"""Rule-based REP SEGMENTATION only.

Detects where each repetition starts and ends from already-extracted pose
skeletons + joint angles. It does NOT judge form: there is no form_label and no
mistake_type anywhere in this script. Output columns are strictly:

    video,exercise,rep_start,rep_end

It is conservative: when a video is unclear it goes to a review file instead of
producing guessed reps. It never touches data/annotations/reps.csv.

Run
---
    python scripts/auto_segment_reps.py \
      --config configs/default.yaml \
      --skeleton_dir data/processed/skeletons \
      --splits_dir data/splits \
      --out data/annotations/reps_segments.csv \
      --review_out data/annotations/reps_review_needed.csv \
      --report data/annotations/rep_segmentation_report.csv \
      --plots_dir data/annotations/debug_plots \
      --fps 30 \
      --make_plots

The heavy lifting (loading, smoothing, side selection, adaptive peak/valley
detection) is reused from ``src/data/auto_reps.py`` so the biomechanical rules
live in one place.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.auto_reps import (  # noqa: E402
    ANGLE_IDX,
    KP,
    DetectConfig,
    EXERCISE_REP_SPEC,
    _robust_rom,
    assess_pose_quality,
    detect_reps,
    load_skeleton_and_angles,
    select_side,
)
from src.feedback.form_rules import get_exercise_names  # noqa: E402

SEGMENT_COLUMNS = ["video", "exercise", "rep_start", "rep_end"]
REVIEW_COLUMNS = ["video", "exercise", "reason"]
REPORT_COLUMNS = [
    "video", "exercise", "rep_index", "rep_start", "rep_end",
    "duration_frames", "duration_seconds", "confidence",
    "driver_signal", "selected_side", "reason",
]


def _clamp01(v):
    return float(max(0.0, min(1.0, v)))


# ---------------------------------------------------------------------------
# Exercise inference (split CSVs are authoritative; fall back to filename prefix)
# ---------------------------------------------------------------------------
def build_exercise_map(splits_dir, exercise_names):
    mapping = {}
    for split in ("train", "val", "test"):
        path = os.path.join(splits_dir, f"{split}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            try:
                idx = int(row["exercise"])
            except (ValueError, TypeError, KeyError):
                continue
            if 0 <= idx < len(exercise_names):
                mapping[str(row["filename"])] = exercise_names[idx]
    return mapping


def infer_exercise(stem, split_map, exercise_names):
    if stem in split_map:
        return split_map[stem]
    for name in sorted(exercise_names, key=len, reverse=True):
        if stem.startswith(name + "_") or stem == name:
            return name
    return None


def list_skeleton_stems(skeleton_dir):
    stems = []
    for f in sorted(os.listdir(skeleton_dir)):
        if f.endswith(".npy") and not f.endswith("_angles.npy") and not f.endswith("_raw.npy"):
            stems.append(f[:-4])
    return stems


# ---------------------------------------------------------------------------
# Per-rep confidence (segmentation quality only — NOT form quality)
# ---------------------------------------------------------------------------
def rep_confidence(driver, skel, s, pk, e, side, spec, cfg, video_rom):
    """Confidence in [0,1] that (s,e) is a real, cleanly-segmented rep.

    Built from: movement range, signal clarity (separation from video noise),
    pose confidence, duration reasonableness, and cycle completeness (does the
    signal start and end near the same rest level it returns to?).
    """
    seg = driver[s:e + 1]
    rep_rom = float(seg.max() - seg.min())

    # signal clarity: this rep's swing relative to the whole video's swing
    separation = rep_rom / max(video_rom, 1e-6)

    # pose confidence over the active side's shoulder/elbow/wrist
    sides = ["l", "r"] if side == "LR" else [side.lower()]
    joints = ([KP[f"{sd}_shoulder"] for sd in sides]
              + [KP[f"{sd}_elbow"] for sd in sides]
              + [KP[f"{sd}_wrist"] for sd in sides])
    pose_conf = float(np.mean(skel[s:e + 1][:, joints, 2]))
    pose_score = _clamp01((pose_conf - cfg.side_min_confidence)
                          / max(1e-6, 1.0 - cfg.side_min_confidence))

    rom_score = _clamp01(rep_rom / spec["good_rom_deg"])

    dur_s = (e - s) / cfg.fps
    if dur_s < cfg.good_min_seconds:
        tempo = _clamp01(dur_s / cfg.good_min_seconds)
    elif dur_s > cfg.good_max_seconds:
        tempo = _clamp01(1.0 - (dur_s - cfg.good_max_seconds) / cfg.good_max_seconds)
    else:
        tempo = 1.0

    # cycle completeness: endpoints should be close to each other (returned to
    # start) and the peak should sit clearly between them.
    rest_level = (seg[0] + seg[-1]) / 2.0
    endpoint_gap = abs(seg[0] - seg[-1]) / max(rep_rom, 1e-6)
    extreme = seg.max() if abs(seg.max() - rest_level) >= abs(seg.min() - rest_level) else seg.min()
    excursion = abs(extreme - rest_level) / max(rep_rom, 1e-6)
    completeness = _clamp01(1.0 - endpoint_gap) * _clamp01(excursion / 0.5)

    confidence = (
        0.30 * _clamp01(separation / 0.8)
        + 0.25 * pose_score
        + 0.20 * rom_score
        + 0.10 * tempo
        + 0.15 * completeness
    )
    if dur_s < cfg.min_rep_seconds * 1.2 or dur_s > cfg.max_rep_seconds * 0.9:
        confidence *= 0.5

    reason = (f"rom={rep_rom:.0f}deg sep={separation:.2f} dur={dur_s:.2f}s "
              f"pose={pose_conf:.2f} cycle={completeness:.2f}")
    return _clamp01(confidence), reason


def _safe_write(df, path):
    """Write a CSV, surviving a locked target (OneDrive sync / file open in editor).

    Falls back to ``<path>.new.csv`` so one locked file never aborts the run or
    loses the other outputs.
    """
    import time
    for attempt in range(3):
        try:
            df.to_csv(path, index=False)
            return path
        except PermissionError:
            time.sleep(1.0)
    alt = path + ".new.csv"
    df.to_csv(alt, index=False)
    print(f"  WARNING: '{path}' was locked; wrote '{alt}' instead "
          f"(close the file, then rename it).")
    return alt


def dedupe_non_overlapping(reps, overlap_frac=0.30):
    """Drop a rep only when it genuinely OVERLAPS its neighbour.

    Full-cycle reps are adjacent by design (one rep's rest-return is the next
    rep's rest-start), so adjacency is NOT a duplicate. We only discard when two
    reps overlap by more than ``overlap_frac`` of the shorter rep, keeping the
    higher-confidence one. Peak spacing already prevents double-counting.
    """
    reps = sorted(reps, key=lambda r: r["rep_start"])
    kept = []
    for r in reps:
        if kept:
            prev = kept[-1]
            overlap = min(prev["rep_end"], r["rep_end"]) - max(prev["rep_start"], r["rep_start"])
            shortest = min(prev["rep_end"] - prev["rep_start"], r["rep_end"] - r["rep_start"])
            if overlap > overlap_frac * max(1, shortest):
                if r["confidence"] > prev["confidence"]:
                    kept[-1] = r
                continue
            # clamp tiny overlaps so reps never cross (rep_end <= next rep_start)
            if overlap > 0:
                mid = (prev["rep_end"] + r["rep_start"]) // 2
                prev["rep_end"] = min(prev["rep_end"], mid)
                r["rep_start"] = max(r["rep_start"], mid)
        kept.append(r)
    return kept


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plot(stem, exercise, status, driver_raw, driver_smooth, side,
              accepted, rejected, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 4))
    if driver_raw is not None:
        ax.plot(driver_raw, color="0.78", lw=1, label="driver (raw L)")
    ax.plot(driver_smooth, color="C0", lw=1.6, label=f"driver smoothed ({side})")
    ymax = float(np.nanmax(driver_smooth))

    for r in accepted:
        ax.axvline(r["rep_start"], color="green", ls="--", lw=1)
        ax.axvline(r["rep_end"], color="green", ls=":", lw=1)
        ax.text((r["rep_start"] + r["rep_end"]) / 2, ymax,
                f"#{r['rep_index']}\nc={r['confidence']:.2f}",
                ha="center", va="top", fontsize=7, color="green")
    for r in rejected:
        ax.axvspan(r["rep_start"], r["rep_end"], color="red", alpha=0.07)
        ax.text((r["rep_start"] + r["rep_end"]) / 2, ymax * 0.55,
                f"rej\nc={r['confidence']:.2f}", ha="center", va="top",
                fontsize=6, color="red")

    ax.set_title(f"{stem}  [{exercise}]  status={status}  "
                 f"accepted={len(accepted)} rejected={len(rejected)}", fontsize=9)
    ax.set_xlabel("frame")
    ax.set_ylabel("joint angle (deg)")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{stem}.png"), dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Rule-based rep segmentation (boundaries only)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--skeleton_dir", default="data/processed/skeletons")
    p.add_argument("--splits_dir", default="data/splits")
    p.add_argument("--out", default="data/annotations/reps_segments.csv")
    p.add_argument("--review_out", default="data/annotations/reps_review_needed.csv")
    p.add_argument("--report", default="data/annotations/rep_segmentation_report.csv")
    p.add_argument("--plots_dir", default="data/annotations/debug_plots")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--make_plots", action="store_true")
    p.add_argument("--min_confidence", type=float, default=0.75)
    p.add_argument("--min_rep_seconds", type=float, default=0.5)
    p.add_argument("--max_rep_seconds", type=float, default=8.0)
    args = p.parse_args()

    config = None
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    exercise_names = get_exercise_names(config)  # read from config, not hardcoded

    cfg = DetectConfig(fps=args.fps,
                       min_rep_seconds=args.min_rep_seconds,
                       max_rep_seconds=args.max_rep_seconds)
    # optional config overrides
    autolab_cfg = (config or {}).get("data", {}).get("autolabel", {}) if config else {}
    for k, v in (autolab_cfg or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    if not os.path.isdir(args.skeleton_dir):
        print(f"ERROR: skeleton dir not found: {args.skeleton_dir}")
        sys.exit(1)

    split_map = build_exercise_map(args.splits_dir, exercise_names)
    stems = list_skeleton_stems(args.skeleton_dir)
    if not stems:
        print(f"ERROR: no skeleton .npy files in {args.skeleton_dir}")
        sys.exit(1)

    segment_rows, review_rows, report_rows = [], [], []
    n_videos = n_accepted_videos = n_review_videos = 0
    n_reps_total = n_reps_accepted = n_reps_rejected = 0

    for stem in stems:
        exercise = infer_exercise(stem, split_map, exercise_names)
        spec = EXERCISE_REP_SPEC.get(exercise) if exercise else None
        if spec is None:
            continue

        skel, angles, err = load_skeleton_and_angles(stem, args.skeleton_dir)
        n_videos += 1
        if err:
            n_review_videos += 1
            review_rows.append({"video": stem, "exercise": exercise or "?", "reason": err})
            continue

        if skel.shape[0] < 15:
            n_review_videos += 1
            review_rows.append({"video": stem, "exercise": exercise,
                                "reason": f"too few frames ({skel.shape[0]})"})
            continue

        avg_conf, jitter, q_reason = assess_pose_quality(skel, cfg)
        side, driver, _ = select_side(skel, angles, spec["primary"], cfg)
        driver_raw = angles[:, ANGLE_IDX[spec["primary"]]["L"]]
        video_rom = _robust_rom(driver)
        driver_name = f"{spec['primary']} angle"

        # whole-video review gates (still detect for the plot/report)
        video_status = "ok"
        video_reason = ""
        if q_reason:
            video_status, video_reason = "review", q_reason
        elif video_rom < spec["min_rom_deg"]:
            video_status, video_reason = "review", (
                f"movement range too small ({video_rom:.0f}deg < {spec['min_rom_deg']:.0f}deg)")

        raw_reps, _ = detect_reps(driver, spec["rest_is_high"], cfg)
        min_f = int(cfg.min_rep_seconds * cfg.fps)
        max_f = int(cfg.max_rep_seconds * cfg.fps)

        candidates = []
        for i, (s, pk, e) in enumerate(raw_reps):
            # hard validity: in-range integers, start<end, duration sane
            if not (0 <= s < e < skel.shape[0]):
                n_reps_rejected += 1
                continue
            dur = e - s
            if dur < min_f or dur > max_f:
                n_reps_rejected += 1
                continue
            conf, reason = rep_confidence(driver, skel, s, pk, e, side, spec, cfg, video_rom)
            candidates.append({
                "rep_index": i, "rep_start": int(s), "rep_end": int(e),
                "confidence": round(conf, 3), "reason": reason,
            })

        candidates = dedupe_non_overlapping(candidates)

        accepted, rejected = [], []
        for r in candidates:
            n_reps_total += 1
            ok = video_status == "ok" and r["confidence"] >= args.min_confidence
            (accepted if ok else rejected).append(r)

        # reindex accepted for clean output
        for new_i, r in enumerate(accepted):
            r["rep_index"] = new_i

        for r in accepted:
            n_reps_accepted += 1
            segment_rows.append({
                "video": stem, "exercise": exercise,
                "rep_start": r["rep_start"], "rep_end": r["rep_end"],
            })
        for r in rejected:
            n_reps_rejected += 1

        # report: every candidate (accepted + rejected)
        for r in accepted + rejected:
            dur = r["rep_end"] - r["rep_start"]
            report_rows.append({
                "video": stem, "exercise": exercise, "rep_index": r["rep_index"],
                "rep_start": r["rep_start"], "rep_end": r["rep_end"],
                "duration_frames": dur, "duration_seconds": round(dur / cfg.fps, 2),
                "confidence": r["confidence"], "driver_signal": driver_name,
                "selected_side": side, "reason": r["reason"],
            })

        if accepted:
            n_accepted_videos += 1
            if video_status == "review":
                # had a quality flag but still produced reps — note for review too
                review_rows.append({"video": stem, "exercise": exercise,
                                    "reason": f"{video_reason} (produced {len(accepted)} low-trust reps)"})
        else:
            n_review_videos += 1
            review_rows.append({
                "video": stem, "exercise": exercise,
                "reason": video_reason or "no confident reps detected",
            })

        if args.make_plots:
            try:
                make_plot(stem, exercise, video_status, driver_raw, driver, side,
                          accepted, rejected, args.plots_dir)
            except Exception as e:  # pragma: no cover
                print(f"  plot failed for {stem}: {e}")

    # ---- write outputs (never touches reps.csv) ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    _safe_write(pd.DataFrame(segment_rows, columns=SEGMENT_COLUMNS), args.out)
    _safe_write(pd.DataFrame(review_rows, columns=REVIEW_COLUMNS), args.review_out)
    _safe_write(pd.DataFrame(report_rows, columns=REPORT_COLUMNS), args.report)

    print("\n" + "=" * 50)
    print("Rep segmentation complete.\n")
    print(f"Videos processed:      {n_videos}")
    print(f"Videos accepted:       {n_accepted_videos}")
    print(f"Videos needing review: {n_review_videos}")
    print(f"Total reps detected:   {n_reps_total}")
    print(f"Accepted reps:         {n_reps_accepted}")
    print(f"Rejected reps:         {n_reps_rejected}")
    print(f"Output:                {args.out}")
    print(f"Review file:           {args.review_out}")
    print(f"Report:                {args.report}")
    if args.make_plots:
        print(f"Plots:                 {args.plots_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
