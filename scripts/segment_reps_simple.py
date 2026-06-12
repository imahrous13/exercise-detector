"""Bulletproof rep start/end segmentation via elbow-angle turning points.

Boundaries are anchored on the EXACT turning points (extrema) of the smoothed
elbow angle, not on fixed threshold crossings. Turning points are
threshold-independent and reproducible, and a per-video adaptive prominence
adapts to each clip's own range of motion (so low-ROM / odd-angle videos are
still recovered instead of silently dropped).

Anchor model (rep_start = first frame at the starting anchor, rep_end = return
to the same anchor; one rep is a full cycle):

  bench_press / shoulder_press / triceps : anchor = elbow VALLEY (most flexed)
      rep = valley -> peak -> valley
  biceps                                 : anchor = elbow PEAK (arm extended)
      rep = peak -> valley -> peak

Output:
  data/annotations/reps_segments.csv       <- video,exercise,rep_start,rep_end
  data/annotations/reps_review_needed.csv  <- video,exercise,reason
  data/annotations/debug_plots/*.png       <- with --make_plots

Run:
  python scripts/segment_reps_simple.py --make_plots
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feedback.form_rules import get_exercise_names  # noqa: E402

# ---------------------------------------------------------------------------
# Per-exercise anchor: which elbow-angle extreme marks rep_start / rep_end.
#   "valley" = elbow local MINIMUM (most flexed)   -> rep = valley->peak->valley
#   "peak"   = elbow local MAXIMUM (arm extended)  -> rep = peak->valley->peak
# The exercise list itself is read from configs/default.yaml; this map only
# assigns the anchor direction for the 4 supported exercises.
# ---------------------------------------------------------------------------
ANCHOR = {
    "bench_press":    "valley",   # bottom (bar at chest) -> top -> bottom
    "shoulder_press": "valley",   # bottom (hands at shoulders) -> top -> bottom
    "triceps":        "valley",   # top (elbow flexed) -> bottom (extended) -> top
    "biceps":         "peak",     # bottom (arm extended) -> top (curled) -> bottom
}

# Indices in the 12-angle array (see src/preprocessing/normalize.py)
L_ELBOW_IDX = 2
R_ELBOW_IDX = 3

# ---------------------------------------------------------------------------
# Tunable defaults (overridable from the CLI)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "prominence_frac": 0.35,   # an anchor's opposite extreme must swing this
                               # fraction of the video's ROM to be a real rep
    "prominence_min_deg": 15,  # absolute floor for prominence (degrees)
    "min_rom_deg": 25,         # whole-video elbow ROM below this -> review only
    "min_rep_seconds": 0.5,    # shorter -> impossible / noise
    "max_rep_seconds": 8.0,    # longer  -> not a single rep
    "smoothing_window": 7,     # moving-average window (frames)
    "refine_window": 4,        # +/- frames to snap an anchor to the true extremum
    "fps": 30.0,
}


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def smooth(signal: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing (reflect padding to avoid edge collapse)."""
    if window <= 1 or len(signal) < window:
        return signal.astype(np.float32)
    pad = window // 2
    padded = np.pad(signal.astype(np.float64), pad, mode="edge")
    kernel = np.ones(window) / window
    out = np.convolve(padded, kernel, mode="same")[pad:pad + len(signal)]
    return out.astype(np.float32)


def refine_extremum(signal: np.ndarray, frame: int, anchor: str, win: int) -> int:
    """Snap an anchor frame to the true local extremum within +/- win frames."""
    lo = max(0, frame - win)
    hi = min(len(signal), frame + win + 1)
    seg = signal[lo:hi]
    if len(seg) == 0:
        return frame
    off = int(np.argmin(seg)) if anchor == "valley" else int(np.argmax(seg))
    return lo + off


# ---------------------------------------------------------------------------
# Per-video rep detection (turning-point method)
# ---------------------------------------------------------------------------
def detect_reps_for_video(angles: np.ndarray, exercise: str, params: dict) -> dict:
    """Detect reps anchored on elbow-angle turning points.

    Returns a dict with keys:
      reps        : list of (rep_start, rep_end) integer frame tuples
      elbow       : smoothed elbow signal (for plotting)
      anchor      : "valley" or "peak"
      anchors     : list of accepted anchor frames (for plotting)
      opp_extrema : list of opposite-extreme frames between accepted anchors
      reason      : "" if a clean accept, else explanation
      low_quality : True if reps exist but the video should still be reviewed
    """
    anchor = ANCHOR[exercise]
    out = {"reps": [], "elbow": None, "anchor": anchor, "anchors": [],
           "opp_extrema": [], "reason": "", "low_quality": False}

    if angles is None or angles.shape[0] < 5:
        out["reason"] = "too few frames"
        return out

    L = angles[:, L_ELBOW_IDX].astype(np.float32)
    R = angles[:, R_ELBOW_IDX].astype(np.float32)
    elbow_raw = (L + R) / 2.0

    elbow = smooth(elbow_raw, params["smoothing_window"])
    elbow_light = smooth(elbow_raw, 3)  # near-raw, for sub-frame refinement
    out["elbow"] = elbow

    # Robust range of motion for this video.
    rom = float(np.percentile(elbow, 90) - np.percentile(elbow, 10))
    if rom < params["min_rom_deg"]:
        out["reason"] = (f"too little movement (elbow ROM {rom:.0f}° < "
                         f"{params['min_rom_deg']}°)")
        return out

    prominence = max(params["prominence_min_deg"], params["prominence_frac"] * rom)
    distance = max(1, int(params["min_rep_seconds"] * params["fps"]))

    # Find the anchor extrema. find_peaks finds maxima, so negate for valleys.
    sig = -elbow if anchor == "valley" else elbow
    peaks, _ = find_peaks(sig, prominence=prominence, distance=distance)
    anchors = [refine_extremum(elbow_light, int(p), anchor, params["refine_window"])
               for p in peaks]
    anchors = sorted(set(anchors))

    if len(anchors) < 2:
        out["reason"] = (f"fewer than 2 clean {anchor} turning points "
                         f"(ROM {rom:.0f}°, prominence {prominence:.0f}°)")
        return out

    min_frames = int(params["min_rep_seconds"] * params["fps"])
    max_frames = int(params["max_rep_seconds"] * params["fps"])
    amp_floor = params["prominence_frac"] * rom

    reps, opp_extrema, durations, amps = [], [], [], []
    for i in range(len(anchors) - 1):
        s, e = anchors[i], anchors[i + 1]
        if s >= e:
            continue
        dur = e - s
        if dur < min_frames or dur > max_frames:
            continue
        # The opposite extreme between the two anchors must swing far enough
        # for this to be a genuine full rep (not a ripple between two minima).
        seg = elbow[s:e + 1]
        if anchor == "valley":
            opp = int(s + np.argmax(seg))
            amp = float(elbow[opp] - max(elbow[s], elbow[e]))
        else:
            opp = int(s + np.argmin(seg))
            amp = float(min(elbow[s], elbow[e]) - elbow[opp])
        if amp < amp_floor:
            continue
        reps.append((int(s), int(e)))
        opp_extrema.append(opp)
        durations.append(dur)
        amps.append(amp)

    if not reps:
        out["reason"] = (f"{len(anchors)} {anchor} points but no rep passed "
                         f"duration [{params['min_rep_seconds']}–"
                         f"{params['max_rep_seconds']}s] + swing checks")
        return out

    out["reps"] = reps
    out["anchors"] = sorted(set([f for r in reps for f in r]))
    out["opp_extrema"] = opp_extrema

    # ---- per-video quality: flag (but keep) borderline videos for review ----
    reasons = []
    if rom < 1.5 * params["min_rom_deg"]:
        reasons.append(f"low ROM ({rom:.0f}°)")
    mean_amp = float(np.mean(amps))
    if mean_amp < 0.5 * rom:
        reasons.append(f"shallow reps (mean swing {mean_amp:.0f}° vs ROM {rom:.0f}°)")
    if len(durations) >= 2:
        cv = float(np.std(durations) / (np.mean(durations) + 1e-6))
        if cv > 0.5:
            reasons.append(f"irregular rep durations (CV {cv:.2f})")
    if reasons:
        out["low_quality"] = True
        out["reason"] = "recovered, double-check: " + "; ".join(reasons)

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plot(stem, exercise, result, plots_dir):
    """Plot the elbow signal with rep_start/rep_end anchors and opposite extrema."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    elbow = result["elbow"]
    if elbow is None:
        return
    anchor = result["anchor"]
    reps = result["reps"]

    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.plot(elbow, color="C0", lw=1.4, label="elbow angle (L+R avg, smoothed)")

    for k, (s, e) in enumerate(reps):
        ax.axvline(s, color="green", ls="-", lw=1.3)
        ax.axvline(e, color="green", ls="-", lw=1.3)
        ax.text((s + e) / 2, elbow.max(), f"#{k}", ha="center", va="top",
                fontsize=8, color="green")
    # opposite extreme (mid-rep) markers
    if result["opp_extrema"]:
        ox = result["opp_extrema"]
        marker = "^" if anchor == "valley" else "v"
        ax.plot(ox, elbow[ox], marker, color="black", ms=5,
                label="mid-rep extreme")
    # anchor markers
    if result["anchors"]:
        ax_ = result["anchors"]
        marker = "v" if anchor == "valley" else "^"
        ax.plot(ax_, elbow[ax_], marker, color="green", ms=6,
                label=f"anchor ({anchor}) = rep_start/end")

    title = (f"{stem} [{exercise}] — {len(reps)} reps — anchor={anchor} "
             f"(green lines = rep_start/rep_end)")
    if result["low_quality"]:
        title += "  [REVIEW]"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("frame")
    ax.set_ylabel("elbow angle (deg)")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{stem}.png"), dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loading / exercise inference
# ---------------------------------------------------------------------------
def load_angles(skeleton_dir: str, stem: str) -> tuple:
    """Load the angle .npy file (or compute from skeleton if missing)."""
    ang_path = os.path.join(skeleton_dir, f"{stem}_angles.npy")
    skel_path = os.path.join(skeleton_dir, f"{stem}.npy")

    if os.path.isfile(ang_path):
        return np.load(ang_path).astype(np.float32), ""

    if not os.path.isfile(skel_path):
        return None, "missing .npy file"

    try:
        from src.preprocessing.normalize import compute_angles
        skel = np.load(skel_path).astype(np.float32)
        return compute_angles(skel[:, :, :3]).astype(np.float32), ""
    except Exception as e:
        return None, f"could not compute angles: {e}"


def build_split_map(splits_dir: str, exercise_names: list) -> dict:
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


def infer_exercise(stem: str, split_map: dict, exercise_names: list):
    if stem in split_map:
        return split_map[stem]
    for name in sorted(exercise_names, key=len, reverse=True):
        if stem.startswith(name + "_") or stem == name:
            return name
    return None


def list_stems(skeleton_dir: str) -> list:
    return sorted(
        f[:-4] for f in os.listdir(skeleton_dir)
        if f.endswith(".npy")
        and not f.endswith("_angles.npy")
        and not f.endswith("_raw.npy")
    )


def safe_write_csv(df: pd.DataFrame, path: str) -> str:
    """Write a CSV. If the target is locked, write to a sibling tmp file then
    atomically replace the target so the editor reloads it automatically."""
    import tempfile, shutil
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        pass
    # Write to a sibling temp file, then replace.
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp.csv")
    try:
        os.close(fd)
        df.to_csv(tmp, index=False)
        # os.replace works even when the target is open for reading on Windows.
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            alt = path[:-4] + ".new.csv" if path.endswith(".csv") else path + ".new"
            shutil.move(tmp, alt)
            print(f"  WARNING: {path} was locked; wrote {alt} instead.")
            return alt
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Bulletproof turning-point rep segmentation")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--skeleton_dir", default="data/processed/skeletons")
    p.add_argument("--splits_dir", default="data/splits")
    p.add_argument("--out", default="data/annotations/reps_segments_new.csv")
    p.add_argument("--review_out", default="data/annotations/reps_review_needed.csv")
    p.add_argument("--plots_dir", default="data/annotations/debug_plots")
    p.add_argument("--make_plots", action="store_true")
    p.add_argument("--fps", type=float, default=DEFAULTS["fps"])
    p.add_argument("--min_rep_seconds", type=float, default=DEFAULTS["min_rep_seconds"])
    p.add_argument("--max_rep_seconds", type=float, default=DEFAULTS["max_rep_seconds"])
    p.add_argument("--prominence_frac", type=float, default=DEFAULTS["prominence_frac"])
    p.add_argument("--prominence_min_deg", type=float, default=DEFAULTS["prominence_min_deg"])
    p.add_argument("--min_rom_deg", type=float, default=DEFAULTS["min_rom_deg"])
    p.add_argument("--smoothing_window", type=int, default=DEFAULTS["smoothing_window"])
    args = p.parse_args()

    params = {
        "prominence_frac": args.prominence_frac,
        "prominence_min_deg": args.prominence_min_deg,
        "min_rom_deg": args.min_rom_deg,
        "min_rep_seconds": args.min_rep_seconds,
        "max_rep_seconds": args.max_rep_seconds,
        "smoothing_window": args.smoothing_window,
        "refine_window": DEFAULTS["refine_window"],
        "fps": args.fps,
    }

    config = None
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    exercise_names = get_exercise_names(config)  # exercise list comes from config

    if not os.path.isdir(args.skeleton_dir):
        print(f"ERROR: skeleton dir not found: {args.skeleton_dir}")
        sys.exit(1)

    split_map = build_split_map(args.splits_dir, exercise_names)
    stems = list_stems(args.skeleton_dir)

    segment_rows, review_rows = [], []
    n_videos = n_ok = n_review = n_reps = n_recovered = 0

    for stem in stems:
        exercise = infer_exercise(stem, split_map, exercise_names)
        if exercise not in ANCHOR:
            continue

        n_videos += 1
        angles, err = load_angles(args.skeleton_dir, stem)
        if err or angles is None:
            n_review += 1
            review_rows.append({"video": stem, "exercise": exercise or "?", "reason": err})
            continue

        result = detect_reps_for_video(angles, exercise, params)

        if args.make_plots:
            try:
                make_plot(stem, exercise, result, args.plots_dir)
            except Exception as e:
                print(f"  plot failed for {stem}: {e}")

        reps = result["reps"]
        if not reps:
            n_review += 1
            review_rows.append({"video": stem, "exercise": exercise,
                                "reason": result["reason"] or "no reps detected"})
            continue

        # recover_review: accept the reps, but flag low-quality videos too.
        n_ok += 1
        for s, e in reps:
            n_reps += 1
            segment_rows.append({"video": stem, "exercise": exercise,
                                 "rep_start": int(s), "rep_end": int(e)})
        if result["low_quality"]:
            n_recovered += 1
            review_rows.append({"video": stem, "exercise": exercise,
                                "reason": result["reason"]})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_path = safe_write_csv(
        pd.DataFrame(segment_rows, columns=["video", "exercise", "rep_start", "rep_end"]),
        args.out)
    review_path = safe_write_csv(
        pd.DataFrame(review_rows, columns=["video", "exercise", "reason"]),
        args.review_out)

    print("\nRep segmentation complete.\n")
    print(f"Videos processed:        {n_videos}")
    print(f"Videos accepted:         {n_ok}")
    print(f"  of which flagged review:{n_recovered}")
    print(f"Videos review-only:      {n_review}")
    print(f"Total accepted reps:     {n_reps}")
    print(f"Output:                  {out_path}")
    print(f"Review file:             {review_path}")


if __name__ == "__main__":
    main()
