"""Build data/annotations/unified_reps.csv from all available sources.

Priority order for training:
  1. RepCount         -> rep_start / rep_end supervision (rep counter)
  2. MM-Fit           -> exercise classification + pose (no per-rep timestamps)
  3. Countix-Fitness  -> temporal exercise localization
  4. Real-Time Exer.  -> exercise classification (no rep timestamps)
  5. Roboflow         -> form / phase / mistake labels (image-level, no reps)
  6. Captured         -> optional fine-tuning only (read from existing unified_reps.csv)

Run:
    python scripts/build_unified_reps.py

Output:
    data/annotations/unified_reps.csv

Columns:
    source_dataset, video, exercise, rep_start, rep_end,
    form_label, mistake_type, confidence, split
"""

from __future__ import annotations

import os
import re
import sys
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_EXT     = PROJECT_ROOT / "data" / "external"
DATA_ANN     = PROJECT_ROOT / "data" / "annotations"
DATA_SPLITS  = PROJECT_ROOT / "data" / "splits"
OUT_CSV      = DATA_ANN / "unified_reps.csv"
CONFIG_PATH  = PROJECT_ROOT / "configs" / "default.yaml"

UNIFIED_COLS = [
    "source_dataset", "video", "exercise",
    "rep_start", "rep_end",
    "form_label", "mistake_type",
    "confidence", "split",
]

# ---------------------------------------------------------------------------
# Exercise name normalisation map
# Keys: any variant found in public datasets (lowercase, underscores/spaces ok)
# Values: one of our 4 canonical names
# ---------------------------------------------------------------------------
EXERCISE_MAP: dict[str, str] = {
    # bench press
    "bench_press": "bench_press",
    "bench press": "bench_press",
    "benchpress": "bench_press",
    "flat_dumbbell_bench_press": "bench_press",
    "incline_dumbbell_bench_press": "bench_press",
    "barbell_bench_press": "bench_press",
    # biceps
    "bicep_curl": "biceps",
    "biceps": "biceps",
    "bicep curl": "biceps",
    "biceps_curl": "biceps",
    "barbell_bicep_curl": "biceps",
    "dumbbell_bicep_curl": "biceps",
    "barbell bicep curl": "biceps",
    "bicep_curls": "biceps",
    "curl": "biceps",
    # shoulder press
    "shoulder_press": "shoulder_press",
    "shoulder press": "shoulder_press",
    "overhead_press": "shoulder_press",
    "seated_dumbbell_shoulder_press": "shoulder_press",
    "seated overhead dumbbell press": "shoulder_press",
    "ohp": "shoulder_press",
    # triceps
    "tricep_extension": "triceps",
    "triceps": "triceps",
    "tricep extension": "triceps",
    "tricep_extensions": "triceps",
    "tricep_dip": "triceps",
    "seated_overhead_dumbbell_tricep_extension": "triceps",
    "seated dumbbell overhead tricep extension": "triceps",
    "rope_pushdown": "triceps",
    "rope push down": "triceps",
}


def normalize_exercise(raw: str) -> Optional[str]:
    if not raw or pd.isna(raw):
        return None
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    # try direct
    if key in EXERCISE_MAP:
        return EXERCISE_MAP[key]
    # try without spaces/underscores
    flat = key.replace("_", "")
    for k, v in EXERCISE_MAP.items():
        if k.replace("_", "").replace(" ", "") == flat:
            return v
    return None


def _assign_splits(videos: list[str], train=0.70, val=0.15) -> dict[str, str]:
    """Assign split labels to a list of unique video stems."""
    rng = random.Random(42)
    shuffled = sorted(set(videos))
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val   = int(n * val)
    splits = {}
    for i, v in enumerate(shuffled):
        if i < n_train:
            splits[v] = "train"
        elif i < n_train + n_val:
            splits[v] = "val"
        else:
            splits[v] = "test"
    return splits


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=UNIFIED_COLS)


def _warn(msg: str):
    print(f"  [WARNING] {msg}")


def _info(msg: str):
    print(f"  {msg}")


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------

def adapt_repcount(dataset_dir: Path) -> pd.DataFrame:
    """RepCount — main rep_start/rep_end supervision source.

    Expects one or more JSON annotation files:
      annotation_train.json, annotation_val.json, annotation_test.json

    Two supported formats:
      A) {"video.mp4": {"count": N, "frame_number": [end1, end2, ...]}}
      B) {"video.mp4": {"count": N, "begin": [...], "end": [...]}}
    """
    json_files = {
        "train": dataset_dir / "annotation_train.json",
        "val":   dataset_dir / "annotation_val.json",
        "test":  dataset_dir / "annotation_test.json",
    }
    found = {split: p for split, p in json_files.items() if p.exists()}
    if not found:
        _warn("RepCount: no annotation_*.json files found. Skipping.")
        return _empty_df()

    rows = []
    for split, path in found.items():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for video_name, ann in data.items():
            stem = Path(video_name).stem
            # infer exercise from video name
            exercise = normalize_exercise(stem.split("_")[0]) or normalize_exercise(
                "_".join(stem.split("_")[:2])
            )
            if exercise is None:
                # try whole stem
                exercise = normalize_exercise(stem)
            if exercise is None:
                continue  # not one of our 4 exercises

            begins = ann.get("begin", [])
            ends   = ann.get("end",   ann.get("frame_number", []))

            if begins and ends and len(begins) == len(ends):
                for b, e in zip(begins, ends):
                    rows.append({
                        "source_dataset": "repcount",
                        "video": stem,
                        "exercise": exercise,
                        "rep_start": int(b),
                        "rep_end": int(e),
                        "form_label": "",
                        "mistake_type": "",
                        "confidence": 0.85,
                        "split": split,
                    })
            elif ends:
                # only end frames — build begin from previous end
                prev = 0
                for e in ends:
                    rows.append({
                        "source_dataset": "repcount",
                        "video": stem,
                        "exercise": exercise,
                        "rep_start": int(prev),
                        "rep_end": int(e),
                        "form_label": "",
                        "mistake_type": "",
                        "confidence": 0.70,
                        "split": split,
                    })
                    prev = int(e)

    if not rows:
        _warn("RepCount: JSON files found but no matching exercises. Skipping.")
        return _empty_df()

    _info(f"RepCount: {len(rows)} rep rows from {len(found)} annotation files.")
    return pd.DataFrame(rows, columns=UNIFIED_COLS)


def adapt_mmfit(dataset_dir: Path) -> pd.DataFrame:
    """MM-Fit — exercise classification + pose (no per-rep timestamps).

    Expects: workouts.csv  OR  annotations.csv  OR  *.csv in dataset_dir.
    Required columns (flexible naming):
      subject / subject_id, exercise / activity, start_frame, end_frame
    """
    candidates = [
        dataset_dir / "workouts.csv",
        dataset_dir / "annotations.csv",
        dataset_dir / "workout_annotations.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        # try any CSV
        csvs = list(dataset_dir.glob("*.csv"))
        csv_path = csvs[0] if csvs else None
    if csv_path is None:
        _warn("MM-Fit: no CSV annotation file found. Skipping.")
        return _empty_df()

    df = pd.read_csv(csv_path)
    # flexible column aliases
    col_aliases = {
        "exercise": ["exercise", "activity", "action", "label", "class"],
        "start_frame": ["start_frame", "start", "begin_frame", "frame_start"],
        "end_frame": ["end_frame", "end", "finish_frame", "frame_end"],
        "subject": ["subject", "subject_id", "person", "participant"],
        "split": ["split", "set", "partition"],
    }
    col_map = {}
    for canonical, aliases in col_aliases.items():
        for alias in aliases:
            if alias in df.columns:
                col_map[canonical] = alias
                break

    if "exercise" not in col_map:
        _warn(f"MM-Fit: cannot find exercise column in {csv_path.name}. Skipping.")
        return _empty_df()

    rows = []
    for i, row in df.iterrows():
        raw_ex = row.get(col_map["exercise"], "")
        exercise = normalize_exercise(str(raw_ex))
        if exercise is None:
            continue

        subject  = str(row.get(col_map.get("subject", "subject"), f"subj_{i}"))
        video    = f"mmfit_{subject}_{i:04d}"
        split    = str(row.get(col_map.get("split", "split"), "")).strip()
        if split not in ("train", "val", "test"):
            split = ""

        rows.append({
            "source_dataset": "mmfit",
            "video": video,
            "exercise": exercise,
            "rep_start": "",
            "rep_end": "",
            "form_label": "",
            "mistake_type": "",
            "confidence": 0.50,
            "split": split,
        })

    if not rows:
        _warn("MM-Fit: CSV found but no matching exercises. Skipping.")
        return _empty_df()

    result = pd.DataFrame(rows, columns=UNIFIED_COLS)
    # assign splits if not in data
    if result["split"].eq("").all():
        split_map = _assign_splits(result["video"].tolist())
        result["split"] = result["video"].map(split_map)

    _info(f"MM-Fit: {len(result)} classification rows.")
    return result


def adapt_countix_fitness(dataset_dir: Path) -> pd.DataFrame:
    """Countix-Fitness — temporal exercise localization.

    Expects: countix_fitness_train.csv, countix_fitness_val.csv
    OR: annotations/train.csv, annotations/val.csv
    OR: any *train*.csv / *val*.csv

    Columns: video_id (or name), class (or label), start_frame, end_frame
    """
    candidates = {
        "train": [
            dataset_dir / "countix_fitness_train.csv",
            dataset_dir / "annotations" / "train.csv",
            dataset_dir / "train.csv",
        ],
        "val": [
            dataset_dir / "countix_fitness_val.csv",
            dataset_dir / "annotations" / "val.csv",
            dataset_dir / "val.csv",
        ],
    }
    # also scan for any CSV with train/val in name
    for csv in dataset_dir.glob("*.csv"):
        n = csv.stem.lower()
        if "train" in n:
            candidates["train"].insert(0, csv)
        elif "val" in n:
            candidates["val"].insert(0, csv)

    found = {split: next((p for p in paths if p.exists()), None)
             for split, paths in candidates.items()}
    found = {s: p for s, p in found.items() if p}
    if not found:
        _warn("Countix-Fitness: no annotation CSV files found. Skipping.")
        return _empty_df()

    col_aliases = {
        "video": ["video_id", "video", "name", "filename", "id"],
        "exercise": ["class", "label", "exercise", "action", "category"],
        "start_frame": ["start_frame", "start", "begin"],
        "end_frame": ["end_frame", "end", "finish"],
    }

    rows = []
    for split, path in found.items():
        df = pd.read_csv(path)
        col_map = {}
        for canonical, aliases in col_aliases.items():
            for alias in aliases:
                if alias in df.columns:
                    col_map[canonical] = alias
                    break

        if "exercise" not in col_map or "video" not in col_map:
            _warn(f"Countix-Fitness: missing required columns in {path.name}. Skipping split.")
            continue

        for _, row in df.iterrows():
            exercise = normalize_exercise(str(row.get(col_map["exercise"], "")))
            if exercise is None:
                continue
            video = Path(str(row[col_map["video"]])).stem
            start = row.get(col_map.get("start_frame", ""), "")
            end   = row.get(col_map.get("end_frame", ""), "")

            rows.append({
                "source_dataset": "countix_fitness",
                "video": video,
                "exercise": exercise,
                "rep_start": int(start) if str(start).isdigit() else "",
                "rep_end":   int(end)   if str(end).isdigit()   else "",
                "form_label": "",
                "mistake_type": "",
                "confidence": 0.75,
                "split": split,
            })

    if not rows:
        _warn("Countix-Fitness: CSV files found but no matching exercises. Skipping.")
        return _empty_df()

    _info(f"Countix-Fitness: {len(rows)} rows.")
    return pd.DataFrame(rows, columns=UNIFIED_COLS)


def adapt_realtime_exercise_recognition(dataset_dir: Path) -> pd.DataFrame:
    """Real-Time Exercise Recognition — exercise classification only.

    Expects either:
      A) manifest.csv with columns: filename, label (or class)
      B) videos/{exercise_class}/ folder structure
    """
    manifest = dataset_dir / "manifest.csv"
    rows = []

    if manifest.exists():
        df = pd.read_csv(manifest)
        label_col = next((c for c in ["label", "class", "exercise", "category"] if c in df.columns), None)
        file_col  = next((c for c in ["filename", "file", "video", "name"] if c in df.columns), None)
        if label_col is None or file_col is None:
            _warn(f"RealTime: manifest.csv missing label/file columns. Skipping.")
        else:
            for _, row in df.iterrows():
                exercise = normalize_exercise(str(row[label_col]))
                if exercise is None:
                    continue
                video = Path(str(row[file_col])).stem
                rows.append({
                    "source_dataset": "realtime_exercise_recognition",
                    "video": video,
                    "exercise": exercise,
                    "rep_start": "",
                    "rep_end": "",
                    "form_label": "",
                    "mistake_type": "",
                    "confidence": 0.60,
                    "split": "",
                })

    # also scan video subdirectories
    videos_dir = dataset_dir / "videos"
    if videos_dir.exists():
        for ex_dir in videos_dir.iterdir():
            if not ex_dir.is_dir():
                continue
            exercise = normalize_exercise(ex_dir.name)
            if exercise is None:
                continue
            for vf in ex_dir.iterdir():
                if vf.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
                    rows.append({
                        "source_dataset": "realtime_exercise_recognition",
                        "video": vf.stem,
                        "exercise": exercise,
                        "rep_start": "",
                        "rep_end": "",
                        "form_label": "",
                        "mistake_type": "",
                        "confidence": 0.60,
                        "split": "",
                    })

    if not rows:
        _warn("RealTime Exercise Recognition: no data found. Skipping.")
        return _empty_df()

    result = pd.DataFrame(rows, columns=UNIFIED_COLS)
    # deduplicate by video
    result = result.drop_duplicates(subset=["video"])
    # assign splits
    split_map = _assign_splits(result["video"].tolist())
    result.loc[result["split"] == "", "split"] = result.loc[result["split"] == "", "video"].map(split_map)

    _info(f"RealTime Exercise Recognition: {len(result)} classification rows.")
    return result


def adapt_roboflow(dataset_dir: Path) -> pd.DataFrame:
    """Roboflow supplement — image-level form/phase/mistake labels.

    Scans all subdirs for _annotations.csv files.
    These contribute form_label only (no rep_start/rep_end).
    """
    FORM_INCORRECT_KEYWORDS = {
        "wrong", "incorrect", "bad", "error", "fail",
        "wrong_squat", "wrong_lunge", "wrong_push",
    }
    FORM_CORRECT_KEYWORDS = {
        "correct", "good", "start", "end", "mid",
        "squats_start", "squats_end", "pushup_start",
    }
    MISTAKE_CLASSES = {
        "wrong_squat": "bad_squat_form",
        "wrong_lunge": "bad_lunge_form",
        "wrong_push_up": "bad_pushup_form",
        "wrong_push": "bad_pushup_form",
    }

    rows = []
    for ann_csv in dataset_dir.rglob("_annotations.csv"):
        df = pd.read_csv(ann_csv)
        label_col = next((c for c in ["class", "label", "category", "name"] if c in df.columns), None)
        file_col  = next((c for c in ["filename", "file", "image"] if c in df.columns), None)
        if label_col is None or file_col is None:
            _warn(f"Roboflow: skipping {ann_csv.name} — missing label/file columns.")
            continue

        for _, row in df.iterrows():
            raw_class = str(row[label_col]).strip().lower().replace("-", "_").replace(" ", "_")
            filename  = Path(str(row[file_col])).stem

            # determine exercise (best-effort from class name)
            exercise = normalize_exercise(raw_class)

            # form label from class name keywords
            form_label = ""
            mistake_type = ""
            if any(k in raw_class for k in FORM_INCORRECT_KEYWORDS):
                form_label = 0
                mistake_type = MISTAKE_CLASSES.get(raw_class, raw_class)
            elif any(k in raw_class for k in FORM_CORRECT_KEYWORDS):
                form_label = 1

            if exercise is None and form_label == "":
                continue  # nothing useful

            rows.append({
                "source_dataset": "roboflow",
                "video": filename,
                "exercise": exercise or "",
                "rep_start": "",
                "rep_end": "",
                "form_label": form_label,
                "mistake_type": mistake_type,
                "confidence": 0.65,
                "split": "",
            })

    if not rows:
        _warn("Roboflow: no _annotations.csv files found. Skipping.")
        return _empty_df()

    result = pd.DataFrame(rows, columns=UNIFIED_COLS)
    result = result.drop_duplicates(subset=["video"])
    split_map = _assign_splits(result["video"].tolist())
    result.loc[result["split"] == "", "split"] = result.loc[result["split"] == "", "video"].map(split_map)
    _info(f"Roboflow: {len(result)} image-level form rows.")
    return result


def adapt_captured(splits_dir: Path, existing_unified: Optional[Path]) -> pd.DataFrame:
    """Captured videos — read existing unified_reps.csv rows tagged as 'captured'.

    These are fine-tuning data only. Merged last so public datasets take priority.
    """
    if existing_unified and existing_unified.exists():
        try:
            df = pd.read_csv(existing_unified)
            if "source_dataset" in df.columns:
                captured = df[df["source_dataset"] == "captured"].copy()
                if len(captured) > 0:
                    _info(f"Captured (fine-tune): {len(captured)} rows from existing unified_reps.csv.")
                    return captured[UNIFIED_COLS] if all(c in captured.columns for c in UNIFIED_COLS) else _empty_df()
        except Exception:
            pass

    # fallback: build from splits CSVs (no rep labels, exercise-classification only)
    rows = []
    for split in ("train", "val", "test"):
        csv = splits_dir / f"{split}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            rows.append({
                "source_dataset": "captured",
                "video": row["filename"],
                "exercise": "",  # will be filled by exercise index
                "rep_start": "",
                "rep_end": "",
                "form_label": "",
                "mistake_type": "",
                "confidence": 0.90,
                "split": split,
            })

    if not rows:
        _warn("Captured: no splits CSVs or existing unified_reps.csv found.")
        return _empty_df()

    _info(f"Captured (fine-tune): {len(rows)} rows from splits CSVs.")
    return pd.DataFrame(rows, columns=UNIFIED_COLS)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_and_print(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("UNIFIED REPS VALIDATION")
    print("=" * 60)

    total = len(df)
    print(f"\nTotal rows: {total}")
    if total == 0:
        print("  ERROR: unified_reps.csv is empty!")
        return False

    # rows per source
    print("\nRows per source_dataset:")
    for src, cnt in df["source_dataset"].value_counts().items():
        print(f"  {src:<40} {cnt:>5}")

    # rows per exercise
    print("\nRows per exercise:")
    ex_counts = df["exercise"].replace("", pd.NA).dropna().value_counts()
    for ex, cnt in ex_counts.items():
        print(f"  {ex:<40} {cnt:>5}")
    empty_ex = df["exercise"].eq("").sum() + df["exercise"].isna().sum()
    if empty_ex:
        print(f"  (empty exercise)                         {empty_ex:>5}")

    # rep counting rows
    has_rep = df["rep_start"].notna() & df["rep_start"].astype(str).str.strip().ne("") & \
              df["rep_end"].notna()   & df["rep_end"].astype(str).str.strip().ne("")
    print(f"\nRows with rep_start + rep_end (rep counting):  {has_rep.sum():>5}")
    print(f"Rows without rep timestamps (classif. only):   {(~has_rep).sum():>5}")

    # form label rows
    has_form = df["form_label"].notna() & df["form_label"].astype(str).str.strip().ne("")
    print(f"Rows with form_label (form correction):        {has_form.sum():>5}")

    # split distribution
    print("\nRows per split:")
    for sp, cnt in df["split"].value_counts().items():
        print(f"  {sp:<10} {cnt:>5}")

    # per-dataset task coverage
    print("\nTask support per source:")
    print(f"  {'Dataset':<40} {'exercise_clf':>12} {'rep_counting':>12} {'form_corr':>10}")
    print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*10}")
    for src in df["source_dataset"].unique():
        sub = df[df["source_dataset"] == src]
        has_ex   = sub["exercise"].notna() & sub["exercise"].astype(str).str.strip().ne("")
        sub_rep  = sub["rep_start"].notna() & sub["rep_start"].astype(str).str.strip().ne("")
        sub_form = sub["form_label"].notna() & sub["form_label"].astype(str).str.strip().ne("")
        print(f"  {src:<40} {'yes' if has_ex.any() else 'no':>12} "
              f"{'yes' if sub_rep.any() else 'no':>12} "
              f"{'yes' if sub_form.any() else 'no':>10}")

    # integrity checks
    print("\nIntegrity checks:")
    errors = []

    empty_src = df["source_dataset"].isna() | df["source_dataset"].astype(str).str.strip().eq("")
    if empty_src.any():
        errors.append(f"  FAIL: {empty_src.sum()} rows have empty source_dataset")

    empty_vid = df["video"].isna() | df["video"].astype(str).str.strip().eq("")
    if empty_vid.any():
        errors.append(f"  FAIL: {empty_vid.sum()} rows have empty video")

    # rep_start < rep_end where both exist
    rep_rows = df[has_rep].copy()
    if len(rep_rows):
        rep_rows["_s"] = pd.to_numeric(rep_rows["rep_start"], errors="coerce")
        rep_rows["_e"] = pd.to_numeric(rep_rows["rep_end"],   errors="coerce")
        bad_order = rep_rows["_s"] >= rep_rows["_e"]
        if bad_order.any():
            errors.append(f"  FAIL: {bad_order.sum()} rows have rep_start >= rep_end")

    # form_label only 0 or 1
    form_rows = df[has_form].copy()
    if len(form_rows):
        form_rows["_f"] = pd.to_numeric(form_rows["form_label"], errors="coerce")
        bad_form = ~form_rows["_f"].isin([0, 1])
        if bad_form.any():
            errors.append(f"  FAIL: {bad_form.sum()} rows have form_label not in {{0,1}}")

    # confidence in [0,1]
    conf = pd.to_numeric(df["confidence"], errors="coerce")
    bad_conf = conf.notna() & ((conf < 0) | (conf > 1))
    if bad_conf.any():
        errors.append(f"  FAIL: {bad_conf.sum()} rows have confidence outside [0,1]")

    # split values
    bad_split = ~df["split"].astype(str).isin(["train", "val", "test", ""])
    if bad_split.any():
        errors.append(f"  FAIL: {bad_split.sum()} rows have invalid split value")

    if errors:
        for e in errors:
            print(e)
        print("\nValidation: FAILED")
        return False
    else:
        print("  PASS: all checks passed")

    # training readiness
    print("\nTraining readiness:")
    print(f"  Total rows loaded:                      {total:>5}")
    print(f"  Usable for exercise classification:     "
          f"{(df['exercise'].notna() & df['exercise'].astype(str).ne('')).sum():>5}")
    print(f"  Usable for rep counting:                {has_rep.sum():>5}")
    print(f"  Usable for form correction:             {has_form.sum():>5}")
    print(f"  Fallback to rules for unlabeled videos: yes (fallback_to_rules=true in config)")
    print(f"  Training unblocked:                     {'YES' if total > 0 else 'NO'}")
    print("=" * 60)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building unified_reps.csv")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  External data: {DATA_EXT}")
    print()

    # load config for reference
    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}

    parts: list[pd.DataFrame] = []

    # --- Priority 1: RepCount ---
    print("[1/6] RepCount (rep counting supervision):")
    parts.append(adapt_repcount(DATA_EXT / "repcount"))

    # --- Priority 2: MM-Fit ---
    print("[2/6] MM-Fit (exercise classification + pose):")
    parts.append(adapt_mmfit(DATA_EXT / "mmfit"))

    # --- Priority 3: Countix-Fitness ---
    print("[3/6] Countix-Fitness (temporal localization):")
    parts.append(adapt_countix_fitness(DATA_EXT / "countix_fitness"))

    # --- Priority 4: Real-Time Exercise Recognition ---
    print("[4/6] Real-Time Exercise Recognition (classification):")
    parts.append(adapt_realtime_exercise_recognition(
        DATA_EXT / "realtime_exercise_recognition"
    ))

    # --- Priority 5: Roboflow ---
    print("[5/6] Roboflow supplements (form / phase / mistake labels):")
    parts.append(adapt_roboflow(DATA_EXT / "roboflow"))

    # --- Priority 6: Captured (optional fine-tuning) ---
    print("[6/6] Captured videos (optional fine-tuning):")
    existing_unified = OUT_CSV if OUT_CSV.exists() else None
    parts.append(adapt_captured(DATA_SPLITS, existing_unified))

    # combine
    non_empty = [p for p in parts if len(p) > 0]
    if not non_empty:
        print("\nERROR: No data from any source. unified_reps.csv NOT written.")
        print("Download at least one public dataset into data/external/ and re-run.")
        sys.exit(1)

    unified = pd.concat(non_empty, ignore_index=True)

    # clean up: strip whitespace in string columns
    for col in ["source_dataset", "video", "exercise", "mistake_type", "split"]:
        unified[col] = unified[col].astype(str).str.strip().replace("nan", "")

    # ensure numeric types where appropriate
    for col in ["rep_start", "rep_end"]:
        unified[col] = pd.to_numeric(unified[col], errors="coerce").where(
            unified[col].astype(str).str.strip().ne(""), other=pd.NA
        )
    unified["form_label"] = pd.to_numeric(unified["form_label"], errors="coerce").where(
        unified["form_label"].astype(str).str.strip().ne(""), other=pd.NA
    )
    unified["confidence"] = pd.to_numeric(unified["confidence"], errors="coerce").fillna(0.50)

    # write
    DATA_ANN.mkdir(parents=True, exist_ok=True)
    unified.to_csv(OUT_CSV, index=False)
    print(f"\nWritten: {OUT_CSV}  ({len(unified)} rows)")

    # validate
    ok = validate_and_print(unified)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
