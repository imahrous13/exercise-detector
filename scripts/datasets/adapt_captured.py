"""Adapter — locally captured videos (existing project data).

Converts data/annotations/reps.csv + data/splits/*.csv into the unified schema.
This is always available and forms the baseline for unified_reps.csv.

Source: your own recorded videos (bench_press, biceps, shoulder_press, triceps)
Tasks: exercise_classification, rep_counting, form_correction
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.datasets.utils import UNIFIED_COLUMNS, normalize_exercise, empty_unified_df

SOURCE  = "captured"
ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPS    = os.path.join(ROOT, "data", "annotations", "reps.csv")
SPLITS  = os.path.join(ROOT, "data", "splits")

TASKS_SUPPORTED = ["exercise_classification", "rep_counting", "form_correction"]


def check_available() -> bool:
    return os.path.isfile(REPS)


def download_info() -> str:
    return (
        "No download required — this is your own captured data.\n"
        "Run:  python scripts/build_clean_reps.py\n"
        "to regenerate data/annotations/reps.csv from the segmentation report."
    )


def convert() -> pd.DataFrame:
    """Convert reps.csv + splits CSVs → unified DataFrame."""
    if not check_available():
        print(f"  [captured] reps.csv not found at {REPS} — skipping.")
        return empty_unified_df()

    reps = pd.read_csv(REPS)
    reps = reps[reps["video"].notna() & (reps["video"].astype(str).str.strip() != "")]

    # Build video → split mapping
    split_map: dict[str, str] = {}
    for split in ("train", "val", "test"):
        csv = os.path.join(SPLITS, f"{split}.csv")
        if os.path.isfile(csv):
            df = pd.read_csv(csv)
            for fn in df["filename"].astype(str):
                split_map[fn] = split

    rows = []
    for _, r in reps.iterrows():
        video = str(r["video"]).strip()
        rows.append({
            "source_dataset": SOURCE,
            "video":          video,
            "exercise":       normalize_exercise(str(r.get("exercise", ""))),
            "rep_start":      int(r["rep_start"]) if pd.notna(r.get("rep_start")) else "",
            "rep_end":        int(r["rep_end"])   if pd.notna(r.get("rep_end"))   else "",
            "form_label":     int(r["form_label"]) if pd.notna(r.get("form_label")) else "",
            "mistake_type":   str(r.get("mistake_type", "") or "").strip(),
            "confidence":     0.85,   # derived from segmentation report filters
            "split":          split_map.get(video, "train"),
        })

    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    print(f"  [captured]        {len(df):5d} rows | "
          f"{df['video'].nunique()} videos | "
          f"exercises: {sorted(df['exercise'].unique().tolist())}")
    return df
