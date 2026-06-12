"""Build data/annotations/unified_reps.csv from all available dataset adapters.

Runs every adapter in scripts/datasets/. Available datasets produce rows;
unavailable ones print a download notice and contribute zero rows.

The captured (local) dataset is ALWAYS included as baseline — training is
never blocked even before external datasets are downloaded.

Output:
  data/annotations/unified_reps.csv

Columns:
  source_dataset, video, exercise, rep_start, rep_end,
  form_label, mistake_type, confidence, split

Usage:
  python scripts/build_unified_reps.py
  python scripts/build_unified_reps.py --dry_run
  python scripts/build_unified_reps.py --min_confidence 0.60
"""

from __future__ import annotations

import argparse
import os
import sys
import random

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.datasets.utils import (
    UNIFIED_COLUMNS,
    DATASET_TASK_SUPPORT,
    TASKS,
    row_tasks,
    empty_unified_df,
)
from scripts.datasets import (
    adapt_captured,
    adapt_repcount,
    adapt_countix,
    adapt_mmfit,
    adapt_realtime,
    adapt_roboflow,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_PATH     = os.path.join(PROJECT_ROOT, "data", "annotations", "unified_reps.csv")

# Ordered list of adapters — captured is always first (baseline)
ADAPTERS = [
    adapt_captured,
    adapt_repcount,
    adapt_countix,
    adapt_mmfit,
    adapt_realtime,
    adapt_roboflow,
]


def _reassign_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """For sources that don't provide splits, assign train/val/test 70/15/15.

    Sources that already have split labels (captured, repcount test, countix splits)
    keep their original split assignment. Only rows with split="" or rows from
    sources that don't pre-assign splits are re-partitioned at the VIDEO level
    (all reps from one video stay in the same split).
    """
    needs_split_mask = df["split"].isin(["", None, float("nan")]) | df["split"].isna()
    needs_split_mask = needs_split_mask.fillna(True)

    if not needs_split_mask.any():
        return df

    # Group videos that need splitting
    videos_needing = df.loc[needs_split_mask, "video"].unique().tolist()
    random.seed(seed)
    random.shuffle(videos_needing)

    n = len(videos_needing)
    n_val  = max(1, int(n * 0.15))
    n_test = max(1, int(n * 0.15))
    val_set  = set(videos_needing[:n_val])
    test_set = set(videos_needing[n_val:n_val + n_test])

    def _assign(row):
        if not needs_split_mask.loc[row.name]:
            return row["split"]
        v = row["video"]
        if v in val_set:
            return "val"
        if v in test_set:
            return "test"
        return "train"

    df = df.copy()
    df["split"] = df.apply(_assign, axis=1)
    return df


def main():
    p = argparse.ArgumentParser(description="Build unified_reps.csv from all dataset adapters")
    p.add_argument("--out",             default=OUT_PATH)
    p.add_argument("--min_confidence",  type=float, default=0.0,
                   help="Drop rows below this confidence (default: keep all)")
    p.add_argument("--dry_run",         action="store_true",
                   help="Print summary without writing the file")
    args = p.parse_args()

    print("Building unified_reps.csv")
    print("=" * 60)
    print("Running adapters:")

    frames = []
    for adapter in ADAPTERS:
        try:
            df = adapter.convert()
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:
            name = getattr(adapter, "SOURCE", adapter.__name__)
            print(f"  [{name}] ERROR: {e}")

    if not frames:
        print("\nNo data from any adapter. Check that captured reps.csv exists.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Normalise types
    combined["rep_start"] = combined["rep_start"].apply(
        lambda x: int(x) if str(x).strip().lstrip("-").isdigit() else "")
    combined["rep_end"]   = combined["rep_end"].apply(
        lambda x: int(x) if str(x).strip().lstrip("-").isdigit() else "")
    combined["confidence"] = pd.to_numeric(combined["confidence"], errors="coerce").fillna(0.5)

    # Apply confidence filter (if requested)
    if args.min_confidence > 0:
        before = len(combined)
        combined = combined[combined["confidence"] >= args.min_confidence]
        print(f"\nDropped {before - len(combined)} rows below confidence {args.min_confidence}")

    # Assign missing splits
    combined = _reassign_splits(combined)

    # Validate
    bad_video    = combined["video"].isna() | (combined["video"].astype(str).str.strip() == "")
    bad_exercise = combined["exercise"].isna() | (combined["exercise"].astype(str).str.strip() == "")
    if bad_video.any() or bad_exercise.any():
        combined = combined[~bad_video & ~bad_exercise]
        print(f"Dropped {bad_video.sum() + bad_exercise.sum()} rows with empty video/exercise")

    # Task annotation
    combined["_tasks"] = combined.apply(row_tasks, axis=1)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("UNIFIED DATASET SUMMARY")
    print("=" * 60)
    print(f"Total rows           : {len(combined)}")
    print(f"Unique videos        : {combined['video'].nunique()}")
    print(f"Unique exercises     : {combined['exercise'].nunique()}")
    print()

    print("Rows per source dataset:")
    for src, grp in combined.groupby("source_dataset"):
        n_with_reps = grp[grp["rep_start"].astype(str).str.strip() != ""].shape[0]
        n_with_form = grp["form_label"].astype(str).str.strip().apply(lambda x: x in ("0","1")).sum()
        print(f"  {src:30s}: {len(grp):6d} rows | "
              f"{grp['video'].nunique():5d} videos | "
              f"rep_times={n_with_reps} | form_labels={n_with_form}")

    print()
    print("Rows per exercise:")
    for ex, grp in combined.groupby("exercise"):
        srcs = sorted(grp["source_dataset"].unique())
        print(f"  {ex:28s}: {len(grp):6d} rows  (sources: {', '.join(srcs)})")

    print()
    print("Split distribution:")
    split_counts = combined["split"].value_counts()
    for split, n in split_counts.items():
        print(f"  {split:5s}: {n:6d} rows")

    print()
    print("Task coverage:")
    for task in TASKS:
        n = sum(1 for tasks in combined["_tasks"] if task in tasks)
        print(f"  {task:30s}: {n:6d} rows")

    print()
    print("Task support per dataset:")
    for src, support in DATASET_TASK_SUPPORT.items():
        status = "AVAILABLE" if any(
            getattr(a, "SOURCE", "") == src and a.check_available() for a in ADAPTERS
        ) else "NOT DOWNLOADED"
        print(f"\n  [{src}]  ({status})")
        for task, desc in support.items():
            print(f"    {task:30s}: {desc}")

    # Drop internal column before saving
    combined = combined.drop(columns=["_tasks"], errors="ignore")

    if args.dry_run:
        print("\n[dry_run] No file written.")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"\nWrote {len(combined)} rows -> {args.out}")


if __name__ == "__main__":
    main()
