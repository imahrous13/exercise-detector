"""Validate data/annotations/unified_reps.csv for training readiness.

Checks:
  - File exists and is non-empty
  - Required columns present
  - No empty video/exercise fields
  - rep_start < rep_end for all rows that have both values
  - form_label is 0, 1, or "" only
  - Task coverage per source and per exercise
  - Which tasks each dataset can actually train

Usage:
  python scripts/validate_unified.py
  python scripts/validate_unified.py --file data/annotations/unified_reps.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.datasets.utils import UNIFIED_COLUMNS, DATASET_TASK_SUPPORT, TASKS, row_tasks

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_FILE = os.path.join(PROJECT_ROOT, "data", "annotations", "unified_reps.csv")

SEP = "=" * 65

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=DEFAULT_FILE)
    args = p.parse_args()

    # ----------------------------------------------------------------
    # 1. File existence
    # ----------------------------------------------------------------
    section("1. File check")
    if not os.path.isfile(args.file):
        print(f"  FAIL: {args.file} does not exist")
        print("  Run:  python scripts/build_unified_reps.py")
        sys.exit(1)

    df = pd.read_csv(args.file)
    print(f"  File   : {args.file}")
    print(f"  Rows   : {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    assert len(df) > 0, "FAIL: file is empty"
    print("  [OK] file is non-empty")

    # ----------------------------------------------------------------
    # 2. Required columns
    # ----------------------------------------------------------------
    section("2. Schema check")
    required = {"source_dataset", "video", "exercise", "rep_start",
                "rep_end", "form_label", "mistake_type", "confidence", "split"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(f"  FAIL: missing columns: {sorted(missing_cols)}")
        sys.exit(1)
    print(f"  [OK] all {len(required)} required columns present")

    # ----------------------------------------------------------------
    # 3. No empty video / exercise
    # ----------------------------------------------------------------
    section("3. Empty field check")
    bad_video    = df["video"].isna() | (df["video"].astype(str).str.strip() == "")
    bad_exercise = df["exercise"].isna() | (df["exercise"].astype(str).str.strip() == "")
    issues = []
    if bad_video.any():
        issues.append(f"  FAIL: {bad_video.sum()} rows with empty video")
    if bad_exercise.any():
        issues.append(f"  FAIL: {bad_exercise.sum()} rows with empty exercise")
    if issues:
        for i in issues:
            print(i)
        sys.exit(1)
    print(f"  [OK] no empty video or exercise fields")

    # ----------------------------------------------------------------
    # 4. rep_start < rep_end
    # ----------------------------------------------------------------
    section("4. Temporal bounds check")
    has_bounds = (
        df["rep_start"].astype(str).str.strip().ne("") &
        df["rep_end"].astype(str).str.strip().ne("")
    )
    bounded = df[has_bounds].copy()
    bounded["rep_start"] = pd.to_numeric(bounded["rep_start"], errors="coerce")
    bounded["rep_end"]   = pd.to_numeric(bounded["rep_end"],   errors="coerce")
    bad_bounds = bounded[bounded["rep_start"] >= bounded["rep_end"]]
    if not bad_bounds.empty:
        print(f"  FAIL: {len(bad_bounds)} rows where rep_start >= rep_end:")
        print(bad_bounds[["video", "rep_start", "rep_end"]].head(10).to_string(index=False))
        sys.exit(1)
    print(f"  [OK] {has_bounds.sum()} rows with temporal bounds, all valid "
          f"({(~has_bounds).sum()} rows without bounds — OK for classification-only)")

    # ----------------------------------------------------------------
    # 5. form_label values
    # ----------------------------------------------------------------
    section("5. form_label check")
    fl = df["form_label"].astype(str).str.strip()
    invalid_fl = fl[~fl.isin(["0", "1", ""])]
    if not invalid_fl.empty:
        print(f"  FAIL: {len(invalid_fl)} rows with invalid form_label values:")
        print(invalid_fl.value_counts().head(10).to_string())
        sys.exit(1)
    n_form_0 = (fl == "0").sum()
    n_form_1 = (fl == "1").sum()
    n_form_x = (fl == "").sum()
    print(f"  [OK] form_label: {n_form_1} correct | {n_form_0} incorrect | "
          f"{n_form_x} unlabeled")

    # ----------------------------------------------------------------
    # 6. Counts per source dataset
    # ----------------------------------------------------------------
    section("6. Samples per source dataset")
    print(f"  {'Source':30s}  {'Rows':>6s}  {'Videos':>6s}  {'RepTimes':>8s}  {'FormLabels':>10s}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*10}")
    for src, grp in df.groupby("source_dataset"):
        n_rep   = (grp["rep_start"].astype(str).str.strip() != "").sum()
        n_form  = grp["form_label"].astype(str).str.strip().isin(["0","1"]).sum()
        print(f"  {src:30s}  {len(grp):6d}  {grp['video'].nunique():6d}  "
              f"{n_rep:8d}  {n_form:10d}")

    # ----------------------------------------------------------------
    # 7. Counts per exercise
    # ----------------------------------------------------------------
    section("7. Samples per exercise")
    print(f"  {'Exercise':28s}  {'Rows':>6s}  {'Videos':>6s}  {'Sources'}")
    print(f"  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*30}")
    for ex, grp in df.groupby("exercise"):
        srcs = ", ".join(sorted(grp["source_dataset"].unique()))
        print(f"  {ex:28s}  {len(grp):6d}  {grp['video'].nunique():6d}  {srcs}")

    # ----------------------------------------------------------------
    # 8. Task coverage
    # ----------------------------------------------------------------
    section("8. Task coverage (rows that can train each task)")
    df["_tasks"] = df.apply(row_tasks, axis=1)
    for task in TASKS:
        n       = sum(1 for tasks in df["_tasks"] if task in tasks)
        sources = sorted({
            row["source_dataset"]
            for _, row in df.iterrows()
            if task in row["_tasks"]
        })
        pct = 100 * n / len(df) if len(df) else 0
        print(f"  {task:30s}: {n:6d} rows ({pct:.0f}%)  from {sources}")

    # ----------------------------------------------------------------
    # 9. Task support matrix
    # ----------------------------------------------------------------
    section("9. Task support matrix per dataset")
    available = set()
    try:
        from scripts.datasets import (
            adapt_captured, adapt_repcount, adapt_countix,
            adapt_mmfit, adapt_realtime, adapt_roboflow,
        )
        for a in (adapt_captured, adapt_repcount, adapt_countix,
                  adapt_mmfit, adapt_realtime, adapt_roboflow):
            src = getattr(a, "SOURCE", "")
            if src and a.check_available():
                available.add(src)
    except ImportError:
        pass

    for src, support in DATASET_TASK_SUPPORT.items():
        status = "[AVAILABLE]" if src in available else "[NOT DOWNLOADED]"
        print(f"\n  {src}  {status}")
        for task, desc in support.items():
            print(f"    {task:30s}: {desc}")

    # ----------------------------------------------------------------
    # 10. Split distribution
    # ----------------------------------------------------------------
    section("10. Split distribution")
    split_counts = df["split"].value_counts()
    for split, n in split_counts.items():
        pct = 100 * n / len(df)
        print(f"  {split:5s}: {n:6d} rows ({pct:.0f}%)")

    print(f"\n{SEP}")
    print("  VALIDATION PASSED")
    print(SEP)
    print(f"  unified_reps.csv is ready for training.")
    print(f"  Training config:  source=unified, annotations_file=data/annotations/unified_reps.csv")
    print()


if __name__ == "__main__":
    main()
