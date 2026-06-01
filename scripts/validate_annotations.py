"""Validate manual rep annotation files against split CSVs and skeleton files.

Usage:
    python scripts/validate_annotations.py --annotations data/annotations/reps.csv
    python scripts/validate_annotations.py --annotations data/annotations/reps.csv \\
        --splits_dir data/splits --skeleton_dir data/processed/skeletons --config configs/default.yaml
"""

import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.annotations import load_rep_annotations, resolve_skeleton_filename
from src.feedback.form_rules import get_exercise_names


def main():
    parser = argparse.ArgumentParser(description='Validate manual rep annotations')
    parser.add_argument('--annotations', type=str, required=True,
                        help='Path to reps.csv or reps.json')
    parser.add_argument('--splits_dir', type=str, default='data/splits')
    parser.add_argument('--skeleton_dir', type=str, default=None,
                        help='If set, verify rep_end < video frame count')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    exercise_names = get_exercise_names(config)

    index = load_rep_annotations(args.annotations, exercise_names)
    print(f"Annotation file: {args.annotations}")
    print(f"Videos with reps: {len(index)}")
    total_reps = sum(len(v.reps) for v in index.values())
    correct_reps = sum(
        1 for v in index.values() for r in v.reps if r.form_label == 1
    )
    print(f"Total rep segments: {total_reps} ({correct_reps} correct, {total_reps - correct_reps} incorrect)")

    split_filenames = set()
    if os.path.isdir(args.splits_dir):
        for split in ('train', 'val', 'test'):
            csv_path = os.path.join(args.splits_dir, f'{split}.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                split_filenames.update(df['filename'].astype(str))

    if split_filenames:
        missing_in_ann = [f for f in split_filenames if f not in index]
        ann_not_in_split = [k for k in index if k not in split_filenames]
        print(f"\nSplit videos: {len(split_filenames)}")
        if missing_in_ann:
            print(f"  WARNING: {len(missing_in_ann)} split videos lack annotations (first 10):")
            for f in missing_in_ann[:10]:
                print(f"    - {f}")
        else:
            print("  All split videos have at least one annotation entry (OK)")
        if ann_not_in_split:
            print(f"  Note: {len(ann_not_in_split)} annotated videos not in any split CSV")

    if args.skeleton_dir:
        import numpy as np
        errors = []
        for video_key, va in index.items():
            skel_path = os.path.join(args.skeleton_dir, f"{video_key}.npy")
            if not os.path.exists(skel_path):
                errors.append((video_key, "missing skeleton .npy"))
                continue
            T = np.load(skel_path).shape[0]
            for rep in va.reps:
                if rep.rep_end >= T:
                    errors.append((
                        video_key,
                        f"rep_end {rep.rep_end} >= video length {T}",
                    ))
                if rep.rep_start < 0:
                    errors.append((video_key, f"rep_start {rep.rep_start} < 0"))

        if errors:
            print(f"\n*** FRAME BOUNDS ERRORS ({len(errors)}): ***")
            for vk, msg in errors[:20]:
                print(f"  {vk}: {msg}")
        else:
            print("\nFrame bounds vs skeleton length: OK")

    mistake_types = {}
    for va in index.values():
        for r in va.reps:
            mistake_types[r.mistake_type] = mistake_types.get(r.mistake_type, 0) + 1
    if mistake_types:
        print("\nMistake type distribution:")
        for mt, n in sorted(mistake_types.items(), key=lambda x: -x[1]):
            print(f"  {mt}: {n}")

    print("\nHybrid labeling: manual rows take priority; unlabeled videos use rules at train time.")
    print("Export rule rows: python scripts/auto_label_unannotated.py --dry-run")


if __name__ == '__main__':
    main()
