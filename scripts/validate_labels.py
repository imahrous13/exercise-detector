"""Validate training data labels: exercise consistency and form distribution.

Checks that CSV exercise index matches the filename prefix (exercise name).
Optionally checks for missing skeleton files.

Usage:
    python scripts/validate_labels.py --splits_dir data/splits
    python scripts/validate_labels.py --splits_dir data/splits --skeleton_dir data/processed/skeletons
"""

import argparse
import os
import sys
import yaml
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feedback.form_rules import EXERCISE_NAMES, get_exercise_names


def validate_split(csv_path, skeleton_dir=None, exercise_names=None):
    """Validate labels in a split CSV.

    Returns:
        (mismatches, missing, stats)
        - mismatches: list of (filename, csv_exercise_idx, expected_prefix)
        - missing: list of filenames with missing .npy in skeleton_dir (if provided)
        - stats: dict with total, form counts, exercise counts
    """
    names = exercise_names if exercise_names is not None else EXERCISE_NAMES
    df = pd.read_csv(csv_path)
    mismatches = []
    missing = []

    for _, row in df.iterrows():
        filename = row['filename']
        exercise_idx = int(row['exercise'])
        if exercise_idx < 0 or exercise_idx >= len(names):
            mismatches.append((filename, exercise_idx, f"exercise index {exercise_idx} out of range [0, {len(names)-1}]"))
            continue

        expected_prefix = names[exercise_idx] + "_"
        if not filename.startswith(expected_prefix):
            mismatches.append((filename, exercise_idx, expected_prefix))

        if skeleton_dir:
            skel_path = os.path.join(skeleton_dir, f"{filename}.npy")
            if not os.path.exists(skel_path):
                missing.append(filename)

    total = len(df)
    exercise_counts = df['exercise'].value_counts().sort_index()

    stats = {
        'total': total,
        'exercise_counts': exercise_counts,
    }
    return mismatches, missing, stats


def main():
    parser = argparse.ArgumentParser(description='Validate training data labels')
    parser.add_argument('--splits_dir', type=str, default='data/splits')
    parser.add_argument('--skeleton_dir', type=str, default=None,
                        help='If set, check that skeleton .npy files exist')
    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML; if set, exercise list from data.exercises')
    args = parser.parse_args()

    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    exercise_names = get_exercise_names(config)

    if not os.path.isdir(args.splits_dir):
        print(f"ERROR: Splits directory not found: {args.splits_dir}")
        return

    all_mismatches = []
    all_missing = []

    for split_name in ['train', 'val', 'test']:
        csv_path = os.path.join(args.splits_dir, f'{split_name}.csv')
        if not os.path.exists(csv_path):
            continue

        mismatches, missing, stats = validate_split(csv_path, args.skeleton_dir, exercise_names=exercise_names)
        all_mismatches.extend([(split_name, f, ex, pre) for f, ex, pre in mismatches])
        all_missing.extend(missing)

        print(f"\n{'='*60}")
        print(f"Split: {split_name}.csv")
        print(f"{'='*60}")
        print(f"Total videos: {stats['total']}")
        if mismatches:
            print(f"\n*** LABEL MISMATCHES ({len(mismatches)}): ***")
            for fn, ex_idx, expected in mismatches[:20]:
                name = exercise_names[ex_idx] if ex_idx < len(exercise_names) else "?"
                print(f"  {fn}  -> exercise={ex_idx} ({name})  expected prefix: {expected}")
            if len(mismatches) > 20:
                print(f"  ... and {len(mismatches)-20} more")
        else:
            print("Exercise labels: all filenames match exercise index (OK)")

        if missing:
            print(f"\n*** MISSING FILES ({len(missing)}): ***")
            for fn in missing[:10]:
                print(f"  {fn}.npy")
            if len(missing) > 10:
                print(f"  ... and {len(missing)-10} more")
        elif args.skeleton_dir:
            print("Skeleton files: all present (OK)")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if all_mismatches:
        print(f"Total exercise label mismatches across splits: {len(all_mismatches)}")
        print("  -> Fix: ensure prepare_data.py uses correct DIR_TO_EXERCISE and folder names.")
    else:
        print("Exercise labels: no mismatches found.")
    if all_missing:
        print(f"Total missing skeleton files: {len(all_missing)}")
    else:
        if args.skeleton_dir:
            print("All skeleton files present.")
    print("\nForm/rep training labels come from MANUAL annotations (data/annotations/reps.csv).")
    print("Validate annotations: python scripts/validate_annotations.py --annotations data/annotations/reps.csv")


if __name__ == '__main__':
    main()
