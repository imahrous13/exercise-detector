"""Append rule-based rep rows to annotations CSV for videos not yet manually labeled.

Human rows in reps.csv are never overwritten. Run after prepare_data when you want
a filled annotations file to review/edit.

Usage:
    python scripts/auto_label_unannotated.py
    python scripts/auto_label_unannotated.py --dry-run
"""

import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.annotations import get_annotations_path, load_rep_annotations
from src.data.labeling import load_annotations_index_safe, reps_from_rules
from src.feedback.form_rules import get_exercise_names


def collect_split_filenames(splits_dir):
    names = set()
    for split in ('train', 'val', 'test'):
        path = os.path.join(splits_dir, f'{split}.csv')
        if os.path.exists(path):
            names.update(pd.read_csv(path)['filename'].astype(str))
    return names


def main():
    parser = argparse.ArgumentParser(description='Auto-fill unannotated videos in reps.csv')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--skeleton_dir', type=str, default='data/processed/skeletons')
    parser.add_argument('--splits_dir', type=str, default='data/splits')
    parser.add_argument('--output', type=str, default=None,
                        help='Defaults to config data.labeling.annotations_file')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ann_path = args.output or get_annotations_path(config, project_root=project_root)
    if not ann_path:
        ann_path = os.path.join(project_root, 'data', 'annotations', 'reps.csv')

    exercise_names = get_exercise_names(config)
    manual_index = load_annotations_index_safe(ann_path, exercise_names)
    split_files = collect_split_filenames(args.splits_dir)

    new_rows = []
    for filename in sorted(split_files):
        if filename in manual_index and manual_index[filename].reps:
            continue

        skel_path = os.path.join(args.skeleton_dir, f'{filename}.npy')
        ang_path = os.path.join(args.skeleton_dir, f'{filename}_angles.npy')
        if not os.path.exists(skel_path) or not os.path.exists(ang_path):
            continue

        ex_idx = None
        for split in ('train', 'val', 'test'):
            p = os.path.join(args.splits_dir, f'{split}.csv')
            if not os.path.exists(p):
                continue
            sdf = pd.read_csv(p)
            if filename in sdf['filename'].values:
                ex_idx = int(sdf.loc[sdf['filename'] == filename, 'exercise'].iloc[0])
                break
        if ex_idx is None:
            continue

        ex_name = exercise_names[ex_idx]
        import numpy as np
        angles = np.load(ang_path)
        reps = reps_from_rules(ex_name, angles)
        if not reps:
            continue

        for rep in reps:
            new_rows.append({
                'video': filename,
                'exercise': ex_name,
                'rep_start': rep.rep_start,
                'rep_end': rep.rep_end,
                'form_label': 'correct' if rep.form_label == 1 else 'incorrect',
                'mistake_type': 'rules_auto',
            })

    print(f"Videos already manually labeled: {len(manual_index)}")
    print(f"New rule-based rep rows to add: {len(new_rows)}")

    if args.dry_run or not new_rows:
        return

    cols = ['video', 'exercise', 'rep_start', 'rep_end', 'form_label', 'mistake_type']
    if os.path.isfile(ann_path):
        existing = pd.read_csv(ann_path)
        if existing.empty:
            combined = pd.DataFrame(new_rows, columns=cols)
        else:
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        os.makedirs(os.path.dirname(ann_path) or '.', exist_ok=True)
        combined = pd.DataFrame(new_rows, columns=cols)

    combined.to_csv(ann_path, index=False)
    print(f"Wrote {len(combined)} total rows to {ann_path}")
    print("Review rows with mistake_type=rules_auto and replace with human labels when ready.")


if __name__ == '__main__':
    main()
