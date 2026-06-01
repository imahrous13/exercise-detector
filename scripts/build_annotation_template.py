"""Emit a CSV template for manual rep annotation from split CSVs.

Usage:
    python scripts/build_annotation_template.py --splits_dir data/splits \\
        --output data/annotations/reps_template.csv
"""

import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feedback.form_rules import get_exercise_names


def main():
    parser = argparse.ArgumentParser(description='Build manual annotation CSV template')
    parser.add_argument('--splits_dir', type=str, default='data/splits')
    parser.add_argument('--output', type=str, default='data/annotations/reps_template.csv')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    config = None
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    names = get_exercise_names(config)

    rows = []
    seen = set()
    for split in ('train', 'val', 'test'):
        csv_path = os.path.join(args.splits_dir, f'{split}.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            fn = row['filename']
            if fn in seen:
                continue
            seen.add(fn)
            ex_idx = int(row['exercise'])
            ex_name = names[ex_idx] if ex_idx < len(names) else f"class_{ex_idx}"
            rows.append({
                'video': fn,
                'exercise': ex_name,
                'rep_start': '',
                'rep_end': '',
                'form_label': '',
                'mistake_type': 'none',
            })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} video rows to {args.output}")
    print("Fill rep_start, rep_end, form_label (correct/incorrect) for each rep in each video.")


if __name__ == '__main__':
    main()
