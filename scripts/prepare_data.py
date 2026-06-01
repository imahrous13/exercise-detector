"""Prepare the full dataset: extract keypoints from all videos, create train/val/test splits.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --video_dir ../workoutfitness-video --skip_extraction
"""

import argparse
import os
import sys
import random
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pose_extraction.extractor import PoseExtractor
from src.preprocessing.normalize import preprocess_skeleton
from src.feedback.form_rules import DIR_TO_EXERCISE, EXERCISE_NAMES, get_exercise_names


def build_dir_to_exercise(config=None):
    """Build folder name -> exercise index from config, or return built-in DIR_TO_EXERCISE."""
    if config and isinstance(config.get('data'), dict):
        exercises = config['data'].get('exercises')
        if exercises:
            mapping = {}
            for idx, ex in enumerate(exercises):
                mapping[ex] = idx
                mapping[ex.replace('_', ' ')] = idx  # allow "bench press" or "bench_press"
            return mapping
    return DIR_TO_EXERCISE


def extract_all_keypoints(video_dir, output_dir, model_size='s', confidence=0.5,
                          smoothing_alpha=0.35, resume=False, config=None):
    """Extract keypoints from all videos organized in subdirectories."""
    dir_to_exercise = build_dir_to_exercise(config)
    exercise_names = get_exercise_names(config)
    pose_cfg = config.get("pose", {}) if config else {}
    extractor = PoseExtractor(
        model_size=model_size,
        confidence_threshold=confidence,
        smoothing_alpha=smoothing_alpha,
        roi_padding_ratio=pose_cfg.get("roi_padding_ratio", 0.2),
    )

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    results = []  # (filename, exercise_idx, form_label)
    failed = []

    # If resuming, find already-processed files
    existing_files = set()
    if resume and os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith('.npy') and not f.endswith('_angles.npy') and not f.endswith('_raw.npy'):
                existing_files.add(f[:-4])  # remove .npy
        print(f"Resume mode: found {len(existing_files)} already-processed files")

    # Walk through exercise subdirectories
    for subdir in sorted(os.listdir(video_dir)):
        subdir_path = os.path.join(video_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # Map directory name to exercise index (config-driven or built-in)
        if subdir not in dir_to_exercise:
            print(f"WARNING: Unknown exercise directory '{subdir}', skipping")
            continue

        exercise_idx = dir_to_exercise[subdir]
        exercise_name = exercise_names[exercise_idx] if exercise_idx < len(exercise_names) else f"class_{exercise_idx}"

        # Get all video files
        video_files = [
            f for f in os.listdir(subdir_path)
            if Path(f).suffix.lower() in video_extensions
        ]

        if not video_files:
            continue

        # In resume mode, check how many need processing
        if resume:
            to_process = []
            already_done = []
            for vf in video_files:
                unique_name = f"{exercise_name}_{Path(vf).stem}"
                if unique_name in existing_files:
                    already_done.append((unique_name, exercise_idx))
                else:
                    to_process.append(vf)
            results.extend(already_done)
            if not to_process:
                print(f"[{exercise_name}] All {len(video_files)} videos already processed, skipping")
                continue
            print(f"\n[{exercise_name}] {len(already_done)} done, processing {len(to_process)} remaining...")
            video_files = to_process
        else:
            print(f"\n[{exercise_name}] Processing {len(video_files)} videos...")

        for vf in tqdm(video_files, desc=exercise_name):
            video_path = os.path.join(subdir_path, vf)
            video_stem = Path(vf).stem
            # Make filename unique with exercise prefix
            unique_name = f"{exercise_name}_{video_stem}"

            try:
                # Extract keypoints
                keypoints, fps = extractor.extract_from_video(video_path)

                if keypoints.shape[0] < 10:
                    print(f"  SKIP {vf}: too few frames ({keypoints.shape[0]})")
                    failed.append((vf, "too few frames"))
                    continue

                # Preprocess
                normalized_kpts, angles = preprocess_skeleton(keypoints)

                # Save
                os.makedirs(output_dir, exist_ok=True)
                np.save(os.path.join(output_dir, f"{unique_name}.npy"), normalized_kpts)
                np.save(os.path.join(output_dir, f"{unique_name}_angles.npy"), angles)

                # Form/rep labels come from manual annotations (data/annotations/reps.csv)
                results.append((unique_name, exercise_idx))

            except Exception as e:
                print(f"  ERROR {vf}: {e}")
                failed.append((vf, str(e)))

    print(f"\n{'='*50}")
    print(f"Extraction complete: {len(results)} succeeded, {len(failed)} failed")
    return results, failed


def create_splits(results, splits_dir, train_ratio=0.7, val_ratio=0.15, config=None):
    """Create train/val/test CSV splits."""
    os.makedirs(splits_dir, exist_ok=True)
    ex_names = get_exercise_names(config) if config else EXERCISE_NAMES

    # Shuffle
    random.seed(42)
    random.shuffle(results)

    n = len(results)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = results[:n_train]
    val_data = results[n_train:n_train + n_val]
    test_data = results[n_train + n_val:]

    # Save CSVs
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        df = pd.DataFrame(split_data, columns=['filename', 'exercise'])
        csv_path = os.path.join(splits_dir, f'{split_name}.csv')
        df.to_csv(csv_path, index=False)
        print(f"{split_name}: {len(split_data)} videos")

        # Exercise distribution
        ex_counts = df['exercise'].value_counts().sort_index()
        for idx, count in ex_counts.items():
            name = ex_names[idx] if idx < len(ex_names) else f"class_{idx}"
            print(f"  {name}: {count}")

    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description='Prepare exercise dataset')
    parser.add_argument('--video_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', '..', 'workoutfitness-video'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'skeletons'))
    parser.add_argument('--splits_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'splits'))
    parser.add_argument('--model_size', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--smoothing_alpha', type=float, default=0.35,
                        help='EMA smoothing (0=off, 0.2-0.5 typical)')
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Skip keypoint extraction, only create splits from existing .npy files')
    parser.add_argument('--resume', action='store_true',
                        help='Resume extraction, skip already processed videos')
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml'),
                        help='Config YAML (defines data.exercises for folder mapping)')
    args = parser.parse_args()

    # Load config for exercise list (robust to any data)
    config = None
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Resolve paths
    video_dir = os.path.abspath(args.video_dir)
    output_dir = os.path.abspath(args.output_dir)
    splits_dir = os.path.abspath(args.splits_dir)

    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Splits directory: {splits_dir}")

    if args.skip_extraction:
        # Rebuild results from existing .npy files
        print("\nSkipping extraction, reading existing .npy files...")
        exercise_names = get_exercise_names(config)
        results = []
        for f in sorted(os.listdir(output_dir)):
            if f.endswith('.npy') and not f.endswith('_angles.npy') and not f.endswith('_raw.npy'):
                name = f[:-4]  # remove .npy
                for ex_idx, ex_name in enumerate(exercise_names):
                    if name.startswith(ex_name + '_'):
                        results.append((name, ex_idx))
                        break
        print(f"Found {len(results)} skeleton files")
    else:
        # Full extraction
        if not os.path.exists(video_dir):
            print(f"ERROR: Video directory not found: {video_dir}")
            return

        results, failed = extract_all_keypoints(
            video_dir=video_dir,
            output_dir=output_dir,
            model_size=args.model_size,
            confidence=args.confidence,
            smoothing_alpha=args.smoothing_alpha,
            resume=args.resume,
            config=config,
        )

    if not results:
        print("ERROR: No data to create splits from!")
        return

    # Create splits
    print(f"\nCreating train/val/test splits...")
    create_splits(results, splits_dir, config=config)
    print("\nDone! Labeling (hybrid — train immediately, refine labels over time):")
    print("  • Unlabeled videos: auto rule-based reps at train time (segment_reps)")
    print("  • Optional: python scripts/build_annotation_template.py")
    print("  • Optional: fill data/annotations/reps.csv for human ground truth")
    print("  • Optional: python scripts/auto_label_unannotated.py  (export rules to CSV)")
    print("  • Train: python scripts/train.py --config configs/default.yaml")


if __name__ == '__main__':
    main()
