import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.feedback.form_rules import EXERCISE_NAMES, get_exercise_names
from src.data.annotations import get_annotations_path
from src.data.labeling import (
    get_video_reps,
    label_window,
    labeling_config,
    load_annotations_index_safe,
)


COCO_FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
]

COCO_PARENTS = [
    -1, 0, 0, 1, 2, -1, -1, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14,
]


def compute_extra_features(skeleton):
    """(T, 17, 3) → (T, 17, 6) with velocity and bone lengths."""
    T, V, C = skeleton.shape
    result = np.zeros((T, V, 6), dtype=np.float32)
    result[:, :, :3] = skeleton

    if T > 1:
        result[1:, :, 3] = skeleton[1:, :, 0] - skeleton[:-1, :, 0]
        result[1:, :, 4] = skeleton[1:, :, 1] - skeleton[:-1, :, 1]
        result[0, :, 3:5] = result[1, :, 3:5]

    for j in range(V):
        parent = COCO_PARENTS[j]
        if parent >= 0:
            dx = skeleton[:, j, 0] - skeleton[:, parent, 0]
            dy = skeleton[:, j, 1] - skeleton[:, parent, 1]
            result[:, j, 5] = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)

    return result


class SkeletonAugmentor:
    """Skeleton-level data augmentations applied during training."""

    def __init__(self, config):
        aug_cfg = config.get('data', {}).get('augmentation', {})
        self.enabled = aug_cfg.get('enabled', False)
        self.noise_std = aug_cfg.get('gaussian_noise_std', 0.015)
        self.horizontal_flip = aug_cfg.get('horizontal_flip', True)
        self.scale_range = aug_cfg.get('random_scale_range', [0.9, 1.1])
        self.crop_min = aug_cfg.get('temporal_crop_min', 25)
        self.joint_drop_prob = aug_cfg.get('joint_dropout_prob', 0.1)
        self.joint_drop_max = aug_cfg.get('joint_dropout_max', 2)

    def __call__(self, skeleton, angles):
        if not self.enabled:
            return skeleton, angles

        skeleton = skeleton.copy()
        angles = angles.copy()

        if self.horizontal_flip and np.random.random() < 0.5:
            skeleton, angles = self._flip(skeleton, angles)

        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, skeleton[:, :, :2].shape)
            skeleton[:, :, :2] += noise.astype(np.float32)

        lo, hi = self.scale_range
        skeleton[:, :, :2] *= np.random.uniform(lo, hi)

        T = skeleton.shape[0]
        crop_len = np.random.randint(self.crop_min, T + 1)
        if crop_len < T:
            start = np.random.randint(0, T - crop_len + 1)
            skeleton = self._temporal_resize(skeleton[start:start + crop_len], T)
            angles = self._temporal_resize_2d(angles[start:start + crop_len], T)

        if self.joint_drop_prob > 0 and np.random.random() < self.joint_drop_prob:
            n_drop = np.random.randint(1, self.joint_drop_max + 1)
            skeleton[:, np.random.choice(17, size=n_drop, replace=False), :] = 0.0

        return skeleton, angles

    def _flip(self, skeleton, angles):
        skeleton[:, :, 0] = -skeleton[:, :, 0]
        for left, right in COCO_FLIP_PAIRS:
            skeleton[:, [left, right], :] = skeleton[:, [right, left], :]
        for i in range(0, 12, 2):
            angles[:, [i, i + 1]] = angles[:, [i + 1, i]]
        return skeleton, angles

    @staticmethod
    def _temporal_resize(arr, target_len):
        L = arr.shape[0]
        if L == target_len:
            return arr
        old_idx = np.linspace(0, L - 1, L)
        new_idx = np.linspace(0, L - 1, target_len)
        result = np.zeros((target_len, arr.shape[1], arr.shape[2]), dtype=np.float32)
        for j in range(arr.shape[1]):
            for c in range(arr.shape[2]):
                result[:, j, c] = np.interp(new_idx, old_idx, arr[:, j, c])
        return result

    @staticmethod
    def _temporal_resize_2d(arr, target_len):
        L = arr.shape[0]
        if L == target_len:
            return arr
        old_idx = np.linspace(0, L - 1, L)
        new_idx = np.linspace(0, L - 1, target_len)
        result = np.zeros((target_len, arr.shape[1]), dtype=np.float32)
        for c in range(arr.shape[1]):
            result[:, c] = np.interp(new_idx, old_idx, arr[:, c])
        return result


class SkeletonDataset(Dataset):
    """Skeleton windows with hybrid labels: manual annotations or rule fallback.

    - Videos with rows in reps.csv → human ground truth (priority).
    - Unlabeled videos → segment_reps() + score_form() automatically.
    - Training can start immediately; replace rule labels with manual over time.
    """

    def __init__(self, skeleton_dir, labels_file, window_size=30, stride=15,
                 augment=False, config=None, annotations_index=None,
                 labeling_cfg=None, rep_cache=None):
        self.skeleton_dir = skeleton_dir
        self.window_size = window_size
        self.stride = stride
        self.augmentor = SkeletonAugmentor(config) if augment and config else None
        self.exercise_names = get_exercise_names(config) if config else EXERCISE_NAMES

        self.labeling_cfg = labeling_cfg or labeling_config(config)
        self.source_mode = self.labeling_cfg["source"]
        self.fallback_to_rules = self.labeling_cfg["fallback_to_rules"]
        self.rules_rep_completion = self.labeling_cfg["rules_rep_completion_in_window"]

        if self.source_mode == "rules":
            self.fallback_to_rules = True
            self.manual_index = {}
        elif annotations_index is not None:
            self.manual_index = annotations_index
        else:
            ann_path = get_annotations_path(config)
            self.manual_index = load_annotations_index_safe(ann_path, self.exercise_names)

        if self.source_mode == "manual" and not self.manual_index:
            raise FileNotFoundError(
                "labeling.source=manual requires a populated annotations file. "
                "Use source: hybrid (default) to auto-label unannotated videos with rules."
            )

        self.rep_cache = rep_cache if rep_cache is not None else {}
        self.labels_df = pd.read_csv(labels_file)
        self.windows = []
        self.video_sources = {}  # filename -> manual|rules|none
        self._build_windows()

    def _build_windows(self):
        for _, row in self.labels_df.iterrows():
            filename = row['filename']
            exercise = int(row['exercise'])
            exercise_name = (
                self.exercise_names[exercise]
                if exercise < len(self.exercise_names)
                else f"class_{exercise}"
            )

            skel_path = os.path.join(self.skeleton_dir, f"{filename}.npy")
            angles_path = os.path.join(self.skeleton_dir, f"{filename}_angles.npy")
            if not os.path.exists(skel_path):
                continue

            skeleton = np.load(skel_path)
            T = skeleton.shape[0]
            angles = np.load(angles_path) if os.path.exists(angles_path) else None

            allow_rules = self.fallback_to_rules and self.source_mode in ("hybrid", "rules", "unified")
            reps, src = get_video_reps(
                filename, exercise_name, angles, self.manual_index,
                allow_rules_fallback=allow_rules,
                cache=self.rep_cache,
            )
            self.video_sources[filename] = src

            def _add_window(start, end, skel_slice, ang_slice):
                win_angles = ang_slice
                rep_l, form_l, form_valid, mistake = label_window(
                    start, end, reps,
                    exercise_name=exercise_name,
                    win_angles=win_angles,
                    use_completion_frame_for_rules=self.rules_rep_completion,
                    label_source=src,
                )
                self.windows.append({
                    'skeleton': skel_slice,
                    'angles': ang_slice,
                    'exercise': exercise,
                    'form': form_l,
                    'rep': rep_l,
                    'form_valid': form_valid,
                    'mistake_type': mistake or 'none',
                    'label_source': src,
                    'filename': filename,
                    'start_frame': start,
                })

            if T < self.window_size:
                pad_size = self.window_size - T
                skeleton = np.pad(skeleton, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
                if angles is not None:
                    angles = np.pad(angles, ((0, pad_size), (0, 0)), mode='edge')
                win_angles = (
                    angles[:self.window_size]
                    if angles is not None
                    else np.zeros((self.window_size, 12), dtype=np.float32)
                )
                _add_window(0, self.window_size, skeleton[:self.window_size], win_angles)
            else:
                for start in range(0, T - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    win_angles = (
                        angles[start:end]
                        if angles is not None
                        else np.zeros((self.window_size, 12), dtype=np.float32)
                    )
                    _add_window(start, end, skeleton[start:end], win_angles)

        if not self.windows:
            raise ValueError(
                f"No training windows built from {self.labels_df.shape[0]} split rows. "
                "Check skeleton_dir paths and .npy files."
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        skeleton = window['skeleton'].copy()
        angles = window['angles'].copy()

        if self.augmentor is not None:
            skeleton, angles = self.augmentor(skeleton, angles)

        return {
            'skeleton': torch.FloatTensor(compute_extra_features(skeleton)),
            'angles': torch.FloatTensor(angles),
            'exercise': torch.LongTensor([window['exercise']])[0],
            'form': torch.LongTensor([window['form']])[0],
            'rep': torch.LongTensor([window['rep']])[0],
            'form_valid': torch.tensor(window['form_valid'], dtype=torch.bool),
        }

    def get_class_counts(self):
        labels = [w['exercise'] for w in self.windows]
        counts = Counter(labels)
        weights = [1.0 / counts[label] for label in labels]
        return counts, weights

    def get_label_stats(self):
        rep_pos = sum(1 for w in self.windows if w['rep'] == 1)
        form_pos = sum(1 for w in self.windows if w['form'] == 1 and w['form_valid'])
        form_valid = sum(1 for w in self.windows if w['form_valid'])
        by_source = Counter(self.video_sources.values())
        return {
            'total_windows': len(self.windows),
            'rep_positive': rep_pos,
            'form_correct': form_pos,
            'form_valid_windows': form_valid,
            'videos_manual': by_source.get('manual', 0),
            'videos_rules': by_source.get('rules', 0),
            'videos_none': by_source.get('none', 0),
        }


def create_dataloaders(skeleton_dir, splits_dir, window_size=30, stride=15,
                       batch_size=32, num_workers=0, config=None, project_root=None):
    """Create dataloaders with hybrid manual + rule-based labeling."""
    cfg = labeling_config(config)
    ann_path = get_annotations_path(config, project_root=project_root)
    manual_index = load_annotations_index_safe(ann_path, get_exercise_names(config))

    if ann_path and manual_index:
        print(f"Manual annotations: {ann_path} ({len(manual_index)} videos)")
    elif ann_path:
        print(f"Manual annotations file exists but empty: {ann_path}")
    else:
        print("No manual annotations file — all videos use rule-based fallback.")

    print(
        f"Labeling mode: {cfg['source']} | "
        f"rule fallback: {cfg['fallback_to_rules']}"
    )

    rep_cache = {}
    loaders = {}
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(splits_dir, f'{split}.csv')
        if not os.path.exists(csv_path):
            continue

        is_train = (split == 'train')
        dataset = SkeletonDataset(
            skeleton_dir=skeleton_dir,
            labels_file=csv_path,
            window_size=window_size,
            stride=stride if is_train else window_size,
            augment=is_train,
            config=config,
            annotations_index=manual_index,
            labeling_cfg=cfg,
            rep_cache=rep_cache,
        )
        stats = dataset.get_label_stats()
        print(
            f"  {split}: {stats['total_windows']} windows | "
            f"rep+ {stats['rep_positive']} | "
            f"videos manual={stats['videos_manual']} rules={stats['videos_rules']} "
            f"none={stats['videos_none']}"
        )

        sampler = None
        shuffle = is_train
        if (
            is_train
            and config
            and config.get('training', {}).get('class_balanced_sampling', False)
        ):
            _, sample_weights = dataset.get_class_counts()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True,
            )
            shuffle = False

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
        )

    if not loaders:
        raise FileNotFoundError(
            f"No split CSVs found in {splits_dir}. Run prepare_data.py first."
        )

    return loaders
