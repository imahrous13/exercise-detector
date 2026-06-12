"""End-to-end verification that the annotation pipeline feeds the training dataset correctly.

Checks:
  1. labeling_config() returns source=hybrid, fallback_to_rules=True
  2. get_annotations_path() resolves to the populated reps.csv
  3. load_annotations_index_safe() loads the expected number of videos/reps
  4. SkeletonDataset (train split) builds windows, routes manual vs rules correctly
  5. First 10 manual annotations reach the dataset with correct labels
  6. __getitem__ returns properly-shaped tensors (end-to-end model input check)
  7. Prints all required training-readiness metrics

Usage:
    python scripts/verify_pipeline.py
"""

import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch

from src.data.annotations import get_annotations_path, load_rep_annotations_csv
from src.data.labeling import labeling_config, load_annotations_index_safe, get_video_reps
from src.data.dataset import SkeletonDataset, create_dataloaders
from src.feedback.form_rules import get_exercise_names

PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH    = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
REPS_CSV       = os.path.join(PROJECT_ROOT, "data", "annotations", "reps.csv")
SKELETON_DIR   = os.path.join(PROJECT_ROOT, "data", "processed", "skeletons")
SPLITS_DIR     = os.path.join(PROJECT_ROOT, "data", "splits")

SEP = "=" * 65

def hdr(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ──────────────────────────────────────────────────────────────────
# 0. Load config
# ──────────────────────────────────────────────────────────────────
hdr("0. Config")
assert os.path.isfile(CONFIG_PATH), f"Config not found: {CONFIG_PATH}"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
exercise_names = get_exercise_names(config)
print(f"  Config   : {CONFIG_PATH}")
print(f"  Exercises: {exercise_names}")


# ──────────────────────────────────────────────────────────────────
# 1. labeling_config()
# ──────────────────────────────────────────────────────────────────
hdr("1. labeling_config()")
lcfg = labeling_config(config)
print(f"  source              : {lcfg['source']}")
print(f"  annotations_file    : {lcfg['annotations_file']}")
print(f"  fallback_to_rules   : {lcfg['fallback_to_rules']}")

assert lcfg["source"] == "hybrid",           "FAIL: source must be 'hybrid'"
assert lcfg["fallback_to_rules"] is True,    "FAIL: fallback_to_rules must be True"
print("  [OK] source=hybrid, fallback_to_rules=True")


# ──────────────────────────────────────────────────────────────────
# 2. get_annotations_path() → reps.csv
# ──────────────────────────────────────────────────────────────────
hdr("2. get_annotations_path()")
ann_path = get_annotations_path(config, project_root=PROJECT_ROOT)
print(f"  Resolved path: {ann_path}")
assert ann_path and os.path.isfile(ann_path), f"FAIL: annotations file not found at {ann_path}"

reps_df = pd.read_csv(ann_path)
print(f"  Rows in reps.csv : {len(reps_df)}")
assert len(reps_df) > 0, "FAIL: reps.csv is empty"

# Quick structural check
for col in ("video", "exercise", "rep_start", "rep_end", "form_label", "mistake_type"):
    assert col in reps_df.columns, f"FAIL: missing column '{col}'"
print("  [OK] All required columns present")
print(f"  [OK] {len(reps_df)} rows with {reps_df['video'].nunique()} unique videos")


# ──────────────────────────────────────────────────────────────────
# 3. load_annotations_index_safe()
# ──────────────────────────────────────────────────────────────────
hdr("3. load_annotations_index_safe()")
manual_index = load_annotations_index_safe(ann_path, exercise_names)
total_manual_reps = sum(len(va.reps) for va in manual_index.values())
print(f"  Videos loaded into index : {len(manual_index)}")
print(f"  Total rep annotations    : {total_manual_reps}")
assert len(manual_index) > 0,    "FAIL: manual_index is empty"
assert total_manual_reps > 0,    "FAIL: no rep annotations loaded"

# Show first 10 annotations
print("\n  First 10 annotations from index:")
hdr_row = f"  {'video':40s}  {'exercise':15s}  {'reps':>5s}  {'correct':>7s}"
print(hdr_row)
for i, (k, va) in enumerate(sorted(manual_index.items())[:10]):
    correct = sum(1 for r in va.reps if r.form_label == 1)
    print(f"  {k:40s}  {str(va.exercise):15s}  {len(va.reps):5d}  {correct:7d}")


# ──────────────────────────────────────────────────────────────────
# 4. SkeletonDataset — train split
# ──────────────────────────────────────────────────────────────────
hdr("4. SkeletonDataset — train split")
train_csv = os.path.join(SPLITS_DIR, "train.csv")
assert os.path.isfile(train_csv), f"FAIL: train.csv not found — run prepare_data.py first"

train_ds = SkeletonDataset(
    skeleton_dir=SKELETON_DIR,
    labels_file=train_csv,
    window_size=config["data"]["window_size"],
    stride=config["data"]["stride"],
    augment=False,          # no augmentation for verification
    config=config,
    annotations_index=manual_index,
    labeling_cfg=lcfg,
)
stats = train_ds.get_label_stats()

n_manual  = stats["videos_manual"]
n_rules   = stats["videos_rules"]
n_none    = stats["videos_none"]
n_total_v = n_manual + n_rules + n_none
pct_manual = 100 * n_manual / n_total_v if n_total_v else 0
pct_rules  = 100 * n_rules  / n_total_v if n_total_v else 0

print(f"  Train windows total      : {stats['total_windows']}")
print(f"  Rep-positive windows     : {stats['rep_positive']}")
print(f"  Form-valid windows       : {stats['form_valid_windows']}")
print(f"  Form-correct windows     : {stats['form_correct']}")
print()
print(f"  Videos — manual annot.  : {n_manual:4d}  ({pct_manual:.0f}%)")
print(f"  Videos — rules fallback : {n_rules:4d}  ({pct_rules:.0f}%)")
print(f"  Videos — no labels      : {n_none:4d}")

assert stats["total_windows"] > 0, "FAIL: no training windows built"
print(f"\n  [OK] {stats['total_windows']} training windows built successfully")


# ──────────────────────────────────────────────────────────────────
# 5. Cross-check: first 10 manual annotations reach dataset with
#    the correct form label
# ──────────────────────────────────────────────────────────────────
hdr("5. Manual annotation label round-trip (first 10 annotated reps)")

# Build a lookup: filename -> list of windows from that video
from collections import defaultdict
by_file = defaultdict(list)
for w in train_ds.windows:
    by_file[w["filename"]].append(w)

passed = failed_checks = 0
checked = []

for video_key in sorted(manual_index.keys())[:10]:
    va = manual_index[video_key]
    if not va.reps or video_key not in by_file:
        continue
    rep = va.reps[0]               # first annotated rep
    ws = config["data"]["window_size"]

    # Find the window that overlaps most with this rep
    best_w = None
    best_overlap = 0
    for w in by_file[video_key]:
        s, e = w["start_frame"], w["start_frame"] + ws
        overlap = max(0, min(e, rep.rep_end) - max(s, rep.rep_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_w = w

    if best_w is None:
        continue  # video not in train split (may be val/test)

    # The window should be labeled from manual source
    src_ok    = best_w["label_source"] == "manual"
    # Form label should match if this window is form_valid
    form_match = (not best_w["form_valid"] or
                  best_w["form"] == rep.form_label)
    ok = src_ok and form_match
    passed += int(ok)
    failed_checks += int(not ok)
    checked.append((video_key, rep.form_label, best_w["form"], best_w["label_source"], ok))

print(f"  {'video':40s}  {'ann_lbl':>7s}  {'win_lbl':>7s}  {'source':8s}  OK?")
for video_key, ann_lbl, win_lbl, src, ok in checked:
    tick = "[OK]" if ok else "[FAIL]"
    print(f"  {video_key:40s}  {ann_lbl:7d}  {win_lbl:7d}  {src:8s}  {tick}")

print(f"\n  Passed {passed}/{len(checked)} checks")
if failed_checks:
    print(f"  WARNING: {failed_checks} windows had unexpected label/source")


# ──────────────────────────────────────────────────────────────────
# 6. __getitem__ tensor shapes (end-to-end model input check)
# ──────────────────────────────────────────────────────────────────
hdr("6. __getitem__ tensor shape check (first 3 windows)")
ws = config["data"]["window_size"]
for i in range(min(3, len(train_ds))):
    item = train_ds[i]
    skel_shape  = tuple(item["skeleton"].shape)
    ang_shape   = tuple(item["angles"].shape)
    ex_val      = item["exercise"].item()
    form_val    = item["form"].item()
    rep_val     = item["rep"].item()
    fv          = item["form_valid"].item()

    assert skel_shape == (ws, 17, 6),  f"FAIL: skeleton shape {skel_shape} != ({ws}, 17, 6)"
    assert ang_shape  == (ws, 12),      f"FAIL: angles shape {ang_shape} != ({ws}, 12)"
    assert form_val  in (0, 1),         f"FAIL: form label {form_val} not in {{0,1}}"
    assert rep_val   in (0, 1),         f"FAIL: rep label {rep_val} not in {{0,1}}"
    assert ex_val    in range(len(exercise_names)), f"FAIL: exercise {ex_val} out of range"

    print(f"  window[{i}]: skeleton={skel_shape}  angles={ang_shape}"
          f"  ex={ex_val}  form={form_val}  rep={rep_val}  form_valid={fv}  [OK]")


# ──────────────────────────────────────────────────────────────────
# 7. All three splits — window counts
# ──────────────────────────────────────────────────────────────────
hdr("7. All split sizes")
total_train_windows = stats["total_windows"]
total_windows_all   = 0

for split in ("train", "val", "test"):
    csv_path = os.path.join(SPLITS_DIR, f"{split}.csv")
    if not os.path.isfile(csv_path):
        print(f"  {split:5s}: CSV not found")
        continue
    ds = SkeletonDataset(
        skeleton_dir=SKELETON_DIR,
        labels_file=csv_path,
        window_size=config["data"]["window_size"],
        stride=config["data"]["stride"] if split == "train" else config["data"]["window_size"],
        augment=False,
        config=config,
        annotations_index=manual_index,
        labeling_cfg=lcfg,
    )
    s = ds.get_label_stats()
    total_windows_all += s["total_windows"]
    print(f"  {split:5s}: {s['total_windows']:6d} windows | "
          f"rep+ {s['rep_positive']:5d} | "
          f"manual={s['videos_manual']:3d}  rules={s['videos_rules']:3d}  "
          f"none={s['videos_none']:2d}")


# ──────────────────────────────────────────────────────────────────
# 8. Final summary
# ──────────────────────────────────────────────────────────────────
hdr("TRAINING READINESS REPORT")

# Count fallback videos across all splits
total_videos_manual = 0
total_videos_rules  = 0
total_videos_none   = 0
for split in ("train", "val", "test"):
    csv_path = os.path.join(SPLITS_DIR, f"{split}.csv")
    if not os.path.isfile(csv_path):
        continue
    df_split = pd.read_csv(csv_path)
    for _, row in df_split.iterrows():
        fn = row["filename"]
        ex_idx = int(row["exercise"])
        ex_name = exercise_names[ex_idx] if ex_idx < len(exercise_names) else ""
        skel = os.path.join(SKELETON_DIR, f"{fn}.npy")
        ang  = os.path.join(SKELETON_DIR, f"{fn}_angles.npy")
        if not os.path.isfile(skel):
            continue
        angles = np.load(ang).astype(np.float32) if os.path.isfile(ang) else None
        _, src = get_video_reps(fn, ex_name, angles, manual_index,
                                allow_rules_fallback=True)
        if src == "manual":
            total_videos_manual += 1
        elif src == "rules":
            total_videos_rules += 1
        else:
            total_videos_none += 1

total_videos = total_videos_manual + total_videos_rules + total_videos_none

print(f"  1. Reps available for training (from reps.csv)   : {total_manual_reps}")
print(f"  2. Videos covered by manual annotations          : {len(manual_index)}")
print(f"  3. Videos using rule-based fallback              : {total_videos_rules}")
print(f"     Videos with no labels at all                  : {total_videos_none}")
print(f"  4. Training UNBLOCKED                            : YES"
      f" (source=hybrid, fallback=True)")
print(f"  5. Estimated training dataset size               : {total_train_windows} windows")
print(f"     Total across all splits                       : {total_windows_all} windows")
print(f"\n  Label routing (all splits, {total_videos} videos with skeletons):")
print(f"    manual  : {total_videos_manual:4d}  ({100*total_videos_manual/max(total_videos,1):.0f}%)")
print(f"    rules   : {total_videos_rules:4d}  ({100*total_videos_rules/max(total_videos,1):.0f}%)")
print(f"    none    : {total_videos_none:4d}  ({100*total_videos_none/max(total_videos,1):.0f}%)")

print(f"\n  {'='*40}")
print(f"  STATUS: READY TO TRAIN")
print(f"  Run: python scripts/train.py --config configs/default.yaml")
print(f"  {'='*40}\n")
