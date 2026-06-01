"""Evaluate the trained ExerciseSTGCN model on a test set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.classifier import build_model
from src.data.dataset import create_dataloaders
from src.feedback.form_rules import get_exercise_names


def main():
    parser = argparse.ArgumentParser(description='Evaluate ExerciseSTGCN model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--skeleton_dir', type=str, default='data/processed/skeletons')
    parser.add_argument('--splits_dir', type=str, default='data/splits')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model and load checkpoint
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint['model_state_dict']
    use_rep_head = any(k.startswith('rep_head') for k in state)
    model.load_state_dict(state, strict=use_rep_head)
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"Checkpoint val loss: {checkpoint.get('val_loss', '?')}" + (" (with rep head)" if use_rep_head else ""))

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Create test dataloader (manual annotation labels + extra features)
    loaders = create_dataloaders(
        skeleton_dir=args.skeleton_dir,
        splits_dir=args.splits_dir,
        window_size=config['data']['window_size'],
        stride=config['data']['window_size'],  # No overlap for evaluation
        batch_size=config['training']['batch_size'],
        config=config,
        project_root=project_root,
    )

    if 'test' not in loaders:
        print("No test split found. Using val split.")
        if 'val' not in loaders:
            print("ERROR: No test or val data found.")
            return
        test_loader = loaders['val']
    else:
        test_loader = loaders['test']

    print(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate
    all_exercise_preds = []
    all_exercise_labels = []
    all_form_preds = []
    all_form_labels = []
    all_rep_preds = []
    all_rep_labels = []

    with torch.no_grad():
        for batch in test_loader:
            skeleton = batch['skeleton'].to(device)    # (B, 30, 17, 6)
            angles = batch['angles'].to(device)         # (B, 30, 12)
            exercise_labels = batch['exercise']
            form_labels = batch['form']
            rep_labels = batch['rep']

            exercise_logits, form_logits, rep_logits = model(skeleton, angles)

            all_exercise_preds.extend(exercise_logits.argmax(1).cpu().numpy())
            all_exercise_labels.extend(exercise_labels.numpy())
            all_form_preds.extend(form_logits.argmax(1).cpu().numpy())
            all_form_labels.extend(form_labels.numpy())
            all_rep_preds.extend(rep_logits.argmax(1).cpu().numpy())
            all_rep_labels.extend(rep_labels.numpy())

    all_exercise_preds = np.array(all_exercise_preds)
    all_exercise_labels = np.array(all_exercise_labels)
    all_form_preds = np.array(all_form_preds)
    all_form_labels = np.array(all_form_labels)
    all_rep_preds = np.array(all_rep_preds)
    all_rep_labels = np.array(all_rep_labels)

    # Exercise classification report
    print("\n" + "=" * 60)
    print("EXERCISE CLASSIFICATION RESULTS")
    print("=" * 60)
    ex_names = get_exercise_names(config)
    num_ex = len(ex_names)
    ex_labels_range = list(range(num_ex))
    print(classification_report(all_exercise_labels, all_exercise_preds,
                                labels=ex_labels_range, target_names=ex_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_exercise_labels, all_exercise_preds, labels=ex_labels_range))

    # Form quality report
    print("\n" + "=" * 60)
    print("FORM QUALITY RESULTS")
    print("=" * 60)
    form_labels_range = [0, 1]
    print(classification_report(all_form_labels, all_form_preds,
                                labels=form_labels_range,
                                target_names=['incorrect', 'correct'], digits=4,
                                zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_form_labels, all_form_preds, labels=form_labels_range))

    # Rep head results (when checkpoint has rep head)
    if use_rep_head:
        print("\n" + "=" * 60)
        print("REP (rep-in-window) RESULTS")
        print("=" * 60)
        rep_labels_range = [0, 1]
        print(classification_report(all_rep_labels, all_rep_preds,
                                    labels=rep_labels_range,
                                    target_names=['no_rep', 'rep'], digits=4,
                                    zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(all_rep_labels, all_rep_preds, labels=rep_labels_range))

    # Overall accuracy
    ex_acc = (all_exercise_preds == all_exercise_labels).mean()
    form_acc = (all_form_preds == all_form_labels).mean()
    print(f"\nExercise Accuracy: {ex_acc:.4f}")
    print(f"Form Accuracy:     {form_acc:.4f}")
    if use_rep_head:
        rep_acc = (all_rep_preds == all_rep_labels).mean()
        print(f"Rep Accuracy:      {rep_acc:.4f}")


if __name__ == '__main__':
    main()
