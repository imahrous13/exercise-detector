"""Train the ExerciseSTGCN model.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.classifier import build_model
from src.training.losses import MultiTaskLoss
from src.training.trainer import Trainer
from src.data.dataset import create_dataloaders


def main():
    parser = argparse.ArgumentParser(description='Train ExerciseSTGCN model')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--skeleton_dir', type=str, default='data/processed/skeletons')
    parser.add_argument('--splits_dir', type=str, default='data/splits')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
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
    print(f"Using device: {device}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Create dataloaders (manual annotations + augmentation + class-balanced sampling)
    loaders = create_dataloaders(
        skeleton_dir=args.skeleton_dir,
        splits_dir=args.splits_dir,
        window_size=config['data']['window_size'],
        stride=config['data']['stride'],
        batch_size=config['training']['batch_size'],
        config=config,
        project_root=project_root,
    )

    if 'train' not in loaders:
        print("ERROR: No training data found. Check skeleton_dir and splits_dir.")
        print(f"  skeleton_dir: {args.skeleton_dir}")
        print(f"  splits_dir: {args.splits_dir}")
        return

    print(f"Train samples: {len(loaders['train'].dataset)}")
    if 'val' in loaders:
        print(f"Val samples: {len(loaders['val'].dataset)}")

    # Build model
    model = build_model(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Compute class weights for loss function (inverse frequency)
    train_cfg = config['training']
    exercise_class_weights = None
    if train_cfg.get('exercise_class_weights', False):
        class_counts, _ = loaders['train'].dataset.get_class_counts()
        num_classes = config['model']['num_exercises']
        total = sum(class_counts.values())
        weights = torch.zeros(num_classes)
        for cls_idx, count in class_counts.items():
            weights[cls_idx] = total / (num_classes * count)
        exercise_class_weights = weights.to(device)
        print(f"Using exercise class weights (min={weights.min():.2f}, max={weights.max():.2f})")

    # Loss function with class weights + label smoothing
    label_smoothing = train_cfg.get('label_smoothing', 0.0)
    loss_fn = MultiTaskLoss(
        exercise_class_weights=exercise_class_weights,
        label_smoothing=label_smoothing,
    )
    print(f"Label smoothing: {label_smoothing}")

    # Optimizer
    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.Adam(
        all_params,
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg.get('weight_decay', 0.0001),
    )

    # Scheduler: warmup + cosine annealing
    warmup_epochs = train_cfg.get('warmup_epochs', 0)
    total_epochs = train_cfg['epochs']

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=train_cfg['learning_rate'] * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        print(f"Scheduler: {warmup_epochs}-epoch linear warmup + cosine annealing")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=train_cfg['learning_rate'] * 0.01,
        )
        print("Scheduler: cosine annealing")

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        state = checkpoint['model_state_dict']
        model.load_state_dict(state, strict=any(k.startswith('rep_head') for k in state))
        ckpt_loss = checkpoint['loss_fn_state_dict']
        old_optimizer = checkpoint.get('optimizer_state_dict')
        same_param_count = (old_optimizer is not None and
                            len(old_optimizer.get('param_groups', [{}])[0].get('params', [])) == len(optimizer.param_groups[0]['params']))
        if ckpt_loss.get('log_vars', torch.zeros(1)).shape == loss_fn.log_vars.shape:
            loss_fn.load_state_dict(ckpt_loss)
        else:
            # Old 2-task checkpoint: copy first two log_vars, leave third (rep) at 0
            if 'log_vars' in ckpt_loss and ckpt_loss['log_vars'].shape[0] == 2:
                with torch.no_grad():
                    loss_fn.log_vars.data[:2] = ckpt_loss['log_vars'].to(device)
                print("Loaded 2-task loss weights; rep task log_var initialized at 0.")
        if same_param_count and old_optimizer is not None:
            optimizer.load_state_dict(old_optimizer)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("Optimizer/scheduler not loaded (new params); starting fresh.")
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    # Restore best val loss if resuming
    if start_epoch > 0:
        trainer.best_val_loss = best_val_loss

    # Train
    val_loader = loaders.get('val', loaders['train'])
    trainer.train(
        train_loader=loaders['train'],
        val_loader=val_loader,
        num_epochs=train_cfg['epochs'],
        start_epoch=start_epoch,
    )


if __name__ == '__main__':
    main()
