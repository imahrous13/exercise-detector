import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    """Training loop for ExerciseSTGCN with multi-task loss."""

    def __init__(self, model, loss_fn, optimizer, scheduler, device, config,
                 checkpoint_dir='checkpoints', log_dir='logs'):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_exercise = 0
        correct_form = 0
        correct_rep = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            skeleton = batch['skeleton'].to(self.device)      # (B, 30, 17, 6)
            angles = batch['angles'].to(self.device)           # (B, 30, 12)
            exercise_labels = batch['exercise'].to(self.device)  # (B,)
            form_labels = batch['form'].to(self.device)          # (B,)
            rep_labels = batch['rep'].to(self.device)            # (B,)
            form_valid = batch.get('form_valid')
            if form_valid is not None:
                form_valid = form_valid.to(self.device)

            # Forward pass
            exercise_logits, form_logits, rep_logits = self.model(skeleton, angles)

            # Compute loss
            loss, loss_dict = self.loss_fn(
                exercise_logits, form_logits, rep_logits,
                exercise_labels, form_labels, rep_labels,
                form_valid=form_valid,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_clip = self.config['training'].get('gradient_clip', 1.0)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

            self.optimizer.step()

            # Metrics
            total_loss += loss.item() * skeleton.size(0)
            correct_exercise += (exercise_logits.argmax(1) == exercise_labels).sum().item()
            correct_form += (form_logits.argmax(1) == form_labels).sum().item()
            correct_rep += (rep_logits.argmax(1) == rep_labels).sum().item()
            total_samples += skeleton.size(0)

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ex_acc': f"{correct_exercise/total_samples:.3f}",
                'form_acc': f"{correct_form/total_samples:.3f}",
            })

        avg_loss = total_loss / total_samples
        ex_acc = correct_exercise / total_samples
        form_acc = correct_form / total_samples
        rep_acc = correct_rep / total_samples

        # Log to tensorboard
        self.writer.add_scalar('train/loss', avg_loss, epoch)
        self.writer.add_scalar('train/exercise_acc', ex_acc, epoch)
        self.writer.add_scalar('train/form_acc', form_acc, epoch)
        self.writer.add_scalar('train/rep_acc', rep_acc, epoch)

        return avg_loss, ex_acc, form_acc, rep_acc

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """Validate on the validation set."""
        self.model.eval()
        total_loss = 0.0
        correct_exercise = 0
        correct_form = 0
        correct_rep = 0
        total_samples = 0

        for batch in val_loader:
            skeleton = batch['skeleton'].to(self.device)
            angles = batch['angles'].to(self.device)
            exercise_labels = batch['exercise'].to(self.device)
            form_labels = batch['form'].to(self.device)
            rep_labels = batch['rep'].to(self.device)
            form_valid = batch.get('form_valid')
            if form_valid is not None:
                form_valid = form_valid.to(self.device)

            exercise_logits, form_logits, rep_logits = self.model(skeleton, angles)
            loss, _ = self.loss_fn(
                exercise_logits, form_logits, rep_logits,
                exercise_labels, form_labels, rep_labels,
                form_valid=form_valid,
            )

            total_loss += loss.item() * skeleton.size(0)
            correct_exercise += (exercise_logits.argmax(1) == exercise_labels).sum().item()
            correct_form += (form_logits.argmax(1) == form_labels).sum().item()
            correct_rep += (rep_logits.argmax(1) == rep_labels).sum().item()
            total_samples += skeleton.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        ex_acc = correct_exercise / max(total_samples, 1)
        form_acc = correct_form / max(total_samples, 1)
        rep_acc = correct_rep / max(total_samples, 1)

        # Log to tensorboard
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/exercise_acc', ex_acc, epoch)
        self.writer.add_scalar('val/form_acc', form_acc, epoch)
        self.writer.add_scalar('val/rep_acc', rep_acc, epoch)

        return avg_loss, ex_acc, form_acc, rep_acc

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        path = os.path.join(self.checkpoint_dir, 'last.pt')
        torch.save(state, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pt')
            torch.save(state, best_path)

    def train(self, train_loader, val_loader, num_epochs, start_epoch=0):
        """Full training loop with early stopping."""
        patience = self.config['training'].get('early_stopping_patience', 15)

        for epoch in range(start_epoch + 1, num_epochs + 1):
            # Train
            train_loss, train_ex_acc, train_form_acc, train_rep_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_ex_acc, val_form_acc, val_rep_acc = self.validate(val_loader, epoch)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f} | Ex: {train_ex_acc:.3f} | Form: {train_form_acc:.3f} | Rep: {train_rep_acc:.3f}")
            print(f"  Val   - Loss: {val_loss:.4f} | Ex: {val_ex_acc:.3f} | Form: {val_form_acc:.3f} | Rep: {val_rep_acc:.3f}")

            # Early stopping check
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, val_loss, is_best=is_best)

            if self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        self.writer.close()
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")
