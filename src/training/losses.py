import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall et al., CVPR 2018).

    Automatically learns the relative importance of each task via
    learnable log-variance parameters.

    loss = exp(-s1)*L_exercise + s1 + exp(-s2)*L_form + s2 + exp(-s3)*L_rep + s3

    Supports optional class weights and label smoothing for the exercise head.
    """

    def __init__(self, exercise_class_weights=None, label_smoothing=0.0):
        super().__init__()
        # Learnable log-variance for each task (exercise, form, rep)
        self.log_vars = nn.Parameter(torch.zeros(3))

        self.ce_exercise = nn.CrossEntropyLoss(
            weight=exercise_class_weights,
            label_smoothing=label_smoothing,
        )
        self.ce_form = nn.CrossEntropyLoss()
        self.ce_rep = nn.CrossEntropyLoss()

    def forward(self, exercise_logits, form_logits, rep_logits,
               exercise_labels, form_labels, rep_labels, form_valid=None):
        """
        Args:
            exercise_logits: (B, num_exercises) predictions
            form_logits: (B, num_form_classes) predictions
            rep_logits: (B, num_rep_classes) predictions
            exercise_labels: (B,) ground truth exercise classes
            form_labels: (B,) ground truth form quality (0/1)
            rep_labels: (B,) ground truth rep-in-window (0/1)
            form_valid: (B,) optional mask — only windows overlapping a manual rep
                        contribute to form loss (manual annotation pipeline)

        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss values
        """
        loss_exercise = self.ce_exercise(exercise_logits, exercise_labels)
        if form_valid is not None and form_valid.any():
            loss_form = self.ce_form(form_logits[form_valid], form_labels[form_valid])
        elif form_valid is not None and not form_valid.any():
            loss_form = exercise_logits.new_tensor(0.0)
        else:
            loss_form = self.ce_form(form_logits, form_labels)
        loss_rep = self.ce_rep(rep_logits, rep_labels)

        # Uncertainty weighting
        precision_exercise = torch.exp(-self.log_vars[0])
        precision_form = torch.exp(-self.log_vars[1])
        precision_rep = torch.exp(-self.log_vars[2])

        total_loss = (
            precision_exercise * loss_exercise +
            precision_form * loss_form +
            precision_rep * loss_rep +
            self.log_vars.sum()
        )

        loss_dict = {
            'exercise_loss': loss_exercise.item(),
            'form_loss': loss_form.item(),
            'rep_loss': loss_rep.item(),
            'total_loss': total_loss.item(),
            'weight_exercise': precision_exercise.item(),
            'weight_form': precision_form.item(),
            'weight_rep': precision_rep.item(),
        }

        return total_loss, loss_dict
