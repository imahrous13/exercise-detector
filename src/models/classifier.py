import torch
import torch.nn as nn
from .stgcn import STGCNBlock
from .graph_conv import build_adjacency_matrix


class ExerciseSTGCN(nn.Module):
    """Exercise detection model: ST-GCN backbone + angle branch + BiLSTM + multi-task heads.

    Pipeline:
        Input skeleton (B, T, V, C) ->
        ST-GCN spatial-temporal feature extraction ->
        Pool over joints ->
        [Optional] Angle branch processes (B, T, 12) joint angles ->
        Concatenate ST-GCN features + angle features ->
        BiLSTM temporal modeling ->
        Exercise classification head + Form quality head

    Input:  (B, 30, 17, 6) skeleton + (B, 30, 12) angles
    Output: exercise_logits (B, num_exercises), form_logits (B, num_form_classes),
            rep_logits (B, num_rep_classes)
    """

    def __init__(self, num_exercises=4, num_form_classes=2, num_rep_classes=2,
                 in_channels=6, gcn_channels=None, tcn_kernel_size=9,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3,
                 num_angles=12, angle_hidden=64):
        super().__init__()

        if gcn_channels is None:
            gcn_channels = [64, 128, 256]

        # Build adjacency matrix (fixed, not learnable)
        adj = build_adjacency_matrix()
        self.register_buffer('adj', adj)

        # ST-GCN backbone
        self.stgcn_layers = nn.ModuleList()
        prev_ch = in_channels
        for ch in gcn_channels:
            self.stgcn_layers.append(
                STGCNBlock(prev_ch, ch, kernel_size=tcn_kernel_size, dropout=dropout)
            )
            prev_ch = ch

        # Angle branch: processes (B, T, 12) angles into (B, T, angle_hidden)
        self.angle_branch = nn.Sequential(
            nn.Linear(num_angles, angle_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(angle_hidden, angle_hidden),
            nn.ReLU(inplace=True),
        )
        self.num_angles = num_angles
        self.angle_hidden = angle_hidden

        # BiLSTM input = ST-GCN output channels + angle branch output
        lstm_input_size = gcn_channels[-1] + angle_hidden

        # BiLSTM for temporal sequence modeling
        self.bilstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        lstm_output_size = lstm_hidden * 2  # bidirectional

        # Exercise classification head
        self.exercise_head = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_exercises),
        )

        # Form quality head (binary: correct/incorrect)
        self.form_head = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_form_classes),
        )

        # Rep head (binary: no rep completed in window / rep completed in window)
        self.rep_head = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_rep_classes),
        )

    def forward(self, x, angles=None):
        """
        Args:
            x: (B, T, V, C) skeleton tensor — e.g. (B, 30, 17, 6)
            angles: (B, T, 12) joint angle tensor (optional, zeros if not provided)

        Returns:
            exercise_logits: (B, num_exercises)
            form_logits: (B, num_form_classes)
            rep_logits: (B, num_rep_classes)
        """
        B, T, V, C = x.shape

        # Permute to channels-first: (B, C, T, V)
        x = x.permute(0, 3, 1, 2).contiguous()

        # ST-GCN feature extraction
        for layer in self.stgcn_layers:
            x = layer(x, self.adj)
        # x: (B, C_last, T, V) e.g. (B, 512, 30, 17)

        # Pool over joints (mean over V dimension)
        x = x.mean(dim=3)  # (B, C_last, T)

        # Permute for concatenation: (B, T, C_last)
        x = x.permute(0, 2, 1).contiguous()

        # Angle branch
        if angles is None:
            angles = torch.zeros(B, T, self.num_angles, device=x.device)
        angle_features = self.angle_branch(angles)  # (B, T, angle_hidden)

        # Concatenate ST-GCN features + angle features
        x = torch.cat([x, angle_features], dim=2)  # (B, T, C_last + angle_hidden)

        # BiLSTM temporal modeling
        lstm_out, _ = self.bilstm(x)  # (B, T, lstm_hidden*2)

        # Mean pool over time for final representation
        features = lstm_out.mean(dim=1)  # (B, lstm_hidden*2)

        # Multi-task predictions
        exercise_logits = self.exercise_head(features)
        form_logits = self.form_head(features)
        rep_logits = self.rep_head(features)

        return exercise_logits, form_logits, rep_logits


def build_model(config):
    """Build ExerciseSTGCN model from config dict.

    Args:
        config: dict with 'model' key; optional 'data.exercises' to derive num_exercises

    Returns:
        ExerciseSTGCN model
    """
    model_cfg = config['model']
    num_exercises = model_cfg.get('num_exercises')
    if num_exercises is None and isinstance(config.get('data'), dict):
        exercises = config['data'].get('exercises')
        if exercises:
            num_exercises = len(exercises)
    if num_exercises is None:
        num_exercises = 24  # fallback
    return ExerciseSTGCN(
        num_exercises=num_exercises,
        num_form_classes=model_cfg['num_form_classes'],
        num_rep_classes=model_cfg.get('num_rep_classes', 2),
        in_channels=model_cfg.get('in_channels', 6),
        gcn_channels=model_cfg['gcn_channels'],
        tcn_kernel_size=model_cfg['tcn_kernel_size'],
        lstm_hidden=model_cfg['lstm_hidden'],
        lstm_layers=model_cfg['lstm_layers'],
        dropout=model_cfg['dropout'],
        num_angles=model_cfg.get('num_angles', 12),
        angle_hidden=model_cfg.get('angle_hidden', 64),
    )
