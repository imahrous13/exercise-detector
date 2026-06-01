import torch
import torch.nn as nn
from .graph_conv import GraphConvolution


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution Block.

    Performs:
        1. Spatial graph convolution across joints (within each frame)
        2. Temporal convolution across time (within each joint)
        3. Residual connection + BatchNorm + Dropout

    Input:  (B, C_in, T, V)  — channels-first
    Output: (B, C_out, T, V)
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Spatial graph convolution
        self.gcn = GraphConvolution(in_channels, out_channels)
        self.bn_gcn = nn.BatchNorm2d(out_channels)

        # Temporal convolution
        padding = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj):
        """
        Args:
            x: (B, C_in, T, V) input tensor
            adj: (V, V) normalized adjacency matrix

        Returns:
            (B, C_out, T, V)
        """
        residual = self.residual(x)

        # Spatial graph convolution
        # Reshape to (B, T, V, C) for GraphConvolution
        B, C, T, V = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # (B, T, V, C)
        x_gcn = self.gcn(x_perm, adj)                  # (B, T, V, C_out)
        x_gcn = x_gcn.permute(0, 3, 1, 2).contiguous() # (B, C_out, T, V)
        x_gcn = self.bn_gcn(x_gcn)
        x_gcn = self.relu(x_gcn)

        # Temporal convolution
        x_tcn = self.tcn(x_gcn)  # (B, C_out, T, V)

        # Residual + activation
        out = self.relu(x_tcn + residual)
        return out
