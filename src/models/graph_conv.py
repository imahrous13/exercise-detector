import torch
import torch.nn as nn
import numpy as np


# COCO 17-keypoint skeleton edges
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6),                                      # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),            # arms
    (5, 11), (6, 12), (11, 12),                  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),      # legs
]

NUM_JOINTS = 17


def build_adjacency_matrix(edges=None, num_joints=NUM_JOINTS, self_loops=True, normalize=True):
    """Build symmetric adjacency matrix from skeleton edge list.

    Args:
        edges: list of (src, dst) tuples. Defaults to COCO_EDGES.
        num_joints: number of joints
        self_loops: whether to add self-connections
        normalize: apply symmetric normalization D^{-1/2} A D^{-1/2}

    Returns:
        (num_joints, num_joints) float tensor
    """
    if edges is None:
        edges = COCO_EDGES

    adj = np.zeros((num_joints, num_joints), dtype=np.float32)
    for src, dst in edges:
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0

    if self_loops:
        adj += np.eye(num_joints, dtype=np.float32)

    if normalize:
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        degree = adj.sum(axis=1)
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        adj = D_inv_sqrt @ adj @ D_inv_sqrt

    return torch.FloatTensor(adj)


class GraphConvolution(nn.Module):
    """Spatial graph convolution layer.

    Performs: X' = A_norm @ X @ W + bias

    Input:  (B, T, V, C_in)  — batch, time, vertices/joints, channels
    Output: (B, T, V, C_out)
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Args:
            x: (B, T, V, C_in) input features
            adj: (V, V) normalized adjacency matrix

        Returns:
            (B, T, V, C_out) output features
        """
        # Graph convolution: A @ X @ W
        # x: (B, T, V, C_in)
        # adj: (V, V)
        support = torch.matmul(x, self.weight)        # (B, T, V, C_out)
        output = torch.einsum('btvc,vw->btwc', support, adj)  # (B, T, V_out=W, C_out)
        # Actually we want adj @ support along the V dimension:
        # output[b,t,v,c] = sum_w adj[v,w] * support[b,t,w,c]
        output = torch.einsum('vw,btwc->btvc', adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output
