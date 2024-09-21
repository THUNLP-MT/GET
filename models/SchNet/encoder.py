#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from .schnet import SchNet

class SchNetEncoder(nn.Module):
    def __init__(self, hidden_size, edge_size, n_layers=3) -> None:
        super().__init__()

        self.num_gaussians = 50

        self.encoder = SchNet(hidden_size, num_interactions=n_layers, num_gaussians=self.num_gaussians)

        if edge_size != 0:
            self.edge_linear = nn.Linear(edge_size, self.num_gaussians)

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H, Z = scatter_mean(H, block_id, dim=0), scatter_mean(Z, block_id, dim=0)
        Z = Z.squeeze()
        if edge_attr is not None:
            edge_attr = self.edge_linear(edge_attr)
        block_repr = self.encoder(H, Z, batch_id, edges, edge_attr)  # [Nb, hidden]
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)
        return H, block_repr, graph_repr, None