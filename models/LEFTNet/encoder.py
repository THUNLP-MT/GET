#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from .model import LEFTNet


LEFTNetData = namedtuple('LEFTNetData', ['z', 'posc', 'batch', 'edge_index'])


class LEFTNetEncoder(nn.Module):
    def __init__(self, hidden_size, n_rbf, cutoff, n_layers=3) -> None:
        super().__init__()

        self.save_state_dict = True

        self.encoder = LEFTNet(
            pos_require_grad=False,
            cutoff=cutoff,
            num_layers=n_layers,
            hidden_channels=hidden_size,
            num_radial=n_rbf
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H, Z = scatter_mean(H, block_id, dim=0), scatter_mean(Z, block_id, dim=0)
        Z = Z.squeeze()
        # delete self-loop
        self_loop = edges[0] == edges[1]
        edges = edges.T[~self_loop].T
        data = LEFTNetData(
            posc=Z,
            batch=batch_id,
            z=H,
            edge_index=edges
        )
        block_repr, _ = self.encoder(data)
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)
        return H, block_repr, graph_repr, None
