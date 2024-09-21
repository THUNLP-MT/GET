#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from utils.nn_utils import graph_to_batch

from .equiformer import Equiformer


class EquiformerEncoder(nn.Module):
    def __init__(self, hidden_size, edge_size, heads=4, n_layers=3) -> None:
        super().__init__()

        dim_feats = (hidden_size, hidden_size // 2, hidden_size // 4)
        dim_head = tuple([f // heads for f in dim_feats])

        self.encoder = Equiformer(
            dim=dim_feats,
            num_degrees=3,
            heads=heads,
            dim_head=dim_head,
            depth=n_layers,
            reversible=True,
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H, Z = scatter_mean(H, block_id, dim=0), scatter_mean(Z, block_id, dim=0)
        Z = Z.squeeze()
        H, mask = graph_to_batch(H, batch_id, mask_is_pad=False)
        Z, _ = graph_to_batch(Z, batch_id, mask_is_pad=False)
        out = self.encoder(H, Z, mask)
        block_repr = out.type0[mask] + out.type1.sum() * 0 # cheat the autograd check
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)
        return H, block_repr, graph_repr, None
