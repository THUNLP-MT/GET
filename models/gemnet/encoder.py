#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from utils.nn_utils import graph_to_batch

from .gemnet import GemNetT


GemnetData = namedtuple('GemnetData', ['pos', 'batch', 'h', 'edge_index', 'natoms'])


class GemNetEncoder(nn.Module):
    def __init__(self, hidden_size, radial_size, edge_size, n_layers=3, k_neighbors=9) -> None:
        super().__init__()

        self.encoder = GemNetT(
            num_atoms=0,  # unused
            bond_feat_dim=0,  # unused
            num_targets=hidden_size,  # output size
            num_spherical=7,
            num_radial=radial_size,
            num_blocks=n_layers,
            emb_size_atom=hidden_size,
            emb_size_edge=edge_size,
            emb_size_trip=64,
            emb_size_rbf=64,
            emb_size_cbf=16,
            emb_size_bil_trip=64,
            num_before_skip=1,
            num_after_skip=2,
            num_concat=1,
            num_atom=3,
            regress_forces=False,
            max_neighbors=k_neighbors,
            use_pbc=False,
            otf_graph=True
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H, Z = scatter_mean(H, block_id, dim=0), scatter_mean(Z, block_id, dim=0)
        Z = Z.squeeze()
        data = GemnetData(
            pos=Z,
            batch=batch_id,
            h=H,
            edge_index=edges,
            natoms=scatter_sum(torch.ones_like(batch_id), batch_id, dim=0)
        )
        block_repr = self.encoder(data)
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)
        return H, block_repr, graph_repr, None
