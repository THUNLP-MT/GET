#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from e3nn import o3

from .model import ScaleShiftMACE
from .modules.blocks import RealAgnosticResidualInteractionBlock, RealAgnosticInteractionBlock


class MACEEncoder(nn.Module):
    def __init__(self, hidden_size, n_rbf, cutoff, n_layers=2) -> None:
        super().__init__()

        self.encoder = ScaleShiftMACE(
            atomic_inter_scale=1.0,  # from https://github.com/ACEsuit/mace/blob/main/mace/cli/run_train.py#L250
            atomic_inter_shift=0.0,  # from https://github.com/ACEsuit/mace/blob/main/mace/cli/run_train.py#L265
            r_max=cutoff,  # RBF cutoff
            num_bessel=n_rbf,
            num_polynomial_cutoff=5, # from https://github.com/ACEsuit/mace/blob/main/mace/tools/arg_parser.py#L102
            max_ell=3,               # from https://github.com/ACEsuit/mace/blob/main/mace/tools/arg_parser.py#L130
            interaction_cls=RealAgnosticResidualInteractionBlock,  # from https://github.com/ACEsuit/mace/blob/main/mace/tools/arg_parser.py#L108
            interaction_cls_first=RealAgnosticInteractionBlock, # from https://github.com/ACEsuit/mace/blob/main/mace/cli/run_train.py#L260
            num_interactions=n_layers,  # default 2, from https://github.com/ACEsuit/mace/blob/main/mace/tools/arg_parser.py#L135
            num_elements=hidden_size,   # pass in embedded vector
            hidden_irreps=o3.Irreps(f'{hidden_size}x0e + {hidden_size}x1o'),
            MLP_irreps=o3.Irreps('16x0e'),
            atomic_energies=None,
            avg_num_neighbors=1.0,  # from https://github.com/ACEsuit/mace/blob/main/mace/tools/arg_parser.py#L183
            atomic_numbers=[1,1],
            correlation=3,          # from https://github.com/ACEsuit/mace/blob/main/mace/tools/arg_parser.py#L132
            gate=F.silu,            # from https://github.com/ACEsuit/mace/blob/main/mace/tools/arg_parser.py#L169
        )


    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        H, Z = scatter_mean(H, block_id, dim=0), scatter_mean(Z, block_id, dim=0)
        Z = Z.squeeze()
        self_loop = edges[0] == edges[1]
        edges = edges.T[~self_loop].T
        data = {
            'positions': Z,
            'node_attrs': H,
            'num_graphs': batch_id.max() + 1,
            'edge_index': edges,
            'batch': batch_id,
            'cell': None,
            'shifts': torch.zeros((edges.shape[1], 3), dtype=torch.float, device=edges.device)
        }
        out = self.encoder(data)
        block_repr = out['node_feats']
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)
        return H, block_repr, graph_repr, None
