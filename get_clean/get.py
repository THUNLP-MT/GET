#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_mean, scatter_sum, scatter_std

from .tools import _unit_edges_from_block_edges
from .radial_basis import RadialBasis


class GET(nn.Module):
    '''Generalist Equivariant Transformer'''

    def __init__(self, d_hidden, d_radial, n_channel, n_rbf, cutoff=7.0, d_edge=0, n_layers=4, n_head=4,
                 act_fn=nn.SiLU(), residual=True, dropout=0.1, z_requires_grad=True, pre_norm=False,
                 sparse_k=3):
        super().__init__()
        '''
        :param d_hidden: Number of hidden features
        :param d_radial: Number of features for calculating geometric relations
        :param n_channel: Number of channels of coordinates of each unit
        :param n_rbf: Dimension of RBF feature, 1 for not using rbf
        :param cutoff: cutoff for RBF
        :param d_edge: Number of features for the edge features
        :param n_layers: Number of layer
        :param n_head: Number of heads for multi-head attention
        :param act_fn: Non-linearity
        :param residual: Use residual connections, we recommend not changing this one
        :param dropout: Probability of dropout
        :z_requires_grad: Whether the final coordinates requires gradients
        :pre_norm: If True, normalize the coordinates first
        :sparse_k: Number of sparse connection between two blocks
        '''

        self.n_layers = n_layers
        self.pre_norm = pre_norm
        self.sparse_k = sparse_k

        if self.pre_norm:
            self.pre_layernorm = EquivariantLayerNorm(d_hidden, n_channel, n_rbf, cutoff, act_fn)

        for i in range(0, n_layers):
            self.add_module(f'layer_{i}', GETLayer(
                d_hidden, d_radial, n_channel, n_rbf, cutoff, d_edge, n_head, act_fn, residual
            ))
            self.add_module(f'layernorm0_{i}', EquivariantLayerNorm(d_hidden, n_channel, n_rbf, cutoff, act_fn))
            self.add_module(f'ffn_{i}', EquivariantFFN(
                d_hidden, 4 * d_hidden, d_hidden, n_channel,
                n_rbf, act_fn, residual, dropout,
                z_requires_grad=z_requires_grad if i == n_layers - 1 else True
            ))
            self.add_module(f'layernorm1_{i}', EquivariantLayerNorm(d_hidden, n_channel, n_rbf, cutoff, act_fn))

        if not z_requires_grad:
            self._modules[f'layernorm1_{n_layers - 1}'].sigma.requires_grad = False

    def recover_scale(self, Z, block_id, batch_id, record_scale):
        with torch.no_grad():
            unit_batch_id = batch_id[block_id]
        Z_c = scatter_mean(Z, unit_batch_id, dim=0)  # [bs, n_channel, 3]
        Z_c = Z_c[unit_batch_id]  # [N, n_channel, 3]
        Z_centered = Z - Z_c

        Z = Z_c + Z_centered / record_scale[unit_batch_id]

        return Z

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, cached_unit_edge_info=None):
        if cached_unit_edge_info is None:
            with torch.no_grad():
                cached_unit_edge_info = _unit_edges_from_block_edges(block_id, edges.T, Z, k=self.sparse_k) # [Eu], Eu = \sum_{i, j \in E} n_i * n_j
        
        batch_size, n_channel = batch_id.max() + 1, Z.shape[1]
        record_scale = torch.ones((batch_size, n_channel, 1), dtype=torch.float, device=Z.device)

        if self.pre_norm:
            H, Z, rescale = self.pre_layernorm(H, Z, block_id, batch_id)
            record_scale *= rescale

        for i in range(self.n_layers):

            H, Z = self._modules[f'layer_{i}'](
                H, Z, block_id, edges, edge_attr, cached_unit_edge_info
            )
            H, Z, rescale = self._modules[f'layernorm0_{i}'](H, Z, block_id, batch_id)
            record_scale *= rescale
            H, Z = self._modules[f'ffn_{i}'](H, Z, block_id)
            H, Z, rescale = self._modules[f'layernorm1_{i}'](H, Z, block_id, batch_id)
            record_scale *= rescale
        
        Z = self.recover_scale(Z, block_id, batch_id, record_scale)

        return H, Z


'''
Below are the implementation of the equivariant adaptive block message passing mechanism
'''

class GETLayer(nn.Module):
    '''
    Generalist Equivariant Transformer layer
    '''

    def __init__(self, d_hidden, d_radial, n_channel, n_rbf, cutoff=7.0,
                 d_edge=0, n_head=4, act_fn=nn.SiLU(), residual=True):
        super(GETLayer, self).__init__()

        self.residual = residual
        self.reci_sqrt_d = 1 / math.sqrt(d_radial)
        self.epsilon = 1e-8

        self.n_rbf = n_rbf
        self.cutoff = cutoff

        self.n_head = n_head
        assert d_radial % self.n_head == 0, f'd_radial not compatible with n_head ({d_radial} and {self.n_head})'
        assert n_rbf % self.n_head == 0, f'n_rbf not compatible with n_head ({n_rbf} and {self.n_head})'

        d_hidden_head, d_radial_head = d_hidden // self.n_head, d_radial // self.n_head
        n_rbf_head = n_rbf // self.n_head

        self.linear_qk = nn.Linear(d_hidden_head, d_radial_head * 2, bias=False)
        self.linear_v = nn.Linear(d_hidden_head, d_radial_head)

        if n_rbf > 1:
            self.rbf = RadialBasis(num_radial=n_rbf, cutoff=cutoff)

        self.att_mlp = nn.Sequential(
            nn.Linear(d_radial_head * 3 + n_channel * n_rbf_head, d_radial_head), # radial*3 means H_q, H_k and edge_attr
            act_fn,
            nn.Linear(d_radial_head, d_radial_head),
            act_fn
        )
        self.unit_att_linear = nn.Linear(d_radial_head, 1)
        self.block_att_linear = nn.Linear(d_radial_head, 1)

        if d_edge != 0:
            self.edge_linear = nn.Linear(d_edge, d_radial)

        self.node_mlp = nn.Sequential(
            nn.Linear(d_radial, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden),
            act_fn
        )

        self.node_out_linear = nn.Linear(d_hidden, d_hidden)

        self.coord_mlp = nn.Sequential(
            nn.Linear(d_radial, d_hidden),
            act_fn,
            nn.Linear(d_hidden, n_head * n_channel),
            act_fn
        )

        self.unit_msg_mlp = nn.Sequential(
            nn.Linear(d_radial_head + n_channel * n_rbf_head, d_radial_head),
            act_fn,
            nn.Linear(d_radial_head, d_radial_head),
            act_fn
        )

        self.unit_msg_coord_mlp = nn.Sequential(
            nn.Linear(d_radial_head + n_channel * n_rbf_head, d_radial_head),
            act_fn,
            nn.Linear(d_radial_head, d_radial_head),
            act_fn
        )

        self.unit_msg_coord_linear = nn.Linear(d_radial_head, n_channel)

    def attention(self, H, Z, edges, edge_attr, cached_unit_edge_info):
        row, col = edges
        (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = cached_unit_edge_info

        # multi-head
        H = H.view(H.shape[0], self.n_head, -1) # [N, n_head, hidden_size / n_head]

        # calculate attention
        H_qk = self.linear_qk(H)
        H_q, H_k = H_qk[..., 0::2][unit_row], H_qk[..., 1::2][unit_col]  # [Eu, n_head, d_radial / n_head]

        dZ = Z[unit_row] - Z[unit_col]  # [E_u, n_channel, 3]

        D = torch.norm(dZ, dim=-1)  # [Eu, n_channel]
        if self.n_rbf > 1:
            n_channel = D.shape[-1]
            D = self.rbf(D.view(-1)).view(D.shape[0], n_channel, self.n_head, -1)  # [Eu, n_channel, n_head, n_rbf / n_head]
            D = D.transpose(1, 2).reshape(D.shape[0], self.n_head, -1)  # [Eu, n_head, n_channel * n_rbf / n_head]
        else:
            D = D.unsqueeze(1).repeat(1, self.n_head, 1)  # [Eu, n_head, n_channel]

        if edge_attr is None:
            R_repr = torch.concat([H_q, H_k, D], dim=-1) # [Eu, n_head, (d_radial * 2 + n_channel * n_rbf) / n_head]
        else:
            edge_attr = self.edge_linear(edge_attr).view(edge_attr.shape[0], self.n_head, -1)
            R_repr = torch.concat([H_q, H_k, D, edge_attr[block_edge_id]], dim=-1)
        R_repr = self.att_mlp(R_repr) # [Eu, n_head, d_radial / n_head]
        R = self.unit_att_linear(R_repr).squeeze(-1) # [Eu, n_head]

        alpha = scatter_softmax(R, unit_edge_src_id, dim=0).unsqueeze(-1) # [Eu, n_head, 1], unit-level attention within block-level edges

        # directly use mean of R is not reasonble as the value before softmax has different scales in different pairs
        # using max(R) - min(R) or max(R) - mean(R) are also not reasonable as the lowerbound will be 0 instead of -inf
        # so we use pooling on the representation of unit attention
        beta = self.block_att_linear(scatter_mean(R_repr, block_edge_id, dim=0)).squeeze(-1) # [Eb, n_head]
        beta = scatter_softmax(beta, row, dim=0)  # [Eb, n_head], block-level edge attention
        beta = beta[block_edge_id[unit_edge_src_start]].unsqueeze(-1)  # [Em, n_head, 1], Em = \sum_{i, j \in E} n_i

        return alpha, beta, (D, R, dZ)

    def invariant_update(self, H_v, H, alpha, beta, D, cached_unit_edge_info):
        (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = cached_unit_edge_info
        unit_agg_row = unit_row[unit_edge_src_start]

        # update invariant feature
        H_v = self.unit_msg_mlp(torch.cat([H_v[unit_col], D], dim=-1))  # [Eu, n_head, d_radial / n_head]

        H_agg = scatter_sum(alpha * H_v, unit_edge_src_id, dim=0)  # [Em, n_head, hidden_size / n_head]
        H_agg = H_agg.view(H_agg.shape[0], -1)  # [Em, hidden_size]
        H_agg = self.node_mlp(H_agg)  # [Em, hidden_size]
        H_agg = H_agg.view(H_agg.shape[0], self.n_head, -1) # [Em, n_head, hidden_size / n_head]
        H_agg = scatter_sum(beta * H_agg, unit_agg_row, dim=0, dim_size=H.shape[0])  # [N, n_head, hidden_size / n_head]
        H_agg = H_agg.view(H_agg.shape[0], -1)  # [N, hidden_size]
        H_agg = self.node_out_linear(H_agg)
        
        H = H + H_agg if self.residual else H_agg

        return H
    
    def equivariant_update(self, H_v, Z, alpha, beta, D, dZ, cached_unit_edge_info):
        (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = cached_unit_edge_info
        unit_agg_row = unit_row[unit_edge_src_start]

        # update equivariant feature
        H_v = self.unit_msg_coord_mlp(torch.cat([H_v[unit_col], D], dim=-1))  # [Eu, n_head, d_radial / n_head]

        Z_agg = scatter_sum(
            (alpha * self.unit_msg_coord_linear(H_v)).unsqueeze(-1) * dZ.unsqueeze(1),
            unit_edge_src_id, dim=0)  # [Em, n_head, n_channel, 3]
        Z_H_agg = scatter_sum(alpha * H_v, unit_edge_src_id, dim=0) # [Em, n_head, d_radial / n_head]
        Z_H_agg = self.coord_mlp(Z_H_agg.view(Z_H_agg.shape[0], -1))  # [Em, d_radial]
        Z_H_agg = Z_H_agg.view(Z_H_agg.shape[0], self.n_head, -1) # [Em, n_head, n_channel]
        Z_agg = scatter_sum(
            (beta * Z_H_agg).unsqueeze(-1) * Z_agg, unit_agg_row,
            dim=0, dim_size=Z.shape[0])  # [N, n_head, n_channel, 3]
        Z_agg = Z_agg.sum(dim=1) # [N, n_channel, 3]

        Z = Z + Z_agg

        return Z

    def forward(self, H, Z, block_id, edges, edge_attr=None, cached_unit_edge_info=None):
        '''
        H: [N, hidden_size],
        Z: [N, n_channel, 3],
        block_id: [N],
        edges: [2, E], list of [n_row] and [n_col] where n_row == n_col == E, nodes from col are used to update nodes from row
        edge_attr: [E]
        cached_unit_edge_info: unit level (row, col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) calculated from block edges
        '''
        with torch.no_grad():
            if cached_unit_edge_info is None:
                cached_unit_edge_info = _unit_edges_from_block_edges(block_id, edges.T) # [Eu], Eu = \sum_{i, j \in E} n_i * n_j

        alpha, beta, (D, R, dZ) = self.attention(H, Z, edges, edge_attr, cached_unit_edge_info)
        
        H_v = self.linear_v(H.view(H.shape[0], self.n_head, -1))  # [N, n_head, d_radial / n_head]

        H = self.invariant_update(H_v, H, alpha, beta, D, cached_unit_edge_info)

        Z = self.equivariant_update(H_v, Z, alpha, beta, D, dZ, cached_unit_edge_info)

        return H, Z


class EquivariantFFN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_channel, n_rbf=16, act_fn=nn.SiLU(),
                 residual=True, dropout=0.1, constant=1, z_requires_grad=True) -> None:
        super().__init__()
        self.constant = constant
        self.residual = residual
        self.n_rbf = n_rbf

        self.mlp_h = nn.Sequential(
            nn.Linear(d_in * 2 + n_channel * n_rbf, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
            nn.Dropout(dropout)
        )

        self.mlp_z = nn.Sequential(
            nn.Linear(d_in * 2 + n_channel * n_rbf, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_channel),
            nn.Dropout(dropout)
        )

        if not z_requires_grad:
            for param in self.mlp_z.parameters():
                param.requires_grad = False
        
        self.rbf = RadialBasis(n_rbf, 7.0)
    
    def forward(self, H, Z, block_id):
        '''
        :param H: [N, d_in]
        :param Z: [N, n_channel, 3]
        :param block_id: [Nu]
        '''
        radial, (Z_c, Z_o) = self._radial(Z, block_id)  # [N, n_hidden_channel], ([N, 1, 3], [N, n_channel, 3]
        H_c = scatter_mean(H, block_id, dim=0)[block_id]  # [N, d_in]
        inputs = torch.cat([H, H_c, radial], dim=-1)  # [N, d_in + d_in + d_in]

        H_update = self.mlp_h(inputs)
        
        H = H + H_update if self.residual else H_update

        Z = Z_c + self.mlp_z(inputs).unsqueeze(-1) * Z_o

        return H, Z

    def _radial(self, Z, block_id):
        Z_c = scatter_mean(Z, block_id, dim=0)  # [Nb, n_channel, 3]
        Z_c = Z_c[block_id]
        Z_o = Z - Z_c  # [N, n_channel, 3], no translation

        D = torch.norm(Z_o, dim=-1)  # [N, n_channel]
        radial = self.rbf(D.view(-1)).view(D.shape[0], -1) # [N, n_channel * n_rbf]

        return radial, (Z_c, Z_o)


class EquivariantLayerNorm(nn.Module):
    
    def __init__(self, d_hidden, n_channel, n_rbf=16, cutoff=7.0, act_fn=nn.SiLU()) -> None:
        super().__init__()

        # invariant
        self.fuse_scale_ffn = nn.Sequential(
            nn.Linear(n_channel * n_rbf, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden),
            act_fn
        )
        self.layernorm = nn.LayerNorm(d_hidden)

        # geometric
        sigma = torch.ones((1, n_channel, 1))
        self.sigma = nn.Parameter(sigma, requires_grad=True)
        self.rbf = RadialBasis(num_radial=n_rbf, cutoff=cutoff)

    def forward(self, H, Z, block_id, batch_id):
        with torch.no_grad():
            _, n_channel, n_axis = Z.shape
            unit_batch_id = batch_id[block_id]
            unit_axis_batch_id = unit_batch_id.unsqueeze(-1).repeat(1, n_axis).flatten()  # [N * 3]
        # H = self.layernorm(H)
        Z_c = scatter_mean(Z, unit_batch_id, dim=0)  # [bs, n_channel, 3]
        Z_c = Z_c[unit_batch_id]  # [N, n_channel, 3]
        Z_centered = Z - Z_c
        var = scatter_std(
            Z_centered.transpose(1, 2).reshape(-1, n_channel).contiguous(),
            unit_axis_batch_id, dim=0)  # [bs, n_channel]
        # var = var[unit_batch_id].unsqueeze(-1)  # [N, n_channel, 1]
        # Z = Z_c + Z_centered / var * self.sigma
        rescale = (1 / var).unsqueeze(-1) * self.sigma  # [bs, n_channel, 1]
        Z = Z_c + Z_centered * rescale[unit_batch_id]

        rescale_rbf = self.rbf(rescale.view(-1)).view(rescale.shape[0], -1) # [bs, n_channel * n_rbf]
        H = H + self.fuse_scale_ffn(rescale_rbf)[unit_batch_id]
        H = self.layernorm(H)
        return H, Z, rescale



if __name__ == '__main__':
    d_hidden = 64
    d_radial = 16
    n_channel = 2
    d_edge = 16
    n_rbf = 16
    n_head= 4
    device = torch.device('cuda:0')
    model = GET(d_hidden, d_radial, n_channel, n_rbf, d_edge=d_edge, n_head=n_head)
    model.to(device)
    model.eval()
    
    block_id = torch.tensor([0,0,1,1,1,1,2,2,2,3,4,4,5,6,6,6,6,7,7], dtype=torch.long).to(device)
    batch_id = torch.tensor([0,0,0,0,0,1,1,1], dtype=torch.long).to(device)
    src_dst = torch.tensor([[0,1], [2,3], [1,3], [2,4], [3, 0], [3, 3], [5,7], [7,6], [5,6], [6,7]], dtype=torch.long).to(device)
    src_dst = src_dst.T
    edge_attr = torch.randn(len(src_dst[0]), d_edge).to(device)
    n_unit = block_id.shape[0]

    H = torch.randn(n_unit, d_hidden, device=device)
    Z = torch.randn(n_unit, n_channel, 3, device=device)

    H1, Z1 = model(H, Z, block_id, batch_id, src_dst, edge_attr)

    # random rotaion matrix
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q1, t1 = U.mm(V), torch.randn(3, device=device)
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q2, t2 = U.mm(V), torch.randn(3, device=device)

    unit_batch_id = batch_id[block_id]
    Z[unit_batch_id == 0] = torch.matmul(Z[unit_batch_id == 0], Q1) + t1
    Z[unit_batch_id == 1] = torch.matmul(Z[unit_batch_id == 1], Q2) + t2
    # Z = torch.matmul(Z, Q) + t

    H2, Z2 = model(H, Z, block_id, batch_id, src_dst, edge_attr)

    print(f'invariant feature: {torch.abs(H1 - H2).sum()}')
    Z1[unit_batch_id == 0] = torch.matmul(Z1[unit_batch_id == 0], Q1) + t1
    Z1[unit_batch_id == 1] = torch.matmul(Z1[unit_batch_id == 1], Q2) + t2
    print(f'equivariant feature: {torch.abs(Z1 - Z2).sum()}')