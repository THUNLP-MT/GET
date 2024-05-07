#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from .prediction_model import PredictionModel, ReturnValue


class AffinityPredictor(PredictionModel):

    def __init__(self, model_type, hidden_size, n_channel, **kwargs) -> None:
        super().__init__(model_type, hidden_size, n_channel, **kwargs)
        # self.affinity_ffn = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 1, bias=False)
        # )
    
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=False) -> ReturnValue:
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise)
        energy = return_value.energy
        return F.mse_loss(-energy, label)  # since we are supervising pK=-log_10(Kd), whereas the energy is RTln(Kd)
        aff = scatter_sum(self.affinity_ffn(return_value.block_repr).squeeze(-1), return_value.batch_id)
        return F.mse_loss(aff, label)
    
    def infer(self, batch):
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=None
        )
        return -return_value.energy
        aff = scatter_sum(self.affinity_ffn(return_value.block_repr).squeeze(-1), return_value.batch_id)
        return aff
    

class NoisedAffinityPredictor(AffinityPredictor):
    def __init__(self, model_type, hidden_size, n_channel, sigma: float=0.1, **kwargs) -> None:
        super().__init__(model_type, hidden_size, n_channel, **kwargs)
        self.sigma = sigma

    def add_noise(self, X):
        return X + torch.randn_like(X) * self.sigma

    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=False) -> ReturnValue:
        Z = self.add_noise(Z)
        return super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise)

    def infer(self, batch):
        batch['X'] = self.add_noise(batch['X'])
        return super().infer(batch)