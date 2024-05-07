#!/usr/bin/python
# -*- coding:utf-8 -*-
from get_clean import GET, fully_connect_edges, knn_edges
import torch

# use GPU 0
device = torch.device('cuda:0')

# Dummy model parameters
d_hidden = 64   # hidden size
d_radial = 16   # mapped size of key/value in attention (greatly influence space complexity)
n_channel = 1   # number of channels for coordinates, usually 1 as one atom only has one coordinate
d_edge = 16     # edge feature size
n_rbf = 16      # RBF kernal size
n_head= 4       # number of heads for multi-head attention

# Dummy variables h, x and fully connected edges
# 19 atoms, divided into 8 blocks
block_ids = torch.tensor([0,0,1,1,1,1,2,2,2,3,4,4,5,6,6,6,6,7,7], dtype=torch.long).to(device)
# 8 blocks, divided into 2 graphs
batch_ids = torch.tensor([0,0,0,0,0,1,1,1], dtype=torch.long).to(device)
n_atoms, n_blocks = block_ids.shape[0], batch_ids.shape[0]
H = torch.randn(n_atoms, d_hidden, device=device)
X = torch.randn(n_atoms, n_channel, 3, device=device)
# fully connect edges
src_dst = fully_connect_edges(batch_ids)
# if you want to test knn_edges, you can try:
# src_dst = knn_edges(block_ids, batch_ids, X, k_neighbors=5)
edge_attr = torch.randn(len(src_dst[0]), d_edge).to(device)

# Initialize GET
model = GET(d_hidden, d_radial, n_channel, n_rbf, d_edge=d_edge, n_head=n_head)
model.to(device)
model.eval()

# Run GET
H, X = model(H, X, block_ids, batch_ids, src_dst, edge_attr)

print('Done!')