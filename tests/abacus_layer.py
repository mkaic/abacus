# %%
import torch
from ..src.sparse_abacus import SparseAbacusLayer

layer = SparseAbacusLayer(n_in=64, n_out=32)

input = torch.rand(4, 64)


output = layer(input)

print(output.shape)

loss = output.mean()

loss.backward()

print(layer.sample_points.grad)
