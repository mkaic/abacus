# %%
import torch
from ..src.sparse_abacus import SparseAbacusLayer

layer = SparseAbacusLayer(input_dims=64, output_dims=32)

input = torch.rand(4, 64)


output = layer(input)

print(output.shape)

loss = output.mean()

loss.backward()

print(layer.sample_points.grad)
