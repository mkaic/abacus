# %%
import torch
from ..src.sparse_abacus import SparseAbacusLayer
import itertools


shapes = [(5, 6, 7), (6, 7), (6,)]
for input_shape, output_shape in itertools.product(shapes, shapes):
    print(input_shape, output_shape)
    layer = SparseAbacusLayer(
        input_shape=input_shape, output_shape=output_shape, degree=2
    )
    input = torch.rand(4, *input_shape)

    output = layer(input)
    print(output.shape)
