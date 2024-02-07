# %%
import torch
from ..src.interpolator import interp_nd
import itertools


shapes = [((5, 5, 5), (10, 10, 10))]  # , (6, 7), (6,)]
for input_shape, output_shape in shapes:
    print(input_shape, output_shape)

    input = torch.rand(16, *input_shape)
    output_points = torch.rand(16, 64, len(output_shape))
    output = interp_nd(input, output_points)
