import torch
from ..src.interpolators import LinearInterpolator, SciPyLinearInterpolator
import itertools
import numpy as np

BATCH_SIZE = 8

input_shapes = [(4,), (4, 4), (4, 4, 4), (4, 4, 4, 4)]
output_shapes = [(16,), (16, 16), (16, 16, 16), (16, 16, 16, 16)]


for input_shape, output_shape in itertools.product(input_shapes, output_shapes):
    print(input_shape, output_shape)

    interpolator = LinearInterpolator(input_shape, output_shape)
    scipy_interpolator = SciPyLinearInterpolator(input_shape, output_shape)

    input_values = torch.rand(BATCH_SIZE, *input_shape)
    output_points = torch.rand(BATCH_SIZE, *output_shape, len(input_shape))

    output = interpolator(input_values, output_points)
    reference = scipy_interpolator(input_values, output_points)

    if len(output_shape) == 2 and len(input_shape) == 2:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        axes[0].imshow(input_values[0])
        axes[1].imshow(output[0])
        axes[2].imshow(reference[0])

        plt.savefig("abacus/tests/interpn_test.png")

    assert torch.allclose(
        output, reference, atol=1e-6
    ), f"Interpolation failed. Error: {torch.abs(output - reference).max()}"
