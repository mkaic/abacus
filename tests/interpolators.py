import torch
from ..src.interpolators import LinearInterpolator, SciPyLinearInterpolator, FourierInterpolator
import matplotlib.pyplot as plt
import itertools

BATCH_SIZE = 8

input_shapes = [(4,), (4, 4), (4, 4, 4), (4, 4, 4, 4)]
output_shapes = [(16,), (16, 16), (16, 16, 16), (16, 16, 16, 16)]


for input_shape, output_shape in itertools.product(input_shapes, output_shapes):
    print(input_shape, output_shape)

    linear_interpolator = LinearInterpolator(input_shape, output_shape)
    scipy_interpolator = SciPyLinearInterpolator(input_shape, output_shape)
    fourier_interpolator = FourierInterpolator(input_shape, output_shape)

    input_values = torch.rand(BATCH_SIZE, *input_shape)
    output_points = torch.rand(BATCH_SIZE, *output_shape, len(input_shape))

    linear_output = linear_interpolator(input_values, output_points)
    linear_reference = scipy_interpolator(input_values, output_points)
    fourier_output = fourier_interpolator(input_values, output_points)

    assert torch.allclose(
        linear_output, linear_reference, atol=1e-6
    ), f"Interpolation failed. Error: {torch.abs(linear_output - linear_reference).max()}"


# Now we will visually inspect the 2D to 2D case
input_shape = (4, 4)
output_shape = (64, 64)

linear_interpolator = LinearInterpolator(input_shape, output_shape)
scipy_interpolator = SciPyLinearInterpolator(input_shape, output_shape)
fourier_interpolator = FourierInterpolator(input_shape, output_shape)

input_values = torch.rand(BATCH_SIZE, *input_shape)
output_points = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing="ij")
output_points = torch.stack(output_points, dim=-1).unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1)

linear_output = linear_interpolator(input_values, output_points)
linear_reference = scipy_interpolator(input_values, output_points)
fourier_output = fourier_interpolator(input_values, output_points)

fig, axes = plt.subplots(2, 2, figsize=(10, 3))
axes[0, 0].imshow(input_values[0])
axes[0, 1].imshow(linear_output[0])
axes[1, 0].imshow(linear_reference[0])
axes[1, 1].imshow(fourier_output[0])


plt.savefig("abacus/tests/interpn_test.png")