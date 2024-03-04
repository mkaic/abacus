import torch
import matplotlib.pyplot as plt

from ..src.samplers import n_nonisotropic_gaussian_pdf

coordinates = torch.stack(
    torch.meshgrid(
        *[torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32)], indexing="ij"
    ),
    dim=-1,
).expand(4, -1, -1, -1)

output_coordinates = coordinates.view(4, -1, 2)

to_plot = [coordinates[0].sum(dim=-1), output_coordinates[0].sum(dim=-1).view(32, 32)]

pdf = n_nonisotropic_gaussian_pdf(
    coordinates,
    mu=torch.rand(4, 2),
    sigma=torch.rand(4, 2) * 0.3,
)

print(pdf.shape)

to_plot.extend([pdf[0, :, :, 0], pdf[1, :, :, 1]])

print(pdf.max(), pdf.min())

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for i, data in enumerate(to_plot):
    ax = axes[i // 2, i % 2]
    plt.colorbar(ax.imshow(data), ax=ax)

plt.savefig("abacus/tests/images/gaussian_sanity_check.png", facecolor="black")
