# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# %%
x = torch.linspace(0, 1, 32).expand(32, -1)
y = torch.linspace(1, 0, 32).expand(32, -1).T
z = (x + y) / 2

plt.figure()
plt.imshow(z)

z = z.view(1, 1, *z.shape)
print(z.shape)

new_dim = 16
new_x = torch.linspace(0, 1, new_dim)
new_y = torch.linspace(0, 1, new_dim)

new_coords = torch.meshgrid(new_x, new_y, indexing="xy")
new_coords = torch.stack(new_coords, dim=-1)
new_coords = new_coords.view(new_dim, new_dim, 2)

new_coords = (
    new_coords - 0.5
) * 2  # map from [0, 1] to [-1, 1] and flip axes to match with grid_sample

new_coords = torch.rand_like(new_coords) * 2 - 1
plt.figure()
plt.imshow(new_coords.sum(-1))

new_coords = new_coords.view(1, *new_coords.shape)
print(new_coords.shape)

gridsampled = F.grid_sample(
    z, new_coords, mode="bilinear", padding_mode="reflection", align_corners=False
)
print(gridsampled.shape)

plt.figure()
plt.imshow(gridsampled.squeeze())

# %%
