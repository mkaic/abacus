import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from ..src.model import SamplerModel
from ..src.layers import BinaryTreeSparseAbacusLayer, GaussianLayer
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

import warnings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
DEGREE = 4

LAYER_CLASS = BinaryTreeSparseAbacusLayer

INPUT_SHAPES = [
    tuple([2] * 12)
]  # (3 x 32 x 32) padded to (4 x 32 x 32), represented as a binary tree with depth 12.
MID_BLOCK_SHAPES = [
    *[tuple([2] * 2) for _ in range(2)],
]
OUTPUT_SHAPES = [(100,)]

COMPILE = False
SAVE = True

print(
    f"""
{INPUT_SHAPES=}, 
{MID_BLOCK_SHAPES=}, 
{OUTPUT_SHAPES=}, 
{DEGREE=}, 
{BATCH_SIZE=}, 
{EPOCHS=}
"""
)

if not Path("abacus/weights").exists():
    Path("abacus/weights").mkdir(parents=True)

model = SamplerModel(
    input_shapes=INPUT_SHAPES,
    mid_block_shapes=MID_BLOCK_SHAPES,
    output_shapes=OUTPUT_SHAPES,
    degree=DEGREE,
    layer_class=LAYER_CLASS,
)
model = model.to(DEVICE)

print(model.layers)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    model = torch.compile(model, disable=not COMPILE)
    model: SamplerModel

# Load the MNIST dataset
train = CIFAR100(root="./abacus/data", train=True, download=True, transform=ToTensor())
test = CIFAR100(root="./abacus/data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4
)
test_loader = DataLoader(
    test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4
)

# Train the model
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss().to(DEVICE)

test_accuracy = 0
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, leave=False)

    for x, y in pbar:
        optimizer.zero_grad()

        x, y = x.to(DEVICE), y.to(DEVICE)

        # Need input shape to be a power of 2 for BinaryTreeLinearInterp
        x = torch.cat([x, torch.zeros(x.shape[0], 1, 32, 32, device=DEVICE)], dim=1)
        x = x.reshape(-1, *[2 for _ in range(12)])

        y_hat = model(x)

        loss = criterion(y_hat, y)
        loss: torch.Tensor
        loss.backward()

        optimizer.step()

        model.clamp_params()

        pbar.set_description(
            f"Epoch {epoch}. Train: {loss.item():.4f}, Test: {test_accuracy:.2%}"
        )

    model.eval()
    if SAVE:
        torch.save(model.state_dict(), f"abacus/weights/{epoch:03d}.ckpt")

    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, leave=False):

            x: torch.Tensor
            y: torch.Tensor

            x, y = x.to(DEVICE), y.to(DEVICE)

            x = torch.cat([x, torch.zeros(x.shape[0], 1, 32, 32, device=DEVICE)], dim=1)
            x = x.reshape(-1, *[2 for _ in range(12)])

            y_hat = model(x)

            _, predicted = torch.max(y_hat, dim=1)
            total += y.shape[0]
            correct += (predicted == y).sum().item()

    test_accuracy = correct / total
    print(f"Epoch {epoch + 1}: {test_accuracy:.2%} accuracy on test set")
