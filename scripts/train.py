import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from ..src.model import SparseAbacusModel
from ..src.interpolators import LinearInterpolator, FourierInterpolator
from ..src.aggregators import LinearCombination, FuzzyNAND
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

import warnings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
LR = 1e-3
DEGREE = 32
INTERPOLATOR = LinearInterpolator
AGGREGATOR = LinearCombination
DATA_DEPENDENT = [False for _ in range(9)]
# DATA_DEPENDENT[-1] = True
# DATA_DEPENDENT[-2] = True

COMPILE = True

INPUT_SHAPES = [(3, 32, 32)]
MID_BLOCK_SHAPES = [(256,)] #[(8,8) for _ in range(1)]
OUTPUT_SHAPES = [(100,)]

LOOKBEHIND = 1
EPOCHS = 100

SAVE = False

print(
    f"{INPUT_SHAPES=}, {MID_BLOCK_SHAPES=}, {OUTPUT_SHAPES=}, {DEGREE=}, {BATCH_SIZE=}, {LR=}, {INTERPOLATOR=}, {AGGREGATOR=}, {LOOKBEHIND=}, {EPOCHS=}, {DATA_DEPENDENT=}"
)

if not Path("abacus/weights").exists():
    Path("abacus/weights").mkdir(parents=True)

model = SparseAbacusModel(
    input_shapes=INPUT_SHAPES,
    mid_block_shapes=MID_BLOCK_SHAPES,
    output_shapes=OUTPUT_SHAPES,
    degree=DEGREE,
    interpolator_class=INTERPOLATOR,
    aggregator_class=AGGREGATOR,
    lookbehind=LOOKBEHIND,
    data_dependent=DATA_DEPENDENT,
)
model = model.to(DEVICE)

print(model.layers)
print(model.lookbehinds_list)
print(model.data_dependent)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    model = torch.compile(model, disable=not COMPILE)

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

        y_hat = model(x)

        loss = criterion(y_hat, y)
        loss.backward()

        optimizer.step()

        model.clamp_params() 
            
        pbar.set_description(
            f"Epoch {epoch}. Train: {loss.item():.4f}, Test: {test_accuracy:.2%}"
        )

    if SAVE:
        torch.save(model.state_dict(), f"abacus/weights/{epoch:03d}.ckpt")
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)

            y_hat = model(x)

            _, predicted = torch.max(y_hat, dim=1)
            total += y.shape[0]
            correct += (predicted == y).sum().item()

    test_accuracy = correct / total
    print(f"Epoch {epoch + 1}: {test_accuracy:.2%} accuracy on test set")
