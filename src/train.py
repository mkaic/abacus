import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from .model import SparseAbacusModel
from .interpolators import LinearInterpolator, FourierInterpolator
from .aggregators import LinearCombination, FuzzyNAND
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

DATA_SHAPES = [
    (3, 32, 32),
    (8, 8, 8),
    (8, 8, 8),
    (8, 8, 8),
    (8, 8, 8),
    128,
    100,
]
LR = 1e-4
DEGREE = 2
INTERPOLATOR = FourierInterpolator
AGGREGATOR = FuzzyNAND

print(f"{DATA_SHAPES=}, {DEGREE=}, {BATCH_SIZE=}, {LR=}, {INTERPOLATOR=}, {AGGREGATOR=}")

model = SparseAbacusModel(
    data_shapes=DATA_SHAPES,
    degree=DEGREE,
    interpolator_class=INTERPOLATOR,
    aggregator_class=AGGREGATOR,
)
model = model.to(DEVICE)

# Load the MNIST dataset
train = CIFAR100(root="./abacus/data", train=True, download=True, transform=ToTensor())
test = CIFAR100(root="./abacus/data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# Train the model
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss().to(DEVICE)

test_accuracy = 0
for epoch in range(5):
    model.train()
    pbar = tqdm(train_loader, leave=False)
    for x, y in pbar:
        optimizer.zero_grad()

        x, y = x.to(DEVICE), y.to(DEVICE)

        y_hat = model(x)

        loss = criterion(y_hat, y)
        loss.backward()

        # print(model.layers[0].sample_points.grad[0])

        optimizer.step()

        model.clamp_params()

        pbar.set_description(
            f"Epoch {epoch}. Train: {loss.item():.4f}, Test: {test_accuracy:.2%}"
        )

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)

            y_hat = model(x)

            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    test_accuracy = correct / total
    print(f"Epoch {epoch + 1}: {test_accuracy:.2%} accuracy on test set")
