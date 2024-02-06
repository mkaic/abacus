import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from .model import SparseAbacusModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

imsize = 32 * 32 * 3
layer_dims = [imsize // 2, imsize // 4, imsize // 8, imsize // 16]
layer_dims = [s * 8 for s in layer_dims]
model = SparseAbacusModel(input_dim=imsize, layer_dims=layer_dims, output_dim=100)
model = model.to(DEVICE)

# Load the MNIST dataset
train = CIFAR100(root="./abacus/data", train=True, download=True, transform=ToTensor())
test = CIFAR100(root="./abacus/data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

# Train the model
optimizer = Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss().to(DEVICE)

test_accuracy = 0
for epoch in range(100):
    model.train()
    pbar = tqdm(train_loader, leave=False)
    for x, y in pbar:
        optimizer.zero_grad()

        x, y = x.to(DEVICE), y.to(DEVICE)

        B = x.shape[0]
        y_hat = model(x.view(B, -1))

        loss = criterion(y_hat, y)
        loss.backward()

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

            B = x.shape[0]
            y_hat = model(x.view(B, -1))

            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    test_accuracy = correct / total
    print(f"Epoch {epoch + 1}: {test_accuracy:.2%} accuracy on test set")
