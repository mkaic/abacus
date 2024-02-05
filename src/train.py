import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from .model import SparseAbacusModel
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import torch.nn as nn
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SparseAbacusModel(activation_dims=[784, 1024, 1024, 10])
model = model.to(DEVICE)

# Load the MNIST dataset
train = FashionMNIST(root="./abacus/data", train=True, download=True, transform=ToTensor())
test = FashionMNIST(root="./abacus/data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader = DataLoader(test, batch_size=128, shuffle=False)

# Train the model
optimizer = SGD(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss().to(DEVICE)

test_accuracy = 0
for epoch in range(10):
    model.train()
    pbar = tqdm(train_loader, leave=False)
    for x, y in pbar:
        optimizer.zero_grad()

        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = model(x.view(-1, 784))
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        model.clamp_params()

        pbar.set_description(f"Epoch {epoch}. Train: {loss.item():.4f}, Test: {test_accuracy:.2%}")

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat = model(x.view(-1, 784))
            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    test_accuracy = correct / total
    print(f"Epoch {epoch + 1}: {test_accuracy:.2%} accuracy on test set")
