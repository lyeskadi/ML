#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 512 13:42:44 2025

@author: lyes
"""
# 2 layers, 10 epochs
# Neurons in each hidden layer
# 16384 each: 90.7%, 1412sec
# 8192 each: 89.7%
# 4096 each: 88.5%
# 2048 each: 87.4%
# 1024 each: 85.8%
# 512 each:  84.9% (Default)
# 256 each: 83.2%
# 128 each: 81.2%
# 64 each: 79.5%
# 32 each: 77.1%, 50sec
# 16 each: 62.8%

# batch_size at hidden_neurons = 512
# 1024: 84.3 51sec
# 512: 84.8% 50.7
# 256: 83.8% 50.55sec
# 128: 84.9% 51sec (Optimal?)
# 64: 84.4% 50sec (Default)
# 32: 84.5% 50sec
# 16: 84.6% 51sec
# 8: 84.2% 49.7sec
# -> Accuracy is almost independent of batch_size

# learning_rate 
# 1e-0: 96.1% 50.6sec
# 3e-1: 98.2% 49.7sec (no longer monotonically increasing after each epoch)
# 1e-1: 97.8% 49.8sec (reached 97.7% at epoch 6)
# 3e-2: 97.2% 50.20sec 
# 1e-2: 94.0% 49.89sec
# 3e-3: 90.7% 49.82sec
# 1e-3: 84.7% 50.5sec (Default)
# 3e-4: 71.9% 50.0sec
# 1e-4: 41.8% 49.9sec
# High impact on accuracy, while not affecting training time
# Optimal: 1e-1 to avoid instability?

# Neurons, learning rate
# 8192, 1e-1: 97.9% 335sec -> At optimal learning rate, number of neurons has little effect
# 64, 1e-1: 97.3% 50sec

# Max achieved for MLP: ~98% accuracy with optimal params

# Train network to recognise digits from MNIST

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

import time
start_time = time.time()

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
####### Define hyperparameters

learning_rate = 1e-1 # Step size for optimizer (too high -> unstable, too low -> slow)
batch_size = 64 # Update weights after this number of training samples (RAM-limited?)
epochs = 10 # Number of times model is retrained with entire dataset   
hidden_neurons = 64 # todo: separate for layer 1 and 2
model = NeuralNetwork(hidden_size=hidden_neurons).to(device)

####### Define training and testing loops
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Move data to the same device as model
        X, y = X.to(device), y.to(device)
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

accuracies = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    acc = test_loop(test_dataloader, model, loss_fn)
    accuracies.append(acc)
print("Done!")

end_time = time.time()

print(f"Runtime: {end_time-start_time}")

# Plot accuracy over training time
epochs_range = range(1, epochs + 1)

plt.plot(epochs_range, accuracies, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracy vs Epoch")
plt.grid(True)
plt.show()

# Plot an image from training data
# Get one sample
img, label = training_data[0]  # first image

# Plot it
plt.imshow(img.squeeze(), cmap="gray")  # remove channel dimension
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()

######## Save model
torch.save(model.state_dict(), 'model_weights_digits_256neurons.pth')