#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 15:49:58 2025

@author: lyes
"""

# 10 epochs, default settings (learning_rate = 1e-3, batch_size = 64)
# With SGD
# 1e-3: 90.9% 56.7sec
# 1e-1: 98.9% (99.0% in some epochs)
# 6e-2: 98.8% 55.7sec
# 3e-2: 98.7%

# With Adam
# 1e-1 -> error
# 1e-3 -> 98.7%
# 5e-4 -> 99.0% 
# 3e-4 -> 98.9%
# 1e-4 -> 98.7%

# kernel_size
# 3: 98.8% 55sec (Default)
# 2: 98.5% 55sec
# Error for 1 and 4

# Adding 3rd convolution layer 
# with optimized Adam (lr=5e-4): 99.1%, 68.5sec
# with optimized SGD (lr=1e-1): 99.1%, 64.9sec
# SGD without batchnorm (above are with): 98.7%

# Adding scheduler, 3layers, SGD, 1e-1: 99.4%
# Scheduler adapts learning rate, starting from provided value
# then decreasing as epochs progress and model is getting closer to optimum

# Scheduler and 64*128*128 neurons: 99.5%
# Current max with CNN: 99.5% with 3 convolutional layers

# Train network to recognise digits from MNIST

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

import time
start_time=time.time()

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

####### Define hyperparameters

learning_rate = 1e-1
batch_size = 64
epochs = 10
conv_layer1 = 64 # Default: 32,64(,64)
conv_layer2 = 128
conv_layer3 = 128
kernel_size = 3

class CNN(nn.Module):
    def __init__(self,conv_layer1, conv_layer2, conv_layer3, kernelsize):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, conv_layer1, kernel_size=kernelsize, padding=1),
            nn.BatchNorm2d(conv_layer1), # Add for increased accuracy, not really seen
            nn.ReLU(),
            #nn.MaxPool2d(2),

            nn.Conv2d(conv_layer1, conv_layer2, kernel_size=kernelsize, padding=1),
            nn.BatchNorm2d(conv_layer2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(conv_layer2, conv_layer3, kernel_size=kernelsize, padding=1),
            nn.BatchNorm2d(conv_layer3),
            nn.ReLU(),
            nn.MaxPool2d(2) # Needs change to 7*7 below
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_layer3 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.classifier(x)
        return x
    
model = CNN(conv_layer1=conv_layer1,conv_layer2=conv_layer2, conv_layer3=conv_layer3, kernelsize=kernel_size).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # SGD vs Adam
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    steps_per_epoch=len(train_dataloader),
    epochs=epochs,
)

####### Define training and testing loops
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):        
        X, y = X.to(device), y.to(device)       
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step() # if using scheduler
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
torch.save(model.state_dict(), 'model_weights_digits_CNN.pth')