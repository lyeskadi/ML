#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 21:27:43 2025

@author: lyes
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
# Create instance of model
model = NeuralNetwork().to(device)
print(model)

# Run a random image through it
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Print weights
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

#####

##### Detailed breakdown of how image goes through model
##### self.__init__
input_image = torch.rand(3,28,28)
print(input_image.size())

# Flatten to simplify dims
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# Apply linear transformation
# z = w*a + b?
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# Apply ReLU activations
# sigma(z)
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

######

###### Access weights and biases
print(f"\n-------------\nModel structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# Can copy it into standalone tensor with detach.clone
print("\n-------------\n Weights and biases in a layer")
weights = model.linear_relu_stack[4].weight.detach().clone()
print( weights)
biases = model.linear_relu_stack[4].bias.detach().clone()
print("\nBiases\n",biases)