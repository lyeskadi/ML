#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 03:52:29 2024

@author: lyes
"""

# MNIST dataset for digit recognition

# =============================================================================
# import torchvision.datasets as datasets
# mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
# mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
# =============================================================================

import torchvision.datasets as datasets
import numpy as np
from PIL import Image  # For converting images to NumPy arrays

# Download MNIST dataset
mnist_trainset = datasets.MNIST(root='/home/lyes/Documents/ML/MNIST', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='/home/lyes/Documents/ML/MNIST', train=False, download=True, transform=None)

# Convert training data to NumPy arrays
train_images = []
train_labels = []

for img, label in mnist_trainset:
    # Convert the image (PIL Image) to a NumPy array
    np_img = np.array(img, dtype=np.float32)  # Convert to float32 for numerical operations
    train_images.append(np_img)
    train_labels.append(label)

# Convert lists to NumPy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Convert test data to NumPy arrays
test_images = []
test_labels = []

for img, label in mnist_testset:
    np_img = np.array(img, dtype=np.float32)
    test_images.append(np_img)
    test_labels.append(label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Print the shapes to verify
print(f"Train images shape: {train_images.shape}")  # (60000, 28, 28)
print(f"Train labels shape: {train_labels.shape}")  # (60000,)
print(f"Test images shape: {test_images.shape}")    # (10000, 28, 28)
print(f"Test labels shape: {test_labels.shape}")    # (10000,)

# Save data
np.savez('/home/lyes/Documents/ML/MNIST/mnist_data.npz', 
         train_images=train_images, 
         train_labels=train_labels, 
         test_images=test_images, 
         test_labels=test_labels)