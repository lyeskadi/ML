#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:24:07 2024

@author: lyes
"""

### Plot images from MNIST
import numpy as np
import matplotlib.pyplot as plt

# Load data 
data = np.load('/home/lyes/Documents/ML/MNIST/mnist_data.npz')
train_images = data['train_images']
train_labels = data['train_labels']
test_images = data['test_images']
test_labels = data['test_labels']

select_ind = 16756
plt.imshow(train_images[select_ind],cmap='gray')
plt.axis('off')

print("Digit ", train_labels[select_ind], "    Index", select_ind)
print("Max value", np.max(train_images[select_ind]))