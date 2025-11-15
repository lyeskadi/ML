#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 20:45:50 2024

@author: lyes
"""
###############################################
### Define neural network for digit recognition

import numpy as np
import matplotlib.pyplot as plt # For plotting

def sigmoid(vector):
    vector = np.clip(vector, -500, 500)
    result = 1/(1+np.exp(-vector))
    return result

# Fixed parameters:
    # Input image 28x28 pixels
    # Output 1x10 vector, one value per digit
    # 2 hidden layers with 16 neurons each

# Arbitrary parameters
weightrange = 5 # Range of values allowed for weights and biases

# Input image and output guesses
image = np.random.rand(28,28)
output = np.zeros(10)

# Plot image
plt.imshow(image,cmap='gray')

# Define hidden layers
# Neuron = activation, bias (both doubles)
layer1 = np.zeros(16)
bias1 = np.random.uniform(-weightrange,weightrange,16)

layer2 = np.zeros(16)
bias2 = np.random.uniform(-weightrange,weightrange,16)

# Define weights between layers
weights1 = np.random.uniform(-weightrange,weightrange,(16,28*28))
weights2 = np.random.uniform(-weightrange,weightrange,(16,16))
weights3 = np.random.uniform(-weightrange,weightrange,(10,16))

### Calculate activation of hidden layer neurons
# Reshape image into a vector for convenience
imagevect = image.reshape(-1)
layer1 = sigmoid(weights1 @ imagevect + bias1)
layer2 = sigmoid(weights2 @ layer1 + bias2)
output = sigmoid(weights3 @ layer2)

# Plot output
plt.imshow(output.reshape(1,-1),cmap='gray')
print("\nGuess",np.argmax(output))

##########################################################################
# Plot both image and output, for completeness

fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [28, 1]})
axes[0].imshow(image, cmap='gray', aspect='equal')
axes[0].set_title('Input')
axes[0].axis('off') 

axes[1].imshow(output.reshape(-1, 1), cmap='gray', aspect='auto')
axes[1].set_title('Output')
axes[1].axis('off') 

plt.tight_layout()
plt.show()