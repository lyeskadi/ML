#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 22:15:56 2025

@author: lyes
"""

###############################################
### Define neural network for digit recognition

import numpy as np
import matplotlib.pyplot as plt # For plotting
import time

def sigmoid(vector):
    vector = np.clip(vector, -500, 500)
    result = 1/(1+np.exp(-vector))
    return result

def sigmap(vector):
    #vector = np.clip(vector, -500, 500)
    #result = np.exp(-vector)/((1+np.exp(-vector))**2)
    #return result
    return sigmoid(vector)*(1-sigmoid(vector))

start_time = time.time()

# Fixed parameters:
    # Input image 28x28 pixels
    # Output 1x10 vector, one value per digit
    # 2 hidden layers with 16 neurons each

# Arbitrary parameters
weightrange = 5 # Range of values allowed for weights and biases

### Load MNIST data 
data = np.load('/home/lyes/Documents/ML/MNIST/mnist_data.npz')
train_images = data['train_images']
train_labels = data['train_labels']
test_images = data['test_images']
test_labels = data['test_labels']


# Input image and output guesses
#select_ind = 25938
#image = train_images[select_ind]/255 #np.random.rand(28,28)
#imagevect = image.reshape(-1) # Reshape image into a vector for convenience

#plt.imshow(train_images[select_ind],cmap='gray')
#plt.axis('off')

### Initialize neural network
# Neurons
layer1 = np.zeros(16)
bias1 = np.random.uniform(-weightrange,weightrange,16)
layer2 = np.zeros(16)
bias2 = np.random.uniform(-weightrange,weightrange,16)
bias3 = np.random.uniform(-weightrange,weightrange,10)

# Weights
W1 = np.random.uniform(-weightrange,weightrange,(16,28*28))
W2 = np.random.uniform(-weightrange,weightrange,(16,16))
W3 = np.random.uniform(-weightrange,weightrange,(10,16))

### Backpropagation
gradb1full = np.zeros(16)
gradW1full = np.zeros([16,28*28])
gradb2full = np.zeros(16)
gradW2full = np.zeros([16,16])
gradb3full = np.zeros(10)
gradW3full = np.zeros([10,16])

train_inds = np.arange(0,10)
for i in train_inds:
    
    # Load image
    train_ind = train_inds[i] #25938
    image = train_images[train_ind]/255 #np.random.rand(28,28)
    imagevect = image.reshape(-1) # Reshape image into a vector for convenience
        
    # Calculate activation of hidden layer neurons
    layer1 = sigmoid(W1 @ imagevect + bias1)
    layer2 = sigmoid(W2 @ layer1 + bias2)
    output = sigmoid(W3 @ layer2 + bias3)
    
    #train_ind = 25938
    y = np.zeros(10)
    y[train_labels[train_ind]] = 1
    
    cost = np.sum((output - y)**2)
    sigmap1 = sigmap(W1 @ imagevect + bias1)
    sigmap2 = sigmap(W2 @ layer1 + bias2)
    sigmap3 = sigmap(W3 @ layer2 + bias3)
    
    # Layer 1
    gradb1 = np.zeros(16)
    gradW1 = np.zeros([16,28*28])
    gradW1sum = np.zeros(16)
    for p in range(16):
        for s in range(28*28):
            for q in range(16):
                for r in range(10):
                    gradW1sum[p] += (2*output[r]-y[r])*W3[r,q]*sigmap2[q]*W2[q,p]*sigmap3[r]
            gradW1[p,s] = gradW1sum[p]*imagevect[s]*sigmap1[p]
        gradb1[p] = gradW1sum[p]*sigmap1[p]
    
    # Layer 2
    gradb2 = np.zeros(16)
    gradW2 = np.zeros([16,16])
    gradW2sum = np.zeros(16)
    for p in range(16):
            for q in range(16):
                for r in range(10):
                    gradW2sum[q] += (2*output[r]-y[r])*W3[r,q]*sigmap3[r]
                gradW2[q,p] = gradW2sum[q]*layer1[p]*sigmap2[q]
            gradb2[q] = gradW2sum[q]*sigmap2[q]
    
    # Layer 3
    gradb3 = np.zeros(10)
    gradW3 = np.zeros([10,16])
    for q in range(16):
        for r in range(10):
            gradW3[r,q] = layer2[q]*sigmap3[r]*(2*output[r]-y[r])
        gradb3[r] = sigmap3[r]*(2*output[r]-y[r])
    
    # Add to final gradient
    gradb1full += gradb1/np.size(train_inds)
    gradW1full += gradW1/np.size(train_inds)
    gradb2full += gradb2/np.size(train_inds)
    gradW2full += gradW2/np.size(train_inds)
    gradb3full += gradb3/np.size(train_inds)
    gradW3full += gradW3/np.size(train_inds)

# Modify weights and biases
W1 += gradW1full
W2 += gradW2full
W3 += gradW3full
bias1 += gradb1full
bias2 += gradb2full
bias3 += gradb3full

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:",elapsed_time, "sec for", np.size(train_inds), "training images")

### Plot output
#plt.imshow(output.reshape(1,-1),cmap='gray')
#print("\nGuess",np.argmax(output))


