#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 19:57:33 2025

@author: lyes
"""
# Pytorch turorial: Tensors

import torch
import numpy as np

# Initialize from explicit data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

# Initialize from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Initialize to match size of another tensor
x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)

###### Attributes
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the current accelerator if available
# Not working, maybe must be enabled first somewhere
if 0: #torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
    print("Moved tensor to accelerator")

####### Operations
# Indexing and slicing
tensor = torch.ones(3,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")
print(f"Last column: {tensor[:,-1]}")
tensor[:,1] = 0
print(tensor)

# Joining tensors
t1 = torch.cat([tensor,tensor,tensor],dim=1)
print(t1)

### Arithmetic operators
# Matrix multiplication
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.zeros_like(y1)
torch.matmul(tensor,tensor.T,out=y3)

# Elementwise multiplication
z1 = tensor*tensor
z2 = tensor.mul(tensor)
z3 = torch.zeros_like(tensor)
torch.mul(tensor,tensor, out=z3)

# Single element tensors: convert to standard Python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item,type(agg_item))

### Bridge with NumPy
# Both torch.numpy and torch.from_numpy generate a pointer to the same memory
# -> Modifying tensor or np array affects both
t = torch.ones(5)
n = t.numpy()
print(f"Original t: {t}")
print(f"Original n: {n}")

t.add_(1)
print(f"Modified t: {t}")
print(f"Modified n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)
print(f"Original t: {t}")
print(f"Original n: {n}")

np.add(n,1, out=n)
print(f"Modified t: {t}")
print(f"Modified n: {n}")