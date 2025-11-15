#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 15:19:17 2025

@author: lyes
"""
# Automatic differentiation to find grad
import torch

# Define minimal, single-layer model
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Compute gradients (same for multi-layer models)
loss.backward()
print("\nWeight grad:\n",w.grad)
print("\nBias grad:\n", b.grad)