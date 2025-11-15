#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 20:25:11 2024

@author: lyes
"""

import numpy as np

# Define vectors
v1 = np.array([1, 2, 3])
v2 = np.array([11, 12, 13])

print("Vector 1:", v1)
print("Vector 2:", v2,"\n")

# Define matrices
M1 = np.array([[11,12,13],[21,22,23], [31,32,33]])
M2 = np.array([[1,2,3],[2,3,4], [3,4,5]])

print("Matrix 1:\n", M1, "\n")
print("Matrix 2:\n", M2, "\n")

# Addition and subtraction

v3 = v1 + v2
v4 = v1 - v2

M3 = M1 + M2
M4 = M1 - M2

print("M1 + M2:\n", M3, "\n")
print("Matrix 2:\n", M4) 

# Elementwise multiplication

v5 = v1*v2
M5 = M1*M2

# Matrix multiplication

v6 = np.dot(v1,v2) # Dot product
v62 = v1 @ v2
print("\nDot product:",v6)
print("\nDot product @:",v62)

M6 = np.dot(M1,M2)
M62 = M1 @ M2

print(M6,"\n")
print(M62,"\n")

# Dimensions

M2row3col = np.array([[11,12,13],[21,22,23]])
print("\n",M2row3col)
print("\nDims:", M2row3col.shape)