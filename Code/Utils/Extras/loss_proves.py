import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

shape = (2, 1, 3, 4)

# Generate random integer tensor
random_tensor=torch.tensor([[[[16.2,  7.3, 17.4,  2.6],
          [15.1, 14.8,  4.6,  5.6],
          [ 2.2, 15.6, 19.8,  9.9]]],


        [[[ 7.4, 15.3,  5.2, 10.1],
          [ 9.5, 11.5, 18.4,  0.1],
          [12.5,  3.5,  1.2,  0.8]]]])
#random_tensor = torch.tensor([0.1, 13.5, 2.2, 3.4, 5.1, 4.2, 8.3, 10.9, 8.7, 5.6])
random_tensor = random_tensor.float()
print(random_tensor)

bin_edges = torch.arange(0, math.ceil(random_tensor.max())+1, 1,dtype=torch.float32)
print('edges',bin_edges)
p_y = torch.histogram(random_tensor.view(-1), bins=bin_edges)

print('histogram',p_y[0])

h_invers = p_y[0]/random_tensor.numel()

print(h_invers)

weight1 = 1 / (h_invers+1e-1)
print('weight',weight1)
weight = weight1 / weight1.sum()
print('sum',weight1.sum())
print('weight2',weight)

pixel_weights = {}
for pixel_value, weight_value in enumerate(weight):
        pixel_weights[pixel_value] = weight_value.item()

print(pixel_weights)

mask_tensor = torch.zeros(shape)
for pixel_value, probability in pixel_weights.items():
            mask_tensor[(pixel_value<=random_tensor) & (random_tensor<pixel_value+1)] = probability

print(mask_tensor)

