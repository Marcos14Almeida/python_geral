# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 19:05:54 2023

@author: marcos
"""

import torch
import numpy as np

print(torch.__version__)
print(np.__version__)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    z = z.to("cpu")

x = torch.empty(2, 3)
x = torch.rand(2, 3)
# print(x)
# print(x[:, 0])
# print(x[1, 1])
# print(x[1, 1].item())
x = torch.zeros(2, 3)
x = torch.ones(2, 3)
# print(x.dtype)
x = torch.ones(2, 3, dtype=torch.int)
# print(x)
# print(x.size())

x = torch.tensor([2.5, 0.3])
# print(x)
y = torch.rand(2, 3)
z = torch.rand(2, 3)
x = y + z
x = torch.add(y, z)
x = torch.sub(y, z)
x = torch.mul(y, z)
x = torch.div(y, z)
# print(x)

#######################################
# Change Shape
print("\nChange Shape")
x = torch.rand(3, 3)
# print(x.view(1,3,3))
# print(x.view(9))
print(x.view(-1, 9))

# To numpy
# -> Share the same memory location
# To avoid that:
x = torch.ones(5)
y = x.detach().numpy()

# From Numpy
a = np.ones(5)
b = torch.from_numpy(a)


#######################################
# Gradients
x = torch.ones(5, required_grad=True)
