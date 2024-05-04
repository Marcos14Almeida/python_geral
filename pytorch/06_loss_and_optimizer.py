# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 00:17:13 2023

@author: marcos
"""
# https://www.youtube.com/watch?v=c36lUUr864M&ab_channel=PatrickLoeber

# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights

import torch
import torch.nn as nn

# Linear regression
# f = w * x

# here : f = 2 * x

# 0) Training samples
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# 1) Design Model: Weights to optimize and forward function
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def forward(x):
    return w * x


print(f"\nPrediction before training: f(5) = {forward(5).item():.3f}\n")

# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 100

# callable function
loss = nn.MSELoss()

optimizer = torch.optim.SGD([w], lr=learning_rate)

# 3) Training loop
for epoch in range(n_iters):
    # predict = forward pass
    y_predicted = forward(X)

    # loss
    l = loss(Y, y_predicted)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print("epoch ", epoch + 1, ": w = ", w, " loss = ", l)

print(f"\nPrediction after training: f(5) = {forward(5).item():.3f}\n")
