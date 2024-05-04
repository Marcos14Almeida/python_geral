# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 09:53:49 2023

@author: marcos
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare data
X_numpy, y_numpy = datasets.make_regression(
    n_samples=100, n_features=6, noise=20, random_state=4
)

# cast to float Tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) Model
# Linear model f = wx + b
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) Loss and optimizer
learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")

# Plot
predicted = model(X).detach().numpy()

pred = X_numpy[:, 0]
pred = pred.reshape((100, 1))
y_numpy = y_numpy.reshape((100, 1))
# pred.sort()
# pred = X_numpy
pred = np.append(pred, y_numpy, axis=1)
pred = np.append(pred, predicted, axis=1)
pred = pred[pred[:, 0].argsort()]

plt.plot(pred[:, 0], pred[:, 1], "ro")
plt.plot(pred[:, 0], pred[:, 2], "b")
plt.show()
