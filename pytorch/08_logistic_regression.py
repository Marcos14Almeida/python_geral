# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 10:23:27 2023

@author: marcos
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1) Model
# Linear model f = wx + b , sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred


model = LogisticRegression(n_features, 5)

# 2) Loss and optimizer
num_epochs = 200
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy: {acc.item():.4f}")

###############################################################################
# PLOT PREDICTIONS

import matplotlib.pyplot as plt
import numpy as np
import torch


y_pred_binary = torch.where(
    y_predicted > 0.5, torch.ones_like(y_predicted), torch.zeros_like(y_predicted)
)
y_pred_binary = y_pred_binary.flatten()
# Convert the tensors to NumPy arrays for plotting
y_test = y_test.flatten()

# find the indices where y_test > 0 and y_test == y_pred_binary
indices = np.where((y_test > 0) & (y_test == y_pred_binary))[0]

# create the plot
fig, ax = plt.subplots()
ax.plot(np.arange(len(y_test)), y_test, color="blue", label="test")
ax.plot(np.arange(len(y_pred_binary)), y_pred_binary, color="red", label="predict")

for i in range(len(y_test)):
    # if y_test[i] > 0 and y_test[i] == y_pred_binary[i]:
    #     ax.axvspan(i - 0.5, i + 0.5, facecolor="gray", alpha=0.5)
    if y_test[i] > 0 and y_test[i] != y_pred_binary[i]:
        ax.axvspan(i - 0.5, i + 0.5, facecolor="red", alpha=0.5)
    if y_test[i] == 0 and y_test[i] != y_pred_binary[i]:
        ax.axvspan(i - 0.5, i + 0.5, facecolor="purple", alpha=0.5)

plt.title("Binary Classification")
plt.show()
