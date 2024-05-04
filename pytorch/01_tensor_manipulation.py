# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 19:04:40 2023

@author: marcos
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

###############################################################################
# REDUCE AND INCREASE DIMENSIONS
# Create a 4x3x2 tensor
tensor = np.random.rand(4, 3, 6, 6)
tensor = torch.tensor(tensor)

print(tensor.flatten().size())
print(tensor.view(4, -1).size())
print(tensor.unsqueeze(-1).size())
print(tensor.unsqueeze(-1).view(4, 3, 6, -1).size())


# %%
###############################################################################
# CREATE TENSORS WITH BINARY CLASSIFICATION
labels = np.random.rand(20)
labels = (labels > 0.5).astype(int)
labels = torch.tensor(labels)
predicted = np.random.rand(20)
predicted = (predicted > 0.5).astype(int)
predicted = torch.tensor(predicted)

# %%
###############################################################################
# WRITE PREDICTIONS
cat = torch.cat((predicted.unsqueeze(1), labels.unsqueeze(1)), dim=1)
for i, sample in enumerate(cat):
    if sample[0] != sample[1]:
        correctness = "ERRADO"
    else:
        correctness = ""
    print(
        f"{i} - Valor esperado: {sample[0].item()}, Predito: {sample[1].item()} {correctness}"
    )

# %%
###############################################################################
# PLOT PREDICTIONS

y_pred_binary = np.where(predicted > 0.5, 1, 0)
# Convert the tensors to NumPy arrays for plotting
y_test = labels.flatten().numpy()
y_pred_binary = y_pred_binary.flatten()
x = np.arange(len(y_test))

# find the indices where y_test > 0 and y_test == y_pred_binary
indices = np.where((y_test > 0) & (y_test == y_pred_binary))[0]

# create the plot
fig, ax = plt.subplots()
ax.plot(np.arange(len(y_test)), y_test, color="blue", label="test")
ax.plot(np.arange(len(y_pred_binary)), y_pred_binary, color="red", label="predict")

for i in range(len(y_test)):
    if y_test[i] > 0 and y_test[i] == y_pred_binary[i]:
        ax.axvspan(i - 0.5, i + 0.5, facecolor="green", alpha=0.5)
    if y_test[i] > 0 and y_test[i] != y_pred_binary[i]:
        ax.axvspan(i - 0.5, i + 0.5, facecolor="red", alpha=0.5)
    if y_test[i] == 0 and y_test[i] != y_pred_binary[i]:
        ax.axvspan(i - 0.5, i + 0.5, facecolor="purple", alpha=0.5)

plt.legend()
plt.title("Random Binary Classification")
plt.show()

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

print("\nConfusion Matrix")
print(cm)
