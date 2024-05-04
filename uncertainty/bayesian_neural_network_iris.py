
# Bayesian Neural Network Classification
# https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/demos/Bayesian%20Neural%20Network%20Classification.ipynb
# =============================================================================
# ================================= Libraries =================================
# =============================================================================
from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim

import torchbnn as bnn
import matplotlib.pyplot as plt

# =============================================================================
#                                1. Load Iris Data
# =============================================================================

print("\nLOAD IRIS DATASET...\n")
iris = datasets.load_iris()
X = iris.data
Y = iris.target
x, y = torch.from_numpy(X).float(), torch.from_numpy(Y).long()
x.shape, y.shape

# =============================================================================
#                                2. Define Model
# =============================================================================

print("\nDEFINE MODEL...\n")
model = nn.Sequential(
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=4, out_features=100),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=3),
)
ce_loss = nn.CrossEntropyLoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01

optimizer = optim.Adam(model.parameters(), lr=0.01)

# =============================================================================
#                                3. Train Model
# =============================================================================

print("\nTRAIN MODEL...\n")
kl_weight = 0.1
max_steps = 3000
for step in range(max_steps):
    print("step: " + str(step) + "/" + str(max_steps))
    pre = model(x)
    ce = ce_loss(pre, y)
    kl = kl_loss(model)
    cost = ce + kl_weight*kl

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

_, predicted = torch.max(pre.data, 1)
total = y.size(0)
correct = (predicted == y).sum()
print('- Accuracy: %f %%' % (100 * float(correct) / total))
print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item()))

# =============================================================================
#                              4. Test Model
# =============================================================================

print("\nPLOTS...\n")


def draw_plot(predicted):
    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    z1_plot = ax1.scatter(X[:, 0], X[:, 1], c=Y)
    z2_plot = ax2.scatter(X[:, 0], X[:, 1], c=predicted)

    plt.colorbar(z1_plot, ax=ax1)
    plt.colorbar(z2_plot, ax=ax2)

    ax1.set_title("REAL")
    ax2.set_title("PREDICT")

    plt.show()


pre = model(x)
_, predicted = torch.max(pre.data, 1)
draw_plot(predicted)

# Bayesian Neural Network will return different outputs even if inputs are same.
# In other words, different plots will be shown every time forward method is called.
pre = model(x)
_, predicted = torch.max(pre.data, 1)
draw_plot(predicted)
