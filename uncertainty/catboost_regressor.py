# Catboost Regressor to predict Uncertainty
# https://towardsdatascience.com/a-new-way-to-predict-probability-distributions-e7258349f464

# =============================================================================
# ================================= Libraries =================================
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor

# =============================================================================
#                                     Main
# =============================================================================

sns.set()

# Number of training and testing examples
n = 1000

# Generate random x values between 0 and 1
x_train = np.random.rand(n)
x_test = np.random.rand(n)

# Generate random noise for the target
noise_train = np.random.normal(0, 0.3, n)
noise_test = np.random.normal(0, 0.3, n)

# Set the slope and y-intercept of the line
a, b = 2, 3

# Generate y values according to the equation y = ax + b + noise
y_train = a * x_train + b + noise_train
y_test = a * x_test + b + noise_test

# Store quantiles 0.01 through 0.99 in a list
quantiles = [q/100 for q in range(1, 100)]

# Format the quantiles as a string for Catboost
quantile_str = str(quantiles).replace('[', '').replace(']', '')

# Fit the multi quantile model
model = CatBoostRegressor(
    iterations=100, loss_function=f'MultiQuantile:alpha={quantile_str}')

model.fit(x_train.reshape(-1, 1), y_train)

# Make predictions on the test set
preds = model.predict(x_test.reshape(-1, 1))
preds = pd.DataFrame(preds, columns=[f'pred_{q}' for q in quantiles])

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_test, y_test)

for col in ['pred_0.05', 'pred_0.5', 'pred_0.95']:
    ax.scatter(x_test.reshape(-1, 1), preds[col], alpha=0.50, label=col)

ax.legend()
plt.title("CatBoost Regressor to predict uncertainty")
plt.show()

coverage_90 = np.mean((y_test <= preds['pred_0.95']) & (y_test >= preds['pred_0.05']))*100
print("Coverage 90%: "+coverage_90)

# Give the model a new input of x = 0.4
x = np.array([0.4])

# We expect the mean of this array to be about 2*0.4 + 3 = 3.8
# We expect the standard deviation of this array to be about 0.30
y_pred = model.predict(x.reshape(-1, 1))

mu = np.mean(y_pred)
sigma = np.std(y_pred)
print("Mean: " + mu)  # Output: 3.836147287742427
print("Sigma: " + sigma)  # Output: 0.3283984093786787

# Plot the predicted distribution
fig, ax = plt.subplots(figsize=(10, 6))
_ = ax.hist(y_pred.reshape(-1, 1), density=True)
ax.set_xlabel('$y$')
ax.set_title(f'Predicted Distribution $P(y|x=4)$, $\mu$ = {round(mu, 3)}, $\sigma$ = {round(sigma, 3)}')
