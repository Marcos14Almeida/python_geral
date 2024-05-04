
# Using Monte Carlo Dropout Model
# to get the uncertainty of the model for each prediction

# =============================================================================
# ================================= Libraries =================================
# =============================================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

# =============================================================================
#                                     Main
# =============================================================================

print()
print("Monte Carlo Dropout Model")
print()


# Define the Monte Carlo Dropout model
class MCDDropoutModel(keras.Model):
    def __init__(self, rate):
        super(MCDDropoutModel, self).__init__()
        self.rate = rate
        self.dropout_layer = tf.keras.layers.Dropout(rate)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(8, activation='relu')
        self.dense4 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.dropout_layer(inputs, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


# Generate data
n_points = 200
X = np.linspace(-5, 5, num=n_points).reshape(-1, 1)
y = 20 * np.sin(X) + np.random.normal(size=(n_points, 1)) * 3

# Normalize inputs
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Create the Monte Carlo Dropout model
rate = 0.2  # Dropout rate
model = MCDDropoutModel(rate)

# Number of Monte Carlo samples
n_samples = 500

# Training parameters
epochs = 100
batch_size = 64
optimizer = tf.keras.optimizers.Adam()

# Training loop
for epoch in range(epochs):
    print("Epoch: " + str(epoch))
    indices = np.random.permutation(X.shape[0])
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for batch_start in range(0, X.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, X.shape[0])
        X_batch = X_shuffled[batch_start:batch_end]
        y_batch = y_shuffled[batch_start:batch_end]

        with tf.GradientTape() as tape:
            predictions = model(X_batch, training=True)
            loss = tf.keras.losses.mean_squared_error(y_batch, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Perform Monte Carlo Dropout predictions
predictions = []
for _ in range(n_samples):
    predictions.append(model(X, training=True))  # Enable dropout during testing


# Stack predictions and calculate mean and standard deviation
predictions = np.hstack(predictions)
mean_prediction = np.mean(predictions, axis=1)
std_deviation = np.std(predictions, axis=1)

# Denormalize predictions
mean_prediction = (mean_prediction * X_std) + X_mean
std_deviation = std_deviation * X_std

# Print mean prediction and standard deviation
for i in range(len(X)):
    print(f"Input: {X[i][0]:.2f}  Mean Prediction: {mean_prediction[i]:.4f}  Std Deviation: {std_deviation[i]:.4f}")

# Denormalize predictions
mean_prediction = (mean_prediction * X_std) + X_mean
std_deviation = std_deviation * X_std

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X.flatten(), y.flatten(), label='Data')
plt.plot(X.flatten(), mean_prediction.flatten(), color='red', label='Mean Prediction')
plt.fill_between(X.flatten(), (mean_prediction - 2 * std_deviation).flatten(),
                 (mean_prediction + 2 * std_deviation).flatten(), color='gray', alpha=0.3, label='Uncertainty')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Monte Carlo Dropout Predictions')
plt.legend()
plt.show()
