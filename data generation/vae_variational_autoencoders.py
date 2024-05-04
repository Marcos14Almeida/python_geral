
# Using Variational Autoencoders (VAE)
# Generate synthetic data from the `mtcars` dataset
# Represents each generated row for each variable

# =============================================================================
# ================================= Libraries =================================
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# =============================================================================
#                                     Main
# =============================================================================

# Load and preprocess the dataset
url = 'https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv'  # noqa
mtcars = pd.read_csv(url)

# Select the numerical columns
numeric_cols = ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
dataset = mtcars[numeric_cols].values

# Normalize the data (example: scaling between 0 and 1)
dataset = (dataset - dataset.min(axis=0)) / (dataset.max(axis=0) - dataset.min(axis=0))

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
train_data = dataset[:train_size]
test_data = dataset[train_size:]

# Define the VAE model
input_dim = dataset.shape[1]
latent_dim = 2  # lower-dimensional representation of the input data, capturing its essential features.

# Encoder
encoder_inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(256, activation='relu')(encoder_inputs)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)


# Reparameterization trick
# It takes z_mean and z_log_var as inputs and generates a random sample z based on the mean and log variance.
def sampling(args):
    z_mean, z_log_var = args
    epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
# The decoder takes the samples from the latent space and maps them back to the original data space.
decoder_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(256, activation='relu')(decoder_inputs)
decoder_outputs = layers.Dense(input_dim, activation='sigmoid')(x)

# Define the VAE model
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, vae_outputs, name='vae')

# Define the loss function for VAE
reconstruction_loss = keras.losses.mean_squared_error(encoder_inputs, vae_outputs)
reconstruction_loss *= input_dim
kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
kl_loss = keras.backend.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compile and train the VAE model
vae.compile(optimizer='adam')
vae.fit(train_data, epochs=1000, batch_size=64)

# Make predictions
predictions = vae.predict(test_data)

# Print the generated samples
print(test_data)
print(predictions)

# plot the data as a heat map
fig, ax = plt.subplots()
im = ax.imshow(abs(test_data-predictions), cmap='Reds')
# Add colorbar
cbar = plt.colorbar(im)

# Adjust spacing between subplots and display the plot
plt.tight_layout()
plt.title("Difference between real data and generated data")
plt.show()

# plot the data as a heat map
fig, ax = plt.subplots()
im = ax.imshow(predictions, cmap='YlGn')
# Add colorbar
cbar = plt.colorbar(im)

# Adjust spacing between subplots and display the plot
plt.tight_layout()
plt.title("Generated Data")
plt.show()
