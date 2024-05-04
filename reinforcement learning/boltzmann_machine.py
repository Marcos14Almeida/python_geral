# =============================================================================
# ================================= Libraries =================================
# =============================================================================
import numpy as np
from sklearn.neural_network import BernoulliRBM

# =============================================================================
# Use Case 1
# Generate synthetic binary data
# =============================================================================
print("-"*50)
print(" "*10 + "Generate synthetic binary data")
print("-"*50)
num_samples = 1000
num_features = 10

data = np.random.randint(0, 2, size=(num_samples, num_features))

# Create and train the RBM
rbm = BernoulliRBM(n_components=10, n_iter=100, learning_rate=0.1)
rbm.fit(data)

# Generate samples from the trained RBM
generated_samples = rbm.gibbs(np.ones((num_samples, num_features)))

# Print the generated samples
print("Generated Samples:")
print(generated_samples)
print(data[0])
print(generated_samples[0])

# =============================================================================
# Use Case 2
# Music reccomendation system
# =============================================================================
print("-"*50)
print(" "*10 + "Music reccomendation system")
print("-"*50)
# Assuming you have a dataset of user listening history represented as binary vectors
# Each row represents a user and each column represents a song

# Load your dataset here
# Ex: Green Day, Anti-Flag, RHCP, Offspring, Metallica
mapa = {
    0: "Green Day",
    1: "Anti-Flag",
    2: "Red Hot",
    3: "Offspring",
    4: "Metallica",
    5: "Queen",
    6: "Nirvana",
}

dataset = np.array([[1, 1, 0, 1, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 1, 1, 0],
                    [1, 0, 1, 0, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1],
                    ])

# Create and train the RBM model
rbm = BernoulliRBM(n_components=7, n_iter=100)
rbm.fit(dataset)

# Generate recommendations for a new user
new_user = np.array([[0, 1, 0, 0, 0, 0, 1]])  # New user listening history
recommendations = rbm.gibbs(new_user)  # Generate recommendations


def words(lista):
    show = []
    for i in range(len(lista[0])):
        if str(lista[0][i]) == "True" or lista[0][i] == 1:
            show.append(mapa[i])
    return show


print("TEST:\n")
print("Listens:")
print(words(new_user))
print("Recommendations:")
print(words(recommendations))
