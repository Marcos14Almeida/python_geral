
# =============================================================================
# ================================= Libraries =================================
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#                                   Functions
# =============================================================================


def positional_encoding(position, d_model):
    """
    Generate positional encodings for a given position and dimensionality.

    Args:
        position (int): The position for which to generate the encoding.
        d_model (int): The dimensionality of the model's embeddings.

    Returns:
        np.ndarray: The positional encoding vector.
    """
    encoding = np.zeros((d_model,))
    for i in range(d_model):
        if i % 2 == 0:
            encoding[i] = np.sin(position / (10000 ** (i / d_model)))
        else:
            encoding[i] = np.cos(position / (10000 ** ((i - 1) / d_model)))
    return encoding


def plot(d_model, positions, positional_encodings):
    # Plot the positional encodings for each dimension
    plt.figure(figsize=(10, 6))
    for i in range(d_model):
        plt.plot(positions, positional_encodings[:, i], label=f'Dimension {i+1}')

    plt.title(f'Positional Encodings (Dimension = {d_model})')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================================
#                                   Main
# =============================================================================

# Define the dimensionality of the model's embeddings
d_model = 4

# Generate positional encodings for positions from 1 to 100
positions = np.arange(1, 101)
positional_encodings = np.array([positional_encoding(pos, d_model) for pos in positions])

plot(d_model, positions, positional_encodings)
