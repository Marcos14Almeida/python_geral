# =============================================================================
# ================================= Libraries =================================
# =============================================================================

import numpy as np
import os
from sklearn.metrics.pairwise import euclidean_distances
from main_script import image_to_vector

# =============================================================================
#                                   Functions
# =============================================================================

# =============================================================================
#                               Compare images
# =============================================================================


def compare_images(new_image, reference_image, base_network):
    print()
    # Create pairs of anchor and positive images
    positive_input = np.expand_dims(reference_image, axis=0)

    x, _ = image_to_vector(new_image)
    anchor_image = x
    anchor_input = np.expand_dims(anchor_image, axis=0)
    # Calculate the Euclidean distance between the embeddings
    distance = np.linalg.norm(base_network.predict(anchor_input) - base_network.predict(positive_input))
    # Set a threshold to determine similarity
    threshold = 0.5  # Adjust as needed

    if distance < threshold:
        print("Images are similar.")
        print(f"Distance: {distance}")
        return True
    else:
        print("Images are different.")
        print(f"Distance: {distance}")
        return False


reference_image = os.path.join("marcelinho", "Marcelinho_0002_0000_inferencia.jpg")
x, _ = image_to_vector(reference_image)
new_image = x


new_image = os.path.join("pictures", "Mike_Matheny_0004_0000.jpg")
compare_images(new_image, reference_image, base_network)

new_image = os.path.join("pictures", "Michael_Jordan_0001_0003.jpg")
compare_images(new_image, reference_image, base_network)

new_image = os.path.join("marcelinho", "Marcelinho_0001_0000.jpg")
compare_images(new_image, reference_image, base_network)


# =============================================================================
#                               Closest label
# =============================================================================

def get_closest_label(image_path, base_network, X_train, Y_train):
    new_image, _ = image_to_vector(image_path)

    # Calculate the feature embedding of the new image
    new_embedding = base_network.predict(np.expand_dims(new_image, axis=0))

    # Calculate distances between the new embedding and all embeddings in the dataset
    distances = euclidean_distances(new_embedding, base_network.predict(X_train))

    # Set a threshold to determine similarity
    threshold = 0.5  # Adjust as needed

    # Find the index of the most similar image (smallest distance)
    print()
    print("GET CLOSEST LABEL")
    print("Distances between the current image and the other images")
    print(distances)
    most_similar_index = np.argmin(distances)

    print()
    if distances[0][most_similar_index] < threshold:
        # The new image is similar to the image at most_similar_index
        true_label = Y_train[most_similar_index]
        print(f"Predicted label: {true_label}")
    else:
        print("No similar image found in the dataset.")


image_path = os.path.join("pictures", "Michael_Jordan_0003_0000.jpg")
get_closest_label(image_path, base_network, X_train, Y_train)
