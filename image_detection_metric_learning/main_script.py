
# =============================================================================
# ================================= Libraries =================================
# =============================================================================

from tensorflow.keras.layers import Lambda
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
import matplotlib.pyplot as plt


# =============================================================================
#                                   Functions
# =============================================================================
def show_grid_images(images, titles="", global_title=""):

    titlesize = 14

    print()
    print("SHOW IMAGE")
    if isinstance(images, dict):
        titles = list(images.keys())
        images = list(images.values())
    elif isinstance(images, np.ndarray):
        print("ARRAY")
        print(images.shape)
        # numpy_array = np.random.rand(*images.shape)
        # Convert the NumPy array to a list of items
        # images = [numpy_array[i] for i in range(images.shape[0])]
        if not isinstance(titles, np.ndarray):
            titles = [""] * len(images)
        else:
            titlesize = 8
    elif isinstance(images, list):
        if len(titles) == 0:
            titles = [""] * len(images)
    else:
        print("1 image")
        show_1image(images, titles)
        return

    if len(images) > 150:
        images = images[0:150]
        titles = titles[0:150]

    num_images = len(images)

    print(f"Num. Images Plot: {num_images}")

    if (num_images > 25):
        num_columns = 10
    elif (num_images < 5):
        num_columns = num_images - 1
    else:
        num_columns = 5
    num_rows = -(-num_images // num_columns)  # Ceiling division to ensure enough rows

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 15))

    # Loop through the images and display them in subplots
    for i in range(num_rows):
        for j in range(num_columns):
            index = i * num_columns + j
            if index < num_images:
                ax = axes[i, j]
                ax.imshow(images[index], cmap='gray')  # Display the image in grayscale
                ax.set_title(titles[index], fontsize=titlesize, pad=4)
                ax.axis('off')  # Turn off axis labels

    # Remove any empty subplots
    for i in range(num_images, num_rows * num_columns):
        fig.delaxes(axes.flatten()[i])

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    if len(global_title) > 0:
        fig.text(
            0.5, 0.95, global_title,
            ha='center', va='center', fontsize=16, weight='bold')
    # Show the grid of images
    plt.show()


def show_1image(X, y=""):
    # Display the image using matplotlib
    plt.imshow(X)
    plt.title(y, fontsize=16, pad=16)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()


def plot_loss_curve(history):
    # Extract loss values from the history
    train_loss = history.history['loss']

    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curve')
    plt.grid(True)
    plt.show()


# Load and preprocess your .jpg image
def image_to_vector(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust the target size
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    x = x.squeeze()

    person_name = img_path.split("_0")[0].split("\\")[1]
    person_name = person_name.replace("_", " ")

    print(f" - {person_name}")

    return x, person_name


# =============================================================================
#                                     Main
# =============================================================================

folder_path = 'pictures'

# List all files in the folder
all_files = os.listdir(folder_path)

# Filter for image files (e.g., .jpg, .png, .jpeg, etc.)
image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

X = []
Y = []

# Loop through the image files
print("\nLoading image:")
for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(folder_path, image_file)
    x, y = image_to_vector(image_path)
    X.append(x)
    Y.append(y)

# Convert X and Y to NumPy arrays
X = np.array(X)
Y = np.array(Y)

print()
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
print()

# show_grid_images(X, Y, "DATASET")

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


def get_anchors(X, Y):

    anchor_labels = []
    anchor_images = []
    positive_images = []
    negative_images = []
    negative_labels = []

    for i in range(len(Y)):
        anchor_image = X[i]
        anchor_label = Y[i]

        positive_indices = np.where(Y == anchor_label)[0]
        negative_indices = np.where(Y != anchor_label)[0]

        if (len(positive_indices) > 1):
            # Make the positive indice different from the current image
            positive_indices = positive_indices[positive_indices != i]
        if len(positive_indices) == 0 or len(negative_indices) == 0:
            continue

        positive_index = random.choice(positive_indices)
        negative_index = random.choice(negative_indices)

        positive_image = X[positive_index]
        negative_image = X[negative_index]
        negative_label = Y[negative_index]

        anchor_images.append(anchor_image)
        anchor_labels.append(anchor_label)
        positive_images.append(positive_image)
        negative_images.append(negative_image)
        negative_labels.append(negative_label)

    anchor_images = np.array(anchor_images)
    positive_images = np.array(positive_images)
    negative_images = np.array(negative_images)
    anchor_labels = np.array(anchor_labels)
    negative_labels = np.array(negative_labels)

    return anchor_images, positive_images, negative_images, anchor_labels, negative_labels


def get_triplets(X, Y):
    anchors, positives, negatives, anchor_labels, neg_labels = get_anchors(X, Y)
    triplets = [anchors, positives, negatives]
    triplets = np.array(triplets)
    return triplets, anchor_labels, neg_labels


anchors, positives, negatives, labels, neg_labels = get_anchors(X, Y)
# show_grid_images(anchors, labels, "Anchor Images")
# show_grid_images(positives, labels, "Positive Images")
# show_grid_images(negatives, labels, "Negative Images")

triplets, triplets_labels, neg_labels = get_triplets(X, Y)
triplets_train, triplets_labels_train, neg_labels_train = get_triplets(X_train, Y_train)
triplets_test, triplets_labels_test, neg_labels_test = get_triplets(X_test, Y_test)
# for i in range(len(triplets[0])):
#     show_grid_images(
#         [triplets[0][i], triplets[1][i], triplets[2][i]],
#         ["Anchor: "+triplets_labels[i], "Positive: "+triplets_labels[i], "Negative: "+neg_labels_test[i]],
#         "Triplet Images " + triplets_labels[i])

print()
print("Get Images Triplets")
print(f"triplets size: {triplets.shape}")
print(f"triplets_train size: {triplets_train.shape}")
print(f"triplets_test size: {triplets_test.shape}")

# Prepare triplet inputs and labels
triplet_inputs = [triplets_train[i] for i in range(3)]
# All the triplets contains valid triplets images therefore: 1
triplet_labels = np.ones(len(triplet_inputs[0]))

# =============================================================================
#                             Model
# =============================================================================


# Define a Siamese network model
def create_siamese_model(input_shape):
    input = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(input)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='relu')(x)

    model = keras.Model(inputs=input, outputs=x)

    # print(model.summary())

    return model


# Create two identical subnetworks (Siamese twins)
input_shape = (224, 224, 3)
base_network = create_siamese_model(input_shape)

anchor_input = keras.layers.Input(shape=input_shape)
positive_input = keras.layers.Input(shape=input_shape)
negative_input = keras.layers.Input(shape=input_shape)

# Get the embeddings for each input
anchor_embeddings = base_network(anchor_input)
positive_embeddings = base_network(positive_input)
negative_embeddings = base_network(negative_input)

# Calculate the Euclidean distance between the embeddings
distance_ap = tf.norm(anchor_embeddings - positive_embeddings, axis=-1)
distance_an = tf.norm(anchor_embeddings - negative_embeddings, axis=-1)

# Combine anchor, positive, and negative embeddings into a single model
merged = Lambda(
    lambda x: [x[0], x[1], x[2]], output_shape=lambda x: x[0]
    )([anchor_embeddings, positive_embeddings, negative_embeddings])

# Define the Siamese model with triplet loss
siamese_model = keras.Model(
    inputs=[anchor_input, positive_input, negative_input],
    outputs=merged
    )


# Define the triplet loss function
def triplet_loss(y_true, y_pred, margin=0.2):
    return tf.reduce_mean(
        tf.maximum(
            y_true * tf.square(y_pred),
            (1.0 - y_true) * tf.square(tf.maximum(margin - y_pred, 0.0))
        )
    )


# Train the Siamese network
print("\nFIT MODEL")

# Compile the Siamese model with triplet loss
siamese_model.compile(optimizer='adam', loss=triplet_loss)

# Create an instance of the EarlyStopping callback
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

history = siamese_model.fit(triplet_inputs, triplet_labels,
                            epochs=2, batch_size=4, shuffle=True, callbacks=[early_stopping])
# plot_loss_curve(history)

# Save the model
siamese_model.save('models/siamese_model.h5')

print("\n")

# =============================================================================
#                             Train Evaluation
# =============================================================================


def evaluate(base_network, X, Y):
    # Generate embeddings for X using the trained base_network
    train_embeddings = base_network.predict(X)

    # Initialize variables for accuracy calculation
    total_correct = 0
    total_samples = len(X)

    # Iterate through each image in X
    for i in range(total_samples):
        anchor_embedding = train_embeddings[i]  # Get the embedding of the current image

        # Calculate distances or similarities to all other images in X
        distances = np.linalg.norm(train_embeddings - anchor_embedding, axis=1)  # Using Euclidean distance

        # Exclude the current image when finding the closest one
        distances[i] = np.inf  # Set the distance to itself as infinity

        # Find the index of the closest image
        closest_index = np.argmin(distances)

        # Get the true label for the current image
        true_label = Y[i]

        # Get the true label for the closest image
        predicted_label = Y[closest_index]

        print(f"true: {true_label}     pred: {predicted_label}")

        # Check if the predicted label matches the true label
        if predicted_label == true_label:
            total_correct += 1

    # Calculate accuracy
    accuracy = total_correct / total_samples

    print()
    print(f"Accuracy on X_train: {accuracy * 100:.2f}%")
    print()


print()
print("TRAIN EVALUATION")
evaluate(base_network, X_train, Y_train)


# =============================================================================
#                             Test Evaluation
# =============================================================================


# Initialize an empty list to store the embeddings
embeddings_of_all_images = []

# Assuming you have a list or array of images called 'all_images'
for X_image in X:
    # Generate embeddings for each image using the trained base_network
    image_embedding = base_network.predict(X_image.reshape(1, 224, 224, 3))

    # Append the embedding to the list
    embeddings_of_all_images.append(image_embedding)


# Convert the list of embeddings to a NumPy array
embeddings_of_all_images = np.array(embeddings_of_all_images)

# Embeddings for test
# Generate embeddings for X_test using the trained base_network
test_embeddings = base_network.predict(X_test)

# Initialize an empty list to store the predicted Y labels
predicted_labels = []

# Iterate through each element in X_test
for test_index, test_embedding in enumerate(test_embeddings):
    min_distance = float('inf')  # Initialize minimum distance as infinity
    closest_label = None

    # Iterate through all images in your dataset
    for dataset_index, dataset_embedding in enumerate(embeddings_of_all_images):
        # Calculate the Euclidean distance between the test embedding and the dataset embedding
        distance = np.linalg.norm(test_embedding - dataset_embedding)

        # Check if this distance is the smallest encountered so far
        if distance < min_distance:
            min_distance = distance
            closest_label = Y[dataset_index]  # Get the label associated with the closest embedding

    # Append the closest label to the predicted_labels list
    predicted_labels.append(closest_label)

# Convert the list of predicted labels to a NumPy array
# Now, 'predicted_labels' contains the predicted Y labels for each element in X_test
predicted_labels = np.array(predicted_labels)

# Compare predicted labels with ground truth labels
correct_predictions = (predicted_labels == Y_test).astype(int)

# Calculate accuracy
accuracy = np.mean(correct_predictions)

print()
print("TEST EVALUATION")
for i in range(len(predicted_labels)):
    print(f"true: {Y_test[i]}     pred:{predicted_labels[i]}")
print(f"\nAccuracy: {accuracy * 100:.2f}%")


# =============================================================================
#                              Add image
# =============================================================================

print()
print("ADD IMAGE TO DATABASE")
new_image_path = os.path.join("marcelinho", "Marcelinho_0001_0000.jpg")
new_image, new_label = image_to_vector(new_image_path)

X = np.append(X, [new_image], axis=0)
Y = np.append(Y, [new_label], axis=0)

# =============================================================================
#                               Predict new image
# =============================================================================

print()
print("PREDICT NEW IMAGE")


def distance_to_score(distance):
    return 100 - (distance * 100)


def predict_image(image_path):
    new_image, new_label = image_to_vector(image_path)

    # Obtain embeddings for the existing dataset (X) and the new image
    #               ->    vetor descritor do modelo    <-
    existing_embeddings = base_network.predict(X)
    new_image_embedding = base_network.predict(np.expand_dims(new_image, axis=0))

    # Calculate distances or similarities between the new image's embedding and existing embeddings
    distances = np.linalg.norm(existing_embeddings - new_image_embedding, axis=1)  # Using Euclidean distance

    # Find the index of the closest image
    closest_index = np.argmin(distances)

    # Predict the label of the new image based on the closest match
    predicted_label = Y[closest_index]

    print(f"Predicted Label: {predicted_label}")
    print(len(new_image))
    show_1image(
            new_image,
            "Real: " + new_label
            + "\nPred: " + predicted_label
            + " Score: {:.2f}%".format(distance_to_score(closest_index)),
    )


image_path = os.path.join("marcelinho", "Marcelinho_0002_0000_inferencia.jpg")
predict_image(image_path)


# =============================================================================
#                                     END
# =============================================================================
print()
print("-------------------   END   ------------------")
print()
