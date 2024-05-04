from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import backend as K
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

print()
print("SHAPE")
print(x_train.shape)
print(x_test.shape)
print()


# Define the Siamese network
def build_siamese_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, (2, 2), activation='relu')(input)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    embeddings = Dense(64, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=embeddings)
    return model


# Define the triplet loss function
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_distance = K.sum(K.square(anchor - positive), axis=-1)
    neg_distance = K.sum(K.square(anchor - negative), axis=-1)
    loss = K.maximum(0.0, pos_distance - neg_distance + alpha)
    return loss


def triplet_loss_original(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_distance = K.sum(K.square(anchor - positive), axis=-1)
    neg_distance = K.sum(K.square(anchor - negative), axis=-1)
    loss = K.maximum(0.0, pos_distance - neg_distance + alpha)
    return loss


# Build the Siamese model
input_shape = (28, 28, 1)  # Fashion MNIST images are 28x28
siamese_network = build_siamese_network(input_shape)

# Create the anchor, positive, and negative input tensors
anchor_input = Input(shape=input_shape)
positive_input = Input(shape=input_shape)
negative_input = Input(shape=input_shape)

# Get the embeddings for each input
anchor_embeddings = siamese_network(anchor_input)
positive_embeddings = siamese_network(positive_input)
negative_embeddings = siamese_network(negative_input)

# Combine anchor, positive, and negative embeddings into a single model
merged = Lambda(
    lambda x: [x[0], x[1], x[2]], output_shape=lambda x: x[0]
    )([anchor_embeddings, positive_embeddings, negative_embeddings])
siamese_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged)

# Compile the Siamese model with triplet loss
siamese_model.compile(loss=triplet_loss_original, optimizer='adam')


# Prepare triplets from the dataset
def create_triplets(data, labels):

    triplets = []
    num_classes = len(np.unique(labels))

    for i in range(num_classes):
        class_indices = np.where(labels == i)[0]
        if len(class_indices) < 2:
            continue
        for j in range(len(class_indices) - 1):
            anchor = data[class_indices[j]]
            positive = data[class_indices[j + 1]]
            negative = data[np.random.choice(np.delete(np.arange(len(data)), class_indices), 1)[0]]
            triplets.append((anchor, positive, negative))

    return np.array(triplets)


# Create triplets from the training data
triplets = create_triplets(x_train, y_train)

loadModel = False

model_path = 'models/siamese_model_mnist.h5'

if (loadModel):
    siamese_model = load_model(model_path, custom_objects={'triplet_loss_original': triplet_loss_original})
else:
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    # Train the Siamese network
    siamese_model.fit(
        [triplets[:, 0], triplets[:, 1], triplets[:, 2]],
        np.zeros(len(triplets)), epochs=6, batch_size=128,
        callbacks=[early_stopping]
    )

    # Save the model
    siamese_model.save('models/siamese_model_mnist.h5')

# =============================================================================
#                             Test Evaluation
# =============================================================================
print()
print("TEST MODELS")
print()


triplets_train = create_triplets(x_train, y_train)
train_embeddings = siamese_network.predict(triplets_train[:, 0].reshape(-1, 28, 28, 1))

# Calculate embeddings for test images
test_embeddings = siamese_network.predict(x_test.reshape(-1, 28, 28, 1))

# Initialize an empty array to store predicted labels
y_pred = []

# Iterate through each test image
for test_idx in range(len(x_test)):
    print(test_idx)
    # Calculate cosine similarities between the test image's embeddings and training embeddings
    test_embeddings = np.array(test_embeddings)
    similarities = cosine_similarity(
        test_embeddings[test_idx].reshape(1, -1), train_embeddings)

    # Find the index of the training image with the highest similarity
    closest_idx = np.argmax(similarities)

    # Assign the label of the closest training image as the predicted label
    predicted_label = y_train[closest_idx]
    y_pred.append(predicted_label)

    if (y_pred[test_idx] == y_test[test_idx]):
        print(" -------------- IGUAL")
    # else:
        # print("diferente")


# Convert the list of predicted labels to a NumPy array
y_pred = np.array(y_pred)

# Calculate accuracy by comparing y_pred with y_test
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Accuracy: 0.1
