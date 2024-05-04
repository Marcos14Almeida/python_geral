
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_decision_regions

# Generate some toy data
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a logistic regression model to the training data
model = LogisticRegression()
model.fit(X_train, y_train)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Fit a neural network classifier to the training data
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=4000, random_state=42)
model.fit(X_train, y_train)

# Fit a SVM classifier to the training data
model = SVC(kernel='linear', C=3.0, random_state=42)
model.fit(X_train, y_train)

# Plot the decision boundaries
plot_decision_regions(X_test, y_test, clf=model, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary Plot')
plt.show()

###############################################################################
# 3 CLASSES
###############################################################################
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

# Generate some toy data
X, y = make_blobs(n_samples=300, centers=[[2, 2], [-2, -2], [2, -2]], cluster_std=0.8, random_state=42)

# Fit a SVM classifier to the data
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X, y)

# Plot the decision regions
plot_decision_regions(X, y, clf=model, legend=3)

# Add axis labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions for 3 Classes')

# Show the plot
plt.show()
