"""
Partial Dependency Plot
https://scikit-learn.org/stable/modules/partial_dependence.html
24/05/2023
"""

from sklearn.datasets import load_iris
from sklearn.inspection import partial_dependence
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay

# -----------------------------------------------------------------------------
X, y = make_hastie_10_2(random_state=0)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
features = [0, 1, (0, 1)]
# PartialDependenceDisplay.from_estimator(clf, X, features)

# -----------------------------------------------------------------------------
# Raw values

results = partial_dependence(clf, X, [0])
print(results["average"])
print(results["values"])


# -----------------------------------------------------------------------------
# Iris Dataset
iris = load_iris()
print(iris.feature_names)
mc_clf = GradientBoostingClassifier(n_estimators=10, max_depth=1).fit(iris.data, iris.target)
features = [3, 2, (3, 2)]
# PartialDependenceDisplay.from_estimator(mc_clf, iris.data, features, target=0)

# -----------------------------------------------------------------------------
"""
Individual conditional expectation (ICE) plot
"""
iris = load_iris()
features = [3, 2]
clf = GradientBoostingClassifier(n_estimators=10, max_depth=1)
clf.fit(iris.data, iris.target)
PartialDependenceDisplay.from_estimator(clf, iris.data, features, kind='both', centered=True, target=0)
