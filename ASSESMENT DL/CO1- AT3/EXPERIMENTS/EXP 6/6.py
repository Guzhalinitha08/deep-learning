from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Linearly Separable Dataset
X1, y1 = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)

pred = perceptron.predict(X_test)
print("Perceptron Accuracy (Linear Dataset):", accuracy_score(y_test, pred))

# Non-Linearly Separable Dataset
X2, y2 = make_circles(n_samples=500, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

perceptron.fit(X_train, y_train)
pred = perceptron.predict(X_test)

print("Perceptron Accuracy (Non-Linear Dataset):", accuracy_score(y_test, pred))

# Multilayer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, random_state=42)

mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)

print("MLP Accuracy:", accuracy_score(y_test, pred))
