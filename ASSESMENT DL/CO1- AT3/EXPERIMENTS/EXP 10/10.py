import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dimensions = [2, 5, 10, 20, 50]

for dim in dimensions:

    X, y = make_classification(
        n_samples=1000,
        n_features=dim,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    average_distance = np.mean(
        np.linalg.norm(X_train[:100] - X_train[100:200], axis=1)
    )

    print("Dimensions:", dim)
    print("Average Distance:", average_distance)
    print("Model Accuracy:", accuracy_score(y_test, predictions))
    print("--------------------------------")
