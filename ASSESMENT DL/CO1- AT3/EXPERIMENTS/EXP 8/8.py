import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Gradient Descent
x = 10
learning_rate = 0.1
history = []

for i in range(50):
    gradient = 2 * x
    x = x - learning_rate * gradient
    history.append(x)

plt.plot(history)
plt.title("Gradient Descent")
plt.xlabel("Iterations")
plt.ylabel("x")
plt.show()

# Stochastic Gradient Descent
X, y = make_regression(
    n_samples=500,
    n_features=1,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sgd = SGDRegressor(max_iter=1000)
sgd.fit(X_train, y_train)
pred1 = sgd.predict(X_test)

batch = LinearRegression()
batch.fit(X_train, y_train)
pred2 = batch.predict(X_test)

print("SGD Mean Squared Error:", mean_squared_error(y_test, pred1))
print("Batch Gradient Descent Mean Squared Error:", mean_squared_error(y_test, pred2))
