import numpy as np
import matplotlib.pyplot as plt

# Sample Dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 4, 5, 4, 5], dtype=float)

# Initialize parameters
m = 0
c = 0
learning_rate = 0.01
epochs = 1000
n = len(X)

loss_history = []

# Gradient Descent
for i in range(epochs):
    y_pred = m * X + c

    dm = (-2 / n) * np.sum(X * (Y - y_pred))
    dc = (-2 / n) * np.sum(Y - y_pred)

    m = m - learning_rate * dm
    c = c - learning_rate * dc

    loss = np.mean((Y - y_pred) ** 2)
    loss_history.append(loss)

print("Slope (m):", m)
print("Intercept (c):", c)

# Plot Regression Line
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, m * X + c, color='red', label='Regression Line')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()

# Plot Learning Curve
plt.plot(loss_history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.show()
