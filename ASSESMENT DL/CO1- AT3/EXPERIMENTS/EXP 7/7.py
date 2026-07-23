import numpy as np

# Inputs
X = np.array([[1, 2]])

# Initial Weights
W = np.array([[0.5],
              [0.3]])

# Bias
b = np.array([[0.1]])

# Target
y = np.array([[1]])

# Learning Rate
lr = 0.1

# Forward Propagation
z = np.dot(X, W) + b
output = 1 / (1 + np.exp(-z))

print("Forward Propagation Output:")
print(output)

# Backpropagation
error = output - y

dW = np.dot(X.T, error)
db = error

W = W - lr * dW
b = b - lr * db

print("\nUpdated Weights:")
print(W)

print("\nUpdated Bias:")
print(b)
