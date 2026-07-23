import numpy as np

inputs = np.array([1, 2, 3])

weights = np.array([0.5, -0.6, 0.8])

bias = 0.2

net = np.dot(inputs, weights) + bias

sigmoid = 1 / (1 + np.exp(-net))

relu = max(0, net)

print("Net Input:", net)

print("Sigmoid Output:", sigmoid)

print("ReLU Output:", relu)
