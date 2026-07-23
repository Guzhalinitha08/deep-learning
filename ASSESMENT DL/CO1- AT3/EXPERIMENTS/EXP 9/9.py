import numpy as np

X = np.random.rand(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1) * 0.1

w = np.random.randn(1, 1)
b = 0

learning_rate = 0.1
batch_size = 10
epochs = 100

for epoch in range(epochs):

    for i in range(0, len(X), batch_size):

        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        predictions = X_batch.dot(w) + b

        dw = -2 * X_batch.T.dot(y_batch - predictions) / len(X_batch)
        db = -2 * np.mean(y_batch - predictions)

        w = w - learning_rate * dw
        b = b - learning_rate * db

print("Final Weight:", w)
print("Final Bias:", b)
