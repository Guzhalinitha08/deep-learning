import numpy as np

np.random.seed(0)

true_mean = 10
true_variance = 4

data = np.random.normal(true_mean, np.sqrt(true_variance), 1000)

estimated_mean = np.mean(data)
estimated_variance = np.mean((data - estimated_mean) ** 2)

print("Actual Mean:", true_mean)
print("Estimated Mean:", estimated_mean)

print("Actual Variance:", true_variance)
print("Estimated Variance:", estimated_variance)
