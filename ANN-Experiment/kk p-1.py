# KRISHNA KUMAR TRIVEDI
# TE-B-25
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 200)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Sigmoid
plt.figure()
plt.plot(x, sigmoid(x))
plt.title("Sigmoid")
plt.grid()

# Tanh
plt.figure()
plt.plot(x, tanh(x))
plt.title("Tanh")
plt.grid()

# ReLU
plt.figure()
plt.plot(x, relu(x))
plt.title("ReLU")
plt.grid()

# Softmax
x_soft = np.linspace(-5, 5, 50)
plt.figure()
plt.plot(x_soft, softmax(x_soft))
plt.title("Softmax")
plt.grid()

# Show all graphs together
plt.show()
