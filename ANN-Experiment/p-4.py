# KRISHNA KUMAR TRIVEDI
# TE-B-25
# Manual Perceptron Learning Law with Decision Boundary

import numpy as np
import matplotlib.pyplot as plt

# AND Gate Data
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([0,0,0,1])

# Initialize weights and bias
w = np.zeros(2)
b = 0
learning_rate = 0.1

# Training
for epoch in range(20):
    for i in range(len(X)):
        yin = np.dot(w, X[i]) + b
        
        # Step Activation Function
        if yin >= 0:
            y_pred = 1
        else:
            y_pred = 0
        
        error = y[i] - y_pred
        
        # Update rule
        w = w + learning_rate * error * X[i]
        b = b + learning_rate * error

print("Final Weights:", w)
print("Final Bias:", b)

# Plot Data Points
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')

# Decision Boundary
x_vals = np.linspace(-0.5,1.5,100)
y_vals = -(w[0]*x_vals + b)/w[1]

plt.plot(x_vals, y_vals)

plt.title("Manual Perceptron Decision Boundary")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid()

plt.show()