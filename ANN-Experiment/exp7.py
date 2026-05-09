# KRISHNA KUMAR TRIVEDI
# B-25

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,1,1,0]).reshape(-1,1)

learning_rate=0.1
epochs=10000

w1=np.random.uniform(-1, 1, (2, 4))
b1=np.random.uniform(-1, 1, (1, 4))

w2=np.random.uniform(-1, 1, (4, 1))
b2=np.random.uniform(-1, 1, (1, 1))

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

loss_history=[]

for epoch in range(epochs):
    y1=np.dot(x,w1)+b1
    a1=sigmoid(y1)  

    y2=np.dot(a1,w2)+b2
    y_pred=sigmoid(y2)

    loss=np.mean((y-y_pred)**2)
    loss_history.append(loss)

    error_output = y - y_pred
    d_output = error_output * sigmoid_derivative(y_pred)

    error_hidden = np.dot(d_output, w2.T)
    d_hidden = error_hidden * sigmoid_derivative(a1)

    # -------- Update Weights & Bias --------
    w2 += np.dot(a1.T, d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    w1 += np.dot(x.T, d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

print("\nTruth Table (XOR)")
print("A  B  |  Expected  |  Predicted")

for i in range(len(x)):
    print(x[i][0], x[i][1], " |     ",
          int(y[i][0]), "     |     ",
          int(round(y_pred[i][0])))

plt.plot(loss_history)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

