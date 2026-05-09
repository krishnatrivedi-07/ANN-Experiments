#KRISHNA KUMAR TRIVEDI
#B-25
#PRAC-B-1

import numpy as np
import matplotlib.pyplot as plt

# Network parameters
input_neuron = 2
hidden_neuron = 4
output_neuron = 1
learning_rate = 0.5
epochs = 20000

# XOR Data
X = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])

Y = np.array([[0], 
              [1], 
              [1], 
              [0]])

# Initialize weights and bias
W1 = np.random.randn(input_neuron, hidden_neuron)
W2 = np.random.randn(hidden_neuron, output_neuron)

b1 = np.zeros((1, hidden_neuron))
b2 = np.zeros((1, output_neuron))

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Store loss values for graph
losses = []

# Training using Backpropagation
for i in range(epochs):

    # Forward Pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # Error Calculation
    error = Y - final_output

    # Calculate loss (Mean Squared Error)
    loss = np.mean(np.square(error))
    losses.append(loss)

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Print Loss every 2000 epochs
    if i % 2000 == 0:
        print("Epoch:", i, "Loss:", round(loss,6))

# Final Forward Pass
hidden_input = np.dot(X, W1) + b1
hidden_output = sigmoid(hidden_input)
final_input = np.dot(hidden_output, W2) + b2
final_output = sigmoid(final_input)

# Display Results
print("\nFinal Results:")
for i in range(len(X)):
    raw_output = final_output[i][0]
    binary_output = 1 if raw_output >= 0.5 else 0
    
    print("Input:", X[i],
          "| Raw Output:", round(raw_output,4),
          "| Binary Output:", binary_output,
          "| Expected:", Y[i][0])

# Plot Loss Graph
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()