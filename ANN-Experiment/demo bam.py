# KRISHNA KUMAR TRIVEDI
# TE-B-25
# BAM 

import numpy as np

# Sign Activation Function
def sign(x):
    result = []
    for i in x:
        if i >= 0:
            result.append(1)
        else:
            result.append(-1)
    return np.array(result)

# -------------------------
# Training Pairs
# -------------------------

X1 = np.array([1, -1, 1])
Y1 = np.array([1, 1, -1])

X2 = np.array([-1, 1, -1])
Y2 = np.array([-1, -1, 1])

# -------------------------
# Weight Matrix Calculation
# W = X1^T Y1 + X2^T Y2
# -------------------------

W1 = np.outer(X1, Y1)
W2 = np.outer(X2, Y2)

W = W1 + W2

print("Weight Matrix W (3x3):")
print(W)

# -------------------------
# Recall Y from X
# -------------------------

print("\nRecall Y from X1:")
Y_recall1 = sign(np.dot(X1, W))
print(Y_recall1)

print("\nRecall Y from X2:")
Y_recall2 = sign(np.dot(X2, W))
print(Y_recall2)

# -------------------------
# Recall X from Y
# -------------------------

print("\nRecall X from Y1:")
X_recall1 = sign(np.dot(Y1, W.T))
print(X_recall1)

print("\nRecall X from Y2:")
X_recall2 = sign(np.dot(Y2, W.T))
print(X_recall2)