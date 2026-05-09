# KRISHNA KUMAR TRIVEDI
# TE-B-25
from sklearn.linear_model import Perceptron
import numpy as np

# Training Data (number % 2)
X = np.array([[0],[1],[0],[1],[0],[1],[0],[1],[0],[1]])
y = np.array([0,1,0,1,0,1,0,1,0,1])

model = Perceptron(max_iter=1000)
model.fit(X, y)

print("Type -1 to Exit")

while True:
    num = int(input("Enter number 0-9: "))

    if num == -1:
        print("Program Ended")
        break

    feature = [[num % 2]]
    prediction = model.predict(feature)

    if prediction[0] == 0:
        print("Even Number")
    else:
        print("Odd Number")
