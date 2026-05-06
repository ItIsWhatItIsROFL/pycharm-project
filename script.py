import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

data = [
    (0,0,0),
    (0,1,0),
    (1,0,0),
    (1,1,1)
]

w1 = 5
w2 = 2
bias = -10
learning_rate = 0.1

for iteration in range(100):

    for x1, x2, target in data:

        output = sigmoid(w1 * x1 + w2 * x2 + bias)
        error = target - output

        w1 = w1 + (x1 * error * learning_rate)
        w2 = w2 + (x2 * error * learning_rate)
        bias = bias + (error * learning_rate)

        print(output, error, w1, w2, bias)
