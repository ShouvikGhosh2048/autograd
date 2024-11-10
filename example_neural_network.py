import random
from autodiff import Variable, relu, sigmoid, log
import copy
from math import sqrt

def random_circle():
    img = [[1.0 for _ in range(28)] for _ in range(28)]

    cx = 28 * random.random()
    cy = 28 * random.random()
    r = 5 + 5 * random.random()
    for i in range(28):
        for j in range(28):
            if (i - cx) ** 2 + (j - cy) ** 2 < r**2:
                img[j][i] = 0.0
    return img

def random_plus():
    img = [[1.0 for _ in range(28)] for _ in range(28)]

    a = random.randint(1, 26)
    b = random.randint(1, 26)
    for i in range(28):
        img[a-1][i] = 0.0
        img[a][i] = 0.0
        img[a+1][i] = 0.0
        img[i][b-1] = 0.0
        img[i][b] = 0.0
        img[i][b+1] = 0.0
    return img

def flatten(img):
    res = []
    for row in img:
        for val in row:
            res.append(val)
    return res

N = 10000
BATCHES = 100
train_x = [Variable(
    [[flatten(random_circle())] for _ in range(N // (2 * BATCHES))]
    + [[flatten(random_plus())] for _ in range(N // (2 * BATCHES))]
) for _ in range(BATCHES)]
train_y = Variable([[0.0]] * (N // (2 * BATCHES)) + [[1.0]] * (N // (2 * BATCHES)))
test_x = Variable(
    [[flatten(random_circle())] for _ in range(N // 2)]
    + [[flatten(random_plus())] for _ in range(N // 2)]
)
test_y = Variable([[0.0]] * (N // 2) + [[1.0]] * (N // 2))

hidden_size = 50
w1 = Variable([
    [0.1 * (2 * random.random() - 1) for _ in range(hidden_size)]
    for _ in range(784)
])
b1 = Variable([
    0.1 * (2 * random.random() - 1) for _ in range(hidden_size)
])
w2 = Variable([0.1 * (2 * random.random() - 1) for _ in range(hidden_size)])
b2 = Variable([0.1 * (2 * random.random() - 1)])

def sum_of_squares(val):
    res = 0
    for elem in val:
        if type(elem) == list:
            res += sum_of_squares(elem)
        else:
            res += elem ** 2
    return res

def divide(val, c):
    for i in range(len(val)):
        if type(val[i]) == list:
            divide(val[i], c)
        else:
            val[i] /= c

def normalize(val):
    val = copy.deepcopy(val)
    divide(val, sqrt(sum_of_squares(val)))
    return val

def clamp(x, a, b):
    if x < a:
        return a
    elif b < x:
        return b
    else:
        return x

for i in range(10):
    print("Epoch", i)
    for batch in train_x:
        w1.set_derivative([[0.0 for _ in range(hidden_size)] for _ in range(784)])
        b1.set_derivative([0.0 for _ in range(hidden_size)])
        w2.set_derivative([0.0 for _ in range(hidden_size)])
        b2.set_derivative([0.0])

        pred = sigmoid(relu(batch @ w1 + b1) @ w2 + b2)
        loss = (train_y * log(pred) + (Variable([1.0]) - train_y) * log(Variable([1.0]) - pred)) * Variable([-1 / (N // BATCHES)])
        loss.backward()

        mult = 0.01
        values = w1.get_values()
        derivative = w1.get_derivative()
        w1.set_values([[values[i][j] - mult * derivative[i][j] for j in range(hidden_size)] for i in range(784)])
        values = b1.get_values()
        derivative = b1.get_derivative()
        b1.set_values([values[i] - mult * derivative[i] for i in range(hidden_size)])
        values = w2.get_values()
        derivative = w2.get_derivative()
        w2.set_values([values[i] - mult * derivative[i] for i in range(hidden_size)])
        values = b2.get_values()
        derivative = b2.get_derivative()
        b2.set_values([values[i] - mult * derivative[i] for i in range(1)])
    
    correct = 0
    for batch in train_x:
        pred = sigmoid(relu(batch @ w1 + b1) @ w2 + b2)
        correct += sum([1 for (label, pred) in zip(train_y.calc(), pred.calc()) if (label[0] - 0.5) * (pred[0] - 0.5) > 0.0])
    print("Accuracy", correct / N)
    print()

pred = sigmoid(relu(test_x @ w1 + b1) @ w2 + b2)
correct = sum([1 for (label, pred) in zip(test_y.calc(), pred.calc()) if (label[0] - 0.5) * (pred[0] - 0.5) > 0.0])
print("Accuracy", correct / N)