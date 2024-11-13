import random
from autodiff import Tensor, relu, sigmoid, log, flatten

def random_circle():
    img = [[1.0 for _ in range(28)] for _ in range(28)]
    cx = 28 * random.random()
    cy = 28 * random.random()
    r = 5 + 10 * random.random()
    for i in range(28):
        for j in range(28):
            if (i - cx) ** 2 + (j - cy) ** 2 < r**2:
                img[j][i] = 0.0
    return img

def random_plus():
    img = [[1.0 for _ in range(28)] for _ in range(28)]
    a = random.randint(1, 26)
    b = random.randint(1, 26)
    for i in range(random.randint(0, b-1), random.randint(b+2, 28)):
        img[a-1][i] = 0.0
        img[a][i] = 0.0
        img[a+1][i] = 0.0
    for i in range(random.randint(0, a-1), random.randint(a+2, 28)):
        img[i][b-1] = 0.0
        img[i][b] = 0.0
        img[i][b+1] = 0.0
    return img

N = 10000
BATCHES = 100
train_x = [Tensor(
    [flatten(random_circle()) for _ in range(N // (2 * BATCHES))]
    + [flatten(random_plus()) for _ in range(N // (2 * BATCHES))]
) for _ in range(BATCHES)]
train_y = Tensor([0.0] * (N // (2 * BATCHES)) + [1.0] * (N // (2 * BATCHES)))
test_x = Tensor(
    [flatten(random_circle()) for _ in range(N // 2)]
    + [flatten(random_plus()) for _ in range(N // 2)]
)
test_y = Tensor([0.0] * (N // 2) + [1.0] * (N // 2))

hidden_size = 10
w1 = Tensor([
    [0.1 * (2 * random.random() - 1) for _ in range(hidden_size)]
    for _ in range(784)
])
b1 = Tensor([
    0.1 * (2 * random.random() - 1) for _ in range(hidden_size)
])
w2 = Tensor([0.1 * (2 * random.random() - 1) for _ in range(hidden_size)])
b2 = Tensor([0.1 * (2 * random.random() - 1)])

for i in range(20):
    print("Epoch", i)
    for batch in train_x:
        w1.set_derivative_zero()
        b1.set_derivative_zero()
        w2.set_derivative_zero()
        b2.set_derivative_zero()

        pred = sigmoid(relu(batch @ w1 + b1) @ w2 + b2)
        loss = (train_y * log(pred) + (Tensor([1.0]) - train_y) * log(Tensor([1.0]) - pred)) * Tensor([-1 / (N // BATCHES)])
        loss.backward()

        mult = -0.01
        w1.derivative_step(mult)
        b1.derivative_step(mult)
        w2.derivative_step(mult)
        b2.derivative_step(mult)
    
    correct = 0
    for batch in train_x:
        pred = sigmoid(relu(batch @ w1 + b1) @ w2 + b2)
        correct += sum([1 for (label, pred) in zip(train_y.calc(), pred.calc()) if (label - 0.5) * (pred - 0.5) > 0.0])
    print("Accuracy", correct / N)
    print()

pred = sigmoid(relu(test_x @ w1 + b1) @ w2 + b2)
correct = sum([1 for (label, pred) in zip(test_y.calc(), pred.calc()) if (label - 0.5) * (pred - 0.5) > 0.0])
print("Accuracy", correct / N)