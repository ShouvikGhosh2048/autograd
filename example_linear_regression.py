from autodiff import Variable
import random

def clamp(x, a, b):
    if x < a:
        return a
    elif b < x:
        return b
    else:
        return x

x1 = [20 * random.random() - 10 for _ in range(100)]
x2 = [20 * random.random() - 10 for _ in range(100)]
y = [x1[i] * 10 + x2[i] * 5 - 3 for i in range(100)]
x1 = Variable(x1)
x2 = Variable(x2)
y = Variable(y)
b1 = Variable([0.0])
b2 = Variable([0.0])
b3 = Variable([0.0])
vars = [b1, b2, b3]
loss = (y - b1 * x1 - b2 * x2 - b3) * (y - b1 * x1 - b2 * x2 - b3)

for _ in range(1000):
    print('Weights', [var.get_values()[0] for var in vars])
    for var in vars:
        var.set_derivative([0.0])
    loss.backward()
    for var in vars:
        val = var.get_values()[0]
        derivative = clamp(var.get_derivative()[0], -10.0, 10.0)
        var.set_values([val - 0.001 * derivative])