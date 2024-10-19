from autodiff import Variable
import random

def clamp(x, a, b):
    if x < a:
        return a
    elif b < x:
        return b
    else:
        return x

actual = [20 * random.random() - 10, 20 * random.random() - 10, 20 * random.random() - 10]
x = [[20 * random.random() - 10, 20 * random.random() - 10] for _ in range(100)]
y = [x[i][0] * actual[0] + x[i][1] * actual[1] + actual[2] for i in range(100)]
x = Variable(x)
y = Variable(y)
m = Variable([0.0, 0.0])
c = Variable([0.0])
loss = (y - x @ m - c) * (y - x @ m - c)

for _ in range(1000):
    print('Weights', [m.get_values(), c.get_values()])
    m.set_derivative([0.0] * len(m.get_values()))
    c.set_derivative([0.0])
    loss.backward()

    m_values = m.get_values()
    m_derivative = [clamp(m.get_derivative()[i], -10.0, 10.0) for i in range(len(m_values))]
    m.set_values([m_values[i] - 0.001 * m_derivative[i] for i in range(len(m_values))])

    c_value = c.get_values()
    c_derivative = [clamp(c.get_derivative()[0], -10.0, 10.0)]
    c.set_values([c_value[0] - 0.001 * c_derivative[0]])

print("Actual: ", actual)