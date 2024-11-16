# To use this library, you need to first compile autodiff.c.
# Use this on Linux/WSL:
# gcc autodiff.c -lm -fPIC -shared -o autodiff_lib.so

from autodiff import Tensor
import random

actual = [20 * random.random() - 10, 20 * random.random() - 10, 20 * random.random() - 10]
x = [[20 * random.random() - 10, 20 * random.random() - 10] for _ in range(100)]
y = [x[i][0] * actual[0] + x[i][1] * actual[1] + actual[2] for i in range(100)]
x = Tensor(x)
y = Tensor(y)
m = Tensor([0.0, 0.0])
c = Tensor([0.0])
loss = (y - x @ m - c) ** 2 * Tensor([1 / 100])

for _ in range(1000):
    print('Weights:', [m.get_values(), c.get_values()])
    print('loss[0]:', loss.calc()[0]) # Loss has shape [100].
    m.set_derivative_zero()
    c.set_derivative_zero()
    loss.backward()
    m.derivative_step(-0.01)
    c.derivative_step(-0.01)
    print()

print("Actual: ", actual)