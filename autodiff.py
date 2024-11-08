# https://www.digitalocean.com/community/tutorials/calling-c-functions-from-python
# https://docs.python.org/3/library/ctypes.html

# gcc autodiff.c -lm -fPIC -shared -o autodiff_lib.so
# https://stackoverflow.com/a/65356749
# https://stackoverflow.com/a/13452473

from ctypes import *

autodiff = CDLL("./autodiff_lib.so")

class Shape(Structure):
    _fields_ = [
        ("sizes", POINTER(c_size_t)),
        ("sizes_length", c_size_t),
    ]

class CExpression(Structure):
    pass
CExpression._fields_ = [
    ("values", POINTER(c_double)),
    ("derivative", POINTER(c_double)),
    ("shape", Shape),
    ("arg1", POINTER(CExpression)),
    ("arg2", POINTER(CExpression)),
    ("type", c_int), # TODO: Is there a way to use enum?
]

autodiff.var.argtypes = [POINTER(c_double), POINTER(c_size_t), c_size_t]
autodiff.var.restype = POINTER(CExpression)
autodiff.exp_sin.argtypes = [POINTER(CExpression)]
autodiff.exp_sin.restype = POINTER(CExpression)
autodiff.exp_relu.argtypes = [POINTER(CExpression)]
autodiff.exp_relu.restype = POINTER(CExpression)
autodiff.exp_add.argtypes = [POINTER(CExpression), POINTER(CExpression)]
autodiff.exp_add.restype = POINTER(CExpression)
autodiff.exp_mul.argtypes = [POINTER(CExpression), POINTER(CExpression)]
autodiff.exp_mul.restype = POINTER(CExpression)
autodiff.exp_matmul.argtypes = [POINTER(CExpression), POINTER(CExpression)]
autodiff.exp_matmul.restype = POINTER(CExpression)
autodiff.exp_sub.argtypes = [POINTER(CExpression), POINTER(CExpression)]
autodiff.exp_sub.restype = POINTER(CExpression)
autodiff.free_exp.argtypes = [POINTER(CExpression)]
autodiff.free_exp.restype = None
autodiff.calc.argtypes = [POINTER(CExpression)]
autodiff.calc.restype = POINTER(c_double)
autodiff.free_pointer.argtypes = [c_void_p]
autodiff.free_pointer.restype = None
autodiff.backward.argtypes = [POINTER(CExpression)]
autodiff.backward.restype = None

# TODO: Might be inefficient due to list copies, fix this.
def multi_dim_list(values: list[int], shape: list[int]):
    if len(shape) == 1:
        return values[:]
    stride = len(values) // shape[0]
    return [multi_dim_list(values[(i*stride):((i+1)*stride)], shape[1:]) for i in range(shape[0])]

def flatten(values):
    if type(values[0]) != list:
        return values[:]
    else:
        res = []
        for value in values:
            res += flatten(value)
        return res

class Expression():
    def __init__(self, _exp, _dependencies):
        self._exp = _exp
        self._dependencies = _dependencies

    def calc(self) -> list[float]:
        shape_sizes_length = self._exp.contents.shape.sizes_length
        shape = [self._exp.contents.shape.sizes[i] for i in range(shape_sizes_length)]
        total_length = 1
        for s in shape:
            total_length *= s

        calc_res = autodiff.calc(self._exp)
        res = multi_dim_list(
            [calc_res[i] for i in range(total_length)],
            shape
        )
        autodiff.free_pointer(calc_res)
        return res

    def backward(self):
        autodiff.backward(self._exp)

    def __add__(self, other):
        assert isinstance(other, Expression)
        return Expression(autodiff.exp_add(self._exp, other._exp), [self, other])

    def __mul__(self, other):
        assert isinstance(other, Expression)
        return Expression(autodiff.exp_mul(self._exp, other._exp), [self, other])
    
    def __matmul__(self, other):
        assert isinstance(other, Expression)
        return Expression(autodiff.exp_matmul(self._exp, other._exp), [self, other])

    def __sub__(self, other):
        assert isinstance(other, Expression)
        return Expression(autodiff.exp_sub(self._exp, other._exp), [self, other])
    
    def __del__(self):
        autodiff.free_exp(self._exp)

def sin(exp: Expression):
    return Expression(autodiff.exp_sin(exp._exp), [exp])

def relu(exp: Expression):
    return Expression(autodiff.exp_relu(exp._exp), [exp])

class Variable(Expression):
    def __init__(self, values):
        values_flat = flatten(values)
        shape = []
        a = values
        while type(a) == list:
            shape.append(len(a))
            a = a[0]

        self._exp = autodiff.var(
            (c_double * len(values_flat))(*values_flat),
            (c_size_t * len(shape))(*shape),
            c_size_t(len(shape)),
        )
        self._dependencies = []

    def get_values(self):
        shape_sizes_length = self._exp.contents.shape.sizes_length
        shape = [self._exp.contents.shape.sizes[i] for i in range(shape_sizes_length)]
        total_length = 1
        for s in shape:
            total_length *= s

        return multi_dim_list(
            [self._exp.contents.values[i] for i in range(total_length)],
            shape
        )

    def set_values(self, values):
        values_flat = flatten(values)
        values_shape = []
        a = values
        while type(a) == list:
            values_shape.append(len(a))
            a = a[0]

        exp_sizes_length = self._exp.contents.shape.sizes_length
        exp_shape = [self._exp.contents.shape.sizes[i] for i in range(exp_sizes_length)]

        assert(values_shape == exp_shape)
        for i in range(len(values_flat)):
            self._exp.contents.values[i] = c_double(values_flat[i])

    def get_derivative(self):
        shape_sizes_length = self._exp.contents.shape.sizes_length
        shape = [self._exp.contents.shape.sizes[i] for i in range(shape_sizes_length)]
        total_length = 1
        for s in shape:
            total_length *= s

        return multi_dim_list(
            [self._exp.contents.derivative[i] for i in range(total_length)],
            shape
        )

    def set_derivative(self, derivative):
        derivative_flat = flatten(derivative)
        derivative_shape = []
        a = derivative
        while type(a) == list:
            derivative_shape.append(len(a))
            a = a[0]

        exp_sizes_length = self._exp.contents.shape.sizes_length
        exp_shape = [self._exp.contents.shape.sizes[i] for i in range(exp_sizes_length)]

        assert(derivative_shape == exp_shape)
        for i in range(len(derivative_flat)):
            self._exp.contents.derivative[i] = c_double(derivative_flat[i])