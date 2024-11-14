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
    ("arg3", c_double),
    ("type", c_int), # TODO: Is there a way to use enum?
]

autodiff.tensor.argtypes = [POINTER(c_double), POINTER(c_size_t), c_size_t]
autodiff.tensor.restype = POINTER(CExpression)
autodiff.exp_sin.argtypes = [POINTER(CExpression)]
autodiff.exp_sin.restype = POINTER(CExpression)
autodiff.exp_relu.argtypes = [POINTER(CExpression)]
autodiff.exp_relu.restype = POINTER(CExpression)
autodiff.exp_sigmoid.argtypes = [POINTER(CExpression)]
autodiff.exp_sigmoid.restype = POINTER(CExpression)
autodiff.exp_log.argtypes = [POINTER(CExpression)]
autodiff.exp_log.restype = POINTER(CExpression)
autodiff.exp_power.argtypes = [POINTER(CExpression), c_double]
autodiff.exp_power.restype = POINTER(CExpression)
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
autodiff.calc.restype = None
autodiff.backward.argtypes = [POINTER(CExpression)]
autodiff.backward.restype = None
autodiff.set_derivative_zero.argtypes = [POINTER(CExpression)]
autodiff.set_derivative_zero.restype = None
autodiff.derivative_step.argtypes = [POINTER(CExpression), c_double]
autodiff.derivative_step.restype = None

# TODO: Might be inefficient due to list copies, fix this.
def multi_dim_list(values: list[int], shape: list[int]):
    if len(shape) == 1:
        return values[:]
    stride = len(values) // shape[0]
    return [multi_dim_list(values[(i*stride):((i+1)*stride)], shape[1:]) for i in range(shape[0])]

def shape(values):
    if len(values) == 0:
        return None
    elif type(values[0]) == float or type(values[0]) == int:
        for elem in values:
            if type(elem) != float and type(elem) != int:
                return None
        return [len(values)]
    else:
        elem_shape = shape(values[0])
        if elem_shape == None:
            return None
        for elem in values[1:]:
            if shape(elem) != elem_shape:
                return None
        return [len(values), *elem_shape]

def flatten(values):
    if type(values[0]) != list:
        return values[:]
    else:
        res = []
        for value in values:
            res += flatten(value)
        return res

def can_broadcast(shape1, shape2):
    for i in range(min(len(shape1), len(shape2))):
        if shape1[-i-1] > 1 and shape2[-i-1] > 1 and shape1[-i-1] != shape2[-i-1]:
            return False
    return True

def can_matmul(shape1, shape2):
    common_dim_1 = shape1[-1]
    common_dim_2 = shape2[-2] if len(shape2) > 1 else shape2[0]
    if common_dim_1 != common_dim_2:
        return False

    if len(shape1) > 2 and len(shape2) > 2:
        if not can_broadcast(shape1[:-2], shape2[:-2]):
            return False

    return True

class Expression():
    def __init__(self, _exp, _dependencies):
        assert _exp, "Null pointer"
        self._exp = _exp
        self._dependencies = _dependencies
    
    def get_shape(self):
        shape_sizes_length = self._exp.contents.shape.sizes_length
        return [self._exp.contents.shape.sizes[i] for i in range(shape_sizes_length)]

    def calc(self) -> list[float]:
        shape = self.get_shape()
        total_length = 1
        for s in shape:
            total_length *= s

        autodiff.calc(self._exp)
        res = multi_dim_list(
            [self._exp.contents.values[i] for i in range(total_length)],
            shape
        )
        return res

    def backward(self):
        autodiff.backward(self._exp)

    def __pow__(self, exponent):
        assert(type(exponent) == int or type(exponent) == float)
        return Expression(autodiff.exp_power(self._exp, float(exponent)), [self])

    def __add__(self, other):
        assert isinstance(other, Expression)
        assert can_broadcast(self.get_shape(), other.get_shape())
        return Expression(autodiff.exp_add(self._exp, other._exp), [self, other])

    def __mul__(self, other):
        assert isinstance(other, Expression)
        assert can_broadcast(self.get_shape(), other.get_shape())
        return Expression(autodiff.exp_mul(self._exp, other._exp), [self, other])
    
    def __matmul__(self, other):
        assert isinstance(other, Expression)
        assert can_matmul(self.get_shape(), other.get_shape())
        return Expression(autodiff.exp_matmul(self._exp, other._exp), [self, other])

    def __sub__(self, other):
        assert isinstance(other, Expression)
        assert can_broadcast(self.get_shape(), other.get_shape())
        return Expression(autodiff.exp_sub(self._exp, other._exp), [self, other])
    
    def __del__(self):
        if self._exp:
            autodiff.free_exp(self._exp)

def sin(exp: Expression):
    return Expression(autodiff.exp_sin(exp._exp), [exp])

def relu(exp: Expression):
    return Expression(autodiff.exp_relu(exp._exp), [exp])

def sigmoid(exp: Expression):
    return Expression(autodiff.exp_sigmoid(exp._exp), [exp])

def log(exp: Expression):
    return Expression(autodiff.exp_log(exp._exp), [exp])

class Tensor(Expression):
    def __init__(self, values):
        values_flat = flatten(values)
        values_shape = shape(values)
        # https://stackoverflow.com/a/3808078
        assert values_shape != None, "values isn't tensor shaped"

        self._exp = autodiff.tensor(
            (c_double * len(values_flat))(*values_flat),
            (c_size_t * len(values_shape))(*values_shape),
            c_size_t(len(values_shape)),
        )
        assert self._exp, "Null pointer"
        self._dependencies = []

    def get_values(self):
        shape = self.get_shape()
        total_length = 1
        for s in shape:
            total_length *= s

        return multi_dim_list(
            [self._exp.contents.values[i] for i in range(total_length)],
            shape
        )

    def set_values(self, values):
        values_flat = flatten(values)
        values_shape = shape(values)
        assert values_shape != None, "values isn't tensor shaped"
        assert(values_shape == self.get_shape())
        for i in range(len(values_flat)):
            self._exp.contents.values[i] = c_double(values_flat[i])

    def get_derivative(self):
        shape = self.get_shape()
        total_length = 1
        for s in shape:
            total_length *= s

        return multi_dim_list(
            [self._exp.contents.derivative[i] for i in range(total_length)],
            shape
        )

    def set_derivative(self, derivative):
        derivative_flat = flatten(derivative)
        derivative_shape = shape(derivative)
        assert derivative_shape != None
        assert(derivative_shape == self.get_shape())
        for i in range(len(derivative_flat)):
            self._exp.contents.derivative[i] = c_double(derivative_flat[i])
    
    def set_derivative_zero(self):
        autodiff.set_derivative_zero(self._exp)
    
    def derivative_step(self, multiplier):
        autodiff.derivative_step(self._exp, multiplier)