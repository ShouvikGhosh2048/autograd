#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// https://stackoverflow.com/a/19188730
// https://stackoverflow.com/a/19188749

typedef enum {
    VAR,
    SIN,
    ADD,
    MUL,
    SUB,
} expression_type;

typedef struct {
    size_t *sizes;
    size_t sizes_length;
} shape;

size_t shape_total_length(shape shape) {
    size_t length = 1;
    for (size_t i = 0; i < shape.sizes_length; i++) {
        length *= shape.sizes[i];
    }
    return length;
}

size_t max_size(size_t a, size_t b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

// Returns a new broadcast shape, sizes will be NULL if not possible.
shape broadcast_shape(shape shape1, shape shape2) {
    size_t i = shape1.sizes_length - 1;
    size_t j = shape2.sizes_length - 1;

    shape res = { NULL };
    while(1) {
        if (shape1.sizes[i] != shape2.sizes[j] && shape1.sizes[i] != 1 && shape2.sizes[j] != 1) {
            return res;
        } 
        if (i == 0 || j == 0) {
            break;
        }
        i--;
        j--;
    }

    res.sizes_length = max_size(shape1.sizes_length, shape2.sizes_length);
    res.sizes = malloc(sizeof(size_t) * res.sizes_length);
    for (size_t i = 0; i < res.sizes_length; i++) {
        res.sizes[res.sizes_length - 1 - i] = 1;
        if (i < shape1.sizes_length) {
            res.sizes[res.sizes_length - 1 - i] = max_size(res.sizes[res.sizes_length - 1 - i], shape1.sizes[shape1.sizes_length - 1 - i]);
        }
        if (i < shape2.sizes_length) {
            res.sizes[res.sizes_length - 1 - i] = max_size(res.sizes[res.sizes_length - 1 - i], shape2.sizes[shape2.sizes_length - 1 - i]);
        }
    }
    return res;
}

typedef struct expression {
    double *values;
    double *derivative;
    shape shape;
    struct expression *arg1;
    struct expression *arg2;
    expression_type type;
} expression;

expression* var(double *values, size_t *sizes, size_t sizes_length) {
    if (sizes_length == 0) {
        return NULL;
    }

    expression* var = malloc(sizeof(expression));

    size_t length = 1;
    for (size_t i = 0; i < sizes_length; i++) {
        length *= sizes[i];
    }
    // TODO: Maybe memcpy?
    var -> values = malloc(sizeof(double) * length);
    for (size_t i = 0; i < length; i++) {
        (var -> values)[i] = values[i];
    }
    // TODO: Does calloc initialize double to 0.0?
    var -> derivative = malloc(sizeof(double) * length);
    for (size_t i = 0; i < length; i++) {
        (var -> derivative)[i] = 0.0;
    }
    var -> shape.sizes_length = sizes_length;
    var -> shape.sizes = malloc(sizeof(size_t) * sizes_length);
    for (size_t i = 0; i < sizes_length; i++) {
        var -> shape.sizes[i] = sizes[i];
    }
    var -> arg1 = NULL;
    var -> arg2 = NULL;
    var -> type = VAR;
    return var;
}

expression* exp_sin(expression* arg) {
    expression* exp = malloc(sizeof(expression));
    exp -> values = NULL;
    exp -> derivative = NULL;
    exp -> shape.sizes_length = arg -> shape.sizes_length;
    exp -> shape.sizes = malloc(sizeof(size_t) * arg -> shape.sizes_length);
    for (size_t i = 0; i < arg -> shape.sizes_length; i++) {
        exp -> shape.sizes[i] = arg -> shape.sizes[i];
    }
    exp -> arg1 = arg;
    exp -> arg2 = NULL;
    exp -> type = SIN;
    return exp;
}

expression* exp_add(expression* arg1, expression* arg2) {
    // TODO: Should I add an assertion for equal lengths?
    shape exp_shape = broadcast_shape(arg1 -> shape, arg2 -> shape);
    if (exp_shape.sizes == NULL) {
        return NULL;
    }

    expression* exp = malloc(sizeof(expression));
    exp -> values = NULL;
    exp -> derivative = NULL;
    exp -> shape = exp_shape;
    exp -> arg1 = arg1;
    exp -> arg2 = arg2;
    exp -> type = ADD;
    return exp;
}

expression* exp_mul(expression* arg1, expression* arg2) {
    shape exp_shape = broadcast_shape(arg1 -> shape, arg2 -> shape);
    if (exp_shape.sizes == NULL) {
        return NULL;
    }

    expression* exp = malloc(sizeof(expression));
    exp -> values = NULL;
    exp -> derivative = NULL;
    exp -> shape = exp_shape;
    exp -> arg1 = arg1;
    exp -> arg2 = arg2;
    exp -> type = MUL;
    return exp;
}

expression* exp_sub(expression* arg1, expression* arg2) {
    shape exp_shape = broadcast_shape(arg1 -> shape, arg2 -> shape);
    if (exp_shape.sizes == NULL) {
        return NULL;
    }

    expression* exp = malloc(sizeof(expression));
    exp -> values = NULL;
    exp -> derivative = NULL;
    exp -> shape = exp_shape;
    exp -> arg1 = arg1;
    exp -> arg2 = arg2;
    exp -> type = SUB;
    return exp;
}

void free_exp(expression* exp) {
    if (exp -> type == VAR) {
        free(exp -> values);
        free(exp -> derivative);
    }
    free(exp -> shape.sizes);
    free(exp);
}

size_t broadcast_index(size_t index, shape larger, shape smaller) {
    size_t res = 0;
    size_t stride = 1;
    for (size_t j = 0; j < smaller.sizes_length; j++) {
        size_t res_position_index = index % larger.sizes[larger.sizes_length - 1 - j];
        size_t position_index = (smaller.sizes[smaller.sizes_length - 1 - j] == 1) ? 0 : res_position_index;
        res += position_index * stride;
        stride *= smaller.sizes[smaller.sizes_length - 1 - j];
        index /= larger.sizes[larger.sizes_length - 1 - j];
    }
    return res;
}

double* calc(expression* exp) {
    size_t length = 1;
    for (size_t i = 0; i < exp -> shape.sizes_length; i++) {
        length *= exp -> shape.sizes[i];
    }

    switch (exp -> type) {
        case VAR: {
            double *res = malloc(sizeof(double) * length);
            for (size_t i = 0; i < length; i++) {
                res[i] = (exp -> values)[i];
            }
            return res;
        }
        case SIN: {
            double *res = calc(exp -> arg1);
            for (size_t i = 0; i < length; i++) {
                res[i] = sin(res[i]);
            }
            return res;
        }
        case ADD: {
            double *arg1 = calc(exp -> arg1);
            double *arg2 = calc(exp -> arg2);
            double *res = malloc(sizeof(double) * length);
            for (size_t i = 0; i < length; i++) {
                size_t arg1_index = broadcast_index(i, exp -> shape, exp -> arg1 -> shape);
                size_t arg2_index = broadcast_index(i, exp -> shape, exp -> arg2 -> shape);
                res[i] = arg1[arg1_index] + arg2[arg2_index];
            }
            free(arg1);
            free(arg2);
            return res;
        }
        case MUL: {
            double *arg1 = calc(exp -> arg1);
            double *arg2 = calc(exp -> arg2);
            double *res = malloc(sizeof(double) * length);
            for (size_t i = 0; i < length; i++) {
                size_t arg1_index = broadcast_index(i, exp -> shape, exp -> arg1 -> shape);
                size_t arg2_index = broadcast_index(i, exp -> shape, exp -> arg2 -> shape);
                res[i] = arg1[arg1_index] * arg2[arg2_index];
            }
            free(arg1);
            free(arg2);
            return res;
        }
        case SUB: {
            double *arg1 = calc(exp -> arg1);
            double *arg2 = calc(exp -> arg2);
            double *res = malloc(sizeof(double) * length);
            for (size_t i = 0; i < length; i++) {
                size_t arg1_index = broadcast_index(i, exp -> shape, exp -> arg1 -> shape);
                size_t arg2_index = broadcast_index(i, exp -> shape, exp -> arg2 -> shape);
                res[i] = arg1[arg1_index] - arg2[arg2_index];
            }
            free(arg1);
            free(arg2);
            return res;
        }
    }
}

void free_pointer(void* p) {
    free(p);
}

void backward_recursive(expression* exp, double* mult) {
    size_t length = 1;
    for (size_t i = 0; i < exp -> shape.sizes_length; i++) {
        length *= exp -> shape.sizes[i];
    }

    switch (exp -> type) {
        case VAR: {
            for (size_t i = 0; i < length; i++) {
                exp -> derivative[i] += mult[i];
            }
            break;
        }
        case SIN: {
            double *arg1 = calc(exp -> arg1);
            for (size_t i = 0; i < length; i++) {
                mult[i] *= cos(arg1[i]);
            }
            free(arg1);
            backward_recursive(exp -> arg1, mult);
            break;
        }
        case ADD: {
            double *arg1 = calc(exp -> arg1);
            size_t arg1_length = shape_total_length(exp -> arg1 -> shape);
            double *arg1_mult = malloc(sizeof(double) * arg1_length);
            for (size_t i = 0; i < arg1_length; i++) {
                arg1_mult[i] = 0.0;
            }
            double *arg2 = calc(exp -> arg2);
            size_t arg2_length = shape_total_length(exp -> arg2 -> shape);
            double *arg2_mult = malloc(sizeof(double) * arg2_length);
            for (size_t i = 0; i < arg2_length; i++) {
                arg2_mult[i] = 0.0;
            }
            for (size_t i = 0; i < length; i++) {
                size_t arg1_index = broadcast_index(i, exp -> shape, exp -> arg1 -> shape);
                size_t arg2_index = broadcast_index(i, exp -> shape, exp -> arg2 -> shape);
                arg1_mult[arg1_index] += mult[i];
                arg2_mult[arg2_index] += mult[i];
            }
            free(arg1);
            free(arg2);

            backward_recursive(exp -> arg1, arg1_mult);
            backward_recursive(exp -> arg2, arg2_mult);
            break;
        }
        case MUL: {
            double *arg1 = calc(exp -> arg1);
            size_t arg1_length = shape_total_length(exp -> arg1 -> shape);
            double *arg1_mult = malloc(sizeof(double) * arg1_length);
            for (size_t i = 0; i < arg1_length; i++) {
                arg1_mult[i] = 0.0;
            }
            double *arg2 = calc(exp -> arg2);
            size_t arg2_length = shape_total_length(exp -> arg2 -> shape);
            double *arg2_mult = malloc(sizeof(double) * arg2_length);
            for (size_t i = 0; i < arg2_length; i++) {
                arg2_mult[i] = 0.0;
            }
            for (size_t i = 0; i < length; i++) {
                size_t arg1_index = broadcast_index(i, exp -> shape, exp -> arg1 -> shape);
                size_t arg2_index = broadcast_index(i, exp -> shape, exp -> arg2 -> shape);
                arg1_mult[arg1_index] += mult[i] * arg2[arg2_index];
                arg2_mult[arg2_index] += mult[i] * arg1[arg1_index];
            }
            free(arg1);
            free(arg2);

            backward_recursive(exp -> arg1, arg1_mult);
            backward_recursive(exp -> arg2, arg2_mult);
            break;
        }
        case SUB: {
            double *arg1 = calc(exp -> arg1);
            size_t arg1_length = shape_total_length(exp -> arg1 -> shape);
            double *arg1_mult = malloc(sizeof(double) * arg1_length);
            for (size_t i = 0; i < arg1_length; i++) {
                arg1_mult[i] = 0.0;
            }
            double *arg2 = calc(exp -> arg2);
            size_t arg2_length = shape_total_length(exp -> arg2 -> shape);
            double *arg2_mult = malloc(sizeof(double) * arg2_length);
            for (size_t i = 0; i < arg2_length; i++) {
                arg2_mult[i] = 0.0;
            }
            for (size_t i = 0; i < length; i++) {
                size_t arg1_index = broadcast_index(i, exp -> shape, exp -> arg1 -> shape);
                size_t arg2_index = broadcast_index(i, exp -> shape, exp -> arg2 -> shape);
                arg1_mult[arg1_index] += mult[i];
                arg2_mult[arg2_index] -= mult[i];
            }
            free(arg1);
            free(arg2);

            backward_recursive(exp -> arg1, arg1_mult);
            backward_recursive(exp -> arg2, arg2_mult);
            break;
        }
    }
}

void backward(expression* exp) {
    size_t length = 1;
    for (size_t i = 0; i < exp -> shape.sizes_length; i++) {
        length *= exp -> shape.sizes[i];
    }

    double *mult = malloc(sizeof(double) * length);
    for (size_t i = 0; i < length; i++) {
        mult[i] = 1.0;
    }
    backward_recursive(exp, mult);
    free(mult);
}