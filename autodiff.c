#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// https://stackoverflow.com/a/19188730
// https://stackoverflow.com/a/19188749

typedef enum {
    VAR,
    SIN,
    RELU,
    SIGMOID,
    LOG,
    ADD,
    MUL,
    MATMUL,
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

shape matmul_shape(shape shape1, shape shape2) {
    shape res = { NULL };
    if (shape1.sizes_length == 1) {
        if (shape2.sizes_length == 1) {
            if (shape1.sizes[0] != shape2.sizes[0]) {
                return res;
            }
            else {
                res.sizes_length = 1;
                res.sizes = malloc(sizeof(size_t) * 1);
                res.sizes[0] = 1;
                return res;
            }
        } else {
            if (shape1.sizes[0] != shape2.sizes[shape2.sizes_length - 2]) {
                return res;
            } else {
                res.sizes_length = shape2.sizes_length - 1;
                res.sizes = malloc(sizeof(size_t) * res.sizes_length);
                for (size_t i = 0; i < res.sizes_length - 1; i++) {
                    res.sizes[i] = shape2.sizes[i];
                }
                res.sizes[res.sizes_length - 1] = shape2.sizes[shape2.sizes_length - 1];
                return res;
            }
        }
    }
    else if (shape2.sizes_length == 1) {
        if (shape2.sizes[0] != shape1.sizes[shape1.sizes_length - 1]) {
            return res;
        } else {
            res.sizes_length = shape1.sizes_length - 1;
            res.sizes = malloc(sizeof(size_t) * res.sizes_length);
            for (size_t i = 0; i < res.sizes_length; i++) {
                res.sizes[i] = shape1.sizes[i];
            }
            return res;
        }
    }

    if (shape1.sizes[shape1.sizes_length - 1] != shape2.sizes[shape2.sizes_length - 2]) {
        return res;
    }

    if (shape1.sizes_length > 2 && shape2.sizes_length > 2) {
        size_t i = shape1.sizes_length - 3;
        size_t j = shape2.sizes_length - 3;
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
    }

    res.sizes_length = max_size(shape1.sizes_length, shape2.sizes_length);
    res.sizes = malloc(sizeof(size_t) * res.sizes_length);
    res.sizes[res.sizes_length - 2] = shape1.sizes[shape1.sizes_length - 2];
    res.sizes[res.sizes_length - 1] = shape2.sizes[shape2.sizes_length - 1];
    for (size_t i = 2; i < res.sizes_length; i++) {
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
    if (values) {
        for (size_t i = 0; i < length; i++) {
            (var -> values)[i] = values[i];
        }
    } else {
        for (size_t i = 0; i < length; i++) {
            (var -> values)[i] = 0.0;
        }
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

    size_t length = 1;
    for (size_t i = 0; i < arg -> shape.sizes_length; i++) {
        length *= arg -> shape.sizes[i];
    }

    exp -> values = malloc(sizeof(double) * length);
    for (size_t i = 0; i < length; i++) {
        exp -> values[i] = 0.0;
    }

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

expression* exp_relu(expression* arg) {
    expression* exp = malloc(sizeof(expression));

    size_t length = 1;
    for (size_t i = 0; i < arg -> shape.sizes_length; i++) {
        length *= arg -> shape.sizes[i];
    }

    exp -> values = malloc(sizeof(double) * length);
    for (size_t i = 0; i < length; i++) {
        exp -> values[i] = 0.0;
    }

    exp -> derivative = NULL;
    exp -> shape.sizes_length = arg -> shape.sizes_length;
    exp -> shape.sizes = malloc(sizeof(size_t) * arg -> shape.sizes_length);
    for (size_t i = 0; i < arg -> shape.sizes_length; i++) {
        exp -> shape.sizes[i] = arg -> shape.sizes[i];
    }
    exp -> arg1 = arg;
    exp -> arg2 = NULL;
    exp -> type = RELU;
    return exp;
}

expression* exp_sigmoid(expression* arg) {
    expression* exp = malloc(sizeof(expression));

    size_t length = 1;
    for (size_t i = 0; i < arg -> shape.sizes_length; i++) {
        length *= arg -> shape.sizes[i];
    }

    exp -> values = malloc(sizeof(double) * length);
    for (size_t i = 0; i < length; i++) {
        exp -> values[i] = 0.0;
    }

    exp -> derivative = NULL;
    exp -> shape.sizes_length = arg -> shape.sizes_length;
    exp -> shape.sizes = malloc(sizeof(size_t) * arg -> shape.sizes_length);
    for (size_t i = 0; i < arg -> shape.sizes_length; i++) {
        exp -> shape.sizes[i] = arg -> shape.sizes[i];
    }
    exp -> arg1 = arg;
    exp -> arg2 = NULL;
    exp -> type = SIGMOID;
    return exp;
}

expression* exp_log(expression* arg) {
    expression* exp = malloc(sizeof(expression));

    size_t length = 1;
    for (size_t i = 0; i < arg -> shape.sizes_length; i++) {
        length *= arg -> shape.sizes[i];
    }

    exp -> values = malloc(sizeof(double) * length);
    for (size_t i = 0; i < length; i++) {
        exp -> values[i] = 0.0;
    }

    exp -> derivative = NULL;
    exp -> shape.sizes_length = arg -> shape.sizes_length;
    exp -> shape.sizes = malloc(sizeof(size_t) * arg -> shape.sizes_length);
    for (size_t i = 0; i < arg -> shape.sizes_length; i++) {
        exp -> shape.sizes[i] = arg -> shape.sizes[i];
    }
    exp -> arg1 = arg;
    exp -> arg2 = NULL;
    exp -> type = LOG;
    return exp;
}

expression* exp_add(expression* arg1, expression* arg2) {
    // TODO: Should I add an assertion for equal lengths?
    shape exp_shape = broadcast_shape(arg1 -> shape, arg2 -> shape);
    if (exp_shape.sizes == NULL) {
        return NULL;
    }

    expression* exp = malloc(sizeof(expression));

    size_t length = 1;
    for (size_t i = 0; i < exp_shape.sizes_length; i++) {
        length *= exp_shape.sizes[i];
    }
    exp -> values = malloc(sizeof(double) * length);
    for (size_t i = 0; i < length; i++) {
        exp -> values[i] = 0.0;
    }

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

    size_t length = 1;
    for (size_t i = 0; i < exp_shape.sizes_length; i++) {
        length *= exp_shape.sizes[i];
    }
    exp -> values = malloc(sizeof(double) * length);
    for (size_t i = 0; i < length; i++) {
        exp -> values[i] = 0.0;
    }

    exp -> derivative = NULL;
    exp -> shape = exp_shape;
    exp -> arg1 = arg1;
    exp -> arg2 = arg2;
    exp -> type = MUL;
    return exp;
}

expression* exp_matmul(expression* arg1, expression* arg2) {
    shape exp_shape = matmul_shape(arg1 -> shape, arg2 -> shape);
    if (exp_shape.sizes == NULL) {
        return NULL;
    }

    expression* exp = malloc(sizeof(expression));

    size_t length = 1;
    for (size_t i = 0; i < exp_shape.sizes_length; i++) {
        length *= exp_shape.sizes[i];
    }
    exp -> values = malloc(sizeof(double) * length);
    for (size_t i = 0; i < length; i++) {
        exp -> values[i] = 0.0;
    }

    exp -> derivative = NULL;
    exp -> shape = exp_shape;
    exp -> arg1 = arg1;
    exp -> arg2 = arg2;
    exp -> type = MATMUL;
    return exp;
}

expression* exp_sub(expression* arg1, expression* arg2) {
    shape exp_shape = broadcast_shape(arg1 -> shape, arg2 -> shape);
    if (exp_shape.sizes == NULL) {
        return NULL;
    }

    expression* exp = malloc(sizeof(expression));

    size_t length = 1;
    for (size_t i = 0; i < exp_shape.sizes_length; i++) {
        length *= exp_shape.sizes[i];
    }
    exp -> values = malloc(sizeof(double) * length);
    for (size_t i = 0; i < length; i++) {
        exp -> values[i] = 0.0;
    }

    exp -> derivative = NULL;
    exp -> shape = exp_shape;
    exp -> arg1 = arg1;
    exp -> arg2 = arg2;
    exp -> type = SUB;
    return exp;
}

void free_exp(expression* exp) {
    free(exp -> values);
    if (exp -> type == VAR) {
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

// TODO: The length calculations can overflow, add checks for them.
void calc(expression* expr) {
    size_t length = 1;
    for (size_t i = 0; i < expr -> shape.sizes_length; i++) {
        length *= expr -> shape.sizes[i];
    }

    switch (expr -> type) {
        case VAR: {
            break;
        }
        case SIN: {
            calc(expr -> arg1);
            for (size_t i = 0; i < length; i++) {
                expr -> values[i] = sin(expr -> arg1 -> values[i]);
            }
            break;
        }
        case RELU: {
            calc(expr -> arg1);
            for (size_t i = 0; i < length; i++) {
                expr -> values[i] = (expr -> arg1 -> values[i] > 0) ? expr -> arg1 -> values[i] : 0;
            }
            break;
        }
        case SIGMOID: {
            calc(expr -> arg1);
            for (size_t i = 0; i < length; i++) {
                expr -> values[i] = 1 / (1 + exp(-(expr -> arg1 -> values[i])));
            }
            break;
        }
        case LOG: {
            calc(expr -> arg1);
            for (size_t i = 0; i < length; i++) {
                expr -> values[i] = (expr -> arg1 -> values[i] > 0) ? log(expr -> arg1 -> values[i]) : -INFINITY;
            }
            break;
        }
        case ADD: {
            calc(expr -> arg1);
            calc(expr -> arg2);
            for (size_t i = 0; i < length; i++) {
                size_t arg1_index = broadcast_index(i, expr -> shape, expr -> arg1 -> shape);
                size_t arg2_index = broadcast_index(i, expr -> shape, expr -> arg2 -> shape);
                expr -> values[i] = expr -> arg1 -> values[arg1_index] + expr -> arg2 -> values[arg2_index];
            }
            break;
        }
        case MUL: {
            calc(expr -> arg1);
            calc(expr -> arg2);
            for (size_t i = 0; i < length; i++) {
                size_t arg1_index = broadcast_index(i, expr -> shape, expr -> arg1 -> shape);
                size_t arg2_index = broadcast_index(i, expr -> shape, expr -> arg2 -> shape);
                expr -> values[i] = expr -> arg1 -> values[arg1_index] * expr -> arg2 -> values[arg2_index];
            }
            break;
        }
        case MATMUL: {
            calc(expr -> arg1);
            calc(expr -> arg2);

            size_t one[] = { 1 };
            size_t a = 1, b = 1, c = 1;
            if (expr -> arg1 -> shape.sizes_length > 1) {
                a = expr -> arg1 -> shape.sizes[expr -> arg1 -> shape.sizes_length - 2];
            }
            if (expr -> arg2 -> shape.sizes_length > 1) {
                c = expr -> arg2 -> shape.sizes[expr -> arg2 -> shape.sizes_length - 1];
            }
            b = expr -> arg1 -> shape.sizes[expr -> arg1 -> shape.sizes_length - 1];
            
            shape broadcast_res_shape = {
                .sizes = one,
                .sizes_length = 1
            };
            shape broadcast_shape1 = {
                .sizes = one,
                .sizes_length = 1,
            };
            shape broadcast_shape2 = {
                .sizes = one,
                .sizes_length = 1,
            };
            size_t broadcast_res_shape_sizes_length = expr -> shape.sizes_length - 1;
            if (expr -> arg1 -> shape.sizes_length > 1 && expr -> arg2 -> shape.sizes_length > 1) {
                broadcast_res_shape_sizes_length = expr -> shape.sizes_length - 2;
            }
            if (broadcast_res_shape_sizes_length > 0) {
                broadcast_res_shape.sizes = expr -> shape.sizes;
                broadcast_res_shape.sizes_length = broadcast_res_shape_sizes_length;
            }
            if (expr -> arg1 -> shape.sizes_length > 2) {
                broadcast_shape1.sizes = expr -> arg1 -> shape.sizes;
                broadcast_shape1.sizes_length = expr -> arg1 -> shape.sizes_length - 2;
            }
            if (expr -> arg2 -> shape.sizes_length > 2) {
                broadcast_shape2.sizes = expr -> arg2 -> shape.sizes;
                broadcast_shape2.sizes_length = expr -> arg2 -> shape.sizes_length - 2;
            }

            for (size_t s = 0; s < length / (a * c); s++) {
                size_t res_index = s * a * c;
                size_t arg1_index = broadcast_index(s, broadcast_res_shape, broadcast_shape1) * a * b;
                size_t arg2_index = broadcast_index(s, broadcast_res_shape, broadcast_shape2) * b * c;
                for (size_t i = 0; i < a; i++) {
                    for (size_t j = 0; j < c; j++) {
                        expr -> values[res_index + i * c + j] = 0.0;
                        for (size_t k = 0; k < b; k++) {
                            expr -> values[res_index + i * c + j]
                                += expr -> arg1 -> values[arg1_index + i * b + k] * expr -> arg2 -> values[arg2_index + k * c + j];
                        }
                    }
                }
            }
            break;
        }
        case SUB: {
            calc(expr -> arg1);
            calc(expr -> arg2);
            for (size_t i = 0; i < length; i++) {
                size_t arg1_index = broadcast_index(i, expr -> shape, expr -> arg1 -> shape);
                size_t arg2_index = broadcast_index(i, expr -> shape, expr -> arg2 -> shape);
                expr -> values[i] = expr -> arg1 -> values[arg1_index] - expr -> arg2 -> values[arg2_index];
            }
            break;
        }
    }
}

void free_pointer(void* p) {
    free(p);
}

void backward_recursive(expression* expr, double* mult) {
    size_t length = 1;
    for (size_t i = 0; i < expr -> shape.sizes_length; i++) {
        length *= expr -> shape.sizes[i];
    }

    switch (expr -> type) {
        case VAR: {
            for (size_t i = 0; i < length; i++) {
                expr -> derivative[i] += mult[i];
            }
            break;
        }
        case SIN: {
            for (size_t i = 0; i < length; i++) {
                mult[i] *= cos(expr -> arg1 -> values[i]);
            }
            backward_recursive(expr -> arg1, mult);
            break;
        }
        case RELU: {
            for (size_t i = 0; i < length; i++) {
                mult[i] *= (expr -> arg1 -> values[i] > 0.0) ? 1.0 : 0.0;
            }
            backward_recursive(expr -> arg1, mult);
            break;
        }
        case SIGMOID: {
            for (size_t i = 0; i < length; i++) {
                double exp_val = exp(-(expr -> arg1 -> values[i]));
                mult[i] *= exp_val / ((1 + exp_val) * (1 + exp_val));
            }
            backward_recursive(expr -> arg1, mult);
            break;
        }
        case LOG: {
            for (size_t i = 0; i < length; i++) {
                // TODO: What should I do for negative values?
                mult[i] *= 1 / expr -> arg1 -> values[i];
            }
            backward_recursive(expr -> arg1, mult);
            break;
        }
        case ADD: {
            size_t arg1_length = shape_total_length(expr -> arg1 -> shape);
            double *arg1_mult = malloc(sizeof(double) * arg1_length);
            for (size_t i = 0; i < arg1_length; i++) {
                arg1_mult[i] = 0.0;
            }
            size_t arg2_length = shape_total_length(expr -> arg2 -> shape);
            double *arg2_mult = malloc(sizeof(double) * arg2_length);
            for (size_t i = 0; i < arg2_length; i++) {
                arg2_mult[i] = 0.0;
            }
            for (size_t i = 0; i < length; i++) {
                size_t arg1_index = broadcast_index(i, expr -> shape, expr -> arg1 -> shape);
                size_t arg2_index = broadcast_index(i, expr -> shape, expr -> arg2 -> shape);
                arg1_mult[arg1_index] += mult[i];
                arg2_mult[arg2_index] += mult[i];
            }

            backward_recursive(expr -> arg1, arg1_mult);
            backward_recursive(expr -> arg2, arg2_mult);
            free(arg1_mult);
            free(arg2_mult);
            break;
        }
        case MUL: {
            size_t arg1_length = shape_total_length(expr -> arg1 -> shape);
            double *arg1_mult = malloc(sizeof(double) * arg1_length);
            for (size_t i = 0; i < arg1_length; i++) {
                arg1_mult[i] = 0.0;
            }
            size_t arg2_length = shape_total_length(expr -> arg2 -> shape);
            double *arg2_mult = malloc(sizeof(double) * arg2_length);
            for (size_t i = 0; i < arg2_length; i++) {
                arg2_mult[i] = 0.0;
            }
            for (size_t i = 0; i < length; i++) {
                size_t arg1_index = broadcast_index(i, expr -> shape, expr -> arg1 -> shape);
                size_t arg2_index = broadcast_index(i, expr -> shape, expr -> arg2 -> shape);
                arg1_mult[arg1_index] += mult[i] * expr -> arg2 -> values[arg2_index];
                arg2_mult[arg2_index] += mult[i] * expr -> arg1 -> values[arg1_index];
            }

            backward_recursive(expr -> arg1, arg1_mult);
            backward_recursive(expr -> arg2, arg2_mult);
            free(arg1_mult);
            free(arg2_mult);
            break;
        }
        case MATMUL: {
            size_t arg1_length = shape_total_length(expr -> arg1 -> shape);
            double *arg1_mult = malloc(sizeof(double) * arg1_length);
            for (size_t i = 0; i < arg1_length; i++) {
                arg1_mult[i] = 0.0;
            }
            size_t arg2_length = shape_total_length(expr -> arg2 -> shape);
            double *arg2_mult = malloc(sizeof(double) * arg2_length);
            for (size_t i = 0; i < arg2_length; i++) {
                arg2_mult[i] = 0.0;
            }

            size_t one[] = { 1 };
            size_t a = 1, b = 1, c = 1;
            if (expr -> arg1 -> shape.sizes_length > 1) {
                a = expr -> arg1 -> shape.sizes[expr -> arg1 -> shape.sizes_length - 2];
            }
            if (expr -> arg2 -> shape.sizes_length > 1) {
                c = expr -> arg2 -> shape.sizes[expr -> arg2 -> shape.sizes_length - 1];
            }
            b = expr -> arg1 -> shape.sizes[expr -> arg1 -> shape.sizes_length - 1];
            
            shape broadcast_res_shape = {
                .sizes = one,
                .sizes_length = 1
            };
            shape broadcast_shape1 = {
                .sizes = one,
                .sizes_length = 1,
            };
            shape broadcast_shape2 = {
                .sizes = one,
                .sizes_length = 1,
            };
            size_t broadcast_res_shape_sizes_length = expr -> shape.sizes_length - 1;
            if (expr -> arg1 -> shape.sizes_length > 1 && expr -> arg2 -> shape.sizes_length > 1) {
                broadcast_res_shape_sizes_length = expr -> shape.sizes_length - 2;
            }
            if (broadcast_res_shape_sizes_length > 0) {
                broadcast_res_shape.sizes = expr -> shape.sizes;
                broadcast_res_shape.sizes_length = broadcast_res_shape_sizes_length;
            }
            if (expr -> arg1 -> shape.sizes_length > 2) {
                broadcast_shape1.sizes = expr -> arg1 -> shape.sizes;
                broadcast_shape1.sizes_length = expr -> arg1 -> shape.sizes_length - 2;
            }
            if (expr -> arg2 -> shape.sizes_length > 2) {
                broadcast_shape2.sizes = expr -> arg2 -> shape.sizes;
                broadcast_shape2.sizes_length = expr -> arg2 -> shape.sizes_length - 2;
            }

            for (size_t s = 0; s < length / (a * c); s++) {
                size_t res_index = s * a * c;
                size_t arg1_index = broadcast_index(s, broadcast_res_shape, broadcast_shape1) * a * b;
                size_t arg2_index = broadcast_index(s, broadcast_res_shape, broadcast_shape2) * b * c;
                for (size_t i = 0; i < a; i++) {
                    for (size_t j = 0; j < c; j++) {
                        for (size_t k = 0; k < b; k++) {
                            arg1_mult[arg1_index + i * b + k] += mult[res_index + i * c + j] * expr -> arg2 -> values[arg2_index + k * c + j];
                            arg2_mult[arg2_index + k * c + j] += mult[res_index + i * c + j] * expr -> arg1 -> values[arg1_index + i * b + k];
                        }
                    }
                }
            }

            backward_recursive(expr -> arg1, arg1_mult);
            backward_recursive(expr -> arg2, arg2_mult);
            free(arg1_mult);
            free(arg2_mult);
            break;
        }
        case SUB: {
            size_t arg1_length = shape_total_length(expr -> arg1 -> shape);
            double *arg1_mult = malloc(sizeof(double) * arg1_length);
            for (size_t i = 0; i < arg1_length; i++) {
                arg1_mult[i] = 0.0;
            }
            size_t arg2_length = shape_total_length(expr -> arg2 -> shape);
            double *arg2_mult = malloc(sizeof(double) * arg2_length);
            for (size_t i = 0; i < arg2_length; i++) {
                arg2_mult[i] = 0.0;
            }
            for (size_t i = 0; i < length; i++) {
                size_t arg1_index = broadcast_index(i, expr -> shape, expr -> arg1 -> shape);
                size_t arg2_index = broadcast_index(i, expr -> shape, expr -> arg2 -> shape);
                arg1_mult[arg1_index] += mult[i];
                arg2_mult[arg2_index] -= mult[i];
            }

            backward_recursive(expr -> arg1, arg1_mult);
            backward_recursive(expr -> arg2, arg2_mult);
            free(arg1_mult);
            free(arg2_mult);
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

    calc(exp);
    backward_recursive(exp, mult);
    free(mult);
}