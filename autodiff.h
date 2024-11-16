#ifndef AUTODIFF
#define AUTODIFF

#include <stdlib.h>
#include <stdint.h>

// https://stackoverflow.com/a/228757
// https://stackoverflow.com/a/228691

typedef enum {
    TENSOR,
    SIN,
    RELU,
    SIGMOID,
    LOG,
    POWER,
    ADD,
    MUL,
    MATMUL,
    SUB,
} expression_type;

typedef struct {
    size_t *sizes;
    size_t sizes_length;
} shape;

typedef struct expression {
    double *values;
    double *derivative;
    shape shape;
    struct expression *arg1;
    struct expression *arg2;
    // Additional argument for expression, for example the exponent for power.
    double arg3;
    expression_type type;
} expression;

expression* tensor(double *values, size_t *sizes, size_t sizes_length);
expression* exp_sin(expression* arg);
expression* exp_relu(expression* arg);
expression* exp_sigmoid(expression* arg);
expression* exp_log(expression* arg);
expression* exp_power(expression* arg, double exponent);
expression* exp_add(expression* arg1, expression* arg2);
expression* exp_mul(expression* arg1, expression* arg2);
expression* exp_matmul(expression* arg1, expression* arg2);
expression* exp_sub(expression* arg1, expression* arg2);
void free_exp(expression* exp);
void calc(expression* expr);
void backward(expression* exp);
void set_derivative_zero(expression* expr);
void derivative_step(expression* expr, double multiplier);

#endif