// Example on how to use the library.
// Compile this example using:
// gcc autodiff.c example.c -lm -fPIC

#include "autodiff.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));

    // In this example, we will perform linear regression.

    // Actual values
    double a = 10.0 * ((rand() % 2000) / 1000.0 - 1.0);
    double b = 10.0 * ((rand() % 2000) / 1000.0 - 1.0);
    double c = 10.0 * ((rand() % 2000) / 1000.0 - 1.0);

    // We create a tensor using the tensor function which takes:
    // values: If non null, it copies the values, else it zero initializes.
    // sizes: The sizes of the tensors.
    // sizes_length: The length of the sizes.
    // It copies both values and sizes.
    size_t x_size[] = { 100, 2 };
    size_t y_size[] = { 100 };
    expression* x = tensor(NULL, x_size, 2);
    expression* y = tensor(NULL, y_size, 1);

    for (size_t i = 0; i < 100; i++) {
        // For tensors, you can directly modify the values / derivatives they point too.
        x -> values[2*i] = 10.0 * ((rand() % 2000) / 1000.0 - 1.0);
        x -> values[2*i + 1] = 10.0 * ((rand() % 2000) / 1000.0 - 1.0);
        y -> values[i] = a * (x -> values[2*i]) + b * (x -> values[2*i+1]) + c;
    }

    // Parameters we will learn.
    size_t coeff_size[] = { 2 };
    size_t intercept_size[] = { 1 };
    expression* coeff = tensor(NULL, coeff_size, 1);
    expression* intercept = tensor(NULL, intercept_size, 1);

    printf("Hello\n");
    // Create expression using the exp_ functions.
    // If the creation failed (due to memory / shape mismatch),
    // NULL is returned.
    expression* exp1 = exp_matmul(x, coeff);
    expression* exp2 = exp_add(exp1, intercept);
    expression* exp3 = exp_sub(y, exp2);
    expression* exp4 = exp_power(exp3, 2.0);
    double exp5_values[] = { 1.0 / 100.0 };
    size_t exp5_shape[] = { 1 };
    expression* exp5 = tensor(exp5_values, exp5_shape, 1);
    expression *loss = exp_mul(exp4, exp5);

    for (size_t i = 0 ; i < 1000; i++) {
        printf("Coefficients: %f %f\n", coeff->values[0], coeff->values[1]);
        printf("Intercept: %f\n", intercept->values[0]);

        // If you want the value of an expression,
        // you calc and then read the values.
        calc(loss);
        printf("loss[0]: %f\n\n", loss->values[0]); // Loss has shape [100].

        // Set derivative zero for tensors.
        set_derivative_zero(coeff);
        set_derivative_zero(intercept);

        // This will add the partial derivatives of the expression,
        // to the derivative field of the tensors.
        // This means multiple calls of backward will add up.
        // If the expression is multi-dimensional, we consider the
        // sum of all entries in the expression and consider its partial derivatives.
        backward(loss);

        // Take a step along the derivative.
        derivative_step(coeff, -0.01);
        derivative_step(intercept, -0.01);
    }

    printf("Actual values: %f %f %f\n", a, b, c);

    // Free the expressions once done.
    // free_exp just frees the specific expression,
    // not its children.
    free_exp(loss);
    free_exp(exp5);
    free_exp(exp4);
    free_exp(exp3);
    free_exp(exp2);
    free_exp(exp1);
    free_exp(intercept);
    free_exp(coeff);
    free_exp(y);
    free_exp(x);
}