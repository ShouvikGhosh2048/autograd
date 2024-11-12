// TODO: Create header files for general use.
#include "autodiff.c"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// https://stackoverflow.com/a/5141996
// https://stackoverflow.com/a/5142028

int main(void) {
    {
        // https://www.geeksforgeeks.org/how-arrays-are-passed-to-functions-in-cc/
        size_t train_x_dimensions[] = {10000, 1, 784};
        expression* train_x = tensor(NULL, train_x_dimensions, 3);
        size_t train_y_dimensions[] = {10000, 1};
        expression* train_y = tensor(NULL, train_y_dimensions, 2);
        size_t w1_dimensions[] = {784, 50};
        expression* w1 = tensor(NULL, w1_dimensions, 2);
        size_t b1_dimensions[] = {50};
        expression* b1 = tensor(NULL, b1_dimensions, 1);
        size_t w2_dimensions[] = {50};
        expression* w2 = tensor(NULL, w2_dimensions, 1);
        size_t b2_dimensions[] = {1};
        expression* b2 = tensor(NULL, b2_dimensions, 1);

        expression* res = exp_matmul(train_x, w1);
        res = exp_add(res, b1);
        res = exp_relu(res);
        res = exp_matmul(res, w2);
        res = exp_add(res, b2);
        res = exp_sub(res, train_y);
        expression* res2 = exp_matmul(train_x, w1);
        res2 = exp_add(res2, b1);
        res2 = exp_relu(res2);
        res2 = exp_matmul(res2, w2);
        res2 = exp_add(res2, b2);
        res2 = exp_sub(res2, train_y);
        res = exp_mul(res, res2);

        {
            time_t start = time(NULL);
            for (int iter = 0; iter < 10; iter++) {
                calc(res);
            }
            time_t end = time(NULL);
            printf("%ld\n", (end - start));
        }
        {
            time_t start = time(NULL);
            for (int iter = 0; iter < 10; iter++) {
                backward(res);
            }
            time_t end = time(NULL);
            printf("%ld\n", (end - start));
        }
    }

    {
        size_t a = 10000;
        size_t b = 784;
        size_t c = 50;

        double* buffer1 = malloc(sizeof(double) * a * b);
        double* buffer2 = malloc(sizeof(double) * b * c);
        double* buffer3 = malloc(sizeof(double) * a * c);
        for (int i = 0; i < a * b; i++) {
            buffer1[i] = 0.0;
        }
        for (int i = 0; i < b * c; i++) {
            buffer2[i] = 0.0;
        }

        time_t start = time(NULL);
        for (int iter = 0; iter < 10; iter++) {
            for (int i = 0; i < a; i++) {
                for (int j = 0; j < c; j++) {
                    buffer3[c * i + j] = 0;
                    for (int k = 0; k < b; k++) {
                        buffer3[c * i + j] += buffer1[b * i + k] * buffer2[c * k + j];
                    }
                }
            }
        }
        time_t end = time(NULL);
        printf("%ld\n", (end - start));
    }
}