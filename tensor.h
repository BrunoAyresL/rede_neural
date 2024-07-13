#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "functions.h"

class Function;

class Tensor {
    public:
        int n_dim;
        int size;
        float* data;
        Tensor* grad;
        int* strides;
        int* shape;

        bool requires_grad;
        Function* grad_fn;


        // constructor
        Tensor(float* new_data, int* new_shape, int new_n_dim, bool req_grad=false) {

            requires_grad = req_grad;
            grad = NULL;
            grad_fn = new NullFunction();

            n_dim = new_n_dim;
            size = 1;
            shape = new int[n_dim];
            for (int i = 0; i < n_dim; i++) {
                shape[i] = new_shape[i];
                size *= new_shape[i];
            }

            data = new_data;        
            strides = new int[n_dim];
            int stride = 1;
            for (int i = n_dim - 1; i >= 0; i--) {
                strides[i] = stride;
                stride *= shape[i];
            }               
        }

        // destructor
        ~Tensor() {
            delete[] data;
            delete[] grad;
            delete[] strides;
            delete[] shape;
        }

        // operações
        Tensor* operator + (Tensor* other)  {
            // verificação
            if (n_dim != other->n_dim) {
                printf("\nERRO: Soma de tensores de diferentes dimensões.");
                exit(1);
            } 
            for (int i = 0; i < n_dim; i++) {
                if (shape[i] != other->shape[i]) {
                    printf("\nERRO: Soma de tensores de formatos diferentes.");
                    exit(1);
                }
            }

            float* new_data = new float[size];
            for (int i = 0; i < size; i++) {
                new_data[i] = data[i] + other->data[i];
            }

            Tensor* result = new Tensor(new_data, shape, n_dim, true);
            if (requires_grad) {
                result->grad_fn = new Add(this, other);
            }
            return result;
        }

        Tensor* operator - (const Tensor& other) const {
            // verificação
            if (n_dim != other.n_dim) {
                printf("\nERRO: Subtração de tensores de diferentes dimensões.");
                exit(1);
            } 
            for (int i = 0; i < n_dim; i++) {
                if (shape[i] != other.shape[i]) {
                    printf("\nERRO: Subtração de tensores de formatos diferentes.");
                    exit(1);
                }
            }

            float* new_data = new float[size];
            for (int i = 0; i < size; i++) {
                new_data[i] = data[i] - other.data[i];
            }

            return new Tensor(new_data, shape, n_dim);
        }

        void operator += (Tensor* other) {
            // verificação
            if (n_dim != other->n_dim) {
                printf("\nERRO: Soma de tensores de diferentes dimensões.");
                exit(1);
            } 
            for (int i = 0; i < n_dim; i++) {
                if (shape[i] != other->shape[i]) {
                    printf("\nERRO: Soma de tensores de formatos diferentes.");
                    exit(1);
                }
            }

            for (int i = 0; i < size; i++) {
                data[i] += other->data[i];
            }
        }

        Tensor* operator * (Tensor* other) {
            // verificação
            if (n_dim != other->n_dim) {
                printf("\nERRO: Multiplicação escalar de tensores de diferentes dimensões.");
                exit(1);
            } 
            for (int i = 0; i < n_dim; i++) {
                if (shape[i] != other->shape[i]) {
                    printf("\nERRO: Multiplicação escalar de tensores de formatos diferentes.");
                    exit(1);
                }
            }

            float* new_data = new float[size];
            for (int i = 0; i < size; i++) {
                new_data[i] = data[i] * other->data[i];
            }
            Tensor* result = new Tensor(new_data, shape, n_dim, true);
            if (requires_grad) {
                result->grad_fn = new Mul(this, other);
            }
            return result;
        }

        void operator *= (Tensor* other) {
            // verificação
            if (n_dim != other->n_dim) {
                printf("\nERRO: Multiplicação escalar de tensores de diferentes dimensões.");
                exit(1);
            } 
            for (int i = 0; i < n_dim; i++) {
                if (shape[i] != other->shape[i]) {
                    printf("\nERRO: Multiplicação escalar de tensores de formatos diferentes.");
                    exit(1);
                }
            }
            for (int i = 0; i < size; i++) {
                data[i] *= other->data[i];
            }
        }

        void backward(Tensor* grad_output = NULL) {
            if (requires_grad) {
                if (grad_output == NULL) {
                    float* new_data = new float[size];
                    for (int i = 0; i < size; i++) {
                        new_data[i] = 1.0;
                    }
                    grad_output = new Tensor(new_data, shape, n_dim, false);

                }
                if (grad == NULL) {
                    float* new_data = new float[size];
                    for (int i = 0; i < size; i++) {
                        new_data[i] = 0.0;
                    }
                    grad = new Tensor(new_data, shape, n_dim, false);
                }
                grad_fn->backward(grad_output);
            }
        }



        void add_scalar(float x);
        void expand(int* new_shape);


        Tensor* abs();
        Tensor* neg();
        Tensor* sqrt();
        Tensor* tanh();
        Tensor* mul_scalar(float x);
        Tensor* mean();
        Tensor* pow(float x);
        Tensor* sin();
        Tensor* cos();
        Tensor* tan();
        Tensor* softmax();
        void print() const {

            printf("\nTensor(");

            for (int i = 0; i < n_dim; i++) {
                printf("%d", shape[i]);
                if (i < n_dim -1) printf(", ");
            }

            printf("): [");

            for(int i = 0; i < shape[0] * shape[1]; i++) {
                printf("%f", data[i]);
                if (i < shape[0] * shape[1] - 1) printf(", ");
            }

            printf("]");
        }
        

};
Tensor* create_Tensor_full(float x, int* shape, int n_dim);

#endif