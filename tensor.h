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
        Tensor* grad;
        int size;
        float* data;
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

        // operações Tensor x Tensor
        Tensor* operator + (Tensor* other)  {
            // verificação


            // broadcast (teste)
            if (n_dim == 1 && other->n_dim > 1) {
                broadcast(other);
            }
            if (n_dim > 1 && other->n_dim == 1) {
                other->broadcast(this);
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

            Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
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


            // broadcast (teste)
            if (n_dim == 1 && other->n_dim > 1) {
                broadcast(other);
            }
            if (n_dim > 1 && other->n_dim == 1) {
                other->broadcast(this);
            }


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


            // broadcast (teste)
            if (n_dim == 1 && other->n_dim > 1) {
                broadcast(other);
            }
            if (n_dim > 1 && other->n_dim == 1) {
                other->broadcast(this);
            }

            if (n_dim != other->n_dim) {
                printf("\nERRO: Multiplicação escalar de tensores de diferentes dimensões.\nTENSOR 1:");
                this->print();
                printf("\nTENSOR 2:");
                other->print();
                exit(1);
            } 
            for (int i = 0; i < n_dim; i++) {
                if (shape[i] != other->shape[i]) {
                    printf("\nERRO: Multiplicação escalar de tensores de formatos diferentes. (%d - %d)", shape[i], other->shape[i]);
                    exit(1);
                }
            }

            float* new_data = new float[size];
            for (int i = 0; i < size; i++) {
                new_data[i] = data[i] * other->data[i];
            }
            Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
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

        Tensor* operator / (Tensor* other) {
            if (n_dim != other->n_dim) {
                printf("\nERRO: Divisão de tensores de diferentes dimensões.");
                exit(1);
            } 
            for (int i = 0; i < n_dim; i++) {
                if (shape[i] != other->shape[i]) {
                    printf("\nERRO: Divisão de tensores de formatos diferentes.");
                    exit(1);
                }
            }

            float* new_data = new float[size];
            for (int i = 0; i < size; i++) {
                new_data[i] = data[i] / other->data[i];
            }
            Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
            if (requires_grad) {
                result->grad_fn = new Div(this, other);
            }
            return result;            
        }

        Tensor* operator & (Tensor* other) {
            // verificação
            if (n_dim != other->n_dim) {
                printf("\nERRO: Produto de tensores de diferentes dimensões.");
                exit(1);
            } 
            if (shape[0] != other->shape[1]) {
                this->t_();
                printf("\nERRO: Produto de tensores de formatos diferentes. (%d - %d)", shape[0], other->shape[1]);
                this->print("THIS");
                other->print("OTHER");
            }

            float* new_data = new float[size];
            // assumir 2 dims

            int m = shape[0];
            int n = other->shape[1];
            int p = shape[1];
            
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    float  sum = 0.0;
                    for (int k = 0; k < p; k++) {
                        sum += data[i * strides[0] + k] * other->data[k * other->strides[0] + j * other->strides[1]];
                    }
                    new_data[i * n + j] = sum;
                }
            }
            
            printf("\n%d %d %d", m, n, p);
            int new_shape[] = {n, p};
            Tensor* result = new Tensor(new_data, new_shape, n_dim, requires_grad);
            if (requires_grad) {
                result->grad_fn = new MatMul(this, other);
            }
            
            return result;
        }

        // operações Tensor x float
        Tensor* operator + (float scalar)  {
            float* new_data = new float[size];
            for (int i = 0; i < size; i++) {
                new_data[i] = data[i] + scalar;
            }
            Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
            return result;
        }

        Tensor* operator - (float scalar)  {
            float* new_data = new float[size];
            for (int i = 0; i < size; i++) {
                new_data[i] = data[i] - scalar;
            }
            Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
            return result;
        }

        Tensor* operator * (float scalar)  {
            float* new_data = new float[size];
            for (int i = 0; i < size; i++) {
                new_data[i] = data[i] * scalar;
            }
            Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
            if (requires_grad) {
                result->grad_fn = new Scalar_Mul(this, scalar);
            }
            return result;
        }

        void operator *= (float scalar) {
            for (int i = 0; i < size; i++) {
                data[i] *= scalar;
            }
        }

        void operator += (float scalar)  {
            for (int i = 0; i < size; i++) {
                data[i] = data[i] + scalar;
            }
        }

        void operator -= (float scalar)  {
            for (int i = 0; i < size; i++) {
                data[i] = data[i] - scalar;
            }
        }

        friend Tensor* operator / (float scalar, Tensor other) {
            float* new_data = new float[other.size];
            for (int i = 0; i < other.size; i++) {
                new_data[i] = scalar / other.data[i];
            }
            Tensor* t_new = new Tensor(new_data, other.shape, other.n_dim);
            return t_new;
        }

        Tensor* operator-() {
            float* new_data = new float[size];
            for (int i = 0; i < size; i++) {
                new_data[i] = -data[i];
            }
            Tensor* t_new = new Tensor(new_data, shape, n_dim);
            return t_new;
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
        void expand(int* new_shape);

        void broadcast(Tensor* other);

        Tensor* t();
        void t_();
        Tensor* abs();
        Tensor* sqrt();
        Tensor* tanh();
        Tensor* mean();
        Tensor* pow(float x);
        Tensor* sin();
        Tensor* cos();
        Tensor* tan();
        Tensor* softmax();

        void print(char* str = " ") {

            printf("\n%s -> Tensor(", str);

            for (int i = 0; i < n_dim; i++) {
                printf("%d", shape[i]);
                if (i < n_dim -1) printf(", ");
            }

            printf("): [");
            
            if (n_dim == 1) {
                for (int i = 0; i < size; i++) {
                    printf("%f", data[i]);
                    if (i < size - 1) printf(", ");
                }
                printf("]");
                return;
            }


            for(int i = 0; i < shape[0] * shape[1]; i++) {
                printf("%f", data[i]);
                if (i < shape[0] * shape[1] - 1) printf(", ");
            }

            

            printf("]");
        }
        

};
Tensor* create_Tensor_full(float x, int* shape, int n_dim);

#endif