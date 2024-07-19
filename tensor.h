#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include "functions.h"

#include "registry.h"

#include <iostream>
#include <cstring>

class Function;

class Tensor {
public:
    int n_dim;
    
    int size;
    float* data;
    int* strides;
    int* shape;

    bool requires_grad;
    Function* grad_fn;
    Tensor* grad;
    bool is_scalar;


    // constructor
    Tensor(float* new_data, int* new_shape, int new_n_dim, bool req_grad=false) {
        is_scalar = false;
        requires_grad = req_grad;

        if (req_grad) {
            TensorRegistry::add(this);
        }

        grad = nullptr;
        grad_fn = new NullFunction();
        n_dim = new_n_dim;
        size = 1;
        shape = new int[n_dim];
        if (shape == nullptr) {
            printf("kkkkkkkkkk");
            throw std::runtime_error("Error: Failed to allocate memory for tensor shape.");
        }
        for (int i = 0; i < n_dim; i++) {
            shape[i] = new_shape[i];
            size *= new_shape[i];
        }
        data = new_data; 
        if (data == nullptr) {
            printf("kkkkkkkkkk");
            throw std::runtime_error("Error: Failed to allocate memory for tensor data.");
        }

        strides = new int[n_dim];
        if (strides == nullptr) {
            printf("kkkkkkkkkk");
            throw std::runtime_error("Error: Failed to allocate memory for tensor strides.");
        }
        int stride = 1;
        for (int i = n_dim - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }         
    }

    // destructor
    ~Tensor() {
        //print("$Deleted ");
        delete[] data;
        if (grad != nullptr) {
            delete grad;
        }
        delete[] strides;
        delete[] shape;
    }

    void zero_grad() {
        delete grad;
        grad = nullptr;
    }


    Tensor* operator + (Tensor* other);
    Tensor* operator - (Tensor* other);
    void operator += (Tensor* other);
    Tensor* operator * (Tensor* other);
    void operator *= (Tensor* other);
    Tensor* operator / (Tensor* other);
    Tensor* operator & (Tensor* other);
    Tensor* operator + (float scalar);
    Tensor* operator - (float scalar);
    Tensor* operator * (float scalar);
    void operator *= (float scalar);
    void operator += (float scalar);
    void operator -= (float scalar);

    Tensor* scalar_div (float scalar);

    Tensor* operator-();
    Tensor* index(float* x, float* y, int l);

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
            //printf("\ngrad_output: size=%d, n_dim=%d, shape[0]=%d", grad_output->size, grad_output->n_dim, grad_output->shape[0]);
            //grad_output->print("first grad_output");
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
    Tensor* sum(int dims);
    Tensor* pow(float x);
    Tensor* exp();
    Tensor* log();
    Tensor* sin();
    Tensor* cos();
    Tensor* tan();
    Tensor* softmax();
    Tensor* one_hot(int size);
    void print(const char str[]  = " ") {

        // modo compacto
        if (str[0] == '$' && n_dim > 1) {

             printf("\n%s -> (", str);

            for (int i = 0; i < n_dim; i++) {
                printf("%d", shape[i]);
                if (i < n_dim -1) printf(", ");
            }

            printf("): (");

            for(int i = 0; i < shape[0]; i++) {
                printf("[");
                for (int j = 0; j < shape[1]; j++) {
                    printf("%.1f", data[j + i * strides[0]]);
                    if (j <  shape[1] - 1) printf(",");
                }
                printf("]");
                if (i < shape[0] - 1) {
                    printf(",\n");
                    for (int k = 0; k < 2 * n_dim + strlen(str) + 11; k++) {
                        printf(" ");
                    }
                } else printf(")\n");

            }   
            return;     
        }




        // tem que possibilitar printar mais de 3 dimensões
        printf("\n%s -> Tensor(", str);

        for (int i = 0; i < n_dim; i++) {
            printf("%d", shape[i]);
            if (i < n_dim -1) printf(", ");
        }

        printf("): (");
        
        if (is_scalar) {
            printf("%f", data[0]);
            printf("]");
            return;      
        }
        if (n_dim == 1) {
            for (int i = 0; i < size; i++) {
                printf("%f", data[i]);
                if (i < size - 1) printf(", ");
            }
            printf("])\n");
            return;
        }

        for(int i = 0; i < shape[0]; i++) {
            printf("[");
            for (int j = 0; j < shape[1]; j++) {
                printf("%10f", data[j + i * strides[0]]);
                if (j <  shape[1] - 1) printf(", ");
            }
            printf("]");
            if (i < shape[0] - 1) {
                printf(",\n");
                for (int k = 0; k < 2 * n_dim + strlen(str) + 15; k++) {
                    printf(" ");
                }
            } else printf(")\n");

        }
    }
    

};

// nome ruim
Tensor* tensor_fill(float x, int* shape, int n_dim);

#endif