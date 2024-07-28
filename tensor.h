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
    std::vector<float> data;
    std::vector<int> strides;
    std::vector<int> shape;

    bool requires_grad;
    bool keep_grad;

    Function* grad_fn;
    Tensor* grad;
    std::string op;


    // constructor
    Tensor(std::vector<float> new_data, std::vector<int> new_shape, bool req_grad=false) {

        data = new_data;
        shape = new_shape;
        requires_grad = req_grad;
        op = "NULL";
        grad = nullptr;
        grad_fn = new NullFunction();
        keep_grad = false;

        n_dim = new_shape.size();

        size = 1;
        for (auto dim : shape) {
            size *= dim;
        }

        strides.resize(n_dim);

        int stride = 1;
        for (int i = n_dim - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }   

        TensorRegistry::add(this);   
        
        //printf("\nnew Tensor: size=%d shape=(%d, %d) n_dim=%d strides=(%d, %d)", size, shape[0], shape[1], n_dim, strides[0], strides[1]);  
    }

    // destructor
    ~Tensor() {
        TensorRegistry::remove(this);
        if (grad != nullptr) {
            delete grad;
        }
        delete grad_fn;
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

    void backward(Tensor* grad_output);

    Tensor* broadcast(Tensor* other);
    Tensor* t();
    void t_();
    Tensor* abs();
    Tensor* sqrt();
    Tensor* tanh();
    Tensor* mean();
    Tensor* sum(int dim);
    Tensor* pow(float x);
    Tensor* exp();
    Tensor* log();
    Tensor* sin();
    Tensor* cos();
    Tensor* tan();
    Tensor* softmax();
    Tensor* nll(Tensor* y);
    Tensor* cross_entropy(Tensor* targets);
    Tensor* max(int dim);
    Tensor* one_hot(int size);
    Tensor* index(Tensor* X, Tensor* Y);
    Tensor* reshape(std::vector<int> new_shape);


    void print(const char str[]  = " ") {

        // modo compacto
        if (str[0] == '$' && n_dim > 1) {

            printf("\n%s -> ", str);
            std::cout << op << "";
            printf("(");

            for (int i = 0; i < n_dim; i++) {
                printf("%d", shape[i]);
                if (i < n_dim -1) printf(", ");
            }

            printf("): (");

            for(int i = 0; i < shape[0]; i++) {
                printf("[");
                for (int j = 0; j < shape[1]; j++) {

                    // remover isso dps
                    if (data[j + i * strides[0]] == 0.00000) {
                        printf("0");
                    } else {
                        printf("%.1f", data[j + i * strides[0]]);
                    }
                    




                    if (j <  shape[1] - 1) printf(",");
                }
                printf("]");
                if (i < shape[0] - 1) {
                    printf(",\n");
                    for (int k = 0; k < 2 * n_dim + strlen(str) + 11 + op.length(); k++) {
                        printf(" ");
                    }
                } else printf(")\n");

            }   
            return;     
        }




        // tem que possibilitar printar mais de 3 dimensões
        printf("\n%s -> ", str);
        std::cout << op << "";
        printf("(");

        for (int i = 0; i < n_dim; i++) {
            printf("%d", shape[i]);
            if (i < n_dim -1) printf(", ");
        }

        printf("): (");
        
        if (n_dim == 1) {
            printf("[");
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
                printf("%9f", data[j + i * strides[0]]);
                if (j <  shape[1] - 1) printf(", ");
            }
            printf("]");
            if (i < shape[0] - 1) {
                printf(",\n");
                for (int k = 0; k < 2 * n_dim + strlen(str) + 9 + op.length(); k++) {
                    printf(" ");
                }
            } else printf(")\n");

        }
    }
    
    void sprint(const char str[]  = " ") {
        printf("\n%s -> ", str);
        printf("%f", data[0]);
    }

};

// nome ruim
Tensor* tensor_fill(float x, std::vector<int> shape,  bool req_grad);

#endif