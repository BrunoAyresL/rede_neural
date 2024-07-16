#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include <math.h>
#include "functions.h"

    // operações Tensor x Tensor
    Tensor* Tensor::operator + (Tensor* other)  {
        // verificação


        // broadcast (teste, refazer depois)
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

    Tensor* Tensor::operator - (Tensor* other)  {
        // verificação
        if (n_dim != other->n_dim) {
            printf("\nERRO: Subtração de tensores de diferentes dimensões.");
            exit(1);
        } 
        for (int i = 0; i < n_dim; i++) {
            if (shape[i] != other->shape[i]) {
                printf("\nERRO: Subtração de tensores de formatos diferentes.");
                exit(1);
            }
        }

        float* new_data = new float[size];
        for (int i = 0; i < size; i++) {
            new_data[i] = data[i] - other->data[i];
        }

        return new Tensor(new_data, shape, n_dim);
    }

    void Tensor::operator += (Tensor* other) {

        // nao ta completo !!!!!!!

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

    Tensor* Tensor::operator * (Tensor* other) {
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

    void Tensor::operator *= (Tensor* other) {

        // nao ta completo !!!!!!!

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

    Tensor* Tensor::operator / (Tensor* other) {
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

    Tensor* Tensor::operator & (Tensor* other) {
        // verificação (tem que melhorar isso)
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
    Tensor* Tensor::operator + (float scalar)  {
        float* new_data = new float[size];
        for (int i = 0; i < size; i++) {
            new_data[i] = data[i] + scalar;
        }
        Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
        return result;
    }

    Tensor* Tensor::operator - (float scalar)  {
        float* new_data = new float[size];
        for (int i = 0; i < size; i++) {
            new_data[i] = data[i] - scalar;
        }
        Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
        return result;
    }

    Tensor* Tensor::operator * (float scalar)  {
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

    void Tensor::operator *= (float scalar) {

        // nao ta completo !!!!!!!

        for (int i = 0; i < size; i++) {
            data[i] *= scalar;
        }
    }

    void Tensor::operator += (float scalar)  {
        for (int i = 0; i < size; i++) {
            data[i] = data[i] + scalar;
        }
    }

    void Tensor::operator -= (float scalar)  {
        for (int i = 0; i < size; i++) {
            data[i] = data[i] - scalar;
        }
    }
    
    Tensor* Tensor::operator-() {

        // tem que derivar isso???????????

        float* new_data = new float[size];
        for (int i = 0; i < size; i++) {
            new_data[i] = -data[i];
        }
        Tensor* t_new = new Tensor(new_data, shape, n_dim);
        return t_new;
    }