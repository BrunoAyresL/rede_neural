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

        return *this + -*other; 
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
        if ((shape[0] == 1 || shape[1] == 1 || n_dim == 1) && other->n_dim > 1) {
            broadcast(other);
        }
        if (n_dim > 1 && (other->shape[0] == 1 || other->shape[1] == 1 || other->n_dim == 1)) {
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

        if ((shape[0] == 1 || shape[1] == 1 || n_dim == 1) && other->n_dim > 1) {
            printf("\nbroadcast 1");
            broadcast(other);
        }
        if (n_dim > 1 && (other->shape[0] == 1 || other->shape[1] == 1 || other->n_dim == 1)) {
            printf("\nbroadcast 2");
            other->broadcast(this);
        }



        if (n_dim != other->n_dim) {
            printf("\nERRO: Divisão de tensores de diferentes dimensões.");
            this->print("$Div 1");
            other->print("$Div 2");
            exit(1);
        } 
        for (int i = 0; i < n_dim; i++) {
            if (shape[i] != other->shape[i]) {
                printf("\nERRO: Divisão de tensores de formatos diferentes.");
                this->print("Div 1");
                other->print("Div 2");
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

        if (n_dim == 1 && other->n_dim == 1) {
            // vetor x vetor
            return (*this * other);
        }

        if (n_dim == 1 && other->n_dim > 1) {
            // new dim - shape x,1
            if (shape[0] != other->shape[1]) {
                printf("\nERRO: Produto de matriz e vetor incompatível.");
                printf("\n1 dim, 2 dim");
                printf("\n(%d) x (%d, %d)", shape[0], other->shape[0], other->shape[1]);
                exit(1);
            }

            int result_size = other->shape[0];
            float* result_data = new float[result_size];

            // matriz x vetor
            for (int i = 0; i < result_size; i++) {
                result_data[i] = 0.0;
                for (int j = 0; j < other->shape[1]; j++) {
                    result_data[i] += other->data[j * other->strides[0] + i] * data[j];
                }
            }
            
            int result_shape[] = {result_size};
            Tensor* result = new Tensor(result_data, result_shape, 1, other->requires_grad);
            if (requires_grad) {
                result->grad_fn = new MatMul(this, other);
            }
            return result;
        }


        if (n_dim > 1 && other->n_dim == 1) {

            if (other->shape[0] != shape[1]) {
                printf("\nERRO: Produto de matriz e vetor incompatível.");
                printf("\n2 dim, 1 dim");
                exit(1);
            }

            int result_size = shape[0];
            float* result_data = new float[result_size];

            // matriz x vetor
            for (int i = 0; i < result_size; i++) {
                result_data[i] = 0.0;
                for (int j = 0; j < shape[1]; j++) {
                    result_data[i] += data[j * strides[0] + i] * other->data[j];
                }
            }
            int result_shape[] = {result_size};
            Tensor* result = new Tensor(result_data, result_shape, 1, requires_grad);
            if (requires_grad) {
                result->grad_fn = new MatMul(this, other);
            }
            return result;
        }

        if (shape[1] != other->shape[0]) {
            printf("\nERRO: Produto de tensores de formatos diferentes. (%d - %d)", shape[1], other->shape[0]);
            //this->t_();
            this->print("THIS");
            other->print("OTHER");
            exit(1);
        }
        // X            W         A
        // (32,64) x (64,27) = (32,27)
        // (a,p) x (p,b) = (a,b) 
        // assumir 2 dims

        int m = shape[0];
        int n = other->shape[1];
        int p = shape[1];
    
        float* new_data = new float[m * n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float  sum = 0.0;
                for (int k = 0; k < p; k++) {
                    sum += data[i * strides[0] + k] * other->data[k * other->strides[0] + j * other->strides[1]];
                }
                new_data[i * n + j] = sum;
            }
        }
        int new_shape[] = {m, n};
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

        return (*this * -1);
    }


    Tensor* Tensor::index(float* x, float* y, int l){
        int* pos = new int[l];
        float* new_data = new float[l];
        for (int i = 0; i < l; i++) {
            int idx = (int) (y[i] + (int) x[i] * strides[0]); 
            printf("\n (%d, %d) >>> %d -   %f", (int)x[i], (int)y[i],idx, data[idx]);
            new_data[i] = data[idx];
            pos[i] = idx;
        }

        int new_shape[1] = {l};
        Tensor* result = new Tensor(new_data, new_shape, 1, requires_grad);
        printf("\n\n");
        if (requires_grad) {
            result->grad_fn = new Indexing(this, pos, l);
        }
        return result;
    }