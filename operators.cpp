#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include <math.h>
#include "functions.h"

    // operações Tensor x Tensor
    Tensor* Tensor::operator + (Tensor* other)  {

        Tensor* other_ = other;
        Tensor* this_ = this;
        this_ = handle_broadcast(this, other);
        other_ = handle_broadcast(other, this_);

        if (this_->shape != other_->shape) {
            printf("\nERRO: Soma de tensores de formatos diferentes.");
            this_->print("this_");
            other_->print("other_");
            exit(1);
        }
        

        std::vector<float> result_data(this_->size);
        for (int i = 0; i < this_->size; i++) {
            result_data[i] = this_->data[i] + other_->data[i];
        }
        std::vector<int> result_shape = this_->shape;
        Tensor* result = new Tensor(result_data, result_shape, requires_grad);
        if (requires_grad) {
            result->op = "Add";
            result->grad_fn = new Add(this_, other_);
        }
        return result;
    }

    Tensor* Tensor::operator - (Tensor* other)  {
        return *this + -*other; 
    }



    void Tensor::operator += (Tensor* other) {

        Tensor* other_ = other;
        Tensor* this_ = this;
        this_ = handle_broadcast(this, other);
        other_ = handle_broadcast(other, this_);

        if (this_->shape != other_->shape) {
            printf("\nERRO: Soma (+=) de tensores de formatos diferentes.");
            this->print("this");
            other->print("other");
            exit(1);
        }

        for (int i = 0; i < this_->size; i++) {
            this_->data[i] += other_->data[i];
        }
    }

    Tensor* Tensor::operator * (Tensor* other) {

        Tensor* other_ = other;
        Tensor* this_ = this;
        this_ = handle_broadcast(this, other);
        other_ = handle_broadcast(other, this_);

        if (this_->shape != other_->shape) {
            printf("\nERRO: Multiplicação escalar de tensores de formatos diferentes. ");
            this_->print("this");
            other_->print("other");
            exit(1);
        }

        std::vector<float> result_data(this_->size);
        for (int i = 0; i < this_->size; i++) {
            result_data[i] = this_->data[i] * other_->data[i];
        }

        std::vector<int> result_shape = this_->shape;
        Tensor* result = new Tensor(result_data, result_shape, requires_grad);
        if (requires_grad) {
            result->op = "Mul";
            result->grad_fn = new Mul(this_, other_);
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

        Tensor* other_ = other;
        Tensor* this_ = this;
        this_ = handle_broadcast(this, other);
        other_ = handle_broadcast(other, this_);

        if (this_->shape != other_->shape) {
            printf("\nERRO: Divisão de tensores de formatos diferentes.");
            this_->print("Div 1");
            other_->print("Div 2");
            exit(1);
        }
        
        std::vector<float> result_data(this_->size);
        for (int i = 0; i < this_->size; i++) {
            result_data[i] = this_->data[i] / other_->data[i];
        }

        std::vector<int> result_shape = this_->shape;
        Tensor* result = new Tensor(result_data, result_shape, requires_grad);
        if (requires_grad) {
            result->op = "Div";
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

        if ((n_dim == 1 && other->n_dim > 1) || (size == 1 && other->n_dim > 1)) {
            // new dim - shape x,1
            if (shape[0] != other->shape[0]) {
                printf("\nERRO: Produto de matriz e vetor incompatível.");
                printf("\n1 dim, 2 dim");
                printf("\n(%d) x (%d, %d)", shape[0], other->shape[0], other->shape[1]);
                this->print("this");
                other->print("other");
                exit(1);
            }
            int result_size = other->shape[1];
            std::vector<float> result_data(result_size);

            // matriz x vetor
            for (int i = 0; i < result_size; i++) {
                result_data[i] = 0.0;
                for (int j = 0; j < other->shape[0]; j++) {
                    result_data[i] += other->data[j * other->strides[0] + i] * data[j];
                }
            }
            
            std::vector<int> result_shape = {1, result_size};
            Tensor* result = new Tensor(result_data, result_shape, other->requires_grad);
            if (requires_grad) {
                result->op = "Matmul1";
                result->grad_fn = new MatMul(this, other);
            }
            return result;
        }


        if ((n_dim > 1 && other->n_dim == 1) || (other->size == 1 && n_dim > 1)) {

            if (other->shape[0] != shape[1]) {
                printf("\nERRO: Produto de matriz e vetor incompatível.");
                printf("\n2 dim, 1 dim");
                this->print("THIS");
                other->print("OTHER");
                exit(1);
            }

            int result_size = shape[0];
            std::vector<float> result_data(result_size);

            // matriz x vetor
            for (int i = 0; i < result_size; i++) {
                result_data[i] = 0.0;
                for (int j = 0; j < shape[1]; j++) {
                    result_data[i] += data[j * strides[0] + i] * other->data[j];
                }
            }
            std::vector<int> result_shape = {result_size};
            Tensor* result = new Tensor(result_data, result_shape, requires_grad);
            if (requires_grad) {
                result->op = "Matmul2";
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
    
        std::vector<float> result_data(m * n);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0;
                for (int k = 0; k < p; k++) {
                    sum += data[i * strides[0] + k] * other->data[k * other->strides[0] + j * other->strides[1]];
                }
                result_data[i * n + j] = sum;
            }
        }

        std::vector<int> result_shape = {m, n};
        Tensor* result = new Tensor(result_data, result_shape, requires_grad);
        if (requires_grad) {
            result->op = "Matmul3";
            result->grad_fn = new MatMul(this, other);
        }
        return result;
    }

    // operações Tensor x float
    Tensor* Tensor::operator + (float scalar)  {

        std::vector<float> result_data(size);
        for (int i = 0; i < size; i++) {
            result_data[i] = data[i] + scalar;
        }

        std::vector<int> result_shape = shape;
        Tensor* result = new Tensor(result_data, result_shape, requires_grad);
        return result;
    }

    Tensor* Tensor::operator - (float scalar)  {
        std::vector<float> result_data(size);
        for (int i = 0; i < size; i++) {
            result_data[i] = data[i] - scalar;
        }

        std::vector<int> result_shape = shape;
        Tensor* result = new Tensor(result_data, result_shape, requires_grad);
        return result;
    }

    Tensor* Tensor::operator * (float scalar)  {
        std::vector<float> result_data(size);
        for (int i = 0; i < size; i++) {
            result_data[i] = data[i] * scalar;
        }

        std::vector<int> result_shape = shape;
        Tensor* result = new Tensor(result_data, result_shape, requires_grad);
        if (requires_grad) {
            result->op = "ScalarMul";
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
    
    Tensor* Tensor::scalar_div (float scalar) {

        std::vector<float> result_data(size);
        for (int i = 0; i < size; i++) {
            result_data[i] = scalar / data[i];
        }

        std::vector<int> result_shape = shape;
        Tensor* result = new Tensor(result_data, result_shape, requires_grad);
        return result;
    }


    Tensor* Tensor::operator-() {

        // tem que derivar isso???????????

        return (*this * -1);
    }
