#include "functions.h"

NullFunction::NullFunction() {

}
void NullFunction::backward(Tensor* grad) {
}


Add::Add(Tensor* a, Tensor* b) {
    a_ = a;
    b_ = b;
    printf("\nADD");
}
void Add::backward(Tensor* grad) {
    printf("\nAdd backward: ");
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }

        *a_->grad += grad;
        a_->backward(a_->grad); 
    }
    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }
        *b_->grad += grad;
        b_->backward(b_->grad);
    }

}


Mul::Mul(Tensor* a, Tensor* b) {
    a_ = a;
    b_ = b;
    printf("\nMUL");


}

void Mul::backward(Tensor* grad) {
        printf("\nMul backward: ");
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }

        *a_->grad += (*grad * b_);
        a_->backward(a_->grad);  
    }
    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }
        *b_->grad += (*grad * a_);
        b_->backward(b_->grad);
    }
}

Tanh::Tanh(Tensor* a, Tensor* result) {
    a_ = a;
    result_ = result;
    printf("\nTANH");
}

void Tanh::backward(Tensor* grad) {
    printf("\nTANH backward: ");
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }

        *a_->grad += (*grad * (result_->pow(2)));
        a_->backward(a_->grad);  
    }
}

Pow::Pow(Tensor* a, float x) {
    a_ = a;
    x_ = x;
    printf("\nPOW");
}

void Pow::backward(Tensor* grad) {
    printf("\nPOW backward: ");
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }

        *a_->grad += (*grad * (a_->mul_scalar(x_))->pow(x_ - 1));
        a_->backward(a_->grad);  
    }
}