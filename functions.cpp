#include "functions.h"

NullFunction::NullFunction() {

}
void NullFunction::backward(Tensor* grad) {
}


Add::Add(Tensor* a, Tensor* b) {
    a_ = a;
    b_ = b;
    //printf("\nADD");
}
void Add::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }

        *a_->grad += grad;
        //a_->grad->print("Add grad a");
        a_->backward(a_->grad); 
    }
    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }
        *b_->grad += grad;
        //b_->grad->print("Add grad b");
        b_->backward(b_->grad);
    }

}


Mul::Mul(Tensor* a, Tensor* b) {
    a_ = a;
    b_ = b;
    //printf("\nMUL");
}

void Mul::backward(Tensor* grad) {
        //printf("\nMul backward: ");
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

Div::Div(Tensor* a, Tensor* b) {
    a_ = a;
    b_ = b;
    //printf("\nDIV");
}

void Div::backward(Tensor* grad) {
        //printf("\nDiv backward: ");
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }
        *a_->grad += (*grad * (1 / *b_));
        a_->backward(a_->grad);  
    }
    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }
        *b_->grad += (*grad * (*(-*a_) / b_->pow(2)));
        b_->backward(b_->grad);
    }
}

Scalar_Mul::Scalar_Mul(Tensor* a, float scalar) {
    a_ = a;
    scalar_ = scalar;
    //printf("\nScalar Multiplication");
}

void Scalar_Mul::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }

        *a_->grad += (*grad * scalar_);
        //a_->grad->print("Scalar mul grad");
        a_->backward(a_->grad);  
    }
}




MatMul::MatMul(Tensor* a, Tensor* b) {
    a_ = a;
    b_ = b;
    //printf("\nMATMUL");
}

void MatMul::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }
        *a_->grad += (*grad & (b_->t()) );
        //a_->grad->print("Matmul grad a");
        a_->backward(a_->grad);  
    }
    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }
        *b_->grad += (*(a_->t()) & grad);
        //b_->grad->print("Matmul grad b");
        b_->backward(b_->grad);
    }
}





Tanh::Tanh(Tensor* a, Tensor* result) {
    a_ = a;
    result_ = result;
    //printf("\nTANH");
}

void Tanh::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }

        *a_->grad += (*grad * (result_->pow(2)));
        //a_->grad->print("Tanh grad");
        a_->backward(a_->grad);  
    }
}

Pow::Pow(Tensor* a, float x) {
    a_ = a;
    x_ = x;
    //printf("\nPOW");
}

void Pow::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }

        *a_->grad += (*grad * (*a_ * x_)->pow(x_ - 1));
        //a_->grad->print("Pow grad");
        a_->backward(a_->grad);  
    }
}

Mean::Mean(Tensor* a) {
    a_ = a;
    //printf("\nMEAN");
}

void Mean::backward(Tensor* grad) {

    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = create_Tensor_full(0, grad->shape, grad->n_dim);
        }
        float x = 1.0 / ((float) a_->size);
        *a_->grad += (*grad * x);
        //a_->grad->print("Mean grad");
        a_->backward(a_->grad);  
    }
}