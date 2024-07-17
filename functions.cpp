#include "functions.h"

NullFunction::NullFunction() {

}
void NullFunction::backward(Tensor* grad) {
}


Add::Add(Tensor* a, Tensor* b) {
    a_ = a;
    b_ = b;
}
void Add::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }

        *a_->grad += grad;
        a_->backward(a_->grad); 
    }
    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }
        *b_->grad += grad;
        b_->backward(b_->grad);
    }
}


Mul::Mul(Tensor* a, Tensor* b) {
    a_ = a;
    b_ = b;
}

void Mul::backward(Tensor* grad) {

    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }

        *a_->grad += (*grad * b_);
        a_->backward(a_->grad);  
    }
    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }
        *b_->grad += (*grad * a_);
        b_->backward(b_->grad);
    }
}

Div::Div(Tensor* a, Tensor* b) {
    a_ = a;
    b_ = b;
}

void Div::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }
        *a_->grad += (*grad * (1 / *b_));
        a_->backward(a_->grad);  
    }
    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }
        *b_->grad += (*grad * (*(-*a_) / b_->pow(2)));
        b_->backward(b_->grad);
    }
}

Scalar_Mul::Scalar_Mul(Tensor* a, float scalar) {
    a_ = a;
    scalar_ = scalar;
}

void Scalar_Mul::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }

        *a_->grad += (*grad * scalar_);
        a_->backward(a_->grad);  
    }
}




MatMul::MatMul(Tensor* a, Tensor* b) {
    a_ = a;
    b_ = b;
}

void MatMul::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, a_->shape, a_->n_dim, false);
        }
        *a_->grad += (*grad & (b_->t()));
        a_->backward(a_->grad);
    }
    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = tensor_fill(0, b_->shape, b_->n_dim, false);
        }
        *b_->grad += (*(a_->t()) & grad);
        b_->backward(b_->grad);
    }
}





Tanh::Tanh(Tensor* a, Tensor* result) {
    a_ = a;
    result_ = result;
}

void Tanh::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }

        *a_->grad += (*grad * (result_->pow(2)));
        a_->backward(a_->grad);  
    }
}

Pow::Pow(Tensor* a, float x) {
    a_ = a;
    x_ = x;
}

void Pow::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }
        *a_->grad += (*grad * (*a_ * x_)->pow(x_ - 1));
        a_->backward(a_->grad);  
    }
}

Mean::Mean(Tensor* a) {
    a_ = a;
}

void Mean::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }
        float x = 1.0 / ((float) a_->size);
        *a_->grad += (*grad * x);
        a_->backward(a_->grad);  
    }
}

Sum::Sum(Tensor* a) {
    a_ = a;
}

void Sum::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }

        *a_->grad += grad;
        a_->backward(a_->grad);  
    }
}

Exp::Exp(Tensor* a) {
    a_ = a;
}

void Exp::backward(Tensor* grad) {
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }
        *a_->grad += (*grad * a_->exp());
        a_->backward(a_->grad);  
    }
}
