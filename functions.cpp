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
            a_->grad = tensor_fill(0, a_->shape, a_->n_dim, false);
        }

        *a_->grad += grad;
        a_->backward(a_->grad); 
    }
    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = tensor_fill(0, b_->shape, b_->n_dim, false);
        }

        if (b_->grad->shape[0] != a_->shape[0] || b_->grad->shape[1] != a_->shape[1]) {
            *b_->grad += grad->sum(1);
        } else {
            *b_->grad += grad;
        }
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
            a_->grad = tensor_fill(0, a_->shape, a_->n_dim, false);
        }

        *a_->grad += (*grad * b_);
        a_->backward(a_->grad);  
    }
    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = tensor_fill(0, b_->shape, b_->n_dim, false);
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
            a_->grad = tensor_fill(0, a_->shape, a_->n_dim, false);
        }
        *a_->grad += (*grad / b_);

        a_->backward(a_->grad);  
    }

    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = tensor_fill(0, b_->shape, b_->n_dim, false);
        }   

        if (b_->grad->shape[0] != a_->shape[0] || b_->grad->shape[1] != a_->shape[1]) {
            *b_->grad += (*grad * -*(*(a_) / b_->pow(2)))->sum(1);
        } else {
            *b_->grad += (*grad * -*(*(a_) / b_->pow(2)));
        }


        
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
    //grad->print("$Matmul grad");
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, a_->shape, a_->n_dim, false);
        }
        *a_->grad += (*grad & (b_->t()));
        //a_->grad->print("$Matmul a_ grad");
        a_->backward(a_->grad);
    }
    if (b_->requires_grad) {
        if (b_->grad == NULL) {
            b_->grad = tensor_fill(0, b_->shape, b_->n_dim, false);
        }
        *b_->grad += (*(a_->t()) & grad);
        //b_->grad->print("$Matmul b_ grad");
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
        *a_->grad += (*grad * (*(a_)->pow(x_ - 1) * x_));
        a_->backward(a_->grad);  
    }
}

Mean::Mean(Tensor* a) {
    a_ = a;
}

void Mean::backward(Tensor* grad) {
    //grad->print("$mean grad");
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }
        float x = 1.0 / ((float) a_->size);
        *a_->grad += (*grad * x);
        //a_->grad->print("$mean a_ grad");
        a_->backward(a_->grad);  
    }
}

Sum::Sum(Tensor* a, int dim) {
    a_ = a;
    dim_ = dim;

}

void Sum::backward(Tensor* grad) {

    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }
        if (dim_ == 1) {
            *a_->grad += grad->sum(1);
        } else {
            *a_->grad += grad->sum(0);
        }
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
        //grad->print("$grad exp");
        *a_->grad += (*grad * a_->exp());
        //a_->grad->print("$a_ grad exp");
        a_->backward(a_->grad);  
    }
}

Log::Log(Tensor* a) {
    a_ = a;
    
}

void Log::backward(Tensor* grad) {
    //grad->print("$log grad");
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, grad->shape, grad->n_dim, false);
        }

        *a_->grad += (*grad / a_);
        //a_->grad->print("$log a_ grad");
        a_->backward(a_->grad);  
    }
}


Indexing::Indexing(Tensor* a, int* pos, int l) {
    a_ = a;
    pos_ = pos;
    l_ = l;
 
}

void Indexing::backward(Tensor* grad) {
    //a_->print("$a_ of indexing");
    //grad->print("$indexing grad");
    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, a_->shape, a_->n_dim, false);
        }
        //a_->print("a_");
        //grad->print("grad");
        for (int i = 0; i < l_; i++) {
            a_->grad->data[pos_[i]] += grad->data[i];
            //printf("\ndata[%d] += %f", pos_[i], grad->data[i]);
        }
        //a_->grad->print("$indexing a_ grad");
        a_->backward(a_->grad);  
    }
}

Max::Max(Tensor* a, int dim, int* pos) {
    a_ = a;
    dim_ = dim;
    pos_ = pos;
}

void Max::backward(Tensor* grad) {

    if (a_->requires_grad) {
        if (a_->grad == NULL) {
            a_->grad = tensor_fill(0, a_->shape, a_->n_dim, false);
        }
        if (dim_ == 0) {
            a_->grad->data[pos_[0]] = a_->data[0];
        } else {

            for (int i = 0; i < a_->shape[0]; i++) {
                a_->grad->data[pos_[i]] += a_->data[i];
            }
        }


        a_->backward(a_->grad);  
    }
}

