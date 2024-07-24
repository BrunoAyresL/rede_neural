#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "function.h"
#include <random>
#include <ctime>

void Tensor::backward(Tensor* grad_output = NULL) {
        if (requires_grad) {
            if (grad_output == NULL) {
                grad_output = tensor_fill(1.0, shape, n_dim, false);
                grad = tensor_fill(1.0, shape, n_dim, false);
            }
            if (grad == NULL) {
                grad = tensor_fill(0.0, shape, n_dim, false);
            }
            //grad_output->print("out");
            //this->print("this out");
            //std::cout << op << std::endl;
            grad_fn->backward(grad_output);
        }
    }


Tensor* tensor_rand(int* shape, int n_dim, bool req_grad = false) {
    int size = 1;
    for (int i = 0; i < n_dim; i++) {
        size *= shape[i];
    }
    float* data = new float[size];

    // random
    std::random_device rd;
    std::mt19937 gen(rd());
    //std::normal_distribution<> distrib(-1.0, 1.0);
    std::uniform_real_distribution<> distrib(-0.5, 0.5);

    for (int i = 0; i < size; i++) {
        data[i] = (float) distrib(gen);
    }
    Tensor* t = new Tensor(data, shape, n_dim, req_grad);
    t->op = "Rand";
    return t;
}

Tensor* tensor_range(int first, int last, bool req_grad = false) {
    int size = last - first;
    float* data = new float[size];
    for (int i = 0; i < size; i++) {
        data[i] = (float) first + i;
    }
    int shape[1] = {size}; 
    Tensor* t = new Tensor(data, shape, 1, req_grad);
    if (req_grad) t->op = "Range";
    else t->op = "Grad";
    return t;
}

Tensor* tensor_fill(float x, int* shape, int n_dim, bool req_grad = false) {
    int size = 1;
    for (int i = 0; i < n_dim; i++) {
        size *= shape[i];
    }
    float* data = new float[size];
    for (int i = 0; i < size; i++) {
        data[i] = x;
    }
    Tensor* t = new Tensor(data, shape, n_dim, req_grad);
    if (req_grad) t->op = "Fill";
    else t->op = "Grad";
    return t;
}

Tensor* Tensor::t() {
    if (size == 0) {
        return this;
    }

    if (n_dim == 1) {
        int new_shape[2] = {shape[0], 1};
        Tensor* result = new Tensor(data, new_shape, 2, requires_grad);
        if (requires_grad) {
            result->op = "Transpose";
            result->grad_fn = new Transpose(this);
        }

        return result;
    }

    float* new_data = new float[size];
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; ++j) {
            new_data[j * shape[0] + i] = data[i * strides[0] + j * strides[1]];
        }
    }  
    int new_shape[] = {shape[1], shape[0]};
    Tensor* result = new Tensor(new_data, new_shape, n_dim, requires_grad);
    if (requires_grad) {
        result->op = "Transpose";
        result->grad_fn = new Transpose(this);
    }
    return result;
}

void Tensor::t_() {
    if (n_dim != 2) {
        printf("\nNúmero de dimensões deve ser 2 para transpor.");
    }
    float* new_data = new float[size];

    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            new_data[j * shape[0] + i] = data[i * strides[0] + j * strides[1]];
        }
    }  
    int new_shape[] = {shape[1], shape[0]};

    memcpy(data, new_data, size * sizeof(float));
    memcpy(shape, new_shape, n_dim * sizeof(int));

    int stride = 1;
    for (int i = n_dim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    
}


Tensor* Tensor::abs() {
    float* new_data = new float[size];

    for (int i = 0; i < size; i++) {
        if (data[i] < 0) {
            new_data[i] = -data[i];
        } else {
            new_data[i] = data[i];
        }
    }

    Tensor* t_new = new Tensor(new_data, shape, n_dim);
    return t_new;
}
Tensor* Tensor::sqrt() {
    float* new_data = new float[size];

    for (int i = 0; i < size; i++) {
        new_data[i] = sqrtf(data[i]);
    }
    Tensor* t_new = new Tensor(new_data, shape, n_dim);
    return t_new;
}
Tensor* Tensor::tanh() {
    float* new_data = new float[size];

    for (int i = 0; i < size; i++) {
        new_data[i] = tanhf(data[i]);
    }
    Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
    if (requires_grad) {
        result->op = "Tanh";
        result->grad_fn = new Tanh(this, result);
    }
    return result;
}

Tensor* Tensor::mean() {

    if (n_dim == 2 && shape[1] == 1) {
        Tensor* result = *this->sum(0) * powf( (float) size, -1);
        if (requires_grad) {
            result->op = "Mean";
        }
        return result; 
    }

    Tensor* result = *this->sum(-1) * powf( (float) size, -1);

    if (requires_grad) {
        result->op = "Mean";
    }
    return result; 
}

Tensor* Tensor::sum(int dim = 0) {


    if (dim == 0) {
        int one_shape[2] = {1, shape[0]};
        Tensor* one = tensor_fill(1.0, one_shape, 2, requires_grad);
        Tensor* result = *one & this;  
        if (requires_grad) {
            result->op = "Sum0";
            result->grad_fn = new Sum(this, one, dim);
        }
        return result;    
    }
    if (dim == 1) {
        int one_shape[2] = {shape[1], 1};
        Tensor* one = tensor_fill(1.0, one_shape, 2, requires_grad);
        Tensor* result = *this & one;
        if (requires_grad) {
            result->op = "Sum1";
            result->grad_fn = new Sum(this, one, dim);
        }
        return result;    
    }
    
    // if: sum all
    if (n_dim > 1) {
        return this->sum(1)->sum(0);
    } else {
        return this->sum(0);
    }
    
}

Tensor* Tensor::pow(float x) {
    float* new_data = new float[size];

    for (int i = 0; i < size; i++) {
        new_data[i] = powf(data[i], x);
    }
    Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
    if (requires_grad) {
        result->op = "Pow";
        result->grad_fn = new Pow(this, x);
    }
    return result;
}

Tensor* Tensor::exp() {
    float* new_data = new float[size];

    for (int i = 0; i < size; i++) {
        new_data[i] = expf(data[i]);
    }
    Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
    if (requires_grad) {
        result->op = "Exp";
        result->grad_fn = new Exp(this, result);
    }
    return result;
}

Tensor* Tensor::log() {
    float* new_data = new float[size];

    for (int i = 0; i < size; i++) {
        new_data[i] = logf(data[i]);
    }
    Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
    if (requires_grad) {
        result->op = "Log";
        result->grad_fn = new Log(this);
    }
    return result;
}

Tensor* Tensor::softmax() {
    // softmax
    Tensor* counts = this->exp();
    Tensor* counts_sum = (counts->sum(1));
    Tensor* probs = *counts / counts_sum; 
    return probs;
}     

Tensor* Tensor::nll(Tensor* y) {
    Tensor* result = -*((*y * this->log())->sum(1))->mean();
    return result;
}

Tensor* Tensor::cross_entropy(Tensor* targets) {
    Tensor* result = this->softmax()->nll(targets);
    if (requires_grad) {
        result->op = "CE Loss";
        result->grad_fn = new CrossEntropy(this, targets);
    }
    return result;
}

Tensor* Tensor::max(int dim = 0) {
        if (dim == 1) {

        float* new_data = new float[shape[0]];
        int* pos = new int[shape[0]];
        for (int i = 0; i < shape[0]; i++) {
            int max_i = 0;
            for (int j = 0; j < shape[1]; j++) {
                int idx = j + i * strides[0];
                if (data[max_i] < data[idx]) {
                    max_i = idx;
                }
            }
            new_data[i] = data[max_i];
            pos[i] = max_i;
        }
        
        int new_shape[2] = {shape[0], 1};

        Tensor* result = new Tensor(new_data, new_shape, 2, requires_grad);
        if (requires_grad) {
            result->op = "Max1";
            result->grad_fn = new Max(this, dim, pos);
        }
        
        return result; 
    }

    int max_i = 0;
    for (int i = 0; i < size; i++) {
        if (data[max_i] < data[i]) {
            max_i = i;
        }
    }

    float* new_data = new float[size];
    int* pos = new int[1];
    pos[0] = max_i;
    for (int i = 0; i < size; i++) {
        new_data[i] = data[max_i];
    }

    Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
    result->is_scalar = true;
    if (requires_grad) {
        result->op = "Max0";
        result->grad_fn = new Max(this, dim, pos);
    }
    return result; 
}

Tensor* Tensor::one_hot(int length) {
    if (n_dim > 1) {
        printf("\nOne Hot impossível.");
    }
    int new_shape[2] = {shape[0], length};
    float* new_data = new float[size * length];
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0 ; j < length; j++) {
            if (j == (int) data[i]) new_data[j + i * length] = 1.0;
            else new_data[j + i * length] = 0.0; 
            
        }
    }
    Tensor* result = new Tensor(new_data, new_shape, 2, true);
    result->op = "Onehot";
    return result;

}


// broadcast
// faltam casos de +2 dimensões
/*
Tensor* Tensor::broadcast(Tensor* other) {
    // 0,3,2
    // 2,3,2
    if (grad != nullptr) {
        printf("\n\n BROADCAST WITH TENSOR THAT HAS GRAD");
    }

    if (n_dim == 1) {
        float* new_data = new float[other->size];
        for (int i = 0; i < other->shape[0]; i++) {
            for (int j = 0; j <other->shape[1]; j++) {

                int dest = j + i * other->strides[0];
                new_data[dest] = data[j];
            }
        }

        Tensor* result = new Tensor(new_data, other->shape, other->n_dim, requires_grad);
        result->origin = this;
        return result;
    }

    if (n_dim == 2) {

        float* new_data = new float[other->size];
        for (int i = 0; i < other->shape[0]; i++) {
            for (int j = 0; j < other->shape[1]; j++) {

                int dest = j + i * other->strides[0];
                new_data[dest] = data[i];
            }
        }
        Tensor* result = new Tensor(new_data, other->shape, other->n_dim, requires_grad);
        result->origin = this;
        return result;
    }

}
*/

Tensor* handle_broadcast(Tensor* a, Tensor* b) {

    if (a->size == 1 && b->size > 1) {
        return a->broadcast(b);
    }

    if (a->n_dim == 1 && b->n_dim == 1) {
        //printf("\caso 1");
        if (a->shape[0] < b->shape[0]) {
            return a->broadcast(b);
        }
    }
    if (a->n_dim > b->n_dim) {
        //printf("\caso 2");
        return a;
    }

    if (a->n_dim < b->n_dim) {

        if (b->shape[0] == 1) {
            return a->broadcast(b);
        }

        if (a->shape[0] == b->shape[1]) {
            return a->broadcast(b);
        } else {
            printf("\nBroadcast not possible");
            a->print("a");
            b->print("b");
            exit(1);
        }
    }

    if (a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1]) {
        //printf("\caso 4");
        return a;
    }
    if(a->shape[0] == b->shape[0] && a->shape[1] < b->shape[1]) {
        //printf("\ncaso 5:");
        return a->broadcast(b);
    }

    if (a->shape[1] == b->shape[1] && a->shape[0] < b->shape[0]) {
        return a->broadcast(b);
    }
    // 32, 100
    //  1, 100
    //printf("\ncaso 6:");
    //printf("\n%d, %d - %d, %d", a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
    return a;
}


Tensor* Tensor::broadcast(Tensor* other) {

    if (shape[0] == other->shape[0] && shape[1] == other->shape[1]) {
        printf("all correct");
    }

    if (size == 1) {
        Tensor* one = tensor_fill(1.0, this->shape, this->n_dim, requires_grad);
        Tensor* result = tensor_fill(this->data[0], other->shape, other->n_dim, requires_grad);
        if (requires_grad) {
            result->grad_fn = new Broadcast(this, one);
        }   
        return result;
    }

    if (shape[0] == 1) {
        int one_shape[2] = {other->shape[0], 1};
        Tensor* one = tensor_fill(1.0, one_shape, 2, requires_grad);
        Tensor* result = *one & this;
        if (requires_grad) {
            result->grad_fn = new Broadcast(this, one);
        }
        return result;
    }



    if (n_dim == 1) {

        if (other->shape[0] == 1) {
            int new_shape[2] = {1, shape[0]}; 
            Tensor* result = new Tensor(data, new_shape, 2, requires_grad);
            return result; 
        }

        int one_shape[2] = {1, other->shape[1]};
        Tensor* one = tensor_fill(1.0, one_shape, 2, requires_grad);
        Tensor* result = *this->t() & one;
        if (requires_grad) {
            result->grad_fn = new Broadcast(this, one);
        }
        return result;

    } else {
    
        int one_shape[2] = {1, other->shape[1]};
        Tensor* one = tensor_fill(1.0, one_shape, 2, requires_grad);
        Tensor* result = *this & one;
        if (requires_grad) {
            result->grad_fn = new Broadcast(this, one);
        }
        //printf("\n%d, %d - %d, %d", shape[0], shape[1], other->shape[0], other->shape[1]);

        return result;
    }

}


Tensor* Tensor::index(Tensor* X, Tensor* Y) {
    float* new_data = new float[X->size];
    for (int i = 0; i < X->size; i++) {
        int idx = Y->data[i] + X->data[i] * strides[0];
        new_data[i] = data[idx];
    }
    
    Tensor* result = new Tensor(new_data, X->shape, 1, requires_grad);
    if (requires_grad) {
        result->op = "Indexing";
        result->grad_fn = new Indexing(this, X, Y);
    }
    return result;
}

Tensor* Tensor::reshape(int* new_shape, int new_n_dim) {
    int new_size = 1;
    for (int i = 0; i < new_n_dim; i++) {
        new_size *= new_shape[i];
    }
    if (new_size != size) {
        printf("\nNovo formato para o tensor não coincide.");
        exit(1);
    }

    Tensor* result = new Tensor(data, new_shape, new_n_dim, requires_grad);
    if (requires_grad) {
        result->op = "Reshape";
        result->grad_fn = new Reshape(this);
    }
    return result;
}



// criar outro arquivo dps:

int multinomial(float* probs, int n_classes) {
    static std::default_random_engine generator(static_cast<unsigned>(std::time(nullptr)));
    std::discrete_distribution<> dist(probs, probs + n_classes);
    return dist(generator); 
}