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
            std::vector<int> grad_shape = shape;
            if (grad_output == NULL) {
                grad_output = tensor_fill(1.0, grad_shape, false);
                grad = tensor_fill(1.0, grad_shape, false);
            }
            if (grad == NULL) {
                grad = tensor_fill(0.0, grad_shape, false);
            }

            grad_fn->backward(grad_output);
        }
    }

Tensor* tensor_rand(std::vector<int> shape, bool req_grad = false) {
    int result_size = 1;
    for (auto dim : shape) {
        result_size *= dim;
    }

    std::vector<float> result_data(result_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-0.5, 0.5);

    for (int i = 0; i < result_size; i++) {
        result_data[i] = static_cast<float>(distrib(gen));
    }

    std::vector<int> result_shape = shape;
    auto result = new Tensor(result_data, result_shape, req_grad);
    result->op = req_grad ? "Rand" : "Grad";
    return result;
}

Tensor* tensor_range(int first, int last, bool req_grad = false) {

    int result_size = last - first;

    std::vector<float> result_data(result_size);

    for (int i = 0; i < result_size; i++) {
        result_data[i] = static_cast<float>(first + i);
    }

    std::vector<int> result_shape = {result_size}; 

    auto result = new Tensor(result_data, result_shape, req_grad);
    result->op = req_grad ? "Range" : "Grad";
    return result;
}

Tensor* tensor_fill(float x, std::vector<int> shape, bool req_grad = false) {

    int result_size = 1;
    for (auto const& dim : shape) {
        result_size *= dim;
    }

    std::vector<float> result_data(result_size, x);

    auto result = new Tensor(result_data, shape, req_grad);
    result->op = req_grad ? "Fill" : "Grad";
    return result;
}


Tensor* Tensor::t() {
    if (size == 1) {
        return this;
    }

    if (n_dim == 1) {
        std::vector<int> result_shape = {shape[0], 1};
        Tensor* result = new Tensor(data, result_shape, requires_grad);
        if (requires_grad) {
            result->op = "Transpose";
            result->grad_fn = new Transpose(this);
        }

        return result;
    }

    std::vector<float> result_data(size);
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; ++j) {
            result_data[j * shape[0] + i] = data[i * strides[0] + j * strides[1]];
        }
    }  

    std::vector<int> result_shape = {shape[1], shape[0]};

    Tensor* result = new Tensor(result_data, result_shape, requires_grad);
    if (requires_grad) {
        result->op = "Transpose";
        result->grad_fn = new Transpose(this);
    }
    return result;
}

Tensor* Tensor::tanh() {

    std::vector<float> result_data(size);
    for (int i = 0; i < size; i++) {
        result_data[i] = tanhf(data[i]);
    }
    
    std::vector<int> result_shape = shape;
    Tensor* result = new Tensor(result_data, result_shape, requires_grad);
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

        std::vector<int> one_shape = {1, shape[0]};
        Tensor* one = tensor_fill(1.0, one_shape, requires_grad);
        Tensor* result = *one & this;  
        if (requires_grad) {
            result->op = "Sum0";
            result->grad_fn = new Sum(this, one, dim);
        }
        return result;    
    }
    if (dim == 1) {
        std::vector<int> one_shape = {shape[1], 1};
        Tensor* one = tensor_fill(1.0, one_shape, requires_grad);
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
    std::vector<float> result_data(size);
    for (int i = 0; i < size; i++) {
        result_data[i] = powf(data[i], x);
    }
    std::vector<int> result_shape = shape;
    Tensor* result = new Tensor(result_data, result_shape, requires_grad);
    if (requires_grad) {
        result->op = "Pow";
        result->grad_fn = new Pow(this, x);
    }
    return result;
}

Tensor* Tensor::exp() {
    std::vector<float> result_data(size);

    for (int i = 0; i < size; i++) {
        result_data[i] = expf(data[i]);
    }
    std::vector<int> result_shape = shape;
    Tensor* result = new Tensor(result_data, result_shape, requires_grad);
    if (requires_grad) {
        result->op = "Exp";
        result->grad_fn = new Exp(this, result);
    }
    return result;
}

Tensor* Tensor::log() {
    std::vector<float> result_data(size);

    for (int i = 0; i < size; i++) {
        result_data[i] = logf(data[i]);
    }
    std::vector<int> result_shape = shape;
    Tensor* result = new Tensor(result_data, result_shape, requires_grad);
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
/*
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
*/

Tensor* Tensor::one_hot(int length) {
    if (n_dim > 1) {
        printf("\nOne Hot impossível.");
    }
    std::vector<int> result_shape = {shape[0], length};
    std::vector<float> result_data(size * length);
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0 ; j < length; j++) {
            if (j == (int) data[i]) result_data[j + i * length] = 1.0;
            else result_data[j + i * length] = 0.0; 
            
        }
    }
    Tensor* result = new Tensor(result_data, result_shape, true);
    result->op = "Onehot";
    return result;

}

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
        std::vector<int> one_shape = shape;
        std::vector<int> result_shape = other->shape;
        Tensor* one = tensor_fill(1.0, one_shape, requires_grad);
        Tensor* result = tensor_fill(this->data[0], result_shape, requires_grad);
        if (requires_grad) {
            result->grad_fn = new Broadcast(this, one);
        }   
        return result;
    }

    if (shape[0] == 1) {
        std::vector<int> one_shape = {other->shape[0], 1};
        Tensor* one = tensor_fill(1.0, one_shape, requires_grad);
        Tensor* result = *one & this;
        if (requires_grad) {
            result->grad_fn = new Broadcast(this, one);
        }
        return result;
    }



    if (n_dim == 1) {

        if (other->shape[0] == 1) {
            std::vector<int> result_shape = {1, shape[0]}; 
            Tensor* result = new Tensor(data, result_shape, requires_grad);
            return result; 
        }

        std::vector<int> one_shape = {1, other->shape[1]};
        Tensor* one = tensor_fill(1.0, one_shape, requires_grad);
        Tensor* result = *this->t() & one;
        if (requires_grad) {
            result->grad_fn = new Broadcast(this, one);
        }
        return result;

    } else {
    
        std::vector<int> one_shape = {1, other->shape[1]};
        Tensor* one = tensor_fill(1.0, one_shape, requires_grad);
        Tensor* result = *this & one;
        if (requires_grad) {
            result->grad_fn = new Broadcast(this, one);
        }
        //printf("\n%d, %d - %d, %d", shape[0], shape[1], other->shape[0], other->shape[1]);

        return result;
    }

}


Tensor* Tensor::index(Tensor* X, Tensor* Y) {
    std::vector<float> result_data(X->size);
    for (int i = 0; i < X->size; i++) {
        int idx = Y->data[i] + X->data[i] * strides[0];
        result_data[i] = data[idx];
    }
    std::vector<int> result_shape = X->shape;
    Tensor* result = new Tensor(result_data, result_shape, requires_grad);
    if (requires_grad) {
        result->op = "Indexing";
        result->grad_fn = new Indexing(this, X, Y);
    }
    return result;
}

Tensor* Tensor::reshape(std::vector<int> new_shape) {
    int new_size = 1;
    for (int i = 0; i < new_shape.size(); i++) {
        new_size *= new_shape[i];
    }
    if (new_size != size) {
        printf("\nNovo formato para o tensor não coincide.");
        exit(1);
    }

    std::vector<float> result_data = data;
    
    std::vector<int> result_shape = new_shape;
    Tensor* result = new Tensor(result_data, result_shape, requires_grad);
    if (requires_grad) {
        result->op = "Reshape";
        result->grad_fn = new Reshape(this);
    }
    return result;
}



// criar outro arquivo dps:

int multinomial(std::vector<float> probs, int n_classes, float bias_factor) {
    for (int i = 0; i < probs.size(); i++) {
        probs[i] = powf(probs[i], bias_factor);    
    }
    static std::default_random_engine generator(static_cast<unsigned>(std::time(nullptr)));
    std::discrete_distribution<> dist(probs.begin(), probs.begin() + n_classes);
    return dist(generator); 
}