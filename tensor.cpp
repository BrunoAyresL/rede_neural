#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "function.h"
#include <random>


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
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    for (int i = 0; i < size; i++) {
        data[i] = (float) distrib(gen);
    }
    Tensor* t = new Tensor(data, shape, n_dim, req_grad);
    t->op = "Rand";
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

void Tensor::expand(int* new_shape) {
    int i = 0;
    for (; i < n_dim; i++) {
        if (shape[i] == 1) {
            break;
        }
    }
    if (i == n_dim - 1) {
        return;
    }

    strides[i] = 0;
    shape = new_shape;
}

Tensor* Tensor::t() {
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
    Tensor* result = new Tensor(new_data, new_shape, n_dim, requires_grad);
    result->op = "Transpose";
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
    float mean = 0.0;
    for (int i = 0; i < size; i++) {
        mean += data[i];
    }
    mean /= (float)size;

    float* new_data = new float[size];
    for (int i = 0; i < size; i++) {
        new_data[i] = mean;
    }

    Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
    result->is_scalar = true;
    if (requires_grad) {
        result->op = "Mean";
        result->grad_fn = new Mean(this);
    }
    return result; 
}

Tensor* Tensor::sum(int dim = 0) {
    if (dim == 1) {

        float* new_data = new float[shape[0]];

        for (int i = 0; i < shape[0]; i++) {
            float sum = 0.0;
            for (int j = 0; j < shape[1]; j++) {
                sum += data[j + i * strides[0]];
            }
            new_data[i] = sum;
        }
        int new_shape[2] = {shape[0], 1};

        Tensor* result = new Tensor(new_data, new_shape, 2, requires_grad);
        if (requires_grad) {
            result->op = "Sum1";
            result->grad_fn = new Sum(this, dim);
        }
        
        return result; 
    }

    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }

    float* new_data = new float[size];
    for (int i = 0; i < size; i++) {
        new_data[i] = sum;
    }

    Tensor* result = new Tensor(new_data, shape, n_dim, requires_grad);
    result->is_scalar = true;
    if (requires_grad) {
        result->op = "Sum0";
        result->grad_fn = new Sum(this, dim);
    }
    return result; 
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
        result->grad_fn = new Exp(this);
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

Tensor* Tensor::sin() {
 
    float* new_data = new float[size];

    // seno
    //
    //
    //

    Tensor* t_new = new Tensor(new_data, shape, n_dim);
    return t_new;
}
Tensor* Tensor::cos() {
    float* new_data = new float[size];

    // cosseno
    //
    //
    //

    Tensor* t_new = new Tensor(new_data, shape, n_dim);
    return t_new;
}
Tensor* Tensor::tan() {
    float* new_data = new float[size];

    // tangente
    //
    //
    //

    Tensor* t_new = new Tensor(new_data, shape, n_dim);
    return t_new;
}    
Tensor* Tensor::softmax() {
    float* new_data = new float[size];

    // softmax
    //
    //
    //
    //

    Tensor* t_new = new Tensor(new_data, shape, n_dim);
    return t_new;
}     

// ------------------------------------------------------------------------------------------------------------
// refazer ou apagar
float get_item(Tensor* t, int* indexes) {
    int pos = 0;
    for (int i = 0; i < t->n_dim; i++) {
        pos += indexes[i] * t->strides[i];
    }
    float result;
    result = t->data[pos];
    return result;
}
Tensor* reshape_Tensor(Tensor* t, int* new_shape, int new_n_dim) {
    int n_dim = new_n_dim;
    int* shape = (int*) malloc(n_dim * sizeof(int));

    int size = 1;
    for (int i = 0; i < n_dim; i++) {
        shape[i] = new_shape[i];
        size *= shape[i];
    }

    if (size != t->size) {
        printf("\nTamanho do novo Tensor diferente do original.");
        exit(1);
    }

    float* data = (float*) malloc(size * sizeof(float));;
    if (data == NULL || shape == NULL) {
        printf("\n--- ERRO: Falha ao usar reshape ---\n");
        exit(1);
    }

    // copiar (cpu)
    memcpy(data, t->data, size * sizeof(float));

    Tensor* t_new = new Tensor(data, shape, n_dim);
    return t_new;
}

// onehot -> 5 -> [0,0,0,0,0,1]

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

