#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "function.h"


Tensor* tensor_fill(float x, int* shape, int n_dim) {
    int size = 1;
    for (int i = 0; i < n_dim; i++) {
        size *= shape[i];
    }

    float* data = new float[size];
    for (int i = 0; i < size; i++) {
        data[i] = x;
    }
    Tensor* t = new Tensor(data, shape, n_dim, false);
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
        result->grad_fn = new Tanh(this, result);
    }
    return result;
}

Tensor* Tensor::mean() {
    float* new_data = new float[1];
    new_data[0] = 0.0;
    for (int i = 0; i < size; i++) {
        new_data[i] += data[i];
    }
    new_data[0] /= size;
    int* new_shape = new int[1];
    new_shape[0] = 1;

    Tensor* result = new Tensor(new_data, new_shape, 1, requires_grad);
    if (requires_grad) {
        result->grad_fn = new Mean(this);
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
        result->grad_fn = new Pow(this, x);
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



// broadcast
// tem que refazer isso tudo, n sei fazer direito:
void Tensor::broadcast(Tensor* other) {
    // 0,3,2
    // 2,3,2

    // caso base (1)

    if (n_dim == 1) {
        size = other->size;
        n_dim = other->n_dim;
        memcpy(strides, other->strides, n_dim * sizeof(int));
        memcpy(shape, other->shape, n_dim * sizeof(int));
        for (int i = 1; i < other->size; i++) {
            data[i] = data[0];
        }
        return;
    }

    for (int i = 0; i < n_dim; i++) {
        if (shape[i] != other->shape[i] && shape[i] != 1 && other->shape[i] != 1) {
            printf("\nImpossível transmitir Tensor.");
        } 
    }

}

