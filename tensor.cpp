#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "function.h"



// funções derivadas




// ------------------------------------------------------------------------------------------------------------
// criação de Tensores:
Tensor* create_Tensor_full(float x, int* shape, int n_dim) {
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

// ------------------------------------------------------------------------------------------------------------
// funções in-place:

    void Tensor::add_scalar(float x) {
        for (int i = 0; i < size; i++) {
           data[i] += x;
        }   
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
// ------------------------------------------------------------------------------------------------------------
// funções padrão
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
    Tensor* Tensor::neg() {
        float* new_data = new float[size];

        for (int i = 0; i < size; i++) {
            new_data[i] = -data[i];
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
    Tensor* Tensor::mul_scalar(float x) {
        float* new_data = new float[size];

        for (int i = 0; i < size; i++) {
            new_data[i] = x * data[i];
        }
        Tensor* t_new = new Tensor(new_data, shape, n_dim);
        return t_new;
    }
    Tensor* Tensor::mean() {
        float* new_data = new float[1];
        new_data[0] = 0;
        for (int i = 0; i < size; i++) {
            new_data[i] += data[i];
        }
        new_data[0] /= size;

        int* new_shape = new int[1];
        new_shape[0] = 1;

        Tensor* t_new = new Tensor(new_data, new_shape, 1);
        return t_new; 
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

        for (int i = 0; i < size; i++) {
            new_data[i] = sinf(data[i]);
        }
        Tensor* t_new = new Tensor(new_data, shape, n_dim);
        return t_new;
    }
    Tensor* Tensor::cos() {
        float* new_data = new float[size];

        for (int i = 0; i < size; i++) {
            new_data[i] = cosf(data[i]);
        }
        Tensor* t_new = new Tensor(new_data, shape, n_dim);
        return t_new;
    }
    Tensor* Tensor::tan() {
        float* new_data = new float[size];

        for (int i = 0; i < size; i++) {
            new_data[i] = tanf(data[i]);
        }
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

