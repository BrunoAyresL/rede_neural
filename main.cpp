#include "tensor.h"
#include "tensor.cpp"
#include "functions.cpp"
#include "operators.cpp"

// tudo aqui é pra teste

int main() {

    int w_shape[2] = {64,27};
    int x_shape[2] = {32,64};
    int b_shape[1] = {   27};
    Tensor* W = tensor_rand(w_shape, 2, true);
    Tensor* X = tensor_rand(x_shape, 2, true);
    Tensor* B = tensor_fill(0, b_shape, 1, true);


    Tensor* A = *(*X & W) + B;
    Tensor* T = A->tanh();
    Tensor* Loss = T->mean();
    Loss->backward();
    Loss->print("Loss");

    // gradient descent
    *W *= *W->grad * -1;

    A = *(*X & W) + B;
    T = A->tanh();
    Loss = T->mean();
    Loss->backward();
    Loss->print("Loss");

}