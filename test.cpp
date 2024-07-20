#include "tensor.h"
#include "tensor.cpp"
#include "functions.cpp"
#include "operators.cpp"
#include "registry.h"
#include "registry.cpp"


int main() {
    int l_shape[1] = {5};
    Tensor* L = *tensor_rand(l_shape, 1, true) * 4;
    Tensor* H = *tensor_rand(l_shape, 1, true) * 4;

    int w_shape[2] = {27,27};
    int x_shape[2] = {27,5};
    int b_shape[2] = {27,1};
    Tensor* W = tensor_fill(1, w_shape, 2, true);
    W->print("W");
    Tensor* X = tensor_fill(2, x_shape, 2, true);
    X->print("X");   
    Tensor* M = (*W & X);
    M->print("M");
    Tensor* B = tensor_fill(3, b_shape, 2, true); 
    B->print("B");
    Tensor* A = *M + B;
    A->print("A");
    Tensor* Id = A->index(L, H);
    Id->print("Id");
    Tensor* Log = Id->log();
    Log->print("Log");
    Tensor* Pow = Log->pow(-1);
    Pow->print("Pow");
    Tensor* Exp = Pow->exp();
    Exp->print("Exp");
    Tensor* Sum = Exp->sum(1);
    Sum->print("Sum");
    Tensor* Loss = *Sum * 2;
    Loss->backward();

    W->grad->print("W");
    X->grad->print("X");
    B->grad->print("B");
    M->grad->print("M");
    A->grad->print("A");
    Id->grad->print("Id");
    Log->grad->print("Log");
    Pow->grad->print("Pow");
    Exp->grad->print("Exp");
    Sum->grad->print("Sum");
    Loss->grad->print("Loss");

}