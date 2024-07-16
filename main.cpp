#include "tensor.h"
#include "tensor.cpp"
#include "functions.cpp"

// tudo aqui é pra teste

int main() {

    float a[6] = {0.1,0.5,0.4,0.1,-0.05,-1.0};
    int shape[2] = {2,3};
    int n_dim = 2;
    float b[6] = {0.132,0.25,-0.37, 0.21,-0.9, 0.129};
    float c[6] = {3,2,1,4,5,6};

    Tensor* W = new Tensor(a,shape, n_dim, true);
    Tensor* X = new Tensor(b, shape, n_dim, true);
    Tensor* B = new Tensor(c, shape, n_dim, true);

    Tensor* C = *(*X & W) + B;
    Tensor* T = C->tanh();
    Tensor* Loss = T->mean();
    Loss->backward();
    W->print("W");
    W->grad->print("W.grad");
    Loss->print("Loss");
    Loss->grad->print("Loss.grad");




    *W *= *W->grad * -0.2;
    printf("\n");
    /*
    C = *X * W;
    R = *C + B;
    T = R->tanh();
    T->backward();
    printf("\nW: ");
    W->print();
    printf("\nW.grad: ");
    W->grad->print();
    printf("\nT: ");
    T->print(); 
    */
}