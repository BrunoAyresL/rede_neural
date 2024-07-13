#include "tensor.h"
#include "tensor.cpp"
#include "functions.cpp"

int main() {

    float a[4] = {0,1,2,3};
    int shape[2] = {2,2};
    int n_dim = 2;
    float b[4] = {1,2,3,2};
    float c[4] = {1,1,1,1};

    Tensor* W = new Tensor(a,shape, n_dim, true);
    Tensor* X = new Tensor(b, shape, n_dim, true);
    Tensor* B = new Tensor(c, shape, n_dim, true);
    Tensor* C = *X * W;
    Tensor* R = *C + B;
    Tensor* T = R->tanh();

    T->backward();
    printf("\nW: ");
    W->print();
    printf("\nW.grad: ");
    W->grad->print();
    printf("\nT: ");
    T->print();
    *W *= W->grad->neg()->mul_scalar(0.2);
    printf("\n");
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
    
}