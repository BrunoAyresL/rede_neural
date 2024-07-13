#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "function.h"
#include "tensor.h"

class NullFunction : public Function {
    public:
        NullFunction();
        void backward(Tensor* grad);
    private: 
        Tensor* a_;
        Tensor* b_;
};

class Add : public Function {
    public:
        Add(Tensor* a, Tensor* b);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
        Tensor* b_;
};

class Mul : public Function {
    public:
        Mul(Tensor* a, Tensor* b);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
        Tensor* b_;
};

class Tanh : public Function {
    public:
        Tanh(Tensor* a, Tensor* result);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
        Tensor* result_;
};

class Pow : public Function {
    public:
        Pow(Tensor* a, float x);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
        float x_;
};


#endif