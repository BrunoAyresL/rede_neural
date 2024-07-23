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

class Scalar_Mul : public Function {
    public:
        Scalar_Mul(Tensor* a, float scalar);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
        float scalar_;
};

class Div : public Function {
    public:
        Div(Tensor* a, Tensor* b);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
        Tensor* b_;
};

class MatMul : public Function {
    public:
        MatMul(Tensor* a, Tensor* b);
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

class Mean : public Function {
    public:
        Mean(Tensor* a);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
};

class Sum : public Function {
    public:
        Sum(Tensor* a, Tensor* one, int dim);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
        Tensor* one_;
        int dim_;
};

class Exp : public Function {
    public:
        Exp(Tensor* a, Tensor* result);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
        Tensor* result_;
};

class Log : public Function {
    public:
        Log(Tensor* a);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
};

class Indexing : public Function {
    public:
        Indexing(Tensor* a, Tensor* X, Tensor* Y);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
        Tensor* X_;
        Tensor* Y_;
};

class Max : public Function {
    public:
        Max(Tensor* a, int dim, int* pos);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
        int* pos_;
        int dim_;
};

class Transpose : public Function {
    public:
        Transpose(Tensor* a);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
};

class Broadcast: public Function {
    public:
        Broadcast(Tensor* a, Tensor* one);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
        Tensor* one_;
};

class CrossEntropy: public Function {
    public:
        CrossEntropy(Tensor* a, Tensor* targets);
        void backward(Tensor* grad) override;
    private: 
        Tensor* a_;
        Tensor* targets_;
};

#endif