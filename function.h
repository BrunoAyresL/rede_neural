#ifndef FUNCTION_H
#define FUNCTION_H

class Tensor;

class Function {
    public:
        virtual ~Function() = default;
        virtual void backward(Tensor* prev) = 0;
};

#endif