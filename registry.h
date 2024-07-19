#ifndef REGISTRY_H
#define REGISTRY_H

#include <vector>

#include "tensor.h"

class TensorRegistry {

private:

    static std::vector<Tensor*> tensors;

public:

    static void add(Tensor* tensor);
    static void clear();
    static void zero_grad();

};


#endif
