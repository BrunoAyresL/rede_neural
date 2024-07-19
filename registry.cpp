#include "registry.h"

std::vector<Tensor*> TensorRegistry::tensors;

void TensorRegistry::add(Tensor* tensor) {
    if (tensor->requires_grad) {
        tensors.push_back(tensor);
    }
}

void TensorRegistry::clear() {
    tensors.clear();
}

void TensorRegistry::zero_grad() {
    for (Tensor* tensor : tensors) {
        tensor->zero_grad();
    }
}