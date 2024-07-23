#include "registry.h"

std::vector<Tensor*> TensorRegistry::tensors;

void TensorRegistry::add(Tensor* tensor) {
    if (tensor->requires_grad) {
        for (Tensor* exist_tensor : tensors) {
            if (exist_tensor == tensor) {
                return;
            }
        }
        tensors.push_back(tensor);
    }
}

void TensorRegistry::remove(Tensor* tensor) {
    auto it = std::find(tensors.begin(), tensors.end(), tensor);
    if (it != tensors.end()) {
        tensors.erase(it);
    } 
}

void TensorRegistry::clear() {
    auto it = tensors.begin();
    while (it != tensors.end()) {
        Tensor* tensor = *it;
        tensor->zero_grad();
        if (!tensor->keep_grad) {
            delete tensor;
        } else {
            ++it;
        }
    }
}

void TensorRegistry::zero_grad() {
    auto it = tensors.begin();
    while (it != tensors.end()) {
        Tensor* tensor = *it;
        tensor->zero_grad();
        if (!tensor->keep_grad) {
            it = tensors.erase(it);
        } else {
            ++it;
        }
    }  
    
}
int TensorRegistry::length() {
    return (int) tensors.size();
}