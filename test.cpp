#include "tensor.h"
#include "tensor.cpp"
#include "functions.cpp"
#include "operators.cpp"
#include "registry.h"
#include "registry.cpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

std::string fileLine(const std::string& conteudo, int numLinhas) {
    std::istringstream iss(conteudo);
    std::string linha;
    std::ostringstream result;
    int count = 0;
    while (std::getline(iss, linha) && count < numLinhas) {
        result << "." << linha << ".\n";
        count++;
    }
    return result.str();
}

char itol(int number) {
    if (number == 0) {
        return '.';
    }
    return 'A' + (number - 1);
}

int ltoi(char c) {
    c = std::tolower(c);
    if (c == '.') {
        return 0;
    } 
    int number = c - 'a' + 1;
    return number;
}

int main() {
    std::ifstream namesFile("nomes_brasileiros.txt");
    std::stringstream buffer;
    std::string words;
    buffer << namesFile.rdbuf();

    words = buffer.str();
    namesFile.close();
    int n_lines = 10000;
    std::string str = fileLine(words, n_lines);

    int batch_size = 10;

    float* data_xs = new float[batch_size];
    float* data_ys = new float[batch_size];
    int j = 0;
    for (int i = 0; i < str.length(); i++) {
        if (str[i] == '\n' || str[i + 1] == '\n') continue;       
        //printf(" %c%c", str[i], str[i+1]);
        data_xs[j] = (float) ltoi(str[j]);
        data_ys[j] = (float) ltoi(str[j + 1]);
        j++;
        if (j == batch_size) {
            break;
        }   
    }
    int xy_shape[1] = {j};
    Tensor* xs = new Tensor(data_xs, xy_shape, 1, true);
    xs->print("xs");
    Tensor* ys = new Tensor(data_ys, xy_shape, 1, true);
    ys->print("ys");
    printf("\nnumero de exemplos: %d", j);   

    int w_shape[2] = {27, 27};
    Tensor* xenc = xs->one_hot(27);
    xenc->print("xenc");
    Tensor* W = tensor_rand(w_shape, 2, true);
    W->print("W");
    Tensor* logits = (*xenc & W);
    logits->print("logits");
    Tensor* logits_maxes = logits->max(1);
    logits_maxes->print("logits_maxes");
    Tensor* norm_logits = *logits - logits_maxes; 
    norm_logits->print("norm_logits");
    Tensor* counts = norm_logits->exp();
    counts->print("counts");
    Tensor* counts_sum = counts->sum(1);
    counts_sum->print("counts_sum");
    Tensor* probs = *counts / counts_sum;
    probs->print("probs");
    Tensor* logprobs = probs->log();
    logprobs->print("logprobs");

    Tensor* range_x = tensor_range(0, j, true);
    range_x->print("range_x");
    Tensor* idx_logprobs = logprobs->index(range_x, ys);
    idx_logprobs->print("idx_logprobs");
    Tensor* loss = -*(idx_logprobs)->mean();
    loss->print("loss");

    TensorRegistry::zero_grad();

    loss->backward();

    float learning_rate = -0.01;
    for (int i = 0; i < 100; i++) {
        // forward pass
        logits = (*xenc & W);
        logits_maxes = logits->max(1);
        norm_logits = *logits - logits_maxes; 
        counts = norm_logits->exp();
        counts_sum = counts->sum(1);
        probs = *counts / counts_sum;
        logprobs = probs->log();
        idx_logprobs = logprobs->index(range_x, ys);
        loss = -*(idx_logprobs)->mean();
        // zero grad
        TensorRegistry::zero_grad();
        // backward pass
        loss->backward();
        // update
        *W += *W->grad * learning_rate;
        
        loss->sprint("loss");
        printf("[%d] ", i);
    }



    /*
    W->grad->print("W");
    logits->grad->print("logits");
    logits_maxes->grad->print("logits_maxes");
    norm_logits->grad->print("norm_logits");
    counts->grad->print("counts");
    counts_sum->grad->print("counts_sum");
    probs->grad->print("probs");
    logprobs->grad->print("logprobs");
    idx_logprobs->grad->print("idx_logprobs");
    loss->grad->print("loss");
    */









    /*
    Tensor* L = tensor_range(0, 5, true);
    Tensor* H = tensor_range(0, 5, true);
    L->print("L");
    H->print("H");
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
    Tensor* Sum = Exp->sum(0);
    Sum->print("Sum");
    Tensor* Loss = *Sum * 2;
    Loss->print("Loss");
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
    */
}