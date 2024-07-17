#include "tensor.h"
#include "tensor.cpp"
#include "functions.cpp"
#include "operators.cpp"

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

int ltoi(char c) {
    printf("  c: %c", c);
    c = std::tolower(c);
    if (c == '.') {
        printf("\t 0");
        return 0;
    } 
    int number = c - 'a' + 1;
    printf("\t %d", number);
    return number;
}

// tudo aqui é pra teste

int main() {

    // abrir arquivo

    std::ifstream namesFile("nomes_brasileiros.txt");
    std::stringstream buffer;
    std::string words;
    buffer << namesFile.rdbuf();

    words = buffer.str();
    namesFile.close();
    int n_lines = 50;
    std::string str = fileLine(words, n_lines);
    int batch_size = 8;

    float* data_xs = new float[batch_size];
    float* data_ys = new float[batch_size];
    int j = 0;
    for (int i = 0; i < batch_size; i++) {
        if (str[i] == '\n' || str[i + 1] == '\n') continue;        
        printf("\n%c%c", str[i], str[i+1]);
        data_xs[j] = (float) ltoi(str[i]);
        data_ys[j] = (float) ltoi(str[i + 1]);

        j++;
    }
    int xs_shape[1] = {j};
    Tensor* xs = new Tensor(data_xs, xs_shape, 1, false);
    Tensor* ys = new Tensor(data_ys, xs_shape, 1, false);
    xs->print("xs");
    Tensor* xenc = xs->one_hot(27);   
    //xenc->print("$xenc");
    ys->print("ys");
    Tensor* yenc = ys->one_hot(27);   
    //yenc->print("$yenc");
    int w_shape[2] = {27, 27};
    Tensor* W = tensor_rand(w_shape, 2, true);
    Tensor* logits = (*xenc & W);
    // SOFTMAX:
    Tensor* counts = logits->exp();
    Tensor* probs = *counts / counts->sum(1);
    probs->print("$probs");











    /*
    int w_shape[2] = {64,27};
    int x_shape[2] = {32,64};
    int b_shape[1] = {   27};
    Tensor* W = tensor_rand(w_shape, 2, true);
    Tensor* X = tensor_rand(x_shape, 2, true);
    Tensor* B = tensor_fill(0, b_shape, 1, true);


    Tensor* A = *(*X & W) + B;
    Tensor* Out = A->tanh();
    Tensor* Target = tensor_fill(1.0, Out->shape, Out->n_dim, true);
    Tensor* M = (*Target - Out)->pow(2);
    Tensor* Loss = M->mean();
    Loss->backward();
    Loss->print("Loss");

   
    for (int i = 0; i < 10; i++) {
     // gradient descent    
    *W += *W->grad * (-100);
    
    A = *(*X & W) + B;
    Out = A->tanh();
    M = (*Out - Target)->pow(2);
    Loss = M->mean();
    Loss->backward();
    Loss->print("Loss");
    }
    */
}