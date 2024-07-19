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

// tudo aqui é pra teste

int main() {

    // abrir arquivo

        std::ifstream namesFile("nomes_brasileiros.txt");
        std::stringstream buffer;
        std::string words;
        buffer << namesFile.rdbuf();

        words = buffer.str();
        namesFile.close();
        int n_lines = 10000;
        std::string str = fileLine(words, n_lines);
        //int batch_size = str.size() / 100;
        int batch_size = str.size() / 500;

        float* data_xs = new float[batch_size];
        float* data_ys = new float[batch_size];
        int j = 0;
        for (int i = 0; i < batch_size - 1; i++) {
            if (str[i] == '\n' || str[i + 1] == '\n') continue;        
            //printf(" %c%c", str[i], str[i+1]);
            data_xs[j] = (float) ltoi(str[i]);
            data_ys[j] = (float) ltoi(str[i + 1]);

            j++;
        }
        int xs_shape[1] = {j};

        // j é o tamanho do input
        float range[j];
        for (int i = 0; i < j; i++) {
            range[i] = (float) i;
        }

        printf("\nnumero de elementos: %d", j);

        Tensor* xs = new Tensor(data_xs, xs_shape, 1, true);
        Tensor* ys = new Tensor(data_ys, xs_shape, 1, true);
        int w_shape[2] = {27, 27};
        Tensor* W = tensor_fill(1.0, w_shape, 2, true);

        Tensor* xenc = xs->one_hot(27);   
        
        Tensor* logits = (*xenc & W);
        Tensor* counts = logits->exp();
        Tensor* counts_sum = counts->sum(1);
        Tensor* probs = *counts / counts_sum;

        Tensor* loss = -*((probs->index(range, ys->data, j)->log()))->mean();
        loss->backward();
        printf("\nloss -> %f", loss->data[0]);
        *W += *W->grad * (-0.2);

        for (int i = 0 ; i < 5; i++) {
            xenc = xs->one_hot(27);
            logits = (*xenc & W);
            counts = logits->exp();
            counts_sum = counts->sum(1);
            probs = *counts / counts_sum;
            loss =-*((probs->index(range, ys->data, j)->log()))->mean();

            TensorRegistry::zero_grad();

            loss->backward();  
            printf("\nloss -> %f", loss->data[0]);
            *W += *W->grad * (-1);
        }
    W->print("$W");
    W->grad->print("W grad");
    probs->index(range, ys->data, j)->print("idx");
    //logits->print("$logits");
    //logits->grad->print("$logits");

    //for (int i = 0; i < 100; i++) {
    //    printf("\n%c\t%c", itol((int)xs->data[i]), itol((int)ys->data[i]));
    //}
    int k = 0;
    for (int i = 0; i < W->size; i++) {
        if (W->grad->data[i] >= 0.001 || W->grad->data[i] < -0.001) {
            k++;
        }
    }
    printf("W: %d  non-zero = %d", W->size, k);


    //ys->print("$ys");
    //W->print("$W");
    //W->grad->print("$W grad");



}