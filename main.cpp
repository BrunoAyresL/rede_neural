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
        int batch_size = 100;

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

        printf("\nnumero de exemplos: %d", j);

        Tensor* xs = new Tensor(data_xs, xs_shape, 1, true);
        Tensor* ys = new Tensor(data_ys, xs_shape, 1, true);
        int w_shape[2] = {27, 27};
        Tensor* W = tensor_rand(w_shape, 2, true);

        float learning_rate = 0.01;

        Tensor* xenc = xs->one_hot(27);   

        Tensor* logits = (*xenc & W);
        Tensor* logits_maxes = logits->max(1);
        Tensor* norm_logits = *logits - logits_maxes; 
        // softmax
        Tensor* counts = norm_logits->exp();
        Tensor* counts_sum = counts->sum(1);
        Tensor* probs = *counts / counts_sum;
        Tensor* logprobs = probs->log();
        // cross-entropy
        Tensor* loss = -*( logprobs->index(range, ys->data, j) )->mean();
        // zero-grad
        TensorRegistry::zero_grad();
        loss->backward();
        printf("\nloss -> %f", loss->data[0]);

        for (int i = 0 ; i < 40; i++) {
            logits = (*xenc & W);
            logits_maxes = logits->max(1);
            norm_logits = *logits - logits_maxes;
        // softmax
            counts = norm_logits->exp();
            counts_sum = counts->sum(1);
            probs = *counts / counts_sum;
            logprobs = probs->log();
            // cross-entropy
            loss = -*( logprobs->index(range, ys->data, j) )->mean();
            // zero-grad
            TensorRegistry::zero_grad();
            loss->backward();
            printf("\nloss -> %f", loss->data[0]);
            
            // gradient descent
            *W += *W->grad * (-learning_rate);


            if (i % 10 == 0) {
                //W->grad->print("$W.grad");
                //printf("\nloss -> %f", loss->data[0]);
                //W->mean()->print("mean");
                //probs->index(range, ys->data, j)->print("$predicts");
            }
        }

            //counts->print("Counts");
            //counts_sum->print("Counts Sum");
            //counts->grad->print("Counts GRAD");
            //counts_sum->grad->print("Counts Sum GRAD");
            //logits->print("Logits");
            //logits->grad->print("Logits");
            W->print("W");
            W->grad->print("W");
            /*
            W->print("W");
            logits->print("Logits");
            logits_maxes->print("Logits Maxes");
            norm_logits->print("Norm Logits");
            counts->print("Counts");
            counts_sum->print("Counts Sum");
            counts_sum_inv->print("Counts Sum Inv");
            probs->print("Probs");
            logprobs->print("Log Probs");
            loss->print("Loss");

            W->grad->print("W GRAD");
            logits->grad->print("Logits GRAD");
            logits_maxes->grad->print("Logits Maxes GRAD");
            norm_logits->grad->print("Norm Logits GRAD");
            counts->grad->print("Counts GRAD");
            counts_sum->grad->print("Counts Sum GRAD");
            counts_sum_inv->grad->print("Counts Sum Inv GRAD");
            probs->grad->print("Probs GRAD");
            logprobs->grad->print("Log Probs GRAD");
            loss->grad->print("Loss GRAD");
             */



}