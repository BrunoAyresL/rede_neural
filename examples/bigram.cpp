#include "../tensor.h"
#include "../tensor.cpp"
#include "../functions.cpp"
#include "../operators.cpp"
#include "../registry.h"
#include "../registry.cpp"

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
        result << "." << linha;
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
    if (!isalpha(c)) {
        return 0;
    } 
    c = std::tolower(c);
    int number = c - 'a' + 1;
    return number;
}

int main() {
    std::ifstream namesFile("../nomes_brasileiros.txt");
    std::stringstream buffer;
    std::string words;
    buffer << namesFile.rdbuf();

    words = buffer.str();
    namesFile.close();
    int n_lines = 10000;
    std::string str = fileLine(words, n_lines);
    //std::cout << str << std::endl;
    int batch_size = str.length();

    float* data_xs = new float[batch_size];
    float* data_ys = new float[batch_size];
    int j = 0;
    for (int i = 0; i < (int)str.length(); i++) {
        if (str[i] == '\n' || str[i + 1] == '\n' || str[i] == ' ' || str[i + 1] == ' ') continue;   
        //printf(" %c%c", str[i], str[i+1]);
        data_xs[j] = (float) ltoi(str[j]);
        data_ys[j] = (float) ltoi(str[j + 1]);
        //printf(" (%f, %f)", data_xs[j], data_xs[j+1]);
        j++;
        if (j == batch_size) {
            break;
        }   
    }

    int xy_shape[1] = {batch_size};

    Tensor* xs = new Tensor(data_xs, xy_shape, 1, true);
    //xs->print("xs");
    Tensor* ys = new Tensor(data_ys, xy_shape, 1, true);
    //ys->print("ys");
    printf("\nnumero de exemplos: %d", j);   
    
    int w_shape[2] = {27, 27};
    Tensor* xenc = xs->one_hot(27);
    Tensor* yenc = ys->one_hot(27);
    //xenc->print("xenc");
    Tensor* W = tensor_rand(w_shape, 2, true);

    
    // tem que automatizar isso (criar optimizer)
    W->keep_grad = true;
    xenc->keep_grad = true;
    yenc->keep_grad = true;

    Tensor* logits = *xenc & W;
    //W->print("W")
    Tensor* loss = logits->cross_entropy(yenc);
    loss->backward();

    float learning_rate = -40.0;
    int iterations = 200;

    for (int i = 0; i < iterations; i++) {
        logits = *xenc & W;
        loss = logits->cross_entropy(yenc);
        TensorRegistry::zero_grad();
        loss->backward();
        *W += *W->grad * learning_rate;
        loss->sprint("loss");
        TensorRegistry::clear();
    }
    printf("\n\nFINAL LOSS: %f\n\n\n", loss->data[0]);
    // sample
    for (int i = 0; i < 50; i++) {
        int k = 0;
        std::string out = "";
        int ix = 0;
        while (true) {
            int sample_x_shape[1] = {1};
            Tensor* sample_x = tensor_fill(ix, sample_x_shape, 1, true);
            Tensor* sample_xenc = sample_x->one_hot(27);
            Tensor* sample_logits = *sample_xenc & W;
            Tensor* sample_probs = sample_logits->softmax();
            ix = multinomial(sample_probs->data, 27);
            if (ix == 0 || k > 10) {
                break;
            }
            k++;
            out += itol(ix);
        }
        if (k > 2){
            std::cout << out << std::endl;
        }
        
    }


}