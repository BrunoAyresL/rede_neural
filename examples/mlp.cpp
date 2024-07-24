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
        result << "..." << linha;
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
    std::string str = fileLine(words, 10000);
    //std::cout << str << std::endl;


    int batch_size = str.length();
    int block_size = 3;

    float* data_xs = new float[batch_size * block_size];
    float* data_ys = new float[batch_size];

    int j = 0;
    for (int i = 0; i < (int)str.length(); i++) {
        for (int l = 0; l < block_size; l++) {
            if (str[i + l] == '\n' || str[i + l] == ' ') i++;
            if (str[i + block_size -1] == '.' && str[i] != '.') i++;
        }
           
        int k = 0;
        for (; k < block_size; k++) {
            data_xs[j*block_size + k] = (float) ltoi(str[i + k]);   
            //printf("%c", str[i + k]);
        }
        //printf(" = %c\n", str[i + k]);
        data_ys[j] = (float) ltoi(str[i + k]);
        j++;
        if (j == batch_size) {
            break;
        }   
    }

    printf("\nnumero de exemplos: %d", j);  

    // --------

    int x_shape[1] = {batch_size * block_size};
    int y_shape[1] = {batch_size};
    Tensor* xs = new Tensor(data_xs, x_shape, 1, true);
    Tensor* ys = new Tensor(data_ys, y_shape, 1, true);
    Tensor* xenc = xs->one_hot(27);
    Tensor* yenc = ys->one_hot(27);

    int c_shape[2] = {27, 2};
    int w1_shape[2] = {block_size * 2, 100};
    int b1_shape[2] = {1, 100};
    int w2_shape[2] = {100, 27};
    int b2_shape[2] = {1, 27};
    int emb_shape[2] = {batch_size, block_size * 2};

    Tensor* C  = tensor_rand(c_shape,  2, true);
    Tensor* W1 = tensor_rand(w1_shape, 2, true);
    Tensor* b1 = tensor_rand(b1_shape, 2, true);
    Tensor* W2 = tensor_rand(w2_shape, 2, true);
    Tensor* b2 = tensor_rand(b2_shape, 2, true);

    Tensor* parameters[5] = {C, W1, b1, W2, b2};
    int n_params = 0;
    for (int i = 0; i < 5; i++) {
        parameters[i]->keep_grad = true;
        n_params += parameters[i]->size;
    }
    printf("\nnumero de parametros: %d", n_params);

    Tensor* emb = (*xenc & C)->reshape(emb_shape, 2);
    Tensor* h = (*(*emb & W1) + b1)->tanh();
    Tensor* logits = *(*h & W2) + b2;
    Tensor* loss = logits->cross_entropy(yenc);
    TensorRegistry::zero_grad();
    loss->backward();

    float learning_rate = -0.1;
    int iterations = 10 ;
    
    // minibatch

    std::default_random_engine generator(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_int_distribution<int> distribution(0, xs.shape[0]);
    int* ix = new int[32];
    for (int i = 0; i < 32; i++) {
        ix[i] = distribution(generator);
    }

    
    for (int i = 0; i < iterations; i++) {

        emb = (*xenc & C)->reshape(emb_shape, 2);
        h = (*(*emb & W1) + b1)->tanh();
        logits = *(*h & W2) + b2;
        loss = logits->cross_entropy(yenc);
        //
        
        TensorRegistry::zero_grad();

        loss->backward();
        for (int n = 0; n < 5; n++) {
            *parameters[n] += *parameters[n]->grad * learning_rate;
        }
        
        loss->sprint("loss");
        TensorRegistry::clear();
    }



    }