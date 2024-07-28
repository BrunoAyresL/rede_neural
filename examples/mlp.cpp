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
    return 'a' + (number - 1);
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

    std::vector<float> data_xs(batch_size * block_size);
    std::vector<float> data_ys(batch_size);

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
    batch_size = j;
    // --------

    std::vector<int> x_shape = {batch_size * block_size};
    std::vector<int> y_shape = {batch_size};
    Tensor* xs = new Tensor(data_xs, x_shape, true);
    Tensor* ys = new Tensor(data_ys, y_shape, true);

    std::vector<int> c_shape = {27, 2};
    std::vector<int> w1_shape = {block_size * 2, 100};
    std::vector<int> b1_shape = {1, 100};
    std::vector<int> w2_shape = {100, 27};
    std::vector<int> b2_shape = {1, 27};
    Tensor* C  = tensor_rand(c_shape, true);
    Tensor* W1 = tensor_rand(w1_shape, true);
    Tensor* b1 = tensor_rand(b1_shape, true);
    Tensor* W2 = tensor_rand(w2_shape, true);
    Tensor* b2 = tensor_rand(b2_shape, true);

    std::vector<Tensor*> parameters = {C, W1, b1, W2, b2};
    int n_params = 0;
    for (int i = 0; i < 5; i++) {
        parameters[i]->keep_grad = true;
        n_params += parameters[i]->size;
    }
    printf("\nnumero de parametros: %d", n_params);

    int mb_size = 64;

    std::default_random_engine generator(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_int_distribution<int> distribution(0, batch_size);
    std::vector<int> newx_shape = {mb_size * block_size};
    std::vector<int> newy_shape = {mb_size};
    std::vector<int> emb_shape = {mb_size, block_size * 2};
    float learning_rate = -0.1;
    int iterations = 50000;
    for (int iter = 0; iter < iterations; iter++) {

        std::vector<float> newx_data(mb_size * block_size);
        std::vector<float> newy_data(mb_size);
        for (int i = 0; i < mb_size; i++) {
            int ix = distribution(generator);
            for (int k = 0; k < block_size; k++) {
                newx_data[i*block_size + k] = xs->data[ix*block_size + k];
            }
            newy_data[i] = ys->data[ix];
        }

        Tensor* batch_X = new Tensor(newx_data, newx_shape, true);
        Tensor* batch_Y = new Tensor(newy_data, newy_shape, true);
        Tensor* xenc = batch_X->one_hot(27);
        Tensor* yenc = batch_Y->one_hot(27);
        //
        Tensor* emb = (*xenc & C)->reshape(emb_shape);
        Tensor* h = (*(*emb & W1) + b1)->tanh();
        Tensor* logits = *(*h & W2) + b2;
        Tensor* loss = logits->cross_entropy(yenc);
        //
        
        TensorRegistry::zero_grad();
        loss->backward();
        for (int n = 0; n < 5; n++) {
            *parameters[n] += *parameters[n]->grad * learning_rate;
        }
        if (iter % 1000 == 0) {
            learning_rate *= 0.95;
            printf("\n\n---- [ %d ] ----\nlr = %f", iter, learning_rate);
            loss->sprint("loss");
        }

        TensorRegistry::clear();
        delete batch_X;
        delete batch_Y;
        delete xenc;
        delete yenc;
        delete emb;
        delete h;
        delete logits;
        delete loss;
    }
    printf("\n Training done.");
    // final loss: 
    std::vector<int> final_emb_shape = {batch_size, block_size * 2};
    Tensor* xenc = xs->one_hot(27);
    Tensor* yenc = ys->one_hot(27);
    Tensor* emb = (*xenc & C)->reshape(final_emb_shape);
    Tensor* h = (*(*emb & W1) + b1)->tanh();
    Tensor* logits = *(*h & W2) + b2;
    Tensor* loss = logits->cross_entropy(yenc);
    loss->sprint("final loss");

    printf("\n\n");
    // sample
    float bias_factor = 1.7;
    std::vector<int> sample_emb_shape = {1, block_size * 2};
    for (int i = 0; i < 80; i++) {
        int k = 3;
        std::string out = "";
        std::vector<float> ix(3.0, 0.0);
        while (true) {
            //printf("\n");
            for (int g = 0; g < ix.size(); g++) {
                //printf("%c", itol(ix[g]));
            }
            std::vector<int> sample_x_shape = {3};
            Tensor* sample_x = new Tensor(ix, sample_x_shape, true);
            Tensor* sample_xenc = sample_x->one_hot(27);
            Tensor* emb = (*sample_xenc & C)->reshape(sample_emb_shape);
            Tensor* h = (*(*emb & W1) + b1)->tanh();
            Tensor* sample_logits = *(*h & W2) + b2;
            Tensor* sample_probs = sample_logits->softmax();
            ix[k] = multinomial(sample_probs->data, 27, bias_factor);
            if (ix[k] == 0) {
                break;
            }

            for (int l = 0; l < ix.size(); l++) {
                ix[l] = ix[l + 1];
            }
            out += itol(ix[k-1]);
        }
        out[0] = std::toupper(out[0]);
        if (out.length() > 3 && out.length() < 13) {
            std::cout << out << std::endl;
        }
    }

    int close;
    scanf("%d", &close);
    }