#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "tensor.h"

int main() {
    std::ifstream inputFile("ibge-mas-10000.csv");
    std::ofstream outputFile("nomes_brasileiros.txt");
    std::string line, nome;

    if (!inputFile.is_open()) {
        std::cerr << "Erro ao abrir o arquivo de entrada." << std::endl;
        return 1;
    }

    if (!outputFile.is_open()) {
        std::cerr << "Erro ao abrir o arquivo de saída." << std::endl;
        return 1;
    }

    std::getline(inputFile, line);

    while (std::getline(inputFile, line)) {
        std::stringstream ss(line);
        std::getline(ss, nome, ',');
        nome.erase(0, 1); 
        nome.erase(nome.size() - 1);
        outputFile << nome << std::endl;
    }

    inputFile.close();
    outputFile.close();
}