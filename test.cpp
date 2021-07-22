#include <iostream>
#include <vector>
#include <memory>
#include "NeuralNetwork.h"

using namespace std;

int main(int argc, char **argv) {
	std::vector<int> top{2, 5, 2};
	std::vector<double> input{1, 1};
	auto nn = std::make_unique<NeuralNetwork>(top);
	nn->setCurrentInput(input);
	nn->feedForward();
	nn->printToConsole();
}
