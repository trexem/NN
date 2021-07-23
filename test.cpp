#include <iostream>
#include <vector>
#include <memory>
#include "NeuralNetwork.hpp"

using namespace std;

int main(int argc, char **argv) {
	vector<int> top{2, 2, 1};
	vector<double> input{1, 1};
	auto nn = std::make_unique<NeuralNetwork>(top);
	nn->setCurrentInput(input);
	nn->feedForward();
	nn->printToConsole();
	nn->backPropagation(1);
}
