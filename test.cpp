#include <iostream>
#include <vector>
#include <memory>
#include "NeuralNetwork.h"

using namespace std;

int main(int argc, char **argv) {
	std::vector<int> top{4, 5, 2};
	auto nn = std::make_unique<NeuralNetwork>(top);
	nn->printToConsole();
}
