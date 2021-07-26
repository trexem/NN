#include <iostream>
#include <vector>
#include <memory>
#include "NeuralNetwork.hpp"

using namespace std;

int main(int argc, char **argv) {
	vector<int> top{2, 2, 1};
	vector<vector<double> > input{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
	    //vector<vector<double> > out{{0, 0}, {1, 0}, {1, 0}, {0, 1}};
	vector<vector<double> > out{{0}, {1}, {1}, {0}};
	auto nn = make_unique<NeuralNetwork>(top, 2);
	char a, b;

	nn->m_learning_rate = 10;
	for (int k = 0; k < 5; k++) {
		for (int j = 0; j < input.size(); j++) {
			nn->setCurrentInput(input.at(j));
			nn->setCurrentTarget(out.at(j));
			for (size_t i = 0; i < 500; i++) {
				cout << "\nEpoch: \t" << i << '\n';
				nn->feedForward();
				    //nn->printToConsole();
				cout << "Errors:" << '\n';
				nn->setErrors();
				nn->backPropagation(1);
			}
		}
	}
	for (int j = 0; j < input.size(); j++) {
		nn->setCurrentInput(input.at(j));
		nn->setCurrentTarget(out.at(j));
		nn->feedForward();
		nn->printOutput();
	}
}
