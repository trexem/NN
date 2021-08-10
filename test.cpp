#include <iostream>
#include <vector>
#include <memory>
#include "NeuralNetwork.hpp"

using namespace std;
//RELU 1
//SIGM 2
//SFMX 3
//PRELU 4
//SIGM2 5


int main(int argc, char **argv) {
	auto top = make_unique<vector<int> >();
	auto input = make_unique<vector<vector<double> > >();
	auto output = make_unique<vector<vector<double> > >();
	NeuralNetwork::askInitializers(top, input, output);
	    //vector<vector<double> > input{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
	    //vector<vector<double> > out{ {1, 0}, {1, 0}, {0, 1}};
	    //vector<vector<double> > out{{0}, {1}, {1}, {0}};
	auto nn = make_unique<NeuralNetwork>(move(top), false, 1, 2);
	char a, b;

	nn->m_learning_rate = .3;

	for (long i = 1; i <= 100000; i++) {
		for (int j = 0; j < input->size(); j++) {
			nn->setCurrentInput(input->at(j));
			nn->setCurrentTarget(output->at(j));
			    //nn->printToConsole();
			    //cin.get();
			nn->feedForward();
			//nn->printToConsole();

			nn->setErrors();
			nn->backPropagation(1);

		}
		if (!(i % 10000)) {
			cout << "\nEpoch: \t" << i << '\n';
			for (int j = 0; j < input->size(); j++) {
				nn->setCurrentInput(input->at(j));
				nn->setCurrentTarget(output->at(j));
				nn->feedForward();
				nn->printToConsole();
				nn->printOutput();
				cin.get();
			}
		}
	}

}
