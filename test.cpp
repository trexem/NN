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
	vector<int> top{2, 6, 2, 1};
	vector<vector<double> > input{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
	    //vector<vector<double> > out{ {1, 0}, {1, 0}, {0, 1}};
	vector<vector<double> > out{{0}, {1}, {1}, {0}};
	auto nn = make_unique<NeuralNetwork>(top, true, 4, 5);
	char a, b;

	nn->m_learning_rate = .1;

	for (long i = 1; i <= 50000; i++) {
		for (int j = 0; j < input.size(); j++) {
			nn->setCurrentInput(input.at(j));
			nn->setCurrentTarget(out.at(j));


			nn->feedForward();
			//nn->printToConsole();
			//cin.get();
			//nn->printToConsole();

			nn->setErrors();
			nn->backPropagation(1);

		}
		if (!(i % 10000)) {
			cout << "\nEpoch: \t" << i << '\n';
			for (int j = 0; j < input.size(); j++) {
				nn->setCurrentInput(input.at(j));
				nn->setCurrentTarget(out.at(j));
				nn->feedForward();
				nn->printToConsole();
				nn->printOutput();
				cin.get();
			}
		}
	}

}
