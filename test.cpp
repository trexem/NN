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
void askInitializers(unique_ptr < vector<int> > & p_topology,
                     unique_ptr <vector<vector<double> > > & p_input,
                     unique_ptr <vector<vector<double> > > & p_output) {
	int layers, temp, test_cases;
	double test;
	    //Topology
	cout << "How many Layers?" << '\t';
	cin >> layers;
	for (int i = 1; i <= layers; i++) {
		if (i == 1) {
			cout << "How many input nodes?" << '\t';
		} else if (i < layers) {
			cout << "How many nodes for layer " << i << "?" << '\t';
		} else {
			cout << "How many output nodes?" << '\t';
		}
		cin >> temp;
		p_topology->push_back(temp);
	}
	    //Test inputs and outputs
	cout << "How many test cases?" << '\t';
	cin >> test_cases;
	vector<double> temp_vector;
	for (int i = 1; i <= test_cases; i++) {
		cout << "Test case " << i << '\n';
		for (int j = 0; j < p_topology->at(0); j++) {
			cout << "Input " << j << ":\t";
			cin >> test;
			temp_vector.push_back(test);
		}
		p_input->push_back(temp_vector);
		temp_vector.clear();
		for (int j = 0; j < p_topology->at(p_topology->size() - 1); j++) {
			cout << "Output " << j << ":\t";
			cin >> test;
			temp_vector.push_back(test);
		}
		p_output->push_back(temp_vector);
		temp_vector.clear();
	}
}

int main(int argc, char **argv) {
	auto top = make_unique<vector<int> >();
	auto input = make_unique<vector<vector<double> > >();
	auto output = make_unique<vector<vector<double> > >();
	askInitializers(top, input, output);
	    //vector<vector<double> > input{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
	    //vector<vector<double> > out{ {1, 0}, {1, 0}, {0, 1}};
	    //vector<vector<double> > out{{0}, {1}, {1}, {0}};
	auto nn = make_unique<NeuralNetwork>(move(top), true, 4, 2);
	char a, b;

	nn->m_learning_rate = .1;

	for (long i = 1; i <= 50000; i++) {
		for (int j = 0; j < input->size(); j++) {
			nn->setCurrentInput(input->at(j));
			nn->setCurrentTarget(output->at(j));


			nn->feedForward();
			//nn->printToConsole();
			//cin.get();
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
