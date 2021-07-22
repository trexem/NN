#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include <iostream>
#include <vector>
#include <assert.h>
#include <memory>
#include "Matrix.h"
#include "Layer.h"


class NeuralNetwork {
private:
	int m_topology_size{0};
	std::vector<int> m_topology;
	std::vector<Layer> m_layers;
	std::vector<Matrix> m_weight_matrices;
	std::vector<double> m_input;
	std::vector<double> m_target;
	double m_error;
	std::vector<double> m_errors;
	std::vector<double> m_historical_errors;
	std::vector<double> m_derived_errors;
public:
	NeuralNetwork(std::vector<int> t_topology) {
		m_topology = t_topology;
		m_topology_size = m_topology.size();
		m_bias = 1;
		m_learning_rate = .01;
		m_momentum = 1;
		for (int i = 0; i < m_topology_size; i++) {
			if (i == m_topology_size - 1 || i == 0) {
				auto lay = std::make_unique<Layer>(m_topology.at(i), SFMX);
				m_layers.push_back(*lay);
			} else {
				auto lay = std::make_unique<Layer>(m_topology.at(i), PRELU);
				m_layers.push_back(*lay);
			}
		}
		for (int i = 0; i < m_topology_size - 1; i++) {
			if (i < m_topology_size - 2) {
				auto mat = std::make_unique<Matrix>(m_topology.at(i), m_topology.at(i + 1), true, true);
				m_weight_matrices.push_back(*mat);
			} else {
				auto mat = std::make_unique<Matrix>(m_topology.at(i), m_topology.at(i + 1), true);
				m_weight_matrices.push_back(*mat);
			}
		}
	}

	    //Functions
	void printToConsole() {
		for (int i = 0; i < m_layers.size(); i++) {
			std::cout << "Layer: " << i << std::endl;
			if (i == 0) {
				auto mat = m_layers.at(i).matrixifyVals();
				mat.printToConsole();
			} else {
				auto mat = m_layers.at(i).matrixifyActiveVals();
				mat.printToConsole();
			}
			std::cout << "=======================" << std::endl;
			if (i != m_layers.size() - 1) {
				std::cout << "Weight Matrix: " << std::endl;
				m_weight_matrices.at(i).printToConsole();
			}
		}
	}
	void feedForward() {
		for (int i = 0; i < m_layers.size() - 1; i++) {
			auto mat_a = std::make_unique<Matrix>(1, getLayerSize(i), false);
		}
	}

	    //Getters
	int getLayerSize(int t_i) {
		return m_layers.at(t_i).getSize();
	}

	    //variables
	double m_bias{1};
	double m_momentum{0};
	double m_learning_rate{0};
};
#endif
