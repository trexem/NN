#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include <iostream>
#include <vector>
#include <assert.h>
#include <memory>
#include "Layer.h"
#include "Math.cpp"


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
			if (i) {
				*mat_a = getActiveNeuronMatrix(i);
			} else{
				*mat_a = getNeuronMatrix(i);
			}
			auto mat_b = std::make_unique<Matrix>(getLayerSize(i), getLayerSize(i + 1), false);
			*mat_b = getWeightMatrix(i);
			auto mat_c = std::make_unique<Matrix>(1, getLayerSize(i + 1), false);
			::utils::Math::multiplyMatrix(&*mat_a, &*mat_b, &*mat_c);
			for (int c_index = 0; c_index < getLayerSize(i + 1); c_index++) {
				setNeuronValue(i + 1, c_index, mat_c->getValue(0, c_index));
			}
			if (i == m_layers.size() - 2) {
				*mat_c = getSoftMaxOutput();
				mat_c->printToConsole();
				for (int c_index = 0; c_index < getLayerSize(i + 1); c_index++) {
					setNeuronValue(m_layers.size() - 1, c_index, mat_c->getValue(0, c_index));
				}
			}
		}
	}

//Setters
	void setCurrentInput(std::vector<double> t_input) {
		m_input = t_input;
		for (int i = 0; i < m_input.size(); i++) {
			m_layers.at(0).setVal(i, m_input.at(i));
		}
	}
	void setNeuronValue(int t_index_layer, int t_index_neuron, double t_val) {
		m_layers.at(t_index_layer).setVal(t_index_neuron, t_val);
	}
	void setCurrentTarget(std::vector<double> t_target) {
		m_target = t_target;
	}
	void setErrors() {
		if (m_errors.size() == 0) {
			for (int qw = 0; qw < m_target.size(); qw++) {
				m_errors.push_back(0);
				m_derived_errors.push_back(0);
			}
		}
		int output_layer_index = m_layers.size() - 1;
		std::vector<Neuron> output_neurons = m_layers.at(output_layer_index).getNeurons();
		m_error = 0.0;
		for (int i = 0; i < m_target.size(); i++) {
			double t = m_target.at(i);
			double y = output_neurons.at(i).getActiveVal();
			m_errors.at(i) = 0.5 * pow(abs(t - y), 2);
			m_derived_errors.at(i) = y - t;
			m_error += m_errors.at(i);
		}
		m_historical_errors.push_back(m_error);
	}
	void setErrors(std::vector<double> t_errors) {
		if (m_errors.size() == 0) {
			for (int qw = 0; qw < m_target.size(); qw++) {
				m_errors.push_back(0);
				m_derived_errors.push_back(0);
			}
		}
		for (int i = 0; i < m_target.size(); i++) {
			m_derived_errors.at(i) = m_errors.at(i);
			m_errors.at(i) = t_errors.at(i);
			m_error += m_errors.at(i);
		}
		m_historical_errors.push_back(m_error);
	}
	    //Getters
	int getLayerSize(int t_i) {
		return m_layers.at(t_i).getSize();
	}
	Matrix getNeuronMatrix(int t_i) {
		return m_layers.at(t_i).matrixifyVals();
	}
	Matrix getActiveNeuronMatrix(int t_i) {
		return m_layers.at(t_i).matrixifyActiveVals();
	}
	Matrix getDerivedNeuronMatrix(int t_i) {
		return m_layers.at(t_i).matrixifyDerivedVals();
	}
	Matrix getWeightMatrix(int t_i) {
		return m_weight_matrices.at(t_i);
	}
	Matrix getSoftMaxOutput() {
		return m_layers.at(m_topology.size() - 1).getSoftMaxValues();
	}
	double getTotalError() {
		return m_error;
	}
	std::vector<double> getErrors() {
		return m_errors;
	}

	    //variables
	double m_bias{1};
	double m_momentum{0};
	double m_learning_rate{0};
};
#endif
