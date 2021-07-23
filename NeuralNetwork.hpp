#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include <iostream>
#include <vector>
#include <assert.h>
#include <memory>
#include "Layer.hpp"
#include "math.hpp"


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
	int m_out_activation{1};
	int m_hidden_activation{1};
public:
	    //Constructors
	    //We ask for the topology of the nn, the activation function for layers
	NeuralNetwork(std::vector<int> t_topology, int t_hidden_activation) {
		m_topology = t_topology;
		m_out_activation = t_out_activation;
		m_hidden_activation = t_hidden_activation;
		m_topology_size = m_topology.size();
		m_bias = 1;
		m_learning_rate = .01;
		m_momentum = 1;
		for (int i = 0; i < m_topology_size; i++) {
			auto lay = std::make_unique<Layer>(m_topology.at(i), m_hidden_activation);
			m_layers.push_back(*lay);
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
	    //Overload function to specify output layer activation function
	NeuralNetwork(std::vector<int> t_topology, int t_hidden_activation, int t_out_activation) {
		m_topology = t_topology;
		m_out_activation = t_out_activation;
		m_hidden_activation = t_hidden_activation;
		m_topology_size = m_topology.size();
		m_bias = 1;
		m_learning_rate = .01;
		m_momentum = 1;
		for (int i = 0; i < m_topology_size; i++) {
			if (i == m_topology_size - 1) {
				auto lay = std::make_unique<Layer>(m_topology.at(i), m_out_activation);
				m_layers.push_back(*lay);
			} else {
				auto lay = std::make_unique<Layer>(m_topology.at(i), m_hidden_activation);
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
	void backPropagation(int t_reward) {
		std::vector<int> output = getSoftMaxOutput().matrixMax();
		int op_max = output.at(1);

		    // Output to last Hidden Layer
		int index_output_layer = m_topology.size() - 1;
		    //Initialize gradients matrix
		auto gradients = std::make_unique<Matrix>(
			1,
			m_topology.at(index_output_layer),
			false
			);
		    //Give vaulues to gradients matrix
		for (int i = 0; i < m_topology.at(index_output_layer); i++) {
			if (i == op_max) {
				gradients->setValue(
					0,
					i,
					-t_reward / getSoftMaxOutput().getValue(0, i)
					);
			} else{
				gradients->setValue(0, i, 0);
			}
			    //We set a top por the gradients in 20, so it won't go to inf
			if (gradients->getValue(0, i) > 20) {
				gradients->setValue(0, i, 20);
			}
		}
		    //Initialize delta_weights matrix
		auto delta_weights = std::make_unique<Matrix>(
			m_topology.at(index_output_layer - 1),
			m_topology.at(index_output_layer),
			false
			);
		for (int r = 0; r < delta_weights->getNumRows(); r++) {
			for (int c = 0; c < delta_weights->getNumCols(); c++) {
				if (c == op_max) {
					delta_weights->setValue(
						r,
						c,
						-t_reward *
						(1 - getSoftMaxOutput().getValue(0, c)) *
						getActiveNeuronMatrix(index_output_layer - 1).getValue(0, r)
						);
				} else{
					delta_weights->setValue(
						r,
						c,
						-t_reward *
						(0 - getSoftMaxOutput().getValue(0, c)) *
						getActiveNeuronMatrix(index_output_layer - 1).getValue(0, r)
						);
				}
				    //We set a top por the delta_weights in 20||-20, so it won't go to inf
				if (delta_weights->getValue(r, c) > 20) {
					delta_weights->setValue(r, c, 20);
				} else if (delta_weights->getValue(r, c) < -20) {
					delta_weights->setValue(r, c, -20);
				}
			}
		}
		    //Compute new weights for lasthiddenlayer <-> outputlayer
		auto temp_new_weights = std::make_unique<Matrix>(
			m_topology.at(index_output_layer - 1),
			m_topology.at(index_output_layer),
			false
			);

		for (int r = 0; r < m_topology.at(index_output_layer - 1); r++) {
			for (int c = 0; c < m_topology.at(index_output_layer); c++) {
				double original_value = m_weight_matrices.at(index_output_layer - 1).getValue(r, c);
				double delta_value = delta_weights->getValue(r, c);
				original_value *= m_momentum;
				delta_value *= m_learning_rate;
				temp_new_weights->setValue(r, c, original_value - delta_value);
			}
		}
		std::vector<Matrix *> new_weights;
		new_weights.push_back(&*temp_new_weights);

		//Last HiddenLayer to InputLayer

		    //We iterate from lasthiddenlayer to the input layer
		for (int i = index_output_layer - 1; i > 0; i--) {
			auto p_gradients = std::make_unique<Matrix>();
			*p_gradients = *gradients;


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
