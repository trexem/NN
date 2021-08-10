#ifndef _LAYER_H_
#define _LAYER_H_

#include <iostream>
#include <vector>
#include <memory>
#include "Matrix.hpp"

class Layer {
private:
	int m_size{0};
	std::vector<Neuron> m_neurons;
public:
	    //Constructors
	Layer() {
	}
	Layer(int t_size, bool t_is_output) {
		m_size = t_size;
		for (int r = 0; r < m_size; r++) {
			auto neuron = std::make_unique<Neuron>(0.0);
			m_neurons.push_back(*neuron);
		}
		    //bias node
		if (!t_is_output) {
			auto neuron = std::make_unique<Neuron>(1.0, RELU);
			neuron->activate();
			m_neurons.push_back(*neuron);
		}
	}

	Layer(int t_size, bool t_is_output, int t_activation_type) {
		m_size = t_size;
		for (int r = 0; r < m_size; r++) {
			auto neuron = std::make_unique<Neuron>(0.0, t_activation_type);
			m_neurons.push_back(*neuron);
		}
		    //bias node
		if (!t_is_output) {
			auto neuron = std::make_unique<Neuron>(1.0, RELU);
			neuron->activate();
			m_neurons.push_back(*neuron);
		}
	}

	    //Functions
	Matrix matrixifyVals() {
		auto mat = std::make_unique<Matrix>(1, m_neurons.size(), false);
		for (int i = 0; i < m_neurons.size(); i++) {
			mat->setValue(0, i, m_neurons.at(i).getVal());
		}
		return *mat;
	}
	Matrix matrixifyActiveVals() {
		auto mat = std::make_unique<Matrix>(1, m_neurons.size(), false);
		for (int i = 0; i < m_neurons.size(); i++) {
			mat->setValue(0, i, m_neurons.at(i).getActiveVal());
		}
		return *mat;
	}
	Matrix matrixifyDerivedVals() {
		auto mat = std::make_unique<Matrix>(1, m_neurons.size(), false);
		for (int i = 0; i < m_neurons.size(); i++) {
			mat->setValue(0, i, m_neurons.at(i).getDerivedVal());
		}
		return *mat;
	}
	Matrix getSoftMaxValues() {
		auto mat = std::make_unique<Matrix>(1, m_size, false);
		double sum = 0, max = m_neurons.at(0).getVal(), shiftmax = 0;
		for (int i = 0; i < m_size; i++) {
			if (m_neurons.at(i).getVal() > max) {
				max = m_neurons.at(i).getVal();
			}
		}
		for (int i = 0; i < m_size; i++) {
			sum += exp(m_neurons.at(i).getVal() - max);
		}
		for (int i = 0; i < m_size; i++) {
			mat->setValue(0, i, exp(m_neurons.at(i).getVal() - max) / sum);
		}
		return *mat;
	}

	    //Setters
	void setVal(int t_i, double t_val) {
		m_neurons.at(t_i).setVal(t_val);
	}

	    //Getters
	std::vector<Neuron> getNeurons() {
		return m_neurons;
	}
	int getSize() {
		return m_size;
	}
};
#endif
