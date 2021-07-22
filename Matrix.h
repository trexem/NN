#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include "Neuron.h"

class Matrix {
private:
	int m_num_rows{0};
	int m_num_cols{0};
	std::vector<std::vector<double> > m_values;
public:
	    //Constructors
	Matrix() {
	}
	Matrix(int t_num_rows, int t_num_cols, bool t_is_random) {
		m_num_rows = t_num_rows;
		m_num_cols = t_num_cols;
		for (int i = 0; i < m_num_rows; i++) {
			std::vector<double> col_values;
			for (int j = 0; j < m_num_cols; j++) {
				double r = 0.0;
				if (t_is_random) {
					r = getRandomNumber();
				}
				col_values.push_back(r);
			}
			m_values.push_back(col_values);
		}
	}
	Matrix(int t_num_rows, int t_num_cols, bool t_is_random, bool t_altern_positive) {
		m_num_rows = t_num_rows;
		m_num_cols = t_num_cols;
		for (int r = 0; r < m_num_rows; r++) {
			std::vector<double> col_values;
			for (int c = 0; c < m_num_cols; c++) {
				double num = 0.0;
				if (t_is_random) {
					num = getRandomNumber();
				}
				if (t_altern_positive && c % 2) {
					num = -num;
				}
				col_values.push_back(num);
			}
			m_values.push_back(col_values);
		}
	}
	    //Functions
	Matrix transpose() {
		auto mat = std::make_unique<Matrix>(m_num_cols, m_num_rows, false);
		for (int r = 0; r < m_num_rows; r++) {
			for (int c = 0; c < m_num_cols; c++) {
				mat->setValue(c, r, getValue(r, c));
			}
		}
		return *mat;
	}
	std::vector<int> matrixMax() {
		double max = 0;
		std::vector<int> max_index{0, 0};
		for (int r = 0; r < m_num_rows; r++) {
			for (int c = 0; c < m_num_cols; c++) {
				if (getValue(r, c) > max) {
					max = getValue(r, c);
					max_index.at(0) = r;
					max_index.at(1) = c;
				}
			}
		}
		return max_index;
	}
	void printToConsole() {
		for (int r = 0; r < m_num_rows; r++) {
			for (int c = 0; c < m_num_cols; c++) {
				std::cout << m_values.at(r).at(c) << "\t";
			}
			std::cout << std::endl;
		}
	}

	    //Getters
	double getValue(int t_r, int t_c) {
		return m_values.at(t_r).at(t_c);
	}
	int getNumRows() {
		return m_num_rows;
	}
	int getNumCols() {
		return m_num_cols;
	}
	double getRandomNumber() {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(0, 1);
		return dis(gen);
	}

	    //Setters
	void setValue(int t_r, int t_c, double v) {
		m_values.at(t_r).at(t_c) = v;
	}
};
#endif
