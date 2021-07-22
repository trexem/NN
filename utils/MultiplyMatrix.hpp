#ifndef _MULTIPLY_MATRIX_H_
#define _MULTIPLY_MATRIX_H_

#include <iostream>
#include <vector>
#include <assert.h>
#include "../Matrix.h"

namespace utils
{
class MultiplyMatrix {
public:
	    //Constructor
	MultiplyMatrix(Matrix *a, Matrix *b) {
		this->a = a;
		this->b = b;
		if (a->getNumCols() != b->getNumRows())
			assert(false);
		this->c = new Matrix(a->getNumRows(), b->getNumCols(), false);

	}

	    //Functions
	Matrix *execute() {
		for (int i = 0; i < a->getNumRows(); i++) {
			for (int j = 0; j < b->getNumCols(); j++) {
				for (int k = 0; k < b->getNumRows(); k++) {
					double p = this->a->getValue(i, k) * this->b->getValue(k, j);
					double newVal = this->c->getValue(i, j) + p;
					this->c->setValue(i, j, newVal);
				}
			}
		}
		return this->c;
	}
	//Getters

	//Setters


private:
	Matrix *a;
	Matrix *b;
	Matrix *c;

};
}
#endif
