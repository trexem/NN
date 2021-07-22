#ifndef _MATH_HPP_
#define _MATH_HPP_

#include <iostream>
#include "Matrix.hpp"

namespace utils
{
class Math
{
public:
	static void multiplyMatrix(Matrix *a, Matrix *b, Matrix *c) {
		for (int i = 0; i < a->getNumRows(); i++) {
			for (int j = 0; j < b->getNumCols(); j++) {
				for (int k = 0; k < b->getNumRows(); k++) {
					double p = a->getValue(i, k) * b->getValue(k, j);
					double newVal = c->getValue(i, j) + p;
					c->setValue(i, j, newVal);
				}
				    //std::cout << "multi\n" << c->getValue(i, j) << std::endl;
				c->setValue(i, j, c->getValue(i, j));
			}
		}
	};
	static void softMaxMatrix(Matrix *a, Matrix *b, Matrix *c) {
		for (int i = 0; i < a->getNumRows(); i++) {
			for (int j = 0; j < b->getNumCols(); j++) {
				double max, sum = 0, emax;
				for (int k = 0; k < b->getNumRows(); k++) {
					    //cout << "numrowsK\t" << k << endl;
					double p = a->getValue(i, k) * b->getValue(k, j);
					if (k == 0) {
						max = p;
						emax = exp(max);
					}
					if (p > max) {
						max = p;
						emax = exp(max);
					}
					sum += exp(p);
				}
				    //cout << exp(max) << "\t" << sum << endl;
				if (sum != 0)
					c->setValue(i, j, emax / sum);
				else
					c->setValue(i, j, emax);
				if (c->getValue(i, j) >= 1)
					c->setValue(i, j, .999);
			}
		}
	};
};
}

#endif
