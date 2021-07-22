#ifndef _MATH_HPP_
#define _MATH_HPP_

#include <iostream>
#include "Matrix.h"

namespace utils
{
class Math
{
public:
	static void multiplyMatrix(Matrix *a, Matrix *b, Matrix *c);
	static void softMaxMatrix(Matrix *a, Matrix *b, Matrix *c);
};
}

#endif
