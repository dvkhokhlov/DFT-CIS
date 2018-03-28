#ifndef ALIASES_H_INCLUDED
#define ALIASES_H_INCLUDED

#include <Eigen/Dense>

using Matrix = 
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 
using Vector = Eigen::VectorXd;

#endif
