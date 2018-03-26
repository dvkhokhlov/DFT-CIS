#ifndef INTEGRALS_INCLUDED_H
#define INTEGRALS_INCLUDED_H

#include "libint2.hpp"
#include <Eigen/Dense>

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
	calc_1body_ints (libint2::Operator op, 
					const std::vector<libint2::Shell>& obs,
					const std::vector<libint2::Atom>& atoms = std::vector<libint2::Atom>()
					);

#endif