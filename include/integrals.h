#ifndef INTEGRALS_INCLUDED_H
#define INTEGRALS_INCLUDED_H

#include "libint2.hpp"
#include <Eigen/Dense>

#include "aliases.h"

Matrix compute_1body_ints (libint2::Operator op, 
					const std::vector<libint2::Shell>& obs,
					const std::vector<libint2::Atom>& atoms = std::vector<libint2::Atom>()
					);
					
Matrix compute_2body_fock(const std::vector<libint2::Shell>& shells,
                       const Matrix& D);
                       
std::vector<Matrix> compute_2body_fock_like_batch(const std::vector<libint2::Shell>& shells,
                          const std::vector<Matrix>& batch_P);
#endif
