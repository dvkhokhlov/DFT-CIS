// multi-root Davidson
// M.Crouzeix, B. Philippe, M. Sadkane // SIAM J. Sci. Comput. 15 (1994) 62-76
#ifndef DAVIDSON_INCLUDED_H
#define DAVIDSON_INCLUDED_H

#include <iostream>
#include <iomanip>
#include <ctgmath>
#include <random>
#include <chrono>
#include <tuple>
//#include "lapacke.h"
//#define EIGEN_USE_BLAS
//#define EIGEN_USE_LAPACKE
#include <Eigen/Dense>

#include "aliases.h"

namespace david_pars{
	const size_t max_davison_iter = 200;
	const size_t ntrial_per_state = 2;
	const size_t trial_space_mult = 4*ntrial_per_state;
	const size_t ntrial_per_state0 = ntrial_per_state;
	const double tolerance = 1E-5;
}

class davidson_inderect
{        
	public:
	davidson_inderect() = delete;
	davidson_inderect(const Matrix& _A, size_t _nstate, double _tol, size_t _max_iter);
	~davidson_inderect() = default;
	
	bool solve();
	Vector eigenvalues();
	Matrix eigenvectors();
	 
	private:
	const Matrix& A;
	size_t dim, nstate;
	double tol;
	size_t max_iter;
	
	size_t max_trial;
	
	Matrix evec;
	Vector eval;
	
	bool converged;
};

/*
class davidson
{
   	public:
	davidson_inderect() = delete;
	davidson_inderect(const Matrix& _A, size_t _nstate, double _tol, size_t _max_iter);
	~davidson_inderect() = default;
	
	bool solve();
	Vector eigenvalues();
	Matrix eigenvectors();
	 
	private:
	const Matrix& A;
	size_t dim, nstate;
	double tol;
	size_t max_iter;
	
	size_t max_trial;
	
	Matrix evec;
	Vector eval;
	
	bool converged;
};
*/

void dvs_example();


#endif
