#include "davidson.h"

using Matrix = 
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector =
	Eigen::VectorXd;
		
// indirect Davidson
davidson_inderect::davidson_inderect(const Matrix& _A, size_t _nstate, 
	double _tol=david_pars::tolerance, size_t _max_iter=david_pars::max_davison_iter) : 
	A(_A), dim(A.cols()), nstate(_nstate), 
	tol(_tol), max_iter(_max_iter), max_trial(david_pars::trial_space_mult*_nstate),
	evec(A.cols(), _nstate), eval(_nstate)
{
	assert(A.cols() == A.rows() && "davidson(): matrix should be square");
}
	
bool davidson_inderect::solve()
{
	//initial guess
	size_t ntrial = nstate*david_pars::ntrial_per_state0;
	Matrix V(dim, ntrial);	
	for(size_t i = 0; i < ntrial; ++i)
		V(i, i) = 1.;
	
	// Ritz vectors, residuals, and trial vectors
	Matrix x(dim, ntrial);
	Matrix r(dim, ntrial);
	Matrix t(dim, ntrial);
	
	size_t k = 1;
	size_t nvec_used = ntrial;
	do{
		// compute Wk = A*V
		Matrix W = A*V;
		// compute Rayleigh matrix H = Vt*W
		Matrix H = V.transpose()*W;

		// diagonalize 
		Eigen::SelfAdjointEigenSolver<Matrix> R_eigensolver(H);
		auto y = R_eigensolver.eigenvectors();
		auto lambda = R_eigensolver.eigenvalues();
		
		// compute Ritz vectors
		for(size_t i = 0; i < ntrial; ++i){
			x.col(i) = V*y.col(i);
		}
		
		// compute residuals
		converged = true;
		std::cout << "niter = " << k; 
		std::cout << "; vectors used = " << nvec_used << std::endl;
		for(size_t i = 0; i < ntrial; ++i){
			r.col(i) = lambda[i]*x.col(i) - W*y.col(i);
			
			if(i < nstate){
				auto residue = r.col(i).norm()/sqrt(V.cols());
				if(residue > tol) converged = false;
				std::cout << "state " << i << "; residue = " << residue << std::endl;
			}
		}
		
		if(converged){
			for(size_t i = 0; i < nstate; ++i){
				eval[i] = lambda[i];
				evec = x;
			}
			std::cout << "Davidson converged" << std::endl;	
			break;
		}
		
		// compute new directions
		for(size_t i = 0; i < ntrial; ++i){
			t.col(i) = r.col(i)/(lambda[i] - A(i, i));
		}
		
		k++;
		ntrial = nstate*david_pars::ntrial_per_state;
		nvec_used += ntrial;
		
		if(static_cast<size_t>(V.cols()) < max_trial - ntrial){
			V.conservativeResize(dim, V.cols() + ntrial);
			for(size_t i = 0; i < ntrial; ++i)
				V.col(V.cols() - ntrial + i) = t.col(i);
		}
		else{
			V.resize(dim, 2*ntrial);
			for(size_t i = 0; i < ntrial; ++i)
				V.col(i) = x.col(i);
			for(size_t i = 0; i < ntrial; ++i)
				V.col(i + ntrial) = t.col(i);
		}
		
		auto QR = V.fullPivHouseholderQr();
		V = QR.matrixQ().adjoint().topRows(V.cols()).transpose(); 
			
	}while(k < max_iter);
	
	return converged;
}

Vector davidson_inderect::eigenvalues()
{
	assert(converged && "Davidson not converged, cannot return eigenvalues");
	return eval;
}
	
Matrix davidson_inderect::eigenvectors()
{
	assert(converged && "Davidson not converged, cannot return eigenvalues");
	return evec;
}	
	
	

// indirect Davidson example
#define DIM 1000
#define NSTATE 2
#define W18 std::setw(18)

void dvs_example()
{
	using Matrix = 
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 

// generate Hamiltonian-like random Hermiatian matrix
	std::random_device rd; 
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> nondiag(-5, -3);
 
	Matrix A(DIM, DIM);
	
	for(size_t i = 0; i < DIM; ++i){
		A(i,i) = static_cast<double>(i + 1);
		for(size_t j = i + 1; j < DIM; ++j){
			A(i,j) = A(j,i) = pow(-1, static_cast<int>(nondiag(gen))) * pow(10, nondiag(gen))/(i+j+1);
		}
	}

// reference diagonalization	
	auto start = std::chrono::high_resolution_clock::now();

	Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(A);
	
	auto ref_eval = eigensolver.eigenvalues();
	auto ref_evec = eigensolver.eigenvectors();

	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "full_diag time = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;

// multi-root Davidson diagonalizer
	
	davidson_inderect dvs_solver(A, NSTATE);
	
	start = std::chrono::high_resolution_clock::now();
	
	dvs_solver.solve();
	
	auto evec = dvs_solver.eigenvectors();
	auto eval = dvs_solver.eigenvalues();

	stop = std::chrono::high_resolution_clock::now();
	std::cout << "davidson time = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;

	std::cout << "Eigenvalues:" << std::endl;
	std::cout << W18 << "reference" << W18 << "davidson" << W18 << "diff" << std::endl;
	for(size_t i = 0; i < NSTATE; ++i){
		std::cout.precision(10);
		std::cout << W18 << ref_eval[i] << W18 <<  eval[i] << W18 << ref_eval[i] - eval[i] << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Eigenvectors:" << std::endl;
	std::cout << W18 << "reference" << W18 << "davidson" << W18 << "diff" << std::endl;
	for(size_t i = 0; i < NSTATE; ++i){
		std::cout << "State " << i << ":\n"; 
		for(size_t j = 0; j < DIM; j++){
			if(fabs(evec.col(i)[j]) > david_pars::tolerance/10){
				std::cout.precision(10);
				std::cout << W18 << evec.col(i)[j] << W18 << ref_evec.col(i)[j] 
					<< W18 << fabs(evec.col(i)[j]) - fabs(ref_evec.col(i)[j]) << std::endl;
			}
		}
	}

}
