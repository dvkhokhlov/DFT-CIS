// multi-root Davidson
// M.Crouzeix, B. Philippe, M. Sadkane // SIAM J. Sci. Comput. 15 (1994) 62-76
#ifndef DAVIDSON_INCLUDED_H
#define DAVIDSON_INCLUDED_H

#include <tuple>

#include <Eigen/Dense>

class davidson
{
	using Matrix = 
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vector =
		Eigen::VectorXd;
        
	public:
	davidson() = delete;
	davidson(const Matrix& _A, size_t _nstate, double _tol=1E-4, size_t _max_iter=100) : 
		A(_A), dim(A.cols()), nstate(_nstate), tol(_tol), max_iter(_max_iter),
		evec(A.cols(), _nstate), eval(_nstate)
	{
		assert(A.cols() == A.rows() && "davidson(): matrix should be square");
	}
	~davidson() = default;
	
	bool solve()
	{	
		const size_t ntrial_per_state = 5;
		size_t ntrial = nstate*ntrial_per_state;
		
		//initial guess
		Matrix V(dim, ntrial);	
		for(size_t i = 0; i < ntrial; ++i)
			V(i, i) = 1.;
		
		size_t k = 1;
		
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
			Matrix x(dim, ntrial);
			for(size_t i = 0; i < ntrial; ++i){
				x.col(i) = V*y.col(i);
			}
			
			// compute residuals
			Matrix r(dim, ntrial);	
			converged = true;
			std::cout << "niter = " << k; 
			std::cout << "; vectors used = " << k*ntrial << std::endl;
			for(size_t i = 0; i < nstate; ++i){
				r.col(i) = lambda[i]*x.col(i) - W*y.col(i);
				
				auto residue = r.col(i).norm()/sqrt(dim);
				if(residue > tol) converged = false;
	
				std::cout << "state " << i << "; residue = " << residue << std::endl;
			}
			
			if(converged){
				for(size_t i = 0; i < nstate; i++){
					eval[i] = lambda[i];
				}
				std::cout << "Davidson converged" << std::endl;	
				break;
			}
			
			Matrix t(dim, ntrial);
			// compute new directions
			for(size_t i = 0; i < ntrial; ++i){
				t.col(i) = r.col(i)/(lambda[i] - A(i,i));
			}
			
			k++;
			
			if(V.cols() < max_trial - ntrial){
				V.conservativeResize(dim, V.cols() + ntrial);
				for(size_t i = 0; i < ntrial; ++i){
					V.col(V.cols() - ntrial + i) = t.col(i);
				}
			}
			else{
				V.resize(dim, 2*ntrial);
				for(size_t i = 0; i < ntrial; ++i){
					V.col(i) = x.col(i);
				}
				for(size_t i = 0; i < ntrial; ++i){
					V.col(i + ntrial) = t.col(i);
				}
			}
			
			auto QR = V.fullPivHouseholderQr();
			V = QR.matrixQ().adjoint().topRows(V.cols()).transpose(); 
				
		}while(k < max_iter);
		
		return converged;
	}
	
	Vector eigenvalues()
	{
		assert(converged && "Davidson not converged, cannot return eigenvalues");
		return eval;
	}
	
	Matrix eigenvectors()
	{
		assert(converged && "Davidson not converged, cannot return eigenvalues");
		return evec;
	}	
	 
	private:
	const Matrix& A;
	size_t dim, nstate;
	double tol;
	size_t max_iter;
	
	size_t max_trial = 100;
	
	Matrix evec;
	Vector eval;
	
	bool converged;
};


#endif
