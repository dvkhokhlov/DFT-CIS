//#include "qm_residue.h"

// multi-root Davidson

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <ctgmath>
#include <random>
#include <cassert>
#include <tuple>

#include <Eigen/Dense>

#define DIM 200

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Matrix;
        
int main()
{
// generate Hamiltonian-like random Hermiatian matrix
	std::random_device rd; 
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> nondiag(1E-2, 5E-1);
 
	Matrix A(DIM, DIM);
	
	for(size_t i = 0; i < DIM; ++i){
		A(i,i) = static_cast<double>(i);
		for(size_t j = i + 1; j < DIM; ++j){
			A(i,j) = A(j,i) = nondiag(gen);
		}
	}

// reference diagonalization	
	Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(A);
	
	auto ref_eval = eigensolver.eigenvalues();
	auto ref_evec = eigensolver.eigenvectors();
		
// multi-root Davidson diagonalizer
	
	const double tol = 1E-6;
	
	size_t neigenpair = 2;
	size_t maxiter = 1000;
	double residual;
	
	size_t k = 1;
	Matrix V(DIM, neigenpair);
	Matrix W(DIM, neigenpair);
	Matrix H(neigenpair, neigenpair);
	std::vector<Eigen::VectorXd> x(neigenpair);
	std::vector<Eigen::VectorXd> r(neigenpair);
	std::vector<Eigen::VectorXd> t(neigenpair);
	
	bool converged = false;
	//initial guess
	for(size_t i = 0; i < neigenpair; ++i)
		V(i, i) = 1.;
	
	do{
		// compute Wk = A*V
		W = A*V;
		// compute Rayleigh matrix H = Vt*W
		H = V.transpose()*W;
//		std::cout << H << std::endl;

		// diagonalize 
		Eigen::SelfAdjointEigenSolver<Matrix> R_eigensolver(H);
		auto y = R_eigensolver.eigenvectors();
		auto lambda = R_eigensolver.eigenvalues();
//		std::cout << R_eigensolver.eigenvalues() << std::endl;
//		std::cout << R_eigensolver.eigenvectors() << std::endl;

		// compute Ritz vectors
		for(size_t i = 0; i < neigenpair; ++i){
			x[i] = V*y.col(i);
		}
		// compute residuals
		converged = true;
		for(size_t i = 0; i < neigenpair; ++i){
			r[i] = lambda[i]*x[i] - W*y.col(i);
			
			auto residue = r[i].norm();
			if(residue > tol) converged = false;
			 
			std::cout << "state " << i << "; residue = " << residue << std::endl;
		}
		
		if(k==100){
//		if(converged){
			std::cout << "Davidson converged" << std::endl;
			
			for(size_t i = 0; i < neigenpair; ++i){
				std::cout.precision(5);
				std::cout << "Davidson L = " << lambda[i] << "; reference L = " << ref_eval[i] << std::endl;
				getchar();
				for(size_t j = 0; j < DIM; ++j){
					std::cout << y(i,j) << "  " << ref_evec(i,j) << std::endl;
				}
				getchar();
			}
			
//			std::cout << y.col(i) << std::endl;		
			break;
		}
		
		// compute new directions
		for(size_t i = 0; i < neigenpair; ++i){
			t[i] = r[i]/(lambda[i] - A(i,i));
		}
		
		//std::cout << r << std::endl;
		k++;
		
		V.conservativeResize(DIM, k*neigenpair);
		for(size_t i = 0; i < neigenpair; ++i){
			V.col((k-1)*neigenpair + i) = t[i];
		}
		Eigen::HouseholderQR<Matrix> qr(V);
		V = qr.householderQ(); 
		
		
		x.resize(k*neigenpair);
		r.resize(k*neigenpair);
		t.resize(k*neigenpair);
		
					
	}while(k < maxiter);
	

	

}


/*
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Matrix; 
        
size_t nbasis(const std::vector<libint2::Shell>& shells) {
  size_t n = 0;
  for (const auto& shell: shells)
    n += shell.size();
  return n;
}

size_t max_nprim(const std::vector<libint2::Shell>& shells) {
  size_t n = 0;
  for (auto shell: shells)
    n = std::max(shell.nprim(), n);
  return n;
}

int max_l(const std::vector<libint2::Shell>& shells) {
  int l = 0;
  for (auto shell: shells)
    for (auto c: shell.contr)
      l = std::max(c.l, l);
  return l;
}

std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells) {
  std::vector<size_t> result;
  result.reserve(shells.size());

  size_t n = 0;
  for (auto shell: shells) {
    result.push_back(n);
    n += shell.size();
  }

  return result;
}

Matrix compute_2body_fock(const std::vector<libint2::Shell>& shells,
                          const Matrix& D) {

  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  std::chrono::duration<double> time_elapsed = std::chrono::duration<double>::zero();

  const auto n = nbasis(shells);
  Matrix G = Matrix::Zero(n,n);

  // construct the 2-electron repulsion integrals engine
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);

  auto shell2bf = map_shell_to_basis_function(shells);

  const auto& buf = engine.results();

  // The problem with the simple Fock builder is that permutational symmetries of the Fock,
  // density, and two-electron integrals are not taken into account to reduce the cost.
  // To make the simple Fock builder efficient we must rearrange our computation.
  // The most expensive step in Fock matrix construction is the evaluation of 2-e integrals;
  // hence we must minimize the number of computed integrals by taking advantage of their permutational
  // symmetry. Due to the multiplicative and Hermitian nature of the Coulomb kernel (and realness
  // of the Gaussians) the permutational symmetry of the 2-e ints is given by the following relations:
  //
  // (12|34) = (21|34) = (12|43) = (21|43) = (34|12) = (43|12) = (34|21) = (43|21)
  //
  // (here we use chemists' notation for the integrals, i.e in (ab|cd) a and b correspond to
  // electron 1, and c and d -- to electron 2).
  //
  // It is easy to verify that the following set of nested loops produces a permutationally-unique
  // set of integrals:
  // foreach a = 0 .. n-1
  //   foreach b = 0 .. a
  //     foreach c = 0 .. a
  //       foreach d = 0 .. (a == c ? b : c)
  //         compute (ab|cd)
  //
  // The only complication is that we must compute integrals over shells. But it's not that complicated ...
  //
  // The real trick is figuring out to which matrix elements of the Fock matrix each permutationally-unique
  // (ab|cd) contributes. STOP READING and try to figure it out yourself. (to check your answer see below)

  // loop over permutationally-unique set of shells
  for(auto s1=0; s1!=shells.size(); ++s1) {

    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();   // number of basis functions in this shell

    for(auto s2=0; s2<=s1; ++s2) {

      auto bf2_first = shell2bf[s2];
      auto n2 = shells[s2].size();

      for(auto s3=0; s3<=s1; ++s3) {

        auto bf3_first = shell2bf[s3];
        auto n3 = shells[s3].size();

        const auto s4_max = (s1 == s3) ? s2 : s3;
        for(auto s4=0; s4<=s4_max; ++s4) {

          auto bf4_first = shell2bf[s4];
          auto n4 = shells[s4].size();

          // compute the permutational degeneracy (i.e. # of equivalents) of the given shell set
          auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
          auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
          auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
          auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

          const auto tstart = std::chrono::high_resolution_clock::now();

          engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
          const auto* buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          const auto tstop = std::chrono::high_resolution_clock::now();
          time_elapsed += tstop - tstart;

          // ANSWER
          // 1) each shell set of integrals contributes up to 6 shell sets of the Fock matrix:
          //    F(a,b) += (ab|cd) * D(c,d)
          //    F(c,d) += (ab|cd) * D(a,b)
          //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
          //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
          //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
          //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
          // 2) each permutationally-unique integral (shell set) must be scaled by its degeneracy,
          //    i.e. the number of the integrals/sets equivalent to it
          // 3) the end result must be symmetrized
          for(auto f1=0, f1234=0; f1!=n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for(auto f2=0; f2!=n2; ++f2) {
              const auto bf2 = f2 + bf2_first;
              for(auto f3=0; f3!=n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                  const auto bf4 = f4 + bf4_first;

                  const auto value = buf_1234[f1234];

                  const auto value_scal_by_deg = value * s1234_deg;

                  G(bf1,bf2) += D(bf3,bf4) * value_scal_by_deg;
                  G(bf3,bf4) += D(bf1,bf2) * value_scal_by_deg;
                  G(bf1,bf3) -= 0.25 * D(bf2,bf4) * value_scal_by_deg;
                  G(bf2,bf4) -= 0.25 * D(bf1,bf3) * value_scal_by_deg;
                  G(bf1,bf4) -= 0.25 * D(bf2,bf3) * value_scal_by_deg;
                  G(bf2,bf3) -= 0.25 * D(bf1,bf4) * value_scal_by_deg;
                }
              }
            }
          }

        }
      }
    }
  }

  // symmetrize the result and return
  Matrix Gt = G.transpose();
  return 0.5 * (G + Gt);
}


int main()
{
	QM_residue p1("gms_7amc.out");	
	
	libint2::initialize();
	
	Matrix D = Matrix::Zero(p1.ncgto,p1.ncgto);
	
	compute_2body_fock(p1.get_basis(), D);
	
	libint2::finalize();
	
}
*/
