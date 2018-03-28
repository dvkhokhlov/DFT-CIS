#include <iostream>
#include <iomanip>
#include <ctgmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <utility>

#include "integrals.h"
#include "qm_residue.h"

#define W18 std::setw(18)

namespace pars{
	const double orto_cutoff = 2E-7;
	const double saps_printf_cutoff = 0.05;
	const std::string inpdir = "inp/";
	const std::string outdir = "out/";
	const std::string head = " ======== CI-Singles ========";
};

namespace david_pars{
	const size_t max_davison_iter = 200;
	const size_t ntrial_per_state = 2;
	const size_t trial_space_mult = 4*ntrial_per_state;
	const size_t ntrial_per_state0 = ntrial_per_state;
	const double tolerance = 1E-5;
};

int main(int argc, char *argv[])
{

	std::string inpfile;
	std::string outfile;
	for(size_t i = 0; i < static_cast<size_t>(argc); ++i){
        if(std::string{argv[i]} == "-inp"){
			inpfile = pars::inpdir + std::string{argv[i+1]} + ".inp";
			outfile = pars::outdir + std::string{argv[i+1]} + ".out";
		}
    }

	QM_residue mol(inpfile);
	std::ofstream outf{outfile};
	outf << pars::head << std::endl;
	outf.precision(12);
		
	Matrix& MOcoef = mol.get_MOs();

	libint2::initialize();
	
// check orthonormality of MOs 	
	Matrix S = compute_1body_ints(libint2::Operator::overlap, mol.get_basis());
	
	for(size_t i = 0; i < mol.nmo; ++i){
		Eigen::VectorXd MOi = MOcoef.row(i);
		Eigen::VectorXd SMOi = S*MOi;
		auto norm = MOi.dot(SMOi);
		if(fabs(norm - 1.) > pars::orto_cutoff){
			outf << "warning: normalization of MO" << i << " = " << norm << std::endl;
		}
	}	
	
// direct CIS for small molecules
// build density matrix
	auto nocc = mol.nelec()/2;
	std::cout << nocc << std::endl;
// GAMESS molecular orbitals are stored in rows
	auto MOcoef_occ = MOcoef.topRows(nocc);
    Matrix D = MOcoef_occ.transpose() * MOcoef_occ;

	auto start = std::chrono::high_resolution_clock::now();
// build Fock matrix and compute orbiat energies
	Matrix T = compute_1body_ints(libint2::Operator::kinetic, mol.get_basis());
	Matrix V = compute_1body_ints(libint2::Operator::nuclear, mol.get_basis(), mol.get_atoms());
	
	Matrix F = T + V + compute_2body_fock(mol.get_basis(), D);

	auto stop = std::chrono::high_resolution_clock::now();
	outf << "Fock time = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;
	
	Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> eigensolver(F, S);
	auto orbital_e = eigensolver.eigenvalues();
	
	outf << "orbital energies:" << std::endl;
	for(size_t i = 0; i < mol.nmo; ++i){
		outf << orbital_e[i] << std::endl;
	}
	
/*
 * 
 * CI-S for singlet states 
 * 
*/
	
// determine chemical core
	auto ncore_atomic = [](size_t atomic_number)
	{
		assert(atomic_number <= 18 && "ncore_atomic(): 4th period and higher ones are not supported");
		
		auto ncore = 0ul;
		
		if(atomic_number <= 2){
			ncore = 0;
		}
		else if(atomic_number <= 10){
			ncore = 1;
		}
		else if(atomic_number <= 18){
			ncore = 5; 
		}
		
		return ncore;
	};
	
	size_t ncore = 0;
	for(size_t i = 0, i_max = mol.get_atoms().size(); i < i_max; ++i){
		ncore += ncore_atomic(mol.get_atoms()[i].atomic_number);
	}

// create SAPS
	auto nao = mol.ncgto;
	auto nmo = mol.nmo;
	auto nval = nmo - nocc;
	auto nact = nocc - ncore;
	auto nsaps = nval*nact;
	
	std::vector<std::pair<size_t, size_t>> saps;
	
	for(size_t i = 0; i < nact; ++i){
		for(size_t j = 0; j < nval; ++j){
			saps.emplace_back(std::make_pair(ncore + i, nocc + j));
		}
	}
	
// sort SAPS
	auto energy_sort = [=](std::pair<size_t, size_t> sap1, std::pair<size_t, size_t> sap2){
		auto e1 = orbital_e[sap1.second] - orbital_e[sap1.first];
		auto e2 = orbital_e[sap2.second] - orbital_e[sap2.first];
		return e1 < e2;
	};
	
	std::sort(saps.begin(), saps.end(), energy_sort);

	outf << "SAPS diagonal eneries" << std::endl;
	for(const auto& sap : saps){
		outf << sap.first << " -> " << sap.second << "; E = " 
			<< orbital_e[sap.second] - orbital_e[sap.first] << std::endl;
	}


/*
 *  Davidson procedure	
 */

	size_t nstate = 2;
	size_t ntrial = nstate*david_pars::ntrial_per_state; //12;
	size_t model_dim = ntrial;
	size_t nvec_used = ntrial;
	
// eigenvectors and eigenvalues
	Matrix cis_evec(nsaps, nstate);
	Vector cis_eval(nstate);
	 
// estimated Hcis(i,i) for preconditioner
	Vector Hdiag(nsaps);
	for(size_t k = 0; k < nsaps; ++k)
		Hdiag(k) = orbital_e[saps[k].second] - orbital_e[saps[k].first];
	
	size_t niter = 0;
	
// initial guess	
	Matrix V_trial = Matrix::Zero(nsaps, ntrial);
	for(size_t tr = 0; tr < ntrial; ++tr)
		V_trial(tr, tr) = 1.0;	

// 	evaluation of CIS state density; in form of lambda
	auto compute_Tcis = [&](const Matrix& V_st){
		Matrix Tcis = Matrix::Zero(nao, nao);
		
		for(size_t i = 0; i < nao; ++i){
			for(size_t j = 0; j < nao; ++j){
				for(size_t k = 0; k < nsaps; ++k){
					Tcis(i,j) += V_st(k)*MOcoef(saps[k].first, i)*MOcoef(saps[k].second, j);
				}
			}
		}
		
		return std::move(Tcis);
	};
	
// old eigenvectors for consistency check	
	Matrix evec_old;
	do{
	
	std::vector<Matrix> Tcis(ntrial);
	for(size_t tr = 0; tr < ntrial; ++tr)
		Tcis[tr] = compute_Tcis(V_trial.col(tr));

//	outf << "CIS-S trial density for state 0:" << std::endl;
//	outf << Tcis[0] << std::endl;
	
	start = std::chrono::high_resolution_clock::now();

// Fock-like matrices; batched evaluation
	auto Flike_batch = compute_2body_fock_like_batch_s(mol.get_basis(), Tcis);

	stop = std::chrono::high_resolution_clock::now();
	outf << "Fock-like time = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;

//	outf << "Fock-like build for trial vector 0:" << std::endl;
//	outf << Flike_batch[0] << std::endl;
		
// W_trial = H*V_trial	
	Matrix W_trial = Matrix::Zero(nsaps, ntrial);
	for(size_t tr = 0; tr < ntrial; ++tr){
		
		for(size_t k = 0; k < nsaps; ++k){
			
			for(size_t i = 0; i < nao; ++i){
				for(size_t j = 0; j < nao; ++j){
					W_trial(k, tr) += MOcoef(saps[k].first, i)*MOcoef(saps[k].second, j)*Flike_batch[tr](i, j);
				}
			}
			
			W_trial(k, tr) += (orbital_e[saps[k].second] - orbital_e[saps[k].first])*V_trial(k, tr);
			
		}
		
	}

// projection to model Hamiltonian
	auto model_H = V_trial.transpose()*W_trial;
	
//	outf << "Model CI-S Hamiltonian; ntrial = " << ntrial << "; nsaps = " << nsaps << std::endl;
//	outf << model_H << std::endl;

// diagonalization
	Eigen::SelfAdjointEigenSolver<Matrix> eigensolverH(model_H);
	
//	outf << "Eigenvalues in a.u.:" << std::endl;	
//	outf << eigensolverH.eigenvalues() << std::endl;
//	outf << eigensolverH.eigenvectors() << std::endl;
	
	auto evec = eigensolverH.eigenvectors();

// check consistency of model_H	
	if(niter > 0){
		outf << "evec - evec_old residues:" << std::endl;
		for(size_t tr = 0; tr < ntrial - nstate*david_pars::ntrial_per_state; ++tr){
			auto res = 0.;
			for(size_t i = 0; i < ntrial - nstate*david_pars::ntrial_per_state; ++i){
				res += pow(fabs(evec(i, tr)) - fabs(evec_old(i, tr)), 2.);			
			}
			outf << sqrt(res) << std::endl;
		}
	}
	
	evec_old = evec;
// end check
	auto eval = eigensolverH.eigenvalues();

// Ritz vectors	
	Matrix x = Matrix::Zero(nsaps, ntrial);
	for(size_t tr = 0; tr < ntrial; ++tr)
		x.col(tr) = V_trial*evec.col(tr);
	
// residual vectors
//	outf << "Residues:"  << std::endl;
	Matrix res = Matrix::Zero(nsaps, ntrial);
	for(size_t tr = 0; tr < ntrial; ++tr){
		res.col(tr) = eval[tr]*x.col(tr) - W_trial*evec.col(tr);
//		outf << res.col(tr).norm() << std::endl;
	}
	
// convergence check
	outf << "Iteration " << niter << " :" << std::endl;
	outf << W18 << "Energy" << W18 << "Residue" << std::endl;
	bool converged = true;
	for(size_t i = 0; i < nstate; ++i){
		auto residue = res.col(i).norm();
		outf << "state " << i + 1 << W18 << eval[i] << W18 << residue << std::endl;
		if(residue > david_pars::tolerance) converged = false;
	}
	if(converged){
		for(size_t i = 0; i < nstate; ++i){
			cis_eval[i] = eval[i];
			cis_evec.col(i) = x.col(i);
		}
		outf << "Davidson converged" << std::endl;
		break;
	}
	
// correction vectors
	Matrix corr = Matrix::Zero(nsaps, ntrial);
	for(size_t tr = 0; tr < ntrial; ++tr){
		for(size_t k = 0; k < nsaps; ++k){
			corr(k, tr) = res(k, tr)/(eval[tr] - Hdiag[k]);
		}
	}
//	outf << corr << std::endl;
	
	nvec_used += ntrial;
	assert(nvec_used < nsaps && "number of used trial vectors exceeded number of SAPS");
		
	auto ntrial_add = nstate*david_pars::ntrial_per_state;
//	if(ntrial <= nstate*david_pars::trial_space_mult - ntrial_add){
		auto ntrial_old = ntrial;
		ntrial += ntrial_add;
		V_trial.conservativeResize(nsaps, ntrial);
		for(size_t i = 0; i < ntrial_add; ++i)
			V_trial.col(ntrial_old + i) = corr.col(i);
/*	}
	else{
		V_trial.resize(nsaps, 2*ntrial_add);
		ntrial = 2*ntrial_add;
		for(size_t i = 0; i < ntrial_add; ++i)
			V_trial.col(i) = x.col(i);
		for(size_t i = 0; i < ntrial_add; ++i)
			V_trial.col(i + ntrial_add) = corr.col(i);
	}*/
		
	auto QR = V_trial.fullPivHouseholderQr();
	V_trial = QR.matrixQ().adjoint().topRows(V_trial.cols()).transpose(); 

	++niter;
	}while(niter < david_pars::max_davison_iter);
	
	
	for(size_t i = 0; i < nstate; ++i){
		outf << "State " << i + 1 << std::endl;
		outf << "Energy = " << cis_eval[i] << std::endl;
		for(size_t k = 0; k < nsaps; ++k){
			if(fabs(cis_evec(k, i)) > pars::saps_printf_cutoff){
				outf << saps[k].first << " -> " << saps[k].second << ' ' << cis_evec(k, i) << std::endl;
			}
		}
	}
/*
	outf << "Eigenvectors of model space" << std::endl;
	for(size_t i = 0; i < ntrial; ++i){
		outf << saps[i].first << " -> " << saps[i].second << ' ';
		for(size_t j = 0; j < ntrial; ++j){
			outf << std::setw(16) << evec(i, j);
		}
		outf << std::endl;
	}
*/	
	
	
	
	libint2::finalize();	

}


