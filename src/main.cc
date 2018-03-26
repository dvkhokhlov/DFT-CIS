#include <iostream>
#include <iomanip>
#include <ctgmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <utility>

#include "integrals.h"
#include "davidson.h"
#include "qm_residue.h"

namespace pars{
	const double orto_cutoff = 2E-7;
};

int main(int argc, char *argv[])
{

	std::string inpfile;
	for(size_t i = 0; i < static_cast<size_t>(argc); ++i){
        if(std::string{argv[i]} == "-inp"){inpfile = std::string{argv[i+1]};}
    }

	QM_residue mol(inpfile);
	
	Matrix& MOcoef = mol.get_MOs();
	
	libint2::initialize();
	
// check orthonormality of MOs 	
	Matrix S = compute_1body_ints(libint2::Operator::overlap, mol.get_basis());
	
	for(size_t i = 0; i < mol.nmo; ++i){
		Eigen::VectorXd MOi = MOcoef.row(i);
		Eigen::VectorXd SMOi = S*MOi;
		auto norm = MOi.dot(SMOi);
		if(fabs(norm - 1.) > pars::orto_cutoff){
			std::cout.precision(10);
			std::cout << "warning: normalization of MO" << i << " = " << norm << std::endl;
		}
	}	
	
// direct CIS for small molecules
// build density matrix
	auto nocc = mol.nelec()/2;
	std::cout << nocc << std::endl;
// GAMESS molecular orbitals are stored in rows
	auto MOcoef_occ = MOcoef.topRows(nocc);
    Matrix D = MOcoef_occ.transpose() * MOcoef_occ;

// build Fock matrix and compute orbiat energies
	Matrix T = compute_1body_ints(libint2::Operator::kinetic, mol.get_basis());
	Matrix V = compute_1body_ints(libint2::Operator::nuclear, mol.get_basis(), mol.get_atoms());
	
	Matrix F = T + V + compute_2body_fock(mol.get_basis(), D);
	
	Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> eigensolver(F, S);
	auto orbital_e = eigensolver.eigenvalues();
	
	for(size_t i = 0; i < mol.nmo; ++i){
		std::cout << orbital_e[i] << std::endl;
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

/*	
	for(const auto& sap : saps){
		std::cout << sap.first << " -> " << sap.second << "; E = " 
			<< orbital_e[sap.second] - orbital_e[sap.first] << std::endl;
	}
*/
	size_t nstate = 2;

/*
 *  Davidson procedure	
 */

	libint2::finalize();	

}


