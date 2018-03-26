#include "integrals.h"

using namespace libint2;
using Matrix = 
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 

size_t nbasis(const std::vector<Shell>& shells) {
  size_t n = 0;
  for (const auto& shell: shells)
    n += shell.size();
  return n;
}

Matrix calc_1body_ints (Operator op, const std::vector<Shell>& obs)
{
	initialize();

	size_t max_nprim = BasisSet::max_nprim(obs);
	size_t max_l = BasisSet::max_l(obs);
	auto shell2bf = BasisSet::compute_shell2bf(obs);
	
	Engine engine(op, max_nprim, max_l);
	const auto& buf_vec = engine.results();
	
	Matrix M(nbasis(obs), nbasis(obs));
	
	for(size_t i = 0, i_max = obs.size(); i < i_max; ++i){
		for(size_t j = i, j_max = obs.size(); j < j_max; ++j){
			
			engine.compute(obs[i], obs[j]);
			
			auto ints_set = buf_vec[0];
			
			auto nf1 = obs[i].size();
			auto nf2 = obs[j].size();
			auto bf1 = shell2bf[i];
			auto bf2 = shell2bf[j];
			
			for(size_t k = 0; k < nf1; k++){
				for(size_t l = 0; l < nf2; l++){
					M(bf1 + k, bf2 + l) = ints_set[k*nf2 + l];
					M(bf2 + l, bf1 + k) = ints_set[k*nf2 + l];
				}
			}
		}
	}

	finalize();
	
	return std::move(M);
}
