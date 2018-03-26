#include "integrals.h"

using namespace libint2;

size_t nbasis(const std::vector<Shell>& shells) {
  size_t n = 0;
  for (const auto& shell: shells)
    n += shell.size();
  return n;
}

Matrix compute_1body_ints (Operator op, const std::vector<Shell>& obs, const std::vector<Atom>& atoms)
{

	auto shell2bf = BasisSet::compute_shell2bf(obs);
	
	Engine engine(op, BasisSet::max_nprim(obs), BasisSet::max_l(obs));
	if(op == Operator::nuclear){
		assert(atoms.size()
			&& "calc_1body_ints: provide atoms for Operator::nuclear");
			
		std::vector<std::pair<double,std::array<double,3>>> q;
		for(const auto& atom : atoms) {
			q.push_back({static_cast<double>(atom.atomic_number), {{atom.x, atom.y, atom.z}}} );
		}
		engine.set_params(q);
	}
	
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
	
	return std::move(M);
}

Matrix compute_2body_fock(const std::vector<Shell>& shells,
                          const Matrix& D) {

  const auto n = nbasis(shells);
  Matrix G = Matrix::Zero(n,n);

  // construct the 2-electron repulsion integrals engine
  Engine engine(Operator::coulomb, BasisSet::max_nprim(shells), BasisSet::max_l(shells), 0);

  auto shell2bf = BasisSet::compute_shell2bf(shells);

  const auto& buf = engine.results();

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

          engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
          const auto* buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

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

std::vector<Matrix> compute_2body_fock_like_batch(const std::vector<Shell>& shells,
                          const std::vector<Matrix>& batch_P) {

  const auto n = nbasis(shells);
  std::vector<Matrix> batch_G(batch_P.size());
  
	for(size_t i = 0, i_max = batch_P.size(); i < i_max; ++i){
		batch_G[i] = Matrix::Zero(n,n);
	}

  // construct the 2-electron repulsion integrals engine
  Engine engine(Operator::coulomb, BasisSet::max_nprim(shells), BasisSet::max_l(shells), 0);

  auto shell2bf = BasisSet::compute_shell2bf(shells);

  const auto& buf = engine.results();

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

          engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
          const auto* buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

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
					
					// batch cycle
					
					for(size_t batch_i = 0, batch_max_i = batch_P.size(); batch_i < batch_max_i; ++batch_i){
						batch_G[batch_i](bf1,bf2) += batch_P[batch_i](bf3,bf4) * value_scal_by_deg;
						batch_G[batch_i](bf3,bf4) += batch_P[batch_i](bf1,bf2) * value_scal_by_deg;
						batch_G[batch_i](bf1,bf3) -= 0.25 * batch_P[batch_i](bf2,bf4) * value_scal_by_deg;
						batch_G[batch_i](bf2,bf4) -= 0.25 * batch_P[batch_i](bf1,bf3) * value_scal_by_deg;
						batch_G[batch_i](bf1,bf4) -= 0.25 * batch_P[batch_i](bf2,bf3) * value_scal_by_deg;
						batch_G[batch_i](bf2,bf3) -= 0.25 * batch_P[batch_i](bf1,bf4) * value_scal_by_deg;
					}
					
                }
              }
            }
          }

        }
      }
    }
  }

  // symmetrize the result and return
	for(size_t batch_i = 0, batch_max_i = batch_P.size(); batch_i < batch_max_i; ++batch_i){
		Matrix Gt = batch_G[batch_i].transpose();
		batch_G[batch_i] = 0.5*(Gt + batch_G[batch_i]);
	}

  return batch_G;
}


