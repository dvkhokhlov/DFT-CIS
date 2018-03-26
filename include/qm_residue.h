#ifndef QM_RESIDUE_H_INCLUDED
#define QM_RESIDUE_H_INCLUDED

#include "libint2.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <memory>
#include <limits>
#include <ctgmath>

class QM_residue
{
	using Matrix = 
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 
	
	public:
	// constructors
	QM_residue() = delete;
	QM_residue(const std::string& _qm_fname);
	
	~QM_residue(){};
	
	inline std::vector<libint2::Shell>& get_basis()
	{
		return basis;
	}
	
	const inline std::vector<libint2::Shell>& get_basis() const
	{
		return basis;
	}

	inline std::vector<libint2::Atom>& get_atoms()
	{
		return atoms;
	}
	
	const inline std::vector<libint2::Atom>& get_atoms() const
	{
		return atoms;
	}
	
	inline Matrix& get_MOs()
	{
		return MOcoef;
	}
	
	const inline Matrix& get_MOs() const
	{
		return MOcoef;
	}
	
	size_t ncgto;
	size_t natom;
	size_t nshell;
	size_t nmo;
	
	private:
	// add reading mode!
	bool pure = false;
	std::string qm_fname;
		
	std::vector<libint2::Shell> basis;
	std::vector<libint2::Atom> atoms;
	Matrix MOcoef;
	
// auxiliary functions/variables
	bool parsQ = false;
	std::string tmp;
	std::ifstream qm_file;
	void read_shell (const std::streampos&, const std::streampos&, size_t atom);	
	void read_pars ();
	void read_atoms();
	void read_basis ();
	void read_MOs ();
	void resort_MOs();
};

#endif
