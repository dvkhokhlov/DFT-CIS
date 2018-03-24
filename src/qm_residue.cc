#include "qm_residue.h"

QM_residue::QM_residue(const std::string& _qm_fname) : qm_fname(_qm_fname) 
{
	qm_file.open (qm_fname);
	if(!qm_file){
		throw std::runtime_error ("QM_residue.fopen(): cannot open QM file");
	}
	
	try{
		read_pars();
		read_atoms();
		read_basis();
		read_MOs();
	}
	catch (const std::exception &e){
		std::cerr << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
	
	qm_file.close();
}

void QM_residue::read_pars ()
{
	const size_t GMS_DATA_STRIDE = 47;
	const size_t GMS_DATA_WIDTH = 5;
	
	if(qm_file){	
				
		while(std::getline(qm_file, tmp)){
			if(!tmp.compare(0, 33, " TOTAL NUMBER OF BASIS SET SHELLS")){
				nshell = std::stoul (tmp.substr(GMS_DATA_STRIDE, GMS_DATA_WIDTH));
				break;
			}
		}
		
		while(std::getline(qm_file, tmp)){
			if(!tmp.compare(0, 45, " NUMBER OF CARTESIAN GAUSSIAN BASIS FUNCTIONS")){
				ncgto = std::stoul (tmp.substr(GMS_DATA_STRIDE, GMS_DATA_WIDTH));
				break;
			}
		}
		
		while(std::getline(qm_file, tmp)){
			if(!tmp.compare(0, 22, " TOTAL NUMBER OF ATOMS")){
				natom = std::stoul (tmp.substr(GMS_DATA_STRIDE, GMS_DATA_WIDTH));
				break;
			}
		}
		
		if(natom && ncgto && nshell){
			parsQ = true;
		}	
		else{
			throw std::runtime_error ("QM_residue.read_pars(): parameters block not found");
		}
		
//		std::cout << natom << ' ' << ncgto << ' ' << nshell << std::endl;
		
	}
	else{
		throw std::runtime_error ("QM_residue.read_pars(): QM file stream closed");
	}
}

void QM_residue::read_atoms()
{
	if(qm_file && parsQ){
		const size_t GMS_CHRG_STRIDE = 11;
		const size_t GMS_CHRG_WIDTH = 3;
		const size_t GMS_RX_STRIDE = 16;
		const size_t GMS_RY_STRIDE = 36;
		const size_t GMS_RZ_STRIDE = 56;
		const size_t GMS_R_WIDTH = 17;	
				
		qm_file.seekg(0, qm_file.beg);
		
		bool atomsQ = false;
		while(std::getline(qm_file, tmp)){
			if(!tmp.compare(" ATOM      ATOMIC                      COORDINATES (BOHR)")){
				atomsQ = true;
				break;
			}
		}
		
		if(!atomsQ){
			throw std::runtime_error ("QM_residue.read_atoms(): geometry block not found");
		}
		
		atoms.resize (natom);
		
		std::getline(qm_file, tmp);
		
		for(auto& atom : atoms){
			std::getline(qm_file, tmp);
			atom.atomic_number = std::stoul (tmp.substr(GMS_CHRG_STRIDE, GMS_CHRG_WIDTH));
			atom.x = std::stod (tmp.substr(GMS_RX_STRIDE, GMS_R_WIDTH));
			atom.y = std::stod (tmp.substr(GMS_RY_STRIDE, GMS_R_WIDTH));
			atom.z = std::stod (tmp.substr(GMS_RZ_STRIDE, GMS_R_WIDTH));
			
//			std::cout << atom.atomic_number << ' ' << atom.x << ' ' << atom.y << ' ' << atom.z << std::endl;
		}
		
	}
	else{
		if(!qm_file) throw std::runtime_error ("QM_residue.read_atoms(): QM file stream closed");
		if(!parsQ) throw std::runtime_error ("QM_residue.read_atoms(): read parameters before");
	}
	
}

void QM_residue::read_shell (const std::streampos& p_beg, const std::streampos& p_end, size_t n)
{
	const size_t GMS_EXP_STRIDE = 24;
	const size_t GMS_EXP_WIDTH = 17;
	const size_t GMS_CONTR0_STRIDE = 44;
	const size_t GMS_CONTR1_STRIDE = 62;
	const size_t GMS_CONTR_WIDTH = 15;
		
	qm_file.seekg(p_beg, qm_file.beg);
	
	std::getline(qm_file, tmp);
		
	int am;
	auto am_name = tmp.at(10);
	switch (am_name){
		case 'S' : am = 0; break;
		case 'P' : am = 1; break;
		case 'D' : am = 2; break;
		case 'F' : am = 3; break;
		case 'L' : am = 0; break;
		default: throw std::runtime_error ("QM_residue.read_basis(): unsupported angular moment\n");
	}
	
	std::vector<double> exps;
    std::vector<double> coeffs_0;
    std::vector<double> coeffs_1;
    
	while(qm_file.tellg() < p_end){
		exps.push_back (std::stod (tmp.substr(GMS_EXP_STRIDE, GMS_EXP_WIDTH)));
		
		coeffs_0.push_back (std::stod (tmp.substr(GMS_CONTR0_STRIDE, GMS_CONTR_WIDTH)));
		if(am_name == 'L'){
			coeffs_1.push_back (std::stod (tmp.substr(GMS_CONTR1_STRIDE, GMS_CONTR_WIDTH)));
		}
		
		std::getline(qm_file, tmp);
	}
	
	basis.push_back(
		libint2::Shell{
			exps,
            {
				{am, pure, coeffs_0}
            },
            {{atoms[n].x, atoms[n].y, atoms[n].z}}
		}
	);
	
	if(am_name == 'L'){
		basis.push_back(
			libint2::Shell{
				exps,
				{
					{1, pure, coeffs_1}
				},
				{{atoms[n].x, atoms[n].y, atoms[n].z}}
			}
		);
	}	
	
}

void QM_residue::read_basis ()
{
	const size_t GMS_NAME_WIDTH = 11;
	const size_t GMS_SHELLNUM_POS = 6;			
			
	if(qm_file && parsQ){
		
		qm_file.seekg(0, qm_file.beg);
		
		while(std::getline(qm_file, tmp)){
			if(!tmp.compare("  SHELL TYPE  PRIMITIVE        EXPONENT          CONTRACTION COEFFICIENT(S)")){
				break;
			}
		}
				
		std::streampos p, p_beg;
		bool shellQ = false;	
		size_t natom_found = -1;
		while(true){
			p = qm_file.tellg();
			
			if(!std::getline(qm_file, tmp)){
				throw std::runtime_error ("QM_residue.read_basis(): EOF reached before the end of basis\n");
			}
			
			if(!tmp.size()){
				if(shellQ){
					read_shell(p_beg, qm_file.tellg(), natom_found);
					shellQ = false;
				}
				continue;
			}
			
			if(tmp.size() == GMS_NAME_WIDTH){
				natom_found++;
				continue;
			}

			if(!tmp.compare(0, 4, "#end")){
				if(natom_found == natom - 1){
					break;
				}
				else{
					throw std::runtime_error ("QM_residue.read_basis(): end of basis reached before the end of atoms\n");
				}
			}
						
			if(!isspace(tmp.at(GMS_SHELLNUM_POS)) && !shellQ){
				shellQ = true;
				p_beg = p;
			} 
		}
		
	}	
	else{
		if(!qm_file) throw std::runtime_error ("QM_residue.read_basis(): QM file stream closed");
		if(!parsQ) throw std::runtime_error ("QM_residue.read_basis(): read parameters before");
	}

	if(nshell != basis.size()){
		std::cout << "QM_residue.read_basis(): number of shells increased (and = " << basis.size() << ") due to splitting of SP shells" << std::endl;
		nshell = basis.size();
	}
	
//	for(auto& shell : basis){
//		std::cout << shell;
//	}

}

void QM_residue::resort_MOs ()
{
	// resort from GAMESS order (XX, YY, ZZ, ..) into libint one
	auto map = [](size_t am, size_t ndx) -> size_t {
		const static size_t d_map[]{0, 3, 4, 1, 5, 2};
		const static size_t f_map[]{0, 3, 4, 5, 9, 7, 1, 6, 8, 2};
		switch(am){
			case 0: return ndx;
			case 1: return ndx;
			case 2: return d_map[ndx];
			case 3: return f_map[ndx];
			default: throw std::runtime_error ("QM_residue.resort_MOs(): unsupported angular moment\n");
		}
	};
	
	// embed normalization coefficients for non-axial primitives
	auto scale = [](size_t am, size_t i) -> double {
		switch(am){
			case 0: return 1.;
			case 1: return 1.;
			case 2: 
				if(i == 0) return 1.;
				if(i == 3) return 1.;
				if(i == 5) return 1.;
				return pow(3., 0.5);
			case 3: 
				if (i == 0) return 1.;
				if (i == 6) return 1.;
				if (i == 9) return 1.;
				if (i == 1) return pow(5., 0.5);
				if (i == 2) return pow(5., 0.5);
				if (i == 3) return pow(5., 0.5);
				if (i == 5) return pow(5., 0.5);
				if (i == 7) return pow(5., 0.5);
				if (i == 8) return pow(5., 0.5);
				return pow(15., 0.5);
			default: throw std::runtime_error ("QM_residue.resort_MOs(): unsupported angular moment\n");
		}
	};
			
	auto shell2bf = libint2::BasisSet::compute_shell2bf(basis);
	// resort molecular orbitals
	for(size_t i = 0; i < nmo; i++){
		for(size_t j = 0; j < nshell; j++){
			auto am = basis[j].contr[0].l;

			size_t ngss = basis[j].size();
			
			if(am > 1 ){
				double tmp[ngss];
				
				for(size_t k = 0; k < ngss; k++){
						tmp[k] = MOcoef(i, shell2bf[j] + k);
				}
				
				for(size_t k = 0; k < ngss; k++){
						MOcoef(i, shell2bf[j] + k) = tmp[map(am, k)]*scale(am, k);
				}
			}
		}
	}
	
}

void QM_residue::read_MOs ()
{
	const size_t GMS_NCOL_MO = 5;
	const size_t GMS_MO_STRIDE = 5;
	const size_t GMS_MO_WIDTH = 15;
	
	if(qm_file && parsQ){
		qm_file.seekg(0, qm_file.beg);
			
		bool vecfound = false;
		while(std::getline(qm_file, tmp)){
			if(!tmp.compare(0, 4, "#vec")){
				vecfound = true;
				break;
			}
		}
		if(!vecfound) throw std::runtime_error ("QM_residue.read_MOs(): #vec group not found\n");
		
// !!! assuption NMO = NAO			
		nmo = ncgto;
		MOcoef.resize(nmo, nmo);
		for(size_t i = 0; i < nmo; ++i){
			for(size_t j = 0; j < nmo/GMS_NCOL_MO; ++j){
				
				std::getline(qm_file, tmp);
				if(!tmp.compare("#end")) throw std::runtime_error ("QM_residue.read_MOs(): #end reached before the end of MOs");
				
				for(size_t k = 0; k < GMS_NCOL_MO; ++k){
					MOcoef(i, j*GMS_NCOL_MO + k) 
						= std::stod(tmp.substr(GMS_MO_STRIDE + k*GMS_MO_WIDTH, GMS_MO_WIDTH));
				}
			}
			if(nmo%GMS_NCOL_MO){
				std::getline(qm_file, tmp);
				if(!tmp.compare("#end")) throw std::runtime_error ("QM_residue.read_MOs(): #end reached before the end of MOs");
				
				for(size_t k = 0; k < nmo%GMS_NCOL_MO; ++k){
					MOcoef(i, (nmo/GMS_NCOL_MO)*GMS_NCOL_MO + k) 
						= std::stod(tmp.substr(GMS_MO_STRIDE + k*GMS_MO_WIDTH, GMS_MO_WIDTH));
				}				
			}
		}
		
	}
	else{
		if(!qm_file) throw std::runtime_error ("QM_residue.read_mos(): QM file stream closed");
		if(!parsQ) throw std::runtime_error ("QM_residue.read_mos(): read parameters before");
	}
		
	resort_MOs();
/*	
	// eigenvectors of MOs are now stored in columns
	// as in Eigen
	MOcoef.transpose();
	*/
}
