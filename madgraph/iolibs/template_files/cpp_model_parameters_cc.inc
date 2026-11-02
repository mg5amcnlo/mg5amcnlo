//==========================================================================
// This file has been automatically generated for C++ by
%(info_lines)s
//==========================================================================

#include <algorithm>
#include <cctype>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "Parameters_%(model_name)s.h"

    void Parameters_%(model_name)s::setIndependentParameters(SLHAReader& slha){
	// Define "zero"
	zero=0;
	ZERO=0;
	// Prepare a vector for indices
	vector<int> indices(2,0);
	m_slha = slha;
	registerParameters();
	%(set_independent_parameters)s
    }
    void Parameters_%(model_name)s::setIndependentCouplings(){
	%(set_independent_couplings)s
    %(set_flv_couplings)s
    }
    void Parameters_%(model_name)s::setDependentParameters(){
	%(set_dependent_parameters)s
    }
    void Parameters_%(model_name)s::setDependentCouplings(){
	%(set_dependent_couplings)s
    }

    // Routines for printing out parameters
    void Parameters_%(model_name)s::printIndependentParameters(){
	cout << "%(model_name)s model parameters independent of event kinematics:" << endl;
	%(print_independent_parameters)s
    }
    void Parameters_%(model_name)s::printIndependentCouplings(){
	cout << "%(model_name)s model couplings independent of event kinematics:" << endl;
	%(print_independent_couplings)s
    }
    void Parameters_%(model_name)s::printDependentParameters(){
	cout << "%(model_name)s model parameters dependent on event kinematics:" << endl;
	%(print_dependent_parameters)s
    }
    void Parameters_%(model_name)s::printDependentCouplings(){
	cout << "%(model_name)s model couplings dependent on event kinematics:" << endl;
	%(print_dependent_couplings)s
    }

    void Parameters_%(model_name)s::registerParameters(){
	%(register_parameters)s
	%(register_slha_keys)s
    }

    bool Parameters_%(model_name)s::resolveSlhaKey(const std::string& name, std::string& resolved) const {
	std::istringstream iss(name);
	std::string block;
	if (!(iss >> block)) return false;
	std::transform(block.begin(), block.end(), block.begin(),
	               [](unsigned char c){ return std::tolower(c); });
	std::string key = block;
	bool any_index = false;
	int index;
	while (iss >> index) {
	    key += " " + std::to_string(index);
	    any_index = true;
	}
	if (!any_index) return false;
	std::map<std::string, std::string>::const_iterator it = m_slha_key_to_name.find(key);
	if (it == m_slha_key_to_name.end()) return false;
	resolved = it->second;
	return true;
    }

    bool Parameters_%(model_name)s::setParameterByName(std::string name, double real, double imag){
	(void)imag; // settable (SLHA-card-level) parameters are always real
	std::string resolved;
	if (resolveSlhaKey(name, resolved)) name = resolved;
	if (m_settable_params.find(name) == m_settable_params.end())
	    return false;
	m_param_overrides[name] = real;
	setIndependentParameters(m_slha);
	setIndependentCouplings();
	return true;
    }

    bool Parameters_%(model_name)s::getParameterByName(std::string name, double* real, double* imag){
	std::string resolved;
	if (resolveSlhaKey(name, resolved)) name = resolved;
	map<std::string, double*>::iterator real_it = m_real_params.find(name);
	if (real_it != m_real_params.end()){
	    *real = *(real_it->second);
	    if (imag) *imag = 0.;
	    return true;
	}
	map<std::string, std::complex<double> *>::iterator complex_it = m_complex_params.find(name);
	if (complex_it != m_complex_params.end()){
	    *real = complex_it->second->real();
	    if (imag) *imag = complex_it->second->imag();
	    return true;
	}
	return false;
    }

    void Parameters_%(model_name)s::resetParameterOverrides(){
	m_param_overrides.clear();
    }

