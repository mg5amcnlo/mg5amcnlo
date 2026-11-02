//==========================================================================
// This file has been automatically generated for C++
%(info_lines)s
//==========================================================================

#ifndef Parameters_%(model_name)s_H
#define Parameters_%(model_name)s_H

#include <complex>
#include <map>
#include <set>
#include <string>

#include "read_slha.h"
using namespace std;

struct FLV_COUPLING
{
     int partner[4] = {-1,-1,-1,-1};
     int partner2[4] = {-1,-1,-1,-1};
     std::complex<double>* val[4];
};

class Parameters_%(model_name)s
{
public:

// Define "zero"
double zero, ZERO;
%(independent_parameters)s
%(independent_couplings)s
%(dependent_parameters)s
%(dependent_couplings)s
%(flavor_independent_couplings)s
%(flavor_dependent_couplings)s

// Set parameters that are unchanged during the run
void setIndependentParameters(SLHAReader& slha);
// Set couplings that are unchanged during the run
void setIndependentCouplings();
// Set parameters that are changed event by event
void setDependentParameters();
// Set couplings that are changed event by event
void setDependentCouplings();

// Print parameters that are unchanged during the run
void printIndependentParameters();
// Print couplings that are unchanged during the run
void printIndependentCouplings();
// Print parameters that are changed event by event
void printDependentParameters();
// Print couplings that are changed event by event
void printDependentCouplings();

// Set the value of an independent, SLHA-card-level parameter by name at
// runtime (overriding the value read from the param card), and recompute
// every parameter/coupling that depends on it. Returns false if "name" is
// not the name of a settable independent parameter (e.g. it names an
// internal parameter derived from others, or a coupling, neither of which
// can be set directly since they would just be recomputed from the
// underlying parameters on the next call). "name" may be either the
// parameter's model name (e.g. "mdl_MT") or its SLHA card location, given as
// "<block> <index> [<index2>]" (e.g. "mass 6"); the block name is matched
// case-insensitively.
bool setParameterByName(std::string name, double real, double imag);
// Retrieve the current value of any parameter or coupling by name (real or
// complex, independent or dependent). As with setParameterByName, "name" may
// be either the parameter's model name or its SLHA card location. Returns
// false if "name" is unknown.
bool getParameterByName(std::string name, double* real, double* imag);
// Drop all overrides previously set via setParameterByName(), so that the
// next setIndependentParameters(slha) call starts from a clean slate (e.g.
// when (re-)initializing from a param card). Does not itself recompute
// anything; the caller is expected to call setIndependentParameters()/
// setIndependentCouplings() afterward (as CPPProcess's constructor does).
void resetParameterOverrides();

private:

// If "name" is of the form "<block> <index> [<index2>]", resolve it (via
// m_slha_key_to_name) to the corresponding parameter's model name and return
// true; otherwise leave "resolved" untouched and return false.
bool resolveSlhaKey(const std::string& name, std::string& resolved) const;

// Name-keyed lookup tables populated by registerParameters(), backing
// setParameterByName()/getParameterByName() above.
void registerParameters();
std::map<std::string, double*> m_real_params;
std::map<std::string, std::complex<double> *> m_complex_params;
std::set<std::string> m_settable_params;
std::map<std::string, double> m_param_overrides;
std::map<std::string, std::string> m_slha_key_to_name;
SLHAReader m_slha;
};

#endif // Parameters_%(model_name)s_H
