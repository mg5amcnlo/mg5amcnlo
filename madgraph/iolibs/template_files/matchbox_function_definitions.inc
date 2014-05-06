//==========================================================================
// Class member functions for calculating the matrix elements for
%(process_lines)s

//--------------------------------------------------------------------------
// Initialize process. 
  
void CPPProcess::initProc(string param_card_name) {
// Instantiate the model class and set parameters that stay fixed during run
    pars = Parameters_%(model_name)s::getInstance();
    SLHAReader slha(param_card_name);
    pars->setIndependentParameters(slha);
    pars->setIndependentCouplings();
    pars->setDependentParameters();
    pars->setDependentCouplings();
    %(initProc_lines)s
} 

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

%(all_sigmaKin)s
