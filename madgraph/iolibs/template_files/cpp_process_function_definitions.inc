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
    pars->printIndependentParameters();
    pars->printIndependentCouplings();
    %(initProc_lines)s
} 

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour. 

void CPPProcess::sigmaKin( int* flavor) { 
    // Set the parameters which change event by event
    pars->setDependentParameters();
    pars->setDependentCouplings();
    static bool firsttime = true;
    if (firsttime){
	pars->printDependentParameters();
	pars->printDependentCouplings();
	firsttime = false;
    }

    // Reset color flows
    %(reset_jamp_lines)s
    %(sigmaKin_lines)s
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, including incoming flavour dependence. 

double CPPProcess::sigmaHat() {  
    %(sigmaHat_lines)s
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

%(all_sigmaKin)s

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

int CPPProcess::broken_sym(int* flavor)
{
    int old_factor = %(get_old_symmmetry_value)s;
    %(get_pid)s
    int nincoming = %(nincoming)d;
    for (int i = 0; i < (nexternal - nincoming); i++)
    {
        if(pid[i] == 0)
            continue;
        int n_tot = 1;
        for (int j = i + 1; j < (nexternal - nincoming); j++)
        {
            if((pid[i] == pid[j]) && (flavor[nincoming + i] == flavor[nincoming + j]))
            {
                pid[j] = 0;
                n_tot = n_tot + 1;
                old_factor = old_factor / n_tot;
            }
        }
    }
    return old_factor;
}
