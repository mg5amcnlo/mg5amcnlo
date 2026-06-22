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
    const int n_components = %(broken_sym_ncomponents)d;
    const int n_entries = %(broken_sym_nentries)d;
    const int comp_beg[n_components] = {%(broken_sym_component_starts)s};
    const int comp_end[n_components] = {%(broken_sym_component_ends)s};
    const int comp_old[n_components] = {%(broken_sym_component_old_factors)s};
    const int pid_list[n_entries] = {%(broken_sym_pid_list)s};
    const int block_start[n_entries] = {%(broken_sym_block_starts)s};
    const int block_len[n_entries] = {%(broken_sym_block_lengths)s};
    int pid_work[n_entries];
    for (int i = 0; i < n_entries; i++)
        pid_work[i] = pid_list[i];

    int total_factor = 1;
    for (int icomp = 0; icomp < n_components; icomp++)
    {
        int old_factor = comp_old[icomp];
        if (comp_old[icomp] > 1)
        {
        for (int i = comp_beg[icomp] - 1; i < comp_end[icomp]; i++)
        {
            if (pid_work[i] == 0)
                continue;
            int n_tot = 1;
            for (int j = i + 1; j < comp_end[icomp]; j++)
            {
                if (pid_work[i] != pid_work[j])
                    continue;
                bool same_block = (block_len[i] == block_len[j]);
                for (int k = 0; same_block && k < block_len[i]; k++)
                {
                    if (flavor[block_start[i] - 1 + k] != flavor[block_start[j] - 1 + k])
                        same_block = false;
                }
                if (same_block)
                {
                    pid_work[j] = 0;
                    n_tot = n_tot + 1;
                    old_factor = old_factor / n_tot;
                }
            }
        }
        }
        total_factor = total_factor * old_factor;
    }
    return total_factor;
}
