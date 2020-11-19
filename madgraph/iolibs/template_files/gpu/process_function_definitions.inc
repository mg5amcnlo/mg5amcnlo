//==========================================================================
// Class member functions for calculating the matrix elements for
%(process_lines)s

__constant__ int cHel[%(ncomb)i][%(nexternal)i];
//__constant__ double cmME[%(nexternal)i]; value hardcoded now
//extern __constant__ int cPerm[4];
// %(coupling_list)s
__constant__ fptype cIPC[%(ncouplingstimes2)i]; // coupling ?
__constant__ fptype cIPD[%(nparams)i]; 


// Evaluate |M|^2 for each subprocess

%(all_sigmaKin)s


CPPProcess::CPPProcess(int numiterations, int gpublocks, int gputhreads,
                       bool verbose, bool debug)
    : m_numiterations(numiterations), gpu_nblocks(gpublocks),
      gpu_nthreads(gputhreads), m_verbose(verbose), m_debug(debug),
      dim(gpu_nblocks * gpu_nthreads) {


  // Helicities for the process - nodim
  %(all_helicities)s
  gpuErrchk3(cudaMemcpyToSymbol(cHel, tHel, ncomb * nexternal * sizeof(int)));
  // perm - nodim
  //static int perm[nexternal] = {0, 1, 2, 3};
}

CPPProcess::~CPPProcess() {}

const std::vector<fptype> &CPPProcess::getMasses() const { return mME; }

//--------------------------------------------------------------------------
// Initialize process. 
  
void CPPProcess::initProc(string param_card_name) {
// Instantiate the model class and set parameters that stay fixed during run
    pars = Parameters_%(model_name)s::getInstance();
    SLHAReader slha(param_card_name, m_verbose);
    pars->setIndependentParameters(slha);
    pars->setIndependentCouplings();
    if (m_verbose) {
        pars->printIndependentParameters();
    	pars->printIndependentCouplings();
    }
    pars->setDependentParameters();
    pars->setDependentCouplings();
    %(initProc_lines)s
    %(assign_coupling)s
    gpuErrchk3(cudaMemcpyToSymbol(cIPC, tIPC, %(ncouplings)i * sizeof(thrust::complex<fptype>)));
    gpuErrchk3(cudaMemcpyToSymbol(cIPD, tIPD, %(nparams)i * sizeof(fptype)));
} 

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour. 

__global__ void sigmaKin(fptype * allmomenta, fptype * output) { 
    // Set the parameters which change event by event
    // Need to discuss this with Stefan
    //pars->setDependentParameters();
    //pars->setDependentCouplings();

    // Reset color flows
    %(reset_jamp_lines)s
    %(sigmaKin_lines)s
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
