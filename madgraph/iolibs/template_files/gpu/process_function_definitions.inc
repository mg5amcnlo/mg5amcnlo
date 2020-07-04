//==========================================================================
// Class member functions for calculating the matrix elements for
%(process_lines)s

__constant__ int cHel[%(ncomb)i][%(nexternal)i];
__constant__ double cmME[%(nexternal)i]; 
//extern __constant__ int cPerm[4];
// %(coupling_list)s
__constant__ double cIPC[%(ncouplingstimes2)i]; // coupling ?
__constant__ double cIPD[%(nparams)i]; 


// Evaluate |M|^2 for each subprocess

%(all_sigmaKin)s


CPPProcess::CPPProcess(int numiterations, int gpublocks, int gputhreads,
                       bool verbose, bool debug)
    : m_numiterations(numiterations), gpu_nblocks(gpublocks),
      gpu_nthreads(gputhreads), dim(gpu_nblocks * gpu_nthreads), mME(%(nexternal)d, 0.00) {


  // Helicities for the process - nodim
  %(all_helicities)s
  // perm - nodim
  //static int perm[nexternal] = {0, 1, 2, 3};
}

CPPProcess::~CPPProcess() {}

const std::vector<double> &CPPProcess::getMasses() const { return mME; }

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
    %(assign_coupling)s
    cudaMemcpyToSymbol(cIPC, tIPC, %(ncouplings)i * sizeof(thrust::complex<double>));
    cudaMemcpyToSymbol(cIPD, tIPD, %(nparams)i * sizeof(double));
} 

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour. 

__global__ void sigmaKin(cudaPitchedPtr tp, double *meDevPtr, size_t mePitch) { 
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
