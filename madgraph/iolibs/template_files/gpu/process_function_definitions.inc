//==========================================================================
// Class member functions for calculating the matrix elements for
%(process_lines)s

#ifdef __CUDACC__
namespace gProc
#else
namespace Proc
#endif
{

  using mgOnGpu::np4; // 4: the dimension of 4-momenta (E,px,py,pz)
  using mgOnGpu::npar; // number of particles in total (initial + final)
  using mgOnGpu::ncomb; // number of helicity combinations

#ifdef __CUDACC__
  __device__ __constant__ int cHel[ncomb][npar];
  __device__ __constant__ fptype cIPC[%(ncouplingstimes2)i];
  __device__ __constant__ fptype cIPD[%(nparams)i];
  __device__ __constant__ int cNGoodHel[1];
  __device__ __constant__ int cGoodHel[ncomb];
#else
  static int cHel[ncomb][npar];
  static fptype cIPC[%(ncouplingstimes2)i];
  static fptype cIPD[%(nparams)i];
#endif

  //--------------------------------------------------------------------------
  using mgOnGpu::nwf;
  using mgOnGpu::nw6;
  //--------------------------------------------------------------------------

// Evaluate |M|^2 for each subprocess
// NB: calculate_wavefunctions ADDS |M|^2 for a given ihel to the running sum of |M|^2 over helicities for the given event

%(all_sigmaKin)s


CPPProcess::CPPProcess(int numiterations, int gpublocks, int gputhreads,
                       bool verbose, bool debug)
    : m_numiterations(numiterations), gpu_nblocks(gpublocks),
      gpu_nthreads(gputhreads), m_verbose(verbose),
      dim(gpu_nblocks * gpu_nthreads) {


  // Helicities for the process - nodim
  %(all_helicities)s
#ifdef __CUDACC__
    checkCuda( cudaMemcpyToSymbol( cHel, tHel, ncomb * nexternal * sizeof(int) ) );
#else
    memcpy( cHel, tHel, ncomb * nexternal * sizeof(int) );
#endif

    // SANITY CHECK: GPU memory usage may be based on casts of fptype[2] to cxtype
    assert( sizeof(cxtype) == 2*sizeof(fptype) );
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
#ifdef __CUDACC__
    checkCuda(cudaMemcpyToSymbol(cIPC, tIPC, %(ncouplings)i * sizeof(cxtype)));
    checkCuda(cudaMemcpyToSymbol(cIPD, tIPD, %(nparams)i * sizeof(fptype)));
#else
    memcpy( cIPC, tIPC, %(ncouplings)i * sizeof(cxtype) );
    memcpy( cIPD, tIPD, %(nparams)i * sizeof(fptype) );
#endif

} 

//--------------------------------------------------------------------------

#ifdef __CUDACC__
  __global__
  void sigmaKin_getGoodHel( const fptype* allmomenta, // input: momenta as AOSOA[npagM][npar][4][neppM] with nevt=npagM*neppM
                            bool* isGoodHel )         // output: isGoodHel[ncomb] - device array
  {
    const int nprocesses = 1; // FIXME: assume process.nprocesses == 1
    fptype meHelSum[nprocesses] = { 0 }; // all zeros
    fptype meHelSumLast = 0;
    for ( int ihel = 0; ihel < ncomb; ihel++ )
    {
      // NB: calculate_wavefunctions ADDS |M|^2 for a given ihel to the running sum of |M|^2 over helicities for the given event
      calculate_wavefunctions( ihel, allmomenta, meHelSum[0] );
      if ( meHelSum[0]!=meHelSumLast ){
         isGoodHel[ihel] = true;
         meHelSumLast = meHelSum[0];
      }
    }
  }
#endif

  //--------------------------------------------------------------------------

#ifdef __CUDACC__
  void sigmaKin_setGoodHel( const bool* isGoodHel ) // input: isGoodHel[ncomb] - host array
  {
    int nGoodHel[1] = { 0 };
    int goodHel[ncomb] = { 0 };
    for ( int ihel = 0; ihel < ncomb; ihel++ )
    {
      //std::cout << "sigmaKin_setGoodHel ihel=" << ihel << ( isGoodHel[ihel] ? " true" : " false" ) << std::endl;
      if ( isGoodHel[ihel] )
      {
        goodHel[nGoodHel[0]] = ihel;
        nGoodHel[0]++;
      }
    }
    checkCuda( cudaMemcpyToSymbol( cNGoodHel, nGoodHel, sizeof(int) ) );
    checkCuda( cudaMemcpyToSymbol( cGoodHel, goodHel, ncomb*sizeof(int) ) );
  }
#endif



//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour. 

__global__ void sigmaKin(const fptype * allmomenta, fptype * allMEs
#ifndef __CUDACC__
                 , const int nevt          // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
#endif
) { 
    // Set the parameters which change event by event
    // Need to discuss this with Stefan
    //pars->setDependentParameters();
    //pars->setDependentCouplings();

#ifndef __CUDACC__
    const int maxtry = 10;
    static unsigned long long sigmakin_itry = 0; // first iteration over nevt events
    static bool sigmakin_goodhel[ncomb] = { false };
#endif

    // Reset color flows
    %(reset_jamp_lines)s

    //start sigmakin_lines
    %(sigmaKin_lines)s
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
