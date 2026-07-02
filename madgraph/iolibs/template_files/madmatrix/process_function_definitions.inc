! Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
! Created by: J. Alwall (Jul 2010) for the MG5aMC CPP backend.
!==========================================================================
! Copyright (C) 2020-2026 CERN and UCLouvain.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Modified originally by: A. Valassi (Sep 2021) for the MG5aMC CUDACPP plugin.
! Further modified by: J. Teig, A. Valassi, Z. Wettersten (2021-2025).
! Integrated with the MadGraph7 project in Feb 2026.
!==========================================================================
//==========================================================================
// Class member functions for calculating the matrix elements for
%(process_lines)s

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  __device__ INLINE unsigned int getChannelId( const unsigned int* allChannelIds
#ifndef MGONGPUCPP_GPUIMPL
                                               ,
                                               const int ievt00,
                                               bool sanityCheckMixedPrecision = true
#endif
  )
  {
    unsigned int channelId = 0; // disable multichannel single-diagram enhancement unless allChannelIds != nullptr
#ifdef MGONGPUCPP_GPUIMPL
    using CID_ACCESS = DeviceAccessChannelIds; // non-trivial access: buffer includes all events
    // SCALAR channelId for the current event (CUDA) or for the whole SIMD event page (C++)
    // The cudacpp implementation ASSUMES (and checks! #898) that all channelIds are the same in a SIMD event page
    if( allChannelIds != nullptr )
    {
      const unsigned int* channelIds = allChannelIds;                            // fix #899 (distinguish channelIds and allChannelIds)
      const uint_sv channelIds_sv = CID_ACCESS::kernelAccessConst( channelIds ); // fix #895 (compute this only once for all diagrams)
      // NB: channelIds_sv is a scalar in CUDA
      channelId = channelIds_sv;
      assert( channelId > 0 ); // SANITY CHECK: scalar channelId must be > 0 if multichannel is enabled (allChannelIds != nullptr)
    }
#else // Cuda or C++
    using CID_ACCESS = HostAccessChannelIds; // non-trivial access: buffer includes all events
    // SCALAR channelId for the whole SIMD neppV2 event page (C++), i.e. one or two neppV event page(s)
    // The cudacpp implementation ASSUMES (and checks! #898) that all channelIds are the same in a neppV2 SIMD event page
    // **NB! in "mixed" precision, using SIMD, calculate_wavefunctions computes MEs for TWO neppV pages with a single channelId! #924
    if( allChannelIds != nullptr )
    {
      // First - and/or only - neppV page of channels (iParity=0 => ievt0 = ievt00 + 0 * neppV)
      const unsigned int* channelIds = CID_ACCESS::ieventAccessRecordConst( allChannelIds, ievt00 ); // fix bug #899/#911
      uint_sv channelIds_sv = CID_ACCESS::kernelAccessConst( channelIds );                           // fix #895 (compute this only once for all diagrams)
#ifndef MGONGPU_CPPSIMD
      // NB: channelIds_sv is a scalar in no-SIMD C++
      channelId = channelIds_sv;
#else
      // NB: channelIds_sv is a vector in SIMD C++
      channelId = channelIds_sv[0];    // element[0]
      for( int i = 1; i < neppV; ++i ) // elements[1...neppV-1]
      {
        assert( channelId == channelIds_sv[i] ); // SANITY CHECK #898: check that all events in a SIMD vector have the same channelId
      }
#endif
      assert( channelId > 0 ); // SANITY CHECK: scalar channelId must be > 0 if multichannel is enabled (allChannelIds != nullptr)
      if( sanityCheckMixedPrecision )
      {
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        // Second neppV page of channels (iParity=1 => ievt0 = ievt00 + 1 * neppV)
        const unsigned int* channelIds2 = CID_ACCESS::ieventAccessRecordConst( allChannelIds, ievt00 + neppV ); // fix bug #899/#911
        uint_v channelIds2_v = CID_ACCESS::kernelAccessConst( channelIds2 );                                    // fix #895 (compute this only once for all diagrams)
        // **NB! in "mixed" precision, using SIMD, calculate_wavefunctions computes MEs for TWO neppV pages with a single channelId! #924
        for( int i = 0; i < neppV; ++i )
        {
          assert( channelId == channelIds2_v[i] ); // SANITY CHECKS #898 #924: all events in the 2nd SIMD vector have the same channelId as that of the 1st SIMD vector
        }
#endif
      }
    }
#endif // MGONGPUCPP_GPUIMPL
    return channelId;
  }

  constexpr int np4 = CPPProcess::np4;     // dimensions of 4-momenta (E, px, py, pz)
  constexpr int nw6 = CPPProcess::nw6;       // dimensions of each wavefunction (HELAS KEK 91-11): e.g. 6 for e+ e- -> mu+ mu- (fermions and vectors)
  constexpr int npar = CPPProcess::npar;     // #particles in total (external = initial + final): e.g. 4 for e+ e- -> mu+ mu-
  constexpr int ncomb = CPPProcess::ncomb;   // #helicity combinations: e.g. 16 for e+ e- -> mu+ mu- (2**4 = fermion spin up/down ** npar)
  constexpr int ncolor = CPPProcess::ncolor; // the number of leading colors
  constexpr int nmaxflavor = CPPProcess::nmaxflavor; // the maximum number of flavor combinations

  // [NB: I am currently unable to get the right value of nwf in CPPProcess.h - will hardcode it in CPPProcess.cc instead (#644)]
  //using CPPProcess::nwf; // #wavefunctions = #external (npar) + #internal: e.g. 5 for e+ e- -> mu+ mu- (1 internal is gamma or Z)

  using Parameters_dependentCouplings::ndcoup;   // #couplings that vary event by event (depend on running alphas QCD)
  using Parameters_independentCouplings::nicoup; // #couplings that are fixed for all events (do not depend on running alphas QCD)

  // The number of SIMD vectors of events processed by calculate_jamps
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
  constexpr int nParity = 2;
#else
  constexpr int nParity = 1;
#endif

  // Physics parameters (masses, coupling, etc...)
  // For CUDA performance, hardcoded constexpr's would be better: fewer registers and a tiny throughput increase
  // However, physics parameters are user-defined through card files: use CUDA constant memory instead (issue #39)
  // [NB if hardcoded parameters are used, it's better to define them here to avoid silent shadowing (issue #263)]
  constexpr int nIPD = %(nipd)i; // SM independent parameters used in this CPPProcess.cc (FIXME? rename as sm_IndepParam?)
  // Note: in the Python code generator, nIPD == nparam, while nIPC <= nicoup, because (see #823)
  // nIPC may vary from one P*/CPPProcess.cc to another, while nicoup is defined in src/Param.h and is common to all P*
  constexpr int nIPC = %(nipc)i; // SM independent couplings used in this CPPProcess.cc (FIXME? rename as sm_IndepCoupl?)
  // nIPF are the number of SM independent flavor couplings, of type FLV_COUPLING
  constexpr int nMF = FLV_COUPLING::max_flavor;
  constexpr int nIPF = %(nipf)i;
  static_assert( nIPC <= nicoup );
  static_assert( nIPD >= 0 ); // Hack to avoid build warnings when nIPD==0 is unused
  static_assert( nIPC >= 0 ); // Hack to avoid build warnings when nIPC==0 is unused
  static_assert( nMF >= 0 );  // Hack to avoid build warnings when nMF ==0 is unused
  static_assert( nIPF >= 0 ); // Hack to avoid build warnings when nIPF==0 is unused
#ifdef MGONGPU_HARDCODE_PARAM
  %(cipdhrdcod)s
  %(cipchrdcod)s
  %(cipfhrdcod)s
#else
#ifdef MGONGPUCPP_GPUIMPL
  %(cipddevice)s
  %(cipcdevice)s
  %(cipfdevice)s
#else
  %(cipdstatic)s
  %(cipcstatic)s
  %(cipfstatic)s
#endif
#endif

  // AV Jan 2024 (PR #625): this ugly #define was the only way I found to avoid creating arrays[nBsm] in CPPProcess.cc if nBsm is 0
  // The problem is that nBsm is determined when generating Parameters.h, which happens after CPPProcess.cc has already been generated
  // For simplicity, keep this code hardcoded also for SM processes (a nullptr is needed as in the case nBsm == 0)
#ifdef MGONGPUCPP_NBSMINDEPPARAM_GT_0
#ifdef MGONGPU_HARDCODE_PARAM
  __device__ const double* bsmIndepParam = Parameters::mdl_bsmIndepParam;
#else
#ifdef MGONGPUCPP_GPUIMPL
  __device__ __constant__ double bsmIndepParam[Parameters::nBsmIndepParam];
#else
  static double bsmIndepParam[Parameters::nBsmIndepParam];
#endif
#endif
#else
#ifdef MGONGPU_HARDCODE_PARAM
  __device__ const double* bsmIndepParam = nullptr;
#else
#ifdef MGONGPUCPP_GPUIMPL
  __device__ __constant__ double* bsmIndepParam = nullptr;
#else
  static double* bsmIndepParam = nullptr;
#endif
#endif
#endif

  // Helicity combinations (and filtering of "good" helicity combinations)
#ifdef MGONGPUCPP_GPUIMPL
  __device__ __constant__ short cHel[ncomb][npar];
  __device__ __constant__ int dcNGoodHel;
  __device__ __constant__ int dcGoodHel[ncomb];
  __device__ __constant__ short cFlavors[nmaxflavor][npar];
#else
  static short cHel[ncomb][npar];
  static short cFlavors[nmaxflavor][npar];
#endif
  static int cNGoodHel;
  static int cGoodHel[ncomb];

  // Host-side flavor table: single source of truth for PDG ids (used by both the
  // constructor copy into cFlavors and the public CPPProcess::flavorPDG accessor).
%(all_flavors)s

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  class DeviceAccessJamp2
  {
  public:
    static __device__ inline fptype&
    kernelAccessIcol( fptype* buffer, const int icol )
    {
      const int nevt = gridDim.x * blockDim.x;
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      return buffer[icol * nevt + ievt];
    }
    static __device__ inline const fptype&
    kernelAccessIcolConst( const fptype* buffer, const int icol )
    {
      const int nevt = gridDim.x * blockDim.x;
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      return buffer[icol * nevt + ievt];
    }
  };
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  __device__ INLINE unsigned int
  gpu_channelId( const unsigned int* allChannelIds )
  {
    unsigned int channelId = 0; // disable multichannel single-diagram enhancement unless allChannelIds != nullptr
    using CID_ACCESS = DeviceAccessChannelIds; // non-trivial access: buffer includes all events
    // SCALAR channelId for the current event (CUDA)
    if( allChannelIds != nullptr )
    {
      const unsigned int* channelIds = allChannelIds;                            // fix #899 (distinguish channelIds and allChannelIds)
      const uint_sv channelIds_sv = CID_ACCESS::kernelAccessConst( channelIds ); // fix #895 (compute this only once for all diagrams)
      // NB: channelIds_sv is a scalar in CUDA
      channelId = channelIds_sv;
      assert( channelId > 0 ); // SANITY CHECK: scalar channelId must be > 0 if multichannel is enabled (allChannelIds != nullptr)
    }
    return channelId;
  }
#endif

  //--------------------------------------------------------------------------
%(all_sigmaKin)s
  //--------------------------------------------------------------------------

  CPPProcess::CPPProcess( bool verbose,
                          bool debug )
    : m_verbose( verbose )
    , m_debug( debug )
#ifndef MGONGPU_HARDCODE_PARAM
    , m_pars( 0 )
#endif
    , m_masses()
  {
    // Helicities for the process [NB do keep 'static' for this constexpr array, see issue #283]
    // *** NB There is no automatic check yet that these are in the same order as Fortran! #569 ***
%(all_helicities)s
#ifdef MGONGPUCPP_GPUIMPL
    gpuMemcpyToSymbol( cHel, tHel, ncomb * npar * sizeof( short ) );
    gpuMemcpyToSymbol( cFlavors, tFlavors, nmaxflavor * npar * sizeof( short ) );
#else
    memcpy( cHel, tHel, ncomb * npar * sizeof( short ) );
    memcpy( cFlavors, tFlavors, nmaxflavor * npar * sizeof( short ) );
#endif

    // Enable SIGFPE traps for Floating Point Exceptions
#ifdef MGONGPUCPP_DEBUG
    fpeEnable();
#endif
  }

  //--------------------------------------------------------------------------

  CPPProcess::~CPPProcess() {}

  //--------------------------------------------------------------------------

  // PDG id for the given flavor combination and particle index (host side).
  int
  CPPProcess::flavorPDG( int iflavor, int ipar )
  {
    return flavorPDGs[iflavor][ipar];
  }

  //--------------------------------------------------------------------------

#ifndef MGONGPU_HARDCODE_PARAM
  // Initialize process (with parameters read from user cards)
  void
  CPPProcess::initProc( const std::string& param_card_name )
  {
    // Instantiate the model class and set parameters that stay fixed during run
    m_pars = Parameters::getInstance();
    SLHAReader slha( param_card_name, m_verbose );
    m_pars->setIndependentParameters( slha );
    m_pars->setIndependentCouplings();
    //m_pars->setDependentParameters(); // now computed event-by-event (running alphas #373)
    //m_pars->setDependentCouplings(); // now computed event-by-event (running alphas #373)
    if( m_verbose )
    {
      m_pars->printIndependentParameters();
      m_pars->printIndependentCouplings();
      //m_pars->printDependentParameters(); // now computed event-by-event (running alphas #373)
      //m_pars->printDependentCouplings(); // now computed event-by-event (running alphas #373)
    }
    %(initProc_lines)s
#ifdef MGONGPUCPP_GPUIMPL
    // Create the normalized color matrix in device memory
    createNormalizedColorMatrix();
#endif
    // Read physics parameters like masses and couplings from user configuration files (static: initialize once)
    // Then copy them to CUDA constant memory (issue #39) or its C++ emulation in file-scope static memory
    %(cipdassign)s
    %(cipcassign)s
    %(cipfassign)s
#ifdef MGONGPUCPP_GPUIMPL
    %(cipd2tipdSym)s
    %(cipc2tipcSym)s
    %(cipf2tipfSym)s
#ifdef MGONGPUCPP_NBSMINDEPPARAM_GT_0
    if( Parameters::nBsmIndepParam > 0 )
      gpuMemcpyToSymbol( bsmIndepParam, m_pars->mdl_bsmIndepParam, Parameters::nBsmIndepParam * sizeof( double ) );
#endif
#else
    %(cipd2tipd)s
    %(cipc2tipc)s
    %(cipf2tipf)s
#ifdef MGONGPUCPP_NBSMINDEPPARAM_GT_0
    if( Parameters::nBsmIndepParam > 0 )
      memcpy( bsmIndepParam, m_pars->mdl_bsmIndepParam, Parameters::nBsmIndepParam * sizeof( double ) );
#endif
#endif%(cipddump)s%(cipcdump)s%(cipfdump)s
    //for ( int i=0; i<Parameters::nBsmIndepParam; i++ ) std::cout << std::setprecision(17) << "m_pars->mdl_bsmIndepParam[i] = " << m_pars->mdl_bsmIndepParam[i] << std::endl;
  }
#else
  // Initialize process (with hardcoded parameters)
  void
  CPPProcess::initProc( const std::string& /*param_card_name*/ )
  {
    // Use hardcoded physics parameters
    if( m_verbose )
    {
      Parameters::printIndependentParameters();
      Parameters::printIndependentCouplings();
      //Parameters::printDependentParameters(); // now computed event-by-event (running alphas #373)
      //Parameters::printDependentCouplings(); // now computed event-by-event (running alphas #373)
    }
    %(hardcoded_initProc_lines)s
#ifdef MGONGPUCPP_GPUIMPL
    // Create the normalized color matrix in device memory
    createNormalizedColorMatrix();
#endif
  }
#endif

  //--------------------------------------------------------------------------

  // Retrieve the compiler that was used to build this module
  const std::string
  CPPProcess::getCompiler()
  {
    std::stringstream out;
    // HIP version (HIPCC)
    // [Use __HIPCC__ instead of MGONGPUCPP_GPUIMPL here!]
    // [This tests if 'hipcc' was used even to build a .cc file, even if not necessarily 'nvcc -x cu' for a .cu file]
    // [Check 'hipcc -dM -E -x hip -I ../../src CPPProcess.cc | grep HIP']
#ifdef __HIPCC__
#if defined HIP_VERSION_MAJOR && defined HIP_VERSION_MINOR && defined HIP_VERSION_PATCH
    out << "hipcc " << HIP_VERSION_MAJOR << "." << HIP_VERSION_MINOR << "." << HIP_VERSION_PATCH;
#else
    out << "hipcc UNKNOWN";
#endif
    out << " (";
#endif
    // CUDA version (NVCC)
    // [Use __NVCC__ instead of MGONGPUCPP_GPUIMPL here!]
    // [This tests if 'nvcc' was used even to build a .cc file, even if not necessarily 'nvcc -x cu' for a .cu file]
    // [Check 'nvcc --compiler-options -dM -E dummy.c | grep CUDA': see https://stackoverflow.com/a/53713712]
#ifdef __NVCC__
#if defined __CUDACC_VER_MAJOR__ && defined __CUDACC_VER_MINOR__ && defined __CUDACC_VER_BUILD__
    out << "nvcc " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__;
#else
    out << "nvcc UNKNOWN";
#endif
    out << " (";
#endif
    // ICX version (either as CXX or as host compiler inside NVCC)
#if defined __INTEL_COMPILER
#error "icc is no longer supported: please use icx"
#elif defined __INTEL_LLVM_COMPILER // alternative: __INTEL_CLANG_COMPILER
    out << "icx " << __INTEL_LLVM_COMPILER;
#ifdef __NVCC__
    out << ", ";
#else
    out << " (";
#endif
#endif
    // CLANG version (either as CXX or as host compiler inside NVCC or inside ICX)
#if defined __clang__
#if defined __clang_major__ && defined __clang_minor__ && defined __clang_patchlevel__
#ifdef __APPLE__
    out << "Apple clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#else
    out << "clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
    /*
    // === AV 26-Jan-2024 DISABLE THIS CODE (START)
    // === AV 26-Jan-2024 First, it is totally wrong to assume that the CXX environment variable is used in the build!
    // === AV 26-Jan-2024 Second and worse, here we need build time values, while CXX in this code is evaluated at runtime!
    // GCC toolchain version inside CLANG
    std::string tchainout;
    std::string tchaincmd = "readelf -p .comment $(${CXX} -print-libgcc-file-name) |& grep 'GCC: (GNU)' | grep -v Warning | sort -u | awk '{print $5}'";
    std::unique_ptr<FILE, decltype( &pclose )> tchainpipe( popen( tchaincmd.c_str(), "r" ), pclose );
    if( !tchainpipe ) throw std::runtime_error( "`readelf ...` failed?" );
    std::array<char, 128> tchainbuf;
    while( fgets( tchainbuf.data(), tchainbuf.size(), tchainpipe.get() ) != nullptr ) tchainout += tchainbuf.data();
    tchainout.pop_back(); // remove trailing newline
#if defined __NVCC__ or defined __INTEL_LLVM_COMPILER
    out << ", gcc " << tchainout;
#else
    out << " (gcc " << tchainout << ")";
#endif
    // === AV 26-Jan-2024 DISABLE THIS CODE (END)
    */
#endif
#else
    out << "clang UNKNOWKN";
#endif
#else
    // GCC version (either as CXX or as host compiler inside NVCC)
#if defined __GNUC__ && defined __GNUC_MINOR__ && defined __GNUC_PATCHLEVEL__
    out << "gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#else
    out << "gcc UNKNOWKN";
#endif
#endif
#if defined __HIPCC__ or defined __NVCC__ or defined __INTEL_LLVM_COMPILER
    out << ")";
#endif
    return out.str();
  }

  //--------------------------------------------------------------------------

  __global__ void /* clang-format off */
  computeDependentCouplings( const fptype* allgs, // input: Gs[nevt]
                             fptype* allcouplings // output: couplings[nevt*ndcoup*2]
#ifndef MGONGPUCPP_GPUIMPL
                             , const int nevt     // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
#endif
  ) /* clang-format on */
  {
#ifdef MGONGPUCPP_GPUIMPL
    using namespace mg5amcGpu;
    using G_ACCESS = DeviceAccessGs;
    using C_ACCESS = DeviceAccessCouplings;
    G2COUP<G_ACCESS, C_ACCESS>( allgs, allcouplings, bsmIndepParam );
#else
    using namespace mg5amcCpu;
    using G_ACCESS = HostAccessGs;
    using C_ACCESS = HostAccessCouplings;
    for( int ipagV = 0; ipagV < nevt / neppV; ++ipagV )
    {
      const int ievt0 = ipagV * neppV;
      const fptype* gs = MemoryAccessGs::ieventAccessRecordConst( allgs, ievt0 );
      fptype* couplings = MemoryAccessCouplings::ieventAccessRecord( allcouplings, ievt0 );
      G2COUP<G_ACCESS, C_ACCESS>( gs, couplings, bsmIndepParam );
    }
#endif
  }

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  void /* clang-format off */
  sigmaKin_getGoodHel( const fptype* allmomenta,       // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,     // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec, // input: indices of the flavor combinations
                       fptype* allMEs,                 // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype* allNumerators,          // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype* allDenominators,        // output: multichannel denominators[nevt], running_sum_over_helicities
                       fptype_sv* allJamps,            // tmp: jamp[ncolor*2*nevt] _for one helicity_ (reused in the getGoodHel helicity loop)
                       bool* isGoodHel,                // output: isGoodHel[ncomb] - host array
                       const int nevt )                // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  { /* clang-format on */
    const int maxtry0 = 16;
    fptype hstMEs[maxtry0];
    const int maxtry = std::min( maxtry0, nevt ); // 16, but at most nevt (avoid invalid memory access if nevt<maxtry0)
    //std::cout << "sigmaKin_getGoodHel nevt=" << nevt << " maxtry=" << maxtry << std::endl;
    // Per-flavor good-helicity union (merged flavors, e.g. PDG=81): sample every
    // flavor combination on the same momenta and OR the result, so cGoodHel is
    // the union over all flavors (see the C++ branch below for the rationale).
    for( int ihel = 0; ihel < ncomb; ihel++ ) isGoodHel[ihel] = false;
    (void)iflavorVec; // flavor is forced below to scan every flavor combination
    unsigned int hstFlavorVec[maxtry0] = {};
    unsigned int* devFlavorVec = nullptr;
    gpuMalloc( (void**)&devFlavorVec, maxtry * sizeof( unsigned int ) );
    for( int iflav = 0; iflav < nmaxflavor; ++iflav )
    {
    for( int i = 0; i < maxtry; ++i ) hstFlavorVec[i] = (unsigned int)iflav;
    gpuMemcpy( devFlavorVec, hstFlavorVec, maxtry * sizeof( unsigned int ), gpuMemcpyHostToDevice );
    for( int ihel = 0; ihel < ncomb; ihel++ )
    {
      const int gpublocks = 1;
      const int gputhreads = maxtry;
      constexpr int nOneHel = 1; // use a jamp buffer for a single helicity
      gpuMemcpyToSymbol( dcNGoodHel, &nOneHel, sizeof( int ) );
      // NEW IMPLEMENTATION OF GETGOODHEL (#630): RESET THE RUNNING SUM OVER HELICITIES TO 0 BEFORE ADDING A NEW HELICITY
      gpuMemset( allMEs, 0, maxtry * sizeof( fptype ) );
      // NB: color_sum ADDS |M|^2 for one helicity to the running sum of |M|^2 over helicities for the given event(s)
      constexpr fptype_sv* allJamp2s = nullptr; // no need for color selection during helicity filtering
      gpuLaunchKernel( calculate_jamps, gpublocks, gputhreads, ihel, allmomenta, allcouplings, devFlavorVec, allJamps, false, allNumerators, allDenominators, allJamp2s, gpublocks * gputhreads, false );
      gpuLaunchKernel( color_sum_kernel, gpublocks, gputhreads, allMEs, allJamps, nOneHel, 0 );
      gpuMemcpy( hstMEs, allMEs, maxtry * sizeof( fptype ), gpuMemcpyDeviceToHost );
      //std::cout << "sigmaKin_getGoodHel ihel=" << ihel << std::endl;
      for( int ievt = 0; ievt < maxtry; ++ievt )
      {
        //std::cout << "sigmaKin_getGoodHel hstMEs[ievt]=" << hstMEs[ievt] << std::endl;
        if( hstMEs[ievt] != 0 ) // NEW IMPLEMENTATION OF GETGOODHEL (#630): COMPARE EACH HELICITY CONTRIBUTION TO 0
        {
          //if ( !isGoodHel[ihel] ) std::cout << "sigmaKin_getGoodHel ihel=" << ihel << " TRUE" << std::endl;
          isGoodHel[ihel] = true;
        }
      }
    }
    } // end loop over flavor combinations (per-flavor good-helicity union)
    gpuFree( devFlavorVec );
  }
#else
  void
  sigmaKin_getGoodHel( const fptype* allmomenta,       // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,     // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec, // input: indices of the flavor combinations
                       fptype* allMEs,                 // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype* allNumerators,          // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype* allDenominators,        // output: multichannel denominators[nevt], running_sum_over_helicities
                       bool* isGoodHel,                // output: isGoodHel[ncomb] - host array
                       const int nevt )                // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  {
    //assert( (size_t)(allmomenta) %% mgOnGpu::cppAlign == 0 ); // SANITY CHECK: require SIMD-friendly alignment [COMMENT OUT TO TEST MISALIGNED ACCESS]
    //assert( (size_t)(allMEs) %% mgOnGpu::cppAlign == 0 ); // SANITY CHECK: require SIMD-friendly alignment [COMMENT OUT TO TEST MISALIGNED ACCESS]
    // Allocate arrays at build time to contain at least 16 events (or at least neppV events if neppV>16, e.g. in future VPUs)
    constexpr int maxtry0 = std::max( 16, neppV ); // 16, but at least neppV (otherwise the npagV loop does not even start)
    // Loop over only nevt events if nevt is < 16 (note that nevt is always >= neppV)
    assert( nevt >= neppV );
    const int maxtry = std::min( maxtry0, nevt ); // 16, but at most nevt (avoid invalid memory access if nevt<maxtry0)
    //std::cout << "sigmaKin_getGoodHel nevt=" << nevt << " maxtry=" << maxtry << std::endl;
    // HELICITY LOOP: CALCULATE WAVEFUNCTIONS
    const int npagV = maxtry / neppV;
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT /* clang-format off */
    // Mixed fptypes #537: float for color algebra and double elsewhere
    // Delay color algebra and ME updates (only on even pages)
    assert( npagV %% 2 == 0 ); // SANITY CHECK for mixed fptypes: two neppV-pages are merged to one 2*neppV-page
    const int npagV2 = npagV / 2; // loop on two SIMD pages (neppV events) at a time
#else
    const int npagV2 = npagV; // loop on one SIMD page (neppV events) at a time
#endif /* clang-format on */
    // Per-flavor good-helicity union (merged flavors, e.g. PDG=81): a helicity
    // that vanishes for the sampled flavor may be non-zero for another merged
    // flavor and must not be dropped. Sample every flavor combination on the
    // same momenta and OR the result, so cGoodHel becomes the union over all
    // flavors (extra helicities simply contribute 0 for a given flavor at run
    // time, exactly as in the scalar standalone_cpp per-flavor good-hel filter).
    for( int ihel = 0; ihel < ncomb; ihel++ ) isGoodHel[ihel] = false;
    (void)iflavorVec; // flavor is forced below to scan every flavor combination
    unsigned int hgFlavorVec[maxtry0] = {}; // forced single-flavor index buffer
    for( int iflav = 0; iflav < nmaxflavor; ++iflav )
    {
    for( int i = 0; i < maxtry0; ++i ) hgFlavorVec[i] = (unsigned int)iflav;
    for( int ipagV2 = 0; ipagV2 < npagV2; ++ipagV2 )
    {
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT /* clang-format off */
      const int ievt00 = ipagV2 * neppV * 2; // loop on two SIMD pages (neppV events) at a time
#else
      const int ievt00 = ipagV2 * neppV; // loop on one SIMD page (neppV events) at a time
#endif /* clang-format on */
      for( int ihel = 0; ihel < ncomb; ihel++ )
      {
        //std::cout << "sigmaKin_getGoodHel ihel=" << ihel << std::endl;
        // NEW IMPLEMENTATION OF GETGOODHEL (#630): RESET THE RUNNING SUM OVER HELICITIES TO 0 BEFORE ADDING A NEW HELICITY
        for( int ieppV = 0; ieppV < neppV; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          allMEs[ievt] = 0;
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          const int ievt2 = ievt00 + ieppV + neppV;
          allMEs[ievt2] = 0;
#endif
        }
        constexpr fptype_sv* jamp2_sv = nullptr; // no need for color selection during helicity filtering
        //std::cout << "sigmaKin_getGoodHel ihel=" << ihel << ( isGoodHel[ihel] ? " true" : " false" ) << std::endl;
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        cxtype_sv jamp_sv[2 * ncolor] = {}; // all zeros
#else
        cxtype_sv jamp_sv[ncolor] = {};  // all zeros
#endif
        calculate_jamps( ihel, allmomenta, allcouplings, hgFlavorVec, jamp_sv, false, allNumerators, allDenominators, jamp2_sv, ievt00 ); //maxtry?
        color_sum_cpu( allMEs, jamp_sv, ievt00 );
        for( int ieppV = 0; ieppV < neppV; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          //std::cout << "sigmaKin_getGoodHel allMEs[ievt]=" << allMEs[ievt] << std::endl;
          if( allMEs[ievt] != 0 ) // NEW IMPLEMENTATION OF GETGOODHEL (#630): COMPARE EACH HELICITY CONTRIBUTION TO 0
          {
            //if ( !isGoodHel[ihel] ) std::cout << "sigmaKin_getGoodHel ihel=" << ihel << " TRUE" << std::endl;
            isGoodHel[ihel] = true;
          }
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          const int ievt2 = ievt00 + ieppV + neppV;
          if( allMEs[ievt2] != 0 ) // NEW IMPLEMENTATION OF GETGOODHEL (#630): COMPARE EACH HELICITY CONTRIBUTION TO 0
          {
            //if ( !isGoodHel[ihel] ) std::cout << "sigmaKin_getGoodHel ihel=" << ihel << " TRUE" << std::endl;
            isGoodHel[ihel] = true;
          }
#endif
        }
      }
    }
    } // end loop over flavor combinations (per-flavor good-helicity union)
  }
#endif

  //--------------------------------------------------------------------------

  int                                          // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( const bool* isGoodHel ) // input: isGoodHel[ncomb] - host array (CUDA and C++)
  {
    int nGoodHel = 0;
    int goodHel[ncomb] = { 0 }; // all zeros https://en.cppreference.com/w/c/language/array_initialization#Notes
    for( int ihel = 0; ihel < ncomb; ihel++ )
    {
      //std::cout << "sigmaKin_setGoodHel ihel=" << ihel << ( isGoodHel[ihel] ? " true" : " false" ) << std::endl;
      if( isGoodHel[ihel] )
      {
        goodHel[nGoodHel] = ihel;
        nGoodHel++;
      }
    }
#ifdef MGONGPUCPP_GPUIMPL
    gpuMemcpyToSymbol( dcNGoodHel, &nGoodHel, sizeof( int ) );
    gpuMemcpyToSymbol( dcGoodHel, goodHel, ncomb * sizeof( int ) );
#endif
    cNGoodHel = nGoodHel;
    for( int ihel = 0; ihel < ncomb; ihel++ ) cGoodHel[ihel] = goodHel[ihel];
    return nGoodHel;
  }

  //--------------------------------------------------------------------------

  __device__ int
  broken_symmetry_factor( const int iflavor )
  {
    // Decay-aware identical-particle (broken-)symmetry factor, shared with the
    // Fortran / standalone_cpp exporters (_get_broken_symmetry_data). Two
    // entries contribute to the over-counting factor only when they have the
    // same top-level PID AND the same full decay/flavour block, so e.g. two Z
    // bosons decaying to different families are correctly distinguished.
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
                    if (cFlavors[iflavor][block_start[i] - 1 + k] != cFlavors[iflavor][block_start[j] - 1 + k])
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

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL /* clang-format off */
  __global__ void
  normalise_output( fptype* allMEs,                    // output: allMEs[nevt], |M|^2 running_sum_over_helicities
                    const unsigned int* iflavorVec,
                    fptype* ghelAllNumerators,         // input/tmp: allNumerators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
                    fptype* ghelAllDenominators,       // input/tmp: allNumerators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
                    const unsigned int* allChannelIds, // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable SDE enhancement (fix #899/#911)
                    bool storeChannelWeights,           // if true, compute final multichannel weights
                    bool mulChannelWeight,             // if true, multiply matrix element by channel weight
                    const fptype globaldenom) /* clang-format on */
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread)
    allMEs[ievt] = allMEs[ievt] * broken_symmetry_factor(iflavorVec[ievt]) / globaldenom;
    const int nevt = gridDim.x * blockDim.x;
    if( storeChannelWeights ) // fix segfault #892 (not 'channelIds[0] != 0')
    {
      fptype* totAllNumerators = ghelAllNumerators;     // reuse "helicity #0" buffer to compute the total over all helicities
      fptype* totAllDenominators = ghelAllDenominators; // reuse "helicity #0" buffer to compute the total over all helicities
      for( int ighel = 1; ighel < dcNGoodHel; ighel++ ) // NB: the loop starts at ighel=1
      {
        fptype* hAllDenominators = ghelAllDenominators + ighel * nevt;
        totAllDenominators[ievt] += hAllDenominators[ievt];
        fptype* hAllNumerators = ghelAllNumerators + ( ievt + ighel * nevt ) * processConfig::ndiagrams;
        fptype* firstNumerator = ghelAllNumerators + ievt * processConfig::ndiagrams;
        for( int idiag = 0; idiag < processConfig::ndiagrams; ++idiag )
        {
          firstNumerator[idiag] += hAllNumerators[idiag];
        }
      }
      if( mulChannelWeight )
      {
        unsigned int channelId = allChannelIds[ievt];
        allMEs[ievt] *= totAllNumerators[channelId - 1 + ievt * processConfig::ndiagrams] / totAllDenominators[ievt];
      }
    }
    return;
  }
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  add_and_select_hel( int* allselhel,          // output: helicity selection[nevt]
                      const fptype* allrndhel, // input: random numbers[nevt] for helicity selection
                      fptype* ghelAllMEs,      // input/tmp: allMEs for nGoodHel <= ncomb individual/runningsum helicities (index is ighel)
                      fptype* allMEs,          // output: allMEs[nevt], final sum over helicities
                      const int nevt )         // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread)
    // Compute the sum of MEs over all good helicities (defer this after the helicity loop to avoid breaking streams parall>
    for( int ighel = 0; ighel < dcNGoodHel; ighel++ )
    {
      allMEs[ievt] += ghelAllMEs[ighel * nevt + ievt];
      ghelAllMEs[ighel * nevt + ievt] = allMEs[ievt]; // reuse the buffer to store the running sum for helicity selection
    }
    // Event-by-event random choice of helicity #403
    //printf( "select_hel: ievt=%%4d rndhel=%%f\n", ievt, allrndhel[ievt] );
    for( int ighel = 0; ighel < dcNGoodHel; ighel++ )
    {
      if( allrndhel[ievt] < ( ghelAllMEs[ighel * nevt + ievt] / allMEs[ievt] ) )
      {
        const int ihelF = dcGoodHel[ighel] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
        allselhel[ievt] = ihelF;
        //printf( "select_hel: ievt=%%4d ihel=%%4d\n", ievt, ihelF );
        break;
      }
    }
    return;
  }
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  select_col_and_diag( int* allselcol,                    // output: color selection[nevt]
                       unsigned int* allDiagramIdsOut,    // output: sampled diagram ids
                       const fptype* allrndcol,           // input: random numbers[nevt] for color selection
                       const fptype* allrnddiagram,       // input: random numbers[nevt] for diagram selection
                       const unsigned int* allChannelIds, // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable SDE enhancement (fix #899/#911)
                       const fptype_sv* allJamp2s,        // input: jamp2[ncolor][nevt] for color choice (nullptr if disabled)
                       const fptype* allNumerators,       // input: all numerators
                       const fptype* allDenominators,     // input: all denominators
                       const int nevt )                   // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread)
    // SCALAR channelId for the current event (CUDA)
    unsigned int channelId = gpu_channelId( allChannelIds );
    // Event-by-event random choice of color #402

    // Event-by-event random choice of channel
    if( allrnddiagram != nullptr )
    {
      fptype numerator_sum = 0., normalization = 0.;
      for( unsigned int ichan = 0; ichan < mgOnGpu::nchannels; ichan++ )
      {
        if( mgOnGpu::channel2iconfig[ichan] == -1 ) continue;
        normalization += allNumerators[ievt * processConfig::ndiagrams + ichan];
      }
      channelId = mgOnGpu::nchannels;
      for( unsigned int ichan = 0; ichan < mgOnGpu::nchannels; ichan++ )
      {
        if( mgOnGpu::channel2iconfig[ichan] == -1 ) continue;
        numerator_sum += allNumerators[ievt * processConfig::ndiagrams + ichan];
        if( allrnddiagram[ievt] < numerator_sum / normalization )
        {
          channelId = ichan + 1;
          break;
        }
      }
      allDiagramIdsOut[ievt] = channelId;
    }

    if( channelId != 0 ) // no event-by-event choice of color if channelId == 0 (fix FPE #783)
    {
      if( channelId > mgOnGpu::nchannels )
      {
        printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%%d which is greater than nchannels=%%d\n", channelId, mgOnGpu::nchannels );
        assert( channelId <= mgOnGpu::nchannels ); // SANITY CHECK #919 #910
      }
      // Determine the jamp2 for this event (TEMPORARY? could do this with a dedicated memory accessor instead...)
      fptype_sv jamp2_sv[ncolor] = { 0 };
      assert( allJamp2s != nullptr ); // sanity check
      using J2_ACCESS = DeviceAccessJamp2;
      for( int icolC = 0; icolC < ncolor; icolC++ )
        jamp2_sv[icolC] = J2_ACCESS::kernelAccessIcolConst( allJamp2s, icolC );
      // NB (see #877): in the array channel2iconfig, the input index uses C indexing (channelId -1), the output index uses F indexing (iconfig)
      // NB (see #917): mgOnGpu::channel2iconfig returns an int (which may be -1), not an unsigned int!
      const int iconfig = mgOnGpu::channel2iconfig[channelId - 1]; // map N_diagrams to N_config <= N_diagrams configs (fix LHE color mismatch #856: see also #826, #852, #853)
      if( iconfig <= 0 )
      {
        printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%%d which has no associated SDE iconfig\n", channelId );
        assert( iconfig > 0 ); // SANITY CHECK #917
      }
      else if( iconfig > (int)mgOnGpu::nconfigSDE )
      {
        printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%%d (invalid SDE iconfig=%%d\n > nconfig=%%d)", channelId, iconfig, mgOnGpu::nconfigSDE );
        assert( iconfig <= (int)mgOnGpu::nconfigSDE ); // SANITY CHECK #917
      }
      fptype targetamp[ncolor] = { 0 };
      // NB (see #877): explicitly use 'icolC' rather than 'icol' to indicate that icolC uses C indexing in [0, N_colors-1]
      for( int icolC = 0; icolC < ncolor; icolC++ )
      {
        if( icolC == 0 )
          targetamp[icolC] = 0;
        else
          targetamp[icolC] = targetamp[icolC - 1];
        // NB (see #877): in the array icolamp, the input index uses C indexing (iconfig -1)
        if( mgOnGpu::icolamp[iconfig - 1][icolC] ) targetamp[icolC] += jamp2_sv[icolC];
      }
      //printf( "sigmaKin: ievt=%%4d rndcol=%%f\n", ievt, allrndcol[ievt] );
      for( int icolC = 0; icolC < ncolor; icolC++ )
      {
        if( allrndcol[ievt] < ( targetamp[icolC] / targetamp[ncolor - 1] ) )
        {
          allselcol[ievt] = icolC + 1; // NB Fortran [1,ncolor], cudacpp [0,ncolor-1]
          //printf( "sigmaKin: ievt=%%d icol=%%d\n", ievt, icolC+1 );
          break;
        }
      }
    }
    else
    {
      allselcol[ievt] = 0; // no color selected in Fortran range [1,ncolor] if channelId == 0 (see #931)
    }
    return;
  }
#endif

  //--------------------------------------------------------------------------
  // Evaluate |M|^2, part independent of incoming flavour

  void /* clang-format off */
  sigmaKin( const fptype* allmomenta,           // input: momenta[nevt*npar*4]
            const fptype* allcouplings,         // input: couplings[nevt*ndcoup*2]
            const unsigned int* iflavorVec,     // input: indices of the flavor combinations
            const fptype* allrndhel,            // input: random numbers[nevt] for helicity selection
            const fptype* allrndcol,            // input: random numbers[nevt] for color selection
            const unsigned int* allChannelIds,  // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable single-diagram enhancement (fix #899/#911)
            const fptype* allrnddiagram,        // input: random numbers[nevt] for diagram sampling
            fptype* allMEs,                     // output: allMEs[nevt], |M|^2 final_avg_over_helicities
            int* allselhel,                     // output: helicity selection[nevt]
#ifdef MGONGPUCPP_GPUIMPL
            int* allselcol,                     // output: helicity selection[nevt]
            fptype* colAllJamp2s,               // tmp: allJamp2s super-buffer for ncolor individual colors, running sum over colors and helicities
            fptype* ghelAllNumerators,          // tmp: allNumerators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            fptype* ghelAllDenominators,        // tmp: allDenominators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            unsigned int* allDiagramIdsOut,     // output: multichannel channelIds[nevt] (1 to #diagrams)
            bool mulChannelWeight,              // if true, multiply channel weight to ME output
            fptype* ghelAllMEs,                 // tmp: allMEs super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            fptype* ghelAllJamps,               // tmp: jamp[2*ncolor*nGoodHel*nevt] super-buffer for nGoodHel <= ncomb individual helicities
            fptype2* ghelAllBlasTmp,            // tmp: allBlasTmp super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            gpuBlasHandle_t* pBlasHandle,       // input: cuBLAS/hipBLAS handle
            gpuStream_t* ghelStreams,           // input: cuda streams (index is ighel: only the first nGoodHel <= ncomb are non-null)
            const bool async,
            const int gpublocks,                // input: cuda gpublocks
            const int gputhreads                // input: cuda gputhreads
#else
            int* allselcol,                     // output: helicity selection[nevt]
            fptype* allNumerators,              // output: multichannel numerators[nevt], running_sum_over_helicities
            fptype* allDenominators,            // output: multichannel denominators[nevt], running_sum_over_helicities
            unsigned int* allDiagramIdsOut,     // output: multichannel channelIds[nevt] (1 to #diagrams)
            bool mulChannelWeight,              // if true, multiply channel weight to ME output
            const int nevt                      // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
#endif
            ) /* clang-format on */
  {
    mgDebugInitialise();

    // SANITY CHECKS for cudacpp code generation (see issues #272 and #343 and PRs #619, #626, #360, #396 and #754)
    // These variable are not used anywhere else in the code and their scope is limited to this sanity check
    {
      // nprocesses == 2 may happen for "mirror processes" such as P0_uux_ttx within pp_tt012j (see PR #754)
      constexpr int nprocesses = %(nproc)i;
      static_assert( nprocesses == 1 || nprocesses == 2, "Assume nprocesses == 1 or 2" );
      constexpr int process_id = %(proc_id)i; // code generation source: %(proc_id_source)s
      static_assert( process_id == 1, "Assume process_id == 1" );
    }

    // Denominators: spins, colors and identical particles
    constexpr int helcolDenominators[1] = { %(den_factors)s }; // assume nprocesses == 1 (#272 and #343)

#ifndef MGONGPUCPP_GPUIMPL
    //assert( (size_t)(allmomenta) %% mgOnGpu::cppAlign == 0 ); // SANITY CHECK: require SIMD-friendly alignment [COMMENT OUT TO TEST MISALIGNED ACCESS]
    //assert( (size_t)(allMEs) %% mgOnGpu::cppAlign == 0 ); // SANITY CHECK: require SIMD-friendly alignment [COMMENT OUT TO TEST MISALIGNED ACCESS]
    using E_ACCESS = HostAccessMatrixElements; // non-trivial access: buffer includes all events
    using NUM_ACCESS = HostAccessNumerators;   // non-trivial access: buffer includes all events
    using DEN_ACCESS = HostAccessDenominators; // non-trivial access: buffer includes all events
#endif

    // Start sigmaKin_lines
%(sigmaKin_lines)s
} // end namespace
