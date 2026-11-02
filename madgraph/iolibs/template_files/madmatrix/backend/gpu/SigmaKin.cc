// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Backend-owned driver: sigmaKin, calculate_jamps, good-helicity filtering.
// The diagram/vertex-call sequence (helas_calls) is process-specific and
// lives in the P1-generated EvaluateDiagrams.inc, #include'd below.
//
// This is a first, foundational slice: the process-independent storage the
// rest of this file operates on (cHel/cFlavors/cIPD/cIPC/cIPF/bsmIndepParam),
// its setters, and the two helpers (gpu_channelId, computeDependentCouplings)
// that don't depend on calculate_jamps/sigmaKin, which land in a later slice.

#include "SigmaKin.h"

#include "CPPProcess.h" // ProcessData.h, Parameters.h, HelAmps_<model>.h transitively
#include "ProcessTables.h"

#include "GpuRuntime.h"
#include "MemoryAccessAmplitudes.h"
#include "MemoryAccessChannelIds.h"
#include "MemoryAccessCouplings.h"
#include "MemoryAccessCouplingsFixed.h"
#include "MemoryAccessGs.h"
#include "MemoryAccessIflavorVec.h"
#include "MemoryAccessMomenta.h"
#include "MemoryAccessNumerators.h"
#include "MemoryAccessWavefunctions.h"
#include "color_sum.h" // for DeviceAccessJamp
#include "ColorData.h" // for mgOnGpu::nchannels/channel2iconfig/icolamp/nconfigSDE

#include <cfloat>
#include <cmath>
#include <vector>

namespace madmatrix
{
  using namespace ProcessData;
  using namespace ProcessTables;
  using Parameters_dependentCouplings::ndcoup;   // #couplings that vary event by event (depend on running alphas QCD)
  using Parameters_independentCouplings::nicoup; // #couplings that are fixed for all events (do not depend on running alphas QCD)

  // ncolor_flow (unlike ncolor) is not in ProcessData.h: the color sum can run
  // on a smaller (DDM) basis than the color flow probabilities do, so this stays
  // a CPPProcess-generated constant (see process_class.inc/set_color_flow_lines_cpp).
  constexpr int ncolor_flow = CPPProcess::ncolor_flow;

  // Squared split orders (nampso > 1): the jamps carry an amplitude-order index,
  // njampso = ncolor * nampso of them per helicity (EvaluateDiagrams.inc offsets each
  // amplitude onto the vector of its order), and color_sum_kernel pairs them, as
  // backend/{cpu,simd} do. The device jamp buffers and DeviceAccessJamp are sized
  // njampso, which is ncolor without split orders.

  // Helicity/flavor tables and SM parameter/coupling storage, populated once
  // by CPPProcess's constructor/initProc via the setters below.
  __device__ __constant__ short cHel[ncomb][npar];
  __device__ __constant__ int dcNGoodHel;
  __device__ __constant__ int dcGoodHel[ncomb];
  __device__ __constant__ short cFlavors[nmaxflavor][npar];
  static int cNGoodHel;
  static int cGoodHel[ncomb];
  __device__ __constant__ fptype cIPD[nIPD > 0 ? nIPD : 1];
  __device__ __constant__ fptype cIPC[nIPC > 0 ? nIPC * 2 : 1];
  __device__ __constant__ int cIPF_partner1[ProcessTables::nMF * nIPF > 0 ? ProcessTables::nMF * nIPF : 1];
  __device__ __constant__ int cIPF_partner2[ProcessTables::nMF * nIPF > 0 ? ProcessTables::nMF * nIPF : 1];
  __device__ __constant__ fptype cIPF_value[ProcessTables::nMF * nIPF * 2 > 0 ? ProcessTables::nMF * nIPF * 2 : 1];
  __device__ __constant__ double bsmIndepParam[Parameters::nBsmIndepParam > 0 ? Parameters::nBsmIndepParam : 1];
  // BWCUTOFF of the $-excluded propagators (ALOHA P1D tag), see setBwCutoff
  __device__ __constant__ fptype cBWCUTOFF = 15.;

  void setHelicitiesAndFlavors( const short* tHel, const short* tFlavors )
  {
    gpuMemcpyToSymbol( cHel, tHel, ncomb * npar * sizeof( short ) );
    gpuMemcpyToSymbol( cFlavors, tFlavors, nmaxflavor * npar * sizeof( short ) );
  }

  void setIndependentParams( const fptype* tIPD )
  {
    if( nIPD > 0 ) gpuMemcpyToSymbol( cIPD, tIPD, nIPD * sizeof( fptype ) );
  }

  void setIndependentCouplings( const cxtype* tIPC )
  {
    if( nIPC > 0 ) gpuMemcpyToSymbol( cIPC, tIPC, nIPC * sizeof( cxtype ) );
  }

  void setFlavorCouplings( const int* tIPF_partner1, const int* tIPF_partner2, const cxtype* tIPF_value )
  {
    if( nIPF == 0 ) return;
    gpuMemcpyToSymbol( cIPF_partner1, tIPF_partner1, ProcessTables::nMF * nIPF * sizeof( int ) );
    gpuMemcpyToSymbol( cIPF_partner2, tIPF_partner2, ProcessTables::nMF * nIPF * sizeof( int ) );
    gpuMemcpyToSymbol( cIPF_value, tIPF_value, ProcessTables::nMF * nIPF * sizeof( cxtype ) );
  }

  void setBsmIndepParam( const double* values, int n )
  {
    if( n > 0 ) gpuMemcpyToSymbol( bsmIndepParam, values, n * sizeof( double ) );
  }

  void setBwCutoff( const double bwcutoff )
  {
    const fptype value = bwcutoff;
    gpuMemcpyToSymbol( cBWCUTOFF, &value, sizeof( fptype ) );
  }

  //--------------------------------------------------------------------------

  // Per-event access into the ncolor_flow super-buffer of jamp2 values (one fptype_amp per event per color)
  class DeviceAccessJamp2
  {
  public:
    static __device__ inline fptype_amp&
    kernelAccessIcol( fptype_amp* buffer, const int icol )
    {
      const int nevt = gridDim.x * blockDim.x;
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      return buffer[icol * nevt + ievt];
    }
    static __device__ inline const fptype_amp&
    kernelAccessIcolConst( const fptype_amp* buffer, const int icol )
    {
      const int nevt = gridDim.x * blockDim.x;
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      return buffer[icol * nevt + ievt];
    }
  };

  //--------------------------------------------------------------------------

  // SCALAR channelId for the current event (CUDA)
  __device__ INLINE unsigned int
  gpu_channelId( const unsigned int* allChannelIds )
  {
    unsigned int channelId = 0; // disable multichannel single-diagram enhancement unless allChannelIds != nullptr
    using CID_ACCESS = DeviceAccessChannelIds; // non-trivial access: buffer includes all events
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

  //--------------------------------------------------------------------------

  __global__ void
  computeDependentCouplings( const fptype* allgs, fptype* allcouplings )
  {
    using G_ACCESS = DeviceAccessGs;
    using C_ACCESS = DeviceAccessCouplings;
    G2COUP<G_ACCESS, C_ACCESS>( allgs, allcouplings, bsmIndepParam );
  }

  //--------------------------------------------------------------------------

  // The mass an external leg marked off-shell ('*': MadSpin's decaying particles)
  // is evaluated with: the invariant mass of its momentum, as in the Fortran
  // standalone output (SQRT(P(0,i)**2-...)) and the C++ backends; one event per
  // thread. The pole mass comes first only for the code generator, which maps
  // it to its parameter.
  template<class M_ACCESS>
  __device__ inline fptype_amp
  offshell_mass( const fptype_amp poleMass, const fptype_momenta* momenta, const int ipar )
  {
    (void)poleMass;
    const fptype_momenta e = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_momenta px = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_momenta py = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_momenta pz = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    const fptype_momenta m2 = e * e - px * px - py * py - pz * pz;
    return static_cast<fptype_amp>( m2 > 0 ? sqrt( m2 ) : 0 );
  }

  //--------------------------------------------------------------------------

  // Accumulate a multichannel numerator contribution in place. All good-helicity
  // blocks/streams for a given event race on the same numerator slot (the helicity
  // dimension has been removed to save memory), so an atomicAdd is mandatory.
#define NUM_ATOMIC_ADD( DST, VAL ) atomicAdd( &( DST ), VAL )
// The --hel_recycling warm-up probe hook that a recycled output's diagram calls
// carry after every amplitude: the probe and the recycled build are CPU only
// (see backend/{cpu,simd}/SigmaKin.cc), so it is nothing here. The GPU backend
// never builds the recycled block, by design: it makes every helicity row of an
// event in one thread and keeps every amplitude of every row until the color
// flows (g g > t t~ g g g: 3.9 MB of amplitudes and 0.5 MB of wavefunctions per
// event), where a GPU thread holds one helicity of one event; and the GPU
// already runs the helicities in parallel (blockIdx.y). A recycled output
// built for cuda/hip is the plain matrix element (madmatrix.mk says so).
#define MG_HR_PROBE_AMP( NAMP )

  // Evaluate QCD partial amplitudes jamps for this given helicity from Feynman diagrams.
  // Also compute running sums over helicities adding jamp2, numerator, denominator
  // (NB: this function no longer handles matrix elements as the color sum has now been
  // moved to a separate function/kernel). This is a kernel function: it processes a
  // single event (the CUDA thread), and takes a channelId array as input. The helicity
  // may differ from one thread to the next (allihel): the external wavefunctions are
  // branch-free in the helicity (see helas.h), so the threads of a warp do not diverge.
  __global__ void /* clang-format off */
  calculate_jamps( int ihel,                           // input: helicity index [0,ncomb) of all events (if !processAllHelicities and allihel == nullptr)
                   const int* allihel,                 // input: helicity index [0,ncomb) of each event[nevt] (nullptr: ihel, or the good helicity blockIdx.y)
                   const fptype_momenta* allmomenta,   // input: momenta[nevt*npar*4]
                   const fptype* allcouplings,         // input: couplings[nevt*ndcoup*2]
                   const unsigned int* iflavorVec,     // input: indices of the flavor combinations
                   fptype_amp* allJamps,               // output: jamp[2*njampso*nevt] buffer for one helicity _within a super-buffer for nhel helicities_
                   bool storeChannelWeights,
                   fptype_amp* allNumerators,          // input/output: multichannel numerators[nevt], add helicity ihel
                   fptype_amp* allDenominators,        // input/output: multichannel denominators[nevt], add helicity ihel
                   fptype_amp* colAllJamp2s,           // output: allJamp2s[ncolor_flow][nevt] super-buffer, sum over col/hel (nullptr to disable)
                   const int nevt,                     // input: #events (nevt == ndim == gpublocks*gputhreads)
                   const int nhel,                     // input: #helicities in the allJamps super-buffer (nGoodHel, or 1 with allihel)
                   const bool processAllHelicities )   // input: if true, use blockIdx.y to index helicities
  /* clang-format on */
  {
    using M_ACCESS = DeviceAccessMomenta;         // non-trivial access: buffer includes all events
    using W_ACCESS = DeviceAccessWavefunctions;   // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using A_ACCESS = DeviceAccessAmplitudes;      // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using CD_ACCESS = DeviceAccessCouplings;      // non-trivial access (dependent couplings): buffer includes all events
    using CI_ACCESS = DeviceAccessCouplingsFixed; // TRIVIAL access (independent couplings): buffer for one event
    using CDPF_ACCESS = CI_ACCESS;                // TRIVIAL access: the running flavor couplings gathered into dpf_value for this event
    using F_ACCESS = DeviceAccessIflavorVec;      // non-trivial access: buffer includes all events
    using NUM_ACCESS = DeviceAccessNumerators;    // non-trivial access: buffer includes all events
    mgDebug( 0, __FUNCTION__ );

    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread) in grid
    if( processAllHelicities )
    {
      int ighel = blockIdx.y;
      if( allihel == nullptr ) ihel = dcGoodHel[ighel];
      allJamps = allJamps + ighel * nevt;
      // NB: the numerators buffer has NO helicity dimension anymore: all good-helicity blocks
      // for a given event accumulate in place into the same [nevt][ndiagrams] slot via atomicAdd.
      // The denominators are no longer accumulated here (derived as the sum of numerators later).
    }
    // One helicity per event (nhel == 1 and gridDim.y == 1): the given helicity of this event
    if( allihel != nullptr ) ihel = allihel[ievt];
    // The helicities of the external legs for this event, as the external wavefunctions take them
    fptype_amp_sv hel_sv[npar];
    for( int ipar = 0; ipar < npar; ipar++ ) hel_sv[ipar] = cHel[ihel][ipar];

    // Local TEMPORARY variables for a subset of Feynman diagrams in the given CUDA event
    // (ievt) [NB these variables are reused several times (and re-initialised each time)
    // within the same event]. Create memory for both momenta and wavefunctions separately,
    // and later wrap them in ALOHAOBJ.
    fptype_momenta_sv pvec_sv[nwf][np4];
    cxtype_amp_sv w_sv[nwf][nw6]; // particle wavefunctions within Feynman diagrams
    cxtype_amp_sv amp_sv[1];      // invariant amplitude for one given Feynman diagram
    ALOHAOBJ aloha_obj[nwf];
    for( int iwf = 0; iwf < nwf; iwf++ ) aloha_obj[iwf] = ALOHAOBJ{ pvec_sv[iwf], w_sv[iwf] };
    fptype_amp* amp_fp = reinterpret_cast<fptype_amp*>( amp_sv );

    // special temporary ALOHAOBJ to hold F/Vtmp values in the combined vertex functions
    // while using the FD gauge (harmless, unused, when the model doesn't need it)
    fptype_momenta_sv pvec_sv_tmp[1][np4];
    cxtype_amp_sv w_sv_tmp[1][nw6];
    ALOHAOBJ aloha_obj_tmp[1];
    aloha_obj_tmp[0] = ALOHAOBJ{ pvec_sv_tmp[0], w_sv_tmp[0] };
    cxtype_amp_sv amp_tmp_sv[1]; // to ensure proper alignment for vector instructions
    fptype_amp* amp_tmp_fp = reinterpret_cast<fptype_amp*>( amp_tmp_sv );

    // jamp: sum (for one event) of the invariant amplitudes for all Feynman diagrams in a
    // given color combination (NB: vector cxtype_v IS initialized to 0, but scalar cxtype
    // is NOT, if "= {}" is missing!)
    // (njampso = ncolor * nampso: one vector per amplitude split order, just ncolor without them)
    cxtype_amp_sv jamp_sv[njampso] = {};
    // jampTmp: partial sums of amplitudes that several color flows share, so that they are
    // computed only once (see MadMatrixUFOHelasCallWriter.build_jamp_plan); no "= {}", each
    // one is assigned before it is ever read.
    cxtype_amp_sv jampTmp_sv[ProcessTables::nb_tmp_jamp > 0 ? ProcessTables::nb_tmp_jamp : 1];

    // === Calculate wavefunctions and amplitudes for all diagrams in all processes

    constexpr size_t nxcoup = ndcoup + nIPC; // both dependent and independent couplings
    // nxcoup can be 0: a process with no alpha_s-dependent coupling whose couplings are
    // all flavor couplings (cIPF, e.g. u u~ > u u~ QED^2==4 with flavor grouping), and a
    // zero-sized array is not allowed in device code (the host compilers take it as an
    // extension): size them at least 1, as cIPD/cIPC above
    const fptype* allCOUPs[nxcoup > 0 ? nxcoup : 1];
#ifdef __CUDACC__ // this must be __CUDACC__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 186 // e.g. <<warning #186-D: pointless comparison of unsigned integer with zero>>
#endif
    for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ )
      allCOUPs[idcoup] = CD_ACCESS::idcoupAccessBufferConst( allcouplings, idcoup ); // dependent couplings, vary event-by-event
    for( size_t iicoup = 0; iicoup < nIPC; iicoup++ )
      allCOUPs[ndcoup + iicoup] = CI_ACCESS::iicoupAccessBufferConst( cIPC, iicoup ); // independent couplings, fixed for all events
#ifdef __CUDACC__ // this must be __CUDACC__
#pragma nv_diagnostic pop
#endif
    // CUDA kernels take input/output buffers with momenta/MEs for all events
    const fptype_momenta* momenta = allmomenta;
    const fptype* COUPs[nxcoup > 0 ? nxcoup : 1];
    for( size_t ixcoup = 0; ixcoup < nxcoup; ixcoup++ ) COUPs[ixcoup] = allCOUPs[ixcoup];
    fptype_amp* numerators = &allNumerators[ievt * ndiagrams];
    // Create an array of views over the Flavor Couplings
    FLV_COUPLING_ARRAY<nIPF, nMF> flvCOUPs{ cIPF_partner1, cIPF_partner2, cIPF_value };

    // Dependent (event-by-event, running-alphas) flavor couplings (Step 3): the per-flavor
    // values are NOT baked in (they run per event). Gather the current values of the
    // underlying dependent couplings for this event into dpf_value, one complex per
    // (coupling,flavor) slot, then build an ordinary value-based view over it. This is the
    // direct analogue of Fortran's FLV_xx%VAL(k)%P => GC_yyy(J). A GPU thread is one event,
    // so dpf_value is a one-event buffer, read with the trivial CDPF_ACCESS (= CI_ACCESS) by
    // the vertex routines; CD_ACCESS would add the offset of this event in the all-events
    // coupling buffer, outside dpf_value for every event past the first page.
    constexpr int ndpfbuf = ( nDPF > 0 ? nDPF * nMF * CDPF_ACCESS::flv_stride : 1 );
    fptype dpf_value[ndpfbuf]{};
    for( int idpf = 0; idpf < nDPF; idpf++ )
      for( int imf = 0; imf < nMF; imf++ )
      {
        const int idc = cDPF_idcoup[idpf * nMF + imf];
        if( idc >= 0 )
        {
          const cxtype coup = CD_ACCESS::kernelAccessConst( COUPs[idc] );
          fptype* slot = dpf_value + ( idpf * nMF + imf ) * CDPF_ACCESS::flv_stride;
          slot[0] = cxreal( coup );
          slot[1] = cximag( coup );
        }
      }
    FLV_COUPLING_ARRAY<nDPF, nMF, CDPF_ACCESS::flv_stride> flvCOUPs_dep{ cDPF_partner1, cDPF_partner2, dpf_value };

    // Reset color flows (reset jamp_sv) at the beginning of a new event or event page
    for( int i = 0; i < njampso; i++ ) { jamp_sv[i] = cxzero_sv<cxtype_amp_sv>(); }

    // Numerators for the current event (CUDA); denominators are no longer
    // accumulated here: they are derived as the sum of numerators later.
    fptype_amp_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
    // Scalar iflavor for the current event
    const unsigned int iflavor = F_ACCESS::kernelAccessConst( iflavorVec );
#include "EvaluateDiagrams.inc"
#include "ColorFlows.inc" // defines jampflow_sv[ncolor_flow], which is not jamp_sv on the DDM basis

    // *** COLOR CHOICE BELOW ***
    // Store the leading color flows for choice of color
    if( colAllJamp2s ) // disable color choice if nullptr
    {
      using J2_ACCESS = DeviceAccessJamp2;
      for( int icol = 0; icol < ncolor_flow; icol++ )
        // NB: atomicAdd is needed after moving to cuda streams with one helicity per stream!
        atomicAdd( &J2_ACCESS::kernelAccessIcol( colAllJamp2s, icol ), cxabs2( jampflow_sv[icol] ) ); // may underflow #831
    }

    // *** PREPARE OUTPUT JAMPS ***
    // allJamps already points at this helicity's slot (ighel * nevt) in the
    // [2][ncolor][nhel][nevt] super-buffer, hence ihel0 = 0; nhel stays the
    // number of helicities because it is the color stride, and color_sum reads with it.
    {
      constexpr int ihel0 = 0;
      using J_ACCESS = DeviceAccessJamp;
      for( int icol = 0; icol < njampso; icol++ )
        J_ACCESS::kernelAccessIcolIhelNhel( allJamps, icol, ihel0, nhel ) = jamp_sv[icol];
    }

    mgDebug( 1, __FUNCTION__ );
  }

#undef NUM_ATOMIC_ADD
#undef MG_HR_PROBE_AMP

  //--------------------------------------------------------------------------

  void /* clang-format off */
  sigmaKin_getGoodHel( const fptype_momenta* allmomenta, // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,       // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec,   // input: indices of the flavor combinations
                       fptype* allMEs,                   // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype_amp* allNumerators,        // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype_amp* allDenominators,      // output: multichannel denominators[nevt], running_sum_over_helicities
                       fptype_amp_sv* allJamps,          // tmp: jamp[njampso*2*nevt] _for one helicity_ (reused in the getGoodHel helicity loop)
                       bool* isGoodHel,                  // output: isGoodHel[ncomb] - host array
                       const int nevt )                  // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  { /* clang-format on */
    const int maxtry0 = 16;
    fptype hstMEs[maxtry0];
    const int maxtry = std::min( maxtry0, nevt ); // 16, but at most nevt (avoid invalid memory access if nevt<maxtry0)
    // Per-flavor good-helicity union (merged flavors, e.g. PDG=81): sample every
    // flavor combination on the same momenta and OR the result, so cGoodHel is
    // the union over all flavors (see the C++ branch below for the rationale).
    for( int ihel = 0; ihel < ncomb; ihel++ ) isGoodHel[ihel] = false;
    (void)iflavorVec; // flavor is forced below to scan every flavor combination
    unsigned int hstFlavorVec[maxtry0] = {};
    // |M|^2 of every helicity at every sampled event (of the current flavor): a
    // helicity is good if, at one event, it is above limhel/ncomb of the sum over
    // helicities there -- madevent's filter (matrix_madevent_group_v4.inc, limhel
    // its hidden run_card parameter, 1e-8 by default). Comparing to an exact 0 kept
    // the helicities whose amplitudes only cancel to rounding.
    constexpr double limhel = 1e-8;
    std::vector<fptype> helMEs( (size_t)ncomb * maxtry0, 0. );
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
      // NEW IMPLEMENTATION OF GETGOODHEL (#630): RESET THE RUNNING SUM OVER HELICITIES TO 0 BEFORE ADDING A NEW HELICITY
      gpuMemset( allMEs, 0, maxtry * sizeof( fptype ) );
      // NB: color_sum ADDS |M|^2 for one helicity to the running sum of |M|^2 over helicities for the given event(s)
      constexpr fptype_amp_sv* allJamp2s = nullptr; // no need for color selection during helicity filtering
      gpuLaunchKernel( calculate_jamps, gpublocks, gputhreads, ihel, nullptr, allmomenta, allcouplings, devFlavorVec, allJamps, false, allNumerators, allDenominators, allJamp2s, gpublocks * gputhreads, nOneHel, false );
      gpuLaunchKernel( color_sum_kernel, gpublocks, gputhreads, allMEs, allJamps, nOneHel, 0 );
      gpuMemcpy( hstMEs, allMEs, maxtry * sizeof( fptype ), gpuMemcpyDeviceToHost );
      for( int ievt = 0; ievt < maxtry; ++ievt )
        helMEs[(size_t)ihel * maxtry0 + ievt] = hstMEs[ievt];
    }
    for( int ievt = 0; ievt < maxtry; ++ievt )
    {
      fptype sumhel = 0;
      for( int ihel = 0; ihel < ncomb; ihel++ ) sumhel += std::abs( helMEs[(size_t)ihel * maxtry0 + ievt] );
      for( int ihel = 0; ihel < ncomb; ihel++ )
        if( std::abs( helMEs[(size_t)ihel * maxtry0 + ievt] ) > sumhel * limhel / ncomb ) isGoodHel[ihel] = true;
    }
    } // end loop over flavor combinations (per-flavor good-helicity union)
    gpuFree( devFlavorVec );
  }

  //--------------------------------------------------------------------------

  int                                          // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( const bool* isGoodHel ) // input: isGoodHel[ncomb] - host array (CUDA and C++)
  {
    int nGoodHel = 0;
    int goodHel[ncomb] = { 0 }; // all zeros https://en.cppreference.com/w/c/language/array_initialization#Notes
    for( int ihel = 0; ihel < ncomb; ihel++ )
    {
      if( isGoodHel[ihel] )
      {
        goodHel[nGoodHel] = ihel;
        nGoodHel++;
      }
    }
    gpuMemcpyToSymbol( dcNGoodHel, &nGoodHel, sizeof( int ) );
    gpuMemcpyToSymbol( dcGoodHel, goodHel, ncomb * sizeof( int ) );
    cNGoodHel = nGoodHel;
    for( int ihel = 0; ihel < ncomb; ihel++ ) cGoodHel[ihel] = goodHel[ihel];
    return nGoodHel;
  }

  //--------------------------------------------------------------------------

  int
  sigmaKin_getNGoodHel()
  {
    return cNGoodHel; // the host copy of dcNGoodHel
  }

  //--------------------------------------------------------------------------

  // Decay-aware identical-particle (broken-)symmetry factor, shared with the
  // Fortran / standalone_cpp exporters (_get_broken_symmetry_data). Two
  // entries contribute to the over-counting factor only when they have the
  // same top-level PID AND the same full decay/flavour block, so e.g. two Z
  // bosons decaying to different families are correctly distinguished.
  __device__ int
  broken_symmetry_factor( const int iflavor )
  {
    int pid_work[broken_sym_nentries];
    for( int i = 0; i < broken_sym_nentries; i++ )
      pid_work[i] = broken_sym_pid_list[i];

    int total_factor = 1;
    for( int icomp = 0; icomp < broken_sym_ncomponents; icomp++ )
    {
      int old_factor = broken_sym_component_old_factors[icomp];
      if( broken_sym_component_old_factors[icomp] > 1 )
      {
        for( int i = broken_sym_component_starts[icomp] - 1; i < broken_sym_component_ends[icomp]; i++ )
        {
          if( pid_work[i] == 0 )
            continue;
          int n_tot = 1;
          for( int j = i + 1; j < broken_sym_component_ends[icomp]; j++ )
          {
            if( pid_work[i] != pid_work[j] )
              continue;
            bool same_block = ( broken_sym_block_lengths[i] == broken_sym_block_lengths[j] );
            for( int k = 0; same_block && k < broken_sym_block_lengths[i]; k++ )
            {
              if( cFlavors[iflavor][broken_sym_block_starts[i] - 1 + k] != cFlavors[iflavor][broken_sym_block_starts[j] - 1 + k] )
                same_block = false;
            }
            if( same_block )
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

  __global__ void
  normalise_output( fptype* allMEs,                    // output: allMEs[nevt], |M|^2 running_sum_over_helicities
                    const unsigned int* iflavorVec,
                    fptype_amp* allNumerators,          // input: multichannel numerators[nevt][ndiagrams], already summed over helicities (atomicAdd)
                    fptype_amp* allDenominators,        // output: multichannel denominators[nevt], derived here as the sum of numerators
                    const unsigned int* allChannelIds,  // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable SDE enhancement (fix #899/#911)
                    bool storeChannelWeights,            // if true, compute final multichannel weights
                    bool mulChannelWeight,               // if true, multiply matrix element by channel weight
                    const fptype globaldenom )
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread)
    allMEs[ievt] = allMEs[ievt] * broken_symmetry_factor( iflavorVec[ievt] ) / globaldenom;
    if( storeChannelWeights ) // fix segfault #892 (not 'channelIds[0] != 0')
    {
      // The numerators have already been accumulated over all good helicities in place (atomicAdd in
      // calculate_jamps), so there is no helicity dimension to sum here. The denominator is just the
      // sum of all numerators for this event: derive it once and store it for the downstream consumers.
      fptype_amp* numerators = allNumerators + ievt * ndiagrams;
      fptype denominator = 0;
      for( int idiag = 0; idiag < ndiagrams; ++idiag )
        denominator += numerators[idiag];
      allDenominators[ievt] = denominator;
      if( mulChannelWeight )
      {
        unsigned int channelId = allChannelIds[ievt];
        // denominator == 0 means every diagram's |amp|^2 vanishes for this event (a
        // subprocess whose matrix element is identically zero); 0/0 would turn a zero
        // matrix element into a nan. Floored rather than tested, so that -ffast-math
        // cannot drop the guard - see backend/cpu/SigmaKin.cc for the reasoning.
#if defined MGONGPU_FPTYPE_DOUBLE
        constexpr fptype fptypeMin = DBL_MIN; // smallest normal double
#elif defined MGONGPU_FPTYPE_FLOAT
        constexpr fptype fptypeMin = FLT_MIN; // smallest normal float
#endif
        allMEs[ievt] *= numerators[channelId - 1] / ( denominator + fptypeMin );
      }
    }
  }

  //--------------------------------------------------------------------------

  __global__ void
  add_and_select_hel( int* allselhel,                   // output: helicity selection[nevt]
                      const fptype* allrndhel,          // input: random numbers[nevt] for helicity selection
                      fptype* ghelAllMEs,               // input/tmp: allMEs for nGoodHel <= ncomb individual/runningsum helicities (index is ighel)
                      fptype* allMEs,                   // output: allMEs[nevt], final sum over helicities
                      const int nevt,                   // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
                      const bool sampleSignedHelicity,  // input: see sigmaKin
                      const int* allihel )              // input: helicity index [0,ncomb) of each event[nevt] (nullptr: choose one)
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread)
    if( allihel != nullptr )
    {
      // No sum and no choice: each event was evaluated for its given helicity only, whose
      // |M|^2 (signed, with an interference) is the one entry of ghelAllMEs (nhel == 1)
      allMEs[ievt] = ghelAllMEs[ievt];
      allselhel[ievt] = allihel[ievt] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
      return;
    }
    // no helicity (0) unless one is chosen below: a vanishing or NaN |M|^2 chooses none, and
    // allselhel is not cleared beforehand (the CPU buffers are zero-initialised)
    allselhel[ievt] = 0;
    if constexpr( nampso > 1 )
    {
      if( sampleSignedHelicity )
      {
        // A helicity can contribute negatively to an interference: choose it as madevent
        // does (see select_helicity_signed in backend/{cpu,simd}/SigmaKin.cc), helicity i
        // with probability |T_i| / sum_j |T_j|, and give the event sign(T_i) * sum_j |T_j|,
        // whose average over the choice is sum_j T_j. ghelAllMEs holds the T_i themselves.
        fptype absTotal = 0;
        for( int ighel = 0; ighel < dcNGoodHel; ighel++ )
          absTotal += fabs( ghelAllMEs[ighel * nevt + ievt] );
        allMEs[ievt] = 0;
        if( absTotal == 0 ) return; // no helicity, as in the positive case
        fptype absSum = 0;
        int ighelLast = -1; // last non-vanishing helicity, in case rndhel is not below 1
        for( int ighel = 0; ighel < dcNGoodHel; ighel++ )
        {
          const fptype t = ghelAllMEs[ighel * nevt + ievt];
          if( t == 0 ) continue;
          ighelLast = ighel;
          absSum += fabs( t );
          if( allrndhel[ievt] < absSum / absTotal ) break; // the last ratio is exactly 1
        }
        allselhel[ievt] = dcGoodHel[ighelLast] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
        allMEs[ievt] = copysign( absTotal, ghelAllMEs[ighelLast * nevt + ievt] );
        return;
      }
    }
    // Compute the sum of MEs over all good helicities (defer this after the helicity loop to avoid breaking streams parallelism)
    for( int ighel = 0; ighel < dcNGoodHel; ighel++ )
    {
      allMEs[ievt] += ghelAllMEs[ighel * nevt + ievt];
      ghelAllMEs[ighel * nevt + ievt] = allMEs[ievt]; // reuse the buffer to store the running sum for helicity selection
    }
    // Event-by-event random choice of helicity #403
    for( int ighel = 0; ighel < dcNGoodHel; ighel++ )
    {
      if( allrndhel[ievt] < ( ghelAllMEs[ighel * nevt + ievt] / allMEs[ievt] ) )
      {
        const int ihelF = dcGoodHel[ighel] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
        allselhel[ievt] = ihelF;
        break;
      }
    }
  }

  //--------------------------------------------------------------------------

  __global__ void
  select_col_and_diag( int* allselcol,                    // output: color selection[nevt]
                       unsigned int* allDiagramIdsOut,    // output: sampled diagram ids
                       const fptype* allrndcol,           // input: random numbers[nevt] for color selection
                       const fptype* allrnddiagram,       // input: random numbers[nevt] for diagram selection
                       const unsigned int* allChannelIds, // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable SDE enhancement (fix #899/#911)
                       const fptype_amp_sv* allJamp2s,    // input: jamp2[ncolor_flow][nevt] for color choice (nullptr if disabled)
                       const fptype_amp* allNumerators,   // input: all numerators
                       const fptype_amp* allDenominators, // input: all denominators
                       const int nevt )                   // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread)
    // SCALAR channelId for the current event (CUDA)
    unsigned int channelId = gpu_channelId( allChannelIds );

    // Event-by-event random choice of channel
    if( allrnddiagram != nullptr )
    {
      fptype numerator_sum = 0., normalization = 0.;
      for( unsigned int ichan = 0; ichan < mgOnGpu::nchannels; ichan++ )
      {
        if( mgOnGpu::channel2iconfig[ichan] == -1 ) continue;
        normalization += allNumerators[ievt * ndiagrams + ichan];
      }
      channelId = mgOnGpu::nchannels;
      for( unsigned int ichan = 0; ichan < mgOnGpu::nchannels; ichan++ )
      {
        if( mgOnGpu::channel2iconfig[ichan] == -1 ) continue;
        numerator_sum += allNumerators[ievt * ndiagrams + ichan];
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
        printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%d which is greater than nchannels=%d\n", channelId, mgOnGpu::nchannels );
        assert( channelId <= mgOnGpu::nchannels ); // SANITY CHECK #919 #910
      }
      // Determine the jamp2 for this event (TEMPORARY? could do this with a dedicated memory accessor instead...)
      fptype_amp_sv jamp2_sv[ncolor_flow] = { 0 };
      assert( allJamp2s != nullptr ); // sanity check

      using J2_ACCESS = DeviceAccessJamp2;
      for( int icolC = 0; icolC < ncolor_flow; icolC++ )
        jamp2_sv[icolC] = J2_ACCESS::kernelAccessIcolConst( allJamp2s, icolC );
      // NB (see #877): in the array channel2iconfig, the input index uses C indexing (channelId -1), the output index uses F indexing (iconfig)
      const int iconfig = mgOnGpu::channel2iconfig[channelId - 1]; // map N_diagrams to N_config <= N_diagrams configs (fix LHE color mismatch #856: see also #826, #852, #853)
      if( iconfig <= 0 )
      {
        printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%d which has no associated SDE iconfig\n", channelId );
        assert( iconfig > 0 ); // SANITY CHECK #917
      }
      else if( iconfig > (int)mgOnGpu::nconfigSDE )
      {
        printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%d (invalid SDE iconfig=%d\n > nconfig=%d)", channelId, iconfig, mgOnGpu::nconfigSDE );
        assert( iconfig <= (int)mgOnGpu::nconfigSDE ); // SANITY CHECK #917
      }
      fptype_amp targetamp[ncolor_flow] = { 0 };
      // NB (see #877): explicitly use 'icolC' rather than 'icol' to indicate that icolC uses C indexing in [0, N_colors-1]
      for( int icolC = 0; icolC < ncolor_flow; icolC++ )
      {
        if( icolC == 0 )
          targetamp[icolC] = 0;
        else
          targetamp[icolC] = targetamp[icolC - 1];
        // NB (see #877): in the array icolamp, the input index uses C indexing (iconfig -1)
        if( mgOnGpu::icolamp[iconfig - 1][icolC] ) targetamp[icolC] += jamp2_sv[icolC];
      }
      allselcol[ievt] = 0; // no color unless one is chosen below (vanishing or NaN jamp2 sum)
      for( int icolC = 0; icolC < ncolor_flow; icolC++ )
      {
        if( allrndcol[ievt] < ( targetamp[icolC] / targetamp[ncolor_flow - 1] ) )
        {
          allselcol[ievt] = icolC + 1; // NB Fortran [1,ncolor], cudacpp [0,ncolor-1]
          break;
        }
      }
    }
    else
    {
      allselcol[ievt] = 0; // no color selected in Fortran range [1,ncolor] if channelId == 0 (see #931)
    }
  }

  //--------------------------------------------------------------------------
  // Evaluate |M|^2, part independent of incoming flavour

  void /* clang-format off */
  sigmaKin( const fptype_momenta* allmomenta,   // input: momenta[nevt*npar*4]
            const fptype* allcouplings,         // input: couplings[nevt*ndcoup*2]
            const unsigned int* iflavorVec,     // input: index of the flavor combination
            const fptype* allrndhel,            // input: random numbers[nevt] for helicity selection
            const fptype* allrndcol,            // input: random numbers[nevt] for color selection
            const unsigned int* allChannelIds,  // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable single-diagram enhancement (fix #899/#911)
            const fptype* allrnddiagram,        // input: random numbers[nevt] for channel sampling
            fptype* allMEs,                     // output: allMEs[nevt], |M|^2 final_avg_over_helicities
            int* allselhel,                     // output: helicity selection[nevt]
            int* allselcol,                     // output: helicity selection[nevt]
            fptype_amp* colAllJamp2s,           // tmp: allJamp2s super-buffer for ncolor individual colors, running sum over colors and helicities
            fptype_amp* ghelAllNumerators,      // tmp: allNumerators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            fptype_amp* ghelAllDenominators,    // tmp: allDenominators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            unsigned int* allDiagramIdsOut,     // output: multichannel channelIds[nevt] (1 to #diagrams)
            bool mulChannelWeight,              // if true, multiply channel weight to ME output
            fptype* ghelAllMEs,                 // tmp: allMEs super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            fptype_amp* ghelAllJamps,           // tmp: jamp[2*ncolor*nGoodHel*nevt] super-buffer for nGoodHel <= ncomb individual helicities
            fptype_colour* ghelAllBlasTmp,      // tmp: allBlasTmp super-buffer for nGoodHel <= ncomb individual helicities
            gpuBlasHandle_t* pBlasHandle,       // input: cuBLAS/hipBLAS handle
            gpuStream_t* ghelStreams,           // input: cuda streams (index is ighel: only the first nGoodHel <= ncomb are non-null)
            const bool async,
            const int gpublocks,                // input: cuda gpublocks
            const int gputhreads,               // input: cuda gputhreads
            const bool sampleSignedHelicity,    // input: choose the helicity on |T_i| and return sign(T_i)*sum|T| (split orders only)
            const int* allihel )                // input: helicity index [0,ncomb) of each event[nevt]; nullptr to sum over the good helicities
  /* clang-format on */
  {
    mgDebugInitialise();

    // SANITY CHECKS for cudacpp code generation (see issues #272 and #343 and PRs #619, #626, #360, #396 and #754)
    {
      // nprocesses == 2 may happen for "mirror processes" such as P0_uux_ttx within pp_tt012j (see PR #754)
      static_assert( nproc == 1 || nproc == 2, "Assume nprocesses == 1 or 2" );
      static_assert( proc_id == 1, "Assume process_id == 1" );
    }

    // === PART 0 - INITIALISATION (before calculate_jamps) ===
    // In async mode (umami) all the work goes to the caller's stream ghelStreams[0], in stream
    // order with what the caller does before and after, also when that stream does not
    // synchronise with the default stream (a non-blocking stream, e.g. a torch side stream).
    // Otherwise the default stream, which the helicity streams synchronise with.
    gpuStream_t stream = async ? ghelStreams[0] : 0;
    // Reset the "matrix elements" - running sums of |M|^2 over helicities for the given event
    // (the sizes are computed in size_t, from sizeof: the int product could overflow)
    const int nevt = gpublocks * gputhreads;
    // With allihel, each event has a single helicity of its own instead of the sum over the good
    // helicities: one evaluation (nhel == 1) in which the threads carry different helicities in
    // lock-step, and the color sum, the color and channel choice all run on that one helicity.
    const int nhel = ( allihel != nullptr ? 1 : cNGoodHel );
    gpuMemsetAsync( allMEs, 0, sizeof( fptype ) * nevt, stream );
    gpuMemsetAsync( ghelAllJamps, 0, sizeof( fptype_amp ) * nhel * njampso * mgOnGpu::nx2 * nevt, stream );
    gpuMemsetAsync( colAllJamp2s, 0, sizeof( fptype_amp ) * ncolor_flow * nevt, stream );
    // The numerators buffer has NO helicity dimension: all good helicities accumulate in place via
    // atomicAdd, so it is zeroed once as [nevt][ndiagrams]. The denominators are derived from the
    // numerators in normalise_output, so the buffer is just [nevt].
    gpuMemsetAsync( ghelAllNumerators, 0, sizeof( fptype_amp ) * ndiagrams * nevt, stream );
    gpuMemsetAsync( ghelAllDenominators, 0, sizeof( fptype_amp ) * nevt, stream );
    gpuMemsetAsync( ghelAllMEs, 0, sizeof( fptype ) * nhel * nevt, stream );

    // === PART 1 - HELICITY LOOP: CALCULATE WAVEFUNCTIONS (one event per GPU thread) ===

    // Use CUDA/HIP streams to process different helicities in parallel (one good helicity per stream)
    // (1) First, within each helicity stream, compute the QCD partial amplitudes jamp's for each helicity
    // In multichannel mode, also compute the running sums over helicities of numerators, denominators and squared jamp2s
    // No good helicity (a subprocess whose |M|^2 vanishes identically, e.g. an FCNC channel with
    // every Wilson coefficient at zero): nothing to compute, the |M|^2 stays 0 as on the CPU (a
    // launch with gridDim.y = 0 is an invalid configuration)
    bool storeChannelWeights = allChannelIds != nullptr || allrnddiagram != nullptr;
    if( nhel > 0 && async )
    {
      gpuLaunchKernel2D( calculate_jamps, gpublocks, nhel, gputhreads, ghelStreams[0], 0, allihel, allmomenta, allcouplings, iflavorVec, ghelAllJamps, storeChannelWeights, ghelAllNumerators, ghelAllDenominators, colAllJamp2s, nevt, nhel, true );
      color_sum_gpu( ghelAllMEs, ghelAllJamps, ghelAllBlasTmp, pBlasHandle, ghelStreams, nhel, gpublocks, gputhreads, true );
    }
    else if( nhel > 0 )
    {
      for( int ighel = 0; ighel < nhel; ighel++ )
      {
        const int ihel = ( allihel != nullptr ? 0 : cGoodHel[ighel] ); // unused with allihel
        fptype_amp* hAllJamps = ghelAllJamps + ighel * nevt; // HACK: bypass DeviceAccessJamp (consistent with layout defined there)
        // NB: the numerators buffer has no helicity dimension: every helicity stream accumulates in place
        // into the same [nevt][ndiagrams] slot via atomicAdd. The denominators are derived later.
        gpuLaunchKernelStream( calculate_jamps, gpublocks, gputhreads, ghelStreams[ighel], ihel, allihel, allmomenta, allcouplings, iflavorVec, hAllJamps, storeChannelWeights, ghelAllNumerators, ghelAllDenominators, colAllJamp2s, nevt, nhel, false );
      }
      // (2) Then compute the ME for that helicity from the color sum of QCD partial amplitudes jamps
      color_sum_gpu( ghelAllMEs, ghelAllJamps, ghelAllBlasTmp, pBlasHandle, ghelStreams, nhel, gpublocks, gputhreads, false );
      checkGpu( gpuDeviceSynchronize() ); // do not start helicity/color selection until the loop over helicities has completed
      // (3) Wait for all helicity streams to complete, then finally compute the ME sum over all helicities and choose one helicity and one color
    }
    // Event-by-event random choice of helicity #403 and ME sum over helicities (defer this after the helicity loop to avoid breaking streams parallelism)
    gpuLaunchKernelStream( add_and_select_hel, gpublocks, gputhreads, stream, allselhel, allrndhel, ghelAllMEs, allMEs, gpublocks * gputhreads, sampleSignedHelicity, allihel );

    gpuLaunchKernelStream( normalise_output, gpublocks, gputhreads, stream, allMEs, iflavorVec, ghelAllNumerators, ghelAllDenominators, allChannelIds, storeChannelWeights, mulChannelWeight, helcolDenominators[0] );

    // Event-by-event random choice of color and diagram #402
    gpuLaunchKernelStream( select_col_and_diag, gpublocks, gputhreads, stream, allselcol, allDiagramIdsOut, allrndcol, allrnddiagram, allChannelIds, colAllJamp2s, ghelAllNumerators, ghelAllDenominators, gpublocks * gputhreads );

    mgDebugFinalise();
  }

  //--------------------------------------------------------------------------

} // end namespace
