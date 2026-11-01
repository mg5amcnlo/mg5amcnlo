// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Backend-owned driver: sigmaKin, calculate_jamps, good-helicity filtering.
// The diagram/vertex-call sequence (helas_calls) is process-specific and
// lives in the P1-generated EvaluateDiagrams.inc, #include'd below.
//
// The parameters of the process (cIPD/cIPC/cIPF/bsmIndepParam, the good
// helicities) are an argument of every function here (ProcessParameters.h):
// they belong to the CPPProcess, and so to the UMAMI instance, evaluated.

#include "SigmaKin.h"

#include "ProcessParameters.h"

#include "CPPProcess.h" // ProcessData.h, Parameters.h, HelAmps_<model>.h transitively
#include "ProcessTables.h"

#include "MemoryAccessAmplitudes.h"
#include "MemoryAccessChannelIds.h"
#include "MemoryAccessCouplings.h"
#include "MemoryAccessCouplingsFixed.h"
#include "MemoryAccessDenominators.h"
#include "MemoryAccessGs.h"
#include "MemoryAccessIflavorVec.h"
#include "MemoryAccessMatrixElements.h"
#include "MemoryAccessMomenta.h"
#include "MemoryAccessNumerators.h"
#include "MemoryAccessWavefunctions.h"
#include "ColorData.h"       // for shouldUseBlas/mgOnGpu::nchannels/channel2iconfig/icolamp/nconfigSDE
#include "color_sum.h"       // for color_sum_cpu/color_sum_cpu_blas

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace madmatrix
{
  using namespace ProcessData;
  using namespace ProcessTables;
  using Parameters_dependentCouplings::ndcoup;   // #couplings that vary event by event (depend on running alphas QCD)
  using Parameters_independentCouplings::nicoup; // #couplings that are fixed for all events (do not depend on running alphas QCD)

  // The number of SIMD vectors of events processed by calculate_jamps
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
  constexpr int nParity = 2;
#else
  constexpr int nParity = 1;
#endif

  // ncolor_flow (unlike ncolor) is not in ProcessData.h: the color sum can run
  // on a smaller (DDM) basis than the color flow probabilities do, so this stays
  // a CPPProcess-generated constant (see process_class.inc/set_color_flow_lines_cpp).
  constexpr int ncolor_flow = CPPProcess::ncolor_flow;

  // The helicity and flavor tables of the process (ProcessData.h), which are the
  // same for every CPPProcess
  static constexpr const short ( &cHel )[ncomb][npar] = tHel;
  static constexpr const short ( &cFlavors )[nmaxflavor][npar] = tFlavors;

  const short* getHelicities()
  {
    return &cHel[0][0];
  }

  //--------------------------------------------------------------------------

  // SCALAR channelId for the whole SIMD neppV2 event page (C++), i.e. one or two neppV event page(s)
  // The cudacpp implementation ASSUMES (and checks! #898) that all channelIds are the same in a neppV2 SIMD event page
  // **NB! in "mixed" precision, using SIMD, calculate_wavefunctions computes MEs for TWO neppV pages with a single channelId! #924
  __device__ INLINE unsigned int
  getChannelId( const unsigned int* allChannelIds, const int ievt00, bool sanityCheckMixedPrecision = true )
  {
    unsigned int channelId = 0; // disable multichannel single-diagram enhancement unless allChannelIds != nullptr
    using CID_ACCESS = HostAccessChannelIds; // non-trivial access: buffer includes all events
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
#else
        (void)sanityCheckMixedPrecision; // no second SIMD page to cross-check outside mixed precision
#endif
      }
    }
    return channelId;
  }

  //--------------------------------------------------------------------------

  __global__ void
  computeDependentCouplings( const ProcessParameters& params, const fptype* allgs, fptype* allcouplings, const int nevt )
  {
    using G_ACCESS = HostAccessGs;
    using C_ACCESS = HostAccessCouplings;
    for( int ipagV = 0; ipagV < nevt / neppV; ++ipagV )
    {
      const int ievt0 = ipagV * neppV;
      const fptype* gs = MemoryAccessGs::ieventAccessRecordConst( allgs, ievt0 );
      fptype* couplings = MemoryAccessCouplings::ieventAccessRecord( allcouplings, ievt0 );
      G2COUP<G_ACCESS, C_ACCESS>( gs, couplings, params.bsmIndepParam );
    }
  }

  //--------------------------------------------------------------------------

  // The mass an external leg marked off-shell ('*': MadSpin's decaying particles)
  // is evaluated with: the invariant mass of its momentum, as in the Fortran
  // standalone output (SQRT(P(0,i)**2-...)), so that the wavefunction is the one
  // of the momentum the leg actually has. The pole mass comes first only for the
  // code generator, which maps it to its parameter.
  template<class M_ACCESS>
  inline fptype_amp
  offshell_mass( const fptype_amp poleMass, const fptype_momenta* momenta, const int ipar )
  {
    (void)poleMass;
    const double e = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const double px = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const double py = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const double pz = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    return static_cast<fptype_amp>( std::sqrt( std::max( e * e - px * px - py * py - pz * pz, 0. ) ) );
  }

  //--------------------------------------------------------------------------

  // The helicities of the npar external legs for the events of one page (neppV events,
  // one SIMD lane each, or one event without SIMD), as the external wavefunctions take
  // them: the helicities hel[npar] for every event, or the helicity allihel[ievt] of each event.
  static inline void
  fillHelicities( fptype_amp_sv* hel_sv,  // output: hel_sv[npar] helicities of the external legs
                  const short* hel,       // input: helicity of each leg, hel[npar], of all events (if allihel == nullptr)
                  const int* allihel,     // input: helicity index [0,ncomb) of each event[nevt] (nullptr: hel for all events)
                  const int ievt0 )       // input: first event number in current C++ event page
  {
#ifdef MGONGPU_CPPSIMD
    for( int ieppV = 0; ieppV < neppV; ieppV++ )
    {
      const short* heli = ( allihel != nullptr ? cHel[allihel[ievt0 + ieppV]] : hel );
      for( int ipar = 0; ipar < npar; ipar++ ) hel_sv[ipar][ieppV] = heli[ipar];
    }
#else
    const short* heli = ( allihel != nullptr ? cHel[allihel[ievt0]] : hel );
    for( int ipar = 0; ipar < npar; ipar++ ) hel_sv[ipar] = heli[ipar];
#endif
  }

  //--------------------------------------------------------------------------

  // Accumulate a multichannel numerator contribution in place. In C++ each
  // event page is processed serially within the helicity loop, so a plain
  // sum suffices (CUDA needs atomicAdd instead: see backend/gpu/SigmaKin.cc).
#define NUM_ATOMIC_ADD( DST, VAL ) ( DST ) += ( VAL )

  // Evaluate QCD partial amplitudes jamps for this given helicity from Feynman diagrams.
  // Also compute running sums over helicities adding jamp2, numerator, denominator
  // (NB: this function no longer handles matrix elements as the color sum has now been
  // moved to a separate function/kernel). This function processes a single event "page"
  // or SIMD vector (or for two in "mixed" precision mode, nParity=2). Accepts a SCALAR
  // channelId because it is GUARANTEED that all events in a SIMD vector have the same
  // channelId #898. The helicity, on the other hand, may differ from one event (SIMD lane)
  // to the next: the external wavefunctions are branch-free in the helicity (see helas.h),
  // so that the events of a page with different helicities are computed in lock-step.
  void
  calculate_jamps( const ProcessParameters& params,    // input: parameters of the process
                   const short* hel,                   // input: helicity of each leg, hel[npar], of all events (if allihel == nullptr): a row of cHel, or not (see Density.cc)
                   const int* allihel,                 // input: helicity index [0,ncomb) of each event[nevt] (nullptr: hel for all events)
                   const fptype_momenta* allmomenta,   // input: momenta[nevt*npar*4]
                   const fptype* allcouplings,         // input: couplings[nevt*ndcoup*2]
                   const unsigned int* iflavorVec,     // input: indices of the flavor combinations
                   cxtype_amp_sv* allJamp_sv,          // output: jamp_sv[njampso] (float/double) or jamp_sv[2*njampso] (mixed) for this helicity
                   bool storeChannelWeights,
                   fptype_amp* allNumerators,          // input/output: multichannel numerators[nevt], add helicity ihel
                   fptype_amp* allDenominators,        // input/output: multichannel denominators[nevt], add helicity ihel
                   fptype_amp_sv* jamp2_sv,            // output: jamp2[nParity][ncolor_flow][neppV] for color choice (nullptr if disabled)
                   const int ievt00 )                  // input: first event number in current C++ event page
  {
    using M_ACCESS = HostAccessMomenta;         // non-trivial access: buffer includes all events
    using W_ACCESS = HostAccessWavefunctions;   // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using A_ACCESS = HostAccessAmplitudes;      // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using CD_ACCESS = HostAccessCouplings;      // non-trivial access (dependent couplings): buffer includes all events
    using CI_ACCESS = HostAccessCouplingsFixed; // TRIVIAL access (independent couplings): buffer for one event
    using CDPF_ACCESS = CD_ACCESS;              // the running flavor couplings gathered into dpf_value (one event page, as CD_ACCESS)
    using F_ACCESS = HostAccessIflavorVec;      // non-trivial access: buffer includes all events
    using NUM_ACCESS = HostAccessNumerators;    // non-trivial access: buffer includes all events
    mgDebug( 0, __FUNCTION__ );

    // The parameters of the process, under the names EvaluateDiagrams.inc uses. The
    // independent parameters are copied: a local array cannot alias the buffers written
    // below, so that they are not read again after every store.
    fptype cIPD[ProcessParameters::nIPD];
    for( int i = 0; i < ProcessParameters::nIPD; i++ ) cIPD[i] = params.IPD[i];
    const fptype* cIPC = params.IPC;
    const int* cIPF_partner1 = params.IPF_partner1;
    const int* cIPF_partner2 = params.IPF_partner2;
    const fptype* cIPF_value = params.IPF_value;
    [[maybe_unused]] const fptype cBWCUTOFF = params.bwcutoff; // the $-excluded propagators (ALOHA P1D tag)
#ifndef MGONGPU_HARDCODE_PARAM
    Parameters::current = params.model; // for the HelAmps routines that read a model parameter
#endif

    // Local TEMPORARY variables for a subset of Feynman diagrams in the given C++ event
    // page (ipagV) [NB these variables are reused several times (and re-initialised each
    // time) within the same event or event page]. Create memory for both momenta and
    // wavefunctions separately, and later wrap them in ALOHAOBJ.
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

    // jamp: sum (for one event or event page) of the invariant amplitudes for all Feynman
    // diagrams in a given color combination (NB: vector cxtype_v IS initialized to 0, but
    // scalar cxtype is NOT, if "= {}" is missing!)
    // (njampso = ncolor * nampso: one vector per amplitude split order, just ncolor without them)
    cxtype_amp_sv jamp_sv[njampso] = {};
    // jampTmp: partial sums of amplitudes that several color flows share, so that they are
    // computed only once (see MadMatrixUFOHelasCallWriter.build_jamp_plan); no "= {}", each
    // one is assigned before it is ever read.
    cxtype_amp_sv jampTmp_sv[ProcessTables::nb_tmp_jamp > 0 ? ProcessTables::nb_tmp_jamp : 1];

    // === Calculate wavefunctions and amplitudes for all diagrams in all processes
    // === (for one event page in C++, or for two in mixed mode)

    // START LOOP ON IPARITY
    for( int iParity = 0; iParity < nParity; ++iParity )
    {
      const int ievt0 = ievt00 + iParity * neppV;

      constexpr size_t nxcoup = ndcoup + nIPC; // both dependent and independent couplings
      // nxcoup can be 0 (no alpha_s-dependent coupling, and only flavor couplings, e.g.
      // u u~ > u u~ QED^2==4 with flavor grouping): a zero-sized array is only a compiler
      // extension (and refused in GPU device code), so size them at least 1, as cIPD/cIPC
      const fptype* allCOUPs[nxcoup > 0 ? nxcoup : 1];
      for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ )
        allCOUPs[idcoup] = CD_ACCESS::idcoupAccessBufferConst( allcouplings, idcoup ); // dependent couplings, vary event-by-event
      for( size_t iicoup = 0; iicoup < nIPC; iicoup++ )
        allCOUPs[ndcoup + iicoup] = CI_ACCESS::iicoupAccessBufferConst( cIPC, iicoup ); // independent couplings, fixed for all events
      // C++ kernels take input/output buffers with momenta/MEs for one specific event (the first in the current event page)
      const fptype_momenta* momenta = M_ACCESS::ieventAccessRecordConst( allmomenta, ievt0 );
      const fptype* COUPs[nxcoup > 0 ? nxcoup : 1];
      for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ )
        COUPs[idcoup] = CD_ACCESS::ieventAccessRecordConst( allCOUPs[idcoup], ievt0 ); // dependent couplings, vary event-by-event
      for( size_t iicoup = 0; iicoup < nIPC; iicoup++ )
        COUPs[ndcoup + iicoup] = allCOUPs[ndcoup + iicoup]; // independent couplings, fixed for all events
      fptype_amp* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 * ndiagrams );
      // The helicities of the external legs for the events of this page, one per SIMD lane:
      // the same helicities hel for all of them, or the helicity allihel of each event
      fptype_amp_sv hel_sv[npar];
      fillHelicities( hel_sv, hel, allihel, ievt0 );
      // Create an array of views over the Flavor Couplings
      FLV_COUPLING_ARRAY<nIPF, nMF> flvCOUPs{ cIPF_partner1, cIPF_partner2, cIPF_value };

      // Dependent (event-by-event, running-alphas) flavor couplings (Step 3): the per-flavor
      // values are NOT baked in (they run per event). Gather the current values of the
      // underlying dependent couplings for this event page into an AOSOA buffer dpf_value
      // (one nx2*neppC SIMD record per (coupling,flavor) slot, matching CD_ACCESS), then build
      // an ordinary value-based view over it. The flavor index is constant across a SIMD lane
      // (guaranteed by the phase-space integrator), so each lane gets its own running value
      // while sharing the same flavor selection. This is the direct analogue of Fortran's
      // FLV_xx%VAL(k)%P => GC_yyy(J). The vertex routines are instantiated with CDPF_ACCESS
      // (= CD_ACCESS here) so get_coupling_def reads dpf_value with the right per-flavor stride.
      constexpr int ndpfbuf = ( nDPF > 0 ? nDPF * nMF * CDPF_ACCESS::flv_stride : 1 );
      // cppAlign is only defined for SIMD
      alignas( mgOnGpu::cppAlign ) fptype dpf_value[ndpfbuf]{};
      for( int idpf = 0; idpf < nDPF; idpf++ )
        for( int imf = 0; imf < nMF; imf++ )
        {
          const int idc = cDPF_idcoup[idpf * nMF + imf];
          if( idc >= 0 )
            CDPF_ACCESS::kernelAccess( dpf_value + ( idpf * nMF + imf ) * CDPF_ACCESS::flv_stride ) =
              CD_ACCESS::kernelAccessConst( COUPs[idc] );
        }
      FLV_COUPLING_ARRAY<nDPF, nMF, CDPF_ACCESS::flv_stride> flvCOUPs_dep{ cDPF_partner1, cDPF_partner2, dpf_value };

      // Reset color flows (reset jamp_sv) at the beginning of a new event or event page
      for( int i = 0; i < njampso; i++ ) { jamp_sv[i] = cxzero_sv<cxtype_amp_sv>(); }

      // Numerators for the current event page (C++); denominators are no longer
      // accumulated here: they are derived as the sum of numerators later.
      fptype_amp_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
      // Scalar iflavor for the current event page (constant across the SIMD vector)
      const unsigned int* iflavor_rec = F_ACCESS::ieventAccessRecordConst( iflavorVec, ievt0 );
      const uint_sv iflavor_sv = F_ACCESS::kernelAccessConst( iflavor_rec );
      const unsigned int iflavor = reinterpret_cast<const unsigned int*>( &iflavor_sv )[0];
#include "EvaluateDiagrams.inc"
#include "ColorFlows.inc" // defines jampflow_sv[ncolor_flow], which is not jamp_sv on the DDM basis

      // *** COLOR CHOICE BELOW ***
      // Store the leading color flows for choice of color
      if( jamp2_sv ) // disable color choice if nullptr
      {
        for( int icol = 0; icol < ncolor_flow; icol++ )
          jamp2_sv[ncolor_flow * iParity + icol] += cxabs2( jampflow_sv[icol] ); // may underflow #831
      }

      // *** PREPARE OUTPUT JAMPS ***
      // In C++, copy the local jamp to the output array passed as function argument
      for( int icol = 0; icol < njampso; icol++ )
        allJamp_sv[iParity * njampso + icol] = jamp_sv[icol];
    }
    // END LOOP ON IPARITY

    mgDebug( 1, __FUNCTION__ );
  }

#undef NUM_ATOMIC_ADD

  //--------------------------------------------------------------------------

  void
  sigmaKin_getGoodHel( const ProcessParameters& params,   // input: parameters of the process
                       const fptype_momenta* allmomenta, // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,       // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec,   // input: index of the flavor combination
                       fptype* allMEs,                   // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype_amp* allNumerators,        // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype_amp* allDenominators,      // output: multichannel denominators[nevt], running_sum_over_helicities
                       bool* isGoodHel,                  // output: isGoodHel[ncomb] - host array
                       const int nevt )                  // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  {
    // Allocate arrays at build time to contain at least 16 events (or at least neppV events if neppV>16, e.g. in future VPUs)
    constexpr int maxtry0 = std::max( 16, neppV ); // 16, but at least neppV (otherwise the npagV loop does not even start)
    // Loop over only nevt events if nevt is < 16 (note that nevt is always >= neppV)
    assert( nevt >= neppV );
    const int maxtry = std::min( maxtry0, nevt ); // 16, but at most nevt (avoid invalid memory access if nevt<maxtry0)
    // HELICITY LOOP: CALCULATE WAVEFUNCTIONS
    const int npagV = maxtry / neppV;
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT /* clang-format off */
    // Mixed fptypes #537: float for color algebra and double elsewhere
    // Delay color algebra and ME updates (only on even pages)
    assert( npagV % 2 == 0 ); // SANITY CHECK for mixed fptypes: two neppV-pages are merged to one 2*neppV-page
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
        constexpr fptype_amp_sv* jamp2_sv = nullptr; // no need for color selection during helicity filtering
#if defined MGONGPU_CPPSIMD and !( defined MGONGPU_FPTYPE_AMP_FLOAT ) and defined MGONGPU_FPTYPE2_FLOAT
        cxtype_amp_sv jamp_sv[2 * njampso] = {}; // all zeros
#else
        cxtype_amp_sv jamp_sv[njampso] = {}; // all zeros
#endif
        calculate_jamps( params, cHel[ihel], nullptr, allmomenta, allcouplings, hgFlavorVec, jamp_sv, false, allNumerators, allDenominators, jamp2_sv, ievt00 );
        color_sum_cpu( allMEs, jamp_sv, ievt00 );
        for( int ieppV = 0; ieppV < neppV; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          if( allMEs[ievt] != 0 ) // NEW IMPLEMENTATION OF GETGOODHEL (#630): COMPARE EACH HELICITY CONTRIBUTION TO 0
          {
            isGoodHel[ihel] = true;
          }
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          const int ievt2 = ievt00 + ieppV + neppV;
          if( allMEs[ievt2] != 0 ) // NEW IMPLEMENTATION OF GETGOODHEL (#630): COMPARE EACH HELICITY CONTRIBUTION TO 0
          {
            isGoodHel[ihel] = true;
          }
#endif
        }
      }
    }
    } // end loop over flavor combinations (per-flavor good-helicity union)
  }

  //--------------------------------------------------------------------------

  int                                          // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( ProcessParameters& params, // output: the good helicities of the process
                       const bool* isGoodHel )   // input: isGoodHel[ncomb] - host array
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
    params.nGoodHel = nGoodHel;
    for( int ihel = 0; ihel < ncomb; ihel++ ) params.goodHel[ihel] = goodHel[ihel];
    return nGoodHel;
  }

  //--------------------------------------------------------------------------

  // Decay-aware identical-particle (broken-)symmetry factor, shared with the
  // Fortran / standalone_cpp exporters (_get_broken_symmetry_data). Two
  // entries contribute to the over-counting factor only when they have the
  // same top-level PID AND the same full decay/flavour block, so e.g. two Z
  // bosons decaying to different families are correctly distinguished.
  int
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
  // Helicity choice when a helicity can contribute negatively: a squared split-order
  // selection that keeps an interference (nampso > 1). As madevent does
  // (matrix_madevent_group_v4.inc), helicity i is chosen with probability
  // |T_i| / sum_j |T_j| and the event gets sign(T_i) * sum_j |T_j|, whose average over
  // the choice is sum_j T_j: the chosen helicity then reproduces the signed per-helicity
  // |M|^2. Choosing on the signed running sum, as the positive case does, would not:
  // with mixed signs it is not a distribution, and a negative total turns it upside down.
  // ighelSum[ighel] is the running sum of T_j over the good helicities j <= ighel, as the
  // helicity loop leaves it. Returns the |M|^2 of the event (before the helicity/color
  // average and the channel weight) and sets selhel; a vanishing |M|^2 returns 0 and
  // leaves selhel alone, as the positive case does.
  [[maybe_unused]] static inline fptype
  select_helicity_signed( const ProcessParameters& params, // input: parameters of the process (good helicities)
                          const fptype* ighelSum, // input: running sums over the good helicities
                          const fptype rndhel,    // input: random number for the helicity choice
                          int& selhel )           // output: chosen helicity (Fortran range [1,ncomb])
  {
    const int cNGoodHel = params.nGoodHel;
    const int* cGoodHel = params.goodHel;
    fptype absTotal = 0;
    for( int ighel = 0; ighel < cNGoodHel; ighel++ )
      absTotal += std::abs( ighelSum[ighel] - ( ighel > 0 ? ighelSum[ighel - 1] : 0 ) );
    if( absTotal == 0 ) return 0;
    fptype absSum = 0;
    int ighelLast = -1; // last non-vanishing helicity, in case rndhel is not below 1
    for( int ighel = 0; ighel < cNGoodHel; ighel++ )
    {
      const fptype t = ighelSum[ighel] - ( ighel > 0 ? ighelSum[ighel - 1] : 0 );
      if( t == 0 ) continue;
      ighelLast = ighel;
      absSum += std::abs( t );
      if( rndhel < absSum / absTotal ) // the last ratio is exactly 1 (same terms, same order)
        break;
    }
    const fptype tLast = ighelSum[ighelLast] - ( ighelLast > 0 ? ighelSum[ighelLast - 1] : 0 );
    selhel = cGoodHel[ighelLast] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
    return std::copysign( absTotal, tLast );
  }

  //--------------------------------------------------------------------------
  // Evaluate |M|^2, part independent of incoming flavour

  void
  sigmaKin( const ProcessParameters& params,   // input: parameters of the process
            const fptype_momenta* allmomenta,  // input: momenta[nevt*npar*4]
            const fptype* allcouplings,        // input: couplings[nevt*ndcoup*2]
            const unsigned int* iflavorVec,    // input: index of the flavor combination
            const fptype* allrndhel,           // input: random numbers[nevt] for helicity selection
            const fptype* allrndcol,           // input: random numbers[nevt] for color selection
            const unsigned int* allChannelIds, // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable single-diagram enhancement (fix #899)
            const fptype* allrnddiagram,       // input: random numbers[nevt] for channel sampling
            fptype* allMEs,                    // output: allMEs[nevt], |M|^2 final_avg_over_helicities
            int* allselhel,                    // output: helicity selection[nevt]
            int* allselcol,                    // output: helicity selection[nevt]
            fptype_amp* allNumerators,         // tmp: multichannel numerators[nevt], running_sum_over_helicities
            fptype_amp* allDenominators,       // tmp: multichannel denominators[nevt], running_sum_over_helicities
            unsigned int* allDiagramIdsOut,    // output: multichannel channelIds[nevt] (1 to #diagrams)
            bool mulChannelWeight,             // if true, multiply channel weight to ME output
            const int nevt,                    // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
            const bool sampleSignedHelicity,   // input: choose the helicity on |T_i| and return sign(T_i)*sum|T| (split orders only)
            const int* allihel )               // input: helicity index [0,ncomb) of each event[nevt]; nullptr to sum over the good helicities
  {
    mgDebugInitialise();

    // SANITY CHECKS for cudacpp code generation (see issues #272 and #343 and PRs #619, #626, #360, #396 and #754)
    {
      // nprocesses == 2 may happen for "mirror processes" such as P0_uux_ttx within pp_tt012j (see PR #754)
      static_assert( nproc == 1 || nproc == 2, "Assume nprocesses == 1 or 2" );
      static_assert( proc_id == 1, "Assume process_id == 1" );
    }

    using E_ACCESS = HostAccessMatrixElements; // non-trivial access: buffer includes all events
    using NUM_ACCESS = HostAccessNumerators;   // non-trivial access: buffer includes all events
    using DEN_ACCESS = HostAccessDenominators; // non-trivial access: buffer includes all events

    // The good helicities of the process
    const int cNGoodHel = params.nGoodHel;
    const int* cGoodHel = params.goodHel;

    // === PART 0 - INITIALISATION (before calculate_jamps) ===
    // Reset the "matrix elements" - running sums of |M|^2 over helicities for the given event
    const int npagV = nevt / neppV;
    for( int ipagV = 0; ipagV < npagV; ++ipagV )
    {
      const int ievt0 = ipagV * neppV;
      fptype* MEs = E_ACCESS::ieventAccessRecord( allMEs, ievt0 );
      fptype_sv& MEs_sv = E_ACCESS::kernelAccess( MEs );
      MEs_sv = fptype_sv{ 0 };
      fptype_amp* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 * ndiagrams );
      fptype_amp* denominators = DEN_ACCESS::ieventAccessRecord( allDenominators, ievt0 );
      fptype_amp_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
      fptype_amp_sv& denominators_sv = DEN_ACCESS::kernelAccess( denominators );
      for( int i = 0; i < ndiagrams; ++i )
      {
        numerators_sv[i] = fptype_amp_sv{ 0 };
      }
      denominators_sv = fptype_amp_sv{ 0 };
    }

    // === PART 1 - HELICITY LOOP: CALCULATE WAVEFUNCTIONS ===
    // (using precomputed good helicities)
    // With allihel, each event has a single helicity of its own instead: the "loop" is one
    // evaluation in which the events of a SIMD page carry different helicities in lock-step,
    // and everything below (color sum, color and channel choice) runs on that one helicity.
    const int nhelEval = ( allihel != nullptr ? 1 : cNGoodHel );
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    // Mixed fptypes #537: float for color algebra and double elsewhere
    // Delay color algebra and ME updates (only on even pages)
    assert( npagV % 2 == 0 );     // SANITY CHECK for mixed fptypes: two neppV-pages are merged to one 2*neppV-page
    const int npagV2 = npagV / 2; // loop on two SIMD pages (neppV events) at a time
#else
    const int npagV2 = npagV; // loop on one SIMD page (neppV events) at a time
#endif
#ifdef _OPENMP
    // OMP multithreading #575 (NB: tested only with gcc11 so far)
#define _OMPLIST0 allcouplings, allMEs, allmomenta, allrndcol, allrndhel, allselcol, allselhel, allihel, params, cNGoodHel, cGoodHel, nhelEval, npagV2, sampleSignedHelicity
#define _OMPLIST1 , allDenominators, allNumerators, allChannelIds, mgOnGpu::icolamp, mgOnGpu::channel2iconfig
#pragma omp parallel for default( none ) shared( _OMPLIST0 _OMPLIST1 )
#undef _OMPLIST0
#undef _OMPLIST1
#endif // _OPENMP
    for( int ipagV2 = 0; ipagV2 < npagV2; ++ipagV2 )
    {
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      const int ievt00 = ipagV2 * neppV * 2; // loop on two SIMD pages (neppV events) at a time
#else
      const int ievt00 = ipagV2 * neppV; // loop on one SIMD page (neppV events) at a time
#endif
      // Running sum of partial amplitudes squared for event by event color selection (#402)
      fptype_amp_sv jamp2_sv[nParity * ncolor_flow] = {};
      fptype_sv MEs_ighel[ncomb] = {}; // sum of MEs for all good helicities up to ighel (first - and/or only - neppV page)
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      fptype_sv MEs_ighel2[ncomb] = {}; // sum of MEs for all good helicities up to ighel (second neppV page)
#endif

      // The color matrix does not depend on the helicity, so for a large-enough color matrix
      // (ColorMatrixData::shouldUseBlas) it pays to keep the jamps of every good helicity and
      // hand them all to BLAS in one call after the loop instead of the per-helicity color sum.
#ifdef MGONGPU_CPP_HAS_BLAS
      // The BLAS buffers below hold one ncolor jamp vector per helicity (cpp_blas_wanted is false once split)
      static_assert( !ColorMatrixData::shouldUseBlas || nampso == 1, "the BLAS color sum does not pair split amplitude orders" );
      if( ColorMatrixData::shouldUseBlas )
      {
        static thread_local std::vector<cxtype_sv> ghelJamp_sv( (size_t)ncomb * nParity * ncolor );
        for( int ighel = 0; ighel < nhelEval; ighel++ )
        {
          const int ihel = ( allihel != nullptr ? -1 : cGoodHel[ighel] ); // unused with allihel
          cxtype_sv* jamp_sv = ghelJamp_sv.data() + (size_t)ighel * nParity * ncolor;
          for( int i = 0; i < nParity * ncolor; i++ ) jamp_sv[i] = cxzero_sv<cxtype_sv>(); // calculate_jamps accumulates into jamp_sv
          // **NB! in "mixed" precision, using SIMD, calculate_jamps computes MEs for TWO neppV pages with a single channelId! #924
          bool storeChannelWeights = allChannelIds != nullptr || allrnddiagram != nullptr;
          calculate_jamps( params, ( allihel != nullptr ? nullptr : cHel[ihel] ), allihel, allmomenta, allcouplings, iflavorVec, jamp_sv, storeChannelWeights, allNumerators, allDenominators, jamp2_sv, ievt00 );
        }
        // nhelEval may be 0 (no good helicity): the color sum of no helicity leaves allMEs at 0
        if( nhelEval > 0 )
        {
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          color_sum_cpu_blas( allMEs, MEs_ighel, MEs_ighel2, ghelJamp_sv.data(), nhelEval, ievt00 );
#else
          color_sum_cpu_blas( allMEs, MEs_ighel, nullptr, ghelJamp_sv.data(), nhelEval, ievt00 );
#endif
        }
      }
      else
#endif // MGONGPU_CPP_HAS_BLAS
      {
        for( int ighel = 0; ighel < nhelEval; ighel++ )
        {
          const int ihel = ( allihel != nullptr ? -1 : cGoodHel[ighel] ); // unused with allihel
          cxtype_amp_sv jamp_sv[nParity * njampso] = {}; // fixed nasty bug (omitting 'nParity' caused memory corruptions after calling calculate_jamps)
          // **NB! in "mixed" precision, using SIMD, calculate_jamps computes MEs for TWO neppV pages with a single channelId! #924
          bool storeChannelWeights = allChannelIds != nullptr || allrnddiagram != nullptr;
          calculate_jamps( params, ( allihel != nullptr ? nullptr : cHel[ihel] ), allihel, allmomenta, allcouplings, iflavorVec, jamp_sv, storeChannelWeights, allNumerators, allDenominators, jamp2_sv, ievt00 );
          color_sum_cpu( allMEs, jamp_sv, ievt00 );
          MEs_ighel[ighel] = E_ACCESS::kernelAccess( E_ACCESS::ieventAccessRecord( allMEs, ievt00 ) );
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          MEs_ighel2[ighel] = E_ACCESS::kernelAccess( E_ACCESS::ieventAccessRecord( allMEs, ievt00 + neppV ) );
#endif
        }
      }

      // Event-by-event random choice of helicity #403
      for( int ieppV = 0; ieppV < neppV; ++ieppV )
      {
        const int ievt = ievt00 + ieppV;
        if( allihel != nullptr )
        {
          // No choice: each event was evaluated for its given helicity only, whose |M|^2 (signed,
          // with an interference) is the color sum above
          allselhel[ievt] = allihel[ievt] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          allselhel[ievt + neppV] = allihel[ievt + neppV] + 1;
#endif
          continue;
        }
        if constexpr( nampso > 1 )
        {
          if( sampleSignedHelicity )
          {
            fptype ighelSum[ncomb];
            for( int ighel = 0; ighel < cNGoodHel; ighel++ )
#if defined MGONGPU_CPPSIMD
              ighelSum[ighel] = MEs_ighel[ighel][ieppV];
#else
              ighelSum[ighel] = MEs_ighel[ighel];
#endif
            *E_ACCESS::ieventAccessRecord( allMEs, ievt ) = select_helicity_signed( params, ighelSum, allrndhel[ievt], allselhel[ievt] );
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
            const int ievt2 = ievt00 + ieppV + neppV;
            for( int ighel = 0; ighel < cNGoodHel; ighel++ )
              ighelSum[ighel] = MEs_ighel2[ighel][ieppV];
            *E_ACCESS::ieventAccessRecord( allMEs, ievt2 ) = select_helicity_signed( params, ighelSum, allrndhel[ievt2], allselhel[ievt2] );
#endif
            continue;
          }
        }
        for( int ighel = 0; ighel < cNGoodHel; ighel++ )
        {
#if defined MGONGPU_CPPSIMD
          const bool okhel = allrndhel[ievt] < ( MEs_ighel[ighel][ieppV] / MEs_ighel[cNGoodHel - 1][ieppV] );
#else
          const bool okhel = allrndhel[ievt] < ( MEs_ighel[ighel] / MEs_ighel[cNGoodHel - 1] );
#endif
          if( okhel )
          {
            const int ihelF = cGoodHel[ighel] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
            allselhel[ievt] = ihelF;
            break;
          }
        }
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        const int ievt2 = ievt00 + ieppV + neppV;
        for( int ighel = 0; ighel < cNGoodHel; ighel++ )
        {
          if( allrndhel[ievt2] < ( MEs_ighel2[ighel][ieppV] / MEs_ighel2[cNGoodHel - 1][ieppV] ) )
          {
            const int ihelF = cGoodHel[ighel] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
            allselhel[ievt2] = ihelF;
            break;
          }
        }
#endif
      }
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      const int vecsize = 2 * neppV;
#else
      const int vecsize = neppV;
#endif
      unsigned int channelIdVec[vecsize];
      if( allChannelIds != nullptr )
      {
        for( int ieppV = 0; ieppV < vecsize; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          channelIdVec[ieppV] = allChannelIds[ievt];
        }
      }

      // Event-by-event random choice of channel
      if( allrnddiagram != nullptr )
      {
        for( int ieppV = 0; ieppV < vecsize; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          fptype numerator_sum = 0., normalization = 0.;
          for( unsigned int ichan = 0; ichan < mgOnGpu::nchannels; ichan++ )
          {
            if( mgOnGpu::channel2iconfig[ichan] == -1 ) continue;
            normalization += allNumerators[ievt / neppV * neppV * ndiagrams +
                                           ichan * neppV + ieppV % neppV];
          }
          channelIdVec[ieppV] = mgOnGpu::nchannels;
          for( unsigned int ichan = 0; ichan < mgOnGpu::nchannels; ichan++ )
          {
            if( mgOnGpu::channel2iconfig[ichan] == -1 ) continue;
            numerator_sum += allNumerators[ievt / neppV * neppV * ndiagrams +
                                           ichan * neppV + ieppV % neppV];
            if( allrnddiagram[ievt] < numerator_sum / normalization )
            {
              channelIdVec[ieppV] = ichan + 1;
              break;
            }
          }
          allDiagramIdsOut[ievt] = channelIdVec[ieppV];
        }
      }

      // Event-by-event random choice of color #402
      if( allChannelIds != nullptr || allrnddiagram != nullptr ) // no event-by-event choice of color if channelId == 0 (fix FPE #783)
      {
        for( int ieppV = 0; ieppV < vecsize; ++ieppV )
        {
          unsigned int channelId = channelIdVec[ieppV];
          if( channelId > mgOnGpu::nchannels )
          {
            printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%d which is greater than nchannels=%d\n", channelId, mgOnGpu::nchannels );
            assert( channelId <= mgOnGpu::nchannels ); // SANITY CHECK #919 #910
          }
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
#ifdef MGONGPU_CPPSIMD
            if( mgOnGpu::icolamp[iconfig - 1][icolC] ) targetamp[icolC] +=
              jamp2_sv[icolC + ncolor_flow * ( ieppV / neppV )][ieppV % neppV];
#else
            if( mgOnGpu::icolamp[iconfig - 1][icolC] ) targetamp[icolC] +=
              jamp2_sv[icolC + ncolor_flow * ( ieppV / neppV )];
#endif
          }
          const int ievt = ievt00 + ieppV;
          for( int icolC = 0; icolC < ncolor_flow; icolC++ )
          {
            if( allrndcol[ievt] < ( targetamp[icolC] / targetamp[ncolor_flow - 1] ) )
            {
              allselcol[ievt] = icolC + 1; // NB Fortran [1,ncolor], cudacpp [0,ncolor-1]
              break;
            }
          }
        }
      }
      else
      {
        for( int ieppV = 0; ieppV < neppV; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          allselcol[ievt] = 0; // no color selected in Fortran range [1,ncolor] if channelId == 0 (see #931)
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          const int ievt2 = ievt00 + ieppV + neppV;
          allselcol[ievt2] = 0; // no color selected in Fortran range [1,ncolor] if channelId == 0 (see #931)
#endif
        }
      }
    }
    // *** END OF PART 1 - C++ (loop on event pages)

    // PART 2 - FINALISATION (after calculate_jamps)
    // Get the final |M|^2 as an average over helicities/colors of the running sum of |M|^2 over helicities for the given event
    const bool storeChannelWeights = allChannelIds != nullptr || allrnddiagram != nullptr;
    for( int ipagV = 0; ipagV < npagV; ++ipagV )
    {
      const int ievt0 = ipagV * neppV;
      fptype* MEs = E_ACCESS::ieventAccessRecord( allMEs, ievt0 );
      fptype_sv& MEs_sv = E_ACCESS::kernelAccess( MEs );
      MEs_sv = MEs_sv * static_cast<fptype>( broken_symmetry_factor( iflavorVec[ievt0] ) ) / static_cast<fptype>( helcolDenominators[0] );
      if( storeChannelWeights ) // fix segfault #892 (not 'channelIds[0] != 0')
      {
        // The numerators have already been accumulated over all good helicities in place (running sum
        // over the helicity loop), so there is no helicity dimension to sum here. The denominator is
        // just the sum of all numerators for this event page: derive it once and store it for the
        // downstream consumers (e.g. the multichannel amp2 output).
        fptype_amp* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 * ndiagrams );
        fptype_amp* denominators = DEN_ACCESS::ieventAccessRecord( allDenominators, ievt0 );
        fptype_amp_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
        fptype_amp_sv& denominators_sv = DEN_ACCESS::kernelAccess( denominators );
        denominators_sv = fptype_amp_sv{ 0 };
        for( int idiag = 0; idiag < ndiagrams; ++idiag )
          denominators_sv += numerators_sv[idiag];
        if( mulChannelWeight && allChannelIds != nullptr )
        {
          const unsigned int channelId = getChannelId( allChannelIds, ievt0, false );
          // A vanishing denominator means every diagram's |amp|^2 is zero for this event; clamp to the
          // smallest normal value so the channel-weight division stays finite (0/tiny is 0) rather than
          // turning a matrix element that is simply zero into a nan (see color_sum.cc fix #435 note).
          const fptype_sv safe_den = denominators_sv + std::numeric_limits<fptype>::min();
          MEs_sv *= numerators_sv[channelId - 1] / safe_den;
        }
      }
    }
    mgDebugFinalise();
  }

  //--------------------------------------------------------------------------

} // end namespace
