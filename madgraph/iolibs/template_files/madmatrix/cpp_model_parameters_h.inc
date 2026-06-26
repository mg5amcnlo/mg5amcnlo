// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MG5aMC CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: A. Valassi (Sep 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2021-2024) for the MG5aMC CUDACPP plugin.
// Integrated with the MadGraph7 project in Feb 2026.
// ==========================================================================
// This file has been automatically generated for CUDA/C++ standalone by
%(info_lines)s
//==========================================================================

#ifndef Parameters_H
#define Parameters_H

#include "mgOnGpuConfig.h"

#include "mgOnGpuCxtypes.h"
#include "mgOnGpuVectors.h"

#include "constexpr_math.h"

//==========================================================================

// AV Jan 2024 (PR #625): this ugly #define was the only way I found to avoid creating arrays[nBsm] in CPPProcess.cc if nBsm is 0
// The problem is that nBsm is determined when generating Parameters.h, which happens after CPPProcess.cc has already been generated
// For simplicity, keep this code hardcoded also for SM processes (a nullptr is needed as in the case nBsm == 0)
%(bsmdefine)s

#ifndef MGONGPU_HARDCODE_PARAM%(eftwarn0)s

#include "read_slha.h"

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  // Define FLV_COUPLING struct
  struct FLV_COUPLING {
    static constexpr int max_flavor = %(max_flavor)d;

    int partner1[max_flavor];
    int partner2[max_flavor];
    cxtype* value[max_flavor]; // because it will refer already calculated couplings

    FLV_COUPLING() {
      for (int i = 0; i < max_flavor; ++i) {
          partner1[i] = -1;
          partner2[i] = -1;
      }
    }
  };

  class Parameters
  {
  public:

    static Parameters* getInstance();

    // Define "zero"
    double zero, ZERO;

    %(independent_parameters)s

    %(independent_couplings)s

    %(dependent_parameters)s

    %(dependent_couplings)s

    %(flavor_independent_couplings)s

    %(flavor_dependent_couplings)s

    // Set parameters that are unchanged during the run
    void setIndependentParameters( SLHAReader& slha );

    // Set couplings that are unchanged during the run
    void setIndependentCouplings();

    // Set parameters that are changed event by event
    //void setDependentParameters(); // now computed event-by-event (running alphas #373)

    // Set couplings that are changed event by event
    //void setDependentCouplings(); // now computed event-by-event (running alphas #373)

    // Print parameters that are unchanged during the run
    void printIndependentParameters();

    // Print couplings that are unchanged during the run
    void printIndependentCouplings();

    // Print parameters that are changed event by event
    //void printDependentParameters(); // now computed event-by-event (running alphas #373)

    // Print couplings that are changed event by event
    //void printDependentCouplings(); // now computed event-by-event (running alphas #373)

    // BSM parameters that do not depend on alphaS but are needed in the computation of alphaS-dependent couplings;
    static constexpr int nBsmIndepParam = %(nbsmip)i;
    %(hasbsmip)sdouble mdl_bsmIndepParam[nBsmIndepParam];

  private:

    static Parameters* instance;
  };

} // end namespace mg5amcGpu/mg5amcCpu

#else%(eftwarn1)s

#include <cassert>
#include <limits>

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  // Hardcoded constexpr physics parameters
  namespace Parameters // keep the same name rather than HardcodedParameters for simplicity
  {
    // Model parameters independent of aS
    constexpr double zero = 0;
    constexpr double ZERO = 0;
    %(hardcoded_independent_parameters)s
    // Model couplings independent of aS
    %(hardcoded_independent_couplings)s
    // Model parameters dependent on aS
    %(hardcoded_dependent_parameters)s
    // Model couplings dependent on aS
    %(hardcoded_dependent_couplings)s
    // Print parameters that are unchanged during the run
    void printIndependentParameters();

    // Print couplings that are unchanged during the run
    void printIndependentCouplings();

    // Print parameters that are changed event by event
    //void printDependentParameters(); // now computed event-by-event (running alphas #373)

    // Print couplings that are changed event by event
    //void printDependentCouplings(); // now computed event-by-event (running alphas #373)

    // BSM parameters that do not depend on alphaS but are needed in the computation of alphaS-dependent couplings;
    constexpr int nBsmIndepParam = %(nbsmip)i;
    %(hasbsmip)s__device__ constexpr double mdl_bsmIndepParam[nBsmIndepParam] = { %(bsmip)s };
  }

} // end namespace mg5amcGpu/mg5amcCpu

#endif

//==========================================================================

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  namespace Parameters_dependentCouplings
  {
    constexpr size_t ndcoup = %(ndcoup)d; // #couplings that vary event by event because they depend on the running alphas QCD
%(idcoup)s
    struct DependentCouplings_sv
    {
%(dcoupdecl)s
    };
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"        // e.g. <<warning: unused parameter ‘G’ [-Wunused-parameter]>>
#pragma GCC diagnostic ignored "-Wunused-variable"         // e.g. <<warning: unused variable ‘mdl_G__exp__2’ [-Wunused-variable]>>
#pragma GCC diagnostic ignored "-Wunused-but-set-variable" // e.g. <<warning: variable ‘mdl_G__exp__2’ set but not used [-Wunused-but-set-variable]>>
#ifdef MGONGPUCPP_GPUIMPL
#pragma nv_diagnostic push
#pragma nv_diag_suppress 177 // e.g. <<warning #177-D: variable "mdl_G__exp__2" was declared but never referenced>>
#endif
    __host__ __device__ inline const DependentCouplings_sv computeDependentCouplings_fromG( const fptype_sv& G_sv, const double* bsmIndepParamPtr )
    {
#ifdef MGONGPU_HARDCODE_PARAM
      using namespace Parameters;
#else%(eftspecial0)s
#endif
      // NB: hardcode cxtype cI(0,1) instead of cxtype (or hardcoded cxsmpl) mdl_complexi (which exists in Parameters) because:
      // (1) mdl_complexi is always (0,1); (2) mdl_complexi is undefined in device code; (3) need cxsmpl conversion to cxtype in code below
      const cxtype cI( 0., 1. );
      DependentCouplings_sv out;
#if not( defined MGONGPU_CPPSIMD && defined MGONGPU_FPTYPE_FLOAT )
      // Couplings are (scalar, or vector of) doubles, or scalar floats - default implementation
      {
        const fptype_sv& G = G_sv;
        // Model parameters dependent on aS
%(dcoupsetdpar)s
        // Model couplings dependent on aS
%(dcoupsetdcoup)s
      }
#else
      // Couplings are VECTORS OF FLOATS: #439 special handling is needed (variable Gs are vector floats, fixed parameters are scalar doubles)
      // Use an explicit loop to avoid <<error: conversion of scalar ‘double’ to vector ‘fptype_sv’ {aka ‘__vector(8) float’} involves truncation>>
      // Problems may come e.g. in EFTs from multiplying a vector float (related to aS-dependent G) by a scalar double (aS-independent parameters)
      // (NB in pure SM processes this special handling is not needed, but we keep it here for simplicity, see PR #824)%(dcoupoutfptypev2)s
      for( int i = 0; i < neppV; i++ )
      {
        const fptype& G = G_sv[i];
        // Model parameters dependent on aS
%(dcoupsetdpar2)s
        // Model couplings dependent on aS
  %(dcoupsetdcoup2)s
      }%(dcoupoutdcoup2)s
#endif
      return out;
    }
#ifdef MGONGPUCPP_GPUIMPL
#pragma GCC diagnostic pop
#pragma nv_diagnostic pop
#endif
  }

  //==========================================================================

  namespace Parameters_independentCouplings
  {
    constexpr size_t nicoup = %(nicoup)d; // #couplings that are fixed for all events because they do not depend on the running alphas QCD
%(iicoup)s
  }

  //==========================================================================

#pragma GCC diagnostic push
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wunused-but-set-variable" // e.g. <<warning: variable ‘couplings_sv’ set but not used [-Wunused-but-set-variable]>>
#endif
  // Compute the output couplings (e.g. gc10 and gc11) from the input gs
  template<class G_ACCESS, class C_ACCESS>
  __device__ inline void
  G2COUP( const fptype gs[],
          fptype couplings[],
          const double* bsmIndepParamPtr )
  {
    mgDebug( 0, __FUNCTION__ );
    using namespace Parameters_dependentCouplings;
    const fptype_sv& gs_sv = G_ACCESS::kernelAccessConst( gs );
    DependentCouplings_sv couplings_sv = computeDependentCouplings_fromG( gs_sv, bsmIndepParamPtr );
%(dcoupaccessbuffer)s%(dcoupkernelaccess)s%(dcoupcompute)s
    mgDebug( 1, __FUNCTION__ );
    return;
  }
#pragma GCC diagnostic pop

} // end namespace mg5amcGpu/mg5amcCpu

//==========================================================================

#endif // Parameters_H
