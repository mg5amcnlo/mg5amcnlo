// Copyright (C) 2010 The MadGraph7 development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MadGraph7 CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: A. Valassi (Sep 2021) for the MadGraph7 CUDACPP plugin.
// Further modified by: A. Valassi (2021-2024) for the MadGraph7 CUDACPP plugin.
// Integrated with the MadGraph7 project in Feb 2026.
// ==========================================================================
// This file has been automatically generated for CUDA/C++ standalone by
%(info_lines)s
//==========================================================================

#ifndef Parameters_H
#define Parameters_H

#include "mgOnGpuConfig.h"

#include "mgOnGpuCxtypes.h"
#ifdef MGONGPU_HAS_VECTORS_H // the simd backend (see its mgOnGpuConfig.h)
#include "mgOnGpuVectors.h"
#endif

#include "constexpr_math.h"

//==========================================================================

// AV Jan 2024 (PR #625): this ugly #define was the only way I found to avoid creating arrays[nBsm] in CPPProcess.cc if nBsm is 0
// The problem is that nBsm is determined when generating Parameters.h, which happens after CPPProcess.cc has already been generated
// For simplicity, keep this code hardcoded also for SM processes (a nullptr is needed as in the case nBsm == 0)
%(bsmdefine)s

#ifndef MGONGPU_HARDCODE_PARAM%(eftwarn0)s

#include "read_slha.h"

#include <map>
#include <set>
#include <string>

//One namespace. Split ber backend.
namespace madmatrix
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
          value[i] = nullptr; // flavor slots with no coupling stay null so the
                              // cIPF_value setup (value[j] ? *value[j] : 0) does
                              // not dereference an uninitialised pointer
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

    // Set the value of an independent, SLHA-card-level parameter by name at
    // runtime (overriding the value read from the param card), and recompute
    // every parameter/coupling that depends on it. Returns false if "name" is
    // not the name of a settable independent parameter (e.g. it names an
    // internal parameter derived from others, or a coupling, neither of which
    // can be set directly since they would just be recomputed from the
    // underlying parameters on the next call).
    // "name" may be either the parameter's model name (e.g. "mdl_MT") or its
    // SLHA card location, given as "<block> <index> [<index2>]" (e.g. "mass
    // 6"); the block name is matched case-insensitively.
    bool setParameterByName( std::string name, double real, double imag );
    // Retrieve the current value of any independent parameter or coupling by
    // name (real or complex; "name" accepts the same two forms as
    // setParameterByName). NB: dependent (aS-dependent) parameters and
    // couplings are not stored here at all in this backend (they are
    // recomputed fresh every event batch by computeDependentCouplings()), so
    // they cannot be retrieved by name; only independent ones can. Returns
    // false if "name" is unknown.
    bool getParameterByName( std::string name, double* real, double* imag );
    // Drop all overrides previously set via setParameterByName(), so that the
    // next setIndependentParameters(slha) call starts from a clean slate (e.g.
    // when (re-)initializing from a param card). Does not itself recompute
    // anything; the caller is expected to call setIndependentParameters()/
    // setIndependentCouplings() afterward (as initProc() does).
    void resetParameterOverrides();

  private:

    // If "name" is of the form "<block> <index> [<index2>]", resolve it (via
    // m_slha_key_to_name) to the corresponding parameter's model name and
    // return true; otherwise leave "resolved" untouched and return false.
    bool resolveSlhaKey( const std::string& name, std::string& resolved ) const;

    // Name-keyed lookup tables populated by registerParameters(), backing
    // setParameterByName()/getParameterByName() above.
    void registerParameters();
    std::map<std::string, double*> m_real_params;
    // Two separate maps, not one: independent *couplings* (eg GC_72 below)
    // are declared cxtype, which resolves to cxsmpl<float> at FPTYPE=f, so
    // this map's value_type has to be cxtype too (a mismatched hardcoded
    // cxsmpl<double> here fails to compile the moment any coupling is
    // actually registered below, eg complex effective couplings like
    // ggH's) -- while independent model *parameters* (eg mdl_I1x33) always
    // stay plain cxsmpl<double>, matching m_real_params' own always-double
    // real parameters, and go into m_complex_model_params below instead;
    // see write_register_parameters()'s own doc comment (export_cpp.py)
    // for why these can't share one map/type.
    std::map<std::string, cxtype*> m_complex_params;
    std::map<std::string, cxsmpl<double>*> m_complex_model_params;
    std::set<std::string> m_settable_params;
    std::map<std::string, double> m_param_overrides;
    std::map<std::string, std::string> m_slha_key_to_name;
    SLHAReader m_slha;

    static Parameters* instance;
  };

  // Generic readers for initProc: dereference an array of pointer-to-Parameters-member
  // (a per-process compile-time table) into a matching array of tIPD/tIPC values.
  template<int N>
  inline void gatherFptype( const Parameters* pars, double Parameters::* const ( &members )[N], fptype ( &out )[N] )
  {
    for( int i = 0; i < N; ++i ) out[i] = (fptype)( pars->*members[i] );
  }

  template<int N>
  inline void gatherCxtype( const Parameters* pars, cxtype Parameters::* const ( &members )[N], cxtype ( &out )[N] )
  {
    for( int i = 0; i < N; ++i ) out[i] = cxmake( pars->*members[i] );
  }

} // end namespace madmatrix

#else%(eftwarn1)s

#include <cassert>
#include <limits>

//One namespace. Split ber backend.
namespace madmatrix
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

} // end namespace madmatrix

#endif

//==========================================================================

//One namespace. Split ber backend.
namespace madmatrix
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

} // end namespace madmatrix

//==========================================================================

#endif // Parameters_H
