// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Sep 2025) for the MadGraph7 CUDACPP plugin.
// Further modified by: A. Valassi (2025).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef COLOR_SUM_H
#define COLOR_SUM_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuCxtypes.h"

#include "ProcessData.h"

#include <cstddef>
#ifdef MGONGPU_CPP_HAS_BLAS
#include <vector> // the batched C++ color sum keeps the jamps of every good helicity
#endif

namespace madmatrix
{
  //--------------------------------------------------------------------------

  void
  color_sum_cpu( fptype* allMEs,              // output: allMEs[nevt], add |M|^2 for one specific helicity
                 const cxtype_amp_sv* allJamp_sv, // input: jamp_sv[njampso] (float/double) or jamp_sv[2*njampso] (mixed) for one specific helicity
                 const int ievt0 );           // input: first event number in current C++ event page (for CUDA, ievt depends on threadid)

  //--------------------------------------------------------------------------

  // Only defined for processes whose color matrix is large enough that the
  // BLAS call is worth setting up (see blas_wanted): the color sum for every
  // good helicity of one event page in one go.
#ifdef MGONGPU_CPP_HAS_BLAS
  void
  color_sum_cpu_blas( fptype* allMEs,                  // input/output: allMEs[nevt], add |M|^2 summed over all good helicities
                      fptype_sv* MEs_ighel,            // output: [ncomb] running sum of |M|^2 up to ighel (first - and/or only - neppV page)
                      fptype_sv* MEs_ighel2,           // output: [ncomb] the same for the second neppV page (mixed mode only)
                      const cxtype_sv* ghelAllJamp_sv, // input: jamp_sv[nGoodHel][nParity*ncolor] for all good helicities
                      const int nGoodHel,              // input: number of good helicities
                      const int ievt0 );               // input: first event number in current C++ event page
#endif

  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------
}

#endif // COLOR_SUM_H
