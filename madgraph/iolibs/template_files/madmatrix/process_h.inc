// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MG5aMC CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: S. Roiser (Feb 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: D. Massaro, O. Mattelaer, S. Roiser, J. Teig, A. Thete, A. Valassi (2020-2025).
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
// This file has been automatically generated for CUDA/C++ standalone by
%(info_lines)s
//==========================================================================

#ifndef MG5_%(process_file_name)s_H
#define MG5_%(process_file_name)s_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuVectors.h"

#include "GpuAbstraction.h"
#include "Parameters.h"

#include <vector>

//--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
%(process_class_definitions)s
  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  computeDependentCouplings( const fptype* allgs,    // input: Gs[nevt]
                             fptype* allcouplings ); // output: couplings[nevt*ndcoup*2]
#else
  __global__ void
  computeDependentCouplings( const fptype* allgs,  // input: Gs[nevt]
                             fptype* allcouplings, // output: couplings[nevt*ndcoup*2]
                             const int nevt );     // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL /* clang-format off */
  void
  sigmaKin_getGoodHel( const fptype* allmomenta,       // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,     // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec, // input: index of the flavor combination
                       fptype* allMEs,                 // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype* allNumerators,          // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype* allDenominators,        // output: multichannel denominators[nevt], running_sum_over_helicities
                       fptype_sv* allJamps,            // output: jamp[ncolor*2*nevt]
                       bool* isGoodHel,                // output: isGoodHel[ncomb] - device array (GPU device implementation)
                       const int nevt );               // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
#else
  void
  sigmaKin_getGoodHel( const fptype* allmomenta,       // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,     // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec, // input: index of the flavor combination
                       fptype* allMEs,                 // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype* allNumerators,          // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype* allDenominators,        // output: multichannel denominators[nevt], running_sum_over_helicities
                       bool* isGoodHel,                // output: isGoodHel[ncomb] - host array (C++ implementation)
                       const int nevt );               // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
#endif /* clang-format on */

  //--------------------------------------------------------------------------

  int                                           // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( const bool* isGoodHel ); // input: isGoodHel[ncomb] - host array

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL /* clang-format off */
  void
  sigmaKin( const fptype* allmomenta,           // input: momenta[nevt*npar*4]
            const fptype* allcouplings,         // input: couplings[nevt*ndcoup*2]
            const unsigned int* iflavorVec,     // input: index of the flavor combination
            const fptype* allrndhel,            // input: random numbers[nevt] for helicity selection
            const fptype* allrndcol,            // input: random numbers[nevt] for color selection
            const unsigned int* allChannelIds,  // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable single-diagram enhancement (fix #899/#911)
            const fptype* allrnddiagram,        // input: random numbers[nevt] for channel sampling
            fptype* allMEs,                     // output: allMEs[nevt], |M|^2 final_avg_over_helicities
            int* allselhel,                     // output: helicity selection[nevt]
            int* allselcol,                     // output: helicity selection[nevt]
            fptype* colAllJamp2s,               // tmp: allJamp2s super-buffer for ncolor individual colors, running sum over colors and helicities
            fptype* ghelAllNumerators,          // tmp: allNumerators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            fptype* ghelAllDenominators,        // tmp: allDenominators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            unsigned int* allDiagramIdsOut,     // output: multichannel channelIds[nevt] (1 to #diagrams)
            bool mulChannelWeight,              // if true, multiply channel weight to ME output
            fptype* ghelAllMEs,                 // tmp: allMEs super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            fptype* ghelAllJamps,               // tmp: allJamps super-buffer[2][ncol][nGoodHel][nevt] for nGoodHel <= ncomb individual helicities
            fptype2* ghelAllBlasTmp,            // tmp: allBlasTmp super-buffer for nGoodHel <= ncomb individual helicities
            gpuBlasHandle_t* pBlasHandle,       // input: cuBLAS/hipBLAS handle
            gpuStream_t* ghelStreams,           // input: cuda streams (index is ighel: only the first nGoodHel <= ncomb are non-null)
            const bool async,                   // input: if true, run everything asynchronously in first stream in ghelStreams
            const int gpublocks,                // input: cuda gpublocks
            const int gputhreads );             // input: cuda gputhreads
#else
  void
  sigmaKin( const fptype* allmomenta,           // input: momenta[nevt*npar*4]
            const fptype* allcouplings,         // input: couplings[nevt*ndcoup*2]
            const unsigned int* iflavorVec,     // input: index of the flavor combination
            const fptype* allrndhel,            // input: random numbers[nevt] for helicity selection
            const fptype* allrndcol,            // input: random numbers[nevt] for color selection
            const unsigned int* allChannelIds,  // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable single-diagram enhancement (fix #899)
            const fptype* allrnddiagram,        // input: random numbers[nevt] for channel sampling
            fptype* allMEs,                     // output: allMEs[nevt], |M|^2 final_avg_over_helicities
            int* allselhel,                     // output: helicity selection[nevt]
            int* allselcol,                     // output: helicity selection[nevt]
            fptype* allNumerators,              // tmp: multichannel numerators[nevt], running_sum_over_helicities
            fptype* allDenominators,            // tmp: multichannel denominators[nevt], running_sum_over_helicities
            unsigned int* allDiagramIdsOut,     // output: multichannel channelIds[nevt] (1 to #diagrams)
            bool mulChannelWeight,              // if true, multiply channel weight to ME output
            const int nevt );                   // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
#endif /* clang-format on */

  //--------------------------------------------------------------------------
}

#endif // MG5_%(process_file_name)s_H
