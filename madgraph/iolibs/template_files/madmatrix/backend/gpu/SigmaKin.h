// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Backend-owned driver: sigmaKin and everything it calls (calculate_jamps,
// good-helicity filtering, color/channel selection). Declared here so
// CPPProcess.cc's constructor/initProc (P1-generated) can call the setters
// that populate this file's storage, and so umami.cc/MatrixElementKernels.cc
// can call sigmaKin/computeDependentCouplings.

#ifndef SIGMAKIN_H
#define SIGMAKIN_H 1

#include "mgOnGpuConfig.h"
#include "mgOnGpuCxtypes.h" // for cxtype, fptype_amp_sv
#include "GpuAbstraction.h"

namespace madmatrix
{
  __global__ void
  computeDependentCouplings( const fptype* allgs,    // input: Gs[nevt]
                             fptype* allcouplings ); // output: couplings[nevt*ndcoup*2]

  void
  sigmaKin_getGoodHel( const fptype_momenta* allmomenta, // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,       // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec,   // input: index of the flavor combination
                       fptype* allMEs,                   // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype_amp* allNumerators,         // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype_amp* allDenominators,       // output: multichannel denominators[nevt], running_sum_over_helicities
                       fptype_amp_sv* allJamps,           // output: jamp[ncolor*2*nevt]
                       bool* isGoodHel,                  // output: isGoodHel[ncomb] - device array (GPU device implementation)
                       const int nevt );                 // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)

  int                                          // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( const bool* isGoodHel ); // input: isGoodHel[ncomb] - host array

  int                    // output: nGoodHel as last set by sigmaKin_setGoodHel (0 before)
  sigmaKin_getNGoodHel();

  void
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
            fptype_amp* colAllJamp2s,            // tmp: allJamp2s super-buffer for ncolor individual colors, running sum over colors and helicities
            fptype_amp* ghelAllNumerators,       // tmp: allNumerators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            fptype_amp* ghelAllDenominators,     // tmp: allDenominators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            unsigned int* allDiagramIdsOut,      // output: multichannel channelIds[nevt] (1 to #diagrams)
            bool mulChannelWeight,               // if true, multiply channel weight to ME output
            fptype* ghelAllMEs,                  // tmp: allMEs super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            fptype_amp* ghelAllJamps,            // tmp: allJamps super-buffer[2][njampso][nGoodHel][nevt] for nGoodHel <= ncomb individual helicities
            fptype_colour* ghelAllBlasTmp,       // tmp: allBlasTmp super-buffer for nGoodHel <= ncomb individual helicities
            gpuBlasHandle_t* pBlasHandle,        // input: cuBLAS/hipBLAS handle
            gpuStream_t* ghelStreams,            // input: cuda streams (index is ighel: only the first nGoodHel <= ncomb are non-null)
            const bool async,                    // input: if true, run everything asynchronously in first stream in ghelStreams
            const int gpublocks,                 // input: cuda gpublocks
            const int gputhreads,                // input: cuda gputhreads
            // input: when the |M|^2 keeps an interference (squared split orders, nampso > 1),
            // choose the helicity on |T_i| and return sign(T_i) * sum_j |T_j| as madevent does,
            // so that the chosen helicity is exact; false returns sum_j T_j (no effect otherwise)
            const bool sampleSignedHelicity = false,
            // input: the helicity index [0,ncomb) of each event (a device array), which is then
            // evaluated for that helicity alone (and returns its contribution to the helicity-summed
            // |M|^2, see umami_matrix_element); nullptr, the default, sums over the good helicities
            const int* allihel = nullptr );

  // Setters: called once by CPPProcess (P1-generated) to populate this file's
  // otherwise-internal storage, since it can no longer be written directly
  // from a different translation unit.
  void setHelicitiesAndFlavors( const short* tHel, const short* tFlavors );
  void setIndependentParams( const fptype* tIPD );
  void setIndependentCouplings( const cxtype* tIPC );
  void setFlavorCouplings( const int* tIPF_partner1, const int* tIPF_partner2, const cxtype* tIPF_value );
  void setBsmIndepParam( const double* values, int n );
  // Window (in widths) of the $-excluded propagators, the run card bw_cutoff
  void setBwCutoff( const double bwcutoff );
}

#endif // SIGMAKIN_H
