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

namespace madmatrix
{
  __global__ void
  computeDependentCouplings( const fptype* allgs,  // input: Gs[nevt]
                             fptype* allcouplings, // output: couplings[nevt*ndcoup*2]
                             const int nevt );     // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)

  void
  sigmaKin_getGoodHel( const fptype_momenta* allmomenta, // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,       // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec,   // input: index of the flavor combination
                       fptype* allMEs,                   // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype_amp* allNumerators,         // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype_amp* allDenominators,       // output: multichannel denominators[nevt], running_sum_over_helicities
                       bool* isGoodHel,                  // output: isGoodHel[ncomb] - host array (C++ implementation)
                       const int nevt );                 // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)

  int                                          // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( const bool* isGoodHel ); // input: isGoodHel[ncomb] - host array

  int                    // output: nGoodHel as last set by sigmaKin_setGoodHel (0 before)
  sigmaKin_getNGoodHel();

  void
  sigmaKin( const fptype_momenta* allmomenta,  // input: momenta[nevt*npar*4]
            const fptype* allcouplings,        // input: couplings[nevt*ndcoup*2]
            const unsigned int* iflavorVec,    // input: index of the flavor combination
            const fptype* allrndhel,           // input: random numbers[nevt] for helicity selection
            const fptype* allrndcol,           // input: random numbers[nevt] for color selection
            const unsigned int* allChannelIds, // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable single-diagram enhancement (fix #899)
            const fptype* allrnddiagram,       // input: random numbers[nevt] for channel sampling
            fptype* allMEs,                    // output: allMEs[nevt], |M|^2 final_avg_over_helicities
            int* allselhel,                    // output: helicity selection[nevt]
            int* allselcol,                    // output: helicity selection[nevt]
            fptype_amp* allNumerators,          // tmp: multichannel numerators[nevt], running_sum_over_helicities
            fptype_amp* allDenominators,        // tmp: multichannel denominators[nevt], running_sum_over_helicities
            unsigned int* allDiagramIdsOut,    // output: multichannel channelIds[nevt] (1 to #diagrams)
            bool mulChannelWeight,             // if true, multiply channel weight to ME output
            const int nevt,                    // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
            // input: when the |M|^2 keeps an interference (squared split orders, nampso > 1),
            // choose the helicity on |T_i| and return sign(T_i) * sum_j |T_j| as madevent does,
            // so that the chosen helicity is exact; false returns sum_j T_j (no effect otherwise)
            const bool sampleSignedHelicity = false,
            // input: the helicity index [0,ncomb) of each event, which is then evaluated for that
            // helicity alone (and returns its contribution to the helicity-summed |M|^2, see
            // umami_matrix_element); nullptr, the default, sums over the good helicities
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

  // What the spin-density matrix (Density.cc) needs on top of sigmaKin: the
  // jamps of one event page for any helicity combination hel[npar], the
  // helicity table cHel[ncomb][npar] (row major) the combinations are built
  // from, and the identical-particle factor of a flavor combination.
  void
  calculate_jamps( const short* hel,                 // input: helicity of each leg, hel[npar], of all events (if allihel == nullptr)
                   const int* allihel,               // input: helicity index [0,ncomb) of each event[nevt] (nullptr: hel for all events)
                   const fptype_momenta* allmomenta, // input: momenta[nevt*npar*4]
                   const fptype* allcouplings,       // input: couplings[nevt*ndcoup*2]
                   const unsigned int* iflavorVec,   // input: indices of the flavor combinations
                   cxtype_amp_sv* allJamp_sv,        // output: jamp_sv[njampso] (float/double) or jamp_sv[2*njampso] (mixed)
                   bool storeChannelWeights,
                   fptype_amp* allNumerators,        // input/output: multichannel numerators[nevt]
                   fptype_amp* allDenominators,      // input/output: multichannel denominators[nevt]
                   fptype_amp_sv* jamp2_sv,          // output: jamp2 for the color choice (nullptr if disabled)
                   const int ievt00 );               // input: first event number in the event page
  const short* getHelicities();
  int broken_symmetry_factor( const int iflavor );
}

#endif // SIGMAKIN_H
