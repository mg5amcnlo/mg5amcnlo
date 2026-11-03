// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Nov 2026.
//
// The parameters the matrix element of one CPPProcess is evaluated with: the
// independent parameters and couplings of the model it uses, the run card
// bw_cutoff and the good helicities. Each CPPProcess, and so each UMAMI instance,
// has its own: two instances of the same library can have different parameters,
// and (on GPUs) live on different devices. The kernels take them as an argument,
// see backend/{cpu,simd,gpu}/SigmaKin.cc.

#ifndef PROCESSPARAMETERS_H
#define PROCESSPARAMETERS_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuCxtypes.h"

#include "Parameters.h"    // nBsmIndepParam
#include "ProcessData.h"   // ncomb, nIPD, nIPC, nIPF
#include "ProcessTables.h" // nMF

namespace madmatrix
{
  struct ProcessParameters
  {
    // The arrays are sized at least 1: a zero-sized array is only a compiler extension
    static constexpr int nIPD = ProcessData::nIPD > 0 ? ProcessData::nIPD : 1;
    static constexpr int nIPC = ProcessData::nIPC > 0 ? ProcessData::nIPC : 1;
    static constexpr int nIPF = ProcessTables::nMF * ProcessData::nIPF > 0 ? ProcessTables::nMF * ProcessData::nIPF : 1;
    static constexpr int nBsm = Parameters::nBsmIndepParam > 0 ? Parameters::nBsmIndepParam : 1;

    fptype IPD[nIPD];          // independent parameters (cIPD in EvaluateDiagrams.inc)
    fptype IPC[nIPC * 2];      // independent couplings, real and imaginary parts (cIPC)
    int IPF_partner1[nIPF];    // flavor couplings (cIPF)
    int IPF_partner2[nIPF];
    fptype IPF_value[nIPF * 2];
    double bsmIndepParam[nBsm]; // BSM parameters needed by the alphas-dependent couplings
    fptype bwcutoff = 15.;     // window (in widths) of the $-excluded propagators, the run card bw_cutoff
    int nGoodHel = 0;          // the good helicities: goodHel[0..nGoodHel) index the helicity table
    int goodHel[ProcessData::ncomb] = {};
#ifndef MGONGPU_HARDCODE_PARAM
    // The model parameters these were taken from (host memory), for the HelAmps routines
    // that read a model parameter on CPUs (see Parameters::current)
    const Parameters* model = nullptr;
#endif
  };
}

#endif // PROCESSPARAMETERS_H
