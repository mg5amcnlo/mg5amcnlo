// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
//==========================================================================
// --hel_recycling (CPU only): what the recycled chunks need of SigmaKin.cc.
// A big recycled block is cut into functions (hr_chunk_<i>) that
// calculate_all_jamps calls in order; they are written into translation units
// of their own, HelRecyclingChunks_<k>.cc next to HelRecycling.inc, so that
// the build compiles them in parallel instead of in one SigmaKin.cc. The
// run-time tables of SigmaKin.cc they read (cIPD, cFlavors, cBWCUTOFF) reach
// them as arguments; this is the rest of SigmaKin.cc's scope.
//==========================================================================
#ifndef HELRECYCLINGCHUNKS_H
#define HELRECYCLINGCHUNKS_H 1

#include "SigmaKin.h"

#include "CPPProcess.h" // ProcessData.h, Parameters.h, HelAmps_<model>.h transitively
#include "ProcessTables.h"

#include "MemoryAccessAmplitudes.h"
#include "MemoryAccessCouplings.h"
#include "MemoryAccessCouplingsFixed.h"
#include "MemoryAccessIflavorVec.h"
#include "MemoryAccessMomenta.h"
#include "MemoryAccessWavefunctions.h"
#include "mgOnGpuConfig.h"

namespace madmatrix
{
  using namespace ProcessData;
  using namespace ProcessTables;

  // As in SigmaKin.cc: the chunks index the jamp tables with it
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
  constexpr int nParity = 2;
#else
  constexpr int nParity = 1;
#endif

  // the tabulated P1N groups (a fragment of this namespace, after nParity)
#include "HelRecyclingTables.h"
}

#endif // HELRECYCLINGCHUNKS_H
