// Copyright (C) 2010 The MadGraph7 development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MadGraph7 CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: S. Roiser (Feb 2020) for the MadGraph7 CUDACPP plugin.
// Further modified by: D. Massaro, O. Mattelaer, S. Roiser, J. Teig, A. Thete, A. Valassi (2020-2025).
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
// This file has been automatically generated for CUDA/C++ standalone by
%(info_lines)s
//==========================================================================

#ifndef MG5_%(process_file_name)s_H
#define MG5_%(process_file_name)s_H 1

#include "mgOnGpuConfig.h"

#ifdef MGONGPU_HAS_VECTORS_H // the simd backend (see its mgOnGpuConfig.h)
#include "mgOnGpuVectors.h"
#else
#include "mgOnGpuCxtypes.h"
#endif

#include "Parameters.h"
#include "ProcessData.h"
#include "ProcessParameters.h"
#include "ProcessTables.h" // for nMF (flavor-coupling setup in initProc)
#include "SigmaKin.h"
%(helamps_h)s

#include <memory>
#include <vector>

//--------------------------------------------------------------------------

namespace madmatrix
{
%(process_class_definitions)s
  //--------------------------------------------------------------------------
  // sigmaKin, computeDependentCouplings and friends are backend-owned:
  // see backend/{cpu,simd,gpu}/SigmaKin.h for their declarations.
  //--------------------------------------------------------------------------
}

#endif // MG5_%(process_file_name)s_H
