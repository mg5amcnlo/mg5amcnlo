// Copyright (C) 2010 The MadGraph7 development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MadGraph7 CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: S. Roiser (Feb 2020) for the MadGraph7 CUDACPP plugin.
// Further modified by: S. Hageboeck, O. Mattelaer, S. Roiser, J. Teig, A. Valassi, Z. Wettersten (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
// This file has been automatically generated for CUDA/C++ standalone by
%(info_lines)s
//==========================================================================

#include "CPPProcess.h"

#include "mgOnGpuConfig.h"

%(hel_amps_h)s
#include "MemoryAccessAmplitudes.h"
#include "MemoryAccessIflavorVec.h"
#include "MemoryAccessChannelIds.h"
#include "MemoryAccessCouplings.h"
#include "MemoryAccessCouplingsFixed.h"
#include "MemoryAccessGs.h"
#include "MemoryAccessMatrixElements.h"
#include "MemoryAccessMomenta.h"
#include "MemoryAccessWavefunctions.h"
#include "color_sum.h"

#include "MemoryAccessDenominators.h"
#include "MemoryAccessNumerators.h"
#include "ColorData.h"
#include "fpe.h"           // for fpeEnable (constructor, MGONGPUCPP_DEBUG builds)

#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>

// Test ncu metrics for CUDA thread divergence
#undef MGONGPU_TEST_DIVERGENCE
//#define MGONGPU_TEST_DIVERGENCE 1

//--------------------------------------------------------------------------

%(process_function_definitions)s
//==========================================================================
