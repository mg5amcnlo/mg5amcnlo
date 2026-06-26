// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (May 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessDenominators_H
#define MemoryAccessDenominators_H 1

#include "MemoryAccessGs.h"

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //----------------------------------------------------------------------------

  // A class describing the internal layout of memory buffers for denominators
  // This implementation reuses the plain ARRAY[nevt] implementation of MemoryAccessGs

  typedef KernelAccessGs<false> HostAccessDenominators;
  typedef KernelAccessGs<true> DeviceAccessDenominators;

  //----------------------------------------------------------------------------

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MemoryAccessDenominators_H
