// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (May 2022) for the MadGraph7 CUDACPP plugin.
// Further modified by: A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessDenominators_H
#define MemoryAccessDenominators_H 1

#include "MemoryAccessGs.h"

//One namespace. Split ber backend.
namespace madmatrix
{
  //----------------------------------------------------------------------------

  // A class describing the internal layout of memory buffers for denominators
  // This implementation reuses the plain ARRAY[nevt] implementation of MemoryAccessGs

  typedef KernelAccessGs<false> HostAccessDenominators;
  typedef KernelAccessGs<true> DeviceAccessDenominators;

  //----------------------------------------------------------------------------

} // end namespace madmatrix

#endif // MemoryAccessDenominators_H
