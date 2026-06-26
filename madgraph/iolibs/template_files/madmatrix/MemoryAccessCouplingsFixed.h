// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Apr 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessCouplingsFixed_H
#define MemoryAccessCouplingsFixed_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuCxtypes.h"
#include "mgOnGpuVectors.h"

//#include "MemoryAccessHelpers.h"

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //----------------------------------------------------------------------------

  // A class describing the internal layout of memory buffers for fixed couplings
  // This implementation uses a STRUCT[ndcoup][nx2] "super-buffer" layout: in practice, the cIPC global array
  // From the "super-buffer" for ndcoup different couplings, use idcoupAccessBuffer to access the buffer for one specific coupling
  // [If many implementations are used, a suffix _Sv1 should be appended to the class name]
  class MemoryAccessCouplingsFixedBase //_Sv1
  {
  public:

    // Locate the buffer for a single coupling (output) in a memory super-buffer (input) from the given coupling index (input)
    // [Signature (const) ===> const fptype* iicoupAccessBufferConst( const fptype* buffer, const int iicoup ) <===]
    static __host__ __device__ inline const fptype*
    iicoupAccessBufferConst( const fptype* buffer, // input "super-buffer": in practice, the cIPC global array
                             const int iicoup )
    {
      constexpr int ix2 = 0;
      // NB! this effectively adds an offset "iicoup * nx2"
      return &( buffer[iicoup * nx2 + ix2] ); // STRUCT[idcoup][ix2]
    }

  private:

    // The number of floating point components of a complex number
    static constexpr int nx2 = mgOnGpu::nx2;
  };

  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on implicit kernel rules
  // Its methods use the KernelAccessHelper template - note the use of the template keyword in template function instantiations
  template<bool onDevice>
  class KernelAccessCouplingsFixed
  {
  public:

    // Expose selected functions from MemoryAccessCouplingsFixedBase
    static constexpr auto iicoupAccessBufferConst = MemoryAccessCouplingsFixedBase::iicoupAccessBufferConst;

    // Locate a field (output) in a memory buffer (input) from a kernel event-indexing mechanism (internal) and the given field indexes (input)
    // [Signature (const, SCALAR OR VECTOR) ===> cxtype_sv kernelAccessConst( const fptype* buffer ) <===]
    static __host__ __device__ inline const cxtype_sv
    kernelAccessConst( const fptype* buffer )
    {
      // TRIVIAL ACCESS to fixed-couplings buffers!
      //return cxmake( fptype_sv{ buffer[0] }, fptype_sv{ buffer[1] } ); // NO! BUG #339!
      const fptype_sv r_sv = fptype_sv{ 0 } + buffer[0];
      const fptype_sv i_sv = fptype_sv{ 0 } + buffer[1];
      return cxmake( r_sv, i_sv ); // ugly but effective
    }
  };

  //----------------------------------------------------------------------------

  typedef KernelAccessCouplingsFixed<false> HostAccessCouplingsFixed;
  typedef KernelAccessCouplingsFixed<true> DeviceAccessCouplingsFixed;

  //----------------------------------------------------------------------------

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MemoryAccessCouplingsFixed_H
