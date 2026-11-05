// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Sep 2025) for the MadGraph7 CUDACPP plugin.
// Further modified by: A. Valassi (2025).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef COLOR_SUM_H
#define COLOR_SUM_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuCxtypes.h"

#include "ProcessData.h"
#include "GpuAbstraction.h"

#include <cstddef>
#ifdef MGONGPU_CPP_HAS_BLAS
#include <vector> // the batched C++ color sum keeps the jamps of every good helicity
#endif

namespace madmatrix
{
  //--------------------------------------------------------------------------

#ifndef MGONGPU_HAS_NO_BLAS
  // The size of the ghelAllBlasTmp scratch buffer color_sum_blas needs, in fptype2 elements:
  // one fptype2[ncolor*nx2*nhel*nevt] buffer for the BLAS intermediate results and, in mixed
  // precision mode only, one more for the jamps converted from double to float plus one
  // fptype2[nhel*nevt] buffer for the MEs, which are fptype elsewhere. This is the one place
  // the size is defined: both the allocation (MatrixElementKernels.cc) and the reset
  // (color_sum_gpu) come here.
  constexpr std::size_t
  blasColorSumTmpSize( const int nhel, const int nevt )
  {
    std::size_t nfptype2PerEvent = ProcessData::ncolor * mgOnGpu::nx2;
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    nfptype2PerEvent *= 2;  // the jamps converted to float need a buffer of their own
    nfptype2PerEvent += 1;  // the fptype2 matrix elements
#endif
    return nfptype2PerEvent * (std::size_t)nhel * (std::size_t)nevt;
  }
#endif

  //--------------------------------------------------------------------------

  class DeviceAccessJamp
  {
  public:
    static __device__ inline cxtype_amp_ref
    kernelAccessIcolIhelNhel( fptype_amp* buffer, const int icol, const int ihel, const int nhel )
    {
      // the number of jamps: ncolor per amplitude split order (njampso == ncolor
      // without split orders); the real and the imaginary planes are that long
      const int ncolor = ProcessData::njampso;
      const int nevt = gridDim.x * blockDim.x;
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      // (ONE HELICITY) Original "old" striding for CUDA kernels: ncolor separate 2*nevt matrices for each color (ievt last)
      //return cxtype_ref( buffer[icol * 2 * nevt + ievt], buffer[icol * 2 * nevt + nevt + ievt] ); // "old"
      // (ONE HELICITY) New "new1" striding for cuBLAS: two separate ncolor*nevt matrices for each of real and imag (ievt last)
      // The "new1" striding was used for both HASBLAS=hasBlas and hasNoBlas builds and for both CUDA kernels and cuBLAS
      //return cxtype_ref( buffer[0 * ncolor * nevt + icol * nevt + ievt], buffer[1 * ncolor * nevt + icol * nevt + ievt] ); // "new1"
      // (ALL HELICITIES) New striding for cuBLAS: two separate ncolor*nhel*nevt matrices for each of real and imag (ievt last)
      return cxtype_amp_ref( buffer[0 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt],
                         buffer[1 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt] );
    }
    static __device__ inline const cxtype
    kernelAccessIcolIhelNhelConst( const fptype_amp* buffer, const int icol, const int ihel, const int nhel )
    {
      // the number of jamps: ncolor per amplitude split order (njampso == ncolor
      // without split orders); the real and the imaginary planes are that long
      const int ncolor = ProcessData::njampso;
      const int nevt = gridDim.x * blockDim.x;
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      // (ONE HELICITY) Original "old" striding for CUDA kernels: ncolor separate 2*nevt matrices for each color (ievt last)
      //return cxtype_ref( buffer[icol * 2 * nevt + ievt], buffer[icol * 2 * nevt + nevt + ievt] ); // "old"
      // (ONE HELICITY) New "new1" striding for cuBLAS: two separate ncolor*nevt matrices for each of real and imag (ievt last)
      // The "new1" striding was used for both HASBLAS=hasBlas and hasNoBlas builds and for both CUDA kernels and cuBLAS
      //return cxtype_ref( buffer[0 * ncolor * nevt + icol * nevt + ievt], buffer[1 * ncolor * nevt + icol * nevt + ievt] ); // "new1"
      // (ALL HELICITIES) New striding for cuBLAS: two separate ncolor*nhel*nevt matrices for each of real and imag (ievt last)
      return cxtype_amp( buffer[0 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt],
                     buffer[1 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt] );
    }
  };

  //--------------------------------------------------------------------------

  void
  color_sum_gpu( fptype* ghelAllMEs,               // output: allMEs super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
                 const fptype_amp* ghelAllJamps,   // input: allJamps super-buffer[2][ncol][nGoodHel][nevt] for nGoodHel <= ncomb individual helicities
                 fptype_colour* ghelAllBlasTmp,    // tmp: allBlasTmp super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
                 gpuBlasHandle_t* pBlasHandle,     // input: cuBLAS/hipBLAS handle
                 gpuStream_t* ghelStreams,         // input: cuda streams (index is ighel: only the first nGoodHel <= ncomb are non-null)
                 const int nGoodHel,               // input: number of good helicities
                 const int gpublocks,              // input: cuda gpublocks
                 const int gputhreads,             // input: cuda gputhreads
                 const bool processAllHelicities); // input: if true, use blockIdx.y to index helicities

  //--------------------------------------------------------------------------

  __global__ void
  color_sum_kernel( fptype* allMEs,                 // output: allMEs[nevt], add |M|^2 for one specific helicity
                    const fptype_amp* allJamps,     // input: jamp[ncolor*2*nevt] for one specific helicity
                    const int nGoodHel,             // input: number of good helicities
                    const int nevtIfAllHelicities); // input: zero in single-helicity mode, number of events in multi-helicity mode

  //--------------------------------------------------------------------------
}

#endif // COLOR_SUM_H
