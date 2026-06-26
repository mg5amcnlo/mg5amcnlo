// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Sep 2025) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2025).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef COLOR_SUM_H
#define COLOR_SUM_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuVectors.h"

#include "CPPProcess.h"
#include "GpuAbstraction.h"

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  class DeviceAccessJamp
  {
  public:
    static __device__ inline cxtype_ref
    kernelAccessIcolIhelNhel( fptype* buffer, const int icol, const int ihel, const int nhel )
    {
      const int ncolor = CPPProcess::ncolor; // the number of leading colors
      const int nevt = gridDim.x * blockDim.x;
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      // (ONE HELICITY) Original "old" striding for CUDA kernels: ncolor separate 2*nevt matrices for each color (ievt last)
      //return cxtype_ref( buffer[icol * 2 * nevt + ievt], buffer[icol * 2 * nevt + nevt + ievt] ); // "old"
      // (ONE HELICITY) New "new1" striding for cuBLAS: two separate ncolor*nevt matrices for each of real and imag (ievt last)
      // The "new1" striding was used for both HASBLAS=hasBlas and hasNoBlas builds and for both CUDA kernels and cuBLAS
      //return cxtype_ref( buffer[0 * ncolor * nevt + icol * nevt + ievt], buffer[1 * ncolor * nevt + icol * nevt + ievt] ); // "new1"
      // (ALL HELICITIES) New striding for cuBLAS: two separate ncolor*nhel*nevt matrices for each of real and imag (ievt last)
      return cxtype_ref( buffer[0 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt],
                         buffer[1 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt] );
    }
    static __device__ inline const cxtype
    kernelAccessIcolIhelNhelConst( const fptype* buffer, const int icol, const int ihel, const int nhel )
    {
      const int ncolor = CPPProcess::ncolor; // the number of leading colors
      const int nevt = gridDim.x * blockDim.x;
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      // (ONE HELICITY) Original "old" striding for CUDA kernels: ncolor separate 2*nevt matrices for each color (ievt last)
      //return cxtype_ref( buffer[icol * 2 * nevt + ievt], buffer[icol * 2 * nevt + nevt + ievt] ); // "old"
      // (ONE HELICITY) New "new1" striding for cuBLAS: two separate ncolor*nevt matrices for each of real and imag (ievt last)
      // The "new1" striding was used for both HASBLAS=hasBlas and hasNoBlas builds and for both CUDA kernels and cuBLAS
      //return cxtype_ref( buffer[0 * ncolor * nevt + icol * nevt + ievt], buffer[1 * ncolor * nevt + icol * nevt + ievt] ); // "new1"
      // (ALL HELICITIES) New striding for cuBLAS: two separate ncolor*nhel*nevt matrices for each of real and imag (ievt last)
      return cxtype( buffer[0 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt],
                     buffer[1 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt] );
    }
  };
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  void createNormalizedColorMatrix();
#endif

  //--------------------------------------------------------------------------

#ifndef MGONGPUCPP_GPUIMPL
  void
  color_sum_cpu( fptype* allMEs,              // output: allMEs[nevt], add |M|^2 for one specific helicity
                 const cxtype_sv* allJamp_sv, // input: jamp_sv[ncolor] (float/double) or jamp_sv[2*ncolor] (mixed) for one specific helicity
                 const int ievt0 );           // input: first event number in current C++ event page (for CUDA, ievt depends on threadid)
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  void
  color_sum_gpu( fptype* ghelAllMEs,               // output: allMEs super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
                 const fptype* ghelAllJamps,       // input: allJamps super-buffer[2][ncol][nGoodHel][nevt] for nGoodHel <= ncomb individual helicities
                 fptype2* ghelAllBlasTmp,          // tmp: allBlasTmp super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
                 gpuBlasHandle_t* pBlasHandle,     // input: cuBLAS/hipBLAS handle
                 gpuStream_t* ghelStreams,         // input: cuda streams (index is ighel: only the first nGoodHel <= ncomb are non-null)
                 const int nGoodHel,               // input: number of good helicities
                 const int gpublocks,              // input: cuda gpublocks
                 const int gputhreads,             // input: cuda gputhreads
                 const bool processAllHelicities); // input: if true, use blockIdx.y to index helicities
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  color_sum_kernel( fptype* allMEs,                 // output: allMEs[nevt], add |M|^2 for one specific helicity
                    const fptype* allJamps,         // input: jamp[ncolor*2*nevt] for one specific helicity
                    const int nGoodHel,             // input: number of good helicities
                    const int nevtIfAllHelicities); // input: zero in single-helicity mode, number of events in multi-helicity mode
#endif

  //--------------------------------------------------------------------------
}

#endif // COLOR_SUM_H
