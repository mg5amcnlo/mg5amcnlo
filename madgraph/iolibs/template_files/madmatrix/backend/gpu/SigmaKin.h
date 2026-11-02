// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Backend-owned driver: sigmaKin and everything it calls (calculate_jamps,
// good-helicity filtering, color/channel selection). Declared here so that
// umami.cc/MatrixElementKernels.cc can call them. They all take the parameters
// of the process they evaluate (ProcessParameters.h) as an argument.

#ifndef SIGMAKIN_H
#define SIGMAKIN_H 1

#include "mgOnGpuConfig.h"
#include "mgOnGpuCxtypes.h" // for cxtype, fptype_amp_sv
#include "GpuAbstraction.h"
#include "ProcessParameters.h"

#include <type_traits>

// A kernel argument that the kernel reads in place, without a copy per thread: from
// the constant bank, as __constant__ memory (CUDA 11.7 and later; HIP passes the
// kernel arguments in a read-only segment of their own)
#if defined __CUDACC__ && ( __CUDACC_VER_MAJOR__ > 11 || ( __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 7 ) )
#define MGONGPU_GRID_CONSTANT __grid_constant__
#else
#define MGONGPU_GRID_CONSTANT
#endif

namespace madmatrix
{
  // The parameters of the process as the kernels take them: by value, which puts them
  // in the constant bank (as fast as __constant__ memory, and private to each launch,
  // hence to each instance and device), unless they do not fit in the kernel arguments
  // (4 kB): a pointer to a copy in global memory (DeviceProcessParameters) then. The
  // latter can be forced with -DMGONGPU_PARAMS_IN_GLOBAL_MEMORY.
#ifdef MGONGPU_PARAMS_IN_GLOBAL_MEMORY
  constexpr bool processParametersByValue = false;
#else
  constexpr bool processParametersByValue = sizeof( ProcessParameters ) <= 4000;
#endif
  using ProcessParametersArg = std::conditional_t<processParametersByValue, ProcessParameters, const ProcessParameters*>;

  __device__ inline const ProcessParameters&
  deref( const ProcessParameters& params ) { return params; }
  __device__ inline const ProcessParameters&
  deref( const ProcessParameters* params ) { return *params; }

  // The copy of the parameters of a process on the current device, for the kernels
  // when they take them through a pointer, and the kernel argument made of them
  class DeviceProcessParameters
  {
  public:
    DeviceProcessParameters();
    ~DeviceProcessParameters();
    DeviceProcessParameters( const DeviceProcessParameters& ) = delete;
    DeviceProcessParameters& operator=( const DeviceProcessParameters& ) = delete;
    // Copies params to the device (when the kernels take them through a pointer)
    void update( const ProcessParameters& params );
    // The kernel argument for params, which update() must have been given
    ProcessParametersArg arg( const ProcessParameters& params ) const;
  private:
    ProcessParameters* m_device; // nullptr when the kernels take the parameters by value
  };

  __global__ void
  computeDependentCouplings( MGONGPU_GRID_CONSTANT const ProcessParametersArg params, // input: parameters of the process
                             const fptype* allgs,                                 // input: Gs[nevt]
                             fptype* allcouplings );                              // output: couplings[nevt*ndcoup*2]

  void
  sigmaKin_getGoodHel( const ProcessParametersArg params,  // input: parameters of the process
                       const fptype_momenta* allmomenta, // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,       // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec,   // input: index of the flavor combination
                       fptype* allMEs,                   // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype_amp* allNumerators,         // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype_amp* allDenominators,       // output: multichannel denominators[nevt], running_sum_over_helicities
                       fptype_amp_sv* allJamps,           // output: jamp[ncolor*2*nevt]
                       bool* isGoodHel,                  // output: isGoodHel[ncomb] - device array (GPU device implementation)
                       const int nevt );                 // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)

  int                                          // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( ProcessParameters& params, // output: the good helicities of the process
                       const bool* isGoodHel );  // input: isGoodHel[ncomb] - host array

  void
  sigmaKin( const ProcessParameters& params,     // input: parameters of the process (host)
            const ProcessParametersArg kparams,  // input: the same, as the kernels take them
            const fptype_momenta* allmomenta,   // input: momenta[nevt*npar*4]
            const fptype* allcouplings,         // input: couplings[nevt*ndcoup*2]
            const unsigned int* iflavorVec,     // input: index of the flavor combination
            const fptype* allrndhel,            // input: random numbers[nevt] for helicity selection
            const fptype* allrndcol,            // input: random numbers[nevt] for color selection
            const unsigned int* allChannelIds,  // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable single-diagram enhancement (fix #899/#911)
            const fptype* allrnddiagram,        // input: random numbers[nevt] for channel sampling
            fptype* allMEs,                     // output: allMEs[nevt], |M|^2 final_avg_over_helicities
            int* allselhel,                     // output: helicity selection[nevt]
            int* allselcol,                     // output: helicity selection[nevt]
            fptype_amp* colAllJamp2s,            // tmp: allJamp2s super-buffer for ncolor individual colors, running sum over colors and helicities
            fptype_amp* ghelAllNumerators,       // tmp: allNumerators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            fptype_amp* ghelAllDenominators,     // tmp: allDenominators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            unsigned int* allDiagramIdsOut,      // output: multichannel channelIds[nevt] (1 to #diagrams)
            bool mulChannelWeight,               // if true, multiply channel weight to ME output
            fptype* ghelAllMEs,                  // tmp: allMEs super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            fptype_amp* ghelAllJamps,            // tmp: allJamps super-buffer[2][njampso][nGoodHel][nevt] for nGoodHel <= ncomb individual helicities
            fptype_colour* ghelAllBlasTmp,       // tmp: allBlasTmp super-buffer for nGoodHel <= ncomb individual helicities
            gpuBlasHandle_t* pBlasHandle,        // input: cuBLAS/hipBLAS handle
            gpuStream_t* ghelStreams,            // input: cuda streams (index is ighel: only the first nGoodHel <= ncomb are non-null)
            const bool async,                    // input: if true, run everything asynchronously in first stream in ghelStreams
            const int gpublocks,                 // input: cuda gpublocks
            const int gputhreads,                // input: cuda gputhreads
            // input: when the |M|^2 keeps an interference (squared split orders, nampso > 1),
            // choose the helicity on |T_i| and return sign(T_i) * sum_j |T_j| as madevent does,
            // so that the chosen helicity is exact; false returns sum_j T_j (no effect otherwise)
            const bool sampleSignedHelicity = false,
            // input: the helicity index [0,ncomb) of each event (a device array), which is then
            // evaluated for that helicity alone (and returns its contribution to the helicity-summed
            // |M|^2, see umami_matrix_element); nullptr, the default, sums over the good helicities
            const int* allihel = nullptr );
}

#endif // SIGMAKIN_H
