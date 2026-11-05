// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Sep 2025) for the MadGraph7 CUDACPP plugin.
// Further modified by: A. Valassi (2025).
// Integrated with the MadGraph7 project in Feb 2026.

#include "color_sum.h"

#include "mgOnGpuConfig.h"

#include "ColorData.h" // P1-generated: colorMatrix/colorDenom
#include "MemoryAccessMatrixElements.h"
#include "mgOnGpuCxtypes.h"

namespace madmatrix
{
  using namespace ColorMatrixData; // colorMatrix, colorDenom, ncolor

  //--------------------------------------------------------------------------

  // The normalized color matrix (divide each column by denom)
  template<typename T>
  struct NormalizedColorMatrix
  {
    constexpr __host__ __device__ NormalizedColorMatrix()
      : value()
    {
      for( int icol = 0; icol < ncolor; icol++ )
        for( int jcol = 0; jcol < ncolor; jcol++ )
          value[icol * ncolor + jcol] = colorMatrix[icol][jcol] / colorDenom[icol];
    }
    T value[ncolor * ncolor];
  };
  // The fptype_colour version is the default used by kernels (supporting mixed floating point mode).
  // Statically initialized: every device has it as soon as the module is loaded.
  static __device__ const NormalizedColorMatrix<fptype_colour> s_normalizedColorMatrix2{};

  //--------------------------------------------------------------------------

  //--------------------------------------------------------------------------

  __global__ void
  color_sum_kernel( fptype* allMEs,                 // output: allMEs[nevt], add |M|^2 for one specific helicity
                    const fptype* allJamps,         // input: jamp[ncolor*2*nevt] for one specific helicity
                    const int nGoodHel,             // input: number of good helicities
                    const int nevtIfAllHelicities ) // input: zero in single-helicity mode, number of events in multi-helicity mode
  {
    if (nevtIfAllHelicities) {
      int ighel = blockIdx.y;
      allMEs = allMEs + ighel * nevtIfAllHelicities; // MEs for one specific helicity ighel
      allJamps = allJamps + ighel * nevtIfAllHelicities; // Jamps for one specific helicity ighel
    }
    using J_ACCESS = DeviceAccessJamp;
    using E_ACCESS = DeviceAccessMatrixElements; // non-trivial access: buffer includes all events
    constexpr int ihel0 = 0; // the input buffer allJamps already points to a specific helicity
    fptype deltaMEs = { 0 };
    if constexpr( nampso > 1 )
    {
      // Squared split orders: the jamps carry an amplitude-order index (njampso =
      // ncolor * nampso, the order striding by ncolor, as calculate_jamps wrote them)
      // and are paired here, as color_sum_cpu_splitorders in backend/{cpu,simd} does.
      // The color contraction stays triangular, but the loop over amplitude-order
      // pairs must run over ALL ordered pairs: for two different jamp vectors a pair
      // and its transpose are not each other's conjugate, only their sum is the
      // truth (a triangle gives about twice the answer). Masking stays safe because
      // sqSoIndex is symmetric: a pair and its transpose are kept or dropped together.
      // (in the color algebra precision, as below and as backend/{cpu,simd} do)
      fptype_colour jampR[nampso][ncolor];
      fptype_colour jampI[nampso][ncolor];
      for( int iao = 0; iao < nampso; iao++ )
      {
        for( int icol = 0; icol < ncolor; icol++ )
        {
          cxtype_amp jamp = J_ACCESS::kernelAccessIcolIhelNhelConst( allJamps, iao * ncolor + icol, ihel0, nGoodHel );
          jampR[iao][icol] = jamp.real();
          jampI[iao][icol] = jamp.imag();
        }
      }
      for( int iao = 0; iao < nampso; iao++ )
      {
        for( int jao = 0; jao < nampso; jao++ )
        {
          // the squared order this pair contributes to, dropped if the process asked
          // for a contribution that does not include it
          if( !chosenSqso[sqSoIndex[iao][jao]] ) continue;
          for( int icol = 0; icol < ncolor; icol++ )
          {
            // ztemp from the jamps of order iao, contracted with those of order jao
            // (Fortran: ZTEMP from JAMP(:,M), times DCONJG(JAMP(I,N)))
            fptype_colour ztempR = s_normalizedColorMatrix2.value[icol * ncolor + icol] * jampR[iao][icol];
            fptype_colour ztempI = s_normalizedColorMatrix2.value[icol * ncolor + icol] * jampI[iao][icol];
            for( int jcol = 0; jcol < icol; jcol++ )
            {
              ztempR += 2 * s_normalizedColorMatrix2.value[icol * ncolor + jcol] * jampR[iao][jcol];
              ztempI += 2 * s_normalizedColorMatrix2.value[icol * ncolor + jcol] * jampI[iao][jcol];
            }
            deltaMEs += ztempR * jampR[jao][icol];
            deltaMEs += ztempI * jampI[jao][icol];
          }
        }
      }
    }
    else // one amplitude split order (an else, not a return: no unreachable code for nvcc)
    {
      fptype_amp jampR[ncolor];
      fptype_amp jampI[ncolor];
      for( int icol = 0; icol < ncolor; icol++ )
      {
        cxtype_amp jamp = J_ACCESS::kernelAccessIcolIhelNhelConst( allJamps, icol, ihel0, nGoodHel );
        jampR[icol] = jamp.real();
        jampI[icol] = jamp.imag();
      }
      for( int icol = 0; icol < ncolor; icol++ )
      {
        fptype_colour ztempR = { 0 };
        fptype_colour ztempI = { 0 };
        fptype_colour jampRi = jampR[icol];
        fptype_colour jampIi = jampI[icol];
        // OLD IMPLEMENTATION (ihel3: symmetric square matrix) - Loop over all jcol
        //for( int jcol = 0; jcol < ncolor; jcol++ )
        //{
        //  fptype_colour jampRj = jampR[jcol];
        //  fptype_colour jampIj = jampI[jcol];
        //  ztempR += s_normalizedColorMatrix2.value[icol * ncolor + jcol] * jampRj; // use fptype_colour version of color matrix
        //  ztempI += s_normalizedColorMatrix2.value[icol * ncolor + jcol] * jampIj; // use fptype_colour version of color matrix
        //}
        // NEW IMPLEMENTATION #475 (ihel3p1: triangular lower diagonal matrix) - Loop over jcol < icol
        ztempR += s_normalizedColorMatrix2.value[icol * ncolor + icol] * jampRi; // use fptype_colour version of color matrix
        ztempI += s_normalizedColorMatrix2.value[icol * ncolor + icol] * jampIi; // use fptype_colour version of color matrix
        for( int jcol = 0; jcol < icol; jcol++ )
        {
          fptype_colour jampRj = jampR[jcol];
          fptype_colour jampIj = jampI[jcol];
          ztempR += 2 * s_normalizedColorMatrix2.value[icol * ncolor + jcol] * jampRj; // use fptype_colour version of color matrix
          ztempI += 2 * s_normalizedColorMatrix2.value[icol * ncolor + jcol] * jampIj; // use fptype_colour version of color matrix
        }
        deltaMEs += ztempR * jampRi;
        deltaMEs += ztempI * jampIi;
      }
    }
    // *** STORE THE RESULTS ***
    // NB: color_sum ADDS |M|^2 for one helicity to the running sum of |M|^2 over helicities for the given event(s)
    // (once, after the if constexpr: on HIP, a template first named in the discarded branch
    // was never instantiated for the device, "undefined hidden symbol ... kernelAccessField")
    E_ACCESS::kernelAccess( allMEs ) += deltaMEs; // fix #435
  }

  //--------------------------------------------------------------------------

#ifndef MGONGPU_HAS_NO_BLAS
  // Convert the jamps from fptype to fptype2, which is only ever double to float in mixed
  // floating point mode: the color sum reads them where compute_jamps wrote them otherwise.
  __global__ void
  convertD2F_Jamps( fptype_colour* allJamps2, // output: jamp[2][ncolor][ihel][nevt] for one specific helicity ihel
                    const fptype_amp* allJamps, // input: jamp[2][ncolor][ihel][nevt] for one specific helicity ihel
                    const int nhel )        // input: number of good helicities nGoodHel
  {
    const int nevt = gridDim.x * blockDim.x;
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int ihel = 0; // the input buffer allJamps already points to a specific helicity
    // NB! The same striding as in compute_jamps and cuBLAS is used here
    for( int ix2 = 0; ix2 < mgOnGpu::nx2; ix2++ )
      for( int icol = 0; icol < ncolor; icol++ )
        allJamps2[ix2 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt] =
          allJamps[ix2 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt];
  }

  // Convert the jamps of every good helicity into ghelAllJampsBuf, and return it
  fptype2*
  convertD2F_AllJamps( fptype2* ghelAllJampsBuf,    // output: allJamps super-buffer[2][ncolor][nhel][nevt]
                       const fptype* ghelAllJamps,  // input: allJamps super-buffer[2][ncolor][nhel][nevt]
                       gpuStream_t* ghelStreams,    // input: cuda streams (index is ighel)
                       const int nhel,              // input: number of good helicities
                       const int gpublocks,         // input: cuda gpublocks
                       const int gputhreads )       // input: cuda gputhreads
  {
    const int nevt = gpublocks * gputhreads;
    for( int ighel = 0; ighel < nhel; ighel++ )
    {
      const fptype* hAllJamps = ghelAllJamps + ighel * nevt;         // jamps for a single helicity ihel
      fptype2* hAllJamps2 = ghelAllJampsBuf + ighel * nevt;          // converted jamps for a single helicity ihel
      gpuLaunchKernelStream( convertD2F_Jamps, gpublocks, gputhreads, ghelStreams[ighel], hAllJamps2, hAllJamps, nhel );
    }
    return ghelAllJampsBuf;
  }
#endif

  //--------------------------------------------------------------------------

#ifndef MGONGPU_HAS_NO_BLAS
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
  __global__ void
  convertF2D_MEs( fptype* allMEs,             // output: allMEs[nevt] for one specific helicity
                  const fptype_colour* allMEsFpt2 ) // input: allMEs[nevt] for one specific helicity
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
    allMEs[ievt] = allMEsFpt2[ievt];
  }
#endif
#endif

  //--------------------------------------------------------------------------

#ifndef MGONGPU_HAS_NO_BLAS
  void
  color_sum_blas( fptype* ghelAllMEs,           // output: allMEs super-buffer[nhel][nevt], add |M|^2 separately for each helicity
                  const fptype_amp* ghelAllJamps,   // input: allJamps super-buffer[2][ncol][nhel][nevt] for nhel good helicities
                  fptype_colour* ghelAllBlasTmp,      // tmp: allBlasTmp super-buffer for nhel good helicities
                  gpuBlasHandle_t* pBlasHandle, // input: cuBLAS/hipBLAS handle
                  gpuStream_t* ghelStreams,     // input: cuda streams (index is ighel: only the first nhel <= ncomb are non-null)
                  const int nhel,               // input: number of good helicities (nhel == nGoodHel)
                  const int gpublocks,          // input: cuda gpublocks
                  const int gputhreads )        // input: cuda gputhreads
  {
    const int nevt = gpublocks * gputhreads;

    // Get the address associated with the normalized color matrix in the memory of the
    // current device (each device has its own: not to be kept from one call to the next)
    fptype_colour* devNormColMat = nullptr;
    gpuGetSymbolAddress( (void**)&devNormColMat, s_normalizedColorMatrix2 );

    // The scratch buffer holds the BLAS intermediate results and, in mixed precision mode,
    // the converted jamps and the fptype2 MEs: see the layout in blasColorSumTmpSize, which
    // is what MatrixElementKernels.cc allocates.
    fptype2* ghelAllZtempBoth = ghelAllBlasTmp;                                        // start of the fptype2[ncolor*2*nhel*nevt] buffer
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    // Mixed precision mode: the jamps go through a buffer of their own, as they must be
    // converted from double to float on the way
    fptype_colour* ghelAllJampsBuf = ghelAllBlasTmp + ncolor * mgOnGpu::nx2 * nhel * nevt;    // start of the second one
    fptype_colour* ghelAllMEsFpt2 = ghelAllBlasTmp + 2 * ncolor * mgOnGpu::nx2 * nhel * nevt; // start of the fptype_colour[nhel*nevt] buffer
    const fptype_colour* ghelAllJamps2 = convertD2F_AllJamps( ghelAllJampsBuf, ghelAllJamps, ghelStreams, nhel, gpublocks, gputhreads );
#else
    static_assert( std::is_same<fptype_colour, fptype>::value );
    fptype_colour* ghelAllMEsFpt2 = ghelAllMEs;
    // Nothing to convert: read the jamps where compute_jamps wrote them (this is not a
    // cast, the two types are identical)
    const fptype_colour* ghelAllJamps2 = ghelAllJamps;
#endif
    // Real and imaginary components
    const fptype_colour* ghelAllJampsReal = ghelAllJamps2;
    const fptype_colour* ghelAllJampsImag = ghelAllJamps2 + ncolor * nhel * nevt;
    fptype_colour* ghelAllZtempReal = ghelAllZtempBoth;
    fptype_colour* ghelAllZtempImag = ghelAllZtempBoth + ncolor * nhel * nevt;

    // Note: striding for cuBLAS (that of DeviceAccessJamp, over ncolor):
    // - ghelAllJamps2(icol,ihel,ievt).real is ghelAllJamps2[0 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt]
    // - ghelAllJamps2(icol,ihel,ievt).imag is ghelAllJamps2[1 * ncolor * nhel * nevt + icol * nhel * nevt + ihel * nevt + ievt]

    // Step 1: Compute Ztemp[ncolor][nhel*nevt] = ColorMatrix[ncolor][ncolor] * JampsVector[ncolor][nhel*nevt] for both real and imag
    // In this case alpha=1 and beta=0: the operation is Ztemp = alpha * ColorMatrix * JampsVector + beta * Ztemp
    fptype_colour alpha1 = 1;
    fptype_colour beta1 = 0;
    const int ncolorM = ncolor;
    const int nevtN = nhel*nevt;
    const int ncolorK = ncolor;
    checkGpuBlas( gpuBlasTgemm( *pBlasHandle,
                                GPUBLAS_OP_N,                  // do not transpose ColMat
                                GPUBLAS_OP_T,                  // transpose JampsV (new1)
                                ncolorM, nevtN, ncolorK,
                                &alpha1,
                                devNormColMat, ncolorM,        // ColMat is ncolorM x ncolorK
                                ghelAllJampsReal, nevtN,       // JampsV is nevtN x ncolorK
                                &beta1,
                                ghelAllZtempReal, ncolorM ) ); // Ztemp is ncolorM x nevtN
    checkGpuBlas( gpuBlasTgemm( *pBlasHandle,
                                GPUBLAS_OP_N,                  // do not transpose ColMat
                                GPUBLAS_OP_T,                  // transpose JampsV (new1)
                                ncolorM, nevtN, ncolorK,
                                &alpha1,
                                devNormColMat, ncolorM,        // ColMat is ncolorM x ncolorK
                                ghelAllJampsImag, nevtN,       // JampsV is nevtN x ncolorK (new1)
                                &beta1,
                                ghelAllZtempImag, ncolorM ) ); // Ztemp is ncolorM x nevtN

    // Step 2: For each ievt, compute the dot product of JampsVector[ncolor][ievt] dot tmp[ncolor][ievt]
    // In this case alpha=1 and beta=1: the operation is ME = alpha * ( Tmp dot JampsVector ) + beta * ME
    // Use cublasSgemmStridedBatched to perform these batched dot products in one call
    fptype_colour alpha2 = 1;
    fptype_colour beta2 = 1;
    checkGpuBlas( gpuBlasTgemmStridedBatched( *pBlasHandle,
                                              GPUBLAS_OP_N,                             // do not transpose JampsV (new1)
                                              GPUBLAS_OP_N,                             // do not transpose Tmp
                                              1, 1, ncolor,                         // result is 1x1 (dot product)
                                              &alpha2,
                                              ghelAllJampsReal, nevtN, 1,               // allJamps is nevtN x ncolor, stride 1 for each ievt column
                                              ghelAllZtempReal, ncolor, ncolor, // allZtemp is ncolor x nevtN, with stride ncolor for each ievt column
                                              &beta2,
                                              ghelAllMEsFpt2, 1, 1,                     // output is a 1x1 result for each "batch" (i.e. for each ievt)
                                              nevtN ) );                                // there are nevtN (nhel*nevt) "batches"
    checkGpuBlas( gpuBlasTgemmStridedBatched( *pBlasHandle,
                                              GPUBLAS_OP_N,                             // do not transpose JampsV (new1)
                                              GPUBLAS_OP_N,                             // do not transpose Tmp
                                              1, 1, ncolor,                         // result is 1x1 (dot product)
                                              &alpha2,
                                              ghelAllJampsImag, nevtN, 1,               // allJamps is nevtN x ncolor, stride 1 for each ievt column (new1)
                                              ghelAllZtempImag, ncolor, ncolor, // allZtemp is ncolor x nevtN, with stride ncolor for each ievt column
                                              &beta2,
                                              ghelAllMEsFpt2, 1, 1,                     // output is a 1x1 result for each "batch" (i.e. for each ievt)
                                              nevtN ) );                                // there are nevt (nhel*nevt) "batches"

#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    // Convert MEs from float to double
    for( int ighel = 0; ighel < nhel; ighel++ )
    {
      fptype* hAllMEs = ghelAllMEs + ighel * nevt;          // MEs for a single helicity ihel
      fptype_colour* hAllMEsFpt2 = ghelAllMEsFpt2 + ighel * nevt; // MEs for a single helicity ihel      
      gpuLaunchKernelStream( convertF2D_MEs, gpublocks, gputhreads, ghelStreams[ighel], hAllMEs, hAllMEsFpt2 );
    }
#endif
  }
#endif /* clang-format on */

  //--------------------------------------------------------------------------

  void
  color_sum_gpu( fptype* ghelAllMEs,               // output: allMEs super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
                 const fptype_amp* ghelAllJamps,       // input: allJamps super-buffer[2][ncol][nGoodHel][nevt] for nGoodHel <= ncomb individual helicities
                 fptype_colour* ghelAllBlasTmp,          // tmp: allBlasTmp super-buffer for nGoodHel <= ncomb individual helicities
                 gpuBlasHandle_t* pBlasHandle,     // input: cuBLAS/hipBLAS handle
                 gpuStream_t* ghelStreams,         // input: cuda streams (index is ighel: only the first nGoodHel <= ncomb are non-null)
                 const int nGoodHel,               // input: number of good helicities
                 const int gpublocks,              // input: cuda gpublocks
                 const int gputhreads,             // input: cuda gputhreads
                 const bool processAllHelicities ) // input: if true, use blockIdx.y to index helicities
  {
    const int nevt = gpublocks * gputhreads;
    // CASE 1: KERNEL
    if( !pBlasHandle )
    {
      assert( ghelAllBlasTmp == nullptr );  // sanity check for HASBLAS=hasNoBlas or CUDACPP_RUNTIME_BLASCOLORSUM not set
      if (processAllHelicities) {
        gpuLaunchKernel2D( color_sum_kernel, gpublocks, nGoodHel, gputhreads, ghelStreams[0], ghelAllMEs, ghelAllJamps, nGoodHel, nevt );
      } else {
        // Loop over helicities
        for( int ighel = 0; ighel < nGoodHel; ighel++ )
        {
          fptype* hAllMEs = ghelAllMEs + ighel * nevt;           // MEs for one specific helicity ighel
          const fptype_amp* hAllJamps = ghelAllJamps + ighel * nevt; // Jamps for one specific helicity ighel
          gpuStream_t hStream = ghelStreams[ighel];
          gpuLaunchKernelStream( color_sum_kernel, gpublocks, gputhreads, hStream, hAllMEs, hAllJamps, nGoodHel, 0 );
        }
      }
    }
    // CASE 2: BLAS
    else
    {
#ifdef MGONGPU_HAS_NO_BLAS
      assert( false ); // sanity check: no path to this statement for HASBLAS=hasNoBlas
#else
      // sanity check: MatrixElementKernelDevice never enables BLAS for split amplitude
      // orders, whose jamps the BLAS color sum cannot pair
      assert( nampso == 1 );
      if (processAllHelicities) {
        assert( false ); // BLAS in async mode not supported for now
      } else {
        checkGpu( gpuDeviceSynchronize() ); // do not start the BLAS color sum for all helicities until the loop over helicities has completed
        // Reset the tmp buffer (same size as the one MatrixElementKernelDevice allocated)
        gpuMemset( ghelAllBlasTmp, 0, blasColorSumTmpSize( nGoodHel, nevt ) * sizeof( fptype_colour ) );
        // Delegate the color sum to BLAS for 
        color_sum_blas( ghelAllMEs, ghelAllJamps, ghelAllBlasTmp, pBlasHandle, ghelStreams, nGoodHel, gpublocks, gputhreads );
      }
#endif
    }
  }

  //--------------------------------------------------------------------------

} // end namespace
