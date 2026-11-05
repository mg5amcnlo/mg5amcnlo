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
  using namespace ColorMatrixData; // colorMatrix, colorDenom, ncolor, and the split-order nampso/njampso/sqSoIndex/chosenSqso


  //--------------------------------------------------------------------------

  // SQUARED SPLIT ORDERS. The color sum of a process whose '^2' constraint leaves
  // more than one amplitude split order (ProcessData::nampso > 1): the jamps arrive
  // as nampso vectors end to end (njampso = nampso*ncolor) and the color sum pairs
  // them, which is the Fortran GET_MATRIX contract JAMP(NCOLOR,NAMPSO) -> RES(NSQAMPSO).
  //
  // THE PAIR LOOP RUNS OVER ALL nampso*nampso ORDERED PAIRS, never a triangle.
  // The color contraction below keeps the triangular matrix whose off-diagonal
  // is doubled, which for a single jamp vector is just the statement that the
  // (i,j) and (j,i) terms are conjugates. For two DIFFERENT jamp vectors they
  // are not, so the value computed for one ordered pair (m,n) is NOT that pair's
  // contribution -- only (m,n) and (n,m) together are, and they are equal to the
  // true sum of the two. That is why both orderings must be summed, and why it
  // is safe to mask on the squared order: sqSoIndex is symmetric (a squared
  // order is the SUM of the two amplitude orders), so a pair and its transpose
  // are always kept or dropped together and the masked total stays real.
  [[maybe_unused]] static void
  color_sum_cpu_splitorders( fptype* allMEs,                  // output: allMEs[nevt], add |M|^2 for one specific helicity
                             const cxtype_amp_sv* allJamp_sv, // input: jamp_sv[njampso] (float/double) or jamp_sv[2*njampso] (mixed) for one specific helicity
                             const int ievt0 )                // input: first event number in current C++ event page
  {
    // Pre-compute a constexpr triangular color matrix properly normalized #475
    struct TriangularNormalizedColorMatrix
    {
      // See https://stackoverflow.com/a/34465458
      __host__ __device__ constexpr TriangularNormalizedColorMatrix()
        : value()
      {
        for( int icol = 0; icol < ncolor; icol++ )
        {
          // Diagonal terms
          value[icol][icol] = colorMatrix[icol][icol] / colorDenom[icol];
          // Off-diagonal terms
          for( int jcol = icol + 1; jcol < ncolor; jcol++ )
            value[icol][jcol] = 2 * colorMatrix[icol][jcol] / colorDenom[icol];
        }
      }
      fptype_colour value[ncolor][ncolor];
    };
    static constexpr auto cf2 = TriangularNormalizedColorMatrix();
    fptype_sv deltaMEs = { 0 };
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    fptype_sv deltaMEs_next = { 0 };
#endif
    // Gather the color flows the sum runs over, for each amplitude split order.
    // The order index strides by ncolor, exactly as calculate_jamps wrote them.
    // NB in mixed mode the two neppV vectors of allJamp_sv, at ijamp and at
    // njampso+ijamp, are two halves of the event page and not two colors.
    fptype_colour_sv jampR_sv[nampso][ncolor];
    fptype_colour_sv jampI_sv[nampso][ncolor];
    for( int iao = 0; iao < nampso; iao++ )
    {
      for( int icol = 0; icol < ncolor; icol++ )
      {
        const int ijamp = iao * ncolor + icol;
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        jampR_sv[iao][icol] = fpvmerge( cxreal( allJamp_sv[ijamp] ), cxreal( allJamp_sv[njampso + ijamp] ) );
        jampI_sv[iao][icol] = fpvmerge( cximag( allJamp_sv[ijamp] ), cximag( allJamp_sv[njampso + ijamp] ) );
#else
        jampR_sv[iao][icol] = (fptype_colour_sv)( cxreal( allJamp_sv[ijamp] ) );
        jampI_sv[iao][icol] = (fptype_colour_sv)( cximag( allJamp_sv[ijamp] ) );
#endif
      }
    }
    // Loop over ORDERED pairs of amplitude split orders (see the note above:
    // this must not be turned into a triangle over iao/jao)
    for( int iao = 0; iao < nampso; iao++ )
    {
      for( int jao = 0; jao < nampso; jao++ )
      {
        // The squared order this pair contributes to, dropped if the process
        // asked for a contribution that does not include it
        if( !chosenSqso[sqSoIndex[iao][jao]] ) continue;
        for( int icol = 0; icol < ncolor; icol++ )
        {
          // Diagonal terms (the color contraction is still triangular)
          fptype_colour_sv ztempR_sv = cf2.value[icol][icol] * jampR_sv[iao][icol];
          fptype_colour_sv ztempI_sv = cf2.value[icol][icol] * jampI_sv[iao][icol];
          for( int jcol = icol + 1; jcol < ncolor; jcol++ )
          {
            // Off-diagonal terms
            ztempR_sv += cf2.value[icol][jcol] * jampR_sv[iao][jcol];
            ztempI_sv += cf2.value[icol][jcol] * jampI_sv[iao][jcol];
          }
          // ztemp comes from the jamps of order iao, and is contracted with those
          // of order jao (Fortran: ZTEMP from JAMP(:,M), times DCONJG(JAMP(I,N)))
          fptype_colour_sv deltaMEs2 = ( jampR_sv[jao][icol] * ztempR_sv + jampI_sv[jao][icol] * ztempI_sv ); // may underflow #831
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          deltaMEs += fpvsplit0( deltaMEs2 );
          deltaMEs_next += fpvsplit1( deltaMEs2 );
#else
          deltaMEs += deltaMEs2;
#endif
        }
      }
    }
    // *** STORE THE RESULTS ***
    using E_ACCESS = HostAccessMatrixElements; // non-trivial access: buffer includes all events
    fptype* MEs = E_ACCESS::ieventAccessRecord( allMEs, ievt0 );
    // NB: color_sum ADDS |M|^2 for one helicity to the running sum of |M|^2 over helicities for the given event(s)
    fptype_sv& MEs_sv = E_ACCESS::kernelAccess( MEs );
    MEs_sv += deltaMEs; // fix #435
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    fptype* MEs_next = E_ACCESS::ieventAccessRecord( allMEs, ievt0 + neppV );
    fptype_sv& MEs_sv_next = E_ACCESS::kernelAccess( MEs_next );
    MEs_sv_next += deltaMEs_next;
#endif
  }

  //--------------------------------------------------------------------------

  void
  color_sum_cpu( fptype* allMEs,              // output: allMEs[nevt], add |M|^2 for one specific helicity
                 const cxtype_amp_sv* allJamp_sv, // input: jamp_sv[njampso] (float/double) or jamp_sv[2*njampso] (mixed) for one specific helicity
                 const int ievt0 )            // input: first event number in current C++ event page (for CUDA, ievt depends on threadid)
  {
    if constexpr( nampso > 1 )
    {
      // The jamps carry an amplitude-order index: pair them instead (see above)
      color_sum_cpu_splitorders( allMEs, allJamp_sv, ievt0 );
      return;
    }
    // Pre-compute a constexpr triangular color matrix properly normalized #475
    struct TriangularNormalizedColorMatrix
    {
      // See https://stackoverflow.com/a/34465458
      __host__ __device__ constexpr TriangularNormalizedColorMatrix()
        : value()
      {
        for( int icol = 0; icol < ncolor; icol++ )
        {
          // Diagonal terms
          value[icol][icol] = colorMatrix[icol][icol] / colorDenom[icol];
          // Off-diagonal terms
          for( int jcol = icol + 1; jcol < ncolor; jcol++ )
            value[icol][jcol] = 2 * colorMatrix[icol][jcol] / colorDenom[icol];
        }
      }
      fptype_colour value[ncolor][ncolor];
    };
    static constexpr auto cf2 = TriangularNormalizedColorMatrix();
    // Use the property that M is a real matrix (see #475):
    // we can rewrite the quadratic form (A-iB)(M)(A+iB) as AMA - iBMA + iBMA + BMB = AMA + BMB
    // In addition, on C++ use the property that M is symmetric (see #475),
    // and also use constexpr to compute "2*" and "/colorDenom[icol]" once and for all at compile time:
    // we gain (not a factor 2...) in speed here as we only loop over the up diagonal part of the matrix.
    // Strangely, CUDA is slower instead, so keep the old implementation for the moment.
    fptype_sv deltaMEs = { 0 };
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    fptype_sv deltaMEs_next = { 0 };
#endif
    // NB in mixed mode the two neppV vectors of allJamp_sv, at icol and at ncolor+icol, are
    // two halves of the event page and not two colors: it is the color index inside each of
    // them which is gathered, and the two are merged into one neppV2 vector.
    fptype_colour_sv jampR_sv[ncolor];
    fptype_colour_sv jampI_sv[ncolor];
    for( int icol = 0; icol < ncolor; icol++ )
    {
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      jampR_sv[icol] = fpvmerge( cxreal( allJamp_sv[icol] ), cxreal( allJamp_sv[ncolor + icol] ) );
      jampI_sv[icol] = fpvmerge( cximag( allJamp_sv[icol] ), cximag( allJamp_sv[ncolor + icol] ) );
#else
      jampR_sv[icol] = (fptype_colour_sv)( cxreal( allJamp_sv[icol] ) );
      jampI_sv[icol] = (fptype_colour_sv)( cximag( allJamp_sv[icol] ) );
#endif
    }
    for( int icol = 0; icol < ncolor; icol++ )
    {
      // Diagonal terms
      fptype_colour_sv& jampRi_sv = jampR_sv[icol];
      fptype_colour_sv& jampIi_sv = jampI_sv[icol];
      fptype_colour_sv ztempR_sv = cf2.value[icol][icol] * jampRi_sv;
      fptype_colour_sv ztempI_sv = cf2.value[icol][icol] * jampIi_sv;
      for( int jcol = icol + 1; jcol < ncolor; jcol++ )
      {
        // Off-diagonal terms
        fptype_colour_sv& jampRj_sv = jampR_sv[jcol];
        fptype_colour_sv& jampIj_sv = jampI_sv[jcol];
        ztempR_sv += cf2.value[icol][jcol] * jampRj_sv;
        ztempI_sv += cf2.value[icol][jcol] * jampIj_sv;
      }
      fptype_colour_sv deltaMEs2 = ( jampRi_sv * ztempR_sv + jampIi_sv * ztempI_sv ); // may underflow #831
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      deltaMEs += fpvsplit0( deltaMEs2 );
      deltaMEs_next += fpvsplit1( deltaMEs2 );
#else
      deltaMEs += deltaMEs2;
#endif
    }
    // *** STORE THE RESULTS ***
    using E_ACCESS = HostAccessMatrixElements; // non-trivial access: buffer includes all events
    fptype* MEs = E_ACCESS::ieventAccessRecord( allMEs, ievt0 );
    // NB: color_sum ADDS |M|^2 for one helicity to the running sum of |M|^2 over helicities for the given event(s)
    fptype_sv& MEs_sv = E_ACCESS::kernelAccess( MEs );
    MEs_sv += deltaMEs; // fix #435
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    fptype* MEs_next = E_ACCESS::ieventAccessRecord( allMEs, ievt0 + neppV );
    fptype_sv& MEs_sv_next = E_ACCESS::kernelAccess( MEs_next );
    MEs_sv_next += deltaMEs_next;
#endif
  }

  //--------------------------------------------------------------------------

#ifndef MGONGPUCPP_GPUIMPL
#ifdef MGONGPU_CPP_HAS_BLAS

#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
  constexpr int nParityCS = 2; // mixed mode merges two neppV pages into one call
#else
  constexpr int nParityCS = 1;
#endif

  // The color matrix does not depend on the helicity, so the jamps of every
  // good helicity (and of every event in the SIMD page) are columns of one
  // right hand side and the whole color sum becomes two SYMM calls. SYMM is
  // real, so the real and the imaginary part of JAMP go through separately,
  // which is the same property of M being real that the scalar sum uses to
  // rewrite (A-iB)M(A+iB) as AMA + BMB (see #475).
  //
  // The scalar sum walks the upper triangle with its off diagonal doubled;
  // SYMM wants the whole symmetric matrix with each entry counted once. What
  // is written out is normalized row by row by colorDenom[icol], which is not
  // symmetric when the denominators differ; only the symmetric part of a
  // matrix contributes to a quadratic form, so take it explicitly. When all
  // the denominators agree (the usual case) this is exactly colorMatrix/denom.
  struct SymmetricNormalizedColorMatrix
  {
    constexpr SymmetricNormalizedColorMatrix()
      : value()
    {
      for( int icol = 0; icol < ncolor; icol++ )
        for( int jcol = 0; jcol < ncolor; jcol++ )
          value[icol * ncolor + jcol] =
            ( colorMatrix[icol][jcol] / colorDenom[icol] + colorMatrix[jcol][icol] / colorDenom[jcol] ) / 2;
    }
    fptype2 value[ncolor * ncolor];
  };

  // The Fortran BLAS interface, which every implementation exports (the
  // reference BLAS ships no CBLAS of its own). Column major, as in the
  // Fortran color sum.
  extern "C"
  {
    void dsymm_( const char* side, const char* uplo, const int* m, const int* n,
                 const double* alpha, const double* a, const int* lda,
                 const double* b, const int* ldb,
                 const double* beta, double* c, const int* ldc );
    void ssymm_( const char* side, const char* uplo, const int* m, const int* n,
                 const float* alpha, const float* a, const int* lda,
                 const float* b, const int* ldb,
                 const float* beta, float* c, const int* ldc );
  }

  [[maybe_unused]] static inline void
  blas_symm( const int m, const int n, const double* a, const double* b, double* c )
  {
    const char side = 'L', uplo = 'U';
    const double alpha = 1, beta = 0;
    dsymm_( &side, &uplo, &m, &n, &alpha, a, &m, b, &m, &beta, c, &m );
  }

  [[maybe_unused]] static inline void
  blas_symm( const int m, const int n, const float* a, const float* b, float* c )
  {
    const char side = 'L', uplo = 'U';
    const float alpha = 1, beta = 0;
    ssymm_( &side, &uplo, &m, &n, &alpha, a, &m, b, &m, &beta, c, &m );
  }

  void
  color_sum_cpu_blas( fptype* allMEs,                  // input/output: allMEs[nevt], add |M|^2 summed over all good helicities
                      fptype_sv* MEs_ighel,            // output: [ncomb] running sum of |M|^2 up to ighel (first - and/or only - neppV page)
                      fptype_sv* MEs_ighel2,           // output: [ncomb] the same for the second neppV page (mixed mode only)
                      const cxtype_sv* ghelAllJamp_sv, // input: jamp_sv[nGoodHel][nParity*ncolor] for all good helicities
                      const int nGoodHel,              // input: number of good helicities
                      const int ievt0 )                // input: first event number in current C++ event page
  {
    static constexpr auto cfsym = SymmetricNormalizedColorMatrix();
    constexpr int nevtB = nParityCS * neppV; // events covered by one call
    const int ncol = nGoodHel * nevtB;       // number of BLAS right hand side columns
    // Column major scratch: JR/JI hold the ncolor x ncol jamps, ZR/ZI take
    // the SYMM results and MEcol one |M|^2 per column. Kept on the heap and grown
    // once per thread: for ncolor=60 and ncomb=128 this is a few hundred kB.
    static thread_local std::vector<fptype2> scratch;
    const size_t need = 4 * (size_t)ncolor * ncol + ncol;
    if( scratch.size() < need ) scratch.resize( need );
    fptype2* JR = scratch.data();
    fptype2* JI = JR + (size_t)ncolor * ncol;
    fptype2* ZR = JI + (size_t)ncolor * ncol;
    fptype2* ZI = ZR + (size_t)ncolor * ncol;
    fptype2* MEcol = ZI + (size_t)ncolor * ncol;
    // Transpose the jamps into the column major right hand side: colour is the
    // fast index, (helicity, event) the slow one.
    for( int ighel = 0; ighel < nGoodHel; ighel++ )
    {
      const cxtype_sv* jamp_sv = ghelAllJamp_sv + (size_t)ighel * nParityCS * ncolor;
      for( int ip = 0; ip < nParityCS; ip++ )
        for( int ieppV = 0; ieppV < neppV; ieppV++ )
        {
          const size_t off = (size_t)( ighel * nevtB + ip * neppV + ieppV ) * ncolor;
          for( int icol = 0; icol < ncolor; icol++ )
          {
#ifdef MGONGPU_CPPSIMD
            JR[off + icol] = cxreal( jamp_sv[ip * ncolor + icol] )[ieppV];
            JI[off + icol] = cximag( jamp_sv[ip * ncolor + icol] )[ieppV];
#else
            JR[off + icol] = cxreal( jamp_sv[ip * ncolor + icol] );
            JI[off + icol] = cximag( jamp_sv[ip * ncolor + icol] );
#endif
          }
        }
    }
    // Ztemp[ncolor][ncol] = ColorMatrix[ncolor][ncolor] * Jamps[ncolor][ncol], real and imaginary parts apart
    blas_symm( ncolor, ncol, cfsym.value, JR, ZR );
    blas_symm( ncolor, ncol, cfsym.value, JI, ZI );
    // |M|^2 for one (helicity, event) is the dot product of one column of Jamps with one column of Ztemp
    for( int j = 0; j < ncol; j++ )
    {
      const size_t off = (size_t)j * ncolor;
      fptype2 me = 0;
      for( int icol = 0; icol < ncolor; icol++ )
        me += JR[off + icol] * ZR[off + icol] + JI[off + icol] * ZI[off + icol];
      MEcol[j] = me; // may underflow #831
    }
    // *** STORE THE RESULTS ***
    // NB: MEs_ighel carries the running sum over helicities of |M|^2, which the
    // event by event choice of helicity needs. The color sum is no longer added
    // to allMEs one helicity at a time, so build those running sums here,
    // starting from whatever allMEs already held (fix #435).
    using E_ACCESS = HostAccessMatrixElements; // non-trivial access: buffer includes all events
    for( int ip = 0; ip < nParityCS; ip++ )
    {
      fptype_sv* running = ( ip == 0 ? MEs_ighel : MEs_ighel2 );
      fptype* MEsp = E_ACCESS::ieventAccessRecord( allMEs, ievt0 + ip * neppV );
      fptype_sv& MEsp_sv = E_ACCESS::kernelAccess( MEsp );
      for( int ieppV = 0; ieppV < neppV; ieppV++ )
      {
#ifdef MGONGPU_CPPSIMD
        fptype sum = MEsp_sv[ieppV];
        for( int ighel = 0; ighel < nGoodHel; ighel++ )
        {
          sum += MEcol[ighel * nevtB + ip * neppV + ieppV];
          running[ighel][ieppV] = sum;
        }
#else
        fptype sum = MEsp_sv;
        for( int ighel = 0; ighel < nGoodHel; ighel++ )
        {
          sum += MEcol[ighel * nevtB + ip * neppV + ieppV];
          running[ighel] = sum;
        }
#endif
      }
      MEsp_sv = running[nGoodHel - 1];
    }
  }
#endif
#endif

  //--------------------------------------------------------------------------

} // end namespace
