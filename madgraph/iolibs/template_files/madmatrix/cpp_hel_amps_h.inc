// Copyright (C) 2010 The ALOHA Development team and Contributors.
// Copyright (C) 2010 The MadGraph7 development team and contributors.
// Created by: J. Alwall (Sep 2010) for the MadGraph7 backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: A. Valassi (Sep 2021) for the MadGraph7 CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024) for the MadGraph7 CUDACPP plugin.
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
// This file has been automatically generated for %(output_name)s by
%(info_lines)s
//==========================================================================

#ifndef HelAmps_%(model_name)s_H
#define HelAmps_%(model_name)s_H 1

#include "mgOnGpuConfig.h"

#ifdef MGONGPU_HAS_VECTORS_H // the simd backend (see its mgOnGpuConfig.h)
#include "mgOnGpuVectors.h"
#else
#include "mgOnGpuCxtypes.h"
#endif
#include "MadMatrixDenom.h"

#include "Parameters.h"

#include <cassert>
//#include <cmath>
//#include <cstdlib>
//#include <iomanip>
//#include <iostream>

namespace madmatrix
{

  // ALOHA-style object for easy flavor consolidation and non-template API

  struct ALOHAOBJ {

      static constexpr int np4 = 4; // dimensions of 4-momenta (E,px,py,pz)
      static constexpr int nw6 = %(nwave)d; // dimensions of each wavefunction (notice, this is +1 in case of FD gauge)
      fptype_momenta_sv * pvec;
      fptype_amp * w;
      int flv_index;

      __host__ __device__ ALOHAOBJ() {}
      __host__ __device__ ALOHAOBJ(fptype_momenta_sv * pvec_sv, cxtype_amp_sv * w_sv, int flv = -1)
          : pvec(pvec_sv), w(reinterpret_cast<fptype_amp*>(w_sv)), flv_index(flv) {}
  };

  // Sum two currents standing for the same off shell line: the four gluon
  // current and the pair of three gluon vertices it factorises into carry the
  // same colour factor, so the amplitude reading the sum gets both
  // contributions from a single call. See
  // HelasMatrixElement.get_quartic_current_sums. The two share their momentum,
  // so only the wavefunction is added and the rest is taken from the first.
  template<class W_ACCESS>
  __device__ inline void
  SUMW_1( const ALOHAOBJ& V2, const ALOHAOBJ& V3, ALOHAOBJ& V1 )
  {
    const cxtype_sv* wV2 = W_ACCESS::kernelAccessConst( V2.w );
    const cxtype_sv* wV3 = W_ACCESS::kernelAccessConst( V3.w );
    cxtype_sv* wV1 = W_ACCESS::kernelAccess( V1.w );
    for( int i = 0; i < ALOHAOBJ::np4; i++ ) V1.pvec[i] = V2.pvec[i];
    for( int i = 0; i < ALOHAOBJ::nw6; i++ ) wV1[i] = wV2[i] + wV3[i];
    V1.flv_index = V2.flv_index;
    return;
  }

  // As SUMW_1, for the contributions which enter with a minus sign.
  template<class W_ACCESS>
  __device__ inline void
  SUBW_1( const ALOHAOBJ& V2, const ALOHAOBJ& V3, ALOHAOBJ& V1 )
  {
    const cxtype_sv* wV2 = W_ACCESS::kernelAccessConst( V2.w );
    const cxtype_sv* wV3 = W_ACCESS::kernelAccessConst( V3.w );
    cxtype_sv* wV1 = W_ACCESS::kernelAccess( V1.w );
    for( int i = 0; i < ALOHAOBJ::np4; i++ ) V1.pvec[i] = V2.pvec[i];
    for( int i = 0; i < ALOHAOBJ::nw6; i++ ) wV1[i] = wV2[i] - wV3[i];
    V1.flv_index = V2.flv_index;
    return;
  }

  struct FLV_COUPLING_VIEW {

      const int* const partner1;
      const int* const partner2;
      const fptype* const value;

      __host__ __device__
      FLV_COUPLING_VIEW(const int* p1, const int* p2, const fptype* v)
      : partner1(p1), partner2(p2), value(v) {}
  };

  // FSTRIDE is the number of fptype's used to store one flavor slot of the value buffer:
  //  - independent (fixed) flavored couplings: FSTRIDE = nx2 = 2 (a single scalar complex, broadcast across the SIMD vector)
  //  - dependent (event-by-event, running-alphas) flavored couplings: FSTRIDE = nx2*neppC (an AOSOA SIMD record)
  // It must match C_ACCESS::flv_stride of the access type the consuming vertex routine is instantiated with.
  template<int SIZE, int STRIDE, int FSTRIDE = 2>
  class FLV_COUPLING_ARRAY {

      static_assert(SIZE >= 0, "flvCOUPs SIZE must be non-negative");
      static_assert(STRIDE > 0, "flvCOUPs STRIDE must be positive");
      static_assert(FSTRIDE > 0, "flvCOUPs FSTRIDE must be positive");
      const int* const partner1;
      const int* const partner2;
      const fptype* const value;

    public:
      __host__ __device__
      FLV_COUPLING_ARRAY(const int* p1, const int* p2, const fptype* v)
      : partner1(p1), partner2(p2), value(v) {}

      __host__ __device__
      FLV_COUPLING_VIEW operator[](const int i) const {
        return FLV_COUPLING_VIEW{
          partner1 + i*STRIDE,
          partner2 + i*STRIDE,
          value + i*FSTRIDE*STRIDE
        };
      }
  };
%(function_declarations)s
