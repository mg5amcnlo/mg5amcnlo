// Copyright (C) 2010 The ALOHA Development team and Contributors.
// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Sep 2010) for the MG5aMC backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: A. Valassi (Sep 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024) for the MG5aMC CUDACPP plugin.
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
// This file has been automatically generated for %(output_name)s by
%(info_lines)s
//==========================================================================

#ifndef HelAmps_%(model_name)s_H
#define HelAmps_%(model_name)s_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuVectors.h"

#include "Parameters.h"

#include <cassert>
//#include <cmath>
//#include <cstdlib>
//#include <iomanip>
//#include <iostream>

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{

  // ALOHA-style object for easy flavor consolidation and non-template API
  struct ALOHAOBJ {

      static constexpr int np4 = 4; // dimensions of 4-momenta (E,px,py,pz)
      static constexpr int nw6 = %(nwave)d; // dimensions of each wavefunction (notice, this is +1 in case of FD gauge)
      fptype_sv * pvec;
      fptype * w;
      int flv_index;

      __host__ __device__ ALOHAOBJ() {}
      __host__ __device__ ALOHAOBJ(fptype_sv * pvec_sv, cxtype_sv * w_sv, int flv = -1)
          : pvec(pvec_sv), w(reinterpret_cast<fptype*>(w_sv)), flv_index(flv) {}
  };

  struct FLV_COUPLING_VIEW {

      const int* const partner1;
      const int* const partner2;
      const fptype* const value;

      __host__ __device__
      FLV_COUPLING_VIEW(const int* p1, const int* p2, const fptype* v)
      : partner1(p1), partner2(p2), value(v) {}
  };

  template<int SIZE, int STRIDE>
  class FLV_COUPLING_ARRAY {

      static_assert(SIZE >= 0, "flvCOUPs SIZE must be non-negative");
      static_assert(STRIDE > 0, "flvCOUPs STRIDE must be positive");
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
          value + i*2*STRIDE
        };
      }
  };
%(function_declarations)s
