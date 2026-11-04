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
// This file has been automatically generated for CUDA/C++ standalone by
//  MadGraph7 v. %(version)s, %(date)s
//  By the MadGraph7 Development Team
//  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef HelAmps_sm_H
#define HelAmps_sm_H 1

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
      static constexpr int nw6 = 5; // dimensions of each wavefunction (notice, this is +1 in case of FD gauge)
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

  //--------------------------------------------------------------------------

#ifdef MGONGPU_INLINE_HELAMPS
#define INLINE inline
#define ALWAYS_INLINE __attribute__( ( always_inline ) )
#else
#define INLINE
#define ALWAYS_INLINE
#endif

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  ixxxxx( const fptype_momenta momenta[], // input: momenta
          const fptype_amp fmass,     // input: fermion mass
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fi,          // output: aloha objects
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  ipzxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype_amp fmass,   // [skip: ASSUME fermion mass==0]
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavor index
          ALOHAOBJ & fi,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == -PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  imzxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype_amp fmass,   // [skip: ASSUME fermion mass==0]
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavor index
          ALOHAOBJ & fi,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PT > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  ixzxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype_amp fmass,   // [skip: ASSUME fermion mass==0]
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavor index
          ALOHAOBJ & fi,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction vc[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  vxxxxx( const fptype_momenta momenta[], // input: momenta
          const fptype_amp vmass,     // input: vector boson mass
          const fptype_amp_sv nhel, // input: -1, 0 (only if vmass!=0) or +1 (helicity of vector boson, per SIMD lane/GPU thread)
          const int nsv,          // input: +1 (final) or -1 (initial)
          const int flv,          // input: flavor index
          ALOHAOBJ & vc,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction sc[3] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  sxxxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype,                 // WARNING: input "smass" unused (missing in Fortran) - scalar boson mass
          //const int,                    // WARNING: input "nhel" unused (missing in Fortran) - scalar has no helicity!
          const int nss,          // input: +1 (final) or -1 (initial)
          const int flv,          // input: flavor index
          ALOHAOBJ & sc,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  oxxxxx( const fptype_momenta momenta[], // input: momenta
          const fptype_amp fmass,     // input: fermion mass
          const fptype_amp_sv nhel, // input: -1, 0 (only if vmass!=0) or +1 (helicity of vector boson, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavor index
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  opzxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype_amp fmass,   // [skip: ASSUME fermion mass==0]
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavor index
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == -PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  omzxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype_amp fmass,   // [skip: ASSUME fermion mass==0]
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavor index
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  oxzxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype_amp fmass,   // [skip: ASSUME fermion mass==0]
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavor index
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;


//--------------------------------------------------------------------------

  // Compute the direction n[5] of the gauge q[5]
  __host__ __device__ INLINE void
  define_gauge_dir( const fptype_amp_sv q[], // input: gauge
                    fptype_amp_sv n[]        // output: direction
                    ) ALWAYS_INLINE;


  //--------------------------------------------------------------------------
  // Compute a propagator factor d out of gauge q[5] and a mass
  __host__ __device__ INLINE void
  calculate_propagator_factor( const fptype_amp_sv q[5], // input: gauge
                               const fptype_amp_sv mass, // input: mass
                               fptype_amp_sv *d          // output: propagator factor
                               ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------
  // multiply by propagation factor from m and wawefunctionsin[] and output them
  // as wavefunctionout[]
  template< class W_ACCESS>
  __host__ __device__ INLINE void
  multiply_propagator_factor( const fptype_amp wavefunctionsin[], // input: wavefunctions
                              const fptype_amp m,                 // input: mass
                              fptype_amp wavefunctionsout[]       // output: wavefunctions
                              ) ALWAYS_INLINE;
//==========================================================================

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  ixxxxx( const fptype_momenta momenta[], // input: momenta
          const fptype_amp fmass,     // input: fermion mass
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fi,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    // NEW IMPLEMENTATION FIXING FLOATING POINT EXCEPTIONS IN SIMD CODE (#701)
    // Variables xxxDENOM are a hack to avoid division-by-0 FPE while preserving speed (#701 and #727)
    // Variables xxxDENOM are declared as 'volatile' to make sure they are not optimized away on clang! (#724)
    // A few additional variables are declared as 'volatile' to avoid sqrt-of-negative-number FPEs (#736)
    const fptype_momenta_sv& pvec0_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_momenta_sv& pvec1_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_momenta_sv& pvec2_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_momenta_sv& pvec3_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    const fptype_amp_sv pvec0 = fpamp_of_mom(pvec0_);
    const fptype_amp_sv pvec1 = fpamp_of_mom(pvec1_);
    const fptype_amp_sv pvec2 = fpamp_of_mom(pvec2_);
    const fptype_amp_sv pvec3 = fpamp_of_mom(pvec3_);
    cxtype_amp_sv* w = W_ACCESS::kernelAccess( fi.w );
    fi.pvec[0] = -pvec0_ * static_cast<fptype_momenta>(nsf);
    fi.pvec[1] = -pvec1_ * static_cast<fptype_momenta>(nsf);
    fi.pvec[2] = -pvec2_ * static_cast<fptype_momenta>(nsf);
    fi.pvec[3] = -pvec3_ * static_cast<fptype_momenta>(nsf);
    fi.flv_index = flv;
    const fptype_amp_sv nh = nhel * static_cast<fptype_amp>( nsf );
    // BRANCHLESS HELICITY: the helicity is not a branch condition but enters as the truth values
    // tp (nh == +1) and tm (nh == -1), 1 or 0, which multiply the two alternatives. Different
    // SIMD lanes or GPU threads can then carry different helicities and still run in lock-step.
    const fptype_amp_sv tp = ( 1 + nh ) * (fptype_amp)0.5;
    const fptype_amp_sv tm = 1 - tp;
    if( fmass != 0. )
    {
#ifndef MGONGPU_CPPSIMD
      const fptype_amp_sv pp = fpmin( pvec0, fpsqrt( pvec1 * pvec1 + pvec2 * pvec2 + pvec3 * pvec3 ) );
#else
      volatile fptype_amp_sv p2 = pvec1 * pvec1 + pvec2 * pvec2 + pvec3 * pvec3; // volatile fixes #736
      const fptype_amp_sv pp = fpmin( pvec0, fpsqrt( p2 ) );
#endif
      // The Fortran ip/im indices ( 1 + nh ) / 2 and ( 1 - nh ) / 2 (for pp > 0, ( 3 +- nh ) / 2 as omega(2) has
      // indexes 1,2) are the truth values tp and tm: an array element x[ip] is tm * x[0] + tp * x[1], and
      // ip * x[ip] is just tp * x[1] (and likewise for im)
#ifndef MGONGPU_CPPSIMD
      if( pp == 0. )
      {
        // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
        fptype_amp sqm[2] = { fpsqrt( std::abs( fmass ) ), 0. }; // possibility of negative fermion masses
        //sqm[1] = ( fmass < 0. ? -abs( sqm[0] ) : abs( sqm[0] ) ); // AV: why abs here?
        sqm[1] = ( fmass < 0. ? -sqm[0] : sqm[0] ); // AV: removed an abs here
        w[0] = cxmake( tp * sqm[1], 0 );
        w[1] = cxmake( tm * nsf * sqm[0], 0 );
        w[2] = cxmake( tp * nsf * sqm[0], 0 );
        w[3] = cxmake( tm * sqm[1], 0 );
      }
      else
      {
        const fptype_amp sf[2] = { fptype_amp( 1 + nsf + ( 1 - nsf ) * nh ) * (fptype_amp)0.5,
                               fptype_amp( 1 + nsf - ( 1 - nsf ) * nh ) * (fptype_amp)0.5 };
        fptype_amp omega[2] = { fpsqrt( pvec0 + pp ), 0. };
        omega[1] = fmass / omega[0];
        const fptype_amp sfomega[2] = { sf[0] * ( tm * omega[0] + tp * omega[1] ), sf[1] * ( tp * omega[0] + tm * omega[1] ) };
        const fptype_amp pp3 = fpmax( fpternary( fpsignbit( pvec3 ), ( pvec1 * pvec1 + pvec2 * pvec2 ) / ( pp - pvec3 ), pp + pvec3 ), 0. );
        const cxtype_amp chi[2] = { cxmake( fpsqrt( pp3 * (fptype_amp)0.5 / pp ), 0. ),
                                ( pp3 == 0. ? cxmake( -nh, 0. ) : cxmake( nh * pvec1, pvec2 ) / fpsqrt( (fptype_amp)2. * pp * pp3 ) ) };
        const cxtype_amp chi_im = tp * chi[0] + tm * chi[1]; // chi[im]
        const cxtype_amp chi_ip = tm * chi[0] + tp * chi[1]; // chi[ip]
        w[0] = sfomega[0] * chi_im;
        w[1] = sfomega[0] * chi_ip;
        w[2] = sfomega[1] * chi_im;
        w[3] = sfomega[1] * chi_ip;
      }
#else
      // Branch A: pp == 0.
      // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
      fptype_amp sqm[2] = { fpsqrt( std::abs( fmass ) ), 0 }; // possibility of negative fermion masses (NB: SCALAR!)
      sqm[1] = ( fmass < 0 ? -sqm[0] : sqm[0] );          // AV: removed an abs here (as above)
      const cxtype_amp_v fiA_2 = cxmake( tp * sqm[1], 0 );
      const cxtype_amp_v fiA_3 = cxmake( tm * (fptype_amp)nsf * sqm[0], 0 );
      const cxtype_amp_v fiA_4 = cxmake( tp * (fptype_amp)nsf * sqm[0], 0 );
      const cxtype_amp_v fiA_5 = cxmake( tm * sqm[1], 0 );
      // Branch B: pp != 0.
      const fptype_amp_v sf[2] = { ( (fptype_amp)( 1 + nsf ) + (fptype_amp)( 1 - nsf ) * nh ) * (fptype_amp)0.5,
                                   ( (fptype_amp)( 1 + nsf ) - (fptype_amp)( 1 - nsf ) * nh ) * (fptype_amp)0.5 };
      fptype_v omega[2] = { fpsqrt( pvec0 + pp ), 0 };
      omega[1] = fmass / omega[0];
      const fptype_amp_v sfomega[2] = { sf[0] * ( tm * omega[0] + tp * omega[1] ), sf[1] * ( tp * omega[0] + tm * omega[1] ) };
      const fptype_amp_v pp3 = fpmax( fpternary( fpsignbit( pvec3 ), ( pvec1 * pvec1 + pvec2 * pvec2 ) / ( pp - pvec3 ), pp + pvec3 ), 0 );
      volatile fptype_amp_v ppDENOM = fpternary( pp != 0, pp, 1. );    // hack: ppDENOM[ieppV]=1 if pp[ieppV]==0
      volatile fptype_amp_v pp3DENOM = fpternary( pp3 != 0, pp3, 1. ); // hack: pp3DENOM[ieppV]=1 if pp3[ieppV]==0
      volatile fptype_amp_v chi0r2 = pp3 * 0.5 / ppDENOM;              // volatile fixes #736
      const cxtype_amp_v chi[2] = { cxmake( fpsqrt( chi0r2 ), 0 ),     // hack: dummy[ieppV] is not used if pp[ieppV]==0
                                cxternary( ( pp3 == 0. ),
                                           cxmake( -nh, 0 ),
                                           cxmake( nh * pvec1, pvec2 ) / fpsqrt( (fptype_amp)2. * ppDENOM * pp3DENOM ) ) }; // hack: dummy[ieppV] is not used if pp[ieppV]==0
      const cxtype_amp_v chi_im = tp * chi[0] + tm * chi[1]; // chi[im]
      const cxtype_amp_v chi_ip = tm * chi[0] + tp * chi[1]; // chi[ip]
      const cxtype_amp_v fiB_2 = sfomega[0] * chi_im;
      const cxtype_amp_v fiB_3 = sfomega[0] * chi_ip;
      const cxtype_amp_v fiB_4 = sfomega[1] * chi_im;
      const cxtype_amp_v fiB_5 = sfomega[1] * chi_ip;
      // Choose between the results from branch A and branch B
      const bool_v mask = ( pp == 0. );
      w[0] = cxternary( mask, fiA_2, fiB_2 );
      w[1] = cxternary( mask, fiA_3, fiB_3 );
      w[2] = cxternary( mask, fiA_4, fiB_4 );
      w[3] = cxternary( mask, fiA_5, fiB_5 );
#endif
    }
    else
    {
#ifdef MGONGPU_CPPSIMD
      // pvec0+pvec3 is a cancelling difference of two ~|p| numbers for a
      // backward-moving massless fermion, and everything below divides by it.
      // Take it from the light-cone identity p+ p- = pT^2 instead.
      const fptype_amp_sv p0mp3 = pvec0 - pvec3;
      volatile fptype_amp_sv p0mp3DENOM = fpternary( p0mp3 > 0, p0mp3, 1. ); // hack: dummy p0mp3DENOM[ieppV]=1 if p0mp3[ieppV]<=0
      volatile fptype_amp_sv p0p3 = fpternary( ( pvec3 < 0. and p0mp3 > 0. ),
                                              ( pvec1 * pvec1 + pvec2 * pvec2 ) / (const fptype_amp_sv)p0mp3DENOM,
                                              fpmax( pvec0 + pvec3, 0 ) ); // volatile fixes #736
      volatile fptype_amp_sv sqp0p3 = fpternary( ( pvec1 == 0. and pvec2 == 0. and pvec3 < 0. ),
                                             fptype_sv{ 0 },
                                             fpsqrt( p0p3 ) * (fptype_amp)nsf );
      volatile fptype_amp_sv sqp0p3DENOM = fpternary( sqp0p3 != 0, (fptype_sv)sqp0p3, 1. ); // hack: dummy sqp0p3DENOM[ieppV]=1 if sqp0p3[ieppV]==0
      cxtype_sv chi[2] = { cxmake( (fptype_v)sqp0p3, 0. ),
                           cxternary( sqp0p3 == 0,
                                      cxmake( -nhel * fpsqrt( (fptype_amp)2. * pvec0 ), 0. ),
                                      cxmake( nh * pvec1, pvec2 ) / (const fptype_v)sqp0p3DENOM ) }; // hack: dummy[ieppV] is not used if sqp0p3[ieppV]==0
#else
      // pvec0+pvec3 is a cancelling difference of two ~|p| numbers for a
      // backward-moving massless fermion, and everything below divides by it.
      // Take it from the light-cone identity p+ p- = pT^2 instead.
      const fptype_amp_sv p0mp3 = pvec0 - pvec3;
      const fptype_amp_sv p0p3stable = fpternary( ( pvec3 < 0. and p0mp3 > 0. ),
                                              ( pvec1 * pvec1 + pvec2 * pvec2 ) / fpternary( p0mp3 > 0., p0mp3, fptype_amp_sv{ 1 } ),
                                              fpmax( pvec0 + pvec3, 0. ) );
      const fptype_amp_sv sqp0p3 = fpternary( ( pvec1 == 0. and pvec2 == 0. and pvec3 < 0. ),
                                          fptype_sv{ 0 },
                                          fpsqrt( p0p3stable ) * (fptype_amp)nsf );
      const cxtype_amp_sv chi[2] = { cxmake( sqp0p3, 0. ),
                                 ( sqp0p3 == 0. ? cxmake( -nhel * fpsqrt( (fptype_amp)2. * pvec0 ), 0. ) : cxmake( nh * pvec1, pvec2 ) / sqp0p3 ) };
#endif
      w[0] = tm * chi[1];
      w[1] = tm * chi[0];
      w[2] = tp * chi[0];
      w[3] = tp * chi[1];
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  ipzxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype_amp fmass,   // [skip: ASSUME fermion mass==0]
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fi,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_momenta_sv& pvec3_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    const fptype_amp_sv pvec3 = fpamp_of_mom(pvec3_);
    cxtype_amp_sv* w = W_ACCESS::kernelAccess( fi.w );
    fi.pvec[0] = -pvec3_ * static_cast<fptype_momenta>(nsf);
    fi.pvec[1] = fptype_momenta_sv{ 0 };
    fi.pvec[2] = fptype_momenta_sv{ 0 };
    fi.pvec[3] = -pvec3_ * static_cast<fptype_momenta>(nsf);
    fi.flv_index = flv;
    const fptype_amp_sv nh = nhel * static_cast<fptype_amp>( nsf );
    // BRANCHLESS HELICITY: the helicity is not a branch condition but enters as the truth values
    // tp (nh == +1) and tm (nh == -1), 1 or 0, which multiply the two alternatives. Different
    // SIMD lanes or GPU threads can then carry different helicities and still run in lock-step.
    const fptype_amp_sv tp = ( 1 + nh ) * (fptype_amp)0.5;
    const fptype_amp_sv tm = 1 - tp;
    const cxtype_amp_sv sqp0p3 = cxmake( fpsqrt( (fptype_amp)2. * pvec3 ) * (fptype_amp)nsf, 0. );
    w[0] = cxzero_sv();
    w[1] = tm * sqp0p3;
    w[2] = tp * sqp0p3;
    w[3] = cxzero_sv();
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == -PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  imzxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype_amp fmass,   // [skip: ASSUME fermion mass==0]
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fi,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_momenta_sv& pvec3_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    const fptype_amp_sv pvec3 = fpamp_of_mom(pvec3_);
    cxtype_amp_sv* w = W_ACCESS::kernelAccess( fi.w );
    fi.pvec[0] =  pvec3_ * static_cast<fptype_momenta>(nsf);
    fi.pvec[1] = fptype_momenta_sv{ 0 };
    fi.pvec[2] = fptype_momenta_sv{ 0 };
    fi.pvec[3] = -pvec3_ * static_cast<fptype_momenta>(nsf);
    fi.flv_index = flv;
    const fptype_amp_sv nh = nhel * static_cast<fptype_amp>( nsf );
    // BRANCHLESS HELICITY: the helicity is not a branch condition but enters as the truth values
    // tp (nh == +1) and tm (nh == -1), 1 or 0, which multiply the two alternatives. Different
    // SIMD lanes or GPU threads can then carry different helicities and still run in lock-step.
    const fptype_amp_sv tp = ( 1 + nh ) * (fptype_amp)0.5;
    const fptype_amp_sv tm = 1 - tp;
    const cxtype_amp_sv chi = cxmake( -nhel * fpsqrt( -(fptype_amp)2. * pvec3 ), 0. );
    w[0] = tm * chi;
    w[1] = cxzero_sv();
    w[2] = cxzero_sv();
    w[3] = tp * chi;
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PT > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  ixzxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype_amp fmass,   // [skip: ASSUME fermion mass==0]
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fi,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_momenta_sv& pvec0_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_momenta_sv& pvec1_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_momenta_sv& pvec2_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_momenta_sv& pvec3_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    const fptype_amp_sv pvec0 = fpamp_of_mom(pvec0_);
    const fptype_amp_sv pvec1 = fpamp_of_mom(pvec1_);
    const fptype_amp_sv pvec2 = fpamp_of_mom(pvec2_);
    const fptype_amp_sv pvec3 = fpamp_of_mom(pvec3_);
    cxtype_amp_sv* w = W_ACCESS::kernelAccess( fi.w );
    fi.pvec[0] = -pvec0_ * static_cast<fptype_momenta>(nsf);
    fi.pvec[1] = -pvec1_ * static_cast<fptype_momenta>(nsf);
    fi.pvec[2] = -pvec2_ * static_cast<fptype_momenta>(nsf);
    fi.pvec[3] = -pvec3_ * static_cast<fptype_momenta>(nsf);
    fi.flv_index = flv;

    const fptype_amp_sv nh = nhel * static_cast<fptype_amp>( nsf );
    // BRANCHLESS HELICITY: the helicity is not a branch condition but enters as the truth values
    // tp (nh == +1) and tm (nh == -1), 1 or 0, which multiply the two alternatives. Different
    // SIMD lanes or GPU threads can then carry different helicities and still run in lock-step.
    const fptype_amp_sv tp = ( 1 + nh ) * (fptype_amp)0.5;
    const fptype_amp_sv tm = 1 - tp;
    //const float sqp0p3 = sqrtf( pvec0 + pvec3 ) * nsf; // AV: why force a float here?
    // pvec0+pvec3 is a cancelling difference of two ~|p| numbers for a
    // backward-moving massless fermion, and everything below divides by it.
    // Take it from the light-cone identity p+ p- = pT^2 instead (PT>0 here by
    // assumption, so no pT==0 branch is needed).
    const fptype_amp_sv p0mp3 = pvec0 - pvec3;
    const fptype_amp_sv p0p3stable = fpternary( ( pvec3 < 0. and p0mp3 > 0. ),
                                            ( pvec1 * pvec1 + pvec2 * pvec2 ) / fpternary( p0mp3 > 0., p0mp3, fptype_amp_sv{ 1 } ),
                                            pvec0 + pvec3 );
    const fptype_amp_sv sqp0p3 = fpsqrt( p0p3stable ) * static_cast<fptype_amp>(nsf);
    const cxtype_amp_sv chi0 = cxmake( sqp0p3, 0. );
    const cxtype_amp_sv chi1 = cxmake( nh * pvec1 / sqp0p3, pvec2 / sqp0p3 );
    w[0] = tm * chi1;
    w[1] = tm * chi0;
    w[2] = tp * chi0;
    w[3] = tp * chi1;
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction vc[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  vxxxxx( const fptype_momenta momenta[], // input: momenta
          const fptype_amp vmass,     // input: vector boson mass
          const fptype_amp_sv nhel, // input: -1, 0 (only if vmass!=0) or +1 (helicity of vector boson, per SIMD lane/GPU thread)
          const int nsv,          // input: +1 (final) or -1 (initial)
          const int flv,          // input: flavour
          ALOHAOBJ & vc,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    // NEW IMPLEMENTATION FIXING FLOATING POINT EXCEPTIONS IN SIMD CODE (#701)
    // Variables xxxDENOM are a hack to avoid division-by-0 FPE while preserving speed (#701 and #727)
    // Variables xxxDENOM are declared as 'volatile' to make sure they are not optimized away on clang! (#724)
    // A few additional variables are declared as 'volatile' to avoid sqrt-of-negative-number FPEs (#736)
    const fptype_momenta_sv& pvec0_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_momenta_sv& pvec1_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_momenta_sv& pvec2_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_momenta_sv& pvec3_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    const fptype_amp_sv pvec0 = fpamp_of_mom(pvec0_);
    const fptype_amp_sv pvec1 = fpamp_of_mom(pvec1_);
    const fptype_amp_sv pvec2 = fpamp_of_mom(pvec2_);
    const fptype_amp_sv pvec3 = fpamp_of_mom(pvec3_);
    cxtype_amp_sv* w = W_ACCESS::kernelAccess( vc.w );
    vc.pvec[0] = pvec0_ * static_cast<fptype_momenta>(nsv);
    vc.pvec[1] = pvec1_ * static_cast<fptype_momenta>(nsv);
    vc.pvec[2] = pvec2_ * static_cast<fptype_momenta>(nsv);
    vc.pvec[3] = pvec3_ * static_cast<fptype_momenta>(nsv);
    vc.flv_index = flv;
    const fptype_amp sqh = fpsqrt( 0.5 ); // AV this is > 0!
    const fptype_amp_sv hel = nhel;
    // BRANCHLESS HELICITY: |hel| is hel * hel for hel in { -1, 0, +1 }, so that the helicity can
    // differ from one SIMD lane or GPU thread to the next with no branch and no std::abs
    const fptype_amp_sv abshel = hel * hel;

    // FD gauge
     const cxtype_amp_sv cI = cxmake( 0. + fptype_amp_sv{ 0. },  1. + fptype_amp_sv{ 0. }  );
    fptype_amp_sv n[5];
    fptype_amp_sv nk;
    const fptype_amp_sv zero = 0. + fptype_amp_sv{ 0 };
    const fptype_amp_sv one = 1. + fptype_amp_sv{ 0 };

    if( vmass != 0. )
    {
      const fptype_amp_sv nsvahl = static_cast<fptype_amp>( nsv ) * abshel;
      const fptype_amp_sv hel0 = 1 - abshel; // truth value of hel == 0 (longitudinal polarisation)
#ifndef MGONGPU_CPPSIMD
      const fptype_amp_sv pt2 = ( pvec1 * pvec1 ) + ( pvec2 * pvec2 );
      const fptype_amp_sv pp = fpmin( pvec0, fpsqrt( pt2 + ( pvec3 * pvec3 ) ) );
      const fptype_amp_sv pt = fpmin( pp, fpsqrt( pt2 ) );
      if( pp == 0. )
      {
        w[0] = cxmake( 0., 0. );
        w[1] = cxmake( -hel * sqh, 0. );
        w[2] = cxmake( 0., nsvahl * sqh );
        w[3] = cxmake( hel0, 0. );
      }
      else
      {
        //printf( "DEBUG1011 (before emp); pvec0=%f vmass=%f pp=%f vmass*pp=%f\n", pvec0, vmass, pp, vmass * pp );
        //const fptype_amp emp = pvec / ( vmass * pp ); // this may give a FPE #1011 (why?! maybe when vmass=+-epsilon?)
        const fptype_amp emp = pvec0 / vmass / pp; // workaround for FPE #1011
        //printf( "DEBUG1011 (after emp); emp=%f\n", emp );
        w[0] = cxmake( hel0 * pp / vmass, 0. );
        w[3] = cxmake( hel0 * pvec3 * emp + hel * pt / pp * sqh, 0. );
        if( pt != 0. )
        {
          const fptype_amp pzpt = pvec3 / ( pp * pt ) * sqh * hel;
          w[1] = cxmake( hel0 * pvec1 * emp - pvec1 * pzpt, -nsvahl * pvec2 / pt * sqh );
          w[2] = cxmake( hel0 * pvec2 * emp - pvec2 * pzpt, nsvahl * pvec1 / pt * sqh );
        }
        else
        {
          w[1] = cxmake( -hel * sqh, 0. );
          // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
          //vc[4] = cxmake( 0., nsvahl * ( pvec3 < 0. ? -std::abs( sqh ) : std::abs( sqh ) ) ); // AV: why abs here?
          w[2] = cxmake( 0., nsvahl * ( pvec3 < 0. ? -sqh : sqh ) ); // AV: removed an abs here
        }
      }

      //FD gauge
      if( pp > 0. )
      {
        n[0] = ( pvec0 >= zero) ? one : -one;
        n[1] = -pvec1/pp;
        n[2] = -pvec2/pp;
        n[3] = -pvec3/pp;
        n[4] = zero;
      }
      else
      {
        n[0] = ( pvec0 >= zero) ? one : -one;
        n[1] = zero;
        n[2] = zero;
        n[3] = ( pvec0 >= zero) ? -one : one;
      }


      nk = n[0]*pvec0 - n[1]*pvec1 - n[2]*pvec2 - n[3]*pvec3;

      // BRANCHLESS HELICITY: |hel| = 1 keeps w[0..3] with w[4] = 0, hel = 0 (hel0 = 1) takes the
      // FD-gauge longitudinal polarisation instead (nk > 0 for a massive momentum, so the unused
      // alternative is finite and vanishes when multiplied by its zero truth value)
      w[0] = abshel * w[0] + hel0 * cxmake( -vmass / nk * n[0], zero );
      w[1] = abshel * w[1] + hel0 * cxmake( -vmass / nk * n[1], zero );
      w[2] = abshel * w[2] + hel0 * cxmake( -vmass / nk * n[2], zero );
      w[3] = abshel * w[3] + hel0 * cxmake( -vmass / nk * n[3], zero );
      w[4] = hel0 * ( -static_cast<fptype_amp>( nsv ) * cI );

#else
      volatile fptype_amp_sv pt2 = ( pvec1 * pvec1 ) + ( pvec2 * pvec2 );
      volatile fptype_amp_sv p2 = pt2 + ( pvec3 * pvec3 ); // volatile fixes #736
      const fptype_amp_sv pp = fpmin( pvec0, fpsqrt( p2 ) );
      const fptype_amp_sv pt = fpmin( pp, fpsqrt( pt2 ) );
      // Branch A: pp == 0.
      const cxtype_amp vcA_2 = cxmake( 0, 0 );
      const cxtype_amp_v vcA_3 = cxmake( -hel * sqh, 0 );
      const cxtype_amp_v vcA_4 = cxmake( 0., nsvahl * sqh );
      const cxtype_amp_v vcA_5 = cxmake( hel0, 0 );
      // Branch B: pp != 0.
      volatile fptype_amp_v ppDENOM = fpternary( pp != 0, pp, 1. ); // hack: ppDENOM[ieppV]=1 if pp[ieppV]==0
      const fptype_amp_v emp = pvec0 / ( vmass * ppDENOM );         // hack: dummy[ieppV] is not used if pp[ieppV]==0
      const cxtype_amp_v vcB_2 = cxmake( hel0 * pp / vmass, 0 );
      const cxtype_amp_v vcB_5 = cxmake( hel0 * pvec3 * emp + hel * pt / ppDENOM * sqh, 0 ); // hack: dummy[ieppV] is not used if pp[ieppV]==0
      // Branch B1: pp != 0. and pt != 0.
      volatile fptype_amp_v ptDENOM = fpternary( pt != 0, pt, 1. );                                                     // hack: ptDENOM[ieppV]=1 if pt[ieppV]==0
      const fptype_amp_v pzpt = pvec3 / ( ppDENOM * ptDENOM ) * sqh * hel;                                              // hack: dummy[ieppV] is not used if pp[ieppV]==0
      const cxtype_amp_v vcB1_3 = cxmake( hel0 * pvec1 * emp - pvec1 * pzpt, -nsvahl * pvec2 / ptDENOM * sqh ); // hack: dummy[ieppV] is not used if pt[ieppV]==0
      const cxtype_amp_v vcB1_4 = cxmake( hel0 * pvec2 * emp - pvec2 * pzpt, nsvahl * pvec1 / ptDENOM * sqh );  // hack: dummy[ieppV] is not used if pt[ieppV]==0
      // Branch B2: pp != 0. and pt == 0.
      const cxtype_amp_v vcB2_3 = cxmake( -hel * sqh, 0. );
      const cxtype_amp_v vcB2_4 = cxmake( 0., nsvahl * fpternary( ( pvec3 < 0 ), -sqh, sqh ) ); // AV: removed an abs here
      // Choose between the results from branch A and branch B (and from branch B1 and branch B2)
      const bool_v mask = ( pp == 0. );
      const bool_v maskB = ( pt != 0. );
      w[0] = cxternary( mask, vcA_2, vcB_2 );
      w[1] = cxternary( mask, vcA_3, cxternary( maskB, vcB1_3, vcB2_3 ) );
      w[2] = cxternary( mask, vcA_4, cxternary( maskB, vcB1_4, vcB2_4 ) );
      w[3] = cxternary( mask, vcA_5, vcB_5 );

      //FD gauge: same two branches as the scalar code above, selected lane by
      //lane. The division uses ppDENOM (=1 where pp==0) so that the lanes that
      //do not take it are not poisoned: a select discards the other value, but
      //a nan surviving an arithmetic combination (nan*0 is nan) would not be.
      const bool_v maskFD = ( pp > zero );
      n[0] = fpternary( pvec0 >= zero , one , -one );
      n[1] = fpternary( maskFD, -pvec1 / ppDENOM, zero );
      n[2] = fpternary( maskFD, -pvec2 / ppDENOM, zero );
      n[3] = fpternary( maskFD, -pvec3 / ppDENOM, -n[0] );
      n[4] = zero;

      nk = n[0]*pvec0 - n[1]*pvec1 - n[2]*pvec2 - n[3]*pvec3;

      // BRANCHLESS HELICITY: |hel| = 1 keeps w[0..3] with w[4] = 0, hel = 0 (hel0 = 1) takes the
      // FD-gauge longitudinal polarisation instead (nk > 0 for a massive momentum, so the unused
      // alternative is finite and vanishes when multiplied by its zero truth value)
      w[0] = abshel * w[0] + hel0 * cxmake( -vmass / nk * n[0], zero );
      w[1] = abshel * w[1] + hel0 * cxmake( -vmass / nk * n[1], zero );
      w[2] = abshel * w[2] + hel0 * cxmake( -vmass / nk * n[2], zero );
      w[3] = abshel * w[3] + hel0 * cxmake( -vmass / nk * n[3], zero );
      w[4] = hel0 * ( -static_cast<fptype_amp>( nsv ) * cI );
#endif
    }
    else
    {
      const fptype_amp_sv& pp = pvec0; // NB: rewrite the following as in Fortran, using pp instead of pvec0
#ifndef MGONGPU_CPPSIMD
      const fptype_amp_sv pt = fpsqrt( ( pvec1 * pvec1 ) + ( pvec2 * pvec2 ) );
#else
      volatile fptype_amp_sv pt2 = pvec1 * pvec1 + pvec2 * pvec2; // volatile fixes #736
      const fptype_amp_sv pt = fpsqrt( pt2 );
#endif
      w[0] = cxzero_sv();
      w[3] = cxmake( hel * pt / pp * sqh, 0. );
#ifndef MGONGPU_CPPSIMD
      if( pt != 0. )
      {
        const fptype_amp pzpt = pvec3 / ( pp * pt ) * sqh * hel;
        w[1] = cxmake( -pvec1 * pzpt, -nsv * pvec2 / pt * sqh );
        w[2] = cxmake( -pvec2 * pzpt, nsv * pvec1 / pt * sqh );
      }
      else
      {
        w[1] = cxmake( -hel * sqh, 0. );
        // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
        //w[2] = cxmake( 0, nsv * ( pvec3 < 0. ? -std::abs( sqh ) : std::abs( sqh ) ) ); // AV why abs here?
        w[2] = cxmake( 0., nsv * ( pvec3 < 0. ? -sqh : sqh ) ); // AV: removed an abs here
      }
#else
      // Branch A: pt != 0.
      volatile fptype_amp_v ptDENOM = fpternary( pt != 0, pt, 1. );                             // hack: ptDENOM[ieppV]=1 if pt[ieppV]==0
      const fptype_amp_v pzpt = pvec3 / ( pp * ptDENOM ) * sqh * hel;                           // hack: dummy[ieppV] is not used if pt[ieppV]==0
      const cxtype_amp_v vcA_3 = cxmake( -pvec1 * pzpt, -(fptype_amp)nsv * pvec2 / ptDENOM * sqh ); // hack: dummy[ieppV] is not used if pt[ieppV]==0
      const cxtype_amp_v vcA_4 = cxmake( -pvec2 * pzpt, (fptype_amp)nsv * pvec1 / ptDENOM * sqh );  // hack: dummy[ieppV] is not used if pt[ieppV]==0
      // Branch B: pt == 0.
      const cxtype_amp_v vcB_3 = cxmake( -hel * sqh, 0 );
      const cxtype_amp_v vcB_4 = cxmake( 0, (fptype_amp)nsv * fpternary( ( pvec3 < 0 ), -sqh, sqh ) ); // AV: removed an abs here
      // Choose between the results from branch A and branch B
      const bool_v mask = ( pt != 0. );
      w[1] = cxternary( mask, vcA_3, vcB_3 );
      w[2] = cxternary( mask, vcA_4, vcB_4 );
#endif
      //FD gauge
      w[4] = cxzero_sv();
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction sc[3] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  sxxxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype,                 // WARNING: input "smass" unused (missing in Fortran) - scalar boson mass
          //const int,                    // WARNING: input "nhel" unused (missing in Fortran) - scalar has no helicity!
          const int nss,          // input: +1 (final) or -1 (initial)
          const int flv,          // input: flavour
          ALOHAOBJ &sc,           // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_momenta_sv& pvec0_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_momenta_sv& pvec1_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_momenta_sv& pvec2_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_momenta_sv& pvec3_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    const fptype_amp_sv pvec0 = fpamp_of_mom(pvec0_);
    const fptype_amp_sv pvec1 = fpamp_of_mom(pvec1_);
    const fptype_amp_sv pvec2 = fpamp_of_mom(pvec2_);
    const fptype_amp_sv pvec3 = fpamp_of_mom(pvec3_);
    cxtype_amp_sv* w = W_ACCESS::kernelAccess( sc.w );
    sc.pvec[0] = pvec0_ * static_cast<fptype_momenta>(nss);
    sc.pvec[1] = pvec1_ * static_cast<fptype_momenta>(nss);
    sc.pvec[2] = pvec2_ * static_cast<fptype_momenta>(nss);
    sc.pvec[3] = pvec3_ * static_cast<fptype_momenta>(nss);
    sc.flv_index = flv;
    w[0] = cxmake( 1 + fptype_sv{ 0 }, 0 );
    //FD gauge
    w[1] = cxmake( 0 + fptype_sv{ 0 }, 0 );
    w[2] = cxmake( 0 + fptype_sv{ 0 }, 0 );
    w[3] = cxmake( 0 + fptype_sv{ 0 }, 0 );
    w[4] = cxmake( 1 + fptype_sv{ 0 }, 0 );

    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  oxxxxx( const fptype_momenta momenta[], // input: momenta
          const fptype_amp fmass,     // input: fermion mass
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          int flv,                // input: flavour
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    // NEW IMPLEMENTATION FIXING FLOATING POINT EXCEPTIONS IN SIMD CODE (#701)
    // Variables xxxDENOM are a hack to avoid division-by-0 FPE while preserving speed (#701 and #727)
    // Variables xxxDENOM are declared as 'volatile' to make sure they are not optimized away on clang! (#724)
    // A few additional variables are declared as 'volatile' to avoid sqrt-of-negative-number FPEs (#736)
    const fptype_momenta_sv& pvec0_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_momenta_sv& pvec1_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_momenta_sv& pvec2_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_momenta_sv& pvec3_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    const fptype_amp_sv pvec0 = fpamp_of_mom(pvec0_);
    const fptype_amp_sv pvec1 = fpamp_of_mom(pvec1_);
    const fptype_amp_sv pvec2 = fpamp_of_mom(pvec2_);
    const fptype_amp_sv pvec3 = fpamp_of_mom(pvec3_);
    cxtype_amp_sv* w = W_ACCESS::kernelAccess( fo.w );

    fo.pvec[0] = pvec0_ * static_cast<fptype_momenta>(nsf);
    fo.pvec[1] = pvec1_ * static_cast<fptype_momenta>(nsf);
    fo.pvec[2] = pvec2_ * static_cast<fptype_momenta>(nsf);
    fo.pvec[3] = pvec3_ * static_cast<fptype_momenta>(nsf);
    fo.flv_index = flv;
    const fptype_amp_sv nh = nhel * static_cast<fptype_amp>( nsf );
    // BRANCHLESS HELICITY: the helicity is not a branch condition but enters as the truth values
    // tp (nh == +1) and tm (nh == -1), 1 or 0, which multiply the two alternatives. Different
    // SIMD lanes or GPU threads can then carry different helicities and still run in lock-step.
    const fptype_amp_sv tp = ( 1 + nh ) * (fptype_amp)0.5;
    const fptype_amp_sv tm = 1 - tp;
    if( fmass != 0. )
    {
#ifndef MGONGPU_CPPSIMD
      const fptype_amp_sv pp = fpmin( pvec0, fpsqrt( ( pvec1 * pvec1 ) + ( pvec2 * pvec2 ) + ( pvec3 * pvec3 ) ) );
      if( pp == 0. )
      {
        // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
        fptype_amp sqm[2] = { fpsqrt( std::abs( fmass ) ), 0. }; // possibility of negative fermion masses
        //sqm[1] = ( fmass < 0. ? -abs( sqm[0] ) : abs( sqm[0] ) ); // AV: why abs here?
        sqm[1] = ( fmass < 0. ? -sqm[0] : sqm[0] ); // AV: removed an abs here
        // The Fortran ip = -( ( 1 - nh ) / 2 ) * nhel and im = ( 1 + nh ) / 2 * nhel are -tm * nhel and
        // tp * nhel, so that |ip| = tm and |im| = tp: sqm[|ip|] is sqm[1] wherever ip != 0, and so on
        w[0] = cxmake( tp * nhel * sqm[0], 0 );
        w[1] = cxmake( -tm * nhel * nsf * sqm[1], 0 );
        w[2] = cxmake( tp * nhel * nsf * sqm[1], 0 );
        w[3] = cxmake( -tm * nhel * sqm[0], 0 );
      }
      else
      {
        const fptype_amp sf[2] = { fptype_amp( 1 + nsf + ( 1 - nsf ) * nh ) * (fptype_amp)0.5,
                               fptype_amp( 1 + nsf - ( 1 - nsf ) * nh ) * (fptype_amp)0.5 };
        fptype_amp omega[2] = { fpsqrt( pvec0 + pp ), 0. };
        omega[1] = fmass / omega[0];
        // The Fortran ip/im indices ( 3 +- nh ) / 2 (omega(2) has indexes 1,2) are the truth values tp/tm
        const fptype_amp sfomeg[2] = { sf[0] * ( tm * omega[0] + tp * omega[1] ), sf[1] * ( tp * omega[0] + tm * omega[1] ) };
        const fptype_amp pp3 = fpmax( fpternary( fpsignbit( pvec3 ), ( pvec1 * pvec1 + pvec2 * pvec2 ) / ( pp - pvec3 ), pp + pvec3 ), 0. );
        const cxtype_amp chi[2] = { cxmake( fpsqrt( pp3 * (fptype_amp)0.5 / pp ), 0. ),
                                ( ( pp3 == 0. ) ? cxmake( -nh, 0. )
                                                : cxmake( nh * pvec1, -pvec2 ) / fpsqrt( (fptype_amp)2. * pp * pp3 ) ) };
        const cxtype_amp chi_im = tp * chi[0] + tm * chi[1]; // chi[im]
        const cxtype_amp chi_ip = tm * chi[0] + tp * chi[1]; // chi[ip]
        w[0] = sfomeg[1] * chi_im;
        w[1] = sfomeg[1] * chi_ip;
        w[2] = sfomeg[0] * chi_im;
        w[3] = sfomeg[0] * chi_ip;
      }
#else
      volatile fptype_amp_sv p2 = pvec1 * pvec1 + pvec2 * pvec2 + pvec3 * pvec3; // volatile fixes #736
      const fptype_amp_sv pp = fpmin( pvec0, fpsqrt( p2 ) );
      // Branch A: pp == 0.
      // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
      fptype_amp sqm[2] = { fpsqrt( std::abs( fmass ) ), 0 }; // possibility of negative fermion masses
      sqm[1] = ( fmass < 0 ? -sqm[0] : sqm[0] );          // AV: removed an abs here (as above)
      // (see the scalar code above for the truth values tp/tm in place of the Fortran ip/im indices)
      const cxtype_amp_v foA_2 = cxmake( tp * nhel * sqm[0], 0 );
      const cxtype_amp_v foA_3 = cxmake( -tm * nhel * (fptype_amp)nsf * sqm[1], 0 );
      const cxtype_amp_v foA_4 = cxmake( tp * nhel * (fptype_amp)nsf * sqm[1], 0 );
      const cxtype_amp_v foA_5 = cxmake( -tm * nhel * sqm[0], 0 );
      // Branch B: pp != 0.
      const fptype_amp_v sf[2] = { ( (fptype_amp)( 1 + nsf ) + (fptype_amp)( 1 - nsf ) * nh ) * (fptype_amp)0.5,
                                   ( (fptype_amp)( 1 + nsf ) - (fptype_amp)( 1 - nsf ) * nh ) * (fptype_amp)0.5 };
      fptype_v omega[2] = { fpsqrt( pvec0 + pp ), 0 };
      omega[1] = fmass / omega[0];
      const fptype_amp_v sfomeg[2] = { sf[0] * ( tm * omega[0] + tp * omega[1] ), sf[1] * ( tp * omega[0] + tm * omega[1] ) };
      const fptype_amp_v pp3 = fpmax( fpternary( fpsignbit( pvec3 ), ( pvec1 * pvec1 + pvec2 * pvec2 ) / ( pp - pvec3 ), pp + pvec3 ), 0. );
      volatile fptype_amp_v ppDENOM = fpternary( pp != 0, pp, 1. );    // hack: ppDENOM[ieppV]=1 if pp[ieppV]==0
      volatile fptype_amp_v pp3DENOM = fpternary( pp3 != 0, pp3, 1. ); // hack: pp3DENOM[ieppV]=1 if pp3[ieppV]==0
      volatile fptype_amp_v chi0r2 = pp3 * 0.5 / ppDENOM;              // volatile fixes #736
      const cxtype_amp_v chi[2] = { cxmake( fpsqrt( chi0r2 ), 0. ),    // hack: dummy[ieppV] is not used if pp[ieppV]==0
                                ( cxternary( ( pp3 == 0. ),
                                             cxmake( -nh, 0. ),
                                             cxmake( nh * pvec1, -pvec2 ) / fpsqrt( (fptype_amp)2. * ppDENOM * pp3DENOM ) ) ) }; // hack: dummy[ieppV] is not used if pp[ieppV]==0
      const cxtype_amp_v chi_im = tp * chi[0] + tm * chi[1]; // chi[im]
      const cxtype_amp_v chi_ip = tm * chi[0] + tp * chi[1]; // chi[ip]
      const cxtype_amp_v foB_2 = sfomeg[1] * chi_im;
      const cxtype_amp_v foB_3 = sfomeg[1] * chi_ip;
      const cxtype_amp_v foB_4 = sfomeg[0] * chi_im;
      const cxtype_amp_v foB_5 = sfomeg[0] * chi_ip;
      // Choose between the results from branch A and branch B
      const bool_v mask = ( pp == 0. );
      w[0] = cxternary( mask, foA_2, foB_2 );
      w[1] = cxternary( mask, foA_3, foB_3 );
      w[2] = cxternary( mask, foA_4, foB_4 );
      w[3] = cxternary( mask, foA_5, foB_5 );
#endif
    }
    else
    {
#ifdef MGONGPU_CPPSIMD
      // pvec0+pvec3 is a cancelling difference of two ~|p| numbers for a
      // backward-moving massless fermion, and everything below divides by it.
      // Take it from the light-cone identity p+ p- = pT^2 instead.
      const fptype_amp_sv p0mp3 = pvec0 - pvec3;
      volatile fptype_amp_sv p0mp3DENOM = fpternary( p0mp3 > 0, p0mp3, 1. ); // hack: dummy p0mp3DENOM[ieppV]=1 if p0mp3[ieppV]<=0
      volatile fptype_amp_sv p0p3 = fpternary( ( pvec3 < 0. and p0mp3 > 0. ),
                                              ( pvec1 * pvec1 + pvec2 * pvec2 ) / (const fptype_amp_sv)p0mp3DENOM,
                                              fpmax( pvec0 + pvec3, 0 ) ); // volatile fixes #736
      volatile fptype_amp_sv sqp0p3 = fpternary( ( pvec1 == 0. and pvec2 == 0. and pvec3 < 0. ),
                                             fptype_sv{ 0 },
                                             fpsqrt( p0p3 ) * (fptype_amp)nsf );
      volatile fptype_amp_v sqp0p3DENOM = fpternary( sqp0p3 != 0, (fptype_sv)sqp0p3, 1. ); // hack: sqp0p3DENOM[ieppV]=1 if sqp0p3[ieppV]==0
      const cxtype_amp_v chi[2] = { cxmake( (fptype_v)sqp0p3, 0. ),
                                cxternary( ( sqp0p3 == 0. ),
                                           cxmake( -nhel, 0. ) * fpsqrt( (fptype_amp)2. * pvec0 ),
                                           cxmake( nh * pvec1, -pvec2 ) / (const fptype_amp_sv)sqp0p3DENOM ) }; // hack: dummy[ieppV] is not used if sqp0p3[ieppV]==0
#else
      // pvec0+pvec3 is a cancelling difference of two ~|p| numbers for a
      // backward-moving massless fermion, and everything below divides by it.
      // Take it from the light-cone identity p+ p- = pT^2 instead.
      const fptype_amp_sv p0mp3 = pvec0 - pvec3;
      const fptype_amp_sv p0p3stable = fpternary( ( pvec3 < 0. and p0mp3 > 0. ),
                                              ( pvec1 * pvec1 + pvec2 * pvec2 ) / fpternary( p0mp3 > 0., p0mp3, fptype_amp_sv{ 1 } ),
                                              fpmax( pvec0 + pvec3, 0. ) );
      const fptype_amp_sv sqp0p3 = fpternary( ( pvec1 == 0. ) and ( pvec2 == 0. ) and ( pvec3 < 0. ),
                                          fptype_sv{ 0 },
                                          fpsqrt( p0p3stable ) * (fptype_amp)nsf );
      const cxtype_amp_sv chi[2] = { cxmake( sqp0p3, 0. ),
                                 ( sqp0p3 == 0. ? cxmake( -nhel, 0. ) * fpsqrt( (fptype_amp)2. * pvec0 ) : cxmake( nh * pvec1, -pvec2 ) / sqp0p3 ) };
#endif
      w[0] = tp * chi[0];
      w[1] = tp * chi[1];
      w[2] = tm * chi[1];
      w[3] = tm * chi[0];
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  opzxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype_amp fmass,   // [skip: ASSUME fermion mass==0]
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_momenta_sv& pvec3_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    const fptype_amp_sv pvec3 = fpamp_of_mom(pvec3_);
    cxtype_amp_sv* w = W_ACCESS::kernelAccess( fo.w );
    fo.pvec[0] = pvec3_ * static_cast<fptype_momenta>(nsf);
    fo.pvec[1] = fptype_momenta_sv{ 0 };
    fo.pvec[2] = fptype_momenta_sv{ 0 };
    fo.pvec[3] = pvec3_ * static_cast<fptype_momenta>(nsf);
    fo.flv_index = flv;
    const fptype_amp_sv nh = nhel * static_cast<fptype_amp>( nsf );
    // BRANCHLESS HELICITY: the helicity is not a branch condition but enters as the truth values
    // tp (nh == +1) and tm (nh == -1), 1 or 0, which multiply the two alternatives. Different
    // SIMD lanes or GPU threads can then carry different helicities and still run in lock-step.
    const fptype_amp_sv tp = ( 1 + nh ) * (fptype_amp)0.5;
    const fptype_amp_sv tm = 1 - tp;
    const cxtype_amp_sv csqp0p3 = cxmake( fpsqrt( (fptype_amp)2. * pvec3 ) * (fptype_amp)nsf, 0. );
    w[0] = tp * csqp0p3;
    w[1] = cxzero_sv();
    w[2] = cxzero_sv();
    w[3] = tm * csqp0p3;
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == -PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  omzxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype_amp fmass,   // [skip: ASSUME fermion mass==0]
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_momenta_sv& pvec3_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    const fptype_amp_sv pvec3 = fpamp_of_mom(pvec3_);
    cxtype_amp_sv* w = W_ACCESS::kernelAccess( fo.w );
    fo.pvec[0] = -pvec3_ * static_cast<fptype_momenta>(nsf);
    fo.pvec[1] = fptype_momenta_sv{ 0 };
    fo.pvec[2] = fptype_momenta_sv{ 0 };
    fo.pvec[3] = pvec3_ * static_cast<fptype_momenta>(nsf);
    fo.flv_index = flv;
    const fptype_amp_sv nh = nhel * static_cast<fptype_amp>( nsf );
    // BRANCHLESS HELICITY: the helicity is not a branch condition but enters as the truth values
    // tp (nh == +1) and tm (nh == -1), 1 or 0, which multiply the two alternatives. Different
    // SIMD lanes or GPU threads can then carry different helicities and still run in lock-step.
    const fptype_amp_sv tp = ( 1 + nh ) * (fptype_amp)0.5;
    const fptype_amp_sv tm = 1 - tp;
    const cxtype_amp_sv chi1 = cxmake( -nhel, 0. ) * fpsqrt( -(fptype_amp)2. * pvec3 );
    w[0] = cxzero_sv();
    w[1] = tp * chi1;
    w[2] = tm * chi1;
    w[3] = cxzero_sv();
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PT > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  oxzxxx( const fptype_momenta momenta[], // input: momenta
          //const fptype_amp fmass,   // [skip: ASSUME fermion mass==0]
          const fptype_amp_sv nhel, // input: -1 or +1 (helicity of fermion, per SIMD lane/GPU thread)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_momenta_sv& pvec0_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_momenta_sv& pvec1_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_momenta_sv& pvec2_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_momenta_sv& pvec3_ = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    const fptype_amp_sv pvec0 = fpamp_of_mom(pvec0_);
    const fptype_amp_sv pvec1 = fpamp_of_mom(pvec1_);
    const fptype_amp_sv pvec2 = fpamp_of_mom(pvec2_);
    const fptype_amp_sv pvec3 = fpamp_of_mom(pvec3_);
    cxtype_amp_sv* w = W_ACCESS::kernelAccess( fo.w );

    fo.pvec[0] = pvec0_ * static_cast<fptype_momenta>(nsf);
    fo.pvec[1] = pvec1_ * static_cast<fptype_momenta>(nsf);
    fo.pvec[2] = pvec2_ * static_cast<fptype_momenta>(nsf);
    fo.pvec[3] = pvec3_ * static_cast<fptype_momenta>(nsf);
    fo.flv_index = flv;
    const fptype_amp_sv nh = nhel * static_cast<fptype_amp>( nsf );
    // BRANCHLESS HELICITY: the helicity is not a branch condition but enters as the truth values
    // tp (nh == +1) and tm (nh == -1), 1 or 0, which multiply the two alternatives. Different
    // SIMD lanes or GPU threads can then carry different helicities and still run in lock-step.
    const fptype_amp_sv tp = ( 1 + nh ) * (fptype_amp)0.5;
    const fptype_amp_sv tm = 1 - tp;
    //const float sqp0p3 = sqrtf( pvec0 + pvec3 ) * nsf; // AV: why force a float here?
    // pvec0+pvec3 is a cancelling difference of two ~|p| numbers for a
    // backward-moving massless fermion, and everything below divides by it.
    // Take it from the light-cone identity p+ p- = pT^2 instead (PT>0 here by
    // assumption, so no pT==0 branch is needed).
    const fptype_amp_sv p0mp3 = pvec0 - pvec3;
    const fptype_amp_sv p0p3stable = fpternary( ( pvec3 < 0. and p0mp3 > 0. ),
                                            ( pvec1 * pvec1 + pvec2 * pvec2 ) / fpternary( p0mp3 > 0., p0mp3, fptype_amp_sv{ 1 } ),
                                            pvec0 + pvec3 );
    const fptype_amp_sv sqp0p3 = fpsqrt( p0p3stable ) * static_cast<fptype_amp>(nsf);
    const cxtype_amp_sv chi0 = cxmake( sqp0p3, 0. );
    const cxtype_amp_sv chi1 = cxmake( nh * pvec1 / sqp0p3, -pvec2 / sqp0p3 );
    w[0] = tp * chi0;
    w[1] = tp * chi1;
    w[2] = tm * chi1;
    w[3] = tm * chi0;
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------
  // Compute the direction n[5] of the gauge q[5]
  // TODO: Utilise pvec instead of the whole q
  // Genuinely inline (unlike INLINE elsewhere): defined here, not in helas.cu.
  __host__ __device__ inline void
  define_gauge_dir( const cxtype_amp_sv q[5], // input: gauge
                    fptype_amp_sv n[5] )      // output: direction
 {
   const fptype_amp_sv qabs2 = q[1].real()*q[1].real()
                       + q[2].real()*q[2].real()
                       + q[3].real()*q[3].real();

   const fptype_amp_sv one = 1. + fptype_amp_sv{0};
   const fptype_amp_sv zero = 0. + fptype_amp_sv{0};

#ifndef MGONGPU_CPPSIMD

    if (qabs2 > 0.f)
    {
      const fptype_amp_sv qabs = fpsqrt(qabs2);

      n[0] = fpternary( q[0].real() >= 0.f , one , -one);
      n[1] = -q[1].real() / qabs;
      n[2] = -q[2].real() / qabs;
      n[3] = -q[3].real() / qabs;
      n[4] = zero;
    }
    else
    {
      n[0] = fpternary( q[0].real() >= 0.f , one , -one );
      n[1] = zero;
      n[2] = zero;
      n[3] = fpternary( q[0].real() >= 0.f , -one , one); // -sign(q0), as in fortran and python define_gauge_dir
      n[4] = zero;
    }
#else
    const fptype_amp_sv qabs = fpsqrt(qabs2);
    const bool_v qsign = (qabs2 > 0.f);
    n[0] = fpternary( q[0].real() >= 0.f , one , -one);
    n[1] = fpternary( qsign , -q[1].real() / qabs , zero );
    n[2] = fpternary( qsign , -q[2].real() / qabs , zero );
    n[3] = fpternary( qsign , -q[3].real() / qabs , fpternary( q[0].real() >= 0.f , -one , one));
    n[4] = zero;
#endif
 }

//--------------------------------------------------------------------------
// Compute propagator factor d  from the gauge q[5] and mass
// Genuinely inline (unlike INLINE elsewhere): defined here, not in helas.cu.
  __host__ __device__ inline void
  calculate_propagator_factor( const cxtype_amp_sv q[5], // input: gauge
                               const fptype_amp_sv mass,    // input: mass
                               fptype_sv *d )        // output: propagator factor
  {
    const fptype_amp_sv one = 1. + fptype_amp_sv{0};
    const fptype_amp_sv  q2 = q[0].real()*q[0].real() - ( q[1].real()*q[1].real() + q[2].real()*q[2].real() + q[3].real()*q[3].real() );
    *d = one / (q2 - mass*mass);
  }

//--------------------------------------------------------------------------
// Multiply the wavefunction by propagator factor from momenta and m
// TODO: check if d should not be used
  template< class W_ACCESS>
  __host__ __device__ INLINE void
  multiply_propagator_factor( const ALOHAOBJ & Ain, // input: wavefunctions
                              const fptype_amp m,       // input: mass
                              ALOHAOBJ Aout )       // output: wavefunctions
  {

    const cxtype_amp_sv* win = W_ACCESS::kernelAccessConst( Ain.w );
    cxtype_amp_sv* wout = W_ACCESS::kernelAccess( Aout.w );

    cxtype_amp_sv q[5];
    fptype_amp_sv n[5];
    cxtype_amp_sv w0[5], w1[5];

    const cxtype_amp_sv cI = cxmake( 0. + fptype_amp_sv{ 0 },  1. + fptype_amp_sv{ 0 }  );

    // Construct q from momenta
    q[0] = cxmake( fpamp_of_mom( -Ain.pvec[0] ), 0. );
    q[1] = cxmake( fpamp_of_mom( -Ain.pvec[1] ), 0. );
    q[2] = cxmake( fpamp_of_mom( -Ain.pvec[2] ), 0. );
    q[3] = cxmake( fpamp_of_mom( -Ain.pvec[3] ), 0. );
    q[4] = -cI*m;

    // Copy the momenta 
    Aout.pvec[0] = Ain.pvec[0];
    Aout.pvec[0] = Ain.pvec[0];
    Aout.pvec[0] = Ain.pvec[0];
    Aout.pvec[0] = Ain.pvec[0];

    define_gauge_dir(q, n);

    w0[0] = win[0];
    w0[1] = win[1];
    w0[2] = win[2];
    w0[3] = win[3];
    w0[4] = win[4];

    fptype_amp_sv nq =
          n[0]*q[0].real()
        - n[1]*q[1].real()
        - n[2]*q[2].real()
        - n[3]*q[3].real();

    cxtype_amp_sv js1 =
        ( n[0]*w0[0]
        - n[1]*w0[1]
        - n[2]*w0[2]
        - n[3]*w0[3] ) / nq;

    cxtype_amp_sv js2 =
        ( q[0]*w0[0]
        - q[1]*w0[1]
        - q[2]*w0[2]
        - q[3]*w0[3]
        - cxconj(q[4]) * w0[4] ) / nq ;

    w1[0] = w0[0] - q[0]*js1 - n[0]*js2;
    w1[1] = w0[1] - q[1]*js1 - n[1]*js2;
    w1[2] = w0[2] - q[2]*js1 - n[2]*js2;
    w1[3] = w0[3] - q[3]*js1 - n[3]*js2;
    w1[4] = w0[4] - q[4]*js1 - n[4]*js2;

    wout[0] = w1[0];
    wout[1] = w1[1];
    wout[2] = w1[2];
    wout[3] = w1[3];
    wout[4] = w1[4];
  }
  //--------------------------------------------------------------------------
  //==========================================================================

  // Compute the output wavefunction 'V3[6]' from the input wavefunctions 
  template<class W_ACCESS, class C_ACCESS>
  __device__ INLINE void
  FFV1MP0_3( const ALOHAOBJ  & F1,
             const ALOHAOBJ  & F2,
             const FLV_COUPLING_VIEW &MCOUP,
             const fptype Ccoeff,
             const fptype & M3,
             const fptype & W3,
             ALOHAOBJ  & V3 ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction 'V3[6]' from the input wavefunctions 
  template<class W_ACCESS, class C_ACCESS>
  __device__ INLINE void
  FFV6M_3( const ALOHAOBJ  & F1,
           const ALOHAOBJ  & F2,
           const FLV_COUPLING_VIEW &MCOUP,
           const fptype Ccoeff,
           const fptype & M3,
           const fptype & W3,
           ALOHAOBJ  & V3 ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------
  // Compute the output wavefunction 'V3[6]' from the input wavefunctions 
  template<class W_ACCESS, class C_ACCESS>
  __device__ INLINE void
  FFV6_2M_3( const ALOHAOBJ  & F1,
             const ALOHAOBJ  & F2,
             const FLV_COUPLING_VIEW &MCOUP1,
             const fptype Ccoeff1,
             const FLV_COUPLING_VIEW &MCOUP2,
             const fptype Ccoeff2,
             const fptype & M3,
             const fptype & W3,
             ALOHAOBJ  & V3 ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ INLINE void
  FFV2M_0( const ALOHAOBJ  & F1,
           const ALOHAOBJ  & F2,
           const ALOHAOBJ  & V3,
           const FLV_COUPLING_VIEW &MCOUP,
           const fptype Ccoeff,
           fptype_amp allvertexes[] ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction 'F2[6]' from the input wavefunctions 
  template<class W_ACCESS, class C_ACCESS>
  __device__ INLINE void
  FFV2M_2( const ALOHAOBJ  & F1,
           const ALOHAOBJ  & V3,
           const FLV_COUPLING_VIEW &MCOUP,
           const fptype Ccoeff,
           const fptype & M2,
           const fptype & W2,
           ALOHAOBJ  & F2 ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction 'V3[6]' from the input wavefunctions 
  template<class W_ACCESS, class C_ACCESS>
  __device__ INLINE void
  FFV2M_3( const ALOHAOBJ  & F1,
           const ALOHAOBJ  & F2,
           const FLV_COUPLING_VIEW &MCOUP,
           const fptype Ccoeff,
           const fptype & M3,
           const fptype & W3,
           ALOHAOBJ  & V3 ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ INLINE void
  VVV1_0( const ALOHAOBJ  & V1,
          const ALOHAOBJ  & V2,
          const ALOHAOBJ  & V3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------
  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ INLINE void
  VVV1_VVS1_VSV2_VSS1_0( const ALOHAOBJ  & V1,
                         const ALOHAOBJ  & V2,
                         const ALOHAOBJ  & V3,
                         const fptype allCOUP1[],
                         const fptype Ccoeff1,
                         const fptype allCOUP2[],
                         const fptype Ccoeff2,
                         const fptype allCOUP3[],
                         const fptype Ccoeff3,
                         const fptype allCOUP4[],
                         const fptype Ccoeff4,
                         fptype_amp allvertexes[] ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------
  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ INLINE void
  VVV1_VSV2_VSS2_SVV2_SVS2_SSV3_0( const ALOHAOBJ  & V1,
                                   const ALOHAOBJ  & V2,
                                   const ALOHAOBJ  & V3,
                                   const fptype allCOUP1[],
                                   const fptype Ccoeff1,
                                   const fptype allCOUP2[],
                                   const fptype Ccoeff2,
                                   const fptype allCOUP3[],
                                   const fptype Ccoeff3,
                                   const fptype allCOUP4[],
                                   const fptype Ccoeff4,
                                   const fptype allCOUP5[],
                                   const fptype Ccoeff5,
                                   const fptype allCOUP6[],
                                   const fptype Ccoeff6,
                                   fptype_amp allvertexes[] ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ INLINE void
  VVS1_0( const ALOHAOBJ  & V1,
          const ALOHAOBJ  & V2,
          const ALOHAOBJ  & S3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ INLINE void
  VSV2_0( const ALOHAOBJ  & V1,
          const ALOHAOBJ  & S2,
          const ALOHAOBJ  & V3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ INLINE void
  VSS1_0( const ALOHAOBJ  & V1,
          const ALOHAOBJ  & S2,
          const ALOHAOBJ  & S3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ INLINE void
  VSS2_0( const ALOHAOBJ  & V1,
          const ALOHAOBJ  & S2,
          const ALOHAOBJ  & S3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ INLINE void
  SVV2_0( const ALOHAOBJ  & S1,
          const ALOHAOBJ  & V2,
          const ALOHAOBJ  & V3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ INLINE void
  SVS2_0( const ALOHAOBJ  & S1,
          const ALOHAOBJ  & V2,
          const ALOHAOBJ  & S3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ INLINE void
  SSV3_0( const ALOHAOBJ  & S1,
          const ALOHAOBJ  & S2,
          const ALOHAOBJ  & V3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] ) ALWAYS_INLINE;

  //==========================================================================

  // Compute the output wavefunction 'V3[6]' from the input wavefunctions 
  template<class W_ACCESS, class C_ACCESS>
  __device__ void
  FFV1MP0_3( const ALOHAOBJ  & F1,
             const ALOHAOBJ  & F2,
             const FLV_COUPLING_VIEW &MCOUP,
             const fptype Ccoeff,
             const fptype & M3,
             const fptype & W3,
             ALOHAOBJ  & V3 )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wF1 = W_ACCESS::kernelAccessConst( F1.w );
    const cxtype_amp_sv* wF2 = W_ACCESS::kernelAccessConst( F2.w );
    cxtype_sv COUP;
    cxtype_amp_sv* wV3 = W_ACCESS::kernelAccess( V3.w );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    V3.pvec[0] = +F1.pvec[0] + F2.pvec[0];
    V3.pvec[1] = +F1.pvec[1] + F2.pvec[1];
    V3.pvec[2] = +F1.pvec[2] + F2.pvec[2];
    V3.pvec[3] = +F1.pvec[3] + F2.pvec[3];
    const fptype_amp_sv P3[4] = { fpamp_of_mom(-V3.pvec[0]), fpamp_of_mom(-V3.pvec[1]), fpamp_of_mom(-V3.pvec[2]), fpamp_of_mom(-V3.pvec[3]) };
    const fptype_denom_sv dP3[4] = { -V3.pvec[0], -V3.pvec[1], -V3.pvec[2], -V3.pvec[3] };
    wV3[0] = CZERO ;
    wV3[1] = CZERO ;
    wV3[2] = CZERO ;
    wV3[3] = CZERO ;
    wV3[4] = CZERO ;
    cxtype_sv FDQ[5] = { cxmake( fpamp_of_mom( -V3.pvec[0] ), 0. ), cxmake( fpamp_of_mom( -V3.pvec[1] ), 0. ), cxmake( fpamp_of_mom( -V3.pvec[2] ), 0. ), cxmake( fpamp_of_mom( -V3.pvec[3] ), 0. ), cxmake( fptype_sv{ 0 }, -M3 + fptype_sv{ 0 } ) };
    fptype_sv FDN[5];
    define_gauge_dir( FDQ, FDN );
    const fptype_sv FDNQ = FDN[0] * FDQ[0].real() - FDN[1] * FDQ[1].real() - FDN[2] * FDQ[2].real() - FDN[3] * FDQ[3].real();
    const int & flv_index1 = F1.flv_index;
    const int & flv_index2 = F2.flv_index;
    if(flv_index1 == -1 || flv_index2 == -1) {
      for(int i=0; i<V3.np4; i++) { wV3[i] = cxzero_sv(); }
      return;
    }
    int flv_sel = -1;
    if(MCOUP.partner1[flv_index1] == flv_index2) flv_sel = flv_index1;
    else if(MCOUP.partner1[flv_index2] == flv_index1) flv_sel = flv_index2;
    if(flv_sel == -1) {
      for(int i=0; i<V3.np4; i++) { wV3[i] = cxzero_sv(); }
      return;
    }
    COUP = C_ACCESS::kernelAccessConst( MCOUP.value + C_ACCESS::flv_stride*flv_sel );
    const cxtype_amp_sv denom = madmatrix::denom( Ccoeff, COUP, dP3, M3, W3 );
    wV3[0] = denom * ( -cI ) * ( wF2[2] * wF1[0] + wF2[3] * wF1[1] + wF2[0] * wF1[2] + wF2[1] * wF1[3] );
    wV3[1] = denom * ( -cI ) * ( -wF2[3] * wF1[0] - wF2[2] * wF1[1] + wF2[1] * wF1[2] + wF2[0] * wF1[3] );
    wV3[2] = denom * ( -cI ) * ( -cI * ( wF2[3] * wF1[0] + wF2[0] * wF1[3] ) + cI * ( wF2[2] * wF1[1] + wF2[1] * wF1[2] ) );
    wV3[3] = denom * ( -cI ) * ( -wF2[2] * wF1[0] - wF2[1] * wF1[3] + wF2[3] * wF1[1] + wF2[0] * wF1[2] );
    const cxtype_sv FDJS1 = ( FDN[0] * wV3[0] - FDN[1] * wV3[1] - FDN[2] * wV3[2] - FDN[3] * wV3[3] ) / FDNQ;
    const cxtype_sv FDJS2 = ( FDQ[0] * wV3[0] - FDQ[1] * wV3[1] - FDQ[2] * wV3[2] - FDQ[3] * wV3[3] - cxconj( FDQ[4] ) * wV3[4] ) / FDNQ;
    wV3[0] = wV3[0] - FDQ[0] * FDJS1 - FDN[0] * FDJS2;
    wV3[1] = wV3[1] - FDQ[1] * FDJS1 - FDN[1] * FDJS2;
    wV3[2] = wV3[2] - FDQ[2] * FDJS1 - FDN[2] * FDJS2;
    wV3[3] = wV3[3] - FDQ[3] * FDJS1 - FDN[3] * FDJS2;
    wV3[4] = wV3[4] - FDQ[4] * FDJS1 - FDN[4] * FDJS2;
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction 'V3[6]' from the input wavefunctions 
  template<class W_ACCESS, class C_ACCESS>
  __device__ void
  FFV6M_3( const ALOHAOBJ  & F1,
           const ALOHAOBJ  & F2,
           const FLV_COUPLING_VIEW &MCOUP,
           const fptype Ccoeff,
           const fptype & M3,
           const fptype & W3,
           ALOHAOBJ  & V3 )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wF1 = W_ACCESS::kernelAccessConst( F1.w );
    const cxtype_amp_sv* wF2 = W_ACCESS::kernelAccessConst( F2.w );
    cxtype_sv COUP;
    cxtype_amp_sv* wV3 = W_ACCESS::kernelAccess( V3.w );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    V3.pvec[0] = +F1.pvec[0] + F2.pvec[0];
    V3.pvec[1] = +F1.pvec[1] + F2.pvec[1];
    V3.pvec[2] = +F1.pvec[2] + F2.pvec[2];
    V3.pvec[3] = +F1.pvec[3] + F2.pvec[3];
    const fptype_amp_sv P3[4] = { fpamp_of_mom(-V3.pvec[0]), fpamp_of_mom(-V3.pvec[1]), fpamp_of_mom(-V3.pvec[2]), fpamp_of_mom(-V3.pvec[3]) };
    const fptype_denom_sv dP3[4] = { -V3.pvec[0], -V3.pvec[1], -V3.pvec[2], -V3.pvec[3] };
    wV3[0] = CZERO ;
    wV3[1] = CZERO ;
    wV3[2] = CZERO ;
    wV3[3] = CZERO ;
    wV3[4] = CZERO ;
    cxtype_sv FDQ[5] = { cxmake( fpamp_of_mom( -V3.pvec[0] ), 0. ), cxmake( fpamp_of_mom( -V3.pvec[1] ), 0. ), cxmake( fpamp_of_mom( -V3.pvec[2] ), 0. ), cxmake( fpamp_of_mom( -V3.pvec[3] ), 0. ), cxmake( fptype_sv{ 0 }, -M3 + fptype_sv{ 0 } ) };
    fptype_sv FDN[5];
    define_gauge_dir( FDQ, FDN );
    const fptype_sv FDNQ = FDN[0] * FDQ[0].real() - FDN[1] * FDQ[1].real() - FDN[2] * FDQ[2].real() - FDN[3] * FDQ[3].real();
    const int & flv_index1 = F1.flv_index;
    const int & flv_index2 = F2.flv_index;
    if(flv_index1 == -1 || flv_index2 == -1) {
      for(int i=0; i<V3.np4; i++) { wV3[i] = cxzero_sv(); }
      return;
    }
    int flv_sel = -1;
    if(MCOUP.partner1[flv_index1] == flv_index2) flv_sel = flv_index1;
    else if(MCOUP.partner1[flv_index2] == flv_index1) flv_sel = flv_index2;
    if(flv_sel == -1) {
      for(int i=0; i<V3.np4; i++) { wV3[i] = cxzero_sv(); }
      return;
    }
    COUP = C_ACCESS::kernelAccessConst( MCOUP.value + C_ACCESS::flv_stride*flv_sel );
    const cxtype_amp_sv denom = madmatrix::denom( Ccoeff, COUP, dP3, M3, W3 );
    wV3[0] = denom * ( -cI ) * ( wF2[0] * wF1[2] + wF2[1] * wF1[3] );
    wV3[1] = denom * ( -cI ) * ( wF2[1] * wF1[2] + wF2[0] * wF1[3] );
    wV3[2] = denom * ( -cI ) * ( +cI * ( wF2[1] * wF1[2] ) - cI * ( wF2[0] * wF1[3] ) );
    wV3[3] = denom * ( -cI ) * ( wF2[0] * wF1[2] - wF2[1] * wF1[3] );
    const cxtype_sv FDJS1 = ( FDN[0] * wV3[0] - FDN[1] * wV3[1] - FDN[2] * wV3[2] - FDN[3] * wV3[3] ) / FDNQ;
    const cxtype_sv FDJS2 = ( FDQ[0] * wV3[0] - FDQ[1] * wV3[1] - FDQ[2] * wV3[2] - FDQ[3] * wV3[3] - cxconj( FDQ[4] ) * wV3[4] ) / FDNQ;
    wV3[0] = wV3[0] - FDQ[0] * FDJS1 - FDN[0] * FDJS2;
    wV3[1] = wV3[1] - FDQ[1] * FDJS1 - FDN[1] * FDJS2;
    wV3[2] = wV3[2] - FDQ[2] * FDJS1 - FDN[2] * FDJS2;
    wV3[3] = wV3[3] - FDQ[3] * FDJS1 - FDN[3] * FDJS2;
    wV3[4] = wV3[4] - FDQ[4] * FDJS1 - FDN[4] * FDJS2;
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------
  // Compute the output wavefunction 'V3[6]' from the input wavefunctions 
  template<class W_ACCESS, class C_ACCESS>
  __device__ void
  FFV6_2M_3( const ALOHAOBJ  & F1,
             const ALOHAOBJ  & F2,
             const FLV_COUPLING_VIEW &MCOUP1,
             const fptype Ccoeff1,
             const FLV_COUPLING_VIEW &MCOUP2,
             const fptype Ccoeff2,
             const fptype & M3,
             const fptype & W3,
             ALOHAOBJ  & V3 )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wF1 = W_ACCESS::kernelAccessConst( F1.w );
    const cxtype_amp_sv* wF2 = W_ACCESS::kernelAccessConst( F2.w );
    cxtype_sv COUP1;
    cxtype_sv COUP2;
    cxtype_amp_sv* wV3 = W_ACCESS::kernelAccess( V3.w );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    V3.pvec[0] = +F1.pvec[0] + F2.pvec[0];
    V3.pvec[1] = +F1.pvec[1] + F2.pvec[1];
    V3.pvec[2] = +F1.pvec[2] + F2.pvec[2];
    V3.pvec[3] = +F1.pvec[3] + F2.pvec[3];
    const fptype_amp_sv P3[4] = { fpamp_of_mom(-V3.pvec[0]), fpamp_of_mom(-V3.pvec[1]), fpamp_of_mom(-V3.pvec[2]), fpamp_of_mom(-V3.pvec[3]) };
    const fptype_denom_sv dP3[4] = { -V3.pvec[0], -V3.pvec[1], -V3.pvec[2], -V3.pvec[3] };
    wV3[0] = CZERO ;
    wV3[1] = CZERO ;
    wV3[2] = CZERO ;
    wV3[3] = CZERO ;
    wV3[4] = CZERO ;
    cxtype_sv FDQ[5] = { cxmake( fpamp_of_mom( -V3.pvec[0] ), 0. ), cxmake( fpamp_of_mom( -V3.pvec[1] ), 0. ), cxmake( fpamp_of_mom( -V3.pvec[2] ), 0. ), cxmake( fpamp_of_mom( -V3.pvec[3] ), 0. ), cxmake( fptype_sv{ 0 }, -M3 + fptype_sv{ 0 } ) };
    fptype_sv FDN[5];
    define_gauge_dir( FDQ, FDN );
    const fptype_sv FDNQ = FDN[0] * FDQ[0].real() - FDN[1] * FDQ[1].real() - FDN[2] * FDQ[2].real() - FDN[3] * FDQ[3].real();
    const int & flv_index1 = F1.flv_index;
    const int & flv_index2 = F2.flv_index;
    int zero_coup1 = 0;
    int zero_coup2 = 0;
    if(flv_index1 != flv_index2 || flv_index1 == -1) {
      for(int i=0; i<V3.np4; i++) { wV3[i] = cxzero_sv(); }
      return;
    }
    if(flv_index1 == -1 || flv_index2 == -1) {
      for(int i=0; i<V3.np4; i++) { wV3[i] = cxzero_sv(); }
      return;
    }
    if(MCOUP1.partner1[flv_index1] != flv_index2 || MCOUP1.partner2[flv_index1] != flv_index2) {
      zero_coup1 = 1;
      COUP1 = cxzero_sv();
    }
    if(MCOUP2.partner1[flv_index1] != flv_index2 || MCOUP2.partner2[flv_index1] != flv_index2) {
      zero_coup2 = 1;
      COUP2 = cxzero_sv();
    }
    if(zero_coup1 ==0) { COUP1 = C_ACCESS::kernelAccessConst( MCOUP1.value + C_ACCESS::flv_stride*flv_index1 ); }
    if(zero_coup2 ==0) { COUP2 = C_ACCESS::kernelAccessConst( MCOUP2.value + C_ACCESS::flv_stride*flv_index1 ); }
    const cxtype_amp_sv denom1 = madmatrix::denom( Ccoeff1, COUP1, dP3, M3, W3 );
    wV3[0] = wV3[0] + denom1 * ( -cI ) * ( wF2[0] * wF1[2] + wF2[1] * wF1[3] );
    wV3[1] = wV3[1] + denom1 * ( -cI ) * ( wF2[1] * wF1[2] + wF2[0] * wF1[3] );
    wV3[2] = wV3[2] + denom1 * ( -cI ) * ( +cI * ( wF2[1] * wF1[2] ) - cI * ( wF2[0] * wF1[3] ) );
    wV3[3] = wV3[3] + denom1 * ( -cI ) * ( wF2[0] * wF1[2] - wF2[1] * wF1[3] );
    const cxtype_amp_sv denom2 = madmatrix::denom( Ccoeff2, COUP2, dP3, M3, W3 );
    wV3[0] = wV3[0] + denom2 * ( -cI ) * ( wF2[2] * wF1[0] + wF2[3] * wF1[1] );
    wV3[1] = wV3[1] + denom2 * ( -cI ) * ( -wF2[3] * wF1[0] - wF2[2] * wF1[1] );
    wV3[2] = wV3[2] + denom2 * ( -cI ) * ( -cI * ( wF2[3] * wF1[0] ) + cI * ( wF2[2] * wF1[1] ) );
    wV3[3] = wV3[3] + denom2 * ( -cI ) * ( -wF2[2] * wF1[0] + wF2[3] * wF1[1] );
    const cxtype_sv FDJS1 = ( FDN[0] * wV3[0] - FDN[1] * wV3[1] - FDN[2] * wV3[2] - FDN[3] * wV3[3] ) / FDNQ;
    const cxtype_sv FDJS2 = ( FDQ[0] * wV3[0] - FDQ[1] * wV3[1] - FDQ[2] * wV3[2] - FDQ[3] * wV3[3] - cxconj( FDQ[4] ) * wV3[4] ) / FDNQ;
    wV3[0] = wV3[0] - FDQ[0] * FDJS1 - FDN[0] * FDJS2;
    wV3[1] = wV3[1] - FDQ[1] * FDJS1 - FDN[1] * FDJS2;
    wV3[2] = wV3[2] - FDQ[2] * FDJS1 - FDN[2] * FDJS2;
    wV3[3] = wV3[3] - FDQ[3] * FDJS1 - FDN[3] * FDJS2;
    wV3[4] = wV3[4] - FDQ[4] * FDJS1 - FDN[4] * FDJS2;
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------
  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ void
  FFV2M_0( const ALOHAOBJ  & F1,
           const ALOHAOBJ  & F2,
           const ALOHAOBJ  & V3,
           const FLV_COUPLING_VIEW &MCOUP,
           const fptype Ccoeff,
           fptype_amp allvertexes[] )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wF1 = W_ACCESS::kernelAccessConst( F1.w );
    const cxtype_amp_sv* wF2 = W_ACCESS::kernelAccessConst( F2.w );
    const cxtype_amp_sv* wV3 = W_ACCESS::kernelAccessConst( V3.w );
    cxtype_sv COUP;
    cxtype_amp_sv* vertex = A_ACCESS::kernelAccess( allvertexes );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    const int & flv_index1 = F1.flv_index;
    const int & flv_index2 = F2.flv_index;
    if(flv_index1 == -1 || flv_index2 == -1) {
      *vertex = cxzero_sv();
      return;
    }
    int flv_sel = -1;
    if(MCOUP.partner1[flv_index1] == flv_index2) flv_sel = flv_index1;
    else if(MCOUP.partner1[flv_index2] == flv_index1) flv_sel = flv_index2;
    if(flv_sel == -1) {
      *vertex = cxzero_sv();
      return;
    }
    COUP = C_ACCESS::kernelAccessConst( MCOUP.value + C_ACCESS::flv_stride*flv_sel );
    const cxtype_amp_sv TMP0 = ( wF1[0] * ( wF2[2] * ( wV3[0] + wV3[3] ) + wF2[3] * ( wV3[1] + cI * wV3[2] ) ) + wF1[1] * ( wF2[2] * ( wV3[1] - cI * wV3[2] ) + wF2[3] * ( wV3[0] - wV3[3] ) ) );
    ( *vertex ) = (cxtype_amp_sv)( Ccoeff * COUP * -cI * TMP0 );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction 'F2[6]' from the input wavefunctions 
  template<class W_ACCESS, class C_ACCESS>
  __device__ void
  FFV2M_2( const ALOHAOBJ  & F1,
           const ALOHAOBJ  & V3,
           const FLV_COUPLING_VIEW &MCOUP,
           const fptype Ccoeff,
           const fptype & M2,
           const fptype & W2,
           ALOHAOBJ  & F2 )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wF1 = W_ACCESS::kernelAccessConst( F1.w );
    const cxtype_amp_sv* wV3 = W_ACCESS::kernelAccessConst( V3.w );
    cxtype_sv COUP;
    cxtype_amp_sv* wF2 = W_ACCESS::kernelAccess( F2.w );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    F2.pvec[0] = +F1.pvec[0] + V3.pvec[0];
    F2.pvec[1] = +F1.pvec[1] + V3.pvec[1];
    F2.pvec[2] = +F1.pvec[2] + V3.pvec[2];
    F2.pvec[3] = +F1.pvec[3] + V3.pvec[3];
    const fptype_amp_sv P2[4] = { fpamp_of_mom(-F2.pvec[0]), fpamp_of_mom(-F2.pvec[1]), fpamp_of_mom(-F2.pvec[2]), fpamp_of_mom(-F2.pvec[3]) };
    const fptype_denom_sv dP2[4] = { -F2.pvec[0], -F2.pvec[1], -F2.pvec[2], -F2.pvec[3] };
    int flv_index1 = F1.flv_index;
    if(flv_index1 == -1) {
      for(int i=0; i<F2.nw6; i++) { wF2[i] = cxzero_sv(); }
      F2.flv_index = -1;
      return;
    }
    int flv_index2 = MCOUP.partner1[flv_index1];
    if(flv_index2 == -1){
      for(int i=0; i<F2.nw6; i++) { wF2[i] = cxzero_sv(); }
      F2.flv_index = -1;
      return;
    }
    F2.flv_index = flv_index2;
    COUP = C_ACCESS::kernelAccessConst( MCOUP.value + C_ACCESS::flv_stride*flv_index1 );
    constexpr fptype_amp one( 1. );
    constexpr fptype_denom oned( 1. );
    const cxtype_amp_sv denom = madmatrix::denom( Ccoeff, COUP, dP2, M2, W2 );
    wF2[0] = denom * cI * ( wF1[0] * ( P2[0] * ( wV3[0] + wV3[3] ) + ( P2[1] * ( -one ) * ( wV3[1] + cI * wV3[2] ) + ( P2[2] * ( +cI * wV3[1] - wV3[2] ) - P2[3] * ( wV3[0] + wV3[3] ) ) ) ) + wF1[1] * ( P2[0] * ( wV3[1] - cI * wV3[2] ) + ( P2[1] * ( -wV3[0] + wV3[3] ) + ( P2[2] * ( +cI * wV3[0] - cI * wV3[3] ) + P2[3] * ( -wV3[1] + cI * wV3[2] ) ) ) ) );
    wF2[1] = denom * cI * ( wF1[0] * ( P2[0] * ( wV3[1] + cI * wV3[2] ) + ( P2[1] * ( -one ) * ( wV3[0] + wV3[3] ) + ( P2[2] * ( -one ) * ( +cI * ( wV3[0] + wV3[3] ) ) + P2[3] * ( wV3[1] + cI * wV3[2] ) ) ) ) + wF1[1] * ( P2[0] * ( wV3[0] - wV3[3] ) + ( P2[1] * ( -wV3[1] + cI * wV3[2] ) + ( P2[2] * ( -one ) * ( +cI * wV3[1] + wV3[2] ) + P2[3] * ( wV3[0] - wV3[3] ) ) ) ) );
    wF2[2] = denom * -cI * M2 * ( wF1[0] * ( -one ) * ( wV3[0] + wV3[3] ) + wF1[1] * ( -wV3[1] + cI * wV3[2] ) );
    wF2[3] = denom * cI * M2 * ( wF1[0] * ( wV3[1] + cI * wV3[2] ) + wF1[1] * ( wV3[0] - wV3[3] ) );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction 'V3[6]' from the input wavefunctions 
  template<class W_ACCESS, class C_ACCESS>
  __device__ void
  FFV2M_3( const ALOHAOBJ  & F1,
           const ALOHAOBJ  & F2,
           const FLV_COUPLING_VIEW &MCOUP,
           const fptype Ccoeff,
           const fptype & M3,
           const fptype & W3,
           ALOHAOBJ  & V3 )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wF1 = W_ACCESS::kernelAccessConst( F1.w );
    const cxtype_amp_sv* wF2 = W_ACCESS::kernelAccessConst( F2.w );
    cxtype_sv COUP;
    cxtype_amp_sv* wV3 = W_ACCESS::kernelAccess( V3.w );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    V3.pvec[0] = +F1.pvec[0] + F2.pvec[0];
    V3.pvec[1] = +F1.pvec[1] + F2.pvec[1];
    V3.pvec[2] = +F1.pvec[2] + F2.pvec[2];
    V3.pvec[3] = +F1.pvec[3] + F2.pvec[3];
    const fptype_amp_sv P3[4] = { fpamp_of_mom(-V3.pvec[0]), fpamp_of_mom(-V3.pvec[1]), fpamp_of_mom(-V3.pvec[2]), fpamp_of_mom(-V3.pvec[3]) };
    const fptype_denom_sv dP3[4] = { -V3.pvec[0], -V3.pvec[1], -V3.pvec[2], -V3.pvec[3] };
    wV3[0] = CZERO ;
    wV3[1] = CZERO ;
    wV3[2] = CZERO ;
    wV3[3] = CZERO ;
    wV3[4] = CZERO ;
    cxtype_sv FDQ[5] = { cxmake( fpamp_of_mom( -V3.pvec[0] ), 0. ), cxmake( fpamp_of_mom( -V3.pvec[1] ), 0. ), cxmake( fpamp_of_mom( -V3.pvec[2] ), 0. ), cxmake( fpamp_of_mom( -V3.pvec[3] ), 0. ), cxmake( fptype_sv{ 0 }, -M3 + fptype_sv{ 0 } ) };
    fptype_sv FDN[5];
    define_gauge_dir( FDQ, FDN );
    const fptype_sv FDNQ = FDN[0] * FDQ[0].real() - FDN[1] * FDQ[1].real() - FDN[2] * FDQ[2].real() - FDN[3] * FDQ[3].real();
    const int & flv_index1 = F1.flv_index;
    const int & flv_index2 = F2.flv_index;
    if(flv_index1 == -1 || flv_index2 == -1) {
      for(int i=0; i<V3.np4; i++) { wV3[i] = cxzero_sv(); }
      return;
    }
    int flv_sel = -1;
    if(MCOUP.partner1[flv_index1] == flv_index2) flv_sel = flv_index1;
    else if(MCOUP.partner1[flv_index2] == flv_index1) flv_sel = flv_index2;
    if(flv_sel == -1) {
      for(int i=0; i<V3.np4; i++) { wV3[i] = cxzero_sv(); }
      return;
    }
    COUP = C_ACCESS::kernelAccessConst( MCOUP.value + C_ACCESS::flv_stride*flv_sel );
    const cxtype_amp_sv denom = madmatrix::denom( Ccoeff, COUP, dP3, M3, W3 );
    wV3[0] = denom * ( -cI ) * ( wF2[2] * wF1[0] + wF2[3] * wF1[1] );
    wV3[1] = denom * ( -cI ) * ( -wF2[3] * wF1[0] - wF2[2] * wF1[1] );
    wV3[2] = denom * ( -cI ) * ( -cI * ( wF2[3] * wF1[0] ) + cI * ( wF2[2] * wF1[1] ) );
    wV3[3] = denom * ( -cI ) * ( -wF2[2] * wF1[0] + wF2[3] * wF1[1] );
    const cxtype_sv FDJS1 = ( FDN[0] * wV3[0] - FDN[1] * wV3[1] - FDN[2] * wV3[2] - FDN[3] * wV3[3] ) / FDNQ;
    const cxtype_sv FDJS2 = ( FDQ[0] * wV3[0] - FDQ[1] * wV3[1] - FDQ[2] * wV3[2] - FDQ[3] * wV3[3] - cxconj( FDQ[4] ) * wV3[4] ) / FDNQ;
    wV3[0] = wV3[0] - FDQ[0] * FDJS1 - FDN[0] * FDJS2;
    wV3[1] = wV3[1] - FDQ[1] * FDJS1 - FDN[1] * FDJS2;
    wV3[2] = wV3[2] - FDQ[2] * FDJS1 - FDN[2] * FDJS2;
    wV3[3] = wV3[3] - FDQ[3] * FDJS1 - FDN[3] * FDJS2;
    wV3[4] = wV3[4] - FDQ[4] * FDJS1 - FDN[4] * FDJS2;
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ void
  VVV1_0( const ALOHAOBJ  & V1,
          const ALOHAOBJ  & V2,
          const ALOHAOBJ  & V3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wV1 = W_ACCESS::kernelAccessConst( V1.w );
    const cxtype_amp_sv* wV2 = W_ACCESS::kernelAccessConst( V2.w );
    const cxtype_amp_sv* wV3 = W_ACCESS::kernelAccessConst( V3.w );
    const cxtype_sv COUP = C_ACCESS::kernelAccessConst( allCOUP );
    cxtype_amp_sv* vertex = A_ACCESS::kernelAccess( allvertexes );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    const fptype_amp_sv P1[4] = { fpamp_of_mom(+V1.pvec[0]), fpamp_of_mom(+V1.pvec[1]), fpamp_of_mom(+V1.pvec[2]), fpamp_of_mom(+V1.pvec[3]) };
    const fptype_amp_sv P2[4] = { fpamp_of_mom(+V2.pvec[0]), fpamp_of_mom(+V2.pvec[1]), fpamp_of_mom(+V2.pvec[2]), fpamp_of_mom(+V2.pvec[3]) };
    const fptype_amp_sv P3[4] = { fpamp_of_mom(+V3.pvec[0]), fpamp_of_mom(+V3.pvec[1]), fpamp_of_mom(+V3.pvec[2]), fpamp_of_mom(+V3.pvec[3]) };
    const cxtype_amp_sv TMP1 = ( wV2[0] * wV1[0] - wV2[1] * wV1[1] - wV2[2] * wV1[2] - wV2[3] * wV1[3] );
    const cxtype_amp_sv TMP2 = ( wV3[0] * P1[0] - wV3[1] * P1[1] - wV3[2] * P1[2] - wV3[3] * P1[3] );
    const cxtype_amp_sv TMP3 = ( wV3[0] * P2[0] - wV3[1] * P2[1] - wV3[2] * P2[2] - wV3[3] * P2[3] );
    const cxtype_amp_sv TMP4 = ( wV2[0] * P1[0] - wV2[1] * P1[1] - wV2[2] * P1[2] - wV2[3] * P1[3] );
    const cxtype_amp_sv TMP5 = ( wV3[0] * wV1[0] - wV3[1] * wV1[1] - wV3[2] * wV1[2] - wV3[3] * wV1[3] );
    const cxtype_amp_sv TMP6 = ( wV2[0] * P3[0] - wV2[1] * P3[1] - wV2[2] * P3[2] - wV2[3] * P3[3] );
    const cxtype_amp_sv TMP7 = ( wV3[0] * wV2[0] - wV3[1] * wV2[1] - wV3[2] * wV2[2] - wV3[3] * wV2[3] );
    const cxtype_amp_sv TMP8 = ( P2[0] * wV1[0] - P2[1] * wV1[1] - P2[2] * wV1[2] - P2[3] * wV1[3] );
    const cxtype_amp_sv TMP9 = ( wV1[0] * P3[0] - wV1[1] * P3[1] - wV1[2] * P3[2] - wV1[3] * P3[3] );
    ( *vertex ) = (cxtype_amp_sv)( Ccoeff * COUP * ( TMP1 * ( -cI * TMP2 + cI * TMP3 ) + ( TMP5 * ( +cI * TMP4 - cI * TMP6 ) + TMP7 * ( -cI * TMP8 + cI * TMP9 ) ) ) );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------
  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ void
  VVV1_VVS1_VSV2_VSS1_0( const ALOHAOBJ  & V1,
                         const ALOHAOBJ  & V2,
                         const ALOHAOBJ  & V3,
                         const fptype allCOUP1[],
                         const fptype Ccoeff1,
                         const fptype allCOUP2[],
                         const fptype Ccoeff2,
                         const fptype allCOUP3[],
                         const fptype Ccoeff3,
                         const fptype allCOUP4[],
                         const fptype Ccoeff4,
                         fptype_amp allvertexes[] )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wV1 = W_ACCESS::kernelAccessConst( V1.w );
    const cxtype_amp_sv* wV2 = W_ACCESS::kernelAccessConst( V2.w );
    const cxtype_amp_sv* wV3 = W_ACCESS::kernelAccessConst( V3.w );
    const cxtype_sv COUP1 = C_ACCESS::kernelAccessConst( allCOUP1 );
    const cxtype_sv COUP2 = C_ACCESS::kernelAccessConst( allCOUP2 );
    const cxtype_sv COUP3 = C_ACCESS::kernelAccessConst( allCOUP3 );
    const cxtype_sv COUP4 = C_ACCESS::kernelAccessConst( allCOUP4 );
    cxtype_amp_sv* vertex = A_ACCESS::kernelAccess( allvertexes );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    const fptype_amp_sv P1[4] = { fpamp_of_mom(+V1.pvec[0]), fpamp_of_mom(+V1.pvec[1]), fpamp_of_mom(+V1.pvec[2]), fpamp_of_mom(+V1.pvec[3]) };
    const fptype_amp_sv P2[4] = { fpamp_of_mom(+V2.pvec[0]), fpamp_of_mom(+V2.pvec[1]), fpamp_of_mom(+V2.pvec[2]), fpamp_of_mom(+V2.pvec[3]) };
    const fptype_amp_sv P3[4] = { fpamp_of_mom(+V3.pvec[0]), fpamp_of_mom(+V3.pvec[1]), fpamp_of_mom(+V3.pvec[2]), fpamp_of_mom(+V3.pvec[3]) };
    ( *vertex ) = cxzero_sv();
    const cxtype_amp_sv TMP1 = ( wV2[0] * wV1[0] - wV2[1] * wV1[1] - wV2[2] * wV1[2] - wV2[3] * wV1[3] );
    const cxtype_amp_sv TMP2 = ( wV3[0] * P1[0] - wV3[1] * P1[1] - wV3[2] * P1[2] - wV3[3] * P1[3] );
    const cxtype_amp_sv TMP3 = ( wV3[0] * P2[0] - wV3[1] * P2[1] - wV3[2] * P2[2] - wV3[3] * P2[3] );
    const cxtype_amp_sv TMP4 = ( wV2[0] * P1[0] - wV2[1] * P1[1] - wV2[2] * P1[2] - wV2[3] * P1[3] );
    const cxtype_amp_sv TMP5 = ( wV3[0] * wV1[0] - wV3[1] * wV1[1] - wV3[2] * wV1[2] - wV3[3] * wV1[3] );
    const cxtype_amp_sv TMP6 = ( wV2[0] * P3[0] - wV2[1] * P3[1] - wV2[2] * P3[2] - wV2[3] * P3[3] );
    const cxtype_amp_sv TMP7 = ( wV3[0] * wV2[0] - wV3[1] * wV2[1] - wV3[2] * wV2[2] - wV3[3] * wV2[3] );
    const cxtype_amp_sv TMP8 = ( P2[0] * wV1[0] - P2[1] * wV1[1] - P2[2] * wV1[2] - P2[3] * wV1[3] );
    const cxtype_amp_sv TMP9 = ( wV1[0] * P3[0] - wV1[1] * P3[1] - wV1[2] * P3[2] - wV1[3] * P3[3] );
    ( *vertex ) = (cxtype_amp_sv)( ( *vertex ) + Ccoeff1 * COUP1 * ( TMP1 * ( -cI * TMP2 + cI * TMP3 ) + ( TMP5 * ( +cI * TMP4 - cI * TMP6 ) + TMP7 * ( -cI * TMP8 + cI * TMP9 ) ) ) );
    ( *vertex ) = (cxtype_amp_sv)( ( *vertex ) + Ccoeff2 * COUP2 * -cI * TMP1 * wV3[4] );
    ( *vertex ) = (cxtype_amp_sv)( ( *vertex ) + Ccoeff3 * COUP3 * -cI * TMP5 * wV2[4] );
    ( *vertex ) = (cxtype_amp_sv)( ( *vertex ) + Ccoeff4 * COUP4 * wV2[4] * wV3[4] * ( -cI * TMP8 + cI * TMP9 ) );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ void
  VVV1_VSV2_VSS2_SVV2_SVS2_SSV3_0( const ALOHAOBJ  & V1,
                                   const ALOHAOBJ  & V2,
                                   const ALOHAOBJ  & V3,
                                   const fptype allCOUP1[],
                                   const fptype Ccoeff1,
                                   const fptype allCOUP2[],
                                   const fptype Ccoeff2,
                                   const fptype allCOUP3[],
                                   const fptype Ccoeff3,
                                   const fptype allCOUP4[],
                                   const fptype Ccoeff4,
                                   const fptype allCOUP5[],
                                   const fptype Ccoeff5,
                                   const fptype allCOUP6[],
                                   const fptype Ccoeff6,
                                   fptype_amp allvertexes[] )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wV1 = W_ACCESS::kernelAccessConst( V1.w );
    const cxtype_amp_sv* wV2 = W_ACCESS::kernelAccessConst( V2.w );
    const cxtype_amp_sv* wV3 = W_ACCESS::kernelAccessConst( V3.w );
    const cxtype_sv COUP1 = C_ACCESS::kernelAccessConst( allCOUP1 );
    const cxtype_sv COUP2 = C_ACCESS::kernelAccessConst( allCOUP2 );
    const cxtype_sv COUP3 = C_ACCESS::kernelAccessConst( allCOUP3 );
    const cxtype_sv COUP4 = C_ACCESS::kernelAccessConst( allCOUP4 );
    const cxtype_sv COUP5 = C_ACCESS::kernelAccessConst( allCOUP5 );
    const cxtype_sv COUP6 = C_ACCESS::kernelAccessConst( allCOUP6 );
    cxtype_amp_sv* vertex = A_ACCESS::kernelAccess( allvertexes );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    const fptype_amp_sv P1[4] = { fpamp_of_mom(+V1.pvec[0]), fpamp_of_mom(+V1.pvec[1]), fpamp_of_mom(+V1.pvec[2]), fpamp_of_mom(+V1.pvec[3]) };
    const fptype_amp_sv P2[4] = { fpamp_of_mom(+V2.pvec[0]), fpamp_of_mom(+V2.pvec[1]), fpamp_of_mom(+V2.pvec[2]), fpamp_of_mom(+V2.pvec[3]) };
    const fptype_amp_sv P3[4] = { fpamp_of_mom(+V3.pvec[0]), fpamp_of_mom(+V3.pvec[1]), fpamp_of_mom(+V3.pvec[2]), fpamp_of_mom(+V3.pvec[3]) };
    ( *vertex ) = cxzero_sv();
    const cxtype_amp_sv TMP1 = ( wV2[0] * wV1[0] - wV2[1] * wV1[1] - wV2[2] * wV1[2] - wV2[3] * wV1[3] );
    const cxtype_amp_sv TMP2 = ( wV3[0] * P1[0] - wV3[1] * P1[1] - wV3[2] * P1[2] - wV3[3] * P1[3] );
    const cxtype_amp_sv TMP3 = ( wV3[0] * P2[0] - wV3[1] * P2[1] - wV3[2] * P2[2] - wV3[3] * P2[3] );
    const cxtype_amp_sv TMP4 = ( wV2[0] * P1[0] - wV2[1] * P1[1] - wV2[2] * P1[2] - wV2[3] * P1[3] );
    const cxtype_amp_sv TMP5 = ( wV3[0] * wV1[0] - wV3[1] * wV1[1] - wV3[2] * wV1[2] - wV3[3] * wV1[3] );
    const cxtype_amp_sv TMP6 = ( wV2[0] * P3[0] - wV2[1] * P3[1] - wV2[2] * P3[2] - wV2[3] * P3[3] );
    const cxtype_amp_sv TMP7 = ( wV3[0] * wV2[0] - wV3[1] * wV2[1] - wV3[2] * wV2[2] - wV3[3] * wV2[3] );
    const cxtype_amp_sv TMP8 = ( P2[0] * wV1[0] - P2[1] * wV1[1] - P2[2] * wV1[2] - P2[3] * wV1[3] );
    const cxtype_amp_sv TMP9 = ( wV1[0] * P3[0] - wV1[1] * P3[1] - wV1[2] * P3[2] - wV1[3] * P3[3] );
    ( *vertex ) = (cxtype_amp_sv)( ( *vertex ) + Ccoeff1 * COUP1 * ( TMP1 * ( -cI * TMP2 + cI * TMP3 ) + ( TMP5 * ( +cI * TMP4 - cI * TMP6 ) + TMP7 * ( -cI * TMP8 + cI * TMP9 ) ) ) );
    ( *vertex ) = (cxtype_amp_sv)( ( *vertex ) + Ccoeff2 * COUP2 * -cI * TMP5 * wV2[4] );
    ( *vertex ) = (cxtype_amp_sv)( ( *vertex ) + Ccoeff3 * COUP3 * wV2[4] * wV3[4] * ( -cI * TMP9 + cI * TMP8 ) );
    ( *vertex ) = (cxtype_amp_sv)( ( *vertex ) + Ccoeff4 * COUP4 * -cI * TMP7 * wV1[4] );
    ( *vertex ) = (cxtype_amp_sv)( ( *vertex ) + Ccoeff5 * COUP5 * wV1[4] * wV3[4] * ( -cI * TMP6 + cI * TMP4 ) );
    ( *vertex ) = (cxtype_amp_sv)( ( *vertex ) + Ccoeff6 * COUP6 * wV1[4] * wV2[4] * ( -cI * TMP2 + cI * TMP3 ) );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------
  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ void
  VVS1_0( const ALOHAOBJ  & V1,
          const ALOHAOBJ  & V2,
          const ALOHAOBJ  & S3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wV1 = W_ACCESS::kernelAccessConst( V1.w );
    const cxtype_amp_sv* wV2 = W_ACCESS::kernelAccessConst( V2.w );
    const cxtype_amp_sv* wS3 = W_ACCESS::kernelAccessConst( S3.w );
    const cxtype_sv COUP = C_ACCESS::kernelAccessConst( allCOUP );
    cxtype_amp_sv* vertex = A_ACCESS::kernelAccess( allvertexes );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    const cxtype_amp_sv TMP1 = ( wV2[0] * wV1[0] - wV2[1] * wV1[1] - wV2[2] * wV1[2] - wV2[3] * wV1[3] );
    ( *vertex ) = (cxtype_amp_sv)( Ccoeff * COUP * -cI * TMP1 * wS3[4] );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ void
  VSV2_0( const ALOHAOBJ  & V1,
          const ALOHAOBJ  & S2,
          const ALOHAOBJ  & V3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wV1 = W_ACCESS::kernelAccessConst( V1.w );
    const cxtype_amp_sv* wS2 = W_ACCESS::kernelAccessConst( S2.w );
    const cxtype_amp_sv* wV3 = W_ACCESS::kernelAccessConst( V3.w );
    const cxtype_sv COUP = C_ACCESS::kernelAccessConst( allCOUP );
    cxtype_amp_sv* vertex = A_ACCESS::kernelAccess( allvertexes );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    const cxtype_amp_sv TMP5 = ( wV3[0] * wV1[0] - wV3[1] * wV1[1] - wV3[2] * wV1[2] - wV3[3] * wV1[3] );
    ( *vertex ) = (cxtype_amp_sv)( Ccoeff * COUP * -cI * TMP5 * wS2[4] );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ void
  VSS1_0( const ALOHAOBJ  & V1,
          const ALOHAOBJ  & S2,
          const ALOHAOBJ  & S3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wV1 = W_ACCESS::kernelAccessConst( V1.w );
    const cxtype_amp_sv* wS2 = W_ACCESS::kernelAccessConst( S2.w );
    const cxtype_amp_sv* wS3 = W_ACCESS::kernelAccessConst( S3.w );
    const cxtype_sv COUP = C_ACCESS::kernelAccessConst( allCOUP );
    cxtype_amp_sv* vertex = A_ACCESS::kernelAccess( allvertexes );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    const fptype_amp_sv P2[4] = { fpamp_of_mom(+S2.pvec[0]), fpamp_of_mom(+S2.pvec[1]), fpamp_of_mom(+S2.pvec[2]), fpamp_of_mom(+S2.pvec[3]) };
    const fptype_amp_sv P3[4] = { fpamp_of_mom(+S3.pvec[0]), fpamp_of_mom(+S3.pvec[1]), fpamp_of_mom(+S3.pvec[2]), fpamp_of_mom(+S3.pvec[3]) };
    const cxtype_amp_sv TMP8 = ( P2[0] * wV1[0] - P2[1] * wV1[1] - P2[2] * wV1[2] - P2[3] * wV1[3] );
    const cxtype_amp_sv TMP9 = ( wV1[0] * P3[0] - wV1[1] * P3[1] - wV1[2] * P3[2] - wV1[3] * P3[3] );
    ( *vertex ) = (cxtype_amp_sv)( Ccoeff * COUP * wS2[4] * wS3[4] * ( -cI * TMP8 + cI * TMP9 ) );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ void
  VSS2_0( const ALOHAOBJ  & V1,
          const ALOHAOBJ  & S2,
          const ALOHAOBJ  & S3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wV1 = W_ACCESS::kernelAccessConst( V1.w );
    const cxtype_amp_sv* wS2 = W_ACCESS::kernelAccessConst( S2.w );
    const cxtype_amp_sv* wS3 = W_ACCESS::kernelAccessConst( S3.w );
    const cxtype_sv COUP = C_ACCESS::kernelAccessConst( allCOUP );
    cxtype_amp_sv* vertex = A_ACCESS::kernelAccess( allvertexes );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    const fptype_amp_sv P2[4] = { fpamp_of_mom(+S2.pvec[0]), fpamp_of_mom(+S2.pvec[1]), fpamp_of_mom(+S2.pvec[2]), fpamp_of_mom(+S2.pvec[3]) };
    const fptype_amp_sv P3[4] = { fpamp_of_mom(+S3.pvec[0]), fpamp_of_mom(+S3.pvec[1]), fpamp_of_mom(+S3.pvec[2]), fpamp_of_mom(+S3.pvec[3]) };
    const cxtype_amp_sv TMP8 = ( P2[0] * wV1[0] - P2[1] * wV1[1] - P2[2] * wV1[2] - P2[3] * wV1[3] );
    const cxtype_amp_sv TMP9 = ( wV1[0] * P3[0] - wV1[1] * P3[1] - wV1[2] * P3[2] - wV1[3] * P3[3] );
    ( *vertex ) = (cxtype_amp_sv)( Ccoeff * COUP * wS2[4] * wS3[4] * ( -cI * TMP9 + cI * TMP8 ) );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ void
  SVV2_0( const ALOHAOBJ  & S1,
          const ALOHAOBJ  & V2,
          const ALOHAOBJ  & V3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wS1 = W_ACCESS::kernelAccessConst( S1.w );
    const cxtype_amp_sv* wV2 = W_ACCESS::kernelAccessConst( V2.w );
    const cxtype_amp_sv* wV3 = W_ACCESS::kernelAccessConst( V3.w );
    const cxtype_sv COUP = C_ACCESS::kernelAccessConst( allCOUP );
    cxtype_amp_sv* vertex = A_ACCESS::kernelAccess( allvertexes );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    const cxtype_amp_sv TMP7 = ( wV3[0] * wV2[0] - wV3[1] * wV2[1] - wV3[2] * wV2[2] - wV3[3] * wV2[3] );
    ( *vertex ) = (cxtype_amp_sv)( Ccoeff * COUP * -cI * TMP7 * wS1[4] );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ void
  SVS2_0( const ALOHAOBJ  & S1,
          const ALOHAOBJ  & V2,
          const ALOHAOBJ  & S3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wS1 = W_ACCESS::kernelAccessConst( S1.w );
    const cxtype_amp_sv* wV2 = W_ACCESS::kernelAccessConst( V2.w );
    const cxtype_amp_sv* wS3 = W_ACCESS::kernelAccessConst( S3.w );
    const cxtype_sv COUP = C_ACCESS::kernelAccessConst( allCOUP );
    cxtype_amp_sv* vertex = A_ACCESS::kernelAccess( allvertexes );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    const fptype_amp_sv P1[4] = { fpamp_of_mom(+S1.pvec[0]), fpamp_of_mom(+S1.pvec[1]), fpamp_of_mom(+S1.pvec[2]), fpamp_of_mom(+S1.pvec[3]) };
    const fptype_amp_sv P3[4] = { fpamp_of_mom(+S3.pvec[0]), fpamp_of_mom(+S3.pvec[1]), fpamp_of_mom(+S3.pvec[2]), fpamp_of_mom(+S3.pvec[3]) };
    const cxtype_amp_sv TMP4 = ( wV2[0] * P1[0] - wV2[1] * P1[1] - wV2[2] * P1[2] - wV2[3] * P1[3] );
    const cxtype_amp_sv TMP6 = ( wV2[0] * P3[0] - wV2[1] * P3[1] - wV2[2] * P3[2] - wV2[3] * P3[3] );
    ( *vertex ) = (cxtype_amp_sv)( Ccoeff * COUP * wS1[4] * wS3[4] * ( -cI * TMP6 + cI * TMP4 ) );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output amplitude 'vertex' from the input wavefunctions 
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ void
  SSV3_0( const ALOHAOBJ  & S1,
          const ALOHAOBJ  & S2,
          const ALOHAOBJ  & V3,
          const fptype allCOUP[],
          const fptype Ccoeff,
          fptype_amp allvertexes[] )
  {
    mgDebug( 0, __FUNCTION__ );
    const cxtype_amp_sv* wS1 = W_ACCESS::kernelAccessConst( S1.w );
    const cxtype_amp_sv* wS2 = W_ACCESS::kernelAccessConst( S2.w );
    const cxtype_amp_sv* wV3 = W_ACCESS::kernelAccessConst( V3.w );
    const cxtype_sv COUP = C_ACCESS::kernelAccessConst( allCOUP );
    cxtype_amp_sv* vertex = A_ACCESS::kernelAccess( allvertexes );
    cxtype_amp_sv CZERO = cxzero_sv<cxtype_amp_sv>(); 
    const cxtype_amp_sv cI = cxmake( 0., 1. );
    const fptype_amp_sv P1[4] = { fpamp_of_mom(+S1.pvec[0]), fpamp_of_mom(+S1.pvec[1]), fpamp_of_mom(+S1.pvec[2]), fpamp_of_mom(+S1.pvec[3]) };
    const fptype_amp_sv P2[4] = { fpamp_of_mom(+S2.pvec[0]), fpamp_of_mom(+S2.pvec[1]), fpamp_of_mom(+S2.pvec[2]), fpamp_of_mom(+S2.pvec[3]) };
    const cxtype_amp_sv TMP2 = ( wV3[0] * P1[0] - wV3[1] * P1[1] - wV3[2] * P1[2] - wV3[3] * P1[3] );
    const cxtype_amp_sv TMP3 = ( wV3[0] * P2[0] - wV3[1] * P2[1] - wV3[2] * P2[2] - wV3[3] * P2[3] );
    ( *vertex ) = (cxtype_amp_sv)( Ccoeff * COUP * wS1[4] * wS2[4] * ( -cI * TMP2 + cI * TMP3 ) );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

} // end namespace

#endif // HelAmps_sm_H
