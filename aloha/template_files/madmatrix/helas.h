! Copyright (C) 2010 The ALOHA Development team and Contributors.
! Copyright (C) 2010 The MadGraph7 development team and contributors.
! Created by: J. Alwall (Sep 2010) for the MadGraph7 CPP backend.
!==========================================================================
! Copyright (C) 2020-2026 CERN and UCLouvain.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Modified originally by: O. Mattelaer (Mar 2020) for the MadGraph7 CUDACPP plugin.
! Further modified by:  F. Stloukal, D. Massaro, O. Mattelaer, A. Thete, A. Valassi (2020-2024).
! Integrated with the MadGraph7 project in Feb 2026.
!==========================================================================

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

  // Compute the output wavefunction vc[6] from the input momenta[npar*4*nevt],
  // in the axial gauge of the momentum of particle iref (see the definition)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  vxxxxxr( const fptype_momenta momenta[], // input: momenta
           const fptype_amp vmass,     // input: vector boson mass
           const fptype_amp_sv nhel, // input: -1, 0 (only if vmass!=0) or +1 (helicity of vector boson, per SIMD lane/GPU thread)
           const int nsv,          // input: +1 (final) or -1 (initial)
           const int flv,          // input: flavor index
           ALOHAOBJ & vc,          // output: wavefunctions
           const int ipar,         // input: particle# out of npar
           const int iref          // input: particle# of the gauge reference momentum
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
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction vc[6] from the input momenta[npar*4*nevt],
  // in the axial gauge of the momentum r of particle iref, i.e. with r.eps = 0.
  // This is vxxxxx followed by the residual gauge transformation
  //   eps'^mu = eps^mu - (r.eps)/(r.p) p^mu
  // which, for a massless p, keeps eps'.p = 0, eps'.eps' = 0 and eps'.eps'* = -1
  // and so returns the same helicity state up to a phase that cancels in |M|^2.
  // With r the (lightlike) momentum of another external particle, eps.r = 0 and,
  // for legs sharing r, eps_i.eps_j = 0 for same-helicity pairs: whole currents
  // and amplitudes vanish for given helicities (arXiv:2312.07447).
  // r is ignored for a massive vector and wherever r.p = 0, in particular when
  // iref is ipar itself (see vxxxxxr in aloha_functions.f)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  vxxxxxr( const fptype_momenta momenta[], // input: momenta
           const fptype_amp vmass,     // input: vector boson mass
           const fptype_amp_sv nhel, // input: -1, 0 (only if vmass!=0) or +1 (helicity of vector boson, per SIMD lane/GPU thread)
           const int nsv,          // input: +1 (final) or -1 (initial)
           const int flv,          // input: flavour
           ALOHAOBJ & vc,          // output: wavefunctions
           const int ipar,         // input: particle# out of npar
           const int iref )        // input: particle# of the gauge reference momentum
  {
    vxxxxx<M_ACCESS, W_ACCESS>( momenta, vmass, nhel, nsv, flv, vc, ipar );
    // the shift is for the transverse helicities, the only ones of a massless vector
    if( vmass != 0. || iref == ipar ) return;
    mgDebug( 0, __FUNCTION__ );
    const fptype_momenta_sv rm0 = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, iref );
    const fptype_momenta_sv rm1 = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, iref );
    const fptype_momenta_sv rm2 = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, iref );
    const fptype_momenta_sv rm3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, iref );
    // vc.pvec is nsv * p: the sign cancels between r.p and the shift along p.
    // r.p of two lightlike momenta cancels between its terms when they are close
    // to collinear, as p^2 does: it is made in the precision of the momenta (FP64
    // in FPTYPE=v, as the propagator denominators), the shift in that of the amplitudes
    const fptype_momenta_sv rdotp_mom = rm0 * vc.pvec[0] - rm1 * vc.pvec[1] - rm2 * vc.pvec[2] - rm3 * vc.pvec[3];
    const fptype_amp_sv rdotp = fpamp_of_mom( rdotp_mom );
    const fptype_amp_sv r0 = fpamp_of_mom( rm0 );
    const fptype_amp_sv r1 = fpamp_of_mom( rm1 );
    const fptype_amp_sv r2 = fpamp_of_mom( rm2 );
    const fptype_amp_sv r3 = fpamp_of_mom( rm3 );
    const fptype_amp_sv p0 = fpamp_of_mom( vc.pvec[0] );
    const fptype_amp_sv p1 = fpamp_of_mom( vc.pvec[1] );
    const fptype_amp_sv p2 = fpamp_of_mom( vc.pvec[2] );
    const fptype_amp_sv p3 = fpamp_of_mom( vc.pvec[3] );
    cxtype_amp_sv* w = W_ACCESS::kernelAccess( vc.w );
    const cxtype_amp_sv rdote = w[0] * r0 - w[1] * r1 - w[2] * r2 - w[3] * r3;
#ifndef MGONGPU_CPPSIMD
    if( rdotp == 0. )
    {
      mgDebug( 1, __FUNCTION__ );
      return;
    }
    const cxtype_amp_sv cfac = rdote / rdotp;
#else
    // as in vxxxxx: no division by zero in the lanes where r.p = 0 (#701), which keep eps
    volatile fptype_amp_v rpDENOM = fpternary( rdotp != 0, rdotp, 1. );
    const fptype_amp_v rpINV = fptype_amp( 1 ) / rpDENOM;
    const cxtype_amp_v cfac = cxternary( rdotp != 0, rdote * rpINV, cxzero_sv() );
#endif
    w[0] = w[0] - cfac * p0;
    w[1] = w[1] - cfac * p1;
    w[2] = w[2] - cfac * p2;
    w[3] = w[3] - cfac * p3;
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

  //==========================================================================
