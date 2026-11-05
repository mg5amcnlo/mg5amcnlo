! Copyright (C) 2010 The ALOHA Development team and Contributors.
! Copyright (C) 2010 The MadGraph7 development team and contributors.
! Created by: J. Alwall (Sep 2010) for the MadGraph7 CPP backend.
!==========================================================================
! Copyright (C) 2020-2026 CERN and UCLouvain.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Modified originally by: F. Stloukal (Mar 2026) for the MadGraph7 CUDACPP plugin.
! Further modified by: D. Massaro, F. Stloukal
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
