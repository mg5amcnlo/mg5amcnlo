! Copyright (C) 2010 The ALOHA Development team and Contributors.
! Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
! Created by: J. Alwall (Sep 2010) for the MG5aMC CPP backend.
!==========================================================================
! Copyright (C) 2020-2026 CERN and UCLouvain.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Modified originally by: F. Stloukal (Mar 2026) for the MG5aMC CUDACPP plugin.
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
  ixxxxx( const fptype momenta[], // input: momenta
          const fptype fmass,     // input: fermion mass
          const int nhel,         // input: -1 or +1 (helicity of fermion)
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
  ipzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
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
  imzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
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
  ixzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavor index
          ALOHAOBJ & fi,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction vc[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  vxxxxx( const fptype momenta[], // input: momenta
          const fptype vmass,     // input: vector boson mass
          const int nhel,         // input: -1, 0 (only if vmass!=0) or +1 (helicity of vector boson)
          const int nsv,          // input: +1 (final) or -1 (initial)
          const int flv,          // input: flavor index
          ALOHAOBJ & vc,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction sc[3] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  sxxxxx( const fptype momenta[], // input: momenta
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
  oxxxxx( const fptype momenta[], // input: momenta
          const fptype fmass,     // input: fermion mass
          const int nhel,         // input: -1, 0 (only if vmass!=0) or +1 (helicity of vector boson)
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
  opzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
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
  omzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavor index
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  oxzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavor index
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;


//--------------------------------------------------------------------------

  // Compute the direction n[5] of the gauge q[5]
  __host__ __device__ INLINE void
  define_gauge_dir( const fptype q[], // input: gauge
                    fptype n[]        // output: direction
                    ) ALWAYS_INLINE;


  //--------------------------------------------------------------------------
  // Compute a propagator factor d out of gauge q[5] and a mass
  __host__ __device__ INLINE void
  calculate_propagator_factor( const fptype_sv q[5], // input: gauge
                               const fptype_sv mass, // input: mass
                               fptype_sv *d          // output: propagator factor
                               ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------
  // multiply by propagation factor from m and wawefunctionsin[] and output them
  // as wavefunctionout[]
  template< class W_ACCESS>
  __host__ __device__ INLINE void
  multiply_propagator_factor( const fptype wavefunctionsin[], // input: wavefunctions
                              const fptype m,                 // input: mass
                              fptype wavefunctionsout[]       // output: wavefunctions
                              ) ALWAYS_INLINE;
//==========================================================================

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  ixxxxx( const fptype momenta[], // input: momenta
          const fptype fmass,     // input: fermion mass
          const int nhel,         // input: -1 or +1 (helicity of fermion)
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
    const fptype_sv& pvec0 = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_sv& pvec1 = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_sv& pvec2 = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* w = W_ACCESS::kernelAccess( fi.w );
    fi.pvec[0] = -pvec0 * (fptype)nsf;
    fi.pvec[1] = -pvec1 * (fptype)nsf;
    fi.pvec[2] = -pvec2 * (fptype)nsf;
    fi.pvec[3] = -pvec3 * (fptype)nsf;
    fi.flv_index = flv;
    const int nh = nhel * nsf;
    if( fmass != 0. )
    {
#ifndef MGONGPU_CPPSIMD
      const fptype_sv pp = fpmin( pvec0, fpsqrt( pvec1 * pvec1 + pvec2 * pvec2 + pvec3 * pvec3 ) );
#else
      volatile fptype_sv p2 = pvec1 * pvec1 + pvec2 * pvec2 + pvec3 * pvec3; // volatile fixes #736
      const fptype_sv pp = fpmin( pvec0, fpsqrt( p2 ) );
#endif
      // In C++ ixxxxx, use a single ip/im numbering that is valid both for pp==0 and pp>0, which have two numbering schemes in Fortran ixxxxx:
      // for pp==0, Fortran sqm(0:1) has indexes 0,1 as in C++; but for Fortran pp>0, omega(2) has indexes 1,2 and not 0,1
      // NB: this is only possible in ixxxx, but in oxxxxx two different numbering schemes must be used
      const int ip = ( 1 + nh ) / 2; // NB: same as in Fortran pp==0, differs from Fortran pp>0, which is (3+nh)/2 because omega(2) has indexes 1,2
      const int im = ( 1 - nh ) / 2; // NB: same as in Fortran pp==0, differs from Fortran pp>0, which is (3-nh)/2 because omega(2) has indexes 1,2
#ifndef MGONGPU_CPPSIMD
      if( pp == 0. )
      {
        // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
        fptype sqm[2] = { fpsqrt( std::abs( fmass ) ), 0. }; // possibility of negative fermion masses
        //sqm[1] = ( fmass < 0. ? -abs( sqm[0] ) : abs( sqm[0] ) ); // AV: why abs here?
        sqm[1] = ( fmass < 0. ? -sqm[0] : sqm[0] ); // AV: removed an abs here
        w[0] = cxmake( ip * sqm[ip], 0 );
        w[1] = cxmake( im * nsf * sqm[ip], 0 );
        w[2] = cxmake( ip * nsf * sqm[im], 0 );
        w[3] = cxmake( im * sqm[im], 0 );
      }
      else
      {
        const fptype sf[2] = { fptype( 1 + nsf + ( 1 - nsf ) * nh ) * (fptype)0.5,
                               fptype( 1 + nsf - ( 1 - nsf ) * nh ) * (fptype)0.5 };
        fptype omega[2] = { fpsqrt( pvec0 + pp ), 0. };
        omega[1] = fmass / omega[0];
        const fptype sfomega[2] = { sf[0] * omega[ip], sf[1] * omega[im] };
        const fptype pp3 = fpmax( pp + pvec3, 0. );
        const cxtype chi[2] = { cxmake( fpsqrt( pp3 * (fptype)0.5 / pp ), 0. ),
                                ( pp3 == 0. ? cxmake( -nh, 0. ) : cxmake( nh * pvec1, pvec2 ) / fpsqrt( 2. * pp * pp3 ) ) };
        w[0] = sfomega[0] * chi[im];
        w[1] = sfomega[0] * chi[ip];
        w[2] = sfomega[1] * chi[im];
        w[3] = sfomega[1] * chi[ip];
      }
#else
      // Branch A: pp == 0.
      // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
      fptype sqm[2] = { fpsqrt( std::abs( fmass ) ), 0 }; // possibility of negative fermion masses (NB: SCALAR!)
      sqm[1] = ( fmass < 0 ? -sqm[0] : sqm[0] );          // AV: removed an abs here (as above)
      const cxtype fiA_2 = ip * sqm[ip];                  // scalar cxtype: real part initialised from fptype, imag part = 0
      const cxtype fiA_3 = im * nsf * sqm[ip];            // scalar cxtype: real part initialised from fptype, imag part = 0
      const cxtype fiA_4 = ip * nsf * sqm[im];            // scalar cxtype: real part initialised from fptype, imag part = 0
      const cxtype fiA_5 = im * sqm[im];                  // scalar cxtype: real part initialised from fptype, imag part = 0
      // Branch B: pp != 0.
      const fptype sf[2] = { fptype( 1 + nsf + ( 1 - nsf ) * nh ) * (fptype)0.5,
                             fptype( 1 + nsf - ( 1 - nsf ) * nh ) * (fptype)0.5 };
      fptype_v omega[2] = { fpsqrt( pvec0 + pp ), 0 };
      omega[1] = fmass / omega[0];
      const fptype_v sfomega[2] = { sf[0] * omega[ip], sf[1] * omega[im] };
      const fptype_v pp3 = fpmax( pp + pvec3, 0 );
      volatile fptype_v ppDENOM = fpternary( pp != 0, pp, 1. );    // hack: ppDENOM[ieppV]=1 if pp[ieppV]==0
      volatile fptype_v pp3DENOM = fpternary( pp3 != 0, pp3, 1. ); // hack: pp3DENOM[ieppV]=1 if pp3[ieppV]==0
      volatile fptype_v chi0r2 = pp3 * 0.5 / ppDENOM;              // volatile fixes #736
      const cxtype_v chi[2] = { cxmake( fpsqrt( chi0r2 ), 0 ),     // hack: dummy[ieppV] is not used if pp[ieppV]==0
                                cxternary( ( pp3 == 0. ),
                                           cxmake( -nh, 0 ),
                                           cxmake( (fptype)nh * pvec1, pvec2 ) / fpsqrt( 2. * ppDENOM * pp3DENOM ) ) }; // hack: dummy[ieppV] is not used if pp[ieppV]==0
      const cxtype_v fiB_2 = sfomega[0] * chi[im];
      const cxtype_v fiB_3 = sfomega[0] * chi[ip];
      const cxtype_v fiB_4 = sfomega[1] * chi[im];
      const cxtype_v fiB_5 = sfomega[1] * chi[ip];
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
      volatile fptype_sv p0p3 = fpmax( pvec0 + pvec3, 0 ); // volatile fixes #736
      volatile fptype_sv sqp0p3 = fpternary( ( pvec1 == 0. and pvec2 == 0. and pvec3 < 0. ),
                                             fptype_sv{ 0 },
                                             fpsqrt( p0p3 ) * (fptype)nsf );
      volatile fptype_sv sqp0p3DENOM = fpternary( sqp0p3 != 0, (fptype_sv)sqp0p3, 1. ); // hack: dummy sqp0p3DENOM[ieppV]=1 if sqp0p3[ieppV]==0
      cxtype_sv chi[2] = { cxmake( (fptype_v)sqp0p3, 0. ),
                           cxternary( sqp0p3 == 0,
                                      cxmake( -(fptype)nhel * fpsqrt( 2. * pvec0 ), 0. ),
                                      cxmake( (fptype)nh * pvec1, pvec2 ) / (const fptype_v)sqp0p3DENOM ) }; // hack: dummy[ieppV] is not used if sqp0p3[ieppV]==0
#else
      const fptype_sv sqp0p3 = fpternary( ( pvec1 == 0. and pvec2 == 0. and pvec3 < 0. ),
                                          fptype_sv{ 0 },
                                          fpsqrt( fpmax( pvec0 + pvec3, 0. ) ) * (fptype)nsf );
      const cxtype_sv chi[2] = { cxmake( sqp0p3, 0. ),
                                 ( sqp0p3 == 0. ? cxmake( -(fptype)nhel * fpsqrt( 2. * pvec0 ), 0. ) : cxmake( (fptype)nh * pvec1, pvec2 ) / sqp0p3 ) };
#endif
      if( nh == 1 )
      {
        w[0] = cxzero_sv();
        w[1] = cxzero_sv();
        w[2] = chi[0];
        w[3] = chi[1];
      }
      else
      {
        w[0] = chi[1];
        w[1] = chi[0];
        w[2] = cxzero_sv();
        w[3] = cxzero_sv();
      }
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  ipzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fi,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* w = W_ACCESS::kernelAccess( fi.w );
    fi.pvec[0] = -pvec3 * (fptype)nsf;
    fi.pvec[1] = fptype_sv{ 0 };
    fi.pvec[2] = fptype_sv{ 0 };
    fi.pvec[3] = -pvec3 * (fptype)nsf;
    fi.flv_index = flv;
    const int nh = nhel * nsf;
    const cxtype_sv sqp0p3 = cxmake( fpsqrt( 2. * pvec3 ) * (fptype)nsf, 0. );
    w[0] = cxmake( fi.pvec[1], fi.pvec[2] );
    if( nh == 1 )
    {
      w[1] = cxmake( fi.pvec[1], fi.pvec[2] );
      w[2] = sqp0p3;
    }
    else
    {
      w[1] = sqp0p3;
      w[2] = cxmake( fi.pvec[1], fi.pvec[2] );
    }
    w[3] = cxmake( fi.pvec[1], fi.pvec[2] );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == -PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  imzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fi,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* w = W_ACCESS::kernelAccess( fi.w );
    fi.pvec[0] =  pvec3 * (fptype)nsf;
    fi.pvec[1] = fptype_sv{ 0 };
    fi.pvec[2] = fptype_sv{ 0 };
    fi.pvec[3] = -pvec3 * (fptype)nsf;
    fi.flv_index = flv;
    const int nh = nhel * nsf;
    const cxtype_sv chi = cxmake( -(fptype)nhel * fpsqrt( -2. * pvec3 ), 0. );
    w[1] = cxzero_sv();
    w[2] = cxzero_sv();
    if( nh == 1 )
    {
      w[0] = cxzero_sv();
      w[3] = chi;
    }
    else
    {
      w[0] = chi;
      w[3] = cxzero_sv();
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PT > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  ixzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fi,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec0 = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_sv& pvec1 = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_sv& pvec2 = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* w = W_ACCESS::kernelAccess( fi.w );
    fi.pvec[0] = -pvec0 * (fptype)nsf;
    fi.pvec[1] = -pvec1 * (fptype)nsf;
    fi.pvec[2] = -pvec2 * (fptype)nsf;
    fi.pvec[3] = -pvec3 * (fptype)nsf;
    fi.flv_index = flv;
    const int nh = nhel * nsf;
    //const float sqp0p3 = sqrtf( pvec0 + pvec3 ) * nsf; // AV: why force a float here?
    const fptype_sv sqp0p3 = fpsqrt( pvec0 + pvec3 ) * (fptype)nsf;
    const cxtype_sv chi0 = cxmake( sqp0p3, 0. );
    const cxtype_sv chi1 = cxmake( (fptype)nh * pvec1 / sqp0p3, pvec2 / sqp0p3 );
    if( nh == 1 )
    {
      w[0] = cxzero_sv();
      w[1] = cxzero_sv();
      w[2] = chi0;
      w[3] = chi1;
    }
    else
    {
      w[0] = chi1;
      w[1] = chi0;
      w[2] = cxzero_sv();
      w[3] = cxzero_sv();
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction vc[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  vxxxxx( const fptype momenta[], // input: momenta
          const fptype vmass,     // input: vector boson mass
          const int nhel,         // input: -1, 0 (only if vmass!=0) or +1 (helicity of vector boson)
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
    const fptype_sv& pvec0 = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_sv& pvec1 = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_sv& pvec2 = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* w = W_ACCESS::kernelAccess( vc.w );
    vc.pvec[0] = pvec0 * (fptype)nsv;
    vc.pvec[1] = pvec1 * (fptype)nsv;
    vc.pvec[2] = pvec2 * (fptype)nsv;
    vc.pvec[3] = pvec3 * (fptype)nsv;
    vc.flv_index = flv;
    const fptype sqh = fpsqrt( 0.5 ); // AV this is > 0!
    const fptype hel = nhel;

    // FD gauge
     const cxtype_sv cI = cxmake( 0 + fptype_sv{ 0 },  1 + fptype_sv{ 0 }  );
#ifdef MGONGPU_CPPSIMD
    fptype_sv nA[5];
    fptype_sv nB[5];
#endif
    fptype_sv n[5];
    fptype_sv nk;
    const fptype_sv zero{0.};
    const fptype_sv one{1.};

    if( vmass != 0. )
    {
      const int nsvahl = nsv * std::abs( hel );
      const fptype hel0 = 1. - std::abs( hel );
#ifndef MGONGPU_CPPSIMD
      const fptype_sv pt2 = ( pvec1 * pvec1 ) + ( pvec2 * pvec2 );
      const fptype_sv pp = fpmin( pvec0, fpsqrt( pt2 + ( pvec3 * pvec3 ) ) );
      const fptype_sv pt = fpmin( pp, fpsqrt( pt2 ) );
      if( pp == 0. )
      {
        w[0] = cxmake( 0., 0. );
        w[1] = cxmake( -hel * sqh, 0. );
        w[2] = cxmake( 0., nsvahl * sqh );
        w[3] = cxmake( hel0, 0. );
      }
      else
      {
        //printf( "DEBUG1011 (before emp): pvec0=%f vmass=%f pp=%f vmass*pp=%f\n", pvec0, vmass, pp, vmass * pp );
        //const fptype emp = pvec / ( vmass * pp ); // this may give a FPE #1011 (why?! maybe when vmass=+-epsilon?)
        const fptype emp = pvec0 / vmass / pp; // workaround for FPE #1011
        //printf( "DEBUG1011 (after emp): emp=%f\n", emp );
        w[0] = cxmake( hel0 * pp / vmass, 0. );
        w[3] = cxmake( hel0 * pvec3 * emp + hel * pt / pp * sqh, 0. );
        if( pt != 0. )
        {
          const fptype pzpt = pvec3 / ( pp * pt ) * sqh * hel;
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

      if ( abs(nhel) == 1)
      {
        w[4] = cxzero_sv();
      }
      else{
        w[0] = cxmake( -vmass/nk * n[0], zero );
        w[1] = cxmake( -vmass/nk * n[1], zero );
        w[2] = cxmake( -vmass/nk * n[2], zero );
        w[3] = cxmake( -vmass/nk * n[3], zero );
        w[4] = static_cast<fptype>(nsv)*cI;
      }

#else

      volatile fptype_sv pt2 = ( pvec1 * pvec1 ) + ( pvec2 * pvec2 );
      volatile fptype_sv p2 = pt2 + ( pvec3 * pvec3 ); // volatile fixes #736
      const fptype_sv pp = fpmin( pvec0, fpsqrt( p2 ) );
      const fptype_sv pt = fpmin( pp, fpsqrt( pt2 ) );
      // Branch A: pp == 0.
      const cxtype vcA_2 = cxmake( 0, 0 );
      const cxtype vcA_3 = cxmake( -hel * sqh, 0 );
      const cxtype vcA_4 = cxmake( 0, nsvahl * sqh );
      const cxtype vcA_5 = cxmake( hel0, 0 );
      // Branch B: pp != 0.
      volatile fptype_v ppDENOM = fpternary( pp != 0, pp, 1. ); // hack: ppDENOM[ieppV]=1 if pp[ieppV]==0
      const fptype_v emp = pvec0 / ( vmass * ppDENOM );         // hack: dummy[ieppV] is not used if pp[ieppV]==0
      const cxtype_v vcB_2 = cxmake( hel0 * pp / vmass, 0 );
      const cxtype_v vcB_5 = cxmake( hel0 * pvec3 * emp + hel * pt / ppDENOM * sqh, 0 ); // hack: dummy[ieppV] is not used if pp[ieppV]==0
      // Branch B1: pp != 0. and pt != 0.
      volatile fptype_v ptDENOM = fpternary( pt != 0, pt, 1. );                                                     // hack: ptDENOM[ieppV]=1 if pt[ieppV]==0
      const fptype_v pzpt = pvec3 / ( ppDENOM * ptDENOM ) * sqh * hel;                                              // hack: dummy[ieppV] is not used if pp[ieppV]==0
      const cxtype_v vcB1_3 = cxmake( hel0 * pvec1 * emp - pvec1 * pzpt, -(fptype)nsvahl * pvec2 / ptDENOM * sqh ); // hack: dummy[ieppV] is not used if pt[ieppV]==0
      const cxtype_v vcB1_4 = cxmake( hel0 * pvec2 * emp - pvec2 * pzpt, (fptype)nsvahl * pvec1 / ptDENOM * sqh );  // hack: dummy[ieppV] is not used if pt[ieppV]==0
      // Branch B2: pp != 0. and pt == 0.
      const cxtype vcB2_3 = cxmake( -hel * sqh, 0. );
      const cxtype_v vcB2_4 = cxmake( 0., (fptype)nsvahl * fpternary( ( pvec3 < 0 ), -sqh, sqh ) ); // AV: removed an abs here
      // Choose between the results from branch A and branch B (and from branch B1 and branch B2)
      const bool_v mask = ( pp == 0. );
      const bool_v maskB = ( pt != 0. );
      w[0] = cxternary( mask, vcA_2, vcB_2 );
      w[1] = cxternary( mask, vcA_3, cxternary( maskB, vcB1_3, vcB2_3 ) );
      w[2] = cxternary( mask, vcA_4, cxternary( maskB, vcB1_4, vcB2_4 ) );
      w[3] = cxternary( mask, vcA_5, vcB_5 );

      //FD gauge
      //branch A
      nA[0] = fpternary( pvec0 >= zero , one , -one);
      nA[1] = -pvec1/pp;
      nA[2] = -pvec2/pp;
      nA[3] = -pvec3/pp;
      nA[4] = zero;

      //branch B
      nB[0] = nA[0];
      nB[1] = zero;
      nB[2] = zero;
      nB[3] = -nA[0];

      const fptype_sv b_A = fpternary(pp > zero, one , zero);
      const fptype_sv b_B = fpternary(pp <= zero , one , zero);

      n[0] = nA[0]*b_A + nB[0]*b_B;
      n[1] = nA[1]*b_A + nB[1]*b_B;
      n[2] = nA[2]*b_A + nB[2]*b_B;
      n[3] = nA[3]*b_A + nB[3]*b_B;
      n[4] = nA[4];

      nk = n[0]*pvec0 - n[1]*pvec1 - n[2]*pvec2 - n[3]*pvec3;

      const bool_v mask3 = { (abs(nhel) == 1 ? 1 : 0) }; // first element replicated
      w[0] = cxternary( mask3, w[0], cxmake( -vmass/nk * n[0], zero));
      w[1] = cxternary( mask3, w[1], cxmake( -vmass/nk * n[1], zero));
      w[2] = cxternary( mask3, w[2], cxmake( -vmass/nk * n[2], zero));
      w[3] = cxternary( mask3, w[3], cxmake( -vmass/nk * n[3], zero));
      w[4] = cxternary( mask3, cxzero_sv(), -static_cast<fptype>(nsv)*cI);
#endif
    }
    else
    {
      const fptype_sv& pp = pvec0; // NB: rewrite the following as in Fortran, using pp instead of pvec0
#ifndef MGONGPU_CPPSIMD
      const fptype_sv pt = fpsqrt( ( pvec1 * pvec1 ) + ( pvec2 * pvec2 ) );
#else
      volatile fptype_sv pt2 = pvec1 * pvec1 + pvec2 * pvec2; // volatile fixes #736
      const fptype_sv pt = fpsqrt( pt2 );
#endif
      w[0] = cxzero_sv();
      w[3] = cxmake( hel * pt / pp * sqh, 0. );
#ifndef MGONGPU_CPPSIMD
      if( pt != 0. )
      {
        const fptype pzpt = pvec3 / ( pp * pt ) * sqh * hel;
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
      volatile fptype_v ptDENOM = fpternary( pt != 0, pt, 1. );                             // hack: ptDENOM[ieppV]=1 if pt[ieppV]==0
      const fptype_v pzpt = pvec3 / ( pp * ptDENOM ) * sqh * hel;                           // hack: dummy[ieppV] is not used if pt[ieppV]==0
      const cxtype_v vcA_3 = cxmake( -pvec1 * pzpt, -(fptype)nsv * pvec2 / ptDENOM * sqh ); // hack: dummy[ieppV] is not used if pt[ieppV]==0
      const cxtype_v vcA_4 = cxmake( -pvec2 * pzpt, (fptype)nsv * pvec1 / ptDENOM * sqh );  // hack: dummy[ieppV] is not used if pt[ieppV]==0
      // Branch B: pt == 0.
      const cxtype vcB_3 = cxmake( -(fptype)hel * sqh, 0 );
      const cxtype_v vcB_4 = cxmake( 0, (fptype)nsv * fpternary( ( pvec3 < 0 ), -sqh, sqh ) ); // AV: removed an abs here
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
  sxxxxx( const fptype momenta[], // input: momenta
          //const fptype,                 // WARNING: input "smass" unused (missing in Fortran) - scalar boson mass
          //const int,                    // WARNING: input "nhel" unused (missing in Fortran) - scalar has no helicity!
          const int nss,          // input: +1 (final) or -1 (initial)
          const int flv,          // input: flavour
          ALOHAOBJ &sc,           // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec0 = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_sv& pvec1 = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_sv& pvec2 = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* w = W_ACCESS::kernelAccess( sc.w );

    sc.pvec[0] = pvec0 * (fptype)nss;
    sc.pvec[1] = pvec1 * (fptype)nss;
    sc.pvec[2] = pvec2 * (fptype)nss;
    sc.pvec[3] = pvec3 * (fptype)nss;

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
  oxxxxx( const fptype momenta[], // input: momenta
          const fptype fmass,     // input: fermion mass
          const int nhel,         // input: -1 or +1 (helicity of fermion)
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
    const fptype_sv& pvec0 = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_sv& pvec1 = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_sv& pvec2 = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* w = W_ACCESS::kernelAccess( fo.w );
    fo.pvec[0] = pvec0 * (fptype)nsf;
    fo.pvec[1] = pvec1 * (fptype)nsf;
    fo.pvec[2] = pvec2 * (fptype)nsf;
    fo.pvec[3] = pvec3 * (fptype)nsf;
    fo.flv_index = flv;
    const int nh = nhel * nsf;
    if( fmass != 0. )
    {
#ifndef MGONGPU_CPPSIMD
      const fptype_sv pp = fpmin( pvec0, fpsqrt( ( pvec1 * pvec1 ) + ( pvec2 * pvec2 ) + ( pvec3 * pvec3 ) ) );
      if( pp == 0. )
      {
        // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
        fptype sqm[2] = { fpsqrt( std::abs( fmass ) ), 0. }; // possibility of negative fermion masses
        //sqm[1] = ( fmass < 0. ? -abs( sqm[0] ) : abs( sqm[0] ) ); // AV: why abs here?
        sqm[1] = ( fmass < 0. ? -sqm[0] : sqm[0] ); // AV: removed an abs here
        const int ip = -( ( 1 - nh ) / 2 ) * nhel;  // NB: Fortran sqm(0:1) also has indexes 0,1 as in C++
        const int im = ( 1 + nh ) / 2 * nhel;       // NB: Fortran sqm(0:1) also has indexes 0,1 as in C++
        w[0] = cxmake( im * sqm[std::abs( ip )], 0 );
        w[1] = cxmake( ip * nsf * sqm[std::abs( ip )], 0 );
        w[2] = cxmake( im * nsf * sqm[std::abs( im )], 0 );
        w[3] = cxmake( ip * sqm[std::abs( im )], 0 );
      }
      else
      {
        const fptype sf[2] = { fptype( 1 + nsf + ( 1 - nsf ) * nh ) * (fptype)0.5,
                               fptype( 1 + nsf - ( 1 - nsf ) * nh ) * (fptype)0.5 };
        fptype omega[2] = { fpsqrt( pvec0 + pp ), 0. };
        omega[1] = fmass / omega[0];
        const int ip = ( 1 + nh ) / 2; // NB: Fortran is (3+nh)/2 because omega(2) has indexes 1,2 and not 0,1
        const int im = ( 1 - nh ) / 2; // NB: Fortran is (3-nh)/2 because omega(2) has indexes 1,2 and not 0,1
        const fptype sfomeg[2] = { sf[0] * omega[ip], sf[1] * omega[im] };
        const fptype pp3 = fpmax( pp + pvec3, 0. );
        const cxtype chi[2] = { cxmake( fpsqrt( pp3 * (fptype)0.5 / pp ), 0. ),
                                ( ( pp3 == 0. ) ? cxmake( -nh, 0. )
                                                : cxmake( nh * pvec1, -pvec2 ) / fpsqrt( 2. * pp * pp3 ) ) };
        w[0] = sfomeg[1] * chi[im];
        w[1] = sfomeg[1] * chi[ip];
        w[2] = sfomeg[0] * chi[im];
        w[3] = sfomeg[0] * chi[ip];
      }
#else
      volatile fptype_sv p2 = pvec1 * pvec1 + pvec2 * pvec2 + pvec3 * pvec3; // volatile fixes #736
      const fptype_sv pp = fpmin( pvec0, fpsqrt( p2 ) );
      // Branch A: pp == 0.
      // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
      fptype sqm[2] = { fpsqrt( std::abs( fmass ) ), 0 }; // possibility of negative fermion masses
      sqm[1] = ( fmass < 0 ? -sqm[0] : sqm[0] );          // AV: removed an abs here (as above)
      const int ipA = -( ( 1 - nh ) / 2 ) * nhel;
      const int imA = ( 1 + nh ) / 2 * nhel;
      const cxtype foA_2 = imA * sqm[std::abs( ipA )];
      const cxtype foA_3 = ipA * nsf * sqm[std::abs( ipA )];
      const cxtype foA_4 = imA * nsf * sqm[std::abs( imA )];
      const cxtype foA_5 = ipA * sqm[std::abs( imA )];
      // Branch B: pp != 0.
      const fptype sf[2] = { fptype( 1 + nsf + ( 1 - nsf ) * nh ) * (fptype)0.5,
                             fptype( 1 + nsf - ( 1 - nsf ) * nh ) * (fptype)0.5 };
      fptype_v omega[2] = { fpsqrt( pvec0 + pp ), 0 };
      omega[1] = fmass / omega[0];
      const int ipB = ( 1 + nh ) / 2;
      const int imB = ( 1 - nh ) / 2;
      const fptype_v sfomeg[2] = { sf[0] * omega[ipB], sf[1] * omega[imB] };
      const fptype_v pp3 = fpmax( pp + pvec3, 0. );
      volatile fptype_v ppDENOM = fpternary( pp != 0, pp, 1. );    // hack: ppDENOM[ieppV]=1 if pp[ieppV]==0
      volatile fptype_v pp3DENOM = fpternary( pp3 != 0, pp3, 1. ); // hack: pp3DENOM[ieppV]=1 if pp3[ieppV]==0
      volatile fptype_v chi0r2 = pp3 * 0.5 / ppDENOM;              // volatile fixes #736
      const cxtype_v chi[2] = { cxmake( fpsqrt( chi0r2 ), 0. ),    // hack: dummy[ieppV] is not used if pp[ieppV]==0
                                ( cxternary( ( pp3 == 0. ),
                                             cxmake( -nh, 0. ),
                                             cxmake( (fptype)nh * pvec1, -pvec2 ) / fpsqrt( 2. * ppDENOM * pp3DENOM ) ) ) }; // hack: dummy[ieppV] is not used if pp[ieppV]==0
      const cxtype_v foB_2 = sfomeg[1] * chi[imB];
      const cxtype_v foB_3 = sfomeg[1] * chi[ipB];
      const cxtype_v foB_4 = sfomeg[0] * chi[imB];
      const cxtype_v foB_5 = sfomeg[0] * chi[ipB];
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
      volatile fptype_sv p0p3 = fpmax( pvec0 + pvec3, 0 ); // volatile fixes #736
      volatile fptype_sv sqp0p3 = fpternary( ( pvec1 == 0. and pvec2 == 0. and pvec3 < 0. ),
                                             fptype_sv{ 0 },
                                             fpsqrt( p0p3 ) * (fptype)nsf );
      volatile fptype_v sqp0p3DENOM = fpternary( sqp0p3 != 0, (fptype_sv)sqp0p3, 1. ); // hack: sqp0p3DENOM[ieppV]=1 if sqp0p3[ieppV]==0
      const cxtype_v chi[2] = { cxmake( (fptype_v)sqp0p3, 0. ),
                                cxternary( ( sqp0p3 == 0. ),
                                           cxmake( -nhel, 0. ) * fpsqrt( 2. * pvec0 ),
                                           cxmake( (fptype)nh * pvec1, -pvec2 ) / (const fptype_sv)sqp0p3DENOM ) }; // hack: dummy[ieppV] is not used if sqp0p3[ieppV]==0
#else
      const fptype_sv sqp0p3 = fpternary( ( pvec1 == 0. ) and ( pvec2 == 0. ) and ( pvec3 < 0. ),
                                          0,
                                          fpsqrt( fpmax( pvec0 + pvec3, 0. ) ) * (fptype)nsf );
      const cxtype_sv chi[2] = { cxmake( sqp0p3, 0. ),
                                 ( sqp0p3 == 0. ? cxmake( -nhel, 0. ) * fpsqrt( 2. * pvec0 ) : cxmake( (fptype)nh * pvec1, -pvec2 ) / sqp0p3 ) };
#endif
      if( nh == 1 )
      {
        w[0] = chi[0];
        w[1] = chi[1];
        w[2] = cxzero_sv();
        w[3] = cxzero_sv();
      }
      else
      {
        w[0] = cxzero_sv();
        w[1] = cxzero_sv();
        w[2] = chi[1];
        w[3] = chi[0];
      }
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  opzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* w = W_ACCESS::kernelAccess( fo.w );
    fo.pvec[0] = pvec3 * (fptype)nsf;
    fo.pvec[1] = fptype_sv{ 0 };
    fo.pvec[2] = fptype_sv{ 0 };
    fo.pvec[3] = pvec3 * (fptype)nsf;
    fo.flv_index = flv;
    const int nh = nhel * nsf;
    const cxtype_sv csqp0p3 = cxmake( fpsqrt( 2. * pvec3 ) * (fptype)nsf, 0. );
    w[1] = cxzero_sv();
    w[2] = cxzero_sv();
    if( nh == 1 )
    {
      w[0] = csqp0p3;
      w[3] = cxzero_sv();
    }
    else
    {
      w[0] = cxzero_sv();
      w[3] = csqp0p3;
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == -PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  omzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* w = W_ACCESS::kernelAccess( fo.w );
    fo.pvec[0] = -pvec3 * (fptype)nsf;
    fo.pvec[1] = fptype_sv{ 0 };
    fo.pvec[2] = fptype_sv{ 0 };
    fo.pvec[3] = pvec3 * (fptype)nsf;
    fo.flv_index = flv;
    const int nh = nhel * nsf;
    const cxtype_sv chi1 = cxmake( -nhel, 0. ) * fpsqrt( -2. * pvec3 );
    if( nh == 1 )
    {
      w[0] = cxzero_sv();
      w[1] = chi1;
      w[2] = cxzero_sv();
      w[3] = cxzero_sv();
    }
    else
    {
      w[0] = cxzero_sv();
      w[1] = cxzero_sv();
      w[2] = chi1;
      //w[3] = chi1; // AV: BUG!
      w[3] = cxzero_sv(); // AV: BUG FIX
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PT > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  oxzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ & fo,          // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec0 = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_sv& pvec1 = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_sv& pvec2 = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* w = W_ACCESS::kernelAccess( fo.w );
    fo.pvec[0] = pvec0 * (fptype)nsf;
    fo.pvec[1] = pvec1 * (fptype)nsf;
    fo.pvec[2] = pvec2 * (fptype)nsf;
    fo.pvec[3] = pvec3 * (fptype)nsf;
    fo.flv_index = flv;
    const int nh = nhel * nsf;
    //const float sqp0p3 = sqrtf( pvec0 + pvec3 ) * nsf; // AV: why force a float here?
    const fptype_sv sqp0p3 = fpsqrt( pvec0 + pvec3 ) * (fptype)nsf;
    const cxtype_sv chi0 = cxmake( sqp0p3, 0. );
    const cxtype_sv chi1 = cxmake( (fptype)nh * pvec1 / sqp0p3, -pvec2 / sqp0p3 );
    if( nh == 1 )
    {
      w[0] = chi0;
      w[1] = chi1;
      w[2] = cxzero_sv();
      w[3] = cxzero_sv();
    }
    else
    {
      w[0] = cxzero_sv();
      w[1] = cxzero_sv();
      w[2] = chi1;
      w[3] = chi0;
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------
  // Compute the direction n[5] of the gauge q[5]
  // TODO: Utilise pvec instead of the whole q
  __host__ __device__ INLINE void
  define_gauge_dir( const cxtype_sv q[5], // input: gauge
                    fptype_sv n[5] )      // output: direction
 {
   const fptype_sv qabs2 = q[1].real()*q[1].real()
                       + q[2].real()*q[2].real()
                       + q[3].real()*q[3].real();

   const fptype_sv one = 1. + fptype_sv{0};
   const fptype_sv zero = 0. + fptype_sv{0};

#ifndef MGONGPU_CPPSIMD

    if (qabs2 > 0.f)
    {
      const fptype_sv qabs = fpsqrt(qabs2);

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
      n[3] = fpternary( q[0].real() >= 0.f , -one , one); //possible error in Fortran
      n[4] = zero;
    }
#else
    const fptype_sv qabs = fpsqrt(qabs2);
    const bool_v qsign = (qabs2 > 0.f);
    n[0] = fpternary( q[0].real() >= 0.f , one , -one);
    n[1] = fpternary( qsign , -q[1].real() / qabs , zero );
    n[2] = fpternary( qsign , -q[2].real() / qabs , zero );
    n[3] = fpternary( qsign , -q[3].real() / qabs , fpternary( q[0].real() >= 0.f , one , -one));
    n[4] = zero;
#endif
 }

//--------------------------------------------------------------------------
// Compute propagator factor d  from the gauge q[5] and mass
  __host__ __device__ INLINE void
  calculate_propagator_factor( const cxtype_sv q[5], // input: gauge
                               const fptype mass,    // input: mass
                               fptype_sv *d )        // output: propagator factor
  {
    const fptype_sv one = 1. + fptype_sv{0};
    const fptype_sv  q2 = q[0].real()*q[0].real() - ( q[1].real()*q[1].real() + q[2].real()*q[2].real() + q[3].real()*q[3].real() );
    *d = one / (q2 - mass*mass);
  }

//--------------------------------------------------------------------------
// Multiply the wavefunction by propagator factor from momenta and m
// TODO: check if d should not be used
  template< class W_ACCESS>
  __host__ __device__ INLINE void
  multiply_propagator_factor( const ALOHAOBJ & Ain, // input: wavefunctions
                              const fptype m,       // input: mass
                              ALOHAOBJ Aout )       // output: wavefunctions
  {

    const cxtype_sv* win = W_ACCESS::kernelAccessConst( Ain.w );
    cxtype_sv* wout = W_ACCESS::kernelAccess( Aout.w );

    cxtype_sv q[5];
    fptype_sv n[5];
    cxtype_sv w0[5], w1[5];

    const cxtype_sv cI = cxmake( 0 + fptype_sv{ 0 },  1. + fptype_sv{ 0 }  );

    // Construct q from momenta
    q[0] = cxmake( -Ain.pvec[0], 0.);
    q[1] = cxmake( -Ain.pvec[1], 0.);
    q[2] = cxmake( -Ain.pvec[2], 0.);
    q[3] = cxmake( -Ain.pvec[3], 0.);
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

    fptype_sv nq =
          n[0]*q[0].real()
        - n[1]*q[1].real()
        - n[2]*q[2].real()
        - n[3]*q[3].real();

    cxtype_sv js1 =
        ( n[0]*w0[0]
        - n[1]*w0[1]
        - n[2]*w0[2]
        - n[3]*w0[3] ) / nq;

    cxtype_sv js2 =
        ( q[0]*w0[0]
        - q[1]*w0[1]
        - q[2]*w0[2]
        - q[3]*w0[3]
        - cxconj(q[4]) * w0[4] ) / nq;

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
