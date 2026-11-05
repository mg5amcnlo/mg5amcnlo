// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022, based on earlier work by D. Smith) for the MadGraph7 CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUCXTYPES_H
#define MGONGPUCXTYPES_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuFptypes.h"

#include <cassert>
#include <iostream>
#include <type_traits>

//==========================================================================
// COMPLEX TYPES: (PLATFORM-SPECIFIC) HEADERS
//==========================================================================

#include <complex>

// Complex type in c++: std::complex or cxsmpl
#if defined MGONGPU_CPPCXTYPE_STDCOMPLEX
#include <cmath>
#elif not defined MGONGPU_CPPCXTYPE_CXSMPL
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_CPPCXTYPE_STDCOMPLEX or MGONGPU_CPPCXTYPE_CXSMPL
#endif

//==========================================================================
// COMPLEX TYPES: SIMPLE COMPLEX CLASS (cxsmpl)
//==========================================================================

// NB: namespace mgOnGpu includes types which are defined in exactly the same way for CPU and GPU builds (see #318 and #725)
namespace mgOnGpu /* clang-format off */
{
  // The number of floating point types in a complex type (real, imaginary)
  constexpr int nx2 = 2;

  // --- Type definition (simple complex type derived from cxtype_v)
  template<typename FP>
  class cxsmpl
  {
  public:
    __host__ __device__ constexpr cxsmpl() : m_real( 0 ), m_imag( 0 ) {}
    cxsmpl( const cxsmpl& ) = default;
    cxsmpl( cxsmpl&& ) = default;
    __host__ __device__ constexpr cxsmpl( const FP& r, const FP& i = 0 ) : m_real( r ), m_imag( i ) {}
    __host__ __device__ constexpr cxsmpl( const std::complex<FP>& c ) : m_real( c.real() ), m_imag( c.imag() ) {}
    cxsmpl& operator=( const cxsmpl& ) = default;
    cxsmpl& operator=( cxsmpl&& ) = default;
    __host__ __device__ constexpr cxsmpl& operator+=( const cxsmpl& c ) { m_real += c.real(); m_imag += c.imag(); return *this; }
    __host__ __device__ constexpr cxsmpl& operator-=( const cxsmpl& c ) { m_real -= c.real(); m_imag -= c.imag(); return *this; }
    __host__ __device__ constexpr const FP& real() const { return m_real; }
    __host__ __device__ constexpr const FP& imag() const { return m_imag; }
    template<typename FP2> __host__ __device__ constexpr operator cxsmpl<FP2>() const { return cxsmpl<FP2>( m_real, m_imag ); }
#ifdef MGONGPU_CPPCXTYPE_STDCOMPLEX
    template<typename FP2> __host__ __device__ constexpr operator std::complex<FP2>() const { return std::complex<FP2>( m_real, m_imag ); }
#endif
  private:
    FP m_real, m_imag; // RI
  };

  template<typename FP>
  constexpr // (NB: now valid code? in the past this failed as "a constexpr function cannot have a nonliteral return type mgOnGpu::cxsmpl")
  inline __host__ __device__ cxsmpl<FP>
  conj( const cxsmpl<FP>& c )
  {
    return cxsmpl<FP>( c.real(), -c.imag() );
  }
} /* clang-format on */

// Expose the cxsmpl class outside the namespace
using mgOnGpu::cxsmpl;

// Printout to stream for user defined types
namespace madmatrix
{
  template<typename FP>
  inline __host__ std::ostream&
  operator<<( std::ostream& out, const cxsmpl<FP>& c )
  {
    //out << std::complex<FP>( c.real(), c.imag() );
    out << "(" << c.real() << ", " << c.imag() << ")"; // add a space after the comma
    return out;
  }

  // Operators for cxsmpl
  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator+( const cxsmpl<FP> a )
  {
    return a;
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator-( const cxsmpl<FP>& a )
  {
    return cxsmpl<FP>( -a.real(), -a.imag() );
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator+( const cxsmpl<FP>& a, const cxsmpl<FP>& b )
  {
    return cxsmpl<FP>( a.real() + b.real(), a.imag() + b.imag() );
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator+( const FP& a, const cxsmpl<FP>& b )
  {
    return cxsmpl<FP>( a, 0 ) + b;
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator-( const cxsmpl<FP>& a, const cxsmpl<FP>& b )
  {
    return cxsmpl<FP>( a.real() - b.real(), a.imag() - b.imag() );
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator-( const FP& a, const cxsmpl<FP>& b )
  {
    return cxsmpl<FP>( a, 0 ) - b;
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator*( const cxsmpl<FP>& a, const cxsmpl<FP>& b )
  {
    return cxsmpl<FP>( a.real() * b.real() - a.imag() * b.imag(), a.imag() * b.real() + a.real() * b.imag() );
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator*( const FP& a, const cxsmpl<FP>& b )
  {
    return cxsmpl<FP>( a, 0 ) * b;
  }

  inline __host__ __device__ constexpr cxsmpl<float>
  operator*( const double& a, const cxsmpl<float>& b )
  {
    return cxsmpl<float>( a, 0 ) * b;
  }

  inline __host__ __device__ constexpr cxsmpl<float>
  operator*( const cxsmpl<float>& a, const double& b )
  {
    return a * cxsmpl<float>( b, 0 );
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator/( const cxsmpl<FP>& a, const cxsmpl<FP>& b )
  {
    FP bnorm = b.real() * b.real() + b.imag() * b.imag();
    return cxsmpl<FP>( ( a.real() * b.real() + a.imag() * b.imag() ) / bnorm,
                       ( a.imag() * b.real() - a.real() * b.imag() ) / bnorm );
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator/( const FP& a, const cxsmpl<FP>& b )
  {
    return cxsmpl<FP>( a, 0 ) / b;
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator+( const cxsmpl<FP>& a, const FP& b )
  {
    return a + cxsmpl<FP>( b, 0 );
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator-( const cxsmpl<FP>& a, const FP& b )
  {
    return a - cxsmpl<FP>( b, 0 );
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator*( const cxsmpl<FP>& a, const FP& b )
  {
    return a * cxsmpl<FP>( b, 0 );
  }

  template<typename FP>
  inline __host__ __device__ constexpr cxsmpl<FP>
  operator/( const cxsmpl<FP>& a, const FP& b )
  {
    return a / cxsmpl<FP>( b, 0 );
  }
}

//==========================================================================
// COMPLEX TYPES: (PLATFORM-SPECIFIC) TYPEDEFS
//==========================================================================

namespace madmatrix
{
  // --- Type definitions (complex type: cxtype)
#if defined MGONGPU_CPPCXTYPE_STDCOMPLEX
  typedef std::complex<fptype> cxtype;
#else
  typedef cxsmpl<fptype> cxtype;
#endif

  // SANITY CHECK: memory access may be based on casts of fptype[2] to cxtype (e.g. for wavefunctions)
  static_assert( sizeof( cxtype ) == mgOnGpu::nx2 * sizeof( fptype ), "sizeof(cxtype) is not 2*sizeof(fptype)" );

  // --- Multi-precision complex types (same platform logic as cxtype)
#if defined MGONGPU_CPPCXTYPE_STDCOMPLEX
  typedef std::complex<fptype_momenta> cxtype_momenta;
  typedef cxtype_momenta cxtype_denom;
  typedef std::complex<fptype_amp> cxtype_amp;
  typedef std::complex<fptype_colour> cxtype_colour;
#else
  typedef cxsmpl<fptype_momenta> cxtype_momenta;
  typedef cxtype_momenta cxtype_denom;
  typedef cxsmpl<fptype_amp> cxtype_amp;
  typedef cxsmpl<fptype_colour> cxtype_colour;
#endif

  // SANITY CHECKS
  static_assert( sizeof( cxtype_momenta ) == mgOnGpu::nx2 * sizeof( fptype_momenta ), "sizeof(cxtype_momenta) is not 2*sizeof(fptype_momenta)" );
  static_assert( sizeof( cxtype_amp ) == mgOnGpu::nx2 * sizeof( fptype_amp ), "sizeof(cxtype_amp) is not 2*sizeof(fptype_amp)" );
  static_assert( sizeof( cxtype_colour ) == mgOnGpu::nx2 * sizeof( fptype2 ), "sizeof(cxtype_colour) is not 2*sizeof(fptype2)" );
}

// DANGEROUS! this was mixing different cxtype definitions for CPU and GPU builds (see #318 and #725)
// DO NOT expose typedefs and operators outside the namespace
//using mgOnGpu::cxtype;

//==========================================================================
// COMPLEX TYPES: (PLATFORM-SPECIFIC) FUNCTIONS AND OPERATORS
//==========================================================================

namespace madmatrix
{
#if defined MGONGPU_CPPCXTYPE_CXSMPL

  //------------------------------
  // C++ - using cxsmpl
  //------------------------------

  inline __host__ __device__ cxtype
  cxmake( const fptype& r, const fptype& i )
  {
    return cxtype( r, i ); // cxsmpl constructor
  }

  inline __host__ __device__ fptype
  cxreal( const cxtype& c )
  {
    return c.real(); // cxsmpl::real()
  }

  inline __host__ __device__ fptype
  cximag( const cxtype& c )
  {
    return c.imag(); // cxsmpl::imag()
  }

  inline __host__ __device__ cxtype
  cxconj( const cxtype& c )
  {
    return conj( c ); // conj( cxsmpl )
  }

  inline __host__ cxtype                 // NOT __device__
  cxmake( const std::complex<float>& c ) // std::complex to cxsmpl (float-to-float or float-to-double)
  {
    return cxmake( c.real(), c.imag() );
  }

  inline __host__ cxtype                  // NOT __device__
  cxmake( const std::complex<double>& c ) // std::complex to cxsmpl (double-to-float or double-to-double)
  {
    return cxmake( c.real(), c.imag() );
  }

  inline __host__ __device__ const cxtype&
  cxmake( const cxtype& c ) // cxsmpl to cxsmpl (identity)
  {
    return c;
  }

  template<typename FP, typename = std::enable_if_t<std::is_floating_point<FP>::value>>
  inline __host__ __device__ cxsmpl<FP>
  cxmake( const FP& r, const FP& i ) { return cxsmpl<FP>( r, i ); }

  template<typename FP>
  inline __host__ __device__ FP
  cxreal( const cxsmpl<FP>& c ) { return c.real(); }

  template<typename FP>
  inline __host__ __device__ FP
  cximag( const cxsmpl<FP>& c ) { return c.imag(); }

  template<typename FP>
  inline __host__ __device__ cxsmpl<FP>
  cxconj( const cxsmpl<FP>& c ) { return conj( c ); }

#endif // #if defined MGONGPU_CPPCXTYPE_CXSMPL

  //==========================================================================

#if defined MGONGPU_CPPCXTYPE_STDCOMPLEX

  //------------------------------
  // C++ - using std::complex
  //------------------------------

  inline cxtype
  cxmake( const fptype& r, const fptype& i )
  {
    return cxtype( r, i ); // std::complex<fptype> constructor
  }

  inline fptype
  cxreal( const cxtype& c )
  {
    return c.real(); // std::complex<fptype>::real()
  }

  inline fptype
  cximag( const cxtype& c )
  {
    return c.imag(); // std::complex<fptype>::imag()
  }

  inline cxtype
  cxconj( const cxtype& c )
  {
    return conj( c ); // conj( std::complex<fptype> )
  }

  inline const cxtype&
  cxmake( const cxtype& c ) // std::complex to std::complex (float-to-float or double-to-double)
  {
    return c;
  }

#if defined MGONGPU_FPTYPE_FLOAT
  inline cxtype
  cxmake( const std::complex<double>& c ) // std::complex to std::complex (cast double-to-float)
  {
    return cxmake( (fptype)c.real(), (fptype)c.imag() );
  }
#endif

  template<typename FP, typename = std::enable_if_t<std::is_floating_point<FP>::value>>
  inline std::complex<FP>
  cxmake( const FP& r, const FP& i ) { return std::complex<FP>( r, i ); }

  template<typename FP>
  inline FP
  cxreal( const std::complex<FP>& c ) { return c.real(); }

  template<typename FP>
  inline FP
  cximag( const std::complex<FP>& c ) { return c.imag(); }

  template<typename FP>
  inline std::complex<FP>
  cxconj( const std::complex<FP>& c ) { return conj( c ); }

#endif // #if defined MGONGPU_CPPCXTYPE_STDCOMPLEX

} // end namespace madmatrix

//==========================================================================
// COMPLEX TYPES: WRAPPER OVER RI FLOATING POINT PAIR (cxtype_ref)
//==========================================================================

//One namespace. Split ber backend.
namespace madmatrix
{
  // The cxtype_ref class (a const reference to two non-const fp variables) was originally designed for cxtype_v::operator[]
  // It used to be included in the code only when MGONGPU_HAS_CPPCXTYPEV_BRK (originally MGONGPU_HAS_CPPCXTYPE_REF) is defined
  // It is now always included in the code because it is needed also to access an fptype wavefunction buffer as a cxtype
  class cxtype_ref
  {
  public:
    cxtype_ref() = delete;
    cxtype_ref( const cxtype_ref& ) = delete;
    cxtype_ref( cxtype_ref&& ) = default; // copy const refs
    __host__ __device__ cxtype_ref( fptype& r, fptype& i )
      : m_preal( &r ), m_pimag( &i ) {} // copy (create from) const refs
    cxtype_ref& operator=( const cxtype_ref& ) = delete;
    //__host__ __device__ cxtype_ref& operator=( cxtype_ref&& c ) {...} // REMOVED! Should copy refs or copy values? No longer needed in cxternary
    __host__ __device__ cxtype_ref& operator=( const cxtype& c )
    {
      *m_preal = cxreal( c );
      *m_pimag = cximag( c );
      return *this;
    } // copy (assign) non-const values
    __host__ __device__ operator cxtype() const { return cxmake( *m_preal, *m_pimag ); }
  private:
    fptype* const m_preal; // const pointer to non-const fptype R
    fptype* const m_pimag; // const pointer to non-const fptype I
  };

  // Printout to stream for user defined types
  inline __host__ __device__ std::ostream&
  operator<<( std::ostream& out, const cxtype_ref& c )
  {
    out << (cxtype)c;
    return out;
  }

  // The cxtype_amp_ref class: same as cxtype_ref but for fptype_amp buffers
  class cxtype_amp_ref
  {
  public:
    cxtype_amp_ref() = delete;
    cxtype_amp_ref( const cxtype_amp_ref& ) = delete;
    cxtype_amp_ref( cxtype_amp_ref&& ) = default;
    __host__ __device__ cxtype_amp_ref( fptype_amp& r, fptype_amp& i )
      : m_preal( &r ), m_pimag( &i ) {}
    cxtype_amp_ref& operator=( const cxtype_amp_ref& ) = delete;
    __host__ __device__ cxtype_amp_ref& operator=( const cxtype_amp& c )
    {
      *m_preal = cxreal( c );
      *m_pimag = cximag( c );
      return *this;
    }
    __host__ __device__ operator cxtype_amp() const { return cxmake( *m_preal, *m_pimag ); }
  private:
    fptype_amp* const m_preal;
    fptype_amp* const m_pimag;
  };

  inline __host__ __device__ std::ostream&
  operator<<( std::ostream& out, const cxtype_amp_ref& c )
  {
    out << (cxtype_amp)c;
    return out;
  }

  //--------------------------------------------------------------------------

  // all needed from mgOnGpuVectors.h for cpu
  const int neppV = 1;

#ifndef MGONGPU_CPPCXTYPE_CXSMPL // operator<< for cxsmpl has already been defined!
  inline std::ostream&
  operator<<( std::ostream& out, const cxtype& c )
  {
    out << "[" << cxreal( c ) << "," << cximag( c ) << "]";
    return out;
  }
#endif

  template<typename FP>
  inline FP
  fpternary( const bool& mask, const FP& a, const FP& b )
  {
    return ( mask ? a : b );
  }

  template<typename CX>
  inline CX
  cxternary( const bool& mask, const CX& a, const CX& b )
  {
    return ( mask ? a : b );
  }

  inline bool
  maskand( const bool& mask )
  {
    return mask;
  }

  // Scalar-or-vector types: scalar in CUDA, vector or scalar in C++.
  // 3 mixed-precision stages: _amp (== fptype), _momenta (== _denom), _colour (== fptype2).
  typedef bool bool_sv;
  typedef fptype fptype_sv;
  typedef fptype2 fptype2_sv;
  typedef unsigned int uint_sv;
  typedef cxtype cxtype_sv;
  typedef cxtype_ref cxtype_sv_ref;
  typedef fptype_momenta fptype_momenta_sv;   typedef fptype_momenta fptype_momenta_v;
  typedef fptype_denom fptype_denom_sv;       typedef fptype_denom fptype_denom_v;
  typedef fptype_amp fptype_amp_sv;           typedef fptype_amp fptype_amp_v;
  typedef fptype_colour fptype_colour_sv;     typedef fptype_colour fptype_colour_v;
  typedef cxtype_momenta cxtype_momenta_sv;   typedef cxtype_momenta cxtype_momenta_v;
  typedef cxtype_denom cxtype_denom_sv;       typedef cxtype_denom cxtype_denom_v;
  typedef cxtype_amp cxtype_amp_sv;           typedef cxtype_amp cxtype_amp_v;
  typedef cxtype_colour cxtype_colour_sv;     typedef cxtype_colour cxtype_colour_v;

  // narrowing/casting operators for unification
#ifdef MGONGPU_SIMD_DENOM64
  inline fptype_amp_sv fpamp_of_mom( const fptype_momenta_sv& p ) { return fpdenom_narrow( p ); }
#else
  inline __host__ __device__ fptype_amp_sv fpamp_of_mom( const fptype_momenta_sv& p ) { return static_cast<fptype_amp_sv>( p ); }
#endif
  inline fptype_denom_sv fpdenom_bcast( fptype_denom s ) { return fptype_denom_sv{} + s; }
  template<typename T>
  inline __host__ __device__ fptype_amp fpamp_scalar( const T& x ) { return static_cast<fptype_amp>( x ); }

  // Scalar-or-vector zeros: scalar in CUDA, vector or scalar in C++
  // Template version for multi-precision (explicit template parameter required)
  template<typename CX = cxtype>
  inline CX cxzero_sv() { return CX{}; }

  // Functions and operators for cxtype_sv (and multi-precision variant)
  template<typename CX>
  inline __host__ __device__ auto
  cxabs2( const CX& c ) -> decltype( cxreal( c ) * cxreal( c ) + cximag( c ) * cximag( c ) )
  {
    return cxreal( c ) * cxreal( c ) + cximag( c ) * cximag( c );
  }

  // ALOHA raises the denominator of a custom propagator to an integer power (the
  // squared Breit-Wigner of the SMEFTsim width corrections, for instance). There
  // is no std::pow overload for cxtype_v, so expand the power by repeated
  // multiplication: the exponent comes from the UFO propagator and is a small
  // positive integer.
  template<class T>
  inline __host__ __device__ T
  cxpow( const T& base, const fptype n )
  {
    const int k = (int)n;
    assert( (fptype)k == n && k >= 1 ); // only positive integer powers are supported
    T out = base;
    for( int i = 1; i < k; i++ ) out = out * base;
    return out;
  }

  // Heaviside step of the $-excluded propagator (ALOHA 'P1D' tag): valtrue
  // where cond >= 0 (outside the on-shell window), valfalse inside it. Same
  // Theta(0) = 1 convention as the Fortran THETA_FUNCTIONR. Only the real part
  // of cond is physical.
  template<class CX>
  inline __host__ __device__ CX
  theta_functionr( const CX& cond, const fptype valtrue, const fptype valfalse )
  {
    return ( cxreal( cond ) >= 0 ) ? CX( valtrue, 0 ) : CX( valfalse, 0 );
  }

} // end namespace madmatrix

//==========================================================================

#endif // MGONGPUCXTYPES_H
