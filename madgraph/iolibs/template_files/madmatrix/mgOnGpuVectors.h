// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Nov 2020) for the MadGraph7 CUDACPP plugin.
// Further modified by: S. Roiser, A. Valassi, Z. Wettersten (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUVECTORS_H
#define MGONGPUVECTORS_H 1

#include "mgOnGpuCxtypes.h"
#include "mgOnGpuFptypes.h"

#include <cassert>
#include <iostream>
#include <type_traits>

//==========================================================================

//------------------------------
// Vector types - C++
//------------------------------

#ifdef __clang__
// If set: return a pair of (fptype&, fptype&) by non-const reference in cxtype_v::operator[]
// This is forbidden in clang ("non-const reference cannot bind to vector element")
// See also https://stackoverflow.com/questions/26554829
//#define MGONGPU_HAS_CPPCXTYPEV_BRK 1 // clang test (compilation fails also on clang 12.0, issue #182)
#undef MGONGPU_HAS_CPPCXTYPEV_BRK // clang default
#elif defined __INTEL_COMPILER
//#define MGONGPU_HAS_CPPCXTYPEV_BRK 1 // icc default?
#undef MGONGPU_HAS_CPPCXTYPEV_BRK // icc test
#else
#define MGONGPU_HAS_CPPCXTYPEV_BRK 1 // gcc default
//#undef MGONGPU_HAS_CPPCXTYPEV_BRK // gcc test (very slightly slower? issue #172)
#endif

//One namespace. Split ber backend.
namespace madmatrix
{
#ifdef MGONGPU_CPPSIMD

  const int neppV = MGONGPU_CPPSIMD;

  // SANITY CHECK: cppAlign must be a multiple of neppV * sizeof(fptype)
  static_assert( mgOnGpu::cppAlign % ( neppV * sizeof( fptype ) ) == 0 );

  // SANITY CHECK: check that neppV is a power of two
  static_assert( ispoweroftwo( neppV ), "neppV is not a power of 2" );

  // --- Type definition (using vector compiler extensions: need -march=...)
  // For gcc: https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html
  // For clang: https://clang.llvm.org/docs/LanguageExtensions.html#vectors-and-extended-vectors
#ifdef __clang__
  typedef fptype fptype_v __attribute__( ( ext_vector_type( neppV ) ) ); // RRRR
#else
  typedef fptype fptype_v __attribute__( ( vector_size( neppV * sizeof(fptype) ), aligned( neppV * sizeof(fptype) ) ) ); // RRRR
#endif

  // Mixed fptypes #537: float for color algebra and double elsewhere
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
  const int neppV2 = MGONGPU_CPPSIMD * 2;
  static_assert( mgOnGpu::cppAlign % ( neppV2 * sizeof( fptype2 ) ) == 0 );
  static_assert( ispoweroftwo( neppV2 ), "neppV2 is not a power of 2" );
#ifdef __clang__
  typedef fptype2 fptype2_v __attribute__( ( ext_vector_type( neppV2 ) ) ); // RRRRRRRR
#else
  typedef fptype2 fptype2_v __attribute__( ( vector_size( neppV2 * sizeof( fptype2 ) ), aligned( neppV2 * sizeof( fptype2 ) ) ) ); // RRRRRRRR
#endif
#else
  typedef fptype_v fptype2_v;
#endif

  // type to hold 2x the SIMD lane, operators work on both halves (autovector) + narrow/widen operators defs
#ifdef MGONGPU_SIMD_DENOM64
  constexpr int neppVD = neppV / 2;
  static_assert( neppV % 2 == 0, "neppV must be even for SIMD denom64" );
#ifdef __clang__
  typedef fptype_momenta fptype_denom_hv __attribute__( ( ext_vector_type( neppVD ) ) ); // one native FP64 register
#else
  typedef fptype_momenta fptype_denom_hv __attribute__( ( vector_size( neppVD * sizeof( fptype_momenta ) ), aligned( neppVD * sizeof( fptype_momenta ) ) ) );
#endif

  struct fptype_denom_sv // 2x FP64 half (v[0], v[1])
  {
    fptype_denom_hv v[2];
    fptype_denom_sv() : v{ fptype_denom_hv{}, fptype_denom_hv{} } {}
    fptype_denom_sv( const fptype_denom_hv& lo, const fptype_denom_hv& hi ) : v{ lo, hi } {}
    fptype_denom_sv( fptype_momenta s ) : v{ fptype_denom_hv{} + s, fptype_denom_hv{} + s } {} // broadcast (M, W, Ccoeff, {0})
  };
  static_assert( sizeof( fptype_denom_sv ) == 2 * sizeof( fptype_denom_hv ), "fptype_denom_sv must be exactly two FP64 halves" );
  inline fptype_denom_sv operator+( const fptype_denom_sv& a ) { return a; }
  inline fptype_denom_sv operator-( const fptype_denom_sv& a ) { return { -a.v[0], -a.v[1] }; }
  inline fptype_denom_sv operator+( const fptype_denom_sv& a, const fptype_denom_sv& b ) { return { a.v[0] + b.v[0], a.v[1] + b.v[1] }; }
  inline fptype_denom_sv operator-( const fptype_denom_sv& a, const fptype_denom_sv& b ) { return { a.v[0] - b.v[0], a.v[1] - b.v[1] }; }
  inline fptype_denom_sv operator*( const fptype_denom_sv& a, const fptype_denom_sv& b ) { return { a.v[0] * b.v[0], a.v[1] * b.v[1] }; }
  inline fptype_denom_sv operator/( const fptype_denom_sv& a, const fptype_denom_sv& b ) { return { a.v[0] / b.v[0], a.v[1] / b.v[1] }; }
  inline fptype_denom_sv operator*( const fptype_denom_sv& a, fptype_momenta s ) { return { a.v[0] * s, a.v[1] * s }; }
  inline fptype_denom_sv operator*( fptype_momenta s, const fptype_denom_sv& a ) { return a * s; }
  inline fptype_denom_sv operator/( const fptype_denom_sv& a, fptype_momenta s ) { return { a.v[0] / s, a.v[1] / s }; }
  inline fptype_denom_sv operator/( fptype_momenta s, const fptype_denom_sv& a ) { return { s / a.v[0], s / a.v[1] }; }
  inline fptype_denom_sv operator+( const fptype_denom_sv& a, fptype_momenta s ) { return { a.v[0] + s, a.v[1] + s }; }
  inline fptype_denom_sv operator+( fptype_momenta s, const fptype_denom_sv& a ) { return a + s; }
  inline fptype_denom_sv operator-( const fptype_denom_sv& a, fptype_momenta s ) { return { a.v[0] - s, a.v[1] - s }; }
  inline fptype_denom_sv operator-( fptype_momenta s, const fptype_denom_sv& a ) { return { s - a.v[0], s - a.v[1] }; }

  inline fptype_v fpdenom_narrow( const fptype_denom_sv& p )
  {
    fptype_v out = {};
    for( int l = 0; l < neppVD; l++ ) { out[l] = (fptype)p.v[0][l]; out[neppVD + l] = (fptype)p.v[1][l]; }
    return out;
  }
  inline fptype_denom_sv fpdenom_widen( const fptype_v& f )
  {
    fptype_denom_hv lo = {}, hi = {};
    for( int l = 0; l < neppVD; l++ ) { lo[l] = (fptype_momenta)f[l]; hi[l] = (fptype_momenta)f[neppVD + l]; }
    return { lo, hi };
  }
#endif

  // --- Type definition (using vector compiler extensions: need -march=...)
  class cxtype_v // no need for "class alignas(2*sizeof(fptype_v)) cxtype_v"
  {
  public:
    // Array initialization: zero-out as "{0}" (C and C++) or as "{}" (C++ only)
    // See https://en.cppreference.com/w/c/language/array_initialization#Notes
    cxtype_v()
      : m_real{ 0 }, m_imag{ 0 } {} // RRRR=0000 IIII=0000
    cxtype_v( const cxtype_v& ) = default;
    cxtype_v( cxtype_v&& ) = default;
    cxtype_v( const fptype_v& r, const fptype_v& i )
      : m_real( r ), m_imag( i ) {}
    cxtype_v( const fptype_v& r )
      : m_real( r ), m_imag{ 0 } {} // IIII=0000
    cxtype_v( const fptype& r )
      : m_real( fptype_v{} + r ), m_imag{ 0 } {} // IIII=0000
    template<typename FP2>
    cxtype_v( const mgOnGpu::cxsmpl<FP2>& c ) // broadcast a scalar complex (amp precision)
      : m_real( fptype_v{} + fptype( c.real() ) ), m_imag( fptype_v{} + fptype( c.imag() ) ) {}
    cxtype_v& operator=( const cxtype_v& ) = default;
    cxtype_v& operator=( cxtype_v&& ) = default;
    cxtype_v& operator+=( const cxtype_v& c )
    {
      m_real += c.real();
      m_imag += c.imag();
      return *this;
    }
    cxtype_v& operator-=( const cxtype_v& c )
    {
      m_real -= c.real();
      m_imag -= c.imag();
      return *this;
    }
#ifdef MGONGPU_HAS_CPPCXTYPEV_BRK
    // NB: THIS IS THE FUNDAMENTAL DIFFERENCE BETWEEN MGONGPU_HAS_CPPCXTYPEV_BRK DEFINED AND NOT DEFINED
    // NB: the alternative "clang" implementation is simpler: it simply does not have any bracket operator[]
    //cxtype_ref operator[]( size_t i ) const { return cxtype_ref( m_real[i], m_imag[i] ); } // gcc14.2 build fails #1004
    cxtype_ref operator[]( size_t i ) { return cxtype_ref( m_real[i], m_imag[i] ); }
    cxtype operator[]( size_t i ) const { return cxtype( m_real[i], m_imag[i] ); }
#endif
    const fptype_v& real() const
    {
      return m_real;
    }
    const fptype_v& imag() const { return m_imag; }
  private:
    fptype_v m_real, m_imag; // RRRRIIII
  };

#ifdef MGONGPU_SIMD_DENOM64
  // complex type to hold 2x the SIMD lane, operators work on both halves (autovector) narrow to cxtype
  class cxtype_denom_sv
  {
  public:
    cxtype_denom_sv() {}
    cxtype_denom_sv( const fptype_denom_sv& r, const fptype_denom_sv& i ) : m_real( r ), m_imag( i ) {}
    cxtype_denom_sv( fptype_momenta r, fptype_momenta i ) : m_real( r ), m_imag( i ) {}          // cId( 0., 1. )
    cxtype_denom_sv( const cxtype_v& c ) : m_real( fpdenom_widen( c.real() ) ), m_imag( fpdenom_widen( c.imag() ) ) {} // COUP (FP32 complex)
    template<typename FP2>
    cxtype_denom_sv( const mgOnGpu::cxsmpl<FP2>& c ) : m_real( fptype_momenta( c.real() ) ), m_imag( fptype_momenta( c.imag() ) ) {} // scalar complex
    const fptype_denom_sv& real() const { return m_real; }
    const fptype_denom_sv& imag() const { return m_imag; }
    explicit operator cxtype_v() const { return cxtype_v( fpdenom_narrow( m_real ), fpdenom_narrow( m_imag ) ); }
  private:
    fptype_denom_sv m_real, m_imag;
  };
  inline cxtype_denom_sv operator+( const cxtype_denom_sv& a ) { return a; }
  inline cxtype_denom_sv operator-( const cxtype_denom_sv& a ) { return { -a.real(), -a.imag() }; }
  inline cxtype_denom_sv operator+( const cxtype_denom_sv& a, const cxtype_denom_sv& b ) { return { a.real() + b.real(), a.imag() + b.imag() }; }
  inline cxtype_denom_sv operator-( const cxtype_denom_sv& a, const cxtype_denom_sv& b ) { return { a.real() - b.real(), a.imag() - b.imag() }; }
  inline cxtype_denom_sv operator+( const fptype_denom_sv& a, const cxtype_denom_sv& b ) { return { a + b.real(), b.imag() }; }
  inline cxtype_denom_sv operator+( const cxtype_denom_sv& a, const fptype_denom_sv& b ) { return { a.real() + b, a.imag() }; }
  inline cxtype_denom_sv operator-( const fptype_denom_sv& a, const cxtype_denom_sv& b ) { return { a - b.real(), -b.imag() }; }
  inline cxtype_denom_sv operator-( const cxtype_denom_sv& a, const fptype_denom_sv& b ) { return { a.real() - b, a.imag() }; }
  inline cxtype_denom_sv operator*( const cxtype_denom_sv& a, const fptype_denom_sv& b ) { return { a.real() * b, a.imag() * b }; }
  inline cxtype_denom_sv operator*( const fptype_denom_sv& a, const cxtype_denom_sv& b ) { return b * a; }
  inline cxtype_denom_sv operator*( const cxtype_denom_sv& a, const cxtype_denom_sv& b )
  {
    return { a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real() };
  }
  inline cxtype_denom_sv operator/( const cxtype_denom_sv& a, const cxtype_denom_sv& b )
  {
    const fptype_denom_sv n = b.real() * b.real() + b.imag() * b.imag();
    return { ( a.real() * b.real() + a.imag() * b.imag() ) / n, ( a.imag() * b.real() - a.real() * b.imag() ) / n };
  }
  inline cxtype_denom_sv operator/( const cxtype_denom_sv& a, const fptype_denom_sv& b ) { return { a.real() / b, a.imag() / b }; }
  inline cxtype_denom_sv operator/( const fptype_denom_sv& a, const cxtype_denom_sv& b )
  {
    const fptype_denom_sv n = b.real() * b.real() + b.imag() * b.imag();
    return { a * b.real() / n, -a * b.imag() / n };
  }
#endif

  // --- Type definition (using vector compiler extensions: need -march=...)
#ifdef __clang__ // https://clang.llvm.org/docs/LanguageExtensions.html#vectors-and-extended-vectors
  typedef unsigned int uint_v __attribute__( ( ext_vector_type( neppV ) ) );
#if defined MGONGPU_FPTYPE_DOUBLE
  typedef long int bool_v __attribute__( ( ext_vector_type( neppV ) ) ); // bbbb
#elif defined MGONGPU_FPTYPE_FLOAT
  typedef int bool_v __attribute__( ( ext_vector_type( neppV ) ) );                         // bbbb
#endif
#else // gcc
  typedef unsigned int uint_v __attribute__( ( vector_size( neppV * sizeof( unsigned int ) ), aligned( neppV * sizeof( unsigned int ) ) ) );
#if defined MGONGPU_FPTYPE_DOUBLE
  typedef long int bool_v __attribute__( ( vector_size( neppV * sizeof( long int ) ), aligned( neppV * sizeof( long int ) ) ) ); // bbbb
#elif defined MGONGPU_FPTYPE_FLOAT
  typedef int bool_v __attribute__( ( vector_size( neppV * sizeof( int ) ), aligned( neppV * sizeof( int ) ) ) ); // bbbb
#endif
#endif

#else // i.e #ifndef MGONGPU_CPPSIMD (this includes #ifdef MGONGPUCPP_GPUIMPL)

  const int neppV = 1;

#endif // #ifdef MGONGPU_CPPSIMD
}

//--------------------------------------------------------------------------

// DANGEROUS! this was mixing different cxtype definitions for CPU and GPU builds (see #318 and #725)
// DO NOT expose typedefs outside the namespace
//using mgOnGpu::neppV;
//#ifdef MGONGPU_CPPSIMD
//using mgOnGpu::fptype_v;
//using mgOnGpu::fptype2_v;
//using mgOnGpu::cxtype_v;
//using mgOnGpu::bool_v;
//#endif

//==========================================================================

//One namespace. Split ber backend.
namespace madmatrix
{
#ifndef MGONGPUCPP_GPUIMPL

  // Printout to stream for user defined types

#ifndef MGONGPU_CPPCXTYPE_CXSMPL // operator<< for cxsmpl has already been defined!
  inline std::ostream&
  operator<<( std::ostream& out, const cxtype& c )
  {
    out << "[" << cxreal( c ) << "," << cximag( c ) << "]";
    //out << cxreal(c) << "+i" << cximag(c);
    return out;
  }
#endif

  /*
#ifdef MGONGPU_CPPSIMD
  inline std::ostream&
  operator<<( std::ostream& out, const bool_v& v )
  {
    out << "{ " << v[0];
    for ( int i=1; i<neppV; i++ ) out << ", " << (bool)(v[i]);
    out << " }";
    return out;
  }
#endif
  */

#ifdef MGONGPU_CPPSIMD
  inline std::ostream&
  operator<<( std::ostream& out, const fptype_v& v )
  {
    out << "{ " << v[0];
    for( int i = 1; i < neppV; i++ ) out << ", " << v[i];
    out << " }";
    return out;
  }
#endif

#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
  inline std::ostream&
  operator<<( std::ostream& out, const fptype2_v& v )
  {
    out << "{ " << v[0];
    for( int i = 1; i < neppV2; i++ ) out << ", " << v[i];
    out << " }";
    return out;
  }
#endif

#ifdef MGONGPU_CPPSIMD
  inline std::ostream&
  operator<<( std::ostream& out, const cxtype_v& v )
  {
#ifdef MGONGPU_HAS_CPPCXTYPEV_BRK
    out << "{ " << v[0];
    for( int i = 1; i < neppV; i++ ) out << ", " << v[i];
#else
    out << "{ " << cxmake( v.real()[0], v.imag()[0] );
    for( int i = 1; i < neppV; i++ ) out << ", " << cxmake( v.real()[i], v.imag()[i] );
#endif
    out << " }";
    return out;
  }
#endif

#ifdef MGONGPU_CPPSIMD
  inline std::ostream&
  operator<<( std::ostream& out, const uint_v& v )
  {
    out << "{ " << v[0];
    for( int i = 1; i < neppV; i++ ) out << ", " << v[i];
    out << " }";
    return out;
  }
#endif

  //--------------------------------------------------------------------------

  /*
  // Printout to std::cout for user defined types

  inline void print( const fptype& f ) { std::cout << f << std::endl; }

#ifdef MGONGPU_CPPSIMD
  inline void print( const fptype_v& v ) { std::cout << v << std::endl; }
#endif

  inline void print( const cxtype& c ) { std::cout << c << std::endl; }

#ifdef MGONGPU_CPPSIMD
  inline void print( const cxtype_v& v ) { std::cout << v << std::endl; }
#endif
  */

  //--------------------------------------------------------------------------

  // Functions and operators for fptype_v

#ifdef MGONGPU_CPPSIMD
  inline fptype_v
  fpsqrt( const volatile fptype_v& v ) // volatile fixes #736
  {
    // See https://stackoverflow.com/questions/18921049/gcc-vector-extensions-sqrt
    fptype_v out = {}; // avoid warning 'out' may be used uninitialized: see #594
    for( int i = 0; i < neppV; i++ )
    {
      volatile fptype outi = 0; // volatile fixes #736
      if( v[i] > 0 ) outi = fpsqrt( (fptype)v[i] );
      out[i] = outi;
    }
    return out;
  }

  inline fptype_v
  fpsqrt( const fptype_v& v )
  {
    // See https://stackoverflow.com/questions/18921049/gcc-vector-extensions-sqrt
    fptype_v out = {}; // avoid warning 'out' may be used uninitialized: see #594
    for( int i = 0; i < neppV; i++ ) out[i] = fpsqrt( v[i] );
    return out;
  }

  inline bool_v
  fpsignbit( const fptype_v& v ) // per-lane std::signbit
  {
    bool_v out = {};
    for( int i = 0; i < neppV; i++ ) out[i] = std::signbit( v[i] );
    return out;
  }
#endif

  /*
#ifdef MGONGPU_CPPSIMD
  inline fptype_v
  fpvmake( const fptype v[neppV] )
  {
    fptype_v out = {}; // see #594
    for ( int i=0; i<neppV; i++ ) out[i] = v[i];
    return out;
  }
#endif
  */

  //--------------------------------------------------------------------------

  // Functions and operators for cxtype_v

#ifdef MGONGPU_CPPSIMD

  /*
  inline cxtype_v
  cxvmake( const cxtype c )
  {
    cxtype_v out;
    for ( int i=0; i<neppV; i++ ) out[i] = c;
    return out;
  }
  */

  inline cxtype_v
  cxmake( const fptype_v& r, const fptype_v& i )
  {
    return cxtype_v( r, i );
  }

  inline cxtype_v
  cxmake( const fptype_v& r, const fptype& i )
  {
    //return cxtype_v( r, fptype_v{i} ); // THIS WAS A BUG! #339
    return cxtype_v( r, fptype_v{} + i ); // IIII=0000+i=iiii
  }

  inline cxtype_v
  cxmake( const fptype& r, const fptype_v& i )
  {
    //return cxtype_v( fptype_v{r}, i ); // THIS WAS A BUG! #339
    return cxtype_v( fptype_v{} + r, i ); // IIII=0000+r=rrrr
  }

  inline const fptype_v&
  cxreal( const cxtype_v& c )
  {
    return c.real(); // returns by reference
  }

  inline const fptype_v&
  cximag( const cxtype_v& c )
  {
    return c.imag(); // returns by reference
  }

  inline const cxtype_v
  cxconj( const cxtype_v& c )
  {
    return cxtype_v( c.real(), -c.imag() );
  }

  inline cxtype_v
  operator+( const cxtype_v& a, const cxtype_v& b )
  {
    return cxtype_v( a.real() + b.real(), a.imag() + b.imag() );
  }

  inline cxtype_v
  operator+( const fptype_v& a, const cxtype_v& b )
  {
    return cxtype_v( a + b.real(), b.imag() );
  }

  inline cxtype_v
  operator+( const cxtype_v& a, const fptype_v& b )
  {
    return cxtype_v( a.real() + b, a.imag() );
  }

  inline cxtype_v
  operator+( const cxtype& a, const cxtype_v& b )
  {
    return cxtype_v( a.real() + b.real(), a.imag() + b.imag() );
  }

  inline cxtype_v
  operator+( const cxtype_v& a, const cxtype& b )
  {
    return cxtype_v( a.real() + b.real(), a.imag() + b.imag() );
  }

  inline const cxtype_v&
  operator+( const cxtype_v& a )
  {
    return a;
  }

  inline cxtype_v
  operator-( const cxtype_v& a, const cxtype_v& b )
  {
    return cxtype_v( a.real() - b.real(), a.imag() - b.imag() );
  }

  inline cxtype_v
  operator-( const fptype& a, const cxtype_v& b )
  {
    return cxtype_v( a - b.real(), -b.imag() );
  }

  inline cxtype_v
  operator-( const cxtype_v& a )
  {
    return 0 - a;
  }

  inline cxtype_v
  operator-( const cxtype_v& a, const fptype& b )
  {
    return cxtype_v( a.real() - b, a.imag() );
  }

  inline cxtype_v
  operator-( const fptype_v& a, const cxtype_v& b )
  {
    return cxtype_v( a - b.real(), -b.imag() );
  }

  inline cxtype_v
  operator-( const cxtype_v& a, const fptype_v& b )
  {
    return cxtype_v( a.real() - b, a.imag() );
  }

  inline cxtype_v
  operator-( const fptype_v& a, const cxtype& b )
  {
    return cxtype_v( a - b.real(), fptype_v{} - b.imag() ); // IIII=0000-b.imag()
  }

  inline cxtype_v
  operator-( const cxtype& a, const cxtype_v& b )
  {
    return cxtype_v( a.real() - b.real(), a.imag() - b.imag() );
  }

  inline cxtype_v
  operator-( const cxtype_v& a, const cxtype& b )
  {
    return cxtype_v( a.real() - b.real(), a.imag() - b.imag() );
  }

  inline cxtype_v
  operator*( const cxtype_v& a, const cxtype_v& b )
  {
    return cxtype_v( a.real() * b.real() - a.imag() * b.imag(), a.imag() * b.real() + a.real() * b.imag() );
  }

  inline cxtype_v
  operator*( const cxtype& a, const cxtype_v& b )
  {
    return cxtype_v( a.real() * b.real() - a.imag() * b.imag(), a.imag() * b.real() + a.real() * b.imag() );
  }

  inline cxtype_v
  operator*( const cxtype_v& a, const cxtype& b )
  {
    return cxtype_v( a.real() * b.real() - a.imag() * b.imag(), a.imag() * b.real() + a.real() * b.imag() );
  }

  inline cxtype_v
  operator*( const fptype& a, const cxtype_v& b )
  {
    return cxtype_v( a * b.real(), a * b.imag() );
  }

  inline cxtype_v
  operator*( const cxtype_v& a, const fptype& b )
  {
    return cxtype_v( a.real() * b, a.imag() * b );
  }

  inline cxtype_v
  operator*( const fptype_v& a, const cxtype_v& b )
  {
    return cxtype_v( a * b.real(), a * b.imag() );
  }

  inline cxtype_v
  operator*( const cxtype_v& a, const fptype_v& b )
  {
    return cxtype_v( a.real() * b, a.imag() * b );
  }

  inline cxtype_v
  operator*( const fptype_v& a, const cxtype& b )
  {
    return cxtype_v( a * b.real(), a * b.imag() );
  }

  inline cxtype_v
  operator*( const cxtype& a, const fptype_v& b )
  {
    return cxtype_v( a.real() * b, a.imag() * b );
  }

  inline cxtype_v
  operator/( const cxtype_v& a, const cxtype_v& b )
  {
    fptype_v bnorm = b.real() * b.real() + b.imag() * b.imag();
    return cxtype_v( ( a.real() * b.real() + a.imag() * b.imag() ) / bnorm,
                     ( a.imag() * b.real() - a.real() * b.imag() ) / bnorm );
  }

  inline cxtype_v
  operator/( const cxtype& a, const cxtype_v& b )
  {
    fptype_v bnorm = b.real() * b.real() + b.imag() * b.imag();
    return cxtype_v( ( cxreal( a ) * b.real() + cximag( a ) * b.imag() ) / bnorm,
                     ( cximag( a ) * b.real() - cxreal( a ) * b.imag() ) / bnorm );
  }

  inline cxtype_v
  operator/( const fptype& a, const cxtype_v& b )
  {
    fptype_v bnorm = b.real() * b.real() + b.imag() * b.imag();
    return cxtype_v( ( a * b.real() ) / bnorm, ( -a * b.imag() ) / bnorm );
  }

  inline cxtype_v
  operator/( const cxtype_v& a, const fptype_v& b )
  {
    return cxtype_v( a.real() / b, a.imag() / b );
  }

  inline cxtype_v
  operator/( const cxtype_v& a, const fptype& b )
  {
    return cxtype_v( a.real() / b, a.imag() / b );
  }

  inline cxtype_v
  operator/( const cxtype_v& a, const cxtype& b )
  {
    const fptype bnorm = b.real() * b.real() + b.imag() * b.imag();
    return cxtype_v( ( a.real() * b.real() + a.imag() * b.imag() ) / bnorm,
                     ( a.imag() * b.real() - a.real() * b.imag() ) / bnorm );
  }

#endif // #ifdef MGONGPU_CPPSIMD

  //--------------------------------------------------------------------------

  // Functions and operators for bool_v (ternary and masks)

#ifdef MGONGPU_CPPSIMD

  inline fptype_v
  fpternary( const bool_v& mask, const fptype_v& a, const fptype_v& b )
  {
    fptype_v out = {}; // see #594
    for( int i = 0; i < neppV; i++ ) out[i] = ( mask[i] ? a[i] : b[i] );
    return out;
  }

  inline fptype_v
  fpternary( const bool_v& mask, const fptype_v& a, const fptype& b )
  {
    fptype_v out = {}; // see #594
    for( int i = 0; i < neppV; i++ ) out[i] = ( mask[i] ? a[i] : b );
    return out;
  }

  inline fptype_v
  fpternary( const bool_v& mask, const fptype& a, const fptype_v& b )
  {
    fptype_v out = {}; // see #594
    for( int i = 0; i < neppV; i++ ) out[i] = ( mask[i] ? a : b[i] );
    return out;
  }

  inline fptype_v
  fpternary( const bool_v& mask, const fptype& a, const fptype& b )
  {
    fptype_v out = {}; // see #594
    for( int i = 0; i < neppV; i++ ) out[i] = ( mask[i] ? a : b );
    return out;
  }

  inline cxtype_v
  cxternary( const bool_v& mask, const cxtype_v& a, const cxtype_v& b )
  {
#ifdef MGONGPU_HAS_CPPCXTYPEV_BRK
    cxtype_v out;
    //for( int i = 0; i < neppV; i++ ) out[i] = ( mask[i] ? a[i] : b[i] ); // OLD error-prone depends on "cxtype_ref& operator=( cxtype_ref&& c )"
    for( int i = 0; i < neppV; i++ ) out[i] = cxtype( mask[i] ? a[i] : b[i] );
    return out;
#else
    fptype_v outr = {}; // see #594
    fptype_v outi = {}; // see #594
    for( int i = 0; i < neppV; i++ )
    {
      outr[i] = ( mask[i] ? a.real()[i] : b.real()[i] );
      outi[i] = ( mask[i] ? a.imag()[i] : b.imag()[i] );
    }
    return cxtype_v( outr, outi );
#endif
  }

  inline cxtype_v
  cxternary( const bool_v& mask, const cxtype_v& a, const cxtype& b )
  {
#ifdef MGONGPU_HAS_CPPCXTYPEV_BRK
    cxtype_v out;
    for( int i = 0; i < neppV; i++ ) out[i] = ( mask[i] ? a[i] : b );
    return out;
#else
    fptype_v outr = {}; // see #594
    fptype_v outi = {}; // see #594
    for( int i = 0; i < neppV; i++ )
    {
      outr[i] = ( mask[i] ? a.real()[i] : b.real() );
      outi[i] = ( mask[i] ? a.imag()[i] : b.imag() );
    }
    return cxtype_v( outr, outi );
#endif
  }

  inline cxtype_v
  cxternary( const bool_v& mask, const cxtype& a, const cxtype_v& b )
  {
#ifdef MGONGPU_HAS_CPPCXTYPEV_BRK
    cxtype_v out;
    for( int i = 0; i < neppV; i++ ) out[i] = ( mask[i] ? a : b[i] );
    return out;
#else
    fptype_v outr = {}; // see #594
    fptype_v outi = {}; // see #594
    for( int i = 0; i < neppV; i++ )
    {
      outr[i] = ( mask[i] ? a.real() : b.real()[i] );
      outi[i] = ( mask[i] ? a.imag() : b.imag()[i] );
    }
    return cxtype_v( outr, outi );
#endif
  }

  inline cxtype_v
  cxternary( const bool_v& mask, const cxtype& a, const cxtype& b )
  {
#ifdef MGONGPU_HAS_CPPCXTYPEV_BRK
    cxtype_v out;
    for( int i = 0; i < neppV; i++ ) out[i] = ( mask[i] ? a : b );
    return out;
#else
    fptype_v outr = {}; // see #594
    fptype_v outi = {}; // see #594
    for( int i = 0; i < neppV; i++ )
    {
      outr[i] = ( mask[i] ? a.real() : b.real() );
      outi[i] = ( mask[i] ? a.imag() : b.imag() );
    }
    return cxtype_v( outr, outi );
#endif
  }

  /*
  inline bool
  maskor( const bool_v& mask )
  {
    bool out = false;
    for ( int i=0; i<neppV; i++ ) out = out || mask[i];
    return out;
  }
  */

  inline bool
  maskand( const bool_v& mask )
  {
    bool out = true;
    for( int i = 0; i < neppV; i++ ) out = out && mask[i];
    return out;
  }

#else // i.e. #ifndef MGONGPU_CPPSIMD

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

  /*
  inline bool
  maskor( const bool& mask )
  {
    return mask;
  }
  */

  inline bool
  maskand( const bool& mask )
  {
    return mask;
  }

#endif // #ifdef MGONGPU_CPPSIMD

  //--------------------------------------------------------------------------

  // Functions and operators for fptype_v (min/max)

#ifdef MGONGPU_CPPSIMD

  inline fptype_v
  fpmax( const fptype_v& a, const fptype_v& b )
  {
    return fpternary( ( b < a ), a, b );
  }

  inline fptype_v
  fpmax( const fptype_v& a, const fptype& b )
  {
    return fpternary( ( b < a ), a, b );
  }

  /*
  inline fptype_v
  fpmax( const fptype& a, const fptype_v& b )
  {
    return fpternary( ( b < a ), a, b );
  }
  */

  inline fptype_v
  fpmin( const fptype_v& a, const fptype_v& b )
  {
    return fpternary( ( a < b ), a, b );
  }

  /*
  inline fptype_v
  fpmin( const fptype_v& a, const fptype& b )
  {
    return fpternary( ( a < b ), a, b );
  }

  inline fptype_v
  fpmin( const fptype& a, const fptype_v& b )
  {
    return fpternary( ( a < b ), a, b );
  }
  */

  //--------------------------------------------------------------------------

  // Vector wrapper over RRRRIIII floating point vectors (cxtype_v_ref)
  // The cxtype_v_ref class (a non-const reference to two fptype_v variables) was originally designed for MemoryAccessCouplings.
  class cxtype_v_ref
  {
  public:
    cxtype_v_ref() = delete;
    cxtype_v_ref( const cxtype_v_ref& ) = delete;
    cxtype_v_ref( cxtype_v_ref&& ) = default; // copy refs
    cxtype_v_ref( fptype_v& r, fptype_v& i )
      : m_preal( &r ), m_pimag( &i ) {} // copy refs
    cxtype_v_ref& operator=( const cxtype_v_ref& ) = delete;
    cxtype_v_ref& operator=( cxtype_v_ref&& c ) = delete;
    cxtype_v_ref& operator=( const cxtype_v& c )
    {
      *m_preal = cxreal( c );
      *m_pimag = cximag( c );
      return *this;
    } // copy values
    __host__ __device__ operator cxtype_v() const { return cxmake( *m_preal, *m_pimag ); }
  private:
    fptype_v *m_preal, *m_pimag; // RRRRIIII
  };

#endif // #ifdef MGONGPU_CPPSIMD

  //--------------------------------------------------------------------------

  // Functions and operators for fptype2_v

#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT

  inline fptype2_v
  fpvmerge( const fptype_v& v1, const fptype_v& v2 )
  {
    // This code is not very efficient! It makes mixed precision FFV/color not faster than double on C++ (#537).
    // I considered various alternatives, including
    // - in gcc12 and clang, __builtin_shufflevector (works with different vector lengths, BUT the same fptype...)
    // - casting vector(4)double to vector(4)float and then assigning via reinterpret_cast... but how to do the cast?
    // Probably the best solution is intrinsics?
    // - see https://stackoverflow.com/questions/5139363
    // - see https://stackoverflow.com/questions/54518744
    /*
    fptype2_v out;
    for( int ieppV = 0; ieppV < neppV; ieppV++ )
    {
      out[ieppV] = v1[ieppV];
      out[ieppV+neppV] = v2[ieppV];
    }
    return out;
    */
#if MGONGPU_CPPSIMD == 2
    fptype2_v out =
      { (fptype2)v1[0], (fptype2)v1[1], (fptype2)v2[0], (fptype2)v2[1] };
#elif MGONGPU_CPPSIMD == 4
    fptype2_v out =
      { (fptype2)v1[0], (fptype2)v1[1], (fptype2)v1[2], (fptype2)v1[3], (fptype2)v2[0], (fptype2)v2[1], (fptype2)v2[2], (fptype2)v2[3] };
#elif MGONGPU_CPPSIMD == 8
    fptype2_v out =
      { (fptype2)v1[0], (fptype2)v1[1], (fptype2)v1[2], (fptype2)v1[3], (fptype2)v1[4], (fptype2)v1[5], (fptype2)v1[6], (fptype2)v1[7], (fptype2)v2[0], (fptype2)v2[1], (fptype2)v2[2], (fptype2)v2[3], (fptype2)v2[4], (fptype2)v2[5], (fptype2)v2[6], (fptype2)v2[7] };
#endif
    return out;
  }

  inline fptype_v
  fpvsplit0( const fptype2_v& v )
  {
    /*
    fptype_v out = {}; // see #594
    for( int ieppV = 0; ieppV < neppV; ieppV++ )
    {
      out[ieppV] = v[ieppV];
    }
    */
#if MGONGPU_CPPSIMD == 2
    fptype_v out =
      { (fptype)v[0], (fptype)v[1] };
#elif MGONGPU_CPPSIMD == 4
    fptype_v out =
      { (fptype)v[0], (fptype)v[1], (fptype)v[2], (fptype)v[3] };
#elif MGONGPU_CPPSIMD == 8
    fptype_v out =
      { (fptype)v[0], (fptype)v[1], (fptype)v[2], (fptype)v[3], (fptype)v[4], (fptype)v[5], (fptype)v[6], (fptype)v[7] };
#endif
    return out;
  }

  inline fptype_v
  fpvsplit1( const fptype2_v& v )
  {
    /*
    fptype_v out = {}; // see #594
    for( int ieppV = 0; ieppV < neppV; ieppV++ )
    {
      out[ieppV] = v[ieppV+neppV];
    }
    */
#if MGONGPU_CPPSIMD == 2
    fptype_v out =
      { (fptype)v[2], (fptype)v[3] };
#elif MGONGPU_CPPSIMD == 4
    fptype_v out =
      { (fptype)v[4], (fptype)v[5], (fptype)v[6], (fptype)v[7] };
#elif MGONGPU_CPPSIMD == 8
    fptype_v out =
      { (fptype)v[8], (fptype)v[9], (fptype)v[10], (fptype)v[11], (fptype)v[12], (fptype)v[13], (fptype)v[14], (fptype)v[15] };
#endif
    return out;
  }

#endif // #if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT

#endif // #ifndef MGONGPUCPP_GPUIMPL

  //==========================================================================

#ifdef MGONGPUCPP_GPUIMPL

  //------------------------------
  // Vector types - CUDA
  //------------------------------

  // Printout to std::cout for user defined types
  inline __host__ __device__ void
  print( const fptype& f )
  {
    printf( "%f\n", f );
  }
  inline __host__ __device__ void
  print( const cxtype& c )
  {
    printf( "[%f, %f]\n", cxreal( c ), cximag( c ) );
  }

  /*
  inline __host__ __device__ const cxtype&
  cxvmake( const cxtype& c )
  {
    return c;
  }
  */

  template<typename FP>
  inline __host__ __device__ FP
  fpternary( const bool& mask, const FP& a, const FP& b )
  {
    return ( mask ? a : b );
  }

  template<typename CX>
  inline __host__ __device__ CX
  cxternary( const bool& mask, const CX& a, const CX& b )
  {
    return ( mask ? a : b );
  }

  inline __host__ __device__ bool
  maskand( const bool& mask )
  {
    return mask;
  }

#endif // #ifdef MGONGPUCPP_GPUIMPL

  //==========================================================================

  // Scalar-or-vector types: scalar in CUDA, vector or scalar in C++.
  // 3 mixed-precision stages: _amp (== fptype), _momenta (== _denom), _colour (== fptype2).
#ifdef MGONGPUCPP_GPUIMPL
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
#elif defined MGONGPU_CPPSIMD
  typedef bool_v bool_sv;
  typedef fptype_v fptype_sv;
  typedef fptype2_v fptype2_sv;
  typedef uint_v uint_sv;
  typedef cxtype_v cxtype_sv;
  typedef cxtype_v_ref cxtype_sv_ref;
  typedef fptype_v fptype_amp_v;              typedef fptype_v fptype_amp_sv;
  typedef cxtype_v cxtype_amp_v;              typedef cxtype_v cxtype_amp_sv;
  typedef fptype2_v fptype_colour_v;          typedef fptype2_v fptype_colour_sv;
  typedef cxtype_v cxtype_colour_v;           typedef cxtype_v cxtype_colour_sv;
#ifdef MGONGPU_SIMD_DENOM64
  // FPTYPE=v: momenta computed 2x to fill for rest
  typedef fptype_denom_sv fptype_momenta_v;     typedef fptype_denom_sv fptype_momenta_sv;
  typedef fptype_denom_sv fptype_denom_v;       // fptype_denom_sv is the struct defined above
  typedef cxtype_denom_sv cxtype_momenta_v;     typedef cxtype_denom_sv cxtype_momenta_sv;
  typedef cxtype_denom_sv cxtype_denom_v;       // cxtype_denom_sv is the class defined above
#else
  // Other SIMD modes: momenta/denom == amp width.
  typedef fptype_v fptype_momenta_v;          typedef fptype_v fptype_momenta_sv;
  typedef fptype_v fptype_denom_v;            typedef fptype_v fptype_denom_sv;
  typedef cxtype_v cxtype_momenta_v;          typedef cxtype_v cxtype_momenta_sv;
  typedef cxtype_v cxtype_denom_v;            typedef cxtype_v cxtype_denom_sv;
#endif
#else
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
#endif

  // narrowing/casting operators for unification
#ifdef MGONGPU_SIMD_DENOM64
  inline fptype_amp_sv fpamp_of_mom( const fptype_momenta_sv& p ) { return fpdenom_narrow( p ); } 
#else
  inline __host__ __device__ fptype_amp_sv fpamp_of_mom( const fptype_momenta_sv& p ) { return static_cast<fptype_amp_sv>( p ); } 
#endif
  inline __host__ __device__ fptype_denom_sv fpdenom_bcast( fptype_denom s ) { return fptype_denom_sv{} + s; }
  // Squared propagator mass m2 = M^2 - i M W at denom precision: real mass and width, or complex mass (CMS).
  inline __host__ __device__
  cxtype_denom mass2( fptype M, fptype W )
  {
    const fptype_denom m = M;
    return cxtype_denom( m * m, -( m * static_cast<fptype_denom>( W ) ) );
  }

  inline __host__ __device__
  cxtype_denom mass2( const cxtype& M )
  {
    const cxtype_denom m( M.real(), M.imag() );
    return m * m;
  }

  // Virtuality P^2 - m2 (the propagator denominator, width included): only P^2 - Re(m2) at denom precision (it cancels near the pole).
  inline __host__ __device__
  cxtype_amp_sv virtuality( const fptype_denom_sv P[4], const cxtype_denom& m2 )
  {
    const fptype_denom_sv p2 = ( P[0] * P[0] ) - ( P[1] * P[1] ) - ( P[2] * P[2] ) - ( P[3] * P[3] );
    return cxmake( fpamp_of_mom( p2 - fpdenom_bcast( m2.real() ) ), static_cast<fptype_amp>( -m2.imag() ) );
  }

  // Propagator coupling Ccoeff * COUP / ( P^2 - m2 ), the division at vertex precision; mass as in mass2.
  template<typename C, typename... Mass>
  inline __host__ __device__
  cxtype_amp_sv denom( fptype ccoeff, const C& coup, const fptype_denom_sv P[4], const Mass&... mass )
  {
    return static_cast<fptype_amp>( ccoeff ) * coup / virtuality( P, mass2( mass... ) );
  }

  // Without coupling: 1 / ( P^2 - m2 ).
  template<typename... Mass>
  inline __host__ __device__
  cxtype_amp_sv denom( const fptype_denom_sv P[4], const Mass&... mass )
  {
    return fptype_amp( 1 ) / virtuality( P, mass2( mass... ) );
  }
  template<typename T>
  inline __host__ __device__ fptype_amp fpamp_scalar( const T& x ) { return static_cast<fptype_amp>( x ); }

  // Scalar-or-vector zeros: scalar in CUDA, vector or scalar in C++
  // Template version for multi-precision (explicit template parameter required)
#ifdef MGONGPUCPP_GPUIMPL /* clang-format off */
  template<typename CX = cxtype>
  inline __host__ __device__ CX cxzero_sv(){ return CX( 0, 0 ); }
#elif defined MGONGPU_CPPSIMD
  inline cxtype_v cxzero_sv() { return cxtype_v(); } // RRRR=0000 IIII=0000
  template<typename CX>
  inline CX cxzero_sv() { return CX{}; }
#else
  template<typename CX = cxtype>
  inline CX cxzero_sv() { return CX{}; }
#endif /* clang-format on */

  //==========================================================================


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

#ifdef MGONGPU_CPPSIMD
  // SIMD version of theta_functionr: a lane-by-lane select
  inline cxtype_v
  theta_functionr( const cxtype_v& cond, const fptype valtrue, const fptype valfalse )
  {
    const fptype_v zero = {};
    return cxternary( cond.real() >= zero, cxmake( valtrue, 0 ), cxmake( valfalse, 0 ) );
  }
#endif

  //==========================================================================

} // end namespace madmatrix

#endif // MGONGPUVECTORS_H
