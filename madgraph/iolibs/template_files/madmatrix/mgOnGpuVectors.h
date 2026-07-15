// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Nov 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Roiser, A. Valassi, Z. Wettersten (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUVECTORS_H
#define MGONGPUVECTORS_H 1

#include "mgOnGpuCxtypes.h"
#include "mgOnGpuFptypes.h"

#include <iostream>

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

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
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

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
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

  inline fptype
  fpternary( const bool& mask, const fptype& a, const fptype& b )
  {
    return ( mask ? a : b );
  }

  inline cxtype
  cxternary( const bool& mask, const cxtype& a, const cxtype& b )
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

  inline __host__ __device__ fptype
  fpternary( const bool& mask, const fptype& a, const fptype& b )
  {
    return ( mask ? a : b );
  }

  inline __host__ __device__ cxtype
  cxternary( const bool& mask, const cxtype& a, const cxtype& b )
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

  // Scalar-or-vector types: scalar in CUDA, vector or scalar in C++
#ifdef MGONGPUCPP_GPUIMPL
  typedef bool bool_sv;
  typedef fptype fptype_sv;
  typedef fptype2 fptype2_sv;
  typedef unsigned int uint_sv;
  typedef cxtype cxtype_sv;
  typedef cxtype_ref cxtype_sv_ref;
#elif defined MGONGPU_CPPSIMD
  typedef bool_v bool_sv;
  typedef fptype_v fptype_sv;
  typedef fptype2_v fptype2_sv;
  typedef uint_v uint_sv;
  typedef cxtype_v cxtype_sv;
  typedef cxtype_v_ref cxtype_sv_ref;
#else
  typedef bool bool_sv;
  typedef fptype fptype_sv;
  typedef fptype2 fptype2_sv;
  typedef unsigned int uint_sv;
  typedef cxtype cxtype_sv;
  typedef cxtype_ref cxtype_sv_ref;
#endif

  // Scalar-or-vector zeros: scalar in CUDA, vector or scalar in C++
#ifdef MGONGPUCPP_GPUIMPL /* clang-format off */
  inline __host__ __device__ cxtype cxzero_sv(){ return cxtype( 0, 0 ); }
#elif defined MGONGPU_CPPSIMD
  inline cxtype_v cxzero_sv() { return cxtype_v(); } // RRRR=0000 IIII=0000
#else
  inline cxtype cxzero_sv() { return cxtype( 0, 0 ); }
#endif /* clang-format on */

  //==========================================================================

  // Functions and operators for cxtype_sv
  inline __host__ __device__ fptype_sv
  cxabs2( const cxtype_sv& c )
  {
    return cxreal( c ) * cxreal( c ) + cximag( c ) * cximag( c );
  }

  //==========================================================================

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MGONGPUVECTORS_H
