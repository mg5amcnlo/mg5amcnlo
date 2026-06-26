// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022, based on earlier work by D. Smith) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUCXTYPES_H
#define MGONGPUCXTYPES_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuFptypes.h"

#include <iostream>

//==========================================================================
// COMPLEX TYPES: (PLATFORM-SPECIFIC) HEADERS
//==========================================================================

#include <complex>

// Complex type in cuda: thrust or cucomplex or cxsmpl
#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
#if defined MGONGPU_CUCXTYPE_THRUST
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare" // for icpx2021/clang13 (https://stackoverflow.com/a/15864661)
#include <thrust/complex.h>
#pragma clang diagnostic pop
#elif defined MGONGPU_CUCXTYPE_CUCOMPLEX
#include <cuComplex.h>
#elif not defined MGONGPU_CUCXTYPE_CXSMPL
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_CUCXTYPE_THRUST or MGONGPU_CUCXTYPE_CUCOMPLEX or MGONGPU_CUCXTYPE_CXSMPL
#endif
// Complex type in HIP: cxsmpl
#elif defined __HIPCC__
#if not defined MGONGPU_HIPCXTYPE_CXSMPL
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_HIPCXTYPE_CXSMPL
#endif
#else
// Complex type in c++ or HIP: std::complex or cxsmpl
#if defined MGONGPU_CPPCXTYPE_STDCOMPLEX
#include <cmath>
#elif not defined MGONGPU_CPPCXTYPE_CXSMPL
#error You must CHOOSE (ONE AND) ONLY ONE of MGONGPU_CPPCXTYPE_STDCOMPLEX or MGONGPU_CPPCXTYPE_CXSMPL
#endif
#endif

//==========================================================================
// COMPLEX TYPES: INSTRUMENTED CUCOMPLEX CLASS (cucomplex)
//==========================================================================

#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
#if defined MGONGPU_CUCXTYPE_CUCOMPLEX
namespace mg5amcGpu
{
#if defined MGONGPU_FPTYPE_DOUBLE
  class cucomplex
  {
  public:
    __host__ __device__ cucomplex( const double& r = 0, const double& i = 0 )
      : m_ri( make_cuDoubleComplex( r, i ) ) {}
    __host__ __device__ constexpr cucomplex( const cuDoubleComplex& ri )
      : m_ri( ri ) {}
    //__host__ __device__ operator cuDoubleComplex&() { return m_ri; }
    __host__ __device__ constexpr operator cuDoubleComplex() const { return m_ri; }
    __host__ __device__ double real() const { return cuCreal( m_ri ); }
    __host__ __device__ double imag() const { return cuCimag( m_ri ); }
    inline __host__ __device__ cucomplex& operator+=( const cucomplex& c )
    {
      m_ri = cuCadd( m_ri, c );
      return *this;
    }
    inline __host__ __device__ cucomplex& operator-=( const cucomplex& c )
    {
      m_ri = cuCsub( m_ri, c );
      return *this;
    }
  private:
    cuDoubleComplex m_ri;
  };
#elif defined MGONGPU_FPTYPE_FLOAT
  class cucomplex
  {
  public:
    __host__ __device__ cucomplex( const float& r = 0, const float& i = 0 )
      : m_ri( make_cuFloatComplex( r, i ) ) {}
    __host__ __device__ constexpr cucomplex( const cuFloatComplex& ri )
      : m_ri( ri ) {}
    //__host__ __device__ operator cuFloatComplex&() { return m_ri; }
    __host__ __device__ constexpr operator cuFloatComplex() const { return m_ri; }
    __host__ __device__ float real() const { return cuCrealf( m_ri ); }
    __host__ __device__ float imag() const { return cuCimagf( m_ri ); }
    inline __host__ __device__ cucomplex& operator+=( const cucomplex& c )
    {
      m_ri = cuCaddf( m_ri, c );
      return *this;
    }
    inline __host__ __device__ cucomplex& operator-=( const cucomplex& c )
    {
      m_ri = cuCsubf( m_ri, c );
      return *this;
    }
  private:
    cuFloatComplex m_ri;
  };
#endif
}
#endif
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
#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
#ifdef MGONGPU_CUCXTYPE_THRUST
    template<typename FP2> __host__ __device__ constexpr operator thrust::complex<FP2>() const { return thrust::complex<FP2>( m_real, m_imag ); }
#elif defined MGONGPU_CUCXTYPE_CUCOMPLEX 
    __host__ __device__ constexpr operator mg5amcGpu::cucomplex() const { return mg5amcGpu::cucomplex( m_real, m_imag ); }
#endif
#else
#ifdef MGONGPU_CPPCXTYPE_STDCOMPLEX
    template<typename FP2> __host__ __device__ constexpr operator std::complex<FP2>() const { return std::complex<FP2>( m_real, m_imag ); }
#endif
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
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
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

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  // --- Type definitions (complex type: cxtype)
#ifdef __CUDACC__ // this must be __CUDACC__ (not MGONGPUCPP_GPUIMPL)
#if defined MGONGPU_CUCXTYPE_THRUST
  typedef thrust::complex<fptype> cxtype;
#elif defined MGONGPU_CUCXTYPE_CUCOMPLEX
  typedef cucomplex cxtype;
#else
  typedef cxsmpl<fptype> cxtype;
#endif
#else // c++
#if defined MGONGPU_CPPCXTYPE_STDCOMPLEX
  typedef std::complex<fptype> cxtype;
#else
  typedef cxsmpl<fptype> cxtype;
#endif
#endif

  // SANITY CHECK: memory access may be based on casts of fptype[2] to cxtype (e.g. for wavefunctions)
  static_assert( sizeof( cxtype ) == mgOnGpu::nx2 * sizeof( fptype ), "sizeof(cxtype) is not 2*sizeof(fptype)" );
}

// DANGEROUS! this was mixing different cxtype definitions for CPU and GPU builds (see #318 and #725)
// DO NOT expose typedefs and operators outside the namespace
//using mgOnGpu::cxtype;

//==========================================================================
// COMPLEX TYPES: (PLATFORM-SPECIFIC) FUNCTIONS AND OPERATORS
//==========================================================================

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
#if defined MGONGPU_CUCXTYPE_CXSMPL or defined MGONGPU_HIPCXTYPE_CXSMPL or defined MGONGPU_CPPCXTYPE_CXSMPL

  //------------------------------
  // CUDA or C++ - using cxsmpl
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

#endif // #if defined MGONGPU_CUCXTYPE_CXSMPL or defined MGONGPU_HIPCXTYPE_CXSMPL or defined MGONGPU_CPPCXTYPE_CXSMPL

  //==========================================================================

#if defined __CUDACC__ and defined MGONGPU_CUCXTYPE_THRUST // cuda + thrust (this must be __CUDACC__ and not MGONGPUCPP_GPUIMPL)

  //------------------------------
  // CUDA - using thrust::complex
  //------------------------------

  inline __host__ __device__ cxtype
  cxmake( const fptype& r, const fptype& i )
  {
    return cxtype( r, i ); // thrust::complex<fptype> constructor
  }

  inline __host__ __device__ fptype
  cxreal( const cxtype& c )
  {
    return c.real(); // thrust::complex<fptype>::real()
  }

  inline __host__ __device__ fptype
  cximag( const cxtype& c )
  {
    return c.imag(); // thrust::complex<fptype>::imag()
  }

  inline __host__ __device__ cxtype
  cxconj( const cxtype& c )
  {
    return conj( c ); // conj( thrust::complex<fptype> )
  }

  inline __host__ __device__ const cxtype&
  cxmake( const cxtype& c )
  {
    return c;
  }

#endif // #if defined __CUDACC__ and defined MGONGPU_CUCXTYPE_THRUST

  //==========================================================================

#if defined __CUDACC__ and defined MGONGPU_CUCXTYPE_CUCOMPLEX // cuda + cucomplex (this must be __CUDACC__ and not MGONGPUCPP_GPUIMPL)

  //------------------------------
  // CUDA - using cuComplex
  //------------------------------

#if defined MGONGPU_FPTYPE_DOUBLE // cuda + cucomplex + double

  //+++++++++++++++++++++++++
  // cuDoubleComplex ONLY
  //+++++++++++++++++++++++++

  inline __host__ __device__ cxtype
  cxmake( const fptype& r, const fptype& i )
  {
    return make_cuDoubleComplex( r, i );
  }

  inline __host__ __device__ fptype
  cxreal( const cxtype& c )
  {
    return cuCreal( c ); // returns by value
  }

  inline __host__ __device__ fptype
  cximag( const cxtype& c )
  {
    return cuCimag( c ); // returns by value
  }

  inline __host__ __device__ cxtype
  operator+( const cxtype& a, const cxtype& b )
  {
    return cuCadd( a, b );
  }

  inline __host__ __device__ cxtype
  operator-( const cxtype& a, const cxtype& b )
  {
    return cuCsub( a, b );
  }

  inline __host__ __device__ cxtype
  operator*( const cxtype& a, const cxtype& b )
  {
    return cuCmul( a, b );
  }

  inline __host__ __device__ cxtype
  operator/( const cxtype& a, const cxtype& b )
  {
    return cuCdiv( a, b );
  }

  inline __host__ std::ostream&
  operator<<( std::ostream& out, const cxtype& c )
  {
    //out << std::complex<double>( cxreal( c ), cximag( c ) );
    out << "(" << cxreal( c ) << ", " << cximag( c ) << ")"; // add a space after the comma
    return out;
  }

#elif defined MGONGPU_FPTYPE_FLOAT // cuda + cucomplex + float

  //+++++++++++++++++++++++++
  // cuFloatComplex ONLY
  //+++++++++++++++++++++++++

  inline __host__ __device__ cxtype
  cxmake( const fptype& r, const fptype& i )
  {
    return make_cuFloatComplex( r, i );
  }

  inline __host__ __device__ fptype
  cxreal( const cxtype& c )
  {
    return cuCrealf( c ); // returns by value
  }

  inline __host__ __device__ fptype
  cximag( const cxtype& c )
  {
    return cuCimagf( c ); // returns by value
  }

  inline __host__ __device__ cxtype
  operator+( const cxtype& a, const cxtype& b )
  {
    return cuCaddf( a, b );
  }

  inline __host__ __device__ cxtype
  operator-( const cxtype& a, const cxtype& b )
  {
    return cuCsubf( a, b );
  }

  inline __host__ __device__ cxtype
  operator*( const cxtype& a, const cxtype& b )
  {
    return cuCmulf( a, b );
  }

  inline __host__ __device__ cxtype
  operator/( const cxtype& a, const cxtype& b )
  {
    return cuCdivf( a, b );
  }

  inline __host__ cxtype                  // NOT __device__
  cxmake( const std::complex<double>& c ) // std::complex to cucomplex (cast double-to-float)
  {
    return cxmake( (fptype)c.real(), (fptype)c.imag() );
  }

  inline __host__ std::ostream&
  operator<<( std::ostream& out, const cxtype& c )
  {
    //out << std::complex<float>( cxreal( c ), cximag( c ) );
    out << "(" << cxreal( c ) << ", " << cximag( c ) << ")"; // add a space after the comma
    return out;
  }

#endif

  //+++++++++++++++++++++++++
  // cuDoubleComplex OR
  // cuFloatComplex
  //+++++++++++++++++++++++++

  inline __host__ __device__ cxtype
  operator+( const cxtype a )
  {
    return a;
  }

  inline __host__ __device__ cxtype
  operator-( const cxtype& a )
  {
    return cxmake( -cxreal( a ), -cximag( a ) );
  }

  inline __host__ __device__ cxtype
  operator+( const fptype& a, const cxtype& b )
  {
    return cxmake( a, 0 ) + b;
  }

  inline __host__ __device__ cxtype
  operator-( const fptype& a, const cxtype& b )
  {
    return cxmake( a, 0 ) - b;
  }

  inline __host__ __device__ cxtype
  operator*( const fptype& a, const cxtype& b )
  {
    return cxmake( a, 0 ) * b;
  }

  inline __host__ __device__ cxtype
  operator/( const fptype& a, const cxtype& b )
  {
    return cxmake( a, 0 ) / b;
  }

  inline __host__ __device__ cxtype
  operator+( const cxtype& a, const fptype& b )
  {
    return a + cxmake( b, 0 );
  }

  inline __host__ __device__ cxtype
  operator-( const cxtype& a, const fptype& b )
  {
    return a - cxmake( b, 0 );
  }

  inline __host__ __device__ cxtype
  operator*( const cxtype& a, const fptype& b )
  {
    return a * cxmake( b, 0 );
  }

  inline __host__ __device__ cxtype
  operator/( const cxtype& a, const fptype& b )
  {
    return a / cxmake( b, 0 );
  }

  inline __host__ __device__ cxtype
  cxconj( const cxtype& c )
  {
    return cxmake( cxreal( c ), -cximag( c ) );
  }

  inline __host__ cxtype                  // NOT __device__
  cxmake( const std::complex<fptype>& c ) // std::complex to cucomplex (float-to-float or double-to-double)
  {
    return cxmake( c.real(), c.imag() );
  }

#endif // #if defined __CUDACC__ and defined MGONGPU_CUCXTYPE_CUCOMPLEX

  //==========================================================================

#if not defined __CUDACC__ and defined MGONGPU_CPPCXTYPE_STDCOMPLEX // c++/hip + stdcomplex (this must be __CUDACC__ and not MGONGPUCPP_GPUIMPL)

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

#endif // #if not defined __CUDACC__ and defined MGONGPU_CPPCXTYPE_STDCOMPLEX

  //==========================================================================

  inline __host__ __device__ const cxtype
  cxmake( const cxsmpl<float>& c ) // cxsmpl to cxtype (float-to-float or float-to-double)
  {
    return cxmake( c.real(), c.imag() );
  }

  inline __host__ __device__ const cxtype
  cxmake( const cxsmpl<double>& c ) // cxsmpl to cxtype (double-to-float or double-to-double)
  {
    return cxmake( c.real(), c.imag() );
  }

} // end namespace mg5amcGpu/mg5amcCpu

//==========================================================================
// COMPLEX TYPES: WRAPPER OVER RI FLOATING POINT PAIR (cxtype_ref)
//==========================================================================

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
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

} // end namespace mg5amcGpu/mg5amcCpu

//==========================================================================

#endif // MGONGPUCXTYPES_H
