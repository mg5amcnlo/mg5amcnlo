// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: F.Stloukal (May 2026) for the MadGraph7 MadMatrix.
// Further modified by: F.Stloukal  


#ifndef TEST_FD_FPEXPANSION_H
#define TEST_FD_FPEXPANSION_H
#include "errorFreeOPs.h"

#ifdef __CADNA__
#include <cadna.h>
#endif

namespace MG_ARITHM{

template< typename T >
class alignas( 2 * sizeof( T ) ) Double
{

  static_assert( GoodFloatType< T >
                 ,"Double<T> can only be instantiated with float or double (float_st or double_st under CADNA)." );

  private:
  T data[ 2 ];
  public:
  using BaseType = T;

  __cuda_callable__
  constexpr Double( T hi, T lo );

  __cuda_callable__
  constexpr Double() = default;

  __cuda_callable__
  constexpr Double( const Double& other ) = default;

  __cuda_callable__
  constexpr Double( Double&& other ) noexcept = default;

  template<SingleFloatType U>
  __cuda_callable__
  constexpr Double(U rhs);

  template<DoubleFloatType U>
  __cuda_callable__
  constexpr Double(U rhs);

#ifdef __CADNA__
   __cuda_callable__
   explicit constexpr
   operator float_st() const;
#else
   __cuda_callable__
   explicit constexpr
   operator float() const;
#endif

   //--------------ADDS AND SUBS-------------
   __cuda_callable__
   constexpr static Double< T >
   add( T a, T b );

   __cuda_callable__
   constexpr static Double< T >
   ieee_add( const Double< T >& a, const Double< T >& b );

   __cuda_callable__
   constexpr static Double< T >
   sloppy_add( const Double< T >& a, const Double< T >& b );

   //UNARY MINUS
   __cuda_callable__
   constexpr Double< T >
   operator+() const;

   //UNARY MINUS
   __cuda_callable__
   constexpr Double< T >
   operator-() const;

  // Element access
  __cuda_callable__
  constexpr T operator[]( int i ) const { return data[ i ]; }

  __cuda_callable__
  constexpr T& operator[]( int i ) { return data[ i ]; }

};
  // --------------- ADDITION --------------
  template< typename T >
  __cuda_callable__
  constexpr Double< T >
  operator+( const Double< T >& A, const Double< T >& B );

  // --------------- SUBTRACTION -------------
  template< typename T >
  __cuda_callable__
  constexpr Double< T >
  operator-( const Double< T >& A, const Double< T >& B );

  // ------------- MULTIPLICATION ------------

  template< typename T >
  __cuda_callable__
  constexpr Double< T >
  operator*( const Double< T >& A, const Double< T >& B );

  //----------------------------------------
  //-------------- DEFINITIONS -------------
  //----------------------------------------

  template< typename T >
  __cuda_callable__
  constexpr Double< T >::Double( T hi, T lo )
     : data{hi, lo}
  {}

  template< typename T >
  template< SingleFloatType U >
  __cuda_callable__
  constexpr Double< T >::Double( const U rhs )
  {
#ifdef __CADNA__
    if constexpr( std::is_same_v< T, float_st > ) {
#else
    if constexpr( std::is_same_v< T, float > ) {
#endif
      data[ 0 ] = rhs;
      data[ 1 ] = 0.0F;
    }
    else {
      data[ 0 ] = static_cast< T >( rhs );
      data[ 1 ] = 0.0;
    }
  }

template< typename T >
template< DoubleFloatType U >
__cuda_callable__
constexpr Double< T >::Double( const U rhs )
{
#ifdef __CADNA__
    if constexpr( std::is_same_v< T, float_st > ) {
#else
    if constexpr( std::is_same_v< T, float > ) {
#endif
      const auto sp = split< double, float >( rhs );
      data[ 0 ] = sp.sum;
      data[ 1 ] = sp.error;
   }
   else {
      data[ 0 ] = rhs;
      data[ 1 ] = 0.0;
   }
}

#ifdef __CADNA__
template< typename T >
__cuda_callable__
constexpr Double< T >::operator float_st() const
  {
   if constexpr( std::is_same_v< T, float_st > )
      return data[0];
   else
      return static_cast<float_st>(data[ 0 ]);
  }
#else

   template< typename T >
   __cuda_callable__
   constexpr Double< T >::operator float() const
  {
     return static_cast<float>(data[ 0 ]);
  }

#endif

 template< typename T >
__cuda_callable__
constexpr Double< T >
Double< T >::add( T a, T b )
{
   auto tsRes = two_sum( a, b );
   return Double< T >( tsRes.sum, tsRes.error );
}
template< typename T >
__cuda_callable__
constexpr Double< T >
Double< T >::ieee_add( const Double< T >& a, const Double< T >& b )
{
   auto tsRes = two_sum( a[ 0 ], b[ 0 ] );
   auto tsRes2 = two_sum( a[ 1 ], b[ 1 ] );
   auto qtsRes = quick_two_sum( tsRes.sum, add_rn( tsRes.error, tsRes2.sum ) );
   auto qtsRes2 = quick_two_sum( qtsRes.sum, add_rn( qtsRes.error, tsRes2.error ) );
   return Double< T >( qtsRes2.sum, qtsRes2.error );
}

template< typename T >
__cuda_callable__
constexpr Double< T >
Double< T >::sloppy_add( const Double< T >& a, const Double< T >& b )
{
   auto tsRes = two_sum( a[ 0 ], b[ 0 ] );
   const T temp = add_rn( a[ 1 ], b[ 1 ] );
   auto qtsRes = quick_two_sum( tsRes.sum, add_rn( tsRes.error, temp ) );
   return Double< T >( qtsRes.sum, qtsRes.error );
}

   template< typename T >
   __cuda_callable__
   constexpr Double< T >
   operator+( const Double< T >& A, const Double< T >& B )
  {
#ifdef ARITHMETICS_SLOPPY_ADD
     return Double< T >::sloppy_add( A, B );
#else
     return Double< T >::ieee_add( A, B );
#endif
  }

   template< typename T >
   __cuda_callable__
   constexpr Double< T >
   Double< T >::operator+() const
  {
     return *this;
  }

   template< typename T >
   __cuda_callable__
   constexpr Double< T >
   operator-( const Double< T >& A, const Double< T >& B )
  {
#ifdef ARITHMETICS_SLOPY_ADD
     auto td = two_diff( A[ 0 ], B[ 0 ] );
     td.error = add_rn( td.error, A[ 1 ] );
     auto qtsRes = quick_two_sum( td.sum, add_rn( td.error, -B[ 1 ] ) );
     return Double< T >( qtsRes.sum, qtsRes.error );
#else
     auto td1 = two_diff( A[ 0 ], B[ 0 ] );
     auto td2 = two_diff( A[ 1 ], B[ 1 ] );
     auto qtsRes = quick_two_sum( td1.sum, add_rn( td1.error, td2.sum ) );
     auto qtsRes2 = quick_two_sum( qtsRes.sum, add_rn( qtsRes.error, td2.error ) );
     return Double< T >( qtsRes2.sum, qtsRes2.error );
#endif
  }

   template< typename T >
   __cuda_callable__
   constexpr Double< T >
   Double< T >::operator-() const
  {
     return Double< T >( -this->data[ 0 ], -this->data[ 1 ] );
  }

   template< typename T >
   __cuda_callable__
   constexpr Double< T >
   operator*( const Double< T >& A, const Double< T >& B )
  {
     auto tp = two_prod( A[ 0 ], B[ 0 ] );
     /*volatile*/ const T temp = add_rn( mul_rn( A[ 1 ], B[ 0 ] ), mul_rn( A[ 0 ], B[ 1 ] ) );
     auto qtsRes = quick_two_sum( tp.sum, add_rn( tp.error, temp ) );
     return Double< T >( qtsRes.sum, qtsRes.error );
  }

}

#endif //TEST_FD_FPEXPANSION_H
