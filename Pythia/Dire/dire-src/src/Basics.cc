
#include "Dire/Basics.h"

namespace Pythia8 {

//--------------------------------------------------------------------------

// Helper function to calculate dilogarithm.

double polev(double x,double* coef,int N ) {
  double ans;
  int i;
  double *p;

  p = coef;
  ans = *p++;
  i = N;
    
  do
    ans = ans * x  +  *p++;
  while( --i );
    
  return ans;
}
  
//--------------------------------------------------------------------------

// Function to calculate dilogarithm.

double dilog(double x) {

  static double cof_A[8] = {
    4.65128586073990045278E-5,
    7.31589045238094711071E-3,
    1.33847639578309018650E-1,
    8.79691311754530315341E-1,
    2.71149851196553469920E0,
    4.25697156008121755724E0,
    3.29771340985225106936E0,
    1.00000000000000000126E0,
  };
  static double cof_B[8] = {
    6.90990488912553276999E-4,
    2.54043763932544379113E-2,
    2.82974860602568089943E-1,
    1.41172597751831069617E0,
    3.63800533345137075418E0,
    5.03278880143316990390E0,
    3.54771340985225096217E0,
    9.99999999999999998740E-1,
  };

  if( x >1. ) {
    return -dilog(1./x)+M_PI*M_PI/3.-0.5*pow2(log(x));
  }

  x = 1.-x;
  double w, y, z;
  int flag;
  if( x == 1.0 )
    return( 0.0 );
  if( x == 0.0 )
    return( M_PI*M_PI/6.0 );
    
  flag = 0;
    
  if( x > 2.0 ) {
    x = 1.0/x;
    flag |= 2;
  }
    
  if( x > 1.5 ) {
    w = (1.0/x) - 1.0;
    flag |= 2;
  }
    
  else if( x < 0.5 ) {
    w = -x;
    flag |= 1;
  }
    
  else
    w = x - 1.0;
    
  y = -w * polev( w, cof_A, 7) / polev( w, cof_B, 7 );
    
  if( flag & 1 )
    y = (M_PI * M_PI)/6.0  - log(x) * log(1.0-x) - y;
    
  if( flag & 2 ) {
    z = log(x);
    y = -0.5 * z * z  -  y;
  }
    
  return y;

}

}
