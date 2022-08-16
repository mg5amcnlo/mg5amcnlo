// DireBasics.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for Dire basics.

#include "Pythia8/DireBasics.h"

namespace Pythia8 {

bool checkSIJ(const Event& e, double minSIJ) {
  double sijmin=1e10;
  for (int i=0; i < e.size(); ++i) {
    if (!e[i].isFinal() && e[i].mother1() !=1 && e[i].mother1() !=2) continue;
    for (int j=0; j < e.size(); ++j) {
      if (i==j) continue;
      if (!e[j].isFinal() && e[j].mother1() !=1 && e[j].mother1() !=2)
        continue;
      sijmin=min(sijmin,abs(2.*e[i].p()*e[j].p()));
    }
  }
  return (sijmin>minSIJ);
}

//--------------------------------------------------------------------------

void printSI(const Event& e) {
  for (int i=0; i < e.size(); ++i) {
    if (!e[i].isFinal() && e[i].mother1() !=1 && e[i].mother1() !=2) continue;
    cout << "  [" << e[i].isFinal()
         << " s("<< i << ")="
         << e[i].p().m2Calc() << "],\n";
  }
}

//--------------------------------------------------------------------------

void printSIJ(const Event& e) {
  for (int i=0; i < e.size(); ++i) {
    if (!e[i].isFinal() && e[i].mother1() !=1 && e[i].mother1() !=2) continue;
    for (int j=0; j < e.size(); ++j) {
      if (i==j) continue;
      if (!e[j].isFinal() && e[j].mother1() !=1 && e[j].mother1() !=2)
        continue;
      cout << "  [" << e[i].isFinal() << e[j].isFinal()
           << " s("<< i << "," << j << ")="
           << 2.*e[i].p()*e[j].p() << "],\n";
    }
  }
}

//--------------------------------------------------------------------------

// Function to hash string into long integer.

ulong shash(const std::string& str) {
    ulong hash = 5381;
    for (size_t i = 0; i < str.size(); ++i)
        hash = 33 * hash + (unsigned char)str[i];
    return hash;
}

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

double lABC(double a, double b, double c) { return pow2(a-b-c) - 4.*b*c;}
double bABC(double a, double b, double c) {
  double ret = 0.;
  if      ((a-b-c) > 0.) ret = sqrt(lABC(a,b,c));
  else if ((a-b-c) < 0.) ret =-sqrt(lABC(a,b,c));
  else                   ret = 0.;
  return ret; }
double gABC(double a, double b, double c) { return 0.5*(a-b-c+bABC(a,b,c));}

//==========================================================================

}  // end namespace Pythia8
