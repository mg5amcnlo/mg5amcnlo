
#ifndef Pythia8_DireBasics_H
#define Pythia8_DireBasics_H

#define DIRE_BASICS_VERSION "2.000"

// Pythia includes.
#include "Pythia8/Pythia.h"
#include <limits>

namespace Pythia8 {

//==========================================================================

// Template to make initializing maps simpler, while not relying on C++11.
// Usage: map(createmap<T,U>(a,b)(c,d)(e,f));

template <typename T, typename U> class createmap {

private:

  map<T, U> m_map;

public:

  createmap(const T& key, const U& val) { m_map[key] = val; }
  createmap<T, U>& operator()(const T& key, const U& val) {
    m_map[key] = val;
    return *this;
  }
  operator map<T, U>() { return m_map; }

};

//--------------------------------------------------------------------------

// Helper function to calculate dilogarithm.

double polev(double x,double* coef,int N );
// Function to calculate dilogarithm.
double dilog(double x);

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

class Function {

public:

  Function() {};
  virtual ~Function() {};

  virtual double f(double, vector<double> = vector<double>()) { return 0.; }

  double findRootSecant1D( double xmin, double xmax, double constant,
    vector<double> xb = vector<double>(), int N=10 ) {
    vector<double> x;
    x.push_back(xmin);
    x.push_back(xmax);
    for ( int i=2; i < N; ++i ) {
      double xn = x[i-1]
      - ( f(x[i-1],xb) - constant)
      * ( x[i-1] - x[i-2] )
      / ( f(x[i-1],xb) - f(x[i-2],xb) );
      x.push_back(xn);
    }
    return x.back();
  }

  double findRoot1D( double xmin, double xmax, double constant,
    vector<double> xx = vector<double>(), int N=10, double tol = 1e-10 ) {

    double a(xmin), b(xmax), c(xmax), d(0.), e(0.),
      fa(f(a,xx)-constant), fb(f(b,xx)-constant), fc(fb),
      p(0.), q(0.), r(0.), s(0.),
      tol1(tol), xm(0.);
    double EPS = std::numeric_limits<double>::epsilon();
    //double EPS = 1e-20;

    // No root.
    if ( (fa>0. && fb>0.) || (fa<0. && fb<0.) ) {
     cout << "no root " << constant << " " << f(a,xx) << " " << f(b,xx) << endl; abort();
     return std::numeric_limits<double>::quiet_NaN();
    }

    for ( int i=0; i < N; ++i ) {

      if ( (fb>0. && fc>0.) || (fb<0. && fc<0.) ) {
        c  = a;
        fc = fa;
        e  = d = b-a;
      }

      if ( abs(fc) < abs(fb) ) {
        a = b;
        b = c;
        c = a;
        fa = fb;
        fb = fc;
        fc = fa;
      }

      tol1 = 2.*EPS*abs(b) + 0.5*tol;
      xm = 0.5*(c-b);

      if (abs(xm) <= tol1 || fb == 0.) return b;
      //if (abs(xm) <= tol1 || fb < EPS) return b;
      //if (abs(xm) <= tol1 || abs(fb) < EPS) return b;

      if (abs(e) >= tol1 && abs(fa) > abs(fb) ) {
        s = fb/fa;
        if ( a == c ) {
        //if ( abs(abs(a)-abs(c)) < EPS ) {
          p = 2.*xm*s;
          q = 1.-s;
        } else {
          q = fa/fc;
          r = fb/fc;
          p = s*(2.*xm*q*(q-r) - (b-a)*(r-1.));
          q = (q-1.)*(r-1.)*(s-1.);
        }
        if (p>0.) q = -q;
        p = abs(p);
        double min1 = 3.*xm*q - abs(tol1*q);
        double min2 = abs(e*q);
        if (2.*p < ((min1 < min2) ? min1 : min2)) {
          e = d;
          d = p/q;
        } else {
          d = xm;
          e = d;
        }

      } else {
        d = xm;
        e = d;
      }

      a = b;
      fa = fb;

      if (abs(d) > tol1) { b += d; }
      else {
        b += (xm> 0.) ? tol1 : -tol1;
      //  fb = f(b,xx)-constant;
      }
      fb = f(b,xx)-constant;
    }

    // Failed. Return NaN
    return std::numeric_limits<double>::quiet_NaN();

  }

};

//==========================================================================

} // end namespace Pythia8

#endif
