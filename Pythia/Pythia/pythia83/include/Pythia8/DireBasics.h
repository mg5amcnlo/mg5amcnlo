// DireBasics.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for Dire basics.

#ifndef Pythia8_DireBasics_H
#define Pythia8_DireBasics_H

#define DIRE_BASICS_VERSION "2.002"

#include "Pythia8/Event.h"
#include <limits>
#include <unordered_map>

using std::unordered_map;

namespace Pythia8 {

bool checkSIJ(const Event& e, double minSIJ=0.0);
void printSI(const Event& e);
void printSIJ(const Event& e);

typedef unsigned long ulong;

//==========================================================================

// Function to hash string into long integer.

ulong shash(const string& str);

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

//==========================================================================

// Template to make initializing maps simpler, while not relying on C++11.
// Usage: map(createmap<T,U>(a,b)(c,d)(e,f));

template <typename T, typename U> class create_unordered_map {

private:

  unordered_map<T, U> um_map;

public:

  create_unordered_map(const T& key, const U& val) { um_map[key] = val; }
  create_unordered_map<T, U>& operator()(const T& key, const U& val) {
    um_map[key] = val;
    return *this;
  }
  operator unordered_map<T, U>() { return um_map; }

};

//==========================================================================

// Template to make initializing maps simpler, while not relying on C++11.
// Usage: map(createmap<T,U>(a,b)(c,d)(e,f));

template <typename T> class createvector {

private:

  vector<T> m_vector;

public:

  createvector(const T& val) { m_vector.push_back(val); }
  createvector<T>& operator()(const T& val) {
    m_vector.push_back(val);
    return *this;
  }
  operator vector<T>() { return m_vector; }

};

//==========================================================================

// Helper function to calculate dilogarithm.

double polev(double x,double* coef,int N );
// Function to calculate dilogarithm.
double dilog(double x);

//==========================================================================

// Kallen function and derived quantities.

double lABC(double a, double b, double c);
double bABC(double a, double b, double c);
double gABC(double a, double b, double c);

//==========================================================================

class DireFunction {

public:

  virtual double f(double) { return 0.; }
  virtual double f(double, double) { return 0.; }
  virtual double f(double, vector<double>) { return 0.; }

};

class DireRootFinder {

public:

  DireRootFinder() {};
  virtual ~DireRootFinder() {};

  double findRootSecant1D( DireFunction* f, double xmin, double xmax,
    double constant, vector<double> xb = vector<double>(), int N=10 ) {
    vector<double> x;
    x.push_back(xmin);
    x.push_back(xmax);
    for ( int i=2; i < N; ++i ) {
      double xn = x[i-1]
      - ( f->f(x[i-1],xb) - constant)
      * ( x[i-1] - x[i-2] )
      / ( f->f(x[i-1],xb) - f->f(x[i-2],xb) );
      x.push_back(xn);
    }
    return x.back();
  }

  double findRoot1D( DireFunction* f, double xmin, double xmax,
    double constant, vector<double> xx = vector<double>(), int N=10,
    double tol = 1e-10 ) {

    double a(xmin), b(xmax), c(xmax), d(0.), e(0.),
      fa(f->f(a,xx)-constant), fb(f->f(b,xx)-constant), fc(fb),
      p(0.), q(0.), r(0.), s(0.),
      tol1(tol), xm(0.);
    double EPS = numeric_limits<double>::epsilon();

    // No root.
    if ( (fa>0. && fb>0.) || (fa<0. && fb<0.) ) {
     cout << "no root " << constant << " " << f->f(a,xx) << " " << f->f(b,xx)
     << endl;
     return numeric_limits<double>::quiet_NaN();
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

      if (abs(e) >= tol1 && abs(fa) > abs(fb) ) {
        s = fb/fa;
        if ( a == c ) {
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
      }
      fb = f->f(b,xx)-constant;
    }

    // Failed. Return NaN
    return numeric_limits<double>::quiet_NaN();

  }

};

//==========================================================================

class DireEventInfo {

  public:

  DireEventInfo() {}

  // Bookkeeping of soft particles.
  int sizeSoftPos () const { return softPosSave.size(); }
  int getSoftPos(unsigned int i) const {
    return (i > softPosSave.size()-1) ? -1 : softPosSave[i]; }
  bool isSoft(int iPos) {
    vector<int>::iterator it = find( softPosSave.begin(),
      softPosSave.end(), iPos);
    return (it != softPosSave.end());
  }
  void addSoftPos(int iPos) { if (!isSoft(iPos)) softPosSave.push_back(iPos); }
  void removeSoftPos(int iPos) {
    vector<int>::iterator it = find( softPosSave.begin(),
      softPosSave.end(), iPos);
    if (it != softPosSave.end()) softPosSave.erase(it);
  }
  void updateSoftPos(int iPosOld, int iPosNew) {
    if (isSoft(iPosOld)) removeSoftPos(iPosOld);
    if (isSoft(iPosNew)) removeSoftPos(iPosNew);
    addSoftPos(iPosNew);
  }
  void updateSoftPosIfMatch(int iPosOld, int iPosNew) {
    if (isSoft(iPosOld)) {
      vector<int>::iterator it
        = find (softPosSave.begin(), softPosSave.end(), iPosOld);
      *it = iPosNew;
    }
  }
  vector<int> softPos () const { return softPosSave; }
  void clearSoftPos () { softPosSave.clear(); }
  void listSoft() const {
    cout << " 'Soft' particles: ";
    for (int i=0; i < sizeSoftPos(); ++i) cout << setw(5) << getSoftPos(i);
    cout << endl;
  }

  // Bookkeeping of resonances.
  void removeResPos(int iPos) {
    vector<int>::iterator it = find (iPosRes.begin(), iPosRes.end(), iPos);
    if (it == iPosRes.end()) return;
    iPosRes.erase(it);
    sort (iPosRes.begin(), iPosRes.end());
  }
  void addResPos(int iPos) {
    vector<int>::iterator it = find (iPosRes.begin(), iPosRes.end(), iPos);
    if (it != iPosRes.end()) return;
    iPosRes.push_back(iPos);
    sort (iPosRes.begin(), iPosRes.end());
  }
  void updateResPos(int iPosOld, int iPosNew) {
    vector<int>::iterator it = find (iPosRes.begin(), iPosRes.end(), iPosOld);
    if (it == iPosRes.end()) iPosRes.push_back(iPosNew);
    else                    *it = iPosNew;
    sort (iPosRes.begin(), iPosRes.end());
  }
  void updateResPosIfMatch(int iPosOld, int iPosNew) {
    vector<int>::iterator it = find (iPosRes.begin(), iPosRes.end(), iPosOld);
    if (it != iPosRes.end()) {
      iPosRes.erase(it);
      iPosRes.push_back(iPosNew);
      sort (iPosRes.begin(), iPosRes.end());
    }
  }
  bool isRes(int iPos) {
    vector<int>::iterator it = find (iPosRes.begin(), iPosRes.end(), iPos);
    return (it != iPosRes.end());
  }
  int sizeResPos() const { return iPosRes.size(); }
  int getResPos(unsigned int i) const {
    return (i > iPosRes.size()-1) ? -1 : iPosRes[i]; }
  void clearResPos() { iPosRes.resize(0); }
  void listRes() const {
    cout << " 'Resonant' particles: ";
    for (int i=0; i < sizeResPos(); ++i) cout << setw(5) <<  getResPos(i);
    cout << endl;
  }

  // Data members.
  vector<int> softPosSave;
  vector<int> iPosRes;

};


//==========================================================================

class DireDebugInfo {

  public:

  DireDebugInfo() = default;

  void clearMessages() {
    messageStream0.str("");
    messageStream1.str("");
    messageStream2.str("");
  }

  void printMessages( int verbosity = 0) {
    cout << "\n"
      << "*------------------------------------------------------------*\n"
      << "*----------------- Begin diagnostic output ------------------*\n\n";
    if (verbosity == 0) cout << scientific << setprecision(8)
    << messageStream0.str();
    if (verbosity == 1) cout << scientific << setprecision(8)
    << messageStream1.str();
    if (verbosity == 2) cout << scientific << setprecision(8)
    << messageStream2.str();
    cout << "\n\n"
      << "*----------------- End diagnostic output -------------------*\n"
      << "*-----------------------------------------------------------*"
      << endl;
  }

  void printHistograms() {}

  void fillHistograms(int type, int nfinal, double mec, double pT, double z) {
    if (false) cout << type*nfinal*mec*pT*z;}

  // Add debug messages to message stream.
  ostream & message ( int verbosity = 0) {
    if (verbosity == 0) return messageStream0;
    if (verbosity == 1) return messageStream1;
    if (verbosity == 2) return messageStream2;
    return messageStream0;
  }

  // Debug message streams.
  ostringstream messageStream0, messageStream1, messageStream2;

};

//==========================================================================

class DireInfo {

  public:

  DireInfo() {}

  void clearAll() {
    direEventInfo.clearResPos();
    direEventInfo.clearSoftPos();
    direDebugInfo.clearMessages();
  }

  // Resonance info forwards.
  void removeResPos(int iPos)   { return direEventInfo.removeResPos(iPos); }
  void addResPos(int iPos)      { return direEventInfo.addResPos(iPos); }
  bool isRes(int iPos)          { return direEventInfo.isRes(iPos); }
  void clearResPos ()           { return direEventInfo.clearResPos(); }
  int sizeResPos() const        { return direEventInfo.sizeResPos(); }
  void listRes() const          { return direEventInfo.listRes(); }
  int getResPos(unsigned int i) const { return direEventInfo.getResPos(i); }
  void updateResPos(int iPosOld, int iPosNew) {
    return direEventInfo.updateResPos(iPosOld,iPosNew); }
  void updateResPosIfMatch(int iPosOld, int iPosNew) {
    return direEventInfo.updateResPosIfMatch(iPosOld,iPosNew); }

  // Debug info forwards.
  void printMessages( int verbosity = 0) {
    return direDebugInfo.printMessages(verbosity); }
  ostream & message ( int verbosity = 0) {
    return direDebugInfo.message(verbosity); }
  void clearMessages() { direDebugInfo.clearMessages(); }

  void fillHistograms(int type, int nfinal, double mec, double pT, double z) {
    direDebugInfo.fillHistograms(type, nfinal,  mec, pT, z); }
  void printHistograms() { direDebugInfo.printHistograms(); }

  // Soft particle info forwards.
  bool isSoft(int iPos)          { return direEventInfo.isSoft(iPos); }
  void addSoftPos(int iPos)      { return direEventInfo.addSoftPos(iPos); }
  void removeSoftPos(int iPos)   { return direEventInfo.removeSoftPos(iPos); }
  vector<int> softPos ()         { return direEventInfo.softPos(); }
  void clearSoftPos ()           { return direEventInfo.clearSoftPos(); }
  int sizeSoftPos () const       { return direEventInfo.sizeSoftPos(); }
  void listSoft() const          { return direEventInfo.listSoft(); }
  int getSoftPos(unsigned int i) const { return direEventInfo.getSoftPos(i); }
  void updateSoftPos(int iPosOld, int iPosNew) {
    return direEventInfo.updateSoftPos(iPosOld, iPosNew);
  }
  void updateSoftPosIfMatch(int iPosOld, int iPosNew) {
    return direEventInfo.updateSoftPosIfMatch(iPosOld, iPosNew);
  }

  DireEventInfo direEventInfo;
  DireDebugInfo direDebugInfo;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_DireBasics_H
