// Basics.cc is a part of the PYTHIA event generator.

// Access time information.
#include <ctime>
// Stdlib header files for mathematics.
#include <cmath>
#include <cstdlib>
#include <algorithm>

// Stdlib header files for strings and containers.
//#include <string>
//#include <vector>
//#include <map>
//#include <deque>
//#include <set>

// Stdlib header file for input and output.
//#include <iostream>
//#include <iomanip>
//#include <fstream>
//#include <sstream>

// Define pi if not yet done.
#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif

//==========================================================================

// Forward reference to RotBstMatrix class; needed in Vec4 class.
class RotBstMatrix;

//--------------------------------------------------------------------------

// Vec4 class.
// This class implements four-vectors, in energy-momentum space.
// (But can equally well be used to hold space-time four-vectors.)

class Vec4 {

public:

  // Constructors.
  Vec4(double xIn = 0., double yIn = 0., double zIn = 0., double tIn = 0.)
    : xx(xIn), yy(yIn), zz(zIn), tt(tIn) { }
  Vec4(const Vec4& v) : xx(v.xx), yy(v.yy), zz(v.zz), tt(v.tt) { }
  Vec4& operator=(const Vec4& v) { if (this != &v) { xx = v.xx; yy = v.yy; 
    zz = v.zz; tt = v.tt; } return *this; }
  Vec4& operator=(double value) { xx = value; yy = value; zz = value; 
    tt = value; return *this; }
      
  // Member functions for input.
  void reset() {xx = 0.; yy = 0.; zz = 0.; tt = 0.;}
  void p(double xIn, double yIn, double zIn, double tIn) 
    {xx = xIn; yy = yIn; zz = zIn; tt = tIn;}
  void p(Vec4 pIn) {xx = pIn.xx; yy = pIn.yy; zz = pIn.zz; tt = pIn.tt;} 
  void px(double xIn) {xx = xIn;}
  void py(double yIn) {yy = yIn;}
  void pz(double zIn) {zz = zIn;}
  void e(double tIn) {tt = tIn;}

  // Member functions for output.
  double px() const {return xx;}
  double py() const {return yy;}
  double pz() const {return zz;}
  double e() const {return tt;}
  double mCalc() const {double temp = tt*tt - xx*xx - yy*yy - zz*zz;
    return (temp >= 0.) ? sqrt(temp) : -sqrt(-temp);}
  double m2Calc() const {return tt*tt - xx*xx - yy*yy - zz*zz;}
  double pT() const {return sqrt(xx*xx + yy*yy);}
  double pT2() const {return xx*xx + yy*yy;}
  double pAbs() const {return sqrt(xx*xx + yy*yy + zz*zz);}
  double pAbs2() const {return xx*xx + yy*yy + zz*zz;}
  double theta() const {return atan2(sqrt(xx*xx + yy*yy), zz);}
  double phi() const {return atan2(yy,xx);}
  double rap() const {return 0.5 * log( (tt + zz) / (tt - zz) );}
  double eta() const {double xyz = sqrt(xx*xx + yy*yy + zz*zz);
    return 0.5 * log( (xyz + zz) / (xyz - zz) );}

  // Member functions that perform operations.
  void rot(double thetaIn, double phiIn); 
  void bst(double betaX, double betaY, double betaZ); 
  void bst(double betaX, double betaY, double betaZ, double gamma); 
  void bst(const Vec4& pIn); 
  void bst(const Vec4& pIn, double mIn); 
  void bstback(const Vec4& pIn); 
  void bstback(const Vec4& pIn, double mIn); 
  void rotbst(const RotBstMatrix& M); 

  // Operator overloading with member functions
  Vec4 operator-() {Vec4 tmp; tmp.xx = -xx; tmp.yy = -yy; tmp.zz = -zz; 
    tmp.tt = -tt; return tmp;}
  Vec4& operator+=(const Vec4& v) {xx += v.xx; yy += v.yy; zz += v.zz; 
    tt += v.tt; return *this;}
  Vec4& operator-=(const Vec4& v) {xx -= v.xx; yy -= v.yy; zz -= v.zz; 
    tt -= v.tt; return *this;}
  Vec4& operator*=(double f) {xx *= f; yy *= f; zz *= f; 
    tt *= f; return *this;}
  Vec4& operator/=(double f) {xx /= f; yy /= f; zz /= f; 
    tt /= f; return *this;}

  // Operator overloading with friends
  friend Vec4 operator+(const Vec4& v1, const Vec4& v2);
  friend Vec4 operator-(const Vec4& v1, const Vec4& v2);
  friend Vec4 operator*(double f, const Vec4& v1);
  friend Vec4 operator*(const Vec4& v1, double f);
  friend Vec4 operator/(const Vec4& v1, double f);
  friend double operator*(const Vec4& v1, const Vec4& v2);

  // Invariant mass of a pair and its square.
  friend double m(const Vec4& v1, const Vec4& v2);
  friend double m2(const Vec4& v1, const Vec4& v2);

private:

  // Constants: could only be changed in the code itself.
  static const double TINY;

  // The four-vector data members.
  double xx, yy, zz, tt;

};

//--------------------------------------------------------------------------

// Namespace function declarations; friends of Vec4 class.

// Implementation of operator overloading with friends.

inline Vec4 operator+(const Vec4& v1, const Vec4& v2) 
  {Vec4 v = v1 ; return v += v2;}

inline Vec4 operator-(const Vec4& v1, const Vec4& v2) 
  {Vec4 v = v1 ; return v -= v2;}

inline Vec4 operator*(double f, const Vec4& v1) 
  {Vec4 v = v1; return v *= f;}

inline Vec4 operator*(const Vec4& v1, double f) 
  {Vec4 v = v1; return v *= f;}

inline Vec4 operator/(const Vec4& v1, double f) 
  {Vec4 v = v1; return v /= f;}

inline double operator*(const Vec4& v1, const Vec4& v2)
  {return v1.tt*v2.tt - v1.xx*v2.xx - v1.yy*v2.yy - v1.zz*v2.zz;}  

// Invariant mass of a pair and its square.
double m(const Vec4& v1, const Vec4& v2);
double m2(const Vec4& v1, const Vec4& v2);

//==========================================================================

// RotBstMatrix class.
// This class implements 4 * 4 matrices that encode an arbitrary combination
// of rotations and boosts, that can be applied to Vec4 four-vectors.

class RotBstMatrix {

public:

  // Constructors.
  RotBstMatrix() {for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) 
    { M[i][j] = (i==j) ? 1. : 0.; } } } 
  RotBstMatrix(const RotBstMatrix& Min) {
    for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) {
    M[i][j] = Min.M[i][j]; } } }
  RotBstMatrix& operator=(const RotBstMatrix& Min) {if (this != &Min) {
    for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) {
    M[i][j] = Min.M[i][j]; } } } return *this; }

  // Member functions.
  void rot(double = 0., double = 0.);
  void rot(const Vec4& p);
  void bst(double = 0., double = 0., double = 0.);
  void bst(const Vec4&);
  void bstback(const Vec4&);
  void bst(const Vec4&, const Vec4&);
  void toCMframe(const Vec4&, const Vec4&);
  void fromCMframe(const Vec4&, const Vec4&);
  void rotbst(const RotBstMatrix&);

  // Private members to be accessible from Vec4. 
  friend class Vec4;

private:

  // Constants: could only be changed in the code itself.
  static const double TINY;

  // The rotation-and-boost matrix data members.
  double M[4][4];

};

// Small number to avoid division by zero.
const double Vec4::TINY = 1e-20;

//--------------------------------------------------------------------------

// Boost (simple).

void Vec4::bst(double betaX, double betaY, double betaZ) {

  double beta2 = betaX*betaX + betaY*betaY + betaZ*betaZ;
  double gamma = 1. / sqrt(1. - beta2);
  double prod1 = betaX * xx + betaY * yy + betaZ * zz;
  double prod2 = gamma * (gamma * prod1 / (1. + gamma) + tt);
  xx += prod2 * betaX;
  yy += prod2 * betaY;
  zz += prod2 * betaZ;
  tt = gamma * (tt + prod1);

}

//--------------------------------------------------------------------------

// Boost (simple, given gamma).

void Vec4::bst(double betaX, double betaY, double betaZ, double gamma) {

  double prod1 = betaX * xx + betaY * yy + betaZ * zz;
  double prod2 = gamma * (gamma * prod1 / (1. + gamma) + tt);
  xx += prod2 * betaX;
  yy += prod2 * betaY;
  zz += prod2 * betaZ;
  tt = gamma * (tt + prod1);

}

//--------------------------------------------------------------------------

// Boost given by a Vec4 p.

void Vec4::bst(const Vec4& pIn) {

  double betaX = pIn.xx / pIn.tt;
  double betaY = pIn.yy / pIn.tt;
  double betaZ = pIn.zz / pIn.tt;
  double beta2 = betaX*betaX + betaY*betaY + betaZ*betaZ;
  double gamma = 1. / sqrt(1. - beta2);
  double prod1 = betaX * xx + betaY * yy + betaZ * zz;
  double prod2 = gamma * (gamma * prod1 / (1. + gamma) + tt);
  xx          += prod2 * betaX;
  yy          += prod2 * betaY;
  zz          += prod2 * betaZ;
  tt           = gamma * (tt + prod1);

}

//--------------------------------------------------------------------------

// Boost given by a Vec4 p and double m.

void Vec4::bst(const Vec4& pIn, double mIn) {

  double betaX = pIn.xx / pIn.tt;
  double betaY = pIn.yy / pIn.tt;
  double betaZ = pIn.zz / pIn.tt;
  double gamma = pIn.tt / mIn;
  double prod1 = betaX * xx + betaY * yy + betaZ * zz;
  double prod2 = gamma * (gamma * prod1 / (1. + gamma) + tt);
  xx          += prod2 * betaX;
  yy          += prod2 * betaY;
  zz          += prod2 * betaZ;
  tt           = gamma * (tt + prod1);

}

//--------------------------------------------------------------------------

// Boost given by a Vec4 p; boost in opposite direction.

void Vec4::bstback(const Vec4& pIn) {

  double betaX = -pIn.xx / pIn.tt;
  double betaY = -pIn.yy / pIn.tt;
  double betaZ = -pIn.zz / pIn.tt;
  double beta2 = betaX*betaX + betaY*betaY + betaZ*betaZ;
  double gamma = 1. / sqrt(1. - beta2);
  double prod1 = betaX * xx + betaY * yy + betaZ * zz;
  double prod2 = gamma * (gamma * prod1 / (1. + gamma) + tt);
  xx          += prod2 * betaX;
  yy          += prod2 * betaY;
  zz          += prod2 * betaZ;
  tt           = gamma * (tt + prod1);

}

//--------------------------------------------------------------------------

// Boost given by a Vec4 p and double m; boost in opposite direction.

void Vec4::bstback(const Vec4& pIn, double mIn) {

  double betaX = -pIn.xx / pIn.tt;
  double betaY = -pIn.yy / pIn.tt;
  double betaZ = -pIn.zz / pIn.tt;
  double gamma = pIn.tt / mIn;
  double prod1 = betaX * xx + betaY * yy + betaZ * zz;
  double prod2 = gamma * (gamma * prod1 / (1. + gamma) + tt);
  xx          += prod2 * betaX;
  yy          += prod2 * betaY;
  zz          += prod2 * betaZ;
  tt           = gamma * (tt + prod1);

}

//--------------------------------------------------------------------------

// Arbitrary combination of rotations and boosts defined by 4 * 4 matrix.

void Vec4::rotbst(const RotBstMatrix& M) {

  double x = xx; double y = yy; double z = zz; double t = tt; 
  tt = M.M[0][0] * t + M.M[0][1] * x + M.M[0][2] * y +  M.M[0][3] * z;
  xx = M.M[1][0] * t + M.M[1][1] * x + M.M[1][2] * y +  M.M[1][3] * z;
  yy = M.M[2][0] * t + M.M[2][1] * x + M.M[2][2] * y +  M.M[2][3] * z;
  zz = M.M[3][0] * t + M.M[3][1] * x + M.M[3][2] * y +  M.M[3][3] * z;

} 

//--------------------------------------------------------------------------

// The invariant mass of two four-vectors.

double m(const Vec4& v1, const Vec4& v2) {
  double m2 = pow(v1.tt + v2.tt,2) - pow(v1.xx + v2.xx,2)
     - pow(v1.yy + v2.yy,2) - pow(v1.zz + v2.zz,2);
  return (m2 > 0.) ? sqrt(m2) : 0.; 
}

//--------------------------------------------------------------------------

// The squared invariant mass of two four-vectors.

double m2(const Vec4& v1, const Vec4& v2) {
  double m2 = pow(v1.tt + v2.tt,2) - pow(v1.xx + v2.xx,2)
     - pow(v1.yy + v2.yy,2) - pow(v1.zz + v2.zz,2);
  return m2; 
}

//--------------------------------------------------------------------------

//==========================================================================

// RotBstMatrix class.
// This class implements 4 * 4 matrices that encode an arbitrary combination
// of rotations and boosts, that can be applied to Vec4 four-vectors.

//--------------------------------------------------------------------------

// Constants: could be changed here if desired, but normally should not.
// These are of technical nature, as described for each.

// Small number to avoid division by zero.
const double RotBstMatrix::TINY = 1e-20;

//--------------------------------------------------------------------------

// Rotate by polar angle theta and azimuthal angle phi.

void RotBstMatrix::rot(double theta, double phi) {

  // Set up rotation matrix.
  double cthe = cos(theta); 
  double sthe = sin(theta);
  double cphi = cos(phi); 
  double sphi = sin(phi);
  double Mrot[4][4] = { 
    {1.,           0.,         0.,          0.}, 
    {0.,  cthe * cphi,     - sphi, sthe * cphi},
    {0.,  cthe * sphi,       cphi, sthe * sphi},
    {0., -sthe,                0., cthe       } };

  // Rotate current matrix accordingly.
  double Mtmp[4][4];
  for (int i = 0; i < 4; ++i)  
  for (int j = 0; j < 4; ++j) 
    Mtmp[i][j] = M[i][j];  
  for (int i = 0; i < 4; ++i) 
  for (int j = 0; j < 4; ++j) 
    M[i][j] = Mrot[i][0] * Mtmp[0][j] + Mrot[i][1] * Mtmp[1][j]
            + Mrot[i][2] * Mtmp[2][j] + Mrot[i][3] * Mtmp[3][j]; 

}

//--------------------------------------------------------------------------

// Rotate so that vector originally along z axis becomes parallel with p.

void RotBstMatrix::rot(const Vec4& p) {

  double theta = p.theta();
  double phi = p.phi();
  rot(0., -phi);
  rot(theta, phi);

}

//--------------------------------------------------------------------------

// Boost with velocity vector (betaX, betaY, betaZ).

void RotBstMatrix::bst(double betaX, double betaY, double betaZ) {

  // Set up boost matrix.  
  double gm = 1. / sqrt( max( TINY, 1. - betaX*betaX - betaY*betaY 
    - betaZ*betaZ ) );
  double gf = gm*gm / (1. + gm);
  double Mbst[4][4] = { 
    { gm,           gm*betaX,           gm*betaY,          gm*betaZ },
    { gm*betaX, 1. + gf*betaX*betaX, gf*betaX*betaY, gf*betaX*betaZ },
    { gm*betaY, gf*betaY*betaX, 1. + gf*betaY*betaY, gf*betaY*betaZ },
    { gm*betaZ, gf*betaZ*betaX, gf*betaZ*betaY, 1. + gf*betaZ*betaZ } };

  // Boost current matrix correspondingly.
  double Mtmp[4][4];
  for (int i = 0; i < 4; ++i)  
  for (int j = 0; j < 4; ++j) 
    Mtmp[i][j] = M[i][j]; 
  for (int i = 0; i < 4; ++i) 
  for (int j = 0; j < 4; ++j) 
    M[i][j] = Mbst[i][0] * Mtmp[0][j] + Mbst[i][1] * Mtmp[1][j]
            + Mbst[i][2] * Mtmp[2][j] + Mbst[i][3] * Mtmp[3][j];

}

//--------------------------------------------------------------------------

// Boost so that vector originally at rest obtains same velocity as p.

void RotBstMatrix::bst(const Vec4& p) {
  double betaX = p.px() / p.e();  
  double betaY = p.py() / p.e();  
  double betaZ = p.pz() / p.e();  
  bst(betaX, betaY, betaZ);
}

//--------------------------------------------------------------------------

// Boost so vector originally with same velocity as p is brought to rest.

void RotBstMatrix::bstback(const Vec4& p) {
  double betaX = -p.px() / p.e();  
  double betaY = -p.py() / p.e();  
  double betaZ = -p.pz() / p.e();  
  bst(betaX, betaY, betaZ);
}

//--------------------------------------------------------------------------

// Boost that transforms p1 to p2, where p1^2 = p2^2 is assumed.

void RotBstMatrix::bst(const Vec4& p1, const Vec4& p2) {
  double eSum = p1.e() + p2.e();
  double betaX = (p2.px() - p1.px()) / eSum;
  double betaY = (p2.py() - p1.py()) / eSum;
  double betaZ = (p2.pz() - p1.pz()) / eSum;
  double fac = 2. / (1. + betaX*betaX + betaY*betaY + betaZ*betaZ);
  betaX *= fac; betaY *= fac; betaZ *= fac;
  bst(betaX, betaY, betaZ);
}

//--------------------------------------------------------------------------

// Boost and rotation that transforms from p1 and p2 
// to their rest frame with p1 along +z axis.

void RotBstMatrix::toCMframe(const Vec4& p1, const Vec4& p2) {
  Vec4 pSum = p1 + p2; 
  Vec4 dir  = p1;
  dir.bstback(pSum);
  double theta = dir.theta();
  double phi   = dir.phi();
  bstback(pSum);
  rot(0., -phi);
  rot(-theta, phi);
}

//--------------------------------------------------------------------------

// Rotation and boost that transforms from rest frame of p1 and p2
// with p1 along +z axis to actual frame of p1 and p2. (Inverse of above.)

void RotBstMatrix::fromCMframe(const Vec4& p1, const Vec4& p2) {
  Vec4 pSum = p1 + p2;
  Vec4 dir  = p1;
  dir.bstback(pSum);
  double theta = dir.theta();
  double phi   = dir.phi();
  rot(0., -phi);
  rot(theta, phi);
  bst(pSum);
}

//--------------------------------------------------------------------------

// Combine existing rotation/boost matrix with another one.

void RotBstMatrix::rotbst(const RotBstMatrix& Mrb) {
  double Mtmp[4][4];
  for (int i = 0; i < 4; ++i) 
  for (int j = 0; j < 4; ++j) 
    Mtmp[i][j] = M[i][j]; 
  for (int i = 0; i < 4; ++i)  
  for (int j = 0; j < 4; ++j) 
    M[i][j] = Mrb.M[i][0] * Mtmp[0][j] + Mrb.M[i][1] * Mtmp[1][j]
            + Mrb.M[i][2] * Mtmp[2][j] + Mrb.M[i][3] * Mtmp[3][j]; 
}

//==========================================================================
