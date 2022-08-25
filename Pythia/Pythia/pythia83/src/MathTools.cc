// MathTools.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for some mathematical tools.

#include "Pythia8/MathTools.h"

namespace Pythia8 {

//==========================================================================

// The Gamma function for real arguments, using the Lanczos approximation.
// Code based on http://en.wikipedia.org/wiki/Lanczos_approximation

double GammaCoef[9] = {
     0.99999999999980993,     676.5203681218851,   -1259.1392167224028,
      771.32342877765313,   -176.61502916214059,    12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};

double gammaReal(double x) {

  // Reflection formula (recursive!) for x < 0.5.
  if (x < 0.5) return M_PI / (sin(M_PI * x) * gammaReal(1 - x));

  // Iterate through terms.
  double z = x - 1.;
  double gamma = GammaCoef[0];
  for (int i = 1; i < 9; ++i) gamma += GammaCoef[i] / (z + i);

  // Answer.
  double t = z + 7.5;
  gamma *= sqrt(2. * M_PI) * pow(t, z + 0.5) * exp(-t);
  return gamma;

}

//--------------------------------------------------------------------------

// Polynomial approximation for modified Bessel function of the first kind.
// Based on Abramowitz & Stegun, Handbook of mathematical functions (1964).

double besselI0(double x){

  // Parametrize in terms of t.
  double result = 0.;
  double t  = x / 3.75;
  double t2 = pow2(t);

  // Only positive values relevant.
  if      ( t < 0.) ;
  else if ( t < 1.) {
    result = 1.0 + 3.5156229 * t2 + 3.0899424 * pow2(t2)
           + 1.2067492 * pow3(t2) + 0.2659732 * pow4(t2)
           + 0.0360768 * pow5(t2) + 0.0045813 * pow6(t2);
  } else {
    double u = 1. / t;
    result = exp(x) / sqrt(x) * ( 0.39894228 + 0.01328592 * u
           + 0.00225319 * pow2(u) - 0.00157565 * pow3(u)
           + 0.00916281 * pow4(u) - 0.02057706 * pow5(u)
           + 0.02635537 * pow6(u) - 0.01647633 * pow7(u)
           + 0.00392377 * pow8(u) );
  }

  return result;
}

//--------------------------------------------------------------------------

// Polynomial approximation for modified Bessel function of the first kind.
// Based on Abramowitz & Stegun, Handbook of mathematical functions (1964).

double besselI1(double x){

  // Parametrize in terms of t.
  double result = 0.;
  double t = x / 3.75;
  double t2 = pow2(t);

  // Only positive values relevant.
  if      ( t < 0.) ;
  else if ( t < 1.) {
    result = x * ( 0.5 + 0.87890594 * t2 + 0.51498869 * pow2(t2)
           + 0.15084934 * pow3(t2) + 0.02658733 * pow4(t2)
           + 0.00301532 * pow5(t2) + 0.00032411 * pow6(t2) );
  } else {
    double u = 1. / t;
    result = exp(x) / sqrt(x) * ( 0.39894228 - 0.03988024 * u
           - 0.00368018 * pow2(u) + 0.00163801 * pow3(u)
           - 0.01031555 * pow4(u) + 0.02282967 * pow5(u)
           - 0.02895312 * pow6(u) + 0.01787654 * pow7(u)
           - 0.00420059 * pow8(u) );
  }

  return result;
}

//--------------------------------------------------------------------------

// Polynomial approximation for modified Bessel function of a second kind.
// Based on Abramowitz & Stegun, Handbook of mathematical functions (1964).

double besselK0(double x){

  double result = 0.;

  // Polynomial approximation valid ony for x > 0.
  if      ( x < 0.) ;
  else if ( x < 2.) {
    double x2 = pow2(0.5 * x);
    result = -log(0.5 * x) * besselI0(x) - 0.57721566
           + 0.42278420 * x2 + 0.23069756 * pow2(x2)
           + 0.03488590 * pow3(x2) + 0.00262698 * pow4(x2)
           + 0.00010750 * pow5(x2) + 0.00000740 * pow6(x2);
  } else {
    double z = 2. / x;
    result = exp(-x) / sqrt(x) * ( 1.25331414 - 0.07832358 * z
           + 0.02189568 * pow2(z) - 0.01062446 * pow3(z)
           + 0.00587872 * pow4(z) - 0.00251540 * pow5(z)
           + 0.00053208 * pow6(z) );
  }

  return result;
}

//--------------------------------------------------------------------------

// Polynomial approximation for modified Bessel function of a second kind.
// Based on Abramowitz & Stegun, Handbook of mathematical functions (1964).

double besselK1(double x){

  double result = 0.;

  // Polynomial approximation valid ony for x > 0.
  if      ( x < 0.) ;
  else if ( x < 2.) {
    double x2 = pow2(0.5 * x);
    result = log(0.5 * x) * besselI1(x) + 1./x * ( 1. + 0.15443144 * x2
           - 0.67278579 * pow2(x2) - 0.18156897 * pow3(x2)
           - 0.01919402 * pow4(x2) - 0.00110404 * pow5(x2)
           - 0.00004686 * pow6(x2) );
  } else {
    double z = 2. / x;
    result = exp(-x) / sqrt(x) * ( 1.25331414 + 0.23498619 * z
           - 0.03655620 * pow2(z) + 0.01504268 * pow3(z)
           - 0.00780353 * pow4(z) + 0.00325614 * pow5(z)
           - 0.00068245 * pow6(z) );
  }

  return result;
}

//==========================================================================

// Integrate f over the specified range.
// Note that f must be a function of one variable. In order to integrate one
// variable of function with multiple arguments, like integrating f(x, y) with
// respect to x when y is fixed, the other variables can be captured using a
// lambda function.

bool integrateGauss(double& resultOut, function<double(double)> f,
  double xLo, double xHi, double tol) {

  // Boundary check: return zero if xLo >= xHi.
  if (xLo >= xHi) {
    resultOut = 0.0;
    return true;
  }

  // Initialize temporary result.
  double result = 0.0;

  // 8-point unweighted.
  static double x8[4]={  0.96028985649753623, 0.79666647741362674,
    0.52553240991632899, 0.18343464249564980};
  static double w8[4]={  0.10122853629037626, 0.22238103445337447,
    0.31370664587788729, 0.36268378337836198};
  // 16-point unweighted.
  static double x16[8]={ 0.98940093499164993, 0.94457502307323258,
    0.86563120238783174, 0.75540440835500303, 0.61787624440264375,
    0.45801677765722739, 0.28160355077925891, 0.09501250983763744};
  static double w16[8]={  0.027152459411754095, 0.062253523938647893,
    0.095158511682492785, 0.12462897125553387, 0.14959598881657673,
    0.16915651939500254,  0.18260341504492359, 0.18945061045506850};

  // Set up integration region.
  double c = 0.001/abs(xHi-xLo);
  double zLo = xLo;
  double zHi = xHi;

  bool nextbin = true;
  while ( nextbin ) {

    double zMid = 0.5*(zHi+zLo); // midpoint
    double zDel = 0.5*(zHi-zLo); // midpoint, relative to zLo

    // Calculate 8-point and 16-point quadratures.
    double s8=0.0;
    for (int i=0;i<4;i++) {
      double dz = zDel * x8[i];
      double f1 = f(zMid+dz);
      double f2 = f(zMid-dz);
      s8 += w8[i]*(f1 + f2);
    }
    s8 *= zDel;
    double s16=0.0;
    for (int i=0;i<8;i++) {
      double dz = zDel * x16[i];
      double f1 = f(zMid+dz);
      double f2 = f(zMid-dz);
      s16 += w16[i]*(f1 + f2);
    }
    s16 *= zDel;

    // Precision in this bin OK, add to cumulative and go to next.
    if (abs(s16-s8) < tol*(1+abs(s16))) {
      nextbin=true;
      result += s16;
      // Next bin: LO = end of current, HI = end of integration region.
      zLo=zHi;
      zHi=xHi;
      if ( zLo == zHi ) nextbin = false;

    // Precision in this bin not OK, subdivide.
    } else {
      if (1.0 + c*abs(zDel) == 1.0) {
        // Cannot subdivide further at double precision. Fail.
        result = 0.0 ;
        return false;
      }
      zHi = zMid;
      nextbin = true;
    }
  }

  // Write result and return success.
  resultOut = result;
  return true;
}

//==========================================================================

// Solve f(x) = target for x in the specified range. Note that f must
// be a function of one variable. In order to solve an equation with a
// multivariable function, like solving f(x, y) = target for x when y
// is fixed, the other variables can be captured using a lambda
// function.

bool brent(double& solutionOut, function<double(double)> f,  double target,
  double xLo, double xHi, double tol, int maxIter) {

  // Range checks.
  if (xLo > xHi) return false;

  // Evaluate function - targetValue at lower boundary.
  double f1 = f(xLo) - target;
  if (abs(f1) < tol) {
    solutionOut = xLo;
    return true;
  }
  // Evaluate function - targetValue at upper boundary.
  double f2 = f(xHi) - target;
  if (abs(f2) < tol) {
    solutionOut = xHi;
    return true;
  }

  // Check if root is bracketed.
  if ( f1 * f2 > 0.0) return false;

  // Start searching for root.
  double x1 = xLo;
  double x2 = xHi;
  double x3 = 0.5 * (xLo + xHi);

  int iter=0;
  while(++iter < maxIter) {
    // Now check at x = x3.
    double f3 = f(x3) - target;
    // Check if tolerance on f has been reached.
    if (abs(f3) < tol) {
      solutionOut = x3;
      return true;
    }
    // Is root bracketed in lower or upper half?
    if (f1 * f3 < 0.0) xHi = x3;
    else xLo = x3;
    // Check if tolerance on x has been reached.
    if ((xHi - xLo) < tol * (abs(xHi) < 1.0 ? xHi : 1.0)) {
      solutionOut = 0.5 * (xLo + xHi);
      return true;
    }

    // Work out next step to take in x.
    double den = (f2 - f1) * (f3 - f1) * (f2 - f3);
    double num = x3 * (f1 - f2) * (f2 - f3 + f1) + f2 * x1 * (f2 - f3)
               + f1 * x2 * (f3 - f1);
    double dx = xHi - xLo;
    if (den != 0.0) dx = f3 * num / den;

    // First attempt, using gradient
    double x = x3 + dx;

    // If this was too far, just step to the middle
    if ((xHi - x) * (x - xLo) < 0.0) {
      dx = 0.5 * (xHi - xLo);
      x = xLo + dx;
    }
    if (x < x3) {
        x2 = x3;
        f2 = f3;
    }
    else {
        x1 = x3;
        f1 = f3;
    }
    x3 = x;
  }

  // Maximum number of iterations exceeded.
  return false;
}

//==========================================================================

// Gram determinant, invariants used in the argument = 2*pi*pj.

double gramDet( double s01tilde, double s12tilde, double s02tilde,
  double m0, double m1, double m2) {
  return ((s01tilde*s12tilde*s02tilde - pow2(s01tilde)*pow2(m2)
           - pow2(s02tilde)*pow2(m1) - pow2(s12tilde)*pow2(m0))/4
          + pow2(m0)*pow2(m1)*pow2(m2));
}

double gramDet(Vec4 p0, Vec4 p1, Vec4 p2) {
  return gramDet(2*p0*p1, 2*p1*p2, 2*p0*p2, p0.mCalc(), p1.mCalc(),
    p2.mCalc());
}

//==========================================================================

// Dilogarithm.

double Li2(const double x, const double kmax, const double xerr) {
  if (x < 0.0) return 0.5*Li2(x*x) - Li2(-x);
  if (x <= 0.5) {
    double sum(x), term(x);
    for (int k = 2; k < kmax; k++) {
      double rk = (k-1.0)/k;
      term *= x*rk*rk;
      sum += term;
      if (abs(term/sum) < xerr) return sum;
    }
    cout << "Maximum number of iterations exceeded in Li2" << endl;
    return sum;
  }
  if (x < 1.0)  return M_PI*M_PI/6.0 - Li2(1.0 - x) - log(x)*log(1.0 - x);
  if (x == 1.0) return M_PI*M_PI/6.0;
  if (x <= 1.01) {
    const double eps(x - 1.0), lne(log(eps)),
      c0(M_PI*M_PI/6.0),         c1(  1.0 - lne),
      c2(-(1.0 - 2.0*lne)/4.0),  c3( (1.0 - 3.0*lne)/9.0),
      c4(-(1.0 - 4.0*lne)/16.0), c5( (1.0 - 5.0*lne)/25.0),
      c6(-(1.0 - 6.0*lne)/36.0), c7( (1.0 - 7.0*lne)/49.0),
      c8(-(1.0 - 8.0*lne)/64.0);
    return c0 + eps*(c1 + eps*(c2 + eps*(c3 + eps*(c4 + eps*(c5 + eps*(
                     c6 + eps*(c7 + eps*c8)))))));
  }
  double logx = log(x);
  if (x<=2.0) return M_PI*M_PI/6.0 + Li2(1.0 - 1.0/x) -
                logx*(log(1.0 - 1.0/x) + 0.5*logx);
  return M_PI*M_PI/3.0 - Li2(1.0/x) - 0.5*logx*logx;
}


//==========================================================================

// Standard factorial.

double factorial(const int n) {
  double fac = 1;
  for (int i = 2; i <= n; i++) fac *= i;
  return fac;
}

//==========================================================================

// Binomial coefficient.

int binomial(const int n, const int m) {
  if (m < 0 || m > n) return 0;
  else if (m == n || m == 0) return 1;
  else if (m == 1 || m == n - 1) return n;
  else return factorial(n)/factorial(m)/factorial(n - m) + 0.01;
}

//==========================================================================

// Lambert W function using the rational fit from Darko Veberic's
// paper, arXiv:1209.0735v2.  Should give 5 digits of precision for
// positive arguments x not too large (fit region was 0.3, 2e, but
// still has 5-digit accuracy at zero).  Precision quickly drops for
// negative values, but he has extra functions that can be implemented
// if those are needed, and for very large values the asymptotic
// log(x), log(log(x)) form could be used if precise solutions for
// large values are needed. For now just write a warning if we are
// ever asked for a value far outside region of validity.

double lambertW(const double x) {
  if (x == 0.) return 0.;
  if (x < -0.2) {
    cout << "Warning in lambertW"
         << ": Accuracy less than three decimal places for x < -0.2";
  } else if (x > 10.) {
    cout << "Warning in lambertW"
         <<": Accuracy less than three decimal places for x > 10.";
  }
  return x*(1. + x*(2.445053 + x*(1.343664 + x*(0.14844 + 0.000804*x))))
    /(1. + x*(3.444708 + x*(3.292489 + x*(0.916460 + x*(0.053068)))));
}

//==========================================================================

// The Kallen function.

double kallenFunction(const double x, const double y, const double z) {
  return x*x + y*y + z*z - 2.*(x*y + x*z + y*z);
}

//==========================================================================

// LinearInterpolator class.
// Used to interpolate between values in linearly spaced data.

//--------------------------------------------------------------------------

// Operator to get interpolated value at the specified point.

double LinearInterpolator::operator()(double xIn) const {

  if (xIn == rightSave)
    return ysSave.back();

  // Select interpolation bin.
  double t = (xIn - leftSave) / (rightSave - leftSave);
  int lastIdx = ysSave.size() - 1;
  int j = (int)floor(t * lastIdx);

  // Return zero outside of interpolation range.
  if (j < 0 || j >= lastIdx) return 0.;
  // Select position in bin and return linear interpolation.
  else {
    double s = (xIn - (leftSave + j * dx())) / dx();
    return (1 - s) * ysSave[j] + s * ysSave[j + 1];
  }
}

//--------------------------------------------------------------------------

// Plot the data points of this LinearInterpolator in a histogram.

Hist LinearInterpolator::plot(string title) const {
  return plot(title, leftSave, rightSave);
}

Hist LinearInterpolator::plot(string title, double xMin, double xMax) const {

  int nBins = ceil((xMax - xMin) / (rightSave - leftSave) * ysSave.size());

  Hist result(title, nBins, xMin, xMax, false);
  double dxNow = (xMax - xMin) / nBins;

  for (int i = 0; i < nBins; ++i) {
    double x = xMin + dxNow * (0.5 + i);
    result.fill(x, operator()(x));
  }

  return result;
}

//==========================================================================

// Class for the "Hungarian" pairing algorithm.

//--------------------------------------------------------------------------

// A single function wrapper for solving assignment problem.

double HungarianAlgorithm::solve(vector <vector<double> >& distMatrix,
  vector<int>& assignment) {

  int nRows = distMatrix.size();
  int nCols = distMatrix[0].size();
  vector<double> distMatrixIn(nRows * nCols);
  vector<int> solution(nRows);
  double cost = 0.0;

  // Fill in the distMatrixIn. Mind the index is "i + nRows * j".
  // Here the cost matrix of size MxN is defined as a double precision
  // array of N*M elements. In the solving functions matrices are seen
  // to be saved MATLAB-internally in row-order. (i.e. the matrix [1
  // 2; 3 4] will be stored as a vector [1 3 2 4], NOT [1 2 3 4]).
  for (int row = 0; row < nRows; row++)
    for (int col = 0; col < nCols; col++)
      distMatrixIn[row + nRows * col] = distMatrix[row][col];

  // Call solving function.
  optimal(solution, cost, distMatrixIn, nRows, nCols);
  assignment.clear();
  for (int r = 0; r < nRows; r++) assignment.push_back(solution[r]);
  return cost;

}

//--------------------------------------------------------------------------

// Solve optimal solution for assignment.

void HungarianAlgorithm::optimal(vector<int> &assignment, double& cost,
  vector<double>& distMatrixIn, int nOfRows, int nOfColumns) {

  // Initialization.
  int nOfElements(nOfRows * nOfColumns), minDim(0);
  vector<double> distMatrix(nOfElements);
  vector<bool> coveredColumns(nOfColumns), coveredRows(nOfRows),
    starMatrix(nOfElements), newStarMatrix(nOfElements),
    primeMatrix(nOfElements);
  cost = 0;
  for (int row = 0; row<nOfRows; row++) assignment[row] = -1;

  // Generate working copy of distance matrix. Check if all matrix
  // elements are positive.
  for (int row = 0; row < nOfElements; row++) {
    double value(distMatrixIn[row]);
    if (value < 0) std::cerr << "HungarianAlgorithm::assigmentoptimal(): All"
                             << " matrix elements have to be non-negative.\n";
    distMatrix[row] = value;
  }

  // Preliminary steps.
  if (nOfRows <= nOfColumns) {
    minDim = nOfRows;
    for (int row = 0; row < nOfRows; row++) {
      // Find the smallest element in the row.
      int idx(row);
      double minValue(distMatrix[idx]);
      idx += nOfRows;
      while (idx < nOfElements) {
        double value(distMatrix[idx]);
        if (value < minValue) minValue = value;
        idx += nOfRows;
      }

      // Subtract the smallest element from each element of the row.
      idx = row;
      while (idx < nOfElements) {
        distMatrix[idx] -= minValue;
        idx += nOfRows;
      }
    }

    // Steps 1 and 2a.
    for (int row = 0; row < nOfRows; row++)
      for (int col = 0; col < nOfColumns; col++)
        if (abs(distMatrix[row + nOfRows*col]) <
          numeric_limits<double>::epsilon())
          if (!coveredColumns[col]) {
            starMatrix[row + nOfRows*col] = true;
            coveredColumns[col] = true;
            break;
          }
  } else {
    minDim = nOfColumns;
    for (int col = 0; col < nOfColumns; col++) {
      // Find the smallest element in the column.
      int idx(nOfRows*col);
      int columnEnd(idx + nOfRows);
      double minValue = distMatrix[idx++];
      while (idx < columnEnd) {
        double value(distMatrix[idx++]);
        if (value < minValue) minValue = value;
      }

      // Subtract the smallest element from each element of the column.
      idx = nOfRows*col;
      while (idx < columnEnd) distMatrix[idx++] -= minValue;
    }

    // Steps 1 and 2a.
    for (int col = 0; col < nOfColumns; col++)
      for (int row = 0; row < nOfRows; row++)
        if (abs(distMatrix[row + nOfRows*col]) <
          numeric_limits<double>::epsilon())
          if (!coveredRows[row]) {
            starMatrix[row + nOfRows*col] = true;
            coveredColumns[col] = true;
            coveredRows[row] = true;
            break;
          }
    for (int row = 0; row < nOfRows; row++) coveredRows[row] = false;
  }

  // Move to step 2b.
  step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
    coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

  // Compute cost and remove invalid assignments.
  calcCost(assignment, cost, distMatrixIn, nOfRows);
  return;

}

//--------------------------------------------------------------------------

// Build the assignment vector.

void HungarianAlgorithm::vect(vector<int>& assignment,
  vector<bool>& starMatrix, int nOfRows, int nOfColumns) {
  for (int row = 0; row < nOfRows; row++)
    for (int col = 0; col < nOfColumns; col++)
      if (starMatrix[row + nOfRows*col]) {assignment[row] = col; break;}

}

//--------------------------------------------------------------------------

// Calculate the assignment cost.

void HungarianAlgorithm::calcCost(vector<int>& assignment, double& cost,
  vector<double>& distMatrix, int nOfRows) {
  for (int row = 0; row < nOfRows; row++) {
    int col(assignment[row]);
    if (col >= 0) cost += distMatrix[row + nOfRows*col];
  }

}

//--------------------------------------------------------------------------

// Factorized step 2a of the algorithm.

void HungarianAlgorithm::step2a(vector<int>& assignment,
  vector<double>& distMatrix, vector<bool>& starMatrix,
  vector<bool>& newStarMatrix, vector<bool>& primeMatrix,
  vector<bool>& coveredColumns, vector<bool>& coveredRows,
  int nOfRows, int nOfColumns, int minDim) {

  // Cover every column containing a starred zero.
  for (int col = 0; col < nOfColumns; col++) {
    int idx(nOfRows*col);
    int columnEnd(idx + nOfRows);
    while (idx < columnEnd) {
      if (starMatrix[idx++]) {coveredColumns[col] = true; break;}
    }
  }

  // Move to step 2b (note, original comment changed by Skands).
  step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
    coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

}

//--------------------------------------------------------------------------

// Factorized step 2b of the algorithm.

void HungarianAlgorithm::step2b(
  vector<int>& assignment, vector<double>& distMatrix,
  vector<bool>& starMatrix, vector<bool>& newStarMatrix,
  vector<bool>& primeMatrix, vector<bool>& coveredColumns,
  vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim) {

  // Count covered columns.
  int nOfCoveredColumns(0);
  for (int col = 0; col < nOfColumns; col++)
    if (coveredColumns[col]) nOfCoveredColumns++;

  // Algorithm finished.
  if (nOfCoveredColumns == minDim)
    vect(assignment, starMatrix, nOfRows, nOfColumns);
  // Move to step 3.
  else step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
    coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

}

//--------------------------------------------------------------------------

// Factorized step 3 of the algorithm.

void HungarianAlgorithm::step3(
  vector<int>& assignment, vector<double>& distMatrix,
  vector<bool>& starMatrix, vector<bool>& newStarMatrix,
  vector<bool>& primeMatrix, vector<bool>& coveredColumns,
  vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim) {

  bool zerosFound(true);
  while (zerosFound) {
    zerosFound = false;
    for (int col = 0; col < nOfColumns; col++)
      if (!coveredColumns[col])
        for (int row = 0; row < nOfRows; row++)
          if ((!coveredRows[row]) &&
            (abs(distMatrix[row + nOfRows*col]) <
              numeric_limits<double>::epsilon())) {
            // Prime zero.
            primeMatrix[row + nOfRows*col] = true;

            // Find starred zero in current row.
            int starCol(0);
            for (; starCol < nOfColumns; starCol++)
              if (starMatrix[row + nOfRows*starCol]) break;

            // No starred zero found, move to step 4.
            if (starCol == nOfColumns) {
              step4(assignment, distMatrix, starMatrix, newStarMatrix,
                primeMatrix, coveredColumns, coveredRows, nOfRows,
                nOfColumns, minDim, row, col);
              return;
            } else {
              coveredRows[row] = true;
              coveredColumns[starCol] = false;
              zerosFound = true;
              break;
            }
          }
  }

  // Move to step 5.
  step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
    coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

}

//--------------------------------------------------------------------------

// Factorized step 4 of the algorithm.

void HungarianAlgorithm::step4(
  vector<int>& assignment, vector<double>& distMatrix,
  vector<bool>& starMatrix, vector<bool>& newStarMatrix,
  vector<bool>& primeMatrix, vector<bool>& coveredColumns,
  vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim,
  int row, int col) {

  // Generate temporary copy of starMatrix.
  int nOfElements(nOfRows*nOfColumns);
  for (int n = 0; n < nOfElements; n++) newStarMatrix[n] = starMatrix[n];
  // Star current zero.
  newStarMatrix[row + nOfRows*col] = true;
  // Find starred zero in current column.
  int starCol(col), starRow(0);
  for (; starRow < nOfRows; starRow++)
    if (starMatrix[starRow + nOfRows*starCol]) break;
  while (starRow < nOfRows) {
    // Unstar the starred zero.
    newStarMatrix[starRow + nOfRows*starCol] = false;
    // Find primed zero in current row.
    int primeRow(starRow), primeCol(0);
    for (; primeCol < nOfColumns; primeCol++)
      if (primeMatrix[primeRow + nOfRows*primeCol]) break;
      // Star the primed zero.
    newStarMatrix[primeRow + nOfRows*primeCol] = true;
    // Find starred zero in current column.
    starCol = primeCol;
    for (starRow = 0; starRow < nOfRows; starRow++)
      if (starMatrix[starRow + nOfRows*starCol]) break;
  }

  // Use temporary copy as new starMatrix, delete all primes, uncover
  // all rows.
  for (int n = 0; n < nOfElements; n++) {
    primeMatrix[n] = false;
    starMatrix[n] = newStarMatrix[n];
  }
  for (int n = 0; n < nOfRows; n++) coveredRows[n] = false;

  // Move to step 2a.
  step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
    coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

}

//--------------------------------------------------------------------------

// Factorized step 5 of the algorithm.

void HungarianAlgorithm::step5(
  vector<int>& assignment, vector<double>& distMatrix,
  vector<bool>& starMatrix, vector<bool>& newStarMatrix,
  vector<bool>& primeMatrix, vector<bool>& coveredColumns,
  vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim) {

  // Find smallest uncovered element h.
  double h(numeric_limits<double>::max());
  for (int row = 0; row < nOfRows; row++)
    if (!coveredRows[row])
      for (int col = 0; col < nOfColumns; col++)
        if (!coveredColumns[col]) {
          double value(distMatrix[row + nOfRows*col]);
          if (value < h) h = value;
        }

  // Add h to each covered row.
  for (int row = 0; row < nOfRows; row++)
    if (coveredRows[row])
      for (int col = 0; col < nOfColumns; col++)
        distMatrix[row + nOfRows*col] += h;

  // Subtract h from each uncovered column.
  for (int col = 0; col < nOfColumns; col++)
    if (!coveredColumns[col])
      for (int row = 0; row < nOfRows; row++)
        distMatrix[row + nOfRows*col] -= h;

  // Move to step 3.
  step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
    coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

}

//==========================================================================

} // end namespace Pythia8
