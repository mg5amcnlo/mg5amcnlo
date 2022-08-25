// MathTools.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for some mathematics tools, like special functions.
#ifndef Pythia8_MathTools_H
#define Pythia8_MathTools_H

// Header file for the MathTools methods.
#include "Pythia8/Basics.h"
#include "Pythia8/PythiaStdlib.h"

namespace Pythia8 {

//==========================================================================

// The Gamma function for real argument.
double gammaReal(double x);

// Modified Bessel functions of the first and second kinds.
double besselI0(double x);
double besselI1(double x);
double besselK0(double x);
double besselK1(double x);

// Integrate f(x) dx over the specified range.
bool integrateGauss(double& resultOut, function<double(double)> f,
  double xLo, double xHi, double tol=1e-6);

// Solve f(x) = target for x in the specified range.
bool brent(double& solutionOut, function<double(double)> f,
  double target, double xLo, double xHi, double tol=1e-6, int maxIter = 10000);

// Gram determinant, invariants used in the argument = 2*pi*pj.
double gramDet(double s01tilde, double s12tilde, double s02tilde,
  double m0, double m1, double m2);
double gramDet(Vec4 p0, Vec4 p1, Vec4 p2);

// Dilogarithm.
double Li2 (const double, const double kmax = 100.0, const double xerr = 1e-9);

// Standard factorial.
double factorial(const int);

// Binomial coefficient.
int binomial (const int,int);

// Lambert W function.
double lambertW (const double x);

// Kallen function.
double kallenFunction(const double x, const double y, const double z);

//==========================================================================

// LinearInterpolator class.
// Used to interpolate between values in linearly spaced data.

class LinearInterpolator {

public:

  LinearInterpolator() = default;

  // Constructor.
  LinearInterpolator(double leftIn, double rightIn, vector<double> ysIn)
    : leftSave(leftIn), rightSave(rightIn), ysSave(ysIn) { }

  // Function to get y-values of interpolation data.
  const vector<double>& data() const { return ysSave; }

  // x-values are linearly spaced on the interpolation region.
  double left()  const { return leftSave; }
  double right() const { return rightSave; }
  double dx()    const { return (rightSave - leftSave) / (ysSave.size() - 1); }

  // Operator to get interpolated value at the specified point.
  double operator()(double x) const;

  // Plot the data points of this LinearInterpolator in a histogram.
  Hist plot(string title) const;
  Hist plot(string title, double xMin, double xMax) const;

private:

  // Data members
  double leftSave, rightSave;
  vector<double> ysSave;

};

//==========================================================================

// Class for the "Hungarian" pairing algorithm. Adapted for Vincia
// from an implementation by M. Buehren and C. Ma, see notices below.

// This is a C++ wrapper with slight modification of a hungarian algorithm
// implementation by Markus Buehren. The original implementation is a few
// mex-functions for use in MATLAB, found here:
// http://www.mathworks.com/matlabcentral/fileexchange/
//    6543-functions-for-the-rectangular-assignment-problem
//
// Both this code and the orignal code are published under the BSD license.
// by Cong Ma, 2016.
//
// Copyright (c) 2014, Markus Buehren
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in
// the documentation and/or other materials provided with the distribution
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

class HungarianAlgorithm {

public:

  // Function wrapper for solving assignment problem.
  double solve(vector <vector<double> >& distMatrix, vector<int>& assignment);

 private:

  // Solve optimal solution for assignment problem using Munkres algorithm,
  // also known as the Hungarian algorithm.
  void optimal(vector<int>& assignment, double& cost,
    vector<double>& distMatrix, int nOfRows, int nOfColumns);
  // Build the assignment vector.
  void vect(vector<int>& assignment, vector<bool>& starMatrix, int nOfRows,
    int nOfColumns);
  // Calculate the assignment cost.
  void calcCost(vector<int>& assignment, double& cost,
    vector<double>& distMatrix, int nOfRows);
  // Factorized step 2a of the algorithm.
  void step2a(vector<int>& assignment, vector<double>& distMatrix,
    vector<bool>& starMatrix, vector<bool>& newStarMatrix,
    vector<bool>& primeMatrix, vector<bool>& coveredColumns,
    vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);
  // Factorized step 2b of the algorithm.
  void step2b(vector<int>& assignment, vector<double>& distMatrix,
    vector<bool>& starMatrix, vector<bool>& newStarMatrix,
    vector<bool>& primeMatrix, vector<bool>& coveredColumns,
    vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);
  // Factorized step 3 of the algorithm.
  void step3(vector<int>& assignment, vector<double>& distMatrix,
    vector<bool>& starMatrix, vector<bool>& newStarMatrix,
    vector<bool>& primeMatrix, vector<bool>& coveredColumns,
    vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);
  // Factorized step 4 of the algorithm.
  void step4(vector<int>& assignment, vector<double>& distMatrix,
    vector<bool>& starMatrix, vector<bool>& newStarMatrix,
    vector<bool>& primeMatrix, vector<bool>& coveredColumns,
    vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim,
    int row, int col);
  // Factorized step 5 of the algorithm.
  void step5(vector<int>& assignment, vector<double>& distMatrix,
    vector<bool>& starMatrix, vector<bool>& newStarMatrix,
    vector<bool>& primeMatrix, vector<bool>& coveredColumns,
    vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_MathTools_H
