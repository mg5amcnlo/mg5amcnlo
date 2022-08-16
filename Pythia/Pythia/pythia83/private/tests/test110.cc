// test110.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This is a simple test program.
// It calculates low energy cross sections for many hadron-hadron combinations
// and all process types.

#include "Pythia8/Pythia.h"

using namespace Pythia8;

//==========================================================================

// Number of points to use for each process
static constexpr int nPoints = 10;

//--------------------------------------------------------------------------

// Class to keep track of the stream and to write test results 

class LowEnergyCrossSectionsTest {
public:

  LowEnergyCrossSectionsTest(Pythia& pythiaIn, ostream& streamIn)
    : pythia(pythiaIn), stream(streamIn) { }

  void test(int idA, int idB) {

    // Define energy range.
    double mA = pythia.particleData.m0(idA), mB = pythia.particleData.m0(idB);
    double eMin = mA + mB + 0.1;
    double eMax = eMin + 5.0;
    double de = (eMax - eMin) / (nPoints - 1);

    // Write header.
    stream << idA << " " << idB << endl;

    // Iterate through processes.
    for (int iType = 0; iType <= 9; ++iType) {
      stream << iType << ": ";
      for (double ie = 0; ie < nPoints; ++ie) {
        double eNow = eMin + ie * de;
        stream << pythia.getLowEnergySigma(idA, idB, eNow, mA, mB, iType);
        stream << " ";
      }
      stream << endl;
    }

    // Also write resonances, if available.
    for (int iRes : pythia.hadronWidths.possibleResonances(idA, idB)) {
      stream << iRes << ": ";
      for (double ie = 0; ie < nPoints; ++ie) {
        double eNow = eMin + ie * de;
        stream << pythia.getLowEnergySigma(idA, idB, eNow, mA, mB, iRes);
        stream << " ";
      }
      stream << endl;
    }

    stream << endl;

  }

private:

  Pythia& pythia;
  ostream& stream;

};

//--------------------------------------------------------------------------

int main() {

  // Setup pythia without any processes
  Pythia pythia;
  pythia.readString("Print:quiet = on");
  pythia.readString("ProcessLevel:all = off");
  if (!pythia.init()) {
    cout << "Failed to initialize Pythia." << endl;
    return -1;
  }

  ofstream stream("test110.dat");
  if (!stream.good()) {
    cout << "Failed to open output stream." << endl;
    return -2;
  }

  LowEnergyCrossSectionsTest tester(pythia, stream);

  // NN
  tester.test(2212, 2212);
  tester.test(2112, 2112);
  tester.test(2112, 2212);

  // SigmaN, LambdaN
  for (int idA : { 3222, 3212, 3112, 3122 })
  for (int idB : { 2212, 2112 })
    tester.test(idA, idB);
  
  // Other BB cases: Delta, Sigma, Xi, Omega, N*, Delta*
  for (int idA : { 2224, 2214, 3212, 3322, 3334, 202112, 212224 })
  for (int idB : { 2224, 2214, 3212, 3322, 3334, 202112, 212224 })
    tester.test(idA, idB);

  // NNbar
  tester.test(-2212, 2212);
  tester.test(-2212, 2112);

  // Other BBbar cases: Delta, Sigma, Lambda, Xi, N*, Delta*
  // Annihilation is impossible for some cases, e.g. Delta++ Sigma-
  for (int idA : { 2224, 2214, 3112, 3212, 3322, 3334, 202112, 212224 })
  for (int idB : { 2224, 2214, 3112, 3212, 3322, 3334, 202112, 212224 }) {
    // Test both orderings
    tester.test(idA, -idB);
    tester.test(-idA, idB);
  }
  
  // N/S/L + pi/K
  for (int idA : { 211, 111, -211,  321, 311, -321, -311 })
  for (int idB : { 2212, 2112,  3222, 3212, 3112, 3122 }) {
    tester.test(idA, idB);
    tester.test(idB, idA);
    tester.test(idA, -idB);
  }
  
  // Other BM cases
  for (int idA : { 221, 223, 225, 333, 213, 113, 331, 323, 313, -323, -313 })
  for (int idB : { 2212, 2112, 3222, 3212, 3112, 3122, 3312, 3322, 3334,
                   2224, 2214 }) {
    // Test both orderings
    tester.test(idA, idB);
    tester.test(idB, idA);
  }
  
  // MM cases
  for (int idA : { 211, 111, -211, 321, 311, -321, -311 })
  for (int idB : { 211, 111, -211, 321, 311, -321, -311 })
    tester.test(idA, idB);

}
