// main156.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Marius Utheim <marius.utheim@thep.lu.se>.

// Keywords: hadron widths;

// Code to create parameterization tables for hadron widths.
// Useful if resonances are added or particle properties are changed.

#include "Pythia8/Pythia.h"
using namespace Pythia8;

//--------------------------------------------------------------------------

int main(int argc, char* argv[]) {

  // Get precision from command line, if provided.
  int precision = (argc == 2) ? atoi(argv[1]) : 50;
  if ((argc != 1 && argc != 2) || precision <= 1) {
    cerr << " Unexpected number of command-line arguments. \n"
         << " You are expected to either provide the precision as an \n"
         << " integer (precision >= 2), or no argument to indicate the \n"
         << " default value (50). \n"
         << " Program stopped! " << endl;
    return 1;
  }

  // Initialize Pythia.
  Pythia pythia;
  pythia.readString("ProcessLevel:all = off");

  if (!pythia.init()) {
    cout << endl << " Pythia failed to initialize. \n"
     " If this happened because hadron widths are unavailable or invalid,\n"
     " particle data should still be loaded. In this case, this code should\n"
     " still be used to generate hadron widths. Therefore, execution will\n"
     " continue." << endl << endl;
  }

  // Perform parameterization.
  HadronWidths& hadronWidths = pythia.hadronWidths;
  hadronWidths.parameterizeAll(precision);
  hadronWidths.save("HadronWidths.dat");

  // Done.
  return 0;

}
