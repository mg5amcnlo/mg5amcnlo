// test030.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This is a simple test program.
// It outputs all particle data. This tests that the way Pythia reads data is
// unchanged between versions.

#include "Pythia8/Pythia.h"

using namespace Pythia8;

int main() {

  // Setup pythia without any processes
  Pythia pythia;
  pythia.readString("Print:quiet = on");
  pythia.readString("ProcessLevel:all = off");
  if (!pythia.init()) {
    cout << "Failed to initialize Pythia." << endl;
    return -1;
  }

  ofstream stream("test030.dat");
  if (!stream.good()) {
    cout << "Failed to open output stream." << endl;
    return -2;
  }
  
  pythia.particleData.listAll(stream);

  return 0;
}