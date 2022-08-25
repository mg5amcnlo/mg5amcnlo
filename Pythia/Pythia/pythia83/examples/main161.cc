// main161.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Marius Utheim <marius.utheim@thep.lu.se>.

// Keywords: parallelism;

// This is a simple test program to illustrate the usage of PythiaParallel.
// The program is equivalent to main01, but in parallel.
// This program is shortened to fit on a single slide. See main162 for a
// more heavily commented version.

#include "Pythia8/Pythia.h"
#include "Pythia8/PythiaParallel.h"
using namespace Pythia8;
int main() {
  // Use the PythiaParallel class for parallel generation.
  PythiaParallel pythia;
  pythia.readString("Beams:eCM = 8000.");
  pythia.readString("HardQCD:all = on");
  pythia.readString("PhaseSpace:pTHatMin = 20.");
  pythia.init();
  Hist mult("charged multiplicity", 100, -0.5, 799.5);
  // Use PythiaParallel::run to generate the specified number of events.
  pythia.run(10000, [&](Pythia& pythiaNow) {
    // Find number of all final charged particles and fill histogram.
    int nCharged = 0;
    for (int i = 0; i < pythiaNow.event.size(); ++i)
      if (pythiaNow.event[i].isFinal() && pythiaNow.event[i].isCharged())
        ++nCharged;
    mult.fill( nCharged );
    // End of event loop. Statistics. Histogram. Done.
  });
  pythia.stat();
  cout << mult;
  return 0;
}
