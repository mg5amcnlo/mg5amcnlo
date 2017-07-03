// main01.cc is a part of the PYTHIA event generator.
// Copyright (C) 2016 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This is a simple test program. It fits on one slide in a talk.
// It studies the charged multiplicity distribution at the LHC.

#include "Pythia8/Pythia.h"
using namespace Pythia8;
int main() {
  // Generator. Process selection. LHC initialization. Histogram.
  Pythia pythia;
  //pythia.readString("Beams:eCM = 8000.");
  //pythia.readString("HardQCD:all = on");
  //pythia.readString("PhaseSpace:pTHatMin = 20.");

  pythia.readString("Beams:idA = 11");                   
  pythia.readString("Beams:idB = -11");                 
  pythia.readString("Beams:eA = 9.");                 
  pythia.readString("Beams:eB = 3.1");
  pythia.readString("Beams:frameType = 2");
  pythia.readString("HiddenValley:ffbar2Zv = on");
  pythia.readString("HiddenValley:fragment = on");
  pythia.readString("HiddenValley:doKinMix = on");
  pythia.readString("4900023:onMode = off");
  pythia.readString("4900023:onIfAny = 4900101 11");
  pythia.readString("4900023:mWidth = 0.0001");
  pythia.readString("4900023:m0 = 5.0");  //! dark photon?
  pythia.readString("4900101:m0 = 0.1");  //! dark quark
  pythia.readString("4900111:m0 = 0.2");  //! flavor diag pi d
  pythia.readString("4900113:m0 = 0.2");  //! flavor diag rho d
  pythia.readString("4900211:m0 = 0.2");  //! flavor off diag pi d
  pythia.readString("4900213:m0 = 0.2");  //! flavor off diag rho d
  pythia.readString("4900023:mMin = 0.0");  //! flavor off diag rho d
  pythia.readString("4900101:onMode = off");  //! dark quark
  pythia.readString("4900101:mayDecay = off");  //! dark quark
  pythia.readString("4900101:isResonance = off");  //! dark quark

  pythia.init();
  Hist mult("charged multiplicity", 100, -0.5, 799.5);
  // Begin event loop. Generate event. Skip if error. List first one.
  for (int iEvent = 0; iEvent < 100; ++iEvent) {
    if (!pythia.next()) continue;
    // Find number of all final charged particles and fill histogram.
    int nCharged = 0;
    for (int i = 0; i < pythia.event.size(); ++i)
      if (pythia.event[i].isFinal() && pythia.event[i].isCharged())
        ++nCharged;
    mult.fill( nCharged );
  // End of event loop. Statistics. Histogram. Done.
  }
  pythia.stat();
  cout << mult;
  return 0;
}
