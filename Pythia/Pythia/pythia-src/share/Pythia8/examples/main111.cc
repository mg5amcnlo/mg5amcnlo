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
  pythia.readString("Beams:frameType = 4");
  //pythia.readString("Beams:LHEF = wbj_lhef3.lhe");

  //pythia.readString("LesHouches:setLeptonMass   = 0");
  //pythia.readString("LesHouches:setQuarkMass    = 0");
  //pythia.readString("LesHouches:mRecalculate    = -1.0");
  pythia.readString("LesHouches:matchInOut      = off");
  //pythia.readString("Check:epTolErr = 1e-2");

  ifstream ifs;
  ifs.open("wbj_lhef3.lhe");
  istream* is = (istream*) &ifs;

  ifstream ifs2;
  ifs2.open("wbj_lhef3.lhe");
  istream* isHead = (istream*) &ifs2;

  pythia.init(is,isHead);

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
