// main31.cc is a part of the PYTHIA event generator.
// Copyright (C) 2018 Richard Corke, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Example how to perform merging with PWOHEG-BOX events,
// based on the code found in include/Pythia8Plugins/PowhegHooks.h.

#include "Pythia8/Pythia.h"
using namespace Pythia8;
#include "TestHooks.h"

//==========================================================================

int main(  int, char* argv[] ) {

  // Generator
  Pythia pythia;

  // Load configuration file
  pythia.readFile("main-test-scales.cmnd");

  ostringstream file;
  file << argv[1];

  pythia.readString("Beams:LHEF = " + file.str());

  ostringstream npartons;
  npartons << argv[2];
  pythia.readString("TimeShower:nPartonsInBorn = " + npartons.str()); 


  int nEvent      = pythia.settings.mode("Main:numberOfEvents");

  Hist tH("tH",100,0.0,200.0);
  Hist tS("tS",100,0.0,200.0);
  Hist tH2("tH2",100,0.0,200.0);
  Hist tS2("tS2",100,0.0,200.0);

  TestHook* testHook = new TestHook(1,&tH, &tS, &tH2, &tS2); // veto after first emission
  pythia.setUserHooksPtr((UserHooks *) testHook);

  // Initialise and list settings
  pythia.init();

  // Begin event loop; generate until nEvent events are processed
  // or end of LHEF file
  int iEvent = 0;
  while (true) {
    // Generate the next event
    if (!pythia.next()) {
      // If failure because reached end of file then exit event loop
      if (pythia.info.atEndOfFile()) break;
      // Otherwise count event failure and continue/exit as necessary
      cout << "Warning: event " << iEvent << " failed" << endl;
      continue;
    }
    // If nEvent is set, check and exit loop if necessary
    ++iEvent;
    if (nEvent != 0 && iEvent == nEvent) break;
  } // End of event loop.

  // Statistics, histograms and veto information
  pythia.stat();

  // Done.
  delete testHook;

  tH /= nEvent;
  tS /= nEvent;
  tH2 /= nEvent;
  tS2 /= nEvent;

  ostringstream prefix;
  prefix.str("");
  prefix << argv[3];

  ofstream write;
  write.open(prefix.str()+"_avgscale_vs_tH.dat");
  tH.table(write);
  write.close();

  write.open(prefix.str()+"_avgscale_vs_tS.dat");
  tS.table(write);
  write.close();

  write.open(prefix.str()+"_tH.dat");
  tH2.table(write);
  write.close();

  write.open(prefix.str()+"_tS.dat");
  tS2.table(write);
  write.close();



  return 0;
}
