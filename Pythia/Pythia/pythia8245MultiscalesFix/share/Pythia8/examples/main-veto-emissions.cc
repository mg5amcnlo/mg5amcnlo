// main31.cc is a part of the PYTHIA event generator.
// Copyright (C) 2018 Richard Corke, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Example how to perform merging with PWOHEG-BOX events,
// based on the code found in include/Pythia8Plugins/PowhegHooks.h.

#include "Pythia8/Pythia.h"
using namespace Pythia8;
#include "VetoAfterEmissionNumber.h"

//==========================================================================

int main() {

  // Generator
  Pythia pythia;

  // Load configuration file
  pythia.readFile("main31.cmnd");
  int nEvent      = pythia.settings.mode("Main:numberOfEvents");

  VetoAfterEmissionNumber* vetoHook = new VetoAfterEmissionNumber(1); // veto after first emission
  //VetoAfterEmissionNumber* vetoHook = new VetoAfterEmissionNumber(2); // veto after 2nd emission
  pythia.setUserHooksPtr((UserHooks *) vetoHook);

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
  delete vetoHook;

  return 0;
}
