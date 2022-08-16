// VinciaDiagnostics.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for the VinciaDiagnostics class.

#ifndef Pythia8_VinciaDiagnostics_H
#define Pythia8_VinciaDiagnostics_H

#include "Pythia8/UserHooks.h"

namespace Pythia8 {

//==========================================================================

// Vincia diagnostics.

class VinciaDiagnostics : public UserHooks {

 public:

  // Initialise pointers.
  void initPtr(Info* infoPtrIn) {infoPtr = infoPtrIn;}

  // Initialise.
  void init() {};

  // Define and increment a counter (default is increment by 1).
  void increment(string methodName, string variableName, double inc = 1);

  // Called when "name" starts.
  void start(string name);

  // Called when "name" stops.
  void stop(string name, string counter = "", double inc = 1);

  // Print diagnostics.
  void print();

 private:

  map<string, bool> isRunning;
  map<string, clock_t> startTime;
  map<string, double> nStarts, nRestarts;
  map<string, double> runTime;
  map<string, Hist> hRunTime;
  map<string, map<string, double> > counters;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_VinciaDiagnostics_H
