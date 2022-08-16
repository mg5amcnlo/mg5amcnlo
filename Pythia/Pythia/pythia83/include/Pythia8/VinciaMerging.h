// VinciaMerging.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Helen Brooks, Christian T Preuss
// This file contains the VinciaMerging class.

#ifndef Pythia8_VinciaMerging_H
#define Pythia8_VinciaMerging_H

#include "Pythia8/Merging.h"
#include "Pythia8/VinciaCommon.h"

namespace Pythia8 {

//==========================================================================

// Merging wrapper class for Vincia.

class VinciaMerging : public Merging {

 public:

  // Overridden methods.
  void init() override;
  void statistics() override;
  int mergeProcess(Event& process) override;

  // Set the verbosity level.
  void setVerbose(int verboseIn) {verbose=verboseIn;};

 private:

  // Steers the sector merging.
  int mergeProcessSector(Event& process);

  // Insert resonances in event record if not there.
  bool insertResonances(Event& process);

  // Flags that affect behaviour.
  bool doMerging, doSectorMerging, includeWtInXsec, doXSecEstimate,
    doMergeRes, doInsertRes;

  // Maximum additional jets (inclusive total).
  int nMaxJets;
  // Maximum additional jets in each res system.
  int nMaxJetsRes;
  // Number of resonance systems allowed to produce additional jets.
  int nMergeResSys;

  // Event counters.
  int nAbort, nBelowMS, nVeto, nTotal;
  vector<int> nVetoByMult, nTotalByMult;

  // Statistics.
  map<int, double> historyCompTime;
  map<int, int> nHistories;

  // Debug verbosity.
  int verbose;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_VinciaMerging_H
