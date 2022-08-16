// DireSplittings.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the
// DireSplitting class.

#include "Pythia8/DireSplittings.h"
#include "Pythia8/DireSpace.h"
#include "Pythia8/DireTimes.h"

namespace Pythia8 {

//==========================================================================

// The Splitting class.

//--------------------------------------------------------------------------

void DireSplitting::init() {

  renormMultFac      = 1.;
  if (id.find("Dire_isr_") != string::npos)
    renormMultFac    = settingsPtr->parm("SpaceShower:renormMultFac");
  else
    renormMultFac    = settingsPtr->parm("TimeShower:renormMultFac");

  if ( id.find("_qcd_")      != string::npos) is_qcd  = true;
  if ( id.find("_qed_")      != string::npos) is_qed  = true;
  if ( id.find("_ew_")       != string::npos) is_ewk  = true;
  if ( id.find("Dire_")      != string::npos) is_dire = true;
  if ( id.find("Dire_isr_")  != string::npos) is_isr  = true;
  if ( id.find("Dire_fsr_")  != string::npos) is_fsr  = true;

  nameHash = shash(id);

}

//--------------------------------------------------------------------------

double DireSplitting::getKernel(string key) {
  unordered_map<string, double>::iterator it = kernelVals.find(key);
  if ( it == kernelVals.end() ) return 0./0.;
  return it->second;
}

//==========================================================================

} // end namespace Pythia8
