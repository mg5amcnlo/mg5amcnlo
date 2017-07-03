// ResonanceWidthsDM.cc is a part of the PYTHIA event generator.
// Copyright (C) 2017 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for
// the ResonanceWidthsDM class and classes derived from it.

#include "Pythia8/ParticleData.h"
#include "Pythia8/ResonanceWidthsDM.h"
#include "Pythia8/PythiaComplex.h"

namespace Pythia8 {

//==========================================================================

// The ResonanceZp class.
// Derived class for Z'^0 properties (using PDG id 55, DMmed(s=1)).

//--------------------------------------------------------------------------

// Initialize constants.

void ResonanceZp::initConstants() {

  // Locally stored properties and couplings.
  gq = settingsPtr->parm("Zp:vq");
  gX = settingsPtr->parm("Zp:vX");

}

//--------------------------------------------------------------------------

// Calculate various common prefactors for the current mass.

void ResonanceZp::calcPreFac(bool) {

  // Common coupling factors.
  preFac      = mRes / 12.0 / M_PI;

}

//--------------------------------------------------------------------------

// Calculate width for currently considered channel.

void ResonanceZp::calcWidth(bool) {

  // Check that above threshold.
  if (ps == 0. || id1 * id2 > 0) return;

  double mRat2 = pow2(mf1 / mRes);
  double kinfac = (1 - 4 * mRat2) * (1. + 2 * mRat2);

  widNow = 0.;

  if(id1Abs < 7)
    widNow = 3. * pow2(gq) * preFac * kinfac;

  if(id1Abs == 52)
    widNow = pow2(gX) * preFac * kinfac;

}

//==========================================================================

} // end namespace Pythia8
