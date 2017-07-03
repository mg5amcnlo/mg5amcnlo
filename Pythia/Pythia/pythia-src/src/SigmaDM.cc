// SigmaDM.cc is a part of the PYTHIA event generator.
// Copyright (C) 2017 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the
// Dark Matter simulation classes.

#include "Pythia8/SigmaDM.h"

namespace Pythia8 {

//==========================================================================

// Sigma2ffbar2Zp2XX class.
// Cross section for f fbar' -> Zprime -> X X.  (Zprime a.k.a. DMmed(s=1).)

//--------------------------------------------------------------------------

// Initialize process.

void Sigma2ffbar2Zp2XX::initProc() {

  // Store mass and width for propagator.
  mRes      = particleDataPtr->m0(55);
  GammaRes  = particleDataPtr->mWidth(55);
  m2Res     = mRes*mRes;

  // Coupling strengths appear in prefactor.
  double gq = settingsPtr->parm("Zp:vq");
  double gX = settingsPtr->parm("Zp:vX");
  preFac = pow2(gq * gX);

  // Set pointer to particle properties and decay table.
  particlePtr = particleDataPtr->particleDataEntryPtr(32);

}

//--------------------------------------------------------------------------

// Evaluate sigmaHat(sHat), part independent of incoming flavour.

void Sigma2ffbar2Zp2XX::sigmaKin() {

  double propZp = 1.0 / ( pow2(sH - m2Res) + pow2(mRes * GammaRes) );
  sigma0        = preFac * propZp;

}

//--------------------------------------------------------------------------

// Evaluate sigmaHat(sHat), including incoming flavour dependence.

double Sigma2ffbar2Zp2XX::sigmaHat() {

  // Check for allowed flavour combinations
  if (id1 + id2 != 0 || abs(id1) > 6 ) return 0.;

  // Calculate kinematics dependence.
  double trace  = 8.0 * pow2(s3 - tH) + pow2(s3 - uH) + 2.0 * s3 * sH ;
  double sigma = sigma0 * trace;

  // Colour factors.
  if (abs(id1) < 7) sigma /= 3.;

  // Answer.
  return sigma;

}

//--------------------------------------------------------------------------

// Select identity, colour and anticolour.

void Sigma2ffbar2Zp2XX::setIdColAcol() {

  setId(id1, id2, 52, -52);

  // Colour flow topologies. Swap when antiquarks.
  if (abs(id1) < 9) setColAcol( 1, 0, 0, 1, 0, 0, 0, 0);
  else              setColAcol( 0, 0, 0, 0, 0, 0, 0, 0);
  if (id1 < 0) swapColAcol();

}

//==========================================================================

} // end namespace Pythia8
