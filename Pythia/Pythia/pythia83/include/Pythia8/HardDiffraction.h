// HardDiffraction.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Author: Christine Rasmussen.

// Header file for the HardDiffraction class.

#ifndef Pythia8_HardDiffraction_H
#define Pythia8_HardDiffraction_H

#include "Pythia8/Basics.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/BeamRemnants.h"
#include "Pythia8/Info.h"
#include "Pythia8/MultipartonInteractions.h"
#include "Pythia8/PhysicsBase.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/SpaceShower.h"
#include "Pythia8/TimeShower.h"

namespace Pythia8 {

//==========================================================================

// HardDiffraction class.
// This class handles hard diffraction, together with PartonLevel.

class HardDiffraction : public PhysicsBase {

public:

  // Constructor and destructor.
  HardDiffraction() : isGammaA(), isGammaB(), isGammaGamma(), usePomInPhoton(),
    pomFlux(), iBeam(), idA(), idB(), rescale(), normPom(), sigTotRatio(),
    a1(), a2(), a3(), A1(), A2(), A3(), a0(), ap(), b0(), mA(), mB(), s(),
    s1(), s2(), s3(), s4(), xPomA(), xPomB(), tPomA(), tPomB(), thetaPomA(),
    thetaPomB(), tmpPomPtr() {};
  ~HardDiffraction() {}

  // Initialise constant and the beams to be considered.
  void init(BeamParticle* beamAPtrIn,  BeamParticle* beamBPtrIn);

  // Main routine to check if event is from diffractive PDF.
  bool isDiffractive(int iBeamIn = 1, int partonIn = 0,
    double xIn = 0., double Q2In = 0., double xfIncIn = 0.);

  // Get diffractive values.
  double getXPomeronA()     {return xPomA;}
  double getXPomeronB()     {return xPomB;}
  double getTPomeronA()     {return tPomA;}
  double getTPomeronB()     {return tPomB;}
  double getThetaPomeronA() {return thetaPomA;}
  double getThetaPomeronB() {return thetaPomB;}

private:

  // Constants: could only be changed in the code itself.
  static const double TINYPDF;
  static const double POMERONMASS;
  static const double RHOMASS;
  static const double PROTONMASS;
  static const double DIFFMASSMARGIN;

  // Initialization and event data.
  bool isGammaA, isGammaB, isGammaGamma, usePomInPhoton;
  int    pomFlux, iBeam, idA, idB;
  double rescale, normPom, sigTotRatio,
         a1, a2, a3, A1, A2, A3, a0, ap, b0,
         mA, mB, s, s1, s2, s3, s4,
         xPomA, xPomB, tPomA, tPomB, thetaPomA, thetaPomB;

  // Pointer to temporary Pomeron PDF.
  BeamParticle*   tmpPomPtr;

  // Return Pomeron flux inside proton, integrated over t.
  double xfPom(double xIn = 0.);

  // Pick a t value for a given x.
  double pickTNow(double xIn = 0.);

  // Return Pomeron flux inside proton, differential in t.
  double xfPomWithT(double xIn = 0., double tIn = 0.);

  // Make t range available as a pair.
  pair<double, double> tRange(double xIn = 0.);

  // Calculate scattering angle from  given x and t.
  double getThetaNow(double xIn = 0., double tIn = 0.);

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_HardDiffraction_H
