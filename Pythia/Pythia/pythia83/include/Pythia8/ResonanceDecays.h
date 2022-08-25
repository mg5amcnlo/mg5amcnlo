// ResonanceDecays.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the main class for performing resonance decays.
// ResonanceDecays: handles the sequential decay of resonances in process.

#ifndef Pythia8_ResonanceDecays_H
#define Pythia8_ResonanceDecays_H

#include "Pythia8/Basics.h"
#include "Pythia8/Event.h"
#include "Pythia8/Info.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PhysicsBase.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/ResonanceWidths.h"
#include "Pythia8/Settings.h"

namespace Pythia8 {

//==========================================================================

// The ResonanceDecays class handles the sequential decay of resonances
// that are part of the hard process (t, W, Z, H, SUSY,...).

class ResonanceDecays : public PhysicsBase {

public:

  // Constructor.
  ResonanceDecays() : id0(), mult(), m0() {}

  // Store pointers to Info and Rndm for error messages and random numbers.
  void init() {}

  // Generate the next decay sequence.
  bool next( Event& process, int iDecNow = 0);

private:

  // Constants: could only be changed in the code itself.
  static const int    NTRYCHANNEL, NTRYMASSES;
  static const double MSAFETY, WIDTHCUT, TINY, TINYBWRANGE,
                      WTCORRECTION[11];

  // Select masses of decay products.
  bool pickMasses();

  // Select colours of decay products.
  bool pickColours(int iDec, Event& process);

  // Select kinematics isotropic in phase space.
  bool pickKinematics();

  // Flavour, colour and momentum information.
  int            id0, mult;
  double         m0;
  vector<int>    idProd, cols, acols;
  vector<double> mProd;
  vector<Vec4>   pProd;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_ResonanceDecays_H
