// SusyResonanceWidths.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand
// Main author of this file: N. Desai
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for resonance properties: dynamical widths etc.
// WidthFunctions: base class for SUSY 3-body decay width functions.

#ifndef Pythia8_SusyWidthFunctions_H
#define Pythia8_SusyWidthFunctions_H

#include "Pythia8/MathTools.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/SusyCouplings.h"

namespace Pythia8 {

//==========================================================================

class WidthFunction {

public:

  // Constructor and destructor.
  WidthFunction() : infoPtr(), particleDataPtr(), coupSMPtr(), coupSUSYPtr(),
    idRes(), idInt(), id1(), id2(), id3(), id4(), mRes(),
    mInt(), gammaInt(), m1(), m2(), m3(), m4() { };
  virtual ~WidthFunction() { };

  // Public methods.
  void setPointers(Info* infoPtrIn);
  virtual double getWidth( int, int) { return 0.0; };

  // Definition of width function.
  virtual double f(double xIn) = 0;

protected:

  Info*         infoPtr;
  ParticleData* particleDataPtr;
  CoupSM*       coupSMPtr;
  CoupSUSY*     coupSUSYPtr;
  int idRes, idInt, id1, id2, id3, id4;
  double mRes, mInt, gammaInt, m1, m2 , m3, m4;

};

//==========================================================================

// Class StauWidths.

class StauWidths : public WidthFunction {

public:

  // Constructor.
  StauWidths() : fnSwitch(), delm(), f0(), gf(), cons(), wparam() {}

  // Destructor.
  ~StauWidths() { };

  // Public method.
  double getWidth(int idResIn, int idIn) override;

protected:

  int fnSwitch; // Switch between multiple functions
  void setChannel(int idResIn, int idIn);
  double f(double xIn) override;

  double delm, f0, gf, cons, wparam;
  complex gL, gR;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_SusyResonanceWidths_H
