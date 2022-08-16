// DireHooks.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for Dire user hooks.

#ifndef Pythia8_DireHooks_H
#define Pythia8_DireHooks_H

#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/DireSplittings.h"

namespace Pythia8 {

//==========================================================================

// Hooks is base class for user access to program execution.

class DireHooks {

public:

  // Destructor.
  virtual ~DireHooks() {}

  // Initialize pointers and workEvent. Note: not virtual.
  void initPtr( Info* infoPtrIn, BeamParticle* beamAPtrIn,
    BeamParticle* beamBPtrIn) {
    infoPtr       = infoPtrIn;
    settingsPtr      = infoPtr->settingsPtr;
    particleDataPtr  = infoPtr->particleDataPtr;
    rndmPtr          = infoPtr->rndmPtr;
    beamAPtr         = beamAPtrIn;
    beamBPtr         = beamBPtrIn;
    coupSMPtr        = infoPtr->coupSMPtr;
    partonSystemsPtr = infoPtr->partonSystemsPtr;
  }

  // Initialisation after beams have been set by Pythia::init().
  virtual bool init() { return true; }

  // Possibility for user-defined splitting kernels.
  virtual bool canLoadFSRKernels() {return false;}
  virtual bool doLoadFSRKernels(
    std::unordered_map< string, DireSplitting* >&) {return false;}
  virtual bool canLoadISRKernels() {return false;}
  virtual bool doLoadISRKernels(
    std::unordered_map< string, DireSplitting* >&) {return false;}

  // Possibility for user-defined scale setting.
  virtual bool canSetRenScale()    {return false;}
  virtual bool canSetFacScale()    {return false;}
  virtual bool canSetStartScale()  {return false;}
  virtual double doGetRenScale(double x1, double x2, double sH, double tH,
   double uH, bool massless, double m1sq, double m2sq, double m3sq,
   double m4sq) {
   if (false) cout << x1*x2*sH*tH*uH*massless*m1sq*m2sq*m3sq*m4sq;
   return -1.0;
  }
  virtual double doGetFacScale(double x1, double x2, double sH, double tH,
    double uH, bool massless, double m1sq, double m2sq, double m3sq,
    double m4sq) {
    if (false) cout << x1*x2*sH*tH*uH*massless*m1sq*m2sq*m3sq*m4sq;
    return -1.0;
  }
  virtual double doGetStartScale(double x1, double x2, double sH, double tH,
    double uH, bool massless, double m1sq, double m2sq, double m3sq,
    double m4sq) {
    if (false) cout << x1*x2*sH*tH*uH*massless*m1sq*m2sq*m3sq*m4sq;
    return -1.0;
  }

protected:

  // Constructor.
  DireHooks() : infoPtr(0), settingsPtr(0), particleDataPtr(0), rndmPtr(0),
    beamAPtr(0), beamBPtr(0), coupSMPtr(0), partonSystemsPtr(0) {}

  // Pointer to various information on the generation.
  Info*          infoPtr;

  // Pointer to the settings database.
  Settings*      settingsPtr;

  // Pointer to the particle data table.
  ParticleData*  particleDataPtr;

 // Pointer to the random number generator.
  Rndm*          rndmPtr;

  // Pointers to the two incoming beams and to Pomeron beam-inside-beam.
  BeamParticle*  beamAPtr;
  BeamParticle*  beamBPtr;

  // Pointers to Standard Model couplings.
  CoupSM*        coupSMPtr;

  // Pointer to information on subcollision parton locations.
  PartonSystems* partonSystemsPtr;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_DireHooks_H
