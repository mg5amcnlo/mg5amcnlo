// Hooks.h is a part of the DIRE plugin to the PYTHIA event generator.
// Copyright (C) 2018 Stefan Prestel.

#ifndef Pythia8_Hooks_H
#define Pythia8_Hooks_H

#include "Pythia8/PythiaStdlib.h"

#include "Dire/Splittings.h"

namespace Pythia8 {

//==========================================================================

// Hooks is base class for user access to program execution.

class Hooks {

public:

  // Destructor.
  virtual ~Hooks() {}

  // Initialize pointers and workEvent. Note: not virtual.
  void initPtr( Info* infoPtrIn, Settings* settingsPtrIn,
    ParticleData* particleDataPtrIn,  Rndm* rndmPtrIn,
    BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn,
    CoupSM* coupSMPtrIn, PartonSystems* partonSystemsPtrIn) {
    infoPtr          = infoPtrIn;
    settingsPtr      = settingsPtrIn;
    particleDataPtr  = particleDataPtrIn;
    rndmPtr          = rndmPtrIn;
    beamAPtr         = beamAPtrIn;
    beamBPtr         = beamBPtrIn;
    coupSMPtr        = coupSMPtrIn;
    partonSystemsPtr = partonSystemsPtrIn;
  }

  // Initialisation after beams have been set by Pythia::init().
  virtual bool init() { return true; }

  // Possibility for user-defined splitting kernels.
  virtual bool canLoadFSRKernels() {return false;}
  virtual bool doLoadFSRKernels( map< string, Splitting* >&) {return false;}
  virtual bool canLoadISRKernels() {return false;}
  virtual bool doLoadISRKernels( map< string, Splitting* >&) {return false;}

protected:

  // Constructor.
  Hooks() : infoPtr(0), settingsPtr(0), particleDataPtr(0), rndmPtr(0),
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

#endif // Pythia8_Hooks_H
