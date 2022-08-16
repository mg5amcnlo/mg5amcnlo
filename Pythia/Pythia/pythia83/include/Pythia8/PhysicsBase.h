// PhysicsBase.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the base class for physics classes used inside Pyhia8.

// Still to convert:
// BeamParticle
// BeamShape

#ifndef Pythia8_PhysicsBase_H
#define Pythia8_PhysicsBase_H

#include "Pythia8/Info.h"
#include "Pythia8/Settings.h"
#include "Pythia8/SharedPointers.h"

namespace Pythia8 {

//==========================================================================

// Classes that implement physics models should inherit from the PhysicsBase
// class. It includes pointers to objects set up in the controlling Pythia
// object to take care of bookkeeping and simpler service tasks.

class PhysicsBase {

public:

  // Enumerate the different status codes the event generation can have.
  enum Status { INCOMPLETE = -1, COMPLETE = 0, CONSTRUCTOR_FAILED,
    INIT_FAILED, LHEF_END, LOWENERGY_FAILED, PROCESSLEVEL_FAILED,
    PROCESSLEVEL_USERVETO, MERGING_FAILED, PARTONLEVEL_FAILED,
    PARTONLEVEL_USERVETO, HADRONLEVEL_FAILED, CHECK_FAILED,
    OTHER_UNPHYSICAL, HEAVYION_FAILED, HADRONLEVEL_USERVETO };

  // This function is called from above for physics objects used in a run.
  void initInfoPtr(Info& infoPtrIn);

  // Empty virtual destructor.
  virtual ~PhysicsBase() {}

  // Shorthand to read settings values.
  bool   flag(string key) const {return settingsPtr->flag(key);}
  int    mode(string key) const {return settingsPtr->mode(key);}
  double parm(string key) const {return settingsPtr->parm(key);}
  string word(string key) const {return settingsPtr->word(key);}

protected:

  // Default constructor.
  PhysicsBase() {}

  // If an object needs to set up infoPtr for sub objects, override this
  // and call registerSubObject for each object in question.
  virtual void onInitInfoPtr() {}

  // This function is called in the very beginning of each Pythia::next call.
  virtual void onBeginEvent() {}

  // This function is called in the very end of each Pythia::next call
  // with the argument set to the current status of the event.
  virtual void onEndEvent(Status) {}

  // This function is called from the Pythia::stat() call.
  virtual void onStat() {}

  // Register a sub object that should have its information in sync with this.
  void registerSubObject(PhysicsBase& pb);

  // Pointer to various information on the generation.
  // This is also the place from which a number of pointers are recovered.
  Info*          infoPtr       =  {};

  // Pointer to the settings database.
  Settings*      settingsPtr      = {};

  // Pointer to the particle data table.
  ParticleData*  particleDataPtr  = {};

  // Pointer to the hadron widths data table
  HadronWidths*  hadronWidthsPtr  = {};

  // Pointer to the random number generator.
  Rndm*          rndmPtr          = {};

  // Pointers to SM and SUSY couplings.
  CoupSM*        coupSMPtr        = {};
  CoupSUSY*      coupSUSYPtr      = {};

  // Pointers to the two incoming beams and to Pomeron, photon or VMD
  // beam-inside-beam cases.
  BeamParticle*  beamAPtr         = {};
  BeamParticle*  beamBPtr         = {};
  BeamParticle*  beamPomAPtr      = {};
  BeamParticle*  beamPomBPtr      = {};
  BeamParticle*  beamGamAPtr      = {};
  BeamParticle*  beamGamBPtr      = {};
  BeamParticle*  beamVMDAPtr      = {};
  BeamParticle*  beamVMDBPtr      = {};

  // Pointer to information on subcollision parton locations.
  PartonSystems* partonSystemsPtr = {};

  // Pointer to the total/elastic/diffractive cross sections.
  SigmaTotal*    sigmaTotPtr      = {};

  // A set of sub objects that should have their information in sync
  // with This.
  set<PhysicsBase*> subObjects;

  // Pointer to the UserHooks object (needs to be sett to null in
  // classes deriving from UserHooks to avoid closed loop ownership).
  UserHooksPtr   userHooksPtr;

private:

  friend class Pythia;

  // Calls onBeginEvent, then propagates the call to all sub objects
  void beginEvent();

  // Calls onEndEvent, then propagates the call to all sub objects
  void endEvent(Status status);

  // Calls onStat, then propagates the call to all sub objects
  void stat();

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_PhysicsBase_H
