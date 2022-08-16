// HiddenValleyFragmentation.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the classes for Hidden-Valley fragmentation.

#ifndef Pythia8_HiddenValleyFragmentation_H
#define Pythia8_HiddenValleyFragmentation_H

#include "Pythia8/Basics.h"
#include "Pythia8/Event.h"
#include "Pythia8/FragmentationFlavZpT.h"
#include "Pythia8/FragmentationSystems.h"
#include "Pythia8/Info.h"
#include "Pythia8/MiniStringFragmentation.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StringFragmentation.h"

namespace Pythia8 {

//==========================================================================

// The HVStringFlav class is used to select HV-quark and HV-hadron flavours.

class HVStringFlav : public StringFlav {

public:

  // Constructor.
  HVStringFlav() : nFlav(), probVector() {}

  // Destructor.
  ~HVStringFlav() {}

  // Initialize data members.
  void init();

  // Pick a new flavour (including diquarks) given an incoming one.
  FlavContainer pick(FlavContainer& flavOld, double, double, bool);

  // Combine two flavours (including diquarks) to produce a hadron.
  int combine(FlavContainer& flav1, FlavContainer& flav2);

private:

  // Initialization data, to be read from Settings.
  int    nFlav;
  double probVector;

};

//==========================================================================

// The HVStringPT class is used to select select HV transverse momenta.

class HVStringPT : public StringPT {

public:

  // Constructor.
  HVStringPT() {}

  // Destructor.
  ~HVStringPT() {}

  // Initialize data members.
  void init();

};

//==========================================================================

// The HVStringZ class is used to sample the HV fragmentation function f(z).

class HVStringZ : public StringZ {

public:

  // Constructor.
  HVStringZ() : mqv2(), bmqv2(), rFactqv(), mhvMeson() {}

  // Destructor.
  virtual ~HVStringZ() {}

  // Initialize data members.
  void init();

  // Fragmentation function: top-level to determine parameters.
  double zFrag( int idOld, int idNew = 0, double mT2 = 1.);

  // Parameters for stopping in the middle; for now hardcoded.
  virtual double stopMass()    {return 1.5 * mhvMeson;}
  virtual double stopNewFlav() {return 2.0;}
  virtual double stopSmear()   {return 0.2;}

private:

  // Initialization data, to be read from Settings and ParticleData.
  double mqv2, bmqv2, rFactqv, mhvMeson;

};

//==========================================================================

// The HiddenValleyFragmentation class contains the routines
// to fragment a Hidden Valley partonic system.

class HiddenValleyFragmentation : public PhysicsBase {

public:

  // Constructor.
  HiddenValleyFragmentation() : doHVfrag(false), nFlav(), hvOldSize(),
    hvNewSize(), mhvMeson(), mSys() {}

  // Initialize and save pointers.
  bool init();

  // Do the fragmentation: driver routine.
  bool fragment(Event& event);

protected:

  virtual void onInitInfoPtr() override {
    registerSubObject(hvStringFrag);
    registerSubObject(hvMinistringFrag);
    registerSubObject(hvFlavSel);
    registerSubObject(hvPTSel);
    registerSubObject(hvZSel);
  }

private:

  // Data mambers.
  bool          doHVfrag;
  int           nFlav, hvOldSize, hvNewSize;
  double        mhvMeson, mSys;
  vector<int>   ihvParton;

  // Configuration of colour-singlet systems.
  ColConfig     hvColConfig;

  // Temporary event record for the Hidden Valley system.
  Event         hvEvent;

  // The generator class for Hidden Valley string fragmentation.
  StringFragmentation hvStringFrag;

  // The generator class for special low-mass HV string fragmentation.
  MiniStringFragmentation hvMinistringFrag;

  // Pointers to classes for flavour, pT and z generation in HV sector.
  HVStringFlav hvFlavSel;
  HVStringPT   hvPTSel;
  HVStringZ    hvZSel;

  // Extract HV-particles from event to hvEvent. Assign HV-colours.
  bool extractHVevent(Event& event);

  // Collapse of low-mass system to one HV-meson.
  bool collapseToMeson();

  // Insert HV particles from hvEvent to event.
  bool insertHVevent(Event& event);

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_HiddenValleyFragmentation_H
