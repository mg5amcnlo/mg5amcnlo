// StringInteraction.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for the classes involved in the modelling interactions
// between strings. This includes standard colour reconnections,
// perturbative swing, string shoving and rope hadronisation. The
// StringInterations base class is an interface to all of these, and
// can be used both during and after PartonLevel, as well as before,
// during and after string fragmentation.

#ifndef Pythia8_StringInteractions_H
#define Pythia8_StringInteractions_H

#include "Pythia8/SharedPointers.h"
#include "Pythia8/PhysicsBase.h"
#include "Pythia8/FragmentationSystems.h"

namespace Pythia8 {

//==========================================================================

// StringInteractions is the base class for handling colour
// reconnection, swing, shoving, and rope-type models for modifying
// string fragmentation.

class StringInteractions : public PhysicsBase {

public:

  // Empty constructor.
  StringInteractions() = default;

  // Empty virtual destructor
  virtual ~StringInteractions() {}

  // Function called from Pythia after the basic pointers.  The base
  // class will setup the standard behaviour of Pythia 8.2 according
  // to the given Settings. Derived classes should create objects of
  // the specific model objects to be used.
  virtual bool init();

  // Access the pointers to the different models.
  ColRecPtr getColourReconnections() const { return colrecPtr; }
  DipSwingPtr getDipoleSwing() const { return dipswingPtr; }
  StringRepPtr getStringRepulsion() const { return stringrepPtr; }
  FragModPtr getFragmentationModifier() const { return fragmodPtr; }

protected:

  // The object responsible for colour reconections in the end of
  // PartonLevel or in the beginning of HadronLevel.
  ColRecPtr colrecPtr{};

  // The object responsible for the perturbative swing which is always
  // active together with the TimeShower.
  DipSwingPtr dipswingPtr{};

  // The object responsible for repulsion between strings before
  // hadronisation (or calculating the repulsion before but actually
  // pushing the hadrons after fragmentation).
  StringRepPtr stringrepPtr{};

  // The object responsible for modifying fragmentation parameters due
  // to string interactions in each break-up (eg. in rope
  // hadronisation).
  FragModPtr fragmodPtr{};

};

//==========================================================================

// ColourReconnectionBase is responsible for doing standard colour
// reconnection in the end of the PartonLevel or in the beginning of
// HadronLevel.

class ColourReconnectionBase : public PhysicsBase {

public:

  // Empty default constructor.
  ColourReconnectionBase() = default;

  // Empty virtual destructor.
  virtual ~ColourReconnectionBase() {}

  // Called after PhysicsBase initHbPtrs has been called.
  virtual bool init() { return true; }

  // New beams possible for handling of hard diffraction.
  virtual void reassignBeamPtrs( BeamParticle* beamAPtrIn,
    BeamParticle* beamBPtrIn) {beamAPtr = beamAPtrIn;
    beamBPtr = beamBPtrIn;}

  // Do colour reconnection for current event.
  virtual bool next( Event & event, int oldSize) = 0;

};

//==========================================================================

// DipoleSwingBase is responsible for the perturbative swing and is active
// anytime the TimeShower is active.

class DipoleSwingBase : public PhysicsBase {

public:

  // Empty default constructor.
  DipoleSwingBase() = default;

  // Empty virtual destructor.
  virtual ~DipoleSwingBase() {}

  // Called after PhysicsBase initInfoPtr has been called.
  virtual bool init() { return true; }

  // New beams possible for handling of hard diffraction.
  virtual void reassignBeamPtrs( BeamParticle* beamAPtrIn,
    BeamParticle* beamBPtrIn, int beamOffsetIn = 0) {beamAPtr = beamAPtrIn;
    beamBPtr = beamBPtrIn; beamOffset = beamOffsetIn;}

  // Prepare system for evolution after each new interaction; identify ME.
  // Usage: prepare( iSys, event, limitPTmax).
  virtual void prepare( int , Event& , bool = true) = 0;

  // Update dipole list after a multiparton interactions rescattering.
  // Usage: rescatterUpdate( iSys, event).
  virtual void rescatterUpdate( int , Event& ) = 0;

  // Update dipole list after each ISR emission.
  // Usage: update( iSys, event, hasWeakRad).
  virtual void update( int , Event& , bool = false) = 0;

  // Select next pT in downwards evolution.
  // Usage: pTnext( event, pTbegAll, pTendAll, isFirstTrial, doTrialIn).
  virtual double pTnext( Event& , double , double ,
                         bool = false, bool = false) = 0;

  // Perform the swing previousl generated in pTnext.
  virtual bool swing( Event& event) = 0;

protected:

  // Offset the location of the beam pointers in an event.
  int beamOffset{};

};

//==========================================================================

// StringRepulsionBase is responsible for calculating and performing
// the repulsion between strings before the hadronisation.

class StringRepulsionBase : public PhysicsBase {

public:

  // Empty default constructor.
  StringRepulsionBase() = default;

  // Empty virtual destructor.
  virtual ~StringRepulsionBase() {}

  // Called after PhysicsBase initInfoPtr has been called.
  virtual bool init() { return true; }

  // Main virtual function, called before the hadronisation.
  virtual bool stringRepulsion(Event & event, ColConfig & colConfig) = 0;

  // Additional functionality for implementing repulsion post-factum
  // after the actual hadronisation has been done.
  virtual bool hadronRepulsion(Event &) { return true; }
};

//==========================================================================

// FragmentationModifierBase can change the parameters of the string
// fragmentation in each break-up.

class FragmentationModifierBase : public PhysicsBase {

public:

  // Empty default constructor.
  FragmentationModifierBase() = default;

  // Empty virtual destructor.
  virtual ~FragmentationModifierBase() {};

  // Called after PhysicsBase initInfoPtr has been called.
  virtual bool init() { return true; }

  // Called just before hadronisation starts.
  virtual bool initEvent(Event& event, ColConfig& colConfig) = 0;

  // The main virtual function for chaning the fragmentation
  // parameters.
  virtual bool doChangeFragPar(StringFlav* flavPtr, StringZ* zPtr,
   StringPT * pTPtr, double m2Had, vector<int> iParton, int endId) = 0;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_StringInteractions_H
