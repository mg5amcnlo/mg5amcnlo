// ShowerModel.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for the classes involved in the modelling of parton showers.
// A ShowerModel object is supposed to keep track of and give Pythia
// access to objects implementing space-like and time-like showers
// (also separately in resonance decays). Pointers related to the
// matrix-element merging may be overwritten in the derived classes.
// The SimpleShowerModel implements the default Pythia behavior
// with SimpleTimeShower and SimpleSpaceShower.

#ifndef Pythia8_ShowerModel_H
#define Pythia8_ShowerModel_H

#include "Pythia8/SharedPointers.h"
#include "Pythia8/PhysicsBase.h"
#include "Pythia8/FragmentationSystems.h"

namespace Pythia8 {

//==========================================================================

// ShowerModel is the base class for handling parton-shower algorithms,
// including merging methods.

class ShowerModel : public PhysicsBase {

public:

  // Empty constructor.
  ShowerModel() = default;

  // Empty virtual destructor
  virtual ~ShowerModel() {}

  // Function called from Pythia after the basic pointers has been set.
  // Derived classes should create objects of the specific model objects
  // to be used. Pointers to merging and merging hooks may be overwritten
  // in derived classes.
  virtual bool init(MergingPtr mergPtrIn, MergingHooksPtr mergHooksPtrIn,
    PartonVertexPtr partonVertexPtrIn,
    WeightContainer* weightContainerPtrIn) = 0;

  // Function called from Pythia after the beam particles have been set up,
  // so that showers may be initialized after the beams are initialized.
  virtual bool initAfterBeams() = 0;

  // Access the pointers to the different model components.
  virtual TimeShowerPtr   getTimeShower() const { return timesPtr; }
  virtual TimeShowerPtr   getTimeDecShower() const { return timesDecPtr; }
  virtual SpaceShowerPtr  getSpaceShower() const { return spacePtr; }
  virtual MergingPtr      getMerging() const { return mergingPtr; }
  virtual MergingHooksPtr getMergingHooks() const { return mergingHooksPtr; }

protected:

  // The object responsible for generating time-like showers.
  TimeShowerPtr   timesPtr{};

  // The object responsible for generating time-like showers in decays.
  TimeShowerPtr   timesDecPtr{};

  // The object responsible for generating space-like showers.
  SpaceShowerPtr  spacePtr{};

  // The object responsible for merging with matrix elements.
  MergingPtr      mergingPtr{};

  // The object responsible for user modifications to the merging.
  MergingHooksPtr mergingHooksPtr{};

};

//==========================================================================

// The shower model class handling the default Pythia shower model
// with SimpleTimeShower and SimpleSpaceShower classes.

class SimpleShowerModel : public ShowerModel {

public:

  // Empty constructor.
  SimpleShowerModel() = default;

  // Empty virtual destructor
  virtual ~SimpleShowerModel() override {}

  // Function called from Pythia after the basic pointers has been set.
  virtual bool init(MergingPtr mergPtrIn, MergingHooksPtr mergHooksPtrIn,
                    PartonVertexPtr partonVertexPtrIn,
                    WeightContainer* weightContainerPtrIn) override;

  // Function called from Pythia after the beam particles have been set up,
  // so that showers may be initialized after the beams are initialized.
  // Currently only dummy dunction.
  virtual bool initAfterBeams() override { return true; }

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_ShowerModel_H
