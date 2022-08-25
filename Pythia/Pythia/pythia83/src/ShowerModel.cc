// ShowerModel.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the
// ShowerModel class.

#include "Pythia8/ShowerModel.h"
#include "Pythia8/SimpleTimeShower.h"
#include "Pythia8/SimpleSpaceShower.h"
#include "Pythia8/Merging.h"
#include "Pythia8/MergingHooks.h"

namespace Pythia8 {

//==========================================================================

// Initialize the SimpleShowerModel.

bool SimpleShowerModel::init(MergingPtr mergPtrIn,
  MergingHooksPtr mergHooksPtrIn, PartonVertexPtr,
  WeightContainer*) {
  subObjects.clear();
  mergingPtr = mergPtrIn;
  if ( mergingPtr ) registerSubObject(*mergingPtr);
  mergingHooksPtr = mergHooksPtrIn;
  if ( mergingHooksPtr ) registerSubObject(*mergingHooksPtr);
  timesPtr = timesDecPtr = make_shared<SimpleTimeShower>();
  registerSubObject(*timesPtr);
  spacePtr = make_shared<SimpleSpaceShower>();
  registerSubObject(*spacePtr);
  return true;
}

//==========================================================================

} // end namespace Pythia8
