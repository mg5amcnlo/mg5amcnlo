// StringInteractions.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the
// StringInteractions class.

#include "Pythia8/StringInteractions.h"
#include "Pythia8/ColourReconnection.h"

namespace Pythia8 {

//==========================================================================

// The StringInteractions class.

//--------------------------------------------------------------------------

// The base class init() method only implements the ColourReconnection
// model from Pythia 8.2.

bool StringInteractions::init() {
  subObjects.clear();
  if ( flag("ColourReconnection:reconnect") ||
       flag("ColourReconnection:forceHadronLevelCR") ) {
    colrecPtr = make_shared<ColourReconnection>();
    registerSubObject(*colrecPtr);
  }

  return true;
}

//==========================================================================

// Just needs to make sure something ends up in an object file.

// FragmentationModifierBase::~FragmentationModifierBase() {}

//==========================================================================

} // end namespace Pythia8
