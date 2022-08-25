// PythiaStdlib.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for

#include "Pythia8/PythiaStdlib.h"

namespace Pythia8 {

//==========================================================================

// Convert string to lowercase for case-insensitive comparisons.
// By default remove any initial and trailing blanks or escape characters.

string toLower(const string& name, bool trim) {

  // Copy string without initial and trailing blanks or escape characters.
  string temp = name;
  if (trim) {
    if (name.find_first_not_of(" \n\t\v\b\r\f\a") == string::npos) return "";
    int firstChar = name.find_first_not_of(" \n\t\v\b\r\f\a");
    int lastChar  = name.find_last_not_of(" \n\t\v\b\r\f\a");
    temp          = name.substr( firstChar, lastChar + 1 - firstChar);
  }

  // Convert to lowercase letter by letter.
  for (int i = 0; i < int(temp.length()); ++i) temp[i] = tolower(temp[i]);
  return temp;

}

//==========================================================================

} // end namespace Pythia8
