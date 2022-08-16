// ShowerMEs.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Stefan Prestel, Philip Ilten, Torbjorn
// Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Virtual interface for external matrix element plugins for parton showers.

#include "Pythia8/ShowerMEs.h"

namespace Pythia8 {

//==========================================================================

// The parton shower matrix-element interface.

//--------------------------------------------------------------------------

// Set pointers to required PYTHIA 8 objects.

void ShowerMEs::initPtrVincia(Info* infoPtrIn,
  SusyLesHouches* slhaPtrIn, VinciaCommon* vinComPtrIn) {
  infoPtr         = infoPtrIn;
  coupSMPtr       = infoPtr->coupSMPtr;
  particleDataPtr = infoPtr->particleDataPtr;
  settingsPtr     = infoPtr->settingsPtr;
  rndmPtr         = infoPtr->rndmPtr;
  slhaPtr         = slhaPtrIn;
  vinComPtr       = vinComPtrIn;
  isInitPtr       = true;
}

//--------------------------------------------------------------------------

// Set helicities for a particle state.

bool ShowerMEs::selectHelicitiesVincia(vector<Particle>& state, int nIn,
  bool force) {
  // If force = true, erase any existing helicities.
  if (force) {
    for (int i=0; i<(int)state.size(); ++i) state[i].pol(9);
  }

  // Get the matrix element (automatically sums over any h=9 particles).
  double me2sum = me2Vincia(state, nIn);
  if (verbose >= VinciaConstants::DEBUG)
    cout << " ShowerMEs::selectHelicities(): "
         << scientific << me2sum << endl;

  // Did we find the ME, (me2() returns -1 if no ME found).
  if (me2sum <= 0.) return false;

  // Check how many helicity configurations we summed over.
  int nHelConf = me2hel.size();
  if (nHelConf <= 0) return false;

  // Add up all helicity matrix
  double me2helsum = 0.;
  for (auto it = me2hel.begin(); it!=me2hel.end(); it++) {
    me2helsum += it->second;
  }

  // Random number between zero and me2sum (trivial if only one helConf).
  double ranHelConf = 0.0;
  vector<int> hSelected;
  if (nHelConf >= 2) ranHelConf = rndmPtr->flat() * me2helsum;
  for (map< vector<int>, double>::iterator it=me2hel.begin();
       it != me2hel.end(); ++it) {
    // Progressively subtract each me2hel and check when we cross zero.
    ranHelConf -= it->second;
    if (ranHelConf <= 0.0) {
      hSelected = it->first;
      break;
    }
  }
  if (ranHelConf > 0.) return false;

  // Set helicity configuration.
  for (int i = 0; i < (int)state.size(); ++i) state[i].pol(hSelected[i]);
  if (verbose >= VinciaConstants::DEBUG)
    cout << " ShowerMEs::selectHelicities(): selected " <<
      makeLabelVincia(hSelected, nIn, false) << endl;
  return true;

}

//--------------------------------------------------------------------------

// Convert process id codes (or helicity values) to string.

string ShowerMEs::makeLabelVincia(vector<int>& id, int nIn, bool useNames)
  const {
  string label = "{";
  for (int i = 0; i < (int)id.size(); ++i) {
    string idName;
    if (useNames && id[i] != 0) idName = particleDataPtr->name(id[i]);
    else idName = num2str(id[i]);
    if (i == nIn-1) idName += " ->";
    label += idName+" ";
  }
  label += "}";
  return label;
}

//--------------------------------------------------------------------------

// Fill a vector of IDs.

void ShowerMEs::fillIds(const Event& event, vector<int>& in, vector<int>& out)
  const {
  in.push_back(event[3].id());
  in.push_back(event[4].id());
  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].isFinal() ) out.push_back(event[i].id());
  }
}

//--------------------------------------------------------------------------

// Fill a vector of momenta.

void ShowerMEs::fillMoms(const Event& event, vector<Vec4>& p) const {
  p.push_back(event[3].p());
  p.push_back(event[4].p());
  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].isFinal() ) p.push_back(event[i].p());
  }
}

//--------------------------------------------------------------------------

// Fill a vector of colors.

void ShowerMEs::fillCols(const Event& event, vector<int>& colors) const {
  colors.push_back(event[3].col()); colors.push_back(event[3].acol());
  colors.push_back(event[4].col()); colors.push_back(event[4].acol());
  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].isFinal() ) {
      colors.push_back(event[i].col());
      colors.push_back(event[i].acol());
    }
  }
}


//--------------------------------------------------------------------------

// Return the momenta.

vector<vector<double> > ShowerMEs::fillMoms(const Event& event) const {
  vector<Vec4> p;
  fillMoms(event, p);
  vector< vector<double> > ret;
  for (int i = 0; i < int(p.size()); i++ ) {
    vector<double> p_tmp(4, 0.);
    p_tmp[0] = isnan(p[i].e())  ? 0.0 : p[i].e() ;
    p_tmp[1] = isnan(p[i].px()) ? 0.0 : p[i].px();
    p_tmp[2] = isnan(p[i].py()) ? 0.0 : p[i].py();
    p_tmp[3] = isnan(p[i].pz()) ? 0.0 : p[i].pz();
    ret.push_back(p_tmp);
  }
  return ret;
}

//==========================================================================

// Interface to external matrix elements for parton shower matrix
// element corrections.

//--------------------------------------------------------------------------

// Set pointers to required PYTHIA 8 objects.

void ShowerMEsPlugin::initPtrVincia(Info* infoPtrIn,
  SusyLesHouches* slhaPtrIn, VinciaCommon* vinComPtrIn) {
  infoPtr         = infoPtrIn;
  coupSMPtr       = infoPtr->coupSMPtr;
  particleDataPtr = infoPtr->particleDataPtr;
  settingsPtr     = infoPtr->settingsPtr;
  rndmPtr         = infoPtr->rndmPtr;
  slhaPtr         = slhaPtrIn;
  vinComPtr       = vinComPtrIn;
  isInitPtr       = true;
  if (mesPtr != nullptr)
    mesPtr->initPtrVincia(infoPtrIn, slhaPtrIn, vinComPtrIn);
}

//--------------------------------------------------------------------------

// Initialize the matrix element.

bool ShowerMEsPlugin::initVincia() {

  // Load the plugin library if needed.
  if (name.size() == 0) return false;
  if (libPtr == nullptr) {
    if (infoPtr != nullptr) libPtr = infoPtr->plugin(name);
    else libPtr = make_shared<Plugin>(name);
    if (!libPtr->isLoaded()) return false;

    // Create a new ME.
    NewShowerMEs* newShowerMEs = (NewShowerMEs*)libPtr->symbol("newShowerMEs");
    if (!newShowerMEs) return false;
    mesPtr = newShowerMEs();
    mesPtr->initPtrVincia(infoPtr, slhaPtr, vinComPtr);
  }

  // Initialize the ME if it exists.
  if (mesPtr != nullptr) return mesPtr->initVincia();
  else return false;

}

//--------------------------------------------------------------------------

// Initialize the matrix element.

bool ShowerMEsPlugin::initDire(Info* infoPtrIn, string card) {
  infoPtr = infoPtrIn;

  // Load the plugin library if needed.
  if (name.size() == 0) return false;
  if (libPtr == nullptr) {
    if (infoPtr != nullptr) libPtr = infoPtr->plugin(name);
    else libPtr = make_shared<Plugin>(name);
    if (!libPtr->isLoaded()) return false;

    // Create a new ME.
    NewShowerMEs* newShowerMEs = (NewShowerMEs*)libPtr->symbol("newShowerMEs");
    if (!newShowerMEs) return false;
    mesPtr = newShowerMEs();
  }

  // Initialize the ME if it exists.
  if (mesPtr != nullptr) return mesPtr->initDire(infoPtr, card);
  else return false;

}

//--------------------------------------------------------------------------

// Destructor.

ShowerMEsPlugin::~ShowerMEsPlugin() {

  // Delete the MEs pointer.
  if (mesPtr == nullptr || libPtr == nullptr || !libPtr->isLoaded()) return;
  DeleteShowerMEs* deleteShowerMEs =
    (DeleteShowerMEs*)libPtr->symbol("deleteShowerMEs");
  if (deleteShowerMEs) deleteShowerMEs(mesPtr);

}

//==========================================================================

} // end namespace Pythia8
