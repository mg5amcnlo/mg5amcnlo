// ShowerMEs.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Stefan Prestel, Philip Ilten, Torbjorn
// Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the functionality to interface external matrix
// elements for matrix element corrections to parton showers.

#ifndef Pythia8_ShowerMEs_H
#define Pythia8_ShowerMEs_H

// Include Pythia headers.
#include "Pythia8/Basics.h"
#include "Pythia8/PythiaComplex.h"
#include "Pythia8/Event.h"
#include "Pythia8/Info.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StandardModel.h"
#include "Pythia8/SusyLesHouches.h"

// Include Vincia.
#include "Pythia8/VinciaCommon.h"

namespace Pythia8 {

//==========================================================================

// The parton shower matrix-element interface.

class ShowerMEs {

public:

  // Constructor, destructor, and assignment.
  ShowerMEs() = default;
  ~ShowerMEs() = default;

  // VINCIA methods.
  // Set pointers to required PYTHIA 8 objects.
  virtual void initPtrVincia(Info* infoPtrIn, SusyLesHouches* slhaPtrIn,
    VinciaCommon* vinComPtrIn);
  // Initialise the MG5 model, parameters, and couplings.
  virtual bool initVincia() = 0;
  // Check if the process is available.
  virtual bool hasProcessVincia
  (vector<int> idIn, vector<int> idOut, set<int> sChan) = 0;
  // Get the matrix element squared for a particle state.
  virtual double me2Vincia(vector<Particle> state, int nIn) = 0;
  // Use me2 to set helicities for a state. Takes a reference as
  // input and operates on it.
  virtual bool selectHelicitiesVincia(vector<Particle>& state, int nIn,
    bool force);
  // Set and get colour depth.
  virtual void setColourDepthVincia(int colourDepthIn) {
    colourDepth = colourDepthIn;}
  virtual int getColourDepthVincia() {return colourDepth;}
  // Convert a process label to a string, e.g. for printing to stdout.
  string makeLabelVincia
  (vector<int>& id, int nIn, bool convertToNames = false) const;
  // Set verbosity level.
  virtual void setVerboseVincia(int verboseIn) {verbose = verboseIn;}

  // Dire methods.
  virtual bool initDire(Info* infoPtrIn, string card) = 0;
  virtual bool isAvailableMEDire(vector <int> in, vector<int> out) = 0;
  virtual bool isAvailableMEDire(const Pythia8::Event& event) = 0;
  virtual double calcMEDire(const Pythia8::Event& event) = 0;

  // Common methods.
  // Fill a vector of IDs.
  void fillIds(const Event& event, vector<int>& in, vector<int>& out) const;
  // Fill a vector of momenta.
  void fillMoms(const Event& event, vector<Vec4>& p) const;
  // Fill a vector of colors.
  void fillCols(const Event& event, vector<int>& colors) const;
  // Return the momenta.
  vector<vector<double> > fillMoms(const Event& event) const;

protected:

  // Is initialized.
  bool isInitPtr{false}, isInit{false};

  // Saved list of helicity components for last ME evaluated.
  map< vector<int> , double > me2hel{};

  // Pointers to VINCIA and Pythia 8 objects.
  Info*           infoPtr{};
  CoupSM*         coupSMPtr{};
  ParticleData*   particleDataPtr{};
  Rndm*           rndmPtr{};
  Settings*       settingsPtr{};
  VinciaCommon*   vinComPtr{};
  SusyLesHouches* slhaPtr{};

  // Colour mode (0: leading colour, 1: Vincia colour).
  int colourDepth{0};

  // Verbosity level.
  int verbose{0};

};

//==========================================================================

// Interface to external matrix elements for parton shower matrix
// element corrections.

class ShowerMEsPlugin : public ShowerMEs {

public:

  // Constructor and destructor.
  ShowerMEsPlugin(const ShowerMEsPlugin &me) : ShowerMEs(), mesPtr(nullptr),
    libPtr(nullptr), name(me.name) {};
  ShowerMEsPlugin(string nameIn = "") : ShowerMEs(), mesPtr(nullptr),
    libPtr(nullptr), name(nameIn) {};
  ~ShowerMEsPlugin();
  ShowerMEsPlugin &operator=(const ShowerMEsPlugin &me) {
    mesPtr = nullptr; libPtr = nullptr; name = me.name; return *this;}

  // VINCIA methods.
  // Set pointers to required PYTHIA 8 objects.
  void initPtrVincia(Info* infoPtrIn, SusyLesHouches* slhaPtrIn,
    VinciaCommon* vinComPtrIn) override;
  // Initialise the MG5 model, parameters, and couplings.
  bool initVincia() override;
  // Get the matrix element squared for a particle state.
  double me2Vincia(vector<Particle> state, int nIn) override {
    return mesPtr != nullptr ? mesPtr->me2Vincia(state, nIn) : -1;}
  // Check if the process is available.
  bool hasProcessVincia(vector<int> idIn, vector<int> idOut,
    set<int> sChan) override {return mesPtr != nullptr ?
      mesPtr->hasProcessVincia(idIn, idOut, sChan) : false;}
  // Use me2 to set helicities for a state. Takes a reference as
  // input and operates on it.
  bool selectHelicitiesVincia(vector<Particle>& state, int nIn,
    bool force) override {
    return mesPtr != nullptr ?
      mesPtr->selectHelicitiesVincia(state, nIn, force) : false;}
  // Set and get colour depth.
  void setColourDepthVincia(int colourDepthIn) override {
    if (mesPtr != nullptr) mesPtr->setColourDepthVincia(colourDepthIn);}
  int getColourDepthVincia() override {
    return mesPtr != nullptr ? mesPtr->getColourDepthVincia() : 0;}
  // Set verbosity level.
  void setVerboseVincia(int verboseIn) override {
    if (mesPtr != nullptr) mesPtr->setVerboseVincia(verboseIn);}

  // Dire methods.
  bool initDire(Info* infoPtrIn, string card) override;
  bool isAvailableMEDire(vector <int> in, vector<int> out) override {
    return mesPtr != nullptr ? mesPtr->isAvailableMEDire(in, out) : false;}
  bool isAvailableMEDire(const Pythia8::Event& event) override {
    return mesPtr != nullptr ? mesPtr->isAvailableMEDire(event) : false;}
  double calcMEDire(const Pythia8::Event& event) override {
    return mesPtr != nullptr ? mesPtr->calcMEDire(event) : 0;}

private:

  // Typedefs of the hooks used to access the plugin.
  typedef ShowerMEs* NewShowerMEs();
  typedef void DeleteShowerMEs(ShowerMEs*);

  // The loaded MEs object, plugin library, and plugin name.
  ShowerMEs *mesPtr;
  PluginPtr  libPtr;
  string     name;

};

//==========================================================================

} // end namespace Pythia8

#endif // end Pythia8_ShowerMEs_H
