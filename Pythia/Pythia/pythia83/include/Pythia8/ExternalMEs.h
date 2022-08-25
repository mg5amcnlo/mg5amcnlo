// ExternalMEs.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Peter Skands, Stefan Prestel, Philip Ilten, Torbjorn
// Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the functionality to interface external matrix
// elements for matrix element corrections to parton showers.

#ifndef Pythia8_ExternalMEs_H
#define Pythia8_ExternalMEs_H

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

namespace Pythia8 {

//==========================================================================

// Base class for external matrix-element interfaces.

class ExternalMEs {

public:

  // Constructor, destructor, and assignment.
  ExternalMEs() = default;
  ~ExternalMEs() = default;

  // Initialisers for pointers.
  virtual void initPtrs(Info* infoPtrIn, SusyLesHouches* slhaPtrIn = nullptr);

  // Initialisers.
  virtual bool init() = 0;
  virtual bool initVincia() = 0;
  virtual bool initDire(Info* infoPtrIn, string card) = 0;

  // Methods to check availability of matrix elements.
  virtual bool isAvailable(vector<int> idIn, vector<int> idOut) = 0;
  virtual bool isAvailable(const Pythia8::Event& event) = 0;
  virtual bool isAvailable(const vector<Particle>& state) = 0;

  // Calculate the matrix element squared for a particle state.
  virtual double calcME2(const vector<Particle>& state) = 0;
  virtual double calcME2(const Event& event) = 0;

  // Setters.
  virtual void setColourMode(int colModeIn) {
    colMode = colModeIn;}
  virtual void setHelicityMode(int helModeIn) {
    helMode = helModeIn;}
  virtual void setIncludeSymmetryFac(bool doInclIn) {
    inclSymFac = doInclIn;}
  virtual void setIncludeHelicityAvgFac(bool doInclIn) {
    inclHelAvgFac = doInclIn;}
  virtual void setIncludeColourAvgFac(bool doInclIn) {
    inclColAvgFac = doInclIn;}

  // Getters.
  virtual int  colourMode() {return colMode;}
  virtual int  helicityMode() {return helMode;}
  virtual bool includeSymmetryFac() {return inclSymFac;}
  virtual bool includeHelicityAvgFac() {return inclHelAvgFac;}
  virtual bool includeColourAvgFac() {return inclColAvgFac;}

  // Saved list of helicity components for last ME evaluated.
  map<vector<int>, double> me2hel;

protected:

  // Fill a vector of IDs.
  void fillIds(const Event& event, vector<int>& in, vector<int>& out) const;
  // Fill a vector of momenta.
  void fillMoms(const Event& event, vector<Vec4>& p) const;
  // Fill a vector of colors.
  void fillCols(const Event& event, vector<int>& colors) const;
  // Return the momenta.
  vector<vector<double> > fillMoms(const Event& event) const;

  // Colour mode (0: strict LC, 1: LC, 2: LC sum, 3: FC).
  int colMode{1};

  // Helicity mode (0: explicit helicity sum, 1: implicit helicity sum).
  int helMode{1};

  // Symmetry and averaging factors.
  bool inclSymFac{false}, inclHelAvgFac{false}, inclColAvgFac{false};

  // Pointers to VINCIA and Pythia 8 objects.
  Info*           infoPtr{};
  CoupSM*         coupSMPtr{};
  ParticleData*   particleDataPtr{};
  Rndm*           rndmPtr{};
  Settings*       settingsPtr{};
  SusyLesHouches* slhaPtr{};

  // Is initialized.
  bool isInitPtr{false}, isInit{false};

};

//==========================================================================

// Interface to external matrix elements.

class ExternalMEsPlugin : public ExternalMEs {

public:

  // Constructor and destructor.
  ExternalMEsPlugin(const ExternalMEsPlugin &me) : ExternalMEs(),
    mesPtr(nullptr), libPtr(nullptr), name(me.name) {}
  ExternalMEsPlugin(string nameIn = "") : ExternalMEs(), mesPtr(nullptr),
    libPtr(nullptr), name(nameIn) {}
  ~ExternalMEsPlugin();
  ExternalMEsPlugin &operator=(const ExternalMEsPlugin &me) {
    mesPtr = nullptr; libPtr = nullptr; name = me.name; return *this;}

  // Initialisers for pointers.
  void initPtrs(Info* infoPtrIn, SusyLesHouches* slhaPtrIn = nullptr) override;

  // Initialisers.
  bool init() override;
  bool initVincia() override;
  bool initDire(Info* infoPtrIn, string card) override;

  // Methods to check availability of matrix elements.
  virtual bool isAvailable(vector<int> idIn, vector<int> idOut) override {
    return mesPtr != nullptr ? mesPtr->isAvailable(idIn, idOut) : false;}
  virtual bool isAvailable(const Pythia8::Event& event) override {
    return mesPtr != nullptr ? mesPtr->isAvailable(event) : false;}
  virtual bool isAvailable(const vector<Particle>& state) override {
    return mesPtr != nullptr ? mesPtr->isAvailable(state) : false;}

  // Calculate the matrix element squared for a particle state.
  double calcME2(const vector<Particle>& state) override {
    return mesPtr != nullptr ? mesPtr->calcME2(state) : -1;}
  virtual double calcME2(const Event& event) override {
    return mesPtr != nullptr ? mesPtr->calcME2(event) : -1;}

  // Get previously calculated results.
  map<vector<int>, double> getHelicityAmplitudes() {
    return mesPtr != nullptr ? mesPtr-> me2hel : map<vector<int>, double>();}

  // Setters.
  void setColourMode(int colModeIn) override {
    if (mesPtr != nullptr) mesPtr->setColourMode(colModeIn);}
  void setHelicityMode(int helModeIn) override {
    if (mesPtr != nullptr) mesPtr->setHelicityMode(helModeIn);}
  void setIncludeSymmetryFac(bool doInclIn) override {
    if (mesPtr != nullptr) mesPtr->setIncludeSymmetryFac(doInclIn);}
  void setIncludeHelicityAvgFac(bool doInclIn) override {
    if (mesPtr != nullptr)
      mesPtr->setIncludeHelicityAvgFac(doInclIn);}
  void setIncludeColourAvgFac(bool doInclIn) override {
    if (mesPtr != nullptr) mesPtr->setIncludeColourAvgFac(doInclIn);}

  // Getters.
  int colourMode() override {
    return mesPtr != nullptr ? mesPtr->colourMode() : 0;}
  int helicityMode() override {
    return mesPtr != nullptr ? mesPtr->helicityMode() : 0;}
  bool includeSymmetryFac() override {
    return mesPtr != nullptr ? mesPtr->includeSymmetryFac() : false;}
  bool includeHelicityAvgFac() override {
    return mesPtr != nullptr ? mesPtr->includeHelicityAvgFac() : false;}
  bool includeColourAvgFac() override {
    return mesPtr != nullptr ? mesPtr->includeColourAvgFac() : false;}


private:

  // Typedefs of the hooks used to access the plugin.
  typedef ExternalMEs* NewExternalMEs();
  typedef void DeleteExternalMEs(ExternalMEs*);

  // The loaded MEs object, plugin library, and plugin name.
  ExternalMEs *mesPtr;
  PluginPtr  libPtr;
  string     name;

};

//==========================================================================

// A helicity sampler using external matrix elements.

class HelicitySampler {

 public:
  // Constructor, destructor, and assignment.
  HelicitySampler() {isInitPtr = false;}
  ~HelicitySampler() = default;

  // Initialise pointers to required Pythia objects.
  void initPtrs(ExternalMEsPlugin* mePluginPtrIn, Rndm* rndmPtrIn) {
    mePluginPtr = mePluginPtrIn;
    rndmPtr     = rndmPtrIn;
    isInitPtr   = true;}

  // Set helicities for a particle state.
  bool selectHelicities(vector<Particle>& state, bool force);

 private:

  // Pointers to Pythia objects.
  ExternalMEsPlugin* mePluginPtr;
  Rndm* rndmPtr;

  // Flag whether we have all pointers.
  bool isInitPtr;

};

//==========================================================================

} // end namespace Pythia8

#endif // end Pythia8_ExternalMEs_H
