// VinciaWeights.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains header information for the VinciaWeights class.

#ifndef Vincia_VinciaWeights_H
#define Vincia_VinciaWeights_H

#include "Pythia8/Info.h"
#include "Pythia8/Settings.h"
#include "Pythia8/VinciaCommon.h"
#include "Pythia8/Weights.h"

namespace Pythia8 {

//==========================================================================

// Class for storing Vincia weights.

class VinciaWeights : public WeightsShower {

public:

  // Friends for internal private members.
  friend class VinciaFSR;
  friend class VinciaISR;

  // Initilize pointers.
  bool initPtr(Info* infoPtrIn, VinciaCommon* vinComPtrIn);

  // Initialize.
  void init( bool doMerging ) override;

  // Reset all internal values;
  void clear() override {
    for (size_t i=0; i < weightValues.size(); ++i) weightValues[i] = 1.;}

  // Access the weight labels.
  string weightLabel(int iWeightIn = 0) {
    return iWeightIn == 0 ? "Vincia" : varLabels[iWeightIn-1];}

  // Scale the uncertainty band weights.
  void scaleWeightVar(vector<double> pAccept, bool accept, bool isHard);

  // Scale the uncertainty band weights if branching is accepted.
  void scaleWeightVarAccept(vector<double> pAccept);

  // Scale the uncertainty band weights if branching is rejected.
  void scaleWeightVarReject(vector<double> pAccept);

  // Enhanced kernels: reweight if branching is accepted.
  void scaleWeightEnhanceAccept(double enhanceFac = 1.);

  // Enhanced kernels: reweight if branching is rejected.
  void scaleWeightEnhanceReject(double pAcceptUnenhanced,
    double enhanceFac = 1.);

  // Helper function for keyword evaluation.
  int doVarNow(string keyIn, enum AntFunType antFunTypePhys, bool isFSR) ;

  // Helper function for antenna function.
  double ant(double antIn, double cNSIn) {return (antIn+cNSIn);}

private:

  // Verbosity.
  int verbose{0};

  // Pointers.
  Settings*     settingsPtr{};
  VinciaCommon* vinComPtr{};

  // Internal flag.
  bool isInitPtr{false};

  // Constants.
  static const double TINYANT, PACCEPTVARMAX, MINVARWEIGHT;

  // Parameters taken from settings.
  bool uncertaintyBands{false};
  vector<string> varLabels;
  vector<vector<string> > varKeys;
  vector<vector<double> > varVals;

  // Helper parameters.
  vector<string> allKeywords;
  map<enum AntFunType, string> antFunTypeToKeyFSR, antFunTypeToKeyISR;
  bool doMerging{false}, doAlphaSvar{false}, doFiniteVar{false};

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_VinciaWeights_H
