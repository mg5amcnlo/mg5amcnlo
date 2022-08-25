// VinciaWeights.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the VinciaWeights class.

#include "Pythia8/VinciaWeights.h"

namespace Pythia8 {

using namespace VinciaConstants;

//==========================================================================

// The VinciaWeights class.

//--------------------------------------------------------------------------

// Constants.

const double VinciaWeights::PACCEPTVARMAX = 0.99;
const double VinciaWeights::MINVARWEIGHT  = 0.01;

//--------------------------------------------------------------------------

// Initilize pointers.

bool VinciaWeights::initPtr(Info* infoPtrIn,
  VinciaCommon* vinComPtrIn) {
  infoPtr     = infoPtrIn;
  settingsPtr = infoPtr->settingsPtr;
  vinComPtr   = vinComPtrIn;
  isInitPtr   = true;
  return true;
}

//--------------------------------------------------------------------------

// Initialize.

void VinciaWeights::init( bool doMergingIn ) {

  // Check initPtr.
  if (!isInitPtr) {
    printOut("VinciaWeights::init","Error! pointers not initialized");
    return;
  }
  verbose          = settingsPtr->mode("Vincia:verbose");
  uncertaintyBands = settingsPtr->flag("UncertaintyBands:doVariations");
  varLabels.resize(0);
  bookWeight("Vincia", 1.);

  // Merging on/off ?
  doMerging = doMergingIn;

  // List of all keywords recognised by Vincia.
  allKeywords.clear(); allKeywords.resize(0);
  string ffKeys[3] = { ":", ":emit:", ":split:" };
  string iiKeys[4] = { ":", ":emit:", ":split:", ":conv:" };
  for (int i = 0; i < 3; i++) allKeywords.push_back("fsr"+ffKeys[i]+"murfac");
  for (int i = 0; i < 3; i++) allKeywords.push_back("fsr"+ffKeys[i]+"cns");
  for (int i = 0; i < 4; i++) allKeywords.push_back("isr"+iiKeys[i]+"murfac");
  for (int i = 0; i < 4; i++) allKeywords.push_back("isr"+iiKeys[i]+"cns");

  // Convert antFunTypePhys to keyword.
  antFunTypeToKeyFSR.clear();
  antFunTypeToKeyFSR[QQemitFF]  = "emit";
  antFunTypeToKeyFSR[QGemitFF]  = "emit";
  antFunTypeToKeyFSR[GQemitFF]  = "emit";
  antFunTypeToKeyFSR[GGemitFF]  = "emit";
  antFunTypeToKeyFSR[GXsplitFF] = "split";
  antFunTypeToKeyFSR[QQemitRF]  = "emit";
  antFunTypeToKeyFSR[QGemitRF]  = "emit";
  antFunTypeToKeyFSR[XGsplitRF] = "split";
  antFunTypeToKeyISR.clear();
  antFunTypeToKeyISR[QQemitII]  = "emit";
  antFunTypeToKeyISR[GQemitII]  = "emit";
  antFunTypeToKeyISR[GGemitII]  = "emit";
  antFunTypeToKeyISR[QXsplitII] = "split";
  antFunTypeToKeyISR[GXconvII]  = "conv";
  antFunTypeToKeyISR[QQemitIF]  = "emit";
  antFunTypeToKeyISR[QGemitIF]  = "emit";
  antFunTypeToKeyISR[GQemitIF]  = "emit";
  antFunTypeToKeyISR[GGemitIF]  = "emit";
  antFunTypeToKeyISR[QXsplitIF] = "split";
  antFunTypeToKeyISR[GXconvIF]  = "conv";
  antFunTypeToKeyISR[XGsplitIF] = "split";

  // Clean up the names of requested variations.
  for (int i = 0; i < (int)varLabels.size(); i++) {
    int strBegin = varLabels[i].find_first_not_of(" ");
    if ((i == 0) && (varLabels[i].find("{") != string::npos)) strBegin += 1;
    int strEnd   = varLabels[i].find_last_not_of(" ");
    if ((i == (int)varLabels.size()-1) && (varLabels[i].find("}") !=
        string::npos)) strEnd -= 1;
    int strRange = strEnd - strBegin + 1;
    varLabels[i] = toLower(varLabels[i].substr(strBegin, strRange));
  }

  // Parse names of requested variations and check for keywords.
  varKeys.clear(); varKeys.resize(varLabels.size());
  varVals.clear(); varVals.resize(varLabels.size());
  for (int i = 0; i < (int)varLabels.size(); i++) {
    varKeys[i].clear(); varKeys[i].resize(0);
    varVals[i].clear(); varVals[i].resize(0);
    string var      = varLabels[i];
    int    iEndName = var.find(" ", 0);
    string varName  = var.substr(0, iEndName);
    string left     = var.substr(iEndName, var.length());
    left     = left.substr(left.find_first_not_of(" "), left.length());
    varLabels[i] = varName;
    while (left.length() > 1) {
      int    iEnd = left.find("=", 0);
      int    iAlt = left.find(" ", 0);
      if ( (iAlt < iEnd) && (iAlt > 0) ) iEnd = iAlt;
      string key  = left.substr(0, iEnd);
      if (1+key.find_last_not_of(" ") < key.length())
        key = key.substr(0, 1+key.find_last_not_of(" "));
      varKeys[i].push_back(key);
      string val  = left.substr(iEnd+1, left.length());
      val  = val.substr(val.find_first_not_of(" "), val.length());
      val  = val.substr(0, val.find_first_of(" "));
      varVals[i].push_back(atof(val.c_str()));
      left = left.substr(iEnd+1, left.length());
      if (left.find_first_of(" ") >= left.length()) break;
      left = left.substr(left.find_first_of(" "), left.length());
      if (left.find_first_not_of(" ") >= left.length()) break;
      left = left.substr(left.find_first_not_of(" "), left.length());
    }
  }

  if (uncertaintyBands && (verbose >= REPORT)) {
    printOut("VinciaWeights", "List of variations, keywords and values:");
    for (int i = 0; i < (int)varLabels.size(); i++) {
      cout << "  " << varLabels[i] << " : ";
      for (int j=0; j<(int)varVals[i].size(); j++) {
        cout << " " << varKeys[i][j] << " -> " << varVals[i][j];
        if (j < (int)varVals[i].size() - 1) cout << ",";
      }
      cout << endl;
    }
  }

  // Safety check for non-sensible input.
  if (uncertaintyBands) {
    for (int i = 0; i < (int)varKeys.size(); i++) {
      for (int j = 0; j < (int)varKeys[i].size(); j++) {
        // Check input keywords against known ones.
        bool foundValidKey = false;
        for (int k = 0; k < (int)allKeywords.size(); k++) {
          if (allKeywords[k] == varKeys[i][j]) {
            foundValidKey = true;
            // If a weight with this name does not yet exist, book one.
            if (findIndexOfName(varKeys[i][j]) < 0)
              bookWeight(varKeys[i][j],1.);
            break;
          }
        }
        if (!foundValidKey)
          printOut("VinciaWeights", "Error! Unknown key " +
            varKeys[i][j] + " found, please check!");
      }
    }
  }

}

//--------------------------------------------------------------------------

// Scaling functions. They determine what needs to be done, to which
// weight(s), and then call the relevant base class method:
// reweightValueByIndex.

void VinciaWeights::scaleWeightVar(vector<double> pAccept, bool accept,
  bool isHard) {
  if (!uncertaintyBands) return;
  if (getWeightsSize() <= 1) return;
  // Variations only pertain to hard process and resonance decays.
  if (!isHard) return;
  if (accept) scaleWeightVarAccept(pAccept);
  else scaleWeightVarReject(pAccept);
}

void VinciaWeights::scaleWeightVarAccept(vector<double> pAccept) {
  for (int iWeight = 1; iWeight<getWeightsSize(); iWeight++) {
    double pAcceptVar = pAccept[iWeight];
    if (pAcceptVar > PACCEPTVARMAX) pAcceptVar = PACCEPTVARMAX;
    reweightValueByIndex( iWeight, pAcceptVar/pAccept[0] );
  }
}

void VinciaWeights::scaleWeightVarReject(vector<double> pAccept) {
  for (int iWeight = 1; iWeight<getWeightsSize(); iWeight++) {
    double pAcceptVar = pAccept[iWeight];
    if (pAcceptVar > PACCEPTVARMAX) pAcceptVar = PACCEPTVARMAX;
    double reWeight = (1.0-pAcceptVar)/(1.0-pAccept[0]);
    if (reWeight < MINVARWEIGHT) reWeight = MINVARWEIGHT;
    reweightValueByIndex( iWeight, reWeight );
  }
}

void VinciaWeights::scaleWeightEnhanceAccept(double enhanceFac) {
  if (enhanceFac == 1.0) return;
  else reweightValueByIndex( 0, 1./enhanceFac);
}

void VinciaWeights::scaleWeightEnhanceReject(double pAcceptUnenhanced,
  double enhanceFac) {
  if (enhanceFac == 1.0) return;
  if (enhanceFac > 1.0) {
    double rRej =
      (1. - pAcceptUnenhanced/enhanceFac)/(1. - pAcceptUnenhanced);
    reweightValueByIndex( 0, rRej );
  } else {
    double rRej =
      (1. - pAcceptUnenhanced)/(1. - enhanceFac*pAcceptUnenhanced);
    reweightValueByIndex( 0, rRej );
  }
}

//--------------------------------------------------------------------------

// Helper function for keyword evaluation

int VinciaWeights::doVarNow(string keyIn, enum AntFunType antFunTypePhys,
  bool isFSR) {

  // Check variation for all branching types.
  string asKey = ":murfac";
  string nsKey = ":cns";
  string type = (isFSR ? "fsr" : "isr");
  if (type + asKey == keyIn) return 1;
  if (type + nsKey == keyIn) return 2;
  // Check variation for specific branching type.
  map<enum AntFunType, string> keyCvt = (isFSR ? antFunTypeToKeyFSR :
    antFunTypeToKeyISR);
  if (type + ":" + keyCvt[antFunTypePhys] + asKey == keyIn) return 1;
  if (type + ":" + keyCvt[antFunTypePhys] + nsKey == keyIn) return 2;
  return -1;

}

//==========================================================================

} // end namespace Pythia8
