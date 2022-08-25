// Weights.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the Weight classes.

#include "Pythia8/Info.h"
#include "Pythia8/Settings.h"
#include "Pythia8/Weights.h"
#include <limits>

namespace Pythia8 {

//==========================================================================

// WeightsBase class.

//--------------------------------------------------------------------------

// Function to return processed weights to weight container, e.g. if
// weights should be combined before proceeding.

void WeightsBase::collectWeightValues(vector<double>& outputWeights,
  double norm) {
  for (int iwt=1; iwt < getWeightsSize(); ++iwt) {
    double value = getWeightsValue(iwt)*norm;
    outputWeights.push_back(value);
  }
  return;
}

//--------------------------------------------------------------------------

// Similar function to return processed weight names.

void WeightsBase::collectWeightNames(vector<string>& outputNames) {
  for (int iwt=1; iwt < getWeightsSize(); ++iwt) {
    string name  = getWeightsName(iwt);
    outputNames.push_back(name);
  }
  return;
}

//==========================================================================

// WeightsSimpleShower class.

//--------------------------------------------------------------------------

// Reset all internal values.

void WeightsSimpleShower::clear() {
  for (size_t i=0; i < weightValues.size(); ++i) weightValues[i] = 1.;
}

//--------------------------------------------------------------------------

// Initialize shower weights.

void WeightsSimpleShower::init(bool doMerging ) {

  // Empty weight vector, relevant to avoid double init of ISR variations
  weightValues.resize(0);
  weightNames.resize(0);
  mergingVarNames.resize(0);
  // Remember the nominal weight, since this might be needed for splitting
  // enhancement handling.
  bookWeight("Baseline");

  // Force shower variations if needed by merging but not requested by user
  if (!infoPtr->settingsPtr->flag("UncertaintyBands:doVariations") &&
      infoPtr->weightContainerPtr->weightsMerging.getMuRVarFactors().size()
      && doMerging ) {
    infoPtr->settingsPtr->flag("UncertaintyBands:doVariations", true);
    // In this case, also empty requested variations, to not do default ones
    infoPtr->settingsPtr->wvec("UncertaintyBands:List",vector<string>(0));
  }

  // Assemble shower variation strings needed for merging
  if (doMerging)
    for (double fac: infoPtr->weightContainerPtr->weightsMerging.
        getMuRVarFactors()) {
          string stringfsr = "fsr:murfac=" + std::to_string(fac);
          string stringisr = "isr:murfac=" + std::to_string(fac);
          mergingVarNames.push_back({stringfsr,stringisr});
    }

}

//--------------------------------------------------------------------------

// Store the current event information.

void WeightsSimpleShower::bookVectors(vector<double> weights,
  vector<string> names) {
  replaceWhitespace(names);
  for (size_t i = 0; i < weights.size(); ++i) bookWeight(names[i], weights[i]);
}

//--------------------------------------------------------------------------

// Replace whitespace with underscore in wieght names, so that the names
// transferred to HepMC do not contain whitespace.

void WeightsSimpleShower::replaceWhitespace( vector<string>& namesIn) {
  vector<string> ret;
  for (size_t i=0; i < namesIn.size(); ++i) {
    string name=namesIn[i];
    replace(name.begin(), name.end(), ' ', '_');
    ret.push_back(name);
    namesIn[i] = name;
  }
}

//--------------------------------------------------------------------------

// Uncertainty variations initialization.

void WeightsSimpleShower::initWeightGroups(bool isISR) {
  vector<string> variationListIn = infoPtr->settingsPtr->
    wvec("UncertaintyBands:List");
  size_t vNames = weightNames.size();
  externalVariations.clear();
  externalVarNames.clear();
  externalGroupNames.clear();
  externalMap.clear();
  initialNameSave.clear();
  externalVariations.push_back("Baseline");
  initialNameSave.push_back("Baseline");
  for(vector<string>::const_iterator v=variationListIn.begin();
      v != variationListIn.end(); ++v) {
    string line = *v;
    // Remove initial blank spaces
    while (line.find(" ") == 0) line.erase( 0, 1);
    size_t pos=0;
    // Search for pdf:family keyword for SpaceShower
    if( isISR && ((pos = line.find("isr:pdf:family")) != string::npos) ) {
      size_t posEnd = line.find(" ",pos);
      if( posEnd == string::npos ) posEnd = line.size();
      for(size_t i=0; i < vNames; ++i) {
        string local = weightNames[i];
        if( local.find("isr:pdf:member") != string::npos ) {
          size_t iEqual = local.find("=")+1;
          string nMember = local.substr(iEqual,local.size());
          nMember.append(" ");
          string tmpLine = line;
          tmpLine.replace(pos,posEnd-pos,local);
          size_t iBlank = line.find_first_of(" ");
          tmpLine.replace(iBlank,1,nMember);
          externalVariations.push_back(tmpLine);
          initialNameSave.push_back(line.substr(0,line.find_first_of(" ")));
        }
      }
    } else {
      externalVariations.push_back(line);
      initialNameSave.push_back(line.substr(0,line.find_first_of(" ")));
    }
  }
  externalVariationsSize = externalVariations.size();
  size_t nNames = externalVariationsSize;
  externalVarNames.resize(nNames);
  externalVarNames[0].push_back("Baseline");
  externalGroupNames.resize(nNames);
  externalGroupNames[0]="Baseline";
  for(size_t iWeight = 0; iWeight < nNames; ++iWeight) {
    string uVarString = toLower(externalVariations[iWeight]);
    size_t firstBlank  = uVarString.find_first_of(" ");
    size_t endLine = uVarString.size();
    if( firstBlank > endLine) continue;
    externalGroupNames[iWeight] = uVarString.substr(0,firstBlank);
    uVarString  = uVarString.substr(firstBlank+1,endLine);
    size_t pos = 0;
    while ((pos = uVarString.find(" ")) != string::npos) {
      string token = uVarString.substr(0, pos);
      externalVarNames[iWeight].push_back(token);
      uVarString.erase(0, pos + 1);
    }
    if (uVarString == "" || uVarString == " ") continue;
    externalVarNames[iWeight].push_back(uVarString);
  }
  externalMap.resize(nNames);
  for(size_t iWeight = 0; iWeight < vNames; ++iWeight) {
    for(size_t iV = 0; iV < nNames; ++iV) {
      for(size_t iW = 0; iW < externalVarNames[iV].size(); ++iW) {
        if( externalVarNames[iV][iW] == weightNames[iWeight] ) {
          externalMap[iV].push_back(iWeight);
        } else if ( isISR && externalVarNames[iV][iW].find("isr:pdf:family")
        != string::npos && weightNames[iWeight].find("isr:pdf:member")
        != string::npos ) {
          externalMap[iV].push_back(iWeight);
        }
      }
    }
  }
}

//--------------------------------------------------------------------------

// Return weight group name.

string WeightsSimpleShower::getGroupName(int iGN) const {
  string tmpString("Null");
  if( iGN < 0 || iGN >= externalVariationsSize )
    return tmpString;
  return externalGroupNames[iGN];
}

//--------------------------------------------------------------------------

// Return weight group value.

double WeightsSimpleShower::getGroupWeight(int iGW) const {
  double tempWeight(1.0);
  if( iGW < 0 || iGW >= externalVariationsSize )
    return tempWeight;
  for( vector<int>::const_iterator cit = externalMap[iGW].
      begin(); cit < externalMap[iGW].end(); ++cit )
    tempWeight *= getWeightsValue(*cit);
  return tempWeight;
}

//--------------------------------------------------------------------------

// Initialise list of atomic uncertainty variations to be done by shower
// from list specified by uVars.

bool WeightsSimpleShower::initUniqueShowerVars() {

  // Reset saved list of uncertainty variations.
  uniqueShowerVars.clear();

  // Parse each string in uVars list to look for recognized keywords.
  vector<string> uVarsListIn = infoPtr->settingsPtr->
    wvec("UncertaintyBands:List");
  size_t size = uVarsListIn.size();
  for (size_t iWeight = 0; iWeight < size; ++iWeight) {
    // Convert to lowercase (to be case-insensitive). Also remove
    // extra spaces, so everything is mapped to "key=value"
    string uVarString = toLower(uVarsListIn[iWeight]);
    while (uVarString.find(" ") == 0) uVarString.erase( 0, 1);
    int iEnd = uVarString.find(" ", 0);
    uVarString.erase(0,iEnd+1);
    while (uVarString.find("=") != string::npos) {
      iEnd = uVarString.find_first_of(" ", 0);
      if( iEnd<0 ) iEnd = uVarString.length();
      string insertString = uVarString.substr(0,iEnd);
      if( uniqueShowerVars.size() == 0 ) {
        uniqueShowerVars.push_back(insertString);
      } else if ( find(uniqueShowerVars.begin(), uniqueShowerVars.end(),
          insertString) == uniqueShowerVars.end() ) {
        uniqueShowerVars.push_back(insertString);
      }
      uVarString.erase(0,iEnd+1);
    }
  }

  // Also attach weights needed for merging
  for (vector<string> mergingVariation: mergingVarNames) {
    for (string varString: mergingVariation) {
      uniqueShowerVars.push_back(varString);
    }
  }

  if (uniqueShowerVars.size() == 0) return false;
  else return true;
}

//--------------------------------------------------------------------------

// Extract those variations that match a specified list of keys.

vector<string> WeightsSimpleShower::getUniqueShowerVars(vector<string> keys) {

  // Define return vector and return immediately if nothing to do.
  vector<string> uniqueVarsNow;
  size_t sizeAll = uniqueShowerVars.size();
  if (keys.size() == 0 || sizeAll == 0) return uniqueVarsNow;

  // Parse each string in uniqueShowerVars to look for recognized keywords.
  for (string uVarString: uniqueShowerVars) {
    int firstEqual = uVarString.find_first_of("=");
    string testString = uVarString.substr(0, firstEqual);
    // does the key match an fsr one?
    if( find(keys.begin(), keys.end(), testString) != keys.end() ) {
      if( uniqueVarsNow.size() == 0 ) {
        uniqueVarsNow.push_back(uVarString);
      } else if ( find(uniqueVarsNow.begin(), uniqueVarsNow.end(), uVarString)
      == uniqueVarsNow.end() ) {
        uniqueVarsNow.push_back(uVarString);
      }
    }
  }
  return uniqueVarsNow;
}

//--------------------------------------------------------------------------

// Initialise list of enhancement factors to be applied to shower trials.

bool WeightsSimpleShower::initEnhanceFactors() {

  // Get uncertainty variations from Settings (as list of strings to parse).
  vector<string> listIn = infoPtr->settingsPtr->
    wvec("EnhancedSplittings:List");
  if (listIn.size() == 0) return false;

  string delim = "=";
  // Parse each string in list to look for recognized keywords.
  for ( auto &word : listIn ) {
    auto end = word.find(delim);
    string name = word.substr(0,end);
    string value = word.substr(end+1);
    name.erase(remove_if(name.begin(),name.end(),isspace),name.end());
    value.erase(remove_if(value.begin(),value.end(),isspace),value.end());
    double rvalue;
    istringstream iss(value);
    iss >> rvalue;
    pair<string,double> t(name,rvalue);
    enhanceFactors.insert(t);
  }
  return true;
}

//--------------------------------------------------------------------------

// Get vector of muR weight combinations (isr & fsr) needed by merging
vector<double> WeightsSimpleShower::getMuRWeightVector() {
  int nVarCombs = mergingVarNames.size();
  vector<double> ret( nVarCombs, 1. );
  for (int iVarComb = 0; iVarComb < nVarCombs; ++iVarComb) {
    int nVars = mergingVarNames[iVarComb].size();
    for (int iVar = 0; iVar < nVars; ++iVar) {
      int index = findIndexOfName(mergingVarNames[iVarComb][iVar]);
      if (index != -1) ret[iVarComb] *= getWeightsValue(index);
    }
  }
  // Insert central shower weight
  ret.insert(ret.begin(),getWeightsValue(0));
  return ret;
}

//--------------------------------------------------------------------------

// Collect shower weight names.

void WeightsSimpleShower::collectWeightNames(vector<string>& outputNames) {
  for (int iwt=1; iwt < getWeightsSize(); ++iwt)
    outputNames.push_back("AUX_" + getWeightsName(iwt));
  for (int iwtGrp = 1; iwtGrp < nWeightGroups(); ++iwtGrp)
    outputNames.push_back("AUX_" + getGroupName(iwtGrp));
}

//--------------------------------------------------------------------------

// Collect shower weight values.

void WeightsSimpleShower::collectWeightValues(vector<double>& outputWeights,
  double norm) {
  for (int iwt=1; iwt < getWeightsSize(); ++iwt)
    outputWeights.push_back(getWeightsValue(iwt)*norm);
  for (int iwtGrp = 1; iwtGrp < nWeightGroups(); ++iwtGrp)
    outputWeights.push_back(getGroupWeight(iwtGrp)*norm);
}

//==========================================================================

// WeightsLHEF class.

//--------------------------------------------------------------------------

// Reset all internal values;

void WeightsLHEF::clear() {weightValues.resize(0); weightNames.resize(0);}

//--------------------------------------------------------------------------

// Store the current event information.

void WeightsLHEF::bookVectors(vector<double> weights, vector<string> names) {
  weightValues = weights;
  // Normalize values relative to eventWeightLHEF.
  double norm = 1./infoPtr->eventWeightLHEF;
  for (double& value: weightValues) value *= norm;
  weightNames = convertNames(names);
}

//--------------------------------------------------------------------------

// Function to return processed weights to weight container.

void WeightsLHEF::collectWeightValues(vector<double>& ret, double norm) {

  // Attach the LHEF weights, starting with well-defined MUF and MUR
  // variations, and then followed by any other LHEF weight.
  for (int iwt=0; iwt < getWeightsSize(); ++iwt) {
    double value = getWeightsValue(iwt);
    string name  = getWeightsName(iwt);
    if (name.find("MUR") == string::npos || name.find("MUF") == string::npos)
      continue;
    ret.push_back(value*norm);
  }
  for (int iwt=0; iwt < getWeightsSize(); ++iwt) {
    double value = getWeightsValue(iwt);
    string name  = getWeightsName(iwt);
    if (name.find("MUR") != string::npos || name.find("MUF") != string::npos)
      continue;
    ret.push_back(value*norm);
  }

  // Done.
  return;
}

//--------------------------------------------------------------------------

// Function to return processed weight names to weight container.

void WeightsLHEF::collectWeightNames(vector<string>& ret) {

  // Attach the LHEF weights, starting with well-defined MUF and MUR
  // variations, and then followed by any other LHEF weight.
  for (int iwt = 0; iwt < getWeightsSize(); ++iwt) {
    string name = getWeightsName(iwt);
    if (name.find("MUR") == string::npos || name.find("MUF") == string::npos)
      continue;
    ret.push_back("AUX_"+name);
  }
  for (int iwt=0; iwt < getWeightsSize(); ++iwt) {
    string name  = getWeightsName(iwt);
    if (name.find("MUR") != string::npos || name.find("MUF") != string::npos)
      continue;
    ret.push_back("AUX_"+name);
  }
}


//--------------------------------------------------------------------------

// Convert weight names in MadGraph5 convention to the convention outlined
// in https://arxiv.org/pdf/1405.1067.pdf, page 162ff.

vector<string> WeightsLHEF::convertNames(vector<string> names) {
  vector<string> ret;
  for (size_t i = 0; i < names.size(); ++i) {
    string name = names[i];
    if (name=="1001") name="MUR1.0_MUF1.0";
    if (name=="1002") name="MUR1.0_MUF2.0";
    if (name=="1003") name="MUR1.0_MUF0.5";
    if (name=="1004") name="MUR2.0_MUF1.0";
    if (name=="1005") name="MUR2.0_MUF2.0";
    if (name=="1006") name="MUR2.0_MUF0.5";
    if (name=="1007") name="MUR0.5_MUF1.0";
    if (name=="1008") name="MUR0.5_MUF2.0";
    if (name=="1009") name="MUR0.5_MUF0.5";
    ret.push_back(name);
  }
  return ret;
}

//--------------------------------------------------------------------------

// Identify muR (and muF) variations in LHEF weights. This mapping is needed
// to later combine with the respective shower and merging weights.

void WeightsLHEF::identifyVariationsFromLHAinit(map<string, LHAweight>*
  weights) {
  muRvars.clear();
  int index = 0;
  for (map<string,LHAweight>::const_iterator it =
        weights->begin(); it != weights->end(); ++it) {
    string cont = it->second.contents;
    double muR = 0, muF = 0;
    // Go through all tags of one weight
    while (true) {
      // Erase leading blanks, skip irrelevant tags
      while (cont.substr(0,4) != "muR=" && cont.substr(0,4) != "muF=") {
        if (cont.find_first_of(" ") != string::npos) {
          cont = cont.substr(cont.find_first_of(" ") + 1);
        } else break;
      }
      // Parse muF and muR
      if (cont.substr(0,4) == "muR=") {
        muR = stof(cont.substr(4,cont.find_first_of(" ")));
        cont = cont.substr(cont.find_first_of(" ") + 1);
      }
      if (cont.substr(0,4) == "muF=") {
        muF = stof(cont.substr(4,cont.find_first_of(" ")));
        cont = cont.substr(cont.find_first_of(" ") + 1);
      }
      // Stop if both muF and muR set
      if (muR && muF) break;
      // Also stop if end of contents reached
      if (cont.find_first_of(" ") == string::npos) break;
      }
    // For now, only save muR values for corresponding muF=1.
    if (muF == 1.) muRvars[index] = muR;
    ++index;
  }
}

//==========================================================================

// WeightsMerging class.

//--------------------------------------------------------------------------

// Reset all internal values.

void WeightsMerging::clear() {
  for (size_t i=0; i < weightValues.size(); ++i) {
    weightValues[i] = 1.;
    weightValuesFirst[i] = 0.;
  }
  for (size_t i=0; i < weightValuesP.size(); ++i) {
    weightValuesP[i] = 1.;
    weightValuesFirstP[i] = 0.;
    weightValuesPC[i] = 1.;
    weightValuesFirstPC[i] = 0.;
  }
}

//--------------------------------------------------------------------------

// Initialize merging weights.

void WeightsMerging::init() {

  // Reset weight vectors
  weightValues.resize(0);
  weightNames.resize(0);
  weightValuesFirst.resize(0);
  weightValuesP.resize(0);
  weightValuesPC.resize(0);
  weightValuesFirstP.resize(0);
  weightValuesFirstPC.resize(0);

  // Initialization of all required variation weights done in MergingHooks.cc
  bookWeight("MUR1.0_MUF1.0", 1., 0.);
  isNLO = (infoPtr->settingsPtr->flag("Merging:doUNLOPSLoop") ||
           infoPtr->settingsPtr->flag("Merging:doUNLOPSSubtNLO") ||
           infoPtr->settingsPtr->flag("Merging:doNL3LOOP") );
}

//--------------------------------------------------------------------------

// Function to create a new, synchronized, pair of weight name and value.

void WeightsMerging::bookWeight(string name, double value, double valueFirst) {
  weightNames.push_back(name);
  weightValues.push_back(value);
  weightValuesFirst.push_back(valueFirst);
}

//--------------------------------------------------------------------------

// Store the current event information.

void WeightsMerging::bookVectors(vector<double> weights, vector<string> names){
  weightValues.resize(0);
  weightNames.resize(0);
  weightValuesFirst.resize(0);
  weightValuesP.resize(0);
  weightValuesPC.resize(0);
  weightValuesFirstP.resize(0);
  weightValuesFirstPC.resize(0);
  for (size_t i=0; i < weights.size(); ++i)
    bookWeight(names[i], weights[i], 0.);
}

//--------------------------------------------------------------------------

// Store the current event information, including first order weights for
// NLO merging.

void WeightsMerging::bookVectors(vector<double> weights,
                    vector<double> weightsFirst, vector<string> names) {
  weightValues.resize(0);
  weightNames.resize(0);
  weightValuesFirst.resize(0);
  weightValuesP.resize(0);
  weightValuesPC.resize(0);
  weightValuesFirstP.resize(0);
  weightValuesFirstPC.resize(0);
  for (size_t i=0; i < weights.size(); ++i)
    bookWeight(names[i], weights[i], weightsFirst[i]);
}

//--------------------------------------------------------------------------

// Functions to set values of weights. Does not apply to first order weights or
// scheme variation weights.

void WeightsMerging::reweightValueByIndex(int iPos, double val) {
  weightValues[iPos] *= val;
}

//--------------------------------------------------------------------------

// Reweigth merging weights by name. Does not apply to first order weights or
// scheme variation weights.

void WeightsMerging::reweightValueByName(string name, double val) {
  reweightValueByIndex(findIndexOfName(name), val);}

//--------------------------------------------------------------------------

// Functions to set values of first order weights. Does not apply to scheme
// variation weights.

void WeightsMerging::setValueFirstByIndex(int iPos, double val) {
  weightValuesFirst[iPos] = val;}

//--------------------------------------------------------------------------

// Set values of first order weights by name. Does not apply to scheme
// variation weights.

void WeightsMerging::setValueFirstByName(string name, double val) {
  setValueFirstByIndex(findIndexOfName(name), val);}

//--------------------------------------------------------------------------

// Set values as whole vector.

void WeightsMerging::setValueVector(vector<double> valueVector) {
  weightValues = valueVector;}

//--------------------------------------------------------------------------

// Set first order weight values as vector.

void WeightsMerging::setValueFirstVector(vector<double> valueFirstVector) {
  weightValuesFirst = valueFirstVector;}

//--------------------------------------------------------------------------

// Function telling merging which muR variations to perform, read from
// Merging:muRfactors setting.

vector<double> WeightsMerging::getMuRVarFactors() {
  return infoPtr->settingsPtr->pvec("Merging:muRfactors");}

//--------------------------------------------------------------------------

// Function to return processed weights to weight container.

void WeightsMerging::collectWeightValues(vector<double>& ret, double norm) {
  vector<double> showerWeights =
    infoPtr->weightContainerPtr->weightsSimpleShower.getMuRWeightVector();
  for (int iwt=1; iwt < getWeightsSize(); ++iwt) {
    double value = getWeightsValue(iwt)*norm;
    if (getWeightsValue(0) != 0 ) value /= getWeightsValue(0);
    // Combine with corresponding LHE variation
    if (isNLO) {
      // Check if corresponding muR variation is available
      if (muRVarLHEFindex.find(iwt) == muRVarLHEFindex.end()) {
        string errormsg = "Error in WeightsMerging::collectWeightValues: "
        "Requested muR variation ";
        errormsg += std::to_string(getMuRVarFactors()[iwt-1]) +
        " not found in LHE file.";
        infoPtr->errorMsg(errormsg);
      } else
        value *= infoPtr->weightContainerPtr->weightsLHEF.
          getWeightsValue(muRVarLHEFindex[iwt]);
    }
    // Combine with corresponding shower weight
    value *= showerWeights[iwt];
    ret.push_back(value);
  }

  // Include scheme variation weights if present
  if (weightValuesP.size()) {
    for (int iwt=0; iwt < getWeightsSize(); ++iwt) {
      // Normalize with UNLOPS-1 central merging weight, since that is
      // collected into the nominal weight.
      double valueP = getWeightsValueP(iwt)*norm;
      double valuePC = getWeightsValuePC(iwt)*norm;
      if (getWeightsValue(0) != 0 ) {
        valueP /= getWeightsValue(0);
        valuePC /= getWeightsValue(0);
      }

      // Combine with corresponding LHE variation
      if (isNLO) {
        // Check if corresponding muR variation is available
        if (muRVarLHEFindex.find(iwt) != muRVarLHEFindex.end()) {
          double fact = infoPtr->weightContainerPtr->weightsLHEF.
            getWeightsValue(muRVarLHEFindex[iwt]);
          valueP *= fact;
          valuePC *= fact;
        }
      }
      // Combine with corresponding shower weight
      if (iwt != 0) {
        valueP *= showerWeights[iwt-1];
        valuePC *= showerWeights[iwt-1];
      }
      ret.push_back(valueP);
      ret.push_back(valuePC);
    }
  }

  // Done.
  return;
}

//--------------------------------------------------------------------------

// Similar function to return processed weight names.

void WeightsMerging::collectWeightNames(vector<string>& outputNames) {
  for (int iwt=1; iwt < getWeightsSize(); ++iwt) {
    string name  = getWeightsName(iwt);
    outputNames.push_back(name);
  }

  // Include scheme variation names if present
  if (weightValuesP.size()) {
    for (int iwt=0; iwt < getWeightsSize(); ++iwt) {
      string nameP  = getWeightsName(iwt)+"_SCHEMEP";
      string namePC  = getWeightsName(iwt)+"_SCHEMEPC";
      outputNames.push_back(nameP);
      outputNames.push_back(namePC);
    }
  }
  return;
}

//--------------------------------------------------------------------------

// Set up mapping between LHEF variations and weights.

void WeightsMerging::setLHEFvariationMapping() {
  if (!isNLO) return;
  map<int,double> muRvarsLHEF = infoPtr->weightContainerPtr->weightsLHEF.
    muRvars;
  vector<double> muRvarsMerging = getMuRVarFactors();
  for (unsigned int iMerVar = 0; iMerVar < muRvarsMerging.size(); ++iMerVar) {
    for (pair<int,double> muRvarLHEF: muRvarsLHEF) {
      if (abs(muRvarLHEF.second - muRvarsMerging[iMerVar]) < 1e-10) {
        muRVarLHEFindex[iMerVar+1] = muRvarLHEF.first;
        continue;
      }
    }
  }
}

//==========================================================================

// The WeightContainer class.

//--------------------------------------------------------------------------

// Set nominal weight.

void WeightContainer::setWeightNominal(double weightNow) {
  weightNominal = weightNow;}

//--------------------------------------------------------------------------

// Assemble nominal weight, including nominal parton shower and merging weight.
// PS and Merging weight variations are stored relative to this return value.

double WeightContainer::collectWeightNominal() {
  return weightNominal * weightsShowerPtr->getWeightsValue(0)
                       * weightsMerging.getWeightsValue(0);
}


//--------------------------------------------------------------------------

// Functions to retrieve the stored information.

int WeightContainer::numberOfWeights() {
  // Get total number of merging weights.
  int nMergingWeights = weightsMerging.getWeightsSize() - 1;
  if (weightsMerging.weightValuesP.size())
    nMergingWeights += 2*weightsMerging.weightValuesP.size();
  // Get total number of shower weights.
  int nShowerWeights       = weightsShowerPtr->getWeightsSize() - 1;
  int nShowerWeightGroups  = weightsShowerPtr->nWeightGroups() > 0 ?
    weightsShowerPtr->nWeightGroups() - 1 : 0;
  if (doSuppressAUXweights) return 1 + nMergingWeights;
  else return (1 + weightsLHEF.getWeightsSize()
                 + nShowerWeights + nShowerWeightGroups + nMergingWeights);
}

double WeightContainer::weightValueByIndex(int key) {
  return weightValueVector()[key];
}

string WeightContainer::weightNameByIndex(int key) {
  return weightNameVector()[key];
}

//--------------------------------------------------------------------------

// Function to return the vector of weight values, combining all weights from
// all subcategories.

vector<double> WeightContainer::weightValueVector() {
  vector<double> ret;

  // The very first entry in the vector should always be the nominal weight.
  double collWgtNom = collectWeightNominal();
  ret.push_back(collWgtNom);

  // Let all weights attach the relative weight values to the return vector.
  // Second argument allows for normalization.
  if (!doSuppressAUXweights) weightsLHEF.collectWeightValues(ret,collWgtNom);
  if (!doSuppressAUXweights)
    weightsShowerPtr->collectWeightValues(ret,collWgtNom);
  weightsMerging.collectWeightValues(ret,collWgtNom);

  // Done
  return ret;

}

//--------------------------------------------------------------------------

// Function to return the vector of weight names, combining all names from
// all subcategories, cf. weightValueVector function.

vector<string> WeightContainer::weightNameVector() {
  vector<string> ret;

   // The very first entry in the vector should always be the nominal weight.
  ret.push_back("Weight");

  // Let all weights attach the weight names to the return vector.
  if (!doSuppressAUXweights) weightsLHEF.collectWeightNames(ret);
  if (!doSuppressAUXweights) weightsShowerPtr->collectWeightNames(ret);
  weightsMerging.collectWeightNames(ret);

  // Done
  return ret;

}

//--------------------------------------------------------------------------

// Reset all members to default status.

void WeightContainer::clear() {
  weightNominal = 1.;
  weightsLHEF.clear();
  if (weightsShowerPtr != nullptr) weightsShowerPtr->clear();
  weightsMerging.clear();
}

//--------------------------------------------------------------------------

// Reset total cross section estimate.

void WeightContainer::clearTotal() {
  if (sigmaTotal.size()) {
    sigmaTotal = vector<double>(sigmaTotal.size(),0.);
    errorTotal = vector<double>(errorTotal.size(),0.);
  }
}

//--------------------------------------------------------------------------

// Function to set Pointers in weight classes.

void WeightContainer::initPtrs(Info* infoPtrIn) {
  infoPtr = infoPtrIn;
  // Default pointer to shower weights = weightsSimpleShower.
  weightsShowerPtr = &weightsSimpleShower;
  weightsLHEF.setPtrs(infoPtrIn);
  weightsShowerPtr->setPtrs(infoPtrIn);
  weightsMerging.setPtrs(infoPtrIn);
}

//--------------------------------------------------------------------------

// Function to initialize weight classes.

void WeightContainer::init( bool doMerging ) {
  weightsShowerPtr->init(doMerging);
  weightsMerging.init();
  doSuppressAUXweights = infoPtr->settingsPtr->
    flag("Weights:suppressAUX");
  if (xsecIsInit) {
    sigmaSample = vector<double>(sigmaSample.size(),0.);
    errorSample = vector<double>(errorSample.size(),0.);
  }
}

//--------------------------------------------------------------------------

// Initialize accumulation of cross section.

void WeightContainer::initXsecVec() {
  if (!xsecIsInit) {
    sigmaTotal = vector<double>(weightNameVector().size(),0.);
    sigmaSample = vector<double>(weightNameVector().size(),0.);
    errorTotal = vector<double>(weightNameVector().size(),0.);
    errorSample = vector<double>(weightNameVector().size(),0.);
    xsecIsInit = true;
  }
}

//--------------------------------------------------------------------------

// Return cross section for current sample.

vector<double> WeightContainer::getSampleXsec() {return sigmaSample;}

//--------------------------------------------------------------------------

// Return cross section error for current sample.

vector<double> WeightContainer::getSampleXsecErr() {
  vector<double> ret;
  for (double error: errorSample) ret.push_back(sqrt(error));
  return ret;
}

//--------------------------------------------------------------------------

// Return cross section estimate for total run.

vector<double> WeightContainer::getTotalXsec() {return sigmaTotal;}

//--------------------------------------------------------------------------

// Return cross section error estimate for total run.
vector<double> WeightContainer::getTotalXsecErr() {
  vector<double> ret;
  for (double error: errorTotal) ret.push_back(sqrt(error));
  return ret;
}
//--------------------------------------------------------------------------

// Accumulate cross section for all weights. Provide cross section estimate
// for whole sample if lhaStrategy != 4 or -4.

void WeightContainer::accumulateXsec(double norm) {
  if (!xsecIsInit) initXsecVec();
  vector<double> weights = weightValueVector();
  for (unsigned int iWgt = 0; iWgt < weights.size(); ++iWgt) {
    sigmaTotal[iWgt] += weights[iWgt]*norm;
    sigmaSample[iWgt] += weights[iWgt]*norm;
    errorTotal[iWgt] += pow2(weights[iWgt]*norm);
    errorSample[iWgt] += pow2(weights[iWgt]*norm);
  }
}

//==========================================================================

} // end namespace Pythia8
