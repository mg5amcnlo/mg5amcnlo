// DireWeightContainer.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the
// DireWeightContainer class.

#include "Pythia8/DireWeightContainer.h"
#include "Pythia8/DireSpace.h"
#include "Pythia8/DireTimes.h"

namespace Pythia8 {

//==========================================================================

const double DireWeightContainer::LARGEWT = 2e0;

// Container for all shower weights, including handling.

void DireWeightContainer::setup() {

  // Clear everything.
  init();
  enhanceFactors.clear();

  // Initialize MG5 MEs interface.
  card            = settingsPtr->word("Dire:MG5card");
  string mePlugin = settingsPtr->word("Dire:MEplugin");
  if (mePlugin.size() > 0) {
    if (!hasMEs) {
      matrixElements = ShowerMEsPlugin("libpythia8mg5" + mePlugin + ".so");
    }
    hasMEs = matrixElements.initDire(infoPtr,card);
  }

  // Initialize additional user-defined enhancements of splitting kernel
  // overestimates.
  //int sizeNames = 48;
  int sizeNames = 100;
  const char* names[] = {
    // QCD FSR
    "Dire_fsr_qcd_1->1&21",
    "Dire_fsr_qcd_1->21&1",
    "Dire_fsr_qcd_21->21&21a",
    "Dire_fsr_qcd_21->21&21b",
    "Dire_fsr_qcd_21->1&1a",
    "Dire_fsr_qcd_21->1&1b",
    "Dire_fsr_qcd_1->2&1&2",
    "Dire_fsr_qcd_1->1&1&1",
    "Dire_fsr_qcd_1->1&21_notPartial",
    "Dire_fsr_qcd_21->21&21_notPartial",
    "Dire_fsr_qcd_21->1&1_notPartial",
    "Dire_fsr_qcd_1->1&21&21",
    "Dire_fsr_qcd_1->1&d&dbar",
    "Dire_fsr_qcd_1->1&dbar&d",
    "Dire_fsr_qcd_1->1&u&ubar",
    "Dire_fsr_qcd_1->1&ubar&u",
    "Dire_fsr_qcd_1->1&s&sbar",
    "Dire_fsr_qcd_1->1&sbar&s",
    "Dire_fsr_qcd_1->1&c&cbar",
    "Dire_fsr_qcd_1->1&cbar&c",
    "Dire_fsr_qcd_1->1&b&bbar",
    "Dire_fsr_qcd_1->1&bbar&b",
    "Dire_fsr_qcd_21->21&21&21",
    "Dire_fsr_qcd_21->21&d&dbar",
    "Dire_fsr_qcd_21->21&dbar&d",
    "Dire_fsr_qcd_21->21&u&ubar",
    "Dire_fsr_qcd_21->21&ubar&u",
    "Dire_fsr_qcd_21->21&s&sbar",
    "Dire_fsr_qcd_21->21&sbar&s",
    "Dire_fsr_qcd_21->21&c&cbar",
    "Dire_fsr_qcd_21->21&cbar&c",
    "Dire_fsr_qcd_21->21&b&bbar",
    "Dire_fsr_qcd_21->21&bbar&b",
    // QCD ISR
    "Dire_isr_qcd_1->1&21",
    "Dire_isr_qcd_21->1&1",
    "Dire_isr_qcd_21->21&21a",
    "Dire_isr_qcd_21->21&21b",
    "Dire_isr_qcd_1->21&1",
    "Dire_isr_qcd_1->2&1&2",
    "Dire_isr_qcd_1->1&1&1",
    // QED FSR
    "Dire_fsr_qed_1->1&22",
    "Dire_fsr_qed_1->22&1",
    "Dire_fsr_qed_11->11&22",
    "Dire_fsr_qed_11->22&11",
    "Dire_fsr_qed_22->1&1a",
    "Dire_fsr_qed_22->1&1b",
    "Dire_fsr_qed_22->2&2a",
    "Dire_fsr_qed_22->2&2b",
    "Dire_fsr_qed_22->3&3a",
    "Dire_fsr_qed_22->3&3b",
    "Dire_fsr_qed_22->4&4a",
    "Dire_fsr_qed_22->4&4b",
    "Dire_fsr_qed_22->5&5a",
    "Dire_fsr_qed_22->5&5b",
    "Dire_fsr_qed_22->11&11a",
    "Dire_fsr_qed_22->11&11b",
    "Dire_fsr_qed_22->13&13a",
    "Dire_fsr_qed_22->13&13b",
    "Dire_fsr_qed_22->15&15a",
    "Dire_fsr_qed_22->15&15b",
    // QED ISR
    "Dire_isr_qed_1->1&22",
    "Dire_isr_qed_1->22&1",
    "Dire_isr_qed_22->1&1",
    "Dire_isr_qed_11->11&22",
    "Dire_isr_qed_11->22&11",
    "Dire_isr_qed_22->11&11",
    // EW FSR
    "Dire_fsr_ew_1->1&23",
    "Dire_fsr_ew_1->23&1",
    "Dire_fsr_ew_23->1&1a",
    "Dire_fsr_ew_23->1&1b",
    "Dire_fsr_ew_24->1&1a",
    "Dire_fsr_ew_24->1&1b",
    // New U(1) FSR
    "Dire_fsr_u1new_1->1&22",
    "Dire_fsr_u1new_1->22&1",
    "Dire_fsr_u1new_11->11&22",
    "Dire_fsr_u1new_11->22&11",
    "Dire_fsr_u1new_22->1&1a",
    "Dire_fsr_u1new_22->1&1b",
    "Dire_fsr_u1new_22->2&2a",
    "Dire_fsr_u1new_22->2&2b",
    "Dire_fsr_u1new_22->3&3a",
    "Dire_fsr_u1new_22->3&3b",
    "Dire_fsr_u1new_22->4&4a",
    "Dire_fsr_u1new_22->4&4b",
    "Dire_fsr_u1new_22->5&5a",
    "Dire_fsr_u1new_22->5&5b",
    "Dire_fsr_u1new_22->11&11a",
    "Dire_fsr_u1new_22->11&11b",
    "Dire_fsr_u1new_22->13&13a",
    "Dire_fsr_u1new_22->13&13b",
    "Dire_fsr_u1new_22->15&15a",
    "Dire_fsr_u1new_22->15&15b",
    "Dire_fsr_u1new_22->211&211a",
    "Dire_fsr_u1new_22->211&211b",
    // New U(1) ISR
    "Dire_isr_u1new_1->1&22",
    "Dire_isr_u1new_1->22&1",
    "Dire_isr_u1new_22->1&1",
    "Dire_isr_u1new_11->11&22",
    "Dire_isr_u1new_11->22&11",
    "Dire_isr_u1new_22->11&11"
  };

  for (int i=0; i < sizeNames; ++i) {
    if (settingsPtr->parm("Enhance:"+string(names[i])) > 1.0) {
      enhanceFactors.insert(
        make_pair( string(names[i]),
        settingsPtr->parm("Enhance:"+string(names[i]))) );
    }
  }

  string vkey = "base";
  rejectWeight.insert( make_pair(vkey, map<ulong, DirePSWeight>() ));
  acceptWeight.insert( make_pair(vkey, map<ulong, DirePSWeight>() ));
  showerWeight.insert( make_pair(vkey, 1.) );
  weightNames.push_back( vkey );

  bool doVar = settingsPtr->flag("Variations:doVariations");
  if ( doVar ) {
    vector<string> group;
    // Down-variations of renormalization scale.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.) {
      bookWeightVar("Variations:muRisrDown");
      group.push_back("Variations:muRisrDown");
    }
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.) {
      bookWeightVar("Variations:muRfsrDown");
      group.push_back("Variations:muRfsrDown");
    }
    if (int(group.size()) > 0) {
      weightCombineList.insert ( make_pair("scaleDown", group) );
      weightCombineListNames.push_back ("scaleDown");
    }

    group.resize(0);
    // Up-variations of renormalization scale.
    if (settingsPtr->parm("Variations:muRisrUp") != 1.) {
      bookWeightVar("Variations:muRisrUp");
      group.push_back("Variations:muRisrUp");
    }
    if (settingsPtr->parm("Variations:muRfsrUp") != 1.) {
      bookWeightVar("Variations:muRfsrUp");
      group.push_back("Variations:muRfsrUp");
    }
    if (int(group.size()) > 0) {
      weightCombineList.insert ( make_pair("scaleUp", group) );
      weightCombineListNames.push_back ("scaleUp");
    }

    // PDF variations.
    group.resize(0);
    if (settingsPtr->flag("Variations:PDFup")) {
      bookWeightVar("Variations:PDFup",false);
      group.push_back("Variations:PDFup");
     weightCombineList.insert ( make_pair("PDFup", group) );
     weightCombineListNames.push_back ("PDFup");
    }
    group.resize(0);
    if (settingsPtr->flag("Variations:PDFdown")) {
      bookWeightVar("Variations:PDFdown",false);
      group.push_back("Variations:PDFdown");
      weightCombineList.insert ( make_pair("PDFdown", group) );
      weightCombineListNames.push_back ("PDFdown");
    }

    // ME variations.
    if (settingsPtr->parm("Variations:muRmeUp") != 1.)
      bookWeightVar("Variations:muRmeUp");
    if (settingsPtr->parm("Variations:muRmeDown") != 1.)
      bookWeightVar("Variations:muRmeDown");
    if (settingsPtr->parm("Variations:muFmeUp") != 1.)
      bookWeightVar("Variations:muFmeUp");
    if (settingsPtr->parm("Variations:muFmeDown") != 1.)
      bookWeightVar("Variations:muFmeDown");

  }

  // Remember groups of weights that should be combined into one weight.
  //vector<string> group;
  //group = createvector<string>("Variations:muRfsrUp")
  //  ("Variations:muRisrUp");
  //weightCombineList.insert ( make_pair("scaleUp", group) );
  //weightCombineListNames.push_back ("scaleUp");
  //group = createvector<string>("Variations:muRfsrDown")
  //  ("Variations:muRisrDown");
  //weightCombineList.insert ( make_pair("scaleDown", group) );
  //weightCombineListNames.push_back ("scaleDown");

}

//--------------------------------------------------------------------------

void DireWeightContainer::bookWeightVar( string vkey, bool checkSettings) {
  bool insert = !checkSettings || settingsPtr->parm(vkey) != 1.0;
  if (insert) {
    rejectWeight.insert( make_pair(vkey, map<ulong, DirePSWeight>() ));
    acceptWeight.insert( make_pair(vkey, map<ulong, DirePSWeight>() ));
    showerWeight.insert( make_pair(vkey, 1.) );
    weightNames.push_back( vkey );
  }
}

//--------------------------------------------------------------------------

void DireWeightContainer::setWeight( string varKey, double value) {
  unordered_map<string, double>::iterator it = showerWeight.find( varKey );
  if ( it == showerWeight.end() ) return;
  it->second = value;
}

//--------------------------------------------------------------------------

void DireWeightContainer::resetAcceptWeight( double pT2key, double value,
  string varKey) {
  unordered_map<string, map<ulong, DirePSWeight> >::iterator
    it0 = acceptWeight.find( varKey );
  if ( it0 == acceptWeight.end() ) return;
  map<ulong, DirePSWeight>::iterator
    it = acceptWeight[varKey].find( key(pT2key) );
  if ( it == acceptWeight[varKey].end() ) return;
  acceptWeight[varKey].erase(it);
  acceptWeight[varKey].insert( make_pair( key(pT2key),
                                          DirePSWeight(value,1,0,pT2key,"")));
}

//--------------------------------------------------------------------------

void DireWeightContainer::resetRejectWeight( double pT2key, double value,
  string varKey) {
  unordered_map<string, map<ulong, DirePSWeight> >::iterator
    it0 = rejectWeight.find( varKey );
  if ( it0 == rejectWeight.end() ) return;
  map<ulong, DirePSWeight>::iterator
    it = rejectWeight[varKey].find( key(pT2key) );
  if ( it == rejectWeight[varKey].end() ) return;
  rejectWeight[varKey].erase(it);
  rejectWeight[varKey].insert( make_pair( key(pT2key),
                                          DirePSWeight(value,1,0,pT2key,"")));
}

//--------------------------------------------------------------------------

void DireWeightContainer::eraseAcceptWeight( double pT2key, string varKey) {
  unordered_map<string, map<ulong, DirePSWeight> >::iterator
    it0 = acceptWeight.find( varKey );
  if ( it0 == acceptWeight.end() ) return;
  map<ulong, DirePSWeight>::iterator
    it = acceptWeight[varKey].find( key(pT2key) );
  if ( it == acceptWeight[varKey].end() ) return;
  acceptWeight[varKey].erase(it);
}

//--------------------------------------------------------------------------

void DireWeightContainer::eraseRejectWeight( double pT2key, string varKey) {
  unordered_map<string, map<ulong, DirePSWeight> >::iterator it0 =
    rejectWeight.find( varKey );
  if ( it0 == rejectWeight.end() ) return;
  map<ulong, DirePSWeight>::iterator it =
    rejectWeight[varKey].find( key(pT2key) );
  if ( it == rejectWeight[varKey].end() ) return;
  rejectWeight[varKey].erase(it);
}

//--------------------------------------------------------------------------

double DireWeightContainer::getAcceptWeight( double pT2key, string varKey) {
  unordered_map<string, map<ulong, DirePSWeight> >::iterator it0 =
    acceptWeight.find( varKey );
  if ( it0 == acceptWeight.end() ) return 0./0.;
  map<ulong, DirePSWeight>::iterator it =
    acceptWeight[varKey].find( key(pT2key) );
  if ( it == acceptWeight[varKey].end() ) return 0./0.;
  return it->second.weight();
}


//--------------------------------------------------------------------------

double DireWeightContainer::getRejectWeight( double pT2key, string varKey) {
  unordered_map<string, map<ulong, DirePSWeight> >::iterator it0 =
    rejectWeight.find( varKey );
  if ( it0 == rejectWeight.end() ) return 0./0.;
  map<ulong, DirePSWeight>::iterator it =
    rejectWeight[varKey].find( key(pT2key) );
  if ( it == rejectWeight[varKey].end() ) return 0./0.;
  return it->second.weight();
}

//--------------------------------------------------------------------------

// Attach accept/reject probabilities for a proposed shower step.
void DireWeightContainer::insertWeights( map<double,double> aWeight,
  multimap<double,double> rWeight, string varKey ) {

  unordered_map<string, map<ulong, DirePSWeight> >::iterator itA0 =
    acceptWeight.find( varKey );
  if ( itA0 == acceptWeight.end() ) return;
  unordered_map<string, map<ulong, DirePSWeight> >::iterator itR0 =
    rejectWeight.find( varKey );
  if ( itR0 == rejectWeight.end() ) return;

  // New accept weights.
  for ( map<double,double>::iterator it = aWeight.begin();
    it != aWeight.end(); ++it ){
    map<ulong, DirePSWeight>::iterator itLo
      = acceptWeight[varKey].find( key(it->first) );
    if (itLo == acceptWeight[varKey].end())
      acceptWeight[varKey].insert(make_pair( key(it->first),
                                   DirePSWeight(it->second,1,0,it->first,"")));
    else
      itLo->second *= it->second;
  }
  // New reject weights.
  for ( multimap<double,double>::iterator it = rWeight.begin();
    it != rWeight.end(); ++it ){
    map<ulong, DirePSWeight>::iterator itLo
      = rejectWeight[varKey].find( key(it->first) );
    if (itLo == rejectWeight[varKey].end())
      rejectWeight[varKey].insert
        (make_pair( key(it->first),
                    DirePSWeight(it->second,-1,0,it->first,"")));
    else
      itLo->second *= it->second;
  }
}

//--------------------------------------------------------------------------

// Function to calculate the weight of the shower evolution step.
void DireWeightContainer::calcWeight(double pT2, bool includeAcceptAtPT2,
  bool includeRejectAtPT2) {

  // Loop though weights.
  for ( unordered_map<string, map<ulong, DirePSWeight> >::iterator
    it = rejectWeight.begin(); it != rejectWeight.end(); ++it ) {
    // Set accept weight.
    bool hasAccept  = ( acceptWeight[it->first].find(key(pT2))
                     != acceptWeight[it->first].end());
    double acceptWt = (hasAccept && includeAcceptAtPT2)
                     ? acceptWeight[it->first].find(key(pT2))->second.weight()
                     : 1.;

    // Now multiply rejection weights.
    double rejectWt = 1.;
    for ( map<ulong, DirePSWeight>::reverse_iterator itR
      = it->second.rbegin(); itR != it->second.rend(); ++itR ){
        if ( includeRejectAtPT2 && itR->first == key(pT2) ) {
          rejectWt *= itR->second.weight(); }
        if ( itR->first > key(pT2) ) { rejectWt *= itR->second.weight(); }
        if ( itR->first < key(pT2) || itR->first-key(pT2) == 0) break;
    }

    // Remember weights
    unordered_map<string, double>::iterator itW = showerWeight.find(it->first);

    if (itW != showerWeight.end()) itW->second *= acceptWt*rejectWt;

  }

}

//--------------------------------------------------------------------------

// Function to calculate the weight of the shower evolution step.
pair<double,double> DireWeightContainer::getWeight(double pT2, string varKey) {

  // Set accept weight.
  bool hasAccept  = ( acceptWeight[varKey].find(key(pT2))
                   != acceptWeight[varKey].end());
  double acceptWt = (hasAccept)
                   ? acceptWeight[varKey].find(key(pT2))->second.weight()
                   : 1.;

  // Now multiply rejection weights.
  double rejectWt = 1.;
  unordered_map<string, map<ulong, DirePSWeight> >::iterator itRW
    = rejectWeight.find(varKey);
  if (itRW != rejectWeight.end()) {

    // Now multiply rejection weights.
    for ( map<ulong, DirePSWeight>::reverse_iterator itR
      = itRW->second.rbegin(); itR != itRW->second.rend();
      ++itR ){
        if ( itR->first > key(pT2) ) rejectWt *= itR->second.weight();
        if ( itR->first < key(pT2) || itR->first-key(pT2) == 0) break;
    }

  }

  // Remember weights
  unordered_map<string, double>::iterator itW = showerWeight.find(varKey);

  if (itW != showerWeight.end()
    && abs(itW->second) > LARGEWT) direInfoPtr->message(1) << scientific
    << setprecision(8) << __FILE__ << " " << __func__ << " "
    << __LINE__ << " : Found large shower weight=" << itW->second
    << " at pT2=" << pT2 << endl;

  if (itW != showerWeight.end()) rejectWt *= itW->second;

  // Diagnostic messages.
  if (abs(acceptWt) > LARGEWT) direInfoPtr->message(1) << scientific
    << setprecision(8) << __FILE__ << " " << __func__ << " "
    << __LINE__ << " : Found large accept weight=" << acceptWt
    << " at pT2=" << pT2 << endl;
  if ( abs(rejectWt) > LARGEWT) {
    for ( map<ulong, DirePSWeight>::reverse_iterator itR
      = itRW->second.rbegin(); itR != itRW->second.rend(); ++itR ){
      if ( itR->first > key(pT2) ) {
        if ( abs(itR->second.weight()) > LARGEWT) direInfoPtr->message(1)
          << scientific << setprecision(8) << __FILE__ << " " << __func__
          << " " << __LINE__ << " : Found large reject weight="
          << itR->second.weight() << " at index=" << itR->first
          << " (pT2 approx. " << dkey(itR->first) << ")" << endl;
      }
      if ( itR->first < key(pT2) || itR->first-key(pT2) == 0) break;
    }
  }

  // Done.
  return make_pair(acceptWt,rejectWt);

}

//--------------------------------------------------------------------------

// Returns additional user-supplied enhancements factors.

double DireWeightContainer::enhanceOverestimate( string name ) {
  unordered_map<string, double>::iterator it = enhanceFactors.find(name );
  if ( it == enhanceFactors.end() ) return 1.;
  return it->second;
}

//--------------------------------------------------------------------------

double DireWeightContainer::getTrialEnhancement( double pT2key ) {
  map<ulong,double>::iterator it = trialEnhancements.find(key(pT2key));
  if ( it == trialEnhancements.end() ) return 1.;
  return it->second;
}

//--------------------------------------------------------------------------

bool DireWeightContainer::hasME(const Event& event) {
  if (hasMEs) return matrixElements.isAvailableMEDire(event);
  return false;
}

bool DireWeightContainer::hasME(vector <int> in_pdgs, vector<int> out_pdgs) {
  if (hasMEs) return matrixElements.isAvailableMEDire(in_pdgs, out_pdgs);
  return false;
}

double DireWeightContainer::getME(const Event& event) {
  if (hasMEs) return matrixElements.calcMEDire(event);
  return 0.0;
}

//==========================================================================

} // end namespace Pythia8
