
#include "Dire/WeightContainer.h"
#include "Dire/DireSpace.h"
#include "Dire/DireTimes.h"

namespace Pythia8 {

//==========================================================================

// Container for all shower weights, including handling.

void WeightContainer::setup() {

  // Clear everything.
  init();
  enhanceFactors.clear();
 
  // Initialize MG5 interface.
  card=settingsPtr->word("Dire:MG5card");
  PY8MEs_accessor.initModelFromSLHACard(card);
   

  // Initialize additional user-defined enhancements of splitting kernel
  // overestimates.
  int sizeNames = 32;
  const char* names[] = {
    "fsr_qcd_1->1&21_CS",
    "fsr_qcd_1->21&1_CS",
    "fsr_qcd_21->21&21a_CS",
    "fsr_qcd_21->21&21b_CS",
    "fsr_qcd_21->1&1a_CS",
    "fsr_qcd_21->1&1b_CS",
    "fsr_qcd_1->2&1&2_CS",
    "fsr_qcd_1->1&1&1_CS",
    "fsr_qed_1->1&22_CS",
    "fsr_qed_1->22&1_CS",
    "fsr_qed_22->1&1a_CS",
    "fsr_qed_22->1&1b_CS",
    "fsr_ew_1->1&23_CS",
    "fsr_ew_1->23&1_CS",
    "fsr_ew_23->1&1a_CS",
    "fsr_ew_23->1&1b_CS",
    "fsr_ew_24->1&1a_CS",
    "fsr_ew_24->1&1b_CS",
    "isr_qcd_1->1&21_CS",
    "isr_qcd_21->1&1_CS",
    "isr_qcd_21->21&21a_CS",
    "isr_qcd_21->21&21b_CS",
    "isr_qcd_1->21&1_CS",
    "isr_qcd_1->2&1&2_CS",
    "isr_qcd_1->1&1&1_CS",
    "isr_qed_1->1&22_CS",
    "isr_qed_1->22&1_CS",
    "isr_qed_22->1&1_CS",
    "isr_qed_1->22&1_CS",
    "isr_qed_11->11&22_CS",
    "isr_qed_11->22&11_CS",
    "isr_qed_22->11&11_CS"};

  for (int i=0; i < sizeNames; ++i) {
    if (settingsPtr->parm("Enhance:"+string(names[i])) > 1.0) {
      enhanceFactors.insert(
        make_pair( string(names[i]),
        settingsPtr->parm("Enhance:"+string(names[i]))) );
    }
  }

  string vkey = "base";
  rejectWeight.insert( make_pair(vkey, map<unsigned long, PSWeight>() ));
  acceptWeight.insert( make_pair(vkey, map<unsigned long, PSWeight>() ));
  showerWeight.insert( make_pair(vkey, 1.) );
  weightNames.push_back( vkey );

  bool doVar = settingsPtr->flag("Variations:doVariations");
  if ( doVar ) {
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      bookWeightVar("Variations:muRisrDown");
    if (settingsPtr->parm("Variations:muRisrUp") != 1.)
      bookWeightVar("Variations:muRisrUp");
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      bookWeightVar("Variations:muRfsrDown");
    if (settingsPtr->parm("Variations:muRfsrUp") != 1.)
      bookWeightVar("Variations:muRfsrUp");
  }

  // Remember groups of weights that should be combined into one weight.
  vector<string> group;
  group = createvector<string>("Variations:muRfsrUp")
    ("Variations:muRisrUp");
  weightCombineList.insert ( make_pair("scaleUp", group) );
  group = createvector<string>("Variations:muRfsrDown")
    ("Variations:muRisrDown");
  weightCombineList.insert ( make_pair("scaleDown", group) );

}

//--------------------------------------------------------------------------

void WeightContainer::bookWeightVar( string vkey, bool checkSettings) {
  bool insert =  !checkSettings
              || (checkSettings && settingsPtr->parm(vkey) != 1.0);
  if (insert) {
    rejectWeight.insert( make_pair(vkey, map<unsigned long, PSWeight>() ));
    acceptWeight.insert( make_pair(vkey, map<unsigned long, PSWeight>() ));
    showerWeight.insert( make_pair(vkey, 1.) );
    weightNames.push_back( vkey );
  }
}

//--------------------------------------------------------------------------

void WeightContainer::resetAcceptWeight( double pT2key, double value,
  string varKey) {
  map<string, map<unsigned long, PSWeight> >::iterator it0 = acceptWeight.find( varKey );
  if ( it0 == acceptWeight.end() ) return;
  map<unsigned long, PSWeight>::iterator it = acceptWeight[varKey].find( key(pT2key) );
  if ( it == acceptWeight[varKey].end() ) return;
  acceptWeight[varKey].erase(it);
  acceptWeight[varKey].insert( make_pair( key(pT2key), PSWeight(value,1,0,pT2key,"")));
}

//--------------------------------------------------------------------------

void WeightContainer::resetRejectWeight( double pT2key, double value,
  string varKey) {
  map<string, map<unsigned long, PSWeight> >::iterator it0 = rejectWeight.find( varKey );
  if ( it0 == rejectWeight.end() ) return;
  map<unsigned long, PSWeight>::iterator it = rejectWeight[varKey].find( key(pT2key) );
  if ( it == rejectWeight[varKey].end() ) return;
  rejectWeight[varKey].erase(it);
  rejectWeight[varKey].insert( make_pair( key(pT2key), PSWeight(value,1,0,pT2key,"")));
}

//--------------------------------------------------------------------------

void WeightContainer::eraseAcceptWeight( double pT2key, string varKey) {
  map<string, map<unsigned long, PSWeight> >::iterator it0 = acceptWeight.find( varKey );
  if ( it0 == acceptWeight.end() ) return;
  map<unsigned long, PSWeight>::iterator it = acceptWeight[varKey].find( key(pT2key) );
  if ( it == acceptWeight[varKey].end() ) return;
  acceptWeight[varKey].erase(it);
}

//--------------------------------------------------------------------------

void WeightContainer::eraseRejectWeight( double pT2key, string varKey) {
  map<string, map<unsigned long, PSWeight> >::iterator it0 = rejectWeight.find( varKey );
  if ( it0 == rejectWeight.end() ) return;
  map<unsigned long, PSWeight>::iterator it = rejectWeight[varKey].find( key(pT2key) );
  if ( it == rejectWeight[varKey].end() ) return;
  rejectWeight[varKey].erase(it);
}

//--------------------------------------------------------------------------

double WeightContainer::getAcceptWeight( double pT2key, string varKey) {
  map<string, map<unsigned long, PSWeight> >::iterator it0 = acceptWeight.find( varKey );
  if ( it0 == acceptWeight.end() ) return 0./0.;
  map<unsigned long, PSWeight>::iterator it = acceptWeight[varKey].find( key(pT2key) );
  if ( it == acceptWeight[varKey].end() ) return 0./0.;
  return it->second.weight();
}


//--------------------------------------------------------------------------

double WeightContainer::getRejectWeight( double pT2key, string varKey) {
  map<string, map<unsigned long, PSWeight> >::iterator it0 = rejectWeight.find( varKey );
  if ( it0 == rejectWeight.end() ) return 0./0.;
  map<unsigned long, PSWeight>::iterator it = rejectWeight[varKey].find( key(pT2key) );
  if ( it == rejectWeight[varKey].end() ) return 0./0.;
  return it->second.weight();
}

//--------------------------------------------------------------------------

// Attach accept/reject probabilities for a proposed shower step.
void WeightContainer::insertWeights( map<double,double> aWeight,
  multimap<double,double> rWeight, string varKey ) {

  map<string, map<unsigned long, PSWeight> >::iterator itA0 = acceptWeight.find( varKey );
  if ( itA0 == acceptWeight.end() ) return;
  map<string, map<unsigned long, PSWeight> >::iterator itR0 = rejectWeight.find( varKey );
  if ( itR0 == rejectWeight.end() ) return;

  // New accept weights.
  for ( map<double,double>::iterator it = aWeight.begin();
    it != aWeight.end(); ++it ){
    map<unsigned long, PSWeight>::iterator itLo
      = acceptWeight[varKey].find( key(it->first) );
    if (itLo == acceptWeight[varKey].end())
      acceptWeight[varKey].insert(make_pair( key(it->first),
                                   PSWeight(it->second,1,0,it->first,"")) );
    else
      itLo->second *= it->second;
  }
  // New reject weights.
  for ( multimap<double,double>::iterator it = rWeight.begin();
    it != rWeight.end(); ++it ){
    map<unsigned long, PSWeight>::iterator itLo
      = rejectWeight[varKey].find( key(it->first) );
    if (itLo == rejectWeight[varKey].end())
      rejectWeight[varKey].insert(make_pair( key(it->first),
                                   PSWeight(it->second,-1,0,it->first,"")) );
    else
      itLo->second *= it->second;
  }
}

//--------------------------------------------------------------------------

// Function to calculate the weight of the shower evolution step.
void WeightContainer::calcWeight(double pT2) {

  // Loop though weights.
  for ( map<string, map<unsigned long, PSWeight> >::iterator
    it = rejectWeight.begin(); it != rejectWeight.end(); ++it ) {
    // Set accept weight.
    bool hasAccept  = ( acceptWeight[it->first].find(key(pT2))
                     != acceptWeight[it->first].end());
    double acceptWt = (hasAccept)
                     ? acceptWeight[it->first].find(key(pT2))->second.weight()
                     : 1.;

    // Now multiply rejection weights.
    double rejectWt = 1.;
    for ( map<unsigned long, PSWeight>::reverse_iterator itR
      = it->second.rbegin(); itR != it->second.rend(); ++itR ){
        if ( itR->first > key(pT2) ) rejectWt *= itR->second.weight();
        if ( itR->first < key(pT2) || itR->first-key(pT2) == 0) break;
    }

    // Remember weights
    map<string, double>::iterator itW = showerWeight.find(it->first);
    if (itW != showerWeight.end()) itW->second *= acceptWt*rejectWt;

    // Diagnostic messages.
    if (abs(acceptWt) > 5.) debugPtr->message(1) << scientific
      << setprecision(8) << __FILE__ << " " << __func__ << " "
      << __LINE__ << " : Found large accept weight=" << acceptWt
      << " at pT2=" << pT2 << endl;
    if ( abs(rejectWt) > 5.) {
      for ( map<unsigned long, PSWeight>::reverse_iterator itR
        = it->second.rbegin(); itR != it->second.rend(); ++itR ){
        if ( itR->first > key(pT2) ) {
          if ( abs(itR->second.weight()) > 5.) debugPtr->message(1)
            << scientific << setprecision(8) << __FILE__ << " " << __func__
            << " " << __LINE__ << " : Found large reject weight="
            << itR->second.weight() << " at index=" << itR->first
            << " (pT2 approx. " << dkey(itR->first) << ")" << endl;
        }
        if ( itR->first < key(pT2) || itR->first-key(pT2) == 0) break;
      }
    }

  }

}

//--------------------------------------------------------------------------

// Function to calculate the weight of the shower evolution step.
pair<double,double> WeightContainer::getWeight(double pT2, string varKey) {

  // Set accept weight.
  bool hasAccept  = ( acceptWeight[varKey].find(key(pT2))
                   != acceptWeight[varKey].end());
  double acceptWt = (hasAccept)
                   ? acceptWeight[varKey].find(key(pT2))->second.weight()
                   : 1.;

  // Now multiply rejection weights.
  double rejectWt = 1.;
  map<string, map<unsigned long, PSWeight> >::iterator itRW
    = rejectWeight.find(varKey);
  if (itRW != rejectWeight.end()) {

    // Now multiply rejection weights.
    for ( map<unsigned long, PSWeight>::reverse_iterator itR
      = itRW->second.rbegin(); itR != itRW->second.rend();
      ++itR ){
        if ( itR->first > key(pT2) ) rejectWt *= itR->second.weight();
        if ( itR->first < key(pT2) || itR->first-key(pT2) == 0) break;
    }

  }

  // Remember weights
  map<string, double>::iterator itW = showerWeight.find(varKey);
  if (itW != showerWeight.end()) rejectWt *= itW->second;

  // Diagnostic messages.
  if (abs(acceptWt) > 5.) debugPtr->message(1) << scientific
    << setprecision(8) << __FILE__ << " " << __func__ << " "
    << __LINE__ << " : Found large accept weight=" << acceptWt
    << " at pT2=" << pT2 << endl;
  if ( abs(rejectWt) > 5.) {
    for ( map<unsigned long, PSWeight>::reverse_iterator itR
      = itRW->second.rbegin(); itR != itRW->second.rend(); ++itR ){
      if ( itR->first > key(pT2) ) {
        if ( abs(itR->second.weight()) > 5.) debugPtr->message(1)
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

double WeightContainer::enhanceOverestimate( string name ) {
  map<string, double>::iterator it = enhanceFactors.find(name );
  if ( it == enhanceFactors.end() ) return 1.;
  return it->second;
}

//--------------------------------------------------------------------------

bool WeightContainer::hasME(const Event& event) {

  int nfinal(0), ng(0), nq(0), nlq(0);
  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].isFinal() && event[i].colType()!=0) nfinal++;
    if ( event[i].isFinal() && event[i].idAbs()<6) nlq++;
    if ( event[i].isFinal() && event[i].idAbs()==21) nlq++;
    if ( event[i].isFinal() && event[i].id()==21) ng++;
    if ( event[i].isFinal() && event[i].idAbs()<10) nq++;
  }
  if (nlq>2) return false;
  //if (nfinal>2) return false;

  return isAvailableME(PY8MEs_accessor,event);

}

double WeightContainer::getME(const Event& event) {
  int nfinal(0);
  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].isFinal() && event[i].colType()!=0) nfinal++;
  }
  return ( calcME(PY8MEs_accessor,event) );
}

//==========================================================================

} // end namespace Pythia8
