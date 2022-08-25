// DireWeightContainer.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for weight constainers for Dire parton shower.

#ifndef Pythia8_DireWeightContainer_H
#define Pythia8_DireWeightContainer_H

#define DIRE_WEIGHTCONTAINER_VERSION "2.002"

#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/DireBasics.h"
#include "Pythia8/ExternalMEs.h"

namespace Pythia8 {

//==========================================================================

// Container for a single weight with auxiliary information.

class DirePSWeight {

public:

  // Constructors.
  DirePSWeight()
    : wt(1.0), type(0), iAtt(0), dAtt(0.0), sAtt("") {}

  DirePSWeight( double w)
    : wt(w), type(0), iAtt(0), dAtt(0.0), sAtt("") {}

  DirePSWeight( double w, int typeIn, int iAttIn=0, double dAttIn=0.0,
    string sAttIn="") : wt(w), type(typeIn), iAtt(iAttIn), dAtt(dAttIn),
    sAtt(sAttIn) {}

  DirePSWeight( const DirePSWeight& wgt) : wt(wgt.wt), type(wgt.type),
    iAtt(wgt.iAtt), dAtt(wgt.dAtt), sAtt(wgt.sAtt), auxwt(wgt.auxwt) {}

  DirePSWeight( double w, int typeIn, double full, double over, double aux,
    int iAttIn=0, double dAttIn=0.0, string sAttIn="") : wt(w), type(typeIn),
    iAtt(iAttIn), dAtt(dAttIn), sAtt(sAttIn) { auxwt.push_back(full);
    auxwt.push_back(over); auxwt.push_back(aux); }

  // Return functions.
  double weight()     { return wt; }
  int    iAttribute() { return iAtt; }
  double dAttribute() { return dAtt; }
  string sAttribute() { return sAtt; }

  // Set functions.
  void setWeight  ( double w) { wt  = w; }
  inline DirePSWeight& operator*=(double f) { wt *= f; return *this; }

private:

  // Value of the weight.
  double wt;

  // Remember type: 1-> Accept weight, 0-> Not assigned, -1->Reject weight
  int type;

  // Auxiliary attributes and identifiers.
  int iAtt;
  double dAtt;
  string sAtt;

  // Auxiliary weights, to e.g. store kernel, overestimate and auxiliary
  // estimate separately.
  // Ordering:
  // auxwt[0] -> full kernel,
  // auxwt[1] -> overestimate,
  // auxwt[2] -> auxiliary overestimate
  vector<double> auxwt;

};

// Container for all shower weights, including handling.

class DireWeightContainer {

public:

  // Constructor.
  DireWeightContainer() : card(""), hasMEs(false), settingsPtr(nullptr),
    beamA(nullptr), beamB(nullptr), infoPtr(nullptr), direInfoPtr(nullptr)
    { init(); }

  DireWeightContainer(Settings* settingsPtrIn) : card(""), hasMEs(false),
    settingsPtr(settingsPtrIn), beamA(nullptr), beamB(nullptr),
    infoPtr(nullptr), direInfoPtr(nullptr)
    { init(); }

  // Initialize weights.
  void init() {
    reset();
    for ( unordered_map<string, double>::iterator itw = showerWeight.begin();
      itw != showerWeight.end(); ++itw ) itw->second = 1.;
  }
  void setup();

  void initPtrs(BeamParticle* beamAIn, BeamParticle* beamBIn,
    Settings* settingsPtrIn, Info* infoPtrIn, DireInfo* direInfoPtrIn) {
    beamA    = beamAIn;
    beamB    = beamBIn;
    settingsPtr = settingsPtrIn;
    infoPtr = infoPtrIn;
    direInfoPtr = direInfoPtrIn;
    return;
  }

  // Reset current accept/reject probabilities.
  void reset() {
    for ( unordered_map<string, map<ulong, DirePSWeight> >::iterator
      it = rejectWeight.begin(); it != rejectWeight.end(); ++it )
      it->second.clear();
    for ( unordered_map<string, map<ulong, DirePSWeight> >::iterator
      it = acceptWeight.begin(); it != acceptWeight.end(); ++it )
      it->second.clear();
  }
  void clear() {
    reset();
    for ( unordered_map<string, double >::iterator it = showerWeight.begin();
      it != showerWeight.end(); ++it ) it->second = 1.0;
  }

  // Function to initialize new maps for a new shower variation.
  void bookWeightVar(string varKey, bool checkSettings = true);

  // To avoid rounding problems, maps will be indexed with long keys.
  // Round double inputs to four decimals, as long will should be >10 digits.
  ulong key(double a) { return (ulong)(a*1e8+0.5); }
  double dkey(ulong a) { return (double(a)/1e8); }

  void setWeight( string varKey, double value);
  void resetAcceptWeight( double pT2key, double value, string varKey);
  void resetRejectWeight( double pT2key, double value, string varKey);
  void eraseAcceptWeight( double pT2key, string varKey);
  void eraseRejectWeight( double pT2key, string varKey);
  double getAcceptWeight( double pT2key, string varKey);
  double getRejectWeight( double pT2key, string varKey);

  // Attach accept/reject probabilities for a proposed shower step.
  void insertWeights( map<double,double> aWeight,
    multimap<double,double> rWeight, string varKey );

  // Function to calculate the weight of the shower evolution step.
  void calcWeight(double pT2, bool includeAcceptAtPT2 = true,
    bool includeRejectAtPT2 = false);

  pair<double,double> getWeight(double pT2, string valKey = "base");

  // Function to return weight of the shower evolution.
  double getShowerWeight(string valKey = "base") {
    // First try to return an individual shower weight indexed by "valKey".
    unordered_map<string, double>::iterator it1 = showerWeight.find( valKey );
    if ( it1 != showerWeight.end() ) return it1->second;

    // If not possible, return a product of shower weights indexed by "valKey".
    unordered_map<string, vector<string> >::iterator it2
      = weightCombineList.find(valKey);
    if ( it2 != weightCombineList.end() ) {
      double wtNow = 1.;
      // Loop through group of weights and combine all weights into one weight.
      for (int iwgtname=0; iwgtname < int(it2->second.size()); ++iwgtname) {
        unordered_map<string, double>::iterator it3
          = showerWeight.find( it2->second[iwgtname] );
       if ( it3 != showerWeight.end() ) wtNow *= it3->second;
      }
      return wtNow;
    }
    // Done.
    return 0.;
  }

  unordered_map<string,double>* getShowerWeights() { return &showerWeight; }
  double sizeWeights() const { return showerWeight.size(); }
  string weightName (int i) const  { return weightNames[i]; }

  double sizeWeightgroups() const { return weightCombineList.size(); }
  string weightgroupName (int i) const  {
    return weightCombineListNames[i];
  }

  // Returns additional user-supplied enhancements factors.
  double enhanceOverestimate( string name );
  double getTrialEnhancement(double pT2key);
  void   clearTrialEnhancements() { trialEnhancements.clear(); }
  void   addTrialEnhancement( double pT2key, double value) {
    trialEnhancements.insert(make_pair(key(pT2key), value));
  }

  // MG5 matrix element access.
  string card;
  ExternalMEsPlugin matrixElements;
  bool hasMEs;

  bool hasME(vector <int> in_pdgs = vector<int>(),
    vector<int> out_pdgs = vector<int>() );
  bool hasME(const Event& event);
  double getME(const Event& event);

private:

  static const double LARGEWT;

  Settings* settingsPtr;

  unordered_map<string, map<ulong, DirePSWeight> > acceptWeight;
  unordered_map<string, map<ulong, DirePSWeight> > rejectWeight;
  unordered_map<string, double> showerWeight;
  vector<string> weightNames;
  unordered_map<string, vector<string> > weightCombineList;
  vector<string> weightCombineListNames;

  // Additonal enhancement factors to boost emission probabilities.
  unordered_map<string,double> enhanceFactors;
  map<ulong, double> trialEnhancements;

  BeamParticle* beamA;
  BeamParticle* beamB;
  Info* infoPtr;
  DireInfo* direInfoPtr;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_DireWeightContainer_H
