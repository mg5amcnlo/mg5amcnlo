
#ifndef Pythia8_WeightContainer_H
#define Pythia8_WeightContainer_H

#define DIRE_WEIGHTCONTAINER_VERSION "2.000"

#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Dire/MEwrap.h"

namespace Pythia8 {

//==========================================================================

// Container for a single weight with auxiliary information.

class PSWeight {

public: 

  // Constructors.
  PSWeight()
    : wt(1.0), type(0), iAtt(0), dAtt(0.0), sAtt("") {}

  PSWeight( double w)
    : wt(w), type(0), iAtt(0), dAtt(0.0), sAtt("") {}

  PSWeight( double w, int typeIn, int iAttIn=0, double dAttIn=0.0,
    string sAttIn="") : wt(w), type(typeIn), iAtt(iAttIn), dAtt(dAttIn),
    sAtt(sAttIn) {}

  PSWeight( const PSWeight& wgt) : wt(wgt.wt), type(wgt.type), iAtt(wgt.iAtt),
    dAtt(wgt.dAtt), sAtt(wgt.sAtt), auxwt(wgt.auxwt) {}

  PSWeight( double w, int typeIn, double full, double over, double aux, 
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
  inline PSWeight& operator*=(double f) { wt *= f; return *this; }

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

class WeightContainer {

public:

  // Constructor.
  WeightContainer() : card(""), PY8MEs_accessor(card) { init(); }

  WeightContainer(Settings* settingsPtrIn) :
    card(""), PY8MEs_accessor(card), settingsPtr(settingsPtrIn)
    { init(); }

  // Destructor.
  virtual ~WeightContainer() {}

  // Initialize weights.
  void init() { 
    reset();
    for ( map<string, double>::iterator itw = showerWeight.begin();
      itw != showerWeight.end(); ++itw ) itw->second = 1.;
  }
  void setup();

  // Reset current accept/reject probabilities.
  void reset() {
    for ( map<string, map<unsigned long, PSWeight> >::iterator
      it = rejectWeight.begin(); it != rejectWeight.end(); ++it )
      it->second.clear();
    for ( map<string, map<unsigned long, PSWeight> >::iterator
      it = acceptWeight.begin(); it != acceptWeight.end(); ++it )
      it->second.clear();
  }

  // Function to initialize new maps for a new shower variation.
  void bookWeightVar(string varKey);

  // To avoid rounding problems, maps will be indexed with long keys.
  // Round double inputs to four decimals, as long will should be >10 digits.
  unsigned long key(double a) { return (int)(a*1e4+0.5); }
  double dkey(unsigned long a) { return (double(a)/1e4-0.5); }

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
  void calcWeight(double pT2);

  pair<double,double> getWeight(double pT2, string valKey = "base");

  // Function to return weight of the shower evolution.
  double getShowerWeight(string valKey = "base") { 
    map<string, double>::iterator it = showerWeight.find( valKey );
    if ( it == showerWeight.end() ) return 0.;
    return showerWeight[valKey];
  }
  map<string,double>* getShowerWeights() { return &showerWeight; }
  double sizeWeights() { return showerWeight.size(); }

  // Returns additional user-supplied enhancements factors.
  double enhanceOverestimate( string name );

  // PY8MEs accessor
  string card;
  PY8MEs_namespace::PY8MEs PY8MEs_accessor;
  //PY8MEs_sm::PY8MEs PY8MEs_accessor;
  bool hasME(const Event& event);
  double getME(const Event& event);

private:

  Settings* settingsPtr;

  map<string, map<unsigned long, PSWeight> > acceptWeight;
  map<string, map<unsigned long, PSWeight> > rejectWeight;
  map<string, double> showerWeight;

  // Additonal enhancement factors to boost emission probabilities.
  map<string,double> enhanceFactors;

};

//==========================================================================

} // end namespace Pythia8

#endif
