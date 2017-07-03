
#ifndef Pythia8_WeightContainer_H
#define Pythia8_WeightContainer_H

#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StandardModel.h"

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
  WeightContainer() { init(); }
  //WeightContainer(Settings* settingsPtrIn) : settingsPtr(settingsPtrIn)
  //  { init(); }

  // Destructor.
  virtual ~WeightContainer() {}

  // Initialize weights.
  void init() { 
    reset();
    for ( map<string, double>::iterator itw = showerWeight.begin();
      itw != showerWeight.end(); ++itw ) itw->second = 1.;
  }
  void initPtr(Settings* settingsPtrIn) { settingsPtr = settingsPtrIn;} 
  virtual void setup();

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
  virtual void bookWeightVar(string varKey);

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

  // Get list of variations, restricted by input string.
  vector<string> getVariationKeys( string restriction = "") {
    vector<string> ret;
    for (int i=0; i < int(variationKeys.size()); ++i)
      if (variationKeys[i].find(restriction) != string::npos) 
        ret.push_back(variationKeys[i]);
    return ret;
  }

  // Attach accept/reject probabilities for a proposed shower step.
  void insertWeights( map<double,double> aWeight,
    multimap<double,double> rWeight, string varKey );

  // Function to calculate the weight of the shower evolution step.
  virtual void calcWeight(double pT2);

  // Function to return weight of the shower evolution.
  double getShowerWeight(string valKey = "base") { 
    map<string, double>::iterator it = showerWeight.find( valKey );
    if ( it == showerWeight.end() ) return 0.;
    return showerWeight[valKey];
  }
  map<string,double>* getShowerWeights() { return &showerWeight; }
  double sizeWeights() { return showerWeight.size(); }

  // Returns additional user-supplied enhancements factors.
  virtual double enhanceOverestimate( string name );

  // Returns the alpha_s associated to a particular evolution scale
  // and scaling factor
  double alphasWeight( AlphaStrong* alphaS, double pT2now,
    double renormMultFacNow );

  // Fills, for a proposed branching at scale pT2now,  the map of weight
  // variations (fullWeightsNow), based on the input weight (fullWeightVal),
  // a vector of keys (variationKeys), using some auxiliary variables (alphaS,
  // renormMultFacNow).
  virtual bool fillVariations( map<string,double>& fullWeightsNow,
    double fullWeightVal, vector<string> variationKeys, double pT2now,
    AlphaStrong* alphaS, double renormMultFacNow );

public:

  Settings* settingsPtr;

  map<string, map<unsigned long, PSWeight> > acceptWeight;
  map<string, map<unsigned long, PSWeight> > rejectWeight;
  map<string, double> showerWeight;
  vector<string> variationKeys;

  // Additonal enhancement factors to boost emission probabilities.
  map<string,double> enhanceFactors;

};

//==========================================================================

} // end namespace Pythia8

#endif
