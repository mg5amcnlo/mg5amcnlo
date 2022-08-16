// DireSplittings.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for the OverheadInfo and DireSplitting classes.

#ifndef Pythia8_DireSplittings_H
#define Pythia8_DireSplittings_H

#define DIRE_SPLITTINGS_VERSION "2.002"

#include "Pythia8/Basics.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StandardModel.h"
#include "Pythia8/Info.h"
#include "Pythia8/DireSplitInfo.h"
#include "Pythia8/DireBasics.h"

namespace Pythia8 {

class DireSpace;
class DireTimes;

//==========================================================================

class OverheadInfo {

public:

  OverheadInfo(int nFinalIn, int idIn, double valIn, double xIn, double pT2In)
    : nFinal(nFinalIn), id(idIn), val(valIn), x(xIn), pT2(pT2In) {}

  int nFinal, id;
  double val, x, pT2;

  bool match(int idIn, int nfIn) { return (idIn==id && nfIn==nFinal); }

  string list () const {
    ostringstream os;
    os << scientific << setprecision(6)
    << "pT2 " << setw(10) << pT2 << " x " << setw(10) << x
    << " id " << setw(4) << id << " nf " << setw(4) << nFinal
    << " val=" << val;
    return os.str();
  }

};

class DireSplitting {

public:

  // Constructor and destructor.
  DireSplitting() : renormMultFac(0),
      id("void"), correctionOrder(0), settingsPtr(0),
      particleDataPtr(0), rndmPtr(0), beamAPtr(0),
      beamBPtr(0),  coupSMPtr(0), infoPtr(0), direInfoPtr(0),
      is_qcd(false), is_qed(false), is_ewk(false), is_fsr(false),
      is_isr(false), is_dire(false), nameHash(0) {}
  DireSplitting(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSMPtrIn, Info* infoPtrIn,
                DireInfo* direInfo) :
      renormMultFac(0), id(idIn), correctionOrder(softRS),
      settingsPtr(settings), particleDataPtr(particleData), rndmPtr(rndm),
      beamAPtr(beamA), beamBPtr(beamB), coupSMPtr(coupSMPtrIn),
      infoPtr(infoPtrIn), direInfoPtr(direInfo), is_qcd(false), is_qed(false),
      is_ewk(false), is_fsr(false), is_isr(false), is_dire(false),
       nameHash(0) { init(); splitInfo.storeName(name()); }
  virtual ~DireSplitting() {}

  void init();

public:

  double renormMultFac;

  string id;
  int correctionOrder;
  Settings* settingsPtr;
  ParticleData* particleDataPtr;
  Rndm* rndmPtr;
  BeamParticle* beamAPtr;
  BeamParticle* beamBPtr;
  CoupSM* coupSMPtr;
  Info* infoPtr;
  DireInfo* direInfoPtr;

  // Some short-cuts and string hashes to help avoid string comparisons.
  bool is_qcd, is_qed, is_ewk, is_fsr, is_isr, is_dire;
  ulong nameHash;
  bool is( ulong pattern ) {
    if (pattern == nameHash) return true;
    return false;
  }

  unordered_map<string,double> kernelVals;

  string name () {return id;}

  virtual bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL)
    {return false;}

  // Discard below the cut-off for the splitting.
  virtual bool aboveCutoff( double, const Particle&, const Particle&, int,
    PartonSystems* = NULL) { return true; }

  virtual bool useFastFunctions() { return false; }
  virtual bool canRadiate ( const Event&, int, int,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL)
    {return false;}

  // Function to return an identifier for the phase space mapping
  // that is used for setting up this splitting.
  // return values: 1 --> Default Dire mapping.
  //                2 --> Dire 1->3 mapping.
  virtual int kinMap () {return 1;}

  // Return id of mother after splitting.
  virtual int motherID(int) {return 0;}

  // Return id of emission.
  virtual int sisterID(int) {return 0;}

  // Return a pair of ids for the radiator and emission after
  // the splitting.
  virtual vector <int> radAndEmt(int, int) { return vector<int>(); }
  virtual vector < pair<int,int> > radAndEmtCols( int, int, Event)
    { return vector<pair<int,int> >(); }
  virtual bool canUseForBranching() { return false; }
  virtual bool isPartial()  { return false; }
  virtual int  nEmissions() { return 0; }

  virtual bool swapRadEmt() { return false; }
  virtual bool isSymmetric( const Particle* = NULL, const Particle* = NULL)
    { return false; }

  // Return a vector of all possible recoiler positions, given the
  // positions of the radiator and emission after the splitting.
  virtual vector <int> recPositions( const Event&, int, int)
    {return vector<int>();}

  // Return id of recombined radiator (before splitting!)
  virtual int radBefID(int, int) {return 0;}

  // Return colours of recombined radiator (before splitting!)
  virtual pair<int,int> radBefCols(int, int, int, int)
    {return make_pair(0,0);}

  // Return color factor for splitting.
  virtual double gaugeFactor (int, int) {return 1.;}

  // Return symmetry factor for splitting.
  virtual double symmetryFactor (int, int) {return 1.;}

  // Return an identifier for the interaction that causes the
  // branching.
  // return values -1 --> Type not defined.
  //                1 --> QCD splitting (i.e. proportional to alphaS)
  //                2 --> QED splitting (i.e. proportional to alphaEM)
  //                3 --> EW splitting (i.e. proportional to sinThetaW)
  //                4 --> Yukawa splitting (i.e. proportional to y)
  virtual int couplingType (int, int) {return -1;}

  // Return the value of the coupling that should be used for this branching.
  // Note that the last input allows easy access to the PS evolution variable.
  // return values -1         --> Coupling value not defined.
  //               double > 0 --> Value to be used for this branching.
  virtual double coupling (double = 0., double = 0., double = 0.,
    double = -1., pair<int,bool> = pair<int,bool>(),
    pair<int,bool> = pair<int,bool>()) {
    return -1.;
  }
  virtual double couplingScale2 (double = 0., double = 0., double = 0.,
    pair<int,bool> = pair<int,bool>(), pair<int,bool> = pair<int,bool>()) {
    return -1.;
  }

  // Pick z for new splitting.
  virtual double zSplit(double, double, double) {return 0.5;}

  // New overestimates, z-integrated versions.
  virtual double overestimateInt(double, double, double, double, int = -1)
    { return 0.;}

  // Return kernel for new splitting.
  virtual double overestimateDiff(double, double, int = -1) {return 1.;}

  // Functions to store and retrieve all the variants of the kernel.
  virtual double getKernel(string = "");
  virtual unordered_map<string,double> getKernelVals() { return kernelVals; }
  virtual void   clearKernels()         { kernelVals.clear(); }

  DireSplitInfo splitInfo;

  // Functions to calculate the kernel from SplitInfo information.
  virtual bool calc(const Event& = Event(), int = -1) { return false; }

  shared_ptr<DireSpace> isr;
  shared_ptr<DireTimes> fsr;
  shared_ptr<DireTimes> fsrDec;
  void setTimesPtr(shared_ptr<DireTimes> fsrIn)    { fsr=fsrIn;}
  void setTimesDecPtr(shared_ptr<DireTimes> fsrIn) { fsrDec=fsrIn;}
  void setSpacePtr(shared_ptr<DireSpace> isrIn)    { isr=isrIn;}

  // Functions that allow different ordering variables for emissions.
  // Note: Only works after splitInfo has been properly filled.
  virtual double getJacobian( const Event& = Event(),
    PartonSystems* = 0) { return 0.;}
  // Map filled identical to shower state variables map.
  virtual unordered_map<string, double> getPhasespaceVars(
    const Event& = Event(),
    PartonSystems* = 0) { return unordered_map<string,double>(); }

  // Treatment of additional virtual corrections.
  virtual bool allow_z_endpoint_for_kinematics()   { return false; }
  virtual bool allow_pT2_endpoint_for_kinematics() { return false; }
  virtual bool allow_sai_endpoint_for_kinematics() { return false; }
  virtual bool allow_xa_endpoint_for_kinematics()  { return false; }
  // Functions to set if kernel should contribute to a kinematical endpoint.
  virtual void try_z_endpoint()                    { return; }
  virtual void try_pT2_endpoint()                  { return; }
  virtual void try_sai_endpoint()                  { return; }
  virtual void try_xa_endpoint()                   { return; }
  // Return endpoint information.
  virtual bool is_z_endpoint()                     { return false; }
  virtual bool is_pT2_endpoint()                   { return false; }
  virtual bool is_sai_endpoint()                   { return false; }
  virtual bool is_xa_endpoint()                    { return false; }

  // Functions to calculate Dire variables from the evolution variables.
  virtual double tdire_ff(double, double t, double) { return t; }
  virtual double zdire_ff(double z, double, double) { return z; }
  virtual double tdire_fi(double, double t, double) { return t; }
  virtual double zdire_fi(double z, double, double) { return z; }
  virtual double tdire_if(double, double t, double) { return t; }
  virtual double zdire_if(double z, double, double) { return z; }
  virtual double tdire_ii(double, double t, double) { return t; }
  virtual double zdire_ii(double z, double, double) { return z; }

  virtual bool hasMECBef(const Event&, double) { return false; }
  virtual bool hasMECAft(const Event&, double) { return false; }

  virtual void storeOverhead(double pT2, double x, int radid, int nf,
    double val) { overhead_map.insert(make_pair(pT2, OverheadInfo(nf, radid,
    val, x, pT2))); }
  multimap<double,OverheadInfo> overhead_map;

  virtual double overhead(double pT2, int idd, int nf) {

    if (overhead_map.empty()) return 1.;

    multimap<double,OverheadInfo>::iterator lo = overhead_map.lower_bound(pT2);
    if (lo != overhead_map.begin()) lo--;
    if (lo != overhead_map.begin()) lo--;
    multimap<double,OverheadInfo>::iterator hi = overhead_map.upper_bound(pT2);
    if (hi != overhead_map.end()) hi++;
    if (hi == overhead_map.end()) hi--;

    int n(0);
    double sum = 0.;
    for ( multimap<double,OverheadInfo>::iterator it = lo;
      it != hi; ++it ){
      if (!it->second.match(idd,nf)) continue;
      sum += it->second.val;
      n++;
    }

    if (hi->second.match(idd,nf)) {
      sum += hi->second.val;
      n++;
    }

    return max(sum/max(1,n),1.);

  }

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_DireSplittings_H
