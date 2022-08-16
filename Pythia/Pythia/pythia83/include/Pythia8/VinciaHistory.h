// VinciaHistory.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Helen Brooks, Christian T Preuss
// This file contains the VinciaHistory class.

#ifndef VINCIA_History_H
#define VINCIA_History_H

#include "Pythia8/History.h"
#include "Pythia8/Info.h"
#include "Pythia8/Vincia.h"
#include "Pythia8/VinciaAntennaFunctions.h"
#include "Pythia8/VinciaMergingHooks.h"

namespace Pythia8 {

//==========================================================================

// Convenient shorthand for storing ordered list of chains.

struct PseudoChain {
  // Ordered list of concatenated chains.
  vector<int> chainlist;
  // Index unique up to chain content (not ordering).
  int index;
  // Charge Index.
  int cindex;
  // Does any of the chains contain an initial state parton.
  bool hasInitial;
  // Flavour at start of first and end of last.
  int flavStart;
  int flavEnd;
  // Charge.
  int charge;
};

//==========================================================================

// A class for associating disconnected colour chains with either
// resonances or the hard QCD scattering. Once fixed, the sector
// history is deterministic.

class ColourFlow {

 public:

  // Constructor.
  ColourFlow() : nChains(0), nBeamChainsMin(0), nBeamChainsMax(0), nRes(0) {
    for(int i=0; i<4; i++) {countChainsByChargeIndex[i]=0;
      countResByChargeIndex[i]=0;}}

  // Methods.

  // Add or assign chains.
  void addChain(int charge, int flavStart, int flavEnd, bool hasInitialIn);
  void selectResChains(int index,int iorder, int id);
  void selectBeamChains(int index,int iorder);

  // Initialise from hard process information.
  bool initHard(map<int, map<int,int> >& countRes,
    shared_ptr<VinciaMergingHooks> vinMergingHooksPtr);

  // Get information about current status of (pseudo)chains.
  bool checkChains(int cIndex);
  bool checkChains();
  int getNChainsLeft();
  int maxLength();
  int minLength();

  // Print information about (pseudo)chains.
  void print(bool printpsch = false);

  // Members.

  // Chains that arise from the decay of a resonance.
  map<int, vector<PseudoChain> > resChains;

  // Remaining list of ordered chains after resonances stripped off.
  vector<PseudoChain> beamChains;

  // Maps to all viable combinations of chains (of each charge)
  // from an index which depends on the identities of chains in it
  // - will be empty when resChains and beamChains set.
  map< int , vector< PseudoChain > > pseudochains;

  // Easy way to look up all pseudochains which contain a given chain.
  // Map from chain number to pseudochain index.
  map<int,vector<int> > chainToIndices;

  // Map from chain to the flavour at the start or end of that chain.
  map<int,int> chainStartToFlav;
  map<int,int> chainEndToFlav;
  // Keep track of chains which contain an initial state parton.
  map<int,bool> hasInitial;
  // Keep track of charge information of chains.
  map<int,int> chainToCharge;

 private:

  // Add or assign chains.
  void addChain(int oldIndex, int chainsIndex, int iChain,
    vector<int> & newIndices);
  void selectPseudochain(vector<int> & psch);
  void selectChain(int iChain);

  // Create empty vectors in resChains, and count each type.
  void addResonances(vector<int>& idsIn, map<int, map<int,int> >& idCounter,
    int charge, bool fc);

  // Convert charge information to an index.
  int getChargeIndex(int charge, bool fc);

  // Convert pseudochain to unique ID.
  int getID(PseudoChain& psc) {
    int id = 0;
    int iChains = psc.chainlist.size()-1;
    for (int iPwr(iChains); iPwr>=0; --iPwr)
      id += (psc.chainlist.at(iPwr)+1)*pow(10,iChains-iPwr);
    return id;
  }

  // List of found pseudochains to not double count.
  vector<int> pseudochainIDs;

  // Counters.
  int nChains;
  int nBeamChainsMin, nBeamChainsMax, nRes;
  map<int,int> countChainsByChargeIndex;
  map<int,int> countResByChargeIndex;

};

//==========================================================================

// Class for a single step in the history of a process.

class HistoryNode {

 public:

  // Constructors.
  HistoryNode() {};
  HistoryNode(Event& stateIn, vector< vector<int> > chainsIn,
    double scaleIn) : HistoryNode() {
    state = stateIn;
    clusterableChains = chainsIn;
    QevolNow = scaleIn;
    hasRes = false;
    iRes = 0;
    idRes = 0;
    isInitPtr=false;
  };

  // Methods.
  void initPtr(VinciaCommon* vinComPtrIn, Resolution* resPtrIn,
    AntennaSetFSR* antSetPtrIn) {
    resPtr = resPtrIn;
    vinComPtr = vinComPtrIn;
    antSetFSRptr = antSetPtrIn;
    isInitPtr=true;
    nMinQQbar = 0;
  }

  // Set clusterList.
  int getNClusterings(shared_ptr<VinciaMergingHooks> vinMergingHooksPtr,
    Info* infoPtr, int verboseIn);
  void setClusterList(shared_ptr<VinciaMergingHooks> vinMergingHooksPtr,
    Info* infoPtr, int verboseIn);

  // Perform the clusterings according to the resolution criterion.
  bool cluster(HistoryNode& nodeClus,
    Info* infoPtr, int verboseIn);

  // Get energy fractions (used in PDF ratios).
  double xA() const {return 2. * state[3].e() / state[0].e();}
  double xB() const {return 2. * state[4].e() / state[0].e();}

  // Get flavours of beams (used in PDF ratios).
  int idA() const {return state[3].id();}
  int idB() const {return state[4].id();}

  // Get colour types of beams (used in PDF ratios).
  int colTypeA() const {return state[3].colType();}
  int colTypeB() const {return state[4].colType();}

  // Get evolution scale (used in trial shower).
  double getEvolNow() const {return QevolNow;}

  // Setter methods.
  void setEvolScale(double scaleIn) {QevolNow = scaleIn;}

  // Current state.
  Event state;

  // Resonance info.
  bool hasRes;
  int iRes;
  int idRes;

  // Minimal number of qqbar pairs.
  int nMinQQbar;

  // List of unclusterable lists of colour-connected partons.
  vector<vector<int>> clusterableChains;

  // Information corresponding to the last clustering.
  VinciaClustering lastClustering;

 private:

  // Perform a clustering.
  bool doClustering(VinciaClustering& clus, Event& clusEvent,
    vector<vector<int>>& clusChains, Info* infoPtr, int verboseIn);

  // Methods to calculate resolution and evolution scales.
  double calcResolution(VinciaClustering& clusIn) {
    return resPtr->q2sector(clusIn);}
  double calcEvolScale(VinciaClustering& clusIn) {
    return resPtr->q2evol(clusIn);}

  // Members.

  // Vincia pointers.
  Resolution*    resPtr;
  VinciaCommon*  vinComPtr;
  AntennaSetFSR* antSetFSRptr;

  bool isInitPtr;

  // The value of the evolution scale.
  double QevolNow;

  // List of next possible clusterings.
  // Map is from corresponding resolution criterion.
  map<double, VinciaClustering> clusterList;

};
typedef map<int, vector<HistoryNode> > HistoryNodes;

//==========================================================================

// History class for the Vincia shower.

class VinciaHistory {

public:

  // Constructor.
  VinciaHistory(Event &stateIn,
    BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn,
    MergingHooksPtr mergingHooksPtrIn,
    PartonLevel* trialPartonLevelPtrIn,
    ParticleData* particleDataPtrIn,
    Info* infoPtrIn);

  // Check if a valid history was found.
  bool isValid() {return foundValidHistory;}

  // Check if history failed merging scale cut.
  bool isBelowMS() {return failedMSCut;}

  // Perform a trial shower and get the ckkwl weight.
  double getWeightCKKWL();

  // What was the multiplicity of this event?
  int getNClusterSteps();

  // Get the shower starting scale.
  double getRestartScale();

  // Should we overwrite the original event?
  // E.g. if an MPI was generated.
  bool hasNewProcess() {return hasNewProcessSav;}
  Event getNewProcess() {return newProcess;}
  bool doAbort() {return aborted;}

private:

  // Loop over colPerms and set historyBest.
  void findBestHistory();

  // Find all viable colour orderings and return number.
  unsigned int countPerms();

  // Fetch the colour chains from the hard event.
  bool getColChains();

  // Find all selections of chains for colour singlets.
  bool assignResChains(map<int, map<int,int>>& idCounter,
    vector<ColourFlow>& flowsSoFar);

  // Find all selections of chains for colour singlets.
  bool assignResFromEvent(map<int, map<int,int>>& idCounter,
    vector<ColourFlow>& flowsSoFar);

  // Find all selections of chains for everything that is
  // not a colour singlet resonance.
  bool assignBeamChains(vector<ColourFlow>& flowsSoFar);

  // Make a single selection in all possible ways.
  bool assignNext(vector<ColourFlow>& flowsSoFar, bool isRes = false,
    int id = 0, int cIndex = 0);

  // Make a specific selection for a resonance.
  bool assignThis(vector<ColourFlow>& flowsSoFar, int id, int cIndex,
    vector<int>& chains);

  // Is this (completed flow) compatible with the number of
  // resonances of each type?
  bool check(ColourFlow& flow);

  // Construct history for a given colour permutation.
  tuple<bool, double, HistoryNodes> findHistoryPerm(ColourFlow& flow);

  // Check if history failed merging scale cut.
  bool checkMergingCut(HistoryNodes& history);

  // Initialise history nodes for each system.
  HistoryNodes initHistoryNodes(ColourFlow& flow );

  // Translate abstract book-keeping of colourordering into
  // systems of particles.
  map<int, vector<vector<int>>> getSystems(ColourFlow& flow,
    map<int, int>& sysToRes);

  // Decide if state is the Born topology.
  bool isBorn(const HistoryNode& nodeIn, bool isRes);

  // Set up beams for given history node.
  bool setupBeams(const HistoryNode* node, double scale2);

  // Calculate criterion for testing whether to keep history.
  double calcME2guess(vector<HistoryNode>& history,bool isRes);
  // Calculate ME for Born-level event.
  double calcME2Born(const HistoryNode& bornNode, bool isRes);

  // Calculate the antenna function for a given clustering.
  double calcAntFun(const VinciaClustering& clusNow);

  // Calculate PDF ratio to multiply CKKW-L weight.
  double calcPDFRatio(const HistoryNode* nodeNow,
    double pT2now, double pT2next);

  // Calculate alphaS ratio to multiply CKKW-L weight.
  double calcAlphaSRatio(const HistoryNode& node);

  // Return the kinematic maximum for this event.
  double getStartScale(Event& event, bool isRes);

  // Perform a trial branching and return scale.
  double qNextTrial(double qStart, Event& evtIn);

  // Print colour chains.
  void printChains();

  // Verbosity.
  int verbose;

  // Beams -- for PDFs.
  BeamParticle beamA, beamB;

  // MergingHooks.
  shared_ptr<VinciaMergingHooks> vinMergingHooksPtr{};

  // PartonLevel pointer.
  PartonLevel* trialPartonLevel{};

  // Particle data pointer.
  ParticleData* particleDataPtr{};

  // Other Pythia pointers.
  Info* infoPtr{};

  // Vincia pointers.
  shared_ptr<VinciaFSR> fsrShowerPtr{};
  shared_ptr<VinciaISR> isrShowerPtr{};
  MECs*                 mecsPtr{};
  VinciaCommon*         vinComPtr{};
  Resolution*           resPtr{};
  AntennaSetFSR*        antSetFSRptr{};

  // Check we found a valid history at all.
  bool foundValidHistory;

  // Check we vetoed due failing the merging scale cut.
  bool failedMSCut;

  // This is the best PS history so far.
  HistoryNodes historyBest;

  // Criterion to minimise (best so far).
  // Call calcME2guess(..) to evaluate.
  double ME2guessBest;

  // Initial colour permutation, stored as
  // list of sets of colour connected partons.
  vector<vector<int>> colChainsSav;

  // Track if chain contains initial state quarks.
  map<int, bool> chainHasInitial;

  // Keep track of resonances in the event record.
  map<int, vector<int>> resIDToIndices;
  map<int, vector<int>> resIndexToChains;

  // All colour flows compatible with Born process.
  vector<ColourFlow> colPerms;

  // ME generated event.
  Event state;

  // Number of quarks  and gluon loops in event.
  int nQSave, nGluonLoopsSave;

  // The merging scale and whether it is the evolution variable.
  double qms;
  bool msIsEvolVar;

  // The maximum multiplicity of our me-generator.
  int nMax, nMaxRes;

  // Possible new hard process info (if MPI was generated).
  bool hasNewProcessSav;
  Event newProcess;
  double newProcessScale;

  // Flag to signal if something went wrong.
  bool aborted;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_VinciaHistory_H
