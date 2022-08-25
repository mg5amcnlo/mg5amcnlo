// VinciaMergingHooks.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand, Peter Skands.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUidELINES for details.

// This file was authored by Helen Brooks, Christian T Preuss.

#ifndef Pythia8_VinciaMergingHooks_H
#define Pythia8_VinciaMergingHooks_H

#include "Pythia8/MergingHooks.h"
#include "Pythia8/PartonLevel.h"
#include "Pythia8/UserHooks.h"
#include "Pythia8/VinciaCommon.h"

namespace Pythia8 {

//==========================================================================

// Storage device for multiparticle.

struct MultiParticle {
  vector<int> pidList;
  vector<int> coltypes;
  // id, if it has a unique one, otherwise 0.
  int id;
  // QED charge,if it has a unique one, otherwise 999.
  int charge;
  bool isRes, isFCN;
};


// Helper structure to identify where to find a given
// hard process particle inside HardProcessParticleList.

struct ParticleLocator {
  // Increment level for every resonance decay.
  int level;
  // Position in vector at a given level.
  int pos;
};


//==========================================================================

// Class to store information about a particle in the hard process.

class HardProcessParticleList;
class HardProcessParticle {

  friend class HardProcessParticleList;

 public:

  // Construct from particle data.
  HardProcessParticle(int idIn, ParticleDataEntryPtr pdata,
    ParticleLocator locIn, HardProcessParticleList* listPtrIn,
    vector<ParticleLocator>& mothersIn) :
    isMultiparticle(false), pid(idIn), multiPtr(nullptr),
    loc(locIn), listPtr(listPtrIn), mothers(mothersIn) {
    isResSav = pdata->isResonance(); coltype = pdata->colType(pid);
    charge = pdata->chargeType(pid); isColSav = coltype != 0;
    nameSav = pdata->name(pid);}

  // Construct from multiparticle.
  HardProcessParticle(string nameIn, const MultiParticle* multiPtrIn,
    ParticleLocator locIn, HardProcessParticleList * listPtrIn,
    vector<ParticleLocator> & mothersIn) :
    isMultiparticle(true), nameSav(nameIn), multiPtr(multiPtrIn),
      loc(locIn), listPtr(listPtrIn), mothers(mothersIn) {
    pid = multiPtr->id; charge = multiPtr->charge;
    isResSav = multiPtr->isRes;
    coltype = multiPtr->coltypes.size() != 0 ? multiPtr->coltypes.at(0) : 0;
    isColSav = coltype !=0;}

  // Check if final.
  bool isFinal() const {return daughters.size() == 0;}

  // Check if beam particle.
  bool isBeam() const {return loc.level == 0;}

  // Check if intermediate particle.
  bool isIntermediate() const {return !isBeam() && !isFinal();}

  // Getter methods.
  bool isRes() const {return isResSav;}
  bool isCol() const {return isColSav;}
  bool isMulti() const { return isMultiparticle;}
  string name() const {return nameSav;}
  int id() const {return pid;}
  int chargeType() const {return charge;}
  int colType() const {return coltype;}
  vector<ParticleLocator> getDaughters() const {return daughters;}
  const MultiParticle* getMulti() const {return multiPtr;}

  // Print the particle.
  void print() const;

 private:

  bool isMultiparticle;
  bool isResSav;
  bool isColSav;
  string nameSav;

  // pid and coltype if not multiparticle.
  int pid;
  int coltype;

  // QED charge if it has a unique value (even if multiparticle).
  int charge;

  // Pointer to a multiparticle if is one,
  // null pointer otherwise.
  const MultiParticle* multiPtr;

  // Location of this particle - only used by particle list.
  ParticleLocator loc;
  // Pointer to the list in which this particle lives.
  HardProcessParticleList* listPtr;

  // Location of this particle's mother(s).
  vector<ParticleLocator> mothers;

  // Location of this particle's daughter(s).
  vector<ParticleLocator> daughters;

};

//==========================================================================

// List of hard particles.

class HardProcessParticleList {

 public:

  // List the hard particles.
  void list() const;

  // Fetch pointers to the beams.
  pair<HardProcessParticle*, HardProcessParticle*> getBeams() {
    HardProcessParticle* beamAPtr{nullptr}, *beamBPtr{nullptr};
    if (!particles.empty() && particles[0].size() == 2) {
      beamAPtr = &particles[0][0]; beamBPtr = &particles[0][1];}
    return make_pair(beamAPtr,beamBPtr);}

  // Fetch pointer to particles at i-th level.
  vector<HardProcessParticle>* getLevel(int i) {
    if (particles.find(i) != particles.end()) return &particles[i];
    else return nullptr;}

  // Get a single particle, given a location.
  HardProcessParticle* getPart(ParticleLocator loc) {
    if (particles.find(loc.level) != particles.end() &&
        int(particles[loc.level].size()) > loc.pos)
      return &particles[loc.level].at(loc.pos);
    return nullptr;}

  // Add multiparticle to list.
  ParticleLocator add(int level, string nameIn, const MultiParticle* multiPtr,
    vector<ParticleLocator>& mothersIn);

  // Add particle to list from data.
  ParticleLocator add(int level, int idIn, ParticleDataEntryPtr pdata,
    vector<ParticleLocator>& mothersIn);

  // Set daughters of particle at mother location.
  void setDaughters(ParticleLocator& mother,
    vector<ParticleLocator>& daughters);

  // Get the next location.
  ParticleLocator getNextLoc(int level) {
   // Does this level exist yet? Create.
   if (particles.find(level) == particles.end())
     particles[level] = vector<HardProcessParticle>();
   ParticleLocator loc; loc.level = level; loc.pos = particles[level].size();
   return loc;}

 private:

  // This is the list of all particles, by level (each nested decay
  // creates a new level).
  map<int, vector<HardProcessParticle>> particles;

};

//==========================================================================

// Storage device for containing colour structure of hard process.

struct ColourStructure {
  // Pointers to beam particles.
  HardProcessParticle* beamA{};
  HardProcessParticle* beamB{};

  // Pointers to coloured partons, leptons and resonances.
  vector<HardProcessParticle*> coloured;
  vector<HardProcessParticle*> leptons;

  // IDs of hadronically decaying resonances.
  vector<int> resPlusHad;
  vector<int> resMinusHad;
  vector<int> resNeutralFCHad;
  vector<int> resNeutralFNHad;

  // IDs of leptonically decaying resonances.
  vector<int> resPlusLep;
  vector<int> resMinusLep;
  vector<int> resNeutralFCLep;
  vector<int> resNeutralFNLep;

  // IDs of charged undecayed resonances
  // (only for hard process specification when MergeInResSystems = off).
  vector<int> resPlusUndecayed;
  vector<int> resMinusUndecayed;
  vector<int> resNeutralUndecayed;

  // Counters for partons (after all col res decayed).
  int nQQbarPairs{0};
  int nColoured{0};

  // Minimum and maximum number of chains associated with beams.
  int nMinBeamChains{0};
  int nMaxBeamChains{0};
};

//==========================================================================

// Container for the hard process used in Vincia merging.

class VinciaHardProcess : public HardProcess {

 public:

  // Constructor.
  VinciaHardProcess(Info* infoPtrIn, int verboseIn, bool resolveDecaysIn,
    bool doHEFTIn, bool doVBFIn) :
    verbose(verboseIn), infoPtr(infoPtrIn), resolveDecays(resolveDecaysIn),
      doHEFT(doHEFTIn), doVBF(doVBFIn), isInit(false) {defineMultiparticles();}

  // Initialise process.
  void initOnProcess(string process, ParticleData* particleData) override;

  // Redundant inherited methods - only dummy versions here.
  void storeCandidates(const Event&, string) override {;}
  bool matchesAnyOutgoing(int, const Event&) override {return false;}
  bool findOtherCandidates(int, const Event&, bool) override {return false;}

  // Print functions.
  void list() const {parts.list();}
  void listLookup() const;

  // Set verbosity.
  void setVerbose(int verboseIn) {verbose = verboseIn;}

  // Check if initialised.
  bool initSuccess() {return isInit;}

  // Return the colour structure of the hard process.
  void getColourStructure(ColourStructure& colStructNow);

private:

  // Initialised multiparticle definitions.
  void defineMultiparticles();
  // Check if ID may be a beam particle.
  bool isBeamID(int id);
  // Initialise map of names to IDs.
  void initLookup(ParticleData* particleData);

  // Split process string into incoming, outgoing using ">".
  bool splitProcess(string process, vector<string>& inWords,
    vector<string>& outWords);
  // Split string into words by spaces, and append to the back
  // (or front) of wordsOut.
  void splitbyWhitespace(string wordIn, vector<string>& wordsOut,
    bool atFront = false);

  // Set the list of particles in the hard process.
  bool getParticles(ParticleData* particleDataPtr,
    vector<string> inWords,
    vector<string> outWords);

  // Recursive version (if decays are found).
  bool getParticles(ParticleData* particleDataPtr,
    vector<string> inWords,
    vector<string> outWords,
    int levelNow,
    vector<ParticleLocator>& mothersIn,
    vector<ParticleLocator>& mothersNow);

  // Add a particle to list, and save location in loc if successful.
  bool addParticle(ParticleData* particleDataPtr,int level, bool isIncoming,
    string name, vector<ParticleLocator>& mothersIn, ParticleLocator& loc);

  // Set daughters of particle at mother location.
  void setDaughters(ParticleLocator& mother,
    vector<ParticleLocator>& daughters) {parts.setDaughters(mother,daughters);}

  // Verbosity.
  int verbose;

  // Pointer to info in Pythia.
  Info* infoPtr{};

  // Flag to control how resonances are handled.
  bool resolveDecays;

  // Flags to control how HEFT and VBF are handled.
  bool doHEFT, doVBF;

  // Provide a way to look up a particle data entry by its name.
  map<string, int> lookupIDfromName;

  // Store neutral flavour-changing resonances.
  map<int, bool> isFCNres;

  // Provide a way to define multparticles.
  map<string, MultiParticle> multiparticles;

  // Main way of storing the hard process particles.
  HardProcessParticleList parts;

  // Did initialisation succeed?
  bool isInit;

};

//==========================================================================

// Class for Vincia to perform merging.

class VinciaMergingHooks : public MergingHooks {

 public:

  // Constructor.
  VinciaMergingHooks() : vinHardProcessPtr(nullptr), isInit(false) {;}
  // Destructor.
  ~VinciaMergingHooks() {if (hardProcess) delete hardProcess;}

  // Initialise.
  void init() override;

  // Set starting scales.
  bool setShowerStartingScales(bool isTrial, bool,
    double& pTscaleIn, const Event& event, double& pTmaxFSRIn, bool&,
    double& pTmaxISRIn, bool&, double& pTmaxMPIIn, bool&) override;

  // This MergingHooks is for Vincia only.
  virtual bool usesVincia() override {return true;}

  // Calculate merging scale of current state.
  virtual double tmsNow(const Event& event) override;

  // Check whether an event should be vetoed due to branching above tMS.
  virtual bool canVetoStep() override;
  virtual bool doVetoStep(const Event& process, const Event& event, bool)
    override;

  // Overridden base class methods.
  virtual bool doVetoEmission(const Event&) override {return false;}
  virtual bool canVetoEmission() override {return false;}
  virtual double dampenIfFailCuts(const Event& ) override {return 0.;}
  virtual int getNumberOfClusteringSteps(const Event&, bool) override {
    return 0;}
  virtual bool canCutOnRecState() override {return false;}
  virtual bool doCutOnRecState(const Event&) override {return false;}
  virtual bool canVetoTrialEmission() override {return false;}
  virtual bool doVetoTrialEmission(const Event&, const Event&) override {
    return false;}
  virtual bool useShowerPlugin() override {return false;}
  virtual double hardProcessME(const Event&) override {return 0;}

  // Others.
  virtual double tmsDefinition( const Event&) override {return 0.;}

  // Set and get verbosity.
  void setVerbose(int verboseIn) {verbose = verboseIn;}
  int getVerbose() {return verbose;}

  // Check if initialisation succeeded.
  bool initSuccess() {return isInit;}

  // Get list of leptons in the hard process.
  vector<HardProcessParticle*> getLeptons() {return colStructSav.leptons;}

  // Get number of undecayed resonances.
  int getNResPlusUndecayed() {
    return int(colStructSav.resPlusUndecayed.size());}
  int getNResMinusUndecayed() {
    return int(colStructSav.resMinusUndecayed.size());}
  int getNResNeutralUndecayed() {
    return int(colStructSav.resNeutralUndecayed.size());}

  // Get list of undecayed resonances in the hard process.
  vector<int> getResPlusUndecayed() {
    return colStructSav.resPlusUndecayed;}
  vector<int> getResMinusUndecayed() {
    return colStructSav.resMinusUndecayed;}
  vector<int> getResNeutralUndecayed() {
    return colStructSav.resNeutralUndecayed;}

  // Get list of leptonically decaying resonances in the hard process.
  vector<int> getResPlusLep() {return colStructSav.resPlusLep;}
  vector<int> getResMinusLep() {return colStructSav.resMinusLep;}
  vector<int> getResNeutralFCLep() {return colStructSav.resNeutralFCLep;}
  vector<int> getResNeutralFNLep() {return colStructSav.resNeutralFNLep;}

  // Get list of hadronically decaying resonances in the hard process.
  vector<int> getResPlusHad() {return colStructSav.resPlusHad;}
  vector<int> getResMinusHad() {return colStructSav.resMinusHad;}
  vector<int> getResNeutralFCHad() {return colStructSav.resNeutralFCHad;}
  vector<int> getResNeutralFNHad() {return colStructSav.resNeutralFNHad;}

  // Get number of hadronically decaying resonances in the hard process.
  int getNResPlusHad() {return colStructSav.resPlusHad.size();}
  int getNResMinusHad() {return colStructSav.resMinusHad.size();}
  int getNResNeutralFCHad() {return colStructSav.resNeutralFCHad.size();}
  int getNResNeutralFNHad() {return colStructSav.resNeutralFNHad.size();}
  int getNResHad() {return getNResPlusHad() + getNResMinusHad() +
      getNResNeutralFCHad() + getNResNeutralFNHad();}

  // Get numbers of (either hadronically or leptonically decaying)
  // resonances in the hard process.
  int getNResPlus() {return colStructSav.resPlusHad.size() +
      colStructSav.resPlusLep.size();}
  int getNResMinus() {return colStructSav.resMinusHad.size() +
      colStructSav.resMinusLep.size();}
  int getNResNeutralFC() {return colStructSav.resNeutralFCHad.size() +
      colStructSav.resNeutralFCLep.size();}
  int getNResNeutralFN() {return colStructSav.resNeutralFNHad.size() +
      colStructSav.resNeutralFNLep.size();}

  // Get information about the beam chains.
  int getNChainsMin() {return colStructSav.nMinBeamChains;}
  int getNChainsMax() {return colStructSav.nMaxBeamChains;}
  int getNPartons() {return colStructSav.nColoured;}
  int getNQPairs() {return colStructSav.nQQbarPairs;}

  // Get informations about whether colour structure has been set yet.
  bool hasSetColourStructure() {return hasColStruct;}

  // Check if we are merging in resonance systems.
  bool canMergeRes() {return doMergeRes;}

  // Check if we are merging in VBF system.
  bool doMergeInVBF() {return doVBF;}

  // Check if we are allowing HEFT couplings.
  bool allowHEFT() {return doHEFT;}

  // Get maximum number of additional jets from resonance decay systems.
  int nMaxJetsRes() {return nJetMaxResSave;}

  // Fetch shower restarting scale for resonances.
  void setScaleRes(int iRes, double scale) {resSysRestartScale[iRes] = scale;}

  // Fetch shower starting scale for resonances.
  double getScaleRes(int iRes, const Event&) {
    return resSysRestartScale.find(iRes) != resSysRestartScale.end() ?
      resSysRestartScale[iRes] : tmsCut();}

  // Check if clusterings are allowed.
  bool canClusFF() {return doFF;}
  bool canClusRF() {return doRF;}
  bool canClusII() {return doII;}
  bool canClusIF() {return doIF;}

  // Veto showered event if branching is above merging scale.
  bool isAboveMS(const Event& event);

  // Same as HardProcess, but explicitly of VinciaHardProcess type.
  // Note both point to the same object.
  VinciaHardProcess* vinHardProcessPtr{};

 protected:

  // Maximal number of additional jets per resonance system.
  int nJetMaxResSave;

  // Number of resonance systems allowed to produce additional jets.
  int nMergeResSys;

  // Flag to decide if we can merge in resonance systems.
  bool doMergeRes;

  // Tell Vincia whether to veto emissions from non-resonance systems.
  bool doVetoNotInResSav;

  // Saved information about resonance restart scales.
  map<int,double> resSysRestartScale;

 private:

  // Merging scale implementations.
  double kTmin(const Event& event);
  vector<double> cutsMin(const Event& event);

  // Find the colour structure of the hard process.
  ColourStructure getColourStructure();
  bool setColourStructure();
  void printColStruct();

  // Check whether a particle is a resonance decay product.
  bool isResDecayProd(int iPtcl, const Event& event);

  // Get list of jets in event record according to jet definition.
  vector<int> getJetsInEvent(const Event& event);

  // Did we suceed in initialising?
  bool isInit;

  // Verbosity.
  int verbose;

  // Colour strucuture of the hard process.
  bool hasColStruct;
  ColourStructure colStructSav;

  // Flags to turn on/off FSR or ISR.
  bool doFF, doRF, doII, doIF;

  // Flags to turn on/off specific event topologies.
  bool doHEFT{false}, doVBF{false};

};

//==========================================================================

// Mini UserHooks class for setting the scale of main shower in
// resonance systems (if doing merging in these).

class MergeResScaleHook : public UserHooks {

public:

  // Constructor.
  MergeResScaleHook( MergingHooksPtr mergingHooksPtrIn) {
    // Cast as a VinciaMergingHooks object.
    vinMergingHooksPtr
      = dynamic_pointer_cast<VinciaMergingHooks>(mergingHooksPtrIn);
    if (vinMergingHooksPtr == nullptr || !vinMergingHooksPtr->initSuccess() )
      canMergeRes = false;
    else canMergeRes = vinMergingHooksPtr->canMergeRes();}

  // Start resonance showers at a scale of m.
  bool canSetResonanceScale() override {return canMergeRes;}
  double scaleResonance(int iRes, const Event& event) override {
    return vinMergingHooksPtr->getScaleRes(iRes,event);}

 private:

  bool canMergeRes;
  shared_ptr<VinciaMergingHooks> vinMergingHooksPtr;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_MergingHooks_H
