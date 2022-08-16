// VinciaFSR.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains header information for the VinciaFSR class for
// QCD final-state antenna showers, and auxiliary classes.

#ifndef Pythia8_VinciaFSR_H
#define Pythia8_VinciaFSR_H

#include "Pythia8/TimeShower.h"
#include "Pythia8/VinciaAntennaFunctions.h"
#include "Pythia8/VinciaCommon.h"
#include "Pythia8/VinciaISR.h"
#include "Pythia8/VinciaMergingHooks.h"
#include "Pythia8/VinciaTrialGenerators.h"
#include "Pythia8/VinciaQED.h"
#include "Pythia8/VinciaEW.h"
#include "Pythia8/VinciaWeights.h"
#include "Pythia8/VinciaDiagnostics.h"

namespace Pythia8 {

// Forward declarations.
class VinciaISR;

//==========================================================================

// Helper struct to store information about junctions that involved
// resonances that have now decayed.

struct ResJunctionInfo {

  // Number of junction in event record.
  int iJunction;
  // Which of the three ends of iJunction are we talking about.
  int iEndCol;
  // What is the colour tag of iEndCol in iJunctions.
  int iEndColTag;
  // Pythia index of quark that should be on the end. Specifically it
  // should be the quark from the decay of the resonance.
  int iEndQuark;
  // iEndColTag and iEnd Quark will get updated as the quark radiates.
  // Cector of colours lines between res and end quark, needed in
  // order to track if a gluon in chain splits to find correct
  // junction end.
  vector<int> colours;

};

//==========================================================================

// The Brancher class, base class containing a generic set of "parent
// partons" as well as virtual methods for generating trial
// branchings.

class Brancher {

public:

  // Main base class constructor.
  Brancher(int iSysIn, Event& event, bool sectorShowerIn,
vector<int> iIn) : sectorShower(sectorShowerIn) {
    reset(iSysIn, event, iIn);
  }

  // Wrapper for 2- and 3-parton parents.
  Brancher(int iSysIn, Event& event, bool sectorShowerIn,
    int iIn0, int iIn1, int iIn2=0) : sectorShower(sectorShowerIn) {
    reset(iSysIn, event, iIn0, iIn1, iIn2);
  }

  // Base class must have a virtual destructor.
  virtual ~Brancher() {}

  // Reset (common functionality implemented in base class).
  void reset(int iSysIn, Event& event, vector<int> iIn);

  // Wrapper for simple 2- (or 3-) parton antennae.
  void reset(int iSysIn, Event& event, int i0In, int i1In, int i2In = 0) {
    vector<int> iIn {i0In, i1In}; if (i2In >= 1) iIn.push_back(i2In);
    reset(iSysIn,event,iIn);}

  // Methods to get (explicit for up to 3 parents, otherwise just use iVec).
  int i0() const {return (iSav.size() >= 1) ? iSav[0] : -1;}
  int i1() const {return (iSav.size() >= 2) ? iSav[1] : -1;}
  int i2() const {return (iSav.size() >= 3) ? iSav[2] : -1;}
  int iVec(unsigned int i) const {return (iSav.size() > i) ? iSav[i] : -1;}
  vector<int> iVec() {return iSav;}
  int id0() const {return (idSav.size() >= 1) ? idSav[0] : -1;}
  int id1() const {return (idSav.size() >= 2) ? idSav[1] : -1;}
  int id2() const {return (idSav.size() >= 3) ? idSav[2] : -1;}
  vector<int> idVec() const {return idSav;}
  int colType0() const {return (colTypeSav.size() >= 1) ? colTypeSav[0] : -1;}
  int colType1() const {return (colTypeSav.size() >= 2) ? colTypeSav[1] : -1;}
  int colType2() const {return (colTypeSav.size() >= 3) ? colTypeSav[2] : -1;}
  vector<int> colTypeVec() const {return colTypeSav;}
  int col0() const {return (colSav.size() >= 1) ? colSav[0] : 0;}
  int col1() const {return (colSav.size() >= 2) ? colSav[1] : 0;}
  int col2() const {return (colSav.size() >= 3) ? colSav[2] : 0;}
  vector<int> colVec() const {return colSav;}
  int acol0() const {return (acolSav.size() >= 1) ? acolSav[0] : 0;}
  int acol1() const {return (acolSav.size() >= 2) ? acolSav[1] : 0;}
  int acol2() const {return (acolSav.size() >= 3) ? acolSav[2] : 0;}
  vector<int> acolVec() const {return acolSav;}
  int h0() const {return (hSav.size() >= 1) ? hSav[0] : -1;}
  int h1() const {return (hSav.size() >= 2) ? hSav[1] : -1;}
  int h2() const {return (hSav.size() >= 3) ? hSav[2] : -1;}
  vector<int> hVec() const {return hSav;}
  double m0() const {return (mSav.size() >= 1) ? mSav[0] : -1;}
  double m1() const {return (mSav.size() >= 2) ? mSav[1] : -1;}
  double m2() const {return (mSav.size() >= 3) ? mSav[2] : -1;}
  vector<double> mVec() const {return mSav;}
  vector<double> getmPostVec() {return mPostSav;}
  int colTag() {return colTagSav;}

  // Method to get maximum value of evolution scale for this brancher.
  // Must be redefined in base classes.
  virtual double getQ2Max(int) = 0;

  // Methods to return saved/derived quantities.
  int    system()     const {return systemSav;}
  double mAnt()       const {return mAntSav;}
  double m2Ant()      const {return m2AntSav;}
  double sAnt()       const {return sAntSav;}
  double kallenFac()  const {return kallenFacSav;}
  double enhanceFac() const {return enhanceSav;}
  double q2Trial()    const {return q2NewSav;}
  enum AntFunType antFunTypePhys()   const {return antFunTypeSav;}

  // Generate a new Q2 scale.
  virtual double genQ2(int evTypeIn, double Q2MaxNow, Rndm* rndmPtr,
    Info* infoPtr, const EvolutionWindow* evWindowPtrIn, double colFac,
    vector<double> headroomIn, vector<double> enhanceFacIn,
    int verboseIn) = 0;

  // Generate complementary invariant(s) for saved trial. Return false
  // if no physical kinematics possible. Base class returns false.
  virtual bool genInvariants(vector<double>& invariants, Rndm*, int,
    Info* /*infoPtr*/) {
    invariants.clear(); return false;}

  // Compute antPhys/antTrial, given an input value for antPhys. Base
  // class returns 0.
  virtual double pAccept(const double, Info* /*infoPtr*/, int = 0) {return 0.;}

  // Compute pT scale of trial branching.
  virtual double getpTscale();

  // Return Xj.
  virtual double getXj();

  // What kind of particle(s) are being created, e.g. return 21 for
  // gluon emission, quark flavour for splitting, etc.
  virtual int idNew() const {return 0;}
  virtual double mNew() const {return 0.0;}

  // Return new particles, must be implemented by derived class.
  virtual bool getNewParticles(Event& event, vector<Vec4> momIn,
    vector<int> hIn, vector<Particle> &pNew,Rndm* rndmPtr,
    VinciaColour* colourPtr) = 0;

  // Simple print utility, showing the contents of the Brancher. Base
  // class implementation allows for up to three explicit parents.
  virtual void list(string header="none", bool withLegend=true) const;

  // Set post-branching IDs and masses. Base class is for gluon emission.
  virtual void setidPost();
  virtual vector<double> setmPostVec();
  virtual void setStatPost();
  virtual void setMaps(int);

  // Return index of new particle (slightly arbitrary choice for splittings).
  virtual int iNew();

  // Method returns pos of resonance if there is one participating in
  // decay, -1 otherwise.
  virtual int posR() const {return -1;}

  // Method returns pos of colour-connected daughter to resonance if
  // there is one participating in decay, -1 otherwise.
  virtual int posF() const {return -1;}

  // Return sector label.
  virtual int getSector(){
    return iSectorWinner;
  };

  // Return branch type.
  enum BranchType getBranchType() {return branchType;}
  // Check if swapped.
  bool isSwapped() {return swapped;}
  // Return the saved invariants.
  vector<double> getInvariants() {return invariantsSav;}

  // This method allows to reset enhanceFac if we do an accept/reject.
  void resetEnhanceFac(const double enhanceIn) {enhanceSav = enhanceIn;}

  // Method to see if a saved trial has been generated, and to erase
  // memory of it.
  bool hasTrial() const {return hasTrialSav;}
  // Method to mark new trial needed *without* erasing current one.
  void needsNewTrial() {
    hasTrialSav = false;
    if (trialGenPtr != nullptr) {
      trialGenPtr->needsNewTrial();
    }
  }
  // Method to mark new trial needed *and* erase current one.
  void eraseTrial() {
    hasTrialSav = false;
    q2NewSav = 0.;
    if (trialGenPtr != nullptr) {
      trialGenPtr->resetTrial();
    }
  }
  // Object to perform trial generation.
  shared_ptr<TrialGenerator> trialGenPtr = {};

  // Publicly accessible members for storing mother/daughter connections.
  map<int, pair<int, int> > mothers2daughters;
  map<int, pair<int, int> > daughters2mothers;

protected:

  // Data members for storing information about parent partons.
  int systemSav{};
  vector<int> iSav, idSav, colTypeSav, hSav, colSav, acolSav;
  vector<int> idPostSav, statPostSav;
  vector<double> mSav, mPostSav;
  int colTagSav{}, evTypeSav{};

  // All alphaS information.
  const EvolutionWindow* evWindowSav{};

  // Saved antenna mass parameters.
  double mAntSav{}, m2AntSav{}, kallenFacSav{}, sAntSav{};

  // Data members for storing information about generated trial branching.
  bool   hasTrialSav{false};
  double headroomSav{1.}, enhanceSav{1.}, q2BegSav{}, q2NewSav{};
  vector<double> invariantsSav;

  // Store which branching type we are doing.
  enum BranchType branchType{BranchType::Void};

  // Index of FF antenna function.
  enum AntFunType antFunTypeSav{NoFun};

  // If true, flip identities of A and B.
  bool swapped{false};

  // Parameters for the sector shower.
  bool sectorShower{};
  int iSectorWinner{};

};

//==========================================================================

// Class BrancherEmitFF, branch elemental for 2->3 gluon emissions.

class BrancherEmitFF : public Brancher {

public:

  // Create branch elemental for antenna(e) with parents in iIn.
  BrancherEmitFF(int iSysIn, Event& event, bool sectorShowerIn,
    vector<int> iIn, ZetaGeneratorSet& zetaGenSet) :
    Brancher(iSysIn, event, sectorShowerIn, iIn) {
    // Initialise derived-class members and set up trial generator.
    initBrancher(zetaGenSet);
  }

  // Wrapper to provide simple 2-parton systems as parents.
  BrancherEmitFF(int iSysIn, Event& event, bool sectorShowerIn,
    int iIn0, int iIn1, ZetaGeneratorSet& zetaGenSet) :
    Brancher(iSysIn, event, sectorShowerIn, iIn0, iIn1) {
    // Initialise derived-class members and set up trial generator.
    initBrancher(zetaGenSet);
  }

  // Method to initialise members specific to BrancherEmitFF.
  void initBrancher(ZetaGeneratorSet& zetaGenSet);

  // Generate a new Q2 value, soft-eikonal 2/yij/yjk implementation.
  double genQ2(int evTypeIn, double Q2MaxNow, Rndm* rndmPtr,
    Info* infoPtr, const EvolutionWindow* evWindowPtrIn, double colFac,
    vector<double> headroomIn, vector<double> enhanceFacIn,int verboseIn);

  // Generate invariants. Method to generate complementary
  // invariant(s) for saved trial scale for gluon emission. Return
  // false if no physical kinematics possible.
  virtual bool genInvariants(vector<double>& invariants, Rndm* rndmPtr,
    int verboseIn, Info* /*infoPtr*/);

  // Compute antPhys/antTrial for gluon emissions, given
  // antPhys. Note, antPhys should be normalised to include charge and
  // coupling factors.
  virtual double pAccept(const double antPhys, Info* infoPtr, int verboseIn);

  // Return the maximum Q2.
  double getQ2Max(int evType);

  // Method to make mothers2daughters and daughters2mothers pairs.
  virtual void setMaps(int sizeOld);

  // Flavour and mass of emitted particle
  virtual int idNew() const {return 21;}
  virtual double mNew() const {return 0.0;}

  // Generic getter method. Assumes setter methods called earlier.
  virtual bool getNewParticles(Event& event, vector<Vec4> momIn,
    vector<int> hIn, vector<Particle> &pNew, Rndm* rndmPtr,
    VinciaColour* colourPtr);

private:

  // Data members to store information about generated trials.
  double colFacSav{};

};

//==========================================================================

// Class BrancherSplitFF, branch elemental for 2->3 gluon splittings.

class BrancherSplitFF : public Brancher {

public:

  // Create branch elemental for antenna(e) with parents in iIn.
  BrancherSplitFF(int iSysIn, Event& event, bool sectorShowerIn,
    vector<int> iIn, ZetaGeneratorSet& zetaGenSet) :
    Brancher(iSysIn, event, sectorShowerIn, iIn) {
    // Initialise derived-class members and set up trial generator.
    initBrancher(zetaGenSet);
  }

  // Wrapper to provide simple 2-parton systems as parents. Set if it
  // is the anticolour or colour side of the gluon that participates
  // in the antenna (used to decide pTj or pTi measure).
  BrancherSplitFF(int iSysIn, Event& event, bool sectorShowerIn, int iIn0,
    int iIn1, bool col2acol, ZetaGeneratorSet& zetaGenSet) :
    Brancher(iSysIn, event, sectorShowerIn, iIn0, iIn1) {
    // Initialise derived-class members and set up trial generator.
    initBrancher(zetaGenSet, col2acol);
  }

  // Method to initialise data members specific to BrancherSplitFF.
  void initBrancher(ZetaGeneratorSet& zetaGenSet, bool col2acolIn=true);

  // Method to check if it is the colour (false) or anticolour (true)
  // side of the gluon that is splitting, corresponding to the quark
  // or antiquark being regarded as the emitted parton respectively.
  virtual bool isXG() const {return isXGsav;}

  // Flavour and mass of emitted particle.
  virtual int idNew() const {return idFlavSav;}
  virtual double mNew() const {return mFlavSav;}

  // Generate a new Q2 scale (collinear 1/(2q2) implementation) with
  // constant trial alphaS.
  double genQ2(int evTypeIn, double Q2MaxNow, Rndm* rndmPtr,
    Info* infoPtr, const EvolutionWindow* evWindowPtrIn, double colFac,
    vector<double> headroomIn, vector<double> enhanceFacIn,int verboseIn);

  // Generate complementary invariant(s) for saved trial scale for
  // gluon splitting. Return false if no physical kinematics possible.
  virtual bool genInvariants(vector<double>& invariants, Rndm* rnmdPtr,
    int verboseIn, Info* /*infoPtr*/);

  // Compute antPhys/antTrial for gluon splittings, given antPhys.
  // Note, antPhys should be normalised to include charge and coupling
  // factors.
  virtual double pAccept(const double antPhys, Info* infoPtr, int verboseIn);

  // Getter and setter methods.
  double getQ2Max(int evType);
  virtual vector<double> setmPostVec();
  virtual void setidPost();
  virtual void setStatPost();
  virtual void setMaps(int sizeOld);

  // Generic getter method. Assumes setter methods called earlier.
  virtual bool getNewParticles(Event& event, vector<Vec4> momIn,
    vector<int> hIn, vector<Particle> &pNew, Rndm*, VinciaColour*);

 private:

  // Data members for storing information on generated trials.
  int     idFlavSav{};
  double  mFlavSav{};

  // Data member to store whether this is a GXbar (false) or XG (true) antenna,
  // i.e. if it is the colour side (false) of the gluon or the anticolour
  // side which is participating in the LC antenna. In the former case, we
  // use pT(q) as the measure, else pT(qbar) = pTj.
  bool isXGsav{};

};

//==========================================================================

// BrancherRF base class for resonance-final antennae.

class BrancherRF : public Brancher {

public:

  // Base class constructor = inherited from Brancher.
  BrancherRF(int iSysIn, Event& event, bool sectorShowerIn,
    vector<int> allIn) : Brancher(iSysIn, event, sectorShowerIn, allIn) {}

  // RF branchers have a different initBrancher structure.
  virtual void initRF(Event& event, vector<int> allIn,
    unsigned int posResIn, unsigned int posFIn, double Q2cut,
    ZetaGeneratorSet& zetaGenSet) = 0;

  // Reset the brancher.
  void resetRF(int iSysIn, Event& event, vector<int> allIn,
    unsigned int posResIn, unsigned int posFIn, double Q2cut,
    ZetaGeneratorSet& zetaGenSet) {
    reset(iSysIn, event, allIn);
    initRF(event, allIn, posResIn, posFIn, Q2cut, zetaGenSet);
  }

  // Setter methods that are common for all derived RF classes.
  int iNew();
  void setMaps(int sizeOld);

  // Return position of resonance and colour-connected final-state parton.
  int posR() const {return int(posRes);}
  int posF() const {return int(posFinal);}

  // Return maximum Q2.
  double getQ2Max(int evType) {return evType == 1 ? Q2MaxSav : 0.;}

protected:

  // Protected helper methods for internal class use.
  double getsAK(double mA, double mK, double mAK);
  double calcQ2Max(double mA, double mAK, double mK);

  // Veto point if outside available phase space.
  bool vetoPhSpPoint(const vector<double>& invariants, int verboseIn);

  // Save reference to position in vectors of resonance and colour
  // connected parton.
  unsigned int posRes{}, posFinal{};
  // Mass of resonance.
  double mRes{};
  // Mass of final state parton in antennae.
  double mFinal{};
  // Collective mass of rest of downstream decay products of resonance
  // these will just take recoil.
  double mRecoilers{};
  double sAK{};

  // Max Q2 for this brancher, still an overestimate.
  double Q2MaxSav{};
  double colFacSav{};
  // Store whether the colour flow is from R to F (e.g. t -> bW+) or F
  // to R (e.g. tbar -> bbarW-).
  bool colFlowRtoF{};
  map<unsigned int,unsigned int> posNewtoOld{};

};


//==========================================================================

// BrancherEmitRF class for storing information on antennae between a
// coloured resonance and final state parton, and generating a new
// emission.

class BrancherEmitRF : public BrancherRF {

public:

  // Constructor.
  BrancherEmitRF(int iSysIn, Event& event, bool sectorShowerIn,
    vector<int> allIn, unsigned int posResIn, unsigned int posFIn,
    double Q2cut, ZetaGeneratorSet& zetaGenSet) :
    BrancherRF(iSysIn, event, sectorShowerIn, allIn) {
    // Initialise derived-class members and set up trial generator.
    initBrancher(event, allIn, posResIn, posFIn, Q2cut, zetaGenSet);
  }

  // Method to initialise data members specific to BrancherEmitRF.
  void initBrancher(Event& event, vector<int> allIn, unsigned int posResIn,
    unsigned int posFIn, double Q2cut, ZetaGeneratorSet& zetaGenSet);
  void initRF(Event& event, vector<int> allIn, unsigned int posResIn,
    unsigned int posFIn, double Q2cut, ZetaGeneratorSet& zetaGenSet) override {
    initBrancher(event, allIn, posResIn, posFIn, Q2cut, zetaGenSet);}

  // Setter methods.
  vector<double> setmPostVec() override;
  void setidPost() override;
  void setStatPost() override;

  // Generic method, assumes setter methods called earlier.
  bool getNewParticles(Event& event, vector<Vec4> momIn, vector<int> hIn,
    vector<Particle> &pNew, Rndm* rndmPtr, VinciaColour*) override;

  // Generate a new Q2 scale.
  double genQ2(int evTypeIn, double Q2MaxNow, Rndm* rndmPtr,
    Info* infoPtr, const EvolutionWindow* evWindowPtrIn, double colFac,
    vector<double> headroomIn, vector<double> enhanceFacIn,
    int verboseIn) override;

  // Generate complementary invariant(s) for saved trial scale. Return
  // false if no physical kinematics possible.
  bool genInvariants(vector<double>& invariants,Rndm* rndmPtr,
    int verboseIn, Info* /*infoPtr*/) override;

  // Compute antPhys/antTrial, given antPhys. Note, antPhys should be
  // normalised to include charge and coupling factors.
  double pAccept(const double, Info* infoPtr, int verboseIn) override;

};

//==========================================================================

// BrancherSplitRF class for storing information on antennae between a
// coloured resonance and final state parton, and generating a new
// emission.

class BrancherSplitRF : public BrancherRF {

public:

  // Constructor.
  BrancherSplitRF(int iSysIn, Event& event, bool sectorShowerIn,
    vector<int> allIn, unsigned int posResIn, unsigned int posFIn,
    double Q2cut, ZetaGeneratorSet& zetaGenSet) :
    BrancherRF(iSysIn, event, sectorShowerIn, allIn) {
    // Initialise derived-class members and set up trial generator.
    initBrancher(event, allIn, posResIn, posFIn, Q2cut, zetaGenSet);
  }

  // Method to initialise data members specific to BrancherSplitRF.
  void initBrancher(Event& event, vector<int> allIn, unsigned int posResIn,
    unsigned int posFIn, double Q2cut, ZetaGeneratorSet& zetaGenSet);
  void initRF(Event& event, vector<int> allIn, unsigned int posResIn,
    unsigned int posFIn, double Q2cut, ZetaGeneratorSet& zetaGenSet) override {
    initBrancher(event, allIn, posResIn, posFIn, Q2cut, zetaGenSet);}

  // Setter methods.
  vector<double> setmPostVec() override;
  void setidPost() override;
  void setStatPost() override;

  // Generic method, assumes setter methods called earlier.
  bool getNewParticles(Event& event, vector<Vec4> momIn,
    vector<int> hIn, vector<Particle>& pNew, Rndm*, VinciaColour*) override;

  // Generate a new Q2 scale.
  double genQ2(int evTypeIn, double Q2MaxNow, Rndm* rndmPtr,
    Info* infoPtr, const EvolutionWindow* evWindowPtrIn, double colFac,
    vector<double> headroomIn, vector<double> enhanceFacIn,
    int verboseIn) override;

protected:

  // Members.
  int     idFlavSav{};
  double  mFlavSav{};

};

//==========================================================================

// The VinciaFSR class for resonant decays. Inherits from TimeShower
// in Pythia 8 so can be used as alternative to TimeShower.
// Methods that must replace ones in TimeShower are marked with override.

class VinciaFSR : public TimeShower {

  // Allow VinciaISR and VinciaHistory to access private information.
  friend class VinciaISR;
  friend class VinciaHistory;

public:

  // Constructor.
  VinciaFSR() :
    zetaGenSetRF(ZetaGeneratorSet(TrialGenType::RF)),
      zetaGenSetFF(ZetaGeneratorSet(TrialGenType::FF)) {
    verbose = 0; headerIsPrinted = false; isInit = false;
    isPrepared = false; diagnosticsPtr = 0;}

  // Destructor.
  ~VinciaFSR() {;}

  // The following methods control main flow of shower and are
  // inherited from TimeShower. Any method re-implementing a
  // TimeShower method is appended with (TimeShower).

  // Initialize alphaStrong and related pTmin parameters (TimeShower).
  void init(BeamParticle* beamAPtrIn = 0, BeamParticle* beamBPtrIn = 0)
    override;

  // Force reset at beginning of each event.
  void onBeginEvent() override { isPrepared = false; }

  // Possible limitation of first emission (TimeShower, last two
  // arguments purely dummy in Vincia implementation). Determines if
  // max pT limit should be imposed on first emission. Note, not set
  // up to handle Pythia's explicit DPS processes yet.
  bool limitPTmax(Event& event, double Q2Fac = 0., double Q2Ren = 0.) override;

  // Top-level routine to do a full time-like shower in resonance
  // decay (TimeShower).
  int shower( int iBeg, int iEnd, Event& event, double pTmax,
    int nBranchMax = 0) override;

  // Method to add QED showers in hadron decays (TimeShower).
  int showerQED(int iBeg, int iEnd, Event& event, double pTmax) override;

  // Method to add QED showers to partons below colour resolution
  // scale (TimeShower).
  int showerQEDafterRemnants(Event& event) override;

  // Prepare process-level event for shower + interleaved resonance decays.
  // Usage: prepareProcess( process, event, iPos).
  // iPos provides mapping from process to event entries (before showering).
  void prepareProcess( Event& process, Event& event,
    vector<int>& iPosBefShow) override;

  // Used for global recoil scheme (TimeShower, no Vincia implementation yet).
  // void prepareGlobal(Event&);

  // Prepare system for evolution (TimeShower).
  void prepare( int iSys, Event& event, bool limitPTmaxIn) override;

  // Update antenna list after a multiple interactions rescattering
  // (TimeShower, no Vincia implementation yet).
  // void rescatterUpdate( int iSys, Event& event);

  // Update antenna list after each ISR emission (TimeShower).
  void update( int iSys, Event& event, bool hasWeakRad = false) override;

  // Select next pT in downwards evolution (TimeShower).
  double pTnext( Event& event, double pTbegAll, double pTendAll,
    bool isFirstTrial = false, bool doTrialIn = false) override;

  // Select next pT for interleaved resonance decays.
  double pTnextResDec() override {
    double pTresDecMax =  0.;
    iHardResDecSav = -1 ;
    for (size_t i=0; i<pTresDecSav.size(); ++i) {
      if (pTresDecSav[i] > pTresDecMax) {
        pTresDecMax    = pTresDecSav[i];
        iHardResDecSav = i;
      }
    }
    return pTresDecMax;
  }

  // Branch event, including accept/reject veto (TimeShower).
  bool branch( Event& event, bool isInterleaved = false) override;

  // Do a resonance decay + resonance shower (including any nested decays).
  // May be called recursively for nested decays.
  // Usage: resonanceShower( process, event, iPos, pTmerge), where iPos
  // maps process to event entries, and pTmerge is the scale at which this
  // system should be merged into its parent system.
  bool resonanceShower(Event& process, Event& event, vector<int>& iPos,
    double qRestart) override;

  // Utility to print antenna list; for debug mainly (TimeShower).
  void list() const override;

  // Initialize data members for calculation of uncertainty bands
  // (TimeShower, no Vincia implementation yet).
  // virtual bool initUncertainties();

  // Tell whether FSR has done a weak emission.
  bool getHasWeaklyRadiated() override {return hasWeaklyRadiated;}

  // Tell which system was the last processed one (TimeShower).
  int system() const override {return iSysWin;}

  // Potential enhancement factor of pTmax scale for hardest emission.
  // Used if limitPTmax = true (TimeShower).
  double enhancePTmax() override {return pTmaxFudge;}

  // Provide the pT scale of the last branching in the above shower
  // (TimeShower).
  double pTLastInShower() override {return pTLastAcceptedSav;}

  // The following methods for merging not yet implemented in Vincia:
  //   Event clustered()
  //   map<string, double> getStateVariables (const Event &, int, int, int,
  //     string)
  //   bool isTimelike()
  //   vector<string> getSplittingName()
  //   double getSplittingProb()
  //   bool allowedSplitting()
  //   vector<int> getRecoilers()

  // The remaining public functions Vincia only, i.e. not inherited
  // from Pythia 8.

  // Initialise pointers to Vincia objects.
  void initVinciaPtrs(VinciaColour* colourPtrIn,
    shared_ptr<VinciaISR> isrPtrIn, MECs* mecsPtrIn,
    Resolution* resolutionPtrIn, VinciaCommon* vinComPtrIn,
    VinciaWeights* vinWeightsPtrIn);

  // Set pointer to object to use for diagnostics and profiling.
  void setDiagnosticsPtr(shared_ptr<VinciaDiagnostics> diagnosticsPtrIn) {
    diagnosticsPtr = diagnosticsPtrIn;
  }

  // Set EW shower module.
  void setEWShowerPtr(VinciaModulePtr ewShowerPtrIn) {
    ewShowerPtr = ewShowerPtrIn;
  }

  // Set QED shower module for hard scattering (and resonance decays).
  void setQEDShowerHardPtr(VinciaModulePtr qedShowerPtrIn) {
    qedShowerHardPtr = qedShowerPtrIn;
  }

  // Set QED shower module for MPI and hadronization.
  void setQEDShowerSoftPtr(VinciaModulePtr qedShowerPtrIn) {
    qedShowerSoftPtr = qedShowerPtrIn;
  }

  // Initialize pointers to antenna sets.
  void initAntPtr(AntennaSetFSR* antSetIn) {antSetPtr = antSetIn;}

  // Wrapper function to return a specific antenna inside an antenna set.
  AntennaFunction* getAntFunPtr(enum AntFunType antFunType) {
    return antSetPtr->getAntFunPtr(antFunType);}
  // Wrapper to return all AntFunTypes that are contained in antSetPtr.
  vector<enum AntFunType> getAntFunTypes() {
    return antSetPtr->getAntFunTypes();}

  // Print header information (version, settings, parameters, etc.).
  void header();

  // Communicate information about trial showers for merging.
  void setIsTrialShower(bool isTrialShowerIn){
    isTrialShower=isTrialShowerIn;
  }
  void setIsTrialShowerRes(bool isTrialShowerResIn){
    isTrialShowerRes=isTrialShowerResIn;
  }

  // Save the flavour content of system in Born state
  // (needed for sector shower).
  void saveBornState(int iSys, Event& born);
  // Save the flavour content of Born for trial shower.
  void saveBornForTrialShower(Event& born);

  // Check self-consistency, for specific iSys >= 0 or all systems (iSys < 0).
  bool check(int iSys, Event &event);

  // Set verbosity level.
  void setVerbose(int verboseIn) {verbose = verboseIn;}

private:

  // Constants
  static const int NLOOPMAX;

  // Initialize evolution windows.
  void initEvolutionWindows(void);
  // Return window Q2.
  double getQ2Window(int iWindow, double q2cutoff);
  // Return Lambda value.
  double getLambda(int nFin, AlphaStrong* aSptr);
  // Method to return renormalisation-scale prefactor.
  double getkMu2(bool isEmit);
  // Method to return renormalisation scale. Default scale is kMu *
  // evolution scale.
  double getMu2(bool isEmit);
  // Reset (or clear) sizes of all containers.
  void clearContainers();
  // Method to set up QCD antennae, called in prepare.
  bool setupQCDantennae(int iSys, Event& event);
  // Set starting scale of shower (power vs wimpy) for system iSys.
  void setStartScale(int iSys, Event& event);

  // Auxiliary method to compute scale for interleaved resonance decays
  virtual double calcPTresDec(Particle& res) {
    if (resDecScaleChoice == 0) return res.mWidth();
    double virt = pow2(res.m()) - pow2(res.m0());
    if (resDecScaleChoice == 1) return abs(virt) / res.m0();
    else if (resDecScaleChoice == 2) return sqrt(abs(virt));
    return 0.;
  }

  // Template method for generating the next Q2 for a generic Brancher.
  template <class Brancher> bool q2NextQCD( vector<Brancher> &brancherVec,
    const map<double, EvolutionWindow>& evWindows, const int evType,
    const double q2Begin, const double q2End, bool isEmit);

  // Methods to generate trial scales for each QCD shower component.
  // Based on the generic template method above.
  bool q2NextEmitQCD(const double q2Begin, double q2End);
  bool q2NextSplitQCD(const double q2Begin, double q2End);
  bool q2NextEmitResQCD(const double q2Begin, const double q2End);
  bool q2NextSplitResQCD(const double q2Begin, const double q2End);
  bool q2NextEmitQED(double q2Begin, const double q2End);
  bool q2NextSplitQED(double q2Begin, const double q2End);

  // Perform a QCD branching.
  bool branchQCD(Event& event);
  // Perform an EW/QED branching.
  bool branchEW(Event& event);

  // Perform an early antenna rejection.
  bool rejectEarly(AntennaFunction* &antFunPtr,bool doMEC);
  // Compute physical antenna function.
  double getAntFunPhys(AntennaFunction* &antFunPtr);
  // Calculate acceptance probability.
  double pAcceptCalc(double antPhys);
  // Generate the full kinematics.
  bool genFullKinematics(int kineMap, Event event, vector<Vec4> &pPost);
  // Check if a trial is accepted.
  bool acceptTrial(Event& event);
  // Generate new particles for the antenna.
  bool getNewParticles(Event& event, AntennaFunction* antFunPtr,
    vector<Particle> &newParts);

  // Generate new helicities for the antenna. Default is to set same
  // as before, with a new unpolarised emission inserted. If helicity
  // shower is off, this will correspond to all unpolarised. Do it
  // this way because in case of resonance decays rest of vector
  // includes whole resonance system, whose polarisations we don't
  // want to change, i.e. they only recoil kinematically.
  vector<int> genHelicities(AntennaFunction* antFunPtr);

  // ME corrections.
  double getMEC(int iSys, const Event& event,
    const vector<Particle>& statePost, VinciaClustering& thisClus);

  // Update the event.
  bool updateEvent(Event& event,ResJunctionInfo & junctionInfoIn);
  // Update the parton systems.
  void updatePartonSystems();
  // Create a new emission brancher.
  void saveEmitterFF(int iSysIn, Event& event, int i0, int i1);
  // Create a new resonance emission brancher.
  void saveEmitterRF(int iSys, Event& event, vector<int> allIn,
    unsigned int posResIn, unsigned int posFIn, bool colMode);
  // Create a new resonance splitter.
  void saveSplitterRF(int iSysIn, Event& event, vector<int> allIn,
    unsigned int posResIn, unsigned int posFIn,bool colMode);
  // Create a new splitter brancher.
  void saveSplitterFF(int iSysIn, Event& event, int i0, int i1, bool col2acol);
  // Update emission branchers due to a recoiled parton.
  void updateEmittersFF(Event& event, int iOld, int iNew);
  // Update emission brancher due to an emission.
  void updateEmitterFF(Event& event, int iOld1, int iOld2, int iNew1,
    int iNew2);
  // Update splitter branchers due to a recoiled parton.
  void updateSplittersFF(Event& event, int iOld, int iNew);
  // Update splitter brancher due to an emission.
  void updateSplitterFF(Event& event, int iOld1, int iOld2, int iNew1,
    int iNew2, bool col2acol);
  // Remove a splitter due to a gluon that has branched, assumes that
  // iRemove is splitting gluon.
  void removeSplitterFF(int iRemove);
  // Update resonance emitter due to changed downstream decay products.
  bool updateEmittersRF(int iSysRes, Event& event, int iRes);
  // Update resonance emitter due to changed downstream decay products.
  void updateEmittersRF(int iSysRes, Event& event, vector<int> resSysAll,
    unsigned int posRes, unsigned int posPartner, bool isCol);
  // Update the antennae.
  bool updateAntennae(Event& event);
  // Update systems of QCD antennae after an EW/QED branching.
  bool updateAfterEW(Event& event, int sizeOld);
  // Print a brancher lookup.
  void printLookup(map< pair<int, bool>, unsigned int >& lookupEmitter,
    string name);
  // Print the brancher lookup maps.
  void printLookup();
  // Calculate the headroom factor.
  vector<double> getHeadroom(int iSys, bool isEmit, double q2Next);
  // Calculate the enhancement factor.
  vector<double> getEnhance(int iSys, bool isEmit, double q2Next);

  // Flags if initialized and prepared.
  bool isInit{}, isPrepared{};

  // Beam info.
  double eCMBeamsSav{}, m2BeamsSav{};

  // Main on/off switches.
  bool doFF{}, doRF{}, doII{}, doIF{}, doQED{}, doWeak{};
  int ewMode{}, ewModeMPI{};

  // Parameter setting which kind of 2->4 modifications (if any) are used.
  int mode2to4{};

  // Shower parameters.
  bool helicityShower{}, sectorShower{};
  int evTypeEmit{}, evTypeSplit{}, nGluonToQuark{};
  double q2CutoffEmit{}, q2CutoffSplit{};
  int nFlavZeroMass{};
  map<int,int> resSystems;
  int kMapResEmit{};
  int kMapResSplit{};

  // Factorization scale and shower starting settings.
  int    pTmaxMatch{};
  double pTmaxFudge{}, pT2maxFudge{}, pT2maxFudgeMPI{};

  // AlphaS parameters.
  bool useCMW{};
  int alphaSorder{};
  double alphaSvalue{}, alphaSmax{}, alphaSmuFreeze{}, alphaSmuMin{};
  double aSkMu2Emit{}, aSkMu2Split{};

  // Calculated alphaS values.
  double mu2freeze{}, mu2min{};

  // Map of qmin evolution window.
  map<double, EvolutionWindow> evWindowsEmit;
  map<double, EvolutionWindow> evWindowsSplit;

  // Lists of different types of antennae.
  vector<BrancherEmitRF> emittersRF;
  vector<BrancherSplitRF> splittersRF;
  vector<BrancherEmitFF> emittersFF;
  vector<BrancherSplitFF> splittersFF;

  // Look up resonance emitter, bool switches between R (true) or F
  // (false), n.b. multiply resonance index by sign of colour index
  // involved in branching to avoid a multiple-valued map.
  map< pair<int, bool>, unsigned int > lookupEmitterRF;
  map< pair<int, bool>, unsigned int > lookupSplitterRF;
  // Look up emitter, bool switches between col and anticol end
  map< pair<int, bool>, unsigned int > lookupEmitterFF;
  // Lookup splitter, bool switches between splitter and recoiler.
  map< pair<int, bool>, unsigned int > lookupSplitterFF;

  // Current winner.
  Brancher* winnerQCD{};
  VinciaModulePtr winnerEW{};
  double q2WinSav{}, pTLastAcceptedSav{};

  // Variables set by branch().
  int iSysWin{};
  enum AntFunType antFunTypeWin{AntFunType::NoFun};
  bool hasWeaklyRadiated{false};

  // Index of latest emission (slightly arbritrary for splittings but
  // only used to populate some internal histograms.
  int iNewSav{};

  // Storage of the post-branching configuration while it is being built.
  vector<Particle> pNew;
  // Total and MEC accept probability.
  vector<double> pAccept;

  // Colour reconnection parameters.
  bool doCR{}, CRjunctions{};

  // Enhancement switches and parameters.
  bool enhanceInHard{}, enhanceInResDec{}, enhanceInMPI{};
  double enhanceAll{}, enhanceBottom{}, enhanceCharm{}, enhanceCutoff{};

  // Possibility to allow user veto of emission step.
  bool hasUserHooks{}, canVetoEmission{}, canVetoISREmission{};

  // Flags to tell a few basic properties of each parton system.
  map<int, bool> isHardSys, isResonanceSys, polarisedSys, doMECsSys;
  map<int, bool> stateChangeSys;
  bool stateChangeLast;

  // Save initial FSR starting scale system by system.
  map<int, double> Q2hat;

  // Count the number of branchings in the system.
  map<int, int> nBranch, nBranchFSR;

  // Total mass of showering system.
  map<int, double> mSystem;

  // Count numbers of quarks and gluons.
  map<int, int> nG, nQ, nLep, nGam;

  // Partons present in the Born (needed in sector shower).
  map<int, bool> savedBorn;
  map<int, bool> resolveBorn;
  map<int, map<int, int>> nFlavsBorn;

  // Save headroom and enhancement factors for each system for both
  // emission and splitting branchers.
  map< pair<int, pair<bool,bool> > , vector<double> > headroomSav;
  map< pair<int, pair<bool,bool> > , vector<double> > enhanceSav;

  // Information about resonances that participate in junctions.
  map<int, bool> hasResJunction;
  map<int, ResJunctionInfo> junctionInfo;

  // Flags used in merging.
  bool doMerging, isTrialShower, isTrialShowerRes;

  // Verbose settings.
  int verbose{};
  bool headerIsPrinted{};

  // Diagnostics.
  shared_ptr<VinciaDiagnostics> diagnosticsPtr{};

  // Debug settings.
  bool allowforceQuit{}, forceQuit{};
  int nBranchQuit{};

  // Zeta generators for trial generators.
  ZetaGeneratorSet      zetaGenSetRF;
  ZetaGeneratorSet      zetaGenSetFF;

  // Pointers to VINCIA objects.
  AntennaSetFSR*        antSetPtr{};
  MECs*                 mecsPtr{};
  VinciaColour*         colourPtr{};
  Resolution*           resolutionPtr{};
  shared_ptr<VinciaISR> isrPtr{};
  VinciaCommon*         vinComPtr{};
  VinciaWeights*        weightsPtr{};

  // Electroweak shower pointers.
  VinciaModulePtr       ewShowerPtr{};
  VinciaModulePtr       qedShowerHardPtr{};
  VinciaModulePtr       qedShowerSoftPtr{};
  // Pointer to either ewShowerPtr or qedShowerHardPtr depending on ewMode.
  VinciaModulePtr       ewHandlerHard{};

  // Pointer to AlphaS instances.
  AlphaStrong* aSemitPtr{};
  AlphaStrong* aSsplitPtr{};

  // Settings and member variables for interleaved resonance decays
  bool doFSRinResonances{true}, doInterleaveResDec{true};
  int resDecScaleChoice{-1}, iHardResDecSav{}, nRecurseResDec{};
  vector<int> idResDecSav, iPosBefSav;
  vector<double> pTresDecSav;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_VinciaFSR_H
