// VinciaQED.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the QED antenna-shower class and auxiliary
// classes. Main author is Rob Verheyen.

#ifndef Pythia8_VinciaQED_H
#define Pythia8_VinciaQED_H

// PYTHIA 8 headers.
#include "Pythia8/BeamParticle.h"
#include "Pythia8/Event.h"
#include "Pythia8/StandardModel.h"
#include "Pythia8/PartonSystems.h"

// VINCIA headers.
#include "Pythia8/VinciaCommon.h"
#include "Pythia8/VinciaWeights.h"

namespace Pythia8 {

//==========================================================================

// Class for QED emissions.

class QEDemitElemental {

public:

  // Friends for internal private members.
  friend class QEDemitSystem;

  // Constuctor.
  QEDemitElemental() : rndmPtr(nullptr), partonSystemsPtr(nullptr), q2Sav(0.),
    zetaSav(0.), phiSav(0.), sxjSav(0.), syjSav(0.), alpha(0.), c(0.),
    hasTrial(false), x(0), y(0), idx(0), idy(0), mx2(0.), my2(0.),
    ex(0.), ey(0.), m2Ant(0.), sAnt(0.), QQ(0.), isII(false), isIF(false),
    isFF(false), isRF(false), isIA(true), isDip(false), shh(0.),
    isInitPtr(false), isInit(false), verbose(1) {;}

  // Initialize the pointers.
  void initPtr(Rndm* rndmPtrIn, PartonSystems* partonSystemsPtrIn);
  // Initialize.
  void init(Event &event, int xIn, int yIn, double shhIn,
    double verboseIn);
  // Initialize.
  void init(Event &event, int xIn, vector<int> iRecoilIn, double shhIn,
    double verboseIn);
  // Generate a trial point.
  double generateTrial(Event &event, double q2Start, double q2Low,
    double alphaIn, double cIn);

private:

  // Random pointer.
  Rndm* rndmPtr{};

  // Parton system pointer.
  PartonSystems* partonSystemsPtr{};

  // Trial variables.
  double q2Sav, zetaSav, phiSav, sxjSav, syjSav, alpha, c;
  bool hasTrial;

  // Particle indices.
  int x, y;
  // Recoiler indices.
  vector<int> iRecoil;
  // IDs.
  int idx, idy;
  // Particle masses.
  double mx2, my2;
  // Particle energies.
  double ex, ey;
  // Antenna invariant mass.
  double m2Ant;
  // Antenna dot product.
  double sAnt;
  // The negative of the product of charges.
  double QQ;

  // Type switches.
  bool isII, isIF, isFF, isRF, isIA, isDip;

  // Hadronic invariant mass.
  double shh;

  // Initialization.
  bool isInitPtr, isInit;
  int verbose;

};

//==========================================================================

// Base class for QED systems.

class QEDsystem {

 public:

  // Constructor.
  QEDsystem() : infoPtr(nullptr), partonSystemsPtr(nullptr),
    particleDataPtr(nullptr), rndmPtr(nullptr), settingsPtr(nullptr),
    vinComPtr(nullptr), isInitPtr(false), iSys(-1), verbose(0), jNew(0),
    shat(0.) {;}

  // Destructor.
  virtual ~QEDsystem() = default;

  // Initialize pointers.
  void initPtr(Info* infoPtrIn, ParticleData* particleDataPtrIn,
    PartonSystems* partonSystemsPtrIn, Rndm* rndmPtrIn,
    Settings* settingsPtrIn, VinciaCommon* vinComPtrIn);

  // Initialise settings for current run.
  virtual void init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn,
    int verboseIn) = 0;
  virtual void setVerbose(int verboseIn) { verbose = verboseIn; }
  // Prepare a parton system for evolution.
  virtual void prepare(int iSysIn, Event &event, double q2CutIn,
    bool isBelowHadIn, vector<double> evolutionWindowsIn, AlphaEM alIn) = 0;
  // Build parton system.
  virtual void buildSystem(Event &event) = 0;
  // Generate a trial scale.
  virtual double q2Next(Event &event, double q2Start) = 0 ;
  // Generate kinematics and check for veto of branching.
  virtual bool acceptTrial(Event &event) = 0;
  // Update the envent.
  virtual void updateEvent(Event &event) = 0;
  // Update the parton systems.
  virtual void updatePartonSystems();
  // Print information about the system.
  virtual void print() = 0;

  // Methods to tell which type of brancher this is.
  virtual bool isSplitting() {return false;};
  virtual bool isInitial() {return false;};

 protected:

  // Pointers.
  Info* infoPtr{};
  PartonSystems* partonSystemsPtr{};
  ParticleData* particleDataPtr{};
  Rndm* rndmPtr{};
  Settings* settingsPtr{};
  VinciaCommon* vinComPtr{};
  bool isInitPtr;

  // Event system.
  int iSys;
  vector<Vec4> pNew;

  // Verbose setting.
  int verbose;

  // Information for partonSystems.
  int jNew;
  map<int,int> iReplace;
  double shat;

};

//==========================================================================

// Class for a QED emission system.

class QEDemitSystem : public QEDsystem {

public:

  QEDemitSystem() : shh(-1.), cMat(0.), trialIsVec(false), beamAPtr(nullptr),
    beamBPtr(nullptr), qedMode(-1), qedModeMPI(-1), useFullWkernel(false),
    isBelowHad(false), emitBelowHad(false), q2Cut(-1.), isInit(false),
    TINYPDF(-1.), kMapTypeFinal(0) {;}

  // Initialise settings for current run.
  void init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn, int verboseIn)
    override;
  // Prepare a parton system for photon emission evolution
  void prepare(int iSysIn, Event &event, double q2CutIn, bool isBelowHadIn,
    vector<double> evolutionWindowsIn, AlphaEM alIn) override;
  // Set up antenna pairing for incoherent mode.
  void buildSystem(Event &event) override;
  // Generate a trial scale.
  double q2Next(Event &event, double q2Start) override;
  // Generate kinematics and check veto.
  bool acceptTrial(Event &event) override;
  // Update the envent
  void updateEvent(Event &event) override;
  // Branching type.
  bool isInitial() override {return eleTrial->isII || eleTrial->isIF;}
  // Print the QED emit internal system.
  void print() override;

  // Trial antenna function.
  double aTrial(QEDemitElemental* ele, double sxj, double syj, double sxy);
  // Physical antenna function.
  double aPhys (QEDemitElemental* ele, double sxj, double syj, double sxy);
  // Ratio between PDFs.
  double pdfRatio(bool isA, double eOld, double eNew, int id, double Qt2);

  private:

  // Event system.
  double shh;

  // Internal storage.
  vector<vector<QEDemitElemental> > eleMat;
  vector<int> iCoh;
  double cMat;
  vector<QEDemitElemental> eleVec;

  // AlphaEM.
  AlphaEM al;

  // Evolution window.
  vector<double> evolutionWindows;

  // Trial pointer.
  QEDemitElemental* eleTrial{};
  bool trialIsVec;

  // Pointers.
  BeamParticle* beamAPtr{};
  BeamParticle* beamBPtr{};

  // Settings.
  int qedMode, qedModeMPI;
  bool useFullWkernel, isBelowHad, emitBelowHad;
  double q2Cut;

  // Initialization.
  bool isInit;

  // PDF check.
  double TINYPDF;

  // Recoil strategy.
  int kMapTypeFinal;

  // Global recoil momenta.
  Vec4 pRecSum;
  vector<Vec4> pRec;
  vector<int>  iRec;

};

//==========================================================================

// Class for trial QED splittings.

class QEDsplitElemental {

public:

  // Friends for internal private members.
  friend class QEDsplitSystem;

  // Constructor.
  QEDsplitElemental(Event &event, int iPhotIn, int iSpecIn):
    iPhot(iPhotIn), iSpec(iSpecIn), ariWeight(0) {
    m2Ant = max(VinciaConstants::PICO,
      m2(event[iPhotIn], event[iSpecIn]));
    sAnt = max(VinciaConstants::PICO,
      2.*event[iPhotIn].p()*event[iSpecIn].p());
    m2Spec = max(0., event[iSpecIn].m2());
  }

  // Kallen function.
  double getKallen() {return m2Ant/(m2Ant - m2Spec);}

private:

  // Internal members.
  int iPhot, iSpec;
  double m2Spec, m2Ant, sAnt;
  double ariWeight;
};

//==========================================================================

// Class for a QED splitting system.

class QEDsplitSystem : public QEDsystem {

public:

  // Constructor.
  QEDsplitSystem() :
    totIdWeight(-1.), hasTrial(false),
    q2Trial(-1.), zTrial(-1.), phiTrial(-1.), idTrial(0), nQuark(-1),
    nLepton(-1), q2Max(-1.), q2Cut(-1.), isBelowHad(false),
    beamAPtr(nullptr), beamBPtr(nullptr), isInit(false), kMapTypeFinal(0) {;}

  // Initialize.
  void init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn, int verboseIn)
    override;
  // Prepare list of final-state photons - with recoilers - for splittings.
  void prepare(int iSysIn, Event &event, double q2CutIn, bool isBelowHadIn,
    vector<double> evolutionWindowsIn, AlphaEM alIn) override;
  // Build the splitting system.
  void buildSystem(Event &event) override;
  // Generate a scale for the system.
  double q2Next(Event &event, double q2Start) override;
  // Generate kinematics and check veto.
  bool acceptTrial(Event &event) override;
  // Update Event.
  void updateEvent(Event &event) override;
  // Branching type: isSplitting() = true.
  bool isSplitting() override { return true;}
  // Print the system.
  void print() override;

private:

  // AlphaEM.
  AlphaEM al;

  // Evolution window.
  vector<double> evolutionWindows;

  // Weights for splitting IDs.
  vector<int> ids;
  vector<double> idWeights;
  double totIdWeight;

  // Antennae.
  vector<QEDsplitElemental> eleVec;

  // Trial variables.
  bool hasTrial;
  double q2Trial, zTrial, phiTrial, idTrial;
  QEDsplitElemental* eleTrial{};

  // Settings.
  int nQuark, nLepton;
  double q2Max, q2Cut;
  bool isBelowHad;

  // Pointers.
  BeamParticle*  beamAPtr;
  BeamParticle*  beamBPtr;

  // Initialization.
  bool isInit;

  // Recoil strategy.
  int kMapTypeFinal;

};

//==========================================================================

// Class for a QED conversion system.

class QEDconvSystem : public QEDsystem {

public:

  // Constructor.
  QEDconvSystem() : totIdWeight(-1.), maxIdWeight(-1.), shh(-1.), s(-1.),
    iA(-1), iB(-1), isAPhot(false), isBPhot(false), hasTrial(false),
    iPhotTrial(-1), iSpecTrial(-1), q2Trial(-1.), zTrial(-1.), phiTrial(-1.),
    idTrial(-1), nQuark(-1), q2Cut(-1.),
    isBelowHad(false), beamAPtr(nullptr), beamBPtr(nullptr),isInit(false),
    TINYPDF(-1.) {;}

  // Initialize.
  void init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn, int verboseIn)
    override;
  // Prepare for backwards-evolution of photons.
  void prepare(int iSysIn, Event &event, double q2CutIn, bool isBelowHadIn,
    vector<double> evolutionWindowsIn, AlphaEM alIn) override;
  // Build the system.
  void buildSystem(Event &event) override;
  // Generate a trial scale.
  double q2Next(Event &event, double q2Start) override;
  // Generate kinematics and check veto.
  bool acceptTrial(Event &event) override;
  // Update.
  void updateEvent(Event &event) override;
  // Branching type: isInitial() = true.
  bool isInitial() override {return true;};
  // Print.
  void print() override;

private:

  // Trial pdf ratios for conversion.
  map<int,double> Rhat;

  // AlphaEM.
  AlphaEM al;

  // Evolution window.
  vector<double> evolutionWindows;

  // Weights for conversion IDs.
  vector<int> ids;
  vector<double> idWeights;
  double totIdWeight, maxIdWeight;
  double shh;

  // Antenna parameters.
  double s;
  int iA, iB;
  bool isAPhot, isBPhot;

  // Trial variables.
  bool hasTrial;
  int iPhotTrial, iSpecTrial;
  double q2Trial, zTrial, phiTrial, idTrial;

  // Settings.
  int nQuark;
  double q2Cut;
  bool isBelowHad;

  // Pointers.
  BeamParticle*  beamAPtr;
  BeamParticle*  beamBPtr;

  // Initialization.
  bool isInit;

  // PDF check.
  double TINYPDF;

  // Global recoil momenta
  vector<Vec4> pRec;
  vector<int>  iRec;

};

//==========================================================================

// Base class for Vincia's QED and EW shower modules.

class VinciaModule {

public:

  // Default constructor.
  VinciaModule(): verbose(-1), isInitPtr(false), isInitSav(false) {;};

  // Default destructor.
  virtual ~VinciaModule() = default;

  // Initialise pointers (called at construction time).
  virtual void initPtr(Info* infoPtrIn, VinciaCommon* vinComPtrIn) = 0;

  //Some early initialisation (needed for EW shower).
  virtual void load() {;}

  // Initialise settings for current run (called as part of Pythia::init()).
  virtual void init(BeamParticle* beamAPtrIn = 0, BeamParticle* beamBPtrIn = 0)
    = 0;
  bool isInit() {return isInitSav;}

  // Select helicities for a system of particles.
  virtual bool polarise(vector<Particle>&) {return false;}

  // Prepare to shower a system.
  virtual bool prepare(int iSysIn, Event &event, bool isBelowHadIn) = 0;

  // Update shower system each time something has changed in event.
  virtual void update(Event &event, int iSys) = 0;

  // Set verbosity level.
  virtual void setVerbose(int verboseIn) {verbose = verboseIn;}

  // Clear everything.
  virtual void clear(int iSys=-1) = 0;

  // Generate a trial scale.
  virtual double q2Next(Event &event, double q2Start, double q2End) = 0;

  // Which system does the winner belong to?
  virtual int sysWin() = 0;

  // Get information about the latest branching.
  virtual bool lastIsSplitting() = 0;
  virtual bool lastIsInitial() = 0;
  virtual bool lastIsResonanceDecay() {return false;}

  // Generate kinematics and check veto.
  virtual bool acceptTrial(Event &event) = 0;

  // Update event after branching accepted.
  virtual void updateEvent(Event &event) = 0;

  // Update partonSystems after branching accepted.
  virtual void updatePartonSystems(Event &event) = 0;

  // End scales.
  virtual double q2minColoured() = 0;
  virtual double q2min() = 0;

  // Get number of branchers / systems.
  virtual unsigned int nBranchers() = 0;
  virtual unsigned int nResDec() = 0;

  // Members.
  BeamParticle* beamAPtr{};
  BeamParticle* beamBPtr{};
  Info* infoPtr{};
  ParticleData* particleDataPtr{};
  PartonSystems* partonSystemsPtr{};
  Rndm* rndmPtr{};
  Settings* settingsPtr{};
  VinciaCommon* vinComPtr{};

 protected:

  int verbose;
  bool isInitPtr, isInitSav;

};

//==========================================================================

// Class for performing QED showers.

class VinciaQED : public VinciaModule {

public:

  // Friends for internal private members.
  friend class VinciaFSR;

  // Constructor.
  VinciaQED() {;}

  // Initialise pointers (called at construction time).
  void initPtr(Info* infoPtrIn, VinciaCommon* vinComPtrIn) override;
  // Initialise settings for current run (called as part of Pythia::init()).
  void init(BeamParticle* beamAPtrIn = 0, BeamParticle* beamBPtrIn = 0)
    override;
  // Prepare to shower a system.
  bool prepare(int iSysIn, Event& event, bool isBelowHadIn) override;
  // Update QED shower system(s) each time something has changed in event.
  void update(Event& event, int iSys) override;
  // Set or change verbosity level, and propagate to QED systems.
  virtual void setVerbose(int verboseIn) override {
    verbose = verboseIn;
    emptyQEDemitSystem.setVerbose(verboseIn);
    emptyQEDsplitSystem.setVerbose(verboseIn);
    emptyQEDconvSystem.setVerbose(verboseIn);
  }
  // Clear everything, or specific system.
  void clear(int iSys = -1) override {
    if (iSys < 0) {emitSystems.clear(); splitSystems.clear();
      convSystems.clear();}
    else {emitSystems.erase(iSys); splitSystems.erase(iSys);
      convSystems.erase(iSys);}
    qedTrialSysPtr = nullptr;}

  // Generate a trial scale.
  double q2Next(Event& event, double q2Start, double) override;
  // Return the system window.
  int  sysWin() override {return iSysTrial;}
  // Information about last branching.
  bool lastIsSplitting() override {
    if (qedTrialSysPtr != nullptr) return  qedTrialSysPtr->isSplitting();
    else return false;}
  bool lastIsInitial() override {
    if (qedTrialSysPtr != nullptr) return qedTrialSysPtr->isInitial();
    else return false;}
  // Generate kinematics and check veto.
  bool acceptTrial(Event &event) override;
  // Update event after branching accepted.
  void updateEvent(Event &event) override;
  // Update partonSystems after branching accepted.
  void updatePartonSystems(Event &event) override;
  // Return scales.
  double q2minColoured() override {return q2minColouredSav;}
  double q2min() override {return q2minSav;}

  // Getter for number of systems.
  unsigned int nBranchers() override {
    int sizeNow = emitSystems.size();
    sizeNow = max(sizeNow, (int)splitSystems.size());
    sizeNow = max(sizeNow, (int)convSystems.size());
    return sizeNow;}

  // This module does not implement resonance decays.
  unsigned int nResDec() override { return 0; }

private:

  // Get Q2 for QED system.
  template <class T>
  void q2NextSystem(map<int, T>& QEDsystemList, Event& event, double q2Start);

  // Systems.
  QEDemitSystem emptyQEDemitSystem;
  QEDsplitSystem emptyQEDsplitSystem;
  QEDconvSystem emptyQEDconvSystem;
  map< int, QEDemitSystem> emitSystems;
  map< int, QEDsplitSystem> splitSystems;
  map< int, QEDconvSystem> convSystems;

  // Settings.
  bool doQED, doEmission;
  int  nGammaToLepton, nGammaToQuark;
  bool doConvertGamma, doConvertQuark;

  // Scales.
  double q2minSav, q2minColouredSav;

  // Trial information
  int iSysTrial;
  double q2Trial;
  QEDsystem* qedTrialSysPtr{};

  // AlphaEM.
  AlphaEM al;

  // Evolution windows
  vector<double> evolutionWindows;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_VinciaQED_H
