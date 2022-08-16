// PartonLevel.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the main class for parton-level event generation
// PartonLevel: administrates showers, multiparton interactions and remnants.

#ifndef Pythia8_PartonLevel_H
#define Pythia8_PartonLevel_H

#include "Pythia8/Basics.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/BeamRemnants.h"
#include "Pythia8/Event.h"
#include "Pythia8/HardDiffraction.h"
#include "Pythia8/Info.h"
#include "Pythia8/JunctionSplitting.h"
#include "Pythia8/MergingHooks.h"
#include "Pythia8/MultipartonInteractions.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PartonSystems.h"
#include "Pythia8/PartonVertex.h"
#include "Pythia8/PhysicsBase.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/ResonanceDecays.h"
#include "Pythia8/RHadrons.h"
#include "Pythia8/Settings.h"
#include "Pythia8/Settings.h"
#include "Pythia8/SharedPointers.h"
#include "Pythia8/SigmaTotal.h"
#include "Pythia8/SpaceShower.h"
#include "Pythia8/StandardModel.h"
#include "Pythia8/StringLength.h"
#include "Pythia8/TimeShower.h"
#include "Pythia8/UserHooks.h"

namespace Pythia8 {

//==========================================================================

// The PartonLevel class contains the top-level routines to generate
// the partonic activity of an event.

class PartonLevel : public PhysicsBase {

public:

  // Constructor.
  PartonLevel() = default;

  // Initialization of all classes at the parton level.
  bool init( TimeShowerPtr timesDecPtrIn, TimeShowerPtr timesPtrIn,
    SpaceShowerPtr spacePtrIn, RHadrons* rHadronsPtrIn,
    MergingHooksPtr mergingHooksPtr, PartonVertexPtr partonVertexPtrIn,
    StringIntPtr stringInteractionPtrIn,
    bool useAsTrial);

  // Generate the next parton-level process.
  bool next( Event& process, Event& event);

  // Perform showers in resonance decay chains. (For special cases.)
  void setupShowerSys( Event& process, Event& event);
  bool resonanceShowers( Event& process, Event& event, bool skipForR);

  // Perform decays and showers of W and Z emitted in shower.
  bool wzDecayShowers( Event& event);

  // Tell whether failure was due to vetoing.
  bool hasVetoed() const {return doVeto;}
  bool hasVetoedDiff() const {return doDiffVeto;}
  bool hasVetoedMerging() const {return doMergingVeto;}

  // Accumulate, print and reset statistics.
  void accumulate() {if (isResolved && !isDiff) multiPtr->accumulate();}
  void statistics(bool reset = false) {
    if (doMPI) multiMB.statistics(reset);}
  void resetStatistics() { if (doMPI) multiMB.resetStatistics(); }

  // Reset PartonLevel object for trial shower usage.
  void resetTrial();
  // Provide the pT scale of the last branching in the shower.
  double pTLastInShower() { return pTLastBranch; }
  // Provide the type of the last branching in the shower.
  int typeLastInShower() { return typeLastBranch; }

  // Check of any trial emissions could have been enhanced.
  bool canEnhanceTrial() { return doEnhanceTrial; }
  // Get enhanced trial emission evolution variable.
  double getEnhancedTrialPT() {
    if (canEnhanceTrial()) return infoPtr->weightContainerPtr->
      weightsSimpleShower.getEnhancedTrialPT();
    return 0.;
  }
  // Get enhanced trial emission weight.
  double getEnhancedTrialWeight() {
    if (canEnhanceTrial()) return infoPtr->weightContainerPtr->
      weightsSimpleShower.getEnhancedTrialWeight();
    return 1.;
  }

  // Spare copies of normal beam pointers.
  BeamParticle*  beamHadAPtr{};
  BeamParticle*  beamHadBPtr{};

  // Pointers to timelike showers for resonance decays and the rest.
  TimeShowerPtr  timesDecPtr{};
  TimeShowerPtr  timesPtr{};

  // Pointer to spacelike showers.
  SpaceShowerPtr spacePtr{};

protected:

  virtual void onInitInfoPtr() override {
    registerSubObject(multiMB);
    registerSubObject(multiSDA);
    registerSubObject(multiSDB);
    registerSubObject(multiCD);
    registerSubObject(multiGmGm);
    registerSubObject(remnants);
    registerSubObject(resonanceDecays);
    registerSubObject(junctionSplitting);
    registerSubObject(hardDiffraction);
  }

private:

  // Constants: could only be changed in the code itself.
  static const int NTRY;

  // Initialization data, mainly read from Settings.
  bool   doNonDiff{}, doDiffraction{}, doMPI{}, doMPIMB{}, doMPISDA{},
         doMPISDB{}, doMPICD{}, doMPIinit{}, doISR{}, doFSRduringProcess{},
         doFSRafterProcess{}, doFSRinResonances{}, doInterleaveResDec{},
         doRemnants{}, doSecondHard{}, hasOneLeptonBeam{}, hasTwoLeptonBeams{},
         hasPointLeptons{}, canVetoPT{}, canVetoStep{}, canVetoMPIStep{},
         canVetoEarly{}, canSetScale{}, allowRH{}, earlyResDec{},
         vetoWeakJets{}, canReconResSys{}, doReconnect{}, doHardDiff{},
         forceResonanceCR{}, doNDgamma{}, doMPIgmgm{}, showUnresGamma{};
  int    pTmaxMatchMPI;
  double mMinDiff{}, mWidthDiff{}, pMaxDiff{}, vetoWeakDeltaR2{};

  // Event generation strategy. Number of steps. Maximum pT scales.
  bool   doVeto{};
  int    nMPI{}, nISR{}, nFSRinProc{}, nFSRinRes{}, nISRhard{}, nFSRhard{},
         typeLatest{}, nVetoStep{}, typeVetoStep{}, nVetoMPIStep{}, iSysNow{},
         reconnectMode{}, hardDiffSide{}, sampleTypeDiff{};
  double pTsaveMPI{}, pTsaveISR{}, pTsaveFSR{}, pTvetoPT{};

  // Current event properties.
  bool   isNonDiff{}, isDiffA{}, isDiffB{}, isDiffC{}, isDiff{},
         isSingleDiff{}, isDoubleDiff{}, isCentralDiff{},
         isResolved{}, isResolvedA{}, isResolvedB{}, isResolvedC{},
         isHardDiffA{}, isHardDiffB{}, isHardDiff{}, doDiffVeto{},
         hardDiffSet{}, isElastic{}, twoHard{};
  int    sizeProcess{}, sizeEvent{}, nHardDone{}, nHardDoneRHad{}, iDS{};
  double eCMsave;
  vector<bool> inRHadDecay;
  vector<int>  iPosBefShow;

  // Variables for photon inside electron.
  bool   hasGammaA{}, hasGammaB{}, beamHasGamma{}, beamAisGamma{},
         beamBisGamma{}, beamAhasGamma{}, beamBhasGamma{}, beamAhasResGamma{},
         beamBhasResGamma{}, beamHasResGamma{}, isGammaHadronDir{},
         sampleQ2gamma{};
  int    gammaMode{}, gammaModeEvent{}, gammaOffset{};
  double eCMsaveGamma{};

  // Pointer to assign space-time vertices during parton evolution.
  PartonVertexPtr partonVertexPtr{};

  // The generator classes for multiparton interactions.
  MultipartonInteractions  multiMB;
  MultipartonInteractions  multiSDA;
  MultipartonInteractions  multiSDB;
  MultipartonInteractions  multiCD;
  MultipartonInteractions* multiPtr{};
  MultipartonInteractions  multiGmGm;

  // The generator class to construct beam-remnant kinematics.
  BeamRemnants remnants;

  // The RHadrons class is used to fragment off and decay R-hadrons.
  RHadrons*    rHadronsPtr{};

  // ResonanceDecay object does sequential resonance decays.
  ResonanceDecays resonanceDecays;

  // The Colour reconnection class used to do colour reconnection.
  ColRecPtr colourReconnectionPtr{};

  // The Junction splitting class used to split junctions systems.
  JunctionSplitting junctionSplitting;

  // The Diffraction class is for hard diffraction selection.
  HardDiffraction hardDiffraction;

  // Resolved diffraction: find how many systems should have it.
  int decideResolvedDiff( Event& process);

  // Set up an unresolved process, i.e. elastic or diffractive.
  bool setupUnresolvedSys( Event& process, Event& event);

  // Set up the hard process, excluding subsequent resonance decays.
  void setupHardSys( Event& process, Event& event);

  // Resolved diffraction: pick whether to have it and set up for it.
  void setupResolvedDiff( Event& process);

  // Resolved diffraction: restore normal behaviour.
  void leaveResolvedDiff( int iHardLoop, Event& process, Event& event);

  // Hard diffraction: set up the process record.
  void setupHardDiff( Event& process);

  // Hard diffraction: leave the process record.
  void leaveHardDiff( Event& process, Event& event, bool physical = true);

  // Photon beam inside lepton beam: set up the parton level generation.
  bool setupResolvedLeptonGamma( Event& process);

  // Photon beam inside lepton beam: recover the whole event and
  // add scattered leptons.
  void leaveResolvedLeptonGamma( Event& process, Event& event,
    bool physical = true);

  // Set the photon collision mode for the current event.
  void saveGammaModeEvent( int gammaModeA, int gammaModeB);

  // Photon beam inside lepton beam: set up the parton level generation.
  void cleanEventFromGamma( Event& event);

  // Pointer to MergingHooks object for user interaction with the merging.
  MergingHooksPtr mergingHooksPtr{};
  // Parameters to specify trial shower usage.
  bool doTrial{};
  bool doEnhanceTrial{};
  int nTrialEmissions{};
  // Parameters to store to veto trial showers.
  double pTLastBranch{};
  int typeLastBranch{};
  // Parameters to specify merging usage.
  bool canRemoveEvent{}, canRemoveEmission{}, doMergingVeto{};

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_PartonLevel_H
