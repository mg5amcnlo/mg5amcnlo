// HadronLevel.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the main class for hadron-level generation.
// HadronLevel: handles administration of fragmentation and decay.

#ifndef Pythia8_HadronLevel_H
#define Pythia8_HadronLevel_H

#include "Pythia8/Basics.h"
#include "Pythia8/BoseEinstein.h"
#include "Pythia8/ColourTracing.h"
#include "Pythia8/DeuteronProduction.h"
#include "Pythia8/Event.h"
#include "Pythia8/FragmentationFlavZpT.h"
#include "Pythia8/FragmentationSystems.h"
#include "Pythia8/HadronWidths.h"
#include "Pythia8/HiddenValleyFragmentation.h"
#include "Pythia8/Info.h"
#include "Pythia8/JunctionSplitting.h"
#include "Pythia8/LowEnergyProcess.h"
#include "Pythia8/LowEnergySigma.h"
#include "Pythia8/MiniStringFragmentation.h"
#include "Pythia8/NucleonExcitations.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/ParticleDecays.h"
#include "Pythia8/PartonVertex.h"
#include "Pythia8/PhysicsBase.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/RHadrons.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StringFragmentation.h"
#include "Pythia8/TimeShower.h"
#include "Pythia8/UserHooks.h"

namespace Pythia8 {

//==========================================================================

// The HadronLevel class contains the top-level routines to generate
// the transition from the partonic to the hadronic stage of an event.

class HadronLevel : public PhysicsBase {

public:

  // Constructor.
  HadronLevel() = default;

  // Initialize HadronLevel classes as required.
  bool init( TimeShowerPtr timesDecPtr, RHadrons* rHadronsPtrIn,
    DecayHandlerPtr decayHandlePtr, vector<int> handledParticles,
    StringIntPtr stringInteractionsPtrIn, PartonVertexPtr partonVertexPtrIn);

  // Get pointer to StringFlav instance (needed by BeamParticle).
  StringFlav* getStringFlavPtr() {return &flavSel;}

  // Generate the next event.
  bool next(Event& event);

  // Special routine to allow more decays if on/off switches changed.
  bool moreDecays(Event& event);

  // Prepare and pick process for a low-energy hadron-hadron scattering.
  bool initLowEnergyProcesses();
  int pickLowEnergyProcess(int idA, int idB, double eCM, double mA, double mB);

  // Special routine to do a low-energy hadron-hadron scattering.
  bool doLowEnergyProcess(int i1, int i2, int type, Event& event) {
    if (!lowEnergyProcess.collide( i1, i2, type, event)) {
      infoPtr->errorMsg("Error in HadronLevel::doLowEnergyProcess: "
        "Low energy collision failed");
      return false;
    }
    return true;
  }

  // Routine to allow user access to low-energy cross sections.
  double getLowEnergySigma( int idA, int idB, double eCM,  double mA,
    double mB, int type = 0) {
    return lowEnergySigma.sigmaPartial( idA, idB, eCM, mA, mB, type); }

  // Give access to b slope in elastic and diffractive interactions.
  double getLowEnergySlope( int idA, int idB, double eCM, double mA,
    double mB, int type = 2) {
    return lowEnergyProcess.bSlope( idA, idB, eCM, mA, mB, type); }

  // Tell if we did an early user-defined veto of the event.
  bool hasVetoedHadronize() const {return doHadronizeVeto; }

protected:

  virtual void onInitInfoPtr() override{
    registerSubObject(flavSel);
    registerSubObject(pTSel);
    registerSubObject(zSel);
    registerSubObject(stringFrag);
    registerSubObject(ministringFrag);
    registerSubObject(decays);
    registerSubObject(lowEnergyProcess);
    registerSubObject(lowEnergySigma);
    registerSubObject(nucleonExcitations);
    registerSubObject(boseEinstein);
    registerSubObject(hiddenvalleyFrag);
    registerSubObject(junctionSplitting);
    registerSubObject(deuteronProd);
  }

private:

  // Constants: could only be changed in the code itself.
  static const double MTINY;

  // Initialization data, read from Settings.
  bool doHadronize{}, doDecay{}, doPartonVertex{}, doBoseEinstein{},
    doDeuteronProd{}, allowRH{}, closePacking{}, doNonPertAll{};
  double mStringMin{}, eNormJunction{}, widthSepBE{}, widthSepRescatter{};
  vector<int> nonPertProc{};

  // Configuration of colour-singlet systems.
  ColConfig      colConfig;

  // Colour and mass information.
  vector<int>    iParton{}, iJunLegA{}, iJunLegB{}, iJunLegC{},
                 iAntiLegA{}, iAntiLegB{}, iAntiLegC{}, iGluLeg{};
  vector<double> m2Pair{};

  // The generator class for normal string fragmentation.
  StringFragmentation stringFrag;

  // The generator class for special low-mass string fragmentation.
  MiniStringFragmentation ministringFrag;

  // The generator class for normal decays.
  ParticleDecays decays;

  // The generator class for Bose-Einstein effects.
  BoseEinstein boseEinstein;

  // The generator class for deuteron production.
  DeuteronProduction deuteronProd;

  // Classes for flavour, pT and z generation.
  StringFlav flavSel;
  StringPT   pTSel;
  StringZ    zSel;

  // Class for colour tracing.
  ColourTracing colTrace;

  // Junction splitting class.
  JunctionSplitting junctionSplitting;

  // The RHadrons class is used to fragment off and decay R-hadrons.
  RHadrons*  rHadronsPtr;

  // Special class for Hidden-Valley hadronization. Not always used.
  HiddenValleyFragmentation hiddenvalleyFrag;
  bool useHiddenValley{};

  // Special case: colour-octet onium decays, to be done initially.
  bool decayOctetOnia(Event& event);

  // Trace colour flow in the event to form colour singlet subsystems.
  // Option to keep junctions, needed for rope hadronization.
  bool findSinglets(Event& event, bool keepJunctions = false);

  // Class to displace hadron vertices from parton impact-parameter picture.
  PartonVertexPtr partonVertexPtr;

  // Hadronic rescattering.
  class PriorityNode;
  bool doRescatter{}, scatterManyTimes{}, scatterQuickCheck{},
    scatterNeighbours{}, delayRegeneration{};
  double b2Max, tauRegeneration{};
  void queueDecResc(Event& event, int iStart,
    priority_queue<HadronLevel::PriorityNode>& queue);
  int boostDir;
  double boost;
  bool doBoost;
  bool useVelocityFrame;

  // User veto performed right after string hadronization.
  bool doHadronizeVeto;

  // The generator class for low-energy hadron-hadron collisions.
  LowEnergyProcess lowEnergyProcess;
  int    impactModel{};
  double impactOpacity{};

  // Cross sections for low-energy processes.
  LowEnergySigma lowEnergySigma;

  // Nucleon excitations data.
  NucleonExcitations nucleonExcitations = {};

  // Class for event geometry for Rope Hadronization. Production vertices.
  StringRepPtr stringRepulsionPtr;
  FragModPtr fragmentationModifierPtr;

  // Extract rapidity pairs.
  vector< vector< pair<double,double> > > rapidityPairs(Event& event);

  // Calculate the rapidity for string ends, protected against too large y.
  double yMax(Particle pIn, double mTiny) {
    double temp = log( ( pIn.e() + abs(pIn.pz()) ) / max( mTiny, pIn.mT()) );
    return (pIn.pz() > 0) ? temp : -temp; }

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_HadronLevel_H
