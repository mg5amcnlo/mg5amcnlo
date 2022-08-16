// LowEnergyProcess.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for low-energy hadronic collisions, as used for rescattering.

#ifndef Pythia8_LowEnergyProcess_H
#define Pythia8_LowEnergyProcess_H

#include "Pythia8/Basics.h"
#include "Pythia8/Event.h"
#include "Pythia8/FragmentationSystems.h"
#include "Pythia8/LowEnergySigma.h"
#include "Pythia8/MiniStringFragmentation.h"
#include "Pythia8/HadronWidths.h"
#include "Pythia8/NucleonExcitations.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/StringFragmentation.h"

namespace Pythia8 {

//==========================================================================

// LowEnergyProcess class.
// Is used to describe the low-energy collision between two hadrons.

class LowEnergyProcess : public PhysicsBase {

public:

  // Constructor.
  LowEnergyProcess() = default;

  // Initialize the class.
  void init( StringFlav* flavSelPtrIn, StringFragmentation* stringFragPtrIn,
    MiniStringFragmentation* ministringFragPtrIn,
    LowEnergySigma* lowEnergySigmaPtrIn,
    NucleonExcitations* nucleonExcitationsPtrIn);

  // Produce outgoing primary hadrons from collision of incoming pair.
  bool collide( int i1, int i2, int typeIn, Event& event, Vec4 vtx = Vec4(),
    Vec4 vtx1 = Vec4(), Vec4 vtx2 = Vec4());

  // Event record to handle hadronization.
  Event         leEvent;

  // Give access to b slope in elastic and diffractive interactions.
  double bSlope( int id1In, int id2In, double eCMIn, double mAIn, double mBIn,
    int typeIn = 2) { id1 = id1In; id2 = id2In; eCM = eCMIn, sCM = eCM * eCM;
    mA = mAIn; mB = mBIn; type = typeIn; return bSlope();}

private:

  // Initialization flag.
  bool isInit = false;

  // Parameters of the generation process.
  double probStoUD, fracEtass, fracEtaPss, xPowMes, xPowBar, xDiqEnhance,
         sigmaQ, mStringMin, sProton, probDoubleAnn;

  // Properties of the current collision. 1 or 2 is two incoming hadrons.
  // "c" or "ac" is colour or anticolour component of hadron.
  bool   isBaryon1, isBaryon2;
  int    type, sizeOld, id1, id2, idc1, idac1, idc2, idac2, nHadron,
         id1sv = {}, id2sv = {};
  double m1, m2, eCM, sCM, mThr1, mThr2, z1, z2, mT1, mT2, mA, mB,
         mc1, mac1, px1, py1, pTs1, mTsc1, mTsac1, mTc1, mTac1,
         mc2, mac2, px2, py2, pTs2, mTsc2, mTsac2, mTc2, mTac2, bA, bB;

  // Pointer to class for flavour generation.
  StringFlav* flavSelPtr;

  // Pointer to the generator for normal string fragmentation.
  StringFragmentation* stringFragPtr;

  // Pointer to the generator for special low-mass string fragmentation.
  MiniStringFragmentation* ministringFragPtr;

  // Separate configuration for simple collisions.
  ColConfig simpleColConfig;

  // Cross sections for low-energy processes.
  LowEnergySigma* lowEnergySigmaPtr;

  // Pointer to class for handling nucleon excitations
  NucleonExcitations* nucleonExcitationsPtr;

  // Handle inelastic nondiffractive collision.
  bool nondiff();

  // Handle elastic and diffractive collisions.
  bool eldiff();

  // Handle excitation collisions.
  bool excitation();

  // Handle annihilation collisions.
  bool annihilation();

  // Handle resonant collisions.
  bool resonance();

  // Simple version of hadronization for low-energy hadronic collisions.
  bool simpleHadronization();

  // Special case with isotropic two-body final state.
  bool twoBody();

  // Special case with isotropic three-body final state.
  bool threeBody();

  // Split up hadron A or B into a colour pair, with masses and pT values.
  bool splitA( double mMax, double redMpT, bool splitFlavour = true);
  bool splitB( double mMax, double redMpT, bool splitFlavour = true);

  // Split a hadron inte a colour and an anticolour part.
  pair< int, int> splitFlav( int id);

  // Choose relative momentum of colour and anticolour constituents in hadron.
  double splitZ( int iq1, int iq2, double mRat1, double mRat2);

  // Estimate lowest possible mass state for flavour combination.
  double mThreshold( int iq1, int iq2);

  // Estimate lowest possible mass state for diffractive excitation.
  double mDiffThr( int idNow, double mNow);

  // Pick slope b of exp(b * t) for elastic and diffractive events.
  double bSlope();

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_LowEnergyProcess_H
