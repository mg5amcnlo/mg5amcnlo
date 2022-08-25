// DireSplittingLibrary.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for the Dire splitting library.

#ifndef Pythia8_DireSplittingLibrary_H
#define Pythia8_DireSplittingLibrary_H

#define DIRE_SPLITTINGLIBRARY_VERSION "2.002"

#include "Pythia8/Basics.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StandardModel.h"

#include "Pythia8/DireSplittings.h"
#include "Pythia8/DireSplittingsQCD.h"
#include "Pythia8/DireSplittingsQED.h"
#include "Pythia8/DireSplittingsEW.h"
#include "Pythia8/DireSplittingsU1new.h"
#include "Pythia8/DireBasics.h"
#include "Pythia8/DireHooks.h"

namespace Pythia8 {

class DireSpace;
class DireTimes;

//==========================================================================

class DireSplittingLibrary {

public:

  // Constructor and destructor.
  DireSplittingLibrary() :  infoPtr(0), settingsPtr(0), particleDataPtr(0),
    rndmPtr(0), beamAPtr(0), beamBPtr(0), coupSMPtr(0), hooksPtr(0),
    hasExternalHook(false) {}
 ~DireSplittingLibrary() { clear();}

  // Initialisation.
  void init(Info* infoPtr, BeamParticle* beamA, BeamParticle* beamB,
    DireInfo* direInfo, DireHooks* hooks = NULL);
  void initFSR();
  void initISR();
  void clear();

  void setTimesPtr (shared_ptr<DireTimes> fsrIn) {
    for (unordered_map<string,DireSplitting*>::iterator
           it = splittings.begin(); it != splittings.end(); ++it )
      it->second->setTimesPtr(fsrIn);
  }
  void setTimesDecPtr (shared_ptr<DireTimes> fsrIn) {
    for (unordered_map<string,DireSplitting*>::iterator
           it = splittings.begin(); it != splittings.end(); ++it )
      it->second->setTimesDecPtr(fsrIn);
  }
  void setSpacePtr (shared_ptr<DireSpace> isrIn) {
    for (unordered_map<string,DireSplitting*>::iterator
           it = splittings.begin(); it != splittings.end(); ++it )
      it->second->setSpacePtr(isrIn);
  }

  unordered_map< string, DireSplitting* > getSplittings() { return splittings;}

  // Overload index operator to access element of splitting vector.
  DireSplitting* operator[](string id);
  const DireSplitting* operator[](string id) const;

  // Generate name for a splitting
  vector<int> getSplittingRadBefID(const Event& event, int rad, int emt);
  vector<int> getSplittingRadBefID_new(const Event& event, int rad, int emt);

  // Generate name for a splitting
  vector<string> getSplittingName(const Event& event, int rad, int emt);
  vector<string> getSplittingName_new(const Event& event, int rad, int emt);

  // Check number of particles produced in splitting.
  int nEmissions(string name);

  void setKernelHooks(DireHooks* hooks) {hooksPtr = hooks;}

  // Some string name hashes, to avoid string conparisons.
  ulong fsrQCD_1_to_1_and_21,
        fsrQCD_1_to_21_and_1,
        fsrQCD_21_to_21_and_21a,
        fsrQCD_21_to_21_and_21b,
        fsrQCD_21_to_1_and_1a,
        fsrQCD_21_to_1_and_1b,
        fsrQCD_1_to_2_and_1_and_2,
        fsrQCD_1_to_1_and_1_and_1,
        fsrQCD_1_to_1_and_21_notPartial,
        fsrQCD_21_to_21_and_21_notPartial,
        fsrQCD_21_to_1_and_1_notPartial,
        fsrQCD_1_to_1_and_21_and_21,
        fsrQCD_1_to_1_and_1_and_1a,
        fsrQCD_1_to_1_and_1_and_1b,
        fsrQCD_1_to_1_and_2_and_2a,
        fsrQCD_1_to_1_and_2_and_2b,
        fsrQCD_1_to_1_and_3_and_3a,
        fsrQCD_1_to_1_and_3_and_3b,
        fsrQCD_1_to_1_and_4_and_4a,
        fsrQCD_1_to_1_and_4_and_4b,
        fsrQCD_1_to_1_and_5_and_5a,
        fsrQCD_1_to_1_and_5_and_5b,
        fsrQCD_21_to_21_and_21_and_21,
        fsrQCD_21_to_21_and_1_and_1a,
        fsrQCD_21_to_21_and_1_and_1b,
        fsrQCD_21_to_21_and_2_and_2a,
        fsrQCD_21_to_21_and_2_and_2b,
        fsrQCD_21_to_21_and_3_and_3a,
        fsrQCD_21_to_21_and_3_and_3b,
        fsrQCD_21_to_21_and_4_and_4a,
        fsrQCD_21_to_21_and_4_and_4b,
        fsrQCD_21_to_21_and_5_and_5a,
        fsrQCD_21_to_21_and_5_and_5b,
        isrQCD_1_to_1_and_21,
        isrQCD_21_to_1_and_1,
        isrQCD_21_to_21_and_21a,
        isrQCD_21_to_21_and_21b,
        isrQCD_1_to_21_and_1,
        isrQCD_1_to_2_and_1_and_2,
        isrQCD_1_to_1_and_1_and_1;

  // Some string name hashes, to avoid string conparisons.
  ulong fsrQED_1_to_1_and_22,
        fsrQED_1_to_22_and_1,
        fsrQED_11_to_11_and_22,
        fsrQED_11_to_22_and_11,
        fsrQED_22_to_1_and_1a,
        fsrQED_22_to_1_and_1b,
        fsrQED_22_to_2_and_2a,
        fsrQED_22_to_2_and_2b,
        fsrQED_22_to_3_and_3a,
        fsrQED_22_to_3_and_3b,
        fsrQED_22_to_4_and_4a,
        fsrQED_22_to_4_and_4b,
        fsrQED_22_to_5_and_5a,
        fsrQED_22_to_5_and_5b,
        fsrQED_22_to_11_and_11a,
        fsrQED_22_to_11_and_11b,
        fsrQED_22_to_13_and_13a,
        fsrQED_22_to_13_and_13b,
        fsrQED_22_to_15_and_15a,
        fsrQED_22_to_15_and_15b,
        fsrQED_1_to_1_and_22_notPartial,
        fsrQED_11_to_11_and_22_notPartial,
        isrQED_1_to_1_and_22,
        isrQED_11_to_11_and_22,
        isrQED_1_to_22_and_1,
        isrQED_11_to_22_and_11,
        isrQED_22_to_1_and_1,
        isrQED_22_to_11_and_11;

  ulong fsrEWK_1_to_1_and_23,
        fsrEWK_1_to_23_and_1,
        fsrEWK_23_to_1_and_1a,
        fsrEWK_23_to_1_and_1b,
        fsrEWK_24_to_1_and_1a,
        fsrEWK_24_to_1_and_1b,
        fsrEWK_25_to_24_and_24,
        fsrEWK_25_to_22_and_22,
        fsrEWK_25_to_21_and_21,
        fsrEWK_24_to_24_and_22,
        isrEWK_1_to_1_and_23;

  ulong fsrU1N_1_to_1_and_22,
        fsrU1N_1_to_22_and_1,
        fsrU1N_11_to_11_and_22,
        fsrU1N_11_to_22_and_11,
        fsrU1N_22_to_1_and_1a,
        fsrU1N_22_to_1_and_1b,
        fsrU1N_22_to_2_and_2a,
        fsrU1N_22_to_2_and_2b,
        fsrU1N_22_to_3_and_3a,
        fsrU1N_22_to_3_and_3b,
        fsrU1N_22_to_4_and_4a,
        fsrU1N_22_to_4_and_4b,
        fsrU1N_22_to_5_and_5a,
        fsrU1N_22_to_5_and_5b,
        fsrU1N_22_to_11_and_11a,
        fsrU1N_22_to_11_and_11b,
        fsrU1N_22_to_13_and_13a,
        fsrU1N_22_to_13_and_13b,
        fsrU1N_22_to_15_and_15a,
        fsrU1N_22_to_15_and_15b,
        fsrU1N_22_to_211_and_211a,
        fsrU1N_22_to_211_and_211b,
        isrU1N_1_to_1_and_22,
        isrU1N_1_to_22_and_1,
        isrU1N_22_to_1_and_1,
        isrU1N_11_to_11_and_22,
        isrU1N_11_to_22_and_11,
        isrU1N_22_to_11_and_11;

private:

  unordered_map< string, DireSplitting* > splittings;
  Info* infoPtr;
  Settings* settingsPtr;
  ParticleData* particleDataPtr;
  Rndm* rndmPtr;
  BeamParticle* beamAPtr;
  BeamParticle* beamBPtr;
  CoupSM* coupSMPtr;
  DireInfo* direInfoPtr;

  // User may load additional kernels.
  DireHooks* hooksPtr;
  bool hasExternalHook;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_DireSplittingLibrary_H
