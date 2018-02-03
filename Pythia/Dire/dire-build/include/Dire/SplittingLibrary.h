
#ifndef Pythia8_SplittingLibrary_H
#define Pythia8_SplittingLibrary_H

#define DIRE_SPLITTINGLIBRARY_VERSION "2.002"

#include "Pythia8/Basics.h"
#include "Pythia8/Pythia.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StandardModel.h"

#include "Dire/Splittings.h"
#include "Dire/SplittingsQCD.h"
#include "Dire/SplittingsQED.h"
#include "Dire/SplittingsEW.h"
#include "Dire/Hooks.h"

namespace Pythia8 {

class DireSpace;
class DireTimes;
 
//==========================================================================

class SplittingLibrary {

public:  

  // Constructor and destructor.
  SplittingLibrary() : settingsPtr(0), particleDataPtr(0), rndmPtr(0),
    beamAPtr(0), beamBPtr(0), coupSMPtr(0), infoPtr(0), hooksPtr(0),
    hasExternalHook(false) {}
 ~SplittingLibrary() { clear();}

  // Initialisation.
  void init(Settings* settings, ParticleData* particleData, Rndm* rndm,
    BeamParticle* beamA, BeamParticle* beamB, CoupSM* coupSMPtr, Info* infoPtr,
    Hooks* hooks = NULL);
  void initFSR();
  void initISR();
  void clear();

  void setTimesPtr (DireTimes* fsrIn) {
    for ( map<string,Splitting*>::iterator it = splittings.begin();
    it != splittings.end(); ++it ) it->second->setTimesPtr(fsrIn);

  }
  void setSpacePtr (DireSpace* isrIn) {
    for ( map<string,Splitting*>::iterator it = splittings.begin();
    it != splittings.end(); ++it ) it->second->setSpacePtr(isrIn);
  }

  map< string, Splitting* > getSplittings() { return splittings;}

  // Overload index operator to access element of splitting vector.
  Splitting* operator[](string id);
  const Splitting* operator[](string id) const;

  // Generate name for a splitting
  vector<int> getSplittingRadBefID(const Event& event, int rad, int emt);
  vector<int> getSplittingRadBefID_new(const Event& event, int rad, int emt);

  // Generate name for a splitting
  vector<string> getSplittingName(const Event& event, int rad, int emt);
  vector<string> getSplittingName_new(const Event& event, int rad, int emt);

  // Check number of particles produced in splitting.
  int nEmissions(string name);

  void setKernelHooks(Hooks* hooks) {hooksPtr = hooks;}

  // Some string name hashes, to avoid string conparisons.
  ulong fsrQCD_1_to_1_and_21, fsrQCD_1_to_21_and_1, fsrQCD_21_to_21_and_21a,
        fsrQCD_21_to_21_and_21b, fsrQCD_21_to_1_and_1a, fsrQCD_21_to_1_and_1b,
        fsrQCD_1_to_2_and_1_and_2, fsrQCD_1_to_1_and_1_and_1,
        fsrQCD_1_to_1_and_21_notPartial, fsrQCD_1_to_1_and_21_and_21,
        fsrQCD_21_to_21_and_21_and_21, fsrQCD_21_to_1_and_1_and_21,
        fsrQCD_21_to_2_and_2_and_21, fsrQED_1_to_1_and_22,
        fsrQED_1_to_22_and_1, fsrQED_11_to_11_and_22, fsrQED_11_to_22_and_11,
        fsrQED_22_to_1_and_1a, fsrQED_22_to_1_and_1b, fsrQED_22_to_2_and_2a,
        fsrQED_22_to_2_and_2b, fsrQED_22_to_3_and_3a, fsrQED_22_to_3_and_3b,
        fsrQED_22_to_4_and_4a, fsrQED_22_to_4_and_4b, fsrQED_22_to_5_and_5a,
        fsrQED_22_to_5_and_5b, fsrQED_22_to_11_and_11a,
        fsrQED_22_to_11_and_11b, fsrQED_22_to_13_and_13a,
        fsrQED_22_to_13_and_13b, fsrQED_22_to_15_and_15a,
        fsrQED_22_to_15_and_15b, fsrEWK_1_to_1_and_23, fsrEWK_1_to_23_and_1,
        fsrEWK_23_to_1_and_1a, fsrEWK_23_to_1_and_1b, fsrEWK_24_to_1_and_1a,
        fsrEWK_24_to_1_and_1b, fsrEWK_25_to_24_and_24, isrQCD_1_to_1_and_21,
        isrQCD_21_to_1_and_1, isrQCD_21_to_21_and_21a, isrQCD_21_to_21_and_21b,
        isrQCD_1_to_21_and_1, isrQCD_1_to_2_and_1_and_2,
        isrQCD_1_to_1_and_1_and_1, isrQED_1_to_1_and_22,
        isrQED_11_to_11_and_22, isrQED_1_to_22_and_1, isrQED_11_to_22_and_11,
        isrQED_22_to_1_and_1, isrQED_22_to_11_and_11, isrEWK_1_to_1_and_23;

private:

  map< string, Splitting* > splittings;
  Settings* settingsPtr;
  ParticleData* particleDataPtr;
  Rndm* rndmPtr;
  BeamParticle* beamAPtr;
  BeamParticle* beamBPtr;
  CoupSM* coupSMPtr;
  Info* infoPtr;

  // User may load additional kernels.
  Hooks* hooksPtr;
  bool hasExternalHook;

};

} // end namespace Pythia8

#endif
