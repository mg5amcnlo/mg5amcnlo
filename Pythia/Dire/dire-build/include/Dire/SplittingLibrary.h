
#ifndef Pythia8_SplittingLibrary_H
#define Pythia8_SplittingLibrary_H

#define DIRE_SPLITTINGLIBRARY_VERSION "2.000"

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

  // Generate name for a splitting
  vector<string> getSplittingName(const Event& event, int rad, int emt);

  // Check number of particles produced in splitting.
  int nEmissions(string name);

  void setKernelHooks(Hooks* hooks) {hooksPtr = hooks;}

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
