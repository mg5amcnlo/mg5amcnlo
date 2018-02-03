
#ifndef Pythia8_Dire_H
#define Pythia8_Dire_H

#define DIRE_VERSION "2.002"

// DIRE includes.
#include "Dire/SplittingLibrary.h"
#include "Dire/Merging.h"
#include "Dire/DireTimes.h"
#include "Dire/DireSpace.h"
#include "Dire/WeightContainer.h"
#include "Dire/Hooks.h"
#include "Dire/DireProcesses.h"

// Pythia includes.
#include "Pythia8/Pythia.h"
#include <iostream>
#include <sstream>

namespace Pythia8 {

//==========================================================================

class Dire {

  public:

  Dire() : weightsPtr(NULL), timesPtr(NULL), timesDecPtr(NULL), spacePtr(NULL),
    splittings(NULL), hooksPtr(NULL), userHooksPtr(NULL), hasOwnWeights(false),
    hasOwnTimes(false), hasOwnTimesDec(false), hasOwnSpace(false),
    hasOwnSplittings(false), hasOwnHooks(false), hasUserHooks(false),
    initNewSettings(false) {}

 ~Dire() {
    if (hasOwnWeights)    delete weightsPtr;
    if (hasOwnTimes)      delete timesPtr;
    if (hasOwnTimesDec)   delete timesDecPtr;
    if (hasOwnSpace)      delete spacePtr;
    if (hasOwnSplittings) delete splittings;
  }

  void init(Pythia& pythia, char const* settingsFile = "", int subrun = -999,
    UserHooks* userHooks = NULL, Hooks* hooks = NULL);
  void initSettings(Pythia& pythia);
  void initTune(Pythia& pythia);
  void initShowersAndWeights(Pythia& pythia, UserHooks* userHooks,
    Hooks* hooks);
  void setup(Pythia& pythia);

  WeightContainer* weightsPtr;
  DireTimes* timesPtr;
  DireTimes* timesDecPtr;
  DireSpace* spacePtr;
  SplittingLibrary* splittings;

  Hooks* hooksPtr;
  UserHooks* userHooksPtr;

  DebugInfo debugInfo;

  bool hasOwnWeights, hasOwnTimes, hasOwnTimesDec, hasOwnSpace,
       hasOwnSplittings, hasOwnHooks, hasUserHooks, initNewSettings;

  // Dire-internal hard processes, to overwrite Pythia's hard processes.
  vector<SigmaProcess*> sigmaPtr;

};

//==========================================================================

} // end namespace Pythia8

#endif
