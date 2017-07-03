
#ifndef Pythia8_Dire_H
#define Pythia8_Dire_H

#define DIRE_VERSION "2.000"

// DIRE includes.
#include "Dire/SplittingLibrary.h"
#include "Dire/Merging.h"
#include "Dire/DireTimes.h"
#include "Dire/DireSpace.h"
#include "Dire/WeightContainer.h"
#include "Dire/Hooks.h"

// Pythia includes.
#include "Pythia8/Pythia.h"
#include <iostream>
#include <sstream>

namespace Pythia8 {

//==========================================================================

// Use UserHooks as trick to initialse the Weight container.

class WeightHooks : public UserHooks {

public:  

  // Constructor and destructor.
  WeightHooks(WeightContainer* weightsIn) : weights(weightsIn) {}
 ~WeightHooks() {}

  bool canVetoResonanceDecays() { return true;}
  bool doVetoResonanceDecays(Event& process) {
    return false;
/*
    if ( !settingsPtr->flag("Top:gg2ttbar")
      && !settingsPtr->flag("Top:qqbar2ttbar"))
      return false;

    if ( settingsPtr->mode("Beams:frameType") == 4)
      return false;

//cout << "blaaaaaaaaaaa" << endl;

    // Count number of b-quarks, leptons and light quarks.
    int nb(0), nl(0), nj(0);
    for (int i=0;i<process.size(); ++i) {
      if (!process[i].isFinal()) continue;
      if (process[i].idAbs() == 5) nb++;
      if (process[i].idAbs() == 11 || process[i].idAbs() == 13 || process[i].idAbs() == 15) nl++;
      if (process[i].idAbs() < 5) nj++;
    }
    if (nb==2 && nl==1 && nj==2) return false;
    if (nb==2 && nl==2 && nj==0) return false;
    return true; */
  }

  bool canVetoProcessLevel() { return true;}
  bool doVetoProcessLevel( Event&) {
    weights->init();
    return false;
  }

  WeightContainer* weights;

};

class Dire {

  public:

  Dire(){}
 ~Dire() { delete wts; delete weightsPtr; delete timesPtr; delete timesDecPtr;
    delete spacePtr; delete splittings;}

  void init(Pythia& pythia, char const* settingsFile, int subrun = -999,
    Hooks* hooks = NULL);
  void initSettings(Pythia& pythia);
  void initShowersAndWeights(Pythia& pythia, Hooks* hooks);
  void setup(Pythia& pythia);

  WeightContainer* weightsPtr;
  DireTimes* timesPtr;
  DireTimes* timesDecPtr;
  DireSpace* spacePtr;
  SplittingLibrary* splittings;

  Hooks* hooksPtr;
  UserHooks* wts;

};

//==========================================================================

} // end namespace Pythia8

#endif
