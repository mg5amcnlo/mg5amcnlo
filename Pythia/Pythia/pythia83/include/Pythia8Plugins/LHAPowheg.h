// LHAPowheg.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.
// Author: Philip Ilten, May 2015.

#ifndef Pythia8_LHAPowheg_H
#define Pythia8_LHAPowheg_H

#include "Pythia8Plugins/LHAFortran.h"
#include <sys/stat.h>
#include <unistd.h>

namespace Pythia8 {

//==========================================================================

// Give access to the POWHEG commonblocks and subroutines.

extern "C" {

  // The random number common block.
  extern struct {
    int rnd_numseeds, rnd_initialseed, rnd_iwhichseed;
    char rnd_cwhichseed[4];
    int rnd_i1, rnd_i2;
  } pwhg_rnd_;

  // The RANMAR (R48 modification) common block.
  extern struct {
    double u[97];
    double c;
    int i97, j97;
  } r48st1_;

  // Initialize Powheg.
  void pwhginit_();

  // Reset the counters.
  void resetcnt_(const char *string, int length);

  // Generate an event.
  void pwhgevent_();

  // Access Powheg input data.
  double powheginput_(const char *string, int length);

}

//==========================================================================

// A derived class from LHAupFortran which allows a POWHEGBOX binary
// to be directly interfaced with Pythia via a plugin structure. The
// class PowhegProcs handles the loading of these plugin libraries.

class LHAupPowheg : public LHAupFortran {

public:

  // Constructor.
  LHAupPowheg(Pythia *pythiaIn = nullptr);

  // Call pwhginit and fill the HEPRUP commonblock.
  bool fillHepRup();

  // Call pwhgevent and fill the HEEUP commonblock.
  bool fillHepEup();

private:

  // The external random number generator.
  Rndm* rndm;

  // The run directory.
  string dir;

  // The current working directory.
  char cwd[FILENAME_MAX];

};

//--------------------------------------------------------------------------

// Constructor.

LHAupPowheg::LHAupPowheg(Pythia *pythia) : dir("./") {

  if (pythia && pythia->settings.isWord("POWHEG:dir"))
    dir = pythia->settings.word("POWHEG:dir");
  if (pythia && pythia->settings.flag("POWHEG:pythiaRandom"))
    rndm = &pythia->rndm;
  mkdir(dir.c_str(), 0777);

}

//--------------------------------------------------------------------------

// Call pwhginit and fill the HEPRUP commonblock.

bool LHAupPowheg::fillHepRup() {

  // Set multiple random seeds to none.
  getcwd(cwd, sizeof(cwd));
  chdir(dir.c_str());
  strcpy(pwhg_rnd_.rnd_cwhichseed, "none");

  // Initialize Powheg.
  pwhginit_();

  // Reset all the counters.
  resetcnt_("upper bound failure in inclusive cross section", 46);
  resetcnt_("vetoed calls in inclusive cross section", 39);
  resetcnt_("upper bound failures in generation of radiation", 47);
  resetcnt_("vetoed radiation", 16);
  chdir(cwd);
  return fillHepEup();

}

//--------------------------------------------------------------------------

// Set the random numbers, call pwhgevent, and fill the HEPEUP commonblock.

bool LHAupPowheg::fillHepEup() {

  // Change directory.
  getcwd(cwd, sizeof(cwd));
  chdir(dir.c_str());

  // Reset the random block if requested.
  if (rndm != nullptr) {
    r48st1_.i97 = 97;
    r48st1_.j97 = 33;
    r48st1_.c = rndm->flat();
    for (int i = 0; i < 97; ++i) r48st1_.u[i] = rndm->flat();
  }

  // Generate the event.
  pwhgevent_();
  chdir(cwd);
  return true;

}

//--------------------------------------------------------------------------

// Define external handles to the plugin for dynamic loading.

extern "C" {

  LHAupPowheg* newLHAup(Pythia *pythia) {return new LHAupPowheg(pythia);}

  void deleteLHAup(LHAupPowheg* lha) {delete lha;}

}

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_LHAPowheg_H
