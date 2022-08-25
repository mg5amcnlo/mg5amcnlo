// Vincia.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains global header information for Vincia.

#ifndef Pythia8_Vincia_H
#define Pythia8_Vincia_H

// Maths headers.
#include <limits>
#include <cmath>

// Include Pythia 8 headers.
#include "Pythia8/Event.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PartonSystems.h"
#include "Pythia8/PhaseSpace.h"
#include "Pythia8/StandardModel.h"
#include "Pythia8/ShowerModel.h"
#include "Pythia8/ExternalMEs.h"

// Include Vincia headers.
#include "Pythia8/VinciaAntennaFunctions.h"
#include "Pythia8/VinciaCommon.h"
#include "Pythia8/VinciaDiagnostics.h"
#include "Pythia8/VinciaEW.h"
#include "Pythia8/VinciaFSR.h"
#include "Pythia8/VinciaISR.h"
#include "Pythia8/VinciaMerging.h"
#include "Pythia8/VinciaMergingHooks.h"
#include "Pythia8/VinciaQED.h"

// Define namespace inside which Vincia lives.
namespace Pythia8 {

//==========================================================================

// The Vincia class. Top-level handler class for the Vincia antenna
// shower model.

class Vincia : public ShowerModel {

public:

  // Constructor.
  Vincia() = default;

  // Empty virtual destructor.
  virtual ~Vincia() override = default;

  // Initialize.
  bool init(MergingPtr mrgPtrIn, MergingHooksPtr mrgHooksPtrIn,
            PartonVertexPtr partonVertexPtrIn,
            WeightContainer* weightContainerPtrIn) override;

  // Function called from Pythia after the beam particles have been set up,
  // so that showers may be initialized after the beams are initialized.
  bool initAfterBeams() override {
    // Initialise QED showers with beams initialised.
    qedShowerHardPtr->init(beamAPtr, beamBPtr);
    qedShowerSoftPtr->init(beamAPtr, beamBPtr);
    ewShowerPtr->init(beamAPtr, beamBPtr);
    return true;
  }

  // Methods to get
  TimeShowerPtr  getTimeShower() const override { return timesPtr; }
  TimeShowerPtr  getTimeDecShower() const override { return timesDecPtr; }
  SpaceShowerPtr getSpaceShower() const override { return spacePtr; }
  MergingHooksPtr getMergingHooks() const override { return mergingHooksPtr; }
  MergingPtr getMerging() const override { return mergingPtr; }

  // End-of-run statistics.
  void onStat() override {
    if (verbose >= VinciaConstants::REPORT) diagnosticsPtr->print(); }

  // Automatically set verbose level in all members.
  void setVerbose(int verboseIn);

  // Public Vincia objects.
  VinciaCommon          vinCom{};
  Resolution            resolution{};
  VinciaModulePtr       ewShowerPtr{};
  VinciaModulePtr       qedShowerHardPtr{};
  VinciaModulePtr       qedShowerSoftPtr{};
  VinciaColour          colour{};
  VinciaWeights         vinWeights{};
  MECs                  mecs{};

  // Auxiliary objects.
  ExternalMEsPlugin     mg5mes{};
  Rambo                 rambo{};

  // Vectors of antenna functions.
  DGLAP         dglap{};
  AntennaSetFSR antennaSetFSR{};
  AntennaSetISR antennaSetISR{};

  // Pointers to Pythia classes.
  SusyLesHouches*    slhaPtr{};
  WeightContainer*   weightContainerPtr{};

 protected:

  // Method to initialise Vincia tune settings
  bool initTune(int iTune);

  // Members for the FSR and ISR showers.
  shared_ptr<VinciaFSR> timesPtr{};
  shared_ptr<VinciaFSR> timesDecPtr{};
  shared_ptr<VinciaISR> spacePtr{};

  // Merging pointers.
  shared_ptr<VinciaMergingHooks> mergingHooksPtr{};
  shared_ptr<VinciaMerging> mergingPtr{};

  // Pointer for diagnostics and profiling.
  shared_ptr<VinciaDiagnostics> diagnosticsPtr{};

 private:

  // Verbosity level.
  int verbose{0};

  // Merging flag.
  bool doMerging{false};

};

//==========================================================================

} // end Pythia8 namespace

#endif // Pythia8_Vincia_H
