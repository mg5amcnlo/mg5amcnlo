// VinciaCommon.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the headers) for the Vincia class.

#include "Pythia8/Vincia.h"
#include "Pythia8/Merging.h"
#include "Pythia8/MergingHooks.h"

namespace Pythia8 {

using namespace VinciaConstants;

//==========================================================================

// Vincia parton shower class.

//--------------------------------------------------------------------------

// Initialize.

bool Vincia::init(MergingPtr mrgPtrIn, MergingHooksPtr mrgHooksPtrIn,
                  PartonVertexPtr partonVertexPtrIn,
                  WeightContainer* weightContainerPtrIn) {

  // Verbosity output.
  verbose = settingsPtr->mode("Vincia:verbose");
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Create diagnostics pointer.
  diagnosticsPtr = make_shared<VinciaDiagnostics>();
  diagnosticsPtr->initPtr(infoPtr);
  if (verbose >= REPORT) diagnosticsPtr->start(__METHOD_NAME__);

  // Clear Vincia's register of PhysicsBase objects
  subObjects.clear();

  bool vinciaOn     = (settingsPtr->mode("PartonShowers:model") == 2);
  bool doCutMerging = flag("Merging:doCutBasedMerging");
  bool doKTMerging  = flag("Merging:doKTMerging");
  bool doMGMerging  = flag("Merging:doMGMerging");
  doMerging         = flag("Merging:doMerging");
  if ((doCutMerging || doKTMerging || doMGMerging) && !doMerging) {
    doMerging = true;
    settingsPtr->readString("Merging:doMerging = on");
  }
  doMerging       = ( doMerging && vinciaOn );

  // Setup Vincia's merging if requested.
  if (doMerging) {
    // Ensure consistency in settings with merging.
    if (mode("Vincia:ewMode") > 2) {
      infoPtr->errorMsg("Warning from "+__METHOD_NAME__+": Reverting to"
        " default QED mode. EW shower not yet supported by merging.");
      // Use readString so change is reapplied after Vincia tune setting.
      int ewModeDef = settingsPtr->modeDefault("Vincia:ewMode");
      settingsPtr->readString("Vincia:ewMode = "+to_string(ewModeDef));
    }
    if (flag("Vincia:interleaveResDec")) {
      infoPtr->errorMsg("Warning from "+__METHOD_NAME__+": Switching off"
        " interleaved resonance decays. Not yet supported by merging.");
      // Must switch both Vincia and TimeShower flags off, since PartonLevel
      // uses the TimeShower one.
      settingsPtr->readString("Vincia:interleaveResDec = off");
      settingsPtr->readString("TimeShower:interleaveResDec = off");
    }
    // TODO this could be fixed relatively easily.
    if (mode("Vincia:kineMapFFsplit") != 1) {
      infoPtr->errorMsg("Info from "+__METHOD_NAME__+": Forcing"
        " kineMapFFsplit = 1. Others not yet supported"
        " by merging.");
      settingsPtr->readString("Vincia:kineMapFFsplit = 1");
    }

    // Set and register merging pointers
    mergingHooksPtr = make_shared<VinciaMergingHooks>();
    registerSubObject(*mergingHooksPtr);
    mrgHooksPtrIn = mergingHooksPtr;
    mergingPtr = make_shared<VinciaMerging>();
    registerSubObject(*mergingPtr);
    mrgPtrIn = mergingPtr;

    // Initialise Vincia's mergingHookPtr.
    mergingHooksPtr->init();

    if (!mergingHooksPtr->initSuccess()) {
      string msg= ": MergingHooks initialisation failed.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
      return false;
    }

    // Create Vincia's own userhook.
    shared_ptr<MergeResScaleHook> mergeResHookPtr =
      make_shared<MergeResScaleHook>(mergingHooksPtr);

    // Update userHooksPtr.
    if ( !userHooksPtr )
      userHooksPtr = mergeResHookPtr;
    else {
      shared_ptr<UserHooksVector> uhv =
        dynamic_pointer_cast<UserHooksVector>(userHooksPtr);
      if ( !uhv ) {
        uhv = make_shared<UserHooksVector>();
        uhv->hooks.push_back(userHooksPtr);
        userHooksPtr = uhv;
      }
      uhv->hooks.push_back(mergeResHookPtr);
    }

    // Update infoPtr's pointer to userhooks.
    infoPtr->userHooksPtr = userHooksPtr;
  }

  // Set weightContainerPtr and tell weightContainer where to find our weights.
  weightContainerPtr = weightContainerPtrIn;
  if (vinciaOn) weightContainerPtr->weightsShowerPtr = &vinWeights;

  // Create EW/QED Shower module(s).
  int ewMode = settingsPtr->mode("Vincia:EWmode");
  // Create the QED and EW shower pointers.
  ewShowerPtr      = std::make_shared<VinciaEW>();
  qedShowerHardPtr = std::make_shared<VinciaQED>();
  qedShowerSoftPtr = std::make_shared<VinciaQED>();

  if (vinciaOn && ewMode >= 3 && settingsPtr->flag("Vincia:EWOverlapVeto")) {
    // Initialize the overlap veto
    shared_ptr<VinciaEWVetoHook> EWvetoPtr = make_shared<VinciaEWVetoHook>();
    registerSubObject(*EWvetoPtr);
    EWvetoPtr->init(dynamic_pointer_cast<VinciaEW>(ewShowerPtr));

    // Update userHooksPtr.
    if ( !userHooksPtr ) {
      userHooksPtr = EWvetoPtr;
    }
    else {
      shared_ptr<UserHooksVector> uhv =
        dynamic_pointer_cast<UserHooksVector>(userHooksPtr);
      if ( !uhv ) {
        uhv = make_shared<UserHooksVector>();
        uhv->hooks.push_back(userHooksPtr);
        userHooksPtr = uhv;
      }
      uhv->hooks.push_back(EWvetoPtr);
    }

    // Update infoPtr's pointer to userhooks.
    infoPtr->userHooksPtr = userHooksPtr;
  }

  // Create and register VinciaFSR and VinciaISR instances.
  timesPtr = make_shared<VinciaFSR>() ;
  registerSubObject(*timesPtr);
  spacePtr = make_shared<VinciaISR>() ;
  registerSubObject(*spacePtr);
  timesDecPtr = timesPtr;

  // Set pointers in showers.
  timesPtr->initPtrs( mergingHooksPtr, partonVertexPtrIn,
    weightContainerPtr);
  spacePtr->initPtrs( mergingHooksPtr, partonVertexPtrIn,
    weightContainerPtr);

  // Pass verbose settings to members
  setVerbose(verbose);
  if (verbose >= REPORT) printOut(__METHOD_NAME__,
    "setting Vincia pointers...");

  // Init FSR shower pointers and default settings, beyond those set
  // by the non-virtual TimeShower::initPtr().
  timesPtr->initVinciaPtrs(&colour, spacePtr, &mecs,
    &resolution, &vinCom, &vinWeights);
  timesPtr->setDiagnosticsPtr(diagnosticsPtr);

  // Init ISR shower pointers and default settings, beyond those set
  // by the non-virtual SpaceShower::initPtr().
  spacePtr->initVinciaPtrs(&colour, timesPtr, &mecs,
    &resolution, &vinCom, &vinWeights);
  spacePtr->setDiagnosticsPtr(diagnosticsPtr);

  // FSR and ISR antenna sets.
  antennaSetFSR.initPtr(infoPtr, &dglap);
  antennaSetISR.initPtr(infoPtr, &dglap);

  // Hand antenna set pointers to shower and matching objects.
  timesPtr->initAntPtr(&antennaSetFSR);
  spacePtr->initAntPtr(&antennaSetISR);
  mecs.initAntPtr(&antennaSetFSR, &antennaSetISR);

  // Set SLHA pointer
  slhaPtr = coupSUSYPtr->slhaPtr;
  if (slhaPtr == nullptr)
    printOut(__METHOD_NAME__, "Warning: SLHA pointer is null pointer.");

  // Load the matrix element correction plugin.
  string melib = settingsPtr->word("Vincia:MEplugin");
  if (melib.size() > 0)
    mg5mes = ExternalMEsPlugin("libpythia8mg5" + melib + ".so");

  // Pass pointers on to objects that require them.
  rambo.initPtr(rndmPtr);
  vinCom.initPtr(infoPtr);
  resolution.initPtr(settingsPtr, infoPtr, &vinCom);
  mg5mes.initPtrs(infoPtr, slhaPtr);
  mecs.initPtr(infoPtr, &mg5mes, &vinCom, &resolution);
  colour.initPtr(infoPtr);
  vinWeights.initPtr(infoPtr, &vinCom);

  // Initialize pointers in EW shower modules.
  // Set EW/QED Shower module in timesPtr and spacePtr.
  // QED shower for hard interaction + resonance decays.
  qedShowerHardPtr->initPtr(infoPtr, &vinCom);
  timesPtr->setQEDShowerHardPtr(qedShowerHardPtr);
  spacePtr->setQEDShowerHardPtr(qedShowerHardPtr);

  // QED shower for MPI and hadronisation.
  qedShowerSoftPtr->initPtr(infoPtr, &vinCom);
  timesPtr->setQEDShowerSoftPtr(qedShowerSoftPtr);
  spacePtr->setQEDShowerSoftPtr(qedShowerSoftPtr);

  // Electroweak shower.
  ewShowerPtr->initPtr(infoPtr, &vinCom);
  // Save some information on resonances locally,
  // and modify particleDataPtr if doing resonance decays.
  if (ewMode >= 3) ewShowerPtr->load();
  timesPtr->setEWShowerPtr(ewShowerPtr);
  spacePtr->setEWShowerPtr(ewShowerPtr);

  // Now set tune parameters
  int baseTune = settingsPtr->mode("Vincia:Tune");
  if (vinciaOn && baseTune >= 0) {
    // Store user-specified settings before overwriting with tune parameters
    vector<string> userSettings = settingsPtr->getReadHistory();
    if (initTune(baseTune)) {
      // Reapply user settings
      for (int i=0; i<(int)userSettings.size(); ++i) {
        string lineNow      = userSettings[i];
        string lineNowLower = toLower(lineNow);
        if (lineNowLower.find("tune:ee") == string::npos &&
          lineNowLower.find("tune:pp") == string::npos)
          settingsPtr->readString(lineNow);
      }
    }
  }

  // If Vincia is on, allow to override some Pythia settings by
  // Vincia-specific ones.
  if (vinciaOn) {
    // PartonLevel only checks TimeShower:interleaveResDec, so set that to
    // agree with the corresponding Vincia flag.
    bool interleaveResDec = settingsPtr->flag("Vincia:interleaveResDec");
    settingsPtr->flag("TimeShower:interleaveResDec",interleaveResDec);
  }

  // Initialise Vincia auxiliary classes (showers initialised by Pythia).
  vinCom.init();
  resolution.init();
  colour.init();
  vinWeights.init( doMerging );

  // MECs depend on Pythia/SLHA Couplings.
  mecs.init();
  if (!mecs.isInitialised()) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__
      + ": MECs could not be initialised.");
    return false;
  }

  // Print VINCIA header and list of parameters
  if (verbose >= NORMAL && vinciaOn) timesPtr->header();

  // Diagnostics
  if (verbose >= REPORT) diagnosticsPtr->stop(__METHOD_NAME__);

  // Verbose output.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return true;

}

//--------------------------------------------------------------------------

// Vincia tune settings.

bool Vincia::initTune(int iTune) {

  // iTune = 0 : default Vincia tune from Pythia 8.302
  if (iTune == 0) {
    // Z fractions in string breaks
    settingsPtr->parm("StringZ:aLund            ", 0.45 );
    settingsPtr->parm("StringZ:bLund            ", 0.80 );
    settingsPtr->parm("StringZ:aExtraDiquark    ", 0.90 );
    // Z fractions for heavy quarks
    settingsPtr->parm("StringZ:rFactC           ", 1.15 );
    settingsPtr->parm("StringZ:rFactB           ", 0.85 );
    // pT in string breaks
    settingsPtr->parm("StringPT:sigma",            0.305);
    settingsPtr->parm("StringPT:enhancedFraction", 0.01);
    settingsPtr->parm("StringPT:enhancedWidth",    2.0);
    // String breakup flavour parameters
    settingsPtr->parm("StringFlav:probStoUD     ", 0.205);
    settingsPtr->parm("StringFlav:mesonUDvector ", 0.42 );
    settingsPtr->parm("StringFlav:mesonSvector  ", 0.53 );
    settingsPtr->parm("StringFlav:mesonCvector  ", 1.3  );
    settingsPtr->parm("StringFlav:mesonBvector  ", 2.2  );
    settingsPtr->parm("StringFlav:probQQtoQ     ", 0.077);
    settingsPtr->parm("StringFlav:probSQtoQQ    ", 1.0  );
    settingsPtr->parm("StringFlav:probQQ1toQQ0  ", 0.025);
    settingsPtr->parm("StringFlav:etaSup        ", 0.5  );
    settingsPtr->parm("StringFlav:etaPrimeSup   ", 0.1  );
    settingsPtr->parm("StringFlav:decupletSup   ", 1.0  );
    settingsPtr->parm("StringFlav:popcornSpair  ", 0.75 );
    settingsPtr->parm("StringFlav:popcornSmeson ", 0.75 );
    // Primordial kT
    settingsPtr->parm("BeamRemnants:primordialKThard ", 0.4 );
    settingsPtr->parm("BeamRemnants:primordialKTsoft ", 0.25);
    // MB/UE tuning parameters (MPI)
    // Use a "low" alphaS and 2-loop running everywhere, also for MPI
    settingsPtr->parm("SigmaProcess:alphaSvalue ", 0.119);
    settingsPtr->mode("SigmaProcess:alphaSorder ", 2);
    settingsPtr->parm("MultiPartonInteractions:alphaSvalue", 0.119);
    settingsPtr->mode("MultiPartonInteractions:alphaSorder", 2);
    settingsPtr->parm("MultiPartonInteractions:pT0ref     ", 2.24);
    settingsPtr->parm("MultiPartonInteractions:expPow     ", 1.75);
    settingsPtr->parm("MultiPartonInteractions:ecmPow     ", 0.21);
    // Use PYTHIA 8's baseline CR model
    settingsPtr->flag("ColourReconnection:reconnect", true);
    settingsPtr->parm("ColourReconnection:range    ", 1.75);
    // Diffraction: switch off Pythia's perturbative MPI
    // (colours in diffractive systems not yet handled by Vincia)
    settingsPtr->parm("Diffraction:mMinPert", 1000000.0);
    return true;
  }
  // Unknown iTune.
  else return false;
}

//--------------------------------------------------------------------------

// Automatically set verbose level in all members.

void Vincia::setVerbose(int verboseIn) {

  verbose = verboseIn;
  vinCom.setVerbose(verboseIn);
  resolution.setVerbose(verboseIn);
  timesPtr->setVerbose(verboseIn);
  spacePtr->setVerbose(verboseIn);
  colour.setVerbose(verboseIn);
  mecs.setVerbose(verboseIn);
  if (doMerging) {
    mergingHooksPtr->setVerbose(verboseIn);
    mergingPtr->setVerbose(verboseIn);
  }
  if (ewShowerPtr != nullptr) ewShowerPtr->setVerbose(verboseIn);
  if (qedShowerHardPtr != nullptr) qedShowerHardPtr->setVerbose(verboseIn);
  if (qedShowerSoftPtr != nullptr) qedShowerSoftPtr->setVerbose(verboseIn);

}

//==========================================================================

} // end namespace Pythia8
