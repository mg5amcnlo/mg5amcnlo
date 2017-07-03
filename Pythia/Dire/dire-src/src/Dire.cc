
// DIRE includes.
#include "Dire/Dire.h"

namespace Pythia8 {


void Dire::initSettings( Pythia& pythia ) {

  // Teach Pythia the additional DIRE input settings.
  pythia.settings.addFlag("ShowerPDF:usePDFalphas",false);
  pythia.settings.addFlag("ShowerPDF:usePDFmasses",true);
  pythia.settings.addFlag("ShowerPDF:useSummedPDF",true);
  pythia.settings.addFlag("DireSpace:useGlobalMapIF",false);
  pythia.settings.addFlag("DireSpace:forceMassiveMap",false);
  pythia.settings.addMode("DireTimes:nFinalMax",-10,true,false,-1,10000000);
  pythia.settings.addMode("DireSpace:nFinalMax",-10,true,false,-1,10000000);
  pythia.settings.addMode("DireTimes:kernelOrder",1,true,false,0,10);
  pythia.settings.addMode("DireSpace:kernelOrder",1,true,false,0,10);
  pythia.settings.addMode("DireTimes:kernelOrderMPI",1,true,false,0,10);
  pythia.settings.addMode("DireSpace:kernelOrderMPI",1,true,false,0,10);
  pythia.settings.forceParm("SpaceShower:pT0Ref",0.0);

  // Teach Pythia some enhance factors.
  pythia.settings.addParm("Enhance:fsr_qcd_1->1&21_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_qcd_1->21&1_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_qcd_21->21&21a_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_qcd_21->21&21b_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_qcd_21->1&1a_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_qcd_21->1&1b_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_qcd_1->2&1&2_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_qcd_1->1&1&1_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_qed_1->1&22_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_qed_1->22&1_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_qed_22->1&1a_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_qed_22->1&1b_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_ew_1->1&23_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_ew_1->23&1_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_ew_23->1&1a_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_ew_23->1&1b_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_ew_24->1&1a_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:fsr_ew_24->1&1b_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qcd_1->1&21_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qcd_21->1&1_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qcd_21->21&21a_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qcd_21->21&21b_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qcd_1->21&1_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qcd_1->2&1&2_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qcd_1->1&1&1_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qed_1->1&22_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qed_1->22&1_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qed_22->1&1_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qed_1->22&1_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qed_11->11&22_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qed_11->22&11_CS",1.0,false,false,-1e5,1e5);
  pythia.settings.addParm("Enhance:isr_qed_22->11&11_CS",1.0,false,false,-1e5,1e5);

  // Teach Pythia scale variations.
  pythia.settings.addFlag("Variations:doVariations",false);
  pythia.settings.addParm("Variations:muRisrDown",1.0,false,false,1e-2,1.0);
  pythia.settings.addParm("Variations:muRisrUp",1.0,false,false,1.0,1e2);
  pythia.settings.addParm("Variations:muRfsrDown",1.0,false,false,1e-2,1.0);
  pythia.settings.addParm("Variations:muRfsrUp",1.0,false,false,1.0,1e2);

  // Teach Pythia merging.
  pythia.settings.addFlag("Dire:doMerging",false);
  pythia.settings.addFlag("Dire:doGenerateSubtractions",false);
  pythia.settings.addFlag("Dire:doGenerateSudakovs",false);
  pythia.settings.addFlag("Dire:doMECs",false);
  pythia.settings.addFlag("Dire:doMOPS",false);

  // Teach Pythia MG5 inputs for external MEs
  pythia.settings.addWord("Dire:MG5card", "");

}

void Dire::initShowersAndWeights(Pythia& pythia, Hooks* hooks) {

  // Construct showers.
  if (!weightsPtr)  weightsPtr  = new WeightContainer(&pythia.settings);
  if (!timesPtr)    timesPtr    = new DireTimes(&pythia);
  if (!spacePtr)    spacePtr    = new DireSpace(&pythia);
  if (!timesDecPtr) timesDecPtr = new DireTimes(&pythia);
  hooksPtr = hooks;

  // Initialise weight book-keeping.
  if (!wts) wts = new WeightHooks(weightsPtr);
  timesPtr->setWeightContainerPtr(weightsPtr);
  spacePtr->setWeightContainerPtr(weightsPtr);
  timesDecPtr->setWeightContainerPtr(weightsPtr);

  // Feed new DIRE showers to Pythia.
  pythia.setShowerPtr( timesDecPtr, timesPtr, spacePtr);
  pythia.setUserHooksPtr(wts);

}

void Dire::setup( Pythia& pythia) {

  // Initialise library of splitting functions.
  if (!splittings) splittings = new SplittingLibrary();

  // Reinitialise showers to ensure that pointers are
  // correctly set.
  timesPtr->reinitPtr(&pythia.info,
    &pythia.settings,
    &pythia.particleData,
    &pythia.rndm,
    &pythia.partonSystems,
     wts, pythia.mergingHooksPtr,
     splittings);
  spacePtr->reinitPtr(&pythia.info,
    &pythia.settings,
    &pythia.particleData,
    &pythia.rndm,
    &pythia.partonSystems,
     wts, pythia.mergingHooksPtr,
     splittings);
  timesDecPtr->reinitPtr(&pythia.info,
    &pythia.settings,
    &pythia.particleData,
    &pythia.rndm,
    &pythia.partonSystems,
     wts, pythia.mergingHooksPtr,
     splittings);

  // Reset Pythia masses if necessary.
  if ( pythia.settings.flag("ShowerPDF:usePDFmasses")
    && ( spacePtr->getBeamA() != NULL || spacePtr->getBeamB() != NULL) ) {
    for (int i=1; i <= 5; ++i) {
      // Try to get masses from the hadron beams.
      double mPDF = (abs(spacePtr->getBeamA()->id()) > 30)
                  ? spacePtr->getBeamA()->mQuarkPDF(i)
                  : (abs(spacePtr->getBeamB()->id()) > 30) 
                    ? spacePtr->getBeamB()->mQuarkPDF(i) : -1.0;
      // If there are no hadron beams, get the masses from either beam.
      if (spacePtr->getBeamA() != NULL && mPDF < 0.)
        mPDF = spacePtr->getBeamA()->mQuarkPDF(i);
      if (spacePtr->getBeamB() != NULL && mPDF < 0.)
        mPDF = spacePtr->getBeamB()->mQuarkPDF(i);
      if (mPDF > -1.) {
        stringstream resetMass;
        resetMass << i << ":m0 = " << mPDF;
        pythia.readString(resetMass.str());
      }
    }
  }

  // Switch off all showering and MPI when estimating the cross section,
  if (hooksPtr)
    hooksPtr->initPtr( &pythia.info, &pythia.settings, &pythia.particleData,
      &pythia.rndm, spacePtr->getBeamA(), spacePtr->getBeamB(),
      &pythia.couplings, &pythia.partonSystems);

  splittings->setKernelHooks(hooksPtr);

  // Initialise splitting function library here so that beam pointers
  // are already correctly initialised.
  splittings->init(&pythia.settings, &pythia.particleData, &pythia.rndm,
    spacePtr->getBeamA(), spacePtr->getBeamB(), spacePtr->getCoupSM(),
    &pythia.info, hooksPtr);

  // Feed the splitting functions to the showers.
  splittings->setTimesPtr(timesPtr);
  splittings->setSpacePtr(spacePtr);

  // Initialize splittings in showers again (!), now that splittings are
  // properly set up.
  timesPtr->initSplits();
  spacePtr->initSplits();

}

void Dire::init(Pythia& pythia, char const* settingsFile, int subrun,
  Hooks* hooks) {

  // Initialize new settings.
  initSettings(pythia);

  // Construct showers.
  initShowersAndWeights(pythia, hooks); 

  // Redirect output so that Pythia banner will not be printed twice.
  //std::streambuf *old = cout.rdbuf();       
  //stringstream ss;
  //cout.rdbuf (ss.rdbuf());

  // Read Pythia settings from file and initialise
  // (needed so that we have a well-defined beam particle
  // pointer that can be fed to the splitting functions).
  pythia.readFile(settingsFile, subrun);

  // Setup weight container (after user-defined enhance factors have been read)
  weightsPtr->setup();

  pythia.init();

  // Restore print-out.
  //cout.rdbuf (old);

  setup(pythia);

}

//==========================================================================

} // end namespace Pythia8
