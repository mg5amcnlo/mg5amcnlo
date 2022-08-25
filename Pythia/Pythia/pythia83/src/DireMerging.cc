// DireMerging.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) used for Dire merging.

#include "Pythia8/DireMerging.h"
#include "Pythia8/DireSpace.h"
#include "Pythia8/DireTimes.h"
#include <ctime>

namespace Pythia8 {

//==========================================================================

// The Merging class.

//--------------------------------------------------------------------------


// Check colour/flavour correctness of state.

bool validEvent( const Event& event ) {

  bool validColour  = true;
  bool validCharge  = true;
  bool validMomenta = true;
  double mTolErr=1e-2;

  // Check charge sum in initial and final state
  double initCharge = event[3].charge() + event[4].charge();
  double finalCharge = 0.0;
  for(int i = 0; i < event.size(); ++i)
    if (event[i].isFinal()) finalCharge += event[i].charge();
  if (abs(initCharge-finalCharge) > 1e-12) validCharge = false;

  // Check that overall pT is vanishing.
  Vec4 pSum(0.,0.,0.,0.);
  for ( int i = 0; i < event.size(); ++i) {
    if ( event[i].status() == -21 ) pSum -= event[i].p();
    if ( event[i].isFinal() )       pSum += event[i].p();
  }
  if ( abs(pSum.px()) > mTolErr || abs(pSum.py()) > mTolErr) {
    validMomenta = false;
  }

  if ( event[3].status() == -21
    && (abs(event[3].px()) > mTolErr || abs(event[3].py()) > mTolErr)){
    validMomenta = false;
  }
  if ( event[4].status() == -21
    && (abs(event[4].px()) > mTolErr || abs(event[4].py()) > mTolErr)){
    validMomenta = false;
  }

  return (validColour && validCharge && validMomenta);

}

//--------------------------------------------------------------------------

// Initialise Merging class

void DireMerging::init(){
  // Reset minimal tms value.
  tmsNowMin             = infoPtr->eCM();

  enforceCutOnLHE = settingsPtr->flag("Merging:enforceCutOnLHE");
  doMOPS          = settingsPtr->flag("Dire:doMOPS");
  applyTMSCut     = settingsPtr->flag("Merging:doXSectionEstimate");
  doMerging       = settingsPtr->flag("Dire:doMerging");
  usePDF          = settingsPtr->flag("ShowerPDF:usePDF");
  allowReject     = settingsPtr->flag("Merging:applyVeto");
  doMECs          = settingsPtr->flag("Dire:doMECs");
  doMEM           = settingsPtr->flag("Dire:doMEM");
  doGenerateSubtractions
    = settingsPtr->flag("Dire:doGenerateSubtractions");
  doGenerateMergingWeights
    = settingsPtr->flag("Dire:doGenerateMergingWeights");
  doExitAfterMerging
    = settingsPtr->flag("Dire:doExitAfterMerging");
  allowIncompleteReal
    = settingsPtr->flag("Merging:allowIncompleteHistoriesInReal");
  nQuarksMerge    =  settingsPtr->mode("Merging:nQuarksMerge");

  first = true;

}

//--------------------------------------------------------------------------

// Function to print information.
void DireMerging::statistics() {

  // Recall merging scale value.
  double tmsval         = mergingHooksPtr->tms();
  bool printBanner      = enforceCutOnLHE && tmsNowMin > TMSMISMATCH*tmsval
                        && tmsval > 0.;
  // Reset minimal tms value.
  tmsNowMin             = infoPtr->eCM();

  if (doMOPS) printBanner = false;
  if (doMEM)  printBanner = false;
  if (doMECs) printBanner = false;

  if (!printBanner) return;

  // Header.
  cout << "\n *-------  PYTHIA Matrix Element Merging Information  ------"
       << "-------------------------------------------------------*\n"
       << " |                                                            "
       << "                                                     |\n";
  // Print warning if the minimal tms value of any event was significantly
  // above the desired merging scale value.
  cout << " | Warning in DireMerging::statistics: All Les Houches events"
       << " significantly above Merging:TMS cut. Please check.       |\n";

  // Listing finished.
  cout << " |                                                            "
       << "                                                     |\n"
       << " *-------  End PYTHIA Matrix Element Merging Information -----"
       << "-----------------------------------------------------*" << endl;
}

//--------------------------------------------------------------------------

void DireMerging::storeInfos() {

  // Clear previous information.
  clearInfos();

  // Store information on every possible last clustering.
  for ( int i = 0 ; i < int(myHistory->children.size()); ++i) {
    // Just store pT and mass for now.
    stoppingScalesSave.push_back(myHistory->children[i]->clusterIn.pT());
    radSave.push_back(myHistory->children[i]->clusterIn.radPos());
    emtSave.push_back(myHistory->children[i]->clusterIn.emtPos());
    recSave.push_back(myHistory->children[i]->clusterIn.recPos());
    mDipSave.push_back(myHistory->children[i]->clusterIn.mass());
  }

}

//--------------------------------------------------------------------------

void DireMerging::getStoppingInfo(double scales [100][100],
  double masses [100][100]) {

  int posOffest=2;
  for (unsigned int i=0; i < radSave.size(); ++i){
    scales[radSave[i]-posOffest][recSave[i]-posOffest] = stoppingScalesSave[i];
    masses[radSave[i]-posOffest][recSave[i]-posOffest] = mDipSave[i];
  }

}

double DireMerging::generateSingleSudakov ( double pTbegAll,
  double pTendAll, double m2dip, int idA, int type, double s, double x) {
  return isr->noEmissionProbability( pTbegAll, pTendAll, m2dip, idA,
    type, s, x);
}

//--------------------------------------------------------------------------

void DireMerging::reset() {
  partonSystemsPtr->clear();
  isr->clear();
  fsr->clear();
  beamAPtr->clear();
  beamBPtr->clear();
}

// Function to steer different merging prescriptions.

bool DireMerging::generateUnorderedPoint(Event& process){

  cout << "generate new unordered point" << endl;

  Event newProcess( mergingHooksPtr->bareEvent( process, true) );

  bool found = false;

  // Dryrun to find overestimate enhancements if necessary.
  if (first) {

    isr->dryrun = true;
    fsr->dryrun = true;

    for (int iii=0; iii<50; ++iii) {

      int sizeOld = newProcess.size();
      int sizeNew = newProcess.size();
      int nSplit = 2;

      while (sizeNew < sizeOld+nSplit) {

        // Incoming partons to hard process are stored in slots 3 and 4.
        int inS = 0;
        int inP = 3;
        int inM = 4;
        int sizeProcess = newProcess.size();
        int sizeEvent   = newProcess.size();
        int nOffset  = sizeEvent - sizeProcess;
        int nHardDone = sizeProcess;
        double scale = newProcess.scale();
        // Store participating partons as first set in list of all systems.
        partonSystemsPtr->addSys();
        partonSystemsPtr->setInA(0, inP + nOffset);
        partonSystemsPtr->setInB(0, inM + nOffset);
        for (int i = inM + 1; i < nHardDone; ++i)
          partonSystemsPtr->addOut(0, i + nOffset);
        partonSystemsPtr->setSHat( 0,
          (newProcess[inP + nOffset].p() +
           newProcess[inM + nOffset].p()).m2Calc() );
        partonSystemsPtr->setPTHat( 0, scale);

        // Add incoming hard-scattering partons to list in beam remnants.
        double x1 = newProcess[inP].pPos() / newProcess[inS].m();
        double x2 = newProcess[inM].pNeg() / newProcess[inS].m();
        beamAPtr->append( inP + nOffset, newProcess[inP].id(), x1);
        beamBPtr->append( inM + nOffset, newProcess[inM].id(), x2);
        // Scale. Find whether incoming partons are valence or sea. Store.
        beamAPtr->xfISR( 0, newProcess[inP].id(), x1, scale*scale);
        int vsc1 = beamAPtr->pickValSeaComp();
        beamBPtr->xfISR( 0, newProcess[inM].id(), x2, scale*scale);
        int vsc2 = beamBPtr->pickValSeaComp();
        bool isVal1 = (vsc1 == -3);
        bool isVal2 = (vsc2 == -3);
        infoPtr->setValence( isVal1, isVal2);

        isr->prepare(0, newProcess, true);
        fsr->prepare(0, newProcess, true);

        Event in(newProcess);
        double pTnowFSR = fsr->newPoint(in);
        double pTnowISR = isr->newPoint(in);
        if (pTnowFSR==0. && pTnowISR==0.) { reset(); continue; }
        bool branched=false;
        if (pTnowFSR > pTnowISR) branched = fsr->branch(in);
        else                     branched = isr->branch(in);
        if (!branched) { reset(); continue; }
        if (pTnowFSR > pTnowISR) isr->update(0, in, false);
        else                     fsr->update(0, in, false);

        pTnowISR = isr->newPoint(in);
        pTnowFSR = fsr->newPoint(in);
        if (pTnowFSR==0. && pTnowISR==0.) { reset(); continue; }
        branched=false;
        if (pTnowFSR > pTnowISR) branched = fsr->branch(in);
        else                     branched = isr->branch(in);
        if (!branched) { reset(); continue; }
        if (pTnowFSR > pTnowISR) isr->update(0, in, false);
        else                     fsr->update(0, in, false);

        // Done. Clean up event and return.
        Event bla = fsr->makeHardEvent(0, in, false);
        sizeNew = bla.size();
        partonSystemsPtr->clear();
        isr->clear();
        fsr->clear();
        beamAPtr->clear();
        beamBPtr->clear();
      }
    }

  // Generate arbitrary phase-space point with two additional particles.
  } else {

    int sizeOld = newProcess.size();
    int sizeNew = newProcess.size();
    int nSplit = 2;

    while (sizeNew < sizeOld+nSplit) {

      // Incoming partons to hard process are stored in slots 3 and 4.
      int inS = 0;
      int inP = 3;
      int inM = 4;
      int sizeProcess = newProcess.size();
      int sizeEvent   = newProcess.size();
      int nOffset  = sizeEvent - sizeProcess;
      int nHardDone = sizeProcess;
      double scale = newProcess.scale();
      // Store participating partons as first set in list of all systems.
      partonSystemsPtr->addSys();
      partonSystemsPtr->setInA(0, inP + nOffset);
      partonSystemsPtr->setInB(0, inM + nOffset);
      for (int i = inM + 1; i < nHardDone; ++i)
        partonSystemsPtr->addOut(0, i + nOffset);
      partonSystemsPtr->setSHat( 0,
        (newProcess[inP + nOffset].p() +
      newProcess[inM + nOffset].p()).m2Calc() );
      partonSystemsPtr->setPTHat( 0, scale);

      // Add incoming hard-scattering partons to list in beam remnants.
      double x1 = newProcess[inP].pPos() / newProcess[inS].m();
      double x2 = newProcess[inM].pNeg() / newProcess[inS].m();
      beamAPtr->append( inP + nOffset, newProcess[inP].id(), x1);
      beamBPtr->append( inM + nOffset, newProcess[inM].id(), x2);

      // Scale. Find whether incoming partons are valence or sea. Store.
      // When an x-dependent matter profile is used with nonDiffractive,
      // trial interactions mean that the valence/sea choice has already
      // been made and should be restored here.
      beamAPtr->xfISR( 0, newProcess[inP].id(), x1, scale*scale);
      int vsc1 = beamAPtr->pickValSeaComp();
      beamBPtr->xfISR( 0, newProcess[inM].id(), x2, scale*scale);
      int vsc2 = beamBPtr->pickValSeaComp();
      bool isVal1 = (vsc1 == -3);
      bool isVal2 = (vsc2 == -3);
      infoPtr->setValence( isVal1, isVal2);

      isr->prepare(0, newProcess, true);
      fsr->prepare(0, newProcess, true);

      Event in(newProcess);

      double pTnowFSR = fsr->newPoint(in);
      double pTnowISR = isr->newPoint(in);

      if (pTnowFSR==0. && pTnowISR==0.) { reset(); continue; }
      bool branched=false;
      if (pTnowFSR > pTnowISR) branched = fsr->branch(in);
      else                     branched = isr->branch(in);
      if (!branched) { reset(); continue; }
      if (pTnowFSR > pTnowISR) isr->update(0, in, false);
      else                     fsr->update(0, in, false);

      pTnowISR = isr->newPoint(in);
      pTnowFSR = fsr->newPoint(in);
      if (pTnowFSR==0. && pTnowISR==0.) { reset(); continue; }
      branched=false;
      if (pTnowFSR > pTnowISR) branched = fsr->branch(in);
      else                     branched = isr->branch(in);
      if (!branched) { reset(); continue; }
      if (pTnowFSR > pTnowISR) isr->update(0, in, false);
      else                     fsr->update(0, in, false);

      // Done. Clean up event and return.
      in = fsr->makeHardEvent(0, in, false);
      reset();

      // Loop through event and count.
      int nPartons(0);
      for(int i=0; i < int(in.size()); ++i)
        if ( in[i].isFinal()
          && in[i].colType()!= 0
          && ( in[i].id() == 21 || in[i].idAbs() <= nQuarksMerge))
          nPartons++;
      nPartons -= mergingHooksPtr->hardProcess->nQuarksOut();

      // Set number of requested partons.
      settingsPtr->mode("Merging:nRequested", nPartons);
      mergingHooksPtr->nRequestedSave
        = settingsPtr->mode("Merging:nRequested");
      mergingHooksPtr->reattachResonanceDecays(in);

      generateHistories(in, false);
      reset();
      if (myHistory->foundAnyOrderedPaths()) continue;
      newProcess = in;
      sizeNew = newProcess.size();
      found = true;

    }
  }

  if (!found) {
    // Loop through event and count.
    int nPartons(0);
    for(int i=0; i < int(newProcess.size()); ++i)
      if ( newProcess[i].isFinal()
        && newProcess[i].colType()!= 0
        && ( newProcess[i].id() == 21
             || newProcess[i].idAbs() <= nQuarksMerge))
        nPartons++;
    nPartons -= mergingHooksPtr->hardProcess->nQuarksOut();

    // Set number of requested partons.
    settingsPtr->mode("Merging:nRequested", nPartons);
    mergingHooksPtr->nRequestedSave
      = settingsPtr->mode("Merging:nRequested");
    mergingHooksPtr->reattachResonanceDecays(newProcess);
  }
  process = newProcess;

  cout << "found unordered point" << endl;

  first=false;
  isr->dryrun=false;
  fsr->dryrun=false;

  return true;

}

//--------------------------------------------------------------------------

// Function to steer different merging prescriptions.

int DireMerging::mergeProcess(Event& process){

  // Clear all previous event-by-event information.
  clearInfos();

  int vetoCode = 1;

  // Reinitialise hard process.
  mergingHooksPtr->hardProcess->clear();
  string processNow = settingsPtr->word("Merging:Process");
  mergingHooksPtr->hardProcess->initOnProcess(processNow, particleDataPtr);

  // Remove whitespace from process string
  while(processNow.find(" ", 0) != string::npos)
    processNow.erase(processNow.begin()+processNow.find(" ",0));
  mergingHooksPtr->processSave = processNow;

  mergingHooksPtr->doUserMergingSave
    = settingsPtr->flag("Merging:doUserMerging");
  mergingHooksPtr->doMGMergingSave
    = settingsPtr->flag("Merging:doMGMerging");
  mergingHooksPtr->doKTMergingSave
    = settingsPtr->flag("Merging:doKTMerging");
  mergingHooksPtr->doPTLundMergingSave
    = settingsPtr->flag("Merging:doPTLundMerging");
  mergingHooksPtr->doCutBasedMergingSave
    = settingsPtr->flag("Merging:doCutBasedMerging");
  mergingHooksPtr->doNL3TreeSave
    = settingsPtr->flag("Merging:doNL3Tree");
  mergingHooksPtr->doNL3LoopSave
    = settingsPtr->flag("Merging:doNL3Loop");
  mergingHooksPtr->doNL3SubtSave
    = settingsPtr->flag("Merging:doNL3Subt");
  mergingHooksPtr->doUNLOPSTreeSave
    = settingsPtr->flag("Merging:doUNLOPSTree");
  mergingHooksPtr->doUNLOPSLoopSave
    = settingsPtr->flag("Merging:doUNLOPSLoop");
  mergingHooksPtr->doUNLOPSSubtSave
    = settingsPtr->flag("Merging:doUNLOPSSubt");
  mergingHooksPtr->doUNLOPSSubtNLOSave
    = settingsPtr->flag("Merging:doUNLOPSSubtNLO");
  mergingHooksPtr->doUMEPSTreeSave
    = settingsPtr->flag("Merging:doUMEPSTree");
  mergingHooksPtr->doUMEPSSubtSave
    = settingsPtr->flag("Merging:doUMEPSSubt");
  mergingHooksPtr->nReclusterSave
    = settingsPtr->mode("Merging:nRecluster");

  mergingHooksPtr->hasJetMaxLocal  = false;
  mergingHooksPtr->nJetMaxLocal
    = mergingHooksPtr->nJetMaxSave;
  mergingHooksPtr->nJetMaxNLOLocal
    = mergingHooksPtr->nJetMaxNLOSave;
  mergingHooksPtr->nRequestedSave
    = settingsPtr->mode("Merging:nRequested");

  // Reset to default merging scale.
  mergingHooksPtr->tms(mergingHooksPtr->tmsCut());

  // Ensure that merging weight is not counted twice.
  bool includeWGT = mergingHooksPtr->includeWGTinXSEC();

  // Directly retrive Sudakov w/o PDF factors from showers and exit.
  //if (doMcAtNloDelta && !usePDF) return genSud(process);

  // Possibility to apply merging scale to an input event.
  if ( applyTMSCut && cutOnProcess(process) ) {
    if (includeWGT) infoPtr->weightContainerPtr->setWeightNominal(0.);
    return -1;
  }

  // Done if only a cut should be applied.
  if ( applyTMSCut ) return 1;

  if (doMerging){

    int nPartons = 0;

    // Do not include resonance decay products in the counting.
    Event newp( mergingHooksPtr->bareEvent( process, false) );
    // Loop through event and count.
    for(int i=0; i < int(newp.size()); ++i)
      if ( newp[i].isFinal()
        && newp[i].colType()!= 0
        && ( newp[i].id() == 21 || newp[i].idAbs() <= nQuarksMerge))
        nPartons++;

    int nSteps = mergingHooksPtr->getNumberOfClusteringSteps( newp, false);

    nPartons -= mergingHooksPtr->hardProcess->nQuarksOut();

    // Set number of requested partons.
    settingsPtr->mode("Merging:nRequested", nPartons);

    mergingHooksPtr->hasJetMaxLocal  = false;
    mergingHooksPtr->nJetMaxLocal
      = mergingHooksPtr->nJetMaxSave;
    mergingHooksPtr->nJetMaxNLOLocal
      = mergingHooksPtr->nJetMaxNLOSave;
    mergingHooksPtr->nRequestedSave
      = settingsPtr->mode("Merging:nRequested");

    if (doMEM) {
      int nFinal(0), nQuarks(0), nGammas(0);
      for (int i=0; i < newp.size(); ++i) {
        if (newp[i].idAbs() <   7) nQuarks++;
        if (newp[i].idAbs() == 22) nGammas++;
        if (newp[i].isFinal()) nFinal++;
      }
      settingsPtr->mode("DireSpace:nFinalMax",nFinal-1);
      settingsPtr->mode("DireTimes:nFinalMax",nFinal-1);
      if (nQuarks > 4) return 1;
    }

    // Reset to default merging scale.
    mergingHooksPtr->tms(mergingHooksPtr->tmsCut());

    // For ME corrections, only do mergingHooksPtr reinitialization here,
    // and do not perform any veto.
    if (doMECs) return 1;

    if (doMEM) mergingHooksPtr->orderHistories(false);

    clock_t begin = clock();

    if (doMOPS) generateUnorderedPoint(process);

    bool foundHistories = generateHistories(process, false);
    int returnCode = (foundHistories) ? 1 : 0;

    if (doMOPS && myHistory->foundAnyOrderedPaths() && nSteps > 0)
      returnCode = 0;

    std::clock_t end = std::clock();
    double elapsed_secs_1 = double(end - begin) / CLOCKS_PER_SEC;
    sum_paths += myHistory->goodBranches.size();
    sum_time_1 += elapsed_secs_1;

    // For interative resummed matrix element method, tag histories and exit.
    if (doMEM) {
      tagHistories();
      return 1;
    }

    if (doGenerateSubtractions) calculateSubtractions();
    bool useAll = doMOPS;

    double RNpath = getPathIndex(useAll);
    if ((doMOPS && returnCode > 0) || doGenerateMergingWeights)
      returnCode = calculateWeights(RNpath, useAll);

    end = std::clock();
    double elapsed_secs_2 = double(end - begin) / CLOCKS_PER_SEC;
    sum_time_2 += elapsed_secs_2;

    int tmp_code = getStartingConditions( RNpath, process);
    if (returnCode > 0) returnCode = tmp_code;

    // Ensure that merging weight is not counted twice.
    if (returnCode == 0) {
      mergingHooksPtr->setWeightCKKWL({0.});
      if ( includeWGT) infoPtr->weightContainerPtr->setWeightNominal(0.);
    }

    if (!allowReject && returnCode < 1) returnCode=1;

    // Store information before leaving.
    if (foundHistories) storeInfos();

    if (doMOPS) {
      if (returnCode < 1) mergingHooksPtr->setWeightCKKWL({0.});
     return returnCode;
    }

    // Veto if we do not want to do event generation.
    if (doExitAfterMerging) return -1;

    return 1;
  }

  // Possibility to perform CKKW-L merging on this event.
  if ( mergingHooksPtr->doCKKWLMerging() )
    vetoCode = mergeProcessCKKWL(process);

  // Possibility to perform UMEPS merging on this event.
  if ( mergingHooksPtr->doUMEPSMerging() )
     vetoCode = mergeProcessUMEPS(process);

  // Possibility to perform NL3 NLO merging on this event.
  if ( mergingHooksPtr->doNL3Merging() )
    vetoCode = mergeProcessNL3(process);

  // Possibility to perform UNLOPS merging on this event.
  if ( mergingHooksPtr->doUNLOPSMerging() )
    vetoCode = mergeProcessUNLOPS(process);

  return vetoCode;

}

//--------------------------------------------------------------------------

// Function to perform CKKW-L merging on this event.

int DireMerging::mergeProcessCKKWL( Event& process) {

  // Ensure that merging hooks to not veto events in the trial showers.
  mergingHooksPtr->doIgnoreStep(true);
  // For pp > h, allow cut on state, so that underlying processes
  // can be clustered to gg > h
  if ( mergingHooksPtr->getProcessString().compare("pp>h") == 0 )
    mergingHooksPtr->allowCutOnRecState(true);

  // Construct all histories.
  // This needs to be done because MECs can depend on unordered paths if
  // these unordered paths are ordered up to some point.
  mergingHooksPtr->orderHistories(false);

  // Ensure that merging weight is not counted twice.
  bool includeWGT = mergingHooksPtr->includeWGTinXSEC();

  // Reset weight of the event.
  double wgt = 1.0;
  mergingHooksPtr->setWeightCKKWL({1.});
  mergingHooksPtr->muMI(-1.);

  // Prepare process record for merging. If Pythia has already decayed
  // resonances used to define the hard process, remove resonance decay
  // products.
  Event newProcess( mergingHooksPtr->bareEvent( process, true) );
  // Reset any incoming spins for W+-.
  if (mergingHooksPtr->doWeakClustering())
    for (int i = 0;i < newProcess.size();++i)
      newProcess[i].pol(9);
  // Store candidates for the splitting V -> qqbar'.
  mergingHooksPtr->storeHardProcessCandidates( newProcess);

  // Check if event passes the merging scale cut.
  // Get merging scale in current event.
  // Calculate number of clustering steps.
  int nSteps = mergingHooksPtr->getNumberOfClusteringSteps( newProcess, true);

  double tmsnow = mergingHooksPtr->tmsNow( newProcess );

  // Too few steps can be possible if a chain of resonance decays has been
  // removed. In this case, reject this event, since it will be handled in
  // lower-multiplicity samples.
  int nRequested = mergingHooksPtr->nRequested();

  // Store hard event cut information, reset veto information.
  mergingHooksPtr->setHardProcessInfo(nSteps, tmsnow);
  mergingHooksPtr->setEventVetoInfo(-1, -1.);

  if (nSteps < nRequested && allowReject) {
    if (!includeWGT) mergingHooksPtr->setWeightCKKWL({0.});
    if ( includeWGT) infoPtr->weightContainerPtr->setWeightNominal(0.);
    return -1;
  }

  // Reset the minimal tms value, if necessary.
  tmsNowMin     = (nSteps == 0) ? 0. : min(tmsNowMin, tmsnow);

  // Set dummy process scale.
  newProcess.scale(0.0);
  // Generate all histories.
  DireHistory FullHistory( nSteps, 0.0, newProcess, DireClustering(),
    mergingHooksPtr, (*beamAPtr), (*beamBPtr), particleDataPtr, infoPtr,
    trialPartonLevelPtr, fsr, isr, psweights, coupSMPtr, true, true,
    1.0, 1.0, 1.0, 1.0, 0);

  // Project histories onto desired branches, e.g. only ordered paths.
  FullHistory.projectOntoDesiredHistories();

  // Setup to choose shower starting conditions randomly.
  double sumAll(0.), sumFullAll(0.);
  for ( map<double, DireHistory*>::iterator
    it  = FullHistory.goodBranches.begin();
    it != FullHistory.goodBranches.end(); ++it ) {
    sumAll     += it->second->prodOfProbs;
    sumFullAll += it->second->prodOfProbsFull;
  }
  // Store a double with which to access each of the paths.
  double lastp(0.);
  vector<double> path_index;
  for ( map<double, DireHistory*>::iterator
    it  = FullHistory.goodBranches.begin();
    it != FullHistory.goodBranches.end(); ++it ) {
      // Double to access path.
      double indexNow =  (lastp + 0.5*(it->first - lastp))/sumAll;
      path_index.push_back(indexNow);
      lastp = it->first;
  }
  // Randomly pick path.
  int sizeBranches = FullHistory.goodBranches.size();
  int iPosRN = (sizeBranches > 0)
             ? rndmPtr->pick(
                 vector<double>(sizeBranches, 1./double(sizeBranches)) )
             : 0;
  double RN  = (sizeBranches > 0) ? path_index[iPosRN] : rndmPtr->flat();

  // Setup the selected path. Needed for
  FullHistory.select(RN)->setSelectedChild();

  // Do not apply cut if the configuration could not be projected onto an
  // underlying born configuration.
  bool applyCut = allowReject
                && nSteps > 0 && FullHistory.select(RN)->nClusterings() > 0;

  Event core( FullHistory.lowestMultProc(RN) );
  // Set event-specific merging scale cut. Q2-dependent for DIS.
  if ( mergingHooksPtr->getProcessString().compare("e+p>e+j") == 0
    || mergingHooksPtr->getProcessString().compare("e-p>e-j") == 0 ) {

    // Set dynamical merging scale for DIS
    if (FullHistory.isDIS2to2(core)) {
      int iInEl(0), iOutEl(0);
      for ( int i=0; i < core.size(); ++i )
        if ( core[i].idAbs() == 11 ) {
          if ( core[i].status() == -21 ) iInEl  = i;
          if ( core[i].isFinal() )       iOutEl = i;
        }
      double Q      = sqrt( -(core[iInEl].p() - core[iOutEl].p() ).m2Calc());
      double tmsCut = mergingHooksPtr->tmsCut();
      double tmsEvt = tmsCut / sqrt( 1. + pow( tmsCut/ ( 0.5*Q ), 2)  );
      mergingHooksPtr->tms(tmsEvt);

    } else if (FullHistory.isMassless2to2(core)) {
      double mT(1.);
      for ( int i=0; i < core.size(); ++i )
        if ( core[i].isFinal() ) mT *= core[i].mT();
      double Q      = sqrt(mT);
      double tmsCut = mergingHooksPtr->tmsCut();
      double tmsEvt = tmsCut / sqrt( 1. + pow( tmsCut/ ( 0.5*Q ), 2)  );
      mergingHooksPtr->tms(tmsEvt);
    }
  }
  double tmsval = mergingHooksPtr->tms();

  // Enfore merging scale cut if the event did not pass the merging scale
  // criterion.
  if ( enforceCutOnLHE && applyCut && tmsnow < tmsval ) {
    string message="Warning in DireMerging::mergeProcessCKKWL: "
      "Les Houches Event";
    message+=" fails merging scale cut. Reject event.";
    infoPtr->errorMsg(message);
    if (!includeWGT) mergingHooksPtr->setWeightCKKWL({0.});
    if ( includeWGT) infoPtr->weightContainerPtr->setWeightNominal(0.);
    return -1;
  }

  // Check if more steps should be taken.
  int nFinalP(0), nFinalW(0), nFinalZ(0);
  for ( int i = 0; i < core.size(); ++i )
    if ( core[i].isFinal() ) {
      if ( core[i].colType() != 0 ) nFinalP++;
      if ( core[i].idAbs() == 24 )  nFinalW++;
      if ( core[i].idAbs() == 23 )  nFinalZ++;
    }
  bool complete = (FullHistory.select(RN)->nClusterings() == nSteps) ||
    ( mergingHooksPtr->doWeakClustering() && nFinalP == 2
      && nFinalW+nFinalZ == 0);
  if ( !complete ) {
    string message="Warning in DireMerging::mergeProcessCKKWL: No clusterings";
    message+=" found. History incomplete.";
    infoPtr->errorMsg(message);
  }

  // Calculate CKKWL reweighting for all paths.
  double wgtsum(0.);
  lastp = 0.;
  for ( map<double, DireHistory*>::iterator it =
          FullHistory.goodBranches.begin();
      it != FullHistory.goodBranches.end(); ++it ) {

      // Double to access path.
      double indexNow =  (lastp + 0.5*(it->first - lastp))/sumAll;
      lastp = it->first;

      // Probability of path.
      double probPath = it->second->prodOfProbsFull/sumFullAll;

      FullHistory.select(indexNow)->setSelectedChild();

      // Calculate CKKWL weight:
      double w = FullHistory.weightTREE( trialPartonLevelPtr,
        mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
        mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(),
        indexNow);

      wgtsum += probPath*w;
  }

  wgt = wgtsum;

  // Event with production scales set for further (trial) showering
  // and starting conditions for the shower.
  FullHistory.getStartingConditions( RN, process );
  // If necessary, reattach resonance decay products.
  mergingHooksPtr->reattachResonanceDecays(process);

  // Allow to dampen histories in which the lowest multiplicity reclustered
  // state does not pass the lowest multiplicity cut of the matrix element.
  double dampWeight = mergingHooksPtr->dampenIfFailCuts(
           FullHistory.lowestMultProc(RN) );
  // Save the weight of the event for histogramming. Only change the
  // event weight after trial shower on the matrix element
  // multiplicity event (= in doVetoStep).
  wgt *= dampWeight;

  // Save the weight of the event for histogramming.
  if (!includeWGT) mergingHooksPtr->setWeightCKKWL({wgt});

  // Update the event weight.
  if ( includeWGT) infoPtr->weightContainerPtr->
    setWeightNominal(infoPtr->weight()*wgt);

  // Allow merging hooks to veto events from now on.
  mergingHooksPtr->doIgnoreStep(false);

  // If no-emission probability is zero.
  if ( allowReject && wgt == 0. ) return 0;

  // Done
  return 1;

}

//--------------------------------------------------------------------------

// Function to perform UMEPS merging on this event.

int DireMerging::mergeProcessUMEPS( Event& process) {

  // Initialise which part of UMEPS merging is applied.
  bool doUMEPSTree                = settingsPtr->flag("Merging:doUMEPSTree");
  bool doUMEPSSubt                = settingsPtr->flag("Merging:doUMEPSSubt");
  // Save number of looping steps
  mergingHooksPtr->nReclusterSave = settingsPtr->mode("Merging:nRecluster");
  int nRecluster                  = settingsPtr->mode("Merging:nRecluster");

  // Ensure that merging hooks does not remove emissions.
  mergingHooksPtr->doIgnoreEmissions(true);
  // For pp > h, allow cut on state, so that underlying processes
  // can be clustered to gg > h
  if ( mergingHooksPtr->getProcessString().compare("pp>h") == 0 )
    mergingHooksPtr->allowCutOnRecState(true);
  // For now, prefer construction of ordered histories.
  mergingHooksPtr->orderHistories(true);

  // Ensure that merging weight is not counted twice.
  bool includeWGT = mergingHooksPtr->includeWGTinXSEC();

  // Reset any incoming spins for W+-.
  if (mergingHooksPtr->doWeakClustering())
    for (int i = 0;i < process.size();++i)
      process[i].pol(9);

  // Reset weights of the event.
  double wgt   = 1.;
  mergingHooksPtr->setWeightCKKWL({1.});
  mergingHooksPtr->muMI(-1.);

  // Prepare process record for merging. If Pythia has already decayed
  // resonances used to define the hard process, remove resonance decay
  // products.
  Event newProcess( mergingHooksPtr->bareEvent( process, true) );
  // Store candidates for the splitting V -> qqbar'.
  mergingHooksPtr->storeHardProcessCandidates( newProcess );

  // Check if event passes the merging scale cut.
  double tmsval   = mergingHooksPtr->tms();
  // Get merging scale in current event.
  double tmsnow  = mergingHooksPtr->tmsNow( newProcess );
  // Calculate number of clustering steps.
  int nSteps = mergingHooksPtr->getNumberOfClusteringSteps( newProcess, true);
  int nRequested = mergingHooksPtr->nRequested();

  // Too few steps can be possible if a chain of resonance decays has been
  // removed. In this case, reject this event, since it will be handled in
  // lower-multiplicity samples.
  if (nSteps < nRequested) {
    if (!includeWGT) mergingHooksPtr->setWeightCKKWL({0.});
    if ( includeWGT) infoPtr->weightContainerPtr->setWeightNominal(0.);
    return -1;
  }

  // Reset the minimal tms value, if necessary.
  tmsNowMin      = (nSteps == 0) ? 0. : min(tmsNowMin, tmsnow);

  // Get random number to choose a path.
  double RN = rndmPtr->flat();
  // Set dummy process scale.
  newProcess.scale(0.0);
  // Generate all histories.
  DireHistory FullHistory( nSteps, 0.0, newProcess, DireClustering(),
            mergingHooksPtr,
            (*beamAPtr), (*beamBPtr), particleDataPtr, infoPtr,
            trialPartonLevelPtr, fsr, isr, psweights, coupSMPtr, true, true,
            1.0, 1.0, 1.0, 1.0, 0);
  // Project histories onto desired branches, e.g. only ordered paths.
  FullHistory.projectOntoDesiredHistories();

  // Do not apply cut if the configuration could not be projected onto an
  // underlying born configuration.
  bool applyCut = nSteps > 0 && FullHistory.select(RN)->nClusterings() > 0;

  // Enfore merging scale cut if the event did not pass the merging scale
  // criterion.
  if ( enforceCutOnLHE && applyCut && tmsnow < tmsval ) {
    string message="Warning in DireMerging::mergeProcessUMEPS: "
      "Les Houches Event";
    message+=" fails merging scale cut. Reject event.";
    infoPtr->errorMsg(message);
    if (!includeWGT) mergingHooksPtr->setWeightCKKWL({0.});
    if ( includeWGT) infoPtr->weightContainerPtr->setWeightNominal(0.);
    return -1;
  }

  // Check reclustering steps to correctly apply MPI.
  int nPerformed = 0;
  if ( nSteps > 0 && doUMEPSSubt
    && !FullHistory.getFirstClusteredEventAboveTMS( RN, nRecluster,
          newProcess, nPerformed, false ) ) {
    // Discard if the state could not be reclustered to a state above TMS.
    if (!includeWGT) mergingHooksPtr->setWeightCKKWL({0.});
    if ( includeWGT) infoPtr->weightContainerPtr->setWeightNominal(0.);
    return -1;
  }

  mergingHooksPtr->nMinMPI(nSteps - nPerformed);

  // Calculate CKKWL weight:
  // Perform reweighting with Sudakov factors, save alpha_s ratios and
  // PDF ratio weights.
  if ( doUMEPSTree ) {
    wgt = FullHistory.weight_UMEPS_TREE( trialPartonLevelPtr,
      mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
      mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(), RN);
  } else {
    wgt = FullHistory.weight_UMEPS_SUBT( trialPartonLevelPtr,
      mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
      mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(), RN);
  }

  // Event with production scales set for further (trial) showering
  // and starting conditions for the shower.
  if ( doUMEPSTree ) FullHistory.getStartingConditions( RN, process );
  // Do reclustering (looping) steps.
  else FullHistory.getFirstClusteredEventAboveTMS( RN, nRecluster, process,
    nPerformed, true );

  // Allow to dampen histories in which the lowest multiplicity reclustered
  // state does not pass the lowest multiplicity cut of the matrix element
  double dampWeight = mergingHooksPtr->dampenIfFailCuts(
           FullHistory.lowestMultProc(RN) );
  // Save the weight of the event for histogramming. Only change the
  // event weight after trial shower on the matrix element
  // multiplicity event (= in doVetoStep)
  wgt *= dampWeight;

  // Save the weight of the event for histogramming.
  if (!includeWGT) mergingHooksPtr->setWeightCKKWL({wgt});

  // Update the event weight.
  if ( includeWGT) infoPtr->weightContainerPtr->
    setWeightNominal(infoPtr->weight()*wgt);

  // Set QCD 2->2 starting scale different from arbitrary scale in LHEF!
  // --> Set to minimal mT of partons.
  int nFinal = 0;
  double muf = process[0].e();
  for ( int i=0; i < process.size(); ++i )
  if ( process[i].isFinal()
    && (process[i].colType() != 0 || process[i].id() == 22 ) ) {
    nFinal++;
    muf = min( muf, abs(process[i].mT()) );
  }

  // For pure QCD dijet events (only!), set the process scale to the
  // transverse momentum of the outgoing partons.
  // Calculate number of clustering steps.
  int nStepsNew = mergingHooksPtr->getNumberOfClusteringSteps( process );
  if ( nStepsNew == 0
    && ( mergingHooksPtr->getProcessString().compare("pp>jj") == 0
      || mergingHooksPtr->getProcessString().compare("pp>aj") == 0) )
    process.scale(muf);

  // Reset hard process candidates (changed after clustering a parton).
  mergingHooksPtr->storeHardProcessCandidates( process );
  // If necessary, reattach resonance decay products.
  mergingHooksPtr->reattachResonanceDecays(process);

  // Allow merging hooks to remove emissions from now on.
  mergingHooksPtr->doIgnoreEmissions(false);

  // If no-emission probability is zero.
  if ( wgt == 0. ) return 0;

  // Done
  return 1;

}

//--------------------------------------------------------------------------

// Function to perform NL3 NLO merging on this event.

int DireMerging::mergeProcessNL3( Event& process) {

  // Initialise which part of NL3 merging is applied.
  bool doNL3Tree = settingsPtr->flag("Merging:doNL3Tree");
  bool doNL3Loop = settingsPtr->flag("Merging:doNL3Loop");
  bool doNL3Subt = settingsPtr->flag("Merging:doNL3Subt");

  // Ensure that hooks (NL3 part) to not remove emissions.
  mergingHooksPtr->doIgnoreEmissions(true);
  // Ensure that hooks (CKKWL part) to not veto events in trial showers.
  mergingHooksPtr->doIgnoreStep(true);
  // For pp > h, allow cut on state, so that underlying processes
  // can be clustered to gg > h
  if ( mergingHooksPtr->getProcessString().compare("pp>h") == 0)
    mergingHooksPtr->allowCutOnRecState(true);
  // For now, prefer construction of ordered histories.
  mergingHooksPtr->orderHistories(true);

  // Reset weight of the event
  double wgt      = 1.;
  mergingHooksPtr->setWeightCKKWL({1.});
  // Reset the O(alphaS)-term of the CKKW-L weight.
  double wgtFIRST = 0.;
  mergingHooksPtr->setWeightFIRST({0.});
  mergingHooksPtr->muMI(-1.);

  // Prepare process record for merging. If Pythia has already decayed
  // resonances used to define the hard process, remove resonance decay
  // products.
  Event newProcess( mergingHooksPtr->bareEvent( process, true) );
  // Store candidates for the splitting V -> qqbar'
  mergingHooksPtr->storeHardProcessCandidates( newProcess);

  // Check if event passes the merging scale cut.
  double tmsval  = mergingHooksPtr->tms();
  // Get merging scale in current event.
  double tmsnow  = mergingHooksPtr->tmsNow( newProcess );
  // Calculate number of clustering steps
  int nSteps = mergingHooksPtr->getNumberOfClusteringSteps( newProcess, true);
  int nRequested = mergingHooksPtr->nRequested();

  // Too few steps can be possible if a chain of resonance decays has been
  // removed. In this case, reject this event, since it will be handled in
  // lower-multiplicity samples.
  if (nSteps < nRequested) {
    mergingHooksPtr->setWeightCKKWL({0.});
    mergingHooksPtr->setWeightFIRST({0.});
    return -1;
  }

  // Reset the minimal tms value, if necessary.
  tmsNowMin = (nSteps == 0) ? 0. : min(tmsNowMin, tmsnow);

  // Enfore merging scale cut if the event did not pass the merging scale
  // criterion.
  if ( enforceCutOnLHE && nSteps > 0 && nSteps == nRequested
    && tmsnow < tmsval ) {
    string message = "Warning in DireMerging::mergeProcessNL3: Les Houches";
    message += " Event fails merging scale cut. Reject event.";
    infoPtr->errorMsg(message);
    mergingHooksPtr->setWeightCKKWL({0.});
    mergingHooksPtr->setWeightFIRST({0.});
    return -1;
  }

  // Get random number to choose a path.
  double RN = rndmPtr->flat();
  // Set dummy process scale.
  newProcess.scale(0.0);
  // Generate all histories
  DireHistory FullHistory( nSteps, 0.0, newProcess, DireClustering(),
            mergingHooksPtr,
            (*beamAPtr), (*beamBPtr), particleDataPtr, infoPtr,
            trialPartonLevelPtr, fsr, isr, psweights, coupSMPtr, true, true,
            1.0, 1.0, 1.0, 1.0, 0);
  // Project histories onto desired branches, e.g. only ordered paths.
  FullHistory.projectOntoDesiredHistories();

  // Discard states that cannot be projected unto a state with one less jet.
  if ( nSteps > 0 && doNL3Subt
    && FullHistory.select(RN)->nClusterings() == 0 ){
    mergingHooksPtr->setWeightCKKWL({0.});
    mergingHooksPtr->setWeightFIRST({0.});
    return -1;
  }

  // Potentially recluster real emission jets for powheg input containing
  // "too many" jets, i.e. real-emission kinematics.
  bool containsRealKin = nSteps > nRequested && nSteps > 0;

  // Perform one reclustering for real emission kinematics, then apply merging
  // scale cut on underlying Born kinematics.
  if ( containsRealKin ) {
    Event dummy = Event();
    // Initialise temporary output of reclustering.
    dummy.clear();
    dummy.init( "(hard process-modified)", particleDataPtr );
    dummy.clear();
    // Recluster once.
    if ( !FullHistory.getClusteredEvent( RN, nSteps, dummy )) {
      mergingHooksPtr->setWeightCKKWL({0.});
      mergingHooksPtr->setWeightFIRST({0.});
      return -1;
    }
    double tnowNew  = mergingHooksPtr->tmsNow( dummy );
    // Veto if underlying Born kinematics do not pass merging scale cut.
    if ( enforceCutOnLHE && nRequested > 0 && tnowNew < tmsval ) {
      mergingHooksPtr->setWeightCKKWL({0.});
      mergingHooksPtr->setWeightFIRST({0.});
      return -1;
    }
  }

  // Remember number of jets, to include correct MPI no-emission probabilities.
  if ( doNL3Subt || containsRealKin ) mergingHooksPtr->nMinMPI(nSteps - 1);
  else mergingHooksPtr->nMinMPI(nSteps);

  // Calculate weight
  // Do LO or first part of NLO tree-level reweighting
  if( doNL3Tree ) {
    // Perform reweighting with Sudakov factors, save as ratios and
    // PDF ratio weights
    wgt = FullHistory.weightTREE( trialPartonLevelPtr,
      mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
      mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(), RN);
  } else if( doNL3Loop || doNL3Subt ) {
    // No reweighting, just set event scales properly and incorporate MPI
    // no-emission probabilities.
    wgt = FullHistory.weightLOOP( trialPartonLevelPtr, RN);
  }

  // Event with production scales set for further (trial) showering
  // and starting conditions for the shower
  if ( !doNL3Subt && !containsRealKin )
    FullHistory.getStartingConditions(RN, process);
  // For sutraction of nSteps-additional resolved partons from
  // the nSteps-1 parton phase space, recluster the last parton
  // in nSteps-parton events, and sutract later
  else {
    // Function to return the reclustered event
    if ( !FullHistory.getClusteredEvent( RN, nSteps, process )) {
      mergingHooksPtr->setWeightCKKWL({0.});
      mergingHooksPtr->setWeightFIRST({0.});
      return -1;
    }
  }

  // Allow to dampen histories in which the lowest multiplicity reclustered
  // state does not pass the lowest multiplicity cut of the matrix element
  double dampWeight = mergingHooksPtr->dampenIfFailCuts(
           FullHistory.lowestMultProc(RN) );
  // Save the weight of the event for histogramming. Only change the
  // event weight after trial shower on the matrix element
  // multiplicity event (= in doVetoStep)
  wgt *= dampWeight;

  // For tree level samples in NL3, rescale with k-Factor
  if (doNL3Tree ){
    // Find k-factor
    double kFactor = 1.;
    if( nSteps > mergingHooksPtr->nMaxJetsNLO() )
      kFactor = mergingHooksPtr->kFactor( mergingHooksPtr->nMaxJetsNLO() );
    else kFactor = mergingHooksPtr->kFactor(nSteps);
    // For NLO merging, rescale CKKW-L weight with k-factor
    wgt *= kFactor;
  }

  // Save the weight of the event for histogramming
  mergingHooksPtr->setWeightCKKWL({wgt});

  // Check if we need to subtract the O(\alpha_s)-term. If the number
  // of additional partons is larger than the number of jets for
  // which loop matrix elements are available, do standard CKKW-L
  bool doOASTree = doNL3Tree && nSteps <= mergingHooksPtr->nMaxJetsNLO();

  // Now begin NLO part for tree-level events
  if ( doOASTree ) {
    // Calculate the O(\alpha_s)-term of the CKKWL weight
    wgtFIRST = FullHistory.weightFIRST( trialPartonLevelPtr,
      mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
      mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(), RN,
      rndmPtr );
    // If necessary, also dampen the O(\alpha_s)-term
    wgtFIRST *= dampWeight;
    // Set the subtractive weight to the value calculated so far
    mergingHooksPtr->setWeightFIRST({wgtFIRST});
    // Subtract the O(\alpha_s)-term from the CKKW-L weight
    // If PDF contributions have not been included, subtract these later
    wgt = wgt - wgtFIRST;
  }

  // Set qcd 2->2 starting scale different from arbirtrary scale in LHEF!
  // --> Set to pT of partons
  double pT = 0.;
  for( int i=0; i < process.size(); ++i)
    if(process[i].isFinal() && process[i].colType() != 0) {
      pT = sqrt(pow(process[i].px(),2) + pow(process[i].py(),2));
      break;
    }
  // For pure QCD dijet events (only!), set the process scale to the
  // transverse momentum of the outgoing partons.
  if ( nSteps == 0
    && mergingHooksPtr->getProcessString().compare("pp>jj") == 0)
    process.scale(pT);

  // Reset hard process candidates (changed after clustering a parton).
  mergingHooksPtr->storeHardProcessCandidates( process );
  // If necessary, reattach resonance decay products.
  mergingHooksPtr->reattachResonanceDecays(process);

  // Allow merging hooks (NL3 part) to remove emissions from now on.
  mergingHooksPtr->doIgnoreEmissions(false);
  // Allow merging hooks (CKKWL part) to veto events from now on.
  mergingHooksPtr->doIgnoreStep(false);

  // Done
  return 1;

}

//--------------------------------------------------------------------------

// Function to perform UNLOPS merging on this event.

int DireMerging::mergeProcessUNLOPS( Event& process) {

  // Initialise which part of UNLOPS merging is applied.
  bool nloTilde         = settingsPtr->flag("Merging:doUNLOPSTilde");
  bool doUNLOPSTree     = settingsPtr->flag("Merging:doUNLOPSTree");
  bool doUNLOPSLoop     = settingsPtr->flag("Merging:doUNLOPSLoop");
  bool doUNLOPSSubt     = settingsPtr->flag("Merging:doUNLOPSSubt");
  bool doUNLOPSSubtNLO  = settingsPtr->flag("Merging:doUNLOPSSubtNLO");
  // Save number of looping steps
  mergingHooksPtr->nReclusterSave = settingsPtr->mode("Merging:nRecluster");
  int nRecluster        = settingsPtr->mode("Merging:nRecluster");

  // Ensure that merging hooks to not remove emissions
  mergingHooksPtr->doIgnoreEmissions(true);
  // For now, prefer construction of ordered histories.
  mergingHooksPtr->orderHistories(true);
  // For pp > h, allow cut on state, so that underlying processes
  // can be clustered to gg > h
  if ( mergingHooksPtr->getProcessString().compare("pp>h") == 0)
    mergingHooksPtr->allowCutOnRecState(true);

  // Reset weight of the event.
  double wgt      = 1.;
  mergingHooksPtr->setWeightCKKWL({1.});
  // Reset the O(alphaS)-term of the UMEPS weight.
  double wgtFIRST = 0.;
  mergingHooksPtr->setWeightFIRST({0.});
  mergingHooksPtr->muMI(-1.);

  // Prepare process record for merging. If Pythia has already decayed
  // resonances used to define the hard process, remove resonance decay
  // products.
  Event newProcess( mergingHooksPtr->bareEvent( process, true) );
  // Store candidates for the splitting V -> qqbar'
  mergingHooksPtr->storeHardProcessCandidates( newProcess );

  // Check if event passes the merging scale cut.
  double tmsval  = mergingHooksPtr->tms();
  // Get merging scale in current event.
  double tmsnow  = mergingHooksPtr->tmsNow( newProcess );
  // Calculate number of clustering steps
  int nSteps = mergingHooksPtr->getNumberOfClusteringSteps( newProcess, true);
  int nRequested = mergingHooksPtr->nRequested();

  // Too few steps can be possible if a chain of resonance decays has been
  // removed. In this case, reject this event, since it will be handled in
  // lower-multiplicity samples.
  if (nSteps < nRequested) {
    string message="Warning in DireMerging::mergeProcessUNLOPS: "
      "Les Houches Event";
    message+=" after removing decay products does not contain enough partons.";
    infoPtr->errorMsg(message);
    mergingHooksPtr->setWeightCKKWL({0.});
    mergingHooksPtr->setWeightFIRST({0.});
    return -1;
  }

  // Reset the minimal tms value, if necessary.
  tmsNowMin = (nSteps == 0) ? 0. : min(tmsNowMin, tmsnow);

  // Get random number to choose a path.
  double RN = rndmPtr->flat();
  // Set dummy process scale.
  newProcess.scale(0.0);
  // Generate all histories
  DireHistory FullHistory( nSteps, 0.0, newProcess, DireClustering(),
             mergingHooksPtr,
            (*beamAPtr), (*beamBPtr), particleDataPtr, infoPtr,
            trialPartonLevelPtr, fsr, isr, psweights, coupSMPtr, true, true,
            1.0, 1.0, 1.0, 1.0, 0);
  // Project histories onto desired branches, e.g. only ordered paths.
  FullHistory.projectOntoDesiredHistories();

  // Do not apply cut if the configuration could not be projected onto an
  // underlying born configuration.
  bool applyCut = nSteps > 0 && FullHistory.select(RN)->nClusterings() > 0;

  // Enfore merging scale cut if the event did not pass the merging scale
  // criterion.
  if ( enforceCutOnLHE && applyCut && nSteps == nRequested
    && tmsnow < tmsval && tmsval > 0.) {
    string message="Warning in DireMerging::mergeProcessUNLOPS: Les Houches";
    message+=" Event fails merging scale cut. Reject event.";
    infoPtr->errorMsg(message);
    mergingHooksPtr->setWeightCKKWL({0.});
    mergingHooksPtr->setWeightFIRST({0.});
    return -1;
  }

  // Potentially recluster real emission jets for powheg input containing
  // "too many" jets, i.e. real-emission kinematics.
  bool containsRealKin = nSteps > nRequested && nSteps > 0;
  if ( containsRealKin ) nRecluster += nSteps - nRequested;

  // Remove real emission events without underlying Born configuration from
  // the loop sample, since such states will be taken care of by tree-level
  // samples.
  if ( doUNLOPSLoop && containsRealKin && !allowIncompleteReal
    && FullHistory.select(RN)->nClusterings() == 0 ) {
    mergingHooksPtr->setWeightCKKWL({0.});
    mergingHooksPtr->setWeightFIRST({0.});
    return -1;
  }

  // Discard if the state could not be reclustered to any state above TMS.
  int nPerformed = 0;
  if ( nSteps > 0 && !allowIncompleteReal
    && ( doUNLOPSSubt || doUNLOPSSubtNLO || containsRealKin )
    && !FullHistory.getFirstClusteredEventAboveTMS( RN, nRecluster,
          newProcess, nPerformed, false ) ) {
    mergingHooksPtr->setWeightCKKWL({0.});
    mergingHooksPtr->setWeightFIRST({0.});
    return -1;
  }

  // Check reclustering steps to correctly apply MPI.
  mergingHooksPtr->nMinMPI(nSteps - nPerformed);

  // Perform one reclustering for real emission kinematics, then apply
  // merging scale cut on underlying Born kinematics.
  if ( containsRealKin ) {
    Event dummy = Event();
    // Initialise temporary output of reclustering.
    dummy.clear();
    dummy.init( "(hard process-modified)", particleDataPtr );
    dummy.clear();
    // Recluster once.
    FullHistory.getClusteredEvent( RN, nSteps, dummy );
    double tnowNew  = mergingHooksPtr->tmsNow( dummy );
    // Veto if underlying Born kinematics do not pass merging scale cut.
    if (enforceCutOnLHE && nRequested > 0 && tnowNew < tmsval && tmsval > 0.) {
      string message="Warning in DireMerging::mergeProcessUNLOPS: Les Houches";
      message+=" Event fails merging scale cut. Reject event.";
      infoPtr->errorMsg(message);
      mergingHooksPtr->setWeightCKKWL({0.});
      mergingHooksPtr->setWeightFIRST({0.});
      return -1;
    }
  }

  // New UNLOPS strategy based on UN2LOPS.
  bool doUNLOPS2 = false;
  int depth = -1;

  // Calculate weights.
  // Do LO or first part of NLO tree-level reweighting
  if( doUNLOPSTree ) {
    // Perform reweighting with Sudakov factors, save as ratios and
    // PDF ratio weights
    wgt = FullHistory.weight_UNLOPS_TREE( trialPartonLevelPtr,
            mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
            mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(),
            RN, depth);
  } else if( doUNLOPSLoop ) {
    // Set event scales properly, reweight for new UNLOPS
    wgt = FullHistory.weight_UNLOPS_LOOP( trialPartonLevelPtr,
            mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
            mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(),
            RN, depth);
  } else if( doUNLOPSSubtNLO ) {
    // Set event scales properly, reweight for new UNLOPS
    wgt = FullHistory.weight_UNLOPS_SUBTNLO( trialPartonLevelPtr,
            mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
            mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(),
            RN, depth);
  } else if( doUNLOPSSubt ) {
    // Perform reweighting with Sudakov factors, save as ratios and
    // PDF ratio weights
    wgt = FullHistory.weight_UNLOPS_SUBT( trialPartonLevelPtr,
            mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
            mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(),
            RN, depth);
  }

  // Event with production scales set for further (trial) showering
  // and starting conditions for the shower.
  if (!doUNLOPSSubt && !doUNLOPSSubtNLO && !containsRealKin )
    FullHistory.getStartingConditions(RN, process);
  // Do reclustering (looping) steps.
  else FullHistory.getFirstClusteredEventAboveTMS( RN, nRecluster, process,
    nPerformed, true );

  // Allow to dampen histories in which the lowest multiplicity reclustered
  // state does not pass the lowest multiplicity cut of the matrix element
  double dampWeight = mergingHooksPtr->dampenIfFailCuts(
           FullHistory.lowestMultProc(RN) );
  // Save the weight of the event for histogramming. Only change the
  // event weight after trial shower on the matrix element
  // multiplicity event (= in doVetoStep)
  wgt *= dampWeight;

  // For tree-level or subtractive sammples, rescale with k-Factor
  if ( doUNLOPSTree || doUNLOPSSubt ){
    // Find k-factor
    double kFactor = 1.;
    if ( nSteps > mergingHooksPtr->nMaxJetsNLO() )
      kFactor = mergingHooksPtr->kFactor( mergingHooksPtr->nMaxJetsNLO() );
    else kFactor = mergingHooksPtr->kFactor(nSteps);
    // For NLO merging, rescale CKKW-L weight with k-factor
    wgt *= (nRecluster == 2 && nloTilde) ? 1. : kFactor;
  }

  // Save the weight of the event for histogramming
  mergingHooksPtr->setWeightCKKWL({wgt});

  // Check if we need to subtract the O(\alpha_s)-term. If the number
  // of additional partons is larger than the number of jets for
  // which loop matrix elements are available, do standard UMEPS.
  int nMaxNLO     = mergingHooksPtr->nMaxJetsNLO();
  bool doOASTree  = doUNLOPSTree && nSteps <= nMaxNLO;
  bool doOASSubt  = doUNLOPSSubt && nSteps <= nMaxNLO+1 && nSteps > 0;

  // Now begin NLO part for tree-level events
  if ( doOASTree || doOASSubt ) {

    // Decide on which order to expand to.
    int order = ( nSteps > 0 && nSteps <= nMaxNLO) ? 1 : -1;

    // Exclusive inputs:
    // Subtract only the O(\alpha_s^{n+0})-term from the tree-level
    // subtraction, if we're at the highest NLO multiplicity (nMaxNLO).
    if ( nloTilde && doUNLOPSSubt && nRecluster == 1
      && nSteps == nMaxNLO+1 ) order = 0;

    // Exclusive inputs:
    // Do not remove the O(as)-term if the number of reclusterings
    // exceeds the number of NLO jets, or if more clusterings have
    // been performed.
    if (nloTilde && doUNLOPSSubt && ( nSteps > nMaxNLO+1
      || (nSteps == nMaxNLO+1 && nPerformed != nRecluster) ))
        order = -1;

    // Calculate terms in expansion of the CKKW-L weight.
    wgtFIRST = FullHistory.weight_UNLOPS_CORRECTION( order,
      trialPartonLevelPtr, mergingHooksPtr->AlphaS_FSR(),
      mergingHooksPtr->AlphaS_ISR(), mergingHooksPtr->AlphaEM_FSR(),
      mergingHooksPtr->AlphaEM_ISR(), RN, rndmPtr );

    // Exclusive inputs:
    // Subtract the O(\alpha_s^{n+1})-term from the tree-level
    // subtraction, not the O(\alpha_s^{n+0})-terms.
    if ( nloTilde && doUNLOPSSubt && nRecluster == 1
      && nPerformed == nRecluster && nSteps <= nMaxNLO )
      wgtFIRST += 1.;

    // If necessary, also dampen the O(\alpha_s)-term
    wgtFIRST *= dampWeight;

    // Set the subtractive weight to the value calculated so far
    mergingHooksPtr->setWeightFIRST({wgtFIRST});
    // Subtract the O(\alpha_s)-term from the CKKW-L weight
    // If PDF contributions have not been included, subtract these later
    // New UNLOPS based on UN2LOPS.
    if (doUNLOPS2 && order > -1) wgt = -wgt*(wgtFIRST-1.);
    else if (order > -1) wgt = wgt - wgtFIRST;

  }

  // Set QCD 2->2 starting scale different from arbitrary scale in LHEF!
  // --> Set to minimal mT of partons.
  int nFinal = 0;
  double muf = process[0].e();
  for ( int i=0; i < process.size(); ++i )
  if ( process[i].isFinal()
    && (process[i].colType() != 0 || process[i].id() == 22 ) ) {
    nFinal++;
    muf = min( muf, abs(process[i].mT()) );
  }
  // For pure QCD dijet events (only!), set the process scale to the
  // transverse momentum of the outgoing partons.
  if ( nSteps == 0 && nFinal == 2
    && ( mergingHooksPtr->getProcessString().compare("pp>jj") == 0
      || mergingHooksPtr->getProcessString().compare("pp>aj") == 0) )
    process.scale(muf);

  // Reset hard process candidates (changed after clustering a parton).
  mergingHooksPtr->storeHardProcessCandidates( process );

  // Check if resonance structure has been changed
  //  (e.g. because of clustering W/Z/gluino)
  vector <int> oldResonance;
  for ( int i=0; i < newProcess.size(); ++i )
    if ( newProcess[i].status() == 22 )
      oldResonance.push_back(newProcess[i].id());
  vector <int> newResonance;
  for ( int i=0; i < process.size(); ++i )
    if ( process[i].status() == 22 )
      newResonance.push_back(process[i].id());
  // Compare old and new resonances
  for ( int i=0; i < int(oldResonance.size()); ++i )
    for ( int j=0; j < int(newResonance.size()); ++j )
      if ( newResonance[j] == oldResonance[i] ) {
        oldResonance[i] = 99;
        break;
      }
  bool hasNewResonances = (newResonance.size() != oldResonance.size());
  for ( int i=0; i < int(oldResonance.size()); ++i )
    hasNewResonances = (oldResonance[i] != 99);

  // If necessary, reattach resonance decay products.
  if (!hasNewResonances) mergingHooksPtr->reattachResonanceDecays(process);

  // Allow merging hooks to remove emissions from now on.
  mergingHooksPtr->doIgnoreEmissions(false);

  // If no-emission probability is zero.
  if ( wgt == 0. ) return 0;

  // If the resonance structure of the process has changed due to reclustering,
  // redo the resonance decays in Pythia::next()
  if (hasNewResonances) return 2;

  // Done
  return 1;

}

//--------------------------------------------------------------------------

// Function to set up all histories for an event.

bool DireMerging::generateHistories( const Event& process, bool orderedOnly) {

  // Input not valid.
  if (!validEvent(process)) {
    cout << "Warning in DireMerging::generateHistories: Input event "
         << "has invalid flavour or momentum structure, thus reject. " << endl;
    return false;
  }

  // Clear previous history.
  if (myHistory) delete myHistory;

  // For now, prefer construction of ordered histories.
  mergingHooksPtr->orderHistories(orderedOnly);

  if (doMOPS) mergingHooksPtr->orderHistories(false);

  // For pp > h, allow cut on state, so that underlying processes
  // can be clustered to gg > h
  if ( mergingHooksPtr->getProcessString().compare("pp>h") == 0)
    mergingHooksPtr->allowCutOnRecState(true);

  // Prepare process record for merging. If Pythia has already decayed
  // resonances used to define the hard process, remove resonance decay
  // products.
  Event newProcess( mergingHooksPtr->bareEvent( process, true) );
  // Store candidates for the splitting V -> qqbar'
  mergingHooksPtr->storeHardProcessCandidates( newProcess );

  // Calculate number of clustering steps
  int nSteps = mergingHooksPtr->getNumberOfClusteringSteps( newProcess, true);

  nSteps++;

  // Set dummy process scale.
  newProcess.scale(0.0);
  // Generate all histories
  myHistory = new DireHistory( nSteps, 0.0, newProcess, DireClustering(),
            mergingHooksPtr,
            (*beamAPtr), (*beamBPtr), particleDataPtr, infoPtr,
            trialPartonLevelPtr, fsr, isr, psweights, coupSMPtr, true, true,
            1.0, 1.0, 1.0, 1.0, 0);
  // Project histories onto desired branches, e.g. only ordered paths.
  bool foundHistories = myHistory->projectOntoDesiredHistories();

  // Done
  return (doMOPS ? foundHistories : true);

}

//--------------------------------------------------------------------------

void DireMerging::tagHistories() {

  // Tag history paths as "signal" or "background"
  for ( map<double, DireHistory*>::iterator it =
    myHistory->goodBranches.begin();
    it != myHistory->goodBranches.end(); ++it )
    it->second->tagPath(it->second);

  double sumAll(0.), lastp(0.);
  for ( map<double, DireHistory*>::iterator it =
    myHistory->goodBranches.begin();
    it != myHistory->goodBranches.end(); ++it ) {
    sumAll     += it->second->prodOfProbs;
  }

  // Sum up signal and background probabilities.
  vector<double> sumSignalProb(createvector<double>(0.)(0.)(0.)),
    sumBkgrndProb(createvector<double>(0.)(0.)(0.));

  for ( map<double, DireHistory*>::iterator it =
          myHistory->goodBranches.begin();
    it != myHistory->goodBranches.end(); ++it ) {

    if (it->second == myHistory) continue;

    // Get ME weight.
    double prob = it->second->prodOfProbsFull;
    // Reweight with Sudakovs, couplings and PDFs.
    double indexNow =  (lastp + 0.5*(it->first - lastp))/sumAll;
    lastp = it->first;
    myHistory->select(indexNow)->setSelectedChild();
    vector<double> w = myHistory->weightMEM( trialPartonLevelPtr,
      mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaEM_FSR(), indexNow);
    for (unsigned int i=0; i < w.size(); ++i) {
      totalProbSave[i] += prob*w[i];
      if (it->second->hasTag("higgs")) signalProbSave["higgs"][i] += prob*w[i];
      else                             bkgrndProbSave["higgs"][i] += prob*w[i];
      if (it->second->hasTag("qed"))   signalProbSave["qed"][i]   += prob*w[i];
      else                             bkgrndProbSave["qed"][i]   += prob*w[i];
      if (it->second->hasTag("qcd"))   signalProbSave["qcd"][i]   += prob*w[i];
      else                             bkgrndProbSave["qcd"][i]   += prob*w[i];

      if (it->second->hasTag("higgs") ) signalProbSave["higgs-subt"][i]
                                          += prob*(w[i]-1.);
      else                              bkgrndProbSave["higgs-subt"][i]
                                          += prob*(w[i]-1.);
      if (it->second->hasTag("higgs") ) signalProbSave["higgs-nosud"][i]
                                          += prob;
      else                              bkgrndProbSave["higgs-nosud"][i]
                                          += prob;
    }
  }

}

//--------------------------------------------------------------------------

double DireMerging::getPathIndex( bool useAll) {

  if (!useAll) return rndmPtr->flat();

  // Setup to choose shower starting conditions randomly.
  double sumAll(0.);
  for ( map<double, DireHistory*>::iterator it =
    myHistory->goodBranches.begin();
    it != myHistory->goodBranches.end(); ++it ) {
    sumAll     += it->second->prodOfProbs;
  }
  // Store a double with which to access each of the paths.
  double lastp(0.);
  vector<double> path_index;
  for ( map<double, DireHistory*>::iterator it =
      myHistory->goodBranches.begin();
      it != myHistory->goodBranches.end(); ++it ) {
      // Double to access path.
      double indexNow =  (lastp + 0.5*(it->first - lastp))/sumAll;
      path_index.push_back(indexNow);
      lastp = it->first;
  }
  // Randomly pick path.
  int sizeBranches = myHistory->goodBranches.size();
  int iPosRN = (sizeBranches > 0)
             ? rndmPtr->pick(
                 vector<double>(sizeBranches, 1./double(sizeBranches)) )
             : 0;
  double RN  = (sizeBranches > 0) ? path_index[iPosRN] : rndmPtr->flat();
  return RN;
}

//--------------------------------------------------------------------------

// Function to set up all histories for an event.

bool DireMerging::calculateSubtractions() {

  // Store shower subtractions.
  clearSubtractions();
  for ( int i = 0 ; i < int(myHistory->children.size()); ++i) {

    // Need to reattach resonance decays, if necessary.
    Event psppoint = myHistory->children[i]->state;
    // Reset hard process candidates (changed after clustering a parton).
    mergingHooksPtr->storeHardProcessCandidates( psppoint );

    // Check if resonance structure has been changed
    //  (e.g. because of clustering W/Z/gluino)
    vector <int> oldResonance;
    for ( int n=0; n < myHistory->state.size(); ++n )
      if ( myHistory->state[n].status() == 22 )
        oldResonance.push_back(myHistory->state[n].id());
    vector <int> newResonance;
    for ( int n=0; n < psppoint.size(); ++n )
      if ( psppoint[n].status() == 22 )
        newResonance.push_back(psppoint[n].id());
    // Compare old and new resonances
    for ( int n=0; n < int(oldResonance.size()); ++n )
      for ( int m=0; m < int(newResonance.size()); ++m )
        if ( newResonance[m] == oldResonance[n] ) {
          oldResonance[n] = 99;
          break;
        }
    bool hasNewResonances = (newResonance.size() != oldResonance.size());
    for ( int n=0; n < int(oldResonance.size()); ++n )
      hasNewResonances = (oldResonance[n] != 99);

    // If necessary, reattach resonance decay products.
    if (!hasNewResonances) mergingHooksPtr->reattachResonanceDecays(psppoint);
    else {
      cout << "Warning in DireMerging::generateHistories: Resonance "
           << "structure changed due to clustering. Cannot attach decay "
           << "products correctly." << endl;
    }

    double prob = myHistory->children[i]->clusterProb;

    // Switch from 4pi to 8pi convention
    prob *= 2.;

    // Get clustering variables.
    map<string,double> stateVars;
    int rad = myHistory->children[i]->clusterIn.radPos();
    int emt = myHistory->children[i]->clusterIn.emtPos();
    int rec = myHistory->children[i]->clusterIn.recPos();

    bool isFSR = myHistory->showers->timesPtr->isTimelike(myHistory->state,
      rad, emt, rec, "");
    if (isFSR)
      stateVars = myHistory->showers->timesPtr->getStateVariables(
        myHistory->state,rad,emt,rec,"");
    else
      stateVars = myHistory->showers->spacePtr->getStateVariables(
        myHistory->state,rad,emt,rec,"");

    double z = stateVars["z"];
    double t = stateVars["t"];

    double m2dip = abs
      (-2.*myHistory->state[emt].p()*myHistory->state[rad].p()
       -2.*myHistory->state[emt].p()*myHistory->state[rec].p()
       +2.*myHistory->state[rad].p()*myHistory->state[rec].p());
    double kappa2 = t/m2dip;
    double xCS        = (z*(1-z) - kappa2) / (1 -z);

    // For II dipoles, scale with 1/xCS.
    prob *= 1./xCS;

    // Multiply with ME correction.
    prob *= myHistory->MECnum/myHistory->MECden;

    // Attach point to list of shower subtractions.
    appendSubtraction( prob, psppoint);

  }

  // Restore stored hard process candidates
  mergingHooksPtr->storeHardProcessCandidates(  myHistory->state );

  // Done
  return true;

}

//--------------------------------------------------------------------------

// Function to calulate the weights used for UNLOPS merging.

int DireMerging::calculateWeights( double RNpath, bool useAll ) {

  // Initialise which part of UNLOPS merging is applied.
  bool nloTilde         = settingsPtr->flag("Merging:doUNLOPSTilde");
  bool doUNLOPSTree     = settingsPtr->flag("Merging:doUNLOPSTree");
  bool doUNLOPSLoop     = settingsPtr->flag("Merging:doUNLOPSLoop");
  bool doUNLOPSSubt     = settingsPtr->flag("Merging:doUNLOPSSubt");
  bool doUNLOPSSubtNLO  = settingsPtr->flag("Merging:doUNLOPSSubtNLO");
  // Save number of looping steps
  mergingHooksPtr->nReclusterSave = settingsPtr->mode("Merging:nRecluster");
  int nRecluster        = settingsPtr->mode("Merging:nRecluster");

  // Ensure that merging hooks to not remove emissions
  mergingHooksPtr->doIgnoreEmissions(true);
  mergingHooksPtr->setWeightCKKWL({1.});
  mergingHooksPtr->setWeightFIRST({0.});

  // Reset weight of the event.
  double wgt      = 1.;
  // Reset the O(alphaS)-term of the UMEPS weight.
  double wgtFIRST = 0.;
  mergingHooksPtr->muMI(-1.);

  // Check if event passes the merging scale cut.
  double tmsval  = mergingHooksPtr->tms();

  if (doMOPS) tmsval = 0.;

  // Get merging scale in current event.
  double tmsnow  = mergingHooksPtr->tmsNow( myHistory->state );
  // Calculate number of clustering steps
  int nSteps = mergingHooksPtr->getNumberOfClusteringSteps( myHistory->state,
                                                            true);
  int nRequested = mergingHooksPtr->nRequested();

  if (doMOPS && nSteps == 0) { return 1; }

  // Too few steps can be possible if a chain of resonance decays has been
  // removed. In this case, reject this event, since it will be handled in
  // lower-multiplicity samples.
  if (nSteps < nRequested) {
    string message="Warning in DireMerging::calculateWeights: "
      "Les Houches Event";
    message+=" after removing decay products does not contain enough partons.";
    infoPtr->errorMsg(message);
    if (allowReject) return -1;
  }

  // Reset the minimal tms value, if necessary.
  tmsNowMin = (nSteps == 0) ? 0. : min(tmsNowMin, tmsnow);

  // Do not apply cut if the configuration could not be projected onto an
  // underlying born configuration.
  bool applyCut = nSteps > 0 && myHistory->select(RNpath)->nClusterings() > 0;

  // Enfore merging scale cut if the event did not pass the merging scale
  // criterion.
  if ( enforceCutOnLHE && applyCut && nSteps == nRequested
    && tmsnow < tmsval && tmsval > 0.) {
    string message="Warning in DireMerging::calculateWeights: Les Houches";
    message+=" Event fails merging scale cut. Reject event.";
    infoPtr->errorMsg(message);
    if (allowReject) return -1;
    //return -1;
  }

  // Potentially recluster real emission jets for powheg input containing
  // "too many" jets, i.e. real-emission kinematics.
  bool containsRealKin = nSteps > nRequested && nSteps > 0;
  if ( containsRealKin ) nRecluster += nSteps - nRequested;

  // Remove real emission events without underlying Born configuration from
  // the loop sample, since such states will be taken care of by tree-level
  // samples.
  if ( doUNLOPSLoop && containsRealKin && !allowIncompleteReal
    && myHistory->select(RNpath)->nClusterings() == 0 ) {
    if (allowReject) return -1;
  }

  // Discard if the state could not be reclustered to any state above TMS.
  int nPerformed = 0;
  if ( nSteps > 0 && !allowIncompleteReal
    && ( doUNLOPSSubt || doUNLOPSSubtNLO || containsRealKin )
    && !myHistory->getFirstClusteredEventAboveTMS( RNpath, nRecluster,
          myHistory->state, nPerformed, false ) ) {
    if (allowReject) return -1;
  }

  // Check reclustering steps to correctly apply MPI.
  mergingHooksPtr->nMinMPI(nSteps - nPerformed);

  // Perform one reclustering for real emission kinematics, then apply
  // merging scale cut on underlying Born kinematics.
  if ( containsRealKin ) {
    Event dummy = Event();
    // Initialise temporary output of reclustering.
    dummy.clear();
    dummy.init( "(hard process-modified)", particleDataPtr );
    dummy.clear();
    // Recluster once.
    myHistory->getClusteredEvent( RNpath, nSteps, dummy );
    double tnowNew  = mergingHooksPtr->tmsNow( dummy );
    // Veto if underlying Born kinematics do not pass merging scale cut.
    if (enforceCutOnLHE && nRequested > 0 && tnowNew < tmsval && tmsval > 0.) {
      string message="Warning in DireMerging::calculateWeights: Les Houches";
      message+=" Event fails merging scale cut. Reject event.";
      infoPtr->errorMsg(message);
      if (allowReject) return -1;
      //return -1;
    }
  }

  // Setup to choose shower starting conditions randomly.
  double sumAll(0.), sumFullAll(0.);
  for ( map<double, DireHistory*>::iterator it =
    myHistory->goodBranches.begin();
    it != myHistory->goodBranches.end(); ++it ) {
    sumAll     += it->second->prodOfProbs;
    sumFullAll += it->second->prodOfProbsFull;
  }

  // New UNLOPS strategy based on UN2LOPS.
  bool doUNLOPS2 = false;
  int depth = -1;

  if (!useAll) {

    // Calculate weights.
    if (doMOPS)
      wgt = myHistory->weightMOPS( trialPartonLevelPtr,
              mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaEM_FSR(),
              RNpath);
    else if ( mergingHooksPtr->doCKKWLMerging() )
      wgt = myHistory->weightTREE( trialPartonLevelPtr,
              mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
              mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(),
              RNpath);
    else if (  mergingHooksPtr->doUMEPSTreeSave )
      wgt = myHistory->weight_UMEPS_TREE( trialPartonLevelPtr,
              mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
              mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(),
              RNpath);
    else if ( mergingHooksPtr->doUMEPSSubtSave )
      wgt = myHistory->weight_UMEPS_SUBT( trialPartonLevelPtr,
              mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
              mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(),
              RNpath);
    else if ( mergingHooksPtr->doUNLOPSTreeSave )
      wgt = myHistory->weight_UNLOPS_TREE( trialPartonLevelPtr,
              mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
              mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(),
              RNpath, depth);
    else if ( mergingHooksPtr->doUNLOPSLoopSave )
      wgt = myHistory->weight_UNLOPS_LOOP( trialPartonLevelPtr,
              mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
              mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(),
              RNpath, depth);
    else if ( mergingHooksPtr->doUNLOPSSubtNLOSave )
      wgt = myHistory->weight_UNLOPS_SUBTNLO( trialPartonLevelPtr,
              mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
              mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(),
              RNpath, depth);
    else if ( mergingHooksPtr->doUNLOPSSubtSave )
      wgt = myHistory->weight_UNLOPS_SUBT( trialPartonLevelPtr,
              mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaS_ISR(),
              mergingHooksPtr->AlphaEM_FSR(), mergingHooksPtr->AlphaEM_ISR(),
              RNpath, depth);

    // For tree-level or subtractive sammples, rescale with k-Factor
    if ( doUNLOPSTree || doUNLOPSSubt ){
      // Find k-factor
      double kFactor = 1.;
      if ( nSteps > mergingHooksPtr->nMaxJetsNLO() )
        kFactor = mergingHooksPtr->kFactor( mergingHooksPtr->nMaxJetsNLO() );
      else kFactor = mergingHooksPtr->kFactor(nSteps);
      // For NLO merging, rescale CKKW-L weight with k-factor
      wgt *= (nRecluster == 2 && nloTilde) ? 1. : kFactor;
    }

  } else if (doMOPS) {
    // Calculate CKKWL reweighting for all paths.
    double wgtsum(0.);
    double lastp(0.);

    for ( map<double, DireHistory*>::iterator it =
      myHistory->goodBranches.begin();
      it != myHistory->goodBranches.end(); ++it ) {

      // Double to access path.
      double indexNow =  (lastp + 0.5*(it->first - lastp))/sumAll;
      lastp = it->first;

      // Probability of path.
      double probPath = it->second->prodOfProbsFull/sumFullAll;

      myHistory->select(indexNow)->setSelectedChild();

      // Calculate CKKWL weight:
      double w = myHistory->weightMOPS( trialPartonLevelPtr,
        mergingHooksPtr->AlphaS_FSR(), mergingHooksPtr->AlphaEM_FSR(),
        indexNow);

      wgtsum += probPath*w;
    }
    wgt = wgtsum;

  }

  mergingHooksPtr->setWeightCKKWL({wgt});

  // Check if we need to subtract the O(\alpha_s)-term. If the number
  // of additional partons is larger than the number of jets for
  // which loop matrix elements are available, do standard UMEPS.
  int nMaxNLO     = mergingHooksPtr->nMaxJetsNLO();
  bool doOASTree  = doUNLOPSTree && nSteps <= nMaxNLO;
  bool doOASSubt  = doUNLOPSSubt && nSteps <= nMaxNLO+1 && nSteps > 0;

  // Now begin NLO part for tree-level events
  if ( doOASTree || doOASSubt ) {

    // Decide on which order to expand to.
    int order = ( nSteps > 0 && nSteps <= nMaxNLO) ? 1 : -1;

    // Exclusive inputs:
    // Subtract only the O(\alpha_s^{n+0})-term from the tree-level
    // subtraction, if we're at the highest NLO multiplicity (nMaxNLO).
    if ( nloTilde && doUNLOPSSubt && nRecluster == 1
      && nSteps == nMaxNLO+1 ) order = 0;

    // Exclusive inputs:
    // Do not remove the O(as)-term if the number of reclusterings
    // exceeds the number of NLO jets, or if more clusterings have
    // been performed.
    if (nloTilde && doUNLOPSSubt && ( nSteps > nMaxNLO+1
      || (nSteps == nMaxNLO+1 && nPerformed != nRecluster) ))
        order = -1;

    // Calculate terms in expansion of the CKKW-L weight.
    wgtFIRST = myHistory->weight_UNLOPS_CORRECTION( order,
      trialPartonLevelPtr, mergingHooksPtr->AlphaS_FSR(),
      mergingHooksPtr->AlphaS_ISR(), mergingHooksPtr->AlphaEM_FSR(),
      mergingHooksPtr->AlphaEM_ISR(), RNpath, rndmPtr );

    // Exclusive inputs:
    // Subtract the O(\alpha_s^{n+1})-term from the tree-level
    // subtraction, not the O(\alpha_s^{n+0})-terms.
    if ( nloTilde && doUNLOPSSubt && nRecluster == 1
      && nPerformed == nRecluster && nSteps <= nMaxNLO )
      wgtFIRST += 1.;

    // Subtract the O(\alpha_s)-term from the CKKW-L weight
    // If PDF contributions have not been included, subtract these later
    // New UNLOPS based on UN2LOPS.
    if (doUNLOPS2 && order > -1) wgt = -wgt*(wgtFIRST-1.);
    else if (order > -1) wgt = wgt - wgtFIRST;

  }

  // If no-emission probability is zero.
  if ( allowReject && wgt == 0. ) return 0;
  //if ( wgt == 0. ) return 0;

  // Done
  return 1;

}

//--------------------------------------------------------------------------

// Function to perform UNLOPS merging on this event.

int DireMerging::getStartingConditions( double RNpath, Event& process) {

  // Initialise which part of UNLOPS merging is applied.
  bool doUNLOPSSubt     = settingsPtr->flag("Merging:doUNLOPSSubt");
  bool doUNLOPSSubtNLO  = settingsPtr->flag("Merging:doUNLOPSSubtNLO");
  // Save number of looping steps
  mergingHooksPtr->nReclusterSave = settingsPtr->mode("Merging:nRecluster");
  int nRecluster        = settingsPtr->mode("Merging:nRecluster");

  // Calculate number of clustering steps
  int nSteps = mergingHooksPtr->getNumberOfClusteringSteps( myHistory->state,
                                                            true);
  int nRequested = mergingHooksPtr->nRequested();

  // Potentially recluster real emission jets for powheg input containing
  // "too many" jets, i.e. real-emission kinematics.
  bool containsRealKin = nSteps > nRequested && nSteps > 0;
  if ( containsRealKin ) nRecluster += nSteps - nRequested;

  // Event with production scales set for further (trial) showering
  // and starting conditions for the shower.
  int nPerformed = 0;
  if (!doUNLOPSSubt && !doUNLOPSSubtNLO && !containsRealKin )
    myHistory->getStartingConditions(RNpath, process);
  // Do reclustering (looping) steps.
  else myHistory->getFirstClusteredEventAboveTMS( RNpath, nRecluster, process,
    nPerformed, true );

  // Set QCD 2->2 starting scale different from arbitrary scale in LHEF!
  // --> Set to minimal mT of partons.
  int nFinal = 0;
  double muf = process[0].e();
  for ( int i=0; i < process.size(); ++i )
  if ( process[i].isFinal()
    && (process[i].colType() != 0 || process[i].id() == 22 ) ) {
    nFinal++;
    muf = min( muf, abs(process[i].mT()) );
  }
  // For pure QCD dijet events (only!), set the process scale to the
  // transverse momentum of the outgoing partons.
  if ( nSteps == 0 && nFinal == 2
    && ( mergingHooksPtr->getProcessString().compare("pp>jj") == 0
      || mergingHooksPtr->getProcessString().compare("pp>aj") == 0) )
    process.scale(muf);

  // Reset hard process candidates (changed after clustering a parton).
  mergingHooksPtr->storeHardProcessCandidates( process );

  // Check if resonance structure has been changed
  //  (e.g. because of clustering W/Z/gluino)
  vector <int> oldResonance;
  for ( int i=0; i < myHistory->state.size(); ++i )
    if ( myHistory->state[i].status() == 22 )
      oldResonance.push_back(myHistory->state[i].id());
  vector <int> newResonance;
  for ( int i=0; i < process.size(); ++i )
    if ( process[i].status() == 22 )
      newResonance.push_back(process[i].id());
  // Compare old and new resonances
  for ( int i=0; i < int(oldResonance.size()); ++i )
    for ( int j=0; j < int(newResonance.size()); ++j )
      if ( newResonance[j] == oldResonance[i] ) {
        oldResonance[i] = 99;
        break;
      }
  bool hasNewResonances = (newResonance.size() != oldResonance.size());
  for ( int i=0; i < int(oldResonance.size()); ++i )
    hasNewResonances = (oldResonance[i] != 99);

  // If necessary, reattach resonance decay products.
  if (!hasNewResonances) mergingHooksPtr->reattachResonanceDecays(process);

  // Allow merging hooks to remove emissions from now on.
  mergingHooksPtr->doIgnoreEmissions(false);

  // If the resonance structure of the process has changed due to reclustering,
  // redo the resonance decays in Pythia::next()
  if (hasNewResonances) return 2;

  // Done
  return 1;

}

//--------------------------------------------------------------------------

// Function to apply the merging scale cut on an input event.

bool DireMerging::cutOnProcess( Event& process) {

  // Save number of looping steps
  mergingHooksPtr->nReclusterSave = settingsPtr->mode("Merging:nRecluster");

  // For now, prefer construction of ordered histories.
  mergingHooksPtr->orderHistories(true);
  // For pp > h, allow cut on state, so that underlying processes
  // can be clustered to gg > h
  if ( mergingHooksPtr->getProcessString().compare("pp>h") == 0)
    mergingHooksPtr->allowCutOnRecState(true);

  // Reset any incoming spins for W+-.
  if (mergingHooksPtr->doWeakClustering())
    for (int i = 0;i < process.size();++i)
      process[i].pol(9);

  // Prepare process record for merging. If Pythia has already decayed
  // resonances used to define the hard process, remove resonance decay
  // products.
  Event newProcess( mergingHooksPtr->bareEvent( process, true) );
  // Store candidates for the splitting V -> qqbar'
  mergingHooksPtr->storeHardProcessCandidates( newProcess );

  // Check if event passes the merging scale cut.
  double tmsval  = mergingHooksPtr->tms();
  // Get merging scale in current event.
  double tmsnow  = mergingHooksPtr->tmsNow( newProcess );
  // Calculate number of clustering steps
  int nSteps = mergingHooksPtr->getNumberOfClusteringSteps( newProcess, true);

  // Too few steps can be possible if a chain of resonance decays has been
  // removed. In this case, reject this event, since it will be handled in
  // lower-multiplicity samples.
  int nRequested = mergingHooksPtr->nRequested();
  if (nSteps < nRequested) return true;

  // Reset the minimal tms value, if necessary.
  tmsNowMin = (nSteps == 0) ? 0. : min(tmsNowMin, tmsnow);

  // Potentially recluster real emission jets for powheg input containing
  // "too many" jets, i.e. real-emission kinematics.
  bool containsRealKin = nSteps > nRequested && nSteps > 0;

  // Get random number to choose a path.
  double RN = rndmPtr->flat();
  // Set dummy process scale.
  newProcess.scale(0.0);
  // Generate all histories
  DireHistory FullHistory( nSteps, 0.0, newProcess, DireClustering(),
            mergingHooksPtr,
            (*beamAPtr), (*beamBPtr), particleDataPtr, infoPtr,
            trialPartonLevelPtr, fsr, isr, psweights, coupSMPtr, true, true,
            1.0, 1.0, 1.0, 1.0, 0);
  // Project histories onto desired branches, e.g. only ordered paths.
  FullHistory.projectOntoDesiredHistories();

  // Remove real emission events without underlying Born configuration from
  // the loop sample, since such states will be taken care of by tree-level
  // samples.
  if ( containsRealKin && !allowIncompleteReal
    && FullHistory.select(RN)->nClusterings() == 0 )
    return true;

  // Cut if no history passes the cut on the lowest-multiplicity state.
  double dampWeight = mergingHooksPtr->dampenIfFailCuts(
           FullHistory.lowestMultProc(RN) );
  if ( dampWeight == 0. ) return true;

  // Do not apply cut if the configuration could not be projected onto an
  // underlying born configuration.
  if ( nSteps > 0 && FullHistory.select(RN)->nClusterings() == 0 )
    return false;

  // Now enfore merging scale cut if the event did not pass the merging scale
  // criterion.
  if ( nSteps > 0 && nSteps == nRequested && tmsnow < tmsval && tmsval > 0.) {
    string message="Warning in DireMerging::cutOnProcess: Les Houches Event";
    message+=" fails merging scale cut. Reject event.";
    infoPtr->errorMsg(message);
    return true;
  }

  // Check if more steps should be taken.
  int nFinalP = 0;
  int nFinalW = 0;
  Event coreProcess = Event();
  coreProcess.clear();
  coreProcess.init( "(hard process-modified)", particleDataPtr );
  coreProcess.clear();
  coreProcess = FullHistory.lowestMultProc(RN);
  for ( int i = 0; i < coreProcess.size(); ++i )
    if ( coreProcess[i].isFinal() ) {
      if ( coreProcess[i].colType() != 0 )
        nFinalP++;
      if ( coreProcess[i].idAbs() == 24 )
        nFinalW++;
    }

  bool complete = (FullHistory.select(RN)->nClusterings() == nSteps) ||
    ( mergingHooksPtr->doWeakClustering() && nFinalP == 2 && nFinalW == 0 );

  if ( !complete ) {
    string message="Warning in DireMerging::cutOnProcess: No clusterings";
    message+=" found. History incomplete.";
    infoPtr->errorMsg(message);
  }

  // Done if no real-emission jets are present.
  if ( !containsRealKin ) return false;

  // Now cut on events that contain an additional real-emission jet.
  // Perform one reclustering for real emission kinematics, then apply merging
  // scale cut on underlying Born kinematics.
  Event dummy = Event();
  // Initialise temporary output of reclustering.
  dummy.clear();
  dummy.init( "(hard process-modified)", particleDataPtr );
  dummy.clear();
  // Recluster once.
  FullHistory.getClusteredEvent( RN, nSteps, dummy );
  double tnowNew  = mergingHooksPtr->tmsNow( dummy );
  // Veto if underlying Born kinematics do not pass merging scale cut.
  if ( nRequested > 0 && tnowNew < tmsval && tmsval > 0.) {
    string message="Warning in DireMerging::cutOnProcess: Les Houches Event";
    message+=" fails merging scale cut. Reject event.";
    infoPtr->errorMsg(message);
    return true;
  }

  // Done if only interested in cross section estimate after cuts.
  return false;

}

//==========================================================================

} // end namespace Pythia8
