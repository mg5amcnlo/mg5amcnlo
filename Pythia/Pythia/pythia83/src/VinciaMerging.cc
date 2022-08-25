// VinciaMerging.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Helen Brooks, Christian T Preuss.

#include "Pythia8/VinciaMerging.h"
#include "Pythia8/VinciaHistory.h"
#include <ctime>

namespace Pythia8 {

using namespace VinciaConstants;

//==========================================================================

// The VinciaMerging class.

//--------------------------------------------------------------------------

// Initialize the merging.

void VinciaMerging::init() {

  // Verbosity.
  verbose = mode("Vincia:verbose");

  // Are we doing merging?
  bool vinciaOn     = (mode("PartonShowers:model")==2);
  bool sectorShower = flag("Vincia:sectorShower");
  doMerging         = flag("Merging:doMerging");
  doMerging         = ( doMerging && vinciaOn );
  doSectorMerging   = ( doMerging && sectorShower );

  // Check consistency.
  if (doMerging && !doSectorMerging && verbose >= NORMAL) {
    string msg = "Please set Vincia:sectorShower = on ";
    msg += "to perform merging with Vincia.";
    printOut(__METHOD_NAME__,msg);
  }

  // Flag to check if merging weight should directly be included
  // in the cross section.
  includeWtInXsec = flag("Merging:includeWeightInXsection");

  // Check if we are just estimating the cross section.
  doXSecEstimate = flag("Merging:doXSectionEstimate");

  // Check if we are doing merging in resonance systems.
  doMergeRes = flag("Vincia:MergeInResSystems");

  // Check if we need to insert resonances.
  doInsertRes = settingsPtr->flag("Vincia:InsertResInMerging");

  // What is the maximum multiplicity of the ME-generator?
  nMaxJets = mode("Merging:nJetMax");
  nMaxJetsRes = 0;
  nMergeResSys = 0;
  if (doMergeRes) {
    nMaxJetsRes = mode("Vincia:MergeNJetMaxRes");
    nMergeResSys = mode("Vincia:MergeNResSys");
  }
  nMaxJets += nMaxJetsRes*nMergeResSys;

  // Initialise counters.
  nVeto=0;
  nBelowMS=0;
  nTotal=0;
  nAbort=0;
  nVetoByMult=vector<int>(nMaxJets+1,0);
  nTotalByMult=vector<int>(nMaxJets+1,0);

}

//--------------------------------------------------------------------------

// Print some stats.

void VinciaMerging::statistics() {

  if (doMerging && verbose >= NORMAL) {
    int nVetoInMain = mergingHooksPtr->getNumberVetoedInMainShower();
    int nc1 = 0, nc2 = 0;
    cout << endl;
    cout << " *--------  VINCIA Merging Statistics  -----------------------"
         << "-----------------------------------------------------* \n";
    cout << " |                                                       "
         << "                                                          | \n";
    nc1 = int((std::to_string(nBelowMS)).size());
    nc2 = int((std::to_string(nTotal)).size());
    cout << " | Failed merging scale cut " << nBelowMS << " / " << nTotal
         << " events";
    for (int ws(0); ws<77-nc1-nc2; ++ws) cout << " ";
    cout << "|" << endl;
    cout << " |                                                       "
         << "                                                          | \n";
    nc1 = int((std::to_string(nVeto+nVetoInMain)).size());
    cout << " | Vetoed in total          " << nVeto + nVetoInMain
         << " / " << nTotal << " events";
    for (int ws(0); ws<77-nc1-nc2; ++ws) cout << " ";
    cout << "|" << endl;
    nc1 = int((std::to_string(nVeto)).size());
    cout << " |        in trial shower   " << nVeto << " / " << nTotal
         << " events";
    for (int ws(0); ws<77-nc1-nc2; ++ws) cout << " ";
    cout << "|" << endl;
    nc1 = int((std::to_string(nVetoInMain)).size());
    cout << " |        in main shower    " << nVetoInMain << " / " << nTotal
         << " events";
    for (int ws(0); ws<77-nc1-nc2; ++ws) cout << " ";
    cout << "|" << endl;
    cout << " |                                                       "
         << "                                                          | \n";
    cout << " | Vetoed in trial shower by multiplicity:               "
         << "                                                          | \n";
    for (int i=0; i<=nMaxJets; ++i) {
      nc1 = int((std::to_string(nVetoByMult[i])).size());
      nc2 = int((std::to_string(nTotalByMult[i])).size());
      cout <<" |   Born + "<< i<< " jets: "
           <<" vetoed " << nVetoByMult[i]
           << " / " << nTotalByMult[i];
      for (int ws(0); ws<84-nc1-nc2; ++ws) cout << " ";
      cout << "|" << endl;
    }
    cout << " |                                                       "
         << "                                                          | \n";
    string nAbortStr = std::to_string(nAbort);
    cout << " | Aborted " << nAbortStr << " events ";
    for (int ws(0); ws<96-int(nAbortStr.size()); ++ws) cout << " ";
    cout << "|" << endl;
    cout << " |                                                       "
         << "                                                          | \n";
    // For REPORT, print computing times.
    if (verbose >= REPORT) {
      cout << " | CPU time to construct histories:                      "
           << "                                                          | \n";
      auto it = historyCompTime.begin();
      for ( ; it != historyCompTime.end(); ++it) {
        int nClus = it->first;
        double cTime = it->second / (nHistories[nClus] / 1.e3) / 1.e3;
        string cpuTime = std::to_string(cTime);
        nc1 = (int)cpuTime.size();
        cout << " |   Born + "<< nClus << " jets: "
             << cpuTime << " seconds / 1k histories";
        for (int ws(0); ws<69-nc1; ++ws) cout << " ";
        cout << "   |" << endl;
      }
      cout << " |                                                       "
           << "                                                          | \n";
    }
    cout << " *---------------------------------------------------------------"
         <<"--------------------------------------------------*";
    cout << endl;
  }

}

//--------------------------------------------------------------------------

// Merge the event.

int VinciaMerging::mergeProcess(Event& process) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  int vetoCode = 1;

  // If we just want to calculate the cross section,
  // check cut here and exit.
  if (doXSecEstimate) {
    shared_ptr<VinciaMergingHooks> vinMergingHooksPtr
      = dynamic_pointer_cast<VinciaMergingHooks>(mergingHooksPtr);
    // Check whether we have a pointer to Vincia's own MergingHooks object now.
    if (!vinMergingHooksPtr) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Could not fetch Vincia's MergingHooks pointer.");
      vetoCode = -1;
    } else {
      // Check whether event is above merging scale.
      if (!vinMergingHooksPtr->isAboveMS(process))
        vetoCode = 0;
    }
  // Sector shower merging.
  } else if (doSectorMerging) {
    vetoCode = mergeProcessSector(process);
  }
  // Could add other types of merging here in future?
  // E.g. merging for regular shower.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return vetoCode;

}


//--------------------------------------------------------------------------

// Basically a simpler version of CKKW-L merging for sector showers.

int VinciaMerging::mergeProcessSector(Event& process) {

  bool doVeto = false;
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "begin", dashLen);
    string msg = "Raw process:";
    printOut(__METHOD_NAME__,msg);
    process.list();
  }

  // Reset weight of the event.
  int nWts = mergingHooksPtr->nWgts;
  vector<double> wts(nWts, 1.);
  if (!includeWtInXsec) mergingHooksPtr->setWeightCKKWL(wts);

  // Copy event record.
  Event newProcess = process;

  // Insert resonances in event record if needed.
  if (doInsertRes) {
    if (!insertResonances(newProcess)) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Could not insert resonances in event record.");
      return -1;
    }
  }

  // If not merging in resonance systems, remove their decays.
  if (!doMergeRes) {
    newProcess = mergingHooksPtr->bareEvent(newProcess, true);
    if (verbose >= DEBUG) {
      string msg = "Process with resonances stripped:";
      printOut(__METHOD_NAME__,msg);
      newProcess.list();
    }
  }

  // Find the (best) history - deterministic!
  auto start = std::clock();
  VinciaHistory history(newProcess, beamAPtr, beamBPtr,
    mergingHooksPtr, trialPartonLevelPtr, particleDataPtr, infoPtr);
  auto stop = std::clock();

  //TODO implement accept for unordered histories for MOPS-like merging.

  // Check if the event is below merging scale.
  if (history.isBelowMS()) {
    ++nBelowMS;
    ++nTotal;
    // Save the weight of the event for histogramming.
    if (!includeWtInXsec)
      mergingHooksPtr->setWeightCKKWL(vector<double>(nWts, 0.));
    else infoPtr->weightContainerPtr->setWeightNominal(0.);
    return 0;
  }

  if (!history.isValid()) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": No valid history found.");
    ++nAbort;
    return -1;
  }

  // Get number of clustering steps and save.
  int nClus = history.getNClusterSteps();
  if (nClus > nMaxJets) {
    // Something went wrong.
    if (verbose >= NORMAL) {
      string msg = ": Multiplicity exceeded expected maximum.";
      msg += " Please check.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    }
    return -1;
  }
  nTotalByMult[nClus]++;
  ++nTotal;

  // Statistics for computation time and size.
  double compTime = 1.e3*(stop - start)/CLOCKS_PER_SEC;
  historyCompTime[nClus] += compTime;
  nHistories[nClus]++;

  // Get CKKW-L weight.
  wts[0] = history.getWeightCKKWL();
  // Check that the weight is non-vanishing.
  if (wts[0] <= MICRO) wts[0] = 0.;
  if (verbose>=DEBUG) {
    stringstream ss;
    ss << "CKKW-L weight is " << wts[0];
    printOut(__METHOD_NAME__,ss.str());
  }
  // For now no variations implemented.
  for (int iWt(1); iWt<nWts; ++iWt) wts[iWt] = wts[0];

  // Save the weight of the event for histogramming.
  if (!includeWtInXsec) mergingHooksPtr->setWeightCKKWL(wts);
  else {
    // In this case, central merging weight goes into nominal weight, all
    // variations are saved relative to central merging weight
    vector<double> relWts(1, 1.);
    for (int iVar(1); iVar<nWts; ++iVar) {
      double wtVar = (wts[0] != 0) ? wts[iVar]/wts[0] :
        numeric_limits<double>::infinity();
      relWts.push_back(wtVar);
    }
    infoPtr->weightContainerPtr->setWeightNominal(infoPtr->weight()*wts[0]);
    mergingHooksPtr->setWeightCKKWL(relWts);
  }

  // If no-emission probability is zero, veto.
  if (wts[0] == 0.) {
    doVeto = true;
    // Check for abort.
    if (history.doAbort()) {
      nAbort++;
      nTotalByMult[nClus]--;
      nTotal--;
      if (verbose >= REPORT) printOut(__METHOD_NAME__,"Aborting merging");
      return -1;
    }
    nVetoByMult[nClus]++;
    ++nVeto;
  }
  else if (history.hasNewProcess()) {

    // We need to overwrite the hard process.
    // (e.g. because an MPI was generated).
    process = history.getNewProcess();

    // TODO remove any resonance that was inserted.

    // Now need to add back any resonances we removed.
    // TODO check this works
    mergingHooksPtr->reattachResonanceDecays(process);
  }

  // Set the scale at which to restart the shower.
  if (!doVeto) {
    process.scale(history.getRestartScale());
    // Tell MergingHooks whether we should veto the first emission.
    bool vetoFirst = (nClus < nMaxJets) ? true : false;
    mergingHooksPtr->doIgnoreStep(!vetoFirst);
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Shower restart scale: "
        +num2str(process.scale())+", can veto first step: "
        +(mergingHooksPtr->canVetoStep() ? " yes" : "  no"));
  }

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return (doVeto) ? 0 : 1;

}

//--------------------------------------------------------------------------

// Insert Resonances in event record if needed.

bool VinciaMerging::insertResonances(Event& process) {

  // Get lists of all leptons that are attached to the beams.
  vector<int> leptonsPos;
  vector<int> leptonsNeg;
  vector<int> leptonsNeutral;
  for (int iPtcl(5); iPtcl<process.size(); ++iPtcl) {
    Particle* ptcl = &process[iPtcl];
    // Check if particle is a final-state lepton.
    if (!ptcl->isFinal() || !ptcl->isLepton()) continue;
    // Check if it has beams as parents.
    if (ptcl->mother1() == 3 || ptcl->mother1() == 4
      || ptcl->mother2() == 3 || ptcl->mother2() == 4) {
      if (ptcl->isNeutral()) leptonsNeutral.push_back(iPtcl);
      else if (ptcl->charge() > 0.) leptonsPos.push_back(iPtcl);
      else leptonsNeg.push_back(iPtcl);
    }
  }
  const int nLepPos = leptonsPos.size();
  const int nLepNeg = leptonsNeg.size();
  const int nLepNeutral = leptonsNeutral.size();

  // No "free" leptons - all good!
  if (nLepPos == 0 && nLepNeg == 0 && nLepNeutral == 0) return true;
  // Currently, we can deal with only a single lepton pair.
  else if (nLepPos + nLepNeg + nLepNeutral > 2) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Cannot insert resonances for more than one lepton pair.");
    return false;
  } else if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Found " << nLepNeutral << " neutral, " << nLepPos << " positive, "
       << nLepNeg << " negative final state leptons.";
    printOut(__METHOD_NAME__, ss.str());
  }

  // Otherwise we have to insert them explicitly.

  // We need specific Vincia merging functions.
  shared_ptr<VinciaMergingHooks> vinMergingHooksPtr
    = dynamic_pointer_cast<VinciaMergingHooks>(mergingHooksPtr);
  if (!vinMergingHooksPtr) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Could not fetch VinciaMergingHooks pointer.");
    return false;
  }

  // Reconstruct resonance in hard process.
  // Note: only for one lepton pair!
  int iLep1 = -1;
  int iLep2 = -1;
  Particle resonance;
  if (nLepPos == 1 && nLepNeg == 1 && nLepNeutral == 0) {
    // Check which neutral resonance is specified in the hard process.
    int nResNeutral = vinMergingHooksPtr->getNResNeutralUndecayed();
    if (nResNeutral != 1) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Could not match event record to hard process.");
      return false;
    }

    // Construct resonance.
    iLep1 = leptonsPos.at(0);
    iLep2 = leptonsNeg.at(0);
    int idRes = (vinMergingHooksPtr->getResNeutralUndecayed()).at(0);
    Vec4 pRes = process[iLep1].p() + process[iLep2].p();
    double mRes = pRes.mCalc();
    resonance = Particle(idRes, -22, 3, 4, iLep1, iLep2,
      0, 0, pRes, mRes);
  } else if (nLepPos == 1 && nLepNeg == 0 && nLepNeutral == 1) {
    // Check which positive resonance is specified in the hard process.
    int nResPos = vinMergingHooksPtr->getNResPlusUndecayed();
    if (nResPos != 1) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Could not match event record to hard process.");
      return false;
    }

    // Construct resonance.
    iLep1 = leptonsPos.at(0);
    iLep2 = leptonsNeutral.at(0);
    int idRes = (vinMergingHooksPtr->getResPlusUndecayed()).at(0);
    Vec4 pRes = process[iLep1].p() + process[iLep2].p();
    double mRes = pRes.mCalc();
    resonance = Particle(idRes, -22, 3, 4, iLep1, iLep2,
      0, 0, pRes, mRes, mRes);
  } else if (nLepPos == 0 && nLepNeg == 1 && nLepNeutral == 1) {
    // Check which positive resonance is specified in the hard process.
    int nResNeg = vinMergingHooksPtr->getNResMinusUndecayed();
    if (nResNeg != 1) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Could not match event record to hard process.");
      return false;
    }

    // Construct resonance.
    iLep1 = leptonsNeg.at(0);
    iLep2 = leptonsNeutral.at(0);
    int idRes = (vinMergingHooksPtr->getResMinusUndecayed()).at(0);
    Vec4 pRes = process[iLep1].p() + process[iLep2].p();
    double mRes = pRes.mCalc();
    resonance = Particle(idRes, -22, 3, 4, iLep1, iLep2,
      0, 0, pRes, mRes, mRes);
  } else {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Could not reconstruct resonance from final-state leptons.");
    return false;
  }

  // Construct new event record.
  if (iLep1 > 0 && iLep2 > 0) {
    Event newProcess = process;
    newProcess.reset();
    newProcess[0] = process[0];
    // Insert beams and beam particles.
    for (int i(1); i<5; ++i)
      newProcess.append(process[i]);
    // Insert new resonance.
    int iRes = newProcess.append(resonance);
    // Insert other particles.
    int iStop = iRes;
    for (int i(5); i<process.size(); ++i)
      if (i != iLep1 && i != iLep2) iStop = newProcess.append(process[i]);
    // Insert leptons.
    int iDtr1 = newProcess.append(process[iLep1]);
    int iDtr2 = newProcess.append(process[iLep2]);
    // Fix daughters.
    newProcess[3].daughters(iRes, iStop);
    newProcess[4].daughters(iRes, iStop);
    newProcess[iRes].daughters(iDtr1,iDtr2);
    // Fix mothers.
    newProcess[iDtr1].mothers(iRes,iRes);
    newProcess[iDtr2].mothers(iRes,iRes);
    // Set process to new process.
    process = newProcess;
  } else {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Could not find leptons in event record.");
    return false;
  }
  return true;

}

//==========================================================================

} // end namespace Pythia8
