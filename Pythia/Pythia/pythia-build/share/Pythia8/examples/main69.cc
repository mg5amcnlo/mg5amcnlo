// main69.cc is a part of the PYTHIA event generator.
// Copyright (C) 2017 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Main program to generate charged hadron spectra from photon-initiated
// hard processes, by combining four sub-runs with direct or resolved photons.
// The first sub-run generates hard QCD processes with two resolved photons,
// second and third processes where one beam is a unresolved and the other
// resolved, and the last run the contribution from processes where both
// photons are unresolved and act as initiators of the process. Events can be
// generated either with photon beams or with photons emitted from lepton
// beams, and a pT weight can be set to emphasize higher values of pT.

#include "Pythia8/Pythia.h"

using namespace Pythia8;

int main() {

  // Generator.
  Pythia pythia;

  // Decrease the output.
  pythia.readString("Init:showChangedSettings = off");
  pythia.readString("Init:showChangedParticleData = off");
  pythia.readString("Next:numberCount = 0");
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");

  // Shorthand for some public members of pythia (also static ones).
  Settings& settings = pythia.settings;
  Info& info = pythia.info;

  // Generate photon-photon events in leptonic or photon beams.
  bool photonsFromElectrons = false;
  bool biasSampling         = false;

  // Optionally use different PDFs for hard process.
  // pythia.readString("PDF:useHard = on");
  // pythia.readString("PDF:GammaHardSet = LHAPDF5:SASG.LHgrid/5");

  // Beam parameters.
  pythia.readString("Beams:eCM = 200.");

  // Either electron beams with photons or photon beams.
  if ( photonsFromElectrons) {
    pythia.readString("Beams:idA = -11");
    pythia.readString("Beams:idB =  11");
    pythia.readString("PDF:lepton2gamma = on");

    // Cuts on photon virtuality and invariant mass of gamma-gamma pair.
    pythia.readString("Photon:Q2max = 1.0");
    pythia.readString("Photon:Wmin  = 10.0");

  } else {
    pythia.readString("Beams:idA = 22");
    pythia.readString("Beams:idB = 22");
  }

  // Number of events per run.
  int nEvent = 10000;

  // Limit partonic pThat.
  settings.parm("PhaseSpace:pTHatMin", 5.0);

  // Hard processes with pT-weight.
  if ( biasSampling) {
    pythia.readString("PhaseSpace:bias2Selection = on");
    pythia.readString("PhaseSpace:bias2SelectionPow = 3.");
    pythia.readString("PhaseSpace:bias2SelectionRef = 5.");
  }

  // Reset statistics after each subrun.
  pythia.readString("Stat:reset = on");

  // Parameters for histograms.
  double pTmin = 0.0;
  double pTmax = 40.0;
  int nBinsPT  = 40;

  // Initialize the histograms.
  Hist pTtot("Total charged hadron pT distribution", nBinsPT, pTmin, pTmax);
  Hist pThard("Hard QCD contribution from resolved", nBinsPT, pTmin, pTmax);
  Hist pTresdir("Resolved-direct contribution", nBinsPT, pTmin, pTmax);
  Hist pTdirres("Direct-resolved contribution", nBinsPT, pTmin, pTmax);
  Hist pTdirdir("Direct-direct contribution", nBinsPT, pTmin, pTmax);
  Hist pTiRun("Contribution from Run i", nBinsPT, pTmin, pTmax);

  // Loop over relevant processes.
  for ( int iRun = 1; iRun < 5; ++iRun) {

    // Set the type of gamma-gamma process:
    // 1 = resolved-resolved,
    // 2 = resolved-direct,
    // 3 = direct-resolved,
    // 4 = direct-direct.
    settings.mode("Photon:ProcessType", iRun);

    // First run: hard QCD processes for resolved photons.
    if ( iRun == 1 ) {
      pythia.readString("HardQCD:all = on");

    // Second and third run: direct-resolved processes.
    } else if ( (iRun == 2) || (iRun == 3) ) {
      pythia.readString("HardQCD:all = off");
      pythia.readString("PhotonParton:all = on");
      pythia.readString("PartonLevel:MPI = off");

    // Fourth run: direct-direct QCD processes.
    } else {
      pythia.readString("PhotonParton:all = off");
      pythia.readString("PhotonCollision:gmgm2qqbar = on");
      pythia.readString("PhotonCollision:gmgm2ccbar = on");
      pythia.readString("PhotonCollision:gmgm2bbbar = on");

      // No need for pT weight for direct-direct gm+gm collisions.
      if ( !photonsFromElectrons && biasSampling )
        pythia.readString("PhaseSpace:bias2Selection = off");
    }

    // Initialize the generator.
    pythia.init();

    // Clear the histogram.
    pTiRun.null();

    // Begin event loop. Skip if fails.
    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

      // Generate next event.
      if (!pythia.next()) continue;

      // List the first process and event for each contribution.
      if (iEvent == 0) {
        pythia.process.list();
        pythia.event.list();
      }

      // Possible event weights.
      double weight = info.weight();

      // Loop over event record and find charged final state particles.
      for (int i = 0; i < pythia.event.size(); ++i){
        if ( pythia.event[i].isFinal() && pythia.event[i].isCharged() ) {

          // Store pT and add pT to the histogram.
          double pTch = pythia.event[i].pT();
          pTiRun.fill(pTch, weight);
        }
      }

    }

    // Show statistics after each run (erorrs cumulate).
    pythia.stat();

    // Normalize to cross section [mb].
    double sigmaNorm = info.sigmaGen() / info.weightSum();
    double pTBin     = (pTmax - pTmin) / (1. * nBinsPT);
    pTiRun          *= sigmaNorm / pTBin;

    if (iRun == 1) pThard   = pTiRun;
    if (iRun == 2) pTresdir = pTiRun;
    if (iRun == 3) pTdirres = pTiRun;
    if (iRun == 4) pTdirdir = pTiRun;
    pTtot += pTiRun;

  // End of loop over runs.
  }

  // Print histograms.
  cout << pThard << pTresdir << pTdirres << pTdirdir << pTtot;

  // Done.
  return 0;
}
