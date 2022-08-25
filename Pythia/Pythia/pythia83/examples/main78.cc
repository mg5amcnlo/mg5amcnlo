// main78.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Ilkka Helenius <ilkka.m.helenius@jyu.fi>.

// Keywords: photon beam; UPC; photoproduction; exclusive;

// Main program to demonstrate how to generate different types of
// photon-initiated dilepton events in proton-proton collisions.

#include "Pythia8/Pythia.h"

using namespace Pythia8;

// The main program.

int main() {

  // Generator.
  Pythia pythia;

  // Decrease the output.
  pythia.readString("Init:showChangedSettings = off");
  pythia.readString("Init:showChangedParticleData = off");
  pythia.readString("Next:numberCount = 1000");
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 1");
  pythia.readString("Next:numberShowEvent = 1");

  // Beam parameters.
  pythia.readString("Beams:frameType = 1");
  pythia.readString("Beams:eCM = 13000");
  pythia.readString("Beams:idA = 2212");
  pythia.readString("Beams:idB = 2212");

  // Four possible contributions: double-dissociative = 1;
  // single-dissociative on side A (B) = 2 (3); elastic-elastic = 4.
  int processType = 4;

  // Enable elastic photon beams according to processType above.
  // For dissociative beams photons from PDFs (NNPDF2.3 by default) are used.
  if ( processType == 4 || processType == 3 )
    pythia.readString("PDF:beamA2gamma = on");
  if ( processType == 4 || processType == 2 )
    pythia.readString("PDF:beamB2gamma = on");

  // Need to use virtuality dependent photon flux to obtain realistic
  // acoplanarity distribution also for elastic-elastic case.
  pythia.readString("PDF:Proton2gammaSet = 2");

  // Set outgoing lepton-pair id and switch on relevant process.
  int idLep = 13;
  if      (idLep == 11) pythia.readString("PhotonCollision:gmgm2ee = on");
  else if (idLep == 13) pythia.readString("PhotonCollision:gmgm2mumu = on");

  // Need to prepare for MPIs only for double-dissociative production.
  if ( processType != 1 ) pythia.readString("PartonLevel:MPI = off");

  // Use dipole shower as appropriate for the case where no colour connection
  // between remnants.
  pythia.readString("SpaceShower:dipoleRecoil = on");
  pythia.readString("SpaceShower:pTmaxMatch = 2");
  pythia.readString("SpaceShower:pTdampMatch = 1");

  // Kinematical cuts.
  pythia.readString("PhaseSpace:mHatMin = 10.0");

  // Initialize the histogram for acoplanarity.
  Hist acoHist("Acoplanarity of the lepton pair", 30, 0.0, 0.06);

  // Initialize the generator.
  pythia.init();

  // Number of events.
  int nEvent = 10000;

  // Begin event loop. Skip if fails.
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

    // Generate next event.
    if (!pythia.next()) continue;

    // Event weight.
    double weight = pythia.info.weight();

    // Find the final muons (last ones in the event record).
    int iPosPlus = 0, iPosMinus = 0;
    for (int i = 0; i < pythia.event.size();++i) {
      if (pythia.event[i].id() == idLep) iPosPlus = i;
      if (pythia.event[i].id() == -idLep) iPosMinus = i;
    }

    // Derive 4-momenta of leptons.
    Vec4 p1 = pythia.event[iPosPlus].p();
    Vec4 p2 = pythia.event[iPosMinus].p();

    // Calculate acoplanarity (linear in Delta phi).
    double aco = 1. - abs( p1.phi() - p2.phi() ) / M_PI;

    // Fill histrogram with possible weights.
    acoHist.fill(aco, weight);

  } // End of event loop.

  // Show statistics.
  pythia.stat();

  // Normalize to cross section [fb].
  double sigmaNorm = pythia.info.sigmaGen() / pythia.info.weightSum() * 1.e12;
  acoHist *= sigmaNorm;

  // Print histogram.
  cout << acoHist;

  // Done.
  return 0;
}
