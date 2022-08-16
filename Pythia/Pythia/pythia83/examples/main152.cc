// main152.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Torbjorn Sjostrand <torbjorn.sjostrand@thep.lu.se>.

// Keywords: rescattering; low energy; pT spectra;

// Compare multiplicities and hadron spectra with and without rescattering,
// the former with or without rescattering between nearest string neighbours.
// Note: many other parameters could be used to vary the rescattering rate.

#include "Pythia8/Pythia.h"
using namespace Pythia8;

//--------------------------------------------------------------------------

int main() {

  // Main settings. All particles with tau0 > maxTau0 fm are set stable.
  int nEvent = 5000;
  double maxTau0 = 100.;

  // Histograms.
  Hist multFin0("n_final, no rescattering", 100, 0., 800.);
  Hist multRes0("n_resc, no rescattering ", 100, 0., 300.);
  Hist pTpi0("pT pions no rescattering",    100, 0., 5.);
  Hist pTk0("pT kaons no rescattering",     100, 0., 5.);
  Hist pTp0("pT protons no rescattering",   100, 0., 5.);
  Hist multFin1("n_final, no neighbours ",  100, 0., 800.);
  Hist multRes1("n_resc, no neighbours  ",  100, 0., 300.);
  Hist pTpi1("pT pions, no neighbours ",    100, 0., 5.);
  Hist pTk1("pT kaons, no neighbours ",     100, 0., 5.);
  Hist pTp1("pT protons, no neighbours ",   100, 0., 5.);
  Hist multFin2("n_final, all rescatter",   100, 0., 800.);
  Hist multRes2("n_resc, all rescatter ",   100, 0., 300.);
  Hist pTpi2("pT pions, all rescatter ",    100, 0., 5.);
  Hist pTk2("pT kaons, all rescatter ",     100, 0., 5.);
  Hist pTp2("pT protons, all rescatter ",   100, 0., 5.);

  // Loop over the three possible scenarios being compared.
  for (int ic = 0; ic < 3; ++ic) {

    // Create Pythia instance. Shorthand for event.
    Pythia pythia;
    Event& event = pythia.event;

    // Process selection: nondiffractive pp.
    pythia.readString("SoftQCD:nonDiffractive = on");
    pythia.readString("Beams:eCM = 13000.");

    // Turn rescattering on, which also requires vertices on.
    if (ic > 0) {
      pythia.readString("HadronLevel:Rescatter = on");
      pythia.readString("Fragmentation:setVertices = on");
      pythia.readString("PartonVertex:setVertex = on");

      // Rescattering details.
      //pythia.readString("Rescattering:quickCheck = off");
      if (ic == 1) pythia.readString("Rescattering:nearestNeighbours = off");
    }

    // Switch off all decays with lifetime > maxTau0.
    pythia.readString("ParticleDecays:limitTau0 = on");
    pythia.settings.parm("ParticleDecays:tau0Max", maxTau0 * FM2MM);

    // Initialize Pythia.
    if (!pythia.init()) {
      cout << " Pythia failed to initialize." << endl;
      return -1;
    }

    // References to relevant histograms.
    Hist& multFin = (ic == 0) ? multFin0 : ((ic == 1) ? multFin1 : multFin2);
    Hist& multRes = (ic == 0) ? multRes0 : ((ic == 1) ? multRes1 : multRes2);
    Hist& pTpi    = (ic == 0) ? pTpi0    : ((ic == 1) ? pTpi1    : pTpi2);
    Hist& pTk     = (ic == 0) ? pTk0     : ((ic == 1) ? pTk1     : pTk2);
    Hist& pTp     = (ic == 0) ? pTp0     : ((ic == 1) ? pTp1     : pTp2);

    // Generate events.
    int nSuccess = 0;
    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
      if (!pythia.next()) continue;
      ++nSuccess;

        // Final multiplicity and number of rescatterings.
      int nFin = event.nFinal(false);
      int nRes = 0;
      for (Particle& particle : event) {
        if (particle.isHadron() && !particle.isFinal()) {
          Particle& daughter = event[particle.daughter1()];
          if (daughter.isHadron() && daughter.statusAbs()/10 == 15) nRes += 1;
        }
      }
      // Exactly two hadrons participate in each rescattering. Fill histograms.
      nRes /= 2;
      multFin.fill(nFin + 0.5);
      multRes.fill(nRes + 0.5);

      // Hadron pT spectra at central rapidities.
      for (Particle& particle : event) {
        if (particle.isHadron() && particle.isFinal()
        && abs(particle.y()) < 2.) {
          int idAbs = particle.idAbs();
          double pT = particle.pT();
          if (idAbs == 211 || idAbs == 111) pTpi.fill(pT);
          else if (idAbs == 311 || idAbs == 321 || idAbs == 310
            || idAbs ==  130) pTk.fill(pT);
          else if (idAbs == 2212) pTp.fill(pT);
        }
      }

    // End of event loop. Normalize and print histograms.
    }
    pythia.stat();
    multFin *= 1. / (8. * nSuccess);
    multRes *= 1. / (3. * nSuccess);
    pTpi    *= 20. / nSuccess;
    pTk     *= 20. / nSuccess;
    pTp     *= 20. / nSuccess;
    cout << multFin << multRes << pTpi << pTk << pTp;

  // End loop over three cases.
  }

  // Plot histograms.
  HistPlot hpl("main152plot");
  hpl.frame("out152plot", "Hadronic multiplicity", "n", "dN/dn");
  hpl.add( multFin0, "", "no rescattering");
  hpl.add( multFin1, "", "partial rescattering");
  hpl.add( multFin2, "", "full rescattering");
  hpl.plot(true);
  hpl.frame("", "Number of rescatterings", "n", "dN/dn");
  //hpl.add( multRes0, "", "no rescattering");
  hpl.add( multRes1, ",g", "partial rescattering");
  hpl.add( multRes2, ",r", "full rescattering");
  hpl.plot(true);
  hpl.frame("", "Pion pT distribution for |y| < 2", "pT (GeV)", "dN/dpT");
  hpl.add( pTpi0, "", "no rescattering");
  hpl.add( pTpi1, "", "partial rescattering");
  hpl.add( pTpi2, "", "full rescattering");
  hpl.plot(true);
  hpl.frame("", "Kaon pT distribution for |y| < 2", "pT (GeV)", "dN/dpT");
  hpl.add( pTk0, "", "no rescattering");
  hpl.add( pTk1, "", "partial rescattering");
  hpl.add( pTk2, "", "full rescattering");
  hpl.plot(true);
  hpl.frame("", "Proton pT distribution for |y| < 2", "pT (GeV)", "dN/dpT");
  hpl.add( pTp0, "", "no rescattering");
  hpl.add( pTp1, "", "partial rescattering");
  hpl.add( pTp2, "", "full rescattering");
  hpl.plot(true);

  // Done.
  return 0;
}
