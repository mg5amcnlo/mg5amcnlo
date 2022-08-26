// main183.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Marius Utheim <marius.utheim@thep.lu.se>.
//          Torbjorn Sjostrand <torbjorn.sjostrand@thep.lu.se>.

// Keywords: cosmic ray cascade; switch beam; switch collision energy;
// arXiv:2108.03481 [hep-ph]

#include "Pythia8/Pythia.h"
using namespace Pythia8;

//==========================================================================

// Note: when a hadron interacts with a medium particle, the latter is added
// to the event record. This program uses these additional status codes for
// target particles in the medium:
//  181: the first (or only) target nucleon in a collision.
//  182: nucleons from further subcollisions with the same target nucleus.

// Conversion factor between mm and km.
constexpr double KM2MM = 1.e6;

// Conversion factor for kg/m^3 to GeV/(c^2 mm mb)
constexpr double kg_m3_to_GeV_c2mm1mb1
  =  5.60958865e26 // kg to GeV/c2
    * 0.001         // m^-1 to mm^-1
    * 1e-31;        // m^-2 to mb^-1

// Medium parameters
constexpr double mAir = 0.9315;
constexpr double mediumDensity = 1.225 * kg_m3_to_GeV_c2mm1mb1 / mAir;
constexpr double rhoAir = 1.225e-4; // g cm^-2 mm^-1
constexpr double mediumHeight = 100 * KM2MM;
constexpr double H = 10.4 * KM2MM;

//==========================================================================

// Configuration of the atmospheric model.

struct Configuration {
  // Description of the configuration, to be used when plotting.
  string legend;
  // Whether the atmosphere should be exponentially attenuated, or uniform.
  bool doExponential;
  // Whether the medium particles are clustered into ions or a p/n gas.
  bool doHeavyIon;
  // The zenith angle of the primary particle.
  double zenithAngle;
  // Hadrons below the kinetic energy threshold cannot interact with medium.
  double eKinMin;

  // Get the height above ground of the primary particle when cascade begins.
  double initialHeight() {
    return doExponential ? mediumHeight : H;
  }

  // Get the medium depth of a particle at the specified height.
  double getDepth(double h) {
    return 1. / cos(zenithAngle)
        * ( doExponential ? rhoAir * H * exp(-h / H)
                          : rhoAir * (initialHeight() - h) );
  }
};

//==========================================================================

int main() {

  // Set up four different "atmospheres" to compare.
  constexpr int nCases = 4;
  vector<Configuration> configurations = {
    { "Uniform p/n atmosphere",                false, false, 0.,       0.2 },
    { "Uniform nitrogen",                      false, true,  0.,       0.2 },
    { "Exponential nitrogen",                  true,  true,  0.,       0.2 },
    { "Exponential nitrogen at 45$^{\\circ}$", true,  true,  M_PI / 4, 0.2 },
  };
  vector<string> col = { "r", "b", "k", "g" };

  // Energy of primary incident proton (in GeV).
  double pPri = 1e6;

  // Number of events per case. Only do a few since each shower is so big.
  int nEvent = 5;

  // Main Pythia object for managing the cascade evolution and particle decays.
  Pythia pythiaMain;
  Event& eventMain = pythiaMain.event;
  Rndm& rndm = pythiaMain.rndm;
  double mp = pythiaMain.particleData.m0(2212);
  // Prepare to do decays but no hard processes.
  pythiaMain.readString("ProcessLevel:all = off");
  pythiaMain.readString("211:mayDecay = on");
  pythiaMain.readString("13:mayDecay  = on");
  pythiaMain.readString("321:mayDecay = on");
  pythiaMain.readString("130:mayDecay = on");
  // Initialize.
  pythiaMain.init();

  // Secondary Pythia object for performing individual collisions.
  Pythia pythiaColl;
  Event& eventColl = pythiaColl.event;
  // Variable incoming beam type and energy.
  pythiaColl.readString("Beams:allowVariableEnergy = on");
  pythiaColl.readString("Beams:allowIDAswitch = on");
  // Set up for fixed-target collisions.
  pythiaColl.readString("Beams:frameType = 3");
  pythiaColl.settings.parm("Beams:pzA", -pPri);
  pythiaColl.readString("Beams:pzB = 0.");
  // Must use the soft and low-energy QCD processes.
  pythiaColl.readString("SoftQCD:all = on");
  pythiaColl.readString("LowEnergyQCD:all = on");
  // Decays to be done by pythiaMain.
  pythiaColl.readString("HadronLevel:Decay = off");
  // Reduce printout and relax energy-momentum conservation.
  pythiaColl.readString("Print:quiet = on");
  pythiaColl.readString("Check:epTolErr = 0.1");
  // Reuse MPI initialization file if it exists; else create a new one.
  pythiaColl.readString("MultipartonInteractions:reuseInit = 3");
  pythiaColl.readString("MultipartonInteractions:initFile = main183.mpi");
  // Initialize.
  pythiaColl.init();

  // Book histograms.
  double depthMax = 1.5 * rhoAir * H;
  Hist nInt[nCases], diffHad[nCases], diffMuon[nCases], prodnue[nCases],
       prodnumu[nCases];
  for (int iCase = 0; iCase < nCases; ++iCase) {
    nInt[iCase].book("depth of interactions",            100, 0., depthMax);
    diffHad[iCase].book("hadron production-decay depth", 100, 0., depthMax);
    diffMuon[iCase].book("muon production-decay depth",  100, 0., depthMax);
    prodnue[iCase].book("nu_e production depth",         100, 0., depthMax);
    prodnumu[iCase].book("nu_mu production depth",       100, 0., depthMax);
  }

  // Begin loops over cases and events.
  for (int iCase = 0; iCase < nCases; ++iCase)
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

    // Four-momentum of incoming initiator.
    Configuration& config = configurations[iCase];
    double pxPri = 0.;
    double pyPri = pPri * sin(config.zenithAngle);
    double pzPri = pPri * cos(config.zenithAngle);
    Vec4 p0(pxPri, pyPri, -pzPri, sqrt(pow2(pPri) + pow2(mp)));

    // Insert primary particle in cleared main event record.
    eventMain.clear();
    eventMain.append(90,  -11, 0, 0, 1, 1, 0, 0, p0, mp);
    eventMain.append(2212, 12, 0, 0, 0, 0, 0, 0, p0, mp);

    // Set initial location of initiator, where z is distance above ground.
    double heightNow = config.initialHeight();
    eventMain[0].yProd(-heightNow * tan(config.zenithAngle));
    eventMain[0].zProd(heightNow);
    eventMain[1].yProd(-heightNow * tan(config.zenithAngle));
    eventMain[1].zProd(heightNow);

    // Loop over particles in the main event record.
    for (int iHad = 1; iHad < eventMain.size(); ++iHad) {
      Particle& hadNow = eventMain[iHad];

      // Skip already fragmented/decayed or upwards-moving particles.
      if (!hadNow.isFinal() || hadNow.pz() > 0.) continue;

      // Find decay vertex for unstable hadrons. (Below ground if no decay.)
      Vec4 vDec = hadNow.canDecay() ? hadNow.vDec() : Vec4( 0., 0., -1., 0.);

      // Low energy hadrons should not interact with medium.
      // Decay non-hadrons or low-energy ones if decay happens above ground.
      if (!hadNow.isHadron() || hadNow.e() - hadNow.m() < config.eKinMin) {
        if (vDec.pz() > 0.) pythiaMain.moreDecays(iHad);
        continue;
      }

      // Invariant mass of particle with a p/n nucleon in atmosphere.
      int idNow     = hadNow.id();
      Vec4 pNow     = hadNow.p();
      double eCMNow = (pNow + Vec4(0, 0, 0, mp)).mCalc();

      // Get hadron-nucleon cross section.
      double sigmaNow = pythiaColl.getSigmaTotal(idNow, 2212, eCMNow);

      // If the cross section vanishes, decay is the only option.
      if (sigmaNow <= 0.) {
        if (vDec.pz() > 0.) pythiaMain.moreDecays(iHad);
        continue;
      }

      // Average number of hadron-nucleon collisions in a hadron-nitrogen one.
      double nCollAvg = (sigmaNow < 31.) ? 1. + 0.017 * sigmaNow
                      : 1.2 + 0.0105 * sigmaNow;
      double probMore = 1. - 1. / nCollAvg;

      // Medium density is expressed in terms of nucleons per volume, so if
      // collisions are clustered to nuclei then sigma must be compensated.
      if (config.doHeavyIon) sigmaNow /= nCollAvg;

      // Calculate potential interaction vertex, depending on medium.
      Vec4 vInt;
      Vec4 dirNow = pNow / pNow.pAbs();
      // Exponential atmosphere.
      if (config.doExponential) {
        double zNow  = hadNow.zProd();
        double dzds  = hadNow.pz() / hadNow.pAbs();
        double logR  = log(rndm.flat());
        double zNext = -H * log( exp(-zNow / H)
                     + dzds / (H * sigmaNow * mediumDensity) * logR );
        vInt = hadNow.vProd() + (zNext - zNow) * dirNow / dzds;
      // Homogeneous atmosphere.
      } else {
        double freePath = rndm.exp() / (mediumDensity * sigmaNow);
        vInt = hadNow.vProd() + freePath * dirNow;
      }

      // Done if hadron reaches surface before both interaction and decay.
      if (vInt.pz() < 0. && vDec.pz() < 0.) continue;

      // Do decay if that happens first.
      if (vDec.pz() > vInt.pz()) {
        pythiaMain.moreDecays(iHad);
        continue;
      }

      // Set up for collisions on a nucleus.
      int np = 7;
      int nn = 7;
      int sizeOld = 0;
      int sizeNew = 0;
      double probSD = 0.3;

      // Loop over varying number of hit nucleons in target nucleus.
      for (int iColl = 1; iColl < 10; ++iColl) {
        if (!config.doHeavyIon && iColl == 2) break;
        if (iColl > 1 && rndm.flat() > probMore) break;

        // Pick incoming projectile: trivial for first subcollision, else ...
        int iProj = iHad;
        int procType = 0;

        // ... find highest-pLongitudinal particle from latest subcollision.
        if (iColl > 1) {
          iProj = 0;
          double pMax = 0.;
          for (int i = sizeOld; i < sizeNew; ++i)
          if ( eventMain[i].isFinal() && eventMain[i].isHadron()) {
            double pp = dot3(dirNow, eventMain[i].p());
            if (pp > pMax) {
              iProj = i;
              pMax  = pp;
            }
          }

          // No further subcollision if no particle with enough energy.
          if ( iProj == 0 || eventMain[iProj].e() - eventMain[iProj].m()
            < config.eKinMin) break;

          // Choose process; only SD or ND at perturbative energies.
          double eCMSub = (eventMain[iProj].p() + Vec4(0, 0, 0, mp)).mCalc();
          if (eCMSub > 10.) procType = (rndm.flat() < probSD) ? 4 : 1;
        }

        // Pick one p or n from target.
        int idProj = eventMain[iProj].id();
        bool doProton = rndm.flat() < (np / double(np + nn));
        if (doProton) np -= 1;
        else          nn -= 1;
        int idNuc = (doProton) ? 2212 : 2112;

        // Perform the projectile-nucleon subcollision.
        pythiaColl.setBeamIDs(idProj, idNuc);
        pythiaColl.setKinematics(eventMain[iProj].p(), Vec4());
        pythiaColl.next(procType);

        // Insert target nucleon. Mothers are (0,iProj) to mark who it
        // interacted with. Always use proton mass for simplicity.
        int statusNuc = (iColl == 1) ? -181 : -182;
        int iNuc = eventMain.append( idNuc, statusNuc, 0, iProj, 0, 0, 0, 0,
          0., 0., 0., mp, mp);
        eventMain[iNuc].vProdAdd(vInt);

        // Insert secondary produced particles (but skip intermediate partons)
        // into main event record and shift to correct production vertex.
        sizeOld = eventMain.size();
        for (int iSub = 3; iSub < eventColl.size(); ++iSub) {
          if (!eventColl[iSub].isFinal()) continue;
          int iNew = eventMain.append(eventColl[iSub]);
          eventMain[iNew].mothers(iNuc, iProj);
          eventMain[iNew].vProdAdd(vInt);
        }
        sizeNew = eventMain.size();

        // Update daughters of colliding hadrons and other history.
        eventMain[iProj].daughters(sizeOld, sizeNew - 1);
        eventMain[iNuc].daughters(sizeOld, sizeNew - 1);
        eventMain[iProj].statusNeg();
        double dTau = (iColl == 1) ? (vInt.e() - eventMain[iHad].tProd())
          * eventMain[iHad].m() / eventMain[iHad].e() : 0.;
        eventMain[iProj].tau(dTau);

      // End of loop over interactions in a nucleus.
      }

    // End of loop over interactions + decays inside a single cascade.
    }

    // Begin analysis. Loop over all particles to find interaction depths.
    for (Particle& h : eventMain) {
      double depthProd = config.getDepth(h.zProd());
      double depthDec  = config.getDepth(h.isFinal() ? 0. : h.zDec());

      // If particle came from the medium, record the interaction depth.
      if (h.status() == -181 || h.status() == -182) {
        nInt[iCase].fill(depthProd);
      }
      // Otherwise, track depths where particles are created/destroyed.
      else if (h.e() - h.m() > config.eKinMin) {
        if (h.isHadron()) {
          if (h.isFinal())
            diffHad[iCase].fill(depthProd, 1.);
          else {
            diffHad[iCase].fill(min(depthProd, depthDec), 1.);
            diffHad[iCase].fill(max(depthProd, depthDec), -1.);
          }
        }
        else if (h.idAbs() == 13) {
          if (h.isFinal())
            diffMuon[iCase].fill(depthProd, 1.);
          else {
            diffMuon[iCase].fill(min(depthProd, depthDec), 1.);
            diffMuon[iCase].fill(max(depthProd, depthDec), -1.);
          }
        }
        else if (h.idAbs() == 12)
          prodnue[iCase].fill(depthProd, 1.);
        else if (h.idAbs() == 14)
          prodnumu[iCase].fill(depthProd, 1.);
      }
    }

  // End loops over events and cases.
  }

  // Print statistics, mainly for errors.
  pythiaMain.stat();
  pythiaColl.stat();

  // Book histograms.
  Hist nHad[nCases], nMuon[nCases], nnue[nCases], nnumu[nCases];
  for (int iCase = 0; iCase < nCases; ++iCase) {
    nHad[iCase] .book("", 100, 0., depthMax);
    nMuon[iCase].book("", 100, 0., depthMax);
    nnue[iCase] .book("", 100, 0., depthMax);
    nnumu[iCase].book("", 100, 0., depthMax);

    // Integrate production minus depletion to find particle number by depth.
    double nHadSum = 0., nMuonSum = 0., nnueSum = 0., nnumuSum = 0.;
    for (int i = 1; i <= 100; ++i) {
      double depthNow = depthMax * (i - 0.5) / 100.;
      if (depthNow > configurations[iCase].getDepth(0.)) break;
      nHadSum  += diffHad[iCase] .getBinContent(i);
      nMuonSum += diffMuon[iCase].getBinContent(i);
      nnueSum  += prodnue[iCase] .getBinContent(i);
      nnumuSum += prodnumu[iCase].getBinContent(i);
      nHad[iCase] .fill(depthNow, nHadSum );
      nMuon[iCase].fill(depthNow, nMuonSum);
      nnue[iCase] .fill(depthNow, nnueSum );
      nnumu[iCase].fill(depthNow, nnumuSum);
    }

    // Normalize histograms.
    nInt[iCase].normalizeSpectrum(nEvent);
    nHad[iCase]  /= nEvent;
    nMuon[iCase] /= nEvent;
    nnue[iCase]  /= nEvent;
    nnumu[iCase] /= nEvent;
  }

  // Normalize and plot histograms.
  HistPlot plt("main183plot");
  plt.frame("out183plot", "Atmospheric depth of p/n interactions",
    "$X$ (g/cm$^2$)", "$(1/n_{ev}) dn_{int}/dX$", 6.4, 4.8);
  for (int iCase = 0; iCase < nCases; ++iCase)
    plt.add(nInt[iCase], "-,"+col[iCase], configurations[iCase].legend);
  plt.plot(true);
  plt.frame("out183plot", "Number of hadrons at depth",
    "$X$ (g/cm$^2$)", "$(1/n_{ev}) \\int_0^{X} dn_{had}$", 6.4, 4.8);
  for (int iCase = 0; iCase < nCases; ++iCase)
    plt.add(nHad[iCase], "-,"+col[iCase], configurations[iCase].legend);
  plt.plot(true);
  plt.frame("out183plot", "Number of muons at depth",
    "$X$ (g/cm$^2$)", "$(1/n_{ev}) \\int_0^{X} dn_{\\mu}$", 6.4, 4.8);
  for (int iCase = 0; iCase < nCases; ++iCase)
    plt.add(nMuon[iCase], "-,"+col[iCase], configurations[iCase].legend);
  plt.plot(true);
  plt.frame("out183plot", "Number of e neutrinos at depth",
    "$X$ (g/cm$^2$)", "$(1/n_{ev}) \\int_0^{X} dn_{\\nu_e}$", 6.4, 4.8);
  for (int iCase = 0; iCase < nCases; ++iCase)
    plt.add(nnue[iCase], "-,"+col[iCase], configurations[iCase].legend);
  plt.plot(true);
  plt.frame("out183plot", "Number of mu neutrinos at depth",
    "$X$ (g/cm$^2$)", "$(1/n_{ev}) \\int_0^{X} dn_{\\nu_\\mu}$", 6.4, 4.8);
  for (int iCase = 0; iCase < nCases; ++iCase)
    plt.add(nnumu[iCase], "-,"+col[iCase], configurations[iCase].legend);
  plt.plot(true);

}
