// main154.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Marius Utheim <marius.utheim@thep.lu.se>.

// Keywords: rescattering; low energy; cross sections;

// Calculate all cross sections for the specified process.

#include "Pythia8/Pythia.h"
using namespace Pythia8;

//--------------------------------------------------------------------------

int main(int argc, char* argv[]) {

  // Initialize Pythia.
  Pythia pythia;
  pythia.readFile("main154.cmnd");
  if (!pythia.init()) {
    cout << " Pythia failed to initialize." << endl;
    return 1;
  }

  int idA = argc == 3 ? atoi(argv[1]) : pythia.mode("Main:spareMode1");
  int idB = argc == 3 ? atoi(argv[2]) : pythia.mode("Main:spareMode2");
  double eMin = pythia.parm("Main:spareParm1");
  double eMax = pythia.parm("Main:spareParm2");
  if (eMin < pythia.particleData.m0(idA) + pythia.particleData.m0(idB)) {
    eMin = pythia.particleData.m0(idA) + pythia.particleData.m0(idB);
    cout << "Warning, setting eMin to nominal mass sum of " << eMin << ".\n";
  }

  int nBin = 300;

  ParticleData& particleData = pythia.particleData;

  HistPlot plt("main154plot");
  plt.frame("out154plot", "Cross section for " + particleData.name(idA)
    + " + " + particleData.name(idB), "$\\sqrt{s}$ (GeV)", "$\\sigma$ (mb)");

  // Basic cross sections: non-diffractive, elastic and diffractive.
  Hist sigND = Hist::plotFunc(
    [&](double eCM) { return pythia.getLowEnergySigma(idA, idB, eCM, 1); },
    "Non-diffractive", nBin, eMin, eMax);
  plt.add(sigND, "-");

  Hist sigEla = Hist::plotFunc(
    [&](double eCM) { return pythia.getLowEnergySigma(idA, idB, eCM, 2); },
    "Elastic", nBin, eMin, eMax);
  plt.add(sigEla, "-");

  Hist sigSD = Hist::plotFunc(
    [&](double eCM) { return pythia.getLowEnergySigma(idA, idB, eCM, 3)
                           + pythia.getLowEnergySigma(idA, idB, eCM, 4); },
    "Single diffractive", nBin, eMin, eMax);
  plt.add(sigSD, "-");

  Hist sigDD = Hist::plotFunc(
    [&](double eCM) { return pythia.getLowEnergySigma(idA, idB, eCM, 5); },
    "Double diffractive", nBin, eMin, eMax);
  plt.add(sigDD, "-");

  // Add nucleon excitation cross section for NN.
  if ((abs(idA) == 2212 || abs(idA) == 2112)
   && (abs(idB) == 2212 || abs(idB) == 2112)
   && idA * idB > 0) {
    Hist sigEx = Hist::plotFunc(
      [&](double eCM) { return pythia.getLowEnergySigma(idA, idB, eCM, 7); },
      "Excitation", nBin, eMin, eMax);
    plt.add(sigEx, "-");
  }

  // Add annihilation cross section of baryon-antibaryon.
  if (idA * idB < 0 && particleData.isBaryon(idA)
                    && particleData.isBaryon(idB)) {
    Hist sigAnn = Hist::plotFunc(
      [&](double eCM) { return pythia.getLowEnergySigma(idA, idB, eCM, 8); },
      "Annihilation", nBin, eMin, eMax);
    plt.add(sigAnn, "-");
  }

  // Add resonance cross sections if applicable.
  if (pythia.hadronWidths.hasResonances(idA, idB)) {
    Hist sigRes = Hist::plotFunc(
      [&](double eCM) { return pythia.getLowEnergySigma(idA, idB, eCM, 9); },
      "Resonant", nBin, eMin, eMax);
    plt.add(sigRes, "-");
  }

  // Add total cross section at the end.
  Hist sigTot = Hist::plotFunc(
    [&](double eCM) { return pythia.getLowEnergySigma(idA, idB, eCM); },
    "Total", nBin, eMin, eMax);
  plt.add(sigTot, "k-");

  // Plot.
  plt.plot();

}
