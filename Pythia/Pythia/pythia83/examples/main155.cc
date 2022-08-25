// main155.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Marius Utheim <marius.utheim@thep.lu.se>.

// Keywords: rescattering; low energy; cross sections; resonances;

// Calculate and plot resonance cross sections for the specified process.

#include "Pythia8/Pythia.h"
using namespace Pythia8;

//--------------------------------------------------------------------------

int main(int argc, char* argv[]) {

  // Initialize Pythia.
  Pythia pythia;
  pythia.readFile("main155.cmnd");
  if (!pythia.init()) {
    cout << " Pythia failed to initialize." << endl;
    return -1;
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

  // Get possible resonances.
  vector<int> resonances = pythia.hadronWidths.possibleResonances(idA, idB);

  if (resonances.size() == 0) {
    cout << "No resonances for input particles " << idA << " " << idB << endl;
    return -1;
  }

  // Define plot.
  HistPlot plt("main155plot");
  plt.frame("out155plot", "", "$\\sqrt{s}$ [GeV]", "$\\sigma$ [mb]");

  // Plot all resonances.
  for (int res : resonances) {
    Hist sigRes = Hist::plotFunc( [&](double eCM) {
      return pythia.getSigmaPartial(idA, idB, eCM, res);
    }, pythia.particleData.name(res), nBin, eMin, eMax);

    plt.add(sigRes, "-");
  }

  // Add total and miscellaneous cross sections at the end.
  Hist sigTot = Hist::plotFunc(
    [&](double eCM) {
      // type == 0 is equivalent to getSigmaTotal.
      return pythia.getSigmaPartial(idA, idB, eCM, 0); },
    "Total", nBin, eMin, eMax);
  Hist sigTotRes = Hist::plotFunc(
    [&](double eCM) { return pythia.getSigmaPartial(idA, idB, eCM, 9); },
    "Sum of resonances", nBin, eMin, eMax);
  Hist sigOther = sigTot - sigTotRes;

  plt.add(sigTotRes, "-");
  plt.add(sigOther, "-", "Other");
  plt.add(sigTot, "k-");

  // Plot.
  plt.plot();

  // Done.
  return 0;
}
