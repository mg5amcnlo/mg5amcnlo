// test202.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This test program is a basic check of Vincia showers for pp > tt at LHC.
// Most of the components of Vincia get called: ISR, FSR, resonance decays,
// and QED showers. (Helicity dependence not yet feasible to test with this
// program since Born unpolarised.)

// Author: Peter Skands <peter.skands@monash.edu>

#include <time.h>
#include "Pythia8/Pythia.h"
using namespace Pythia8;

int main() {

  //************************************************************************
  // Generator.
  Pythia pythia;

  // Define run settings.
  int nEvent = 20;
  pythia.readString("Next:numberShowInfo    = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent   = 0");
  pythia.readString("Main:timesAllowErrors = 0");
  pythia.readString("Beams:eCM = 7000.");
  pythia.readString("Top:qqbar2ttbar = on");
  pythia.readString("Top:gg2ttbar = on");
  pythia.readString("24:onMode = off");
  pythia.readString("24:onIfAny = 13");
  pythia.readString("Check:event = on");
  // Vincia settings.
  pythia.readString("PartonShowers:Model = 2");
  pythia.readString("Vincia:verbose = 0");
  
  //************************************************************************

  // Extract settings to be used in the main program.
  // Number of events, generated and listed ones.

  //************************************************************************

  // Initialize
  if(!pythia.init()) { return EXIT_FAILURE; }

  //************************************************************************

  // Measure the cpu runtime.
  clock_t start, stop;
  double t = 0.0;
  start = clock();

  // Averages to test. 
  double sumWeights = 0.;
  double nFinal(0.);
  // Test ISR and FSR (maily QCD)
  double n43(0.);
  double n44(0.);
  double n51(0.);
  // Test QED shower.
  double nGam43(0.);
  double nGam51(0.);
  double nGam52(0.);
  
  // Begin event loop. Generate event. Abort if error.
  Event& event    = pythia.event;
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

    if (!pythia.next()) {
      cout << " Event generation aborted prematurely, owing to error!\n";
      cout<< "Event number was : "<<iEvent<<endl;
      break;
    }

    // Check for weights
    double weight = pythia.info.weight();
    sumWeights += weight;

    // Do counts
    for (int i=1; i<event.size(); ++i) {
      if (event[i].isFinal()) ++nFinal;
      if (abs(event[i].status()) == 43) ++n43;
      if (abs(event[i].status()) == 44) ++n44;
      if (abs(event[i].status()) == 51) ++n51;
      if (event[i].idAbs() == 22) {
        if (abs(event[i].status()) == 51) ++nGam51;
        else if (abs(event[i].status()) == 52) ++nGam52;
        else if (abs(event[i].status()) == 43) ++nGam43;
      }
    }
  }
  
  // End of event loop. Determine run time.
  stop = clock(); // Stop timer
  t = (double) (stop-start)/CLOCKS_PER_SEC;
  
  // Statistics.
  pythia.stat();
  ofstream ofs("test202.dat");
  ofs << " Test results :" << endl;
  ofs << "\n <weight>      = " << fixed << setprecision(3)
      << sumWeights/nEvent <<endl;
  ofs << " <nFinal>      = " << setprecision(0) << nFinal/sumWeights << endl;
  ofs << "\n Average number of partons by status code: "<<endl;
  ofs << " <n51>         = " << setprecision(0) << n51/sumWeights << endl;
  ofs << " <n43>         = " << setprecision(0) << n43/sumWeights << endl;
  ofs << " <n44>         = " << setprecision(0) << n44/sumWeights << endl;
  ofs << " <nGam51>      = " << setprecision(1) << nGam51/sumWeights << endl;
  ofs << " <nGam52>      = " << setprecision(2) << nGam52/sumWeights << endl;
  ofs << " <nGam43>      = " << setprecision(2) << nGam43/sumWeights << endl;
  
  // Print runtime
  cout << "\n" << "|----------------------------------------|" << endl;
  cout << "| CPU Runtime = " << fixed << setprecision(1) << t << " sec" << endl;
  cout << "|----------------------------------------|" << "\n" << endl;

  // Done.
  return 0;
}
