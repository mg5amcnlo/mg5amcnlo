// main63.cc is a part of the PYTHIA event generator.
// Copyright (C) 2016 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Example how you can use UserHooks to enhance rare emission rates,
// in this case q -> q gamma.
// To concentrate on the photons from the showers, MPI and hadronization
// are switched off by default.

#include "Pythia8/Pythia.h"
using namespace Pythia8;

//==========================================================================

int main() {

  // Histogram pT spectrum of photons and event weights.
  Hist gamWithEnh( "gamma pT spectrum, with enhancement", 100, 0., 500.);
  Hist eventWt(    "log10(event weight)",                 100, -7., 3.);

    // Generator.
    Pythia pythia;

    pythia.settings.addParm("Weights:fsr:Q2QG:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:G2GG:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:G2QQ:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:Q2QA:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:A2LL:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:A2QQ:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:Q2QW:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:Q2QHV:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:G2GG:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:Q2GQ:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:Q2QG:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:G2QQ:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:A2QQ:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:Q2QA:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:Q2AQ:enhance",1.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:Q2QW:enhance",1.0,true,false,0.,0.);

    pythia.settings.addParm("Weights:fsr:Q2QG:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:G2GG:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:G2QQ:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:Q2QA:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:A2LL:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:A2QQ:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:Q2QW:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:fsr:Q2QHV:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:G2GG:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:Q2GQ:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:Q2QG:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:G2QQ:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:A2QQ:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:Q2QA:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:Q2AQ:vetoProb",0.0,true,false,0.,0.);
    pythia.settings.addParm("Weights:isr:Q2QW:vetoProb",0.0,true,false,0.,0.);

    pythia.readFile("main65.cmnd");
    int nEvent = pythia.mode("Main:numberOfEvents");

    // LHC initialization.
    pythia.init();

    // Begin event loop.
    double sumWt = 0.;
    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

      // Generate events. Find and histogram event weight.
      pythia.next();
      //double weight = pythia.weights.getEnhancedEventWeight();
      double weight = pythia.weights.getWeight("enhance");
      eventWt.fill( log10(weight) );
      sumWt += weight;

      // Find all final-state photons and histogram them.
      for (int i = 0; i < pythia.event.size(); ++i)
      if (pythia.event[i].isFinal() && pythia.event[i].id() == 22) {
        double pT = pythia.event[i].pT();
        gamWithEnh.fill( pT, weight);
      }

    // End of event loop.
    }

    // Statistics.
    pythia.stat();
    cout << "\n Average event weight = " << scientific
         << sumWt / nEvent << endl;

  gamWithEnh *= pythia.info.sigmaGen()/pythia.info.nAccepted()/5.;

  // Write histograms to output stream.
  cout << gamWithEnh << eventWt;

  // Write histogram data to files.
  ofstream write;
  write.open("PTA.dat");
  gamWithEnh.table(write);
  write.close();

  // Done.
  return 0;
}
