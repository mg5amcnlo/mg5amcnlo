
// DIRE includes.
#include "Dire/Dire.h"

// Pythia includes.
#include "Pythia8/Pythia.h"

using namespace Pythia8;

//==========================================================================

int main( int argc, char* argv[]  ){

  // Check that correct number of command-line arguments
  if (argc != 1) {
    cerr << " Unexpected number of command-line arguments ("<<argc-1<<"). \n"
         << " This example program uses no arguments, but "
         << argc-1 << " arguments provided:";
         for ( int i=1; i<argc; ++i) cerr << " " << argv[i];
         cerr << "\n Program stopped. " << endl;
    return 1;
  }

  Pythia pythia;

  // Create and initialize DIRE shower plugin.
  Dire dire;
  //dire.init(pythia, "lep.cmnd");
  dire.init(pythia, "lhc.cmnd");

  // Jet finder for some of the histograms.
  SlowJet* slowJet = new SlowJet(-1, 1.0, 20., 4.4, 2, 2, NULL, false);
  Hist njets("njets",7,-0.5,6.5);

  double wmax =-1e15;
  double wmin = 1e15;
  double sumwt = 0.;
  double sumwtsq = 0.;

  // Start generation loop
  int nEvent = pythia.settings.mode("Main:numberOfEvents");
  for( int iEvent=0; iEvent<nEvent; ++iEvent ){

    // Generate next event
    if( !pythia.next() ) {
      if( pythia.info.atEndOfFile() )
        break;
      else continue;
    }

    // Get event weight(s).
    double evtweight         = pythia.info.weight();

    // Do not print zero-weight events.
    if ( evtweight == 0. ) continue;

    // Retrieve the shower weight.
    dire.weightsPtr->calcWeight(0.);
    dire.weightsPtr->reset();
    double wt = dire.weightsPtr->getShowerWeight();

    if (abs(wt) > 1e3) {
      cout << scientific << setprecision(8)
      << "Warning in DIRE main program dire00.cc: Large shower weight wt="
      << wt << endl;
      if (abs(wt) > 1e4) { 
        cout << "Warning in DIRE main program dire00.cc: Shower weight larger"
        << " than 10000. Discard event with rare shower weight fluctuation."
        << endl;
        evtweight = 0.;
      }
    }
    // Do not print zero-weight events.
    if ( evtweight == 0. ) continue;

    evtweight *= wt;

    wmin = min(wmin,wt);
    wmax = max(wmax,wt);
    sumwt += wt;
    sumwtsq+=pow2(wt);

    // Construct input for the jet algorithm.
    Event jetInput;
    jetInput.init("jet input",&pythia.particleData);
    jetInput.clear();
    for (int i =0; i < pythia.event.size(); ++i)
      if ( pythia.event[i].isFinal()
        && (pythia.event[i].colType() != 0 || pythia.event[i].isHadron()) )
        jetInput.append(pythia.event[i]);
    // Run jet algorithm.
    slowJet->analyze(jetInput);
    // Histogram number of jets.
    int nJet = slowJet->sizeJet();
    for (int i =0; i <= nJet; ++i) njets.fill(i, evtweight); 

  } // end loop over events to generate

  // print cross section, errors
  pythia.stat();

  // Normalise histograms
  njets *= pythia.info.sigmaGen() / pythia.info.nAccepted();

  // Print histograms.
  cout << njets;

  cout << endl
       << "\t Minimal shower weight=" << wmin
       << "\n\t Maximal shower weight=" << wmax
       << "\n\t Mean shower weight=" << sumwt/double(nEvent)
       << "\n\t Variance of shower weight="
       << sqrt(1/double(nEvent)*(sumwtsq - pow(sumwt,2)/double(nEvent)))
       << endl << endl;

  // Done
  return 0;

}
