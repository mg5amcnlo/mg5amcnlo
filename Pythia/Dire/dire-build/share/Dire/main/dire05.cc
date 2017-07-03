
// DIRE includes.
#include "Dire/Dire.h"

// Pythia includes.
#include "Pythia8/Pythia.h"

using namespace Pythia8;

// Include analysis. You can replace this include with another file
// to include your own analysis.
#include "DirePlugins/analyses/AnalysisDummy.h"

//==========================================================================

int main( int argc, char* argv[] ){

  // Check that correct number of command-line arguments
  if (argc < 2) {
    cerr << " Unexpected number of command-line arguments ("<<argc-1<<"). \n"
         << " You are expected to provide the arguments." << endl
         << argc-1 << " arguments provided:";
         for ( int i=1; i<argc; ++i) cerr << " " << argv[i];
         cerr << "\n Program stopped. " << endl;
    return 1;
  }

  Pythia pythia;

  // Create and initialize DIRE shower plugin.
  Dire dire;
  dire.init(pythia, argv[1]);

  // Histogram the weight.
  Hist histWT("weight",100000,-5000.,5000.);

  // Initialize Pythia analysis.
  MyAnalysis analysis;
  analysis.init();

  cout << endl << endl << endl;
  cout << "Start generating events" << endl;

  int nEvent = pythia.settings.mode("Main:numberOfEvents");

  double wmax =-1e15;
  double wmin = 1e15;
  double sumwt = 0.;
  double sumwtsq = 0.;

  // Start generation loop
  for( int iEvent=0; iEvent<nEvent; ++iEvent ){

    // Generate next event
    if( !pythia.next() ) {
      if( pythia.info.atEndOfFile() )
        break;
      else continue;
    }

    // Get event weight(s).
    double evtweight         = pythia.info.weight();

    // Retrieve the shower weight.
    dire.weightsPtr->calcWeight(0.);
    dire.weightsPtr->reset();
    double wt = dire.weightsPtr->getShowerWeight();
    evtweight *= wt;

    if (abs(wt) > 1e3) {
      cout << scientific << setprecision(8)
      << "Warning in DIRE main program dire03.cc: Large shower weight wt="
      << wt << endl;
      if (abs(wt) > 1e4) { 
        cout << "Warning in DIRE main program dire03.cc: Shower weight larger"
        << " than 10000. Discard event with rare shower weight fluctuation."
        << endl;
        evtweight = 0.;
      }
    }
    // Do not print zero-weight events.
    if ( evtweight == 0. ) continue;

    wmin = min(wmin,wt);
    wmax = max(wmax,wt);
    sumwt += wt;
    sumwtsq+=pow2(wt);
    histWT.fill( wt, 1.0);

    analysis.fill(pythia.event,wt);

  } // end loop over events to generate

  // print cross section, errors
  pythia.stat();

  cout << scientific << setprecision(6)
       << "\t Minimal shower weight     = " << wmin << "\n"
       << "\t Maximal shower weight     = " << wmax << "\n"
       << "\t Mean shower weight        = " << sumwt/double(nEvent) << "\n"
       << "\t Variance of shower weight = "
       << sqrt(1/double(nEvent)*(sumwtsq - pow(sumwt,2)/double(nEvent)))
       << endl;

  ofstream write;
  // Write histograms to file
  write.open("wt.dat");
  histWT.table(write);
  write.close();

  analysis.finalize();
  analysis.print();

  // Done
  return 0;

}
