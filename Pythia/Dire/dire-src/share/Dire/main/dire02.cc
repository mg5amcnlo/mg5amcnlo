
// DIRE includes.
#include "Dire/Dire.h"

// Pythia includes.
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

using namespace Pythia8;

//==========================================================================

int main( int argc, char* argv[] ){

  // Check that correct number of command-line arguments
  if (argc < 3) {
    cerr << " Unexpected number of command-line arguments ("<<argc-1<<"). \n"
         << " You are expected to provide the arguments" << endl
         << " 1. Input file for settings" << endl
         << " 2. Output HepMC file" << endl
         << argc-1 << " arguments provided:";
         for ( int i=1; i<argc; ++i) cerr << " " << argv[i];
         cerr << "\n Program stopped. " << endl;
    return 1;
  }

  Pythia pythia;

  // New setting to allow processing of multiple input LHEFs.
  pythia.settings.addMode("LHEFInputs:nSubruns",0,true,false,0,100);

  // Create and initialize DIRE shower plugin.
  Dire dire;
  dire.init(pythia, argv[1]);

  // Interface for conversion from Pythia8::Event to HepMC one. 
  HepMC::Pythia8ToHepMC ToHepMC;
  // Specify file where HepMC events will be stored.
  HepMC::IO_GenEvent ascii_io(argv[2], std::ios::out);
  // Switch off warnings for parton-level events.
  ToHepMC.set_print_inconsistency(false);
  ToHepMC.set_free_parton_exception(false);
  // Do not store cross section information, as this will be done manually.
  ToHepMC.set_store_pdf(false);
  ToHepMC.set_store_proc(false);
  ToHepMC.set_store_xsec(false);

  // Get number of subruns.
  int nMerge = pythia.mode("LHEFInputs:nSubruns");
  int nEvent = pythia.settings.mode("Main:numberOfEvents");
  double sigmaTotal = 0.0;

  // Histogram the weight.
  Hist histWT("weight",10000,-500.,500.);

  // Loop over subruns with varying number of jets.
  for (int iMerge = 0; iMerge < nMerge; ++iMerge) {

    // Initialize with the LHE file for the current subrun.
    dire.init(pythia, argv[1], iMerge);

    double wmax =-1e15;
    double wmin = 1e15;
    double sumwt = 0.;
    double sumwtsq = 0.;

    // Get the inclusive x-section by summing over all process x-sections.
    double xs = 0.;
    for (int i=0; i < pythia.info.nProcessesLHEF(); ++i)
      xs += pythia.info.sigmaLHEF(i);

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

      // Work with weighted (LHA strategy=-4) events.
      double normhepmc = 1.;
      if (abs(pythia.info.lhaStrategy()) == 4)
        normhepmc = 1. / double(1e9*nEvent);
      // Work with unweighted events.
      else
        normhepmc = xs / double(1e9*nEvent);

      // Construct new empty HepMC event.
      HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
      // Set event weight
      hepmcevt->weights().push_back(evtweight*normhepmc);
      // Fill HepMC event
      ToHepMC.fill_next_event( pythia, hepmcevt );
 
      // Add the weight of the current event to the cross section.
      sigmaTotal  += evtweight*normhepmc;
      // Report cross section to hepmc
      HepMC::GenCrossSection xsec;
      xsec.set_cross_section( sigmaTotal*1e9, pythia.info.sigmaErr()*1e9 );
      hepmcevt->set_cross_section( xsec );
      // Write the HepMC event to file. Done with it.
      ascii_io << hepmcevt;
      delete hepmcevt;

    } // end loop over events to generate

    // print cross section, errors
    pythia.stat();

    cout << "\t Minimal shower weight=" << wmin
       << "\n\t Maximal shower weight=" << wmax
       << "\n\t Mean shower weight=" << sumwt/double(nEvent)
       << "\n\t Variance of shower weight="
       << sqrt(1/double(nEvent)*(sumwtsq - pow(sumwt,2)/double(nEvent)))
       << endl;

  }

  ofstream write;
  // Write histograms to file
  write.open("wt.dat");
  histWT.table(write);
  write.close();

  // Done
  return 0;

}
