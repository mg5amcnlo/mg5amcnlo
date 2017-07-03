
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
         << " 2. Output hepmc file name" << endl
         << argc-1 << " arguments provided:";
         for ( int i=1; i<argc; ++i) cerr << " " << argv[i];
         cerr << "\n Program stopped. " << endl;
    return 1;
  }

  Pythia pythia;

  // Create and initialize DIRE shower plugin.
  Dire dire;
  dire.init(pythia, argv[1]);

  int nEventEst = pythia.settings.mode("Main:numberOfEvents");

  // Switch off all showering and MPI when estimating the cross section,
  // and re-initialise (unfortunately).
  bool fsr = pythia.flag("PartonLevel:FSR");
  bool isr = pythia.flag("PartonLevel:ISR");
  bool mpi = pythia.flag("PartonLevel:MPI");
  bool had = pythia.flag("HadronLevel:all");
  bool rem = pythia.flag("PartonLevel:Remnants");
  bool chk = pythia.flag("Check:Event");
  pythia.settings.flag("PartonLevel:FSR",false);
  pythia.settings.flag("PartonLevel:ISR",false);
  pythia.settings.flag("PartonLevel:MPI",false);
  pythia.settings.flag("HadronLevel:all",false);
  pythia.settings.flag("PartonLevel:Remnants",false);
  pythia.settings.flag("Check:Event",false);
  pythia.init();

  // Cross section estimate run.
  double sumSH = 0.;
  double nAcceptSH = 0.;
  for( int iEvent=0; iEvent<nEventEst; ++iEvent ){
    // Generate next event
    if( !pythia.next() ) {
      if( pythia.info.atEndOfFile() )
        break;
      else continue;
    }

    sumSH     += pythia.info.weight();
    map <string,string> eventAttributes;
    if (pythia.info.eventAttributes)
      eventAttributes = *(pythia.info.eventAttributes);
    string trials = (eventAttributes.find("trials") != eventAttributes.end())
                  ?  eventAttributes["trials"] : "";
    if (trials != "") nAcceptSH += atof(trials.c_str());
  }
  pythia.stat();
  double xs = pythia.info.sigmaGen();
  int nA    = pythia.info.nAccepted();

  // Histogram the weight.
  Hist histWT("weight",100000,-5000.,5000.);

  // Interface for conversion from Pythia8::Event to HepMC one. 
  HepMC::Pythia8ToHepMC ToHepMC;
  // Specify file where HepMC events will be stored.
  HepMC::IO_GenEvent ascii_io(argv[2], std::ios::out);
  //HepMC::IO_GenEvent ascii_io("/dev/null", std::ios::out);
  // Switch off warnings for parton-level events.
  ToHepMC.set_print_inconsistency(false);
  ToHepMC.set_free_parton_exception(false);
  // Do not store cross section information, as this will be done manually.
  ToHepMC.set_store_pdf(false);
  ToHepMC.set_store_proc(false);
  ToHepMC.set_store_xsec(false);

  int nEvent = pythia.settings.mode("Main:numberOfEvents");

  // Cross section an error.
  double sigmaTotal  = 0.;
  double errorTotal  = 0.;

  cout << endl << endl << endl;
  cout << "Start generating events" << endl;

  double sigmaSample = 0., errorSample = 0.;

  // Switch showering and multiple interaction back on.
  pythia.settings.flag("PartonLevel:FSR",fsr);
  pythia.settings.flag("PartonLevel:ISR",isr);
  pythia.settings.flag("HadronLevel:all",had);
  pythia.settings.flag("PartonLevel:MPI",mpi);
  pythia.settings.flag("PartonLevel:Remnants",rem);
  pythia.settings.flag("Check:Event",chk);
  pythia.init();

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

    // Do not print zero-weight events.
    if ( evtweight == 0. ) continue;

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

    // Construct new empty HepMC event.
    HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();

    double normhepmc = xs / double(nA);

    // Weighted events with additional number of trial events to consider.
    if ( pythia.info.lhaStrategy() != 0
      && pythia.info.lhaStrategy() != 3
      && nAcceptSH > 0)
      normhepmc = 1. / (1e9*nAcceptSH);
    // Weighted events.
    else if ( pythia.info.lhaStrategy() != 0
      && pythia.info.lhaStrategy() != 3
      && nAcceptSH == 0)
      normhepmc = 1. / (1e9*nA);

    if(pythia.event.size() > 3){
    // Set event weight
    hepmcevt->weights().push_back(evtweight*normhepmc);
    // Fill HepMC event
    ToHepMC.fill_next_event( pythia, hepmcevt );
 
    // Add the weight of the current event to the cross section.
    sigmaTotal  += evtweight*normhepmc;
    sigmaSample += evtweight*normhepmc;
    errorTotal  += pow2(evtweight*normhepmc);
    errorSample += pow2(evtweight*normhepmc);
    // Report cross section to hepmc
    HepMC::GenCrossSection xsec;
    xsec.set_cross_section( sigmaTotal*1e9, pythia.info.sigmaErr()*1e9 );
    hepmcevt->set_cross_section( xsec );
    // Write the HepMC event to file. Done with it.
    ascii_io << hepmcevt;
    delete hepmcevt;
    }

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

  // Done
  return 0;

}
