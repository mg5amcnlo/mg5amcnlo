
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

  int nExtraEst = 100;
  int nEventEst = nExtraEst*pythia.settings.mode("Main:numberOfEvents");

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

  // For DIS, force PhaseSpace:pTHatMinDiverge to something very small.
  pythia.settings.forceParm("PhaseSpace:pTHatMinDiverge",1e-6);

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
  int nA    = pythia.info.nAccepted() / nExtraEst;

  // Histogram the weight.
  Hist histWT("weight",100000,-5000.,5000.);
  vector<double> wtmax;

  string hepmcfile = string(argv[2]);
  // Interface for conversion from Pythia8::Event to HepMC one. 
  HepMC::Pythia8ToHepMC ToHepMC;
  // Specify file where HepMC events will be stored.
  HepMC::IO_GenEvent ascii_io(hepmcfile, std::ios::out);
  // Switch off warnings for parton-level events.
  ToHepMC.set_print_inconsistency(false);
  ToHepMC.set_free_parton_exception(false);
  // Do not store cross section information, as this will be done manually.
  ToHepMC.set_store_pdf(false);
  ToHepMC.set_store_proc(false);
  ToHepMC.set_store_xsec(false);
  vector< HepMC::IO_GenEvent* > ev;
  vector<double> sigmaTot, errorTot;
  if ( pythia.settings.flag("Variations:doVariations") ) { 
    //for (int iwt=0; iwt < dire.weightsPtr->sizeWeights(); ++iwt) {
    //  string newfile = hepmcfile + "-" + dire.weightsPtr->weightName(iwt);
    //  std::replace(newfile.begin(), newfile.end(),' ', '_');
    //  std::replace(newfile.begin(), newfile.end(),':', '_');
    //  ev.push_back( new HepMC::IO_GenEvent(newfile, std::ios::out));
    //  sigmaTot.push_back(0.);
    //  errorTot.push_back(0.);
    //}
    for (int iwt=0; iwt < 3; ++iwt) {
      ostringstream c; c << iwt;
      string newfile = hepmcfile + c.str();
      ev.push_back( new HepMC::IO_GenEvent(newfile, std::ios::out));
      sigmaTot.push_back(0.);
      errorTot.push_back(0.);
    }
  }

  int nEvent = pythia.settings.mode("Main:numberOfEvents");

  // Cross section an error.
  double sigmaInc(0.), sigmaTotal(0.), errorTotal(0.);

  cout << endl << endl << endl;
  cout << "Start generating events" << endl;

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
    double pswt = dire.weightsPtr->getShowerWeight();

    if (abs(pswt) > 1e3) {
      cout << scientific << setprecision(8)
      << "Warning in DIRE main program dire03.cc: Large shower weight wt="
      << pswt << endl;
      if (abs(pswt) > 1e4) { 
        cout << "Warning in DIRE main program dire03.cc: Shower weight larger"
        << " than 10000. Discard event with rare shower weight fluctuation."
        << endl;
        evtweight = 0.;
      }
      // Print diagnostic output.
      dire.debugInfo.print(1);
      evtweight = 0.;
    }
    // Do not print zero-weight events.
    if ( evtweight == 0. ) continue;

    // Now retrieve additional shower weights, and combine these
    // into muR-up and muR-down variations.
    vector<double> pswts;
    //if (pythia.settings.flag("Variations:doVariations")) { 
    //  for (int iwt=0; iwt < dire.weightsPtr->sizeWeights(); ++iwt) {
    //    string key = dire.weightsPtr->weightName(iwt);
    //    pswts.push_back(dire.weightsPtr->getShowerWeight(key));
    //  }
    //}

    if (pythia.settings.flag("Variations:doVariations")) { 
      pswts.push_back(dire.weightsPtr->getShowerWeight("base"));
      bool hasupvar(false), hasdownvar(false);
      double uvar(1.), dvar(1.);
      // Get ISR variations.
      if ( pythia.settings.flag("PartonLevel:ISR")) {
        if ( pythia.settings.parm("Variations:muRisrUp") != 1.) {
          hasupvar=true;
          uvar *= dire.weightsPtr->getShowerWeight("Variations:muRisrUp");
        }
        if ( pythia.settings.parm("Variations:muRisrDown") != 1.) {
          hasdownvar=true;
          dvar *= dire.weightsPtr->getShowerWeight("Variations:muRisrDown");
        }
      }
      // Get FSR variations.
      if ( pythia.settings.flag("PartonLevel:FSR")) {
        if ( pythia.settings.parm("Variations:muRfsrUp") != 1.) {
          hasupvar=true;
          uvar *= dire.weightsPtr->getShowerWeight("Variations:muRfsrUp");
        }
        if ( pythia.settings.parm("Variations:muRfsrDown") != 1.) {
          hasdownvar=true;
          dvar *= dire.weightsPtr->getShowerWeight("Variations:muRfsrDown");
        }
      }
      if (hasupvar && abs(uvar) < 1e3)   pswts.push_back(uvar);
      else            pswts.push_back(0.0);
      if (hasdownvar && abs(dvar) < 1e3) pswts.push_back(dvar);
      else            pswts.push_back(0.0);
    }

    wmin = min(wmin,pswt);
    wmax = max(wmax,pswt);

    sumwt += pswt;
    sumwtsq+=pow2(pswt);
    histWT.fill( pswt, 1.0);

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

    if (pythia.settings.flag("PhaseSpace:bias2Selection"))
      normhepmc = xs / (sumSH);

    if (pythia.event.size() > 3) {

      // Set event weight
      hepmcevt->weights().push_back(evtweight*pswt*normhepmc);
      // Fill HepMC event
      ToHepMC.fill_next_event( pythia, hepmcevt );
       // Add the weight of the current event to the cross section.
      sigmaInc    += evtweight*normhepmc;
      sigmaTotal  += evtweight*pswt*normhepmc;
      errorTotal  += pow2(evtweight*pswt*normhepmc);
      // Report cross section to hepmc
      HepMC::GenCrossSection xsec;
      xsec.set_cross_section( sigmaTotal*1e9, pythia.info.sigmaErr()*1e9 );
      hepmcevt->set_cross_section( xsec );
      // Write the HepMC event to file. Done with it.
      ascii_io << hepmcevt;
      delete hepmcevt;

      // Write additional HepMC events.
      for (int iwt=0; iwt < int(pswts.size()); ++iwt) {
        HepMC::GenEvent* evt = new HepMC::GenEvent();
        // Set event weight
        double w = evtweight*pswts[iwt]*normhepmc;
        evt->weights().push_back(w);
        // Fill HepMC event
        ToHepMC.fill_next_event( pythia, evt );
        // Add the weight of the current event to the cross section.
        sigmaTot[iwt]  += w;
        errorTot[iwt]  += pow2(w);
        // Report cross section to hepmc
        HepMC::GenCrossSection xss;
        xss.set_cross_section( sigmaTot[iwt]*1e9, pythia.info.sigmaErr()*1e9 );
        evt->set_cross_section( xss );
        // Write the HepMC event to file. Done with it.
        *ev[iwt] << evt;
        delete evt;
      } 
    }

  } // end loop over events to generate

  // print cross section, errors
  pythia.stat();

  int nAccepted = pythia.info.nAccepted();
  cout << scientific << setprecision(6)
       << "\t Minimal shower weight     = " << wmin << "\n"
       << "\t Maximal shower weight     = " << wmax << "\n"
       << "\t Mean shower weight        = " << sumwt/double(nAccepted) << "\n"
       << "\t Variance of shower weight = "
       << sqrt(1/double(nAccepted)*(sumwtsq - pow(sumwt,2)/double(nAccepted)))
       << endl;

  cout << "Inclusive cross section    : " << sigmaInc << endl;
  cout << "Cross section after shower : " << sigmaTotal << endl;

  //ofstream writewt;
  //// Write histograms to file
  //writewt.open("wt.dat");
  //histWT.table(writewt);
  //writewt.close();

  if ( pythia.settings.flag("Variations:doVariations") ) { 
    //for (int iwt=0; iwt < dire.weightsPtr->sizeWeights(); ++iwt) {
    for (int iwt=0; iwt < 3; ++iwt) {
      delete ev[iwt];
    }
  }

  // Done
  return 0;

}
