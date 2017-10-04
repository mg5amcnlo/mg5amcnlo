
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

  // Allow Pythia to use Dire merging classes. 
  MyMerging* merging           = new MyMerging();
  MyHardProcess* hardProcess   = new MyHardProcess();
  MyMergingHooks* mergingHooks = new MyMergingHooks();
  mergingHooks->setHardProcessPtr(hardProcess);
  pythia.setMergingHooksPtr(mergingHooks);
  pythia.setMergingPtr(merging);

  // Create and initialize DIRE shower plugin.
  Dire dire;
  //dire.init(pythia, "lep.cmnd");
  //dire.init(pythia, "lhc-dev.cmnd");
  dire.init(pythia, "nu-dev.cmnd");

  // Transfer initialized shower weights pointer to merging class. 
  merging->setWeightsPtr(dire.weightsPtr);

  // Gluon histograms.
  Hist zglue("zglue",50,0.,1.0);
  Hist zglueMax("zglue_max",50,0.,1.0);
  Hist zglueMin("zglue_min",50,0.,1.0);

  // Z-boson histograms.
  Hist ptz("ptz",100,0.,100.0);
  Hist ptzMax("ptz_max",100,0.,100.0);
  Hist ptzMin("ptz_min",100,0.,100.0);

  // Photon histograms.
  Hist ngamma("ngamma",10,0.,10.0);
  Hist zgamma("zgamma",200,0.,0.5);

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

    double wtMax = dire.weightsPtr->getShowerWeight("scaleUp");
    double wtMin = dire.weightsPtr->getShowerWeight("scaleDown");

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
      dire.debugInfo.print(1);
    }
    // Do not print zero-weight events.
    if ( evtweight == 0. ) continue;

    double evtweightMax = evtweight*wtMax;
    double evtweightMin = evtweight*wtMin;

    evtweight *= wt;

    wmin = min(wmin,wt);
    wmax = max(wmax,wt);
    sumwt += wt;
    sumwtsq+=pow2(wt);

    // Fill gluon histograms.
    for (int i =0; i < pythia.event.size(); ++i) if (pythia.event[i].isFinal()
      && pythia.event[i].id() == 21) {
      double z = 2.*pythia.event[i].p()*pythia.event[5].p()
                  / pythia.event[5].p().m2Calc();
      zglue.fill( z, evtweight );
      zglueMax.fill( z, evtweightMax );
      zglueMin.fill( z, evtweightMin );
    }

    // Fill Z-boson histograms.
    int iz=0;
    for (int i =0; i < pythia.event.size(); ++i)
      if (pythia.event[i].id() == 23) iz =i;
    double pTz = pythia.event[iz].pT();
    ptz.fill( pTz, evtweight );
    ptzMax.fill( pTz, evtweightMax );
    ptzMin.fill( pTz, evtweightMin );

    // Fill photon histograms.
    int na=0;
    for (int i =0; i < pythia.event.size(); ++i) if (pythia.event[i].isFinal()
      && pythia.event[i].id() == 22) {
      double z = 2.*pythia.event[i].e() / pythia.event[0].m();
      zgamma.fill( z, evtweight );
      na++;
    }
    ngamma.fill( na, evtweight );

  } // end loop over events to generate

  // print cross section, errors
  pythia.stat();

  // Normalise histograms
  zglue    *= pythia.info.sigmaGen() / pythia.info.nAccepted();
  zglueMin *= pythia.info.sigmaGen() / pythia.info.nAccepted();
  zglueMax *= pythia.info.sigmaGen() / pythia.info.nAccepted();
  ptz      *= pythia.info.sigmaGen() / pythia.info.nAccepted();
  ptzMin   *= pythia.info.sigmaGen() / pythia.info.nAccepted();
  ptzMax   *= pythia.info.sigmaGen() / pythia.info.nAccepted();
  zgamma   *= pythia.info.sigmaGen() / pythia.info.nAccepted();
  ngamma   *= pythia.info.sigmaGen() / pythia.info.nAccepted();

  // Write histograms to data table files.
  ofstream write;
  write.open("zglue.dat");
  zglue.table(write);
  write.close();
  write.open("zglueMax.dat");
  zglueMax.table(write);
  write.close();
  write.open("zglueMin.dat");
  zglueMin.table(write);
  write.close();
  write.open("ptz.dat");
  ptz.table(write);
  write.close();
  write.open("ptzMax.dat");
  ptzMax.table(write);
  write.close();
  write.open("ptzMin.dat");
  ptzMin.table(write);
  write.close();
  write.open("ngamma.dat");
  ngamma.table(write);
  write.close();
  write.open("zgamma.dat");
  zgamma.table(write);
  write.close();

  cout << endl
       << "\t Minimal shower weight=" << wmin
       << "\n\t Maximal shower weight=" << wmax
       << "\n\t Mean shower weight=" << sumwt/double(nEvent)
       << "\n\t Variance of shower weight="
       << sqrt(1/double(nEvent)*(sumwtsq - pow(sumwt,2)/double(nEvent)))
       << endl << endl;

  // Clean-up
  if (merging) delete merging;
  if (hardProcess) delete hardProcess;
  if (mergingHooks) delete mergingHooks;

  // Done
  return 0;

}
