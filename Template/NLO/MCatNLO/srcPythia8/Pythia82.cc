// Driver for Pythia 8. Reads an input file dynamically created on
// the basis of the inputs specified in MCatNLO_MadFKS_PY8.Script 
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"
#include "Pythia8Plugins/aMCatNLOHooks.h"
#include "Pythia8Plugins/CombineMatchingInput.h"
#include "HepMC/GenEvent.h"
#include "HepMC/IO_GenEvent.h"
#include "HepMC/IO_BaseClass.h"
#include "HepMC/IO_HEPEVT.h"
#include "HepMC/HEPEVT_Wrapper.h"
#include "fstream"
#include "LHEFRead.h"

using namespace Pythia8;

extern "C" {
  extern struct {
    double EVWGT;
  } cevwgt_;
}
#define cevwgt cevwgt_

extern "C" { 
  void pyabeg_(int&,char(*)[50]);
  void pyaend_(double&);
  void pyanal_(int&,double(*));
}

int main() {
  Pythia pythia;

  int cwgtinfo_nn;
  char cwgtinfo_weights_info[1024][50];
  double cwgt_ww[1024];

  string inputname="Pythia8.cmd",outputname="Pythia8.hep";

  pythia.readFile(inputname.c_str());

  //Create UserHooks pointer for the FxFX matching. Stop if it failed. Pass pointer to Pythia.
  CombineMatchingInput* combined = NULL;
  UserHooks* matching            = NULL;

  string filename = pythia.word("Beams:LHEF");

  MyReader read(filename);
  read.lhef_read_wgtsinfo_(cwgtinfo_nn,cwgtinfo_weights_info);
  pyabeg_(cwgtinfo_nn,cwgtinfo_weights_info);

  int nAbort=10;
  int nPrintLHA=1;
  int iAbort=0;
  int iPrintLHA=0;
  int nstep=5000;
  int iEventtot=pythia.mode("Main:numberOfEvents");
  int iEventshower=pythia.mode("Main:spareMode1");
  string evt_norm=pythia.word("Main:spareWord1");
  int iEventtot_norm=iEventtot;
  if (evt_norm == "average"){
    iEventtot_norm=1;
  }

  //FxFx merging
  bool isFxFx=pythia.flag("JetMatching:doFxFx");
  if (isFxFx) {
    matching = combined->getHook(pythia);
    if (!matching) {
      std::cout << " Failed to initialise jet matching structures.\n"
                << " Program stopped.";
      return 1;
    }
    pythia.setUserHooksPtr(matching);
    int nJmax=pythia.mode("JetMatching:nJetMax");
    double Qcut=pythia.parm("JetMatching:qCut");
    double PTcut=pythia.parm("JetMatching:qCutME");
    if (Qcut <= PTcut || Qcut <= 0.) {
      std::cout << " \n";
      std::cout << "Merging scale (shower_card.dat) smaller than pTcut (run_card.dat)"
		<< Qcut << " " << PTcut << "\n";
      return 0;
    }
  }

  pythia.init();

  HepMC::IO_BaseClass *_hepevtio;
  HepMC::Pythia8ToHepMC ToHepMC;
  HepMC::IO_GenEvent ascii_io(outputname.c_str(), std::ios::out);
  double nSelected;
  int nTry;
  double norm;

  // Cross section
  double sigmaTotal  = 0.;
  int iLHEFread=0;

  for (int iEvent = 0; ; ++iEvent) {
    if (!pythia.next()) {
      if (++iAbort < nAbort) continue;
      break;
    }
    // the number of events read by Pythia so far
    nSelected=double(pythia.info.nSelected());
    // normalisation factor for the default analyses defined in pyanal_
    norm=iEventtot_norm*iEvent/nSelected;

    if (nSelected >= iEventshower) break;
    if (pythia.info.isLHA() && iPrintLHA < nPrintLHA) {
      pythia.LHAeventList();
      pythia.info.list();
      pythia.process.list();
      pythia.event.list();
      ++iPrintLHA;
    }

    double evtweight = pythia.info.weight();
    double normhepmc;
    // Add the weight of the current event to the cross section.
    normhepmc = 1. / double(iEventshower);
    if (evt_norm == "average") {
      sigmaTotal  += evtweight*normhepmc;
    } else {
      sigmaTotal  += evtweight*normhepmc*iEventtot;
    }

    HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
    ToHepMC.fill_next_event( pythia, hepmcevt );

    //define the IO_HEPEVT
    _hepevtio = new HepMC::IO_HEPEVT;
    _hepevtio->write_event(hepmcevt);
    
    //event weight
    cevwgt.EVWGT=hepmcevt->weights()[0];

    //call the FORTRAN analysis for this event. First, make sure to
    //re-synchronize the reading of the weights with the reading of
    //the event. (They get desynchronised if an event was rejected).
    nTry=pythia.info.nTried();
    for (; iLHEFread<nTry ; ++iLHEFread) {
      read.lhef_read_wgts_(cwgt_ww);
    }
    cwgt_ww[0]=cevwgt.EVWGT;
    pyanal_(cwgtinfo_nn,cwgt_ww);

    if (iEvent % nstep == 0 && iEvent >= 100){
      pyaend_(norm);
    }
    delete hepmcevt;
  }
  pyaend_(norm);

  pythia.stat();
  if (isFxFx){
    std::cout << " \n";
    std::cout << "*********************************************************************** \n";
    std::cout << "*********************************************************************** \n";
    std::cout << "Cross section, including FxFx merging is: "
	      << sigmaTotal << "\n";
    std::cout << "*********************************************************************** \n";
    std::cout << "*********************************************************************** \n";
  }

  return 0;
}
