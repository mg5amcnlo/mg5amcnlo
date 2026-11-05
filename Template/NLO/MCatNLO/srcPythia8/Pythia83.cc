// Driver for Pythia 8. Reads an input file dynamically created on
// the basis of the inputs specified in MCatNLO_MadFKS_PY8.Script 
// The events are passed to the Fortran analysis through the HEPEVT common
// block. With -DHEPMC3 (shower_card: hepmc_format) it is filled directly from
// the Pythia8 event record, otherwise through HepMC2.
#include "Pythia8/Pythia.h"
#ifndef HEPMC3
#include "Pythia8Plugins/HepMC2.h"
#endif
#include "Pythia8Plugins/CombineMatchingInput.h"
#ifndef HEPMC3
#include "HepMC/GenEvent.h"
#include "HepMC/IO_GenEvent.h"
#include "HepMC/IO_BaseClass.h"
#include "HepMC/IO_HEPEVT.h"
#include "HepMC/HEPEVT_Wrapper.h"
#include "HEPEVTSize.h"
#endif
#include "fstream"
#include "LHEFRead.h"

using namespace Pythia8;

#ifdef HEPMC3
// The HEPEVT common block of the analysis (MCatNLO/include/HEPMC.INC).
const int NMXHEP = 4000;
extern "C" {
  extern struct {
    int nevhep, nhep, isthep[NMXHEP], idhep[NMXHEP];
    int jmohep[NMXHEP][2], jdahep[NMXHEP][2];
    double phep[NMXHEP][5], vhep[NMXHEP][4];
  } hepevt_;
}

// Copy the event record (without its system entry 0) to HEPEVT, with the
// HepMC status codes, so that HEPEVT entry i is Pythia8 entry i.
void fillHEPEVT(Event & event, int iEvent) {
  static bool warned = false;
  int n = 0;
  for (int i = 1; i < event.size(); ++i) {
    if (n == NMXHEP) {
      if (!warned) cout << "Warning: events truncated to " << NMXHEP
                        << " entries in HEPEVT" << endl;
      warned = true;
      break;
    }
    Particle & p = event[i];
    hepevt_.isthep[n] = p.statusHepMC();
    hepevt_.idhep[n] = p.id();
    hepevt_.jmohep[n][0] = p.mother1();
    hepevt_.jmohep[n][1] = p.mother2();
    hepevt_.jdahep[n][0] = p.daughter1();
    hepevt_.jdahep[n][1] = p.daughter2();
    hepevt_.phep[n][0] = p.px();
    hepevt_.phep[n][1] = p.py();
    hepevt_.phep[n][2] = p.pz();
    hepevt_.phep[n][3] = p.e();
    hepevt_.phep[n][4] = p.m();
    hepevt_.vhep[n][0] = p.xProd();
    hepevt_.vhep[n][1] = p.yProd();
    hepevt_.vhep[n][2] = p.zProd();
    hepevt_.vhep[n][3] = p.tProd();
    ++n;
  }
  hepevt_.nhep = n;
  hepevt_.nevhep = iEvent;
}
#endif

extern "C" {
  extern struct {
    double EVWGT;
  } cevwgt_;
}
#define cevwgt cevwgt_

extern "C" { 
  void pyabeg_(int&,char(*)[wgts_info_len_used]);
  void pyaend_(double&);
  void pyanal_(int&,double(*));
}

int main() {
  Pythia pythia;

  int cwgtinfo_nn;
  char cwgtinfo_weights_info[1024][wgts_info_len_used];
  double cwgt_ww[1024];

  string inputname="Pythia8.cmd",outputname="Pythia8.hep";

  if (!pythia.readFile(inputname.c_str())) {
    std::cerr << "Error: invalid or unsupported Pythia settings" << std::endl;
    return 1;
  }

  //Create UserHooks pointer for the FxFX matching. Stop if it failed. Pass pointer to Pythia.
  CombineMatchingInput combined;
  //UserHooks* matching            = NULL;

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
  if (evt_norm != "sum"){
    iEventtot_norm=1;
  }

  //FxFx merging
  bool isFxFx=pythia.flag("JetMatching:doFxFx");
  if (isFxFx) {
    combined.setHook(pythia);
    //matching = combined->getHook(pythia);
    //if (!matching) {
    //  std::cout << " Failed to initialise jet matching structures.\n"
    //            << " Program stopped.";
    //  return 1;
    //}
    //pythia.setUserHooksPtr(matching);
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

  // Initialise Pythia.
  if (!pythia.init()) {
    cout << "Error: could not initialise Pythia" << endl;
    return 0;
  };

#ifndef HEPMC3
  HepMC::IO_BaseClass *_hepevtio;
  HepMC::Pythia8ToHepMC ToHepMC;
  HepMC::IO_GenEvent ascii_io(outputname.c_str(), std::ios::out);
#endif
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
    if (evt_norm != "sum") {
      sigmaTotal  += evtweight*normhepmc;
    } else {
      sigmaTotal  += evtweight*normhepmc*iEventtot;
    }

#ifdef HEPMC3
    fillHEPEVT(pythia.event, iEvent);
    //event weight: the first weight that HepMC2 stores for the event
    cevwgt.EVWGT=pythia.info.weightValueByIndex(0);
#else
    HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
    ToHepMC.fill_next_event( pythia, hepmcevt );

    //define the IO_HEPEVT
    _hepevtio = new HepMC::IO_HEPEVT;
    _hepevtio->write_event(hepmcevt);
    
    //event weight
    cevwgt.EVWGT=hepmcevt->weights()[0];
#endif

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
#ifndef HEPMC3
    delete hepmcevt;
#endif
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
