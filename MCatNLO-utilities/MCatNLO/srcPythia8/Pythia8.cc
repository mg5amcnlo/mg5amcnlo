// Driver for Pythia 8. Reads an input file dynamically created on
// the basis of the inputs specified in MCatNLO_MadFKS_PY8.Script 
#include "Pythia.h"
#include "HepMCInterface.h"
#include "HepMC/GenEvent.h"
#include "HepMC/IO_GenEvent.h"

using namespace Pythia8;
int main() {
  Pythia pythia;
  string inputname="Pythia8.cmd",outputname="Pythia8.hep";
  pythia.readFile(inputname.c_str());
  pythia.init();

  //read this from Pythia8.cmd 
  int nAbort=10;
  int nPrintLHA=1;
  int iAbort=0;
  int iPrintLHA=0;

  HepMC::I_Pythia8 ToHepMC;
  //ToHepMC.set_crash_on_problem();
  //Specify file where HepMC events will be stored.
  HepMC::IO_GenEvent ascii_io(outputname.c_str(), std::ios::out);


  for (int iEvent = 0; ; ++iEvent) {
    if (!pythia.next()) {
      if (++iAbort < nAbort) continue;
      break;
    }
    if (pythia.info.isLHA() && iPrintLHA < nPrintLHA) {
      pythia.LHAeventList();
      pythia.info.list();
      pythia.process.list();
      pythia.event.list();
      ++iPrintLHA;
    }
    HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
    ToHepMC.fill_next_event( pythia, hepmcevt );
    ascii_io << hepmcevt;
    delete hepmcevt;
  }

  pythia.stat();
  return 0;
}
