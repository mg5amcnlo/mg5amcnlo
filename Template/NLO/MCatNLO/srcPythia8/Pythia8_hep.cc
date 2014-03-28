// Driver for Pythia 8. Reads an input file dynamically created on
// the basis of the inputs specified in MCatNLO_MadFKS_PY8.Script 
#include "Pythia8/Pythia.h"
#include "Pythia8/Pythia8ToHepMC.h"
#include "HepMC/GenEvent.h"
#include "HepMC/IO_GenEvent.h"
#include "../examples/CombineMatchingInput.h"

using namespace Pythia8;

int main() {
  Pythia pythia;

  // Teach Pythia some additional settings (for FxFx).
  pythia.settings.addFlag("JetMatching:doFxFx",false);
  pythia.settings.addMode("JetMatching:nPartonsNow",0,true,false,0,10);
  pythia.settings.addParm("JetMatching:qCutME",5.,true,false,0.,-1.);

  string inputname="Pythia8.cmd",outputname="Pythia8.hep";

  pythia.readFile(inputname.c_str());

  //Create UserHooks pointer for the FxFX matching. Stop if it failed. Pass pointer to Pythia.
  CombineMatchingInput combined;
  UserHooks* matching = combined.getHook(pythia);
  if (!matching) return 1;
  pythia.setUserHooksPtr(matching);

  pythia.init();

  int nAbort=10;
  int nPrintLHA=1;
  int iAbort=0;
  int iPrintLHA=0;
  int iEventshower=pythia.mode("Main:spareMode1");

  //FxFx merging
  bool isFxFx=pythia.flag("JetMatching:doFxFx");
  if (isFxFx) {
    int nJnow=pythia.mode("JetMatching:nPartonsNow");
    int nJmax=pythia.mode("JetMatching:nJetMax");
    int iExcl=pythia.mode("JetMatching:exclusive");
    double Qcut=pythia.parm("JetMatching:qCut");
    double PTcut=pythia.parm("JetMatching:qCutME");
    if (Qcut <= PTcut || Qcut <= 0.) {
      std::cout << " \n";
      std::cout << "Merging scale (shower_card.dat) smaller than pTcut (run_card.dat)"
		<< Qcut << " " << PTcut << "\n";
      return 0;
    }
    if (nJnow >= 10 || nJnow < 0 || nJmax >= 10 || nJmax < 0 || nJnow > nJmax) {
      std::cout << " \n";
      std::cout << "Wrong inputs njmax and/or njnow in shower_card.dat "
		<< nJmax << " " << nJnow << "\n";
      return 0;
    }
    if (nJnow != nJmax && iExcl == 0) {
      std::cout << " \n";
      std::cout << "Inclusive merging required for a sample with non-max multiplicity\n";
      return 0;
    }
  }

  HepMC::Pythia8ToHepMC ToHepMC;
  HepMC::IO_GenEvent ascii_io(outputname.c_str(), std::ios::out);

  for (int iEvent = 0; ; ++iEvent) {
    if (!pythia.next()) {
      if (++iAbort < nAbort) continue;
      break;
    }
    if (iEvent >= iEventshower) break;
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

  if (isFxFx) {
    delete matching;
  }
  return 0;
}
