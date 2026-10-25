// Driver for Pythia 8. Reads an input file dynamically created on
// the basis of the inputs specified in MCatNLO_MadFKS_PY8.Script 
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"
#include "Pythia8Plugins/aMCatNLOHooks.h"
#include "Pythia8Plugins/CombineMatchingInput.h"
#include "HepMC/GenEvent.h"
#include "HepMC/IO_GenEvent.h"
#include "LHEF.h"

using namespace Pythia8;

int main() {
  Pythia pythia;

  string inputname="Pythia8.cmd",outputname="Pythia8.hep";

  if (!pythia.readFile(inputname.c_str())) {
    std::cerr << "Error: invalid or unsupported Pythia settings" << std::endl;
    return 1;
  }

  //Create UserHooks pointer for the FxFX matching. Stop if it failed. Pass pointer to Pythia.
  CombineMatchingInput* combined = NULL;
  UserHooks* matching            = NULL;

  int nAbort=10;
  int nPrintLHA=1;
  int iAbort=0;
  int iPrintLHA=0;
  int iEventtot=pythia.mode("Main:numberOfEvents");
  int iEventshower=pythia.mode("Main:spareMode1");
  // A continuation consumes the accepted LHE once, preserving its weights.
  const bool continuation=pythia.flag("Main:spareFlag1");
  if (continuation) pythia.settings.forceParm("SpaceShower:pT0Ref", 0.);
  LHEF::Reader* sourceWeights=continuation
      ? new LHEF::Reader(pythia.word("Beams:LHEF")) : NULL;
  string evt_norm=pythia.word("Main:spareWord1");

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

  if (!pythia.init() && continuation) return 1;

  HepMC::Pythia8ToHepMC ToHepMC;
  HepMC::IO_GenEvent ascii_io(outputname.c_str(), std::ios::out);
  // Do not store cross section information, as this will be done manually.
  ToHepMC.set_store_pdf(false);
  ToHepMC.set_store_proc(false);
  ToHepMC.set_store_xsec(false);

  // Cross section an error.
  double sigmaTotal  = 0.;
  double errorTotal  = 0.;

  for (int iEvent = 0; ; ++iEvent) {
    if (continuation && iEvent >= iEventshower) break;
    if (!pythia.next()) {
      if (continuation) {
        std::cerr << "Continuation shower failed at input event " << iEvent+1 << "\n";
        return 1;
      }
      if (++iAbort < nAbort) continue;
      break;
    }
    // the number of events read by Pythia so far
    int nSelected=pythia.info.nSelected();

    if (!continuation && nSelected > iEventshower) break;
    if (continuation && (nSelected != iEvent+1 || pythia.info.nTried() != iEvent+1)) {
      std::cerr << "Continuation shower skipped an input event\n";
      return 1;
    }
    if (pythia.info.isLHA() && iPrintLHA < nPrintLHA) {
      pythia.LHAeventList();
      pythia.info.list();
      pythia.process.list();
      pythia.event.list();
      ++iPrintLHA;
    }

    HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
    double evtweight = pythia.info.weight();
    if (continuation) {
      if (!sourceWeights->readEvent()) {
        std::cerr << "Missing continuation source weight at event " << iEvent+1 << "\n";
        return 1;
      }
      evtweight=sourceWeights->hepeup.XWGTUP;
    }
    double normhepmc;
    // ALWAYS NORMALISE HEPMC WEIGHTS TO SUM TO THE CROSS SECTION
    if (evt_norm != "sum") {
      normhepmc = 1. / double(iEventshower);
    } else {
      normhepmc = double(iEventtot) / double(iEventshower);
    }
    sigmaTotal += evtweight*normhepmc;
    if (!continuation) hepmcevt->weights().push_back(evtweight*normhepmc);
    ToHepMC.fill_next_event( pythia, hepmcevt );
    if (continuation) {
      // Stock strategy -3 exposes a sign through Info. Restore the retained
      // LHE normalization for the nominal and any shower variation weights.
      const double factor=pythia.info.weight()!=0.
          ? evtweight*normhepmc/pythia.info.weight() : 0.;
      for (size_t i=0; i<hepmcevt->weights().size(); ++i)
        hepmcevt->weights()[i]*=factor;
      if (hepmcevt->weights().empty()) hepmcevt->weights().push_back(evtweight*normhepmc);
      else hepmcevt->weights()[0]=evtweight*normhepmc;
    }
    // Add the weight of the current event to the cross section.
    // Report cross section to hepmc
    HepMC::GenCrossSection xsec;
    xsec.set_cross_section( sigmaTotal, pythia.info.sigmaErr() );
    hepmcevt->set_cross_section( xsec );
    // Write the HepMC event to file. Done with it.
    ascii_io << hepmcevt;    
    delete hepmcevt;
  }

  delete sourceWeights;
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
