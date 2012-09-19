// Driver for Pythia 8. Reads an input file dynamically created on
// the basis of the inputs specified in MCatNLO_MadFKS_PY8.Script 
#include "Pythia.h"
using namespace Pythia8;
int main(int argc, char* argv[]) {
  Pythia pythia;                            
  if (argc != 2) {
    cerr << "Unexpected number of arguments, program stopped!" << endl;
    return 1;
  }
  pythia.readFile(argv[1]);
  pythia.init();

  int nAbort=10;
  int nPrintLHA=1;
  int iAbort=0;
  int iPrintLHA=0;

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

    // implement the analysis step!!
    for (int i = 0; i < pythia.event.size(); ++i) {
    }
  }
  pythia.stat();
  return 0;
}
