
#include "Dire/Splittings.h"
#include "Dire/DireSpace.h"
#include "Dire/DireTimes.h"

namespace Pythia8 {

//==========================================================================

// The Splitting class.

//--------------------------------------------------------------------------

void Splitting::init() {

  renormMultFac      = 1.;
  if (name().find("isr_") != string::npos)
    renormMultFac    = settingsPtr->parm("SpaceShower:renormMultFac");
  else
    renormMultFac    = settingsPtr->parm("TimeShower:renormMultFac");

}

//--------------------------------------------------------------------------

double Splitting::getKernel(string key) {
  map<string, double>::iterator it = kernelVals.find(key);
  if ( it == kernelVals.end() ) return 0./0.;
  return it->second;
}

//==========================================================================

} // end namespace Pythia8
