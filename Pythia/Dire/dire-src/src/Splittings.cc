
#include "Dire/Splittings.h"
#include "Dire/DireSpace.h"
#include "Dire/DireTimes.h"

namespace Pythia8 {

//==========================================================================

// The Splitting class.

//--------------------------------------------------------------------------

void Splitting::init() {

  renormMultFac      = 1.;
  if (id.find("isr_") != string::npos)
    renormMultFac    = settingsPtr->parm("SpaceShower:renormMultFac");
  else
    renormMultFac    = settingsPtr->parm("TimeShower:renormMultFac");

  if (id.find("_qcd_") != string::npos) is_qcd  = true;
  if (id.find("_qed_") != string::npos) is_qed  = true;
  if (id.find("_ew_")  != string::npos) is_ewk  = true;
  if (id.find("_CS")   != string::npos) is_dire = true;
  if (id.find("isr_")  != string::npos) is_isr  = true;
  if (id.find("fsr_")  != string::npos) is_fsr  = true;

  nameHash = shash(id);

}

//--------------------------------------------------------------------------

double Splitting::getKernel(string key) {
  map<string, double>::iterator it = kernelVals.find(key);
  if ( it == kernelVals.end() ) return 0./0.;
  return it->second;
}

//==========================================================================

} // end namespace Pythia8
