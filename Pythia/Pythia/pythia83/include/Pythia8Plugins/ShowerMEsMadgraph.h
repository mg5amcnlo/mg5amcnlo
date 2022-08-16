// ShowerMEsMadgraph.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Stefan Prestel, Philip Ilten, Torbjorn
// Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the Madgraph parton shower ME plugin class which
// interfaces with matrix elements generated with the
// PY8Kernels/MG5MES plugin to MadGraph 5.

#ifndef Pythia8_ShowerMEsMadgraph_H
#define Pythia8_ShowerMEsMadgraph_H

// Include Pythia headers.
#include "Pythia8/ShowerMEs.h"

// Include Madgraph PY8MEs plugin headers.
#include "PY8ME.h"
#include "PY8MEs.h"

namespace Pythia8 {

//==========================================================================


class ShowerMEsMadgraph : public ShowerMEs {

public:

  // Constructor.
  ShowerMEsMadgraph() {isInitPtr = false; isInit = false;
    libPtr = nullptr; modelPtr = nullptr;}

  // Destructor.
  ~ShowerMEsMadgraph() {if (libPtr != nullptr) delete libPtr;
    if (modelPtr != nullptr) delete modelPtr;}

  // VINCIA methods.
  // Initialise the Madgraph model, parameters, and couplings.
  bool initVincia() override;
  // Get the matrix element squared for a particle state.
  double me2Vincia(vector<Particle> state, int nIn) override;
  // Check if the process is available.
  bool hasProcessVincia(vector<int> idIn, vector<int> idOut,
    set<int> sChan) override;

  // Dire methods.
  bool initDire(Info*, string card) override;
  bool isAvailableMEDire(vector <int> in, vector<int> out) override;
  bool isAvailableMEDire(const Pythia8::Event& event) override;
  double calcMEDire(const Pythia8::Event& event) override;

  // Common methods.
  // Get pointer to matrix element, e.g. to check if process exists in
  // library.
  PY8MEs_namespace::PY8ME* getProcess(vector<int> idIn, vector<int> idOut,
    set<int> sChan);

private:

  PY8MEs_namespace::PY8MEs* libPtr;
  PARS* modelPtr;

};

//--------------------------------------------------------------------------

// Initialise the Madgraph model, parameters, and couplings.

bool ShowerMEsMadgraph::initVincia() {

  // Check if pointers initialized.
  verbose = settingsPtr->mode("Vincia:verbose");
  if (verbose > VinciaConstants::NORMAL)
    printOut("ShowerMEsMadgraph::init", "begin", VinciaConstants::dashLen);
  if (!isInitPtr) {
    printOut("ShowerMEsMadgraph::init",
             "Cannot initialize, pointers not set.");
    return false;
  }
  isInit = true;

  // Set colour depth.
  colourDepth = 0.;

  // Create new model instance.
  if (verbose >= VinciaConstants::REPORT) printOut("ShowerMEsMadgraph::init",
    "setting MG5 C++ masses, widths, couplings...");
  if (modelPtr != nullptr) delete modelPtr;
  modelPtr = new PARS();
  modelPtr->setIndependentParameters(particleDataPtr,coupSMPtr,slhaPtr);
  modelPtr->setIndependentCouplings();
  if (verbose >= VinciaConstants::DEBUG) {
    modelPtr->printIndependentParameters();
    modelPtr->printIndependentCouplings();
  }

  // In the VINCIA context, alphaS_MGME = 1/4pi (- > gS = 1; we
  // control the running separately). Thus, even the Madgraph "dependent"
  // parameters only need to be set once.

  // Alternatively, we could evaluate the QCD coupling at MZ but should
  // then use a coupling definition from a Vincia parameter list rather
  // than PYTHIA's couplings.
  //    double muDefME  = 91.2;
  //    double alpS = coupSMPtr->alphaS(pow2(muDefME));

  // The following is equivalent to
  // PY8MEs::updateModelDependentCouplingsWithPY8(...)
  double alpS = 1.0 / ( 4 * M_PI );
  modelPtr->setDependentParameters(particleDataPtr, coupSMPtr, slhaPtr,
    alpS);
  modelPtr->setDependentCouplings();

  // Construct Madgraph process library.
  if (verbose >= VinciaConstants::DEBUG) printOut("ShowerMEsMadgraph::init()",
      "   attempting to construct lib");
  if (libPtr != nullptr) delete libPtr;
  libPtr = new PY8MEs_namespace::PY8MEs(modelPtr);
  // Set whether to include averaging and symmetry factors.
  //TODO: could be read in from flags.
  libPtr->seProcessesIncludeSymmetryFactors(false);
  libPtr->seProcessesIncludeHelicityAveragingFactors(false);
  libPtr->seProcessesIncludeColorAveragingFactors(false);

  return true;

}

//--------------------------------------------------------------------------

// Check if the process is available.

bool ShowerMEsMadgraph::hasProcessVincia(vector<int> idIn, vector<int> idOut,
  set<int> sChan) {return getProcess(idIn, idOut, sChan) != nullptr;}

//--------------------------------------------------------------------------

// Get pointer to matrix element, e.g. to check if process exists in
// library.

PY8MEs_namespace::PY8ME* ShowerMEsMadgraph::getProcess(
  vector<int> idIn, vector<int> idOut, set<int> sChan) {
    if (verbose >= VinciaConstants::DEBUG) {
      cout << " ShowerMEsMadgraph::getProcess(): checking for process";
      for (int i = 0; i < (int)idIn.size(); ++i) cout << " " << idIn[i];
      cout << " > ";
      for (int i = 0; i < (int)idOut.size(); ++i) cout << " " << idOut[i];
      cout << endl;
    }
    if (libPtr != nullptr && libPtr != 0)
      return libPtr->getProcess(idIn, idOut, sChan);
    cout << "      returning NULL" << endl;
    return nullptr;
}

//--------------------------------------------------------------------------

// Get the matrix element squared for a particle state.

double ShowerMEsMadgraph::me2Vincia(vector<Particle> state, int nIn) {

  // Prepare vector of incoming ID codes.
  if (nIn <= 0) return -1;
  else if (state.size() - nIn < 1) return -1;
  vector<int> idIn, helOrg, col, acol;
  vector<Vec4> momenta;
  idIn.push_back(state[0].id());
  momenta.push_back(state[0].p());
  helOrg.push_back(state[0].pol());
  col.push_back(state[0].col());
  acol.push_back(state[0].acol());
  if (nIn == 2) {
    idIn.push_back(state[1].id());
    momenta.push_back(state[1].p());
    helOrg.push_back(state[1].pol());
    col.push_back(state[1].col());
    acol.push_back(state[1].acol());
  }
  // Prepare vector of outgoing ID codes.
  vector<int> idOut;
  for (int i=nIn; i<(int)state.size(); ++i) {
    idOut.push_back(state[i].id());
    momenta.push_back(state[i].p());
    helOrg.push_back(state[i].pol());
    col.push_back(state[i].col());
    acol.push_back(state[i].acol());
  }
  // Currently not using the option to request specific s-channels.
  set<int> sChannels;

  // Access the process.
  PY8MEs_namespace::process_specifier proc_spec =
    libPtr->getProcessSpecifier(idIn, idOut, sChannels);
  PY8MEs_namespace::process_accessor proc_handle =
    libPtr->getProcess(proc_spec);

  // Return right away if unavailable.
  if (proc_handle.second.second < 0) return -1;

  // Convert momenta and colours to Madgraph format (energy first entry).
  vector< vector<double> > momentaMG5;
  vector< int > colacolMG5;
  for (int i = 0; i < (int)momenta.size(); ++i) {
    vector<double> pNow;
    pNow.push_back(momenta[i].e());
    pNow.push_back(momenta[i].px());
    pNow.push_back(momenta[i].py());
    pNow.push_back(momenta[i].pz());
    momentaMG5.push_back(pNow);
    colacolMG5.push_back(col[i]);
    colacolMG5.push_back(acol[i]);
  }

  vector<int> i9;
  // Check if we are doing a (partial) helicity sum.
  for (int i = 0; i < (int)helOrg.size(); ++i) {
    // Save indices of unpolarised partons.
    if (helOrg[i] == 9) i9.push_back(i);
  }

  // Explicitly sum over any hel = 9 helicities.
  vector< vector<int> > helConf;
  helConf.push_back(helOrg);
  while (i9.size() > 0) {
    int i  = i9.back();
    int id = (i < nIn) ? idIn[i] : idOut[i-nIn];
    // How many spin states.
    int nS = particleDataPtr->spinType(id);
    // Massless particles max have max 2 (physical) spin states.
    if (particleDataPtr->m0(id) == 0.0) nS=min(nS,2);
    // Create nS copies of helConf, one for each spin state.
    int helConfSizeNow = helConf.size();
    for (int iCopy = 1; iCopy <= nS; ++iCopy) {
      // Set hel for this particle in this copy.
      // Start from -1, then 1, then 0 (if 3 states).
      int h = -1;
      if (nS == 1) h = 0;
      else if (iCopy == 2) h = 1;
      else if (iCopy == 3) h = 0;
      else if (iCopy == 4) h = -2;
      else if (iCopy == 5) h = 2;
      for (int iHelConf=0; iHelConf<helConfSizeNow; ++iHelConf) {
        vector<int> helNow = helConf[iHelConf];
        helNow[i] = h;
        // First copy: use existing.
        if (iCopy == 1) helConf[iHelConf] = helNow;
        // Subsequent copies: create new.
        else helConf.push_back(helNow);
      }
    }
    // Remove the particle whose helicities have been summed over.
    i9.pop_back();
  }
  if (verbose >= VinciaConstants::DEBUG) {
    cout << " in = ";
    for (int i = 0; i < (int)idIn.size(); ++i) cout << idIn[i] << " ";
    cout << "   out = ";
    for (int i = 0; i < (int)idOut.size(); ++i) cout << idOut[i] << " ";
    cout << endl;
    cout << " number of helicity configurations = " << helConf.size() << endl;
    for (int i = 0; i < (int)helConf.size(); ++i) {
      cout << "   helConf " << i;
      for (int j = 0; j < (int)helConf[i].size(); ++j)
        cout << " " << helConf[i][j];
      cout << endl;
    }
  }

  // Set properties and return ME2 value.
  PY8MEs_namespace::PY8ME* proc_ptr = proc_handle.first;
  vector<int> perms = proc_handle.second.first;
  int proc_ID = proc_handle.second.second;
  proc_ptr->setMomenta(momentaMG5);
  proc_ptr->setProcID(proc_ID);
  proc_ptr->setPermutation(perms);
  proc_ptr->setColors(colacolMG5);

  // Compute helicity sum (and save helicity components if needed later).
  double me2 = 0.0;
  me2hel.clear();
  for (int iHC=0; iHC<(int)helConf.size(); ++iHC) {

    // Note. could check here if the helConf is physical (consistent
    // with spins of particles and conserving angular momentum).
    proc_ptr->setHelicities(helConf[iHC]);
    double me2now = proc_ptr->sigmaKin();
    // MG may produce inf/nan for unphysical hel combinations.
    if ( !isnan(me2now) && !isinf(me2now) ) {
      // Save helicity matrix element for possible later use.
      me2hel[helConf[iHC]] = me2now;
      // Add this helicity ME to me2
      me2 += me2now;
    }
  }
  me2 *= double(proc_ptr->getSymmetryFactor());
  return me2;

}

//--------------------------------------------------------------------------

bool ShowerMEsMadgraph::initDire(Info*, string card) {

  // Redirect output so that Dire can print MG5 initialization.
  std::streambuf *old = cout.rdbuf();
  stringstream ss;
  cout.rdbuf (ss.rdbuf());
  if (libPtr != nullptr) delete libPtr;
  libPtr = new PY8MEs_namespace::PY8MEs(card);
  libPtr->seProcessesIncludeSymmetryFactors(false);
  libPtr->seProcessesIncludeHelicityAveragingFactors(false);
  libPtr->seProcessesIncludeColorAveragingFactors(false);
  libPtr->setProcessesExternalMassesMode(1);
  // Restore print-out.
  cout.rdbuf (old);

  return true;

}

//--------------------------------------------------------------------------

// Check if a matrix element is available.

bool ShowerMEsMadgraph::isAvailableMEDire(vector <int> in, vector<int> out) {
  set<int> req_s_channels;
  PY8MEs_namespace::PY8ME * query
    = libPtr->getProcess(in, out, req_s_channels);
  return (query != 0);
}

//--------------------------------------------------------------------------

// Check if a matrix element is available.

bool ShowerMEsMadgraph::isAvailableMEDire(const Pythia8::Event& event) {

  vector <int> in, out;
  fillIds(event, in, out);
  set<int> req_s_channels;

  PY8MEs_namespace::PY8ME* query
    = libPtr->getProcess(in, out, req_s_channels);
  return (query != 0);
}

//--------------------------------------------------------------------------

// Calcuate the matrix element.

double ShowerMEsMadgraph::calcMEDire(const Pythia8::Event& event) {

  vector <int> in, out;
  fillIds(event, in, out);
  vector<int> cols;
  fillCols(event, cols);
  vector< vector<double> > pvec = fillMoms(event);
  set<int> req_s_channels;
  vector<int> helicities;

  bool success = true;
  pair < double, bool > res;
  try {
    res = libPtr->calculateME(in, out, pvec, req_s_channels, cols,
                               helicities);
  } catch (const std::exception& e) {
    success = false;
  }
  if (!success) return 0.0;
  if (res.second) {
    double me = res.first;
    PY8MEs_namespace::PY8ME* query
      = libPtr->getProcess(in, out, req_s_channels);
    me *= 1./query->getHelicityAveragingFactor();
    me *= 1./query->getColorAveragingFactor();
    return me;
  }
  // Done
  return 0.0;

}

//--------------------------------------------------------------------------

// Define external handles to the plugin for dynamic loading.

extern "C" {

  ShowerMEsMadgraph* newShowerMEs() {return new ShowerMEsMadgraph();}

  void deleteShowerMEs(ShowerMEsMadgraph* mes) {delete mes;}

}

//==========================================================================

} // end namespace Pythia8

#endif // end Pythia8_ShowerMEsMadgraph_H
