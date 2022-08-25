// ExternalMEsMadgraph.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Peter Skands, Stefan Prestel, Philip Ilten, Torbjorn
// Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the Madgraph parton shower ME plugin class which
// interfaces with matrix elements generated with the
// PY8Kernels/MG5MES plugin to MadGraph 5.

#ifndef Pythia8_ExternalMEsMadgraph_H
#define Pythia8_ExternalMEsMadgraph_H

// Include Pythia headers.
#include "Pythia8/ExternalMEs.h"

// Include Madgraph PY8MEs plugin headers.
#include "PY8ME.h"
#include "PY8MEs.h"

namespace Pythia8 {

//==========================================================================


class ExternalMEsMadgraph : public ExternalMEs {

public:

  // Constructor.
  ExternalMEsMadgraph() {isInitPtr = false; isInit = false;
    libPtr = nullptr; modelPtr = nullptr;}

  // Destructor.
  ~ExternalMEsMadgraph() {if (libPtr != nullptr) delete libPtr;
    if (modelPtr != nullptr) delete modelPtr;}

  // Initialisers.
  bool init() override;
  bool initVincia() override;
  bool initDire(Info*, string card) override;

  // Methods to check availability of matrix elements.
  bool isAvailable(vector<int> idIn, vector<int> idOut) override;
  bool isAvailable(const Pythia8::Event& event) override;
  bool isAvailable(const vector<Particle>& state) override;

  // Get the matrix element squared for a particle state.
  double calcME2(const vector<Particle>& state) override;
  double calcME2(const Event& event) override;

private:

  // Fill lists of IDs, momenta, colours, and helicities in MG5 format.
  void fillLists(const vector<Particle>& state, vector<int>& idsIn,
    vector<int>& idsOut, vector<int>& hels, vector<int>& cols,
    vector<vector<double>>& pn) const;

  // Calculate ME2 from pre-set lists.
  double calcME2(vector<int>& idIn, vector<int>& idOut,
    vector< vector<double> >& pn, vector<int>& hels, vector<int>& cols,
    set<int> sChannels = {});

  PY8MEs_namespace::PY8MEs* libPtr;
  PARS* modelPtr;

};

//--------------------------------------------------------------------------

// Initialise the Madgraph model, parameters, and couplings for use in Vincia.

bool ExternalMEsMadgraph::init() {return true;}

bool ExternalMEsMadgraph::initVincia() {

  // Check if pointers initialized.
  int verbose = settingsPtr->mode("Vincia:verbose");
  if (verbose > 1)
    cout << " (ExternalMEsMadgraph::init()) begin -------" << endl;
  if (!isInitPtr) {
    infoPtr->errorMsg("Error in ExternalMEsMadgraph::init:"
      " Cannot initialize, pointers not set.");
    return false;
  }
  isInit = true;

  // Create new model instance.
  if (verbose >= 2) cout << " (ExternalMEsMadgraph::init())"
    << "setting MG5 C++ masses, widths, couplings..." << endl;
  if (modelPtr != nullptr) delete modelPtr;
  modelPtr = new PARS();
  modelPtr->setIndependentParameters(particleDataPtr,coupSMPtr,slhaPtr);
  modelPtr->setIndependentCouplings();
  if (verbose >= 3) {
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
  if (verbose >= 3) cout << " (ExternalMEsMadgraph::init())"
    << " attempting to construct lib" << endl;
  if (libPtr != nullptr) delete libPtr;
  libPtr = new PY8MEs_namespace::PY8MEs(modelPtr);
  // Do not include averaging or symmetry factors in MG5.
  libPtr->seProcessesIncludeSymmetryFactors(false);
  libPtr->seProcessesIncludeHelicityAveragingFactors(false);
  libPtr->seProcessesIncludeColorAveragingFactors(false);
  // Set whether symmetry and averaging factors are applied in calcME2().
  inclSymFac    = false;
  inclHelAvgFac = true;
  inclColAvgFac = true;
  // Leading-colour colour-ordered amplitude only (can be reset later).
  colMode = 1;
  // Implicitly sum over helicities (can be reset later).
  helMode = 1;

  return true;

}

bool ExternalMEsMadgraph::initDire(Info*, string card) {

  // Redirect output so that Dire can print MG5 initialization.
  std::streambuf *old = cout.rdbuf();
  stringstream ss;
  cout.rdbuf (ss.rdbuf());
  if (libPtr != nullptr) delete libPtr;
  libPtr = new PY8MEs_namespace::PY8MEs(card);
  // Do not include averaging or symmetry factors in MG5.
  libPtr->seProcessesIncludeSymmetryFactors(false);
  libPtr->seProcessesIncludeHelicityAveragingFactors(false);
  libPtr->seProcessesIncludeColorAveragingFactors(false);
  libPtr->setProcessesExternalMassesMode(1);
  // Set whether symmetry and averaging factors are applied in calcME2().
  inclSymFac    = false;
  inclHelAvgFac = true;
  inclColAvgFac = true;
  // Leading-colour colour-ordered amplitude only (can be reset later).
  colMode = 1;
  // Implicitly sum over helicities (can be reset later).
  helMode = 1;
  // Restore print-out.
  cout.rdbuf (old);

  return true;

}

//--------------------------------------------------------------------------

// Check if a matrix element is available.

bool ExternalMEsMadgraph::isAvailable(vector<int> idIn, vector<int> idOut) {
  set<int> req_s_channels;
  PY8MEs_namespace::PY8ME * query
    = libPtr->getProcess(idIn, idOut, req_s_channels);
  return (query != nullptr);
}

bool ExternalMEsMadgraph::isAvailable(const Event& event) {

  vector <int> in, out;
  fillIds(event, in, out);
  set<int> req_s_channels;

  PY8MEs_namespace::PY8ME* query
    = libPtr->getProcess(in, out, req_s_channels);
  return (query != nullptr);
}

bool ExternalMEsMadgraph::isAvailable(const vector<Particle>& state) {

  vector <int> idIn, idOut;
  for (const Particle& ptcl : state) {
    if (ptcl.isFinal()) idOut.push_back(ptcl.id());
    else idIn.push_back(ptcl.id());
  }
  set<int> req_s_channels;

  PY8MEs_namespace::PY8ME* query
    = libPtr->getProcess(idIn, idOut, req_s_channels);
  return (query != nullptr);
}

//--------------------------------------------------------------------------

// Calcuate the squared matrix element.

double ExternalMEsMadgraph::calcME2(const vector<Particle>& state) {

  // Prepare lists.
  vector<int> idIn, idOut, hels, cols;
  vector<vector<double>> pn;
  fillLists(state, idIn, idOut, hels, cols, pn);
  int nIn = idIn.size();
  if (nIn <= 0) return -1.;
  else if (state.size() - nIn < 1) return -1.;

  return calcME2(idIn, idOut, pn, hels, cols);

}

double ExternalMEsMadgraph::calcME2(const Pythia8::Event& event) {

  // Prepare lists.
  vector<int> in, out;
  fillIds(event, in, out);
  vector<int> cols;
  fillCols(event, cols);
  vector< vector<double> > pvec = fillMoms(event);
  vector<int> helicities;

  return calcME2(in, out, pvec, helicities, cols);

}

double ExternalMEsMadgraph::calcME2(vector<int>& idIn, vector<int>& idOut,
  vector< vector<double> >& pn, vector<int>& hels, vector<int>& cols,
  set<int> sChannels) {

  // Access the process.
  PY8MEs_namespace::process_specifier proc_spec =
    libPtr->getProcessSpecifier(idIn, idOut, sChannels);
  PY8MEs_namespace::process_accessor proc_handle =
    libPtr->getProcess(proc_spec);
  // Return right away if unavailable.
  if (proc_handle.second.second < 0) return 0.0;
  PY8MEs_namespace::PY8ME* proc_ptr = proc_handle.first;
  proc_ptr->setPermutation(proc_handle.second.first);
  int procID = proc_handle.second.second;
  proc_ptr->setProcID(procID);
  int nIn = idIn.size();

  // Save current helicity configuration.
  vector< vector<int> > helConf;
  helConf.push_back(hels);
  // Check if we are doing an explicit helicity sum.
  vector<int> i9;
  if (helMode == 0) {
    // Save indices of unpolarised partons.
    for (int i(0); i<(int)hels.size(); ++i)
      if (hels[i] == 9) i9.push_back(i);
  }
  // Manually calculate helicity average.
  double helAvgNorm = i9.size()>0 ? 1:proc_ptr->getHelicityAveragingFactor();
  while (i9.size() > 0) {
    int i  = i9.back();
    int id = (i < nIn) ? idIn[i] : idOut[i-nIn];
    // How many spin states.
    int nS = particleDataPtr->spinType(id);
    // Massless particles have max 2 (physical) spin states.
    if (particleDataPtr->m0(id) == 0.0) nS=min(nS,2);
    if (i < nIn) helAvgNorm *= (double)nS;
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

  // Set colour configurations according to requested colour mode.
  vector<vector<int>> colConf;
  if (colMode >= 3) {
    // For FC sum use empty colour vector to communicate it with MG5.
    cols.clear();
    colConf.push_back(cols);
  } else if (colMode == 2) {
    // For LC sum, fetch all LC colour configurations.
    vector<vector<int>> colorConfigs = proc_ptr->getColorConfigs(procID);
    for (const auto& cc : colorConfigs) {
      int colID = proc_ptr->getColorIDForConfig(cc);
      if (proc_ptr->getColorFlowRelativeNCPower(colID) == 0)
        colConf.push_back(cc);
    }
  } else colConf.push_back(cols);

  // Compute sum over colours and helicities.
  // (Save helicity components if needed later).
  double me2 = 0.0;
  me2hel.clear();
  proc_ptr->setMomenta(pn);
  for (const auto& cc : colConf) {
    proc_ptr->setColors(cc);
    for (int iHC(0); iHC<(int)helConf.size(); ++iHC) {
      proc_ptr->setHelicities(helConf[iHC]);
      double me2now = proc_ptr->sigmaKin();
      // MG5 may produce inf/nan for unphysical hel combinations.
      if (!isnan(me2now) && !isinf(me2now)) {
        // Save helicity matrix element for possible later use.
        me2hel[helConf[iHC]] = me2now;
        me2 += me2now;
      }
    }
  }

  // Potentially include symmetry and averaging factors.
  if (inclSymFac) me2 /= proc_ptr->getSymmetryFactor();
  if (inclHelAvgFac) me2 /= helAvgNorm;
  if (inclColAvgFac) me2 /= proc_ptr->getColorAveragingFactor();
  return me2;

}

//--------------------------------------------------------------------------

// Fill lists.

void ExternalMEsMadgraph::fillLists(const vector<Particle>& state,
  vector<int>& idsIn, vector<int>& idsOut, vector<int>& hels,
  vector<int>& cols, vector<vector<double>>& pn) const {
  for (const Particle& ptcl : state) {
    if (ptcl.isFinal()) idsOut.push_back(ptcl.id());
    else idsIn.push_back(ptcl.id());
    vector<double> pNow = {ptcl.e(), ptcl.px(), ptcl.py(), ptcl.pz()};
    pn.push_back(pNow);
    hels.push_back(ptcl.pol());
    cols.push_back(ptcl.col());
    cols.push_back(ptcl.acol());
  }
}

//--------------------------------------------------------------------------

// Define external handles to the plugin for dynamic loading.

extern "C" {

  ExternalMEsMadgraph* newExternalMEs() {return new ExternalMEsMadgraph();}

  void deleteExternalMEs(ExternalMEsMadgraph* mes) {delete mes;}

}

//==========================================================================

} // end namespace Pythia8

#endif // end Pythia8_ExternalMEsMadgraph_H
