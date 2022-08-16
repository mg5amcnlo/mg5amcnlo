// VinciaEW.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the EW antenna-shower class and auxiliary
// classes. Main author is Rob Verheyen.

#ifndef Pythia8_VinciaEW_H
#define Pythia8_VinciaEW_H

// PYTHIA 8 headers.
#include "Pythia8/Event.h"
#include "Pythia8/StandardModel.h"
#include "Pythia8/PartonSystems.h"
#include "Pythia8/PythiaComplex.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/UserHooks.h"

// VINCIA functionality.
#include "Pythia8/VinciaCommon.h"
#include "Pythia8/VinciaQED.h"

namespace Pythia8 {

//==========================================================================

// Simple class to save information about particles.

class EWParticle {

 public:

  // Constructor.
  EWParticle() = default;
  EWParticle(double massIn, double widthIn, bool isResIn) :
    mass(massIn), width(widthIn), isRes(isResIn) {};

  // Members.
  double mass{0.}, width{0.};
  bool isRes{false};

};

//==========================================================================

// Locally saved particle data in glorified map.

class EWParticleData {

 public:

  // Find particle data.
  bool find(int id, int pol){
    return data.find(make_pair(id, pol)) != data.end();}

  // Add particle data.
  void add(int id, int pol, double massIn, double widthIn, bool isResIn) {
    data[make_pair(id, pol)] = EWParticle(massIn, widthIn, isResIn);}

  // Return particle mass.
  double mass(int id, int pol) {
    return find(id, pol) ? data[make_pair(id, pol)].mass : 0.;}

  // Return particle mass.
  double mass(int id) {
    // Every particle has either pol = 1 or pol = 0.
    if (find(id, 1)) return data[make_pair(id, 1)].mass;
    return find(id, 0) ? data[make_pair(id, 0)].mass : 0.;
  }

  // Return particle width.
  double width(int id, int pol) {
    return find(id, pol) ? data[make_pair(id, pol)].width : 0.;}

  // Return if particle is resonance.
  bool isRes(int id, int pol) {
    return find(id, pol) && data[make_pair(id, pol)].isRes;}

  // Return if particle is resonance.
  bool isRes(int id) {
    // Every particle has either pol = 1 or pol = 0.
    if (find(id, 1)) return data[make_pair(id, 1)].isRes;
    else if (find(id, 0)) return data[make_pair(id, 0)].isRes;
    else return false;
  }

  // Data access.
  EWParticle* get(int id, int pol) { return &data.at(make_pair(id, pol)); }
  unordered_map<pair<int, int>, EWParticle>::iterator begin() {
    return data.begin();}
  unordered_map<pair<int, int>, EWParticle>::iterator end() {
    return data.end();}

  // Member.
  unordered_map<pair<int,int>, EWParticle> data;

};

//==========================================================================

// Class to contain an antenna function and two outgoing polarizations.

class AntWrapper {

public:

  // Constructor.
  AntWrapper(double valIn, int poliIn, int poljIn):
    val(valIn), poli(poliIn), polj(poljIn) {}

  // Print.
  void print() {cout << "(" << poli << ", " << polj << ") " << val;}

  // Members.
  double val;
  int poli, polj;

};

//==========================================================================

// Class to contain an amplitude and two outgoing polarizations.

class AmpWrapper {

 public:

  // Constructor.
  AmpWrapper(complex valIn, int poliIn, int poljIn):
    val(valIn), poli(poliIn), polj(poljIn) {}

  // Normalization.
  AntWrapper norm() {return AntWrapper(std::norm(val), poli, polj);}

  // Operators.
  AmpWrapper& operator+=(complex c) {this->val += c; return *this;}
  AmpWrapper& operator*=(complex c) {this->val *= c; return *this;}
  void print() {cout << "(" << poli << ", " << polj << ") " << val;}

  // Members.
  complex val;
  int poli, polj;

};

//==========================================================================

// Calculator class for amplitudes, antennae, and Breit-Wigners.

class AmpCalculator {

public:

  // Initialize the pointers.
  void initPtr(Info* infoPtrIn, AlphaEM* alphaPtrIn,
    AlphaStrong* alphaSPtrIn) {
    infoPtr          = infoPtrIn;
    partonSystemsPtr = infoPtr->partonSystemsPtr;
    rndmPtr          = infoPtr->rndmPtr;
    settingsPtr      = infoPtr->settingsPtr;
    alphaPtr         = alphaPtrIn;
    alphaSPtr        = alphaSPtrIn;
    isInitPtr        = true;
  }

  // Initialize with maps.
  void init(EWParticleData* dataIn, unordered_map< pair<int, int>,
            vector<pair<int, int> > >* cluMapFinalIn,
            unordered_map< pair<int, int>, vector<pair<int, int> > >*
            cluMapInitialIn);

  // Set verbosity level.
  void setVerbose(int verboseIn) {verbose = verboseIn;}

  // Spinor products used for the calculation of the amplitudes.
  complex spinProd(int pol, const Vec4& ka, const Vec4& kb);
  complex spinProd(int pol, const Vec4& ka, const Vec4& pa, const Vec4& kb);
  complex spinProd(int pol, const Vec4& ka, const Vec4& pa, const Vec4& pb,
                   const Vec4& kb);
  complex spinProd(int pol, const Vec4& ka, const Vec4& pa, const Vec4& pb,
                   const Vec4& pc, const Vec4& kb);
  complex spinProd(int pol, const Vec4& ka, const Vec4& pa, const Vec4& pb,
                   const Vec4& pc, const Vec4& pd, const Vec4& kb);
  Vec4    spinProdFlat(string method, const Vec4& ka, const Vec4& pa);

  // Initialize couplings.
  void initCoup(bool va, int id1, int id2, int pol, bool m);

  // Initialize an FSR branching amplitude.
  void initFSRAmp(bool va, int id1, int id2, int pol,
    const Vec4& pi, const Vec4 &pj, const double& mMot, const double& widthQ2);

  // Check for zero denominator in an FSR amplitude.
  bool zdenFSRAmp(const string& method, const Vec4& pi, const Vec4& pj,
    bool check);

  // Initialize an ISR branching amplitude.
  void initISRAmp(bool va, int id1, int id2, int pol,
    const Vec4& pa, const Vec4 &pj, double& mA);

  // Check for zero denominator in an ISR amplitude.
  bool zdenISRAmp(const string& method, const Vec4& pa, const Vec4& pj,
    bool check);

  // Branching amplitude formalism.
  // Naming scheme: f = fermion, v = vector boson, h = higgs.

  // Final-state branching amplitudes.
  complex ftofvFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);
  complex ftofhFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);
  complex fbartofbarvFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);
  complex fbartofbarhFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);
  complex vTtoffbarFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);
  complex vTtovhFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);
  complex vTtovvFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);
  complex vLtoffbarFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);
  complex vLtovhFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);
  complex vLtovvFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);
  complex htoffbarFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);
  complex htovvFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);
  complex htohhFSRAmp(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj);

  // Initial-state branching amplitudes.
  complex ftofvISRAmp(const Vec4& pa, const Vec4& pj, int idA, int ida,
    int idj, double mA, int polA, int pola, int polj);
  complex ftofhISRAmp(const Vec4& pa, const Vec4& pj, int idA, int ida,
    int idj, double mA, int polA, int pola, int polj);
  complex fbartofbarvISRAmp(const Vec4& pa, const Vec4& pj, int idA, int ida,
    int idj, double mA, int polA, int pola, int polj);
  complex fbartofbarhISRAmp(const Vec4& pa, const Vec4& pj, int idA, int ida,
    int idj, double mA, int polA, int pola, int polj);

  // Branching amplitude selector.
  complex branchAmpFSR(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli=9, int polj=9);
  complex branchAmpISR(const Vec4& pa, const Vec4& pj, int idA, int ida,
    int idj, double mA, int polA, int pola=9, int polj=9);

  // Compute FF antenna function from amplitudes.
  double branchKernelFF(const Vec4& pi, const Vec4& pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot, int poli, int polj) {
    return norm(branchAmpFSR(pi, pj, idMot, idi, idj, mMot, widthQ2,
                             polMot, poli, polj));}

  // Compute FF antenna functions from amplitudes for all polarizations.
  vector<AntWrapper> branchKernelFF(Vec4 pi, Vec4 pj, int idMot, int idi,
    int idj, double mMot, double widthQ2, int polMot);

  // Compute II antenna function from amplitudes.
  double branchKernelII(Vec4 pa, Vec4 pj, int idA, int ida,
    int idj, double mA, int polA, int pola, int polj) {
    return norm(branchAmpISR(pa, pj, idA, ida, idj, mA, polA, pola, polj));}

  // Compute II antenna functions from amplitudes for all polarizations.
  vector<AntWrapper> branchKernelII(Vec4 pa, Vec4 pj, int idA, int ida,
    int idj, double mA, int polA);

  // Initialize an FF antenna function.
  void initFFAnt(bool va, int id1, int id2, int pol, const double& Q2,
    const double& widthQ2, const double& xi, const double& xj,
    const double& mMot, const double& miIn, const double& mjIn);

  // Report helicity combination error for an FF antenna function.
  void hmsgFFAnt(int polMot, int poli, int polj) {
    stringstream ss;
    ss << ": helicity combination was not found:\n    "
       << "polMot = " << polMot << " poli = " << poli << " polj = " << polj;
    infoPtr->errorMsg("Error in " + __METHOD_NAME__, ss.str());}

  // Initialize an II antenna function.
  void initIIAnt(int id1, int id2, int pol, const double& Q2,
    const double& xA, const double& xj,
    const double& mA, const double& maIn, const double& mjIn);

  // Report helicity combination error for an II antenna function.
  void hmsgIIAnt(int polA, int pola, int polj) {
    stringstream ss;
    ss << ": helicity combination was not found:\n    "
       << "polA = " << polA << " pola = " << pola << " polj = " << polj;
    infoPtr->errorMsg("Error in " + __METHOD_NAME__, ss.str());}

  // FF Antenna functions for branching process I (K) -> i j (k).
  // Q2 is the offshellness of I.
  // xi and xj are energy fractions of pi and pj in the collinear limit.
  double ftofvFFAnt(double Q2, double widthQ2, double xi, double xj,
    int idMot, int idi, int idj, double mMot, double miIn, double mjIn,
    int polMot, int poli, int polj);
  double ftofhFFAnt(double Q2, double widthQ2, double xi, double xj,
    int idMot, int idi, int idj, double mMot, double miIn, double mjIn,
    int polMot, int poli, int polj);
  double fbartofbarvFFAnt(double Q2, double widthQ2, double xi, double xj,
    int idMot, int idi, int idj, double mMot, double miIn, double mjIn,
    int polMot, int poli, int polj);
  double fbartofbarhFFAnt(double Q2, double widthQ2, double xi, double xj,
    int idMot, int idi, int idj, double mMot, double miIn, double mjIn,
    int polMot, int poli, int polj);
  double vtoffbarFFAnt(double Q2, double widthQ2, double xi, double xj,
    int idMot, int idi, int idj, double mMot, double miIn, double mjIn,
    int polMot, int poli, int polj);
  double vtovhFFAnt(double Q2, double widthQ2, double xi, double xj,
    int idMot, int idi, int idj, double mMot, double miIn, double mjIn,
    int polMot, int poli, int polj);
  double vtovvFFAnt(double Q2, double widthQ2, double xi, double xj,
    int idMot, int idi, int idj, double mMot, double miIn, double mjIn,
    int polMot, int poli, int polj);
  double htoffbarFFAnt(double Q2, double widthQ2, double xi, double xj,
    int idMot, int idi, int idj, double mMot, double miIn, double mjIn,
    int polMot, int poli, int polj);
  double htovvFFAnt(double Q2, double widthQ2, double xi, double xj,
    int idMot, int idi, int idj, double mMot, double miIn, double mjIn,
    int polMot, int poli, int polj);
  double htohhFFAnt(double Q2, double widthQ2, double xi, double xj,
    int idMot, int idi, int idj, double mMot, double miIn, double mjIn,
    int polMot, int poli, int polj);

  // II Antenna functions for branching process A (B) -> a j (b).
  // Q2 is the offshellness of A.
  // xA and xj are energy fractions of pA and pj in the collinear limit.
  double ftofvIIAnt(double Q2, double xA, double xj,
    int idA, int ida, int idj, double mA, double maIn, double mjIn,
    int polA, int pola, int polj);
  double fbartofbarvIIAnt(double Q2, double xA, double xj,
    int idA, int ida, int idj, double mA, double maIn, double mjIn,
    int polA, int pola, int polj);

  // FF antenna function calculator.
  double antFuncFF(double Q2, double widthQ2, double xi, double xj,
    int idMot, int idi, int idj, double mMot, double miIn, double mjIn,
    int polMot, int poli, int polj);

  // FF antenna function calculator for all outgoing polarizations.
  vector<AntWrapper> antFuncFF(double Q2, double widthQ2, double xi,
    double xj, int idMot, int idi, int idj, double mMot, double miIn,
    double mjIn, int polMot);

  // II antenna function calculator.
  double antFuncII(double Q2, double xA, double xj,
    int idA, int ida, int idj, double mA, double maIn, double mjIn,
    int polA, int pola, int polj);

  // II antenna function calculator for all outgoing polarizations.
  vector<AntWrapper> antFuncII(double Q2, double xA, double xj,
    int idA, int ida, int idj, double mA, double maIn, double mjIn, int polA);

  // Initialize an FSR splitting kernel.
  void initFSRSplit(bool va, int id1, int id2, int pol,
    const double& mMot, const double& miIn, const double& mjIn) {
    mi = miIn; mj = mjIn; mMot2 = pow2(mMot); mi2 = pow2(mi); mj2 = pow2(mj);
    initCoup(va, id1, id2, pol, true);}

  // Check for zero denominator in an FSR splitting kernel.
  bool zdenFSRSplit(const string& method, const double& Q2, const double& z,
    bool check);

  // Report helicty combination error for an FSR splitting kernel.
  void hmsgFSRSplit(int polMot, int poli, int polj) {
    stringstream ss;
    ss << ": helicity combination was not found:\n    "
       << "polMot = " << polMot << " poli = " << poli << " polj = " << polj;
    infoPtr->errorMsg("Error in " + __METHOD_NAME__, ss.str());}

  // Initialize an ISR splitting kernel.
  void initISRSplit(bool va, int id1, int id2, int pol,
    const double& mA, const double& maIn, const double& mjIn) {
    ma = maIn; mj = mjIn; mA2 = pow2(mA); ma2 = pow2(ma); mj2 = pow2(mj);
    initCoup(va, id1, id2, pol, mA > VinciaConstants::NANO);}

  // Check for zero denominator in an ISR splitting kernel.
  bool zdenISRSplit(const string& method, const double& Q2, const double& z,
    bool flip, bool check);

  // Report helicty combination error for an ISR splitting kernel.
  void hmsgISRSplit(int polA, int pola, int polj) {
    stringstream ss;
    ss << ": helicity combination was not found:\n    "
       << "polA = " << polA << " pola = " << pola << " polj = " << polj;
    infoPtr->errorMsg("Error in " + __METHOD_NAME__, ss.str());}

  // Final-state splitting kernels.
  double ftofvFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double ftofhFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double fbartofbarvFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double fbartofbarhFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double vTtoffbarFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double vTtovhFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double vTtovvFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double vLtoffbarFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double vLtovhFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double vLtovvFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double htoffbarFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double htovvFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double htohhFSRSplit(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);

  // Initial-state splitting kernels.
  double ftofvISRSplit(double Q2, double z, int idA, int ida, int idj,
    double mA, double maIn, double mjIn, int polA, int pola, int polj);
  double ftofhISRSplit(double Q2, double z, int idA, int ida, int idj,
    double mA, double maIn, double mjIn, int polA, int pola, int polj);
  double fbartofbarvISRSplit(double Q2, double z, int idA, int ida, int idj,
    double mA, double maIn, double mjIn, int polA, int pola, int polj);
  double fbartofbarhISRSplit(double Q2, double z, int idA, int ida, int idj,
    double mA, double maIn, double mjIn, int polA, int pola, int polj);

  // Splitting kernel caller.
  double splitFuncFSR(double Q2, double z, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj);
  double splitFuncISR(double Q2, double z, int idA, int ida, int idj,
    double mA, double maIn, double mjIn, int polA, int pola, int polj);

  // Compute partial decay width.
  double getPartialWidth(int idMot, int idi, int idj, double mMot, int polMot);

  // Compute total decay width.
  double getTotalWidth(int idMot, double mMot, int polMot);

  // Breit-Wigner calculators.
  double getBreitWigner(int id, double m, int pol);
  double getBreitWignerOverestimate(int id, double m, int pol);

  // Generate Breit-Wigner mass.
  double sampleMass(int id, int pol);

  // Bosonic interference factor.
  void applyBosonInterferenceFactor(Event &event, int XYEv, Vec4 pi, Vec4 pj,
    int idi, int idj, int poli, int polj);

  // Polarise a resonance decay.
  bool polarise(vector<Particle> &state);

  // EW event weight.
  double eventWeight() {return eventWeightSave;}
  void   eventWeight(double eventWeightIn) {eventWeightSave = eventWeightIn;}

  // Public data members.

  // EW data.
  EWParticleData* dataPtr{};

  // Maps of coupling constants.
  // TODO: read this from data file (rather than hard code).
  unordered_map<pair<int, int>, double> vMap, aMap, gMap, vCKM;

private:

  // Data members.

  // Constants used for Breit-Wigner sampling.
  unordered_map<int, vector<double> > cBW;
  // BW normalizations.
  unordered_map<pair<int,int>, double> nBW, np;

  // Event weight.
  double eventWeightSave{1};

  // Electroweak constants.
  double mw, mw2, sw, sw2;

  // Mode of matching to the Breit-Wigner.
  int bwMatchMode;

  // Maps of EW clusterings.
  unordered_map<pair<int, int>, vector<pair<int, int> > >* cluMapFinal{};
  unordered_map<pair<int, int>, vector<pair<int, int> > >* cluMapInitial{};

  // Vectors with spin assignments.
  vector<int> fermionPols, vectorPols, scalarPols;

  // Couplings.
  double v, a, vPls, vMin, g;

  // Masses and scale (FSR and ISR).
  double mMot2, mi, mi2, mj, mj2, mA2, ma, ma2, isrQ2;
  complex M, fsrQ2;

  // Reference vectors (FSR and ISR).
  Vec4 kij, ki, kj, pij, kaj, ka, paj;
  double wij, wi, wj, wij2, wi2, wj2, waj, wa, waj2, wa2;

  // Antenna function members.
  double Q4, Q4gam, Q2til, ant;

  // Pointers.
  Info* infoPtr{};
  PartonSystems* partonSystemsPtr{};
  Rndm* rndmPtr{};
  Settings* settingsPtr{};
  AlphaEM* alphaPtr{};
  AlphaStrong* alphaSPtr{};

  // Initializations.
  bool isInit{false};
  bool isInitPtr{false};
  int verbose{0};
};

//==========================================================================

// Class that contains an electroweak branching.

class EWBranching {

 public:

  // Constructor.
  EWBranching(int idMotIn, int idiIn, int idjIn, int polMotIn,
    double c0In = 0, double c1In = 0, double c2In = 0, double c3In = 0):
    idMot(idMotIn), idi(idiIn), idj(idjIn), polMot(polMotIn), c0(c0In),
    c1(c1In), c2(c2In), c3(c3In),
    isSplitToFermions(abs(idMot) > 20 && abs(idi) < 20 && abs(idj) < 20) {;}

  // Print.
  void print() {cout <<"    (" << idMot << ", " << polMot << ") -> " << idi <<
    "," << idj << ": (" << c0 << ", " << c1 << ", " << c2 << ", " << c3 <<
    ") \n";}

  // For a branching I->ij, particle IDs and polarisation of I.
  int idMot, idi, idj, polMot;
  // Overestimate constants used to sample antennae.
  double c0, c1, c2, c3;
  // Store if branching is a splitting.
  bool isSplitToFermions;

};

//==========================================================================

// Base class for an electroweak antenna.

class EWAntenna {

 public:

  // Constructor and destructor.
  EWAntenna(): iSys(-1), shat(0), doBosonInterference(false) {};
  virtual ~EWAntenna() = default;

  // Print, must be implemented by base classes.
  void print() {
    stringstream ss;
    ss << "Brancher = (" << iMot << ", " << polMot
       << "), Recoiler = " << iRec;
    printOut(__METHOD_NAME__, ss.str());
    for (int i = 0; i < (int)brVec.size(); i++) brVec[i].print();}

  // Initialize pointers.
  void initPtr(Info* infoPtrIn, VinciaCommon* vinComPtrIn, AlphaEM* alphaPtrIn,
    AmpCalculator* ampCalcPtrIn) {
    infoPtr          = infoPtrIn;
    rndmPtr          = infoPtr->rndmPtr;
    partonSystemsPtr = infoPtr->partonSystemsPtr;
    vinComPtr        = vinComPtrIn;
    alphaPtr         = alphaPtrIn;
    ampCalcPtr       = ampCalcPtrIn;
  };

  // Initialize, must be implemented by base classes.
  virtual bool init(Event &event, int iMotIn, int iRecIn, int iSysIn,
    vector<EWBranching> &branchings, Settings* settingsPtr) = 0;

  // Set verbosity level.
  void setVerbose(int verboseIn) {verbose = verboseIn;}

  // Generate a trial scale to be compared against other QCD and EW trials.
  virtual double generateTrial(double q2Start, double q2End,
    double alphaIn) = 0;

  // Accept/reject step to accept a trial.
  virtual bool acceptTrial(Event& event) = 0;

  // Update an event and parton system.
  virtual void updateEvent(Event& event) = 0;
  virtual void updatePartonSystems(Event& event);

  // Return index.
  int getIndexMot() {return iMot;};
  int getIndexRec() {return iRec;};

  // Check if splitting to fermions, inital, or resoance.
  bool isSplitToFermions() {
    return brTrial != nullptr && brTrial->isSplitToFermions;}
  virtual bool isInitial() {return false;}
  virtual bool isResonanceDecay() {return false;}

  // Select a channel.
  bool selectChannel(int idx, const double& cSum, const map<double, int>&
    cSumSoFar, int& idi, int& idj, double& mi2, double& mj2);

protected:

  // Indices, PID, and polarization of I, K in Pythia event record.
  int iMot, iRec, idMot, idRec, polMot;
  // Momenta of antenna constituents.
  Vec4 pMot, pRec;
  // Masses and invariants of antenna constituents.
  double sAnt, mMot, mMot2, mRec, mRec2;
  // Overestimate of QED coupling.
  double alpha;
  // Parton system this antenna is part of.
  int iSys;

  // EW branchings.
  vector<EWBranching> brVec;

  // Trial variables.
  bool hasTrial;
  double q2Trial, sijTrial, sjkTrial;
  int poliTrial, poljTrial;

  // Outgoing momenta after branching.
  vector<Vec4> pNew;

  // Info on coefficents.
  double c0Sum, c1Sum, c2Sum, c3Sum;
  map<double, int> c0SumSoFar, c1SumSoFar, c2SumSoFar, c3SumSoFar;

  // Matching scale.
  double q2Match;

  // Information for partonSystems.
  int jNew;
  unordered_map<int,int> iReplace;
  double shat;

  // Pointers.
  EWBranching* brTrial{};
  Info* infoPtr{};
  Rndm* rndmPtr{};
  PartonSystems* partonSystemsPtr{};
  VinciaCommon* vinComPtr{};
  AlphaEM* alphaPtr{};
  AmpCalculator* ampCalcPtr{};

  // Settings.
  bool doBosonInterference;

  // Verbosity.
  int verbose;

};

//==========================================================================

// Final-final electroweak antenna.

class EWAntennaFF : public EWAntenna {
 public:

  // Overridden virtual functions.
  virtual bool init(Event &event, int iMotIn, int iRecIn, int iSysIn,
    vector<EWBranching> &branchings, Settings* settingsPtr) override;
  virtual double generateTrial(double q2Start, double q2End, double alphaIn)
    override;
  virtual bool acceptTrial(Event &event) override;
  virtual void updateEvent(Event &event) override;

 private:

  // Data members.
  double mAnt2, sqrtKallen;
  // Kinematic map.
  int kMapFinal;
  // Controls if resonances with too large offshellness are vetoed.
  bool vetoResonanceProduction;

};

//==========================================================================

// Final-final electroweak resonance antenna.

class EWAntennaFFres : public EWAntennaFF {

public:

  // Overridden virtual functions.
  bool init(Event &event, int iMotIn, int iRecIn, int iSysIn,
    vector<EWBranching> &branchings, Settings* settingsPtr) override;
  bool isResonanceDecay() override {return true;}
  double generateTrial(double q2Start, double q2End, double alphaIn) override;
  bool acceptTrial(Event &event) override;
  void updateEvent(Event &event) override;

  // Generate the kinematics and channel for decays below matching scale.
  bool genForceDecay(Event &event);

private:

  // Check if the trial was a resonance decay.
  bool trialIsResDecay;
  // Matching mode.
  int bwMatchMode;
  // Offshellness of the resonance and EW scale.
  double q2Dec, q2EW;
  // Switch for the special case of a resonance without a recoiler.
  bool doDecayOnly{false};

};

//==========================================================================

// Initial-initial electroweak antenna.

class EWAntennaII : public EWAntenna {

public:

  // Constructor.
  EWAntennaII(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn):
    beamAPtr(beamAPtrIn), beamBPtr(beamBPtrIn), shh(0), xMot(0), xRec(0),
      vetoResonanceProduction(false), TINYPDFtrial(1e-10) {;}

  // Overridden virtual functions.
  bool init(Event &event, int iMotIn, int iRecIn, int iSysIn,
    vector<EWBranching> &branchings,Settings* settingsPtr) override;
  bool isInitial() override {return true;}
  double generateTrial(double q2Start, double q2End, double alphaIn) override;
  bool acceptTrial(Event &event) override;
  void updateEvent(Event &event) override;
  void updatePartonSystems(Event &event) override;

private:

  // Members.
  // Beam pointers.
  BeamParticle* beamAPtr{};
  BeamParticle* beamBPtr{};
  // Hadronic invariant mass.
  double shh;
  // Antenna hadronic momentum fractions.
  double xMot, xRec;
  // Controls if resonances with too large offshellness are vetoed.
  bool vetoResonanceProduction;
  // Global recoil momenta.
  vector<Vec4> pRecVec;
  vector<int>  iRecVec;

  // Tolerance for PDFs.
  double TINYPDFtrial;

};

//==========================================================================

// Class that performs electroweak showers in a single parton system.

class EWSystem {

public:

  // Constructors.
  EWSystem(): antTrial(nullptr) {clearLastTrial();}
  EWSystem(unordered_map<pair<int, int>, vector<EWBranching> >* brMapFinalIn,
    unordered_map<pair<int, int>, vector<EWBranching> >* brMapInitialIn,
    unordered_map<pair<int, int>, vector<EWBranching> >* brMapResonanceIn,
    unordered_map<pair<int, int>, vector<pair<int, int> > >* cluMapFinalIn,
    unordered_map<pair<int, int>, vector<pair<int, int> > >* cluMapInitialIn,
    AmpCalculator * ampCalcIn):
    shh(0), iSysSav(0), resDecOnlySav(false), q2Cut(0), q2Trial(0),
    lastWasSplitSav(false), lastWasDecSav(false), lastWasInitialSav(false),
    lastWasBelowCut(false), ISav(0), KSav(0), brMapFinal(brMapFinalIn),
    brMapInitial(brMapInitialIn), brMapResonance(brMapResonanceIn),
    cluMapFinal(cluMapFinalIn), cluMapInitial(cluMapInitialIn),
    ampCalcPtr(ampCalcIn), isInit(false), doVetoHardEmissions(false),
    verbose(false), vetoHardEmissionsDeltaR2(0) {clearLastTrial();}

  // Initialize pointers.
  void initPtr(Info* infoPtrIn, VinciaCommon* vinComPtrIn, AlphaEM* alIn) {
    infoPtr = infoPtrIn;
    partonSystemsPtr = infoPtr->partonSystemsPtr;
    rndmPtr          = infoPtr->rndmPtr;
    settingsPtr      = infoPtr->settingsPtr;
    vinComPtr = vinComPtrIn;
    al = alIn;}

  // Initialize.
  void init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn)  {
  beamAPtr = beamAPtrIn; beamBPtr = beamBPtrIn;
  doVetoHardEmissions = settingsPtr->flag("Vincia:EWoverlapVeto");
  vetoHardEmissionsDeltaR2 =
    pow2(settingsPtr->parm("Vincia:EWoverlapVetoDeltaR"));
  isInit = true;}

  // Set verbosity.
  void setVerbose(int verboseIn) {verbose = verboseIn;}

  // Prepare an event.
  bool prepare(Event &event, int iSysIn, double q2CutIn, bool resDecOnlyIn) {
    iSysSav = iSysIn; resDecOnlySav = resDecOnlyIn; q2Cut = q2CutIn;
    shh = infoPtr->s(); return buildSystem(event);}

  // Build a system.
  bool buildSystem(Event &event);

  // Return which system we are handling.
  int system() {return iSysSav;}

  // Generate the next Q2.
  double q2Next(double q2Start, double q2End);

  // Add an antenna.
  template <class T> void addAntenna(T ant, vector<T>& antVec,
    Event &event, int iMot, int iRec,
    unordered_map<pair<int, int>, vector<EWBranching> >* brMapPtr) {
    if (iMot == 0) return; // Check mother.
    int idA(event[iMot].id()), polA(event[iMot].pol());
    if (idA == 21) return; // Skip gluons.
    auto it = brMapPtr->find(make_pair(idA, polA));
    if (it != brMapPtr->end()) {
      // Found. Pass verbosity.
      ant.setVerbose(verbose);
      // Pass pointers.
      ant.initPtr(infoPtr, vinComPtr, al, ampCalcPtr);
      // Initialise and if success, store.
      if (ant.init(event, iMot, iRec, iSysSav, it->second, settingsPtr)) {
        antVec.push_back(move(ant));
        if (verbose >= VinciaConstants::DEBUG) {
          stringstream ss;
          ss << "Added EW antenna with iEv = "
             << iMot << " and iRec = "<< iRec<< " in system "<< iSysSav;
          printOut(__METHOD_NAME__, ss.str());}}}}

  // Generate a trial.
  template <class T> void generateTrial(vector<T> & antVec, double q2Start,
    double q2End, double alphaIn) {
    if (q2End > q2Start) return;
    for (int i = 0; i < (int)antVec.size(); i++) {
      // Generate a trial scale for the current antenna.
      double q2New = antVec[i].generateTrial(q2Start,q2End,alphaIn);
      // Current winner.
      if (q2New > q2Trial && q2New > q2End) {
        // Save trial information.
        q2Trial = q2New;
        antTrial = &(antVec[i]);
        lastWasDecSav = antTrial->isResonanceDecay();
        lastWasInitialSav = antTrial->isInitial();
        // This is done to avoid issues with resonance antennae
        // not deciding their channel until acceptTrial.
        lastWasSplitSav = lastWasDecSav ? true : antTrial->isSplitToFermions();
        lastWasBelowCut = (q2Trial < q2Cut)? true : false;
        ISav = antTrial->getIndexMot(); KSav = antTrial->getIndexRec();}}}

  // Overloaded version passing event, just for resonance decays.
  void generateTrial(Event &event, vector<EWAntenna> & antVec,double q2Start,
    double q2End, double alphaIn);

  // Accept a trial.
  bool acceptTrial(Event &event) {
    bool passed = antTrial->acceptTrial(event);
    if (verbose >= VinciaConstants::DEBUG)
      printOut(__METHOD_NAME__, passed ? "Passed veto" : "Vetoed branching");
    return passed;}

  // Update an event.
  void updateEvent(Event &event) {
    if (verbose >= VinciaConstants::DEBUG)
      printOut(__METHOD_NAME__, "begin", VinciaConstants::dashLen);
    if (antTrial != nullptr) antTrial->updateEvent(event);
    else infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": trial doesn't exist!");
    if (verbose >= VinciaConstants::DEBUG)
      printOut(__METHOD_NAME__, "end", VinciaConstants::dashLen);}

  // Update parton systems.
  void updatePartonSystems(Event &event) {
    if (verbose >= VinciaConstants::DEBUG)
      printOut(__METHOD_NAME__, "begin", VinciaConstants::dashLen);
    if (antTrial!=nullptr) antTrial->updatePartonSystems(event);
    else infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": trial doesn't exist!");
    if (verbose >= VinciaConstants::DEBUG)
      printOut(__METHOD_NAME__, "end", VinciaConstants::dashLen);}

  // Print the antennas.
  void printAntennae() {
    for (int i = 0; i < (int)antVecFinal.size(); i++) antVecFinal[i].print();
    for (int i = 0; i < (int)antVecRes.size(); i++) antVecRes[i].print();
    for (int i = 0; i < (int)antVecInitial.size(); i++)
      antVecInitial[i].print();}

  // Get number of branchers (total and res-dec only)
  unsigned int nBranchers() {return antVecFinal.size() +
      antVecInitial.size() + antVecRes.size();}
  unsigned int nResDec() {return antVecRes.size();}

  // Check if system has a trial pointer.
  bool hasTrial() {return antTrial != nullptr;}

  // Check if split to fermions, initial, or resonance.
  bool lastWasSplitToFermions() {return lastWasSplitSav;}
  bool lastWasInitial() {return lastWasInitialSav;}
  bool lastWasResonanceDecay() {return lastWasDecSav;}

  // Clear the antennae.
  void clearAntennae() {antVecFinal.clear(); antVecInitial.clear();
    antVecRes.clear(); clearLastTrial();}

  // Clear the last saved trial (but keep the antennae).
  void clearLastTrial() {
    q2Trial = 0; antTrial = nullptr; lastWasSplitSav = false;
    lastWasDecSav = false; lastWasInitialSav = false; lastWasBelowCut = false;
    ISav = 0; KSav = 0;}

private:

  // Members.
  // Hadronic invariant mass.
  double shh;

  // System and whether to do only resonance decays or full EW shower.
  int iSysSav;
  bool resDecOnlySav;

  // Cutoff scale.
  double q2Cut;

  // Pointers.
  BeamParticle* beamAPtr{}, *beamBPtr{};
  Info* infoPtr{};
  PartonSystems* partonSystemsPtr{};
  Rndm* rndmPtr{};
  Settings* settingsPtr{};
  VinciaCommon* vinComPtr{};
  AlphaEM* al{};

  // Antennae.
  vector<EWAntennaFF> antVecFinal;
  vector<EWAntennaII> antVecInitial;
  vector<EWAntennaFFres> antVecRes;

  // Trial information.
  EWAntenna* antTrial{};
  double q2Trial;
  bool lastWasSplitSav, lastWasDecSav, lastWasInitialSav, lastWasBelowCut;
  int ISav, KSav;

  // Map of branchings.
  unordered_map<pair<int, int>, vector<EWBranching> >
    *brMapFinal{}, *brMapInitial{}, *brMapResonance{};

  // Cluster maps for spectator selection.
  unordered_map<pair<int, int>, vector<pair<int, int> > >
    *cluMapFinal{}, *cluMapInitial{};

  // Amplitude calculator.
  AmpCalculator* ampCalcPtr{};

  // Flags.
  bool isInit, doVetoHardEmissions;
  int verbose;
  double vetoHardEmissionsDeltaR2;

};

//==========================================================================

// Top-level class for the electroweak shower module.

class VinciaEW : public VinciaModule {

public:

  // Constructor.
  VinciaEW(): isLoaded{false} {;}

  // Initialize pointers (called at construction time).
  void initPtr(Info* infoPtrIn, VinciaCommon* vinComPtrIn) override {
    infoPtr = infoPtrIn;
    particleDataPtr  = infoPtr->particleDataPtr;
    partonSystemsPtr = infoPtr->partonSystemsPtr;
    rndmPtr          = infoPtr->rndmPtr;
    settingsPtr      = infoPtr->settingsPtr;
    vinComPtr = vinComPtrIn;
    ampCalc.initPtr(infoPtr, &al, &vinComPtr->alphaStrong);
    isInitPtr = true;}

  // Initialise settings for current run (called as part of Pythia::init()).
  void init(BeamParticle* beamAPtrIn = 0, BeamParticle* beamBPtrIn = 0)
    override;

  // Select helicities for a resonance-decay system.
  bool polarise(vector<Particle> &state) override {
    if (isLoaded) return ampCalc.polarise(state);
    else return false;}

  // Prepare to shower a system.
  // (If isBelowHadIn = true, assume only resonance decays may be left to do.)
  bool prepare(int iSysIn, Event &event, bool isBelowHadIn=false) override;

  // Update EW shower system each time something has changed.
  void update(Event &event, int iSysIn) override {
    if (verbose >= VinciaConstants::DEBUG)
      printOut(__METHOD_NAME__, "begin", VinciaConstants::dashLen);
    if (iSysIn != ewSystem.system()) return;
    else ewSystem.buildSystem(event);
    if (verbose >= VinciaConstants::DEBUG)
      printOut(__METHOD_NAME__, "end", VinciaConstants::dashLen);}

  // Set verbose level.
  void setVerbose(int verboseIn) override {
    verbose = verboseIn; ampCalc.setVerbose(verboseIn);}

  // Save information on masses and widths, and load branchings.
  void load() override;

  // Generate a trial scale.
  double q2Next(Event&, double q2Start,double q2End) override;

  // Query last branching.
  bool lastIsSplitting() override { return ewSystem.lastWasSplitToFermions(); }
  bool lastIsInitial() override { return ewSystem.lastWasInitial(); }
  bool lastIsResonanceDecay() override {
    return ewSystem.lastWasResonanceDecay(); }

  // Check veto.
  bool acceptTrial(Event& event) override {
    if (verbose >= VinciaConstants::DEBUG)
      printOut(__METHOD_NAME__, "begin", VinciaConstants::dashLen);
    bool success = false;
    if (ewSystem.hasTrial()) success = ewSystem.acceptTrial(event);
    else infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": trial doesn't exist!");
    if (verbose >= VinciaConstants::DEBUG)
      printOut(__METHOD_NAME__, "end", VinciaConstants::dashLen);
    return success;}

  // Update event after branching accepted.
  void updateEvent(Event& event) override {
    if (verbose >= VinciaConstants::DEBUG)
      printOut(__METHOD_NAME__, "begin", VinciaConstants::dashLen);
    if (ewSystem.hasTrial()) ewSystem.updateEvent(event);
    else infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": trial doesn't exist!");
    if (verbose >=VinciaConstants::DEBUG) {
      printOut(__METHOD_NAME__,"Event after update:"); event.list();
      printOut(__METHOD_NAME__, "end", VinciaConstants::dashLen);}}

  // Update partonSystems after branching accepted.
  void updatePartonSystems(Event& event) override {
    if (verbose >= VinciaConstants::DEBUG)
      printOut(__METHOD_NAME__, "begin", VinciaConstants::dashLen);
    if (ewSystem.hasTrial()) ewSystem.updatePartonSystems(event);
    else infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": trial doesn't exist!");
    if (verbose >= VinciaConstants::DEBUG)
      printOut(__METHOD_NAME__, "end", VinciaConstants::dashLen);}

  // Clear EW system.
  void clear(int) override {
    ewSystem.clearAntennae();
    ampCalc.eventWeight(1);
  }

  // Get number of EW branchers.
  unsigned int nBranchers() override {return ewSystem.nBranchers();}
  unsigned int nResDec() override {return ewSystem.nResDec();}

  // Print loaded data on masses and widths.
  void printData();

  // Print branchings.
  void printBranchings();

  // Override additional base class methods.
  double q2min() override {return q2minSav;}
  double q2minColoured() override {return q2minSav;}
  double eventWeight() {return ampCalc.eventWeight();}

  // Setter for q2minSav
  void q2min(double q2minSavIn) {q2minSav = q2minSavIn;}

  // We only have one system.
  int sysWin() override {return ewSystem.system();}

  // Check if a particle is a resonance according to EW shower database.
  bool isResonance(int pid) {return ewData.isRes(pid);}

  // Public members.

  // Cluster maps for spectator selection.
  unordered_map<pair<int, int>, vector<pair<int, int> > >
    cluMapFinal, cluMapInitial;

  // Map from (PID, polarisation) of I to all possible branchings where I is:
  // final, initial or resonance.
  unordered_map<pair<int, int>, vector<EWBranching> >
    brMapFinal, brMapInitial, brMapResonance;

  // Locally store data about EW particles.
  EWParticleData ewData;

  // Amplitude calculator.
  AmpCalculator ampCalc;

private:

  // Read in data for overestimate function coefficients for EW branchings
  // from XML and store.
  bool readFile(string file);
  bool readLine(string line);
  bool attributeValue(string line, string attribute, string& val);
  template <class T> bool attributeValue(
    string line, string attribute, T& val) {
    string valString;
    if (!attributeValue(line, attribute, valString)) return false;
    istringstream valStream(valString);
    if ( !(valStream >> val) ) {
      infoPtr->errorMsg("Error in " + __METHOD_NAME__,
        ": failed to store attribute " + attribute + " " + valString);
      return false;
    } else return true;
  }
  bool addBranching(string line, unordered_map< pair<int, int>,
    vector<EWBranching> > & branchings, unordered_map< pair<int, int>,
    vector<pair<int, int> > > & clusterings, double headroom, bool decay);
  bool addParticle(int idIn, int polIn, bool isRes);

  // Members.

  // Cutoff scale.
  double q2minSav;

  // AlphaEM.
  AlphaEM al;

  // System.
  EWSystem ewSystem;

  // Trial information.
  double q2Trial;

  // Switches.
  bool isLoaded, doEW, doFFbranchings, doIIbranchings, doRFbranchings,
    doBosonInterference;

  // Massless quark flavors.
  int nFlavZeroMass;

  // Overestimate headroom.
  double headroomFinal, headroomInitial;

};

//==========================================================================

// Class to do the veto for overlapping QCD/EW shower phase space.

class VinciaEWVetoHook : public UserHooks {

 public:

  // Constructor.
  VinciaEWVetoHook() = default;

  // Define to activate veto.
  bool canVetoISREmission() override {return mayVeto;}
  bool canVetoFSREmission() override {return mayVeto;}

  // Methods to veto EW/QCD emissions.
  bool doVetoISREmission(int sizeOld, const Event &event, int iSys) override;
  bool doVetoFSREmission(int sizeOld, const Event &event, int iSys,
    bool inResonance = false) override;

  // Initialize.
  void init(shared_ptr<VinciaEW> ewShowerPtrIn);

 private:

  // Universal method called by doVeto(FSR/ISR)Emission.
  bool doVetoEmission(int sizeOld, const Event &event, int iSys);
  // Find all QCD clusterings and evaluate their kt measure.
  double findQCDScale(int sizeOld, const Event& event, int iSys);
  // Find all EW clusterings and evaluate their kt measure.
  double findEWScale(int sizeOld, const Event& event, int iSys);
  // Evaluate Durham kt measure for two particles i and j.
  double ktMeasure(const Event& event, int indexi, int indexj, double mI2);
  // Evaluate a QCD clustering - returns -1 if not a valid clustering.
  double findktQCD(const Event& event, int indexi, int indexj);
  // Evaluate an EW clustering - returns -1 if not a valid clustering.
  double findktEW(const Event& event, int indexi, int indexj);
  // Look up last FSR emission info from event record.
  bool setLastFSREmission(int sizeOld, const Event& event);
  // Look up last ISR emission info from event record.
  bool setLastISREmission(int sizeOld, const Event& event);

  // Members.

  // Verbosity.
  int verbose;

  // Flag to control if vetoing occurs.
  bool mayVeto{true};

  // delta R parameter used in kt measure.
  double deltaR;

  // Electroweak scale
  double q2EW;

  // Last emission info.
  bool lastIsQCD;
  double lastkT2;

  // Pointer to the EW shower
  shared_ptr<VinciaEW> ewShowerPtr{};

};


//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_VinciaEW_H
