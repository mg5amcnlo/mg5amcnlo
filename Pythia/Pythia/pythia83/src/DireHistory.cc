// DireHistory.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for Dire history classes.

#include "Pythia8/DireHistory.h"
#include "Pythia8/DireSpace.h"
#include "Pythia8/DireTimes.h"

namespace Pythia8 {

//==========================================================================

string stringFlavs(const Event& event) {
  ostringstream os;
  os << " (";
  for (int i=0; i < event.size(); ++i)
    if (event[i].status() == -21) os << " " << event[i].id();
  os << " ) -->> (";
  for (int i=0; i < event.size(); ++i) {
    if (event[i].status() ==  23) os << " " << event[i].id();
    if (event[i].status() ==  22) os << " " << event[i].id();
  }
  os << " ) ";
  return os.str();
}

void listFlavs(const Event& event, bool includeEndl = false) {
  cout << std::left << setw(30) << stringFlavs(event);
  if (includeEndl) cout << endl;
}

//--------------------------------------------------------------------------

// Helper class for setEffectiveScales.

class DireCouplFunction : public DireFunction {

public:

  DireCouplFunction(AlphaStrong* asIn) :
    as(asIn), aem(nullptr), asPow(1), aemPow(1) {};
  DireCouplFunction(AlphaEM* aemIn) :
    as(nullptr), aem(aemIn), asPow(1), aemPow(1) {};
  DireCouplFunction(AlphaStrong* asIn, int asPowIn,
    AlphaEM* aemIn, int aemPowIn) : as(asIn), aem(aemIn), asPow(asPowIn),
    aemPow(aemPowIn) {};
  double f(double x) {
    double ret = (as==nullptr)  ? 1. : pow(as->alphaS(x),asPow);
    ret       *= (aem==nullptr) ? 1. : pow(aem->alphaEM(x),aemPow);
    return ret;
  }
  double f(double x, double) {
    double ret = (as==nullptr)  ? 1. : pow(as->alphaS(x),asPow);
    ret       *= (aem==nullptr) ? 1. : pow(aem->alphaEM(x),aemPow);
    return ret;
  }
  double f(double x, vector<double>) {
    double ret = (as==nullptr)  ? 1. : pow(as->alphaS(x),asPow);
    ret       *= (aem==nullptr) ? 1. : pow(aem->alphaEM(x),aemPow);
    return ret;
  }
  AlphaStrong* as;
  AlphaEM*     aem;
  int asPow, aemPow;

};

//==========================================================================

// The Clustering class.

//--------------------------------------------------------------------------

// Declaration of Clustering class
// This class holds information about one radiator, recoiler,
// emitted system.
// This class is a container class for DireHistory class use.

// print for debug
void DireClustering::list() const {
  cout << " emt " << emitted
       << " rad " << emittor
       << " rec " << recoiler
       << " partner " << partner
       << " pTscale " << pTscale
       << " name " << name()  << endl;
}

//==========================================================================

// The DireHistory class.

// A DireHistory object represents an event in a given step in the CKKW-L
// clustering procedure. It defines a tree-like recursive structure,
// where the root node represents the state with n jets as given by
// the matrix element generator, and is characterized by the member
// variable mother being null. The leaves on the tree corresponds to a
// fully clustered paths where the original n-jets has been clustered
// down to the Born-level state. Also states which cannot be clustered
// down to the Born-level are possible - these will be called
// incomplete. The leaves are characterized by the vector of children
// being empty.

//--------------------------------------------------------------------------

// Number of trial emission to use for calculating the average number of
// emissions
const int DireHistory::NTRIAL = 1;

const double DireHistory::MCWINDOW = 0.1;
const double DireHistory::MBWINDOW = 0.1;
const double DireHistory::PROBMAXFAC = 0.0;

//--------------------------------------------------------------------------

// Declaration of DireHistory class
// The only constructor. Default arguments are used when creating
// the initial history node. The depth is the maximum number of
// clusterings requested. scalein is the scale at which the \a
// statein was clustered (should be set to the merging scale for the
// initial history node. beamAIn and beamBIn are needed to
// calcutate PDF ratios, particleDataIn to have access to the
// correct masses of particles. If isOrdered is true, the previous
// clusterings has been ordered. is the PDF ratio for this
// clustering (=1 for FSR clusterings). probin is the accumulated
// probabilities for the previous clusterings, and \ mothin is the
// previous history node (null for the initial node).

DireHistory::DireHistory( int depthIn,
         double scalein,
         Event statein,
         DireClustering c,
         MergingHooksPtr mergingHooksPtrIn,
         BeamParticle beamAIn,
         BeamParticle beamBIn,
         ParticleData* particleDataPtrIn,
         Info* infoPtrIn,
         PartonLevel* showersIn,
         shared_ptr<DireTimes> fsrIn,
         shared_ptr<DireSpace> isrIn,
         DireWeightContainer* psweightsIn,
         CoupSM* coupSMPtrIn,
         bool isOrdered = true,
         bool isAllowed = true,
         double clusterProbIn = 1.0,
         double clusterCouplIn = 1.0,
         double prodOfProbsIn = 1.0,
         double prodOfProbsFullIn = 1.0,
         DireHistory * mothin = 0)
    : state(statein),
      generation(depthIn),
      mother(mothin),
      selectedChild(-1),
      sumpath(0.0),
      sumGoodBranches(0.0),
      sumBadBranches(0.0),
      foundOrderedPath(false),
      foundAllowedPath(false),
      foundCompletePath(false),
      foundOrderedChildren(true),
      scale(scalein),
      scaleEffective(0.),
      couplEffective(1.),
      clusterProb(clusterProbIn),
      clusterCoupl(clusterCouplIn),
      prodOfProbs(prodOfProbsIn),
      prodOfProbsFull(prodOfProbsFullIn),
      clusterIn(c),
      iReclusteredOld(0),
      doInclude(true),
      hasMEweight(false),
      MECnum(1.0),
      MECden(1.0),
      MECcontrib(1.0),
      mergingHooksPtr(mergingHooksPtrIn),
      beamA(beamAIn),
      beamB(beamBIn),
      particleDataPtr(particleDataPtrIn),
      infoPtr(infoPtrIn),
      showers(showersIn),
      fsr(fsrIn),
      isr(isrIn),
      coupSMPtr(coupSMPtrIn),
      psweights(psweightsIn),
      doSingleLegSudakovs(
        infoPtr->settingsPtr->flag("Dire:doSingleLegSudakovs")),
      probMaxSave(-1.),
      depth(depthIn),
      minDepthSave(-1) {

  fsr->direInfoPtr->message(1)
    << scientific << setprecision(15)
    << __FILE__ << " " << __func__
    << " " << __LINE__ << " : New history node "
    << stringFlavs(state) << " at "
    << clusterIn.name() << " " << clusterIn.pT() << " "
    << (clusterIn.radSave? clusterIn.radSave->id() : 0) << " "
    << (clusterIn.emtSave? clusterIn.emtSave->id() : 0) << " "
    << (clusterIn.recSave? clusterIn.recSave->id() : 0) << " "
    << clusterProb << " "
    << clusterCoupl << " "
    << prodOfProbs << " "
    << "\t\t bare prob = " << clusterProb*pow2(clusterIn.pT())
    << " pT = " << clusterIn.pT()
    << endl;

  // Initialize.
  goodBranches.clear();
  badBranches.clear();
  paths.clear();

  // Remember how many steps in total were supposed to be taken.
  if (!mother) nStepsMax = depth;
  else         nStepsMax = mother->nStepsMax;

  // Initialise beam particles
  setupBeams();

  // Update probability with PDF ratio
  if (mother && mergingHooksPtr->includeRedundant()) {
    double pdfFac    = pdfForSudakov();
    clusterProb     *= pdfFac;
    prodOfProbs     *= pdfFac;
    prodOfProbsFull *= pdfFac;
  }

  // Remember reclustered radiator in lower multiplicity state
  if ( mother ) iReclusteredOld = mother->iReclusteredNew;

  // Check if more steps should be taken.
  int nFinalHeavy = 0, nFinalLight = 0;
  for ( int i = 0; i < int(state.size()); ++i )
    if ( state[i].status() > 0) {
      if ( state[i].idAbs() == 23
        || state[i].idAbs() == 24
        || state[i].idAbs() == 25)
        nFinalHeavy++;
      if ( state[i].colType() != 0
        || state[i].idAbs() == 22
        || (state[i].idAbs() > 10 && state[i].idAbs() < 20) )
        nFinalLight++;
    }
  if (nFinalHeavy == 1 && nFinalLight == 0) depth = 0;

  // Update generation index.
  generation = depth;

  // If this is not the fully clustered state, try to find possible
  // QCD clusterings.
  vector<DireClustering> clusterings;
  if ( depth > 0 ) clusterings = getAllClusterings(state);

  if (nFinalHeavy == 0 && nFinalLight == 2 && clusterings.empty()) depth = 0;

  if ( clusterings.empty() ) {
    hasMEweight = psweights->hasME(state);
    if (hasMEweight) MECnum = psweights->getME(state);
    else MECnum    = hardProcessME(state);
    fsr->direInfoPtr->message(1)
    << scientific << setprecision(15)
    << __FILE__ << " " << __func__
    << " " << __LINE__ << " : Hard ME for "
    << stringFlavs(state) << " found? " << hasMEweight << " ME "
    << MECnum << endl;
  } else {
    // Check if fixed-order ME calculation for this state exists.
    hasMEweight = psweights->hasME(state);
    // Calculate ME
    if (hasMEweight) MECnum = psweights->getME(state);
    fsr->direInfoPtr->message(1)
    << scientific << setprecision(15)
    << __FILE__ << " " << __func__
    << " " << __LINE__ << " : Hard ME for "
    << stringFlavs(state) << " found? " << hasMEweight << " ME "
    << MECnum << endl;
  }

  int na=0, nf = 0;
  for ( int i = 0; i < int(state.size()); ++i ) {
    if ( state[i].status() > 0 ) nf++;
    if ( state[i].status() > 0 && state[i].idAbs() == 22) na++;
  }

  // Check if more steps should be taken.
  int nfqq = 0, nfhh = 0, nfgg = 0;
  for ( int i = 0; i < int(state.size()); ++i )
    if ( state[i].status() > 0) {
      if ( state[i].idAbs() < 10) nfqq++;
      if ( state[i].idAbs() == 21) nfgg++;
      if ( state[i].idAbs() == 25) nfhh++;
    }

  // If no clusterings were found, the recursion is done and we
  // register this node.
  if ( clusterings.empty() ) {

    // Multiply with hard process matrix element.
    prodOfProbs     *= MECnum;
    prodOfProbsFull *= MECnum;

    // Divide out all couplings in hard process matrix element.
    double MECnumCoupl = hardProcessCouplings(state);
    if (MECnumCoupl != 0.0) {
      prodOfProbs     /= MECnumCoupl;
      prodOfProbsFull /= MECnumCoupl;
    // If the coupling exactly vanishes, force weight to zero.
    } else {
      prodOfProbs     = 0.0;
      prodOfProbsFull = 0.0;
    }

    // Additional ordering requirement between shower starting scale and
    // scale of first emission.
    if ( mergingHooksPtr->orderHistories()
      || ( infoPtr->settingsPtr->flag("Dire:doMOPS")
        && infoPtr->settingsPtr->mode("Merging:nRequested") < 2) )
      isOrdered = isOrdered && (scale < hardStartScale(state) );

    if (registerPath( *this, isOrdered, isAllowed, depth == 0 ))
      updateMinDepth(depth);

    return;
  }

  // Now we sort the possible clusterings so that we try the
  // smallest scale first.
  multimap<double, DireClustering *> sorted;
  for ( int i = 0, N = clusterings.size(); i < N; ++i ) {
    sorted.insert(make_pair(clusterings[i].pT(), &clusterings[i]));
  }

  bool foundChild = false;
  for ( multimap<double, DireClustering *>::iterator it = sorted.begin();
  it != sorted.end(); ++it ) {

    // Check if reclustering follows ordered sequence.
    bool ordered = isOrdered;
    if ( mergingHooksPtr->orderHistories() ) {
      // If this path is not ordered in pT and we already have found an
      // ordered path, then we don't need to continue along this path, unless
      // we have not yet found an allowed path.
      if ( !ordered || ( mother && (it->first < scale) ) ) {
        if ( depth >= minDepth() && onlyOrderedPaths() && onlyAllowedPaths() )
          continue;
        ordered = false;
      }
    }

    if ( !ordered || ( mother && (it->first < scale) ) ) ordered = false;

    Event newState(cluster(*it->second));

    if ( newState.size() == 0) continue;

    // Check if reclustered state should be disallowed.
    bool doCut = mergingHooksPtr->canCutOnRecState()
              || mergingHooksPtr->allowCutOnRecState();
    bool allowed = isAllowed;
    if (  doCut
      && mergingHooksPtr->doCutOnRecState(newState) ) {
      if ( onlyAllowedPaths()  ) continue;
      allowed = false;
    }

    pair <double,double> probs = getProb(*it->second);

    // Skip clustering with vanishing probability.
    if ( probs.second == 0. || hardProcessCouplings(newState) == 0.
      || (psweights->hasME(newState) && psweights->getME(newState) < 1e-20))
      continue;
    // Skip if this branch is already strongly suppressed.
    if (abs(probs.second)*prodOfProbs < PROBMAXFAC*probMax())
      continue;

    // Perform the clustering and recurse and construct the next
    // history node.
    children.push_back(new DireHistory(depth - 1,it->first, newState,
           *it->second, mergingHooksPtr, beamA, beamB, particleDataPtr,
           infoPtr, showers, fsr, isr, psweights, coupSMPtr, ordered,
           allowed,
           probs.second, probs.first, abs(probs.second)*prodOfProbs,
           probs.second*prodOfProbsFull, this ));
    foundChild = true;
  }

  // Register as valid history if no children allowed.
  if (!foundChild) {

    // Multiply with hard process matrix element.
    prodOfProbs     *= MECnum;
    prodOfProbsFull *= MECnum;

    // Divide out all couplings in hard process matrix element.
    double MECnumCoupl = hardProcessCouplings(state);
    if (MECnumCoupl != 0.0) {
      prodOfProbs     /= MECnumCoupl;
      prodOfProbsFull /= MECnumCoupl;
    // If the coupling exactly vanishes, force weight to zero.
    } else {
      prodOfProbs     = 0.0;
      prodOfProbsFull = 0.0;
    }

    if (registerPath( *this, isOrdered, isAllowed, depth == 0 ))
      updateMinDepth(depth);
  }

}

//--------------------------------------------------------------------------

// Function to project all possible paths onto only the desired paths.

bool DireHistory::projectOntoDesiredHistories() {

  bool foundGoodMOPS=true;

  // In MOPS, discard states that yield clusterings below the shower cut-off.
  if ( infoPtr->settingsPtr->flag("Dire:doMOPS")) {
    for ( map<double, DireHistory*>::iterator it = paths.begin();
      it != paths.end(); ++it ) {
      if (!it->second->hasScalesAboveCutoff()) { foundGoodMOPS=false; break; }
    }
  }

  // Loop through good branches and set the set of "good" children in mother
  // nodes.
  for ( map<double, DireHistory*>::iterator it = paths.begin();
    it != paths.end(); ++it )
    it->second->setGoodChildren();

  // Set good sisters.
  setGoodSisters();

  // Multiply couplings and ME corrections to probability.
  for ( map<double, DireHistory*>::iterator it = paths.begin();
    it != paths.end(); ++it ) {
    it->second->setCouplingOrderCount(it->second);
  }

  // Loop through the good branches and set their correct probabilities, if
  // necessary.
  if (paths.size() > 0) {
    //paths.begin()->second->mother->setProbabilities();
    // Set probabilities from next-to-lowest multi --> highest multi. If
    // lowest multi == highest multi, no need to set probabilities.
    DireHistory* deepest = nullptr;

    // Set probabilities from next-to-lowest multi --> highest multi. If
    // lowest multi == highest multi, no need to set probabilities.
    int generationMin = 1000000000;
    for ( map<double, DireHistory*>::iterator it = paths.begin();
      it != paths.end(); ++it )
      if (it->second->generation < generationMin) {
        generationMin = it->second->generation;
        deepest = it->second;
      }
    if (deepest->mother) deepest->mother->setProbabilities();
    if (deepest->mother) deepest->mother->setEffectiveScales();

  }

  // Multiply couplings and ME corrections to probability.
  for ( map<double, DireHistory*>::iterator it = paths.begin();
    it != paths.end(); ++it ) {
    it->second->multiplyMEsToPath(it->second);
  }

  // Trim to desirable histories.
  bool foundGood = trimHistories();

  //return foundGood;
  return (infoPtr->settingsPtr->flag("Dire:doMOPS")
    ? foundGoodMOPS : foundGood);

}

//--------------------------------------------------------------------------

// In the initial history node, select one of the paths according to
// the probabilities. This function should be called for the initial
// history node.
// IN  trialShower*    : Previously initialised trialShower object,
//                       to perform trial showering and as
//                       repository of pointers to initialise alphaS
//     PartonSystems* : PartonSystems object needed to initialise
//                      shower objects
// OUT double         : (Sukadov) , (alpha_S ratios) , (PDF ratios)

double DireHistory::weightMOPS(PartonLevel* trial, AlphaStrong * /*as*/,
  AlphaEM * /*aem*/, double RN) {

  // Read alpha_S in ME calculation and maximal scale (eCM)
  double maxScale = (foundCompletePath) ? infoPtr->eCM()
                  : mergingHooksPtr->muFinME();

  // Select a path of clusterings
  DireHistory *  selected = select(RN);

  // Set scales in the states to the scales pythia would have set
  selected->setScalesInHistory();

  // Keep only unordered paths, since all ordered paths have been corrected
  // with matrix element corrections.
  if (foundOrderedPath) { return 0.;}

  // Calculate no-emission probability with trial shower.
  bool nZero = false;
  vector<double> ret(createvector<double>(1.)(1.)(1.));
  vector<double> noemwt = selected->weightEmissionsVec(trial,1,-1,-1,maxScale);
  for (size_t i=0; i < ret.size(); ++i) ret[i] *= noemwt[i];
  for (size_t i=0; i < ret.size(); ++i) if (abs(ret[i]) > 1e-12) nZero = true;

  double sudakov = noemwt.front();

  // Calculate PDF ratios.
  double pdfwt = 1.;
  if (nZero) pdfwt = selected->weightPDFs( maxScale, selected->clusterIn.pT());
  for (size_t i=0; i < ret.size(); ++i) ret[i] *= pdfwt;
  nZero = false;
  for (size_t i=0; i < ret.size(); ++i) if (abs(ret[i]) > 1e-12) nZero = true;

  // Calculate coupling ratios.
  vector<double> couplwt(createvector<double>(1.)(1.)(1.));
  if (nZero) couplwt = selected->weightCouplingsDenominator();
  for (size_t i=0; i < ret.size(); ++i) ret[i] *= couplwt[i];
  nZero = false;
  for (size_t i=0; i < ret.size(); ++i) if (abs(ret[i]) > 1e-12) nZero = true;

  double coupwt = couplEffective/couplwt.front();

  // MPI no-emission probability
  int njetsMaxMPI = mergingHooksPtr->nMinMPI();
  double mpiwt = 1.;

  if (infoPtr->settingsPtr->flag("PartonLevel:MPI")) mpiwt
    = selected->weightEmissions( trial, -1, 0, njetsMaxMPI, maxScale );

  // Done
  return (sudakov*coupwt*pdfwt*mpiwt);

}

//--------------------------------------------------------------------------

vector<double> DireHistory::weightMEM(PartonLevel* trial, AlphaStrong * as,
  AlphaEM * aem, double RN) {

  // Read alpha_S in ME calculation and maximal scale (eCM)
  double maxScale = (foundCompletePath) ? infoPtr->eCM()
                  : mergingHooksPtr->muFinME();

  // Select a path of clusterings
  DireHistory *  selected = select(RN);

  // Set scales in the states to the scales pythia would have set
  selected->setScalesInHistory();

  // Calculate no-emission probability with trial shower.
  bool nZero = false;
  vector<double> ret(createvector<double>(1.)(1.)(1.));
  vector<double> noemwt = selected->weightEmissionsVec(trial,1,-1,-1,maxScale);
  for (size_t i=0; i < ret.size(); ++i) ret[i] *= noemwt[i];
  for (size_t i=0; i < ret.size(); ++i) if (abs(ret[i]) > 1e-12) nZero = true;

  // Calculate PDF ratios.
  double pdfwt = 1.;
  if (nZero) pdfwt = selected->weightPDFs( maxScale, selected->clusterIn.pT());
  for (size_t i=0; i < ret.size(); ++i) ret[i] *= pdfwt;
  nZero = false;
  for (size_t i=0; i < ret.size(); ++i) if (abs(ret[i]) > 1e-12) nZero = true;

  // Calculate coupling ratios.
  vector<double> couplwt(createvector<double>(1.)(1.)(1.));
  if (nZero) couplwt = selected->weightCouplings();
  for (size_t i=0; i < ret.size(); ++i) ret[i] *= couplwt[i];
  nZero = false;
  for (size_t i=0; i < ret.size(); ++i) if (abs(ret[i]) > 1e-12) nZero = true;

  if (nZero) {
    vector<double> vars(createvector<double>(1.)(0.25)(4.));
    double QRen  = selected->hardProcessScale(selected->state);
    double coupl = selected->hardProcessCouplings(selected->state, 1,
      QRen*QRen, as, aem);
    for (size_t i=0; i < vars.size(); ++i) {
      double ratio = selected->hardProcessCouplings(selected->state, 1,
        vars[i]*QRen*QRen, as, aem) / coupl;
      ret[i] *= ratio;
    }
  }
  return ret;

}

//--------------------------------------------------------------------------

// In the initial history node, select one of the paths according to
// the probabilities. This function should be called for the initial
// history node.
// IN  trialShower*    : Previously initialised trialShower object,
//                       to perform trial showering and as
//                       repository of pointers to initialise alphaS
//     PartonSystems* : PartonSystems object needed to initialise
//                      shower objects
// OUT double         : (Sukadov) , (alpha_S ratios) , (PDF ratios)

double DireHistory::weightTREE(PartonLevel* trial, AlphaStrong * asFSR,
  AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN) {

  if ( mergingHooksPtr->canCutOnRecState() && !foundAllowedPath ) {
    string message="Warning in DireHistory::weightTREE: No allowed history";
    message+=" found. Using disallowed history.";
    infoPtr->errorMsg(message);
  }

  if ( mergingHooksPtr->orderHistories() && !foundOrderedPath ) {
    string message="Warning in DireHistory::weightTREE: No ordered history";
    message+=" found. Using unordered history.";
    infoPtr->errorMsg(message);
  }
  if ( mergingHooksPtr->canCutOnRecState()
    && mergingHooksPtr->orderHistories()
    && !foundAllowedPath && !foundOrderedPath ) {
    string message="Warning in DireHistory::weightTREE: No allowed or ordered";
    message+=" history found.";
    infoPtr->errorMsg(message);
  }

  // Read alpha_S in ME calculation and maximal scale (eCM)
  double asME     = infoPtr->alphaS();
  double aemME    = infoPtr->alphaEM();
  double maxScale = (foundCompletePath) ? infoPtr->eCM()
                  : mergingHooksPtr->muFinME();

  // Select a path of clusterings
  DireHistory *  selected = select(RN);

  // Set scales in the states to the scales pythia would have set
  selected->setScalesInHistory();

  // Get weight.
  double sudakov   = 1.;
  double asWeight  = 1.;
  double aemWeight = 1.;
  double pdfWeight = 1.;

  // Do trial shower, calculation of alpha_S ratios, PDF ratios
  sudakov  = selected->weight( trial, asME, aemME, maxScale,
    selected->clusterIn.pT(), asFSR, asISR, aemFSR, aemISR, asWeight,
    aemWeight, pdfWeight );

  // MPI no-emission probability
  int njetsMaxMPI = mergingHooksPtr->nMinMPI();
  //double mpiwt = selected->weightEmissions( trial, -1, 0, njetsMaxMPI,
  //                 maxScale );
  double mpiwt = 1.;

  if (infoPtr->settingsPtr->flag("PartonLevel:MPI")) mpiwt
    = selected->weightEmissions( trial, -1, 0, njetsMaxMPI, maxScale );

  // Set hard process renormalisation scale to default Pythia value.
  bool resetScales = mergingHooksPtr->resetHardQRen();

  // For pure QCD dijet events, evaluate the coupling of the hard process at
  // a more reasonable pT, rather than evaluation \alpha_s at a fixed
  // arbitrary scale.
  if ( resetScales
    && mergingHooksPtr->getProcessString().compare("pp>jj") == 0) {
    // Reset to a running coupling. Here we choose FSR for simplicity.
    double newQ2Ren = pow2( selected->hardRenScale(selected->state) );
    double runningCoupling = (*asFSR).alphaS(newQ2Ren) / asME;
    asWeight *= pow2(runningCoupling);
  } else if (mergingHooksPtr->doWeakClustering()
    && isQCD2to2(selected->state)) {
    // Reset to a running coupling. Here we choose FSR for simplicity.
    double newQ2Ren = pow2( selected->hardRenScale(selected->state) );
    double runningCoupling = (*asFSR).alphaS(newQ2Ren) / asME;
    asWeight *= pow2(runningCoupling);
  }

  // For W clustering, correct the \alpha_em.
  if (mergingHooksPtr->doWeakClustering() && isEW2to1(selected->state)) {
    // Reset to a running coupling. Here we choose FSR for simplicity.
    double newQ2Ren = pow2( selected->hardRenScale(selected->state) );
    double runningCoupling = (*aemFSR).alphaEM(newQ2Ren) / aemME;
    aemWeight *= runningCoupling;
  }

  // For prompt photon events, evaluate the coupling of the hard process at
  // a more reasonable pT, rather than evaluation \alpha_s at a fixed
  // arbitrary scale.
  if ( resetScales
    && mergingHooksPtr->getProcessString().compare("pp>aj") == 0) {
    // Reset to a running coupling. In prompt photon always ISR.
    double newQ2Ren = pow2( selected->hardRenScale(selected->state) );
    double runningCoupling =
      (*asISR).alphaS( newQ2Ren + pow(mergingHooksPtr->pT0ISR(),2) ) / asME;
    asWeight *= runningCoupling;
  }

  // For DIS, set the hard process scale to Q2.
  if ( resetScales
    && ( mergingHooksPtr->getProcessString().compare("e+p>e+j") == 0
      || mergingHooksPtr->getProcessString().compare("e-p>e-j") == 0)) {
    double newQ2Ren = pow2( selected->hardRenScale(selected->state) );
    double pT20     = pow(mergingHooksPtr->pT0ISR(),2);
    if ( isMassless2to2(selected->state) ) {
      int nIncP(0), nOutP(0);
      for ( int i=0; i < selected->state.size(); ++i ) {
        if ( selected->state[i].isFinal()
          && selected->state[i].colType() != 0)
          nOutP++;
        if ( selected->state[i].status() == -21
          && selected->state[i].colType() != 0)
          nIncP++;
        }
      if (nIncP == 2 && nOutP == 2)
        asWeight *= pow2( (*asISR).alphaS(newQ2Ren+pT20) / asME );
      if (nIncP == 1 && nOutP == 2)
        asWeight *= (*asISR).alphaS(newQ2Ren+pT20) / asME
                  * (*aemFSR).alphaEM(newQ2Ren) / aemME;
    }
  }

  // Done
  return (sudakov*asWeight*aemWeight*pdfWeight*mpiwt);

}

//--------------------------------------------------------------------------

// Function to return weight of virtual correction and subtractive events
// for NL3 merging

double DireHistory::weightLOOP(PartonLevel* trial, double RN ) {

  if ( mergingHooksPtr->canCutOnRecState() && !foundAllowedPath ) {
    string message="Warning in DireHistory::weightLOOP: No allowed history";
    message+=" found. Using disallowed history.";
    infoPtr->errorMsg(message);
  }

  // Select a path of clusterings
  DireHistory *  selected = select(RN);
  // Set scales in the states to the scales pythia would have set
  selected->setScalesInHistory();

  // So far, no reweighting
  double wt = 1.;

  // Only reweighting with MPI no-emission probability
  double maxScale = (foundCompletePath) ? infoPtr->eCM()
                  : mergingHooksPtr->muFinME();
  int njetsMaxMPI = mergingHooksPtr->nMinMPI();
  double mpiwt = selected->weightEmissions( trial, -1, 0, njetsMaxMPI,
                   maxScale );
  wt = mpiwt;
  // Done
  return wt;
}

//--------------------------------------------------------------------------

// Function to calculate O(\alpha_s)-term of CKKWL-weight for NLO merging

double DireHistory::weightFIRST(PartonLevel* trial, AlphaStrong* asFSR,
  AlphaStrong* asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN,
  Rndm* rndmPtr ) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << aemFSR << aemISR;

  // Read alpha_S in ME calculation and maximal scale (eCM)
  double asME   = infoPtr->alphaS();
  double muR      = mergingHooksPtr->muRinME();
  double maxScale = (foundCompletePath)
                  ? infoPtr->eCM()
                  : mergingHooksPtr->muFinME();

  // Pick path of clusterings
  DireHistory *  selected = select(RN);
  // Set scales in the states to the scales pythia would have set
  selected->setScalesInHistory();

  int nSteps = mergingHooksPtr->getNumberOfClusteringSteps(state);

  // Get the lowest order k-factor and add first two terms in expansion
  double kFactor = asME * mergingHooksPtr->k1Factor(nSteps);

  // If using Bbar, which includes a tree-level part, subtract an
  // additional one, i.e. the O(\as^0) contribution as well
  double wt = 1. + kFactor;

  // Calculate sum of O(alpha) terms
  wt += selected->weightFirst(trial,asME, muR, maxScale, asFSR, asISR,
          rndmPtr );

  // Get starting scale for trial showers.
  double startingScale = (selected->mother) ? state.scale() : infoPtr->eCM();

  // Count emissions: New variant
  // Generate true average, not only one-point
  double nWeight1 = 0.;
  for(int i=0; i < NTRIAL; ++i) {
    // Get number of emissions
    vector<double> unresolvedEmissionTerm = countEmissions
      ( trial, startingScale, mergingHooksPtr->tms(), 2, asME, asFSR, asISR, 3,
        true, true );
    nWeight1 += unresolvedEmissionTerm[1];
  }

  wt += nWeight1/double(NTRIAL);

  // Done
  return wt;

}

//--------------------------------------------------------------------------

double DireHistory::weight_UMEPS_TREE(PartonLevel* trial, AlphaStrong * asFSR,
  AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN) {
  // No difference to CKKW-L. Recycle CKKW-L function.
  return weightTREE( trial, asFSR, asISR, aemFSR, aemISR, RN);
}

//--------------------------------------------------------------------------

// Function to return weight of virtual correction events for NLO merging

double DireHistory::weight_UMEPS_SUBT(PartonLevel* trial, AlphaStrong * asFSR,
  AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN ) {

  // Read alpha_S in ME calculation and maximal scale (eCM)
  double asME     = infoPtr->alphaS();
  double aemME    = infoPtr->alphaEM();
  double maxScale = (foundCompletePath) ? infoPtr->eCM()
                  : mergingHooksPtr->muFinME();
  // Select a path of clusterings
  DireHistory *  selected = select(RN);
  // Set scales in the states to the scales pythia would have set
  selected->setScalesInHistory();

  // Get weight.
  double sudakov   = 1.;
  double asWeight  = 1.;
  double aemWeight = 1.;
  double pdfWeight = 1.;

  // Do trial shower, calculation of alpha_S ratios, PDF ratios
  sudakov   = selected->weight(trial, asME, aemME, maxScale,
    selected->clusterIn.pT(), asFSR, asISR, aemFSR, aemISR, asWeight,
    aemWeight, pdfWeight);

  // MPI no-emission probability.
  int njetsMaxMPI = mergingHooksPtr->nMinMPI()+1;
  double mpiwt = selected->weightEmissions( trial, -1, 0, njetsMaxMPI,
                   maxScale );

  // Set hard process renormalisation scale to default Pythia value.
  bool resetScales = mergingHooksPtr->resetHardQRen();
  // For pure QCD dijet events, evaluate the coupling of the hard process at
  // a more reasonable pT, rather than evaluation \alpha_s at a fixed
  // arbitrary scale.
  if ( resetScales
    && mergingHooksPtr->getProcessString().compare("pp>jj") == 0) {
    // Reset to a running coupling. Here we choose FSR for simplicity.
    double newQ2Ren = pow2( selected->hardRenScale(selected->state) );
    double runningCoupling = (*asFSR).alphaS(newQ2Ren) / asME;
    asWeight *= pow(runningCoupling,2);
  }

  // For prompt photon events, evaluate the coupling of the hard process at
  // a more reasonable pT, rather than evaluation \alpha_s at a fixed
  // arbitrary scale.
  if ( resetScales
    && mergingHooksPtr->getProcessString().compare("pp>aj") == 0) {
    // Reset to a running coupling. In prompt photon always ISR.
    double newQ2Ren = pow2( selected->hardRenScale(selected->state) );
    double runningCoupling =
      (*asISR).alphaS( newQ2Ren + pow(mergingHooksPtr->pT0ISR(),2) ) / asME;
    asWeight *= runningCoupling;
  }

  // Done
  return (sudakov*asWeight*aemWeight*pdfWeight*mpiwt);

}

//--------------------------------------------------------------------------

double DireHistory::weight_UNLOPS_TREE(PartonLevel* trial, AlphaStrong * asFSR,
  AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN,
  int depthIn) {

  // Read alpha_S in ME calculation and maximal scale (eCM)
  double asME     = infoPtr->alphaS();
  double aemME    = infoPtr->alphaEM();
  double maxScale = (foundCompletePath) ? infoPtr->eCM()
                  : mergingHooksPtr->muFinME();
  // Select a path of clusterings
  DireHistory *  selected = select(RN);
  // Set scales in the states to the scales pythia would have set
  selected->setScalesInHistory();

  // Get weight.
  double asWeight  = 1.;
  double aemWeight = 1.;
  double pdfWeight = 1.;

  // Do trial shower, calculation of alpha_S ratios, PDF ratios
  double wt = 1.;
  if (depthIn < 0) wt = selected->weight(trial, asME, aemME, maxScale,
    selected->clusterIn.pT(), asFSR, asISR, aemFSR, aemISR, asWeight,
    aemWeight, pdfWeight);
  else {
    wt   = selected->weightEmissions( trial, 1, 0, depthIn, maxScale );
    if (wt != 0.) {
      asWeight  = selected->weightALPHAS( asME, asFSR, asISR, 0, depthIn);
      aemWeight = selected->weightALPHAEM( aemME, aemFSR, aemISR, 0, depthIn);
      pdfWeight = selected->weightPDFs
        ( maxScale, selected->clusterIn.pT(), 0, depthIn);
    }
  }

  // MPI no-emission probability.
  int njetsMaxMPI = mergingHooksPtr->nMinMPI();
  double mpiwt = selected->weightEmissions( trial, -1, 0, njetsMaxMPI,
                   maxScale );

  // Set hard process renormalisation scale to default Pythia value.
  bool resetScales = mergingHooksPtr->resetHardQRen();
  // For pure QCD dijet events, evaluate the coupling of the hard process at
  // a more reasonable pT, rather than evaluation \alpha_s at a fixed
  // arbitrary scale.
  if ( resetScales
    && mergingHooksPtr->getProcessString().compare("pp>jj") == 0) {
    // Reset to a running coupling. Here we choose FSR for simplicity.
    double newQ2Ren = pow2( selected->hardRenScale(selected->state) );
    double runningCoupling = (*asFSR).alphaS(newQ2Ren) / asME;
    asWeight *= pow(runningCoupling,2);
  }

  // For prompt photon events, evaluate the coupling of the hard process at
  // a more reasonable pT, rather than evaluation \alpha_s at a fixed
  // arbitrary scale.
  if ( resetScales
    && mergingHooksPtr->getProcessString().compare("pp>aj") == 0) {
    // Reset to a running coupling. In prompt photon always ISR.
    double newQ2Ren = pow2( selected->hardRenScale(selected->state) );
    double runningCoupling =
      (*asISR).alphaS( newQ2Ren + pow(mergingHooksPtr->pT0ISR(),2) ) / asME;
    asWeight *= runningCoupling;
  }

  // Done
  return (wt*asWeight*aemWeight*pdfWeight*mpiwt);

}

//--------------------------------------------------------------------------

double DireHistory::weight_UNLOPS_LOOP(PartonLevel* trial, AlphaStrong * asFSR,
  AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN,
  int depthIn) {
  // No difference to default NL3
  if (depthIn < 0) return weightLOOP(trial, RN);
  else return weight_UNLOPS_TREE(trial, asFSR,asISR, aemFSR,
                                 aemISR, RN, depthIn);
}

//--------------------------------------------------------------------------

double DireHistory::weight_UNLOPS_SUBT(PartonLevel* trial, AlphaStrong * asFSR,
  AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN,
  int depthIn) {

  // Select a path of clusterings
  DireHistory *  selected = select(RN);
  // Set scales in the states to the scales pythia would have set
  selected->setScalesInHistory();
  // So far, no reweighting
  double wt = 1.;

  // Read alpha_S in ME calculation and maximal scale (eCM)
  double asME     = infoPtr->alphaS();
  double aemME    = infoPtr->alphaEM();
  double maxScale = (foundCompletePath)
                  ? infoPtr->eCM()
                  : mergingHooksPtr->muFinME();

  // Only allow two clusterings if all intermediate states above the
  // merging scale.
  double nSteps = mergingHooksPtr->getNumberOfClusteringSteps(state);
  if ( nSteps == 2 && mergingHooksPtr->nRecluster() == 2
    && ( !foundCompletePath
      || !selected->allIntermediateAboveRhoMS( mergingHooksPtr->tms() )) )
    return 0.;

  // Get weights: alpha_S ratios and PDF ratios
  double asWeight  = 1.;
  double aemWeight = 1.;
  double pdfWeight = 1.;
  // Do trial shower, calculation of alpha_S ratios, PDF ratios
  double sudakov = 1.;
  if (depthIn < 0)
    sudakov = selected->weight(trial, asME, aemME, maxScale,
      selected->clusterIn.pT(), asFSR, asISR, aemFSR, aemISR, asWeight,
      aemWeight, pdfWeight);
  else {
    sudakov   = selected->weightEmissions( trial, 1, 0, depthIn, maxScale );
    if (sudakov > 0.) {
      asWeight  = selected->weightALPHAS( asME, asFSR, asISR, 0, depthIn);
      aemWeight = selected->weightALPHAEM( aemME, aemFSR, aemISR, 0, depthIn);
      pdfWeight = selected->weightPDFs
        ( maxScale, selected->clusterIn.pT(), 0, depthIn);
    }
  }

  // MPI no-emission probability.
  int njetsMaxMPI = mergingHooksPtr->nMinMPI()+1;
  double mpiwt = selected->weightEmissions( trial, -1, 0, njetsMaxMPI,
                   maxScale );

  // Set weight
  wt = ( mergingHooksPtr->nRecluster() == 2 ) ? 1.
     : asWeight*aemWeight*pdfWeight*sudakov*mpiwt;

  // Done
  return wt;

}

//--------------------------------------------------------------------------

double DireHistory::weight_UNLOPS_SUBTNLO(PartonLevel* trial,
  AlphaStrong * asFSR,
  AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN,
  int depthIn) {

  if (depthIn < 0) {

    // Select a path of clusterings
    DireHistory *  selected = select(RN);
    // Set scales in the states to the scales pythia would have set
    selected->setScalesInHistory();
    // So far, no reweighting
    double wt = 1.;
    // Only reweighting with MPI no-emission probability
    double maxScale = (foundCompletePath) ? infoPtr->eCM()
                    : mergingHooksPtr->muFinME();
    int njetsMaxMPI = mergingHooksPtr->nMinMPI()+1;
    double mpiwt = selected->weightEmissions( trial, -1, 0, njetsMaxMPI,
                     maxScale );
    wt = mpiwt;
    // Done
    return wt;

  } else return weight_UNLOPS_SUBT(trial, asFSR, asISR, aemFSR, aemISR, RN,
                                   depthIn);

}

//--------------------------------------------------------------------------

// Function to calculate O(\alpha_s)-term of CKKWL-weight for NLO merging

double DireHistory::weight_UNLOPS_CORRECTION( int order, PartonLevel* trial,
  AlphaStrong* asFSR, AlphaStrong* asISR, AlphaEM * aemFSR, AlphaEM * aemISR,
  double RN, Rndm* rndmPtr ) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << aemFSR << aemISR;

  // Already done if no correction should be calculated
  if ( order < 0 ) return 0.;

  // Read alpha_S in ME calculation and maximal scale (eCM)
  double asME     = infoPtr->alphaS();
  //double aemME    = infoPtr->alphaEM();
  double muR      = mergingHooksPtr->muRinME();
  double maxScale = (foundCompletePath)
                  ? infoPtr->eCM()
                  : mergingHooksPtr->muFinME();

  // Pick path of clusterings
  DireHistory *  selected = select(RN);
  // Set scales in the states to the scales pythia would have set
  selected->setScalesInHistory();

  double nSteps = mergingHooksPtr->getNumberOfClusteringSteps(state);

  // Get the lowest order k-factor and add first two terms in expansion
  double kFactor = asME * mergingHooksPtr->k1Factor(nSteps);

  // If using Bbar, which includes a tree-level part, subtract an
  // additional one, i.e. the O(\as^0) contribution as well
  double wt = 1.;

  // If only O(\alpha_s^0)-term is to be calculated, done already.
  if ( order == 0 ) return wt;

  // Start by adding the O(\alpha_s^1)-term of the k-factor.
  wt += kFactor;

  // Calculate sum of O(\alpha_s^1)-terms of the ckkw-l weight WITHOUT
  // the O(\alpha_s^1)-term of the last no-emission probability.
  // Get first term in expansion of alpha_s ratios.
  double wA = selected->weightFirstALPHAS( asME, muR, asFSR, asISR );
  // Add logarithm from \alpha_s expansion to weight.
  wt += wA;
  // Generate true average, not only one-point.
  double nWeight = 0.;
  for ( int i = 0; i < NTRIAL; ++i ) {
    // Get average number of emissions.
    double wE = selected->weightFirstEmissions
      ( trial,asME, maxScale, asFSR, asISR, true, true );
    // Add average number of emissions off reconstructed states to weight.
    nWeight  += wE;
    // Get first term in expansion of PDF ratios.
    double pscale = selected->clusterIn.pT();
    double wP = selected->weightFirstPDFs(asME, maxScale, pscale, rndmPtr);
    // Add integral of DGLAP shifted PDF ratios from \alpha_s expansion to wt.
    nWeight  += wP;
  }
  wt += nWeight/double(NTRIAL);

  // If O(\alpha_s^1)-term + O(\alpha_s^1)-term is to be calculated, done.
  if ( order == 1 ) return wt;

  // So far, no calculation of  O(\alpha_s^2)-term
  return 0.;

}

//--------------------------------------------------------------------------

// Function to set the state with complete scales for evolution.

void DireHistory::getStartingConditions( const double RN, Event& outState ) {

  // Select the history
  DireHistory *  selected = select(RN);

  // Set scales in the states to the scales pythia would have set
  selected->setScalesInHistory();

  // Get number of clustering steps.
  int nSteps = mergingHooksPtr->getNumberOfClusteringSteps(state);

  // Update the lowest order process.
  if (!selected->mother) {
    int nFinal = 0;
    for(int i=0; i < int(state.size()); ++i)
      if ( state[i].isFinal()) nFinal++;

    if (nSteps == 0) {
      double startingScale = hardStartScale(state);
      state.scale(startingScale);
      for (int i = 3;i < state.size();++i)
        state[i].scale(startingScale);
    }
  }

  // Save information on last splitting, to allow the next
  // emission in the shower to have smaller rapidity with
  // respect to the last ME splitting.
  infoPtr->zNowISR(0.5);
  infoPtr->pT2NowISR(pow(state[0].e(),2));
  infoPtr->hasHistory(true);

  // Copy the output state.
  outState = state;

  // Save MPI starting scale.
  if (nSteps == 0)
    mergingHooksPtr->muMI(infoPtr->eCM());
  else
    mergingHooksPtr->muMI(outState.scale());

  mergingHooksPtr->setShowerStoppingScale(0.0);

}

//--------------------------------------------------------------------------

// Function to print the history that would be chosen from the number RN.

void DireHistory::printHistory( const double RN ) {
  DireHistory *  selected = select(RN);
  selected->printStates();
  // Done
}

//--------------------------------------------------------------------------

// Function to print the states in a history, starting from the hard process.

void DireHistory::printStates() {
  if ( !mother ) {
    cout << scientific << setprecision(4) << "Probability="
         << prodOfProbs << endl;
    cout << "State:\t\t\t"; listFlavs(state,true);
    return;
  }

  // Print.
  double p = prodOfProbs/mother->prodOfProbs;
  cout << scientific << setprecision(4) << "Probabilities:"
       << "\n\t Product =              "
       << prodOfProbs << " " << prodOfProbsFull
       << "\n\t Single with coupling = " << p
       << "\n\t Cluster probability  = " << clusterProb << "\t\t"
       << clusterIn.name()
       << "\nScale=" << clusterIn.pT() << endl;
  cout << "State:\t\t\t"; listFlavs(state,true);
  cout << "rad=" << clusterIn.radPos()
       << " emt=" << clusterIn.emtPos()
       << " rec=" << clusterIn.recPos() << endl;
  // Recurse
  mother->printStates();
  // Done
  return;
}

//--------------------------------------------------------------------------

// Function to set the state with complete scales for evolution.

bool DireHistory::getClusteredEvent( const double RN, int nSteps,
                Event& outState) {

  // Select history
  DireHistory *  selected = select(RN);
  // Set scales in the states to the scales pythia would have set
  // (Only needed if not done before in calculation of weights or
  //  setting of starting conditions)
  selected->setScalesInHistory();
  // If the history does not allow for nSteps clusterings (e.g. because the
  // history is incomplete), return false
  if (nSteps > selected->nClusterings()) return false;
  // Return event with nSteps-1 additional partons (i.e. recluster the last
  // splitting) and copy the output state
  outState = selected->clusteredState(nSteps-1);
  // Done.
  return true;

}

//--------------------------------------------------------------------------

bool DireHistory::getFirstClusteredEventAboveTMS( const double RN,
  int nDesired, Event& process, int& nPerformed, bool doUpdate ) {

  // Do reclustering (looping) steps. Remember process scale.
  int nTried  = nDesired - 1;
  // Get number of clustering steps.
  int nSteps   = select(RN)->nClusterings();
  // Set scales in the states to the scales pythia would have set.
  select(RN)->setScalesInHistory();

  // Recluster until reclustered event is above the merging scale.
  Event dummy = Event();
  do {
    // Initialise temporary output of reclustering.
    dummy.clear();
    dummy.init( "(hard process-modified)", particleDataPtr );
    dummy.clear();
    // Recluster once more.
    nTried++;
    // If reclustered event does not exist, exit.
    if ( !getClusteredEvent( RN, nSteps-nTried+1, dummy ) ) return false;
    if ( nTried >= nSteps ) break;

    // Continue loop if reclustered event has unresolved partons.
  } while ( mergingHooksPtr->getNumberOfClusteringSteps(dummy) > 0
         && mergingHooksPtr->tmsNow( dummy) < mergingHooksPtr->tms() );

  // Update the hard process.
  if ( doUpdate ) process = dummy;

  // Failed to produce output state.
  if ( nTried > nSteps ) return false;

  nPerformed = nTried;
  if ( doUpdate ) {
    // Update to the actual number of steps.
    mergingHooksPtr->nReclusterSave = nPerformed;
    // Save MPI starting scale
    if (mergingHooksPtr->getNumberOfClusteringSteps(state) == 0)
      mergingHooksPtr->muMI(infoPtr->eCM());
    else
      mergingHooksPtr->muMI(state.scale());
  }

  // Done
  return true;

}

//--------------------------------------------------------------------------

// Calculate and return pdf ratio.

double DireHistory::getPDFratio( int side, bool forSudakov, bool useHardPDFs,
                    int flavNum, double xNum, double muNum,
                    int flavDen, double xDen, double muDen) {

  // Do nothing for e+e- beams
  if ( particleDataPtr->colType(flavNum) == 0) return 1.0;
  if ( particleDataPtr->colType(flavDen) == 0) return 1.0;

  // Now calculate PDF ratio if necessary
  double pdfRatio = 1.0;

  // Get mother and daughter pdfs
  double pdfNum = 0.0;
  double pdfDen = 0.0;

  // Use hard process PDFs (i.e. PDFs NOT used in ISR, FSR or MPI).
  if ( useHardPDFs ) {
    if (side == 1) {
      if (forSudakov)
        pdfNum = mother->beamA.xfHard( flavNum, xNum, muNum*muNum);
      else pdfNum = beamA.xfHard( flavNum, xNum, muNum*muNum);
      pdfDen = max(1e-10, beamA.xfHard( flavDen, xDen, muDen*muDen));
    } else {
      if (forSudakov)
        pdfNum = mother->beamB.xfHard( flavNum, xNum, muNum*muNum);
      else pdfNum = beamB.xfHard( flavNum, xNum, muNum*muNum);
      pdfDen = max(1e-10,beamB.xfHard( flavDen, xDen, muDen*muDen));
    }

  // Use rescaled PDFs in the presence of multiparton interactions
  } else {
    if (side == 1) {
      if (forSudakov)
        pdfNum = mother->beamA.xfISR(0, flavNum, xNum, muNum*muNum);
      else pdfNum = beamA.xfISR(0, flavNum, xNum, muNum*muNum);
      pdfDen = max(1e-10, beamA.xfISR(0, flavDen, xDen, muDen*muDen));
    } else {
      if (forSudakov)
        pdfNum = mother->beamB.xfISR(0, flavNum, xNum, muNum*muNum);
      else pdfNum = beamB.xfISR(0, flavNum, xNum, muNum*muNum);
      pdfDen = max(1e-10,beamB.xfISR(0, flavDen, xDen, muDen*muDen));
    }
  }

  // Cut out charm threshold.
  if ( forSudakov && abs(flavNum) ==4 && abs(flavDen) == 4 && muDen == muNum
    && muNum < particleDataPtr->m0(4))
    pdfDen = pdfNum = 1.0;

  // Return ratio of pdfs
  if ( pdfNum > 1e-15 && pdfDen > 1e-10 ) {
    pdfRatio *= pdfNum / pdfDen;
  } else if ( pdfNum < pdfDen ) {
    pdfRatio = 0.;
  } else if ( pdfNum > pdfDen ) {
    pdfRatio = 1.;
  }

  // Done
  return pdfRatio;

}

//--------------------------------------------------------------------------

// Methods used for only one path of history nodes.

// Function to set all scales in the sequence of states. This is a
// wrapper routine for setScales and setEventScales methods

void DireHistory::setScalesInHistory() {
  // Find correct links from n+1 to n states (mother --> child), as
  // needed for enforcing ordered scale sequences
  vector<int> ident;
  findPath(ident);

  // Set production scales in the states to the scales pythia would
  // have set and enforce ordering
  setScales(ident,true);

  // Set the overall event scales to the scale of the last branching
  setEventScales();

}

//--------------------------------------------------------------------------

// Function to find the index (in the mother histories) of the
// child history, thus providing a way access the path from both
// initial history (mother == 0) and final history (all children == 0)
// IN vector<int> : The index of each child in the children vector
//                  of the current history node will be saved in
//                  this vector
// NO OUTPUT

void DireHistory::findPath(vector<int>& out) {

  // If the initial and final nodes are identical, return
  if (!mother && int(children.size()) < 1) return;

  // Find the child by checking the children vector for the perfomed
  // clustering
  int iChild=-1;
  if ( mother ) {
    int size = int(mother->children.size());
    // Loop through children and identify child chosen
    for ( int i=0; i < size; ++i) {
      if ( mother->children[i]->scale == scale
        && mother->children[i]->prodOfProbs == prodOfProbs
        && equalClustering(mother->children[i]->clusterIn,clusterIn)) {
        iChild = i;
        break;
      }
    }
    // Save the index of the child in the children vector and recurse
    if (iChild >-1)
      out.push_back(iChild);
    mother->findPath(out);
  }
}

//--------------------------------------------------------------------------

// Functions to set the  parton production scales and enforce
// ordering on the scales of the respective clusterings stored in
// the History node:
// Method will start from lowest multiplicity state and move to
// higher states, setting the production scales the shower would
// have used.
// When arriving at the highest multiplicity, the method will switch
// and go back in direction of lower states to check and enforce
// ordering for unordered histories.
// IN vector<int> : Vector of positions of the chosen child
//                  in the mother history to allow to move
//                  in direction initial->final along path
//    bool        : True: Move in direction low->high
//                       multiplicity and set production scales
//                  False: Move in direction high->low
//                       multiplicity and check and enforce
//                       ordering
// NO OUTPUT

void DireHistory::setScales( vector<int> index, bool forward) {

  // Scale setting less conventional for MOPS --> separate code.

  // CKKW-L scale setting.
  if ( !infoPtr->settingsPtr->flag("Dire:doMOPS")) {
  // First, set the scales of the hard process to the kinematial
  // limit (=s)
  if ( children.empty() && forward ) {
    // New "incomplete" configurations showered from mu
    if (!mother) {
      double scaleNew = 1.;
      if (mergingHooksPtr->incompleteScalePrescip()==0) {
        scaleNew = mergingHooksPtr->muF();
      } else if (mergingHooksPtr->incompleteScalePrescip()==1) {
        Vec4 pOut;
        pOut.p(0.,0.,0.,0.);
        for(int i=0; i<int(state.size()); ++i)
          if (state[i].isFinal())
            pOut += state[i].p();
        scaleNew = pOut.mCalc();
      } else if (mergingHooksPtr->incompleteScalePrescip()==2) {
        scaleNew = state[0].e();
      }

      scaleNew = max( mergingHooksPtr->pTcut(), scaleNew);

      state.scale(scaleNew);
      for(int i=3; i < int(state.size());++i)
        if (state[i].colType() != 0)
          state[i].scale(scaleNew);
    } else {
      // 2->2 with non-parton particles showered from eCM
      state.scale( state[0].e() );
      // Count final partons
      bool isLEP = ( state[3].isLepton() && state[4].isLepton() );
      int nFinal = 0;
      int nFinalPartons = 0;
      int nFinalPhotons = 0;
      for ( int i=0; i < int(state.size()); ++i ) {
        if ( state[i].isFinal() ) {
          nFinal++;
          if ( state[i].colType() != 0 ) nFinalPartons++;
          if ( state[i].id() == 22 )     nFinalPhotons++;
        }
      }
      bool isQCD = ( nFinal == 2 && nFinal == nFinalPartons );
      bool isPPh = ( nFinal == 2 && nFinalPartons == 1 && nFinalPhotons == 1);
      // If 2->2, purely partonic, set event scale to kinematic pT
      if ( !isLEP && ( isQCD || isPPh ) ) {
        double scaleNew = hardFacScale(state);
        state.scale( scaleNew );
      }

      // For DIS, set the hard process scale to Q2.
      if ( ( isDIS2to2(state) || isMassless2to2(state) )
        && ( mergingHooksPtr->getProcessString().compare("e+p>e+j") == 0
          || mergingHooksPtr->getProcessString().compare("e-p>e-j") == 0) )
        state.scale( hardFacScale(state) );

    }
  }
  // Set all particle production scales, starting from lowest
  // multiplicity (final) state
  if (mother && forward) {
    // When choosing splitting scale, beware of unordered splittings:
    double scaleNew = 1.;
    if ( mergingHooksPtr->unorderedScalePrescip() == 0) {
      // Use larger scale as common splitting scale for mother and child
      scaleNew = max( mergingHooksPtr->pTcut(), max(scale,mother->scale));
    } else if ( mergingHooksPtr->unorderedScalePrescip() == 1) {
      // Use smaller scale as common splitting scale for mother and child
      if (scale < mother->scale)
        scaleNew = max( mergingHooksPtr->pTcut(), min(scale,mother->scale));
      else
        scaleNew = max( mergingHooksPtr->pTcut(), max(scale,mother->scale));
    }

    // Rescale the mother state partons to the clustering scales
    // that have been found along the path
    mother->state[clusterIn.emtPos()].scale(scaleNew);
    mother->state[clusterIn.radPos()].scale(scaleNew);
    mother->state[clusterIn.recPos()].scale(scaleNew);

    // Find unchanged copies of partons in higher multiplicity states
    // and rescale those
    mother->scaleCopies(clusterIn.emtPos(), mother->state, scaleNew);
    mother->scaleCopies(clusterIn.radPos(), mother->state, scaleNew);
    mother->scaleCopies(clusterIn.recPos(), mother->state, scaleNew);

    // Recurse
    mother->setScales(index,true);
  }

  // Now, check and correct ordering from the highest multiplicity
  // state backwards to all the clustered states
  if (!mother || !forward) {
    // Get index of child along the path
    int iChild = -1;
    if ( int(index.size()) > 0 ) {
      iChild = index.back();
      index.pop_back();
    }

    // Check that the reclustered scale is above the shower cut
    if (mother) {
      scale = max(mergingHooksPtr->pTcut(), scale);
    }
    // If this is NOT the 2->2 process, check and enforce ordering
    if (iChild != -1 && !children.empty()) {
      if (scale > children[iChild]->scale ) {

        if ( mergingHooksPtr->unorderedScalePrescip() == 0) {
          // Use larger scale as common splitting scale for mother and child
          double scaleNew = max( mergingHooksPtr->pTcut(),
                              max(scale,children[iChild]->scale));
          // Enforce ordering in particle production scales
          for( int i = 0; i < int(children[iChild]->state.size()); ++i)
            if (children[iChild]->state[i].scale() == children[iChild]->scale)
              children[iChild]->state[i].scale(scaleNew);
          // Enforce ordering in saved clustering scale
          children[iChild]->scale = scaleNew;

        } else if ( mergingHooksPtr->unorderedScalePrescip() == 1) {
           // Use smaller scale as common splitting scale for mother & child
           double scaleNew = max(mergingHooksPtr->pTcut(),
                               min(scale,children[iChild]->scale));
           // Enforce ordering in particle production scales
           for( int i = 0; i < int(state.size()); ++i)
             if (state[i].scale() == scale)
               state[i].scale(scaleNew);
           // Enforce ordering in saved clustering scale
           scale = scaleNew;
        }

      // Just set the overall event scale to the minimal scale
      } else {

        double scalemin = state[0].e();
        for( int i = 0; i < int(state.size()); ++i)
          if (state[i].colType() != 0)
            scalemin = max(mergingHooksPtr->pTcut(),
                         min(scalemin,state[i].scale()));
        state.scale(scalemin);
        scale = max(mergingHooksPtr->pTcut(), scale);
      }
      //Recurse
      children[iChild]->setScales(index, false);
    }
  }

  // Done with CKKW-L scale setting.
  // MOPS scale setting.
  } else {

  // First, set the scales of the hard process to the kinematial
  // limit (=s)
  if ( children.empty() && forward ) {
    // New "incomplete" configurations showered from mu
    if (!mother) {
      double scaleNew = 1.;
      if (mergingHooksPtr->incompleteScalePrescip()==0) {
        scaleNew = mergingHooksPtr->muF();
      } else if (mergingHooksPtr->incompleteScalePrescip()==1) {
        Vec4 pOut;
        pOut.p(0.,0.,0.,0.);
        for(int i=0; i<int(state.size()); ++i)
          if (state[i].isFinal())
            pOut += state[i].p();
        scaleNew = pOut.mCalc();
      } else if (mergingHooksPtr->incompleteScalePrescip()==2) {
        scaleNew = state[0].e();
      }

      scaleNew = max( mergingHooksPtr->pTcut(), scaleNew);

      state.scale(scaleNew);
      for(int i=3; i < int(state.size());++i)
        if (state[i].colType() != 0)
          state[i].scale(scaleNew);
    } else {
      // 2->2 with non-parton particles showered from eCM
      state.scale( state[0].e() );
      // Count final partons
      bool isLEP = ( state[3].isLepton() && state[4].isLepton() );
      int nFinal = 0;
      int nFinalPartons = 0;
      int nFinalPhotons = 0;
      for ( int i=0; i < int(state.size()); ++i ) {
        if ( state[i].isFinal() ) {
          nFinal++;
          if ( state[i].colType() != 0 ) nFinalPartons++;
          if ( state[i].id() == 22 )     nFinalPhotons++;
        }
      }
      bool isQCD = ( nFinal == 2 && nFinal == nFinalPartons );
      bool isPPh = ( nFinal == 2 && nFinalPartons == 1 && nFinalPhotons == 1);
      // If 2->2, purely partonic, set event scale to kinematic pT
      if ( !isLEP && ( isQCD || isPPh ) ) {
        double scaleNew = hardFacScale(state);
        state.scale( scaleNew );
      }

      // For DIS, set the hard process scale to Q2.
      if ( ( isDIS2to2(state) || isMassless2to2(state) )
        && ( mergingHooksPtr->getProcessString().compare("e+p>e+j") == 0
          || mergingHooksPtr->getProcessString().compare("e-p>e-j") == 0) )
        state.scale( hardFacScale(state) );

      double hardScale = hardStartScale(state);
      state.scale(hardScale);


    }
  }

  // Set all particle production scales, starting from lowest
  // multiplicity (final) state
  if (mother && forward) {
    double scaleNew = (scaleEffective > 0.) ? scaleEffective : scale;
    scale = max(mergingHooksPtr->pTcut(), scaleNew);

    double scaleProduction = max( mergingHooksPtr->pTcut(),
                                  mother->scaleEffective);
    scaleProduction = max(scaleProduction,scaleNew);

    // Rescale the mother state partons to the clustering scales
    // that have been found along the path
    mother->state[clusterIn.emtPos()].scale(scaleProduction);
    mother->state[clusterIn.radPos()].scale(scaleProduction);
    mother->state[clusterIn.recPos()].scale(scaleProduction);

    // Find unchanged copies of partons in higher multiplicity states
    // and rescale those
    mother->scaleCopies(clusterIn.emtPos(), mother->state, scaleProduction);
    mother->scaleCopies(clusterIn.radPos(), mother->state, scaleProduction);
    mother->scaleCopies(clusterIn.recPos(), mother->state, scaleProduction);

    // Recurse
    mother->setScales(index,true);
  }

  // Now, check and correct ordering from the highest multiplicity
  // state backwards to all the clustered states
  if (!mother || !forward) {

    // Get index of child along the path
    int iChild = -1;
    if ( int(index.size()) > 0 ) {
      iChild = index.back();
      index.pop_back();
    }

    // Check that the reclustered scale is above the shower cut
    if (mother) {
      scale = max(mergingHooksPtr->pTcut(), scale);
      if (infoPtr->settingsPtr->flag("Dire:doMOPS"))
        scale = max(scale,mother->scaleEffective);
    }
    // If this is NOT the 2->2 process, check and enforce ordering
    if (iChild != -1 && !children.empty()) {
      if (scale > children[iChild]->scale ) {

        double scaleNew = max( mergingHooksPtr->pTcut(),
          max(children[iChild]->scale, scaleEffective));
        if (mother) scaleNew = max(scaleNew, mother->scaleEffective);

        // Enforce ordering in particle production scales
        for( int i = 0; i < int(children[iChild]->state.size()); ++i) {
          if (children[iChild]->state[i].scale() == children[iChild]->scale)
            children[iChild]->state[i].scale(scaleNew);
        }
        // Enforce ordering in saved clustering scale
        children[iChild]->scale = scaleNew;

      } else {
        double scalemin = state[0].e();
        for( int i = 0; i < int(state.size()); ++i)
          if (state[i].colType() != 0)
            scalemin = max(mergingHooksPtr->pTcut(),
                         min(scalemin,state[i].scale()));
        state.scale(scalemin);
        scale = max(mergingHooksPtr->pTcut(), scale);
      }
      //Recurse
      children[iChild]->setScales(index, false);
    }
  }

  // Done with MOPS scale setting.
  }


}

//--------------------------------------------------------------------------

// Function to find a particle in all higher multiplicity events
// along the history path and set its production scale to the input
// scale
// IN  int iPart       : Parton in refEvent to be checked / rescaled
//     Event& refEvent : Reference event for iPart
//     double scale    : Scale to be set as production scale for
//                       unchanged copies of iPart in subsequent steps

void DireHistory::scaleCopies(int iPart, const Event& refEvent, double rho) {

  // Check if any parton recently rescaled is found unchanged:
  // Same charge, colours in mother->state
  if ( mother ) {
    for( int i=0; i < mother->state.size(); ++i) {
      if ( ( mother->state[i].id()         == refEvent[iPart].id()
          && mother->state[i].colType()    == refEvent[iPart].colType()
          && mother->state[i].chargeType() == refEvent[iPart].chargeType()
          && mother->state[i].col()        == refEvent[iPart].col()
          && mother->state[i].acol()       == refEvent[iPart].acol() )
         ) {
        // Rescale the unchanged parton
        mother->state[i].scale(rho);
        // Recurse
         if (mother->mother)
          mother->scaleCopies( iPart, refEvent, rho );
       } // end if found unchanged parton case
    } // end loop over particle entries in event
  }
}

//--------------------------------------------------------------------------

// Functions to set the OVERALL EVENT SCALES [=state.scale()] to
// the scale of the last clustering
// NO INPUT
// NO OUTPUT

void DireHistory::setEventScales() {
  // Set the event scale to the scale of the last clustering,
  // except for the very lowest multiplicity state
  if (mother) {
    mother->state.scale(scale);
    // Recurse
    mother->setEventScales();
  }
}

//--------------------------------------------------------------------------

// Function to return the depth of the history (i.e. the number of
// reclustered splittings)
// NO INPUT
// OUTPUT int  : Depth of history

int DireHistory::nClusterings() {
  if (!mother) return 0;
  int w = mother->nClusterings();
  w += 1;
  return w;
}

//--------------------------------------------------------------------------

// Functions to return the event after nSteps splittings of the 2->2 process
// Example: nSteps = 1 -> return event with one additional parton
// INPUT  int   : Number of splittings in the event,
//                as counted from core 2->2 process
// OUTPUT Event : event with nSteps additional partons

Event DireHistory::clusteredState(int nSteps) {

  // Save state
  Event outState = state;
  // As long as there are steps to do, recursively save state
  if (mother && nSteps > 0)
    outState = mother->clusteredState(nSteps - 1);
  // Done
  return outState;

}

//--------------------------------------------------------------------------

// Function to choose a path from all paths in the tree
// according to their splitting probabilities
// IN double    : Random number
// OUT DireHistory* : Leaf of history path chosen

DireHistory * DireHistory::select(double rnd) {

  // No need to choose if no paths have been constructed.
  if ( goodBranches.empty() && badBranches.empty() ) return this;

  // Choose amongst paths allowed by projections.
  double sum = 0.;
  map<double, DireHistory*> selectFrom;
  if ( !goodBranches.empty() ) {
    selectFrom = goodBranches;
    sum        = sumGoodBranches;
  } else {
    selectFrom = badBranches;
    sum        = sumBadBranches;
  }

  // Choose history according to probability, be careful about upper bound
  if ( rnd != 1. ) {
    return selectFrom.upper_bound(sum*rnd)->second;
  } else {
    return selectFrom.lower_bound(sum*rnd)->second;
  }

  // Done
}

//--------------------------------------------------------------------------

// Function to project paths onto desired paths.

bool DireHistory::trimHistories() {
  // Do nothing if no paths have been constructed.
  if ( paths.empty() ) return false;
  // Loop through all constructed paths. Check all removal conditions.
  for ( map<double, DireHistory*>::iterator it = paths.begin();
    it != paths.end(); ++it ) {
    // Check if history is allowed.
    if ( it->second->keep() && !it->second->keepHistory() )
      it->second->remove();
  }

  // Project onto desired / undesired branches.
  double sumold(0.), sumnew(0.), mismatch(0.);
  // Loop through all constructed paths and store allowed paths.
  // Skip undesired paths.
  for ( map<double, DireHistory*>::iterator it = paths.begin();
    it != paths.end(); ++it ) {
    // Update index
    sumnew = it->second->prodOfProbs;
    if ( it->second->keep() ) {
      // Fill branches with allowed paths.
      goodBranches.insert( make_pair( sumnew - mismatch, it->second) );
      // Add probability of this path.
      sumGoodBranches = sumnew - mismatch;
    } else {
      // Update mismatch in probabilities resulting from not including this
      // path
      double mismatchOld = mismatch;
      mismatch += sumnew - sumold;
      // Fill branches with allowed paths.
      badBranches.insert( make_pair( mismatchOld + sumnew - sumold,
        it->second ) );
      // Add probability of this path.
      sumBadBranches = mismatchOld  + sumnew - sumold;
    }
    // remember index of this path in order to caclulate probability of
    // subsequent path.
    sumold = it->second->prodOfProbs;
  }

  // Done
  return !goodBranches.empty();
}

//--------------------------------------------------------------------------

// Function implementing checks on a paths, deciding if the path is valid.

bool DireHistory::keepHistory() {
  bool keepPath = true;

  double hardScale = hardStartScale(state);

  // Tag unordered paths for removal.
  if ( mergingHooksPtr->getProcessString().compare("pp>jj") == 0
    || mergingHooksPtr->getProcessString().compare("pp>aj") == 0
    || isQCD2to2(state)   ) {
    // Tag unordered paths for removal. Include scale of hard 2->2 process
    // into the ordering definition.
    hardScale = hardStartScale(state);
  }

  // Set starting scale to mass of Drell-Yan for 2->1.
  if (isEW2to1(state)) {
    Vec4 pSum(0,0,0,0);
    for (int i = 0;i < state.size(); ++i)
      if (state[i].isFinal()) pSum += state[i].p();
    hardScale = pSum.mCalc();
  }

  // For DIS, set the hard process scale to Q2.
  if ( mergingHooksPtr->getProcessString().compare("e+p>e+j") == 0
    || mergingHooksPtr->getProcessString().compare("e-p>e-j") == 0 ) {
    // Tag unordered paths for removal. Include scale of hard 2->2 process
    // into the ordering definition.
    hardScale = hardFacScale(state);
  }

  keepPath = isOrderedPath( hardScale );

  if ( !mergingHooksPtr->orderHistories() ) keepPath = true;

  //Done
  return keepPath;
}

//--------------------------------------------------------------------------

// Function to check if a path is ordered in evolution pT.

bool DireHistory::isOrderedPath( double maxscale ) {
  double newscale = clusterIn.pT();
  if ( !mother ) return true;
  bool ordered = mother->isOrderedPath(newscale);
  if ( !ordered || maxscale < newscale) return false;
  return ordered;
}

//--------------------------------------------------------------------------

// Function to check if all reconstucted states in a path pass the merging
// scale cut.

bool DireHistory::allIntermediateAboveRhoMS( double rhoms, bool good ) {
  // If one state below the merging scale has already been found, no need to
  // check further.
  if ( !good ) return false;
  // Check merging scale for states with more than 0 jets
  int nFinal = 0;
  for ( int i = 0; i < state.size(); ++i )
    if ( state[i].isFinal() && state[i].colType() != 0 )
      nFinal++;
  double rhoNew = (nFinal > 0 ) ? mergingHooksPtr->tmsNow( state )
                : state[0].e();
  // Assume state from ME generator passes merging scale cut.
  if ( !mother ) return good;
  // Recurse.
  return mother->allIntermediateAboveRhoMS( rhoms, (rhoNew > rhoms) );
}

//--------------------------------------------------------------------------

// Function to check if any ordered paths were found (and kept).

bool DireHistory::foundAnyOrderedPaths() {
  // Do nothing if no paths were found
  if ( paths.empty() ) return false;
  double maxscale = hardStartScale(state);
  // Loop through paths. Divide probability into ordered and unordered pieces.
  for ( map<double, DireHistory*>::iterator it = paths.begin();
    it != paths.end(); ++it )
    if ( it->second->isOrderedPath(maxscale) )
      return true;
  // Done
  return false;
}

//--------------------------------------------------------------------------

// Function to check if a path contains any clustering scales below the
// shower cut-off.

bool DireHistory::hasScalesAboveCutoff() {
  if ( !mother ) return true;
  return ( clusterIn.pT() > mergingHooksPtr->pTcut()
        && mother->hasScalesAboveCutoff() );
}

//--------------------------------------------------------------------------

// For a full path, find the weight calculated from the ratio of
// couplings, the no-emission probabilities, and possible PDF
// ratios. This function should only be called for the last history
// node of a full path.
// IN  TimeShower : Already initialised shower object to be used as
//                  trial shower
//     double     : alpha_s value used in ME calculation
//     double     : Maximal mass scale of the problem (e.g. E_CM)
//     AlphaStrong: Initialised shower alpha_s object for FSR
//                  alpha_s ratio calculation
//     AlphaStrong: Initialised shower alpha_s object for ISR
//                  alpha_s ratio calculation (can be different from previous)

double DireHistory::weight(PartonLevel* trial, double as0, double aem0,
  double maxscale, double pdfScale, AlphaStrong * asFSR, AlphaStrong * asISR,
  AlphaEM * aemFSR, AlphaEM * aemISR, double& asWeight, double& aemWeight,
  double& pdfWeight) {

  // Use correct scale
  double newScale = scale;

  // For ME state, just multiply by PDF ratios
  if ( !mother ) {

    int sideRad = (state[3].pz() > 0) ? 1 :-1;
    int sideRec = (state[4].pz() > 0) ? 1 :-1;

    // Calculate PDF first leg
    if (state[3].colType() != 0) {
      // Find x value and flavour
      double x = 2.*state[3].e() / state[0].e();
      int flav = state[3].id();
      // Find numerator/denominator scale
      double scaleNum = (children.empty()) ? hardFacScale(state) : maxscale;
      double scaleDen = mergingHooksPtr->muFinME();
      // For initial parton, multiply by PDF ratio
      double ratio = getPDFratio(sideRad, false, false, flav, x, scaleNum,
                       flav, x, scaleDen);
      pdfWeight *= ratio;
    }

    // Calculate PDF ratio for second leg
    if (state[4].colType() != 0) {
      // Find x value and flavour
      double x = 2.*state[4].e() / state[0].e();
      int flav = state[4].id();
      // Find numerator/denominator scale
      double scaleNum = (children.empty()) ? hardFacScale(state) : maxscale;
      double scaleDen = mergingHooksPtr->muFinME();
      // For initial parton, multiply with PDF ratio
      double ratio = getPDFratio(sideRec, false, false, flav, x, scaleNum,
                       flav, x, scaleDen);
      pdfWeight *= ratio;
    }

    return 1.0;
  }

  // Remember new PDF scale n case true scale should be used for un-ordered
  // splittings.
  double newPDFscale = newScale;
  if ( !infoPtr->settingsPtr->flag("Dire:doMOPS")
    &&  mergingHooksPtr->unorderedPDFscalePrescip() == 1)
    newPDFscale = clusterIn.pT();

  // Recurse
  double w = mother->weight(trial, as0, aem0, newScale, newPDFscale,
    asFSR, asISR, aemFSR, aemISR, asWeight, aemWeight, pdfWeight);

  // Do nothing for empty state
  if (state.size() < 3) return 1.0;
  // If up to now, trial shower was not successful, return zero
  // Do trial shower on current state, return zero if not successful
  w *= doTrialShower(trial, 1, maxscale).front();

  int emtType = mother->state[clusterIn.emtPos()].colType();
  bool isQCD = emtType != 0;
  bool isQED = emtType == 0;

  pair<int,double> coup = getCoupling(mother->state, clusterIn.emittor,
    clusterIn.emtPos(), clusterIn.recoiler, clusterIn.name());

  if (coup.first > 0) {
    isQCD = isQED = false;
    if (coup.first == 1)
      asWeight  *= coup.second * 2.*M_PI / as0;
    if (coup.first == 2 || coup.first == 3)
      aemWeight *= coup.second * 2.*M_PI / aem0;
  }

  // Calculate alpha_s ratio for current state.
  if ( asFSR && asISR && isQCD) {
    double asScale = pow2( newScale );
    if ( !infoPtr->settingsPtr->flag("Dire:doMOPS")
      &&  mergingHooksPtr->unorderedASscalePrescip() == 1)
      asScale = pow2( clusterIn.pT() );

    // Add regularisation scale to initial state alpha_s.
    bool FSR = mother->state[clusterIn.emittor].isFinal();
    if (!FSR) asScale += pow2(mergingHooksPtr->pT0ISR());

    // Directly get argument of running alpha_s from shower plugin.
    asScale = getShowerPluginScale(mother->state, clusterIn.emittor,
      clusterIn.emtPos(), clusterIn.recoiler, clusterIn.name(),
      "scaleAS", asScale);

    if (infoPtr->settingsPtr->flag("Dire:doMOPS"))
      asScale = pow2(newScale);

    double alphaSinPS = (FSR) ? (*asFSR).alphaS(asScale)
                              : (*asISR).alphaS(asScale);
    asWeight *= alphaSinPS / as0;
  }

  // Calculate alpha_em ratio for current state.
  if ( aemFSR && aemISR && isQED ) {
    double aemScale = pow2( newScale );
    if ( !infoPtr->settingsPtr->flag("Dire:doMOPS")
      &&  mergingHooksPtr->unorderedASscalePrescip() == 1)
      aemScale = pow2( clusterIn.pT() );

    // Add regularisation scale to initial state alpha_s.
    bool FSR = mother->state[clusterIn.emittor].isFinal();
    if (!FSR) aemScale += pow2(mergingHooksPtr->pT0ISR());

    // Directly get argument of running alpha_em from shower plugin.
    aemScale = getShowerPluginScale(mother->state, clusterIn.emittor,
      clusterIn.emtPos(), clusterIn.recoiler, clusterIn.name(),
      "scaleEM", aemScale);

    double alphaEMinPS = (FSR) ? (*aemFSR).alphaEM(aemScale)
                               : (*aemISR).alphaEM(aemScale);
    aemWeight *= alphaEMinPS / aem0;
  }

  // Calculate pdf ratios: Get both sides of event
  int inP = 3;
  int inM = 4;
  int sideP = (mother->state[inP].pz() > 0) ? 1 :-1;
  int sideM = (mother->state[inM].pz() > 0) ? 1 :-1;

  if ( mother->state[inP].colType() != 0 ) {
    // Find x value and flavour
    double x = getCurrentX(sideP);
    int flav = getCurrentFlav(sideP);
    // Find numerator scale
    double scaleNum = (children.empty())
                    ? hardFacScale(state)
                    : ( (!infoPtr->settingsPtr->flag("Dire:doMOPS")
                       && mergingHooksPtr->unorderedPDFscalePrescip() == 1)
                      ? pdfScale : maxscale );
    double scaleDen = (  !infoPtr->settingsPtr->flag("Dire:doMOPS")
                       && mergingHooksPtr->unorderedPDFscalePrescip() == 1)
                    ? clusterIn.pT() : newScale;
    // Multiply PDF ratio
    double ratio = getPDFratio(sideP, false, false, flav, x, scaleNum,
                     flav, x, scaleDen);

    pdfWeight *= ratio;
  }

  if ( mother->state[inM].colType() != 0 ) {
    // Find x value and flavour
    double x = getCurrentX(sideM);
    int flav = getCurrentFlav(sideM);
    // Find numerator scale
    double scaleNum = (children.empty())
                    ? hardFacScale(state)
                    : ( (!infoPtr->settingsPtr->flag("Dire:doMOPS")
                       && mergingHooksPtr->unorderedPDFscalePrescip() == 1)
                      ? pdfScale : maxscale );
    double scaleDen = (  !infoPtr->settingsPtr->flag("Dire:doMOPS")
                       && mergingHooksPtr->unorderedPDFscalePrescip() == 1)
                    ? clusterIn.pT() : newScale;
    // Multiply PDF ratio
    double ratio = getPDFratio(sideM, false, false, flav, x, scaleNum,
                     flav, x, scaleDen);

    pdfWeight *= ratio;
  }

  // Done
  return w;
}

//--------------------------------------------------------------------------

// Function to return the \alpha_s-ratio part of the CKKWL weight of a path.

double DireHistory::weightALPHAS( double as0, AlphaStrong * asFSR,
  AlphaStrong * asISR, int njetMin, int njetMax ) {

  // For ME state, do nothing.
  if ( !mother ) return 1.;
  // Recurse
  double w = mother->weightALPHAS( as0, asFSR, asISR, njetMin, njetMax );
  // Do nothing for empty state
  if (state.size() < 3) return w;

  // If this node has too many jets, no not calculate no-emission probability.
  int njetNow = mergingHooksPtr->getNumberOfClusteringSteps( state) ;
  if (njetNow >= njetMax) return 1.0;

  // Store variables for easy use.
  bool FSR = mother->state[clusterIn.emittor].isFinal();
  int emtID = mother->state[clusterIn.emtPos()].id();

  // Do not correct alphaS if it is an EW emission.
  if (abs(emtID) == 22 || abs(emtID) == 23 || abs(emtID) == 24) return w;

  if (njetNow < njetMin ) w *= 1.0;
  else {
  // Calculate alpha_s ratio for current state
  if ( asFSR && asISR ) {
    double asScale = pow2( scale );
    if (!infoPtr->settingsPtr->flag("Dire:doMOPS")
      && mergingHooksPtr->unorderedASscalePrescip() == 1)
      asScale = pow2( clusterIn.pT() );

    // Add regularisation scale to initial state alpha_s.
    if (!FSR) asScale += pow2(mergingHooksPtr->pT0ISR());

    // Directly get argument of running alpha_s from shower plugin.
    asScale = getShowerPluginScale(mother->state, clusterIn.emittor,
      clusterIn.emtPos(), clusterIn.recoiler, clusterIn.name(),
      "scaleAS", asScale);

    double alphaSinPS = (FSR) ? (*asFSR).alphaS(asScale)
                              : (*asISR).alphaS(asScale);
    w *= alphaSinPS / as0;
  }
  }

  // Done
  return w;
}

//--------------------------------------------------------------------------

// Function to return the \alpha_s-ratio part of the CKKWL weight of a path.

vector<double> DireHistory::weightCouplings() {

  // For ME state, do nothing.
  if ( !mother ) return createvector<double>(1.)(1.)(1.);
  // Recurse
  vector<double> w = mother->weightCouplings();
  // Do nothing for empty state
  if (state.size() < 3) return w;

  // Get local copies of input system
  int rad     = clusterIn.radPos();
  int rec     = clusterIn.recPos();
  int emt     = clusterIn.emtPos();
  string name = clusterIn.name();

  if (!(fsr && isr)) return createvector<double>(1.)(1.)(1.);
  bool isFSR = fsr->isTimelike(mother->state, rad, emt, rec, "");
  bool isISR = isr->isSpacelike(mother->state, rad, emt, rec, "");
  double mu2Ren = pow2(mergingHooksPtr->muR());
  double t      = pow2(scale);
  double renormMultFacFSR
    = infoPtr->settingsPtr->parm("TimeShower:renormMultFac");
  double renormMultFacISR
    = infoPtr->settingsPtr->parm("SpaceShower:renormMultFac");
  if      (isFSR) t *= renormMultFacFSR;
  else if (isISR) t *= renormMultFacISR;

  double couplingOld(1.), couplingNew(1.);
  if (isFSR) couplingOld = fsr->getCoupling( mu2Ren, name);
  if (isISR) couplingOld = isr->getCoupling( mu2Ren, name);
  vector<double> variations(createvector<double>(1.)(0.25)(4.));
  for (size_t i=0; i<variations.size(); ++i) {
    if (isFSR) couplingNew = fsr->getCoupling( variations[i]*t, name);
    if (isISR) couplingNew = fsr->getCoupling( variations[i]*t, name);
    w[i] *= couplingNew / couplingOld;
  }

  // Done
  return w;
}

//--------------------------------------------------------------------------

// Function to return the \alpha_s-ratio part of the CKKWL weight of a path.

vector<double> DireHistory::weightCouplingsDenominator() {

  // For ME state, do nothing.
  if ( !mother ) return createvector<double>(1.)(1.)(1.);
  // Recurse
  vector<double> w = mother->weightCouplingsDenominator();
  // Do nothing for empty state
  if (state.size() < 3) return w;
  // Get local copies of input system
  if (!(fsr && isr)) return createvector<double>(1.)(1.)(1.);
  for (size_t i=0; i<w.size(); ++i) {
    w[i] *= clusterCoupl*2.*M_PI;
  }

  // Done
  return w;
}

//--------------------------------------------------------------------------

// Function to return the \alpha_em-ratio part of the CKKWL weight of a path.

double DireHistory::weightALPHAEM( double aem0, AlphaEM * aemFSR,
  AlphaEM * aemISR, int njetMin, int njetMax ) {

  // For ME state, do nothing.
  if ( !mother ) return 1.;
  // Recurse
  double w = mother->weightALPHAEM( aem0, aemFSR, aemISR, njetMin, njetMax);
  // Do nothing for empty state
  if (state.size() < 3) return w;

  // If this node has too many jets, no not calculate no-emission probability.
  int njetNow = mergingHooksPtr->getNumberOfClusteringSteps( state) ;
  if (njetNow >= njetMax) return 1.0;

  // Store variables for easy use.
  bool FSR = mother->state[clusterIn.emittor].isFinal();
  int emtID = mother->state[clusterIn.emtPos()].id();

  // Do not correct alpha EM if it not an EW emission.
  if (!(abs(emtID) == 22 || abs(emtID) == 23 || abs(emtID) == 24)) return w;

  if (njetNow < njetMin ) w *= 1.0;
  else {
  // Calculate alpha_s ratio for current state
  if ( aemFSR && aemISR ) {
    double aemScale = pow2( scale );
    if (!infoPtr->settingsPtr->flag("Dire:doMOPS")
      && mergingHooksPtr->unorderedASscalePrescip() == 1)
      aemScale = pow2( clusterIn.pT() );

    // Add regularisation scale to initial state alpha_em.
    if (!FSR) aemScale += pow2(mergingHooksPtr->pT0ISR());

    // Directly get argument of running alpha_em from shower plugin.
    aemScale = getShowerPluginScale(mother->state, clusterIn.emittor,
      clusterIn.emtPos(), clusterIn.recoiler, clusterIn.name(),
      "scaleEM", aemScale);

    double alphaEMinPS = (FSR) ? (*aemFSR).alphaEM(aemScale)
                               : (*aemISR).alphaEM(aemScale);
    w *= alphaEMinPS / aem0;
  }
  }

  // Done
  return w;
}

//--------------------------------------------------------------------------

// Function to return the PDF-ratio part of the CKKWL weight of a path.

double DireHistory::weightPDFs( double maxscale, double pdfScale,
  int njetMin, int njetMax ) {

  // Use correct scale
  double newScale = scale;
  int njetNow = mergingHooksPtr->getNumberOfClusteringSteps( state);

  // For ME state, just multiply by PDF ratios
  if ( !mother ) {

    // If this node has too many jets, no not calculate PDF ratio.
    if (njetMax > -1 && njetNow > njetMax) return 1.0;

    double wt = 1.;
    int sideRad = (state[3].pz() > 0) ? 1 :-1;
    int sideRec = (state[4].pz() > 0) ? 1 :-1;

    // Calculate PDF first leg
    if (state[3].colType() != 0) {
      // Find x value and flavour
      double x = 2.*state[3].e() / state[0].e();
      int flav = state[3].id();
      // Find numerator/denominator scale
      double scaleNum = (children.empty()) ? hardFacScale(state) : maxscale;
      double scaleDen = mergingHooksPtr->muFinME();
      // For initial parton, multiply by PDF ratio
      if (njetMin > -1 && njetNow >= njetMin ) wt *= getPDFratio(sideRad,
        false, false, flav, x, scaleNum, flav, x, scaleDen);
      else if (njetMin == -1)                  wt *= getPDFratio(sideRad,
        false, false, flav, x, scaleNum, flav, x, scaleDen);
    }

    // Calculate PDF ratio for second leg
    if (state[4].colType() != 0) {
      // Find x value and flavour
      double x = 2.*state[4].e() / state[0].e();
      int flav = state[4].id();
      // Find numerator/denominator scale
      double scaleNum = (children.empty()) ? hardFacScale(state) : maxscale;
      double scaleDen = mergingHooksPtr->muFinME();
      if (njetMin > -1 && njetNow >= njetMin ) wt *= getPDFratio(sideRec,
        false, false, flav, x, scaleNum, flav, x, scaleDen);
      else if (njetMin == -1)                  wt *= getPDFratio(sideRec,
        false, false, flav, x, scaleNum, flav, x, scaleDen);
    }

    return wt;
  }

  // Remember new PDF scale n case true scale should be used for un-ordered
  // splittings.
  double newPDFscale = newScale;
  if ( !infoPtr->settingsPtr->flag("Dire:doMOPS")
    &&  mergingHooksPtr->unorderedPDFscalePrescip() == 1)
    newPDFscale = clusterIn.pT();

  // Recurse
  double w = mother->weightPDFs( newScale, newPDFscale, njetMin, njetMax);

  // Do nothing for empty state
  if (state.size() < 3) return w;

  // Calculate pdf ratios: Get both sides of event
  int inP = 3;
  int inM = 4;
  int sideP = (mother->state[inP].pz() > 0) ? 1 :-1;
  int sideM = (mother->state[inM].pz() > 0) ? 1 :-1;

  if ( mother->state[inP].colType() != 0 ) {
    // Find x value and flavour
    double x = getCurrentX(sideP);
    int flav = getCurrentFlav(sideP);
    // Find numerator scale
    double scaleNum = (children.empty())
                ? hardFacScale(state)
                  : ( (!infoPtr->settingsPtr->flag("Dire:doMOPS")
                    && mergingHooksPtr->unorderedPDFscalePrescip() == 1)
                  ? pdfScale : maxscale );
    double scaleDen = (  !infoPtr->settingsPtr->flag("Dire:doMOPS")
                && mergingHooksPtr->unorderedPDFscalePrescip() == 1)
                ? clusterIn.pT() : newScale;
    double xDen = (njetMax > -1 && njetNow == njetMax)
                ? mother->getCurrentX(sideP) : x;
    int flavDen = (njetMax > -1 && njetNow == njetMax)
                ? mother->getCurrentFlav(sideP) : flav;
    double sDen = (njetMax > -1 && njetNow == njetMax)
                ? mergingHooksPtr->muFinME() : scaleDen;
    if (njetMin > -1 && njetNow >= njetMin ) w *= getPDFratio(sideP,
      false, false, flav, x, scaleNum, flavDen, xDen, sDen);
    else if (njetMin == -1)                  w *= getPDFratio(sideP,
      false, false, flav, x, scaleNum, flavDen, xDen, sDen);
  }

  if ( mother->state[inM].colType() != 0 ) {
    // Find x value and flavour
    double x = getCurrentX(sideM);
    int flav = getCurrentFlav(sideM);
    // Find numerator scale
    double scaleNum = (children.empty())
                ? hardFacScale(state)
                  : ( (!infoPtr->settingsPtr->flag("Dire:doMOPS")
                    && mergingHooksPtr->unorderedPDFscalePrescip() == 1)
                  ? pdfScale : maxscale );
    double scaleDen = (  !infoPtr->settingsPtr->flag("Dire:doMOPS")
                && mergingHooksPtr->unorderedPDFscalePrescip() == 1)
                ? clusterIn.pT() : newScale;
    double xDen = (njetMax > -1 && njetNow == njetMax)
                ? mother->getCurrentX(sideM) : x;
    int flavDen = (njetMax > -1 && njetNow == njetMax)
                ? mother->getCurrentFlav(sideM) : flav;
    double sDen = (njetMax > -1 && njetNow == njetMax)
                ? mergingHooksPtr->muFinME() : scaleDen;
    if (njetMin > -1 && njetNow >= njetMin ) w *= getPDFratio(sideM,
      false, false, flav, x, scaleNum, flavDen, xDen, sDen);
    else if (njetMin == -1)                  w *= getPDFratio(sideM,
      false, false, flav, x, scaleNum, flavDen, xDen, sDen);
  }

  // Done
  return w;
}

//--------------------------------------------------------------------------

// Function to return the no-emission probability part of the CKKWL weight.

double DireHistory::weightEmissions( PartonLevel* trial, int type,
  int njetMin, int njetMax, double maxscale ) {

  // Use correct scale
  double newScale = scale;
  // For ME state, just multiply by PDF ratios

  if ( !mother ) return 1.0;
  // Recurse
  double w = mother->weightEmissions(trial,type,njetMin,njetMax,newScale);
  // Do nothing for empty state
  if (state.size() < 3) return 1.0;
  // If up to now, trial shower was not successful, return zero
  if ( w < 1e-12 ) return 0.0;
  // If this node has too many jets, no not calculate no-emission probability.
  int njetNow = mergingHooksPtr->getNumberOfClusteringSteps( state) ;
  if (njetMax > -1 && njetNow >= njetMax) return 1.0;
  if (njetMin > -1 && njetNow < njetMin ) w *= 1.0;
  // Do trial shower on current state, return zero if not successful
  else w *= doTrialShower(trial, type, maxscale).front();

  if ( abs(w) < 1e-12 ) return 0.0;
  // Done
  return w;

}

//--------------------------------------------------------------------------

// Function to return the no-emission probability part of the CKKWL weight.

vector<double> DireHistory::weightEmissionsVec( PartonLevel* trial, int type,
  int njetMin, int njetMax, double maxscale ) {

  // Use correct scale
  double newScale = scale;

  // Done if at the highest multiplicity node.
  if (!mother) return createvector<double>(1.)(1.)(1.);

  // Recurse
  vector<double> w = mother->weightEmissionsVec(trial, type, njetMin, njetMax,
    newScale);
  // Do nothing for empty state
  if (state.size() < 3) return createvector<double>(1.)(1.)(1.);
  // If up to now, trial shower was not successful, return zero
  bool nonZero = false;
  for (size_t i=0; i < w.size(); ++i) if (abs(w[i]) > 1e-12) nonZero = true;
  if (!nonZero) return createvector<double>(0.)(0.)(0.);
  // If this node has too many jets, no not calculate no-emission probability.
  int njetNow = mergingHooksPtr->getNumberOfClusteringSteps(state);
  if (njetMax > -1 && njetNow >= njetMax)
    return createvector<double>(1.)(1.)(1.);

  // Do nothing for too few jets.
  if (njetMin > -1 && njetNow < njetMin ) ;
  // Do trial shower on current state, return zero if not successful
  else {
    vector<double> wem = doTrialShower(trial, type, maxscale);
    for (size_t i=0; i < w.size(); ++i) w[i] *= wem[i];
  }

  nonZero = false;
  for (size_t i=0; i < w.size(); ++i) if (abs(w[i]) > 1e-12) nonZero = true;
  if (!nonZero) return createvector<double>(0.)(0.)(0.);

  // Done
  return w;

}

//--------------------------------------------------------------------------

// Function to generate the O(\alpha_s)-term of the CKKWL-weight.

double DireHistory::weightFirst(PartonLevel* trial, double as0, double muR,
  double maxscale, AlphaStrong * asFSR, AlphaStrong * asISR, Rndm* rndmPtr ) {

  // Use correct scale
  double newScale = scale;

  if ( !mother ) {

    double wt = 0.;

    // Calculate PDF first leg
    if (state[3].colType() != 0) {
      // Find x value and flavour
      double x = 2.*state[3].e() / state[0].e();
      int flav = state[3].id();
      // Find numerator/denominator scale
      double scaleNum = (children.empty()) ? hardFacScale(state) : maxscale;
      double scaleDen = mergingHooksPtr->muFinME();
      // Monte Carlo integrand.
      double intPDF4 = monteCarloPDFratios(flav, x, scaleNum, scaleDen,
                         mergingHooksPtr->muFinME(), as0, rndmPtr);
      wt += intPDF4;
    }

    // Calculate PDF ratio for second leg
    if (state[4].colType() != 0) {
      // Find x value and flavour
      double x = 2.*state[4].e() / state[0].e();
      int flav = state[4].id();
      // Find numerator/denominator scale
      double scaleNum = (children.empty()) ? hardFacScale(state) : maxscale;
      double scaleDen = mergingHooksPtr->muFinME();
      // Monte Carlo integrand.
      double intPDF4 = monteCarloPDFratios(flav, x, scaleNum, scaleDen,
                         mergingHooksPtr->muFinME(), as0, rndmPtr);
      wt += intPDF4;
    }

    return wt;
  }

  // Recurse
  double w = mother->weightFirst(trial, as0, muR, newScale, asFSR, asISR,
               rndmPtr );

  // Do nothing for empty state
  if (state.size() < 3) return 0.0;

  // Find right scale
  double b = 1.;
  double asScale2 = newScale*newScale;
  int showerType = (mother->state[clusterIn.emittor].isFinal() ) ? 1 : -1;
  if (showerType == -1) asScale2 += pow(mergingHooksPtr->pT0ISR(),2);

  // Directly get argument of running alpha_s from shower plugin.
  asScale2 = getShowerPluginScale(mother->state, clusterIn.emittor,
    clusterIn.emtPos(), clusterIn.recoiler, clusterIn.name(),
    "scaleAS", asScale2);

  // Find summand beta_0 / 2 * ln(muR^2/t_i) due to as expansion.
  double NF = 4.;
  double BETA0 = 11. - 2./3.* NF;
  // For fixed \alpha_s in matrix element
  w += as0 / (2.*M_PI) * 0.5 * BETA0 * log( (muR*muR) / (b*asScale2) );

  // Count emissions: New variant
  // Generate true average, not only one-point.
  bool fixpdf = true;
  bool fixas  = true;
  double nWeight1 = 0.;
  double nWeight2 = 0.;

  for(int i=0; i < NTRIAL; ++i) {
    // Get number of emissions
    vector<double> unresolvedEmissionTerm = countEmissions(trial, maxscale,
      newScale, 2, as0, asFSR, asISR, 3, fixpdf, fixas);
    nWeight1 += unresolvedEmissionTerm[1];
  }
  w += nWeight1/double(NTRIAL) + nWeight2/double(NTRIAL);

  // Calculate pdf ratios: Get both sides of event
  int inP = 3;
  int inM = 4;
  int sideP = (mother->state[inP].pz() > 0) ? 1 :-1;
  int sideM = (mother->state[inM].pz() > 0) ? 1 :-1;

  if ( mother->state[inP].colType() != 0 ) {
    // Find x value and flavour
    double x = getCurrentX(sideP);
    int flav = getCurrentFlav(sideP);
    // Find numerator scale
    double scaleNum = (children.empty()) ? hardFacScale(state) : maxscale;
    // Monte Carlo integrand.
    double intPDF4 = monteCarloPDFratios(flav, x, scaleNum, newScale,
                       mergingHooksPtr->muFinME(), as0, rndmPtr);
    w += intPDF4;

  }

  if ( mother->state[inM].colType() != 0 ) {
    // Find x value and flavour
    double x = getCurrentX(sideM);
    int flav = getCurrentFlav(sideM);
    // Find numerator scale
    double scaleNum = (children.empty()) ? hardFacScale(state) : maxscale;
    // Monte Carlo integrand.
    double intPDF4 = monteCarloPDFratios(flav, x, scaleNum, newScale,
                       mergingHooksPtr->muFinME(), as0, rndmPtr);
    w += intPDF4;

  }

  // Done
  return w;

}

//--------------------------------------------------------------------------

// Function to generate the O(\alpha_s)-term of the \alpha_s-ratios
// appearing in the CKKWL-weight.

double DireHistory::weightFirstALPHAS( double as0, double muR,
  AlphaStrong * asFSR, AlphaStrong * asISR ) {

  // Use correct scale
  double newScale = scale;
  // Done
  if ( !mother ) return 0.;
  // Recurse
  double w = mother->weightFirstALPHAS( as0, muR, asFSR, asISR );
  // Find right scale
  int showerType = (mother->state[clusterIn.emittor].isFinal() ) ? 1 : -1;
  double b = 1.;
  double asScale = pow2( newScale );
  if ( mergingHooksPtr->unorderedASscalePrescip() == 1 )
    asScale = pow2( clusterIn.pT() );
  if (showerType == -1)
    asScale += pow2( mergingHooksPtr->pT0ISR() );

  // Directly get argument of running alpha_s from shower plugin.
  asScale = getShowerPluginScale(mother->state, clusterIn.emittor,
    clusterIn.emtPos(), clusterIn.recoiler, clusterIn.name(),
    "scaleAS", asScale);

  // Find summand beta_0 / 2 * ln(muR^2/t_i) due to as expansion.
  double NF = 4.;
  double BETA0 = 11. - 2./3.* NF;
  // For fixed \alpha_s in matrix element
  w += as0 / (2.*M_PI) * 0.5 * BETA0 * log( (muR*muR) / (b*asScale) );

  // Done
  return w;

}

//--------------------------------------------------------------------------

// Function to generate the O(\alpha_s)-term of the PDF-ratios
// appearing in the CKKWL-weight.

double DireHistory::weightFirstPDFs( double as0, double maxscale,
  double pdfScale, Rndm* rndmPtr ) {

  // Use correct scale
  double newScale = scale;

  if ( !mother ) {

    double wt = 0.;

    // Calculate PDF first leg
    if (state[3].colType() != 0) {
      // Find x value and flavour
      double x        = 2.*state[3].e() / state[0].e();
      int flav        = state[3].id();
      // Find numerator/denominator scale
      double scaleNum = (children.empty()) ? hardFacScale(state) : maxscale;
      double scaleDen = mergingHooksPtr->muFinME();
      // Monte Carlo integrand.
      wt += monteCarloPDFratios(flav, x, scaleNum, scaleDen,
                          mergingHooksPtr->muFinME(), as0, rndmPtr);
    }
    // Calculate PDF ratio for second leg
    if (state[4].colType() != 0) {
      // Find x value and flavour
      double x        = 2.*state[4].e() / state[0].e();
      int flav        = state[4].id();
      // Find numerator/denominator scale
      double scaleNum = (children.empty()) ? hardFacScale(state) : maxscale;
      double scaleDen = mergingHooksPtr->muFinME();
      // Monte Carlo integrand.
      wt += monteCarloPDFratios(flav, x, scaleNum, scaleDen,
                         mergingHooksPtr->muFinME(), as0, rndmPtr);
    }

    // Done
    return wt;
  }

  // Remember new PDF scale n case true scale should be used for un-ordered
  // splittings.
  double newPDFscale = newScale;
  if (mergingHooksPtr->unorderedPDFscalePrescip() == 1)
    newPDFscale      = clusterIn.pT();

  // Recurse
  double w = mother->weightFirstPDFs( as0, newScale, newPDFscale, rndmPtr);

  // Calculate pdf ratios: Get both sides of event
  int inP   = 3;
  int inM   = 4;
  int sideP = (mother->state[inP].pz() > 0) ? 1 :-1;
  int sideM = (mother->state[inM].pz() > 0) ? 1 :-1;

  if ( mother->state[inP].colType() != 0 ) {
    // Find x value and flavour
    double x        = getCurrentX(sideP);
    int flav        = getCurrentFlav(sideP);
    // Find numerator / denominator scales
    double scaleNum = (children.empty())
                    ? hardFacScale(state)
                    : ( (mergingHooksPtr->unorderedPDFscalePrescip() == 1)
                      ? pdfScale : maxscale );
    double scaleDen = (mergingHooksPtr->unorderedPDFscalePrescip() == 1)
                    ? clusterIn.pT() : newScale;
    // Monte Carlo integrand.
    w += monteCarloPDFratios(flav, x, scaleNum, scaleDen,
                        mergingHooksPtr->muFinME(), as0, rndmPtr);
  }

  if ( mother->state[inM].colType() != 0 ) {
    // Find x value and flavour
    double x        = getCurrentX(sideM);
    int flav        = getCurrentFlav(sideM);
    // Find numerator / denominator scales
    double scaleNum = (children.empty())
                    ? hardFacScale(state)
                    : ( (mergingHooksPtr->unorderedPDFscalePrescip() == 1)
                      ? pdfScale : maxscale );
    double scaleDen = (mergingHooksPtr->unorderedPDFscalePrescip() == 1)
                    ? clusterIn.pT() : newScale;
    // Monte Carlo integrand.
    w += monteCarloPDFratios(flav, x, scaleNum, scaleDen,
                        mergingHooksPtr->muFinME(), as0, rndmPtr);
  }

  // Done
  return w;

}


//--------------------------------------------------------------------------

// Function to generate the O(\alpha_s)-term of the no-emission
// probabilities appearing in the CKKWL-weight.

double DireHistory::weightFirstEmissions(PartonLevel* trial, double as0,
  double maxscale, AlphaStrong * asFSR, AlphaStrong * asISR,
  bool fixpdf, bool fixas ) {

  // Use correct scale
  double newScale = scale;
  if ( !mother ) return 0.0;
  // Recurse
  double w = mother->weightFirstEmissions(trial, as0, newScale, asFSR, asISR,
                                          fixpdf, fixas );
  // Do nothing for empty state
  if (state.size() < 3) return 0.0;
  // Generate true average.
  double nWeight1 = 0.;
  double nWeight2 = 0.;
  for(int i=0; i < NTRIAL; ++i) {
    // Get number of emissions
    vector<double> unresolvedEmissionTerm = countEmissions(trial, maxscale,
      newScale, 2, as0, asFSR, asISR, 3, fixpdf, fixas);
    nWeight1 += unresolvedEmissionTerm[1];
  }

  w += nWeight1/double(NTRIAL) + nWeight2/double(NTRIAL);

  // Done
  return w;

}

//--------------------------------------------------------------------------

// Function to return the factorisation scale of the hard process in Pythia.

double DireHistory::hardFacScale(const Event& event) {

  // Declare output scale.
  double hardscale = 0.;
  // If scale should not be reset, done.
  if ( !mergingHooksPtr->resetHardQFac() ) return mergingHooksPtr->muF();

  // For pure QCD dijet events, calculate the hadronic cross section
  // of the hard process at the pT of the dijet system, rather than at fixed
  // arbitrary scale.
  if ( mergingHooksPtr->getProcessString().compare("pp>jj") == 0
    || mergingHooksPtr->getProcessString().compare("pp>aj") == 0
    || isQCD2to2(event)) {
    // Find the mT in the hard sub-process.
    vector <double> mT;
    for ( int i=0; i < event.size(); ++i)
      if ( event[i].isFinal() && event[i].colType() != 0 )
        mT.push_back( abs(event[i].mT2()) );
    if ( int(mT.size()) != 2 )
      hardscale = infoPtr->QFac();
    else
      hardscale = sqrt( min( mT[0], mT[1] ) );

  // For DIS, set the hard process scale to Q2.
  } else if ( mergingHooksPtr->getProcessString().compare("e+p>e+j") == 0
           || mergingHooksPtr->getProcessString().compare("e-p>e-j") == 0) {
    // Use Q2 as core scale.
    if ( isDIS2to2(event)) {
      int iInEl(0), iOutEl(0);
      for ( int i=0; i < event.size(); ++i )
        if ( event[i].idAbs() == 11 ) {
          if ( event[i].status() == -21 ) iInEl  = i;
          if ( event[i].isFinal() )       iOutEl = i;
        }
      hardscale = sqrt( -(event[iInEl].p()-event[iOutEl].p()).m2Calc() );

    // Use pT2 as core scale.
    } else if (isMassless2to2(event)) {

      // Find the mT in the hard sub-process.
      vector <double> mT;
      for ( int i=0; i < event.size(); ++i)
        if ( event[i].isFinal() && event[i].colType() != 0 )
          mT.push_back( abs(event[i].mT2()) );
      if ( int(mT.size()) != 2 )
        hardscale = infoPtr->QFac();
      else
        hardscale = sqrt( min( mT[0], mT[1] ) );

    } else hardscale = mergingHooksPtr->muF();

  } else {
    hardscale = mergingHooksPtr->muF();
  }
  // Done
  return hardscale;
}

//--------------------------------------------------------------------------

// Function to return the factorisation scale of the hard process in Pythia.

double DireHistory::hardRenScale(const Event& event) {
  // Declare output scale.
  double hardscale = 0.;
  // If scale should not be reset, done.
  if ( !mergingHooksPtr->resetHardQRen() ) return mergingHooksPtr->muR();
  // For pure QCD dijet events, calculate the hadronic cross section
  // of the hard process at the pT of the dijet system, rather than at fixed
  // arbitrary scale.
  if ( mergingHooksPtr->getProcessString().compare("pp>jj") == 0
       || mergingHooksPtr->getProcessString().compare("pp>aj") == 0
       || isQCD2to2(event)) {
    // Find the mT in the hard sub-process.
    vector <double> mT;
    for ( int i=0; i < event.size(); ++i)
      if ( event[i].isFinal()
        && ( event[i].colType() != 0 || event[i].id() == 22 ) )
        mT.push_back( abs(event[i].mT()) );
    if ( int(mT.size()) != 2 )
      hardscale = infoPtr->QRen();
    else
      hardscale = sqrt( mT[0]*mT[1] );

  // For DIS, set the hard process scale to Q2.
  } else if ( mergingHooksPtr->getProcessString().compare("e+p>e+j") == 0
           || mergingHooksPtr->getProcessString().compare("e-p>e-j") == 0) {
    // Use Q2 as core scale.
    if ( isDIS2to2(event)) {
      int iInEl(0), iOutEl(0);
      for ( int i=0; i < state.size(); ++i )
        if ( state[i].idAbs() == 11 ) {
          if ( state[i].status() == -21 ) iInEl  = i;
          if ( state[i].isFinal() )       iOutEl = i;
        }
      hardscale = sqrt( -(state[iInEl].p()-state[iOutEl].p()).m2Calc() );

    // Use pT2 as core scale.
    } else if (isMassless2to2(event)) {

      // Find the mT in the hard sub-process.
      vector <double> mT;
      for ( int i=0; i < event.size(); ++i)
        if ( event[i].isFinal() && event[i].colType() != 0 )
          mT.push_back( abs(event[i].mT2()) );
      if ( int(mT.size()) != 2 )
        hardscale = infoPtr->QFac();
      else
        hardscale = sqrt( min( mT[0], mT[1] ) );

    } else hardscale = mergingHooksPtr->muF();

  } else {
    hardscale = mergingHooksPtr->muR();
  }
  // Done
  return hardscale;
}

//--------------------------------------------------------------------------

// Function to return the factorisation scale of the hard process in Pythia.

double DireHistory::hardStartScale(const Event& event) {

  // Starting scale of initial state showers.
  map<string,double> stateVarsISR;

  if ( showers && showers->spacePtr) stateVarsISR
    = showers->spacePtr->getStateVariables(event,0,0,0,"");
  if (!showers && isr) stateVarsISR
    = isr->getStateVariables(event,0,0,0,"");

  // Starting scale of final state showers.
  map<string,double> stateVarsFSR;
  if ( showers && showers->timesPtr ) stateVarsFSR
    = showers->timesPtr->getStateVariables(event,0,0,0,"");
  if (!showers && fsr) stateVarsFSR
    = fsr->getStateVariables(event,0,0,0,"");

  // Find maximal scale.
  double hardscale = 0.;
  for ( map<string,double>::iterator it = stateVarsISR.begin();
    it != stateVarsISR.end(); ++it )
    if ( it->first.find("scalePDF") != string::npos )
      hardscale = max( hardscale, sqrt(it->second) );
  for ( map<string,double>::iterator it = stateVarsFSR.begin();
    it != stateVarsFSR.end(); ++it )
    if ( it->first.find("scalePDF") != string::npos )
      hardscale = max( hardscale, sqrt(it->second) );

  // Done
  return hardscale;
}

//--------------------------------------------------------------------------

// Perform a trial shower using the pythia object between
// maxscale down to this scale and return the corresponding Sudakov
// form factor.
// IN  trialShower : Shower object used as trial shower
//     double     : Maximum scale for trial shower branching
// OUT  0.0       : trial shower emission outside allowed pT range
//      1.0       : trial shower successful (any emission was below
//                  the minimal scale )

vector<double> DireHistory::doTrialShower( PartonLevel* trial, int type,
  double maxscaleIn, double minscaleIn ) {

  // Copy state to local process
  Event process        = state;
  // Set starting scale.
  double startingScale = maxscaleIn;
  // Careful when setting shower starting scale for pure QCD and prompt
  // photon case.
  if ( mergingHooksPtr->getNumberOfClusteringSteps(process) == 0
    && ( mergingHooksPtr->getProcessString().compare("pp>jj") == 0
         || mergingHooksPtr->getProcessString().compare("pp>aj") == 0
         || isQCD2to2(state) ) )
      startingScale = min( startingScale, hardFacScale(process) );

  // For DIS, set starting scale to Q2 or pT2.
  if ( mergingHooksPtr->getNumberOfClusteringSteps(process) == 0
    && ( mergingHooksPtr->getProcessString().compare("e+p>e+j") == 0
      || mergingHooksPtr->getProcessString().compare("e-p>e-j") == 0))
      //startingScale = min( startingScale, hardFacScale(process) );
      startingScale = hardFacScale(process);

  if ( mergingHooksPtr->getNumberOfClusteringSteps(process) == 0 )
    startingScale = hardStartScale(process);

  // Set output.
  double wt            = 1.;
  vector <double> wtv(createvector<double>(1.)(1.)(1.));
  int nFSRtry(0), nISRtry(0), nMPItry(0);

  while (true) {

    // Reset trialShower object
    psweights->reset();
    trial->resetTrial();
    // Construct event to be showered
    Event event = Event();
    event.init("(hard process-modified)", particleDataPtr);
    event.clear();

    // Reset process scale so that shower starting scale is correctly set.
    process.scale(startingScale);
    //doVeto = false;

    // Get pT before reclustering
    double minScale = (minscaleIn > 0.) ? minscaleIn : scale;

    mergingHooksPtr->setShowerStoppingScale(minScale);

    // Give up generating no-MPI probability if ISR completely dominates.
    //if (type == -1 && nFSRtry+nISRtry > 500) {doVeto=false; break;}
    if (type == -1 && nFSRtry+nISRtry > 500) { break;}

    // If the maximal scale and the minimal scale coincide (as would
    // be the case for the corrected scales of unordered histories),
    // do not generate Sudakov
    if (minScale >= startingScale) break;

    // Find z and pT values at which the current state was formed, to
    // ensure that the showers can order the next emission correctly in
    // rapidity, if required.
    // NOT CORRECTLY SET FOR HIGHEST MULTIPLICITY STATE!
    double z = ( mergingHooksPtr->getNumberOfClusteringSteps(state) == 0
               || !mother )
             ? 0.5
             : mother->getCurrentZ(clusterIn.emittor,clusterIn.recoiler,
                 clusterIn.emtPos(), clusterIn.flavRadBef);
    // Store z and pT values at which the current state was formed.
    infoPtr->zNowISR(z);
    infoPtr->pT2NowISR(pow(startingScale,2));
    infoPtr->hasHistory(true);

    // Perform trial shower emission
    trial->next(process,event);
    // Get trial shower pT.
    double pTtrial   = trial->pTLastInShower();
    int typeTrial    = trial->typeLastInShower();

    if      (typeTrial == 1) nMPItry++;
    else if (typeTrial == 2) nISRtry++;
    else                     nFSRtry++;

    // Clear parton systems.
    trial->resetTrial();

    double t = (pTtrial <= 0.) ? pow2(minScale) : pow2(pTtrial);
    pair<double,double> wtShower = psweights->getWeight(t);
    pair<double,double> wt_isr_1 = psweights->getWeight
      (t, "Variations:muRisrDown");
    pair<double,double> wt_isr_2 = psweights->getWeight
      (t, "Variations:muRisrUp");
    pair<double,double> wt_fsr_1 = psweights->getWeight
      (t, "Variations:muRfsrDown");
    pair<double,double> wt_fsr_2 = psweights->getWeight
      (t, "Variations:muRfsrUp");

    double enhancement = 1.;
    if ( pTtrial > minScale) enhancement
      = psweights->getTrialEnhancement( pow2(pTtrial));
    psweights->reset();
    if (pTtrial>0.) psweights->init();
    psweights->clearTrialEnhancements();

    // Get veto (merging) scale value
    double vetoScale  = (mother) ? 0. : mergingHooksPtr->tms();
    // Get merging scale in current event
    double tnow = mergingHooksPtr->tmsNow( event );

    // Done if evolution scale has fallen below minimum
    if ( pTtrial < minScale ) {
      wt     *= wtShower.second;
      wtv[0] *= wtShower.second;
      wtv[1] *= wt_isr_1.second*wt_fsr_1.second;
      wtv[2] *= wt_isr_2.second*wt_fsr_2.second;
      break;
    }

    // Reset starting scale.
    startingScale = pTtrial;

    // Continue if this state is below the veto scale
    if ( tnow < vetoScale && vetoScale > 0. ) continue;

    // Retry if the trial emission was not allowed.
    if ( mergingHooksPtr->canVetoTrialEmission()
      && mergingHooksPtr->doVetoTrialEmission( process, event) ) continue;

    int iRecAft = event.size() - 1;
    int iEmt    = event.size() - 2;
    int iRadAft = event.size() - 3;
    if ( (event[iRecAft].status() != 52 && event[iRecAft].status() != -53) ||
         event[iEmt].status() != 51 || event[iRadAft].status() != 51)
      iRecAft = iEmt = iRadAft = -1;
    for (int i = event.size() - 1; i > 0; i--) {
      if      (iRadAft == -1 && event[i].status() == -41) iRadAft = i;
      else if (iEmt    == -1 && event[i].status() ==  43) iEmt    = i;
      else if (iRecAft == -1 && event[i].status() == -42) iRecAft = i;
      if (iRadAft != -1 && iEmt != -1 && iRecAft != -1) break;
    }

    // Check if the splitting occured in a small window around a flavour
    // threshold.
    bool onCthreshold(false), onBthreshold(false);
    if (process[3].colType() != 0 || process[4].colType() != 0 ) {
      bool usePDFalphas
        = infoPtr->settingsPtr->flag("ShowerPDF:usePDFalphas");
      BeamParticle* beam = (particleDataPtr->isHadron(beamA.id())) ? &beamA
                         : (particleDataPtr->isHadron(beamB.id())) ? &beamB
                                                        : nullptr;
      double m2cPhys     = (usePDFalphas) ? pow2(max(0.,beam->mQuarkPDF(4)))
                         : mergingHooksPtr->AlphaS_ISR()->muThres2(4);
      double m2bPhys     = (usePDFalphas) ? pow2(max(0.,beam->mQuarkPDF(5)))
                         : mergingHooksPtr->AlphaS_ISR()->muThres2(5);
      if ( event[iEmt].idAbs() == 4 && minScale < sqrt(m2cPhys)
        && pTtrial > (1. - MCWINDOW)*sqrt(m2cPhys)
        && pTtrial < (1. + MCWINDOW)*sqrt(m2cPhys)) onCthreshold = true;
      if ( event[iEmt].idAbs() == 5 && minScale < sqrt(m2bPhys)
        && pTtrial > (1. - MBWINDOW)*sqrt(m2bPhys)
        && pTtrial < (1. + MBWINDOW)*sqrt(m2bPhys)) onBthreshold = true;
    }

    // Only consider allowed emissions for veto:
    // Only allow MPI for MPI no-emission probability.
    if ( type == -1 && typeTrial != 1 ) {
      // If an initial-state splitting occured because of a flavour threshold,
      // then the showers will always win competition against MPI, meaning that
      // no MPI emission will be produced, i.e. the no-MPI-probability = 1
      //if (onCthreshold || onBthreshold) { doVeto=false; break; }
      if (onCthreshold || onBthreshold) { break; }
      continue;
    }
    // Only allow ISR or FSR for radiative no-emission probability.
    if ( type ==  1 && !(typeTrial == 2 || typeTrial >= 3) ) continue;

    if (pTtrial > minScale) {
      wt     *= wtShower.first*wtShower.second * (1.-1./enhancement);
      wtv[0] *= wtShower.first*wtShower.second * (1.-1./enhancement);
      wtv[1] *= wt_isr_1.first*wt_isr_1.second*wt_fsr_1.first*wt_fsr_1.second
                *(1.-1./enhancement);
      wtv[2] *= wt_isr_2.first*wt_isr_2.second*wt_fsr_2.first*wt_fsr_2.second
                *(1.-1./enhancement);
    }
    if (wt == 0.) break;

    if (pTtrial > minScale) continue;

    // For 2 -> 2 pure QCD state, do not allow multiparton interactions
    // above the kinematical pT of the 2 -> 2 state.
    if ( type == -1
      && typeTrial == 1
      && mergingHooksPtr->getNumberOfClusteringSteps(process) == 0
      && ( mergingHooksPtr->getProcessString().compare("pp>jj") == 0
        || mergingHooksPtr->getProcessString().compare("pp>aj") == 0
           || isQCD2to2(state))
      && pTtrial > hardFacScale(process) )
      return createvector<double>(0.)(0.)(0.);

    // Done
    break;

  }

  // Reset trialShower object
  psweights->reset();
  trial->resetTrial();

  // Done
  return wtv;

}

//--------------------------------------------------------------------------

// Assume we have a vector of i elements containing indices into
// another vector with N elements. Update the indices so that all
// unique combinations (starting from 0,1,2,3, ...) are
// covered. Return false when all combinations have been ehausted.

bool DireHistory::updateind(vector<int> & ind, int i, int N) {
  if ( i < 0 ) return false;
  if ( ++ind[i] < N ) return true;
  if ( !updateind(ind, i - 1, N - 1) ) return false;
  ind[i] = ind[i - 1] + 1;
  return true;
}

//--------------------------------------------------------------------------

// Return the expansion of the no-emission probability up to the Nth
// term. Optionally calculate the the terms using fixed alphaS
// and/or PDF ratios.

vector<double>
DireHistory::countEmissions(PartonLevel* trial, double maxscale,
                        double minscale, int showerType, double as0,
                        AlphaStrong * asFSR, AlphaStrong * asISR, int N = 1,
                        bool fixpdf = true, bool fixas = true) {

  if ( N < 0 ) return vector<double>();
  vector<double> result(N+1);
  result[0] = 1.0;
  if ( N < 1 ) return result;

  // Copy state to local process
  Event process = state;

  double startingScale   = maxscale;
  // Careful when setting shower starting scale for pure QCD and prompt
  // photon case.
  if ( mergingHooksPtr->getNumberOfClusteringSteps(process) == 0
    && ( mergingHooksPtr->getProcessString().compare("pp>jj") == 0
      || mergingHooksPtr->getProcessString().compare("pp>aj") == 0
         || isQCD2to2(state) ) )
      startingScale = min( startingScale, hardFacScale(process) );

  vector<double> wts;
  bool canEnhanceTrial = trial->canEnhanceTrial();

  while ( true ) {

    // Reset trialShower object
    psweights->reset();
    trial->resetTrial();
    // Construct event to be showered
    Event event = Event();
    event.init("(hard process-modified)", particleDataPtr);
    event.clear();

    // Reset process scale
    process.scale(startingScale);

    // If the maximal scale and the minimal scale coincide (as would
    // be the case for the corrected scales of unordered histories),
    // do not generate Sudakov
    if (minscale >= startingScale) return result;

    // Find z and pT values at which the current state was formed, to
    // ensure that the showers can order the next emission correctly in
    // rapidity, if required
    if ( mother ) {
      double z = ( mergingHooksPtr->getNumberOfClusteringSteps(state) == 0)
               ? 0.5
               : mother->getCurrentZ(clusterIn.emittor,clusterIn.recoiler,
                   clusterIn.emtPos());
      // Store z and pT values at which the current state was formed
      infoPtr->zNowISR(z);
      infoPtr->pT2NowISR(pow(startingScale,2));
      infoPtr->hasHistory(true);
    }

    // Perform trial shower emission
    trial->next(process,event);

    // Get trial shower pT
    double pTtrial = trial->pTLastInShower();
    int typeTrial  = trial->typeLastInShower();

    // Clear parton systems.
    trial->resetTrial();

    // Get enhanced trial emission weight.
    double pTEnhanced = trial->getEnhancedTrialPT();
    double wtEnhanced = trial->getEnhancedTrialWeight();
    if ( canEnhanceTrial && pTEnhanced > 0.) pTtrial = pTEnhanced;

    // Get veto (merging) scale value
    double vetoScale  = (mother) ? 0. : mergingHooksPtr->tms();
    // Get merging scale in current event
    double tnow = mergingHooksPtr->tmsNow( event );

    // Save scale of current state.
    startingScale   = pTtrial;
    // If the scale of the current state is below the minimal scale, exit.
    if ( pTtrial < minscale ) break;
    // If this state is below the merging scale, do not count emission.
    if ( tnow < vetoScale && vetoScale > 0. ) continue;
    // Retry if the trial emission was not allowed.
    if ( mergingHooksPtr->canVetoTrialEmission()
      && mergingHooksPtr->doVetoTrialEmission( process, event) ) continue;

    // Set weight of enhanced emission.
    double enhance = (canEnhanceTrial && pTtrial > minscale) ? wtEnhanced : 1.;

    // Check if a new emission should be generated, either because
    // the latest emission was not of the desired kind or if the
    // emission was above the minimal scale
    double alphaSinPS = as0;
    double pdfs = 1.0;

    double asScale2 = pTtrial*pTtrial;
    // Directly get argument of running alpha_s from shower plugin.
    asScale2 = getShowerPluginScale(mother->state, clusterIn.emittor,
      clusterIn.emtPos(), clusterIn.recoiler, clusterIn.name(),
      "scaleAS", asScale2);

    // Initial state splittings.
    if ( (showerType == -1 || showerType == 2) && typeTrial == 2 ) {
      // Get weight to translate to alpha_s at fixed renormalisation scale.
      if ( fixas ) alphaSinPS = (*asISR).alphaS(asScale2);
      // Get weight to translate to PDFs at fixed factorisation scale.
      if ( fixpdf )
        //pdfs = pdfFactor( event, typeTrial, pTtrial,
        //                  mergingHooksPtr->muFinME() );
        pdfs = pdfFactor( process, event, typeTrial, pTtrial,
                          mergingHooksPtr->muFinME() );
    // Final state splittings.
    } else if ( (showerType == 1 || showerType == 2) && typeTrial >= 3 ) {
      // Get weight to translate to alpha_s at fixed renormalisation scale.
      if ( fixas ) alphaSinPS = (*asFSR).alphaS(asScale2);
      // Get weight to translate to PDFs at fixed factorisation scale. Needed
      // for final state splittings with initial state recoiler.
      if ( fixpdf )
        pdfs = pdfFactor( process, event, typeTrial, pTtrial,
                          mergingHooksPtr->muFinME() );
    }

    // Save weight correcting to emission generated with fixed scales.
    if ( typeTrial == 2 || typeTrial >= 3 )
      wts.push_back(as0/alphaSinPS * pdfs * 1./enhance);

  }

  for ( int n = 1; n <= min(N, int(wts.size())); ++n ) {
    vector<int> ind(N);
    for ( int i = 0; i < N; ++i ) ind[i] = i;
    do {
      double x = 1.0;
      for ( int j = 0; j < n; ++j ) x *= wts[ind[j]];
      result[n] += x;
    }  while ( updateind(ind, n - 1, wts.size()) );
    if ( n%2 ) result[n] *= -1.0;
  }

  // Reset trialShower object
  psweights->reset();
  trial->resetTrial();

  // Done
  return result;
}

//--------------------------------------------------------------------------

// Function to integrate PDF ratios between two scales over x and t,
// where the PDFs are always evaluated at the lower t-integration limit

double DireHistory::monteCarloPDFratios(int flav, double x, double maxScale,
         double minScale, double pdfScale, double asME, Rndm* rndmPtr) {

  // Perform numerical integration for PDF ratios
  // Prefactor is as/2PI
  double factor = asME / (2.*M_PI);
  // Scale integration just produces a multiplicative logarithm
  factor *= log(maxScale/minScale);

  // For identical scales, done
  if (factor == 0.) return 0.;

  // Declare constants
  double CF = 4./3.;
  double CA = 3.;
  double NF = 4.;
  double TR = 1./2.;

  double integral = 0.;
  double RN = rndmPtr->flat();

  if (flav == 21) {
    double zTrial = pow(x,RN);
    integral  = -log(x) * zTrial *
                integrand(flav, x, pdfScale, zTrial);
    integral += 1./6.*(11.*CA - 4.*NF*TR)
              + 2.*CA*log(1.-x);
  } else {
    double zTrial = x + RN*(1. - x);
    integral  = (1.-x) *
                integrand(flav, x, pdfScale, zTrial);
    integral += 3./2.*CF
              + 2.*CF*log(1.-x);
  }

  // Done
  return (factor*integral);
}

//--------------------------------------------------------------------------

// Methods used for construction of all histories.

// Check if a ordered (and complete) path has been found in the
// initial node, in which case we will no longer be interested in
// any unordered paths.

bool DireHistory::onlyOrderedPaths() {
  if ( !mother || foundOrderedPath ) return foundOrderedPath;
  return  foundOrderedPath = mother->onlyOrderedPaths();
}

//--------------------------------------------------------------------------

// Check if an allowed (according to user-criterion) path has been found in
// the initial node, in which case we will no longer be interested in
// any forbidden paths.

bool DireHistory::onlyAllowedPaths() {
  if ( !mother || foundAllowedPath ) return foundAllowedPath;
  return foundAllowedPath = mother->onlyAllowedPaths();
}

//--------------------------------------------------------------------------

// When a full path has been found, register it with the initial
// history node.
// IN  History : History to be registered as path
//     bool    : Specifying if clusterings so far were ordered
//     bool    : Specifying if path is complete down to 2->2 process
// OUT true if History object forms a plausible path (eg prob>0 ...)

bool DireHistory::registerPath(DireHistory & l, bool isOrdered,
       bool isAllowed, bool isComplete) {

  // We are not interested in improbable paths.
  if ( l.prodOfProbs <= 0.0)
    return false;
  // We only register paths in the initial node.
  if ( mother ) return mother->registerPath(l, isOrdered,
                         isAllowed, isComplete);

  // Again, we are not interested in improbable paths.
  if ( sumpath == sumpath + l.prodOfProbs )
    return false;
  if ( mergingHooksPtr->canCutOnRecState()
    && foundAllowedPath && !isAllowed )
    return false;
  if ( mergingHooksPtr->orderHistories()
    && foundOrderedPath && !isOrdered ) {
    // Prefer complete or allowed paths to ordered paths.
    if ( (!foundCompletePath && isComplete)
      || (!foundAllowedPath && isAllowed) ) ;
    else return false;
  }

  if ( foundCompletePath && !isComplete)
    return false;
  if ( !mergingHooksPtr->canCutOnRecState()
    && !mergingHooksPtr->allowCutOnRecState() )
    foundAllowedPath = true;

  if ( mergingHooksPtr->canCutOnRecState() && isAllowed && isComplete) {
    if ( !foundAllowedPath || !foundCompletePath ) {
      // If this is the first complete, allowed path, discard the
      // old, disallowed or incomplete ones.
      paths.clear();
      sumpath = 0.0;
    }
    foundAllowedPath = true;

  }

  if ( mergingHooksPtr->orderHistories() && isOrdered && isComplete ) {
    if ( !foundOrderedPath || !foundCompletePath ) {
      // If this is the first complete, ordered path, discard the
      // old, non-ordered or incomplete ones.
      paths.clear();
      sumpath = 0.0;
    }
    foundOrderedPath = true;
    foundCompletePath = true;

  }

  if ( isComplete ) {
    if ( !foundCompletePath ) {
      // If this is the first complete path, discard the old,
      // incomplete ones.
      paths.clear();
      sumpath = 0.0;
    }
    foundCompletePath = true;
  }

  if ( isOrdered ) foundOrderedPath = true;

  // Index path by probability
  sumpath += l.prodOfProbs;
  paths[sumpath] = &l;

  updateProbMax(l.prodOfProbs, isComplete);

  return true;
}

//--------------------------------------------------------------------------

// For one given state, find all possible clusterings.
// IN  Event : state to be investigated
// OUT vector of all (rad,rec,emt) systems in the state

vector<DireClustering> DireHistory::getAllClusterings( const Event& event) {

  vector<DireClustering> ret;
  vector<DireClustering> systems;

  for (int i=0; i < event.size(); ++i) {
    if ( event[i].isFinal() ) {
      for (int j=0; j < event.size(); ++j) {
        if ( i == j) continue;
        bool isInitial = (event[j].status() == -21
                     || event[j].status() == -41 || event[j].status() == -42
                     || event[j].status() == -53
                     || event[j].status() == -31 || event[j].status() == -34);
        if (!isInitial && !event[j].isFinal() ) continue;
        systems = getClusterings( i, j, event);
        ret.insert(ret.end(), systems.begin(), systems.end());
        systems.resize(0);
      }
    }
  }

  // Now remove any clustering that appears more than once.
  vector<int> iRemove;
  for (unsigned int i=0; i < ret.size(); ++i) {
    for (unsigned int j=i; j < ret.size(); ++j) {
      if (i == j) continue;
      if (find(iRemove.begin(), iRemove.end(), j) != iRemove.end()) continue;
      if ( equalClustering(ret[i], ret[j])) iRemove.push_back(j);
    }
  }
  sort (iRemove.begin(), iRemove.end());
  for (int i = iRemove.size()-1; i >= 0; --i) {
    ret[iRemove[i]] = ret.back();
    ret.pop_back();
  }

  return ret;
}

//--------------------------------------------------------------------------

// Function to attach (spin-dependent duplicates of) a clustering.

void DireHistory::attachClusterings (vector<DireClustering>& clus, int iEmt,
    int iRad,
    int iRec, int iPartner, double pT, string name, const Event& event) {

  // Do nothing for unphysical clustering.
  if (pT <= 0.) return;

  if ( !mergingHooksPtr->doWeakClustering() ) {

    clus.push_back( DireClustering(iEmt, iRad, iRec, iPartner, pT,
      &event[iRad], &event[iEmt], &event[iRec], name, 0, 0, 0, 0));

  } else {

    // Get ID of radiator before the splitting.
    map<string,double> stateVars;
    bool hasPartonLevel(showers && showers->timesPtr && showers->spacePtr),
         hasShowers(fsr && isr);
    if (hasPartonLevel) {
      bool isFSR = showers->timesPtr->isTimelike(event, iRad, iEmt, iRec, "");
      if (isFSR) stateVars = showers->timesPtr->getStateVariables(event,iRad,
        iEmt,iRec,name);
      else       stateVars = showers->spacePtr->getStateVariables(event,iRad,
        iEmt,iRec,name);
    } else if (hasShowers) {
      bool isFSR = fsr->isTimelike(event, iRad, iEmt, iRec, "");
      if (isFSR) stateVars = fsr->getStateVariables(event,iRad,iEmt,iRec,name);
      else       stateVars = isr->getStateVariables(event,iRad,iEmt,iRec,name);
    }

    // Get flavour of radiator after potential clustering
    int radBeforeFlav = int(stateVars["radBefID"]);

    clus.push_back( DireClustering(iEmt, iRad, iRec, iPartner, pT,
      &event[iRad], &event[iEmt], &event[iRec], name, radBeforeFlav, 0, 0, 0));

  } // doWeakClustering

  return;

}


//--------------------------------------------------------------------------

// Function to construct (rad,rec,emt) triples from the event
// IN  int   : Position of Emitted in event record for which
//             dipoles should be constructed
//     int   : Colour topogy to be tested
//             1= g -> qqbar, causing 2 -> 2 dipole splitting
//             2= q(bar) -> q(bar) g && g -> gg,
//              causing a 2 -> 3 dipole splitting
//     Event : event record to be checked for ptential partners
// OUT vector of all allowed radiator+recoiler+emitted triples

vector<DireClustering> DireHistory::getClusterings (int emt, int rad,
                      const Event& event ) {

  vector<DireClustering> clus;

  // Check if this configuration is result of a splitting.
  bool isFSR(false), isISR(false), hasShowers(fsr && isr),
       hasPartonLevel(showers && showers->timesPtr && showers->spacePtr);
  if (hasPartonLevel) {
    isFSR = showers->timesPtr->allowedSplitting(event, rad, emt);
    isISR = showers->spacePtr->allowedSplitting(event, rad, emt);
  } else if (hasShowers) {
    isFSR = fsr->allowedSplitting(event, rad, emt);
    isISR = isr->allowedSplitting(event, rad, emt);
  }

  if ( isFSR ) {
    vector<string> names = hasPartonLevel
      ? showers->timesPtr->getSplittingName(event,rad,emt,0)
      : hasShowers ? fsr->getSplittingName(event,rad,emt,0) : vector<string>();
    for ( int iName=0; iName < int(names.size()); ++iName) {
      vector<int> recsNow = hasPartonLevel
        ? showers->timesPtr->getRecoilers(event, rad, emt, names[iName])
        : (hasShowers ? fsr->getRecoilers(event, rad, emt, names[iName])
                      : vector<int>());
      for ( int i = 0; i < int(recsNow.size()); ++i ) {
        if ( allowedClustering( rad, emt, recsNow[i], recsNow[i],
          names[iName], event) ) {
          double pT = pTLund(event, rad, emt, recsNow[i], names[iName]);
          attachClusterings (clus, emt, rad, recsNow[i], recsNow[i], pT,
            names[iName], event);
        }
      }
    }
  }

  if ( isISR ) {
    vector<string> names = hasPartonLevel
      ? showers->spacePtr->getSplittingName(event,rad,emt,0)
      : hasShowers ? isr->getSplittingName(event,rad,emt,0) : vector<string>();

    for ( int iName=0; iName < int(names.size()); ++iName) {
      vector<int> recsNow = hasPartonLevel
        ? showers->spacePtr->getRecoilers(event, rad, emt, names[iName])
        : (hasShowers ? isr->getRecoilers(event, rad, emt, names[iName])
                      : vector<int>());
      for ( int i = 0; i < int(recsNow.size()); ++i ) {
        if ( allowedClustering( rad, emt, recsNow[i], recsNow[i],
          names[iName], event) ) {
          attachClusterings (clus, emt, rad, recsNow[i], recsNow[i],
            pTLund(event, rad, emt, recsNow[i], names[iName]),
            names[iName], event);
        }
      }
    }
  }

  // Done
  return clus;
}

//--------------------------------------------------------------------------

// Calculate and return the probability of a clustering.
// IN  Clustering : rad,rec,emt - System for which the splitting
//                  probability should be calcuated
// OUT splitting probability

pair<double,double> DireHistory::getProb(const DireClustering & SystemIn) {

  // Get local copies of input system
  int rad     = SystemIn.radPos();
  int rec     = SystemIn.recPos();
  int emt     = SystemIn.emtPos();
  string name = SystemIn.name();

  // If the splitting resulted in disallowed evolution variable,
  // disallow the splitting
  if (SystemIn.pT() <= 0.) { return make_pair(1.,0.);}

  double pr(0.), coupling(1.);

  bool isFSR(false), isISR(false), hasShowers(fsr && isr),
       hasPartonLevel(showers && showers->timesPtr && showers->spacePtr);
  if (hasPartonLevel) {
    isFSR = showers->timesPtr->isTimelike(state, rad, emt, rec, "");
    isISR = showers->spacePtr->isSpacelike(state, rad, emt, rec, "");
  } else if (hasShowers) {
    isFSR = fsr->isTimelike(state, rad, emt, rec, "");
    isISR = isr->isSpacelike(state, rad, emt, rec, "");
  }

  name += "-0";

  if (isFSR) {

    // Ask shower for splitting probability.
    pr += hasPartonLevel
      ? showers->timesPtr->getSplittingProb( state, rad, emt, rec, name)
      : hasShowers ? fsr->getSplittingProb( state, rad, emt, rec, name) : 0.;

    // Scale with correct coupling factor.
    double mu2Ren = pow2(mergingHooksPtr->muR());
    name=name.substr( 0, name.size()-2);
    coupling      = fsr->getCoupling( mu2Ren, name);

  }

  if (isISR) {

    // Ask shower for splitting probability.
    pr += hasPartonLevel
       ? showers->spacePtr->getSplittingProb( state, rad, emt, rec, name)
       : hasShowers ? isr->getSplittingProb( state, rad, emt, rec, name) : 0.;

    // Scale with correct coupling factor.
    double mu2Ren = pow2(mergingHooksPtr->muR());
    name=name.substr( 0, name.size()-2);
    coupling      = isr->getCoupling( mu2Ren, name);

  }

  // Done.
  return make_pair(coupling,pr);

}

//--------------------------------------------------------------------------

// Set up the beams (fill the beam particles with the correct
// current incoming particles) to allow calculation of splitting
// probability.
// For interleaved evolution, set assignments dividing PDFs into
// sea and valence content. This assignment is, until a history path
// is chosen and a first trial shower performed, not fully correct
// (since content is chosen form too high x and too low scale). The
// assignment used for reweighting will be corrected after trial
// showering

void DireHistory::setupBeams() {

  // Do nothing for empty event, possible if sequence of
  // clusterings was ill-advised in that it results in
  // colour-disconnected states
  if (state.size() < 4) return;

  // Do nothing for e+e- beams
  if ( state[3].colType() == 0 && state[4].colType() == 0 ) return;

  // Incoming partons to hard process are stored in slots 3 and 4.
  int inS = 0;
  int inP = 0;
  int inM = 0;
  for(int i=0;i< int(state.size()); ++i) {
    if (state[i].mother1() == 1) inP = i;
    if (state[i].mother1() == 2) inM = i;
  }

  // Save some info before clearing beams
  // Mothers of incoming partons companion code
  int motherPcompRes = -1;
  int motherMcompRes = -1;

  bool sameFlavP = false;
  bool sameFlavM = false;

  if (mother) {
    int inMotherP = 0;
    int inMotherM = 0;
    for(int i=0;i< int(mother->state.size()); ++i) {
      if (mother->state[i].mother1() == 1) inMotherP = i;
      if (mother->state[i].mother1() == 2) inMotherM = i;
    }
    sameFlavP = (state[inP].id() == mother->state[inMotherP].id());
    sameFlavM = (state[inM].id() == mother->state[inMotherM].id());

    motherPcompRes = (sameFlavP) ? beamA[0].companion() : -2;
    motherMcompRes = (sameFlavM) ? beamB[0].companion() : -2;
  }

  // Append the current incoming particles to the beam
  beamA.clear();
  beamB.clear();

  // Get energy of incoming particles
  double Ep = 2. * state[inP].e();
  double Em = 2. * state[inM].e();

  // If incoming partons are massive then recalculate to put them massless.
  if (state[inP].m() != 0. || state[inM].m() != 0.) {
    Ep = state[inP].pPos() + state[inM].pPos();
    Em = state[inP].pNeg() + state[inM].pNeg();
  }

  // Add incoming hard-scattering partons to list in beam remnants.
  double x1 = Ep / state[inS].m();
  beamA.append( inP, state[inP].id(), x1);
  double x2 = Em / state[inS].m();
  beamB.append( inM, state[inM].id(), x2);

  // Scale. For ME multiplicity history, put scale to mu_F
  // (since sea/valence quark content is chosen from this scale)
  double scalePDF = (mother) ? scale : infoPtr->QFac();
  // Find whether incoming partons are valence or sea. Store.
  // Can I do better, e.g. by setting the scale to the hard process
  // scale (= M_W) or by replacing one of the x values by some x/z??
  beamA.xfISR( 0, state[inP].id(), x1, scalePDF*scalePDF);
  if (!mother) {
    beamA.pickValSeaComp();
  }  else {
    beamA[0].companion(motherPcompRes);
  }
  beamB.xfISR( 0, state[inM].id(), x2, scalePDF*scalePDF);
  if (!mother) {
    beamB.pickValSeaComp();
  } else {
    beamB[0].companion(motherMcompRes);
  }

}

//--------------------------------------------------------------------------

// Calculate the PDF ratio used in the argument of the no-emission
// probability

double DireHistory::pdfForSudakov() {

  // Do nothing for e+e- beams
  if ( state[3].colType() == 0 ) return 1.0;
  if ( state[4].colType() == 0 ) return 1.0;

  // Check if splittings was ISR or FSR
  bool FSR = (   mother->state[clusterIn.emittor].isFinal()
             && mother->state[clusterIn.recoiler].isFinal());
  bool FSRinRec = (   mother->state[clusterIn.emittor].isFinal()
                  && !mother->state[clusterIn.recoiler].isFinal());

  // Done for pure FSR
  if (FSR) return 1.0;

  int iInMother = (FSRinRec)? clusterIn.recoiler : clusterIn.emittor;
  //  Find side of event that was reclustered
  int side = ( mother->state[iInMother].pz() > 0 ) ? 1 : -1;

  int inP = 0;
  int inM = 0;
  for(int i=0;i< int(state.size()); ++i) {
    if (state[i].mother1() == 1) inP = i;
    if (state[i].mother1() == 2) inM = i;
  }

  // Save mother id
  int idMother = mother->state[iInMother].id();
  // Find daughter position and id
  int iDau = (side == 1) ? inP : inM;
  int idDaughter = state[iDau].id();
  // Get mother x value
  double xMother = 2. * mother->state[iInMother].e() / mother->state[0].e();
  // Get daughter x value of daughter
  double xDaughter = 2.*state[iDau].e() / state[0].e(); // x1 before isr

  // Calculate pdf ratio
  double ratio = getPDFratio(side, true, false, idMother, xMother, scale,
                   idDaughter, xDaughter, scale);

  // For FSR with incoming recoiler, maximally return 1.0, as
  // is done in Pythia::TimeShower.
  // For ISR, return ratio
  return ( (FSRinRec)? min(1.,ratio) : ratio);
}

//--------------------------------------------------------------------------

// Calculate the hard process matrix element to include in the selection
// probability.

double DireHistory::hardProcessME( const Event& event ) {

  // Calculate prob for Drell-Yan process.
  if (isEW2to1(event)) {

    // qqbar -> W.
    if (event[5].idAbs() == 24) {
      int idIn1  = event[3].id();
      int idIn2  = event[4].id();
      double mW = particleDataPtr->m0(24);
      double gW = particleDataPtr->mWidth(24) / mW;
      double sH = (event[3].p()+event[4].p()).m2Calc();

      double thetaWRat = 1. / (12. * coupSMPtr->sin2thetaW());
      double ckmW = coupSMPtr->V2CKMid(abs(idIn1), abs(idIn2));

      double bwW = 12. * M_PI / ( pow2(sH - pow2(mW)) + pow2(sH * gW) );
      double preFac = thetaWRat * sqrt(sH) * particleDataPtr->mWidth(24);
      return ckmW * preFac * bwW;
    }

    // qqbar -> Z. No interference with gamma included.
    else if (event[5].idAbs() == 23) {
      double mZ = particleDataPtr->m0(23);
      double gZ = particleDataPtr->mWidth(23) / mZ;
      double sH = (event[3].p()+event[4].p()).m2Calc();
      int flav  = (mother) ? abs(clusterIn.flavRadBef) : event[3].idAbs();
      double thetaZRat =
        (pow2(coupSMPtr->rf( flav )) + pow2(coupSMPtr->lf( flav ))) /
        (24. * coupSMPtr->sin2thetaW() * coupSMPtr->cos2thetaW());
      double bwW = 12. * M_PI / ( pow2(sH - pow2(mZ)) + pow2(sH * gZ) );
      double preFac = thetaZRat * sqrt(sH) * particleDataPtr->mWidth(23);
      return preFac * bwW;
    }

    else {
      string message="Warning in DireHistory::hardProcessME: Only Z/W are";
      message+=" supported as 2->1 processes. Skipping history.";
      infoPtr->errorMsg(message);
      return 0;
    }
  }
  // 2 to 2 process, assume QCD.
    else if (isQCD2to2(event)) {
    int idIn1  = event[3].id();
    int idIn2  = event[4].id();
    int idOut1 = event[5].id();
    int idOut2 = event[6].id();

    double sH = (event[3].p()+event[4].p()).m2Calc();
    double tH = (event[3].p()-event[5].p()).m2Calc();
    double uH = (event[3].p()-event[6].p()).m2Calc();

    // Verify that it is QCD.
    bool isQCD = true;
    if (!(abs(idIn1) < 10 || abs(idIn1) == 21) ) isQCD = false;
    if (!(abs(idIn2) < 10 || abs(idIn2) == 21) ) isQCD = false;
    if (!(abs(idOut1) < 10 || abs(idOut1) == 21) ) isQCD = false;
    if (!(abs(idOut2) < 10 || abs(idOut2) == 21) ) isQCD = false;

    // Overall phase-space constant (dsigma/dcos(theta)).
    //double cor = M_PI / (9. * pow2(sH));
    double cor = 1. / (9. * pow2(sH));

    // Multiply with overall factor (g_s^4) / (16Pi^2) = as^2
    double mu2Ren = pow2(mergingHooksPtr->muR());
    cor *= pow2( mergingHooksPtr->AlphaS_ISR()->alphaS(mu2Ren) );

    // If it is QCD calculate cross section.
    if (isQCD) {
      // Find out which 2->2 process it is.

      // incoming gluon pair.
      if (abs(idIn1) == 21 && abs(idIn2) == 21) {
        if (abs(idOut1) == 21 && abs(idOut2) == 21)
          return cor * weakShowerMEs.getMEgg2gg(sH, tH, uH);
        else return cor * weakShowerMEs.getMEgg2qqbar(sH, tH, uH);

      // Incoming single gluon
      } else if (abs(idIn1) == 21 || abs(idIn2) == 21) {
        if (idIn1 != idOut1) swap(uH, tH);
        return cor * weakShowerMEs.getMEqg2qg(sH, tH, uH);
      }

      // Incoming quarks
      else {
        if (abs(idOut1) == 21 && abs(idOut2) == 21) {
          return cor * weakShowerMEs.getMEqqbar2gg(sH, tH, uH);
        }

        if (idIn1 == -idIn2) {
          if (abs(idIn1) == abs(idOut1)) {
            if (idIn1 != idOut1) swap(uH, tH);
            return cor * weakShowerMEs.getMEqqbar2qqbar(sH, tH, uH, true);
          }
          else {
            return cor * weakShowerMEs.getMEqqbar2qqbar(sH, tH, uH, false);
          }
        }
        else if (idIn1 == idIn2)
          return cor * weakShowerMEs.getMEqq2qq(sH, tH, uH, true);
        else {
          if (idIn1 == idOut1) swap(uH,tH);
          return cor * weakShowerMEs.getMEqq2qq(sH, tH, uH, false);
        }
      }
    }
  }

  // Hard process MEs for DIS.
  if ( isDIS2to2(event) ) {

    //int iIncEl(0), iOutEl(0), iIncP(0), iOutP(0);
    int iIncEl(0), iOutEl(0), iIncP(0);
    for ( int i=0; i < event.size(); ++i ) {
      if ( event[i].idAbs() == 11 ) {
        if ( event[i].status() == -21 ) iIncEl = i;
        if ( event[i].isFinal() )       iOutEl = i;
      }
      if ( event[i].colType() != 0 ) {
        if ( event[i].status() == -21 ) iIncP  = i;
        //if ( event[i].isFinal() )       iOutP  = i;
      }
    }
    Vec4 pgam( event[iIncEl].p() - event[iOutEl].p() );
    Vec4 pprot( (event[iIncP].mother1() == 1) ? event[1].p() : event[2].p() );
    double s  = pow2(event[0].m());
    double Q2 = -pgam.m2Calc();
    double y  = (pprot*pgam) / (pprot*event[iIncEl].p());
    double x  = Q2 / (2.*pprot*pgam);
    double res = 4.*M_PI / (s*pow2(x)*pow2(y))*(1. - y + 0.5*pow2(y));
    return res;

  // 2 to 2 process, assume QCD.
  } else if (isMassless2to2(event)) {
    int idIn1  = event[3].id();
    int idIn2  = event[4].id();
    int idOut1 = event[5].id();
    int idOut2 = event[6].id();

    double sH = (event[3].p()+event[4].p()).m2Calc();
    double tH = (event[3].p()-event[5].p()).m2Calc();
    double uH = (event[3].p()-event[6].p()).m2Calc();

    // Verify that it is QCD.
    int inc1Type = particleDataPtr->colType(idIn1);
    int inc2Type = particleDataPtr->colType(idIn2);
    int out1Type = particleDataPtr->colType(idOut1);
    int out2Type = particleDataPtr->colType(idOut2);
    bool isQCD = (inc1Type*inc2Type*out1Type*out2Type != 0);

    // Overall phase-space constant (dsigma/dcos(theta)).
    double cor = M_PI / (9. * pow2(sH));

    // If it is QCD calculate cross section.
    if (isQCD) {
      // Find out which 2->2 process it is.

      // incoming gluon pair.
      if (abs(idIn1) == 21 && abs(idIn2) == 21) {
        if (abs(idOut1) == 21 && abs(idOut2) == 21)
          return cor * weakShowerMEs.getMEgg2gg(sH, tH, uH);
        else return cor * weakShowerMEs.getMEgg2qqbar(sH, tH, uH);

      // Incoming single gluon
      } else if (abs(idIn1) == 21 || abs(idIn2) == 21) {
        if (idIn1 != idOut1) swap(uH, tH);
        return cor * weakShowerMEs.getMEqg2qg(sH, tH, uH);
      }

      // Incoming quarks
      else {
        if (abs(idOut1) == 21 && abs(idOut2) == 21)
          return cor * weakShowerMEs.getMEqqbar2gg(sH, tH, uH);
        if (idIn1 == -idIn2) {
          if (abs(idIn1) == abs(idOut1)) {
            if (idIn1 != idOut1) swap(uH, tH);
            return cor * weakShowerMEs.getMEqqbar2qqbar(sH, tH, uH, true);
          }
          else {
            return cor * weakShowerMEs.getMEqqbar2qqbar(sH, tH, uH, false);
          }
        }
        else if (idIn1 == idIn2)
          return cor * weakShowerMEs.getMEqq2qq(sH, tH, uH, true);
        else {
          if (idIn1 == idOut1) swap(uH,tH);
          return cor * weakShowerMEs.getMEqq2qq(sH, tH, uH, false);
        }
      }
    }

    // Photon-gluon scattering, use gg->qq~ as proxy.
    if ( (idIn1 == 21 && idIn2 == 22) || (idIn1 == 22 && idIn2 == 21) )
      return cor * weakShowerMEs.getMEgg2qqbar(sH, tH, uH);

    // Photon-quark scattering, use gq->gq as proxy.
    if ( (abs(idIn1) < 10 && idIn2 == 22) || (idIn1 == 22 && abs(idIn2) < 10)){
        if (idIn1 != idOut1) swap(uH, tH);
        return cor * weakShowerMEs.getMEqg2qg(sH, tH, uH);
    }

  }

  // Get hard process.
  string process = mergingHooksPtr->getProcessString();
  double result = 1.;

  if ( process.compare("pp>e+ve") == 0
    || process.compare("pp>e-ve~") == 0
    || process.compare("pp>LEPTONS,NEUTRINOS") == 0 ) {
    // Do nothing for incomplete process.
    int nFinal = 0;
    for ( int i=0; i < int(event.size()); ++i )
      if ( event[i].isFinal() ) nFinal++;
    if ( nFinal != 2 ) return 1.;
    // Get W-boson mass and width.
    double mW = particleDataPtr->m0(24);
    double gW = particleDataPtr->mWidth(24) / mW;
    // Get incoming particles.
    int inP = (event[3].pz() > 0) ? 3 : 4;
    int inM = (event[3].pz() > 0) ? 4 : 3;
    // Get outgoing particles.
    int outP = 0;
    for ( int i=0; i < int(event.size()); ++i ) {
      if ( event[i].isFinal() && event[i].px() > 0 ) outP = i;
    }
    // Get Mandelstam variables.
    double sH = (event[inP].p() + event[inM].p()).m2Calc();
    double tH = (event[inP].p() - event[outP].p()).m2Calc();
    double uH = - sH - tH;

    // Return kinematic part of matrix element.
    result = ( 1. + (tH - uH)/sH ) / ( pow2(sH - mW*mW) + pow2(sH*gW) );
  } else
    result = mergingHooksPtr->hardProcessME(event);

  return result;

}

//--------------------------------------------------------------------------

// Function to return the couplings present in the hard process ME (for correct
// relative normalization of histories with different hard process, coupling
// should be stripped off).

double DireHistory::hardProcessCouplings( const Event& event, int order,
  double scale2, AlphaStrong* alphaS, AlphaEM* alphaEM,
  bool fillCouplCounters, bool with2Pi) {

  vector<int> nwp, nwm, nz, nh, na, nl, nlq, ng, nq, nqb;
  int in1(0), in2(0);
  for (int i=0; i < event.size(); ++i) {
    if (event[i].mother1() == 1 && event[i].mother2() == 0) in1 = i;
    if (event[i].mother1() == 2 && event[i].mother2() == 0) in2 = i;
    if (event[i].isFinal()) {
      if (event[i].id() == 21) ng.push_back(i);
      if (event[i].id() == 22) na.push_back(i);
      if (event[i].id() == 23) nz.push_back(i);
      if (event[i].id() == 24) nwp.push_back(i);
      if (event[i].id() ==-24) nwm.push_back(i);
      if (event[i].id() == 25) nh.push_back(i);
      if (event[i].isLepton()) nl.push_back(i);
      if (event[i].colType() == 1) nq.push_back(i);
      if (event[i].colType() ==-1) nqb.push_back(i);
    }
  }

  double twopi = (with2Pi) ? 2.*M_PI : 1.;
  double as2pi  = (order == 0)
    ? infoPtr->settingsPtr->parm("SigmaProcess:alphaSvalue")/twopi
    : alphaS->alphaS(scale2)/twopi;
  double aem2pi = (order == 0)
    ? infoPtr->settingsPtr->parm("StandardModel:alphaEM0")/twopi
    : alphaEM->alphaEM(scale2)/twopi;

  double result = 1.;
  // One power of aEM for each outgoing photon.
  result *= pow(aem2pi,na.size());
  if (fillCouplCounters) couplingPowCount["qed"]+=na.size();
  // One power of aEM for each outgoing W- and Z-boson.
  result *= pow(aem2pi,nwp.size()+nwm.size()+nz.size());
  if (fillCouplCounters) couplingPowCount["qed"]+=nwp.size()+nwm.size()+
                           nz.size();
  // One power of aS for each outgoing gluon.
  result *= pow(as2pi,ng.size());
  if (fillCouplCounters) couplingPowCount["qcd"]+=ng.size();

  // Couplings for outgoing quarks.
  if (
       (event[in1].colType() == 0 && event[in2].colType() == 0)
    && (nq.size() == 1 && nqb.size() == 1)
    && (event[nq[0]].id() == -event[nqb[0]].id()) ) {
    // Two powers of aEM for single quark pair coupling to incoming
    // lepton pair.
    result *= pow(aem2pi,2.0);
    if (fillCouplCounters) couplingPowCount["qed"]+=2;
  } else if (
       (event[in1].colType() == 0 && event[in2].colType() == 1)
    && (nq.size() == 1 && event[in2].id() == event[nq[0]].id()) ) {
    // Two powers of aEM for eq->eq scattering.
    result *= pow(aem2pi,2.0);
    if (fillCouplCounters) couplingPowCount["qed"]+=2;
  } else if (
       (event[in2].colType() == 0 && event[in1].colType() == 1)
    && (nq.size() == 1 && event[in1].id() == event[nq[0]].id()) ) {
    // Two powers of aEM for eq->eq scattering.
    result *= pow(aem2pi,2.0);
    if (fillCouplCounters) couplingPowCount["qed"]+=2;
  } else if (
       (event[in1].colType() == 0 && event[in2].colType() ==-1)
    && (nqb.size() == 1 && event[in2].id() == event[nqb[0]].id()) ) {
    // Two powers of aEM for eqbar->eqbar scattering.
    result *= pow(aem2pi,2.0);
    if (fillCouplCounters) couplingPowCount["qed"]+=2;
  } else if (
       (event[in2].colType() == 0 && event[in1].colType() ==-1)
    && (nqb.size() == 1 && event[in1].id() == event[nqb[0]].id()) ) {
    // Two powers of aEM for eq->eq scattering.
    result *= pow(aem2pi,2.0);
    if (fillCouplCounters) couplingPowCount["qed"]+=2;
  } else {
    // One power of aS for each outgoing quark/antiquark.
    result *= pow(as2pi,nq.size()+nqb.size());
    if (fillCouplCounters) couplingPowCount["qcd"]+=nq.size()+nqb.size();
  }

  // Coupling for outgoing Higgs to initial state.
  if ( nh.size() > 0 ) {

    double sH = event[nh.front()].m2Calc();
    double mH = sqrt(sH);

    double width = 0.;
    if (event[in1].id() == event[in2].id() && event[in1].id()  == 21)
      width = particleDataPtr->particleDataEntryPtr(25)->resWidthChan(
        mH,21,21)/64;
    else if (event[in1].id() == -event[in2].id() && event[in1].idAbs() < 9)
      width = particleDataPtr->particleDataEntryPtr(25)->resWidthChan(
        mH, event[in1].id(), -event[in1].id()) / 9.;
    else if (event[in1].id() == 21 && event[in2].idAbs() < 9)
      width = max(particleDataPtr->particleDataEntryPtr(25)->resWidthChan(
        mH, 21, 21) / 64,
        particleDataPtr->particleDataEntryPtr(25)->resWidthChan(
        mH, event[in1].id(), -event[in1].id()) / 9.);
    else if (event[in2].id() == 21 && event[in1].idAbs() < 9)
      width = max(particleDataPtr->particleDataEntryPtr(25)->resWidthChan(
        mH, 21, 21) / 64,
        particleDataPtr->particleDataEntryPtr(25)->resWidthChan(
        mH, event[in2].id(), -event[in2].id()) / 9.);

    double m2Res = pow2(particleDataPtr->m0(25));
    double widthTot = particleDataPtr->particleDataEntryPtr(25)->
      resWidth(25,mH);

    // Check if Higgs can couple to final state
    if (width/widthTot < 1e-4) {

      for (int i=0; i < event.size(); ++i) {
        if (i != nh.front() && event[i].isFinal()) {
          int sign = particleDataPtr->hasAnti(event[i].id()) ? -1 : 1;
          double widthNew = particleDataPtr->particleDataEntryPtr(25)->
            resWidthChan( mH, event[i].id(), sign*event[i].id());
          if (event[i].id()  == 21) widthNew /= 64.;
          if (event[i].idAbs() < 9) widthNew /= 9.;
          if (widthNew/widthTot > 1e-4 && widthNew/widthTot > width/widthTot){
            width = widthNew; break;
          }
        }
      }

    }

    // Also remove Breit-Wigner (since contained in clustering probability)
    double sigBW  = 8. * M_PI/ ( pow2(sH - m2Res) + pow2(mH * widthTot) );

    // Discard things with extremely small branching fraction.
    if (width/widthTot < 1e-4) width = 0.;

    double asRatio = (order==0) ? 1.
      : pow2(alphaS->alphaS(scale2)/alphaS->alphaS(125.*125.));
    double res = pow(width*sigBW*asRatio,nh.size());

    result *= res;
    if (fillCouplCounters) {
      couplingPowCount["qcd"]+=2;
      couplingPowCount["heft"]++;
    }
  }

  return result;

}

//--------------------------------------------------------------------------

double DireHistory::hardProcessScale( const Event& event) {

  // Find the mT in the hard sub-process.
  double nFinal(0.), mTprod(1.);
  for ( int i=0; i < event.size(); ++i)
    if ( event[i].isFinal() ) {
      nFinal += 1.;
      mTprod *= abs(event[i].mT());
    }
  double hardScale = (mTprod!=1.) ? pow(mTprod, 1./nFinal) : infoPtr->QRen();

  // Done.
  return hardScale;

}

//--------------------------------------------------------------------------

// Perform the clustering of the current state and return the
// clustered state.
// IN Clustering : rad,rec,emt triple to be clustered to two partons
// OUT clustered state

Event DireHistory::cluster( DireClustering & inSystem ) {

  // Initialise tags of particles to be changed
  int rad     = inSystem.radPos();
  int rec     = inSystem.recPos();
  int emt     = inSystem.emtPos();
  string name = inSystem.name();

  // Construct the clustered event
  Event newEvent = Event();
  newEvent.init("(hard process-modified)", particleDataPtr);
  newEvent.clear();

  bool isFSR(false), hasShowers(fsr && isr),
       hasPartonLevel(showers && showers->timesPtr && showers->spacePtr);
  if (hasPartonLevel) {
    isFSR = showers->timesPtr->isTimelike(state, rad, emt, rec, "");
  } else if (hasShowers) {
    isFSR = fsr->isTimelike(state, rad, emt, rec, "");
  }

  if (isFSR) {
    newEvent = (hasPartonLevel
      ? showers->timesPtr->clustered( state, rad, emt, rec, name)
      : hasShowers ? fsr->clustered( state, rad, emt, rec, name)
                   : newEvent);
  } else {
    newEvent = (hasPartonLevel
      ? showers->spacePtr->clustered( state, rad, emt, rec, name)
      : hasShowers ? isr->clustered( state, rad, emt, rec, name)
                   : newEvent);
  }

  // Store radiator and recoiler positions.
  if (newEvent.size() > 0) {
    inSystem.recBef = newEvent[0].mother2();
    inSystem.radBef = newEvent[0].mother1();
    newEvent[0].mothers(0,0);
  }

  // Done
  return newEvent;
}

//--------------------------------------------------------------------------

// Function to get the flavour of the radiator before the splitting
// for clustering
// IN int  : Flavour of the radiator after the splitting
//    int  : Flavour of the emitted after the splitting
// OUT int : Flavour of the radiator before the splitting

int DireHistory::getRadBeforeFlav(const int RadAfter, const int EmtAfter,
      const Event& event) {

  int type = event[RadAfter].isFinal() ? 1 :-1;
  int emtID  = event[EmtAfter].id();
  int radID  = event[RadAfter].id();
  int emtCOL = event[EmtAfter].col();
  int radCOL = event[RadAfter].col();
  int emtACL = event[EmtAfter].acol();
  int radACL = event[RadAfter].acol();

  bool colConnected = ((type == 1) && ( (emtCOL !=0 && (emtCOL ==radACL))
                                     || (emtACL !=0 && (emtACL ==radCOL)) ))
                    ||((type ==-1) && ( (emtCOL !=0 && (emtCOL ==radCOL))
                                     || (emtACL !=0 && (emtACL ==radACL)) ));
  // QCD splittings
  // Gluon radiation
  if ( emtID == 21 )
    return radID;
  // Final state gluon splitting
  if ( type == 1 && emtID == -radID && !colConnected )
    return 21;
  // Initial state s-channel gluon splitting
  if ( type ==-1 && radID == 21 )
    return -emtID;
  // Initial state t-channel gluon splitting
  if ( type ==-1 && !colConnected && radID != 21 && abs(emtID) < 10
       && abs(radID) < 10)
    return 21;

  // SQCD splittings
  int radSign = (radID < 0) ? -1 : 1;
  int offsetL = 1000000;
  int offsetR = 2000000;
  // Gluino radiation
  if ( emtID == 1000021 ) {
    // Gluino radiation combined with quark yields squark.
    if (abs(radID) < 10 ) {
      int offset = offsetL;
      // Check if righthanded squark present. If so, make the reclustered
      // squark match. Works for squark pair production + gluino emission.
      for (int i=0; i < int(event.size()); ++i)
        if ( event[i].isFinal()
          && event[i].idAbs() < offsetR+10 && event[i].idAbs() > offsetR)
          offset = offsetR;
      return radSign*(abs(radID)+offset);
    }
    // Gluino radiation combined with squark yields quark.
    if (abs(radID) > offsetL && abs(radID) < offsetL+10 )
      return radSign*(abs(radID)-offsetL);
    if (abs(radID) > offsetR && abs(radID) < offsetR+10 )
      return radSign*(abs(radID)-offsetR);
    // Gluino radiation off gluon yields gluino.
    if (radID == 21 ) return emtID;
  }

  int emtSign = (emtID < 0) ? -1 : 1;
  // Get PDG numbering offsets.
  int emtOffset = 0;
  if ( abs(emtID) > offsetL && abs(emtID) < offsetL+10 )
    emtOffset = offsetL;
  if ( abs(emtID) > offsetR && abs(emtID) < offsetR+10 )
    emtOffset = offsetR;
  int radOffset = 0;
  if ( abs(radID) > offsetL && abs(radID) < offsetL+10 )
    radOffset = offsetL;
  if ( abs(radID) > offsetR && abs(radID) < offsetR+10 )
    radOffset = offsetR;

  // Final state gluino splitting
  if ( type == 1 && !colConnected ) {
    // Emitted squark, radiating quark.
    if ( emtOffset > 0 && radOffset == 0
      && emtSign*(abs(emtID) - emtOffset) == -radID )
      return 1000021;
    // Emitted quark, radiating squark.
    if ( emtOffset == 0 && radOffset > 0
      && emtID == -radSign*(abs(radID) - radOffset) )
      return 1000021;
  }

  // Initial state s-channel gluino splitting
  if ( type ==-1 && radID == 1000021 ) {
    // Quark entering underlying hard process.
    if ( emtOffset > 0 ) return -emtSign*(abs(emtID) - emtOffset);
    // Squark entering underlying hard process.
    else return -emtSign*(abs(emtID) + emtOffset);
  }

  // Initial state t-channel gluino splitting.
  if ( type ==-1
    && ( (abs(emtID) > offsetL && abs(emtID) < offsetL+10)
      || (abs(emtID) > offsetR && abs(emtID) < offsetR+10))
    && ( (abs(radID) > offsetL && abs(radID) < offsetL+10)
      || (abs(radID) > offsetR && abs(radID) < offsetR+10))
    && emtSign*(abs(emtID)+emtOffset) == radSign*(abs(radID) - radOffset)
    && !colConnected ) {
    return 1000021;
  }

  // Electroweak splittings splittings
  // Photon / Z radiation: Calculate invariant mass of system
  double m2final = (event[RadAfter].p()+ event[EmtAfter].p()).m2Calc();

  if ( emtID == 22 || emtID == 23 ) return radID;
  // Final state Photon splitting
  if ( type == 1 && emtID == -radID && colConnected && sqrt(m2final) <= 10. )
    return 22;
  // Final state Photon splitting
  if ( type == 1 && emtID == -radID && colConnected && sqrt(m2final)  > 10. )
    return 23;
  // Initial state s-channel photon/ Z splitting
  if ( type ==-1 && (radID == 22 || radID == 23) )
    return -emtID;
  // Initial state t-channel photon / Z splitting: Always bookkeep as photon
  if ( type ==-1 && abs(emtID) < 10 && abs(radID) < 10 && colConnected )
    return 22;

  // W+ radiation
  // Final state W+ splitting
  if ( emtID == 24 && radID < 0 ) return radID + 1;
  if ( emtID == 24 && radID > 0 ) return radID + 1;

  // W- radiation
  // Final state W- splitting
  if ( emtID ==-24 && radID < 0 ) return radID - 1;
  if ( emtID ==-24 && radID > 0 ) return radID - 1;

  // Done.
  return 0;

}

//--------------------------------------------------------------------------

// Function to get the spin of the radiator before the splitting
// IN int  : Spin of the radiator after the splitting
//    int  : Spin of the emitted after the splitting
// OUT int : Spin of the radiator before the splitting

int DireHistory::getRadBeforeSpin(const int radAfter, const int emtAfter,
      const int spinRadAfter, const int spinEmtAfter,
      const Event& event) {

  // Get flavour before the splitting.
  int radBeforeFlav = getRadBeforeFlav(radAfter, emtAfter, event);

  // Gluon in final state g-> q qbar
  if ( event[radAfter].isFinal()
    && event[radAfter].id() == -event[emtAfter].id())
    return (spinRadAfter == 9) ? spinEmtAfter : spinRadAfter;

  // Quark in final state q -> q g
  if ( event[radAfter].isFinal() && abs(radBeforeFlav) < 10
    && event[radAfter].idAbs() < 10)
    // Special oddity: Gluon does not change spin.
    return spinRadAfter;

  // Quark in final state q -> g q
  if ( event[radAfter].isFinal() && abs(radBeforeFlav) < 10
    && event[emtAfter].idAbs() < 10)
    // Special oddity: Gluon does not change spin.
    return spinEmtAfter;

  // Gluon in final state g -> g g
  if ( event[radAfter].isFinal() && radBeforeFlav == 21
    && event[radAfter].id() == 21)
    // Special oddity: Gluon does not change spin.
    return (spinRadAfter == 9) ? spinEmtAfter : spinRadAfter;

  // Gluon in initial state g-> q qbar
  if ( !event[radAfter].isFinal()
    && radBeforeFlav == -event[emtAfter].id())
    return (spinRadAfter == 9) ? spinEmtAfter : spinRadAfter;

  // Quark in initial state q -> q g
  if ( !event[radAfter].isFinal() && abs(radBeforeFlav) < 10
    && event[radAfter].idAbs() < 10)
    // Special oddity: Gluon does not change spin.
    return spinRadAfter;

  // Gluon in initial state q -> g q
  if ( !event[radAfter].isFinal() && radBeforeFlav == 21
    && event[emtAfter].idAbs() < 10)
    // Special oddity: Gluon does not change spin.
    return spinEmtAfter;

  // Done. Return default value.
  return 9;

}

//--------------------------------------------------------------------------

// Function to properly colour-connect the radiator to the rest of
// the event, as needed during clustering
// IN  Particle& : Particle to be connected
//     Particle  : Recoiler forming a dipole with Radiator
//     Event     : event to which Radiator shall be appended
// OUT true               : Radiator could be connected to the event
//     false              : Radiator could not be connected to the
//                          event or the resulting event was
//                          non-valid

bool DireHistory::connectRadiator( Particle& Radiator, const int RadType,
                      const Particle& Recoiler, const int RecType,
                      const Event& event ) {

  // Start filling radiator colour indices with dummy values
  Radiator.cols( -1, -1 );

  // Radiator should always be colour-connected to recoiler.
  // Three cases (rad = Anti-Quark, Quark, Gluon) to be considered
  if ( Radiator.colType() == -1 ) {
    // For final state antiquark radiator, the anticolour is fixed
    // by the final / initial state recoiler colour / anticolour
    if ( RadType + RecType == 2 )
      Radiator.cols( 0, Recoiler.col());
    else if ( RadType + RecType == 0 )
      Radiator.cols( 0, Recoiler.acol());
    // For initial state antiquark radiator, the anticolour is fixed
    // by the colour of the emitted gluon (which will be the
    // leftover anticolour of a final state particle or the leftover
    // colour of an initial state particle ( = the recoiler))
    else {
      // Set colour of antiquark radiator to zero
      Radiator.col( 0 );
      for (int i = 0; i < event.size(); ++i) {
        int col = event[i].col();
        int acl = event[i].acol();

        if ( event[i].isFinal()) {
          // Search for leftover anticolour in final / initial state
          if ( acl > 0 && FindCol(acl,i,0,event,1,true) == 0
              && FindCol(acl,i,0,event,2,true) == 0 )
            Radiator.acol(event[i].acol());
        } else {
          // Search for leftover colour in initial / final state
          if ( col > 0 && FindCol(col,i,0,event,1,true) == 0
              && FindCol(col,i,0,event,2,true) == 0 )
            Radiator.acol(event[i].col());
        }
      } // end loop over particles in event record
    }

  } else if ( Radiator.colType() == 1 ) {
    // For final state quark radiator, the colour is fixed
    // by the final / initial state recoiler anticolour / colour
    if ( RadType + RecType == 2 )
      Radiator.cols( Recoiler.acol(), 0);

    else if ( RadType + RecType == 0 )
      Radiator.cols( Recoiler.col(), 0);
    // For initial state quark radiator, the colour is fixed
    // by the anticolour of the emitted gluon (which will be the
    // leftover colour of a final state particle or the leftover
    // anticolour of an initial state particle ( = the recoiler))

    else {
      // Set anticolour of quark radiator to zero
      Radiator.acol( 0 );
      for (int i = 0; i < event.size(); ++i) {
        int col = event[i].col();
        int acl = event[i].acol();

        if ( event[i].isFinal()) {
          // Search for leftover colour in final / initial state
          if ( col > 0 && FindCol(col,i,0,event,1,true) == 0
              && FindCol(col,i,0,event,2,true) == 0)
            Radiator.col(event[i].col());
        } else {
          // Search for leftover anticolour in initial / final state
          if ( acl > 0 && FindCol(acl,i,0,event,1,true) == 0
              && FindCol(acl,i,0,event,2,true) == 0)
            Radiator.col(event[i].acol());
        }
      } // end loop over particles in event record

    } // end distinction between fsr / fsr+initial recoiler / isr

  } else if ( Radiator.colType() == 2 ) {
    // For a gluon radiator, one (anticolour) colour index is defined
    // by the recoiler colour (anticolour).
    // The remaining index is chosen to match the free index in the
    // event
    // Search for leftover colour (anticolour) in the final state
    for (int i = 0; i < event.size(); ++i) {
      int col = event[i].col();
      int acl = event[i].acol();
      int iEx = i;

      if ( event[i].isFinal()) {
        if ( col > 0 && FindCol(col,iEx,0,event,1,true) == 0
          && FindCol(col,iEx,0,event,2,true) == 0) {
          if (Radiator.status() < 0 ) Radiator.col(event[i].col());
          else Radiator.acol(event[i].col());
        }
        if ( acl > 0 && FindCol(acl,iEx,0,event,2,true) == 0
          && FindCol(acl,iEx,0,event,1,true) == 0 ) {
          if (Radiator.status() < 0 )  Radiator.acol(event[i].acol());
          else Radiator.col(event[i].acol());
        }
      } else {
        if ( col > 0 && FindCol(col,iEx,0,event,1,true) == 0
          && FindCol(col,iEx,0,event,2,true) == 0) {
          if (Radiator.status() < 0 ) Radiator.acol(event[i].col());
          else Radiator.col(event[i].col());
        }
        if ( acl > 0 && (FindCol(acl,iEx,0,event,2,true) == 0
          && FindCol(acl,iEx,0,event,1,true) == 0)) {
          if (Radiator.status() < 0 ) Radiator.col(event[i].acol());
          else Radiator.acol(event[i].acol());
        }
      }
    } // end loop over particles in event record
  } // end cases of different radiator colour type

  // If either colour or anticolour has not been set, return false
  if (Radiator.col() < 0 || Radiator.acol() < 0) return false;
  // Done
  return true;
}

//--------------------------------------------------------------------------

// Function to find a colour (anticolour) index in the input event
// IN  int col       : Colour tag to be investigated
//     int iExclude1 : Identifier of first particle to be excluded
//                     from search
//     int iExclude2 : Identifier of second particle to be excluded
//                     from  search
//     Event event   : event to be searched for colour tag
//     int type      : Tag to define if col should be counted as
//                      colour (type = 1) [->find anti-colour index
//                                         contracted with col]
//                      anticolour (type = 2) [->find colour index
//                                         contracted with col]
// OUT int           : Position of particle in event record
//                     contraced with col [0 if col is free tag]

int DireHistory::FindCol(int col, int iExclude1, int iExclude2,
            const Event& event, int type, bool isHardIn) {

  bool isHard = isHardIn;
  int index = 0;

  if (isHard) {
    // Search event record for matching colour & anticolour
    for(int n = 0; n < event.size(); ++n) {
      if ( n != iExclude1 && n != iExclude2
        && event[n].colType() != 0
        &&(   event[n].status() > 0          // Check outgoing
           || event[n].status() == -21) ) {  // Check incoming
         if ( event[n].acol() == col ) {
          index = -n;
          break;
        }
        if ( event[n].col()  == col ) {
          index =  n;
          break;
        }
      }
    }
  } else {

    // Search event record for matching colour & anticolour
    for(int n = 0; n < event.size(); ++n) {
      if (  n != iExclude1 && n != iExclude2
        && event[n].colType() != 0
        &&(   event[n].status() == 43        // Check outgoing from ISR
           || event[n].status() == 51        // Check outgoing from FSR
           || event[n].status() == -41       // first initial
           || event[n].status() == -42) ) {  // second initial
        if ( event[n].acol() == col ) {
          index = -n;
          break;
        }
        if ( event[n].col()  == col ) {
          index =  n;
          break;
        }
      }
    }
  }
  // if no matching colour / anticolour has been found, return false
  if ( type == 1 && index < 0) return abs(index);
  if ( type == 2 && index > 0) return abs(index);

  return 0;
}

//--------------------------------------------------------------------------

// Function to in the input event find a particle with quantum
// numbers matching those of the input particle
// IN  Particle : Particle to be searched for
//     Event    : Event to be searched in
// OUT int      : > 0 : Position of matching particle in event
//                < 0 : No match in event

int DireHistory::FindParticle( const Particle& particle, const Event& event,
  bool checkStatus ) {

  int index = -1;

  for ( int i = int(event.size()) - 1; i > 0; --i )
    if ( event[i].id()         == particle.id()
      && event[i].colType()    == particle.colType()
      && event[i].chargeType() == particle.chargeType()
      && event[i].col()        == particle.col()
      && event[i].acol()       == particle.acol()
      && event[i].charge()     == particle.charge() ) {
      index = i;
      break;
    }

  if ( checkStatus && event[index].status() != particle.status() )
    index = -1;

  return index;
}

//--------------------------------------------------------------------------

// Function to get the colour of the radiator before the splitting
// for clustering
// IN  int   : Position of the radiator after the splitting, in the event
//     int   : Position of the emitted after the splitting, in the event
//     Event : Reference event
// OUT int   : Colour of the radiator before the splitting

int DireHistory::getRadBeforeCol(const int rad, const int emt,
      const Event& event) {

  // Save type of splitting
  int type = (event[rad].isFinal()) ? 1 :-1;
  // Get flavour of radiator after potential clustering
  int radBeforeFlav = getRadBeforeFlav(rad,emt,event);
  // Get colours of the radiator before the potential clustering
  int radBeforeCol = -1;
  // Get reconstructed gluon colours
  if (radBeforeFlav == 21) {

    // Start with quark emissions in FSR
    if (type == 1 && event[emt].id() != 21) {
      radBeforeCol = (event[rad].col()  > 0)
                   ? event[rad].col() : event[emt].col();
    // Quark emissions in ISR
    } else if (type == -1 && event[emt].id() != 21) {
      radBeforeCol = (event[rad].col()  > 0)
                   ? event[rad].col() : event[emt].acol();
    //Gluon emissions in FSR
    } else if (type == 1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour
      int colRemove = (event[rad].col() == event[emt].acol())
                    ? event[rad].col() : event[rad].acol();
      radBeforeCol  = (event[rad].col()  == colRemove)
                    ? event[emt].col() : event[rad].col();
    //Gluon emissions in ISR
    } else if (type == -1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour
      int colRemove = (event[rad].col() == event[emt].col())
                    ? event[rad].col() : event[rad].acol();
      radBeforeCol  = (event[rad].col()  == colRemove)
                    ? event[emt].acol() : event[rad].col();
    }

  // Get reconstructed quark colours
  } else if ( radBeforeFlav > 0) {

    // Quark emission in FSR
    if (type == 1 && event[emt].id() != 21) {
      // If radiating is a quark, remove the repeated index, and take
      // the remaining indices as colour and anticolour
      int colRemove = (event[rad].col() == event[emt].acol())
                    ? event[rad].acol() : 0;
      radBeforeCol  = (event[rad].col()  == colRemove)
                    ? event[emt].col() : event[rad].col();
    //Gluon emissions in FSR
    } else if (type == 1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour
      int colRemove = (event[rad].col() == event[emt].acol())
                    ? event[rad].col() : 0;
      radBeforeCol  = (event[rad].col()  == colRemove)
                    ? event[emt].col() : event[rad].col();
    //Quark emissions in ISR
    } else if (type == -1 && event[emt].id() != 21) {
      // If emitted is a quark, remove the repeated index, and take
      // the remaining indices as colour and anticolour
      int colRemove = (event[rad].col() == event[emt].col())
                    ? event[rad].col() : 0;
      radBeforeCol  = (event[rad].col()  == colRemove)
                    ? event[emt].acol() : event[rad].col();
    //Gluon emissions in ISR
    } else if (type == -1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour
      int colRemove = (event[rad].col() == event[emt].col())
                    ? event[rad].col() : 0;
      radBeforeCol  = (event[rad].col()  == colRemove)
                    ? event[emt].acol() : event[rad].col();
    }
  // Other particles are assumed uncoloured
  } else {
    radBeforeCol = 0;
  }

  return radBeforeCol;

}

//--------------------------------------------------------------------------

// Function to get the anticolour of the radiator before the splitting
// for clustering
// IN  int   : Position of the radiator after the splitting, in the event
//     int   : Position of the emitted after the splitting, in the event
//     Event : Reference event
// OUT int   : Anticolour of the radiator before the splitting

int DireHistory::getRadBeforeAcol(const int rad, const int emt,
      const Event& event) {

  // Save type of splitting
  int type = (event[rad].isFinal()) ? 1 :-1;
  // Get flavour of radiator after potential clustering

  int radBeforeFlav = getRadBeforeFlav(rad,emt,event);
  // Get colours of the radiator before the potential clustering
  int radBeforeAcl = -1;
  // Get reconstructed gluon colours
  if (radBeforeFlav == 21) {

    // Start with quark emissions in FSR
    if (type == 1 && event[emt].id() != 21) {
      radBeforeAcl = (event[rad].acol() > 0)
                   ? event[rad].acol() : event[emt].acol();
    // Quark emissions in ISR
    } else if (type == -1 && event[emt].id() != 21) {
      radBeforeAcl = (event[rad].acol() > 0)
                   ? event[rad].acol() : event[emt].col();
    //Gluon emissions in FSR
    } else if (type == 1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour
      int colRemove = (event[rad].col() == event[emt].acol())
                    ? event[rad].col() : event[rad].acol();
      radBeforeAcl  = (event[rad].acol() == colRemove)
                    ? event[emt].acol() : event[rad].acol();
    //Gluon emissions in ISR
    } else if (type == -1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour
      int colRemove = (event[rad].col() == event[emt].col())
                    ? event[rad].col() : event[rad].acol();
      radBeforeAcl  = (event[rad].acol() == colRemove)
                    ? event[emt].col() : event[rad].acol();
    }

  // Get reconstructed anti-quark colours
  } else if ( radBeforeFlav < 0) {

    // Antiquark emission in FSR
    if (type == 1 && event[emt].id() != 21) {
      // If radiating is a antiquark, remove the repeated index, and take
      // the remaining indices as colour and anticolour
      int colRemove = (event[rad].col() == event[emt].acol())
                    ? event[rad].acol() : 0;
      radBeforeAcl  = (event[rad].acol()  == colRemove)
                    ? event[emt].acol() : event[rad].acol();
    //Gluon emissions in FSR
    } else if (type == 1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour
      int colRemove = (event[rad].acol() == event[emt].col())
                    ? event[rad].acol() : 0;
      radBeforeAcl  = (event[rad].acol()  == colRemove)
                    ? event[emt].acol() : event[rad].acol();
    //Antiquark emissions in ISR
    } else if (type == -1 && event[emt].id() != 21) {
      // If emitted is an antiquark, remove the repeated index, and take
      // the remaining indices as colour and anticolour
      int colRemove = (event[rad].acol() == event[emt].acol())
                    ? event[rad].acol() : 0;
      radBeforeAcl  = (event[rad].acol()  == colRemove)
                    ? event[emt].col() : event[rad].acol();
    //Gluon emissions in ISR
    } else if (type == -1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour
      int colRemove = (event[rad].acol() == event[emt].acol())
                    ? event[rad].acol() : 0;
      radBeforeAcl  = (event[rad].acol()  == colRemove)
                    ? event[emt].col() : event[rad].acol();
    }
  // Other particles are considered uncoloured
  } else {
    radBeforeAcl = 0;
  }

  return radBeforeAcl;

}

//--------------------------------------------------------------------------

  // Function to get the parton connected to in by a colour line
  // IN  int   : Position of parton for which partner should be found
  //     Event : Reference event
  // OUT int   : If a colour line connects the "in" parton with another
  //             parton, return the Position of the partner, else return 0

int DireHistory::getColPartner(const int in, const Event& event) {

  if (event[in].col() == 0) return 0;

  int partner = 0;
  // Try to find anticolour index first
  partner = FindCol(event[in].col(),in,0,event,1,true);
  // If no anticolour index has been found, try colour
  if (partner == 0)
   partner = FindCol(event[in].col(),in,0,event,2,true);

  return partner;

}

//--------------------------------------------------------------------------


  // Function to get the parton connected to in by an anticolour line
  // IN  int   : Position of parton for which partner should be found
  //     Event : Reference event
  // OUT int   : If an anticolour line connects the "in" parton with another
  //             parton, return the Position of the partner, else return 0

int DireHistory::getAcolPartner(const int in, const Event& event) {

  if (event[in].acol() == 0) return 0;

  int partner = 0;
  // Try to find colour index first
  partner = FindCol(event[in].acol(),in,0,event,2,true);
  // If no colour index has been found, try anticolour
  if (partner == 0)
   partner = FindCol(event[in].acol(),in,0,event,1,true);

  return partner;

}

//--------------------------------------------------------------------------

// Function to get the list of partons connected to the particle
// formed by reclusterinf emt and rad by colour and anticolour lines
// IN  int          : Position of radiator in the clustering
// IN  int          : Position of emitted in the clustering
//     Event        : Reference event
// OUT vector<int>  : List of positions of all partons that are connected
//                    to the parton that will be formed
//                    by clustering emt and rad.

vector<int> DireHistory::getReclusteredPartners(const int rad, const int emt,
  const Event& event) {

  // Save type
  int type = event[rad].isFinal() ? 1 : -1;
  // Get reclustered colours
  int radBeforeCol = getRadBeforeCol(rad, emt, event);
  int radBeforeAcl = getRadBeforeAcol(rad, emt, event);
  // Declare output
  vector<int> partners;


  // Start with FSR clusterings
  if (type == 1) {

    for(int i=0; i < int(event.size()); ++i) {
      // Check all initial state partons
      if ( i != emt && i != rad
        && event[i].status() == -21
        && event[i].col() > 0
        && event[i].col() == radBeforeCol)
          partners.push_back(i);
      // Check all final state partons
      if ( i != emt && i != rad
        && event[i].isFinal()
        && event[i].acol() > 0
        && event[i].acol() == radBeforeCol)
          partners.push_back(i);
      // Check all initial state partons
      if ( i != emt && i != rad
        && event[i].status() == -21
        && event[i].acol() > 0
        && event[i].acol() == radBeforeAcl)
          partners.push_back(i);
      // Check all final state partons
      if ( i != emt && i != rad
        && event[i].isFinal()
        && event[i].col() > 0
        && event[i].col() == radBeforeAcl)
          partners.push_back(i);
    }
  // Start with ISR clusterings
  } else {

    for(int i=0; i < int(event.size()); ++i) {
      // Check all initial state partons
      if ( i != emt && i != rad
        && event[i].status() == -21
        && event[i].acol() > 0
        && event[i].acol() == radBeforeCol)
          partners.push_back(i);
      // Check all final state partons
      if ( i != emt && i != rad
        && event[i].isFinal()
        && event[i].col() > 0
        && event[i].col() == radBeforeCol)
          partners.push_back(i);
      // Check all initial state partons
      if ( i != emt && i != rad
        && event[i].status() == -21
        && event[i].col() > 0
        && event[i].col() == radBeforeAcl)
          partners.push_back(i);
      // Check all final state partons
      if ( i != emt && i != rad
        && event[i].isFinal()
        && event[i].acol() > 0
        && event[i].acol() == radBeforeAcl)
          partners.push_back(i);
    }

  }
  // Done
  return partners;
}

//--------------------------------------------------------------------------

// Function to extract a chain of colour-connected partons in
// the event
// IN     int          : Type of parton from which to start extracting a
//                       parton chain. If the starting point is a quark
//                       i.e. flavType = 1, a chain of partons that are
//                       consecutively connected by colour-lines will be
//                       extracted. If the starting point is an antiquark
//                       i.e. flavType =-1, a chain of partons that are
//                       consecutively connected by anticolour-lines
//                       will be extracted.
// IN      int         : Position of the parton from which a
//                       colour-connected chain should be derived
// IN      Event       : Refernence event
// IN/OUT  vector<int> : Partons that should be excluded from the search.
// OUT     vector<int> : Positions of partons along the chain
// OUT     bool        : Found singlet / did not find singlet

bool DireHistory::getColSinglet( const int flavType, const int iParton,
  const Event& event, vector<int>& exclude, vector<int>& colSinglet) {

  // If no possible flavour to start from has been found
  if (iParton < 0) return false;

  // If no further partner has been found in a previous iteration,
  // and the whole final state has been excluded, we're done
  if (iParton == 0) {

    // Count number of final state partons
    int nFinal = 0;
    for(int i=0; i < int(event.size()); ++i)
      if ( event[i].isFinal() && event[i].colType() != 0)
        nFinal++;

    // Get number of initial state partons in the list of
    // excluded partons
    int nExclude = int(exclude.size());
    int nInitExclude = 0;
    if (!event[exclude[2]].isFinal())
      nInitExclude++;
    if (!event[exclude[3]].isFinal())
      nInitExclude++;

    // If the whole final state has been considered, return
    if (nFinal == nExclude - nInitExclude)
      return true;
    else
      return false;

  }

  // Declare colour partner
  int colP = 0;
  // Save the colour partner
  colSinglet.push_back(iParton);
  // Remove the partner from the list
  exclude.push_back(iParton);
  // When starting out from a quark line, follow the colour lines
  if (flavType == 1)
    colP = getColPartner(iParton,event);
  // When starting out from an antiquark line, follow the anticolour lines
  else
    colP = getAcolPartner(iParton,event);

  // Do not count excluded partons twice
  for(int i = 0; i < int(exclude.size()); ++i)
    if (colP == exclude[i])
      return true;

  // Recurse
  return getColSinglet(flavType,colP,event,exclude,colSinglet);

}

//--------------------------------------------------------------------------

// Function to check that a set of partons forms a colour singlet
// IN  Event       : Reference event
// IN  vector<int> : Positions of the partons in the set
// OUT bool        : Is a colour singlet / is not

bool DireHistory::isColSinglet( const Event& event,
  vector<int> system ) {

  // Check if system forms a colour singlet
  for(int i=0; i < int(system.size()); ++i ) {
    // Match quark and gluon colours
    if ( system[i] > 0
      && (event[system[i]].colType() == 1
       || event[system[i]].colType() == 2) ) {
      for(int j=0; j < int(system.size()); ++j)
        // If flavour matches, remove both partons and continue
        if ( system[j] > 0
          && event[system[i]].col() == event[system[j]].acol()) {
          // Remove index and break
          system[i] = 0;
          system[j] = 0;
          break;
        }
    }
    // Match antiquark and gluon anticolours
    if ( system[i] > 0
      && (event[system[i]].colType() == -1
       || event[system[i]].colType() == 2) ) {
      for(int j=0; j < int(system.size()); ++j)
        // If flavour matches, remove both partons and continue
        if ( system[j] > 0
          && event[system[i]].acol() == event[system[j]].col()) {
          // Remove index and break
          system[i] = 0;
          system[j] = 0;
          break;
        }
    }

  }

  // The system is a colour singlet if for all colours,
  // an anticolour was found
  bool isColSing = true;
  for(int i=0; i < int(system.size()); ++i)
    if ( system[i] != 0 )
      isColSing = false;

  // Return
  return isColSing;


}

//--------------------------------------------------------------------------

// Function to check that a set of partons forms a flavour singlet
// IN  Event       : Reference event
// IN  vector<int> : Positions of the partons in the set
// IN  int         : Flavour of all the quarks in the set, if
//                   all quarks in a set should have a fixed flavour
// OUT bool        : Is a flavour singlet / is not

bool DireHistory::isFlavSinglet( const Event& event,
  vector<int> system, int flav) {

  // If a decoupled colour singlet has been found, check if this is also
  // a flavour singlet
  // Check that each quark matches an antiquark
  for(int i=0; i < int(system.size()); ++i)
    if ( system[i] > 0 ) {
      for(int j=0; j < int(system.size()); ++j) {
        // If flavour of outgoing partons matches,
        // remove both partons and continue.
        // Skip all bosons
        if ( event[i].idAbs() != 21
          && event[i].idAbs() != 22
          && event[i].idAbs() != 23
          && event[i].idAbs() != 24
          && system[j] > 0
          && event[system[i]].isFinal()
          && event[system[j]].isFinal()
          && event[system[i]].id() == -1*event[system[j]].id()) {
          // If we want to check if only one flavour of quarks
          // exists
          if (abs(flav) > 0 && event[system[i]].idAbs() != flav)
            return false;
          // Remove index and break
          system[i] = 0;
          system[j] = 0;
          break;
        }
        // If flavour of outgoing and incoming partons match,
        // remove both partons and continue.
        // Skip all bosons
        if ( event[i].idAbs() != 21
          && event[i].idAbs() != 22
          && event[i].idAbs() != 23
          && event[i].idAbs() != 24
          && system[j] > 0
          && event[system[i]].isFinal() != event[system[j]].isFinal()
          && event[system[i]].id() == event[system[j]].id()) {
          // If we want to check if only one flavour of quarks
          // exists
          if (abs(flav) > 0 && event[system[i]].idAbs() != flav)
            return false;
          // Remove index and break
          system[i] = 0;
          system[j] = 0;
          break;
        }

      }
    }

  // The colour singlet is a flavour singlet if for all quarks,
  // an antiquark was found
  bool isFlavSing = true;
  for(int i=0; i < int(system.size()); ++i)
    if ( system[i] != 0 )
      isFlavSing = false;

  // Return
  return isFlavSing;

}

//--------------------------------------------------------------------------

// Function to check if rad,emt,rec triple is allowed for clustering
// IN int rad,emt,rec : Positions (in event record) of the three
//                      particles considered for clustering
//    Event event     : Reference event

bool DireHistory::allowedClustering( int rad, int emt, int rec, int partner,
  string name, const Event& event ) {

  // Declare output
  bool allowed = true;

  // CONSTRUCT SOME PROPERTIES FOR LATER INVESTIGATION

  // Check if the triple forms a colour singlett
  bool isSing    = isSinglett(rad,emt,partner,event);
  bool hasColour = event[rad].colType() != 0 || event[emt].colType() != 0;
  int type       = (event[rad].isFinal()) ? 1 :-1;

  // Use external shower for merging.
  map<string,double> stateVars;

  if (name.compare("Dire_fsr_qcd_1->21&1") == 0) swap(rad,emt);
  if (name.compare("Dire_fsr_qcd_1->22&1") == 0) swap(rad,emt);
  if (name.compare("Dire_fsr_qcd_11->22&11") == 0) swap(rad,emt);

  bool hasPartonLevel(showers && showers->timesPtr && showers->spacePtr),
       hasShowers(fsr && isr);
  if (hasPartonLevel) {
    bool isFSR = showers->timesPtr->isTimelike(event, rad, emt, rec, "");
    if (isFSR) stateVars = showers->timesPtr->getStateVariables
                 (event,rad,emt,rec,name);
    else       stateVars = showers->spacePtr->getStateVariables
                 (event,rad,emt,rec,name);
  } else if (hasShowers) {
    bool isFSR = fsr->isTimelike(event, rad, emt, rec, "");
    if (isFSR) stateVars = fsr->getStateVariables(event,rad,emt,rec,name);
    else       stateVars = isr->getStateVariables(event,rad,emt,rec,name);
  }

  // Get flavour of radiator after potential clustering
  int radBeforeFlav = int(stateVars["radBefID"]);
  // Get colours of the radiator before the potential clustering
  int radBeforeCol = int(stateVars["radBefCol"]);
  int radBeforeAcl = int(stateVars["radBefAcol"]);

  // Get colour partner of reclustered parton
  vector<int> radBeforeColP = getReclusteredPartners(rad, emt, event);

  // Only allow clustering if the evolution scale is well-defined.
  if ( stateVars["t"] < 0.0) return false;

  // Count coloured partons in hard process
  int nPartonInHard = 0;
  for(int i=0; i < int(event.size()); ++i)
    // Check all final state partons
    if ( event[i].isFinal()
      && event[i].colType() != 0
      && mergingHooksPtr->hardProcess->matchesAnyOutgoing(i, event) )
      nPartonInHard++;

  // Count coloured final state partons in event, excluding
  // rad, rec, emt and hard process
  int nPartons = 0;
  for(int i=0; i < int(event.size()); ++i)
    // Check all final state partons
    if ( i!=emt && i!=rad && i!=rec
      &&  event[i].isFinal()
      &&  event[i].colType() != 0
      && !mergingHooksPtr->hardProcess->matchesAnyOutgoing(i, event) )
      nPartons++;

  // Count number of initial state partons
  int nInitialPartons = 0;
  for(int i=0; i < int(event.size()); ++i)
    if ( event[i].status() == -21
      && event[i].colType() != 0 )
      nInitialPartons++;

  // Get number of non-charged final state particles
  int nFinalEW = 0;
  for(int i=0; i < int(event.size()); ++i)
    if ( event[i].isFinal()
      &&(  event[i].id() == 22
        || event[i].id() == 23
        || event[i].idAbs() == 24
        ||(event[i].idAbs() > 10 && event[i].idAbs() < 20)
        ||(event[i].idAbs() > 1000010 && event[i].idAbs() < 1000020)
        ||(event[i].idAbs() > 2000010 && event[i].idAbs() < 2000020) ))
      nFinalEW++;

  int nFinalH = 0;
  for(int i=0; i < int(event.size()); ++i)
    if ( event[i].isFinal() && event[i].id() == 25)
      nFinalH++;

  // Check if event after potential clustering contains an even
  // number of quarks and/or antiquarks
  // (otherwise no electroweak vertex could be formed!)
  // Get number of final quarks
  int nFinalQuark = 0;
  // Get number of excluded final state quarks as well
  int nFinalQuarkExc = 0;
  for(int i=0; i < int(event.size()); ++i) {
    if (i !=rad && i != emt && i != rec) {
      if (event[i].isFinal() && abs(event[i].colType()) == 1 ) {
        if ( !mergingHooksPtr->hardProcess->matchesAnyOutgoing(i,event) )
          nFinalQuark++;
        else
          nFinalQuarkExc++;
      }
    }
  }

  // Add recoiler to number of final quarks
  if (event[rec].isFinal() && event[rec].isQuark()) nFinalQuark++;
  // Add radiator after clustering to number of final quarks
  if (event[rad].isFinal() && abs(radBeforeFlav) < 10) nFinalQuark++;

  // Get number of initial quarks
  int nInitialQuark = 0;
  if (type == 1) {
    if (event[rec].isFinal()) {
      if (event[3].isQuark())        nInitialQuark++;
      if (event[4].isQuark())        nInitialQuark++;
    } else {
      int iOtherIn = (rec == 3) ? 4 : 3;
      if (event[rec].isQuark())      nInitialQuark++;
      if (event[iOtherIn].isQuark()) nInitialQuark++;
    }
  } else {
    if (event[rec].isFinal()) {
      int iOtherIn = (rad == 3) ? 4 : 3;
      if (abs(radBeforeFlav) < 10)   nInitialQuark++;
      if (event[iOtherIn].isQuark()) nInitialQuark++;
    } else {
      if (abs(radBeforeFlav) < 10)   nInitialQuark++;
      if (event[rec].isQuark())      nInitialQuark++;
    }
  }

  // Get number of initial leptons.
  int nInitialLepton = 0;
  if (type == 1) {
    if (event[rec].isFinal()) {
      if (event[3].isLepton()) nInitialLepton++;
      if (event[4].isLepton()) nInitialLepton++;
    } else {
      int iOtherIn = (rec == 3) ? 4 : 3;
      if (event[rec].isLepton()) nInitialLepton++;
      if (event[iOtherIn].isLepton()) nInitialLepton++;
    }
  } else {
    // Add recoiler to number of initial quarks
    if (event[rec].isLepton()) nInitialLepton++;
    // Add radiator after clustering to number of initial quarks
    if (abs(radBeforeFlav) > 10 && abs(radBeforeFlav) < 20 ) nInitialLepton++;
  }

  // Store incoming and outgoing flavours after clustering.
  vector<int> in;
  for(int i=0; i < int(event.size()); ++i)
    if ( i!=emt && i!=rad && i!=rec
      && (event[i].mother1() == 1 || event[i].mother1() == 2))
      in.push_back(event[i].id());
  if (!event[rad].isFinal()) in.push_back(radBeforeFlav);
  if (!event[rec].isFinal()) in.push_back(event[rec].id());
  vector<int> out;
  for(int i=0; i < int(event.size()); ++i)
    if ( i!=emt && i!=rad && i!=rec && event[i].isFinal())
      out.push_back(event[i].id());
  if (event[rad].isFinal()) out.push_back(radBeforeFlav);
  if (event[rec].isFinal()) out.push_back(event[rec].id());

  // BEGIN CHECKING THE CLUSTERING

  // Do not allow clusterings that lead to a disallowed proton content.
  int proton[] = {1,2,3,4,5,21,22,23,24};
  bool isInProton = false;
  for(int i=0; i < 9; ++i)
    if (abs(radBeforeFlav) == proton[i]) isInProton = true;
  if ( type == -1 && particleDataPtr->colType(radBeforeFlav) != 0
    && !isInProton) return false;

  // Check if colour is conserved
  vector<int> unmatchedCol;
  vector<int> unmatchedAcl;
  // Check all unmatched colours
  for ( int i = 0; i < event.size(); ++i)
    if ( i != emt && i != rad
      && (event[i].isFinal() || event[i].status() == -21)
      && event[i].colType() != 0 ) {

      int colP = getColPartner(i,event);
      int aclP = getAcolPartner(i,event);

      if (event[i].col() > 0
        && (colP == emt || colP == rad || colP == 0) )
        unmatchedCol.push_back(i);
      if (event[i].acol() > 0
        && (aclP == emt || aclP == rad || aclP == 0) )
        unmatchedAcl.push_back(i);

    }

  // If more than one colour or more than one anticolour are unmatched,
  // there is no way to make this clustering work
  if (int(unmatchedCol.size()) + int(unmatchedAcl.size()) > 2)
    return false;

  // If triple forms colour singlett, check that resulting state
  // matches hard core process
  if (hasColour && isSing)
    allowed = false;
  if (hasColour && isSing && (abs(radBeforeFlav)<10 && event[rec].isQuark()) )
    allowed = true;

  // Colour singlet in DIS hard process
  if (hasColour && isSing && abs(radBeforeFlav)<10 && nPartons == 0
    && nInitialPartons == 1)
    allowed = true;

  // Never recluster any outgoing partons of the core V -> qqbar' splitting!
  if ( mergingHooksPtr->hardProcess->matchesAnyOutgoing(emt,event) ) {
    // Check if any other particle could replace "emt" as part of the candidate
    // core process. If so, replace emt with the new candidate and allow the
    // clustering.
    bool canReplace = mergingHooksPtr->hardProcess->findOtherCandidates(emt,
                        event, true);

    if (canReplace) allowed = true;
    else allowed = false;
  }

  // Never remove so many particles that the hard process cannot
  // be set up afterwards.
  //int nIncPartHardProc = mergingHooksPtr->hardProcess->nQuarksIn();
  int nOutPartHardProc = mergingHooksPtr->hardProcess->nQuarksOut();
  // Add recoiler to number of final quarks
  int nOutPartNow(nPartons);
  // Add non-participating hard process partons.
  for(int i=0; i < int(event.size()); ++i)
    // Check all final state partons
    if ( i!=emt && i!=rad && i!=rec
      &&  event[i].isFinal()
      &&  event[i].colType() != 0
      &&  mergingHooksPtr->hardProcess->matchesAnyOutgoing(i, event) )
      nOutPartNow++;
  if (event[rec].isFinal() && event[rec].colType() != 0) nOutPartNow++;
  // Add radiator after clustering to number of final quarks
  if (event[rad].isFinal() && particleDataPtr->colType(radBeforeFlav) != 0)
    nOutPartNow++;
  if (nOutPartNow < nOutPartHardProc) allowed = false;

  // Never allow clustering of any outgoing partons of the hard process
  // which would change the flavour of one of the hard process partons!
  if ( mergingHooksPtr->hardProcess->matchesAnyOutgoing(rad,event)
      && event[rad].id() != radBeforeFlav )
    allowed = false;

  // If only gluons in initial state and no quarks in final state,
  // reject (no electroweak vertex can be formed)
  if ( nFinalEW   != 0     && nInitialQuark == 0 && nFinalQuark == 0
    && nFinalQuarkExc == 0 && nInitialLepton == 0
    && !mayHaveEffectiveVertex( mergingHooksPtr->getProcessString(), in, out))
    allowed = false;

  if ( (nInitialQuark + nFinalQuark + nFinalQuarkExc)%2 != 0 )
    allowed = false;

  map<int,int> nIncIDs, nOutIDs;
  for ( int i = 0; i < event.size(); ++i) {
    if ( i != emt && i != rad && event[i].isFinal() ) {
      if (nOutIDs.find(event[i].id()) != nOutIDs.end() )
        nOutIDs[event[i].id()]++;
      else
        nOutIDs.insert(make_pair(event[i].id(),1));
    }
    if ( i != emt && i != rad && event[i].status() == -21 ){
      if (nIncIDs.find(event[i].id()) != nIncIDs.end() )
        nIncIDs[event[i].id()]++;
      else
        nIncIDs.insert(make_pair(event[i].id(),1));
    }
  }
  if (type > 0 ) {
    if (nOutIDs.find(radBeforeFlav) != nOutIDs.end()) nOutIDs[radBeforeFlav]++;
    else nOutIDs.insert(make_pair(radBeforeFlav,1));
  }
  if (type < 0 ) {
    if (nIncIDs.find(radBeforeFlav) != nIncIDs.end()) nIncIDs[radBeforeFlav]++;
    else nIncIDs.insert(make_pair(radBeforeFlav,1));
  }

  if (!canConnectFlavs(nIncIDs,nOutIDs) ) allowed = false;

  // Disallow clusterings that lead to a 2->1 massless state.
  // To check, only look at final state flavours.
  int nMassless(0), nOther(0);
  for ( map<int, int>::iterator it = nOutIDs.begin();
    it != nOutIDs.end(); ++it )
    if ( abs(it->first) < 20 || abs(it->first) == 21
      || abs(it->first) == 22) nMassless += it->second;
    else nOther++;
  if (nMassless == 1 && nOther == 0) allowed = false;

  // Disallow final state splittings that lead to a purely gluonic final
  // state, while having a completely colour-connected initial state.
  // This means that the clustering is discarded if it does not lead to the
  // t-channel gluon needed to connect the final state to a qq~ initial state.
  // Here, partons excluded from clustering are not counted as possible
  // partners to form a t-channel gluon
  if (event[3].col() == event[4].acol()
    && event[3].acol() == event[4].col()
    && !mayHaveEffectiveVertex( mergingHooksPtr->getProcessString(), in, out)
    && nFinalQuark == 0){
    // Careful if rad and rec are the only quarks in the final state, but
    // were both excluded from the list of final state quarks.
    int nTripletts = abs(event[rec].colType())
                   + abs(particleDataPtr->colType(radBeforeFlav));
    if (event[3].isGluon())                            allowed = false;
    else if (nTripletts != 2 && nFinalQuarkExc%2 == 0) allowed = false;
  }

  // Minimal phase space checks.
  if ( event[rad].isFinal() && event[rec].isFinal()
    && abs( ( event[rad].p() + event[emt].p() + event[rec].p()).pz())
          > ( event[rad].p() + event[emt].p() + event[rec].p()).e() )
    return false;

  if ( !event[rad].isFinal() && !event[rec].isFinal()
    && abs( ( event[rad].p() - event[emt].p() + event[rec].p()).pz())
          > ( event[rad].p() - event[emt].p() + event[rec].p()).e() )
    return false;

  if ( !event[rad].isFinal() && event[rec].isFinal()
    && -(-event[rad].p() + event[emt].p() + event[rec].p()).m2Calc() < 0.)
    return false;

  // Check that invariant mass of dipole is positive.
  // Initial-initial configuration.
  if ( !event[rad].isFinal() && !event[rec].isFinal()
    &&     (event[rad].p() - event[emt].p() + event[rec].p()).m2Calc() < 0.
    && abs((event[rad].p() - event[emt].p() + event[rec].p()).m2Calc()) > 1e-5)
    return false;

  // Initial-final configuration.
  if ( !event[rad].isFinal() && event[rec].isFinal()
    &&    -(event[rad].p() - event[emt].p() - event[rec].p()).m2Calc() < 0.
    && abs((event[rad].p() - event[emt].p() - event[rec].p()).m2Calc()) > 1e-5)
    return false;

  // Final-initial configuration.
  if ( event[rad].isFinal() && !event[rec].isFinal()
    &&    -(-event[rad].p() - event[emt].p() + event[rec].p()).m2Calc() < 0.
    && abs((-event[rad].p() - event[emt].p() + event[rec].p()).m2Calc())
       > 1e-5)
    return false;

  // Final-final configuration.
  if ( event[rad].isFinal() && event[rec].isFinal()
    && (event[rad].p() + event[emt].p() + event[rec].p()).m2Calc() < 0.)
    return false;

  // No problems with gluon radiation
  if (event[emt].id() == 21)
    return allowed;

  // No problems with photon / Z / Higgs radiation
  if (event[emt].id() == 22 || event[emt].id() == 23 || event[emt].id() == 25)
   return allowed;

  // No problems with gluino radiation
  if (event[emt].id() == 1000021)
    return allowed;

  // No problems if radiator is gluon, emitted is (anti)quark.

  // No problems if radiator is photon/Z/Higgs, and emitted is fermion.

  // No problems if radiator is gluino, and emitted is (anti)quark.

  // Save all hard process candidates
  vector<int> outgoingParticles;
  int nOut1 = int(mergingHooksPtr->hardProcess->PosOutgoing1.size());
  for ( int i=0; i < nOut1;  ++i ) {
    int iPos = mergingHooksPtr->hardProcess->PosOutgoing1[i];
    outgoingParticles.push_back(
                      mergingHooksPtr->hardProcess->state[iPos].id() );
  }
  int nOut2 = int(mergingHooksPtr->hardProcess->PosOutgoing2.size());
  for ( int i=0; i < nOut2; ++i ) {
    int iPos = mergingHooksPtr->hardProcess->PosOutgoing2[i];
    outgoingParticles.push_back(
                      mergingHooksPtr->hardProcess->state[iPos].id() );
  }

  // Start more involved checks. g -> q_1 qbar_1 splittings are
  // particularly problematic if more than one quark of the emitted
  // flavour is present.
  // Count number of initial quarks of radiator or emitted flavour
  vector<int> iInQuarkFlav;
  for(int i=0; i < int(event.size()); ++i)
    // Check all initial state partons
    if ( i != emt && i != rad
      && event[i].status() == -21
      && event[i].idAbs() == event[emt].idAbs() )
      iInQuarkFlav.push_back(i);

  // Count number of final quarks of radiator or emitted flavour
  vector<int> iOutQuarkFlav;
  for(int i=0; i < int(event.size()); ++i)
  // Check all final state partons
  if ( i != emt && i != rad
    && event[i].isFinal()
    && event[i].idAbs() == event[emt].idAbs() ) {

    // Loop through final state hard particles. If one matches, remove the
    // matching one, and do not count.
    bool matchOut = false;
    for (int j = 0; j < int(outgoingParticles.size()); ++j)
    if ( event[i].idAbs() == abs(outgoingParticles[j])) {
      matchOut = true;
      outgoingParticles[j] = 99;
    }
    if (!matchOut) iOutQuarkFlav.push_back(i);

  }

  // Save number of potentially dangerous quarks
  int nInQuarkFlav  = int(iInQuarkFlav.size());
  int nOutQuarkFlav = int(iOutQuarkFlav.size());

  // Easiest problem 0:
  // Radiator before splitting exactly matches the partner
  // after the splitting
  if ( event[partner].isFinal()
    && event[partner].id()   == 21
    && radBeforeFlav         == 21
    && event[partner].col()  == radBeforeCol
    && event[partner].acol() == radBeforeAcl)
    return false;

  // If there are no ambiguities in qqbar pairs, return
  if (nInQuarkFlav + nOutQuarkFlav == 0)
    return allowed;

  // Save all quarks and gluons that will not change colour
  vector<int> gluon;
  vector<int> quark;
  vector<int> antiq;
  vector<int> partons;
  for(int i=0; i < int(event.size()); ++i)
    // Check initial and final state partons
    if ( i!=emt && i!=rad
      && event[i].colType() != 0
      && (event[i].isFinal() || event[i].status() == -21) ) {
      // Save index
      partons.push_back(i);
      // Split into components
      if (event[i].colType() == 2)
        gluon.push_back(i);
      else if (event[i].colType() ==  1)
        quark.push_back(i);
      else if (event[i].colType() == -1)
        antiq.push_back(i);
    }

  // We split up the test of the g->qq splitting into final state
  // and initial state problems
  bool isFSRg2qq = ( (type == 1) && radBeforeFlav == 21
                  && (event[rad].id() == -1*event[emt].id()) );
  bool isISRg2qq = ( (type ==-1) && radBeforeFlav == 21
                  && (event[rad].id() ==    event[emt].id()) );

  // First check general things about colour connections
  // Check that clustering does not produce a gluon that is exactly
  // matched in the final state, or does not have any colour connections
  if ( (isFSRg2qq || isISRg2qq)
    && int(quark.size()) + int(antiq.size())
     + int(gluon.size()) > nPartonInHard ) {

      vector<int> colours;
      vector<int> anticolours;
      // Add the colour and anticolour of the gluon before the emission
      // to the list, bookkeep initial colour as final anticolour, and
      // initial anticolour as final colour
      if (type == 1) {
        colours.push_back(radBeforeCol);
        anticolours.push_back(radBeforeAcl);
      } else {
        colours.push_back(radBeforeAcl);
        anticolours.push_back(radBeforeCol);
      }
      // Now store gluon colours and anticolours.
      for(int i=0; i < int(gluon.size()); ++i)
        if (event[gluon[i]].isFinal()) {
          colours.push_back(event[gluon[i]].col());
          anticolours.push_back(event[gluon[i]].acol());
        } else {
          colours.push_back(event[gluon[i]].acol());
          anticolours.push_back(event[gluon[i]].col());
        }

      // Loop through colours and check if any match with
      // anticolours. If colour matches, remove from list
      for(int i=0; i < int(colours.size()); ++i)
        for(int j=0; j < int(anticolours.size()); ++j)
          if (colours[i] > 0 && anticolours[j] > 0
            && colours[i] == anticolours[j]) {
            colours[i] = 0;
            anticolours[j] = 0;
          }


      // If all gluon anticolours and all colours matched, disallow
      // the clustering
      bool allMatched = true;
      for(int i=0; i < int(colours.size()); ++i)
        if (colours[i] != 0)
          allMatched = false;
      for(int i=0; i < int(anticolours.size()); ++i)
        if (anticolours[i] != 0)
          allMatched = false;

      if (allMatched)
        return false;

      // Now add the colours of the hard process, and check if all
      // colours match.
      for(int i=0; i < int(quark.size()); ++i)
        if ( event[quark[i]].isFinal()
        && mergingHooksPtr->hardProcess->matchesAnyOutgoing(quark[i], event) )
          colours.push_back(event[quark[i]].col());

      for(int i=0; i < int(antiq.size()); ++i)
        if ( event[antiq[i]].isFinal()
        && mergingHooksPtr->hardProcess->matchesAnyOutgoing(antiq[i], event) )
          anticolours.push_back(event[antiq[i]].acol());

      // Loop through colours again and check if any match with
      // anticolours. If colour matches, remove from list
      for(int i=0; i < int(colours.size()); ++i)

        for(int j=0; j < int(anticolours.size()); ++j)
          if (colours[i] > 0 && anticolours[j] > 0
            && colours[i] == anticolours[j]) {
            colours[i] = 0;
            anticolours[j] = 0;
          }

      // Check if clustering would produce the hard process
      int nNotInHard = 0;
      for ( int i=0; i < int(quark.size()); ++i )
        if ( !mergingHooksPtr->hardProcess->matchesAnyOutgoing( quark[i],
              event) )
          nNotInHard++;
      for ( int i=0; i < int(antiq.size()); ++i )
        if ( !mergingHooksPtr->hardProcess->matchesAnyOutgoing( antiq[i],
              event) )
          nNotInHard++;
      for(int i=0; i < int(gluon.size()); ++i)
        if ( event[gluon[i]].isFinal() )
          nNotInHard++;
      if ( type == 1 )
          nNotInHard++;

      // If all colours are matched now, and since we have more quarks than
      // present in the hard process, disallow the clustering
      allMatched = true;
      for(int i=0; i < int(colours.size()); ++i)
        if (colours[i] != 0)
          allMatched = false;
      for(int i=0; i < int(anticolours.size()); ++i)
        if (anticolours[i] != 0)
          allMatched = false;

      if (allMatched && nNotInHard > 0)
        return false;

  }

  // FSR PROBLEMS

  if (isFSRg2qq && nInQuarkFlav + nOutQuarkFlav > 0) {

    // Easiest problem 1:
    // RECLUSTERED FINAL STATE GLUON MATCHES INITIAL STATE GLUON
    for(int i=0; i < int(gluon.size()); ++i) {
      if (!event[gluon[i]].isFinal()
        && event[gluon[i]].col()  == radBeforeCol
        && event[gluon[i]].acol() == radBeforeAcl)
        return false;
    }

    // Easiest problem 2:
    // RECLUSTERED FINAL STATE GLUON MATCHES FINAL STATE GLUON
    for(int i=0; i < int(gluon.size()); ++i) {
      if (event[gluon[i]].isFinal()
        && event[gluon[i]].col()  == radBeforeAcl
        && event[gluon[i]].acol() == radBeforeCol)
        return false;
    }

    // Easiest problem 3:
    // RECLUSTERED FINAL STATE GLUON MATCHES FINAL STATE Q-QBAR PAIR
    if ( int(radBeforeColP.size()) == 2
      && event[radBeforeColP[0]].isFinal()
      && event[radBeforeColP[1]].isFinal()
      && event[radBeforeColP[0]].id() == -1*event[radBeforeColP[1]].id() ) {

      // This clustering is allowed if there is no colour in the
      // initial state
      if (nInitialPartons > 0)
        return false;
    }

    // Next-to-easiest problem 1:
    // RECLUSTERED FINAL STATE GLUON MATCHES ONE FINAL STARE Q_1
    // AND ONE INITIAL STATE Q_1
    if ( int(radBeforeColP.size()) == 2
      && ((  event[radBeforeColP[0]].status() == -21
          && event[radBeforeColP[1]].isFinal())
        ||(  event[radBeforeColP[0]].isFinal()
          && event[radBeforeColP[1]].status() == -21))
      && event[radBeforeColP[0]].id() == event[radBeforeColP[1]].id() ) {

      // In principle, clustering this splitting can disconnect
      // the colour lines of a graph. However, the colours can be connected
      // again if a final or initial partons of the correct flavour exists.

      // Check which of the partners are final / initial
      int incoming = (event[radBeforeColP[0]].isFinal())
                   ? radBeforeColP[1] : radBeforeColP[0];
      int outgoing = (event[radBeforeColP[0]].isFinal())
                   ? radBeforeColP[0] : radBeforeColP[1];

      // Loop through event to find "recovery partons"
      bool clusPossible = false;
      for(int i=0; i < int(event.size()); ++i)
        if (  i != emt && i != rad
          &&  i != incoming && i != outgoing
          && !mergingHooksPtr->hardProcess->matchesAnyOutgoing(i,event) ) {
          // Check if an incoming parton matches
          if ( event[i].status() == -21
            && (event[i].id() ==    event[outgoing].id()
              ||event[i].id() == -1*event[incoming].id()) )
          clusPossible = true;
          // Check if a final parton matches
          if ( event[i].isFinal()
            && (event[i].id() == -1*event[outgoing].id()
              ||event[i].id() ==    event[incoming].id()) )
          clusPossible = true;
        }

      // There can be a further complication: If e.g. in
      // t-channel photon exchange topologies, both incoming
      // partons are quarks, and form colour singlets with any
      // number of final state partons, at least try to
      // recluster as much as possible.
      // For this, check if the incoming parton
      // connected to the radiator is connected to a
      // colour and flavour singlet
      vector<int> excludeIn1;
      for(int i=0; i < 4; ++i)
        excludeIn1.push_back(0);
      vector<int> colSingletIn1;
      int flavIn1Type = (event[incoming].id() > 0) ? 1 : -1;
      // Try finding colour singlets
      bool isColSingIn1  = getColSinglet(flavIn1Type,incoming,event,
                             excludeIn1,colSingletIn1);
      // Check if colour singlet also is a flavour singlet
      bool isFlavSingIn1 = isFlavSinglet(event,colSingletIn1);

      // If the incoming particle is a lepton, just ensure lepton number
      // conservation.
      bool foundLepton = false;
      for(int i=0; i < int(event.size()); ++i)
        if ( i != emt && i != rad && i != incoming
          && event[i].isLepton() ) foundLepton = true;
      if ( abs(radBeforeFlav)%10 == 1 ) foundLepton = true;
      if ( foundLepton && event[incoming].isLepton() )
        isColSingIn1 = isFlavSingIn1 = true;

      // Check if the incoming parton not
      // connected to the radiator is connected to a
      // colour and flavour singlet
      int incoming2 = (incoming == 3) ? 4 : 3;
      vector<int> excludeIn2;
      for(int i=0; i < 4; ++i)
        excludeIn2.push_back(0);
      vector<int> colSingletIn2;
      int flavIn2Type = (event[incoming2].id() > 0) ? 1 : -1;
      // Try finding colour singlets
      bool isColSingIn2  = getColSinglet(flavIn2Type,incoming2,event,
                             excludeIn2,colSingletIn2);
      // Check if colour singlet also is a flavour singlet
      bool isFlavSingIn2 = isFlavSinglet(event,colSingletIn2);

      // If the incoming particle is a lepton, just ensure lepton number
      // conservation.
      foundLepton = false;
      for(int i=0; i < int(event.size()); ++i)
        if ( i != emt && i != rad && i != incoming2
          && event[i].isLepton() ) foundLepton = true;
      if ( abs(radBeforeFlav)%10 == 1 ) foundLepton = true;
      if ( foundLepton && event[incoming2].isLepton() )
        isColSingIn2 = isFlavSingIn2 = true;

      // If no "recovery clustering" is possible, reject clustering
      if (!clusPossible
        && (!isColSingIn1 || !isFlavSingIn1
         || !isColSingIn2 || !isFlavSingIn2))
        return false;

    }

    // Next-to-easiest problem 2:
    // FINAL STATE Q-QBAR CLUSTERING DISCONNECTS SINGLETT SUBSYSTEM WITH
    // FINAL STATE Q-QBAR PAIR FROM GRAPH

    if ( int(radBeforeColP.size()) == 2 ) {

      // Prepare to check for colour singlet combinations of final state quarks
      // Start by building a list of partons to exclude when checking for
      // colour singlet combinations
      int flav = event[emt].id();
      vector<int> exclude;
      exclude.push_back(emt);
      exclude.push_back(rad);
      exclude.push_back(radBeforeColP[0]);
      exclude.push_back(radBeforeColP[1]);
      vector<int> colSinglet;

      // Now find parton from which to start checking colour singlets
      int iOther = -1;
      // Loop through event to find a parton of correct flavour
      for(int i=0; i < int(event.size()); ++i)
        // Check final state for parton equalling emitted flavour.
        // Exclude the colour system coupled to the clustering
        if ( i != emt
          && i != rad
          && i != radBeforeColP[0]
          && i != radBeforeColP[1]
          && event[i].isFinal() ) {
          // Stop if one parton of the correct flavour is found
          if (event[i].id() == flav) {
            iOther = i;
            break;
          }
        }
      // Save the type of flavour
      int flavType = (iOther > 0 && event[iOther].id() > 0) ? 1
                   : (iOther > 0) ? -1 : 0;
      // Try finding colour singlets
      bool isColSing = getColSinglet(flavType,iOther,event,exclude,colSinglet);
      // Check if colour singlet also is a flavour singlet
      bool isFlavSing = isFlavSinglet(event,colSinglet);

      // Check if colour singlet is precisely contained in the hard process.
      // If so, then we're safe to recluster.
      bool isHardSys = true;
      for(int i=0; i < int(colSinglet.size()); ++i)
        isHardSys =
        mergingHooksPtr->hardProcess->matchesAnyOutgoing(colSinglet[i], event);

      // Nearly there...
      // If the decoupled colour singlet system is NOT contained in the hard
      // process, we need to check the whole final state.
      if (isColSing && isFlavSing && !isHardSys) {

        // In a final check, ensure that the final state does not only
        // consist of colour singlets that are also flavour singlets
        // of the identical (!) flavours
        // Loop through event and save all final state partons
        vector<int> allFinal;
        for(int i=0; i < int(event.size()); ++i)
          if ( event[i].isFinal() )
            allFinal.push_back(i);

        // Check if all final partons form a colour singlet
        bool isFullColSing  = isColSinglet(event,allFinal);
        // Check if all final partons form a flavour singlet
        bool isFullFlavSing = isFlavSinglet(event,allFinal,flav);

        // If all final quarks are of identical flavour,
        // no possible clustering should be discriminated.
        // Otherwise, disallow
        if (!isFullColSing || !isFullFlavSing)
          return false;
      }

    }

  }

  // ISR PROBLEMS

  if (isISRg2qq && nInQuarkFlav + nOutQuarkFlav > 0) {

    // Easiest problem 1:
    // RECLUSTERED INITIAL STATE GLUON MATCHES FINAL STATE GLUON
    for(int i=0; i < int(gluon.size()); ++i) {
      if (event[gluon[i]].isFinal()
        && event[gluon[i]].col()  == radBeforeCol
        && event[gluon[i]].acol() == radBeforeAcl)
        return false;
    }

    // Easiest problem 2:
    // RECLUSTERED INITIAL STATE GLUON MATCHES INITIAL STATE GLUON
    for(int i=0; i < int(gluon.size()); ++i) {
      if (event[gluon[i]].status() == -21
        && event[gluon[i]].acol()  == radBeforeCol
        && event[gluon[i]].col() == radBeforeAcl)
        return false;
    }

    // Next-to-easiest problem 1:
    // RECLUSTERED INITIAL STATE GLUON MATCHES FINAL STATE Q-QBAR PAIR
    if ( int(radBeforeColP.size()) == 2
      && event[radBeforeColP[0]].isFinal()
      && event[radBeforeColP[1]].isFinal()
      && event[radBeforeColP[0]].id() == -1*event[radBeforeColP[1]].id() ) {

      // In principle, clustering this splitting can disconnect
      // the colour lines of a graph. However, the colours can be connected
      // again if final state partons of the correct (anti)flavour, or
      // initial state partons of the correct flavour exist
      // Loop through event to check
      bool clusPossible = false;
      for(int i=0; i < int(event.size()); ++i)
        if ( i != emt && i != rad
          && i != radBeforeColP[0]
          && i != radBeforeColP[1]
          && !mergingHooksPtr->hardProcess->matchesAnyOutgoing(i,event) ) {
          if (event[i].status() == -21
            && ( event[radBeforeColP[0]].id() == event[i].id()
              || event[radBeforeColP[1]].id() == event[i].id() ))

            clusPossible = true;
          if (event[i].isFinal()
            && ( event[radBeforeColP[0]].id() == -1*event[i].id()
              || event[radBeforeColP[1]].id() == -1*event[i].id() ))
            clusPossible = true;
        }

      // There can be a further complication: If e.g. in
      // t-channel photon exchange topologies, both incoming
      // partons are quarks, and form colour singlets with any
      // number of final state partons, at least try to
      // recluster as much as possible.
      // For this, check if the incoming parton
      // connected to the radiator is connected to a
      // colour and flavour singlet
      int incoming1 = 3;
      vector<int> excludeIn1;
      for(int i=0; i < 4; ++i)
        excludeIn1.push_back(0);
      vector<int> colSingletIn1;
      int flavIn1Type = (event[incoming1].id() > 0) ? 1 : -1;
      // Try finding colour singlets
      bool isColSingIn1  = getColSinglet(flavIn1Type,incoming1,event,
                             excludeIn1,colSingletIn1);
      // Check if colour singlet also is a flavour singlet
      bool isFlavSingIn1 = isFlavSinglet(event,colSingletIn1);

      // If the incoming particle is a lepton, just ensure lepton number
      // conservation.
      bool foundLepton = false;
      for(int i=0; i < int(event.size()); ++i)
        if ( i != emt && i != rad && i != incoming1
          && event[i].isLepton() ) foundLepton = true;
      if ( abs(radBeforeFlav)%10 == 1 ) foundLepton = true;
      if ( foundLepton && event[incoming1].isLepton() )
        isColSingIn1 = isFlavSingIn1 = true;

      // Check if the incoming parton not
      // connected to the radiator is connected to a
      // colour and flavour singlet
      int incoming2 = 4;
      vector<int> excludeIn2;
      for(int i=0; i < 4; ++i)
        excludeIn2.push_back(0);
      vector<int> colSingletIn2;
      int flavIn2Type = (event[incoming2].id() > 0) ? 1 : -1;
      // Try finding colour singlets
      bool isColSingIn2  = getColSinglet(flavIn2Type,incoming2,event,
                             excludeIn2,colSingletIn2);
      // Check if colour singlet also is a flavour singlet
      bool isFlavSingIn2 = isFlavSinglet(event,colSingletIn2);

      // If the incoming particle is a lepton, just ensure lepton number
      // conservation.
      foundLepton = false;
      for(int i=0; i < int(event.size()); ++i)
        if ( i != emt && i != rad && i != incoming2
          && event[i].isLepton() ) foundLepton = true;
      if ( abs(radBeforeFlav)%10 == 1 ) foundLepton = true;
      if ( foundLepton && event[incoming2].isLepton() )
        isColSingIn2 = isFlavSingIn2 = true;

      // If no "recovery clustering" is possible, reject clustering
      if (!clusPossible
        && (!isColSingIn1 || !isFlavSingIn1
         || !isColSingIn2 || !isFlavSingIn2))
        return false;

    }

  }

  // Done
  return allowed;
}

//--------------------------------------------------------------------------

bool DireHistory::hasConnections( int, int nIncIDs[], int nOutIDs[]) {

  bool foundQuarks = false;
  for (int i=-6; i < 6; i++)
    if (nIncIDs[i] > 0 || nOutIDs[i] > 0) foundQuarks = true;

  if ( nIncIDs[-11] == 1 && nOutIDs[-11] == 1 && !foundQuarks) return false;

  return true;
}


bool DireHistory::canConnectFlavs(map<int,int> nIncIDs, map<int,int> nOutIDs) {

  bool foundIncQuarks(false), foundOutQuarks(false);
  for (int i=-6; i < 6; i++) {
    if (nIncIDs[i] > 0) foundIncQuarks = true;
    if (nOutIDs[i] > 0) foundOutQuarks = true;
  }

  int nIncEle = (nIncIDs.find(11)  != nIncIDs.end()) ? nIncIDs[11]  : 0;
  int nIncPos = (nIncIDs.find(-11) != nIncIDs.end()) ? nIncIDs[-11] : 0;
  int nOutEle = (nOutIDs.find(11)  != nOutIDs.end()) ? nOutIDs[11]  : 0;
  int nOutPos = (nOutIDs.find(-11) != nOutIDs.end()) ? nOutIDs[-11] : 0;

  // Cannot couple positron to other electric charge.
  if ( nIncPos == 1 && nOutPos == 1 && !foundOutQuarks && !foundIncQuarks)
    return false;

  // Cannot couple electron to other electric charge.
  if ( nIncEle == 1 && nOutEle == 1 && !foundOutQuarks && !foundIncQuarks)
    return false;

  return true;
}

//--------------------------------------------------------------------------

// Function to check if rad,emt,rec triple is results in
// colour singlet radBefore+recBefore
// IN int rad,emt,rec : Positions (in event record) of the three
//                      particles considered for clustering
//    Event event     : Reference event

bool DireHistory::isSinglett( int rad, int emt, int rec, const Event& event ) {

  int radCol = event[rad].col();
  int emtCol = event[emt].col();
  int recCol = event[rec].col();
  int radAcl = event[rad].acol();
  int emtAcl = event[emt].acol();
  int recAcl = event[rec].acol();
  int recType = event[rec].isFinal() ? 1 : -1;

  bool isSing = false;

  if ( ( recType == -1
       && radCol + emtCol == recCol && radAcl + emtAcl == recAcl)
    ||( recType == 1
       && radCol + emtCol == recAcl && radAcl + emtAcl == recCol) )
    isSing = true;

  return isSing;

}

//--------------------------------------------------------------------------

// Function to check if event is sensibly constructed: Meaning
// that all colour indices are contracted and that the charge in
// initial and final states matches
// IN  event : event to be checked
// OUT TRUE  : event is properly construced
//     FALSE : event not valid

bool DireHistory::validEvent( const Event& event ) {

  // Check if event is coloured
  bool validColour = true;
  for ( int i = 0; i < event.size(); ++i)
   // Check colour of quarks
   if ( event[i].isFinal() && event[i].colType() == 1
          // No corresponding anticolour in final state
       && ( FindCol(event[i].col(),i,0,event,1,true) == 0
          // No corresponding colour in initial state
         && FindCol(event[i].col(),i,0,event,2,true) == 0 )) {
     validColour = false;
     break;
   // Check anticolour of antiquarks
   } else if ( event[i].isFinal() && event[i].colType() == -1
          // No corresponding colour in final state
       && ( FindCol(event[i].acol(),i,0,event,2,true) == 0
          // No corresponding anticolour in initial state
         && FindCol(event[i].acol(),i,0,event,1,true) == 0 )) {
     validColour = false;
     break;
   // No uncontracted colour (anticolour) charge of gluons
   } else if ( event[i].isFinal() && event[i].colType() == 2
          // No corresponding anticolour in final state
       && ( FindCol(event[i].col(),i,0,event,1,true) == 0
          // No corresponding colour in initial state
         && FindCol(event[i].col(),i,0,event,2,true) == 0 )
          // No corresponding colour in final state
       && ( FindCol(event[i].acol(),i,0,event,2,true) == 0
          // No corresponding anticolour in initial state
         && FindCol(event[i].acol(),i,0,event,1,true) == 0 )) {
     validColour = false;
     break;
   }

  // Check charge sum in initial and final state
  bool validCharge = true;
  double initCharge = event[3].charge() + event[4].charge();
  double finalCharge = 0.0;
  for(int i = 0; i < event.size(); ++i)
    if (event[i].isFinal()) finalCharge += event[i].charge();
  if (abs(initCharge-finalCharge) > 1e-12) validCharge = false;

  return (validColour && validCharge);

}

//--------------------------------------------------------------------------

// Function to check whether two clusterings are identical, used
// for finding the history path in the mother -> children direction

bool DireHistory::equalClustering( DireClustering c1 , DireClustering c2 ) {

  // Check if clustering members are equal.
  bool isIdenticalClustering
    =  (c1.emittor     == c2.emittor)
    && (c1.emitted     == c2.emitted)
    && (c1.recoiler    == c2.recoiler)
    && (c1.partner     == c2.partner)
    && (c1.pT()        == c2.pT())
    && (c1.spinRadBef  == c2.spinRadBef)
    && (c1.flavRadBef  == c2.flavRadBef)
    && (c1.splitName   == c2.splitName);
  if (isIdenticalClustering) return true;

  // Require identical recoiler.
  if (c1.recoiler != c2.recoiler) return false;
  // Require same names.
  if (c1.name() != c2.name())     return false;

  // For unequal clusterings, splitting can still be identical, if symmetric
  // in radiator and emission.
  if (c1.emittor != c2.emitted || c1.emitted != c2.emittor) return false;

  bool isIdenticalSplitting = false;
  if (fsr && c1.rad()->isFinal() && c2.rad()->isFinal())
    isIdenticalSplitting = fsr->isSymmetric(c1.name(),c1.rad(),c1.emt());
  else if (isr && !c1.rad()->isFinal() && !c2.rad()->isFinal())
    isIdenticalSplitting = isr->isSymmetric(c1.name(),c1.rad(),c1.emt());

  return isIdenticalSplitting;

}

//--------------------------------------------------------------------------

// Chose dummy scale for event construction. By default, choose
//     sHat     for 2->Boson(->2)+ n partons processes and
//     M_Boson  for 2->Boson(->)             processes

double DireHistory::choseHardScale( const Event& event ) const {

  // Get sHat
  double mHat = (event[3].p() + event[4].p()).mCalc();

  // Find number of final state particles and bosons
  int nFinal = 0;
  int nFinBos= 0;
  int nBosons= 0;
  double mBos = 0.0;
  for(int i = 0; i < event.size(); ++i)
    if ( event[i].isFinal() ) {
      nFinal++;
      // Remember final state unstable bosons
      if ( event[i].idAbs() == 23
        || event[i].idAbs() == 24 ) {
          nFinBos++;
          nBosons++;
          mBos += event[i].m();
      }
    } else if ( abs(event[i].status()) == 22
             && (  event[i].idAbs() == 23
                || event[i].idAbs() == 24 )) {
      nBosons++;
      mBos += event[i].m(); // Real mass
    }

  // Return averaged boson masses
  if ( nBosons > 0 && (nFinal + nFinBos*2) <= 3)
    return (mBos / double(nBosons));
  else return
    mHat;
}


//--------------------------------------------------------------------------

// If the state has an incoming hadron return the flavour of the
// parton entering the hard interaction. Otherwise return 0

int DireHistory::getCurrentFlav(const int side) const {
  int in = (side == 1) ? 3 : 4;
  return state[in].id();
}

//--------------------------------------------------------------------------

double DireHistory::getCurrentX(const int side) const {
  int in = (side == 1) ? 3 : 4;
  return ( 2.*state[in].e()/state[0].e() );
}

//--------------------------------------------------------------------------

double DireHistory::getCurrentZ(const int rad,
  const int rec, const int emt, int idRadBef) const {

  int type = state[rad].isFinal() ? 1 : -1;
  double z = 0.;

  if (type == 1) {

    Vec4 radAfterBranch(state[rad].p());
    Vec4 recAfterBranch(state[rec].p());
    Vec4 emtAfterBranch(state[emt].p());

    // Store masses both after and prior to emission.
    double m2RadAft = radAfterBranch.m2Calc();
    double m2EmtAft = emtAfterBranch.m2Calc();
    double m2RadBef = 0.;
    if ( state[rad].idAbs() != 21 && state[rad].idAbs() != 22
      && state[emt].idAbs() != 24 && state[rad].idAbs() != state[emt].idAbs())
      m2RadBef = m2RadAft;
    else if ( state[emt].idAbs() == 24) {
      if (idRadBef != 0)
        m2RadBef = pow2(particleDataPtr->m0(abs(idRadBef)));
    }

    double Qsq   = (radAfterBranch + emtAfterBranch).m2Calc();

    // Calculate dipole invariant mass.
    double m2final
      = (radAfterBranch + recAfterBranch + emtAfterBranch).m2Calc();
    // More complicated for initial state recoiler.
    if ( !state[rec].isFinal() ){
      double mar2  = m2final - 2. * Qsq + 2. * m2RadBef;
       recAfterBranch *=  (1. - (Qsq - m2RadBef)/(mar2 - m2RadBef))
                         /(1. + (Qsq - m2RadBef)/(mar2 - m2RadBef));
       // If Qsq is larger than mar2 the event is not kinematically possible.
       // Just return random z, since clustering will be discarded.
       if (Qsq > mar2) return 0.5;
    }

    Vec4   sum   = radAfterBranch + recAfterBranch + emtAfterBranch;
    double m2Dip = sum.m2Calc();
    // Construct 2->3 variables for FSR
    double x1 = 2. * (sum * radAfterBranch) / m2Dip;
    double x2 = 2. * (sum * recAfterBranch) / m2Dip;

    // Prepare for more complicated z definition for massive splittings.
    double lambda13 = sqrt( pow2(Qsq - m2RadAft - m2EmtAft )
                         - 4.*m2RadAft*m2EmtAft);
    double k1 = ( Qsq - lambda13 + (m2EmtAft - m2RadAft ) ) / ( 2. * Qsq );
    double k3 = ( Qsq - lambda13 - (m2EmtAft - m2RadAft ) ) / ( 2. * Qsq );
    // Calculate z of splitting, different for FSR
    z = 1./ ( 1- k1 -k3) * ( x1 / (2.-x2) - k3);

  } else {
    // Construct momenta of dipole before/after splitting for ISR
    Vec4 qBR(state[rad].p() - state[emt].p() + state[rec].p());
    Vec4 qAR(state[rad].p() + state[rec].p());
    // Calculate z of splitting, different for ISR
    z = (qBR.m2Calc())/( qAR.m2Calc());
  }

  return z;

}

//--------------------------------------------------------------------------

// Function to compute "pythia pT separation" from Particle input

double DireHistory::pTLund(const Event& event, int rad, int emt, int rec,
  string name) {

  // Use external shower for merging.
  map<string,double> stateVars;

  bool hasPartonLevel(showers && showers->timesPtr && showers->spacePtr),
       hasShowers(fsr && isr);
  if (hasPartonLevel) {
    bool isFSR = showers->timesPtr->isTimelike(event, rad, emt, rec, "");
    if (isFSR) stateVars = showers->timesPtr->getStateVariables
                 (event, rad,emt,rec, name);
    else       stateVars = showers->spacePtr->getStateVariables
                 (event, rad,emt,rec, name);
  } else if (hasShowers) {
    bool isFSR = fsr->isTimelike(event, rad, emt, rec, "");
    if (isFSR) stateVars = fsr->getStateVariables(event, rad,emt,rec, name);
    else       stateVars = isr->getStateVariables(event, rad,emt,rec, name);
  }

  return ( (stateVars.size() > 0 && stateVars.find("t") != stateVars.end())
           ? sqrt(stateVars["t"]) : -1.0 );
}

//--------------------------------------------------------------------------

// Function to return the position of the initial line before (or after)
// a single (!) splitting.

int DireHistory::posChangedIncoming(const Event& event, bool before) {

  // Check for initial state splittings.
  // Consider a splitting to exist if both mother and sister were found.
  // Find sister
  int iSister = 0;
  for (int i =0; i < event.size(); ++i)
    if (event[i].status() == 43) {
      iSister = i;
      break;
    }
  // Find mother
  int iMother = 0;
  if (iSister > 0) iMother = event[iSister].mother1();

  // Initial state splitting has been found.
  if (iSister > 0 && iMother > 0) {

    // Find flavour, mother flavour
    int flavSister  = event[iSister].id();
    int flavMother  = event[iMother].id();

    // Find splitting flavour
    int flavDaughter = 0;
    if ( abs(flavMother) < 21 && flavSister     == 21)
      flavDaughter = flavMother;
    else if ( flavMother     == 21 && flavSister     == 21)
      flavDaughter = flavMother;
    else if ( flavMother     == 21 && abs(flavSister) < 21)
      flavDaughter = -1*flavSister;
    else if ( abs(flavMother) < 21 && abs(flavSister) < 21)
      flavDaughter = 21;

    // Find initial state (!) daughter
    int iDaughter = 0;
    for (int i =0; i < event.size(); ++i)
      if ( !event[i].isFinal()
        && event[i].mother1() == iMother
        && event[i].id()      == flavDaughter )
        iDaughter = i;

    // Done for initial state splitting.
    if ( !before ) return iMother;
    else return iDaughter;

  }

  // Check for final state splittings with initial state recoiler.
  // Consider a splitting to exist if both mother and daughter were found.
  // Find new mother
  iMother = 0;
  for (int i =0; i < event.size(); ++i)
    if ( abs(event[i].status()) == 53 || abs(event[i].status()) == 54) {
      iMother = i;
      break;
    }
  // Find daughter
  int iDaughter = 0;
  if (iMother > 0) iDaughter = event[iMother].daughter1();

  // Done if final state splitting has been found.
  if (iDaughter > 0 && iMother > 0) {

    // Done for final state splitting.
    if ( !before ) return iMother;
    else return iDaughter;

  }

  // If no splitting has been found, return zero.
  return 0;

}

//--------------------------------------------------------------------------

vector<int> DireHistory::getSplittingPos(const Event& e, int type) {

  // Get radiators and recoilers before and after splitting.
  int iRadBef(-1), iRecBef(-1), iRadAft(-1), iEmt(-1), iRecAft(-1);
  // ISR
  if (type == 2) {
    // Loop through event to find new particles.
    for (int i = e.size() - 1; i > 0; i--) {
      if      ( iRadAft == -1 && e[i].status() == -41) iRadAft = i;
      else if ( iEmt    == -1 && e[i].status() ==  43) iEmt    = i;
      else if ( iRecAft == -1
             && (e[i].status() == -42 || e[i].status() == 48) ) iRecAft = i;
      if (iRadAft != -1 && iEmt != -1 && iRecAft != -1) break;
    }
    // Radiator before emission.
    iRadBef = (iRadAft > 0) ?  e[iRadAft].daughter2() : -1;
    // Recoiler before emission.
    iRecBef = (iRecAft > 0) ? (e[iRecAft].isFinal()
            ? e[iRecAft].mother1() : e[iRecAft].daughter1()) : -1;

  // FSR
  } else if (type >= 3) {

    // Recoiler after branching.
    if ( e[e.size() - 1].status() ==  52
      || e[e.size() - 1].status() == -53
      || e[e.size() - 1].status() == -54) iRecAft = (e.size() - 1);
    // Emission after branching.
    if ( e[e.size() - 2].status() == 51) iEmt = (e.size() - 2);
    // Radiator after branching.
    if ( e[e.size() - 3].status() == 51) iRadAft = (e.size() - 3);
    // Radiator before emission.
    iRadBef = (iRadAft > 0) ?  e[iRadAft].mother1() : -1;
    // Recoiler before emission.
    iRecBef = (iRecAft > 0) ? (e[iRecAft].isFinal()
            ? e[iRecAft].mother1() : e[iRecAft].daughter1()) : -1;
  }

  vector<int> ret;
  if ( iRadBef != -1
    && iRecBef != -1
    && iRadAft != -1
    && iEmt    != -1
    && iRecAft != -1)
    ret = createvector<int>(iRadBef)(iRecBef)(iRadAft)(iRecAft)(iEmt);

  return ret;

}

double DireHistory::pdfFactor( const Event&, const Event& e, const int type,
  double pdfScale, double mu ) {

  double wt = 1.;

  // Do nothing for MPI
  if (type < 2) return 1.0;

  // Get radiators and recoilers before and after splitting.
  vector<int> splitPos = getSplittingPos(e, type);
  if (splitPos.size() < 5) return 1.0;
  int iRadBef = splitPos[0];
  int iRecBef = splitPos[1];
  int iRadAft = splitPos[2];
  int iRecAft = splitPos[3];

  bool useSummedPDF
    = infoPtr->settingsPtr->flag("ShowerPDF:useSummedPDF");

  // Final-final splittings
  if        ( e[iRadAft].isFinal() &&  e[iRecAft].isFinal() ) {
    return 1.0;

  // Final-initial splittings
  } else if ( e[iRadAft].isFinal() && !e[iRecAft].isFinal() ) {

    // Find flavour and x values.
    int flavAft    = e[iRecAft].id();
    int flavBef    = e[iRecBef].id();
    double xAft    = 2.*e[iRecAft].e() / e[0].e();
    double xBef    = 2.*e[iRecBef].e() / e[0].e();
    bool hasPDFAft = (particleDataPtr->colType(flavAft) != 0);
    bool hasPDFBef = (particleDataPtr->colType(flavBef) != 0);

    // Calculate PDF weight to reweight emission to emission evaluated at
    // constant factorisation scale. No need to include the splitting kernel in
    // the weight, since it will drop out anyway.
    int sideSplit = ( e[iRecAft].pz() > 0.) ? 1 : -1;
    double pdfDen1, pdfDen2, pdfNum1, pdfNum2;
    pdfDen1 = pdfDen2 = pdfNum1 = pdfNum2 = 1.;
    if ( sideSplit == 1 ) {
      pdfDen1 = (!hasPDFBef) ? 1.0 : (useSummedPDF)
              ? beamA.xf(flavBef, xBef, pow2(mu))
              : beamA.xfISR(0, flavBef, xBef, pow2(mu));
      pdfNum1 = (!hasPDFBef) ? 1.0 : (useSummedPDF)
              ? beamA.xf(flavBef, xBef, pow2(pdfScale))
              : beamA.xfISR(0, flavBef, xBef, pow2(pdfScale));
      pdfNum2 = (!hasPDFAft) ? 1.0 : (useSummedPDF)
              ? beamA.xf(flavAft, xAft, pow2(mu))
              : beamA.xfISR(0, flavAft, xAft, pow2(mu));
      pdfDen2 = (!hasPDFAft) ? 1.0 : (useSummedPDF)
              ? beamA.xf(flavAft, xAft, pow2(pdfScale))
              : beamA.xfISR(0, flavAft, xAft, pow2(pdfScale));
    } else {
      pdfDen1 = (!hasPDFBef) ? 1.0 : (useSummedPDF)
              ? beamB.xf(flavBef, xBef, pow2(mu))
              : beamB.xfISR(0, flavBef, xBef, pow2(mu));
      pdfNum1 = (!hasPDFBef) ? 1.0 : (useSummedPDF)
              ? beamB.xf(flavBef, xBef, pow2(pdfScale))
              : beamB.xfISR(0, flavBef, xBef, pow2(pdfScale));
      pdfNum2 = (!hasPDFAft) ? 1.0 : (useSummedPDF)
              ? beamB.xf(flavAft, xAft, pow2(mu))
              : beamB.xfISR(0, flavAft, xAft, pow2(mu));
      pdfDen2 = (!hasPDFAft) ? 1.0 : (useSummedPDF)
              ? beamB.xf(flavAft, xAft, pow2(pdfScale))
              : beamB.xfISR(0, flavAft, xAft, pow2(pdfScale));
    }
    wt = (pdfNum1/pdfDen1) * (pdfNum2)/(pdfDen2);

  // Initial-final splittings
  } else if ( !e[iRadAft].isFinal() &&  e[iRecAft].isFinal() ) {

    // Find flavour and x values.
    int flavAft    = e[iRadAft].id();
    int flavBef    = e[iRadBef].id();
    double xAft    = 2.*e[iRadAft].e() / e[0].e();
    double xBef    = 2.*e[iRadBef].e() / e[0].e();
    bool hasPDFAft = (particleDataPtr->colType(flavAft) != 0);
    bool hasPDFBef = (particleDataPtr->colType(flavBef) != 0);

    // Calculate PDF weight to reweight emission to emission evaluated at
    // constant factorisation scale. No need to include the splitting kernel in
    // the weight, since it will drop out anyway.
    int sideSplit = ( e[iRadAft].pz() > 0.) ? 1 : -1;
    double pdfDen1, pdfDen2, pdfNum1, pdfNum2;
    pdfDen1 = pdfDen2 = pdfNum1 = pdfNum2 = 1.;
    if ( sideSplit == 1 ) {
      pdfDen1 = (!hasPDFBef) ? 1.0 : (useSummedPDF)
              ? beamA.xf(flavBef, xBef, pow2(mu))
              : beamA.xfISR(0, flavBef, xBef, pow2(mu));
      pdfNum1 = (!hasPDFBef) ? 1.0 : (useSummedPDF)
              ? beamA.xf(flavBef, xBef, pow2(pdfScale))
              : beamA.xfISR(0, flavBef, xBef, pow2(pdfScale));
      pdfNum2 = (!hasPDFAft) ? 1.0 : (useSummedPDF)
              ? beamA.xf(flavAft, xAft, pow2(mu))
              : beamA.xfISR(0, flavAft, xAft, pow2(mu));
      pdfDen2 = (!hasPDFAft) ? 1.0 : (useSummedPDF)
              ? beamA.xf(flavAft, xAft, pow2(pdfScale))
              : beamA.xfISR(0, flavAft, xAft, pow2(pdfScale));
    } else {
      pdfDen1 = (!hasPDFBef) ? 1.0 : (useSummedPDF)
              ? beamB.xf(flavBef, xBef, pow2(mu))
              : beamB.xfISR(0, flavBef, xBef, pow2(mu));
      pdfNum1 = (!hasPDFBef) ? 1.0 : (useSummedPDF)
              ? beamB.xf(flavBef, xBef, pow2(pdfScale))
              : beamB.xfISR(0, flavBef, xBef, pow2(pdfScale));
      pdfNum2 = (!hasPDFAft) ? 1.0 : (useSummedPDF)
              ? beamB.xf(flavAft, xAft, pow2(mu))
              : beamB.xfISR(0, flavAft, xAft, pow2(mu));
      pdfDen2 = (!hasPDFAft) ? 1.0 : (useSummedPDF)
              ? beamB.xf(flavAft, xAft, pow2(pdfScale))
              : beamB.xfISR(0, flavAft, xAft, pow2(pdfScale));
    }
    wt = (pdfNum1/pdfDen1) * (pdfNum2)/(pdfDen2);


  // Initial-initial splittings
  } else if ( !e[iRadAft].isFinal() && !e[iRecAft].isFinal() ) {

    // Find flavour and x values.
    int flavAft    = e[iRadAft].id();
    int flavBef    = e[iRadBef].id();
    double xAft    = 2.*e[iRadAft].e() / e[0].e();
    double xBef    = 2.*e[iRadBef].e() / e[0].e();

    // Calculate PDF weight to reweight emission to emission evaluated at
    // constant factorisation scale. No need to include the splitting kernel
    // in the weight, since it will drop out anyway.
    int sideSplit = ( e[iRadAft].pz() > 0.) ? 1 : -1;
    double ratio1 = getPDFratio( sideSplit, false, false, flavBef,
                      xBef, pdfScale, flavBef, xBef, mu );
    double ratio2 = getPDFratio( sideSplit, false, false, flavAft,
                      xAft, mu, flavAft, xAft, pdfScale );

    wt = ratio1*ratio2;

  }

  // Done
  return wt;
}

//--------------------------------------------------------------------------

// Function giving the product of splitting kernels and PDFs so that the
// resulting flavour is given by flav. This is used as a helper routine
// to dgauss

double DireHistory::integrand(int flav, double x, double scaleInt, double z) {

  // Colour factors.
  double CA = infoPtr->settingsPtr->parm("DireColorQCD:CA") > 0.0
            ? infoPtr->settingsPtr->parm("DireColorQCD:CA") : 3.0;
  double CF = infoPtr->settingsPtr->parm("DireColorQCD:CF") > 0.0
            ? infoPtr->settingsPtr->parm("DireColorQCD:CF") : 4./3.;
  double TR = infoPtr->settingsPtr->parm("DireColorQCD:TR") > 0.
            ? infoPtr->settingsPtr->parm("DireColorQCD:TR") : 0.5;

  double result = 0.;

  // Integrate NLL sudakov remainder
  if (flav==0) {

    AlphaStrong* as = mergingHooksPtr->AlphaS_ISR();
    double asNow = (*as).alphaS(z);
    result = 1./z *asNow*asNow* ( log(scaleInt/z) -3./2. );

  // Integrand for PDF ratios. Careful about factors if 1/z, since formulae
  // are expressed in terms if f(x,mu), while Pythia uses x*f(x,mu)!
  } else if (flav==21) {

    double measure1 = 1./(1. - z);
    double measure2 = 1.;

    double integrand1 =
      2.*CA
      * z * beamB.xf( 21,x/z,pow(scaleInt,2))
          / beamB.xf( 21,x,  pow(scaleInt,2))
    - 2.*CA;

    double integrand2 =
      // G -> G terms
      2.*CA  *((1. -z)/z + z*(1.-z))
      * beamB.xf( 21,x/z,pow(scaleInt,2))
      / beamB.xf( 21,x,  pow(scaleInt,2))
      // G -> Q terms
    + CF * ((1+pow(1-z,2))/z)
      *( beamB.xf(  1, x/z,pow(scaleInt,2))
       / beamB.xf( 21, x,  pow(scaleInt,2))
       + beamB.xf( -1, x/z,pow(scaleInt,2))
       / beamB.xf( 21, x,  pow(scaleInt,2))
       + beamB.xf(  2, x/z,pow(scaleInt,2))
       / beamB.xf( 21, x,  pow(scaleInt,2))
       + beamB.xf( -2, x/z,pow(scaleInt,2))
       / beamB.xf( 21, x,  pow(scaleInt,2))
       + beamB.xf(  3, x/z,pow(scaleInt,2))
       / beamB.xf( 21, x,  pow(scaleInt,2))
       + beamB.xf( -3, x/z,pow(scaleInt,2))
       / beamB.xf( 21, x,  pow(scaleInt,2))
       + beamB.xf(  4, x/z,pow(scaleInt,2))
       / beamB.xf( 21, x,  pow(scaleInt,2))
       + beamB.xf( -4, x/z,pow(scaleInt,2))
       / beamB.xf( 21, x,  pow(scaleInt,2)) );

    // Done
    result = integrand1*measure1 + integrand2*measure2;

  } else {

    double measure1 = 1./(1. -z);
    double measure2 = 1.;

    // Q -> Q terms
    double integrand1 =
      CF * (1+pow(z,2))
      * beamB.xf( flav, x/z, pow(scaleInt,2))
      / beamB.xf( flav, x,   pow(scaleInt,2))
    - 2.*CF;

    // Q -> G terms
    double integrand2 =
    + TR * (pow(z,2) + pow(1-z,2))
      * beamB.xf( 21,   x/z, pow(scaleInt,2))
      / beamB.xf( flav, x,   pow(scaleInt,2));

    // Done
    result = measure1*integrand1 + measure2*integrand2;
  }

  return result;

}

//--------------------------------------------------------------------------

// Function providing a list of possible new flavours after a w emssion
// from the input flavour.

vector<int> DireHistory::posFlavCKM(int flav) {

  // absolute values!
  int flavAbs = abs(flav);

  vector<int> flavRadBefs;
  // (e,mu,tau)
  if (flavAbs > 10 && flavAbs % 2 == 1)
    flavRadBefs.push_back(flavAbs + 1);
  // (neutrinoes)
  else if (flavAbs > 10 && flavAbs % 2 == 0)
    flavRadBefs.push_back(flavAbs - 1);
  // Full CKM for quarks.
  else if (flavAbs < 10 && flavAbs % 2 == 1) {
    flavRadBefs.push_back(2);
    flavRadBefs.push_back(4);
    flavRadBefs.push_back(6);
  }
  else if (flavAbs < 10 && flavAbs % 2 == 0) {
    flavRadBefs.push_back(1);
    flavRadBefs.push_back(3);
    flavRadBefs.push_back(5);
  }

  // Return the possible flavours.
  return flavRadBefs;
}

//--------------------------------------------------------------------------

// Check if the new flavour structure is possible.
// If clusType is 1 final clustering is assumed, otherwise initial clustering
// is assumed.

bool DireHistory::checkFlavour(vector<int>& flavCounts, int flavRad,
                           int flavRadBef, int clusType) {

  // Loop over event.
  for(int k = 0; k < 20; ++k) {
    // Find changes from this W emission.
    int cor = 0;
    if (abs(flavRad) == k) {
      cor = -1;
      if (flavRad < 0) cor = 1;
    }

    if (abs(flavRadBef) == k) {
      cor = 1;
      if (flavRadBef < 0) cor = -1;
    }

    // if flavour and flavRadBef is the same, no correction.
    if (flavRadBef == flavRad) cor = 0;

    // Check if flavour is consistent.
    if (clusType == 1) {
      if (flavCounts[k] + cor != 0) return false;
    }
    else
      if (flavCounts[k] - cor != 0) return false;
  }

 // No more checks.
 return true;

}

//--------------------------------------------------------------------------

// Check if an event reclustered into a 2 -> 2 dijet.
// (Only enabled if W reclustering is used).
bool DireHistory::isQCD2to2(const Event& event) {

  if (!mergingHooksPtr->doWeakClustering()) return false;
  int nFinalPartons = 0, nFinal = 0;
  for (int i = 0;i < event.size();++i)
    if (event[i].isFinal()) {
      nFinal++;
      if ( event[i].idAbs() < 10 || event[i].idAbs() == 21)
        nFinalPartons++;
    }
  if (nFinalPartons == 2 && nFinal == 2) return true;
  else return false;

}

//--------------------------------------------------------------------------

// Check if an event reclustered into a 2 -> 1 Drell-Yan.
// (Only enabled if W reclustering is used).
bool DireHistory::isEW2to1(const Event& event) {

  if (!mergingHooksPtr->doWeakClustering()) return false;
  int nVector = 0;
  for (int i = 0;i < event.size();++i) {
    if (event[i].isFinal()) {
      if (event[i].idAbs() == 23 ||
         event[i].idAbs() == 24 ||
         event[i].idAbs() == 22) nVector++;
      else return false;
    }
  }

  // Only true if a single vector boson as outgoing process.
  if (nVector == 1) return true;

  // Done
  return false;

}

//--------------------------------------------------------------------------

// Check if an event reclustered into massless 2 -> 2.

bool DireHistory::isMassless2to2(const Event& event) {

  int nFmassless(0), nFinal(0), nImassless(0);
  for (int i = 0;i < event.size();++i)
    if (event[i].isFinal()) {
      nFinal++;
      if ( event[i].idAbs() < 10
        || event[i].idAbs() == 21
        || event[i].idAbs() == 22) nFmassless++;
    } else if ( event[i].status() == -21 ) {
      if ( event[i].idAbs() < 10
        || event[i].idAbs() == 21
        || event[i].idAbs() == 22) nImassless++;
    }
  if (nFmassless == 2 && nFinal == 2 && nImassless == 2) return true;

  // Done
  return false;

}

//--------------------------------------------------------------------------

// Check if an event reclustered into DIS 2 -> 2.
bool DireHistory::isDIS2to2(const Event& event) {

  int nFpartons(0), nFleptons(0), nIpartons(0), nIleptons(0), nFinal(0);
  for (int i = 0;i < event.size();++i)
    if (event[i].isFinal()) {
      if ( event[i].isLepton() )     nFleptons++;
      if ( event[i].colType() != 0 ) nFpartons++;
      nFinal++;
    } else if ( event[i].status() == -21 ) {
      if ( event[i].isLepton() )     nIleptons++;
      if ( event[i].colType() != 0 ) nIpartons++;
    }
  bool isDIS =  nFinal == 2 && nFpartons == 1 && nFleptons == 1
             && nIpartons == 1 && nIleptons == 1;
  if (isDIS) return true;

  // Done.
  return false;
}

// Function to allow effective gg -> EW boson couplings.
bool DireHistory::mayHaveEffectiveVertex( string process, vector<int> in,
  vector<int> out) {

  if ( process.compare("ta+ta->jj") == 0
    || process.compare("ta-ta+>jj") == 0 ) {
    int nInFermions(0), nOutFermions(0), nOutBosons(0);
    for (int i=0; i < int(in.size()); ++i)
      if (abs(in[i])<20) nInFermions++;
    for (int i=0; i < int(out.size()); ++i) {
      if (abs(out[i])<20) nOutFermions++;
      if (abs(out[i])>20) nOutBosons++;
    }
    return (nInFermions%2==0 && nOutFermions%2==0);
  }

  int nInG(0), nOutZ(0), nOutWp(0), nOutWm(0), nOutH(0), nOutA(0), nOutG(0);
  for (int i=0; i < int(in.size()); ++i)
    if (in[i]==21) nInG++;
  for (int i=0; i < int(out.size()); ++i) {
    if (out[i] == 21) nOutG++;
    if (out[i] == 22) nOutA++;
    if (out[i] == 23) nOutZ++;
    if (out[i] == 24) nOutWp++;
    if (out[i] ==-24) nOutWm++;
    if (out[i] == 25) nOutH++;
  }

  if ( nInG==2 && nOutWp+nOutWm > 0 && nOutWp+nOutWm == int(out.size())
    && nOutWp-nOutWm == 0)
    return true;
  if (nInG+nOutG>0 && nOutH > 0)
    return true;

  if ( process.find("Hinc") != string::npos
    && process.find("Ainc") != string::npos
    && (nOutH > 0 || nOutA%2==0) )
    return true;

  return false;
}

//--------------------------------------------------------------------------

// Set selected child indices.

void DireHistory::setSelectedChild() {

  if (mother == nullptr) return;
  for (int i = 0;i < int(mother->children.size());++i)
    if (mother->children[i] == this) mother->selectedChild = i;
  mother->setSelectedChild();
}

//--------------------------------------------------------------------------

// Store index of children that pass "trimHistories".

void DireHistory::setGoodChildren() {
  if (mother == nullptr) return;
  for (int i = 0;i < int(mother->children.size());++i)
    if (mother->children[i] == this) {
      // Nothing to be done if good child is already tagged.
      if ( find(mother->goodChildren.begin(), mother->goodChildren.end(), i)
        != mother->goodChildren.end() ) continue;
     mother->goodChildren.push_back(i);
    }
  mother->setGoodChildren();
}

//--------------------------------------------------------------------------

// Store index of children that pass "trimHistories".

void DireHistory::setGoodSisters() {

  for (int i = 0;i < int(goodChildren.size());++i) {
    for (int j = 0;j < int(goodChildren.size());++j) {
      children[i]->goodSisters.push_back(children[j]);
    }
    children[i]->setGoodSisters();
  }
  if (!mother) goodSisters.push_back(this);

}

//--------------------------------------------------------------------------

// Store index of children that pass "trimHistories".

void DireHistory::printMECS() {

  if ( !mother && children.size() > 0 && (MECnum/MECden > 1e2 )) {
    cout << scientific << setprecision(6);
    listFlavs(state);
    cout << " " << goodChildren.size() << " num " << MECnum
         << " den " << MECden << endl;
  }
  if (mother ) mother->printMECS();
  return;

}

//--------------------------------------------------------------------------

void DireHistory::setProbabilities() {

  for (int i = 0;i < int(goodSisters.size());++i) {

    DireHistory* sisterNow = goodSisters[i];
    bool foundOrdered=false;
    double denominator=0.;
    double contrib=0.;
    double denominator_unconstrained=0.;
    double contrib_unconstrained=0.;

    for (int j = 0;j < int(sisterNow->goodChildren.size());++j) {

      DireHistory* childNow = sisterNow->children[j];

      double virtuality = 2.*(childNow->clusterIn.rad()->p()*
                              childNow->clusterIn.emt()->p());

      // Get clustering variables.
      map<string,double> stateVars;
      int rad = childNow->clusterIn.radPos();
      int emt = childNow->clusterIn.emtPos();
      int rec = childNow->clusterIn.recPos();

      bool hasPartonLevel(showers && showers->timesPtr && showers->spacePtr),
           hasShowers(fsr && isr), isFSR(false);
      if (hasPartonLevel) {
        isFSR = showers->timesPtr->isTimelike
          (sisterNow->state, rad, emt, rec, "");
        if (isFSR) stateVars = showers->timesPtr->getStateVariables
                     (sisterNow->state,rad,emt,rec,"");
        else       stateVars = showers->spacePtr->getStateVariables
                     (sisterNow->state,rad,emt,rec,"");
      } else if (hasShowers) {
        isFSR = fsr->isTimelike(sisterNow->state, rad, emt, rec, "");
        if (isFSR) stateVars = fsr->getStateVariables
                     (sisterNow->state,rad,emt,rec,"");
        else       stateVars = isr->getStateVariables
                     (sisterNow->state,rad,emt,rec,"");
      }

      double z   = stateVars["z"];
      double t   = stateVars["t"];
      double Q2  = stateVars["m2dip"];
      double xCS = 1.;
      // For splittings with initial-state particles, remember initial
      // momentum rescaling.
      if        ( !isFSR && !sisterNow->state[rec].isFinal() ) {
        double kappa2 = t/Q2;
        xCS = (z*(1-z) - kappa2) / (1 -z);
      } else if ( !isFSR &&  sisterNow->state[rec].isFinal() ) {
        xCS = z;
      } else if (  isFSR && !sisterNow->state[rec].isFinal() ) {
        double kappa2 = t/Q2;
        xCS    = 1 - kappa2/(1.-z);
        if (abs(Q2) < 1e-5) xCS = 1.;
      }

      double dd = childNow->MECnum
                * childNow->clusterProb
                // clusterProb includes 1/pT2, but here, we only want
                // the propagator factor 1/(2.*pipj)
                * pow2(childNow->clusterIn.pT()) / virtuality
                // extra factor left-over from 8*Pi*alphaS of dipole ME
                // factorization:
                // |Mn+1|^2 ~ 8*Pi*alphaS*1/(2.pipj)*1/x*dipole*|Mn|^2
                //          = 2g^2 *1/(2.pipj)*1/x*dipole*|Mn|^2
                //          compared with using g^2=1 in MG5 MEs.
                // * 2.
                // Part of the definition of dipole factorization.
                * 1 / xCS
                / childNow->MECden * childNow->MECcontrib;

      // Multiply coupling.
      double coupl = 1.;
      string name = childNow->clusterIn.name();
      if (hasPartonLevel) {
        if ( name.find("qcd") != string::npos
          || name.find("qed") != string::npos)
          coupl = childNow->clusterCoupl * 2. * M_PI * 8. * M_PI;
        else
          coupl = childNow->clusterCoupl;
      } else if (hasShowers) {
        if ( isFSR && ( fsr->splits[name]->is_qcd
                     || fsr->splits[name]->is_qed))
          coupl = childNow->clusterCoupl * 2. * M_PI * 8. * M_PI;
        else if (       isr->splits[name]->is_qcd
                     || isr->splits[name]->is_qed)
          coupl = childNow->clusterCoupl * 2. * M_PI * 8. * M_PI;
        else
          coupl = childNow->clusterCoupl;
      }

      dd *= coupl;

      denominator_unconstrained += dd;

      // Remember if this child contributes to the next-higher denominator.
      if (childNow->clusterIn.pT() > sisterNow->clusterIn.pT()) {
        contrib_unconstrained += dd;
        foundOrdered=true;
      }

      // Careful about first emissions above the factorization scale.
      if ( int(childNow->goodChildren.size()) == 0
        && hardStartScale(childNow->state) > childNow->clusterIn.pT()) {
        denominator += dd;
        if (childNow->clusterIn.pT() > sisterNow->clusterIn.pT())
          contrib += dd;
      } else if (int(childNow->goodChildren.size()) > 0) {
        denominator += dd;
        if (childNow->clusterIn.pT() > sisterNow->clusterIn.pT())
          contrib += dd;
      }

    }

    // Update ME correction pieces in sister node.
    if (sisterNow->children.size() > 0) {
      if (denominator != 0.0) goodSisters[i]->MECden = denominator;
      goodSisters[i]->MECcontrib = contrib;
      if (denominator == 0. && contrib == 0.) {
        if (denominator_unconstrained != 0.0)
          goodSisters[i]->MECden = denominator_unconstrained;
        goodSisters[i]->MECcontrib = contrib_unconstrained;
      }

      if (!foundOrdered) goodSisters[i]->foundOrderedChildren = false;

    }

  }

  if (mother) mother->setProbabilities();

  return;

}

//--------------------------------------------------------------------------

void DireHistory::setEffectiveScales() {

  for (int i = 0;i < int(goodSisters.size());++i) {

    DireHistory* sisterNow = goodSisters[i];

    // Nothing to do if there are no children.
    if (sisterNow->goodChildren.size()==0) continue;

    double alphasOftEff_num(0.), alphasOftEff_den(0.), tmin(1e15), tmax(-1.0);

    for (int j = 0;j < int(sisterNow->goodChildren.size());++j) {

      DireHistory* childNow = sisterNow->children[j];

      // Scale with correct coupling factor.
      double t(pow2(childNow->clusterIn.pT()));
      tmax = max(tmax,t);
      tmin = min(tmin,t);

      // We will want to choose an effective scale by solving
      // as(teff) = (sum of kernel*propagator*MEC) . Note that we DO NOT
      // include the Jacobian here, and consequently have to remove its
      // contribution from clusterProb. Also, an extra 1/x-factor has
      // to be included for inital-state splittings.
      double massSign   = childNow->clusterIn.rad()->isFinal() ? 1. : -1.;
      double virtuality = massSign*(childNow->clusterIn.rad()->p()
                         + massSign*childNow->clusterIn.emt()->p()).m2Calc();
      // Get clustering variables.
      map<string,double> stateVars;
      int rad = childNow->clusterIn.radPos();
      int emt = childNow->clusterIn.emtPos();
      int rec = childNow->clusterIn.recPos();
      bool hasPartonLevel(showers && showers->timesPtr && showers->spacePtr),
           hasShowers(fsr && isr), isFSR(false);
      if (hasPartonLevel) {
        isFSR = showers->timesPtr->isTimelike
          (sisterNow->state, rad, emt, rec, "");
        if (isFSR) stateVars = showers->timesPtr->getStateVariables
                     (sisterNow->state,rad,emt,rec,"");
        else       stateVars = showers->spacePtr->getStateVariables
                     (sisterNow->state,rad,emt,rec,"");
      } else if (hasShowers) {
        isFSR = fsr->isTimelike(sisterNow->state, rad, emt, rec, "");
        if (isFSR) stateVars = fsr->getStateVariables
                     (sisterNow->state,rad,emt,rec,"");
        else       stateVars = isr->getStateVariables
                     (sisterNow->state,rad,emt,rec,"");
      }

      double z   = stateVars["z"];
      double Q2  = stateVars["m2dip"];
      double xCS = 1.;
      // For splittings with initial-state particles, remember initial
      // momentum rescaling.
      if        ( !isFSR && !sisterNow->state[rec].isFinal() ) {
        double kappa2 = t/Q2;
        xCS = (z*(1-z) - kappa2) / (1 -z);
      } else if ( !isFSR &&  sisterNow->state[rec].isFinal() ) {
        xCS = z;
      } else if (  isFSR && !sisterNow->state[rec].isFinal() ) {
        double kappa2 = t/Q2;
        xCS    = 1 - kappa2/(1.-z);
      }

      // Construct coupling.
      double coupling(1.);
      string name  = childNow->clusterIn.name();
      int idRadBef = int(stateVars["radBefID"]);
      int idRec    = sisterNow->state[rec].id();
      if (hasPartonLevel) {
        if      ( name.find("qcd") != string::npos)
          coupling = mergingHooksPtr->AlphaS_FSR()->alphaS(t);
        else if ( name.find("qed") != string::npos)
          coupling = pow2(childNow->clusterIn.rad()->charge())
                   * mergingHooksPtr->AlphaEM_FSR()->alphaEM(t);
        else if ( name.find("ew") != string::npos) {
          int flav = abs(childNow->clusterIn.flavRadBef);
          coupling = 4.*M_PI
               * (pow2(coupSMPtr->rf( flav )) + pow2(coupSMPtr->lf( flav )))
               / (coupSMPtr->sin2thetaW() * coupSMPtr->cos2thetaW());
        }
      } else if (hasShowers) {
        if (isFSR) coupling = 2.*M_PI*fsr->splits[name]->coupling(z,t,Q2,-1.,
            make_pair(idRadBef, sisterNow->state[rad].isFinal()),
            make_pair(idRec,    sisterNow->state[rec].isFinal()) );
        else       coupling = 2.*M_PI*isr->splits[name]->coupling(z,t,Q2,-1.,
            make_pair(idRadBef, sisterNow->state[rad].isFinal()),
            make_pair(idRec,    sisterNow->state[rec].isFinal()) );
      }

      //double prob = childNow->clusterProb
      double prob = abs(childNow->clusterProb)
                // clusterProb includes 1/pT2, but here, we only want
                // the propagator factor 1/(2.*pipj)
                * pow2(childNow->clusterIn.pT()) / virtuality
                // extra factor left-over from 8*Pi*alphaS of dipole ME
                // factorization:
                // |Mn+1|^2 ~ 8*Pi*alphaS*1/(2.pipj)*1/x*dipole*|Mn|^2
                //          = 2g^2 *1/(2.pipj)*1/x*dipole*|Mn|^2
                //          compared with using g^2=1 in MG5 MEs.
                * 2.
                // Part of the definition of dipole factorization.
                * 1 / xCS;

      // Include alphaS at effective scale of previous order.
      double alphasOftEffPrev(childNow->couplEffective);
      alphasOftEff_num += prob * childNow->MECnum * alphasOftEffPrev
        * coupling;
      alphasOftEff_den += prob * childNow->MECnum;

    }

    // Calculate the effective scale.
    // Note: For negative probabilities, it might be necessary to increase the
    // scale range for root finding.
    DireCouplFunction  couplFunc(  mergingHooksPtr->AlphaS_FSR(),
      couplingPowCount["qcd"], mergingHooksPtr->AlphaEM_FSR(),
      couplingPowCount["qed"]);
    double as_tmin = couplFunc.f(tmin,vector<double>());
    double as_tmax = couplFunc.f(tmax,vector<double>());
    double headroom = 1.;
    bool failed = false;
    double alphasOftEffRatio = pow(alphasOftEff_num/alphasOftEff_den,1.);
    while ( tmin/headroom < tmax*headroom
      && ( ( as_tmin-alphasOftEffRatio > 0
          && as_tmax-alphasOftEffRatio > 0)
        || ( as_tmin-alphasOftEffRatio < 0
          && as_tmax-alphasOftEffRatio < 0))) {
      if (tmin/headroom < mergingHooksPtr->pTcut()) { failed = true; break;}
      headroom *= 1.01;
      as_tmin = couplFunc.f(tmin/headroom,vector<double>());
      as_tmax = couplFunc.f(tmax*headroom,vector<double>());
    }

    // Include correct power for effective alphaS.
    DireRootFinder direRootFinder;
    double teff = mergingHooksPtr->pTcut();
    double tminNow(tmin/headroom), tmaxNow(tmax*headroom);
    if (!failed) {
      if  (abs(tmaxNow-tminNow)/tmaxNow < 1e-4) teff = tmaxNow;
      else teff = direRootFinder.findRoot1D( &couplFunc, tminNow, tmaxNow,
        alphasOftEffRatio, vector<double>(), 100);
    }

    // Set the effective scale for currect sister.
    sisterNow->scaleEffective = sqrt(teff);
    sisterNow->couplEffective = alphasOftEffRatio;

  }

  if (mother) mother->setEffectiveScales();

  return;

}

//--------------------------------------------------------------------------

// Function to retrieve scale information from external showers.

double DireHistory::getShowerPluginScale(const Event& event, int rad, int emt,
  int rec, string name, string key, double) {

  map<string,double> stateVars;
  bool hasPartonLevel(showers && showers->timesPtr && showers->spacePtr),
       hasShowers(fsr && isr);
  if (hasPartonLevel) {
    bool isFSR = showers->timesPtr->isTimelike(event, rad, emt, rec, "");
    if (isFSR) stateVars = showers->timesPtr->getStateVariables
                 (event, rad, emt, rec, name);
    else       stateVars = showers->spacePtr->getStateVariables
                 (event, rad, emt, rec, name);
  } else if (hasShowers) {
    bool isFSR = fsr->isTimelike(event, rad, emt, rec, "");
    if (isFSR) stateVars = fsr->getStateVariables(event, rad, emt, rec, name);
    else       stateVars = isr->getStateVariables(event, rad, emt, rec, name);
  }

  return ( (stateVars.size() > 0 && stateVars.find(key) != stateVars.end())
           ? stateVars[key] : -1.0 );

}

//--------------------------------------------------------------------------

// Function to retrieve type of splitting from external showers.

pair<int,double> DireHistory::getCoupling(const Event& event, int rad, int emt,
  int rec, string name) {

  // Retrieve state variables.
  map<string,double> stateVars;
  bool hasPartonLevel(showers && showers->timesPtr && showers->spacePtr),
       hasShowers(fsr && isr);
  if (hasPartonLevel) {
    bool isFSR = showers->timesPtr->isTimelike(event, rad, emt, rec, "");
    if (isFSR) stateVars = showers->timesPtr->getStateVariables
                 (event, rad, emt, rec, name);
    else       stateVars = showers->spacePtr->getStateVariables
                 (event, rad, emt, rec, name);
  } else if (hasShowers) {
    bool isFSR = fsr->isTimelike(event, rad, emt, rec, "");
    if (isFSR) stateVars = fsr->getStateVariables(event, rad, emt, rec, name);
    else       stateVars = isr->getStateVariables(event, rad, emt, rec, name);
  }

  // Get coupling type (identifier of interaction), and get coupling value for
  // the current splitting, i.e. 1 / [4\pi] * g^2 {splitting variables}
  int type     = ( (stateVars.size() > 0
                 && stateVars.find("couplingType") != stateVars.end())
               ?  stateVars["couplingType"] : -1);
  double value = ( (stateVars.size() > 0
                 && stateVars.find("couplingValue") != stateVars.end())
               ?  stateVars["couplingValue"] : -1.0);

  // Done.
  return make_pair(type,value);

}

//--------------------------------------------------------------------------

// Store if path is considered "signal" or "background" according to a
// user-defined criterion.

void DireHistory::tagPath(DireHistory* leaf) {

  int nHiggs = 0;
  for (int i=0; i < state.size(); ++i)
    if (state[i].isFinal() && state[i].id() == 25) nHiggs++;
  // Tag as Higgs signal.
  if (nHiggs > 0) leaf->tagSave.push_back("higgs");
  if (leaf == this) {
    int nFinal(0), nFinalPartons(0), nFinalGamma(0);
    for (int i = 0;i < state.size();++i) {
      if (state[i].isFinal()) {
        nFinal++;
        if ( state[i].idAbs() < 10
          || state[i].idAbs() == 21) nFinalPartons++;
        if ( state[i].idAbs() == 22) nFinalGamma++;
      }
    }
    // Tag as QCD signal.
    if (nFinal == 2 && nFinalPartons == 2)
      leaf->tagSave.push_back("qcd");
    // Tag as QED signal.
    if (nFinal == 2 && nFinalGamma == 2)
      leaf->tagSave.push_back("qed");
    // Tag as QCD and QED signal.
    if (nFinal == 2 && nFinalGamma == 1 && nFinalPartons == 1) {
      leaf->tagSave.push_back("qed");
      leaf->tagSave.push_back("qcd");
    }
  }

  if (mother) mother->tagPath(leaf);
  return;

}

//--------------------------------------------------------------------------

// Multiply ME corrections to the probability of the path.

void DireHistory::multiplyMEsToPath(DireHistory* leaf) {

  if (leaf == this) {
    leaf->prodOfProbsFull *= hardProcessCouplings(state)*clusterCoupl;
    leaf->prodOfProbs     *= abs(hardProcessCouplings(state)*clusterCoupl);
  } else {
    leaf->prodOfProbsFull *= MECnum/MECden*clusterCoupl;
    leaf->prodOfProbs     *= abs(MECnum/MECden*clusterCoupl);
  }

  if (mother) mother->multiplyMEsToPath(leaf);

  return;
}

//--------------------------------------------------------------------------

// Set coupling power counters in the path.

void DireHistory::setCouplingOrderCount(DireHistory* leaf,
  map<string,int> count) {

  string name  = clusterIn.name();
  if (leaf == this) {
    // Count hard process couplings.
    hardProcessCouplings(state, 0, 1., nullptr, nullptr, true);
    // Update with coupling order of clustering.
    count = couplingPowCount;

  } else if (couplingPowCount.empty()) {
    couplingPowCount = count;
  }

  if ( name.find("qcd") != string::npos) count["qcd"]++;
  if ( name.find("qed") != string::npos) count["qed"]++;

  if (mother) mother->setCouplingOrderCount(leaf, count);

  return;
}

//==========================================================================

} // end namespace Pythia8
