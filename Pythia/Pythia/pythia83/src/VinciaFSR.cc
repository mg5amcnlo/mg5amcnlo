// VinciaFSR.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the VinciaFSR class
// and auxiliary classes.

#include "Pythia8/VinciaFSR.h"

namespace Pythia8 {

using namespace VinciaConstants;

// Max loop counter (for interleaved resonance decays).
const int VinciaFSR::NLOOPMAX = 10000;

//==========================================================================

// The Brancher class, base class containing a generic set of "parent
// partons" as well as virtual methods for generating trial
// branchings.

//--------------------------------------------------------------------------

// Reset.

void Brancher::reset(int iSysIn, Event& event, vector<int> iIn) {

  // Save info on parents and resize vectors.
  iSav         = iIn;
  hasTrialSav  = false;
  systemSav    = iSysIn;
  Vec4 pSum;
  int nMassive = 0;
  idSav.resize(iIn.size());
  hSav.resize(iIn.size());
  colTypeSav.resize(iIn.size());
  colSav.resize(iIn.size());
  acolSav.resize(iIn.size());
  mSav.resize(iIn.size());
  for (unsigned int i = 0; i < iIn.size(); ++i) {
    idSav[i]      = event[iIn[i]].id();
    hSav[i]       = event[iIn[i]].pol();
    colTypeSav[i] = event[iIn[i]].colType();
    colSav[i]     = event[iIn[i]].col();
    acolSav[i]    = event[iIn[i]].acol();
    mSav[i]       = event[iIn[i]].m();
    if (mSav[i] != 0.0) nMassive += 1;
    // Compute and store antenna invariant mass.
    pSum += event[iIn[i]].p();
  }
  m2AntSav     = pSum.m2Calc();
  mAntSav      = (m2AntSav >= 0) ? sqrt(m2AntSav) : -sqrt(-m2AntSav);
  // Massless parents: sIK = m2IK and kallenFac = 1.0.
  sAntSav      = m2AntSav;
  kallenFacSav = 1.0;
  // Mass corrections to sAnt and kallenFac.
  if (nMassive != 0) {
    // sIK = m2IK - m2I - m2K.
    for (unsigned int i = 0; i < iIn.size(); ++i) sAntSav -= pow2(mSav[i]);
    // Phase-space correction non-unity if both parents massive.
    // Note, so far only defined for 2-parton systems.
    if (nMassive == 2 && iIn.size() == 2)
      kallenFacSav = sAntSav/sqrt(pow2(sAntSav) - 4*pow2(mSav[0]*mSav[1]));
  }

}

//--------------------------------------------------------------------------

// Compute pT scale of trial branching.

double Brancher::getpTscale() {

  if (invariantsSav.size() == 3) {
    double sIK = invariantsSav[0];
    double y12 = invariantsSav[1] / sIK;
    double y23 = invariantsSav[2] / sIK;
    return sIK * y12 * y23;
  } else return 0.;

}

//--------------------------------------------------------------------------

// Return Xj.

double Brancher::getXj() {

  if (invariantsSav.size() == 3) {
    double sIK = invariantsSav[0];
    double y12 = invariantsSav[1] / sIK;
    double y23 = invariantsSav[2] / sIK;
    return y12 + y23;
  } else return 1.0;

}

//--------------------------------------------------------------------------

// Simple print utility, showing the contents of the Brancher.

void Brancher::list(string header, bool withLegend) const {

  // Check if we are asked to output a header.
  if (header != "none") {
    cout << " --------  " << std::left << setw(34) << header
         << "  ---------------------------------------------------- \n";
    if (withLegend) {
      cout << "  sys type           mothers                   ID codes    "
           << "colTypes     hels          m    qNewSav \n";
    }
  }
  cout << fixed << std::right << setprecision(3);
  cout << setw(5) << system() << " ";
  // Mothers in iSav list. Baseline 2->3 but also allow for 3->4.
  // (Set up for right-justified printing.)
  int jP0 = -1;
  int jP1 = 0;
  int jP2 = 1;
  if ( iSav.size() == 3 ) {
    jP0 = 0;
    jP1 = 1;
    jP2 = 2;
  }
  string type = "FF";
  // Resonance-Final antennae R and F mothers + list of recoilers.
  if ( posR() >= 0 ) {
    type  = "RF";
    jP0 = -1;
    jP1 = posR();
    jP2 = posF();
  }
  else if ( iSav.size() == 3 ) type = "FFF";
  else if ( iSav.size() >= 4 ) type = "?";
  cout << setw(4) << type;
  cout << " " << setw(5) << (jP0 >= 0 ? num2str(iSav[jP0],5) : " ")
       << " " << setw(5) << iSav[jP1] << " " << setw(5) << iSav[jP2];
  cout << setw(9) << (jP0 >= 0 ? num2str(idSav[jP0],9) : " ")
       << setw(9) << idSav[jP1] << setw(9) << idSav[jP2];
  cout << " " << setw(3) << (jP0 >= 0 ? num2str(colTypeSav[jP0],3) : " ")
       << " " << setw(3) << colTypeSav[jP1]
       << " " << setw(3) << colTypeSav[jP2];
  cout << " " << setw(2) << (jP0 >= 0 ? num2str(hSav[jP0],2) : " ")
       << " " << setw(2) << hSav[jP1] << " " << setw(2) << hSav[jP2];
  cout << " " << num2str(mAnt(), 10);
  if (hasTrial()) {
    if (q2NewSav > 0.) cout << " " << num2str(sqrt(q2NewSav), 10);
    else cout << " " << num2str(0.0, 10);
  }
  else cout << " " << setw(10) << "-";
  cout << endl;

}

//--------------------------------------------------------------------------

// Set post-branching IDs and masses. Base class is for gluon emission.

void Brancher::setidPost() {
  idPostSav.clear();
  idPostSav.push_back(id0());
  idPostSav.push_back(21);
  idPostSav.push_back(id1());
}

vector<double> Brancher::setmPostVec() {
  mPostSav.clear();
  mPostSav.push_back(mSav[0]); // mi
  mPostSav.push_back(0.0);     // mj
  mPostSav.push_back(mSav[1]); // mk
  return mPostSav;
}

void Brancher::setStatPost() {
  statPostSav.resize(iSav.size() + 1, 51);}

void Brancher::setMaps(int) {
  mothers2daughters.clear(); daughters2mothers.clear();}

//--------------------------------------------------------------------------

// Return index of new particle (slightly arbitrary choice for splittings).

int Brancher::iNew() {

  if (i0() > 0) {
    if (mothers2daughters.find(i0()) != mothers2daughters.end())
      return mothers2daughters[i0()].second;
    else return 0;
  }
  else return 0;

}


//==========================================================================

// Class BrancherEmitFF, branch elemental for 2->3 gluon emissions.

//--------------------------------------------------------------------------

// Method to initialise members specific to BrancherEmitFF.

void BrancherEmitFF::initBrancher(ZetaGeneratorSet& zetaGenSet) {

  branchType = BranchType::Emit;
  if (colType0() == 2 && colType1() == 2) antFunTypeSav = GGemitFF;
  else if (colType1() == 2) antFunTypeSav = QGemitFF;
  else if (colType0() == 2) antFunTypeSav = GQemitFF;
  else antFunTypeSav = QQemitFF;

  trialGenPtr = make_shared<TrialGeneratorFF>(sectorShower, branchType,
    zetaGenSet);
}

//--------------------------------------------------------------------------

// Generate a new Q2 value, soft-eikonal 2/yij/yjk implementation.

double BrancherEmitFF::genQ2(int evTypeIn, double q2BegIn, Rndm* rndmPtr,
  Info* infoPtr, const EvolutionWindow* evWindowIn, double colFacIn,
  vector<double> headroomIn, vector<double> enhanceIn,
  int verboseIn) {

  // Set current phase space limits and active sectors.
  double q2MinNow = pow2(evWindowIn->qMin);
  trialGenPtr->reset(q2MinNow,sAntSav,mSav,antFunTypeSav);

  // Initialise output value and save input parameters.
  evTypeSav       = evTypeIn;
  evWindowSav     = evWindowIn;
  colFacSav       = colFacIn;
  q2BegSav        = q2BegIn;
  headroomSav     = (headroomIn.size() >=1) ?  headroomIn[0] : 1.0 ;
  enhanceSav      = (enhanceIn.size() >=1) ?  enhanceIn[0] : 1.0 ;
  double wtNow    = headroomSav * enhanceSav;

  // Generate Q2 and save winning sector.
  q2NewSav = trialGenPtr->genQ2(q2BegSav,rndmPtr,evWindowIn,
    colFacSav,wtNow,infoPtr,verboseIn);
  iSectorWinner   = trialGenPtr->getSector();

  // Sanity checks.
  if (q2NewSav > q2BegIn) {
    string msg = ": Generated q2New > q2BegIn. Returning 0.";
    infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
    q2NewSav = 0.;
  }
  if (q2NewSav > 0.) {
    // Set flag that this call produces a saved trial.
    hasTrialSav = true;
  }
  return q2NewSav;

}

//--------------------------------------------------------------------------

// Generate invariants.

bool BrancherEmitFF::genInvariants(vector<double>& invariants,
  Rndm* rndmPtr, int verboseIn, Info* infoPtr) {

  // Clear output vector, check if we have a sensible q2New scale.
  invariants.clear();
  if (q2NewSav <= 0.) return false;

  // pT evolution.
  if (evTypeSav == 1) {
    //TODO: better overestimate for constant trial alphaS?
    if (!trialGenPtr->genInvariants(sAntSav,setmPostVec(),
        invariantsSav,rndmPtr, infoPtr, verboseIn)) {
      if (verboseIn >= DEBUG) {
        printOut(__METHOD_NAME__,"Trial failed.");
      }
      return false;
    }
    // Veto if the point outside the available phase space.
    double det = gramDet(invariantsSav[1],invariantsSav[2],invariantsSav[3],
      mPostSav[0],mPostSav[1],mPostSav[2]);
    if (det > 0.) {
      invariants = invariantsSav;
      return true;
    }
    else return false;
  }
  else return false;

}

//--------------------------------------------------------------------------

// Compute antPhys / antTrial for gluon emissions, given antPhys.

double BrancherEmitFF::pAccept(const double antPhys, Info* infoPtr,
  int verboseIn) {

  // pT evolution.
  if (evTypeSav == 1) {
    double antTrial = trialGenPtr->aTrial(invariantsSav,mPostSav,
      verboseIn);
    antTrial *= headroomSav;
    if (verboseIn>=DEBUG) {
      if (antTrial==0.) {
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Trial antenna is zero.");
      }
      if (std::isnan(antTrial)) {
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +"Trial antenna not a number");
      }
    }

    return antPhys/antTrial;
  }

  return 0.;

}

//--------------------------------------------------------------------------

// Return the maximum Q2.

double BrancherEmitFF::getQ2Max(int evType) {

  if      (evType == 1) return sAntSav/4.;
  else if (evType == 2) return sAntSav/9.;
  else if (evType == 3) return sAntSav/2.;
  else return 0.;

}

//--------------------------------------------------------------------------

// Method to make mothers2daughters and daughters2mothers pairs.

void BrancherEmitFF::setMaps(int sizeOld) {

  mothers2daughters.clear();
  daughters2mothers.clear();

  //For updating the children of existing parents.
  mothers2daughters[i0()] = make_pair(sizeOld, sizeOld + 1);
  mothers2daughters[i1()] = make_pair(sizeOld + 1, sizeOld + 2);

  //For adding mothers of new children.
  daughters2mothers[sizeOld]   = make_pair(i0(), 0);
  daughters2mothers[sizeOld+1] = make_pair(i0(), i1());
  daughters2mothers[sizeOld+2] = make_pair(i1(), 0);

}

//--------------------------------------------------------------------------

// Generic getter method. Assumes setter methods called earlier.

bool BrancherEmitFF::getNewParticles(Event& event, vector<Vec4> momIn,
  vector<int> hIn, vector<Particle> &pNew, Rndm* rndmPtr, VinciaColour*
  colourPtr) {

  // Initialize.
  unsigned int nPost = iSav.size() + 1;
  pNew.clear();
  pNew.resize(nPost);
  setidPost();
  setStatPost();
  double scaleNew = sqrt(q2NewSav);
  setMaps(event.size());

  // Check everything set.
  if (momIn.size() != nPost || hIn.size() != nPost ||
    mPostSav.size() != nPost || idPostSav.size() != nPost ||
    statPostSav.size() != nPost || invariantsSav.size() < 3) return false;

  // Who inherits the colour?
  double sij  = invariantsSav[1];
  double sjk  = invariantsSav[2];
  bool inh01  = colourPtr->inherit01(sij,sjk);
  int lastTag = event.lastColTag();
  vector<int> col(nPost, 0);
  vector<int> acol(nPost, 0);
  acol[0] = event[i0()].acol();
  col[0]  = event[i0()].col();
  acol[2] = event[i1()].acol();
  col[2]  = event[i1()].col();

  // Generate a new colour tag.
  int colNew = lastTag + 1 + rndmPtr->flat()*10;
  // 0 keeps colour.
  if (inh01) {
    while (colNew%10 == col[2]%10 || colNew%10 == 0)
      colNew = lastTag + 1 + rndmPtr->flat()*10;
    acol[1]=col[0];
    col[1]=colNew;
    acol[2]=colNew;
  // 2 keeps colour.
  } else {
    while (colNew%10 == acol[0]%10 || colNew%10 == 0)
      colNew = lastTag + 1 + rndmPtr->flat()*10;
    col[0]=colNew;
    acol[1]=colNew;
    col[1]=acol[2];
  }

  // Now populate particle vector.
  for (unsigned int ipart = 0; ipart < nPost; ++ipart) {
    pNew[ipart].status(statPostSav[ipart]);
    pNew[ipart].id(idPostSav[ipart]);
    pNew[ipart].pol(hIn[ipart]);
    pNew[ipart].p(momIn[ipart]);
    pNew[ipart].m(mPostSav[ipart]);
    pNew[ipart].setEvtPtr(&event);
    pNew[ipart].scale(scaleNew);
    pNew[ipart].daughters(0,0);
    pNew[ipart].col(col[ipart]);
    pNew[ipart].acol(acol[ipart]);
  }
  colTagSav = colNew;
  return true;

}

//==========================================================================

// Class BrancherSplitFF, branch elemental for 2->3 gluon splittings.

//--------------------------------------------------------------------------

// Method to initialise data members specific to BrancherSplitFF.

void BrancherSplitFF::initBrancher(ZetaGeneratorSet& zetaGenSet,
  bool col2acolIn) {

  branchType    = BranchType::SplitF;
  antFunTypeSav = GXsplitFF;
  isXGsav       = !col2acolIn;
  swapped       = false;

  trialGenPtr   = make_shared<TrialGeneratorFF>(sectorShower, branchType,
    zetaGenSet);
}

//--------------------------------------------------------------------------

// Generate a new Q2 value .

double BrancherSplitFF::genQ2(int evTypeIn, double q2BegIn,
  Rndm* rndmPtr, Info* infoPtr, const EvolutionWindow* evWindowIn,
  double colFac, vector<double> headroomFlav,
  vector<double> enhanceFlav, int verboseIn) {

  // Set current phase space limits and active sectors.
  double q2MinNow = pow2(evWindowIn->qMin);
  trialGenPtr->reset(q2MinNow, sAntSav, mSav, antFunTypeSav);

  // Initialise output value and save input parameters.
  q2NewSav    = 0.;
  evTypeSav   = evTypeIn;
  q2BegSav    = q2BegIn;
  evWindowSav = evWindowIn;

  // Total splitting weight summed over flavours
  double wtSum = 0.0;
  vector<double> wtFlav;
  unsigned int nFlav = headroomFlav.size();
  if (nFlav != enhanceFlav.size()) {
    if (verboseIn >=NORMAL) {
      string msg = ": inconsistent size of headroom and enhancement vectors.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
    }
    return 0.;
  }

  // First check if there is any phase space open for this flavour
  for (unsigned int iFlav = 0; iFlav < nFlav; ++iFlav) {
    double mFlav = evWindowSav->mass.at(iFlav+1);
    if (mAnt() - m0() - m1() < 2.*mFlav) {
      wtFlav.push_back(0.); continue;
    } else {
      double wt = headroomFlav[iFlav] * enhanceFlav[iFlav];
      wtFlav.push_back(wt);
      wtSum += wt;
    }
  }

  // pT evolution.
  if (evTypeSav == 1) {
    // Generate Q2 and save winning sector.
    q2NewSav = trialGenPtr->genQ2(q2BegSav, rndmPtr, evWindowIn,
      colFac, wtSum, infoPtr, verboseIn);
    iSectorWinner = trialGenPtr->getSector();
  }

  // Select flavour.
  double ranFlav = rndmPtr->flat() * wtSum;
  for (int iFlav = nFlav - 1; iFlav >= 0; --iFlav) {
    ranFlav -= wtFlav[iFlav];
    if (ranFlav < 0) {
      idFlavSav   = iFlav+1;
      // Set quark masses.
      mFlavSav    = evWindowSav->mass.at(idFlavSav);
      // Save corresponding headroom and enhancement factors.
      enhanceSav  = enhanceFlav[iFlav];
      headroomSav = headroomFlav[iFlav];
      break;
    }
  }
  if (q2NewSav > q2BegIn) {
    string msg = ": Generated q2New > q2Beg. Returning 0.";
    infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
    q2NewSav = 0.;
  }

  // Sanity checks.
  if (q2NewSav > q2BegIn) {
    string msg = ": Generated q2New > q2BegIn. Returning 0.";
    infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
    q2NewSav = 0.;
  }
  if (q2NewSav > 0.) {
    // Set flag that this call produces a saved trial.
    hasTrialSav = true;
  }

  return q2NewSav;

}

//--------------------------------------------------------------------------

// Generate complementary invariant(s) for saved trial scale
// for gluon splitting. Return false if no physical kinematics possible.

bool BrancherSplitFF::genInvariants(vector<double>& invariants,
  Rndm* rndmPtr, int verboseIn, Info* infoPtr) {

  // Clear output vector, and check if we have a sensible q2New scale.
  invariants.clear();
  if (q2NewSav <= 0.) return false;

  // pT evolution.
  if (evTypeSav == 1) {
    if (!trialGenPtr->genInvariants(sAntSav,setmPostVec(),invariants,rndmPtr,
        infoPtr, verboseIn)) {
      if (verboseIn >= DEBUG) {
        printOut(__METHOD_NAME__,"Trial Failed.");
      }
      return false;
    }

    // Here i=q, j=qbar is always the case, but change def for sjk,
    // sik depending on who is colour connected to the recoiler.
    if (!isXGsav) std::swap(invariants[1],invariants[2]);
    invariantsSav = invariants;

    // Veto if point outside the available phase space.
    double det = gramDet(invariantsSav[0],invariantsSav[1],invariantsSav[2],
      mPostSav[0],mPostSav[1],mPostSav[2]);
    if (det > 0.) return true;
    else return false;
  }
  else return false;

}

//--------------------------------------------------------------------------

// Compute antPhys/antTrial for gluon splittings, given antPhys.
// Note, antPhys should be normalised to include charge and coupling
// factors.

double BrancherSplitFF::pAccept(const double antPhys, Info* infoPtr,
  int verboseIn) {

  // pT evolution.
  if (evTypeSav == 1) {
    double antTrial = trialGenPtr->aTrial(invariantsSav,mPostSav,
      verboseIn);
    antTrial *= headroomSav;
    if (verboseIn>=DEBUG) {
      if (antTrial==0.) {
        infoPtr->errorMsg("Error in "+__METHOD_NAME__,
          "Trial antenna is zero.");
      }
      if (std::isnan(antTrial)) {
        infoPtr->errorMsg("Error in "+__METHOD_NAME__,
          "Trial antenna not a number");
      }
    }

    return antPhys/antTrial;
  }

  return 0.;
}

//--------------------------------------------------------------------------

// Getter and setter methods.

double BrancherSplitFF::getQ2Max(int evType) {

  if      (evType == 1) return sAntSav/4.;
  else if (evType == 2) return sAntSav;
  else if (evType == 3) return sAntSav;
  else return 0.;

}

vector<double> BrancherSplitFF::setmPostVec() {

  mPostSav.clear();
  mPostSav.push_back(mFlavSav); // mi
  mPostSav.push_back(mFlavSav); // mj
  mPostSav.push_back(mSav[1]);  // mk
  return mPostSav;

}

void BrancherSplitFF::setidPost() {

  idPostSav.clear();
  idPostSav.push_back(idFlavSav);
  idPostSav.push_back(-idFlavSav);
  idPostSav.push_back(id1());

}

void BrancherSplitFF::setStatPost() {

  statPostSav.resize(iSav.size() + 1, 51);
  statPostSav[2] = 52;

}

void BrancherSplitFF::setMaps(int sizeOld) {

  // For updating the children of existing parents.
  mothers2daughters.clear();
  daughters2mothers.clear();
  mothers2daughters[i0()] = make_pair(sizeOld, sizeOld+1);
  mothers2daughters[i1()] = make_pair(sizeOld+2,sizeOld+2);

  // For adding mothers of new children.
  daughters2mothers[sizeOld] = make_pair(i0(),0);
  daughters2mothers[sizeOld+1] = make_pair(i0(),0);
  daughters2mothers[sizeOld+2] = make_pair(i1(),i1());

}

//--------------------------------------------------------------------------

// Generic getter method. Assumes setter methods called earlier.

bool BrancherSplitFF::getNewParticles(Event& event, vector<Vec4> momIn,
  vector<int> hIn, vector<Particle> &pNew, Rndm*, VinciaColour*) {

  // Initialize.
  unsigned int nPost = iSav.size() + 1;
  pNew.clear();
  pNew.resize(nPost);
  setidPost();
  setStatPost();
  double scaleNew = sqrt(q2NewSav);
  setMaps(event.size());

  // Check everything set.
  if (momIn.size()!=nPost || hIn.size()!=nPost ||
      mPostSav.size() !=nPost || idPostSav.size() != nPost ||
      statPostSav.size() != nPost || invariantsSav.size() < 3) return false;
  vector<int> col(nPost,0);
  vector<int> acol(nPost,0);
  acol[0] = 0;
  col[0]  = event[i0()].col();
  acol[1] = event[i0()].acol();
  col[1]  = 0;
  acol[2] = event[i1()].acol();
  col[2]  = event[i1()].col();

  // Now populate particle vector.
  for (unsigned int ipart = 0; ipart < nPost; ++ipart) {
    pNew[ipart].status(statPostSav[ipart]);
    pNew[ipart].id(idPostSav[ipart]);
    pNew[ipart].pol(hIn[ipart]);
    pNew[ipart].p(momIn[ipart]);
    pNew[ipart].m(mPostSav[ipart]);
    pNew[ipart].setEvtPtr(&event);
    pNew[ipart].scale(scaleNew);
    pNew[ipart].daughters(0,0);
    pNew[ipart].col(col[ipart]);
    pNew[ipart].acol(acol[ipart]);
  }
  colTagSav = 0;
  return true;

}

//==========================================================================

// BrancherRF base class for storing information on antennae between a
// coloured resonance and final state parton.

//--------------------------------------------------------------------------

// Return index of new particle (slightly arbitrary choice for splittings).

int BrancherRF::iNew() {
  if (posFinal > 0 && iSav[posFinal] > 0
      && mothers2daughters.find(iSav[posFinal]) != mothers2daughters.end())
    return mothers2daughters[iSav[posFinal]].second;
  return 0;
}

//--------------------------------------------------------------------------

// Method to make mothers2daughters and daughters2mothers pairs.

void BrancherRF::setMaps(int sizeOld) {
  mothers2daughters.clear();
  daughters2mothers.clear();
  posNewtoOld.clear();

  // For updating the children of existing parents.  Save children of
  // F (treat like 1->2 splitting).
  mothers2daughters[iSav[posFinal]] = make_pair(sizeOld, sizeOld + 1);
  daughters2mothers[sizeOld] = make_pair(iSav[posFinal], 0);
  daughters2mothers[sizeOld+1] = make_pair(iSav[posFinal], 0);

  //Save recoilers and insert the new emission at position 1.
  int iInsert = sizeOld + 2;
  unsigned int posNewEmit = 1;
  for (unsigned int pos = 0; pos < iSav.size(); pos++) {
    if (pos >= posNewEmit) posNewtoOld[pos + 1] = pos;
    else posNewtoOld[pos] = pos;
    if (pos == posRes || pos == posFinal) continue;
    else {
      mothers2daughters[iSav[pos]] = make_pair(iInsert, iInsert);
      daughters2mothers[iInsert] = make_pair(iSav[pos], iSav[pos]);
      iInsert++;
    }
  }
}

//--------------------------------------------------------------------------

// Protected helper methods for internal class use.

double BrancherRF::getsAK(double mA, double mK, double mAK) {
  return mA*mA +mK*mK - mAK*mAK;}

double BrancherRF::calcQ2Max(double mA, double mAK, double mK) {
  double aM2 = (mA-mAK)*(mA-mAK) - mK*mK;
  double bM2 = mAK*(mA-mAK) + mK*mK;
  double cM = mA-mAK;
  return aM2*aM2*mA/(2.0*cM*bM2);
}

//--------------------------------------------------------------------------

// Veto point if outside available phase space.

bool BrancherRF::vetoPhSpPoint(const vector<double>& invariants,
  int verboseIn) {
  if (invariants.size()!=4) {
    return false;
  }
  double saj = invariants[1];
  double sjk = invariants[2];
  double sak = invariants[3];

  // Make copies of masses (just for compactness of notation).
  double mAK = mRecoilers;
  double ma  = mPostSav[0];
  double mj  = mPostSav[1];
  double mk  = mPostSav[2];

  // Common sense: saj, sjk > 0. Not an error for splitters - mass
  // effects can make negative and push outside generated phase space.
  if (saj<0. || sjk<0.) {
    if (verboseIn >= DEBUG) {
      stringstream ss;
      ss << "Negative invariants. saj = " << saj << " sjk = " << sjk;
      printOut(__METHOD_NAME__, ss.str());
    }
    return true;
  }

  // On-shell X condition.
  double invDiff = ma*ma + mj*mj + mk*mk - saj - sak + sjk - mAK*mAK;
  if (invDiff > MILLI) {
    if (verboseIn >= DEBUG)
      printOut(__METHOD_NAME__, "Failed on-shell AK condition.");
    return true;
  }

  // On-shell j,k conditions.
  double Ek = sak/(2.0*ma);
  if (Ek*Ek < mk*mk) {
    if (verboseIn >= DEBUG)
      printOut(__METHOD_NAME__, "Failed on-shell k condition.");
    return true;
  }
  double Ej = saj/(2.0*ma);
  if (Ej*Ej < mj*mj) {
    if (verboseIn >= DEBUG)
      printOut(__METHOD_NAME__, "Failed on-shell j condition.");
    return true;
  }

  // When |cosTheta| < 1.
  double cosTheta = costheta(Ej,Ek,mj,mk,sjk);
  if (abs(cosTheta) > 1.0) {
    if (verboseIn >= DEBUG)
      printOut(__METHOD_NAME__, "Failed cos theta condition.");
    return true;
  }

  // This condition may be sufficient to remove above conditions.
  // TODO use gramdet here
  double det = saj*sjk*sak - saj*saj*mk*mk - sjk*sjk*ma*ma - sak*sak*mj*mj
    + 4.0*ma*ma*mj*mj*mk*mk;
  if (det <= 0.) {
    if (verboseIn >= DEBUG)
      printOut(__METHOD_NAME__, "Gram det < 0 : Outside phase space");
  }
  return false;

}

//==========================================================================

// BrancherEmitRF class for storing information on antennae between a
// coloured resonance and final state parton, and generating a new
// emission.

//--------------------------------------------------------------------------

// Method to initialise data members specific to BrancherEmitRF.

void BrancherEmitRF::initBrancher(Event& event, vector<int> allIn,
  unsigned int posResIn, unsigned int posFIn, double Q2cut,
  ZetaGeneratorSet& zetaGenSet) {

  // Get Pythia indices of res and final.
  posRes      = posResIn;
  posFinal    = posFIn;
  int iRes    = allIn.at(posRes);
  int iFinal  = allIn.at(posFinal);
  // Is colour flow from R to F, or from F to R?
  colFlowRtoF = event[iRes].col() == event[iFinal].col() && event[iRes].col()
    != 0;

  // Check if R and F swapped (explicit way to force reverse colour flow
  // e.g., for second antenna in gluino -> gluon neutralino.)
  if (event[iRes].status() > 0 ) {
    posRes      = posFIn;
    posFinal    = posResIn;
    iRes        = allIn.at(posRes);
    iFinal      = allIn.at(posFinal);
    colFlowRtoF = false;
  }

  // Extract the momenta of the (set of) recoiler(s).
  Vec4 recoilVec(0., 0., 0., 0.);
  for (vector<int>::iterator pos = allIn.begin(); pos != allIn.end(); ++pos) {
    if ((*pos == iRes) || (*pos == iFinal)) continue;
    recoilVec += event[*pos].p();
  }

  // This is not necesssarily p(res). In the case where one particle
  // always recieves the recoil e.g. W in t->bWX it is p_t -p_X.
  Vec4 resVec = recoilVec + event[iFinal].p();

  // Calculate the masses.
  mRes = resVec.mCalc();
  mFinal = event[iFinal].mCalc();
  mRecoilers = recoilVec.mCalc();
  sAK = getsAK(mRes, mFinal, mRecoilers);

  vector<double> massesPre;
  massesPre.push_back(mRes);
  massesPre.push_back(mFinal);
  massesPre.push_back(mRecoilers);

  // Calculate Q2max.
  Q2MaxSav = calcQ2Max(mRes, mRecoilers, mFinal);
  branchType = BranchType::Emit;
  // TODO: swapped should be redundant since save posRes, posFinal.
  // R = Q.
  if (abs(colTypeSav[posRes]) == 1) {
    // F = Q.
    if (abs(colTypeSav[posFinal]) == 1) {
      antFunTypeSav = QQemitRF;
      swapped = false;
    // F = g.
    } else if (colTypeSav[posFinal] == 2) {
      antFunTypeSav = QGemitRF;
      swapped = posRes != 0;
    // Some other final state - don't know what to do with this yet!
    } else {
      antFunTypeSav = NoFun;
      swapped = false;
    }
  // Some other resonance. Don't know what to do with this yet!
  } else {
    antFunTypeSav = NoFun;
    swapped = false;
  }

  // Set up and initialise trial generator.
  trialGenPtr = make_shared<TrialGeneratorRF>(sectorShower, branchType,
      zetaGenSet);
  trialGenPtr->reset(Q2cut,sAK, massesPre, antFunTypeSav);

}

//--------------------------------------------------------------------------

// Setter methods.

vector<double> BrancherEmitRF::setmPostVec() {
  mPostSav.clear();
  mPostSav.push_back(mRes);       // ma
  mPostSav.push_back(0.0);        // mj
  mPostSav.push_back(mFinal);     // mk
  mPostSav.push_back(mRecoilers); // mAK
  return mPostSav;
}

void BrancherEmitRF::setidPost() {
  idPostSav.clear();
  idPostSav = idSav;
  // Insert gluon in second position.
  idPostSav.insert(idPostSav.begin() + 1, 21);
}

void BrancherEmitRF::setStatPost() {
  statPostSav.resize(iSav.size() + 1, 52);
  statPostSav[posFinal] = 51;
  statPostSav[posFinal+1] = 51;
}

//--------------------------------------------------------------------------

// Generic method, assumes setter methods called earlier.

bool BrancherEmitRF::getNewParticles(Event& event, vector<Vec4> momIn,
  vector<int> hIn, vector<Particle> &pNew, Rndm* rndmPtr, VinciaColour*) {

  // Initialize.
  unsigned int nPost = iSav.size() + 1;
  pNew.clear();
  setidPost();
  setStatPost();
  double scaleNew = sqrt(q2NewSav);
  setMaps(event.size());

  // Check everything set.
  if (momIn.size() != nPost || hIn.size() != nPost ||
     idPostSav.size() != nPost || statPostSav.size() != nPost) return false;

  // Generate new colour tag.
  int lastTag = event.lastColTag();
  int resTag = 0;
  int newTag = 0;
  if (colFlowRtoF) resTag = event[iSav[posRes]].col();
  else resTag = event[iSav[posRes]].acol();
  // New tag can't be same colour as neighbour.
  while (newTag%10 == resTag%10 || newTag%10 == 0)
    newTag = lastTag + 1 + rndmPtr->flat()*10;

  // Now populate particle vector.
  for (unsigned int ipart = 0; ipart < nPost; ++ipart) {
    Particle newPart;
    // Set mass and colours (we have repurposed mPost for antenna
    // function mass scales). This is new emission.
    if (posNewtoOld.find(ipart) == posNewtoOld.end()) {
      newPart.m(0.0);
      if (colFlowRtoF) newPart.cols(resTag, newTag);
      else newPart.cols(newTag, resTag);
    // Skip the resonance.
    } else if (posNewtoOld[ipart] == posRes) continue;
    else {
      newPart.m(mSav[posNewtoOld[ipart]]);
      int colNow  = event[iSav[posNewtoOld[ipart]]].col();
      int acolNow = event[iSav[posNewtoOld[ipart]]].acol();
      if (posNewtoOld[ipart] == posFinal) {
        if (colFlowRtoF) colNow = newTag;
        else acolNow = newTag;
      }
      newPart.cols(colNow,acolNow);
    }

    // Set other pre-determined particle properties.
    newPart.status(statPostSav[ipart]);
    newPart.id(idPostSav[ipart]);
    newPart.pol(hIn[ipart]);
    newPart.p(momIn[ipart]);
    newPart.setEvtPtr(&event);
    newPart.scale(scaleNew);
    newPart.daughters(0,0);
    if (abs(newPart.m() - newPart.mCalc()) > MILLI) return false;
    pNew.push_back(newPart);
  }
  colTagSav=newTag;
  return true;

}

//--------------------------------------------------------------------------

// Generate a new Q2 scale.

double BrancherEmitRF::genQ2(int, double Q2MaxNow, Rndm* rndmPtr,
  Info* infoPtr, const EvolutionWindow* evWindowPtrIn, double colFac,
  vector<double> headroomIn, vector<double> enhanceIn,
  int verboseIn) {

  // Save headroom and enhancement factors.
  headroomSav = (headroomIn.size() >= 1) ?  headroomIn[0] : 1.0;
  enhanceSav  = (enhanceIn.size() >= 1) ?  enhanceIn[0] : 1.0;

  double wtNow = headroomSav * enhanceSav;

  q2NewSav = trialGenPtr->genQ2(Q2MaxNow,rndmPtr,evWindowPtrIn,
    colFac,wtNow,infoPtr,verboseIn);

  iSectorWinner = trialGenPtr->getSector();

  // Sanity checks.
  if (q2NewSav > Q2MaxNow) {
    string msg = ": Generated q2New > q2BegIn. Returning 0.";
    infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
    q2NewSav = 0.;
  }
  if (q2NewSav > 0.) {
    // Set flag that this call produces a saved trial.
    hasTrialSav = true;
  }

  return q2NewSav;

}

//--------------------------------------------------------------------------

// Generate complementary invariant(s) for saved trial scale. Return
// false if no physical kinematics possible.

bool BrancherEmitRF::genInvariants(vector<double>& invariants,Rndm* rndmPtr,
  int verboseIn, Info* infoPtr) {

  // Initialise and check we have a generated q2.
  invariants.clear();

  if (q2NewSav <= 0.) return false;

  if (!trialGenPtr->genInvariants(sAK,setmPostVec(),invariantsSav,rndmPtr,
      infoPtr, verboseIn)) {
    if (verboseIn >= DEBUG) {
      printOut(__METHOD_NAME__,"Trial failed.");
    }
    return false;
  }

  // Veto if the point is outside the available phase space.
  if (vetoPhSpPoint(invariantsSav, verboseIn)) {
    if (verboseIn >= DEBUG) {
      printOut(__METHOD_NAME__,"Outside phase space.");
    }
    return false;
  }
  else {invariants = invariantsSav; return true;}

}

//--------------------------------------------------------------------------

// Compute antPhys/antTrial, given antPhys.

double BrancherEmitRF::pAccept(const double antPhys, Info* infoPtr,
  int verboseIn) {

  double antTrial = trialGenPtr->aTrial(invariantsSav,mPostSav,
    verboseIn);
  antTrial *= headroomSav;
  if (verboseIn>=DEBUG) {
    if (antTrial==0.) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,
        "Trial antenna is zero.");
    }
    if (std::isnan(antTrial)) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,
        "Trial antenna not a number");
    }
  }

  return antPhys/antTrial;

}

//==========================================================================

// BrancherSplitRF class for storing information on antennae between a
// coloured resonance and final state parton, and generating a new
// emission.

//--------------------------------------------------------------------------

void BrancherSplitRF::initBrancher(Event& event, vector<int> allIn,
  unsigned int posResIn, unsigned int posFIn, double Q2cut,
  ZetaGeneratorSet& zetaGenSet) {

  // Get Pythia indices of res and final.
  posRes     = posResIn;
  posFinal   = posFIn;
  int iRes   = allIn.at(posRes);
  int iFinal = allIn.at(posFinal);
  colFlowRtoF = event[iRes].col() == event[iFinal].col()
    && event[iRes].col() != 0;

  // Extract the momenta of the rest.
  Vec4 recoilVec(0., 0., 0., 0.);
  for (vector<int>::iterator pos=allIn.begin(); pos!=allIn.end(); ++pos) {
    if ((*pos == iRes) || (*pos == iFinal)) continue;
    recoilVec += event[*pos].p();
  }

  // This is not necesssarily p(res). In the case where one particle
  // always recieves the recoil, e.g. W in t->bWX it is p_t - p_X,
  Vec4 resVec = recoilVec + event[iFinal].p();
  mRes = resVec.mCalc();
  mFinal = 0.;
  mRecoilers = recoilVec.mCalc();
  sAK = getsAK(mRes, mFinal, mRecoilers);

  vector<double> massesPre;
  massesPre.push_back(mRes);
  massesPre.push_back(mFinal);
  massesPre.push_back(mRecoilers);

  // Calculate Q2max.
  Q2MaxSav = calcQ2Max(mRes, mRecoilers,mFinal);
  branchType = BranchType::SplitF;
  swapped = false;
  antFunTypeSav = XGsplitRF;

  // Set up and initialise trial generator.
  trialGenPtr = make_shared<TrialGeneratorRF>(sectorShower,
    branchType, zetaGenSet);
  trialGenPtr->reset(Q2cut,sAK, massesPre, antFunTypeSav);
}

//--------------------------------------------------------------------------

// Setter methods.

vector<double> BrancherSplitRF::setmPostVec() {
  mPostSav.clear();
  mPostSav.push_back(mRes);
  mPostSav.push_back(mFlavSav);
  mPostSav.push_back(mFlavSav);
  mPostSav.push_back(mRecoilers);
  return mPostSav;
}

void BrancherSplitRF::setidPost() {
  idPostSav.clear();
  idPostSav = idSav;
  // Modify the splitting gluon to antiquark, insert quark in second position.
  if (colFlowRtoF) {
    idPostSav[posFinal] = -idFlavSav;
    idPostSav.insert(idPostSav.begin() + 1, idFlavSav);
  } else {
    idPostSav[posFinal] = idFlavSav;
    idPostSav.insert(idPostSav.begin() + 1, -idFlavSav);
  }
}

void BrancherSplitRF::setStatPost() {
  statPostSav.resize(iSav.size() + 1, 52);
  statPostSav[1] = 51;
  statPostSav[posFinal+1] = 51;
}

//--------------------------------------------------------------------------

// Generic method, assumes setter methods called earlier.

bool BrancherSplitRF::getNewParticles(Event& event, vector<Vec4> momIn,
  vector<int> hIn, vector<Particle>& pNew, Rndm*, VinciaColour*) {

  // Initialize.
  unsigned int nPost = iSav.size() + 1;
  pNew.clear();
  setidPost();
  setStatPost();
  double scaleNew = sqrt(q2NewSav);
  setMaps(event.size());
  if (momIn.size() != nPost || hIn.size() != nPost ||
      idPostSav.size() != nPost || statPostSav.size() != nPost ) return false;
  int resTag = 0;
  if (colFlowRtoF) resTag = event[iSav[posRes]].col();
  else resTag = event[iSav[posRes]].acol();

  // Now populate particle vector.
  for (unsigned int ipart = 0; ipart < nPost; ++ipart) {
    Particle newPart;
    // Set mass and colours, (we have repurposed mPost for antenna
    // function mass scales). This is new emission.
    if (posNewtoOld.find(ipart) == posNewtoOld.end()) {
      newPart.m(mFlavSav);
      if (colFlowRtoF) newPart.cols(resTag, 0);
      else newPart.cols(0, resTag);
    } else if (posNewtoOld[ipart] == posRes) {continue;
    } else {
      int colNow  = event[iSav[posNewtoOld[ipart]]].col();
      int acolNow = event[iSav[posNewtoOld[ipart]]].acol();
      if (posNewtoOld[ipart] == posFinal) {
        if (colFlowRtoF) colNow = 0;
        else acolNow = 0;
        newPart.m(mFlavSav);
      } else newPart.m(mSav[posNewtoOld[ipart]]);
      newPart.cols(colNow,acolNow);
    }

    //Set other pre-determined particle properties.
    newPart.status(statPostSav[ipart]);
    newPart.id(idPostSav[ipart]);
    newPart.pol(hIn[ipart]);
    newPart.p(momIn[ipart]);
    newPart.setEvtPtr(&event);
    newPart.scale(scaleNew);
    newPart.daughters(0,0);
    pNew.push_back(newPart);
  }
  colTagSav = 0;
  return true;

}

//--------------------------------------------------------------------------

// Generate a new Q2 scale.

double BrancherSplitRF::genQ2(int, double Q2MaxNow, Rndm* rndmPtr,
  Info* infoPtr, const EvolutionWindow* evWindowPtrIn, double colFac,
  vector<double> headroomIn, vector<double> enhanceIn, int verboseIn) {

  // Total splitting weight summed over flavours
  double wtSum = 0.0;
  vector<double> wtFlav;
  unsigned int nFlav = headroomIn.size();
  if (nFlav != enhanceIn.size()) {
    if (verboseIn >= NORMAL) {
      string msg = ": Headroom and enhancement vectors have different sizes.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    }
    return 0.;
  }
  for (unsigned int iFlav = 0; iFlav < nFlav; ++iFlav) {
    double wt = headroomIn[iFlav] * enhanceIn[iFlav];
    wtFlav.push_back(wt);
    wtSum += wt;
  }

  q2NewSav = trialGenPtr->genQ2(Q2MaxNow,rndmPtr,evWindowPtrIn,
    colFac,wtSum,infoPtr,verboseIn);

  // Sanity check.
  if (q2NewSav > Q2MaxNow) {
    string msg = ": Generated q2New > q2BegIn. Returning 0.";
    infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
    q2NewSav = 0.;
  }

  if (q2NewSav > 0.) {

    // Select flavour.
    double ranFlav = rndmPtr->flat() * wtSum;
    for (int iFlav = nFlav - 1; iFlav >= 0; --iFlav) {
      ranFlav -= wtFlav[iFlav];
      if (ranFlav < 0) {
        idFlavSav   = iFlav+1;
        mFlavSav    = evWindowPtrIn->mass.at(idFlavSav);
        enhanceSav  = enhanceIn[iFlav];
        headroomSav = headroomIn[iFlav];
        break;
      }
    }

    // Debugging.
    if (verboseIn >= DEBUG) {
      stringstream ss;
      ss << "Selected splitting flavour: " << idFlavSav;
      printOut(__METHOD_NAME__, ss.str());
    }
    if (q2NewSav > Q2MaxNow) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,
        "Generated impossible Q2");
      q2NewSav = -1.;
    }
    hasTrialSav = true;

  }
  return q2NewSav;

}

//==========================================================================

// The VinciaFSR class for resonant decays.

//--------------------------------------------------------------------------

// Initialize alphaStrong and related pTmin parameters (TimeShower).

void VinciaFSR::init( BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn) {
  // Check if already initialized.
  if (isInit)
    return;
  verbose = settingsPtr->mode("Vincia:verbose");
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin --------------");
  allowforceQuit = false;
  forceQuit = false;
  nBranchQuit = -1;

  // Showers on/off.
  bool doFSR = settingsPtr->flag("PartonLevel:FSR");
  bool doISR = settingsPtr->flag("PartonLevel:ISR");
  doFF       = doFSR && settingsPtr->flag("Vincia:doFF");
  doRF       = doFSR && settingsPtr->flag("Vincia:doRF");
  doII       = doISR && settingsPtr->flag("Vincia:doII");
  doIF       = doISR && settingsPtr->flag("Vincia:doIF");
  ewMode     = settingsPtr->mode("Vincia:EWmode");
  ewModeMPI  = min(settingsPtr->mode("Vincia:EWmodeMPI"),ewMode);
  doQED      = ewMode >= 1;
  doWeak     = ewMode >= 3;

  // TODO: everything is evolved in PT in this version of VINCIA.
  evTypeEmit     = 1;
  evTypeSplit    = 1;

  // Parameters of interleaved resonance decays.
  doInterleaveResDec = settingsPtr->flag("Vincia:interleaveResDec");
  resDecScaleChoice  = settingsPtr->mode("Vincia:resDecScalechoice");
  doFSRinResonances  = settingsPtr->flag("PartonLevel:FSRinResonances");

  // Store input pointers for future use.
  beamAPtr     = beamAPtrIn;
  beamBPtr     = beamBPtrIn;

  // Assume all events in same run have same beam-beam ECM.
  m2BeamsSav  = m2(beamAPtr->p(), beamBPtr->p());
  eCMBeamsSav = sqrt(m2BeamsSav);

  // Possibility to allow user veto of emission step.
  hasUserHooks       = (userHooksPtr != 0);
  canVetoEmission    = (hasUserHooks && userHooksPtr->canVetoFSREmission());
  // canVetoISREmission is part of the overlap veto to avoid filling the same
  // phase space with EW and QCD radiation starting from different hard
  // processes. If there is no weak shower, there is also no overlap and thus
  // no veto needed.
  canVetoISREmission = (hasUserHooks && doWeak &&
    userHooksPtr->canVetoISREmission());

  // Number of active quark flavours
  nGluonToQuark = settingsPtr->mode("Vincia:nGluonToQuark");

  // Number of flavours to be treated as massless (can be made
  // user-specifiable in future if desired).
  nFlavZeroMass = settingsPtr->mode("Vincia:nFlavZeroMass");

  // Global flag for helicity dependence.
  helicityShower = settingsPtr->flag("Vincia:helicityShower");
  if (doWeak && !helicityShower) {
    infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": switching "
      "on helicityShower (required for ewMode = 3).");
    settingsPtr->flag("Vincia:helicityShower", true);
    helicityShower = true;
    isrPtr->helicityShower = true;
  }

  // Global flag for sector showers on/off.
  sectorShower = settingsPtr->flag("Vincia:sectorShower");

  // Merging flags.
  doMerging        = settingsPtr->flag("Merging:doMerging");
  isTrialShower    = false;
  isTrialShowerRes = false;

  // Perturbative cutoff. Since emissions and splittings can have
  // different evolution measures, in principle allow for different
  // cutoff scales, for now forced same.
  q2CutoffEmit  = pow2(settingsPtr->parm("Vincia:cutoffScaleFF"));
  // Allow perturbative g->qq splittings to lower scales.
  q2CutoffSplit = pow2(settingsPtr->parm("Vincia:cutoffScaleFF"));

  // Set shower alphaS pointer.
  useCMW     = settingsPtr->flag("Vincia:useCMW");
  aSemitPtr  = &vinComPtr->alphaStrong;
  aSsplitPtr = &vinComPtr->alphaStrong;
  // Currently, CMW is applied to both emissions and splittings.
  if (useCMW) {
    aSemitPtr  = &vinComPtr->alphaStrongCMW;
    aSsplitPtr = &vinComPtr->alphaStrongCMW;
  }

  // AlphaS parameters.
  alphaSvalue    = settingsPtr->parm("Vincia:alphaSvalue");
  alphaSorder    = settingsPtr->mode("Vincia:alphaSorder");
  aSkMu2Emit     = settingsPtr->parm("Vincia:renormMultFacEmitF");
  aSkMu2Split    = settingsPtr->parm("Vincia:renormMultFacSplitF");
  alphaSmax      = settingsPtr->parm("Vincia:alphaSmax");
  alphaSmuFreeze = settingsPtr->parm("Vincia:alphaSmuFreeze");
  mu2freeze      = pow2(alphaSmuFreeze);

  // Smallest allowed scale for running alphaS.
  alphaSmuMin = 1.05 * max(aSemitPtr->Lambda3(), aSsplitPtr->Lambda3());
  mu2min      = pow2(alphaSmuMin);

  // For constant alphaS, set max = value (for efficiency).
  if (alphaSorder == 0) alphaSmax = alphaSvalue;
  initEvolutionWindows();

  // Settings for enhanced (biased) kernels.
  enhanceInHard   = settingsPtr->flag("Vincia:enhanceInHardProcess");
  enhanceInResDec = settingsPtr->flag("Vincia:enhanceInResonanceDecays");
  enhanceInMPI    = settingsPtr->flag("Vincia:enhanceInMPIshowers");
  enhanceAll      = settingsPtr->parm("Vincia:enhanceFacAll");
  // Explicitly allow heavy-quark enhancements only, not suppression
  enhanceBottom   = max(1., settingsPtr->parm("Vincia:enhanceFacBottom"));
  enhanceCharm    = max(1., settingsPtr->parm("Vincia:enhanceFacCharm"));
  enhanceCutoff   = settingsPtr->parm("Vincia:enhanceCutoff");

  // Resize pAccept to the maximum number of elements.
  pAccept.resize(max(weightsPtr->getWeightsSize(), 1));

  // Initialize parameters for shower starting scale.
  pTmaxMatch     = settingsPtr->mode("Vincia:pTmaxMatch");
  pTmaxFudge     = settingsPtr->parm("Vincia:pTmaxFudge");
  pT2maxFudge    = pow2(pTmaxFudge);
  pT2maxFudgeMPI = pow2(settingsPtr->parm("Vincia:pTmaxFudgeMPI"));

  // Initialize the FSR antenna functions.
  if (verbose >= REPORT)
    printOut(__METHOD_NAME__, "initializing antenna set");
  antSetPtr->init();
  kMapResEmit  = settingsPtr->mode("Vincia:kineMapRFemit");
  kMapResSplit = settingsPtr->mode("Vincia:kineMapRFsplit");

  // All OK.
  isInit=true;
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);

  if (verbose >= QUIET)  header();
}

//--------------------------------------------------------------------------

// Possible limitation of first emission (TimeShower).

bool VinciaFSR::limitPTmax(Event& event, double, double) {

  // Check if limiting pT of first emission.
  if (pTmaxMatch == 1) return true;
  else if (pTmaxMatch == 2) return false;

  // Always restrict SoftQCD processes.
  else if (infoPtr->isNonDiffractive() || infoPtr->isDiffractiveA() ||
           infoPtr->isDiffractiveB() || infoPtr->isDiffractiveC())
    return true;

  // Look if jets or photons in final state of hard system (iSys = 0).
  else {
    const int iSysHard = 0;
    for (int i = 0; i < partonSystemsPtr->sizeOut(iSysHard); ++i) {
      int idAbs = event[partonSystemsPtr->getOut(iSysHard,i)].idAbs();
      if (idAbs <= 5 || idAbs == 21 || idAbs == 22) return true;
      else if (idAbs == 6 && nGluonToQuark == 6) return true;
    }
    // If no QCD/QED partons detected, allow to go to phase-space maximum
    return false;
  }

}

//--------------------------------------------------------------------------

// Top-level routine to do a full time-like shower in resonance decay
// (TimeShower).

int VinciaFSR::shower(int iBeg, int iEnd, Event& event, double pTmax,
  int nBranchMax) {

  // Verbose output.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin" , dashLen);

  // Add new system, automatically with two empty beam slots.
  int iSys = partonSystemsPtr->addSys();

  // Verbose output.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,
      "preparing to shower. System no. " + num2str(iSys));

  // Loop over allowed range to find all final-state particles.
  Vec4 pSum;
  for (int i = iBeg; i <= iEnd; ++i) {
    if (event[i].isFinal()) {
      partonSystemsPtr->addOut( iSys, i);
      pSum += event[i].p();
    }
  }
  partonSystemsPtr->setSHat( iSys, pSum.m2Calc() );

  // Let prepare routine do the setup.
  // (isPrepared = false forces clearing of any previous information.)
  isPrepared = false;
  prepare(iSys, event, false);

  // Begin evolution down in pT from hard pT scale.
  int nBranchNow = 0;
  do {
    // Do a final-state emission (if allowed).
    double pTtimes = pTnext(event, pTmax, 0.);
    if (pTtimes > 0.) {
      if (branch(event)) ++nBranchNow;
      pTmax = pTtimes;
    }

    // Keep on evolving until nothing is left to be done.
    else pTmax = 0.;
  } while (pTmax > 0. && (nBranchMax <= 0 || nBranchNow < nBranchMax));

  // Return number of emissions that were performed.
  return nBranchNow;

}

//--------------------------------------------------------------------------

// Method to add QED showers in hadron decays (TimeShower).

int VinciaFSR::showerQED(int iBeg, int iEnd, Event& event, double pTmax) {

  // Check if we are supposed to do anything.
  if (!doQED || infoPtr->getAbortPartonLevel()) return 0;
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "begin", dashLen);
    event.list();
  }
  // Construct a little QED system out of the given particles.
  partonSystemsPtr->addSys();
  int iSys = partonSystemsPtr->sizeSys()-1;
  // We could check if they all have the same mother and treat as
  // resonance decay, but currently do not.
  if (iBeg > iEnd) {
    partonSystemsPtr->addOut(iSys,iBeg);
    partonSystemsPtr->addOut(iSys,iEnd);
  } else {
    for (int i=iBeg; i<iEnd; ++i) partonSystemsPtr->addOut(iSys,i);
  }
  qedShowerSoftPtr->clear();
  qedShowerSoftPtr->prepare( iSys, event, true);
  double q2      = pow2(pTmax);
  double q2min   = qedShowerSoftPtr->q2min();
  int nBranchNow = 0;
  while (q2 > q2min) {
    q2 = qedShowerSoftPtr->q2Next(event, q2, q2min);
    if (q2 < q2min) break;
    if (qedShowerSoftPtr->acceptTrial(event)) {
      // After branching accepted, update event, partonSystems, and antennae.
      qedShowerSoftPtr->updateEvent(event);
      qedShowerSoftPtr->updatePartonSystems(event);
      qedShowerSoftPtr->update(event, iSys);
      ++nBranchNow;
    }
  }
  return nBranchNow;

}

//--------------------------------------------------------------------------

// Method to add QED showers to partons below colour resolution scale
// (TimeShower).

int VinciaFSR::showerQEDafterRemnants(Event& event) {
  // Check if we are supposed to do anything.
  if (!doQED || infoPtr->getAbortPartonLevel()) return 0;
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "begin", dashLen);
    event.list();
  }

  // Prepare for showering below hadronisation scale. Include partons
  // from all current systems (pass iSys = -1).
  qedShowerSoftPtr->clear();
  qedShowerSoftPtr->prepare( -1, event, true);

  // Retrieve iSys for remnant system.
  int iSysRem    = partonSystemsPtr->sizeSys()-1;
  double q2      = qedShowerSoftPtr->q2minColoured();
  double q2min   = max(qedShowerSoftPtr->q2min(),PICO);

  int nBranchNow = 0;
  if (partonSystemsPtr->sizeOut(iSysRem) >= 2) {

    double nLoop = 0;
    while (q2 > q2min) {
      if (++nLoop >= 1000) {
        infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": "
          "stuck in infinite loop");
        break;
      }
      q2 = qedShowerSoftPtr->q2Next(event, q2, q2min);
      if (q2 <= q2min) break;
      if (qedShowerSoftPtr->acceptTrial(event)) {
        // After branching accepted, update event, partonSystems, and antennae.
        qedShowerSoftPtr->updateEvent(event);
        qedShowerSoftPtr->updatePartonSystems(event);
        qedShowerSoftPtr->update(event, iSysRem);
        ++nBranchNow;
      }
    }
    // Move post-remnant-shower partons back into their respective systems.
    if (verbose >= DEBUG) {
      stringstream ss; ss<<" showered remnant iSysRem = "<<iSysRem;
      printOut(__METHOD_NAME__,ss.str());
      event.list();
      partonSystemsPtr->list();
    }
    for (int iSys = 0; iSys < iSysRem; ++iSys) {
      int sizeOut = partonSystemsPtr->sizeOut(iSys);
      for (int iOut = sizeOut - 1; iOut >= 0; --iOut) {
        int i = partonSystemsPtr->getOut(iSys, iOut);
        // If this parton is still present, keep it and skip to next.
        if (event[i].isFinal()) continue;
        int iBot = event[i].iBotCopyId();
        // Replaced disappeared particle by iBot if latter is final.
        // Else overwrite with the one at the back, and then pop back.
        // (Example of latter: photon that has branched to fermion pair.)
        if (event[iBot].isFinal()) {
          partonSystemsPtr->replace(iSys,i,iBot);
        } else {
          int sizeOutNow = partonSystemsPtr->sizeOut(iSys);
          int iCopy = partonSystemsPtr->getOut(iSys,sizeOutNow-1);
          partonSystemsPtr->setOut(iSys,iOut,iCopy);
          partonSystemsPtr->popBackOut(iSys);
        }
      }
    }
    // Finally only keep partons in iSysRem that are not in any other system.
    int sizeOut = partonSystemsPtr->sizeOut(iSysRem);
    for (int iOut = sizeOut - 1; iOut >= 0; --iOut) {
      int i = partonSystemsPtr->getOut(iSysRem, iOut);
      int iSysNow = partonSystemsPtr->getSystemOf(i);
      if (iSysNow != iSysRem) {
        int sizeOutNow = partonSystemsPtr->sizeOut(iSysRem);
        int iCopy = partonSystemsPtr->getOut(iSysRem,sizeOutNow-1);
        partonSystemsPtr->setOut(iSysRem,iOut,iCopy);
        partonSystemsPtr->popBackOut(iSysRem);
      }
    }
    // If QED shower did not do anything, the updated post-remnant system
    // will only have remnant partons in it. Remove it if no partons left.
    if (partonSystemsPtr->sizeOut(iSysRem) == 0) partonSystemsPtr->popBack();
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__,"updated parton systems");
      partonSystemsPtr->list();
    }
  }

  // Force decays of any left-over resonances from the weak shower.
  if (doWeak) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,
      "Checking for leftover resonances");
    ewShowerPtr->clear();
    ewShowerPtr->prepare(0, event, true);
    if (ewShowerPtr->nResDec() > 0) {
      q2 = 1e6;
      while (q2 > 0.) {
        q2 = ewShowerPtr->q2Next(event, q2, 0.);
        if (q2 <= 0.) break;
        q2WinSav  = q2;
        winnerEW  = ewShowerPtr;
        winnerQCD = nullptr;
        // branch() automatically adds resonance shower.
        if (branch(event)) ++nBranchNow;
      }
    }
  }

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return nBranchNow;
}

//--------------------------------------------------------------------------

// Prepare process-level event for showering + interleaved resonance decays.
// Usage: prepareProcess( process, event, iPos).
// iPos provides mapping from process to event entries (before showering).

void VinciaFSR::prepareProcess( Event& process, Event& event,
  vector<int>& iBefShower) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"begin",dashLen);

  // Signal that prepare() should treat this as a new event.
  isPrepared = false;
  isrPtr->isPrepared = false;
  iPosBefSav = iBefShower;

  // Initialise recursion-depth counter (for nested sequential decays),
  // and vector of resonance-decay pT scales.
  nRecurseResDec = 0;
  pTresDecSav.clear();
  idResDecSav.clear();

  // Find resonances in process-level event record.
  vector<int> resMothers;
  for (int iHard=0; iHard<process.size(); ++iHard) {
    double pTresDec = 0.;
    int    idResDec = 0;
    int iHardMot = process[iHard].mother1();
    if (process[iHard].status() == -22) {
      resMothers.push_back(iHard);
      // Compute pT scales for resonances with decays included in the hard
      // process. (Sequential decays irrelevant until mothers have decayed.)
      if (doInterleaveResDec && !process[iHardMot].isResonance()) {
        // Set interleaving scale: width or offshellness
        pTresDec = calcPTresDec(process[iHard]);
        idResDec = process[iHard].id();
      }
    }
    pTresDecSav.push_back(pTresDec);
    idResDecSav.push_back(idResDec);
  }

  // EW and Helicity Showers require helicity selection.
  if ( doWeak || helicityShower) {

    // Define hard-scattering state (with and then without prompt res decays).
    vector<Particle> state;
    vector<int> iProcess;
    bool needsPol = false;
    // Save Incoming legs first.
    for (int iHard=0; iHard<process.size(); ++iHard) {
      // Add incoming partons to state
      if (process[iHard].status() == -21) {
        state.push_back(process[iHard]);
        iProcess.push_back(iHard);
      }
      // Check if there are any unpolarised partons in this state.
      // Check if any partons in this state need helicities.
      int id     = process[iHard].id();
      int nSpins = particleDataPtr->spinType(abs(id));
      if (nSpins != 0 && process[iHard].pol() == 9) needsPol = true;
    }
    // If no helicities need to be selected, we are done.
    if (!needsPol) return;

    // If doing interleaved resonance decays, first try with
    // prompt decays included, then allow to try without.
    int sizeIn = state.size();
    for (bool interleave : { doInterleaveResDec, false }) {
      // Reset state to incoming only, then add final ones.
      state.resize(sizeIn);
      iProcess.resize(sizeIn);
      // Loop over 2->N Born process.
      int iBeg = state[0].daughter1();
      int iEnd = state[0].daughter2();
      if (iEnd == 0) iEnd = iBeg;
      double scale = process.scale();
      for (int iHard=iBeg; iHard <= iEnd; ++iHard) {
        // If doing interleaving, resonances with decay scales > scale/2
        // are treated as prompt (no showering). Include daughters instead.
        if (interleave && process[iHard].status() == -22 &&
          pTresDecSav[iHard] > scale/2) {
          for (int iDa = process[iHard].daughter1();
               iDa <= process[iHard].daughter2(); ++iDa) {
            state.push_back(process[iDa]);
            iProcess.push_back(iDa);
            // Force outgoing partons in state to have code 23.
            state.back().status(23);
          }
        }
        // Stable final-state particle, or resonance treated as stable.
        else if (process[iHard].isFinal() || process[iHard].status() == -22) {
          state.push_back(process[iHard]);
          iProcess.push_back(iHard);
          // Force outgoing partons in state to have code 23.
          state.back().status(23);
        }
      }

      // Check if we have the relevant Hard-Process Matrix Element.
      if (mecsPtr->meAvailable(state)) {
        if (mecsPtr->polarise(state,true)) {
          // If we pass sanity check,
          if (state.size() == iProcess.size()) {
            needsPol = false;
            break;
          }
          // Else print warning that something unexpected happened.
          else infoPtr->errorMsg("Error in "+__METHOD_NAME__+": "
            "inconsistent state size after helicity selection");
        }
      }
    }

    // If state now contains polarisations, copy to process and event.
    if (!needsPol && state.size() > 0) {
      for (int i=0; i<(int)state.size(); ++i) {
        // Consistency checks.
        int iHard = iProcess[i];
        if (state[i].id() != process[iHard].id()) {
          infoPtr->errorMsg("Error in "+__METHOD_NAME__+": "
            "state does not match process after helicity selection");
          if (verbose >= DEBUG) process.list();
          return;
        }
        // Copy pol() to process.
        process[iHard].pol(state[i].pol());
        // Copy pol() to main event record.
        if (iHard < (int)iBefShower.size()
          && iBefShower[iHard] != 0) {
          int iEvent = iBefShower[iHard];
          if (state[i].id() != event[iEvent].id()) {
            infoPtr->errorMsg("Error in "+__METHOD_NAME__+": "
              "state does not match event after helicity selection");
            if (verbose >= DEBUG) process.list();
            return;
          }
          event[iEvent].pol(state[i].pol());
        }
      }
    } else {
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": "
        "failed to assign helicities for hard process");
      if (verbose >= DEBUG) process.list();
      return;
    }

    // Verbose output. Print new process-level and event-level event records
    // with helicity assignments.
    if (verbose >= DEBUG) {
      process.list(true);
      event.list(true);
    }
  }

  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"end",dashLen);

  // Finished.
  return;

}

//--------------------------------------------------------------------------

// Prepare system for evolution (TimeShower).

void VinciaFSR::prepare(int iSys, Event& event, bool) {

  if (!isInit) return;

  // Check if we are supposed to do anything
  if (!(doFF || doRF)) return;
  if (infoPtr->getAbortPartonLevel()) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Received abort from PartonLevel().","Aborting.");
    return;
  }

  // Last chance to print header if not done already.
  if (!headerIsPrinted && verbose >= NORMAL) header();

  // Resetting for each new event (or non-interleaved decay of a resonance
  // or hadron) to be showered.
  bool hasInRes = partonSystemsPtr->hasInRes(iSys);
  bool hasInAB  = partonSystemsPtr->hasInAB(iSys);
  // Print event and antenna list before cleanup.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "begin (iSys " + num2str(iSys) + ", isPrepared "
      + bool2str(isPrepared) + ", hasInAB " + bool2str(hasInAB) + ", hasInRes "
      + bool2str(hasInRes) + ")", dashLen);
    event.list();
    partonSystemsPtr->list();
  }

  if (!isPrepared || (!doInterleaveResDec && hasInRes) ||
    (!hasInAB && !hasInRes)) {

    // Do the following only once per new event to shower.
    if (!isPrepared) {
      forceQuit = false;

      // Reset counters in new events.
      vinComPtr->resetCounters();
      clearContainers();
      nRecurseResDec = 0;
      nBranch.clear();
      nBranchFSR.clear();

      // Set hard-system handler to EW or QED showers.
      ewHandlerHard = (doWeak) ? ewShowerPtr : qedShowerHardPtr;
    }

    // Do this for each new stage of showering (main, non-interleaved
    // resonance decays, hadron decays).
    emittersRF.clear();
    splittersRF.clear();
    emittersFF.clear();
    splittersFF.clear();
    lookupEmitterRF.clear();
    lookupSplitterRF.clear();
    lookupEmitterFF.clear();
    lookupSplitterFF.clear();

    if (doWeak) ewShowerPtr->clear();
    qedShowerHardPtr->clear();
    qedShowerSoftPtr->clear();

  } else {

    // Verbose output.
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__, "QCD antenna list before cleanup:", dashLen);
      list();
    }

    // Make sure any existing branchers in this system are cleared out.
    int nSys = partonSystemsPtr->sizeSys();
    for (int i=emittersFF.size()-1; i >= 0; --i) {
      if (emittersFF[i].system() == iSys || emittersFF[i].system() >= nSys)
        emittersFF.erase(emittersFF.begin()+i);
    }
    for (int i=emittersRF.size()-1; i >= 0; --i)
      if (emittersRF[i].system() == iSys || emittersRF[i].system() >= nSys)
        emittersRF.erase(emittersRF.begin()+i);
    for (int i=splittersFF.size()-1; i >= 0; --i)
      if (splittersFF[i].system() == iSys || splittersFF[i].system() >= nSys)
        splittersFF.erase(splittersFF.begin()+i);
    for (int i=splittersRF.size()-1; i >= 0; --i)
      if (splittersRF[i].system() == iSys || splittersRF[i].system() >= nSys)
        splittersRF.erase(splittersRF.begin()+i);

    // Reconstruct lookup tables for existing branchers.
    lookupEmitterFF.clear();
    for (unsigned int i=0; i < emittersFF.size(); ++i) {
      // Colour, Anticolour.
      lookupEmitterFF[make_pair(emittersFF[i].i0(),true)] = i;
      lookupEmitterFF[make_pair(emittersFF[i].i1(),false)] = i;
    }
    lookupSplitterFF.clear();
    for (unsigned int i=0; i < splittersFF.size(); ++i) {
      // Gluon(Colour side) + Recoiler.
      if (!splittersFF[i].isXG()) {
        lookupSplitterFF[make_pair(splittersFF[i].i0(),true)] = i;
        lookupSplitterFF[make_pair(splittersFF[i].i1(),false)] = i;
      }
      // Gluon(Anticolour side) + Recoiler.
      else {
        lookupSplitterFF[make_pair(-splittersFF[i].i0(),true)] = i;
        lookupSplitterFF[make_pair(-splittersFF[i].i1(),false)] = i;
      }
    }
    lookupEmitterRF.clear();
    for (unsigned int i=0; i < emittersRF.size(); ++i) {
      bool i0isRes = (emittersRF[i].posR() == 0);
      // Resonance always first. (Negative for reversed colour flow.)
      lookupEmitterRF[make_pair(emittersRF[i].i0(),i0isRes)] = i;
      lookupEmitterRF[make_pair(emittersRF[i].i1(),!i0isRes)] = i;
    }
    lookupSplitterRF.clear();
    for (unsigned int i=0; i < splittersRF.size(); ++i) {
      bool i0isRes = (splittersRF[i].posR() == 0);
      // Resonance always first. (Negative for reversed colour flow.)
      lookupSplitterRF[make_pair(splittersRF[i].i0(),i0isRes)] = i;
      lookupSplitterRF[make_pair(splittersRF[i].i1(),!i0isRes)] = i;
    }

    // Verbose output.
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__, "QCD antenna list after cleanup:", dashLen);
      list();
    }

  }

  // Allow to quit after a certain number of emissions per event (just
  // for testing).
  if (forceQuit) {
    if (verbose >= REPORT) printOut(__METHOD_NAME__,
      "User forced quit early");
    return;
  }

  // Sanity check: at least two particles in system.
  int sizeSystem = partonSystemsPtr->sizeAll(iSys);
  if (sizeSystem <= 1) return;

  // We don't have a starting scale for this system yet.
  Q2hat[iSys] = 0.0;
  // After prepare we always have zero branchings.
  nBranch[iSys] = 0;
  nBranchFSR[iSys] = 0;

  stateChangeSys[iSys] = true;
  stateChangeLast      = true;
  iSysWin = iSys;
  iNewSav = 0;

  // Note, for 2->2 systems, ISR::prepare() is called before
  // FRS::prepare() (if doISR) so ISR may already have done
  // everything.
  if ((doIF || doII) && isrPtr->prepared(iSys)) {

    // Ensure consistency between ISR + FSR lists.
    isHardSys[iSys]      = isrPtr->isHardSys[iSys];
    isResonanceSys[iSys] = false;
    doMECsSys[iSys]      = isrPtr->doMECsSys[iSys];
    polarisedSys[iSys]   = isrPtr->polarisedSys[iSys];

  // If ISR::prepare() not called for this system, prepare it now.
  } else {

    // Assume system 0 is the hard system (if not just a single particle).
    // Allows for case of forced final-state showers off a user-supplied
    // final state without beams or decaying resonance, see shower().
    isHardSys[iSys] = ( iSys == 0 && partonSystemsPtr->sizeOut(iSys) >= 2);
    isResonanceSys[iSys] = partonSystemsPtr->hasInRes(iSys);

    // Make light quarks (and initial-state partons) explicitly massless.
    if (!vinComPtr->mapToMassless(iSys, event, false)) return;
    // Then see if we know how to compute matrix elements for this conf.
    doMECsSys[iSys] = mecsPtr->prepare(iSys, event);
    // Decide if we should be doing ME corrections for next order.
    if (doMECsSys[iSys]) doMECsSys[iSys] = mecsPtr->doMEC(iSys, 1);
    // Initialize polarisation flag.
    polarisedSys[iSys]   = mecsPtr->isPolarised(iSys, event);
    // Colourise hard process and MPI systems first time they are encountered.
    bool doColourise = !isPrepared;
    if (!isHardSys[iSys] && !isResonanceSys[iSys]) doColourise = true;
    if (doColourise) colourPtr->colourise(iSys, event);

  }

  // Set up QCD antennae.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "Finding branchers...");
  if (doFF || doRF) {
    // In merging, allow to only generate emissions inside resonance systems.
    if ( isTrialShowerRes && !isResonanceSys[iSys]) {
      // Just clear what we've got.
      emittersRF.clear();
      splittersRF.clear();
      emittersFF.clear();
      splittersFF.clear();
      lookupEmitterRF.clear();
      lookupSplitterRF.clear();
      lookupEmitterFF.clear();
      lookupSplitterFF.clear();
    } else {
      // Set up QCD antennae.
      if (!setupQCDantennae(iSys,event)) return;
      // Save Born state, used later to reject sectors that involve clustering
      // away the "hard" (Born) partons. (For trial showers, this step is done
      // already, in VinciaMerging::getWeightCKKWL().)
      if (!isTrialShower) saveBornState(iSys, event);
    }
  }

  // Set up QED/EW systems.
  if (doQED) {
    bool isHard = isHardSys[iSys] || isResonanceSys[iSys];
    if (isHard) {
      // Check if doing full EW or "just" QED.
      if (doWeak && polarisedSys[iSys] &&
        ewShowerPtr->prepare(iSys,event,false) ) {
        ewHandlerHard = ewShowerPtr;
      } else {
        qedShowerHardPtr->clear(iSys);
        qedShowerHardPtr->prepare(iSys, event, false);
        ewHandlerHard = qedShowerHardPtr;
      }
      if (verbose >= DEBUG) {
        string msg = "ewHandlerHard = ";
        if (ewHandlerHard == ewShowerPtr) msg += "EW";
        else msg += "QED";
        printOut(__METHOD_NAME__, msg);
      }
    } else {
      // MPI and non-resonance (eg hadron-) decay systems always use QED.
      qedShowerSoftPtr->prepare(iSys, event, false);
    }
  }

  // Set starting scale for this system
  setStartScale(iSys, event);

  // Let others know we got to the end.
  isPrepared = true;
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "QCD antenna list after prepare:", dashLen);
    list();
    printOut(__METHOD_NAME__, "end", dashLen);
  }

}

//--------------------------------------------------------------------------

// Update FF, RF, and QED/EW antenna lists after each ISR emission.

void VinciaFSR::update( int iSys, Event& event, bool) {

  // Do nothing if not prepared for FSR.
  if (!isPrepared) return;
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "begin", dashLen);
    if (verbose >= DEBUG) event.list();
  }

  // Update any EW branchers in system.
  if (doQED) {
    if (isHardSys[iSys] || isResonanceSys[iSys])
      ewHandlerHard->update(event, iSys);
    else
      qedShowerSoftPtr->update(event, iSys);
  }

  // Return now if not doing QCD final-final branchings.
  if (!(doFF || doRF)) return;

  // Sanity check
  if (isResonanceSys[iSys]) {
    if (verbose >=NORMAL) infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Update called unexpectedly in resonance shower.","Exiting.");
    return;
  }

  // Count number of branches.
  nBranch[iSys]++;

  // Particles in the list are already updated by ISR. Find and save
  // all colours and anticolours; find all FF antennae.
  map<int,int> indexOfAcol;
  map<int,int> indexOfCol;
  vector< pair<int,int> > antFF;
  const bool findFF = true;
  const bool findIX = false;
  colourPtr->makeColourMaps(iSys, event, indexOfAcol, indexOfCol,
    antFF, findFF, findIX);

  // In principle, the colour maps could here be used to look for any
  // unmatched tags -> junctions.

  // Sanity check: can only shower QCD systems with more than 1 FF
  // connection.
  if (antFF.size() <= 0) return;

  // Update any final-state antennae with partons changed by ISR
  // branching.
  for (int i = 0; i < (int)emittersFF.size(); i++) {
    Brancher* brancherPtr = &emittersFF[i];
    // Update any antennae with legs that were modified by the ISR
    // branching, i.e. whose current legs have been marked with status
    // < 0.
    int iOld0 = brancherPtr->i0();
    int iOld1 = brancherPtr->i1();
    int iNew0 = iOld0;
    int iNew1 = iOld1;

    if (event[iOld0].status() < 0 || event[iOld1].status() < 0) {
      // Get new positions from indexOfCol, indexOfAcol (could also
      // use daughter information from old i0, i1).
      iNew0 = indexOfCol[event[iOld0].col()];
      iNew1 = indexOfAcol[event[iOld1].acol()];
      // Update emitter (and update pointer if location changed).
      emittersFF[i] = BrancherEmitFF(brancherPtr->system(), event,
        sectorShower, iNew0, iNew1, zetaGenSetFF);
      brancherPtr = &emittersFF[i];

      // Update lookup map and erase old keys.
      pair<int,bool> key = make_pair(iOld0, true);
      if (lookupEmitterFF.find(key)!=lookupEmitterFF.end())
        lookupEmitterFF.erase(key);
      key = make_pair(iOld1, false);
      if (lookupEmitterFF.find(key)!=lookupEmitterFF.end())
        lookupEmitterFF.erase(key);
      // Add new keys.
      key = make_pair(iNew0,true);
      lookupEmitterFF[key] = i;
      key = make_pair(iNew1,false);
      lookupEmitterFF[key] = i;

      // Update splitters.
      if (event[iOld0].isGluon()) {
        if (event[iNew0].isGluon())
          updateSplitterFF(event,iOld0,iOld1,iNew0,iNew1,true);
        else removeSplitterFF(iOld0);
      }
      if (event[iOld1].isGluon()) {
        if (event[iNew1].isGluon())
          updateSplitterFF(event,iOld1,iOld0,iNew1,iNew0,false);
        else removeSplitterFF(iOld1);
      }
    }

    // Remove the antennae out of the list. This way we can check
    // later if ISR added a new FF antenna i0/i1 is colour/anticolour.
    pair<int,int> pairNow = make_pair(iNew0,iNew1);
    vector< pair<int,int> >::iterator iter;
    iter = find (antFF.begin(), antFF.end(), pairNow);
    if (iter != antFF.end()) antFF.erase(iter);
  }

  // Is there a FF connection left?
  for (int i = 0; i < (int)antFF.size(); i++) {
    int i0 = antFF[i].first;  // i0/iNew[0] is colour.
    int i1 = antFF[i].second; // i1/iNew[2] is anticolour.
    // Don't include II or IF antennae.
    if (!event[i0].isFinal() || !event[i1].isFinal()) continue;
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Creating antenna between " << i0 << " , " << i1
         << " col = " << event[i0].col();
      printOut(__METHOD_NAME__, ss.str());
    }
    // Store new trial QCD gluon emission antenna.
    saveEmitterFF(iSys, event, i0, i1);
    // Store new trial QCD gluon splitting antenna(e).
    if (event[i0].isGluon()) saveSplitterFF(iSys, event, i0, i1, true);
    if (event[i1].isGluon()) saveSplitterFF(iSys, event, i1, i0, false);
  }

  // Decide if we should be doing ME corrections for next order.
  if (doMECsSys[iSysWin])
    doMECsSys[iSysWin] = mecsPtr->doMEC(iSysWin, nBranch[iSysWin]+1);

  // Sanity check.
  if (emittersFF.size() + splittersFF.size() <= 0) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "WARNING: Did not find any QCD antennae");
    return;
  }
  if (!check(iSysWin, event)) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": failed to update antennae");
    list();
    if (verbose >= DEBUG) printLookup();
    infoPtr->setAbortPartonLevel(true);
    return;
  }
  if (verbose >=DEBUG) {
    list();
    printLookup();
    printOut(__METHOD_NAME__, "end", dashLen);
  }

}

//--------------------------------------------------------------------------

// Select next pT in downwards evolution (TimeShower).

double VinciaFSR::pTnext(Event& event, double pTevolBegAll,
  double pTevolEndAll, bool, bool) {

  // Check if we are supposed to do anything.
  if (infoPtr->getAbortPartonLevel() || !isPrepared) return 0.;
  if (forceQuit) {
    if (verbose >= REPORT) printOut(__METHOD_NAME__,
      "User forced quit early");
    return 0.;
  }
  if (verbose >= DEBUG) {
    cout<<endl;
    printOut(__METHOD_NAME__, "begin", dashLen);
  }

  // Profiling.
  if (verbose >= REPORT) diagnosticsPtr->start(__METHOD_NAME__);

  // Denote VINCIA scales by "q", PYTHIA ones by "pTevol".
  double q2Begin  = pow2(pTevolBegAll);
  double q2EndAll = pow2(pTevolEndAll);

  // End scale is set wrong for merging in resonances.
  if (isTrialShowerRes) {
    double q2EndMerge = pow2(mergingHooksPtr->getShowerStoppingScale());
    q2EndAll = max(q2EndAll, q2EndMerge);
  }

  // Initialise.
  q2WinSav  = 0.;
  winnerEW  = nullptr;
  winnerQCD = nullptr;

  // Generate next gluon-emission trial scale (above qEndAll).
  if (doFF && emittersFF.size() > 0) {
    if ( !q2NextEmitQCD(q2Begin, q2EndAll) ) return 0.;
  }

  // Generate next gluon-splitting trial scale and compare to current qWin.
  if (doFF && splittersFF.size() > 0) {
    if ( !q2NextSplitQCD(q2Begin, q2EndAll) ) return 0.;
  }

  // Generate next resonance gluon-emission trial and compare to current qWin.
  if (!isTrialShower && doRF && emittersRF.size() > 0) {
    if ( !q2NextEmitResQCD(q2Begin, q2EndAll) ) return 0.;
  }

  // Generate nex resonance gluon-splitting trial and compare to current qWin.
  if (!isTrialShower && doRF && splittersRF.size() > 0) {
    if ( !q2NextSplitResQCD(q2Begin, q2EndAll) ) return 0.;
  }

  // Generate next EW trial scale and compare to current qWin.
  if (!isTrialShower && doQED) {
    double q2EW = 0.;
    // EW emissions in the hard system.
    if (ewHandlerHard->nBranchers() >= 1) {
      q2EW     = ewHandlerHard->q2Next(event, q2Begin, q2EndAll);
      winnerEW = ewHandlerHard;
    }
    // QED emissions in MPI systems (only if we are not in a resonance decay)
    if (nRecurseResDec == 0 && qedShowerSoftPtr->nBranchers() >= 1) {
      if (verbose >= DEBUG) printOut(__METHOD_NAME__,
        "Generating QED off MPI");
      double q2EWmpi = qedShowerSoftPtr->q2Next(event, q2Begin, q2EndAll);
      if (q2EWmpi > q2EW) {
        q2EW = q2EWmpi;
        winnerEW = qedShowerSoftPtr;
      }
    }
    if (q2EW > q2Begin+NANO) {
      stringstream ss;
      ss << "q2Begin = "<<q2Begin<<" q2EW = " << q2EW;
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Genereated q2EW > q2Begin.",ss.str());
      infoPtr->setAbortPartonLevel(true);
      return 0.;
    }
    // Check for winning condition.
    // Note: EW resonance decays can go all the way to zero.
    if (q2EW > q2WinSav && q2EW > 0.) {
      q2WinSav  = q2EW;
      // Mark QCD as the loser.
      winnerQCD = nullptr;
    }
    else {
      // Mark EW as the loser.
      winnerEW  = nullptr;
    }
  }

  // If non-zero branching scale found: continue.
  if (winnerQCD != nullptr && q2WinSav > q2EndAll) {
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << " QCD Winner at scale qWinNow = "
         <<  sqrt(q2WinSav)
         << " col = " << event[winnerQCD->i0()].col()
         << " in System " << winnerQCD->system()
         << " qbegin = "<< pTevolBegAll;
    }
  }
  else if (winnerEW != nullptr && q2WinSav > 0.) {
   if (verbose >= DEBUG) {
     stringstream ss;
     ss << "=== EW Winner at scale qWinNow = "
        << sqrt(q2WinSav);
     if (winnerEW->lastIsResonanceDecay()) ss<<" (resonance decay)"<<endl;
     printOut(__METHOD_NAME__, ss.str());
     list();
   }
  }
  // Else no more branchings. Finalize.
  else {
    q2WinSav  = 0.0;
    winnerQCD = nullptr;
    winnerEW  = nullptr;
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__,
        "=== No FSR trial branchings above cutoff");
      event.list();
    }
  }

  // Profiling.
  if (verbose >= REPORT) diagnosticsPtr->stop(__METHOD_NAME__);

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return (q2WinSav > 0) ? sqrt(q2WinSav) : 0.0;

}

//--------------------------------------------------------------------------

// Branch event, including accept/reject veto (TimeShower).

bool VinciaFSR::branch(Event& event, bool ) {

  // Verbose output.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Diagnostics.
  if (verbose >= REPORT) diagnosticsPtr->start(__METHOD_NAME__);

  // Initialise info on this branching.
  hasWeaklyRadiated = false;

  // EW branchings
  if (winnerEW != nullptr) {

    // Do the EW branching, including a resonance shower if branching
    // is a resonance decay. The idea is that after the branching, the
    // overall event has the same number of parton systems as before,
    // which have all been evolved down to the current evolution scale.
    if (!branchEW(event)) {
      if (verbose >= REPORT)
        diagnosticsPtr->stop(__METHOD_NAME__,"veto(branchEW)");
      return false;
    }
    else hasWeaklyRadiated = true;
  }

  // QCD Branchings.
  else {

    // Do the QCD branching.
    if (!branchQCD(event)) {
      if (verbose >= REPORT)
        diagnosticsPtr->stop(__METHOD_NAME__,"veto(branchQCD)");
      return false;
    }
  }

  // Save info variables.
  stateChangeSys[iSysWin] = true;
  stateChangeLast         = true;
  pTLastAcceptedSav       = sqrt(q2WinSav);

  // Diagnostics.
  if (verbose >= REPORT) diagnosticsPtr->stop(__METHOD_NAME__,"accept");

  // Done.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return true;

}

//--------------------------------------------------------------------------

// Handle a resonance decay, including showering of resonance system (with
// preserved resonance mass) down to the scale pTmerge, at which the
// produced partons are merged back into the system that produced the
// resonance (iSysMot).
// Assumes decay channel and kinematics already selected and present in
// process or event record.
// Note: this method can be called recursively for nested resonance decays.

bool VinciaFSR::resonanceShower(Event& process, Event& event,
  vector<int>& iPosBefShow, double pTmerge) {

  // Verbose output.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin" , dashLen);

  // Keep track of how many times resonanceShower has called itself.
  ++nRecurseResDec;

  int iMother = 0;
  // Is this a decay already in the event record? (E.g., produced by EW
  // shower.) If so, we can ignore the process event record and just need
  // to do the (interleaved) resonance shower off the decay products.
  if (process.size() <= 1) {
    iMother = iPosBefShow[0];
  }

  // Else assume this is a resonance decay from the hard process.
  // Update and copy down decay products (including junctions).
  // Then do interleaved resonance shower.
  else if (iPosBefShow.size() >= 2) {

    // Save which resonance is currently being handled (since iHardResDecSav
    // may be overwritten when called recursively).
    int iHardMother = iHardResDecSav;
    // Do not try this resonance again (to avoid risk of infinite loop).
    // If we fail, PartonLevel will have a (non-interleaved) chance at
    // end of evolution.
    pTresDecSav[iHardMother] = 0.;

    // Mother in hard process and in event recrd. iPosBefShow indicates
    // position in complete event before shower evolution started,
    // so may have moved.
    Particle& hardMother = process[iHardMother];
    int iBefMother       = iPosBefShow[iHardMother];
    iMother              = event[iBefMother].iBotCopyId();
    Particle& aftMother  = event[iMother];
    // If this particle has already been decayed (eg by EW shower), then no
    // need to do anything further.
    if (aftMother.status() < 0) {
      --nRecurseResDec;
      return false;
    }

    // Prepare to move daughters from process to event record
    int sizeOld = event.size();

    // Mother can have been moved by showering (in any of previous steps),
    // so prepare to update colour and momentum information for system.
    int colBef  = hardMother.col();
    int acolBef = hardMother.acol();
    int colAft  = aftMother.col();
    int acolAft = aftMother.acol();
    // Construct boost matrix to go from process to event frame.
    RotBstMatrix M;
    M.bst( hardMother.p(), aftMother.p());

    // Check if this decay contains one (or more) junction structure(s) that
    // should be copied from process to event as part of this decay.
    vector<int> nMatchJun;
    // Check for junctions that were already copied (e.g., by setupHardSys)
    for (int iJun = 0; iJun < process.sizeJunction(); ++iJun) {
      nMatchJun.push_back(0);
      for (int kJun = 0; kJun < event.sizeJunction(); ++kJun) {
        if (process.kindJunction(iJun) != event.kindJunction(kJun)) continue;
        int nMatch = 0;
        for (int jLeg = 0; jLeg <= 2; ++jLeg) {
          if (process.colJunction(iJun,jLeg) == event.colJunction(kJun,jLeg))
            ++nMatch;
          // Mark this junction as already copied (force to be skipped).
          if (nMatch == 3) nMatchJun[iJun] = -999;
        }
      }
    }

    // TODO "colourise": allow to assign subleading colour indices to any new
    // colour lines produced in the decay.

    // Move daughters from process to event and apply boosts + colour updates.
    int iHardDau1 = hardMother.daughter1();
    int iHardDau2 = hardMother.daughter2();
    for (int iHardDau = iHardDau1; iHardDau <= iHardDau2; ++iHardDau) {

      // Copy daughter from process to event.
      int iNow = event.append( process[iHardDau] );

      // Update iPos map from process to event
      iPosBefShow[iHardDau] = iNow;

      // Now set properties of this daughter in event.
      Particle& now = event.back();
      now.mother1(iMother);
      // Currently outgoing ones should not count as decayed.
      if (now.status() == -22) {
        now.statusPos();
        now.daughters(0, 0);
      }

      // Check if this decay contains a junction in hard event.
      for (int iJun = 0; iJun < process.sizeJunction(); ++iJun) {
        // Only consider junctions that can appear in decays.
        int kindJunction = process.kindJunction(iJun);
        if (kindJunction >= 5) continue;
        for (int iLeg = 0; iLeg <= 2; ++iLeg) {
          // Check if colour of hard mother matches an incoming junction leg.
          if (kindJunction >= 3 && iLeg == 0) {
            // Only check mother once (not once for every daughter).
            if (iHardDau != iHardDau1) continue;
            int colLeg = process.colJunction(iJun,iLeg);
            if ( (kindJunction == 3 && hardMother.acol() == colLeg)
              || (kindJunction == 4 && hardMother.col() == colLeg ) )
              nMatchJun[iJun] += 1;
          }
          // Check if daughter colour matches an outgoing junction leg.
          else {
            int colLeg = process.colJunction(iJun,iLeg);
            int colDau = (kindJunction == 1 || kindJunction == 3) ?
              process[iHardDau].col() : process[iHardDau].acol();
            if ( colLeg == colDau ) nMatchJun[iJun] += 1;
          }
        }
        // If we have 3 matches, copy down junction from process to event.
        if ( nMatchJun[iJun] == 3 ) {
          // Check for changed colors and update as necessary.
          Junction junCopy = process.getJunction(iJun);
          for (int iLeg = 0; iLeg <= 2; ++iLeg) {
            int colLeg = junCopy.col(iLeg);
            if (colLeg == colBef) junCopy.col(iLeg, colAft);
            if (colLeg == acolBef) junCopy.col(iLeg, acolAft);
          }
          event.appendJunction(junCopy);
          // Mark junction as copied (to avoid later recopying)
          nMatchJun[iJun] = -999;
        }
      }

      // Update colour and momentum information.
      if (now.col() == colBef) now.col( colAft);
      if (now.acol() == acolBef) now.acol( acolAft);
      // Sextet mothers have additional (negative) tag
      if (now.col() == -acolBef) now.col( -acolAft);
      if (now.acol() == -colBef) now.acol( -colAft);
      now.rotbst( M);

      // Update vertex information.
      if (now.hasVertex()) now.vProd( event[iMother].vDec() );

      // Finally, check if daughter is itself a resonance.
      // If so, add interleaving scale.
      if (process[iHardDau].isResonance() && process[iHardDau].status() < 0) {
        pTresDecSav[iHardDau]   = calcPTresDec(process[iHardDau]);
        idResDecSav[iHardDau]   = process[iHardDau].id();
      }
    } // End loop over resonance daughters.

    // If everything worked, mark mother decayed and set daughters.
    event[iMother].statusNeg();
    event[iMother].daughters(sizeOld, event.size() - 1);

    // Update table of pre-shower process->event indices in case of changes.
    iPosBefSav = iPosBefShow;

  }

  int iSysMot          = partonSystemsPtr->getSystemOf(iMother);
  // If this is a recursive call, upstream system is the one which was added
  // most recently.
  if (nRecurseResDec >= 2) iSysMot = partonSystemsPtr->sizeSys()-1;
  Particle& mother     = event[iMother];
  vector<int> children = mother.daughterList();

  // Sanity check.
  if (iSysMot <= -1) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": "
      "could not identify upstream system for resonance decay");
    if (verbose >= REPORT) {
      if (verbose >= DEBUG) event.list();
      partonSystemsPtr->list();
      cout<<" iMother = "<<iMother<<endl;
    }
    return false;
  }

  // Add new system for this resonance decay + shower, + any nested resonance
  // decays that may be done (recursively) while handling this system.
  // (Partons will be moved from this system to upstream one at end.)
  int iSysRes = partonSystemsPtr->addSys();
  // Insert resonance into system, then add daughters.
  partonSystemsPtr->setInRes( iSysRes, iMother);
  partonSystemsPtr->setSHat(  iSysRes, pow2(mother.m()) );
  partonSystemsPtr->setPTHat( iSysRes, 0.5 * mother.m() );
  for (unsigned int i = 0; i < children.size(); ++i) {
    int iDau = children[i];
    if (event[iDau].isFinal()) partonSystemsPtr->addOut(iSysRes,iDau);
  }

  // Check if we should assign polarisations to this system.
  if (helicityShower) {
    // Force reselect daughter polarisations if mother changed helicity.
    int iPolMotTop = event[mother.iTopCopyId()].pol();
    int iPolMotNow = mother.pol();
    bool forcePolarise = (iPolMotTop != iPolMotNow);
    vector<Particle> stateRes;
    stateRes.push_back(mother);
    // Check if daughters have helicities.
    for (int iOut=0; iOut<partonSystemsPtr->sizeOut(iSysRes); ++iOut) {
      stateRes.push_back(event[partonSystemsPtr->getOut(iSysRes, iOut)]);
      // Force all outgoing particles to have code 23.
      stateRes.back().status(23);
      // Check if this daughter needs to be polarised.
      int id     = stateRes.back().id();
      int nSpins = particleDataPtr->spinType(abs(id));
      if (nSpins == 1) stateRes.back().pol(0);
      if (stateRes.back().pol() == 9) forcePolarise = true;
    }
    if (forcePolarise) {
      // First, step up one system and see if we can polarise iSysMot+iSysRes.
      vector<Particle> stateMot;
      if (partonSystemsPtr->hasInRes(iSysMot))
        stateMot.push_back(event[partonSystemsPtr->getInRes(iSysMot)]);
      else {
        stateMot.push_back(event[partonSystemsPtr->getInA(iSysMot)]);
        stateMot.push_back(event[partonSystemsPtr->getInB(iSysMot)]);
      }
      for (int iOut=0; iOut<partonSystemsPtr->sizeOut(iSysMot); ++iOut) {
        int i = partonSystemsPtr->getOut(iSysMot, iOut);
        // Add all outgoing partons from mother system except resonance.
        if (i != iMother) {
          stateMot.push_back(event[partonSystemsPtr->getOut(iSysMot, iOut)]);
          stateMot.back().status(23);
        }
      }
      // Add resonance daughters to stateMot.
      int iResBeg = stateMot.size();
      stateMot.insert(stateMot.end(), ++stateRes.begin(), stateRes.end());
      // First try if MG interface can select helicities using stateMot.
      if (mecsPtr->meAvailable(stateMot) && mecsPtr->polarise(stateMot)) {
        // Copy daughter polarisations into event record.
        for (int j=0; j<partonSystemsPtr->sizeOut(iSysRes); ++j) {
          int iEvent = partonSystemsPtr->getOut(iSysRes, j);
          // Consistency check.
          if (iResBeg+j >= (int)stateMot.size() ||
            stateMot[iResBeg+j].id() != event[iEvent].id()) {
            infoPtr->errorMsg("Error in "+__METHOD_NAME__+": "
              "daughter mismatch after helicity selection (stateMot)");
            break;
          } else event[iEvent].pol(stateMot[iResBeg + j].pol());
        }
      }
      // Else see if MG or EW shower can select helicities using stateRes.
      // Note: require mother (stateRes[0]) to have a helicity.
      else if ( stateRes[0].pol() == 9 ) { }
      else if ( (mecsPtr->meAvailable(stateRes) && mecsPtr->polarise(stateRes))
        || (ewShowerPtr != nullptr && ewShowerPtr->polarise(stateRes)) ) {
        // Copy daughter polarisations into event record.
        for (int j=0; j<partonSystemsPtr->sizeOut(iSysRes); ++j) {
          int iEvent = partonSystemsPtr->getOut(iSysRes, j);
          // Consistency check.
          if (j+1 >= (int)stateRes.size() ||
            stateRes[1+j].id() != event[iEvent].id()) {
            infoPtr->errorMsg("Error in "+__METHOD_NAME__+": "
              "daughter mismatch after helicity selection (stateRes)");
            break;
          } else event[iEvent].pol(stateRes[1 + j].pol());
        }
      } else {
        if (verbose >= REPORT) {
          infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": "
            +"could not assign daughter helicities");
        }
      }
    }
  }

  double pTmax = 0.5 * mother.m();
  // Userhooks and Trial showers accounted for here in PartonLevel.
  // TODO: discuss whether to include that here and how.
  // For now, just do pure showers.
  prepare(iSysRes, event, false);

  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "pTmax = "<<pTmax<<" pTmerge = "<<pTmerge;
    printOut(__METHOD_NAME__, ss.str());
  }

  // Begin evolution down in pT, allowing for nested resonance decays.
  if (pTmax > pTmerge) {
    int nBranchNow = 0;
    int nLoop = 0;
    do {

      // Interleave (intra-system) FSR and resonance decays.
      double pTtimes  = (doFSRinResonances) ? pTnext( event, pTmax, pTmerge)
        : -1.;
      double pTresDec = pTnextResDec();

      // Do a final-state emission.
      if ( pTtimes > 0. && pTtimes > max( pTresDec, pTmerge) ) {
        if (branch(event)) ++nBranchNow;
        pTmax = pTtimes;
      }

      // Do a resonance decay from hard (process) event, nested (recursively)
      // inside the current one.
      else if (pTresDec > 0. && pTresDec > pTmerge) {
        resonanceShower(process, event, iPosBefShow, pTresDec);
        pTmax = pTresDec;
      }

      // Do nothing.
      else pTmax = -1.;

      // Check loop counter to avoid infinite loop.
      if (++nLoop >= NLOOPMAX) {
        infoPtr->errorMsg("Error in VinciaFSR::resonanceShower: "
          "infinite loop");
        break;
      }

    } while (pTmax > pTmerge && !infoPtr->getAbortPartonLevel());
  }

  // Check for abort condition.
  if (infoPtr->getAbortPartonLevel()) {
    nRecurseResDec--;
    return false;
  }

  // Udate upstream parton system. Replace outgoing resonance by its first
  // daughter. Then add remaining partons.
  for ( int iP = 0; iP < partonSystemsPtr->sizeOut(iSysRes); ++iP) {
    int iNew = partonSystemsPtr->getOut( iSysRes, iP);
    if (iP == 0)
      partonSystemsPtr->replace( iSysMot, iMother, iNew);
    else
      partonSystemsPtr->addOut( iSysMot, iNew);
  }

  // Now delete the new system, otherwise beam remnant handling breaks.
  partonSystemsPtr->popBack();

  // Also clear added QED systems.
  if (doQED) ewHandlerHard->clear(iSysRes);

  // Reset antenna list for continued evolution in upstream system.
  prepare(iSysMot, event, false);

  // Done.
  nRecurseResDec--;
  iSysWin = iSysMot;

  return true;

}

//--------------------------------------------------------------------------

// Utility to print antenna list; for DEBUG mainly (TimeShower).

void VinciaFSR::list() const {

  bool withLegend = true;
  bool withTitle  = true;
  if (emittersRF.size() + emittersFF.size() + splittersRF.size()
    + splittersFF.size() == 0) {
    cout << " --------  No FF or RF QCD antennae  ------------------------"
      "--------------------------------------\n";
  } else {
    // Loop over antenna lists and print them.
    for (unsigned int i = 0; i < emittersRF.size(); ++i) {
      if (i == 0) {
        emittersRF[i].list("FF & RF Gluon Emission Antennae",withLegend);
        withTitle  = false;
        withLegend = false;
      }
      else emittersRF[i].list();
    }
    for (int i = 0; i < (int)emittersFF.size(); ++i) {
      if (i == 0 && withTitle) {
        emittersFF[i].list("FF & RF Gluon Emission Antennae",withLegend);
        withLegend = false;
        withTitle  = false;
      }
      else emittersFF[i].list();
    }
    withTitle = true;
    for (unsigned int i = 0; i < splittersRF.size(); ++i) {
      if (i == 0) {
        splittersRF[i].list("FF & RF Gluon Splitting Antennae", withLegend);
        withLegend = false;
        withTitle  = false;
      }
      else splittersRF[i].list();
    }
    for (int i = 0; i < (int)splittersFF.size(); ++i) {
      if (i == 0 && withTitle) {
        splittersFF[i].list("FF & RF Gluon Splitting Antennae", withLegend);
        withLegend = false;
      }
      else splittersFF[i].list();
    }
  }

  // Are there interleaved resonance decays?
  bool headerPrinted = false;
  for (unsigned int iRD=0; iRD<pTresDecSav.size(); ++iRD) {
    if (pTresDecSav[iRD] <= 0.) continue;
    if (!headerPrinted) {
      cout << " --------  Interleaved Resonance Decays  ----------------"
        "------------------------------------------\n";
      headerPrinted = true;
    }
    cout<<"        RD        process["<<iRD<<"]" << setw(18) << " "
        << num2str(idResDecSav[iRD],9) << setw(11) << " "
        << particleDataPtr->colType(abs(idResDecSav[iRD])) << setw(22) << " "
        << num2str(pTresDecSav[iRD],9) << endl;
  }
  if (!headerPrinted) {
    cout << " --------  No Interleaved Resonance Decays  ----------------"
      "---------------------------------------\n";
  }
  else {
    cout << " --------  End of List of FSR Branchers ----------------------"
      "-------------------------------------\n";
  }
}

//--------------------------------------------------------------------------

// Initialise pointers to Vincia objects.

void VinciaFSR::initVinciaPtrs(VinciaColour* colourPtrIn,
  shared_ptr<VinciaISR> isrPtrIn, MECs* mecsPtrIn,
  Resolution* resolutionPtrIn, VinciaCommon* vinComPtrIn,
  VinciaWeights* vinWeightsPtrIn) {
  colourPtr     = colourPtrIn;
  isrPtr        = isrPtrIn;
  mecsPtr       = mecsPtrIn;
  resolutionPtr = resolutionPtrIn;
  vinComPtr     = vinComPtrIn;
  weightsPtr    = vinWeightsPtrIn;
}

//--------------------------------------------------------------------------

// Print header information (version, settings, parameters, etc.).

void VinciaFSR::header() {

  // Must be initialised before printing header.
  if (!isInit) return;

  // Avoid printing header several times.
  if (headerIsPrinted) return;
  headerIsPrinted = true;

  cout <<setprecision(3);
  cout.setf(ios::left);
  cout << "\n";
  cout << " *-------  VINCIA Global Initialization  ------"
       << "-------------------------------------------------*\n";

  // Print header information about shower.
  cout << " |\n";
  cout << " | QCD Shower:     doII,doIF,doFF,doRF       =   "
       << bool2str(doII,3) <<","<<bool2str(doIF,3)
       <<","<<bool2str(doFF,3)<<","<<bool2str(doRF,3)
       <<"\n";
  cout << " |                 nGluonToQuark (FSR)       = "
       << num2str(settingsPtr->mode("Vincia:nGluonToQuark"),9)<<"\n";
  cout << " |                 convertGluonToQuark (ISR) = "
       << bool2str(settingsPtr->flag("Vincia:convertGluonToQuark"),9)<<"\n";
  cout << " |                 convertQuarkToGluon (ISR) = "
       << bool2str(settingsPtr->flag("Vincia:convertQuarkToGluon"),9)<<"\n";
  cout << " |                 helicityShower            = "
       << bool2str(settingsPtr->flag("Vincia:helicityShower"),9)<<"\n";
  cout << " |                 sectorShower              = "
       << bool2str(settingsPtr->flag("Vincia:sectorShower"),9)<<"\n";

  // Print header information about alphaS
  cout << " |\n"
       << " | Alpha_s:        alphaS(mZ)|MSbar          = "
       << num2str(alphaSvalue,9)<<"\n"
       << " |                 order                     = "
       << num2str(alphaSorder,9)<<"\n";
  if (alphaSorder >= 1) {
    if (useCMW) {
      cout << " |                 LambdaQCD[nF]|MSbar       = "
           << num2str(vinComPtr->alphaStrong.Lambda3(),9)<<"[3] "
           << num2str(vinComPtr->alphaStrong.Lambda4(),7)<<"[4] "
           << num2str(vinComPtr->alphaStrong.Lambda5(),7)<<"[5] "
           << num2str(vinComPtr->alphaStrong.Lambda6(),7)<<"[6]\n";
      cout << " |                 LambdaQCD[nF]|CMW         = "
           << num2str(vinComPtr->alphaStrongCMW.Lambda3(),9)<<"[3] "
           << num2str(vinComPtr->alphaStrongCMW.Lambda4(),7)<<"[4] "
           << num2str(vinComPtr->alphaStrongCMW.Lambda5(),7)<<"[5] "
           << num2str(vinComPtr->alphaStrongCMW.Lambda6(),7)<<"[6]\n";
    } else {
      cout << " |                 LambdaQCD[nF]            = "
           << num2str(vinComPtr->alphaStrong.Lambda3(),9)<<"[3] "
           << num2str(vinComPtr->alphaStrong.Lambda4(),7)<<"[4] "
           << num2str(vinComPtr->alphaStrong.Lambda5(),7)<<"[5] "
           << num2str(vinComPtr->alphaStrong.Lambda6(),7)<<"[6]\n";
    }
    cout << " |                 useCMW                    = "
         << bool2str(settingsPtr->flag("Vincia:useCMW"),9)<<"\n";
    cout << " |                 renormMultFacEmitF        = "
         << num2str(settingsPtr->parm("Vincia:renormMultFacEmitF"),9)
         <<" (muR prefactor for FSR emissions)\n";
    cout << " |                 renormMultFacSplitF       = "
         << num2str(settingsPtr->parm("Vincia:renormMultFacSplitF"),9)
         <<" (muR prefactor for FSR splittings)\n";
    cout << " |                 renormMultFacEmitI        = "
         << num2str(settingsPtr->parm("Vincia:renormMultFacEmitI"),9)
         <<" (muR prefactor for ISR emissions)\n";
    cout << " |                 renormMultFacSplitI       = "
         << num2str(settingsPtr->parm("Vincia:renormMultFacSplitI"),9)
         <<" (muR prefactor for ISR splittings)\n";
    cout << " |                 renormMultFacConvI        = "
         << num2str(settingsPtr->parm("Vincia:renormMultFacConvI"),9)
         <<" (muR prefactor for ISR conversions)\n";

    cout << " |                 alphaSmuFreeze            = "
         << num2str(alphaSmuFreeze,9)<<"\n";
    cout << " |                 alphaSmax                 = "
         << num2str(alphaSmax,9)<<"\n";
  }

  // Print header information about IR regularization.
  cout << " |\n"
       << " |   IR Reg.:      cutoffScaleEmitFF         = "
       << num2str(sqrt(q2CutoffEmit),9)<<"\n"
       << " |                 cutoffScaleSplitFF        = "
       << num2str(sqrt(q2CutoffSplit),9)<<"\n"
       << " |                 cutoffScaleII             = "
       << num2str(settingsPtr->parm("Vincia:cutoffScaleII"),9)<<"\n"
       << " |                 cutoffScaleIF             = "
       << num2str(settingsPtr->parm("Vincia:cutoffScaleIF"),9)<<"\n";

  // Information about EW/QED showers.
  cout << " |\n";
  cout << " |   QED/EW:       EWmode                    = "
       <<num2str(settingsPtr->mode("Vincia:EWmode"),9)<<"\n";
  if (settingsPtr->mode("Vincia:EWmode") >= 1) {
    cout << " |                 nGammaToQuark             = "
         <<num2str(settingsPtr->mode("Vincia:nGammaToQuark"),9)<<"\n"
         << " |                 nGammaToLepton            = "
         <<num2str(settingsPtr->mode("Vincia:nGammaToLepton"),9)<<"\n"
         << " |                 convertGammaToQuark       = "
         <<bool2str(settingsPtr->flag("Vincia:convertGammaToQuark"),9)<<"\n"
         << " |                 convertQuarkToGamma       = "
         <<bool2str(settingsPtr->flag("Vincia:convertQuarkToGamma"),9)<<"\n";
    cout << " |                 EWmodeMPI                 = "
         <<num2str(settingsPtr->mode("Vincia:EWmodeMPI"),9)<<"\n";
    // Further information about EW.
    if (ewMode >=3) {
      cout << " |                 doBosonicInterference     = "
           <<bool2str(settingsPtr->flag("Vincia:doBosonicInterference"),9)
           <<"\n";
    }
  }

  // Print header information about antenna functions.
  if (verbose >= NORMAL) {
    cout<<" |\n"
        <<" | AntennaFunctions:         "
        <<"                      chargeFactor   kineMap"<<endl;
    int modeSLC      = settingsPtr->mode("Vincia:modeSLC");

    // FF and RF antennae.
    vector<enum AntFunType> antFunTypes = antSetPtr->getAntFunTypes();
    for (size_t i=0; i<antFunTypes.size(); ++i) {
      AntennaFunction* antFunPtr =
        antSetPtr->getAntFunPtr(antFunTypes[i]);
      if (antFunPtr == nullptr) continue;
      // Print antenna name.
      cout.setf(ios::left);
      cout << setprecision(2);
      string antName = antFunPtr->vinciaName()+" ["+antFunPtr->humanName()+"]";
      cout << " |                 " << left << setw(32) << antName << "    ";
      // Print colour/charge factor.
      double chargeFac = antFunPtr->chargeFac();
      cout<<fixed<<setw(6)<<chargeFac;
      // Put asterisk next to QG colour factor if using -1/NC2 correction.
      if (modeSLC == 2) {
        if (antFunPtr->vinciaName() == "Vincia:QGEmitFF" ||
            antFunPtr->vinciaName() == "Vincia:GQEmitFF" ||
            antFunPtr->vinciaName() == "Vincia:QGEmitRF" ||
            antFunPtr->vinciaName() == "Vincia:GQEmitRF") cout << "*";
        else cout << " ";
      } else cout << " ";
      int kineMap = antFunPtr->kineMap();
      cout << "    " << right << setw(5) << kineMap << left << "\n";
    }

    // II and IF antennae.
    AntennaSetISR* antSetISRPtr = isrPtr->antSetPtr;
    if (antSetISRPtr != nullptr) {
      vector<enum AntFunType> antFunTypesISR = antSetISRPtr->getAntFunTypes();
      for (size_t i = 0; i < antFunTypesISR.size(); ++i) {
        enum AntFunType antFunTypePhys = antFunTypesISR[i];
        AntennaFunctionIX* antFunPtr =
          antSetISRPtr->getAntFunPtr(antFunTypePhys);
        if (antFunPtr == nullptr) continue;
        // Print antenna name.
        cout.setf(ios::left);
        cout << setprecision(2) << " |                 " << left << setw(32)
             << antFunPtr->vinciaName() + " [" + antFunPtr->humanName() + "]"
             << "    ";
        // Print colour/charge factor.
        double chargeFac = antFunPtr->chargeFac();
        cout << fixed << setw(6) << chargeFac;
        if (modeSLC == 2) {
          if (antFunPtr->vinciaName() == "Vincia:QGEmitII" ||
              antFunPtr->vinciaName() == "Vincia:GQEmitII" ||
              antFunPtr->vinciaName() == "Vincia:QGEmitIF" ||
              antFunPtr->vinciaName() == "Vincia:GQEmitIF") cout<<"*";
          else cout << " ";
        } else cout << " ";
        int kineMap = antFunPtr->kineMap();
        cout << "    " << right << setw(5) << kineMap << left << "\n";
      }
      if (modeSLC == 2)
        cout << " |                 *: GQ antennae interpolate between "
             << "CA and 2CF (modeSLC = 2)" << endl;
    }
  }
  // Print header information about matrix-element Corrections.
  mecsPtr->header();

  // Print references.
  // TODO: Init output should be restructured into a vector<string> where each
  // piece of Vincia adds lines, and a separate vector<string> for
  // references.
  cout << " |\n";
  cout << " |-------------------------------------------"
       << "---------------------------------------------*\n |\n";
  cout << " | References :"<<endl;
  // Vincia QCD shower.
  cout << " |    VINCIA Shower   : Brooks, Preuss, Skands, "
       << "JHEP07(2020)032, arXiv:2003.00702" << endl;
  // Vincia QED multipole shower.
  if (ewMode == 2) {
    cout << " |    VINCIA QED      : Skands, Verheyen, "
         << "PLB811(2020)135878 arXiv:2002.04939" << endl;
  }
  // Vincia Weak shower.
  else if (ewMode == 3) {
    cout << " |    VINCIA EW       : Kleiss, Verheyen, "
         << "EPJC80(2020)10,980 arXiv:2002.09248" << endl;
  }
  // Pythia 8 main reference.
  cout << " |    PYTHIA 8        : Sjostrand et al., CPC191(2015)159, "
       << "arXiv:1410.3012" << endl;
  cout << " |\n *-------  End VINCIA Initialization  "
       << "----------------------------------------------------*\n\n";
  cout.setf(ios::right);

}

//--------------------------------------------------------------------------

// Check event.

bool VinciaFSR::check(int iSys, Event &event) {
  stringstream ss;
  // All FF emitters must be final-state particles.
  for (int i = 0; i < (int)emittersFF.size(); ++i) {
    // If specific iSys requested, only check that system.
    if (iSys >= 0 && emittersFF[i].system() != iSys) continue;
    if (!event[emittersFF[i].i0()].isFinal()) {
      if (verbose >= REPORT) {
        event.list();
        list();
      }
      if (verbose >= NORMAL) {
        ss << "EmitterFF " << i
           << " i0 = " << emittersFF[i].i0() << " not final.";
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Failed to update emitterFF (not final).", ss.str());
      }
      return false;
    } else if (!event[emittersFF[i].i1()].isFinal()) {
      if (verbose >= REPORT) {
        event.list();
        list();
      }
      if (verbose >= NORMAL) {
        ss << "EmitterFF " << i
           << " i1 = " << emittersFF[i].i1() << " not final.";
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Failed to update emitterFF (not final).", ss.str());
      }
      return false;
    }
  }
  // All FF splitters must be final-state particles.
  for (int i = 0; i < (int)splittersFF.size(); ++i) {
    if (iSys >= 0 && splittersFF[i].system() != iSys) continue;
    if (!event[splittersFF[i].i0()].isFinal()) {
      if (verbose >= REPORT) {
        event.list();
        list();
      }
      if (verbose >= NORMAL) {
        ss << "SplitterFF " << i
           << " i0 = " << splittersFF[i].i0() << " not final.";
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Failed to update splitterFF (not final).", ss.str());
      }
      return false;
    } else if (!event[splittersFF[i].i1()].isFinal()) {
      if (verbose >= REPORT) {
        event.list();
        list();
      }
      if (verbose >= NORMAL) {
        ss << "SplitterFF " << i
           << " i1 = " << splittersFF[i].i1() << " not final.";
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Failed to update splitterFF (not final).", ss.str());
      }
      return false;
    }
  }
  // All RF emitters must be initial-final systems
  for (int i = 0; i < (int)emittersRF.size(); ++i) {
    if (iSys >= 0 && emittersRF[i].system() != iSys) continue;
    int iRes = emittersRF[i].i0();
    int iFin = emittersRF[i].i1();
    if (emittersRF[i].posR() == 1) {
      iRes = emittersRF[i].i1();
      iFin = emittersRF[i].i0();
    }
    if (!event[abs(iFin)].isFinal()) {
      if (verbose >= REPORT) {
        event.list();
        list();
      }
      if (verbose >= NORMAL) {
        ss << "EmitterRF " << i << " iF = " << iFin << " not final.";
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Failed to update emitterRF (not final).", ss.str());
      }
      return false;
    } else if (event[abs(iRes)].isFinal()) {
      if (verbose >= REPORT) {
        event.list();
        list();
      }
      if (verbose >= NORMAL) {
        ss << "EmitterRF " << i << " iR = " << iRes << " not incoming.";
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Failed to update emitterRF (not final).", ss.str());
      }
      return false;
    }
  }
  // All RF splitters must be initial-final systems
  for (int i = 0; i < (int)splittersRF.size(); ++i) {
    if (iSys >= 0 && splittersRF[i].system() != iSys) continue;
    int iRes = splittersRF[i].i0();
    int iFin = splittersRF[i].i1();
    if (splittersRF[i].posR() == 1) {
      iRes = splittersRF[i].i1();
      iFin = splittersRF[i].i0();
    }
    if (!event[abs(iFin)].isFinal()) {
      if (verbose >= REPORT) {
        event.list();
        list();
      }
      if (verbose >= NORMAL) {
        ss << "SplitterRF " << i << " iF = " << iFin << " not final.";
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Failed to update splitterRF (not final).", ss.str());
      }
      return false;
    } else if (event[abs(iRes)].isFinal()) {
      if (verbose >= REPORT) {
        event.list();
        list();
      }
      if (verbose >= NORMAL) {
        ss << "SplitterRF " << i << " iR = " << iRes << " not incoming.";
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Failed to update splitterRF (not final).", ss.str());
      }
      return false;
    }
  }
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "Passed all checks on antennae");
  return true;

}

//--------------------------------------------------------------------------

// Save flavour content of Born state.

void VinciaFSR::saveBornState(int iSys, Event& born) {
  // Initialise.
  resolveBorn[iSys] = false;
  map<int, int> nFlavours;
  for (int i(-6); i<=6; ++i) {
    if (i == 0) nFlavours[21] = 0;
    nFlavours[i] = 0;
  }

  // We want to resolve the Born only when we have a non-QCD coupling in Born.
  int nNonQCD = 0;
  int nIn     = 0;
  for (int i(0); i<partonSystemsPtr->sizeAll(iSys); ++i) {
    Particle* partonPtr = &born[partonSystemsPtr->getAll(iSys, i)];
    if (!partonPtr->isFinal()) ++nIn;
    if (partonPtr->isGluon()) nFlavours[partonPtr->id()]++;
    else if (partonPtr->isQuark()) {
      int idNow = partonPtr->isFinal() ? partonPtr->id() : -partonPtr->id();
      nFlavours[idNow]++;
    }
    else ++nNonQCD;
  }

  // If there are non-QCD partons in the system, resolve Born.
  // (Also do this if there are no incoming partons, when using
  // forceTimeShower to shower off specific user-defined configuration.)
  if (nNonQCD > 0 || nIn == 0) {
    resolveBorn[iSys] = true;
    nFlavsBorn[iSys] = nFlavours;
  }

  // Print information.
  if (verbose >= DEBUG) {
    if (resolveBorn[iSys]) {
      printOut(__METHOD_NAME__, "System " + num2str(iSys,2)
        + " with resolved Born configuration:");
      auto it = nFlavsBorn[iSys].begin();
      for ( ; it != nFlavsBorn[iSys].end(); ++it) {
        if (it->second != 0)
          cout << "      " << num2str(it->first,3) << ": "
               << num2str(it->second,2) << endl;
      }
    } else
      printOut(__METHOD_NAME__,"System " + num2str(iSys,2)
        + " without resolving the Born configuration");
  }
}

//--------------------------------------------------------------------------

// Save flavour content of Born state for trial shower (in merging).

void VinciaFSR::saveBornForTrialShower(Event& born) {
  // Initialise.
  map<int, int> nFlavours;
  for (int i(-6); i<=6; ++i) {
    if (i == 0) nFlavours[21] = 0;
    nFlavours[i] = 0;
  }

  // Index of system we do the trial shower for.
  int iSysTrial = 0;

  // Only resolve Born for resonance systems.
  if (isTrialShowerRes) {
    // We have to find the resonance system.
    // NOTE: by convention (!) this will be the only hadronically decaying one.
    for (int iPtcl(2); iPtcl<born.size(); ++iPtcl) {
      if (!born[iPtcl].isResonance()) continue;
      // Otherwise always increase counter.
      ++iSysTrial;
      // Get indices of daughters.
      int iDaughter1 = born[iPtcl].daughter1();
      int iDaughter2 = born[iPtcl].daughter2();
      // Skip if not quarks or gluons.
      bool dtr1isQorG = iDaughter1 > 0 ?
        (born[iDaughter1].isQuark() || born[iDaughter1].isGluon()) : false;
      bool dtr2isQorG = iDaughter2 > 0 ?
        (born[iDaughter2].isQuark() || born[iDaughter2].isGluon()) : false;
      if (!dtr1isQorG && !dtr2isQorG) continue;
      // Otherwise this is our system and we save the Born info.
      resolveBorn[iSysTrial] = true;
      if (born[iDaughter1].isGluon()) nFlavours[21]++;
      else nFlavours[born[iDaughter1].id()]++;
      if (born[iDaughter2].isGluon()) nFlavours[21]++;
      else nFlavours[born[iDaughter2].id()]++;
      break;
    }
  } else resolveBorn[iSysTrial] = false;
  nFlavsBorn[iSysTrial] = nFlavours;

  // Print information.
  if (verbose >= DEBUG) {
    if (resolveBorn[iSysTrial]) {
      printOut(__METHOD_NAME__, "System " + num2str(iSysTrial,2)
        + " with resolved Born configuration:");
      auto it = nFlavsBorn[iSysTrial].begin();
      for ( ; it != nFlavsBorn[iSysTrial].end(); ++it) {
        if (it->second != 0)
          cout << "      " << num2str(it->first,3) << ": "
               << num2str(it->second,2) << endl;
      }
    } else
      printOut(__METHOD_NAME__,"System " + num2str(iSysTrial,2)
        + " without resolving the Born configuration");
  }
}

//--------------------------------------------------------------------------

// Initialize evolution windows.

void VinciaFSR::initEvolutionWindows(void) {

  evWindowsEmit.clear();
  evWindowsSplit.clear();
  EvolutionWindow window;
  window.alphaSmax = alphaSmax;
  window.mass[1]   = 0.;
  window.mass[2]   = 0.;
  window.mass[3]   = 0.;
  window.mass[4]   = (nFlavZeroMass >= 4) ? 0.0 : particleDataPtr->m0(4);
  window.mass[5]   = (nFlavZeroMass >= 5) ? 0.0 : particleDataPtr->m0(5);
  window.mass[6]   = (nFlavZeroMass >= 6) ? 0.0 : particleDataPtr->m0(6);

  for (int iWindow = 0; iWindow < 4; ++iWindow) {
    //Get minimum boundaries of window.
    double qMinNowEmit = getQ2Window(iWindow, q2CutoffEmit);
    double qMinNowSplit = getQ2Window(iWindow, q2CutoffSplit);

    // Lowest window, use constant trial alphaS for scales below charm mass.
    if (iWindow == 0) {
      window.b0        = 0.;
      window.lambda2   = 0.;
      window.kMu2      = 1.;
      window.runMode   = 0 ;
      window.qMin = qMinNowEmit;
      evWindowsEmit[qMinNowEmit]   = window;
      window.qMin = qMinNowSplit;
      evWindowsSplit[qMinNowSplit] = window;
    } else {
      // Emissions.
      window.runMode = alphaSorder;
      int nFnow = 5;
      if (qMinNowEmit < particleDataPtr->m0(4)) nFnow = 3;
      else if (qMinNowEmit < particleDataPtr->m0(5)) nFnow = 4;
      else if (qMinNowEmit >= particleDataPtr->m0(6)) nFnow = 6;
      window.b0 = (33.0 - 2.0*nFnow) / (12.0 * M_PI);
      double lambdaNow = getLambda(nFnow,aSemitPtr);
      window.lambda2 = (lambdaNow*lambdaNow);
      window.kMu2 = aSkMu2Emit;
      window.qMin = qMinNowEmit;
      evWindowsEmit[qMinNowEmit]=window;
      // Splittings.
      nFnow = 5;
      if (qMinNowSplit < particleDataPtr->m0(4)) nFnow = 3;
      else if (qMinNowSplit < particleDataPtr->m0(5)) nFnow = 4;
      else if (qMinNowSplit >= particleDataPtr->m0(6)) nFnow = 6;
      window.b0 = (33.0 - 2.0*nFnow) / (12.0 * M_PI);
      lambdaNow = getLambda(nFnow,aSsplitPtr);
      window.lambda2 = (lambdaNow*lambdaNow);
      window.kMu2 = aSkMu2Split;
      window.qMin = qMinNowSplit;
      evWindowsSplit[qMinNowSplit]=window;
    }
  }

}

//--------------------------------------------------------------------------

// Return window Q2.

double VinciaFSR::getQ2Window(int iWindow, double q2cutoff) {
  double qMinNow = 0.;
  switch (iWindow) {
  case 0:
    // [cutoff, mc]
    qMinNow = min(sqrt(q2cutoff),particleDataPtr->m0(4));
    break;
  case 1:
    // [mc, mb] with 4-flavour running trial alphaS.
    qMinNow = max(1.0,particleDataPtr->m0(4));
    break;
  case 2:
    // [mb, mt] with 5-flavour running trial alphaS.
    qMinNow = max(3.0,particleDataPtr->m0(5));
    break;
  default:
    // [>mt] with 6-flavour running trial alphaS.
    qMinNow = max(100.0,particleDataPtr->m0(6));
    break;
  }
  return qMinNow;
}

//--------------------------------------------------------------------------

// Return Lambda value.

double VinciaFSR::getLambda(int nFin, AlphaStrong* aSptr) {
  if (nFin <= 3) return 0.;
  else if (nFin == 4) return aSptr->Lambda4();
  else if (nFin == 5) return aSptr->Lambda5();
  else return aSptr->Lambda6();
}

//--------------------------------------------------------------------------

// Method to return renormalisation-scale prefactor.

double VinciaFSR::getkMu2(bool isEmit) {
  double kMu2 = 1.;
  if (isEmit) {
    kMu2 = aSkMu2Emit;
    bool muSoftCorr = false;
    if (useCMW && muSoftCorr) {
      // TODO: generalize.
      double xj =  winnerQCD->getXj();
      double muSoftInvPow = 4;
      double a  = 1./muSoftInvPow;
      kMu2      = pow(xj,a) * kMu2 + (1.-pow(xj,a));
    }
  } else kMu2 = aSkMu2Split;
  return kMu2;
}

//--------------------------------------------------------------------------

// Method to return renormalisation scale. Default scale is kMu *
// evolution scale.

double VinciaFSR::getMu2(bool isEmit) {
  double mu2 = winnerQCD->q2Trial();
  double kMu2 = getkMu2(isEmit);
  mu2 = max(mu2min, mu2freeze + mu2*kMu2);
  return mu2;
}

//--------------------------------------------------------------------------

// Reset (or clear) sizes of all containers.

void VinciaFSR::clearContainers() {
  headroomSav.clear();
  enhanceSav.clear();
  Q2hat.clear();
  isHardSys.clear();
  isResonanceSys.clear();
  doMECsSys.clear();
  polarisedSys.clear();
  stateChangeSys.clear();
  nBranch.clear();
  nBranchFSR.clear();
  nFlavsBorn.clear();
  resolveBorn.clear();
  mSystem.clear();
  nG.clear();
  nQ.clear();
  nLep.clear();
  nGam.clear();
}

//--------------------------------------------------------------------------

// Method to set up QCD antennae, called in prepare.

bool VinciaFSR::setupQCDantennae(int iSys, Event& event) {

  // Sanity check.
  if (partonSystemsPtr == nullptr) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": partonSystems pointer is nullptr!");
    return false;
  }
  // Check that iSys is a valid value.
  if (iSys > partonSystemsPtr->sizeSys()) return false;
  // Check that system contains some final state partons.
  if (partonSystemsPtr->sizeOut(iSys) == 0) return false;

  // Fetch index of resonance if any.
  int iMother     = -1;
  // Colour information of resonance (will remain negative if no resonance).
  int resCol      = -1;
  int resACol     = -1;
  int colPartner  = -1;
  int acolPartner = -1;
  if (partonSystemsPtr->hasInRes(iSys)) {
    iMother = partonSystemsPtr->getInRes(iSys);
    // Check that mother no longer is in system.
    if (event[iMother].status() > 0) return false;
    resCol = event[iMother].col();
    resACol = event[iMother].acol();
  }

  // Map of colour index to decay product (Pythia index).
  map<int, int> coltoDecID;
  // Map of anticolour index to decay product (Pythia index).
  map<int, int> aColtoDecID;
  // List of all decay products.
  vector<int> daughters;

  // Loop over members of current system and get colour information.
  Vec4 pSum(0., 0., 0., 0.);
  for (int iPart = 0; iPart < partonSystemsPtr->sizeOut(iSys); iPart++) {

    // Sum total FS momentum.
    int iOut = partonSystemsPtr->getOut(iSys, iPart);
    pSum += event[iOut].p();
    // Require final state.
    if (!event[iOut].isFinal()) return false;
    // Check colour.
    if (event[iOut].col() !=0 ) {
      // Check if colour partner of a resonance.
      if (event[iOut].col() == resCol) colPartner = iOut;
      // Otherwise save.
      else coltoDecID[event[iOut].col()] = iOut;
    }
    if (event[iOut].acol() !=0 ) {
      // Check if colour partner of a resonance.
      if (event[iOut].acol()==resACol) acolPartner = iOut;
      // Otherwise save.
      else aColtoDecID[event[iOut].acol()] = iOut;
    }
    // Save all.
    if (iOut != colPartner && iOut != acolPartner) daughters.push_back(iOut);
  }
  double mSys = m(pSum);

  // Store total invariant mass of the final state.
  mSystem[iSys] = mSys;

  // Don't do RF or EW/QED branchings in trial showers.
  if (!isTrialShower) {
    // Find any resonance antennae.
    if (colPartner > 0) {
      // Get a copy of daughters.
      vector<int> resSysAll = daughters;
      if (acolPartner != colPartner && acolPartner > 0)
        resSysAll.push_back(acolPartner);

      // Insert col partner and res at front (just convention).
      resSysAll.insert(resSysAll.begin(), colPartner);
      resSysAll.insert(resSysAll.begin(), iMother);
      unsigned int posRes(0), posPartner(1);
      saveEmitterRF(iSys, event, resSysAll, posRes, posPartner, true);
      if (event[colPartner].isGluon())
        saveSplitterRF(iSys, event, resSysAll, posRes, posPartner, true);
    }
    if (acolPartner > 0) {
      // Get a copy of daughters.
      vector<int> resSysAll = daughters;
      if (acolPartner != colPartner && colPartner > 0)
        resSysAll.push_back(colPartner);

      // Insert col partner and res at front (just convention).
      resSysAll.insert(resSysAll.begin(), acolPartner);
      resSysAll.insert(resSysAll.begin(), iMother);
      unsigned int posRes(0), posPartner(1);
      saveEmitterRF(iSys, event, resSysAll, posRes, posPartner, false);
      if (event[acolPartner].isGluon())
        saveSplitterRF(iSys, event, resSysAll, posRes, posPartner, false);
    }
  }

  // Find any f-f that are colour connected, but not directly to a
  // resonance create normal branchers for these.
  for (map<int,int>::iterator it = coltoDecID.begin(); it != coltoDecID.end();
       ++it) {
    int col = it->first;
    int i0  = it->second;
    int i1  = aColtoDecID[col];

    // Exclude antennae that are not FF.
    if (!event[i0].isFinal() || !event[i1].isFinal()) continue;

    // Add to list of QCD gluon emission trial antennae.
    saveEmitterFF(iSys, event, i0, i1);

    // Add gluon-splitting antennae. Default, same 2->3 antenna
    // structure as for gluon emissions.
    if (event[i0].isGluon()) saveSplitterFF(iSys, event, i0, i1, true);
    if (event[i1].isGluon()) saveSplitterFF(iSys, event, i1, i0, false);
  }

  // Deal with any resonance junctions, n.b. assumes that these are
  // colour junctions not anticolour.
  hasResJunction[iSys] = false;
  if (isResonanceSys[iSys] && resCol > 0 && colPartner >0) {
    // Loop over junctions.
    for (int iJun = 0; iJun < event.sizeJunction(); ++iJun) {
      // Loop over ends.
      for (int iLeg = 0; iLeg < 3; ++iLeg) {
        if (event.endColJunction(iJun, iLeg) == resCol) {
          // Found a resonance junction.
          hasResJunction[iSys] = true;
          junctionInfo[iSys].iJunction=iJun;
          junctionInfo[iSys].iEndCol=iLeg;
          junctionInfo[iSys].iEndColTag=resCol;
          junctionInfo[iSys].iEndQuark=colPartner;
          junctionInfo[iSys].colours.clear();
          junctionInfo[iSys].colours.push_back(resCol);

          // In context of matching might have many partons in systems
          // already.
          while (!event[junctionInfo[iSys].iEndQuark].isQuark()) {
            int colNow = event[junctionInfo[iSys].iEndQuark].acol();
            if (aColtoDecID.find(colNow) != aColtoDecID.end()) {
              int newPart = coltoDecID[colNow];
              junctionInfo[iSys].colours.push_back(colNow);
              junctionInfo[iSys].iEndQuark=newPart;
              junctionInfo[iSys].iEndColTag=colNow;
            } else {
              infoPtr->errorMsg("Error in "+__METHOD_NAME__
                +": Resonance involved in junction that cannot be traced");
              hasResJunction[iSys] = false;
              break;
            }
          }
          if (event[junctionInfo[iSys].iEndQuark].col() == 0 ||
            !event[junctionInfo[iSys].iEndQuark].isFinal()) {
            infoPtr->errorMsg("Error in "+__METHOD_NAME__
              +": Failed to find end quark in resonance junction");
            hasResJunction[iSys] = false;
            break;
          }
        }
      }
    }
  }

  // Count up number of gluons, quarks, and photons.
  nG[iSys]       = 0;
  nQ[iSys]       = 0;
  nGam[iSys]     = 0;
  nLep[iSys]     = 0;
  for (int i = 0; i < partonSystemsPtr->sizeAll(iSys); ++i) {
    Particle* partonPtr = &event[partonSystemsPtr->getAll(iSys, i)];
    if (partonPtr->isGluon()) nG[iSys]++;
    else if (partonPtr->isQuark()) nQ[iSys]++;
    else if (abs(partonPtr->id()) == 22) nGam[iSys]++;
    else if (partonPtr->isLepton()) nLep[iSys]++;
  }

  // Sanity checks.
  if (verbose >= DEBUG) {
    if (emittersRF.size() + splittersRF.size() +
      emittersFF.size() + splittersFF.size() <= 0)
      printOut(__METHOD_NAME__, "did not find any QCD branchers");
    list();
    printLookup();
  }
  return true;

}

//--------------------------------------------------------------------------

// Set starting scale of shower (power vs wimpy) for system iSys.

void VinciaFSR::setStartScale(int iSys, Event& event) {

  // Set nIn: 1->n or 2->n.
  int nIn = 0;
  if (isResonanceSys[iSys]) nIn = 1;
  else if (partonSystemsPtr->hasInAB(iSys)) nIn = 2;

  // Set FSR starting scale of this system (can be different from qFac).
  // Resonance decay systems always start at Q2 = m2..
  if (isResonanceSys[iSys]) {
    Q2hat[iSys] = pow2(mSystem[iSys]);
    return;

  // Hard system: start at phase-space maximum or factorisation scale.
  } else if (isHardSys[iSys]) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Setting FSR starting scale for hard system");
    // pTmaxMatch = 1 : always start at QF (modulo kFudge).
    if (pTmaxMatch == 1) Q2hat[iSys] = pT2maxFudge * infoPtr->Q2Fac();
    // pTmaxMatch = 2 : always start at eCM.
    else if (pTmaxMatch == 2) Q2hat[iSys] = m2BeamsSav;
    // Else check if this event has final-state jets or photons.
    else {
      bool hasRad = false;
      for (int i = 0; i < partonSystemsPtr->sizeOut(iSys); ++i) {
        int idAbs = event[partonSystemsPtr->getOut(iSys, i)].idAbs();
        if (idAbs <= 5 || idAbs == 21 || idAbs == 22) hasRad = true;
        if (idAbs == 6 && nGluonToQuark == 6) hasRad = true;
        if (hasRad) break;
      }
      // If no QCD/QED partons detected, allow to go to phase-space maximum.
      if (hasRad) Q2hat[iSys] = pT2maxFudge * infoPtr->Q2Fac();
      else Q2hat[iSys] = m2BeamsSav;
    }
  } else if (nIn == 2) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Setting FSR starting scale of MPI system");
    // Set starting scale for MPI systems: min of incoming parton
    // scales. Find positions of incoming colliding partons.
    int in1 = partonSystemsPtr->getInA(iSys);
    int in2 = partonSystemsPtr->getInB(iSys);
    Q2hat[iSys] = pT2maxFudgeMPI
      * pow2(min(event[in1].scale(),event[in2].scale()));
  } else {
    // Assume hadron -> partons decay. Starting scale = mSystem.
    Q2hat[iSys] = pow2(mSystem[iSys]);
  }

}

//--------------------------------------------------------------------------

// Auxiliary methods to generate trial scales for various shower
// components.

bool VinciaFSR::q2NextEmitResQCD(const double q2Begin, const double q2End) {
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  double q2EndNow = max(q2End, q2CutoffEmit);
  bool gen = q2NextQCD<BrancherEmitRF>(emittersRF, evWindowsEmit,
    evTypeEmit, q2Begin, q2EndNow, true);
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "end", dashLen);
  return gen;
}

bool VinciaFSR::q2NextSplitResQCD(const double q2Begin, const double q2End) {
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  double q2EndNow = max(q2End, q2CutoffSplit);
  bool gen = q2NextQCD<BrancherSplitRF>(splittersRF, evWindowsSplit,
    evTypeSplit, q2Begin, q2EndNow, false);
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__,"end", dashLen);
  return gen;
}

bool VinciaFSR::q2NextEmitQCD(const double q2Begin, const double q2End) {
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  double q2EndNow = max(q2End, q2CutoffEmit);
  bool gen = q2NextQCD<BrancherEmitFF>(emittersFF, evWindowsEmit, evTypeEmit,
    q2Begin, q2EndNow, true);
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__,"end", dashLen);
  return gen;
}

bool VinciaFSR::q2NextSplitQCD(const double q2Begin, const double q2End) {
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  double q2EndNow = max(q2End, q2CutoffSplit);
  bool gen = q2NextQCD<BrancherSplitFF>(splittersFF, evWindowsSplit,
    evTypeSplit, q2Begin, q2EndNow, false);
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__,"end", dashLen);
  return gen;
}

//--------------------------------------------------------------------------

// Return the Q2 for the next branching.

template <class Brancher> bool VinciaFSR::q2NextQCD(
  vector<Brancher>& brancherVec, const map<double, EvolutionWindow> &evWindows,
  const int evType, const double q2Begin, const double q2End, bool isEmit) {

  // Sanity check
  if (verbose  >= DEBUG) {
    stringstream ss;
    ss << "qBegin = " << num2str(sqrt(q2Begin))
       << " GeV, with "<<brancherVec.size()<<" branchers.";
    printOut(__METHOD_NAME__, ss.str());
  }
  if (q2Begin <= q2End) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "q2Begin below cutoff. Nothing to do.");
    return true;
  } else if (!isEmit && nGluonToQuark == 0) return true;

  // Loop over resonance antennae.
  unsigned int iAnt = 0;
  for (typename vector<Brancher>::iterator ibrancher = brancherVec.begin();
       ibrancher!=brancherVec.end(); ++ibrancher) {
    iAnt++;
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Brancher " << iAnt <<" / " << brancherVec.size();
      printOut(__METHOD_NAME__,ss.str());
    }

    // Skip antennae outside system if doing resonance shower.
    // (Current resonance system is always the last one.)
    if (nRecurseResDec >= 1 &&
      ibrancher->system() != partonSystemsPtr->sizeSys() -1) continue;

    // Check if there is any phase space left for current antenna.
    double Q2MaxNow = min(q2Begin, ibrancher->getQ2Max(evType));
    if (Q2MaxNow < q2End) {
      if (verbose >= DEBUG) printOut(__METHOD_NAME__,
          "No phase space left for current brancher, continuing.");
      continue;
    }

    // Check if a saved trial exists for this brancher.
    double q2Next = 0.;
    if (ibrancher->hasTrial()) {
      q2Next = ibrancher->q2Trial();
      if (verbose >= DEBUG) {
        stringstream ss;
        ss << "Retrieving saved trial Q=" << sqrt(q2Next);
        printOut(__METHOD_NAME__, ss.str());
      }
    // Else generate new trial scale.
    } else {
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, "Generating new trial");

      // Fetch system and colour factor for current brancher.
      int iSys   = ibrancher->system();
      double colFac = getAntFunPtr(ibrancher->antFunTypePhys())->chargeFac();
      if (verbose >= DEBUG) {
        stringstream ss;
        ss << "Starting shower for current brancher at Q=" << sqrt(Q2MaxNow);
        printOut(__METHOD_NAME__, ss.str());
      }

      // Impose evolution windows (for alphaS running); fetch the
      // current window.
      map<double, EvolutionWindow>::const_iterator
        it = evWindows.lower_bound(sqrt(Q2MaxNow));
      // Cast as a reverse iterator to go downwards in q2.
      map<double, EvolutionWindow>::const_reverse_iterator itWindowNow(it);

      // Go through regions.
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, "Looping over Q2 windows...");
      while(itWindowNow != evWindows.rend()) {

        // Bottom of current window.
        double Q2MinWindow = pow2(itWindowNow->first);
        const EvolutionWindow* windowPtr = &(itWindowNow->second);

        // Set headroom and enhancement factors.
        vector<double> headroomVec = getHeadroom(iSys, isEmit, Q2MaxNow);
        vector<double> enhanceVec = getEnhance(iSys, isEmit, Q2MaxNow);
        double Q2NextWindow = ibrancher->genQ2(evType, Q2MaxNow, rndmPtr,
          infoPtr, windowPtr, colFac, headroomVec, enhanceVec, verbose);
        if (Q2NextWindow < 0.) {
          infoPtr->setAbortPartonLevel(true);
          return false;
        }
        if (verbose >= DEBUG) {
          stringstream ss;
          ss << "Generated QNextWindow = " << sqrt(Q2NextWindow)
             << " (QMinWindow = " << itWindowNow->first << " )";
          printOut(__METHOD_NAME__, ss.str());
        }

        // Check if Q2next is in the current window.
        if (Q2NextWindow > Q2MinWindow || Q2NextWindow <= 0.) {
          q2Next=Q2NextWindow;
          break;
        } else {
          if (verbose >= DEBUG) printOut(__METHOD_NAME__,
              "QNext below window threshold. Continuing to next window.");
        }
        // Else go straight to next window.
        Q2MaxNow = Q2MinWindow;
        // Increment reverse iterator (go down in scale).
        itWindowNow++;
      } // End loop over evolution windows.
      if (verbose >= DEBUG && itWindowNow == evWindows.rend())
        printOut(__METHOD_NAME__, "Out of windows. Continuing to "
          "next brancher.");
    } // End generate new trial for this antenna.

    // Check for winning condition.
    if (q2Next > q2WinSav && q2Next > q2End) {
      q2WinSav = q2Next;
      winnerQCD = &(*ibrancher);
    }
  } // End loop over QCD antennae.

  // Done.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss<<"qWin = "<< sqrt(q2WinSav);
    printOut(__METHOD_NAME__,ss.str());
  }
  return true;

}

//--------------------------------------------------------------------------

// Perform QCD branching.

bool VinciaFSR::branchQCD(Event& event) {

  // Check if we are supposed to do anything.
  if (!(doFF || doRF)) return false;

  // Verbose output
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Profiling.
  if (verbose >= REPORT) diagnosticsPtr->start(__METHOD_NAME__);

  iSysWin                 = winnerQCD->system();
  stateChangeLast         = false;
  stateChangeSys[iSysWin] = false;
  iNewSav = 0;

  // Mark this trial as used so we do not risk reusing it.
  winnerQCD->needsNewTrial();

  // If this is a resonance shower, branching must be in the parton system
  // that was added last.
  if (nRecurseResDec >= 1 && iSysWin != partonSystemsPtr->sizeSys()-1 ) {
    infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": vetoing trial "
      "branching outside resonance shower system");
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"veto(not in res system)");
    return false;
  }

  // Find out which branching type we are doing
  antFunTypeWin   = winnerQCD->antFunTypePhys();

  // Decide whether to accept the trial.
  // Store new particles in pNew if keeping.
  if (!acceptTrial(event)) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,
      "Trial rejected (failed acceptTrial)");
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"veto(acceptTrial)");
    return false;
  }

  // Update event record, add new daughters.
  // Make a backup copy of the event (may want to veto!)
  // Make a copy of junction info.
  Event oldEvent = event;
  ResJunctionInfo junctionInfoCopy;
  const int sizeOld = event.size();
  if (hasResJunction[iSysWin]) junctionInfoCopy=junctionInfo[iSysWin];
  if (!updateEvent(event,junctionInfoCopy)) {
    if (verbose >= REPORT) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Failed to update event");
      diagnosticsPtr->stop(__METHOD_NAME__,"veto(updateEvent)");
    }
    // restore backup event
    event = oldEvent;
    return false;
  }

  // Allow veto by Userhooks.
  // Possibility to allow user veto of emission step.
  if (canVetoEmission) {
    if (userHooksPtr->doVetoFSREmission(sizeOld, event,
        iSysWin, isResonanceSys[iSysWin])) {
      if (verbose >= REPORT) printOut(__METHOD_NAME__,
        "Trial rejected (failed UserHooks::doVetoFSREmission)");
      // restore backup event
      event = oldEvent;
      return false;
    }
  }

  // Everything accepted. Update junctionInfo and partonSystems.
  if (hasResJunction[iSysWin]) junctionInfo[iSysWin] = junctionInfoCopy;
  updatePartonSystems();

  // Update antennae.
  if (!updateAntennae(event)) {
    if (verbose >= REPORT)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Failed to update branchers");
    //Something went wrong.
    infoPtr->setAbortPartonLevel(true);
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"veto(updateAntennae)");
    return false;
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // Count the number of branchings in the system.
  nBranch[iSysWin]++;
  nBranchFSR[iSysWin]++;

  // Check the event after each branching.
  if (verbose >= REPORT && !vinComPtr->showerChecks(event, false)) {
    infoPtr->errorMsg("Error in"+__METHOD_NAME__+": Failed shower checks");
    infoPtr->setAbortPartonLevel(true);
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"veto(showerChecks)");
    return false;
  }

  // Book-keeping for MECs.
  if (doMECsSys[iSysWin]) {
    // Communicate to MECs class that we succesfully branched.
    mecsPtr->hasBranched(iSysWin);
    // Decide if we should be doing ME corrections for next order.
    doMECsSys[iSysWin] = mecsPtr->doMEC(iSysWin, nBranch[iSysWin]+1);
  }

  // When merging, communicate to MergingHooks whether the event may be vetoed
  // due to this branching.
  if (doMerging && !isTrialShower) {
    // We only want to veto the event based on the first branching.
    //TODO: in principle, we could veto later emissions here as well.
    if (nBranch[iSysWin] > 1)
      mergingHooksPtr->doIgnoreStep(true);
  }

  // Force stop by user? (Debugging only)
  if (allowforceQuit) {
    if (nBranchFSR[iSysWin] >= nBranchQuit && nBranchQuit >0) {
      forceQuit=true;
      if (verbose >= REPORT) {
        stringstream ss;
        ss<<"User forced quit after "<<nBranchQuit<<" emissions.";
        printOut(__METHOD_NAME__,ss.str());
      }
    }
  }

  // Profiling.
  if (verbose >= REPORT) diagnosticsPtr->stop(__METHOD_NAME__,"accept");

  // Verbose output
  if (verbose >=DEBUG) printOut(__METHOD_NAME__,"end", dashLen);

  // Done/
  return true;

}

//--------------------------------------------------------------------------

// Perform an EW branching.

bool VinciaFSR::branchEW(Event& event) {

  // EW veto step.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  int sizeOld  = event.size();
  double pTnow = sqrt(q2WinSav);
  if (winnerEW->acceptTrial(event)) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "EW trial accepted. About to update.");

    // Make a backup of the event to update (user may want to veto!).
    Event oldEvent = event;

    // Update event after branching accepted.
    winnerEW->updateEvent(event);

    bool isInitial = winnerEW->lastIsInitial();
    iSysWin = winnerEW->sysWin();

    if (canVetoISREmission && isInitial) {
      if (userHooksPtr->doVetoISREmission(sizeOld, event, iSysWin)) {
        if (verbose >= REPORT) printOut(__METHOD_NAME__,
          "Trial rejected (failed UserHooks::doVetoISREmission)");
        event = oldEvent;
        return false;
      }
    }
    else if (canVetoEmission && !isInitial) {
      bool isResonanceDecay = winnerEW->lastIsResonanceDecay();
      if (userHooksPtr->doVetoFSREmission(sizeOld, event,
          iSysWin, isResonanceDecay)) {
        if (verbose >= REPORT) printOut(__METHOD_NAME__,
          "Trial rejected (failed UserHooks::doVetoFSREmission)");
        event = oldEvent;
        return false;
      }
    }

    // Everything accepted.
    // If this was a resonance decay, do resonance shower.
    if (winnerEW->lastIsResonanceDecay()) {
      // Was this a 1->2 or 2->3 resonance decay?
      // 2->3 resonance decays are modeled in two steps, first IK -> I'K',
      // where the decaying resonance in general acquires a higher virtuality,
      // followed by I' -> i j. This is marked by I' being assigned status 57.
      bool is2to3 = (abs(event[sizeOld].status()) == 57);
      if ( is2to3 ) {
        // Update upstream parton system to step to I'K'.
        for (int iOut = 0; iOut < partonSystemsPtr->sizeOut(iSysWin); ++iOut) {
          int iOld = partonSystemsPtr->getOut(iSysWin,iOut);
          if (event[iOld].isFinal()) continue;
          int iNew = event[iOld].iBotCopyId();
          if (iNew != iOld) partonSystemsPtr->replace(iSysWin,iOld,iNew);
        }
      }
      int iMother = (is2to3) ? sizeOld : event[sizeOld].mother1();
      vector<int> iPosRes = { iMother };
      // Now prepare to do resonance shower.
      Event dummy;
      // Do not do EW shower in current system during resonance shower.
      winnerEW->clear(iSysWin);
      // Do resonance shower (includes updates post-shower).
      if (!resonanceShower(dummy, event, iPosRes, pTnow)) {
        infoPtr->errorMsg("Error in "+__METHOD_NAME__+
          ": resonanceShower() returned false. Aborting.");
        event = oldEvent;
        infoPtr->setAbortPartonLevel(true);
        return false;
      }
    }
    // Else just do updates.
    else {
      // Update parton systems.
      winnerEW->updatePartonSystems(event);
      // Update EW radiators.
      winnerEW->update(event, iSysWin);
      // Update QCD radiators.
      if (!updateAfterEW(event, sizeOld)) {
        infoPtr->errorMsg("Error in "+__METHOD_NAME__+
          ": failed to update QCD branchers. Aborting.");
        event = oldEvent;
        infoPtr->setAbortPartonLevel(true);
        return false;
      }
    }

    // Check PartonSystems in REPORT mode.
    if (verbose >= REPORT) {
      if (partonSystemsPtr->hasInAB(iSysWin)) {
        int inA = partonSystemsPtr->getInA(iSysWin);
        int inB = partonSystemsPtr->getInB(iSysWin);
        if (inA <= 0 || inB <= 0 ) {
          stringstream ss;
          ss << "iSysWin = "<<iSysWin << " non-positive. inA = "<< inA
             << " inB = " << inB;
          infoPtr->errorMsg("Error in "+__METHOD_NAME__
            +": Non-positive incoming parton.", ss.str());
          infoPtr->setAbortPartonLevel(true);
          return false;
        } else if (event[inA].mother1() > 2 || event[inB].mother1() > 2) {
          stringstream ss;
          ss << "iSysWin = "<<iSysWin;
          infoPtr->errorMsg("Error in "+__METHOD_NAME__
            +": Failed to update incoming particles after QED branching.",
            ss.str());
          infoPtr->setAbortPartonLevel(true);
          return false;
        }
      }
    }

  }
  // Else EW trial failed.
  else {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__, "EW trial failed");
    return false;
  }

  // Done.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return true;

}

//--------------------------------------------------------------------------

// Update systems of QCD antennae after an EW/ QED branching.

bool VinciaFSR::updateAfterEW(Event& event, int sizeOld) {
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  iSysWin = winnerEW->sysWin();

  // Create colour-anticolour map for post-branching partons.
  map<int,int> iOfCol, iOfAcol;
  // Also check for coloured partons that were created in the
  // splitting, i.e. with status 51, used to create new emission
  // antennae below.
  vector<int>  status51coloured;
  vector<pair<int,int> > colouredrecoilers;
  for (int i = sizeOld; i < event.size(); ++i) {
    int col  = event[i].col();
    int acol = event[i].acol();
    if (col != 0) iOfCol[col] = i;
    if (acol != 0) iOfAcol[acol] = i;
    // Check which were "created" (as opposed to recoiling) - to see
    // if we need to create splitter.
    if (event[i].colType() != 0 && event[i].status() == 51)
      status51coloured.push_back(i);
    else if (event[i].colType() != 0 &&
             (event[i].status() == 52 || event[i].status() == 43 ||
              event[i].status() == 44)) {
      int moth = event[i].mother1();
      if (moth > 0) colouredrecoilers.push_back(make_pair(moth, i ));
    }
  }

  if (status51coloured.size() == 2) {
    int i1    = status51coloured[0];
    int i2    = status51coloured[1];
    if (event[i1].colType() < 0) {
      i1 = status51coloured[1];
      i2 = status51coloured[0];
    }
    // If this was a splitting to coloured partons, create new
    // emission antenna. Create a QCD emission antenna between the two
    // status-51 partons.
    if (winnerEW->lastIsSplitting() && event[i1].col() != 0 &&
      event[i1].col() == event[i2].acol()) {
      saveEmitterFF(iSysWin,event,i1,i2);
      // In principle even allow for h->gg type.
      if (event[i2].col() != 0 && event[i2].col() == event[i1].acol())
        saveEmitterFF(iSysWin,event,i2,i1);
    }
    // Need to update existing QCD antennae.
    else {
      int moth1 = event[i1].mother1();
      colouredrecoilers.push_back(make_pair(moth1, i1));
      int moth2 = event[i2].mother1();
      colouredrecoilers.push_back(make_pair(moth2, i2));
      if (event[moth1].col() == event[moth2].acol())
        updateEmitterFF(event,moth1,moth2,i1,i2);
    }
  } else if (status51coloured.size() == 1) {
    int i1    = status51coloured[0];
    int moth1 = event[i1].mother1();
    // Check for "special" entries for intermediate resonance decays.
    if (event[moth1].statusAbs() == 57) moth1 = event[moth1].mother1();
    colouredrecoilers.push_back(make_pair(moth1, i1));
  } else if (status51coloured.size() > 2) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Too many status 51 particles");
    infoPtr->setAbortPartonLevel(true);
    return false;
  }

  // Reset any emission antennae involving quarks that have recoiled.
  for (vector<pair<int,int> >::iterator it = colouredrecoilers.begin();
      it!= colouredrecoilers.end(); ++it) {
    int recOld = it->first;
    int recNew = it->second;
    updateEmittersFF(event,recOld,recNew);
    updateSplittersFF(event,recOld,recNew);
  }

  // Update resonance antennae.
  if (isResonanceSys[iSysWin]) {
    if (!updateEmittersRF(iSysWin, event,
        partonSystemsPtr->getInRes(iSysWin))) {
      if (verbose >= NORMAL)
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Failed updateEmittersRF");
      return false;
    }
  }

  // Check all FF branchers are final-state.
  if (!check(iSysWin, event)) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Failed to update branchers");
    list();
    if (verbose >= DEBUG) printLookup();
    infoPtr->setAbortPartonLevel(true);
    return false;
  }

  // Update list of resonance decays if needed.
  for (size_t iHard=0; iHard<pTresDecSav.size(); ++iHard) {
    if (pTresDecSav[iHard] <= 0.) continue;
    int iBef = iPosBefSav[iHard];
    int iAft = event[iBef].iBotCopyId();
    if (!event[iAft].isFinal()) pTresDecSav[iHard] = 0.;
  }

  // Now check if end quark has changed.
  if (hasResJunction[iSysWin]) {
    int iEndQuark = junctionInfo[iSysWin].iEndQuark;
    if (!event[iEndQuark].isFinal()) {
      int d1 = event[iEndQuark].daughter1();
      int d2 = event[iEndQuark].daughter2();
      if (event[d1].isQuark() && event[d1].col() > 0)
        junctionInfo[iSysWin].iEndQuark = d1;
      else if (event[d2].isQuark() && event[d2].col() > 0)
        junctionInfo[iSysWin].iEndQuark = d2;
      else {
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Couldn't update junction information");
        return false;
      }
    }
  }

  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "end", dashLen);

  return true;

}

//--------------------------------------------------------------------------

// Perform an early antenna rejection.

bool VinciaFSR::rejectEarly(AntennaFunction* &antFunPtr, bool doMEC) {

  bool reject = true;
  if (winnerQCD->getBranchType() == BranchType::Void) {
    if (verbose >= REPORT)
      printOut(__METHOD_NAME__, "Warning: could not identify branching type");
    return reject;
  }

  // If enhancement was applied to the trial function but branching is
  // below enhancement cutoff, we do an early accept/reject here with
  // probability trial/enhanced-trial to get back to unenhanced trial
  // probability.

  // Trials only enhanced for enhanceFac > 1.
  if (winnerQCD->enhanceFac() > 1.0 &&
      winnerQCD->q2Trial() <= pow2(enhanceCutoff)) {
    if (rndmPtr->flat() > 1./winnerQCD->enhanceFac()) {
      if (verbose >= DEBUG) printOut(__METHOD_NAME__,
          "Trial rejected (enhance applied below enhanceCutoff)");
      return reject;
    }
    // If passed, save that enhancement factor has now been canceled.
    winnerQCD->resetEnhanceFac(1.0);
  }

  // Generate post-branching invariants. Can check some vetos already
  // at this level, without full kinematics.
  vector<double> invariants;
  if (!winnerQCD->genInvariants(invariants, rndmPtr, verbose, infoPtr)) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Trial rejected (failed genInvariants)");
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"veto(genInvariants)");
    return reject;
  }

  // Impose g->QQ mass thresholds for flavours treated as massless.
  if (antFunTypeWin == GXsplitFF && winnerQCD->idNew() <= nFlavZeroMass) {
    // m2(qq) > 4m2q => s(qq) > 2m2q, but allow for larger factor.
    // Factor 4 roughly matches n(g->bb) for massive b quarks.
    double facM = 4;
    if (invariants[1] < facM*pow2(particleDataPtr->m0(winnerQCD->idNew()))) {
      if (verbose >= REPORT)
        diagnosticsPtr->stop(__METHOD_NAME__,"veto(mQQ)");
      return reject;
    }
  }

  // Compute physical antenna function (summed over possible helicities).
  double antPhys = getAntFunPhys(antFunPtr);
  // Get accept probability.
  pAccept[0] = pAcceptCalc(antPhys);

  // If doing ME corrections, don't allow to reject yet.
  if (!doMEC) {
    // Check if rejecting the trial.
    double R = rndmPtr->flat();
    bool doVeto = ( R > pAccept[0]) ? true : false;

    if (doVeto) {
      // TODO: Note, here we want to put a call to something which computes
      // uncertainty variations for pure-shower branchings. We also
      // may want to take into account if there was an enhancement
      // applied to this branching.
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, "Trial rejected (failed R<pAccept)");
      if (verbose >= REPORT)
        diagnosticsPtr->stop(__METHOD_NAME__,"veto(pAccept)");
      // Reweighting to account for enhancement factor (reject).
      double enhanceFac = winnerQCD->enhanceFac();
      if ( enhanceFac != 1.0)
        weightsPtr->scaleWeightEnhanceReject(pAccept[0],enhanceFac);
      return reject;
    }

    // Set accept probability to 1, so no later opportunity to reject
    // unles we apply an enhancement factor.
    pAccept[0] = 1.;
  }

  //Trial accepted so far, n.b. proper acccept/reject condition later.
  return false;

}

//--------------------------------------------------------------------------

// Compute physica antenna function.
double VinciaFSR::getAntFunPhys(AntennaFunction* &antFunPtr) {

  // Set antenna function pointer and check if this antenna is "on".
  antFunPtr = getAntFunPtr(antFunTypeWin);
  if (antFunPtr->chargeFac() <= 0.) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Trial rejected (chargeFac <= 0)");
    return 0.;
  }

  bool isSplit = (antFunTypeWin == GXsplitFF || antFunTypeWin == XGsplitRF);

  // AlphaS, impose default choice. Can differ slighly from trial even
  // when running inside trial integral, due to flavor
  // thresholds. Here, alphaS(mu) is returned directly, with the
  // number of flavors active at mu, whereas the number of flavors in
  // the trial integral is controlled by the value of the trial scale.
  double alphaSNow = alphaSmax;
  if (alphaSorder >= 1) {
    double mu2 = getMu2(!isSplit);
    AlphaStrong* alphaSptr = (isSplit) ? aSsplitPtr : aSemitPtr;
    alphaSNow = min(alphaSmax, alphaSptr->alphaS(mu2));
  }

  // Compute physical antenna function (summed over final state
  // helicities). Note, physical antenna function can have swapped
  // labels (eg GQ -> GGQ).
  vector<double> mPost = winnerQCD->getmPostVec();
  vector<double> invariants = winnerQCD->getInvariants();
  unsigned int nPre = winnerQCD->iVec().size();
  vector<int> hPre = ( helicityShower && polarisedSys[iSysWin] ) ?
    winnerQCD->hVec() : vector<int>(nPre, 9);
  vector<int> hPost(nPre+1,9);
  double antPhys = antFunPtr->antFun(invariants, mPost, hPre, hPost);
  if (antPhys < 0.) {
    if (verbose >= REPORT) infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Negative Antenna Function.", num2str(antFunTypeWin));
    return 0.;
  }

  antPhys *= antFunPtr->chargeFac();

  antPhys*=alphaSNow;

  return antPhys;
}

//--------------------------------------------------------------------------

// Calculate acceptance probability.

double VinciaFSR::pAcceptCalc(double antPhys) {
  double prob = winnerQCD->pAccept(antPhys,infoPtr,verbose);
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__,"Shower pAccept = " + num2str(prob));
  return prob;
}

//--------------------------------------------------------------------------

// Generate the full kinematics.

bool VinciaFSR::genFullKinematics(int kineMap, Event event,
  vector<Vec4> &pPost) {

  // Generate branching kinematics, starting from antenna parents.
  vector<Vec4> pPre;
  vector<int> iPre          = winnerQCD->iVec();
  int nPre                  = iPre.size();
  int nPost                 = winnerQCD->iVec().size() + 1;
  vector<double> invariants = winnerQCD->getInvariants();
  vector<double> mPost      = winnerQCD->getmPostVec();
  bool isRF                 = winnerQCD->posR() >= 0;
  double phi                = 2 * M_PI * rndmPtr->flat();
  for (int i = 0; i < nPre; ++i) pPre.push_back(event[iPre[i]].p());

  // Special case for resonance decay.
  if (isRF) {
    if (!vinComPtr->map2toNRF(pPost, pPre, winnerQCD->posR(),
          winnerQCD->posF(), invariants,phi,mPost)) {
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, "Trial rejected (failed map2toNRF)");
      return false;
    }
  } else {
    // 2->3 kinematics.
    if (nPre == 2 && nPost == 3) {
      if (!vinComPtr->map2to3FF(pPost, pPre, kineMap, invariants, phi,
          mPost)) {
        if (verbose >=  DEBUG)
          printOut(__METHOD_NAME__, "Trial rejected (failed map2to3)");
        return false;
      }
    // 2->4 kinematics
    } else if (nPre == 2 && nPost == 4) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": 2->4 kinematics map not implemented yet");
      return false;
    // 3->4 kinematics
    } else if (nPre == 3 && nPost == 4) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": 3->4 kinematics map not implemented yet");
      return false;
    }
  }
  return true;

}

//--------------------------------------------------------------------------

// Check if a trial is accepted.

bool VinciaFSR::acceptTrial(Event& event) {

  // Diagnostics.
  if (verbose >= REPORT) diagnosticsPtr->start(__METHOD_NAME__);

  bool doMEC  = doMECsSys[iSysWin];
  AntennaFunction* antFunPtr;

  // Check to see if we veto early before generating full kinematics,
  // i.e. just based on invariants.
  if (rejectEarly(antFunPtr,doMEC)) {
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"veto(rejectEarly)");
    return false;
  }
  if (!getNewParticles(event,antFunPtr,pNew)) {
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"veto(getNewParticles)");
    return false;
  }

  // For sector shower veto branching here if outside of sector.
  vector<Particle> stateNew;
  VinciaClustering minClus;
  if (sectorShower) {
    // If we have a resonance, its momentum is not stored in pNew, so we
    // create a new vector with all post-branching particles, including the
    // resonance, although being unchanged.
    vector<Particle> ptclsPost;
    vector<int> iOld;
    // Check if we have a resonance.
    int iPosRes = winnerQCD->posR();
    if (iPosRes != -1) {
      ptclsPost.push_back(event[winnerQCD->iVec(iPosRes)]);
      iOld.push_back(winnerQCD->iVec(iPosRes));
      for (auto& i : winnerQCD->mothers2daughters)
        iOld.push_back(i.first);
    }
    else iOld = winnerQCD->iVec();
    for (auto& p : pNew) ptclsPost.push_back(p);

    // Get tentative post-branching state.
    stateNew = vinComPtr->makeParticleList(iSysWin, event, ptclsPost, iOld);

    // Save clustering and compute sector resolution for it.
    VinciaClustering thisClus;
    thisClus.setChildren(ptclsPost,0,1,2);
    thisClus.setMothers(winnerQCD->id0(),winnerQCD->id1());
    thisClus.setAntenna(true,antFunTypeWin);
    thisClus.initInvariantAndMassVecs();
    double q2sectorThis = resolutionPtr->q2sector(thisClus);
    // Sanity check.
    if (q2sectorThis < 0.) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,
        "Negative sector resolution");
      return false;
    }
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Branching has sector resolution " << q2sectorThis;
      printOut(__METHOD_NAME__, ss.str());
    }

    // Check sector veto.
    minClus = resolutionPtr->findSector(stateNew, nFlavsBorn[iSysWin]);
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Minimal clustering has sector resolution " << minClus.Q2res;
      printOut(__METHOD_NAME__, ss.str());
    }
    bool isVetoed = resolutionPtr->sectorVeto(minClus, thisClus);
    // Failed sector veto.
    if (isVetoed) {
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, "Branching rejected (outside of sector)");
      if (verbose >= REPORT)
        diagnosticsPtr->stop(__METHOD_NAME__,"veto(sector)");
      return false;
    }
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Branching inside of sector");
  }

  // Check if phase space is closed for getting rid of heavy quarks.
  vector<Particle> stateOld;
  if (!isrPtr->checkHeavyQuarkPhaseSpace(stateOld,iSysWin)) {
    stateOld = vinComPtr->makeParticleList(iSysWin, event);
    if (verbose >= REPORT) {
      printOut(__METHOD_NAME__, "Trial rejected (failed "
        "checkHeavyQuarkPhaseSpace)");
      diagnosticsPtr->stop(__METHOD_NAME__,"veto(HQPS)");
    }
    return false;
  }

  // Matrix element corrections.
  // Note: currently only implemented for the sector shower.
  double pMEC = 1.;
  if (doMEC) {
    // DEBUG output.
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Trying matrix element correction for system "
         << iSysWin << " (" << nBranch[iSysWin]+1 << ". branching).";
      printOut(__METHOD_NAME__, ss.str());
    }

    pMEC = getMEC(iSysWin, event, stateNew, minClus);
  }
  pAccept[0] *= pMEC;

  // Count number of shower-type partons (for diagnostics and headroom
  // factors).
  int nQbef(0), nGbef(0), nBbef(0);
  for (int i = 0; i < partonSystemsPtr->sizeOut(iSysWin); ++i) {
    if (event[partonSystemsPtr->getOut(iSysWin,i)].id() == 21) ++nGbef;
    else if (event[partonSystemsPtr->getOut(iSysWin,i)].idAbs() <= 4) ++nQbef;
    else if (event[partonSystemsPtr->getOut(iSysWin,i)].idAbs() == 5) ++nBbef;
  }

  // Print MC violations.
  if (doMEC && verbose >= DEBUG) {
    stringstream ss;
    ss << " MEC pAccept = " << pAccept[0];
    printOut(__METHOD_NAME__, ss.str());
  }
  if (verbose >= REPORT) {
    bool violation  = (pAccept[0] > 1.0 + NANO);
    bool negPaccept = (pAccept[0] < 0.0);
    if (violation) infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": pAccept > 1");
    if (negPaccept) infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": pAccept < 0");
    //Print more info for bad cases.
    if ((violation || negPaccept) && verbose >= DEBUG) winnerQCD->list();
  }

  // Enhance factors < 1 (radiation inhibition) treated by modifying pAccept.
  const double enhanceFac = winnerQCD->enhanceFac();
  if (rndmPtr->flat() > min(1.0,enhanceFac)*pAccept[0]) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__ , "Trial rejected at veto "
        "step. wPhys/wTrial = " + num2str(pAccept[0])
        + " * enhanceFac = "+num2str(enhanceFac));

    // Reweighting to account for enhancement factor (reject).
    if (enhanceFac != 1.0)
      weightsPtr->scaleWeightEnhanceReject(pAccept[0],enhanceFac);
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"veto(pAccept)");
    return false;

  } else {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__, "Trial accepted");

    // Reweighting to account for enhancement factor (accept).
    if (enhanceFac != 1.0) weightsPtr->scaleWeightEnhanceAccept(enhanceFac);
  }

  // Diagnostics
  if (verbose >= REPORT) diagnosticsPtr->stop(__METHOD_NAME__,"accept");

  return true;
}

//--------------------------------------------------------------------------

// Generate new particles for the antenna.

bool VinciaFSR::getNewParticles(Event& event, AntennaFunction* antFunPtr,
  vector<Particle>& newParts) {

  // Generate full kinematics.
  if (antFunPtr == nullptr) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": antFunPtr is null pointer");
    return false;
  }
  newParts.clear();
  vector<Vec4> pPost;
  int maptype = antFunPtr->kineMap();
  if (!genFullKinematics(maptype, event, pPost)) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Failed to generate kinematics");
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"veto(kinematics)");
    return false;
  }

  // Generate new helicities.
  vector<int> hPost = genHelicities(antFunPtr);
  if (pPost.size() != hPost.size()) {
    if (verbose >= NORMAL) {
      stringstream ss;
      ss << " pPost.size() = "
         << pPost.size() <<"  hPost.size() = " << hPost.size();
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Wrong size containers.", ss.str());
    }
    return false;
  } else if (!winnerQCD->getNewParticles(event, pPost, hPost, newParts,
      rndmPtr, colourPtr)) {
    if (verbose >= REPORT)
      printOut(__METHOD_NAME__, "Failed to generate new particles");
    return false;
  } else return true;

}

//--------------------------------------------------------------------------

// Generate new helicities for the antenna.

vector<int> VinciaFSR::genHelicities(AntennaFunction* antFunPtr) {

  vector<int> hPre = winnerQCD->hVec();
  vector<int> hPost = hPre;
  hPost.insert(hPost.begin() + 1, 9);
  if (hPost.size() >=3) {
    if (helicityShower && polarisedSys[iSysWin]) {
      vector<double> mPost = winnerQCD->getmPostVec();
      vector<double> invariants = winnerQCD->getInvariants();
      double helSum = antFunPtr->antFun(invariants, mPost, hPre, hPost);
      double randHel = rndmPtr->flat() * helSum;
      double aHel = 0.0;
      // Select helicity, n.b. positions hard-coded. hPost may be
      // larger than 3 in case of resonance decays but meaning of
      // first 3 positions is same (rest are unchanged).
      for (int iHel = 0; iHel < 8; ++iHel) {
        hPost[0] = ( (iHel%2)   )*2 -1;
        hPost[1] = ( (iHel/2)%2 )*2 -1;
        hPost[2] = ( (iHel/4)%2 )*2 -1;
        aHel = antFunPtr->antFun(invariants, mPost, hPre, hPost);
        randHel -= aHel;
        if (verbose >= DEBUG) printOut(__METHOD_NAME__, "antPhys(" +
            num2str(int(hPre[0])) + " " + num2str(int(hPre[1])) + "  -> " +
            num2str(hPost[0]) + " " + num2str(hPost[1]) + " " +
            num2str(hPost[2]) + ") = " + num2str(aHel) + ", m(IK,ij,jk) = " +
            num2str(sqrt(invariants[0])) + ", " +
            num2str(sqrt(invariants[1])) + ", " +
            num2str(sqrt(invariants[2])) + "; sum = "+num2str(helSum));
        if (randHel < 0.) break;
      }
    }
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "selected"+num2str((int)(hPre[0]))
        + " " + num2str(int(hPre[1])) + "  -> " + num2str(hPost[0]) + " "
        + num2str(hPost[1]) + " " + num2str(hPost[2]));
  }
  return hPost;

}

//--------------------------------------------------------------------------

// Get matrix element correction factor.

double VinciaFSR::getMEC(int iSys, const Event& event,
  const vector<Particle>& statePost, VinciaClustering& thisClus) {

  double mec = 1.;

  // Check if post-branching state is set.
  if (statePost.size() == 0) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,
        ": Post-branching state not set in system " + num2str(iSysWin,2));
    return mec;
  }

  // Matrix elemnt corrections for the global shower not implemented.
  if (!sectorShower) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__,
        ": Matrix element corrections not implemented for global shower");
    return mec;
  }

  // Matrix element corrections for the sector shower.
  // Get current state as list of particles.
  vector<Particle> stateNow
    = vinComPtr->makeParticleList(iSysWin, event);
  mec = mecsPtr->getMECSector(iSys, stateNow, statePost, thisClus);
  // Sanity check.
  if (mec < 0.) {
    if (verbose >= NORMAL) {
      stringstream ss;
      ss << ": Negative matrix element correction factor";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+ss.str(),
        "("+num2str(mec,6)+")", true);
    }
    return 1.;
  }

  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Found matrix element correction factor " << mec;
    printOut(__METHOD_NAME__, ss.str());
  }
  return mec;
}

//--------------------------------------------------------------------------

// Update the event.

bool VinciaFSR::updateEvent(Event& event, ResJunctionInfo& junctionInfoIn) {

  for (unsigned int i = 0; i < pNew.size(); ++i) event.append(pNew[i]);
  map<int, pair<int,int> >::iterator it;
  for (it = winnerQCD->mothers2daughters.begin();
       it != winnerQCD->mothers2daughters.end(); ++it) {
    int mother    = it->first;
    int daughter1 = (it->second).first;
    int daughter2 = (it->second).second;
    if (mother<event.size() && mother > 0) {
      event[mother].daughters(daughter1,daughter2);
      event[mother].statusNeg();
    } else return false;
  }

  // Add mothers to new daughters.
  for (it = winnerQCD->daughters2mothers.begin();
      it != winnerQCD->daughters2mothers.end(); ++it) {
    int daughter = it->first;
    int mother1  = (it->second).first;
    int mother2  = (it->second).second;
    if (daughter<event.size() && daughter > 0)
      event[daughter].mothers(mother1, mother2);
    else return false;
  }

  // Tell Pythia if we used a colour tag.
  if (winnerQCD->colTag() != 0) {
    int lastTag = event.nextColTag();
    int colMax  = winnerQCD->colTag();
    while (colMax > lastTag) lastTag = event.nextColTag();
  }
  iNewSav = winnerQCD->iNew();

  // Deal with any resonance junctions.
  if (hasResJunction[iSysWin]) {
    vector<int>* colours = &junctionInfoIn.colours;
    if (!event[junctionInfoIn.iEndQuark].isQuark()) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Can't update junction. iEndQuark is not a quark!");
      hasResJunction[iSysWin]=false;
      return false;
    }

    // Check if resonance splitting.
    bool isRF = winnerQCD->posR() >= 0;
    BranchType branchType = winnerQCD->getBranchType();
    if (branchType == BranchType::SplitF) {
      int splitter = winnerQCD->i0();
      if (isRF) splitter = winnerQCD->iVec().at(winnerQCD->posF());
      // First update list of colours.
      int colLeft = event[splitter].col();
      // Find position col index.
      vector<int>::iterator pos = find(colours->begin(), colours->end(),
        colLeft);
      // Check if emission in string.
      if (pos != colours->end()) {
        // Remove part of string that has split off.
        colours->erase(pos + 1, colours->end());
        // Now update the junction info.
        int d1 = event[splitter].daughter1();
        int d2 = event[splitter].daughter2();
        if (event[d1].isQuark() && event[d1].col() > 0) {
          junctionInfoIn.iEndQuark  = d1;
          junctionInfoIn.iEndColTag = event[d1].col();
        } else if (event[d2].isQuark() && event[d2].col() > 0) {
          junctionInfoIn.iEndQuark  = d2;
          junctionInfoIn.iEndColTag = event[d2].col();
        }
        // Update junction.
        event.endColJunction(junctionInfoIn.iJunction, junctionInfoIn.iEndCol,
          junctionInfoIn.iEndColTag);
      }
    } else if (branchType == BranchType::Emit) {
      // First update list of colours.
      int iNew = winnerQCD->iNew();

      // Find radiator (parton whose colours changed).
      int iRad = event[iNew].mother1();
      if (!isRF) {
        // Need to test both mothers.
        int m2 = event[iNew].mother2();
        if (m2 !=0) {
          // Get daughter that isn't iNew.
          int m2after=event[m2].daughter1();
          if (m2after==iNew) m2after = event[m2].daughter2();
          //Check, did this mother change colours or was it the
          // recoiler?
          int colBef    = event[m2].col();
          int acolBef   = event[m2].acol();
          int colAfter  = event[m2after].col();
          int acolAfter = event[m2after].acol();
          if (colBef != colAfter || acolBef != acolAfter) iRad = m2;
        }
      }

      //Find new colour to insert and old colour.
      int colNew = 0;
      int colLeft = event[iRad].col();
      if (event[iNew].col() == colLeft) colNew = event[iNew].acol();
      else colNew = event[iNew].col();
      if (colNew == 0) {
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Couldn't find colour for updating junction info");
        return false;
      }

      // Find position of radiator col index.
      vector<int>::iterator pos = find(colours->begin(), colours->end(),
        colLeft);
      if (pos!=colours->end()) colours->insert(pos+1,colNew);

      // Now check if end quark has changed colour.
      int iEndQuark = junctionInfoIn.iEndQuark;
      if (!event[iEndQuark].isFinal()) {
        int d1 = event[iEndQuark].daughter1();
        int d2 = event[iEndQuark].daughter2();
        if (event[d1].isQuark() && event[d1].col() > 0) {
          junctionInfoIn.iEndQuark  = d1;
          junctionInfoIn.iEndColTag = event[d1].col();
        } else if (event[d2].isQuark() && event[d2].col() > 0) {
          junctionInfoIn.iEndQuark  = d2;
          junctionInfoIn.iEndColTag = event[d2].col();
        } else {
          infoPtr->errorMsg("Error in "+__METHOD_NAME__
            +": Couldn't update junction");
          return false;
        }
        // Update junction.
        event.endColJunction(junctionInfoIn.iJunction,
          junctionInfoIn.iEndCol, junctionInfoIn.iEndColTag);
      }
    }
  }
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Succesfully updated event after emission");
    event.list();
  }
  return true;

}

//--------------------------------------------------------------------------

// Update the parton systems.

void VinciaFSR::updatePartonSystems() {

  // List parton systems.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Parton systems before update: ");
    partonSystemsPtr->list();
  }

  // Loop over mothers.
  vector<int> newpartons;
  for (map<int, pair<int,int> >::iterator it =
         winnerQCD->mothers2daughters.begin();
       it != winnerQCD->mothers2daughters.end(); ++it) {
    int mother    = it->first;
    int daughter1 = (it->second).first;
    int daughter2 = (it->second).second;
    // Two identical non-zero daughters -> recoilers, just update.
    if (daughter1 == daughter2 && daughter1 != 0 && daughter2 != 0) {
      partonSystemsPtr->replace(iSysWin, mother, daughter1);
      newpartons.push_back(daughter1);
    }
    // Two non-identical daughters -> brancher.
    else if (daughter1 != daughter2 && daughter1 != 0 && daughter2 != 0) {
      // Check if we have already added either daughter.
      bool found1 = false;
      bool found2 = false;
      vector<int>::iterator findit = find(newpartons.begin(), newpartons.end(),
        daughter1);
      if (findit != newpartons.end()) found1 = true;
      findit = find(newpartons.begin(), newpartons.end(), daughter2);
      if (findit != newpartons.end()) found2=true;
      // Both added already. Just continue.
      if (found1 && found2) continue;
      // 1 in record already - just update mother with 2.
      else if (found1 && !found2) {
        partonSystemsPtr->replace(iSysWin, mother, daughter2);
        newpartons.push_back(daughter2);
      // 2 in record already - just update mother with 1
      } else if (!found1 && found2) {
        partonSystemsPtr->replace(iSysWin, mother, daughter1);
        newpartons.push_back(daughter1);
      }
      // Neither in record, update mother with 1, add 2.
      else {
        partonSystemsPtr->replace(iSysWin, mother, daughter1);
        partonSystemsPtr->addOut(iSysWin, daughter2);
        newpartons.push_back(daughter1);
        newpartons.push_back(daughter2);
      }
    }
  }
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Parton systems after update: ");
    partonSystemsPtr->list();
  }

}

//--------------------------------------------------------------------------

// Create a new emission brancher.

void VinciaFSR::saveEmitterFF(int iSysIn, Event& event, int i0, int i1) {
  if (i0 <= 0 || i1 <= 0 || i0 > event.size() || i1 > event.size()) return;
  if (event[i0].col() == event[i1].acol()) {
    BrancherEmitFF temp(iSysIn,event,sectorShower,i0,i1,zetaGenSetFF);
    emittersFF.push_back(std::move(temp));
    lookupEmitterFF[make_pair(i0,true)]=(emittersFF.size()-1);
    lookupEmitterFF[make_pair(i1,false)]=(emittersFF.size()-1);
  }
}

//--------------------------------------------------------------------------

// Create a new resonance emission brancher.

void VinciaFSR::saveEmitterRF(int iSysIn, Event& event, vector<int> allIn,
  unsigned int posResIn, unsigned int posFIn, bool colMode) {

  int iRes = allIn[posResIn];
  if (kMapResEmit == 2 && allIn.size() > 3) {
    // Save radiator.
    allIn.clear();
    int iRad = allIn[posFIn];
    int iRec = 0;
    int d1   = event[iRes].daughter1();
    int d2   = event[iRes].daughter2();

    // Find original colour connected.
    if (colMode) {
      // d2 was original recoiler.
      if (event[d1].col() > 0 && event[iRes].col() == event[d1].col())
        iRec = event[d2].iBotCopy();
      // d1 was original recoiler.
      else iRec = event[d1].iBotCopy();
    } else {
      // d2 was original recoiler.
      if (event[d1].acol() > 0 && event[iRes].acol() == event[d1].acol() )
        iRec = event[d2].iBotCopy();
      // d1 was original recoiler.
      else iRec = event[d1].iBotCopy();
    }
    allIn.push_back(iRes);
    allIn.push_back(iRad);
    allIn.push_back(iRec);
    posResIn=0;
    posFIn=1;
  }

  // Discriminate between colour and anticolour res antennae to avoid
  // degeneracy in lookupEmitterRF0 if res is colour octet.
  if (!colMode) iRes *= -1;

  BrancherEmitRF temp(iSysIn,event,sectorShower,allIn,posResIn,posFIn,
    q2CutoffEmit,zetaGenSetRF);
  emittersRF.push_back(std::move(temp));
  lookupEmitterRF[make_pair(iRes,true)]=(emittersRF.size() - 1);
  lookupEmitterRF[make_pair(allIn[posFIn],false)]=(emittersRF.size() - 1);

}

//--------------------------------------------------------------------------

// Create a new resonance splitter.

void VinciaFSR::saveSplitterRF(int iSysIn, Event& event, vector<int> allIn,
  unsigned int posResIn, unsigned int posFIn, bool colMode) {

  int iRes = allIn[posResIn];
  if (kMapResSplit == 2 && allIn.size() > 3) {
    // Save radiator.
    allIn.clear();
    int iRad = allIn[posFIn];
    int iRec = 0;
    int d1   = event[iRes].daughter1();
    int d2   = event[iRes].daughter2();

    // Find original colour connected.
    // colMode = true if the resonance parent is coloured.
    //         = false if the resonance parent is anticoloured.
    if (colMode) {
      // d2 was original recoiler.
      if (event[d1].col() > 0 && event[iRes].col() == event[d1].col())
        iRec = event[d2].iBotCopy();
      // d1 was original recoiler.
      else iRec = event[d1].iBotCopy();
    } else {
      // d2 was original recoiler.
      if (event[d1].acol() > 0 && event[iRes].acol() == event[d1].acol())
        iRec = event[d2].iBotCopy();
      // d1 was original recoiler.
      else iRec = event[d1].iBotCopy();
    }
    allIn.push_back(iRes);
    allIn.push_back(iRad);
    allIn.push_back(iRec);
    posResIn=0;
    posFIn=1;
  }

  // Discriminate between colour and anticolour res antennae to avoid
  // degeneracy in lookupEmitterRF0 if res is colour octet.
  if (!colMode) iRes*=-1;
  BrancherSplitRF temp(iSysIn,event,sectorShower,allIn,posResIn,posFIn,
    q2CutoffSplit,zetaGenSetRF);
  splittersRF.push_back(std::move(temp));
  lookupSplitterRF[make_pair(iRes,true)]=(splittersRF.size() -1);
  lookupSplitterRF[make_pair(allIn[posFIn],false)]=(splittersRF.size() -1);

}

//--------------------------------------------------------------------------

// Create a new splitter brancher.

void VinciaFSR::saveSplitterFF(int iSysIn, Event& event, int i0, int i1,
  bool col2acol) {
  if (i0 > event.size() || i1 > event.size() ) {
    return;
  }
  BrancherSplitFF temp(iSysIn, event, sectorShower, i0, i1, col2acol,
    zetaGenSetFF);
  splittersFF.push_back(temp);
  if (event[i0].isGluon()) {
    // Colour to anti-colour.
    if (col2acol) {
      lookupSplitterFF[make_pair(i0,true)]=(splittersFF.size()-1);
      lookupSplitterFF[make_pair(i1,false)]=(splittersFF.size()-1);
    // Anti-colour to colour.
    } else {
      lookupSplitterFF[make_pair(-i0,true)]=(splittersFF.size()-1);
      lookupSplitterFF[make_pair(-i1,false)]=(splittersFF.size()-1);
    }
  }
}

//--------------------------------------------------------------------------

// Update all FF emitters replacing recoiling parton iOld -> iNew.

void VinciaFSR::updateEmittersFF(Event& event, int iOld, int iNew) {

  pair<int,bool> key = make_pair(iOld, true);
  if (lookupEmitterFF.find(key) != lookupEmitterFF.end()) {
    unsigned int pos = lookupEmitterFF[key];
    int iRec         = emittersFF[pos].i1();
    int iSysNow      = emittersFF[pos].system();
    emittersFF[pos]  = BrancherEmitFF(iSysNow, event, sectorShower,
      abs(iNew), iRec, zetaGenSetFF);
    lookupEmitterFF.erase(key);
    lookupEmitterFF[make_pair(iNew,true)] = pos;
  }
  key = make_pair(iOld, false);
  if (lookupEmitterFF.find(key) != lookupEmitterFF.end()) {
    unsigned int pos = lookupEmitterFF[key];
    int iEmit        = emittersFF[pos].i0();
    int iSysNow      = emittersFF[pos].system();
    emittersFF[pos]  = BrancherEmitFF(iSysNow, event, sectorShower,
      iEmit, abs(iNew), zetaGenSetFF);
    lookupEmitterFF.erase(key);
    lookupEmitterFF[make_pair(iNew,false)]=pos;
  }

}

//--------------------------------------------------------------------------

// Update all FF splitters replacing recoling parton iOld -> iNew.

void VinciaFSR::updateSplittersFF(Event& event, int iOld, int iNew) {

  // Loop over colour and anticolour side of gluon splittings.
  for (int sign = -1; sign <= 1; sign += 2) {
    pair<int,bool> key = make_pair(sign*iOld, true);
    if (lookupSplitterFF.find(key) != lookupSplitterFF.end()) {
      unsigned int pos = lookupSplitterFF[key];
      int iRec         = splittersFF[pos].i1();
      int iSysNow      = splittersFF[pos].system();
      bool col2acol    = !splittersFF[pos].isXG();
      splittersFF[pos] = BrancherSplitFF(iSysNow, event, sectorShower,
        abs(iNew), iRec, col2acol, zetaGenSetFF);
      lookupSplitterFF.erase(key);
      lookupSplitterFF[make_pair(sign*iNew,true)] = pos;
    }
    key = make_pair(sign*iOld, false);
    if (lookupSplitterFF.find(key) != lookupSplitterFF.end()) {
      unsigned int pos = lookupSplitterFF[key];
      int iEmit        = splittersFF[pos].i0();
      int iSysNow      = splittersFF[pos].system();
      bool col2acol    = !splittersFF[pos].isXG();
      splittersFF[pos] = BrancherSplitFF(iSysNow, event, sectorShower,
        iEmit, abs(iNew), col2acol, zetaGenSetFF);
      lookupSplitterFF.erase(key);
      lookupSplitterFF[make_pair(sign*iNew,false)]=pos;
    }
  }

}

//--------------------------------------------------------------------------

// Update emission brancher due to an emission.

void VinciaFSR::updateEmitterFF(Event& event,int iOld1, int iOld2,
  int iNew1, int iNew2) {

  pair<int,bool> key1 = make_pair(iOld1,true);
  pair<int,bool> key2 = make_pair(iOld2,false);
  if (lookupEmitterFF.find(key1) != lookupEmitterFF.end()) {
    unsigned int pos = lookupEmitterFF[key1];
    if (lookupEmitterFF.find(key2) != lookupEmitterFF.end()) {
      unsigned int pos2=lookupEmitterFF[key2];
      if (pos == pos2) {
        lookupEmitterFF.erase(key1);
        lookupEmitterFF.erase(key2);
        int iSysNow = emittersFF[pos].system();
        emittersFF[pos] = BrancherEmitFF(iSysNow, event, sectorShower,
          abs(iNew1), abs(iNew2), zetaGenSetFF);
        lookupEmitterFF[make_pair(iNew1,true)]=pos;
        lookupEmitterFF[make_pair(iNew2,false)]=pos;
      }
    }
  }

}

//--------------------------------------------------------------------------

// Update splitter brancher due to an emission.

void VinciaFSR::updateSplitterFF(Event& event,int iOld1, int iOld2, int iNew1,
  int iNew2, bool col2acol) {

  int sign=1;
  if (!col2acol) sign=-1;

  pair<int,bool> key1 = make_pair(sign*abs(iOld1),true);
  pair<int,bool> key2 = make_pair(sign*abs(iOld2),false);
  if (lookupSplitterFF.find(key1) != lookupSplitterFF.end()) {
    unsigned int pos = lookupSplitterFF[key1];
    if (lookupSplitterFF.find(key2) != lookupSplitterFF.end()) {
      unsigned int pos2=lookupSplitterFF[key2];
      if (pos == pos2) {
        lookupSplitterFF.erase(key1);
        lookupSplitterFF.erase(key2);
        int iSysNow = splittersFF[pos].system();
        splittersFF[pos] = BrancherSplitFF(iSysNow, event, sectorShower,
          abs(iNew1), abs(iNew2), col2acol, zetaGenSetFF);
        lookupSplitterFF[make_pair(sign*abs(iNew1),true)]=pos;
        lookupSplitterFF[make_pair(sign*abs(iNew2),false)]=pos;
      }
    }
  }

}

//--------------------------------------------------------------------------

// Remove a splitter due to a gluon that has branched.

void VinciaFSR::removeSplitterFF(int iRemove) {

  for (int isign = 0; isign < 2; isign++) {
    int sign = 1 - 2*isign;
    pair<int,bool> key = make_pair(sign*iRemove, true);

    // Update map.
    if (lookupSplitterFF.find(key) != lookupSplitterFF.end()) {
      unsigned int pos = lookupSplitterFF[key];
      lookupSplitterFF.erase(key);
      int iRec = splittersFF[pos].i1();
      pair<int,bool> recoilkey = make_pair(sign*iRec, false);
      if (lookupSplitterFF.find(recoilkey) != lookupSplitterFF.end())
        lookupSplitterFF.erase(recoilkey);
      if (pos < splittersFF.size()) {
        splittersFF.erase(splittersFF.begin()+pos);

        // Update map with modified positions.
        for (; pos < splittersFF.size(); pos++) {
          //Get brancher at current pos.
          BrancherSplitFF splitter = splittersFF.at(pos);
          // Find indices.
          int i0(splitter.i0()), i1(splitter.i1());
          // Update lookup map to new pos.
          if (!splitter.isXG()) {
            lookupSplitterFF[make_pair(i0,true)]=pos;
            lookupSplitterFF[make_pair(i1,false)]=pos;
          } else{
            lookupSplitterFF[make_pair(-i0,true)]=pos;
            lookupSplitterFF[make_pair(-i1,false)]=pos;
          }
        } // End loop over splitters.
      }
    }
  } // End loop over signs.

}

//--------------------------------------------------------------------------

// Update resonance emitter due to changed downstream decay products.

bool VinciaFSR::updateEmittersRF(int iSysRes, Event& event, int iRes) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Colour information of resonance. Return if colour singlet.
  int resCol      = event[iRes].col();
  int resACol     = event[iRes].acol();
  if (resCol == 0 && resACol == 0) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "end (resonance is colour singlet)", dashLen);
    return true;
  }

  // Look up decay products using partonSystems, assumed to be updated already.
  int colPartner  = -1;
  int acolPartner = -1;
  vector<int> daughters;

  // Loop over members of current decay system and get colour information.
  int sizeOut = partonSystemsPtr->sizeOut(iSysRes);
  for (int iDecPart = 0; iDecPart < sizeOut; iDecPart++) {
    int iOut = partonSystemsPtr->getOut(iSysRes,iDecPart);

    // Check if coloured partner of the resonance.
    if (event[iOut].col() != 0 && event[iOut].col() == resCol)
      colPartner = iOut;
    if (event[iOut].acol() != 0 && event[iOut].acol() == resACol)
      acolPartner = iOut;
    if (iOut != colPartner && iOut != acolPartner)
      daughters.push_back(iOut);
  }
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "col partner = " << colPartner << " acol partner = " << acolPartner;
    printOut(__METHOD_NAME__,ss.str());
  }

  if (colPartner > 0) {
    // Get a copy of daughters.
    vector<int> resSysAll = daughters;
    if (acolPartner != colPartner && acolPartner > 0)
      resSysAll.push_back(acolPartner);
    // Insert col partner and res at front (just convention).
    resSysAll.insert(resSysAll.begin(),colPartner);
    resSysAll.insert(resSysAll.begin(),iRes);
    unsigned int posRes(0), posPartner(1);
    updateEmittersRF(iSysRes,event,resSysAll,posRes,posPartner,true);
  }
  if (acolPartner > 0) {
    // Get a copy of daughters.
    vector<int> resSysAll = daughters;
    if (acolPartner != colPartner && colPartner > 0)
      resSysAll.push_back(colPartner);
    // Insert col partner and res at front (just convention).
    resSysAll.insert(resSysAll.begin(),acolPartner);
    resSysAll.insert(resSysAll.begin(),iRes);
    unsigned int posRes(0), posPartner(1);
    updateEmittersRF(iSysRes,event,resSysAll,posRes,posPartner,false);
  }
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "end", dashLen);

  return true;

}

//--------------------------------------------------------------------------

// Update resonance emitter due to changed downstream decay products.

void VinciaFSR::updateEmittersRF(int iSysRes, Event& event,
  vector<int> resSysAll, unsigned int posRes, unsigned int posPartner,
  bool isCol) {

  if (posRes >= resSysAll.size() || posPartner >= resSysAll.size()) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": Invalid positions");
    infoPtr->setAbortPartonLevel(true);
    return;
  }
  int iRes      = resSysAll[posRes];
  int iPartner  = resSysAll[posPartner];
  int posREmit  = posRes;
  int posFEmit  = posPartner;
  int posRSplit = posRes;
  int posFSplit = posPartner;
  vector<int> resSysAllEmit;
  vector<int> resSysAllSplit;

  // If "bad" recoil map need to update recoiler system resSysAll.
  if (kMapResEmit == 2 && resSysAll.size() > 3) {
    // Fetch daughters of res.
    int iRec = 0;
    int d1 = event[iRes].daughter1();
    int d2 = event[iRes].daughter2();

    // Find original colour connected.
    if (isCol) {
      // d2 was original recoiler.
      if (event[d1].col() > 0 && event[iRes].col() == event[d1].col())
        iRec = event[d2].iBotCopy();
      // d1 was original recoiler.
      else iRec = event[d1].iBotCopy();
    } else {
      // d2 was original recoiler.
      if (event[d1].acol() > 0 && event[iRes].acol() == event[d1].acol())
        iRec = event[d2].iBotCopy();
      // d1 was original recoiler.
      else iRec = event[d1].iBotCopy();
    }
    resSysAllEmit.push_back(iRes);
    resSysAllEmit.push_back(iPartner);
    resSysAllEmit.push_back(iRec);
    posREmit=0;
    posFEmit=1;
  } else resSysAllEmit = resSysAll;
  if (kMapResSplit == 2) {
    resSysAllSplit = resSysAllEmit;
    posRSplit=0;
    posFSplit=1;
  } else resSysAllSplit = resSysAll;
  if (!isCol) iRes*=-1;

  // First update emission brancher -> always need to update because
  // downstream recoilers will have changed.
  pair<int,bool> branchkey = make_pair(iRes, true);
  if (lookupEmitterRF.find(branchkey) != lookupEmitterRF.end()) {
    unsigned int pos =lookupEmitterRF[branchkey];
    int iRec = (emittersRF[pos].iVec())[emittersRF[pos].posF()];
    pair<int,bool> recoilkey=make_pair(iRec,false);
    // Delete map to recoiler.
    if (lookupEmitterRF.find(recoilkey) != lookupEmitterRF.end())
      lookupEmitterRF.erase(recoilkey);
    // Reset brancher.
    emittersRF[pos].resetRF(iSysRes, event, resSysAllEmit, posREmit,
      posFEmit, q2CutoffEmit, zetaGenSetRF);
    // Add new map.
    recoilkey = make_pair(iPartner,false);
    lookupEmitterRF[recoilkey] = pos;
  }

  // Splitters - treatement depends on latest emission.
  if (lookupSplitterRF.find(branchkey) != lookupSplitterRF.end()) {
    unsigned int pos = lookupSplitterRF[branchkey];
    int iSplit = (splittersRF[pos].iVec())[splittersRF[pos].posF()];
    pair<int,bool> splitkey=make_pair(iSplit,false);

    // Delete map to recoiler.
    if (lookupSplitterRF.find(splitkey) != lookupSplitterRF.end())
      lookupSplitterRF.erase(splitkey);

    //Do we need to remove this splitter, is the splitter still a gluon?
    if (!event[iPartner].isGluon()) {
      lookupSplitterRF.erase(branchkey);
      splittersRF.erase(splittersRF.begin()+pos);
      // Update any other splitters' positions in lookup map.
      for (unsigned int ipos = pos; ipos < splittersRF.size(); ipos++) {
        BrancherSplitRF splitter = splittersRF.at(ipos);
        int itmpSplit = (splittersRF[ipos].iVec())[splittersRF[ipos].posF()];
        // Update lookup map to new pos.
        lookupSplitterRF[make_pair(iRes,true)] = ipos;
        lookupSplitterRF[make_pair(itmpSplit,false)] = ipos;
      }
    // Otherwise just update.
    } else {
      splittersRF[pos].resetRF(iSysRes, event, resSysAllSplit,
        posRSplit, posFSplit, q2CutoffSplit, zetaGenSetRF);
      // Add new map.
      splitkey = make_pair(iPartner,false);
      lookupSplitterRF[splitkey]=pos;
    }
  // Else if last branch type was res branch add new res splitter.
  } else if (winnerQCD!= nullptr) {
    bool isRFsplit = winnerQCD->getBranchType() == BranchType::SplitF
      && winnerQCD->posR() >= 0;
    if (isRFsplit && event[iPartner].isGluon())
      saveSplitterRF(iSysRes,event,resSysAllSplit,posRSplit,posFSplit,isCol);
  }

}

//--------------------------------------------------------------------------

// Update the antennae.

bool VinciaFSR::updateAntennae(Event& event) {

  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "begin", dashLen);
    if (verbose >= DEBUG) printLookup();
  }
  if (winnerQCD == nullptr) {
    if (verbose >= NORMAL) infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": winnerQCD is null pointer");
    return false;
  }

  // Update EW system(s), then QCD.
  if (doQED) {
    if (isHardSys[iSysWin] || isResonanceSys[iSysWin])
      ewHandlerHard->update(event, iSysWin);
    else
      qedShowerSoftPtr->update(event, iSysWin);
  }

  // Was this g->ffbar?
  bool isRF = winnerQCD->posR() >= 0;
  BranchType branchType = winnerQCD->getBranchType();
  if (branchType == BranchType::SplitF && !isRF) {
    // Remove old splitters where g = i0 and update splitters where g is rec.
    int splitOld = winnerQCD->i0();
    int recOld = winnerQCD->i1();
    removeSplitterFF(splitOld);

    // Get daughters.
    int iColSplit = event[splitOld].daughter1();
    int iaColSplit = event[splitOld].daughter2();
    if (event[iColSplit].col() == 0 && event[iaColSplit].acol() == 0 &&
       event[iColSplit].acol() != 0 && event[iaColSplit].col() != 0) {
      iColSplit = event[splitOld].daughter2();
      iaColSplit = event[splitOld].daughter1();
    }

    //Find colour connected partner.
    int iColPartner = 0;
    int iaColPartner = 0;
    pair<int,bool> testkey = make_pair(splitOld,true);
    if (lookupEmitterFF.find(testkey) != lookupEmitterFF.end()) {
      int iTest = emittersFF[lookupEmitterFF[testkey]].i1();
      if (event[iTest].acol() == event[splitOld].col()) iaColPartner = iTest;
    }
    testkey = make_pair(splitOld,false);
    if (lookupEmitterFF.find(testkey) != lookupEmitterFF.end()) {
      int iTest = emittersFF[lookupEmitterFF[testkey]].i0();
      if (event[iTest].col() == event[splitOld].acol()) iColPartner = iTest;
    }

    //Update splitters where g is (anti-)colour-connected recoiler/emitter.
    updateSplitterFF(event,iaColPartner,splitOld,iaColPartner,iColSplit,false);
    updateSplitterFF(event,iColPartner,splitOld,iColPartner,iaColSplit,true);
    updateEmitterFF(event,iColPartner,splitOld,iColPartner,iaColSplit);
    updateEmitterFF(event,splitOld,iaColPartner,iColSplit,iaColPartner);

    // Update recoiler.
    int recNew = event[recOld].daughter1();
    updateSplittersFF(event,recOld,recNew);
    updateEmittersFF(event,recOld,recNew);
  }

  // Emission.
  else if (branchType == BranchType::Emit && !isRF) {
    // Update old splitters.
    int iOld1 = winnerQCD->i0();
    int iOld2 = winnerQCD->i1();
    int iNew1 = event[iOld1].daughter1();
    int iNew2 = event[iOld1].daughter2();
    int iNew3 = event[iOld2].daughter1();

    // Switch 1<->2 so that 2 is repeated daughter.
    if (iNew3 == iNew1) {
      iNew1=iNew2;
      iNew2=iNew3;
      iNew3=event[iOld2].daughter2();
    } else if (iNew3 == iNew2) iNew3=event[iOld2].daughter2();

    // Update emitters, determine antenna to preserve.
    // ab->12.
    if (event[iOld1].col() == event[iNew1].col()) {
      updateEmitterFF(event,iOld1,iOld2,iNew1,iNew2);
      if (event[iNew2].col() == event[iNew3].acol())
        saveEmitterFF(iSysWin,event,iNew2,iNew3);
    // ab->23.
    } else {
      updateEmitterFF(event,iOld1,iOld2,iNew2,iNew3);
      if (event[iNew1].col()==event[iNew2].acol())
        saveEmitterFF(iSysWin,event,iNew1,iNew2);
    }
    if (event[iNew1].isGluon())
      updateSplitterFF(event,iOld1,iOld2,iNew1,iNew2,true);
    if (event[iNew3].isGluon())
      updateSplitterFF(event,iOld2,iOld1,iNew3,iNew2,false);

    // New splitters.
    if (event[iNew2].isGluon()) {
      saveSplitterFF(iSysWin,event,iNew2,iNew3,true);
      saveSplitterFF(iSysWin,event,iNew2,iNew1,false);
    }

    // Update other connected-connected antenna, excluding antenna
    // which branched.
    updateEmittersFF(event,iOld1,iNew1);
    updateEmittersFF(event,iOld2,iNew3);
    updateSplittersFF(event,iOld1,iNew1);
    updateSplittersFF(event,iOld2,iNew3);

  // Resonance emission.
  } else if (isRF) {
    // Update emitters and splitters.
    for (map<int, pair<int, int> >::iterator it =
         winnerQCD->mothers2daughters.begin();
         it!= winnerQCD->mothers2daughters.end(); ++it) {
      int mother    = it->first;
      int daughter1 = (it->second).first;
      int daughter2 = (it->second).second;
      // Recoiler -> just update.
      if (daughter1 == daughter2) {
        updateEmittersFF(event,mother,daughter1);
        updateSplittersFF(event,mother,daughter1);
      // Resonance emitter.
      } else {
        // Convention of res emission: daughter1 is new emission but
        // check anyway.
        if (branchType == BranchType::Emit && event[daughter1].isGluon()) {
          if (event[daughter1].col()==event[daughter2].acol())
            saveEmitterFF(iSysWin,event,daughter1,daughter2);
          else if (event[daughter1].acol()==event[daughter2].col())
            saveEmitterFF(iSysWin,event,daughter2,daughter1);
          // TODO: check colour condition here.
          bool col2acol = false;
          if (event[daughter1].col() == event[daughter2].acol())
            col2acol = true;
          saveSplitterFF(iSysWin,event,daughter1,daughter2,col2acol);
          updateEmittersFF(event,mother,daughter2);
          updateSplittersFF(event,mother,daughter2);
        // Resonant splitter.
        } else if (branchType == BranchType::SplitF && event[mother].isGluon()
          && !event[daughter1].isGluon() && !event[daughter2].isGluon()) {
          removeSplitterFF(mother);
          int iColSplit  = daughter1;
          int iaColSplit = daughter2;
          if (event[mother].col() != event[daughter1].col()) {
            iColSplit  = daughter2;
            iaColSplit = daughter1;
          }

          // Find colour connected partner.
          int iColPartner(0), iaColPartner(0);
          pair<int,bool> testkey = make_pair(mother,true);
          if (lookupEmitterFF.find(testkey) != lookupEmitterFF.end()) {
            int iTest = emittersFF[lookupEmitterFF[testkey]].i1();
            if (event[iTest].acol() == event[mother].col())
              iaColPartner=iTest;
          }
          testkey = make_pair(mother,false);
          if (lookupEmitterFF.find(testkey) != lookupEmitterFF.end()) {
            int iTest = emittersFF[lookupEmitterFF[testkey]].i0();
            if (event[iTest].col() == event[mother].acol())
              iColPartner=iTest;
          }

          // Update splitters where mother was a
          // (anti)colour-connected recoiler/emitter.
          updateSplitterFF(event, iaColPartner, mother, iaColPartner,
            iColSplit, false);
          updateSplitterFF(event, iColPartner, mother, iColPartner,
            iaColSplit, true);
          updateEmitterFF(event, iColPartner, mother, iColPartner,
            iaColSplit);
          updateEmitterFF(event, mother, iaColPartner, iColSplit,
            iaColPartner);

        }
      } // End found branching in mothers2daughters.
    } // End for loop over mothers2daughters.
  } // End resonance brancher case.

  // If system containing a resonance has branched, must always update
  // (because must update downstream recoilers regardless of if last
  // branch was res emission).
  if (isResonanceSys[iSysWin]) {
    if (!updateEmittersRF(iSysWin, event,
        partonSystemsPtr->getInRes(iSysWin))) {
      if (verbose >= NORMAL)
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Failed updateEmittersRF");
      return false;
    }
  }
  if (verbose >= DEBUG) {
    list();
    printLookup();
  }
  if (!check(iSysWin, event)) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Failed to update branchers");
    return false;
  }
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return true;

}

//--------------------------------------------------------------------------

// Print a brancher lookup.

void VinciaFSR::printLookup(map< pair<int, bool>, unsigned int>&
  lookupEmitter, string name) {
  for (map< pair<int, bool>, unsigned int >::iterator ilook =
         lookupEmitter.begin(); ilook != lookupEmitter.end(); ++ilook)
    cout << "  lookup" << name << "[" << (ilook->first).first
         << "," << (ilook->first).second << "] = " << ilook->second << endl;
}

//--------------------------------------------------------------------------

// Print the brancher lookup maps.

void VinciaFSR::printLookup() {
  cout << endl << "  --------" << "  Brancher lookup maps"
       << "  -------------------------------------------------------------"
       << endl;
  printLookup(lookupEmitterRF,"EmitterRF");
  printLookup(lookupSplitterRF,"SplitterRF");
  printLookup(lookupEmitterFF,"EmitterFF");
  printLookup(lookupSplitterFF,"SplitterFF");
  cout << "  --------" << "       End lookup     "
       <<"  -------------------------------------------------------------"
       << endl << endl;
}

//--------------------------------------------------------------------------

// Calculate the headroom factor.

vector<double> VinciaFSR::getHeadroom(int iSys, bool isEmit, double) {

  // TODO: ensure a decent number of failed trials if doing uncertainties.
  bool doUncert = false;

  // Check if we have we encountered this headroom criterion before.
  pair<int, pair<bool,bool> > headroomKey =
    make_pair(iSys,make_pair(isEmit,doUncert));
  if (headroomSav.find(headroomKey) != headroomSav.end())
    return headroomSav[headroomKey];
  // Otherwise calculate, and save for next time we need it.
  else {
    // Emissions.
    vector<double> headroomVec;
    if (isEmit) {
      double headroomFac = 1.0;
      // Increased headroom factor if doing MECs and/or uncertainties.
      if (doMECsSys[iSys] && mecsPtr->doMEC(iSys,nBranch[iSys]+1)) {
        headroomFac = 1.5;
        // More headroom for 2->2 than for resonance decays.
        if (!isResonanceSys[iSys]) headroomFac *= 2.;
        // More headroom for helicity dependence.
        if (helicityShower && polarisedSys[iSys]) headroomFac *= 1.5;
      }
      if (doUncert) headroomFac *= 1.33;
      headroomVec.push_back(headroomFac);

    // Other.
    } else {
      for (int iFlav = 1; iFlav <= nGluonToQuark; ++iFlav) {
        double headroomFac = 1.0;
        // Heavy flavours get 50% larger trial (since mass correction > 0).
        if (iFlav > nFlavZeroMass) headroomFac *= 1.5;
        // MECs also get increased headroom.
        if (doMECsSys[iSys] && mecsPtr->doMEC(iSys,nBranch[iSys]+1)) {
          headroomFac *= 2.;
          // More headroom for 2->2 than for resonance decays.
          if (!isResonanceSys[iSys]) headroomFac *= 2.;
          // More headroom for helicity dependence.
          if (helicityShower && polarisedSys[iSys]) headroomFac *= 2.;
        }
        headroomVec.push_back(headroomFac);
      }
    }
    headroomSav[headroomKey] = headroomVec;
    return headroomVec;
  }

}

//--------------------------------------------------------------------------

// Calculate the enhancement factor.

vector<double> VinciaFSR::getEnhance(int iSys, bool isEmit, double q2In) {

  bool doEnhance = false;
  if (q2In > pow2(enhanceCutoff)) {
    if (isResonanceSys[iSys] && enhanceInResDec) doEnhance = true;
    else if (isHardSys[iSys] && enhanceInHard) doEnhance = true;
    else if (!isHardSys[iSys] && !isResonanceSys[iSys] &&
      partonSystemsPtr->hasInAB(iSys) && enhanceInMPI) doEnhance = true;
  }

  // Check if we have encountered this enhancement criterion before.
  pair<int,pair<bool,bool> > enhanceKey =
    make_pair(iSys,make_pair(isEmit,doEnhance));
  vector<double> enhanceVec;
  if (enhanceSav.find(enhanceKey) != enhanceSav.end())
    enhanceVec = enhanceSav[enhanceKey];
  else {
    double enhanceFac = 1.0;
    // Emissions.
    if (isEmit) {
      if (doEnhance) enhanceFac *= enhanceAll;
      enhanceVec.push_back(enhanceFac);
    // Other.
    } else {
      for (int iFlav = 1; iFlav <= nGluonToQuark; ++iFlav) {
        if (doEnhance) {
          enhanceFac = enhanceAll;
          // Optional extra enhancement for g->cc, g->bb.
          if (iFlav == 4) enhanceFac *= enhanceCharm;
          else if (iFlav == 5) enhanceFac *= enhanceBottom;
        }
        enhanceVec.push_back(enhanceFac);
      }
    }
    enhanceSav[enhanceKey] = enhanceVec;
  }
  return enhanceVec;
}

//==========================================================================

} // end namespace Pythia8
