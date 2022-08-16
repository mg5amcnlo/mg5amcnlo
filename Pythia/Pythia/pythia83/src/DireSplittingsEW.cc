// DireSplitingsEW.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the Dire
// electroweak splittings.

#include "Pythia8/DireSplittingsEW.h"
#include "Pythia8/DireSpace.h"
#include "Pythia8/DireTimes.h"
#include "Pythia8/DireHistory.h"

namespace Pythia8 {

//==========================================================================

// The SplittingEW class.

//--------------------------------------------------------------------------

void DireSplittingEW::init() {

  // Parameters of alphaEM.
  int alphaEMorder = settingsPtr->mode("SpaceShower:alphaEMorder");
  // Initialize alphaEM.
  alphaEM.init( alphaEMorder, settingsPtr);

  // Z0 and W+- properties needed for gamma/Z0 mixing and weak showers.
  mZ                 = particleDataPtr->m0(23);
  gammaZ             = particleDataPtr->mWidth(23);
  thetaW             = 1. / (16. * coupSMPtr->sin2thetaW()
                       * coupSMPtr->cos2thetaW());
  mW                 = particleDataPtr->m0(24);
  gammaW             = particleDataPtr->mWidth(24);

  aem0 = settingsPtr->parm("StandardModel:alphaEM0");

  enhance = settingsPtr->parm("Enhance:"+ id);

  doQEDshowerByQ  = (is_fsr) ? settingsPtr->flag("TimeShower:QEDshowerByQ")
                             : settingsPtr->flag("SpaceShower:QEDshowerByQ");
  doQEDshowerByL  = (is_fsr) ? settingsPtr->flag("TimeShower:QEDshowerByL")
                             : settingsPtr->flag("SpaceShower:QEDshowerByL");

}

//--------------------------------------------------------------------------

// Function to calculate the correct alphaem/2*Pi value, including
// renormalisation scale variations + threshold matching.

double DireSplittingEW::aem2Pi( double pT2 ) {

  double scale       = pT2*renormMultFac;

  // Get alphaEM(k*pT^2) and subtractions.
  double aemPT2pi = alphaEM.alphaEM(scale) / (2.*M_PI);

  // Done.
  return aemPT2pi;

}

//==========================================================================

// Class inheriting from Splitting class.

// Splitting function Q->QZ (FSR)

// Return true if this kernel should partake in the evolution.
bool Dire_fsr_ew_Q2QZ::canRadiate ( const Event& state, pair<int,int> ints,
  unordered_map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  int nFinPartons(0), nFinQ(0), nFinOther(0);
  for(int i=0; i < state.size(); ++i) {
    if (!state[i].isFinal()) continue;
    if ( state[i].colType() !=0) {
      nFinPartons++;
      if ( abs(state[i].colType()) == 1) nFinQ++;
    } else nFinOther++;
  }
  return ( nFinPartons == 2 && nFinQ > 0 && nFinOther == 0
        && state[ints.first].isFinal()
        && state[ints.first].isQuark() );
}

int Dire_fsr_ew_Q2QZ::kinMap()                        { return 1;}
int Dire_fsr_ew_Q2QZ::motherID(int idDaughter)        { return idDaughter;}
int Dire_fsr_ew_Q2QZ::sisterID(int)                   { return 23;}
double Dire_fsr_ew_Q2QZ::gaugeFactor ( int, int)      { return thetaW; }
double Dire_fsr_ew_Q2QZ::symmetryFactor ( int, int )  { return 1.;}
int Dire_fsr_ew_Q2QZ::radBefID(int idRA, int)         { return idRA;}

pair<int,int> Dire_fsr_ew_Q2QZ::radBefCols(
  int colRadAfter, int acolRadAfter,
  int, int) { return make_pair(colRadAfter,acolRadAfter); }

// Pick z for new splitting.
double Dire_fsr_ew_Q2QZ::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double Dire_fsr_ew_Q2QZ::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  // Q -> QG, soft part (currently also used for collinear part).
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2. * 0.5 * log1p(pow2(1.-zMinAbs)/kappa2);
  return wt;
}

// Return overestimate for new splitting.
double Dire_fsr_ew_Q2QZ::overestimateDiff(double z, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double kappaOld2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool Dire_fsr_ew_Q2QZ::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z);

  double wt = 0.;
  if ( fsr->weights->hasME(fsr->makeHardEvent(0,state,true)) ) {

    // Calculate kernel as ratio of matrix elements.
    Event trialEvent(state);
    if (splitInfo.recBef()->isFinal)
      fsr->branch_FF(trialEvent, true, &splitInfo);
    else
      fsr->branch_FI(trialEvent, true, &splitInfo);

    // Calculate kernel as ratio of matrix elements.
    if ( fsr->weights->hasME(fsr->makeHardEvent(0,trialEvent,true)) ) {

      // Ensure that histories do not try to access kernel again, thus
      // potentially producing an infinite loop.
      splitInfo.addExtra("unitKernel",1.0);

      // Numerator is just the p p --> j j W matrix element.
      double wtNum = fsr->weights->getME(trialEvent);

      // Denominator is sum of all p p --> j j matrix elements.
      // --> Use History class to produce all p p --> j j phase space points.

      // Store previous settings.
      string procSave = settingsPtr->word("Merging:process");
      int nRequestedSave = settingsPtr->mode("Merging:nRequested");

      // Reset hard process to p p --> j j
      string procNow = "pp>jj";
      settingsPtr->word("Merging:process", procNow);
      fsr->mergingHooksPtr->hardProcess->clear();
      fsr->mergingHooksPtr->hardProcess->initOnProcess
        (procNow, particleDataPtr);
      fsr->mergingHooksPtr->processSave = procNow;

      // Denominator is sum of all p p --> j j matrix elements.
      fsr->mergingHooksPtr->orderHistories(false);
      Event newProcess( fsr->mergingHooksPtr->bareEvent(
        fsr->makeHardEvent(0, trialEvent, true), true) );

      // Get the maximal quark flavour counted as "additional" parton.
      int nPartons = 0;
      int nQuarksMerge = settingsPtr->mode("Merging:nQuarksMerge");
      // Loop through event and count.
      for(int i=0; i < int(newProcess.size()); ++i)
        if ( newProcess[i].isFinal()
          && newProcess[i].colType()!= 0
          && ( newProcess[i].id() == 21
               || newProcess[i].idAbs() <= nQuarksMerge))
          nPartons++;
      nPartons -= 2;

      // Set number of requested partons.
      settingsPtr->mode("Merging:nRequested", nPartons);
      fsr->mergingHooksPtr->nRequestedSave
        = settingsPtr->mode("Merging:nRequested");

      // Store candidates.
      fsr->mergingHooksPtr->storeHardProcessCandidates( newProcess );

      // Calculate number of clustering steps
      int nSteps = 1;

      // Set dummy process scale.
      newProcess.scale(0.0);
      // Generate all histories
      DireHistory myHistory( nSteps, 0.0, newProcess, DireClustering(),
        fsr->mergingHooksPtr, (*beamAPtr), (*beamBPtr), particleDataPtr,
        infoPtr,
        NULL, fsr, isr, fsr->weights, coupSMPtr, true, true,
        1.0, 1.0, 1.0, 1.0, 0);
      // Project histories onto desired branches, e.g. only ordered paths.
      myHistory.projectOntoDesiredHistories();

      double wtDen(0.);
      for ( map<double, DireHistory*>::iterator it =
              myHistory.goodBranches.begin();
        it != myHistory.goodBranches.end(); ++it ) {
        Event psppoint = it->second->state;
        wtDen += fsr->weights->getME(psppoint);
      }

      // Reset all merging settings.
      settingsPtr->word("Merging:process", procSave);
      settingsPtr->mode("Merging:nRequested",nRequestedSave);
      fsr->mergingHooksPtr->nRequestedSave
        = settingsPtr->mode("Merging:nRequested");
      fsr->mergingHooksPtr->hardProcess->initOnProcess
        (procSave, particleDataPtr);
      fsr->mergingHooksPtr->processSave = procSave;
      unordered_map<string, double>::iterator it =
        splitInfo.extras.find("unitKernel");
      splitInfo.extras.erase(it);

      // No valid underlying processes means vanishing splitting probability.
      if (myHistory.goodBranches.size() == 0) { wtNum = 0.; wtDen = 1.; }

      wt = wtNum/wtDen;
    }
  }

  // Now multiply with z to project out Q->QG,
  // i.e. the gluon is soft and the quark is identified.
  wt *= z;

  // Trivial map of values, since kernel does not depend on coupling.
  unordered_map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( unordered_map<string,double>::iterator it = wts.begin();
        it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from Splitting class.

// Splitting function Q->ZQ (FSR)

// Return true if this kernel should partake in the evolution.
bool Dire_fsr_ew_Q2ZQ::canRadiate ( const Event& state, pair<int,int> ints,
  unordered_map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  int nFinPartons(0), nFinQ(0), nFinOther(0);
  for(int i=0; i < state.size(); ++i) {
    if (!state[i].isFinal()) continue;
    if ( state[i].colType() !=0) {
      nFinPartons++;
      if ( abs(state[i].colType()) == 1) nFinQ++;
    } else nFinOther++;
  }
  return ( nFinPartons == 2 && nFinQ > 0 && nFinOther == 0
        && state[ints.first].isFinal()
        && state[ints.first].isQuark() );
}

int Dire_fsr_ew_Q2ZQ::kinMap()                        { return 1;}
int Dire_fsr_ew_Q2ZQ::motherID(int idDaughter)        { return idDaughter;}
int Dire_fsr_ew_Q2ZQ::sisterID(int)                   { return 23;}
double Dire_fsr_ew_Q2ZQ::gaugeFactor ( int, int)      { return thetaW; }
double Dire_fsr_ew_Q2ZQ::symmetryFactor ( int, int )  { return 1.;}
int Dire_fsr_ew_Q2ZQ::radBefID(int idRA, int)            { return idRA;}

pair<int,int> Dire_fsr_ew_Q2ZQ::radBefCols( int colRadAfter, int acolRadAfter,
  int, int) { return make_pair(colRadAfter,acolRadAfter); }

// Pick z for new splitting.
double Dire_fsr_ew_Q2ZQ::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double Dire_fsr_ew_Q2ZQ::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  // Q -> QG, soft part (currently also used for collinear part).
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2. * 0.5 * log1p(pow2(1.-zMinAbs)/kappa2);
  return wt;
}

// Return overestimate for new splitting.
double Dire_fsr_ew_Q2ZQ::overestimateDiff(double z, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double kappaOld2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool Dire_fsr_ew_Q2ZQ::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z);

  double wt = 0.;
  if ( fsr->weights->hasME(fsr->makeHardEvent(0,state,true)) ) {

    // Calculate kernel as ratio of matrix elements.
    Event trialEvent(state);
    if (splitInfo.recBef()->isFinal)
      fsr->branch_FF(trialEvent, true, &splitInfo);
    else
      fsr->branch_FI(trialEvent, true, &splitInfo);

    // Calculate kernel as ratio of matrix elements.
    if ( fsr->weights->hasME(fsr->makeHardEvent(0,trialEvent,true)) ) {

      // Ensure that histories do not try to access kernel again, thus
      // potentially producing an infinite loop.
      splitInfo.addExtra("unitKernel",1.0);

      // Numerator is just the p p --> j j W matrix element.
      double wtNum = fsr->weights->getME(trialEvent);

      // Denominator is sum of all p p --> j j matrix elements.
      // --> Use History class to produce all p p --> j j phase space points.

      // Store previous settings.
      string procSave = settingsPtr->word("Merging:process");
      int nRequestedSave = settingsPtr->mode("Merging:nRequested");

      // Reset hard process to p p --> j j
      string procNow = "pp>jj";
      settingsPtr->word("Merging:process", procNow);
      fsr->mergingHooksPtr->hardProcess->clear();
      fsr->mergingHooksPtr->hardProcess->initOnProcess
        (procNow, particleDataPtr);
      fsr->mergingHooksPtr->processSave = procNow;

      // Denominator is sum of all p p --> j j matrix elements.
      fsr->mergingHooksPtr->orderHistories(false);
      Event newProcess( fsr->mergingHooksPtr->bareEvent(
        fsr->makeHardEvent(0, trialEvent, true), true) );

      // Get the maximal quark flavour counted as "additional" parton.
      int nPartons = 0;
      int nQuarksMerge = settingsPtr->mode("Merging:nQuarksMerge");
      // Loop through event and count.
      for(int i=0; i < int(newProcess.size()); ++i)
        if ( newProcess[i].isFinal()
          && newProcess[i].colType()!= 0
          && ( newProcess[i].id() == 21
               || newProcess[i].idAbs() <= nQuarksMerge))
          nPartons++;
      nPartons -= 2;

      // Set number of requested partons.
      settingsPtr->mode("Merging:nRequested", nPartons);
      fsr->mergingHooksPtr->nRequestedSave
        = settingsPtr->mode("Merging:nRequested");

      // Store candidates.
      fsr->mergingHooksPtr->storeHardProcessCandidates( newProcess );

      // Calculate number of clustering steps
      int nSteps = 1;

      // Set dummy process scale.
      newProcess.scale(0.0);
      // Generate all histories
      DireHistory myHistory( nSteps, 0.0, newProcess, DireClustering(),
        fsr->mergingHooksPtr, (*beamAPtr), (*beamBPtr), particleDataPtr,
        infoPtr,
        NULL, fsr, isr, fsr->weights, coupSMPtr, true, true,
        1.0, 1.0, 1.0, 1.0, 0);
      // Project histories onto desired branches, e.g. only ordered paths.
      myHistory.projectOntoDesiredHistories();

      double wtDen(0.);
      for ( map<double, DireHistory*>::iterator it =
              myHistory.goodBranches.begin();
        it != myHistory.goodBranches.end(); ++it ) {
        Event psppoint = it->second->state;
        wtDen += fsr->weights->getME(psppoint);
      }

      // Reset all merging settings.
      settingsPtr->word("Merging:process", procSave);
      settingsPtr->mode("Merging:nRequested",nRequestedSave);
      fsr->mergingHooksPtr->nRequestedSave
        = settingsPtr->mode("Merging:nRequested");
      fsr->mergingHooksPtr->hardProcess->initOnProcess
        (procSave, particleDataPtr);
      fsr->mergingHooksPtr->processSave = procSave;

      unordered_map<string, double>::iterator it = splitInfo.extras.find
        ("unitKernel");
      splitInfo.extras.erase(it);

      // No valid underlying processes means vanishing splitting probability.
      if (myHistory.goodBranches.size() == 0) { wtNum = 0.; wtDen = 1.; }

      wt = wtNum/wtDen;
    }
  }

  // Now multiply with (1-z) to project out Q->GQ,
  // i.e. the quark is soft and the gluon is identified.
  wt *= ( 1. - z );

  // Trivial map of values, since kernel does not depend on coupling.
  unordered_map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( unordered_map<string,double>::iterator it = wts.begin();
        it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================
// Class inheriting from Splitting class.

// Splitting function Z->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool Dire_fsr_ew_Z2QQ1::canRadiate ( const Event& state, pair<int,int> ints,
  unordered_map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints.first].isFinal()
        && state[ints.first].idAbs() == 23 );
}

int Dire_fsr_ew_Z2QQ1::kinMap()                 { return 1;}
int Dire_fsr_ew_Z2QQ1::motherID(int)            { return 1;}
int Dire_fsr_ew_Z2QQ1::sisterID(int)            { return 1;}
double Dire_fsr_ew_Z2QQ1::gaugeFactor ( int, int )        { return 1.;}
double Dire_fsr_ew_Z2QQ1::symmetryFactor ( int, int )     { return 1.0;}

int Dire_fsr_ew_Z2QQ1::radBefID(int, int)                { return 23;}
pair<int,int> Dire_fsr_ew_Z2QQ1::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

// Pick z for new splitting.
double Dire_fsr_ew_Z2QQ1::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double Dire_fsr_ew_Z2QQ1::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5 * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double Dire_fsr_ew_Z2QQ1::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool Dire_fsr_ew_Z2QQ1::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip),
    m2Rad(splitInfo.kinematics()->m2RadAft),
    m2Rec(splitInfo.kinematics()->m2Rec),
    m2Emt(splitInfo.kinematics()->m2EmtAft);
  int splitType(splitInfo.type);

  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double kappa2 = pT2/m2dip;
  wt  = preFac
      * (pow(1.-z,2.) + pow(z,2.));

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  if (doMassive) {

    double vijk = 1., pipj = 0.;

    // splitType == 2 -> Massive FF
    if (splitType == 2) {
      // Calculate CS variables.
      double yCS = kappa2 / (1.-z);
      double nu2Rad = m2Rad/m2dip;
      double nu2Emt = m2Emt/m2dip;
      double nu2Rec = m2Rec/m2dip;
      vijk          = pow2(1.-yCS) - 4.*(yCS+nu2Rad+nu2Emt)*nu2Rec;
      vijk          = sqrt(vijk) / (1-yCS);
      pipj          = m2dip * yCS /2.;

    // splitType ==-2 -> Massive FI
    } else if (splitType ==-2) {
      // Calculate CS variables.
      double xCS = 1 - kappa2/(1.-z);
      vijk   = 1.;
      pipj   = m2dip/2. * (1-xCS)/xCS;
    }

    // Reset kernel for massive splittings.
    wt = preFac * 1. / vijk * ( pow2(1.-z) + pow2(z)
                                    + m2Emt / ( pipj + m2Emt) );
  }

  // Multiply with z to project out part where emitted quark is soft,
  // and antiquark is identified.
  wt *= z;

  // Trivial map of values, since kernel does not depend on coupling.
  unordered_map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( unordered_map<string,double>::iterator it = wts.begin();
        it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from Splitting class.

// Splitting function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool Dire_fsr_ew_Z2QQ2::canRadiate ( const Event& state, pair<int,int> ints,
  unordered_map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints.first].isFinal()
        && state[ints.first].idAbs() == 23 );
}

int Dire_fsr_ew_Z2QQ2::kinMap()                 { return 1;}
int Dire_fsr_ew_Z2QQ2::motherID(int)            { return -1;}
int Dire_fsr_ew_Z2QQ2::sisterID(int)            { return -1;}
double Dire_fsr_ew_Z2QQ2::gaugeFactor ( int, int )        { return 1.;}
double Dire_fsr_ew_Z2QQ2::symmetryFactor ( int, int )     { return 1.0;}

int Dire_fsr_ew_Z2QQ2::radBefID(int, int){ return 23;}
pair<int,int> Dire_fsr_ew_Z2QQ2::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

// Pick z for new splitting.
double Dire_fsr_ew_Z2QQ2::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double Dire_fsr_ew_Z2QQ2::overestimateInt(double zMinAbs,double zMaxAbs,
  double pT2old, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double m2z = particleDataPtr->m0(23);
  wt  = 2.* preFac * 0.5 * ( zMaxAbs - zMinAbs)
      / ( pT2old - m2z);
  return wt;
}

// Return overestimate for new splitting.
double Dire_fsr_ew_Z2QQ2::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool Dire_fsr_ew_Z2QQ2::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip),
    m2Rad(splitInfo.kinematics()->m2RadAft),
    m2Rec(splitInfo.kinematics()->m2Rec),
    m2Emt(splitInfo.kinematics()->m2EmtAft);
  int splitType(splitInfo.type);

  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double kappa2 = pT2/m2dip;
  wt  = preFac
      * (pow(1.-z,2.) + pow(z,2.));

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  if (doMassive) {

    double vijk = 1., pipj = 0.;

    // splitType == 2 -> Massive FF
    if (splitType == 2) {
      // Calculate CS variables.
      double yCS = kappa2 / (1.-z);
      double nu2Rad = m2Rad/m2dip;
      double nu2Emt = m2Emt/m2dip;
      double nu2Rec = m2Rec/m2dip;
      vijk          = pow2(1.-yCS) - 4.*(yCS+nu2Rad+nu2Emt)*nu2Rec;
      vijk          = sqrt(vijk) / (1-yCS);
      pipj          = m2dip * yCS /2.;

    // splitType ==-2 -> Massive FI
    } else if (splitType ==-2) {
      // Calculate CS variables.
      double xCS = 1 - kappa2/(1.-z);
      vijk   = 1.;
      pipj   = m2dip/2. * (1-xCS)/xCS;
    }

    // Reset kernel for massive splittings.
    wt = preFac * 1. / vijk * ( pow2(1.-z) + pow2(z)
                                    + m2Emt / ( pipj + m2Emt) );
  }

  // and quark is identified.
  wt *= (1.-z);

  // Trivial map of values, since kernel does not depend on coupling.
  unordered_map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( unordered_map<string,double>::iterator it = wts.begin();
        it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================
// Class inheriting from Splitting class.

// Splitting function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool Dire_fsr_ew_W2QQ1::canRadiate ( const Event& state, pair<int,int> ints,
  unordered_map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints.first].isFinal()
        && state[ints.first].idAbs() == 24 );
}

int Dire_fsr_ew_W2QQ1::kinMap()      { return 1;}
int Dire_fsr_ew_W2QQ1::motherID(int) { return 1;} // Use 1 as dummy variable.
int Dire_fsr_ew_W2QQ1::sisterID(int) { return 1;} // Use 1 as dummy variable.
double Dire_fsr_ew_W2QQ1::gaugeFactor ( int, int )        { return 1.;}
double Dire_fsr_ew_W2QQ1::symmetryFactor ( int, int )     { return 1.0;}

int Dire_fsr_ew_W2QQ1::radBefID(int idRad, int idEmt) {
  int chg = particleDataPtr->charge(idRad) + particleDataPtr->charge(idEmt);
  if (chg > 0) return 24;
  return -24;
}

pair<int,int> Dire_fsr_ew_W2QQ1::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

// Pick z for new splitting.
double Dire_fsr_ew_W2QQ1::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double Dire_fsr_ew_W2QQ1::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5 * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double Dire_fsr_ew_W2QQ1::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool Dire_fsr_ew_W2QQ1::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z);

  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = preFac
      * (pow(1.-z,2.) + pow(z,2.));

  // Multiply with z to project out part where emitted quark is soft,
  // and antiquark is identified.
  wt *= z;

  // Trivial map of values, since kernel does not depend on coupling.
  unordered_map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( unordered_map<string,double>::iterator it = wts.begin();
        it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from Splitting class.

// Splitting function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool Dire_fsr_ew_H2AA::canRadiate ( const Event& state, pair<int,int> ints,
  unordered_map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints.first].isFinal()
        && state[ints.first].idAbs() == 25 );
}

int Dire_fsr_ew_H2AA::couplingType (int, int) { return 2; }
double Dire_fsr_ew_H2AA::coupling (double, double, double, double,
  pair<int,bool>, pair<int,bool>) {
  return widthToAA;
}

bool Dire_fsr_ew_H2AA::canRadiate ( const Event& state, int iRadBef,
  int, Settings*, PartonSystems*, BeamParticle*){
  return ( state[iRadBef].isFinal() && state[iRadBef].id() == 25);
}

int Dire_fsr_ew_H2AA::kinMap()      { return 1;}
int Dire_fsr_ew_H2AA::motherID(int) { return 22;} // Use -1 as dummy variable.
int Dire_fsr_ew_H2AA::sisterID(int) { return 22;} // Use -1 as dummy variable.
double Dire_fsr_ew_H2AA::gaugeFactor ( int, int )        { return widthToAA;}
double Dire_fsr_ew_H2AA::symmetryFactor ( int, int )     { return 1.0;}

int Dire_fsr_ew_H2AA::radBefID(int idRA, int idEA){
  if (idRA == 22 && idEA == 22) return 25;
  return 0;
}

pair<int,int> Dire_fsr_ew_H2AA::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

vector<int>Dire_fsr_ew_H2AA::recPositions(const Event& state, int iRad,
  int iEmt) {

  vector<int> recs;
  if ( !state[iRad].isFinal()
    || state[iRad].id() != 22
    || state[iEmt].id() != 22) return recs;

  // Particles to exclude as recoilers.
  vector<int> iExc(createvector<int>(iRad)(iEmt));
  // Find charged particles.
  for (int i=0; i < state.size(); ++i) {
    if ( find(iExc.begin(), iExc.end(), i) != iExc.end() ) continue;
    if ( state[i].id() == 21) {
      if (state[i].isFinal())
        recs.push_back(i);
      if (state[i].mother1() == 1 && state[i].mother2() == 0)
        recs.push_back(i);
      if (state[i].mother1() == 2 && state[i].mother2() == 0)
        recs.push_back(i);
    }
  }
  // Done.
  return recs;

}

// Pick z for new splitting.
double Dire_fsr_ew_H2AA::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double Dire_fsr_ew_H2AA::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.* preFac * 0.5 * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double Dire_fsr_ew_H2AA::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool Dire_fsr_ew_H2AA::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;
  double preFac = symmetryFactor();

  // Calculate decay width.
  double sH = splitInfo.radBef()->m2;
  double mH = sqrt(sH);
  // Calculate Breit-Wigner
  double m2Res = pow2(particleDataPtr->m0(25));
  double widthTotNow = (widthTot > 0.) ? widthTot
                     : particleDataPtr->particleDataEntryPtr(25)->resWidth
    (25,mH);
  double sigBW  = 8. * M_PI/ ( pow2(sH - m2Res) + pow2(mH * widthTotNow) );


  // Overall result.
  double wt     = preFac * sigBW * pow2(sH);

  // Trivial map of values, since kernel does not depend on coupling.
  unordered_map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( unordered_map<string,double>::iterator it = wts.begin();
        it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from Splitting class.

// Splitting function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool Dire_fsr_ew_H2GG::canRadiate ( const Event& state, pair<int,int> ints,
  unordered_map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints.first].isFinal()
        && state[ints.first].idAbs() == 25 );
}

int Dire_fsr_ew_H2GG::couplingType (int, int) { return 2; }
double Dire_fsr_ew_H2GG::coupling (double, double, double, double,
  pair<int,bool>, pair<int,bool>) {
  return widthToGG;
}

bool Dire_fsr_ew_H2GG::canRadiate ( const Event& state, int iRadBef,
  int, Settings*, PartonSystems*, BeamParticle*){
  return ( state[iRadBef].isFinal() && state[iRadBef].id() == 25);
}

int Dire_fsr_ew_H2GG::kinMap()      { return 1;}
int Dire_fsr_ew_H2GG::motherID(int) { return 21;} // Use -1 as dummy variable.
int Dire_fsr_ew_H2GG::sisterID(int) { return 21;} // Use -1 as dummy variable.
double Dire_fsr_ew_H2GG::gaugeFactor ( int, int )        { return widthToGG;}
double Dire_fsr_ew_H2GG::symmetryFactor ( int, int )     { return 1.0;}

int Dire_fsr_ew_H2GG::radBefID(int idRA, int idEA){
  if (idRA == 21 && idEA == 21) return 25;
  return 0;
}

pair<int,int> Dire_fsr_ew_H2GG::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

vector<int>Dire_fsr_ew_H2GG::recPositions(const Event& state, int iRad,
  int iEmt) {

  vector<int> recs;
  if ( !state[iRad].isFinal()
    || state[iRad].id()   != 21
    || state[iEmt].id()   != 21
    || state[iRad].col()  != state[iEmt].acol()
    || state[iRad].acol() != state[iEmt].col()) return recs;

  // Particles to exclude as recoilers.
  vector<int> iExc(createvector<int>(iRad)(iEmt));
  // Find charged particles.
  for (int i=0; i < state.size(); ++i) {
    if ( find(iExc.begin(), iExc.end(), i) != iExc.end() ) continue;
    if ( state[i].id() == 21) {
      if (state[i].isFinal())
        recs.push_back(i);
      if (state[i].mother1() == 1 && state[i].mother2() == 0)
        recs.push_back(i);
      if (state[i].mother1() == 2 && state[i].mother2() == 0)
        recs.push_back(i);
    }
  }
  // Done.
  return recs;

}

// Pick z for new splitting.
double Dire_fsr_ew_H2GG::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double Dire_fsr_ew_H2GG::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.* preFac * 0.5 * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double Dire_fsr_ew_H2GG::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool Dire_fsr_ew_H2GG::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  double preFac = symmetryFactor();
  // Calculate decay width.
  double sH = splitInfo.radBef()->m2;
  double mH = sqrt(sH);
  // Calculate Breit-Wigner
  double m2Res = pow2(particleDataPtr->m0(25));
  double widthTotNow = (widthTot > 0.) ? widthTot
                     : particleDataPtr->particleDataEntryPtr(25)->resWidth
    (25,mH);
  double sigBW  = 8. * M_PI/ ( pow2(sH - m2Res) + pow2(mH * widthTotNow) );

  // Overall result.
  double wt     = preFac * sigBW * pow2(sH);

  // Trivial map of values, since kernel does not depend on coupling.
  unordered_map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( unordered_map<string,double>::iterator it = wts.begin();
        it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}


//==========================================================================

// Class inheriting from Splitting class.

// Splitting function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool Dire_fsr_ew_H2WW::canRadiate ( const Event& state, pair<int,int> ints,
  unordered_map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints.first].isFinal()
        && state[ints.first].idAbs() == 25 );
}

int Dire_fsr_ew_H2WW::kinMap()      { return 1;}
int Dire_fsr_ew_H2WW::motherID(int) { return 24;} // Use -1 as dummy variable.
int Dire_fsr_ew_H2WW::sisterID(int) { return 24;} // Use -1 as dummy variable.
double Dire_fsr_ew_H2WW::gaugeFactor ( int, int )        { return 1.;}
double Dire_fsr_ew_H2WW::symmetryFactor ( int, int )     { return 1.0;}

int Dire_fsr_ew_H2WW::radBefID(int, int) { return 25; }

pair<int,int> Dire_fsr_ew_H2WW::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

// Pick z for new splitting.
double Dire_fsr_ew_H2WW::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double Dire_fsr_ew_H2WW::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.* preFac * 0.5 * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double Dire_fsr_ew_H2WW::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool Dire_fsr_ew_H2WW::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  Event trialEvent(state);
  if (splitInfo.recBef()->isFinal)
    fsr->branch_FF(trialEvent, true, &splitInfo);
  else
    fsr->branch_FI(trialEvent, true, &splitInfo);

  Vec4 pW1(trialEvent[trialEvent.size()-3].p());
  Vec4 pW2(trialEvent[trialEvent.size()-2].p());
  Vec4 pRec(trialEvent[trialEvent.size()-1].p());

  // Perform resonance decays.
  double m2Bef = pW1.m2Calc();
  double m2Emt = 0.0;
  double m2Rad = 0.0;
  double m2Rec = 0.0;
  double yCS   = (m2Bef - m2Emt - m2Rad)
    / (m2Bef - m2Emt - m2Rad + 2.*pW1*pRec);
  double zCS   = rndmPtr->flat();
  double phi   = 2.*M_PI*rndmPtr->flat();
  pair < Vec4, Vec4 > decayW1( fsr->decayWithOnshellRec( zCS, yCS, phi, m2Rec,
    m2Rad, m2Emt, pW1, pRec) );

  m2Bef = pW2.m2Calc();
  m2Emt = 0.0;
  m2Rad = 0.0;
  m2Rec = 0.0;
  yCS   = (m2Bef - m2Emt - m2Rad) / (m2Bef - m2Emt - m2Rad + 2.*pW2*pRec);
  zCS   = rndmPtr->flat();
  phi   = 2.*M_PI*rndmPtr->flat();
  pair < Vec4, Vec4 > decayW2( fsr->decayWithOnshellRec( zCS, yCS, phi, m2Rec,
    m2Rad, m2Emt, pW2, pRec) );

  if (false) cout << decayW1.first << decayW1.second
                  << decayW2.first << decayW2.second;

  double wt = 0.;
  // Trivial map of values, since kernel does not depend on coupling.
  unordered_map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( unordered_map<string,double>::iterator it = wts.begin();
        it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from Splitting class.

// Splitting function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool Dire_fsr_ew_W2QQ2::canRadiate ( const Event& state, pair<int,int> ints,
  unordered_map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints.first].isFinal()
        && state[ints.first].idAbs() == 24 );
}

int Dire_fsr_ew_W2QQ2::kinMap()      { return 1;}
int Dire_fsr_ew_W2QQ2::motherID(int) { return -1;} // Use -1 as dummy variable.
int Dire_fsr_ew_W2QQ2::sisterID(int) { return -1;} // Use -1 as dummy variable.
double Dire_fsr_ew_W2QQ2::gaugeFactor ( int, int )        { return 1.;}
double Dire_fsr_ew_W2QQ2::symmetryFactor ( int, int )     { return 1.0;}

int Dire_fsr_ew_W2QQ2::radBefID(int idRad, int idEmt) {
  int chg = particleDataPtr->charge(idRad) + particleDataPtr->charge(idEmt);
  if (chg > 0) return 24;
  return -24;
}

pair<int,int> Dire_fsr_ew_W2QQ2::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

// Pick z for new splitting.
double Dire_fsr_ew_W2QQ2::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double Dire_fsr_ew_W2QQ2::overestimateInt(double zMinAbs,double zMaxAbs,
  double pT2old, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double m2z = particleDataPtr->m0(23);
  wt  = 2.* preFac * 0.5 * ( zMaxAbs - zMinAbs)
      / ( pT2old - m2z);
  return wt;
}

// Return overestimate for new splitting.
double Dire_fsr_ew_W2QQ2::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool Dire_fsr_ew_W2QQ2::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z);

  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = preFac
      * (pow(1.-z,2.) + pow(z,2.));

  // and quark is identified.
  wt *= (1.-z);

  // Trivial map of values, since kernel does not depend on coupling.
  unordered_map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( unordered_map<string,double>::iterator it = wts.begin();
        it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Return true if this kernel should partake in the evolution.
bool Dire_fsr_ew_W2WA::canRadiate ( const Event& state, pair<int,int> ints,
  unordered_map<string,bool> bools, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints.first].isFinal()
        && state[ints.first].idAbs() == 24
        && state[ints.second].isCharged()
        && (bools["doQEDshowerByL"] || bools["doQEDshowerByQ"]));
}

bool Dire_fsr_ew_W2WA::canRadiate ( const Event& state, int iRadBef,
  int iRecBef, Settings*, PartonSystems*, BeamParticle*){
  return ( state[iRadBef].isFinal()
        && state[iRadBef].idAbs() == 24
        && state[iRecBef].isCharged()
        && (doQEDshowerByL || doQEDshowerByQ));
}

int Dire_fsr_ew_W2WA::kinMap()                 { return 1;}
int Dire_fsr_ew_W2WA::motherID(int idDaughter) { return idDaughter;}
int Dire_fsr_ew_W2WA::sisterID(int)            { return 22;}

double Dire_fsr_ew_W2WA::gaugeFactor ( int idRadBef, int idRecBef) {
  double chgRad = particleDataPtr->charge(idRadBef);
  double chgRec = particleDataPtr->charge(idRecBef);
  double charge = -1.*chgRad*chgRec;
  if (!splitInfo.radBef()->isFinal) charge *= -1.;
  if (!splitInfo.recBef()->isFinal) charge *= -1.;
  if (idRadBef != 0 && idRecBef != 0) return charge;
  // Set probability to zero.
  return 0.;
}

double Dire_fsr_ew_W2WA::symmetryFactor ( int, int ) { return 1.;}

int Dire_fsr_ew_W2WA::radBefID(int idRad, int idEmt) {
  if (idEmt == 22 && abs(idRad) == 24) return idRad;
  return 0;
}

pair<int,int> Dire_fsr_ew_W2WA::radBefCols(int, int, int, int) {
  return make_pair(0,0);
}

vector<int>Dire_fsr_ew_W2WA::recPositions(const Event&, int, int) {
  // Not yet implemented.
  vector<int> recs;
  return recs;
}

// Pick z for new splitting.
double Dire_fsr_ew_W2WA::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTminChgL"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double Dire_fsr_ew_W2WA::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {
  double wt = 0.;
  double charge = gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id);
  double preFac = symmetryFactor() * abs(charge);
  // Q -> QG, soft part (currently also used for collinear part).
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTminChgL"))/m2dip;
  wt  = enhance * preFac * 2. * 0.5 * log1p(pow2(1.-zMinAbs)/kappa2);
  return wt;
}

// Return overestimate for new splitting.
double Dire_fsr_ew_W2WA::overestimateDiff(double z, double m2dip, int) {
  double wt = 0.;
  double charge = gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id);
  double preFac = symmetryFactor() * abs(charge);
  double kappaOld2 = pow2(settingsPtr->parm("TimeShower:pTminChgL"))/m2dip;
  wt  = enhance * preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool Dire_fsr_ew_W2WA::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip),
    m2RadBef(splitInfo.kinematics()->m2RadBef),
    m2Rad(splitInfo.kinematics()->m2RadAft),
    m2Rec(splitInfo.kinematics()->m2Rec),
    m2Emt(splitInfo.kinematics()->m2EmtAft);
  int splitType(splitInfo.type);

  // Calculate kernel.
  // Note: We are calculating the z <--> 1-z symmetrised kernel here,
  // and later multiply with z to project out Q->QQ,
  // i.e. the gluon is soft and the quark is identified.
  double wt = 0.;

  double chargeFac = gaugeFactor(splitInfo.radBef()->id,
                                 splitInfo.recBef()->id);
  double preFac = symmetryFactor() * chargeFac;
  double kappa2 = pT2/m2dip;
  wt   = preFac * ( 2. * z * (1.-z) / ( pow2(1.-z) + kappa2) );

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  // Add collinear term for massless splittings.
  if (!doMassive && orderNow >= 0) wt  += preFac * ( 1.-z );

  // Add collinear term for massive splittings.
  if (doMassive && orderNow >= 0) {

    double pipj = 0., vijkt = 1., vijk = 1.;

    // splitType == 2 -> Massive FF
    if (splitType == 2) {

      // Calculate CS variables.
      double yCS = kappa2 / (1.-z);
      double nu2RadBef = m2RadBef/m2dip;
      double nu2Rad = m2Rad/m2dip;
      double nu2Emt = m2Emt/m2dip;
      double nu2Rec = m2Rec/m2dip;
      vijk          = pow2(1.-yCS) - 4.*(yCS+nu2Rad+nu2Emt)*nu2Rec;
      double Q2mass = m2dip + m2Rad + m2Rec + m2Emt;
      vijkt         = pow2(Q2mass/m2dip - nu2RadBef - nu2Rec)
                    - 4.*nu2RadBef*nu2Rec;
      vijk          = sqrt(vijk) / (1-yCS);
      vijkt         = sqrt(vijkt)/ (Q2mass/m2dip - nu2RadBef - nu2Rec);
      pipj          = m2dip * yCS/2.;
    // splitType ==-2 -> Massive FI
    } else if (splitType ==-2) {

      // Calculate CS variables.
      double xCS = 1 - kappa2/(1.-z);
      vijk   = 1.;
      vijkt  = 1.;
      pipj   = m2dip/2. * (1-xCS)/xCS;
    }

    // Add B1 for massive splittings.
    double massCorr = vijkt/vijk*( 1. - z - m2RadBef/pipj);
    wt += preFac*massCorr;

  }

  if (orderNow < 0 && chargeFac < 0.) wt = 0.;

  // Trivial map of values, since kernel does not depend on coupling.
  unordered_map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( unordered_map<string,double>::iterator it = wts.begin();
        it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function Q->QG (ISR)

// Return true if this kernel should partake in the evolution.
bool Dire_isr_ew_Q2QZ::canRadiate ( const Event& state, pair<int,int> ints,
  unordered_map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  int nFinPartons(0), nFinOther(0);
  for(int i=0; i < state.size(); ++i) {
    if (!state[i].isFinal()) continue;
    if ( state[i].colType() !=0)
      nFinPartons++;
    else nFinOther++;
  }
  return ( nFinPartons == 2 && nFinOther == 0
        && !state[ints.first].isFinal()
        &&  state[ints.first].isQuark() );
}

int Dire_isr_ew_Q2QZ::kinMap()                 { return 1;}
int Dire_isr_ew_Q2QZ::motherID(int idDaughter) { return idDaughter;}
int Dire_isr_ew_Q2QZ::sisterID(int)            { return 23;}
double Dire_isr_ew_Q2QZ::gaugeFactor ( int, int )        { return thetaW;}
double Dire_isr_ew_Q2QZ::symmetryFactor ( int, int )     { return 1.;}

int Dire_isr_ew_Q2QZ::radBefID(int idRA, int){ return idRA;}
pair<int,int> Dire_isr_ew_Q2QZ::radBefCols(
  int colRadAfter, int acolRadAfter,
  int , int ) {
  return make_pair(colRadAfter,acolRadAfter);
}

// Pick z for new splitting.
double Dire_isr_ew_Q2QZ::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double Dire_isr_ew_Q2QZ::overestimateInt(double zMinAbs, double,
  double, double m2dip, int ) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  wt  = preFac * 2. * 0.5 * log1p(pow2(1.-zMinAbs)/kappa2);
  return wt;
}

// Return overestimate for new splitting.
double Dire_isr_ew_Q2QZ::overestimateDiff(double z, double m2dip, int ) {
  double wt        = 0.;
  double preFac    = symmetryFactor() * gaugeFactor();
  double kappaOld2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  wt  = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool Dire_isr_ew_Q2QZ::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  double wt = 0.;
  if ( isr->weights->hasME(isr->makeHardEvent(0,state,true)) ) {

    // Calculate kernel as ratio of matrix elements.
    Event trialEvent(state);
    if (splitInfo.recBef()->isFinal)
      isr->branch_IF(trialEvent, true, &splitInfo);
    else
      isr->branch_II(trialEvent, true, &splitInfo);

    // Calculate kernel as ratio of matrix elements.
    if ( isr->weights->hasME(isr->makeHardEvent(0,trialEvent,true)) ) {

      // Ensure that histories do not try to access kernel again, thus
      // potentially producing an infinite loop.
      splitInfo.addExtra("unitKernel",1.0);

      // Numerator is just the p p --> j j W matrix element.
      double wtNum = isr->weights->getME(trialEvent);

      // Denominator is sum of all p p --> j j matrix elements.
      // --> Use History class to produce all p p --> j j phase space points.

      // Store previous settings.
      string procSave = settingsPtr->word("Merging:process");
      int nRequestedSave = settingsPtr->mode("Merging:nRequested");

      // Reset hard process to p p --> j j
      string procNow = "pp>jj";
      settingsPtr->word("Merging:process", procNow);
      isr->mergingHooksPtr->hardProcess->clear();
      isr->mergingHooksPtr->hardProcess->
        initOnProcess(procNow, particleDataPtr);
      isr->mergingHooksPtr->processSave = procNow;

      // Denominator is sum of all p p --> j j matrix elements.
      isr->mergingHooksPtr->orderHistories(false);
      Event newProcess( isr->mergingHooksPtr->bareEvent(
        isr->makeHardEvent(0, trialEvent, true), true) );

      // Get the maximal quark flavour counted as "additional" parton.
      int nPartons = 0;
      int nQuarksMerge = settingsPtr->mode("Merging:nQuarksMerge");
      // Loop through event and count.
      for(int i=0; i < int(newProcess.size()); ++i)
        if ( newProcess[i].isFinal()
          && newProcess[i].colType()!= 0
          && ( newProcess[i].id() == 21
               || newProcess[i].idAbs() <= nQuarksMerge))
          nPartons++;
      nPartons -= 2;

      // Set number of requested partons.
      settingsPtr->mode("Merging:nRequested", nPartons);
      isr->mergingHooksPtr->nRequestedSave
        = settingsPtr->mode("Merging:nRequested");

      // Store candidates.
      isr->mergingHooksPtr->storeHardProcessCandidates( newProcess );

      // Calculate number of clustering steps
      int nSteps = 1;

      // Set dummy process scale.
      newProcess.scale(0.0);
      // Generate all histories
      DireHistory myHistory( nSteps, 0.0, newProcess, DireClustering(),
        isr->mergingHooksPtr, (*beamAPtr), (*beamBPtr), particleDataPtr,
        infoPtr,
        NULL, fsr, isr, isr->weights, coupSMPtr, true, true,
        1.0, 1.0, 1.0, 1.0, 0);
      // Project histories onto desired branches, e.g. only ordered paths.
      myHistory.projectOntoDesiredHistories();

      double wtDen(0.);
      for ( map<double, DireHistory*>::iterator it =
              myHistory.goodBranches.begin();
        it != myHistory.goodBranches.end(); ++it ) {
        Event psppoint = it->second->state;
        wtDen += isr->weights->getME(psppoint);
      }

      // Reset all merging settings.
      settingsPtr->word("Merging:process", procSave);
      settingsPtr->mode("Merging:nRequested",nRequestedSave);
      isr->mergingHooksPtr->nRequestedSave
        = settingsPtr->mode("Merging:nRequested");
      isr->mergingHooksPtr->hardProcess->initOnProcess
        (procSave, particleDataPtr);
      isr->mergingHooksPtr->processSave = procSave;

      unordered_map<string, double>::iterator it =
        splitInfo.extras.find("unitKernel");
      splitInfo.extras.erase(it);

      // No valid underlying processes means vanishing splitting probability.
      if (myHistory.goodBranches.size() == 0) { wtNum = 0.; wtDen = 1.; }

      wt = wtNum/wtDen;
    }
  }

  unordered_map<string,double> wts;
  wts.insert( make_pair("base", wt) );

  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", wt));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp",   wt));
  }

  // Store kernel values.
  clearKernels();
  for ( unordered_map<string,double>::iterator it = wts.begin();
        it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================


} // end namespace Pythia8
