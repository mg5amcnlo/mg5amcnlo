
#include "Dire/SplittingsEW.h"
#include "Dire/DireSpace.h"
#include "Dire/DireTimes.h"
#include "Dire/History.h"

namespace Pythia8 {

//==========================================================================

// The SplittingEW class.

//--------------------------------------------------------------------------

void SplittingEW::init() {

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

}

//--------------------------------------------------------------------------

// Function to calculate the correct alphaem/2*Pi value, including
// renormalisation scale variations + threshold matching.

double SplittingEW::aem2Pi( double pT2 ) {

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
bool fsr_ew_Q2QZ::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  int nFinPartons(0), nFinQ(0), nFinOther(0);
  for(int i=0; i < state.size(); ++i) {
    if (!state[i].isFinal()) continue;
    if ( state[i].colType() !=0) {
      nFinPartons++;
      if ( abs(state[i].colType()) == 1) nFinQ++;
    } else nFinOther++;
  }
  return ( nFinPartons == 2 && nFinQ > 0 && nFinOther == 0
        && state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].isQuark() );
}

int fsr_ew_Q2QZ::kinMap()                        { return 1;}
int fsr_ew_Q2QZ::motherID(int idDaughter)        { return idDaughter;}
int fsr_ew_Q2QZ::sisterID(int)                   { return 23;}
double fsr_ew_Q2QZ::gaugeFactor ( int, int)      { return thetaW; }
double fsr_ew_Q2QZ::symmetryFactor ( int, int )  { return 1.;}
int fsr_ew_Q2QZ::radBefID(int idRA, int)            { return idRA;}

pair<int,int> fsr_ew_Q2QZ::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int, int) { return make_pair(colRadAfter,acolRadAfter); }

// Pick z for new splitting.
double fsr_ew_Q2QZ::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_ew_Q2QZ::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  // Q -> QG, soft part (currently also used for collinear part).
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappa2);
  return wt;
}

// Return overestimate for new splitting.
double fsr_ew_Q2QZ::overestimateDiff(double z, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double kappaOld2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool fsr_ew_Q2QZ::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z);
  //  pT2(splitInfo.kinematics()->pT2),
  //  m2dip(splitInfo.kinematics()->m2Dip),
  //  m2RadBef(splitInfo.kinematics()->m2RadBef),
  //  m2Rad(splitInfo.kinematics()->m2RadAft),
  //  m2Rec(splitInfo.kinematics()->m2Rec),
  //  m2Emt(splitInfo.kinematics()->m2EmtAft);
  //int splitType(splitInfo.type);

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
      splitInfo.extras.insert(make_pair("unitKernel",1.0));

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
      fsr->mergingHooksPtr->hardProcess->initOnProcess(procNow, particleDataPtr);
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
          && ( newProcess[i].id() == 21 || newProcess[i].idAbs() <= nQuarksMerge))
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
      MyHistory myHistory( nSteps, 0.0, newProcess, MyClustering(),
        fsr->mergingHooksPtr, (*beamAPtr), (*beamBPtr), particleDataPtr, infoPtr,
        NULL, fsr, isr, fsr->weights, coupSMPtr, true, true, true, true, 1.0, 1.0, 1.0, 0);
      // Project histories onto desired branches, e.g. only ordered paths.
      myHistory.projectOntoDesiredHistories();

      double wtDen(0.);
      for ( map<double, MyHistory*>::iterator it = myHistory.goodBranches.begin();
        it != myHistory.goodBranches.end(); ++it ) {
        Event psppoint = it->second->state;
        //psppoint.list();
        wtDen += fsr->weights->getME(psppoint);
      }

      // Reset all merging settings.
      settingsPtr->word("Merging:process", procSave);
      settingsPtr->mode("Merging:nRequested",nRequestedSave);
      fsr->mergingHooksPtr->nRequestedSave
        = settingsPtr->mode("Merging:nRequested");
      fsr->mergingHooksPtr->hardProcess->initOnProcess(procSave, particleDataPtr);
      fsr->mergingHooksPtr->processSave = procSave;

      map<string, double>::iterator it = splitInfo.extras.find("unitKernel");
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
  map<string,double> wts;
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
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from Splitting class.

// Splitting function Q->ZQ (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_ew_Q2ZQ::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  int nFinPartons(0), nFinQ(0), nFinOther(0);
  for(int i=0; i < state.size(); ++i) {
    if (!state[i].isFinal()) continue;
    if ( state[i].colType() !=0) {
      nFinPartons++;
      if ( abs(state[i].colType()) == 1) nFinQ++;
    } else nFinOther++;
  }
  return ( nFinPartons == 2 && nFinQ > 0 && nFinOther == 0
        && state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].isQuark() );
}

int fsr_ew_Q2ZQ::kinMap()                        { return 1;}
int fsr_ew_Q2ZQ::motherID(int idDaughter)        { return idDaughter;}
int fsr_ew_Q2ZQ::sisterID(int)                   { return 23;}
double fsr_ew_Q2ZQ::gaugeFactor ( int, int)      { return thetaW; }
double fsr_ew_Q2ZQ::symmetryFactor ( int, int )  { return 1.;}
int fsr_ew_Q2ZQ::radBefID(int idRA, int)            { return idRA;}

pair<int,int> fsr_ew_Q2ZQ::radBefCols( int colRadAfter, int acolRadAfter, 
  int, int) { return make_pair(colRadAfter,acolRadAfter); }

// Pick z for new splitting.
double fsr_ew_Q2ZQ::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_ew_Q2ZQ::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  // Q -> QG, soft part (currently also used for collinear part).
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappa2);
  return wt;
}

// Return overestimate for new splitting.
double fsr_ew_Q2ZQ::overestimateDiff(double z, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double kappaOld2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool fsr_ew_Q2ZQ::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z);
  //  pT2(splitInfo.kinematics()->pT2),
  //  m2dip(splitInfo.kinematics()->m2Dip),
  //  m2RadBef(splitInfo.kinematics()->m2RadBef),
  //  m2Rad(splitInfo.kinematics()->m2RadAft),
  //  m2Rec(splitInfo.kinematics()->m2Rec),
  //  m2Emt(splitInfo.kinematics()->m2EmtAft);
  //int splitType(splitInfo.type);

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
      splitInfo.extras.insert(make_pair("unitKernel",1.0));

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
      fsr->mergingHooksPtr->hardProcess->initOnProcess(procNow, particleDataPtr);
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
          && ( newProcess[i].id() == 21 || newProcess[i].idAbs() <= nQuarksMerge))
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
      MyHistory myHistory( nSteps, 0.0, newProcess, MyClustering(),
        fsr->mergingHooksPtr, (*beamAPtr), (*beamBPtr), particleDataPtr, infoPtr,
        NULL, fsr, isr, fsr->weights, coupSMPtr, true, true, true, true, 1.0, 1.0, 1.0, 0);
      // Project histories onto desired branches, e.g. only ordered paths.
      myHistory.projectOntoDesiredHistories();

      double wtDen(0.);
      for ( map<double, MyHistory*>::iterator it = myHistory.goodBranches.begin();
        it != myHistory.goodBranches.end(); ++it ) {
        Event psppoint = it->second->state;
        //psppoint.list();
        wtDen += fsr->weights->getME(psppoint);
      }

      // Reset all merging settings.
      settingsPtr->word("Merging:process", procSave);
      settingsPtr->mode("Merging:nRequested",nRequestedSave);
      fsr->mergingHooksPtr->nRequestedSave
        = settingsPtr->mode("Merging:nRequested");
      fsr->mergingHooksPtr->hardProcess->initOnProcess(procSave, particleDataPtr);
      fsr->mergingHooksPtr->processSave = procSave;

      map<string, double>::iterator it = splitInfo.extras.find("unitKernel");
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
  map<string,double> wts;
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
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================
// Class inheriting from Splitting class.

// Splitting function Z->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_ew_Z2QQ1::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].idAbs() == 23 );
}

int fsr_ew_Z2QQ1::kinMap()                 { return 1;}
int fsr_ew_Z2QQ1::motherID(int)            { return 1;} // Use 1 as dummy variable.
int fsr_ew_Z2QQ1::sisterID(int)            { return 1;} // Use 1 as dummy variable.
double fsr_ew_Z2QQ1::gaugeFactor ( int, int )        { return 1.;}
double fsr_ew_Z2QQ1::symmetryFactor ( int, int )     { return 1.0;}

int fsr_ew_Z2QQ1::radBefID(int, int)                { return 23;}
pair<int,int> fsr_ew_Z2QQ1::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

// Pick z for new splitting.
double fsr_ew_Z2QQ1::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double fsr_ew_Z2QQ1::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5 * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double fsr_ew_Z2QQ1::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool fsr_ew_Z2QQ1::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip),
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
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
  map<string,double> wts;
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
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from Splitting class.

// Splitting function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_ew_Z2QQ2::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].idAbs() == 23 );
}

int fsr_ew_Z2QQ2::kinMap()                 { return 1;}
int fsr_ew_Z2QQ2::motherID(int)            { return -1;} // Use -1 as dummy variable.
int fsr_ew_Z2QQ2::sisterID(int)            { return -1;} // Use -1 as dummy variable.
double fsr_ew_Z2QQ2::gaugeFactor ( int, int )        { return 1.;}
double fsr_ew_Z2QQ2::symmetryFactor ( int, int )     { return 1.0;}

int fsr_ew_Z2QQ2::radBefID(int, int){ return 23;}
pair<int,int> fsr_ew_Z2QQ2::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

// Pick z for new splitting.
double fsr_ew_Z2QQ2::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double fsr_ew_Z2QQ2::overestimateInt(double zMinAbs,double zMaxAbs,
  double pT2old, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double m2z = particleDataPtr->m0(23);
  wt  = 2.* preFac * 0.5 * ( zMaxAbs - zMinAbs)
      / ( pT2old - m2z);
  return wt;
}

// Return overestimate for new splitting.
double fsr_ew_Z2QQ2::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool fsr_ew_Z2QQ2::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip),
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
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
  map<string,double> wts;
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
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================
// Class inheriting from Splitting class.

// Splitting function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_ew_W2QQ1::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].idAbs() == 24 );
}

int fsr_ew_W2QQ1::kinMap()                 { return 1;}
int fsr_ew_W2QQ1::motherID(int)            { return 1;} // Use 1 as dummy variable.
int fsr_ew_W2QQ1::sisterID(int)            { return 1;} // Use 1 as dummy variable.
double fsr_ew_W2QQ1::gaugeFactor ( int, int )        { return 1.;}
double fsr_ew_W2QQ1::symmetryFactor ( int, int )     { return 1.0;}

int fsr_ew_W2QQ1::radBefID(int idRad, int idEmt) {
  int chg = particleDataPtr->charge(idRad) + particleDataPtr->charge(idEmt);
  if (chg > 0) return 24;
  return -24;
}

pair<int,int> fsr_ew_W2QQ1::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

// Pick z for new splitting.
double fsr_ew_W2QQ1::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double fsr_ew_W2QQ1::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5 * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double fsr_ew_W2QQ1::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool fsr_ew_W2QQ1::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z); //pT2(splitInfo.kinematics()->pT2)
    //m2dip(splitInfo.kinematics()->m2Dip),
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    //m2Rec(splitInfo.kinematics()->m2Rec),
    //m2Emt(splitInfo.kinematics()->m2EmtAft);
  //int splitType(splitInfo.type);

  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = preFac 
      * (pow(1.-z,2.) + pow(z,2.));

  // Multiply with z to project out part where emitted quark is soft,
  // and antiquark is identified.
  wt *= z;

  // Trivial map of values, since kernel does not depend on coupling.
  map<string,double> wts;
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
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from Splitting class.

// Splitting function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_ew_H2WW::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].idAbs() == 25 );
}

int fsr_ew_H2WW::kinMap()                 { return 1;}
int fsr_ew_H2WW::motherID(int)            { return 24;} // Use -1 as dummy variable.
int fsr_ew_H2WW::sisterID(int)            { return 24;} // Use -1 as dummy variable.
double fsr_ew_H2WW::gaugeFactor ( int, int )        { return 1.;}
double fsr_ew_H2WW::symmetryFactor ( int, int )     { return 1.0;}

int fsr_ew_H2WW::radBefID(int, int) { return 25; }

pair<int,int> fsr_ew_H2WW::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

// Pick z for new splitting.
double fsr_ew_H2WW::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double fsr_ew_H2WW::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.* preFac * 0.5 * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double fsr_ew_H2WW::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool fsr_ew_H2WW::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  //double z(splitInfo.kinematics()->z); //pT2(splitInfo.kinematics()->pT2),
    //m2dip(splitInfo.kinematics()->m2Dip),
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    //m2Rec(splitInfo.kinematics()->m2Rec),
    //m2Emt(splitInfo.kinematics()->m2EmtAft);
  //int splitType(splitInfo.type);

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
  double yCS   = (m2Bef - m2Emt - m2Rad) / (m2Bef - m2Emt - m2Rad + 2.*pW1*pRec);
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

//state.list();
//trialEvent.list();
//cout << pW1 << decayW1.first + decayW1.second << decayW1.first << decayW1.second;
//cout << pW2 << decayW2.first + decayW2.second << decayW2.first << decayW2.second;
//exit(0);

  double wt = 0.;

  // Trivial map of values, since kernel does not depend on coupling.
  map<string,double> wts;
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
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from Splitting class.

// Splitting function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_ew_W2QQ2::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].idAbs() == 24 );
}

int fsr_ew_W2QQ2::kinMap()                 { return 1;}
int fsr_ew_W2QQ2::motherID(int)            { return -1;} // Use -1 as dummy variable.
int fsr_ew_W2QQ2::sisterID(int)            { return -1;} // Use -1 as dummy variable.
double fsr_ew_W2QQ2::gaugeFactor ( int, int )        { return 1.;}
double fsr_ew_W2QQ2::symmetryFactor ( int, int )     { return 1.0;}

int fsr_ew_W2QQ2::radBefID(int idRad, int idEmt) {
  int chg = particleDataPtr->charge(idRad) + particleDataPtr->charge(idEmt);
  if (chg > 0) return 24;
  return -24;
}

pair<int,int> fsr_ew_W2QQ2::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

// Pick z for new splitting.
double fsr_ew_W2QQ2::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double fsr_ew_W2QQ2::overestimateInt(double zMinAbs,double zMaxAbs,
  double pT2old, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double m2z = particleDataPtr->m0(23);
  wt  = 2.* preFac * 0.5 * ( zMaxAbs - zMinAbs)
      / ( pT2old - m2z);
  return wt;
}

// Return overestimate for new splitting.
double fsr_ew_W2QQ2::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool fsr_ew_W2QQ2::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z); //pT2(splitInfo.kinematics()->pT2),
    //m2dip(splitInfo.kinematics()->m2Dip),
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    //m2Rec(splitInfo.kinematics()->m2Rec),
    //m2Emt(splitInfo.kinematics()->m2EmtAft);
  //int splitType(splitInfo.type);

  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = preFac 
      * (pow(1.-z,2.) + pow(z,2.));

  // and quark is identified.
  wt *= (1.-z);

  // Trivial map of values, since kernel does not depend on coupling.
  map<string,double> wts;
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
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function Q->QG (ISR)

// Return true if this kernel should partake in the evolution.
bool isr_ew_Q2QZ::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  int nFinPartons(0), nFinOther(0);
  for(int i=0; i < state.size(); ++i) {
    if (!state[i].isFinal()) continue;
    if ( state[i].colType() !=0)
      nFinPartons++;
    else nFinOther++;
  }
  return ( nFinPartons == 2 && nFinOther == 0
        && !state[ints["iRad"]].isFinal()
        &&  state[ints["iRad"]].isQuark() );
}

int isr_ew_Q2QZ::kinMap()                 { return 1;}
int isr_ew_Q2QZ::motherID(int idDaughter) { return idDaughter;} 
int isr_ew_Q2QZ::sisterID(int)            { return 23;} 
double isr_ew_Q2QZ::gaugeFactor ( int, int )        { return thetaW;}
double isr_ew_Q2QZ::symmetryFactor ( int, int )     { return 1.;}

int isr_ew_Q2QZ::radBefID(int idRA, int){ return idRA;}
pair<int,int> isr_ew_Q2QZ::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int , int ) {
  return make_pair(colRadAfter,acolRadAfter); 
}

// Pick z for new splitting.
double isr_ew_Q2QZ::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double isr_ew_Q2QZ::overestimateInt(double zMinAbs, double,
  double, double m2dip, int ) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  wt  = preFac * 2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappa2);
  return wt;
}

// Return overestimate for new splitting.
double isr_ew_Q2QZ::overestimateDiff(double z, double m2dip, int ) {
  double wt        = 0.;
  double preFac    = symmetryFactor() * gaugeFactor();
  double kappaOld2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  wt  = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool isr_ew_Q2QZ::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  //double z(splitInfo.kinematics()->z),
    //pT2(splitInfo.kinematics()->pT2),
    //m2dip(splitInfo.kinematics()->m2Dip),
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    //m2Rec(splitInfo.kinematics()->m2Rec),
    //m2Emt(splitInfo.kinematics()->m2EmtAft);
  //int splitType(splitInfo.type);

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
      splitInfo.extras.insert(make_pair("unitKernel",1.0));

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
      isr->mergingHooksPtr->hardProcess->initOnProcess(procNow, particleDataPtr);
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
          && ( newProcess[i].id() == 21 || newProcess[i].idAbs() <= nQuarksMerge))
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
      MyHistory myHistory( nSteps, 0.0, newProcess, MyClustering(),
        isr->mergingHooksPtr, (*beamAPtr), (*beamBPtr), particleDataPtr, infoPtr,
        NULL, fsr, isr, isr->weights, coupSMPtr, true, true, true, true, 1.0, 1.0, 1.0, 0);
      // Project histories onto desired branches, e.g. only ordered paths.
      myHistory.projectOntoDesiredHistories();

      double wtDen(0.);
      for ( map<double, MyHistory*>::iterator it = myHistory.goodBranches.begin();
        it != myHistory.goodBranches.end(); ++it ) {
        Event psppoint = it->second->state;
        //psppoint.list();
        wtDen += isr->weights->getME(psppoint);
      }

      // Reset all merging settings.
      settingsPtr->word("Merging:process", procSave);
      settingsPtr->mode("Merging:nRequested",nRequestedSave);
      isr->mergingHooksPtr->nRequestedSave
        = settingsPtr->mode("Merging:nRequested");
      isr->mergingHooksPtr->hardProcess->initOnProcess(procSave, particleDataPtr);
      isr->mergingHooksPtr->processSave = procSave;

      map<string, double>::iterator it = splitInfo.extras.find("unitKernel");
      splitInfo.extras.erase(it);

      // No valid underlying processes means vanishing splitting probability.
      if (myHistory.goodBranches.size() == 0) { wtNum = 0.; wtDen = 1.; }

      wt = wtNum/wtDen;
    }
  }

  map<string,double> wts;
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
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================


} // end namespace Pythia8
