
#include "Dire/SplittingsQED.h"
#include "Dire/DireSpace.h"
#include "Dire/DireTimes.h"

namespace Pythia8 {

//==========================================================================

// The SplittingQED class.

//--------------------------------------------------------------------------

void SplittingQED::init() {

  int nGammaToQuark  = settingsPtr->mode("TimeShower:nGammaToQuark");
  int nGammaToLepton = settingsPtr->mode("TimeShower:nGammaToLepton");

  sumCharge2L        = max(0, min(3, nGammaToLepton));
  sumCharge2Q        = 0.;
  if      (nGammaToQuark > 4) sumCharge2Q = 11. / 9.;
  else if (nGammaToQuark > 3) sumCharge2Q = 10. / 9.;
  else if (nGammaToQuark > 2) sumCharge2Q =  6. / 9.;
  else if (nGammaToQuark > 1) sumCharge2Q =  5. / 9.;
  else if (nGammaToQuark > 0) sumCharge2Q =  1. / 9.;
  sumCharge2Tot      = sumCharge2L + 3. * sumCharge2Q;

  // Parameters of alphaEM.
  int alphaEMorder = settingsPtr->mode("SpaceShower:alphaEMorder");
  // Initialize alphaEM.
  alphaEM.init( alphaEMorder, settingsPtr);

  aem0 = settingsPtr->parm("StandardModel:alphaEM0");

}

//--------------------------------------------------------------------------

// Function to calculate the correct alphaem/2*Pi value, including
// renormalisation scale variations + threshold matching.

double SplittingQED::aem2Pi( double pT2, int ) {

  double scale       = pT2*renormMultFac;

  // Get alphaEM(k*pT^2) and subtractions.
  double aemPT2pi = alphaEM.alphaEM(scale) / (2.*M_PI);

  // Done.
  return aemPT2pi;

}

//==========================================================================

// Class inheriting from SplittingQED class.

// SplittingQED function Q->QG (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_qed_Q2QA::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].isQuark() && state[ints["iRec"]].isCharged() );
}

int fsr_qed_Q2QA::kinMap()                 { return 1;}
int fsr_qed_Q2QA::motherID(int idDaughter) { return idDaughter;}
int fsr_qed_Q2QA::sisterID(int)            { return 22;}

double fsr_qed_Q2QA::gaugeFactor ( int idRadBef, int idRecBef) {
  double chgRad = particleDataPtr->charge(idRadBef);
  double chgRec = particleDataPtr->charge(idRecBef);
  double charge = -1.*chgRad*chgRec;
  if (idRadBef != 0 && idRecBef != 0) return charge;
  // Set probability to zero.
  return 0.;
}

double fsr_qed_Q2QA::symmetryFactor ( int, int ) { return 1.;}

int fsr_qed_Q2QA::radBefID(int idRad, int idEmt) {
  if (particleDataPtr->isQuark(idRad) && idEmt == 22 ) return idRad;
  return 0;
}

pair<int,int> fsr_qed_Q2QA::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int, int) { return make_pair(colRadAfter,acolRadAfter); }

vector<int>fsr_qed_Q2QA::recPositions(const Event& state, int iRad, int iEmt) {

  vector<int> recs;
  if ( !state[iRad].isFinal()
    || !state[iRad].isQuark() 
    || state[iEmt].id() != 22) return recs;

  // Particles to exclude as recoilers.
  vector<int> iExc(createvector<int>(iRad)(iEmt));
  // Find charged particles.
  for (int i=0; i < state.size(); ++i) {
    if ( find(iExc.begin(), iExc.end(), i) != iExc.end() ) continue;
    if ( state[i].isCharged() ) {
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
double fsr_qed_Q2QA::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qed_Q2QA::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * abs(gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id));
  // Q -> QG, soft part (currently also used for collinear part).
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappa2);

  return wt;
}

// Return overestimate for new splitting.
double fsr_qed_Q2QA::overestimateDiff(double z, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * abs(gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id));
  double kappaOld2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool fsr_qed_Q2QA::calc(const Event& state, int orderNow) { 

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

  double chargeFac = gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id);
  if ( !state[splitInfo.iRecBef].isFinal() ) chargeFac *= -1.;
  vector <int> in, out;
  for (int i=0; i < state.size(); ++i) {
    if (state[i].isFinal()) out.push_back(state[i].id());
    if (state[i].mother1() == 1 && state[i].mother2() == 0)
      in.push_back(state[i].id());
    if (state[i].mother1() == 2 && state[i].mother2() == 0)
      in.push_back(state[i].id());
  }
  out.push_back(22);
  bool hasME = fsr->weights->hasME(in, out);
  if (hasME && chargeFac < 0.0) chargeFac = abs(chargeFac);

  double preFac = symmetryFactor() * chargeFac;
  double kappa2 = pT2/m2dip;
  wt   = preFac * ( 2.* (1.-z) / ( pow2(1.-z) + kappa2) );

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  // Add collinear term for massless splittings.
  if (!doMassive) wt  += -preFac * ( 1.+z );

  // Add collinear term for massive splittings.
  if (doMassive) {

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
    double massCorr = -1.*vijkt/vijk*( 1. + z + m2RadBef/pipj);
    wt += preFac*massCorr;

  }

  // Now multiply with z to project out Q->QG,
  // i.e. the gluon is soft and the quark is identified.
  wt *= z; 

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

// Class inheriting from SplittingQED class.

// SplittingQED function Q->GQ (FSR)
// At leading order, this can be combined with Q->QG because of symmetry. Since
// this is no longer possible at NLO, we keep the kernels separately.

// Return true if this kernel should partake in the evolution.
bool fsr_qed_Q2AQ::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].isQuark() && state[ints["iRec"]].isCharged());
}

int fsr_qed_Q2AQ::kinMap()                 { return 1;}
int fsr_qed_Q2AQ::motherID(int idDaughter) { return idDaughter;}
int fsr_qed_Q2AQ::sisterID(int)            { return 22;}

double fsr_qed_Q2AQ::gaugeFactor ( int idRadBef, int idRecBef) {
  double chgRad = particleDataPtr->charge(idRadBef);
  double chgRec = particleDataPtr->charge(idRecBef);
  double charge = -1.*chgRad*chgRec;
  if (idRadBef != 0 && idRecBef != 0) return charge;
  // Set probability to zero.
  return 0.;
}

double fsr_qed_Q2AQ::symmetryFactor ( int, int ) { return 1.;}

int fsr_qed_Q2AQ::radBefID(int idRad, int idEmt) {
  if (idRad == 22 && particleDataPtr->isQuark(idEmt)) return idEmt;
  if (idEmt == 22 && particleDataPtr->isQuark(idRad)) return idRad;
  return 0;
}

pair<int,int> fsr_qed_Q2AQ::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int, int) { return make_pair(colRadAfter,acolRadAfter); }

vector<int>fsr_qed_Q2AQ::recPositions(const Event& state, int iRad, int iEmt) {

  vector<int> recs;
  if ( !state[iRad].isFinal()
    || !state[iRad].isQuark() 
    || state[iEmt].id() != 22) return recs;

  // Particles to exclude as recoilers.
  vector<int> iExc(createvector<int>(iRad)(iEmt));
  // Find charged particles.
  for (int i=0; i < state.size(); ++i) {
    if ( find(iExc.begin(), iExc.end(), i) != iExc.end() ) continue;
    if ( state[i].isCharged() ) {
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
double fsr_qed_Q2AQ::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qed_Q2AQ::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * abs(gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id));
  // Q -> QG, soft part (currently also used for collinear part).
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappa2);

  return wt;
}

// Return overestimate for new splitting.
double fsr_qed_Q2AQ::overestimateDiff(double z, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * abs(gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id));
  double kappaOld2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool fsr_qed_Q2AQ::calc(const Event& state, int orderNow) {

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
  // and later multiply with 1-z to project out Q->GQ,
  // i.e. the quark is soft and the gluon is identified.
  double wt = 0.;

  double chargeFac = gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id);
  if ( !state[splitInfo.iRecBef].isFinal() ) chargeFac *= -1.;
  vector <int> in, out;
  for (int i=0; i < state.size(); ++i) {
    if (state[i].isFinal()) out.push_back(state[i].id());
    if (state[i].mother1() == 1 && state[i].mother2() == 0)
      in.push_back(state[i].id());
    if (state[i].mother1() == 2 && state[i].mother2() == 0)
      in.push_back(state[i].id());
  }
  out.push_back(22);
  bool hasME = fsr->weights->hasME(in, out);
  if (hasME && chargeFac < 0.0) chargeFac = abs(chargeFac);

  double preFac = symmetryFactor() * chargeFac;
  double kappa2 = pT2/m2dip;
  wt   = preFac * ( 2.* (1.-z) / ( pow2(1.-z) + kappa2) );

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  // Add collinear term for massless splittings.
  if (!doMassive) wt  += -preFac * ( 1.+z );

  // Add collinear term for massive splittings.
  if (doMassive) {

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
    double massCorr = -1.*vijkt/vijk*( 1. + z + m2RadBef/pipj);
    wt += preFac*massCorr;

  }

  // Now multiply with (1-z) to project out Q->GQ,
  // i.e. the quark is soft and the gluon is identified.
  wt *= ( 1. - z ); 

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

// Class inheriting from SplittingQED class.

// SplittingQED function Q->QG (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_qed_L2LA::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].isLepton() && state[ints["iRad"]].isCharged()
        && state[ints["iRec"]].isCharged());
}

int fsr_qed_L2LA::kinMap()                 { return 1;}
int fsr_qed_L2LA::motherID(int idDaughter) { return idDaughter;}
int fsr_qed_L2LA::sisterID(int)            { return 22;}

double fsr_qed_L2LA::gaugeFactor ( int idRadBef, int idRecBef) {
  double chgRad = particleDataPtr->charge(idRadBef);
  double chgRec = particleDataPtr->charge(idRecBef);
  double charge = -1.*chgRad*chgRec;
  if (idRadBef != 0 && idRecBef != 0) return charge;
  // Set probability to zero.
  return 0.;
}

double fsr_qed_L2LA::symmetryFactor ( int, int ) { return 1.;}

int fsr_qed_L2LA::radBefID(int idRad, int idEmt) {
  if (idEmt == 22 && particleDataPtr->isLepton(idRad)
    && particleDataPtr->charge(idRad) != 0) return idRad;
  return 0;
}

pair<int,int> fsr_qed_L2LA::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int, int) { return make_pair(colRadAfter,acolRadAfter); }

vector<int>fsr_qed_L2LA::recPositions(const Event& state, int iRad, int iEmt) {

  vector<int> recs;
  if ( !state[iRad].isFinal()
    || !(state[iRad].isLepton() && state[iRad].isCharged()) 
    || state[iEmt].id() != 22) return recs;

  // Particles to exclude as recoilers.
  vector<int> iExc(createvector<int>(iRad)(iEmt));
  // Find charged particles.
  for (int i=0; i < state.size(); ++i) {
    if ( find(iExc.begin(), iExc.end(), i) != iExc.end() ) continue;
    if ( state[i].isCharged() ) {
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
double fsr_qed_L2LA::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qed_L2LA::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * abs(gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id));
  // Q -> QG, soft part (currently also used for collinear part).
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappa2);

  return wt;
}

// Return overestimate for new splitting.
double fsr_qed_L2LA::overestimateDiff(double z, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * abs(gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id));
  double kappaOld2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool fsr_qed_L2LA::calc(const Event& state, int orderNow) { 

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

  double chargeFac = gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id);
  if ( !state[splitInfo.iRecBef].isFinal() ) chargeFac *= -1.;
  vector <int> in, out;
  for (int i=0; i < state.size(); ++i) {
    if (state[i].isFinal()) out.push_back(state[i].id());
    if (state[i].mother1() == 1 && state[i].mother2() == 0)
      in.push_back(state[i].id());
    if (state[i].mother1() == 2 && state[i].mother2() == 0)
      in.push_back(state[i].id());
  }
  out.push_back(22);
  bool hasME = fsr->weights->hasME(in, out);
  if (hasME && chargeFac < 0.0) chargeFac = abs(chargeFac);

  double preFac = symmetryFactor() * chargeFac;
  double kappa2 = pT2/m2dip;
  wt   = preFac * ( 2.* (1.-z) / ( pow2(1.-z) + kappa2) );

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  // Add collinear term for massless splittings.
  if (!doMassive) wt  += -preFac * ( 1.+z );

  // Add collinear term for massive splittings.
  if (doMassive) {

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
    double massCorr = -1.*vijkt/vijk*( 1. + z + m2RadBef/pipj);
    wt += preFac*massCorr;

  }

  // Now multiply with z to project out Q->QG,
  // i.e. the gluon is soft and the quark is identified.
  wt *= z; 

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

// Class inheriting from SplittingQED class.

// SplittingQED function Q->GQ (FSR)
// At leading order, this can be combined with Q->QG because of symmetry. Since
// this is no longer possible at NLO, we keep the kernels separately.

// Return true if this kernel should partake in the evolution.
bool fsr_qed_L2AL::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].isLepton() && state[ints["iRad"]].isCharged()
        && state[ints["iRec"]].isCharged());
}

int fsr_qed_L2AL::kinMap()                 { return 1;}
int fsr_qed_L2AL::motherID(int idDaughter) { return idDaughter;}
int fsr_qed_L2AL::sisterID(int)            { return 22;}

double fsr_qed_L2AL::gaugeFactor ( int idRadBef, int idRecBef) {
  double chgRad = particleDataPtr->charge(idRadBef);
  double chgRec = particleDataPtr->charge(idRecBef);
  double charge = -1.*chgRad*chgRec;
  if (idRadBef != 0 && idRecBef != 0) return charge;
  // Set probability to zero.
  return 0.;
}

double fsr_qed_L2AL::symmetryFactor ( int, int ) { return 1.;}

int fsr_qed_L2AL::radBefID(int idRad, int idEmt) {
  if (idRad == 22 && particleDataPtr->isLepton(idEmt)
    && particleDataPtr->charge(idEmt) != 0) return idEmt;
  if (idEmt == 22 && particleDataPtr->isLepton(idRad)
    && particleDataPtr->charge(idRad) != 0) return idRad;
  return 0;
}

pair<int,int> fsr_qed_L2AL::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int, int) { return make_pair(colRadAfter,acolRadAfter); }

vector<int>fsr_qed_L2AL::recPositions(const Event& state, int iRad, int iEmt) {

  vector<int> recs;
  if ( !state[iRad].isFinal()
    || !(state[iRad].isLepton() && state[iRad].isCharged())  
    || state[iEmt].id() != 22) return recs;

  // Particles to exclude as recoilers.
  vector<int> iExc(createvector<int>(iRad)(iEmt));
  // Find charged particles.
  for (int i=0; i < state.size(); ++i) {
    if ( find(iExc.begin(), iExc.end(), i) != iExc.end() ) continue;
    if ( state[i].isCharged() ) {
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
double fsr_qed_L2AL::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qed_L2AL::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * abs(gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id));
  // Q -> QG, soft part (currently also used for collinear part).
  double kappa2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappa2);

  return wt;
}

// Return overestimate for new splitting.
double fsr_qed_L2AL::overestimateDiff(double z, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * abs(gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id));
  double kappaOld2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  wt  = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool fsr_qed_L2AL::calc(const Event& state, int orderNow) {

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
  // and later multiply with 1-z to project out Q->GQ,
  // i.e. the quark is soft and the gluon is identified.
  double wt = 0.;

  double chargeFac = gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id);
  if ( !state[splitInfo.iRecBef].isFinal() ) chargeFac *= -1.;
  vector <int> in, out;
  for (int i=0; i < state.size(); ++i) {
    if (state[i].isFinal()) out.push_back(state[i].id());
    if (state[i].mother1() == 1 && state[i].mother2() == 0)
      in.push_back(state[i].id());
    if (state[i].mother1() == 2 && state[i].mother2() == 0)
      in.push_back(state[i].id());
  }
  out.push_back(22);
  bool hasME = fsr->weights->hasME(in, out);
  if (hasME && chargeFac < 0.0) chargeFac = abs(chargeFac);

  double preFac = symmetryFactor() * chargeFac;
  double kappa2 = pT2/m2dip;
  wt   = preFac * ( 2.* (1.-z) / ( pow2(1.-z) + kappa2) );

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  // Add collinear term for massless splittings.
  if (!doMassive) wt  += -preFac * ( 1.+z );

  // Add collinear term for massive splittings.
  if (doMassive) {

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
    double massCorr = -1.*vijkt/vijk*( 1. + z + m2RadBef/pipj);
    wt += preFac*massCorr;

  }

  // Now multiply with (1-z) to project out Q->GQ,
  // i.e. the quark is soft and the gluon is identified.
  wt *= ( 1. - z ); 

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

// Class inheriting from SplittingQED class.

// SplittingQED function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_qed_A2QQ1::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].id() == 22 );
}

int fsr_qed_A2QQ1::kinMap()                 { return 1;}
int fsr_qed_A2QQ1::motherID(int)            { return 1;} // Use 1 as dummy variable.
int fsr_qed_A2QQ1::sisterID(int)            { return 1;} // Use 1 as dummy variable.
double fsr_qed_A2QQ1::gaugeFactor ( int, int )        { return sumCharge2Tot;}
double fsr_qed_A2QQ1::symmetryFactor ( int, int )     { return 1.;}

int fsr_qed_A2QQ1::radBefID(int, int){ return 22;}
pair<int,int> fsr_qed_A2QQ1::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

// Pick z for new splitting.
double fsr_qed_A2QQ1::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double fsr_qed_A2QQ1::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5 * ( zMaxAbs - zMinAbs);

  return wt;
}

// Return overestimate for new splitting.
double fsr_qed_A2QQ1::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool fsr_qed_A2QQ1::calc(const Event& state, int orderNow) {

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

//  // Now divide out alphaS (used in the evolution) and replace with alphaEM.
//  wt = aem0 / as2Pi(pT2); 

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

// Class inheriting from SplittingQED class.

// SplittingQED function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_qed_A2QQ2::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].id() == 22 );
}

int fsr_qed_A2QQ2::kinMap()                 { return 1;}
int fsr_qed_A2QQ2::motherID(int)            { return -1;} // Use -1 as dummy variable.
int fsr_qed_A2QQ2::sisterID(int)            { return -1;} // Use -1 as dummy variable.
double fsr_qed_A2QQ2::gaugeFactor ( int, int )        { return sumCharge2Tot;}
double fsr_qed_A2QQ2::symmetryFactor ( int, int )     { return 1.;}

int fsr_qed_A2QQ2::radBefID(int, int){ return 22;}
pair<int,int> fsr_qed_A2QQ2::radBefCols( int, int, int, int)
  { return make_pair(0,0); }

// Pick z for new splitting.
double fsr_qed_A2QQ2::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double fsr_qed_A2QQ2::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5 * ( zMaxAbs - zMinAbs);

  return wt;
}

// Return overestimate for new splitting.
double fsr_qed_A2QQ2::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool fsr_qed_A2QQ2::calc(const Event& state, int orderNow) {

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

  // Multiply with z to project out part where emitted antiquark is soft,
  // and quark is identified.
  wt *= (1.-z);

//  // Now divide out alphaS (used in the evolution) and replace with alphaEM.
//  wt = aem0 / as2Pi(pT2); 

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

// Class inheriting from SplittingQED class.

// SplittingQED function Q->QG (ISR)

// Return true if this kernel should partake in the evolution.
bool isr_qed_Q2QA::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].isQuark()
        && state[ints["iRec"]].isCharged() );
}

int isr_qed_Q2QA::kinMap()                 { return 1;}
int isr_qed_Q2QA::motherID(int idDaughter) { return idDaughter;} 
int isr_qed_Q2QA::sisterID(int)            { return 22;}

double isr_qed_Q2QA::gaugeFactor ( int idRadBef, int idRecBef) {
  double chgRad = particleDataPtr->charge(idRadBef);
  double chgRec = particleDataPtr->charge(idRecBef);
  double charge = -1.*chgRad*chgRec;
  if (idRadBef != 0 && idRecBef != 0) return charge;
  // Set probability to zero.
  return 0.;
}

double isr_qed_Q2QA::symmetryFactor ( int, int )     { return 1.;}

int isr_qed_Q2QA::radBefID(int idRad, int idEmt) {
  if (particleDataPtr->isQuark(idRad) && idEmt == 22 ) return idRad;
  return 0;
}

pair<int,int> isr_qed_Q2QA::radBefCols( int colRadAfter, int acolRadAfter, int,
  int) {
  bool isQuark  = (colRadAfter > 0);
  if (isQuark) return make_pair(colRadAfter,0); 
  return make_pair(0,acolRadAfter); 
}

vector<int>isr_qed_Q2QA::recPositions(const Event& state, int iRad, int iEmt) {

  vector<int> recs;
  if ( state[iRad].isFinal() || !state[iRad].isQuark()
    || state[iEmt].id() != 22) return recs;

  // Particles to exclude as recoilers.
  vector<int> iExc(createvector<int>(iRad)(iEmt));
  // Find charged particles.
  for (int i=0; i < state.size(); ++i) {
    if ( find(iExc.begin(), iExc.end(), i) != iExc.end() ) continue;
    if ( state[i].isCharged() ) {
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
double isr_qed_Q2QA::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double isr_qed_Q2QA::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * abs(gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id));
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  wt  = preFac * 2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappa2);

  return wt;
}

// Return overestimate for new splitting.
double isr_qed_Q2QA::overestimateDiff(double z, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * abs(gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id));
  double kappaOld2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  wt  = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool isr_qed_Q2QA::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip);
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    //m2Rec(splitInfo.kinematics()->m2Rec),
    //m2Emt(splitInfo.kinematics()->m2EmtAft);
  //int splitType(splitInfo.type), idRad(int(splitInfo.extras["idRadAft"]));

  double wt = 0.;

  double chargeFac = gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id);
  if ( !state[splitInfo.iRecBef].isFinal() ) chargeFac *= -1.;
  if ( !state[splitInfo.iRadBef].isFinal() ) chargeFac *= -1.;

  vector <int> in, out;
  for (int i=0; i < state.size(); ++i) {
    if (state[i].isFinal()) out.push_back(state[i].id());
    if (state[i].mother1() == 1 && state[i].mother2() == 0)
      in.push_back(state[i].id());
    if (state[i].mother1() == 2 && state[i].mother2() == 0)
      in.push_back(state[i].id());
  }
  out.push_back(22);
  bool hasME = isr->weights->hasME(in, out);
  if (hasME && chargeFac < 0.0) chargeFac = abs(chargeFac);

  double preFac = symmetryFactor() * chargeFac;
  double kappa2 = pT2/m2dip;
  wt   =  preFac * ( 2.* (1.-z) / ( pow2(1.-z) + kappa2) );
  wt  += -preFac * (1.+z);

  // Trivial map of values, since kernel does not depend on coupling.
  map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", wt ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQED class.

// SplittingQED function G->QQ (ISR)

// Return true if this kernel should partake in the evolution.
bool isr_qed_A2QQ::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].isQuark() );
}

int isr_qed_A2QQ::kinMap()                 { return 1;}
int isr_qed_A2QQ::motherID(int)            { return 22;} 
int isr_qed_A2QQ::sisterID(int idDaughter) { return -idDaughter;} 
double isr_qed_A2QQ::gaugeFactor ( int, int )        { return 1.;}
double isr_qed_A2QQ::symmetryFactor ( int, int )     { return 1.;}

int isr_qed_A2QQ::radBefID(int, int idEA){ return -idEA;}
pair<int,int> isr_qed_A2QQ::radBefCols( int, int, int colEmtAfter,
  int acolEmtAfter) {
  if ( acolEmtAfter > 0 ) return make_pair(acolEmtAfter,0);
  return make_pair(0, colEmtAfter);
}

// Pick z for new splitting.
double isr_qed_A2QQ::zSplit(double zMinAbs, double zMaxAbs, double) {
  // Note: Combined with PDF ratio, flat overestimate performs
  // better than using the full splitting kernel as overestimate. 
  double res = zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs);
  //double R   = rndmPtr->flat();
  //double res = (
  //(1 + pow(-2 + 4*R - 6*R*zMinAbs + 6*R*pow(zMinAbs,2) - 4*R*pow(zMinAbs,3) + 
  //     sqrt(1 + 4*pow(1 + R*(-1 + zMinAbs)*(2 + zMinAbs*(-1 + 2*zMinAbs)),2)),1./3.) - 
  //   pow(-2 + 4*R - 6*R*zMinAbs + 6*R*pow(zMinAbs,2) - 4*R*pow(zMinAbs,3) + 
  //     sqrt(1 + 4*pow(1 + R*(-1 + zMinAbs)*(2 + zMinAbs*(-1 + 2*zMinAbs)),2)),2./3.))/
  //(2.*pow(-2 + 4*R - 6*R*zMinAbs + 6*R*pow(zMinAbs,2) - 4*R*pow(zMinAbs,3) + 
  //     sqrt(1 + 4*pow(1 + R*(-1 + zMinAbs)*(2 + zMinAbs*(-1 + 2*zMinAbs)),2)),1./3.))
  //);
  return res;
}

// New overestimates, z-integrated versions.
double isr_qed_A2QQ::overestimateInt(double zMinAbs, double zMaxAbs,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  // Note: Combined with PDF ratio, flat overestimate performs
  // better than using the full splitting kernel as overestimate. 
  wt  = preFac 
      * 2. * ( zMaxAbs - zMinAbs);
  //wt  = preFac
  //    * ( 2./3. - zMinAbs + pow(zMinAbs,2) - (2*pow(zMinAbs,3))/3.);

  return wt;
}

// Return overestimate for new splitting.
double isr_qed_A2QQ::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  // Note: Combined with PDF ratio, flat overestimate performs
  // better than using the full splitting kernel as overestimate. 
  wt = preFac 
     * 2.;
  //wt = preFac
  //   * (pow(1.-z,2.) + pow(z,2.));
  return wt;
}

// Return kernel for new splitting.
bool isr_qed_A2QQ::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z);// pT2(splitInfo.kinematics()->pT2);
    //m2dip(splitInfo.kinematics()->m2Dip),
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    //m2Rec(splitInfo.kinematics()->m2Rec),
    //m2Emt(splitInfo.kinematics()->m2EmtAft);
  //int splitType(splitInfo.type), idRad(int(splitInfo.extras["idRadAft"]));

  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = preFac * (pow(1.-z,2.) + pow(z,2.));

//  // Now divide out alphaS (used in the evolution) and replace with alphaEM.
//  wt = aem0 / as2Pi(pT2); 

  // Trivial map of values, since kernel does not depend on coupling.
  map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", wt ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQED class.

// SplittingQED function Q->AQ (ISR)

// Return true if this kernel should partake in the evolution.
bool isr_qed_Q2AQ::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].id() == 22 );
}

int isr_qed_Q2AQ::kinMap()                 { return 1;}
int isr_qed_Q2AQ::motherID(int)            { return 1;} // Use 1 as dummy 
int isr_qed_Q2AQ::sisterID(int)            { return 1;} // Use 1 as dummy
double isr_qed_Q2AQ::gaugeFactor ( int, int )        { return 1.;}
double isr_qed_Q2AQ::symmetryFactor ( int, int )     { return 0.5;}

int isr_qed_Q2AQ::radBefID(int, int){ return 22;}
pair<int,int> isr_qed_Q2AQ::radBefCols( int, int, int, int) {
  return make_pair(0,0); }

// Pick z for new splitting.
double isr_qed_Q2AQ::zSplit(double zMinAbs, double, double) {
  double R = rndmPtr->flat();
  double res = pow(zMinAbs,3./4.)
          / ( pow(1. + R*(-1. + pow(zMinAbs,-3./8.)),2./3.)
             *pow(R - (-1. + R)*pow(zMinAbs,3./8.),2.));
//  double kappaOld2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
//  double res = sqrt( pow( pow2(zMinAbs) + kappaOld2, R)
//                   * pow( 1.            + kappaOld2, 1.- R) - kappaOld2); 
  return res;
}

// New overestimates, z-integrated versions.
double isr_qed_Q2AQ::overestimateInt(double zMinAbs, double,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = preFac * 2./3. * (8.*(-1. + pow(zMinAbs,-3./8.)));
//  double kappaOld2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
//  wt = preFac * log( (1+kappaOld2) / (pow2(zMinAbs) + kappaOld2) );

  return wt;
}

// Return overestimate for new splitting.
double isr_qed_Q2AQ::overestimateDiff(double z, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = preFac * 2. / pow(z,11./8.);
//  double kappaOld2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
//  wt  = preFac * 2. * z / (pow2(z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool isr_qed_Q2AQ::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip),
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    m2Rec(splitInfo.kinematics()->m2Rec);
    //m2Emt(splitInfo.kinematics()->m2EmtAft);
  int splitType(splitInfo.type);

  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double kappa2 = pT2 / m2dip;
  wt   = preFac
       * ( z + 2.*z / (pow2(z)+kappa2) - 2. );

  // Correction for massive IF splittings.
  bool doMassive = ( m2Rec > 0. && splitType == 2);

  if (doMassive) {
    // Construct CS variables.
    double uCS = kappa2 / (1-z);

    double massCorr = -2. * m2Rec / m2dip * uCS / (1.-uCS);
    // Add correction.
    wt += preFac * massCorr;

  }

//  // Now divide out alphaS (used in the evolution) and replace with alphaEM.
//  wt = aem0 / as2Pi(pT2); 

  // Trivial map of values, since kernel does not depend on coupling.
  map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", wt ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQED class.

// SplittingQED function L->LA (ISR)

// Return true if this kernel should partake in the evolution.
bool isr_qed_L2LA::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].isLepton() && state[ints["iRad"]].isCharged()
        && state[ints["iRec"]].isCharged());
}

int isr_qed_L2LA::kinMap()                 { return 1;}
int isr_qed_L2LA::motherID(int idDaughter) { return idDaughter;} 
int isr_qed_L2LA::sisterID(int)            { return 22;} 

double isr_qed_L2LA::gaugeFactor ( int idRadBef, int idRecBef) {
  double chgRad = particleDataPtr->charge(idRadBef);
  double chgRec = particleDataPtr->charge(idRecBef);
  double charge = -1.*chgRad*chgRec;
  if (idRadBef != 0 && idRecBef != 0) return charge;
  // Set probability to zero.
  return 0.;
}

double isr_qed_L2LA::symmetryFactor ( int, int )     { return 1.;}

int isr_qed_L2LA::radBefID(int idRad, int idEmt) {
  if (particleDataPtr->isLepton(idRad) && particleDataPtr->charge(idRad) != 0
    && idEmt == 22 ) return idRad;
  return 0;
}


pair<int,int> isr_qed_L2LA::radBefCols( int colRadAfter, int acolRadAfter, int,
  int) {
  bool isQuark  = (colRadAfter > 0);
  if (isQuark) return make_pair(colRadAfter,0); 
  return make_pair(0,acolRadAfter); 
}

vector<int>isr_qed_L2LA::recPositions(const Event& state, int iRad, int iEmt) {

  vector<int> recs;
  if ( state[iRad].isFinal()
    || !(state[iRad].isLepton() && state[iRad].isCharged())
    || state[iEmt].id() != 22) return recs;

  // Particles to exclude as recoilers.
  vector<int> iExc(createvector<int>(iRad)(iEmt));
  // Find charged particles.
  for (int i=0; i < state.size(); ++i) {
    if ( find(iExc.begin(), iExc.end(), i) != iExc.end() ) continue;
    if ( state[i].isCharged() ) {
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
double isr_qed_L2LA::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double isr_qed_L2LA::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * abs(gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id));
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  wt  = preFac * 2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappa2);

  return wt;
}

// Return overestimate for new splitting.
double isr_qed_L2LA::overestimateDiff(double z, double m2dip, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * abs(gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id));
  double kappaOld2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  wt  = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool isr_qed_L2LA::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip);
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    //m2Rec(splitInfo.kinematics()->m2Rec),
    //m2Emt(splitInfo.kinematics()->m2EmtAft);
  //int splitType(splitInfo.type), idRad(int(splitInfo.extras["idRadAft"]));

  double wt = 0.;

  double chargeFac = gaugeFactor(splitInfo.radBef()->id, splitInfo.recBef()->id);
  if ( !state[splitInfo.iRecBef].isFinal() ) chargeFac *= -1.;
  if ( !state[splitInfo.iRadBef].isFinal() ) chargeFac *= -1.;
  vector <int> in, out;
  for (int i=0; i < state.size(); ++i) {
    if (state[i].isFinal()) out.push_back(state[i].id());
    if (state[i].mother1() == 1 && state[i].mother2() == 0)
      in.push_back(state[i].id());
    if (state[i].mother1() == 2 && state[i].mother2() == 0)
      in.push_back(state[i].id());
  }
  out.push_back(22);
  bool hasME = isr->weights->hasME(in, out);
  if (hasME && chargeFac < 0.0) chargeFac = abs(chargeFac);

  double preFac = symmetryFactor() * chargeFac;
  double kappa2 = pT2/m2dip;
  wt   =  preFac * ( 2.* (1.-z) / ( pow2(1.-z) + kappa2) );
  wt  += -preFac * (1.+z);

  // Trivial map of values, since kernel does not depend on coupling.
  map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", wt ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQED class.

// SplittingQED function A->LL (ISR)

// Return true if this kernel should partake in the evolution.
bool isr_qed_A2LL::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].isLepton() && state[ints["iRad"]].isCharged());
}

int isr_qed_A2LL::kinMap()                 { return 1;}
int isr_qed_A2LL::motherID(int)            { return 22;} 
int isr_qed_A2LL::sisterID(int idDaughter) { return -idDaughter;} 
double isr_qed_A2LL::gaugeFactor ( int, int )        { return 1.;}
double isr_qed_A2LL::symmetryFactor ( int, int )     { return 1.;}

int isr_qed_A2LL::radBefID(int, int idEA){ return -idEA;}
pair<int,int> isr_qed_A2LL::radBefCols( int, int, int colEmtAfter,
  int acolEmtAfter) {
  if ( acolEmtAfter > 0 ) return make_pair(acolEmtAfter,0);
  return make_pair(0, colEmtAfter);
}

// Pick z for new splitting.
double isr_qed_A2LL::zSplit(double zMinAbs, double zMaxAbs, double) {
  // Note: Combined with PDF ratio, flat overestimate performs
  // better than using the full splitting kernel as overestimate. 
  double res = zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs);
  //double R   = rndmPtr->flat();
  //double res = (
  //(1 + pow(-2 + 4*R - 6*R*zMinAbs + 6*R*pow(zMinAbs,2) - 4*R*pow(zMinAbs,3) + 
  //     sqrt(1 + 4*pow(1 + R*(-1 + zMinAbs)*(2 + zMinAbs*(-1 + 2*zMinAbs)),2)),1./3.) - 
  //   pow(-2 + 4*R - 6*R*zMinAbs + 6*R*pow(zMinAbs,2) - 4*R*pow(zMinAbs,3) + 
  //     sqrt(1 + 4*pow(1 + R*(-1 + zMinAbs)*(2 + zMinAbs*(-1 + 2*zMinAbs)),2)),2./3.))/
  //(2.*pow(-2 + 4*R - 6*R*zMinAbs + 6*R*pow(zMinAbs,2) - 4*R*pow(zMinAbs,3) + 
  //     sqrt(1 + 4*pow(1 + R*(-1 + zMinAbs)*(2 + zMinAbs*(-1 + 2*zMinAbs)),2)),1./3.))
  //);
  return res;
}

// New overestimates, z-integrated versions.
double isr_qed_A2LL::overestimateInt(double zMinAbs, double zMaxAbs,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  // Note: Combined with PDF ratio, flat overestimate performs
  // better than using the full splitting kernel as overestimate. 
  wt  = preFac 
      * 2. * ( zMaxAbs - zMinAbs);
  //wt  = preFac
  //    * ( 2./3. - zMinAbs + pow(zMinAbs,2) - (2*pow(zMinAbs,3))/3.);

  return wt;
}

// Return overestimate for new splitting.
double isr_qed_A2LL::overestimateDiff(double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  // Note: Combined with PDF ratio, flat overestimate performs
  // better than using the full splitting kernel as overestimate. 
  wt = preFac 
     * 2.;
  //wt = preFac
  //   * (pow(1.-z,2.) + pow(z,2.));
  return wt;
}

// Return kernel for new splitting.
bool isr_qed_A2LL::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z); //, pT2(splitInfo.kinematics()->pT2);
    //m2dip(splitInfo.kinematics()->m2Dip),
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    //m2Rec(splitInfo.kinematics()->m2Rec),
    //m2Emt(splitInfo.kinematics()->m2EmtAft);
  //int splitType(splitInfo.type), idRad(int(splitInfo.extras["idRadAft"]));

  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = preFac * (pow(1.-z,2.) + pow(z,2.));

//  // Now divide out alphaS (used in the evolution) and replace with alphaEM.
//  wt = aem0 / as2Pi(pT2); 

  // Trivial map of values, since kernel does not depend on coupling.
  map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", wt ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQED class.

// SplittingQED function L->AL (ISR)

// Return true if this kernel should partake in the evolution.
bool isr_qed_L2AL::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*){
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRad"]].id() == 22 );
}

int isr_qed_L2AL::kinMap()                 { return 1;}
int isr_qed_L2AL::motherID(int)            { return 1;} // Use 1 as dummy 
int isr_qed_L2AL::sisterID(int)            { return 1;} // Use 1 as dummy
double isr_qed_L2AL::gaugeFactor ( int, int )        { return 1.;}
double isr_qed_L2AL::symmetryFactor ( int, int )     { return 0.5;}

int isr_qed_L2AL::radBefID(int, int){ return 22;}
pair<int,int> isr_qed_L2AL::radBefCols( int, int, int, int) {
  return make_pair(0,0); }

// Pick z for new splitting.
double isr_qed_L2AL::zSplit(double zMinAbs, double, double) {
  double R = rndmPtr->flat();
  double res = pow(zMinAbs,3./4.)
          / ( pow(1. + R*(-1. + pow(zMinAbs,-3./8.)),2./3.)
             *pow(R - (-1. + R)*pow(zMinAbs,3./8.),2.));
//  double kappaOld2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
//  double res = sqrt( pow( pow2(zMinAbs) + kappaOld2, R)
//                   * pow( 1.            + kappaOld2, 1.- R) - kappaOld2); 
  return res;
}

// New overestimates, z-integrated versions.
double isr_qed_L2AL::overestimateInt(double zMinAbs, double,
  double, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = preFac * 2./3. * (8.*(-1. + pow(zMinAbs,-3./8.)));
//  double kappaOld2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
//  wt = preFac * log( (1+kappaOld2) / (pow2(zMinAbs) + kappaOld2) );

  return wt;
}

// Return overestimate for new splitting.
double isr_qed_L2AL::overestimateDiff(double z, double, int) {
  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt  = preFac * 2. / pow(z,11./8.);
//  double kappaOld2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
//  wt  = preFac * 2. * z / (pow2(z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool isr_qed_L2AL::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip),
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    m2Rec(splitInfo.kinematics()->m2Rec);
    //m2Emt(splitInfo.kinematics()->m2EmtAft);
  int splitType(splitInfo.type);

  double wt = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double kappa2 = pT2 / m2dip;
  wt   = preFac
       * ( z + 2.*z / (pow2(z)+kappa2) - 2. );

  // Correction for massive IF splittings.
  bool doMassive = ( m2Rec > 0. && splitType == 2);

  if (doMassive) {
    // Construct CS variables.
    double uCS = kappa2 / (1-z);

    double massCorr = -2. * m2Rec / m2dip * uCS / (1.-uCS);
    // Add correction.
    wt += preFac * massCorr;

  }

//  // Now divide out alphaS (used in the evolution) and replace with alphaEM.
//  wt = aem0 / as2Pi(pT2); 

  // Trivial map of values, since kernel does not depend on coupling.
  map<string,double> wts;
  wts.insert( make_pair("base", wt ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", wt ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", wt ));
  }

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

} // end namespace Pythia8
