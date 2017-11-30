
#include "Dire/SplittingsQCD.h"
#include "Dire/DireSpace.h"
#include "Dire/DireTimes.h"

namespace Pythia8 {


//==========================================================================

// The SplittingQCD class.

const double SplittingQCD::SMALL_TEVOL = 2.0;

//-------------------------------------------------------------------------

void SplittingQCD::init() {

  CA = 3.;
  TR = 0.5;
  CF = 4./3.;

  NF_qcd_fsr      = settingsPtr->mode("TimeShower:nGluonToQuark");

  // Parameters of alphaS.
  double alphaSvalue = settingsPtr->parm("SpaceShower:alphaSvalue");
  int alphaSorder    = settingsPtr->mode("SpaceShower:alphaSorder");
  int alphaSnfmax    = settingsPtr->mode("StandardModel:alphaSnfmax");
  bool alphaSuseCMW  = settingsPtr->flag("SpaceShower:alphaSuseCMW");
  // Initialize alphaS.
  alphaS.init( alphaSvalue, alphaSorder, alphaSnfmax, alphaSuseCMW);

  // Set up alphaS
  pTmin              = settingsPtr->parm("SpaceShower:pTmin");
  pTmin              = min(pTmin,settingsPtr->parm("TimeShower:pTmin"));
  usePDFalphas       = settingsPtr->flag("ShowerPDF:usePDFalphas");

  BeamParticle* beam = NULL;
  if (beamAPtr != NULL || beamBPtr != NULL) {
    beam = (beamAPtr != NULL && particleDataPtr->isHadron(beamAPtr->id())) ? beamAPtr
         : (beamBPtr != NULL && particleDataPtr->isHadron(beamBPtr->id())) ? beamBPtr : NULL;
    if (beam == NULL && beamAPtr != 0) beam = beamAPtr;
    if (beam == NULL && beamBPtr != 0) beam = beamBPtr;
  }
  alphaS2pi          = (usePDFalphas && beam != NULL)
                        ? beam->alphaS(pTmin*pTmin) * 0.5/M_PI
                        : (alphaSorder > 0)
                        ? alphaS.alphaS(pTmin*pTmin) *0.5/M_PI
                        :  0.5 * 0.5/M_PI;

  doVariations       = settingsPtr->flag("Variations:doVariations");

}

//-------------------------------------------------------------------------

// Auxiliary function to get number of flavours.

double SplittingQCD::getNF(double pT2) {
  double NF = 6.;

  pT2       = max( pT2, pow2(pTmin) );

  BeamParticle* beam = NULL;
  if (beamAPtr != NULL || beamBPtr != NULL) {
    beam = (beamAPtr != NULL && particleDataPtr->isHadron(beamAPtr->id())) ? beamAPtr
         : (beamBPtr != NULL && particleDataPtr->isHadron(beamBPtr->id())) ? beamBPtr : NULL;
    if (beam == NULL && beamAPtr != 0) beam = beamAPtr;
    if (beam == NULL && beamBPtr != 0) beam = beamBPtr;
  }

  // Get current number of flavours.
  if ( !usePDFalphas || beam == NULL) {
    if ( pT2 > pow2( max(0., particleDataPtr->m0(5) ) )
      && pT2 < pow2( particleDataPtr->m0(6)) )                 NF = 5.;
    else if ( pT2 > pow2( max( 0., particleDataPtr->m0(4)) ) ) NF = 4.; 
    else if ( pT2 > pow2( max( 0., particleDataPtr->m0(3)) ) ) NF = 3.; 
  } else {
    if ( pT2 > pow2( max(0., beam->mQuarkPDF(5) ) )
      && pT2 < pow2( particleDataPtr->m0(6)) )                 NF = 5.;
    else if ( pT2 > pow2( max( 0., beam->mQuarkPDF(4)) ) )     NF = 4.; 
    else if ( pT2 > pow2( max( 0., beam->mQuarkPDF(3)) ) )     NF = 3.; 
  }
  return NF;
}

//--------------------------------------------------------------------------

double SplittingQCD::GammaQCD2 (double NF) {
  return (67./18.-pow2(M_PI)/6.)*CA - 10./9.*NF*TR;
}

//--------------------------------------------------------------------------

double SplittingQCD::GammaQCD3 (double NF) {
  return 1./4.* (CA*CA*(245./6.-134./27.*pow2(M_PI)+11./45.*pow(M_PI,4)
                        +22./3.*ZETA3)
                +CA*NF*TR*(-418./27.+40./27.*pow2(M_PI)-56./3.*ZETA3)
                +CF*NF*TR*(-55./3.+16.*ZETA3)-16./27.*pow2(NF*TR));
}

//--------------------------------------------------------------------------

double SplittingQCD::betaQCD0 (double NF)
  { return 11./6.*CA - 2./3.*NF*TR;}

//--------------------------------------------------------------------------

double SplittingQCD::betaQCD1 (double NF)
  { return 17./6.*pow2(CA) - (5./3.*CA+CF)*NF*TR;}

//--------------------------------------------------------------------------

double SplittingQCD::betaQCD2 (double NF)
  { return 2857./432.*pow(CA,3)
    + (-1415./216.*pow2(CA) - 205./72.*CA*CF + pow2(CF)/4.) *TR*NF
    + ( 79.*CA + 66.*CF)/108.*pow2(TR*NF); }

//--------------------------------------------------------------------------

// Function to calculate the correct alphaS/2*Pi value, including
// renormalisation scale variations + threshold matching.

double SplittingQCD::as2Pi( double pT2, int orderNow, double renormMultFacNow){

  // Get beam for PDF alphaS, if necessary.
  BeamParticle* beam = NULL;
  if (beamAPtr != NULL || beamBPtr != NULL) {
    beam = (beamAPtr != NULL && particleDataPtr->isHadron(beamAPtr->id())) ? beamAPtr
         : (beamBPtr != NULL && particleDataPtr->isHadron(beamBPtr->id())) ? beamBPtr : NULL;
    if (beam == NULL && beamAPtr != 0) beam = beamAPtr;
    if (beam == NULL && beamBPtr != 0) beam = beamBPtr;
  }
  double scale       = pT2 * ( (renormMultFacNow > 0.)
                              ? renormMultFacNow : renormMultFac);
  scale              = max(scale, pow2(pTmin) );

  // Get alphaS(k*pT^2) and subtractions.
  double asPT2pi      = (usePDFalphas && beam != NULL)
                      ? beam->alphaS(scale)  / (2.*M_PI)
                      : alphaS.alphaS(scale) / (2.*M_PI);

  //int order = (orderNow > 0) ? orderNow : correctionOrder;
  int order = (orderNow > -1) ? orderNow : correctionOrder;
  order -= 1;

  // Now find the necessary thresholds so that alphaS can be matched
  // correctly.
  double m2cPhys = (usePDFalphas && beam != NULL) 
                 ? pow2(max(0.,beam->mQuarkPDF(4)))
                 : alphaS.muThres2(4);
  if ( !( (scale > m2cPhys && pT2 < m2cPhys)
       || (scale < m2cPhys && pT2 > m2cPhys) ) ) m2cPhys = -1.;
  double m2bPhys = (usePDFalphas && beam != NULL)
                 ? pow2(max(0.,beam->mQuarkPDF(5)))
                 : alphaS.muThres2(5);
  if ( !( (scale > m2bPhys && pT2 < m2bPhys)
       || (scale < m2bPhys && pT2 > m2bPhys) ) ) m2bPhys = -1.;
  vector<double> scales;
  scales.push_back(scale);
  scales.push_back(pT2);
  if (m2cPhys > 0.) scales.push_back(m2cPhys);
  if (m2bPhys > 0.) scales.push_back(m2bPhys);
  sort( scales.begin(), scales.end());
  if (scale > pT2) reverse(scales.begin(), scales.end());

  double asPT2piCorr  = asPT2pi; 
  for ( int i = 1; i< int(scales.size()); ++i) {
    double NF    = getNF( 0.5*(scales[i]+scales[i-1]) );
    double L     = log( scales[i]/scales[i-1] );
    double subt  = 0.;
    if (order > 0) subt += asPT2piCorr * betaQCD0(NF) * L;
    if (order > 2) subt += pow2( asPT2piCorr ) * ( betaQCD1(NF)*L 
                                   - pow2(betaQCD0(NF)*L) );
    if (order > 4) subt += pow( asPT2piCorr, 3) * ( betaQCD2(NF)*L
                                   - 2.5 * betaQCD0(NF)*betaQCD1(NF)*L*L
                                   + pow( betaQCD0(NF)*L, 3) );
    asPT2piCorr *= 1.0 - subt;
  }

  // Done.
  return asPT2piCorr;

}

//--------------------------------------------------------------------------

// Helper function to calculate dilogarithm.

double SplittingQCD::polevl(double x,double* coef,int N ) {
  double ans;
  int i;
  double *p;

  p = coef;
  ans = *p++;
  i = N;
    
  do
    ans = ans * x  +  *p++;
  while( --i );
    
  return ans;
}
  
//--------------------------------------------------------------------------

// Function to calculate dilogarithm.

double SplittingQCD::DiLog(double x) {

  static double cof_A[8] = {
    4.65128586073990045278E-5,
    7.31589045238094711071E-3,
    1.33847639578309018650E-1,
    8.79691311754530315341E-1,
    2.71149851196553469920E0,
    4.25697156008121755724E0,
    3.29771340985225106936E0,
    1.00000000000000000126E0,
  };
  static double cof_B[8] = {
    6.90990488912553276999E-4,
    2.54043763932544379113E-2,
    2.82974860602568089943E-1,
    1.41172597751831069617E0,
    3.63800533345137075418E0,
    5.03278880143316990390E0,
    3.54771340985225096217E0,
    9.99999999999999998740E-1,
  };

  if( x >1. ) {
    return -DiLog(1./x)+M_PI*M_PI/3.-0.5*pow2(log(x));
  }

  x = 1.-x;
  double w, y, z;
  int flag;
  if( x == 1.0 )
    return( 0.0 );
  if( x == 0.0 )
    return( M_PI*M_PI/6.0 );
    
  flag = 0;
    
  if( x > 2.0 ) {
    x = 1.0/x;
    flag |= 2;
  }
    
  if( x > 1.5 ) {
    w = (1.0/x) - 1.0;
    flag |= 2;
  }
    
  else if( x < 0.5 ) {
    w = -x;
    flag |= 1;
  }
    
  else
    w = x - 1.0;
    
  y = -w * polevl( w, cof_A, 7) / polevl( w, cof_B, 7 );
    
  if( flag & 1 )
    y = (M_PI * M_PI)/6.0  - log(x) * log(1.0-x) - y;
    
  if( flag & 2 ) {
    z = log(x);
    y = -0.5 * z * z  -  y;
  }
    
  return y;

}

//--------------------------------------------------------------------------

double SplittingQCD::softRescaleInt(int order) {
  double rescale = 1.;
  if (order > 0) rescale += alphaS2pi*GammaQCD2(3.);
  if (order > 1) rescale += pow2(alphaS2pi)*GammaQCD3(3.);
  return rescale;
}

//--------------------------------------------------------------------------

double SplittingQCD::softRescaleDiff(int order, double pT2,
  double renormMultFacNow) {
  double rescale = 1.;
  // Get alphaS and number of flavours, attach cusp factors.
  double NF      = getNF(pT2 * ( (renormMultFacNow > 0.)
                                ? renormMultFacNow : renormMultFac) );
  double asPT2pi = as2Pi(pT2, order, renormMultFacNow); 
  if (order > 0) rescale += asPT2pi       * GammaQCD2(NF);
  if (order > 1) rescale += pow2(asPT2pi) * GammaQCD3(NF);
  return rescale;
}

//--------------------------------------------------------------------------

vector<int> SplittingQCD::sharedColor(const Event& event, int iRad,
  int iRec) {
  vector<int> ret;
  int radCol(event[iRad].col()), radAcl(event[iRad].acol()),
      recCol(event[iRec].col()), recAcl(event[iRec].acol());
  if ( event[iRad].isFinal() && event[iRec].isFinal() ) {
    if (radCol != 0 && radCol == recAcl) ret.push_back(radCol);
    if (radAcl != 0 && radAcl == recCol) ret.push_back(radAcl);
  } else if ( event[iRad].isFinal() && !event[iRec].isFinal() ) {
    if (radCol != 0 && radCol == recCol) ret.push_back(radCol);
    if (radAcl != 0 && radAcl == recAcl) ret.push_back(radAcl);
  } else if (!event[iRad].isFinal() && event[iRec].isFinal() )  {
    if (radCol != 0 && radCol == recCol) ret.push_back(radCol);
    if (radAcl != 0 && radAcl == recAcl) ret.push_back(radAcl);
  } else if (!event[iRad].isFinal() && !event[iRec].isFinal() ) {
    if (radCol != 0 && radCol == recAcl) ret.push_back(radCol);
    if (radAcl != 0 && radAcl == recCol) ret.push_back(radAcl);
  }
  return ret;
}

//--------------------------------------------------------------------------

int SplittingQCD::findCol(int col, vector<int> iExc, const Event& event,
    int type) {

  int index = 0;

  int inA = 0, inB = 0;
  for (int i=event.size()-1; i > 0; --i) {
    if ( event[i].mother1() == 1 && event[i].status() != -31
      && event[i].status() != -34) { if (inA == 0) inA = i; }
    if ( event[i].mother1() == 2 && event[i].status() != -31
      && event[i].status() != -34) { if (inB == 0) inB = i; }
  }

  // Search event record for matching colour & anticolour
  for(int n = 0; n < event.size(); ++n) {
    // Skip if this index is excluded.
    if ( find(iExc.begin(), iExc.end(), n) != iExc.end() ) continue;
    if ( event[n].colType() != 0 &&  event[n].status() > 0 ) {
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
  // Search event record for matching colour & anticolour
  for(int n = event.size()-1; n > 0; --n) {
    // Skip if this index is excluded.
    if ( find(iExc.begin(), iExc.end(), n) != iExc.end() ) continue;
    if ( index == 0 && event[n].colType() != 0
      && ( n == inA || n == inB) ) {  // Check incoming
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
  // if no matching colour / anticolour has been found, return false
  if ( type == 1 && index < 0) return abs(index);
  if ( type == 2 && index > 0) return abs(index);
  return 0;

}

//==========================================================================

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_Q2QGG::canRadiate (const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].isQuark() );
}

double fsr_qcd_Q2QGG::gaugeFactor ( int, int )        { return CF;}
double fsr_qcd_Q2QGG::symmetryFactor ( int, int )     { return 1.;}

vector<pair<int,int> > fsr_qcd_Q2QGG::radAndEmtCols(int iRad, int colType,
  Event state) { 
  int newCol1     = state.nextColTag();
  int newCol2     = state.nextColTag();
  int colRadAft   = (colType > 0) ? newCol2 : state[iRad].col();
  int acolRadAft  = (colType > 0) ? state[iRad].acol() : newCol2;
  int colEmtAft1  = (colType > 0) ? state[iRad].col() : newCol1;
  int acolEmtAft1 = (colType > 0) ? newCol1 : state[iRad].acol();
  int colEmtAft2  = (colType > 0) ? newCol1 : newCol2;
  int acolEmtAft2 = (colType > 0) ? newCol2 : newCol1;

  // Also remember colors for "intermediate" particles in 1->3 splitting.
  if ( colType > 0) {
    splitInfo.extras.insert(make_pair("colRadInt",  newCol1));
    splitInfo.extras.insert(make_pair("acolRadInt", state[iRad].acol()));
    splitInfo.extras.insert(make_pair("colEmtInt",  state[iRad].col()));
    splitInfo.extras.insert(make_pair("acolEmtInt", newCol1));
  } else {
    splitInfo.extras.insert(make_pair("colRadInt",  state[iRad].col()));
    splitInfo.extras.insert(make_pair("acolRadInt", newCol1));
    splitInfo.extras.insert(make_pair("colEmtInt",  newCol1));
    splitInfo.extras.insert(make_pair("acolEmtInt", state[iRad].acol()));
  }

  return createvector<pair<int,int> >
    (make_pair(colRadAft, acolRadAft))
    (make_pair(colEmtAft1, acolEmtAft1))
    (make_pair(colEmtAft2, acolEmtAft2));
}

int fsr_qcd_Q2QGG::radBefID(int idRA, int) {
  if (particleDataPtr->isQuark(idRA)) return idRA;
  return 0;
}

pair<int,int> fsr_qcd_Q2QGG::radBefCols(
  int, int, 
  int, int) {
  return make_pair(0,0);
}

// Pick z for new splitting.
double fsr_qcd_Q2QGG::zSplit(double zMinAbs, double, double m2dip) {
  double Rz        = rndmPtr->flat();
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p         = pow( 1. + pow2(1-zMinAbs)/kappaMin2, Rz );
  double res       = 1. - sqrt( p - 1. )*sqrt(kappaMin2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qcd_Q2QGG::overestimateInt(double zMinAbs, double,
  double, double m2dip, int orderNow) {
  // Q -> QG, soft part (currently also used for collinear part).
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  //double wt        = 1e2;
  double wt        = preFac * softRescaleInt(order)
                     *2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappaMin2);
  return wt;
}

// Return overestimate for new splitting.
double fsr_qcd_Q2QGG::overestimateDiff(double z, double m2dip, int orderNow) {
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * softRescaleInt(order)
                     *2. * (1.-z) / ( pow2(1.-z) + kappaMin2);
  return wt;
}

// Return kernel for new splitting.
//bool fsr_qcd_Q2QGG::calc(const Event& state, int) { 
bool fsr_qcd_Q2QGG::calc(const Event&, int) { 

  //// Required kinematics.
  //double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2);
  //
  //Event trialEvent(state);
  //bool physical = false;
  //if (splitInfo.recBef()->isFinal)
  //  physical = fsr->branch_FF(trialEvent, true, &splitInfo);
  //else
  //  physical = fsr->branch_FI(trialEvent, true, &splitInfo);
  //
  //// Get invariants.
  //Vec4 pa(trialEvent[splitInfo.iRadAft].p());
  //Vec4 pk(trialEvent[splitInfo.iRecAft].p());
  //Vec4 pi(trialEvent[splitInfo.iEmtAft].p());
  //Vec4 pj(trialEvent[splitInfo.iEmtAft2].p());

  map<string,double> wts;
  wts.insert(make_pair("base", 1e5));

  // Store kernel values and return.
  clearKernels();
  for (map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
    kernelVals.insert(make_pair( it->first, it->second));
  return true;

}

//==========================================================================

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_G2GGG::canRadiate (const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].id() == 21);
}


double fsr_qcd_G2GGG::gaugeFactor ( int, int )        { return 2.*CA;}
double fsr_qcd_G2GGG::symmetryFactor ( int, int )     { return 0.5;}

vector<pair<int,int> > fsr_qcd_G2GGG::radAndEmtCols(int iRad, int colType,
  Event state) { 
  int newCol1     = state.nextColTag();
  int newCol2     = state.nextColTag();
  int colRadAft   = (colType > 0) ? newCol2 : state[iRad].col();
  int acolRadAft  = (colType > 0) ? state[iRad].acol() : newCol2;
  int colEmtAft1  = (colType > 0) ? state[iRad].col() : newCol1;
  int acolEmtAft1 = (colType > 0) ? newCol1 : state[iRad].acol();
  int colEmtAft2  = (colType > 0) ? newCol1 : newCol2;
  int acolEmtAft2 = (colType > 0) ? newCol2 : newCol1;

  // Also remember colors for "intermediate" particles in 1->3 splitting.
  if ( colType > 0) {
    splitInfo.extras.insert(make_pair("colRadInt",  newCol1));
    splitInfo.extras.insert(make_pair("acolRadInt", state[iRad].acol()));
    splitInfo.extras.insert(make_pair("colEmtInt",  state[iRad].col()));
    splitInfo.extras.insert(make_pair("acolEmtInt", newCol1));
  } else {
    splitInfo.extras.insert(make_pair("colRadInt",  state[iRad].col()));
    splitInfo.extras.insert(make_pair("acolRadInt", newCol1));
    splitInfo.extras.insert(make_pair("colEmtInt",  newCol1));
    splitInfo.extras.insert(make_pair("acolEmtInt", state[iRad].acol()));
  }

  return createvector<pair<int,int> >
    (make_pair(colRadAft, acolRadAft))
    (make_pair(colEmtAft1, acolEmtAft1))
    (make_pair(colEmtAft2, acolEmtAft2));
} 

int fsr_qcd_G2GGG::radBefID(int, int) {
  return 21;
}

pair<int,int> fsr_qcd_G2GGG::radBefCols(
  int, int, 
  int, int) {
  return make_pair(0,0);
}

// Pick z for new splitting.
double fsr_qcd_G2GGG::zSplit(double zMinAbs, double, double m2dip) {
  // Just pick according to soft.
  double R         = rndmPtr->flat();
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p         = pow( 1. + pow2(1-zMinAbs)/kappaMin2, R );
  double res       = 1. - sqrt( p - 1. )*sqrt(kappaMin2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qcd_G2GGG::overestimateInt(double zMinAbs, double,
  double, double m2dip, int orderNow) {

  // Overestimate by soft
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  //double wt        = 1e2;
  double wt        = preFac * softRescaleInt(order)
                     *0.5 * log( 1. + pow2(1.-zMinAbs)/kappaMin2);
  return wt;
}

// Return overestimate for new splitting.
double fsr_qcd_G2GGG::overestimateDiff(double z, double m2dip, int orderNow) {
  // Overestimate by soft
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * softRescaleInt(order)
                     *(1.-z) / ( pow2(1.-z) + kappaMin2);
  return wt;
}

// Return kernel for new splitting.
//bool fsr_qcd_G2GGG::calc(const Event& state, int) { 
bool fsr_qcd_G2GGG::calc(const Event&, int) { 

  //// Required kinematics.
  //double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2);
  //
  //Event trialEvent(state);
  //bool physical = false;
  //if (splitInfo.recBef()->isFinal)
  //  physical = fsr->branch_FF(trialEvent, true, &splitInfo);
  //else
  //  physical = fsr->branch_FI(trialEvent, true, &splitInfo);
  //
  //// Get invariants.
  //Vec4 pa(trialEvent[splitInfo.iRadAft].p());
  //Vec4 pk(trialEvent[splitInfo.iRecAft].p());
  //Vec4 pi(trialEvent[splitInfo.iEmtAft].p());
  //Vec4 pj(trialEvent[splitInfo.iEmtAft2].p());

  map<string,double> wts;
  wts.insert(make_pair("base", 1e5));

  // Store kernel values and return.
  clearKernels();
  for (map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
    kernelVals.insert(make_pair( it->first, it->second));
  return true;

}

//==========================================================================

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_G2QQG::canRadiate (const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].id() == 21);
}

double fsr_qcd_G2QQG::gaugeFactor ( int, int )        { return NF_qcd_fsr*TR;}
double fsr_qcd_G2QQG::symmetryFactor ( int, int )     { return 0.5;}


vector<pair<int,int> > fsr_qcd_G2QQG::radAndEmtCols(int iRad, int colType,
  Event state) { 
  int newCol1     = state.nextColTag();
  int colRadAft   = (colType > 0) ? newCol1 : 0;
  int acolRadAft  = (colType > 0) ? 0 : newCol1;
  int colEmtAft1  = (colType > 0) ? 0 : state[iRad].col();
  int acolEmtAft1 = (colType > 0) ? state[iRad].acol() : 0;
  int colEmtAft2  = (colType > 0) ? state[iRad].col() : newCol1;
  int acolEmtAft2 = (colType > 0) ? newCol1 : state[iRad].acol();

  // Also remember colors for "intermediate" particles in 1->3 splitting.
  if ( colType > 0) {
    splitInfo.extras.insert(make_pair("colRadInt", state[iRad].col()));
    splitInfo.extras.insert(make_pair("acolRadInt", 0));
  } else {
    splitInfo.extras.insert(make_pair("colRadInt", 0));
    splitInfo.extras.insert(make_pair("acolRadInt", state[iRad].acol()));
  }
  splitInfo.extras.insert(make_pair("colEmtInt", colEmtAft1));
  splitInfo.extras.insert(make_pair("acolEmtInt", acolEmtAft1));

  return createvector<pair<int,int> >
    (make_pair(colRadAft, acolRadAft))
    (make_pair(colEmtAft1, acolEmtAft1))
    (make_pair(colEmtAft2, acolEmtAft2));
}

int fsr_qcd_G2QQG::radBefID(int, int) {
  return 21;
}

pair<int,int> fsr_qcd_G2QQG::radBefCols(
  int, int, 
  int, int) {
  return make_pair(0,0);
}

// Pick z for new splitting.
double fsr_qcd_G2QQG::zSplit(double zMinAbs, double, double m2dip) {
  // Just pick according to soft.
  double R         = rndmPtr->flat();
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p         = pow( 1. + pow2(1-zMinAbs)/kappaMin2, R );
  double res       = 1. - sqrt( p - 1. )*sqrt(kappaMin2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qcd_G2QQG::overestimateInt(double zMinAbs, double zMaxAbs,
  double, double, int) {
  // Overestimate by soft
  double preFac = symmetryFactor() * gaugeFactor();
  //double wt        = 1e2;
  double wt     = 2.*preFac * 0.5 * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double fsr_qcd_G2QQG::overestimateDiff(double z, double m2dip, int orderNow) {
  // Overestimate by soft
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * softRescaleInt(order)
                     *(1.-z) / ( pow2(1.-z) + kappaMin2);
  return wt;
}

// Return kernel for new splitting.
//bool fsr_qcd_G2QQG::calc(const Event& state, int) { 
bool fsr_qcd_G2QQG::calc(const Event&, int) { 

  //// Required kinematics.
  //double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2);
  //
  //Event trialEvent(state);
  //bool physical = false;
  //if (splitInfo.recBef()->isFinal)
  //  physical = fsr->branch_FF(trialEvent, true, &splitInfo);
  //else
  //  physical = fsr->branch_FI(trialEvent, true, &splitInfo);
  //
  //// Get invariants.
  //Vec4 pa(trialEvent[splitInfo.iRadAft].p());
  //Vec4 pk(trialEvent[splitInfo.iRecAft].p());
  //Vec4 pi(trialEvent[splitInfo.iEmtAft].p());
  //Vec4 pj(trialEvent[splitInfo.iEmtAft2].p());

  map<string,double> wts;
  wts.insert(make_pair("base", 1e5));

  // Store kernel values and return.
  clearKernels();
  for (map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
    kernelVals.insert(make_pair( it->first, it->second));
  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function Q->QG (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_Q2QG::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].isQuark() );
}

int fsr_qcd_Q2QG::kinMap()                 { return 1;}
int fsr_qcd_Q2QG::motherID(int idDaughter) { return idDaughter;}
int fsr_qcd_Q2QG::sisterID(int)            { return 21;}
double fsr_qcd_Q2QG::gaugeFactor ( int, int )        { return CF;}
double fsr_qcd_Q2QG::symmetryFactor ( int, int )     { return 1.;}

int fsr_qcd_Q2QG::radBefID(int idRA, int) {
  if (particleDataPtr->isQuark(idRA)) return idRA;
  return 0;
}

pair<int,int> fsr_qcd_Q2QG::radBefCols(
  int colRadAfter, int, 
  int colEmtAfter, int acolEmtAfter) {
  bool isQuark = (colRadAfter > 0);
  if (isQuark) return make_pair(colEmtAfter,0);
  return make_pair(0,acolEmtAfter);
}

vector <int> fsr_qcd_Q2QG::recPositions( const Event& state, int iRad, int iEmt) {

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();
  int colShared = (colRad  > 0 && colRad == acolEmt) ? colRad
                : (acolRad > 0 && colEmt == acolRad) ? colEmt : 0;
  // Particles to exclude from colour tracing.
  vector<int> iExc(1,iRad); iExc.push_back(iEmt);

  // Find partons connected via emitted colour line.
  vector<int> recs;
  if ( colEmt != 0 && colEmt != colShared) {
    int acolF = findCol(colEmt, iExc, state, 1);
    int  colI = findCol(colEmt, iExc, state, 2);
    if (acolF  > 0 && colI == 0) recs.push_back (acolF);
    if (acolF == 0 && colI >  0) recs.push_back (colI);
  }
  // Find partons connected via emitted anticolour line.
  if ( acolEmt != 0 && acolEmt != colShared) {
    int  colF = findCol(acolEmt, iExc, state, 2);
    int acolI = findCol(acolEmt, iExc, state, 1);
    if ( colF  > 0 && acolI == 0) recs.push_back (colF);
    if ( colF == 0 && acolI >  0) recs.push_back (acolI);
  }
  // Done.
  return recs;
} 

// Pick z for new splitting.
double fsr_qcd_Q2QG::zSplit(double zMinAbs, double /*zMaxAbs*/, double m2dip) {
  double Rz        = rndmPtr->flat();

  /*// For testing only. 
  return zMinAbs + Rz*(zMaxAbs-zMinAbs);*/

  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p         = pow( 1. + pow2(1-zMinAbs)/kappaMin2, Rz );
  double res       = 1. - sqrt( p - 1. )*sqrt(kappaMin2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qcd_Q2QG::overestimateInt(double zMinAbs, double /*zMaxAbs*/,
  double, double m2dip, int orderNow) {

  /*// For testing only. 
  return 0.5*10.*(zMaxAbs - zMinAbs);*/

  // Q -> QG, soft part (currently also used for collinear part).
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * softRescaleInt(order)
                     *2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappaMin2);
  return wt;
}

// Return overestimate for new splitting.
double fsr_qcd_Q2QG::overestimateDiff(double z, double m2dip, int orderNow) {

  /*// For testing only. 
  return 0.5*10.;*/

  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * softRescaleInt(order)
                     *2. * (1.-z) / ( pow2(1.-z) + kappaMin2);
  return wt;
}

// Return kernel for new splitting.
bool fsr_qcd_Q2QG::calc(const Event& state, int orderNow) { 

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

  /*// For testing only. 
  map<string,double> w;
  double ww = 10.*(z-0.5);
  //if (z < 0.3) ww *= -1.;
  //if (z > 0.9) ww *= -z;
  //double ww = 10.*(z-0.5) / 1.2;
  if (z < 0.3) ww *= -1.;
  if (z > 0.9) ww *= -z;

  w.insert( make_pair("base", ww) );
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      w.insert( make_pair("Variations:muRfsrDown", ww ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      w.insert( make_pair("Variations:muRfsrUp", ww ));
  }

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = w.begin(); it != w.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));
  return true;*/

  // Calculate kernel.
  // Note: We are calculating the z <--> 1-z symmetrised kernel here,
  // and later multiply with z to project out Q->QQ,
  // i.e. the gluon is soft and the quark is identified.
  double preFac = symmetryFactor() * gaugeFactor();
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  double kappa2 = pT2/m2dip;

  map<string,double> wts;
  double wt_base_as1 = preFac * ( 2.* (1.-z) / ( pow2(1.-z) + kappa2) );
  //wts.insert( make_pair("base",
  //  preFac * softRescaleDiff( order, pT2, renormMultFac)
  //         * ( 2.* (1.-z) / ( pow2(1.-z) + kappa2) ) ) );
  //
  //if (doVariations) {
  //  // Create muR-variations.
  //  if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
  //    wts.insert( make_pair("Variations:muRfsrDown", preFac
  //      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRfsrDown")) 
  //      *( 2.* (1.-z) / ( pow2(1.-z) + kappa2) ) ));
  //  if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
  //    wts.insert( make_pair("Variations:muRfsrUp", preFac
  //      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRfsrUp")) 
  //      *( 2.* (1.-z) / ( pow2(1.-z) + kappa2) ) ));
  //}

  wts.insert( make_pair("base", softRescaleDiff( order, pT2, renormMultFac)
                                * wt_base_as1 ) );
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt_base_as1
        * softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRfsrDown")) ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt_base_as1
        * softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRfsrUp")) ));
  }

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  // Add collinear term for massless splittings.
  if (!doMassive) {
    wt_base_as1 += -preFac * ( 1.+z );
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second +=  -preFac * ( 1.+z );
  }

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
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second += preFac * massCorr;

    wt_base_as1 += preFac * massCorr;
  }

  // Add NLO term.
  if (!doMassive && order >= 3) {
    for ( map<string,double>::iterator it = wts.begin(); it !=wts.end(); ++it){

      double mukf = 1.;
      if (it->first == "base")
        mukf = renormMultFac;
      else if (it->first == "Variations:muRfsrDown")
        mukf = settingsPtr->parm("Variations:muRfsrDown");
      else if (it->first == "Variations:muRfsrUp")
        mukf = settingsPtr->parm("Variations:muRfsrUp");
      else continue;

      double NF          = getNF(pT2 * mukf);
      double alphasPT2pi = as2Pi(pT2, order, mukf);
      double TF          = TR*NF;
      double pqq1 = preFac / (18*(z-1)) * (
((-1 + z)*(4*TF*(-10 + z*(-37 + z*(29 + 28*z))) + z*(90*CF*(-1 + z) + CA*(53 - 187*z + 3*(1 + z)*pow2(M_PI)))) +
     3*z*log(z)*(34*TF + 12*(CF - CF*z + 2*TF*z) - 2*(9*CF + TF*(17 + 8*z))*pow2(z) - 12*CF*log(1 - z)*(1 + pow2(z)) -
        CA*(17 + 5*pow2(z)) - 3*log(z)*(CA - 3*CF + 2*TF + (CA - 5*CF - 2*TF)*pow2(z))))/z
    ); 
      // replace 1/z term in NLO kernel with z/(z*z+kappa2) to restore sum rule.
      pqq1 += - preFac * 0.5 * 40./9. * TF * ( z /(z*z + kappa2) - 1./z);
      // Add NLO term.
      it->second += alphasPT2pi*pqq1;

    }
  }

  // Now multiply with z to project out Q->QG,
  // i.e. the gluon is soft and the quark is identified.
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    it->second *= z;    

  wt_base_as1 *= z;
  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2",
    wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function Q->GQ (FSR)
// At leading order, this can be combined with Q->QG because of symmetry. Since
// this is no longer possible at NLO, we keep the kernels separately.

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_Q2GQ::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].isQuark() );
}

int fsr_qcd_Q2GQ::kinMap()                 { return 1;}
int fsr_qcd_Q2GQ::motherID(int idDaughter) { return idDaughter;}
int fsr_qcd_Q2GQ::sisterID(int)            { return 21;}
double fsr_qcd_Q2GQ::gaugeFactor ( int, int )        { return CF;}
double fsr_qcd_Q2GQ::symmetryFactor ( int, int )     { return 1.;}

int fsr_qcd_Q2GQ::radBefID(int idRad, int idEmt) {
  if (idRad == 21 && particleDataPtr->isQuark(idEmt)) return idEmt;
  if (idEmt == 21 && particleDataPtr->isQuark(idRad)) return idRad;
  return 0;
  //return (idRad == 21) ? idEmt : idRad;
}

pair<int,int> fsr_qcd_Q2GQ::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  //bool isQuark = (colEmtAfter > 0);
  //if (isQuark) return make_pair(colRadAfter,0);
  //return make_pair(0,acolRadAfter);
  int colE  = (colEmtAfter*acolEmtAfter == 0 && colRadAfter*acolRadAfter != 0)
            ? colEmtAfter : colRadAfter; 
  int colR  = (colEmtAfter*acolEmtAfter == 0 && colRadAfter*acolRadAfter != 0)
            ? colRadAfter : colEmtAfter; 
  int acolR = (colEmtAfter*acolEmtAfter == 0 && colRadAfter*acolRadAfter != 0)
            ? acolRadAfter : acolEmtAfter; 

  bool isQuark = (colE > 0);
  if (isQuark) return make_pair(colR,0);
  return make_pair(0,acolR);
}

vector <int> fsr_qcd_Q2GQ::recPositions( const Event& state, int iRad, int iEmt) {

  // For Q->GQ, swap radiator and emitted, since we now have to trace the
  // radiator's colour connections.
  if ( state[iEmt].idAbs() < 20 && state[iRad].id() == 21) swap( iRad, iEmt);

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();
  int colShared = (colRad  > 0 && colRad == acolEmt) ? colRad
                : (acolRad > 0 && colEmt == acolRad) ? colEmt : 0;
  // Particles to exclude from colour tracing.
  vector<int> iExc(1,iRad); iExc.push_back(iEmt);

  // Find partons connected via emitted colour line.
  vector<int> recs;
  if ( colEmt != 0 && colEmt != colShared) {
    int acolF = findCol(colEmt, iExc, state, 1);
    int  colI = findCol(colEmt, iExc, state, 2);
    if (acolF  > 0 && colI == 0) recs.push_back (acolF);
    if (acolF == 0 && colI >  0) recs.push_back (colI);
  }
  // Find partons connected via emitted anticolour line.
  if ( acolEmt != 0 && acolEmt != colShared) {
    int  colF = findCol(acolEmt, iExc, state, 2);
    int acolI = findCol(acolEmt, iExc, state, 1);
    if ( colF  > 0 && acolI == 0) recs.push_back (colF);
    if ( colF == 0 && acolI >  0) recs.push_back (acolI);
  }
  // Done.
  return recs;
}

// Pick z for new splitting.
double fsr_qcd_Q2GQ::zSplit(double zMinAbs, double, double m2dip) {
  double Rz        = rndmPtr->flat();
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p         = pow( 1. + pow2(1-zMinAbs)/kappaMin2, Rz );
  double res       = 1. - sqrt( p - 1. )*sqrt(kappaMin2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qcd_Q2GQ::overestimateInt(double zMinAbs, double,
  double, double m2dip, int orderNow) {
  // Q -> QG, soft part (currently also used for collinear part).
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * 2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappaMin2);

  // Rescale with soft cusp term only if NLO corrections are absent.
  // This choice is purely heuristical to improve LEP description.
  if ( ( correctionOrder > 0 && correctionOrder <= 2 )
    //|| ( orderNow > 0        && orderNow <= 2 ) )
    || ( orderNow > -1       && orderNow <= 2 ) )
    wt *= softRescaleInt(order);

  return wt;
}

// Return overestimate for new splitting.
double fsr_qcd_Q2GQ::overestimateDiff(double z, double m2dip, int orderNow) {
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * 2.* (1.-z) / ( pow2(1.-z) + kappaMin2);
  // Rescale with soft cusp term only if NLO corrections are absent.
  // This choice is purely heuristical to improve LEP description.
  if ( ( correctionOrder > 0 && correctionOrder <= 2 )
    //|| ( orderNow > 0        && orderNow <= 2 ) )
    || ( orderNow > -1       && orderNow <= 2 ) )
    wt *= softRescaleInt(order);
  return wt;
}

// Return kernel for new splitting.
bool fsr_qcd_Q2GQ::calc(const Event& state, int orderNow) {

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
  double preFac = symmetryFactor() * gaugeFactor();
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  double kappa2 = pT2/m2dip;

  map<string,double> wts;
  double wt_base_as1 = preFac * ( 2.* (1.-z) / ( pow2(1.-z) + kappa2) );
  //wts.insert( make_pair("base", preFac
  //  *( 2.* (1.-z) / ( pow2(1.-z) + kappa2) ) ) );
  //if (doVariations) {
  //  // Create muR-variations.
  //  if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
  //    wts.insert( make_pair("Variations:muRfsrDown", preFac
  //      *( 2.* (1.-z) / ( pow2(1.-z) + kappa2) ) ));
  //  if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
  //    wts.insert( make_pair("Variations:muRfsrUp", preFac
  //      *( 2.* (1.-z) / ( pow2(1.-z) + kappa2) ) ));
  //}

  wts.insert( make_pair("base", wt_base_as1 ));

  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown",  wt_base_as1 ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp",  wt_base_as1 ));
  }


  // Rescale with soft cusp term only if NLO corrections are absent.
  // This choice is purely heuristical to improve LEP description.
  bool doRescale = ( ( correctionOrder > 0 && correctionOrder <= 2 )
                  //|| ( orderNow > 0        && orderNow <= 2 ) );
                  || ( orderNow > -1       && orderNow <= 2 ) );
  if (doRescale) {
  wts["base"] *= softRescaleDiff( order, pT2, renormMultFac);
  if (doVariations && settingsPtr->parm("Variations:muRfsrDown") != 1.)
    wts["Variations:muRfsrDown"] *= softRescaleDiff( order, pT2,
      settingsPtr->parm("Variations:muRfsrDown"));
  if (doVariations && settingsPtr->parm("Variations:muRfsrUp")   != 1.)
    wts["Variations:muRfsrUp"] *= softRescaleDiff( order, pT2,
      settingsPtr->parm("Variations:muRfsrUp")); 
  }

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  // Add collinear term for massless splittings.
  if (!doMassive) {
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second += -preFac * ( 1.+z );
     wt_base_as1 += -preFac * ( 1.+z );
  }

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
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second += preFac * massCorr;

     wt_base_as1 += preFac * massCorr;
  }

  // Add NLO term.
  if (!doMassive && order >= 3){
    for ( map<string,double>::iterator it = wts.begin(); it !=wts.end(); ++it){
      double mukf = 1.;
      if (it->first == "base")
        mukf = renormMultFac;
      else if (it->first == "Variations:muRfsrDown")
        mukf = settingsPtr->parm("Variations:muRfsrDown");
      else if (it->first == "Variations:muRfsrUp")
        mukf = settingsPtr->parm("Variations:muRfsrUp");
      else continue;

      // Evaluate kernel copied from Mathematica with 1-z!
      double x = 1.-z;
      double NF          = getNF(pT2 * mukf);
      double alphasPT2pi = as2Pi(pT2, order, mukf);
      double TF          = TR*NF;
      double pqg1 = preFac * (
 (9*CF*x*(-1 + 9*x) + 144*(CA - CF)*(2 + (-2 + x)*x)*DiLog(x) + 36*CA*(2 + x*(2 + x))*DiLog(1/(1 + x)) -
     2*CA*(-17 + 9*(-5 + x)*x + 44*pow(x,3) + 3*pow2(M_PI)*(2 + pow2(x))) +
     3*(12*log(1 - x)*((3*CA - 2*CF)*(2 + (-2 + x)*x)*log(x) + (-CA + CF)*pow2(x)) +
        log(x)*(3*CF*(-16 + x)*x + 2*CA*(-18 + x*(24 + x*(27 + 8*x))) - 3*log(x)*(CF*(-2 + x)*x + CA*(8 + 4*x + 6*pow2(x)))) -
        6*(CA - CF)*(2 + (-2 + x)*x)*pow2(log(1 - x)) + 6*CA*(2 + x*(2 + x))*pow2(log(1 + x))))/(18.*x) 
      );
      // replace 1/z term in NLO kernel with z/(z*z+kappa2) to restore sum rule.
      pqg1 += preFac * 0.5 * 40./9. * TF * ( x /(x*x + kappa2) - 1./x);
      // Add NLO term.
      it->second  += alphasPT2pi*pqg1;

    } 

  }

  // Now multiply with (1-z) to project out Q->GQ,
  // i.e. the quark is soft and the gluon is identified.
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    it->second *= (1-z);    

  wt_base_as1 *= (1-z);
  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2",
    wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function G->GG (FSR)
// We now split this kernel into two pieces, as the soft emitted gluon
// is identified as NLO. Thus, it is good to have two kernels for g -> g1 g2,
// one where g1 is soft, and one where g2 is soft.

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_G2GG1::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].id() == 21 );
}

int fsr_qcd_G2GG1::kinMap()                 { return 1;}
int fsr_qcd_G2GG1::motherID(int)            { return 21;}
int fsr_qcd_G2GG1::sisterID(int)            { return 21;}
double fsr_qcd_G2GG1::gaugeFactor ( int, int )        { return 2.*CA;}
double fsr_qcd_G2GG1::symmetryFactor ( int, int )     { return 0.5;}

int fsr_qcd_G2GG1::radBefID(int, int){ return 21;}
pair<int,int> fsr_qcd_G2GG1::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  int colRemove = (colRadAfter == acolEmtAfter)
                ? colRadAfter : acolRadAfter;
  int col       = (colRadAfter == colRemove)
                ? colEmtAfter : colRadAfter;
  int acol      = (acolRadAfter == colRemove)
                ? acolEmtAfter : acolRadAfter;
  return make_pair(col,acol);
}

vector <int> fsr_qcd_G2GG1::recPositions( const Event& state, int iRad, int iEmt) {

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();
  int colShared = (colRad  > 0 && colRad == acolEmt) ? colRad
                : (acolRad > 0 && colEmt == acolRad) ? colEmt : 0;
  // Particles to exclude from colour tracing.
  vector<int> iExc(1,iRad); iExc.push_back(iEmt);

  // Find partons connected via emitted colour line.
  vector<int> recs;
  if ( colEmt != 0 && colEmt != colShared) {
    int acolF = findCol(colEmt, iExc, state, 1);
    int  colI = findCol(colEmt, iExc, state, 2);
    if (acolF  > 0 && colI == 0) recs.push_back (acolF);
    if (acolF == 0 && colI >  0) recs.push_back (colI);
  }
  // Find partons connected via emitted anticolour line.
  if ( acolEmt != 0 && acolEmt != colShared) {
    int  colF = findCol(acolEmt, iExc, state, 2);
    int acolI = findCol(acolEmt, iExc, state, 1);
    if ( colF  > 0 && acolI == 0) recs.push_back (colF);
    if ( colF == 0 && acolI >  0) recs.push_back (acolI);
  }
  // Done.
  return recs;
}

// Pick z for new splitting.
double fsr_qcd_G2GG1::zSplit(double zMinAbs, double, double m2dip) {
  // Just pick according to soft.
  double R         = rndmPtr->flat();
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p         = pow( 1. + pow2(1-zMinAbs)/kappaMin2, R );
  double res       = 1. - sqrt( p - 1. )*sqrt(kappaMin2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qcd_G2GG1::overestimateInt(double zMinAbs, double,
  double, double m2dip, int orderNow) {

  // Overestimate by soft
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * softRescaleInt(order)
                     *0.5 * log( 1. + pow2(1.-zMinAbs)/kappaMin2);
  return wt;
}

// Return overestimate for new splitting.
double fsr_qcd_G2GG1::overestimateDiff(double z, double m2dip, int orderNow) {
  // Overestimate by soft
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * softRescaleInt(order)
                     *(1.-z) / ( pow2(1.-z) + kappaMin2);
  return wt;
}

// Return kernel for new splitting.
bool fsr_qcd_G2GG1::calc(const Event& state, int orderNow) {

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

  double preFac = symmetryFactor() * gaugeFactor();
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  double kappa2 = pT2/m2dip;

  // Calculate kernel.
  // Note: We are calculating the z <--> 1-z symmetrised kernel here,
  // and later multiply with z to project out one part.
  map<string,double> wts;
  double wt_base_as1 = preFac * (1.-z) / ( pow2(1.-z) + kappa2);
  //wts.insert( make_pair("base",
  //  preFac * softRescaleDiff( order, pT2, renormMultFac)
  //         * (1.-z) / ( pow2(1.-z) + kappa2) ));
  //if (doVariations) {
  //  // Create muR-variations.
  //  if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
  //    wts.insert( make_pair("Variations:muRfsrDown", preFac
  //      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRfsrDown")) 
  //      *(1.-z) / ( pow2(1.-z) + kappa2) ));
  //  if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
  //    wts.insert( make_pair("Variations:muRfsrUp", preFac
  //      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRfsrUp")) 
  //      *(1.-z) / ( pow2(1.-z) + kappa2) ));
  //}

  wts.insert( make_pair("base", wt_base_as1
    * softRescaleDiff( order, pT2, renormMultFac) ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt_base_as1
      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRfsrDown")) ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt_base_as1
      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRfsrUp")) ));
  }

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  // Add collinear term for massless splittings.
  if (!doMassive) {
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second += preFac * ( -1. + 0.5 * z*(1.-z) );
    wt_base_as1 += preFac * ( -1. + 0.5 * z*(1.-z) );
  }

  // Add collinear term for massive splittings.
  if (doMassive) {

    double vijk = 1.;

    // splitType == 2 -> Massive FF
    if (splitType == 2) {
      // Calculate CS variables.
      double yCS = kappa2 / (1.-z);
      double nu2Rad = m2Rad/m2dip; 
      double nu2Emt = m2Emt/m2dip; 
      double nu2Rec = m2Rec/m2dip; 
      vijk          = pow2(1.-yCS) - 4.*(yCS+nu2Rad+nu2Emt)*nu2Rec;
      vijk          = sqrt(vijk) / (1-yCS);

    // splitType ==-2 -> Massive FI
    } else if (splitType ==-2) {
      // No changes, as initial recoiler is massless!
      vijk          = 1.; 
    }

    // Add correction for massive splittings.
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second += preFac * 1./ vijk * ( -1. + 0.5 * z*(1.-z) );

    wt_base_as1 += preFac * 1./ vijk * ( -1. + 0.5 * z*(1.-z) );
  }

  // Add NLO term.
  if (!doMassive && order >= 3 ) {
    for ( map<string,double>::iterator it = wts.begin(); it !=wts.end(); ++it){
      double mukf = 1.;
      if (it->first == "base")
        mukf = renormMultFac;
      else if (it->first == "Variations:muRfsrDown")
        mukf = settingsPtr->parm("Variations:muRfsrDown");
      else if (it->first == "Variations:muRfsrUp")
        mukf = settingsPtr->parm("Variations:muRfsrUp");
      else continue;

      double NF          = getNF(pT2 * mukf);
      double alphasPT2pi = as2Pi(pT2, order, mukf);
      double TF          = TR*NF;
      // Anatomy of factors of two:
      // One factor of 0.5 since LO kernel is split into "z" and "1-z" parts, 
      // while the NLO kernel is NOT split into these structures.
      // Another factor of 0.5 enters because the LO kernel above does not
      // include a "2" (this "2" is in preFac), while the NLO kernel in the
      // Mathematica file does include the factor of "2".
      double x=z;
      double pgg1   = preFac * 0.5 * 0.5 / ( 18*x*(pow2(x)-1) ) * (
TF*(4*(-1 + x)*(-23 + x*(6 + x*(10 + x*(4 + 23*x)))) + 24*(1 + x)*(2 + (-1 + x)*x*(3 + x*(-3 + 2*x)))*log(x)) +
   (CF*TF*(-12*(1 + x)*(8 + x*(7 - x*(2 + x)*(-3 + 8*x)))*log(x) - 8*(1 + x)*(23 + x*(14 + 41*x))*pow2(-1 + x) +
        36*(-1 + x)*x*pow2(1 + x)*pow2(log(x))))/CA + 72*CA*(-1 + x)*DiLog(1/(1 + x))*pow2(1 + x + pow2(x)) +
   CA*(-6*(1 + x)*(-22 + x*(11 + x*(30 + x*(-19 + 22*x))))*log(x) +
      (1 - x)*(x*(1 + x)*(25 + 109*x) + 6*(2 + x*(1 + 2*x*(1 + x)))*pow2(M_PI)) - 72*(1 + x)*log(1 - x)*log(x)*pow2(1 + (-1 + x)*x) +
      36*(2 + x*(1 + (-4 + x)*(-1 + x)*x*(1 + x)))*pow2(log(x)) + 36*(-1 + x)*pow2(log(1 + x))*pow2(1 + x + pow2(x))) 
      ); 
      // replace 1/z term in NLO kernel with z/(z*z+kappa2) to restore sum rule.
      // Note: Colour factor is CA, not CF!
      pgg1 += preFac * 0.5 * 0.5 * 40./9. * TF * ( x /(x*x + kappa2) - 1./x);
      // Add NLO term.
      it->second += alphasPT2pi*pgg1;
    }
  }

  // Multiply with z to project out part where emitted gluon is soft.
  // (the radiator is identified)
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
    it->second *= z;    

  wt_base_as1 *= z;
  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2",
    wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function G->GG (FSR)
// We now split this kernel into two pieces, as the soft emitted gluon
// is identified as NLO. Thus, it is good to have two kernels for g -> g1 g2,
// one where g1 is soft, and one where g2 is soft.

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_G2GG2::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].id() == 21 );
}

int fsr_qcd_G2GG2::kinMap()                 { return 1;}
int fsr_qcd_G2GG2::motherID(int)            { return 21;}
int fsr_qcd_G2GG2::sisterID(int)            { return 21;}
double fsr_qcd_G2GG2::gaugeFactor ( int, int )        { return 2.*CA;}
double fsr_qcd_G2GG2::symmetryFactor ( int, int )     { return 0.5;}

int fsr_qcd_G2GG2::radBefID(int, int){ return 21;}
pair<int,int> fsr_qcd_G2GG2::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  int colRemove = (colRadAfter == acolEmtAfter)
                ? colRadAfter : acolRadAfter;
  int col       = (colRadAfter == colRemove)
                ? colEmtAfter : colRadAfter;
  int acol      = (acolRadAfter == colRemove)
                ? acolEmtAfter : acolRadAfter;
  return make_pair(col,acol);
}

vector <int> fsr_qcd_G2GG2::recPositions( const Event& state, int iRad, int iEmt) {

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();
  int colShared = (colRad  > 0 && colRad == acolEmt) ? colRad
                : (acolRad > 0 && colEmt == acolRad) ? colEmt : 0;
  // Particles to exclude from colour tracing.
  vector<int> iExc(1,iRad); iExc.push_back(iEmt);

  // Find partons connected via emitted colour line.
  vector<int> recs;
  // Find partons connected via radiator colour line.
  if ( colRad != 0 && colRad != colShared) {
    int acolF = findCol(colRad, iExc, state, 1);
    int  colI = findCol(colRad, iExc, state, 2);
    if (acolF  > 0 && colI == 0) recs.push_back (acolF);
    if (acolF == 0 && colI >  0) recs.push_back (colI);
  }

  // Find partons connected via radiator anticolour line.
  if ( acolRad != 0 && acolRad != colShared) {
    int  colF = findCol(acolRad, iExc, state, 2);
    int acolI = findCol(acolRad, iExc, state, 1);
    if ( colF  > 0 && acolI == 0) recs.push_back (colF);
    if ( colF == 0 && acolI >  0) recs.push_back (acolI);
  }

  // Done.
  return recs;
}

// Pick z for new splitting.
double fsr_qcd_G2GG2::zSplit(double zMinAbs, double, double m2dip) {
  // Just pick according to soft.
  double R         = rndmPtr->flat();
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p         = pow( 1. + pow2(1-zMinAbs)/kappaMin2, R );
  double res       = 1. - sqrt( p - 1. )*sqrt(kappaMin2);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qcd_G2GG2::overestimateInt(double zMinAbs, double,
  double, double m2dip, int orderNow) {
  // Overestimate by soft
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * softRescaleInt(order)
                     *0.5 * log( 1. + pow2(1.-zMinAbs)/kappaMin2);

  return wt;
}

// Return overestimate for new splitting.
double fsr_qcd_G2GG2::overestimateDiff(double z, double m2dip, int orderNow) {
  // Overestimate by soft
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaMin2 = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * softRescaleInt(order)
                     *(1.-z) / ( pow2(1.-z) + kappaMin2);
  return wt;
}

// Return kernel for new splitting.
bool fsr_qcd_G2GG2::calc(const Event& state, int orderNow) {

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

  double preFac = symmetryFactor() * gaugeFactor();
  int order     = (orderNow > 0) ? orderNow : correctionOrder;
  double kappa2 = pT2/m2dip;

  // Calculate kernel.
  // Note: We are calculating the z <--> 1-z symmetrised kernel here,
  // and later multiply with 1-z to project out one part.
  map<string,double> wts;
  double wt_base_as1 = preFac * (1.-z) / ( pow2(1.-z) + kappa2);
  //wts.insert( make_pair("base",
  //  preFac * softRescaleDiff( order, pT2, renormMultFac)
  //         * (1.-z) / ( pow2(1.-z) + kappa2) ));
  //if (doVariations) {
  //  // Create muR-variations.
  //  if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
  //    wts.insert( make_pair("Variations:muRfsrDown", preFac
  //      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRfsrDown")) 
  //      *(1.-z) / ( pow2(1.-z) + kappa2) ));
  //  if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
  //    wts.insert( make_pair("Variations:muRfsrUp", preFac
  //      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRfsrUp")) 
  //      *(1.-z) / ( pow2(1.-z) + kappa2) ));
  //}
  wts.insert( make_pair("base", wt_base_as1
    * softRescaleDiff( order, pT2, renormMultFac) ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt_base_as1
      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRfsrDown")) ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt_base_as1
      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRfsrUp")) ));
  }

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  // Add collinear term for massless splittings.
  if (!doMassive) {
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
      it->second += preFac * ( -1. + 0.5 *z*(1.-z) );    
    wt_base_as1 += preFac * ( -1. + 0.5 *z*(1.-z) );
  }

  // Add collinear term for massive splittings.
  if (doMassive) {

    double vijk = 1.;

    // splitType == 2 -> Massive FF
    if (splitType == 2) {
      // Calculate CS variables.
      double yCS = kappa2 / (1.-z);
      double nu2Rad = m2Rad/m2dip; 
      double nu2Emt = m2Emt/m2dip; 
      double nu2Rec = m2Rec/m2dip; 
      vijk          = pow2(1.-yCS) - 4.*(yCS+nu2Rad+nu2Emt)*nu2Rec;
      vijk          = sqrt(vijk) / (1-yCS);

    // splitType ==-2 -> Massive FI
    } else if (splitType ==-2) {
      // No changes, as initial recoiler is massless!
      vijk          = 1.; 
    }

    // Add correction for massive splittings.
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second += preFac * 1./ vijk * ( -1. + 0.5 * z*(1.-z) );    

    wt_base_as1 += preFac * 1./ vijk * ( -1. + 0.5 * z*(1.-z) );

  }

  // Add NLO term.
  if (!doMassive && order >= 3) {
    for ( map<string,double>::iterator it = wts.begin(); it !=wts.end(); ++it){
      double mukf = 1.;
      if (it->first == "base")
        mukf = renormMultFac;
      else if (it->first == "Variations:muRfsrDown")
        mukf = settingsPtr->parm("Variations:muRfsrDown");
      else if (it->first == "Variations:muRfsrUp")
        mukf = settingsPtr->parm("Variations:muRfsrUp");
      else continue;

      double NF          = getNF(pT2 * mukf);
      double alphasPT2pi = as2Pi(pT2, order, mukf);
      double TF          = TR*NF;
      // Evaluate everything at x = 1-z, because this is the kernel where
      // the radiating gluon becomes soft and the emission is identified.
      double x = 1.-z;
      // Anatomy of factors of two:
      // One factor of 0.5 since LO kernel is split into "z" and "1-z" parts, 
      // while the NLO kernel is NOT split into these structures.
      // Another factor of 0.5 enters because the LO kernel above does not
      // include a "2" (this "2" is in preFac), while the NLO kernel in the
      // Mathematica file does include the factor of "2".
      double pgg1   = preFac * 0.5 * 0.5 / ( 18*x*(pow2(x)-1) ) * (
TF*(4*(-1 + x)*(-23 + x*(6 + x*(10 + x*(4 + 23*x)))) + 24*(1 + x)*(2 + (-1 + x)*x*(3 + x*(-3 + 2*x)))*log(x)) +
   (CF*TF*(-12*(1 + x)*(8 + x*(7 - x*(2 + x)*(-3 + 8*x)))*log(x) - 8*(1 + x)*(23 + x*(14 + 41*x))*pow2(-1 + x) +
        36*(-1 + x)*x*pow2(1 + x)*pow2(log(x))))/CA + 72*CA*(-1 + x)*DiLog(1/(1 + x))*pow2(1 + x + pow2(x)) +
   CA*(-6*(1 + x)*(-22 + x*(11 + x*(30 + x*(-19 + 22*x))))*log(x) +
      (1 - x)*(x*(1 + x)*(25 + 109*x) + 6*(2 + x*(1 + 2*x*(1 + x)))*pow2(M_PI)) - 72*(1 + x)*log(1 - x)*log(x)*pow2(1 + (-1 + x)*x) +
      36*(2 + x*(1 + (-4 + x)*(-1 + x)*x*(1 + x)))*pow2(log(x)) + 36*(-1 + x)*pow2(log(1 + x))*pow2(1 + x + pow2(x))) 
      ); 
      // replace 1/z term in NLO kernel with z/(z*z+kappa2) to restore sum rule.
      // Note: Colour factor is CA, not CF!
      pgg1 += preFac * 0.5 * 0.5 * 40./9. * TF * ( x /(x*x + kappa2) - 1./x);
      // Add NLO term.
      it->second  += alphasPT2pi*pgg1;
    }
  }

  // Multiply with 1-z to project out part where radiating gluon is soft.
  // (the emission is identified)
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    it->second *= (1-z);    

  wt_base_as1 *= (1-z);
  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2",
    wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_G2QQ1::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].id() == 21 );
}

int fsr_qcd_G2QQ1::kinMap()                 { return 1;}
int fsr_qcd_G2QQ1::motherID(int)            { return 1;} // Use 1 as dummy variable.
int fsr_qcd_G2QQ1::sisterID(int)            { return 1;} // Use 1 as dummy variable.
double fsr_qcd_G2QQ1::gaugeFactor ( int, int )        { return NF_qcd_fsr*TR;}
double fsr_qcd_G2QQ1::symmetryFactor ( int, int )     { return 0.5;}

int fsr_qcd_G2QQ1::radBefID(int, int){ return 21;}
pair<int,int> fsr_qcd_G2QQ1::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  int col  = (colRadAfter  > 0) ? colRadAfter  : colEmtAfter;
  int acol = (acolRadAfter > 0) ? acolRadAfter : acolEmtAfter;
  return make_pair(col,acol);
}

vector <int> fsr_qcd_G2QQ1::recPositions( const Event& state, int iRad, int iEmt) {

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();
  int colShared = (colRad  > 0 && colRad == acolEmt) ? colRad
                : (acolRad > 0 && colEmt == acolRad) ? colEmt : 0;
  // Particles to exclude from colour tracing.
  vector<int> iExc(1,iRad); iExc.push_back(iEmt);

  // Find partons connected via emitted colour line.
  vector<int> recs;

  // Find partons connected via emitted colour line.
  if ( colEmt != 0 && colEmt != colShared) {
    int acolF = findCol(colEmt, iExc, state, 1);
    int  colI = findCol(colEmt, iExc, state, 2);
    if (acolF  > 0 && colI == 0) recs.push_back (acolF);
    if (acolF == 0 && colI >  0) recs.push_back (colI);
  }
  // Find partons connected via emitted anticolour line.
  if ( acolEmt != 0 && acolEmt != colShared) {
    int  colF = findCol(acolEmt, iExc, state, 2);
    int acolI = findCol(acolEmt, iExc, state, 1);
    if ( colF  > 0 && acolI == 0) recs.push_back (colF);
    if ( colF == 0 && acolI >  0) recs.push_back (acolI);
  }

  // Done.
  return recs;
}

// Pick z for new splitting.
double fsr_qcd_G2QQ1::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double fsr_qcd_G2QQ1::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt            = 2.*preFac * 0.5 * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double fsr_qcd_G2QQ1::overestimateDiff(double, double, int) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt            = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool fsr_qcd_G2QQ1::calc(const Event& state, int orderNow) {

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

  double preFac = symmetryFactor() * gaugeFactor();
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  double kappa2 = pT2/m2dip;

  map<string,double> wts;
  double wt_base_as1 = preFac * ( pow(1.-z,2.) + pow(z,2.) );
  //wts.insert( make_pair("base", preFac
  //  * (pow(1.-z,2.) + pow(z,2.)) ));
  //if (doVariations) {
  //  // Create muR-variations.
  //  if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
  //    wts.insert( make_pair("Variations:muRfsrDown", preFac
  //      * (pow(1.-z,2.) + pow(z,2.)) ));
  //  if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
  //    wts.insert( make_pair("Variations:muRfsrUp", preFac
  //      * (pow(1.-z,2.) + pow(z,2.)) ));
  //}

  wts.insert( make_pair("base", wt_base_as1 ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt_base_as1 ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt_base_as1 ));
  }

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
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second =  preFac * 1. / vijk * ( pow2(1.-z) + pow2(z)
                                         + m2Emt / ( pipj + m2Emt) );

    wt_base_as1 =  preFac * 1. / vijk * ( pow2(1.-z) + pow2(z)
                                        + m2Emt / ( pipj + m2Emt) );
  }

  // Add NLO term.
  if (!doMassive && order >= 3) {
    for ( map<string,double>::iterator it = wts.begin(); it !=wts.end(); ++it){
      double mukf = 1.;
      if (it->first == "base")
        mukf = renormMultFac;
      else if (it->first == "Variations:muRfsrDown")
        mukf = settingsPtr->parm("Variations:muRfsrDown");
      else if (it->first == "Variations:muRfsrUp")
        mukf = settingsPtr->parm("Variations:muRfsrUp");
      else continue;

      double NF          = getNF(pT2 * mukf);
      double alphasPT2pi = as2Pi(pT2, order, mukf);
      double TF          = TR*NF;
      double pgq1 = preFac * (
(TF*(-8./3. - (8*(1 + 2*(-1 + z)*z)*(2 + 3*log(1 - z) + 3*log(z)))/9.) +
     CF*(-2 + 3*z - 4*log(1 - z) + (-7 + 8*z)*log(z) + (1 - 2*z)*pow2(log(z)) -
        (2*(1 + 2*(-1 + z)*z)*(15 - 24*DiLog(z) + 3*log(-1 + 1/z) - 24*log(1 - z)*log(z) + pow2(M_PI) + 3*pow2(log(-((-1 + z)*z)))))/3.) +
     (CA*(-152 - 40/z + 166*z + 36*log(1 - z) - 12*(1 + 19*z)*log(z) +
          (1 + 2*(-1 + z)*z)*(178 - 144*DiLog(z) + log(1 - z)*(30 - 72*log(z)) - 3*log(z)*(4 + 3*log(z)) + 3*pow2(M_PI) +
             18*pow2(log(1 - z))) + 9*(2 + 8*z)*pow2(log(z)) +
          3*(1 + 2*z*(1 + z))*(-12*DiLog(1/(1 + z)) + pow2(M_PI) + 3*pow2(log(z)) - 6*pow2(log(1 + z)))))/9.)/2. 
      );
      // replace 1/z term in NLO kernel with z/(z*z+kappa2) to restore sum rule.
      // Include additional factor of 0.5 as we have two g->qq kernels.
      pgq1 += - preFac * 0.5 * 40./9. * CA * ( z /(z*z + kappa2) - 1./z);
      // Add NLO term.
      it->second  += alphasPT2pi*pgq1;
    }
  }

  // Multiply with z to project out part where emitted quark is soft,
  // and antiquark is identified.
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    it->second *= z;    

  wt_base_as1 *= z;
  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2",
    wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_G2QQ2::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].id() == 21 );
}

int fsr_qcd_G2QQ2::kinMap()                 { return 1;}
int fsr_qcd_G2QQ2::motherID(int)            { return -1;} // Use -1 as dummy variable.
int fsr_qcd_G2QQ2::sisterID(int)            { return -1;} // Use -1 as dummy variable.
double fsr_qcd_G2QQ2::gaugeFactor ( int, int )        { return NF_qcd_fsr*TR;}
double fsr_qcd_G2QQ2::symmetryFactor ( int, int )     { return 0.5;}

int fsr_qcd_G2QQ2::radBefID(int, int){ return 21;}
pair<int,int> fsr_qcd_G2QQ2::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  int col  = (colRadAfter  > 0) ? colRadAfter  : colEmtAfter;
  int acol = (acolRadAfter > 0) ? acolRadAfter : acolEmtAfter;
  return make_pair(col,acol);
}

vector <int> fsr_qcd_G2QQ2::recPositions( const Event& state, int iRad, int iEmt) {

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();
  int colShared = (colRad  > 0 && colRad == acolEmt) ? colRad
                : (acolRad > 0 && colEmt == acolRad) ? colEmt : 0;
  // Particles to exclude from colour tracing.
  vector<int> iExc(1,iRad); iExc.push_back(iEmt);

  // Find partons connected via emitted colour line.
  vector<int> recs;

  // Find partons connected via radiator colour line.
  if ( colRad != 0 && colRad != colShared) {
    int acolF = findCol(colRad, iExc, state, 1);
    int  colI = findCol(colRad, iExc, state, 2);
    if (acolF  > 0 && colI == 0) recs.push_back (acolF);
    if (acolF == 0 && colI >  0) recs.push_back (colI);
  }
  // Find partons connected via radiator anticolour line.
  if ( acolRad != 0 && acolRad != colShared) {
    int  colF = findCol(acolRad, iExc, state, 2);
    int acolI = findCol(acolRad, iExc, state, 1);
    if ( colF  > 0 && acolI == 0) recs.push_back (colF);
    if ( colF == 0 && acolI >  0) recs.push_back (acolI);
  }

  // Done.
  return recs;
}

// Pick z for new splitting.
double fsr_qcd_G2QQ2::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double fsr_qcd_G2QQ2::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt            = 2.*preFac * 0.5 * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double fsr_qcd_G2QQ2::overestimateDiff(double, double, int) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt            = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool fsr_qcd_G2QQ2::calc(const Event& state, int orderNow) {

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

  double preFac = symmetryFactor() * gaugeFactor();
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  double kappa2 = pT2/m2dip;

  map<string,double> wts;
  double wt_base_as1 = preFac * ( pow(1.-z,2.) + pow(z,2.) );
  //wts.insert( make_pair("base", preFac
  //  * (pow(1.-z,2.) + pow(z,2.)) ));
  //if (doVariations) {
  //  // Create muR-variations.
  //  if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
  //    wts.insert( make_pair("Variations:muRfsrDown", preFac
  //      * (pow(1.-z,2.) + pow(z,2.)) ));
  //  if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
  //    wts.insert( make_pair("Variations:muRfsrUp", preFac
  //      * (pow(1.-z,2.) + pow(z,2.)) ));
  //}

  wts.insert( make_pair("base", wt_base_as1  ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt_base_as1 ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt_base_as1 ));
  }
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
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second =  preFac * 1. / vijk * ( pow2(1.-z) + pow2(z)
                                         + m2Emt / ( pipj + m2Emt) );  

    wt_base_as1 =  preFac * 1. / vijk * ( pow2(1.-z) + pow2(z)
                                        + m2Emt / ( pipj + m2Emt) );
  }

  // Add NLO term.
  if (!doMassive && order >= 3) {
    for ( map<string,double>::iterator it = wts.begin(); it !=wts.end(); ++it){
      double mukf = 1.;
      if (it->first == "base")
        mukf = renormMultFac;
      else if (it->first == "Variations:muRfsrDown")
        mukf = settingsPtr->parm("Variations:muRfsrDown");
      else if (it->first == "Variations:muRfsrUp")
        mukf = settingsPtr->parm("Variations:muRfsrUp");
      else continue;

      double NF          = getNF(pT2 * mukf);
      double alphasPT2pi = as2Pi(pT2, order, mukf);
      double TF          = TR*NF;
      double x = 1-z;
      double pgq1 = preFac * (
(TF*(-8./3. - (8*(1 + 2*(-1 + x)*x)*(2 + 3*log(1 - x) + 3*log(x)))/9.) +
     CF*(-2 + 3*x - 4*log(1 - x) + (-7 + 8*x)*log(x) + (1 - 2*x)*pow2(log(x)) -
        (2*(1 + 2*(-1 + x)*x)*(15 - 24*DiLog(x) + 3*log(-1 + 1/x) - 24*log(1 - x)*log(x) + pow2(M_PI) + 3*pow2(log(-((-1 + x)*x)))))/3.) +
     (CA*(-152 - 40/x + 166*x + 36*log(1 - x) - 12*(1 + 19*x)*log(x) +
          (1 + 2*(-1 + x)*x)*(178 - 144*DiLog(x) + log(1 - x)*(30 - 72*log(x)) - 3*log(x)*(4 + 3*log(x)) + 3*pow2(M_PI) +
             18*pow2(log(1 - x))) + 9*(2 + 8*x)*pow2(log(x)) +
          3*(1 + 2*x*(1 + x))*(-12*DiLog(1/(1 + x)) + pow2(M_PI) + 3*pow2(log(x)) - 6*pow2(log(1 + x)))))/9.)/2. 
      );
      // replace 1/z term in NLO kernel with z/(z*z+kappa2) to restore sum rule.
      // Include additional factor of 0.5 as we have two g->qq kernels.
      pgq1 += - preFac * 0.5 * 40./9. * CA * ( x /(x*x + kappa2) - 1./x);
      // Add NLO term.
      it->second += alphasPT2pi*pgq1;
    }
  }

  // Multiply with z to project out part where emitted antiquark is soft,
  // and quark is identified.
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    it->second *= (1-z);    

  wt_base_as1 *= (1-z);
  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2",
    wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function Q-> q Q qbar (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_Q2qQqbarDist::canRadiate ( const Event& state,
  map<string,int> ints, map<string,bool>, Settings*, PartonSystems*,
  BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].isQuark() );
}

int fsr_qcd_Q2qQqbarDist::kinMap()                 { return 2;}
int fsr_qcd_Q2qQqbarDist::motherID(int idDaughter) { return idDaughter;}
int fsr_qcd_Q2qQqbarDist::sisterID(int)            { return 1;}
double fsr_qcd_Q2qQqbarDist::gaugeFactor ( int, int )        { return CF;}
double fsr_qcd_Q2qQqbarDist::symmetryFactor ( int, int )     { return 1.;}

int fsr_qcd_Q2qQqbarDist::radBefID(int idRA, int) {
  if (particleDataPtr->isQuark(idRA)) return idRA;
  return 0;
  //return idRA;
}
pair<int,int> fsr_qcd_Q2qQqbarDist::radBefCols(
  int colRadAfter, int, 
  int colEmtAfter, int acolEmtAfter) {
  bool isQuark = (colRadAfter > 0);
  if (isQuark) return make_pair(colEmtAfter,0);
  return make_pair(0,acolEmtAfter);
}

// Pick z for new splitting.
double fsr_qcd_Q2qQqbarDist::zSplit(double zMinAbs, double zMaxAbs,
  double m2dip) {

  double Rz         = rndmPtr->flat();
  double kappa4  = pow(settingsPtr->parm("TimeShower:pTmin"), 4) / pow2(m2dip);
  double res     = 1.;
  // z est from 1/(z + kappa^4)
  res = pow( (kappa4 + zMaxAbs)/(kappa4 + zMinAbs), -Rz )
      * (kappa4 + zMaxAbs - kappa4
                           *pow((kappa4 + zMaxAbs)/(kappa4 + zMinAbs), Rz));

  return res;

}

// New overestimates, z-integrated versions.
double fsr_qcd_Q2qQqbarDist::overestimateInt(double zMinAbs, double zMaxAbs,
  double, double m2dip, int orderNow) {

  // Do nothing without other NLO kernels!
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  if (order < 3) return 0.0;

  double preFac  = symmetryFactor() * gaugeFactor();
  double pT2min  = pow2(settingsPtr->parm("TimeShower:pTmin"));
  double kappa4  = pow2(pT2min/m2dip);
  // Overestimate chosen to have accept weights below one for kappa~0.1
  // z est from 1/(z + kappa^4)
  double wt = preFac * TR * 2. * ( NF_qcd_fsr - 1. ) * 20./9.
            * log( ( kappa4 + zMaxAbs) / ( kappa4 + zMinAbs) );

  // This splitting is down by one power of alphaS !
  wt *= as2Pi(pT2min);

  return wt;

}

// Return overestimate for new splitting.
double fsr_qcd_Q2qQqbarDist::overestimateDiff(double z, double m2dip,
  int orderNow) {

  // Do nothing without other NLO kernels!
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  if (order < 3) return 0.0;

  double preFac    = symmetryFactor() * gaugeFactor();
  double pT2min    = pow2(settingsPtr->parm("TimeShower:pTmin"));
  double kappa4    = pow2(pT2min/m2dip);
  // Overestimate chosen to have accept weights below one for kappa~0.1
  double wt = preFac * TR * 2. * ( NF_qcd_fsr - 1. ) * 20./ 9. * 1 / (z + kappa4);

  // This splitting is down by one power of alphaS !
  wt *= as2Pi(pT2min);

  return wt;

}

// Return kernel for new splitting.
bool fsr_qcd_Q2qQqbarDist::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z),
    pT2(splitInfo.kinematics()->pT2),
    //m2dip(splitInfo.kinematics()->m2Dip),
    xa(splitInfo.kinematics()->xa),
    sai(splitInfo.kinematics()->sai),
    m2aij(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    //m2Rec(splitInfo.kinematics()->m2Rec),
    m2a(splitInfo.kinematics()->m2RadAft),
    m2i(splitInfo.kinematics()->m2EmtAft),
    m2j(splitInfo.kinematics()->m2EmtAft2),
    m2k(splitInfo.kinematics()->m2Rec);

  //int splitType(splitInfo.type);

  // Do nothing without other NLO kernels!
  map<string,double> wts;
  //int order          = (orderNow > 0) ? orderNow : correctionOrder;
  int order          = (orderNow > -1) ? orderNow : correctionOrder;
  if (order < 3 || m2aij > 0. || m2a > 0. || m2i > 0. || m2j > 0. || m2k > 0.){
    wts.insert( make_pair("base", 0.) );
    if (doVariations && settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", 0.));
    if (doVariations && settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", 0.));
    clearKernels();
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
      kernelVals.insert(make_pair( it->first, it->second ));
    return true;
  }

  // Choose if simulating endpoint or differential 1->3 (latter containing
  // both sai=0 and sai !=0). For choice of endpoint, set sai=0 later.
  bool isEndpoint = (rndmPtr->flat() < 0.5);

  Event trialEvent(state);
  bool physical = false;
  if (splitInfo.recBef()->isFinal)
    physical = fsr->branch_FF(trialEvent, true, &splitInfo);
  else
    physical = fsr->branch_FI(trialEvent, true, &splitInfo);

  // Get invariants.
  Vec4 pa(trialEvent[splitInfo.iRadAft].p());
  Vec4 pk(trialEvent[splitInfo.iRecAft].p());
  Vec4 pi(trialEvent[splitInfo.iEmtAft].p());
  Vec4 pj(trialEvent[splitInfo.iEmtAft2].p());

  // Use only massless for now!
  if ( abs(pa.m2Calc()-m2a) > sai || abs(pi.m2Calc()-m2i) > sai
    || abs(pj.m2Calc()-m2j) > sai || abs(pk.m2Calc()-m2k) > sai)
    physical = false;

  if (!physical) {
    wts.insert( make_pair("base", 0.) );
    if (doVariations && settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", 0.));
    if (doVariations && settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", 0.));
    clearKernels();
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
      kernelVals.insert(make_pair( it->first, it->second ));
    return true;
  }

  double sign = (splitInfo.recBef()->isFinal) ? 1. : -1.;
  double p2ai(sai + m2a + m2i),
         p2aj((pa+pj).m2Calc()),
         p2ak(sign*(pa+sign*pk).m2Calc()),
         p2ij((pi+pj).m2Calc()),
         p2ik(sign*(pi+sign*pk).m2Calc()),
         p2jk(sign*(pj+sign*pk).m2Calc());
  double q2   = sign*(pa+pi+pj+sign*pk).m2Calc();
  double saij = (pa+pi+pj).m2Calc();
  double yaij = (splitInfo.recBef()->isFinal) ? saij / q2 : 0.;

  double prob = 0.0;
  double z1(z/(1.-yaij)), z2( z/xa/(1-yaij) - z1 ), z3(1-z1-z2);

  if (isEndpoint) {

    prob = CF*TR*((1.0+z3*z3)/(1.0-z3)
                 +(1.0-2.0*z1*z2/pow2(z1+z2))*(1.0-z3+(1.0+z3*z3)/(1.0-z3)
                 *(log(z1*z2*z3)-1.0)));
    prob-= CF*TR*2.0*((1.0+z3*z3)/(1.0-z3)*log(z3*(1.0-z3)) +1.0-z3)
                     *(1.0-2.0*z1*z2/pow2(z1+z2));

    // there are 2nf-2 such kernels.
    prob *= 2. * ( NF_qcd_fsr - 1. );
    // From xa integration volume.
    prob *= log(1/z1);
    // Multiply by 2 since we randomly chose endpoint or fully differential.
    prob *= 2.0;
    // Weight of sai-selection. Note: Use non-zero sai here!
    prob *= 1. / (1.-p2ai/saij);

  } else {

    double s12(p2ai), s13(p2aj), s23(p2ij), s123(saij);
    double t123 = 2.*(z1*s23 - z2*s13)/(z1+z2) + (z1-z2)/(z1+z2)*s12;
    double CG = 0.5*CF*TR*s123/s12
               *( - pow2(t123)/ (s12*s123)
                  + (4.*z3 + pow2(z1-z2))/(z1+z2) + z1 + z2 - s12/s123 );
    double cosPhiKT1KT3 = pow2(p2ij*p2ak - p2aj*p2ik + p2ai*p2jk)
                        / (4.*p2ai*p2ij*p2ak*p2jk);
    double subt = CF*TR*s123/s12
                * ( (1.+z3*z3) / (1.-z3) * (1.-2.*z1*z2/pow2(1-z3))
                   + 4.*z1*z2*z3 / pow(1.-z3,3) * (1-2.*cosPhiKT1KT3) );
    prob = CG - subt;

    if ( abs(s12) < 1e-10) prob = 0.0;

    // there are 2nf-2 such kernels.
    prob *= 2. * ( NF_qcd_fsr - 1. );
    // From xa integration volume.
    prob *= log(1/z1);
    // Multiply by 2 since we randomly chose endpoint or fully differential.
    prob *= 2.0;
    // Weight of sai-selection.
    prob *= 1. / (1.-p2ai/saij);

  }

  // Remember that this might be an endpoint with vanishing sai.
  if (isEndpoint) { splitInfo.set_sai(0.0); }

  // Insert value of kernel into kernel list.
  wts.insert( make_pair("base", prob * as2Pi(pT2, order, renormMultFac) ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", prob
        * as2Pi(pT2, order, settingsPtr->parm("Variations:muRfsrDown")) ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp",   prob
        * as2Pi(pT2, order, settingsPtr->parm("Variations:muRfsrUp")) ));
  }

  // Multiply with z to project out part where emitted antiquark is soft,
  // and quark is identified.
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    it->second *= z;    

  // Store higher order correction separately.
  wts.insert( make_pair("base_order_as2", wts["base"] ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;
}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function Q-> Qbar Q Q (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_Q2QbarQQId::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].isQuark() );
}

int fsr_qcd_Q2QbarQQId::kinMap()                 { return 2;}
int fsr_qcd_Q2QbarQQId::motherID(int idDaughter) { return idDaughter;}
int fsr_qcd_Q2QbarQQId::sisterID(int)            { return 1;}
double fsr_qcd_Q2QbarQQId::gaugeFactor ( int, int )        { return CF;}
double fsr_qcd_Q2QbarQQId::symmetryFactor ( int, int )     { return 1.;}

int fsr_qcd_Q2QbarQQId::radBefID(int idRA, int) { 
  if (particleDataPtr->isQuark(idRA)) return idRA;
  return 0;
  //return idRA;
}
pair<int,int> fsr_qcd_Q2QbarQQId::radBefCols(
  int colRadAfter, int, 
  int colEmtAfter, int acolEmtAfter) {
  bool isQuark = (colRadAfter > 0);
  if (isQuark) return make_pair(colEmtAfter,0);
  return make_pair(0,acolEmtAfter);
}

// Pick z for new splitting.
double fsr_qcd_Q2QbarQQId::zSplit(double zMinAbs, double zMaxAbs,
  double m2dip) {

  // z est from 1/4 z/(z^2 + kappa^2)
  double Rz         = rndmPtr->flat();
  double kappaMin2  = pow2(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p          = (kappaMin2 + zMaxAbs*zMaxAbs)
                    / (kappaMin2 + zMinAbs*zMinAbs);
  double res        = sqrt( (kappaMin2 + zMaxAbs*zMaxAbs - kappaMin2*pow(p,Rz))
                             /pow(p,Rz) );
  return res;

}

// New overestimates, z-integrated versions.
double fsr_qcd_Q2QbarQQId::overestimateInt(double zMinAbs, double zMaxAbs,
  double, double m2dip, int orderNow) {

  // Do nothing without other NLO kernels!
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  if (order < 3) return 0.0;

  // z est from 1/4 z/(z^2 + kappa^2)
  double preFac     = symmetryFactor() * gaugeFactor();
  double pT2min     = pow2(settingsPtr->parm("TimeShower:pTmin"));
  double kappaMin2  = pT2min/m2dip;
  double wt         = preFac * TR * 20./9. 
                      * 0.5 * log( ( kappaMin2 + zMaxAbs*zMaxAbs)
                                 / ( kappaMin2 + zMinAbs*zMinAbs) );
  // This splitting is down by one power of alphaS !
  wt *= as2Pi(pT2min);
  return wt;

}

// Return overestimate for new splitting.
double fsr_qcd_Q2QbarQQId::overestimateDiff(double z, double m2dip,
  int orderNow) {

  // Do nothing without other NLO kernels!
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  if (order < 3) return 0.0;

  double preFac     = symmetryFactor() * gaugeFactor();
  double pT2min     = pow2(settingsPtr->parm("TimeShower:pTmin"));
  double kappaMin2  = pT2min/m2dip;
  double wt         = preFac * TR * 20./ 9. * z / (z*z + kappaMin2);
  // This splitting is down by one power of alphaS !
  wt *= as2Pi(pT2min);
  return wt;

}

// Return kernel for new splitting.
bool fsr_qcd_Q2QbarQQId::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z),
    pT2(splitInfo.kinematics()->pT2),
    //m2dip(splitInfo.kinematics()->m2Dip),
    xa(splitInfo.kinematics()->xa),
    sai(splitInfo.kinematics()->sai),
    m2aij(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    //m2Rec(splitInfo.kinematics()->m2Rec),
    m2a(splitInfo.kinematics()->m2RadAft),
    m2i(splitInfo.kinematics()->m2EmtAft),
    m2j(splitInfo.kinematics()->m2EmtAft2),
    m2k(splitInfo.kinematics()->m2Rec);

  map<string,double> wts;
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  // Do nothing without other NLO kernels!
  //if (order < 3) { 
  if (order < 3 || m2aij > 0. || m2a > 0. || m2i > 0. || m2j > 0. || m2k > 0.){
    wts.insert( make_pair("base", 0.) );
    if (doVariations && settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", 0.));
    if (doVariations && settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", 0.));
    clearKernels();
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
      kernelVals.insert(make_pair( it->first, it->second ));
    return true;
  }

  // Choose if simulating endpoint or differential 1->3 (latter containing
  // both sai=0 and sai !=0). For choice of endpoint, set sai=0 later.
  bool isEndpoint = (rndmPtr->flat() < 0.5);

  Event trialEvent(state);
  bool physical = false;
  if (splitInfo.recBef()->isFinal)
    physical = fsr->branch_FF(trialEvent, true, &splitInfo);
  else
    physical = fsr->branch_FI(trialEvent, true, &splitInfo);

  // Get invariants.
  Vec4 pa(trialEvent[splitInfo.iRadAft].p());
  Vec4 pk(trialEvent[splitInfo.iRecAft].p());
  Vec4 pi(trialEvent[splitInfo.iEmtAft].p());
  Vec4 pj(trialEvent[splitInfo.iEmtAft2].p());

  // Use only massless for now!
  if ( abs(pa.m2Calc()-m2a) > sai || abs(pi.m2Calc()-m2i) > sai
    || abs(pj.m2Calc()-m2j) > sai || abs(pk.m2Calc()-m2k) > sai)
    physical = false;

  if (!physical) {
    wts.insert( make_pair("base", 0.) );
    if (doVariations && settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", 0.));
    if (doVariations && settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", 0.));
    clearKernels();
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
      kernelVals.insert(make_pair( it->first, it->second ));
    return true;
  }

  double sign = (splitInfo.recBef()->isFinal) ? 1. : -1.;
  double p2ai(sai + m2a + m2i),
         p2aj((pa+pj).m2Calc()),
         p2ak(sign*(pa+sign*pk).m2Calc()),
         p2ij((pi+pj).m2Calc()),
         p2ik(sign*(pi+sign*pk).m2Calc()),
         p2jk(sign*(pj+sign*pk).m2Calc());
  double q2   = sign*(pa+pi+pj+sign*pk).m2Calc();
  double saij = (pa+pi+pj).m2Calc();
  double yaij = (splitInfo.recBef()->isFinal) ? saij / q2 : 0.;

  double prob = 0.0;
  double z1(z/(1.-yaij)), z2( z/xa/(1-yaij) - z1 ), z3(1-z1-z2);

  if (isEndpoint) {

    prob = CF*TR*((1.0+z3*z3)/(1.0-z3)
                 +(1.0-2.0*z1*z2/pow2(z1+z2))*(1.0-z3+(1.0+z3*z3)/(1.0-z3)
                 *(log(z1*z2*z3)-1.0)));
    // Swapped contribution.
    prob+= CF*TR*((1.0+z2*z2)/(1.0-z2)
                 +(1.0-2.0*z1*z3/pow2(z1+z3))*(1.0-z2+(1.0+z2*z2)/(1.0-z2)
                 *(log(z1*z3*z2)-1.0)));
    // Subtraction.
    prob-= CF*TR*2.0*((1.0+z3*z3)/(1.0-z3)*log(z3*(1.0-z3)) +1.0-z3)
                     *(1.0-2.0*z1*z2/pow2(z1+z2));
    // Swapped subtraction.
    prob-= CF*TR*2.0*((1.0+z2*z2)/(1.0-z2)*log(z2*(1.0-z2)) +1.0-z2)
                     *(1.0-2.0*z1*z3/pow2(z1+z3));

    // From xa integration volume.
    prob *= log(1/z1);
    // Multiply by 2 since we randomly chose endpoint or fully differential.
    prob *= 2.0;
    // Weight of sai-selection. Note: Use non-zero sai here!
    prob *= 1. / (1.-p2ai/saij);

  } else {

    double s12(p2ai), s13(p2aj), s23(p2ij), s123(saij);
    double t123 = 2.*(z1*s23 - z2*s13)/(z1+z2) + (z1-z2)/(z1+z2)*s12;
    double CG = 0.5*CF*TR*s123/s12
               *( - pow2(t123)/ (s12*s123)
                  + (4.*z3 + pow2(z1-z2))/(z1+z2) + z1 + z2 - s12/s123 );
    // Swapped kernel.
    double t132 = 2.*(z1*s23 - z3*s12)/(z1+z3) + (z1-z3)/(z1+z3)*s13;
    CG       += 0.5*CF*TR*s123/s13
               *( - pow2(t132)/ (s13*s123)
                  + (4.*z2 + pow2(z1-z3))/(z1+z3) + z1 + z3 - s13/s123 );
    // Interference term.
    CG       += CF*(CF-0.5*CA)
              * ( 2.*s23/s12
                + s123/s12 * ( (1.+z1*z1)/(1-z2) - 2.*z2/(1.-z3) )
                - s123*s123/(s12*s13) * 0.5*z1*(1.+z1*z1) / ((1.-z2)*(1.-z3)) );
    // Swapped interference term.
    CG       += CF*(CF-0.5*CA)
              * ( 2.*s23/s13
                + s123/s13 * ( (1.+z1*z1)/(1-z3) - 2.*z3/(1.-z2) )
                - s123*s123/(s13*s12) * 0.5*z1*(1.+z1*z1) / ((1.-z3)*(1.-z2)) );
    // Subtraction.
    double cosPhiKT1KT3 = pow2(p2ij*p2ak - p2aj*p2ik + p2ai*p2jk)
                        / (4.*p2ai*p2ij*p2ak*p2jk);
    double subt = CF*TR*s123/s12
                * ( (1.+z3*z3) / (1.-z3) * (1.-2.*z1*z2/pow2(1-z3))
                   + 4.*z1*z2*z3 / pow(1.-z3,3) * (1-2.*cosPhiKT1KT3) );
    // Swapped subtraction.
    double cosPhiKT1KT2 = pow2(p2ij*p2ak + p2aj*p2ik - p2ai*p2jk)
                        / (4.*p2aj*p2ij*p2ak*p2ik);
    subt       += CF*TR*s123/s13
                * ( (1.+z2*z2) / (1.-z2) * (1.-2.*z1*z3/pow2(1-z2))
                   + 4.*z1*z3*z2 / pow(1.-z2,3) * (1-2.*cosPhiKT1KT2) );
    prob = CG - subt;

    if ( abs(s12) < 1e-10) prob = 0.0;

    // From xa integration volume.
    prob *= log(1/z1);
    // Multiply by 2 since we randomly chose endpoint or fully differential.
    prob *= 2.0;
    // Weight of sai-selection.
    prob *= 1. / (1.-p2ai/saij);

  }

  // Desymmetrize in i and j.
  prob *= (z/xa - z) / ( 1- z);

  // Remember that this might be an endpoint with vanishing sai.
  if (isEndpoint) { splitInfo.set_sai(0.0); }

  wts.insert( make_pair("base", prob*as2Pi(pT2, order, renormMultFac) ));

  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", prob
        * as2Pi(pT2, order, settingsPtr->parm("Variations:muRfsrDown")) ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp",   prob
        * as2Pi(pT2, order, settingsPtr->parm("Variations:muRfsrUp")) ));
  }

  // Multiply with z to project out part where emitted antiquark is soft,
  // and quark is identified.
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    it->second *= z;    

  // Store higher order correction separately.
  wts.insert( make_pair("base_order_as2", wts["base"] ));

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
bool isr_qcd_Q2QG::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].isQuark() );
}

int isr_qcd_Q2QG::kinMap()                 { return 1;}
int isr_qcd_Q2QG::motherID(int idDaughter) { return idDaughter;} 
int isr_qcd_Q2QG::sisterID(int)            { return 21;} 
double isr_qcd_Q2QG::gaugeFactor ( int, int )        { return CF;}
double isr_qcd_Q2QG::symmetryFactor ( int, int )     { return 1.;}

int isr_qcd_Q2QG::radBefID(int idRA, int) { 
  if (particleDataPtr->isQuark(idRA)) return idRA;
  return 0;
  //return idRA;
}
pair<int,int> isr_qcd_Q2QG::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  bool isQuark  = (colRadAfter > 0);
  int colRemove = (colRadAfter == colEmtAfter)
                ? colRadAfter : 0;
  int col       = (colRadAfter  == colRemove)
                ? acolEmtAfter : colRadAfter;
  if (isQuark) return make_pair(col,0); 
  colRemove = (acolRadAfter == acolEmtAfter)
                ? acolRadAfter : 0;
  int acol      = (acolRadAfter  == colRemove)
                ? colEmtAfter : acolRadAfter;
  return make_pair(0,acol); 
}

vector <int> isr_qcd_Q2QG::recPositions( const Event& state, int iRad, int iEmt) {

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();
  int colShared = (colRad  > 0 && colRad == colEmt) ? colEmt
                : (acolRad > 0 && acolEmt == acolRad) ? acolEmt : 0;
  // Particles to exclude from colour tracing.
  vector<int> iExc(1,iRad); iExc.push_back(iEmt);

  // Find partons connected via emitted colour line.
  vector<int> recs;
  if ( colEmt != 0 && colEmt != colShared) {
    int acolF = findCol(colEmt, iExc, state, 1);
    int  colI = findCol(colEmt, iExc, state, 2);
    if (acolF  > 0 && colI == 0) recs.push_back (acolF);
    if (acolF == 0 && colI >  0) recs.push_back (colI);
  }
  // Find partons connected via emitted anticolour line.
  if ( acolEmt != 0 && acolEmt != colShared) {
    int  colF = findCol(acolEmt, iExc, state, 2);
    int acolI = findCol(acolEmt, iExc, state, 1);
    if ( colF  > 0 && acolI == 0) recs.push_back (colF);
    if ( colF == 0 && acolI >  0) recs.push_back (acolI);
  }
  // Done.
  return recs;
} 

// Pick z for new splitting.
double isr_qcd_Q2QG::zSplit(double zMinAbs, double, double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  double p = pow( 1. + pow2(1-zMinAbs)/kappa2, Rz );
  double res = 1. - sqrt( p - 1. )*sqrt(kappa2);
  return res;
}

// New overestimates, z-integrated versions.
double isr_qcd_Q2QG::overestimateInt(double zMinAbs, double,
  double, double m2dip, int orderNow) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  wt  = preFac * softRescaleInt(order)
      * 2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappa2);
  return wt;
}

// Return overestimate for new splitting.
double isr_qcd_Q2QG::overestimateDiff(double z, double m2dip, int orderNow) {
  double wt        = 0.;
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaOld2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  wt  = preFac * softRescaleInt(order)
      * 2.* (1.-z) / ( pow2(1.-z) + kappaOld2);
  return wt;
}

// Return kernel for new splitting.
bool isr_qcd_Q2QG::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip);
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    //m2Rec(splitInfo.kinematics()->m2Rec),
    //m2Emt(splitInfo.kinematics()->m2EmtAft);
  //int splitType(splitInfo.type);

  double preFac = symmetryFactor() * gaugeFactor();
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  double kappa2 = pT2/m2dip;

  map<string,double> wts;
  double wt_base_as1 = preFac * ( 2.*(1.-z)/(pow2(1.-z) + kappa2) - (1.+z) );
  //wts.insert( make_pair("base",
  //  preFac * softRescaleDiff( order, pT2, renormMultFac)
  //         * ( 2.* (1.-z) / ( pow2(1.-z) + kappa2) ) - preFac * (1.+z ) ));
  //if (doVariations) {
  //  // Create muR-variations.
  //  if (settingsPtr->parm("Variations:muRisrDown") != 1.)
  //    wts.insert( make_pair("Variations:muRisrDown", preFac
  //      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRisrDown")) 
  //      *( 2.* (1.-z) / ( pow2(1.-z) + kappa2) ) - preFac * (1.+z ) ));
  //  if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
  //    wts.insert( make_pair("Variations:muRisrUp", preFac
  //      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRisrUp")) 
  //      *( 2.* (1.-z) / ( pow2(1.-z) + kappa2) ) - preFac * (1.+z ) ));
  //}

  wts.insert( make_pair("base", wt_base_as1
    * softRescaleDiff( order, pT2, renormMultFac) ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", wt_base_as1
      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRisrDown")) ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", wt_base_as1
      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRisrUp")) ));
  }

  // Add NLO term, subtracted by ~ 1/(1-z)*Gamma2,
  // since latter already present in soft rescaling term.
  if (order >= 3) {
    for ( map<string,double>::iterator it = wts.begin(); it !=wts.end(); ++it){

      double mukf = 1.;
      if (it->first == "base")
        mukf = renormMultFac;
      else if (it->first == "Variations:muRisrDown")
        mukf = settingsPtr->parm("Variations:muRisrDown");
      else if (it->first == "Variations:muRisrUp")
        mukf = settingsPtr->parm("Variations:muRisrUp");
      else continue;

      double NF          = getNF(pT2 * mukf);
      double alphasPT2pi = as2Pi(pT2, order, mukf);
      double TF          = TR*NF;
      double pqq1   = preFac * 1 / ( 18*z*(z-1) ) * (
      (-1 + z)*(-8*TF*(-5 + (-1 + z)*z*(-5 + 14*z)) 
                + z*(90*CF*(-1 + z) + CA*(53 - 187*z + 3*(1 + z)*pow2(M_PI))))
      +3*z*log(z)*(-2*(TF + CF*(-9 + 6*(-1 + z)*z) + TF*z*(12 - z*(9 + 8*z)))
                  + 12*CF*log(1 - z)*(1 + pow2(z)) - CA*(17 + 5*pow2(z)))
      -9*z*(CA - CF - 2*TF + (CA + CF + 2*TF)*pow2(z))*pow2(log(z)));
      // replace 1/z term in NLO kernel with z/(z^2+kappa^2)
      pqq1 += preFac * 20./9.*TF * ( z/(pow2(z)+kappa2) - 1./z); 
      // Add NLO term.
      it->second += alphasPT2pi*pqq1;
    }
  }

  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2",
    wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function G->GG (ISR)

// Return true if this kernel should partake in the evolution.
bool isr_qcd_G2GG1::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].id() == 21 );
}

int isr_qcd_G2GG1::kinMap()                 { return 1;}
int isr_qcd_G2GG1::motherID(int)            { return 21;} 
int isr_qcd_G2GG1::sisterID(int)            { return 21;} 
double isr_qcd_G2GG1::gaugeFactor ( int, int )        { return 2.*CA;}
double isr_qcd_G2GG1::symmetryFactor ( int, int )     { return 0.5;}

int isr_qcd_G2GG1::radBefID(int idRA, int){
  if (idRA == 21) return 21;
  return 0;
}
pair<int,int> isr_qcd_G2GG1::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  int colRemove = (colRadAfter == colEmtAfter)
                ? colRadAfter : acolRadAfter;
  int col       = (colRadAfter  == colRemove)
                ? acolEmtAfter : colRadAfter;
  int acol      = (acolRadAfter == colRemove)
                ? colEmtAfter : acolRadAfter;
  return make_pair(col,acol);
}

vector <int> isr_qcd_G2GG1::recPositions( const Event& state, int iRad, int iEmt) {

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();
  int colShared = (colRad  > 0 && colRad == colEmt) ? colEmt
                : (acolRad > 0 && acolEmt == acolRad) ? acolEmt : 0;
  // Particles to exclude from colour tracing.
  vector<int> iExc(1,iRad); iExc.push_back(iEmt);

  // Find partons connected via emitted colour line.
  vector<int> recs;
  if ( colEmt != 0 && colEmt != colShared) {
    int acolF = findCol(colEmt, iExc, state, 1);
    int  colI = findCol(colEmt, iExc, state, 2);
    if (acolF  > 0 && colI == 0) recs.push_back (acolF);
    if (acolF == 0 && colI >  0) recs.push_back (colI);
  }
  // Find partons connected via emitted anticolour line.
  if ( acolEmt != 0 && acolEmt != colShared) {
    int  colF = findCol(acolEmt, iExc, state, 2);
    int acolI = findCol(acolEmt, iExc, state, 1);
    if ( colF  > 0 && acolI == 0) recs.push_back (colF);
    if ( colF == 0 && acolI >  0) recs.push_back (acolI);
  }
  // Done.
  return recs;
}

// Pick z for new splitting.
double isr_qcd_G2GG1::zSplit(double zMinAbs, double, double m2dip) {
  double R = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  // Pick according to soft + 1/z
  double res = (-2.*pow(kappa2,R)*pow(zMinAbs,2.*R) +
             sqrt(4.*pow(kappa2,2.*R)
                   *pow(zMinAbs,4.*R)
                + 4.*(pow(kappa2,R) + pow(kappa2,1. + R))
                   *pow(zMinAbs,2.*R)
                   *(-(pow(kappa2,R)*pow(zMinAbs,2.*R))
                     + kappa2
                       *pow(1. + kappa2 - 2.*zMinAbs + pow(zMinAbs,2.),R))))
          / (2.*(-(pow(kappa2,R)*pow(zMinAbs,2.*R))
                 + kappa2
                   *pow(1. + kappa2 - 2.*zMinAbs + pow(zMinAbs,2.),R)));
  return res;
}

// New overestimates, z-integrated versions.
double isr_qcd_G2GG1::overestimateInt(double zMinAbs, double,
  double, double m2dip, int orderNow) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  // Overestimate by soft + 1/z
  wt   = preFac * softRescaleInt(order)
       *0.5*( log(1./pow2(zMinAbs) + pow2(1.-zMinAbs)/(kappa2*pow2(zMinAbs))));

  return wt;
}

// Return overestimate for new splitting.
double isr_qcd_G2GG1::overestimateDiff(double z, double m2dip, int orderNow) {
  double wt        = 0.;
  double preFac    = symmetryFactor() * gaugeFactor();
  //int order        = (orderNow > 0) ? orderNow : correctionOrder;
  int order        = (orderNow > -1) ? orderNow : correctionOrder;
  double kappaOld2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;
  // Overestimate by soft + 1/z
  wt  = preFac * softRescaleInt(order)
      * ((1.-z) / ( pow2(1.-z) + kappaOld2) + 1./z);
  return wt;
}

// Return kernel for new splitting.
bool isr_qcd_G2GG1::calc(const Event& state, int orderNow) {

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

  double preFac = symmetryFactor() * gaugeFactor();
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  double kappa2 = pT2/m2dip;

  map<string,double> wts;
  double wt_base_as1 = preFac * ( (1.-z) / (pow2(1.-z)+kappa2) );
  //wts.insert( make_pair("base", preFac * softRescaleDiff( order, pT2,
  //  renormMultFac) * (1.-z) / (pow2(1.-z)+kappa2) ));
  //if (doVariations) {
  //  // Create muR-variations.
  //  if (settingsPtr->parm("Variations:muRisrDown") != 1.)
  //    wts.insert( make_pair("Variations:muRisrDown", preFac
  //      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRisrDown")) 
  //      * (1.-z) / ( pow2(1.-z) + kappa2)  ));
  //  if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
  //    wts.insert( make_pair("Variations:muRisrUp", preFac
  //      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRisrUp")) 
  //      * (1.-z) / ( pow2(1.-z) + kappa2)  ));
  //}

  wts.insert( make_pair("base", wt_base_as1
    * softRescaleDiff( order, pT2, renormMultFac) ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", wt_base_as1
      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRisrDown")) ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", wt_base_as1
      *softRescaleDiff( order, pT2, settingsPtr->parm("Variations:muRisrUp")) ));
  }

  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    it->second += preFac * 0.5 * ( z / ( pow2(z) + kappa2) - 1. ) - preFac;

  wt_base_as1 += preFac * 0.5 * ( z / ( pow2(z) + kappa2) - 1. ) - preFac;

  // Correction for massive IF splittings.
  bool doMassive = ( m2Rec > 0. && splitType == 2);

  if (doMassive) {
    // Construct CS variables.
    double uCS = kappa2 / (1-z);
    double massCorr = - m2Rec / m2dip * uCS / (1.-uCS);
    // Mass correction shared in equal parts between both g->gg kernels.
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second += preFac * 0.5 * massCorr;

    wt_base_as1 += preFac * 0.5 * massCorr;

  }

  // Add NLO term, subtracted by 1/(1-z)*(Gamma2+beta0*log(z)),
  // since latter already present in soft rescaling term.
  if (!doMassive && order >= 3) {
    for ( map<string,double>::iterator it = wts.begin(); it !=wts.end(); ++it){

      double mukf = 1.;
      if (it->first == "base")
        mukf = renormMultFac;
      else if (it->first == "Variations:muRisrDown")
        mukf = settingsPtr->parm("Variations:muRisrDown");
      else if (it->first == "Variations:muRisrUp")
        mukf = settingsPtr->parm("Variations:muRisrUp");
      else continue;

      double NF          = getNF(pT2 * mukf);
      double alphasPT2pi = as2Pi(pT2, order, mukf);
      double TF          = TR*NF;
      // SplittingQCD function directly taken from Mathematica file.
      // Note: After removal of the cusp anomalous dimensions, the NLO kernel
      // is a purely collinear term. As such, it should not distinguish between
      // colour structures, and hence should contribute equally to isr_qcd_G2GG1
      // and isr_qcd_G2GG2. Hence one factor of 0.5 . Then, another factor of 0.5
      // is necessary, since the NLO kernel in the Mathematica file is normalised
      // to CA, and not 2*CA (as is the case for the LO kernel above).
      double pgg1   = preFac * 0.5 / ( 18*z*(pow2(z)-1) ) * 0.5 * (
      TF*(-1 + pow2(z))*((4*(-1 + z)*(-23 + z*(6 + z*(10 + z*(4 + 23*z)))))/(-1 + pow2(z))
                        +(24*(1 - z)*z*log(z)*pow2(1 + z))/(-1 + pow2(z)))
     +(CF*TF*(-1 + pow2(z))*((36*(1 - z)*z*(1 + z)*(3 + 5*z)*log(z))/(-1 + pow2(z))
                            +(24*(1 + z)*(-1 + z*(11 + 5*z))*pow2(-1 + z))/(-1 + pow2(z))
                            -(36*(-1 + z)*z*pow2(1 + z)*pow2(log(z)))/(-1 + pow2(z))))/CA
     -72*CA*(-1 + z)*DiLog(1/(1 + z))*pow2(1 + z + pow2(z))
     +CA*(-1 + pow2(z))*((6*(1 - z)*z*(1 + z)*(25 + 11*z*(-1 + 4*z))*log(z))/(-1 + pow2(z))
                        +((1 - z)*(z*(1 + z)*(25 + 109*z) + 6*(2 + z*(1 + 2*z*(1 + z)))*pow2(M_PI)))/(-1 + pow2(z))
                        +(72*(1 + z)*log(1 - z)*log(z)*pow2(1 + (-1 + z)*z))/(-1 + pow2(z))
                        -(36*z*pow2(log(z))*pow2(1 + z - pow2(z)))/(-1 + pow2(z))
                        +(144*DiLog(1/(1 + z))*pow2(1 + z + pow2(z)))/(1 + z)
                        +(36*(-1 + z)*pow2(log(1 + z))*pow2(1 + z + pow2(z)))/(-1 + pow2(z))) );

      // replace 1/x term in NLO kernel with x/(x^2+kappa^2)
      pgg1 += -preFac * 0.5 * 40./9.*TF * 0.5 * ( z/(pow2(z)+kappa2) - 1./z); 
      // Add NLO term.
      it->second += alphasPT2pi*pgg1;
    }
  }

  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2",
    wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function G->GG (ISR)

// Return true if this kernel should partake in the evolution.
bool isr_qcd_G2GG2::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].id() == 21 );
}

int isr_qcd_G2GG2::kinMap()                 { return 1;}
int isr_qcd_G2GG2::motherID(int)            { return 21;} 
int isr_qcd_G2GG2::sisterID(int)            { return 21;} 
double isr_qcd_G2GG2::gaugeFactor ( int, int )        { return 2.*CA;}
double isr_qcd_G2GG2::symmetryFactor ( int, int )     { return 0.5;}

int isr_qcd_G2GG2::radBefID(int idRA, int){
 if (idRA==21) return 21;
 return 0;
}
pair<int,int> isr_qcd_G2GG2::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  int colRemove = (colRadAfter == colEmtAfter)
                ? colRadAfter : acolRadAfter;
  int col       = (colRadAfter  == colRemove)
                ? acolEmtAfter : colRadAfter;
  int acol      = (acolRadAfter == colRemove)
                ? colEmtAfter : acolRadAfter;
  return make_pair(col,acol);
}

vector <int> isr_qcd_G2GG2::recPositions( const Event& state, int iRad, int iEmt) {

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();
  int colShared = (colRad  > 0 && colRad == colEmt) ? colEmt
                : (acolRad > 0 && acolEmt == acolRad) ? acolEmt : 0;
  // Particles to exclude from colour tracing.
  vector<int> iExc(1,iRad); iExc.push_back(iEmt);

  // Find partons connected via emitted colour line.
  vector<int> recs;
  if ( colRad != 0 && colRad != colShared) {
    int acolF = findCol(colRad, iExc, state, 1);
    int  colI = findCol(colRad, iExc, state, 2);
    if (acolF  > 0 && colI == 0) recs.push_back (acolF);
    if (acolF == 0 && colI >  0) recs.push_back (colI);
  }
  // Find partons connected via emitted anticolour line.
  if ( acolRad != 0 && acolRad != colShared) {
    int  colF = findCol(acolRad, iExc, state, 2);
    int acolI = findCol(acolRad, iExc, state, 1);
    if ( colF  > 0 && acolI == 0) recs.push_back (colF);
    if ( colF == 0 && acolI >  0) recs.push_back (acolI);
  }

  // Done.
  return recs;
}

// Pick z for new splitting.
double isr_qcd_G2GG2::zSplit(double zMinAbs, double, double m2dip) {
  double R      = rndmPtr->flat();
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;

  // For small values of pT, increase overestimate to avoid large weights.
  double tOld(splitInfo.kinematics()->pT2Old);
  if (tOld > 0. && tOld < SMALL_TEVOL) kappa2 *= sqrt(kappa2);

  // Pick according to soft + 1/z
  double res = (-2.*pow(kappa2,R)*pow(zMinAbs,2.*R) +
             sqrt(4.*pow(kappa2,2.*R)
                   *pow(zMinAbs,4.*R)
                + 4.*(pow(kappa2,R) + pow(kappa2,1. + R))
                   *pow(zMinAbs,2.*R)
                   *(-(pow(kappa2,R)*pow(zMinAbs,2.*R))
                     + kappa2
                       *pow(1. + kappa2 - 2.*zMinAbs + pow(zMinAbs,2.),R))))
          / (2.*(-(pow(kappa2,R)*pow(zMinAbs,2.*R))
                 + kappa2
                   *pow(1. + kappa2 - 2.*zMinAbs + pow(zMinAbs,2.),R)));
  return res;
}

// New overestimates, z-integrated versions.
double isr_qcd_G2GG2::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double kappa2 = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;

  // For small values of pT, increase overestimate to avoid large weights.
  double tOld(splitInfo.kinematics()->pT2Old);
  if (tOld > 0. && tOld < SMALL_TEVOL) kappa2 *= sqrt(kappa2);

  // Overestimate by soft + 1/z
  wt   = preFac
       *0.5*( log(1./pow2(zMinAbs) + pow2(1.-zMinAbs)/(kappa2*pow2(zMinAbs))));

  return wt;
}

// Return overestimate for new splitting.
double isr_qcd_G2GG2::overestimateDiff(double z, double m2dip, int) {
  double wt        = 0.;
  double preFac    = symmetryFactor() * gaugeFactor();
  double kappa2    = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;

  // For small values of pT, increase overestimate to avoid large weights.
  double tOld(splitInfo.kinematics()->pT2Old);
  if (tOld > 0. && tOld < SMALL_TEVOL) kappa2 *= sqrt(kappa2);

  // Overestimate by soft + 1/z
  wt  = preFac
      * ((1.-z) / ( pow2(1.-z) + kappa2) + 1./z);
  return wt;
}

// Return kernel for new splitting.
bool isr_qcd_G2GG2::calc(const Event& state, int orderNow) {

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

  double preFac = symmetryFactor() * gaugeFactor();
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  double kappa2 = pT2/m2dip;

  map<string,double> wts;
  double wt_base_as1 = preFac * 0.5 * ( z / ( pow2(z) + kappa2) - 1. )
                     + preFac * z*(1.-z);
  //wts.insert( make_pair("base",
  //  preFac * 0.5 * ( z / ( pow2(z) + kappa2) - 1. ) + preFac * z*(1.-z) ));
  //if (doVariations) {
  //  // Create muR-variations.
  //  if (settingsPtr->parm("Variations:muRisrDown") != 1.)
  //    wts.insert( make_pair("Variations:muRisrDown", 
  //      preFac * 0.5 * ( z / ( pow2(z) + kappa2) - 1. ) + preFac * z*(1.-z) ));
  //  if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
  //    wts.insert( make_pair("Variations:muRisrUp", 
  //      preFac * 0.5 * ( z / ( pow2(z) + kappa2) - 1. ) + preFac * z*(1.-z) ));
  //}
  wts.insert( make_pair("base", wt_base_as1 ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", wt_base_as1 ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", wt_base_as1 ));
  }

  // Correction for massive IF splittings.
  bool doMassive = ( m2Rec > 0. && splitType == 2);

  if (doMassive) {
    // Construct CS variables.
    double uCS = kappa2 / (1-z);
    double massCorr = - m2Rec / m2dip * uCS / (1.-uCS);
    // Mass correction shared in equal parts between both g->gg kernels.
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second += preFac * 0.5 * massCorr;

    wt_base_as1 += preFac * 0.5 * massCorr;
  }

  // Add NLO term, subtracted by 1/(1-z)*(Gamma2+beta0*log(z)),
  // since latter already present in soft rescaling term.
  if (!doMassive && order >= 3) {
    for ( map<string,double>::iterator it = wts.begin(); it !=wts.end(); ++it){

      double mukf = 1.;
      if (it->first == "base")
        mukf = renormMultFac;
      else if (it->first == "Variations:muRisrDown")
        mukf = settingsPtr->parm("Variations:muRisrDown");
      else if (it->first == "Variations:muRisrUp")
        mukf = settingsPtr->parm("Variations:muRisrUp");
      else continue;

      double NF          = getNF(pT2 * mukf);
      double alphasPT2pi = as2Pi(pT2, order, mukf);
      double TF          = TR*NF;
      // SplittingQCD function directly taken from Mathematica file.
      // Note: After removal of the cusp anomalous dimensions, the NLO kernel
      // is a purely collinear term. As such, it should not distinguish between
      // colour structures, and hence should contribute equally to isr_qcd_G2GG1
      // and isr_qcd_G2GG2. Hence one factor of 0.5 . Then, another factor of 0.5
      // is necessary, since the NLO kernel in the Mathematica file is normalised
      // to CA, and not 2*CA (as is the case for the LO kernel above).
      double pgg1   = preFac * 0.5 / ( 18*z*(pow2(z)-1) ) * 0.5 * (
      TF*(-1 + pow2(z))*((4*(-1 + z)*(-23 + z*(6 + z*(10 + z*(4 + 23*z)))))/(-1 + pow2(z))
                         +(24*(1 - z)*z*log(z)*pow2(1 + z))/(-1 + pow2(z)))
     +(CF*TF*(-1 + pow2(z))*((36*(1 - z)*z*(1 + z)*(3 + 5*z)*log(z))/(-1 + pow2(z))
                            +(24*(1 + z)*(-1 + z*(11 + 5*z))*pow2(-1 + z))/(-1 + pow2(z))
                            -(36*(-1 + z)*z*pow2(1 + z)*pow2(log(z)))/(-1 + pow2(z))))/CA
     -72*CA*(-1 + z)*DiLog(1/(1 + z))*pow2(1 + z + pow2(z))
     +CA*(-1 + pow2(z))*((6*(1 - z)*z*(1 + z)*(25 + 11*z*(-1 + 4*z))*log(z))/(-1 + pow2(z))
                        +((1 - z)*(z*(1 + z)*(25 + 109*z) + 6*(2 + z*(1 + 2*z*(1 + z)))*pow2(M_PI)))/(-1 + pow2(z))
                        +(72*(1 + z)*log(1 - z)*log(z)*pow2(1 + (-1 + z)*z))/(-1 + pow2(z))
                        -(36*z*pow2(log(z))*pow2(1 + z - pow2(z)))/(-1 + pow2(z))
                        +(144*DiLog(1/(1 + z))*pow2(1 + z + pow2(z)))/(1 + z)
                        +(36*(-1 + z)*pow2(log(1 + z))*pow2(1 + z + pow2(z)))/(-1 + pow2(z))) );
      // replace 1/z term in NLO kernel with z/(z^2+kappa^2)
      pgg1 += -preFac * 0.5 * 40./9.*TF * 0.5 * ( z/(pow2(z)+kappa2) - 1./z); 
      // Add NLO term.
      it->second  += alphasPT2pi*pgg1;
    }
  }

  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2",
    wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function G->QQ (ISR)

// Return true if this kernel should partake in the evolution.
bool isr_qcd_G2QQ::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].isQuark() );
}

int isr_qcd_G2QQ::kinMap()                 { return 1;}
int isr_qcd_G2QQ::motherID(int)            { return 21;} 
int isr_qcd_G2QQ::sisterID(int idDaughter) { return -idDaughter;} 
double isr_qcd_G2QQ::gaugeFactor ( int, int )        { return TR;}
double isr_qcd_G2QQ::symmetryFactor ( int, int )     { return 1.0;}

int isr_qcd_G2QQ::radBefID(int, int idEA){ 
  if (particleDataPtr->isQuark(idEA)) return -idEA;
  return 0;
  //return -idEA;
}
pair<int,int> isr_qcd_G2QQ::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  bool isQuark  = (acolEmtAfter > 0);
  int colRemove = (colRadAfter  == colEmtAfter)
                ? colRadAfter : 0;
  int col       = (colRadAfter  == colRemove)
                ? acolEmtAfter : colRadAfter;
  if (isQuark) return make_pair(col,0);
  colRemove     = (acolRadAfter == acolEmtAfter)
                ? acolRadAfter : 0;
  int acol      = (acolRadAfter == colRemove)
                ? colEmtAfter : acolRadAfter;
  return make_pair(0,acol);
}

vector <int> isr_qcd_G2QQ::recPositions( const Event& state, int iRad, int iEmt) {

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();
  int colShared = (colRad  > 0 && colRad == acolEmt) ? colRad
                : (acolRad > 0 && colEmt == acolRad) ? colEmt : 0;
  // Particles to exclude from colour tracing.
  vector<int> iExc(1,iRad); iExc.push_back(iEmt);

  // Find partons connected via emitted colour line.
  vector<int> recs;
  if ( colRad != 0 && colRad != colShared) {
    int acolF = findCol(colRad, iExc, state, 1);
    int  colI = findCol(colRad, iExc, state, 2);
    if (acolF  > 0 && colI == 0) recs.push_back (acolF);
    if (acolF == 0 && colI >  0) recs.push_back (colI);
  }
  // Find partons connected via emitted anticolour line.
  if ( acolRad != 0 && acolRad != colShared) {
    int  colF = findCol(acolRad, iExc, state, 2);
    int acolI = findCol(acolRad, iExc, state, 1);
    if ( colF  > 0 && acolI == 0) recs.push_back (colF);
    if ( colF == 0 && acolI >  0) recs.push_back (acolI);
  }
  // Done.
  return recs;
}

// Pick z for new splitting.
double isr_qcd_G2QQ::zSplit(double zMinAbs, double zMaxAbs, double) {
  // Note: Combined with PDF ratio, flat overestimate performs
  // better than using the full splitting kernel as overestimate. 
  double res = zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs);
  return res;
}

// New overestimates, z-integrated versions.
double isr_qcd_G2QQ::overestimateInt(double zMinAbs, double zMaxAbs,
  double, double, int) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  // Note: Combined with PDF ratio, flat overestimate performs
  // better than using the full splitting kernel as overestimate. 
  wt  = preFac 
      * 2. * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double isr_qcd_G2QQ::overestimateDiff(double, double, int) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  // Note: Combined with PDF ratio, flat overestimate performs
  // better than using the full splitting kernel as overestimate. 
  wt = preFac 
     * 2.;
  return wt;
}

// Return kernel for new splitting.
bool isr_qcd_G2QQ::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip);
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    //m2Rad(splitInfo.kinematics()->m2RadAft),
    //m2Rec(splitInfo.kinematics()->m2Rec),
    //m2Emt(splitInfo.kinematics()->m2EmtAft);
  //int splitType(splitInfo.type);

  double preFac = symmetryFactor() * gaugeFactor();
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  double kappa2 = pT2 / m2dip;

  map<string,double> wts;
  double wt_base_as1 =  preFac * (pow(1.-z,2.) + pow(z,2.));
  //wts.insert( make_pair("base", preFac * (pow(1.-z,2.) + pow(z,2.)) ));
  //if (doVariations) {
  //  // Create muR-variations.
  //  if (settingsPtr->parm("Variations:muRisrDown") != 1.)
  //    wts.insert( make_pair("Variations:muRisrDown", 
  //      preFac *  (pow(1.-z,2.) + pow(z,2.)) ));
  //  if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
  //    wts.insert( make_pair("Variations:muRisrUp", 
  //      preFac *  (pow(1.-z,2.) + pow(z,2.)) ));
  //}
  wts.insert( make_pair("base", wt_base_as1 ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", wt_base_as1 ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", wt_base_as1 ));
  }

  if (order >= 3) {
    for ( map<string,double>::iterator it = wts.begin(); it !=wts.end(); ++it){

      double mukf = 1.;
      if (it->first == "base")
        mukf = renormMultFac;
      else if (it->first == "Variations:muRisrDown")
        mukf = settingsPtr->parm("Variations:muRisrDown");
      else if (it->first == "Variations:muRisrUp")
        mukf = settingsPtr->parm("Variations:muRisrUp");
      else continue;

      double alphasPT2pi = as2Pi(pT2, order, mukf);
      // SplittingQCD function directly taken from Mathematica file.
      double pgq1 = preFac * (
      (CF*(4 - 9*z + 4*log(1 - z) + (-1 + 4*z)*log(z)
          -(2*(1 + 2*(-1 + z)*z)*(-15 - 3*(-2 + log(-1 + 1/z))*log(-1 + 1/z) + pow2(M_PI)))/3.
          +(-1 + 2*z)*pow2(log(z)))
      +(2*CA*(20 - 18*z*(1 + 2*z*(1 + z))*DiLog(1/(1 + z))
             +z*(-18 + (225 - 218*z)*z + pow2(M_PI)*(3 + 6*pow2(z)))
             +3*z*(12*(-1 + z)*z*log(1 - z)
                  +log(z)*(3 + 4*z*(6 + 11*z) - 3*(1 + 2*z)*log(z))
                  +(-3 - 6*(-1 + z)*z)*pow2(log(1 - z))
                  -3*(1 + 2*z*(1 + z))*pow2(log(1 + z)))))/(9.*z))/2. );
      // replace 1/z term in NLO kernel with z/(z^2+kappa^2)
      pgq1 += preFac * 20./9.*CA * ( z/(pow2(z)+kappa2) - 1./z); 
      // Add NLO term.
      it->second += alphasPT2pi*pgq1;
    }
  }

  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2",
    wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function Q->GQ (ISR)

// Return true if this kernel should partake in the evolution.
bool isr_qcd_Q2GQ::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].id() == 21 );
}

int isr_qcd_Q2GQ::kinMap()                 { return 1;}
int isr_qcd_Q2GQ::motherID(int)            { return 1;} // Use 1 as dummy 
int isr_qcd_Q2GQ::sisterID(int)            { return 1;} // Use 1 as dummy
double isr_qcd_Q2GQ::gaugeFactor ( int, int )        { return CF;}
double isr_qcd_Q2GQ::symmetryFactor ( int, int )     { return 0.5;}

int isr_qcd_Q2GQ::radBefID(int idRA, int){
  if (particleDataPtr->isQuark(idRA)) return 21;
  return 0;
}
pair<int,int> isr_qcd_Q2GQ::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  int col  = (colRadAfter  > 0) ? colRadAfter  : acolEmtAfter;
  int acol = (acolRadAfter > 0) ? acolRadAfter : colEmtAfter;
  return make_pair(col,acol);
}

vector <int> isr_qcd_Q2GQ::recPositions( const Event& state, int iRad, int iEmt) {

  // For Q->GQ, swap radiator and emitted, since we now have to trace the
  // radiator's colour connections.
  if ( state[iEmt].idAbs() < 20 && state[iRad].id() == 21) swap( iRad, iEmt);

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();
  int colShared = (colRad  > 0 && colRad == acolEmt) ? colRad
                : (acolRad > 0 && colEmt == acolRad) ? colEmt : 0;
  // Particles to exclude from colour tracing.
  vector<int> iExc(1,iRad); iExc.push_back(iEmt);

  // Find partons connected via emitted colour line.
  vector<int> recs;
  if ( colEmt != 0 && colEmt != colShared) {
    int acolF = findCol(colEmt, iExc, state, 1);
    int  colI = findCol(colEmt, iExc, state, 2);
    if (acolF  > 0 && colI == 0) recs.push_back (acolF);
    if (acolF == 0 && colI >  0) recs.push_back (colI);
  }
  // Find partons connected via emitted anticolour line.
  if ( acolEmt != 0 && acolEmt != colShared) {
    int  colF = findCol(acolEmt, iExc, state, 2);
    int acolI = findCol(acolEmt, iExc, state, 1);
    if ( colF  > 0 && acolI == 0) recs.push_back (colF);
    if ( colF == 0 && acolI >  0) recs.push_back (acolI);
  }
  // Done.
  return recs;
}

// Pick z for new splitting.
double isr_qcd_Q2GQ::zSplit(double zMinAbs, double, double) {
  double R   = rndmPtr->flat();
  double res = pow(zMinAbs,3./4.)
          / ( pow(1. + R*(-1. + pow(zMinAbs,-3./8.)),2./3.)
             *pow(R - (-1. + R)*pow(zMinAbs,3./8.),2.));
  return res;
}

// New overestimates, z-integrated versions.
double isr_qcd_Q2GQ::overestimateInt(double zMinAbs, double,
  double, double, int) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt            = preFac * 2./3. * (8.*(-1. + pow(zMinAbs,-3./8.)));

  return wt;
}

// Return overestimate for new splitting.
double isr_qcd_Q2GQ::overestimateDiff(double z, double, int) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt            = preFac * 2. / pow(z,11./8.);
  return wt;
}

// Return kernel for new splitting.
bool isr_qcd_Q2GQ::calc(const Event& state, int orderNow) {

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

  double preFac = symmetryFactor() * gaugeFactor();
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  double kappa2 = pT2 / m2dip;

  map<string,double> wts;
  double wt_base_as1 = preFac*(z + 2.*z/(pow2(z)+kappa2) - 2.);
  //wts.insert( make_pair("base", preFac*(z + 2.*z/(pow2(z)+kappa2) - 2.) ));
  //if (doVariations) {
  //  // Create muR-variations.
  //  if (settingsPtr->parm("Variations:muRisrDown") != 1.)
  //    wts.insert( make_pair("Variations:muRisrDown", 
  //      preFac * (z + 2.*z/(pow2(z)+kappa2) - 2.) ));
  //  if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
  //    wts.insert( make_pair("Variations:muRisrUp", 
  //      preFac * (z + 2.*z/(pow2(z)+kappa2) - 2.) ));
  //}

  wts.insert( make_pair("base", wt_base_as1 ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", wt_base_as1 ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", wt_base_as1 ));
  }

  // Correction for massive IF splittings.
  bool doMassive = ( m2Rec > 0. && splitType == 2);

  if (doMassive) {
    // Construct CS variables.
    double uCS = kappa2 / (1-z);

    double massCorr = -2. * m2Rec / m2dip * uCS / (1.-uCS);
    // Add correction.
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second += preFac * massCorr;

    wt_base_as1 += preFac * massCorr;

  }

  if (!doMassive && order >= 3) {
    for ( map<string,double>::iterator it = wts.begin(); it !=wts.end(); ++it){

      double mukf = 1.;
      if (it->first == "base")
        mukf = renormMultFac;
      else if (it->first == "Variations:muRisrDown")
        mukf = settingsPtr->parm("Variations:muRisrDown");
      else if (it->first == "Variations:muRisrUp")
        mukf = settingsPtr->parm("Variations:muRisrUp");
      else continue;

      double NF          = getNF(pT2 * mukf);
      double alphasPT2pi = as2Pi(pT2, order, mukf);
      // SplittingQCD function directly taken from Mathematica file.
      double TF          = TR*NF;
      double pqg1 = preFac * (
      (-9*CF*z*(5 + 7*z) - 16*TF*(5 + z*(-5 + 4*z))
       +36*CA*(2 + z*(2 + z))*DiLog(1/(1 + z))
       +2*CA*(9 + z*(19 + z*(37 + 44*z)) - 3*pow2(M_PI)*(2 + pow2(z)))
       +3*(-2*log(1 - z)*(CA*(-22 + (22 - 17*z)*z)
                         +4*TF*(2 + (-2 + z)*z) + 3*CF*(6 + z*(-6 + 5*z))
                         +6*CA*(2 + (-2 + z)*z)*log(z))
       +z*log(z)*(3*CF*(4 + 7*z) - 2*CA*(36 + z*(15 + 8*z))
                 +3*(CF*(-2 + z) + 2*CA*(2 + z))*log(z))
       +6*(CA - CF)*(2 + (-2 + z)*z)*pow2(log(1 - z))
       +6*CA*(2 + z*(2 + z))*pow2(log(1 + z))))/(18.*z) );
      // replace 1/z term in NLO kernel with z/(z^2+kappa^2)
      pqg1 +=  - preFac * 40./9.*TF * ( z/(pow2(z)+kappa2) - 1./z); 
      // Add NLO term.
      it->second += alphasPT2pi*pqg1;
    }
  }

  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2",
    wts["base"] - wt_base_as1 ));

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
bool isr_qcd_Q2qQqbarDist::canRadiate ( const Event& state,
  map<string,int> ints, map<string,bool>, Settings*, PartonSystems*,
  BeamParticle*) {
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].isQuark() );
}

int isr_qcd_Q2qQqbarDist::kinMap()                 { return 2;}
int isr_qcd_Q2qQqbarDist::motherID(int idDaughter) { return idDaughter;} 
int isr_qcd_Q2qQqbarDist::sisterID(int)            { return 1;} 
double isr_qcd_Q2qQqbarDist::gaugeFactor ( int, int )        { return CF;}
double isr_qcd_Q2qQqbarDist::symmetryFactor ( int, int )     { return 1.;}

int isr_qcd_Q2qQqbarDist::radBefID(int idRA, int) { 
  if (particleDataPtr->isQuark(idRA)) return idRA;
  return 0;
  return idRA;
}
pair<int,int> isr_qcd_Q2qQqbarDist::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  bool isQuark  = (colRadAfter > 0);
  int colRemove = (colRadAfter == colEmtAfter)
                ? colRadAfter : 0;
  int col       = (colRadAfter  == colRemove)
                ? acolEmtAfter : colRadAfter;
  if (isQuark) return make_pair(col,0); 
  colRemove = (acolRadAfter == acolEmtAfter)
                ? acolRadAfter : 0;
  int acol      = (acolRadAfter  == colRemove)
                ? colEmtAfter : acolRadAfter;
  return make_pair(0,acol); 
}

// Pick z for new splitting.
double isr_qcd_Q2qQqbarDist::zSplit(double zMinAbs, double zMaxAbs,
  double m2dip) {
  double Rz      = rndmPtr->flat();
  double res     = 1.;
  // z est from 1/(z + kappa^2)
  double kappa2  = pow(settingsPtr->parm("SpaceShower:pTmin"), 2) / m2dip;

  res = pow( (pow(kappa2,1) + zMaxAbs)/(pow(kappa2,1) + zMinAbs), -Rz )
      * (pow(kappa2,1) + zMaxAbs - pow(kappa2,1)
                           *pow((pow(kappa2,1) + zMaxAbs)/(pow(kappa2,1) + zMinAbs), Rz));

  // Conversions to light flavours can have very large PDF
  // ratios at threshold. Thus, choose large overstimate a priori.
  if ( splitInfo.recBef()->isFinal
    && (splitInfo.radBef()->id < 0 || abs(splitInfo.radBef()->id) > 2) ) {
    double k = pow(kappa2,1);
    res = pow(k,0.5)
        * tan(   Rz*atan(zMaxAbs*pow(k,-0.5))
          - (Rz-1.)*atan(zMinAbs*pow(k,-0.5)));
  }

  return res;

}

// New overestimates, z-integrated versions.
double isr_qcd_Q2qQqbarDist::overestimateInt(double zMinAbs, double zMaxAbs,
  double, double m2dip, int orderNow) {

  // Do nothing without other NLO kernels!
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  if (order < 3) return 0.0;

  double preFac  = symmetryFactor() * gaugeFactor();
  double pT2min  = pow2(settingsPtr->parm("SpaceShower:pTmin"));
  // Overestimate chosen to have accept weights below one for kappa~0.1
  // z est from 1/(z + kappa^2)
  double kappa2  = pT2min/m2dip;

  double wt = preFac * TR * 20./9. 
            * log( ( pow(kappa2,1) + zMaxAbs) / ( pow(kappa2,1) + zMinAbs) );

  // Conversions to light flavours can have very large PDF
  // ratios at threshold. Thus, choose large overstimate a priori.
  if ( splitInfo.recBef()->isFinal
    && (splitInfo.radBef()->id < 0 || abs(splitInfo.radBef()->id) > 2) ) {
    double k = pow(kappa2,1);
    wt = preFac * TR * 20./9.
       * ( atan(zMaxAbs*pow(k,-0.5))
         - atan(zMinAbs*pow(k,-0.5)))*pow(k,-0.5);
  }

  // This splitting is down by one power of alphaS !
  wt *= as2Pi(pT2min);
  return wt;

}

// Return overestimate for new splitting.
double isr_qcd_Q2qQqbarDist::overestimateDiff(double z, double m2dip,
  int orderNow) {

  // Do nothing without other NLO kernels!
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  if (order < 3) return 0.0;

  double preFac    = symmetryFactor() * gaugeFactor();
  double pT2min    = pow2(settingsPtr->parm("SpaceShower:pTmin"));
  // Overestimate chosen to have accept weights below one for kappa~0.1
  double kappa2    = pT2min/m2dip;

  double wt = preFac * TR * 20./ 9. * 1 / (z + pow(kappa2,1));

  // Conversions to light flavours can have very large PDF
  // ratios at threshold. Thus, choose large overstimate a priori.
  if ( splitInfo.recBef()->isFinal
    && (splitInfo.radBef()->id < 0 || abs(splitInfo.radBef()->id) > 2) )
    wt = preFac * TR * 20./ 9. * 1. / (z*z + pow(kappa2,1));

  wt *= as2Pi(pT2min);
  return wt;

}

// Return kernel for new splitting.
bool isr_qcd_Q2qQqbarDist::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z),
    pT2(splitInfo.kinematics()->pT2),
    //m2dip(splitInfo.kinematics()->m2Dip),
    xa(splitInfo.kinematics()->xa),
    sai(splitInfo.kinematics()->sai),
    m2aij(splitInfo.kinematics()->m2RadBef),
    m2a(splitInfo.kinematics()->m2RadAft),
    m2i(splitInfo.kinematics()->m2EmtAft),
    m2j(splitInfo.kinematics()->m2EmtAft2),
    m2k(splitInfo.kinematics()->m2Rec);

  // Do nothing without other NLO kernels!
  map<string,double> wts;
  //int order          = (orderNow > 0) ? orderNow : correctionOrder;
  int order          = (orderNow > -1) ? orderNow : correctionOrder;
  //if (order < 3) { 
  if (order < 3 || m2aij > 0. || m2a > 0. || m2i > 0. || m2j > 0. || m2k > 0.){
    wts.insert( make_pair("base", 0.) );
    if (doVariations && settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", 0.));
    if (doVariations && settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", 0.));
    clearKernels();
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
      kernelVals.insert(make_pair( it->first, it->second ));
    return true;
  }

  // Choose if simulating endpoint or differential 1->3 (latter containing
  // both sai=0 and sai !=0). For choice of endpoint, set sai=0 later.
  bool isEndpoint = (rndmPtr->flat() < 0.5);

  Event trialEvent(state);
  bool physical = true;
  if (splitInfo.recBef()->isFinal)
    physical = isr->branch_IF(trialEvent, true, &splitInfo);
  else
    physical = isr->branch_II(trialEvent, true, &splitInfo);

  // Get invariants.
  Vec4 pa(trialEvent[splitInfo.iRadAft].p());
  Vec4 pk(trialEvent[splitInfo.iRecAft].p());
  Vec4 pj(trialEvent[splitInfo.iEmtAft].p());
  Vec4 pi(trialEvent[splitInfo.iEmtAft2].p());

  double sign = (splitInfo.recBef()->isFinal) ? 1. : -1.;
  double p2ai(-sai + m2a + m2i),
         p2aj( (-pa+pj).m2Calc()),
         p2ak( (-pa+sign*pk).m2Calc()),
         p2ij( (pi+pj).m2Calc()),
         p2ik( (pi+sign*pk).m2Calc()),
         p2jk( (pj+sign*pk).m2Calc());
  double saij = (-pa+pi+pj).m2Calc();
  double q2 = (-pa + sign*pk + pi + pj).m2Calc();
  double z1(-sign*pa*pk/(-sign*pk*(pa-pi-pj))),
         z2(sign*pi*pk/(-sign*pk*(pa-pi-pj))),
         z3(1-z1-z2);
  double pT2min = pow2(settingsPtr->parm("SpaceShower:pTmin"));
  if ( z1< 1. || z2 > 0. || z3 > 0.)
    physical = false;
  if ( splitInfo.recBef()->isFinal && -(q2+pT2/xa-p2ai) < pT2min)
    physical = false;

  // Use only massless for now!
  if ( abs(pa.m2Calc()-m2a) > sai || abs(pi.m2Calc()-m2i) > sai
    || abs(pj.m2Calc()-m2j) > sai || abs(pk.m2Calc()-m2k) > sai)
    physical = false;

  // Discard splitting if not in allowed phase space.
  if (!physical) {
    wts.insert( make_pair("base", 0.) );
    if (doVariations && settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", 0.));
    if (doVariations && settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", 0.));
    clearKernels();
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
      kernelVals.insert(make_pair( it->first, it->second ));
    return true;
  }

  // Calculate kernel.
  double prob = 0.0;
  if (isEndpoint) {

    prob = CF*TR*((1.0+z3*z3)/(1.0-z3)
                 +(1.0-2.0*z1*z2/pow2(z1+z2))*(1.0-z3+(1.0+z3*z3)/(1.0-z3)
                 *(log(z2/z1*z3/(1-z3))-1.0)));
    prob-= CF*TR*2.0*((1.0+z3*z3)/(1.0-z3)*log(-z3/(1.0-z3)) +1.0-z3)
                     *(1.0-2.0*z1*z2/pow2(z1+z2));

    // From xa integration volume?
    prob *= log(1/z);
    // Multiply by 2 since we randomly chose endpoint or fully differential.
    prob *= 2.0;
    // Weight of sai-selection?
    prob *= z/xa * 1. / (1.-p2ai/saij);

  } else {

    double s12(p2ai), s13(p2aj), s23(p2ij), s123(saij);
    double t123 = 2.*(z1*s23 - z2*s13)/(z1+z2) + (z1-z2)/(z1+z2)*s12;
    double CG = 0.5*CF*TR*s123/s12
               *( - pow2(t123)/ (s12*s123)
                  + (4.*z3 + pow2(z1-z2))/(z1+z2) + z1 + z2 - s12/s123 );
    double cosPhiKT1KT3 = pow2(p2ij*p2ak - p2aj*p2ik + p2ai*p2jk)
                        / (4.*p2ai*p2ij*p2ak*p2jk);
    double subt = CF*TR*s123/s12
                * ( (1.+z3*z3) / (1.-z3) * (1.-2.*z1*z2/pow2(1-z3))
                   + 4.*z1*z2*z3 / pow(1.-z3,3) * (1-2.*cosPhiKT1KT3) );
    prob = CG - subt;

    if ( abs(s12) < 1e-10) prob = 0.0;

    // From xa integration volume?
    prob *= log(1/z);
    // Multiply by 2 since we randomly chose endpoint or fully differential.
    prob *= 2.0;
    // Weight of sai-selection?
    prob *= z/xa * 1. / (1.-p2ai/saij);

  }

  // Remember that this might be an endpoint with vanishing sai.
  if (isEndpoint) { splitInfo.set_sai(0.0); }

  // Insert value of kernel into kernel list.
  wts.insert( make_pair("base", prob * as2Pi(pT2, order, renormMultFac) ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", prob
        * as2Pi(pT2, order, settingsPtr->parm("Variations:muRisrDown")) ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp",   prob
        * as2Pi(pT2, order, settingsPtr->parm("Variations:muRisrUp")) ));
  }

  // Multiply with z1 because of crossing.
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    it->second *= z;    

  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2", wts["base"] ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function Q-> Qbar Q Q (FSR)

// Return true if this kernel should partake in the evolution.
bool isr_qcd_Q2QbarQQId::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return (!state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() != 0
        && int(sharedColor(state, ints["iRad"], ints["iRec"]).size()) > 0
        && state[ints["iRad"]].isQuark() );
}

int isr_qcd_Q2QbarQQId::kinMap()                 { return 2;}
int isr_qcd_Q2QbarQQId::motherID(int idDaughter) { return -idDaughter;}
int isr_qcd_Q2QbarQQId::sisterID(int)            { return 1;}
double isr_qcd_Q2QbarQQId::gaugeFactor ( int, int )        { return CF;}
double isr_qcd_Q2QbarQQId::symmetryFactor ( int, int )     { return 1.;}

int isr_qcd_Q2QbarQQId::radBefID(int idRA, int) { 
  if (particleDataPtr->isQuark(idRA)) return idRA;
  return 0;
  //return idRA;
}
pair<int,int> isr_qcd_Q2QbarQQId::radBefCols(
  int colRadAfter, int, 
  int colEmtAfter, int acolEmtAfter) {
  bool isQuark = (colRadAfter > 0);
  if (isQuark) return make_pair(colEmtAfter,0);
  return make_pair(0,acolEmtAfter);
}

// Pick z for new splitting.
double isr_qcd_Q2QbarQQId::zSplit(double zMinAbs, double zMaxAbs,
  double m2dip) {
  double Rz = rndmPtr->flat();
  double kappa2  = pow2(settingsPtr->parm("SpaceShower:pTmin"))/m2dip;

  double res = pow( (pow(kappa2,1) + zMaxAbs)/(pow(kappa2,1) + zMinAbs), -Rz )
                  * (pow(kappa2,1) + zMaxAbs - pow(kappa2,1)
                           *pow((pow(kappa2,1) + zMaxAbs)/(pow(kappa2,1) + zMinAbs), Rz));

  // Conversions to light flavours can have very large PDF
  // ratios at threshold. Thus, choose large overstimate a priori.
  if ( splitInfo.recBef()->isFinal && splitInfo.radBef()->id < 0 ) {
    double k = pow(kappa2,1);
    res = pow(k,0.5)
        * tan(   Rz*atan(zMaxAbs*pow(k,-0.5))
          - (Rz-1.)*atan(zMinAbs*pow(k,-0.5)));
  }

  return res;
}

// New overestimates, z-integrated versions.
double isr_qcd_Q2QbarQQId::overestimateInt(double zMinAbs, double zMaxAbs,
  double, double m2dip, int orderNow) {

  // Do nothing without other NLO kernels!
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  if (order < 3) return 0.0;

  //double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  double pT2min = pow2(settingsPtr->parm("SpaceShower:pTmin"));
  double kappa2 = pT2min/m2dip;

  double wt = preFac * TR * 20./9. 
            * log( ( pow(kappa2,1) + zMaxAbs) / ( pow(kappa2,1) + zMinAbs) );

  // Conversions to light flavours can have very large PDF
  // ratios at threshold. Thus, choose large overstimate a priori.
  if ( splitInfo.recBef()->isFinal && splitInfo.radBef()->id < 0 ) {
    double k = pow(kappa2,1);
    wt = preFac * TR * 20./9.
       * ( atan(zMaxAbs*pow(k,-0.5))
         - atan(zMinAbs*pow(k,-0.5)))*pow(k,-0.5);
  }

  // Multiply by number of channels.
  wt *= 2.;

  wt *= as2Pi(pT2min);

  return wt;
}

// Return overestimate for new splitting.
double isr_qcd_Q2QbarQQId::overestimateDiff(double z, double m2dip,
  int orderNow) {

  // Do nothing without other NLO kernels!
  //int order     = (orderNow > 0) ? orderNow : correctionOrder;
  int order     = (orderNow > -1) ? orderNow : correctionOrder;
  if (order < 3) return 0.0;

  double wt      = 0.;
  double preFac  = symmetryFactor() * gaugeFactor();
  double pT2min  = pow2(settingsPtr->parm("SpaceShower:pTmin"));
  double kappa2  = pT2min/m2dip;

  //wt  = preFac * TR * 20./9. * z / ( pow2(z) + kappaOld2);
  wt  = preFac * TR * 20./9. * 1. / ( z + kappa2);

  // Conversions to light flavours can have very large PDF
  // ratios at threshold. Thus, choose large overstimate a priori.
  if ( splitInfo.recBef()->isFinal && splitInfo.radBef()->id < 0 )
    wt = preFac * TR * 20./ 9. * 1. / (z*z + pow(kappa2,1));

  // Multiply by number of channels.
  wt *= 2.;

  wt *= as2Pi(pT2min);

  return wt;
}

// Return kernel for new splitting.
bool isr_qcd_Q2QbarQQId::calc(const Event& state, int orderNow) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << orderNow << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z),
    pT2(splitInfo.kinematics()->pT2),
    //m2dip(splitInfo.kinematics()->m2Dip),
    xa(splitInfo.kinematics()->xa),
    sai(splitInfo.kinematics()->sai),
    m2aij(splitInfo.kinematics()->m2RadBef),
    m2a(splitInfo.kinematics()->m2RadAft),
    m2i(splitInfo.kinematics()->m2EmtAft),
    m2j(splitInfo.kinematics()->m2EmtAft2),
    m2k(splitInfo.kinematics()->m2Rec);
  //int splitType(splitInfo.type);

  // Do nothing without other NLO kernels!
  map<string,double> wts;
  //int order          = (orderNow > 0) ? orderNow : correctionOrder;
  int order          = (orderNow > -1) ? orderNow : correctionOrder;
  //if (order < 3) { 
  if (order < 3 || m2aij > 0. || m2a > 0. || m2i > 0. || m2j > 0. || m2k > 0.){
    wts.insert( make_pair("base", 0.) );
    if (doVariations && settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", 0.));
    if (doVariations && settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", 0.));
    clearKernels();
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
      kernelVals.insert(make_pair( it->first, it->second ));
    return true;
  }

  // Choose if simulating endpoint or differential 1->3 (latter containing
  // both sai=0 and sai !=0). For choice of endpoint, set sai=0 later.
  bool isEndpoint = (rndmPtr->flat() < 0.5);

  Event trialEvent(state);
  bool physical = true;
  if (splitInfo.recBef()->isFinal)
    physical = isr->branch_IF(trialEvent, true, &splitInfo);
  else
    physical = isr->branch_II(trialEvent, true, &splitInfo);

  // Get invariants.
  Vec4 pa(trialEvent[splitInfo.iRadAft].p());
  Vec4 pk(trialEvent[splitInfo.iRecAft].p());
  Vec4 pj(trialEvent[splitInfo.iEmtAft].p());
  Vec4 pi(trialEvent[splitInfo.iEmtAft2].p());
  double sign = (splitInfo.recBef()->isFinal) ? 1. : -1.;
  double p2ai(-sai + m2a + m2i),
         p2aj( (-pa+pj).m2Calc()),
         p2ak( (-pa+sign*pk).m2Calc()),
         p2ij( (pi+pj).m2Calc()),
         p2ik( (pi+sign*pk).m2Calc()),
         p2jk( (pj+sign*pk).m2Calc());
  double saij = (-pa+pi+pj).m2Calc();
  double q2 = (-pa + sign*pk + pi + pj).m2Calc(); 
  double z1(-sign*pa*pk/(-sign*pk*(pa-pi-pj))),
         z2(sign*pi*pk/(-sign*pk*(pa-pi-pj))),
         z3(1-z1-z2);
  double pT2min = pow2(settingsPtr->parm("SpaceShower:pTmin"));
  if ( z1< 1. || z2 > 0. || z3 > 0.)
    physical = false;
  if ( splitInfo.recBef()->isFinal && -(q2+pT2/xa-p2ai) < pT2min)
    physical = false;

  // Use only massless for now!
  if ( abs(pa.m2Calc()-m2a) > sai || abs(pi.m2Calc()-m2i) > sai
    || abs(pj.m2Calc()-m2j) > sai || abs(pk.m2Calc()-m2k) > sai)
    physical = false;

  // Discard splitting if not in allowed phase space.
  if (!physical) {
    wts.insert( make_pair("base", 0.) );
    if (doVariations && settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", 0.));
    if (doVariations && settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp", 0.));
    clearKernels();
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
      kernelVals.insert(make_pair( it->first, it->second ));
    return true;
  }

  // Calculate kernel.
  double prob = 0.0;
  if (isEndpoint) {

    prob = CF*TR*((1.0+z3*z3)/(1.0-z3)
                 +(1.0-2.0*z1*z2/pow2(z1+z2))*(1.0-z3+(1.0+z3*z3)/(1.0-z3)
                 *(log(z2/z1*z3/(1-z3))-1.0)));
    // Swapped contribution.
    prob+= CF*TR*((1.0+z2*z2)/(1.0-z2)
                 +(1.0-2.0*z1*z3/pow2(z1+z3))*(1.0-z2+(1.0+z2*z2)/(1.0-z2)
                 *(log(z3/z1*z2/(1-z2))-1.0)));
    // Subtraction.
    prob-= CF*TR*2.0*((1.0+z3*z3)/(1.0-z3)*log(-z3/(1.0-z3)) +1.0-z3)
                     *(1.0-2.0*z1*z2/pow2(z1+z2));
    // Swapped subtraction.
    prob-= CF*TR*2.0*((1.0+z2*z2)/(1.0-z2)*log(-z2/(1.0-z2)) +1.0-z2)
                     *(1.0-2.0*z1*z3/pow2(z1+z3));

    // From xa integration volume?
    prob *= log(1/z);
    // Multiply by 2 since we randomly chose endpoint or fully differential.
    prob *= 2.0;
    // Weight of sai-selection?
    prob *= z/xa * 1. / (1.-p2ai/saij);

  } else {

    double s12(p2ai), s13(p2aj), s23(p2ij), s123(saij);
    double t123 = 2.*(z1*s23 - z2*s13)/(z1+z2) + (z1-z2)/(z1+z2)*s12;
    double CG = 0.5*CF*TR*s123/s12
               *( - pow2(t123)/ (s12*s123)
                  + (4.*z3 + pow2(z1-z2))/(z1+z2) + z1 + z2 - s12/s123 );
    // Swapped kernel.
    double t132 = 2.*(z1*s23 - z3*s12)/(z1+z3) + (z1-z3)/(z1+z3)*s13;
    CG       += 0.5*CF*TR*s123/s13
               *( - pow2(t132)/ (s13*s123)
                  + (4.*z2 + pow2(z1-z3))/(z1+z3) + z1 + z3 - s13/s123 );
    // Interference term.
    CG       += CF*(CF-0.5*CA)
              * ( 2.*s23/s12
                + s123/s12 * ( (1.+z1*z1)/(1-z2) - 2.*z2/(1.-z3) )
                - s123*s123/(s12*s13) * 0.5*z1*(1.+z1*z1) / ((1.-z2)*(1.-z3)) );
    // Swapped interference term.
    CG       += CF*(CF-0.5*CA)
              * ( 2.*s23/s13
                + s123/s13 * ( (1.+z1*z1)/(1-z3) - 2.*z3/(1.-z2) )
                - s123*s123/(s13*s12) * 0.5*z1*(1.+z1*z1) / ((1.-z3)*(1.-z2)) );
    // Subtraction.
    double cosPhiKT1KT3 = pow2(p2ij*p2ak - p2aj*p2ik + p2ai*p2jk)
                        / (4.*p2ai*p2ij*p2ak*p2jk);
    double subt = CF*TR*s123/s12
                * ( (1.+z3*z3) / (1.-z3) * (1.-2.*z1*z2/pow2(1-z3))
                   + 4.*z1*z2*z3 / pow(1.-z3,3) * (1-2.*cosPhiKT1KT3) );
    // Swapped subtraction.
    double cosPhiKT1KT2 = pow2(p2ij*p2ak + p2aj*p2ik - p2ai*p2jk)
                        / (4.*p2aj*p2ij*p2ak*p2ik);
    subt       += CF*TR*s123/s13
                * ( (1.+z2*z2) / (1.-z2) * (1.-2.*z1*z3/pow2(1-z2))
                   + 4.*z1*z3*z2 / pow(1.-z2,3) * (1-2.*cosPhiKT1KT2) );
    prob = CG - subt;

    if ( abs(s12) < 1e-10) prob = 0.0;

    // From xa integration volume?
    prob *= log(1/z);
    // Multiply by 2 since we randomly chose endpoint or fully differential.
    prob *= 2.0;
    // Weight of sai-selection?
    prob *= z/xa * 1. / (1.-p2ai/saij);

  }

  // Desymmetrize in i and j.
  prob *= (1.-xa) / (1-z);

  // Remember that this might be an endpoint with vanishing sai.
  if (isEndpoint) { splitInfo.set_sai(0.0); }

  // Insert value of kernel into kernel list.
  wts.insert( make_pair("base", prob * as2Pi(pT2, order, renormMultFac) ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRisrDown") != 1.)
      wts.insert( make_pair("Variations:muRisrDown", prob
        * as2Pi(pT2, order, settingsPtr->parm("Variations:muRisrDown")) ));
    if (settingsPtr->parm("Variations:muRisrUp")   != 1.)
      wts.insert( make_pair("Variations:muRisrUp",   prob
        * as2Pi(pT2, order, settingsPtr->parm("Variations:muRisrUp")) ));
  }

  // Multiply with z1 because of crossing.
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    it->second *= z;    

  // Store higher order correction separately.
  if (order > 0) wts.insert( make_pair("base_order_as2", wts["base"] ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_Q2QG_notPartial::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() == 0
        && state[ints["iRad"]].isQuark() );
}

int fsr_qcd_Q2QG_notPartial::kinMap()                 { return 1;}
int fsr_qcd_Q2QG_notPartial::motherID(int idDaughter) { return idDaughter;}
int fsr_qcd_Q2QG_notPartial::sisterID(int)            { return 21;}
double fsr_qcd_Q2QG_notPartial::gaugeFactor ( int, int )        { return CF;}
double fsr_qcd_Q2QG_notPartial::symmetryFactor ( int, int )     { return 1.;}

int fsr_qcd_Q2QG_notPartial::radBefID(int idRA, int) {
  if (particleDataPtr->isQuark(idRA)) return idRA;
  return 0;
}

vector<pair<int,int> > fsr_qcd_Q2QG_notPartial::radAndEmtCols(int iRad,
  int, Event state) {
  vector< pair<int,int> > ret;
  if (!state[iRad].isQuark() || state[splitInfo.iRecBef].colType() != 0)
    return ret;

//state.list();
//  cout << splitInfo.iRecBef << endl;
  int colType = (state[iRad].id() > 0) ? 1 : -1;

  int newCol1     = state.nextColTag();
  int colRadAft   = (colType > 0) ? newCol1 : state[iRad].col();
  int acolRadAft  = (colType > 0) ? state[iRad].acol() : newCol1;
  int colEmtAft1  = (colType > 0) ? state[iRad].col() : newCol1;
  int acolEmtAft1 = (colType > 0) ? newCol1 : state[iRad].acol();

  ret = createvector<pair<int,int> >
    (make_pair(colRadAft, acolRadAft))
    (make_pair(colEmtAft1, acolEmtAft1));

//cout << colRadAft << " " << acolRadAft << "\t" << colEmtAft1 << " " << acolEmtAft1 << endl;
//abort();


  /*if (particleDataPtr->colType(idRadAfterSave) != 0) {
    int sign      = (idRadAfterSave > 0) ? 1 : -1;
    int newCol    = state.nextColTag();
    ret[0].first  = (sign > 0) ? newCol : 0;
    ret[0].second = (sign > 0) ? 0 : newCol;
    ret[1].first  = (sign > 0) ? 0 : newCol;
    ret[1].second = (sign > 0) ? newCol : 0;
  }*/
  return ret;
}

pair<int,int> fsr_qcd_Q2QG_notPartial::radBefCols(
  int colRadAfter, int, 
  int colEmtAfter, int acolEmtAfter) {
  bool isQuark = (colRadAfter > 0);
  if (isQuark) return make_pair(colEmtAfter,0);
  return make_pair(0,acolEmtAfter);
}

vector <int> fsr_qcd_Q2QG_notPartial::recPositions( const Event&,
  int, int) {
  return vector<int>();
} 

// Pick z for new splitting.
double fsr_qcd_Q2QG_notPartial::zSplit(double zMinAbs, double, double m2dip) {
  double Rz        = rndmPtr->flat();

  double kappaMin4 = pow4(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p         = pow( 1. + pow2(1-zMinAbs)/kappaMin4, Rz );
  double res       = 1. - sqrt( p - 1. )*sqrt(kappaMin4);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qcd_Q2QG_notPartial::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {

  // Q -> QG, soft part (currently also used for collinear part).
  double preFac    = symmetryFactor() * gaugeFactor();
  double kappaMin4 = pow4(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac
                     *2. * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappaMin4);
  return wt;
}

// Return overestimate for new splitting.
double fsr_qcd_Q2QG_notPartial::overestimateDiff(double z, double m2dip, int) {

  double preFac    = symmetryFactor() * gaugeFactor();
  double kappaMin4 = pow4(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac
                     *2. * (1.-z) / ( pow2(1.-z) + kappaMin4);
  return wt;
}

// Return kernel for new splitting.
bool fsr_qcd_Q2QG_notPartial::calc(const Event& state, int) { 

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip),
    m2RadBef(splitInfo.kinematics()->m2RadBef),
    m2Rad(splitInfo.kinematics()->m2RadAft),
    m2Rec(splitInfo.kinematics()->m2Rec),
    m2Emt(splitInfo.kinematics()->m2EmtAft);
  int splitType(splitInfo.type);

  // Calculate kernel.
  // Note: We are calculating the z <--> 1-z symmetrised kernel here.
  double preFac = symmetryFactor() * gaugeFactor();
  double kappa2 = pT2/m2dip;

  map<string,double> wts;
  double wt_base_as1 = preFac * 2. / (1.-z);

  wts.insert( make_pair("base", wt_base_as1 ) );
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt_base_as1 ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp",   wt_base_as1 ));
  }

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  // Add collinear term for massless splittings.
  if (!doMassive) {
    wt_base_as1 += -preFac * ( 1.+z );
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second +=  -preFac * ( 1.+z );
  }

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
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second += preFac * massCorr;

    wt_base_as1 += preFac * massCorr;
  }

  // Store higher order correction separately.
  wts.insert( make_pair("base_order_as2", wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_G2GG_notPartial::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() == 0
        && state[ints["iRad"]].id() == 21 );
}

int fsr_qcd_G2GG_notPartial::kinMap()                 { return 1;}
int fsr_qcd_G2GG_notPartial::motherID(int)            { return 21;}
int fsr_qcd_G2GG_notPartial::sisterID(int)            { return 21;}
double fsr_qcd_G2GG_notPartial::gaugeFactor ( int, int )        { return 2.*CA;}
//double fsr_qcd_G2GG_notPartial::symmetryFactor ( int, int )     { return 0.5;}
double fsr_qcd_G2GG_notPartial::symmetryFactor ( int, int )     { return 1.0;}

int fsr_qcd_G2GG_notPartial::radBefID(int, int){ return 21;}
pair<int,int> fsr_qcd_G2GG_notPartial::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  int colRemove = (colRadAfter == acolEmtAfter)
                ? colRadAfter : acolRadAfter;
  int col       = (colRadAfter == colRemove)
                ? colEmtAfter : colRadAfter;
  int acol      = (acolRadAfter == colRemove)
                ? acolEmtAfter : acolRadAfter;
  return make_pair(col,acol);
}

vector <int> fsr_qcd_G2GG_notPartial::recPositions( const Event&, int, int) {
  return vector <int>();
}

// Pick z for new splitting.
double fsr_qcd_G2GG_notPartial::zSplit(double zMinAbs, double, double m2dip) {
  // Just pick according to soft.
  double R         = rndmPtr->flat();
  double kappaMin4 = pow4(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double p         = pow( 1. + pow2(1-zMinAbs)/kappaMin4, R );
  double res       = 1. - sqrt( p - 1. )*sqrt(kappaMin4);
  return res;
}

// New overestimates, z-integrated versions.
double fsr_qcd_G2GG_notPartial::overestimateInt(double zMinAbs, double,
  double, double m2dip, int) {

  // Overestimate by soft
  double preFac    = symmetryFactor() * gaugeFactor();
  double kappaMin4 = pow4(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * 0.5 * log( 1. + pow2(1.-zMinAbs)/kappaMin4);
  return wt;
}

// Return overestimate for new splitting.
double fsr_qcd_G2GG_notPartial::overestimateDiff(double z, double m2dip, int) {
  // Overestimate by soft
  double preFac    = symmetryFactor() * gaugeFactor();
  double kappaMin4 = pow4(settingsPtr->parm("TimeShower:pTmin"))/m2dip;
  double wt        = preFac * (1.-z) / ( pow2(1.-z) + kappaMin4);
  return wt;
}

// Return kernel for new splitting.
bool fsr_qcd_G2GG_notPartial::calc(const Event& state, int) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip),
    //m2RadBef(splitInfo.kinematics()->m2RadBef),
    m2Rad(splitInfo.kinematics()->m2RadAft),
    m2Rec(splitInfo.kinematics()->m2Rec),
    m2Emt(splitInfo.kinematics()->m2EmtAft);
  int splitType(splitInfo.type);

  double preFac = symmetryFactor() * gaugeFactor();
  double kappa2 = pT2/m2dip;

  // Calculate kernel.
  // Note: We are calculating the z <--> 1-z symmetrised kernel here.
  map<string,double> wts;
  double wt_base_as1 = preFac * 1. / (1.-z);

  wts.insert( make_pair("base", wt_base_as1 ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt_base_as1 ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt_base_as1 ));
  }

  // Correction for massive splittings.
  bool doMassive = (abs(splitType) == 2);

  // Add collinear term for massless splittings.
  if (!doMassive) {
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second += preFac * ( -1. + 0.5 * z*(1.-z) );
    wt_base_as1 += preFac * ( -1. + 0.5 * z*(1.-z) );
  }

  // Add collinear term for massive splittings.
  if (doMassive) {

    double vijk = 1.;

    // splitType == 2 -> Massive FF
    if (splitType == 2) {
      // Calculate CS variables.
      double yCS = kappa2 / (1.-z);
      double nu2Rad = m2Rad/m2dip; 
      double nu2Emt = m2Emt/m2dip; 
      double nu2Rec = m2Rec/m2dip; 
      vijk          = pow2(1.-yCS) - 4.*(yCS+nu2Rad+nu2Emt)*nu2Rec;
      vijk          = sqrt(vijk) / (1-yCS);

    // splitType ==-2 -> Massive FI
    } else if (splitType ==-2) {
      // No changes, as initial recoiler is massless!
      vijk          = 1.; 
    }

    // Add correction for massive splittings.
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second += preFac * 1./ vijk * ( -1. + 0.5 * z*(1.-z) );

    wt_base_as1 += preFac * 1./ vijk * ( -1. + 0.5 * z*(1.-z) );
  }

  // Store higher order correction separately.
  wts.insert( make_pair("base_order_as2", wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================

// Class inheriting from SplittingQCD class.

// SplittingQCD function G->QQ (FSR)

// Return true if this kernel should partake in the evolution.
bool fsr_qcd_G2QQ_notPartial::canRadiate ( const Event& state, map<string,int> ints,
  map<string,bool>, Settings*, PartonSystems*, BeamParticle*) {
  return ( state[ints["iRad"]].isFinal()
        && state[ints["iRec"]].colType() == 0
        && state[ints["iRad"]].id() == 21 );
}

int fsr_qcd_G2QQ_notPartial::kinMap()                 { return 1;}
int fsr_qcd_G2QQ_notPartial::motherID(int)            { return 1;} // Use 1 as dummy variable.
int fsr_qcd_G2QQ_notPartial::sisterID(int)            { return 1;} // Use 1 as dummy variable.
double fsr_qcd_G2QQ_notPartial::gaugeFactor ( int, int )        { return NF_qcd_fsr*TR;}
double fsr_qcd_G2QQ_notPartial::symmetryFactor ( int, int )     { return 1.0;}

int fsr_qcd_G2QQ_notPartial::radBefID(int, int){ return 21;}
pair<int,int> fsr_qcd_G2QQ_notPartial::radBefCols(
  int colRadAfter, int acolRadAfter, 
  int colEmtAfter, int acolEmtAfter) {
  int col  = (colRadAfter  > 0) ? colRadAfter  : colEmtAfter;
  int acol = (acolRadAfter > 0) ? acolRadAfter : acolEmtAfter;
  return make_pair(col,acol);
}

vector <int> fsr_qcd_G2QQ_notPartial::recPositions( const Event&, int, int) {
  return vector<int>();
}

// Pick z for new splitting.
double fsr_qcd_G2QQ_notPartial::zSplit(double zMinAbs, double zMaxAbs, double) {
  return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
}

// New overestimates, z-integrated versions.
double fsr_qcd_G2QQ_notPartial::overestimateInt(double zMinAbs,double zMaxAbs,
  double, double, int) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt            = 2.*preFac * 0.5 * ( zMaxAbs - zMinAbs);
  return wt;
}

// Return overestimate for new splitting.
double fsr_qcd_G2QQ_notPartial::overestimateDiff(double, double, int) {
  double wt     = 0.;
  double preFac = symmetryFactor() * gaugeFactor();
  wt            = 2.*preFac * 0.5;
  return wt;
}

// Return kernel for new splitting.
bool fsr_qcd_G2QQ_notPartial::calc(const Event& state, int) {

  // Dummy statement to avoid compiler warnings.
  if (false) cout << state[0].e() << endl;

  // Read all splitting variables.
  double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
    m2dip(splitInfo.kinematics()->m2Dip),
    m2Rad(splitInfo.kinematics()->m2RadAft),
    m2Rec(splitInfo.kinematics()->m2Rec),
    m2Emt(splitInfo.kinematics()->m2EmtAft);
  int splitType(splitInfo.type);

  double preFac = symmetryFactor() * gaugeFactor();
  double kappa2 = pT2/m2dip;

  map<string,double> wts;
  double wt_base_as1 = preFac * ( pow(1.-z,2.) + pow(z,2.) );
  wts.insert( make_pair("base", wt_base_as1 ));
  if (doVariations) {
    // Create muR-variations.
    if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
      wts.insert( make_pair("Variations:muRfsrDown", wt_base_as1 ));
    if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
      wts.insert( make_pair("Variations:muRfsrUp", wt_base_as1 ));
  }

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
    for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it)
      it->second =  preFac * 1. / vijk * ( pow2(1.-z) + pow2(z)
                                         + m2Emt / ( pipj + m2Emt) );

    wt_base_as1 =  preFac * 1. / vijk * ( pow2(1.-z) + pow2(z)
                                        + m2Emt / ( pipj + m2Emt) );
  }

  // Store higher order correction separately.
  wts.insert( make_pair("base_order_as2", wts["base"] - wt_base_as1 ));

  // Store kernel values.
  clearKernels();
  for ( map<string,double>::iterator it = wts.begin(); it != wts.end(); ++it )
    kernelVals.insert(make_pair( it->first, it->second ));

  return true;

}

//==========================================================================



} // end namespace Pythia8
