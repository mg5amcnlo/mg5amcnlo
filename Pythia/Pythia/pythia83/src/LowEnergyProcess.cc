// LowEnergyProcess.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the LowEnergyProcess
// class.

#include "Pythia8/LowEnergyProcess.h"

namespace Pythia8 {

//==========================================================================

// LowEnergyProcess class.
// This class handles low-energy collisions between two hadrons.

//--------------------------------------------------------------------------

// Constants: could be changed here if desired, but normally should not.
// These are of technical nature, as described for each.

// Maximum number of tries to split beam particles before reconnection.
static constexpr int MAXLOOP = 100;

// Gradually reduce assumed quark masses from their constituent values.
static constexpr double MASSREDUCERATE = 0.025;

// Parameters for diffractive mass spectrum, as in the SaS parametrization.
static constexpr double MDIFFMIN = 0.28;
static constexpr double CRES  = 2.0;
static constexpr double MRES0 = 1.062;

// The pion mass; used to check whether there is room for one more hadron.
static constexpr double MPI = 0.14;

// K mass; used to check if eta can split into ssbar.
static constexpr double MK = 0.498;

// Diquark-antidiquark system need extra mass excess for string handling.
static constexpr double MEXTRADIQDIQ = 0.5;

// Pomeron trajectory alpha(t) = 1 + epsilon + alpha' * t.
static constexpr double ALPHAPRIME = 0.25;

//--------------------------------------------------------------------------

// Initialize the LowEnergyProcess class as required.

void LowEnergyProcess::init( StringFlav* flavSelPtrIn,
  StringFragmentation* stringFragPtrIn,
  MiniStringFragmentation* ministringFragPtrIn,
  SigmaLowEnergy* sigmaLowEnergyPtrIn,
  NucleonExcitations* nucleonExcitationsPtrIn) {

  // Save pointers.
  flavSelPtr        = flavSelPtrIn;
  stringFragPtr     = stringFragPtrIn;
  ministringFragPtr = ministringFragPtrIn;
  sigmaLowEnergyPtr = sigmaLowEnergyPtrIn,
  nucleonExcitationsPtr = nucleonExcitationsPtrIn;

  // Relative fraction of s quark production in strin breaks.
  probStoUD       = parm("StringFlav:probStoUD");

  // Mixing for eta and eta'.
  double theta    = parm("StringFlav:thetaPS");
  double alpha    = (theta + 54.7) * M_PI / 180.;
  fracEtass       = pow2(sin(alpha));
  fracEtaPss      = 1. - fracEtass;

  // Longitudinal momentum sharing of valence quarks in hadrons.
  xPowMes         = parm("BeamRemnants:valencePowerMeson");
  xPowBar         = 0.5 * ( parm("BeamRemnants:valencePowerUinP")
                          + parm("BeamRemnants:valencePowerDinP") );
  xDiqEnhance     = parm("BeamRemnants:valenceDiqEnhance");

  // Transverse momentum spread.
  sigmaQ          = parm("StringPT:sigma") / sqrt(2.);

  // Boundary mass between string and ministring handling.
  mStringMin      = parm("HadronLevel:mStringMin");

  // Proton mass used as reference scale in diffraction.
  sProton         = pow2(particleDataPtr->m0(2212));

  // Probability of double annihilation when flavours allow.
  probDoubleAnn   = parm("LowEnergyQCD:probDoubleAnnihilation");

  // Initialize collision event record.
  leEvent.init( "(low energy event)", particleDataPtr);

  // Done.
  isInit = true;
}

//--------------------------------------------------------------------------

// Produce outgoing primary hadrons from collision of incoming pair.
// type | 1: nondiff; | 2 : el; | 3: SD (XB); | 4: SD (AX);
//      | 5: DD;  | 6: CD (AXB, not implemented)
//      | 7: excitation | 8: annihilation | 9: resonant
//      | >100: resonant through the specified resonance particle

bool LowEnergyProcess::collide( int i1, int i2, int typeIn, Event& event,
  Vec4 vtx, Vec4 vtx1, Vec4 vtx2) {

  // Check that class is initialized and that incoming are hadrons.
  if (!isInit) {
    infoPtr->errorMsg("Error in LowEnergyProcess::collide: "
      "not initialized!");
    return false;
  }
  if (!event[i1].isHadron() || !event[i2].isHadron()) return false;
  sizeOld = event.size();

  // Store information about incoming particles.
  type      = typeIn;
  id1       = event[i1].id();
  id2       = event[i2].id();
  isBaryon1 = ( (abs(id1)/1000)%10 > 0 );
  isBaryon2 = ( (abs(id2)/1000)%10 > 0 );
  m1        = event[i1].m();
  m2        = event[i2].m();
  eCM       = (event[i1].p() + event[i2].p()).mCalc();
  sCM       = eCM * eCM;

  // Pick K0/K0bar combination if both particles are K_S/K_L.
  if ((id1 == 310 || id1 == 130) && (id2 == 310 || id2 == 130)) {
    double sigmaSame = sigmaLowEnergyPtr->sigmaPartial(311,  311, eCM,
      m1, m2, type);
    double sigmaMix  = sigmaLowEnergyPtr->sigmaPartial(311, -311, eCM,
      m1, m2, type);
    int choice = rndmPtr->pick({ 0.25 * sigmaSame, 0.25 * sigmaSame,
      0.50 * sigmaMix });
    if (choice == 0)      id1 = id2 = 311;
    else if (choice == 1) id1 = id2 = -311;
    else                { id1 = 311; id2 = -311; }
  }

  // Pick K0 or K0bar if either particle is K_S or K_L.
  if (id1 == 310 || id1 == 130) {
    double sigmaK    = sigmaLowEnergyPtr->sigmaPartial( 311, id2, eCM,
      m1, m2, type);
    double sigmaKbar = sigmaLowEnergyPtr->sigmaPartial(-311, id2, eCM,
      m1, m2, type);
    id1 = (rndmPtr->pick({ sigmaK, sigmaKbar }) == 0) ? 311 : -311;
  }
  else if (id2 == 310 || id2 == 130) {
    double sigmaK    = sigmaLowEnergyPtr->sigmaPartial(id1,  311, eCM,
      m1, m2, type);
    double sigmaKbar = sigmaLowEnergyPtr->sigmaPartial(id1, -311, eCM,
      m1, m2, type);
    id2 = (rndmPtr->pick({ sigmaK, sigmaKbar }) == 0) ? 311 : -311;
  }

  // Reset leEvent event record. Add incoming hadrons as beams in rest frame.
  leEvent.reset();
  leEvent.append( event[i1]);
  leEvent.append( event[i2]);
  leEvent[1].status( -12);
  leEvent[2].status( -12);
  RotBstMatrix MtoCM = toCMframe( leEvent[1].p(), leEvent[2].p());
  leEvent.rotbst( MtoCM);

  // Get code from type; the same except for resonant process.
  int code;
  if (type >= 1 && type <= 8 && type != 6) code = type;
  else if (abs(type) > 100) code = 9;
  else {
    infoPtr->errorMsg( "Error in LowEnergyProcess::collide: "
      "invalid process type", std::to_string(type));
    return false;
  }

  // Perform collision specified by code.
  if      (code == 1) { if (!nondiff())      return false; }
  else if (code <= 5) { if (!eldiff())       return false; }
  else if (code == 7) { if (!excitation())   return false; }
  else if (code == 8) { if (!annihilation()) return false; }
  else                { if (!resonance())    return false; }

  // Store number of direct daughters of the original colliding hadrons.
  int nPrimaryProds = leEvent.size() - 3;
  nHadron = (code == 3 || code == 5) ? 0 : 1;

  // Hadronize new strings if necessary.
  if (code == 1 || code == 3 || code == 4 || code == 5 || code == 8) {
    if (!simpleHadronization()) {
      infoPtr->errorMsg( "Error in LowEnergyProcess::collide: "
        "hadronization failed");
      return false;
    }
  }

  // Mother incides for the direct daughters.
  int mother1 = max(i1, i2), mother2 = min(i1, i2);

  // Offset between position in low energy event and position in full event.
  int indexOffset = sizeOld - 3;

  // Copy particles into regular event record.
  for (int i = 3; i < leEvent.size(); ++i) {
    int iNew = event.append(leEvent[i]);

    // For direct daughters, set mothers to the original particles.
    if ( leEvent[i].mother1() == 1 || leEvent[i].mother1() == 2
      || leEvent[i].mother2() == 1 || leEvent[i].mother2() == 2 ) {
      event[iNew].mothers(mother1, mother2);
    }
    // For subsequent daughters, use offset indices.
    else {
      event[iNew].mother1(indexOffset + leEvent[i].mother1());
      event[iNew].mother2(indexOffset + leEvent[i].mother2());
    }

    // Set status, lifetime if final and daughters if not.
    if (event[iNew].isFinal()) {
      event[iNew].status(150 + code);
      if (event[iNew].isHadron())
        event[iNew].tau( event[iNew].tau0() * rndmPtr->exp() );
    } else {
      event[iNew].status(-(150 + code));
      event[iNew].daughter1(indexOffset + leEvent[i].daughter1());
      event[iNew].daughter2(indexOffset + leEvent[i].daughter2());
    }
  }

  // Set daughters for original particles.
  event[i1].daughters(sizeOld, sizeOld + nPrimaryProds - 1);
  event[i2].daughters(sizeOld, sizeOld + nPrimaryProds - 1);

  // Boost particles from subcollision frame to full event frame.
  RotBstMatrix MfromCM = fromCMframe( event[i1].p(), event[i2].p());
  if (code == 1 || code > 7) {
    for (int i = sizeOld; i < event.size(); ++i) {
      event[i].rotbst( MfromCM);
      event[i].vProdAdd( vtx);
    }

  // Special case for t-channel processes with displaced production vertices.
  // nHadron & nParton is number in first system, i.e. where to switch.
  } else {
    int iHadron = 0;
    int nParton = (code == 3 || code == 5) ? 2 : 0;
    int iParton = 0;
    for (int i = sizeOld; i < event.size(); ++i) {
      event[i].rotbst( MfromCM);
      if (event[i].status() > 0)
           event[i].vProdAdd( (++iHadron <= nHadron) ? vtx1 : vtx2 );
      else event[i].vProdAdd( (++iParton <= nParton) ? vtx1 : vtx2 );
    }
  }

  // Mark incoming colliding hadrons as decayed.
  event[i1].statusNeg();
  event[i2].statusNeg();

  // Done.
  return true;

}

//--------------------------------------------------------------------------

// Do an inelastic nondiffractive scattering.

bool LowEnergyProcess::nondiff() {

  // Resolve flavours and check minimum new hadron masses.
  pair< int, int>  paircac1  = splitFlav( id1 );
  idc1   = paircac1.first;
  idac1  = paircac1.second;
  pair< int, int>  paircac2  = splitFlav( id2 );
  idc2   = paircac2.first;
  idac2  = paircac2.second;
  mThr1  = mThreshold( idc1, idac2);
  mThr2  = mThreshold( idc2, idac1);

  // Special two-body handling if below three-body threshold.
  if (eCM < mThr1 + mThr2 + MPI) return twoBody();

  // Special three-body handling if below four-body threshold.
  if (eCM < mThr1 + mThr2 + 2. * MPI) return threeBody();

  // Check that not stuck in infinite loop. Allow reduced quark masses.
  int    loop = 0;
  double e1, pz1, epz1, pzc1, ec1, epz2, pzc2, ec2, mAbove1, mAbove2;
  Vec4   pc1, pac1, pc2, pac2;
  do {
    do {
      if (++loop == MAXLOOP) return threeBody();
      double redStep = (loop < 10) ? 1. : exp( -MASSREDUCERATE * (loop - 9));

      // Split up hadrons A  and B into q + qbar or q + qq for meson/baryon.
      if (!splitA( eCM, redStep)) continue;
      if (!splitB( eCM, redStep)) continue;

      // Assign relative sharing of longitudinal momentum.
      z1     = splitZ( idc1, idac1, mTc1 / eCM, mTac1 / eCM);
      z2     = splitZ( idc2, idac2, mTc2 / eCM, mTac2 / eCM);
      mT1    = sqrt( mTsc1 / z1 + mTsac1 / (1. - z1));
      mT2    = sqrt( mTsc2 / z2 + mTsac2 / (1. - z2));

    // Ensure that hadron beam remnants are not too massive.
    } while (mT1 + mT2 > eCM);

    // Set up kinematics for outgoing beam remnants.
    e1    = 0.5 * (sCM + mT1 * mT1 - mT2 * mT2) / eCM;
    pz1   = sqrtpos(e1 * e1 - mT1 * mT1);
    epz1  = z1 * (e1 + pz1);
    pzc1  = 0.5 * (epz1 - mTsc1 / epz1 );
    ec1   = 0.5 * (epz1 + mTsc1 / epz1 );
    pc1.p(   px1,  py1,       pzc1,      ec1 );
    pac1.p( -px1, -py1, pz1 - pzc1, e1 - ec1 );
    epz2  = z2 * (eCM - e1 + pz1);
    pzc2  = -0.5 * (epz2 - mTsc2 / epz2 );
    ec2   =  0.5 * (epz2 + mTsc2 / epz2 );
    pc2.p(   px2,  py2,        pzc2,            ec2 );
    pac2.p( -px2, -py2, -pz1 - pzc2, eCM - e1 - ec2 );

    // Catch reconnected systems with too small masses.
    mAbove1 = (pc1 + pac2).mCalc() - mThreshold( idc1, idac2);
    mAbove2 = (pc2 + pac1).mCalc() - mThreshold( idc2, idac1);
  } while ( max( mAbove1, mAbove2) < MPI || min( mAbove1, mAbove2) < 0. );

  // Store new reconnected string systems; lowest excess first.
  if (mAbove1 < mAbove2) {
    leEvent.append(  idc1, 63, 1, 0, 0, 0, 101,   0,  pc1,  mc1);
    leEvent.append( idac2, 63, 2, 0, 0, 0,   0, 101, pac2, mac2);
    leEvent.append(  idc2, 63, 2, 0, 0, 0, 102,   0,  pc2,  mc2);
    leEvent.append( idac1, 63, 1, 0, 0, 0,   0, 102, pac1, mac1);
  } else {
    leEvent.append(  idc2, 63, 2, 0, 0, 0, 102,   0,  pc2,  mc2);
    leEvent.append( idac1, 63, 1, 0, 0, 0,   0, 102, pac1, mac1);
    leEvent.append(  idc1, 63, 1, 0, 0, 0, 101,   0,  pc1,  mc1);
    leEvent.append( idac2, 63, 2, 0, 0, 0,   0, 101, pac2, mac2);
  }

  // Done.
  return true;

}

//--------------------------------------------------------------------------

// Do an elastic or diffractive scattering.

bool LowEnergyProcess::eldiff() {

  // Classify process type.
  bool excite1 = (type == 3 || type == 5);
  bool excite2 = (type == 4 || type == 5);

  // Check if low-mass diffraction partly covered by excitation processes.
  bool hasExcitation = sigmaLowEnergyPtr->hasExcitation( id1, id2);

  // Find excited mass ranges.
  mA           = m1;
  mB           = m2;
  double mAmin = (excite1) ? mDiffThr(id1, m1) : m1;
  double mBmin = (excite2) ? mDiffThr(id2, m2) : m2;
  double mAmax = eCM - mBmin;
  double mBmax = eCM - mAmin;
  if (mAmin + mBmin > eCM) {
    infoPtr->errorMsg("Error in LowEnergyProcess::eldiff: "
      "too low invariant mass for diffraction",
      "for " + to_string(id1) + " " + to_string(id2)
      + " with type=" + to_string(type) + " @ " + to_string(eCM) + " GeV");
    return false;
  }

  // Useful kinematics definitions. Also some for diffraction.
  double s1    = m1 * m1;
  double s2    = m2 * m2;
  double sA    = mA * mA;
  double sB    = mB * mB;
  double lam12 = sqrtpos(pow2( sCM - s1 - s2) - 4. * s1 * s2);
  double sResXB = pow2(mA + MRES0);
  double sResAX = pow2(mB + MRES0);

  // Find maximal t range.
  double sAmin = mAmin * mAmin;
  double sBmin = mBmin * mBmin;
  double lamAB = sqrtpos(pow2( sCM - sAmin - sBmin) - 4. * sAmin * sBmin);
  double tempA = sCM - (s1 + s2 + sAmin + sBmin) + (s1 - s2) * (sAmin - sBmin)
    / sCM;
  double tempB = lam12 * lamAB / sCM;
  double tLowX = -0.5 * (tempA + tempB);
  double wtA, wtB, tempC, tLow, tUpp, tNow;
  double bNow = (type == 2) ? bSlope() : 2.;

  // Outer loop over t values to be matched against masses.
  int  loopT = 0;
  bool failT = false;
  do {
    failT = false;
    if (++loopT == MAXLOOP) {
      infoPtr->errorMsg("Error in LowEnergyProcess::eldiff: "
        "failed to construct valid kinematics (t)");
      return false;
    }

    // Inner loop over masses of either or both excited beams.
    // Check that not stuck in infinite loop. Allow reduced quark masses.
    int  loopM = 0;
    bool failM = false;
    do {
      failM = false;
      if (++loopM == MAXLOOP) {
        infoPtr->errorMsg("Error in LowEnergyProcess::eldiff: "
          "failed to construct valid kinematics (m)");
        return false;
      }
      double redStep = (loopM < 10) ? 1. : exp( -MASSREDUCERATE * (loopM - 9));

      // Split up hadron 1 (on side A) and assign excited A mass.
      if (excite1) {
        do {
          mA = mAmin * pow( mAmax / mAmin, rndmPtr->flat() );
          sA = mA * mA;
          wtA = (hasExcitation) ? 1.
              : (1. + CRES * sResXB / (sResXB + sA)) / (1. + CRES);
        } while (wtA < rndmPtr->flat());
        if (!splitA( mA, redStep)) failM = true;
      }


      // Split up hadron 2 (on side B) and assign excited B mass.
      if (excite2 && !failM) {
        do {
          mB = mBmin * pow( mBmax / mBmin, rndmPtr->flat() );
          sB = mB * mB;
          wtB = (hasExcitation) ? 1.
              : (1. + CRES * sResAX / (sResAX + sB)) / (1. + CRES);
        } while (wtB < rndmPtr->flat());
        if (!splitB( mB, redStep)) failM = true;
      }

      // Ensure that pair of hadron masses not too large. Suppression at limit.
      if (mA + mB > eCM) failM = true;
      if (!failM) {
        double wtM = 1.;
        if      (type == 3) wtM = 1. - sA / sCM;
        else if (type == 4) wtM = 1. - sB / sCM;
        else if (type == 5) wtM = (1. - pow2(mA + mB) / sCM)
           * sCM * sProton / (sCM * sProton + sA * sB);
        if (wtM < rndmPtr->flat()) failM = true;
      }
    } while (failM);

    // Calculate allowed t range.
    lamAB = sqrtpos(pow2( sCM - sA - sB) - 4. * sA * sB);
    tempA = sCM - (s1 + s2 + sA + sB) + (s1 - s2) * (sA - sB) / sCM;
    tempB = lam12 *  lamAB / sCM;
    tempC = (sA - s1) * (sB - s2) + (s1 + sB - s2 - sA)
          * (s1 * sB - s2 * sA) / sCM;
    tLow  = -0.5 * (tempA + tempB);
    tUpp  = tempC / tLow;

    // Pick t in maximal range and check if within allowed range.
    if (type != 2) bNow  = bSlope();
    tNow  = log(1. - rndmPtr->flat() * (1. - exp(bNow * tLowX))) / bNow;
    if (tNow < tLow || tNow > tUpp) failT = true;
  } while (failT);

  // Energies and longitudinal momenta of excited hadrons.
  double eA    = 0.5 * (sCM + sA - sB) / eCM;
  double pzA   = sqrtpos(eA * eA - sA);
  Vec4   pA( 0., 0.,  pzA,       eA);
  Vec4   pB( 0., 0., -pzA, eCM - eA);

  // Internal kinematics on side A, boost to CM frame and store constituents.
  if (excite1) {
    double ec1   = 0.5 * (sA + mTsc1 - mTsac1) / mA;
    double pzc1  = sqrtpos(ec1 * ec1 - mTsc1);
    // Diquark always forward. Randomize for meson.
    if ( abs(idac1) > 10 || (abs(idc1) < 10 && abs(idac1) < 10
      && rndmPtr->flat() > 0.5) ) pzc1 = -pzc1;
    Vec4 pc1(   px1,  py1,  pzc1,      ec1);
    Vec4 pac1( -px1, -py1, -pzc1, mA - ec1);
    pc1.bst(pA);
    pac1.bst(pA);
    leEvent.append(  idc1, 63, 1, 0, 0, 0, 101,   0,  pc1,  mc1);
    leEvent.append( idac1, 63, 1, 0, 0, 0,   0, 101, pac1, mac1);

  // Simple copy if not excited, and set momentum as in collision frame.
  } else {
    int iNew = leEvent.copy( 1, 63);
    leEvent[iNew].p( pA);
    leEvent[iNew].vProd( 0., 0., 0., 0.);
  }

  // Internal kinematics on side B, boost to CM frame and store constituents.
  if (excite2) {
    double ec2   = 0.5 * (sB + mTsc2 - mTsac2) / mB;
    double pzc2  = -sqrtpos(ec2 * ec2 - mTsc2);
    // Diquark always forward (on negative side). Randomize for meson.
    if ( abs(idac2) > 10 || (abs(idc2) < 10 && abs(idac2) < 10
      && rndmPtr->flat() > 0.5) ) pzc2 = -pzc2;
    Vec4 pc2(   px2,  py2,  pzc2,      ec2);
    Vec4 pac2( -px2, -py2, -pzc2, mB - ec2);
    pc2.bst(pB);
    pac2.bst(pB);
    leEvent.append(  idc2, 63, 2, 0, 0, 0, 102,   0,  pc2,  mc2);
    leEvent.append( idac2, 63, 2, 0, 0, 0, 0,   102, pac2, mac2);

  // Simple copy if not excited, and set momentum as in collision frame.
  } else {
    int iNew = leEvent.copy( 2, 63);
    leEvent[iNew].p( pB);
    leEvent[iNew].vProd( 0., 0., 0., 0.);
  }

  // Reconstruct theta angle and rotate outgoing particles accordingly.
  double cosTheta = min(1., max(-1., (tempA + 2. * tNow) / tempB));
  double sinTheta = 2. * sqrtpos( -(tempC + tempA * tNow + tNow * tNow) )
    / tempB;
  double theta = asin( min(1., sinTheta));
  if (cosTheta < 0.) theta = M_PI - theta;
  if (!std::isfinite(theta)) {
    infoPtr->errorMsg("Error in LowEnergyProcess::eldiff: "
      "t is not finite");
    return false;
  }
  double phi      = 2. * M_PI * rndmPtr->flat();
  for (int i = 3; i < leEvent.size(); ++i) leEvent[i].rot( theta, phi);

  // Done.
  return true;

}

//-------------------------------------------------------------------------

// Do an excitation collision.

bool LowEnergyProcess::excitation() {

  // Generate excited hadrons and masses.
  int idA, idB;
  if (!nucleonExcitationsPtr->pickExcitation(id1, id2, eCM, idA, mA, idB, mB))
    return false;

  // Calculate allowed t range.
  double s1    = m1 * m1;
  double s2    = m2 * m2;
  double sA    = mA * mA;
  double sB    = mB * mB;
  double lam12 = sqrtpos(pow2( sCM - s1 - s2) - 4. * s1 * s2);
  double lamAB = sqrtpos(pow2( sCM - sA - sB) - 4. * sA * sB);
  double tempA = sCM - (s1 + s2 + sA + sB) + (s1 - s2) * (sA - sB) / sCM;
  double tempB = lam12 *  lamAB / sCM;
  double tempC = (sA - s1) * (sB - s2) + (s1 + sB - s2 - sA)
               * (s1 * sB - s2 * sA) / sCM;
  double tLow  = -0.5 * (tempA + tempB);
  double tUpp  = tempC / tLow;

  // Find t slope as in diffraction and calculate t.
  int typeSave = type;
  if (idA == id1 && idB == id2) type = 2;
  else if (idB == id2)          type = 3;
  else if (idA == id1)          type = 4;
  else                          type = 5;
  double bNow = bSlope();
  type = typeSave;
  double tNow  = tUpp + log(1. - rndmPtr->flat()
    * (1. - exp(bNow * (tLow - tUpp)))) / bNow;

  // Set up kinematics along the +- z direction.
  double eA    = 0.5 * (sCM + sA - sB) / eCM;
  double pzA   = sqrtpos(eA * eA - sA);
  Vec4   pA( 0., 0.,  pzA,       eA);
  Vec4   pB( 0., 0., -pzA, eCM - eA);
  int iA = leEvent.append(idA, 157, 1,2, 0,0, 0,0, pA, mA);
  int iB = leEvent.append(idB, 157, 1,2, 0,0, 0,0, pB, mB);

  // Rotate suitably,
  double cosTheta = min(1., max(-1., (tempA + 2. * tNow) / tempB));
  double sinTheta = 2. * sqrtpos( -(tempC + tempA * tNow + tNow * tNow) )
    / tempB;
  double theta = asin( min(1., sinTheta));
  if (cosTheta < 0.) theta = M_PI - theta;
  double phi      = 2. * M_PI * rndmPtr->flat();
  leEvent[iA].rot( theta, phi);
  leEvent[iB].rot( theta, phi);

  // Done.
  return true;
}

//--------------------------------------------------------------------------

// Do an annihilation collision of a baryon-antibaryon pair.

bool LowEnergyProcess::annihilation() {

  // Check that indeed baryon-antibaryon collision.
  if (!isBaryon1 || !isBaryon2
    || (id1 > 0 ? 1 : -1) * (id2 > 0 ? 1 : -1) > 0) {
    infoPtr->errorMsg( "Error in LowEnergyProcess::annihilation: "
      "not a baryon-antibaryon incoming pair",
      std::to_string(id1) + " + " + std::to_string(id2));
    return false;
  }

  // Working areas.
  int iqAll[2][10];
  vector<int> iqPair;

  // Split first and second hadron by flavour content.
  for (int iHad = 0; iHad < 2; ++iHad) {
    int idAbs = (iHad == 0) ? abs(id1) : abs(id2);
    iqAll[iHad][0] = (idAbs/1000)%10;
    iqAll[iHad][1] = (idAbs/100)%10;
    iqAll[iHad][2] = (idAbs/10)%10;
  }

  // Find potential annihilating quark-antiquark pairs.
  for (int i1 = 0; i1 < 3; ++i1)
  for (int i2 = 0; i2 < 3; ++i2)
    if (iqAll[1][i2] == iqAll[0][i1]) iqPair.push_back(10 * i1 + i2);

  // Return if no annihilation possible.
  if (iqPair.size() == 0) {
    infoPtr->errorMsg( "Error in LowEnergyProcess::annihilation: "
      "flavour content does not allow annihilation");
    return false;
  }

  // Annihilate one quark-antiquark pair at random among options.
  int iAnn = max( 0, min( int(iqPair.size()) - 1,
    int(iqPair.size() * rndmPtr->flat()) ));
  iqAll[0][iqPair[iAnn]/10] = iqAll[0][2];
  iqAll[1][iqPair[iAnn]%10] = iqAll[1][2];

  // Check if second annihilation is possible and wanted.
  iqPair.clear();
  for (int i1 = 0; i1 < 2; ++i1)
  for (int i2 = 0; i2 < 2; ++i2)
    if (iqAll[1][i2] == iqAll[0][i1]) iqPair.push_back(10 * i1 + i2);

  // Annihilate second pair if possible and chosen.
  if ( (iqPair.size() > 0) && (rndmPtr->flat() < probDoubleAnn) ) {
    iAnn = max( 0, min( int(iqPair.size()) - 1,
      int(iqPair.size() * rndmPtr->flat()) ));
    iqAll[0][iqPair[iAnn]/10] = iqAll[0][1];
    iqAll[1][iqPair[iAnn]%10] = iqAll[1][1];

    // Extract leftover partons and their masses. Scale down if masses too big.
    int id1Ann   = (id1 > 0) ? iqAll[0][0] : -iqAll[0][0];
    int id2Ann   = (id2 > 0) ? iqAll[1][0] : -iqAll[1][0];
    double m1Ann = particleDataPtr->m0( id1Ann);
    double m2Ann = particleDataPtr->m0( id2Ann);
    if (m1Ann + m2Ann > 0.8 * eCM) {
      double scaledown = 0.8 * eCM / (m1Ann + m2Ann);
      m1Ann *= scaledown;
      m2Ann *= scaledown;
    }

    // Set up kinematics and done for two annihilations.
    double e1Ann = 0.5 * (sCM + m1Ann*m1Ann - m2Ann*m2Ann) / eCM;
    double pzAnn = sqrtpos(e1Ann*e1Ann - m1Ann*m1Ann);
    Vec4 p1Ann( 0., 0.,  pzAnn,       e1Ann );
    Vec4 p2Ann( 0., 0., -pzAnn, eCM - e1Ann );
    int col1  = (id1 > 0) ? 101 : 0;
    int acol1 = (id1 > 0) ? 0 : 101;
    leEvent.append( id1Ann, 63, 1, 2, 0, 0, col1, acol1, p1Ann, m1Ann);
    leEvent.append( id2Ann, 63, 1, 2, 0, 0, acol1, col1, p2Ann, m2Ann);
    return true;
  }

  // Begin handling of two strings/pairs. Labels as if each hadron remnant
  // is a colour plus an anticolour quark, so as to reuse nondiffractive code.
  idc1  = (id1 > 0) ? iqAll[0][0] : -iqAll[0][0];
  idac1 = (id1 > 0) ? iqAll[0][1] : -iqAll[0][1] ;
  idc2  = (id2 > 0) ? iqAll[1][0] : -iqAll[1][0];
  idac2 = (id2 > 0) ? iqAll[1][1] : -iqAll[1][1] ;
  if (rndmPtr->flat() < 0.5) swap(idc2, idac2);

  // Check that not stuck in infinite loop. Allow reduced quark masses.
  int    loop = 0;
  double e1, pz1, epz1, pzc1, ec1, epz2, pzc2, ec2, mAbove1, mAbove2;
  Vec4   pc1, pac1, pc2, pac2;
  do {
    do {
      if (++loop == MAXLOOP) {
        infoPtr->errorMsg( "Error in LowEnergyProcess::annihilation: "
          "failed to find working kinematics configuration");
        return false;
      }
      double redStep = (loop < 10) ? 1. : exp( -MASSREDUCERATE * (loop - 9));

      // Split up hadrons A  and B by relative pT.
      if (!splitA( eCM, redStep, false)) continue;
      if (!splitB( eCM, redStep, false)) continue;

      // Assign relative sharing of longitudinal momentum.
      z1     = splitZ( idc1, idac1, mTc1 / eCM, mTac1 / eCM);
      z2     = splitZ( idc2, idac2, mTc2 / eCM, mTac2 / eCM);
      mT1    = sqrt( mTsc1 / z1 + mTsac1 / (1. - z1));
      mT2    = sqrt( mTsc2 / z2 + mTsac2 / (1. - z2));

    // Ensure that hadron beam remnants are not too massive.
    } while (mT1 + mT2 > eCM);

    // Set up kinematics for outgoing beam remnants.
    e1    = 0.5 * (sCM + mT1 * mT1 - mT2 * mT2) / eCM;
    pz1   = sqrtpos(e1 * e1 - mT1 * mT1);
    epz1  = z1 * (e1 + pz1);
    pzc1  = 0.5 * (epz1 - mTsc1 / epz1 );
    ec1   = 0.5 * (epz1 + mTsc1 / epz1 );
    pc1.p(   px1,  py1,       pzc1,      ec1 );
    pac1.p( -px1, -py1, pz1 - pzc1, e1 - ec1 );
    epz2  = z2 * (eCM - e1 + pz1);
    pzc2  = -0.5 * (epz2 - mTsc2 / epz2 );
    ec2   =  0.5 * (epz2 + mTsc2 / epz2 );
    pc2.p(   px2,  py2,        pzc2,            ec2 );
    pac2.p( -px2, -py2, -pz1 - pzc2, eCM - e1 - ec2 );

    // Catch reconnected systems with too small masses.
    mAbove1 = (pc1 + pac2).mCalc() - mThreshold( idc1, idac2);
    mAbove2 = (pc2 + pac1).mCalc() - mThreshold( idc2, idac1);
  } while ( max( mAbove1, mAbove2) < MPI || min( mAbove1, mAbove2) < 0. );

  // Store new reconnected string systems; lowest excess first.
  int col1  = (id1 > 0) ? 101 : 0;
  int acol1 = (id1 > 0) ? 0 : 101;
  int col2  = (id1 > 0) ? 102 : 0;
  int acol2 = (id1 > 0) ? 0 : 102;
  if (mAbove1 < mAbove2) {
    leEvent.append(  idc1, 63, 1, 0, 0, 0,  col1, acol1,  pc1,  mc1);
    leEvent.append( idac2, 63, 2, 0, 0, 0, acol1,  col1, pac2, mac2);
    leEvent.append( idac1, 63, 1, 0, 0, 0,  col2, acol2, pac1, mac1);
    leEvent.append(  idc2, 63, 2, 0, 0, 0, acol2,  col2,  pc2,  mc2);
  } else {
    leEvent.append( idac1, 63, 1, 0, 0, 0,  col2, acol2, pac1, mac1);
    leEvent.append(  idc2, 63, 2, 0, 0, 0, acol2,  col2,  pc2,  mc2);
    leEvent.append(  idc1, 63, 1, 0, 0, 0,  col1, acol1,  pc1,  mc1);
    leEvent.append( idac2, 63, 2, 0, 0, 0, acol1,  col1, pac2, mac2);
  }

  // Done.
  return true;

}

//-------------------------------------------------------------------------

// Do a resonance formation and decay.

bool LowEnergyProcess::resonance() {

  // Create the resonance
  int iNew = leEvent.append(type, 919, 1,2,0,0, 0,0, Vec4(0,0,0,eCM), eCM);

  leEvent[1].statusNeg(); leEvent[1].daughters(iNew, 0);
  leEvent[2].statusNeg(); leEvent[2].daughters(iNew, 0);

  return true;
}

//--------------------------------------------------------------------------

// Simple version of hadronization for low-energy hadronic collisions.
// Only accepts simple q-qbar systems and hadrons.

bool LowEnergyProcess::simpleHadronization() {

  // Find the complete colour singlet configuration of the event.
  simpleColConfig.clear();
  bool fixOrder = (type == 1);
  for (int i = 0; i < leEvent.size(); ++i)
  if (leEvent[i].isQuark() || leEvent[i].isDiquark()) {
    vector<int> qqPair;
    qqPair.push_back(   i);
    qqPair.push_back( ++i);
    simpleColConfig.simpleInsert( qqPair, leEvent, fixOrder);
  }

  // If no quarks are present, the system is already hadronized.
  if (simpleColConfig.size() == 0) return true;

  // Process all colour singlet (sub)systems.
  leEvent.saveSize();
  int nHadronBeg = leEvent.size();
  for (int iSub = 0; iSub < simpleColConfig.size(); ++iSub) {
    if (iSub == 1) nHadron = leEvent.size() - nHadronBeg;

    // Diquark-antidiquark system needs higher mass to count as full string.
    double mExcess = simpleColConfig[iSub].massExcess;
    double mDiqDiq = (  leEvent[simpleColConfig[iSub].iParton[0]].isDiquark()
      && leEvent[simpleColConfig[iSub].iParton[1]].isDiquark() )
      ? MEXTRADIQDIQ : 0.;
    bool fragDone = false;

    // String fragmentation of each colour singlet (sub)system.
    if ( mExcess > mStringMin + mDiqDiq) {
      fragDone = stringFragPtr->fragment( iSub, simpleColConfig, leEvent);
      if (!fragDone && mExcess > mStringMin + mDiqDiq + 4. * MPI) return false;
    }

    // Low-mass string treated separately. Tell if diffractive system.
    if (!fragDone) {
      bool isDiff = (type >= 3 && type <= 5);
      if ( !ministringFragPtr->fragment( iSub, simpleColConfig, leEvent,
        isDiff, false) ) return false;
    }
  }

  // If elastic try last time to find three-body inelastic (or two-body).
  int nHad = 0, id3 = 0, id4 = 0;
  for (int i = 1; i < leEvent.size(); ++i) if (leEvent[i].isFinal()) {
    ++nHad;
    if (nHad == 1) id3 = leEvent[i].id();
    if (nHad == 2) id4 = leEvent[i].id();
  }
  if (type == 1 && nHad == 2 && ( (id3 == id1 && id4 == id2)
    || (id3 == id2 && id4 == id1) )) {
    leEvent.restoreSize();
    return threeBody();
  }

  // Done.
  return true;

}

//--------------------------------------------------------------------------

// Special two-body handling if below three-body threshold or in emergency.

bool LowEnergyProcess::twoBody() {

  // Often impossible to rearrange baryon-antibaryon flavours.
  if ( (abs(idc1) > 10 && abs(idac2) > 10)
    || (abs(idc2) > 10 && abs(idac1) > 10) ) swap(idac1, idac2);

  // Combine to hadrons
  int idH1   = flavSelPtr->combineToLightest( idc1, idac2);
  int idH2   = flavSelPtr->combineToLightest( idc2, idac1);

  // Get masses
  double mH1, mH2;
  if ( (particleDataPtr->mMin(idH1) + particleDataPtr->mMin(idH2) >= eCM)
    || !hadronWidthsPtr->pickMasses(idH1, idH2, eCM, mH1, mH2)) {
    infoPtr->errorMsg("Warning in LowEnergyProcess::twoBody: "
      "below mass threshold, defaulting to elastic collision");
    idH1 = id1;
    idH2 = id2;
    mH1  = leEvent[1].m();
    mH2  = leEvent[2].m();
  }

  // Phase space. Fill particles into the event record and done.
  pair<Vec4, Vec4> ps12 = rndmPtr->phaseSpace2(eCM, mH1, mH2);
  for (int i = 3; i < leEvent.size(); ++i) leEvent[i].statusNeg();
  leEvent.append( idH1, 111, 2, 1, 0, 0, 0, 0, ps12.first, mH1);
  leEvent.append( idH2, 111, 2, 1, 0, 0, 0, 0, ps12.second, mH2);

  // Done.
  return true;

}

//--------------------------------------------------------------------------

// Special three-body handling if below four-body threshold or in emergency.

bool LowEnergyProcess::threeBody() {

  // Impossible to rearrange baryon-antibaryon flavours.
  if ( (abs(idc1) > 10 && abs(idac2) > 10)
    || (abs(idc2) > 10 && abs(idac1) > 10) ) swap(idac1, idac2);

  // Try to find new flavour choice a couple of time.
  int    idc3, idH1, idH2, idH3;
  double mH1, mH2, mH3;
  for (int iTry = 0; iTry < 10; ++iTry) {
    idc3 = (rndmPtr->flat() < 0.5) ? 1 : 2;
    if (iTry < 8 && rndmPtr->flat() < 0.5) {
      idH1 = flavSelPtr->combineToLightest( idc1, -idc3);
      idH2 = flavSelPtr->combineToLightest( idc3, idac2);
      idH3 = flavSelPtr->combineToLightest( idc2, idac1);
    } else if (iTry < 8) {
      idH1 = flavSelPtr->combineToLightest( idc1, idac2);
      idH2 = flavSelPtr->combineToLightest( idc2, -idc3);
      idH3 = flavSelPtr->combineToLightest( idc3, idac1);
    } else {
      idH1 = flavSelPtr->combineToLightest( idc1, idac2);
      idH2 = flavSelPtr->combineToLightest( idc2, idac1);
      idH3 = 111;
    }

    // Check if sufficient energy, else go for two-body.
    mH1  = particleDataPtr->mSel( idH1);
    mH2  = particleDataPtr->mSel( idH2);
    mH3  = particleDataPtr->mSel( idH3);
    if (mH1 + mH2 + mH3 < eCM) break;
    else if (iTry == 9) return twoBody();
  }

  // Kinematical limits for 2+3 mass. Maximum phase-space weight.
  double m23Min  = mH2 + mH3;
  double m23Max  = eCM - mH1;
  double p1Max   = 0.5 * sqrtpos( (eCM - mH1 - m23Min) * (eCM + mH1 + m23Min)
    * (eCM + mH1 - m23Min) * (eCM - mH1 + m23Min) ) / eCM;
  double p23Max  = 0.5 * sqrtpos( (m23Max - mH2 - mH3) * (m23Max + mH2 + mH3)
    * (m23Max + mH2 - mH3) * (m23Max - mH2 + mH3) ) / m23Max;
  double wtPSmax = 0.5 * p1Max * p23Max;

  // Pick an intermediate mass m23 flat in the allowed range.
  double wtPS, m23, p1Abs, p23Abs;
  do {
    m23    = m23Min + rndmPtr->flat() * (m23Max - m23Min);

    // Translate into relative momenta and find phase-space weight.
    p1Abs  = 0.5 * sqrtpos( (eCM - mH1 - m23) * (eCM + mH1 + m23)
      * (eCM + mH1 - m23) * (eCM - mH1 + m23) ) / eCM;
    p23Abs = 0.5 * sqrtpos( (m23 - mH2 - mH3) * (m23 + mH2 + mH3)
      * (m23 + mH2 - mH3) * (m23 - mH2 + mH3) ) / m23;
    wtPS   = p1Abs * p23Abs;

  // If rejected, try again with new invariant masses.
  } while ( wtPS < rndmPtr->flat() * wtPSmax );

  // Set up particle momenta.
  pair<Vec4, Vec4> ps123 = rndmPtr->phaseSpace2(eCM, mH1, m23);
  Vec4 p1 = ps123.first;
  pair<Vec4, Vec4> ps23 = rndmPtr->phaseSpace2(m23, mH2, mH3);
  Vec4 p2 = ps23.first;
  Vec4 p3 = ps23.second;
  p2.bst( ps123.second, m23 );
  p3.bst( ps123.second, m23 );

  // Fill particles into the event record and done.
  for (int i = 3; i < leEvent.size(); ++i) leEvent[i].statusNeg();
  leEvent.append( idH1, 111, 2, 1, 0, 0, 0, 0, p1, mH1);
  leEvent.append( idH2, 111, 2, 1, 0, 0, 0, 0, p2, mH2);
  leEvent.append( idH3, 111, 2, 1, 0, 0, 0, 0, p3, mH3);

  // Done.
  return true;

}

//-------------------------------------------------------------------------

// Split up hadron A into a colour-anticolour pair, with masses and pT values.

bool LowEnergyProcess::splitA( double mMax, double redMpT, bool splitFlavour) {

  // Split up flavour of hadron into a colour and an anticolour constituent.
  if (splitFlavour) {
    pair< int, int>  paircac  = splitFlav( id1 );
    idc1   = paircac.first;
    idac1  = paircac.second;
  }
  if (idc1 == 0 || idac1 == 0) return false;

  // Allow a few tries to find acceptable internal kinematics.
  for (int i = 0; i < 10; ++i) {

    // Find constituent masses and scale down to less than full mass.
    mc1    = particleDataPtr->m0( idc1);
    mac1   = particleDataPtr->m0( idac1);
    double redNow = redMpT * min( 1., m1 / (mc1 + mac1));
    mc1   *= redNow;
    mac1  *= redNow;

    // Select Gaussian relative transverse momenta for constituents.
    pair<double, double> gauss2 = rndmPtr->gauss2();
    px1    = redMpT * sigmaQ * gauss2.first;
    py1    = redMpT * sigmaQ * gauss2.second;
    pTs1   = px1 * px1 + py1 * py1;

    // Construct transverse masses.
    mTsc1  = pow2(mc1)  + pTs1;
    mTsac1 = pow2(mac1) + pTs1;
    mTc1   = sqrt(mTsc1);
    mTac1  = sqrt(mTsac1);

    // Check if solution found, else failed.
    if (mTc1 + mTac1 < mMax) return true;
  }
  return false;

}

//-------------------------------------------------------------------------

// Split up hadron B into a colour-anticolour pair, with masses and pT values.

bool LowEnergyProcess::splitB( double mMax, double redMpT, bool splitFlavour) {

  // Split up flavour of hadron into a colour and an anticolour constituent.
  if (splitFlavour) {
    pair< int, int>  paircac  = splitFlav( id2 );
    idc2   = paircac.first;
    idac2  = paircac.second;
  }
  if (idc2 == 0 || idac2 == 0) return false;

  // Allow a few tries to find acceptable internal kinematics.
  for (int i = 0; i < 10; ++i) {

    // Find constituent masses and scale down to less than full mass.
    mc2    = particleDataPtr->m0( idc2);
    mac2   = particleDataPtr->m0( idac2);
    double redNow = redMpT * min( 1., m2 / (mc2 + mac2));
    mc2   *= redNow;
    mac2  *= redNow;

    // Select Gaussian relative transverse momenta for constituents.
    pair<double, double> gauss2 = rndmPtr->gauss2();
    px2    = redMpT * sigmaQ * gauss2.first;
    py2    = redMpT * sigmaQ * gauss2.second;
    pTs2   = px2 * px2 + py2 * py2;

    // Construct transverse masses.
    mTsc2  = pow2(mc2)  + pTs2;
    mTsac2 = pow2(mac2) + pTs2;
    mTc2   = sqrt(mTsc2);
    mTac2  = sqrt(mTsac2);

    // Check if solution found, else failed.
    if (mTc2 + mTac2 < mMax) return true;
  }
  return false;

}

//-------------------------------------------------------------------------

// Split up a hadron into a colour and an anticolour part, of q or qq kinds.

pair< int, int> LowEnergyProcess::splitFlav( int id) {

  // Hadron flavour content.
  int idAbs = abs(id);
  int iq1   = (idAbs/1000)%10;
  int iq2   = (idAbs/100)%10;
  int iq3   = (idAbs/10)%10;
  int iq4, iq5;

  // Nondiagonal mesons.
  if (iq1 == 0 && iq2 != iq3) {
    if (id != 130 && id != 310) {
      if (iq2%2 == 1) swap( iq2, iq3);
      if (id > 0) return make_pair( iq2, -iq3);
      else        return make_pair( iq3, -iq2);

    // K0S and K0L are mixes d sbar and dbar s.
    } else {
      if (rndmPtr->flat() < 0.5) return make_pair( 3, -1);
      else                       return make_pair( 1, -3);
    }

  // Diagonal mesons: assume complete mixing ddbar and uubar.
  } else if (iq1 == 0) {
    iq4 = iq2;
    // Special cases for 11x, 22x, and eta'
    if (iq2 < 3 || id == 331) {
      iq4 = (rndmPtr->flat() < 0.5) ? 1 : 2;
      // eta and eta' can also be s sbar.
      if (id == 221 && eCM > 2 * MK && rndmPtr->flat() < fracEtass) iq4 = 3;
      if (id == 331 && eCM > 2 * MK && rndmPtr->flat() < fracEtaPss) iq4 = 3;
    }
    return make_pair( iq4, -iq4);

  // Octet baryons.
  } else if (idAbs%10 == 2) {
    // Three identical quarks: emergency in case of higher spin 1/2 multiplet.
    if (iq1 == iq2 && iq2 == iq3) {iq4 = iq1; iq5 = 1100 * iq1 + 3;}
    // Two identical quarks, like normal p or n.
    else if (iq1 == iq2 || iq2 == iq3) {
      double rr6 = 6. * rndmPtr->flat();
      if    (iq1 == iq2 && rr6 < 2.) { iq4 = iq3; iq5 = 1100 * iq1 + 3;}
      else if             (rr6 < 2.) { iq4 = iq1; iq5 = 1100 * iq3 + 3;}
      else if (rr6 < 3.) { iq4 = iq2; iq5 = 1000 * iq1 + 100 * iq3 + 3;}
      else               { iq4 = iq2; iq5 = 1000 * iq1 + 100 * iq3 + 1;}
    // Three nonidentical quarks, Sigma- or Lambda-like.
    } else {
      int isp = (iq2 > iq3) ? 3 : 1;
      if (iq3 > iq1) swap( iq1, iq3);
      if (iq3 > iq2) swap( iq2, iq3);
      double rr12 = 12. * rndmPtr->flat();
      if      (rr12 < 4.) { iq4 = iq1; iq5 = 1000 * iq2 + 100 * iq3 + isp;}
      else if (rr12 < 5.) { iq4 = iq2; iq5 = 1000 * iq1 + 100 * iq3 + isp;}
      else if (rr12 < 6.) { iq4 = iq3; iq5 = 1000 * iq1 + 100 * iq2 + isp;}
      else if (rr12 < 9.) { iq4 = iq2; iq5 = 1000 * iq1 + 100 * iq3 + 4 - isp;}
      else                { iq4 = iq3; iq5 = 1000 * iq1 + 100 * iq2 + 4 - isp;}
    }
    return (id > 0) ? make_pair(iq4, iq5) : make_pair(-iq5, -iq4);

  // Higher spin baryons.
  } else {
    double rr3 = 3. * rndmPtr->flat();
    // Sort quark order, e.g. for Lambdas.
    if (iq3 > iq1) swap( iq1, iq3);
    if (iq3 > iq2) swap( iq2, iq3);
    if (rr3 < 1.)      { iq4 = iq1; iq5 = 1000 * iq2 + 100 * iq3 + 3;}
    else if (rr3 < 2.) { iq4 = iq2; iq5 = 1000 * iq1 + 100 * iq3 + 3;}
    else               { iq4 = iq3; iq5 = 1000 * iq1 + 100 * iq2 + 3;}
    return (id > 0) ? make_pair(iq4, iq5) : make_pair(-iq5, -iq4);
  }

  // Done. (Fake call to avoid unwarranted compiler warning.)
  return make_pair( 0, 0);
}

//-------------------------------------------------------------------------

// Find relative momentum of colour and anticolour constituents in hadron.

double LowEnergyProcess::splitZ(int iq1, int iq2, double mRat1, double mRat2) {

  // Initial values.
  if (mRat1 + mRat2 >= 1.) return mRat1 / ( mRat1 + mRat2);
  int iq1Abs = abs(iq1);
  int iq2Abs = abs(iq2);
  if (iq2Abs > 10) swap( mRat1, mRat2);
  double x1, x2, x1a, x1b;

  // Handle mesons.
  if (iq1Abs < 10 && iq2Abs < 10) {
    do x1 = pow2( mRat1 + (1. - mRat1) * rndmPtr->flat() );
    while ( pow(1. - x1, xPowMes) < rndmPtr->flat() );
    do x2 = pow2( mRat2 + (1. - mRat2) * rndmPtr->flat() );
    while ( pow(1. - x2, xPowMes) < rndmPtr->flat() );

  // Handle baryons.
  } else {
    double mRat1ab = 0.5 * mRat1 / xDiqEnhance;
    do x1a = pow2( mRat1ab + (1. - mRat1ab) * rndmPtr->flat() );
    while ( pow(1. - x1a, xPowBar) < rndmPtr->flat() );
    do x1b = pow2( mRat1ab + (1. - mRat1ab) * rndmPtr->flat() );
    while ( pow(1. - x1b, xPowBar) < rndmPtr->flat() );
    x1 = xDiqEnhance * ( x1a + x1b);
    do x2 = pow2( mRat2 + (1. - mRat2) * rndmPtr->flat() );
    while ( pow(1. - x2, xPowBar) < rndmPtr->flat() );
    if (iq2Abs > 10) swap( x1, x2);
  }

  // Return z value.
  return x1 / (x1 + x2);

}

//-------------------------------------------------------------------------

// Overestimate mass of lightest state for given flavour combination.

double LowEnergyProcess::mThreshold( int iq1, int iq2) {

  // Initial values.
  int iq1Abs = abs(iq1);
  int iq2Abs = abs(iq2);
  if (iq2Abs > 10) swap( iq1Abs, iq2Abs);
  double mThr = 0.;

  // Mesonic or baryonic state.
  if (iq2Abs < 10) mThr
     = particleDataPtr->m0( flavSelPtr->combineToLightest ( iq1, iq2) );

  // Baryon-antibaryon state.
  else mThr = min(
      particleDataPtr->m0( flavSelPtr->combineToLightest ( iq1Abs, 1) )
    + particleDataPtr->m0( flavSelPtr->combineToLightest ( iq2Abs, 1) ),
      particleDataPtr->m0( flavSelPtr->combineToLightest ( iq1Abs, 2) )
    + particleDataPtr->m0( flavSelPtr->combineToLightest ( iq2Abs, 2) ) );

  // Done.
  return mThr;

}

//-------------------------------------------------------------------------

// Minimum mass required for diffraction into two hadrons.
// Note that splitFlav is not deterministic, so neither is mDiffThr.

double LowEnergyProcess::mDiffThr( int idNow, double mNow) {

  // Initial minimal value.
  double mThr = mNow + MDIFFMIN;

  // Split up hadron into color and anticolour.
  pair< int, int>  paircac  = splitFlav( idNow );
  int idcNow  = paircac.first;
  int idacNow = paircac.second;
  if (idcNow == 0 || idacNow == 0) return mThr;
  if (idNow == 221 || idNow == 331) {idcNow = 3; idacNow = -3;}

  // Insert u-ubar or d-dbar pair to find lowest two-body state.
  double mThr2body = min(
      particleDataPtr->m0( flavSelPtr->combineToLightest ( idcNow, -1) )
    + particleDataPtr->m0( flavSelPtr->combineToLightest ( 1, idacNow) ),
      particleDataPtr->m0( flavSelPtr->combineToLightest ( idcNow, -2) )
    + particleDataPtr->m0( flavSelPtr->combineToLightest ( 2, idacNow) ) );

  // Done.
  return max(mThr, mThr2body);

}

//-------------------------------------------------------------------------

// Pick slope b of exp(b * t) for elastic and diffractive events.

double LowEnergyProcess::bSlope() {

  // Steeper slope for baryons than mesons.
  // Scale by AQM factor for strange, charm and bottom.
  if (id1 != id1sv) {
    bA = sigmaLowEnergyPtr->nqEffAQM(id1) * ((isBaryon1) ? 2.3/3. : 1.4/2.);
    id1sv = id1;
  }
  if (id2 != id2sv) {
    bB = sigmaLowEnergyPtr->nqEffAQM(id2) * ((isBaryon1) ? 2.3/3. : 1.4/2.);
    id2sv = id2;
  }

  // Elastic slope.
  if (type == 2)
    return 2. * bA + 2. * bB + 2. * ALPHAPRIME * log(ALPHAPRIME * sCM);

  // Single diffractive slope for XB and AX, respectively.
  if (type == 3) return 2. * bB + 2. * ALPHAPRIME * log(sCM / (mA * mA));
  if (type == 4) return 2. * bA + 2. * ALPHAPRIME * log(sCM / (mB * mB));

  // Double diffractive slope.
  return 2. * ALPHAPRIME * log(exp(4.) + sCM / (ALPHAPRIME * pow2(mA * mB)) );

}

//==========================================================================

} // end namespace Pythia8
