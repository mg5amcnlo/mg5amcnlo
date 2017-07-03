// GammaKinematics.cc is a part of the PYTHIA event generator.
// Copyright (C) 2017 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the GammaKinematics
// class.

#include "Pythia8/GammaKinematics.h"

namespace Pythia8 {

//==========================================================================

// The GammaKinematics class.
// Generates the kinematics of emitted photons according to phase space limits.

//--------------------------------------------------------------------------

// Initialize phase space limits.

bool GammaKinematics::init(Info* infoPtrIn, Settings* settingsPtrIn,
  Rndm* rndmPtrIn, BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn){

  // Store input pointers for future use.
  infoPtr       = infoPtrIn;
  settingsPtr   = settingsPtrIn;
  rndmPtr       = rndmPtrIn;
  beamAPtr      = beamAPtrIn;
  beamBPtr      = beamBPtrIn;

  // Save the applied cuts.
  Q2maxGamma    = settingsPtr->parm("Photon:Q2max");
  Wmin          = settingsPtr->parm("Photon:Wmin");
  Wmax          = settingsPtr->parm("Photon:Wmax");
  theta1Max     = settingsPtr->parm("Photon:thetaAMax");
  theta2Max     = settingsPtr->parm("Photon:thetaBMax");

  // Direct or resolved photons.
  gammaMode     = settingsPtr->mode("Photon:ProcessType");

  // Check if photons from both beams or only from one beam.
  sample1Gamma  = false;
  sample2Gamma  = false;
  sideGammaA    = true;
  if ( beamAPtr->isLepton() && beamBPtr->isLepton() ) {
    sample2Gamma  = true;
  } else if ( (beamAPtr->isLepton() && beamBPtr->isHadron())
           || (beamBPtr->isLepton() && beamAPtr->isHadron()) ) {
    sample1Gamma  = true;
    if ( beamAPtr->isHadron() ) sideGammaA = false;
  }

  // Get the masses and collision energy and derive useful ratios.
  eCM           = infoPtr->eCM();
  sCM           = pow2( eCM);
  m2BeamA       = pow2( beamAPtr->m() );
  m2BeamB       = pow2( beamBPtr->m() );
  sHatNew       = 0.;

  // Calculate the CM-energies of incoming beams.
  eCM2A = 0.25 * pow2( sCM + m2BeamA - m2BeamB ) / sCM;
  eCM2B = 0.25 * pow2( sCM - m2BeamA + m2BeamB ) / sCM;

  // If Wmax below Wmin (negative by default) use the total invariant mass.
  if ( Wmax < Wmin ) Wmax = eCM;

  // Done.
  return true;
}

//--------------------------------------------------------------------------

// Sample kinematics of one or two photon beams from the original beams.

bool GammaKinematics::sampleKTgamma(){

  // Get the sampled x_gamma values from beams.
  xGamma1 = beamAPtr->xGamma();
  xGamma2 = beamBPtr->xGamma();

  // Two photons from the beams.
  if ( sample2Gamma) {

    // Sample kinematics of the first photon.
    if ( !sampleKin(xGamma1, m2BeamA, eCM2A) ) return false;
    Q2min1   = Q2min;
    Q2gamma1 = Q2gamma;
    kT1      = kT;
    kz1      = kz;
    phi1     = phi;
    theta1   = theta;

    // Reject kinematics if a scattering angle above cut.
    if ( theta1Max > 0 && ( theta1 > theta1Max ) ) return false;

    // Sample kinematics of the second photon.
    if ( !sampleKin(xGamma2, m2BeamB, eCM2B) ) return false;
    Q2min2   = Q2min;
    Q2gamma2 = Q2gamma;
    kT2      = kT;
    kz2      = kz;
    phi2     = phi;
    theta2   = theta;

    // Reject kinematics if a scattering angle above cut.
    if ( theta2Max > 0 && ( theta2 > theta2Max ) ) return false;

    // Derive the invariant mass and check the kinematics limits.
    double cosPhi12 = cos(phi1 - phi2);
    m2GmGm = 2. * sqrt(eCM2A * eCM2B) * xGamma1 * xGamma2 - Q2gamma1 - Q2gamma2
           + 2. * kz1 * kz2 - 2. * kT1 * kT2 * cosPhi12;

    // Check if derived value within bounds set by user.
    if ( ( m2GmGm < pow2(Wmin) ) || ( m2GmGm > pow2(Wmax) ) ) return false;

    // Calculate invariant mass now that the square is positive.
    mGmGm = sqrt(m2GmGm);

    return true;

  // One photon from the beams.
  } else if ( sample1Gamma) {
    if ( sideGammaA) {
      if ( !sampleKin(xGamma1, m2BeamA, eCM2A) ) return false;
      Q2min1   = Q2min;
      Q2gamma1 = Q2gamma;
      kT1      = kT;
      kz1      = kz;
      phi1     = phi;
      theta1   = theta;
    } else {
      if ( !sampleKin(xGamma2, m2BeamB, eCM2B) ) return false;
      Q2min2   = Q2min;
      Q2gamma2 = Q2gamma;
      kT2      = kT;
      kz2      = kz;
      phi2     = phi;
      theta2   = theta;
    }

    // Derive the invariant mass and check the limits.
    // Solve the longitudinal momentum of beam particles in CM frame.
    double pz2 = ( pow2(sCM - m2BeamA - m2BeamB) - 4. * m2BeamA * m2BeamB )
               * 0.25 / sCM;
    double pz  = sqrtpos( pz2);

    // Pick the correct beam mass and x_gamma.
    double m2Beam = sideGammaA ? m2BeamB : m2BeamA;
    double xGamma = sideGammaA ? xGamma1 : xGamma2;

    // Calculate the invariant mass of the photon-hadron pair and check limits.
    m2GmGm     = m2Beam - Q2gamma + 2. * ( xGamma * sqrt(eCM2A) * sqrt(eCM2B)
               + kz * pz );
    if ( ( m2GmGm < pow2(Wmin) ) || ( m2GmGm > pow2(Wmax) ) ) return false;
    mGmGm      = sqrt(m2GmGm);

    return true;
  }

  else return false;

}

//--------------------------------------------------------------------------

// Sample the Q2 values and phi angles for each beam and derive kT according
// to sampled x_gamma. Check that sampled values within required limits.

bool GammaKinematics::sampleKin(double xGamma, double m2Beam, double eCM2) {

  // Check if allowed x_gamma. May fail for direct processes.
  double m2s = 4. * m2Beam / sCM;
  double xGamMax = Q2maxGamma / (2. * m2Beam)
    * ( sqrt( (1. + 4. * m2Beam / Q2maxGamma) * (1. - m2s) ) - 1. );
  if ( xGamma > xGamMax ) return false;

  // Derive the Q2 limit.
  Q2min      = 2. * m2Beam * pow2(xGamma) / ( 1. - xGamma - m2s
             + sqrt(1. - m2s) * sqrt( pow2(1. - xGamma) - m2s ) );

  // Sample Q2_gamma value for the beam.
  Q2gamma = Q2min * pow( Q2maxGamma / Q2min, rndmPtr->flat() );

  // Sample azimuthal angle from flat [0,2*pi[.
  phi = 2. * M_PI * rndmPtr->flat();

  // Calculate kT^2 for photon from particle with non-zero mass.
  double kT2gamma = ( ( 1. - xGamma - 0.25 * Q2gamma / eCM2 ) * Q2gamma
    - m2Beam * ( Q2gamma / eCM2 + pow2(xGamma) ) ) / (1.- m2Beam / eCM2);

  // Check that physical values for kT (very rarely fails if ever but may
  // cause numerical issues).
  if ( kT2gamma < 0. ) {
    infoPtr->errorMsg("Error in gammaKinematics::sampleKTgamma: "
                      "unphysical kT value.");
    return false;
  }

  // Calculate the transverse and longitudinal momemta and scattering angle
  // of the beam particle.
  kT = sqrt( kT2gamma );
  theta = atan( sqrt( eCM2* ( Q2gamma * ( 1. - xGamma )
        - m2Beam * pow2(xGamma) ) - m2Beam * Q2gamma - pow2( 0.5 * Q2gamma) )
        / ( eCM2 * ( 1. - xGamma) - m2Beam - 0.5 * Q2gamma ) );
  kz = (xGamma * eCM2 + 0.5 * Q2gamma) / ( sqrt(eCM2 - m2Beam) );

  return true;
}

//--------------------------------------------------------------------------

// Calculates the new sHat for direct-direct and direct-resolved processes.

double GammaKinematics::calcNewSHat(double sHatOld){

  // Need to recalculate only if two photons.
  if ( sample2Gamma) {

    // When mixing contributions determine the mode from updated beam modes.
    if (gammaMode == 0) {
      int beamAmode = beamAPtr->getGammaMode();
      int beamBmode = beamBPtr->getGammaMode();

      // Direct-direct case.
      if ( (beamAmode == 2) && (beamBmode == 2) ) sHatNew = m2GmGm;
      else if ( ( (beamAmode == 1) && (beamBmode == 2) )
                || ( (beamAmode == 2) && (beamBmode == 1) ) )
        sHatNew = sHatOld * m2GmGm / ( xGamma1 * xGamma2 * sCM);
      else sHatNew = 0.;
    }

    // Else use the gammaMode.
    if      (gammaMode == 4) sHatNew = m2GmGm;
    else if (gammaMode == 2 || gammaMode == 3)
    sHatNew = sHatOld * m2GmGm / ( xGamma1 * xGamma2 * sCM);
  }

  // Otherwise no need for a new value.
  else sHatNew = sHatOld;

  return sHatNew;
}

//--------------------------------------------------------------------------

// Save the accepted values for further use.

bool GammaKinematics::finalize(){

  // Propagate the sampled values for beam particles.
  beamAPtr->newGammaKTPhi(kT1, phi1);
  beamBPtr->newGammaKTPhi(kT2, phi2);
  beamAPtr->Q2Gamma(Q2gamma1);
  beamBPtr->Q2Gamma(Q2gamma2);

  // Set the sampled values also to Info object.
  infoPtr->setQ2Gamma1(Q2gamma1);
  infoPtr->setQ2Gamma2(Q2gamma2);
  infoPtr->setX1Gamma(xGamma1);
  infoPtr->setX2Gamma(xGamma2);
  infoPtr->setTheta1(theta1);
  infoPtr->setTheta2(theta2);
  infoPtr->setECMsub(mGmGm);
  infoPtr->setsHatNew(sHatNew);

  // Done.
  return true;
}

//==========================================================================

} // end namespace Pythia8
