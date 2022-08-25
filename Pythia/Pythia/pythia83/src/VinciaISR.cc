// VinciaISR.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the VinciaISR
// class and auxiliary classes.

#include "Pythia8/VinciaFSR.h"
#include "Pythia8/VinciaISR.h"

namespace Pythia8 {

using namespace VinciaConstants;

//==========================================================================

// Base class for initial-state trial generators.

//--------------------------------------------------------------------------

// Initialize pointers.

void TrialGeneratorISR::initPtr(Info* infoPtrIn) {
  infoPtr     = infoPtrIn;
  rndmPtr     = infoPtr->rndmPtr;
  settingsPtr = infoPtr->settingsPtr;
}

//--------------------------------------------------------------------------

// Initialize.

void TrialGeneratorISR::init(double mcIn, double mbIn) {

  TINYPDFtrial     = 1.e-10;
  // TODO: this version of VINCIA uses PT evolution for all branchings.
  useMevolSav      = false;
  // s for hadron hadron.
  shhSav           = infoPtr->s();
  // Number of active quark flavours.
  nGtoQISRSav      = settingsPtr->mode("Vincia:nGluonToQuark");
  // For conversion trial generators.
  if (!settingsPtr->flag("Vincia:convertGluonToQuark")) nGtoQISRSav = 0;
  trialFlavSav     = 0;
  // Masses.
  mbSav            = mbIn;
  mcSav            = mcIn;
  // Sector shower.
  sectorShower     = settingsPtr->flag("Vincia:sectorShower");
  // Saved trialPDF ratio.
  trialPDFratioSav = 1.0;
  verbose          = settingsPtr->mode("Vincia:Verbose");
  isInit           = true;

}

//--------------------------------------------------------------------------

// Trial antenna function.

double TrialGeneratorISR::aTrial(double saj, double sjb, double sAB) {
  if (saj < 0. || sjb < 0.) return 0.;
  const double sab = sAB + saj + sjb;
  const double ant = 2*pow2(sab)/saj/sjb/sAB;
  const double xFactor = sab/sAB;
  return xFactor * ant;
}

//--------------------------------------------------------------------------

// Generate a new Q value, with first-order running alphaS.

double TrialGeneratorISR::genQ2run(double q2, double sAB,
  double zMin, double zMax, double colFac, double PDFratio,
  double b0, double kR, double Lambda, double, double,
  double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAB < 0. || q2 < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac,1.0);

  // Constants.
  double Iz     = getIz(zMin,zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI*b0/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double facLam = pow2(Lambda/kR);

  // Generate new scale.
  double ran    = rndmPtr->flat();
  return exp(pow(ran,comFac) * log(q2/facLam)) * facLam;

}

//--------------------------------------------------------------------------

// Generate a new Q value, with constant trial alphaS.

double TrialGeneratorISR::genQ2(double q2old, double sAB, double zMin,
  double zMax, double colFac, double alphaS, double PDFratio,
  double, double, double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAB < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac,1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin,zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  return q2old * pow(ran,comFac/alphaS);

}

//--------------------------------------------------------------------------

// Generate new Q value, with running of the PDFs towards the mass
// threshold.

double TrialGeneratorISR::genQ2thres(double, double, double,
  double, double, double, double, int, int, double, double, bool, double,
  double) {return 0.0;}

//--------------------------------------------------------------------------

// Generate a new zeta value in [zMin,zMax].

double TrialGeneratorISR::genZ(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return -1.;
  double ran = rndmPtr->flat();
  double invZ = 1.0 + (1.0-zMin)/zMin*pow(zMin*(1.0-zMax)/zMax/(1.0-zMin),ran);
  return 1.0/invZ;
}

//--------------------------------------------------------------------------

// The zeta integral.

double TrialGeneratorISR::getIz(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return 0.0;
  return log(zMax*(1.0-zMin)/zMin/(1.0-zMax));
}

//--------------------------------------------------------------------------

// The zeta boundaries, for a given value of the evolution variable.

double TrialGeneratorISR::getZmin(double Qt2, double sAB, double, double) {
  // Update in case of beam spread.
  shhSav = infoPtr->s();
  double rootArg = pow2(shhSav - sAB) - 4.*Qt2*shhSav;
  double root = (rootArg < NANO) ? 0. : sqrt(rootArg);
  return (shhSav - sAB - root)/(2.*shhSav);
}

double TrialGeneratorISR::getZmax(double Qt2, double sAB, double, double) {
  // Update in case of beam spread.
  shhSav = infoPtr->s();
  double rootArg = pow2(shhSav - sAB) - 4.*Qt2*shhSav;
  double root = (rootArg < NANO) ? 0. : sqrt(rootArg);
  return (shhSav - sAB + root)/(2.*shhSav);
}

//--------------------------------------------------------------------------

// Inverse transforms to obtain saj and sjb from Qt2 and zeta.

double TrialGeneratorISR::getS1j(double Qt2, double zeta, double sAB) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getSj2(Qt2, -zeta, sAB);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": Unphysical input");
    return 0.;
  }
  return Qt2/zeta;
}

double TrialGeneratorISR::getSj2(double Qt2, double zeta, double sAB) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getS1j(Qt2,-zeta,sAB);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": Unphysical input");
    return 0.;
  }
  return (Qt2 + sAB*zeta)/(1.0 - zeta);
}

//--------------------------------------------------------------------------

// Compute trial PDF ratio.

double TrialGeneratorISR::trialPDFratio(BeamParticle*, BeamParticle*,
  int, int, int, double, double, double, double) {
  trialPDFratioSav = 1.0;
  return trialPDFratioSav;
}

//--------------------------------------------------------------------------

// Check initialization.

bool TrialGeneratorISR::checkInit() {
  if (isInit) return true;
  infoPtr->errorMsg("Error in "+__METHOD_NAME__+": Not initialized");
  return false;
}

//==========================================================================

// A collinear trial function for initial-initial.

//--------------------------------------------------------------------------

// Trial antenna function.

double TrialIIGCollA::aTrial(double saj, double sjb, double sAB) {
  if (saj < 0. || sjb < 0.) return 0.;
  const double sab = sAB + saj + sjb;
  const double ant = 2.0 * pow2(sab/sAB)/saj;
  return ant;
}

//--------------------------------------------------------------------------

// Generate a new Q value, with first-order running alphaS.

double TrialIIGCollA::genQ2run(double q2old, double sAB,
  double zMin, double zMax, double colFac, double PDFratio,
  double b0, double kR, double Lambda, double, double,
  double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAB < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI*b0/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  double facLam = pow2(Lambda/kR);
  return exp(pow(ran, comFac) * log(q2old/facLam)) * facLam;

}

//--------------------------------------------------------------------------

// Generate a new Q value, with constant trial alphaS.

double TrialIIGCollA::genQ2(double q2old, double sAB,
  double zMin, double zMax, double colFac, double alphaS, double PDFratio,
  double, double, double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAB < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  return q2old * pow(ran, comFac/alphaS);

}

//--------------------------------------------------------------------------

// Generate a new zeta value in [zMin,zMax].

double TrialIIGCollA::genZ(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return -1.;
  double ran = rndmPtr->flat();
  double z = -1.0 + (1.0+zMin)*pow((1.0+zMax)/(1.0+zMin),ran);
  return z;
}

//--------------------------------------------------------------------------

// The zeta integral.

double TrialIIGCollA::getIz(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return 0.0;
  return log((1.0+zMax)/(1.0+zMin));
}

//--------------------------------------------------------------------------

// The zeta boundaries, for a given value of the evolution variable.

double TrialIIGCollA::getZmin(double Qt2, double sAB, double, double) {
  // Update in case of beam spread.
  shhSav = infoPtr->s();
   if (((shhSav - sAB)*(shhSav - sAB) - (4.0*Qt2*shhSav))
     < NANO ) return 0.5*(shhSav - sAB)/sAB;
  double sajm = 0.5*(shhSav - sAB - sqrt((shhSav - sAB)*(shhSav - sAB) -
    (4.0*Qt2*shhSav)));
  return sajm/sAB;
}

double TrialIIGCollA::getZmax(double Qt2, double sAB, double, double) {
  // Update in case of beam spread.
  shhSav = infoPtr->s();
  if (((shhSav - sAB)*(shhSav - sAB) - (4.0*Qt2*shhSav))
      < NANO) return 0.5*(shhSav - sAB)/sAB;
  double sajp = 0.5*(shhSav - sAB + sqrt((shhSav - sAB)*(shhSav - sAB) -
    (4.0*Qt2*shhSav)));
  return sajp/sAB;
}

//--------------------------------------------------------------------------

// Inverse transforms to obtain saj and sjb from Qt2 and zeta.

double TrialIIGCollA::getS1j(double Qt2, double zeta, double sAB) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getSj2(Qt2,-zeta,sAB);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  return Qt2*(1.0+zeta)/(zeta-Qt2/sAB);

}

double TrialIIGCollA::getSj2(double Qt2, double zeta, double sAB) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getS1j(Qt2,-zeta,sAB);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  return zeta*sAB;

}

//==========================================================================

// A splitting trial function for initial-initial, q -> gqbar.

//--------------------------------------------------------------------------

// Trial antenna function.

double TrialIISplitA::aTrial(double saj, double sjb, double sAB) {
  if (saj < 0. || sjb < 0.) return 0.;
  const double sab = sAB + saj + sjb;
  const double ant = sab/saj/sAB;
  const double xFactor = sab/sAB;
  return xFactor * ant;
}

//--------------------------------------------------------------------------

// Generate a new Q value, with first-order running alphaS.

double TrialIISplitA::genQ2run(double q2old, double sAB,
  double zMin, double zMax, double colFac, double PDFratio,
  double b0, double kR, double Lambda, double, double,
  double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAB < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 4.0*M_PI*b0/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  double facLam = pow2(Lambda/kR);
  return exp(pow(ran, comFac) * log(q2old/facLam)) * facLam;

}

//--------------------------------------------------------------------------

// Generate a new Q value, with constant trial alphaS.

double TrialIISplitA::genQ2(double q2old, double sAB,
  double zMin, double zMax, double colFac, double alphaS, double PDFratio,
  double, double, double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAB < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 4.0*M_PI/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  return q2old * pow(ran, comFac/alphaS);

}

//--------------------------------------------------------------------------

// Generate a new Q value, with running of the PDFs towards the mass
// threshold.

double TrialIISplitA::genQ2thres(double q2old, double sAB, double zMin,
  double zMax, double colFac, double alphaS, double PDFratio, int idA, int,
  double, double, bool, double headroomFac, double enhanceFac) {

  // Use only if the user wants to get rid of c and b quarks and use
  // only in the right evolution window.
  double mQ = (abs(idA) == 4 ? mcSav : mbSav);

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAB < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 4.0*M_PI/Iz/colFac/alphaS/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  return (exp(pow(ran, comFac) * log(q2old/pow2(mQ))) * pow2(mQ));

}

//--------------------------------------------------------------------------

// Generate a new zeta value in [zMin,zMax].

double TrialIISplitA::genZ(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return -1.;
  double ran = rndmPtr->flat();
  if (useMevolSav) return zMin *pow(zMax/zMin, ran);
  else return (-1.0 + (1.0 + zMin)*pow((1.0 + zMax)/(1.0 + zMin), ran));
}

//--------------------------------------------------------------------------

// The zeta integral (with alpha = 0).

double TrialIISplitA::getIz(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return 0.0;
  if (useMevolSav) return log(zMax/zMin);
  else return log((1.0 + zMax)/(1.0 + zMin));
}

//--------------------------------------------------------------------------

// The zeta boundaries, for a given value of the evolution scale.

double TrialIISplitA::getZmin(double Qt2, double sAB, double, double) {
  // Update in case of beam spread.
  shhSav = infoPtr->s();
  if (useMevolSav) return (Qt2+sAB)/sAB;
  if (((shhSav - sAB)*(shhSav - sAB) - (4.0*Qt2*shhSav))
      < NANO) return 0.5*(shhSav - sAB)/sAB;
  double sajm = 0.5*(shhSav - sAB - sqrt((shhSav - sAB)*(shhSav - sAB) -
      (4.0*Qt2*shhSav)));
  return sajm/sAB;
}

double TrialIISplitA::getZmax(double Qt2, double sAB, double, double) {
  // Update in case of beam spread.
  shhSav = infoPtr->s();
  if (useMevolSav) return (shhSav/sAB);
  if (((shhSav - sAB)*(shhSav - sAB) - (4.0*Qt2*shhSav))
      < NANO) return 0.5*(shhSav - sAB)/sAB;
  double sajp = 0.5*(shhSav - sAB + sqrt((shhSav - sAB)*(shhSav - sAB) -
      (4.0*Qt2*shhSav)));
  return sajp/sAB;
}

//--------------------------------------------------------------------------

// Inverse transforms to obtain saj and sjb from Qt2 and zeta.

double TrialIISplitA::getS1j(double Qt2, double zeta, double sAB) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getSj2(Qt2, -zeta, sAB);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  double saj = Qt2*(1.0 + zeta)/(zeta - Qt2/sAB);
  if (useMevolSav) saj = Qt2;
  return saj;

}

double TrialIISplitA::getSj2(double Qt2, double zeta, double sAB) {
  // If zeta < 0, swap invariants.
  if (zeta < 0) return getS1j(Qt2, -zeta, sAB);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  double sjb = zeta*sAB;
  if (useMevolSav) sjb = sAB*(zeta - 1) - Qt2;
  return sjb;

}

//--------------------------------------------------------------------------

// Trial PDF ratio.

double TrialIISplitA::trialPDFratio(BeamParticle* beamAPtr, BeamParticle*,
  int iSys, int idA, int, double eA, double, double Qt2A, double) {
  double xA     = eA/(sqrt(shhSav)/2.0);
  double newPdf = max(beamAPtr->xfISR(iSys,  21, xA, Qt2A), TINYPDFtrial);
  double oldPdf = max(beamAPtr->xfISR(iSys, idA, xA, Qt2A), TINYPDFtrial);
  trialPDFratioSav = 1.0*newPdf/oldPdf;
  return trialPDFratioSav;
}

//==========================================================================

// A conversion trial function for initial-initial, g -> qqbar.

//--------------------------------------------------------------------------

// Trial antenna function.

double TrialIIConvA::aTrial(double saj, double sjb, double sAB) {
  if (saj < 0. || sjb < 0.) return 0.;
  const double sab = sAB + saj + sjb;
  const double ant = pow2(sab/sAB)/saj;
  return ant;
}

//--------------------------------------------------------------------------

// Generate a new Q value, with first-order running alphaS.

double TrialIIConvA::genQ2run(double q2old, double sAB,
  double zMin, double zMax, double colFac, double PDFratio,
  double b0, double kR, double Lambda, double, double,
  double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAB < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac,1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 4.0*M_PI*b0/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  double facLam = pow2(Lambda/kR);
  return exp(pow(ran, comFac) * log(q2old/facLam)) * facLam;

}

//--------------------------------------------------------------------------

// Generate a new Q value, with constant trial alphaS.

double TrialIIConvA::genQ2(double q2old, double sAB,
  double zMin, double zMax, double colFac, double alphaS, double PDFratio,
  double, double, double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAB < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 4.0*M_PI/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  return q2old * pow(ran, comFac/alphaS);

}

//--------------------------------------------------------------------------

// Generate a new zeta value in [zMin,zMax].

double TrialIIConvA::genZ(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return -1.;
  double ran = rndmPtr->flat();
  if (useMevolSav) return zMax*pow(zMin/zMax, ran);
  else return (-1.0 + (1.0 + zMin)*pow((1.0 + zMax)/(1.0 + zMin), ran));
}

//--------------------------------------------------------------------------

// The zeta integral.

double TrialIIConvA::getIz(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return 0.0;
  if (useMevolSav) return log(zMax/zMin);
  else return log((1.0 + zMax)/(1.0 + zMin));
}

//--------------------------------------------------------------------------

// The zeta boundaries, for a given value of the evolution scale.

double TrialIIConvA::getZmin(double Qt2, double sAB, double, double) {
  // Update in case of beam spread.
  shhSav = infoPtr->s();
  if (useMevolSav) return ((Qt2 + sAB)/sAB);
  if (((shhSav - sAB)*(shhSav - sAB) - (4.0*Qt2*shhSav))
      < NANO) return 0.5*(shhSav - sAB)/sAB;
  double sajm = 0.5*(shhSav - sAB - sqrt((shhSav - sAB)*(shhSav - sAB) -
      (4.0*Qt2*shhSav)));
  return sajm/sAB;
}

double TrialIIConvA::getZmax(double Qt2, double sAB, double, double) {
  // Update in case of beam spread.
  shhSav = infoPtr->s();
  if (useMevolSav) return (shhSav/sAB);
  if (((shhSav - sAB)*(shhSav - sAB) - (4.0*Qt2*shhSav))
      < NANO ) return 0.5*(shhSav - sAB)/sAB;
  double sajp = 0.5*(shhSav - sAB + sqrt((shhSav - sAB)*(shhSav - sAB) -
      (4.0*Qt2*shhSav)));
  return sajp/sAB;
}

//--------------------------------------------------------------------------

// Inverse transforms to obtain saj and sjb from Qt2 and zeta.

double TrialIIConvA::getS1j(double Qt2, double zeta, double sAB) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getSj2(Qt2, -zeta, sAB);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  double saj = Qt2*(1.0 + zeta)/(zeta - Qt2/sAB);
  if (useMevolSav) saj = Qt2;
  return saj;

}

double TrialIIConvA::getSj2(double Qt2, double zeta, double sAB) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getS1j(Qt2, -zeta, sAB);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  double sjb = zeta*sAB;
  if (useMevolSav) sjb = sAB*(zeta - 1) - Qt2;
  return sjb;

}

//--------------------------------------------------------------------------

// Trial PDF ratio.

double TrialIIConvA::trialPDFratio(BeamParticle* beamAPtr, BeamParticle*,
  int iSys, int, int, double eA, double, double Qt2A, double) {

  double xA  = eA/(sqrt(shhSav)/2.0);
  int nQuark = nGtoQISRSav;
  if (nQuark >= 4 && Qt2A <= 4.0*mcSav*mcSav) nQuark = 3;
  else if (nQuark >= 5 && Qt2A <= 4.0*mbSav*mbSav) nQuark = 4;
  double oldPdf = max(beamAPtr->xfISR(iSys, 21, xA, Qt2A),TINYPDFtrial);

  // Store trial PDF weights for later use to pick flavour.
  map<int, double> trialPdfWeight;
  double trialPdfWeightSum = 0.0;
  for (int idQ = -nQuark; idQ <= nQuark; idQ++) {
    // Skip gluon.
    if (idQ==0) continue;
    // PDF headroom and valence flavour enhancement.
    double fac = 2.0 + 0.5 * beamAPtr->nValence(idQ);
    trialPdfWeight[idQ] =
      max(fac * beamAPtr->xfISR(iSys, idQ, xA, Qt2A),TINYPDFtrial);
    trialPdfWeightSum += trialPdfWeight[idQ];
  }
  // Pick trial flavour ID and store weight for that flavour, to be
  // used in accept probability.
  double ranFlav = rndmPtr->flat() * trialPdfWeightSum;
  map<int,double>::iterator it;
  for (it = trialPdfWeight.begin(); it != trialPdfWeight.end(); ++it) {
    double newPdf = it->second;
    ranFlav -= newPdf;
    if (ranFlav < 0.) {
      trialFlavSav = it->first;
      trialPDFratioSav = newPdf/oldPdf;
      break;
    }
  }
  // Return sum over all flavours, to be used as evolution coefficient.
  return trialPdfWeightSum/oldPdf;

}

//==========================================================================

// Soft-eikonal trial function for IF (derived base class).

//--------------------------------------------------------------------------

// Trial antenna function.

double TrialIFSoft::aTrial(double saj, double sjk, double sAK) {
  if (saj < 0. || sjk < 0.) return 0.;
  const double ant = 2. * pow2(sAK + sjk)/saj/sjk/sAK;
  return ant;
}

//--------------------------------------------------------------------------

// Generate a new Q value, with first-order running alphaS.

double TrialIFSoft::genQ2run(double q2old, double sAK,
  double zMin, double zMax, double colFac, double PDFratio,
  double b0, double kR, double Lambda, double, double,
  double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI*b0/(Iz*colFac*PDFratio*headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  double facLam = pow2(Lambda/kR);
  return exp(pow(ran, comFac) * log(q2old/facLam)) * facLam;

}

//--------------------------------------------------------------------------

// Generate a new Q value, with constant trial alphaS.

double TrialIFSoft::genQ2(double q2old, double sAK,
  double zMin, double zMax, double colFac, double alphaS, double PDFratio,
  double, double, double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();

  return q2old * pow(ran, comFac/alphaS);

}

//--------------------------------------------------------------------------

// Generate a new zeta value in [zMin,zMax].

double TrialIFSoft::genZ(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return -1.;
  const double ran = rndmPtr->flat();
  const double facRan = pow( zMin * (zMax-1.) / zMax / (zMin -1.), ran );
  const double z = zMin * 1./(zMin - (zMin - 1) * facRan);
  return z;
}

//--------------------------------------------------------------------------

// The zeta integral: dzeta/zeta/(zeta-1).

double TrialIFSoft::getIz(double zMin, double zMax) {
  if (zMin >= zMax || zMin <= 1.) return 0.0;
  const double c  = (zMax - 1) * zMin / ( (zMin - 1) * zMax );
  return log(c);
}

//--------------------------------------------------------------------------

// The zeta boundaries, for a given value of the evolution scale.

double TrialIFSoft::getZmin(double Qt2, double sAK, double, double) {
  return (Qt2 + sAK)/sAK;
}

double TrialIFSoft::getZmax(double, double, double eA, double eBeamUsed) {
  const double xA     = eA/(sqrt(shhSav)/2.0);
  const double eAmax  = ( (sqrt(shhSav)/2.0) - (eBeamUsed-eA) );
  const double xAmax  = eAmax/(sqrt(shhSav)/2.0);
  return xAmax / xA;
}

//--------------------------------------------------------------------------

// Inverse transforms to obtain saj and sjk from Qt2 and zeta.

double TrialIFSoft::getS1j(double Qt2, double zeta, double sAK) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getSj2(Qt2, -zeta, sAK);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  return Qt2*zeta/(zeta - 1.0);

}

double TrialIFSoft::getSj2(double Qt2, double zeta, double sAK) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getS1j(Qt2,-zeta,sAK);

  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  return sAK*(zeta - 1.0);

}

//--------------------------------------------------------------------------

// Trial PDF ratio.

double TrialIFSoft::trialPDFratio(BeamParticle*, BeamParticle*,
  int, int, int, double, double, double, double) {
  trialPDFratioSav = 1.3;
  return trialPDFratioSav;
}

//==========================================================================

// Specialised soft-eikonal trial function for initial-final when
// initial-state parton is a valence quark.

//--------------------------------------------------------------------------

// Trial antenna function. This trial generator uses PDF <= const as
// overestimate => x-factor.

double TrialVFSoft::aTrial(double saj, double sjk, double sAK) {
  if (saj < 0. || sjk < 0.) return 0.;
  const double ant = 2. * pow2(sAK + sjk)/saj/sjk/sAK;
  const double xFactor = (sAK + sjk)/sAK;
  return xFactor * ant;
}

//--------------------------------------------------------------------------

// Generate a new zeta value in [zMin,zMax].

double TrialVFSoft::genZ(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return -1.;
  double ran = rndmPtr->flat();
  double z = 1 + (zMin - 1.) * pow( (zMax - 1.)/(zMin - 1.),ran);
  return z;
}

//--------------------------------------------------------------------------

// The zeta integral: dzeta/(zeta-1).

double TrialVFSoft::getIz(double zMin, double zMax) {
  if (zMin >= zMax || zMin <= 1.) return 0.0;
  const double c  = (zMax - 1) / (zMin - 1);
  return log(c);
}

//==========================================================================

// A gluon collinear trial function for initial-final.

//--------------------------------------------------------------------------

// Trial antenna function.

double TrialIFGCollA::aTrial(double saj, double sjk, double sAK) {
  if (saj < 0. || sjk < 0.) return 0.;
  return 2.*pow2((sAK + sjk)/sAK)/saj;
}

//--------------------------------------------------------------------------

// Generate a new Q value, with first-order running alphaS.

double TrialIFGCollA::genQ2run(double q2old, double sAK,
  double zMin, double zMax, double colFac, double PDFratio,
  double b0, double kR, double Lambda, double, double,
  double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI*b0/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  double facLam = pow2(Lambda/kR) ;
  return exp(pow(ran, comFac) * log(q2old/facLam)) * facLam;

}

//--------------------------------------------------------------------------

// Generate a new Q value, with constant trial alphaS.

double TrialIFGCollA::genQ2(double q2old, double sAK,
  double zMin, double zMax, double colFac, double alphaS, double PDFratio,
  double, double, double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac,1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  return q2old * pow(ran, comFac/alphaS);
}

//--------------------------------------------------------------------------

// Generate a new zeta value in [zMin,zMax].

double TrialIFGCollA::genZ(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return -1.;
  double ran = rndmPtr->flat();
  return zMax*pow(zMin/zMax,ran);
}

//--------------------------------------------------------------------------

// The zeta integral.

double TrialIFGCollA::getIz(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return 0.0;
  return log(zMax/zMin);
}

//--------------------------------------------------------------------------

// The zeta boundaries, for a given value of the evolution scale.

double TrialIFGCollA::getZmin(double Qt2, double sAK, double, double) {
  return (Qt2+sAK)/sAK;
}

double TrialIFGCollA::getZmax(double, double, double eA, double eBeamUsed) {
  const double xA     = eA/(sqrt(shhSav)/2.0);
  const double eAmax  = ( (sqrt(shhSav)/2.0) - (eBeamUsed - eA) );
  const double xAmax  = eAmax/(sqrt(shhSav)/2.0);
  return xAmax/xA;
}

//--------------------------------------------------------------------------

// Inverse transforms to obtain saj and sjk from Qt2 and zeta.

double TrialIFGCollA::getS1j(double Qt2, double zeta, double sAK) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getSj2(Qt2,-zeta,sAK);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  return Qt2*zeta/(zeta - 1.0);

}

double TrialIFGCollA::getSj2(double Qt2, double zeta, double sAK) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getS1j(Qt2,-zeta,sAK);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  return sAK*(zeta - 1.0);

}

//--------------------------------------------------------------------------

// Trial PDF ratio (= just a simple headroom factor).

double TrialIFGCollA::trialPDFratio(BeamParticle*, BeamParticle*,
  int, int, int, double, double, double, double) {
  trialPDFratioSav = 1.3;
  return trialPDFratioSav;
}

//==========================================================================

// K gluon collinear trial function for initial-final sector shower.

//--------------------------------------------------------------------------

// Trial antenna function.

double TrialIFGCollK::aTrial(double saj, double sjk, double sAK) {
  if (saj < 0. || sjk < 0.) return 0.;
  return 2./sAK*pow2(sAK+sjk)/sjk/(sAK+sjk-saj);
}

//--------------------------------------------------------------------------

// Generate a new Q value with first-order running alphaS.

double TrialIFGCollK::genQ2run(double q2old, double sAK,
  double zMin, double zMax, double colFac, double PDFratio,
  double b0, double kR, double Lambda, double, double,
  double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac,1.);

  // Generate new trial scale.
  double Iz     = getIz(zMin,zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI*b0/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  double facLam = pow2(Lambda/kR);
  double q2     = facLam * pow(q2old/facLam,pow(ran,comFac));

  return q2;
}

//--------------------------------------------------------------------------

// Generate a new Q value with constant trial alphaS.

double TrialIFGCollK::genQ2(double q2old, double sAK,
  double zMin, double zMax, double colFac, double alphaS, double PDFratio,
  double, double, double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac,1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin,zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  double powFac = sqrt(pow(ran,comFac/alphaS));

  return q2old * powFac;
}

//--------------------------------------------------------------------------

// Generate a new zeta value in [zMin,zMax].

double TrialIFGCollK::genZ(double zMin, double zMax) {
  if (zMin > zMax || zMin <= 0.) return -1.;
  double ran = rndmPtr->flat();
  double z = 1. - (1. - zMin) * pow((1. - zMax)/(1. - zMin),ran);
  return z;
}

//----------------------------------------------------------------------

// The zeta integral.

double TrialIFGCollK::getIz(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return 0.0;
  return log((1. - zMin) / (1. - zMax));
}

//----------------------------------------------------------------------

// The zeta boundaries, for a given value of the evolution variable.

double TrialIFGCollK::getZmax(double /*Qt2*/, double sAK, double eA,
  double) {
  // Calculate dimensionless invariants and evolution variable.
  double xA = eA/(sqrt(shhSav)/2.0);
  // Need a cutoff here, as we hit the 1-yaj singularity else.
  // This is justified as a value close yaj->1 is always in the
  // aj-collinear sector, where it will be vetoed.
  //TODO A better solution would still be nice, though.
  double Q2cut = 1.;
  return 1./(1.+xA*Q2cut/sAK);
}

double TrialIFGCollK::getZmin(double Qt2, double sAK, double eA,
  double) {
  // Calculate dimensionless invariants and evolution variable.
  double xA = eA/(sqrt(shhSav)/2.0);
  return xA/(1.-xA)*Qt2/sAK;
}

//----------------------------------------------------------------------

// Inverse transforms to obtain saj and sjk from Qt2 and zeta.

double TrialIFGCollK::getS1j(double Qt2, double zeta, double sAK) {
  // If zeta < 0, swap invariants.
  if (zeta < 0) return getSj2(Qt2,-zeta,sAK);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Eror in "+__METHOD_NAME__+
      ": s1j out of range");
    return 0.;
  }
  // Formulated in terms of dimensionless invariants.
  double yaj = zeta;
  double sjk = Qt2/zeta;
  return yaj * (sAK + sjk);
}

//----------------------------------------------------------------------

double TrialIFGCollK::getSj2(double Qt2, double zeta, double sAK) {
  // If zeta < 0, swap invariants.
  if (zeta < 0) return getS1j(Qt2,-zeta,sAK);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Eror in "+__METHOD_NAME__+
      ": sj2 out of range");
    return 0.;
  }
  double sjk = Qt2/zeta;
  return sjk;
}

//----------------------------------------------------------------------

// Trial PDF ratio.

double TrialIFGCollK::trialPDFratio(BeamParticle*, BeamParticle*, int,
  int, int, double, double, double, double) {
  trialPDFratioSav = 1.0;
  return trialPDFratioSav;
}

//==========================================================================

// A splitting trial function for initial-final, q -> gqbar.

//--------------------------------------------------------------------------

// Trial antenna function.

double TrialIFSplitA::aTrial(double saj, double sjk, double sAK) {
  if (saj < 0. || sjk < 0.) return 0.;
  return 2.0/sAK*(sAK + sjk)/saj;
}

//--------------------------------------------------------------------------

// Generate a new Q value, with first-order running alphaS.

double TrialIFSplitA::genQ2run(double q2old, double sAK,
  double zMin, double zMax, double colFac, double PDFratio,
  double b0, double kR, double Lambda, double, double,
  double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale
  double Iz     = getIz(zMin,zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI*b0/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  double facLam = pow2(Lambda/kR);
  return exp(pow(ran, comFac) * log(q2old/facLam)) * facLam;

}

//--------------------------------------------------------------------------

// Generate a new Q value, with constant trial alphaS.

double TrialIFSplitA::genQ2(double q2old, double sAK,
  double zMin, double zMax, double colFac, double alphaS, double PDFratio,
  double, double, double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin,zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  return q2old * pow(ran, comFac/alphaS);

}

//--------------------------------------------------------------------------

// Generate a new Q value, with running of the PDFs towards the mass
// threshold.

double TrialIFSplitA::genQ2thres(double q2old, double sAK,
  double zMin, double zMax, double colFac, double alphaS, double PDFratio,
  int idA, int, double, double, bool,
  double headroomFac, double enhanceFac) {

  // Use only if the user wants to get rid of c and b quarks and use
  // only in the right evolution window.
  double mQ = (abs(idA) == 4 ? mcSav : mbSav);

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 2.0*M_PI/Iz/colFac/alphaS/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  return (exp(pow(ran, comFac) * log(q2old/pow2(mQ))) * pow2(mQ));

}

//--------------------------------------------------------------------------

// Generate a new zeta value in [zMin,zMax].

double TrialIFSplitA::genZ(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return -1.;
  double ran = rndmPtr->flat();
  return pow(ran*(1./zMax - 1./zMin) + 1./zMin, -1.0);
}

//--------------------------------------------------------------------------

// The zeta integral.

double TrialIFSplitA::getIz(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return 0.0;
  return (1./zMin - 1./zMax);
}

//--------------------------------------------------------------------------

// The zeta boundaries, for a given value of the evolution scale.

double TrialIFSplitA::getZmin(double Qt2, double sAK, double, double) {
  if (useMevolSav) return max(1.0, Qt2/sAK);
  else return (Qt2 + sAK)/sAK;
}

double TrialIFSplitA::getZmax(double, double, double eA, double eBeamUsed) {
  double xA     = eA/(sqrt(shhSav)/2.0);
  double eAmax  = ((sqrt(shhSav)/2.0) - (eBeamUsed - eA));
  double xAmax  = eAmax/(sqrt(shhSav)/2.0);
  return xAmax/xA;
}

//--------------------------------------------------------------------------

// Inverse transforms to obtain saj and sjk from Qt2 and zeta.

double TrialIFSplitA::getS1j(double Qt2, double zeta, double sAK) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getSj2(Qt2, -zeta, sAK);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  double saj = Qt2;
  if (!useMevolSav) saj *= zeta/(zeta - 1.0);
  return saj;

}

double TrialIFSplitA::getSj2(double Qt2, double zeta, double sAK) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getS1j(Qt2, -zeta, sAK);
  // Sanity check
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  double sjk = sAK*(zeta - 1.0);
  if (useMevolSav) sjk = (zeta - 1.0)*sAK;
  return sjk;

}

//--------------------------------------------------------------------------

// Trial PDF ratio.

double TrialIFSplitA::trialPDFratio(BeamParticle* beamAPtr, BeamParticle*,
  int iSys, int idA, int, double eA, double, double Qt2A, double) {
  double xA = eA/(sqrt(shhSav)/2.0);
  double newPdf = max(beamAPtr->xfISR(iSys,  21, xA, Qt2A), TINYPDFtrial);
  double oldPdf = max(beamAPtr->xfISR(iSys, idA, xA, Qt2A), TINYPDFtrial);
  trialPDFratioSav = newPdf/oldPdf;
  return trialPDFratioSav;
}

//==========================================================================

// K splitting trial function for IF (derived base class), g->qqbar.

//--------------------------------------------------------------------------

// Trial antenna function.

double TrialIFSplitK::aTrial(double saj, double sjk, double sAK) {
  if (saj < 0. || sjk < 0.) return 0.;
  double sectorFac = (sectorShower ? 2. : 1.);
  return sectorFac/2.0/sjk*pow2((sAK + sjk)/sAK);
}

//--------------------------------------------------------------------------

// Generate a new Q value, with first-order running alphaS.

double TrialIFSplitK::genQ2run(double q2old, double sAK,
  double zMin, double zMax, double colFac, double PDFratio,
  double b0, double kR, double Lambda, double, double,
  double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 8.0*M_PI*b0/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  if (sectorShower) comFac *= 0.5;
  double ran    = rndmPtr->flat();
  double facLam = pow2(Lambda/kR);
  return exp(pow(ran, comFac) * log(q2old/facLam)) * facLam;

}

//--------------------------------------------------------------------------

// Generate a new Q value, with constant trial alphaS.

double TrialIFSplitK::genQ2(double q2old, double sAK,
  double zMin, double zMax, double colFac, double alphaS, double PDFratio,
  double, double, double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 8.0*M_PI/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  if (sectorShower) comFac *= 0.5;
  double ran    = rndmPtr->flat();
  return q2old * pow(ran, comFac/alphaS);

}

//--------------------------------------------------------------------------

// Generate a new zeta value in [zMin,zMax].

double TrialIFSplitK::genZ(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return -1.;
  double ran = rndmPtr->flat();
  return ran*(zMin - zMax)+zMax;
}

//--------------------------------------------------------------------------

// The zeta integral.

double TrialIFSplitK::getIz(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return 0.0;
  return (zMax - zMin);
}

//--------------------------------------------------------------------------

// The zeta boundaries, for a given value of the evolution scale.

double TrialIFSplitK::getZmin(double Qt2, double sAK, double eA,
  double eBeamUsed) {
  if (useMevolSav) return 0.0;
  double xA     = eA/(sqrt(shhSav)/2.0);
  double eAmax  = ( (sqrt(shhSav)/2.0) - (eBeamUsed-eA) );
  double xAmax  = eAmax/(sqrt(shhSav)/2.0);
  double sjkmax = sAK*(xAmax - xA)/xA;
  return Qt2/sjkmax;
}

double TrialIFSplitK::getZmax(double, double, double, double) {
  return 1.0;
}

//--------------------------------------------------------------------------

// Inverse transforms to obtain saj and sjk from Qt2 and zeta.

double TrialIFSplitK::getS1j(double Qt2, double zeta, double sAK) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getSj2(Qt2, -zeta, sAK);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  double saj = Qt2+zeta*sAK;
  if (useMevolSav) saj = zeta*(sAK + Qt2);
  return saj;

}

double TrialIFSplitK::getSj2(double Qt2, double zeta, double sAK) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getS1j(Qt2,-zeta,sAK);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  double sjk = Qt2;
  if (!useMevolSav) sjk /= zeta;
  return sjk;

}

//--------------------------------------------------------------------------

// Trial PDF ratio.

double TrialIFSplitK::trialPDFratio(BeamParticle*, BeamParticle*,
  int, int, int, double, double, double, double) {
  trialPDFratioSav = 1.0;
  return trialPDFratioSav;
}

//==========================================================================

// A conversion trial function for IF (derived base class), g->qqbar.

//--------------------------------------------------------------------------

// Trial antenna function.

double TrialIFConvA::aTrial(double saj, double sjk, double sAK) {
  if (saj < 0. || sjk < 0. || sAK < 0.) return 0.;
  return 1.0/saj*pow2((sAK + sjk)/sAK);
}

//--------------------------------------------------------------------------

// Generate a new Q value, with first-order running alphaS.

double TrialIFConvA::genQ2run(double q2old, double sAK,
  double zMin, double zMax, double colFac, double PDFratio,
  double b0, double kR, double Lambda, double, double,
  double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability.
  enhanceFac = max(enhanceFac,1.0);

  // Generate new trial scale.
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 4.0*M_PI*b0/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  double facLam = pow2(Lambda/kR);
  return exp(pow(ran, comFac) * log(q2old/facLam)) * facLam;

}

//--------------------------------------------------------------------------

// Generate a new Q value, with constant trial alphaS.

double TrialIFConvA::genQ2(double q2old, double sAK,
  double zMin, double zMax, double colFac, double alphaS, double PDFratio,
  double, double, double headroomFac, double enhanceFac) {

  // Sanity checks.
  if (!checkInit()) return 0.0;
  if (sAK < 0. || q2old < 0.) return 0.0;

  // Enhance factors < 1: do not modify trial probability
  enhanceFac = max(enhanceFac, 1.0);

  // Generate new trial scale
  double Iz     = getIz(zMin, zMax);
  if (Iz <= 0.) return 0.;
  double comFac = 4.0*M_PI/Iz/colFac/PDFratio/(headroomFac*enhanceFac);
  double ran    = rndmPtr->flat();
  return q2old * pow(ran, comFac/alphaS);

}

//--------------------------------------------------------------------------

// Generate a new zeta value in [zMin,zMax].

double TrialIFConvA::genZ(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return -1.;
  double ran = rndmPtr->flat();
  return zMax*pow(zMin/zMax, ran);
}

//--------------------------------------------------------------------------

// The zeta integral.

double TrialIFConvA::getIz(double zMin, double zMax) {
  if (zMin > zMax || zMin < 0.) return 0.0;
  return log(zMax/zMin);
}

//--------------------------------------------------------------------------

// The zeta boundaries, for a given value of the evolution scale.

double TrialIFConvA::getZmin(double Qt2, double sAK, double, double) {
  if (useMevolSav) {
    if (Qt2<sAK) return 1.0;
    else return Qt2/sAK;
  }
  return (Qt2+sAK)/sAK;
}

double TrialIFConvA::getZmax(double, double sAK, double eA,
  double eBeamUsed) {
  double xA     = eA/(sqrt(shhSav)/2.0);
  double eAmax  = ((sqrt(shhSav)/2.0) - (eBeamUsed - eA));
  double xAmax  = eAmax/(sqrt(shhSav)/2.0);
  double sjkmax = sAK*(xAmax - xA)/xA;
  return (sjkmax+sAK)/sAK;
}

//--------------------------------------------------------------------------

// Inverse transforms to obtain saj and sjk from Qt2 and zeta.

double TrialIFConvA::getS1j(double Qt2, double zeta, double sAK) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getSj2(Qt2, -zeta, sAK);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  double saj = Qt2;
  if (!useMevolSav) saj *= zeta/(zeta - 1.0);
  return saj;

}

double TrialIFConvA::getSj2(double Qt2, double zeta, double sAK) {

  // If zeta < 0, swap invariants.
  if (zeta < 0) return getS1j(Qt2,-zeta,sAK);
  // Sanity check.
  if (Qt2 < 0. || zeta <= 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": unphysical input");
    return 0.;
  }
  double sjk = sAK*(zeta - 1.0);
  if (useMevolSav) sjk = (zeta - 1.0)*sAK;
  return sjk;

}

//--------------------------------------------------------------------------

// Trial PDF ratio.

double TrialIFConvA::trialPDFratio(BeamParticle* beamAPtr, BeamParticle*,
  int iSys, int, int, double eOldA, double, double Qt2A, double) {

  // Number of active flavours.
  double xOldA = eOldA/(sqrt(shhSav)/2.0);
  int nQuark = nGtoQISRSav;
  if (nQuark >= 4 && Qt2A <= 4.0*mcSav*mcSav) nQuark = 3;
  else if (nQuark >= 5 && Qt2A <= 4.0*mbSav*mbSav) nQuark = 4;

  // Old PDF.
  double oldPdf = max(beamAPtr->xfISR(iSys, 21, xOldA, Qt2A),TINYPDFtrial);

  // Store trial PDF weights for later use to pick flavour.
  map<int, double> trialPdfWeight;
  double trialPdfWeightSum = 0.0;
  for (int idQ = -nQuark; idQ <= nQuark; idQ++) {
    // Skip gluon.
    if (idQ == 0) continue;
    // PDF headroom and valence enhancement factor.
    double fac = 2.0 + 0.5 * beamAPtr->nValence(idQ);
    trialPdfWeight[idQ] =
      max(fac * beamAPtr->xfISR(iSys, idQ, xOldA, Qt2A),TINYPDFtrial);
    trialPdfWeightSum += trialPdfWeight[idQ];
  }
  // Pick trial flavour ID and store weight for that flavour, to be
  // used in accept probability.
  double ranFlav = rndmPtr->flat() * trialPdfWeightSum;
  for (map<int,double>::iterator it = trialPdfWeight.begin();
       it != trialPdfWeight.end(); ++it) {
    double newPdf = it->second;
    ranFlav -= newPdf;
    if (ranFlav < 0.) {
      trialFlavSav = it->first;
      trialPDFratioSav = newPdf/oldPdf;
      break;
    }
  }
  // Return sum over all flavours, to be used as evolution coefficient.
  return trialPdfWeightSum/oldPdf;

}

//==========================================================================

// The BranchElementalISR class, container for 2 -> 3 trial branchings.

//--------------------------------------------------------------------------
// Initialise / reset a branchelemental. See header for further definitions.

void BranchElementalISR::reset(int iSysIn, Event& event, int i1In, int i2In,
  int colIn, bool isVal1In, bool isVal2In) {

  // Save system.
  system   = iSysIn;
  // Distinguish between II and IF types.
  isIIsav  = ( !event[i1In].isFinal() && !event[i2In].isFinal() );
  // Make sure that for II 1 is the guy with p+ and 2 is the guy with p-.
  //                    IF 1 is the initial guy and 2 is the final guy.
  bool swap = false;
  if (isIIsav) swap = (event[i1In].pz() < 0.0);
  else swap = (event[i1In].isFinal());
  if (swap) {
    // Valence.
    isVal1sav = isVal2In;
    isVal2sav = (isIIsav ? isVal1In : false);
    // Indices of parents.
    i1sav     = i2In;
    i2sav     = i1In;
  } else {
    // Valence.
    isVal1sav = isVal1In;
    isVal2sav = (isIIsav ? isVal2In : false);
    // Indices of parents.
    i1sav     = i1In;
    i2sav     = i2In;
  }
  // Distinguish between IF types: I on side A or B.
  is1Asav     = (event[i1sav].pz() > 0);
  id1sav      = event[i1sav].id();
  id2sav      = event[i2sav].id();
  colType1sav = event[i1sav].colType();
  colType2sav = event[i2sav].colType();
  colSav      = colIn;
  h1sav       = event[i1sav].pol();
  h2sav       = event[i2sav].pol();
  e1sav       = event[i1sav].e();
  e2sav       = event[i2sav].e();
  // Compute and store antenna invariant mass.
  m2AntSav    = m2(event[i1sav].p(),event[i2sav].p());
  mAntSav     = m2AntSav >= 0 ? sqrt(m2AntSav) : sqrt(-m2AntSav);
  sAntSav     = 2 * event[i1sav].p() * event[i2sav].p();
  // Trial Generators.
  clearTrialGenerators();
  nVeto       = 0;
  nHull       = 0;
  nHadr       = 0;
  // Default antenna properties.
  // 41 = incoming on spacelike main branch.
  // Emission 43 = outgoing produced by a branching.
  // 44 = outgoing shifted by a branching.
  new1=Particle(0,-41,i1sav,i2sav,0,0,0,0,0.);
  new2=Particle(0,43,i1sav,i2sav,0,0,0,0,0.);
  new3=Particle(0,isIIsav?-41:44,i1sav,i2sav,0,0,0,0,0.);
  // Set pointers.
  new1.setEvtPtr(&event);
  new2.setEvtPtr(&event);
  new3.setEvtPtr(&event);

}

//--------------------------------------------------------------------------

// Function to reset all trial generators for this branch elemental.

void BranchElementalISR::clearTrialGenerators() {
  trialGenPtrsSav.resize(0);
  antFunTypePhysSav.resize(0);
  isSwappedSav.resize(0);
  hasSavedTrial.resize(0);
  scaleSav.resize(0);
  scaleOldSav.resize(0);
  zMinSav.resize(0);
  zMaxSav.resize(0);
  colFacSav.resize(0);
  alphaSav.resize(0);
  physPDFratioSav.resize(0);
  trialPDFratioSav.resize(0);
  trialFlavSav.resize(0);
  extraMassPDFfactorSav.resize(0);
  headroomSav.resize(0);
  enhanceFacSav.resize(0);
  nShouldRescue.resize(0);
  nVeto = 0;
  nHull = 0;
  nHadr = 0;
}

//--------------------------------------------------------------------------

// Add a trial generator to this branch elemental.

void BranchElementalISR::addTrialGenerator(enum AntFunType antFunTypePhysIn,
  bool swapIn, TrialGeneratorISR* trialGenPtrIn) {
  trialGenPtrsSav.push_back(trialGenPtrIn);
  antFunTypePhysSav.push_back(antFunTypePhysIn);
  isSwappedSav.push_back(swapIn);
  hasSavedTrial.push_back(false);
  scaleSav.push_back(-1.0);
  scaleOldSav.push_back(-1.0);
  zMinSav.push_back(0.0);
  zMaxSav.push_back(0.0);
  colFacSav.push_back(0.0);
  alphaSav.push_back(0.0);
  physPDFratioSav.push_back(0.0);
  trialPDFratioSav.push_back(0.0);
  trialFlavSav.push_back(0);
  extraMassPDFfactorSav.push_back(0.0);
  headroomSav.push_back(1.0);
  enhanceFacSav.push_back(1.0);
  nShouldRescue.push_back(0);
}

//--------------------------------------------------------------------------

// Save a generated trial branching.

void BranchElementalISR::saveTrial(int iTrial, double qOld, double qTrial,
  double zMin, double zMax, double colFac,double alphaEff, double pdfRatio,
  int trialFlav, double extraMpdf, double headroom, double enhanceFac) {
  hasSavedTrial[iTrial]         = true;
  scaleOldSav[iTrial]           = qOld;
  scaleSav[iTrial]              = qTrial;
  if (qTrial <= 0.) return;
  zMinSav[iTrial]               = zMin;
  zMaxSav[iTrial]               = zMax;
  colFacSav[iTrial]             = colFac;
  alphaSav[iTrial]              = alphaEff;
  trialPDFratioSav[iTrial]      = pdfRatio;
  trialFlavSav[iTrial]          = trialFlav;
  extraMassPDFfactorSav[iTrial] = extraMpdf;
  headroomSav[iTrial]           = headroom;
  enhanceFacSav[iTrial]         = enhanceFac;
}

//--------------------------------------------------------------------------

// Generate invariants for saved branching.

bool BranchElementalISR::genTrialInvariants(double& s1j, double& sj2,
  double eBeamUsed, int iTrial) {

  // Automatically determine which trial function to use if -1 input.
  int iGen = iTrial;
  if (iGen == -1) iGen = getTrialIndex();
  if (iGen <= -1) return false;
  double z = trialGenPtrsSav[iGen]->genZ(zMinSav[iGen],zMaxSav[iGen]);
  // Check physical phase space (note, this only checks massless hull)
  // (Use absolute z value since negative z values are used to
  // indicate swapped invariants for mD ordering).
  double Q2E = pow2(scaleSav[iGen]);
  if (abs(z) < trialGenPtrsSav[iGen]->getZmin(Q2E,sAntSav,e1sav,eBeamUsed) ||
      abs(z) > trialGenPtrsSav[iGen]->getZmax(Q2E,sAntSav,e1sav,eBeamUsed))
    return false;
  // Convert to s1j, sj2.
  s1j = trialGenPtrsSav[iGen]->getS1j(Q2E,z,sAntSav);
  sj2 = trialGenPtrsSav[iGen]->getSj2(Q2E,z,sAntSav);
  return true;

}

//--------------------------------------------------------------------------

// Get trial function index of winner.

int BranchElementalISR::getTrialIndex() const {
  double qMax = 0.0;
  int iMax    = -1;
  for (int i = 0; i < int(scaleSav.size()); ++i) {
    if (hasSavedTrial[i]) {
      double qSav = scaleSav[i];
      if (qSav > qMax) {
        qMax = qSav;
        iMax = i;
      }
    }
  }
  return iMax;
}

//--------------------------------------------------------------------------

// Get scale of winner.

double BranchElementalISR::getTrialScale() const {
  double qMax = 0.0;
  for (int i = 0; i < int(scaleSav.size()); ++i) {
    if (hasSavedTrial[i]) qMax = max(qMax,scaleSav[i]);
    else {
      printOut(__METHOD_NAME__,
        +"Error! not all trials have saved scales");
    }
  }
  return qMax;
}

//--------------------------------------------------------------------------

// Simple print utility, showing the contents of the ISRBranchElemental.

void BranchElementalISR::list(bool header, bool footer) const {

  if (header) {
    cout<< "\n --------  VINCIA ISR Dipole-Antenna Listing  -------------"
        << "---------  (S=sea, V=val, F=final)  "
        << "----------------------------------"
        << "---\n \n"
        << "  sys type    mothers   colTypes   col           ID codes    hels"
        << "          m  TrialGenerators\n";
  }
  cout << setw(5) << system << "   ";
  // Instead of "I" for initial, print out "V" for valence, "S" for sea.
  if (isIIsav) cout << (isVal1sav ? "V" : "S") << (isVal2sav ? "V" : "S");
  else cout << (isVal1sav ? "V" : "S") << "F";
  cout << setw(5) << i1sav << " " << setw(5) << i2sav << "   ";
  cout << setw(3) << colType1sav << " ";
  cout << setw(3) << colType2sav << " ";
  cout << setw(6) << colSav << " ";
  cout << setw(9) << id1sav;
  cout << setw(9) << id2sav << "   ";
  // Helicities temporarily output as zero.
  cout << setw(2) << h1sav << " " << setw(2) << h2sav << " ";
  cout << setw(10) << mAnt() << " ";
  for (int j = 0; j < (int)trialGenPtrsSav.size(); j++) {
    string trialName = trialGenPtrsSav[j]->name();
    trialName.erase(0, 5);
    cout << " " << trialName;
  }
  cout << "\n";
  if (footer)
    cout << "\n --------  End VINCIA SpaceShower Antenna Listing  --------"
         << "--------------"
         << "-----------------------------------------------------------\n";

}

//==========================================================================

// The VinciaISR class.

//--------------------------------------------------------------------------

// Initialize shower.

void VinciaISR::init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn) {

  // Check if already initialized.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Verbose level.
  verbose         = settingsPtr->mode("Vincia:verbose");

  // Showers on/off.
  bool doISR = settingsPtr->flag("PartonLevel:ISR");
  doII = doISR && settingsPtr->flag("Vincia:doII");
  doIF = doISR && settingsPtr->flag("Vincia:doIF");

  // Beam parameters.
  beamFrameType = settingsPtr->mode("Beams:frameType");

  // Beam particles.
  beamAPtr = beamAPtrIn;
  beamBPtr = beamBPtrIn;
  if (beamAPtr->pz() < 0)
    infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": beamA has pz < 0");

  // Assume all events in same run have same beam-beam energies.
  m2BeamsSav   = m2(beamAPtr->p(), beamBPtr->p());
  eCMBeamsSav  = sqrt(m2BeamsSav);
  eBeamA = beamAPtr->e();
  eBeamB = beamBPtr->e();

  // Possibility to allow user veto of emission step.
  hasUserHooks    = (userHooksPtr != 0);
  canVetoEmission = (hasUserHooks && userHooksPtr->canVetoISREmission());

  // Number of active quark flavours.
  nGluonToQuarkI    = settingsPtr->mode("Vincia:nGluonToQuark");
  nGluonToQuarkF    = settingsPtr->mode("Vincia:nGluonToQuark");
  convGluonToQuarkI = settingsPtr->flag("Vincia:convertGluonToQuark");
  convQuarkToGluonI = settingsPtr->flag("Vincia:convertQuarkToGluon");

  // Mass corrections.
  nFlavZeroMass     = settingsPtr->mode("Vincia:nFlavZeroMass");

  // Global flag for helicity dependence.
  helicityShower    = settingsPtr->flag("Vincia:helicityShower");

  // Global flag for sector showers on/off.
  sectorShower      = settingsPtr->flag("Vincia:sectorShower");

  // Allow to try other IF kinematics map if selected fails.
  kineMapIFretry    = settingsPtr->flag("Vincia:kineMapIFretry");

  // Merging flags.
  doMerging = settingsPtr->flag("Merging:doMerging");
  isTrialShower    = false;
  isTrialShowerRes = false;

  // Mass windows : specifies thresholds for each flavour.
  mt = vinComPtr->mt;
  mb = vinComPtr->mb;
  mc = vinComPtr->mc;
  ms = vinComPtr->ms;
  mtb = sqrt(mt*mb);

  // Perturbative cutoff.
  cutoffScaleII = settingsPtr->parm("Vincia:cutoffScaleII");
  cutoffScaleIF = settingsPtr->parm("Vincia:cutoffScaleIF");

  // Check PDF Q2min value and issue warning if above ISR cutoff scale(s)
  double xTest = 0.1;
  bool insideBounds = true;
  if (beamAPtr->isHadron()) {
    if (doII && !beamAPtr->insideBounds(xTest, pow2(cutoffScaleII)))
      insideBounds = false;
    if (doIF && !beamAPtr->insideBounds(xTest, pow2(cutoffScaleIF)))
      insideBounds = false;
  }
  if (beamBPtr->isHadron()) {
    if (doII && !beamBPtr->insideBounds(xTest, pow2(cutoffScaleII)))
      insideBounds = false;
    if (doIF && !beamBPtr->insideBounds(xTest, pow2(cutoffScaleIF)))
      insideBounds = false;
  }
  if (!insideBounds) {
    infoPtr->errorMsg("Warning in"+__METHOD_NAME__+": PDF QMin scale is "
      "above ISR shower cutoff.","PDFs will be treated as frozen below QMin.");
  }

  // Set shower alphaS pointer.
  useCMW                = settingsPtr->flag("Vincia:useCMW");
  alphaSptr             = &vinComPtr->alphaStrong;
  if (useCMW) alphaSptr = &vinComPtr->alphaStrongCMW;

  // AlphaS parameters.
  alphaSorder    = settingsPtr->mode("Vincia:alphaSorder");
  alphaSvalue    = settingsPtr->parm("Vincia:alphaSvalue");
  aSkMu2EmitI    = settingsPtr->parm("Vincia:renormMultFacEmitI");
  aSkMu2SplitI   = settingsPtr->parm("Vincia:renormMultFacSplitI");
  aSkMu2Conv     = settingsPtr->parm("Vincia:renormMultFacConvI");
  aSkMu2SplitF   = settingsPtr->parm("Vincia:renormMultFacSplitF");
  alphaSmax      = settingsPtr->parm("Vincia:alphaSmax");
  alphaSmuFreeze = settingsPtr->parm("Vincia:alphaSmuFreeze");
  mu2freeze      = pow2(alphaSmuFreeze);

  // Smallest allowed scale.
  alphaSmuMin = max(alphaSmuFreeze, 1.05*alphaSptr->Lambda3());
  mu2min      = pow2(alphaSmuMin);

  // Check largest numerical value of alphaS we can have.
  if (alphaSorder >= 1) alphaSmax = min(alphaSmax, alphaSptr->alphaS(mu2min));

  // If we want to get rid of heavy quarks we need to change the masses
  // to the ones in the pdfs.
  BeamParticle* beamUsePtr =
    ((abs(beamAPtr->id()) < 100) ? beamBPtr : beamAPtr);
  if ((abs(beamUsePtr->id()) > 100) && (nFlavZeroMass < 5)) {
    vector<double> masses;
    masses.resize(2);
    masses[0] = settingsPtr->parm("Vincia:ThresholdMB");
    masses[1] = settingsPtr->parm("Vincia:ThresholdMC");
    for (int i = 0; i < (5 - nFlavZeroMass); i++) {
      // Check if we can get mass from beams directly (LHAPDF6 only).
      if (beamUsePtr->mQuarkPDF(5 - i) > 0.0) {
        masses[i] = beamUsePtr->mQuarkPDF(5 - i);
        continue;
      }
      // If not attempt to find it.
      double startMass  = masses[i];
      int maxTry        = 500;
      double scale2Last = pow2(startMass);
      double xfLast     = beamUsePtr->xf(5 - i ,0.001, scale2Last);
      for (int j = 1; j <= maxTry; j++) {
        double scale2Now = pow2( startMass + 0.005*((double)(j)) );
        double xfNow     = beamUsePtr->xf(5 - i, 0.001 ,scale2Now);
        // Set x = 0.001 and check the gradient.
        if ((xfNow-xfLast) > NANO) {
          masses[i] = sqrt(scale2Last);
          break;
        }
        // Update
        scale2Last = scale2Now;
        xfLast     = xfNow;
      }
    }
    for (int i = 0; i < (int)masses.size(); i++) {
      (i == 0 ? mb : mc) = masses[i];
      if (verbose >= REPORT)
        printOut(__METHOD_NAME__, "Found " + num2str(masses[i]) +
          (i==0 ? " as b mass." : " as c mass."));
    }
    // Reset the derived scales.
    ms  = min(mc, ms);
    mtb = sqrt(mt*mb);
  }

  // Evolution windows.
  regMinScalesMtSav.clear();
  regMinScalesSav.clear();
  regMinScalesNow.clear();
  // Fill constant version with masses.
  regMinScalesMtSav.push_back(mc/16.0);
  regMinScalesMtSav.push_back(mc/5.0);
  regMinScalesMtSav.push_back(mc);
  regMinScalesMtSav.push_back(mb);
  regMinScalesMtSav.push_back(mtb);
  regMinScalesMtSav.push_back(mt);
  // Fill the rest.
  regMinScalesSav = regMinScalesMtSav;
  double qMinNow  = 2.0*mt;
  double multFac  = 2.0;
  while (qMinNow < eCMBeamsSav) {
    int iRegNew = int(log(qMinNow/mt)/log(5) + 5);
    int iRegMax = int(regMinScalesSav.size()) - 1;
    if (iRegNew > iRegMax)
      regMinScalesSav.push_back(pow(5, (double)(iRegMax+1) - 5.0)*mt);
    qMinNow *= multFac;
  }

  // Trial generators.
  vector<TrialGeneratorISR*> trialGenerators;
  trialGenerators.push_back(&trialIISoft);
  trialGenerators.push_back(&trialIIGCollA);
  trialGenerators.push_back(&trialIIGCollB);
  trialGenerators.push_back(&trialIISplitA);
  trialGenerators.push_back(&trialIISplitB);
  trialGenerators.push_back(&trialIIConvA);
  trialGenerators.push_back(&trialIIConvB);
  trialGenerators.push_back(&trialIFSoft);
  trialGenerators.push_back(&trialVFSoft);
  trialGenerators.push_back(&trialIFGCollA);
  trialGenerators.push_back(&trialIFSplitA);
  trialGenerators.push_back(&trialIFSplitK);
  trialGenerators.push_back(&trialIFConvA);
  if (sectorShower) trialGenerators.push_back(&trialIFGCollK);
  for (int indx = 0; indx < int(trialGenerators.size()); ++indx) {
    trialGenerators[indx]->initPtr(infoPtr);
    trialGenerators[indx]->init(mc, mb);
  }

  // Enhance settings.
  enhanceInHard = settingsPtr->flag("Vincia:enhanceInHardProcess");
  enhanceInMPI  = settingsPtr->flag("Vincia:enhanceInMPIshowers");
  enhanceAll    = settingsPtr->parm("Vincia:enhanceFacAll");
  enhanceBottom = settingsPtr->parm("Vincia:enhanceFacBottom");
  enhanceCharm  = settingsPtr->parm("Vincia:enhanceFacCharm");
  enhanceCutoff = settingsPtr->parm("Vincia:enhanceCutoff");

  // Resize Paccept to the maximum number of elements.
  Paccept.resize(max(weightsPtr->getWeightsSize(), 1));

  // Clear containers.
  clearContainers();

  // Rescue levels.
  doRescue  = true;
  nRescue   = 100;
  rescueMin = 1.0e-6;

  // Initialize factorization scale and parameters for shower starting scale.
  pTmaxMatch     = settingsPtr->mode("Vincia:pTmaxMatch");
  pTmaxFudge     = settingsPtr->parm("Vincia:pTmaxFudge");
  pT2maxFudge    = pow2(pTmaxFudge);
  pT2maxFudgeMPI = pow2(settingsPtr->parm("Vincia:pTmaxFudgeMPI"));
  TINYPDF        = pow(10, -10);

  // Initialise the ISR antenna functions.
  if (verbose >= REPORT)
    printOut(__METHOD_NAME__,"initializing antenna set");
  antSetPtr->init();

  // Print VINCIA header and list of parameters, call issued by ISR
  // since we are initialised last.
  if (verbose >= NORMAL) fsrPtr->header();
  isInit = true;
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);

}

//--------------------------------------------------------------------------

// Method to determine if max pT limit should be imposed on first emission.

bool VinciaISR::limitPTmax(Event& event, double, double) {

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
      int idAbs = event[partonSystemsPtr->getOut(iSysHard, i)].idAbs();
      if (idAbs <= 5 || idAbs == 21 || idAbs == 22) return true;
      else if (idAbs == 6 && nGluonToQuarkF == 6) return true;
    }
    // If no QCD/QED partons detected, allow to go to phase-space maximum.
    return false;
  }

}

//--------------------------------------------------------------------------

// Prepare system of partons for evolution; identify ME.

void VinciaISR::prepare( int iSys, Event& event, bool) {

  // Check if we are supposed to do anything.
  if (!(doII || doIF)) return;
  if (infoPtr->getAbortPartonLevel()) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": Aborting.");
    return;
  }

  // Verbose output.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "begin", dashLen);
    stringstream ss;
    ss << "Preparing system " << iSys;
    printOut(__METHOD_NAME__, ss.str());
    event.list();
    partonSystemsPtr->list();
  }

  // Resetting for first time in new event.
  if (!isPrepared) {
    // Reset counters.
    vinComPtr->resetCounters();

    // Reset all vectors for the first time we are called.
    clearContainers();
    eBeamAUsed = 0.0;
    eBeamBUsed = 0.0;

    // Evolution windows, add factorization scale.
    regMinScalesNow.clear();
    regMinScalesNow = regMinScalesMtSav;
    regMinScalesNow.push_back(sqrt(infoPtr->Q2Fac()));
    // Sort to put factorization scale at the right place.
    stable_sort(regMinScalesNow.begin(), regMinScalesNow.end());
    // Fill the rest.
    double qMinOld = regMinScalesNow[(int)regMinScalesNow.size()-1];
    double qMinNow = 2.0*qMinOld;
    double multFac = 2.0;
    while (qMinNow < eCMBeamsSav) {
      int iRegNew = int(log(qMinNow/qMinOld)/log(6) + 6);
      int iRegMax = int(regMinScalesNow.size()) - 1;
      if (iRegNew > iRegMax)
        regMinScalesNow.push_back(pow(6, (double)(iRegMax+1)-6.0)*qMinOld);
      qMinNow *= multFac;
    }
  }

  // Sanity check: at least two particles in system.
  int sizeSystem = partonSystemsPtr->sizeAll(iSys);
  if (sizeSystem <= 1) return;

  // Check if this is a resonance-decay system; if so, let FSR deal with it.
  hasPrepared[iSys] = false;
  if ( !partonSystemsPtr->hasInAB(iSys) ) return;
  if ( isTrialShowerRes ) return;

  // Flag to tell FSR that ISR::prepare() has treated this system. We
  // assume that when both ISR::prepare() and FSR::prepare() are
  // called, the sequence is that ISR::prepare() is always called
  // first.
  hasPrepared[iSys] = true;

  // We don't have a starting scale for this system yet.
  Q2hat[iSys] = 0.0;
  // After prepare we always have zero branchings.
  nBranch[iSys] = 0;
  nBranchISR[iSys] = 0;

  // Set isHardSys and isResonance flags.
  int nIn = 0;
  if (partonSystemsPtr->hasInAB(iSys)) nIn = 2;
  if (nIn == 2 && iSys == 0) isHardSys[iSys] = true;

  // Make light quarks (and initial-state partons) explicitly massless.
  bool makeNewCopies = false;
  if (!vinComPtr->mapToMassless(iSys, event, makeNewCopies))
    return;

  // Update the beam pointers (incoming partons stored as 0,1 in
  // partonSystems). Needed in case makeNewCopies = true; also
  // ensures beamA == positive pZ.
  for (int iBeam = 0; iBeam <= 1; ++iBeam) {
    int iNew = (iBeam == 0) ? partonSystemsPtr->getInA(iSys) :
      partonSystemsPtr->getInB(iSys);
    if (iNew == 0) continue;
    BeamParticle& beamNow = (event[iNew].pz() > 0.0 ?
      *beamAPtr : *beamBPtr);
    double eBeamNow       = (event[iNew].pz() > 0.0 ? eBeamA : eBeamB);
    beamNow[iSys].update( iNew, event[iNew].id(), event[iNew].e()/eBeamNow );
  }

  // Then see if we know how to compute MECs for this conf.
  doMECsSys[iSys] = mecsPtr->prepare(iSys, event);
  // Decide if we should be doing ME corrections for next order.
  if (doMECsSys[iSys]) doMECsSys[iSys] = mecsPtr->doMEC(iSys, 1);
  // Initialise polarisation flag.
  polarisedSys[iSys] = mecsPtr->isPolarised(iSys, event);

  // Communicate with FSR.
  fsrPtr->polarisedSys[iSys]   = polarisedSys[iSys];
  fsrPtr->doMECsSys[iSys]      = doMECsSys[iSys];
  fsrPtr->isHardSys[iSys]      = isHardSys[iSys];
  fsrPtr->isResonanceSys[iSys] = false;

  // Then see if we should colourise this conf.
  colourPtr->colourise(iSys, event);
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Event after colourise:");
    event.list();
    printOut(__METHOD_NAME__,"Making colour maps");
  }

  // Find and save all colors and anticolors.
  map<int,int> indexOfAcol;
  map<int,int> indexOfCol;

  // Loop over event record. Find indices of particles with color lines.
  for (int i = 0; i < sizeSystem; ++i) {
    int i1 = partonSystemsPtr->getAll( iSys, i);
    if ( i1 <= 0 ) continue;
    // Save to colour maps.
    int col  = event[i1].col();
    int acol = event[i1].acol();
    // Cross colours for initial partons.
    if (!event[i1].isFinal()) {
      col  = acol;
      acol = event[i1].col();
      if (event[i1].pz() > 0.0) {
        initialA[iSys] = event[i1];
        eBeamAUsed  += event[i1].e();
      } else {
        initialB[iSys] = event[i1];
        eBeamBUsed        += event[i1].e();
      }
    }
    if (col > 0) indexOfCol[col] = i1;
    else if (col < 0) indexOfAcol[-col] = i1;
    if (acol > 0) indexOfAcol[acol] = i1;
    else if (acol < 0) indexOfCol[-acol] = i1;
  }

  // Now loop over colored particles to create branch elementals (=antennae).
  int sizeOld = branchElementals.size();
  for (map<int,int>::iterator it = indexOfCol.begin();
       it != indexOfCol.end(); ++it) {
    // Colour index.
    int col = it->first;
    // i1 is the colour (or incoming anticolour) carrier.
    // i2 is the anticolour (or incoming colour) carrier.
    int i1 = it->second;
    int i2 = indexOfAcol[col];
    if (col == 0 || i1 == 0 || i2 == 0) continue;
    // Exclude final-final antennae.
    if ((event[i1].isFinal()) && (event[i2].isFinal())) continue;
    if (verbose >= DEBUG ) {
      stringstream ss;
      ss <<"Creating antenna between ";
      ss << i1 << " , " << i2 << " col = " << col;
      printOut(__METHOD_NAME__, ss.str());
    }

    // Check whether i1 is valence (if incoming).
    bool isVal1(false);
    if (!event[i1].isFinal()) {
      BeamParticle& beam1 = (event[i1].pz() > 0.0) ? *beamAPtr : *beamBPtr;
      isVal1              = beam1[iSys].isValence();
    }
    // Check whether i2 is valence (if incoming).
    bool isVal2(false);
    if (!event[i2].isFinal()) {
      BeamParticle& beam2 = (event[i2].pz() > 0.0) ? *beamAPtr : *beamBPtr;
      isVal2              = beam2[iSys].isValence();
    }

    // Store trial QCD antenna and add trial generators depending on type.
    BranchElementalISR trial(iSys, event, i1, i2, col, isVal1, isVal2);
    resetTrialGenerators(&trial);
    trial.renewTrial();
    branchElementals.push_back(trial);
  }

  // Count up number of gluons and quark pairs.
  nG[iSys]  = 0;
  nQQ[iSys] = 0;
  for (int i = 0; i < (int)partsSav[iSys].size(); i++) {
    if (partsSav[iSys][i].id() == 21) nG[iSys]++;
    else if (abs(partsSav[iSys][i].id()) < 7) nQQ[iSys]++;
  }
  // Halve the quarks to get quark pairs.
  nQQ[iSys] = nQQ[iSys]/2;

  // Save information about Born state only when doing a sector shower
  // and only if this is not a trial shower (because then this was set
  // in VinciaMerging::getWeightCKKWL() already).
  if (!isPrepared && sectorShower && !isTrialShower)
    saveBornState(event, iSys);

  // Sanity check.
  if ((int)branchElementals.size() == sizeOld) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "did not find any antennae: exiting.");
    return;
  }

  // Set starting scale for this system.
  setStartScale(iSys,event);

  isPrepared=true;
  if (verbose >= DEBUG) {
    list();
    printOut(__METHOD_NAME__, "end", dashLen);
  }
}

//--------------------------------------------------------------------------

// Update dipole list after each FSR emission.

void VinciaISR::update( int iSys, Event& event, bool) {

  // Skip if the branching system has no incoming partons.
  if (!(doII || doIF) || !isPrepared) return;
  if (!partonSystemsPtr->hasInAB(iSys)) return;
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "begin", dashLen);
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Updating iSys: " << iSys;
      printOut(__METHOD_NAME__, ss.str());
      event.list();
      printOut(__METHOD_NAME__, "list of ISR dipoles before update:");
      list();
      partonSystemsPtr->list();
    }
  }
  int inA = partonSystemsPtr->getInA(iSys);
  int inB = partonSystemsPtr->getInB(iSys);
  if (inA <= 0 || inB <= 0 ) {
    stringstream ss;
    ss << "in system " << iSys << " inA = " << inA << " inB = " << inB;
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Incoming particles nonpositive",ss.str());
    return;
  }

  // Count number of branches.
  nBranch[iSys]++;

  // Particles in the list are already updated by FSR.
  // Find and save all colors and anticolors.
  map<int,int> indexOfAcol;
  map<int,int> indexOfCol;
  // Also count number of gluons and quark pairs.
  nG[iSys]  = 0;
  nQQ[iSys] = 0;
  for (int i = 0; i < partonSystemsPtr->sizeAll(iSys); ++i) {
    int i1 = partonSystemsPtr->getAll( iSys, i);
    if (i1 <= 0) continue;
    // Save to colour maps.
    if (i1 > event.size()) {
      if (verbose >= DEBUG) {
        event.list();
        cout << " iSys = " << iSys << " / nSys = "
             << partonSystemsPtr->sizeSys()
             << " i = " << i << " / n = " << partonSystemsPtr->sizeAll(iSys)
             << " i1 = " << i1 << " / event.size() = " << event.size() << endl;
      }
    }
    int col  = event[i1].col();
    int acol = event[i1].acol();

    // Switch colours for initial partons.
    if (!event[i1].isFinal()) {
      col  = acol;
      acol = event[i1].col();
    }
    if (col > 0) indexOfCol[col] = i1;
    else if (col < 0) indexOfAcol[-col] = i1;
    if (acol > 0) indexOfAcol[acol] = i1;
    else if (acol < 0) indexOfCol[-acol] = i1;

    // Count number of gluons and quark pairs.
    if (event[i1].id() == 21) nG[iSys]++;
    else if (event[i1].idAbs() < 7) nQQ[iSys]++;
  }
  nQQ[iSys] = nQQ[iSys]/2;

  // Loop over the antennae, and look for changed ones.
  // Start from back so that if we remove one it won't mess up loop
  for (vector<BranchElementalISR>::iterator antIt = branchElementals.end() - 1;
       antIt != branchElementals.begin() - 1; --antIt) {
    // Only check antennae in same system.
    if (antIt->system != iSys) continue;
    bool doUpdate = false;
    bool doRemove = false;
    bool foundColour=true;
    int antCol = antIt->col();
    int i1 = antIt->geti1();
    int i2 = antIt->geti2();
    int i1New = i1;
    int i2New = i2;

    // Sanity check. We don't destroy colour lines.
    // Antenna colour should not have disappeared.
    if (indexOfAcol.find(antCol) == indexOfAcol.end()) {
      if (verbose >= NORMAL) infoPtr->errorMsg("Warning in "+__METHOD_NAME__
        +": Could not find antenna colour in list of anti-colour indices.");
      foundColour = false;
      doRemove = true;
    }
    if (indexOfCol.find(antCol) == indexOfCol.end()) {
      if (verbose >= NORMAL) infoPtr->errorMsg("Warning in "+__METHOD_NAME__
        +": Could not find antenna colour in list of colour indices.");
      foundColour = false;
      doRemove = true;
    }
    if (foundColour) {
      // Initial-initial antennae.
      if (antIt->isII()) {

        // Fetch up to date i1.
        // Check if i1 attached on colour or anticolour end of dipole.
        if (event[i1].col() >0 && event[i1].col() == antCol)
          i1New = indexOfAcol[antCol];
        else i1New = indexOfCol[antCol];

        // Fetch up to date i2.
        // Check if i1 attached on colour or anticolour end of dipole.
        if (event[i2].col() >0 && event[i2].col() == antCol)
          i2New = indexOfAcol[antCol];
        else i2New = indexOfCol[antCol]; //col because initial

        // Check if i1 is still the incoming particle.
        if (i1New != inA) {
          // Check if a QED backwards evolution.
          bool QEDbackwards=false;
          if (event[i1New].isFinal() && event[i1New].mother1()==inA &&
              event[inA].id()==22) QEDbackwards=true;
          // Otherwise we dont't know about this case.
          if (!QEDbackwards && verbose >= NORMAL) {
            infoPtr->errorMsg("Warning in "+__METHOD_NAME__
              +": Could not find iA in II antenna! Removing.");
          }
          doRemove=true;
        }

        // First check if i2 is still the incoming particle.
        if (i2New != inB) {
          // Check if a QED backwards evolution.
          bool QEDbackwards = false;
          if (event[i2New].isFinal() && event[i2New].mother1() == inB &&
             event[inB].id() == 22) QEDbackwards = true;
          // Otherwise we dont't know about this case.
          if (!QEDbackwards && verbose >= NORMAL) {
            infoPtr->errorMsg("Warning in "+__METHOD_NAME__+
              ": Could not find iB in II antenna! Removing.");
          }
          doRemove=true;
        }

        // Check if need to update.
        if (!doRemove && (i1 != i1New || i2 != i2New)) doUpdate = true;

      // Initial-final antennae.
      } else {

        // Fetch up to date i1.
        // Ceck if i1 attached on colour or anticolour end of dipole.
        if (event[i1].col() >0 && event[i1].col() == antCol)
          i1New = indexOfAcol[antCol];
        else i1New = indexOfCol[antCol];

        // Fetch up to date i2.
        if (event[i2].col()>0 && event[i2].col()==antCol)
          i2New = indexOfCol[antCol];
        else i2New = indexOfAcol[antCol];

        //Check if i1New is still incoming.
        int inX = antIt->is1A() ? inA : inB;
        if (i1New != inX) {

          // Check if QED backwards evolution.
          bool QEDbackwards = false;
          if (event[i1New].isFinal() && event[i1New].mother1() == inX &&
            event[inX].id() == 22) QEDbackwards = true;
          // Otherwise we dont't know about this case.
          if (!QEDbackwards && verbose >= NORMAL) {
            infoPtr->errorMsg("Warning in "+__METHOD_NAME__
              +": Could not find inA/inB in IF antenna! Removing.");
          }
          doRemove = true;
        }

        // Check if need to update.
        else if (i1 != i1New || i2 != i2New) doUpdate = true;

      }

      // Recompute antenna mass.
      if (doUpdate) {
        antIt->reset(iSys, event, i1New,i2New, antCol,
          antIt->isVal1(), antIt->isVal2());
        resetTrialGenerators(&(*antIt));
      }
      indexOfAcol.erase(antCol);
      indexOfCol.erase(antCol);
    }

    // Remove antenna either because something went wrong or QED
    // backwards evol.
    if (doRemove) branchElementals.erase(antIt);

  } // End loop over branchers.

  // Check leftover colour lines (dismiss any FF).
  // Can occur e.g. if photon backwards evolves into qqbar.
  for (map<int,int>::iterator colIt = indexOfCol.begin();
       colIt != indexOfCol.end(); ++colIt) {
    int colNow = colIt->first;
    int i1Now = colIt->second;
    if (indexOfAcol.find(colNow) == indexOfAcol.end()) {
      if (verbose >= NORMAL) {
        stringstream ss;
        ss << " Colour tag = " << colNow << " event index: " << i1Now;
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Unmatched colour index. Aborting.",ss.str());
        infoPtr->setAbortPartonLevel(true);
        return;
      }
    } else {
      int i2Now=indexOfAcol[colNow];
      if (i1Now <=0 || i2Now <= 0) {
        if (verbose >= NORMAL) infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Colour tag attached to impossible event index!");
        // Exclude final-final antennae.
      } else if (!event[i1Now].isFinal() || !event[i2Now].isFinal()) {
        if (verbose >= DEBUG) {
          stringstream ss;
          ss<<"Creating antenna between ";
          ss << i1Now << " , " << i2Now <<" col = "<< colNow;
          printOut(__METHOD_NAME__, ss.str());
        }
        BeamParticle& beam1   = (event[i1Now].pz() > 0.0) ?
          *beamAPtr : *beamBPtr;
        BeamParticle& beam2   = (event[i2Now].pz() > 0.0) ?
          *beamAPtr : *beamBPtr;
        bool isVal1           = beam1[iSys].isValence();
        bool isVal2           = beam2[iSys].isValence();

        // Store trial QCD antenna and add trial generators depending
        // on type.
        BranchElementalISR trial(iSys, event, i1Now, i2Now, colNow, isVal1,
          isVal2);
        resetTrialGenerators(&trial);
        trial.renewTrial();
        branchElementals.push_back(trial);
      }
      indexOfAcol.erase(colNow);
    }
  }

  // There was an unmatched colour line.
  if (indexOfAcol.size() > 0) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Unmatched anticolour index!");
    infoPtr->setAbortPartonLevel(true);
    return;
  }

  // If we are going from ME-corrected to non-ME-corrected order,
  // renew trials.
  //TODO +1 correct?
  if (doMECsSys[iSys] && !mecsPtr->doMEC(iSys, nBranch[iSys]+1)) {
    doMECsSys[iSys] = false;
    for (int i = 0; i < (int)branchElementals.size(); i++) {
      BranchElementalISR* trial = &branchElementals[i];
      if (trial->system == iSys) trial->renewTrial();
    }
  }

  // Sanity check.
  if (branchElementals.size() <= 0) {
    if (verbose >= DEBUG)
      printOut("VinciaISR::update", "did not find any antennae: exiting.");
    return;
  }
  if (verbose >= DEBUG) {
    if (!checkAntennae(event)) {
      list();
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Failed checkAntennae. Aborting.");
      infoPtr->setAbortPartonLevel(true);
      return;
    }
  }
  if (verbose >= DEBUG) {
    list();
    printOut(__METHOD_NAME__, "end", dashLen);
  }

}

//--------------------------------------------------------------------------

// Select next pT in downwards evolution.

double VinciaISR::pTnext(Event& event, double pTevolBegAll,
  double pTevolEndAll, int, bool) {

  // Check if we are supposed to do anything.
  if (infoPtr->getAbortPartonLevel() || !(doII || doIF)) return 0.;
  if (branchElementals.size() <= 0) return 0.0;

  // Verbose output.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "begin", dashLen);
    stringstream ss;
    ss << "(re)starting evolution between pTevolBegAll = "
       << num2str(pTevolBegAll) << " and pTevolEndAll = " << pTevolEndAll;
    printOut(__METHOD_NAME__, ss.str());
  }

  // Diagnostics
  if (verbose >= REPORT) diagnosticsPtr->start(__METHOD_NAME__);

  // Checks: skipped ants (q[MPI/FSR] > qISR), qEnd < all cutoffs, no ants.
  bool allSkipped        = true;
  bool qEndsmallerCutoff = true;

  // Denote VINCIA scales by "q", PYTHIA ones by "pTevol".
  double qOld    = pTevolBegAll;
  double qEndAll = pTevolEndAll;

  // Initialize winner scale.
  double qWin = 0.0;
  winnerPtr   = nullptr;
  indxWin     = -1;
  iSysWin     = -1;

  // Loop over antennae (in all currently existing parton systems).
  unsigned int nAnt = branchElementals.size();
  if (verbose >= DEBUG) {
    stringstream ss;
    ss <<"Looping over " << nAnt << " antennae.";
    printOut(__METHOD_NAME__, ss.str());
  }
  for (unsigned int iAnt = 0; iAnt < nAnt; iAnt++) {
    // Shorthand for this antenna.
    BranchElementalISR* trialPtr = &branchElementals[iAnt];
    int iSys = trialPtr->system;
    double qMax = min(qOld, sqrt(Q2hat[iSys]));
    double s12  = trialPtr->sAnt();
    int id1     = trialPtr->id1sav;
    int id2     = trialPtr->id2sav;
    double e1   = trialPtr->e1sav;
    double e2   = trialPtr->e2sav;
    bool isII   = trialPtr->isII();
    bool is1A   = trialPtr->is1A();

    // Check if we are skipping this kind of antenna.
    if (isII && !doII) continue;
    else if (!isII && !doIF) continue;
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Processing antenna ";
      ss << iAnt+1 << " / " << nAnt;
      printOut(__METHOD_NAME__, ss.str());
      ss.str("");
      ss << "Sys = " << iSys << " id1 = " << id1 << " id2 = " << id2
         << " nTrialGens = " << trialPtr->nTrialGenerators()
         << " qMax = " << qMax;
      printOut(__METHOD_NAME__, ss.str());
    }

    // Generate new trial branchings, starting from qMax.
    double qBegin = qMax;
    // Lowest evolution boundary: impose hadronization scale.
    double qEnd = qEndAll;
    // Cutoff scale, depends on whether we are II or IF antenna.
    double cutoffScale = (isII ? cutoffScaleII : cutoffScaleIF);
    // No trial generators for this antenna
    if (trialPtr->nTrialGenerators() == 0) continue;

    // Loop over trial functions. Find and save max trial scale.
    double qTrialMax = 0.0;
    int indxMax      = -1;
    for (int indx = 0; indx < (int)trialPtr->nTrialGenerators(); ++indx) {
      // Pointer to trial generator for this trial.
      TrialGeneratorISR* trialGenPtr = trialPtr->trialGenPtrsSav[indx];
      enum AntFunType antFunTypePhys = trialPtr->antFunTypePhys(indx);
      bool swapped       = trialPtr->getIsSwapped(indx);
      double qBeginNow   = qBegin;
      double qEndNow     = qEnd;

      // Phase space limit can never be exceeded.
      qBeginNow = min(qBeginNow, sqrt(trialGenPtr->getQ2max(s12, e1,
        is1A ? eBeamAUsed : eBeamBUsed)));

      // If restart scale < scale we already found continue.
      if (qBeginNow < qWin || qBeginNow < qTrialMax) continue;
      if (qEndAll > cutoffScale) qEndsmallerCutoff = false;

      // Check if any phase space still open.
      if ((qBeginNow <= cutoffScale) || (qBeginNow <= qEndNow)) {
        // Verbose output.
        if (verbose >= DEBUG) printOut(__METHOD_NAME__,
            "skipping this trial since qBeginNow = " + num2str(qBeginNow) +
            " cutoffScale = " + num2str(cutoffScale) +
            " qEndNow = " + num2str(qEndNow));
        continue;
      }
      allSkipped = false;
      double qTrial = qBeginNow;

      // Impose evolution windows.
      bool acceptRegion = false;
      int iRegion = getRegion(qTrial);

      // Go through regions.
      while (!acceptRegion) {

        // Set overestimated Z range for trial generation.
        double qMinNow  = max(cutoffScale, getQmin(iRegion));
        double q2MinNow = pow2(qMinNow);
        double zMinNow  = trialGenPtr->getZmin(q2MinNow, s12, e1,
          is1A ? eBeamAUsed : eBeamBUsed);
        double zMaxNow  = trialGenPtr->getZmax(q2MinNow, s12, e1,
          is1A ? eBeamAUsed : eBeamBUsed);

        // Set headroom factor (= constant multiplying trial probability).
        double headroomFac = getHeadroomFac(iSys, antFunTypePhys, qMinNow);

        // Check for rescue mechanism in case trial gets stuck.
        if (doRescue && trialPtr->getNshouldRescue(indx) >= nRescue) {
          // Multiply headroom.
          double logRescue = ((double)(trialPtr->getNshouldRescue(indx))) /
            ((double)(nRescue));
          headroomFac *= pow(10.0,-logRescue);
          if (verbose >= DEBUG){
            stringstream ss;
            ss << "Applying rescue mechanism, nShouldRescue = "
               << trialPtr->getNshouldRescue(indx)
               << ", multiplying headroom with 10^-" << logRescue;
            printOut(__METHOD_NAME__, ss.str());
          }
        }

        // Set PDFratio. If II the first is always side A. If IF
        // check which side the initial guy is. For g->qq
        // splittings.
        double PDFscale = pow2(qTrial);
        double pdfRatio = trialGenPtr->trialPDFratio(
          ((isII || is1A) ? beamAPtr : beamBPtr),
          ((isII || is1A) ? beamBPtr : beamAPtr),
          iSys, id1, id2, e1, e2, PDFscale, PDFscale);
        // For trial branchings that have multiple flavour combinations
        // (gluon conversion) check which trial flavour was picked and
        // store PDF ratio for that flavour.
        int    trialFlav    = trialGenPtr->trialFlav();
        double pdfRatioFlav = trialGenPtr->getTrialPDFratio();

        // Set color factor for trial.
        double colFac = getAntFunPtr(antFunTypePhys)->chargeFac();
        int nF        = getNf(iRegion);
        if (antFunTypePhys == XGsplitIF) colFac *= min(nF,nGluonToQuarkF);

        // Effective renormalization-scale prefactor.
        double kR = aSkMu2EmitI;
        if (antFunTypePhys == QXsplitII || antFunTypePhys == QXsplitIF)
          kR = aSkMu2SplitI;
        else if (antFunTypePhys == GXconvII || antFunTypePhys == GXconvIF)
          kR = aSkMu2Conv;
        else if (antFunTypePhys == XGsplitIF) kR = aSkMu2SplitF;
        kR = sqrt(kR);

        // Check if we should use running alphaS.
        bool runAlpha = (alphaSorder >= 1);
        // If we are close to lambdaQCD, use constant instead.
        if (qMinNow < 2.5*alphaSptr->Lambda3()/kR) runAlpha = false;

        // Check if we should try to get rid of c and b quarks,
        // iRegion 2 = mc to mb, iRegion 3 = mb to mtb.
        int idCheck = -1;
        // Flavours to check.
        for (int i = 0; i < (5 - nFlavZeroMass); i++) {
          if ((abs(id1) == 5 - i) && (iRegion == 3 - i)) {
            if ((antFunTypePhys == QXsplitII && !swapped) ||
                (antFunTypePhys == QXsplitIF) ) idCheck = 5 - i;
          }
          if (isII && (abs(id2) == 5-i) && (iRegion == 3-i)) {
            if (antFunTypePhys == QXsplitII && swapped) idCheck = 5-i;
          }
        }
        bool usePDFmassThreshold = (idCheck > 0);

        // Enhancements (biased kernels).
        bool   doEnhance  = false;
        double enhanceFac = 1.0;
        if (qTrial > enhanceCutoff) {
          if (isHardSys[iSys] && enhanceInHard) doEnhance = true;
          else if (!isHardSys[iSys] && partonSystemsPtr->hasInAB(iSys) &&
            enhanceInMPI) doEnhance = true;
          if (doEnhance) {
            enhanceFac *= enhanceAll;
            // At the trial level, all gluon splittings and
            // conversions enhanced by max(enhanceCharm,
            // enhanceBottom).
            if (min(nF,nGluonToQuarkF) >= 4 && antFunTypePhys == XGsplitIF)
              enhanceFac *= max(enhanceCharm, enhanceBottom);
            else if ( nGluonToQuarkI >= 4 &&
              (antFunTypePhys == GXconvII || antFunTypePhys == GXconvIF))
              enhanceFac *= max(enhanceCharm,enhanceBottom);
          }
        }

        // Sanity check for zero branching probability.
        if (colFac < NANO || headroomFac < NANO) {
          double qTmp = qTrial;
          qTrial = 0.0;
          trialPtr->saveTrial(indx,qTmp,qTrial,0.,0.,0.,0.,pdfRatioFlav,
            trialFlav, 0.,0.,0.);
        }

        // Mass treatment, use PDFs mass thresholds with constant alphaS.
        else if (usePDFmassThreshold) {
          double qTmp = qTrial;
          // Add extra headroom, should really be multiplying trial PDF
          // ratio.
          headroomFac *= (antFunTypePhys == QXsplitII ? 2.0 : 1.3);
          // Overestimate.
          double mu2eff    = mu2min + pow2(kR*qMinNow);
          // alphaS for overestimate.
          double facAlphaS = min(alphaSmax, alphaSptr->alphaS(mu2eff));
          if (alphaSorder == 0) facAlphaS = alphaSvalue;
          // Generate new q value, with constant alphaS.
          double q2trial = trialGenPtr->genQ2thres(pow2(qTrial), s12,
            zMinNow, zMaxNow, colFac, facAlphaS, pdfRatio, id1, id2,
            e1, e2, true, headroomFac, enhanceFac);
          qTrial = sqrt(q2trial);
          double massNow = (idCheck == 4 ? mc : mb);
          double extraMassPDFfactor = log(q2trial/pow2(massNow));
          // Trial information.
          trialPtr->saveTrial(indx, qTmp, qTrial, zMinNow, zMaxNow, colFac,
            facAlphaS, pdfRatioFlav, trialFlav, extraMassPDFfactor,
            headroomFac, enhanceFac);
          if (verbose >= DEBUG) printOut(__METHOD_NAME__,
              "Using vanishing pdfs towards the mass threshold for id1 " +
              num2str(id1) + " and id2 " + num2str(id2));

        // AlphaS running inside trial integral.
        } else if (runAlpha) {
          double qTmp = qTrial;
          // One-loop beta function (two-loop imposed by veto, below).
          double b0 = (33.0 - 2.0*nF) / (12.0 * M_PI);
          // Use 3-flavour Lambda for overestimate.
          double lambdaEff = alphaSptr->Lambda3();
          // Generate new q value, with alphaS running inside trial integral.
          double q2trial = trialGenPtr->genQ2run(pow2(qTrial), s12,
            zMinNow, zMaxNow, colFac, pdfRatio, b0, kR, lambdaEff,
            e1, e2, headroomFac, enhanceFac);
          qTrial = sqrt(q2trial);
          // Save trial information.
          double muEff    = max(1.01, kR*qTrial/lambdaEff);
          double alphaEff = 1.0/b0/log(pow2(muEff));
          trialPtr->saveTrial(indx, qTmp, qTrial, zMinNow, zMaxNow, colFac,
            alphaEff, pdfRatioFlav, trialFlav, 1.0, headroomFac, enhanceFac);
        // AlphaS outside trial integral.
        } else {
          double qTmp = qTrial;
          // Constant alphaS.
          double facAlphaS = ( (alphaSorder >= 1) ? alphaSmax
            : alphaSvalue );
          // Generate new q value, with constant alphaS.
          double q2trial = trialGenPtr->genQ2(pow2(qTrial), s12,
            zMinNow, zMaxNow, colFac, facAlphaS, pdfRatio, e1, e2,
            headroomFac, enhanceFac);
          qTrial = sqrt(q2trial);
          // Save trial information.
          trialPtr->saveTrial(indx, qTmp, qTrial, zMinNow, zMaxNow, colFac,
            facAlphaS,pdfRatioFlav,trialFlav,1.0,headroomFac,enhanceFac);
        }

        // Check evolution window boundaries.
        if (qTrial > qMinNow) {
          // Do preliminary accept/reject in smaller Z hull.
          double zMinPhys = trialGenPtr->getZmin(pow2(qTrial), s12, e1,
            is1A ? eBeamAUsed : eBeamBUsed);
          double zMaxPhys = trialGenPtr->getZmax(pow2(qTrial), s12, e1,
            is1A ? eBeamAUsed : eBeamBUsed);
          // Note: can insert tighter x < 1 boundaries here too.
          double IzTrial = trialGenPtr->getIz(zMinNow,zMaxNow);
          double IzPhys  = trialGenPtr->getIz(zMinPhys,zMaxPhys);
          double p = IzPhys/IzTrial;
          // If there is no big difference, don't bother.
          if (p > 0.99) acceptRegion = true;
          else {
            // Update Z limits.
            trialPtr->zMinSav[indx] = zMinPhys;
            trialPtr->zMaxSav[indx] = zMaxPhys;
            // Accept narrower limits with probability IzPhys/IzTrial.
            if (rndmPtr->flat() < p) acceptRegion = true;
          }
        }
        else if (qMinNow < qWin || qMinNow < qTrialMax) {
          if (verbose >= DEBUG) printOut(__METHOD_NAME__,
              "stopping evolution, already found scale that is bigger");
          acceptRegion = true;
          trialPtr->renewTrial(indx);
          qTrial = 0.0;
        } else if (iRegion == 0 || qTrial < cutoffScale) {
          acceptRegion = true;
          qTrial       = 0.0;
        } else {
          qTrial = qMinNow;
          iRegion--;
        }

      } // End loop over regions.
      if ((qTrial > qTrialMax) && (qTrial > cutoffScale)) {
        qTrialMax = qTrial;
        indxMax   = indx;
      }

      // Check for rescue mechanism in case trial gets stuck.
      if (doRescue && abs(qBeginNow-qTrial) < rescueMin)
        trialPtr->addRescue(indx);
    } // End loop over trial generators.

    // Check if trial wins.
    if (qTrialMax >= qWin || qWin <= 0.0) {
      winnerPtr = trialPtr;
      qWin      = qTrialMax;
      indxWin   = indxMax;
      iSysWin   = iSys;
    }
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "qTrialMax = " << qTrialMax;
      if (indxMax >= 0)
        ss <<" (" << trialPtr->trialGenPtrsSav[indxMax]->name() << ")";
      if (iAnt + 1 == nAnt) ss << " final qWin = ";
      else ss << " current qWin = ";
      ss << qWin <<" i1 i2 = " << winnerPtr->i1sav << " " << winnerPtr->i2sav
         << " in system " << iSysWin;
      printOut(__METHOD_NAME__, ss.str());
    }
  } // End loop over antennae.

  // If non-zero branching scale found: continue.
  if ((qWin > qEndAll) && (qWin > 0.0)) {
    if (verbose >= DEBUG && winnerPtr != nullptr) {
      stringstream ss;
      ss<<"Winner at scale qWin = ";
      ss << qWin << " trial type "
         << winnerPtr->trialGenPtrsSav[indxWin]->name();
      printOut(__METHOD_NAME__, ss.str());
      if (verbose >= DEBUG && winnerPtr != nullptr) {
        ss.str("pdf ratio ");
        ss << winnerPtr->getPDFratioTrial(indxWin)
           << " col1 = " << event[winnerPtr->i1sav].col()
           << " acol1 = " << event[winnerPtr->i1sav].acol()
           << " col2 = " << event[winnerPtr->i2sav].col()
           << " acol2 = " << event[winnerPtr->i2sav].acol()
           << " in System " << iSysWin;
      }
    }
  } else {
    qWin = 0.0;
    // Check if qEnd < all cutoffs.
    if (qEndsmallerCutoff && verbose >= DEBUG) {
      printOut(__METHOD_NAME__, "=== All trials now below cutoff "
        "qEndAll = " + num2str(qEndAll, 3) + ".");
      printOut(__METHOD_NAME__,"Final configuration was:");
      event.list();
    }
  }

  // Check if we have a heavy quark in the initial state left.
  bool forceSplitNow = false;
  if ((!allSkipped) && heavyQuarkLeft(max(qWin,qEndAll))) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__,
        "Going to force a splitting! qWin was " + num2str(qWin) +
        " with id1 = " + num2str(winnerPtr->id1sav) +
        " with id2 = " + num2str(winnerPtr->id2sav));
    // Set scale to the corresponding mass.
    qWin = winnerPtr->scaleSav[indxWin];
    winnerPtr->forceSplittingSav = true;
    forceSplitNow = true;
  } else if (winnerPtr != nullptr) winnerPtr->forceSplittingSav = false;
  if (verbose >= DEBUG) {
    list();
    printOut(__METHOD_NAME__, "end", dashLen);
  }

  // Diagnostics.
  if (verbose >= REPORT) diagnosticsPtr->stop(__METHOD_NAME__);

  if (qWin > pTevolBegAll && !forceSplitNow) {
    infoPtr->errorMsg("Warning in "+__METHOD_NAME__
      +": Generated scale > pTevolBegAll. Returning 0.");
    return 0.;
  }

  // Make sure we get the next branching.
  if ((qWin > 0.0) && forceSplitNow) return (pTevolEndAll-NANO);
  return qWin;

}

//--------------------------------------------------------------------------

// Perform a branching (as defined by current "winner").

bool VinciaISR::branch(Event& event) {

  // System of index of the winner and extract current QE scales.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Diagnostics
  if (verbose >= REPORT) diagnosticsPtr->start(__METHOD_NAME__);

  int iTrial     = indxWin;
  double qNew    = winnerPtr->getTrialScale(iTrial);
  double q2new   = pow2(qNew);
  enum AntFunType antFunTypePhys = winnerPtr->antFunTypePhys(iTrial);
  bool isII      = winnerPtr->isII();
  bool is1A      = winnerPtr->is1A();
  bool isSwapped = (isII ? winnerPtr->getIsSwapped(iTrial) : false);
  // Set to false for IF because there it was always guy 1 who did
  // gluon splitting/conversion in the initial state.

  // Count up global number of attempted trials.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Processing Branching at scale Q = " << qNew;
    printOut(__METHOD_NAME__, ss.str());
  }

  // Recoiler, allow for both II and IF to have global recoils
  vector<Vec4> recoilers;
  vector<int> iRecs;
  for (int j = 0; j < partonSystemsPtr->sizeOut(iSysWin); ++j) {
    int ip = partonSystemsPtr->getOut(iSysWin, j);
    if (ip != winnerPtr->i1sav && ip != winnerPtr->i2sav) {
      recoilers.push_back(event[ip].p());
      iRecs.push_back(partonSystemsPtr->getOut(iSysWin,j));
    }
  }

  // Check if we have to force a splitting (to get rid of heavy quarks).
  bool forceSplitting = false;
  if (winnerPtr->forceSplittingSav) {
    forceSplitting = true;
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Forcing a splitting at q = " << qNew;
      printOut(__METHOD_NAME__, ss.str());
    }
    // If so, try to force the splitting at the scale we'd like to.
    double e1    = winnerPtr->e1sav;
    double q2max =
      winnerPtr->trialGenPtrsSav[iTrial]->getQ2max(winnerPtr->sAnt(),
        e1, is1A ? eBeamAUsed : eBeamBUsed);
    if (q2max < q2new) {
      q2new = q2max;
      qNew  = sqrt(q2new);
      winnerPtr->scaleSav[iTrial] = 0.99*qNew;
      if (verbose >= DEBUG ) {
        stringstream ss;
        ss << "adjusted scale to q = " << qNew;
        printOut(__METHOD_NAME__,ss.str());
      }
    }
  }

  // Generate full trial kinematics (and reject if outside phase-space).
  if (!generateKinematics(event, winnerPtr, recoilers)) {
    // Mark this trial as "used", will need to generate a new one..
    winnerPtr->renewTrial(iTrial);
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Branching outside of phase space.");
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(kinematics)");
    return false;
  }
  // If no recoilers for this branching, e.g. local IF map, zero iRecs,
  if (recoilers.size() <= 0) iRecs.clear();

  // If we have a force splitting update scale information might have
  // change to ensure kinematics can be generated.
  if (forceSplitting) {
    qNew  = winnerPtr->getTrialScale(iTrial);
    q2new = pow2(qNew);
  }

  // Assign colour flow. Note, using lastColTag here, if branching
  // accepted we tell Pythia.
  bool usedColTag = assignColourFlow(event, winnerPtr);

  // For the sector shower veto branching here if outside of sector.
  stateNew.clear();
  minClus = VinciaClustering();
  if (sectorShower) {
    // Create vector of all post-branching particles and vector of
    // indices of particles to replace in current state. In local
    // kinematic maps both are only mothers, in global maps
    // also all recoilers.
    vector<Particle> ptclsPost = {winnerPtr->new1,winnerPtr->new2,
                                  winnerPtr->new3};
    vector<int> iOld = {winnerPtr->i1sav, winnerPtr->i2sav};
    for (int i(0); i<(int)iRecs.size(); ++i) {
      // Append index of recoiling particle to list of old particles.
      iOld.push_back(iRecs.at(i));
      // Append recoiling particle to list of post-branching particles.
      ptclsPost.push_back(event[iOld.back()]);
      // Change its momentum.
      ptclsPost.back().p(recoilers.at(i));
    }

    // Get tentative post-branching state.
    stateNew = vinComPtr->makeParticleList(iSysWin, event, ptclsPost, iOld);

    // Save clustering and compute sector resolution for it.
    enum AntFunType antFunTypeWin = winnerPtr->antFunTypePhys(indxWin);
    VinciaClustering thisClus;
    // Set children correctly.
    int indA, indB;
    if (isII) {
      indA = isSwapped ? 2 : 0;
      indB = isSwapped ? 0 : 2;
    }
    else {
      bool is2Initial = !ptclsPost.at(2).isFinal();
      indA = is2Initial ? 2 : 0;
      indB = is2Initial ? 0 : 2;
    }
    thisClus.setChildren(ptclsPost,indA,1,indB);
    thisClus.setMothers(winnerPtr->id1sav,winnerPtr->id2sav);
    thisClus.setAntenna(false,antFunTypeWin);
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
      // Mark this trial as "used", will need to generate a new one.
      winnerPtr->renewTrial(iTrial);
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, "Branching rejected (outside of sector).");
      if (verbose >= REPORT)
        diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(sector)");
      return false;
    }
  }

  // Check whether phase-space is closed for getting rid of heavy quarks.
  vector<Particle> parts = vinComPtr->makeParticleList(iSysWin, event);
  if (!forceSplitting)
    if (!checkHeavyQuarkPhaseSpace(parts,iSysWin)) {
      if (verbose >= DEBUG) printOut(__METHOD_NAME__,
          "branching rejected because phase space after branching "
          "does not allow forced splittings");
      // Mark this trial as "used", will need to generate a new one.
      winnerPtr->renewTrial(iTrial);
      if (verbose >= REPORT)
        diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(HQPS)");
      return false;
    }

  // Check if this branching is above cutoff scale (but don't say no
  // to getting rid of a massive flavour).
  double cutoffScale = (isII ? cutoffScaleII : cutoffScaleIF);
  // Check only if we don't force a splitting.
  if (!forceSplitting && sqrt(q2new) < cutoffScale) {
    bool isMassiveQsplit = false;
    if (antFunTypePhys == QXsplitIF)
      isMassiveQsplit = (abs(winnerPtr->id1sav) > nFlavZeroMass);
    else if (antFunTypePhys == QXsplitII) {
      isMassiveQsplit = ( isSwapped
        ? (abs(winnerPtr->id2sav) > nFlavZeroMass)
        : (abs(winnerPtr->id1sav) > nFlavZeroMass) );
    }

    // Reject and mark as "used", will need to generate a new trial.
    if (!isMassiveQsplit) {
      winnerPtr->renewTrial(iTrial);
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__,"Branching is below cutoff: reject.");
      if (verbose >= REPORT)
        diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(cutoff)");
      return false;
    }
  }

  // Veto step, decide whether to accept or reject branching, skip for
  // forced splitting.
  if (!forceSplitting) {
    if (!acceptTrial(event, winnerPtr)) {
      // Mark this trial as "used", will need to generate a new one.
      winnerPtr->renewTrial(iTrial);
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__,"Trial rejected (failed acceptTrial)");
      if (verbose >= REPORT)
        diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(acceptTrial)");
      return false;
    }
  }

  // Put new particles into event record, store a copy of event, to be
  // used if branching vetoed by userHooks.
  Event evtOld   = event;
  int sizeOld    = event.size();
  int i1sav      = winnerPtr->i1sav;
  int i2sav      = winnerPtr->i2sav;
  winnerPtr->new1.scale(qNew);
  winnerPtr->new2.scale(qNew);
  winnerPtr->new3.scale(qNew);
  int iNew1      = event.append(winnerPtr->new1);
  int iNew3      = event.append(winnerPtr->new3);
  int iNew2      = event.append(winnerPtr->new2);
  // Check for recoilers from II (or IF with global recoil map) branchings.
  vector< pair<int,int> > iRecNew; iRecNew.clear(); iRecNew.resize(0);
  if (iRecs.size() >= 1)
    for (int j = 0; j < event.size(); ++j)
      if (event[j].isFinal())
        for (int k = 0; k < (int)iRecs.size(); k++)
          // Copy recoiler change momentum.
          if (iRecs[k] == j) {
            int inew = event.copy(j,44);
            event[inew].p(recoilers[k]);
            iRecNew.push_back(make_pair(iRecs[k],inew));
          }
  // Update event pointers if necessary.
  event.restorePtrs();
  // Check if we went from polarised to unpolarised state If so,
  // depolarise parton state. A more complete alternative here would
  // be to create depolarised copies of all partons and then update
  // everything, but deemed unnecessary for now.
  if ( event[i1sav].pol() != 9 && event[i2sav].pol() != 9 &&
       (event[iNew1].pol() == 9 || event[iNew2].pol() == 9 ||
        event[iNew3].pol() == 9)) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Depolarizing parton state");

    // Depolarise parton state (except the branching mothers, which
    // will be replaced by unpolarised daughters).
    int sizeSystem = partonSystemsPtr->sizeOut(iSysWin);
    for (int i = 0; i < sizeSystem; ++i) {
      int i1 = partonSystemsPtr->getOut( iSysWin, i);
      // Skip if not present in final state.
      if ( i1 <= 0 || !event[i1].isFinal()) continue;
      // Skip if mother parton to be replaced by daughter.
      if ( i1 == winnerPtr->i1sav || i1 == winnerPtr->i2sav) continue;
      // Else depolarise.
      if ( event[i1].pol() != 9 ) event[i1].pol(9);
    }
    // Also make sure all daughters are depolarised as well.
    winnerPtr->new1.pol(9);
    winnerPtr->new2.pol(9);
    winnerPtr->new3.pol(9);
  }

  // Set mothers and daughters, mark original dipole partons as branched.
  event[i1sav].statusNeg();
  event[i2sav].statusNeg();
  if (isII) {
    // New initial partons inherit mothers (beam).
    event[iNew1].mothers(event[i1sav].mother1(), event[i1sav].mother2());
    event[iNew3].mothers(event[i2sav].mother1(), event[i2sav].mother2());
    // Gluon emission.
    if (event[iNew2].id() == 21) {
      // iNew1 a inherits mothers of A, daughters are A and j.
      event[iNew1].daughters(iNew2, i1sav);
      // iNew3 b inherits mothers of B, daughters are B and j.
      event[iNew3].daughters(iNew2, i2sav);
      // iNew2 j gets a and b as mothers, no daughters.
      // Ensure mother1 is the one that changed colour (collinear mother),
      // used eg to determine vertex structure in HepMC output.
      if (event[iNew3].col() == event[i2sav].col() && event[iNew3].acol()
        == event[i2sav].acol()) event[iNew2].mothers(iNew1, iNew3);
      else event[iNew2].mothers(iNew3, iNew1);

    // Gluon splitting or conversion in the initial state: side A.
    } else if (!isSwapped) {
      // iNew1 a inherits mothers of A, daughters are A and j.
      event[iNew1].daughters(iNew2, i1sav);
      // iNew3 b inherits mothers of B, daughter is B.
      event[iNew3].daughters(i2sav, 0);
      // iNew2 j gets a as mother, no daughters.
      event[iNew2].mothers(iNew1, 0);

    // Gluon splitting or conversion in the initial state: side B.
    } else {
      // iNew1 a inherits mothers of A, daughter is A.
      event[iNew1].daughters(i1sav, 0);
      // iNew3 b inherits mothers of B, daughters are B and j.
      event[iNew3].daughters(iNew2, i2sav);
      // iNew2 j gets b as mother, no daughters.
      event[iNew2].mothers(iNew3 ,0);
    }
    // i1sav A keeps its daughters, gets a as mother.
    event[i1sav].mothers(iNew1, 0);
    // i2sav B keeps its daughters, gets b as mother.
    event[i2sav].mothers(iNew3, 0);
    // iNew2 j has no daughters.
    event[iNew2].daughters(0, 0);
    // Put a and b as daughters of the beam for hard process.
    if (isHardSys[iSysWin]) {
      bool founda = false;
      bool foundb = false;
      for (int i=0; i<(int)event.size(); i++) {
        if (!founda)
          if (event[i].daughter1() == i1sav) {
            event[i].daughters(iNew1, 0);
            founda = true;
          }
        if (!foundb)
          if (event[i].daughter1() == i2sav) {
            event[i].daughters(iNew3, 0);
            foundb = true;
          }
        if (founda && foundb) break;
      }
    }
  } else {
    // New initial parton inherits mothers (beam).
    event[iNew1].mothers(event[i1sav].mother1(), event[i1sav].mother2());
    // Gluon emission.
    if (event[iNew2].id()==21) {
      // iNew1 a inherits mothers of A, daughters are j and A.
      event[iNew1].daughters(iNew2, i1sav);
      // i2sav K gets j and k as daughters, keeps its mothers.
      event[i2sav].daughters(iNew2, iNew3);
      // iNew3 k gets K as mother, no daughters.
      event[iNew3].mothers(i2sav, 0);
      // iNew2 j gets a and K as mothers, no daughters.
      // Ensure mother1 is the one that changed colour (collinear mother),
      // used eg to determine vertex structure in HepMC output.
      if (event[i1sav].col() == event[iNew1].col() &&
        event[i1sav].acol() == event[iNew1].acol())
        event[iNew2].mothers(i2sav, iNew1);
      else event[iNew2].mothers(iNew1, i2sav);

    // Gluon splitting or conversion in the initial state
    } else if (antFunTypePhys == QXsplitIF || antFunTypePhys == GXconvIF) {
      // iNew1 a inherits mothers of A, daughters are A and j.
      event[iNew1].daughters(iNew2, i1sav);
      // i2sav K gets k as daughter, keeps its mothers.
      event[i2sav].daughters(iNew3, 0);
      // iNew3 k gets K as mother, no daughters.
      event[iNew3].mothers(i2sav, 0);
      // iNew2 j gets a as mother, no daughters.
      event[iNew2].mothers(iNew1, 0);

    // Gluon splitting in the final state
    } else {
      // iNew1 a inherits mothers of A, daughteris A.
      event[iNew1].daughters(i1sav, 0);
      // i2sav K gets k and j as daughters, keeps its mothers.
      event[i2sav].daughters(iNew2, iNew3);
      // iNew3 k gets K as mother, no daughters.
      event[iNew3].mothers(i2sav, 0);
      // iNew2 j gets K as mother, no daughters.
      event[iNew2].mothers(i2sav, 0);
    }
    // i1sav A keeps its daughters, gets a as mother.
    event[i1sav].mothers(iNew1, 0);
    // iNew3 5 has no daughters.
    event[iNew3].daughters(0, 0);
    // iNew2 j has no daughters.
    event[iNew2].daughters(0, 0);
    // Put a as daughter of the beam for hard process.
    if (isHardSys[iSysWin])
      for (int i = 0; i < (int)event.size(); i++)
        if (event[i].daughter1() == i1sav) {
          event[i].daughters(iNew1, 0);
          break;
        }
  }

  // Veto by userHooks, possibility to allow user veto of emission step.
  if (canVetoEmission)
    if (userHooksPtr->doVetoISREmission(sizeOld, event, iSysWin)) {
      event = evtOld;
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, "Branching vetoed by user.");
      if (verbose >= REPORT)
        diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(userVeto)");
      return false;
    }

  // Everything accepted.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Branching accepted.");
    event.list();
    printOut(__METHOD_NAME__, "PartonSystems before update:");
    partonSystemsPtr->list();
  }
  // Update list of systems.
  partonSystemsPtr->replace(iSysWin, i1sav, iNew1);
  partonSystemsPtr->addOut(iSysWin, iNew2);
  partonSystemsPtr->replace(iSysWin, i2sav, iNew3);
  // Initial partons.
  if (isII) {
    partonSystemsPtr->setInA(iSysWin, iNew1);
    partonSystemsPtr->setInB(iSysWin, iNew3);
  } else if (is1A) partonSystemsPtr->setInA(iSysWin, iNew1);
  else partonSystemsPtr->setInB(iSysWin, iNew1);
  // Recoilers (if any).
  for (int k = 0; k < (int)iRecNew.size(); k++)
    partonSystemsPtr->replace(iSysWin, iRecNew[k].first, iRecNew[k].second);
  double shat = (event[partonSystemsPtr->getInA(iSysWin)].p() +
    event[partonSystemsPtr->getInB(iSysWin)].p()).m2Calc();
  partonSystemsPtr->setSHat(iSysWin, shat);

  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "PartonSystems after update:");
    partonSystemsPtr->list();
  }

  // Update beam particles.
  bool isValN1 = false;
  bool isValN3 = false;
  // Update both sides.
  if (isII) {
    // Side A.
    BeamParticle& beam1 = *beamAPtr;
    beam1[iSysWin].update(iNew1, event[iNew1].id(), event[iNew1].e()/eBeamA);
    // Redo choice of companion kind whenever new flavour.
    if (event[i1sav].id() != event[iNew1].id()) {
      double PDFscale = q2new;
      beam1.xfISR(iSysWin,event[iNew1].id(),event[iNew1].e()/eBeamA,
        PDFscale );
      beam1.pickValSeaComp();
    }
    isValN1 = beam1[iSysWin].isValence();
    // Side B.
    BeamParticle& beam2 = *beamBPtr;
    beam2[iSysWin].update(iNew3, event[iNew3].id(), event[iNew3].e()/eBeamB);
    // Redo choice of companion kind whenever new flavour.
    if (event[i2sav].id() != event[iNew3].id()) {
      double PDFscale = q2new;
      beam2.xfISR( iSysWin, event[iNew3].id(), event[iNew3].e()/eBeamB,
        PDFscale );
      beam2.pickValSeaComp();
    }
    isValN3 = beam2[iSysWin].isValence();

  // Update only one side.
  } else {
    BeamParticle& beam = (is1A ? *beamAPtr : *beamBPtr);
    beam[iSysWin].update( iNew1, event[iNew1].id(), event[iNew1].e()
      /(is1A ? eBeamA : eBeamB) );
    // Redo choice of companion kind whenever new flavour.
    if (event[i1sav].id() != event[iNew1].id()) {
      double PDFscale = q2new;
      beam.xfISR( iSysWin, event[iNew1].id(), event[iNew1].e()
        /(is1A ? eBeamA : eBeamB), PDFscale );
      beam.pickValSeaComp();
    }
    isValN1 = beam[iSysWin].isValence();
  }

  // Update antennae due to recoil.
  if (iRecNew.size() >= 1)
    for (int iAnt = 0; iAnt < (int)branchElementals.size(); ++iAnt) {
      BranchElementalISR* antPtr = &branchElementals[iAnt];
      // No recoil outside system.
      if (antPtr->system != iSysWin) continue;
      int i2  = antPtr->i2sav;
      // Check relevant partons for recoilers.
      for (int k = 0; k < (int)iRecNew.size(); k++)
        // i1 is always initial.
        if (i2  == iRecNew[k].first) antPtr->i2sav = iRecNew[k].second;
    }

  // Update number of gluons and quark pairs.
  if (antFunTypePhys == GXconvII || antFunTypePhys == GXconvIF ||
      antFunTypePhys == XGsplitIF) {
    nG[iSysWin] -= 1;
    ++nQQ[iSysWin];
  } else ++nG[iSysWin];

  // Update list of partons and parton numbers.
  indexSav[iSysWin].clear(); indexSav[iSysWin].resize(0);
  int sizeSystem = partonSystemsPtr->sizeAll(iSysWin);
  for (int i = 0; i < sizeSystem; ++i) {
    int i1 = partonSystemsPtr->getAll( iSysWin, i);
    if ( i1 <= 0 ) continue;
    indexSav[iSysWin].push_back( i1 );
    if (!event[i1].isFinal()) {
      if (event[i1].pz() > 0.0) initialA[iSysWin] = event[i1];
      else initialB[iSysWin] = event[i1];
    }
  }
  eBeamAUsed = 0.0;
  eBeamBUsed = 0.0;
  for (map<int,Particle>::iterator it = initialA.begin();
       it != initialA.end(); ++it) {
    int i = it->first;
    eBeamAUsed += initialA[i].e();
    eBeamBUsed += initialB[i].e();
  }
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "Updating dipole-antenna(e)");

  // Updates of the branched antennae. The BranchElemental that
  // winnerPtr points to is no longer needed.  Update it to store new
  // antenna, and add second new antenna if needed.

  // Gluon emission.
  if (winnerPtr->new2.id() == 21) {
    // II -> we created two IF antennae.
    // IF -> we created an IF and an FF antenna (the latter of which is up
    //       to the FSR shower to find and handle).
    // Update old antenna to be iNew1-iNew2 antenna.
    int col = ( (event[iNew1].col() == event[iNew2].col()) ?
      event[iNew2].col() : event[iNew2].acol() );
    winnerPtr->reset(iSysWin,event,iNew1,iNew2,col,isValN1,false);
    resetTrialGenerators(winnerPtr);
    // Update colour. Second parton in new antenna is a gluon; decide whether
    // the new antenna corresponds to its colour or anticolour tag.
    // If this was an II branching -> add the other IF antenna.
    if (isII) {
      col = ((event[iNew3].col() == event[iNew2].col()) ?
        event[iNew2].col() : event[iNew2].acol());
      BranchElementalISR newTrial(iSysWin,event,iNew3,iNew2,col,isValN3,false);
      resetTrialGenerators(&newTrial);
      // Decide whether the new antenna corresponds to the colour or
      // anticolour tag of the newly emitted gluon. Save
      // branchelemental.
      branchElementals.push_back(newTrial);
    }

  // Gluon splitting in the initial state:
  } else if (antFunTypePhys == QXsplitII || antFunTypePhys == QXsplitIF) {
    // Old antenna should now and add an IF antenna. Decide whether
    // this antenna corresponds to gluon colour or anticolour.
    int col;
    if (winnerPtr->isII())
      col = ((event[iNew1].col() == event[iNew3].acol()) ?
        event[iNew3].acol() : event[iNew3].col() );
    else
      col = ((event[iNew1].col() == event[iNew3].col()) ?
        event[iNew3].col() : event[iNew3].acol() );

    // Update old antenna to be iNew1-iNew3 antenna.
    winnerPtr->reset(iSysWin,event,iNew1,iNew3,col,isValN1,isValN3);
    resetTrialGenerators(winnerPtr);
    // Add the other IF antenna.
    int iSplitGluon = (isSwapped ? iNew3 : iNew1);
    col = ((event[iSplitGluon].col() == event[iNew2].col()) ?
      event[iNew2].col() : event[iNew2].acol() );
    BranchElementalISR newTrial(iSysWin,event,iSplitGluon,iNew2,col,
      false,false);
    resetTrialGenerators(&newTrial);
    // Update colour, old1 is gluon, so both col and acol != 0. Save
    // branchelemental,
    branchElementals.push_back(newTrial);

  // Gluon conversion in the initial state.
  } else if (antFunTypePhys == GXconvII || antFunTypePhys == GXconvIF) {
    // Update branched antenna, check IS or FS quark carry ant colour.
    int iFSQ     = iNew2;
    int colFSQ   = ( event[iFSQ].id() > 0 ? event[iFSQ].col()
      : event[iFSQ].acol() );
    int iISQ     = (isSwapped ? iNew3 : iNew1);
    int colISQ   = ( event[iISQ].id() > 0 ? event[iISQ].col()
      : event[iISQ].acol() );
    int iQLeft   = 0;
    int colQLeft = 0;
    bool isValQL = false;
    if (colFSQ == winnerPtr->col()) {
      // In case this is FF ant: kick out later.
      int iPartner = (isSwapped ? iNew1 : iNew3);
      bool isPval  = (isSwapped ? isValN1 : isValN3);
      winnerPtr->reset(iSysWin,event,iPartner,iFSQ,colFSQ,isPval,false);
      // IS quark is left.
      iQLeft   = iISQ;
      colQLeft = colISQ;
      isValQL  = (isSwapped ? isValN3 : isValN1);
    } else if (colISQ == winnerPtr->col()) {
      // Is II or IF ant.
      winnerPtr->reset(iSysWin,event,iNew1,iNew3,colISQ,isValN1,isValN3);
      // FS quark is left.
      iQLeft   = iFSQ;
      colQLeft = colFSQ;
    }
    // Update trial generators.
    resetTrialGenerators(winnerPtr);
    // Find other antenna, where converted gluon took part in, replace
    // gluon with left quark.
    int iConvGluon = (isSwapped ? i2sav : i1sav);

    // Can only find one ant with old gluon, as winner already updated.
    for (int iAnt = 0; iAnt < (int)branchElementals.size(); iAnt++) {
      BranchElementalISR* antPtr = &branchElementals[iAnt];
      // Only look inside same system.
      if (antPtr->system != iSysWin) continue;
      if (antPtr->i1sav == iConvGluon) {
        antPtr->reset(iSysWin,event,antPtr->i2sav,iQLeft,colQLeft,
          antPtr->isVal2(),isValQL);
        resetTrialGenerators(antPtr);
      } else if (antPtr->i2sav == iConvGluon) {
        // New antenna is IF.
        antPtr->reset(iSysWin,event,antPtr->i1sav,iQLeft,colQLeft,
          antPtr->isVal1(),isValQL);
        resetTrialGenerators(antPtr);
      }
    }
  }

  // Gluon splitting in the final state.
  else if (antFunTypePhys == XGsplitIF) {
    // Keep the old antenna (iNew2 as emission and iNew1 as initial
    // partner) which is IF, no new one created. Update colour, old2
    // is quark so if col2 !=0 that's antenna colour
    int col = ( (event[iNew2].col() != 0) ?
      event[iNew2].col() : event[iNew2].acol() );
    winnerPtr->reset(iSysWin,event,iNew1,iNew2,col,isValN1,false);
    // Update trial generators.
    resetTrialGenerators(winnerPtr);
    // And check the other antenna i2sav was involved: has to be IF.
    for (int iAnt = 0; iAnt < (int)branchElementals.size(); ++iAnt) {
      BranchElementalISR* antPtr = &branchElementals[iAnt];
      // Skip antennae not in same system.
      if (antPtr->system != iSysWin) continue;
      // Map i2sav to iNew3.
      if (antPtr->i2sav == i2sav) {
        col = antPtr->col();
        antPtr->reset(iSysWin,event,antPtr->i1sav,iNew3,col,
          antPtr->isVal1(),false);
        resetTrialGenerators(antPtr);
      }
    }
  }

  // Updates of the other antennae.
  for (int iAnt = 0; iAnt < (int)branchElementals.size(); ++iAnt) {
    BranchElementalISR* antPtr = &branchElementals[iAnt];

    // Only look inside same system.
    if (antPtr->system != iSysWin) continue;

    // Update particles.
    int i1    = antPtr->i1sav;
    int i2    = antPtr->i2sav;
    int i1new = i1;
    int i2new = i2;
    bool isVal1new = antPtr->isVal1();
    bool isVal2new = antPtr->isVal2();
    if ((i1 == i1sav) || (i1 == i2sav) || (i2 == i1sav) || (i2 == i2sav)) {
      if (i1 == i1sav) {
        i1new     = iNew1;
        isVal1new = isValN1;
      } else if (i1 == i2sav) {
        i1new     = iNew3;
        isVal1new = isValN3;
      }
      if (i2 == i1sav) {
        i2new     = iNew1;
        isVal2new = isValN1;
      } else if (i2 == i2sav) {
        i2new     = iNew3;
        isVal2new = isValN3;
      }
      int col = antPtr->col();
      antPtr->reset(iSysWin, event, i1new, i2new, col, isVal1new, isVal2new);
      resetTrialGenerators(antPtr);
    }

    // Update.
    i1  = antPtr->i1sav;
    i2  = antPtr->i2sav;

    // Check relevant partons for recoilers.
    for (int k=0; k<(int)iRecNew.size(); k++)
      // i1 is always initial.
      if (i2 == iRecNew[k].first) {
        int i2now   = iRecNew[k].second;
        bool isVal1 = antPtr->isVal1();
        bool isVal2 = antPtr->isVal2();
        int col     = antPtr->col();
        antPtr->reset(iSysWin, event, i1, i2now, col, isVal1, isVal2);
        resetTrialGenerators(antPtr);
      }

    // Reset rescue counter
    antPtr->resetRescue();
  }

  // Sanity check: Kick out any FF antenna we might have created.
  for (int iAnt = 0; iAnt < (int)branchElementals.size(); ++iAnt) {
    BranchElementalISR* antPtr = &branchElementals[iAnt];
    if (event[antPtr->i1sav].isFinal() && event[antPtr->i2sav].isFinal())
      branchElementals.erase(branchElementals.begin() + iAnt);
  }

  // Renew trials in all other systems since pdfs changed.
  for (int iAnt = 0; iAnt < (int)branchElementals.size(); ++iAnt) {
    // Skip same system.
    if (branchElementals[iAnt].system == iSysWin) continue;
    // Reset rescue counter.
    branchElementals[iAnt].resetRescue();
    if (isII || (is1A && branchElementals[iAnt].is1A()) ||
        (!is1A && !branchElementals[iAnt].is1A()))
      branchElementals[iAnt].renewTrial();
  }

  // Count the number of branchings in the system.
  nBranch[iSysWin]++;
  nBranchISR[iSysWin]++;

  // Book-keeping for MECs.
  if (doMECsSys[iSysWin]) {
    // Communicate to MECs class that we succesfully branched.
    mecsPtr->hasBranched(iSysWin);
    // Decide if we should be doing ME corrections for next order.
    doMECsSys[iSysWin] = mecsPtr->doMEC(iSysWin, nBranch[iSysWin]+1);
    // If going from ME corrected order to uncorrected one, renew trials.
    if (!doMECsSys[iSysWin]) {
      for (int i = 0; i < (int)branchElementals.size(); i++) {
        BranchElementalISR* trial = &branchElementals[i];
        if (trial->system == iSysWin) trial->renewTrial();
      }
    }
  }

  if (verbose >= DEBUG && !checkAntennae(event)) {
    infoPtr->errorMsg("Warning in "+__METHOD_NAME__
      +": Failed checkAntennae. Aborting.");
    infoPtr->setAbortPartonLevel(true);
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(checkAntennae)");
    return false;
  }

  // Let Pythia know how many color tags we used.
  if (usedColTag) {
    event.nextColTag();
    if (event[iNew2].id() == 21) {
      int lastTag = event.lastColTag();
      int colMax  = max(event[iNew2].col(),event[iNew2].acol());
      while (colMax > lastTag) lastTag = event.nextColTag();
    }
  }

  // Check the event after each branching.
  if (verbose >= REPORT && !vinComPtr->showerChecks(event, true)){
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+
      ": Failed shower checks. Aborting.");
    infoPtr->setAbortPartonLevel(true);
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(showerChecks)");
    return false;
  }

  // Merging: is this branching a candidate for a merging veto or not?
  if (doMerging && !isTrialShower) {
    // We only want to veto the event based on the first branching.
    // In principle, later emissions could be vetoed as well, but the
    // current treatment assumes that if the first emission is below the
    // merging scale, all subsequent ones are too.
    // This could explicitly be checked by setting nBranchMergingVeto to
    // a large number.
    int nBranchMaxMergingVeto = 1;

    // Merging veto should ignore branchings after the first.
    if (nBranch[iSysWin] > nBranchMaxMergingVeto)
      mergingHooksPtr->doIgnoreStep(true);
  }

  if (verbose >= DEBUG) {
    event.list();
    list();
    printOut(__METHOD_NAME__, "end", dashLen);
  }

  // Diagnostics.
  if (verbose >= REPORT) diagnosticsPtr->stop(__METHOD_NAME__,"accept");

  return true;

}

//--------------------------------------------------------------------------

// Initialise pointers to Vincia objects.

void VinciaISR::initVinciaPtrs(
  VinciaColour* colourPtrIn, shared_ptr<VinciaFSR> fsrPtrIn,
  MECs* mecsPtrIn, Resolution* resolutionPtrIn,
  VinciaCommon* vinComPtrIn, VinciaWeights* vinWeightsPtrIn) {
  colourPtr     = colourPtrIn;
  fsrPtr        = fsrPtrIn;
  mecsPtr       = mecsPtrIn;
  resolutionPtr = resolutionPtrIn;
  vinComPtr     = vinComPtrIn;
  weightsPtr    = vinWeightsPtrIn;
}

//--------------------------------------------------------------------------

// Clear all containers.

void VinciaISR::clearContainers() {
  hasPrepared.clear();
  branchElementals.clear();
  Q2hat.clear();
  isHardSys.clear();
  isResonanceSys.clear();
  polarisedSys.clear();
  doMECsSys.clear();
  indexSav.clear();
  partsSav.clear();
  nBranch.clear();
  nBranchISR.clear();
  nFlavsBorn.clear();
  resolveBorn.clear();
  nG.clear();
  nQQ.clear();
  initialA.clear();
  initialB.clear();
}

//--------------------------------------------------------------------------

// Set starting scale of shower (power vs wimpy) for system iSys.

void VinciaISR::setStartScale(int iSys, Event& event) {

  // Resonance and hadron decay systems: no ISR.
  if (!partonSystemsPtr->hasInAB(iSys)) Q2hat[iSys] = 0.0;

  // Hard Process System.
  else if (isHardSys[iSys]) {
    // Hard system: start at phase-space maximum or factorisation scale.
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Setting ISR starting scale for hard system");
    // pTmaxMatch = 1 : always start at QF (modulo kFudge).
    if (pTmaxMatch == 1) Q2hat[iSys] = pT2maxFudge * infoPtr->Q2Fac();
    // pTmaxMatch = 2 : always start at eCM.
    else if (pTmaxMatch == 2) Q2hat[iSys] = m2BeamsSav;
    // Else check if this event has final-state jets or photons.
    else {
      bool hasRad = false;
      for (int i = 0; i < partonSystemsPtr->sizeOut(iSys); ++i) {
        int idAbs = event[partonSystemsPtr->getOut(iSys,i)].idAbs();
        if (idAbs <= 5 || idAbs == 21 || idAbs == 22) hasRad = true;
        if (idAbs == 6 && nGluonToQuarkF == 6) hasRad = true;
        if (hasRad) break;
      }
      // If no QCD/QED partons detected, allow to go to phase-space maximum.
      if (hasRad) Q2hat[iSys] = pT2maxFudge * infoPtr->Q2Fac();
      else Q2hat[iSys] = m2BeamsSav;
    }
  }

  // MPI systems.
  else {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Setting ISR starting scale of MPI system");
    // Set starting scale for MPI systems: min of incoming parton scales.
    // Find positions of incoming colliding partons.
    int in1 = partonSystemsPtr->getInA(iSys);
    int in2 = partonSystemsPtr->getInB(iSys);
    Q2hat[iSys] = pT2maxFudgeMPI
      * pow2(min(event[in1].scale(),event[in2].scale()));
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,
        "Renewing all trials since we got non-hard system!");
    for (int iAnt = 0; iAnt < (int)branchElementals.size(); ++iAnt)
      if (branchElementals[iAnt].system != iSys)
        branchElementals[iAnt].renewTrial();
  }
}

//--------------------------------------------------------------------------

// Add trial functions to a BranchElemental.

void VinciaISR::resetTrialGenerators(BranchElementalISR* trial) {

  // Reset.
  trial->clearTrialGenerators();
  // Always p+ for II. Always I for IF.
  int id1            = abs(trial->id1sav);
  // Always p- for II. Always F for IF.
  int id2            = abs(trial->id2sav);
  bool isOctetOnium2 = ( (id2>6 && id2!=21) ? true : false );
  bool isVal1        = trial->isVal1();
  bool isVal2        = trial->isVal2();
  bool isII          = trial->isII();
  bool is1A          = trial->is1A();
  enum AntFunType antFunTypePhys = NoFun;
  int colType1abs    = abs(trial->colType1());
  int colType2abs    = abs(trial->colType2());

  // II antennae.
  if (isII) {
    // QQbar.
    if ( colType1abs == 1 && colType2abs == 1 ) {
      antFunTypePhys = QQemitII;
      if (getAntFunPtr(antFunTypePhys)->chargeFac() > 0.)
        trial->addTrialGenerator(antFunTypePhys, false, &trialIISoft);
      antFunTypePhys = QXsplitII;
      if (convQuarkToGluonI &&
        getAntFunPtr(antFunTypePhys)->chargeFac() > 0.) {
        if (!isVal1) trial->addTrialGenerator(antFunTypePhys, false,
          &trialIISplitA);
        if (!isVal2) trial->addTrialGenerator(antFunTypePhys, true,
          &trialIISplitB);
      }
    // GG.
    } else if ( colType1abs == 2 && colType2abs == 2 ) {
      antFunTypePhys = GGemitII;
      if (getAntFunPtr(antFunTypePhys)->chargeFac() > 0.) {
        trial->addTrialGenerator(antFunTypePhys, false, &trialIISoft);
        trial->addTrialGenerator(antFunTypePhys, false, &trialIIGCollA);
        trial->addTrialGenerator(antFunTypePhys, false, &trialIIGCollB);
      }
      antFunTypePhys = GXconvII;
      if (convGluonToQuarkI &&
        getAntFunPtr(antFunTypePhys)->chargeFac() > 0.) {
        trial->addTrialGenerator(antFunTypePhys, false, &trialIIConvA);
        trial->addTrialGenerator(antFunTypePhys, true, &trialIIConvB);
      }
    // QG.
    } else if ( colType1abs == 1 && colType2abs == 2 ) {
      antFunTypePhys = GQemitII;
      if (getAntFunPtr(antFunTypePhys)->chargeFac() > 0.) {
        trial->addTrialGenerator(antFunTypePhys, true, &trialIISoft);
        trial->addTrialGenerator(antFunTypePhys, true, &trialIIGCollB);
      }
      antFunTypePhys = GXconvII;
      if (convGluonToQuarkI && getAntFunPtr(antFunTypePhys)->chargeFac() > 0.)
        trial->addTrialGenerator(antFunTypePhys, true, &trialIIConvB);
      antFunTypePhys = QXsplitII;
      if (convQuarkToGluonI && getAntFunPtr(antFunTypePhys)->chargeFac() > 0.)
        if (!isVal1) trial->addTrialGenerator(antFunTypePhys, false,
          &trialIISplitA);
    // GQ.
    } else if ( colType1abs == 2 && colType2abs == 1 ) {
      antFunTypePhys = GQemitII;
      if (getAntFunPtr(antFunTypePhys)->chargeFac() > 0.) {
        trial->addTrialGenerator(antFunTypePhys, false, &trialIISoft);
        trial->addTrialGenerator(antFunTypePhys, false, &trialIIGCollA);
      }
      antFunTypePhys = GXconvII;
      if (convGluonToQuarkI && getAntFunPtr(antFunTypePhys)->chargeFac() > 0.)
        trial->addTrialGenerator(antFunTypePhys, false, &trialIIConvA);
      antFunTypePhys = QXsplitII;
      if (convQuarkToGluonI && getAntFunPtr(antFunTypePhys)->chargeFac() > 0.)
        if (!isVal2) trial->addTrialGenerator(antFunTypePhys, true,
          &trialIISplitB);
    }

  // IF antennae.
  } else {
    // QQ.
    if ( colType1abs == 1 && colType2abs == 1 ) {
      antFunTypePhys = QQemitIF;
      if (getAntFunPtr(antFunTypePhys)->chargeFac() > 0.) {
        // Use different trial generator for valence quarks.
        if (!isVal1)
          trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFSoft);
        else
          trial->addTrialGenerator(antFunTypePhys, !is1A, &trialVFSoft);
      }
      antFunTypePhys = QXsplitIF;
      if (convQuarkToGluonI && getAntFunPtr(antFunTypePhys)->chargeFac() > 0.)
        if (!isVal1) trial->addTrialGenerator(antFunTypePhys, !is1A,
          &trialIFSplitA);
    // GG.
    } else if ( colType1abs == 2 && colType2abs == 2 ) {
      antFunTypePhys = GGemitIF;
      if (getAntFunPtr(antFunTypePhys)->chargeFac() > 0.) {
        trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFSoft);
        trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFGCollA);
        // For sector shower add additional K-collinear trial generator.
        if (sectorShower)
          trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFGCollK);
      }
      antFunTypePhys = XGsplitIF;
      if (id2 == 21 && nGluonToQuarkF > 0 &&
        getAntFunPtr(antFunTypePhys)->chargeFac()>0.)
        trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFSplitK);
      antFunTypePhys = GXconvIF;
      if (convGluonToQuarkI && getAntFunPtr(antFunTypePhys)->chargeFac() > 0.)
        trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFConvA);
    // GQ.
    } else if ( colType1abs == 2 && colType2abs == 1 ) {
      antFunTypePhys = GQemitIF;
      if (getAntFunPtr(antFunTypePhys)->chargeFac() > 0.) {
        trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFSoft);
        trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFGCollA);
      }
      antFunTypePhys = GXconvIF;
      if (convGluonToQuarkI && getAntFunPtr(antFunTypePhys)->chargeFac() > 0.)
        trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFConvA);
    // QG.
    } else if ( colType1abs == 1 && colType2abs == 2 ) {
      antFunTypePhys = QGemitIF;
      if (getAntFunPtr(antFunTypePhys)->chargeFac() > 0.) {
        if (!isVal1)
          trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFSoft);
        else
          trial->addTrialGenerator(antFunTypePhys, !is1A, &trialVFSoft);
        // For sector shower add additional K-collinear trial generator.
        if (sectorShower)
          trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFGCollK);
      }
      antFunTypePhys = XGsplitIF;
      if (id2 == 21 && nGluonToQuarkF > 0 &&
        getAntFunPtr(antFunTypePhys)->chargeFac()>0.)
        trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFSplitK);
      antFunTypePhys = QXsplitIF;
      if (convQuarkToGluonI && getAntFunPtr(antFunTypePhys)->chargeFac() > 0.)
        if (!isVal1) trial->addTrialGenerator(antFunTypePhys, !is1A,
          &trialIFSplitA);
    // GOctetOnium.
    } else if ( id1 == 21 && isOctetOnium2 ) {
      antFunTypePhys = GXconvIF;
      if (convGluonToQuarkI && getAntFunPtr(antFunTypePhys)->chargeFac() > 0.)
        trial->addTrialGenerator(antFunTypePhys, !is1A, &trialIFConvA);
    // QOctetOnium.
    } else if ( colType1abs == 1 && isOctetOnium2 ) {
      antFunTypePhys = QXsplitIF;
      if (convQuarkToGluonI && getAntFunPtr(antFunTypePhys)->chargeFac() > 0.)
        if (!isVal1) trial->addTrialGenerator(antFunTypePhys, !is1A,
          &trialIFSplitA);
    }
  }

}

//--------------------------------------------------------------------------

// Function to return headroom factor.

double VinciaISR::getHeadroomFac(int iSys, enum AntFunType antFunTypePhys,
  double) {

  // Increase headroom factor when doing ME corrections.
  double headroomFac = 1.0;
  if (doMECsSys[iSys] && mecsPtr->doMEC(iSys,nBranch[iSys] + 1)) {
    headroomFac = 4.;
    // Gluon splitting MECs may require larger overestimates.
    if (antFunTypePhys == XGsplitIF) headroomFac *= 1.5;
    // Helicity-dependent MECs may require larger headroom.
    if (helicityShower && polarisedSys[iSys]) headroomFac *= 1.5;
  // Headroom factors for pure shower.
  }

  return headroomFac;

}

//--------------------------------------------------------------------------

// Method to check if heavy quark left after passing the evolution window.

bool VinciaISR::heavyQuarkLeft(double qTrial) {
  // We are above mb.
  if (qTrial > 1.02*mb) return false;
  bool foundQuark = false;
  // Loop over antennae.
  for (int iAnt = 0; iAnt < (int)branchElementals.size(); ++iAnt) {
    BranchElementalISR* trialPtr = &branchElementals[iAnt];
    int iSys    = trialPtr->system;
    int id1     = abs(trialPtr->id1sav);
    int id2     = abs(trialPtr->id2sav);
    bool foundQuarkNow = false;
    int splitGenTndex  = -1;
    if ( (id1 > nFlavZeroMass) && (id1 < 6) ) {
      double mass = ((id1 == 4) ? mc : mb);
      // Safety of 2%.
      if (qTrial < (1.02*mass)) {
        foundQuarkNow = true;
        // Find the index of the trial generator for splitting.
        for (int indx = 0; indx < (int)trialPtr->nTrialGenerators(); ++indx) {
          if ( (trialPtr->antFunTypePhys(indx) == QXsplitIF) ||
            (trialPtr->antFunTypePhys(indx) == QXsplitII) ) {
            splitGenTndex = indx;
            trialPtr->scaleSav[indx] = mass;
          }
        }
      }
    }
    // Only check parton 2 if this is an II antenna.
    if ( trialPtr->isII() && (id2 > nFlavZeroMass) && (id2 < 6) ) {
      double mass = ((id2 == 4) ? mc : mb);
      // Safety of 2%.
      if (qTrial < (1.02*mass)) {
        foundQuarkNow = true;
        // Find the index of the trial generator for splitting.
        for (int indx = 0; indx < (int)trialPtr->nTrialGenerators(); ++indx) {
          if (trialPtr->antFunTypePhys(indx) == QXsplitII) {
            splitGenTndex = indx;
            trialPtr->scaleSav[indx] = mass;
          }
        }
      }
    }
    if (foundQuarkNow && (splitGenTndex>=0)) {
      winnerPtr  = trialPtr;
      iSysWin    = iSys;
      indxWin    = splitGenTndex;
      foundQuark = foundQuarkNow;
    } else if (foundQuarkNow) {
      if (verbose >= QUIET) {
        infoPtr->errorMsg("Error in "+__METHOD_NAME__+
          ": Found heavy quark but no splitting trial generator.",
          "Not going to force a splitting.");
        trialPtr->list();
        cout << "     Current scale = " << qTrial << endl;
      }
    }
  }
  return foundQuark;

}

//--------------------------------------------------------------------------

// Generate kinematics (II) and set flavours and masses.

bool VinciaISR::generateKinematicsII(Event& event,
  BranchElementalISR* trialPtr, vector<Vec4>& pRec) {

  // Basic info about trial function and scale
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  int iTrial     = indxWin;
  if (iTrial < 0) return false;
  double qNew    = trialPtr->getTrialScale(iTrial);
  double q2new   = pow2(qNew);
  enum AntFunType antFunTypePhys = trialPtr->antFunTypePhys(iTrial);
  bool isSwapped = trialPtr->getIsSwapped(iTrial);

  // Trial generator.
  TrialGeneratorISR* trialGenPtr = trialPtr->trialGenPtrsSav[iTrial];
  int idA = trialPtr->id1sav;
  int idB = trialPtr->id2sav;

  // Force a splitting, set zMin and zMax accordingly.
  bool forceSplitting = trialPtr->forceSplittingSav;
  if (forceSplitting) {
    trialPtr->zMinSav[iTrial] =
      trialPtr->trialGenPtrsSav[iTrial]->getZmin(q2new, trialPtr->sAnt(),
        trialPtr->e1sav, 0.0);
    trialPtr->zMaxSav[iTrial] =
      trialPtr->trialGenPtrsSav[iTrial]->getZmax(q2new, trialPtr->sAnt(),
        trialPtr->e1sav, 0.0);
  }

  // Generate zeta variable, work out saj, sjb and check initial energies.
  double saj, sjb;
  if (!trialPtr->genTrialInvariants(saj, sjb, 0.0, iTrial)) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "z outside physical range, returning.");
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(kinematics)");
    trialPtr->nHull++;
    return false;
  }

  // Check that sab < shh.
  double sAB = trialPtr->sAnt();
  double sab = sAB + saj + sjb;
  if (sab > m2BeamsSav) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "sab > shh = " + num2str(m2BeamsSav)
        + ", returning.");
    trialPtr->nHull++;
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(sab)");
    return false;
  }

  // Set flavors and masses.
  double mj=0.;
  // Flavour for gluon backwards evolving to (anti)quark.
  int idConv = (antFunTypePhys == GXconvII) ? trialPtr->getTrialFlav() : 0;

  // Gluon emission: inherit parent flavors and add middle gluon.
  if (antFunTypePhys == QQemitII || antFunTypePhys == GQemitII ||
      antFunTypePhys == GGemitII) {
    trialPtr->new1.id(idA);
    trialPtr->new2.id(21);
    trialPtr->new3.id(idB);
    trialPtr->new1.m(0.0);
    trialPtr->new2.m(0.0);
    trialPtr->new3.m(0.0);

  // Gluon splitting in the initial state:
  } else if (antFunTypePhys == QXsplitII) {
    // Side A splitting.
    if (!isSwapped) {
      trialPtr->new1.id(21);
      trialPtr->new2.id(-idA);
      trialPtr->new3.id(idB);
      trialPtr->new1.m(0.0);
      // Final-state leg assigned on-shell mass.
      mj = (abs(idA) <= nFlavZeroMass) ? 0.0 :
        particleDataPtr->m0(abs(idA));
      trialPtr->new2.m(mj);
      trialPtr->new3.m(0.0);
    // Side B splitting.
    } else {
      trialPtr->new1.id(idA);
      trialPtr->new2.id(-idB);
      trialPtr->new3.id(21);
      trialPtr->new1.m(0.0);
      // Final-state leg assigned on-shell mass.
      mj = (abs(idB) <= nFlavZeroMass) ? 0.0 :
        particleDataPtr->m0(abs(idB));
      trialPtr->new2.m(mj);
      trialPtr->new3.m(0.0);
    }

  // Gluon conversion in the initial state (idConv contains flavour)
  } else if (antFunTypePhys == GXconvII) {
    // Final-state leg assigned on-shell mass.
    mj = (abs(idConv) <= nFlavZeroMass) ? 0.0 :
      particleDataPtr->m0(abs(idConv));
    // Side A conversion.
    if (!isSwapped) {
      trialPtr->new1.id(idConv);
      trialPtr->new2.id(idConv);
      trialPtr->new3.id(idB);
      trialPtr->new1.m(0.0);
      trialPtr->new2.m(mj);
      trialPtr->new3.m(0.0);
    // Side B conversion.
    } else {
      trialPtr->new1.id(idA);
      trialPtr->new2.id(idConv);
      trialPtr->new3.id(idConv);
      trialPtr->new1.m(0.0);
      trialPtr->new2.m(mj);
      trialPtr->new3.m(0.0);
    }
  }

  // Correct sab.
  double m2j = mj*mj;
  sab = sAB + saj + sjb - m2j;
  if (sab <0.) return false;

  // Check that x < 1 : side A.
  double zA    = sqrt( sAB * (sab - sjb) / sab / (sab - saj) );
  double xaMax = beamAPtr->xMax(iSysWin);
  double eaMax = 0.98 * xaMax * eBeamA;
  double eOldA = event[trialPtr->i1sav].e();
  double eNewA = eOldA / zA;
  if (eNewA > eaMax && !forceSplitting) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "xa > 1, returning.");
    trialPtr->nHull++;
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(xa>1)");
    return false;
  }
  // Check that x < 1 : side B.
  double zB    = sqrt( sAB * (sab - saj) / sab / (sab - sjb) );
  double xbMax = beamBPtr->xMax(iSysWin);
  double ebMax = 0.98 * xbMax * eBeamB;
  double eOldB = event[trialPtr->i2sav].e();
  double eNewB = eOldB / zB;
  if (eNewB > ebMax && !forceSplitting) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "xb > 1, returning.");
    trialPtr->nHull++;
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(xb>1)");
    return false;
  }

  // Lowering saj and sjb until success.
  if ( (eNewA > eaMax || eNewB > ebMax) && forceSplitting) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Forced splitting, lowering saj and sjb.");
    bool take     = false;
    do {
      saj *= 0.9 + 0.1*rndmPtr->flat();
      sjb *= 0.9 + 0.1*rndmPtr->flat();
      sab = sAB + saj + sjb - m2j;
      eNewA  = sqrt(eOldA/eOldB*(sab - saj)/(sab - sjb)*sab)/2.0;
      eNewB  = sqrt(eOldB/eOldA*(sab - sjb)/(sab - saj)*sab)/2.0;
      double eBeamAUsedNow = eBeamAUsed - eOldA + eNewA;
      double eBeamBUsedNow = eBeamBUsed - eOldB + eNewB;
      if ( (eBeamAUsedNow<0.98*eBeamA)
        && (eBeamBUsedNow<0.98*eBeamB) ) take = true;
    } while (!take && saj>0.0 && sjb>0.0);
    q2new = trialGenPtr->getQ2(saj,sjb,sAB);
    qNew = sqrt(q2new);
    trialPtr->scaleSav[iTrial] = qNew;
  }

  // Generate full kinematics for this trial branching.
  // Generate random (uniform) phi angle.
  double phi = 2 * M_PI * rndmPtr->flat();
  // Generate branching kinematics, starting from dipole-antenna parents.
  vector<Vec4> pOld, pNew;
  pOld.push_back(event[trialPtr->i1sav].p());
  pOld.push_back(event[trialPtr->i2sav].p());
  if (!forceSplitting &&
      !vinComPtr->map2to3II(pNew, pRec, pOld, sAB, saj, sjb, sab, phi, m2j)) {
    if (verbose >= DEBUG ) printOut(__METHOD_NAME__, "Failed map2to3II.");
    trialPtr->nHull++;
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(map2to3II)");
    return false;
  }
  // Save momenta.
  trialPtr->new1.p(pNew[0]);
  trialPtr->new2.p(pNew[1]);
  trialPtr->new3.p(pNew[2]);
  // Set default polarizations: unpolarised (may be assigned later).
  trialPtr->new1.pol(9);
  trialPtr->new2.pol(9);
  trialPtr->new3.pol(9);
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Printing pre-branching momenta");
    for (int i = 0; i < 2; i++) cout << "  " << pOld[i];
    printOut(__METHOD_NAME__, "Printing post-branching momenta and recoiler");
    for (int i = 0; i < 3; i++) cout << "  " << pNew[i];
    for (int i = 0; i < (int)pRec.size(); i++) cout << "  " << pRec[i];
  }

  // Sanity check that z < 1.
  if (pNew[0].e() < eOldA || pNew[2].e() < eOldB) {
    if (verbose >= DEBUG ) printOut(__METHOD_NAME__, "Z > 1.");
    trialPtr->nHull++;
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(z>1)");
    return false;
  }

  // Compute and save physical x*PDF ratio.
  double PDFscale = q2new;

  // Side with positive z momentum.
  int idOldA = idA;
  int idNewA = trialPtr->new1.id();
  double xOldA = eOldA / eBeamA;
  double xNewA = eNewA / eBeamA;
  if (verbose >= REPORT && !beamAPtr->insideBounds(xNewA, PDFscale))
    printf("%s::PDFratio {xa,Q2a} outside boundaries\n",
      trialGenPtr->name().c_str());
  double pdfRatioA =
    max(beamAPtr->xfISR(iSysWin,idNewA,xNewA,PDFscale),TINYPDF)
    / max(beamAPtr->xfISR(iSysWin,idOldA,xOldA,PDFscale),TINYPDF);

  // Side with negative z momentum.
  int idOldB = idB;
  int idNewB = trialPtr->new3.id();
  double xOldB = eOldB / eBeamB;
  double xNewB = eNewB / eBeamB;
  if (verbose >= REPORT && !beamBPtr->insideBounds(xNewB, PDFscale))
    printf("%s::PDFratio {xb,Q2b} outside boundaries\n",
      trialGenPtr->name().c_str());
  double pdfRatioB =
    max(beamBPtr->xfISR(iSysWin,idNewB,xNewB,PDFscale),TINYPDF)
    / max(beamBPtr->xfISR(iSysWin,idOldB,xOldB,PDFscale),TINYPDF);

  // Save. Note: colour flow is not assigned here, since this requires
  // knowledge of which is the next global colour tag available in the
  // event.
  trialPtr->addPDF(iTrial, pdfRatioA*pdfRatioB);
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return true;

}

//--------------------------------------------------------------------------

// Generate kinematics (IF) and set flavours and masses.

bool VinciaISR::generateKinematicsIF(Event& event,
  BranchElementalISR* trialPtr, vector<Vec4>& pRec) {

  // Basic info about trial function and scale.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  int iTrial       = indxWin;
  if (iTrial < 0) return false;
  double qNew      = trialPtr->getTrialScale(iTrial);
  double q2new     = pow2(qNew);
  enum AntFunType antFunTypePhys = trialPtr->antFunTypePhys(iTrial);
  bool is1A        = trialPtr->is1A();
  double eBeamUsed = (is1A ? eBeamAUsed : eBeamBUsed);

  // Trial generator.
  TrialGeneratorISR* trialGenPtr = trialPtr->trialGenPtrsSav[iTrial];
  int idA = trialPtr->id1sav;
  int idK = trialPtr->id2sav;

  // Force a splitting.
  bool forceSplitting = trialPtr->forceSplittingSav;
  if (forceSplitting) {
    trialPtr->zMinSav[iTrial] =
      trialPtr->trialGenPtrsSav[iTrial]->getZmin(q2new, trialPtr->sAnt(),
        trialPtr->e1sav, eBeamUsed);
    trialPtr->zMaxSav[iTrial] =
      trialPtr->trialGenPtrsSav[iTrial]->getZmax(q2new, trialPtr->sAnt(),
        trialPtr->e1sav, eBeamUsed);
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__,"Note: this is a forced splitting");
  }

  // Generate zeta variable, work out saj, sjk.
  double saj;
  double sjk;
  double sAK = trialPtr->sAnt();
  bool pass = trialPtr->genTrialInvariants(saj, sjk, eBeamUsed, iTrial);
  if (!pass) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "z outside physical range, returning.");
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(kinematics)");
    trialPtr->nHull++;
    return false;
  }

  // Self-consistency check.
  if (verbose >= NORMAL) {
    double sAntSav = trialPtr->sAnt();
    double sCheck  = 2 * event[trialPtr->i1sav].p()*event[trialPtr->i2sav].p();
    if (abs(sAntSav - sCheck) > NANO) {
      if (verbose >= DEBUG) event.list();
      list();
      cout << " sAK doesn't look preserved sAnt = " << sAntSav << " sCheck = "
           << sCheck << endl;
    }
  }

  // Set flavors and masses.
  // Flavour for gluon backwards evolving to (anti)quark
  int idConv = (antFunTypePhys == GXconvIF) ? trialPtr->getTrialFlav() : 0;
  double mj=0.;
  double mk=0.;
  double mKold= event[trialPtr->i2sav].m();

  // Gluon emission: inherit parent flavors and add middle gluon.
  if (antFunTypePhys == QQemitIF || antFunTypePhys == QGemitIF ||
      antFunTypePhys == GQemitIF || antFunTypePhys == GGemitIF) {
    // Set ID codes.
    trialPtr->new1.id(idA);
    trialPtr->new2.id(21);
    trialPtr->new3.id(idK);
    // Set masses.
    mk = mKold;
    trialPtr->new1.m(0.0);
    trialPtr->new2.m(0.0);
    trialPtr->new3.m(mk);

  // Gluon splitting in the initial state.
  } else if (antFunTypePhys == QXsplitIF) {
    // Set ID codes.
    trialPtr->new1.id(21);
    trialPtr->new2.id(-idA);
    trialPtr->new3.id(idK);
    // Set masses.
    mj = (abs(idA) <= nFlavZeroMass) ? 0.0
      : particleDataPtr->m0(abs(idA));
    mk = mKold;
    trialPtr->new1.m(0.0);
    trialPtr->new2.m(mj);
    trialPtr->new3.m(mk);

  // Gluon conversion in the initial state.
  } else if (antFunTypePhys == GXconvIF) {
    // Set ID codes.
    trialPtr->new1.id(idConv);
    trialPtr->new2.id(idConv);
    trialPtr->new3.id(idK);
    // Set masses.
    mj = (abs(idConv) <= nFlavZeroMass) ? 0.0
      : particleDataPtr->m0(abs(idConv));
    mk = mKold;
    trialPtr->new1.m(0.0);
    trialPtr->new2.m(mj);
    trialPtr->new3.m(mk);
    if (verbose == DEBUG) {
      stringstream ss;
      ss << "Gluon backwards evolving to id = " << idConv
         << " idK = " << idK
         << " mK = " << trialPtr->new3.m();
      printOut(__METHOD_NAME__,ss.str());
    }

  // Gluon splitting in the final state.
  } else if (antFunTypePhys == XGsplitIF) {
    // Set flavor of splitting.
    double nF       = min((int)trialPtr->getColFac(iTrial),nGluonToQuarkF);
    int splitFlavor = int(rndmPtr->flat() * nF) + 1;
    // Check phase space: sQQ = q2new > 4m^2.
    int nFmax       = (int)nF;
    if (q2new > 4.0*pow2(mb)) nFmax = min(nFmax,5);
    else if (q2new > 4.0*pow2(mc)) nFmax = min(nFmax,4);
    else if (q2new > 4.0*pow2(ms)) nFmax = min(nFmax,3);
    else nFmax = min(nFmax,2);
    if (nFmax < splitFlavor) return false;
    trialPtr->new1.id(idA);
    // XG->XQQbar where Q is the emission because a col line emitted.
    if (abs(event[trialPtr->i1sav].col()) ==
        abs(event[trialPtr->i2sav].col()) ) {
      // IF is c-c or ac-ac.
      trialPtr->new2.id( splitFlavor);
      trialPtr->new3.id(-splitFlavor);
    // XG->XQbarQ where Qbar is the emission because a acol line emitted.
    } else {
      trialPtr->new2.id(-splitFlavor);
      trialPtr->new3.id( splitFlavor);
    }
    // Set masses.
    mj = (abs(splitFlavor) <= nFlavZeroMass) ? 0.0
      : particleDataPtr->m0(abs(splitFlavor));
    mk = mj;
    trialPtr->new1.m(event[trialPtr->i1sav].m());
    trialPtr->new2.m(mj);
    trialPtr->new3.m(mk);
  }

  // Generate full kinematics for this trial branching (needed to work
  // out xa, required for PDF weight).
  // Let antenna tell us which kinematics map to use.
  int kineMap = getAntFunPtr(antFunTypePhys)->kineMap();
  // Generate random (uniform) phi angle.
  double phi = 2 * M_PI * rndmPtr->flat();
  // Last invariant.
  double m2j = mj*mj;
  double m2k = mk*mk;
  double m2Kold = mKold*mKold;
  double sak = sAK + sjk - saj + m2j + m2k -m2Kold;

  // Generate branching kinematics, starting from dipole-antenna parents.
  vector<Vec4> pOld, pNew;
  pOld.push_back(event[trialPtr->i1sav].p());
  pOld.push_back(event[trialPtr->i2sav].p());
  // Decide whether to use local map 100% of the time or allow probabilistic
  // selection of global map.
  bool useLocalMap = (kineMap == 1);
  if (kineMap == 2 && antFunTypePhys == XGsplitIF) useLocalMap = true;
  if (saj >= sAK) useLocalMap = true;
  if (!useLocalMap) {
    // Make probabilistic choice between global and local maps.
    double probGlobal = pow2(sAK - saj) / (pow2(sAK + sjk) + pow2(sAK - saj));
    if (rndmPtr->flat() < probGlobal) {
      // Set B momentum (fixed but used for dot products by global map).
      int iB = partonSystemsPtr->getInB(iSysWin);
      if (iB == trialPtr->i1sav) iB = partonSystemsPtr->getInA(iSysWin);
      Vec4 pB = event[iB].p();
      pass = vinComPtr->map2to3IFglobal(pNew, pRec, pOld, pB,
        sAK, saj, sjk, sak, phi, m2Kold, m2j, m2k);
      // Retry with local map if global one fails.
      if (!pass && kineMapIFretry) useLocalMap = true;
    // 1 - probGlobal to select local map.
    } else useLocalMap = true;
  }
  if (useLocalMap) {
    // Local map: no recoilers outside antenna itself.
    pRec.resize(0);
    pass = vinComPtr->map2to3IFlocal(pNew, pOld, sAK, saj, sjk, sak, phi,
      m2Kold, m2j, m2k);
  }
  if (!pass && !forceSplitting) {
    if (verbose >= DEBUG ) printOut(__METHOD_NAME__, "Failed map2to3IF.");
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(map2to3IF)");
    trialPtr->nHull++;
    return false;
  }

  // Check if enough energy is available in beam.
  double eBeam = (is1A ? eBeamA : eBeamB);
  double eBeamUsedNow = eBeamUsed - pOld[0].e() + pNew[0].e();
  if (eBeamUsedNow > 0.98*eBeam) {
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(x>1)");
    trialPtr->nHull++;
    return false;
  }

  // Save momenta.
  trialPtr->new1.p(pNew[0]);
  trialPtr->new2.p(pNew[1]);
  trialPtr->new3.p(pNew[2]);
  // Set default polarizations: unpolarised (may be assigned later).
  trialPtr->new1.pol(9);
  trialPtr->new2.pol(9);
  trialPtr->new3.pol(9);
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Printing pre-branching momenta");
    for (int i = 0; i < 2; i++) cout << "  " << pOld[i];
    printOut(__METHOD_NAME__, "Printing post-branching momenta and recoiler");
    for (int i = 0; i < 3; i++) cout << "  " << pNew[i];
    for (int i = 0; i < (int)pRec.size(); i++) cout << "  " << pRec[i];
  }

  // PDF ratio. Note, in gluon conversion the q2trial is generated
  // with the sum of quark flavours but at the same time a trial
  // flavour is selected and the saved trial PDF ratio only contains
  // that flavour/glue, so we don't need to know the sum here.
  BeamParticle* beamPtr = is1A ? beamAPtr : beamBPtr;
  double PDFscale = q2new;
  int idOld = idA;
  int idNew = trialPtr->new1.id();
  double eOld = pOld[0].e();
  double xOld = eOld / eBeam;
  double eNew = pNew[0].e();
  double xNew = eNew / eBeam;
  if (verbose >= REPORT && !beamPtr->insideBounds(xNew, PDFscale))
    printf("%s::PDFratio {x,Q2} outside boundaries\n",
      trialGenPtr->name().c_str());
  double newPDF   = beamPtr->xfISR(iSysWin,idNew,xNew,PDFscale);
  // Check PDF > 0; otherwise reject trial.
  if (newPDF < 0.) {
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(PDF<0)");
    trialPtr->nHull++;
    return false;
  }
  double oldPDF   = beamPtr->xfISR(iSysWin,idOld,xOld,PDFscale);
  double pdfRatio = max(newPDF,TINYPDF) / max(oldPDF,TINYPDF);

  // Verbose output if old PDF does not make sense.
  if (oldPDF <= 0.0 && verbose >= DEBUG) {
    cout << "  PDF ratio = " << num2str(pdfRatio)
         << " for idOld = " << idOld << " idNew = " << idNew
         << " xOld = " << num2str(xOld) << " xNew = " << num2str(xNew)
         << " qPDF = " << num2str(sqrt(PDFscale))
         << endl;
    cout << "    Numerator = "
         << num2str(beamPtr->xfISR(iSysWin,idNew,xNew,PDFscale))
         << "    Denom = "
         << num2str(beamPtr->xfISR(iSysWin,idOld,xOld,PDFscale))
         << "    iSys = " << iSysWin << " nSys = "
         << partonSystemsPtr->sizeSys() << endl;
  }

  // Save.
  trialPtr->addPDF(iTrial, pdfRatio);

  // Check the energies of the initial guy.
  double eAotherSys = 0.0;
  double eBeamNow   = 0.0;
  // Sum up energy used by other systems.
  if (is1A) {
    for (map<int, Particle>::iterator it = initialA.begin();
         it != initialA.end(); it++) {
      int i = it->first;
      if (i!=iSysWin) eAotherSys+=initialA[i].e();
    }
    eBeamNow = eBeamA;
  // Sum up energy used by other systems.
  } else {
    for (map<int, Particle>::iterator it = initialB.begin();
         it != initialB.end(); it++) {
      int i = it->first;
      if (i!=iSysWin) eAotherSys+=initialB[i].e();
    }
    eBeamNow = eBeamB;
  }
  if ((eNew+eAotherSys) > 0.98*eBeamNow) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,
        "Energy of incoming partons exceed beam energy, returning.");
    // Lowering sjk (splitting orderd in saj) until success.
    if (forceSplitting) {
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, "Forced splitting, lowering sjk.");
      eNew  = (0.98*eBeamNow - eAotherSys);
      sjk = sAK*(eNew-eOld)/eOld;
      // This should not happen.
      if (sjk <= 0.001) {
        if (sjk <= 0.0) sjk = pow(10.0,-10.0);
        if (verbose >= DEBUG) printOut(__METHOD_NAME__,
            "Need to choose sjk = "+  num2str(sjk) + " probably due to MPI");
      }
      eNew = eOld*(sAK+sjk)/sAK;
      sak  = sAK + sjk - saj + m2j + m2k -m2Kold;
      if (sak <= 0.0) {
        sak = 1.0;
        do {
          sak = sak/10.0;
          saj = sAK + sjk - sak + m2j + m2k -m2Kold;
        } while (saj <= sak);
      }
      q2new = trialGenPtr->getQ2(saj,sjk,sAK);
      qNew  = sqrt(q2new);
      trialPtr->scaleSav[iTrial] = qNew;
    } else {
      if (verbose >= REPORT)
        diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(x>1)");
      trialPtr->nHull++;
      return false;
    }
  }

  // Check local hadronization veto(s) inside this BranchElemental.
  // Require all generated final-state invariants at least above
  // lowest physical meson mass (FF invariant is only the 23 one)
  // (could in principle check only colour-connected invariants but
  // here just check all 23 invariants regardless of colour
  // connection.
  // Only if we do not want to force a splitting.
  if (!forceSplitting) {
    double mMin23 = vinComPtr->mHadMin(trialPtr->new2.id(),
      trialPtr->new3.id());
    if (sjk < pow2(1.01*mMin23)) {
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, "=== Branching Vetoed. m23 < 1.01*mMes.");
      if (verbose >= REPORT)
        diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(cutoff)");
      trialPtr->nHadr++;
      return false;
    }
  }

  // Note, colour flow is not assigned here, since this requires
  // knowledge of which is the next global colour tag available in the
  // event.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return true;

}

//--------------------------------------------------------------------------

// Main method to decide whether to accept or reject a trial branching after
// full branching kinematics have been constructed.

bool VinciaISR::acceptTrial(const Event& event, BranchElementalISR* trialPtr) {

  // Basic info about trial function and scale
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  int iTrial = indxWin;
  if (iTrial < 0) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__, "iTrial < 0 ");
    return false;
  }
  double qNew    = trialPtr->getTrialScale(iTrial);
  enum AntFunType antFunTypePhys = trialPtr->antFunTypePhys(iTrial);
  int iSys       = trialPtr->system;
  bool isII      = trialPtr->isII();
  bool isSwapped = (isII ? trialPtr->getIsSwapped(iTrial) : false);
  // Set to false for IF because there it was always guy 1 who did
  // gluon splitting/conversion in the initial state.

  // Trial generator.
  TrialGeneratorISR* trialGenPtr = trialPtr->trialGenPtrsSav[iTrial];

  // Invariants.
  double S12 = trialPtr->sAnt();
  double s1j = trialPtr->s12();
  double sj2 = trialPtr->s23();

  // Helicity of mother partons.
  // First mother is always initial. Second mother is initial for II.
  double hA = event[trialPtr->i1sav].pol();
  double hB = event[trialPtr->i2sav].pol();
  bool isPolarised = polarisedSys[iSys] && (hA != 9 && hB != 9);

  // Compute spin-summed trial antenna*PDF sum including colour and
  // headroom factors. Sum over individual trial terms
  double antPDFtrialSum = 0.0;
  double nTrialTerms    = 0;
  for (int iTerm = 0; iTerm < (int)trialPtr->nTrialGenerators(); ++iTerm) {
    // Only include terms that correspond to the current physical antenna.
    if (trialPtr->antFunTypePhys(iTerm) != antFunTypePhys) continue;
    // Only include terms that correspond to the same side.
    if (isII && trialPtr->getIsSwapped(iTerm) != isSwapped) continue;
    antPDFtrialSum += trialPtr->headroomSav[iTerm]
      * trialPtr->trialGenPtrsSav[iTerm]->aTrial(s1j, sj2, S12)
      * trialPtr->trialPDFratioSav[iTerm]
      * trialPtr->getColFac(iTerm);
    nTrialTerms++;
  }

  // Enhanced kernels. Note, all trial functions for the same
  // antFunTypePhys must use same enhanceFac.
  double enhanceFac = trialPtr->getEnhanceFac(iTrial);
  // If enhancement was applied but branching is below enhancement
  // cutoff, do accept/reject here with probability
  // trial/enhance-trial to get back to unenhanced trial
  // probability. (Trials only enhanced for enhanceFac > 1.)
  if (enhanceFac > 1.0 && qNew <= enhanceCutoff) {
    if ( rndmPtr->flat() > 1./enhanceFac ) {
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, "Trial vetoed at enhancement stage.");
      return false;
    }
    enhanceFac = 1.0;
  }

  // If physical antenna function is mirror of current trial, translate to
  // swapped invariants for antenna-function evaluation.
  double hAant  = hA;
  double hBant  = hB;
  double s1jant = s1j;
  double sj2ant = sj2;
  double m1ant  = trialPtr->new1.m();
  double mjant  = trialPtr->new2.m();
  double m2ant  = trialPtr->new3.m();
  if (isSwapped) {
    hAant  = hB;
    hBant  = hA;
    s1jant = sj2;
    sj2ant = s1j;
    m1ant  = trialPtr->new3.m();
    m2ant  = trialPtr->new1.m();
  }
  // Fill vectors to use for antenna-function evaluation.
  vector<double> invariants {S12, s1jant, sj2ant, trialPtr->s13()};
  // So far ISR is massless.
  vector<double> mNew {m1ant, mjant, m2ant};
  // Parent helicities.
  // TODO: check! we insert doubles to vector<int>
  vector<int> helBef {static_cast<int>(hAant), static_cast<int>(hBant)};
  // Total accept is summed over daughter helicities (selection done below).
  vector<int> helUnpol {9, 9, 9};

  // Compute spin-summed physical antennae (spin selection below).
  // Define pointer antennae.
  AntennaFunctionIX* antFunPtr = getAntFunPtr(antFunTypePhys);
  // Compute spin-summed antenna function.
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "Evaluating antenna function and PDF ratio");
  double helSum;
  // Unpolarised case: ignore parent helicities.
  if (!isPolarised) helSum = antFunPtr->antFun(invariants,mNew);
  // Polarised case: use parent helicities.
  else helSum = antFunPtr->antFun(invariants, mNew, helBef, helUnpol);
  if (helSum < 0.) {
    if (verbose >= REPORT) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+
        "Negative Antenna Function.","antFunType = "+num2str(antFunTypePhys));
    }
    return false;
  }
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__,"helSum  = "+num2str(helSum));

  // Store (partial) helicity sum.
  double antPhys = helSum;
  // Apply color (charge) factor.
  antPhys *= antFunPtr->chargeFac();
  // PDF factor.
  double PDFphys = trialPtr->physPDFratioSav[iTrial];
  // Extra factor from using running of PDFs close to mass threshold.
  double extraMassPDFfactor = trialPtr->extraMassPDFfactorSav[iTrial];
  PDFphys *= extraMassPDFfactor;

  // Starting value for accept probability = Physical/Trial.
  Paccept[0] = antPhys*PDFphys/antPDFtrialSum;
  if (verbose >= NORMAL && Paccept[0] > 1.05 && qNew > 2.0)
    infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": pAccept > 1");
  if (verbose >= DEBUG ||
    (verbose >= REPORT && Paccept[0] > 1.05 && qNew > 2.0) ) {
    if (nTrialTerms == 1) {
      printOut(__METHOD_NAME__, "Single trial:");
      cout << "   TrialGenerator= " << trialGenPtr->name() << endl;
      cout << "   AntTrial/s    = " << setprecision(6)
           << trialGenPtr->aTrial(s1j, sj2, S12) << " * "
           << trialPtr->headroomSav[iTrial] << " (headroom)"
           << " s1j, s2j = " << s1j << " " << sj2
           << " sOld = " << S12 << endl;
      cout << "   AntPhys/s     = " << setprecision(6)
           << antPhys/antFunPtr->chargeFac() << endl;
      cout << "   massPDFfactor = " << setprecision(6) << extraMassPDFfactor
           << endl;
      cout << "   PDFtrial      = " << setprecision(6)
           << trialPtr->trialPDFratioSav[iTrial] << endl;
      cout << "   PDFphys       = " << setprecision(6)
           << PDFphys << endl;
    } else {
      printOut(__METHOD_NAME__,
        "Combination of several trials (winner = "+trialGenPtr->name()+")");
      cout << setw(21) << "Phys"  << " = "<<setprecision(6) << antPhys*PDFphys;
      cout << "     Ant/s = " << setprecision(6)
           << antPhys/antFunPtr->chargeFac()
           << "     PDF = " << setprecision(6) << PDFphys
           << "     ChgFac = " << setprecision(2) << antFunPtr->chargeFac()
           << endl;
      for (int iTerm = 0; iTerm < (int)trialPtr->nTrialGenerators(); ++iTerm) {
        // Only include terms that correspond to the current physical antenna.
        if (trialPtr->antFunTypePhys(iTerm) != antFunTypePhys) continue;
        // Only include terms that correspond to the same side.
        if (isII && trialPtr->getIsSwapped(iTerm) != isSwapped) continue;
        cout << setw(21) << trialPtr->trialGenPtrsSav[iTerm]->name();
        double term = trialPtr->trialGenPtrsSav[iTerm]->aTrial(s1j, sj2, S12)
          * trialPtr->trialPDFratioSav[iTerm] * trialPtr->getColFac(iTerm);
        cout << " = " << setprecision(6) << term;
        cout << "     Ant/s = " << setprecision(6)
             << trialPtr->trialGenPtrsSav[iTerm]->aTrial(s1j, sj2, S12);
        cout << "     PDF = "<< setprecision(6)
             << trialPtr->trialPDFratioSav[iTerm];
        cout << "     ChgFac = " << setprecision(2) <<
          trialPtr->getColFac(iTerm);
        cout << "    ( "<<setprecision(1)<<term/(antPhys*PDFphys)*100.<<"% ) "
             << endl;
      }
    }

    if (isII) cout << "   II isSwapped  = " << bool2str(isSwapped) << endl;
    else cout << "   IF is1A       = " << bool2str(trialPtr->is1A()) << endl;
    cout << "   idOld         = " << trialPtr->id1sav << " "
         << trialPtr->id2sav;
    cout << "   idNew   = " << trialPtr->new1.id() << " "
         << trialPtr->new2.id() << " " << trialPtr->new3.id() << endl;
    cout << "   xOld          = " << setprecision(6) << trialPtr->e1sav/eBeamA;
    if (isII) cout << " " << trialPtr->e2sav/eBeamB;
    cout << "   xNew   = " << trialPtr->new1.e()/eBeamA;
    if (isII) cout << " " << trialPtr->new3.e()/eBeamB;
    cout << endl;
    cout << "   isVal         = " << bool2str(trialPtr->isVal1());
    if (isII) cout  << " " << bool2str(trialPtr->isVal2());
    cout << endl;
    cout << "   Q             = " <<  qNew << endl;
    cout << "   AntPDFtrial   = "  << setprecision(6) << antPDFtrialSum
         << endl;
    cout << "   Paccept       = "  <<  setprecision(6) << Paccept[0] << endl;
    cout << "   Enhancement   = " << setprecision(6) << enhanceFac << endl;
  }

  // Choose helicities for daughter particles (so far only for
  // massless polarised shower).
  double aHel = 0.0;
  if ( isPolarised ) {
    // Generate random number.
    double randHel = rndmPtr->flat() * helSum;
    // Select helicity.
    int hi(0), hj(0), hk(0);
    for (hi = hAant; abs(hi) <= 1; hi -= 2*hAant) {
      for (hk = hBant; abs(hk) <= 1; hk -= 2*hBant) {
        for (hj = hAant; abs(hj) <= 1; hj -= 2*hAant) {
          vector<int> helNow {hi, hj, hk};
          aHel = antFunPtr->antFun(invariants, mNew, helBef, helNow);
          randHel -= aHel;
          if (verbose >= DEBUG) {
            stringstream ss;
            ss<< "antPhys("<< hAant <<" " << hBant
              <<"  -> "<<  hi << " " << hj << " "<< hk << ") = " << aHel
              << " isSwapped = " << isSwapped
              << " sum = " << helSum;
            printOut(__METHOD_NAME__, ss.str());
          }
          if (randHel < 0.) break;
        }
        if (randHel < 0.) break;
      }
      if (randHel < 0.) break;
    }
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "selected" + num2str(int(hAant)) +
        " " + num2str(int(hBant)) + "  -> " + num2str(hi) + " " +
        num2str(hj) + " " + num2str(hk) + ", isSwapped = " +
        bool2str(isSwapped));
    // Assign helicities (taking swapped invariants into account).
    if (!isSwapped) {
      trialPtr->new1.pol(hi);
      trialPtr->new2.pol(hj);
      trialPtr->new3.pol(hk);
    } else {
      trialPtr->new1.pol(hk);
      trialPtr->new2.pol(hj);
      trialPtr->new3.pol(hi);
    }
  // If not polarised.
  } else {
    trialPtr->new1.pol(9);
    trialPtr->new2.pol(9);
    trialPtr->new3.pol(9);
    polarisedSys[iSys] = false;
  }

  // AlphaS, impose default choice. Can differ slighly from trial even
  // when running inside trial integral, due to flavor
  // thresholds. Here, alphaS(mu) is returned directly, with the
  // number of flavors active at mu, whereas the number of flavors in
  // the trial integral is controlled by the value of the trial scale.
  double alphaTrial = trialPtr->getAlphaTrial(iTrial);
  if (std::isnan(alphaTrial) && verbose >= NORMAL) printOut(__METHOD_NAME__,
    "alphaStrial is NaN");
  if (alphaTrial != alphaTrial) printOut(__METHOD_NAME__,
    "alphaStrial is != alphaStrial");
  double mu2 = pow2(qNew);
  double kMu2Usr = aSkMu2EmitI;
  if (antFunTypePhys == XGsplitIF) kMu2Usr = aSkMu2SplitF;
  else if (antFunTypePhys == QXsplitIF || antFunTypePhys == QXsplitII)
    kMu2Usr = aSkMu2SplitI;
  else if (antFunTypePhys == GXconvIF || antFunTypePhys == GXconvII)
    kMu2Usr = aSkMu2Conv;
  double mu2Usr    = max(mu2min, mu2freeze + mu2*kMu2Usr);
  // alphaS values.
  double alphaSusr = min(alphaSmax, alphaSptr->alphaS(mu2Usr));
  if (verbose >= DEBUG ||
      (verbose >= REPORT && alphaTrial < alphaSusr)) {
    stringstream ss;
    ss << "alphaTrial = " << alphaTrial << ", alphaSusr = " << alphaSusr
       << " at q = " << qNew << ", Trial Generator " << trialGenPtr->name();
    printOut(__METHOD_NAME__, ss.str() );
  }

  // Reweight central accept probability with user choice.
  Paccept[0] *= alphaSusr / alphaTrial;

  // Check number of partons added since Born.
  int branchOrder = 1;
  int showerOrder = branchOrder
    + partonSystemsPtr->sizeOut(iSys) - mecsPtr->sizeOutBorn(iSys);

  // Matrix element corrections.
  // Initialise flag to decide if we want to do a MEC.
  bool doMEC  = doMECsSys[iSys];
  double pMEC = 1.0;
  if (doMEC) {
    // DEBUG output.
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Trying matrix element correction for system "
         << iSysWin << " (" << nBranch[iSysWin]+1 << ". branching).";
      printOut(__METHOD_NAME__, ss.str());
    }

    // NOTE: stateNew and minClus are member variables of VinciaISR
    // and set in VinciaISR::branch(). Different to VinciaFSR!
    //TODO implement a method getMEC() in VinciaISR? Or move to MECs class?
    pMEC = fsrPtr->getMEC(iSysWin, event, stateNew, minClus);
  }
  Paccept[0] *= pMEC;

  // Print MC violations.
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, " Trial Paccept = " + num2str(Paccept[0]));
  bool violation  = (Paccept[0] > 1.0 + NANO);
  bool negPaccept = (Paccept[0] < 0.0);
  if (violation || negPaccept) {
    if (verbose >= REPORT ) {
      stringstream ss;
      if (doMEC) ss << "Paccept (shower and ME, order = ";
      else ss << "Paccept (shower, order = ";
      ss << showerOrder <<" ) = " << Paccept[0] << " at q = " << qNew
         << " in Trial Generator " << trialGenPtr->name();
      printOut(__METHOD_NAME__, ss.str());
      if (extraMassPDFfactor != 1.0)
        printOut(__METHOD_NAME__, " == Note: PDFs close to mass threshold");
      if (verbose >= REPORT || Paccept[0] > 100.0) {
        if (doMEC) cout <<" MEC factor = " << pMEC << endl;
        cout << (isII ? " AB -> ajb = " : " AK -> ajk = ") << trialPtr->id1sav
             << " " << trialPtr->id2sav << " -> " << trialPtr->new1.id() << " "
             << trialPtr->new2.id() << " " << trialPtr->new3.id() << " maj = "
             << sqrt(m2(trialPtr->new1.p()+trialPtr->new2.p()))
             << (isII ? " mjb" :
               " mjk")<<" = "<<sqrt(m2(trialPtr->new2.p()+trialPtr->new3.p()))
             << (isII ? " mAB" : " mAK") << " = " << sqrt(S12)<<endl;
      }
      if (verbose >= DEBUG) trialPtr->list();
      if (Paccept[0] > 100.0) {
        int    nResc        = trialPtr->getNshouldRescue(iTrial);
        double PDFphysTrial = trialPtr->physPDFratioSav[iTrial]/
          trialPtr->trialPDFratioSav[iTrial];
        if (nResc>0 || PDFphysTrial>100.0) {
          if (nResc>0) cout << " Had " << nResc << " pTnext ~ pTold,";
          else
            cout << " Headroom = " << trialPtr->headroomSav[iTrial]
                 << ", PDFphys/ PDFtrial = " << PDFphysTrial << endl;
        }
      }
    }
  }

  // TODO: uncertainty bands.

  // Accept/Reject step.
  // Enhance factors < 1 (radiation inhibition) treated by modifying pAccept.
  if (rndmPtr->flat() > min(1.0,enhanceFac)*Paccept[0]) {

    // Enhancement.
    if (enhanceFac != 1.0)
      weightsPtr->scaleWeightEnhanceReject(Paccept[0],enhanceFac);

    // Count up number of vetoed branchings.
    trialPtr->nVeto++;
    if (verbose >= DEBUG ) {
      printOut(__METHOD_NAME__, "Branching vetoed.");
      if (verbose >= DEBUG) {
        stringstream ss;
        ss << "Paccept = " << Paccept[0] << " Enhancefac = " << enhanceFac;
        printOut(__METHOD_NAME__, ss.str());
      }
    }
    if (verbose >= REPORT)
      diagnosticsPtr->stop(__METHOD_NAME__,"trialVeto(pAccept)");
    return false;
  }
  else if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "Trial accepted.");

  // Enhancement.
  if (enhanceFac != 1.0)
    weightsPtr->scaleWeightEnhanceAccept(enhanceFac);
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end ", dashLen);
  return true;

}

//--------------------------------------------------------------------------

// Paint a trialPtr with colour-flow information,
// according to the type of branching.

bool VinciaISR::assignColourFlow(Event& event, BranchElementalISR* trialPtr) {

  // Basic info about trial function and old partons.
  bool usedColTag = false;
  int iTrial      = indxWin;
  enum AntFunType antFunTypePhys = trialPtr->antFunTypePhys(iTrial);
  bool isSwapped  = trialPtr->getIsSwapped(iTrial);
  int colOld      = trialPtr->col();
  int col1        = event[trialPtr->i1sav].col();
  int acol1       = event[trialPtr->i1sav].acol();
  int col2        = event[trialPtr->i2sav].col();
  int acol2       = event[trialPtr->i2sav].acol();

  // Gluon emission.
  if (trialPtr->new2.id() == 21) {
    double s12 = trialPtr->new1.p()*trialPtr->new2.p();
    double s23 = trialPtr->new2.p()*trialPtr->new3.p();
    // Who inherits the colour.
    bool inh12 = colourPtr->inherit01(s12,s23);
    // Note, use lastColTag here, if branching accepted we tell Pythia
    // that we used one.
    int nTag   = event.lastColTag()+1;
    usedColTag = true;
    // Check other colourtag for gluons.
    int colL   = 0;
    int colR   = 0;
    if (trialPtr->colType1sav == 2) {
      if (colOld == col1) colL = event[trialPtr->i1sav].acol();
      else                colL = event[trialPtr->i1sav].col();
    }
    if (trialPtr->colType2sav == 2) {
      if (colOld == col2) colR = event[trialPtr->i2sav].acol();
      else                colR = event[trialPtr->i2sav].col();
    }
    int nextTag = 10*int(nTag/10) + 10;
    int colNew  = nextTag + int(colOld%10 + rndmPtr->flat()*8)%9 + 1;
    // new1 inherits colour.
    if (inh12) {
      while (colNew%10 == colR%10)
        colNew = nextTag + int(colOld%10 + rndmPtr->flat()*8)%9 + 1;
      trialPtr->new1.acol(acol1);
      trialPtr->new1.col(col1);
      trialPtr->new2.acol(colOld == col1 ? colNew : colOld);
      trialPtr->new2.col(colOld == col1 ? colOld : colNew);
      trialPtr->new3.acol(colOld == acol2 ? colNew : acol2);
      trialPtr->new3.col(colOld == acol2 ? col2 : colNew);
    // new3 inherits colour.
    } else {
      while (colNew%10 == colL%10)
        colNew = nextTag + int(colOld%10 + rndmPtr->flat()*8)%9 + 1;
      trialPtr->new1.acol(colOld == col1 ? acol1 : colNew);
      trialPtr->new1.col(colOld == col1 ? colNew : col1);
      trialPtr->new2.acol(colOld == col1 ? colOld : colNew);
      trialPtr->new2.col(colOld == col1 ? colNew : colOld);
      trialPtr->new3.acol(acol2);
      trialPtr->new3.col(col2);
    }

  // Gluon splitting in the initial state: side A.
  } else if ((antFunTypePhys == QXsplitII && !isSwapped) ||
    antFunTypePhys == QXsplitIF) {
    // II new1 G -> old1 Q/QB (enters hard process) + new2 QB/Q and
    //    new3   -> old2 as recoiler
    // Note, use lastColTag here, if branching accepted we tell
    // Pythia that we used one.
    int nTag   = event.lastColTag()+1;
    usedColTag = true;
    // Splitting quark -> antiquark in FS.
    if (colOld == col1) {
      trialPtr->new1.acol(nTag);
      trialPtr->new1.col(col1);
      trialPtr->new2.acol(nTag);
      trialPtr->new2.col(0);
    // Splitting antiquark -> quark in FS.
    } else {
      trialPtr->new1.acol(acol1);
      trialPtr->new1.col(nTag);
      trialPtr->new2.acol(0);
      trialPtr->new2.col(nTag);
    }
    trialPtr->new3.acol(acol2);
    trialPtr->new3.col(col2);
  }

  // Gluon splitting in the initial state: side B.
  else if (antFunTypePhys == QXsplitII && isSwapped) {
    // II new3 G -> old2 Q/QB (enters hard process) + new2 QB/Q and
    //    new1   -> old1 as recoiler
    // Note, use lastColTag here, if branching accepted we tell
    // Pythia that we used one.
    int nTag   = event.lastColTag()+1;
    usedColTag = true;
    // Splitting quark -> antiquark in FS.
    if (colOld == col2) {
      trialPtr->new2.acol(nTag);
      trialPtr->new2.col(0);
      trialPtr->new3.acol(nTag);
      trialPtr->new3.col(col2);
    // Splitting antiquark -> quark in FS.
    } else {
      trialPtr->new2.acol(0);
      trialPtr->new2.col(nTag);
      trialPtr->new3.acol(acol2);
      trialPtr->new3.col(nTag);
    }
    trialPtr->new1.acol(acol1);
    trialPtr->new1.col(col1);
  }

  // Gluon convsersion in the initial state: side A.
  else if ((antFunTypePhys == GXconvII && !isSwapped) ||
    antFunTypePhys == GXconvIF) {
    // II new1 Q/QB -> old1 G (enters hard process) + new2 Q/QB and
    //    new3      -> old2 as recoiler
    // Quark.
    if (trialPtr->new2.id() > 0) {
      trialPtr->new1.acol(0);
      trialPtr->new1.col(col1);
      trialPtr->new2.acol(0);
      trialPtr->new2.col(acol1);
    // Antiquark.
    } else {
      trialPtr->new1.acol(acol1);
      trialPtr->new1.col(0);
      trialPtr->new2.acol(col1);
      trialPtr->new2.col(0);
    }
    trialPtr->new3.acol(acol2);
    trialPtr->new3.col(col2);
  }

  // Gluon conversion in the initial state: side B.
  else if (antFunTypePhys == GXconvII && isSwapped) {
    // II new3 Q/QB -> old2 G (enters hard process) + new2 Q/QB and
    //    new1      -> old1 as recoiler
    // Quark.
    if (trialPtr->new2.id() > 0) {
      trialPtr->new2.acol(0);
      trialPtr->new2.col(acol2);
      trialPtr->new3.acol(0);
      trialPtr->new3.col(col2);
      // Anitquark.
    } else {
      trialPtr->new2.acol(col2);
      trialPtr->new2.col(0);
      trialPtr->new3.acol(acol2);
      trialPtr->new3.col(0);
    }
    trialPtr->new1.acol(acol1);
    trialPtr->new1.col(col1);
  }

  // Gluon splitting in the final state.
  else if ( antFunTypePhys == XGsplitIF) {
    // IF old2 G -> new3 QB/Q + new2 Q/QB and
    //    new1   -> old1 as recoiler
    // Quark.
    if (trialPtr->new2.id() > 0) {
      trialPtr->new2.acol(0);
      trialPtr->new2.col(col2);
      trialPtr->new3.acol(acol2);
      trialPtr->new3.col(0);
    // Splitting acol line -> emission is antiquark.
    } else {
      trialPtr->new2.acol(acol2);
      trialPtr->new2.col(0);
      trialPtr->new3.acol(0);
      trialPtr->new3.col(col2);
    }
    trialPtr->new1.acol(acol1);
    trialPtr->new1.col(col1);
  }
  return usedColTag;

}

//--------------------------------------------------------------------------

// Method to check if a gluon splitting in the initial state (to get
// rid of heavy quarks) is still possible after the current branching.

bool VinciaISR::checkHeavyQuarkPhaseSpace(vector<Particle> parts, int) {

  vector<int> isToCheck; isToCheck.resize(0);
  for (int i = 0; i < (int)parts.size(); i++)
    if (!parts[i].isFinal() && parts[i].idAbs() > nFlavZeroMass &&
        parts[i].idAbs() < 6) isToCheck.push_back(i);

  // Loop over partons to check.
  for (int i = 0; i < (int)isToCheck.size(); i++) {
    Particle heavyQuark = parts[isToCheck[i]];
    int hQcol           = ( (heavyQuark.col() == 0) ?
      heavyQuark.acol() : heavyQuark.col() );
    double mass         = ((heavyQuark.idAbs() == 4) ? mc : mb);
    bool is1A           = (heavyQuark.pz() > 0.0);
    // Find the colour partner.
    for (int j = 0; j < (int)parts.size(); j++)
      if (j != isToCheck[i]) {
        if ( (parts[j].col() != hQcol) && (parts[j].acol() != hQcol) )
          continue;
        Particle colPartner = parts[i];
        double sHqCp        = m2(heavyQuark, colPartner);
        double Q2max        = 0.0;
        if (colPartner.isFinal())
          Q2max = trialIFSplitA.getQ2max(sHqCp, colPartner.e(), is1A ?
            eBeamAUsed : eBeamBUsed);
        else
          Q2max = trialIISplitA.getQ2max(sHqCp, colPartner.e(), is1A ?
            eBeamAUsed : eBeamBUsed);
        // Phase space limit is below the mass.
        if (sqrt(Q2max) < mass) return false;
        if (colPartner.isFinal()) {
          // Check for energy exceeding beam energy.
          double eA       = heavyQuark.e();
          double eamax    = ( is1A ? (0.98*eBeamA - (eBeamAUsed-eA)) :
            (0.98*eBeamB - (eBeamBUsed-eA)) );
          // Allowed maxima.
          double sjkmax   = sHqCp*(eamax-eA)/eA;
          // sajmin = mass^2.
          double sakmax   = (sHqCp+sjkmax-(mass*mass));
          // Invariant > 0.5 GeV.
          if ( (sjkmax < 0.5) || (sakmax < 0.5) ) return false;
        }
      }
  }
  return true;

}

//--------------------------------------------------------------------------

// Check the antennae.

bool VinciaISR::checkAntennae(const Event& event) {

  map<int,int> nIIAntInSys;
  map<int,int> nIFAntInSys;
  for (vector<BranchElementalISR >::iterator ibrancher =
         branchElementals.begin(); ibrancher!= branchElementals.end();
       ++ibrancher) {
    int i1 = ibrancher->geti1();
    int i2 = ibrancher->geti2();
    int iSysNow = ibrancher->getSystem();
    int inA = 0;
    int inB = 0;

    if (!partonSystemsPtr->hasInAB(iSysNow)) {
      stringstream ss;
      ss << "iSysNow = " << iSysNow;
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+
        ": No incoming particles in system.",ss.str());
      return false;
    } else{
      inA = partonSystemsPtr->getInA(iSysNow);
      inB = partonSystemsPtr->getInB(iSysNow);
    }
    if (inA <= 0 || inB <= 0 ) {
      stringstream ss;
      ss << "iSysNow = " << iSysNow
         << ". inA = " << inA << " inB = " << inB;
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+
        ": Non-positive incoming particles in system.", ss.str());
      return false;
    }

    // Initialise counters for systems we haven't seen yet.
    if (nIIAntInSys.find(iSysNow)==nIIAntInSys.end())
      nIIAntInSys[iSysNow] = 0;
    if (nIFAntInSys.find(iSysNow)==nIFAntInSys.end())
      nIFAntInSys[iSysNow] = 0;
    if (ibrancher->isII()) {
      if (i1 != inA) {
        stringstream ss;
        ss << "iSysNow = "<<iSysNow<<". i1  = " << i1;
        infoPtr->errorMsg("Error in "+__METHOD_NAME__+
          ": i1 not incoming in system.", ss.str());
        return false;
      } else if (i2 != inB) {
        stringstream ss;
        ss << "iSysNow = "<<iSysNow<<". i2  = " << i2;
        infoPtr->errorMsg("Error in "+__METHOD_NAME__+
          ": i2 not incoming in system.", ss.str());
        return false;
      }
      nIIAntInSys[iSysNow]++;
      // Otherwise IF.
    } else {
      if (!event[i2].isFinal()) {
        stringstream ss;
        ss << "iSysNow = "<<iSysNow<<". i2  = " << i2;
        infoPtr->errorMsg("Error in "+__METHOD_NAME__+
          ": i2 not outgoing in system.", ss.str());
        return false;
      }
      // IF with 1 = I from A.
      if (ibrancher->is1A() && i1 != inA) {
        stringstream ss;
        ss << "iSysNow = "<<iSysNow<<". i1  = " << i1;
        infoPtr->errorMsg("Error in "+__METHOD_NAME__+
          ": i1 not incoming from A in system.", ss.str());
        return false;
        // IF with 1 = I from B.
      } else if (!ibrancher->is1A() && i1 != inB) {
        stringstream ss;
        ss << "iSysNow = "<<iSysNow<<". i1  = " << i1;
        infoPtr->errorMsg("Error in "+__METHOD_NAME__+
          ": i1 not incoming from B in system.", ss.str());
        return false;
      }
      nIFAntInSys[iSysNow]++;
    }
  }

  // Check number of initial antennae in each system matches the color
  // type of incoming partons.
  for (int iSysTest = 0; iSysTest < partonSystemsPtr->sizeSys(); ++iSysTest) {
    if (!partonSystemsPtr->hasInAB(iSysTest)) continue;
    int inA = partonSystemsPtr->getInA(iSysTest);
    int inB = partonSystemsPtr->getInB(iSysTest);
    // Count number of antenna ends expected based on colour type of
    // incoming particles.
    int nEndsExpected = abs(event[inA].colType())
      + abs(event[inB].colType());
    // Count number of antennae we have.
    int nAntEnds = 0;
    // II contribute to two ends.
    if (nIIAntInSys.find(iSysTest)!=nIIAntInSys.end())
      nAntEnds += 2*nIIAntInSys[iSysTest];
    if (nIFAntInSys.find(iSysTest)!=nIFAntInSys.end())
      nAntEnds += nIFAntInSys[iSysTest];
    if (nAntEnds < nEndsExpected) {
      stringstream ss;
      ss << "iSys = " << iSysTest;
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+
        ": Too few initial antennae in system",ss.str());
      cout << "colType A: " << event[inA].colType()
           << " colType B: " << event[inB].colType()
           << " nEnds: "  << nAntEnds
           << " nEnds expected: "  << nEndsExpected
           << " nII: " << nIIAntInSys[iSysTest]
           << " nIF: " << nIFAntInSys[iSysTest]
           << endl;
      return false;
    } else if (nAntEnds > nEndsExpected) {
      stringstream ss;
      ss << "iSys = " << iSysTest;
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+
        ": Too many initial antennae in system.",ss.str());
      cout << "colType A: " << event[inA].colType()
           << " colType B: " << event[inB].colType()
           << " nEnds: "  << nAntEnds
           << " nEnds expected: "  << nEndsExpected
           << " nII: " << nIIAntInSys[iSysTest]
           << " nIF: " << nIFAntInSys[iSysTest]
           << endl;
      return false;
    }
  }
  return true;

}

//--------------------------------------------------------------------------

// Save flavour content of system in Born state.

void VinciaISR::saveBornState(Event& born, int iSys) {
  // Initialise.
  resolveBorn[iSys] = false;
  map<int, int> nFlavours;
  for (int i(-6); i<=6; ++i) {
    if (i == 0) nFlavours[21] = 0;
    nFlavours[i] = 0;
  }

  // We want to resolve the Born only when we have a non-QCD coupling in Born.
  int nNonQCD = 0;
  for (int i(0); i<partonSystemsPtr->sizeAll(iSys); ++i) {
    Particle* partonPtr = &born[partonSystemsPtr->getAll(iSys, i)];
    if (partonPtr->isGluon()) nFlavours[21]++;
    else if (partonPtr->isQuark()) {
      int idNow = partonPtr->isFinal() ? partonPtr->id() : -partonPtr->id();
      nFlavours[idNow]++;
    }
    else ++nNonQCD;
  }

  // If there are non-QCD partons in the system, resolve Born.
  if (nNonQCD > 0) {
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
        + " without resolving the Born configuration.");
  }
}

//--------------------------------------------------------------------------

// Save flavour content in Born state for trial shower (in merging).

void VinciaISR::saveBornForTrialShower(Event& born) {
  // Index of system we do the trial shower for.
  // Note: will always be 0 for ISR in merging.
  int iSysTrial = 0;

  // Initialise.
  resolveBorn[iSysTrial] = false;
  map<int, int> nFlavours;
  for (int i(-6); i<=6; ++i) {
    if (i == 0) nFlavours[21] = 0;
    nFlavours[i] = 0;
  }

  // We want to resolve the Born only when we have a non-QCD coupling in Born.
  int nNonQCD = 0;
  for (int i(3); i<born.size(); ++i) {
    Particle* partonPtr = &born[i];
    if (partonPtr->isGluon()) nFlavours[21]++;
    else if (partonPtr->isQuark()) {
      int idNow = partonPtr->isFinal() ? partonPtr->id() : -partonPtr->id();
      nFlavours[idNow]++;
    }
    else ++nNonQCD;
  }

  // If there are non-QCD partons in the system, resolve Born.
  if (nNonQCD > 0) {
    resolveBorn[iSysTrial] = true;
    nFlavsBorn[iSysTrial] = nFlavours;
  }

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
        + " without resolving the Born configuration.");
  }
}

//--------------------------------------------------------------------------

// Print a list of II and IF dipole-antennae.

void VinciaISR::list() const {
  for (int iAnt = 0; iAnt < (int)branchElementals.size(); ++iAnt)
    if (branchElementals.size() == 1) branchElementals[iAnt].list(true, true);
    else if ( iAnt == 0 ) branchElementals[iAnt].list(true, false);
    else if ( iAnt == int(branchElementals.size()) - 1 )
      branchElementals[iAnt].list(false, true);
    else branchElementals[iAnt].list();
}

//==========================================================================

} // end namespace Pythia8
