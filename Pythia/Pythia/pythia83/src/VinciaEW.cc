// VinciaEW.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for Vincia's EW
// shower class and related auxiliary methods. Main author is Rob
// Verheyen.

#include "Pythia8/VinciaEW.h"

namespace Pythia8 {

using namespace VinciaConstants;

//==========================================================================

// Class calculating amplitudes.

//--------------------------------------------------------------------------

// Initialize with maps.

void AmpCalculator::init(EWParticleData* dataIn,
  unordered_map< pair<int, int>, vector<pair<int, int> > >* cluMapFinalIn,
  unordered_map< pair<int, int>, vector<pair<int, int> > >* cluMapInitialIn) {
  if (!isInitPtr) return;

  // Store maps.
  dataPtr = dataIn;
  cluMapFinal = cluMapFinalIn;
  cluMapInitial = cluMapInitialIn;
  if (dataPtr == nullptr) return;
  isInit = true;

  // Set some global EW constants.
  mw  = dataPtr->mass(24);
  mw2 = pow2(mw);
  double mz(dataPtr->mass(23)), mh(dataPtr->mass(25)), cw(mw/mz);
  sw2 = 1. - pow2(cw);
  sw  = sqrt(sw2);
  verbose = settingsPtr->mode("Vincia:verbose");

  // Store the Breit-Wigner matching mode.
  bwMatchMode = settingsPtr->mode("Vincia:bwMatchingMode");

  // Set EW event weight to one.
  eventWeightSave = 1;

  // Get coupling constants - indices are i and j.
  // Photon.
  vMap[make_pair(1,22)] = -1./3.; vMap[make_pair(11,22)] = -1.;
  aMap[make_pair(1,22)] = 0;      aMap[make_pair(11,22)] = 0;
  vMap[make_pair(2,22)] = 2./3.;  vMap[make_pair(12,22)] = 0;
  aMap[make_pair(2,22)] = 0;      aMap[make_pair(12,22)] = 0;
  vMap[make_pair(3,22)] = -1./3.; vMap[make_pair(13,22)] = -1.;
  aMap[make_pair(3,22)] = 0;      aMap[make_pair(13,22)] = 0;
  vMap[make_pair(4,22)] = 2./3.;  vMap[make_pair(14,22)] = 0;
  aMap[make_pair(4,22)] = 0;      aMap[make_pair(14,22)] = 0;
  vMap[make_pair(5,22)] = -1./3.; vMap[make_pair(15,22)] = -1.;
  aMap[make_pair(5,22)] = 0;      aMap[make_pair(15,22)] = 0;
  vMap[make_pair(6,22)] = 2./3.;  vMap[make_pair(16,22)] = 0;
  aMap[make_pair(6,22)] = 0;      aMap[make_pair(16,22)] = 0;

  // Z.
  vMap[make_pair(1,23)] = -(1. - (4./3.)*sw2)/4./sw/cw;
  aMap[make_pair(1,23)] = -1./4./sw/cw;
  vMap[make_pair(2,23)] =  (1. - (8./3.)*sw2)/4./sw/cw;
  aMap[make_pair(2,23)] =  1./4./sw/cw;
  vMap[make_pair(3,23)] = -(1. - (4./3.)*sw2)/4./sw/cw;
  aMap[make_pair(3,23)] = -1./4./sw/cw;
  vMap[make_pair(4,23)] =  (1. - (8./3.)*sw2)/4./sw/cw;
  aMap[make_pair(4,23)] =  1./4./sw/cw;
  vMap[make_pair(5,23)] = -(1. - (4./3.)*sw2)/4./sw/cw;
  aMap[make_pair(5,23)] = -1./4./sw/cw;
  vMap[make_pair(6,23)] =  (1. - (8./3.)*sw2)/4./sw/cw;
  aMap[make_pair(6,23)] =  1./4./sw/cw;
  vMap[make_pair(11,23)] = -(1. - 4.*sw2)/4./sw/cw;
  aMap[make_pair(11,23)] = -1./4./sw/cw;
  vMap[make_pair(12,23)] =  1./4./sw/cw;
  aMap[make_pair(12,23)] = 1./4./sw/cw;
  vMap[make_pair(13,23)] = -(1. - 4.*sw2)/4./sw/cw;
  aMap[make_pair(13,23)] = -1./4./sw/cw;
  vMap[make_pair(14,23)] =  1./4./sw/cw;
  aMap[make_pair(14,23)] = 1./4./sw/cw;
  vMap[make_pair(15,23)] = -(1. - 4.*sw2)/4./sw/cw;
  aMap[make_pair(15,23)] = -1./4./sw/cw;
  vMap[make_pair(16,23)] =  1./4./sw/cw;
  aMap[make_pair(16,23)] =  1./4./sw/cw;

  // W.
  double cW(-1/sqrt(8)/sw);
  vMap[make_pair(1,24)] = cW; vMap[make_pair(11,24)] = cW;
  aMap[make_pair(1,24)] = cW; aMap[make_pair(11,24)] = cW;
  vMap[make_pair(2,24)] = cW; vMap[make_pair(12,24)] = cW;
  aMap[make_pair(2,24)] = cW; aMap[make_pair(12,24)] = cW;
  vMap[make_pair(3,24)] = cW; vMap[make_pair(13,24)] = cW;
  aMap[make_pair(3,24)] = cW; aMap[make_pair(13,24)] = cW;
  vMap[make_pair(4,24)] = cW; vMap[make_pair(14,24)] = cW;
  aMap[make_pair(4,24)] = cW; aMap[make_pair(14,24)] = cW;
  vMap[make_pair(5,24)] = cW; vMap[make_pair(15,24)] = cW;
  aMap[make_pair(5,24)] = cW; aMap[make_pair(15,24)] = cW;
  vMap[make_pair(6,24)] = cW; vMap[make_pair(16,24)] = cW;
  aMap[make_pair(6,24)] = cW; aMap[make_pair(16,24)] = cW;

  // Higgs (note that a mass factor is not included here).
  gMap[make_pair(1,25)] = 1/mw/2./sw; gMap[make_pair(6,25)]  = 1/mw/2./sw;
  gMap[make_pair(2,25)] = 1/mw/2./sw; gMap[make_pair(11,25)] = 1/mw/2./sw;
  gMap[make_pair(3,25)] = 1/mw/2./sw; gMap[make_pair(13,25)] = 1/mw/2./sw;
  gMap[make_pair(4,25)] = 1/mw/2./sw; gMap[make_pair(15,25)] = 1/mw/2./sw;
  gMap[make_pair(5,25)] = 1/mw/2./sw;

  // Bosonic couplings.
  gMap[make_pair(24, 22)]  = 1;      gMap[make_pair(23, -24)] = cw/sw;
  gMap[make_pair(24, 23)]  = cw/sw;  gMap[make_pair(23, 25)]  = mz/cw/sw;
  gMap[make_pair(-24, 22)] = -1;     gMap[make_pair(24, 25)]  = mw/sw;
  gMap[make_pair(-24, 23)] = -cw/sw; gMap[make_pair(-24, 25)] = mw/sw;
  gMap[make_pair(22, -24)] = 1;
  gMap[make_pair(25, 25)]  = 3.*pow2(mh)/2./mw/sw;

  // CKM matrix.
  vCKM[make_pair(1,2)] = settingsPtr->parm("StandardModel:Vud");
  vCKM[make_pair(2,1)] = settingsPtr->parm("StandardModel:Vud");
  vCKM[make_pair(1,4)] = settingsPtr->parm("StandardModel:Vcd");
  vCKM[make_pair(4,1)] = settingsPtr->parm("StandardModel:Vcd");
  vCKM[make_pair(1,6)] = settingsPtr->parm("StandardModel:Vtd");
  vCKM[make_pair(6,1)] = settingsPtr->parm("StandardModel:Vtd");
  vCKM[make_pair(3,2)] = settingsPtr->parm("StandardModel:Vus");
  vCKM[make_pair(2,3)] = settingsPtr->parm("StandardModel:Vus");
  vCKM[make_pair(3,4)] = settingsPtr->parm("StandardModel:Vcs");
  vCKM[make_pair(4,3)] = settingsPtr->parm("StandardModel:Vcs");
  vCKM[make_pair(3,6)] = settingsPtr->parm("StandardModel:Vts");
  vCKM[make_pair(6,3)] = settingsPtr->parm("StandardModel:Vts");
  vCKM[make_pair(5,2)] = settingsPtr->parm("StandardModel:Vub");
  vCKM[make_pair(2,5)] = settingsPtr->parm("StandardModel:Vub");
  vCKM[make_pair(5,4)] = settingsPtr->parm("StandardModel:Vcb");
  vCKM[make_pair(4,5)] = settingsPtr->parm("StandardModel:Vcb");
  vCKM[make_pair(5,6)] = settingsPtr->parm("StandardModel:Vtb");
  vCKM[make_pair(6,5)] = settingsPtr->parm("StandardModel:Vtb");

  // Overestimate constants for Breit-Wigner sampling.
  cBW[6]  = {1.2618863, 1.0986116, 0.0352201, 1.1040597};
  cBW[23] = {1.1699582, 1.0668744, 0.0338234, 1.1567254};
  cBW[24] = {1.2091010, 1.0854375, 0.0317986, 1.1629825};
  cBW[25] = {1.1864117, 1.0818452, 0.0069201, 1.1963023};

  // Compute total width for all resonances and set Breit-Wigner
  // overestimate normalizations.
  for(auto itRes = dataPtr->begin() ; itRes != dataPtr->end(); ++itRes) {
    if (!itRes->second.isRes) continue;
    int id(abs(itRes->first.first)), pol(itRes->first.second);
    double mass(itRes->second.mass);

    // Compute width for the on-shell mass.
    double width = getTotalWidth(id, mass, pol);

    // Update information.
    itRes->second.width = width;
  }

  // Spin assignment vectors.
  fermionPols.push_back(-1);
  fermionPols.push_back(1);
  vectorPols.push_back(1);
  vectorPols.push_back(0);
  vectorPols.push_back(-1);
  scalarPols.push_back(0);

}

//--------------------------------------------------------------------------

// Spinor products used for the calculation of the amplitudes.

complex AmpCalculator::spinProd(int pol, const Vec4& k1, const Vec4& k2) {

  // Check if k1 or k2 accidentally align with the direction of basis vectors.
  if ( ( (k2.e() - k2.px()==0.) || (k1.e() - k1.px()==0.) ) ) {
    infoPtr->errorMsg("Warning in " + __METHOD_NAME__, ": momentum aligned"
                      " exactly with basis direction.");
    return complex(0.,0.);
  }

  // Explicit expression for a spinor product.
  complex sNow(0., 0.);
  if (pol == 1) {
    sNow = complex(k1.py(), k1.pz())
      * sqrt( complex((k2.e() - k2.px())/(k1.e() - k1.px()), 0) )
      - complex(k2.py(), k2.pz())
      * sqrt( complex((k1.e() - k1.px())/(k2.e() - k2.px()), 0) );
  } else if (pol == -1) {
    sNow =  complex(k2.py(), -k2.pz())
      * sqrt( complex((k1.e() - k1.px())/(k2.e() - k2.px()), 0) )
      - complex(k1.py(), -k1.pz())
      * sqrt( complex((k2.e() - k2.px())/(k1.e() - k1.px()), 0) );
  }

  // Check result is valid.
  if (isnan(double(sNow.real())) || isnan(double(sNow.imag()))) {
    infoPtr->errorMsg("Warning in " + __METHOD_NAME__, ": NAN encountered.");
    return complex(0.,0.);
  } else if (isinf(double(sNow.real())) || isinf(double(sNow.imag()))) {
    infoPtr->errorMsg("Warning in " + __METHOD_NAME__, ": INF encountered.");
    return complex(0.,0.);
  }
  return sNow;

}

complex AmpCalculator::spinProd(int pol, const Vec4& k1, const Vec4& p1,
  const Vec4& k2) {
  Vec4 p1Flat = spinProdFlat(__METHOD_NAME__, k1, p1);
  return spinProd(pol, k1, p1Flat)*spinProd(-pol, p1Flat, k2);
}

complex AmpCalculator::spinProd(int pol, const Vec4 &k1, const Vec4& p1,
  const Vec4& p2, const Vec4& k2) {
  Vec4 p1Flat = spinProdFlat(__METHOD_NAME__, k1, p1);
  return spinProd(pol, k1, p1Flat)*spinProd(-pol, p1Flat, p2, k2);
}

complex AmpCalculator::spinProd(int pol, const Vec4& k1, const Vec4& p1,
  const Vec4& p2, const Vec4& p3, const Vec4& k2) {
  Vec4 p1Flat = spinProdFlat(__METHOD_NAME__, k1, p1);
  return spinProd(pol, k1, p1Flat)*spinProd(-pol, p1Flat, p2, p3, k2);
}

complex AmpCalculator::spinProd(int pol, const Vec4& k1, const Vec4& p1,
  const Vec4& p2, const Vec4& p3, const Vec4& p4, const Vec4& k2) {
  Vec4 p1Flat = spinProdFlat(__METHOD_NAME__, k1, p1);
  return spinProd(pol, k1, p1Flat)*spinProd(-pol, p1Flat, p2, p3, p4, k2);
}

Vec4 AmpCalculator::spinProdFlat(
  string method, const Vec4& k1, const Vec4& p1) {
  if (p1*k1 == 0) {
    if (p1.mCalc()/p1.e() > MILLI) {
      stringstream ss;
      ss << ": zero denominator in flattening slashed momentum "
         << "num = " << 0.5*p1.m2Calc() << " denom = " << p1*k1;
      infoPtr->errorMsg("Error in " + method, ss.str());
    }
    return p1;
  } else return p1 - 0.5*p1.m2Calc()/(p1*k1)*k1;
}

//--------------------------------------------------------------------------

// Initialize couplings.

void AmpCalculator::initCoup(bool va, int id1, int id2, int pol, bool m) {
  if (va) {
    v = vMap[make_pair(abs(id1), abs(id2))];
    a = aMap[make_pair(abs(id1), abs(id2))];
    vPls = v + pol*a;
    vMin = v - pol*a;
  } else if (id1 != 0) g = m ? gMap[make_pair(abs(id1), id2)] : 0;
}

//--------------------------------------------------------------------------

// Initialize an FSR branching amplitude.

void AmpCalculator::initFSRAmp(bool va, int id1, int id2, int pol,
  const Vec4& pi, const Vec4 &pj, const double& mMot, const double& widthQ2) {

  // Masses.
  mMot2 = pow2(mMot);
  mi    = max(0., pi.mCalc()); mi2 = pow2(mi);
  mj    = max(0., pj.mCalc()); mj2 = pow2(mj);
  fsrQ2 = complex((pi + pj).m2Calc() - mMot2, mMot*widthQ2);

  // Reference vectors.
  kij = pi + pj; kij.e(1); kij.flip3(); kij.rescale3(1./kij.pAbs());
  ki  = pi;      ki.e(1);  ki.flip3();  ki.rescale3(1./ki.pAbs());
  kj  = pj;      kj.e(1);  kj.flip3();  kj.rescale3(1./kj.pAbs());
  pij = pi + pj;
  wij = sqrt(2*(pij.e() + pij.pAbs())); wij2 = pow2(wij);
  wi  = sqrt(2*(pi.e() + pi.pAbs()));   wi2  = pow2(wi);
  wj  = sqrt(2*(pj.e() + pj.pAbs()));   wj2  = pow2(wj);
  M   = 0;

  // Couplings.
  initCoup(va, id1, id2, pol, true);

}

//--------------------------------------------------------------------------

// Check for zero denominator in an FSR amplitude.

bool AmpCalculator::zdenFSRAmp(const string& method, const Vec4& pi,
  const Vec4& pj, bool check) {
  if (check || (fsrQ2.real() == 0 && fsrQ2.imag() == 0)) {
    if (verbose >= NORMAL) {
      stringstream ss;
      ss << "zero denominator encountered."
         << "\n    wij =" << wij << " wi = " << wi << "  wj2 = " << wj2
         << "\n    mj = " << mj  << " Q2 = " << fsrQ2
         << "\n    pi = " << pi  << "    pj = " << pj;
      infoPtr->errorMsg("Warning in " + method + ": ", ss.str());
    } return true;
  } else return false;
}

//--------------------------------------------------------------------------

// Initialize an ISR branching amplitude.

void AmpCalculator::initISRAmp(bool va, int id1, int id2, int pol,
  const Vec4& pa, const Vec4 &pj, double& mA) {

  // Masses (disable mass corrections in ISR).
  mA    = 0; mA2 = 0;
  ma    = 0; ma2 = 0;
  mj    = max(0., pj.mCalc()); mj2 = pow2(mj);
  isrQ2 = -(pa - pj).m2Calc() + mA2;

  // Reference vectors.
  kaj = pa - pj; kaj.e(1); kaj.flip3(); kaj.rescale3(1./kaj.pAbs());
  ka  = pa;      ka.e(1);  ka.flip3();  ka.rescale3(1./ka.pAbs());
  kj  = pj;      kj.e(1);  kj.flip3();  kj.rescale3(1./kj.pAbs());
  paj = pa - pj;
  waj = sqrt(2*(paj.e() + paj.pAbs())); waj2 = pow2(waj);
  wa  = sqrt(2*(pa.e() + pa.pAbs()));   wa2  = pow2(wa);
  wj  = sqrt(2*(pj.e() + pj.pAbs()));   wj2  = pow2(wj);
  M   = 0;

  // Couplings.
  initCoup(va, id1, id2, pol, false);

}

//--------------------------------------------------------------------------

// Check for zero denominator in an ISR amplitude.

bool AmpCalculator::zdenISRAmp(const string& method, const Vec4& pa,
  const Vec4& pj, bool check) {
  if (check || isrQ2 == 0) {
    if (verbose >= NORMAL) {
      stringstream ss;
      ss << "zero denominator encountered."
         << "\n    waj =" << waj << " wa = " << wa << "  wj2 = " << wj2
         << "\n    mj = " << mj  << " Q2 = " << isrQ2
         << "\n    pa = " << pa  << "    pj = " << pj;
      infoPtr->errorMsg("Warning in " + method + ": ", ss.str());
    } return true;
  } else return false;
}

//--------------------------------------------------------------------------

// Branching Amplitudes.

// These functions still have the option to compute branching kernels with a
// width.

// FSR: f->fV.

complex AmpCalculator::ftofvFSRAmp(const Vec4& pi, const Vec4& pj, int idMot,
  int idi, int idj, double mMot, double widthQ2, int polMot, int poli,
  int polj) {

  // Initialize.
  initFSRAmp(true, idMot, idj, polMot, pi, pj, mMot, widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj, wij == 0 || wi == 0 || wj2 == 0 ||
    (mj == 0. && polj == 0) )) return M;

  // Calculate amplitude.
  // Transverse polarizations, all defined with hel -pol.
  if (abs(polj) == 1) {
    complex kikj       = spinProd(-polMot, ki, kj);
    complex kipikj     = spinProd(-polMot, ki, pi, kj);
    complex kipjkj     = spinProd(-polMot, ki, pj, kj);
    complex kipipjkj   = spinProd(-polMot, ki, pi, pj, kj);
    complex kjkij      = spinProd(-polMot, kj, kij);
    complex kjpijkij   = spinProd(-polMot, kj, pij, kij);
    complex kjpjkij    = spinProd(-polMot, kj, pj, kij);
    complex kjpjpijkij = spinProd(-polMot, kj, pj, pij, kij);
    double  trvFac     = sqrt(2)*polMot/wi/wij/wj2;
    if ((poli == polMot) && (polj == polMot))
      M = -trvFac*(vMin*kipipjkj*kjpijkij
                   - vPls*mi*mMot*kikj*kjpjkij)/fsrQ2;
    else if ((poli == polMot) && (polj == -polMot))
      M = -trvFac*(vMin*kipikj*conj(-kjpjpijkij)
                   - vPls*mi*mMot*kipjkj*conj(-kjkij))/fsrQ2;
    else if ((poli == -polMot) && (polj == polMot))
      M =  trvFac*(mMot*vPls*conj(kipikj)*kjpjkij
                   - mi*vMin*conj(kipjkj)*kjpijkij)/fsrQ2;
    else if ((poli == -polMot) && (polj == -polMot))
      M =  trvFac*(mMot*vPls*conj(-kipipjkj)*conj(-kjkij)
                   - mi*vMin*conj(-kikj)*conj(-kjpjpijkij))/fsrQ2;

  // Longitudinal polarizations.
  } else if (polj == 0) {
    double lonFac = 1./mj/wi/wij;
    if (poli == polMot)
      M = lonFac*(mMot2*vMin*spinProd(-polMot, ki, pi, kij)
                  - mi2*vMin*spinProd(-polMot, ki, pij, kij)
                  + mi*mMot*vPls*spinProd(-polMot, ki, pj, kij)
                  - 2*mj2/wj2*vMin*spinProd(-polMot, ki, pi, kj, pij, kij)
                  - 2*mj2/wj2*vPls*mMot*mi*spinProd(-polMot, ki, kj, kij)
                  )/fsrQ2;
    else if (poli == -polMot)
      M = lonFac*(mi*vMin*( spinProd(-polMot, ki, pj, pij, kij)
                  - 2*mj2/wj2*spinProd(-polMot, ki, kj, pij, kij))
                  + mMot*vPls*( spinProd(-polMot, ki, pi, pj, kij)
                  - 2*mj2/wj2*spinProd(-polMot, ki, pi, kj, kij)) )/fsrQ2;
  }

  // Multiply by CKM matrix - only for q+W.
  if (abs(idj) == 24 && abs(idi) < 7)
    M *= vCKM[make_pair(abs(idMot), abs(idi))];
  return M;

}

//--------------------------------------------------------------------------

// FSR: f->fH.

complex AmpCalculator::ftofhFSRAmp(const Vec4& pi, const Vec4& pj, int idMot,
  int, int idj, double, double widthQ2, int polMot, int poli, int) {

  // Initialize.
  initFSRAmp(false, idMot, idj, polMot, pi, pj, max(0., pi.mCalc()), widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj, wij == 0 || wi == 0)) return M;

  // Calculate amplitude.
  double fac = mi*g/wi/wij;
  if (poli == polMot) M = fac*mi*spinProd(-polMot, ki, pi + pij, kij)/fsrQ2;
  else if (poli == -polMot) M = fac*(spinProd(-polMot, ki, pi, pij, kij)
                                     + mi2*spinProd(-polMot, ki, kij))/fsrQ2;
  return M;

}

//--------------------------------------------------------------------------

// FSR: fbar->fbarV.

complex AmpCalculator::fbartofbarvFSRAmp(const Vec4& pi, const Vec4& pj,
  int idMot, int idi, int idj, double mMot, double widthQ2, int polMot,
  int poli, int polj) {

  // Initialize.
  initFSRAmp(true, idMot, idj, polMot, pi, pj, mMot, widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj, wij == 0 || wi == 0 || wj2 == 0 ||
    (mj == 0. && polj == 0))) return M;

  // Calculate amplitude.
  // Transverse polarizations, all defined with hel pol.
  if (abs(polj) == 1) {
    complex kijkj      = spinProd(polMot, kij, kj);
    complex kijpijkj   = spinProd(polMot, kij, pij, kj);
    complex kijpjkj    = spinProd(polMot, kij, pj, kj);
    complex kijpijpjkj = spinProd(polMot, kij, pij, pj, kj);
    complex kjki       = spinProd(polMot, kj, ki);
    complex kjpiki     = spinProd(polMot, kj, pi, ki);
    complex kjpjki     = spinProd(polMot, kj, pj, ki);
    complex kjpjpiki   = spinProd(polMot, kj, pj, pi, ki);
    double  trvFac     = sqrt(2)*polMot/wi/wij/wj2;
    if ((poli == polMot) && (polj == polMot))
      M = -trvFac*(vPls*kijpijkj*conj(-kjpjpiki)
                   - vMin*mi*mMot*kijpjkj*conj(-kjki))/fsrQ2;
    else if ((poli == polMot) && (polj == -polMot))
      M = -trvFac*(vPls*kijpijpjkj*kjpiki
                   - vMin*mi*mMot*kijkj*kjpjki)/fsrQ2;
    else if ((poli == -polMot) && (polj == polMot))
      M = -trvFac*(mMot*vMin*kijpjkj*conj(kjpiki)
                   - mi*vPls*kijpijkj*conj(kjpjki))/fsrQ2;
    else if ((poli == -polMot) && (polj == -polMot))
      M = -trvFac*(mMot*vMin*kijkj*kjpjpiki
                   - mi*vPls*kijpijpjkj*kjki)/fsrQ2;

  // Longitudinal polarizations.
  } else if (polj == 0) {
    double lonFac = 1./mj/wi/wij;
    if (poli == polMot)
      M = -lonFac*(mMot2*vPls*spinProd(polMot, kij, pi, ki)
                   - mi2*vPls*spinProd(polMot, kij, pij, ki)
                   + mi*mMot*vMin*spinProd(polMot, kij, pj, ki)
                   - 2*mj2/wj2*vPls*spinProd(polMot, kij, pij, kj, pi, ki)
                   - 2*mj2/wj2*vMin*mMot*mi*spinProd(polMot, kij, kj, ki)
                   )/fsrQ2;
    else if (poli == -polMot)
      M = -lonFac*(mi*vPls*( spinProd(polMot, kij, pij, pj, ki)
                   - 2*mj2/wj2*spinProd(polMot, kij, pij, kj, ki))
                   + mMot*vMin*( spinProd(polMot, kij, pj, pi, ki)
                   - 2*mj2/wj2*spinProd(polMot, kij, kj, pi, ki)) )/fsrQ2;
  }

  // Multiply by CKM matrix - only for q+W.
  if (abs(idj) == 24 && abs(idi) < 7)
    M *= vCKM[make_pair(abs(idMot), abs(idi))];
  return M;

}

//--------------------------------------------------------------------------

// FSR: fbar->fbarH.

complex AmpCalculator::fbartofbarhFSRAmp(const Vec4& pi, const Vec4& pj,
  int idMot, int, int idj, double, double widthQ2, int polMot, int poli, int) {

  // Initialize.
  initFSRAmp(false, idMot, idj, polMot, pi, pj, max(0., pi.mCalc()), widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj, wij == 0 || wi == 0)) return M;

  // Calculate amplitude.
  double fac = mi*g/wi/wij;
  if (poli == polMot) M = fac*mi*spinProd(polMot, kij, pij + pi, ki)/fsrQ2;
  else if (poli == -polMot) M = fac*(spinProd(polMot, kij, pij, pi, ki)
                                     + mi2*spinProd(polMot, kij, ki))/fsrQ2;
  return M;

}

//--------------------------------------------------------------------------

// FSR: VT->ffbar.

complex AmpCalculator::vTtoffbarFSRAmp(const Vec4& pi, const Vec4& pj,
  int idMot, int idi, int idj, double mMot, double widthQ2, int polMot,
  int poli, int polj) {

  // Initialize.
  initFSRAmp(true, idi, idMot, polMot, pi, pj, mMot, widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj, wij2 == 0 || wi == 0 || wj == 0))
    return M;

  // Calculate amplitude.
  // All defined with hel -pol.
  complex kikij      = spinProd(-polMot, ki, kij);
  complex kipikij    = spinProd(-polMot, ki, pi, kij);
  complex kipijkij   = spinProd(-polMot, ki, pij, kij);
  complex kipipijkij = spinProd(-polMot, ki, pi, pij, kij);
  complex kijkj      = spinProd(-polMot, kij, kj);
  complex kijpjkj    = spinProd(-polMot, kij, pj, kj);
  complex kijpijkj   = spinProd(-polMot, kij, pij, kj);
  complex kijpijpjkj = spinProd(-polMot, kij, pij, pj, kj);
  double  trvFac     = sqrt(2)*polMot/wi/wj/wij2;
  if (poli == polMot && polj == -polMot)
    M = trvFac*(vMin*kipikij*kijpijpjkj
                + vPls*mi*mj*kipijkij*kijkj)/fsrQ2;
  else if (poli == -polMot && polj == polMot)
    M = trvFac*(vPls*kipipijkij*kijpjkj
                + vMin*mi*mj*kikij*kijpijkj)/fsrQ2;
  else if (poli == polMot && polj == polMot)
    M = trvFac*(vPls*mi*kipijkij*kijpjkj
                + vMin*mj*kipikij*kijpijkj)/fsrQ2;
  else if (poli == -polMot && polj == -polMot)
    M = trvFac*(vMin*mi*kikij*kijpijpjkj
                + vPls*mj*kipipijkij*kijkj)/fsrQ2;

  // Multiply by CKM matrix - only for W->qqbar.
  if (abs(idMot) == 24 && abs(idi) < 7)
    M *= vCKM[make_pair(abs(idi), abs(idj))];
  return M;

}

//--------------------------------------------------------------------------

// FSR: VT->VH.

complex AmpCalculator::vTtovhFSRAmp(const Vec4& pi, const Vec4& pj, int idMot,
  int, int idj, double mMot, double widthQ2, int polMot, int poli, int) {

  // Initialize.
  initFSRAmp(false, idMot, idj, polMot, pi, pj, mMot, widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj, wij2 == 0 || wi2 == 0 ||
    ( mMot == 0. && poli == 0 ))) return M;

  // Calculate amplitude.
  double trtrFac = g/wij2/wi2;
  if (poli == polMot)
    M = -trtrFac*spinProd(-polMot, kij, pij, ki)
      *spinProd(-polMot, kij, pi, ki)/fsrQ2;
  else if (poli == -polMot)
    M = -trtrFac*spinProd(-polMot, ki, kij)
      *spinProd(-polMot, kij, pij, pi, ki)/fsrQ2;
  else if (poli == 0)
    M = -g*polMot/sqrt(2)/wij2/mMot*(spinProd(-polMot, kij, pij, pi, kij)
        - 2*mMot2/wi2*spinProd(-polMot, kij, pij, ki, kij) )/fsrQ2;
  return M;

}

//--------------------------------------------------------------------------

// FSR: VT->VV.

complex AmpCalculator::vTtovvFSRAmp(const Vec4& pi, const Vec4& pj, int idMot,
  int, int idj, double mMot, double widthQ2, int polMot, int poli,
  int polj) {

  // Initialize.
  initFSRAmp(false, idMot, idj, polMot, pi, pj, mMot, widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj, wij2 == 0 || wi2 == 0 || wj2 == 0 ||
    ( mi == 0. && poli == 0 ) || ( mj == 0. && polj == 0 ) )) return M;

  // u = pol, d = -pol, z = 0, p = mom.
  complex uiuj  = -1/wi2/wj2*spinProd(-polMot, ki, pi, pj, kj)
    *spinProd(-polMot, kj, ki);
  complex uidj  = -1/wi2/wj2*spinProd(-polMot, ki, pi, kj)
    *spinProd(-polMot, ki, pj, kj);
  complex diuj  = conj(uidj);
  complex didj  = conj(uiuj);
  complex uidij = -1/wi2/wij2*spinProd(-polMot, ki, pi, kij)
    *spinProd(-polMot, ki, pij, kij);
  complex didij = -1/wi2/wij2*spinProd(polMot, ki, pi, pij, kij)
    *spinProd(polMot, kij, ki);
  complex ujdij = -1/wj2/wij2*spinProd(-polMot, kj, pj, kij)
    *spinProd(-polMot, kj, pij, kij);
  complex djdij = -1/wj2/wij2*spinProd(polMot, kj, pj, pij, kij)
    *spinProd(polMot, kij, kj);
  complex piuj  = polMot/sqrt(2)/wj2*spinProd(-polMot, kj, pj, pi, kj);
  complex pidj  = conj(piuj);
  complex pjui  = polMot/sqrt(2)/wi2*spinProd(-polMot, ki, pi, pj, ki);
  complex pjdi  = conj(pjui);
  complex pidij = -polMot/sqrt(2)/wij2*spinProd(polMot, kij, pij, pi, kij);

  // Calculate amplitude.
  if (abs(poli) == 1 && abs(polj) == 1) {
    if ((poli == polMot) && (polj == polMot))
      M = 2*g*(pjui*ujdij - piuj*uidij + pidij*uiuj)/fsrQ2;
    else if ((poli == polMot) && (polj == -polMot))
      M = 2*g*(pjui*djdij - pidj*uidij + pidij*uidj)/fsrQ2;
    else if ((poli == -polMot) && (polj == polMot))
      M = 2*g*(pjdi*ujdij - piuj*didij + pidij*diuj)/fsrQ2;
    else if ((poli == -polMot) && (polj == -polMot))
      M = 2*g*(pjdi*djdij - pidj*didij + pidij*didj)/fsrQ2;
  } else if (abs(poli) == 1 && polj == 0) {
    complex uizj = polMot/sqrt(2)/wi2/mj*
      (spinProd(-polMot, ki, pi, pj, ki)
       - 2*mj2/wj2*spinProd(-polMot, ki, pi, kj, ki) );
    complex dizj = conj(uizj);
    complex pizj = (0.5*(mMot2 - mi2 - mj2) - 2*mj2/wj2*(pi*kj))/mj;
    complex zjdij = -polMot/sqrt(2)/wij2/mj*(
      spinProd(polMot, kij, pij, pj, kij)
      - 2*mj2/wj2*spinProd(polMot, kij, pij, kj, kij) );
    if (poli == polMot)
      M = 2*g*(pjui*zjdij - pizj*uidij + pidij*uizj)/fsrQ2;
    else if (poli == -polMot)
      M = 2*g*(pjdi*zjdij - pizj*didij + pidij*dizj)/fsrQ2;
  } else if (poli == 0 && abs(polj) == 1) {
    complex ziuj  = polMot/sqrt(2)/wj2/mi*
      (spinProd(-polMot, kj, pj, pi, kj)
       - 2*mi2/wi2*spinProd(-polMot, kj, pj, ki, kj) );
    complex zidj  = conj(ziuj);
    complex zidij = -polMot/sqrt(2)/wij2/mi*
      (spinProd(polMot, kij, pij, pi, kij)
       - 2*mi2/wi2*spinProd(polMot, kij, pij, ki, kij) );
    complex pjzi  = (0.5*(mMot2 - mi2 - mj2) - 2*mi2/wi2*(pj*ki))/mi;
    if (polj == polMot)
      M = 2*g*(pjzi*ujdij - piuj*zidij + pidij*ziuj)/fsrQ2;
    else if (polj == -polMot)
      M = 2*g*(pjzi*djdij - pidj*zidij + pidij*zidj)/fsrQ2;
  } else if (poli == 0 && polj == 0) {
    complex pizj  = (0.5*(mMot2 - mi2 - mj2) - 2*mj2/wj2*(pi*kj))/mj;
    complex pjzi  = (0.5*(mMot2 - mi2 - mj2) - 2*mi2/wi2*(pj*ki))/mi;
    complex zizj  = (0.5*(mMot2 - mi2 - mj2) - 2*mi2/wi2*(ki*pj)
      - 2*mj2/wj2*(kj*pi) - 4*mi2*mj2/wi2/wj2*(ki*kj))/mi/mj;
    complex zidij = -polMot/sqrt(2)/wij2/mi*
      (spinProd(polMot, kij, pij, pi, kij)
       - 2*mi2/wi2*spinProd(polMot, kij, pij, ki, kij) );
    complex zjdij = -polMot/sqrt(2)/wij2/mj*(
      spinProd(polMot, kij, pij, pj, kij)
      - 2*mj2/wj2*spinProd(polMot, kij, pij, kj, kij) );
    M = 2*g*(pjzi*zjdij - pizj*zidij + pidij*zizj)/fsrQ2;
  }
  return M;

}

//--------------------------------------------------------------------------

// FSR: VL->ffbar.

complex AmpCalculator::vLtoffbarFSRAmp(const Vec4& pi, const Vec4& pj,
  int idMot, int idi, int idj, double mMot, double widthQ2, int,
  int poli, int polj) {

  // Initialize.
  initFSRAmp(true, idi, idMot, 1, pi, pj, mMot, widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj, wij2 == 0 || wi == 0 || wj == 0 ||
    mMot == 0)) return M;

  // Calculate amplitude.
  double lonFac = 1/wi/wj/mMot;
  if (poli == 1 && polj == -1)
    M = lonFac*(vMin*pow2(mi)*spinProd(-1, ki, pj, kj)
                + vMin*pow2(mj)*spinProd(-1, ki, pi, kj)
                - vPls*mi*mj*spinProd(-1, ki, pij, kj)
                + 2*vPls*mi*mj*pow2(mMot)/wij2*spinProd(-1, ki, kij, kj)
                - 2*vMin*pow2(mMot)/wij2*spinProd(-1, ki, pi, kij, pj, kj)
                )/fsrQ2;
  else if (poli == -1 && polj == 1)
    M = lonFac*(vPls*pow2(mi)*spinProd(1, ki, pj, kj)
                + vPls*pow2(mj)*spinProd(1, ki, pi, kj)
                - vMin*mi*mj*spinProd(1, ki, pij, kj)
                + 2*vMin*mi*mj*pow2(mMot)/wij2*spinProd(1, ki, kij, kj)
                - 2*vPls*pow2(mMot)/wij2*spinProd(1, ki, pi, kij, pj, kj)
                )/fsrQ2;
  else if (poli == 1 && polj == 1)
    M = lonFac*(mi*vPls*spinProd(-1, ki, pij, pj, kj)
                - 2*mi*vPls*pow2(mMot)/wij2*spinProd(-1, ki, kij, pj, kj)
                - mj*vMin*spinProd(-1, ki, pi, pij, kj)
                + 2*mj*vMin*pow2(mMot)/wij2*spinProd(-1, ki, pi, kij, kj)
                )/fsrQ2;
  else if (poli == -1 && polj == -1)
    M = lonFac*(mi*vMin*spinProd(1, ki, pij, pj, kj)
                - 2*mi*vMin*pow2(mMot)/wij2*spinProd(1, ki, kij, pj, kj)
                - mj*vPls*spinProd(1, ki, pi, pij, kj)
                + 2*mj*vPls*pow2(mMot)/wij2*spinProd(1, ki, pi, kij, kj)
                )/fsrQ2;

  // Multiply by CKM matrix - only for W->qqbar.
  if (abs(idMot) == 24 && abs(idi) < 7)
    M *= vCKM[make_pair(abs(idi), abs(idj))];
  return M;

}

//--------------------------------------------------------------------------

// FSR: VL->VH.

complex AmpCalculator::vLtovhFSRAmp(const Vec4& pi, const Vec4& pj, int idMot,
  int, int idj, double mMot, double widthQ2, int polMot, int poli, int) {

  // Initialize.
  initFSRAmp(false, idMot, idj, polMot, pi, pj, mMot, widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj, wij2 == 0 || wi2 == 0 || wj2 == 0 ||
    ( mMot == 0. && poli == 0 ))) return M;

  // Calculate amplitude.
  if (poli == 1)
    M = -g/sqrt(2)/wi2/mMot*(spinProd(-1, ki, pi, pij, ki)
        - 2*mMot/wij2*spinProd(-1, ki, pi, kij, ki) )/fsrQ2;
  else if (poli == -1)
    M = g/sqrt(2)/wi2/mMot*(spinProd( 1, ki, pi, pij, ki)
        - 2*mMot/wij2*spinProd( 1, ki, pi, kij, ki) )/fsrQ2;
  else if (poli == 0)
    M = -g/mMot2*(0.5*pow2(mj) + mMot2*(wi2/wij2 + wj2/wi2))/fsrQ2;
  return M;

}

//--------------------------------------------------------------------------

// FSR: VL->VV.

complex AmpCalculator::vLtovvFSRAmp(const Vec4& pi, const Vec4& pj, int idMot,
  int, int idj, double mMot, double widthQ2, int polMot, int poli,
  int polj) {

  // Initialize.
  initFSRAmp(false, idMot, idj, polMot, pi, pj, mMot, widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj,
    wij2 == 0 || wi2 == 0 || wj2 == 0 || mMot == 0 ||
    ( mi == 0. && poli == 0 ) || ( mj == 0. && polj == 0 ))) return M;

  // u = pol, d = -pol, z = 0, p = mom.
  complex uiuj = -1/wi2/wj2*spinProd(-1, ki, pi, pj, kj)*spinProd(-1, kj, ki);
  complex uidj = -1/wi2/wj2*spinProd(-1, ki, pi, kj)*spinProd(-1, ki, pj, kj);
  complex diuj = conj(uidj);
  complex didj = conj(uiuj);
  complex uizij = 1/sqrt(2)/wi2/mMot*( spinProd(-1, ki, pi, pij, ki)
    - 2*mMot2/wij2*spinProd(-1, ki, pi, kij, ki) );
  complex dizij = conj(uizij);
  complex zizij = ( 0.5*(mMot2 + mi2 - mj2) - 2*mi2/wi2*(ki*pij)
    - 2*mMot2/wij2*(kij*pi) - 4*mi2*mMot2/wi2/wij2*(ki*kij) )/mi/mMot;
  complex ujzij = 1/sqrt(2)/wj2/mMot*( spinProd(-1, kj, pj, pij, kj)
    - 2*mMot2/wij2*spinProd(-1, kj, pj, kij, kj) );
  complex djzij = conj(ujzij);
  complex piuj  = 1/sqrt(2)/wj2*spinProd(-1, kj, pj, pi, kj);
  complex pidj  = conj(piuj);
  complex pjui  = 1/sqrt(2)/wi2*spinProd(-1, ki, pi, pj, ki);
  complex pjdi  = conj(pjui);
  complex pizij = (0.5*(mMot2 + mi2 - mj2) - 2*mMot2/wij2*(kij*pi))/mMot;

  // Calculate amplitude.
  if (abs(poli) == 1 && abs(polj) == 1) {
    if (poli == 1 && polj == 1)
      M = 2*g*(pjui*ujzij - piuj*uizij + pizij*uiuj)/fsrQ2;
    else if (poli == 1 && polj == -1)
      M = 2*g*(pjui*djzij - pidj*uizij + pizij*uidj)/fsrQ2;
    else if (poli == -1 && polj == 1)
      M = 2*g*(pjdi*ujzij - piuj*dizij + pizij*diuj)/fsrQ2;
    else if (poli == -1 && polj == -1)
      M = 2*g*(pjdi*djzij - pidj*dizij + pizij*didj)/fsrQ2;
  } else if (abs(poli) == 1 && polj == 0) {
    complex uizj  = 1/sqrt(2)/wi2/mj*
      (spinProd(-1, ki, pi, pj, ki) - 2*mj2/wj2*spinProd(-1, ki, pi, kj, ki) );
    complex dizj  = conj(uizj);
    complex zjzij = ( 0.5*(mMot2 - mi2 + mj2) - 2*mj2/wj2*(kj*pij)
      - 2*mMot2/wij2*(kij*pj) - 4*mj2*mMot2/wj2/wij2*(kj*kij) )/mj/mMot;
    complex pizj  = (0.5*(mMot2 - mi2 - mj2) - 2*mj2/wj2*(pi*kj))/mj;
    if (poli == 1) M = 2*g*(pjui*zjzij - pizj*uizij + pizij*uizj)/fsrQ2;
    else if (poli == -1) M = 2*g*(pjdi*zjzij - pizj*dizij + pizij*dizj)/fsrQ2;
  } else if (poli == 0 && abs(polj) == 1) {
    complex ziuj = 1/sqrt(2)/wj2/mi*( spinProd(-1, kj, pj, pi, kj)
      - 2*mi2/wi2*spinProd(-1, kj, pj, ki, kj) );
    complex zidj = conj(ziuj);
    complex pjzi = (0.5*(mMot2 - mi2 - mj2) - 2*mi2/wi2*(pj*ki))/mi;
    if (polj == 1) M = 2*g*(pjzi*ujzij - piuj*zizij + pizij*ziuj)/fsrQ2;
    else if (polj == -1) M = 2*g*(pjzi*djzij - pidj*zizij + pizij*zidj)/fsrQ2;
  } else if (poli == 0 && polj == 0) {
    complex zizj  = ( 0.5*(mMot2 - mi2 - mj2) - 2*mi2/wi2*(ki*pj)
      - 2*mj2/wj2*(kj*pi) - 4*mi2*mj2/wi2/wj2*(ki*kj) )/mi/mj;
    complex pizj  = (0.5*(mMot2 - mi2 - mj2) - 2*mj2/wj2*(pi*kj))/mj;
    complex pjzi  = (0.5*(mMot2 - mi2 - mj2) - 2*mi2/wi2*(pj*ki))/mi;
    complex zjzij = ( 0.5*(mMot2 - mi2 + mj2) - 2*mj2/wj2*(kj*pij)
      - 2*mMot2/wij2*(kij*pj) - 4*mj2*mMot2/wj2/wij2*(kj*kij) )/mj/mMot;
    M = 2*g*(pjzi*zjzij - pizj*zizij + pizij*zizj)/fsrQ2;
  }
  return M;

}

//--------------------------------------------------------------------------

// FSR: H->ffbar.

complex AmpCalculator::htoffbarFSRAmp(const Vec4& pi, const Vec4& pj,
  int idMot, int idi, int, double mMot, double widthQ2, int polMot,
  int poli, int polj) {

  // Initialize.
  initFSRAmp(false, idi, idMot, polMot, pi, pj, mMot, widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj, wi == 0 || wj == 0)) return M;

  // Calculate amplitude (mi = mj).
  double fac = mi*g/wi/wj;
  if (poli == 1 && polj == -1)
    M = fac*(mi*spinProd(-1, ki, pj, kj) - mj*spinProd(-1, ki, pi, kj) )/fsrQ2;
  else if (poli == -1 && polj == 1)
    M = fac*(mi*spinProd(-1, ki, pj, kj) - mj*spinProd(-1, ki, pi, kj) )/fsrQ2;
  else if (poli == 1 && polj == 1)
    M = fac*(spinProd(-1, ki, pi, pj, kj) - mi*mj*spinProd(-1, ki, kj) )/fsrQ2;
  else if (poli == -1 && polj == -1)
    M = fac*(spinProd(1, ki, pi, pj, kj) - mi*mj*spinProd(1, ki, kj) )/fsrQ2;
  return M;

}

//--------------------------------------------------------------------------

// FSR: H->VV.

complex AmpCalculator::htovvFSRAmp(const Vec4& pi, const Vec4& pj, int idMot,
  int idi, int, double mMot, double widthQ2, int polMot, int poli, int polj) {

  // Initialize.
  initFSRAmp(false, idi, idMot, polMot, pi, pj, mMot, widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj, wi2 == 0 || wj2 == 0 || mi == 0 ||
    mj == 0)) return M;

  // Calculate amplitude.
  double trtrFac = g/wi2/wj2;
  if (poli == 1 && polj == 1 )
    M = -trtrFac*spinProd(-1, kj, ki)*spinProd(-1, ki, pi, pj, kj)/fsrQ2;
  else if (poli == -1  && polj == -1 )
    M = -trtrFac*spinProd( 1, kj, ki)*spinProd( 1 , ki, pi, pj, kj)/fsrQ2;
  else if (poli == 1 && polj == -1 )
    M = -trtrFac*spinProd(-1, ki, pi, kj)*spinProd(-1, ki, pj, kj)/fsrQ2;
  else if (poli == -1 && polj == 1 )
    M = -trtrFac*spinProd( 1, ki, pi, kj)*spinProd( 1, ki, pj, kj)/fsrQ2;
  else if (poli == 0 && polj == 1 )
    M =  g/sqrt(2)/wj2/mi*(spinProd(-1, kj, pj, pi, kj)
                           - 2*mi2/wi2*spinProd(-1, kj, pj, ki, kj))/fsrQ2;
  else if (poli == 0 && polj == -1 )
    M = -g/sqrt(2)/wj2/mi*(spinProd( 1, kj, pj, pi, kj)
                           - 2*mi2/wi2*spinProd( 1, kj, pj, ki, kj))/fsrQ2;
  else if (poli == 1 && polj == 0)
    M =  g/sqrt(2)/wi2/mj*(spinProd(-1, ki, pi, pj, ki)
                           - 2*mj2/wi2*spinProd(-1, ki, pi, kj, ki))/fsrQ2;
  else if (poli == -1 && polj == 0)
    M = -g/sqrt(2)/wi2/mj*(spinProd( 1, ki, pi, pj, ki)
                           - 2*mj2/wi2*spinProd(1 , ki, pi, kj, ki))/fsrQ2;
  else if (poli == 0 && polj == 0)
    M =  g/mi/mj*(0.5*(mMot2 - mi2 - mj2) - mj2*wi2/wj2 - mi2*wj2/wi2)/fsrQ2;
  return M;

}

//--------------------------------------------------------------------------

// FSR: H->HH.

complex AmpCalculator::htohhFSRAmp(const Vec4& pi, const Vec4& pj, int idMot,
  int idi, int, double mMot, double widthQ2, int polMot, int, int) {

  // Initialize.
  initFSRAmp(false, idi, idMot, polMot, pi, pj, mMot, widthQ2);
  if (zdenFSRAmp(__METHOD_NAME__, pi, pj, false)) return M;

  // Calculate amplitude.
  M = g/fsrQ2;
  return M;

}

//--------------------------------------------------------------------------

// ISR: f->fV.

complex AmpCalculator::ftofvISRAmp(const Vec4& pa, const Vec4& pj, int idA,
  int ida, int idj, double mA, int polA, int pola, int polj) {

  // Initialize.
  initISRAmp(true, idA, idj, polA, pa, pj, mA);
  if (zdenISRAmp(__METHOD_NAME__, pa, pj, waj == 0 || wa == 0 || wj2 == 0 ||
    ( mj == 0. && polj == 0 ))) return M;

  // Calculate amplitude (all defined with hel -pol).
  if (abs(pola) == 1 && abs(polj) == 1) {
    complex kajkj      = spinProd(-polA, kaj, kj);
    complex kajpajkj   = spinProd(-polA, kaj, paj, kj);
    complex kajpjkj    = spinProd(-polA, kaj, pj, kj);
    complex kajpajpjkj = spinProd(-polA, kaj, paj, pj, kj);
    complex kjka       = spinProd(-polA, kj, ka);
    complex kjpaka     = spinProd(-polA, kj, pa, ka);
    complex kjpjka     = spinProd(-polA, kj, pj, ka);
    complex kjpjpaka   = spinProd(-polA, kj, pj, pa, ka);
    double  trvFac     = sqrt(2)*polA/wa/waj/wj2/isrQ2;
    if (pola == polA && polj == polA)
      M = trvFac*(vMin*kajpajpjkj*kjpaka - vPls*ma*mA*kajkj*kjpjka);
    else if (pola == polA && polj == -polA)
      M = trvFac*(vMin*kajpajkj*conj(-kjpjpaka)
                  - vPls*ma*mA*kajpjkj*conj(-kjka));
    else if (pola == -polA && polj == polA)
      M = trvFac*(mA*vPls*kajkj*kjpjpaka - ma*vMin*kajpajpjkj*kjka);
    else if (pola == -polA && polj == -polA)
      M = trvFac*(mA*vPls*kajpjkj*conj(kjpaka)
                  - ma*vMin*kajpajkj*conj(kjpjka));
  } else if (abs(pola) == 1 && polj == 0) {
    double lonFac = 1./mj/wa/waj/isrQ2;
    if (pola == polA)
      M = -lonFac*(ma2*vMin*spinProd(-polA, kaj, paj, ka)
                   - mA2*vMin*spinProd(-polA, kaj, pa, ka)
                   + ma*mA*vPls*spinProd(-polA, kaj, pj, ka)
                   - 2*mj2/wj2*vMin*spinProd(-polA, kaj, paj, kj, pa, ka)
                   - 2*mj2/wj2*vPls*mA*ma*spinProd(-polA, kaj, kj, ka) );
    else if (pola == -polA)
      M =  -lonFac*(ma*vMin*( spinProd(-polA, kaj, paj, pj, ka)
                    - 2*mj2/wj2*spinProd(-polA, kaj, paj, kj, ka) )
                    + mA*vPls*( spinProd(-polA, kaj, pj, pa, ka)
                    - 2*mj2/wj2*spinProd(-polA, kaj, kj, pa, ka)) );
  }

  // Multiply by CKM matrix - only for q+W.
  if (abs(idj) == 24 && abs(ida) < 7)
    M *= vCKM[make_pair(abs(idA), abs(ida))];
  return M;

}

//--------------------------------------------------------------------------

// ISR: f->fH.

complex AmpCalculator::ftofhISRAmp(const Vec4& pa, const Vec4& pj, int idA,
  int, int idj, double mA, int polA, int pola, int) {

  // Initialize.
  initISRAmp(false, idA, idj, polA, pa, pj, mA);
  if (zdenISRAmp(__METHOD_NAME__, pa, pj, waj == 0 || wa == 0)) return M;

  // Calculate amplitude.
  double fac = mA*g/wa/waj/isrQ2;
  if (pola == polA) M = -fac*ma*( spinProd(-polA, kaj, paj + pa, ka));
  else if (pola == -polA) M = -fac*( spinProd(-polA, kaj, paj, pa, ka)
                                     + ma2*spinProd(-polA, kaj, ka));
  return M;

}

//--------------------------------------------------------------------------

// ISR: fbar->fbarV.

complex AmpCalculator::fbartofbarvISRAmp(const Vec4& pa, const Vec4& pj,
  int idA, int ida, int idj, double mA, int polA, int pola, int polj) {

  // Initialize.
  initISRAmp(true, idA, idj, polA, pa, pj, mA);
  if (zdenISRAmp(__METHOD_NAME__, pa, pj, waj == 0 || wa == 0 || wj2 == 0 ||
    (mj == 0. && polj == 0 ))) return M;

  // Calculate amplitude.
  // All defined with hel pol.
  if (abs(pola) == 1 && abs(polj) == 1) {
    complex kakj       =  spinProd(polA, ka, kj);
    complex kapakj     = spinProd(polA, ka, pa, kj);
    complex kapjkj     = spinProd(polA, ka, pj, kj);
    complex kapapjkj   = spinProd(polA, ka, pa, pj, kj);
    complex kjkaj      = spinProd(polA, kj, kaj);
    complex kjpajkaj   = spinProd(polA, kj, paj, kaj);
    complex kjpjkaj    = spinProd(polA, kj, pj, kaj);
    complex kjpjpajkaj = spinProd(polA, kj, pj, paj, kaj);
    double  trvFac     = sqrt(2)*polA/wa/waj/wj2/isrQ2;
    if (pola == polA && polj == polA)
      M = trvFac*(vPls*kapakj*conj(-kjpjpajkaj)
                  - vMin*ma*mA*kapjkj*conj(-kjkaj));
    else if (pola == polA && polj == -polA)
      M = trvFac*(vPls*kapapjkj*kjpajkaj - vMin*ma*mA*kakj*kjpjkaj);
    else if (pola == -polA && polj == polA)
      M = trvFac*(ma*vMin*kapjkj*conj(kjpajkaj)
                  - mA*vPls*kapakj*conj(kjpjkaj));
    else if (pola == -polA && polj == -polA)
      M = trvFac*(ma*vMin*kakj*kjpjpajkaj - mA*vPls*kapapjkj*kjkaj);
  } else if (abs(pola) == 1 && polj == 0) {
    double lonFac = 1./mj/wa/waj/isrQ2;
    if (pola == polA)
      M = lonFac*(ma2*vPls*spinProd(polA, ka, paj, kaj)
                  - mA2*vPls*spinProd(polA, ka, pa, kaj)
                  + ma*mA*vMin*spinProd(polA, ka, pj, kaj)
                  - 2*mj2/wj2*vPls*spinProd(polA, ka, pa, kj, paj, kaj)
                  - 2*mj2/wj2*vMin*mA*ma*spinProd(polA, ka, kj, kaj) );
    else if (pola == -polA)
      M = -lonFac*(mA*vPls*( spinProd(polA, ka, pa, pj, kaj)
                   - 2*mj2/wj2*spinProd(polA, ka, pa, kj, kaj) )
                   + ma*vMin*( spinProd(polA, ka, pj, paj, kaj)
                   - 2*mj2/wj2*spinProd(polA, ka, kj, paj, kaj)) );
  }

  // Multiply by CKM matrix - only for q+W.
  if (abs(idj) == 24 && abs(ida) < 7)
    M *= vCKM[make_pair(abs(idA), abs(ida))];
  return M;

}

//--------------------------------------------------------------------------

// ISR: fbar->fbarH.

complex AmpCalculator::fbartofbarhISRAmp(const Vec4& pa, const Vec4& pj,
  int idA, int, int idj, double mA, int polA, int pola, int) {

  // Initialize.
  initISRAmp(false, idA, idj, polA, pa, pj, mA);
  if (!zdenISRAmp(__METHOD_NAME__, pa, pj, waj == 0 || wa == 0)) return M;

  // Calculate amplitude.
  double fac = mA*g/wa/waj/isrQ2;
  if (pola == polA) M = -fac*ma*spinProd(polA, ka, pa + paj, kaj);
  else if (pola == -polA) M = -fac*(spinProd(polA, ka, pa, paj, kaj)
                                    + ma2*spinProd(polA, ka, kaj));
  return M;

}

//--------------------------------------------------------------------------

// FSR amplitude selector.

complex AmpCalculator::branchAmpFSR(const Vec4& pi, const Vec4& pj, int idMot,
  int idi, int idj, double mMot, double widthQ2, int polMot, int poli,
  int polj) {

  // I is fermion.
  if (abs(idMot) < 20 && idMot > 0) {
    // j is Higgs.
    if (idj == 25) return ftofhFSRAmp(pi, pj, idMot, idi, idj,
                                      mMot, widthQ2, polMot, poli, polj);
    // j is vector.
    else return ftofvFSRAmp(pi, pj, idMot, idi, idj,
                            mMot, widthQ2, polMot, poli, polj);
  // I is antifermion.
  } else if (abs(idMot) < 20 && idMot < 0) {
    // j is Higgs.
    if (idj == 25) return fbartofbarhFSRAmp(pi, pj, idMot, idi, idj,
                                            mMot, widthQ2, polMot, poli, polj);
    // j is vector.
    else return fbartofbarvFSRAmp(pi, pj, idMot, idi, idj,
                                  mMot, widthQ2, polMot, poli, polj);
  // I is higgs.
  } else if (idMot == 25) {
    // i is Higgs.
    if (idi == 25) return htohhFSRAmp(pi, pj, idMot, idi, idj,
                                      mMot, widthQ2, polMot, poli, polj);
    // i is fermion (add a factor sqrt(3) for splittings to quarks).
    else if (abs(idi) < 20)
      return (idi < 7 ? sqrt(3) : 1)*htoffbarFSRAmp(pi, pj, idMot, idi, idj,
                              mMot, widthQ2, polMot, poli, polj);
    // i is vector.
    else return htovvFSRAmp(pi, pj, idMot, idi, idj,
                            mMot, widthQ2, polMot, poli, polj);
  // I is vector.
  } else {
    // I is transverse.
    if (polMot != 0) {
      // i is fermion (add a factor sqrt(3) for splittings to quarks).
      if (abs(idi) < 20) return (idi < 7 ? sqrt(3) : 1)*
                           vTtoffbarFSRAmp(pi, pj, idMot, idi, idj,
                           mMot, widthQ2, polMot, poli, polj);
      // j is Higgs.
      else if (idj == 25) return vTtovhFSRAmp(pi, pj, idMot, idi, idj,
        mMot, widthQ2, polMot, poli, polj);
      // i is vector.
      else return vTtovvFSRAmp(pi, pj, idMot, idi, idj,
        mMot, widthQ2, polMot, poli, polj);
    // I is longitudinal.
    } else {
      // i is fermion (add a factor sqrt(3) for splittings to quarks).
      if (abs(idi) < 20) return (idi < 7 ? sqrt(3) : 1)*
                           vLtoffbarFSRAmp(pi, pj, idMot, idi, idj,
                            mMot, widthQ2, polMot, poli, polj);
      // j is Higgs.
      else if (idj == 25) return vLtovhFSRAmp(pi, pj, idMot, idi, idj,
        mMot, widthQ2, polMot, poli, polj);
      // i is vector.
      else return vLtovvFSRAmp(pi, pj, idMot, idi, idj,
        mMot, widthQ2, polMot, poli, polj);
    }
  }

}

//--------------------------------------------------------------------------

// ISR amplitude selector.

complex AmpCalculator::branchAmpISR(const Vec4& pa, const Vec4& pj, int idA,
  int ida, int idj, double mA, int polA, int pola, int polj) {

  // A is fermion.
  if (idA > 0) {
    // j is Higgs.
    if (idj == 25)
      return ftofhISRAmp(pa, pj, idA, ida, idj, mA, polA, pola, polj);
    // j is vector.
    else
      return ftofvISRAmp(pa, pj, idA, ida, idj, mA, polA, pola, polj);
  // A is antifermion.
  } else {
    // j is Higgs.
    if (idj == 25)
      return fbartofbarhISRAmp(pa, pj, idA, ida, idj, mA, polA, pola, polj);
    // j is vector.
    else
      return fbartofbarvISRAmp(pa, pj, idA, ida, idj, mA, polA, pola, polj);
  }

}

//--------------------------------------------------------------------------

// Compute FF antenna functions from amplitudes for all polarizations.

vector<AntWrapper> AmpCalculator::branchKernelFF(Vec4 pi, Vec4 pj, int idMot,
  int idi, int idj, double mMot, double widthQ2, int polMot) {

  // Find appropriate spins for i and j.
  vector<int> iPols, jPols;
  if (abs(idi) == 25)                        {iPols = scalarPols;}
  else if (abs(idi) == 23 || abs(idi) == 24) {iPols = vectorPols;}
  else                                       {iPols = fermionPols;}
  if (abs(idj) == 25)                        {jPols = scalarPols;}
  else if (abs(idj) == 23 || abs(idj) == 24) {jPols = vectorPols;}
  else                                       {jPols = fermionPols;}

  // Sum over all final-state spins.
  vector<AmpWrapper> amps;
  for (int i = 0; i < (int)iPols.size(); i++)
    for (int j = 0; j < (int)jPols.size(); j++)
      amps.push_back(AmpWrapper(branchAmpFSR(pi, pj, idMot, idi, idj,
        mMot, widthQ2, polMot, iPols[i], jPols[j]), iPols[i], jPols[j]));

  // Square amplitudes and check size.
  vector<AntWrapper> ants;
  for (int i = 0; i < (int)amps.size(); i++) ants.push_back(amps[i].norm());
  if (ants.size() == 0 && verbose >= NORMAL) {
    stringstream ss;
    ss << ": antenna vector is empty.\n"
       << "    idMot = " << idMot << "  idi = " << idi << "  idj = " << idj;
    infoPtr->errorMsg("Warning in " + __METHOD_NAME__, ss.str());
  }
  return ants;

}

//--------------------------------------------------------------------------

// Compute II antenna functions from amplitudes for all polarizations.

vector<AntWrapper> AmpCalculator::branchKernelII(Vec4 pa, Vec4 pj, int idA,
  int ida, int idj, double mA, int polA) {

  // Find appropriate spins for a and j. Current implementation only
  // has f -> fv.
  vector<int> aPols(fermionPols),
    jPols(abs(idj) == 22 ? fermionPols : vectorPols);
  vector<AmpWrapper> amps;
  for (int i = 0; i < (int)aPols.size(); i++)
    for (int j = 0; j < (int)jPols.size(); j++)
      amps.push_back(AmpWrapper(branchAmpISR(pa, pj, idA, ida, idj,
        mA, polA, aPols[i], jPols[j]), aPols[i], jPols[j]));

  // Square amplitudes and check size.
  vector<AntWrapper> ants;
  for (int i = 0; i < (int)amps.size(); i++)  ants.push_back(amps[i].norm());
  if (ants.size() == 0 && verbose >= NORMAL) {
    stringstream ss;
    ss << ": antenna vector is empty.\n"
       << "    idA = " << idA << "  ida = " << ida << "  idj = " << idj;
    infoPtr->errorMsg("Warning in " + __METHOD_NAME__, ss.str());
  }
  return ants;

}

//--------------------------------------------------------------------------

// Initialize an FF antenna function.

void AmpCalculator::initFFAnt(bool va, int id1, int id2, int pol,
  const double& Q2, const double& widthQ2, const double& xi, const double& xj,
  const double& mMot, const double& miIn, const double& mjIn) {

  // Masses and antenna function.
  mi = miIn; mj = mjIn; mMot2 = pow2(mMot); mi2 = pow2(mi); mj2 = pow2(mj);
  Q4gam = pow2(Q2) + mMot2*pow2(widthQ2);
  Q2til = max(0., Q2 + mMot2 - mj2/xj - mi2/xi);
  ant = 0;

  // Couplings.
  initCoup(va, id1, id2, pol, true);

}

//--------------------------------------------------------------------------

// Initialize an II antenna function.

void AmpCalculator::initIIAnt(int id1, int id2, int pol, const double& Q2,
  const double& xA, const double& xj,
  const double& mA, const double& maIn, const double& mjIn) {

  // Masses and antenna function.
  ma = maIn; mj = mjIn; mA2 = pow2(mA); ma2 = pow2(ma); mj2 = pow2(mj);
  Q4 = pow2(Q2);
  Q2til = max(0., Q2 - mA2 + ma2*xA - mj2*xA/xj);
  ant = 0;

  // Couplings.
  initCoup(true, id1, id2, pol, true);

}

//--------------------------------------------------------------------------

// Antenna functions.

// FF: f->fV.

double AmpCalculator::ftofvFFAnt(double Q2, double widthQ2,
    double xi, double xj, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj) {

  // Initialize.
  initFFAnt(true, idMot, idj, polMot, Q2, widthQ2, xi, xj, mMot, miIn, mjIn);

  // Calculate antenna function.
  if (polMot == poli && polMot == polj)
    ant = 2*pow2(vMin) * (Q2til/Q4gam) * (1./xj);
  else if (polMot == poli && polMot == -polj)
    ant = 2*pow2(vMin) * (Q2til/Q4gam) * (pow2(xi)/xj);
  else if (polMot == -poli && polMot == polj)
    ant = 2*pow2( vMin*mi/sqrt(xi) - vPls*mMot*sqrt(xi)) * (1/Q4gam);
  else if (polMot == -poli && polMot == -polj) ant = 0;
  else if (polMot == poli && polj == 0)
    ant = pow2( vMin*( (mMot2/mj)*sqrt(xi) - (mi2/mj)/sqrt(xi)
      - 2*mj*sqrt(xi)/xj ) + vPls*(mMot*mi/mj)*xj/sqrt(xi)) * (1./Q4gam);
  else if (polMot == -poli && polj == 0)
    ant = (pow2( mMot*vPls - mi*vMin )/mj2) * (Q2til/Q4gam) * xj;
  else hmsgFFAnt(polMot, poli, polj);

  // Multiply by CKM matrix - only for q+W.
  if (abs(idj) == 24 && abs(idi) < 7)
    ant *= pow2(vCKM[make_pair(abs(idMot), abs(idi))]);
  return ant;

}

//--------------------------------------------------------------------------

// FF: f->fH.

double AmpCalculator::ftofhFFAnt(double Q2, double widthQ2,
    double xi, double xj, int, int, int,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj) {

  // Initialize.
  initFFAnt(false, 0, 0, 0, Q2, widthQ2, xi, xj, mMot, miIn, mjIn);

  // Calculate antenna function.
  if (polMot == poli) ant = (1./4./sw2) * (pow2(mi2)/mw2) * (1./Q4gam) *
                        pow2(sqrt(xi) + 1./sqrt(xi));
  else if (polMot == -poli) ant = (1./4./sw2) * (mi2/mw2) * (Q2til/Q4gam) * xj;
  else hmsgFFAnt(polMot, poli, polj);
  return ant;

}

//--------------------------------------------------------------------------

// FF: fbar->fbarV.

double AmpCalculator::fbartofbarvFFAnt(double Q2, double widthQ2,
    double xi, double xj, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj) {

  // Initialize.
  initFFAnt(true, idMot, idj, polMot, Q2, widthQ2, xi, xj, mMot, miIn, mjIn);

  // Calculate antenna function.
  if (polMot == poli && polMot == polj)
    ant = 2*pow2(vPls) * (Q2til/Q4gam) * (1./xj);
  else if (polMot == poli && polMot == -polj)
    ant = 2*pow2(vPls) * (Q2til/Q4gam) * (pow2(xi)/xj);
  else if (polMot == -poli && polMot == polj)
    ant = 2*pow2( vPls*mi/sqrt(xi) - vMin*mMot*sqrt(xi)) * (1/Q4gam);
  else if (polMot == -poli && polMot == -polj) ant = 0;
  else if (polMot == poli && polj == 0)
    ant = pow2( vPls*( (mMot2/mj)*sqrt(xi) - (mi2/mj)/sqrt(xi)
        - 2*mj*sqrt(xi)/xj ) + vMin*(mMot*mi/mj)*xj/sqrt(xi)) * (1./Q4gam);
  else if (polMot == -poli && polj == 0)
    ant = (pow2( mMot*vMin - mi*vPls )/mj2) * (Q2til/Q4gam) * xj;
  else hmsgFFAnt(polMot, poli, polj);

  // Multiply by CKM matrix - only for q+W.
  if (abs(idj) == 24 && abs(idi) < 7)
    ant *= pow2(vCKM[make_pair(abs(idMot), abs(idi))]);
  return ant;

}

//--------------------------------------------------------------------------

// FF: fbar->fbarH.

double AmpCalculator::fbartofbarhFFAnt(double Q2, double widthQ2,
    double xi, double xj, int, int, int,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj) {

  // Initialize.
  initFFAnt(false, 0, 0, 0, Q2, widthQ2, xi, xj, mMot, miIn, mjIn);

  // Calculate antenna function.
  if (polMot == poli) ant = (1./4./sw2) * (pow2(mi2)/mw2) * (1./Q4gam) *
                        pow2(sqrt(xi) + 1./sqrt(xi));
  else if (polMot == -poli) ant = (1./4./sw2) * (mi2/mw2) * (Q2til/Q4gam) * xj;
  else hmsgFFAnt(polMot, poli, polj);
  return ant;

}

//--------------------------------------------------------------------------

// FF: V->ffbar.

double AmpCalculator::vtoffbarFFAnt(double Q2, double widthQ2,
    double xi, double xj, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj) {

  // Initialize.
  initFFAnt(true, idi, idMot, polMot, Q2, widthQ2, xi, xj, mMot, miIn, mjIn);

  // Calculate antenna function.
  if (polMot == poli && polMot == polj)
    ant = 2*pow2( vPls*mi*sqrt(xj/xi) + vMin*mj*sqrt(xi/xj) ) * (1./Q4gam);
  else if (polMot == -poli && polMot == polj)
    ant = 2*pow2(vPls) * (Q2til/Q4gam) * pow2(xj);
  else if (polMot == poli && polMot == -polj)
    ant = 2*pow2(vMin) * (Q2til/Q4gam) * pow2(xi);
  else if (polMot == -poli && polMot == -polj) ant = 0;
  else if (polMot == 0 && poli == polj)
    ant = (pow2( (v + poli*a)*mi - (v - poli*a)*mj )/mMot2) * (Q2til/Q4gam);
  else if (polMot == 0 && poli == -polj)
    ant = pow2( (v - poli*a)*(2*mMot*sqrt(xi*xj) - (mi2/mMot) * sqrt(xj/xi)
          - (mj2/mMot) * sqrt(xi/xj))
          + (v + poli*a)*(mi*mj/mMot) * (1./sqrt(xi*xj)) )
          * (1./Q4gam);
  else hmsgFFAnt(polMot, poli, polj);

  // Multiply by CKM matrix - only for W->qqbar.
  if (abs(idMot) == 24 && abs(idi) < 7)
    ant *= pow2(vCKM[make_pair(abs(idi), abs(idj))]);
  return ant;

}

//--------------------------------------------------------------------------

// FF: V->VH.

double AmpCalculator::vtovhFFAnt(double Q2, double widthQ2,
  double xi, double xj, int, int, int,
  double mMot, double miIn, double mjIn, int polMot, int poli, int polj) {

  // Initialize.
  initFFAnt(false, 0, 0, 0, Q2, widthQ2, xi, xj, mMot, miIn, mjIn);

  // Calculate antenna function. Check longitudinals first.
  if (polMot == 0 && poli == 0)
    ant = (1./4./sw2) * (1./mw2) * (1./Q4gam) *
      pow2(mj2 + 2.*mi2*(xi + xj/xi));
  else if (polMot != 0 && poli == 0)
    ant = (1./2./sw2) * (mi2/mw2) * (Q2til/Q4gam) * xi*xj;
  else if (polMot == 0 && poli != 0)
    ant = (1./2./sw2) * (mi2/mw2) * (Q2til/Q4gam) * (xj/xi);
  else if (polMot == poli)  ant = (1./sw2) * (pow2(mi2)/mw2) * (1/Q4gam);
  else if (polMot == -poli) ant = 0;
  else hmsgFFAnt(polMot, poli, polj);
  return ant;

}

//--------------------------------------------------------------------------

// FF: V->VV.

double AmpCalculator::vtovvFFAnt(double Q2, double widthQ2,
    double xi, double xj, int idMot, int idi, int idj,
    double mMot, double miIn, double mjIn, int polMot, int poli, int polj) {

  // Initialize.
  initFFAnt(false, 0, 0, 0, Q2, widthQ2, xi, xj, mMot, miIn, mjIn);

  // Triple vector coupling.
  g = idMot == 23 || idi == 23 || idj == 23 ? (1 - sw2)/sw2 : 1;

  // Calculate antenna function. Triple longitudinal.
  if (polMot == 0 && poli == 0 && polj == 0)
    ant = (g/4.) * (1./mMot2/mi2/mj2) * (1/pow2(xi)/pow2(xj)) * pow2(
          pow2(mMot2)*xi*xj*(2*xj - 1)
        - 2*mMot2*( mi2*pow2(xj)*(1.+xi) - mj2*pow2(xi)*(1.+xj))
        + (mi2 - mj2)*(mi2*xj*(xi+2*xj) + mj2*xi*(2*xi+xj))
        ) * (1/Q4gam);
  // Double longitudinal.
  else if (polMot == 0 && poli == 0 && polj != 0)
    ant = (g/2.) * (pow2(mMot2 + mi2 - mj2)/mMot2/mi2) * (Q2til/Q4gam)
      * (xi/xj);
  else if (polMot == 0 && poli != 0 && polj == 0)
    ant = (g/2.) * (pow2(mMot2 - mi2 + mj2)/mMot2/mj2) * (Q2til/Q4gam)
      * (xj/xi);
  else if (polMot != 0 && poli == 0 && polj == 0)
    ant = (g/2.) * (pow2(mMot2 - mi2 - mj2)/mi2/mj2) * (Q2til/Q4gam) * xi*xj;
  // Single longitudinal.
  else if (polMot == 0 && poli == polj) ant = 0;
  else if (polMot == 0 && poli == -polj)
    ant = g * ( pow2( (1. - 2.*xi)*mMot2 + mi2 - mj2)/mMot2 ) * (1./Q4gam);
  else if (poli == 0 && polMot == polj)
    ant = g * ( pow2( mMot2 - mj2 - (1+xj)/xi*mi2 )/mi2 ) * (1./Q4gam);
  else if (poli == 0 && polMot == -polj) ant = 0;
  else if (polj == 0 && polMot == poli)
    ant = g * ( pow2( mMot2 - mi2 - (1+xi)/xj*mj2 )/mj2 ) * (1./Q4gam);
  else if (polj == 0 && polMot == -poli) ant = 0;
  // All transverse.
  else if (polMot == poli && polMot == polj)
    ant = 2*g * (Q2til/Q4gam) * (1./xi/xj);
  else if (polMot == poli && polMot == -polj)
    ant = 2*g * (Q2til/Q4gam) * pow2(xi) * (xi/xj);
  else if (polMot == -poli && polMot == polj)
    ant = 2*g * (Q2til/Q4gam) * pow2(xj) * (xj/xi);
  else if (polMot == -poli && polMot == -polj) ant = 0;
  else hmsgFFAnt(polMot, poli, polj);
  return ant;

}

//--------------------------------------------------------------------------

// FF: H->ffbar.

double AmpCalculator::htoffbarFFAnt(double Q2, double widthQ2,
  double xi, double xj, int, int, int,
  double mMot, double miIn, double mjIn, int polMot, int poli, int polj) {

  // Initialize.
  initFFAnt(false, 0, 0, 0, Q2, widthQ2, xi, xj, mMot, miIn, mjIn);

  // Calculate antenna function.
  if (poli == polj) ant = (1./4./sw2) * (mi2/mw2) * (Q2til/Q4gam);
  else if (poli == -polj) ant = (1./4./sw2) * (pow2(mi2)/mw2) * (1/Q4gam)
                            * pow2(sqrt(xi/xj) - sqrt(xj/xi));
  else hmsgFFAnt(polMot, poli, polj);
  return ant;

}

//--------------------------------------------------------------------------

// FF: H->VV.

double AmpCalculator::htovvFFAnt(double Q2, double widthQ2,
  double xi, double xj, int, int, int,
  double mMot, double miIn, double mjIn, int, int poli, int polj) {

  // Initialize.
  initFFAnt(false, 0, 0, 0, Q2, widthQ2, xi, xj, mMot, miIn, mjIn);

  // Calculate antenna function.
  if (poli == 0 && polj == 0)
    ant = (1./4./sw2) / mw2 / Q4gam * pow2(mMot2 - 2*mi2*(1./xi/xj - 1.));
  else if (poli == 0 && polj != 0)
    ant = (1./2./sw2)  * (mi2/mw2) * (Q2til/Q4gam) * (xi/xj);
  else if (poli != 0 && polj == 0)
    ant = (1./2./sw2)  * (mi2/mw2) * (Q2til/Q4gam) * (xj/xi);
  else if (poli == polj) ant = 0;
  else ant = (1./sw2) * (pow2(mi2)/mw2) / Q4gam;
  return ant;

}

//--------------------------------------------------------------------------

// FF: H->HH.

double AmpCalculator::htohhFFAnt(double Q2, double widthQ2,
    double xi, double xj, int, int, int,
    double mMot, double miIn, double mjIn, int, int, int) {
  initFFAnt(false, 0, 0, 0, Q2, widthQ2, xi, xj, mMot, miIn, mjIn);
  return (9./4./sw2) * (pow2(mMot2)/mw2) / Q4gam;
}

//--------------------------------------------------------------------------

// II: f->fV.

double AmpCalculator::ftofvIIAnt(double Q2, double xA, double xj,
  int idA, int ida, int idj, double mA, double maIn, double mjIn,
  int polA, int pola, int polj) {

  // Initialize.
  initIIAnt(idA, idj, polA, Q2, xA, xj, mA, maIn, mjIn);

  // Calculate antenna function.
  if (polA == pola && polA == polj)
    ant = 2*pow2(vMin) * (Q2til/Q4) / xj / xA;
  else if (polA == pola && polA == -polj)
    ant = 2*pow2(vMin) * (Q2til/Q4) * xA/xj;
  else if (polA == -pola && polA == polj)
    ant = 2*pow2( vMin*mA/sqrt(xA) - vPls*ma*sqrt(xA)) / Q4;
  else if (polA == -pola && polA == -polj) ant = 0;
  else if (polA == pola && polj == 0)
    ant = pow2( vMin*( (ma2/mj)*sqrt(xA) - (mA2/mj)/sqrt(xA)
        - 2*mj*sqrt(xA)/xj) + vPls*(ma*mA/mj)*xj/sqrt(xA) ) / Q4;
  else if (polA == -pola && polj == 0)
    ant = (pow2( mA*vMin - ma*vPls )/mj2) * (Q2til/Q4) * xj/xA;
  else hmsgIIAnt(polA, pola, polj);

  // Multiply by CKM matrix - only for q+W.
  if (abs(idj) == 24 && abs(ida) < 7)
    ant *= pow2(vCKM[make_pair(abs(idA), abs(ida))]);
  return ant;

}

//--------------------------------------------------------------------------

// II: fbar->fbarV.

double AmpCalculator::fbartofbarvIIAnt(double Q2, double xA, double xj,
  int idA, int ida, int idj, double mA, double maIn, double mjIn,
  int polA, int pola, int polj) {

  // Initialize.
  initIIAnt(idA, idj, polA, Q2, xA, xj, mA, maIn, mjIn);

  // Calculate antenna function.
  if (polA == pola && polA == polj) ant = 2*pow2(vPls) * (Q2til/Q4) /xj/xA;
  else if (polA == pola && polA == -polj)
    ant = 2*pow2(vPls) * Q2til/Q4 * xA/xj;
  else if (polA == -pola && polA == polj)
    ant = 2*pow2( vPls*mA/sqrt(xA) - vMin*ma*sqrt(xA)) / Q4;
  else if (polA == -pola && polA == -polj) ant = 0;
  else if (polA == pola && polj == 0)
    ant = pow2( vPls*( (ma2/mj)*sqrt(xA) - mA2/mj/sqrt(xA)
          - 2*mj*sqrt(xA)/xj) + vMin*(ma*mA/mj)*xj/sqrt(xA) ) / Q4;
  else if (polA == -pola && polj == 0)
    ant = (pow2( mA*vPls - ma*vMin )/mj2) * (Q2til/Q4) * (xj/xA);
  else hmsgIIAnt(polA, pola, polj);

  // Multiply by CKM matrix - only for q+W.
  if (abs(idj) == 24 && abs(ida) < 7)
    ant *= pow2(vCKM[make_pair(abs(idA), abs(ida))]);
  return ant;

}

//--------------------------------------------------------------------------

// Antenna-function selector for FF branchings.

double AmpCalculator::antFuncFF(double Q2, double widthQ2, double xi,
  double xj, int idMot, int idi, int idj, double mMot, double miIn,
  double mjIn, int polMot, int poli, int polj) {
  ant = 0;
  // I is fermion.
  if (abs(idMot) < 20 && idMot > 0) {
    // j is Higgs.
    if (idj == 25) ant = ftofhFFAnt(Q2, widthQ2, xi, xj, idMot, idi, idj,
      mMot, miIn, mjIn, polMot, poli, polj);
    // j is vector.
    else ant = ftofvFFAnt(Q2, widthQ2, xi, xj, idMot, idi, idj,
      mMot, miIn, mjIn, polMot, poli, polj);
  // I is antifermion.
  } else if (abs(idMot) < 20 && idMot < 0) {
    // j is Higgs.
    if (idj == 25) ant = fbartofbarhFFAnt(Q2, widthQ2, xi, xj, idMot, idi, idj,
      mMot, miIn, mjIn, polMot, poli, polj);
    // j is vector.
    else ant = fbartofbarvFFAnt(Q2, widthQ2, xi, xj, idMot, idi, idj,
      mMot, miIn, mjIn, polMot, poli, polj);
  // I is Higgs.
  } else if (idMot == 25) {
    // i is Higgs.
    if (idi == 25) ant = htohhFFAnt(Q2, widthQ2, xi, xj, idMot, idi, idj,
      mMot, miIn, mjIn, polMot, poli, polj);
    // i is fermion.
    else if (abs(idi) < 20) ant = htoffbarFFAnt(Q2, widthQ2, xi, xj, idMot,
      idi, idj, mMot, miIn, mjIn, polMot, poli, polj);
    // i is vector.
    else ant = htovvFFAnt(Q2, widthQ2, xi, xj, idMot, idi, idj,
      mMot, miIn, mjIn, polMot, poli, polj);
  // I is vector.
  } else {
    // i is fermion.
    if (abs(idi) < 20) ant = vtoffbarFFAnt(Q2, widthQ2, xi, xj, idMot, idi,
      idj, mMot, miIn, mjIn, polMot, poli, polj);
    // j is Higgs.
    else if (idj == 25) ant = vtovhFFAnt(Q2, widthQ2, xi, xj, idMot, idi,
      idj, mMot, miIn, mjIn, polMot, poli, polj);
    // i is vector.
    else ant = vtovvFFAnt(Q2, widthQ2, xi, xj, idMot, idi,
      idj, mMot, miIn, mjIn, polMot, poli, polj);
  }

  // Add on factor 3 to fix colour counting for quarks.
  if (abs(idi) < 7 && abs(idj) < 7) ant *= 3;
  return ant;

}

//--------------------------------------------------------------------------

// FF antenna function calculator for all outgoing polarizations.

vector<AntWrapper> AmpCalculator::antFuncFF(double Q2, double widthQ2,
  double xi, double xj, int idMot, int idi, int idj, double mMot, double miIn,
  double mjIn, int polMot) {

  // Find appropriate spins for i and j.
  vector<int> iPols, jPols;
  if (abs(idi) == 25)                         {iPols = scalarPols;}
  else if (abs(idi) == 23 || abs(idi) == 24)  {iPols = vectorPols;}
  else                                        {iPols = fermionPols;}
  if (abs(idj) == 25)                         {jPols = scalarPols;}
  else if (abs(idj) == 23 || abs(idj) == 24)  {jPols = vectorPols;}
  else                                        {jPols = fermionPols;}

  // Sum over all final-state spins.
  vector<AntWrapper> ants;
  for (int i = 0; i < (int)iPols.size(); i++)
      for (int j = 0; j < (int)jPols.size(); j++)
          ants.push_back(AntWrapper(antFuncFF(Q2, widthQ2, xi, xj, idMot, idi,
            idj, mMot, miIn, mjIn, polMot, iPols[i], jPols[j]), iPols[i],
            jPols[j]));
  return ants;

}

//--------------------------------------------------------------------------

// Antenna-function selector for II branchings.

double AmpCalculator::antFuncII(double Q2, double xA, double xj,
  int idA, int ida, int idj, double mA, double maIn, double mjIn,
  int polA, int pola, int polj) {
  ant = 0;
  // Fermion branching.
  if (idA > 0) ant = ftofvIIAnt(Q2, xA, xj, idA, ida, idj,
    mA, maIn, mjIn, polA, pola, polj);
  // Antifermion branching.
  else ant = fbartofbarvIIAnt(Q2, xA, xj, idA, ida, idj,
    mA, maIn, mjIn, polA, pola, polj);
  return ant;

}

//--------------------------------------------------------------------------

// II antenna function calculator for all outgoing polarizations.

vector<AntWrapper> AmpCalculator::antFuncII(double Q2, double xA, double xj,
  int idA, int ida, int idj, double mA, double maIn, double mjIn, int polA) {

  // Find appropriate spins for a and j. Current implementation only
  // has f -> fv.
  vector<int> aPols(fermionPols);
  vector<int> jPols(abs(idj) == 22 ? fermionPols : vectorPols);

  // Sum over all final-state spins.
  vector<AntWrapper> ants;
  for (int i = 0; i < (int)aPols.size(); i++)
      for (int j = 0; j < (int)jPols.size(); j++)
          ants.push_back(AntWrapper(antFuncII(Q2, xA, xj, idA, ida, idj,
              mA, maIn, mjIn, polA, aPols[i], jPols[j]), aPols[i], jPols[j]));
  return ants;

}

//--------------------------------------------------------------------------

// Check for zero denominator in an FSR splitting kernel.

bool AmpCalculator::zdenFSRSplit(const string& method, const double& Q2,
  const double& z, bool check) {
  if (check || z == 0 || z == 1 || Q2 == 0) {
    if (verbose >= NORMAL) {
      stringstream ss;
      ss << ": zero denominator encountered.\n"
         << "   z = " << z << " Q2  = " << Q2 << " mj = " << mj;
      infoPtr->errorMsg("Warning in " + method, ss.str());
    } return true;
  }
  Q4 = pow2(Q2);
  Q2til = Q2 + mMot2 - mj2/(1-z) - mi2/z;
  return false;
}

//--------------------------------------------------------------------------

// Check for zero denominator in an ISR splitting kernel.

bool AmpCalculator::zdenISRSplit(const string& method, const double& Q2,
  const double& z, bool flip, bool check) {
  if (check || z == 0 || z == 1 || Q2 == 0) {
    if (verbose >= NORMAL) {
      stringstream ss;
      ss << ": zero denominator encountered.\n"
         << "   z = " << z << " Q2  = " << Q2 << " mj = " << mj;
      infoPtr->errorMsg("Warning in " + method, ss.str());
    } return true;
  }
  Q4 = pow2(Q2);
  Q2til = flip ? Q2 + mA2 - ma2/z - mj2/(1-z) : Q2 - mA2 + ma2*z - mj2*z/(1-z);
  return false;
}

//--------------------------------------------------------------------------

// Splitting functions.

// FSR: f->fV.

double AmpCalculator::ftofvFSRSplit(double Q2, double z, int idMot, int,
  int idj, double mMot, double miIn, double mjIn, int polMot, int poli,
  int polj) {

  // Initialize.
  initFSRSplit(true, idMot, idj, polMot, mMot, miIn, mjIn);
  if (zdenFSRSplit(__METHOD_NAME__, Q2, z,
    ( mj ==0. &&  (idj == 23 || abs(idj) == 24) ))) return 0;

  // Calculate kernel.
  if (poli == polMot && polj == polMot)
    return 2*pow2(vMin)*Q2til/(1-z)/pow2(Q2);
  else if (poli == polMot && polj == -polMot)
    return 2*pow2(vMin)*Q2til*z*z/(1-z)/pow2(Q2);
  else if (poli == -polMot && polj == polMot)
    return 2*pow2(mMot*vPls*sqrt(z) - mi*vMin/sqrt(z))/pow2(Q2);
  else if (poli == -polMot && polj == -polMot) return 0;
  else if (poli == polMot && polj == 0)
    return pow2(vMin*(pow2(mMot)/mj*sqrt(z) - pow2(mi)/mj/sqrt(z)
    - 2*mj*sqrt(z)/(1-z)) + vPls*mi*mMot/mj*(1-z)/sqrt(z))/pow2(Q2);
  else if (poli == -polMot && polj == 0)
    return pow2(mi/mj*vMin - mMot/mj*vPls)*(1-z)*Q2til/pow2(Q2);
  else hmsgFSRSplit(polMot, poli, polj);
  return 0;

}


//--------------------------------------------------------------------------

// FSR: f->fH.

double AmpCalculator::ftofhFSRSplit(double Q2, double z, int idMot, int,
  int idj, double mMot, double, double mjIn, int polMot, int poli, int polj) {

  // Initialize.
  initFSRSplit(false, idMot, idj, polMot, mMot, mMot, mjIn);
  if (zdenFSRSplit(__METHOD_NAME__, Q2, z, false)) return 0;

  // Calculate kernel.
  if (polMot == poli) return pow2(mMot*g)*mMot2*pow2(sqrt(z) + 1./sqrt(z))/Q4;
  else if (polMot == -poli) return pow2(mMot*g)*(1-z)*Q2til/Q4;
  else hmsgFSRSplit(polMot, poli, polj);
  return 0;

}

//--------------------------------------------------------------------------

// FSR: fbar->fbarV.

double AmpCalculator::fbartofbarvFSRSplit(double Q2, double z, int idMot,
  int, int idj, double mMot, double miIn, double mjIn, int polMot,
  int poli, int polj) {

  // Initialize.
  initFSRSplit(true, idMot, idj, polMot, mMot, miIn, mjIn);
  if (zdenFSRSplit(__METHOD_NAME__, Q2, z,
    ( mj == 0 &&  (idj == 23 || abs(idj) == 24) ) )) return 0;

  // Calculate kernel.
  if (poli == polMot && polj == polMot) return 2*pow2(vPls)*Q2til/(1-z)/Q4;
  else if (poli == polMot && polj == -polMot)
    return 2*pow2(vPls)*Q2til*z*z/(1-z)/Q4;
  else if (poli == -polMot && polj == polMot)
    return 2*pow2(mMot*vMin*sqrt(z) - mi*vPls/sqrt(z))/Q4;
  else if (poli == -polMot && polj == -polMot) return 0;
  else if (poli == polMot && polj == 0)
    return pow2(vPls*(pow2(mMot)/mj*sqrt(z) - pow2(mi)/mj/sqrt(z)
    - 2*mj*sqrt(z)/(1-z)) + vMin*mi*mMot/mj*(1-z)/sqrt(z))/Q4;
  else if (poli == -polMot && polj == 0)
    return pow2(mi/mj*vPls - mMot/mj*vMin)*(1-z)*Q2til/Q4;
  else hmsgFSRSplit(polMot, poli, polj);
  return 0;

}

//--------------------------------------------------------------------------

// FSR: fbar->fbarH.

double AmpCalculator::fbartofbarhFSRSplit(double Q2, double z, int idMot,
  int, int idj, double mMot, double, double mjIn, int polMot,
  int poli, int polj) {

  // Initialize.
  initFSRSplit(false, idMot, idj, polMot, mMot, mMot, mjIn);
  if (zdenFSRSplit(__METHOD_NAME__, Q2, z, false)) return 0;

  // Calculate kernel.
  if (polMot == poli)
    return pow2(mMot*g)*mMot2*pow2(sqrt(z) + 1./sqrt(z))/pow2(Q2);
  else if (polMot == -poli) return pow2(mMot*g)*(1-z)*Q2til/pow2(Q2);
  else hmsgFSRSplit(polMot, poli, polj);
  return 0;

}

//--------------------------------------------------------------------------

// FSR: VT->ffbar.

double AmpCalculator::vTtoffbarFSRSplit(double Q2, double z, int idMot,
  int idi, int, double mMot, double miIn, double mjIn, int polMot,
  int poli, int polj) {

  // Initialize.
  initFSRSplit(true, idi, idMot, polMot, mMot, miIn, mjIn);
  if (zdenFSRSplit(__METHOD_NAME__, Q2, z, false)) return 0;

  // Calculate kernel.
  if (polMot == poli && polMot == polj) return 0;
  else if (polMot == -poli && polMot == polj)
      return 2*pow2(vMin)*pow2(1-z)*Q2til/Q4;
  else if (polMot == poli && polMot == -polj)
      return 2*pow2(vPls)*pow2(z)*Q2til/Q4;
  else if (polMot == -poli && polMot == -polj)
      return 2*pow2(mi*vMin*sqrt((1-z)/z) + mj*vPls*sqrt(z/(1-z)))/Q4;
  else hmsgFSRSplit(polMot, poli, polj);
  return 0;

}

//--------------------------------------------------------------------------

// FSR: VT->ffbar.

double AmpCalculator::vTtovhFSRSplit(double Q2, double z, int idMot, int,
  int idj, double mMot, double, double mjIn, int polMot, int poli,
  int polj) {

  // Initialize.
  initFSRSplit(false, idMot, idj, polMot, mMot, mMot, mjIn);
  if (zdenFSRSplit(__METHOD_NAME__, Q2, z, false)) return 0;

  // Calculate kernel.
  if (polMot == poli) return pow2(g)/Q4;
  else if (polMot == -poli) return 0;
  else if (poli == 0)
      return pow2(g*sqrt((1-z)*z)/mMot/sqrt(2))*Q2til/Q4;
  else hmsgFSRSplit(polMot, poli, polj);
  return 0;

}

//--------------------------------------------------------------------------

// FSR: VT->VV.

double AmpCalculator::vTtovvFSRSplit(double Q2, double z, int idMot, int idi,
  int idj, double mMot, double miIn, double mjIn, int polMot, int poli,
  int polj) {

  // Initialize.
  initFSRSplit(false, idMot, idj, polMot, mMot, miIn, mjIn);
  if (zdenFSRSplit(__METHOD_NAME__, Q2, z,
    ( mi == 0. && (idi == 23 || abs(idi) == 24) ) ||
    ( mj == 0. && (idj == 23 || abs(idj) == 24) ) )) return 0;

  // Calculate kernel. Double longitudinal.
  if (poli == 0 && polj == 0)
    return 0.5*pow2(g)*pow2((mMot2 - mi2 - mj2)/mi/mj)*z*(1-z)*Q2til/Q4;
  // Single longitudinal.
  else if (poli == 0 && polMot == polj) return pow2(g)*pow2(mi*(1 + 2*(1-z)/z)
    + pow2(mj)/mi - pow2(mMot)/mi)/Q4;
  else if (poli == 0 && polMot == -polj) return 0;
  else if (polj == 0 && polMot == poli) return pow2(g)*pow2(mj*(1 + 2*z/(1-z))
    + pow2(mi)/mj - pow2(mMot)/mj)/Q4;
  else if (polj == 0 && polMot == -poli) return 0;
  // All transverse.
  else if (polMot == poli && polMot == polj)
    return 2*pow2(g)/z/(1-z)*Q2til/Q4;
  else if (polMot == poli && polMot == -polj)
    return 2*pow2(g)*pow3(z)/(1-z)*Q2til/Q4;
  else if (polMot == -poli && polMot == polj)
    return 2*pow2(g)*pow3(1-z)/z*Q2til/Q4;
  else if (polMot == -poli && polMot == -polj) return 0;
  else hmsgFSRSplit(polMot, poli, polj);
  return 0;

}

//--------------------------------------------------------------------------

// FSR: VL->ffbar.

double AmpCalculator::vLtoffbarFSRSplit(double Q2, double z, int idMot,
  int idi, int, double mMot, double miIn, double mjIn, int polMot,
  int poli, int polj) {

  // Initialize.
  initFSRSplit(true, idi, idMot, 1, mMot, miIn, mjIn);
  if (zdenFSRSplit(__METHOD_NAME__, Q2, z, false)) return 0;

  // Calculate kernel.
  if (poli == polj)
    return pow2( (mi*vPls - mj*vMin)/mMot )*Q2til/pow2(Q2);
  else if (poli == -polj) return pow2( vMin*pow2(mi)/mMot*sqrt((1-z)/z)
    + vMin*pow2(mj)/mMot*sqrt(z/(1-z)) - vPls*mi*mj/mMot/sqrt(z*(1-z))
    - 2*vMin*mMot*sqrt(z*(1-z)) ) / pow2(Q2);
  else hmsgFSRSplit(polMot, poli, polj);
  return 0;

}

//--------------------------------------------------------------------------

// FSR: VL->VH.

double AmpCalculator::vLtovhFSRSplit(double Q2, double z, int idMot, int,
  int idj, double mMot, double, double mjIn, int polMot, int poli, int) {

  // Initialize.
  initFSRSplit(false, idMot, idj, polMot, mMot, mMot, mjIn);
  if (zdenFSRSplit(__METHOD_NAME__, Q2, z, false)) return 0;

  // Calculate kernel.
  if (poli == 0)
    return pow2(g*(0.5*pow2(mj)/pow2(mMot) + z + (1-z)/z))/pow2(Q2);
  else
    return pow2(g*sqrt((1-z)/z)/mMot/sqrt(2))*Q2til/pow2(Q2);

}

//--------------------------------------------------------------------------

// FSR: VL->VV.

double AmpCalculator::vLtovvFSRSplit(double Q2, double z, int idMot, int idi,
  int idj, double mMot, double miIn, double mjIn, int polMot, int poli,
  int polj) {

  // Initialize.
  initFSRSplit(false, idMot, idj, polMot, mMot, miIn, mjIn);
  if (zdenFSRSplit(__METHOD_NAME__, Q2, z,
    ( mi == 0. &&  (idi == 23 || abs(idi) == 24) ) ||
    ( mj == 0. &&  (idj == 23 || abs(idj) == 24) ) )) return 0;

  // Calculate kernel. Double longitudinal.
  if (poli == 0 && polj == 0)
    return pow2(g)*pow2(0.5*pow3(mMot)/mi/mj * (2*z-1)
      - pow3(mi)/mj/mMot*(0.5 + (1-z)/z) + pow3(mj)/mi/mMot*(0.5 + z/(1-z))
          + mi*mj/mMot*( (1-z)/z - z/(1-z) ) + mMot*mi/mj*(1-z)*(2 + (1-z)/z)
      - mMot*mj/mi*z*(2 + z/(1-z)) )/pow2(Q2);
  // Single longitudinal.
  else if (poli == 0)
    return 0.5 * pow2(g) * pow2( (mMot2 + mi2 - mj2)/mMot/mi ) / (1-z)*z
      * Q2til / pow2(Q2);
  else if (polj == 0)
    return 0.5 * pow2(g) * pow2( (mMot2 - mi2 + mj2)/mMot/mj ) * (1-z)/z
      * Q2til / pow2(Q2);
  // All transverse.
  else if (poli == polj) return 0;
  else if (poli == -polj)
    return pow2(g)*pow2(mMot*(1 - 2*z) - mj2/mMot + mi2/mMot)/pow2(Q2);
  else hmsgFSRSplit(polMot, poli, polj);
  return 0;

}

//--------------------------------------------------------------------------

// FSR: H->ffbar.

double AmpCalculator::htoffbarFSRSplit(double Q2, double z, int idMot, int idi,
  int, double mMot, double miIn, double mjIn, int polMot, int poli, int polj) {

  // Initialize.
  initFSRSplit(false, idi, idMot, polMot, mMot, miIn, mjIn);
  if (zdenFSRSplit(__METHOD_NAME__, Q2, z, false)) return 0;

  // Calculate kernel.
  if (poli == polj) return pow2(mi*g)*Q2til/pow2(Q2);
  else if (poli == -polj)
    return pow2(mi*g)*pow2(mi*(1-z) - mj*z)/z/(1-z)/pow2(Q2);
  else hmsgFSRSplit(polMot, poli, polj);
  return 0;

}

//--------------------------------------------------------------------------

// FSR: H->VV.

double AmpCalculator::htovvFSRSplit(double Q2, double z, int idMot, int idi,
  int, double mMot, double miIn, double mjIn, int polMot, int poli, int polj) {

  // Initialize.
  initFSRSplit(false, idi, idMot, polMot, mMot, miIn, mjIn);
  if (zdenFSRSplit(__METHOD_NAME__, Q2, z, mi == 0 || mj == 0)) return 0;

  // Calculate kernel.  Double longitudinal.
  if (poli == 0 && polj == 0) return pow2(g*( 0.5*(mMot2 - mi2 - mj2)
      - mi2*(1-z)/z - mj2*z/(1-z)))/mi2/mj2/pow2(Q2);
  // Single longitudinal.
  else if (poli == 0)
    return pow2(g*sqrt(z/(1-z))/mi/sqrt(2.))*Q2til/pow2(Q2);
  else if (polj == 0)
    return pow2(g*sqrt((1-z)/z)/mj/sqrt(2.))*Q2til/pow2(Q2);
  // Transverse.
  else if (poli == polj) return 0;
  else if (poli == -polj) return pow2(g)/pow2(Q2);
  else hmsgFSRSplit(polMot, poli, polj);
  return 0;

}

//--------------------------------------------------------------------------

// FSR: H->HH.

double AmpCalculator::htohhFSRSplit(double Q2, double, int idMot, int idi,
  int, double, double, double, int, int, int) {

  // Initialize and calculate kernel.
  g = gMap[make_pair(abs(idi), idMot)];
  if (zdenFSRSplit(__METHOD_NAME__, Q2, 0.5, false)) return 0;
  return pow2(g)/pow2(Q2);

}

//--------------------------------------------------------------------------

// ISR: f->fV.

double AmpCalculator::ftofvISRSplit(double Q2, double z, int idA, int,
  int idj, double mA, double maIn, double mjIn, int polA, int pola, int polj) {

  // Initialize.
  initISRSplit(true, idA, idj, polA, mA, maIn, mjIn);
  if (zdenISRSplit(__METHOD_NAME__, Q2, z, false,
    (mj == 0. &&  (idj == 23 || abs(idj) == 24) ))) return 0;

  // Calculate kernel.
  if (pola == polA && polj == polA) return 2*pow2(vMin)*Q2til/(1-z)/pow2(Q2)/z;
  else if (pola == polA && polj == -polA)
    return 2*pow2(vMin)*Q2til*z*z/(1-z)/pow2(Q2)/z;
  else if (pola == -polA && polj == polA)
    return 2*pow2(mA*vPls*sqrt(z) - ma*vMin/sqrt(z))/pow2(Q2)/z;
  else if (pola == -polA && polj == -polA) return 0;
  else if (pola == polA && polj == 0)
    return pow2(vMin*(pow2(mA)/mj*sqrt(z) - pow2(ma)/mj/sqrt(z) -
      2*mj*sqrt(z)/(1-z)) + vPls*ma*mA/mj*(1-z)/sqrt(z))/pow2(Q2);
  else if (pola == -polA && polj == 0)
    return pow2(ma/mj*vMin - mA/mj*vPls)*(1-z)*Q2til/pow2(Q2)/z;
  else hmsgFSRSplit(polA, pola, polj);
  return 0;

}

//--------------------------------------------------------------------------

// ISR: f->fH.

double AmpCalculator::ftofhISRSplit(double Q2, double z, int idA, int,
  int idj, double mA, double, double mjIn, int polA, int pola, int polj) {

  // Initialize.
  initISRSplit(false, idA, idj, polA, mA, mA, mjIn);
  if (zdenISRSplit(__METHOD_NAME__, Q2, z, true, false)) return 0;

  // Calculate kernel.
  if (polA == pola) return pow2(mA*g)*mA2*pow2(sqrt(z) + 1./sqrt(z))/Q4/z;
  else if (polA == -pola) return pow2(mA*g)*(1-z)*Q2til/Q4/z;
  else hmsgFSRSplit(polA, pola, polj);
  return 0;

}

//--------------------------------------------------------------------------

// ISR: fbar->fbarV.

double AmpCalculator::fbartofbarvISRSplit(double Q2, double z, int idA, int,
  int idj, double mA, double maIn, double mjIn, int polA, int pola, int polj) {

  // Initialize.
  initISRSplit(true, idA, idj, polA, mA, maIn, mjIn);
  if (zdenISRSplit(__METHOD_NAME__, Q2, z, false, false)) return 0;

  // Calculate kernel.
  if (pola == polA && polj == polA) return 2*pow2(vPls)*Q2til/(1-z)/Q4/z;
  else if (pola == polA && polj == -polA)
    return 2*pow2(vPls)*Q2til*z*z/(1-z)/Q4/z;
  else if (pola == -polA && polj == polA)
    return 2*pow2(mA*vMin*sqrt(z) - ma*vPls/sqrt(z))/Q4/z;
  else if (pola == -polA && polj == -polA)  return 0;
  else if (pola == polA && polj == 0)
    return pow2(vPls*(pow2(mA)/mj*sqrt(z) - pow2(ma)/mj/sqrt(z) -
      2*mj*sqrt(z)/(1-z)) + vPls*ma*mA/mj*(1-z)/sqrt(z))/Q4;
  else if (pola == -polA && polj == 0)
    return pow2(ma/mj*vPls - mA/mj*vMin)*(1-z)*Q2til/Q4/z;
  else hmsgFSRSplit(polA, pola, polj);
  return 0;

}

//--------------------------------------------------------------------------

// ISR: fbar->fbarH.

double AmpCalculator::fbartofbarhISRSplit(double Q2, double z, int idA,
  int, int idj, double mA, double, double mjIn, int polA, int pola, int polj) {

  // Initialize.
  initISRSplit(false, idA, idj, polA, mA, mA, mjIn);
  if (zdenISRSplit(__METHOD_NAME__, Q2, z, true, false)) return 0;

  // Calculate kernel.
  if (polA == pola)
    return pow2(mA*g)*mA2*pow2(sqrt(z) + 1./sqrt(z))/Q4/z;
  else if (polA == -pola) return pow2(mA*g)*(1-z)*Q2til/Q4/z;
  else hmsgFSRSplit(polA, pola, polj);
  return 0;

}

//--------------------------------------------------------------------------

// Splitting-function selector for FSR.

double AmpCalculator::splitFuncFSR(double Q2, double z, int idMot, int idi,
  int idj, double mMot, double miIn, double mjIn, int polMot, int poli,
  int polj) {
  double P;

  // I is fermion.
  if (abs(idMot) < 20 && idMot > 0) {
    // j is Higgs.
    if (idj == 25) P = ftofhFSRSplit(Q2, z, idMot, idi, idj, mMot, miIn, mjIn,
      polMot, poli, polj);
    // j is vector.
    else P = ftofvFSRSplit(Q2, z, idMot, idi, idj, mMot, miIn, mjIn, polMot,
      poli, polj);
  // I is antifermion.
  } else if (abs(idMot) < 20 && idMot < 0) {
    // j is Higgs.
    if (idj == 25) P = fbartofbarhFSRSplit(Q2, z, idMot, idi, idj, mMot, miIn,
      mjIn, polMot, poli, polj);
    // j is vector.
    else P = fbartofbarvFSRSplit(Q2, z, idMot, idi, idj, mMot, miIn, mjIn,
      polMot, poli, polj);
  // I is higgs.
  } else if (idMot == 25) {
    // i is higgs.
    if (idi == 25) P = htohhFSRSplit(Q2, z, idMot, idi, idj, mMot, miIn, mjIn,
      polMot, poli, polj);
    // i is fermion.
    else if (abs(idi) < 20) P = htoffbarFSRSplit(Q2, z, idMot, idi, idj, mMot,
      miIn, mjIn, polMot, poli, polj);
    // i is vector.
    else P = htovvFSRSplit(Q2, z, idMot, idi, idj, mMot, miIn, mjIn, polMot,
      poli, polj);
  // I is vector.
  } else {
    // I is transverse.
    if (polMot!=0) {
      // i is fermion.
      if (abs(idi) < 20) P = vTtoffbarFSRSplit(Q2, z, idMot, idi, idj, mMot,
        miIn, mjIn, polMot, poli, polj);
      // j is Higgs.
      else if (idj == 25) P = vTtovhFSRSplit(Q2, z, idMot, idi, idj, mMot,
        miIn, mjIn, polMot, poli, polj);
      // i is vector.
      else P = vTtovvFSRSplit(Q2, z, idMot, idi, idj, mMot, miIn, mjIn, polMot,
        poli, polj);
    // I is longitudinal.
    } else {
      // i is fermion.
      if (abs(idi) < 20) P = vLtoffbarFSRSplit(Q2, z, idMot, idi, idj, mMot,
        miIn, mjIn, polMot, poli, polj);
      // j is Higgs.
      else if (idj == 25) P = vLtovhFSRSplit(Q2, z, idMot, idi, idj, mMot,
        miIn, mjIn, polMot, poli, polj);
      // i is vector.
      else P = vLtovvFSRSplit(Q2, z, idMot, idi, idj, mMot, miIn, mjIn, polMot,
        poli, polj);
    }
  }

  // Factor 3 for splitting to quarks.
  if (abs(idi) < 7 && abs(idj) < 7 && abs(idMot) > 7) P *= 3;
  return P;

}

//--------------------------------------------------------------------------

// Splitting-function selector for ISR.

double AmpCalculator::splitFuncISR(double Q2, double z, int idA, int ida,
  int idj, double mA, double maIn, double mjIn, int polA, int pola, int polj) {
  double P;

  // A is fermion.
  if (idA > 0) {
    // j is Higgs.
    if (idj == 25) P = ftofhISRSplit(Q2, z, idA, ida, idj, mA, maIn, mjIn,
      polA, pola, polj);
    // j is vector.
    else P = ftofvISRSplit(Q2, z, idA, ida, idj, mA, maIn, mjIn, polA, pola,
      polj);
  // A is antifermion.
  } else {
    // j is Higgs.
    if (idj == 25) P = fbartofbarhISRSplit(Q2, z, idA, ida, idj, mA, maIn,
      mjIn, polA, pola, polj);
    // j is vector.
    else P = fbartofbarvISRSplit(Q2, z, idA, ida, idj, mA, maIn, mjIn,
      polA, pola, polj);
  }

  return P;
}

//--------------------------------------------------------------------------

// Decay width calculators.

// Compute Partial Width.

double AmpCalculator::getPartialWidth(int idMot, int idi, int idj,
  double mMot, int polMot) {
  // Compute the partial width for a single decay channel.
  double partialWidth = 0;

  double yi = pow2(dataPtr->mass(idi))/pow2(mMot);
  double yj = pow2(dataPtr->mass(idj))/pow2(mMot);
  double y0 = pow2(dataPtr->mass(idMot))/pow2(mMot);

  // Check if there is any phase space available.
  if (kallenFunction(1, yi, yj) < 0 || yi > 1 || yj > 1) return 0;

  // Compute values of running couplings.
  double alpha  = alphaPtr->alphaEM(pow2(mMot));
  double alphaS = alphaSPtr->alphaS(pow2(mMot));

  // W or Z width.
  if (abs(idMot) == 23 || abs(idMot) == 24) {
    double v2 = pow2(vMap[make_pair(abs(idi), abs(idMot))]);
    double a2 = pow2(aMap[make_pair(abs(idi), abs(idMot))]);

    // Longitudinal Z or W.
    if (polMot == 0) partialWidth = (alpha/6.)*mMot*sqrt(
      kallenFunction(1,yi,yj))*((v2 + a2)*(2 - 3*(yi + yj) + pow2(yi - yj))
      + 6*(v2 - a2)*sqrt(yi*yj) );
    // Transverse Z or W.
    else  partialWidth = (alpha/3.)*mMot*sqrt(kallenFunction(1,yi,yj))*(
      (v2 + a2)*(1 - pow2(yi - yj)) + 3*(v2 - a2)*sqrt(yi*yj) );

    // Quark correction.
    if (abs(idi) < 7) partialWidth *= 3.*(1. + alphaS / M_PI);

    // CKM matrix.
    if (abs(idMot) == 24 && abs(idi) < 7)
      partialWidth *= pow2(vCKM[make_pair(abs(idi), abs(idj))]);
  // Higgs width.
  } else if (abs(idMot) == 25) {
    // xi = xj always.
    partialWidth = (alpha/8./sw2) * pow3(mMot)/mw2
      * yi * pow(1. - 4.*yi, 3./2.);

    // Quark correction.
    if (abs(idi) < 7) partialWidth *= 3.*(1. + alphaS / M_PI );
  // Top width.
  } else if (abs(idMot) == 6) {
    // Expression are identical for top and antitop
    // i is the bottom, j is the W.
    partialWidth = (alpha/4.) * pow3(mMot)/pow2(mw) *
      ( (y0 + yi + 2*yj)*(1. + yi - yj) - 4*yi*sqrt(y0) ) *
      sqrt(kallenFunction(1,yi,yj));

    // AlphaS correction.
    partialWidth *= 1. - 2.72*alphaSPtr->alphaS(pow2(mMot)) / M_PI;

    // CKM Matrix
    partialWidth *= pow2(vCKM[make_pair(abs(idMot), abs(idi))]);

    // Check if width dropped below zero.
    if (partialWidth < 0) return 0;
  } else {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__, ": attempted to compute "
                      "partial width for non-resonant state");
    return 0;
  }

  // Return.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Computed partial width for "
       << idMot << " -> (" << idi << ", " << idj << ") mMot = " << mMot
       << " Gamma = " << partialWidth;
    printOut(__METHOD_NAME__, ss.str());
  }
  return partialWidth;

}

//--------------------------------------------------------------------------

// Sum over partial widths.

double AmpCalculator::getTotalWidth(int idMot, double mMot, int polMot) {
  double totalWidth = 0;

  // Top only decays to b W.
  if (abs(idMot) == 6) {
    totalWidth += getPartialWidth(6, 5, 24, mMot, polMot);
  // Z decays to all fermion pairs except top.
  } else if (abs(idMot) == 23) {
    // Z -> quarks.
    for (int i = 1; i < 6; i++)
      totalWidth += getPartialWidth(23, i, i, mMot, polMot);
    // Z - > leptons.
    for (int i = 11; i < 17; i++)
      totalWidth += getPartialWidth(23, i, i, mMot, polMot);
  // W decays to fermion doublets.
  } else if (abs(idMot) == 24) {
    // The vector boson width is symmetric in xi and xj
    // So there is no difference between 24 and -24.
    totalWidth += getPartialWidth(24, 1, 2, mMot, polMot);
    totalWidth += getPartialWidth(24, 1, 4, mMot, polMot);
    totalWidth += getPartialWidth(24, 3, 2, mMot, polMot);
    totalWidth += getPartialWidth(24, 3, 4, mMot, polMot);
    totalWidth += getPartialWidth(24, 5, 2, mMot, polMot);
    totalWidth += getPartialWidth(24, 5, 4, mMot, polMot);
    // W -> leptons.
    for (int i = 11; i < 17; i += 2)
      totalWidth += getPartialWidth(24, i, i + 1, mMot, polMot);
  // Higgs decays to all fermion pairs.
  // Some are (almost) zero due to their small mass.
  } else if (abs(idMot) == 25) {
    // h-> quarks except top.
    for (int i = 1; i < 6; i++)
      totalWidth += getPartialWidth(25, i, i, mMot, 0);
    // h->leptons.
    for (int i=11; i<17; i++)
      totalWidth += getPartialWidth(25, i, i, mMot, 0);
  } else {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__, ": attempted to compute "
                      "total width for non-resonant state.");
    return 0;
  }

  // Return.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Computed total width for " << idMot <<
      " m = " << mMot << " width = " << totalWidth;
    printOut(__METHOD_NAME__,ss.str());
  }
  return totalWidth;

}


//--------------------------------------------------------------------------

// Breit-Wigner.

double AmpCalculator::getBreitWigner(int id, double m, int pol) {
  id = abs(id);
  double width(getTotalWidth(id, m, pol)), m0(dataPtr->mass(abs(id)));
  return m0*width/(pow2(pow2(m) - pow2(m0)) + pow2(m0)*pow2(width));
}

//--------------------------------------------------------------------------

// Breit-Wigner overestimate.

double AmpCalculator::getBreitWignerOverestimate(int id, double m, int pol) {
  id = abs(id);
  double m0(dataPtr->mass(id, pol)), m02(pow2(m0)),
    width0(dataPtr->width(id, pol)), m2(pow2(m));
  vector<double> cBWNow = cBW[id];
  double BWover = cBWNow[0]*width0*m0/( pow2(m2 - m02)
    + pow2(cBWNow[1])*m02*pow2(width0) );
  BWover += m2/m02 > cBWNow[3] ? cBWNow[2]*m0/pow(m2 - m02, 3./2.) : 0;
  return BWover;
}

//--------------------------------------------------------------------------

// Breit-Wigner Generator.

// Function to generate masses for newly created particles.
// Returns on-shell masses for other particles.

double AmpCalculator::sampleMass(int id, int pol) {

  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Generating mass for a " << id;
    printOut(__METHOD_NAME__, ss.str());}
  id = abs(id);

  // Check if id is a resonance.
  if (dataPtr->isRes(id) && bwMatchMode < 3) {
    double m0(dataPtr->mass(abs(id))), m02(pow2(m0)),
      width0(dataPtr->width(abs(id), pol));

    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "On-shell mass is " << m0;
      printOut(__METHOD_NAME__, ss.str());}

    // Get parameters for the overestimates.
    if (cBW.find(id) == cBW.end()) {
      infoPtr->errorMsg("Warning in " + __METHOD_NAME__,
                        ": no overestimate for resonance available.");
      return m0;}

    vector<double> cBWNow = cBW[id];
    // Compute the normalizations of the two components
    // of the Breit-Wigner overestimates.
    double nBWNow = cBWNow[0]/cBWNow[1]
      * (M_PI/2 + atan(m0/cBWNow[1]/width0));
    double npNow = 2*cBWNow[2]/sqrt(cBWNow[3] - 1);
    double m2(0), pAccept(0);
    do {
      // Select one of the probs.
      if (rndmPtr->flat() < nBWNow/(nBWNow + npNow))
        m2 = m02 + cBWNow[1]*m0*width0*tan( cBWNow[1]/cBWNow[0] *
          rndmPtr->flat() * nBWNow - atan(m0/cBWNow[1]/width0));
      else
        m2 = m02*( pow2(2*cBWNow[2]*sqrt(cBWNow[3] - 1) /
            (2*cBWNow[2] - rndmPtr->flat()*npNow*sqrt(cBWNow[3] - 1))) + 1);

      // Do rejection sampling.
      double BWover = getBreitWignerOverestimate(id, sqrt(m2), pol);
      double BWreal = getBreitWigner(id, sqrt(m2), pol);
      pAccept = BWreal/BWover;

      // Check if the overestimate was right.
      if (pAccept > 1) {
        stringstream ss;
        ss << ": Breit-Wigner overestimate failed with "
           << "id = " << id << " m2 = " << m2 << " p = " << BWreal
           << " " << BWover;
        infoPtr->errorMsg("Warning in " + __METHOD_NAME__, ss.str());
      }
    } while(rndmPtr->flat() > pAccept);

    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Returning sampled mass m2 = " << m2;
      printOut(__METHOD_NAME__, ss.str());
    }
    return sqrt(m2);

  // Otherwise just return the pole mass.
  } else {
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Returning on-shell mass m = " << dataPtr->mass(id);
      printOut(__METHOD_NAME__, ss.str());
    }
    return dataPtr->mass(id);
  }

}

//--------------------------------------------------------------------------

// Bosonic interference weight calculator.

void AmpCalculator::applyBosonInterferenceFactor(Event &event, int XYEv,
  Vec4 pi, Vec4 pj, int idi, int idj, int poli, int polj) {

  if (verbose >= DEBUG) {
    event.list();
    stringstream ss;
    ss << "Computing interference factor for "
       << XYEv << " splitting to " << idi << ", "<< idj;
    printOut(__METHOD_NAME__, ss.str());
  }
  int iSys = partonSystemsPtr->getSystemOf(XYEv);
  double facNumerator(0), facDenominator(0);

  // The interfering bosons are X and Y. Get momentum and polarization.
  Vec4 pXY = pi + pj;
  int polXY = event[XYEv].pol();

  // For longitudinal bosons, X = Z, Y = Higgs.
  // For transverse bosons, X = Z, Y = Gamma.
  int idX, idY;
  double mX, mY;
  double widthX, widthY;
  if (polXY == 0) {
    idX = 23;
    idY = 25;
    widthX = getTotalWidth(23, (pi + pj).mCalc(), polXY);
    widthY = getTotalWidth(25, (pi + pj).mCalc(), polXY);
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "splitter is longitudinal");
  } else {
    idX = 23;
    idY = 22;
    widthX = getTotalWidth(23, (pi + pj).mCalc(), polXY);
    widthY = 0;
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "splitter is transverse");
  }

  // Interfering boson masses.
  mX = dataPtr->mass(idX);
  mY = dataPtr->mass(idY);

  // Compute splitting amplitudes X/Y -> idi, idj.
  complex MXsplit = branchAmpFSR(pi, pj, idX, idi, idj,
    mX, widthX, polXY, poli, polj);
  complex MYsplit = branchAmpFSR(pi, pj, idY, idi, idj,
    mY, widthY, polXY, poli, polj);
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Split amplitudes");
    stringstream ss;
    ss << "M((" << idX << ", " << polXY << ") -> " << idi << ", "
       << idj << ") = " << MXsplit;
    printOut(__METHOD_NAME__, ss.str());
    ss.str("");
    ss << "M((" << idY << ", " << polXY << ") -> "
       << idi << ", " << idj << ") = " << MYsplit;
    printOut(__METHOD_NAME__, ss.str());
  }

  // Now find all emission amplitudes for the emission of X/Y.
  int sysSize = partonSystemsPtr->sizeAll(iSys);

  // Find all particles that may be clustered with the X/Y boson.
  int nClusterings(0);
  for (int i = 0; i < sysSize; i++) {
    int iCluWithEv = partonSystemsPtr->getAll(iSys, i);
    // Don't cluster with yourself.
    if (iCluWithEv != XYEv) {
      int idCluWith  = event[iCluWithEv].id();
      int polCluWith = event[iCluWithEv].pol();
      Vec4 pCluWith  = event[iCluWithEv].p();
      if (verbose >= DEBUG) {
        stringstream ss;
        ss << "Considering clustering with "
           << iCluWithEv << " which is a " << idCluWith;
        printOut(__METHOD_NAME__, ss.str());
      }

      // Get all clusterings for each particle in the system.
      // Compare with a Z because it clusters with the most things.
      // Is the clustering candidate in the final state?
      if (event[iCluWithEv].isFinal()) {
        auto it = cluMapFinal->find(make_pair(idCluWith, 23));
        if (it != cluMapFinal->end()) {
          vector<pair<int, int> > cluVecZ = it->second;
          for (int j = 0; j < (int)cluVecZ.size(); j++) {
            int idCluTo(cluVecZ[j].first), polCluTo(cluVecZ[j].second);
            double mCluTo = dataPtr->mass(idCluTo);

            // Compute emission amplitudes.
            complex MXemit = branchAmpFSR(pCluWith, pXY, idCluTo, idCluWith,
              idX, mCluTo, 0, polCluTo, polCluWith, polXY);
            complex MYemit = branchAmpFSR(pCluWith, pXY, idCluTo, idCluWith,
              idY, mCluTo, 0, polCluTo, polCluWith, polXY);

            // Don't add to weight if emit amplitudes are zero.
            if (abs(MXemit) == 0. && abs(MYemit) == 0.) continue;

            // Contributions to the numerator and denominator.
            facNumerator += norm(MXemit*MXsplit + MYemit*MYsplit);
            facDenominator += norm(MXemit*MXsplit) + norm(MYemit*MYsplit);

            nClusterings++;

            if (verbose >= DEBUG) {
              printOut(__METHOD_NAME__, "Emission amplitudes");
              stringstream ss;
              ss << "M((" << idCluTo << ", " << polCluTo << ") -> "
                 << idCluWith << ", " << idX << ") = " << MXemit;
              printOut(__METHOD_NAME__, ss.str());
              ss.str("");
              ss << "M((" << idCluTo << ", " << polCluTo << ") -> "
                 << idCluWith << ", " << idY << ") = " << MYemit;
              printOut(__METHOD_NAME__, ss.str());
            }
          }
        }
      // Or is it in the initial state?
      } else {
        auto it = cluMapInitial->find(make_pair(idCluWith, 23));
        if (it != cluMapInitial->end()) {
          vector<pair<int, int> > cluVecZ = it->second;
          for (int j = 0; j < (int)cluVecZ.size(); j++) {
            int idCluTo(cluVecZ[j].first), polCluTo(cluVecZ[j].second);
            // Always treat initial state as massless.
            double mCluTo = 0;

            // Compute emission amplitudes
            complex MXemit = branchAmpISR(pCluWith, pXY, idCluTo, idCluWith,
              idX, mCluTo, polCluTo, polCluWith, polXY);
            complex MYemit = branchAmpISR(pCluWith, pXY, idCluTo, idCluWith,
              idY, mCluTo, polCluTo, polCluWith, polXY);

            // Don't add to weight if emit amplitudes are zero.
            if (abs(MXemit) == 0. && abs(MYemit) == 0.) continue;

            // Contributions to the numerator and denominator.
            facNumerator += norm(MXemit*MXsplit + MYemit*MYsplit);
            facDenominator += norm(MXemit*MXsplit) + norm(MYemit*MYsplit);

            nClusterings++;

            if (verbose >= DEBUG) {
              printOut(__METHOD_NAME__, "Emission amplitudes");
              stringstream ss;
              ss << "M((" << idCluTo << ", " << polCluTo << ") -> "
                 << idCluWith << ", " << idX << ") = " << MXemit;
              printOut(__METHOD_NAME__, ss.str());
              ss.str("");
              ss << "M((" << idCluTo << ", " << polCluTo << ") -> "
                 << idCluWith << ", " << idY << ") = " << MYemit;
              printOut(__METHOD_NAME__, ss.str());
            }
          }
        }
      }
    }
  }

  // Don't do anything if there were no clusterings.
  if (nClusterings == 0) return;

  // Protect against nan weights.
  if (facDenominator == 0.) return;

  // Get total weight.
  double weight = facNumerator/facDenominator;
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Interference weight " << weight;
    printOut(__METHOD_NAME__, ss.str());
  }
  // Let weightContainer know that Vincia modified the (nominal) weight.
  infoPtr->weightContainerPtr->weightsShowerPtr->reweightValueByIndex( 0,
    weight);

}

//--------------------------------------------------------------------------

// Polarise a decay.

bool AmpCalculator::polarise(vector<Particle> &state) {
  // TODO: Some check for state[0].
  if (!isInit) return false;
  if (state.size() != 3) {
    if (verbose >= REPORT) infoPtr->errorMsg("Error in " + __METHOD_NAME__ +
      ": tried to polarise invalid resonance decay.");
    return false;
  }
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // This must be a resonance decay: require one incoming particle.
  if (state[0].isFinal() || !state[1].isFinal()) return false;
  // Incoming particle must have a helicity (unless it is a Higgs).
  if (state[0].id() == 25) state[0].pol(0);
  else if (state[0].pol() == 9) return false;
  // Also, there must be a direction with respect to which to define hel.
  else if (state[0].pAbs() < NANO) return false;

  // Get the static weight to regularize the kernels
  double width = dataPtr->width(state[0].id(), state[0].pol());

  // Branching kernels.
  bool flipped = false;
  vector<AntWrapper> brKer;

  // Check if we know this decay.
  auto cluIt = cluMapFinal->find(make_pair(state[1].id(), state[2].id()));
  // Didn't find it?
  if (cluIt == cluMapFinal->end()) {
    // Status[1] and status[2] may be in the wrong order.
    // Check flipped case
    cluIt = cluMapFinal->find(make_pair(state[2].id(), state[1].id()));
    // Didn't find it again? Give up.
    if (cluIt == cluMapFinal->end()) return false;
    else {
      // Compute brKer with flipped arguments.
      brKer = branchKernelFF(state[2].p(), state[1].p(), state[0].id(),
        state[2].id(), state[1].id(), state[0].mCalc(), width, state[0].pol());
      flipped = true;
    }
  // Compute brKer.
  } else
    brKer = branchKernelFF(state[1].p(), state[2].p(), state[0].id(),
      state[1].id(), state[2].id(), state[0].mCalc(), width, state[0].pol());

  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "Relative final-state polarization weights");
  map<double,pair<int,int> > brKerCumulative;
  double brKerSum = 0;
  for (int iPol = 0; iPol < (int)brKer.size(); iPol++) {
    brKerSum += brKer[iPol].val;
    brKerCumulative.insert({brKerSum, make_pair(brKer[iPol].poli,
      brKer[iPol].polj)});
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "pols = (";
      if (flipped) ss << brKer[iPol].polj << ", " << brKer[iPol].poli;
      else         ss << brKer[iPol].poli << ", " << brKer[iPol].polj;
      ss << ") weight = " << brKer[iPol].val;
      printOut(__METHOD_NAME__, ss.str());
    }
  }

  // Select a helicity state.
  double brKerPolSelect = rndmPtr->flat() * brKerSum;
  auto polIterator = brKerCumulative.upper_bound(brKerPolSelect);
  if (polIterator == brKerCumulative.end()) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__,
      ": logic error cumulative sum < aHelSum");
    return false;
  }
  if (flipped) {
    state[2].pol((polIterator->second).first);
    state[1].pol((polIterator->second).second);
  } else {
    state[1].pol((polIterator->second).first);
    state[2].pol((polIterator->second).second);
  }
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return true;

}


//==========================================================================

//  Base class for an electroweak antenna.

//--------------------------------------------------------------------------

// Update a parton system.

void EWAntenna::updatePartonSystems(Event&) {
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Updating system " << iSys;
    printOut(__METHOD_NAME__, ss.str());
    printOut(__METHOD_NAME__, "Parton systems before update: ");
    if (partonSystemsPtr != nullptr) partonSystemsPtr->list();
  }

  if (iSys >= 0 && partonSystemsPtr != nullptr &&
    iSys < partonSystemsPtr->sizeSys()) {
    int iAOld(0), iBOld(0);
    if (isInitial() && partonSystemsPtr->hasInAB(iSys)) {
      iAOld = partonSystemsPtr->getInA(iSys);
      iBOld = partonSystemsPtr->getInB(iSys);
    }
    // Replace old IDs.
    for(auto it = iReplace.begin(); it!= iReplace.end() ; ++it) {
      int iOld(it->first), iNew(it->second);
      if (iAOld == iOld)  partonSystemsPtr->setInA(iSys, iNew);
      else if (iBOld == iOld) partonSystemsPtr->setInB(iSys, iNew);
      partonSystemsPtr->replace(iSys, iOld, iNew);
    }
    // Add new.
    partonSystemsPtr->addOut(iSys, jNew);
    // Save sHat if we set it.
    if (shat > 0.) partonSystemsPtr->setSHat(iSys, shat);
  }
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Parton systems after update: ");
    partonSystemsPtr->list();
  }

}

//--------------------------------------------------------------------------

// Select a channel.

bool EWAntenna::selectChannel(int idx, const double& cSum, const
  map<double, int>& cSumSoFar, int& idi, int& idj, double& mi2, double& mj2) {
  auto it = cSumSoFar.upper_bound(cSum * rndmPtr->flat());
  if (it == cSumSoFar.end()) {
    stringstream ss;
    ss << ": logic error - c"
       << idx << "SumSoFar < c" << idx << "Sum.";
    infoPtr->errorMsg("Error in " +__METHOD_NAME__, ss.str());
    return false;
  }
  brTrial = &brVec[it->second];
  idi = brTrial->idi; idj = brTrial->idj;
  mi2 = pow2(ampCalcPtr->dataPtr->mass(idi));
  mj2 = pow2(ampCalcPtr->dataPtr->mass(idj));
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Selected channel is "
       << idMot << " -> (" << idi << ", " << idj << ")";
    printOut(__METHOD_NAME__, ss.str());
  }
  return true;
}

//==========================================================================

// Final-final electroweak antenna.

//--------------------------------------------------------------------------

// Initialize.

bool EWAntennaFF::init(Event &event, int iMotIn, int iRecIn, int iSysIn,
  vector<EWBranching> &branchings, Settings* settingsPtr) {

  // Settings.
  doBosonInterference     = settingsPtr->flag("Vincia:doBosonicInterference");
  kMapFinal               = settingsPtr->mode("Vincia:kineMapEWFinal");
  vetoResonanceProduction = settingsPtr->flag("Vincia:BWstrongOrdering");

  // Initialize variables.
  iMot   = iMotIn;
  iRec   = iRecIn;
  idMot  = event[iMot].id();
  idRec  = event[iRec].id();
  polMot = event[iMot].pol();
  pMot   = event[iMot].p();
  pRec   = event[iRec].p();
  sAnt   = 2*pMot*pRec;
  mAnt2  = (pMot + pRec).m2Calc();

  // The shower always uses the on-shell mass of mI.
  // mRec only acts as recoiler, so we just use the kinematic mass.
  mMot = ampCalcPtr->dataPtr->mass(idMot);
  mRec = pRec.mCalc();
  mMot2 = pow2(mMot);
  mRec2 = pow2(mRec);

  // This function is part of phase space,
  // It is computed with the kinematic masses.
  double kallen = kallenFunction((pMot + pRec).m2Calc(), pMot.m2Calc(),
    pRec.m2Calc());
  if (kallen < 0.) return false;
  sqrtKallen = sqrt(kallen);
  hasTrial = false;

  // Store the system and branchings.
  iSys = iSysIn;
  brVec = branchings;

  // Find coefficients for overestimates.
  c0Sum = c1Sum = c2Sum = c3Sum = 0;
  for (int i = 0; i < (int)brVec.size(); i++) {
    if (brVec[i].c0 > 0.) {
      c0Sum += brVec[i].c0; c0SumSoFar.insert({c0Sum, i});}
    if (brVec[i].c1 > 0.) {
      c1Sum += brVec[i].c1; c1SumSoFar.insert({c1Sum, i});}
    if (brVec[i].c2 > 0.) {
      c2Sum += brVec[i].c2; c2SumSoFar.insert({c2Sum, i});}
    if (brVec[i].c3 > 0.) {
      c3Sum += brVec[i].c3; c3SumSoFar.insert({c3Sum, i});}
  }
  return true;

}

//--------------------------------------------------------------------------

// Generate a trial.

double EWAntennaFF::generateTrial(double q2Start, double q2End,
  double alphaIn) {
  if (infoPtr->getAbortPartonLevel()) return 0.;
  if (hasTrial) {
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Returning saved trial for " << iMot << " = "<< q2Trial;
      printOut(__METHOD_NAME__, ss.str());}
    return q2Trial;
  }
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "--- Generating trial scale for " << iMot << " ---";
    printOut(__METHOD_NAME__, ss.str());
  }

  // Auxiliary phase space variable ranges.
  // All branching are generated in one go.
  // This implementation overestimates z ranges in massless approx.
  alpha = alphaIn;
  double q2StartLocal(min(q2Start, mAnt2)), zTrial(0), zMin(0), zMax(0),
    zFrac(q2End/(mAnt2 - mRec2));
  // Antenna mass is too small for radiation.
  if (1. - 4*zFrac < 0) return 0;
  // Expand square root to prevent numerical issues.
  else if (zFrac < 1e-8) {zMin = zFrac; zMax = 1. - zFrac;}
  else {
    zMin = 0.5*(1. - sqrt(1. - 4*zFrac));
    zMax = 0.5*(1. + sqrt(1. - 4*zFrac));
  }

  // Zeta integrals and trial generation weights.
  double Iz0(zMax - zMin), Iz1(log(zMax/zMin)), Iz2(log(zMax/zMin)),
    Iz3(0.5*(pow2(zMax) - pow2(zMin)));
  double w0 = alpha*Iz0*c0Sum*mAnt2/sqrtKallen/4./M_PI;
  double w1 = alpha*Iz1*c1Sum*mAnt2/sqrtKallen/4./M_PI;
  double w2 = alpha*Iz2*c2Sum*mAnt2/sqrtKallen/4./M_PI;
  double w3 = alpha*Iz3*c3Sum*mMot2*mAnt2/sqrtKallen/4./M_PI;
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "q2Start " << q2StartLocal << " q2End " << q2End;
    printOut(__METHOD_NAME__, ss.str());
    ss.str("");
    ss << "zMin = " << zMin << " zMax = " << zMax;
    printOut(__METHOD_NAME__, ss.str());
    ss.str("");
    ss << "Zeta integrals " << Iz0 << " " << Iz1 << " " << Iz2;
    printOut(__METHOD_NAME__, ss.str());
    ss.str("");
    ss << "Weights " << w0 << " " << w1 << " " << w2 << " " << w3;
    printOut(__METHOD_NAME__, ss.str());
  }

  // Find the highest new trial scale and generate scale from c0.
  int idi, idj;
  double mi2, mj2;
  q2Trial = 0;
  if (c0Sum > NANO) {
    double q2_0 = q2StartLocal*pow(rndmPtr->flat(), 1./w0);
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Generating q2Trial from c0: " << q2_0;
      printOut(__METHOD_NAME__, ss.str());
    }

    // Compute the trial invariants. Use the kinematic mass of pMot here.
    if (q2_0 > q2Trial) {
      zTrial = zMin + (zMax - zMin)*rndmPtr->flat();
      if (!selectChannel(0, c0Sum, c0SumSoFar, idi, idj, mi2, mj2)) return 0;
      sijTrial = q2_0/zTrial - mi2 - mj2 + mMot2;
      sjkTrial = zTrial*mAnt2 - mj2;
      q2Trial  = q2_0;
    }
  }

  // Generate scale from c1.
  if (c1Sum > NANO) {
    double q2_1 = q2StartLocal*pow(rndmPtr->flat(), 1./w1);
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Generating q2Trial from c1: " << q2_1;
      printOut(__METHOD_NAME__, ss.str());
    }

    // Compute the trial invariants. Use the kinematic mass of pMot here.
    if (q2_1 > q2Trial) {
      zTrial = zMin * pow(zMax/zMin, rndmPtr->flat());
      if (!selectChannel(1, c1Sum, c1SumSoFar, idi, idj, mi2, mj2)) return 0;
      sijTrial = q2_1/((1-zTrial) - mRec2/mAnt2) - mi2 - mj2 + mMot2;
      sjkTrial = (1 - zTrial)*mAnt2 - mj2 - mRec2;
      q2Trial = q2_1;
    }
  }

  // Generate scale from c2.
  if (c2Sum > NANO) {
    // Local veto is required for this trial.
    double q2_2 = q2StartLocal;
    double sijNow = 0, sjkNow = 0, zNow = 0;
    double pAccept;
    do {
      q2_2 = q2_2*pow(rndmPtr->flat(), 1./w2);
      if (q2_2 < q2End) break;
      if (verbose >= DEBUG) {
        stringstream ss;
        ss << "Generating q2Trial from c2: " << q2_2;
        printOut(__METHOD_NAME__, ss.str());
      }

      // Compute the trial invariants. Use the kinematic mass of pMot here.
      zNow = zMin * pow(zMax/zMin, rndmPtr->flat());
      if (!selectChannel(2, c2Sum, c2SumSoFar, idi, idj, mi2, mj2)) return 0;
      sijNow = q2_2/zNow - mi2 - mj2 + mMot2;
      sjkNow = zNow*mAnt2 - mj2;

      // Local veto probability to fix the trial distribution.
      // Note that pAccept may be > 1 if sijNow < 0.
      // This is not an issue because these trials get vetoed in acceptTrial.
      pAccept = zNow/(zNow + sijNow/mAnt2);
    } while(rndmPtr->flat() > pAccept && q2_2 > q2Trial);

    // Save the trial.
    if (q2_2 > q2Trial) {
      q2Trial  = q2_2;
      zTrial   = zNow;
      sijTrial = sijNow;
      sjkTrial = sjkNow;
    }
  }

  // Generate scale from c3.
  if (c3Sum > NANO) {
    double q2_3 = w3*q2StartLocal/(w3 - q2StartLocal*log(rndmPtr->flat()));
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Generating q2Trial from c3: " << q2_3;
      printOut(__METHOD_NAME__, ss.str());
    }

    // Compute the trial invariants. Use the kinematic mass of pMot here.
    if (q2_3 > q2Trial) {
      zTrial = sqrt(pow2(zMin) + rndmPtr->flat()*(pow2(zMax) - pow2(zMin)));
      if (!selectChannel(3, c3Sum, c3SumSoFar, idi, idj, mi2, mj2)) return 0;
      sijTrial = q2_3/zTrial - mi2 - mj2 + mMot2;
      sjkTrial = zTrial*mAnt2 - mj2;
      q2Trial  = q2_3;
    }
  }

  // Return trial.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Generated q2Trial = "
       << q2Trial <<" zTrial = " << zTrial << " sijTrial = " << sijTrial
       << " sjkTrial = "<< sjkTrial;
    printOut(__METHOD_NAME__, ss.str());
  }
  if (isnan(zTrial)) q2Trial = zTrial = sijTrial = sjkTrial = 0;
  return q2Trial;

}

//--------------------------------------------------------------------------

// Accept trial.

bool EWAntennaFF::acceptTrial(Event &event) {

  // Mark trial as used.
  hasTrial = false;

  // Get some variables.
  int idi(brTrial->idi), idj(brTrial->idj);
  double mi(ampCalcPtr->dataPtr->mass(idi)),
    mj(ampCalcPtr->dataPtr->mass(idj)), mi2(pow2(mi)), mj2(pow2(mj)),
    sij(sijTrial), sjk(sjkTrial), sik(mAnt2 - sij - sjk - mi2 - mj2 - mRec2);
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Branching is ("
       << idMot << ", " << polMot << ") ->" << idi << ", " << idj;
    printOut(__METHOD_NAME__, ss.str());
    ss.str("");
    ss << "Invariants sij = " << sij << ", sjk = " << sjk << ", sik = " << sik;
    printOut(__METHOD_NAME__, ss.str());}

  // Check on-shell phase space.
  if (sij < 0 || sjk < 0 || sik < 0 || sqrt(mAnt2) < mi + mj + mRec ||
    sij*sjk*sik - pow2(sij)*mRec2 - pow2(sik)*mj2 - pow2(sjk)*mi2
    + 4*mi2*mj2*mRec2 < 0) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Outside phase space: On-shell phase space");
    return false;
  }

  // Make sure any new qqbar pair has at least the invariant mass
  // of the lightest meson.
  // sijMin is 0 if these are not quarks.
  double sijMin = vinComPtr->mHadMin(idi, idj);
  if (sij < sijMin) return false;

  // Construct veto probability, alpha ratio.
  double pAccept =  alphaPtr->alphaEM(q2Trial) / alpha;

  // Trial kernel - uses kinematic mass of pMot.
  double Q2 = sij + mi2 + mj2 - mMot2;
  double xi = (sij + sik + mi2)/mAnt2;
  double xj = (sij + sjk + mj2)/mAnt2;
  double aTrial = brTrial->c0/Q2 + brTrial->c1/Q2/xi + brTrial->c2/Q2/xj
    + brTrial->c3*mMot2/Q2/Q2;

  // Physical kernels per final state spin config - width = 0.
  vector<AntWrapper> aPhys = ampCalcPtr->antFuncFF(Q2, 0, xi, xj,
            idMot, idi, idj, mMot, mi, mj, polMot);

  // Sums and cumulants for spin states.
  double aPhysSum = 0;
  map<double,int> aPhysCumulative;
  for (int i = 0; i < (int)aPhys.size(); i++) {
    double aNow = aPhys[i].val;
    if (isnan(aNow) || isinf(aNow)) {
      string msg(": amplitude is " + string(isnan(aNow) ? "NAN" : "infinite"));
      infoPtr->errorMsg("Error in " + __METHOD_NAME__, msg + ".");
      infoPtr->setAbortPartonLevel(true);
      return false;
    }
    if (aNow > 0.) {aPhysSum += aNow; aPhysCumulative.insert({aPhysSum, i});}
  }

  // Antenna ratio.
  pAccept *= aPhysSum/aTrial;

  // Check if veto probability is ok.
  if (pAccept > 1) {
    stringstream ss;
    ss << ": incorrect overestimate ("
       << idMot << ", " << polMot << ") -> " << idi << ", " << idj << ": "
       << aPhysSum/aTrial;
    infoPtr->errorMsg("Warning in " + __METHOD_NAME__, ss.str());
  }
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Accepting with probability " << pAccept;
    printOut(__METHOD_NAME__, ss.str());}

  // Accept/Reject step.
  double rAccept  = rndmPtr->flat();
  if ( rAccept > pAccept) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__, "Failed to pass veto.");
    return false;
  }

  // Passed veto - now select a spin state.
  double aSelect = rndmPtr->flat() * aPhysSum;
  auto it = aPhysCumulative.upper_bound(aSelect);
  if (it == aPhysCumulative.end()) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__,
                      ": logic error - cumulative sum < aPhysSum");
    return false;
  }
  poliTrial = aPhys[it->second].poli;
  poljTrial = aPhys[it->second].polj;

  // Sample Breit-Wigner masses for the kinematics.
  // Don't change resonance mass in an emission.
  double miKin  = idi == idMot ? pMot.mCalc()
    : ampCalcPtr->sampleMass(idi,poliTrial);
  double mi2Kin = pow2(miKin);
  double mjKin  = ampCalcPtr->sampleMass(idj, poljTrial);
  double mj2Kin = pow2(mjKin);

  // Recompute invariants to conserve Q2.
  double sijKin = Q2 + mMot2 - mi2Kin - mj2Kin;

  // Check off-shell phase space.
  if (sijKin < 0 || sjk < 0 || sik < 0
    || sqrt(mAnt2) < miKin + mjKin + mRec ||
    sijKin*sjk*sik - pow2(sijKin)*mRec2 - pow2(sik)*mj2Kin
    - pow2(sjk)*mi2Kin + 4*mi2Kin*mj2Kin*mRec2 < 0) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__, "Outside phase space.");
    return false;
  }

  // Check if we veto resonance production with off-shellness larger than q2.
  if (vetoResonanceProduction) {
    double mi2Onshell  = pow2(ampCalcPtr->dataPtr->mass(idi));
    double mj2Onshell  = pow2(ampCalcPtr->dataPtr->mass(idj));
    double q2Offshelli = fabs(mi2Kin - mi2Onshell);
    double q2Offshellj = fabs(mj2Kin - mj2Onshell);
    if (q2Trial < q2Offshelli || q2Trial < q2Offshellj) {
      infoPtr->errorMsg("Warning in " + __METHOD_NAME__,
                        ": final-state resonance too far offshell.");
      return false;
    }
  }

  // Proceed to do the kinematics.
  // Pre- and post-branching momenta, phi, invariants, masses.
  vector<Vec4> pOld{pMot, pRec};
  pNew.clear();
  double phi = rndmPtr->flat()*2*M_PI;
  vector<double> invariants{sAnt, sijKin, sjk};
  vector<double> masses{miKin, mjKin, mRec};

  // Do kinematics and return.
  if (!vinComPtr->map2to3FF(pNew, pOld, kMapFinal, invariants, phi, masses)) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__,"Failed to generate kinematics.");
    return false;
  }
  // Apply bosonic interference factor.
  if (doBosonInterference && (idMot == 22 || idMot == 23 || idMot == 25) &&
      abs(idi) == abs(idj))
    ampCalcPtr->applyBosonInterferenceFactor(event,
      iMot, pNew[0], pNew[1], idi, idj, poliTrial, poljTrial);
  // Branching accepted!
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "Branching accepted!");
  return true;

}

//--------------------------------------------------------------------------

// Update an event.

void EWAntennaFF::updateEvent(Event &event) {
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "Updating event");

  // Clear information for replacing later in partonSystems.
  iReplace.clear();
  shat = 0.;
  Particle parti, partj, partk;

  // Fetch ids and kinematic masses.
  int idi(brTrial->idi), idj(brTrial->idj);
  double mi(pNew[0].mCalc()), mj(pNew[1].mCalc());

  // If new particles are qqbar pair, we need new color indices.
  if (idi > 0 && idi < 7 && idj < 0 && idj > -7) {
    int colTag = 10*(event.nextColTag()/10 + 1) + 1 + rndmPtr->flat()*10;
    parti = Particle(idi, 51, iMot, 0, 0, 0, colTag, 0, pNew[0], mi, 0,
      poliTrial);
    partj = Particle(idj, 51, iMot, 0, 0, 0, 0, colTag, pNew[1], mj, 0,
      poljTrial);
  // Otherwise carry I color indices to i.
  } else {
    int col(event[iMot].col()), acol(event[iMot].acol());
    parti = Particle(idi, 51, iMot, 0, 0, 0, col, acol, pNew[0], mi, 0,
      poliTrial);
    partj = Particle(idj, 51, iMot, 0, 0, 0, 0, 0, pNew[1], mj, 0,
      poljTrial);
  }
  partk = event[iRec];
  partk.p(pNew[2]);
  partk.statusCode(52);
  partk.mothers(iRec, iRec);
  int iEv(event.append(parti)), jEv(event.append(partj)),
    kEv(event.append(partk));

  // Adjust old particles.
  event[iMot].daughters(iEv, jEv);
  event[iMot].statusNeg();
  event[iRec].daughters(kEv, kEv);
  event[iRec].statusNeg();

  // Save information for parton systems.
  jNew = jEv;
  iReplace[iMot] = iEv;
  iReplace[iRec] = kEv;
  event.restorePtrs();
}

//==========================================================================

// Final-final electroweak resonance antenna.

//--------------------------------------------------------------------------

// Initialize.

bool EWAntennaFFres::init(Event &event, int iMotIn, int iRecIn,  int iSysIn,
  vector<EWBranching> &branchings, Settings* settingsPtr) {

  // Call the FF antenna init function.
  bool succeed = EWAntennaFF::init(event, iMotIn, iRecIn,
    iSysIn, branchings, settingsPtr);

  // Resonance-related settings.
  bwMatchMode = settingsPtr->mode("Vincia:bwMatchingMode");
  q2EW = pow2(settingsPtr->parm("Vincia:EWScale"));
  int resDecScaleChoice = settingsPtr->mode("Vincia:resDecScaleChoice");

  // Check if this resonance has a recoiler.
  if (iRecIn == 0) doDecayOnly = true;

  // Compute offshellness = minimum decay scale.
  // (In principle, choice = 0 should be hardcoded = width, but not
  // sure that makes sense here so defaulting to scale of order width
  // in that case.)
  if (resDecScaleChoice == 2) q2Dec = fabs( pMot.m2Calc() - mMot2 );
  else q2Dec = pow2( pMot.m2Calc() - mMot2 ) / mMot2;

  // Minimum offshellness is NANO to avoid numerical issues.
  // Multiply by 0.999 to make sure Pythia decays get to go first.
  q2Dec = max(0.999*q2Dec, NANO);
  return succeed;

}

//--------------------------------------------------------------------------

// Generate a trial.

double EWAntennaFFres::generateTrial(double q2Start, double q2End,
  double alphaIn) {

  // Reset the current trial.
  q2Trial = 0;
  trialIsResDecay = false;

  // If the offshellness is above the current shower scale, decay immediately.
  if (q2Dec > q2Start) {
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Decaying resonance with"
         << "q2Dec = " << q2Dec << " > q2Start = " << q2Start;
      printOut(__METHOD_NAME__, ss.str());}
    q2Trial = q2Start;
    trialIsResDecay = true;
    return q2Trial;
  }

  // If this is a resonance decay without recoiler or if ewMatchMode
  // == 1, decay immediately.
  if (doDecayOnly || bwMatchMode == 1) {
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Returning q2Dec = " << q2Dec;
      printOut(__METHOD_NAME__, ss.str());}
    q2Trial = q2Dec;
    trialIsResDecay = true;
    return q2Trial;
  }

  // Set the lower bound to q2Dec. Note that for ewMode == 4, q2Dec == 0.
  double q2EndLocal = max(q2End, q2Dec);

  // If applying the suppression factor, add a cutoff.
  if (bwMatchMode == 2) q2EndLocal = max(q2EndLocal, q2EW * 1E-4);

  // Sample a trial scale.
  EWAntennaFF::generateTrial(q2Start, q2EndLocal, alphaIn);

  // Check if the trial scale is above the cutoff.
  if (q2Trial < q2EndLocal) {
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "q2Trial = "
         << q2Trial << " is smaller than q2EndLocal = " << q2EndLocal;
      printOut(__METHOD_NAME__, ss.str());
      ss.str("");
      ss << "Resonance decay at offshellness " << q2Dec;
      printOut(__METHOD_NAME__, ss.str());}

    // Set q2Trial equal to the resonance offshelless.
    q2Trial = q2Dec;
    trialIsResDecay = true;
  }
  return q2Trial;

}

//--------------------------------------------------------------------------

// Accept a trial.

bool EWAntennaFFres::acceptTrial(Event &event) {

  // Check if this is a resonance decay.
  if (trialIsResDecay) {
    // Force the decay. Always return true afterwards.
    if (genForceDecay(event)) return true;
    else {
      infoPtr->errorMsg("Error in " + __METHOD_NAME__,
        ": failed to force resonance decay.");
      infoPtr->setAbortPartonLevel(true);
      return false;
    }
  // If it is a shower branching, we need to veto.
  } else {
    // Shower suppression for matching to the Breit-Wigner distribution.
    if (bwMatchMode == 2) {
      // On-shell masses, off-shellness, and suppression factor.
      double mi2     = pow2(ampCalcPtr->dataPtr->mass(brTrial->idi));
      double mj2     = pow2(ampCalcPtr->dataPtr->mass(brTrial->idj));
      double Q2      = sijTrial + mi2 + mj2 - pMot.m2Calc();
      double pAccept = pow2(Q2)/pow2(fabs(Q2) + q2EW);

      // Veto to add the suppression factor.
      if (rndmPtr->flat() > pAccept) {
        if (verbose >= DEBUG)
          printOut(__METHOD_NAME__, "Failed BW-matching veto.");
        return false;
      }
    }

    // If passed, do regular shower veto.
    return EWAntennaFF::acceptTrial(event);
  }

}

//--------------------------------------------------------------------------

// Update an event.

void EWAntennaFFres::updateEvent(Event &event) {
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "Adding branching to event");

  // Clear information for replacing later in partonSystems.
  iReplace.clear();
  shat = 0.;

  // Fetch ids, masses, trial vectors.
  int idi(brTrial->idi), idj(brTrial->idj);
  double mi(pNew[0].mCalc()), mj(pNew[1].mCalc());
  Vec4 piTrial(pNew[0]), pjTrial(pNew[1]);

  // If this is not a 1->2 decay, we need a temporary entry for resonance just
  // prior to decay with recoiled momentum and with special status code(57).
  int iResNew = iMot;
  if (!trialIsResDecay) {
    Vec4 pRes = piTrial+pjTrial;
    int resCol(event[iMot].col()), resAcol(event[iMot].acol()),
      resPol(event[iMot].pol());
    Particle partRes(idMot, 57, iMot, iMot, 0, 0, resCol, resAcol, pRes,
      pRes.mCalc(), sqrt(q2Trial), resPol);
    iResNew = event.append(partRes);

    // Update the mother particle.
    event[iMot].daughters(iResNew,iResNew);
    event[iMot].statusNeg();

    // Update the recoiler if this was not a 1->2 branching.
    Particle partk;
    partk = event[iRec];
    partk.p(pNew[2]);
    partk.statusCode(52);
    partk.mothers(iRec, iRec);
    int kEv = event.append(partk);
    event[iRec].daughters(kEv, kEv);
    event[iRec].statusNeg();

    // Save information for parton systems.
    iReplace[iRec] = kEv;
  }

  // Add daughters.
  Particle parti, partj;
  // If new particles are qqbar pair, we need new color indices.
  if (idi > 0 && idi < 7 && idj < 0 && idj > -7) {
    int colTag = 10*(event.nextColTag()/10 + 1) + 1 + rndmPtr->flat()*10;
    parti = Particle(idi, 51, iResNew, 0, 0, 0, colTag, 0, piTrial, mi, 0,
      poliTrial);
    partj = Particle(idj, 51, iResNew, 0, 0, 0, 0, colTag, pjTrial, mj, 0,
      poljTrial);
  // Otherwise carry I color indices to i.
  } else {
    int col(event[iMot].col()), acol(event[iMot].acol());
    parti = Particle(idi, 51, iResNew, 0, 0, 0, col, acol, piTrial, mi, 0,
      poliTrial);
    partj = Particle(idj, 51, iResNew, 0, 0, 0, 0, 0, pjTrial, mj, 0,
      poljTrial);
  }
  int iEv(event.append(parti)), jEv(event.append(partj));

  // Update the resonance.
  event[iResNew].daughters(iEv, jEv);
  event[iResNew].statusNeg();

  // Save information for parton systems.
  jNew = jEv;
  iReplace[iMot] = iEv;
  event.restorePtrs();

}

//--------------------------------------------------------------------------

// Generate the kinematics and channel for decays below matching scale.

bool EWAntennaFFres::genForceDecay(Event &event) {
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Performing non-shower resonance decay."
       << " IdI = " << idMot << " q2Dec = " << q2Dec;
    printOut(__METHOD_NAME__, ss.str());}

  // Check if allowed to decay (according to Pythia decay table).
  const ParticleDataEntryPtr motDataPtr =
    infoPtr->particleDataPtr->findParticle(idMot);
  if (!motDataPtr->mayDecay()) return false;

  // Resonance mass to be used in decay, total and partial width.
  double m(pMot.mCalc()), m2(pow2(m)), totWidthOffshell(0);
  map<double, int> partialWidthCumulatives;
  for (int iChannel = 0; iChannel < (int)brVec.size(); iChannel++) {
    // Fetch daughter ids.
    int idi(brVec[iChannel].idi), idj(brVec[iChannel].idj);

    // Check if Pythia thinks this channel should be switched off.
    bool isOn = true;
    for (int i = 0; i < motDataPtr->sizeChannels(); ++i) {
      int multi = motDataPtr->channel(i).multiplicity();
      // EW shower only does 1->2 decays so ignore any others.
      if (multi != 2) continue;
      int nUnmatched = 2;
      for (int j = 0; j < multi; ++j) {
        int idNow =  abs(motDataPtr->channel(i).product(j));
        if (idNow == abs(idi) || idNow == abs(idj)) --nUnmatched;
      }
      if (nUnmatched == 0) {
        // Found mode. Check if it is on.
        if (motDataPtr->channel(i).onMode() <= 0) isOn = false;
        break;
      }
    }

    // Add partial width.
    if (isOn) totWidthOffshell +=
      ampCalcPtr->getPartialWidth(idMot, idi, idj, pMot.mCalc(), polMot);
    else if (verbose >= DEBUG) {
      stringstream ss;
      ss << " Channel "
         << idMot << " -> " << idi << " " << idj
         << " is switched off according to PDT.\n";
      printOut(__METHOD_NAME__, ss.str());
    }

    // Add to cumulative.
    partialWidthCumulatives.insert({totWidthOffshell, iChannel});
  }

  // Check if there is at least a single decay channel.
  if (totWidthOffshell == 0.) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__,
      "No phase space available for decay");
    infoPtr->setAbortPartonLevel(true);
    return false;
  }
  int idi, idj;
  double mi, mi2, mj, mj2, m0, m02;
  double yi, yj, y0;

  // Select a channel with relative prob of its partial width.
  // Subject to which modes are switched on in Pythia.
  do {
    double partialWidthSelect = rndmPtr->flat() * totWidthOffshell;
    auto widthIterator =
      partialWidthCumulatives.upper_bound(partialWidthSelect);
    if (widthIterator == partialWidthCumulatives.end()) {
      infoPtr->errorMsg("Error in " + __METHOD_NAME__,
        ": logic error - cumulative sum < total Width");
      return false;
    }

    // Store selection.
    int iChannel = widthIterator->second;
    brTrial = &brVec.at(iChannel);
    idi = brTrial->idi;
    idj = brTrial->idj;
    mi  = ampCalcPtr->dataPtr->mass(idi);
    mi2 = pow2(mi);
    mj  = ampCalcPtr->dataPtr->mass(idj);
    mj2 = pow2(mj);
    m0  = ampCalcPtr->dataPtr->mass(idMot);
    m02 = pow2(m0);
    yi = mi2/m2;
    yj = mj2/m2;
    y0 = m02/m2;
  } while (kallenFunction(1,yi,yj) < 0 || yi + yj > 1);

  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Selected channel ("
       << brTrial->idMot << " - > " << brTrial->idi << ", " << brTrial->idj
       << ")";
    printOut(__METHOD_NAME__, ss.str());}

  // Next step is to sample angles for the decay in the CM frame.
  // The azimuthal angle is uniformly distributed.
  double phi = 2.0*M_PI*rndmPtr->flat();

  // The polar angle is distributed according to the final-state
  // spin-summed amplitudes.
  // It is not uniform due to the polarization of the resonance.
  // The distribution always looks like r0 + r1 * c + r2 * c^2,
  // where c = cosTheta.
  // First determine these coefficients for this decay.
  double r0, r1, r2;

  // Top decay - amplitudes are identical for t and tbar.
  if (abs(idMot) == 6) {
    r0 =        (1. + yi - yj)*(y0 + yi + 2*yj) - 4*yi*sqrt(y0);
    r1 = polMot*(1. + yi - yj)*(y0 - yi - 2*yj);
    r2 = 0;
  // Higgs decay - no polarization, so uniform distribution.
  } else if (abs(idMot) == 25) {
    r0 = 1;
    r1 = 0;
    r2 = 0;
  // Vector boson decay.
  } else {
    // Get fermion couplings.
    double v = ampCalcPtr->vMap.at(make_pair(abs(idi), abs(idMot)));
    double a = ampCalcPtr->aMap.at(make_pair(abs(idi), abs(idMot)));
    double v2(pow2(v)), a2(pow2(a));

    // Positive transverse polarization.
    if (polMot == 1) {
      r0 = (v2 + a2)*(1 - yi + yj)*(1 + yi - yj) + 4*(v2 - a2)*sqrt(yi*yj);
      r1 = 4*v*a*(1 - yi + yj)*(1 + yi - yj);
      r2 = (v2 + a2)*(1 - yi + yj)*(1 + yi - yj);
    // Negative transverse polarization.
    } else if (polMot == -1) {
      r0 = (v2 + a2)*(1 - yi + yj)*(1 + yi - yj) + 4*(v2 - a2)*sqrt(yi*yj);
      r1 = -4*v*a*(1 - yi + yj)*(1 + yi - yj);
      r2 = (v2 + a2)*(1 - yi + yj)*(1 + yi - yj);
    // Longitudinal polarization.
    } else {
      r0 = (v2 + a2)*(1 - yi - yj) + (v2 - a2)*(2*sqrt(yi)*sqrt(yj));
      r1 = 0;
      r2 = -(v2 + a2)*(1 - yi + yj)*(1 + yi - yj);
    }
  }
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Determining the angular distribution of "
      "the decay");
    stringstream ss;
    ss << "p(cosTheta) = "
       << r0 << " + " << r1 << "*cosTheta + " << r2 << "*cosTheta^2";
    printOut(__METHOD_NAME__, ss.str());}

  // Without too much loss of efficiency,
  // we can overestimate the distribution with a constant function.
  // We find the maximum by considering the function at c=1,0,-1.
  double rOver = 0.;
  if (r0 + r1 + r2 > rOver) {rOver = r0 + r1 + r2;}
  if (r0 > rOver)           {rOver = r0;}
  if (r0 - r1 + r2 > rOver) {rOver = r0 - r1 + r2;}
  if (rOver == 0.) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": failed to set overestimate.");
    infoPtr->setAbortPartonLevel(true);
    return false;
  }

  // Rejection sampling loop.
  double cosTheta(0), pAccept;
  do {
    cosTheta = 2.*rndmPtr->flat() - 1.;
    pAccept  = max(0., (r0 + r1*cosTheta + r2*pow2(cosTheta))/rOver);
    if (isinf(pAccept)) {
      infoPtr->errorMsg("Error in " + __METHOD_NAME__,
        ": pAccept is infinite.");
      infoPtr->setAbortPartonLevel(true);
      return false;
    }
    if (pAccept > 1. || pAccept < 0.) {
      if (verbose >= NORMAL) {
        stringstream ss;
        ss << "Failed to determine overestimate. P = " << pAccept;
        printOut(__METHOD_NAME__, ss.str());}
      return false;
    }
  } while (rndmPtr->flat() > pAccept);

  // Fetch the momenta for this decay channel.
  double theta = acos(cosTheta);
  if (!vinComPtr->map1to2RF(pNew, pMot, mi, mj, theta, phi)) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__,
      ": failed kinematics while phase space is open");
    return false;
  }
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Decay product momenta " << pNew[0] << pNew[1];
    printOut(__METHOD_NAME__, ss.str());}

  // Now select a spin channel - use the polarise function.
  vector<Particle> state;
  state.push_back(Particle(idMot, -22, 0, 0, 0, 0, 0, 0, pMot, pMot.mCalc(), 0,
      polMot));
  state.push_back(Particle(idi, 23, 0, 0, 0, 0, 0, 0, pNew[0], mi));
  state.push_back(Particle(idj, 23, 0, 0, 0, 0, 0, 0, pNew[1], mj));
  ampCalcPtr->polarise(state);
  poliTrial = state[1].pol();
  poljTrial = state[2].pol();
  if (poliTrial == 9 || poljTrial == 9)
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": failed to polarize a decay.");

  // Special case for top decays.
  // Redistribute the W mass using the Breit-Wigner.
  if (ampCalcPtr->dataPtr->isRes(idi) || ampCalcPtr->dataPtr->isRes(idj)) {
    // Keep trying until inside phase space.
    do {
      mi  = ampCalcPtr->sampleMass(idi, poliTrial);
      mi2 = pow2(mi);
      yi  = mi2/pMot.m2Calc();
      mj  = ampCalcPtr->sampleMass(idj, poljTrial);
      mj2 = pow2(mj);
      yj  = mj2/pMot.m2Calc();
    } while (kallenFunction(1, yi, yj) < 0 || yi + yj > 1);

    // Redo the kinematics.
    if (!vinComPtr->map1to2RF(pNew, pMot, mi, mj, theta, phi)) {
      infoPtr->errorMsg("Error in " + __METHOD_NAME__,
        ": failed kinematics while phase space is open.");
      return false;
    }
  }

  // Apply bosonic interference weight.
  if (doBosonInterference && (idMot == 23 || idMot == 25) &&
      abs(idi) == abs(idj)) ampCalcPtr->applyBosonInterferenceFactor(event,
      iMot, pNew[0], pNew[1], idi, idj, poliTrial, poljTrial);
  return true;

}

//==========================================================================

// Initial-initial electroweak antenna.

//--------------------------------------------------------------------------

// Initialize.

bool EWAntennaII::init(Event &event, int iMotIn, int iRecIn, int iSysIn,
  vector<EWBranching> &branchings, Settings* settingsPtr) {

  // Settings.
  doBosonInterference = settingsPtr->flag("Vincia:doBosonicInterference");
  vetoResonanceProduction = settingsPtr->flag("Vincia:BWstrongOrdering");

  // Initialize variables.
  iMot   = iMotIn;
  iRec   = iRecIn;
  idMot  = event[iMot].id();
  idRec  = event[iRec].id();
  polMot = event[iMot].pol();
  pMot   = event[iMot].p();
  pRec   = event[iRec].p();
  sAnt   = 2*pMot*pRec;
  // Masses are set to zero explicitly.
  mMot = mMot2 = mRec = mRec2 = 0;
  // Hadronic invariant mass
  shh = m2(beamAPtr->p(),beamBPtr->p());
  // Hadronic momentum fractions
  xMot = pMot.e()/(0.5*sqrt(shh));
  xRec = pRec.e()/(0.5*sqrt(shh));

  // Sanity check for phase space being open.
  if (fabs(shh-sAnt) < NANO) return false;
  hasTrial = false;
  // Set system.
  iSys = iSysIn;
  // Store branchings.
  brVec = branchings;
  // Find coefficients for overestimates.
  c0Sum = c1Sum = c2Sum = c3Sum = 0;
  // Only use c0 for initial state radiation.
  for (int i = 0; i < (int)brVec.size(); i++)
    if (brVec[i].c0 > 0.) {
      c0Sum += brVec[i].c0;
      c0SumSoFar.insert({c0Sum, i});
  }
  return true;

}

//--------------------------------------------------------------------------

// Generate a trial.

double EWAntennaII::generateTrial(double q2Start, double q2End,
  double alphaIn) {
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  if (infoPtr->getAbortPartonLevel()) return 0;
  if (hasTrial) {
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Returning saved trial for "
         << iMot << " = " << q2Trial;
      printOut(__METHOD_NAME__, ss.str());}
    return q2Trial;
  }

  alpha = alphaIn;
  q2Trial = 0;
  double zTrial = 0;

  // Cutoff value is the same as for FF, but it is not kinematically identical.
  if (q2End > q2Start) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__, "Already below cutoff.");
    return q2Trial;
  }

  // Zeta boundaries.
  double zMin(0), zMax(0);
  if (shh == 0. || sAnt >= shh) {
    if (verbose >=DEBUG) printOut(__METHOD_NAME__,"Phase space is closed.");
    return 0.;
  }

  // Prevent NaN phase space boundaries.
  if (pow2(shh - sAnt) - 4.*q2End*shh < 0) return 0.;
  // Find largest mj2 in branchings.
  // Required due to the asymmetric phase space bounds.
  double mj2Max = 0;
  for (int i = 0; i < (int)brVec.size(); i++) {
    double mj2Test = pow2(ampCalcPtr->dataPtr->mass(brVec[i].idj));
    if (mj2Test > mj2Max) mj2Max = mj2Test;
  }

  // Expand square root to prevent numerical issues.
  if (4.*q2End*shh/pow2(shh - sAnt) < 1E-8) {
    zMin = q2End/(shh-sAnt);
    zMax = 1. - sAnt/shh;
  } else {
    zMin = (shh - sAnt - mj2Max
      - sqrt(pow2(shh - sAnt - mj2Max) - 4.*q2End*shh))/2./shh;
    zMax = (shh - sAnt + sqrt(pow2(shh - sAnt) - 4.*q2End*shh))/2./shh;
  }
  if (zMax==1.0 || zMax==0. || zMin==1.0 || zMin==0.) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": seta limits outside acceptable range.");
    infoPtr->setAbortPartonLevel(true);
    return 0.;
  }

  // Check if there is any phase space available.
  if (zMin > zMax) {
    if (verbose >=DEBUG) printOut(__METHOD_NAME__,"Phase space is closed.");
    return 0.;
  }

  // Zeta integral and trial generation weight; generate a new scale.
  double Iz(log(zMax*(1.-zMin)/zMin/(1.-zMax))), w(alpha*Iz*c0Sum/4./M_PI),
    pAccept;
  q2Trial = q2Start;
  do {
    q2Trial = q2Trial*pow(rndmPtr->flat(), 1./w);

    // Generate z.
    double A = pow(exp(Iz), rndmPtr->flat());
    zTrial = A*zMin / (1. - (1 - A)*zMin);

    // Select channel and get invariants.
    double mi2, mj2; int idi, idj;
    if (!selectChannel(0, c0Sum, c0SumSoFar, idi, idj, mi2, mj2)) return 0;
    sijTrial = (q2Trial + mj2 + zTrial*sAnt)/(1.-zTrial);
    sjkTrial = q2Trial/zTrial + mj2;

    // Note that the accept prob might be negative.
    // In those cases, the point is outside phase space
    // and would be vetoed anyway.
    pAccept = (sjkTrial - mj2)/(sijTrial + sjkTrial - mj2);
  } while(rndmPtr->flat() > pAccept && q2Trial > q2End);

  // Return.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Generating q2Trial from c: " << q2Trial;
    printOut(__METHOD_NAME__, ss.str());
    printOut(__METHOD_NAME__, "end", dashLen);}
  return q2Trial;

}

//--------------------------------------------------------------------------

// Accept a trial.

bool EWAntennaII::acceptTrial(Event &event) {

  // Mark trial as used.
  hasTrial = false;

  // Indices, on-shell mass, phase space invariants.
  int idi(brTrial->idi), idj(brTrial->idj);
  double mj(ampCalcPtr->dataPtr->mass(idj)), mj2(pow2(mj)),
    q2(q2Trial), sij(sijTrial), sjk(sjkTrial), sik(sAnt + sij + sjk - mj2);

  // Check if I is A or B
  bool isIA = pMot.pz() > 0;
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "sAnt = " << sAnt << " sij = " << sij
       << " sjk = " << sjk << " sik = " << sik;
    printOut(__METHOD_NAME__, ss.str());}

  // Check on-shell phase space.
  if (sij < 0 || sjk < 0 || sik < 0 || sik*sij*sjk - pow2(sik)*mj2 < 0) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Outside phase space: Negative invariants");
    return false;
  }

  // Compute post-branching energies of initial state.
  double ei = pMot.e()*sqrt(sik/sAnt * (sik - sij)/(sik - sjk));
  double ek = pRec.e()*sqrt(sik/sAnt * (sik - sjk)/(sik - sij));

  // Get new initial-state momentum fractions.
  double xi = ei/(sqrt(shh)/2.);
  double xk = ek/(sqrt(shh)/2.);
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "New momentum fractions " << xi << " " << xk;
    printOut(__METHOD_NAME__, ss.str());}

  // Check if new energies don't exceed hadronic maxima.
  // First add up all initial state energies.
  double eaUsed(0), ebUsed(0);
  int nSys = partonSystemsPtr->sizeSys();
  for (int i = 0; i < nSys; i++) {
    eaUsed += event[partonSystemsPtr->getInA(i)].e();
    ebUsed += event[partonSystemsPtr->getInB(i)].e();
  }

  // Then adjust for this branching.
  if (isIA) {
    eaUsed +=  ei - pMot.e();
    ebUsed +=  ek - pRec.e();
  } else {
    eaUsed +=  ek - pRec.e();
    ebUsed +=  ei - pMot.e();
  }

  // Set headroom of 2% in accordance with the rest of Vincia.
  if (eaUsed > 0.98*sqrt(shh)/2. || ebUsed > 0.98*sqrt(shh)/2.) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__,"eaUsed or ebUsed too high: veto. ");
    return false;
  }

  // Next, compute accept probability.
  double pAccept = alphaPtr->alphaEM(q2Trial) / alpha;

  // Compute PDF ratios and add it to the accept probability.
  double newPDF, oldPDF;
  if (isIA) {
    newPDF  = max(beamAPtr->xfISR(iSys, idi, xi, q2)/xi, TINYPDFtrial);
    newPDF *= max(beamBPtr->xfISR(iSys, idRec, xk, q2)/xk, TINYPDFtrial);
    oldPDF  = max(beamAPtr->xfISR(iSys, idMot, xMot, q2)/xMot, TINYPDFtrial);
    oldPDF *= max(beamBPtr->xfISR(iSys, idRec, xRec, q2)/xRec, TINYPDFtrial);
  } else {
    newPDF  = max(beamBPtr->xfISR(iSys, idi, xi, q2)/xi, TINYPDFtrial);
    newPDF *= max(beamAPtr->xfISR(iSys, idRec, xk, q2)/xk, TINYPDFtrial);
    oldPDF  = max(beamBPtr->xfISR(iSys, idMot, xMot, q2)/xMot, TINYPDFtrial);
    oldPDF *= max(beamAPtr->xfISR(iSys, idRec, xRec, q2)/xRec, TINYPDFtrial);
  }

  double Rpdf = newPDF/oldPDF;
  pAccept *= Rpdf;

  // Compute the trial kernel and collinear momentum fractions.
  double Q2(sij - mj2), xA((sAnt + sij)/sik), xj((sij + sjk - mj2)/sik),
    aTrial(brTrial->c0*(sik/sAnt)/Q2/xj);

  // Physical kernel.
  vector<AntWrapper> aPhys = ampCalcPtr->antFuncII(Q2, xA, xj,
    idMot, idi, idj, 0, 0, mj, polMot);
  double aPhysSum = 0;
  map<double,int> aPhysCumulative;
  for (int i = 0; i < (int)aPhys.size(); i++) {
    double aNow = aPhys[i].val;
    if (isnan(aNow) || isinf(aNow)) {
      string msg(": amplitude is " + string(isnan(aNow) ? "NAN" : "infinite"));
      infoPtr->errorMsg("Error in " + __METHOD_NAME__, msg + ".");
      infoPtr->setAbortPartonLevel(true);
      return false;
    }
    if (aNow > 0.) {
      aPhysSum += aNow;
      aPhysCumulative.insert({aPhysSum, i});
    }
  }

  // Add antenna ratio.
  pAccept *= aPhysSum/aTrial;
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Rpdf = " << Rpdf << " aPhys/aTrial = "
       << aPhysSum/aTrial << " pAccept = " << pAccept;
    printOut(__METHOD_NAME__, ss.str());}

  // Check if the veto probability is valid.
  if (pAccept > 1) {
    stringstream ss;
    ss << ": incorrect overestimate ("
       << idMot << ", " << polMot << ") -> " << idi << ", " << idj
       << ": aPhys/aTrial = " << aPhysSum/aTrial << " Rpdf = " << Rpdf;
    infoPtr->errorMsg("Warning in " + __METHOD_NAME__, ss.str());
  }

  // Perform veto.
  if (rndmPtr->flat() > pAccept) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,"Failed to pass veto.");
    return false;
  } else if (verbose >= DEBUG) printOut(__METHOD_NAME__, "Passed veto.");

  // Branching accepted! Now select a spin state.
  double aSelect = rndmPtr->flat() * aPhysSum;
  auto it = aPhysCumulative.upper_bound(aSelect);
  if (it == aPhysCumulative.end()) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__,
      ": logic erro -: cumulative sum < aPhysSum");
    return false;
  }
  poliTrial = aPhys[it->second].poli;
  poljTrial = aPhys[it->second].polj;

  // Sample Breit-Wigner mass for the kinematics, check off-shell phase space.
  double mjKin  = ampCalcPtr->sampleMass(idj, poljTrial);
  double mj2Kin = pow2(mjKin);
  double sikKin = sAnt + sij + sjk - mj2Kin;
  if (sikKin*sij*sjk - pow2(sikKin)*mj2Kin < 0) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Outside phase space: Off-shell");
    return false;
  }

  // Check if we veto resonances with offshellness > q2.
  if (vetoResonanceProduction) {
    double mj2Onshell  = pow2(ampCalcPtr->dataPtr->mass(idj));
    double q2Offshellj = fabs(mj2Kin - mj2Onshell);
    if (q2Trial < q2Offshellj) return false;
  }

  // Pre- and post-branching momenta, global recoil momenta.
  vector<Vec4> pOld{pMot, pRec};
  pNew.clear();
  pRecVec.clear();
  iRecVec.clear();

  // Collect the recoiling final state particles.
  int sysSize = partonSystemsPtr->sizeAll(iSys);
  for (int i = 0; i < sysSize; i++) {
    int iEv = partonSystemsPtr->getAll(iSys, i);
    if (iEv < 0 || !event[iEv].isFinal()) continue;
    pRecVec.push_back(event[iEv].p());
    iRecVec.push_back(iEv);
  }

  // Generate phi.
  double phi = rndmPtr->flat()*2*M_PI;
  Vec4 pSumBefore = pOld[0] + pOld[1];
  for (int i=0; i<(int)pRecVec.size(); i++) pSumBefore -= pRecVec[i];

  // Kinematics and return.
  if (!vinComPtr->map2to3II(pNew, pRecVec, pOld, sAnt, sij, sjk, sikKin, phi,
      mj2Kin)) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__,"Failed to generate kinematics.");
    return false;
  }
  return true;

}

//--------------------------------------------------------------------------

// Update an event.

void EWAntennaII::updateEvent(Event &event) {

  // Setup system, indices, masses, colors. Carry I color indices to i.
  iReplace.clear();
  shat = 0.;
  Particle parti, partj, partk;
  int idi( brTrial->idi), idj(brTrial->idj);
  double mj = pNew[1].mCalc();
  int col(event[iMot].col()), acol(event[iMot].acol());
  parti = Particle(idi, -41, 0, 0, 0, 0, col, acol, pNew[0], 0, 0, poliTrial);
  partj = Particle(idj, 43, 0, 0, 0, 0, 0, 0, pNew[1], mj, 0, poljTrial);
  partk = event[iRec];
  partk.p(pNew[2]);
  partk.statusCode(-42);
  int iEv, jEv, kEv;
  if (iMot > iRec) {
    iEv = event.append(parti);
    jEv = event.append(partj);
    kEv = event.append(partk);
  } else {
    kEv = event.append(partk);
    jEv = event.append(partj);
    iEv = event.append(parti);
  }

  // Save information for parton system.
  jNew = jEv;
  iReplace[iMot] = iEv;
  iReplace[iRec] = kEv;

  // Set old particles to negative.
  event[iMot].statusNeg();
  event[iRec].statusNeg();
  event[iEv].mothers(event[iMot].mother1(), event[iMot].mother2());
  event[jEv].mothers(iEv, 0);
  event[kEv].mothers(event[iRec].mother1(), event[iRec].mother2());
  event[iMot].mothers(iEv, 0);
  event[iRec].mothers(kEv, 0);
  event[iEv].daughters(jEv, iMot);
  event[kEv].daughters(iRec);
  event[jEv].daughters(0, 0);

  // Update beam daughters.
  if (iSys == 0) {
    bool founda(false), foundb(false);
    for (int i = 0; i < (int)event.size(); i++) {
      if (!founda && event[i].daughter1() == iMot) {
          event[i].daughters(iEv, 0);
          founda = true;
      }
      if (!foundb && event[i].daughter1() == iRec) {
          event[i].daughters(kEv, 0);
          foundb = true;
      }
      if (founda && foundb) break;
    }
  }

  // Update event for global recoil.
  for (int j = 0; j < event.size(); j++)
    if (event[j].isFinal())
      for (int k=0; k<(int)iRecVec.size(); k++)
        if (iRecVec[k] == j) {
          // Copy the recoiler.
          int inew = event.copy(j, 44);
          // Change the momentum.
          event[inew].p(pRecVec[k]);
          // Save information for parton system.
          iReplace[iRecVec[k]] = inew;
        }

  // Fix sHat for parton system.
  shat = (event[iEv].p() + event[kEv].p()).m2Calc();
  event.restorePtrs();

}

//--------------------------------------------------------------------------

// Update a parton system.

void EWAntennaII::updatePartonSystems(Event &event) {
  EWAntenna::updatePartonSystems(event);

  // Get new initial state indices from the updated parton system.
  int indexA = partonSystemsPtr->getInA(iSys);
  int indexB = partonSystemsPtr->getInB(iSys);

  // Update beams.
  (*beamAPtr)[iSys].update(indexA, event[indexA].id(),
    event[indexA].e()/beamAPtr->e());
  (*beamBPtr)[iSys].update(indexB, event[indexB].id(),
    event[indexB].e()/beamBPtr->e());

}

//==========================================================================

// Class that performs electroweak showers in a single parton system.

//--------------------------------------------------------------------------

// Build a system.

bool EWSystem::buildSystem(Event &event) {

  // Verbose output.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Building system " << iSysSav;
    printOut(__METHOD_NAME__, ss.str());
    event.list(); partonSystemsPtr->list();}

  // Clear out previous antennae and check for initial state.
  // (Assumes iSysSav set by prior call to prepare()).
  clearAntennae();
  if (partonSystemsPtr->hasInAB(iSysSav) && !resDecOnlySav) {
    // Set up initial antennae.
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Setting up initial antennae");

    // Get initial state particles.
    int indexA = partonSystemsPtr->getInA(iSysSav);
    int indexB = partonSystemsPtr->getInB(iSysSav);

    // Check if the initial state is polarized.
    if (event[indexA].pol() == 9 || event[indexB].pol() == 9) {
      if (verbose >= DEBUG) event.list();
      if (verbose >= REPORT) {
        stringstream ss;
        ss << ": cannot build EW system  (iSysSav "
           << iSysSav << ") - incoming parton(s) do not have helicities.";
        infoPtr->errorMsg("Warning in " + __METHOD_NAME__, ss.str());
      }
      clearAntennae();
      return false;
    }
    EWAntennaII antII(beamAPtr, beamBPtr);
    addAntenna(antII,antVecInitial,event,indexA,indexB,brMapInitial);
    addAntenna(antII,antVecInitial,event,indexB,indexA,brMapInitial);
  }

  // Get final-state particles.
  vector<int> indexFinal;
  int sizeOut = partonSystemsPtr->sizeOut(iSysSav);
  for (int iOut = 0; iOut < sizeOut; ++iOut) {
    int i = partonSystemsPtr->getOut(iSysSav, iOut);
    // Skip non-resonances if only doing resonance decays.
    int idAbs = event[i].idAbs();
    if (resDecOnlySav && !ampCalcPtr->dataPtr->isRes(idAbs)) continue;
    if (!event[i].isFinal()) {
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": partonSystems "
        "corrupted (non-final outgoing parton)");
      continue;
    }
    // Make sure the state is polarized (ignore gluons).
    if (event[i].id() != 21 && event[i].pol() == 9) {
      if (resDecOnlySav) {
        // Force select random polarisation.
        stringstream ss; ss<<idAbs;
        infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": assigning "
          +"random polarisation for decaying resonance.","id = "+ss.str());
        // Spin 0:
        if (idAbs == 25) {
          event[i].pol(0.);
        }
        // Spin 1/2 fermions:
        else if (idAbs < 20) {
          double pol = (rndmPtr->flat() > 0.5 ? 1 : -1);
          event[i].pol(pol);
        }
        // Spin 1:
        else {
          double ran = rndmPtr->flat()*3 - 1.5;
          if (ran < -0.5) event[i].pol(-1);
          else if (ran < 0.5) event[i].pol(0);
          else event[i].pol(1);
        }
      } else {
        if (verbose >= DEBUG) event.list();
        if (verbose >= REPORT) {
          stringstream ss;
          ss << ": cannot build EW system (iSysSav "
             << iSysSav << ") - outgoing parton(s) do not have helicities.";
          infoPtr->errorMsg("Warning in " + __METHOD_NAME__, ss.str());
        }
        clearAntennae();
        return false;
      }
    }
    indexFinal.push_back(i);
  }

  // Return all OK if nothing to do.
  if (indexFinal.size() == 0) return true;

  // Set up final antennae. If there is only one final state particle.
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "Finding recoilers for final antennae");
  if (indexFinal.size() == 1) {
    int iEv(indexFinal[0]), idAbs(event[iEv].idAbs());
    // Check if it is a resonance. In that case it can only decay.
    if (ampCalcPtr->dataPtr->isRes(idAbs)) {
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, "Resonance without recoilers");
      EWAntennaFFres antRes;
      addAntenna(antRes, antVecRes,event,iEv,0,brMapResonance);
    // Otherwise don't do anything.
    } else  if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "No recoilers available");
    return true;
  // If two final state particles.
  } else if (indexFinal.size() == 2) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Single recoiler available");
    int iEv(indexFinal[0]), jEv(indexFinal[1]);
    if (!resDecOnlySav) {
      EWAntennaFF antFF;
      addAntenna(antFF, antVecFinal, event, iEv, jEv, brMapFinal);
      addAntenna(antFF, antVecFinal, event, jEv, iEv, brMapFinal);
    }
    EWAntennaFFres antRes;
    addAntenna(antRes, antVecRes, event, iEv, jEv, brMapResonance);
    addAntenna(antRes, antVecRes, event, jEv, iEv, brMapResonance);
  } else {
    for (int i = 0; i < (int)indexFinal.size(); i++) {
      int iEv(indexFinal[i]), idi(event[iEv].id()), poli(event[iEv].pol());
      Vec4 pi = event[iEv].p();

      // Filter out gluons immediately.
      if (idi == 21) continue;

      // If last scale was below q2cut only consider resonances.
      // (as these still have to decay).
      if (lastWasBelowCut && !ampCalcPtr->dataPtr->isRes(idi)) continue;
      if (resDecOnlySav && !ampCalcPtr->dataPtr->isRes(idi)) continue;
      if (verbose >= DEBUG) {
        stringstream ss;
        ss << "Searching for recoiler for "
           << iEv << " which is a " << idi;
        printOut(__METHOD_NAME__, ss.str());}

      // Check if there are any branchings of this particle.
      auto itFinal = brMapFinal->find(make_pair(idi, poli));
      auto itRes = brMapResonance->find(make_pair(idi, poli));
      if (itFinal == brMapFinal->end() && itRes == brMapResonance->end())
        continue;

      // Collect all potential recoilers and compute their weights.
      vector<int> jEvCluster, jEvBackup;
      double aTotalSum = 0.;
      map<double, int> aSumSoFar;
      for (int j = 0; j < (int)indexFinal.size(); j++) {
        int jEv(indexFinal[j]), idj(event[jEv].id()), polj(event[jEv].pol());
        Vec4 pj = event[jEv].p();

        // Don't recoil against yourself.
        if (jEv == iEv) {continue;}
        if (verbose >= DEBUG) {
          stringstream ss;
          ss << "  Candidate " << jEv << " which is a " << idj;
          printOut(__METHOD_NAME__, ss.str());}
        double aSum = 0;
        int idiCluster(idi), idjCluster(idj), poliCluster(poli),
          poljCluster(polj);
        Vec4 piCluster(pi), pjCluster(pj);

        // Check if there are any clusterings.
        auto cluIt = cluMapFinal->find(make_pair(idi, idj));
        // Did we find any clusterings?
        if (cluIt == cluMapFinal->end()) {

          // idi and idj may be in the wrong order.
          // Check the flipped case.
          cluIt = cluMapFinal->find(make_pair(idj, idi));

          // Have to flip everything.
          if (cluIt != cluMapFinal->end() ) {
            idiCluster = idj;
            idjCluster = idi;
            poliCluster = polj;
            poljCluster = poli;
            piCluster = pj;
            pjCluster = pi;
          // We didn't find any clusterings.
          // Add the recoiler to the list of backups.
          } else {jEvBackup.push_back(jEv); continue;}
        }

        // This is a vector of all possible clusterings of idi and idj.
        vector<pair<int, int> > cluVec = cluIt->second;
        if (verbose >= DEBUG) {
          stringstream ss;
          ss << "  Found " << cluVec.size() << " clusterings.";
          printOut(__METHOD_NAME__, ss.str());}

        // Loop over cluster options for k.
        for (int k = 0; k < (int)cluVec.size(); k++) {
          int idMotCluster(cluVec[k].first), polMotCluster(cluVec[k].second);

          // Get the on-shell mass for idMotCluster.
          double mMotCluster(ampCalcPtr->dataPtr->mass(idMotCluster)), aNew(0);

          // Check in case we are considering the products of an on-shell
          // decay. In that case, we will find infinite branching kernel.
          // We should always select this recoiler.
          // Set the contribution to an arbitrary large value.
          if ((piCluster + pjCluster).m2Calc() - pow2(mMotCluster) == 0)
            aNew = 1./NANO;
          // Compute 1->2 branching kernel for this clustering.
          else {
            if (idiCluster == 0 || idjCluster == 0 ||
              abs(poliCluster)> 1 || abs(poljCluster)>1 ) {
              if (verbose >= NORMAL) {
                stringstream ss;
                ss << ": failed to set clustering ids / pols:"
                   << " idiCluster = " << idiCluster
                   << " idjCluster = " << idjCluster
                   << " poliCluster = " << poliCluster
                   << " poljCluster = " << poljCluster;
                infoPtr->errorMsg("Warning in " + __METHOD_NAME__, ss.str());
              }
              aNew = 0.;
            } else{
              aNew = ampCalcPtr->branchKernelFF(piCluster, pjCluster,
                idMotCluster, idiCluster, idjCluster, mMotCluster,
                0, polMotCluster, poliCluster, poljCluster);
            }
          }
          aSum += aNew;
          if (verbose >= DEBUG) {
            stringstream ss;
            ss << "    Clustered to ("
               << cluVec[k].first << ", " << cluVec[k].second << ") a = "
               << aNew;
            printOut(__METHOD_NAME__, ss.str());}
        }

        // Store candidate recoiler if aSum > 0.
        if (aSum > 0) {
          jEvCluster.push_back(jEv);
          aTotalSum += aSum;
          aSumSoFar.insert({aTotalSum, jEv});
        // Else add to backups.
        } else jEvBackup.push_back(jEv);
      }

      // Select a recoiler.
      int jEvRecoiler;
      // If there were no clusterings and no backups, we panic.
      if (jEvCluster.size() == 0 && jEvBackup.size() == 0) {
        infoPtr->errorMsg("Error in " + __METHOD_NAME__,
          ": unable to find a recoiler");
        return false;
      }

      // If there were no clusterings, select a backup at random.
      if (jEvCluster.size() == 0)
        jEvRecoiler = jEvBackup[jEvBackup.size()*rndmPtr->flat()];
      // Pick one with weighted probability.
      else {
        double aClusSelect = aTotalSum*rndmPtr->flat();
        auto it = aSumSoFar.upper_bound(aClusSelect);
        if (it == aSumSoFar.end()) {
          infoPtr->errorMsg("Error in "+__METHOD_NAME__,
            ": logic error: aSumSoFar < aTotalSum");
          return false;
        }
        jEvRecoiler = it->second;
      }
      if (!resDecOnlySav) {
        EWAntennaFF antFF;
        addAntenna(antFF,antVecFinal,event,iEv,jEvRecoiler,brMapFinal);
      }
      EWAntennaFFres antRes;
      addAntenna(antRes,antVecRes,event,iEv,jEvRecoiler,brMapResonance);
    }
  }

  // Return.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Printing system"); printAntennae();}
  return true;

}

//--------------------------------------------------------------------------

// Generate the next Q2.

double EWSystem::q2Next(double q2Start,double q2End) {
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "begin (with " << antVecFinal.size() << " FF radiators, "
       << antVecInitial.size() << " II radiators, and " << antVecRes.size()
       << " resonance decays)";
    printOut(__METHOD_NAME__, ss.str(), dashLen);}

  // Generate a scale, clear saved trial information.
  double alphaMax = al->alphaEM(q2Start);
  clearLastTrial();

  // Generate from FF EW antennae. Stop evolution at the EW cutoff.
  double q2EndLocal = max(q2End,q2Cut);
  generateTrial(antVecFinal,q2Start,q2EndLocal,alphaMax);

  // Generate from II EW antennae. Stop evolution at the EW cutoff.
  generateTrial(antVecInitial,q2Start,q2EndLocal,alphaMax);

  // Generate from res decay EW antennae.
  // Resonances must decay, so permit "evolution" all the way down to zero.
  // (Scales are BW-generated below matching scale).
  // (Note: later may want to keep long-lived resonances undecayed eg until
  // after first pass of hadronisation cf Early vs Late Resonance Decays.)
  generateTrial(antVecRes,q2Start,q2End,alphaMax);

  // Did we abort?
  if (infoPtr->getAbortPartonLevel()) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__, ": abort was called");
    return 0.;
  }
  if (verbose >= DEBUG && hasTrial() &&
    (q2Trial > q2Cut || lastWasResonanceDecay() )) {
    stringstream ss;
    ss << "Winner has particle I = " << antTrial->getIndexMot()
       << " with scale q2 = " << q2Trial;
    printOut(__METHOD_NAME__, ss.str());
  }
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return q2Trial;

}

//==========================================================================

// Top-level class for the electroweak shower module.

//--------------------------------------------------------------------------

// Initialize.

void VinciaEW::init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn) {

  // Did we load the data?
  if (!isLoaded) return;

  // Settings.
  doEW = (settingsPtr->mode("Vincia:ewMode") >= 3);

  // Initialize alphaEM.
  double alpEM0Vincia = settingsPtr->parm("Vincia:alphaEM0");
  double alpEMmzVincia = settingsPtr->parm("Vincia:alphaEMmz");
  double alpEM0Pythia = settingsPtr->parm("StandardModel:alphaEM0");
  double alpEMmzPythia = settingsPtr->parm("StandardModel:alphaEMmZ");
  int alphaEMorder = settingsPtr->mode("Vincia:alphaEMorder");

  // Change Pythia settings, initialize, then change them back.
  settingsPtr->parm("StandardModel:alphaEM0", alpEM0Vincia);
  settingsPtr->parm("StandardModel:alphaEMmZ", alpEMmzVincia);
  al.init(alphaEMorder, settingsPtr);
  settingsPtr->parm("StandardModel:alphaEM0", alpEM0Pythia);
  settingsPtr->parm("StandardModel:alphaEMmz", alpEMmzPythia);
  q2minSav = pow2(settingsPtr->parm("Vincia:QminChgQ"));

  // Set beam pointers.
  beamAPtr = beamAPtrIn;
  beamBPtr = beamBPtrIn;

  // Initialize AmpCalculator.
  ampCalc.init(&ewData, &cluMapFinal, &cluMapInitial);

  // Create EW shower system.
  ewSystem = EWSystem(&brMapFinal,&brMapInitial, &brMapResonance,
    &cluMapFinal, &cluMapInitial, &ampCalc);

  // Initialize the EW system.
  ewSystem.initPtr(infoPtr, vinComPtr, &al);
  ewSystem.init(beamAPtr, beamBPtr);
  ewSystem.setVerbose(verbose);

  isInitSav = true;

}

//--------------------------------------------------------------------------

// Prepare to shower a system.

bool VinciaEW::prepare(int iSysIn, Event &event, bool isBelowHadIn) {

  // Sanity check.
  if (!doEW) return false;
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Prepare system
  if (!ewSystem.prepare(event, iSysIn, q2minSav, isBelowHadIn)) {
    if (verbose >= REPORT) {
      infoPtr->errorMsg("Warning in " + __METHOD_NAME__+": failed "
        "to prepare EW shower system.");
    }
    return false;
  }
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return true;
}

//--------------------------------------------------------------------------

// Save information on masses and widths, and load branchings.

void VinciaEW::load() {

  // Read settings.
  verbose = settingsPtr->mode("Vincia:verbose");
  nFlavZeroMass = settingsPtr->mode("Vincia:nFlavZeroMass");
  doFFbranchings = settingsPtr->flag("PartonLevel:FSR")
    && settingsPtr->flag("Vincia:doFF");
  doIIbranchings = settingsPtr->flag("PartonLevel:ISR")
    && settingsPtr->flag("Vincia:doII");
  doRFbranchings = settingsPtr->flag("PartonLevel:FSR")
    && settingsPtr->flag("Vincia:doRF");

  // Fetch multipliers for overestimate functions.
  headroomFinal = settingsPtr->parm("Vincia:EWheadroomF");
  headroomInitial = settingsPtr->parm("Vincia:EWheadroomI");

  // Check if we are using the bosonic interference factor.
  doBosonInterference = settingsPtr->flag("Vincia:doBosonicInterference");

  // Load possible EW branchings from XML.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "Loading EW branchings.");
  if (!readFile(settingsPtr->word("xmlPath") + "VinciaEW.xml")) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": failed to read XML file.");
    return;
  }

  // In DEBUG mode, print and check for doubles between Final and Res maps.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Successfully read XML file.");
    unordered_map<pair<int, int>, vector<EWBranching> >::iterator itFin, itRes;
    for (itFin = brMapFinal.begin(); itFin != brMapFinal.end(); itFin++) {

      // Find corresponding entry in brMapResonance.
      itRes = brMapResonance.find(itFin->first);
      if (itRes != brMapResonance.end()) {
        vector<EWBranching> brVecFinal = itFin->second;
        vector<EWBranching> brVecResonance = itRes->second;

        // Loop over all branchings and compare.
        for (int i = 0; i < (int)brVecFinal.size(); i++) {
          int idi(brVecFinal[i].idi), idj(brVecFinal[i].idj);
          for (int j = 0; j < (int)brVecResonance.size(); j++) {
            if (idi == brVecResonance[j].idi && idj == brVecResonance[j].idj) {
              infoPtr->errorMsg("Error in " + __METHOD_NAME__,
                ": duplicates between Final and Resonance shower.");
              return;
            }
          }
        }
      }
    }
    printBranchings();
    printData();
  }
  isLoaded=true;

}

//--------------------------------------------------------------------------

// Generate a trial scale.

double VinciaEW::q2Next(Event&, double q2Start, double q2End) {

  if (!doEW) return 0.0;
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "begin (with " << ewSystem.nBranchers() << " branchers)";
    printOut(__METHOD_NAME__, ss.str(), dashLen);
  }
  q2Trial = ewSystem.q2Next(q2Start,q2End);
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "q2Trial = " << num2str(q2Trial);
    printOut(__METHOD_NAME__, ss.str());
    printOut(__METHOD_NAME__, "end", dashLen);
  }
  return q2Trial;
}

//--------------------------------------------------------------------------

// Print loaded data on masses and widths.

void VinciaEW::printData() {
  cout << "\n  *************************************************************"
       << "\n  Saved particle data: \n\n";
  for (auto it = ewData.begin(); it != ewData.end(); ++it)
    cout << "    id = " << it->first.first << "  pol = " << it->first.second
         << "  m = " << it->second.mass << "  w = " << it->second.width
         << "  isRes: " << (it->second.isRes ? "yes": "no") << "\n";
  cout << "\n  *************************************************************"
       << "\n";
}

//--------------------------------------------------------------------------

// Print branchings.

void VinciaEW::printBranchings() {
  string sep = "  *******************************************************"
    "********************\n";
  stringstream ss;
  ss << "Printing branchings in format:\n"
     << "    ( I , polMot ) ->  i, j : (c0, c1, c2, c3) : (mMot, mi, mj)";
  printOut(__METHOD_NAME__,ss.str());

  // Print final-state branchings.
  cout << sep << "  Final-state branchings\n" << sep;
  for (auto it = brMapFinal.begin(); it != brMapFinal.end(); it++) {
    vector<EWBranching> brVec = it->second;
    for (int i = 0; i < (int)brVec.size(); i++) brVec[i].print();
  }

  // Print resonance branchings.
  cout << "\n" << sep << "  Resonance-decay branchings\n" << sep;
  for (auto it = brMapResonance.begin(); it != brMapResonance.end(); it++) {
    vector<EWBranching> brVec = it->second;
    for (int i = 0; i < (int)brVec.size(); i++) brVec[i].print();
  }

  // Print initial-state branchings.
  cout << "\n" << sep << "  Initial-state branchings\n" << sep;
  for (auto it = brMapInitial.begin(); it != brMapInitial.end(); it++) {
    vector<EWBranching> brVec = it->second;
    for (int i = 0; i < (int)brVec.size(); i++) brVec[i].print();
  }
  cout << "\n" << sep;

}

//--------------------------------------------------------------------------

// Read an XML file.

bool VinciaEW::readFile(string file)  {

  ifstream is(file.c_str());
  if (!is.good()) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": couldn't open XML file" + file);
    return false;
  }
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "Reading file " + file);

  string line;
  bool foundBranching = false;
  stringstream branch("");
  bool skip = false;
  while(getline(is, line)) {
    if (line.find("<!--") != string::npos) {
      // Found commented out section.
      skip = true;
      // End commented out section.
      if (line.find("-->") != string::npos) skip = false;
      continue;
    } else if (line.find("-->") != string::npos) {
      // End commented out section.
      skip = false;
      continue;
    }
    if (!skip) {
      if (line.find("<EWBranching") != string::npos) foundBranching = true;
      else if (line.find("</EWBranching") != string::npos) {
        foundBranching = false;
        // Closed XML bracket: now process string.
        if (!readLine(branch.str())) {
          infoPtr->errorMsg("Error in " + __METHOD_NAME__,
            ": couldn't read line:\n" + branch.str());
          return false;
        }
        // Clear stream.
        branch.str("");
      }

      // Not finished yet-> add to current stream.
      if (foundBranching) branch << line;
    }
  }

  // Done.
  return true;

}

//--------------------------------------------------------------------------

// Read an XML line.

bool VinciaEW::readLine(string line)  {
  if (line.find("EWBranchingFinal") != string::npos) {
    if (doFFbranchings) return addBranching(line, brMapFinal,
      cluMapFinal, headroomFinal, false);
    else return true;
  } else if (line.find("EWBranchingInitial") != string::npos) {
    if (doIIbranchings) return addBranching(line, brMapInitial,
      cluMapInitial, headroomInitial, false);
    else return true;
  } else if (line.find("EWBranchingRes") != string::npos) {
    if (doRFbranchings) return addBranching(line, brMapResonance,
      cluMapFinal, headroomFinal, true);
    else return true;
  } else {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": unknown EW branch type in database.");
    return false;
  }
}

//--------------------------------------------------------------------------

// Fill a string with an attribute value.

bool VinciaEW::attributeValue(string line, string attribute, string& val) {
  size_t iBegAttri = line.find(attribute);
  if (iBegAttri > line.length()) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": could not find attribute " + attribute);
    return false;}
  size_t iBegQuote = line.find('"', iBegAttri + 1);
  if (iBegQuote > line.length()) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": could not extract value for attribute " + attribute);
    return false;}
  size_t iEndQuote = line.find('"', iBegQuote + 1);
  if (iEndQuote > line.length()) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": could not extract value for attribute " + attribute);
    return false;}
  val = line.substr(iBegQuote + 1, iEndQuote - iBegQuote - 1);
  return true;
}

//--------------------------------------------------------------------------

// Read data from file and add branching to list.

bool VinciaEW::addBranching(string line, unordered_map< pair<int, int>,
  vector<EWBranching> >& branchings, unordered_map< pair<int, int>,
  vector<pair<int, int> > >& clusterings, double headroom, bool decay) {

  // Particle IDs and overestimate coefficents
  int idMot, idi, idj, polMot;
  double c0, c1, c2, c3;

  // Set attributes.
  if      (!attributeValue<int>   (line,"idMot",  idMot))  return false;
  else if (!attributeValue<int>   (line,"idi",    idi))    return false;
  else if (!attributeValue<int>   (line,"idj",    idj))    return false;
  else if (!attributeValue<int>   (line,"polMot", polMot)) return false;
  else if (!attributeValue<double>(line,"c0",     c0))     return false;
  else if (!attributeValue<double>(line,"c1",     c1))     return false;
  else if (!attributeValue<double>(line,"c2",     c2))     return false;
  else if (!attributeValue<double>(line,"c3",     c3))     return false;

  // Save information about particles to database.
  if (!addParticle(idMot, polMot, decay)) {return false;}

  // Loop over all pols of i.
  if (abs(idi) < 23) {
    if (!addParticle(idi, 1, false))  {return false;}
    if (!addParticle(idi, -1, false)) {return false;}
  } else if (abs(idi) == 25) {
    if (!addParticle(idi, 0, false))  {return false;}
  } else {
    if (!addParticle(idi, 1, false))  {return false;}
    if (!addParticle(idi, 0, false))  {return false;}
    if (!addParticle(idi, -1, false)) {return false;}
  }

  // Loop over all pols of j.
  if (abs(idj) < 23) {
    if (!addParticle(idj, 1, false))  {return false;}
    if (!addParticle(idj, -1, false)) {return false;}
  } else if (abs(idj) == 25) {
    if (!addParticle(idj, 0, false))  {return false;}
  } else {
    if (!addParticle(idj, 1, false))  {return false;}
    if (!addParticle(idj, 0, false))  {return false;}
    if (!addParticle(idj, -1, false)) {return false;}
  }

  // Multiply by overestimate headroom.
  c0 *= headroom;
  c1 *= headroom;
  c2 *= headroom;
  c3 *= headroom;

  // Save.
  pair<int,int> branchKey  = make_pair(idMot, polMot);
  pair<int,int> clusterKey = make_pair(idi, idj);

  // First time we've seen this branching? -> create empty vector.
  if (branchings.find(branchKey) == branchings.end())
    branchings[branchKey] = vector<EWBranching>();
  branchings[branchKey].push_back(EWBranching(idMot,idi,idj,polMot,
      c0,c1,c2,c3));

  // First time we've seen this clustering? -> create empty vector.
  if (clusterings.find(clusterKey) == clusterings.end())
    clusterings[clusterKey] = vector<pair<int,int>>();
  clusterings[clusterKey].push_back(branchKey);
  return true;

}

//--------------------------------------------------------------------------

// Save information locally about this particle from particleData.

bool VinciaEW::addParticle(int idIn, int polIn, bool isRes) {
  ParticleDataEntryPtr p = particleDataPtr->findParticle(idIn);
  if (p == nullptr) return false;

  // Extract properties. nFlavZeroMass quarks are massless.
  double mass  = abs(idIn) <= nFlavZeroMass ? 0 : p->m0();
  double width = p->mWidth();

  // Is the particle already in database.
  if (!ewData.find(idIn, polIn)) ewData.add(idIn, polIn, mass, width, isRes);
  if (isRes) {
    // Update if we previously set but got it wrong.
    auto pew = ewData.get(idIn, polIn);
    if (!pew->isRes) pew->isRes = true;
  }
  return true;

}

//==========================================================================

// Class to do the veto for overlapping QCD/EW shower phase space.

//--------------------------------------------------------------------------

// Initialize.

void VinciaEWVetoHook::init(shared_ptr<VinciaEW> ewShowerPtrIn) {

  // Extract some settings.
  ewShowerPtr = ewShowerPtrIn;
  bool vinOn  = (settingsPtr->mode("PartonShowers:model") == 2);
  bool doWeak = settingsPtr->mode("Vincia:EWmode") >= 3;
  deltaR      = settingsPtr->parm("Vincia:EWoverlapVetoDeltaR");
  q2EW        = pow2(settingsPtr->parm("Vincia:EWscale"));
  verbose     = settingsPtr->mode("Vincia:verbose");

  // Decide whether or not to activate vetoing.
  mayVeto = vinOn && doWeak && settingsPtr->flag("Vincia:EWOverlapVeto");
  if (mayVeto) printOut(__METHOD_NAME__, "EW+QCD PS merging veto is active.");
  else printOut(__METHOD_NAME__, "EW+QCD PS merging veto is NOT active.");

}

//--------------------------------------------------------------------------

// Veto an ISR emission.

bool VinciaEWVetoHook::doVetoISREmission(int sizeOld, const Event& event,
  int iSys) {

  // Was this an emission in an MPI/Resonance system? -> don't veto.
  if (iSys > 0) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Emission in MPI system: pass");
    return false;
  }

  // Obtain information about last branching.
  if (!setLastISREmission(sizeOld, event)) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": could not classify last ISR emission");
    return false;
  }
  bool doVeto = doVetoEmission(sizeOld, event, iSys);
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,
    ": ISR emission " + string(doVeto ?" vetoed." : "passed."));
  return doVeto;

}

//--------------------------------------------------------------------------

// Veto an FSR emission.

bool VinciaEWVetoHook::doVetoFSREmission(int sizeOld, const Event &event,
  int iSys, bool inResonance) {

  // If this is a resonance decay? -> don't veto.
  if (inResonance) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Emission in resonance decay system: pass");
    return false;
  }

  // Was this an emission in an MPI/Resonance system? -> don't veto.
  if (iSys > 0) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Emission in MPI system: pass");
    return false;
  }

  // Obtain information about last branching.
  if (!setLastFSREmission(sizeOld, event)) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": could not classify last FSR emission");
    return false;
  }
  bool doVeto = doVetoEmission(sizeOld, event, iSys);
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,
    ": FSR emission " + string(doVeto ?" vetoed." : "passed."));
  return doVeto;

}

//--------------------------------------------------------------------------

// Universal method called by doVeto(FSR/ISR)Emission.

bool VinciaEWVetoHook::doVetoEmission(int sizeOld, const Event& event,
  int iSys) {

  // Safety check.
  if (!mayVeto) {return false;}
  // Compare to last and decide whether to veto.
  bool doVeto = false;
  // Don't veto if lastkT2 is -1.
  if (lastkT2 < 0) {return doVeto;}
  // Compare to EW winner.
  if (lastIsQCD) {
    double ewWinnerkT2 = findEWScale(sizeOld, event, iSys);
    if (ewWinnerkT2 > 0 && lastkT2 > ewWinnerkT2) doVeto = true;
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Last emission was QCD with kT2 = "
         << lastkT2 << " comparing to lowest EW clustering kT2 = "
         << ewWinnerkT2;
      printOut(__METHOD_NAME__, ss.str());}
  // Compare to QCD winner.
  } else {
    double qcdWinnerkT2 = findQCDScale(sizeOld, event, iSys);
    if (qcdWinnerkT2 > 0 && lastkT2 > qcdWinnerkT2) doVeto = true;
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Last emission was EW with kT2 = "
         << lastkT2 << " comparing to lowest QCD clustering kT2 = "
         << qcdWinnerkT2;
      printOut(__METHOD_NAME__, ss.str());}
  }
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, doVeto ? "Veto emission." : "Pass.");
  return doVeto;

}

//--------------------------------------------------------------------------

// Find all QCD clusterings and evaluate their kt measure.

double VinciaEWVetoHook::findQCDScale(int sizeOld, const Event& event,
  int iSys) {
  unordered_map<int, int> acolToIndex, colToIndex;
  unordered_map<int, vector<int> > qFlavToIndex;

  // Note that partonSystem isn't up to date. We look through all old
  // things in partonSystems and add in the new stuff.
  // First deal with initial state.
  if (partonSystemsPtr->hasInAB(iSys)) {
    int inA = partonSystemsPtr->getInA(iSys);
    int inB = partonSystemsPtr->getInB(iSys);
    if (event[inA].mother1() != 1) inA = event[inA].mother1();
    if (event[inB].mother1() != 2) inB = event[inB].mother1();
    if (event[inA].isQuark()) {
      int id = event[inA].id();
      if (qFlavToIndex.find(-id) == qFlavToIndex.end())
        qFlavToIndex[-id] = vector<int>();
      qFlavToIndex[-id].push_back(inA);
    }
    if (event[inA].col() > 0) acolToIndex[event[inA].col()]  = inA;
    if (event[inA].acol() > 0) colToIndex[event[inA].acol()] = inA;
    if (event[inB].isQuark()) {
      int id = event[inB].id();
      if (qFlavToIndex.find(-id) == qFlavToIndex.end()) {
        qFlavToIndex[-id] = vector<int>();
      }
      qFlavToIndex[-id].push_back(inB);
    }
    if (event[inB].col() > 0) acolToIndex[event[inB].col()]  = inB;
    if (event[inB].acol() > 0) colToIndex[event[inB].acol()] = inB;
  }

  // Loop over parton systems.
  for (int iOut = 0; iOut < partonSystemsPtr->sizeOut(iSys); ++iOut) {
    int indexNow = partonSystemsPtr->getOut(iSys,iOut);

    // Skip if no longer in final state.
    if (!event[indexNow].isFinal()) continue;
    int id = event[indexNow].id();
    if (event[indexNow].col() > 0 || event[indexNow].acol() > 0) {
      // Save quark flavour.
      if (event[indexNow].isQuark()) {
        if (qFlavToIndex.find(id) == qFlavToIndex.end())
          qFlavToIndex[id] = vector<int>();
        qFlavToIndex[id].push_back(indexNow);
      }
      // Save colours.
      if (event[indexNow].col() > 0)
        colToIndex[event[indexNow].col()] = indexNow;
      if (event[indexNow].acol() > 0)
        acolToIndex[event[indexNow].acol()] = indexNow;
    }
  }

  // Loop over new entries and update maps.
  for (int indexNow = sizeOld; indexNow < event.size(); indexNow++) {
    // Skip if not in final state (we already updated incoming).
    if(!event[indexNow].isFinal()) continue;
    int id = event[indexNow].id();
    if (event[indexNow].col() > 0 || event[indexNow].acol() > 0) {
      // Save quark flavour.
      if (event[indexNow].isQuark()) {
        if (qFlavToIndex.find(id) == qFlavToIndex.end())
          qFlavToIndex[id] = vector<int>();
        qFlavToIndex[id].push_back(indexNow);
      }
      // Save colours.
      if (event[indexNow].col() > 0)
        colToIndex[event[indexNow].col()] = indexNow;
      if (event[indexNow].acol() > 0)
        acolToIndex[event[indexNow].acol()] = indexNow;
    }
  }

  // Number of colour lines should match.
  if (acolToIndex.size() != colToIndex.size()) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": unmatched colour lines.");
    return -1.;
  }
  double kT2Smallest = numeric_limits<double>::max();

  // Find all colour- connected clusterings.
  auto it = acolToIndex.begin();
  while(it != acolToIndex.end()) {
    int colTag(it->first), index(it->second);
    if (colToIndex.find(colTag) != colToIndex.end()) {
      int colPartner = colToIndex[colTag];
      // qg or gg.
      if (event[colPartner].isGluon() || event[index].isGluon()) {
        // Evaluate kT measure.
        double kT2Now = findktQCD(event, index, colPartner);
        if (kT2Now < kT2Smallest && kT2Now > 0) kT2Smallest = kT2Now;
      }
    }

    // Found an unmatched color index.
    else {
      infoPtr->errorMsg("Error in " + __METHOD_NAME__,
        ": unmatched colour lines.");
      return -1.;
    }

    // Erase this index and go to the next tag.
    acolToIndex.erase(colTag);
    it = acolToIndex.begin();
  }

  // Find all gluon splittings.  These can't be identified from the
  // colour structure. Loop over all quark flavours.
  for (int flav = 1; flav <= 6; flav++) {
    // No pairings of this flavour: skip.
    if (qFlavToIndex.find(flav) == qFlavToIndex.end() ||
        qFlavToIndex.find(-flav) == qFlavToIndex.end()) continue;
    auto itQuark   = qFlavToIndex[flav].begin();
    auto endQuark  = qFlavToIndex[flav].end();
    auto itaQuark  = qFlavToIndex[-flav].begin();
    auto endaQuark = qFlavToIndex[-flav].end();

    // Make all possible same flavour pairings.
    for (; itQuark != endQuark; ++itQuark) {
      for (; itaQuark != endaQuark; ++itaQuark) {
        // Don't cluster colour singlets.
        int col = event[*itQuark].isFinal() ?
          event[*itQuark].col() : event[*itQuark].acol();
        int acol = event[*itaQuark].isFinal() ?
          event[*itaQuark].acol() : event[*itaQuark].col();

        // Should not be colour singlets
        if (col == acol) continue;
        double kT2Now = findktQCD(event, *itQuark, *itaQuark);
        if (kT2Now < kT2Smallest && kT2Now > 0) kT2Smallest = kT2Now;
      }
    }
  }
  return kT2Smallest;

}

//--------------------------------------------------------------------------

// Find all QCD clusterings and evaluate their kt measure.

double VinciaEWVetoHook::findEWScale(int sizeOld, const Event& event,
  int iSys) {

  // Set up a list of EW charges
  vector<int> ewfsCharges;
  int inA = 0, inB = 0;

  // First deal with initial state.
  bool hasInitialState = partonSystemsPtr->hasInAB(iSys);
  if (hasInitialState) {
    inA = partonSystemsPtr->getInA(iSys);
    inB = partonSystemsPtr->getInB(iSys);

    // They may have updated.
    if (event[inA].mother1() != 1) inA = event[inA].mother1();
    if (event[inB].mother1() != 2) inB = event[inB].mother1();

    // Set to zero if a gluon
    if (event[inA].id() == 21) inA = 0;
    if (event[inB].id() == 21) inB = 0;
  }

  // Loop over parton systems.
  for (int iOut = 0; iOut < partonSystemsPtr->sizeOut(iSys); ++iOut) {
    int indexNow = partonSystemsPtr->getOut(iSys,iOut);

    // Skip if no longer in final state.
    if(!event[indexNow].isFinal()) continue;
    // Keep if not a gluon.
    if (event[indexNow].id() != 21) ewfsCharges.push_back(indexNow);
  }

  // Loop over new entries.
  for (int indexNow = sizeOld; indexNow < event.size(); indexNow++) {
    // Skip if not in final state (we already updated incoming)
    if (!event[indexNow].isFinal()) continue;
    // Keep if not a gluon.
    if (event[indexNow].id() != 21) ewfsCharges.push_back(indexNow);
  }
  double kT2Smallest = numeric_limits<double>::max();

  // Do the initial state first.
  for (int i=0; i < (int)ewfsCharges.size(); i++) {
    int indexi(ewfsCharges[i]), idi(event[indexi].id());

    // Is there a clustering?
    if (inA != 0) {
      int idA = event[inA].id();
      auto it = ewShowerPtr->cluMapInitial.find(make_pair(idA, idi));
      if (it != ewShowerPtr->cluMapInitial.end()) {
        double kT2Now = ktMeasure(event, inA, indexi, 0);
        if (kT2Now < kT2Smallest) kT2Smallest = kT2Now;
      }
    }

    // Is there a clustering?
    if (inB != 0) {
      int idB = event[inB].id();
      auto it = ewShowerPtr->cluMapInitial.find(make_pair(idB, idi));
      if (it != ewShowerPtr->cluMapInitial.end()) {
        double kT2Now = ktMeasure(event, inB, indexi, 0);
        if (kT2Now < kT2Smallest)  kT2Smallest = kT2Now;
      }
    }
  }

  // Loop over all pairs of final-state electroweak charges.
  for (int i = 0; i < (int)ewfsCharges.size(); i++) {
    for (int j = 0; j < i; j++) {
      int indexi(ewfsCharges[i]), indexj(ewfsCharges[j]);
      // Compute kT measure.
      double kT2Now = findktEW(event, indexi, indexj);
      // Returns -1 if not a branching we count
      if (kT2Now > 0 && kT2Now < kT2Smallest) kT2Smallest = kT2Now;
    }
  }
  return kT2Smallest;

}

//--------------------------------------------------------------------------

// Evaluate Durham kt measure for two particles i and j.

double VinciaEWVetoHook::ktMeasure(const Event &event, int indexi, int indexj,
  double mMot2) {
  if(indexi >= event.size() || indexj > event.size()) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": could not find clustering candidates in event record !");
    return -1;
  }
  Vec4 pi(event[indexi].p()), pj(event[indexj].p());
  double mi2(pi.m2Calc()), mj2(pj.m2Calc());
  double pTi_2 = pi.pT2() + fabs(mi2 + mj2 - mMot2);
  double pTj_2 = pj.pT2() + fabs(mi2 + mj2 - mMot2);

  // If one of the particles is in the initial state, just return kt2
  // of the other. If both are in the initial state, return -1.
  if (!event[indexi].isFinal() && event[indexj].isFinal()) return pTj_2;
  else if (event[indexi].isFinal() && !event[indexj].isFinal()) return pTi_2;
  else if (!event[indexi].isFinal() && !event[indexj].isFinal()) return -1.;
  return min(pTi_2, pTj_2) * pow2(RRapPhi(pi, pj))/deltaR;

}

//--------------------------------------------------------------------------

// Evaluate a QCD clustering - returns -1 if not a valid clustering.

double VinciaEWVetoHook::findktQCD(const Event& event, int indexi,
  int indexj) {

  // Both initial state? -> return -1.
  if (!event[indexi].isFinal() && !event[indexj].isFinal()) return -1.;

  // Check if it is a QCD emission.
  if (!event[indexi].isQuark() && !event[indexi].isGluon())
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": expected a QCD branching.");
  if (!event[indexj].isQuark() && !event[indexj].isGluon())
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": expected a QCD branching.");

  // Check if I was a gluon.
  double mMot2 = 0;
  if ( (event[indexi].isQuark() && event[indexj].isQuark()) ||
       (event[indexi].isGluon() && event[indexj].isGluon()) ) mMot2 = 0;
  // One is a quark: get its mass.
  else  mMot2 = max(event[indexi].m2Calc(), event[indexj].m2Calc());
  return ktMeasure(event, indexi, indexj, mMot2);

}

//--------------------------------------------------------------------------

// Evaluate an EW clustering - returns -1 if not a valid clustering.

double VinciaEWVetoHook::findktEW(const Event& event, int indexi,
  int indexj) {
  int idi(event[indexi].id()), idj(event[indexj].id());

  // See if there are any clusterings.
  auto it = ewShowerPtr->cluMapFinal.find(make_pair(idi, idj));
  // No clusterings? -> Maybe the indices were in the wrong order.
  if (it == ewShowerPtr->cluMapFinal.end()) {
    swap(idi, idj);
    it = ewShowerPtr->cluMapFinal.find(make_pair(idi, idj));
  }

  // Is there still no clustering? Then return -1.
  if (it == ewShowerPtr->cluMapFinal.end()) return -1;
  // Only count if it involves a vector boson emission.
  if (abs(idj) < 20) return -1;
  // Also don't count t -> bW
  if (abs(idi) == 5 && abs(idj) == 24) return -1;
  double mMot2 = 0;

  // Neutral boson emission?
  if (abs(idj) != 24) {
    // Is it h->hh or h->ZZ?
    if (abs(idi) == abs(idj)) mMot2 = pow2(ewShowerPtr->ewData.mass(25));
    // Otherwise keep the emitter mass.
    else mMot2 = max(0., event[indexi].m2());
  // It's a W emission.
  } else {
    // gamma/Z/h -> W+ W-. Use the EW scale.
    if (abs(idi) == 24) mMot2 = q2EW;
    // f -> f'W. Get the f id from clustering
    else
      mMot2 = max(0., pow2(ewShowerPtr->ewData.mass(it->second[0].second)));
  }
  return ktMeasure(event, indexi, indexj, mMot2);

}

//--------------------------------------------------------------------------

// Look up last FSR emission info from event record.

bool VinciaEWVetoHook::setLastFSREmission(int sizeOld, const Event &event) {
  lastIsQCD = false;
  lastkT2 = 0.;
  int iEmit = 0;
  vector<int> status51, status52;
  for (int iPart = sizeOld; iPart < event.size(); ++iPart) {
    if (event[iPart].status() == 51) {
      int moth1(event[iPart].mother1()), moth2(event[iPart].mother2());
      // Is it an emission?
      if (moth1 > 0 && moth2 > 0 && moth1 != moth2) iEmit = iPart;
      else status51.push_back(iPart);
    } else if(event[iPart].status() == 52) status52.push_back(iPart);
  }

  if (status51.size() != 2) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": unexpected number of status 51 in last branching.");
    return false;
  }

  // Check if it is a 2->3 QCD g emission.
  if (iEmit != 0 && event[iEmit].isGluon()) {
    lastIsQCD = true;
    double kT2_1 = findktQCD(event, iEmit, status51[0]);
    double kT2_2 = findktQCD(event, iEmit, status51[1]);
    // Save smallest of the two kTs.
    lastkT2 = kT2_1 < kT2_2 ? kT2_1 : kT2_2;
  }

  // Otherwise it should be a 1->2 branching.
  else if (iEmit == 0 && status52.size() == 1 &&
    event[status51[0]].mother1() > 0 && event[status51[0]].mother2() == 0 &&
    event[status51[1]].mother1() > 0 && event[status51[1]].mother2() == 0 &&
    event[status51[0]].mother1() == event[status51[1]].mother1()) {
    int moth(event[status51[0]].mother1()), d1(status51[0]), d2(status51[1]),
      id1(event[d1].id()), id2(event[d2].id());

    // It could be a g->qqbar.
    if (event[moth].isGluon() && event[d1].isQuark()
      && event[d2].isQuark() && id1 == -id2) {
      lastIsQCD = true;
      lastkT2 = findktQCD(event, d1, d2);
    // Otherwise it should be an EW branching. Find the kT scale.
    } else {
      lastIsQCD = false;
      lastkT2 = findktEW(event, d1, d2);
    }
  // Don't know what this is.
  } else {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__, ": unknown branching.");
    return false;
  }
  return true;

}

//--------------------------------------------------------------------------

// Look up last ISR emission info from event record.

bool VinciaEWVetoHook::setLastISREmission(int sizeOld, const Event& event){
  lastIsQCD = false;
  lastkT2 = 0.;
  int iEmit = 0;
  vector<int> status41, status44;
  for (int iPart = sizeOld; iPart < event.size(); ++iPart){
    if (event[iPart].status() == 43) iEmit = iPart;
    else if(event[iPart].statusAbs() == 41) status41.push_back(iPart);
    else if(event[iPart].status() == 44) status44.push_back(iPart);
  }
  if (iEmit == 0){
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": could not find emission in event record.");
    return false;
  }

  // Is this a gluon emission?
  if (event[iEmit].isGluon()) {
    lastIsQCD = true;

    // Find colour-connected partners.
    int rad1(0), rad2(0);

    // Check status 41.
    // 2 x 41: II.
    if(status41.size()==2) {rad1 = status41[0]; rad2 = status41[1];
    // 1 x 41: IF.
    } else if (status41.size() == 1) {
      rad1 = status41[0];
      // Check status 44.
      if (status44.size() == 0) {
        infoPtr->errorMsg("Error in " + __METHOD_NAME__,
          ": wrong number of status 44 in event record.");
        return false;
      }
      // Loop over outgoing until find colour connected leg
      for(int irad = 0; irad < (int)status44.size(); ++irad) {
        int radNow = status44.at(irad);
        if ( (event[radNow].col() > 0 &&
              event[radNow].col() == event[iEmit].acol() ) ||
             (event[radNow].acol() > 0 &&
              event[radNow].acol() == event[iEmit].col() ) ) {
          rad2 = radNow; break;}
      }
    } else {
      infoPtr->errorMsg("Error in " + __METHOD_NAME__,
        ": wrong number of status 41 in event record.");
      return false;
    }

    if (rad1 == 0 || rad2 == 0) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,
        ": couldn't identify radiators of gluon in event record.");
      return false;
    }

    // Compare kt scales and pick smaller.
    double kT2_1 = findktQCD(event, iEmit, rad1);
    double kT2_2 = findktQCD(event, iEmit, rad2);
    lastkT2  = kT2_1 < kT2_2 ? kT2_1 : kT2_2;
  // If it is a quark, it can only be QCD.
  } else if (event[iEmit].isQuark()) {

    // Check if mother is a gluon or quark or something else.
    int moth = event[iEmit].mother1();
    if (moth == 0) {
      infoPtr->errorMsg("Error in " + __METHOD_NAME__,
        ": couldn't mother of quark emission.");
      return false;
    }

    if (event[moth].isGluon()) {
      lastIsQCD = true;
      lastkT2 = findktQCD(event, iEmit, moth);
    } else if (event[moth].isQuark()) {
      // Find other daughter.
      int dOther = event[moth].daughter1();
      if (event[dOther].isQuark()) dOther = event[moth].daughter2();

      // Was this backwards g splitting?
      if (event[dOther].isGluon()) {
        lastIsQCD = true;
        lastkT2 = findktQCD(event, iEmit, moth);
      } else {
        infoPtr->errorMsg("Error in " + __METHOD_NAME__,
          ": unknown branching.");
        return false;
      }
    } else {
      infoPtr->errorMsg("Error in " + __METHOD_NAME__,
        ": unknown branching.");
      return false;
    }
  // Must be an EW emission otherwise.
  } else if (event[iEmit].idAbs() >= 22 && event[iEmit].idAbs() <= 25){
    // Find radiator.
    if (status41.size() == 1) {
      lastkT2 = ktMeasure(event, status41[0], iEmit, 0);
    } else {
      infoPtr->errorMsg("Error in " + __METHOD_NAME__,
        ": wrong number of status 41 in event record.");
      return false;
    }
  } else {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__,
      ": unknown branching.");
    return false;
  }
  return true;

}

//==========================================================================

} // end namespace Pythia8
