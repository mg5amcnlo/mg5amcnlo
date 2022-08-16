// VinciaAntennaFunctions.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the AntennaSet,
// AntennaFunction, and MECs classes for both FSR and ISR, and some
// related global functions.

#include "Pythia8/VinciaAntennaFunctions.h"

namespace Pythia8 {

using namespace VinciaConstants;

//==========================================================================

// DGLAP splitting functions.

//--------------------------------------------------------------------------

// DGLAP splitting kernels with helicity-dependence (from Larkoski,
// Peskin, http://arxiv.org/abs/arXiv:0908.2450). Note, Pg2gg is
// written with the standard factor 2 normalization convention and
// gives a factor 2 difference with respect PYTHIA FSR. The
// difference is related to whether P is interpreted as the
// probability for gluon A to branch to BC (without factor 2), or as
// the probability for gluon B to have originated from A (with factor
// 2). Both quarks and gluons have definite helicities:
//    9: unpolarized
//   +1: right-handed
//   -1: left-handed
// where z is momentum fraction taken by B in A -> BC.

//--------------------------------------------------------------------------

// Pg2gg, written with the standard factor 2 normalization convention.

double DGLAP::Pg2gg(double z, int hA, int hB, int hC) {

  // If A unpolarized, treat as unpolarized.
  if (hA == 9) return 2*pow2(1. - z*(1. - z))/z/(1. - z);

  // Expressions coded for hA = 1. Flip helicities if hA = -1.
  if (hA == -1) {hB *= -1; hC *= -1;}
  if (hB == 1 && hC == 1) return 1./z/(1. - z);
  else if (hB == -1 && hC == 1) return pow3(1. - z)/z;
  else if (hB == 1 && hC == -1) return pow3(z)/(1. - z);
  return 0.;

}

//--------------------------------------------------------------------------

// Pg2qq, B is quark, C is antiquark. Note, mass corrections not
// implemented.

double DGLAP::Pg2qq(double z, int hA, int hB, int hC, double mu2) {

  // If A unpolarized, treat as unpolarized.
  if (hA == 9) return pow2(z) + pow2(1. - z) + 2.0*mu2;

  // Preserve quark helicity.
  if (hB != -hC || abs(hB) != 1) return 0.;

  // Expressions coded for hA = 1. Flip helicities if hA = -1.
  if (hA == -1) {hB *= -1; hC *= -1;}
  if (hB == -1 && hC == 1) return pow2(1. - z);
  else if (hB == 1 && hC == -1) return pow2(z);
  return 0.;

}

//--------------------------------------------------------------------------

// Pq2qg, B is quark, C is gluon. Note, mass corrections not
// implemented for polarised.

double DGLAP::Pq2qg(double z, int hA, int hB, int hC, double mu2) {

  // If A unpolarized, treat as unpolarized.
  if (hA == 9) return (1. + pow2(z))/(1. - z) - 2.0*mu2;

  // Preserve quark helicity
  if (hA != hB || abs(hB) != 1) return 0.;

  // Expressions coded for hA = +1. Flip helicities if hA = -1.
  if (hA == -1) {hB *= -1; hC *= -1;}
  if (hB == 1 && hC == -1) return pow2(z)/(1. - z);
  else if (hB == 1 && hC == 1) return 1./(1. - z);
  return 0.;

}

//--------------------------------------------------------------------------

// Pq2qg, B is gluon, C is quark. Note, mass corrections not
// implemented for polarised.

double DGLAP::Pq2gq(double z, int hA, int hB, int hC, double mu) {
  return Pq2qg(1-z,hA,hC,hB,mu);}

//--------------------------------------------------------------------------

// DGLAP splitting kernels with polarization-dependence (from Ellis,
// Stirling, Webber) Gluons are plane polarized, quarks have
// helicities:
//   9: unpolarized
//  +1: in plane / right-handed
//  -1: out-of-plane / left-handed
// where z is momentum fraction taken by B in A -> BC.

//--------------------------------------------------------------------------

// Pg2ggLin, written without factor 2 normalization convention.

double DGLAP::Pg2ggLin(double z, int polA, int polB, int polC) {

  // If A unpolarized, treat as unpolarized.
  if (polA == 9) return (1. - z + pow2(z))/z/(1 - z);

  // A in-plane polarized.
  else if (polA == 1) {
    if (polB == 1 && polC == 1) return (1. - z)/z + z/(1. - z) + z*(1. - z);
    else if (polB == -1 && polC == -1) return z*(1. - z);
  }

  // A out-of-plane polarized.
  else if (polA == -1) {
    if (polB == 1 && polC == -1) return (1. - z)/z;
    else if (polB == -1 && polC == 1) return z/(1. - z);
  }
  return 0.;

}

//--------------------------------------------------------------------------

// Pg2qqLin, B is quark, C is antiquark. Note, mass corrections not
// implemented.

double DGLAP::Pg2qqLin(double z, int polA, int hB, int hC, double mu) {

  // If A unpolarized, treat as unpolarized.
  if (polA == 9) return Pg2qq(z,9,9,9,mu);

  // Preserve quark helicity.
  if (hB != -hC || abs(hB) != 1) return 0.;

  // A in-plane polarized.
  else if (polA == 1) return pow2(1. - 2*z);

  // A out-of-plane polarized.
  else if (polA == -1) return 1.;
  return 0.;

}

//--------------------------------------------------------------------------

// Pq2qgLin, B is quark, C is gluon. Note, mass corrections not
// implemented.

double DGLAP::Pq2qgLin(double z, int hA, int hB, int polC, double mu) {

  // If A unpolarized, treat as unpolarized.
  if (hA == 9) return Pq2qg(z, 9, 9, 9, mu);

  // Preserve quark helicity.
  if (hA != hB || abs(hB) != 1) return 0.;

  // Gluon polarized in plane.
  else if (polC == 1) return pow2(1. + z)/(1. - z);

  // Gluon polarized out of plane.
  else if (polC == -1) return 1. - z;
  return 0.;

}

//--------------------------------------------------------------------------

// Pq2qgLin, B is gluon, C is quark. Note, mass corrections not
// implemented.

double DGLAP::Pq2gqLin(double z, int hA, int polB, int hC, double mu) {
  return Pq2qgLin(1 - z, hA, hC, polB, mu);}

//==========================================================================

// The AntennaFunction base class.

//--------------------------------------------------------------------------

// Default initialization.

bool AntennaFunction::init() {

  // Check whether pointers are initialized.
  if (!isInitPtr) return false;

  // Verbosity level.
  verbose = settingsPtr->mode("Vincia:verbose");
  // Charge factor
  // (Use same charge factor for AntQGemit and AntGQemit)
  if (vinciaName() == "Vincia:GQemitFF")
    chargeFacSav = settingsPtr->parm("Vincia:QGemitFF:chargeFactor");
  else
    chargeFacSav = settingsPtr->parm(vinciaName() + ":chargeFactor");
  if (chargeFacSav < 0.) chargeFacSav = 0.0;

  // Subleading-colour treatment.
  // modeSLC = 0: all gluon-emission antennae normalised to CA.
  // modeSLC = 1: use colour factors as specified by user.
  // modeSLC = 2: QQ gets CF, GG gets CA, QG gets interpolation.
  modeSLC = settingsPtr->mode("Vincia:modeSLC");
  if (modeSLC == 0 && id1() == 21) chargeFacSav = CA;
  if (modeSLC == 2 && id1() == 21) {
    if (idA() == 21 && idB() == 21) chargeFacSav = CA;
    else if (idA() == 21 || idB() == 21) chargeFacSav = (CA + CF)/2.;
    else chargeFacSav = CF;
  }

  // Kinematics map type (first look for specific, else default to global).
  if (settingsPtr->isMode(vinciaName() + ":kineMap"))
    kineMapSav = settingsPtr->mode(vinciaName()+":kineMap");
  else {
    // Gluon emission antennae.
    if (id1() == 21) {
      kineMapSav = settingsPtr->mode("Vincia:kineMapFFemit");

    // Gluon splitting antennae.
    } else {
      kineMapSav = settingsPtr->mode("Vincia:kineMapFFsplit");
      // For gluon splittings, parton I is always emitter, K is recoiler.
      if (kineMapSav == 2) kineMapSav = -1;
    }
  }

  // Sector shower on/off and sectorDamp parameter.
  sectorShower  = settingsPtr->flag("Vincia:sectorShower");
  sectorDampSav = settingsPtr->parm("Vincia:sectorDamp");

  // Collinear partitioning (for global antennae).
  // Note: forced to 1 for sector antennae.
  if (!sectorShower) alphaSav = settingsPtr->parm("Vincia:octetPartitioning");
  else alphaSav = 1.0;

  // Return.
  isInit = true;
  return isInit;

}

//--------------------------------------------------------------------------

// Method to initialise internal helicity variables. Return value =
// number of helicity configurations to average over.

int AntennaFunction::initHel(vector<int>* helBef, vector<int>* helNew) {

  // Initialise as unpolarised.
  hA = 9; hB = 9; hi = 9; hj = 9; hk = 9;

  // Check if one or more partons have explicit helicities.
  if (helNew->size() >= 3) {
    hi = helNew->at(0); hj = helNew->at(1); hk = helNew->at(2);
  }
  if (helBef->size() >= 2) {
    hA = helBef->at(0); hB = helBef->at(1);
  }

  // Check if helicity assignments make sense.
  bool physHel = true;
  if (hA != 1 && hA != -1 && hA != 9) physHel = false;
  if (hB != 1 && hB != -1 && hB != 9) physHel = false;
  if (hi != 1 && hi != -1 && hi != 9) physHel = false;
  if (hj != 1 && hj != -1 && hj != 9) physHel = false;
  if (hk != 1 && hk != -1 && hk != 9) physHel = false;
  if (!physHel) {
    if (verbose >= NORMAL) {
      stringstream ss;
      ss << hA << " " << hB << " -> " << hi << " " << hj << " " << hk;
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__+
        ": unphysical helicity configuration.",ss.str());
    }
    return 0;
  }

  // How many configurations are we averaging over.
  int nAvg = 1;
  if (hA == 9) nAvg *= 2;
  if (hB == 9) nAvg *= 2;
  return nAvg;

}

//--------------------------------------------------------------------------

// Generic check of antenna function for use during initialisation.

bool AntennaFunction::check() {

  // Check soft singularity against massless dipole factor.
  bool isOK = true;
  if (id1() == 21 || id1() == 22) {
    vector<double> invariants;

    // Test one symmetric and one asymmetric soft phase-space point.
    for (int iTest = 0; iTest <= 1; ++iTest) {
      // Test invariants.
      double yij = 1.0e-3 * pow(10.0,iTest);
      double yjk = 1.0e-3 / pow(10.0,iTest);
      // sIK is arbitrary, going to divide out in antFun anyway,
      // inclusion is just for consistent notation.
      double sIK = 1.0;
      invariants.resize(3);
      invariants[0] = sIK;
      invariants[1] = yij*sIK;
      invariants[2] = yjk*sIK;
      // Compute eikonal.
      double eik = 2.*(1. - yij - yjk)/yij/yjk*(1./sIK);
      // Compute antenna (without subleading-colour corrections).
      int modeSLCsave = modeSLC;
      modeSLC = 1;
      double ant = antFun(invariants);
      modeSLC = modeSLCsave;
      // Compare.
      double ratio = ant/eik;
      if (abs(ratio - 1.) >= 0.001) {
        isOK = false;
        if (verbose >= QUIET) {
          infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": Failed eikonal.",
            "("+num2str(iTest,1) + ")");
        }
      } else if (verbose >= DEBUG) printOut(__METHOD_NAME__,
        vinciaName() + " OK (eikonal " + num2str(iTest, 1) + ")");
    }
  }

  // Test all helicity configurations for collinearity and posivity.
  string helString = " 9 9 -> 9 9 9";
  for (int iHel = -1; iHel < 32; ++iHel) {
    if (iHel >= 0) {
      hj = 1 - 2*(iHel%2);
      hi = 1 - 2*((iHel/2)%2);
      hk = 1 - 2*((iHel/4)%2);
      hA = 1 - 2*((iHel/8)%2);
      hB = 1 - 2*((iHel/16)%2);
      helString = num2str(hA,2) + num2str(hB,2) + " ->"
        + num2str(hi,2) + num2str(hj,2) + num2str(hk,2);
    }
    else {hA = 9; hB = 9; hi = 9; hj = 9; hk = 9;}
    vector<int> helBef, helNew;
    helBef.push_back(hA);
    helBef.push_back(hB);
    helNew.push_back(hi);
    helNew.push_back(hj);
    helNew.push_back(hk);

    // Check collinear singularities against massless AP kernels.
    for (int iTest = 0; iTest <= 4; ++iTest) {
      // Test invariants, for a few points along collinear axis.
      double y1 = 1.0e-5;
      double y2 = 0.1 + iTest*0.2;
      vector<double> invariants1;

      // Test collinear limit on side 1.
      invariants1.resize(3);
      invariants1[0] = 1.;
      invariants1[1] = y1;
      invariants1[2] = y2;

      // Test collinear limit on side 2 (swap invariants).
      vector<double> invariants2 = invariants1;
      invariants2[1] = y2;
      invariants2[2] = y1;

      // Compute DGLAP kernels.
      double AP1  = AltarelliParisi(invariants1, mDum, helBef, helNew);
      double AP2  = AltarelliParisi(invariants2, mDum, helBef, helNew);

      // For g->qq antennae, there is only either an ij or jk collinear term.
      if (abs(id1()) <= 9) {
        // Keep whichever is larger (more singular), kill the other
        if (AP1 > AP2) AP2 = -1;
        else AP1 = -1;
      }
      // There is only an IJ singularity if hB is conserved.
      if (hB != hk) AP1 = -1.;
      if (hA != hi) AP2 = -1.;
      // Gluon emission: no external helicity flips for global antennae.
      if (!sectorShower) {
        if (id1() == 21) {
          if (hA != hi || hB != hk) {
            AP1 = -1.;
            AP2 = -1.;
          }
        }
      }
      // Are there any limits to check?
      if (AP1 < 0. && AP2 < 0.) continue;

      // Compute antennae (without subleading colour corrections).
      int modeSLCsave = modeSLC;
      modeSLC = 1;
      double ant1 = antFun(invariants1,mDum,helBef,helNew);
      double ant2 = antFun(invariants2,mDum,helBef,helNew);
      // For global antennae, add "swapped" terms to represent neighbor.
      if (!sectorShower && idA() == 21) {
        invariants1[1] = y1;
        invariants1[2] = 1. - y1 -y2;
        vector<int> helComp;
        helComp.push_back(hj);
        helComp.push_back(hi);
        helComp.push_back(hk);
        ant1 += antFun(invariants1, mDum, helBef, helComp);
      }
      if (!sectorShower && idB() == 21) {
        invariants2[1] = 1 - y1 - y2;
        invariants2[2] = y1;
        vector<int> helComp;
        helComp.push_back(hi);
        helComp.push_back(hk);
        helComp.push_back(hj);
        ant2 += antFun(invariants2, mDum, helBef, helComp);
      }
      // Restore subleading-colour level.
      modeSLC = modeSLCsave;

      // Compare.
      if (AP1 > 0.) {
        double ratio = ant1/AP1;
        // Require better than 5% agreement unless dominated by nonsingular.
        if (abs(ratio-1.) >= 0.05 && abs(ant1 - AP1) > 10.) {
          isOK = false;
          if (verbose >= NORMAL) {
            printOut(__METHOD_NAME__, "WARNING:" + vinciaName() +
             " FAILED (collinear ij " + num2str(iTest,1) + " " +
             helString+" )");
          }
          if (verbose >= REPORT) {
            cout << setprecision(6);
            printOut(__METHOD_NAME__, "    ant  = " + num2str(ant1, 9) +
              " y1 = " + num2str(y1, 9) +" y2 = " + num2str(y2, 9));
            printOut(__METHOD_NAME__, "    P(z) = "+num2str(AP1, 9) +
              "  zi = " + num2str(zA(invariants1), 9));
          }
        } else if (verbose >= DEBUG) {
          printOut(__METHOD_NAME__, vinciaName() + " OK (collinear ij " +
            num2str(iTest, 1) + " " + helString + " )");
        }
      }

      // Require better than 5% agreement unless dominated by nonsingular.
      if (AP2 > 0.) {
        double ratio = ant2/AP2;
        if (abs(ratio - 1.) >= 0.05 && abs(ant2 - AP2) > 10.) {
          isOK = false;
          if (verbose >= QUIET) {
            printOut(__METHOD_NAME__, "WARNING:" + vinciaName() +
              " FAILED (collinear jk " + num2str(iTest, 1) + " " +
              helString + " )");
          }
          if (verbose >= REPORT) {
            cout << setprecision(6);
            printOut(__METHOD_NAME__, "    ant  = " + num2str(ant2, 9) +
              " y1 = "+num2str(y2, 9) + " y2 = " + num2str(y1, 9));
            printOut(__METHOD_NAME__, "    P(z) = " + num2str(AP2, 9) +
              "  zk = " + num2str(zB(invariants1), 9));
          }
        } else if (verbose >= DEBUG) {
          printOut(__METHOD_NAME__, vinciaName() + " OK (collinear jk "
            + num2str(iTest, 1) + " " + helString+" )");
        }
      }
    }

    // Check positivity.
    int nPointsCheck = settingsPtr->mode("Vincia:nPointsCheck");
    bool isPositive  = true;
    bool isZero      = true;
    bool hasDeadZone = false;
    double deadZoneAvoidance = settingsPtr->parm("Vincia:deadZoneAvoidance");
    for (int iTest = 0; iTest < nPointsCheck; ++iTest) {

      // Generate a random point in phase-space triangle.
      double y1 = rndmPtr->flat();
      double y2 = rndmPtr->flat();
      if (y1 + y2 > 1.0) {
        y1 = 1 - y1;
        y2 = 1 - y2;
      }
      vector<double> invariants;
      invariants.resize(3);
      invariants[0] = 1.;
      invariants[1] = y1;
      invariants[2] = y2;
      double ant = antFun(invariants,mDum,helBef,helNew);

      // Check positivity (strict).
      if (ant < 0.) {
        isPositive = false;
        if (verbose >= REPORT) {
          printOut(__METHOD_NAME__, "ERROR:" + vinciaName() + " ant("
            + num2str(y1, 9) + "," + num2str(y2, 9) + " ; " + helString
            + ") = " + num2str(ant) + " < 0");
        }
      } else if (ant > 0.) isZero = false;
      // Check for dead zones away from phase-space boundary.
      if (1-y1-y2 > 0.05 && y1 > 0.05 && y2 > 0.05
          && ant < deadZoneAvoidance) hasDeadZone = true;

    } // End nPointsCheck sum for positivity checks.
    isOK = isOK && isPositive;

    // Verbose output.
    if (!isPositive && verbose >= QUIET)
      printOut(__METHOD_NAME__, "ERROR:" + vinciaName() +
        " (ant < 0 encountered " + helString + " )");
    else if (isPositive && !isZero && verbose >= DEBUG)
      printOut(__METHOD_NAME__, vinciaName() + " OK (is positive "
        + helString + " )");
    if (!isZero) {
      if (hasDeadZone) {
        if ( (!sectorShower && verbose >= QUIET)
          || (sectorShower && verbose >= DEBUG) )
        printOut(__METHOD_NAME__, "WARNING:" + vinciaName()
          + " (dead zone encountered " + helString+" )");
      }
      else if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, vinciaName() + " OK (no dead zones "
          + helString + " )");
    }
  } // End helicity sum.
  return isOK;

}

//--------------------------------------------------------------------------

// Initialize pointers. ParticleData: required for masses, colour
// charges, etc. Rndm: required to generate random test points to
// check antenna. Settings: required for all coefficients, switches,
// etc.

void AntennaFunction::initPtr(Info* infoPtrIn, DGLAP* dglapPtrIn) {

  infoPtr      = infoPtrIn;
  particleDataPtr = infoPtr->particleDataPtr;
  settingsPtr     = infoPtr->settingsPtr;
  rndmPtr         = infoPtr->rndmPtr;
  dglapPtr        = dglapPtrIn;
  isInitPtr       = true;
  isInit          = false;

}

//--------------------------------------------------------------------------

// Auxiliary function to translate an ID code to a string, for
// constructing antenna names.

string AntennaFunction::id2str(int id) const {

  // Vector bosons.
  if (id == 21) return "g";
  else if (id == 22) return "gamma";
  else if (id == 23) return "Z";
  else if (abs(id) == 24) return "W";

  // Fermions.
  else if (id >= 1 && id <= 4) return "q";
  else if (-id >= 1 && -id <= 4) return "qbar";
  else if (id == 5) return "b";
  else if (id == -5) return "bbar";
  else if (id == 6) return "t";
  else if (id == -6) return "tbar";
  else if (id >= 11 && id <= 20 && id%2 == 1) return "l-";
  else if (id >= 11 && id <= 20 && id%2 == 0) return "nu";
  else if (-id >= 11 && -id <= 20 && id%2 == 1) return "l+";
  else if (-id >= 11 && -id <= 20 && id%2 == 0) return "nubar";

  // Octet fermion: use gluino.
  else if (id == 1000021) return "~g";
  // Charged scalar: use H+.
  else if (id == 37) return "H+";
  else if (id == -37) return "H-";
  // Coloured scalars: use squarks.
  else if (id >= 1000000 && id <= 1000010) return "~q";
  else if (-id >= 1000000 && -id <= 1000010) return "~q*";
  // Unknown particle.
  else return "X";

}

//==========================================================================

// Class AntQQemitFF, final-final antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntQQemitFF::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Make sure we have enough invariants.
  if (invariants.size() <= 2) return 0.;
  double sIK = invariants[0];
  double sij = invariants[1];
  double sjk = invariants[2];

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef, &helNew);
  if (nAvg <= 0) return 0.;

  // Check helicity conservation for massless partons.
  if (mi <= 0. && hi == -hA) return 0.0;
  else if (mk <= 0. && hk == -hB) return 0.0;

  // Dimensionless parameters: yij = 2pi.pj / sIK.
  double yij = sij/sIK;
  double yjk = sjk/sIK;

  // Shorthands.
  double eik    = 1./yij/yjk;
  double mTermI = (mi > 0.) ? pow2(mi)/sij/yij : 0.0;
  double mTermK = (mk > 0.) ? pow2(mk)/sjk/yjk : 0.0;

  // Do helicity sum.
  double hSum(0.);

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {

    // ++ > +++ && -- > --- antennae (MHV).
    term = eik - mTermI/(1. - yjk) - mTermK/(1. - yij);
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // ++ > +-+ && -- > -+- antennae (NMHV).
    term = eik * pow2(1. - yij - yjk) - mTermI*(1. - yjk) - mTermK*(1. - yij);
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > -++ && -- > +-- antennae (mI != 0).
    if (mi != 0) {
      term = mTermI * pow2(yjk)/(1. - yjk);
      if (RH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    }

    // ++ > ++- && -- > --+ antennae (mK != 0).
    if (mk != 0) {
      term = mTermK * pow2(yij)/(1. - yij);
      if (RH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    }
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {

    // +- > ++- && -+ > --+ antennae.
    term = eik * pow2(1. - yij) - mTermI/(1. - yjk) - mTermK*(1. - yij);
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae.
    term = eik * pow2(1. - yjk) - mTermI*(1. - yjk) - mTermK/(1. - yij);
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > -+- && -+ > +-+ antennae (mI != 0).
    if (mi != 0) {
      term = mTermI * pow2(yjk)/(1. - yjk);
      if (RH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    }

    // +- > +-+ && -+ > -+- antennae (mK != 0).
    if (mk != 0) {
      term = mTermK * pow2(yij)/(1. - yij);
      if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;
    }
  }

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg/sIK;

}

//--------------------------------------------------------------------------

// Function to give Altarelli-Parisi limits of this antenna.
// Defined as PI/sij + PK/sjk, i.e. equivalent to antennae.

double AntQQemitFF::AltarelliParisi(vector<double> invariants,
  vector<double>, vector<int> helBef, vector<int> helNew) {

  int h0Now = helNew[0];
  int h1Now = helNew[1];
  int h2Now = helNew[2];
  int hANow = helBef[0];
  int hBNow = helBef[1];

  // Compute (sum of) DGLAP kernel(s)/Q^2
  if (hANow != h0Now || hBNow != h2Now) return -1.;
  else return dglapPtr->Pq2qg(zA(invariants),hANow,h0Now,h1Now)/invariants[1]
         + dglapPtr->Pq2qg(zB(invariants),hBNow,h2Now,h1Now)/invariants[2];

}

//==========================================================================

// Class AntQGemitFF, final-final antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntQGemitFF::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Make sure we have enough invariants.
  if (invariants.size() <= 2) return 0.;
  double sIK = invariants[0];
  double sij = invariants[1];
  double sjk = invariants[2];

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef, &helNew);
  if (nAvg <= 0) return 0.;

  // Check helicity conservation for massless partons.
  if (mi <= 0. && hi == -hA) return 0.0;
  else if (hk == -hB) return 0.0;

  // Dimensionless parameters: yij = 2pi.pj / sIK.
  double yij = sij/sIK;
  double yjk = sjk/sIK;

  // Shorthands.
  double yik   = max(0., 1. - yij - yjk);
  double eik   = 1./yij/yjk;
  double mTerm = pow2(mi)/sij/yij;
  double a     = 1. - alphaSav;
  if (alphaSav == 0.0) a = 1.;
  else if (alphaSav == 1.0) a = 0.;

  // Do helicity sum.
  double hSum(0.);

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {

    // ++ > +++ && -- > --- antennae (MHV).
    term = eik - mTerm/(1. - yjk);
    if (a != 0.) term += a * (1. - yjk) * ( 1. - 2*yij - yjk )/yjk;
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // ++ > +-+ && -- > -+- antennae (NMHV).
    term = eik * pow2(yik) * (1. - yij) - mTerm*(1. - yjk);
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > -++ && -- > +-- antennae (mI != 0).
    if (mi != 0) {
      term = mTerm*pow2(yjk)/(1. - yjk);
      if (RH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    }
  }

  // (+- and -+) parents
  if (hA * hB < 0 || hA == 9 || hB == 9) {

    // +- > ++- && -+ > --+ antennae.
    term = eik * pow3(1. - yij) - mTerm/(1. - yjk);
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae.
    term = eik * pow2(1. - yjk) - mTerm*(1. - yjk);
    if (a != 0.) term += a * (1. - yjk) * ( 1. - 2*yij - yjk )/yjk;
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > -+- && -+ > +-+ antennae (mI != 0).
    if (mi != 0.) {
      term = mTerm * pow2(yjk)/(1. - yjk);
      if (RH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    }
  }

  // Subleading colour correction.
  // Note: has to be applied to sector antennae after symmetrisation!
  if (!sectorShower && modeSLC >= 2)
    hSum *= CF/chargeFacSav * (1 - yij)/(2 - yij - yjk)
      + CA/chargeFacSav * (1 - yjk)/(2 - yij - yjk);

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg/sIK;

}

//--------------------------------------------------------------------------

// Function to give Altarelli-Parisi limits of this antenna.

double AntQGemitFF::AltarelliParisi(vector<double> invariants,
  vector<double>, vector<int> helBef, vector<int> helNew) {

  int h0Now = helNew[0];
  int h1Now = helNew[1];
  int h2Now = helNew[2];
  int hANow = helBef[0];
  int hBNow = helBef[1];
  if (h0Now != hANow) return -1;

  // Compute (sum of) DGLAP kernel(s)/Q^2
  double sum(0.);
  if (h2Now == hBNow)
    sum += dglapPtr->Pq2qg(zA(invariants), hANow, h0Now, h1Now)/invariants[1];
  return sum + dglapPtr->Pg2gg(zB(invariants), hBNow, h2Now, h1Now)
    /invariants[2];

}

//==========================================================================

// Class AntGQemitFF, final-final antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2] (derived from AntQGemit by swapping).

double AntGQemitFF::antFun(vector<double> invariants,
  vector<double> mNew, vector<int> helBef, vector<int> helNew) {

  swap(invariants[1], invariants[2]);
  swap(mNew[0], mNew[2]);
  swap(helBef[0], helBef[1]);
  swap(helNew[0], helNew[2]);
  return AntQGemitFF::antFun(invariants, mNew, helBef, helNew);

}

//--------------------------------------------------------------------------

// Function to give Altarelli-Parisi limits of this antenna.

double AntGQemitFF::AltarelliParisi(vector<double> invariants,
  vector<double>, vector<int> helBef, vector<int> helNew) {

  int h0Now = helNew[0];
  int h1Now = helNew[1];
  int h2Now = helNew[2];
  int hANow = helBef[0];
  int hBNow = helBef[1];
  if (h2Now != hBNow) return -1;

  // Compute (sum of) DGLAP kernel(s)/Q^2
  double sum(0.);
  if (h0Now == hANow)
    sum += dglapPtr->Pq2qg(zB(invariants), hBNow, h2Now, h1Now)/invariants[2];
  return sum + dglapPtr->Pg2gg(zA(invariants), hANow, h0Now, h1Now)
    /invariants[1];

}

//==========================================================================

// Class AntGGemitFF, final-final antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntGGemitFF::antFun(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  // Make sure we have enough invariants.
  if (invariants.size() <= 2) return 0.;
  double sIK = invariants[0];
  double sij = invariants[1];
  double sjk = invariants[2];

  // Dimensionless parameters: yij = 2pi.pj / sIK.
  double yij = sij/sIK;
  double yjk = sjk/sIK;

  // Initialise helicities. Return 0 for unphysical helicities.
  int nAvg = initHel(&helBef, &helNew);
  if (nAvg <= 0) return 0.;

  // Check physical helicity combination.
  if (hi == -hA) return 0.0;
  else if (hk == -hB) return 0.0;

  // Shorthands.
  double eik = 1./yij/yjk;
  double yik = max(0.,1. - yij - yjk);
  double a;
  if (alphaSav == 0.0) a = 1.;
  else if (alphaSav == 1.0) a = 0.;
  else a = 1. - alphaSav;

  // Do helicity sum.
  double hSum(0.);

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {

    // ++ > +++ && -- > --- antennae (MHV).
    term = eik;
    if (a != 0.0) term += a * ((1. - yjk)*(1. - 2*yij - yjk)/yjk
      + (1. - yij)*(1. - 2*yjk - yij)/yij);
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // ++ > +-+ && -- > -+- antennae (NMHV).
    term = eik * pow3(yik);
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {

    // +- > ++- && -+ > --+ antennae.
    term = eik * pow3(1 - yij);
    if (a != 0.) term += a * (1. - yij)*(1. - 2*yjk)/yij;
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae.
    term = eik * pow3(1 - yjk);
    if (a != 0.) term += a * (1. - yjk)*(1. - 2*yij)/yjk;
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;
  }

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg/sIK;

}

//--------------------------------------------------------------------------

// Function to give Altarelli-Parisi limits of this antenna.

double AntGGemitFF::AltarelliParisi(vector<double> invariants,
  vector<double>, vector<int> helBef, vector<int> helNew) {

  int h0Now = helNew[0];
  int h1Now = helNew[1];
  int h2Now = helNew[2];
  int hANow = helBef[0];
  int hBNow = helBef[1];
  double sum(0.);

  // Compute (sum of) DGLAP kernel(s)/Q^2
  if (hBNow == h2Now)
    sum += dglapPtr->Pg2gg(zA(invariants), hANow, h0Now, h1Now)/invariants[1];
  if (hANow == h0Now)
    sum += dglapPtr->Pg2gg(zB(invariants), hBNow, h2Now, h1Now)/invariants[2];
  return sum;

}

//==========================================================================

// Class AntGXsplitFF, final-final antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntGXsplitFF::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Make sure we have enough invariants.
  if (invariants.size() <= 2) return 0.;
  double sIK = invariants[0];
  double sij = invariants[1];
  double sjk = invariants[2];

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef,&helNew);
  if (nAvg <= 0) return 0.;

  // Dimensionless parameters: yij = 2pi.pj / sIK.
  double yij = sij/sIK;
  double yjk = sjk/sIK;
  double yik = 1. - yij - yjk - pow2(mi)/sIK - pow2(mj)/sIK;

  // Check phase space ok.
  if (yij <= 0.0 || yjk <= 0.0 || yik <= 0.0) return 0.0;

  // Shorthands.
  double mu2q  = mj*mi/sIK;
  double mu2ij = yij + 2 * mu2q;
  double termQ = 0.5 * (pow2(yik) - mu2q/mu2ij * yik/(1. - yik)) / mu2ij;
  double termA = 0.5 * (pow2(yjk) - mu2q/mu2ij * yjk/(1. - yjk)) / mu2ij;
  double termF = (mu2q <= 0.) ? 0
    : 0.5 * mu2q/pow2(mu2ij)*(yik/(1. - yik) + yjk/(1. - yjk) + 2.);

  // Do helicity sum.
  double hSum(0.);

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {

    // Quark (i) takes gluon helicity: ++ > +-+ && -- > -+- antennae.
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += termQ;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += termQ;

    // Antiquark (j) takes gluon helicity: ++ > -++ && -- > +-- antennae.
    if (RH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += termA;
    if (LH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += termA;

    // Splitting with helicity flip (mQ != 0): ++ > +++ && -- > ---.
    if (mu2q > 0.) {
      if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += termF;
      if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += termF;
    }
  }

  // (+- and -+) parents.
  if ( hA * hB < 0 || hA == 9 || hB == 9) {

    // Quark (i) takes gluon helicity: +- > +-- && -+ > -++ antennae.
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += termQ;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += termQ;

    // Antiquark (j) takes gluon helicity: +- > -+- && -+ > +-+ antennae.
    if (RH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += termA;
    if (LH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += termA;

    // Splitting with helicity flip (mQ != 0): +- > ++- && -+ > --+.
    if (mu2q > 0.) {
      if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += termF;
      if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += termF;
    }
  }

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg/sIK;

}

//--------------------------------------------------------------------------

// Function to give Altarelli-Parisi limits of this antenna.

double AntGXsplitFF::AltarelliParisi(vector<double> invariants,
  vector<double>, vector<int> helBef, vector<int> helNew) {

  int h0Now = helNew[0];
  int h1Now = helNew[1];
  int h2Now = helNew[2];
  int hANow = helBef[0];
  int hBNow = helBef[1];

  // Compute DGLAP kernel/Q^2
  return hBNow != h2Now ? 0.0 : dglapPtr->Pg2qq(zA(invariants), hANow, h0Now,
    h1Now)/invariants[1];

}

//==========================================================================

// Class AntQGemitFFsec, sector final-final antenna function, explicit
// symmetrisation of AntQGemitFF.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntQGemitFFsec::antFun(vector<double> invariants,
  vector<double> mNew, vector<int> helBef, vector<int> helNew) {

  // Check if helicity vectors empty.
  double ant = AntQGemitFF::antFun(invariants, mNew, helBef, helNew);
  if (helBef.size() < 2) {helBef.push_back(9); helBef.push_back(9);}
  if (helNew.size() < 3) {
    helNew.push_back(9); helNew.push_back(9); helNew.push_back(9);}

  // Save invariants.
  double sIK = invariants[0];
  double sij = invariants[1];
  double sjk = invariants[2];
  double sik = sIK - sij - sjk;
  double yij = sij/sIK;
  double yjk = sjk/sIK;

  // Check if j has same helicity as parent gluon.
  int hG = helBef[1];
  int hjNow = helNew[1];
  if ( hG == hjNow || hjNow == 9) {
    // Define j<->k symmetrisation term with sector damp parameter;
    sik += sectorDampSav * sjk;
    vector<double> invariantsSym = {sIK, sik, sjk};
    // Swap helicities.
    vector<int> helSym = helNew;
    helSym[1] = helNew[2];
    helSym[2] = helNew[1];
    ant += AntQGemitFF::antFun(invariantsSym, mNew, helBef, helSym);
  }

  // Subleading colour correction has to be applied after symmetrisation.
  if (modeSLC >= 2)
    ant *= CF/chargeFacSav * (1 - yij)/(2 - yij - yjk)
      + CA/chargeFacSav * (1 - yjk)/(2 - yij - yjk);

  return ant;
}

//==========================================================================

// Class AntGQemitFFsec, sector final-final antenna function, explicit
// symmetrisation of AntGQemitFF.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2] (derived from AntQGemitFFsec by swapping).

double AntGQemitFFsec::antFun(vector<double> invariants,
  vector<double> mNew, vector<int> helBef, vector<int> helNew) {

  swap(invariants[1], invariants[2]);
  swap(mNew[0], mNew[2]);
  swap(helBef[0], helBef[1]);
  swap(helNew[0], helNew[2]);
  return AntQGemitFFsec::antFun(invariants, mNew, helBef, helNew);

}

//--------------------------------------------------------------------------

// Function to give Altarelli-Parisi limits of this antenna.

double AntGQemitFFsec::AltarelliParisi(vector<double> invariants,
  vector<double>, vector<int> helBef, vector<int> helNew) {

  int h0Now = helNew[0];
  int h1Now = helNew[1];
  int h2Now = helNew[2];
  int hANow = helBef[0];
  int hBNow = helBef[1];
  if (h2Now != hBNow) return -1;

  // Compute (sum of) DGLAP kernel(s)/Q^2.
  double sum(0.);
  if (h0Now == hANow)
    sum += dglapPtr->Pq2qg(zB(invariants), hBNow, h2Now, h1Now)/invariants[2];
  return sum + dglapPtr->Pg2gg(zA(invariants), hANow, h0Now, h1Now)
    /invariants[1];

}

//==========================================================================

// Class GGemitFFsec, sector final-final antenna function, explicit
// symmetrisation of AntGGemitFF.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntGGemitFFsec::antFun(vector<double> invariants, vector<double> mNew,
  vector<int> helBef, vector<int> helNew) {

  // Check if helicity vectors empty
  double ant = AntGGemitFF::antFun(invariants, mNew, helBef, helNew);
  if (helBef.size() < 2) {helBef.push_back(9); helBef.push_back(9);}
  if (helNew.size() < 3) {
    helNew.push_back(9); helNew.push_back(9); helNew.push_back(9);}

  // Check if j has same helicity as parent gluon 0.
  int hjNow = helNew[1];
  if (helBef[0] == hjNow) {
    // Define i<->j symmetrisation term.
    vector<double> invariantsSym = invariants;
    double s02 = invariants[0] - invariants[1] - invariants[2];
    vector<int> helSym = helNew;
    helSym[0] = helNew[1];
    helSym[1] = helNew[0];
    invariantsSym[2] = s02 + sectorDampSav * invariants[1];
    ant += AntGGemitFF::antFun(invariantsSym, mNew, helBef, helSym);
  }

  // Check if j has same helicity as parent gluon 1.
  if (helBef[1] == hjNow) {
    // Define j<->k symmetrisation term.
    vector<double> invariantsSym = invariants;
    double s02 = invariants[0] - invariants[1] - invariants[2];
    vector<int> helSym = helNew;
    helSym[1] = helNew[2];
    helSym[2] = helNew[1];
    invariantsSym[1] = s02 + sectorDampSav * invariants[2];
    ant += AntGGemitFF::antFun(invariantsSym, mNew, helBef, helSym);
  }
  return ant;

}

//==========================================================================

// Class AntGXsplitFFsec, sector final-final antenna function, explicit
// symmetrisation of AntGXsplitFF.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2] (just 2*global).

double AntGXsplitFFsec::antFun(vector<double> invariants, vector<double> mNew,
  vector<int> helBef, vector<int> helNew) {
  return 2*AntGXsplitFF::antFun(invariants,mNew,helBef,helNew);}

//==========================================================================

// AntennaFunctionIX class.

//--------------------------------------------------------------------------

// Method to initialise (can be different than that of the base class).

bool AntennaFunctionIX::init() {

  // Check whether pointers are initialized.
  if (!isInitPtr) return false;
  verbose      = settingsPtr->mode("Vincia:verbose");
  chargeFacSav = settingsPtr->parm(vinciaName()+":chargeFactor");
  if (chargeFacSav < 0.) chargeFacSav = 0.0;

  // Subleading-colour treatment.
  // modeSLC = 0: all gluon-emission antennae normalised to CA.
  // modeSLC = 1: use colour factors as specified by user.
  // modeSLC = 2: QQ gets CF, GG gets CA, QG gets interpolation.
  modeSLC = settingsPtr->mode("Vincia:modeSLC");
  if (modeSLC == 0 && id1() == 21) chargeFacSav = CA;
  if (modeSLC == 2 && id1() == 21) {
    if (idA() == 21 && idB() == 21 ) chargeFacSav = CA;
    else if (idA() == 21 || idB() == 21) chargeFacSav = (CA + CF)/2.;
    else chargeFacSav = CF;
  }

  // Sector shower on/off and sectorDamp parameter.
  sectorShower  = settingsPtr->flag("Vincia:sectorShower");
  sectorDampSav = settingsPtr->parm("Vincia:sectorDamp");

  // Collinear partitioning (for global antennae).
  if (!sectorShower) alphaSav = settingsPtr->parm("Vincia:octetPartitioning");
  else alphaSav = 1.;

  // Return OK.
  isInit = true;
  return isInit;

}

//--------------------------------------------------------------------------

// Function to check singularities, positivity, etc. For
// initial-initial: sab = sAB + saj + sjb.

bool AntennaFunctionIX::check() {

  // For check for positivity.
  bool isOK = true;
  int nPointsCheck = settingsPtr->mode("Vincia:nPointsCheck");
  double shhMax    = pow2(14000.0);
  double deadZoneAvoidance = settingsPtr->parm("Vincia:deadZoneAvoidance");

  // Z and Higgs as testing case.
  double sAB[2] = {pow2(91.0), pow2(125.0)};

  // Check soft singularity against dipole factor.
  if (id1() == 21 || id1() == 22) {
    // Test one symmetric and one asymmetric soft phase-space point.
    for (int iTest = 0; iTest <= 1; ++iTest) {
      // Test two sAB values.
      for (int isAB = 0; isAB <= 1; ++isAB) {
        // Test invariants.
        double yaj = 1.0e-3 * pow(10.0,iTest); // saj/sab
        double yjb = 1.0e-3 / pow(10.0,iTest); // sjb/sab
        double sab = sAB[isAB]/(1.0-yaj-yjb);  // 1 = sAB/sab + yaj + yjb.
        double saj = yaj*sab;
        double sjb = yjb*sab;

        // Compute eikonal.
        double eik = 2.0*sab/saj/sjb;

        // Compute antenna (without subleading-colour corrections).
        vector<double> invariants;
        invariants.push_back(sAB[isAB]);
        invariants.push_back(saj);
        invariants.push_back(sjb);
        int modeSLCsave = modeSLC;
        modeSLC = 1;
        double ant = antFun(invariants);
        modeSLC = modeSLCsave;

        // Compare.
        double ratio = ant/eik;
        if (abs(ratio - 1.0) >= 0.001) {
          isOK = false;
          if (verbose >= QUIET)
            printOut(__METHOD_NAME__,"WARNING:" + vinciaName() + " FAILED "
              "(eikonal " + num2str(iTest, 1) + " and sAB = " +
              num2str((int)sqrt(sAB[isAB])) + "^2)");
        } else if (verbose >= DEBUG) {
          printOut(__METHOD_NAME__,vinciaName() + " OK (eikonal "
            + num2str(iTest, 1) + " and sAB = "
            + num2str((int)sqrt(sAB[isAB])) + "^2)");
        }
      }
    }
  }

  // Test all helicity configurations for collinearity and posivity.
  string helString = " 9 9 -> 9 9 9";
  for (int iHel = -1; iHel < 32; ++iHel) {
    bool isPositive  = true;
    bool isZero      = true;
    bool hasDeadZone = false;
    if (iHel >= 0) {
      hj = 1 - 2*(iHel%2);
      hi = 1 - 2*((iHel/2)%2);
      hk = 1 - 2*((iHel/4)%2);
      hA = 1 - 2*((iHel/8)%2);
      hB = 1 - 2*((iHel/16)%2);
      helString = num2str(hA,2) + num2str(hB,2) + " ->" +
        num2str(hi,2) + num2str(hj,2) + num2str(hk,2);
    }
    vector<int> helBef = {hA, hB};
    vector<int> helNew = {hi, hj, hk};

    // Gluon emission.
    if (id1() == 21) {
      // Massless: quark helicities must be conserved.
      if (idA() != 21 && hi != hA) continue;
      if (idB() != 21 && hk != hB) continue;
      // For gluons, sector terms can give helicity flips
      if (idA() == 21 && (hi != hA && hj != -hA)) continue;
      if (idB() == 21 && (hk != hB && hj != -hB)) continue;
      // Do not allow helicity flips on both sides simultaneously
      if (hi != hA && hk != hB) continue;
    }
    // Gluon conversion.
    else if (idA() == 21) {
      // Helicity of recoiler must be conserved.
      if (hk != hB) continue;
      // Massless: quark helicity must be conserved.
      if (hi != hj) continue;
    }
    // Gluon splitting.
    else {
      // Helicity of recoiler must be conserved.
      if (hk != hB) continue;
      // Massless: quark helicity must be conserved.
      if (hA == hj) continue;
    }

    // Test invariants, for a few points along collinear axis.
    for (int iTest = 0; iTest <= 2; ++iTest) {
      // Test two sAB values.
      for (int isAB = 0; isAB <= 1; ++isAB) {
        // Test invariants.
        double y1  = 1.0e-5;                 // saj/sab
        double y2  = 0.2 + iTest*0.3;        // sjb/sab
        double sab = sAB[isAB]/(1.0-y1-y2);  // 1 = sAB/sab + yaj + yjb
        double s1  = y1*sab;
        double s2  = y2*sab;
        vector<double> invariants1 = {sAB[isAB], s1, s2};
        vector<double> invariants2 = {sAB[isAB], s2, s1};

        // Compute AP kernels.
        double AP1 = AltarelliParisi(invariants1,mDum,helBef,helNew);
        double AP2 = AltarelliParisi(invariants2,mDum,helBef,helNew);

        // Compute antennae (without subleading colour corrections).
        int modeSLCsave = modeSLC;
        modeSLC = 1;
        double ant1 = antFun(invariants1,mDum,helBef,helNew);
        double ant2 = antFun(invariants2,mDum,helBef,helNew);
        modeSLC = modeSLCsave;

        // Check if only a singularity on one side exists (-> z = -1).
        double zs1 = zA(invariants1);
        double zs2 = zB(invariants1);
        if (zs1 != -1.0) {
          double ratio = ant1/AP1;
          // Require better than 5% agreement unless dominated by non-singular.
          if (fabs(ratio - 1.0) >= 0.05 && fabs(ant1 - AP1) > 10.) {
            isOK = false;
            if (verbose >= NORMAL)
              printOut(__METHOD_NAME__,"WARNING: " + vinciaName() + " FAILED "
                + "(collinear aj " + num2str(iTest, 1) + " and sAB = " +
                num2str((int)sqrt(sAB[isAB])) + "^2)");
            if (verbose >= REPORT) {
              cout << setprecision(6);
              printOut(__METHOD_NAME__, "    (" + helString + ")");
              printOut(__METHOD_NAME__, "    ant  = " + num2str(ant1, 9)
                + " y1 = " + num2str(y1,9) + " y2 = " + num2str(y2, 9));
              printOut(__METHOD_NAME__, "    P(z) = " + num2str(AP1, 9)
                + "  z = " + num2str(zs1, 9));
            }
          } else if (verbose >= DEBUG)
            printOut(__METHOD_NAME__, vinciaName() + " OK (collinear aj "
              + num2str(iTest, 1) + " and sAB = "
              + num2str((int)sqrt(sAB[isAB])) + "^2)");
        }
        if (zs2 != -1.0) {
          double ratio = ant2/AP2;
          // Require better than 5% agreement unless dominated by non-singular.
          if (fabs(ratio - 1.0) >= 0.05 && abs(ant2 - AP2) > 10.) {
            isOK = false;
            if (verbose >= NORMAL)
              printOut(__METHOD_NAME__,"WARNING: " + vinciaName() + " FAILED "
                + "(collinear jb " + num2str(iTest, 1) + " and sAB = " +
                num2str((int)sqrt(sAB[isAB])) + "^2)");
            if (verbose >= REPORT) {
              cout << setprecision(6);
              printOut(__METHOD_NAME__, "    (" + helString + ")");
              printOut(__METHOD_NAME__, "    ant  = " + num2str(ant2, 9)
                + " y1 = " + num2str(y1,9) + " y2 = " + num2str(y2, 9));
              printOut(__METHOD_NAME__, "    P(z) = " + num2str(AP2, 9)
                + "  z = " + num2str(zs1, 9));
            }
          } else if (verbose >= DEBUG)
            printOut(__METHOD_NAME__, vinciaName() + " OK (collinear jb "
              + num2str(iTest, 1) + " and sAB = "
              + num2str((int)sqrt(sAB[isAB])) + "^2)");
        }
      }
    }

    // Check for positivity and dead zones.
    for (int iTest = 0; iTest <= nPointsCheck; ++iTest) {
      // sAB^0.5 between 1 and 1000 GeV.
      double sABnow = pow2(1.0+rndmPtr->flat()*999.0);

      // Generate a random point in phase-space triangle.
      double yaj = rndmPtr->flat(); // saj/sab
      double yjb = rndmPtr->flat(); // sjb/sab
      if (yaj + yjb > 1.0) {
        yaj = 1.0 - yaj;
        yjb = 1.0 - yjb;
      }

      // Compute sab from 1 = sAB/sab + yaj + yjb.
      double sab = sABnow/(1.0-yaj-yjb);

      // Check that sab < shhMax.
      if (sab > shhMax) continue;
      double saj = yaj*sab;
      double sjb = yjb*sab;
      vector<double> invariants = {sABnow, saj, sjb};

      // Compute antenna.
      double ant = antFun(invariants, mDum, helBef, helNew);

      // Check positivity (strict).
      if (ant <= 0.0) {
        isPositive = false;
        if (verbose >= REPORT)
          printOut(__METHOD_NAME__, "ERROR: " + vinciaName() + " ant("
            + num2str((int)sqrt(saj)) + "^2," + num2str((int)sqrt(sjb)) + "^2,"
            + num2str((int)sqrt(sABnow)) + "^2 ; " + helString + ") = "
            + num2str(ant,7) + " < 0");
      } else if (ant > 0.0) isZero = false;

      // Check for dead zones away from phase-space boundary
      if ((1.0 - yaj - yjb > 0.05) && (yaj > 0.05) && (yjb > 0.05)
          && (ant*sABnow < deadZoneAvoidance)) hasDeadZone = true;
    }
    isOK = isOK && isPositive;

    // Verbose output
    if (!isPositive && verbose >= QUIET)
      printOut(__METHOD_NAME__, "ERROR: " + vinciaName()
        + " (ant < 0 encountered " + helString + " )");
    else if (isPositive && !isZero && verbose >= DEBUG)
      printOut(__METHOD_NAME__, vinciaName() + " OK (is positive "
        + helString + " )");
    if (!isZero) {
      if (hasDeadZone) {
        if ( (!sectorShower && verbose >= QUIET)
          || (sectorShower && verbose >= REPORT) )
        printOut(__METHOD_NAME__, "WARNING:" + vinciaName()
          + " (dead zone encountered " + helString+" )");
      }
      else if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, vinciaName() + " OK (no dead zones "
          + helString+" )");
    }
  } // End loop over helicities.
  return isOK;

}

//==========================================================================

// Class AntQQemitII, initial-initial antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntQQemitII::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Invariants.
  double sAB = invariants[0];
  double saj = invariants[1];
  double sjb = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjb <= 0.0) || (sAB <= 0.0)) return 0.0;

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef,&helNew);
  if (nAvg <= 0) return 0.;

  // Shorthands.
  double sab  = saj+sjb+sAB;
  double yaj  = saj/sab;
  double yjb  = sjb/sab;
  double yAB  = sAB/sab;
  double eikJ = 1./(sAB*yaj*yjb);

  // Mass corrections.
  double facMa = (mi != 0.) ? pow2(mi) / sab / pow2(yaj) / sAB : 0.;
  double facMb = (mk != 0.) ? pow2(mk) / sab / pow2(yjb) / sAB : 0.;

  // Do helicity sum.
  double hSum = 0.0;

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {
    // ++ > +++ && -- > --- antennae (MHV).
    term = eikJ - facMa - facMb;
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // ++ > +-+ && -- > -+- antennae (NMHV).
    term = pow2(yAB)*eikJ - pow2(1-yjb)*facMa - pow2(1-yaj)*facMb;
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > --+ && -- > ++- antennae (massive helicity flip).
    if (mi != 0.0) {
      term = facMa * pow2(yjb);
      if (RH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    }

    // ++ > +-- && -- > -++ antennae (massive helicity flip).
    if (mk != 0.0) {
      term = facMb * pow2(yaj);
      if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;
    }
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {
    // +- > ++- && -+ > --+ antennae.
    term = pow2(1. - yaj)*eikJ;
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae.
    term = pow2(1. - yjb)*eikJ;
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > --- && -+ > +++ antennae (massive helicity flip)
    if (mi != 0.0) {
      term = facMa * pow2(yjb);
      if (RH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    }

    // +- > -++ && -+ > +-- antennae (massive helicity flip)
    if (mk != 0.0) {
      term = facMb * pow2(yaj);
      if (RH[hA] && LH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    }
  }

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg;

}

//--------------------------------------------------------------------------

// AP splitting kernel for collinear limit checks, P(z)/Q2.

double AntQQemitII::AltarelliParisi(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  double sAB = invariants[0];
  double saj = invariants[1];
  double sjb = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjb <= 0.0) || (sAB <= 0.0)) return 0.0;

  // Store helicities.
  int hBef0 = helBef[0];
  int hBef1 = helBef[1];
  int hNew0 = helNew[0];
  int hNew1 = helNew[1];
  int hNew2 = helNew[2];

  double AP(-1.);
  // Using smaller invariant for collinear limit.
  if (saj < sjb) {
    double z = zA(invariants);
    if (hNew2 == hBef1)
      AP = dglapPtr->Pq2qg(zA(invariants),hNew0,hBef0,hNew1)/z/saj;
  }
  else if (saj > sjb) {
    double z = zB(invariants);
    if (hNew0 == hBef0)
      AP = dglapPtr->Pq2qg(z,hNew2,hBef1,hNew1)/z/sjb;
  }

  return AP;

}

//==========================================================================

// Class AntGQemitII, initial-initial antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntGQemitII::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Invariants and helicities.
  double sAB = invariants[0];
  double saj = invariants[1];
  double sjb = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjb <= 0.0) || (sAB <= 0.0)) return 0.0;

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef, &helNew);
  if (nAvg <= 0) return 0.;

  // Shorthands
  double sab   = saj+sjb+sAB;
  double yaj   = saj/sab;
  double yjb   = sjb/sab;
  double yAB   = sAB/sab;
  double eikJ  = 1./(sAB*yaj*yjb);
  double collA = 1./(sAB*yaj*(1. - yjb));

  // Mass corrections.
  double facMb = (mk != 0.) ? pow2(mk) / sab / pow2(yjb) / sAB : 0.;

  // Do helicity sum.
  double hSum = 0.0;

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {
    // ++ > +++ && -- > --- antennae (MHV).
    term = eikJ + collA - facMb;
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // ++ > +-+ && -- > -+- antennae (NMHV).
    term = (1. - yjb)*pow2(yAB)*eikJ - pow2(1-yaj) * facMb;
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > --+ && -- > ++- antennae.
    term = pow3(yjb)*collA;
    if (RH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > +-- && -- > -++ antennae (massive helicity flip).
    if (mk != 0.) {
      term = facMb * pow2(yaj);
      if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;
    }
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {
    // +- > ++- && -+ > --+ antennae.
    term = pow2(1. - yaj)*eikJ + collA - pow2(1-yaj)*facMb;
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae.
    term = pow3(1. - yjb)*eikJ - facMb;
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > --- && -+ > +++ antennae.
    term = pow3(yjb)*collA;
    if (RH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > +++ && -+ > --- antennae (massive helicity flip).
    if (mk != 0.) {
      term = facMb * pow2(yaj);
      if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
    }
  }

  // Subleading colour correction.
  if (modeSLC >= 2) hSum *= CF/chargeFacSav * (1 - yjb)/(2 - yaj - yjb)
      + CA/chargeFacSav * (1 - yaj)/(2 - yaj - yjb);

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg;

}

//--------------------------------------------------------------------------

// AP splitting kernel for collinear limit checks, P(z)/Q2.

double AntGQemitII::AltarelliParisi(vector<double> invariants,
  vector<double>, vector<int> helBef, vector<int> helNew) {

  double sAB = invariants[0];
  double saj = invariants[1];
  double sjb = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjb <= 0.0) || (sAB <= 0.0)) return 0.0;

  // Store helicities.
  int hBef0 = helBef[0];
  int hBef1 = helBef[1];
  int hNew0 = helNew[0];
  int hNew1 = helNew[1];
  int hNew2 = helNew[2];

  double AP(-1.);
  // Collinear to initial-state gluon.
  if (saj < sjb) {
    double z = zA(invariants);
    if (hNew2 == hBef1)
      AP = dglapPtr->Pg2gg(z,hNew0,hBef0,hNew1)/z/saj;
  }
  // Collinear to initial-state quark.
  else if (saj > sjb) {
    double z = zB(invariants);
    if (hNew0 == hBef0)
      AP = dglapPtr->Pq2qg(z,hNew2,hBef1,hNew1)/z/sjb;
  }

  return AP;

}

//==========================================================================

// Class AntGGemitII, initial-initial antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntGGemitII::antFun(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  // Invariants and helicities.
  double sAB = invariants[0];
  double saj = invariants[1];
  double sjb = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjb <= 0.0) || (sAB <= 0.0)) return 0.0;

  // Initialise helicities. Return 0 for unphysical helicities.
  int nAvg = initHel(&helBef, &helNew);
  if (nAvg <= 0) return 0.;

  // Shorthands.
  double sab   = saj+sjb+sAB;
  double yaj   = saj/sab;
  double yjb   = sjb/sab;
  double yAB   = sAB/sab;
  double eikJ  = 1./(sAB*yaj*yjb);
  double collA = 1./(sAB*yaj*(1.-yjb));
  double collB = 1./(sAB*yjb*(1.-yaj));

  // Do helicity sum.
  double hSum = 0.0;

  // (++ and --) parents.
  if ( hA * hB > 0 || hA == 9 || hB == 9) {
    // ++ > +++ && -- > --- antennae (MHV).
    term = eikJ + collA + collB;
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // ++ > +-+ && -- > -+- antennae (NMHV).
    term = pow3(yAB)*eikJ;
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > --+ && -- > ++- antennae.
    term = pow3(yjb)*collA;
    if (RH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > +-- && -- > -++ antennae.
    term = pow3(yaj)*collB;
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {
    // +- > ++- && -+ > --+ antennae.
    term = pow3(1.-yaj)*eikJ + collA;
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae.
    term = pow3(1 - yjb)*eikJ + collB;
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > +++ && -+ > --- antennae.
    term = pow3(yaj)*collB;
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // +- > --- && -+ > +++ antennae.
    term = pow3(yjb)*collA;
    if (RH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
  }

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg;

}

//--------------------------------------------------------------------------

// AP splitting kernel, P(z)/Q2.

double AntGGemitII::AltarelliParisi(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  double sAB = invariants[0];
  double saj = invariants[1];
  double sjb = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjb <= 0.0) || (sAB <= 0.0)) return 0.0;

  // Store helicities.
  int hBef0 = helBef[0];
  int hBef1 = helBef[1];
  int hNew0 = helNew[0];
  int hNew1 = helNew[1];
  int hNew2 = helNew[2];

  double AP(-1.);
  if (saj < sjb) {
    double z = zA(invariants);
    if (hNew2 == hBef1)
      AP = dglapPtr->Pg2gg(z,hNew0,hBef0,hNew1)/z/saj;
  }
  else if (saj > sjb) {
    double z = zB(invariants);
    if (hNew0 == hBef0)
      AP = dglapPtr->Pg2gg(z,hNew2,hBef1,hNew1)/z/sjb;
  }

  return AP;

}

//==========================================================================

// Class AntQXsplitII, initial-initial antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntQXsplitII::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Invariants and helicities.
  double sAB = invariants[0];
  double saj = invariants[1];
  double sjb = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjb <= 0.0) || (sAB <= 0.0)) return 0.0;

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef,&helNew);
  if (nAvg <= 0) return 0.;

  // Shorthands.
  double sab    = sAB + saj + sjb;
  double yaj    = saj/sab;
  double yAB    = sAB/sab;
  double splitA = 1.0/sAB/yaj;
  double facMj  = (mj != 0. ) ? pow2(mj) / sab / pow2(yaj) / sAB : 0.0;

  // Do helicity sum.
  double hSum = 0.0;

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {
    // ++ > +-+ && -- > -+- antennae.
    term = pow2(yAB) * splitA - pow2(yAB)/(1. - yAB) * facMj;
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > --+ && -- > ++- antennae.
    term = pow2(1. - yAB) * splitA - (1. - yAB) * facMj;
    if (RH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > +++ && -- > --- antennae (massive helicity flip).
    if (mj != 0.0) {
      term = facMj / (1. - yAB);
      if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
    }
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {
    // +- > +-- && -+ > -++ antennae.
    term = pow2(yAB) * splitA - pow2(yAB)/(1. - yAB) * facMj;
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > --- && -+ > +++ antennae.
    term = pow2(1. - yAB) * splitA - (1. - yAB) * facMj;
    if (RH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > ++- && -+ > --+ antennae.
    if (mj != 0.0) {
      term = facMj / (1. - yAB);
      if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
    }
  }

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg;

}

//--------------------------------------------------------------------------

// AP splitting kernel, P(z)/Q2.

double AntQXsplitII::AltarelliParisi(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  double sAB = invariants[0];
  double saj = invariants[1];
  double sjb = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjb <= 0.0) || (sAB <= 0.0)) return 0.0;

  // Store helicities.
  int hBef0 = helBef[0];
  int hBef1 = helBef[1];
  int hNew0 = helNew[0];
  int hNew1 = helNew[1];
  int hNew2 = helNew[2];

  double AP(-1.);
  // There is only the saj collinear limit.
  if (hNew2 == hBef1) {
    double z  = zA(invariants);
    AP = dglapPtr->Pg2qq(z,hNew0,hBef0,hNew1)/z/saj;
  }

  return AP;

}

//==========================================================================

// Class AntGXconvII, initial-initial antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntGXconvII::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Invariants and helicities.
  double sAB = invariants[0];
  double saj = invariants[1];
  double sjb = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjb <= 0.0) || (sAB <= 0.0)) return 0.0;

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef, &helNew);
  if (nAvg <= 0) return 0.;

  // Shorthands.
  //  double sab   = sAB + saj + sjb - 2*pow2(mj);
  double sab   = sAB + saj + sjb - pow2(mj);
  double yaj   = saj/sab;
  double yAB   = sAB/sab;
  double mu2j  = (mj != 0.) ? pow2(mj)/sab : 0.;
  //  double convA = 1.0/(2.*sAB*(yaj - 2.*mu2j)*yAB);
  //  double facMj = (mj != 0.) ? mu2j/(2.*sAB*pow2(yaj - 2.*mu2j)) : 0.;
  double convA = 1.0/(2.*sAB*(yaj - mu2j)*yAB);
  double facMj = (mj != 0.) ? mu2j/(2.*sAB*pow2(yaj - mu2j)) : 0.;

  // Do helicity sum.
  double hSum = 0.0;

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {
    // ++ > +++ && -- > --- antennae.
    term = convA - facMj*yAB/(1. - yAB);
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // ++ > --+ && -- > ++- antennae.
    term = pow2(1. - yAB)*convA - facMj*yAB*(1. - yAB);
    if (RH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > +-+ && -- > -+- antennae (massive helicity flip).
    if (mj != 0.) {
      term = facMj*pow3(yAB)/(1. - yAB);
      if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;
    }
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {
    // +- > ++- && -+ > --+ antennae.
    term = convA - facMj*yAB/(1. - yAB);
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > --- && -+ > +++ antennae.
    term = pow2(1 - yAB)*convA - facMj*yAB*(1. - yAB);
    if (RH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae (massive helicity flip).
    term = facMj*pow3(yAB)/(1. - yAB);
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;
  }

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg;

}

//--------------------------------------------------------------------------

// AP splitting kernel, P(z)/Q2.

double AntGXconvII::AltarelliParisi(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  double sAB = invariants[0];
  double saj = invariants[1];
  double sjb = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjb <= 0.0) || (sAB <= 0.0)) return 0.0;

  // Store helicities.
  int hBef0 = helBef[0];
  int hBef1 = helBef[1];
  int hNew0 = helNew[0];
  int hNew1 = helNew[1];
  int hNew2 = helNew[2];

  double AP(-1.);
  // There is only the saj collinear limit.
  if (hNew2 == hBef1) {
    double z  = zA(invariants);
    AP = dglapPtr->Pq2gq(z,hNew0,hBef0,hNew1)/z/saj;
    // Gluon appears in two antennae.
    AP *= 0.5;
  }

  return AP;

}

//==========================================================================

// Class AntennaFunctionIF, base class for IF antenna functions which
// implements AntQQemitIF.

//--------------------------------------------------------------------------

// Method to initialise (can be different than that of the base class).

bool AntennaFunctionIF::init() {

  // Check whether pointers are initialized.
  if (!isInitPtr) return false;
  verbose             = settingsPtr->mode("Vincia:verbose");
  chargeFacSav        = settingsPtr->parm(vinciaName()+":chargeFactor");
  if (chargeFacSav < 0.) chargeFacSav = 0.0;

  // Subleading-colour treatment.
  // modeSLC = 0: all gluon-emission antennae normalised to CA.
  // modeSLC = 1: use colour factors as specified by user.
  // modeSLC = 2: QQ gets CF, GG gets CA, QG gets interpolation.
  modeSLC = settingsPtr->mode("Vincia:modeSLC");
  if (modeSLC == 0 && id1() == 21) chargeFacSav = CA;
  if (modeSLC == 2 && id1() == 21) {
    if (idA() == 21 && idB() == 21 ) chargeFacSav = CA;
    else if (idA() == 21 || idB() == 21) chargeFacSav = (CA + CF)/2.;
    else chargeFacSav = CF;
  }

  // Set kinematics map, depending on whether this is an IF or RF antenna.
  if (settingsPtr->isMode(vinciaName() + ":kineMap"))
    kineMapSav = settingsPtr->mode(vinciaName()+":kineMap");
  else if (isRFant()) {
    // Gluon emission antennae.
    if (id1() == 21) kineMapSav = settingsPtr->mode("Vincia:kineMapRFemit");
    // Gluon splitting antennae.
    else kineMapSav = settingsPtr->mode("Vincia:kineMapRFsplit");
  }
  // IF antennae.
  else kineMapSav = settingsPtr->mode("Vincia:kineMapIF");

  // Sector shower on/off and sectorDamp parameter.
  sectorShower  = settingsPtr->flag("Vincia:sectorShower");
  sectorDampSav = settingsPtr->parm("Vincia:sectorDamp");

  // Collinear partitioning (for global antennae).
  if (!sectorShower) alphaSav = settingsPtr->parm("Vincia:octetPartitioning");
  else alphaSav = 1.;

  // Return OK.
  isInit = true;
  return isInit;

}

//--------------------------------------------------------------------------

// Function to check singularities, positivity, etc.

bool AntennaFunctionIF::check() {

  // For check for positivity
  bool isOK        = true;
  int nPointsCheck = settingsPtr->mode("Vincia:nPointsCheck");
  double shhMax    = pow2(14000.0);
  double deadZoneAvoidance = settingsPtr->parm("Vincia:deadZoneAvoidance");

  // Resonance-final.
  if (isRFant()) return checkRes();

  // Initial-Final: sAK + sjk = sak + saj.
  else {

    // Two values for sAK.
    double sAK[2] = {pow2(50.0), pow2(150.0)};

    // Check soft singularity against dipole factor.
    if (id1() == 21 || id1() == 22) {
      // Test one symmetric and one asymmetric soft phase-space point.
      for (int iTest = 0; iTest <= 1; ++iTest) {
        // Test two sAK values.
        for (int isAK = 0; isAK <= 1; ++isAK) {
          // Test invariants.
          double yaj = 1.0e-3 * pow(10.0, iTest);   // saj/sAK
          double yjk = 1.0e-3 / pow(10.0, iTest);   // sjk/sAK
          double sak = sAK[isAK]*(1.0 + yjk - yaj); // 1 + yjk = sak/sAK + yaj
          double saj = yaj*sAK[isAK];
          double sjk = yjk*sAK[isAK];
          vector<double> invariants ={sAK[isAK], saj, sjk};
          // Compute eikonal.
          double eik = 2.0*sak/saj/sjk;
          // Compute antenna (without subleading-colour corrections).
          int modeSLCsave = modeSLC;
          modeSLC = 1;
          double ant = antFun(invariants);
          modeSLC = modeSLCsave;
          // Compare.
          double ratio = ant/eik;
          if (abs(ratio - 1.0) >= 0.001) {
            isOK = false;
            if (verbose >= NORMAL) printOut(vinciaName() + ":check",
                "WARNING: FAILED (eikonal " + num2str(iTest, 1) +
                " and sAK = " + num2str((int)sqrt(sAK[isAK])) + "^2)");
          } else if (verbose >= DEBUG) printOut(vinciaName() + ":check",
              "OK (eikonal " + num2str(iTest, 1) + " and sAK = " +
              num2str((int)sqrt(sAK[isAK])) + "^2)");
        }
      }
    }

    // Test all helicity configurations for collinearity and posivity.
    string helString = " 9 9 -> 9 9 9";
    for (int iHel = -1; iHel < 32; ++iHel) {
      bool isPositive  = true;
      bool isZero      = true;
      bool hasDeadZone = false;
      if (iHel >= 0) {
        hj = 1 - 2*(iHel%2);
        hi = 1 - 2*((iHel/2)%2);
        hk = 1 - 2*((iHel/4)%2);
        hA = 1 - 2*((iHel/8)%2);
        hB = 1 - 2*((iHel/16)%2);
        helString = num2str(hA,2) + num2str(hB,2) + " ->" +
          num2str(hi,2) + num2str(hj,2) + num2str(hk,2);
      }
      vector<int> helBef = {hA, hB};
      vector<int> helNew = {hi, hj, hk};

      // Gluon emission.
      if (id1() == 21) {
        // Massless: quark helicities must be conserved.
        if (idA() != 21 && hi != hA) continue;
        if (idB() != 21 && hk != hB) continue;
        // For gluons, sector terms can give helicity flips
        if (idA() == 21 && (hi != hA && hj != -hA)) continue;
        if (sectorShower && idB() == 21 && (hk != hB && hj != hB)) continue;
        if (!sectorShower && idB() == 21 && hk != hB) continue;
        // Do not allow helicity flips on both sides simultaneously
        if (hi != hA && hk != hB) continue;
      }
      // Gluon conversion.
      else if (idA() == 21) {
        // Helicity of recoiler must be conserved.
        if (hk != hB) continue;
        // Massless: quark helicity must be conserved.
        if (hi != hj) continue;
      }
      // Gluon splitting in the final state.
      else if (idB() == 21 && id2() == -id1()) {
        // Helicity of recoiler must be conserved.
        if (hi != hA) continue;
        // Massless: quark helicity must be conserved.
        if (hj != -hk) continue;
      }
      // Gluon splitting in the initial state.
      else {
        // Helicity of recoiler must be conserved.
        if (hk != hB) continue;
        // Massless: quark helicity must be conserved.
        if (hA == hj) continue;
      }


      // Check collinear singularities against massless AP kernels.
      for (int iTest = 0; iTest <= 2; ++iTest) {
        // Test two sAK values.
        for (int isAK = 0; isAK <= 1; ++isAK) {
          // Test invariants for a few points along collinear axis.
          double y1  = 1.0e-5;          // saj/sAK
          double y2  = 0.2 + iTest*0.3; // sjk/sAK
          double s1  = y1*sAK[isAK];
          double s2  = y2*sAK[isAK];
          vector<double> invariants1 = {sAK[isAK], s1, s2};
          vector<double> invariants2 = invariants1;
          invariants2[1] = invariants1[2];
          invariants2[2] = invariants1[1];

          // Compute AP kernels.
          double AP1 = AltarelliParisi(invariants1,mDum,helBef,helNew);
          double AP2 = AltarelliParisi(invariants2,mDum,helBef,helNew);

          // Are there any limits to check?
          if (AP1 < 0. && AP2 < 0.) continue;

          // Compute antennae (without subleading colour corrections).
          int modeSLCsave = modeSLC;
          modeSLC = 1;
          double ant1 = antFun(invariants1,mDum,helBef,helNew);
          double ant2 = antFun(invariants2,mDum,helBef,helNew);
          // For global antennae, add "swapped" term to represent neighbour.
          if (!sectorShower && idB() == 21) {
            invariants2[1] = (1.+y1-y2)*sAK[isAK];
            invariants2[2] = y1*sAK[isAK];
            vector<int> helComp = {hi, hk, hj};
            ant2 += antFun(invariants2,mDum,helBef,helComp);
          }
          // Restore subleading-colour level.
          modeSLC = modeSLCsave;

          // Check if only a singularity on one side exists (-> z = -1).
          double zs1 = zA(invariants1);
          double zs2 = zB(invariants1);
          if (zs1 != -1.0) {
            double ratio = ant1/AP1;
            // Require better than 5% agreement unless dominated
            // by non-singular.
            if (fabs(ratio - 1.0) >= 0.05 && fabs(ant1 - AP1) > 10.) {
              isOK = false;
              if (verbose >= NORMAL)
                printOut(__METHOD_NAME__, "WARNING: " + vinciaName()
                  + " FAILED (collinear aj " + num2str(iTest, 1)
                  + " and sAK = " + num2str((int)sqrt(sAK[isAK])) + "^2)");
              if (verbose >= REPORT) {
                cout << setprecision(6);
                printOut(__METHOD_NAME__, "    (" + helString + ")");
                printOut(__METHOD_NAME__, "    ant  = " + num2str(ant1, 9)
                  + " y1 = " + num2str(y1, 9) + " y2 = " + num2str(y2, 9));
                printOut(__METHOD_NAME__, "    P(z) = " + num2str(AP1, 9)
                  + "  z = " + num2str(zs1, 9));
              }
            } else if (verbose >= DEBUG)
                printOut(__METHOD_NAME__, vinciaName() + " OK (collinear aj "
                  + num2str(iTest, 1) + " and sAK = "
                  + num2str((int)sqrt(sAK[isAK])) + "^2)");
          }
          if (zs2 != -1.0) {
            double ratio = ant2/AP2;
            // Require better than 5% agreement unless dominated
            // by non-singular.
            if (fabs(ratio - 1.0) >= 0.05 && fabs(ant2 - AP2) > 10.) {
              isOK = false;
              if (verbose >= NORMAL)
                printOut(__METHOD_NAME__, "WARNING: " + vinciaName()
                  + " FAILED (collinear jk " + num2str(iTest, 1)
                  + " and sAK = " + num2str((int)sqrt(sAK[isAK])) + "^2)");
              if (verbose >= REPORT) {
                cout << setprecision(6);
                printOut(__METHOD_NAME__, "    (" + helString + ")");
                printOut(__METHOD_NAME__, "    ant  = " + num2str(ant2, 9)
                  + " y1 = " + num2str(y1, 9) + " y2 = " + num2str(y2, 9));
                printOut(__METHOD_NAME__, "    P(z) = " + num2str(AP2, 9)
                  + "  z = " + num2str(zs1, 9));
              }
            } else if (verbose >= DEBUG)
                printOut(__METHOD_NAME__, vinciaName() + " OK (collinear jk "
                  + num2str(iTest, 1) + " and sAK = "
                  + num2str((int)sqrt(sAK[isAK])) + "^2)");
          }
        }
      }

      // Check for positivity and dead zones.
      for (int iTest = 0; iTest <= nPointsCheck; ++iTest) {
        // sAK^0.5 between 1 and 1000 GeV.
        double sAKnow = pow2(1.0+rndmPtr->flat()*999.0);
        // Pick random xA.
        double xA  = rndmPtr->flat();
        // sjk between 0 and sAK(1-xA)/xA, saj between 0 and sAK+sjk.
        double sjk = rndmPtr->flat()*sAKnow*(1.0-xA)/xA;
        double saj = rndmPtr->flat()*(sAKnow+sjk);
        double sak = sAKnow+sjk-saj;
        vector<double> invariants = {sAKnow, saj, sjk};
        // Check that sxy < shhMax.
        if (saj > shhMax || sjk > shhMax || sak > shhMax || sAKnow > shhMax)
          continue;
        // Restrict checks to ordered region; require pT2 < sAK
        double pT2 = saj * sjk / (sAKnow + sjk);
        if (pT2 > sAKnow) continue;
        // Compute antenna.
        double ant = antFun(invariants, mDum, helBef, helNew);

        // Check positivity (strict).
        if (ant < 0.0) {
          isPositive = false;
          if (verbose >= REPORT)
            printOut(__METHOD_NAME__, "ERROR: " + vinciaName() + " ant("
              + num2str(sAKnow) + ","
              + num2str(saj/(sAKnow+sjk)) + ","
              + num2str(sjk/(sAKnow+sjk)) + " ; " + helString + ") = "
              + num2str(sAKnow*ant,16) + " < 0");
        } else if (ant > 0.0) isZero = false;

        // Check for dead zones away from phase-space boundary.
        double yjk = sjk / (sAKnow + sjk);
        double yAK = sAKnow / (sAKnow + sjk);
        double yaj = saj / (sAKnow + sjk);
        double yak = sak / (sAKnow + sjk);
        double yMin = 5*sqrt(deadZoneAvoidance);
        if ((sjk < 0.95*sAKnow*(1.0-xA)/xA)
          && yjk > yMin && yaj > yMin && yak > yMin && yAK > yMin
          && (ant*(sAKnow+sjk) < deadZoneAvoidance)) {
          hasDeadZone = true;
        }
      }
      isOK = isOK && isPositive;

      // Verbose output.
      if (!isPositive && verbose >= QUIET)
        printOut(__METHOD_NAME__, "ERROR: " + vinciaName()
          + " (ant < 0 encountered " + helString + " )");
      else if (isPositive && !isZero && verbose >= DEBUG)
        printOut(__METHOD_NAME__, vinciaName() + " OK (is positive "
          + helString + " )");
      if (!isZero) {
        if (hasDeadZone) {
          if ( (!sectorShower && verbose >= QUIET)
            || (sectorShower && verbose >= REPORT) )
            printOut(__METHOD_NAME__, "WARNING:" + vinciaName()
              + " (dead zone encountered " + helString+" )");
        }
        else if (verbose >= DEBUG)
          printOut(__METHOD_NAME__, vinciaName() + " OK (no dead zones "
            + helString+" )");
      }
    } // End loop over helicities.
  } // End loop over initi-final state.
  return isOK;

}

//--------------------------------------------------------------------------

// Check for resonances.

bool AntennaFunctionIF::checkRes() {

  // Get test masses.
  vector<double> masses;
  getTestMasses(masses);

  // Test soft singularity for gluon emissions.  Only test asymmetric
  // soft phase-space point (masses limit available phase space).
  if (id1() == 21 || id1() == 22) {
    // Get scaled invariants.
    double yaj = 0.01;
    double yjk = 0.0001;

    // Get dimensionful invariants.
    vector<double> invariants;
    if (!getTestInvariants(invariants, masses, yaj, yjk)) return false;

    // Get massive eikonal.
    double antSoft = massiveEikonal(invariants, masses);

    // Get full antenna.
    double antNow = antFun(invariants, masses);

    // Check ratio.
    double ratio= antNow/antSoft;
    if (abs(ratio - 1.) >= 0.001) {
      if (verbose >= QUIET) {
        stringstream ss;
        ss << "WARNING:" + vinciaName() << " FAILED soft eikonal: ratio to "
           << "soft = " << ratio;
        printOut(__METHOD_NAME__, ss.str());
      }
      return false;
    }
    else if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, vinciaName() + " OK (soft eikonal)");
  }

  // Check collinear singularity (only check for final state parton,
  // quasi-collinear limit not relevant for res here). Test
  // invariants, for a few points along collinear axis.
  for (int iTest = 0; iTest < 4; ++iTest) {
    // Get scaled invariants.
    double yaj = 0.2 + double(iTest)*0.2;
    double yjk = 0.01;

    // Get dimensionful invariants.
    vector<double> invariants;
    if (!getTestInvariants(invariants, masses, yaj, yjk)) {
      if (verbose >= NORMAL) infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Failed to get test invariants!");
      return false;
    }

    // Are we still in the available phase space?
    double gramdet = gramDet(invariants, masses);
    if (gramdet<0.) {
      if (verbose >= DEBUG) printOut(__METHOD_NAME__,
        vinciaName() + " not in phase space. Continue.");
      break;
    }

    // Get the antenna (with sum over flipped invariants where
    // appropriate).
    double antNow = antFunCollLimit(invariants, masses);

    // Get AP function.
    double AP = AltarelliParisi(invariants, masses);
    if (AP > 0.) {
      double ratio = antNow/AP;
      // Require better than 1% agreement unless dominated by nonsingular.
      if (abs(ratio - 1.) >= 0.01 && abs(antNow - AP) > 10.) {
        if (verbose >= NORMAL)
          printOut(__METHOD_NAME__, "WARNING:" + vinciaName() +
            "Failed (collinear ij " + num2str(iTest, 1)+" )");
        if (verbose >= REPORT) cout << setprecision(6) << "    ant  = "
              << num2str(antNow, 9) << " yaj = " << num2str(yaj, 9)
              << " yjk = " << num2str(yjk, 9) << " " << endl << "    P(z) = "
              << num2str(AP, 9) << endl;
        return false;
      }
      else if (verbose >= DEBUG) printOut(__METHOD_NAME__, vinciaName()
        + " OK (collinear ij " + num2str(iTest, 1) + " )");
    }
  }
  return true;

}

//--------------------------------------------------------------------------

// Create the test invariants for the checkRes method.

bool AntennaFunctionIF::getTestInvariants(vector<double> &invariants,
  vector<double> masses, double yaj, double yjk) {

  if (masses.size() != 4) return false;
  double mA  = masses[0];
  double mK  = masses[2];
  double mAK = masses[3];
  double sAK = mA*mA + mK*mK - mAK*mAK;
  double sjk = sAK*yjk/(1. - yjk);
  if((sjk+sAK)==0.0) return false;
  double saj = yaj*(sAK+sjk);
  double sak = sAK + sjk -saj;
  double det = saj*sjk*sak - saj*saj*mK*mK - sjk*sjk*mA*mA;
  if (det <0.) return false;
  invariants = {sAK, saj, sjk, sak};
  return true;

}

//--------------------------------------------------------------------------

// Wrapper for comparing to AP functions, sums over flipped
// invariants where appropriate.

double AntennaFunctionIF::antFunCollLimit(vector<double> invariants,
  vector<double> masses) {

  double ant = antFun(invariants,masses);
  if (idB() == 21) {
    vector<double> flipped_invariants = {
      invariants[0], invariants[3], invariants[2], invariants[1]};
    ant += antFun(flipped_invariants,masses);
  }
  return ant;

}

//==========================================================================

// Class AntQQemitIF, initial-final antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntQQemitIF::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Invariants and helicities.
  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef, &helNew);
  if (nAvg <= 0) return 0.;

  // Shorthands.
  double s     = sAK + sjk;
  double yaj   = saj/s;
  double yjk   = sjk/s;
  double eikJ  = 1./(sAK*yaj*yjk);
  double facMa = (mi != 0.) ? pow2(mi)/s/sAK/pow2(yaj) : 0.;
  double facMk = (mk != 0.) ? pow2(mk)/s/sAK/pow2(yjk) : 0.;

  // Do helicity sum.
  double hSum = 0.0;

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {
    // ++ > +++ && -- > --- antennae (MHV).
    term = eikJ - facMa - facMk/(1. - yaj);
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // ++ > +-+ && -- > -+- antennae (NMHV).
    term = (pow2(1. - yaj) + (pow2(1. - yjk) - 1.)*pow2(1. - yaj)) * eikJ
      - facMa*pow2(1. - yjk-yaj) - facMk*(1. - yaj)*pow2(1. - yjk);
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > --+ && -- > ++- antennae (massive helicity flip).
    if (mi != 0.) {
      term = facMa * pow2(yjk);
      if (RH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    }

    // ++ > ++- && -- > --+ antennae (massive helicity flip).
    if (mk != 0.) {
      term = facMk * pow2(yaj)/(1. - yaj);
      if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
    }
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {
    term = pow2(1.-yaj) * eikJ - facMa*(1. - yaj) - facMk*(1. - yaj);

    // +- > ++- && -+ > --+ antennae.
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae.
    term = pow2(1.-yjk) * eikJ - facMa*pow2(1. - yjk)
      - facMk*pow2(1. - yjk)/(1. - yaj);
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > --- && -+ > +++ antennae (massive helicity flip).
    if (mi != 0.) {
      term = facMa * pow2(yjk);
      if (RH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    }

    // +- > +-+ && -+ > -+- antennae (massive helicity flip).
    if (mk != 0.) {
      term = facMk * pow2(yaj) / (1. - yaj);
      if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;
    }
  }

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg;

}

//--------------------------------------------------------------------------

// The AP kernel, P(z)/Q2.

double AntQQemitIF::AltarelliParisi(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return -1.;

  // Store helicities
  int hBef0 = helBef[0];
  int hBef1 = helBef[1];
  int hNew0 = helNew[0];
  int hNew1 = helNew[1];
  int hNew2 = helNew[2];

  double AP(-1.);
  // Collinear to initial-state quark.
  if (saj < sjk) {
    if (hNew2 == hBef1) {
      double z = zA(invariants);
      AP = dglapPtr->Pq2qg(z,hNew0,hBef0,hNew1)/z/saj;
    }
  }
  // Collinear to final-state quark.
  else if (saj > sjk) {
    if (hNew0 == hBef0) {
      double z = zB(invariants);
      AP = dglapPtr->Pq2qg(z,hBef1,hNew2,hNew1)/sjk;
    }
  }

  return AP;

}

//==========================================================================

// Class AntQGemitIF, initial-final antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntQGemitIF::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Invariants and helicities.
  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef, &helNew);
  if (nAvg <= 0) return 0.;

  // Shorthands.
  double s     = sAK + sjk;
  double yaj   = saj/s;
  double yjk   = sjk/s;
  double eikJ  = 1./(sAK*yaj*yjk);
  double colK  = (alphaSav == 1.) ? 0.
    : (1. - alphaSav) * (1. - 2.*yaj) / (sAK * yjk);
  double facMa = (mi != 0.) ? pow2(mi)/s/sAK/pow2(yaj) : 0.;

  // Do helicity sum.
  double hSum = 0.0;

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {
    // ++ > +++ && -- > --- antennae (MHV).
    term = eikJ + colK - facMa;
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // ++ > +-+ && -- > -+- antennae (NMHV).
    term = (pow3(1. - yaj)+pow2(1. - yjk)-1) * eikJ
      - facMa*pow2(1. - yjk - yaj)*(1. - yaj) + (3. - pow2(yaj))/sAK;
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > --+ && -- > ++- antennae (massive helicity flip).
    if (mi != 0.) {
      term = facMa*pow2(yjk);
      if (RH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    }
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {
    // +- > ++- && -+ > --+ antennae.
    term = pow3(1. - yaj) * eikJ - facMa*pow2(1. - yaj);
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae.
    term = pow2(1. - yjk) * eikJ + colK - facMa*pow2(1. - yjk)
      + (2*yaj - yjk)/sAK;
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > --- && -+ > +++ antennae (massive helicity flip).
    if (mi != 0.) {
      term = facMa * pow2(yjk);
      if (RH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    }
  }

  // Subleading colour correction.
  // Note: has to be applied to sector antennae after symmetrisation!
  if (!sectorShower && modeSLC >= 2)
    hSum *= CF/chargeFacSav * (1 - yaj)/(2 - yaj - yjk)
      + CA/chargeFacSav * (1 - yjk)/(2 - yaj - yjk);

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg;

}

//--------------------------------------------------------------------------

// The AP kernel, P(z)/Q2.

double AntQGemitIF::AltarelliParisi(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Store helicities.
  int hBef0 = helBef[0];
  int hBef1 = helBef[1];
  int hNew0 = helNew[0];
  int hNew1 = helNew[1];
  int hNew2 = helNew[2];

  double AP(-1.);
  // Collinear to initial-state quark.
  if (saj < sjk) {
    if (hNew2 == hBef1) {
      double z = zA(invariants);
      AP = dglapPtr->Pq2qg(z,hNew0,hBef0,hNew1)/z/saj;
    }
  }
  // Collinear to final-state gluon.
  else if (saj > sjk) {
    if (hNew0 == hBef0) {
      double z = zB(invariants);
      AP = dglapPtr->Pg2gg(z,hBef1,hNew2,hNew1)/sjk;
    }
  }

  return AP;
}

//==========================================================================

// Class AntGQemitIF, initial-final antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntGQemitIF::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Invariants and helicities.
  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef, &helNew);
  if (nAvg <= 0) return 0.;

  // Shorthands
  double s     = sAK + sjk;
  double yaj   = saj/s;
  double yjk   = sjk/s;
  double eikJ  = 1./(sAK*yaj*yjk);
  double colA  = 1./(sAK*yaj*(1. - yjk));
  double facMk = (mk != 0.) ? pow2(mk)/s/sAK/pow2(yjk) : 0.;

  // Do helicity sum.
  double hSum = 0.0;

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {
    // ++ > +++ && -- > --- antennae (MHV).
    term = eikJ + colA - facMk/(1. - yaj);
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // ++ > +-+ && -- > -+- antennae (NMHV).
    term = (pow2(1. - yaj)+(pow3(1. - yjk) - 1)*(pow2(1. - yaj)))* eikJ
      - facMk*(1. - yaj)*pow3(1 - yjk);
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > --+ && -- > ++- antennae.
    term = pow3(yjk) * colA;
    if (RH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > ++- && -- > --+ antennae (massive helicity flip).
    if (mk != 0.) {
      term = facMk * pow2(yaj)/(1. - yaj);
      if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
    }
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {
    // +- > ++- && -+ > --+ antennae.
    term = pow2(1 - yaj) * eikJ + colA - facMk*(1. - yaj);
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae.
    term = pow3(1. - yjk) * eikJ - facMk*pow3(1. - yjk)/(1. - yaj);
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > --- && -+ > +++ antennae.
    term = pow3(yjk) * colA;
    if (RH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > +-+ && -+ > -+- antennae (massive helicity flip).
    if (mk != 0.) {
      term = facMk * pow2(yaj)/(1. - yaj);
      if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;
    }
  }

  // Subleading colour correction.
  if (modeSLC >= 2)  hSum *= CF/chargeFacSav * (1 - yjk)/(2 - yaj - yjk)
      + CA/chargeFacSav * (1 - yaj)/(2 - yaj - yjk);

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg;

}

//--------------------------------------------------------------------------

// The AP kernel, P(z)/Q2.

double AntGQemitIF::AltarelliParisi(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Store helicities.
  int hBef0 = helBef[0];
  int hBef1 = helBef[1];
  int hNew0 = helNew[0];
  int hNew1 = helNew[1];
  int hNew2 = helNew[2];

  double AP(-1.);
  // Collinear to initial-state gluon.
  if (saj < sjk) {
    if (hNew2 == hBef1) {
      double z = zA(invariants);
      AP = dglapPtr->Pg2gg(z,hNew0,hBef0,hNew1)/z/saj;
    }
  }
  // Collinear to final-state quark.
  else if (saj > sjk) {
    if (hNew0 == hBef0) {
      double z = zB(invariants);
      AP = dglapPtr->Pq2qg(z,hBef1,hNew2,hNew1)/sjk;
    }
  }

  return AP;

}

//==========================================================================

// Class AntGGemitIF, initial-final antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntGGemitIF::antFun(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  // Invariants and helicities.
  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Initialise helicities.
  int nAvg = initHel(&helBef, &helNew);

  // Shorthands.
  double s    = sAK + sjk;
  double yaj  = saj/s;
  double yjk  = sjk/s;
  double yAK  = sAK/s;
  double eikJ = 1./(sAK*yaj*yjk);
  double colA = 1./(sAK*yaj*yAK);
  double colK = (1.-alphaSav) * (1.-2.*yaj) / (sAK * yjk);

  // Do helicity sum.
  double hSum = 0.0;

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {
    // ++ > +++ && -- > --- antennae (MHV).
    term = eikJ + colA + colK;
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // ++ > +-+ && -- > -+- antennae (NMHV).
    term = (pow3(1. - yaj) + pow3(1. - yjk) - 1.) * eikJ
      + ( 6. - 3.*(yaj+yjk) + yaj*yjk )/sAK;
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > --+ && -- > ++- antennae.
    term = pow3(yjk) * colA;
    if (RH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {
    // +- > ++- && -+ > --+ antennae.
    term = pow3(1. - yaj) * eikJ + colA;
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae.
    term = pow3(1. - yjk) * eikJ + colK + (3.*yaj - yjk - yaj*yjk)/sAK;
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > --- && -+ > +++ antennae.
    term = pow3(yjk) * colA;
    if (RH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
  }

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg;
}

//--------------------------------------------------------------------------

// The AP kernel, P(z)/Q2.

double AntGGemitIF::AltarelliParisi(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Store helicities.
  int hBef0 = helBef[0];
  int hBef1 = helBef[1];
  int hNew0 = helNew[0];
  int hNew1 = helNew[1];
  int hNew2 = helNew[2];

  double AP(-1.);
  // Collinear to initial-state gluon.
  if (saj < sjk) {
    if (hNew2 == hBef1) {
      double z = zA(invariants);
      AP = dglapPtr->Pg2gg(z,hNew0,hBef0,hNew1)/z/saj;
    }
  }
  // Collinear to final-state gluon.
  else if (saj > sjk) {
    if (hNew0 == hBef0)
      AP += dglapPtr->Pg2gg(zB(invariants),hBef1,hNew2,hNew1)/sjk;
  }

  return AP;
}

//==========================================================================

// Class AntQXsplitIF, initial-final antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntQXsplitIF::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Invariants and helicities.
  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef, &helNew);
  if (nAvg <= 0) return 0.;

  // Shorthands
  double s     = sAK + sjk;
  double yAK   = sAK/s;
  double yaj   = saj/s;
  double colA  = 1./(sAK * yaj);
  double facMj = (mj != 0.) ? pow2(mj)/s/sAK/pow2(yaj) : 0.;

  // Do helicity sum.
  double hSum = 0.0;

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {
    // ++ > +-+ && -- > -+- antennae.
    term = pow2(yAK) * colA - facMj*pow2(yAK)/(1. - yAK);
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > --+ && -- > ++- antennae.
    term = pow2(1. - yAK) * colA - facMj*(1. - yAK);
    if (RH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > +++ && -- > --- antennae (massive helicity flip).
    if (mj != 0.) {
      term = facMj / (1.-yAK);
      if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
    }
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {
    // +- > +-- && -+ > -++ antennae.
    term = pow2(yAK) * colA - facMj*pow2(yAK)/(1. - yAK);
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > --- && -+ > +++ antennae.
    term = pow2(1. - yAK) * colA - facMj*(1. - yAK);
    if (RH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > ++- && -+ > --+ antennae (massive helicity flip).
    if (mj != 0.) {
      term = facMj/(1. - yAK);
      if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
    }
  }

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg;

}

//--------------------------------------------------------------------------

// The AP kernel, P(z)/Q2.

double AntQXsplitIF::AltarelliParisi(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Store helicities.
  int hBef0 = helBef[0];
  int hBef1 = helBef[1];
  int hNew0 = helNew[0];
  int hNew1 = helNew[1];
  int hNew2 = helNew[2];

  double AP(-1.);
  // There is only the saj collinear limit.
  if (hNew2 == hBef1) {
    double z = zA(invariants);
    AP = dglapPtr->Pg2qq(z,hNew0,hBef0,hNew1)/z/saj;
  }

  return AP;
}

//==========================================================================

// Class AntGXconvIF, initial-final antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntGXconvIF::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Invariants and helicities.
  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef, &helNew);
  if (nAvg <= 0) return 0.;

  // Shorthands
  double s     = sAK + sjk + 2.*pow2(mj);
  double yaj   = saj / s;
  double yAK   = sAK / s;
  double mu2j  = (mj != 0.) ? pow2(mj)/s : 0.;
  double colA  = 1./(2.*sAK*yAK*(yaj - 2.*mu2j));
  double facMj = (mj != 0.) ? mu2j/(2.*sAK)/pow2(yaj - 2*mu2j) : 0.;

  // Do helicity sum.
  double hSum = 0.0;

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {
    // ++ > +++ && -- > --- antennae.
    term = colA - facMj*yAK/(1. - yAK);
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;

    // ++ > --+ && -- > ++- antennae.
    term = pow2(1. - yAK) * colA - facMj*yAK*(1. - yAK);
    if (RH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > +-+ && -- > -+- antennae (massive helicity flip).
    if (mj != 0.) {
      term = facMj * pow3(yAK)/(1. - yAK);
      if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;
    }
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {
    // +- > ++- && -+ > --+ antennae.
    term = colA - facMj*yAK/(1. - yAK);
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > --- && -+ > +++ antennae.
    term = pow2(1. - yAK) * colA - facMj*yAK*(1. - yAK) ;
    if (RH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae (massive helicity flip).
    if (mj != 0.) {
      term = facMj * pow3(yAK)/(1. - yAK);
      if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;
    }
  }

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg;

}

//--------------------------------------------------------------------------

// The AP kernel, P(z)/Q2.

double AntGXconvIF::AltarelliParisi(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Store helicities.
  int hBef0 = helBef[0];
  int hBef1 = helBef[1];
  int hNew0 = helNew[0];
  int hNew1 = helNew[1];
  int hNew2 = helNew[2];

  double AP(-1.);
  // There is only the saj collinear limit.
  if (hNew2 == hBef1) {
    double z = zA(invariants);
    AP = dglapPtr->Pq2gq(z,hNew0,hBef0,hNew1)/z/saj;
    // Gluon appears in two antennae.
    AP *= 0.5;
  }

  return AP;
}

//==========================================================================

// Class AntXGsplitIF, initial-final antenna function. Gluon splitting in
// the final state.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntXGsplitIF::antFun(vector<double> invariants, vector<double> masses,
  vector<int> helBef, vector<int> helNew) {

  // Invariants and helicities
  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Initialise masses and helicities. Return 0 for unphysical helicities.
  initMasses(&masses);
  int nAvg = initHel(&helBef, &helNew);
  if (nAvg <= 0) return 0.;

  // Shorthands.
  double s     = sAK + sjk + 2*pow2(mj);
  double yaj   = saj / s;
  double yak   = 1. - yaj;
  double m2jk  = sjk +  2*pow2(mj);
  double colK  = 1./(2.*m2jk);
  double facMj = pow2(mj)/(2.*pow2(m2jk));

  // Do helicity sum.
  double hSum = 0.0;

  // (++ and --) parents.
  if (hA * hB > 0 || hA == 9 || hB == 9) {
    // ++ > +-+ && -- > -+- antennae.
    term = pow2(yak) * colK - facMj*yak/(1. - yak);
    if (RH[hA] && RH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // ++ > ++- && -- > --+ antennae.
    term = pow2(yaj) * colK - facMj*yaj/(1. - yaj);
    if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // ++ > +++ && -- > --- antennae (massive helicity flip).
    if (mj != 0.) {
      term = facMj*(yaj/(1. - yaj) + yak/(1. - yak) + 2);
      if (RH[hA] && RH[hB] && RH[hi] && RH[hj] && RH[hk]) hSum += term;
      if (LH[hA] && LH[hB] && LH[hi] && LH[hj] && LH[hk]) hSum += term;
    }
  }

  // (+- and -+) parents.
  if (hA * hB < 0 || hA == 9 || hB == 9) {
    // +- > ++- && -+ > --+ antennae.
    term = pow2(yak) * colK - facMj*yak/(1. - yak);
    if (RH[hA] && LH[hB] && RH[hi] && RH[hj] && LH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && LH[hj] && RH[hk]) hSum += term;

    // +- > +-+ && -+ > -+- antennae.
    term = pow2(yaj) * colK - facMj*yaj/(1. - yaj);
    if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && RH[hk]) hSum += term;
    if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && LH[hk]) hSum += term;

    // +- > +-- && -+ > -++ antennae (massive helicity flip).
    if (mj != 0.) {
      term = facMj*(yaj/(1. - yaj) + yak/(1. - yak) + 2);
      if (RH[hA] && LH[hB] && RH[hi] && LH[hj] && LH[hk]) hSum += term;
      if (LH[hA] && RH[hB] && LH[hi] && RH[hj] && RH[hk]) hSum += term;
    }
  }

  // Return helicity sum, averaged over initial helicities.
  return hSum/nAvg;

}

//--------------------------------------------------------------------------

// The AP kernel, P(z)/Q2.

double AntXGsplitIF::AltarelliParisi(vector<double> invariants, vector<double>,
  vector<int> helBef, vector<int> helNew) {

  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];

  // Sanity check. Require positive invariants.
  if ((saj <= 0.0) || (sjk <= 0.0) || (sAK <= 0.0)) return 0.0;

  // Store helicities.
  int hBef0 = helBef[0];
  int hBef1 = helBef[1];
  int hNew0 = helNew[0];
  int hNew1 = helNew[1];
  int hNew2 = helNew[2];

  double AP(-1.);
  // There is only the sjk collinear limit
  if (hNew0 == hBef0)
    AP = dglapPtr->Pg2qq(zB(invariants),hBef1,hNew2,hNew1)/sjk;

  return AP;
}

//==========================================================================

// Class AntQGemitIFsec, derived class for sector initial-final antenna
// functions.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntQGemitIFsec::antFun(vector<double> invariants,
  vector<double> mNew, vector<int> helBef, vector<int> helNew) {

  // Check if helicity vectors empty.
  double ant = AntQGemitIF::antFun(invariants, mNew, helBef, helNew);
  if (helBef.size() < 2) {helBef.push_back(9); helBef.push_back(9);}
  if (helNew.size() < 3) {
    helNew.push_back(9); helNew.push_back(9); helNew.push_back(9);}

  // Save invariants.
  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];
  double sak = sAK - saj + sjk;
  double yaj = saj/(sAK + sjk);
  double yjk = sjk/(sAK + sjk);
  double yak = sak/(sAK + sjk);

  // Check if j has same helicity as parent final-state gluon.
  int hG = helBef[1];
  int hjNow = helNew[1];
  if (hG == hjNow) {
    // Define j<->k symmetrisation term with sector damp parameter.
    sak += sectorDampSav * sjk;
    vector<double> invariantsSym = {sAK, sak, sjk};
    // Save swapped helicities.
    vector<int> helSym = helNew;
    helSym[1] = helNew[2];
    helSym[2] = helNew[1];
    ant += AntQGemitIF::antFun(invariantsSym, mNew, helBef, helSym);

    // Ensure positivity over all of phase space.
    ant += 1./sAK * (yak + yjk);
  }

  // Subleading colour correction has to be applied after symmetrisation!
  if (modeSLC >= 2)
    ant *= CF/chargeFacSav * (1 - yaj)/(2 - yaj - yjk)
      + CA/chargeFacSav * (1 - yjk)/(2 - yaj - yjk);

  return ant;

}

//==========================================================================

// Class AntGGemitIFsec, sector initial-final antenna function.

//--------------------------------------------------------------------------

// The antenna function [GeV^-2].

double AntGGemitIFsec::antFun(vector<double> invariants,
  vector<double> mNew, vector<int> helBef, vector<int> helNew) {

  // Check if helicity vectors empty.
  double ant = AntGGemitIF::antFun(invariants, mNew, helBef, helNew);
  if (helBef.size() < 2) {helBef.push_back(9); helBef.push_back(9);}
  if (helNew.size() < 3) {
    helNew.push_back(9); helNew.push_back(9); helNew.push_back(9);}

  // Save invariants.
  double sAK = invariants[0];
  double saj = invariants[1];
  double sjk = invariants[2];
  double sak = sAK - saj + sjk;
  double yjk = sjk/(sAK + sjk);
  double yak = sak/(sAK + sjk);

  // Check if j has same helicity as parent final-state gluon.
  int hG = helBef[1];
  int hjNow = helNew[1];
  if ( hG == hjNow ) {
    // Define j<->k symmetrisation term with sector damp parameter.
    sak += sectorDampSav * sjk;
    vector<double> invariantsSym = {sAK, sak, sjk};
    // Save swapped helicities.
    vector<int> helSym = helNew;
    helSym[1] = helNew[2];
    helSym[2] = helNew[1];
    ant += AntGGemitIF::antFun(invariantsSym, mNew, helBef, helSym);

    // Ensure positivity over all of phase space.
    ant += 1./sAK * (yak + yjk);
  }

  return ant;

}

//==========================================================================

// Class AntXGsplitIFsec, sector initial-final antenna function. Gluon
// splitting in the final state.

//--------------------------------------------------------------------------

// The antenna function, just 2*global [GeV^-2].

double AntXGsplitIFsec::antFun(vector<double> invariants, vector<double> mNew,
  vector<int> helBef, vector<int> helNew) {
  return 2*AntXGsplitIF::antFun(invariants,mNew,helBef,helNew);}

//==========================================================================

// The AntennaSetFSR class. Simple container of FF and RF antenna functions.

//--------------------------------------------------------------------------

// Initialize pointers.

void AntennaSetFSR::initPtr(Info* infoPtrIn, DGLAP* dglapPtrIn) {

  infoPtr      = infoPtrIn;
  particleDataPtr = infoPtr->particleDataPtr;
  settingsPtr     = infoPtr->settingsPtr;
  rndmPtr         = infoPtr->rndmPtr;
  dglapPtr        = dglapPtrIn;
  isInitPtr       = true;

}

//--------------------------------------------------------------------------

// Initialize antenna set (optionally with min or max variation).

void AntennaSetFSR::init() {

  // Check pointers and initialization.
  if (!isInitPtr) {
    printOut(__METHOD_NAME__, "Cannot initialize, pointers not set.");
    return;
  }
  verbose = settingsPtr->mode("Vincia:verbose");
  if (isInit) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Already initialized antenna set.");
    return;
  }

  // Create antenna objects (in map, so indices don't have to be consecutive).
  antFunPtrs.clear();
  bool sectorShower = settingsPtr->flag("Vincia:sectorShower");
  antFunPtrs[QQemitFF] = sectorShower ? new AntQQemitFFsec() :
    new AntQQemitFF();
  antFunPtrs[QGemitFF] = sectorShower ? new AntQGemitFFsec() :
    new AntQGemitFF();
  antFunPtrs[GQemitFF] = sectorShower ?
    static_cast<AntennaFunction*>(new AntGQemitFFsec()) :
    static_cast<AntennaFunction*>(new AntGQemitFF());
  antFunPtrs[GGemitFF] = sectorShower ? new AntGGemitFFsec() :
    new AntGGemitFF();
  antFunPtrs[GXsplitFF]= sectorShower ? new AntGXsplitFFsec() :
    new AntGXsplitFF();
  // Add RF antenna functions.
  antFunPtrs[QQemitRF] = new AntQQemitRF();
  antFunPtrs[QGemitRF] = sectorShower ?
    static_cast<AntennaFunction*>(new AntQGemitRFsec()) :
    static_cast<AntennaFunction*>(new AntQGemitRF());
  antFunPtrs[XGsplitRF] = sectorShower?
    static_cast<AntennaFunction*>(new AntXGsplitRFsec()) :
    static_cast<AntennaFunction*>(new AntXGsplitRF());
  if (verbose >= REPORT)
    printOut(__METHOD_NAME__, "Defined new antFunPtrs");

  // Loop through antFunPtrs and initialize.
  for (auto antFunPtrIt = antFunPtrs.begin(); antFunPtrIt != antFunPtrs.end();
       ++antFunPtrIt) {

    // Initialize antenna pointers, antenna.
    AntennaFunction* antFunPtr = antFunPtrIt->second;
    antFunPtr->initPtr(infoPtr, dglapPtr);
    bool pass = antFunPtr->init();

    // Optionally check antenna (singularities, positivity, etc).
    if (settingsPtr->flag("Vincia:checkAntennae"))
      pass = pass && antFunPtr->check();

    // Everything OK?
    if (pass) {
      if (verbose >= REPORT) printOut(__METHOD_NAME__,
        "Added to antenna list: " + antFunPtr->vinciaName());
    }
    else if (verbose >= QUIET)
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": "
        +antFunPtr->vinciaName()+" added to antenna list"
        +" (but one or more consistency checks failed).");
  }
  isInit = true;

}

//--------------------------------------------------------------------------

// Method to return a list of all AntFunTypes in this set.

vector<enum AntFunType> AntennaSetFSR::getAntFunTypes () {
  vector<enum AntFunType> antFunTypes;
  for (auto it=antFunPtrs.begin(); it!=antFunPtrs.end(); ++it)
    antFunTypes.push_back(it->first);
  return antFunTypes;
}

//==========================================================================

// The AntennaSetISR class. Simple container of II and IF antenna functions.

//--------------------------------------------------------------------------

// Initialize pointers.

void AntennaSetISR::initPtr(Info* infoPtrIn, DGLAP* dglapPtrIn) {

  infoPtr      = infoPtrIn;
  particleDataPtr = infoPtr->particleDataPtr;
  settingsPtr     = infoPtr->settingsPtr;
  rndmPtr         = infoPtr->rndmPtr;
  dglapPtr        = dglapPtrIn;
  isInitPtr       = true;

}

//--------------------------------------------------------------------------

// Initialize antenna set.

void AntennaSetISR::init() {

  // Check pointers and initialization.
  if (!isInitPtr) {
    printOut(__METHOD_NAME__, "Cannot initialize, pointers not set.");
    return;
  }
  verbose = settingsPtr->mode("Vincia:verbose");
  if (isInit) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Already initialized antenna set.");
    return;
  }

  // Create antenna objects.
  bool sectorShower = settingsPtr->flag("Vincia:sectorShower");
  antFunPtrs[QQemitII]  = new AntQQemitII();
  antFunPtrs[GQemitII]  = new AntGQemitII();
  antFunPtrs[GGemitII]  = new AntGGemitII();
  antFunPtrs[QXsplitII] = new AntQXsplitII();
  antFunPtrs[GXconvII]  = new AntGXconvII();
  antFunPtrs[QQemitIF]  = new AntQQemitIF();
  antFunPtrs[QGemitIF]  = sectorShower ? new AntQGemitIFsec() :
    new AntQGemitIF();
  antFunPtrs[GQemitIF]  = new AntGQemitIF();
  antFunPtrs[GGemitIF]  = sectorShower ? new AntGGemitIFsec() :
    new AntGGemitIF();
  antFunPtrs[QXsplitIF] = new AntQXsplitIF();
  antFunPtrs[GXconvIF]  = new AntGXconvIF();
  antFunPtrs[XGsplitIF] = sectorShower ? new AntXGsplitIFsec() :
    new AntXGsplitIF();

  // Loop through antFunPtrs and initialize.
  for (auto antFunPtrIt = antFunPtrs.begin(); antFunPtrIt != antFunPtrs.end();
       ++antFunPtrIt) {

    // Initialize antenna pointers, antenna.
    AntennaFunction* antFunPtr = antFunPtrIt->second;
    antFunPtr->initPtr(infoPtr, dglapPtr);
    bool pass = antFunPtr->init();

    // Optionally check antenna (singularities, positivity, etc.).
    if (settingsPtr->flag("Vincia:checkAntennae"))
      pass = pass && antFunPtr->check();

    // DEBUG info.
    if (pass) {
      if (verbose >= DEBUG) printOut(__METHOD_NAME__, "Added to antenna list: "
        + antFunPtr->vinciaName());
    } else if (verbose >= QUIET)
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__
        +": one or more consistency checks failed.");
  }
  isInit = true;

}

//--------------------------------------------------------------------------

// Method to return a list of all AntFunTypes in this set.

vector<enum AntFunType> AntennaSetISR::getAntFunTypes () {
  vector<enum AntFunType> antFunTypes;
  for (auto it=antFunPtrs.begin(); it!=antFunPtrs.end(); ++it)
    antFunTypes.push_back(it->first);
  return antFunTypes;
}

//==========================================================================

// Class MECs, for computing matrix-element corrections to antenna
// functions.

//--------------------------------------------------------------------------

// Initialize pointers.

void MECs::initPtr(Info* infoPtrIn, ShowerMEs* mg5mesPtrIn,
  VinciaCommon* vinComPtrIn, Resolution* resPtrIn) {

  infoPtr          = infoPtrIn;
  particleDataPtr  = infoPtr->particleDataPtr;
  partonSystemsPtr = infoPtr->partonSystemsPtr;
  rndmPtr          = infoPtr->rndmPtr;
  settingsPtr      = infoPtr->settingsPtr;
  mg5mesPtr        = mg5mesPtrIn;
  vinComPtr        = vinComPtrIn;
  resolutionPtr    = resPtrIn;
  isInitPtr        = true;

}

//--------------------------------------------------------------------------

// Initialize.

void MECs::init() {

  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "begin", dashLen);

  // MEC settings.
  verbose            = settingsPtr->mode("Vincia:verbose");
  modeMECs           = settingsPtr->mode("Vincia:modeMECs");
  maxMECs2to1        = settingsPtr->mode("Vincia:maxMECs2to1");
  maxMECs2to2        = settingsPtr->mode("Vincia:maxMECs2to2");
  maxMECs2toN        = settingsPtr->mode("Vincia:maxMECs2toN");
  maxMECsResDec      = settingsPtr->mode("Vincia:maxMECsResDec");
  maxMECsMPI         = settingsPtr->mode("Vincia:maxMECsMPI");
  nFlavZeroMass      = settingsPtr->mode("Vincia:nFlavZeroMass");
  sizeOutBornSav.clear();

  // Some settings hardcoded for now.
  matchingFullColour = false;
  matchingScaleIsAbs = false;
  matchingScale      = 0.05;
  q2Match            = pow2(matchingScale);
  matchingRegOrder   = 2;
  matchingRegShape   = 1;
  matchingIRcutoff   = 2.;

  // Sanity checks.
  // MECs currently not supported.
  if (modeMECs > 0) {
    stringstream ss;
    ss << ": Matrix element corretions not yet supported.";
    infoPtr->errorMsg("Error in " + __METHOD_NAME__ + ss.str());
    isInit = false;
    return;
  }

  // Initialise MG5 interface
  if (mg5mesPtr->initVincia()) {
    //TODO fix colour depth in MG5 interface.
    mg5mesPtr->setColourDepthVincia(1);
  } else {
    if (verbose >= REPORT) printOut(__METHOD_NAME__,
      "Could not initialise VinciaMG5MEs interface.");
    // Check if we wanted to do matching.
    if (modeMECs > 0) {
      // Abort. We should not try to do anything.
      isInit = false;
      return;
    }
    // Set everything explicitly to negative values, but continue.
    else {
      maxMECs2to1   = -1;
      maxMECs2to2   = -1;
      maxMECs2toN   = -1;
      maxMECsResDec = -1;
      maxMECsMPI    = -1;
    }
  }
  isInit = true;

  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "end", dashLen);
}

//--------------------------------------------------------------------------

// Function to return ME class (Born type) for a parton configuration.

bool MECs::prepare(const int iSys, Event& event) {

  // If we switched off MECs completely, return here.
  if (modeMECs < 0) return false;

  // Initialise for no MECs, then check if MECs should be applied.
  int nAll    = partonSystemsPtr->sizeAll(iSys);
  int nOut    = partonSystemsPtr->sizeOut(iSys);
  bool isHard = (iSys == 0 && nAll - nOut == 2);
  bool isMPI  = (iSys >= 1 && nAll - nOut == 2);

  // Check if MECs are switched on for this type of Born system.
  if (isHard) {
    if (nOut == 1 && maxMECs2to1 < 0) return false;
    else if (nOut == 2 && maxMECs2to2 < 0) return false;
    else if (nOut >= 3 && maxMECs2toN < 0) return false;
  } else if (isMPI) {
    if (infoPtr->nMPI() > maxMECsMPI) return false;
  } else {
    if (maxMECsResDec < 0) return false;
  }

  // Save Born multiplicity and number of QCD particles in Born.
  sizeOutBornSav[iSys] = nOut;
  sysToBornMultQCD[iSys] = 0;
  int sizeSys = partonSystemsPtr->sizeAll(iSys);
  for (int iPtcl(0); iPtcl<sizeSys; ++iPtcl) {
    int iNow = partonSystemsPtr->getAll(iSys, iPtcl);
    if (event[iNow].isQuark() || event[iNow].isGluon())
      sysToBornMultQCD[iSys]++;
  }
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Saved Born with " << nOut << " outgoing particles"
       << " and " << sysToBornMultQCD[iSys] << " QCD particles.";
    printOut(__METHOD_NAME__, ss.str());
  }

  // When using relative matching scale, save hard scale.
  if (!matchingScaleIsAbs) {
    if (!saveHardScale(iSys, event)) {
      if (verbose >= NORMAL) {
        stringstream ss;
        ss << "Could not save hard scale for system " << iSys;
        infoPtr->errorMsg("Error in " + __METHOD_NAME__ + ss.str());
      }
      return false;
    }
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Saved hard scale " << sqrt(sysToHardScale[iSys])
         << " GeV for system " << iSys << ".";
      printOut(__METHOD_NAME__, ss.str());
    }
  }

  // Save that we do not yet have calculated any matrix elements.
  hasME2now[iSys]  = false;
  hasME2post[iSys] = false;

  // Check whether MG5MEs interface has the process.
  return meAvailable(iSys, event);

}

//--------------------------------------------------------------------------

// Function to assign helicities to particles (using MEs).

bool MECs::polarise(const int iSys, Event& event, bool force) {

  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "begin", dashLen);

  // First check if we should be doing anything at all.
  if (partonSystemsPtr->hasInAB(iSys)) {
    // Hard System.
    if (iSys == 0) {
      int nOut = partonSystemsPtr->sizeOut(iSys);
      if (nOut==1 && maxMECs2to1 <= -1) return false;
      if (nOut==2 && maxMECs2to2 <= -1) return false;
      if (nOut>=3 && maxMECs2toN <= -1) return false;
    // Hardest MPI system
    } else if (iSys == 1 && maxMECsMPI <= -1) return false;
    // Further MPI systems just use unpolarised showers.
    else return false;
  // Resonance-Decay systems
  } else {if (maxMECsResDec <= -1) return false;}

  // Check if all particles in state already have assigned helicities.
  // Else see if we can assign helicities.
  if (!isPolarised(iSys, event) || force) {

    // Extract particles to use for ME evaluations (incoming first)
    vector<Particle> state = vinComPtr->makeParticleList(iSys, event);

    // Check if we can polarise this state.
    if (!polarise(state, force)) return false;

    // Update particles in event record: incoming.
    int nIn = 2;
    if (partonSystemsPtr->hasInAB(iSys)) {
      event[partonSystemsPtr->getInA(iSys)].pol(state[0].pol());
      event[partonSystemsPtr->getInB(iSys)].pol(state[1].pol());
    }
    else {
      nIn = 1;
      event[partonSystemsPtr->getInRes(iSys)].pol(state[0].pol());
    }

    // Update particles in event record: outgoing.
    for (int iOut = 0; iOut < partonSystemsPtr->sizeOut(iSys); ++iOut) {
      int iEvent = partonSystemsPtr->getOut(iSys,iOut);
      int iState = iOut + nIn;
      event[iEvent].pol(state[iState].pol());
    }
  }

  // Verbose output (showing polarisations).
  if (verbose >= DEBUG) {
    event.list(true);
    printOut(__METHOD_NAME__, "end", dashLen);
  }

  // All is well.
  return true;

}

//--------------------------------------------------------------------------

bool MECs::polarise(vector<Particle>& state, bool force) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  if (state.size() <= 2) return false;

  int nIn = 0;
  for (int i=0; i<(int)state.size(); ++i) {
    if (state[i].status() < 0) ++nIn;
    if (nIn == 2) break;
  }

  // Check if MG5MEs interface can do this.
  return mg5mesPtr->selectHelicitiesVincia(state, nIn, force);

}

//--------------------------------------------------------------------------

// Check if all partons in state have helicities.
// Optionally check if *any* particle has helicity (checkAll = false).

bool MECs::isPolarised(int iSys, Event& event, bool checkAll) {

  for (int i = 0; i < partonSystemsPtr->sizeAll(iSys); ++i) {
    int ip = partonSystemsPtr->getAll(iSys,i);
    if (ip == 0) continue;
    if (event[ip].pol() != 9) {
      if (!checkAll) return true;
      else continue;
    }
    int spinType = particleDataPtr->spinType(event[ip].idAbs());
    if (spinType != 1) return false;
  }
  return true;

}

//--------------------------------------------------------------------------

// Function to determine if MECs are requested at this order for this system.

bool MECs::doMEC(int iSys, int nBranch) {

  // If we switched off MECs completely, return here.
  if (modeMECs < 0) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "MECs switched off.");
    return false;
  }

  // MECs in resonance decays.
  bool isResDec = partonSystemsPtr->hasInRes(iSys);
  if (isResDec) {
    if (nBranch <= maxMECsResDec) return true;
  }
  // MECs in Scattering Processes.
  else {
    // Hard processes.
    if (iSys == 0) {
      if (sizeOutBornSav[iSys] == 1 && nBranch <= maxMECs2to1) return true;
      if (sizeOutBornSav[iSys] == 2 && nBranch <= maxMECs2to2) return true;
      if (sizeOutBornSav[iSys] >= 3 && nBranch <= maxMECs2toN) return true;
    }
    // MPI.
    else if (iSys == 1) if (nBranch <= maxMECsMPI) return true;
  }

  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "No MECs at this order.");

  // If nobody said yes by now, return the sad news.
  return false;

}

//--------------------------------------------------------------------------

// Check whether we have a matrix element for a given configuration.

bool MECs::meAvailable(int iSys, const Event& event) {
  // Make vectors of ID codes.
  vector<int> idIn, idOut;
  if (partonSystemsPtr->hasInAB(iSys)) {
    idIn.push_back(event[partonSystemsPtr->getInA(iSys)].id());
    idIn.push_back(event[partonSystemsPtr->getInB(iSys)].id());
  }
  else if (partonSystemsPtr->hasInRes(iSys))
    idIn.push_back(event[partonSystemsPtr->getInRes(iSys)].id());
  for (int i = 0; i < partonSystemsPtr->sizeOut(iSys); ++i)
    idOut.push_back(event[partonSystemsPtr->getOut(iSys, i)].id());
  set<int> sChan;
  bool avail = mg5mesPtr->hasProcessVincia(idIn, idOut, sChan);
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Matrix element for ";
    for (auto& i : idIn) ss << i << " ";
    ss << "-> ";
    for (auto& i : idOut) ss << i << " ";
    ss << (avail ? "is available." : "not available.");
    printOut(__METHOD_NAME__, ss.str());
  }
  return avail;
}

bool MECs::meAvailable(const vector<Particle>& state) {
  // Make vectors of ID codes.
  vector<int> idIn, idOut;
  for (const auto& p : state) {
    if (!p.isFinal()) idIn.push_back(p.id());
    else idOut.push_back(p.id());
  }
  set<int> sChan;
  return mg5mesPtr->hasProcessVincia(idIn, idOut, sChan);
}

//--------------------------------------------------------------------------

// Get squared matrix element.

double MECs::getME2(const vector<Particle>& state, int nIn) {
  return mg5mesPtr->me2Vincia(state, nIn);}

double MECs::getME2(const int iSys, const Event& event) {
  vector<Particle> state = vinComPtr->makeParticleList(iSys, event);
  bool isResDec = partonSystemsPtr->hasInRes(iSys);
  return (isResDec) ?
    mg5mesPtr->me2Vincia(state, 1 ) : mg5mesPtr->me2Vincia(state, 2);
}

//--------------------------------------------------------------------------

// Get matrix element correction factor.

double MECs::getMECSector(int iSys, const vector<Particle>& stateNow,
  const vector<Particle>& statePost, VinciaClustering& clus) {

  // Sanity check.
  if (statePost.size() != stateNow.size() + 1) {
    stringstream ss;
    ss << "Matrix element corrections for direct 2->"
       << 2 + statePost.size() - stateNow.size() << " not yet implemented.";
    infoPtr->errorMsg("Error in " + __METHOD_NAME__, ss.str());
    hasME2post[iSys] = false;
    return 1.;
  }

  // DEBUG output.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__,"Computing MEC factor for:");
    vinComPtr->list(statePost, "Post Branching", false);
    vinComPtr->list(stateNow, "Current");
  }

  // Check whether we have matrix elements for the two states.
  if (!meAvailable(stateNow)) {
    stringstream ss;
    ss << ": Matrix element for current configuration not available.";
    infoPtr->errorMsg("Warning in " + __METHOD_NAME__ + ss.str());
    return 1.;
  }
  if (!meAvailable(statePost)) {
    stringstream ss;
    ss << ": Matrix element for post-branching configuration not available.";
    infoPtr->errorMsg("Warning in " + __METHOD_NAME__ + ss.str());
    hasME2post[iSys] = false;
    return 1.;
  }

  // Check whether we are below IR cutoff.
  double qNow = sqrt(resolutionPtr->q2evol(clus));
  if (qNow < matchingIRcutoff) {
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Evolution scale pT = " << qNow
         << " below IR cutoff (" << matchingIRcutoff << ").";
      printOut(__METHOD_NAME__,ss.str());
    }
    hasME2post[iSys] = false;
    return 1.;
  }

  // If regularising this order, get matching weight.
  double wMatch = 1.;
  if (doRegMatch(iSys, statePost)) {
    wMatch = getMatchReg(iSys, clus);
    // Sanity checks.
    if (wMatch < 0.) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Negative matching weight",
        ": ("+num2str(wMatch,6)+")");
      return 1.;
    }
    if (wMatch > 1.) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Matching weight above unity",
        ": ("+num2str(wMatch,6)+")");
      return 1.;
    }

    // Check whether we want to calculate a MEC.
    if (wMatch == 0.) {
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__, "Below matching scale. No MEC calculated.");
      hasME2post[iSys] = false;
      return 1.;
    }
    else if (verbose >= DEBUG) {
      string reg = " (cutoff)";
      if (matchingRegShape == 1) reg = " (sigmoid regulator)";
      else if (matchingRegShape == 2) reg = " (linear regulator)";
      else if (matchingRegShape == 3) reg = " (logarithmic regulator)";
      stringstream ss;
      ss << "Matching weight: " << wMatch << reg;
      printOut(__METHOD_NAME__, ss.str());
    }
  }
  else if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "Not regularising this order.");

  // Get number of incoming particles.
  bool isResDec = partonSystemsPtr->hasInRes(iSys);
  int nIn = isResDec ? 1 : 2;

  // Get matrix element for post-branching state.
  me2post[iSys] = getME2(statePost, nIn);
  // Sanity check.
  if (me2post[iSys] <= 0.) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Negative post-branching matrix element squared.");
    hasME2post[iSys] = false;
    return 1.;
  }
  else if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Calculated new post-branching ME2 in system "
       << iSys << " (ME2 = " << num2str(me2post[iSys],9) << ").";
    printOut(__METHOD_NAME__, ss.str());
  }
  hasME2post[iSys] = true;

  // Get matrix element for current state
  // (only if state has changed since last matching).
  if (!hasME2now[iSys]) {
    me2now[iSys] = getME2(stateNow, nIn);
    // Sanity check.
    if (me2now[iSys] <= 0.) {
      if (verbose >= NORMAL)
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Negative matrix element squared.");
      return 1.;
    }
    else if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Calculated new ME2 for current state in system "
         << iSys << " (ME2 = " << num2str(me2now[iSys],9) << ").";
      printOut(__METHOD_NAME__, ss.str());
    }
    hasME2now[iSys] = true;
  }
  else if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Using saved ME2 for current state in system "
       << iSys << " (ME2 = " << num2str(me2now[iSys],9) << ").";
    printOut(__METHOD_NAME__, ss.str());
  }

  // Calculate antenna function.
  double ant = getAntApprox(clus);
  // Sanity check.
  if (ant <= 0.) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Negative antenna function",
        " ("+clus.getAntName()+" = "+num2str(ant,6)+")", true);
    return 1.;
  }
  else if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Antenna function in system "
       << iSys << " (ant = " << num2str(ant,6) << ").";
    printOut(__METHOD_NAME__, ss.str());
  }


  // Calculate colour weight. Just 1, when not matching at full colour.
  double wCol = getColWeight(statePost);
  // Sanity check.
  if (wCol < 0.) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Negative colour weight.");
    return 1.;
  }
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Colour weight: " << wCol
       << (matchingFullColour == 1 ? " (Vincia Colour)." : " (LC).");
    printOut(__METHOD_NAME__, ss.str());
  }


  // Calculate shower-subtracted matrix element correction.
  // Note: just a simple ratio for the sector shower.
  double appr = ant * me2now[iSys];
  double mec = (wCol * me2post[iSys]) - appr;

  // Calculate correction factor including matching weight.
  double pMEC = 1. + wMatch * mec / appr;

  return pMEC;

}

//--------------------------------------------------------------------------

// Upon branching, save last post-branching ME2.

void MECs::hasBranched(int iSys) {
  // DEBUG print out.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Saving last post-branching ME2 as current one (ME2 = "
       << num2str(me2post[iSys],9) << ").";
    printOut(__METHOD_NAME__, ss.str());
  }

  // The last post-branching ME2 is now the current ME2.
  if (hasME2post[iSys]) {
    me2now[iSys] = me2post[iSys];
    hasME2now[iSys] = true;
  }
  else hasME2now[iSys] = false;
}

//--------------------------------------------------------------------------

// Print header information.

void MECs::header() {

  // Front matter.
  bool doPolarise = (maxMECs2to1 >= 0) || (maxMECs2to2 >= 0)
    || (maxMECs2toN >= 0) || (maxMECsResDec >= 0);
  cout << " |\n | MECs (-1:off, 0:selectHelicities, >=1:nMECs): ";
  if (doPolarise) {
    cout << endl;
    cout << " |                 modeMECs              = "
         << num2str(modeMECs, 9) << "\n"
         << " |                 maxMECs2to1           = "
         << num2str(maxMECs2to1, 9) << "\n"
         << " |                 maxMECs2to2           = "
         << num2str(maxMECs2to2, 9) << "\n"
         << " |                 maxMECs2toN           = "
         << num2str(maxMECs2toN, 9) << "\n"
         << " |                 maxMECsResDec         = "
         << num2str(maxMECsResDec, 9) <<"\n";

    // Setings.
    if (modeMECs > 0) {
      cout << " |                 matchingFullColour    = "
           << bool2str(matchingFullColour, 9) << "\n";
      cout << " |                 regOrder              = "
           << num2str(matchingRegOrder, 9) << endl;
      if (matchingScaleIsAbs)
        cout << " |                 matchingScale (GeV)   = "
             << num2str(matchingScale, 9) << endl;
      else
        cout << " |                 matchingScale (Ratio) = "
             << num2str(matchingScale, 9) << endl;
      if (verbose >= REPORT)
        cout << " |                 regShape              = "
             << num2str(matchingRegShape, 9) << endl;
      cout << " |                 IR cutoff             = "
           << num2str(matchingIRcutoff, 9) << endl;
    }

    // Print MG5MEs reference.
    cout << " | The MADGRAPH Matrix Element interface relies on:" << endl
         << " |    MADGRAPH 5 : Alwall et al., JHEP06(2011)128, "
         << "arXiv:1106.0522 " << endl;
  }
  else cout << bool2str(false, 9) << "\n";

}

//--------------------------------------------------------------------------

// Private functions.

//--------------------------------------------------------------------------

// Save hard scale in current process.

bool MECs::saveHardScale(int iSys, Event& /*event*/) {

  // Currently only sHat.
  //TODO: better/different choices? pTHat for 2 -> 2?
  double qHard = partonSystemsPtr->getSHat(iSys);

  // Sanity check.
  if (qHard <= 0.) return false;
  sysToHardScale[iSys] = qHard;

  return true;
}

//--------------------------------------------------------------------------

bool MECs::doRegMatch(int iSys, const vector<Particle>& state) {
  // Not regularising corrections at all.
  if (matchingRegOrder == 0) return false;

  // Get number of QCD particles.
  int nQCDnow = 0;
  for (const auto& p : state)
    if (p.isQuark() || p.isGluon()) ++nQCDnow;

  if (nQCDnow - sysToBornMultQCD[iSys] >= matchingRegOrder) return true;
  return false;
}

//--------------------------------------------------------------------------

// Matching regulator.

double MECs::getMatchReg(int iSys, const VinciaClustering& clus) {

  // Get current scale.
  double q2Now = clus.Q2evol;
  if (!matchingScaleIsAbs) q2Now /= sysToHardScale[iSys];
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "MEC requested at scale qNow = "
       << sqrt(q2Now) << (matchingScaleIsAbs ? " GeV." : " (relative).");
    printOut(__METHOD_NAME__, ss.str());
  }

  // Sharp cutoff.
  if (matchingRegShape == 0) {
    if (q2Now < q2Match) return 0.;
    else return 1.;
  }
  // Sigmoidal interpolation.
  else if (matchingRegShape == 1) {
    if (q2Now < q2Match/2.) return 0.;
    else if (q2Now > 2.*q2Match) return 1.;
    else return 1./(1. + exp(16.*(1. - q2Now/q2Match)));
  }
  // Linear interpolation.
  else if (matchingRegShape == 2) {
    if (q2Now < q2Match/2.) return 0.;
    else if(q2Now > 2.*q2Match) return 1.;
    else return 2./3.*q2Now/q2Match - 1./3.;
  }
  // Logarithmic interpolation.
  else if (matchingRegShape == 3) {
    if (q2Now < q2Match/2.) return 0.;
    else if(q2Now > 2.*q2Match) return 1.;
    else return ( log(q2Now/q2Match)/log(2.) + 1.) / 2.;
  }
  // ...implement other shapes here.

  // No supported regulator shape requested.
  if (verbose >= NORMAL) {
    stringstream ss;
    ss << ": Unsupported matching regulator shape "
       << matchingRegShape << " requested.";
    infoPtr->errorMsg("Error in " + __METHOD_NAME__ + ss.str());
  }
  return 0.;
}

//--------------------------------------------------------------------------

// Antenna function including colour factor.

double MECs::getAntApprox(const VinciaClustering& clus) {

  double ant = -1.;

  // Sanity check: are invariants, masses, and helicities set?
  if (clus.invariants.size() < 3) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,
        ": Post-branching invariants not set in clustering.");
    return ant;
  }
  if (clus.massesChildren.size() < 3) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,
        ": Post-branching masses not set in clustering.");
    return ant;
  }
  if (clus.helChildren.size() < 3) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,
        ": Post-branching helicities not set in clustering.");
    return ant;
  }

  // FSR antenna.
  if (clus.isFSR) {
    AntennaFunction* antFunPtr = getAntFunPtrFSR(clus.antFunType);
    if (antFunPtr == nullptr) {
      if (verbose >= NORMAL)
        infoPtr->errorMsg("Error in "+__METHOD_NAME__,
          ": Unknown FSR antenna function with index "
          + num2str(clus.antFunType,2));
      return ant;
    }
    ant = antFunPtr->antFun(clus.invariants, clus.massesChildren,
      clus.helMothers, clus.helChildren);
    ant *= antFunPtr->chargeFac();
  }
  // ISR antenna.
  else {
    AntennaFunction* antFunPtr = getAntFunPtrISR(clus.antFunType);
    if (antFunPtr == nullptr) {
      if (verbose >= NORMAL)
        infoPtr->errorMsg("Error in "+__METHOD_NAME__,
          ": Unknown ISR antenna function with index "
          + num2str(clus.antFunType,2));
      return ant;
    }
    ant = antFunPtr->antFun(clus.invariants, clus.massesChildren,
      clus.helMothers, clus.helChildren);
    ant *= antFunPtr->chargeFac();
  }

  return ant;
}

//--------------------------------------------------------------------------

// Colour weight to match at full colour.

double MECs::getColWeight(const vector<Particle>& state) {

  // If only leading colour requested return here.
  if (!matchingFullColour) return 1.;

  // Get FC colour summed ME2.
  // Second argument tells MG5 to sum over all colour configurations.
  double me2FC = getME2(state, true);
  if (verbose >= DEBUG) {
    stringstream ss;
  }

  // Get LC colour summed ME2.
  // Temporarily set colourdepth to 0 in MG5 interface.
  mg5mesPtr->setColourDepthVincia(0);
  double me2LC = getME2(state, true);
  mg5mesPtr->setColourDepthVincia(1);
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "ME2(LC) = " << me2LC
       << ", ME2(FC) = " << me2FC
       << ", ME2(FC)/ME2(LC) = " << me2FC/me2LC;
    printOut(__METHOD_NAME__, ss.str());
  }

  return me2FC/me2LC;
}

//==========================================================================

} // end namespace Pythia8
