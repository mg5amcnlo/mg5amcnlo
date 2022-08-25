// VinciaTrialGenerators.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

#include "Pythia8/VinciaTrialGenerators.h"
#include "Pythia8/MathTools.h"

namespace Pythia8 {

using namespace VinciaConstants;

//==========================================================================

// TrialGenerator base class functions.

//--------------------------------------------------------------------------

// Add a zeta generator for a given sector.

void TrialGenerator::addGenerator(ZetaGeneratorSet& zetaGenSet,
  Sector sector) {
  ZetaGenerator* zGenPtr = zetaGenSet.getZetaGenPtr(branchType,sector);
  if (zGenPtr != nullptr) zetaGenPtrs[sector] = zGenPtr;
}

//--------------------------------------------------------------------------

// Initialise the zeta generators.

void TrialGenerator::setupZetaGens(ZetaGeneratorSet& zetaGenSet) {

  // Check for incompatible type.
  if (zetaGenSet.getTrialGenType() != trialGenTypeSav) return;

  // Sector shower: add one generator per sector (if it exists).
  if (isSector) {
    addGenerator(zetaGenSet,Sector::ColI);
    addGenerator(zetaGenSet,Sector::Default);
    addGenerator(zetaGenSet,Sector::ColK);
  } else if (trialGenTypeSav == TrialGenType::FF ||
    trialGenTypeSav == TrialGenType::RF) {
    // For global FF and RF only one generator needed.
    addGenerator(zetaGenSet);
  } else if (trialGenTypeSav == TrialGenType::IF) {
    // For global IF need two generators (initial-state leg sectorised).
    addGenerator(zetaGenSet,Sector::ColI);
    addGenerator(zetaGenSet,Sector::Default);
  } else if (trialGenTypeSav == TrialGenType::II) {
    // For global II need three generators (initial-state legs sectorised).
    addGenerator(zetaGenSet,Sector::ColI);
    addGenerator(zetaGenSet,Sector::Default);
    addGenerator(zetaGenSet,Sector::ColK);
  }
  isInit = true;

}

//--------------------------------------------------------------------------

// Re-calculate the current zeta limits using absolute minimum q2.
// N.B. Base class doesn't use x fractions.

void TrialGenerator::reset(double Q2min, double s,
  const vector<double>& masses, AntFunType antFunType, double, double) {

  if (!isInit) return;

  // Throw away any stored trials.
  resetTrial();

  // Calculate common factors.
  calcKallenFac(s, masses);
  calcRpdf(vector<double>());

  sAntSav = s;
  massesSav = masses;

  // Loop over sectors (generators).
  for (auto iGen = zetaGenPtrs.begin(); iGen!= zetaGenPtrs.end(); ++iGen) {

    Sector sectorNow = iGen->first;
    ZetaGenerator* zGenPtr = iGen->second;

    // Check if this sector is active and save.
    bool isActive = zGenPtr != nullptr && zGenPtr->isActive(antFunType);
    isActiveSector[sectorNow] = isActive;
    if (!isActive) continue;

    // Calculate hull of current phase space limits.
    double zMin = zGenPtr->getzMinHull(Q2min,sAntSav,masses);
    double zMax = zGenPtr->getzMaxHull(Q2min,sAntSav,masses);

    // Save hull of zeta limits.
    zetaLimits[sectorNow] = make_pair(zMin,zMax);
  }

}

//--------------------------------------------------------------------------

// Generate the next trial scale.

double TrialGenerator::genQ2(double Q2MaxNow, Rndm* rndmPtr,
  const EvolutionWindow* evWindowPtrIn, double colFac,
  double wtIn, Info* infoPtr, int verboseIn) {

  if (!isInit) {
    if (verboseIn >= NORMAL) infoPtr->errorMsg("Error in "+__METHOD_NAME__,
        "Trial generator is not initialised!");
    return 0.;
  }
  if (hasTrial) {
    if (verboseIn >= DEBUG) printOut(__METHOD_NAME__,"Returning saved trial.");
    return q2Sav;
  }

  // Reset.
  q2Sav = 0.;

  // Save common variables.
  colFacSav   = colFac;
  evWindowSav = evWindowPtrIn;

  // Get multiplicative prefactors to trial integral.
  double prefactor = kallenFacSav;
  prefactor *= Rpdf;
  prefactor *= colFac;
  // Possible enhancement weights.
  prefactor *= wtIn;

  // Fetch minimum q2 for this window.
  double q2MinNow = pow2(evWindowPtrIn->qMin);

  // Loop over sectors (generators).
  if (verboseIn >= DEBUG) printOut(__METHOD_NAME__,"Looping over sectors...");
  for (auto iGen = zetaGenPtrs.begin(); iGen!= zetaGenPtrs.end(); ++iGen) {

    Sector sectorNow = iGen->first;
    ZetaGenerator* zGenPtr = iGen->second;

    // Skip inactive sectors.
    if (!isActiveSector[sectorNow]) continue;

    // Re-calculate hull of current phase space limits for this window.
    double zMin = zGenPtr->getzMinHull(q2MinNow, sAntSav, massesSav);
    double zMax = zGenPtr->getzMaxHull(q2MinNow, sAntSav, massesSav);

    // Save hull of limits.
    zetaLimits[sectorNow] = make_pair(zMin,zMax);

    // Evaluate zeta integral.
    // Note: do not multiply by anything, since used for veto below.
    double Iz = zGenPtr->getIz(zMin,zMax);
    // Get additional factors for zeta integral.
    double kernel = Iz * zGenPtr->getConstFactor(sAntSav, massesSav);

    // Check if phase space is closed.
    if (kernel<=0.) {
      if (verboseIn >= DEBUG)
        printOut(__METHOD_NAME__,"Phase space is closed.");
      continue;
    }

    // Convert to global trial.
    if (!isSector) kernel *= zGenPtr->globalMultiplier();

    // Now generate q2.
    double q2Sector = Q2MaxNow;

    // Optimise by checking narrower (physical) zeta hull after trial.
    bool acceptHull = false;
    while (!acceptHull) {

      // Get log of random number.
      double lnR = log(rndmPtr->flat());

      // Fixed alphaS.
      if (evWindowPtrIn->runMode <= 0) {
        // Use max possible value for alphaS.
        double pref = prefactor * evWindowPtrIn->alphaSmax;
        // Inverse of Q2 integral for fixed alphaS.
        q2Sector = q2Sector*exp(lnR/(pref*kernel));
      } else {
        // Calculate factors for running alphas.
        double muRScaleMod = evWindowPtrIn->kMu2/evWindowPtrIn->lambda2;
        double pref = prefactor / evWindowPtrIn->b0;
        // Inverse of Q2 integral for running alphas.
        double logQ2Ratio = exp(lnR/(pref*kernel));
        double logQ2maxFactor = log(q2Sector*muRScaleMod);
        q2Sector = exp(logQ2maxFactor*logQ2Ratio)/muRScaleMod;
      }

      // If we dropped below window edge, just accept and continue.
      if (q2Sector <= q2MinNow) {
        acceptHull = true;
      } else {
        // Else check if we are inside the narrower zeta range defined
        // by this q2.
        double zMinPhys = zGenPtr->getzMinHull(q2Sector, sAntSav, massesSav);
        double zMaxPhys = zGenPtr->getzMaxHull(q2Sector, sAntSav, massesSav);
        double pAcceptHull = zGenPtr->getIz(zMinPhys,zMaxPhys)/Iz;
        // Accept this range with probability IzPhys/Iz:
        if (rndmPtr->flat() < pAcceptHull) {
          acceptHull = true;
          // Update saved limits.
          zetaLimits[sectorNow] = make_pair(zMinPhys,zMaxPhys);
        }
      }

      // Safety check.
      if (q2Sector > Q2MaxNow) {
        if (verboseIn >= DEBUG) {
          infoPtr->errorMsg("Error in "+__METHOD_NAME__,
            "Generated impossible Q2");
          cout << "   evolution mode = " << evWindowPtrIn->runMode << endl
               << "   prefactor = " << prefactor << " kernel = " << kernel
               << "   ln(R) =  " << lnR << endl
               << "   kmu2 = " << evWindowPtrIn->kMu2 << " lambda2 = "
               << evWindowPtrIn->lambda2 << endl;
        }
        q2Sector = -1.;
      } else if (verboseIn >= DEBUG) {
        stringstream ss;
        ss << "Generated a new trial with Q2 = "
           << q2Sector << " in Sector: " << int(sectorNow);
        printOut(__METHOD_NAME__,ss.str());
      }
    }

    // Winner so far.
    if (q2Sector > q2Sav) {
      q2Sav = q2Sector;
      hasTrial = true;
      // Save which sector.
      sectorSav = sectorNow;
    }
  }

  if (verboseIn >= DEBUG) {
    stringstream ss;
    ss << "Winner now: Q2 = " << q2Sav << " ( " << sqrt(q2Sav)
       << ") in sector: " << int(sectorSav);
    printOut(__METHOD_NAME__,ss.str());
  }
  return q2Sav;

}


//--------------------------------------------------------------------------

// Get the invariants.

bool TrialGenerator::genInvariants(double sAnt, const vector<double>& masses,
  vector<double>& invariants, Rndm* rndmPtr, Info* infoPtr, int verboseIn) {

  if(!isInit) return false;

  if (verboseIn >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  if (q2Sav > 0. && zetaGenPtrs.find(sectorSav) != zetaGenPtrs.end()
    && zetaLimits.find(sectorSav) != zetaLimits.end()) {

    // Retrieve limits of zeta integral.
    pair<double, double> limits = zetaLimits[sectorSav];
    double zMin = limits.first;
    double zMax = limits.second;

    // Generate a value of zeta.
    double zetaNow = zetaGenPtrs[sectorSav]->genZeta(rndmPtr,zMin,zMax);
    // Check that it is inside the physical limits.
    //TODO fix xA and xB;
    double zetaMinPhys =
      zetaGenPtrs[sectorSav]->getzMin(q2Sav, sAnt, masses, 1., 1.);
    double zetaMaxPhys =
      zetaGenPtrs[sectorSav]->getzMax(q2Sav, sAnt, masses, 1., 1.);
    // Check if we generated a physical zeta, now that we know Q2.
    if (zetaNow < zetaMinPhys || zetaNow > zetaMaxPhys) {
      if (verboseIn >= DEBUG) {
        stringstream ss;
        ss << "Generated zeta outside of physical limits: "
           << num2str(zetaNow,5) << " [" << num2str(zetaMinPhys,5)
           << ", " << num2str(zetaMaxPhys,5) << "].";
        printOut(__METHOD_NAME__, ss.str());
        printOut(__METHOD_NAME__, "return false", dashLen);
      }
      return false;
    }
    if (verboseIn >= DEBUG) {
      stringstream ss;
      ss << "Generated zeta = " << zetaNow
         << " in [" << zMin << "," << zMax << "]";
      printOut(__METHOD_NAME__,ss.str());
    }

    // Calculate invariants from q2 and zeta.
    zetaGenPtrs[sectorSav]->genInvariants(q2Sav,zetaNow,sAnt,
      masses,invariants,infoPtr,verboseIn);
    if (invariants.size()==4) {
      if (verboseIn >= DEBUG) {
        stringstream ss;
        ss<< "with sAnt = "<< invariants[0]
          << "  =>  s01 = "<< invariants[1]
          << "  s12 = "<< invariants[2]
          << "  s02 = "<< invariants[3];
        printOut(__METHOD_NAME__,ss.str());
        printOut(__METHOD_NAME__, "end", dashLen);
      }
      return true;
    } else {
      if (verboseIn >= DEBUG) {
        printOut(__METHOD_NAME__,
          "Warning: fewer than 4 invariants were generated!");
        printOut(__METHOD_NAME__, "return false", dashLen);
      }
      return false;
    }
  } else {
    if (verboseIn >= DEBUG) printOut(__METHOD_NAME__,
      "return false", dashLen);
    return false;
  }

}

//--------------------------------------------------------------------------

// Calculate the trial based on invariants and saved quantities.

double TrialGenerator::aTrial(vector<double>& invariants,
  const vector<double>& masses, int verboseIn) {

  if (!isInit) return 0.;

  // Fetch colour and coupling stripped antenna function.
  double antTrial = aTrialStrip(invariants,masses,verboseIn);

  // Multiply by colour factor.
  antTrial *= colFacSav;

  // Multiply by alphaS.
  double alphaSTrial = evWindowSav->alphaSmax;
  if (evWindowSav->runMode >= 1) {
    double mu2 = q2Sav*(evWindowSav->kMu2/evWindowSav->lambda2);
    alphaSTrial = 1.0/log(mu2)/evWindowSav->b0;
  }
  antTrial *= alphaSTrial;

  // Print.
  if (verboseIn >= DEBUG) {
    stringstream ss;
    ss << "colour factor =" << colFacSav;
    printOut(__METHOD_NAME__,ss.str());
    ss.str("");
    ss << "alphaS = " << alphaSTrial;
    printOut(__METHOD_NAME__,ss.str());
  }
  return antTrial;

}

//--------------------------------------------------------------------------

// Calculate the colour and coupling stripped antenna function.

double TrialGenerator::aTrialStrip(vector<double>& invariants,
  const vector<double>& masses, int verboseIn) {

  double antTrial = 0.;
  // Loop over sectors.
  for (auto itSector = zetaGenPtrs.begin(); itSector!= zetaGenPtrs.end();
       ++itSector) {

    Sector sectorNow = itSector->first;
    // Skip inactive sectors.
    if (!isActiveSector[sectorNow]) continue;

    // Add contribution from this sector.
    double aNow = itSector->second->aTrial(invariants,masses);
    if (verboseIn >= DEBUG) {
      itSector->second->print();
      stringstream ss;
      ss << "aTrial = "<< aNow;
      printOut(__METHOD_NAME__, ss.str());
    }
    antTrial+=aNow;
  }
  return antTrial;

}

//--------------------------------------------------------------------------

// Clear all saved trial variables.

void TrialGenerator::resetTrial() {
  hasTrial = false;
  q2Sav = 0.;
  colFacSav = 1.;
  evWindowSav = nullptr;
  sectorSav = Sector::Void;
  kallenFacSav = 1.0;
  Rpdf = 1.0;
  isActiveSector.clear();
  zetaLimits.clear();
}

//--------------------------------------------------------------------------

// Mark trial as used.

void TrialGenerator::needsNewTrial() {hasTrial = false;}

//==========================================================================

// Methods for FF trial generators.

//--------------------------------------------------------------------------

// Calculate the Kallen factor.

void TrialGeneratorFF::calcKallenFac(double sIK,
  const vector<double>& masses) {

  // Calculate f^FF_Kallen/(2pi).
  double mI2(0), mK2(0), mIK2(sIK);
  // Extract masses.
  if (masses.size()>=2) {
    mI2   = pow2(masses[0]);
    mK2   = pow2(masses[1]);
    mIK2 += mI2 + mK2;
  }

  double kallen = kallenFunction(mIK2, mI2, mK2);
  kallenFacSav = sIK/sqrt(kallen);
  kallenFacSav /= (2.*M_PI);

}

//==========================================================================

// Methods for RF trial generators.

//--------------------------------------------------------------------------

void TrialGeneratorRF::calcKallenFac(double sAK,
  const vector<double>& masses) {

  // Calculate f^RF_Kallen/(2pi).
  double mA(0), mK(0), mAK(0), mj(0), mk(0);
  // Extract masses.
  if (masses.size()>=3) {
    mA = masses[0];
    mK = masses[1];
    mAK = masses[2];
  }
  if (masses.size()>=4) {
    mj = masses[3];
    if (masses.size()>=5) mj = masses[4];
    else mk = mj;
  } else mk = mK;
  double m2diff = mj*mj + mk*mk - mK*mK;
  double kallen = kallenFunction(mA*mA, mK*mK, mAK*mAK);
  kallenFacSav = (sAK + m2diff)/sqrt(kallen);
  kallenFacSav /= (2.*M_PI);

}

//==========================================================================

// ZetaGeneratorSet class.

//--------------------------------------------------------------------------

// The constructor: create all possible generators for the
// corresponding parent type.

ZetaGeneratorSet::ZetaGeneratorSet(TrialGenType trialGenTypeIn) :
  trialGenTypeSav(trialGenTypeIn) {
  if (trialGenTypeIn == TrialGenType::FF) {
    addGenerator(new ZGenFFEmitSoft());
    addGenerator(new ZGenFFEmitColI());
    addGenerator(new ZGenFFEmitColK());
    addGenerator(new ZGenFFSplit());
  } else if (trialGenTypeIn == TrialGenType::RF) {
    addGenerator(new ZGenRFEmitSoft());
    // Uncomment for alternative zeta choice for soft sector.
    // addGenerator(new ZGenRFEmitSoftAlt());
    addGenerator(new ZGenRFEmitColK());
    addGenerator(new ZGenRFSplit());
  } else if (trialGenTypeIn == TrialGenType::IF) {
    addGenerator(new ZGenIFEmitSoft());
    addGenerator(new ZGenIFEmitColA());
    addGenerator(new ZGenIFEmitColK());
    addGenerator(new ZGenIFSplitA());
    addGenerator(new ZGenIFSplitK());
    addGenerator(new ZGenIFConv());
  } else if (trialGenTypeIn == TrialGenType::II) {
    addGenerator(new ZGenIIEmitSoft());
    addGenerator(new ZGenIIEmitCol());
    addGenerator(new ZGenIISplit());
    addGenerator(new ZGenIIConv());
  } else {
    string msg = "Unrecognised parent type.";
    printOut(__METHOD_NAME__,msg);
  }
}

//--------------------------------------------------------------------------

// Destructor.

ZetaGeneratorSet::~ZetaGeneratorSet() {
  for (auto it = zetaGenPtrs.begin(); it!=zetaGenPtrs.end(); ++it) {
    delete(it->second);
    zetaGenPtrs.erase(it->first);
  }
}

//--------------------------------------------------------------------------

// Get ptr to ZetaGenerator for a sector.

ZetaGenerator* ZetaGeneratorSet::getZetaGenPtr(BranchType branchType,
  Sector sectIn) {
  pair<BranchType, Sector> key = make_pair(branchType,sectIn);
  if (zetaGenPtrs.find(key)!= zetaGenPtrs.end()) return zetaGenPtrs[key];
  else return nullptr;
}

//--------------------------------------------------------------------------

// Save generator if it is the correct parent type.

void ZetaGeneratorSet::addGenerator(ZetaGenerator* zGenPtr) {
  if (zGenPtr->getTrialGenType() == trialGenTypeSav) {
    const BranchType btype = zGenPtr->getBranchType();
    const Sector sector = zGenPtr->getSector();
    auto key = make_pair(btype,sector);
    zetaGenPtrs[key] = zGenPtr;
  }
}

//==========================================================================

// ZetaGenerator class (base class for all generators).

//--------------------------------------------------------------------------

// Generate a value of zeta.

double ZetaGenerator::genZeta(Rndm* rndmPtr, double zMinIn, double zMaxIn,
  double gammaPDF) {
  double R = rndmPtr->flat();
  double IzMax = zetaIntSingleLim(zMaxIn,gammaPDF);
  double IzMin = zetaIntSingleLim(zMinIn,gammaPDF);
  if (IzMax < IzMin) return zMinIn;
  double Iz = IzMin + R*(IzMax - IzMin);
  double zetaNow  = inverseZetaIntegral(Iz,gammaPDF);
  return zetaNow;
}

//--------------------------------------------------------------------------

// Print the generator.

void ZetaGenerator::print() {
  cout << "  Zeta Generator Information:" << endl << "    Shower: ";
  if (trialGenType == TrialGenType::FF) cout << "FF";
  else if (trialGenType == TrialGenType::RF) cout << "RF";
  else if (trialGenType == TrialGenType::IF) cout << "IF";
  else if (trialGenType == TrialGenType::II) cout << "II";
  else cout << "None";
  cout << "\n    BranchType: ";
  if (branchType == BranchType::Emit) cout << "Emit";
  else if (branchType == BranchType::SplitF) cout << "Split F";
  else if (branchType == BranchType::SplitI) cout << "Split I";
  else if (branchType == BranchType::Conv) cout << "Conv";
  else cout << "None";
  cout << "\n    Sector: ";
  if (sector == Sector::ColI) cout << "ColI";
  else if (sector == Sector::Default) cout << "Soft/Global";
  else if (sector == Sector::ColK) cout << "ColK";
  else cout << "None";
  cout << "\n";
}

//--------------------------------------------------------------------------

// Check if invariants are valid.

bool ZetaGenerator::valid(const string& method, Info* infoPtr, int verbose,
  double zIn) {
  if (zIn == 0.) {
    if (verbose >= DEBUG && infoPtr != nullptr)
      infoPtr->errorMsg("Error in " + method, ": zeta is zero.");
    return false;
  } else if (zIn == 1.) {
    if (verbose >= DEBUG && infoPtr != nullptr)
      infoPtr->errorMsg("Error in " + method, ": zeta is unity.");
    return false;
  }
  return true;
}

bool ZetaGenerator::valid(const string& method, Info* infoPtr, int verbose,
  double zIn, const double& Q2In) {
  if (zIn == 0) {
    if (verbose >= DEBUG && infoPtr != nullptr)
      infoPtr->errorMsg("Error in " + method, ": zeta is zero.");
    return false;
  } else if (zIn < 0.) {
    if (verbose >= DEBUG && infoPtr != nullptr)
        infoPtr->errorMsg("Error in "+ method, ": zeta is negative.");
    return false;
  } else if (Q2In < 0.) {
    if (verbose >= DEBUG && infoPtr != nullptr) infoPtr->errorMsg(
      "Error in "+ method, ": trial Q2 is negative");
    return false;
  }
  return true;
}

//==========================================================================

// The final-final trial generator methods.

//==========================================================================

// ZGenFFEmitSoft.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenFFEmitSoft::getzMin(double Q2, double sAnt, const vector<double>&,
  double, double) {
  double xT   = Q2 / sAnt;
  double root = sqrt(1. - 4. * xT);
  double rootFrac = (1. + root) / (1. - root);
  return rootFrac > 0. ? -log(rootFrac)/2. : 0;
}

double ZGenFFEmitSoft::getzMax(double Q2, double sAnt, const vector<double>&,
  double, double) {
  double xT   = Q2 / sAnt;
  double root = sqrt(1. - 4. * xT);
  double rootFrac = (1. + root) / (1. - root);
  return rootFrac > 0. ? log(rootFrac)/2. : 0;
}

void ZGenFFEmitSoft::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>&, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double xT  = Q2In / sAnt;
  double yij = sqrt(xT) * exp(-zIn);
  double yjk = sqrt(xT) * exp(zIn);
  double sij = yij * sAnt;
  double sjk = yjk * sAnt;
  double sik = sAnt - sij - sjk;
  invariants = {sAnt, sij, sjk, sik};
}

double ZGenFFEmitSoft::aTrial( const vector<double>& invariants,
  const vector<double>&) {
  if (invariants.size() >= 3) {
    double sIK = invariants[0];
    double yij = invariants[1] / sIK;
    double yjk = invariants[2] / sIK;
    return 1./sIK * 2./(yij*yjk);
  } else return 0;
}

double ZGenFFEmitSoft::zetaIntSingleLim(double z, double) {return z;}

double ZGenFFEmitSoft::inverseZetaIntegral(double Iz, double) {return Iz;}

//==========================================================================

// ZGenFFEmitColI.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenFFEmitColI::getzMin(double Q2, double sAnt,
  const vector<double>&, double, double) {
  double xT = Q2 / sAnt;
  return (1. - sqrt(1. - 4. * xT)) / 2.;
}

double ZGenFFEmitColI::getzMax(double Q2, double sAnt,
  const vector<double>&, double, double) {
  double xT = Q2 / sAnt;
  return (1. + sqrt(1. - 4. * xT)) / 2.;
}

void ZGenFFEmitColI::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>&, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double yjk = zIn;
  double sjk = yjk * sAnt;
  double sij = Q2In / zIn;
  double sik = sAnt - sij - sjk;
  invariants = {sAnt, sij, sjk, sik};
}

double ZGenFFEmitColI::aTrial(const vector<double>& invariants,
  const vector<double>&) {
  if (invariants.size() >= 3) {
    double sIK = invariants[0];
    double yij = invariants[1]/sIK;
    double yjk = invariants[2]/sIK;
    return 1./sIK * 2./(yij*(1.-yjk));
  } else return 0;
}

double ZGenFFEmitColI::zetaIntSingleLim(double z, double) {
  return z == 1 ? 0 : -log(1.-z);}

double ZGenFFEmitColI::inverseZetaIntegral(double Iz, double) {
  return 1. - exp(-Iz);}

//==========================================================================

// ZGenFFEmitColK.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenFFEmitColK::getzMin(double Q2, double sAnt,
  const vector<double>&, double, double) {
  double xT = Q2 / sAnt;
  return (1. - sqrt(1. - 4. * xT)) / 2.;
}

double ZGenFFEmitColK::getzMax(double Q2, double sAnt,
  const vector<double>&, double, double) {
  double xT = Q2 / sAnt;
  return (1. + sqrt(1. - 4. * xT)) / 2.;
}

void ZGenFFEmitColK::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>&, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double yij = zIn;
  double sij = yij * sAnt;
  double sjk = Q2In / zIn;
  double sik = sAnt - sij - sjk;
  invariants = {sAnt, sij, sjk, sik};
}

double ZGenFFEmitColK::aTrial( const vector<double>& invariants,
  const vector<double>&) {
  if (invariants.size() >= 3) {
    double sIK = invariants[0];
    double yij = invariants[1]/sIK;
    double yjk = invariants[2]/sIK;
    return 1./sIK * 2./(yjk*(1.-yij));
  } else return 0;
}

double ZGenFFEmitColK::zetaIntSingleLim(double z, double) {
  return z == 1 ? 0 : -log(1.-z);}

double ZGenFFEmitColK::inverseZetaIntegral(double Iz, double) {
  return 1. - exp(-Iz);}

//==========================================================================

// ZGenFFSplit.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenFFSplit::getzMin(double Q2, double sAnt,
  const vector<double>&, double, double) {
  double xTmin = Q2 / sAnt;
  return (1. - sqrt(1. - 4.*xTmin)) / 2.;
}

double ZGenFFSplit::getzMax(double Q2, double sAnt,
  const vector<double>&, double, double) {
  double xTmin = Q2 / sAnt;
  return (1. + sqrt(1. - 4.*xTmin)) / 2.;
}

void ZGenFFSplit::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>& masses, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn, Q2In)) {
    invariants.clear(); return;}
  double mj2  = (masses.size()>=3 ? pow2(masses[1]) : 0.);
  double sij  = Q2In/zIn - 2.*mj2;
  double sjk  = zIn*sAnt - mj2;
  double sik  = sAnt - sij - sjk - 2.*mj2;
  invariants = {sAnt, sij, sjk, sik};
}

double ZGenFFSplit::aTrial( const vector<double>& invariants,
  const vector<double>& masses) {
  if (invariants.size() >= 3) {
    double sIK = invariants[0];
    double sij = invariants[1];
    double yij = sij / sIK;
    double muj2 = (masses.size()>=3 ? pow2(masses[1]) / sIK : 0.);
    return 1./sIK * 1./(yij+2.*muj2);
  } else return 0;
}

double ZGenFFSplit::zetaIntSingleLim(double z, double) {
  return z/2.;}

double ZGenFFSplit::inverseZetaIntegral(double Iz, double) {
  return 2.*Iz;}

//==========================================================================

// The resonance-final trial generator methods.

//==========================================================================

// ZGenRFEmitSoft.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenRFEmitSoft::getzMin(double Q2, double,
  const vector<double>& masses, double, double) {
  if (masses.size() >= 3 && Q2 > 0.) {
    double mA  = masses[0];
    double mK  = masses[1];
    double mAK = masses[2];
    return 1.0/(1.0 - Q2/(mA*mA -(mAK + mK)*(mAK + mK)));
  } else return 2.0;
}

double ZGenRFEmitSoft::getzMax(double, double sAnt,
  const vector<double>& masses, double, double) {
  if(masses.size() >= 3) {
    double mA  = masses[0];
    double mK  = masses[1];
    double mAK = masses[2];
    return 1.0 + ((mA-mAK)*(mA-mAK) - mK*mK)/sAnt;
  } else return 1;
}

void ZGenRFEmitSoft::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>&, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double yjk = 1. - 1./zIn;
  double saj = Q2In/yjk;
  double sjk = sAnt*(zIn - 1.0);
  double sak = sAnt + sjk - saj;
  invariants = {sAnt, saj, sjk, sak};
}

double ZGenRFEmitSoft::aTrial( const vector<double>& invariants,
  const vector<double>&) {
  if (invariants.size() >= 3) {
    double sAK = invariants[0];
    double saj = invariants[1];
    double sjk = invariants[2];
    return 2.0 *(sAK + sjk)/(saj*sjk);
  } else return 0;
}

double ZGenRFEmitSoft::zetaIntSingleLim(double z, double) {
  return z > 1. && z < 2. ? z - 1. + log(z-1.) : 0.;}

double ZGenRFEmitSoft::inverseZetaIntegral(double Iz, double) {
  return 1.0 + lambertW(exp(Iz));}

//==========================================================================

// ZGenRFEmitSoftAlt. (Alternative RFEmit soft trial generator.)

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenRFEmitSoftAlt::getzMin(double Q2, double sAnt,
  const vector<double>&, double, double) {
  return sqrt(0.5*Q2/sAnt);}

double ZGenRFEmitSoftAlt::getzMax(double Q2, double sAnt,
  const vector<double>&, double, double) {
  return Q2/sAnt/(1.-sqrt(1.0 - 2.0*Q2/sAnt));}

void ZGenRFEmitSoftAlt::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>&, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double yaj = zIn;
  double sjk = Q2In/yaj;
  double saj = (sjk+sAnt)*yaj;
  double sak = sAnt + sjk - saj;
  invariants = {sAnt, saj, sjk, sak};
}

double ZGenRFEmitSoftAlt::aTrial( const vector<double>& invariants,
  const vector<double>&) {
  if (invariants.size() >= 3) {
    double sAK  = invariants[0];
    double saj  = invariants[1];
    double sjk  = invariants[2];
    return 2.0 *(sAK + sjk)/(saj*sjk)*pow2(2.0*sAK/(sAK+sjk));
  } else return 0;
}

double ZGenRFEmitSoftAlt::zetaIntSingleLim(double z, double) {
  return z > 0. && z < 1 ? 4.0*log(z) : 0;}

double ZGenRFEmitSoftAlt::inverseZetaIntegral(double Iz, double) {
  return exp(Iz/4.0);}

//==========================================================================

// ZGenRFEmitSoftAlt. (Alternative RFEmit soft trial generator.)

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenRFEmitColK::getzMin(double Q2, double sAnt,
  const vector<double> &,double, double) {
  return sqrt(0.5*Q2/sAnt);}

double ZGenRFEmitColK::getzMax(double Q2, double sAnt,
  const vector<double> &,double, double) {
  return Q2/sAnt/(1.-sqrt(1.0 - 2.0*Q2/sAnt));}

void ZGenRFEmitColK::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>&, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double yaj = zIn;
  double sjk = Q2In/yaj;
  double saj = (sjk+sAnt)*yaj;
  double sak = sAnt + sjk - saj;
  invariants = {sAnt, saj, sjk, sak};
}

double ZGenRFEmitColK::aTrial( const vector<double> & invariants,
  const vector<double> &) {
  if (invariants.size() >= 3) {
    double sAK  = invariants[0];
    double saj  = invariants[1];
    double sjk  = invariants[2];
    double yaj  = saj/(sAK+sjk);
    double yjk  = sjk/(sAK+sjk);
    return 2.0/(sAK*yjk*(1.0-yaj)) * pow3(2.0*(1-yjk));
  } else return 0;
}

double ZGenRFEmitColK::zetaIntSingleLim(double z, double) {
  return z > 0. && z < 1. ? -8.0*log(1.-z) : 0;}

double ZGenRFEmitColK::inverseZetaIntegral(double Iz, double) {
  return 1.-exp(-Iz/8.0);}

//==========================================================================

// ZGenRFSplit.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenRFSplit::getzMin(double Q2, double sAnt, const vector<double>&,
  double, double) {
  return sqrt(0.5*Q2/sAnt);}

double ZGenRFSplit::getzMax(double Q2, double sAnt, const vector<double>&,
  double, double) {
  return Q2/sAnt/(1.-sqrt(1.0 - 2.0*Q2/sAnt));}

void ZGenRFSplit::genInvariants(double Q2In, double zIn, double sAK,
  const vector<double>& masses, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn, Q2In)) {
    invariants.clear(); return;}
  double yaj = zIn;
  double m2q = masses.size()>=2 ? pow2(masses[1]) : 0.;
  double sjk = Q2In/yaj;
  // Massive splittings.
  if (m2q > NANO) {
    double a = 1. - (Q2In + m2q)/(sAK*yaj);
    double b = Q2In/(sAK*yaj);
    double x = 0.5*a*(1.-sqrt(1+4.*b/(a*a)));
    sjk = -2.0*m2q + sAK*x;
  }
  double saj = yaj*(sAK + sjk + 2.*m2q);
  double sak = sAK + sjk +2.*m2q - saj;
  invariants = {sAK, saj, sjk, sak};
}

double ZGenRFSplit::aTrial( const vector<double>& invariants,
  const vector<double>& masses) {
  if(invariants.size() >= 3) {
    double m2q = masses.size() >= 2 ? pow2(masses[1]) : 0.;
    double sAK = invariants[0];
    double sjk = invariants[2];
    double yjk = sjk/(sAK+sjk+2.0*m2q);
    return 2.0*pow2(1.-yjk)/(sjk+2.*m2q);
  } else return 0;
}

double ZGenRFSplit::zetaIntSingleLim(double z, double) {
  return z;}

double ZGenRFSplit::inverseZetaIntegral(double Iz, double) {
  return Iz;}

//==========================================================================

// The initial-final trial generator methods.

//==========================================================================

// ZGenIFEmitSoft.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenIFEmitSoft::getzMin(double Q2, double sAnt,
  const vector<double>&, double, double) {
  return Q2 / (sAnt + Q2);}

double ZGenIFEmitSoft::getzMax(double, double,
  const vector<double>&, double xA, double) {
  // It's actually (1.-xA)/xA * yAK, but don't know the normalisation
  // of sAK, so overestimate yAK by 1.
  return (1. - xA) / xA;}

void ZGenIFEmitSoft::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>&, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double saj = Q2In / zIn;
  double yjk = zIn;
  double sak = sAnt / (1.-yjk) - saj;
  // Note: for gluon emissions sAK + sjk = saj + sak.
  double sjk = yjk * (saj + sak);
  invariants = {sAnt, saj, sjk, sak};
}

double ZGenIFEmitSoft::aTrial(const vector<double>& invariants,
  const vector<double>&) {
  if (invariants.size() == 3) {
    double sAK = invariants[0];
    double saj = invariants[1];
    double sjk = invariants[2];
    // NOTE: for gluon emissions saj + sak = sAK + sjk.
    double yaj = saj / (sAK + sjk);
    double yjk = sjk / (sAK + sjk);
    return 1./sAK * 2./(yaj * yjk);
  } else if (invariants.size() == 4) {
    double sAK = invariants[0];
    double saj = invariants[1];
    double sjk = invariants[2];
    double sak = invariants[3];
    double yaj = saj / (saj + sak);
    double yjk = sjk / (saj + sak);
    return 1./sAK * 2./(yaj * yjk);
  } else return 0;
}

double ZGenIFEmitSoft::zetaIntSingleLim(double z, double gammaPDF) {
  // General solution for real gammaPDF would be a hypergeometric,
  // so only special cases.
  if (gammaPDF == 0.) return z != 1. ? -log((1.-z)*exp(z)) : 0;
  else if (gammaPDF == 1.) return pow2(z)/2.;
  return 0;
}

double ZGenIFEmitSoft::inverseZetaIntegral(double Iz, double gammaPDF) {
  // General solution for real gammaPDF would be a hypergeometric,
  // so only special cases.
  if (gammaPDF == 0.) return 1. + lambertW(-exp(-1.-Iz));
  else if (gammaPDF == 1.) return 2.*sqrt(Iz);
  return 0.;
}

//==========================================================================

// ZGenIFEmitColA.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenIFEmitColA::getzMin(double Q2, double sAnt, const vector<double>&,
  double, double) {
  return Q2 / (sAnt + Q2);}

double ZGenIFEmitColA::getzMax(double, double,
  const vector<double>&, double xA, double) {
  // It's actually (1.-xA)/xA * yAK, but don't know the normalisation
  // of sAK, so overestimate yAK by 1.
  return (1. - xA) / xA;}

void ZGenIFEmitColA::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>&, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double saj = Q2In / zIn;
  double yjk = zIn;
  double sak = sAnt / (1.-yjk) - saj;
  // Note: for gluon emissions sAK + sjk = saj + sak.
  double sjk = yjk * (saj + sak);
  invariants = {sAnt, saj, sjk, sak};
}

double ZGenIFEmitColA::aTrial(const vector<double>& invariants,
  const vector<double>&) {
  if (invariants.size() == 3) {
    double sAK = invariants[0];
    double saj = invariants[1];
    double sjk = invariants[2];
    // NOTE: for gluon emissions saj + sak = sAK + sjk.
    double yaj = saj / (sAK + sjk);
    double yjk = sjk / (sAK + sjk);
    return 1./sAK * 2./(yaj * (1. - yjk));
  } else if (invariants.size() == 4) {
    double sAK = invariants[0];
    double saj = invariants[1];
    double sjk = invariants[2];
    double sak = invariants[3];
    double yaj = saj / (saj + sak);
    double yjk = sjk / (saj + sak);
    return 1./sAK * 2./(yaj * (1. - yjk));
  } return 0;
}

double ZGenIFEmitColA::zetaIntSingleLim(double z, double gammaPDF) {
  if (gammaPDF == 0.) return z != 1. ? 1./(1.-z) : 0;
  if (gammaPDF == 1.) return z != 1. ? -log(1.-z) : 0;
  if (gammaPDF == 2.) return z;
  return -pow(1-z,gammaPDF-1.)/(gammaPDF-1.);
}

double ZGenIFEmitColA::inverseZetaIntegral(double Iz, double gammaPDF) {
  if (gammaPDF == 0.) return (Iz-1.)/Iz;
  if (gammaPDF == 1.) return 1.-exp(-Iz);
  if (gammaPDF == 2.) return Iz;
  return 1.-pow(-Iz*(gammaPDF-1.),1./(gammaPDF-1.));
}

//==========================================================================

// ZGenIFEmitColK.

//--------------------------------------------------------------------------

// Overridden methods.
// TODO: find better overestimate without arbitrary cutoff!

double ZGenIFEmitColK::getzMin(double Q2, double sAnt, const vector<double>&,
  double xA, double) {return xA/(1.-xA)*Q2/sAnt;}

double ZGenIFEmitColK::getzMax(double, double, const vector<double>&,
  double, double) {
  // Would actually be 1., but this is always in the other sector, so take
  // arbitrary cutoff to stay away from the singularity.
  return 0.99;}

void ZGenIFEmitColK::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>&, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double yaj = zIn;
  double sjk = Q2In / zIn;
  // Note: for gluon emissions we have saj + sak = sAK + sjk.
  double saj = yaj * (sAnt + sjk);
  double sak = sAnt + sjk - saj;
  invariants = {sAnt, saj, sjk, sak};
}

double ZGenIFEmitColK::aTrial(const vector<double>& invariants,
  const vector<double>&) {
  if (invariants.size() == 3) {
    double sAK = invariants[0];
    double saj = invariants[1];
    double sjk = invariants[2];
    // NOTE: for gluon emissions saj + sak = sAK + sjk.
    double yaj = saj / (sAK + sjk);
    double yjk = sjk / (sAK + sjk);
    return 1./sAK * 2./(yjk * (1. - yaj));
  } else if (invariants.size() == 4) {
    double sAK = invariants[0];
    double saj = invariants[1];
    double sjk = invariants[2];
    double sak = invariants[3];
    double yaj = saj / (saj + sak);
    double yjk = sjk / (saj + sak);
    return 1./sAK * 2./(yjk * (1. - yaj));
  } return 0;
}

double ZGenIFEmitColK::zetaIntSingleLim(double z, double gammaPDF) {
  // Simple solution only for gammaPDF = 1.
  return gammaPDF == 1. && z != 1. ? -log(1.-z) : 0;}

double ZGenIFEmitColK::inverseZetaIntegral(double Iz, double gammaPDF) {
  // Simple solution only for gammaPDF = 1.
  return gammaPDF == 1. ? 1.-exp(-Iz) : 0;}

//==========================================================================

// ZGenIFSplitA.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenIFSplitA::getzMin(double Q2, double sAnt,
  const vector<double>&, double, double) {
  return Q2/(sAnt+Q2);}

double ZGenIFSplitA::getzMax(double, double sAnt,
  const vector<double>& masses, double xA, double) {
  double muj2 = (masses.size() >= 3 ? pow2(masses.at(1)) / sAnt : 0.);
  // Would actually be (1-xA)/xA*yAK, but we don't know
  // the normalisation of sAK.
  return (1.-xA)/xA + 2.*muj2;
}

void ZGenIFSplitA::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>& masses, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double muj2 = (masses.size() >= 3 ? pow2(masses.at(1)) / sAnt : 0.);
  double saj  = Q2In / zIn;
  double yjk  = zIn - muj2;
  // Note: convention is that for initial-state splittings mA = mj, so
  // sak + saj = sAK + sjk.
  double sak  = sAnt/(1.-yjk) - saj;
  double sjk  = yjk * (saj + sak);
  invariants = {sAnt, saj, sjk, sak};
}

double ZGenIFSplitA::aTrial(const vector<double>& invariants,
  const vector<double>&) {
  if (invariants.size() == 3) {
    double sAK = invariants[0];
    double saj = invariants[1];
    double sjk = invariants[2];
    // Note: convention is that for initial-state splittings mA = mj, so
    // sak + saj = sAK + sjk.
    double yaj = saj / (sAK + sjk);
    return 1./sAK * 1./yaj;
  } else if (invariants.size() == 4) {
    double sAK = invariants[0];
    double saj = invariants[1];
    double sak = invariants[3];
    double yaj = saj / (saj + sak);
    return 1./sAK * 1./yaj;
  } else return 0;
}

double ZGenIFSplitA::zetaIntSingleLim(double z, double gammaPDF) {
  // For now ignore mass terms in zeta integral, but in principle
  // we could have trial generators for each flavour.
  if (gammaPDF == 0.) return z != 1. ? -log(1.-z) : 0.;
  if (gammaPDF == 1.) return z;
  return -pow(1-z,gammaPDF+1.)/(gammaPDF+1.);
}

double ZGenIFSplitA::inverseZetaIntegral(double Iz, double gammaPDF) {
  if (gammaPDF == 0.) return exp(Iz);
  if (gammaPDF == 1.) return Iz;
  return 1.-pow((-gammaPDF-1.)*Iz,1./(gammaPDF+1.));
}

//==========================================================================

// ZGenIFSplitK.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenIFSplitK::getzMin(double Q2, double sAnt,
  const vector<double>&, double xA, double) {
  return xA/(1.-xA)*Q2/sAnt;}

double ZGenIFSplitK::getzMax(double, double,
  const vector<double>&, double, double) {
  return 1.;}

void ZGenIFSplitK::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>& masses, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double mj2 = (masses.size() >= 3 ? pow2(masses.at(1)) : 0.);
  double sjk = Q2In / zIn - 2.*mj2;
  double sak = (1.-zIn)*(sAnt + sjk + 2.*mj2) - mj2;
  // Note: for final-state gluon splittings we have
  // saj + sak = sAK + sjk + 2mj^2.
  double saj = zIn * (sAnt + sjk + 2.*mj2) + mj2;
  invariants = {sAnt, saj, sjk, sak};
}

double ZGenIFSplitK::aTrial(const vector<double>& invariants,
  const vector<double>& masses) {
  // Assume ma = masses[0], mj = masses[1], ( mk = masses[2] ).
  // Correction factor >= 1 to get rid of unwanted factors.
  if (invariants.size() == 3) {
    double mj2 = (masses.size() >= 3 ? pow2(masses.at(1)) : 0.);
    double sAK = invariants[0];
    double sjk = invariants[2];
    double yjk = sjk / (sAK + sjk + 2.*mj2);
    double muj2 = mj2 / (sAK + sjk + 2.*mj2);
    return 1./sAK * 1./(yjk + 2.*muj2) * (sAK + mj2)/sAK;
  } else if (invariants.size() == 4) {
    double mj2 = (masses.size() >= 3 ? pow2(masses.at(1)) : 0.);
    double sAK = invariants[0];
    double saj = invariants[1];
    double sjk = invariants[2];
    double sak = invariants[3];
    double yjk = sjk / (saj + sak);
    double muj2 = mj2 / (saj + sak);
    return 1./sAK * 1./(yjk + 2.*muj2)*(sAK + mj2)/sAK;
  } else return 0;
}

double ZGenIFSplitK::zetaIntSingleLim(double z, double) {
  // Solution only for gammaPDF = 1.
  return z/2.;}

double ZGenIFSplitK::inverseZetaIntegral(double Iz, double) {
  // Solution only for gammaPDF = 1.
  return 2.*Iz;}

//==========================================================================

// ZGenIFConv.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenIFConv::getzMin(double Q2, double sAnt,
  const vector<double>&, double xA, double) {
  // We have yAK >= xA.
  return xA/2. * (sqrt(1 + 4.*Q2/sAnt/xA) - 1.);
}

double ZGenIFConv::getzMax(double, double,
  const vector<double>&, double xA, double) {
  // Would actually be (1-xA)/xA*yAK,
  // but we don't know the normalisation.
  return (1.-xA)/xA;
}

void ZGenIFConv::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>& masses, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double mj2 = (masses.size() >= 3 ? pow2(masses.at(1)) : 0.);
  double saj = Q2In / zIn + mj2;
  double sak = sAnt/(1.-zIn) - saj;
  double sjk = zIn * (saj + sak) - mj2;
  invariants = {sAnt, saj, sjk, sak};
}

double ZGenIFConv::aTrial(const vector<double>& invariants,
  const vector<double>& masses) {
  // Assumption is ma = 0, so saj + sak = sAK + sjk + mj^2.
  // Correction factor >= 1 to get rid of unwanted factors.
  if (invariants.size() == 3) {
    double mj2  = masses.size() >= 3 ? pow2(masses.at(1)) : 0.;
    double sAK  = invariants[0];
    double saj  = invariants[1];
    double sjk  = invariants[2];
    double yAK  = sAK / (sAK + sjk + mj2);
    double yaj  = saj / (sAK + sjk + mj2);
    double muj2 = mj2 / (sAK + sjk + mj2);
    return 1./sAK * 1./(2. * (yaj - muj2) * yAK) * (sAK + mj2)/sAK;
  } else if (invariants.size() == 4) {
    double mj2  = masses.size() >= 3 ? pow2(masses.at(1)) : 0.;
    double sAK  = invariants[0];
    double saj  = invariants[1];
    double sak  = invariants[3];
    double yAK  = sAK / (saj + sak);
    double yaj  = saj / (saj + sak);
    double muj2 = mj2 / (saj + sak);
    return 1./sAK * 1./(2. * (yaj - muj2) * yAK)*(sAK + mj2)/sAK;
  } return 0;
}

double ZGenIFConv::zetaIntSingleLim(double z, double gammaPDF) {
  if (gammaPDF == 2.) return z/4;
  else if (gammaPDF == 1. && z != 1.) return -log(1.-z)/4;
  return -pow(1.-z,gammaPDF-1.)/(gammaPDF-1.)/4;
}

double ZGenIFConv::inverseZetaIntegral(double Iz, double gammaPDF) {
  if (gammaPDF == 2.) return Iz;
  if (gammaPDF == 1.) return 1.-exp(-Iz);
  if (Iz != 0.) return 1.-pow(-Iz*(gammaPDF-1.),1./(gammaPDF-1.));
  return 0.;
}

//==========================================================================

// The initial-initial trial generator methods.

//==========================================================================

// ZGenIIEmitSoft.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenIIEmitSoft::getzMin(double Q2, double sAnt,
  const vector<double>&, double xA, double xB) {
  double xRatio = Q2/sAnt * xA*xB/pow2(1.-xA*xB);
  return (1.-xA*xB)/2. * (1. - sqrt(1. - 4.*xRatio));
}

double ZGenIIEmitSoft::getzMax(double Q2, double sAnt,
  const vector<double>&, double xA, double xB) {
  double xRatio = Q2/sAnt * xA*xB/pow2(1.-xA*xB);
  return (1.-xA*xB)/(2.*xA*xB) * (1. + sqrt(1. - 4.*xRatio));
}

void ZGenIIEmitSoft::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>&, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double saj = Q2In / zIn;
  // Note: zeta = yjb = 1 implies saj = -sAB, so won't happen.
  double sab = (saj + sAnt) / (1. - zIn);
  double sjb = zIn * sab;
  invariants = {sAnt, saj, sjb, sab};
}

double ZGenIIEmitSoft::aTrial(const vector<double>& invariants,
  const vector<double>&) {
  if (invariants.size() == 3) {
    double sAB = invariants[0];
    double saj = invariants[1];
    double sjb = invariants[2];
    // NOTE: for gluon emissions sab = sAB + saj + sjb.
    double sab = sAB + saj + sjb;
    double yaj = saj / sab;
    double yjb = sjb / sab;
    return 1./sAB * 2./(yaj * yjb);
  } else if (invariants.size() == 4) {
    double sAB = invariants[0];
    double saj = invariants[1];
    double sjb = invariants[2];
    double sab = invariants[3];
    double yaj = saj / sab;
    double yjb = sjb / sab;
    return 1./sAB * 2./(yaj * yjb);
  } return 0;
}

double ZGenIIEmitSoft::zetaIntSingleLim(double z, double gammaPDF) {
  // Could only be solved for gammaPDF = 1.
  return gammaPDF == 1. && z != 0. ? log(z) : 0;
}

double ZGenIIEmitSoft::inverseZetaIntegral(double Iz, double gammaPDF) {
  // Could only be solved for gammaPDF = 1.
  return gammaPDF == 1. ? exp(Iz) : 0;
}

//==========================================================================

// ZGenIIEmitSoft. Everything written for the A-collinear side.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenIIEmitCol::getzMin(double Q2, double sAnt,
  const vector<double>&, double xA, double xB) {
  double xRatio = Q2/sAnt * xA*xB/pow2(1.-xA*xB);
  return (1.-xA*xB)/2. * (1. - sqrt(1. - 4.*xRatio));
}

double ZGenIIEmitCol::getzMax(double Q2, double sAnt,
  const vector<double>&, double xA, double xB) {
  double xRatio = Q2/sAnt * xA*xB/pow2(1.-xA*xB);
  return (1.-xA*xB)/(2.*xA*xB) * (1. + sqrt(1. - 4.*xRatio));
}

void ZGenIIEmitCol::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>&, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double saj = Q2In / zIn;
  // Note: zeta = yjb = 1 implies saj = -sAB, so won't happen.
  double sab = (saj + sAnt) / (1. - zIn);
  double sjb = zIn * sab;
  invariants = {sAnt, saj, sjb, sab};
}

double ZGenIIEmitCol::aTrial(const vector<double>& invariants,
  const vector<double>&) {
  if (invariants.size() == 3) {
    double sAB = invariants[0];
    double saj = invariants[1];
    double sjb = invariants[2];
    // NOTE: for gluon emissions sab = sAB + + saj + sjb.
    double sab = sAB + saj + sjb;
    double yaj = saj / sab;
    double yjb = sjb / sab;
    return 1./sAB * 2./(yaj * (1. - yjb));
  } else if (invariants.size() == 4) {
    double sAB = invariants[0];
    double saj = invariants[1];
    double sjb = invariants[2];
    double sab = invariants[3];
    double yaj = saj / sab;
    double yjb = sjb / sab;
    return 1./sAB * 2./(yaj * (1. - yjb));
  } return 0;
}

double ZGenIIEmitCol::zetaIntSingleLim(double z, double gammaPDF) {
  // Could be solved only for gammaPDF = 1.
  return gammaPDF == 1. && z != 1. ? -log(1.-z) : 0;
}

double ZGenIIEmitCol::inverseZetaIntegral(double Iz, double gammaPDF) {
  // Could be solved only for gammaPDF = 1.
  return gammaPDF == 1. ? 1.-exp(-Iz) : 0.;
}

//==========================================================================

// ZGenIISplit. Everything written for the A-collinear side.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenIISplit::getzMin(double Q2, double sAnt,
  const vector<double>&, double xA, double xB) {
  double xRatio = Q2/sAnt * xA*xB/pow2(1.-xA*xB);
  return (1.-xA*xB)/2. * (1. - sqrt(1. - 4.*xRatio));
}

double ZGenIISplit::getzMax(double Q2, double sAnt,
  const vector<double>&, double xA, double xB) {
  double xRatio = Q2/sAnt * xA*xB/pow2(1.-xA*xB);
  return (1.-xA*xB)/(2.*xA*xB) * (1. + sqrt(1. - 4.*xRatio));
}

void ZGenIISplit::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>&, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn, Q2In)) {
    invariants.clear(); return;}
  double saj = Q2In / zIn;
  // Note: zeta = yjb = 1 implies saj = -sAB, so won't happen.
  double sab = (saj + sAnt) / (1. - zIn);
  double sjb = zIn * sab;
  invariants = {sAnt, saj, sjb, sab};
}

double ZGenIISplit::aTrial(const vector<double>& invariants,
  const vector<double>&) {
  if (invariants.size() == 3) {
    double sAB = invariants[0];
    double saj = invariants[1];
    double sjb = invariants[2];
    // NOTE: for ISR gluon splittings sab = sAB + saj + sjb.
    double sab = sAB + saj + sjb;
    double yaj = saj / sab;
    return 1./sAB * 1./yaj;
  } else if (invariants.size() == 4) {
    double sAB = invariants[0];
    double saj = invariants[1];
    double sab = invariants[3];
    double yaj = saj / sab;
    return 1./sAB * 1./yaj;
  } else return 0;
}

double ZGenIISplit::zetaIntSingleLim(double z, double gammaPDF) {
  // Could be solved only for gammaPDF = 1.
  return gammaPDF == 1. ? z/2. : 0;
}

double ZGenIISplit::inverseZetaIntegral(double Iz, double gammaPDF) {
  // Could be solved only for gammaPDF = 1.
  return gammaPDF == 1. ? 2.*Iz : 0;
}

//==========================================================================

// ZGenIIConv. Everything written for the A-collinear side.

//--------------------------------------------------------------------------

// Overridden methods.

double ZGenIIConv::getzMin(double Q2, double sAnt,
  const vector<double>&, double xA, double xB) {
  double xRatio = Q2/sAnt * xA*xB/pow2(1.-xA*xB);
  return (1.-xA*xB)/2. * (1. - sqrt(1. - 4.*xRatio));
}

double ZGenIIConv::getzMax(double Q2, double sAnt,
  const vector<double>&, double xA, double xB) {
  double xRatio = Q2/sAnt * xA*xB/pow2(1.-xA*xB);
  return (1.-xA*xB)/(2.*xA*xB) * (1. + sqrt(1. - 4.*xRatio));
}

double ZGenIIConv::getConstFactor(double sAnt,
  const vector<double>& masses) {
  return masses.size() >= 3 ? sAnt/(sAnt - pow2(masses.at(1))): 1;}

void ZGenIIConv::genInvariants(double Q2In, double zIn, double sAnt,
  const vector<double>& masses, vector<double>& invariants,
  Info* infoPtr, int verboseIn) {
  if (!valid(__METHOD_NAME__, infoPtr, verboseIn, zIn)) {
    invariants.clear(); return;}
  double mj2 = masses.size() >= 3 ? pow2(masses.at(2)) : 0.;
  double saj = Q2In / zIn + mj2;
  // Note: zeta = yjb = 1 implies saj = -sAB, so won't happen.
  double sab = (saj + sAnt) / (1. - zIn);
  double sjb = zIn * sab;
  invariants = {sAnt, saj, sjb, sab};
}

double ZGenIIConv::aTrial(const vector<double>& invariants,
  const vector<double>& masses) {
  if (invariants.size() == 3) {
    double mj2 = masses.size()>=1 ? pow2(masses.at(0)) : 0.;
    double sAB = invariants[0];
    double saj = invariants[1];
    double sjb = invariants[2];
    // Note: initial-state particles assumed massless.
    double sab = sAB + saj + sjb - mj2;
    double yAB = sAB / sab;
    double yaj = saj / sab;
    double muj2 = mj2 / sab;
    return 1./sAB * 1./((yaj - muj2) * yAB);
  } else if (invariants.size() == 4) {
    double mj2 = masses.size()>=1 ? pow2(masses.at(0)) : 0.;
    double sAB = invariants[0];
    double saj = invariants[1];
    double sab = invariants[3];
    double yaj = saj / sab;
    double yAB = sAB / sab;
    double muj2 = mj2 / sab;
    return 1./sAB * 1./((yaj - muj2) * yAB);
  } return 0;
}

double ZGenIIConv::zetaIntSingleLim(double z, double gammaPDF) {
  // Could be solved only for gammaPDF = 1.
  return gammaPDF == 1. ? z/2. : 0;
}

double ZGenIIConv::inverseZetaIntegral(double Iz, double gammaPDF) {
  // Could be solved only for gammaPDF = 1.
  return gammaPDF == 1. ? 2.*Iz : 0;
}

//==========================================================================

} // end namespace Pythia8
