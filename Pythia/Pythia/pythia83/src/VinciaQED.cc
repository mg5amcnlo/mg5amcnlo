// VinciaQED.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for Vincia's QED
// shower class and related auxiliary methods. Main author is Rob
// Verheyen.

#include "Pythia8/VinciaQED.h"

namespace Pythia8 {

using namespace VinciaConstants;

//==========================================================================

// Class for QED emissions.

//--------------------------------------------------------------------------

// Initialize the pointers.

void QEDemitElemental::initPtr(Rndm* rndmPtrIn,
  PartonSystems* partonSystemsPtrIn) {
  rndmPtr = rndmPtrIn;
  partonSystemsPtr = partonSystemsPtrIn;
  isInitPtr = true;
}

//--------------------------------------------------------------------------

// Initialize.

void QEDemitElemental::init(Event &event, int xIn, int yIn, double shhIn,
    double verboseIn) {

  if (!isInitPtr) printOut(__METHOD_NAME__, "initPtr not called");
  x = xIn;
  y = yIn;
  shh = shhIn;
  hasTrial = false;
  isII = false;
  isIF = false;
  isFF = false;
  isRF = false;
  isIA = false;
  isDip = false;

  // If an II antenna, make sure x is the positive pz state.
  if (!event[x].isFinal() && !event[y].isFinal() && event[x].pz() < 0)
      swap(x,y);

  // If an IF/RF antenna, make sure x is the initial state.
  if (event[x].isFinal() && !event[y].isFinal()) swap(x,y);

  // If a dipole, make sure x is the emitting object.
  if (event[x].isFinal() && event[y].isFinal())
    if (!event[x].isCharged() || event[y].isCharged()) swap(x,y);

  idx = event[x].id();
  idy = event[y].id();
  mx2 = max(0., event[x].m2());
  my2 = max(0., event[y].m2());
  ex = event[x].e();
  ey = event[y].e();
  m2Ant = m2(event[x], event[y]);
  sAnt = 2*dot4(event[x], event[y]);
  QQ = - event[x].charge() * event[y].charge();

  // II.
  if (!event[x].isFinal() && !event[y].isFinal()) isII = true;

  // IF/RF.
  if (!event[x].isFinal() && event[y].isFinal()) {
    // QQ is flipped for IF antennae.
    QQ = -QQ;
    // Check if initial state is in a beam.
    int mother1 = event[x].mother1();
    // Check if initial particle is A or B.
    if (mother1 <= 2) {
      isIF = true;
      if (event[x].pz() > 0) isIA = true;
    // Otherwise it's a resonance decay.
    } else isRF = true;
  }

  // FF.
  if (event[x].isFinal() && event[y].isFinal()) isFF = true;
  isInit = true;
  verbose = verboseIn;

}

//--------------------------------------------------------------------------

// Initialize.

void QEDemitElemental::init(Event &event, int xIn, vector<int> iRecoilIn,
    double shhIn, double verboseIn) {

  x = xIn;
  iRecoil = iRecoilIn;
  shh = shhIn;
  hasTrial = false;
  isII = false;
  isIF = false;
  isFF = false;
  isRF = false;
  isIA = false;
  isDip = true;
  idx = event[x].id();
  mx2 = max(0., event[x].m2());

  // Compute total recoiler momentum.
  Vec4 pRecoil;
  for (int i = 0; i < (int)iRecoil.size(); i++)
    pRecoil += event[iRecoil[i]].p();
  my2 = max(0., pRecoil.m2Calc());
  m2Ant = (pRecoil + event[xIn].p()).m2Calc();
  sAnt = 2*pRecoil*event[xIn].p();
  QQ = 1;
  isInit = true;
  verbose = verboseIn;

}

//--------------------------------------------------------------------------

// Generate a trial point.

double QEDemitElemental::generateTrial(Event &event, double q2Start,
  double q2Low, double alphaIn, double cIn) {

  if (!isInit) return 0.;

  if (hasTrial) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__, "Elemental has a trial already.");
    return q2Sav;
  }
  q2Sav = 0.;
  double q2TrialNow = 0.;
  alpha = alphaIn;
  c = cIn;

  // FF.
  if (isFF || isDip) {
    // Adjust starting scale.
    q2Start = min(q2Start, sAnt/4.);
    if (q2Start < q2Low) {
      if (verbose >= DEBUG) printOut(__METHOD_NAME__,
        "No phase space for FF in this window.");
      return 0.;
    }

    // Compute phase space constants.
    double lambda = m2Ant*m2Ant + mx2*mx2 + my2*my2
      - 2.*m2Ant*mx2 - 2.*m2Ant*my2 - 2.*mx2*my2;
    // zMin is identical for all instances.
    double zMin = (4*q2Low/sAnt < 1E-8) ? q2Low/sAnt
      : 0.5*(1. - sqrt(1. - 4*q2Low/sAnt));

    // Generate scale for eikonal piece.
    if (true) {
      double Iz = (zMin < 1E-8) ? -2*log(zMin) - 2*zMin - pow2(zMin)
        : 2*log((1-zMin)/zMin);
      double comFac = 2*M_PI*sqrt(lambda)/alpha/Iz/c/sAnt;
      double q2New  = q2Start*pow(rndmPtr->flat(), comFac);
      if (q2New > q2TrialNow) {
        q2TrialNow = q2New;
        zetaSav = 1/(exp(Iz*(0.5 - rndmPtr->flat())) + 1);
        // Note: return infinity if zeta=1 or =0 within machine precision
        // to make sure it gets vetoed later.
        sxjSav  = zetaSav != 1 ? sqrt(sAnt*q2TrialNow*zetaSav/((1-zetaSav))) :
          numeric_limits<double>::infinity();
        syjSav  = zetaSav != 0 ? sqrt(sAnt*q2TrialNow*(1-zetaSav)/(zetaSav)) :
          numeric_limits<double>::infinity();
      }
    }
    // Generate scale for additional W piece on x.
    if (isFF && abs(idx) == 24) {
      double Iz = (zMin < 1E-8) ?
        -log(zMin) - zMin - pow2(zMin)/2. : log((1-zMin)/zMin);
      double comFac = 3.*M_PI*sqrt(lambda)/alpha/Iz/c/sAnt/2.;
      double q2New = q2Start*pow(rndmPtr->flat(), comFac);
      if (q2New > q2TrialNow) {
        q2TrialNow = q2New;
        double r = rndmPtr->flat();
        zetaSav  = (zMin < 1E-8) ? 1 - pow(zMin,r)*(1. - (1.-r)*zMin)
          : 1 - pow(zMin,r)*pow(1.-zMin, 1.-r);
        sxjSav   = q2TrialNow/zetaSav;
        syjSav   = zetaSav*sAnt;
      }
    }
    // Generate scale for additional W piece on y.
    if (isFF && abs(idy) == 24) {
      double Iz = (zMin < 1E-8) ? -log(zMin) - zMin - pow2(zMin)/2.
        : log((1-zMin)/zMin);
      double comFac = 3.*M_PI*sqrt(lambda)/alpha/Iz/c/sAnt/2.;
      double q2New  = q2Start*pow(rndmPtr->flat(), comFac);
      if (q2New > q2TrialNow) {
        q2TrialNow = q2New;
        double r = rndmPtr->flat();
        zetaSav  = (zMin < 1E-8) ? 1 - pow(zMin,r)*(1. - (1.-r)*zMin)
          : 1 - pow(zMin,r)*pow(1.-zMin, 1.-r);
        sxjSav   = zetaSav*sAnt;
        syjSav   = q2TrialNow/zetaSav;
      }
    }
  }

  // IF.
  if (isIF) {
    // Compute exmax and sjkMax.
    double exUsed = 0;
    int nSys = partonSystemsPtr->sizeSys();
    for (int i=0; i<nSys; i++) {
      int iEv;
      if (isIA) iEv = partonSystemsPtr->getInA(i);
      else iEv = partonSystemsPtr->getInB(i);
      exUsed += event[iEv].e();
    }
    double exMax = sqrt(shh)/2.0 - (exUsed-ex);
    double sjkMax = sAnt*(exMax-ex)/ex;

    // Adjust starting scale.
    q2Start = min(q2Start, sAnt*(exMax - ex)/ex);
    if (q2Start < q2Low) {
      if (verbose >= DEBUG) printOut(__METHOD_NAME__,
        "No phase space for IF in this window.");
      return 0.;
    }
    double zMax = sjkMax/(sjkMax + my2);
    double zMin = q2Low/sjkMax;

    // Check if there is any phase space available.
    if (zMin < zMax) {
      // Generate scale for eikonal piece.
      if (true) {
        double Iz     = log(zMax/zMin);
        double Rpdf   = 1.;
        double comFac = M_PI/alpha/Iz/c/Rpdf;
        double q2New  = q2Start*pow(rndmPtr->flat(), comFac);
        if (q2New > q2TrialNow) {
          q2TrialNow = q2New;
          zetaSav = zMin*pow(zMax/zMin, rndmPtr->flat());
          sxjSav  = sAnt*zetaSav + q2TrialNow;
          syjSav  = q2TrialNow/zetaSav;
        }
      }

      // Generate scale for additional W piece on y. The veto
      // probability for this antenna piece includes an additional
      // factor which is incorporated by a veto locally.
      if (abs(idy) == 24) {
        double Iz = log((1-zMin)/(1-zMax));
        double Rpdf = 1.;
        double comFac = 3.*M_PI/alpha/Iz/c/Rpdf/2.;
        double q2New = q2Start;
        double zetaNew, sxjNew, syjNew;
        while (true) {
          q2New  *= pow(rndmPtr->flat(), comFac);
          if (q2New < q2TrialNow) {break;}
          zetaNew = 1. - (1-zMin)*pow((1-zMax)/(1-zMin),rndmPtr->flat());
          sxjNew  = sAnt*zetaNew + q2New;
          syjNew  = q2New/zetaNew;

          // Veto probability.
          double pVeto = sAnt/(sAnt + syjNew);
          if (rndmPtr->flat() < pVeto) {
            q2TrialNow = q2New;
            zetaSav = zetaNew;
            sxjSav  = sxjNew;
            syjSav  = syjNew;
            break;
          }
        }
      }
    }
  }

  // II.
  if (isII) {
    // Adjust starting scale.
    q2Start = min(q2Start, pow2(shh-sAnt)/shh/4.);
    if (q2Start < q2Low) {
      if (verbose >= DEBUG) printOut(__METHOD_NAME__,
        "No phase space for II in this window.");
      return 0.;
    }

    // Generate scale for eikonal piece.
    if (true) {
      double zMin = 0.5*(shh-sAnt -
        sqrt((shh-sAnt)*(shh-sAnt) - (4.*shh*q2Low)))/shh;
      double zMax = 0.5*(shh-sAnt +
        sqrt((shh-sAnt)*(shh-sAnt) - (4.*shh*q2Low)))/shh;
      if (4.*shh*q2Low/pow2((shh-sAnt)) < 1e-8)
        zMin = q2Low/(shh-sAnt);
      double Iz     = log(zMax*(1-zMin)/(1-zMax)/zMin);
      double Rpdf   = 1.;
      double comFac = M_PI/alpha/Iz/c/Rpdf;
      double q2New  = q2Start*pow(rndmPtr->flat(), comFac);
      if (q2New > q2TrialNow) {
        q2TrialNow = q2New;
        double r = rndmPtr->flat();
        double w = pow(zMax/(1-zMax), r) * pow(zMin/(1-zMin), 1.-r);
        zetaSav  = w/(1.+w);
        sxjSav   = (q2TrialNow + sAnt*zetaSav)/(1.-zetaSav);
        syjSav   = q2TrialNow/zetaSav;
      }
    }
  }

  // RF.
  if (isRF) {
    // Compute phase space constants.
    double mr2 = abs((event[x].p() - event[y].p()).m2Calc());
    double mx = sqrt(mx2);
    double my = sqrt(my2);
    double mr = sqrt(mr2);
    double lambda = mr2*mr2 + mx2*mx2 + my2*my2
      - 2.*mr2*mx2 - 2.*mr2*my2 - 2.*mx2*my2;
    double sjkMax = pow2(mx - mr) - my2;
    double sajMax = mx2 - pow2(my + mr);
    // Adjust starting scale.
    q2Start = min(q2Start, sajMax*sjkMax/(sAnt + sjkMax));

    // Generate scale for eikonal piece.
    if (true) {
      double zMin   = q2Low/sjkMax;
      double zMax   = sajMax/sAnt;
      if (zMin < zMax) {
        double Iz     = log(zMax/zMin);
        double comFac = M_PI*sqrt(lambda)*sAnt/alpha/Iz/c/pow2(sAnt+sjkMax);
        double q2New  = q2Start;
        double zetaNew, sxjNew, syjNew;
        while (true) {
          q2New *= pow(rndmPtr->flat(), comFac);
          if (q2New < q2TrialNow) break;
          zetaNew = zMin*pow(zMax/zMin, rndmPtr->flat());
          sxjNew  = sAnt*zetaNew + q2New;
          syjNew  = q2New/zetaNew;

          // Veto probability.
          double pVeto = pow2(syjNew+sAnt)/pow2(sjkMax+sAnt);
          if (rndmPtr->flat() < pVeto) {
            q2TrialNow = q2New;
            zetaSav = zetaNew;
            sxjSav  = sxjNew;
            syjSav  = syjNew;
            break;
          }
        }
      }
    }

    // Generate scale for W in initial state.
    if (abs(idx) == 24) {
      double zMin   = q2Low/(sajMax - q2Low);
      double zMax   = sjkMax/sAnt;
      if (zMin < zMax && zMin > 0) {
        double Iz     = pow2(zMax) + (1./3.)*pow3(zMax)
          - pow2(zMin) - (1./3.)*pow3(zMin);
        double comFac = 3.*M_PI*sqrt(lambda)/alpha/Iz/c/sAnt/2.;
        double q2New  = q2Start*pow(rndmPtr->flat(), comFac);

        if (q2New > q2TrialNow) {
          double a = rndmPtr->flat()*Iz + pow2(zMin) + (1./3.)*pow3(zMin);
          // Solve for zeta using Newton-Raphson.
          int n = 0;
          zetaSav = zMin;
          while(true) {
            n++;
            double f = pow2(zetaSav) + pow3(zetaSav)/3. - a;
            double fPrime  = 2.*zetaSav + pow2(zetaSav);
            double zetaNew = zetaSav - f/fPrime;
            if (zetaNew > zMax) {zetaSav = zMax; continue;}
            if (zetaNew < zMin) {zetaSav = zMin; continue;}
            if (abs(zetaNew - zetaSav) < 1E-8*zetaNew) {
              zetaSav = zetaNew;
              break;
            }
            if (n > 500) {
              printOut(__METHOD_NAME__,
                "RF(W) failed to find zeta with Newton-Raphson");
              break;
            }
            zetaSav = zetaNew;
          }
          q2TrialNow = q2New;
          sxjSav = (1.+zetaSav)*q2TrialNow/zetaSav;
          syjSav = sAnt*zetaSav;
        }
      }
    }

    // Generate scale for W in final state.
    if (abs(idy) == 24) {
      double zMin   = q2Low/sjkMax;
      double zMax   = sajMax/sAnt;
      if (zMin < zMax) {
        double Iz     = log((1-zMin)/(1-zMax));
        double comFac = 3.*M_PI*sqrt(lambda)/alpha/Iz/c/(sAnt+sjkMax)/2.;
        double q2New  = q2Start;
        double zetaNew, sxjNew, syjNew;
        while (true) {
          q2New *= pow(rndmPtr->flat(), comFac);
          if (q2New < q2TrialNow) {break;}
          zetaNew = 1. - (1-zMin)*pow((1-zMax)/(1-zMin),rndmPtr->flat());
          sxjNew  = sAnt*zetaNew + q2New;
          syjNew  = q2New/zetaNew;

          // Veto probability.
          double pVeto = (syjNew+sAnt)/(sjkMax+sAnt);
          if (rndmPtr->flat() < pVeto) {
            q2TrialNow = q2New;
            zetaSav = zetaNew;
            sxjSav = sxjNew;
            syjSav = syjNew;
            break;
          }
        }
      }
    }
  }
  phiSav = 2.*M_PI*rndmPtr->flat();
  if (q2TrialNow > q2Low) {
    hasTrial = true;
    q2Sav = q2TrialNow;
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,"Generated a new trial.");
  }
  return q2TrialNow;
}

//==========================================================================

// QEDsystem (base class) member functions.

//--------------------------------------------------------------------------

// Initialize pointers.

void QEDsystem::initPtr(Info* infoPtrIn, ParticleData* particleDataPtrIn,
  PartonSystems* partonSystemsPtrIn, Rndm* rndmPtrIn,
  Settings* settingsPtrIn, VinciaCommon* vinComPtrIn) {
  infoPtr = infoPtrIn;
  particleDataPtr = particleDataPtrIn;
  partonSystemsPtr = partonSystemsPtrIn;
  rndmPtr = rndmPtrIn;
  settingsPtr = settingsPtrIn;
  vinComPtr = vinComPtrIn;
  isInitPtr = true;
}

//--------------------------------------------------------------------------

// Update the partons systems.

void QEDsystem::updatePartonSystems() {

  if (partonSystemsPtr == nullptr) return;

  if (verbose >= DEBUG) {
    stringstream ss(" Updating iSys = ");
    ss <<iSys<<" sizeSys = " << partonSystemsPtr->sizeSys();
    printOut(__METHOD_NAME__, ss.str());
  }

  if (iSys < partonSystemsPtr->sizeSys()) {
    int iAOld(0), iBOld(0);
    if (isInitial() && partonSystemsPtr->hasInAB(iSys)) {
      iAOld = partonSystemsPtr->getInA(iSys);
      iBOld = partonSystemsPtr->getInB(iSys);
    }

    // Replace old IDs.
    for (auto it = iReplace.begin(); it!= iReplace.end() ; ++it) {
      int iOld(it->first), iNew(it->second);
      if (iAOld == iOld) partonSystemsPtr->setInA(iSys,iNew);
      else if (iBOld == iOld) partonSystemsPtr->setInB(iSys,iNew);
      partonSystemsPtr->replace(iSys, iOld, iNew);
    }

    // Add new.
    partonSystemsPtr->addOut(iSys, jNew);

    // Save sHat if we set it.
    if (shat > 0.) partonSystemsPtr->setSHat(iSys, shat);
  }

}

//==========================================================================

// QEDemitSystem member functions.

//--------------------------------------------------------------------------

// Initialize settings for current run.

void QEDemitSystem::init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn,
  int verboseIn) {

  // Verbose setting.
  if (!isInitPtr)
    printOut(__METHOD_NAME__, "QEDemitSystem:initPtr not called");
  verbose = verboseIn;

  // Set beam pointers.
  beamAPtr = beamAPtrIn;
  beamBPtr = beamBPtrIn;
  bool isHadronA  = beamAPtr->isHadron();
  bool isHadronB  = beamBPtr->isHadron();
  bool doRemnants = settingsPtr->flag("PartonLevel:Remnants");

  // QED mode for hard systems: pairing or multipole.
  qedMode        = settingsPtr->mode("Vincia:ewMode");
  // (If weak shower used for hard systems, use pairing as fallback.
  if (qedMode == 3) qedMode = 1;
  // QED mode for MPI cannot be more sophisticated than that of hard process.
  qedModeMPI     = min(settingsPtr->mode("Vincia:ewModeMPI"),qedMode);
  // Other QED settings.
  kMapTypeFinal  = settingsPtr->mode("Vincia:kineMapEWFinal");
  useFullWkernel = settingsPtr->flag("Vincia:fullWkernel");
  emitBelowHad   = (isHadronA || isHadronB) ? doRemnants : true;

  // Constants.
  TINYPDF = 1.0e-10;

  // Initialized.
  isInit = true;

}

//--------------------------------------------------------------------------

// Prepare a QED system.

void QEDemitSystem::prepare(int iSysIn, Event &event, double q2CutIn,
  bool isBelowHadIn, vector<double> evolutionWindowsIn, AlphaEM alIn) {

  if (!isInit) {
    infoPtr->errorMsg("Error in " + __METHOD_NAME__, ": not initialised.");
    return;
  }

  // Verbose output.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Input.
  iSys = iSysIn;
  shh = infoPtr->s();
  q2Cut = q2CutIn;
  isBelowHad = isBelowHadIn;
  evolutionWindows = evolutionWindowsIn;
  al = alIn;

  // Build internal system.
  buildSystem(event);

  // Done.
  if (verbose >= DEBUG) print();
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);

}

//--------------------------------------------------------------------------

// Set up antenna pairing for incoherent mode.

void QEDemitSystem::buildSystem(Event &event) {

  // Verbose output.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Clear previous antennae.
  eleVec.clear();
  eleMat.clear();
  iCoh.clear();

  // Construct hungarian algorithm solver.
  HungarianAlgorithm ha;
  // Below hadronization scale.
  if (isBelowHad && emitBelowHad) {
    map<int, vector<int> > posMap, negMap;
    vector<Vec4> posMoms, negMoms;

    // Find all (final-state) quarks and leptons.
    vector<int> iTriplets, iLeptons;
    int sysSize = partonSystemsPtr->sizeOut(iSys);
    for (int i = 0; i < sysSize; ++i) {
      int iEv = partonSystemsPtr->getOut(iSys, i);
      if (event[iEv].col() != 0 && event[iEv].acol()==0 &&
        event[iEv].isFinal()) {
        // For now, ignore quarks that are connected to junctions. In
        // principle, we could add them, and any antijunction dittos.
        bool isJun = false;
        for (int iJun = 0; iJun < event.sizeJunction(); ++iJun) {
          for (int iLeg = 0; iLeg < 3; ++iLeg) {
            if (event[iEv].col() == event.endColJunction(iJun,iLeg)) {
              isJun = true;
              break;
            }
          }
        }
        if (!isJun) iTriplets.push_back(iEv);
      }
      if (event[iEv].isLepton() && event[iEv].isCharged())
        iLeptons.push_back(iEv);
    }

    // Currently no showering below hadronisation scale if no leptons.
    if (iLeptons.size() == 0) return;

    // Sort all leptons into maps.
    for (int i = 0; i < (int)iLeptons.size(); i++) {
      int iEv = iLeptons[i];
      vector<int> iLeptonVec;
      iLeptonVec.push_back(iEv);
      if (event[iEv].chargeType() == 3) {
        posMoms.push_back(event[iEv].p());
        posMap[posMoms.size()-1] = iLeptonVec;
      }
      if (event[iEv].chargeType() == -3) {
        negMoms.push_back(event[iEv].p());
        negMap[negMoms.size()-1] = iLeptonVec;
      }
    }
    // Find all colour strings.
    for (int i = 0; i < (int)iTriplets.size(); i++) {
      // Get initial quark and add to pseudo particle.
      Vec4 pPseudo;
      int iEv = iTriplets[i];
      vector<int> iPseudoVec;
      iPseudoVec.push_back(iEv);
      pPseudo += event[iEv].p();

      // Find next colour-connected particle.
      double nLoop = 0;
      do {
        if (++nLoop > 10000) {
          infoPtr->errorMsg("Error in "+__METHOD_NAME__+": caught in "
            "infinite loop");
          break;
        }
        int colTag = event[iEv].col();
        for (int j = 0; j < sysSize; j++) {
          int jEv = partonSystemsPtr->getOut(iSys, j);
          if (event[jEv].acol() == colTag && event[jEv].isFinal()) {
            iEv = jEv;
            break;
          }
        }
        if (iEv == iPseudoVec.back()) {
          infoPtr->errorMsg("Error in " + __METHOD_NAME__,
            ": colour tracing failed.");
          break;
        }
        iPseudoVec.push_back(iEv);
        pPseudo += event[iEv].p();
      } while(!event[iEv].isQuark()&&!event[iEv].isDiquark());

      // Get charge of pseudoparticle and sort into maps.
      int chargeTypePseudo = event[iPseudoVec.front()].chargeType()
        + event[iPseudoVec.back()].chargeType();
      // Strings with only quarks are total charge 1 or -1.
      if (chargeTypePseudo == 3) {
        posMoms.push_back(pPseudo);
        posMap[posMoms.size()-1] = iPseudoVec;
      } else if (chargeTypePseudo == -3) {
        negMoms.push_back(pPseudo);
        negMap[negMoms.size()-1] = iPseudoVec;
      // Strings with a diquark can be charge 2 or -2. Add these
      // twice to list of recoilers.
      } else if (chargeTypePseudo == 6) {
        posMoms.push_back(pPseudo);
        posMap[posMoms.size()-1] = iPseudoVec;
        posMoms.push_back(pPseudo);
        posMap[posMoms.size()-1] = iPseudoVec;
      } else if (chargeTypePseudo == -6) {
        negMoms.push_back(pPseudo);
        negMap[negMoms.size()-1] = iPseudoVec;
        negMoms.push_back(pPseudo);
        negMap[negMoms.size()-1] = iPseudoVec;
      }
    }

    // If no leptons and overall hadronic system has charge = 0, do nothing.
    if (posMoms.size() == 0) return;

    // Solve assignment problem.
    vector<vector<double> > weights;
    weights.resize(posMoms.size());
    for (int i = 0; i < (int)posMoms.size(); i++) {
      weights[i].resize(negMoms.size());
      for (int j = 0; j < (int)negMoms.size(); j++)
        weights[i][j] =
          posMoms[i]*negMoms[j] - posMoms[i].mCalc()*negMoms[j].mCalc();
    }
    vector<int> assignment;
    ha.solve(weights, assignment);

    for (int i = 0; i < (int)posMoms.size(); i++) {
      int iPos = i;
      int iNeg = assignment[i];
      // Only keep antennae with at least one lepton.
      if (posMap[iPos].size() == 1 || negMap[iNeg].size() == 1) {
        eleVec.push_back(QEDemitElemental());
        eleVec.back().initPtr(rndmPtr, partonSystemsPtr);
        // If two leptons, add regular antenna.
        if (posMap[iPos].size() == 1 && negMap[iNeg].size() == 1)
          eleVec.back().init(event, posMap[iPos][0], negMap[iNeg][0], shh,
            verbose);
        // If lepton + pseudoparticle, add dipole.
        if (posMap[iPos].size() == 1 && negMap[iNeg].size() != 1)
          eleVec.back().init(event, posMap[iPos][0], negMap[iNeg], shh,
            verbose);
        if (posMap[iPos].size()!=1 && negMap[iNeg].size()==1)
          eleVec.back().init(event, negMap[iNeg][0], posMap[iPos], shh,
            verbose);
      }
    }

  // Above hadronization scale.
  } else if (!isBelowHad) {
    // Collect relevant particles.
    int sysSize = partonSystemsPtr->sizeAll(iSys);
    for (int i = 0; i < sysSize; i++) {
      int iEv = partonSystemsPtr->getAll(iSys, i);
      if (event[iEv].isCharged()) iCoh.push_back(iEv);
    }

    // Catch cases (like hadron->partons decays) where an explicit
    // charged mother may not have been added to the partonSystem as a
    // resonance.
    if (partonSystemsPtr->getInA(iSys) == 0 &&
        partonSystemsPtr->getInB(iSys) == 0 &&
        partonSystemsPtr->getInRes(iSys) == 0) {
      // Guess that the decaying particle is mother of first parton.
      int iRes = event[partonSystemsPtr->getOut(iSys, 0)].mother1();
      if (iRes != 0 && event[iRes].isCharged()) {
        // Check daughter list consistent with whole system.
        int ida1 = event[iRes].daughter1();
        int ida2 = event[iRes].daughter2();
        if (ida2 > ida1) {
          bool isOK = true;
          for (int i=0; i<partonSystemsPtr->sizeOut(iSys); ++i)
            if (partonSystemsPtr->getOut(iSys,i) < ida1
              || partonSystemsPtr->getOut(iSys,i) > ida2) isOK = false;
          if (isOK) {iCoh.push_back(iRes);}
        }
      }
    }

    // First check charge conservation.
    int chargeTypeTot = 0;
    for (int i = 0; i < (int)iCoh.size(); i++) {
      double cType = event[iCoh[i]].chargeType();
      chargeTypeTot += (event[iCoh[i]].isFinal() ? cType : -cType);
    }

    if (chargeTypeTot != 0) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Charge not conserved above hadronization scale");
      if (verbose >= REPORT) {
        printOut(__METHOD_NAME__, "Printing events and systems");
        event.list();
        partonSystemsPtr->list();
      }
    }

    // Decide whether to use pairing (1) or coherent (2) algorithm.
    int qedModeSys = qedMode;
    if (iSys > 0 && partonSystemsPtr->hasInAB(iSys)) qedModeSys = qedModeMPI;

    // Dipole-Pairing Algorithm.
    if (qedModeSys == 1) {
      vector<vector<int> > posChargeTypes;
      posChargeTypes.resize(3);
      vector<vector<int> > negChargeTypes;
      negChargeTypes.resize(3);

      for (int i = 0; i < (int)iCoh.size(); i++) {
        int iEv = iCoh[i];
        // Separate particles into charge types.
        double Q = event[iEv].charge();
        // Get index in pos/negChargeTypes.
        int n = abs(event[iEv].chargeType()) - 1;
        // Flip charge contribution of initial state.
        if (!event[iEv].isFinal()) {Q = -Q;}
        if (Q > 0)  posChargeTypes[n].push_back(iEv);
        else negChargeTypes[n].push_back(iEv);
      }

      // Clear list of charged particles.
      iCoh.clear();

      // Solve assignment problems.
      for (int i=0; i<3; i++) {
        int posSize = posChargeTypes[i].size();
        int negSize = negChargeTypes[i].size();
        int maxSize = max(posSize,negSize);
        if (maxSize > 0) {
          vector<vector<double> > weights;
          weights.resize(maxSize);
          // Set up matrix of weights.
          for (int x = 0; x < maxSize; x++) {
            weights[x].resize(maxSize);
            for (int y = 0; y < maxSize; y++) {
              // If either index is out of range. Add some random
              // large weight.
              double wIn = (0.9 + 0.2*rndmPtr->flat())*1E300;
              if (x < posSize && y < negSize) {
                int xEv = posChargeTypes[i][x];
                int yEv = negChargeTypes[i][y];
                wIn = event[xEv].p()*event[yEv].p()
                  - event[xEv].m()*event[yEv].m();
              }
              weights[x][y] = wIn;
            }
          }

          // Find solution.
          vector<int> assignment;
          ha.solve(weights, assignment);

          // Add pairings to list of emitElementals.
          // Add unpaired particles to index list for coherent algorithm.
          for (int j = 0; j < maxSize; j++) {
            int x = j;
            int y = assignment[j];
            if (x < posSize && y < negSize) {
              int xEv = posChargeTypes[i][x];
              int yEv = negChargeTypes[i][y];
              eleVec.push_back(QEDemitElemental());
              eleVec.back().initPtr(rndmPtr, partonSystemsPtr);
              eleVec.back().init(event, xEv, yEv, shh, verbose);
            } else if (x < posSize) {
              int xEv = posChargeTypes[i][x];
              iCoh.push_back(xEv);
            } else if (y < negSize) {
              int yEv = negChargeTypes[i][y];
              iCoh.push_back(yEv);
            }
          }
        }
      }
    }

    // Create eleMat.
    eleMat.resize(iCoh.size());
    for (int i = 0; i < (int)iCoh.size(); i++) {
      eleMat[i].resize(i);
      for (int j = 0; j < i; j++) {
        eleMat[i][j].initPtr(rndmPtr, partonSystemsPtr);
        eleMat[i][j].init(event, iCoh[i], iCoh[j], shh, verbose);
      }
    }

    // Compute overestimate constant.
    cMat = 0;
    for (int i = 0; i < (int)eleMat.size(); i++)
      for (int j = 0; j < i; j++) cMat += max(eleMat[i][j].QQ, 0.);
  }

  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__,"end (nEmitters(II+IF+RF+FF) ="
      + num2str((int)eleVec.size())+" (pairs) + "+num2str((int)eleMat.size())
      + " (multipole))");
  }

}

//--------------------------------------------------------------------------

// Generate a trial scale.

double QEDemitSystem::q2Next(Event &event, double q2Start) {
  // Don't do anything if empty!
  if (eleVec.size() == 0 && eleMat.size()==0 ) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,"Nothing to do.");
    return 0.;
  }

  if (verbose >= DEBUG) {
    stringstream ss;
    ss<<"Starting evolution at q2Start = "<<q2Start;
    printOut(__METHOD_NAME__,ss.str());
  }

  // Check if qTrial is below the cutoff.
  if (q2Start < q2Cut || evolutionWindows.size() == 0) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,"Below cutoff.");
    return 0;
  }

  // Find lower value from evolution window.
  int iEvol = evolutionWindows.size() - 1;
  while (iEvol >= 1 && q2Start <= evolutionWindows[iEvol]) iEvol--;
  double q2Low = evolutionWindows[iEvol];
  if (q2Low < 0)
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": Evolution window < 0");
  double q2Trial = 0;

  // Generate a scale.
  double alphaMax = al.alphaEM(q2Start);

  // Pull scales from eleVec.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss<<"Looping over "<<eleVec.size()<< " emit pairing elementals.";
    printOut(__METHOD_NAME__,ss.str());
  }
  for (int i = 0; i < (int)eleVec.size(); i++) {
    double c = eleVec[i].QQ;
    double q2New = eleVec[i].generateTrial(event, q2Start, q2Low, alphaMax, c);
    if (q2New > q2Low && q2New > q2Trial) {
      q2Trial = q2New;
      eleTrial = &eleVec[i];
      trialIsVec = true;
    }
  }

  // Pull scales from eleMat.
  for (int i = 0; i < (int)eleMat.size(); i++) {
    if (verbose >= DEBUG) {
      stringstream ss;
      ss<<"Looping over "<<eleMat[i].size()<<" coherent elementals.";
      printOut(__METHOD_NAME__,ss.str());
    }
    for (int j = 0; j < i; j++) {
      double q2New = eleMat[i][j].generateTrial(event, q2Start, q2Low,
        alphaMax, cMat);
      if (q2New > q2Low && q2New > q2Trial) {
        q2Trial = q2New;
        eleTrial = &eleMat[i][j];
        trialIsVec = false;
      }
    }
  }

  // Verbose output.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss<<"Generated a new trial = "<< q2Trial;
    ss<<" in window = "<<iEvol << " (q2Low = "<<q2Low<<" )";
    printOut(__METHOD_NAME__,ss.str());
  }

  // Check if evolution window was crossed.
  if (q2Trial < q2Low) {
    if (iEvol == 0) {
      if (verbose >= DEBUG) printOut(__METHOD_NAME__,
        "Dropped below QED cutoff.");
      return 0;
    }
    else if (verbose >= DEBUG) printOut(__METHOD_NAME__,
      "Trial was below window lower bound. Try again. ");
    // Reset all trials.
    for (int i = 0; i < (int)eleVec.size(); i++) eleVec[i].hasTrial = false;
    for (int i=0; i<(int)eleMat.size(); i++)
      for (int j=0; j<i; j++) eleMat[i][j].hasTrial = false;
    return q2Next(event, q2Low);
  }

  // Otherwise return trial scale.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"Done");
  return q2Trial;

}

//--------------------------------------------------------------------------

// Check the veto. Return false if branching should be vetoed.

bool QEDemitSystem::acceptTrial(Event &event) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  // Mark trial as used.
  eleTrial->hasTrial = false;

  // Pre- and post-branching momenta.
  vector<Vec4> pOld;
  pNew.clear();

  // Global recoil momenta.
  pRec.clear();
  iRec.clear();

  // II.
  if (eleTrial->isII) {
    double saj = eleTrial->sxjSav;
    double sbj = eleTrial->syjSav;
    double phi = eleTrial->phiSav;
    double sAB = eleTrial->sAnt;
    double sab = sAB + saj + sbj;

    // Pre-branching momenta.
    pOld.push_back(event[eleTrial->x].p());
    pOld.push_back(event[eleTrial->y].p());

    // Collect the recoiling final state particles.
    int sysSize = partonSystemsPtr->sizeAll(iSys);
    for (int i = 0; i < sysSize; i++) {
      int iEv = partonSystemsPtr->getAll(iSys, i);
      if (iEv < 0 || !event[iEv].isFinal()) continue;
      if (iEv == eleTrial->x || iEv == eleTrial->y) continue;
      pRec.push_back(event[iEv].p());
      iRec.push_back(iEv);
    }

    // Kinematics.
    if (!vinComPtr->map2to3II(pNew, pRec, pOld, sAB,saj,sbj,sab, phi))
      return false;

    // Check if new energies don't exceed hadronic maxima.
    double eaUsed = 0, ebUsed = 0;
    int nSys = partonSystemsPtr->sizeSys();
    for (int i = 0; i < nSys; i++) {
      eaUsed += event[partonSystemsPtr->getInA(i)].e();
      ebUsed += event[partonSystemsPtr->getInB(i)].e();
    }
    if ((eaUsed - pOld[0].e() + pNew[0].e()) > 0.98*sqrt(shh)/2.) return false;
    if ((ebUsed - pOld[1].e() + pNew[2].e()) > 0.98*sqrt(shh)/2.) return false;
  }

  // IF.
  else if (eleTrial->isIF) {
    double saj = eleTrial->sxjSav;
    double sjk = eleTrial->syjSav;
    double phi = eleTrial->phiSav;
    double sAK = eleTrial->sAnt;
    double sak = sAK + sjk - saj;
    double mK2 = eleTrial->my2;

    // Check phase space.
    if (sak < 0 || saj*sjk*sak - saj*saj*mK2 < 0) {return false;}

    // Pre-branching momenta.
    pOld.push_back(event[eleTrial->x].p());
    pOld.push_back(event[eleTrial->y].p());

    // Kinematics.
    // (Could in principle allow for global, but not done for now since
    // more complicated and difference presumably too small to be relevant.)
    if (!vinComPtr->map2to3IFlocal(pNew, pOld, sAK, saj, sjk, sak, phi,
        mK2, 0, mK2)) return false;

    // Check if new energy doesn't exceed the hadronic maximum.
    double eaUsed = 0;
    int nSys = partonSystemsPtr->sizeSys();
    for (int i = 0; i < nSys; i++) {
      int iEv;
      if (eleTrial->isIA) iEv = partonSystemsPtr->getInA(i);
      else iEv = partonSystemsPtr->getInB(i);
      eaUsed += event[iEv].e();
    }
    if ((eaUsed - pOld[0].e() + pNew[0].e()) > 0.98*sqrt(shh)/2.) return false;
  }

  // RF.
  else if (eleTrial->isRF) {
    double saj = eleTrial->sxjSav;
    double sjk = eleTrial->syjSav;
    double sAK = eleTrial->sAnt;
    double sak = sAK + sjk - saj;
    double phi = eleTrial->phiSav;
    double mA2 = eleTrial->mx2;
    double mK2 = eleTrial->my2;

    // Check phase space.
    if (sak < 0 || saj*sjk*sak - saj*saj*mK2 - sjk*sjk*mA2 < 0) return false;

    // Pre-branching momenta.
    pOld.push_back(event[eleTrial->x].p());
    pOld.push_back(event[eleTrial->y].p());

    // Collect the recoiling final state particles.
    int sysSize = partonSystemsPtr->sizeAll(iSys);
    for (int i = 0; i < sysSize; i++) {
      int iEv = partonSystemsPtr->getAll(iSys, i);
      if (iEv < 0 || !event[iEv].isFinal()) {continue;}
      if (iEv == eleTrial->x || iEv == eleTrial->y) {continue;}
      pRec.push_back(event[iEv].p());
      iRec.push_back(iEv);
    }

    // Do kinematics.
    vector<double> masses;
    masses.push_back(sqrt(mA2));
    masses.push_back(0.);
    masses.push_back(sqrt(mK2));
    masses.push_back(sqrtpos(mA2+mK2-sAK));
    vector<double> invariants;
    invariants.push_back(sAK);
    invariants.push_back(saj);
    invariants.push_back(sjk);
    invariants.push_back(sak);
    vector<Vec4> pAfter;
    vector<Vec4> pBefore = pOld;
    pBefore.insert(pBefore.end(), pRec.begin(), pRec.end());
    if (!vinComPtr->map2toNRF(pAfter, pBefore, 0, 1, invariants, phi,
        masses)) return false;
    pNew.push_back(pAfter[0]);
    pNew.push_back(pAfter[1]);
    pNew.push_back(pAfter[2]);

    // Replace momenta with boosted counterpart.
    pRec.clear();
    for (int i = 3; i < (int)pAfter.size(); i++) pRec.push_back(pAfter[i]);

    // Check if nothing got messed up along the way.
    if (pRec.size() != iRec.size()) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": inconsistent recoilers in RF kinematics.");
      return false;
    }
  }

  // FF.
  else if (eleTrial->isFF) {
    double sIK = eleTrial->m2Ant - eleTrial->mx2 - eleTrial->my2;
    double sij = eleTrial->sxjSav;
    double sjk = eleTrial->syjSav;
    double sik = sIK - sij - sjk;
    double mi  = sqrt(eleTrial->mx2);
    double mk  = sqrt(eleTrial->my2);
    double phi = eleTrial->phiSav;

    vector<double> invariants;
    invariants.push_back(sIK);
    invariants.push_back(sij);
    invariants.push_back(sjk);

    vector<double> masses;
    masses.push_back(mi);
    masses.push_back(0);
    masses.push_back(mk);

    // Check phase space.
    if (sik < 0) return false;
    if (sij*sjk*sik - pow2(sij)*pow2(mk) - pow2(sjk)*pow2(mi) < 0)
      return false;

    // Pre-branching momenta.
    pOld.push_back(event[eleTrial->x].p());
    pOld.push_back(event[eleTrial->y].p());

    // Kinematics.
    if (!vinComPtr->map2to3FF(pNew, pOld, kMapTypeFinal, invariants, phi,
        masses)) return false;
  }

  // Dipole.
  else if (eleTrial->isDip) {
    // Construct recoiler momentum.
    Vec4 pk;
    for (int i = 0; i < (int)eleTrial->iRecoil.size(); i++)
      pk += event[eleTrial->iRecoil[i]].p();
    double sIK = eleTrial->m2Ant - eleTrial->mx2 - eleTrial->my2;
    double sij = eleTrial->sxjSav;
    double sjk = eleTrial->syjSav;
    double sik = sIK - sij - sjk;
    double mi  = sqrt(eleTrial->mx2);
    double mk  = pk.mCalc();
    double phi = eleTrial->phiSav;

    vector<double> invariants;
    invariants.push_back(sIK);
    invariants.push_back(sij);
    invariants.push_back(sjk);

    vector<double> masses;
    masses.push_back(mi);
    masses.push_back(0);
    masses.push_back(mk);

    // Check phase space.
    if (sik < 0) {return false;}
    if (sij*sjk*sik - pow2(sij)*pow2(mk) - pow2(sjk)*pow2(mi) < 0)
      return false;

    // Pre-branching momenta.
    pOld.push_back(event[eleTrial->x].p());
    pOld.push_back(pk);

    // Kinematics.
    if (!vinComPtr->map2to3FF(pNew, pOld, kMapTypeFinal, invariants, phi,
        masses)) return false;
  }

  // Save.
  pRecSum = pOld[1];

  Vec4 pPhot = pNew[1];
  Vec4 px = pNew[0];
  Vec4 py = pNew[2];
  int x = eleTrial->x;
  int y = eleTrial->y;
  double sxj = eleTrial->sxjSav;
  double syj = eleTrial->syjSav;
  double sxy = px*py*2.;

  // Compute veto probability.
  double pVeto = 1.;

  // Add alpha veto.
  pVeto *= al.alphaEM(eleTrial->q2Sav) / eleTrial->alpha;
  if (pVeto > 1) printOut(__METHOD_NAME__, "Alpha increased");

  // Add antenna veto. Simple veto for eleTrial in eleVec.
  if (trialIsVec) {
    // Note that charge factor is included at generation step.
    double aTrialNow = aTrial(eleTrial, sxj, syj, sxy);
    double aPhysNow = aPhys(eleTrial, sxj, syj, sxy);

    if (aPhysNow/aTrialNow > 1.001) {
      stringstream ss1;
      ss1 << "at q = " << sqrt(eleTrial->q2Sav)
          <<" GeV,  ratio = " << aPhysNow/aTrialNow;
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": incorrect "
        +"overestimate (dipole)",ss1.str());

      if (verbose >= DEBUG) {
        stringstream ss2, ss3;
        if (eleTrial->isFF) ss2 << "Antenna is FF";
        if (eleTrial->isIF) ss2 << "Antenna is IF";
        if (eleTrial->isRF) ss2 << "Antenna is RF";
        if (eleTrial->isII) ss2 << "Antenna is II";
        printOut(__METHOD_NAME__, ss2.str());
        printOut(__METHOD_NAME__, ss3.str());
      }
    }
    pVeto *= aPhysNow/aTrialNow;

  // Construct full branching kernel for eleTrial in eleMat. Perform
  // sector check too.
  } else {
    double aSectorNow = aPhys(eleTrial, sxj, syj, sxy);
    double aTrialFull = eleTrial->c*aTrial(eleTrial, sxj, syj, sxy);
    double aPhysFull  = 0;

    // Build map of momenta & invariants with new photon.
    map<int, double> sj;
    map<int, Vec4> p;
    // Loop over the first column in eleMat.
    for (int i = 0; i < (int)iCoh.size(); i++) {
      int iEv = iCoh[i];
      // If the particle is in eleTrial, use shower variables.
      if (iEv == x) {
        p[iEv] = px;
        sj[iEv] = sxj;
      } else if (iEv == y) {
        p[iEv] = py;
        sj[iEv] = syj;
      // Otherwise get the momenta elsewhere
      } else {
        // If global recoil, get them from pRec.
        if (eleTrial->isII) {
          // Find index.
          for (int j = 0; j < (int)iRec.size(); j++) {
            if (iEv == iRec[j]) {
              p[iEv] = pRec[j];
              sj[iEv] = 2.*pRec[j]*pPhot;
              break;
            }
          }
        // Otherwise use momentum from event.
        } else {
          p[iEv] = event[iEv].p();
          sj[iEv] = 2.*event[iEv].p()*pPhot;
        }
      }
    }

    // Then build aPhys.
    for (int v=0; v<(int)eleMat.size(); v++) {
      for (int w=0; w<v; w++) {
        double sxjNow = sj[eleMat[v][w].x];
        double syjNow = sj[eleMat[v][w].y];
        double sxyNow = 2.*p[eleMat[v][w].x]*p[eleMat[v][w].y];
        double aPhysNow = aPhys(&eleMat[v][w], sxjNow, syjNow, sxyNow);

        // Sector veto.
        if (aPhysNow > aSectorNow) return false;

        // Add aPhysNew to aPhys.
        aPhysFull += eleMat[v][w].QQ*aPhysNow;
      }
    }
    // Set aPhys to zeto if below zero.
    if (aPhysFull < 0) {aPhysFull = 0;}

    // Check overestimate.
    if (aPhysFull/aTrialFull > 1) {
      stringstream ss1;
      ss1 << "at q = " << sqrt(eleTrial->q2Sav)
          << " GeV,  ratio = " << aPhysFull/aTrialFull;
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": incorrect "
        +"overestimate (multipole)",ss1.str());
    }
    // Add antenna veto.
    pVeto *= aPhysFull/aTrialFull;
  }

  // Add PDF veto.
  if (eleTrial->isIF) {
    pVeto *= pdfRatio(eleTrial->isIA, pOld[0].e(), pNew[0].e(),
      eleTrial->idx, eleTrial->q2Sav);
  }
  if (eleTrial->isII) {
    pVeto *= pdfRatio(true,  pOld[0].e(), pNew[0].e(),
      eleTrial->idx, eleTrial->q2Sav);
    pVeto *= pdfRatio(false, pOld[1].e(), pNew[2].e(),
      eleTrial->idy, eleTrial->q2Sav);
  }

  // Perform veto.
  if (rndmPtr->flat() > pVeto) {
    return false;
  }

  // Done.
  if (verbose >= DEBUG) {
    event.list();
    partonSystemsPtr->list();
    printOut(__METHOD_NAME__,"end", dashLen);
  }
  return true;

}

//--------------------------------------------------------------------------

// Update the event after accepted emission.

void QEDemitSystem::updateEvent(Event &event) {
  // Clear information for replacing later in partonSystems.
  iReplace.clear();
  shat = 0.;

  Vec4 pPhot = pNew[1];
  Vec4 px = pNew[0];
  Vec4 py = pNew[2];
  int x = eleTrial->x;
  int y = eleTrial->y;

  // Invariants to determine order of photon mothers
  double sxj = eleTrial->sxjSav;
  double syj = eleTrial->syjSav;

  // Different procedures for dipoles and antennae.
  // 1) If it is a dipole:
  if (eleTrial->isDip) {
    // Set up new particles.
    Particle newPhoton(22, 51, 0, 0, 0, 0, 0, 0, pPhot);
    Particle newPartx = event[x];
    newPartx.p(px);

    // Add to event and save updates to be done on PartonSystems later.
    int xNew = event.append(newPartx);
    jNew = event.append(newPhoton);
    iReplace[x] = xNew;

    // Set old particles to negative.
    event[x].statusNeg();

    // Update mother-daughter structure.
    event[xNew].mothers(x,0);
    event[jNew].mothers(x,0);
    event[x].daughters(xNew, jNew);
    event[xNew].daughters(0,0);
    event[jNew].daughters(0,0);
    event[xNew].statusCode(51);
    event[jNew].statusCode(51);

    // Boost momenta and update.
    for (int i = 0; i < (int)eleTrial->iRecoil.size(); i++) {
      int iDipRec = eleTrial->iRecoil[i];
      Vec4 pDipRec = event[iDipRec].p();
      pDipRec.bstback(pRecSum);
      pDipRec.bst(pNew[2]);
      // Copy the recoiler.
      int iDipRecNew = event.copy(iDipRec, 52);
      // Change the momentum.
      event[iDipRecNew].p(pDipRec);
      // Save update to be done on PartonSystems later.
      iReplace[iDipRec] = iDipRecNew;
    }

  }
  // 2) If it is an RF:
  else if (eleTrial->isRF) {

    // Set up new particles.
    Particle newPhoton(22, 51, 0, 0, 0, 0, 0, 0, pPhot);
    Particle newParty = event[y];
    newParty.p(py);

    // Add branched particles to event.
    int yNew;
    if (x < y) {
      jNew = event.append(newPhoton);
      yNew = event.append(newParty);
    } else {
      yNew = event.append(newParty);
      jNew = event.append(newPhoton);
    }

    // Save update to be done on PartonSystems later.
    iReplace[y] = yNew;

    // Set old particles to negative.
    event[y].statusNeg();
    event[jNew].mothers(y,0);
    event[yNew].mothers(y,0);
    event[jNew].daughters(0,0);
    event[yNew].daughters(0,0);
    event[y].daughters(yNew,jNew);
    event[yNew].statusCode(51);

    // Update event for global recoil.
    for (int j=0; j<event.size(); j++) {
      if (event[j].isFinal()) {
        for (int k=0; k<(int)iRec.size(); k++) {
          if (iRec[k] == j) {
            // Copy the recoiler.
            int inew = event.copy(j, 52);
            // Change the momentum.
            event[inew].p(pRec[k]);
            // Save update to be done on PartonSystems later.
            iReplace[iRec[k]] = inew;
          }
        }
      }
    }
  }
  // 3) If it is an antenna:
  else {
    // Set up new particles.
    Particle newPhoton(22, 51, 0, 0, 0, 0, 0, 0, pPhot);
    Particle newPartx = event[x];
    newPartx.p(px);
    Particle newParty = event[y];
    newParty.p(py);

    // Add branched particles to event.
    int xNew, yNew;
    if (x < y) {
      xNew = event.append(newPartx);
      jNew = event.append(newPhoton);
      yNew = event.append(newParty);
    } else {
      yNew = event.append(newParty);
      jNew = event.append(newPhoton);
      xNew = event.append(newPartx);
    }

    // Save changes to be done on PartonSystems later.
    iReplace[x] = xNew;
    iReplace[y] = yNew;

    // Set old particles to negative.
    event[x].statusNeg();
    event[y].statusNeg();

    // Update everything.
    if (eleTrial->isII) {
      event[xNew].mothers(event[x].mother1(), event[x].mother2());
      if (sxj < syj) event[jNew].mothers(xNew, yNew);
      else event[jNew].mothers(yNew, xNew);
      event[yNew].mothers(event[y].mother1(), event[y].mother2());
      event[x].mothers(xNew, 0);
      event[y].mothers(yNew, 0);
      event[xNew].daughters(jNew, x);
      event[yNew].daughters(jNew, y);
      event[jNew].daughters(0,0);
      event[xNew].status(-41);
      event[yNew].status(-41);
      event[jNew].status(43);

      // Update beam daughters.
      if (iSys == 0) {
        bool founda = false;
        bool foundb = false;
        for (int i = 0; i < (int)event.size(); i++) {
          if (!founda)
            if (event[i].daughter1() == x) {
              event[i].daughters(xNew, 0);
              founda = true;
            }
          if (!foundb)
            if (event[i].daughter1() == y) {
              event[i].daughters(yNew, 0);
              foundb = true;
            }
          if (founda && foundb) break;
        }
      }

      // Update event for global recoil.
      for (int j=0; j<event.size(); j++) {
        if (event[j].isFinal()) {
          for (int k=0; k<(int)iRec.size(); k++) {
            if (iRec[k] == j) {
              // Copy the recoiler.
              int inew = event.copy(j, 44);
              // Change the momentum.
              event[inew].p(pRec[k]);
              // Save update to be done on PartonSystems later.
              iReplace[iRec[k]] = inew;
            }
          }
        }
      }

      // Save sHat for parton systems.
      shat = (event[xNew].p() +  event[yNew].p()).m2Calc();

      // Update beams.
      BeamParticle& beam1 = *beamAPtr;
      BeamParticle& beam2 = *beamBPtr;

      // Check that x is always a with pz>0.
      if (event[xNew].pz() < 0) {
        printOut(__METHOD_NAME__, "Swapped II  antenna");
        beam1 = *beamBPtr;
        beam2 = *beamAPtr;
      }
      beam1[iSys].update(xNew, event[xNew].id(), event[xNew].e()/beam1.e());
      beam2[iSys].update(yNew, event[yNew].id(), event[yNew].e()/beam2.e());
    }

    if (eleTrial->isIF) {
      event[xNew].mothers(event[x].mother1(), event[x].mother2());
      if (sxj < syj) event[jNew].mothers(xNew,y);
      else event[jNew].mothers(y,xNew);
      event[yNew].mothers(y,0);
      event[x].mothers(xNew,0);
      event[xNew].daughters(jNew,x);
      event[jNew].daughters(0,0);
      event[yNew].daughters(0,0);
      event[y].daughters(jNew, yNew);
      event[xNew].status(-41);
      event[yNew].status(43);
      event[jNew].status(43);

      // Update beam daughter.
      if (iSys == 0)
        for (int i=0; i<(int)event.size(); i++)
          if (event[i].daughter1() == x) {
            event[i].daughters(xNew, 0);
            break;
          }

      // Save sHat for PartonSystems.
      if (eleTrial->isIA) shat = (event[xNew].p()
        + event[partonSystemsPtr->getInB(iSys)].p()).m2Calc();
      else
        shat = (event[xNew].p()
          + event[partonSystemsPtr->getInA(iSys)].p()).m2Calc();

      // Update beams.
      BeamParticle& beam = (eleTrial->isIA ? *beamAPtr : *beamBPtr);
      beam[iSys].update(xNew, event[xNew].id(), event[xNew].e()/beam.e());
    }

    if (eleTrial->isFF) {
      event[xNew].mothers(x,0);
      if (sxj < syj) event[jNew].mothers(x,y);
      else event[jNew].mothers(y,x);
      event[yNew].mothers(y,0);
      event[x].daughters(xNew, jNew);
      event[y].daughters(yNew, jNew);
      event[xNew].daughters(0,0);
      event[jNew].daughters(0,0);
      event[yNew].daughters(0,0);
      event[xNew].statusCode(51);
      event[jNew].statusCode(51);
      event[yNew].statusCode(51);
    }

    // Update event pointers.
    event.restorePtrs();
  }
}

//--------------------------------------------------------------------------

// Print the internal state of a QEDemitSystem

void QEDemitSystem::print() {
  if (eleVec.size() + eleMat.size() == 0) {
    cout<< " --------  No QED Emitters in System";
    return;
  }
  cout << " --------  QEDemitSystem  ---------------------"
       << "----------------------------------------------------" << endl;
  if (eleVec.size() > 0) cout << "  Pairing elementals: " << endl;
  for (int i=0; i<(int)eleVec.size(); i++) {
    if (eleVec[i].isDip) {
      cout << "    Dipole: x = ";
      cout << eleVec[i].x << " Recoilers: (";
      for (int j=0; j<(int)eleVec[i].iRecoil.size(); j++) {
        cout << eleVec[i].iRecoil[j] << ", ";
        if (j==(int)eleVec[i].iRecoil.size()-1) {cout << ")";}
        else {cout << ", ";}
      }
    }
    else {
      cout << "  Antennae: x = " << eleVec[i].x << ", y = " << eleVec[i].y;
    }
    cout << ", QQ = " << eleVec[i].QQ << ", s = " << eleVec[i].sAnt << endl;
  }
  if (eleMat.size() > 0) cout << "  Coherent elementals: " << endl;
  for (int i=0; i<(int)eleMat.size(); i++) {
    for (int j=0; j<i; j++) {
      cout << "    x = " << eleMat[i][j].x << ", y = " << eleMat[i][j].y <<
        "  QxQy = " << num2str(eleMat[i][j].QQ,6) << ",  s = "
           << num2str(eleMat[i][j].sAnt,9) << endl;
    }
  }
  cout << " ----------------------------------------------"
       << "----------------------------------------------------" << endl;
}

//--------------------------------------------------------------------------

// Trial antenna function.

double QEDemitSystem::aTrial(QEDemitElemental* ele, double sxj, double syj,
  double sxy) {
  int idx = ele->idx;
  int idy = ele->idy;
  double ant = 0;

  // FF.
  if (ele->isFF || ele->isDip) {
    double s = sxj + syj + sxy;
    ant += 4*s/sxj/syj;
    if (ele->isFF && abs(idx) == 24) ant += 8.*s/sxj/(s - syj)/3.;
    if (ele->isFF && abs(idy) == 24) ant += 8.*s/syj/(s - sxj)/3.;
  }

  // IF.
  if (ele->isIF) {
    double s = sxj + sxy - syj;
    ant += 4*pow2(s+syj)/(s*sxj*syj);
    if (abs(idy) == 24) ant += 8.*(s + syj)/syj/(s + syj - sxj)/3.;
  }

  // II.
  if (ele->isII) {
    double s = sxy - sxj - syj;
    ant += 4*sxy*sxy/s/sxj/syj;
  }

  // RF.
  if (ele->isRF) {
    double s = sxj + sxy - syj;
    ant += 4*pow2(s+syj)/s/sxj/syj;
    if (abs(idx) == 24) ant += 8*(2.*syj/s + pow2(syj)/pow2(s))/sxj/3.;
    if (abs(idy) == 24) ant += 8.*(s + syj)/syj/(s + syj - sxj)/3.;
  }
  return ant;

}

//--------------------------------------------------------------------------

// Physical antenna function.

double QEDemitSystem::aPhys(QEDemitElemental* ele, double sxj, double syj,
  double sxy) {
  double mx2 = ele->mx2;
  double my2 = ele->my2;
  int idx = ele->idx;
  int idy = ele->idy;
  double ant = 0;

  // FF.
  if (ele->isFF) {
    double s = sxj + syj + sxy;
    // Eikonal.
    ant += 4.*sxy/sxj/syj - 4.*mx2/sxj/sxj - 4.*my2/syj/syj;

    // Check if x is a W or a fermion.
    if (abs(idx) == 24 && useFullWkernel)
      ant += (4./3.)*(syj/(s - syj) + syj*(s - syj)/s/s)/sxj;
    else
      ant += 2.*syj/sxj/s;

    // Check if y is a W or a fermion.
    if (abs(idy) == 24 && useFullWkernel)
      ant += (4./3.)*(sxj/(s - sxj) + sxj*(s - sxj)/s/s)/syj;
    else
      ant += 2.*sxj/syj/s;
  }

  // FF (dipole).
  if (ele->isDip) {
    double s = sxj + syj + sxy;
    ant += 4.*sxy/sxj/(sxj+syj) - 4.*mx2/sxj/sxj + 2.*syj/sxj/s;
  }

  // IF.
  if (ele->isIF) {
    double s = sxj + sxy - syj;
    // Eikonal + initial state fermion.
    // The initial state is never a W and has no mass.
    ant += 4.*sxy/sxj/syj - 4.*my2/syj/syj + 2.*syj/sxj/s;

    if (abs(idy) == 24 && useFullWkernel)
      ant += (8./3.)*( sxj/(sxy + syj) + sxj/(s + syj)
        - pow2(sxj)/pow2(s + syj) )/syj;
    else
      ant += 2.*sxj/s/syj;
  }

  // II.
  if (ele->isII) {
    double s = sxy - sxj - syj;
    // Eikonal + fermion.
    ant = 4.*sxy/sxj/syj + 2.*(sxj/syj + syj/sxj)/s;
  }

  // RF.
  if (ele->isRF) {
    double s = sxj + sxy - syj;
    // Eikonal.
    ant = 4.*sxy/sxj/syj - 4.*mx2/sxj/sxj - 4.*my2/syj/syj;

    // Check if x is a W or a fermion
    if (abs(idx) == 24 && useFullWkernel)
      ant += (8./3.)*( syj/(s+syj) + syj/s + pow2(syj)/pow2(s) )/sxj;
    else
      ant += 2.*syj/sxj/s;

    // Check if y is a W or a fermion.
    if (abs(idy) == 24 && useFullWkernel)
      ant += (8./3.)*( sxj/(sxy + syj) + sxj/(s + syj)
          - pow2(sxj)/pow2(s + syj) )/syj;
    else
      ant += 2.*sxj/syj/s;
  }
  return ant;

}

//--------------------------------------------------------------------------

// Ratio between PDFs.

double QEDemitSystem::pdfRatio(bool isA, double eOld, double eNew, int id,
  double Qt2) {
  double xOld = eOld/(sqrt(shh)/2.0);
  double xNew = eNew/(sqrt(shh)/2.0);
  double newPDF, oldPDF;
  if (isA) {
    newPDF = beamAPtr->xfISR(iSys, id, xNew, Qt2)/xNew;
    oldPDF = beamAPtr->xfISR(iSys, id, xOld, Qt2)/xOld;
    if (abs(newPDF) < TINYPDF) newPDF = TINYPDF;
    if (abs(oldPDF) < TINYPDF) oldPDF = TINYPDF;
  } else {
    newPDF = beamBPtr->xfISR(iSys, id, xNew, Qt2)/xNew;
    oldPDF = beamBPtr->xfISR(iSys, id, xOld, Qt2)/xOld;
    if (abs(newPDF) < TINYPDF) newPDF = TINYPDF;
    if (abs(oldPDF) < TINYPDF) oldPDF = TINYPDF;
  }
  return newPDF/oldPDF;
}

//==========================================================================

// Class for a QED splitting system.

//--------------------------------------------------------------------------

// Initialize.

void QEDsplitSystem::init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn,
  int verboseIn) {
  if (!isInitPtr) printOut(__METHOD_NAME__, "initPtr not called");
  verbose = verboseIn;
  q2Max   = pow2(settingsPtr->parm("Vincia:mMaxGamma"));
  nLepton = settingsPtr->mode("Vincia:nGammaToLepton");
  nQuark  = settingsPtr->mode("Vincia:nGammaToQuark");
  kMapTypeFinal = settingsPtr->mode("Vincia:kineMapEWFinal");
  beamAPtr = beamAPtrIn;
  beamBPtr = beamBPtrIn;
  isInit = true;
}

//--------------------------------------------------------------------------

// Prepare list of final-state photons - with recoilers - for splittings.

void QEDsplitSystem::prepare(int iSysIn, Event &event, double q2CutIn,
  bool isBelowHadIn, vector<double> evolutionWindowsIn, AlphaEM alIn) {

  if (!isInit) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": Not initialised.");
    return;
  }
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Input.
  iSys = iSysIn;
  q2Cut = q2CutIn;
  isBelowHad = isBelowHadIn;
  evolutionWindows = evolutionWindowsIn;
  al = alIn;

  // Set up weights for splitting flavours.
  ids.clear();
  idWeights.clear();
  totIdWeight = 0;

  // Splittings for gamma->lepton+lepton-.
  for (int i = 0; i < nLepton; i++) {
    ids.push_back(11 + 2*i);
    idWeights.push_back(1);
  }
  // Only include gamma->qqbar if above hadronisation scale.
  if (!isBelowHad) {
    for (int i = 1; i <= nQuark; i++) {
      ids.push_back(i);
      idWeights.push_back((i%2==0 ? 4./3. : 1./3.));
    }
  }
  // Total weight.
  for (int i=0; i<(int)ids.size(); i++) totIdWeight += idWeights[i];

  // Build internal system.
  buildSystem(event);

  // Done.
  if (verbose >= DEBUG) {
    print();
    printOut(__METHOD_NAME__, "end", dashLen);
  }

}

//--------------------------------------------------------------------------

// Build the splitting system.

void QEDsplitSystem::buildSystem(Event &event) {

  // Verbose output.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Get rid of saved trial and clear all antennae.
  hasTrial = false;
  eleVec.clear();

  // Build lists of particles.
  vector<int> photList, chSpecList, uchSpecList;
  int sysSize = partonSystemsPtr->sizeAll(iSys);
  photList.reserve(sysSize);
  chSpecList.reserve(sysSize);
  uchSpecList.reserve(sysSize);
  for (int i = 0; i < sysSize; i++) {
    int iEv = partonSystemsPtr->getAll(iSys, i);
    if (iEv > 0) {
      // Only involve final state particles.
      if (event[iEv].isFinal()) {
        // Find photons.
        if (event[iEv].id()==22)    photList.push_back(iEv);
        // Find recoilers.
        if (event[iEv].isCharged()) chSpecList.push_back(iEv);
        else                        uchSpecList.push_back(iEv);
      }
    }
  }

  // If no charged and no uncharged spectators, return.
  if (chSpecList.empty() && uchSpecList.empty()) return;

  // Loop over photons.
  for (int i = 0; i < (int)photList.size(); i++) {
    int iPhot = photList[i];
    // If no charged spectators, use uncharged.
    if (chSpecList.empty()) {
      // Check if there is another spectator than the current photon.
      bool otherSpecAvail = false;
      for (int j = 0; j < (int)uchSpecList.size(); j++)
        if (uchSpecList[j] != iPhot) {otherSpecAvail = true; break;}
      // Continue to next photon if no spectator is available.
      if (!otherSpecAvail) continue;

      // Select one at random that's not the photon itself.
      int iSpec;
      while (true) {
        iSpec = uchSpecList[rndmPtr->flat()*uchSpecList.size()];
        if (iSpec != iPhot) break;
      }
      eleVec.push_back(QEDsplitElemental(event, iPhot, iSpec));
      eleVec.back().ariWeight = 1.;

    // Else use charged spectators.
    } else {
      double ariNorm = 0;
      vector<QEDsplitElemental> tempEleVec;
      for (int j = 0; j < (int)chSpecList.size(); j++) {
        int iSpec = chSpecList[j];
        tempEleVec.push_back(QEDsplitElemental(event, iPhot, iSpec));
        ariNorm += 1./tempEleVec.back().m2Ant;
      }
      // Set up Ariadne factors.
      for (int j = 0; j < (int)tempEleVec.size(); j++)
        tempEleVec[j].ariWeight = 1./(tempEleVec[j].m2Ant*ariNorm);
      eleVec.insert(eleVec.end(), tempEleVec.begin(), tempEleVec.end());
    }
  }

  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__,"end (nComb(Gam+Rec) ="
      + num2str((int)eleVec.size())+")");
  }

}

//--------------------------------------------------------------------------

// Generate a scale for the system.

double QEDsplitSystem::q2Next(Event &event, double q2Start) {

  // Return saved trial if we have one.
  if (hasTrial) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__,"Returning saved trial.");
    return q2Trial;
  }

  // Check if there are any photons left.
  if (eleVec.size() == 0) {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__,"No photons, can't generate a splitting.");
    return 0.;
  }
  // Starting scale - account for cut on mGammaMax.
  q2Trial = min(q2Max, q2Start);

  // Check if qTrial is below the cutoff.
  if (q2Trial <= q2Cut) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,"Below cutoff.");
    return 0.;
  }

  // Find lower value from evolution window.
  int iEvol = evolutionWindows.size() - 1;
  while (q2Start <= evolutionWindows[iEvol]) iEvol--;
  double q2Low = evolutionWindows[iEvol];

  // Compute weights.
  vector<double> weightVec;
  double totWeight(0);
  for (int i = 0; i < (int)eleVec.size(); i++) {
    double Iz = q2Low > eleVec[i].m2Ant ? 0 : 1. - q2Low/eleVec[i].m2Ant;
    double w = totIdWeight*eleVec[i].ariWeight*Iz*eleVec[i].getKallen();
    weightVec.push_back(w);
    totWeight += w;
  }

  // If no antennae are active, don't generate new scale.
  if (totWeight < NANO) q2Trial = 0.;

  // Generate scale and do alpha veto.
  else {
    while (q2Trial > q2Low) {
      double alphaMax = al.alphaEM(q2Trial);
      q2Trial *= pow(rndmPtr->flat(), M_PI/totWeight/alphaMax);
      double alphaNew = al.alphaEM(q2Trial);
      if (alphaNew <= 0.) return 0.;
      if (rndmPtr->flat() < alphaNew/alphaMax) break;
    }
  }

  // Check if evolution window was crossed.
  if (q2Trial <= q2Low) {
    if (iEvol == 0) {
      if (verbose >= DEBUG) printOut(__METHOD_NAME__,
        "Dropped below QED cutoff.");
      return 0.;
    }
    else if (verbose >= DEBUG) printOut(__METHOD_NAME__,
      "Trial was below window lower bound. Try again. ");
    return q2Next(event, q2Low);
  }

  // Select antenna.
  double ranWeight = rndmPtr->flat() * totWeight;
  for (int i=0; i<(int)weightVec.size(); ++i) {
    if ( (ranWeight -= weightVec[i]) < 0 ) {
      eleTrial = &eleVec[i];
      break;
    }
  }

  // Select splitting ID.
  double ranFlav = rndmPtr->flat() * totIdWeight;
  for (int iFlav=0; iFlav<(int)idWeights.size(); ++iFlav) {
    if ( (ranFlav -= idWeights[iFlav]) < 0 ) {
      idTrial = ids[iFlav];
      break;
    }
  }

  // Safety check.
  if (ranFlav >= 0 || ranWeight >= 0) {
    hasTrial = false;
    q2Trial  = 0.;
    return 0.;
  }

  // Generate value of zeta and phi.
  zTrial = (1. - q2Low/eleTrial->m2Ant)*rndmPtr->flat();
  phiTrial = rndmPtr->flat()*2*M_PI;

  // Done.
  hasTrial = true;
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"Done");
  return q2Trial;

}

//--------------------------------------------------------------------------

// Generate kinematics and check veto.

bool QEDsplitSystem::acceptTrial(Event &event) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  // Mark trial as used.
  hasTrial = false;

  // Set up some shorthands.
  int iPhot = eleTrial->iPhot;
  int iSpec = eleTrial->iSpec;
  double m2Ant = eleTrial->m2Ant;

  // New momenta.
  vector<Vec4> pOld;
  pNew.clear();

  // Safety check.
  if (iPhot > event.size() || iSpec > event.size()) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": inconsistent parent(s).");
    return false;
  }

  pOld.push_back(event[iPhot].p());
  pOld.push_back(event[iSpec].p());

  // ij is the new pair, k is the spectator.
  double mFerm = particleDataPtr->m0(idTrial);
  double mSpec = sqrt(eleTrial->m2Spec);
  double sIK = m2Ant - 2*pow2(mFerm) - pow2(mSpec);
  double sij = q2Trial - 2*pow2(mFerm);
  double sjk = zTrial*m2Ant;
  double sik = m2Ant - sij - sjk - 2*pow2(mFerm) - pow2(mSpec);

  // Check phase space.
  if (sik < 0) return false;
  if (sij*sjk*sik - pow2(sij)*pow2(mSpec)
    - (pow2(sjk) + pow2(sik))*pow2(mFerm) < 0) return false;

  // Make sure any new qqbar pair has at least the invariant mass
  // of the lightest meson.
  // sijMin is 0 if these are not quarks.
  double sijMin = vinComPtr->mHadMin(idTrial, -idTrial);
  if (sij < sijMin) return false;

  // Kernel veto.
  double pVeto = ( (pow2(sik) + pow2(sjk))/m2Ant + 2.*pow2(mFerm)/q2Trial)/2.;
  if (rndmPtr->flat() > pVeto) {
    return false;
  }

  vector<double> invariants;
  invariants.push_back(sIK);
  invariants.push_back(sij);
  invariants.push_back(sjk);
  vector<double> masses;
  masses.push_back(mFerm);
  masses.push_back(mFerm);
  masses.push_back(mSpec);

  // Kinematics.
  if (!vinComPtr->map2to3FF(pNew, pOld, kMapTypeFinal, invariants, phiTrial,
      masses)) return false;

  // Done.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"end", dashLen);
  return true;
}

//--------------------------------------------------------------------------

// Update event after splitting.

void QEDsplitSystem::updateEvent(Event &event) {

  //Clear information for replacing later in partonSystems
  iReplace.clear();
  shat = 0.;

  int iPhot = eleTrial->iPhot;
  int iSpec = eleTrial->iSpec;
  double mFerm = particleDataPtr->m0(idTrial);

  // Set up the new fermions. Stochastic colour tag.
  int colTag = idTrial < 10 ? 10*(event.nextColTag()/10 + 1) + 1
                         + rndmPtr->flat()*10 : 0;
  Particle partFermNew(idTrial, 51, iPhot, 0, 0, 0, colTag, 0, pNew[0], mFerm);
  Particle partAFermNew(-idTrial,51,iPhot, 0, 0, 0, 0, colTag, pNew[1], mFerm);
  Particle partSpecNew = event[iSpec];
  partSpecNew.mothers(iSpec, iSpec);
  partSpecNew.p(pNew[2]);
  partSpecNew.statusCode(52);

  // Change the event - add new particles.
  int iFermNew  = event.append(partFermNew);
  int iAFermNew = event.append(partAFermNew);
  int iSpecNew  = event.append(partSpecNew);

  // Adjust old ones.
  event[iPhot].statusNeg();
  event[iPhot].daughters(iFermNew, iAFermNew);
  event[iSpec].statusNeg();
  event[iSpec].daughters(iSpecNew, 0);

  // Save updates to be done on PartonSystems later.
  jNew = iAFermNew;
  iReplace[iPhot] = iFermNew;
  iReplace[iSpec] = iSpecNew;

  event.restorePtrs();

}

//--------------------------------------------------------------------------

// Print the system.

void QEDsplitSystem::print() {
  if (eleVec.size() == 0) {
    cout<< "  --------  No QED Splitters in System"<<endl;
    return;
  }
  cout << "  --------  QEDsplitSystem  ----------------"
       << "----------------------------------------------" << endl;
  for (int i = 0; i < (int)eleVec.size(); i++)
    cout << "    (" << eleVec[i].iPhot << " " << eleVec[i].iSpec << ") "
         << "s = " << eleVec[i].m2Ant << " ariFac = " << eleVec[i].ariWeight
         << endl;
  cout << "  --------------------------------------------------------------"
       << "----------------------------------------------" << endl;
}

//==========================================================================

// Class for a QED conversion system.

//--------------------------------------------------------------------------

// Initialize the system.

void QEDconvSystem::init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn,
  int verboseIn) {

  // Verbosity setting.
  if (!isInitPtr) printOut(__METHOD_NAME__, "initPtr not called");
  verbose = verboseIn;

  // Settings, number of incoming flavours to allow conversions to.
  // Could be extended to allow top quarks in future; for now up to b.
  nQuark = 5;
  if (!settingsPtr->flag("Vincia:convertGammaToQuark")) nQuark = 0;

  // Set trial pdf ratios.
  Rhat[1]  = 77;  Rhat[-1] = 63;
  Rhat[2]  = 140; Rhat[-2] = 65;
  Rhat[3]  = 30;  Rhat[-3] = 30;
  Rhat[4]  = 22;  Rhat[-4] = 30;
  Rhat[5]  = 15;  Rhat[-5] = 16;

  // Constants.
  TINYPDF = 1.0e-10;

  // Beam pointers.
  beamAPtr = beamAPtrIn;
  beamBPtr = beamBPtrIn;
  isInit = true;

}

//--------------------------------------------------------------------------

// Prepare for backwards-evolution of photons.

void QEDconvSystem::prepare(int iSysIn, Event &event, double q2CutIn,
  bool isBelowHadIn, vector<double> evolutionWindowsIn, AlphaEM alIn) {

  if (!isInit) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": Not initialised.");
    return;
  }
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Input.
  iSys = iSysIn;
  shh = infoPtr->s();
  isBelowHad = isBelowHadIn;
  q2Cut = q2CutIn;
  evolutionWindows = evolutionWindowsIn;
  al = alIn;

  // Set up weights for conversion flavours.
  ids.clear();
  idWeights.clear();
  totIdWeight = 0;
  maxIdWeight = 0;

  // If switched off, do nothing.
  if (nQuark == 0) return;

  // Only do conversions to quarks if above hadronisation scale.
  if (!isBelowHad)
    for (int i = 1; i <= nQuark; i++) {
      ids.push_back(i);
      ids.push_back(-i);
      idWeights.push_back((i%2==0 ? 4./9. : 1./9.)*Rhat[i]);
      idWeights.push_back((i%2==0 ? 4./9. : 1./9.)*Rhat[-i]);
    }
  // Total weights.
  for (int i = 0; i < (int)idWeights.size(); i++) {
    totIdWeight += idWeights[i];
    if (idWeights[i] > maxIdWeight) maxIdWeight = idWeights[i];
  }

  // Build system.
  buildSystem(event);

  // Done.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);

}

//--------------------------------------------------------------------------

// Build the system.

void QEDconvSystem::buildSystem(Event &event) {

  // Get rid of saved trial.
  hasTrial = false;

  // Get initial states.
  iA = partonSystemsPtr->getInA(iSys);
  iB = partonSystemsPtr->getInB(iSys);
  isAPhot = event[iA].id() == 22;
  isBPhot = event[iB].id() == 22;
  s = (event[iA].p() + event[iB].p()).m2Calc();

  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, " convA =" + bool2str(isAPhot)
      +", convB =" + bool2str(isBPhot));
  }

}

//--------------------------------------------------------------------------

// Generate a trial scale.

double QEDconvSystem::q2Next(Event &event, double q2Start) {

  // Return saved trial if we have one.
  if (hasTrial) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,
      "Returning saved trial.");
    return q2Trial;
  }
  // Check if there are any initial-state photons.
  if (!isAPhot && !isBPhot) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,
      "No initial-state photons, so can't generate a conversion.");
    return 0.;
  }
  double totWeight = 1.;

  // Select a photon.
  if       (isAPhot && !isBPhot)  {iPhotTrial = iA; iSpecTrial = iB;}
  else  if (isBPhot && !isAPhot)  {iPhotTrial = iB; iSpecTrial = iA;}
  else {
    if (rndmPtr->flat() < 0.5)    {iPhotTrial = iA; iSpecTrial = iB;}
    else                          {iPhotTrial = iB; iSpecTrial = iA;}
    // Two photon antennae -> twice the weight.
    totWeight *= 2.;
  }

  // Starting scale.
  q2Trial = q2Start;

  // Check if qTrial is below the cutoff.
  if (q2Trial <= q2Cut) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,"Below cutoff.");
    return 0.;
  }

  // Find lower value from evolution window.
  int iEvol = evolutionWindows.size() - 1;
  while(q2Start <= evolutionWindows[iEvol]) {iEvol--;}
  double q2Low = evolutionWindows[iEvol];

  // Iz integral.
  double zPlus = shh/s;
  double zMin = 1 + q2Low/s;
  if (zPlus < zMin) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,"Phase space closed");
    return 0.;
  }
  double Iz = log(zPlus/zMin);
  totWeight *= totIdWeight*Iz;

  // If no antennae are active, don't generate new scale.
  if (totWeight < NANO) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,"Below cutoff.");
    return 0.;
  }

  // Generate scale and do alpha veto.
  else
    while(true) {
      double alphaMax = al.alphaEM(q2Trial);
      q2Trial *= pow(rndmPtr->flat(), M_PI/totWeight/alphaMax);
      double alphaNew = al.alphaEM(q2Trial);
      if (rndmPtr->flat() < alphaNew/alphaMax) break;
    }

  // Check if evolution window was crossed.
  if (q2Trial < q2Low) {
    if (iEvol==0) {
      if (verbose >= DEBUG) printOut(__METHOD_NAME__,
        "Dropped below QED cutoff.");
      return 0.;
    }
    else if (verbose >= DEBUG) printOut(__METHOD_NAME__,
      "Trial was below window lower bound. Try again. ");
    return q2Next(event, q2Low);
  }

  // Select conversion ID.
  while( true) {
    int idIndex = rndmPtr->flat()*ids.size();
    idTrial = ids[idIndex];
    if (rndmPtr->flat() < idWeights[idIndex]/maxIdWeight) break;
  }
  zTrial = zMin*pow(zPlus/zMin, rndmPtr->flat());
  phiTrial = rndmPtr->flat()*2*M_PI;
  hasTrial = true;
  return q2Trial;

}

//--------------------------------------------------------------------------

// Check the veto.

bool QEDconvSystem::acceptTrial(Event &event) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  // Mark trial as used.
  hasTrial = false;

  // Conversion mass.
  double mf2 = pow2(particleDataPtr->m0(idTrial));

  // Spectator ID.
  int idSpec = event[iSpecTrial].id();

  // Old momenta.
  vector<Vec4> pOld;
  pOld.push_back(event[iPhotTrial].p());
  pOld.push_back(event[iSpecTrial].p());

  // Clear new momenta.
  pNew.clear();

  // Note that we treat the initial state as massless and the final
  // state as massive. q2 is defined as saj - 2*mf2, but otherwise we
  // adhere to the proper kinematics.
  double saj = q2Trial + 2*mf2;
  double sbj = (zTrial-1)*s - saj + mf2;
  double sab = s + saj + sbj - mf2;

  // Check kinematic boundaries.
  if (sbj < 0) return false;
  if (sab*saj*sbj - mf2*sab*sab < 0) return false;

  // Check if photon is in beam a or b.
  bool isPhotA = (iPhotTrial == iA) ? true : false;

  // Global recoil momenta.
  pRec.clear();
  iRec.clear();
  int sysSize = partonSystemsPtr->sizeAll(iSys);
  for (int i=0; i<sysSize; i++) {
    int iEv = partonSystemsPtr->getAll(iSys, i);
    if (iEv < 0 || !event[iEv].isFinal()) continue;
    pRec.push_back(event[iEv].p());
    iRec.push_back(iEv);
  }

  // Kinematics.
  if (!vinComPtr->map2to3II(pNew, pRec, pOld, s, saj, sbj, sab,
      phiTrial, mf2)) return false;

  // Check if new energies don't exceed hadronic maxima.
  double eaUsed = 0, ebUsed = 0;
  int nSys = partonSystemsPtr->sizeSys();
  for (int i=0; i<nSys; i++) {
    eaUsed += event[partonSystemsPtr->getInA(i)].e();
    ebUsed += event[partonSystemsPtr->getInB(i)].e();
  }
  if (isPhotA) {
    if ((eaUsed - pOld[0].e() + pNew[0].e()) > 0.98*sqrt(shh)/2.) return false;
    if ((ebUsed - pOld[1].e() + pNew[2].e()) > 0.98*sqrt(shh)/2.) return false;
  } else {
    if ((ebUsed - pOld[0].e() + pNew[0].e()) > 0.98*sqrt(shh)/2.) return false;
    if ((eaUsed - pOld[1].e() + pNew[2].e()) > 0.98*sqrt(shh)/2.) return false;
  }

  // Kernel veto probability.
  double pVeto = 0.5*(1. + pow2(sbj)/pow2(sab)
    - 2.*mf2*pow2(s)/pow2(sab)/(saj-2*mf2));

  // Compute pdf ratios.
  double Rpdf = 1.;
  double xPhotOld = pOld[0].e()/(sqrt(shh)/2.);
  double xPhotNew = pNew[0].e()/(sqrt(shh)/2.);
  double xSpecOld = pOld[1].e()/(sqrt(shh)/2.);
  double xSpecNew = pNew[2].e()/(sqrt(shh)/2.);

  if (isPhotA) {
    // Photon pdf.
    double newPDFPhot = beamAPtr->xfISR(iSys, idTrial, xPhotNew, q2Trial);
    double oldPDFPhot = beamAPtr->xfISR(iSys, 22,      xPhotOld, q2Trial);
    if (abs(newPDFPhot) < TINYPDF) newPDFPhot = TINYPDF;
    if (abs(oldPDFPhot) < TINYPDF) oldPDFPhot = TINYPDF;
    Rpdf *= newPDFPhot/oldPDFPhot;

    // Spectator pdf.
    double newPDFSpec = beamBPtr->xfISR(iSys, idSpec, xSpecNew, q2Trial);
    double oldPDFSpec = beamBPtr->xfISR(iSys, idSpec, xSpecOld, q2Trial);
    if (abs(newPDFSpec) < TINYPDF) newPDFSpec = TINYPDF;
    if (abs(oldPDFSpec) < TINYPDF) oldPDFSpec = TINYPDF;
    Rpdf *= newPDFSpec/oldPDFSpec;
  } else {
    // Photon pdf.
    double newPDFPhot = beamBPtr->xfISR(iSys, idTrial, xPhotNew, q2Trial);
    double oldPDFPhot = beamBPtr->xfISR(iSys, 22,      xPhotOld, q2Trial);
    if (abs(newPDFPhot) < TINYPDF) newPDFPhot = TINYPDF;
    if (abs(oldPDFPhot) < TINYPDF) oldPDFPhot = TINYPDF;
    Rpdf *= newPDFPhot/oldPDFPhot;

    // Spectator pdf.
    double newPDFSpec = beamAPtr->xfISR(iSys, idSpec, xSpecNew, q2Trial);
    double oldPDFSpec = beamAPtr->xfISR(iSys, idSpec, xSpecOld, q2Trial);
    if (abs(newPDFSpec) < TINYPDF) newPDFSpec = TINYPDF;
    if (abs(oldPDFSpec) < TINYPDF) oldPDFSpec = TINYPDF;
    Rpdf *= newPDFSpec/oldPDFSpec;
  }

  if (Rpdf > Rhat[idTrial]) {
    stringstream ss;
    ss << "at q = "<<sqrt(q2Trial)<<" GeV,  id = " << idTrial
       << ",  ratio = " << Rpdf/Rhat[idTrial];
    infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": incorrect PDF "
      +"overestimate",ss.str());
  }

  // Pdf ratio veto probability.
  pVeto *= (Rpdf/Rhat[idTrial]);

  // Do veto.
  if (rndmPtr->flat() > pVeto) {
    return false;
  }

  // Done.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"end", dashLen);
  return true;
}

//--------------------------------------------------------------------------

// Update event after QED conversion.

void QEDconvSystem::updateEvent(Event &event) {

  //Clear information for replacing later in partonSystems
  iReplace.clear();
  shat = 0.;

  // Conversion mass
  double mf2 = pow2(particleDataPtr->m0(idTrial));

  // Set up new particles.
  Particle partSpecNew = event[iSpecTrial];
  partSpecNew.p(pNew[2]);
  partSpecNew.statusCode(41);

  // Stochastic colour tag.
  int colTag = 10*(event.nextColTag()/10 + 1) + 1 + rndmPtr->flat()*10;
  Particle partBeamNew  (idTrial, -41, 0, 0, 0, 0, idTrial > 0 ?
    colTag : 0, idTrial > 0 ? 0 : colTag, pNew[0]);
  Particle partFinalNew (idTrial,  43, 0, 0, 0, 0, idTrial > 0 ?
    colTag : 0, idTrial > 0 ? 0 : colTag, pNew[1], sqrt(mf2));
  int iBeamNew = event.append(partBeamNew);
  int iFinalNew = event.append(partFinalNew);
  int iSpecNew = event.append(partSpecNew);
  event[iPhotTrial].statusNeg();
  event[iSpecTrial].statusNeg();
  event[iBeamNew].mothers(event[iPhotTrial].mother1(),
    event[iPhotTrial].mother2());
  event[iFinalNew].mothers(iBeamNew, 0);
  event[iSpecNew].mothers(event[iSpecTrial].mother1(),
    event[iSpecTrial].mother2());
  event[iPhotTrial].mothers(iBeamNew, 0);
  event[iSpecTrial].mothers(iSpecNew, 0);
  event[iBeamNew].daughters(iFinalNew, iPhotTrial);
  event[iFinalNew].daughters(0,0);
  event[iSpecNew].daughters(iSpecTrial);

  // Change daughters of beams for hard process.
  if (iSys == 0) {
    bool foundPhot = false;
    bool foundSpec = false;
    for (int i = 0; i < (int)event.size(); i++) {
      if (!foundPhot)
        if (event[i].daughter1() == iPhotTrial) {
          event[i].daughters(iBeamNew, 0);
          foundPhot = true;
        }
      if (!foundSpec)
        if (event[i].daughter1() == iSpecTrial) {
          event[i].daughters(iSpecNew, 0);
          foundSpec = true;
        }
      if (foundPhot && foundSpec) break;
    }
  }

  // Update event for global recoil.
  for (int j=0; j<event.size(); j++) {
    if (event[j].isFinal()) {
      for (int k=0; k<(int)iRec.size(); k++) {
        if (iRec[k] == j) {
          // Copy the recoiler.
          int inew = event.copy(j, 44);
          // Change the momentum.
          event[inew].p(pRec[k]);
          // Save update to do on PartonSystems later.
          iReplace[iRec[k]] = inew;
        }
      }
    }
  }

  // Save updates to do on PartonSystems later.
  jNew = iFinalNew;
  iReplace[iPhotTrial] = iBeamNew;
  iReplace[iSpecTrial] = iSpecNew;
  shat = (event[iBeamNew].p() + event[iSpecNew].p()).m2Calc();

  // Update beams.
  BeamParticle& beam1 = *beamAPtr;
  BeamParticle& beam2 = *beamBPtr;
  // Check if photon is in beam a or b
  bool isPhotA = (iPhotTrial == iA) ? true : false;
  if (isPhotA) {
    beam1[iSys].update(iBeamNew, event[iBeamNew].id(),
      event[iBeamNew].e()/beam1.e());
    beam2[iSys].update(iSpecNew, event[iSpecNew].id(),
      event[iSpecNew].e()/beam2.e());
  } else {
    beam1[iSys].update(iSpecNew, event[iSpecNew].id(),
      event[iSpecNew].e()/beam1.e());
    beam2[iSys].update(iBeamNew, event[iBeamNew].id(),
      event[iBeamNew].e()/beam2.e());
  }

  // Done.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);

}

//--------------------------------------------------------------------------

// Print.

void QEDconvSystem::print() {
  cout << "  --------  QEDconvSystem  ----------------"
       << "----------------------------------------------" << endl;
  cout << "    s = " << s << endl;
}

//==========================================================================

// Class for performing QED showers.

//--------------------------------------------------------------------------

// Initialize the pointers.

void VinciaQED::initPtr(Info* infoPtrIn, VinciaCommon* vinComPtrIn) {
  infoPtr       = infoPtrIn;
  particleDataPtr  = infoPtr->particleDataPtr;
  partonSystemsPtr = infoPtr->partonSystemsPtr;
  rndmPtr          = infoPtr->rndmPtr;
  settingsPtr      = infoPtr->settingsPtr;
  vinComPtr        = vinComPtrIn;

  // Initialise the empty templates we use to make new QED shower systems.
  emptyQEDemitSystem.initPtr(infoPtr, particleDataPtr, partonSystemsPtr,
    rndmPtr, settingsPtr, vinComPtr);
  emptyQEDsplitSystem.initPtr(infoPtr, particleDataPtr, partonSystemsPtr,
    rndmPtr, settingsPtr, vinComPtr);
  emptyQEDconvSystem.initPtr(infoPtr, particleDataPtr, partonSystemsPtr,
    rndmPtr, settingsPtr, vinComPtr);

  // Done.
  isInitPtr = true;
}

//--------------------------------------------------------------------------

// Initialize settings for current run.

void VinciaQED::init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn) {

  // Verbose setting
  verbose = settingsPtr->mode("Vincia:verbose");

  // Initialize alphaEM
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

  // Get settings.
  doQED          = settingsPtr->mode("Vincia:EWmode") >= 1;
  doEmission     = doQED;
  nGammaToLepton = settingsPtr->mode("Vincia:nGammaToLepton");
  nGammaToQuark  = settingsPtr->mode("Vincia:nGammaToQuark") >= 1;
  doConvertGamma = settingsPtr->flag("Vincia:convertGammaToQuark");
  doConvertQuark = settingsPtr->flag("Vincia:convertQuarkToGamma");

  // QED cutoff for coloured particles/hadronisation scale.
  q2minColouredSav = pow2(settingsPtr->parm("Vincia:QminChgQ"));
  q2minSav         = pow2(settingsPtr->parm("Vincia:QminChgL"));

  // Set beam pointers.
  beamAPtr = beamAPtrIn;
  beamBPtr = beamBPtrIn;

  // Initialise the empty templates we use to make new QED shower systems.
  emptyQEDemitSystem.init(beamAPtrIn, beamBPtrIn, verbose);
  emptyQEDsplitSystem.init(beamAPtrIn, beamBPtrIn, verbose);
  emptyQEDconvSystem.init(beamAPtrIn, beamBPtrIn, verbose);

  // All done.
  isInitSav = true;

}

//--------------------------------------------------------------------------

// Prepare to shower a system.

bool VinciaQED::prepare(int iSysIn, Event &event, bool isBelowHad) {
  // Check if QED is switched on for this system.
  if (!doQED) return false;

  // Verbose output
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "begin", dashLen);
    stringstream ss;
    ss << "Preparing system " << iSysIn;
    printOut(__METHOD_NAME__,ss.str());
  }

  // Above or below hadronisation scale.
  double q2cut = (isBelowHad) ? q2minSav : q2minColouredSav;

  // If below hadronization scale or this is resonance system,
  // clear information about any other systems.
  if ( iSysIn == -1 || isBelowHad || partonSystemsPtr->hasInRes(iSysIn)) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,
      "clearing previous QED systems");
    clear();
  }

  // Initialize windows for the hard system
  // and the final after-beam-remnants system.
  if ( nBranchers()==0 ) {
    // The cutoff scale is the lowest window boundary,
    // then step up to q2Max successively by factor winStep
    double q2BiggestEver  = infoPtr->s();
    double q2Window       = q2cut;
    double winStep        = 100.0;
    evolutionWindows.clear();
    do {
      evolutionWindows.push_back(q2Window);
      q2Window *= winStep;
    } while(q2Window < q2BiggestEver);
  }

  // Special case: iSysIn = -1 implies below hadronisation scale.
  // Collect all final-state particles into one new system.  Note:
  // this system will have sum(charge) != 0 if the sum of the beam
  // charges is nonzero.
  if (iSysIn == -1) {
    iSysIn = partonSystemsPtr->addSys();
    // Loop over all particles in event rather than all parton systems
    // since beam remnant partons not part of any partonSystem.
    for (int i = 1; i < event.size(); ++i) {
      if (!event[i].isFinal()) continue;
      partonSystemsPtr->addOut(iSysIn, i);
    }
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__,"Created new parton system for post-remnant "
        "QED showering:");
      partonSystemsPtr->list();
    }
  }

  // Safety checks for systems that should have been deleted.
  bool forceClear(false);
  for (auto it=emitSystems.begin(); it!=emitSystems.end(); ++it) {
    if (it->first >= partonSystemsPtr->sizeSys()) forceClear = true;
  }
  for (auto it=splitSystems.begin(); it!=splitSystems.end(); ++it) {
    if (it->first >= partonSystemsPtr->sizeSys()) forceClear = true;
  }
  for (auto it=convSystems.begin(); it!=convSystems.end(); ++it) {
    if (it->first >= partonSystemsPtr->sizeSys()) forceClear = true;
  }
  if (forceClear) {
    clear();
    infoPtr->errorMsg("Warning in "+__METHOD_NAME__
    +": cleared inconsistent list of QED systems.");
  }

  // Add and prepare new system for initial- and final-state photon emissions.
  emitSystems[iSysIn] = emptyQEDemitSystem;
  emitSystems[iSysIn].prepare(iSysIn, event, q2cut, isBelowHad,
    evolutionWindows,al);

  // Add and prepare new system for final-state photon splittings.
  splitSystems[iSysIn] = emptyQEDsplitSystem;
  splitSystems[iSysIn].prepare(iSysIn, event, q2cut, isBelowHad,
    evolutionWindows,al);

  // Add and prepare new system for initial-state photon conversions.
  convSystems[iSysIn] = emptyQEDconvSystem;
  convSystems[iSysIn].prepare(iSysIn, event, q2cut, isBelowHad,
    evolutionWindows,al);

  // Done.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"end", dashLen);
  return true;

}

//--------------------------------------------------------------------------

// Update QED shower system(s) each time something has changed in event.

void VinciaQED::update(Event &event, int iSys) {

  // Find index of the system.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "begin (iSys"+num2str(iSys,2)+")", dashLen);
  }

  if (emitSystems.find(iSys) != emitSystems.end())
    emitSystems[iSys].buildSystem(event);
  if (splitSystems.find(iSys) != splitSystems.end())
    splitSystems[iSys].buildSystem(event);
  if (convSystems.find(iSys) != convSystems.end())
    convSystems[iSys].buildSystem(event);

  // Done.
  if (verbose >= DEBUG) {
    event.list();
    printOut(__METHOD_NAME__, "end", dashLen);
  }
}

//--------------------------------------------------------------------------

// Generate a trial scale.

double VinciaQED::q2Next(Event &event, double q2Start, double) {

  // Get a trial from every system.
  qedTrialSysPtr = nullptr;
  q2Trial = 0.;

  // Sanity check.
  if (!doQED) return 0.0;

  // Verbose output.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "begin", dashLen);
    stringstream ss;
    ss << "q2Start = " << q2Start
       << " doEmit = " << bool2str(doEmission)
       << " nSplitGamToLep = " << num2str(nGammaToLepton)
       << " nSplitGamToQuark = " << num2str(nGammaToQuark)
       << " doConv = " << bool2str(doConvertGamma);
    printOut(__METHOD_NAME__,ss.str());
  }

  // Emissions.
  if (doEmission && emitSystems.size() >= 1) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,
      "Generating QED emissions.");
    q2NextSystem(emitSystems,event,q2Start);
  }

  // Splittings (no point trying if starting scale is below electron mass).
  if (q2Start < pow2(2*particleDataPtr->m0(11))) splitSystems.clear();
  else if (nGammaToLepton + nGammaToQuark > 0 && splitSystems.size() >= 1) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,
      "Generating QED splittings.");
    q2NextSystem(splitSystems,event,q2Start);
  }

  // Conversions.
  if (doConvertGamma && convSystems.size() >= 1) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,
      "Generating QED conversions.");
    q2NextSystem(convSystems,event,q2Start);
  }

  // Done.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "end", dashLen);
  return q2Trial;

}

//--------------------------------------------------------------------------

// Check the veto.

bool VinciaQED::acceptTrial(Event &event) {

  // Verbose output.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);
  bool accept = false;

  // Delegate.
  if (qedTrialSysPtr) accept = qedTrialSysPtr->acceptTrial(event);

  // Done.
  if (verbose >= DEBUG) {
    string result = (accept) ? "accept" : "reject";
    printOut(__METHOD_NAME__, "end ("+result+")", dashLen);
  }
  return accept;

}

//--------------------------------------------------------------------------

// Update Event after QED shower branching

void VinciaQED::updateEvent(Event &event) {

  // Verbose output.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"begin", dashLen);

  // Delegate.
  if (qedTrialSysPtr != nullptr) qedTrialSysPtr->updateEvent(event);

  // Done.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"end", dashLen);

  return;

}

//--------------------------------------------------------------------------

// Update PartonSystems after QED shower branching.

void VinciaQED::updatePartonSystems(Event&) {

  // Verbose output.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"begin", dashLen);

  // Delegate.
  if (qedTrialSysPtr!=nullptr) qedTrialSysPtr->updatePartonSystems();

  // Done.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"end", dashLen);
  return;

}

//--------------------------------------------------------------------------

// Find Q2 next for a given system.

template <class T> void VinciaQED::q2NextSystem(
  map<int, T>& systemList, Event& event, double q2Start) {

  // Loop over all QED systems.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Looping over " << systemList.size()
       << " QED systems (q2start=" << q2Start << ")";
    printOut(__METHOD_NAME__, ss.str());
  }
  for(auto it = systemList.begin(); it != systemList.end(); ++it) {
    QEDsystem* qedSysPtrNow = &(it->second);
    double q2TrialNow = qedSysPtrNow->q2Next(event, q2Start);
    // Highest trial so far?
    if (q2TrialNow > q2Trial) {
      // Save
      q2Trial = q2TrialNow;
      iSysTrial = it->first;
      qedTrialSysPtr = qedSysPtrNow;
    }
  }

}

//==========================================================================

} // end namespace Pythia8
