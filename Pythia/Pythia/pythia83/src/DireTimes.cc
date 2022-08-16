// DireTimes.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the DireTimesEnd
// and DireTimes classes.

#include "Pythia8/DireTimes.h"
#include "Pythia8/DireSpace.h"
#include "Pythia8/DireHistory.h"

namespace Pythia8 {

//==========================================================================

// The DireTimesEnd class.

bool operator==(const DireTimesEnd& dip1, const DireTimesEnd& dip2) {
  return dip1.iRadiator == dip2.iRadiator
      && dip1.iRecoiler == dip2.iRecoiler
      && dip1.colType == dip2.colType
      && dip1.isrType == dip2.isrType
      && dip1.allowedEmissions == dip2.allowedEmissions;
}

//==========================================================================

// The DireTimes class.

//--------------------------------------------------------------------------

// Constants: could be changed here if desired, but normally should not.
// These are of technical nature, as described for each.

// Minimal allowed c and b quark masses, for flavour thresholds.
const double DireTimes::MCMIN        = 1.2;
const double DireTimes::MBMIN        = 4.0;

// For small x approximate 1 - sqrt(1 - x) by x/2.
const double DireTimes::SIMPLIFYROOT = 1e-8;

// Do not allow x too close to 0 or 1 in matrix element expressions.
// Warning: cuts into phase space for E_CM > 2 * pTmin * sqrt(1/XMARGIN),
// i.e. will become problem roughly for E_CM > 10^6 GeV.
const double DireTimes::XMARGIN      = 1e-12;
const double DireTimes::XMARGINCOMB  = 1e-4;

// Lower limit on PDF value in order to avoid division by zero.
const double DireTimes::TINYPDF      = 1e-5;

// Leftover companion can give PDF > 0 at small Q2 where other PDF's = 0,
// and then one can end in infinite loop of impossible kinematics.
const int    DireTimes::MAXLOOPTINYPDF = 10;

// Big starting value in search for smallest invariant-mass pair.
const double DireTimes::LARGEM2      = 1e20;

// In g -> q qbar or gamma -> f fbar require m2_pair > this * m2_q/f.
const double DireTimes::THRESHM2     = 4.004;

// Never pick pT so low that alphaS is evaluated too close to Lambda_3.
const double DireTimes::LAMBDA3MARGIN = 1.1;

// Value (in GeV) below which particles are considered massless
const double DireTimes::TINYMASS = 1e-3;

// Lower limit on estimated evolution rate, below which stop.
const double DireTimes::TINYOVERESTIMATE  = 1e-15;

// pT2 above which PDF is used in overestimates for FI splittings.
const double DireTimes::PT2MIN_PDF_IN_OVERESTIMATE = 5.;

// pT2 below which PDF overestimates for FI with gluon recoiler is included.
const double DireTimes::PT2_INCREASE_OVERESTIMATE = 2.;

const double DireTimes::KERNEL_HEADROOM = 10.;

// Maximal energy fraction of lepton in lepton -> lepton photon branching.
// Needed because massive phase space does not cut off zCS->1.
const double DireTimes::LEPTONZMAX     = 1. - 1e-4;

//--------------------------------------------------------------------------

// Initialize alphaStrong, alphaEM and related pTmin parameters.

void DireTimes::init( BeamParticle* beamAPtrIn,
  BeamParticle* beamBPtrIn) {

  dryrun = false;

  // Colour factors.
  CA = settingsPtr->parm("DireColorQCD:CA") > 0.0
     ? settingsPtr->parm("DireColorQCD:CA") : 3.0;
  CF = settingsPtr->parm("DireColorQCD:CF") > 0.0
     ? settingsPtr->parm("DireColorQCD:CF") : 4./3.;
  TR = settingsPtr->parm("DireColorQCD:TR") > 0.
     ? settingsPtr->parm("DireColorQCD:TR") : 0.5;
  NC = settingsPtr->parm("DireColorQCD:NC") > 0.
     ? settingsPtr->parm("DireColorQCD:NC") : 3.0;

  // Alternatively only initialize resonance decays.
  processLevel.initInfoPtr(*infoPtr);
  processLevel.initDecays(nullptr);

  // Store input pointers for future use.
  beamAPtr           = beamAPtrIn;
  beamBPtr           = beamBPtrIn;

  // Main flags.
  doQCDshower        = settingsPtr->flag("TimeShower:QCDshower");
  doQEDshowerByQ     = settingsPtr->flag("TimeShower:QEDshowerByQ");
  doQEDshowerByL     = settingsPtr->flag("TimeShower:QEDshowerByL");
  doDecaysAsShower   = settingsPtr->flag("DireTimes:DecaysAsShower");

  doMEcorrections    = settingsPtr->flag("Dire:doMECs")
                    || settingsPtr->flag("Dire:doMOPS");
  doMEafterFirst     = settingsPtr->flag("TimeShower:MEafterFirst");
  doPhiPolAsym       = settingsPtr->flag("TimeShower:phiPolAsym");
  doInterleave       = settingsPtr->flag("TimeShower:interleave");
  allowBeamRecoil    = settingsPtr->flag("TimeShower:allowBeamRecoil");
  dampenBeamRecoil   = settingsPtr->flag("TimeShower:dampenBeamRecoil");
  recoilToColoured   = settingsPtr->flag("TimeShower:recoilToColoured");

  // Matching in pT of hard interaction or MPI to shower evolution.
  pTmaxMatch         = settingsPtr->mode("TimeShower:pTmaxMatch");
  pTdampMatch        = settingsPtr->mode("TimeShower:pTdampMatch");
  pTmaxFudge         = settingsPtr->parm("TimeShower:pTmaxFudge");
  pTmaxFudgeMPI      = settingsPtr->parm("TimeShower:pTmaxFudgeMPI");
  pTdampFudge        = settingsPtr->parm("TimeShower:pTdampFudge");
  pT2minVariations   = pow2(max(0.,settingsPtr->parm("Variations:pTmin")));
  pT2minEnhance      = pow2(max(0.,settingsPtr->parm("Enhance:pTmin")));
  pT2minMECs         = pow2(max(0.,settingsPtr->parm("Dire:pTminMECs")));
  Q2minMECs          = pow2(max(0.,settingsPtr->parm("Dire:QminMECs")));
  nFinalMaxMECs      = settingsPtr->mode("Dire:nFinalMaxMECs");
  suppressLargeMECs  = settingsPtr->flag("Dire:suppressLargeMECs");
  pT2recombine       =
    pow2(max(0.,settingsPtr->parm("DireTimes:pTrecombine")));

  // Charm and bottom mass thresholds.
  mc                 = max( MCMIN, particleDataPtr->m0(4));
  mb                 = max( MBMIN, particleDataPtr->m0(5));
  m2c                = mc * mc;
  m2b                = mb * mb;

  // Parameters of scale choices (inherited from Pythia).
  renormMultFac     = settingsPtr->parm("TimeShower:renormMultFac");
  factorMultFac     = settingsPtr->parm("TimeShower:factorMultFac");
  useFixedFacScale  = settingsPtr->flag("TimeShower:useFixedFacScale");
  fixedFacScale2    = pow2(settingsPtr->parm("TimeShower:fixedFacScale"));

  // Parameters of alphaStrong generation.
  alphaSvalue        = settingsPtr->parm("TimeShower:alphaSvalue");
  alphaSorder        = settingsPtr->mode("TimeShower:alphaSorder");
  alphaSnfmax        = settingsPtr->mode("StandardModel:alphaSnfmax");
  alphaSuseCMW       = settingsPtr->flag("TimeShower:alphaSuseCMW");
  alphaS2pi          = 0.5 * alphaSvalue / M_PI;
  asScheme           = settingsPtr->mode("DireTimes:alphasScheme");

  // Set flavour thresholds by default Pythia masses, unless zero.
  double mcpy = particleDataPtr->m0(4);
  double mbpy = particleDataPtr->m0(5);
  double mtpy = particleDataPtr->m0(6);
  if (mcpy > 0.0 && mbpy > 0.0 && mtpy > 0.0)
    alphaS.setThresholds(mcpy, mbpy, mtpy);

  // Initialize alphaStrong generation.
  alphaS.init( alphaSvalue, alphaSorder, alphaSnfmax, alphaSuseCMW);

  // Lambda for 5, 4 and 3 flavours.
  Lambda3flav        = alphaS.Lambda3();
  Lambda4flav        = alphaS.Lambda4();
  Lambda5flav        = alphaS.Lambda5();
  Lambda5flav2       = pow2(Lambda5flav);
  Lambda4flav2       = pow2(Lambda4flav);
  Lambda3flav2       = pow2(Lambda3flav);

  // Parameters of QCD evolution. Warn if pTmin must be raised.
  nGluonToQuark      = settingsPtr->mode("TimeShower:nGluonToQuark");
  pTcolCutMin        = settingsPtr->parm("TimeShower:pTmin");
  if (pTcolCutMin > LAMBDA3MARGIN * Lambda3flav / sqrt(renormMultFac))
    pTcolCut         = pTcolCutMin;
  else {
    pTcolCut         = LAMBDA3MARGIN * Lambda3flav / sqrt(renormMultFac);
    ostringstream newPTcolCut;
    newPTcolCut << fixed << setprecision(3) << pTcolCut;
    infoPtr->errorMsg("Warning in DireTimes::init: pTmin too low",
                      ", raised to " + newPTcolCut.str() );
    infoPtr->setTooLowPTmin(true);
  }
  pT2colCut          = pow2(pTcolCut);
  m2colCut           = pT2colCut;
  mTolErr            = settingsPtr->parm("Check:mTolErr");

  double pT2minQED = pow2(settingsPtr->parm("TimeShower:pTminChgQ"));
  pT2minQED = min(pT2minQED, pow2(settingsPtr->parm("TimeShower:pTminChgL")));
  pT2cutSave = create_unordered_map<int,double>
    (21,pT2colCut)
    (1,pT2colCut)(-1,pT2colCut)(2,pT2colCut)(-2,pT2colCut)
    (3,pT2colCut)(-3,pT2colCut)(4,pT2colCut)(-4,pT2colCut)
    (5,pT2colCut)(-5,pT2colCut)(6,pT2colCut)(-6,pT2colCut)
    (22,pT2minQED)
    (11,pT2minQED)(-11,pT2minQED)(13,pT2minQED)(-13,pT2minQED)
    (15,pT2minQED)(-15,pT2minQED)
    (900032,pT2minQED)(900012,pT2minQED)
    (900040,pT2minQED);

  // Parameters of Pythia QED evolution.
  pTchgQCut          = settingsPtr->parm("TimeShower:pTminChgQ");
  pT2chgQCut         = pow2(pTchgQCut);
  pTchgLCut          = settingsPtr->parm("TimeShower:pTminChgL");
  pT2chgLCut         = pow2(pTchgLCut);

  bool_settings = create_unordered_map<string,bool>
    ("doQEDshowerByL",doQEDshowerByL)
    ("doQEDshowerByQ",doQEDshowerByQ);

  usePDFalphas       = settingsPtr->flag("ShowerPDF:usePDFalphas");
  useSummedPDF       = settingsPtr->flag("ShowerPDF:useSummedPDF");
  BeamParticle* beam = nullptr;
  if (beamAPtr != nullptr || beamBPtr != nullptr) {
    beam = (beamAPtr != nullptr && particleDataPtr->isHadron(beamAPtr->id())) ?
      beamAPtr
         : (beamBPtr != nullptr && particleDataPtr->isHadron(beamBPtr->id())) ?
      beamBPtr : nullptr;
    if (beam == nullptr && beamAPtr != 0) beam = beamAPtr;
    if (beam == nullptr && beamBPtr != 0) beam = beamBPtr;
  }
  alphaS2piOverestimate = (usePDFalphas && beam != nullptr)
                        ? beam->alphaS(pT2colCut) * 0.5/M_PI
                        : (alphaSorder > 0) ? alphaS.alphaS(pT2colCut)*0.5/M_PI
                                            :  0.5 * 0.5/M_PI;

  m2cPhys = (usePDFalphas && beam != nullptr) ?
    pow2(max(0.,beam->mQuarkPDF(4))) : alphaS.muThres2(4);
  m2bPhys = (usePDFalphas && beam != nullptr) ?
    pow2(max(0.,beam->mQuarkPDF(5))) : alphaS.muThres2(5);

  // Parameters of alphaEM generation.
  alphaEMorder       = settingsPtr->mode("TimeShower:alphaEMorder");

  // Initialize alphaEM generation.
  alphaEM.init( alphaEMorder, settingsPtr);

  // Parameters of QED evolution, sums of charges, as necessary to pick flavor.
  nGammaToQuark      = settingsPtr->mode("TimeShower:nGammaToQuark");
  nGammaToLepton     = settingsPtr->mode("TimeShower:nGammaToLepton");
  sumCharge2L        = max(0, min(3, nGammaToLepton));
  sumCharge2Q        = 0.;
  if      (nGammaToQuark > 4) sumCharge2Q = 11. / 9.;
  else if (nGammaToQuark > 3) sumCharge2Q = 10. / 9.;
  else if (nGammaToQuark > 2) sumCharge2Q =  6. / 9.;
  else if (nGammaToQuark > 1) sumCharge2Q =  5. / 9.;
  else if (nGammaToQuark > 0) sumCharge2Q =  1. / 9.;
  sumCharge2Tot      = sumCharge2L + 3. * sumCharge2Q;

  // Allow massive incoming particles. Currently not supported by Pythia.
  //useMassiveBeams    = settingsPtr->flag("Beams:massiveLeptonBeams");
  useMassiveBeams    = false;

  // Z0 and W+- properties needed for gamma/Z0 mixing and weak showers.
  mZ                 = particleDataPtr->m0(23);
  gammaZ             = particleDataPtr->mWidth(23);
  thetaW             = 1. / (16. * coupSMPtr->sin2thetaW()
                       * coupSMPtr->cos2thetaW());
  mW                 = particleDataPtr->m0(24);
  gammaW             = particleDataPtr->mWidth(24);

  nFinalMax          = settingsPtr->mode("DireTimes:nFinalMax");
  usePDFmasses       = settingsPtr->flag("ShowerPDF:usePDFmasses");

  // Mode for higher-order kernels.
  kernelOrder        = settingsPtr->mode("DireTimes:kernelOrder");
  kernelOrderMPI     = settingsPtr->mode("DireTimes:kernelOrderMPI");

  // Create maps of accept/reject weights
  string key = "base";
  rejectProbability.insert( make_pair(key, multimap<double,double>() ));
  acceptProbability.insert( make_pair(key, map<double,double>() ));
  doVariations = settingsPtr->flag("Variations:doVariations");
  splittingSelName="";
  splittingNowName="";

  // Number of MPI, in case MPI forces intervention in shower weights.
  nMPI = 0;

  // Set splitting library, if already exists.
  if (splittingsPtr) splits = splittingsPtr->getSplittings();

  overhead.clear();
  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) {
    overhead.insert(make_pair(it->first,1.));
  }

  // May have to fix up recoils related to rescattering.
  allowRescatter     = settingsPtr->flag("PartonLevel:MPI")
    && settingsPtr->flag("MultipartonInteractions:allowRescatter");

  // Possibility of two predetermined hard emissions in event.
  doSecondHard    = settingsPtr->flag("SecondHard:generate");

  // Possibility to allow user veto of emission step.
  hasUserHooks       = (userHooksPtr != 0);
  canVetoEmission    = (userHooksPtr != 0)
                     ? userHooksPtr->canVetoFSREmission() : false;

  // Set initial value, just in case.
  dopTdamp           = false;
  pT2damp            = 0.;

  // Done.
  isInitSave = true;

}

//--------------------------------------------------------------------------

// Initialize bookkeeping of shower variations.

void DireTimes::initVariations() {

  // Create maps of accept/reject weights
  for ( int i=0; i < weights->sizeWeights(); ++i) {
    string key = weights->weightName(i);
    if ( key.compare("base") == 0) continue;
    if ( key.find("isr") != string::npos) continue;
    rejectProbability.insert( make_pair(key, multimap<double,double>() ));
    acceptProbability.insert( make_pair(key, map<double,double>() ));
  }

  for ( unordered_map<string, multimap<double,double> >::iterator
    it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
    it->second.clear();
  for ( unordered_map<string, map<double,double> >::iterator
    it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
    it->second.clear();

  // Done.
}

//--------------------------------------------------------------------------

// Find whether to limit maximum scale of emissions.
// Also allow for dampening at factorization or renormalization scale.

bool DireTimes::limitPTmax( Event& event, double, double) {

  // Find whether to limit pT. Begin by user-set cases.
  bool dopTlimit = false;
  dopTlimit1 = dopTlimit2 = false;
  int nHeavyCol = 0;
  if      (pTmaxMatch == 1) dopTlimit = dopTlimit1 = dopTlimit2 = true;

  // Always restrict SoftQCD processes.
  else if (infoPtr->isNonDiffractive() || infoPtr->isDiffractiveA()
    || infoPtr->isDiffractiveB() || infoPtr->isDiffractiveC() )
    dopTlimit = dopTlimit1 = dopTlimit2 = true;

  // Look if any quark (u, d, s, c, b), gluon or photon in final state.
  // Also count number of heavy coloured particles, like top.
  else {
    int n21 = 0;
    for (int i = 5; i < event.size(); ++i) {
      if (event[i].status() == -21) ++n21;
      else if (n21 == 0) {
        int idAbs = event[i].idAbs();
        if (idAbs <= 5 || idAbs == 21 || idAbs == 22) dopTlimit1 = true;
        if ( (event[i].col() != 0 || event[i].acol() != 0)
          && idAbs > 5 && idAbs != 21 ) ++nHeavyCol;
      } else if (n21 == 2) {
        int idAbs = event[i].idAbs();
        if (idAbs <= 5 || idAbs == 21 || idAbs == 22) dopTlimit2 = true;
      }
    }
    dopTlimit = (doSecondHard) ? (dopTlimit1 && dopTlimit2) : dopTlimit1;
  }

  // Dampening at factorization or renormalization scale; only for hardest.
  dopTdamp   = false;
  pT2damp    = 0.;

  // Done.
  return dopTlimit;

}

//--------------------------------------------------------------------------

void DireTimes::finalize( Event& event ) {

  // Perform resonance decays if some shower-produced resonances are
  // still left-over after evolution terminates.
  int resSize = direInfoPtr->sizeResPos();
  for ( int i=0; i < resSize; ++i){
    int iRes = direInfoPtr->getResPos(i);
    int sizeOld = event.size();
    // Construct the decay products.
    Event decays;
    decays.init("(decays)", particleDataPtr);
    decays.clear();
    int iMother = event[iRes].mother1();
    decays.append(event[iMother]);
    decays.append(event[iRes]);
    decays[decays.size()-1].mother1(0);
    processLevel.nextDecays(decays);
    if (decays.size() < 3) {
      infoPtr->errorMsg("Error in DireTimes::finalize: "
              "Failed to decay shower-induced resonance");
      continue;
    }
    // Attach decay products to event.
    int iDauMin = sizeOld;
    int iDauMax = sizeOld + decays.size()-3;
    event[direInfoPtr->getResPos(i)].statusNeg();
    event[direInfoPtr->getResPos(i)].daughters(iDauMin,iDauMax);
    int iBeg = event.size()-1;
    int iEnd = iBeg;
    for (int iDecay = 2; iDecay < decays.size(); ++iDecay) {
      if (!decays[iDecay].isFinal()) continue;
      event.append(decays[iDecay]);
      event[event.size()-1].mothers(direInfoPtr->getResPos(i),0);
      iEnd++;
    }

    // Add new system, automatically with two empty beam slots.
    int iSys = partonSystemsPtr->addSys();
    // Loop over allowed range to find all final-state particles.
    Vec4 pSum;
    for (int ii = iBeg; ii <= iEnd; ++ii) if (event[ii].isFinal()) {
      partonSystemsPtr->addOut( iSys, ii);
      pSum += event[ii].p();
    }
    partonSystemsPtr->setSHat( iSys, pSum.m2Calc() );
  }
  // Clear resonances (even if not decayed)
  direInfoPtr->clearResPos();

  // Currently, do nothing to not break other things.
  return;

}


//--------------------------------------------------------------------------

// Top-level routine to do a full time-like shower in resonance decay.

int DireTimes::shower( int iBeg, int iEnd, Event& event, double pTmax,
  int nBranchMax) {

  // Add new system, automatically with two empty beam slots.
  int iSys = partonSystemsPtr->addSys();

  // Loop over allowed range to find all final-state particles.
  Vec4 pSum;
  for (int i = iBeg; i <= iEnd; ++i) if (event[i].isFinal()) {
    partonSystemsPtr->addOut( iSys, i);
    pSum += event[i].p();
  }
  partonSystemsPtr->setSHat( iSys, pSum.m2Calc() );

  // Let prepare routine do the setup.
  dopTlimit1        = true;
  dopTlimit2        = true;
  dopTdamp          = false;
  prepare( iSys, event, true);

  // Begin evolution down in pT from hard pT scale.
  int nBranch  = 0;
  pTLastBranch = 0.;
  do {
    double pTtimes = pTnext( event, pTmax, 0.);

    // Do a final-state emission (if allowed).
    if (pTtimes > 0.) {
      if (branch( event)) {
        ++nBranch;
        pTLastBranch = pTtimes;
      }
      pTmax = pTtimes;
    }

    // Keep on evolving until nothing is left to be done.
    else pTmax = 0.;
  } while (pTmax > 0. && (nBranchMax <= 0 || nBranch < nBranchMax));

  // Return number of emissions that were performed.
  return nBranch;

}

//--------------------------------------------------------------------------

// Top-level routine for QED radiation in hadronic decay to two leptons.
// Intentionally only does photon radiation, i.e. no photon branchings.

int DireTimes::showerQED( int i1, int i2, Event& event, double pTmax) {

  // Add new system, automatically with two empty beam slots.
  int iSys = partonSystemsPtr->addSys();
  partonSystemsPtr->addOut( iSys, i1);
  partonSystemsPtr->addOut( iSys, i2);
  partonSystemsPtr->setSHat( iSys, m2(event[i1], event[i2]) );

  // Reset scales to allow setting up dipoles.
  double scale1 = event[i1].scale();
  event[i1].scale(pTmax);
  double scale2 = event[i2].scale();
  event[i2].scale(pTmax);

  // Prepare all dipoles for evolution.
  dopTlimit1        = true;
  dopTlimit2        = true;
  dopTdamp          = false;
  prepare( iSys, event, false);

  // Begin evolution down in pT from hard pT scale.
  int nBranch  = 0;
  pTLastBranch = 0.;
  do {

    double pTsel = pTnext( event, pTmax, 0., false, false);

    // Do a final-state emission (if allowed).
    if (pTsel > 0.) {
      if (branch(event)) {
        // Done with branching
        ++nBranch;
        pTLastBranch = pTsel;
       }
       pTmax = pTsel;
    // Keep on evolving until nothing is left to be done.
    } else pTmax = 0.;
  } while (pTmax > 0.);

  // Undo scale setting.
  event[i1].scale(scale1);
  event[i2].scale(scale2);

  // Return number of emissions that were performed.
  return nBranch;

}

//--------------------------------------------------------------------------

// Global recoil not used, but abuse function to reset some generic things.

void DireTimes::prepareGlobal( Event& ) {

  // Initialize weight container.
  weights->init();

  // Clear event-by-event diagnostic messages.
  direInfoPtr->clearAll();

  // Clear accept/reject weights.
  weights->reset();
  for ( unordered_map<string, multimap<double,double> >::iterator
    it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
    it->second.clear();
  for ( unordered_map<string, map<double,double> >::iterator
    it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
    it->second.clear();

  // Now also attempt to reset ISR weights!
  unordered_map<string,DireSplitting*> tmpSplits = splittingsPtr->
    getSplittings();
  for ( unordered_map<string,DireSplitting*>::iterator it = tmpSplits.begin();
    it != tmpSplits.end(); ++it ) {
    if (it->second->isr) { it->second->isr->resetWeights(); break; }
  }

  return;
}

//--------------------------------------------------------------------------

// Prepare system for evolution; identify ME.

void DireTimes::prepare( int iSys, Event& event, bool limitPTmaxIn) {

  // Calculate remainder shower weight after MPI.
  if (nMPI < infoPtr->getCounter(23)
      && iSys == infoPtr->getCounter(23) ) {
    weights->calcWeight(pow2(infoPtr->pTnow()));
    weights->reset();
    // Clear accept/reject weights.
    for ( unordered_map<string, multimap<double,double> >::iterator
      it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
      it->second.clear();
    for ( unordered_map<string, map<double,double> >::iterator
      it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
      it->second.clear();
  }
  // Store number of MPI, in case a subsequent MPI forces calculation and
  // reset of shower weights.
  nMPI = infoPtr->getCounter(23);

  // Reset dipole-ends list for first interaction and for resonance decays.
  int iInA = partonSystemsPtr->getInA(iSys);
  int iInB = partonSystemsPtr->getInB(iSys);
  if (iSys == 0 || iInA == 0) dipEnd.resize(0);
  int dipEndSizeBeg = dipEnd.size();

  // Set splitting library.
  splits = splittingsPtr->getSplittings();
  overhead.clear();
  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) {
    overhead.insert(make_pair(it->first,1.));
  }

  // No dipoles for 2 -> 1 processes.
  if (partonSystemsPtr->sizeOut(iSys) < 2) {
    // Loop through final state of system to find possible decays.
    for (int i = 0; i < partonSystemsPtr->sizeOut(iSys); ++i) {
      int iRad = partonSystemsPtr->getOut( iSys, i);
      if ( event[iRad].isFinal()
        && event[iRad].scale() > 0.
        && event[iRad].isResonance())
        // Setup decay dipoles.
        if (doDecaysAsShower) setupDecayDip( iSys, iRad, event, dipEnd);
    }
    return;
  }

  // In case of DPS overwrite limitPTmaxIn by saved value.
  if (doSecondHard && iSys == 0) limitPTmaxIn = dopTlimit1;
  if (doSecondHard && iSys == 1) limitPTmaxIn = dopTlimit2;

  dipSel = 0;

  // Loop through final state of system to find possible dipole ends.
  for (int i = 0; i < partonSystemsPtr->sizeOut(iSys); ++i) {
    int iRad = partonSystemsPtr->getOut( iSys, i);

    if (event[iRad].isFinal() && event[iRad].scale() > 0.) {

      // Find dipole end formed by colour index.
      int colTag = event[iRad].col();
      if (doQCDshower && colTag > 0) setupQCDdip( iSys, i,  colTag,  1, event,
        false, limitPTmaxIn);

      // Find dipole end formed by anticolour index.
      int acolTag = event[iRad].acol();
      if (doQCDshower && acolTag > 0) setupQCDdip( iSys, i, acolTag, -1, event,
        false, limitPTmaxIn);

      // Now find non-QCD dipoles and/or update the existing dipoles.
      getGenDip( iSys, i, iRad, event, limitPTmaxIn, dipEnd);

    // End loop over system final state. Have now found the dipole ends.
    }

    // Setup decay dipoles.
    if (doDecaysAsShower && event[iRad].isResonance())
      setupDecayDip( iSys, iRad, event, dipEnd);

  }

  // Loop through dipole ends and set matrix element correction flag.
  for (int iDip = dipEndSizeBeg; iDip < int(dipEnd.size()); ++iDip)
    dipEnd[iDip].MEtype = 0;

  // Now update masses and allowed emissions. (Not necessary here, since
  // ensured in setupQCDdip etc.)
  updateDipoles(event, iSys);

  bornColors.resize(0);
  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip) {
    int colRad  = event[dipEnd[iDip].iRadiator].col();
    int acolRad = event[dipEnd[iDip].iRadiator].acol();
    if ( dipEnd[iDip].colType > 0
     && find(bornColors.begin(), bornColors.end(), colRad) == bornColors.end())
     bornColors.push_back(colRad);
    if ( dipEnd[iDip].colType < 0
      && find(bornColors.begin(), bornColors.end(), acolRad) ==
         bornColors.end())
      bornColors.push_back(acolRad);
  }

  // Update dipole list after a multiparton interactions rescattering.
  if (iSys > 0 && ( (iInA > 0 && event[iInA].status() == -34)
    || (iInB > 0 && event[iInB].status() == -34) ) )
    rescatterUpdate( iSys, event);

  // Counter of proposed emissions.
  nProposedPT.clear();
  if ( nProposedPT.find(iSys) == nProposedPT.end() )
    nProposedPT.insert(make_pair(iSys,0));

  splittingSelName="";
  splittingNowName="";

  // Clear weighted shower book-keeping.
  for ( unordered_map<string, multimap<double,double> >::iterator
    it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
    it->second.clear();
  for ( unordered_map<string, map<double,double> >::iterator
    it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
    it->second.clear();

}

//--------------------------------------------------------------------------

void DireTimes::clear() {

  dipEnd.resize(0);
  weights->reset();
  dipSel = 0;

  splittingSelName="";
  splittingNowName="";

  // Clear weighted shower book-keeping.
  for ( unordered_map<string, multimap<double,double> >::iterator
    it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
    it->second.clear();
  for ( unordered_map<string, map<double,double> >::iterator
    it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
    it->second.clear();

}

//--------------------------------------------------------------------------

// Update dipole list after a multiparton interactions rescattering.

void DireTimes::rescatterUpdate( int iSys, Event& event) {

  // Loop over two incoming partons in system; find their rescattering mother.
  // (iOut is outgoing from old system = incoming iIn of rescattering system.)
  for (int iResc = 0; iResc < 2; ++iResc) {
    int iIn = (iResc == 0) ? partonSystemsPtr->getInA(iSys)
                           : partonSystemsPtr->getInB(iSys);
    if (iIn == 0 || event[iIn].status() != -34) continue;
    int iOut = event[iIn].mother1();

    // Loop over all dipoles.
    int dipEndSize = dipEnd.size();
    for (int iDip = 0; iDip < dipEndSize; ++iDip) {
      DireTimesEnd& dipNow = dipEnd[iDip];

      // Kill dipoles where rescattered parton is radiator.
      if (dipNow.iRadiator == iOut) {
        dipNow.colType = 0;
        dipNow.chgType = 0;
        dipNow.gamType = 0;
        continue;
      }
      // No matrix element for dipoles between scatterings.
      if (dipNow.iMEpartner == iOut) {
        dipNow.MEtype     =  0;
        dipNow.iMEpartner = -1;
      }

      // Update dipoles where outgoing rescattered parton is recoiler.
      if (dipNow.iRecoiler == iOut) {
        int iRad = dipNow.iRadiator;

        // Colour dipole: recoil in final state, initial state or new.
        if (dipNow.colType > 0) {
          int colTag = event[iRad].col();
          bool done  = false;
          for (int i = 0; i < partonSystemsPtr->sizeOut(iSys); ++i) {
            int iRecNow = partonSystemsPtr->getOut( iSys, i);
            if (event[iRecNow].acol() == colTag) {
              dipNow.iRecoiler = iRecNow;
              dipNow.systemRec = iSys;
              dipNow.MEtype    = 0;
              done             = true;
              break;
            }
          }
          if (!done) {
            int iIn2 = (iResc == 0) ? partonSystemsPtr->getInB(iSys)
                                    : partonSystemsPtr->getInA(iSys);
            if (event[iIn2].col() == colTag) {
              dipNow.iRecoiler = iIn2;
              dipNow.systemRec = iSys;
              dipNow.MEtype    = 0;
              int isrType      = event[iIn2].mother1();
              // This line in case mother is a rescattered parton.
              while (isrType > 2 + beamOffset)
                isrType = event[isrType].mother1();
              if (isrType > 2) isrType -= beamOffset;
              dipNow.isrType   = isrType;
              done             = true;
            }
          }
          // If above options failed, then create new dipole.
          if (!done) {
            int iRadNow = partonSystemsPtr->getIndexOfOut(dipNow.system, iRad);
            if (iRadNow != -1)
              setupQCDdip(dipNow.system, iRadNow, event[iRad].col(), 1,
                          event, dipNow.isOctetOnium, true);
            else
              infoPtr->errorMsg("Warning in DireTimes::rescatterUpdate: "
              "failed to locate radiator in system");

            dipNow.colType = 0;
            dipNow.chgType = 0;
            dipNow.gamType = 0;

            infoPtr->errorMsg("Warning in DireTimes::rescatterUpdate: "
            "failed to locate new recoiling colour partner");
          }

        // Anticolour dipole: recoil in final state, initial state or new.
        } else if (dipNow.colType < 0) {
          int  acolTag = event[iRad].acol();
          bool done    = false;
          for (int i = 0; i < partonSystemsPtr->sizeOut(iSys); ++i) {
            int iRecNow = partonSystemsPtr->getOut( iSys, i);
            if (event[iRecNow].col() == acolTag) {
              dipNow.iRecoiler = iRecNow;
              dipNow.systemRec = iSys;
              dipNow.MEtype    = 0;
              done             = true;
              break;
            }
          }
          if (!done) {
            int iIn2 = (iResc == 0) ? partonSystemsPtr->getInB(iSys)
                                    : partonSystemsPtr->getInA(iSys);
            if (event[iIn2].acol() == acolTag) {
              dipNow.iRecoiler = iIn2;
              dipNow.systemRec = iSys;
              dipNow.MEtype    = 0;
              int isrType      = event[iIn2].mother1();
              // This line in case mother is a rescattered parton.
              while (isrType > 2 + beamOffset)
                isrType = event[isrType].mother1();
              if (isrType > 2) isrType -= beamOffset;
              dipNow.isrType   = isrType;
              done             = true;
            }
          }
          // If above options failed, then create new dipole.
          if (!done) {
            int iRadNow = partonSystemsPtr->getIndexOfOut(dipNow.system, iRad);
            if (iRadNow != -1)
              setupQCDdip(dipNow.system, iRadNow, event[iRad].acol(), -1,
                          event, dipNow.isOctetOnium, true);
            else
              infoPtr->errorMsg("Warning in DireTimes::rescatterUpdate: "
              "failed to locate radiator in system");

            dipNow.colType = 0;
            dipNow.chgType = 0;
            dipNow.gamType = 0;

            infoPtr->errorMsg("Warning in DireTimes::rescatterUpdate: "
            "failed to locate new recoiling colour partner");
          }
        }
      }

    // End of loop over dipoles and two incoming sides.
    }
  }

}

//--------------------------------------------------------------------------

// Update dipole list after each ISR emission (so not used for resonances).

void DireTimes::update( int iSys, Event& event, bool) {

  // Store dipoles in other systems.
  vector <DireTimesEnd> dipLT;
  vector <DireTimesEnd> dipGT;
  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip) {
    if (dipEnd[iDip].system < iSys) dipLT.push_back(dipEnd[iDip]);
    if (dipEnd[iDip].system > iSys) dipGT.push_back(dipEnd[iDip]);
  }

  // Reset dipole-ends.
  dipEnd.resize(0);
  dipSel = 0;

  // No dipoles for 2 -> 1 processes.
  if (partonSystemsPtr->sizeOut(iSys) < 2) return;

  // Loop through final state of system to find possible dipole ends.
  for (int i = 0; i < partonSystemsPtr->sizeOut(iSys); ++i) {
    int iRad = partonSystemsPtr->getOut( iSys, i);

    if (event[iRad].isFinal() && event[iRad].scale() > 0.) {

      // Find dipole end formed by colour index.
      int colTag = event[iRad].col();
      if (doQCDshower && colTag > 0) setupQCDdip( iSys, i,  colTag,  1, event,
        false, true);

      // Find dipole end formed by anticolour index.
      int acolTag = event[iRad].acol();
      if (doQCDshower && acolTag > 0) setupQCDdip( iSys, i, acolTag, -1, event,
        false, true);

      // Now find non-QCD dipoles and/or update the existing dipoles.
      getGenDip( iSys, i, iRad, event, false, dipEnd);

    // End loop over system final state. Have now found the dipole ends.
    }

    // Setup decay dipoles.
    if (doDecaysAsShower && event[iRad].isResonance())
      setupDecayDip( iSys, iRad, event, dipEnd);

  }

  dipEnd.insert( dipEnd.begin(), dipLT.begin(), dipLT.end());
  dipEnd.insert( dipEnd.end(),   dipGT.begin(), dipGT.end());

  // Now update masses and allowed emissions.
  updateDipoles(event, iSys);

}

//--------------------------------------------------------------------------

// Setup a dipole end for a QCD colour charge.

void DireTimes::setupQCDdip( int iSys, int i, int colTag, int colSign,
  Event& event, bool isOctetOnium, bool limitPTmaxIn) {

  // Initial values. Find if allowed to hook up beams.
  int iRad     = partonSystemsPtr->getOut(iSys, i);
  int iRec     = 0;
  int sizeAllA = partonSystemsPtr->sizeAll(iSys);
  int sizeOut  = partonSystemsPtr->sizeOut(iSys);
  int sizeAll  = ( allowBeamRecoil ) ? sizeAllA : sizeOut;
  int sizeIn   = sizeAll - sizeOut;
  int sizeInA  = sizeAllA - sizeIn - sizeOut;
  int iOffset  = i + sizeAllA - sizeOut;
  bool otherSystemRec = false;
  bool allowInitial   = (partonSystemsPtr->hasInAB(iSys)) ? true : false;
  vector<int> iRecVec(0);

  // Colour: other end by same index in beam or opposite in final state.
  // Exclude rescattered incoming and not final outgoing.
  if (colSign > 0)
  for (int j = 0; j < sizeAll; ++j) if (j + sizeInA != iOffset) {
    int iRecNow = partonSystemsPtr->getAll(iSys, j + sizeInA);
    if ( ( j <  sizeIn && event[iRecNow].col()  == colTag
      && !event[iRecNow].isRescatteredIncoming() )
      || ( j >= sizeIn && event[iRecNow].acol() == colTag
      && event[iRecNow].isFinal() ) ) {
      iRec = iRecNow;
      break;
    }
  }

  // Anticolour: other end by same index in beam or opposite in final state.
  // Exclude rescattered incoming and not final outgoing.
  if (colSign < 0)
  for (int j = 0; j < sizeAll; ++j) if (j + sizeInA != iOffset) {
    int iRecNow = partonSystemsPtr->getAll(iSys, j + sizeInA);
    if ( ( j <  sizeIn && event[iRecNow].acol()  == colTag
      && !event[iRecNow].isRescatteredIncoming() )
      || ( j >= sizeIn && event[iRecNow].col() == colTag
      && event[iRecNow].isFinal() ) ) {
      iRec = iRecNow;
      break;
    }
  }

  // Resonance decays (= no instate):
  // other end to nearest recoiler in same system final state,
  // by (p_i + p_j)^2 - (m_i + m_j)^2 = 2 (p_i p_j - m_i m_j).
  // (junction colours more involved, so keep track if junction colour)
  bool hasJunction = false;
  if (iRec == 0 && !allowInitial) {
    for (int iJun = 0; iJun < event.sizeJunction(); ++ iJun) {
      // For types 1&2, all legs in final state
      // For types 3&4, two legs in final state
      // For types 5&6, one leg in final state
      int iBeg = (event.kindJunction(iJun)-1)/2;
      for (int iLeg = iBeg; iLeg < 3; ++iLeg)
        if (event.endColJunction( iJun, iLeg) == colTag) hasJunction  = true;
    }
    double ppMin = LARGEM2;
    for (int j = 0; j < sizeOut; ++j) if (j != i) {
        int iRecNow  = partonSystemsPtr->getOut(iSys, j);
        if (!event[iRecNow].isFinal()) continue;
        double ppNow = event[iRecNow].p() * event[iRad].p()
          - event[iRecNow].m() * event[iRad].m();
        if (ppNow < ppMin) {
          iRec  = iRecNow;
          ppMin = ppNow;
        }
      }
  }

  // If no success then look for matching (anti)colour anywhere in final state.
  if ( iRec == 0 ) {
    for (int j = 0; j < event.size(); ++j) if (event[j].isFinal()) {
      if ( (colSign > 0 && event[j].acol() == colTag)
        || (colSign < 0 && event[j].col()  == colTag) ) {
        iRec = j;
        otherSystemRec = true;
        break;
      }
    }

    // If no success then look for match to non-rescattered in initial state.
    if (iRec == 0 && allowInitial) {
      for (int iSysR = 0; iSysR < partonSystemsPtr->sizeSys(); ++iSysR)
      if (iSysR != iSys) {
        int j = partonSystemsPtr->getInA(iSysR);
        if (j > 0 && event[j].isRescatteredIncoming()) j = 0;
        if (j > 0 && ( (colSign > 0 && event[j].col() == colTag)
          || (colSign < 0 && event[j].acol()  == colTag) ) ) {
          iRec = j;
          otherSystemRec = true;
          break;
        }
        j = partonSystemsPtr->getInB(iSysR);
        if (j > 0 && event[j].isRescatteredIncoming()) j = 0;
        if (j > 0 && ( (colSign > 0 && event[j].col() == colTag)
          || (colSign < 0 && event[j].acol()  == colTag) ) ) {
          iRec = j;
          otherSystemRec = true;
          break;
        }
      }
    }
  }

  // Junctions
  // For types 1&2: all legs in final state
  //                half-strength dipoles between all legs
  // For types 3&4, two legs in final state
  //                full-strength dipole between final-state legs
  // For types 5&6, one leg in final state
  //                no final-state dipole end

  if (hasJunction) {
    for (int iJun = 0; iJun < event.sizeJunction(); ++ iJun) {
      int kindJun = event.kindJunction(iJun);
      int iBeg = (kindJun-1)/2;
      for (int iLeg = iBeg; iLeg < 3; ++iLeg) {
        if (event.endColJunction( iJun, iLeg) == colTag) {
          // For types 5&6, no other leg to recoil against. Switch off if
          // no other particles at all, since radiation then handled by ISR.
          // Example: qq -> ~t* : no radiation off ~t*
          // Allow radiation + recoil if unconnected partners available
          // Example: qq -> ~t* -> tbar ~chi0 : allow radiation off tbar,
          //                                    with ~chi0 as recoiler
          if (kindJun >= 5) {
            if (sizeOut == 1) return;
            else break;
          }
          // For junction types 3 & 4, span one full-strength dipole
          // (only look inside same decay system)
          else if (kindJun >= 3) {
            int iLegRec = 3-iLeg;
            int colTagRec = event.endColJunction( iJun, iLegRec);
            for (int j = 0; j < sizeOut; ++j) if (j != i) {
                int iRecNow  = partonSystemsPtr->getOut(iSys, j);
                if (!event[iRecNow].isFinal()) continue;
                if ( (colSign > 0 && event[iRecNow].col()  == colTagRec)
                  || (colSign < 0 && event[iRecNow].acol() == colTagRec) ) {
                  // Only accept if staying inside same system
                  iRec = iRecNow;
                  break;
                }
              }
          }
          // For junction types 1 & 2, span two half-strength dipoles
          // (only look inside same decay system)
          else {
            // Loop over two half-strength dipole connections
            for (int jLeg = 1; jLeg <= 2; jLeg++) {
              int iLegRec = (iLeg + jLeg) % 3;
              int colTagRec = event.endColJunction( iJun, iLegRec);
              for (int j = 0; j < sizeOut; ++j) if (j != i) {
                  int iRecNow  = partonSystemsPtr->getOut(iSys, j);
                  if (!event[iRecNow].isFinal()) continue;
                  if ( (colSign > 0 && event[iRecNow].col()  == colTagRec)
                    || (colSign < 0 && event[iRecNow].acol() == colTagRec) ) {
                    // Store recoilers in temporary array
                    iRecVec.push_back(iRecNow);
                    // Set iRec != 0 for checks below
                    iRec = iRecNow;
                  }
                }
            }

          }     // End if-then-else of junction kinds

        }       // End if leg has right color tag
      }         // End of loop over junction legs
    }           // End loop over junctions

  }             // End main junction if

  // If fail, then other end to nearest recoiler in same system final state,
  // by (p_i + p_j)^2 - (m_i + m_j)^2 = 2 (p_i p_j - m_i m_j).
  if (iRec == 0) {
    double ppMin = LARGEM2;
    for (int j = 0; j < sizeOut; ++j) if (j != i) {
      int iRecNow  = partonSystemsPtr->getOut(iSys, j);
      if (!event[iRecNow].isFinal()) continue;
      double ppNow = event[iRecNow].p() * event[iRad].p()
                   - event[iRecNow].m() * event[iRad].m();
      if (ppNow < ppMin) {
        iRec  = iRecNow;
        ppMin = ppNow;
      }
    }
  }

  // If fail, then other end to nearest recoiler in any system final state,
  // by (p_i + p_j)^2 - (m_i + m_j)^2 = 2 (p_i p_j - m_i m_j).
  if (iRec == 0) {
    double ppMin = LARGEM2;
    for (int iRecNow = 0; iRecNow < event.size(); ++iRecNow)
    if (iRecNow != iRad && event[iRecNow].isFinal()) {
      double ppNow = event[iRecNow].p() * event[iRad].p()
                   - event[iRecNow].m() * event[iRad].m();
      if (ppNow < ppMin) {
        iRec  = iRecNow;
        otherSystemRec = true;
        ppMin = ppNow;
      }
    }
  }

  // PS dec 2010: make sure iRec is stored in iRecVec
  if (iRecVec.size() == 0 && iRec != 0) iRecVec.push_back(iRec);

  // Remove any zero recoilers from normalization
  int nRec = iRecVec.size();
  for (unsigned int mRec = 0; mRec < iRecVec.size(); ++mRec)
    if (iRecVec[mRec] <= 0) nRec--;

  // Check for failure to locate any recoiler
  if ( nRec <= 0 ) {
    infoPtr->errorMsg("Error in DireTimes::setupQCDdip: "
                      "failed to locate any recoiling partner");
    return;
  }

  // Store dipole colour end(s).
  for (unsigned int mRec = 0; mRec < iRecVec.size(); ++mRec) {
    iRec = iRecVec[mRec];
    if (iRec <= 0) continue;
    // Max scale either by parton scale or by dipole mass.
    double pTmax = event[iRad].scale();
    if (limitPTmaxIn) {
      if (iSys == 0 || (iSys == 1 && doSecondHard)) pTmax *= pTmaxFudge;
      else if (sizeIn > 0) pTmax *= pTmaxFudgeMPI;
    //} else pTmax = 0.5 * m( event[iRad], event[iRec]);
    } else pTmax = m( event[iRad], event[iRec]);

    // Force maximal pT to LHEF input value.
    if ( abs(event[iRad].status()) > 20 &&  abs(event[iRad].status()) < 24
      && settingsPtr->flag("Beams:setProductionScalesFromLHEF")
      && event[iRad].scale() > 0.)
      pTmax = event[iRad].scale();

    int colType  = (event[iRad].id() == 21) ? 2 * colSign : colSign;
    int isrType  = (event[iRec].isFinal()) ? 0 : event[iRec].mother1();
    // This line in case mother is a rescattered parton.
    while (isrType > 2 + beamOffset) isrType = event[isrType].mother1();
    if (isrType > 2) isrType -= beamOffset;
    appendDipole( event, iRad, iRec, pTmax, colType, 0, 0, 0, isrType, iSys,
      -1, -1, 0, isOctetOnium, /*false, false, -1, -1,*/ dipEnd);

    // If hooked up with other system then find which.
    if (otherSystemRec) {
      int systemRec = partonSystemsPtr->getSystemOf(iRec, true);
      if (systemRec >= 0) dipEnd.back().systemRec = systemRec;
      dipEnd.back().MEtype = 0;
    }

  }

}

//--------------------------------------------------------------------------

// Setup a dipole end for a QCD colour charge.

void DireTimes::setupDecayDip( int iSys, int iRad, const Event& event,
  vector<DireTimesEnd>& dipEnds) {

  // Initial values. Find if allowed to hook up beams.
  int iRec          = 0;
  int sizeOut       = partonSystemsPtr->sizeOut(iSys);
  bool allowInitial = (partonSystemsPtr->hasInAB(iSys)) ? true : false;

  // First try nearest recoiler in same system in final state,
  // by (p_i + p_j)^2 - (m_i + m_j)^2 = 2 (p_i p_j - m_i m_j).
  double ppMin = LARGEM2;
  for (int j = 0; j < sizeOut; ++j) {
    int iRecNow  = partonSystemsPtr->getOut(iSys, j);
    if (iRecNow == iRad || !event[iRecNow].isFinal()) continue;
    double ppNow = event[iRecNow].p() * event[iRad].p()
      - event[iRecNow].m() * event[iRad].m();
    if (ppNow < ppMin) {
      iRec  = iRecNow;
      ppMin = ppNow;
    }
  }

  // Now try nearest recoiler in same system in initial state,
  // by -(p_i - p_j)^2 - (m_i + m_j)^2 = 2 (p_i p_j - m_i m_j).
  if (iRec == 0 && allowInitial) {
    ppMin = LARGEM2;
    // Check first beam.
    int iRecNow = partonSystemsPtr->getInA(iSys);
    double ppNow = event[iRecNow].p() * event[iRad].p()
          - event[iRecNow].m() * event[iRad].m();
    if (ppNow < ppMin) {
      iRec  = iRecNow;
      ppMin = ppNow;
    }

    // Check second beam.
    iRecNow     = partonSystemsPtr->getInB(iSys);
    ppNow       = event[iRecNow].p() * event[iRad].p()
                - event[iRecNow].m() * event[iRad].m();
    if (ppNow < ppMin) {
      iRec  = iRecNow;
      ppMin = ppNow;
    }
  }

  double pTmax = m( event[iRad], event[iRec]);
  int colType  = event[iRad].colType();
  int isrType  = (event[iRec].isFinal()) ? 0 : event[iRec].mother1();
  // This line in case mother is a rescattered parton.
  while (isrType > 2 + beamOffset) isrType = event[isrType].mother1();
  if (isrType > 2) isrType -= beamOffset;
  if (iRec > 0) {
    appendDipole( event, iRad, iRec, pTmax, colType, 0, 0, 0, isrType, 0,
          -1, -1, 0, false, dipEnds);
  }

}

//--------------------------------------------------------------------------

// Setup a dipole end for a QCD colour charge.

void DireTimes::getGenDip( int iSys, int i, int iRadIn,
  const Event& event, bool limitPTmaxIn, vector<DireTimesEnd>& dipEnds) {

  // Initial values.
  int iRad     = (iSys > -1) ? partonSystemsPtr->getOut(iSys, i) : iRadIn;
  int sizeAllA = (iSys > -1) ? partonSystemsPtr->sizeAll(iSys) : event.size();
  int sizeOut  = (iSys > -1) ? partonSystemsPtr->sizeOut(iSys) : event.size();
  int sizeAll  = (iSys > -1) ? (( allowBeamRecoil ) ? sizeAllA : sizeOut)
               : event.size();
  int sizeIn   = (iSys > -1) ? sizeAll - sizeOut : 0;
  int sizeInA  = (iSys > -1) ? sizeAllA - sizeIn - sizeOut : 0;
  int iOffset  = (iSys > -1) ? i + sizeAllA - sizeOut : 0;

  for (int j = 0; j < sizeAll; ++j) if (iSys < 0 || j + sizeInA != iOffset) {

    int iRecNow = (iSys > -1) ? partonSystemsPtr->getAll(iSys, j+sizeInA) : j;
    if ( !event[iRecNow].isFinal()
       && event[iRecNow].mother1() != 1
       && event[iRecNow].mother1() != 2) continue;

    // Skip radiator.
    if ( iRecNow == iRad) continue;

    // Skip if dipole already exists, attempt to update the dipole end (a)
    // for the current a-b dipole.
    vector<int> iDip;
    for (int k = 0; k < int(dipEnds.size()); ++k)
      if ( dipEnds[k].iRadiator == iRad && dipEnds[k].iRecoiler == iRecNow )
        iDip.push_back(k);
    if ( int(iDip.size()) > 0) {
      for (int k = 0; k < int(iDip.size()); ++k)
        updateAllowedEmissions(event, &dipEnds[iDip[k]]);
      continue;
    }

    double pTmax = event[iRad].scale();
    if (limitPTmaxIn) {
      if (iSys == 0 || (iSys == 1 && doSecondHard)) pTmax *= pTmaxFudge;
      else if (sizeIn > 0) pTmax *= pTmaxFudgeMPI;
    } else pTmax = m( event[iRad], event[iRecNow]);
    int isrType  = (event[iRecNow].isFinal()) ? 0 : event[iRecNow].mother1();
    // This line in case mother is a rescattered parton.
    while (isrType > 2 + beamOffset) isrType = event[isrType].mother1();
    if (isrType > 2) isrType -= beamOffset;

    appendDipole( event, iRad, iRecNow, pTmax, 0, 0, 0, 0, isrType,
      (iSys > -1) ? iSys : 0, -1, -1, 0, false, dipEnds);
  }

  // Done.
  return;

}

//--------------------------------------------------------------------------

// Setup a dipole end for a QCD colour charge.

void DireTimes::getQCDdip( int iRad, int colTag, int colSign,
  const Event& event, vector<DireTimesEnd>& dipEnds) {

  // Initial values. Find if allowed to hook up beams.
  int iRec     = 0;

  // Colour: other end by same index in beam or opposite in final state.
  // Exclude rescattered incoming and not final outgoing.
  if (colSign > 0)
  for (int iRecNow = 0; iRecNow < event.size(); ++iRecNow) {
    if (iRecNow == iRad) continue;
    if ( ( event[iRecNow].col()  == colTag
      && !event[iRecNow].isFinal() && !event[iRecNow].isRescatteredIncoming() )
      || ( event[iRecNow].acol() == colTag
      && event[iRecNow].isFinal() ) ) {
      iRec = iRecNow;
      break;
    }
  }

  // Anticolour: other end by same index in beam or opposite in final state.
  // Exclude rescattered incoming and not final outgoing.
  if (colSign < 0)
  for (int iRecNow = 0; iRecNow < event.size(); ++iRecNow) {
    if (iRecNow == iRad) continue;
    if ( ( event[iRecNow].acol()  == colTag
      && !event[iRecNow].isFinal() && !event[iRecNow].isRescatteredIncoming() )
      || ( event[iRecNow].col() == colTag
      && event[iRecNow].isFinal() ) ) {
      iRec = iRecNow;
      break;
    }
  }

  double pTmax = m( event[iRad], event[iRec]);
  int colType  = (event[iRad].id() == 21) ? 2 * colSign : colSign;
  int isrType  = (event[iRec].isFinal()) ? 0 : event[iRec].mother1();
  // This line in case mother is a rescattered parton.
  while (isrType > 2 + beamOffset) isrType = event[isrType].mother1();
  if (isrType > 2) isrType -= beamOffset;
  if (iRec > 0) {
    appendDipole( event, iRad, iRec, pTmax, colType, 0, 0, 0,
          isrType, 0, -1, -1, 0, false, dipEnds);
  }

}

//--------------------------------------------------------------------------

// Function to set up and append a new dipole.

bool DireTimes::appendDipole( const Event& state, int iRad, int iRec,
  double pTmax, int colType, int chgType, int gamType, int weakType,
  int isrType, int iSys, int MEtype, int iMEpartner, int weakPol,
  bool isOctetOnium, vector<DireTimesEnd>& dipEnds) {

  // Do not append if the dipole is already known!
  for (int i = 0; i < int(dipEnds.size()); ++i)
    if (dipEnds[i].iRadiator == iRad
     && dipEnds[i].iRecoiler == iRec
     && dipEnds[i].colType   == colType)
      return false;

  // Check and reset color type.
  if (colType == 0 && state[iRad].colType() != 0) {
    vector<int> shared = sharedColor(state[iRad], state[iRec]);
    // Loop through dipoles to check if a dipole with the current rad, rec
    // and colType already exists. If not, reset colType.
    int colTypeNow(0);
    for ( int i=0; i < int(shared.size()); ++i) {
      if ( state[iRad].colType() == 2 && state[iRad].col() == shared[i])
        colTypeNow = 2;
      if ( state[iRad].colType() == 2 && state[iRad].acol() == shared[i])
        colTypeNow =-2;
      if ( state[iRad].colType() == 1 && state[iRad].id() > 0
        && state[iRad].col() == shared[i])
        colTypeNow = 1;
      if ( state[iRad].colType() ==-1 && state[iRad].id() < 0
        && state[iRad].acol() == shared[i])
        colTypeNow =-1;
      bool found = false;
      for ( int j=0; j < int(dipEnds.size()); ++j) {
        if ( dipEnds[j].iRadiator == iRad && dipEnds[j].iRecoiler == iRec
          && dipEnds[j].colType == colTypeNow) { found = true; break; }
      }
      // Reset if color tag has not been found.
      if (!found) break;
    }
    colType = colTypeNow;
  }

  // Check and reset isr type.
  if ( isrType == 0 && !state[iRec].isFinal() )
    isrType = state[iRec].mother1();

  // Check if designated color charge is connected.
  if (colType != 0) {
    vector<int> share = sharedColor(state[iRad], state[iRec]);
    if (colType > 0 && find(share.begin(), share.end(), state[iRad].col())
      == share.end()) return false;
    if (colType < 0 && find(share.begin(), share.end(), state[iRad].acol())
      == share.end()) return false;
  }

  // Construct dipole.
  DireTimesEnd dipNow = DireTimesEnd( iRad, iRec, pTmax, colType, chgType,
    gamType, weakType, isrType, iSys, MEtype, iMEpartner, weakPol,
    isOctetOnium);
  dipNow.clearAllowedEmt();
  dipNow.init(state);

  if (updateAllowedEmissions(state, &dipNow)) {
    dipEnds.push_back(dipNow);
    return true;
  }

  // Done.
  return false;
}

//--------------------------------------------------------------------------

vector<int> DireTimes::sharedColor(const Particle& rad, const Particle& rec) {
  vector<int> ret;
  int radCol(rad.col()), radAcl(rad.acol()),
      recCol(rec.col()), recAcl(rec.acol());
  if ( rad.isFinal() && rec.isFinal() ) {
    if (radCol != 0 && radCol == recAcl) ret.push_back(radCol);
    if (radAcl != 0 && radAcl == recCol) ret.push_back(radAcl);
  } else if ( rad.isFinal() && !rec.isFinal() ) {
    if (radCol != 0 && radCol == recCol) ret.push_back(radCol);
    if (radAcl != 0 && radAcl == recAcl) ret.push_back(radAcl);
  } else if (!rad.isFinal() && rec.isFinal() )  {
    if (radCol != 0 && radCol == recCol) ret.push_back(radCol);
    if (radAcl != 0 && radAcl == recAcl) ret.push_back(radAcl);
  } else if (!rad.isFinal() && !rec.isFinal() ) {
    if (radCol != 0 && radCol == recAcl) ret.push_back(radCol);
    if (radAcl != 0 && radAcl == recCol) ret.push_back(radAcl);
  }
  return ret;
}

//--------------------------------------------------------------------------

// Function to update the list of all dipoles.

void DireTimes::saveSiblings(const Event& state, int iSys) {

  int sizeAllSys = partonSystemsPtr->sizeSys();
  for (int iSystem=0; iSystem < sizeAllSys; ++iSystem) {

  if (iSys > -1 && iSystem != iSys) continue;

  // Get kernel order.
  int order = kernelOrder;
  // Use simple kernels for showering secondary scatterings.
  bool hasInA = (partonSystemsPtr->getInA(iSystem) != 0);
  bool hasInB = (partonSystemsPtr->getInB(iSystem) != 0);
  if (iSystem != 0 && hasInA && hasInB) order = kernelOrderMPI;
  if (order != 4) return;

  vector<int> q, qb, g;
  int sizeSystem(partonSystemsPtr->sizeAll(iSystem)), nFinal(0);
  for ( int i = 0; i < sizeSystem; ++i) {

    int iPos = partonSystemsPtr->getAll(iSystem, i);
    if ( state[iPos].isFinal()) nFinal++;

    if (!state[iPos].isFinal()
      && state[iPos].mother1() != 1
      && state[iPos].mother1() != 2) continue;

    // Check if this is a hadron decay system.
    bool hasHadMother=false;
    int iPosNow = iPos;
    while (state[iPosNow].mother1() > 0) {
      hasHadMother = (state[iPosNow].statusAbs() > 60);
      if (hasHadMother) break;
      iPosNow = state[iPosNow].mother1();
    }
    if (hasHadMother) continue;

    if ( state[iPos].isFinal() && state[iPos].colType() == 1
      && find(q.begin(),q.end(),iPos) == q.end())
      q.push_back(iPos);

    if (!state[iPos].isFinal() && state[iPos].colType() ==-1
      && find(q.begin(),q.end(),iPos) == q.end())
      q.push_back(iPos);

    if ( state[iPos].isFinal() && state[iPos].colType() ==-1
      && find(qb.begin(),qb.end(),iPos) == qb.end())
      qb.push_back(iPos);

    if (!state[iPos].isFinal() && state[iPos].colType() == 1
      && find(qb.begin(),qb.end(),iPos) == qb.end())
      qb.push_back(iPos);

    if ( abs(state[iPos].colType()) == 2
      && find(g.begin(),g.end(),iPos) == g.end())
      g.push_back(iPos);
  }

  // Find all chains of gluon-connected dipoles.
  DireColChains chains;
  if (nFinal > 0) {
    // Start with quark ends.
    for (int i = 0; i < int(q.size()); ++i) {
      if (chains.chainOf(q[i]).size() != 0) continue;
      chains.addChain( DireSingleColChain(q[i],state, partonSystemsPtr));
    }
    // Try all antiquark ends.
    for (int i = 0; i < int(qb.size()); ++i) {
      if (chains.chainOf(qb[i]).size() != 0) continue;
      chains.addChain( DireSingleColChain(qb[i],state, partonSystemsPtr));
    }
    // Try all gluon ends.
    for (int i = 0; i < int(g.size()); ++i) {
      // Try both color ends.
      if (chains.chainOf(g[i]).size() != 0) continue;
      chains.addChain( DireSingleColChain(g[i],state, partonSystemsPtr));
    }
  }

  // For each radiator, store siblings (dipole and next adjacent dipole)
  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip) {
    if (dipEnd[iDip].system != iSystem) continue;
    if (dipEnd[iDip].colType == 0) dipEnd[iDip].clearSiblings();
    else {
      int col = dipEnd[iDip].colType > 0
              ? state[dipEnd[iDip].iRadiator].col()
              : state[dipEnd[iDip].iRadiator].acol();
      dipEnd[iDip].setSiblings(chains.chainFromCol( dipEnd[iDip].iRadiator,
        col, 2, state));
    }
  }

  }

  // Done.
}

//--------------------------------------------------------------------------

// Function to update the list of all dipoles.

void DireTimes::updateDipoles(const Event& state, int iSys) {

  // Update the dipoles, and if necesarry, flag inactive dipoles for removal.
  vector<int> iRemove;
  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip) {
    if (!updateAllowedEmissions(state, &dipEnd[iDip])
      && find(iRemove.begin(), iRemove.end(), iDip) == iRemove.end())
      iRemove.push_back(iDip);
    dipEnd[iDip].init(state);
  }

  // Now remove inactive dipoles.
  sort (iRemove.begin(), iRemove.end());
  for (int i = iRemove.size()-1; i >= 0; --i) {
    dipEnd[iRemove[i]] = dipEnd.back();
    dipEnd.pop_back();
  }

  // Check the list of dipoles.
  checkDipoles(state);
  saveSiblings(state, iSys);

}

//--------------------------------------------------------------------------

// Function to check a new dipole.

void DireTimes::checkDipoles(const Event& state) {

  // Update the dipoles, and if necessary, flag inactive dipoles for removal.
  vector<int> iRemove;
  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip) {
    DireTimesEnd& dipi = dipEnd[iDip];
    for (int jDip = iDip+1; jDip < int(dipEnd.size()); ++jDip) {
      DireTimesEnd& dipj = dipEnd[jDip];

      // Remove double-counted dipoles.
      if (dipi == dipj && find(iRemove.begin(), iRemove.end(), iDip)
        == iRemove.end()) iRemove.push_back(iDip);

      // Check with identical radiator.
      if (dipi.iRadiator == dipj.iRadiator) {
        // If both dipoles are allowed to radiate gluons, keep only dipoles
        // with colored recoiler.
        bool iEmtGlue = find(dipi.allowedEmissions.begin(),
          dipi.allowedEmissions.end(), 21) != dipi.allowedEmissions.end();
        bool jEmtGlue = find(dipj.allowedEmissions.begin(),
          dipj.allowedEmissions.end(), 21) != dipj.allowedEmissions.end();
        if (iEmtGlue && jEmtGlue) {
          bool connectI = int(sharedColor(state[dipi.iRadiator],
            state[dipi.iRecoiler]).size()) > 0;
          bool connectJ = int(sharedColor(state[dipj.iRadiator],
            state[dipj.iRecoiler]).size()) > 0;
          if ( connectI && !connectJ) dipj.removeAllowedEmt(21);
          if (!connectI &&  connectJ) dipi.removeAllowedEmt(21);
        }

        // If both dipoles are allowed to radiate quarks, keep only dipoles
        // with colored recoiler.
        bool iEmtQ = find(dipi.allowedEmissions.begin(),
          dipi.allowedEmissions.end(), 1) != dipi.allowedEmissions.end();
        bool jEmtQ = find(dipj.allowedEmissions.begin(),
          dipj.allowedEmissions.end(), 1) != dipj.allowedEmissions.end();
        if ( state[dipi.iRadiator].colType() != 0 && iEmtQ
          && state[dipj.iRadiator].colType() != 0 && jEmtQ) {
          bool connectI = int(sharedColor(state[dipi.iRadiator],
            state[dipi.iRecoiler]).size()) > 0;
          bool connectJ = int(sharedColor(state[dipj.iRadiator],
            state[dipj.iRecoiler]).size()) > 0;
          if ( connectI && !connectJ) dipj.removeAllowedEmt(1);
          if (!connectI &&  connectJ) dipi.removeAllowedEmt(1);
        }

        // If both dipoles are allowed to radiate photons, keep only dipoles
        // with charged recoiler.
        bool iEmtA = find(dipi.allowedEmissions.begin(),
          dipi.allowedEmissions.end(), 22) != dipi.allowedEmissions.end();
        bool jEmtA = find(dipj.allowedEmissions.begin(),
          dipj.allowedEmissions.end(), 22) != dipj.allowedEmissions.end();
        if (iEmtA && jEmtA) {
          bool connectI = state[dipi.iRecoiler].isCharged();
          bool connectJ = state[dipj.iRecoiler].isCharged();
          if ( connectI && !connectJ) dipj.removeAllowedEmt(22);
          if (!connectI &&  connectJ) dipi.removeAllowedEmt(22);
        }

      }
    }
  }

  // Flag inactive dipoles for removal.
  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip) {
    if (!dipEnd[iDip].canEmit() && find(iRemove.begin(), iRemove.end(), iDip)
       == iRemove.end()) iRemove.push_back(iDip);
  }

  // Now remove problematic dipoles.
  sort (iRemove.begin(), iRemove.end());
  for (int i = iRemove.size()-1; i >= 0; --i) {
    dipEnd[iRemove[i]] = dipEnd.back();
    dipEnd.pop_back();
  }

  // Now go through dipole list and perform rudimentary checks.
  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip) {
    DireTimesEnd* dip = &dipEnd[iDip];
    int iRad = dip->iRadiator;
    int iRec = dip->iRecoiler;
    // Check and reset color type.
    if (dip->colType == 0 && state[iRad].colType() != 0) {
      vector<int> shared = sharedColor(state[iRad], state[iRec]);
      // Loop through dipoles to check if a dipole with the current rad, rec
      // and colType already exists. If not, reset colType.
      int colTypeNow(0);
      for ( int i=0; i < int(shared.size()); ++i) {
        if ( state[iRad].colType() == 2 && state[iRad].col() == shared[i])
          colTypeNow = 2;
        if ( state[iRad].colType() == 2 && state[iRad].acol() == shared[i])
          colTypeNow =-2;
        if ( state[iRad].colType() == 1 && state[iRad].id() > 0
          && state[iRad].col() == shared[i])
          colTypeNow = 1;
        if ( state[iRad].colType() ==-1 && state[iRad].id() < 0
          && state[iRad].acol() == shared[i])
          colTypeNow =-1;
      }
      dip->colType = colTypeNow;
    }
    // Check and reset isr type.
    if ( dip->isrType == 0 && !state[iRec].isFinal() )
      dip->isrType = state[iRec].mother1();
  }

}

//--------------------------------------------------------------------------

bool DireTimes::updateAllowedEmissions(const Event& state, DireTimesEnd* dip) {
  // Clear any allowed emissions.
  dip->clearAllowedEmt();
  // Append all possible emissions.
  return appendAllowedEmissions(state, dip);
}

//--------------------------------------------------------------------------

// Function to set up and append a new dipole.

bool DireTimes::appendAllowedEmissions(const Event& state, DireTimesEnd* dip) {

  // Now loop through all splitting kernels to find which emissions are
  // allowed from the current radiator-recoiler combination.
  bool isAllowed = false;
  int iRad(dip->iRadiator), iRec(dip->iRecoiler);
  pair<int,int> iRadRec(make_pair(iRad, iRec));
  pair<int,int> iRecRad(make_pair(iRec, iRad));

  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) {

    // Check if splitting is allowed.
    bool allowed = it->second->useFastFunctions()
                 ? it->second->canRadiate(state,iRad,iRec)
                 : it->second->canRadiate(state,iRadRec,bool_settings);
    if (!allowed) continue;

    // Get emission id.
    vector<int> re = it->second->radAndEmt( state[iRad].id(), dip->colType);

    // Do not decay resonances that were not generated by previous emissions.
    if ( particleDataPtr->isResonance(state[iRad].id())
      && !direInfoPtr->isRes(iRad) && state[iRad].id() != re[0])
      continue;

    for (int iEmtAft=1; iEmtAft < int(re.size()); ++iEmtAft) {
      int idEmtAft = re[iEmtAft];
      if (it->second->is_qcd) {
        idEmtAft = abs(idEmtAft);
        if (idEmtAft<10) idEmtAft = 1;
      }

      if (!it->second->isPartial()) {
        dip->appendAllowedEmt(idEmtAft);
        isAllowed = true;
      } else {
        // Now check that emission also allowed when radiating from recoiler.
        bool isPartialFractioned = false;
        for ( unordered_map<string,DireSplitting*>::iterator
          itRec = splits.begin(); itRec != splits.end(); ++itRec ) {

          if ( isPartialFractioned ) break;
          bool allowedRec = itRec->second->useFastFunctions()
                 ? itRec->second->canRadiate(state,iRec,iRad)
                 : itRec->second->canRadiate(state,iRecRad,bool_settings);

          if (!allowedRec) continue;

          // Get emission id.
          int colTypeRec
            = state[iRec].isFinal() ? -dip->colType : dip->colType;
          vector<int> reRec
            = itRec->second->radAndEmt( state[iRec].id(), colTypeRec);

          for (int iEmtAftRec=1; iEmtAftRec<int(reRec.size()); ++iEmtAftRec) {
            int idEmtAftRec = reRec[iEmtAftRec];
            if (itRec->second->is_qcd) {
              idEmtAftRec = abs(idEmtAftRec);
              if (idEmtAftRec<10) idEmtAftRec = 1;
            }
            if (idEmtAftRec == idEmtAft) { isPartialFractioned = true; break;}
          }
        }
        // Only allow if the emission can be performed from both dipole ends.
        if (isPartialFractioned) {
          dip->appendAllowedEmt(idEmtAft);
          isAllowed = true;
        }
      }
    }
  }

  // Done.
  return isAllowed;
}

//--------------------------------------------------------------------------

// Select next pT in downwards evolution of the existing dipoles.
// Classical Sudakov veto algorithm to produce next set of state variables.

double DireTimes::pTnext( Event& event, double pTbegAll, double pTendAll,
  bool, bool doTrialIn ) {

  direInfoPtr->message(1) << "Next FSR starting from " << pTbegAll << endl;

  // Begin loop over all possible radiating dipole ends.
  dipSel  = 0;
  iDipSel = -1;
  double pT2sel = pTendAll * pTendAll;
  splittingNowName="";
  splittingSelName="";
  splitInfoSel.clear();
  kernelSel.clear();
  kernelNow.clear();
  auxSel = overSel = auxNow = overNow = 0.;
  splittingSel = 0;

  // Remember if this is a trial emission.
  doTrialNow    = doTrialIn;

  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip) {
    DireTimesEnd& dip = dipEnd[iDip];

    // Limit final state multiplicity. For debugging only
    int nfmax = settingsPtr->mode("DireTimes:nFinalMax");
    int nFinal = 0;
    for (int i=0; i < event.size(); ++i)
      if (event[i].isFinal()) nFinal++;
    if (nfmax > -10 && nFinal > nfmax) continue;

    // Dipole properties.
    dip.mRad   = event[dip.iRadiator].m();
    dip.mRec   = event[dip.iRecoiler].m();
    dip.mDip   =
      sqrt( abs(2. * event[dip.iRadiator].p() * event[dip.iRecoiler].p()) );
    dip.m2Rad  = pow2(dip.mRad);
    dip.m2Rec  = pow2(dip.mRec);
    dip.m2Dip  = pow2(dip.mDip);

    // Find maximum mass scale for dipole.
    dip.m2DipCorr    = dip.m2Dip;

    double pT2start = min( dip.m2Dip, pTbegAll*pTbegAll);
    double pT2stop  = max( pT2cutMin(&dip), pTendAll*pTendAll);
    pT2stop         = max( pT2stop, pT2sel);

    // Reset emission properties.
    dip.pT2         =  0.0;
    dip.z           = -1.0;
    dip.phi         = -1.0;
    // Reset properties of 1->3 splittings.
    dip.sa1         =  0.0;
    dip.xa          = -1.0;
    dip.phia1       = -1.0;
    dip.mass.clear();
    dip.idRadAft = 0;
    dip.idEmtAft = 0;

    // Do not try splitting if the corrected dipole mass is negative.
    if (dip.m2DipCorr < 0.) {
      infoPtr->errorMsg("Warning in DireTimes::pTnext: "
      "negative dipole mass.");
      continue;
    }

    // Evolution.
    if (pT2start > pT2sel) {

      // Store start/end scales.
      dip.pT2start = pT2start;
      dip.pT2stop  = pT2stop;

      // Do evolution if it makes sense.
      if ( dip.canEmit() ) pT2nextQCD(pT2start, pT2stop, dip, event);

      // Update if found pT larger than current maximum.
      if (dip.pT2 > pT2sel) {
        pT2sel  = dip.pT2;
        dipSel  = &dip;
        iDipSel = iDip;
        splittingSelName = splittingNowName;
        splitInfoSel.store(splits[splittingSelName]->splitInfo);
        splittingSel = splits[splittingSelName];
        kernelSel = kernelNow;
        auxSel    = auxNow;
        overSel   = overNow;
        boostSel  = boostNow;
      }

    }
  }

  // Update the number of proposed timelike emissions.
  if (dipSel != 0 && nProposedPT.find(dipSel->system) != nProposedPT.end())
    ++nProposedPT[dipSel->system];

  // Insert additional weights.
  for ( unordered_map<string, multimap<double,double> >::iterator
    itR = rejectProbability.begin(); itR != rejectProbability.end(); ++itR)
    weights->insertWeights(acceptProbability[itR->first], itR->second,
      itR->first);

  for ( unordered_map<string, multimap<double,double> >::iterator
    it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
    it->second.clear();
  for ( unordered_map<string, map<double,double> >::iterator
    it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
    it->second.clear();

  resetOverheadFactors();

  // Timelike shower of resonances: Finalize in case resonances where produced.
  bool hasInAB = false;
  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip)
    if (partonSystemsPtr->hasInAB(dipEnd[iDip].system)) hasInAB = true;
  if (dipSel == 0 && !hasInAB ) finalize(event);

  // Return nonvanishing value if found pT bigger than already found.
  return (dipSel == 0) ? 0. : sqrt(pT2sel);

}

//--------------------------------------------------------------------------

double DireTimes::newPoint( const Event& inevt) {

  int asOrderSave = alphaSorder;
  double pT2minMECsSave = pT2minMECs;
  alphaSorder = 0.;
  pT2minMECs = 0.;
  double tFreeze = 1.;

  Event event(inevt);

  // Begin loop over all possible radiating dipole ends.
  dipSel  = 0;
  iDipSel = -1;
  double pTendAll = 0.;
  double pT2sel = pTendAll * pTendAll;
  splittingNowName="";
  splittingSelName="";
  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) it->second->splitInfo.clear();
  splitInfoSel.clear();
  kernelSel.clear();
  kernelNow.clear();
  auxSel = overSel = auxNow = overNow = 0.;
  splittingSel = 0;

  int nTrials=0;
  int nTrialsMax=200;
  while (pT2sel==0.) {

    nTrials++;
    if (nTrials>=nTrialsMax) break;

    for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip) {

      DireTimesEnd& dip = dipEnd[iDip];

      // Dipole properties.
      dip.mRad   = event[dip.iRadiator].m();
      dip.mRec   = event[dip.iRecoiler].m();
      dip.mDip   =
        sqrt( abs(2. * event[dip.iRadiator].p() * event[dip.iRecoiler].p()) );
      dip.m2Rad  = pow2(dip.mRad);
      dip.m2Rec  = pow2(dip.mRec);
      dip.m2Dip  = pow2(dip.mDip);

      // Find maximum mass scale for dipole.
      dip.m2DipCorr    = dip.m2Dip;

      double pT2start = m2Max(iDip, event);
      double pT2stop  = pTendAll*pTendAll;

      // Reset emission properties.
      dip.pT2         =  0.0;
      dip.z           = -1.0;
      dip.phi         = -1.0;
      // Reset properties of 1->3 splittings.
      dip.sa1         =  0.0;
      dip.xa          = -1.0;
      dip.phia1       = -1.0;
      dip.mass.clear();
      dip.idRadAft = 0;
      dip.idEmtAft = 0;

      // Store start/end scales.
      dip.pT2start = pT2start;
      dip.pT2stop  = pT2stop;

      // Do evolution if it makes sense.
      if (dip.canEmit()) pT2nextQCD(pT2start,0.,dip,event,0.,tFreeze,true);

      if (dip.pT2 > pT2sel) {
        pT2sel = dip.pT2;
        dipSel  = &dip;
        iDipSel = iDip;
        splittingSelName = splittingNowName;
        splitInfoSel.store(splits[splittingSelName]->splitInfo);
        splittingSel = splits[splittingSelName];
        kernelSel = kernelNow;
      }

    }

    if (dipSel) {
      bool hasBranched = false;
      if ( event[dipSel->iRecoiler].isFinal())
           hasBranched = branch_FF(event, true, &splitInfoSel);
      else hasBranched = branch_FI(event, true, &splitInfoSel);
      if (!hasBranched) {
        dipSel  = 0;
        iDipSel =-1;
        pT2sel  = 0.;
        splitInfoSel.clear();
        splittingSel = 0;
      }
    }

  }

  alphaSorder = asOrderSave;
  pT2minMECs  = pT2minMECsSave;

  // Unable to produce splitting.
  if (dipSel == 0) return 0.;

  // Return nonvanishing value if found pT bigger than already found.
  return sqrt(pT2sel);

}

//--------------------------------------------------------------------------

double DireTimes::enhanceOverestimateFurther( string name, int, double tOld) {

  if (tOld < pT2minEnhance) return 1.;
  double enhance = weights->enhanceOverestimate(name);
  return enhance;

}

//--------------------------------------------------------------------------

double DireTimes::overheadFactorsMEC( const Event&, DireSplitInfo*, string) {
  return 1.;
}

//--------------------------------------------------------------------------

double DireTimes::overheadFactors( DireTimesEnd* dip, const Event& state,
  string name, double, double tOld, double xOld) {

  double factor = 1.;
  double MARGIN = 1.;
  // For very low cut-offs, reduce headroom factor.
  if (tOld < pT2colCut*1.25) MARGIN = 1.;
  factor *= MARGIN;

  // Additional enhancement if PDFs vary significantly when increasing x.
  if (tOld > PT2MIN_PDF_IN_OVERESTIMATE && tOld > pT2colCut
    && !state[dip->iRecoiler].isFinal()
    && particleDataPtr->colType(state[dip->iRecoiler].id()) != 0) {

    BeamParticle* beam = nullptr;
    if (beamAPtr != nullptr || beamBPtr != nullptr) {
      if (dip->isrType == 1 && beamAPtr != nullptr) beam = beamAPtr;
      if (dip->isrType != 1 && beamBPtr != nullptr) beam = beamBPtr;
    }

    if (beam != nullptr) {

      double idRec       = state[dip->iRecoiler].id();
      int    iSysRec     = dip->systemRec;
      double scale2      = max(tOld, pT2colCut);
      bool   inOld       = beam->insideBounds(xOld, scale2);
      double xPDFOld     = getXPDF(idRec, xOld, scale2, iSysRec, beam);

      // Try to find largest PDF ratio for initial gluon at low scale (where
      // small changes in x can have a very large numerical effect.
      if (idRec == 21 && scale2 < PT2_INCREASE_OVERESTIMATE) {
        double xPDFmother = xPDFOld;
        int NTSTEPS(3), NXSTEPS(3);
        for (int i=1; i <= NTSTEPS; ++i) {
          double tNew = pT2colCut + double(i)/double(NTSTEPS)*
            (scale2-pT2colCut);
          for (int j=1; j <= NXSTEPS; ++j) {
            double xNew = xOld + double(j)/double(NXSTEPS)*(0.999999-xOld);
            double xPDFnew = getXPDF(21, xNew, tNew, iSysRec, beam);
            if ( beam->insideBounds(xNew, tNew) )
              xPDFmother = max(xPDFmother, xPDFnew);
          }
        }
        if ( inOld && abs(xPDFOld) > tinypdf(xOld) && xPDFmother/xPDFOld > 1.)
          factor *= xPDFmother / xPDFOld;

      } else {
        double tNew1       = pT2colCut;
        double tNew2       = pT2colCut + 0.5 * ( scale2 - pT2colCut );
        double xNew1       = xOld;
        double xNew2       = xOld + 0.5 * ( 0.999999 - xOld );
        bool   inNew       =   beam->insideBounds(xNew1, tNew1)
                            || beam->insideBounds(xNew1, tNew2)
                            || beam->insideBounds(xNew2, tNew1)
                            || beam->insideBounds(xNew2, tNew2);
        double xPDFNew1    = getXPDF(idRec, xNew1, tNew1, iSysRec, beam);
        double xPDFNew2    = getXPDF(idRec, xNew1, tNew2, iSysRec, beam);
        double xPDFNew3    = getXPDF(idRec, xNew2, tNew1, iSysRec, beam);
        double xPDFNew4    = getXPDF(idRec, xNew2, tNew2, iSysRec, beam);
        double PDFNew      = max( 1./xNew1 * max(xPDFNew1,xPDFNew2),
                                  1./xNew2 * max(xPDFNew3,xPDFNew4) );
        if ( inOld && inNew && xPDFOld > tinypdf(xOld)
          && abs((PDFNew)/(1./xOld*xPDFOld)) > 10)
          factor *= abs(PDFNew/(1./xOld*xPDFOld));

      }
    }
  }

  if ( !state[dip->iRecoiler].isFinal()
    && max(tOld, pT2colCut) < PT2_INCREASE_OVERESTIMATE
    && ( name == "Dire_fsr_qcd_1->1&21" || name == "Dire_fsr_qcd_21->21&21a"
      || name == "Dire_fsr_qcd_21->1&1a")) factor *= 2.;

  if (!state[dip->iRecoiler].isFinal() && tOld > pT2minMECs && doMEcorrections)
    factor *= 3.;

  // Multiply dynamically adjusted overhead factor.
  if ( overhead.find(name) != overhead.end() ) factor *= overhead[name];

  return factor;

}



//--------------------------------------------------------------------------

// Function to generate new user-defined overestimates to evolution.

void DireTimes::getNewOverestimates( DireTimesEnd* dip, const Event& state,
  double tOld, double xOld, double zMinAbs, double zMaxAbs,
  multimap<double,string>& newOverestimates) {

  double sum=0.;
  pair<int,int> iRadRec(make_pair(dip->iRadiator, dip->iRecoiler));

  // Loop over splitting names and get overestimates.
  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) {

    string name = it->first;

    it->second->splitInfo.clear();

    // Check if splitting should partake in evolution.
    bool allowed = it->second->useFastFunctions()
                 ? it->second->canRadiate(state,dip->iRadiator,dip->iRecoiler)
                 : it->second->canRadiate(state,iRadRec,bool_settings);

    // Skip if splitting is not allowed.
    if (!allowed) continue;

    // Check if dipole end can really radiate this particle.
    vector<int> re = it->second->radAndEmt(state[dip->iRadiator].id(),
      dip->colType);
    if (int(re.size()) < 2) continue;

    for (int iEmtAft=1; iEmtAft < int(re.size()); ++iEmtAft) {
      int idEmtAft = re[iEmtAft];
      if (it->second->is_qcd) {
        idEmtAft = abs(idEmtAft);
        if (idEmtAft<10) idEmtAft = 1;
      }
      if (find(dip->allowedEmissions.begin(), dip->allowedEmissions.end(),
        idEmtAft) == dip->allowedEmissions.end() ) allowed = false;
      // Disallow below cut-off.
      if ( pT2cut(idEmtAft) > tOld) allowed = false;
    }
    // Skip if splitting is not allowed.
    if (!allowed) continue;

    it->second->splitInfo.set_pT2Old  ( tOld );
    it->second->splitInfo.storeRadBef(state[dip->iRadiator]);
    it->second->splitInfo.storeRecBef(state[dip->iRecoiler]);

    // Discard below the cut-off for the splitting.
    if (!it->second->aboveCutoff( tOld, state[dip->iRadiator],
      state[dip->iRecoiler], dip->system, partonSystemsPtr)) continue;

    // Get kernel order.
    int order = kernelOrder;
    // Use simple kernels for showering secondary scatterings.
    bool hasInA = (partonSystemsPtr->getInA(dip->system) != 0);
    bool hasInB = (partonSystemsPtr->getInB(dip->system) != 0);
    if (dip->system != 0 && hasInA && hasInB) order = kernelOrderMPI;

    // Check if this is a hadron decay system.
    bool hasHadMother=false;
    int iPos = dip->iRadiator;
    while (state[iPos].mother1() > 0) {
      hasHadMother = (state[iPos].statusAbs() > 60);
      if (hasHadMother) break;
      iPos = state[iPos].mother1();
    }
    if (hasHadMother) order = kernelOrderMPI;

    double wt = it->second->overestimateInt(zMinAbs, zMaxAbs, tOld,
                                           dip->m2Dip, order);

    // Include artificial enhancements.
    wt *= overheadFactors(dip, state, name, dip->m2Dip, tOld, xOld);

    // Now add user-defined enhance factor.
    double enhanceFurther
      = enhanceOverestimateFurther(name, state[dip->iRadiator].id(), tOld);
    wt *= enhanceFurther;

    //if (it->second->hasMECBef(state, tOld)) wt *= KERNEL_HEADROOM;

    if (!dryrun && it->second->hasMECBef(state, tOld)) wt *= KERNEL_HEADROOM;
    int nFinal = 0;
    for (int i=0; i < state.size(); ++i) if (state[i].isFinal()) nFinal++;
    if (!dryrun) wt *= it->second->overhead
                   (dip->m2Dip*xOld, state[dip->iRadiator].id(), nFinal);

    // Save this overestimate.
    // Do not include zeros (could lead to trouble with lower_bound?)
    if (wt != 0.) {
      sum += abs(wt);
      newOverestimates.insert(make_pair(sum,name));
    }
  }

}

//--------------------------------------------------------------------------

// Function to generate new user-defined overestimates to evolution.

void DireTimes::getNewSplitting( const Event& state, DireTimesEnd* dip,
  double tOld, double xOld, double t,
  double zMinAbs, double zMaxAbs, int idMother, string name, bool forceFixedAs,
  int& idDaughter, int& idSister, double& z, double& wt,
  unordered_map<string,double>& full, double& over) {

  // Pointer to splitting for easy/fast access.
  DireSplitting* splitNow = splits[name];

  splitNow->splitInfo.storeRadBef ( state[dip->iRadiator]);
  splitNow->splitInfo.storeRecBef ( state[dip->iRecoiler]);

  // Return mother and sister ids.
  vector<int> re = splitNow->radAndEmt(idMother, dip->colType);
  // Exit if final state after splitting is not well-defined.
  if (int(re.size()) < 2) { wt = over = 0.; full.clear(); return; }
  idDaughter = re[0];
  idSister   = re[1];
  int nEmissions = splitNow->nEmissions();

  // Set flavours.
  int flavour = 0;
  if (idSister > 20) flavour = idSister;

  // Reject below cut-off.
  if ( pT2cut(idSister) > t) { wt = over = 0.; full.clear(); return; }

  // Return auxiliary variable, mother and sister ids.
  if (z<0.) z = splitNow->zSplit(zMinAbs, zMaxAbs, dip->m2Dip);

  // Hard cut-off on lepton energy fraction.
  if (splitNow->is(splittingsPtr->fsrQED_11_to_11_and_22_notPartial)
    && z > LEPTONZMAX) { wt = over = 0.; full.clear(); return; }

  // Discard below the cut-off for the splitting.
  if (!splitNow->aboveCutoff( t, state[dip->iRadiator], state[dip->iRecoiler],
    dip->system, partonSystemsPtr)) { wt = over = 0.; full.clear(); return; }

  // Flavour choice for g --> q qbar, or for q -> q'
  double RNflav = rndmPtr->flat();
  int sign = (idSister > 0) ? 1 : -1;
  if ( flavour == 0 && splitNow->is_dire && !splitNow->canUseForBranching()) {

    // Choose flavour for g --> qq~ splitting.
    if ( splitNow->is(splittingsPtr->fsrQCD_21_to_1_and_1a)
      || splitNow->is(splittingsPtr->fsrQCD_21_to_1_and_1b))
      idSister = sign*min(5, 1 + int(nGluonToQuark * RNflav));

    // Choose flavour for g --> qq~ splitting.
    if ( splitNow->is(splittingsPtr->fsrQCD_21_to_1_and_1_notPartial)) {
      idSister = sign*min(5, 1 + int(nGluonToQuark * RNflav));
      if (rndmPtr->flat() < 0.5) idSister *= -1;
    }

    // Choose flavour for q --> q' splitting.
    if (splitNow->is(splittingsPtr->fsrQCD_1_to_2_and_1_and_2)) {
      int index = int((2*nGluonToQuark - 2)*RNflav);
      vector<int> quarks;
      for (int i =-nGluonToQuark; i <=nGluonToQuark; ++i)
        if (abs(i) != state[dip->iRadiator].idAbs() && i != 0)
          quarks.push_back(i);
      idSister = quarks[index];
    }

    // Choose flavour for q --> qbar splitting.
    if (splitNow->is(splittingsPtr->fsrQCD_1_to_1_and_1_and_1))
      idSister = -idMother;

    flavour    = idSister;
    idDaughter = -idSister;

    // For 1->3 splittings, radiator keeps flavour.
    if (nEmissions == 2) idDaughter = state[dip->iRadiator].id();

  } else
    flavour    = idSister;

  // Store the flavour of the splitting.
  dip->flavour       = flavour;

  bool canUseSplitInfo = splitNow->canUseForBranching();
  if (canUseSplitInfo) {
    flavour    = re[1];
    idDaughter = re[0];
  }

  // Get particle masses.
  // Radiator before splitting.
  double m2Bef = particleDataPtr->isResonance(state[dip->iRadiator].id())
               ? getMass(state[dip->iRadiator].id(),3,
                         state[dip->iRadiator].mCalc())
               : (state[dip->iRadiator].idAbs() < 6
               || state[dip->iRadiator].id() == 21
               || state[dip->iRadiator].id() == 22)
               ? getMass(state[dip->iRadiator].id(),2)
               : getMass(state[dip->iRadiator].id(),1);
  // Radiator after splitting.
  double m2r   = particleDataPtr->isResonance(idDaughter)
                 && idDaughter == state[dip->iRadiator].id()
               ? getMass(idDaughter,3,state[dip->iRadiator].mCalc())
               : (abs(idDaughter) < 6 || idDaughter == 21 || idDaughter == 22)
               ? getMass(idDaughter,2)
               : getMass(idDaughter,1);
  // Recoiler.
  double m2s = 0.0;
  int type   = (state[dip->iRecoiler].isFinal()) ? 1 : -1;
  if (type == 1) {
    m2s        = particleDataPtr->isResonance(state[dip->iRecoiler].id())
               ? getMass(state[dip->iRecoiler].id(),3,
                         state[dip->iRecoiler].mCalc())
               : (state[dip->iRecoiler].idAbs() < 6
               || state[dip->iRecoiler].id() == 21
               || state[dip->iRecoiler].id() == 22)
               ? getMass(state[dip->iRecoiler].id(),2)
               : getMass(state[dip->iRecoiler].id(),1);
  }
  // Emission.
  double m2e = (abs(flavour) < 6 || flavour == 21 || flavour == 22)
             ? getMass(flavour,2) : getMass(flavour,1);

  // Special case for resonance decay.
  if ( particleDataPtr->isResonance(idDaughter)
    && idDaughter != state[dip->iRadiator].id() ) {
    // Radiator after splitting.
    m2r = pow2(particleDataPtr->mSel(idDaughter));
    // Emission.
    if ( particleDataPtr->isResonance(flavour) ) {
      m2e = pow2(particleDataPtr->mSel(flavour));
    }
    if (particleDataPtr->isResonance(state[dip->iRadiator].id())
      && sqrt(m2Bef) < sqrt(m2r) + sqrt(m2e) )
      m2e = pow2( sqrt(m2Bef) - sqrt(m2r));
    type = 0;
  }

  // Upate type if this is a massive splitting.
  if ( type != 0 && (m2Bef > TINYMASS || m2r > TINYMASS || m2s > TINYMASS
                      || m2e > TINYMASS))
    type = type/abs(type)*2;

  // Recalculate the kinematicaly available dipole mass.
  int massSign = (type > -1) ? 1 : -1;
  // Dipole invariant mass.
  double q2 = ( state[dip->iRecoiler].p()
              + massSign*state[dip->iRadiator].p() ).m2Calc();
  // Recalculate the kinematicaly available dipole mass.
  double Q2 = dip->m2Dip + massSign*(m2Bef - m2r - m2e);

  // Set kinematics mapping, as needed to check limits.
  // 1 --> Dire
  // 2 --> Catani-Seymour
  int kinType = splitNow->kinMap();

  dip->z = z;
  dip->pT2 = t;
  // Already pick phi value here, since we may need to construct the
  // momenta to evaluate the splitting probability.
  dip->phi   = 2.*M_PI*rndmPtr->flat();

  // Remember masses.
  double m2a = getMass(-idSister,2);
  double m2i = getMass( idSister,2);
  double m2j = m2r;
  if (canUseSplitInfo) {
    m2a = getMass(re[0],2);
    // Note: m2i and m2j are swapped, since it is assumed
    // that j is radiated first.
    m2i = getMass(re[1],2);
    if ( int(re.size()) > 2) {
      m2j = getMass(re[2],2);
      swap (m2i, m2j);
    }
  }

  dip->mass.clear();
  dip->idRadAft = 0;
  dip->idEmtAft = 0;
  splitNow->splitInfo.clearRadAft();
  splitNow->splitInfo.clearEmtAft();
  splitNow->splitInfo.clearEmtAft2();

  // Generate additional variables for 1->3 splitting.
  if ( nEmissions == 2 ) {
    dip->mass.push_back(m2a);
    dip->mass.push_back(m2i);
    dip->mass.push_back(m2j);
    dip->mass.push_back(m2s);
    // Consider endpoint projections.
    if (splitNow->allow_sai_endpoint_for_kinematics())
      splitNow->try_sai_endpoint();
    if (splitNow->allow_xa_endpoint_for_kinematics())
      splitNow->try_xa_endpoint();
    // Choose xa flat in [z, 1.0]
    if (!splitNow->is_xa_endpoint())  zCollNextQCD( dip, dip->z, 1. );
    else                              dip->xa = 1.;
    // Choose sai.
    if (!splitNow->is_sai_endpoint()) virtNextQCD( dip, 0.0, dip->m2Dip);
    else                              dip->sa1 = 0.;

    // Choose phi flat in [0, 2Pi]
    dip->phia1 = 2.*M_PI*rndmPtr->flat();
  }

  // Set correct variables for 1->3 splitting.
  vector <double> aux;
  if ( nEmissions == 2 ) {
    type       = (state[dip->iRecoiler].isFinal()) ? 2 : -2;
    aux.push_back( dip->m2Dip );
    if (type > 0) aux.push_back( (state[dip->iRadiator].p()
                                 +state[dip->iRecoiler].p()).m2Calc() );
    else          aux.push_back( (state[dip->iRadiator].p()
                                 -state[dip->iRecoiler].p()).m2Calc() );
    aux.push_back(dip->pT2);
    aux.push_back(dip->sa1);
    aux.push_back(dip->z);
    aux.push_back(dip->xa);
    aux.push_back(m2Bef);
    aux.push_back(m2a);
    aux.push_back(m2i);
    aux.push_back(m2j);
    aux.push_back(m2s);
  }

  // Note: m2i and m2j were swapped, since it is assumed
  // that j is radiated first. For correct storage, swap back.
  if (canUseSplitInfo) swap (m2i, m2j);

  // Setup splitting information.
  double xBef = (type > 0) ? 0.0 : 2.*state[dip->iRecoiler].e()/state[0].m();
  splitNow->splitInfo.storeInfo(name, type, dip->system, dip->systemRec, 0,
    dip->iRadiator, dip->iRecoiler, state,
    dip->flavour, idDaughter, nEmissions, Q2, dip->pT2, dip->pT2Old,
    dip->z, dip->phi, m2Bef, m2s, (nEmissions == 1 ? m2r : m2a),
    (nEmissions == 1 ? m2e : m2i), dip->sa1, dip->xa, dip->phia1, m2j,
    xBef, -1.);
  splitNow->splitInfo.setSiblings(dip->iSiblings);
  if (canUseSplitInfo) {
    splitNow->splitInfo.setRadAft(re[0]);
    splitNow->splitInfo.setEmtAft(re[1]);
    if (nEmissions==2) splitNow->splitInfo.setEmtAft2(re[2]);
    splitNow->splitInfo.canUseForBranching(true);
  } else {
    splitNow->splitInfo.setRadAft(idDaughter);
    splitNow->splitInfo.setEmtAft(idSister);
    if (nEmissions==2) splitNow->splitInfo.setEmtAft2(-idSister);
  }

  // Check phase space limits.
  double zcheck(z), tcheck(t);
  if (kinType==99) {
    zcheck = (type<0)
      ? splitNow->zdire_fi(z, t, Q2) : splitNow->zdire_ff(z, t, Q2);
    tcheck = (type<0)
      ? splitNow->tdire_fi(z, t, Q2) : splitNow->tdire_ff(z, t, Q2);
  }
  if ( !inAllowedPhasespace( kinType, zcheck, tcheck, Q2, q2,
          xOld, type, m2Bef, m2r, m2s, m2e, aux ) )
    { wt = over = 0.; full.clear(); return; }

  // Get kernel order.
  int order = kernelOrder;
  // Use simple kernels for showering secondary scatterings.
  bool hasInA = (partonSystemsPtr->getInA(dip->system) != 0);
  bool hasInB = (partonSystemsPtr->getInB(dip->system) != 0);
  if (dip->system != 0 && hasInA && hasInB) order = kernelOrderMPI;

  // Check if this is a hadron decay system.
  bool hasHadMother=false;
  int iPos = dip->iRadiator;
  while (state[iPos].mother1() > 0) {
    hasHadMother = (state[iPos].statusAbs() > 60);
    if (hasHadMother) break;
    iPos = state[iPos].mother1();
  }
  if (hasHadMother) order = kernelOrderMPI;

  // Set splitting colors, if necessary.
  if (canUseSplitInfo) {
    vector< pair<int,int> > cols
      = splitNow->radAndEmtCols( dip->iRadiator, dip->colType, state);
    splitNow->splitInfo.setRadAft(re[0], cols[0].first, cols[0].second);
    splitNow->splitInfo.setEmtAft(re[1], cols[1].first, cols[1].second);
    if (nEmissions==2) splitNow->splitInfo.setEmtAft2(re[2], cols[2].first,
      cols[2].second);
  }

  dip->idRadAft = idDaughter;
  dip->idEmtAft = idSister;

  // Return overestimate.
  over        = splitNow->overestimateDiff(z, dip->m2Dip, order);

  // Get complete kernel.
  if (splitNow->calc( state, order) ) full = splitNow->getKernelVals();

  if (!dryrun && splitNow->hasMECBef(state, tOld)) over *= KERNEL_HEADROOM;
  if (!dryrun && splitNow->hasMECBef(state, dip->pT2))
    for (unordered_map<string,double>::iterator it=full.begin();
    it != full.end(); ++it) it->second *= KERNEL_HEADROOM;

  // For small values of pT, recombine with Q2QG kernel to avoid large
  // numerical cancellations:
  // - Set Q2GQ kernel to zero, add to Q2QG
  // - Set G2GG2 kernel to zero, add to G2GG1
  // - Set G2QQ2 kernel to zero, add to G2QQ1
  if ( max(tOld, pT2colCut) < pT2recombine ) {
    if ( splitNow->is(splittingsPtr->fsrQCD_1_to_21_and_1)
      || splitNow->is(splittingsPtr->fsrQCD_21_to_21_and_21b)
      || splitNow->is(splittingsPtr->fsrQCD_21_to_1_and_1b))
      for (unordered_map<string,double>::iterator it=full.begin();
           it != full.end(); ++it)
       it->second = 0.;
    string name_recombine="";
    if (splitNow->is(splittingsPtr->fsrQCD_1_to_1_and_21))
      name_recombine="Dire_fsr_qcd_1->21&1";
    if (splitNow->is(splittingsPtr->fsrQCD_21_to_21_and_21a))
      name_recombine="Dire_fsr_qcd_21->21&21b";
    if (splitNow->is(splittingsPtr->fsrQCD_21_to_1_and_1a))
      name_recombine="Dire_fsr_qcd_21->1&1b";
    // Recombine with other kernels.
    if (name_recombine != "" && splits.find(name_recombine) != splits.end() ) {
      splits[name_recombine]->splitInfo.storeRadBef(state[dip->iRadiator]);
      splits[name_recombine]->splitInfo.storeRecBef(state[dip->iRecoiler]);
      splits[name_recombine]->splitInfo.storeInfo(name_recombine, type,
        dip->system, dip->systemRec, 0, dip->iRadiator,
        dip->iRecoiler, state, dip->flavour, idDaughter, nEmissions, Q2,
        dip->pT2, dip->pT2Old, dip->z,
        dip->phi, m2Bef, m2s, (nEmissions == 1 ? m2r : m2a),
        (nEmissions == 1 ? m2e : m2i), dip->sa1, dip->xa, dip->phia1, m2r,
        xBef, -1.);
      splits[name_recombine]->splitInfo.setRadAft(idDaughter);
      splits[name_recombine]->splitInfo.setEmtAft(idSister);
      splits[name_recombine]->splitInfo.setSiblings(dip->iSiblings);

      // Calculate other kernel and add to previous result.
      unordered_map<string,double> full_recombine;
      if (splits[name_recombine]->calc( state, order) )
        full_recombine = splits[name_recombine]->getKernelVals();
      for ( unordered_map<string,double>::iterator it = full_recombine.begin();
        it != full_recombine.end(); ++it ) full[it->first] += it->second;
    }
  }

  direInfoPtr->message(1) << __FILE__ << " " << __func__
    << " " << __LINE__ << " : New splitting "
    << setw(15) << name << " at pT="
    << setw(15) << sqrt(dip->pT2) << " z = "
    << setw(15) << dip->z << " prob = "
    << setw(15) << full["base"] << endl;

  // Reweight with coupling factor if necessary.
  double coupl = splitNow->coupling(dip->z, dip->pT2, Q2, -1.,
      make_pair(state[dip->iRadiator].id(), state[dip->iRadiator].isFinal()),
      make_pair(state[dip->iRecoiler].id(), state[dip->iRecoiler].isFinal()));
  // Retrieve argument of alphaS.
  double scale2 = splits[splittingNowName]->couplingScale2(
    dip->z, dip->pT2, Q2,
    make_pair(state[dip->iRadiator].id(), state[dip->iRadiator].isFinal()),
    make_pair(state[dip->iRecoiler].id(), state[dip->iRecoiler].isFinal()));
  if (scale2 < 0.) scale2 = dip->pT2;
  double talpha           = max(scale2, pT2colCut);
  double renormMultFacNow = renormMultFac;
  if (forceFixedAs) renormMultFacNow = 1.0;

  if (coupl > 0.) {
    full["base"] *= coupl / alphasNow(talpha, renormMultFacNow, dip->system);
    if (name.find("qcd") == string::npos) {
      for ( unordered_map<string,double>::iterator it = full.begin();
        it != full.end(); ++it ) {
        if (it->first == "base") continue;
        it->second *= coupl / alphasNow(talpha, renormMultFacNow, dip->system);
      }
    }
  }

  vector <int> in, out;
  for (int i=0; i < state.size(); ++i) {
    if (i == dip->iRadiator) continue;
    if (state[i].isFinal()) out.push_back(state[i].id());
    if (state[i].mother1() == 1 && state[i].mother2() == 0)
      in.push_back(state[i].id());
    if (state[i].mother1() == 2 && state[i].mother2() == 0)
      in.push_back(state[i].id());
  }
  out.push_back(re[0]);
  for (size_t i=1; i < re.size(); ++i) out.push_back(re[i]);
  bool hasME = dip->pT2 > pT2minMECs
    && doMEcorrections && weights->hasME(in,out);
  if (hasME) for (unordered_map<string,double>::iterator it=full.begin();
    it != full.end(); ++it) it->second = abs(it->second);

  double mecover=1.;
  int nFinal = 0;
  for (int i=0; i < state.size(); ++i) if (state[i].isFinal()) nFinal++;
  if (!dryrun) mecover = splitNow->overhead
                 (dip->m2Dip*xOld, state[dip->iRadiator].id(), nFinal);
  for (unordered_map<string,double>::iterator it=full.begin();
    it != full.end(); ++it) it->second *= mecover;
  over *= mecover;

  // Acceptance weight.
  wt          = full["base"]/over;

  // Divide out artificial enhancements.
  double headRoom = overheadFactors(dip, state, name, dip->m2Dip, tOld, xOld);
  wt   /= headRoom;
  over *= headRoom;

  // Ensure positive weight.
  wt = abs(wt);

}

//--------------------------------------------------------------------------

pair<bool, pair<double,double> >  DireTimes::getMEC ( const Event& state,
  DireSplitInfo* splitInfo) {

  double MECnum(1.0), MECden(1.0);
  //bool hasME = weights->hasME(makeHardEvent(0,state,true));
  bool hasME
    = weights->hasME(makeHardEvent(max(0,splitInfo->system), state, false));

  if (hasME) {

    // Store previous mergingHooks setup.
    mergingHooksPtr->init();

    // For now, prefer construction of ordered histories.
    mergingHooksPtr->orderHistories(false);
    // For pp > h, allow cut on state, so that underlying processes
    // can be clustered to gg > h
    if ( mergingHooksPtr->getProcessString().compare("pp>h") == 0)
      mergingHooksPtr->allowCutOnRecState(true);

    // Prepare process record for merging. If Pythia has already decayed
    // resonances used to define the hard process, remove resonance decay
    // products.
    Event newProcess( mergingHooksPtr->bareEvent(
      makeHardEvent(max(0,splitInfo->system), state, false), true) );
    // Store candidates for the splitting.
    mergingHooksPtr->storeHardProcessCandidates( newProcess );

    // Calculate number of clustering steps
    int nSteps = mergingHooksPtr->
      getNumberOfClusteringSteps( newProcess, true);

    // Set dummy process scale.
    newProcess.scale(0.0);
    // Generate all histories
    DireHistory myHistory( nSteps, 0.0, newProcess, DireClustering(),
      mergingHooksPtr, (*beamAPtr), (*beamBPtr), particleDataPtr, infoPtr,
      nullptr, splits.begin()->second->fsr, splits.begin()->second->isr,
      weights, coupSMPtr, true, true, 1.0, 1.0, 1.0, 1.0, 0);
    // Project histories onto desired branches, e.g. only ordered paths.
    myHistory.projectOntoDesiredHistories();

    MECnum = myHistory.MECnum;
    MECden = myHistory.MECden;

    // Restore to previous mergingHooks setup.
    mergingHooksPtr->init();

  // Done.
  }

  if (abs(MECden) < 1e-15) direInfoPtr->message(1)
    << __FILE__ << " " << __func__
    << " " << __LINE__ << " : Small MEC denominator="
    << MECden << " for numerator=" << MECnum << endl;
  if (abs(MECnum/MECden) > 1e2) { direInfoPtr->message(1)
    << __FILE__ << " " << __func__
    << " " << __LINE__ << " : Large MEC. Denominator="
    << MECden << " Numerator=" << MECnum << " at pT="
    << sqrt(splitInfo->kinematics()->pT2) << " "
    << endl;
  }

  return make_pair(hasME, make_pair(MECnum,MECden));

}

//--------------------------------------------------------------------------

bool DireTimes::applyMEC ( const Event& state, DireSplitInfo* splitInfo,
  vector<Event> auxState) {

  // Get value of ME correction.
  pair<bool, pair<double, double> > mec = getMEC ( state, splitInfo);
  bool hasME    = mec.first;
  double MECnum = mec.second.first;
  double MECden = mec.second.second;
  double MECnumX = mec.second.first;
  double MECdenX = mec.second.second;

  if (!hasME) return false;

  if (abs(MECnum/MECden) > 5e0 && auxState.size()>0) {
    pair<bool, pair<double, double> > mec1 = getMEC ( auxState[0], splitInfo);
    pair<bool, pair<double, double> > mec2 = getMEC ( auxState[1], splitInfo);
    double MECnum1 = mec1.second.first;
    double MECden1 = mec1.second.second;
    double MECnum2 = mec2.second.first;
    double MECden2 = mec2.second.second;
    if (MECnum/MECden > MECnum1/MECden1) {MECnum = MECnum1; MECden = MECden1;}
    if (MECnum/MECden > MECnum2/MECden2) {MECnum = MECnum2; MECden = MECden2;}
    direInfoPtr->message(1) << __FILE__ << " " << __func__
    << " " << __LINE__ << " : Large MEC weight=" << MECnumX/MECdenX
    << " " << MECnum/MECden
    << "\t\t" << splitInfo->kinematics()->pT2/splitInfo->kinematics()->m2Dip
    << " " << splitInfo->kinematics()->z << endl;
    if (MECnum/MECden > (MECnum+MECnum1)/(MECden+MECden1))
      { MECnum += MECnum1; MECden += MECden1; }
    if (MECnum/MECden > (MECnum+MECnum2)/(MECden+MECden2))
      { MECnum += MECnum2; MECden += MECden2; }
  }

  // Remember O(alphaS^2) term and remove from list of variations.
  double kernel = kernelSel["base"];
  bool reject    = false;

  // Remember O(alphaS^2) term and remove from list of variations.
  double oas2    = 0.;
  if (kernelSel.find("base_order_as2") != kernelSel.end() ) {
    oas2 = kernelSel["base_order_as2"];
    kernelSel.erase(kernelSel.find("base_order_as2"));
  }
  double baseNew = ((kernel - oas2) * MECnum/MECden + oas2);

  // Now check if the splitting should be vetoed/accepted given new kernel.
  //double auxNew  = overheadFactorsMEC(state, splitInfo, splittingSelName)
  //               * kernel;
  double auxNew  = kernel;
  double overNew = kernel;

  int nFinal = 0;
  for (int i=0; i < state.size(); ++i)
    if (state[i].isFinal()) nFinal++;

  if (dryrun) splittingSel->storeOverhead(
    splitInfo->kinematics()->m2Dip*splitInfo->kinematics()->xBef,
    splitInfo->kinematics()->xBef, state[splitInfo->iRadBef].id(), nFinal-1,
    max(baseNew/overNew,1.1));

  // Ensure that accept probability is positive.
  if (baseNew/auxNew < 0.) auxNew *= -1.;
  if (suppressLargeMECs)  while (baseNew/auxNew < 5e-2) auxNew /= 5.;

  // Reset overestimate if necessary.
  if (baseNew/auxNew > 1.) {
    double rescale = baseNew/auxNew * 1.5;
    auxNew *= rescale;
  }
  double wt = baseNew/auxNew;

  // New rejection weight.
  double wvNow = auxNew/overNew * (overNew - baseNew)
    / (auxNew  - baseNew);

  // New acceptance weight.
  double waNow = auxNew/overNew;
  if (wt < rndmPtr->flat()) {

    if (abs(wvNow) > 1e0) {
      direInfoPtr->message(1) << __FILE__ << " " << __func__
      << " " << __LINE__ << " : Large reject weight=" << wvNow
      << "\t for kernel=" << baseNew << " overestimate=" << overNew
      << "\t aux. overestimate=" << auxNew << " at pT2="
      << splitInfo->kinematics()->pT2
      <<  " for " << splittingSelName << endl;
    }

    // Loop through and reset weights.
    for (unordered_map<string,double>::iterator it= kernelSel.begin();
         it != kernelSel.end(); ++it) {
      // Get old accept weight.
      double waOld = weights->getAcceptWeight( splitInfo->kinematics()->pT2,
                                               it->first);
      // Remove previous acceptance weight and replace rejection weight.
      weights->eraseAcceptWeight(splitInfo->kinematics()->pT2, it->first);
      weights->resetRejectWeight(splitInfo->kinematics()->pT2, wvNow*waOld,
                                 it->first);
    }
    reject = true;
  } else {

    if (abs(waNow) > 1e0) {
      direInfoPtr->message(1) << __FILE__ << " " << __func__
      << " " << __LINE__ << " : Large accept weight=" << waNow
      << "\t for kernel=" << baseNew << " overestimate=" << overNew
      << "\t aux. overestimate=" << auxNew << " at pT2="
      << splitInfo->kinematics()->pT2
      << " for " << splittingSelName << endl;
    }

    // Loop through and reset weights.
    for (unordered_map<string,double>::iterator it= kernelSel.begin();
         it != kernelSel.end(); ++it) {
      // Get old accept weight.
      double waOld = weights->getAcceptWeight( splitInfo->kinematics()->pT2,
                                               it->first);
      // Remove previous reject weight and replace accept weight.
      weights->eraseRejectWeight(splitInfo->kinematics()->pT2, it->first);
      weights->resetAcceptWeight(splitInfo->kinematics()->pT2, waNow*waOld,
                                 it->first);
    }
  }

  // Done.
  return reject;

}

//--------------------------------------------------------------------------

// Check if variables are in allowed phase space.
// Note that the vector of auxiliary inputs "aux" (needed to check the phase
// space of 1->3 splittings) has the following ordering:
// +2.pRadBef*pRecBef, (pRadBef +- pRecBef)^2, pT2, sa1, za, xa, m_{0,a12)^2,
// m_{0,a}^2, m_{0,1}^2, m_{0,2}^2, m_{0,b}^2

bool DireTimes::inAllowedPhasespace( int kinType, double z, double pT2,
  double m2dip, double q2, double xOld, int splitType, double m2RadBef,
  double m2r, double m2s, double m2e, vector<double> aux) {

  // Simple (massive) 1->2 decay.
  if (splitType == 0) {

    double zCS = z;
    double yCS = (m2RadBef - m2e - m2r)
               / (m2RadBef - m2e - m2r + q2 - m2RadBef - m2s);
    // Calculate derived variables.
    double sij  = yCS * (q2 - m2s) + (1.-yCS)*(m2r+m2e);
    double zbar = (q2-sij-m2s) / bABC(q2,sij,m2s)
               * (zCS - m2s/gABC(q2,sij,m2s)
                       *(sij + m2r - m2e)/(q2-sij-m2s));
    double kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2r - zbar*m2e;
    if (kT2 < 0.0) return false;

  // splitType == 1 -> Massless FF
  } else if (splitType == 1) {

    // Calculate CS variables.
    double yCS = pT2/m2dip / (1.-z);
    double zCS = ( 1. - z - pT2/m2dip - pow2(1.-z) )
               / ( 1. - z - pT2/m2dip);

    // CS variables directly.
    if (kinType == 2) {
      zCS = z;
      yCS = pT2 / (m2dip*z*(1.-z)) ;
    }

    // Forbidden emission if outside allowed z range for given pT2.
    if ( zCS < 0. || zCS > 1. || yCS < 0. || yCS > 1.) return false;

  // splitType == 2 -> Massive FF
  } else if (splitType == 2 && aux.size() == 0) {

    // Phase space limits - CS style.
    // Calculate CS variables.
    double yCS = pT2/m2dip / (1.-z);
    double zCS = ( 1. - z - pT2/m2dip - pow2(1.-z) )
               / ( 1. - z - pT2/m2dip);

    // Evolution in CS variables directly.
    if (kinType == 2) {
      zCS = z;
      yCS = pT2 / (m2dip*z*(1.-z)) ;
    }

    // Calculate derived variables.
    double sij  = yCS * (q2 - m2s) + (1.-yCS)*(m2r+m2e);
    double zbar = (q2-sij-m2s) / bABC(q2,sij,m2s)
                * (zCS - m2s/gABC(q2,sij,m2s)
                        *(sij + m2r - m2e)/(q2-sij-m2s));
    double kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2r - zbar*m2e;

    // Not possible to construct kinematics if kT2 < 0.0
    if (kT2 < 0. || isnan(kT2)) return false;

    // Get yCS-boundaries.
    double mu2Rad = m2r/q2;
    double mu2Emt = m2e/q2;
    double mu2Rec = m2s/q2;
    double yCSminMassive = 2.*sqrt(mu2Rad)*sqrt(mu2Emt)
                         / ( 1 - mu2Rad - mu2Emt - mu2Rec);
    double yCSmaxMassive = 1.
                         - 2.*sqrt(mu2Rec)*( 1 - sqrt(mu2Rec) )
                         / ( 1 - mu2Rad - mu2Emt - mu2Rec);

    // Forbidden emission if outside allowed y range for given pT2.
    if ( yCS < yCSminMassive || yCS > yCSmaxMassive) return false;

    // Get zCS-boundaries.
    double nu2Rad = m2r/m2dip;
    double nu2Emt = m2e/m2dip;
    double nu2Rec = m2s/m2dip;
    double vijk   = pow2(1.-yCS) - 4.*(yCS + nu2Rad + nu2Emt)*nu2Rec;
    double viji   = pow2(yCS) - 4.*nu2Rad*nu2Emt;
    if (vijk < 0. || viji < 0.) return false;
    vijk          = sqrt(vijk) / (1-yCS);
    viji          = sqrt(viji) / (yCS + 2.*nu2Rad);
    double prefac = (m2dip*yCS + 2.*m2r) / (2.*m2dip*yCS + 2.*m2r + 2.*m2e);
    double zCSminMassive = ( 1 - vijk*viji) * prefac;
    double zCSmaxMassive = ( 1 + vijk*viji) * prefac;

    // Forbidden emission if outside allowed z range for given pT2.
    if ( zCS < zCSminMassive || zCS > zCSmaxMassive) return false;

  // splitType == 2 -> Massive 1->3 FF
  } else if (splitType == 2 && aux.size() > 0) {

    // Not correctly set up!
    if ( int(aux.size()) < 11) return false;

    //double Q2     = aux[0];
    double q2_1   = aux[1];
    double t      = aux[2];
    double sai    = aux[3];
    double za     = aux[4];
    double xa     = aux[5];
    double m2aij  = aux[6];
    double m2a    = aux[7];
    double m2i    = aux[8];
    double m2j    = aux[9];
    double m2k    = aux[10];
    double m2ai   = sai + m2a + m2i;

    // Calculate CS variables from 1->3 variables
    double yCS = t / (q2_1 - m2ai - m2j - m2k) * xa / za;
    double zCS = za / (xa *(1. - yCS))
                * (q2_1 - m2aij - m2k) / (q2_1 - m2ai - m2j - m2k);

    // Calculate derived variables.
    double sij  = yCS * (q2_1 - m2k) + (1.-yCS)*(m2ai+m2j);
    double zbar = (q2_1-sij-m2k) / bABC(q2_1,sij,m2k)
                * (zCS - m2k/gABC(q2_1,sij,m2k)
                        *(sij + m2ai - m2j)/(q2_1-sij-m2k));
    double kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2ai - zbar*m2j;

    // Not possible to construct kinematics if kT2 < 0.0
    if (kT2 < 0. || isnan(kT2)) return false;

    // Get yCS-boundaries.
    double mu2Rad = m2ai/q2_1;
    double mu2Emt = m2j/q2_1;
    double mu2Rec = m2k/q2_1;
    double yCSminMassive = 2.*sqrt(mu2Rad)*sqrt(mu2Emt)
                         / ( 1 - mu2Rad - mu2Emt - mu2Rec);
    double yCSmaxMassive = 1.
                         - 2.*sqrt(mu2Rec)*( 1 - sqrt(mu2Rec) )
                         / ( 1 - mu2Rad - mu2Emt - mu2Rec);

    // Forbidden emission if outside allowed y range for given pT2.
    if ( yCS < yCSminMassive || yCS > yCSmaxMassive) return false;

    // Get zCS-boundaries.
    double nu2Rad = m2ai/(q2_1 - m2ai - m2j - m2k + m2aij + m2k);
    double nu2Emt = m2j/(q2_1 - m2ai - m2j - m2k + m2aij + m2k);
    double nu2Rec = m2k/(q2_1 - m2ai - m2j - m2k + m2aij + m2k);
    double vijk   = pow2(1.-yCS) - 4.*(yCS + nu2Rad + nu2Emt)*nu2Rec;
    double viji   = pow2(yCS) - 4.*nu2Rad*nu2Emt;
    if (vijk < 0. || viji < 0.) return false;
    vijk          = sqrt(vijk) / (1-yCS);
    viji          = sqrt(viji) / (yCS + 2.*nu2Rad);
    double prefac = ((q2_1 - m2ai - m2j - m2k + m2aij + m2k)*yCS + 2.*m2ai)
                   / (2.*(q2_1 - m2ai - m2j - m2k + m2aij + m2k)*yCS
                    + 2.*m2ai + 2.*m2j);
    double zCSminMassive = ( 1 - vijk*viji) * prefac;
    double zCSmaxMassive = ( 1 + vijk*viji) * prefac;

    // Forbidden emission if outside allowed z range for given pT2.
    if ( zCS < zCSminMassive || zCS > zCSmaxMassive) return false;

    // Calculate CS variables.
    double q2_2   = za/xa*(q2_1 - m2aij - m2k) + m2ai + m2k;
    double zCS_2  = xa;
    double yCS_2  = (m2ai - m2a - m2i)
                  / (m2ai - m2a - m2i + q2_2 - m2ai - m2k);

    // Calculate derived variables.
    sij  = yCS_2 * (q2_2 - m2k) + (1.-yCS_2)*(m2a+m2i);
    zbar = (q2_2-sij-m2k) / bABC(q2_2,sij,m2k)
               * (zCS_2 - m2k/gABC(q2_2,sij,m2k)
                       *(sij + m2a - m2i)/(q2_2-sij-m2k));
    kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2a - zbar*m2i;

    if (kT2 < 0. || isnan(kT2)) return false;

  // Extremely conservative technical cut-off on z for final-final splittings.
  } else if (splitType == 3) {

    // pT2 can't be larger than the invariant mass of the dipole.
    if (pT2 > m2dip) return false;

    double kappa2 = pow2(pTcolCutMin/10.)
                  / pow2(0.5*(beamAPtr->e() + beamBPtr->e()));
    // Minimal cut on energy fraction for final-final.
    double yCS =  kappa2 / (1.-z);
    double zCS = ( 1. - z - kappa2 - pow2(1.-z) )
               / ( 1. - z - kappa2);
    if ( zCS < 0. || zCS > 1. || yCS < 0. || yCS > 1.) return false;

  // splitType ==-1 -> Massless FI
  } else if (splitType ==-1) {

    // Calculate CS variables.
    double kappa2 =  pT2/m2dip;
    double zCS = z;
    double xCS = 1 - kappa2/(1.-z);

    // CS variables directly.
    if (kinType == 2) {
      zCS = z;
      xCS = m2dip*zCS*(1.-zCS) / ( pT2 + m2dip*zCS*(1.-zCS) ) ;
    }

    // Forbidden emission if outside allowed z range for given pT2.
    if ( zCS < 0. || zCS > 1. || xCS < xOld || xCS > 1.) return false;

  // splitType ==-2 -> Massive FI
  } else if (splitType == -2 && aux.size() == 0) {

    // Phase space limits - CS style
    // Calculate CS variables.
    double kappa2 =  pT2/m2dip;
    double zCS = z;
    double xCS = 1 - kappa2/(1.-z);

    // CS variables directly.
    if (kinType == 2) {
      zCS = z;
      xCS = m2dip*zCS*(1.-zCS) / ( pT2 + m2dip*zCS*(1.-zCS) ) ;
    }

    // Get xCS-boundaries.
    double xCDST = xCS / m2dip * (m2dip + m2RadBef-m2r-m2e);
    double pijpa_tilde = m2dip - m2r - m2e + m2RadBef;
    double pijpa     = pijpa_tilde/xCDST;
    double mu2RadBef = m2RadBef/pijpa;
    double muRad     = sqrt(m2r/pijpa);
    double muEmt     = sqrt(m2e/pijpa);
    double xCSmaxMassive = 1. + mu2RadBef - pow2(muRad + muEmt);

    // Forbidden emission if outside allowed x range for given pT2.
    if ( xCDST < xOld || xCDST > xCSmaxMassive) return false;

    // Get zCS-boundaries.
    double nu2Rad = m2r/m2dip;
    double nu2Emt = m2e/m2dip;
    double viji   = pow2(1.-xCS) - 4. * xCS*nu2Rad * xCS*nu2Emt;
    if (viji < 0.) return false;
    viji   = sqrt( viji ) / (1.-xCS+2.*nu2Rad*xCS);
    double vijk   = 1.;
    double prefac = 0.5 * ( 1.-xCS + 2.*xCS*nu2Rad )
                  / ( 1.-xCS + xCS*nu2Rad + xCS*nu2Emt);
    double zCSminMassive = prefac * ( 1. - viji*vijk );
    double zCSmaxMassive = prefac * ( 1. + viji*vijk );

    // Forbidden emission if outside allowed z range for given pT2.
    if ( zCS < zCSminMassive || zCS > zCSmaxMassive) return false;

  // splitType ==-2 -> Massive 1->3 FI
  } else if (splitType == -2 && aux.size() > 0) {

    // Not correctly set up!
    if ( int(aux.size()) < 11) return false;

    //double Q2     = aux[0];
    double q2_1   = aux[1];
    double t      = aux[2];
    double sai    = aux[3];
    double za     = aux[4];
    double xa     = aux[5];
    double m2aij  = aux[6];
    double m2a    = aux[7];
    double m2i    = aux[8];
    double m2j    = aux[9];
    double m2k    = aux[10];
    double m2ai   = sai + m2a + m2i;

    swap(m2ai,m2j);

    // Calculate CS variables from 1->3 variables
    double zCS = za / xa;
    double xCS = (q2_1 - m2ai - m2j - m2k)
               / (q2_1 - m2ai - m2j - m2k - t * xa/za );

    // Get xCS-boundaries.
    double xCDST = xCS * ( 1. - (m2aij-m2ai-m2j)/ (q2_1-m2ai-m2j-m2k) );

    //double pijpa_tilde = Q2 - m2j - m2ai + m2aij;
    double pijpa_tilde = -q2_1 + m2aij + m2k;
    double pijpa     = pijpa_tilde/xCDST;
    double mu2RadBef = m2aij/pijpa;
    double muRad     = sqrt(m2j/pijpa);
    double muEmt     = sqrt(m2ai/pijpa);
    double xCSmaxMassive = 1. + mu2RadBef - pow2(muRad + muEmt);

    // Forbidden emission if outside allowed x range for given pT2.
    if ( xCDST < xOld || xCDST > xCSmaxMassive) return false;

    // Get zCS-boundaries.
    double root = pow2(1. - xCDST + mu2RadBef - muRad*muRad - muEmt*muEmt)
      - 4.*pow2(muRad*muEmt);
    if (root < 0.) return false;
    double zCSminMassive = (1. - xCDST + mu2RadBef + muRad*muRad - muEmt*muEmt
      - sqrt(root)) / ( 2.*(1. - xCDST + mu2RadBef) );
    double zCSmaxMassive = (1. - xCDST + mu2RadBef + muRad*muRad - muEmt*muEmt
      + sqrt(root)) / ( 2.*(1. - xCDST + mu2RadBef) );

    // Forbidden emission if outside allowed z range for given pT2.
    if ( zCS < zCSminMassive || zCS > zCSmaxMassive) return false;

    // Check validity of second, FF-like step.
    swap(m2ai,m2j);

    // Calculate CS variables.
    double q2_2 = m2ai + m2k - za/xa * ( q2_1 - m2k - m2ai - m2j - t*xa/za);
    double yCS  = (m2ai - m2a - m2i) / (m2ai - m2a - m2i + q2_2 - m2ai - m2k);
    zCS         = xa;
    // Calculate derived variables.
    double sij  = yCS * (q2_2 - m2k) + (1.-yCS)*(m2a+m2i);
    double zbar = (q2_2-sij-m2k) / bABC(q2_2,sij,m2k)
                * (zCS - m2k/gABC(q2_2,sij,m2k)
                       *(sij + m2a - m2i)/(q2_2-sij-m2k));
    double kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2a - zbar*m2i;

    // Not possible to construct second step if kT2 < 0.0
    if (kT2 < 0. || isnan(kT2)) return false;

  // Extremely conservative technical z-cut-off for final-initial splittings.
  } else if (splitType ==-3) {

    // pT2 can't be larger than the invariant mass of the dipole.
    if (pT2 > m2dip) return false;

    double kappa2 = pow2(pTcolCutMin/10.)
                  / pow2(0.5*(beamAPtr->e() + beamBPtr->e()));
    // Minimal cut on energy fraction for final-initial.
    double zCS = z;
    double xCS = 1 - kappa2/(1.-z);

    if ( zCS < 0. || zCS > 1. || xCS < xOld/1000. || xCS > 1.) return false;

  }

  return true;

}

//--------------------------------------------------------------------------

// Function to add user-defined overestimates to old overestimate.

void DireTimes::addNewOverestimates( multimap<double,string> newOverestimates,
  double& oldOverestimate) {

  // No other tricks necessary at the moment.
  if (!newOverestimates.empty())
    oldOverestimate += newOverestimates.rbegin()->first;

  // Done.

}

//--------------------------------------------------------------------------

// Function to attach the correct alphaS weights to the kernels.

void DireTimes::alphasReweight(double, double talpha, int iSys,
  bool forceFixedAs, double& weight, double& fullWeight, double& overWeight,
  double renormMultFacNow) {

  if (forceFixedAs) renormMultFacNow = 1.0;
  talpha = max(talpha, pT2colCut);

  double scale       = talpha*renormMultFacNow;

  // Save-guard against scales below shower cut-off
  scale = max(scale, pT2colCut);

  // Get current alphaS value.
  double asPT2piCorr  = alphasNow(talpha, renormMultFacNow, iSys);

  // Begin with multiplying alphaS to overestimate.
  double asOver = 1.;
  if (usePDFalphas)        asOver = alphaS2piOverestimate;
  else if (alphaSorder==0) asOver = alphaS2pi;
  else                     asOver = alphaS.alphaS(scale) / (2.*M_PI);
  // Multiply alphaS to full splitting kernel.
  double asFull = 1.;
  if (alphaSorder == 0)    asFull = alphaS2pi;
  else                     asFull = asPT2piCorr;

  fullWeight *= asFull;
  overWeight *= asOver;
  weight     *= asFull/asOver;

  // Done.

}

//--------------------------------------------------------------------------

// Evolve a QCD dipole end in pT2.

void DireTimes::pT2nextQCD(double pT2begDip, double pT2sel,
  DireTimesEnd& dip, Event& event, double pT2endForce, double pT2freeze,
  bool forceBranching) {

  if (event[dip.iRecoiler].isFinal())
    pT2nextQCD_FF(pT2begDip, pT2sel, dip, event, pT2endForce, pT2freeze,
    forceBranching);
  else
    pT2nextQCD_FI(pT2begDip, pT2sel, dip, event, pT2endForce, pT2freeze,
    forceBranching);

  // Done.
  return;

}

//--------------------------------------------------------------------------

// Evolve a QCD dipole end.

bool DireTimes::pT2nextQCD_FF(double pT2begDip, double pT2sel,
  DireTimesEnd& dip, const Event& event, double pT2endForce, double pT2freeze,
  bool forceBranching) {

  // Lower cut for evolution. Return if no evolution range.
  double pT2endDip = max( pT2sel, pT2cutMin(&dip));
  if (pT2endForce >= 0.) pT2endDip = pT2endForce;
  if (pT2begDip < pT2endDip) return false;

  // Variables used inside evolution loop. (Mainly dummy start values.)
  dip.pT2              = pT2begDip;
  double zMinAbs       = 0.0;
  double zMaxAbs       = 1.0;
  double teval         = pT2begDip;
  double Lambda2       = Lambda3flav2;
  double emitCoefTot   = 0.;
  double wt            = 0.;
  bool   mustFindRange = true;

  int idRadiator = event[dip.iRadiator].id();
  multimap<double,string> newOverestimates;

  unordered_map<string,double> fullWeightsNow;
  int    nContinue(0), nContinueMax(10000);
  double fullWeightNow(0.), overWeightNow(0.), auxWeightNow(0.), daux(0.);

  // Begin evolution loop towards smaller pT values.
  do {

    wt          = 0.;
    double tnow = (!forceBranching) ? dip.pT2 : pT2begDip;
    // Reset emission properties.
    dip.z        = -1.0;
    dip.phi      = -1.0;
    dip.sa1      =  0.0;
    dip.xa       = -1.0;
    dip.phia1    = -1.0;
    dip.idRadAft = 0;
    dip.idEmtAft = 0;
    dip.mass.clear();

    // Force exit if non-Sudakov style forced branching is stuck.
    if (forceBranching && nContinue >= nContinueMax) {
      wt = 0.0; dip.pT2 = tnow = 0.;
      break;
    }

    // Update event weight after one step.
    if ( fullWeightNow != 0. && overWeightNow != 0. ) {
      double enhanceFurther
        = enhanceOverestimateFurther(splittingNowName, idRadiator, teval);
      if (doTrialNow) enhanceFurther = 1.;
      kernelNow = fullWeightsNow;
      auxNow    = auxWeightNow;
      overNow   = overWeightNow;
      boostNow  = enhanceFurther;
      for ( unordered_map<string,double>::iterator it = fullWeightsNow.begin();
        it != fullWeightsNow.end(); ++it ) {

        // No weight bookkeeping for non-variation components of kernel vector.
        if (it->first == "base_order_as2") continue;

        double wv = auxWeightNow/overWeightNow
                 * (overWeightNow- it->second/enhanceFurther)
                 / (auxWeightNow - fullWeightNow);
        if (abs(wv) > 1e0) {
        direInfoPtr->message(1) << __FILE__ << " " << __func__
        << " " << __LINE__ << " : Large reject weight=" << wv
        << "\t for kernel=" << it->second << " overestimate=" << overNow
        << "\t aux. overestimate=" << auxNow << " enhance=" << enhanceFurther
        << " at pT2=" << tnow
        <<  " for " << splittingNowName << " " << it->first << endl;

        }
        rejectProbability[it->first].insert( make_pair(tnow,wv));
      }
    }

    splittingNowName ="";
    fullWeightsNow.clear();
    fullWeightNow = overWeightNow = auxWeightNow = 0.;

    if (mustFindRange) {

      newOverestimates.clear();
      teval       = tnow;
      emitCoefTot = 0.;

      // Determine overestimated z range; switch at c and b masses.
      if (tnow > m2b) {
        Lambda2  = Lambda5flav2;
      } else if (tnow > m2c) {
        Lambda2  = Lambda4flav2;
      } else {
        Lambda2  = Lambda3flav2;
      }
      // A change of renormalization scale expressed by a change of Lambda.
      Lambda2 /= renormMultFac;

      // Calculate and add user-defined overestimates.
      getNewOverestimates( &dip, event, tnow, 1., zMinAbs, zMaxAbs,
        newOverestimates);
      addNewOverestimates(newOverestimates, emitCoefTot);

      // Store pT at which overestimate has been evaluated.
      dip.pT2Old = teval;

      // Initialization done for current range.
      mustFindRange = false;
    }

    if (emitCoefTot < TINYOVERESTIMATE) { dip.pT2 = 0.0; return false; }
    if (newOverestimates.empty())       { dip.pT2 = 0.0; return false; }

    // Generate next evolution scale.
    bool forceFixedAs = (tnow < pT2colCut);
    tnow  = tNextQCD( &dip, emitCoefTot, tnow, pT2endDip, pT2freeze,
      (forceBranching ? -1 : 1));
    if (tnow < 0.) {
      wt = 0.0; dip.pT2 = tnow = 0.;
      double R0 = emitCoefTot*rndmPtr->flat();
      if (!newOverestimates.empty()) {
        if (newOverestimates.lower_bound(R0) == newOverestimates.end())
          splittingNowName = newOverestimates.rbegin()->second;
        else
          splittingNowName = newOverestimates.lower_bound(R0)->second;
      }
      break;
    }

    wt = 0.0;
    dip.pT2      = tnow;

    // Abort evolution if below cutoff scale, or below another branching.
    if ( tnow < pT2endDip ) { dip.pT2 = tnow = 0.; break; }

    // Try user-defined splittings first.
    double R = emitCoefTot*rndmPtr->flat();
    double z = -1.;
    int idDaughter, idSister;
    idDaughter = idSister = 0;
    if (!newOverestimates.empty()) {

      // Pick splitting.
      if (newOverestimates.lower_bound(R) == newOverestimates.end())
        splittingNowName = newOverestimates.rbegin()->second;
      else
        splittingNowName = newOverestimates.lower_bound(R)->second;

      // Generate z value and calculate splitting probability.
      getNewSplitting( event, &dip, teval, 0., tnow, zMinAbs,
        zMaxAbs, idRadiator, splittingNowName, forceFixedAs, idDaughter,
        idSister, z, wt, fullWeightsNow, overWeightNow);

      dip.z      = z;
    }

    // Done for vanishing accept probability.
    if ( wt == 0. || z < 0.) {
      //mustFindRange = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    fullWeightNow = fullWeightsNow["base"];

    // Get particle masses.
    // Radiator before splitting.
    double m2Bef = particleDataPtr->isResonance(event[dip.iRadiator].id())
                 ? getMass(event[dip.iRadiator].id(),3,
                           event[dip.iRadiator].mCalc())
                 : (abs(event[dip.iRadiator].id()) < 6
                     || event[dip.iRadiator].id() == 21
                     || event[dip.iRadiator].id() == 22)
                 ? getMass(event[dip.iRadiator].id(),2)
                 : getMass(event[dip.iRadiator].id(),1);
    // Radiator after splitting.
    double m2r   = particleDataPtr->isResonance(idDaughter)
                   && idDaughter == event[dip.iRadiator].id()
                 ? getMass(idDaughter,3,event[dip.iRadiator].mCalc())
                 : (abs(idDaughter) < 6 || idDaughter == 21
                     || idDaughter == 22)
                 ? getMass(idDaughter,2)
                 : getMass(idDaughter,1);
    // Recoiler.
    double m2s   = particleDataPtr->isResonance(event[dip.iRecoiler].id())
                 ? getMass(event[dip.iRecoiler].id(),3,
                           event[dip.iRecoiler].mCalc())
                 : (event[dip.iRecoiler].idAbs() < 6
                 || event[dip.iRecoiler].id() == 21
                 || event[dip.iRecoiler].id() == 22)
                 ? getMass(event[dip.iRecoiler].id(),2)
                 : getMass(event[dip.iRecoiler].id(),1);
    // Emission.
    double m2e = (abs(dip.flavour) < 6 || dip.flavour == 21
                   || dip.flavour == 22)
               ? getMass(dip.flavour,2) : getMass(dip.flavour,1);

    bool canUseSplitInfo = splits[splittingNowName]->canUseForBranching();
    if (canUseSplitInfo) {
      m2r = splits[splittingNowName]->splitInfo.kinematics()->m2RadAft;
      m2e = splits[splittingNowName]->splitInfo.kinematics()->m2EmtAft;
    }
    int nEmissions = splits[splittingNowName]->nEmissions();

    // Recalculate the kinematicaly available dipole mass.
    double Q2 = dip.m2Dip + m2Bef - m2r - m2e;
    double q2 = (event[dip.iRadiator].p() +event[dip.iRecoiler].p()).m2Calc();

    // Discard this 1->3 splitting if the pT has fallen below mEmission (since
    // such splittings would not be included in the virtual corrections to the
    // 1->2 kernels. Note that the threshold is pT>mEmission,since alphaS is
    // evaluated at pT, not virtuality sa1).
    if ( nEmissions == 2
      && ( (abs(dip.flavour) == 4 && tnow < m2cPhys)
        || (abs(dip.flavour) == 5 && tnow < m2bPhys))) {
      mustFindRange = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Pick remaining variables for 1->3 splitting.
    double m2aij(m2Bef), m2a(m2e), m2i(m2e), m2j(m2r), m2k(m2s);
    if (canUseSplitInfo)
      m2j = splits[splittingNowName]->splitInfo.kinematics()->m2EmtAft2;

    double jacobian(1.);
    if (canUseSplitInfo) {
      jacobian = splits[splittingNowName]->getJacobian(event,partonSystemsPtr);
    } else {
      // Calculate CS variables and scaled masses.
      double yCS = tnow/Q2 / (1. - z);
      double mu2RadBef = m2Bef/ q2;
      double mu2Rad    = m2r/ q2;
      double mu2Rec    = m2s/ q2;
      double mu2Emt    = m2e/ q2;
      // Calculate Jacobian.
      double jac1 = ( 1. - mu2Rad - mu2Rec - mu2Emt)
                  / sqrt(lABC(1.,mu2RadBef,mu2Rec));
      double jac2 = 1. + ( mu2Rad + mu2Emt - mu2RadBef)
                        /( yCS*(1. - mu2Rad - mu2Rec - mu2Emt));

      // Jacobian for 1->3 splittings, in CS variables.
      if (nEmissions == 2) {
        double sai   = dip.sa1;
        double m2ai  = sai + m2a + m2i;
        // Jacobian for competing steps, i.e. applied to over-all
        // splitting rate.
        jac1 = (q2 - m2aij - m2k) / sqrt( lABC(q2, m2aij, m2k) );
        // Additional jacobian for non-competing steps.
        double m2aik = (dip.sa1 + m2a + m2i) + m2k
                     +  dip.z/dip.xa * (q2 - m2Bef - m2k);
        jac1 *= (m2aik - m2ai - m2k) / sqrt( lABC(m2aik, m2ai, m2k) );
        // Additional factor from massive propagator.
        jac2 = 1 + (m2ai + m2j - m2aij) / (dip.pT2*dip.xa/dip.z);
      }
      jacobian = jac1/jac2;
    }

    // Multiply with Jacobian.
    fullWeightNow *= jacobian;
    for ( unordered_map<string,double>::iterator it = fullWeightsNow.begin();
      it != fullWeightsNow.end(); ++it )
      it->second   *= jacobian;

    // Before generating kinematics: Reset sai if the kernel fell on an
    // endpoint contribution.
    if ( nEmissions == 2
      && splits[splittingNowName]->splitInfo.kinematics()->sai == 0.)
        dip.sa1 = 0.;

    if (fullWeightNow == 0. ) {
      //mustFindRange = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Retrieve argument of alphaS.
    double scale2 =  splits[splittingNowName]->couplingScale2 ( z, tnow, Q2,
      make_pair (event[dip.iRadiator].id(), event[dip.iRadiator].isFinal()),
      make_pair (event[dip.iRecoiler].id(), event[dip.iRecoiler].isFinal()));
    if (scale2 < 0.) scale2 = tnow;
    double talpha = max(scale2, pT2colCut);

    // Reweight to match PDF alpha_s, including corrective terms for
    // renormalisation scale variation.
    alphasReweight( tnow, talpha, dip.system, forceFixedAs, wt, fullWeightNow,
      overWeightNow, renormMultFac);
    auxWeightNow   = overWeightNow;

    // Create muR-variations.
    double asw = 1.;
    alphasReweight(tnow, talpha, dip.system, forceFixedAs, daux, asw, daux,
      renormMultFac);
    fullWeightsNow["base"] *= asw;
    if (fullWeightsNow.find("base_order_as2") != fullWeightsNow.end())
      fullWeightsNow["base_order_as2"] *= asw;
    if (doVariations) {
      if ( splittingNowName.find("qcd") != string::npos
        && settingsPtr->parm("Variations:muRfsrDown") != 1.) {
        asw = 1.;
        alphasReweight(tnow, talpha, dip.system, forceFixedAs, daux, asw, daux,
          (tnow > pT2minVariations) ? settingsPtr->
                       parm("Variations:muRfsrDown") : renormMultFac);
        fullWeightsNow["Variations:muRfsrDown"] *= asw;
      } else if ( splittingNowName.find("qcd") == string::npos )
        fullWeightsNow["Variations:muRfsrDown"] *= asw;
      if ( splittingNowName.find("qcd") != string::npos
        && settingsPtr->parm("Variations:muRfsrUp")   != 1.) {
        asw = 1.;
        alphasReweight(tnow, talpha, dip.system, forceFixedAs, daux, asw, daux,
          (tnow > pT2minVariations) ? settingsPtr->parm("Variations:muRfsrUp")
          : renormMultFac);
        fullWeightsNow["Variations:muRfsrUp"] *= asw;
      } else if ( splittingNowName.find("qcd") == string::npos )
        fullWeightsNow["Variations:muRfsrUp"] *= asw;
    }

    // Ensure that accept probability is positive.
    if (fullWeightNow < 0.) {
      auxWeightNow *= -1.;
    }

    // Reset overestimate if necessary.
    if (fullWeightNow > 0. && fullWeightNow/auxWeightNow > 1.) {
      direInfoPtr->message(1) << __FILE__ << " " << __func__
        << " " << __LINE__ << " : Large acceptance weight="
        << fullWeightNow/auxWeightNow
        << " for splitting " << splittingNowName << " at pT2=" << tnow
        << " and z=" << z << endl;
      //mustFindRange = true;
      if (fullWeightNow/auxWeightNow > 2.)
        scaleOverheadFactor(splittingNowName, 2.);
      double rescale = fullWeightNow/auxWeightNow * 1.15;
      auxWeightNow *= rescale;
      infoPtr->errorMsg("Info in DireTimes::pT2nextQCD_FF: Found large "
                        "acceptance weight for " + splittingNowName);
    }

    wt = fullWeightNow/auxWeightNow;

  // Iterate until acceptable pT (or have fallen below pTmin).
  } while (wt < rndmPtr->flat());

  // Not possible to find splitting.
  if ( wt == 0.) return false;

  // Update accepted event weight. No weighted shower for first
  // "pseudo-emission" step in 1->3 splitting.
  if ( fullWeightNow != 0. && overWeightNow != 0. ) {
    double enhanceFurther
      = enhanceOverestimateFurther(splittingNowName, idRadiator, teval);
    double tnow = dip.pT2;
    if (doTrialNow) {
      weights->addTrialEnhancement(tnow, enhanceFurther);
      enhanceFurther = 1.;
    }
    kernelNow = fullWeightsNow;
    auxNow    = auxWeightNow;
    overNow   = overWeightNow;
    boostNow  = enhanceFurther;
    for ( unordered_map<string,double>::iterator it = fullWeightsNow.begin();
      it != fullWeightsNow.end(); ++it ) {

      // No weight bookkeeping for non-variation components of kernel vector.
      if (it->first == "base_order_as2") continue;

      acceptProbability[it->first].insert(make_pair(tnow,
        auxWeightNow/overWeightNow * 1./enhanceFurther
        * it->second/fullWeightNow ) );
      if (auxWeightNow == fullWeightNow && overWeightNow == fullWeightNow)
        rejectProbability[it->first].insert( make_pair(tnow, 1.0));
      else {
        double wv  = auxWeightNow/overWeightNow
                  * (overWeightNow- it->second/enhanceFurther)
                  / (auxWeightNow - fullWeightNow);
        if (abs(wv) > 1e0) {
        direInfoPtr->message(1) << __FILE__ << " " << __func__
        << " " << __LINE__ << " : Large reject weight=" << wv
        << "\t for kernel=" << it->second << " " << fullWeightNow
        << " overestimate=" << overNow
        << "\t aux. overestimate=" << auxNow  << " enhance=" << enhanceFurther
        << " at pT2="
        << tnow
        <<  " for " << splittingNowName << " " << it->first << endl;
        }
        rejectProbability[it->first].insert( make_pair(tnow, wv));
      }
    }
  }

  // Done
  return true;

}

//--------------------------------------------------------------------------

// Evolve a QCD dipole end.

bool DireTimes::pT2nextQCD_FI(double pT2begDip, double pT2sel,
  DireTimesEnd& dip, const Event& event, double pT2endForce, double pT2freeze,
  bool forceBranching) {

  // Lower cut for evolution. Return if no evolution range.
  //double pT2endDip = max( pT2sel, pT2colCut);
  double pT2endDip = max( pT2sel, pT2cutMin(&dip));
  if (pT2endForce >= 0.) pT2endDip = pT2endForce;
  if (pT2begDip < pT2endDip) { dip.pT2 = 0.; return false; }

  BeamParticle& beam = (dip.isrType == 1) ? *beamAPtr : *beamBPtr;
  /*double m2cPhys = (usePDFalphas) ? pow2(max(0.,beam.mQuarkPDF(4)))
                 : alphaS.muThres2(4);
  double m2bPhys = (usePDFalphas) ? pow2(max(0.,beam.mQuarkPDF(5)))
                 : alphaS.muThres2(5);*/

  // Variables used inside evolution loop. (Mainly dummy start values.)
  dip.pT2              = pT2begDip;
  int    nFlavour      = 3;
  double zMinAbs       = 0.0;
  double zMaxAbs       = 1.0;
  double teval         = pT2begDip;
  double Lambda2       = Lambda3flav2;
  double xPDFrecoiler  = 0.;
  double emitCoefTot   = 0.;
  double wt            = 0.;
  bool   mustFindRange = true;
  int idRadiator       = event[dip.iRadiator].id();
  int idRecoiler       = event[dip.iRecoiler].id();
  int iSysRec          = dip.systemRec;
  double xRecoiler     = beam[iSysRec].x();
  bool   hasPDFrec     = hasPDF(idRecoiler);

  // Get momentum of other beam, since this might be needed to calculate
  // the Jacobian.
  int iOther = (dip.isrType == 1) ? partonSystemsPtr->getInB(iSysRec)
                                  : partonSystemsPtr->getInA(iSysRec);
  Vec4 pOther(event[iOther].p());

  multimap<double,string> newOverestimates;
  unordered_map<string,double> fullWeightsNow;
  double fullWeightNow(0.), overWeightNow(0.), auxWeightNow(0.), daux(0.);

  // Begin evolution loop towards smaller pT values.
  int    loopTinyPDFdau = 0;
  int    nContinue(0), nContinueMax(10000);
  bool   hasTinyPDFdau  = false;
  do {

    wt          = 0.;
    //double tnow = dip.pT2;
    double tnow = (!forceBranching) ? dip.pT2 : pT2begDip;
    dip.z       = -1.;
    dip.xa      = -1.;
    dip.sa1     = 0.;
    dip.phi     = -1.;
    dip.phia1   = -1.;

    // Force exit if non-Sudakov style forced branching is stuck.
    if (forceBranching && nContinue >= nContinueMax) {
      wt = 0.0; dip.pT2 = tnow = 0.;
      break;
    }

    // Update event weight after one step.
    if ( fullWeightNow != 0. && overWeightNow != 0. ) {
      double enhanceFurther
        = enhanceOverestimateFurther(splittingNowName, idRadiator, teval);
      if (doTrialNow) enhanceFurther = 1.;
      kernelNow = fullWeightsNow;
      auxNow    = auxWeightNow;
      overNow   = overWeightNow;
      boostNow  = enhanceFurther;
      for ( unordered_map<string,double>::iterator it = fullWeightsNow.begin();
        it != fullWeightsNow.end(); ++it ) {

        // No weight bookkeeping for non-variation components of kernel vector.
        if (it->first == "base_order_as2") continue;

        double wv = auxWeightNow/overWeightNow
                 * (overWeightNow- it->second/enhanceFurther)
                 / (auxWeightNow - fullWeightNow);
        if (abs(wv) > 1e0) {
        direInfoPtr->message(1) << __FILE__ << " " << __func__
        << " " << __LINE__ << " : Large reject weight=" << wv
        << "\t for kernel=" << it->second << " overestimate=" << overNow
        << "\t aux. overestimate=" << auxNow << " at pT2="
        << tnow
        <<  " for " << splittingNowName << endl;
        }
        rejectProbability[it->first].insert( make_pair(tnow,wv));
      }
    }

    splittingNowName ="";
    fullWeightsNow.clear();
    fullWeightNow = overWeightNow = auxWeightNow = 0.;

    // Leave unconverted for now.
    if ( event[dip.iRecoiler].idAbs() == 4 && tnow <= m2cPhys) {
      dip.pT2 = 0.; return false;
    }
    if ( event[dip.iRecoiler].idAbs() == 5 && tnow <= m2bPhys) {
      dip.pT2 = 0.; return false;
    }

    // Finish evolution if PDF vanishes.
    double tnew = (useFixedFacScale) ? fixedFacScale2 : factorMultFac*tnow;
    bool inNew  = (hasPDFrec)
                ? beam.insideBounds(xRecoiler, max(tnew, pT2colCut) ) : 1.0;
    if (hasPDFrec && !inNew) { dip.pT2 = 0.0; return false; }

    // Bad sign if repeated looping with small daughter PDF, so fail.
    // (Example: if all PDF's = 0 below Q_0, except for c/b companion.)
    if (hasTinyPDFdau) ++loopTinyPDFdau;
    if (hasPDFrec && loopTinyPDFdau > MAXLOOPTINYPDF) {
      infoPtr->errorMsg("Warning in DireTimes::pT2nextQCD_FI: "
      "small daughter PDF");
      dip.pT2 = 0.0;
      return false;
    }

    // Initialize evolution coefficients at the beginning and
    // reinitialize when crossing c and b flavour thresholds.
    if (mustFindRange
      || tnow < evalpdfstep(event[dip.iRecoiler].id(), tnow, m2cPhys, m2bPhys)*
        teval) {

      newOverestimates.clear();
      teval       = tnow;
      emitCoefTot = 0.;

      // Determine overestimated z range; switch at c and b masses.
      if (tnow > m2b) {
        nFlavour = 5;
        Lambda2  = Lambda5flav2;
      } else if (tnow > m2c) {
        nFlavour = 4;
        Lambda2  = Lambda4flav2;
      } else {
        nFlavour = 3;
        Lambda2  = Lambda3flav2;
      }
      // A change of renormalization scale expressed by a change of Lambda.
      Lambda2 /= renormMultFac;

      // Parton density of daughter at current scale.
      pdfScale2    = (useFixedFacScale) ? fixedFacScale2 : factorMultFac*tnow;
      pdfScale2    = max(pdfScale2, pT2colCut);
      xPDFrecoiler = getXPDF(idRecoiler, xRecoiler, pdfScale2, iSysRec, &beam);
      if ( hasPDFrec && xPDFrecoiler != 0.
        && abs(xPDFrecoiler) < 1e-15) {
        int sign      = (xPDFrecoiler > 0.) ? 1 : -1;
        xPDFrecoiler  = sign*tinypdf(xRecoiler);
        hasTinyPDFdau = true;
      }

      double xMin = (hasPDFrec) ? xRecoiler : 0.;

      // Calculate and add user-defined overestimates.
      getNewOverestimates( &dip, event, tnow, xMin, zMinAbs, zMaxAbs,
        newOverestimates);
      addNewOverestimates(newOverestimates, emitCoefTot);

      // Store pT at which overestimate has been evaluated.
      dip.pT2Old = teval;

      // Initialization done for current range.
      mustFindRange = false;
    }

    if (emitCoefTot < TINYOVERESTIMATE) { dip.pT2 = 0.0; return false; }
    if (newOverestimates.empty())       { dip.pT2 = 0.0; return false; }

    // Generate next evolution scale.
    bool forceFixedAs = (tnow < pT2colCut);
    tnow  = tNextQCD( &dip, emitCoefTot, tnow, pT2endDip, pT2freeze,
      (forceBranching ? -1 : 1));
    if (tnow < 0.) {
      wt = 0.0; dip.pT2 = tnow = 0.;
      double R0 = emitCoefTot*rndmPtr->flat();
      if (!newOverestimates.empty()) {
        if (newOverestimates.lower_bound(R0) == newOverestimates.end())
          splittingNowName = newOverestimates.rbegin()->second;
        else
          splittingNowName = newOverestimates.lower_bound(R0)->second;
      }
      break;
    }

    wt      = 0.0;
    dip.pT2 = tnow;

    // If crossed b threshold, continue evolution from this threshold.
    if        ( nFlavour == 5 && tnow < m2bPhys ) {
      mustFindRange = true;
    // If crossed c threshold, continue evolution from this threshold.
    } else if ( nFlavour == 4 && tnow < m2cPhys ) {
      mustFindRange = true;
    }

    // Leave incoming heavy quarks below threshold unconverted for now.
    if ( event[dip.iRecoiler].idAbs() == 4 && tnow <= m2cPhys) {
      dip.pT2 = 0.; return false;
    }
    if ( event[dip.iRecoiler].idAbs() == 5 && tnow <= m2bPhys) {
      dip.pT2 = 0.; return false;
    }

    // Abort evolution if below cutoff scale, or below another branching.
    if ( tnow < pT2endDip ) { dip.pT2 = tnow = 0.; break; }

    // Try user-defined splittings first.
    double R = emitCoefTot*rndmPtr->flat();
    double z = -1.;
    int idDaughter, idSister;
    idDaughter = idSister = 0;

    if (!newOverestimates.empty()) {

      if (newOverestimates.lower_bound(R) == newOverestimates.end())
        splittingNowName = newOverestimates.rbegin()->second;
      else
        splittingNowName = newOverestimates.lower_bound(R)->second;

      // Generate z value and calculate splitting probability.
      double xMin = (hasPDFrec) ? xRecoiler : 0.;
      getNewSplitting( event, &dip, teval, xMin, tnow, zMinAbs,
        zMaxAbs, idRadiator, splittingNowName, forceFixedAs, idDaughter,
        idSister, z, wt, fullWeightsNow, overWeightNow);

      // Store z value for the splitting.
      dip.z      = z;
    }

    // Done for vanishing accept probability.
    if (wt == 0.) {
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    fullWeightNow = fullWeightsNow["base"];

    // Get particle masses.
    // Radiator before splitting.
    double m2Bef = particleDataPtr->isResonance(event[dip.iRadiator].id())
                 ? getMass(event[dip.iRadiator].id(),3,
                           event[dip.iRadiator].mCalc())
                 : (event[dip.iRadiator].idAbs() < 6
                 || event[dip.iRadiator].id() == 21
                 || event[dip.iRadiator].id() == 22)
                 ? getMass(event[dip.iRadiator].id(),2)
                 : getMass(event[dip.iRadiator].id(),1);
    // Radiator after splitting.
    double m2r   = particleDataPtr->isResonance(idDaughter)
                   && idDaughter == event[dip.iRadiator].id()
                 ? getMass(idDaughter,3,event[dip.iRadiator].mCalc())
                 : (abs(idDaughter) < 6
                     || idDaughter == 21
                     || event[dip.iRadiator].id() == 22)
                 ? getMass(idDaughter,2)
                 : getMass(idDaughter,1);
    // Emission.
    double m2e   = (abs(dip.flavour) < 6
                     || dip.flavour == 21
                     || dip.flavour == 22)
                 ? getMass(dip.flavour,2)
                 : getMass(dip.flavour,1);

    bool canUseSplitInfo = splits[splittingNowName]->canUseForBranching();
    if (canUseSplitInfo) {
      m2Bef = splits[splittingNowName]->splitInfo.kinematics()->m2RadBef;
      m2r   = splits[splittingNowName]->splitInfo.kinematics()->m2RadAft;
      m2e   = splits[splittingNowName]->splitInfo.kinematics()->m2EmtAft;
    }
    int nEmissions = splits[splittingNowName]->nEmissions();

    double q2    = (event[dip.iRecoiler].p()
                   -event[dip.iRadiator].p()).m2Calc();
    // Recalculate the kinematicaly available dipole mass.
    double Q2    = dip.m2Dip - m2Bef + m2r + m2e;

    // Disallow gluon -> heavy quarks if pT has fallen below 2*mQuark.
    if ( event[dip.iRecoiler].idAbs() == 5 && nEmissions == 2
      && tnow <= 4.*m2bPhys) {
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    } else if ( event[dip.iRecoiler].idAbs() == 4 && nEmissions == 2
      && tnow <= 4.*m2cPhys) {
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Discard this 1->3 splitting if the pT has fallen below mEmission (since
    // such splittings would not be included in the virtual corrections to the
    // 1->2 kernels. Note that the threshold is pT>mEmission,since alphaS is
    // evaluated at pT, not virtuality sa1).
    if ( nEmissions == 2
      && ( (abs(dip.flavour) == 4 && tnow < m2cPhys)
        || (abs(dip.flavour) == 5 && tnow < m2bPhys))) {
      mustFindRange = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    double m2a(m2e), m2i(m2e), m2j(m2Bef), m2aij(m2Bef), m2k(0.0);
    if (canUseSplitInfo)
      m2j = splits[splittingNowName]->splitInfo.kinematics()->m2EmtAft2;

    // Recalculate the kinematicaly available dipole mass.
    // Calculate CS variables.
    double kappa2 =  tnow/Q2;
    double xCS    = 1 - kappa2/(1.-z);
    double xCDST  = xCS*( 1. + (m2Bef-m2r-m2e)/Q2 );
    double xNew   = xRecoiler / xCDST;

    // Jacobian factors.
    double jacobian = 1.;
    if (canUseSplitInfo) {
      jacobian
        = splits[splittingNowName]->getJacobian(event,partonSystemsPtr);
      unordered_map<string,double> psvars
        = splits[splittingNowName]->getPhasespaceVars(event, partonSystemsPtr);
      xNew = psvars["xInAft"];
    }

    // Firstly reduce by PDF ratio.
    double pdfRatio = 1.;
    pdfScale2 = (useFixedFacScale) ? fixedFacScale2
      : factorMultFac * tnow;
    pdfScale2 = max(pdfScale2, pT2colCut);
    double pdfScale2Old = pdfScale2;
    double pdfScale2New = pdfScale2;
    if (forceBranching)  pdfScale2New = pdfScale2Old = infoPtr->Q2Fac();
    bool inD = hasPDFrec ? beam.insideBounds(xRecoiler, pdfScale2Old) : true;
    bool inM = hasPDFrec ? beam.insideBounds(xNew, pdfScale2New)      : true;
    double pdfOld = getXPDF(idRecoiler, xRecoiler, pdfScale2Old, iSysRec,
      &beam, false, z, dip.m2Dip);
    double pdfNew = getXPDF(idRecoiler, xNew, pdfScale2New, iSysRec,
      &beam, false, z, dip.m2Dip);

    if ( hasPDFrec && pdfOld != 0.
      && abs(pdfOld) < tinypdf(xRecoiler) ) {
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Extreme case of low-scale gluon, for which denominator rapidly
    // approaches zero. In this case, cut off branching probability if
    // daughter PDF fell too rapidly, to avoid large shower weights.
    // (Note: Last resort - would like something more physical here!)
    double xPDFrecoilerLow = getXPDF(idRecoiler, xRecoiler,
      pdfScale2Old*pdfScale2Old/max(teval, pT2colCut), iSysRec, &beam);
    if ( idRecoiler == 21
      && ( abs(pdfOld/xPDFrecoiler) < 1e-4
        || abs(xPDFrecoilerLow/pdfOld) < 1e-4) ) {
      hasTinyPDFdau = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Set PDF ratio to zero if x-/pT-values are out-of-bounds.
    pdfRatio = (inD && inM) ? pdfNew/pdfOld : 0.;

    if (!canUseSplitInfo) {
      // Jacobian for 1->2 splittings, in CS variables.
      if ( nEmissions!= 2 ) jacobian   = ( 1.- xCS) / ( 1. - xCDST);

      // Jacobian for 1->3 splittings, in CS variables.
      if ( nEmissions == 2 ) {
        double m2ai  = dip.sa1 + m2a + m2i;
        xCS          = (q2 - m2ai - m2a - m2i)
                     / (q2 - m2ai - m2a - m2i - dip.pT2 * dip.xa/dip.z);

        // Jacobian for competing steps, i.e. applied to over-all
        // splitting rate.
        double saij = (xCS - 1.)/xCS * (q2 - m2a) + (m2ai + m2j)/xCS;
        double xbar = (q2 - m2aij - m2k) / (q2 - saij - m2k);

        // Calculate the partonic eCM before the splitting.
        double sHatBefore = (event[dip.iRecoiler].p() + pOther).m2Calc();
        double m2OtherBeam = 0.;

        // Now construct the new recoiler momentum.
        Vec4 q(event[dip.iRecoiler].p()-event[dip.iRadiator].p());
        Vec4 pRadBef(event[dip.iRadiator].p());
        Vec4 pRecBef(event[dip.iRecoiler].p());
        Vec4 qpar(q.px()+pRadBef.px(), q.py()+pRadBef.py(), q.pz(), q.e());
        double qpar2 = qpar.m2Calc();
        double pT2ijt = pow2(pRadBef.px()) + pow2(pRadBef.py());
        Vec4 pRec( (pRecBef - (qpar*pRecBef)/qpar2 * qpar)
                  * sqrt( (lABC(q2,saij,m2k)   - 4.*m2k*pT2ijt)
                         /(lABC(q2,m2aij,m2k) - 4.*m2k*pT2ijt))
                  + qpar * (q2+m2k-saij)/(2.*qpar2) );
        // Calculate the partonic eCM after the splitting.
        double sHatAfter = (pOther + pRec).m2Calc();

        // Calculate Jacobian.
        double rho_bai = sqrt( lABC(sHatBefore, m2k, m2OtherBeam)
                             / lABC(sHatAfter,  m2k, m2OtherBeam) );
        jacobian = rho_bai/xbar
                 * (saij + m2k - q2) / sqrt( lABC(saij, m2k, q2) );

        // Additional jacobian for non-competing steps.
        double saib = m2ai + m2k
          + dip.z/dip.xa * (q2 - m2k - m2ai - m2j - dip.pT2*dip.xa/dip.z);
        jacobian *= (m2ai + m2k - saib) / sqrt( lABC(m2ai, m2k, saib) );

        xCDST = xCS * ( 1. - (m2aij-m2ai-m2j)/ (q2-m2ai-m2j-m2k) );
        // Extra correction from massless to massive propagator.
        jacobian   *= ( 1.- xCS) / ( 1. - xCDST);

        // Recalculate PDF ratio.
        xNew     = xRecoiler / xCDST;
        inM      = (!hasPDFrec) ? true : beam.insideBounds(xNew, pdfScale2);
        pdfNew   = getXPDF(idRecoiler, xNew, pdfScale2, iSysRec, &beam);
        pdfRatio = (inD && inM) ? pdfNew/pdfOld : 0.;
      }
    }

    // More last resort.
    if ( idRecoiler == 21 && pdfScale2 == pT2colCut
      && pdfRatio > 50.) pdfRatio = 0.;

    fullWeightNow  *= pdfRatio*jacobian;

    for ( unordered_map<string,double>::iterator it = fullWeightsNow.begin();
      it != fullWeightsNow.end(); ++it )
      it->second   *= pdfRatio*jacobian;

    // Before generating kinematics: Reset sai if the kernel fell on an
    // endpoint contribution.
    if ( nEmissions == 2
      && splits[splittingNowName]->splitInfo.kinematics()->sai == 0.)
      dip.sa1 = 0.;

    if (fullWeightNow == 0. ) {
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Retrieve argument of alphaS.
    double scale2 =  splits[splittingNowName]->couplingScale2 ( z, tnow, Q2,
      make_pair (event[dip.iRadiator].id(), event[dip.iRadiator].isFinal()),
      make_pair (event[dip.iRecoiler].id(), event[dip.iRecoiler].isFinal()));
    if (scale2 < 0.) scale2 = tnow;
    double talpha = max(scale2, pT2colCut);

    // Reweight to match PDF alpha_s, including corrective terms for
    // renormalisation scale variation. For NLO splitting, all coupling
    // factors have already been covered in the competing phase.
    alphasReweight(tnow, talpha, dip.system, forceFixedAs, wt, fullWeightNow,
      overWeightNow, renormMultFac);
    auxWeightNow   = overWeightNow;

    // Create muR-variations.
    double asw = 1.;
    alphasReweight(tnow, talpha, dip.system, forceFixedAs, daux, asw, daux,
      renormMultFac);
    fullWeightsNow["base"] *= asw;
    if (fullWeightsNow.find("base_order_as2") != fullWeightsNow.end())
      fullWeightsNow["base_order_as2"] *= asw;
    if (doVariations) {
      if ( splittingNowName.find("qcd") != string::npos
        && settingsPtr->parm("Variations:muRfsrDown") != 1.) {
        asw = 1.;
        alphasReweight(tnow, talpha, dip.system, forceFixedAs, daux, asw, daux,
          (tnow > pT2minVariations) ? settingsPtr->
                       parm("Variations:muRfsrDown") : renormMultFac);
        fullWeightsNow["Variations:muRfsrDown"] *= asw;
      } else if ( splittingNowName.find("qcd") == string::npos )
        fullWeightsNow["Variations:muRfsrDown"] *= asw;
      if ( splittingNowName.find("qcd") != string::npos
        && settingsPtr->parm("Variations:muRfsrUp")   != 1.) {
        asw = 1.;
        alphasReweight(tnow, talpha, dip.system, forceFixedAs, daux, asw, daux,
          (tnow > pT2minVariations) ? settingsPtr->parm("Variations:muRfsrUp")
          : renormMultFac);
        fullWeightsNow["Variations:muRfsrUp"] *= asw;
      } else if ( splittingNowName.find("qcd") == string::npos )
        fullWeightsNow["Variations:muRfsrUp"] *= asw;

      // PDF variations.
      if (hasPDFrec && settingsPtr->flag("Variations:PDFup") ) {
        int valSea = (beam[iSysRec].isValence()) ? 1 : 0;
        if( beam[iSysRec].isUnmatched() ) valSea = 2;
        beam.calcPDFEnvelope( make_pair(idRecoiler, idRecoiler),
          make_pair(xNew,xRecoiler), pdfScale2, valSea);
        PDF::PDFEnvelope ratioPDFEnv = beam.getPDFEnvelope();
        double deltaPDFplus
          = min(ratioPDFEnv.errplusPDF  / ratioPDFEnv.centralPDF, 10.);
        double deltaPDFminus
          = min(ratioPDFEnv.errminusPDF / ratioPDFEnv.centralPDF, 10.);
        fullWeightsNow["Variations:PDFup"]   = fullWeightsNow["base"]
          * ((tnow > pT2minVariations) ? (1.0 + deltaPDFplus) : 1.0);
        fullWeightsNow["Variations:PDFdown"] = fullWeightsNow["base"]
          * ((tnow > pT2minVariations) ? (1.0 - deltaPDFminus) : 1.0);
      }
    }

    // Ensure that accept probability is positive.
    if (fullWeightNow < 0.) {
      auxWeightNow *= -1.;
    }

    // Reset overestimate if necessary.
    if ( fullWeightNow/auxWeightNow > 1.) {
      direInfoPtr->message(1) << __FILE__ << " " << __func__
        << " " << __LINE__ << " : Large acceptance weight="
        << fullWeightNow/auxWeightNow
        << " for splitting " << splittingNowName << " at pT2=" << tnow
        << " and z=" << z << "\t(PDF ratio=" << pdfRatio << ")" << endl;
      //mustFindRange = true;
      double rescale = fullWeightNow/auxWeightNow * 1.15;
      auxWeightNow *= rescale;
      infoPtr->errorMsg("Info in DireTimes::pT2nextQCD_FI: Found large "
                        "acceptance weight for " + splittingNowName);
    }

    wt = fullWeightNow/auxWeightNow;

  // Iterate until acceptable pT (or have fallen below pTmin).
  } while (wt < rndmPtr->flat());

  // Not possible to find splitting.
  if ( wt == 0.) return false;

  // Update accepted event weight. No weighted shower for first
  // "pseudo-emission" step in 1->3 splitting.
  if ( fullWeightNow != 0. && overWeightNow != 0. ) {
    double enhanceFurther
      = enhanceOverestimateFurther(splittingNowName, idRadiator, teval);
    double tnow = dip.pT2;
    if (doTrialNow) {
      weights->addTrialEnhancement(tnow, enhanceFurther);
      enhanceFurther = 1.;
    }
    kernelNow = fullWeightsNow;
    auxNow    = auxWeightNow;
    overNow   = overWeightNow;
    boostNow  = enhanceFurther;
    for ( unordered_map<string,double>::iterator it = fullWeightsNow.begin();
      it != fullWeightsNow.end(); ++it ) {

      // No weight bookkeeping for non-variation components of kernel vector.
      if (it->first == "base_order_as2") continue;

     acceptProbability[it->first].insert(make_pair(tnow,
        auxWeightNow/overWeightNow * 1./enhanceFurther
        * it->second/fullWeightNow ) );
      if (auxWeightNow == fullWeightNow && overWeightNow == fullWeightNow)
        rejectProbability[it->first].insert( make_pair(tnow, 1.0));
      else {
        double wv  = auxWeightNow/overWeightNow
                  * (overWeightNow- it->second/enhanceFurther)
                  / (auxWeightNow - fullWeightNow);
        if (abs(wv) > 1e0) {
        direInfoPtr->message(1) << __FILE__ << " " << __func__
        << " " << __LINE__ << " : Large reject weight=" << wv
        << "\t for kernel=" << it->second << " overestimate=" << overNow
        << "\t aux. overestimate=" << auxNow << " at pT2="
        << tnow
        <<  " for " << splittingNowName << endl;
        }
        rejectProbability[it->first].insert( make_pair(tnow, wv));
      }
    }
  }

  // Done
  return true;

}

//--------------------------------------------------------------------------

double DireTimes::tNextQCD( DireTimesEnd*, double overestimateInt,
  double tOld, double tMin, double tFreeze, int algoType) {

  // Return if below cut-off.
  bool forceFixedAs = (tOld < pT2colCut);
  double asOver     = (usePDFalphas || forceFixedAs)
                    ? alphaS2piOverestimate : alphaS2pi;
  double rnd        = rndmPtr->flat();

  // Use cut-off on random numbers to account for minimal t. Only implemented
  // for t-independent overestimates.
  if (usePDFalphas || alphaSorder == 0) {
    double rndMin     = pow( tMin/tOld, asOver * overestimateInt);
    if (rnd < rndMin) return -1.*tMin;
  }

  // Determine LambdaQCD.
  double b0            = 4.5;
  double Lambda2       = Lambda3flav2;
  if (tOld > m2b) {
    b0       = 23./6.;
    Lambda2  = Lambda5flav2;
  } else if (tOld > m2c) {
    b0       = 25./6.;
    Lambda2  = Lambda4flav2;
  } else {
    b0       = 27./6.;
    Lambda2  = Lambda3flav2;
  }
  // A change of renormalization scale expressed by a change of Lambda.
  Lambda2 /= renormMultFac;

  // Generate next evolution scale.
  double tForAlphaS=tOld;
  double tnow = tOld;

  if (algoType<0)
    return pow(tMin+tFreeze,rnd) / pow(tnow+tFreeze,rnd-1) - tFreeze;

  if (usePDFalphas || forceFixedAs)
    tnow = (tnow+tFreeze) * pow( rnd,
        1. / (alphaS2piOverestimate * overestimateInt)) - tFreeze;

  else if (alphaSorder == 0)
    tnow = (tnow+tFreeze) * pow( rnd,
      1. / (alphaS2pi * overestimateInt) ) - tFreeze;

  else if (alphaSorder == 1)
    tnow = Lambda2 * pow( (tnow+tFreeze) / Lambda2,
      pow( rnd, b0 / overestimateInt) ) - tFreeze;

  else {
    do {
      tnow = Lambda2 * pow( (tnow+tFreeze) / Lambda2,
        pow(rndmPtr->flat(), b0 / overestimateInt) ) - tFreeze;
      tForAlphaS = renormMultFac * max( tnow+tFreeze,
        pow2(LAMBDA3MARGIN) * Lambda3flav2);
    } while (alphaS.alphaS2OrdCorr(tForAlphaS) < rndmPtr->flat()
      && tnow > tMin);
  }

  // Done.
  return tnow;

}

//--------------------------------------------------------------------------

// Get auxiliary variable for brnaching of a QCD dipole end.

bool DireTimes::zCollNextQCD( DireTimesEnd* dip, double zMin, double zMax,
  double, double ) {

  // Choose logarithmically.
  dip->xa = zMax * pow( zMax/zMin, -rndmPtr->flat());

  // Done
  return true;

}

//--------------------------------------------------------------------------

bool DireTimes::virtNextQCD( DireTimesEnd* dip, double, double,
  double, double) {

  double v   = rndmPtr->flat();
  double m2j = dip->mass[2];
  dip->sa1 = v / (1.-v) * ( dip->pT2*dip->xa/dip->z + m2j);

  // Done
  return true;

}

//--------------------------------------------------------------------------

bool DireTimes::branch( Event& event, bool ) {

  if (abs(dipSel->pT2 - pT2cutMin(dipSel)) < 1e-10) return false;

  // This function is a wrapper for setting up the branching
  // kinematics.
  bool hasBranched = false;
  if ( event[dipSel->iRecoiler].isFinal())
       hasBranched = branch_FF(event, false, &splitInfoSel);
  else hasBranched = branch_FI(event, false, &splitInfoSel);

  // Done.
  return hasBranched;

}

//--------------------------------------------------------------------------

// ME corrections and kinematics that may give failure.
// Notation: radBef, recBef = radiator, recoiler before emission,
//           rad, rec, emt = radiator, recoiler, emitted efter emission.
//           (rad, emt distinguished by colour flow for g -> q qbar.)

bool DireTimes::branch_FF( Event& event, bool trial,
  DireSplitInfo* split ) {

  Event auxevent1 = event;
  Event auxevent2 = event;

  // Check if the first emission should be studied for removal.
  bool physical      = true;
  bool canMergeFirst = (mergingHooksPtr != 0)
                     ? mergingHooksPtr->canVetoEmission() : false;

  // Find initial radiator and recoiler particles in dipole branching.
  int iRadBef      = (!trial) ? dipSel->iRadiator : split->iRadBef;
  int iRecBef      = (!trial) ? dipSel->iRecoiler : split->iRecBef;

  // Find their momenta, with special sum for global recoil.
  Vec4 pRadBef(event[iRadBef].p());
  Vec4 pRecBef(event[iRecBef].p());

  // Get splitting variables.
  string name = (!trial) ? splittingSelName : split->splittingSelName;
  if (!trial) splits[name]->splitInfo.store(*split);

  unordered_map<string,double> psp(splits[name]->
    getPhasespaceVars(event, partonSystemsPtr));
  double pT2    = (!trial) ? dipSel->pT2   : split->kinematics()->pT2;
  double z      = (!trial) ? dipSel->z     : split->kinematics()->z ;
  // Allow splitting kernel to overwrite phase space variables.
  if (split->useForBranching) { pT2 = psp["pT2"]; z = psp["z"]; }

  double m2Dip  = (!trial) ? dipSel->m2Dip : split->kinematics()->m2Dip;
  // Calculate CS variables.
  double yCS    = pT2/m2Dip / (1.-z);
  double zCS    = ( 1. - z - pT2/m2Dip - pow2(1.-z) )
                / ( 1. - z - pT2/m2Dip);

  // Get flavour of splitting.
  int flavour = (!trial) ? dipSel->flavour : split->emtAft()->id;
  // Name of the splitting.
  int nEmissions = splits[name]->nEmissions();

  if ( nEmissions == 2 && !split->useForBranching) flavour = 21;

  // Default flavours and colour tags for new particles in dipole branching.
  int idRad        = event[iRadBef].id();
  int idEmt        = abs(flavour);
  int colRad       = event[iRadBef].col();
  int acolRad      = event[iRadBef].acol();
  int colEmt       = 0;
  int acolEmt      = 0;
  iSysSel          = (!trial) ? dipSel->system : split->system;
  int iSysSelRec   = (!trial) ? dipSel->systemRec : split->system;
  int colType = -2;
  if (!trial && dipSel->colType > 0) colType = 2;
  if ( trial && idRad > 0)           colType = 2;
  if (!trial && (dipSel->gamType == 1 || abs(dipSel->weakType) > 0)) colType=0;
  if ( trial && (idRad==22 || idRad==23 || idRad==24 || idRad==25))  colType=0;

  if ( split->useForBranching
    && (particleDataPtr->colType(split->emtAft()->id)  == 0
    || (particleDataPtr->colType(split->emtAft2()->id) == 0
      && nEmissions == 2)))
    colType = 0;

  if (flavour == 22 || flavour == 23 || flavour == 25) ;

  // New colour tag required for gluon emission.
  else if (flavour == 21 && colType > 0) {
    colEmt  = colRad;
    colRad  = event.nextColTag();
    acolEmt = colRad;
  } else if (flavour == 21) {
    acolEmt = acolRad;
    acolRad = event.nextColTag();
    colEmt  = acolRad;
  // New flavours for g -> q qbar; split colours.
  } else if (colType > 0) {
    idEmt   = abs(flavour);
    idRad   = -idEmt;
    colEmt  = colRad;
    colRad  = 0;
  } else if (colType < 0) {
    idEmt   = -abs(flavour);
    idRad   = -idEmt;
    acolEmt  = acolRad;
    acolRad  = 0;
  }

  if (split->useForBranching) {
    idRad   = event[iRadBef].id();
    colRad  = event[iRadBef].col();
    acolRad = event[iRadBef].acol();
    colEmt  = 0;
    acolEmt = 0;
    // Now reset if splitting information is available.
    if (split->radAft()->id   != 0) idRad   = split->radAft()->id;
    if (split->emtAft()->id   != 0) idEmt   = split->emtAft()->id;
    if (split->radAft()->col  > -1) colRad  = split->radAft()->col;
    if (split->radAft()->acol > -1) acolRad = split->radAft()->acol;
    if (split->emtAft()->col  > -1) colEmt  = split->emtAft()->col;
    if (split->emtAft()->acol > -1) acolEmt = split->emtAft()->acol;

    if (nEmissions == 2) {
      if ( split->extras.find("idRadInt") != split->extras.end() )
        idRad = int(split->extras["idRadInt"]);
      if ( split->extras.find("idEmtInt") != split->extras.end() )
        idEmt = int(split->extras["idEmtInt"]);
      if ( split->extras.find("colRadInt") != split->extras.end() )
        colRad = int(split->extras["colRadInt"]);
      if ( split->extras.find("acolRadInt") != split->extras.end() )
        acolRad = int(split->extras["acolRadInt"]);
      if ( split->extras.find("colEmtInt") != split->extras.end() )
        colEmt = int(split->extras["colEmtInt"]);
      if ( split->extras.find("acolEmtInt") != split->extras.end() )
        acolEmt = int(split->extras["acolEmtInt"]);
    }
  }

  // Get particle masses.
  // Radiator before splitting.
  double m2Bef = particleDataPtr->isResonance(event[iRadBef].id())
               ? getMass(event[iRadBef].id(),3,event[iRadBef].mCalc())
               : (event[iRadBef].idAbs() < 6 || event[iRadBef].id() == 21
               || event[iRadBef].id() == 22)
               ? getMass(event[iRadBef].id(),2)
               : getMass(event[iRadBef].id(),1);
  // Radiator after splitting.
  double m2r   = particleDataPtr->isResonance(idRad)
                 && idRad == event[iRadBef].id()
               ? getMass(idRad,3,event[iRadBef].mCalc())
               : (abs(idRad) < 6 || idRad == 21 || idRad == 22)
               ? getMass(idRad,2)
               : getMass(idRad,1);
  // Recoiler.
  double m2s   = particleDataPtr->isResonance(event[iRecBef].id())
               ? getMass(event[iRecBef].id(),3,event[iRecBef].mCalc())
               : (event[iRecBef].idAbs() < 6 || event[iRecBef].id() == 21
               || event[iRecBef].id() == 22)
               ? getMass(event[iRecBef].id(),2)
               : getMass(event[iRecBef].id(),1);
  // Emission.
  double m2ex = (abs(idEmt) < 6 || idEmt == 21 || idEmt == 22)
              ? getMass(idEmt,2) : getMass(idEmt,1);
  double m2e  = (!trial) ? m2ex
    : ( (split->kinematics()->m2EmtAft > 0.) ? split->kinematics()->m2EmtAft
                                            : m2ex);
  if (split->useForBranching) {
    m2r = split->kinematics()->m2RadAft;
    m2e = split->kinematics()->m2EmtAft;
  }

  // Adjust the dipole kinematical mass to accomodate masses after branching.
  double Q2  = m2Dip + m2Bef - m2r - m2e;
  // Q2 already correctly stored in splitInfo.
  if (trial) Q2 = m2Dip;

  // Calculate CS variables.
  double kappa2 = pT2/Q2;
  yCS           = kappa2 / (1.-z);
  zCS           = ( 1. - z - kappa2 - pow2(1.-z) ) / ( 1. - z - kappa2);
  double m2Emt  = m2e;
  double m2Rad  = m2r;
  double sai    = (!trial) ? dipSel->sa1 : split->kinematics()->sai;
  double xa     = (!trial) ? dipSel->xa  : split->kinematics()->xa;
  // Allow splitting kernel to overwrite phase space variables.
  if (split->useForBranching) { sai = psp["sai"]; xa = psp["xa"]; }

  // Auxiliary angle.
  double phi_kt = (!trial)
    ? ((dipSel->phi >= 0.)              ? dipSel->phi
                                        : 2.*M_PI*rndmPtr->flat())
    : ((split->kinematics()->phi >= 0.) ? split->kinematics()->phi
                                        : 2.*M_PI*rndmPtr->flat());

  double phi_kt1 = phi_kt+2./3.*M_PI;
  if (phi_kt1>2.*M_PI) phi_kt1 -= 2.*M_PI;
  double phi_kt2 = phi_kt-2./3.*M_PI;
  if (phi_kt2<0.)      phi_kt2 += 2.*M_PI;
  if (phi_kt1<phi_kt2) swap(phi_kt1, phi_kt2);

  // Second angle for 1->3 splitting.
  double phiX = 0.0;
  if (nEmissions == 2)
    phiX = (!trial)
      ? ((dipSel->phia1 >= 0.)             ? dipSel->phia1
                                           : 2.*M_PI*rndmPtr->flat())
      : ((split->kinematics()->phi2 >= 0.) ? split->kinematics()->phi2
                                           : 2.*M_PI*rndmPtr->flat());

  // Allow splitting kernel to overwrite phase space variables.
  if (split->useForBranching) { phi_kt = psp["phi"]; phiX = psp["phi2"]; }

  Vec4 pRad, pEmt, pRec;
  Vec4 auxpRad1, auxpEmt1;
  Vec4 auxpRad2, auxpEmt2;
  // Get dipole 4-momentum.
  Vec4 q(pRadBef + pRecBef);
  double q2 = q.m2Calc();

  // 1->3 splittings generated in CS variables directly.
  double m2a(0.), m2i(0.), m2j(0.), m2ai(0.), m2k(0.), m2aij(0.);
  if (nEmissions == 2) {
    m2a   = getMass((!trial) ? dipSel->flavour : split->emtAft()->id,2);
    m2i   = getMass((!trial) ? dipSel->flavour : split->emtAft()->id,2);
    m2j   = m2r;
    m2k   = m2s;
    if (split->useForBranching) {
      m2a = split->kinematics()->m2RadAft;
      m2i = split->kinematics()->m2EmtAft;
      m2j = split->kinematics()->m2EmtAft2;
    }
    m2aij = m2Bef;
    m2ai  = sai + m2a + m2i;
    Q2    = m2Dip + m2aij + m2k - m2ai - m2j - m2k;
    yCS   = pT2/(q2 - m2ai - m2j - m2k) * xa / z;
    zCS   = z / (xa*(1-yCS)) * (q2 - m2aij - m2k) / (q2 - m2ai - m2j - m2k);
    m2Emt = m2Rad;
    m2Rad = m2ai;
    if (split->useForBranching) {
      m2Emt = split->kinematics()->m2EmtAft;
      m2Rad = sai + m2a + m2j;
      m2ai  = sai + m2a + m2j;
    }
  }

  // Calculate derived variables.
  double sij  = yCS * (q2 - m2s) + (1.-yCS)*(m2Rad+m2Emt);

  double zbar = (q2-sij-m2s) / bABC(q2,sij,m2s)
              * (zCS - m2s/gABC(q2,sij,m2s)
                      *(sij + m2Rad - m2Emt)/(q2-sij-m2s));
  double kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2Rad - zbar*m2Emt;

  // Not possible to construct kinematics if kT2 < 0.0
  if (kT2 < 0.) {
    infoPtr->errorMsg("Warning in DireTimes::branch_FF: Reject state "
      "with kinematically forbidden kT^2.");
    physical = false;
  }

  // NaN kT2 can happen for a 1->3 splitting in which the g->QQ~ produces
  // massive quarks Q.
  if (kT2!=kT2 || abs(kT2-kT2) > 1e5) {
    infoPtr->errorMsg("Warning in DireTimes::branch_FF: Reject state "
      "with not-a-number kT^2 for branching " + name);
    physical = false;
  }

  // Now construct the new recoiler momentum in the lab frame.
  pRec.p( (pRecBef - (q*pRecBef)/q2 * q)
            * sqrt(lABC(q2,sij,m2s)/lABC(q2,m2Bef,m2s))
            + q * (q2+m2s-sij)/(2.*q2) );

  // Construct left-over dipole momentum by momentum conservation.
  Vec4 pij(q-pRec);

  // Set up transverse momentum vector by using two perpendicular four-vectors.
  pair<Vec4, Vec4> pTvecs = getTwoPerpendicular(pRec, pij);
  Vec4 kTmom( sqrt(kT2)*sin(phi_kt)*pTvecs.first
         + sqrt(kT2)*cos(phi_kt)*pTvecs.second);

  // Construct new radiator momentum.
  pRad.p( zbar * (gABC(q2,sij,m2s)*pij - sij*pRec) / bABC(q2,sij,m2s)
            + (m2Rad+kT2) / (zbar*bABC(q2,sij,m2s))
              * (pRec - m2s/gABC(q2,sij,m2s)*pij)
            + kTmom);

  // Contruct the emission momentum by momentum conservation.
  pEmt.p(q-pRad-pRec);

  kTmom.p( sqrt(kT2)*sin(phi_kt1)*pTvecs.first
         + sqrt(kT2)*cos(phi_kt1)*pTvecs.second);
  auxpRad1.p( zbar * (gABC(q2,sij,m2s)*pij - sij*pRec) / bABC(q2,sij,m2s)
            + (m2Rad+kT2) / (zbar*bABC(q2,sij,m2s))
              * (pRec - m2s/gABC(q2,sij,m2s)*pij)
            + kTmom);
  auxpEmt1.p(q-auxpRad1-pRec);

  kTmom.p( sqrt(kT2)*sin(phi_kt2)*pTvecs.first
         + sqrt(kT2)*cos(phi_kt2)*pTvecs.second);
  auxpRad2.p( zbar * (gABC(q2,sij,m2s)*pij - sij*pRec) / bABC(q2,sij,m2s)
            + (m2Rad+kT2) / (zbar*bABC(q2,sij,m2s))
              * (pRec - m2s/gABC(q2,sij,m2s)*pij)
            + kTmom);
  auxpEmt2.p(q-auxpRad2-pRec);

  // Ensure that radiator is on mass-shell
  double errMass = abs(pRad.mCalc() - sqrt(m2Rad)) / max( 1.0, pRad.e());
  if ( errMass > mTolErr*1e-2 ) {
    double deltam2 = pRad.m2Calc() - m2Rad;
    pRad.e(sqrtpos(pow2(pRad.e()) - deltam2));
  }
  // Ensure that emission is on mass-shell
  errMass = abs(pEmt.mCalc() - sqrt(m2Emt)) / max( 1.0, pEmt.e());
  if ( errMass > mTolErr*1e-2 ) {
    double deltam2 = pEmt.m2Calc() - m2Emt;
    pEmt.e(sqrtpos(pow2(pEmt.e()) - deltam2));
  }
  // Ensure that recoiler is on mass-shell
  errMass = abs(pRec.mCalc() - sqrt(m2s)) / max( 1.0, pRec.e());
  if ( errMass > mTolErr*1e-2 ) {
    double deltam2 = pRec.m2Calc() - m2s;
    pRec.e(sqrtpos(pow2(pRec.e()) - deltam2));
  }

  // Swap emitted and radiator properties for first part of
  // 1->3 splitting (q -> "massive gluon" + q)
  if ( nEmissions == 2 && !split->useForBranching) {
    swap(idRad,idEmt);
    swap(colRad,colEmt);
    swap(acolRad,acolEmt);
  }

  // For emitted color singlet, redefine the colors of the "massive gluon".
  if ( nEmissions == 2 && split->useForBranching
    && particleDataPtr->colType(split->emtAft()->id)  == 0)
    { colRad = event[iRadBef].col(); acolRad = event[iRadBef].acol(); }

  // Define new particles from dipole branching.
  double pTsel = sqrt(pT2);
  Particle rad = Particle(idRad, 51, iRadBef, 0, 0, 0,
    colRad, acolRad, pRad, sqrt(m2Rad), pTsel);
  Particle auxrad1 = Particle(idRad, 51, iRadBef, 0, 0, 0,
    colRad, acolRad, auxpRad1, sqrt(m2Rad), pTsel);
  Particle auxrad2 = Particle(idRad, 51, iRadBef, 0, 0, 0,
    colRad, acolRad, auxpRad2, sqrt(m2Rad), pTsel);

  // Exempt off-shell radiator from Pythia momentum checks.
  if ( nEmissions == 2) rad.status(59);

  Particle emt = Particle(idEmt, 51, iRadBef, 0, 0, 0,
    colEmt, acolEmt, pEmt, sqrt(m2Emt), pTsel);
  Particle auxemt1 = Particle(idEmt, 51, iRadBef, 0, 0, 0,
    colEmt, acolEmt, auxpEmt1, sqrt(m2Emt), pTsel);
  Particle auxemt2 = Particle(idEmt, 51, iRadBef, 0, 0, 0,
    colEmt, acolEmt, auxpEmt2, sqrt(m2Emt), pTsel);

  // Default to stored color info for intermediate step in 1->3 branching.
  if ( nEmissions == 2 && split->useForBranching ) {
    if ( split->extras.find("colRadInt") != split->extras.end() )
      rad.col(int(split->extras["colRadInt"]));
    if ( split->extras.find("acolRadInt") != split->extras.end() )
      rad.acol(int(split->extras["acolRadInt"]));
    if ( split->extras.find("colEmtInt") != split->extras.end() )
      emt.col(int(split->extras["colEmtInt"]));
    if ( split->extras.find("acolEmtInt") != split->extras.end() )
      emt.acol(int(split->extras["acolEmtInt"]));
  }

  Particle rec = Particle(event[iRecBef].id(), 52, iRecBef, iRecBef, 0, 0,
    event[iRecBef].col(), event[iRecBef].acol(), pRec, sqrt(m2s), pTsel);

  // Special checks to set weak particles status equal to 56.
  // This is needed for decaying the particles. Also set polarisation.
  if (emt.idAbs() == 23 || emt.idAbs() == 24) {
    emt.status(56);
    if (!trial) {
      event[iRadBef].pol(dipSel->weakPol);
      rad.pol(dipSel->weakPol);
    }
  }

  // Save properties to be restored in case of user-hook veto of emission.
  int evSizeOld    = event.size();
  int iRadStatusV  = event[iRadBef].status();
  int iRadDau1V    = event[iRadBef].daughter1();
  int iRadDau2V    = event[iRadBef].daughter2();
  int iRecStatusV  = event[iRecBef].status();
  int iRecDau1V    = event[iRecBef].daughter1();
  int iRecDau2V    = event[iRecBef].daughter2();

  // Shower may occur at a displaced vertex.
  if (event[iRadBef].hasVertex()) {
    rad.vProd( event[iRadBef].vProd() );
    emt.vProd( event[iRadBef].vProd() );
  }
  if (event[iRecBef].hasVertex()) rec.vProd( event[iRecBef].vProd() );

  // Put new particles into the event record.
  // Mark original dipole partons as branched and set daughters/mothers.
  int iRad(event.append(rad));
  int iEmt(event.append(emt));
  int iEmt2(iEmt);
  event[iRadBef].statusNeg();
  event[iRadBef].daughters( iRad, iEmt);
  int iRec(event.append(rec));
  event[iRecBef].statusNeg();
  event[iRecBef].daughters( iRec, iRec);

  int auxiRad1(auxevent1.append(auxrad1));
  int auxiEmt1(auxevent1.append(auxemt1));
  auxevent1[iRadBef].statusNeg();
  auxevent1[iRadBef].daughters( auxiRad1, auxiEmt1);
  int auxiRec1(auxevent1.append(rec));
  auxevent1[iRecBef].statusNeg();
  auxevent1[iRecBef].daughters( auxiRec1, auxiRec1);

  int auxiRad2(auxevent2.append(auxrad2));
  int auxiEmt2(auxevent2.append(auxemt2));
  auxevent2[iRadBef].statusNeg();
  auxevent2[iRadBef].daughters( auxiRad2, auxiEmt2);
  int auxiRec2(auxevent2.append(rec));
  auxevent2[iRecBef].statusNeg();
  auxevent2[iRecBef].daughters( auxiRec2, auxiRec2);

  if ( nEmissions == 2 && !split->useForBranching) swap(iRad,iEmt);

  // Store flavour again, in case dipSel gets removed.
  int flavourNow = (!trial) ? dipSel->flavour : split->emtAft()->id;

  // Check user veto for 1->2 branchings.
  bool inResonance = (partonSystemsPtr->getInA(iSysSel) == 0) ? true : false;
  bool doVeto      = false;
  if (nEmissions != 2)
    doVeto = (( canVetoEmission && userHooksPtr->doVetoFSREmission(
                evSizeOld,event,iSysSel,inResonance) )
          ||  ( canMergeFirst && mergingHooksPtr->doVetoEmission(
                event) ));
  bool doMECreject = false;

  if ( nEmissions != 2) {

    // Check momenta.
    if ( !validMomentum( rad.p(), idRad, 1)
      || !validMomentum( emt.p(), idEmt, 1)
      || !validMomentum( rec.p(), event[iRecBef].id(), 1))
      physical = false;

    // Apply ME correction if necessary.
    bool isHardSystem = partonSystemsPtr->getSystemOf(iRadBef,true) == 0
                     && partonSystemsPtr->getSystemOf(iRecBef,true) == 0;

    // Try to find incoming particle in other systems, i.e. if the current
    // system arose from a resonance decay.
    int sys = partonSystemsPtr->getSystemOf(iRadBef,true);
    int sizeSys = partonSystemsPtr->sizeSys();
    int in1 = partonSystemsPtr->getInA(sys);
    int in2 = partonSystemsPtr->getInB(sys);
    if ( in1 == 0 && in2 == 0 ) {
      int iParentInOther = 0;
      int nSys = partonSystemsPtr->sizeAll(sys);
      for (int iInSys = 0; iInSys < nSys; ++iInSys){
        int iNow = partonSystemsPtr->getAll(sys,iInSys);
        for (int iOtherSys = 0; iOtherSys < sizeSys; ++iOtherSys){
          if (iOtherSys == sys) continue;
          int nOtherSys = partonSystemsPtr->sizeAll(iOtherSys);
          for (int iInOtherSys = 0; iInOtherSys < nOtherSys; ++iInOtherSys){
            int iOtherNow = partonSystemsPtr->getAll(iOtherSys,iInOtherSys);
            if (event[iNow].isAncestor(iOtherNow)) {
              iParentInOther = iOtherNow;
            }
          }
        }
      }
      if (iParentInOther) isHardSystem = true;
    }

    if ( isHardSystem && physical && doMEcorrections
      //&& pT2 > pT2minMECs && checkSIJ(event,1.)) {
      && pT2 > pT2minMECs && checkSIJ(event,Q2minMECs)) {

      // Finally update the list of all partons in all systems.
      partonSystemsPtr->replace(iSysSel, iRadBef, iRad);
      partonSystemsPtr->addOut(iSysSel, iEmt);
      partonSystemsPtr->replace(iSysSelRec, iRecBef, iRec);

      if ( nFinalMaxMECs < 0
        || nFinalMaxMECs > partonSystemsPtr->sizeOut(iSysSel))
        doMECreject = applyMEC (event, split,
          createvector<Event>(auxevent1)(auxevent2));

      partonSystemsPtr->replace(iSysSel, iRad, iRadBef);
      partonSystemsPtr->replace(iSysSelRec, iRec, iRecBef);
      partonSystemsPtr->popBackOut(iSysSel);
    }

    // Update dipoles and beams.
    if (physical && !doVeto && !trial && !doMECreject) updateAfterFF( iSysSel,
      iSysSelRec, event, iRadBef, iRecBef, iRad, iEmt, iRec, flavour, colType,
      pTsel);

  // Heavy particle 1->2 decay for "second step" in 1->3 splitting.
  } else {

    // Check momenta.
    if ( !validMomentum( emt.p(), idEmt, 1)
      || !validMomentum( rec.p(), event[iRecBef].id(), 1))
      physical = false;

    int iRadOld = int(event.size())-3;
    int iEmtOld = int(event.size())-2;
    int iRecOld = int(event.size())-1;

    // Swap emitted and radiator indices.
    swap(iRadOld,iEmtOld);

    if (!split->useForBranching) {
      // Flavours already fixed by 1->3 kernel.
      idEmt        = -flavourNow;
      idRad        =  flavourNow;
      // Colour tags for new particles in branching.
      if (idEmt < 0) {
        colEmt  = 0;
        acolEmt = event[iEmtOld].acol();
        colRad  = event[iEmtOld].col();
        acolRad = 0;
      } else {
        colEmt  = event[iEmtOld].col();
        acolEmt = 0;
        colRad  = 0;
        acolRad = event[iEmtOld].acol();
      }
    // Already correctly read id and colors from SplitInfo object.
    } else {
      idRad   = split->radAft()->id;
      idEmt   = split->emtAft2()->id;
      colRad  = split->radAft()->col;
      acolRad = split->radAft()->acol;
      colEmt  = split->emtAft2()->col;
      acolEmt = split->emtAft2()->acol;

      if (int(split->extras["swapped"]) == 1) {
        idRad   = split->emtAft()->id;
        idEmt   = split->emtAft2()->id;
        colRad  = split->emtAft()->col;
        acolRad = split->emtAft()->acol;
        colEmt  = split->emtAft2()->col;
        acolEmt = split->emtAft2()->acol;
      }

    }

    // Get particle masses.
    m2Bef = m2ai;
    // Radiator after splitting.
    m2r   = particleDataPtr->isResonance(idRad)
              && idRad == event[iRadBef].id()
          ? getMass(idRad,3,event[iRadBef].mCalc())
          : (abs(idRad) < 6 || idRad == 21 || idRad == 22)
          ? getMass(idRad,2)
          : getMass(idRad,1);
    // Recoiler.
    m2s   = particleDataPtr->isResonance(event[iRecBef].id())
          ? getMass(event[iRecBef].id(),3,event[iRecBef].mCalc())
          : (event[iRecBef].idAbs() < 6 || event[iRecBef].id() == 21
          || event[iRecBef].id() == 22)
          ? getMass(event[iRecBef].id(),2)
          : getMass(event[iRecBef].id(),1);
    // Emission.
    m2e   = (abs(idEmt) < 6 || idEmt == 21 || idEmt == 22)
          ? getMass(idEmt,2) : getMass(idEmt,1);

    if (split->useForBranching) {
      m2r = split->kinematics()->m2RadAft;
      m2e = split->kinematics()->m2EmtAft2;
    }

    // Construct FF dipole momentum.
    Vec4 pa1(event[iEmtOld].p());
    q.p(pa1 + pRec);
    q2 = q.m2Calc();

    // Calculate CS variables.
    m2Emt      = m2e;
    m2Rad      = m2e;
    if (split->useForBranching) {
      m2Rad = split->kinematics()->m2RadAft;
      m2Emt = split->kinematics()->m2EmtAft2;
    }
    zCS        = xa;
    yCS = (m2ai-m2Emt-m2Rad) / (m2ai-m2Emt-m2Rad + 2.*pa1*pRec);

    // Calculate derived variables.
    sij  = yCS * (q2 - m2s) + (1.-yCS)*(m2Rad+m2Emt);
    zbar = (q2-sij-m2s) / bABC(q2,sij,m2s)
               * (zCS - m2s/gABC(q2,sij,m2s)
                       *(sij + m2Rad - m2Emt)/(q2-sij-m2s));
    kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2Rad - zbar*m2Emt;

    if(kT2 < 0.) physical = false;

    // Construct left-over dipole momentum by momentum conservation.
    pij.p(q-pRec);

    // Set up transverse momentum vector by using two perpendicular 4-vectors.
    pTvecs = getTwoPerpendicular(pRec, pij);
    kTmom.p( sqrt(kT2)*sin(phiX)*pTvecs.first
           + sqrt(kT2)*cos(phiX)*pTvecs.second);

    // Construct new radiator momentum.
    pRad.p( zbar * (gABC(q2,sij,m2s)*pij - sij*pRec) / bABC(q2,sij,m2s)
              + (m2Rad+kT2) / (zbar*bABC(q2,sij,m2s))
                * (pRec - m2s/gABC(q2,sij,m2s)*pij)
              + kTmom);

    // Contruct the emission momentum by momentum conservation.
    pEmt.p(q-pRad-pRec);

    // Recoiler unchanged.
    pRec.p(event[iRecOld].p());

    // Define new particles from dipole branching.
    Particle rad2 = Particle(idRad, 51, iEmtOld, 0, 0, 0,
      colRad, acolRad, pRad, sqrt(m2r), pTsel);
    Particle emt2 = Particle(idEmt, 51, iEmtOld, 0, 0, 0,
      colEmt, acolEmt, pEmt, sqrt(m2e), pTsel);
    Particle rec2 = Particle(event[iRecOld].id(), 52, iRecOld, iRecOld, 0, 0,
      event[iRecOld].col(), event[iRecOld].acol(), pRec, sqrt(m2s), pTsel);

    // Check momenta.
    if ( !validMomentum( rad2.p(), idRad, 1)
      || !validMomentum( emt2.p(), idEmt, 1)
      || !validMomentum( rec2.p(), event[iRecOld].id(), 1) )
      physical = false;

    // Check invariants.
    if ( false ) {
      Vec4 pa(pRad), pk(pRec), pj(emt.p()), pi(pEmt);
      double saix(2.*pa*pi), sakx(2.*pa*pk), sajx(2.*pa*pj), sikx(2.*pi*pk),
             sjkx(2.*pj*pk), sijx(2.*pi*pj);
      double pptt = (sajx+sijx)*(sakx+sikx)
                  / ( (event[iRadBef].p()+event[iRecBef].p()).m2Calc()
                     - event[iRadBef].m2Calc() - event[iRecBef].m2Calc() );
      double ssaaii = saix;
      double zzaa = sakx
                  / ( (event[iRadBef].p()+event[iRecBef].p()).m2Calc()
                     - event[iRadBef].m2Calc() - event[iRecBef].m2Calc() );
      double xxaa = sakx / ( sakx + sikx );
      if ( physical &&
           (abs(pptt-pT2) > 1e-5 || abs(ssaaii-sai) > 1e-5 ||
            abs(zzaa-z) > 1e-5   || abs(xxaa-xa) > 1e-5) ){
        cout << scientific << setprecision(8);
        cout << "Error in branch_FF: Invariant masses after branching do not "
             << "match chosen values." << endl;
        cout << "Chosen:    "
             << " Q2 " << (event[iRadBef].p()+event[iRecBef].p()).m2Calc()
             << " pT2 " << pT2
             << " sai " << sai
             << " za " << z
             << " xa " << xa << endl;
        cout << "Generated: "
             << " Q2 " << sakx+saix+sajx+sijx+sikx+sjkx
             << " pT2 " << pptt
             << " sai " << ssaaii
             << " za " << zzaa
             << " xa " << xxaa << endl;
        physical = false;
      }
    }

    // Update bookkeeping
    if (physical) {

      // Update dipoles and beams.
      if (!trial) {
        updateAfterFF( iSysSel, iSysSelRec, event,
          iRadBef, iRecBef, iRad, iEmt, iRec, flavour, colType, pTsel);

        // Shower may occur at a displaced vertex.
        if (event[iEmtOld].hasVertex()) {
          rad2.vProd( event[iEmtOld].vProd() );
          emt2.vProd( event[iEmtOld].vProd() );
        }
        if (event[iRecOld].hasVertex()) rec2.vProd( event[iRecOld].vProd() );
      }

      // Put new particles into the event record.
      // Mark original dipole partons as branched and set daughters/mothers.
      iRad = event.append(rad2);
      iEmt = event.append(emt2);
      event[iEmtOld].statusNeg();
      event[iEmtOld].daughters( iRad, iEmt);
      iRec = event.append(rec2);
      event[iRecOld].statusNeg();
      event[iRecOld].daughters( iRec, iRec);

      // Update dipoles and beams.
      if (!trial) {
        int colTypeNow = colType;
        updateAfterFF( iSysSel, iSysSelRec, event, iEmtOld, iRecOld, iRad,
          iEmt,iRec, flavourNow, colTypeNow, pTsel);
      }
    }
  }

  physical = physical && !doVeto;

  if ( physical && !trial && !doMECreject
    && !validMotherDaughter(event)) {
    infoPtr->errorMsg("Error in DireTimes::branch_FF: Mother-daughter "
                      "relations after branching not valid.");
    physical = false;
  }

  // Allow veto of branching. If so restore event record to before emission.
  if ( !physical || doMECreject ) {
    event.popBack( event.size() - evSizeOld);
    event[iRadBef].status( iRadStatusV);
    event[iRadBef].daughters( iRadDau1V, iRadDau2V);
    event[iRecBef].status( iRecStatusV);
    event[iRecBef].daughters( iRecDau1V, iRecDau2V);

    // This case is identical to the case where the probability to accept the
    // emission was indeed zero all along. In this case, neither
    // acceptProbability nor rejectProbability would have been filled. Thus,
    // remove the relevant entries from the weight container!
    if (!trial) {
      for ( unordered_map<string, multimap<double,double> >::iterator
        it = rejectProbability.begin(); it != rejectProbability.end(); ++it){
        weights->eraseAcceptWeight(pT2, it->first);
        weights->eraseRejectWeight(pT2, it->first);
      }
    }

    if (!trial && doMECreject) {
      weights->calcWeight(pT2, false, true);
      weights->reset();
     // Clear accept/reject weights.
      for ( unordered_map<string, multimap<double,double> >::iterator
        it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
        it->second.clear();
      for ( unordered_map<string, map<double,double> >::iterator
        it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
        it->second.clear();
    }

    return false;
  }

  // Store positions of new particles.
  if (trial && !split->useForBranching) split->storePosAfter(iRad, iRec, iEmt,
    (nEmissions < 2) ? 0 : iEmt2);
  if (trial &&  split->useForBranching) split->storePosAfter(iRad, iRec, iEmt2,
    (nEmissions < 2) ? 0 : iEmt);

  // Set shower weight.
  if (!trial) {
    if (!doTrialNow) {
      weights->calcWeight(pT2);
      weights->reset();
      // Store positions of new soft particles.
      if (nEmissions == 1) {
        direInfoPtr->updateSoftPos( iRadBef, iEmt );
        direInfoPtr->updateSoftPosIfMatch( iRecBef, iRec );
      // For 1->3 splitting, do not tag emissions as "soft".
      } else {
        direInfoPtr->updateSoftPos( iRadBef, iRad );
      }

      updateDipoles(event, iSysSel);
    }

    // Clear accept/reject weights.
    for ( unordered_map<string, multimap<double,double> >::iterator
      it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
      it->second.clear();
    for ( unordered_map<string, map<double,double> >::iterator
      it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
      it->second.clear();
  }

  // Done.
  return true;

}

//--------------------------------------------------------------------------

void DireTimes::updateAfterFF( int iSysSelNow, int iSysSelRec,
  Event& event, int iRadBef, int iRecBef, int iRad, int iEmt, int iRec,
  int, int colType, double pTsel) {

  vector<int> iDipEndCorr, iDipEndRem;
  bool inResonance = (partonSystemsPtr->getInA(iSysSelNow)==0) ? true : false;
  int idEmt        = event[iEmt].id();
  bool hasDipSel   = (dipSel != 0);

  // Gluon emission: update both dipole ends and add two new ones.
  if (particleDataPtr->colType(idEmt) == 2) {

    if (hasDipSel) {
      dipSel->iRadiator  = iRad;
      dipSel->iRecoiler  = iEmt;
      dipSel->systemRec  = iSysSelNow;
      dipSel->isrType    = 0;
      dipSel->pTmax      = pTsel;
      dipSel->MEtype     = 0;
    }

    for (int i = 0; i < int(dipEnd.size()); ++i) {
      DireTimesEnd& dip = dipEnd[i];
      if (dip.iRadiator == iRecBef && dip.iRecoiler == iRadBef
        && dip.colType != 0) {
        dip.iRadiator = iRec;
        dip.iRecoiler = iEmt;
        dip.MEtype = 0;
        // Strive to match colour to anticolour inside closed system.
        if ( dip.colType * colType > 0) dip.iRecoiler = iRad;
        dip.pTmax = pTsel;
        iDipEndCorr.push_back(i);
      }
    }

    int colTypeNow = (colType > 0) ? 2 : -2 ;
    // When recoiler was uncoloured particle, in resonance decays,
    // assign recoil to coloured particle.
    int iRecMod = iRec;
    if (recoilToColoured && inResonance && event[iRec].col() == 0
      && event[iRec].acol() == 0) iRecMod = iRad;
    if (appendDipole( event, iEmt, iRecMod, pTsel, colTypeNow, 0, 0, 0, 0,
          iSysSelNow, 0, -1, 0, false, dipEnd)) {
      iDipEndCorr.push_back(dipEnd.size()-1);
      // Set dipole mass properties.
      DireTimesEnd& dip1 = dipEnd.back();
      dip1.systemRec = iSysSelRec;
    }

    if (appendDipole( event, iEmt, iRad, pTsel,-colTypeNow, 0, 0, 0, 0,
          iSysSelNow, 0, -1, 0, false, dipEnd)) {
      iDipEndCorr.push_back(dipEnd.size()-1);
    }

  // Gluon branching to q qbar: update current dipole and other of gluon.
  } else if (particleDataPtr->colType(idEmt) != 0) {

    // Update dipoles for second step in 1->3 splitting.
    if ( splittingsPtr->nEmissions(splittingSelName) == 2 ){
      for (int i = 0; i < int(dipEnd.size()); ++i) {

        DireTimesEnd& dip = dipEnd[i];

        if ( dip.iRadiator == iRecBef ) {
          dip.iRadiator = iRec;
        }
        if ( dip.iRecoiler == iRecBef ) {
          dip.iRecoiler = iRec;
        }

        if ( dip.iRadiator == iRadBef ) {
          if (dip.colType > 0)
            dip.iRadiator = (event[iEmt].id() > 0) ? iEmt : iRad;
          if (dip.colType < 0)
            dip.iRadiator = (event[iEmt].id() < 0) ? iEmt : iRad;

          if (abs(dip.colType) == 2
            && event[dip.iRadiator].id()    > 0
            && event[dip.iRadiator].idAbs() < 10)
            dip.colType = abs(dip.colType)/2;
          if (abs(dip.colType) == 2
            && event[dip.iRadiator].id()    < 0
            && event[dip.iRadiator].idAbs() < 10)
            dip.colType = -abs(dip.colType)/2;

          if (dip.colType == 0) dip.iRadiator = iRad;

          iDipEndCorr.push_back(i);
        }

        if ( dip.iRecoiler == iRadBef ) {
          if (dip.colType > 0)
            dip.iRecoiler = (event[iEmt].id() < 0) ? iEmt : iRad;
          if (dip.colType < 0)
            dip.iRecoiler = (event[iEmt].id() > 0) ? iEmt : iRad;

          if (abs(dip.colType) == 2) dipEnd[i].colType /= 2;

          if (abs(dip.colType) == 1
            && event[dip.iRadiator].id()    > 0
            && event[dip.iRadiator].idAbs() < 10)
            dip.colType = 1;

          if (abs(dip.colType) == 1
            && event[dip.iRadiator].id()    < 0
            && event[dip.iRadiator].idAbs() < 10)
            dip.colType = -1;

          if (dip.colType == 0) dip.iRecoiler = iRad;

          iDipEndCorr.push_back(i);
        }
      }
    }

    for (int i = 0; i < int(dipEnd.size()); ++i) {
      // Nothing to be done if dipole end has already been updated.
      if ( find(iDipEndCorr.begin(), iDipEndCorr.end(), i)
        != iDipEndCorr.end() ) continue;
      DireTimesEnd& dip = dipEnd[i];
      // Strive to match colour to anticolour inside closed system.
      if ( dip.iRecoiler == iRadBef && dip.colType * colType < 0 ) {
        dip.iRecoiler = iEmt;
      }
      if (dip.iRadiator == iRadBef && abs(dip.colType) == 2) {
        dip.colType /= 2;

        if (hasDipSel && &dipEnd[i] == dipSel) dip.iRadiator  = iEmt;
        else                                   dip.iRadiator  = iRad;
        if (hasDipSel && &dipEnd[i] == dipSel) dip.iRecoiler  = iRec;

        dip.colType = (event[dip.iRadiator].id() > 0)
                    ? abs(dip.colType) : -abs(dip.colType);

        // Potentially also update recoiler!
        if ( dip.iRecoiler == iRecBef ) dip.iRecoiler = iRec;

        iDipEndCorr.push_back(i);

        if (dip.system != dip.systemRec) continue;
        dip.MEtype = 0;
        if (hasDipSel && &dipEnd[i] == dipSel) dip.iMEpartner = iRad;
        else                      dip.iMEpartner = iEmt;

      }
    }

    // Nothing to be done if dipole end has already been updated.
    bool updateSel=true;
    for (int j = 0; j < int(iDipEndCorr.size()); ++j)
      if ( hasDipSel && &dipEnd[iDipEndCorr[j]] == dipSel) updateSel = false;

    if (hasDipSel) {
      if (updateSel) {
        dipSel->iRadiator = iEmt;
        dipSel->iRecoiler = iRec;
      }
      // Always update the production pT.
      dipSel->pTmax     = pTsel;
    }

  } else {

    int iRadOld = (hasDipSel) ? dipSel->iRadiator : iRadBef;
    int iRecOld = (hasDipSel) ? dipSel->iRecoiler : iRecBef;
    // Just update old radiator/recoiler to current outgoing particles.
    for (int i = 0; i < int(dipEnd.size()); ++i) {
      DireTimesEnd& dip = dipEnd[i];
      // Update radiator-recoiler end.
      if ( dip.iRecoiler == iRecOld && dip.iRadiator == iRadOld ) {
        dip.iRadiator = iRad;
        dip.iRecoiler = iRec;
        dip.pTmax  = pTsel;
        iDipEndCorr.push_back(i);
      }
      // Update recoiler-radiator end.
      if ( dip.iRecoiler == iRadOld && dip.iRadiator == iRecOld ) {
        dip.iRadiator = iRec;
        dip.iRecoiler = iRad;
        dip.pTmax = pTsel;
        iDipEndCorr.push_back(i);
      }
    }
  }

  // Now update other dipoles that also involved the radiator or recoiler.
  for (int i = 0; i < int(dipEnd.size()); ++i) {
    // Nothing to be done if dipole end has already been updated.
    if ( find(iDipEndCorr.begin(), iDipEndCorr.end(), i)
      != iDipEndCorr.end() ) continue;
    DireTimesEnd& dip = dipEnd[i];
    if (dip.iRadiator  == iRadBef) dip.iRadiator  = iRad;
    if (dip.iRecoiler  == iRadBef) dip.iRecoiler  = iRad;
    if (dip.iMEpartner == iRadBef) dip.iMEpartner = iRad;
    if (dip.iRadiator  == iRecBef) dip.iRadiator  = iRec;
    if (dip.iRecoiler  == iRecBef) dip.iRecoiler  = iRec;
    if (dip.iMEpartner == iRecBef) dip.iMEpartner = iRec;
  }

  // Now update or construct new dipoles if the radiator or emission allow
  // for new types of emissions.
  vector<pair<int, int> > rad_rec (createvector< pair<int,int> >
    (make_pair(iRad,iEmt))
    (make_pair(iEmt,iRec))
    (make_pair(iRec,iEmt))
    (make_pair(iEmt,iRad))
    (make_pair(iRad,iRec))
    (make_pair(iRec,iRad)));
  for (int i=0; i < int(rad_rec.size()); ++i) {
    int iRadNow = rad_rec[i].first;
    int iRecNow = rad_rec[i].second;
    // Now check if a new dipole end a-b should be added:
    // First check if the dipole end is already existing.
    vector<int> iDip;
    for (int j = 0; j < int(dipEnd.size()); ++j)
      if ( dipEnd[j].iRadiator == iRadNow
        && dipEnd[j].iRecoiler == iRecNow )
        iDip.push_back(j);
    // If the dipole end exists, attempt to update the dipole end (a)
    // for the current a-b dipole.
    if ( int(iDip.size()) > 0) for (int j = 0; j < int(iDip.size()); ++j)
      updateAllowedEmissions(event, &dipEnd[iDip[j]]);
    // If no dipole exists and idEmtAfter != 0, create new dipole end (a).
    else appendDipole( event, iRadNow, iRecNow, pTsel, 0, 0, 0, 0, 0,
      iSysSelNow, -1, -1, 0, false, dipEnd);
  }

  // Copy or set lifetime for new final state.
  if (event[iRad].id() == event[iRadBef].id())
    event[iRad].tau( event[iRadBef].tau() );
  else {
    event[iRad].tau( event[iRad].tau0() * rndmPtr->exp() );
    event[iEmt].tau( event[iEmt].tau0() * rndmPtr->exp() );
  }
  event[iRec].tau( event[iRecBef].tau() );

  // Finally update the list of all partons in all systems.
  partonSystemsPtr->replace(iSysSelNow, iRadBef, iRad);
  partonSystemsPtr->addOut(iSysSelNow, iEmt);
  partonSystemsPtr->replace(iSysSelRec, iRecBef, iRec);

  // Loop through final state of system to find possible new dipole ends.
  for (int i = 0; i < partonSystemsPtr->sizeOut(iSysSelNow); ++i) {
    int iNow = partonSystemsPtr->getOut( iSysSelNow, i);
    if (event[iNow].isFinal() && event[iNow].scale() > 0.) {
      // Find dipole end formed by colour index.
      int colTag = event[iNow].col();
      if (doQCDshower && colTag > 0) setupQCDdip( iSysSelNow, i,  colTag,  1,
        event, false, true);
      // Find dipole end formed by anticolour index.
      int acolTag = event[iNow].acol();
      if (doQCDshower && acolTag > 0) setupQCDdip( iSysSelNow, i, acolTag, -1,
        event, false, true);
      // Now find non-QCD dipoles and/or update the existing dipoles.
      getGenDip( iSysSelNow, i, iNow, event, false, dipEnd);
    }
    // Setup decay dipoles.
    if (doDecaysAsShower && event[iNow].isResonance())
      setupDecayDip( iSysSelNow, iNow, event, dipEnd);
  }

  // Now remove inactive dipoles.
  sort (iDipEndRem.begin(), iDipEndRem.end());
  for (int i = iDipEndRem.size()-1; i >= 0; --i) {
    dipEnd[iDipEndRem[i]] = dipEnd.back();
    dipEnd.pop_back();
  }

  // Now update all dipoles.
  dipSel = 0;
  updateDipoles(event, iSysSelNow);

  // Bookkeep shower-induced resonances.
  if ( direInfoPtr->isRes(iRadBef) &&
    event[iRadBef].id() != event[iRad].id() )
    direInfoPtr->removeResPos(iRadBef);
  if ( particleDataPtr->isResonance(event[iRad].id()) ) {
    if ( direInfoPtr->isRes(iRadBef) )
         direInfoPtr->updateResPos(iRadBef,iRad);
    //else direInfoPtr->addResPos(iRad);
  }
  if ( direInfoPtr->isRes(iRecBef) )
    direInfoPtr->updateResPos(iRecBef,iRec);
  if ( particleDataPtr->isResonance(event[iEmt].id()) )
    direInfoPtr->addResPos(iEmt);

  int nFinal = 0;
  for (int i=0; i < event.size(); ++i) if (event[i].isFinal()) nFinal++;
  if (nFinal == 3) {
    int col = (colType> 0) ? event[iRadBef].col() : event[iRadBef].acol();
    vector<int>::iterator iRem =
      find(bornColors.begin(), bornColors.end(), col);
    if (iRem != bornColors.end()) bornColors.erase(iRem++);
  }

  // Done.
}

//--------------------------------------------------------------------------

// ME corrections and kinematics that may give failure.
// Notation: radBef, recBef = radiator, recoiler before emission,
//           rad, rec, emt = radiator, recoiler, emitted efter emission.
//           (rad, emt distinguished by colour flow for g -> q qbar.)

bool DireTimes::branch_FI( Event& event, bool trial,
  DireSplitInfo* split ) {

  Event auxevent1 = event;
  Event auxevent2 = event;

  // Check if the first emission should be studied for removal.
  bool physical      = true;
  bool canMergeFirst = (mergingHooksPtr != 0)
                     ? mergingHooksPtr->canVetoEmission() : false;

  // Find initial radiator and recoiler particles in dipole branching.
  int iRadBef      = (!trial) ? dipSel->iRadiator : split->iRadBef;
  int iRecBef      = (!trial) ? dipSel->iRecoiler : split->iRecBef;
  int isrType      = event[iRecBef].mother1();

  // Find their momenta, with special sum for global recoil.
  Vec4 pRadBef     = event[iRadBef].p();
  Vec4 pRecBef     = event[iRecBef].p();

  // Get splitting variables.
  string name = (!trial) ? splittingSelName : split->splittingSelName;
  double pT2    = (!trial) ? dipSel->pT2   : split->kinematics()->pT2;
  double z      = (!trial) ? dipSel->z     : split->kinematics()->z;
  splits[name]->splitInfo.store(*split);
  unordered_map<string,double> psp(splits[name]->
                                   getPhasespaceVars(event, partonSystemsPtr));
  // Allow splitting kernel to overwrite phase space variables.
  if (split->useForBranching) { pT2 = psp["pT2"]; z = psp["z"]; }
  double m2Dip  = (!trial) ? dipSel->m2Dip : split->kinematics()->m2Dip;

  // Calculate CS variables.
  double kappa2 = pT2/m2Dip;
  double zCS    = z;
  double xCS    = 1 - kappa2/(1.-z);

  // Get flavour of splitting.
  int flavour = (!trial) ? dipSel->flavour : split->emtAft()->id;
  // Store flavour again, in case dipSel gets removed or flavour gets reset.
  int flavourSave = flavour;
  // Name of the splitting.
  int nEmissions = splits[name]->nEmissions();

  if ( nEmissions == 2 && !split->useForBranching) flavour = 21;

  // Default flavours and colour tags for new particles in dipole branching.
  int idRad        = event[iRadBef].id();
  int idEmt        = abs(flavour);
  int colRad       = event[iRadBef].col();
  int acolRad      = event[iRadBef].acol();
  int colEmt       = 0;
  int acolEmt      = 0;
  iSysSel          = (!trial) ? dipSel->system : split->system;
  int iSysSelRec   = (!trial) ? dipSel->systemRec : split->system;

  int colType = -2;
  if (!trial && dipSel->colType > 0) colType = 2;
  if ( trial && idRad > 0)           colType = 2;
  if (!trial && (dipSel->gamType == 1 || abs(dipSel->weakType) > 0)) colType=0;
  if ( trial && (idRad==22 || idRad==23 || idRad==24 || idRad==25))  colType=0;

  if ( split->useForBranching
    && (particleDataPtr->colType(split->emtAft()->id)  == 0
    || (particleDataPtr->colType(split->emtAft2()->id) == 0
      && nEmissions == 2)))
    colType = 0;

  if (flavour == 22 || flavour == 23 || flavour == 25) ;

  // New colour tag required for gluon emission.
  else if (flavour == 21 && colType > 0) {
    colEmt  = colRad;
    colRad  = event.nextColTag();
    acolEmt = colRad;
  } else if (flavour == 21) {
    acolEmt = acolRad;
    acolRad = event.nextColTag();
    colEmt  = acolRad;
  // New flavours for g -> q qbar; split colours.
  } else if (colType > 0) {
    idEmt   = abs(flavour);
    idRad   = -idEmt;
    colEmt  = colRad;
    colRad  = 0;
  } else if (colType < 0) {
    idEmt   = -abs(flavour);
    idRad   = -idEmt;
    acolEmt  = acolRad;
    acolRad  = 0;
  }

  if (split->useForBranching) {
    idRad   = event[iRadBef].id();
    colRad  = event[iRadBef].col();
    acolRad = event[iRadBef].acol();
    colEmt  = 0;
    acolEmt = 0;
    // Now reset if splitting information is available.
    if (split->radAft()->id   != 0) idRad   = split->radAft()->id;
    if (split->emtAft()->id   != 0) idEmt   = split->emtAft()->id;
    if (split->radAft()->col  > -1) colRad  = split->radAft()->col;
    if (split->radAft()->acol > -1) acolRad = split->radAft()->acol;
    if (split->emtAft()->col  > -1) colEmt  = split->emtAft()->col;
    if (split->emtAft()->acol > -1) acolEmt = split->emtAft()->acol;
  }

  // Get particle masses.
  // Radiator before splitting.
  double m2Bef = particleDataPtr->isResonance(event[iRadBef].id())
               ? getMass(event[iRadBef].id(),3,event[iRadBef].mCalc())
               : (event[iRadBef].idAbs() < 6 || event[iRadBef].id() == 21
               || event[iRadBef].id() == 22)
               ? getMass(event[iRadBef].id(),2)
               : getMass(event[iRadBef].id(),1);
  // Radiator after splitting.
  double m2r   = particleDataPtr->isResonance(idRad)
                 && idRad == event[iRadBef].id()
               ? getMass(idRad,3,event[iRadBef].mCalc())
               : (abs(idRad) < 6 || idRad == 21 || idRad == 22)
               ? getMass(idRad,2)
               : getMass(idRad,1);
  // Emission.
  double m2ex = (abs(idEmt) < 6 || idEmt == 21 || idEmt == 22)
              ? getMass(idEmt,2) : getMass(idEmt,1);
  double m2e  = (!trial) ? m2ex
    : ( (split->kinematics()->m2EmtAft > 0.) ? split->kinematics()->m2EmtAft
                                            : m2ex);

  if (split->useForBranching) {
    m2r = split->kinematics()->m2RadAft;
    m2e = split->kinematics()->m2EmtAft;
  }

  // Initial state recoiler always assumed massless.
  double m2s = 0.0;
  // Recoiler mass
  if ( useMassiveBeams && (event[iRecBef].idAbs() == 11
                        || event[iRecBef].idAbs() == 13
                        || event[iRecBef].idAbs() > 900000))
    m2s   = getMass(event[iRecBef].id(),1);

  // Recalculate the kinematicaly available dipole mass.
  double Q2 = m2Dip - m2Bef + m2r + m2e;

  // Calculate CS variables.
  kappa2       =  pT2/Q2;
  xCS          = 1 - kappa2/(1.-z);
  double m2Emt = m2e;
  double m2Rad = m2r;
  double sai   = (!trial) ? dipSel->sa1 : split->kinematics()->sai;
  double xa    = (!trial) ? dipSel->xa  : split->kinematics()->xa;

  // Allow splitting kernel to overwrite phase space variables.
  if (split->useForBranching) { sai = psp["sai"]; xa = psp["xa"]; }

  // Auxiliary angle.
  double phi_kt = (!trial)
    ? ((dipSel->phi >= 0.)              ? dipSel->phi
                                        : 2.*M_PI*rndmPtr->flat())
    : ((split->kinematics()->phi >= 0.) ? split->kinematics()->phi
                                        : 2.*M_PI*rndmPtr->flat());
  double phi_kt1 = phi_kt+2./3.*M_PI;
  if (phi_kt1>2.*M_PI) phi_kt1 -= 2.*M_PI;
  double phi_kt2 = phi_kt-2./3.*M_PI;
  if (phi_kt2<0.)      phi_kt2 += 2.*M_PI;
  if (phi_kt1<phi_kt2) swap(phi_kt1, phi_kt2);

  // Second angle for 1->3 splitting.
  double phiX = 0.0;
  if (nEmissions == 2)
    phiX = (!trial)
      ? ((dipSel->phia1 >= 0.)             ? dipSel->phia1
                                           : 2.*M_PI*rndmPtr->flat())
      : ((split->kinematics()->phi2 >= 0.) ? split->kinematics()->phi2
                                           : 2.*M_PI*rndmPtr->flat());

  // Allow splitting kernel to overwrite phase space variables.
  if (split->useForBranching) { phi_kt = psp["phi"]; phiX = psp["phi2"]; }

  Vec4 pRad, pEmt, pRec;
  Vec4 auxpRad1, auxpEmt1;
  Vec4 auxpRad2, auxpEmt2;

  // Get dipole 4-momentum.
  Vec4 q(pRecBef-pRadBef);
  double q2 = q.m2Calc();

  // 1->3 splittings generated in CS variables directly.
  double m2a(0.), m2i(0.), m2j(0.), m2ai(0.);
  if (nEmissions == 2) {
    m2a   = (abs(flavourSave) < 6
              || flavourSave == 21
              || flavourSave == 22)
          ? getMass(flavourSave,2)
          : getMass(flavourSave,1);
    m2i   = m2a;
    m2j   = m2Bef;
    if (split->useForBranching) {
      m2a = split->kinematics()->m2RadAft;
      m2i = split->kinematics()->m2EmtAft;
      m2j = split->kinematics()->m2EmtAft2;
    }
    m2ai  = sai + m2a + m2i;
    Q2    = m2Dip - m2Bef + m2ai + m2j + m2s;
    zCS   = z / xa;
    xCS   = (q2 - m2ai - m2j - m2s)
          / (q2 - m2ai - m2j - m2s - pT2 * xa/z);
    m2Emt = m2r;
    m2Rad = m2ai;
    if (split->useForBranching) {
      m2Emt = split->kinematics()->m2EmtAft;
      m2Rad = sai + m2a + m2j;
      m2ai  = sai + m2a + m2j;
    }
  }

  Vec4 qpar(q.px()+pRadBef.px(), q.py()+pRadBef.py(), q.pz(), q.e());
  double qpar2 = qpar.m2Calc();
  double pT2ijt = pow2(pRadBef.px()) + pow2(pRadBef.py());

  // Calculate derived variables.
  double sij  = (1.-1./xCS) * (q2 - m2s) + (m2Rad+m2Emt) / xCS;
  double zbar = (q2-sij-m2s) / bABC(q2,sij,m2s)
              * (zCS - m2s/gABC(q2,sij,m2s)
                    *(sij + m2Rad - m2Emt)/(q2-sij-m2s));
  double kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2Rad - zbar*m2Emt;

  // Not possible to construct kinematics if kT2 < 0.0

  if (kT2 < 0.) {
    infoPtr->errorMsg("Warning in DireTimes::branch_FI: Reject state "
      "with kinematically forbidden kT^2.");
    physical = false;
  }

  // NaN kT2 can happen for a 1->3 splitting in which the g->QQ~ produces
  // massive quarks Q.
  if (physical && (kT2!=kT2 || abs(kT2-kT2) > 1e5) ) {
    infoPtr->errorMsg("Warning in DireTimes::branch_FI: Reject state "
      "with not-a-number kT^2 for branching " + name);
    physical = false;
  }

  // Now construct the new recoiler momentum in the lab frame.
  pRec.p( (pRecBef - (q*pRecBef)/qpar2 * qpar)
            * sqrt( (lABC(q2,sij,m2s)   - 4.*m2s*pT2ijt)
                   /(lABC(q2,m2Bef,m2s) - 4.*m2s*pT2ijt))
            + qpar * (q2+m2s-sij)/(2.*qpar2) );

  // Construct left-over dipole momentum by momentum conservation.
  Vec4 pij(-q+pRec);

  // Set up transverse momentum vector by using two perpendicular four-vectors.
  pair<Vec4, Vec4> pTvecs = getTwoPerpendicular(pRec, pij);
  Vec4 kTmom( sqrt(kT2)*sin(phi_kt)*pTvecs.first
         + sqrt(kT2)*cos(phi_kt)*pTvecs.second);

  // Construct new radiator momentum.
  pRad.p( zbar * (gABC(q2,sij,m2s)*pij + sij*pRec) / bABC(q2,sij,m2s)
           + (m2Rad+kT2) / (zbar*bABC(q2,sij,m2s))
             * (-pRec - m2s/gABC(q2,sij,m2s)*pij)
           + kTmom);

  // Contruct the emission momentum by momentum conservation.
  pEmt.p(-q-pRad+pRec);

  kTmom.p( sqrt(kT2)*sin(phi_kt1)*pTvecs.first
         + sqrt(kT2)*cos(phi_kt1)*pTvecs.second);
  auxpRad1.p( zbar * (gABC(q2,sij,m2s)*pij + sij*pRec) / bABC(q2,sij,m2s)
           + (m2Rad+kT2) / (zbar*bABC(q2,sij,m2s))
             * (-pRec - m2s/gABC(q2,sij,m2s)*pij)
           + kTmom);
  auxpEmt1.p(-q-auxpRad1+pRec);

  kTmom.p( sqrt(kT2)*sin(phi_kt2)*pTvecs.first
         + sqrt(kT2)*cos(phi_kt2)*pTvecs.second);
  auxpRad2.p( zbar * (gABC(q2,sij,m2s)*pij + sij*pRec) / bABC(q2,sij,m2s)
           + (m2Rad+kT2) / (zbar*bABC(q2,sij,m2s))
             * (-pRec - m2s/gABC(q2,sij,m2s)*pij)
           + kTmom);
  auxpEmt2.p(-q-auxpRad2+pRec);

  if ( abs(q.m2Calc()) < 1e-5 && pRadBef.m2Calc() > 0.) {
    double yCS = (m2Bef - m2Emt - m2Rad)
               / (m2Bef - m2Emt - m2Rad + 2.*pRadBef*pRecBef);
    // Construct FF dipole momentum.
    q.p(pRadBef + pRecBef);
    q2 = q.m2Calc();
    // Calculate derived variables.
    sij  = yCS * (q2 - m2s) + (1.-yCS)*(m2Rad+m2Emt);
    zbar = (q2-sij-m2s) / bABC(q2,sij,m2s)
              * (zCS - m2s/gABC(q2,sij,m2s)
                      *(sij + m2Rad - m2Emt)/(q2-sij-m2s));
    kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2Rad - zbar*m2Emt;

    if (kT2 > 0. ) {
      if (!trial) infoPtr->errorMsg("Info in DireTimes::branch_FI: "
        "Recued state with previously forbidden kT^2.");
      physical = true;
    }

    pair < Vec4, Vec4 > momsAfter = decayWithOnshellRec( zCS, yCS,
    phi_kt, m2s, m2Rad, m2Emt, pRadBef, pRecBef );
    pRad.p(momsAfter.first);
    pEmt.p(momsAfter.second);
    pRec.p(pRecBef);

    if (isnan(pRad)) physical = false;

  } else if (isnan(pRad)) physical = false;

  // Ensure that radiator is on mass-shell
  double errMass = abs(pRad.mCalc() - sqrt(m2Rad)) / max( 1.0, pRad.e());
  if ( errMass > mTolErr*1e-2 ) {
    double deltam2 = pRad.m2Calc() - m2Rad;
    pRad.e(sqrtpos(pow2(pRad.e()) - deltam2));
  }
  // Ensure that emission is on mass-shell
  errMass = abs(pEmt.mCalc() - sqrt(m2Emt)) / max( 1.0, pEmt.e());
  if ( errMass > mTolErr*1e-2 ) {
    double deltam2 = pEmt.m2Calc() - m2Emt;
    pEmt.e(sqrtpos(pow2(pEmt.e()) - deltam2));
  }
  // Ensure that recoiler is on mass-shell
  errMass = abs(pRec.mCalc() - sqrt(m2s)) / max( 1.0, pRec.e());
  if ( errMass > mTolErr*1e-2 ) {
    double deltam2 = pRec.m2Calc() - m2s;
    pRec.e(sqrtpos(pow2(pRec.e()) - deltam2));
  }

  // New: Return if the x-value for the incoming recoiler is nonsense.
  if ( physical && particleDataPtr->colType(event[iRecBef].id()) != 0
    && 2.*pRec.e()/event[0].m() > 1. ) {
    infoPtr->errorMsg("Error in DireTimes::branch_FI: "
            "Larger than unity Bjorken x value");
    physical = false;
  }

  // Swap emitted and radiator properties for first part of
  // 1->3 splitting (q -> "massive gluon" + q)
  if ( nEmissions == 2 && !split->useForBranching) {
    swap(idRad,idEmt);
    swap(colRad,colEmt);
    swap(acolRad,acolEmt);
  }

  // For emitted color singlet, redefine the colors of the "massive gluon".
  if ( nEmissions == 2 && split->useForBranching
    && particleDataPtr->colType(split->emtAft()->id)  == 0)
    { colRad = event[iRadBef].col(); acolRad = event[iRadBef].acol(); }

  // Define new particles from dipole branching.
  double pTsel = sqrt(pT2);
  Particle rad = Particle(idRad, 51, iRadBef, 0, 0, 0,
    colRad, acolRad, pRad, sqrt(m2Rad), pTsel);
  Particle auxrad1 = Particle(idRad, 51, iRadBef, 0, 0, 0,
    colRad, acolRad, auxpRad1, sqrt(m2Rad), pTsel);
  Particle auxrad2 = Particle(idRad, 51, iRadBef, 0, 0, 0,
    colRad, acolRad, auxpRad2, sqrt(m2Rad), pTsel);

  // Exempt off-shell radiator from Pythia momentum checks.
  if ( nEmissions == 2 ) rad.status(59);

  Particle emt = Particle(idEmt, 51, iRadBef, 0, 0, 0,
    colEmt, acolEmt, pEmt, sqrt(m2Emt), pTsel);
  Particle auxemt1 = Particle(idEmt, 51, iRadBef, 0, 0, 0,
    colEmt, acolEmt, auxpEmt1, sqrt(m2Emt), pTsel);
  Particle auxemt2 = Particle(idEmt, 51, iRadBef, 0, 0, 0,
    colEmt, acolEmt, auxpEmt2, sqrt(m2Emt), pTsel);

  Particle rec = Particle(event[iRecBef].id(), -53, 0, 0, iRecBef, iRecBef,
    event[iRecBef].col(), event[iRecBef].acol(), pRec, 0., pTsel);

  // Special checks to set weak particles status equal to 56.
  // This is needed for decaying the particles. Also set polarisation.
  if (emt.idAbs() == 23 || emt.idAbs() == 24) {
    emt.status(56);
    if (!trial) {
      event[iRadBef].pol( dipSel->weakPol );
      rad.pol( dipSel->weakPol );
    }
  }

  // Default to stored color info for intermediate step in 1->3 branching.
  if ( nEmissions == 2 && split->useForBranching ) {
    if ( split->extras.find("colRadInt") != split->extras.end() )
      rad.col(int(split->extras["colRadInt"]));
    if ( split->extras.find("acolRadInt") != split->extras.end() )
      rad.acol(int(split->extras["acolRadInt"]));
    if ( split->extras.find("colEmtInt") != split->extras.end() )
      emt.col(int(split->extras["colEmtInt"]));
    if ( split->extras.find("acolEmtInt") != split->extras.end() )
      emt.acol(int(split->extras["acolEmtInt"]));
  }

  // Save properties to be restored in case of user-hook veto of emission.
  int evSizeOld    = event.size();
  int iRadStatusV  = event[iRadBef].status();
  int iRadDau1V    = event[iRadBef].daughter1();
  int iRadDau2V    = event[iRadBef].daughter2();
  int iRecMot1V    = event[iRecBef].mother1();
  int iRecMot2V    = event[iRecBef].mother2();
  int beamOff1     = 1 + beamOffset;
  int beamOff2     = 2 + beamOffset;
  int ev1Dau1V     = event[beamOff1].daughter1();
  int ev2Dau1V     = event[beamOff2].daughter1();

  // Shower may occur at a displaced vertex.
  if (event[iRadBef].hasVertex()) {
    rad.vProd( event[iRadBef].vProd() );
    emt.vProd( event[iRadBef].vProd() );
  }
  if (event[iRecBef].hasVertex()) rec.vProd( event[iRecBef].vProd() );

  // Put new particles into the event record.
  // Mark original dipole partons as branched and set daughters/mothers.
  int iRad = int(event.append(rad));
  int iEmt = int(event.append(emt));
  int iEmt2(iEmt);
  event[iRadBef].statusNeg();
  event[iRadBef].daughters( iRad, iEmt);
  int iRec = event.append(rec);
  event[iRecBef].mothers( iRec, iRec);
  event[iRec].mothers( iRecMot1V, iRecMot2V);
  int iBeam1Dau1 = event[beamOff1].daughter1();
  int iBeam2Dau1 = event[beamOff2].daughter1();
  if (iSysSelRec == 0 && iRecMot1V == beamOff1)
    event[beamOff1].daughter1( iRec);
  if (iSysSelRec == 0 && iRecMot1V == beamOff2)
    event[beamOff2].daughter1( iRec);

  int auxiRad1 = int(auxevent1.append(auxrad1));
  int auxiEmt1 = int(auxevent1.append(auxemt1));
  auxevent1[iRadBef].statusNeg();
  auxevent1[iRadBef].daughters( auxiRad1, auxiEmt1);
  int auxiRec1 = auxevent1.append(rec);
  auxevent1[iRecBef].mothers( auxiRec1, auxiRec1);
  auxevent1[auxiRec1].mothers( iRecMot1V, iRecMot2V);

  int auxiRad2 = int(auxevent2.append(auxrad2));
  int auxiEmt2 = int(auxevent2.append(auxemt2));
  auxevent2[iRadBef].statusNeg();
  auxevent2[iRadBef].daughters( auxiRad2, auxiEmt2);
  int auxiRec2 = auxevent2.append(rec);
  auxevent2[iRecBef].mothers( auxiRec2, auxiRec2);
  auxevent2[auxiRec1].mothers( iRecMot1V, iRecMot2V);

  if ( nEmissions == 2 && !split->useForBranching) swap(iRad,iEmt);

  bool doVeto = false;
  bool doMECreject = false;
  if ( nEmissions != 2) {

    // Check momenta.
    if ( !validMomentum( rad.p(), idRad, 1)
      || !validMomentum( emt.p(), idEmt, 1)
      || !validMomentum( rec.p(), event[iRecBef].id(), -1) )
      physical = false;

    bool inResonance = (partonSystemsPtr->getInA(iSysSel) == 0) ? true : false;
    doVeto = (( canVetoEmission && userHooksPtr->doVetoFSREmission(
                evSizeOld,event,iSysSel,inResonance) )
          ||  ( canMergeFirst && mergingHooksPtr->doVetoEmission(
                event) ));

    double xm = 2. * pRec.e() / (beamAPtr->e() + beamBPtr->e());
    // Check that beam still has leftover momentum.
    BeamParticle& beamRec = (isrType == 1) ? *beamAPtr : *beamBPtr;
    if ( particleDataPtr->colType(event[iRecBef].id()) != 0
      && beamRec.size() > 0) {
      double xOld = beamRec[iSysSelRec].x();
      beamRec[iSysSelRec].iPos(iRec);
      beamRec[iSysSelRec].x(xm);
      if (beamRec.xMax(-1) < 0.0) {
        if (!trial) infoPtr->errorMsg("Warning in DireTimes::branch_FI: "
          "used up beam momentum; discard splitting.");
        physical = false;
      }
      // Restore old beams.
      beamRec[iSysSelRec].iPos(iRecBef);
      beamRec[iSysSelRec].x(xOld);
    }

    // Apply ME correction if necessary.
    bool isHardSystem = partonSystemsPtr->getSystemOf(iRadBef,true) == 0
                     && partonSystemsPtr->getSystemOf(iRecBef,true) == 0;
    if ( isHardSystem && physical && doMEcorrections
      //&& pT2 > pT2minMECs && checkSIJ(event,1.)) {
      && pT2 > pT2minMECs && checkSIJ(event,Q2minMECs)) {
      // Temporarily update parton systems.
      partonSystemsPtr->replace(iSysSel, iRadBef, iRad);
      partonSystemsPtr->addOut(iSysSel, iEmt);
      partonSystemsPtr->replace(iSysSelRec, iRecBef, iRec);

      if ( nFinalMaxMECs < 0
        || nFinalMaxMECs > partonSystemsPtr->sizeOut(iSysSel))
        doMECreject = applyMEC (event, split,
          createvector<Event>(auxevent1)(auxevent2));

      // Undo update of parton systems.
      partonSystemsPtr->replace(iSysSel, iRad, iRadBef);
      partonSystemsPtr->replace(iSysSelRec, iRec, iRecBef);
      partonSystemsPtr->popBackOut(iSysSel);
    }


    // Just update dipoles and beams.
    if ( physical && !doVeto && !trial && !doMECreject) updateAfterFI(
      iSysSel, iSysSelRec,
      event, iRadBef, iRecBef, iRad, iEmt, iRec, flavour, colType, pTsel, xm);

  // Heavy particle 1->2 decay for "second step" in 1->3 splitting.
  } else {

    // Check momenta.
    if ( !validMomentum( emt.p(), idEmt, 1)
      || !validMomentum( rec.p(), event[iRecBef].id(), -1))
      physical = false;

    int iRadOld = int(event.size())-3;
    int iEmtOld = int(event.size())-2;
    int iRecOld = int(event.size())-1;

    // Swap emitted and radiator indices.
    swap(iRadOld,iEmtOld);

    if (!split->useForBranching) {
      // Flavours already fixed by 1->3 kernel.
      idEmt        = -flavourSave;
      idRad        =  flavourSave;
      // Colour tags for new particles in branching.
      if (idEmt < 0) {
        colEmt  = 0;
        acolEmt = event[iEmtOld].acol();
        colRad  = event[iEmtOld].col();
        acolRad = 0;
      } else {
        colEmt  = event[iEmtOld].col();
        acolEmt = 0;
        colRad  = 0;
        acolRad = event[iEmtOld].acol();
      }
    // Already correctly read id and colors from SplitInfo object.
    } else {
      idRad   = split->radAft()->id;
      idEmt   = split->emtAft2()->id;
      colRad  = split->radAft()->col;
      acolRad = split->radAft()->acol;
      colEmt  = split->emtAft2()->col;
      acolEmt = split->emtAft2()->acol;
    }

    // Get particle masses.
    m2Bef = m2ai;
    m2r   = particleDataPtr->isResonance(idRad)
              && idRad == event[iRadBef].id()
          ? getMass(idRad,3,event[iRadBef].mCalc())
          : (abs(idRad) < 6 || idRad == 21 || idRad == 22)
          ? getMass(idRad,2)
          : getMass(idRad,1);
    m2e   = (abs(idEmt) < 6 || idEmt == 21 || idEmt == 22)
          ? getMass(idEmt,2) : getMass(idEmt,1);

    if (split->useForBranching) {
      m2r = split->kinematics()->m2RadAft;
      m2e = split->kinematics()->m2EmtAft2;
    }

    // Construct FF dipole momentum.
    Vec4 pa1(event[iEmtOld].p());
    Vec4 pb(event[iRecOld].p());
    q.p(pa1 + pb);
    q2 = q.m2Calc();

    // Calculate CS variables.
    m2Emt      = m2e;
    m2Rad      = m2e;

    if (split->useForBranching) {
      m2Rad = split->kinematics()->m2RadAft;
      m2Emt = split->kinematics()->m2EmtAft2;
    }

    zCS        = xa;
    double yCS = (m2ai - m2Emt - m2Rad) / (m2ai - m2Emt - m2Rad + 2.*pa1*pb);

    // Calculate derived variables.
    sij  = yCS * (q2 - m2s) + (1.-yCS)*(m2Rad+m2Emt);
    zbar = (q2-sij-m2s) / bABC(q2,sij,m2s)
                * (zCS - m2s/gABC(q2,sij,m2s)
                       *(sij + m2Rad - m2Emt)/(q2-sij-m2s));
    kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2Rad - zbar*m2Emt;

    // Not possible to construct kinematics if kT2 < 0.0
    if (kT2 < 0.) {
      infoPtr->errorMsg("Warning in DireTimes::branch_FI: Reject state "
        "with kinematically forbidden kT^2.");
      physical = false;
    }

    // NaN kT2 can happen for a 1->3 splitting in which the g->QQ~ produces
    // massive quarks Q.
    if (physical && (kT2!=kT2 || abs(kT2-kT2) > 1e5) ) {
      infoPtr->errorMsg("Warning in DireTimes::branch_FI: Reject state "
        "with not-a-number kT^2 for branching " + name);
      physical = false;
    }

    // Update dipoles and beams.
    double xm = 2. * event[iRecOld].e() / (beamAPtr->e() + beamBPtr->e());

    // Check that beam still has leftover momentum.
    BeamParticle& beamRec = (isrType == 1) ? *beamAPtr : *beamBPtr;
    if ( particleDataPtr->colType(event[iRecOld].id()) != 0
      && beamRec.size() > 0) {
      double xOld = beamRec[iSysSelRec].x();
      beamRec[iSysSelRec].iPos(iRec);
      beamRec[iSysSelRec].x(xm);
      if (beamRec.xMax(-1) < 0.0) {
        if (!trial) infoPtr->errorMsg("Warning in DireTimes::branch_FI: "
          "used up beam momentum; discard splitting.");
        physical = false;
      }
      // Restore old beams.
      beamRec[iSysSelRec].iPos(iRecBef);
      beamRec[iSysSelRec].x(xOld);
    }

    if ( physical && !trial ) updateAfterFI( iSysSel, iSysSelRec, event,
      iRadBef, iRecBef, iRad, iEmt, iRec, flavour, colType, pTsel, xm);

    // Boost the transverse momentum vector into the lab frame.
    // Construct left-over dipole momentum by momentum conservation.
    pij.p(q-pb);

    // Set up transverse momentum vector by using two perpendicular
    // four-vectors.
    pTvecs = getTwoPerpendicular(pb, pij);
    kTmom.p( sqrt(kT2)*sin(phiX)*pTvecs.first
           + sqrt(kT2)*cos(phiX)*pTvecs.second);

    // Construct new radiator momentum.
    pRad.p( zbar * (gABC(q2,sij,m2s)*pij - sij*pb) / bABC(q2,sij,m2s)
              + (m2Rad+kT2) / (zbar*bABC(q2,sij,m2s))
                * (pb - m2s/gABC(q2,sij,m2s)*pij)
              + kTmom);

    // Contruct the emission momentum by momentum conservation.
    pEmt.p(q-pRad-pb);

    // Recoiler unchanged.
    pRec.p(event[iRecOld].p());

    // Check invariants.
    if ( false ) {
      Vec4 pa(pRad), pk(pRec), pj(emt.p()), pi(pEmt);
      double saix(2.*pa*pi), sakx(2.*pa*pk), sajx(2.*pa*pj), sikx(2.*pi*pk),
             sjkx(2.*pj*pk), sijx(2.*pi*pj);
      double pptt = (sajx+sijx)*(sakx+sikx)/( sakx + sikx + sjkx);
      double ssaaii = saix;
      double zzaa = sakx / ( sakx + sikx + sjkx );
      double xxaa = sakx / ( sakx + sikx );

      if ( physical &&
          (abs(pptt-pT2) > 1e-5 || abs(ssaaii-sai) > 1e-5 ||
           abs(zzaa-z) > 1e-5   || abs(xxaa-xa) > 1e-5) ){

        cout << scientific << setprecision(8);
        cout << "Error in branch_FI: Invariant masses after branching do not "
             << "match chosen values." << endl;
        cout << "Chosen:    "
             << " Q2 " << (event[iRadBef].p()-event[iRecBef].p()).m2Calc()
             << " pT2 " << pT2
             << " sai " << sai
             << " za " << z
             << " xa " << xa << endl;
        cout << "Generated: "
             << " Q2 " << -sakx + saix + sajx +sijx -sikx - sjkx
             << " pT2 " << pptt
             << " sai " << ssaaii
             << " za " << zzaa
             << " xa " << xxaa << endl;
        physical = false;
      }
    }

    // Check momenta.
    if ( !validMomentum( pRad, idRad, 1)
      || !validMomentum( pEmt, idEmt, 1)
      || !validMomentum( pRec, event[iRecOld].id(), -1) )
      physical = false;

    // Check that beam still has leftover momentum.
    if ( particleDataPtr->colType(event[iRecOld].id()) != 0
      && beamRec.size() > 0) {
      double xOld = beamRec[iSysSelRec].x();
      int iOld    = beamRec[iSysSelRec].iPos();
      double xNew = 2.*pRec.e() / (beamAPtr->e() + beamBPtr->e());
      int iNew    = event.append(Particle(event[iRecOld].id(), -53, 0, 0,
        iRecOld, iRecOld, event[iRecOld].col(), event[iRecOld].acol(), pRec,
        0., pTsel));
      beamRec[iSysSelRec].iPos(iNew);
      beamRec[iSysSelRec].x(xNew);
      if (beamRec.xMax(-1) < 0.0) {
        if (!trial) infoPtr->errorMsg("Warning in DireTimes::branch_FI: "
          "used up beam momentum; discard splitting.");
        physical = false;
      }
      // Restore old beams.
      event.popBack();
      beamRec[iSysSelRec].iPos(iOld);
      beamRec[iSysSelRec].x(xOld);
    }

    // Update bookkeeping
    if (physical) {

      // Define new particles from dipole branching.
      Particle rad2 = Particle(idRad, 51, iEmtOld, 0, 0, 0,
        //colRad, acolRad, pRad, sqrt(m2r), pTsel);
        colRad, acolRad, pRad, sqrt(m2Rad), pTsel);
      Particle emt2 = Particle(idEmt, 51, iEmtOld, 0, 0, 0,
        //colEmt, acolEmt, pEmt, sqrt(m2e), pTsel);
        colEmt, acolEmt, pEmt, sqrt(m2Emt), pTsel);
      Particle rec2 = Particle(event[iRecOld].id(), -53, 0, 0, iRecOld,
        iRecOld, event[iRecOld].col(), event[iRecOld].acol(), pRec, 0., pTsel);

      // Shower may occur at a displaced vertex.
      if (!trial) {
        if (event[iEmtOld].hasVertex()) {
          rad2.vProd( event[iEmtOld].vProd() );
          emt2.vProd( event[iEmtOld].vProd() );
        }
        if (event[iRecOld].hasVertex()) rec2.vProd( event[iRecOld].vProd() );
      }

      // Put new particles into the event record.
      // Mark original dipole partons as branched and set daughters/mothers.
      iRad = event.append(rad2);
      iEmt = event.append(emt2);
      event[iEmtOld].statusNeg();
      event[iEmtOld].daughters( iRad, iEmt);

      iRecMot1V    = event[iRecOld].mother1();
      iRecMot2V    = event[iRecOld].mother2();
      iRec = event.append(rec2);
      event[iRecOld].statusNeg();
      event[iRecOld].mothers( iRec, iRec);
      event[iRec].mothers( iRecMot1V, iRecMot2V);
      if (iRecMot1V == beamOff1) event[beamOff1].daughter1( iRec);
      if (iRecMot1V == beamOff2) event[beamOff2].daughter1( iRec);

      // Update dipoles and beams.
      if (!trial) {
        int colTypeNow = colType;
        xm = 2. * event[iRec].e() / (beamAPtr->e() + beamBPtr->e());
        updateAfterFI( iSysSel, iSysSelRec, event, iEmtOld, iRecOld,
          iRad, iEmt, iRec, flavourSave, colTypeNow, pTsel, xm);
      }
    }
  }

  physical = physical && !doVeto;

  // Temporarily set the daughters in the beams to zero, to
  // allow mother-daughter relation checks.
  if (iSysSelRec > 0) {
    if (iRecMot1V == beamOff1) event[beamOff1].daughter1( iRec);
    if (iRecMot1V == beamOff2) event[beamOff2].daughter1( iRec);
  }

  // Check if mother-daughter relations are correctly set. Check only
  // possible if no MPI are present.
  if ( physical && !trial && !doMECreject && !validMotherDaughter(event)) {
    infoPtr->errorMsg("Error in DireTimes::branch_FI: Mother-daughter "
                      "relations after branching not valid.");
    physical = false;
  }

  // Restore correct daughters in the beams.
  if (iSysSelRec > 0) {
    if (iRecMot1V == beamOff1) event[beamOff1].daughter1(iBeam1Dau1);
    if (iRecMot1V == beamOff2) event[beamOff2].daughter1(iBeam2Dau1);
  }

  // Allow veto of branching. If so restore event record to before emission.
  if ( !physical || doMECreject) {

    event.popBack( event.size() - evSizeOld);
    event[iRadBef].status( iRadStatusV);
    event[iRadBef].daughters( iRadDau1V, iRadDau2V);
    event[iRecBef].mothers( iRecMot1V, iRecMot2V);
    if (iSysSelRec == 0 && iRecMot1V == beamOff1)
      event[beamOff1].daughter1( ev1Dau1V);
    if (iSysSelRec == 0 && iRecMot1V == beamOff2)
      event[beamOff2].daughter1( ev2Dau1V);

    // This case is identical to the case where the probability to accept the
    // emission was indeed zero all along. In this case, neither
    // acceptProbability nor rejectProbability would have been filled. Thus,
    // remove the relevant entries from the weight container!
    if (!trial) {
      for ( unordered_map<string, multimap<double,double> >::iterator
        it = rejectProbability.begin(); it != rejectProbability.end(); ++it){
        weights->eraseAcceptWeight(pT2, it->first);
        weights->eraseRejectWeight(pT2, it->first);
      }
    }

    if (!trial && doMECreject) {
      weights->calcWeight(pT2, false, true);
      weights->reset();
     // Clear accept/reject weights.
      for ( unordered_map<string, multimap<double,double> >::iterator
        it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
        it->second.clear();
      for ( unordered_map<string, map<double,double> >::iterator
        it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
        it->second.clear();
    }

    return false;
  }

  // Store positions of new particles.
  if (trial) split->storePosAfter( iRad, iRec, iEmt,
    (nEmissions < 2) ? 0 : iEmt2);

  // Set shower weight.
  if (!trial) {
    if (!doTrialNow) {
      weights->calcWeight(pT2);
      weights->reset();

      // Store positions of new soft particles.
      direInfoPtr->updateSoftPos( iRadBef, iEmt );
      direInfoPtr->updateSoftPosIfMatch( iRecBef, iRec );
      if (nEmissions > 1) direInfoPtr->addSoftPos( iEmt2 );
      updateDipoles(event, iSysSel);
    }

    // Clear accept/reject weights.
    for ( unordered_map<string, multimap<double,double> >::iterator
      it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
      it->second.clear();
    for ( unordered_map<string, map<double,double> >::iterator
      it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
      it->second.clear();
  }

  // Done.
  return true;

}

//--------------------------------------------------------------------------

void DireTimes::updateAfterFI( int iSysSelNow, int iSysSelRec,
  Event& event, int iRadBef, int iRecBef, int iRad, int iEmt, int iRec,
  int, int colType, double pTsel, double xNew) {

  bool hasDipSel   = (dipSel != 0);
  int isrType      = (hasDipSel) ? dipSel->isrType : event[iRec].mother1();
  bool inResonance = (partonSystemsPtr->getInA(iSysSelNow)==0) ? true : false;
  int idEmt        = event[iEmt].id();
  vector<int> iDipEndCorr;

  // For initial-state recoiler also update beam and sHat info.
  BeamParticle& beamRec = (isrType == 1) ? *beamAPtr : *beamBPtr;
  double xOld = beamRec[iSysSelRec].x();
  beamRec[iSysSelRec].iPos( iRec);
  beamRec[iSysSelRec].x( xNew);
  partonSystemsPtr->setSHat( iSysSelRec,
  partonSystemsPtr->getSHat(iSysSelRec) * xNew / xOld);

  if (particleDataPtr->colType(idEmt) == 2) {

    if (hasDipSel) {
      dipSel->iRadiator  = iRad;
      dipSel->iRecoiler  = iEmt;
      dipSel->systemRec  = iSysSel;
      dipSel->pTmax      = pTsel;
      dipSel->MEtype     = 0;
    }

    for (int i = 0; i < int(dipEnd.size()); ++i) {
      if (dipEnd[i].iRadiator == iRecBef && dipEnd[i].iRecoiler == iRadBef
        && dipEnd[i].colType != 0) {
        dipEnd[i].iRadiator = iRec;
        dipEnd[i].iRecoiler = iEmt;
        dipEnd[i].MEtype = 0;
        // Strive to match colour to anticolour inside closed system.
        if ( dipEnd[i].colType * colType > 0)
          dipEnd[i].iRecoiler = iRad;
        dipEnd[i].pTmax = pTsel;
        iDipEndCorr.push_back(i);
      }
    }
    int colTypeNow = (colType > 0) ? 2 : -2 ;
    // When recoiler was uncoloured particle, in resonance decays,
    // assign recoil to coloured particle.
    int iRecMod = iRec;
    if (recoilToColoured && inResonance && event[iRec].col() == 0
      && event[iRec].acol() == 0) iRecMod = iRad;

    if (appendDipole(event, iEmt, iRecMod, pTsel, colTypeNow, 0, 0, 0, isrType,
          iSysSelNow, 0, -1, 0, false, dipEnd)) {
      iDipEndCorr.push_back(dipEnd.size()-1);
      // Set dipole mass properties.
      DireTimesEnd& dip1 = dipEnd.back();
      dip1.systemRec = iSysSelRec;
    }
    if (appendDipole(event, iEmt, iRad, pTsel, -colTypeNow, 0, 0, 0, 0,
          iSysSelNow, 0, -1, 0, false, dipEnd)) {
      iDipEndCorr.push_back(dipEnd.size()-1);
      // Set dipole mass properties.
      DireTimesEnd& dip2 = dipEnd.back();
      dip2.systemRec = iSysSelRec;
    }

  // Gluon branching to q qbar: update current dipole and other of gluon.
  } else if (particleDataPtr->colType(idEmt) != 0) {

    // Update dipoles for second step in 1->3 splitting.
    if ( splittingsPtr->nEmissions(splittingSelName) == 2 ){
      for (int i = 0; i < int(dipEnd.size()); ++i) {

        DireTimesEnd& dip = dipEnd[i];

        if ( dip.iRadiator == iRecBef ) dip.iRadiator = iRec;
        if ( dip.iRecoiler == iRecBef ) dip.iRecoiler = iRec;
        if ( dip.iRadiator == iRadBef ) {
          if (dip.colType > 0)
            dip.iRadiator = (event[iEmt].id() > 0) ? iEmt : iRad;
          if (dip.colType < 0)
            dip.iRadiator = (event[iEmt].id() < 0) ? iEmt : iRad;

          if (abs(dip.colType) == 2
            && event[dip.iRadiator].id()    > 0
            && event[dip.iRadiator].idAbs() < 10)
            dip.colType = abs(dip.colType)/2;
          if (abs(dip.colType) == 2
            && event[dip.iRadiator].id()    < 0
            && event[dip.iRadiator].idAbs() < 10)
            dip.colType = -abs(dip.colType)/2;
          iDipEndCorr.push_back(i);
        }

        if ( dip.iRecoiler == iRadBef ) {
          if (dip.colType > 0)
            dip.iRecoiler = (event[iEmt].id() < 0) ? iEmt : iRad;
          if (dip.colType < 0)
            dip.iRecoiler = (event[iEmt].id() > 0) ? iEmt : iRad;

          if (abs(dip.colType) == 2) dipEnd[i].colType /= 2;

          if (abs(dip.colType) == 1
            && event[dip.iRadiator].id()    > 0
            && event[dip.iRadiator].idAbs() < 10)
            dip.colType = 1;

          if (abs(dip.colType) == 1
            && event[dip.iRadiator].id()    < 0
            && event[dip.iRadiator].idAbs() < 10)
            dip.colType = -1;
          iDipEndCorr.push_back(i);
        }
      }
    }

    for (int i = 0; i < int(dipEnd.size()); ++i) {
      // Nothing to be done if dipole end has already been updated.
      if ( find(iDipEndCorr.begin(), iDipEndCorr.end(), i)
        != iDipEndCorr.end() ) continue;
      // Strive to match colour to anticolour inside closed system.
      if ( dipEnd[i].iRecoiler == iRadBef
        && dipEnd[i].colType * colType < 0 ) {
        dipEnd[i].iRecoiler = iEmt;
      }
      if (dipEnd[i].iRadiator == iRadBef && abs(dipEnd[i].colType) == 2) {
        dipEnd[i].colType /= 2;

        if (hasDipSel && &dipEnd[i] == dipSel) dipEnd[i].iRadiator  = iEmt;
        else                                   dipEnd[i].iRadiator  = iRad;
        if (hasDipSel && &dipEnd[i] == dipSel) dipEnd[i].iRecoiler  = iRec;

        dipEnd[i].colType = (event[dipEnd[i].iRadiator].id() > 0)
                    ? abs(dipEnd[i].colType) : -abs(dipEnd[i].colType);

        iDipEndCorr.push_back(i);

        if (dipEnd[i].system != dipEnd[i].systemRec) continue;
        dipEnd[i].MEtype = 0;
        if (hasDipSel && &dipEnd[i] == dipSel) dipEnd[i].iMEpartner = iRad;
        else                      dipEnd[i].iMEpartner = iEmt;
      }
    }

    // Nothing to be done if dipole end has already been updated.
    bool updateSel=true;
    for (int j = 0; j < int(iDipEndCorr.size()); ++j)
      if ( hasDipSel && &dipEnd[iDipEndCorr[j]] == dipSel) updateSel = false;

    if (hasDipSel) {
      if (updateSel) {
        dipSel->iRadiator = iEmt;
        dipSel->iRecoiler = iRec;
      }
      // Always update the production pT.
      dipSel->pTmax     = pTsel;
    }

  } else {

    int iRadOld = (hasDipSel) ? dipSel->iRadiator : iRadBef;
    int iRecOld = (hasDipSel) ? dipSel->iRecoiler : iRecBef;

    // Just update old radiator/recoiler to current outgoing particles.
    for (int i = 0; i < int(dipEnd.size()); ++i) {
      DireTimesEnd& dip = dipEnd[i];
      // Update radiator-recoiler end.
      if ( dip.iRecoiler == iRecOld && dip.iRadiator == iRadOld ) {
        dip.iRadiator = iRad;
        dip.iRecoiler = iRec;
        dip.pTmax  = pTsel;
        iDipEndCorr.push_back(i);
      }
      // Update recoiler-radiator end.
      if ( dip.iRecoiler == iRadOld && dip.iRadiator == iRecOld ) {
        dip.iRadiator = iRec;
        dip.iRecoiler = iRad;
        dip.pTmax  = pTsel;
        iDipEndCorr.push_back(i);
      }
    }
  }

  // Now update other dipoles that also involved the radiator or recoiler.
  // Note: For 1->3 splittings, this step has already been done earlier!
  for (int i = 0; i < int(dipEnd.size()); ++i) {
    // Nothing to be done if dipole end has already been updated.
    if ( find(iDipEndCorr.begin(), iDipEndCorr.end(), i)
      != iDipEndCorr.end() ) continue;
    if (dipEnd[i].iRadiator  == iRadBef) dipEnd[i].iRadiator  = iRad;
    if (dipEnd[i].iRecoiler  == iRadBef) dipEnd[i].iRecoiler  = iRad;
    if (dipEnd[i].iMEpartner == iRadBef) dipEnd[i].iMEpartner = iRad;
    if (dipEnd[i].iRadiator  == iRecBef) dipEnd[i].iRadiator  = iRec;
    if (dipEnd[i].iRecoiler  == iRecBef) dipEnd[i].iRecoiler  = iRec;
    if (dipEnd[i].iMEpartner == iRecBef) dipEnd[i].iMEpartner = iRec;
  }

  // Now update or construct new dipoles if the radiator or emission allow
  // for new types of emissions. Careful: do not include initial recoiler
  // as potential radiator.
  vector<pair<int, int> > rad_rec (createvector< pair<int,int> >
    (make_pair(iRad,iEmt))
    (make_pair(iEmt,iRec))
    (make_pair(iEmt,iRad))
    (make_pair(iRad,iRec)));
  for (int i=0; i < int(rad_rec.size()); ++i) {
    int iRadNow = rad_rec[i].first;
    int iRecNow = rad_rec[i].second;
    // Now check if a new dipole end a-b should be added:
    // First check if the dipole end is already existing.
    vector<int> iDip;
    for (int j = 0; j < int(dipEnd.size()); ++j)
      if ( dipEnd[j].iRadiator == iRadNow
        && dipEnd[j].iRecoiler == iRecNow )
        iDip.push_back(j);
    // If the dipole end exists, attempt to update the dipole end (a)
    // for the current a-b dipole.
    if ( int(iDip.size()) > 0) for (int j = 0; j < int(iDip.size()); ++j)
      updateAllowedEmissions(event, &dipEnd[iDip[j]]);
    // If no dipole exists and idEmtAfter != 0, create new dipole end (a).
    else appendDipole( event, iRadNow, iRecNow, pTsel, 0, 0, 0, 0, 0,
      iSysSelNow, -1, -1, 0, false, dipEnd);
  }

  // Copy or set lifetime for new final state.
  if (event[iRad].id() == event[iRadBef].id()) {
    event[iRad].tau( event[iRadBef].tau() );
  } else {
    event[iRad].tau( event[iRad].tau0() * rndmPtr->exp() );
    event[iEmt].tau( event[iEmt].tau0() * rndmPtr->exp() );
  }
  event[iRec].tau( event[iRecBef].tau() );

  // Finally update the list of all partons in all systems.
  partonSystemsPtr->replace(iSysSel, iRadBef, iRad);
  partonSystemsPtr->addOut(iSysSel, iEmt);
  partonSystemsPtr->replace(iSysSelRec, iRecBef, iRec);

  // Loop through final state of system to find possible new dipole ends.
  for (int i = 0; i < partonSystemsPtr->sizeOut(iSysSelNow); ++i) {
    int iNow = partonSystemsPtr->getOut( iSysSelNow, i);
    if (event[iNow].isFinal() && event[iNow].scale() > 0.) {
      // Find dipole end formed by colour index.
      int colTag = event[iNow].col();
      if (doQCDshower && colTag > 0) setupQCDdip( iSysSelNow, i,  colTag,  1,
        event, false, true);
      // Find dipole end formed by anticolour index.
      int acolTag = event[iNow].acol();
      if (doQCDshower && acolTag > 0) setupQCDdip( iSysSelNow, i, acolTag, -1,
        event, false, true);
      // Now find non-QCD dipoles and/or update the existing dipoles.
      getGenDip( iSysSelNow, i, iNow, event, false, dipEnd);
    }
    // Setup decay dipoles.
    if (doDecaysAsShower && event[iNow].isResonance())
      setupDecayDip( iSysSelNow, iNow, event, dipEnd);
  }

  // Now update all dipoles.
  dipSel = 0;
  updateDipoles(event, iSysSel);

  // Bookkeep shower-induced resonances.
  if ( direInfoPtr->isRes(iRadBef) &&
    event[iRadBef].id() != event[iRad].id() )
    direInfoPtr->removeResPos(iRadBef);
  if ( particleDataPtr->isResonance(event[iRad].id()) ) {
    if ( direInfoPtr->isRes(iRadBef) )
         direInfoPtr->updateResPos(iRadBef,iRad);
  }
  if ( direInfoPtr->isRes(iRecBef) )
    direInfoPtr->updateResPos(iRecBef,iRec);
  if ( particleDataPtr->isResonance(event[iEmt].id()) )
    direInfoPtr->addResPos(iEmt);


  // Done.
}

//--------------------------------------------------------------------------

pair < Vec4, Vec4 > DireTimes::decayWithOnshellRec( double zCS, double yCS,
  double phi, double m2Rec, double m2RadAft, double m2EmtAft,
  Vec4 pRadBef, Vec4 pRecBef ) {

  // Construct FF dipole momentum.
  Vec4 q(pRadBef + pRecBef);
  double q2 = q.m2Calc();

  // Calculate derived variables.
  double sij  = yCS * (q2 - m2Rec) + (1.-yCS)*(m2RadAft+m2EmtAft);
  double zbar = (q2-sij-m2Rec) / bABC(q2,sij,m2Rec)
             * (zCS - m2Rec/gABC(q2,sij,m2Rec)
                     *(sij + m2RadAft - m2EmtAft)/(q2-sij-m2Rec));
  double kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2RadAft - zbar*m2EmtAft;

  bool physical = true;
  if (kT2 < 0. || isnan(kT2)) physical = false;
  if (abs(kT2) < 1e-9) kT2 = 0.0;

  // Construct left-over dipole momentum by momentum conservation.
  Vec4 pij(q-pRecBef);

  // Set up transverse momentum vector by using two perpendicular 4-vectors.
  pair <Vec4, Vec4> pTvecs = getTwoPerpendicular(pRecBef, pij);
  Vec4 kTmom( sqrt(kT2)*sin(phi)*pTvecs.first
            + sqrt(kT2)*cos(phi)*pTvecs.second);

  // Construct new radiator momentum.
  Vec4 pRad( zbar * (gABC(q2,sij,m2Rec)*pij - sij*pRecBef)/bABC(q2,sij,m2Rec)
            + (m2RadAft+kT2) / (zbar*bABC(q2,sij,m2Rec))
              * (pRecBef - m2Rec/gABC(q2,sij,m2Rec)*pij)
            + kTmom);

  // Contruct the emission momentum by momentum conservation.
  Vec4 pEmt(q-pRad-pRecBef);
  // Recoiler unchanged.
  Vec4 pRec(pRecBef);

  // Store and return.
  pair < Vec4, Vec4 > ret;
  if (physical) ret = make_pair(pRad,pEmt);
  return ret;

}

//--------------------------------------------------------------------------

pair <Vec4, Vec4> DireTimes::decayWithOffshellRec( double zCS, double yCS,
  double phi, double m2RadBef, double m2RadAft, double m2EmtAft,
  Vec4 pRadBef, Vec4 pRecBef ) {

  Vec4 q(pRadBef + pRecBef);
  double q2 = q.m2Calc();
  // Calculate derived variables.
  double sij  = yCS * (q2 - m2RadBef) + (1.-yCS)*(m2RadAft+m2EmtAft);
  double zbar = (q2-sij-m2RadBef) / bABC(q2,sij,m2RadBef)
              * (zCS - m2RadBef/gABC(q2,sij,m2RadBef)
                     *(sij + m2RadAft - m2EmtAft)/(q2-sij-m2RadBef));
  double kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2RadAft - zbar*m2EmtAft;

  // Not possible to construct kinematics if kT2 < 0.0
  bool physical = true;
  if (kT2 < 0. || isnan(kT2)) physical = false;

  // Construct left-over dipole momentum by momentum conservation.
  Vec4 pij(q-pRadBef);

  // Set up transverse momentum vector by using two perpendicular four-vectors.
  pair < Vec4, Vec4> pTvecs = getTwoPerpendicular(pRadBef, pij);
  Vec4 kTmom( sqrt(kT2)*sin(phi)*pTvecs.first
            + sqrt(kT2)*cos(phi)*pTvecs.second);

  // Construct new radiator momentum.
  Vec4 pRec2( zbar * (gABC(q2,sij,m2RadBef)*pij - sij*pRadBef)
              / bABC(q2,sij,m2RadBef)
            + (m2RadAft+kT2) / (zbar*bABC(q2,sij,m2RadBef))
              * (pRadBef - m2RadBef/gABC(q2,sij,m2RadBef)*pij)
            + kTmom);

  // Contruct the emission momentum by momentum conservation.
  Vec4 pRec1(q-pRec2-pRadBef);

  // Store and return.
  pair < Vec4, Vec4 > ret;
  if (physical) ret = make_pair(pRec1,pRec2);
  return ret;

}

//--------------------------------------------------------------------------

pair <Event, pair<int,int> > DireTimes::clustered_internal( const Event& state,
  int iRad, int iEmt, int iRec, string name ) {

  if (name.compare("Dire_fsr_qcd_1->21&1") == 0 && state[iRad].id() == 21)
    swap(iRad,iEmt);
  if (name.compare("Dire_fsr_qed_1->22&1") == 0 && state[iRad].id() == 22)
    swap(iRad,iEmt);
  if (name.compare("Dire_fsr_qed_11->22&11") == 0 && state[iRad].id() == 22)
    swap(iRad,iEmt);

  // Flags for type of radiation
  int radType = state[iRad].isFinal() ? 1 : -1;
  int recType = state[iRec].isFinal() ? 1 : -1;

  // Construct the clustered event
  Event NewEvent = Event();
  NewEvent.init("(hard process-modified)", particleDataPtr);
  NewEvent.clear();
  // Copy all unchanged particles to NewEvent
  for (int i = 0; i < state.size(); ++i)
    if ( i != iRad && i != iRec && i != iEmt )
      NewEvent.append( state[i] );

  // Copy all the junctions one by one
  for (int i = 0; i < state.sizeJunction(); ++i)
    NewEvent.appendJunction( state.getJunction(i) );
  // Find an appropriate scale for the hard process
  double mu = infoPtr->QFac();
  // Initialise scales for new event
  NewEvent.saveSize();
  NewEvent.saveJunctionSize();
  NewEvent.scale(mu);
  NewEvent.scaleSecond(mu);

  // Set properties of radiator/recoiler after the clustering
  // Recoiler properties will be unchanged
  Particle RecBefore = Particle( state[iRec] );
  RecBefore.setEvtPtr(&NewEvent);
  RecBefore.daughters(0,0);

  // Find flavour of radiator before splitting
  int radID = splits[name]->radBefID(state[iRad].id(), state[iEmt].id());
  int recID = state[iRec].id();
  Particle RadBefore = Particle( state[iRad] );
  RadBefore.setEvtPtr(&NewEvent);
  RadBefore.id(radID);
  RadBefore.daughters(0,0);
  // Put dummy values for colours
  RadBefore.cols(RecBefore.acol(),RecBefore.col());

  // Reset status if the reclustered radiator is a resonance.
  if ( particleDataPtr->isResonance(radID) && radType == 1)
    RadBefore.status(22);

  // Put mass for radiator and recoiler
  double radMass = particleDataPtr->m0(radID);
  double recMass = particleDataPtr->m0(recID);
  if (radType == 1 ) RadBefore.m(radMass);
  else RadBefore.m(0.0);
  if (recType == 1 ) RecBefore.m(recMass);
  else RecBefore.m(0.0);

  if ( particleDataPtr->isResonance(radID) && radType == 1 )
    RadBefore.m( (state[iRad].p()+state[iEmt].p()).mCalc() );
  if ( particleDataPtr->isResonance(recID) && radType == 1 )
    RadBefore.m(  state[iRec].mCalc() );

  // Construct momenta and  colours of clustered particles.
  bool validState = false;
  if ( state[iRec].isFinal())
    validState = cluster_FF(state,iRad,iEmt,iRec,radID,RadBefore,RecBefore);
  else
    validState = cluster_FI(state,iRad,iEmt,iRec,radID,RadBefore,RecBefore);

  // Put some dummy production scales for RecBefore, RadBefore
  RecBefore.scale(mu);
  RadBefore.scale(mu);

  // Append new recoiler and find new radiator colour
  NewEvent.append(RecBefore);

  // Assign the correct colour to re-clustered radiator.
  pair<int,int> cols = splits[name]->radBefCols( state[iRad].col(),
    state[iRad].acol(), state[iEmt].col(), state[iEmt].acol());
  RadBefore.cols( cols.first, cols.second );

  // Build the clustered event
  Event outState = Event();
  outState.init("(hard process-modified)", particleDataPtr);
  outState.clear();

  // Clustering might not be possible due to phase space constraints.
  if (!validState) { return make_pair(outState, make_pair(0,0)); }

  // Copy system and incoming beam particles to outState
  for (int i = 0; i < 3; ++i)
    outState.append( NewEvent[i] );
  // Copy all the junctions one by one
  for (int i = 0; i < state.sizeJunction(); ++i)
    outState.appendJunction( state.getJunction(i) );
  // Initialise scales for new event
  outState.saveSize();
  outState.saveJunctionSize();
  outState.scale(mu);
  outState.scaleSecond(mu);
  bool radAppended = false;
  bool recAppended = false;
  int size = int(outState.size());
  // Save position of radiator in new event record
  int radPos(0), recPos(0);

  // Append first incoming particle
  if ( RecBefore.mother1() == 1) {
    recPos = outState.append( RecBefore );
    recAppended = true;
  } else if ( RadBefore.mother1() == 1 ) {
    radPos = outState.append( RadBefore );
    radAppended = true;
  } else {
    // Find second incoming in input event
    int in1 = 0;
    for(int i=0; i < int(state.size()); ++i)
      if (state[i].mother1() == 1) in1 =i;
    outState.append( state[in1] );
    size++;
  }
  // Append second incoming particle
  if ( RecBefore.mother1() == 2) {
    recPos = outState.append( RecBefore );
    recAppended = true;
  } else if ( RadBefore.mother1() == 2 ) {
    radPos = outState.append( RadBefore );
    radAppended = true;
  } else {
    // Find second incoming in input event
    int in2 = 0;
    for(int i=0; i < int(state.size()); ++i)
      if (state[i].mother1() == 2) in2 =i;

    outState.append( state[in2] );
    size++;
  }

  // Append new recoiler if not done already
  if (!recAppended && !RecBefore.isFinal()) {
    recAppended = true;
    recPos = outState.append( RecBefore);
  }
  // Append new radiator if not done already
  if (!radAppended && !RadBefore.isFinal()) {
    radAppended = true;
    radPos = outState.append( RadBefore);
  }

  // Append intermediate particle
  // (careful not to append reclustered recoiler)
  for (int i = 0; i < int(NewEvent.size()-1); ++i) {
    if (NewEvent[i].status() != -22) continue;
    if ( NewEvent[i].daughter1() == NewEvent[i].daughter2()
      && NewEvent[i].daughter1() > 0) continue;
    outState.append( NewEvent[i] );
  }

  // Append final state particles, resonances first
  for (int i = 0; i < int(NewEvent.size()-1); ++i)
    if (NewEvent[i].status() == 22) {
      outState.append( NewEvent[i] );
    }
  // Then start appending partons
  if (!radAppended && RadBefore.statusAbs() == 22)
    radPos = outState.append(RadBefore);
  if (!recAppended)
    recPos= outState.append(RecBefore);
  if (!radAppended && RadBefore.statusAbs() != 22)
    radPos = outState.append(RadBefore);
  // Then partons (not reclustered recoiler)
  for(int i = 0; i < int(NewEvent.size()-1); ++i)
    if ( NewEvent[i].status()  != 22
      && NewEvent[i].colType() != 0
      && NewEvent[i].isFinal())
      outState.append( NewEvent[i] );
  // Then the rest
  for(int i = 0; i < int(NewEvent.size()-1); ++i)
    if ( NewEvent[i].status() != 22
      && NewEvent[i].colType() == 0
      && NewEvent[i].isFinal() )
      outState.append( NewEvent[i]);

  // Find intermediate and respective daughters
  vector<int> PosIntermediate;
  vector<int> PosDaughter1;
  vector<int> PosDaughter2;
  for(int i=0; i < int(outState.size()); ++i) {
    if (outState[i].status() == -22) {
      PosIntermediate.push_back(i);
      int d1 = outState[i].daughter1();
      int d2 = outState[i].daughter2();
      // Find daughters in output state
      int daughter1 = FindParticle( state[d1], outState);
      int daughter2 = FindParticle( state[d2], outState);

      // If both daughters found, done
      // Else put first final particle as first daughter
      // and last final particle as second daughter
      if (daughter1 > 0)
        PosDaughter1.push_back( daughter1);
      else {
        daughter1 = 0;
        while(!outState[daughter1].isFinal() ) daughter1++;
        PosDaughter1.push_back( daughter1);
      }
      if (daughter2 > 0)
        PosDaughter2.push_back( daughter2);
      else {
        daughter2 = outState.size()-1;
        while(!outState[daughter2].isFinal() ) daughter2--;
        PosDaughter2.push_back( daughter2);
      }
    }
  }

  int iOut1 = 0;
  while(!outState[iOut1].isFinal() ) iOut1++;
  int iOut2 = outState.size()-1;
  while(!outState[iOut2].isFinal() ) iOut2--;

  // Set daughters and mothers
  for(int i=0; i < int(PosIntermediate.size()); ++i) {
    outState[PosIntermediate[i]].daughters(PosDaughter1[i],PosDaughter2[i]);
    outState[PosDaughter1[i]].mother1(PosIntermediate[i]);
    outState[PosDaughter2[i]].mother1(PosIntermediate[i]);
    outState[PosDaughter1[i]].mother2(0);
    outState[PosDaughter2[i]].mother2(0);
  }

  // Force outgoing particles to be part of hard process.
  for ( int i=0; i < int(outState.size()); ++i) {
    if (outState[i].isFinal()) outState[i].status(23);
  }

  // Find range of final state partons
  int minParFinal = int(outState.size());
  int maxParFinal = 0;
  for(int i=0; i < int(outState.size()); ++i)
    if (outState[i].mother1() == 3 && outState[i].mother2() == 4) {
      minParFinal = min(i,minParFinal);
      maxParFinal = max(i,maxParFinal);
    }

  if (minParFinal == maxParFinal) maxParFinal = 0;
  outState[3].daughters(minParFinal,maxParFinal);
  outState[4].daughters(minParFinal,maxParFinal);

  // Update event properties
  outState.saveSize();
  outState.saveJunctionSize();

  // Almost there...
  // If an intermediate coloured parton exists which was directly
  // colour connected to the radiator before the splitting, and the
  // radiator before and after the splitting had only one colour, problems
  // will arise since the colour of the radiator will be changed, whereas
  // the intermediate parton still has the old colour. In effect, this
  // means that when setting up a event for trial showering, one colour will
  // be free.
  // Hence, check for an intermediate coloured triplet resonance has been
  // colour-connected to the "old" radiator.
  // Find resonance
  int iColRes = 0;
  if ( radType == -1 && state[iRad].colType() == 1) {
      // Find resonance connected to initial colour
      for(int i=0; i < int(state.size()); ++i)
        if ( i != iRad && i != iEmt && i != iRec
          && state[i].status() == -22
          && state[i].col() == state[iRad].col() )
          iColRes = i;
  } else if ( radType == -1 && state[iRad].colType() == -1) {
      // Find resonance connected to initial anticolour
      for(int i=0; i < int(state.size()); ++i)
        if ( i != iRad && i != iEmt && i != iRec
          && state[i].status() == -22
          && state[i].acol() == state[iRad].acol() )
          iColRes = i;
  } else if ( radType == 1 && state[iRad].colType() == 1) {
      // Find resonance connected to final state colour
      for(int i=0; i < int(state.size()); ++i)
        if ( i != iRad && i != iEmt && i != iRec
          && state[i].status() == -22
          && state[i].col() == state[iRad].col() )
          iColRes = i;
  } else if ( radType == 1 && state[iRad].colType() == -1) {
      // Find resonance connected to final state anticolour
      for(int i=0; i < int(state.size()); ++i)
        if ( i != iRad && i != iEmt && i != iRec
          && state[i].status() == -22
          && state[i].acol() == state[iRad].acol() )
          iColRes = i;
  }

  if (iColRes > 0) {
    // Now find this resonance in the reclustered state
    int iColResNow = FindParticle( state[iColRes], outState);

    // Find reclustered radiator colours
    int radCol = outState[radPos].col();
    int radAcl = outState[radPos].acol();
    // Find resonance radiator colours
    int resCol = outState[iColResNow].col();
    int resAcl = outState[iColResNow].acol();
    // Check if any of the reclustered radiators colours match the resonance
    bool matchesRes =  (radCol > 0
                          && ( radCol == resCol || radCol == resAcl))
                    || (radAcl > 0
                          && ( radAcl == resCol || radAcl == resAcl));

    // If a resonance has been found, but no colours match, change
    // the colour of the resonance
    if (!matchesRes && iColResNow > 0) {
      if ( radType == -1 && outState[radPos].colType() == 1)
        outState[iColResNow].col(radCol);
      else if ( radType ==-1 && outState[radPos].colType() ==-1)
        outState[iColResNow].acol(radAcl);
      else if ( radType == 1 && outState[radPos].colType() == 1)
        outState[iColResNow].col(radCol);
      else if ( radType == 1 && outState[radPos].colType() ==-1)
        outState[iColResNow].acol(radAcl);
    }

    // If a resonance has been found, but no colours match, and the position
    // of the resonance in the event record has been changed, update the
    // radiator mother
    if (!matchesRes && iColResNow > 0 && iColRes != iColResNow)
      outState[radPos].mother1(iColResNow);

  }

  // Force outgoing particles to be part of hard process.
  for ( int i=0; i < int(outState.size()); ++i) {
    if (outState[i].isFinal()) outState[i].status(23);
  }

  // Force incoming to be part of the hard process.
  outState[3].status(-21);
  outState[4].status(-21);

  // Now check event.
  for ( int i = 0; i < outState.size(); ++i) {
    if ( outState[i].status() == 23 && particleDataPtr->
         isResonance(outState[i].id())) outState[i].status(22);
  }

  // Check if the state is valid. If not, return empty state.
  if (!validEvent( outState, true )) { outState.clear(); }

  // Done
  return make_pair(outState, make_pair(radPos,recPos));

}

bool DireTimes::cluster_FF( const Event& state,
  int iRad, int iEmt, int iRec, int idRadBef, Particle& radBef,
  Particle& recBef ) {

  // Calculate CS variables.
  double pT2    = pT2_FF(state[iRad], state[iEmt], state[iRec]);
  double z      = z_FF(state[iRad], state[iEmt], state[iRec]);

  // Get particle masses.
  double m2Bef = ( abs(idRadBef) < 6 || idRadBef == 21 || idRadBef == 22)
               ? getMass(idRadBef,2)
               : (idRadBef == state[iRad].id())
                  ? getMass(idRadBef,3,state[iRad].mCalc())
                  : getMass(idRadBef,2);

  // Set resonance mass to virtuality.
  if ( particleDataPtr->isResonance(idRadBef)
    && !particleDataPtr->isResonance(state[iRad].id())
    && !particleDataPtr->isResonance(state[iEmt].id()) )
    m2Bef = (state[iRad].p()+state[iEmt].p()).m2Calc();

  double m2r   = state[iRad].p().m2Calc();
  double m2e   = state[iEmt].p().m2Calc();
  double m2s   = state[iRec].p().m2Calc();

  double m2D = 2.*state[iRad].p()*state[iRec].p()
             + 2.*state[iRad].p()*state[iEmt].p()
             + 2.*state[iRec].p()*state[iEmt].p();
  double Q2 = m2D + (m2Bef - m2r - m2e);

  // Get dipole 4-momentum.
  Vec4 q(state[iRad].p() + state[iEmt].p() + state[iRec].p());
  double q2 = q.m2Calc();

  int type     = 1;
  // Upate type if this is a massive splitting.
  if ( m2Bef > TINYMASS || m2r > TINYMASS || m2s > TINYMASS
    || m2e > TINYMASS ) type = 2;

  // Check phase space constraints.
  if ( !inAllowedPhasespace(1, z, pT2, Q2, q2, 0.0, type, m2Bef, m2r, m2s,
        m2e) ) return false;

  // Calculate derived variables.
  double sij  = (state[iRad].p() + state[iEmt].p()).m2Calc();

  // Now construct the new recoiler momentum in the lab frame.
  Vec4 pRec(state[iRec].p());
  Vec4 pRecBef( (pRec - (q*pRec)/q2 * q)
            * sqrt(lABC(q2,m2Bef,m2s)/lABC(q2,sij,m2s))
            + q * (q2+m2s-m2Bef)/(2.*q2) );

  // Get momentum of radiator my momentum conservation.
  Vec4 pRadBef(q-pRecBef);

  radBef.p(pRadBef);
  recBef.p(pRecBef);
  radBef.m(sqrtpos(m2Bef));
  recBef.m(sqrtpos(m2s));

  // Done
  return true;
}

bool DireTimes::cluster_FI( const Event& state,
  int iRad, int iEmt, int iRec, int idRadBef, Particle& radBef,
  Particle& recBef ) {

  // Calculate CS variables.
  double pT2    = pT2_FI(state[iRad], state[iEmt], state[iRec]);
  double z      = z_FI(state[iRad], state[iEmt], state[iRec]);

  // Get particle masses.
  double m2Bef = ( abs(idRadBef) < 6 || idRadBef == 21 || idRadBef == 22)
               ? getMass(idRadBef,2)
               : (idRadBef == state[iRad].id())
                  ? getMass(idRadBef,3,state[iRad].mCalc())
                  : getMass(idRadBef,2);

  // Set resonance mass to virtuality.
  if ( particleDataPtr->isResonance(idRadBef)
    && !particleDataPtr->isResonance(state[iRad].id())
    && !particleDataPtr->isResonance(state[iEmt].id()) )
    m2Bef = (state[iRad].p()+state[iEmt].p()).m2Calc();

  double m2r   = state[iRad].p().m2Calc();
  double m2e   = state[iEmt].p().m2Calc();
  double m2s   = state[iRec].p().m2Calc();

  double m2D  = -2.*state[iRad].p()*state[iEmt].p()
             + 2.*state[iRad].p()*state[iRec].p()
             + 2.*state[iRec].p()*state[iEmt].p();
  double Q2 = m2D;

  // Get dipole 4-momentum.
  Vec4 q(-state[iRad].p() - state[iEmt].p() + state[iRec].p());
  double q2 = q.m2Calc();

  vector<int> iOther;
  for (int i=3; i < state.size(); ++i)
    if (i != iRad && i != iEmt && i != iRec) iOther.push_back(i);

  if ( (iOther.size() == 1 || abs(q.m2Calc()) < TINYMASS) && m2Bef > 0.) {

    // Get momentum of radiator my momentum conservation.
    Vec4 pRadBef( state[iRad].p() + state[iEmt].p() );

    // Ensure that radiator is on mass-shell
    double errMass = abs(pRadBef.mCalc() - sqrt(m2Bef))
      / max( 1.0, pRadBef.e());
    if ( errMass > mTolErr*1e-2 ) {
      double deltam2 = pRadBef.m2Calc() - m2Bef;
      pRadBef.e(sqrtpos(pow2(pRadBef.e()) - deltam2));
    }
    Vec4 pRecBef( state[iRec].p() );
    // Ensure that recoiler is on mass-shell
    errMass = abs(pRecBef.mCalc() - sqrt(m2s)) / max( 1.0, pRecBef.e());
    if ( errMass > mTolErr*1e-2 ) {
      double deltam2 = pRecBef.m2Calc() - m2s;
      pRecBef.e(sqrtpos(pow2(pRecBef.e()) - deltam2));
    }
    radBef.p(pRadBef);
    recBef.p(pRecBef);
    radBef.m(sqrtpos(m2Bef));
    recBef.m(sqrtpos(m2s));
    // Done
    return true;
  }

  // Check phase space constraints.
  double xNew = 2. * state[iRec].e() / state[0].m();

  // Calculate CS variables.
  double kappa2 =  pT2/Q2;
  double xCS = 1 - kappa2/(1.-z);
  double xCDST = xCS*( 1. + (m2Bef-m2r-m2e)/Q2 );
  double xOld  = xNew * xCDST;

  int type     = -1;
  // Upate type if this is a massive splitting.
  if ( m2Bef > TINYMASS || m2r > TINYMASS || m2s > TINYMASS
    || m2e > TINYMASS ) type = -2;

  bool   hasPDFrec = (state[iRec].colType() != 0
    || (state[iRec].isLepton() && settingsPtr->flag("PDF:lepton")));
  double xMin = (hasPDFrec) ? xOld : 0.;
  if ( !inAllowedPhasespace( 1, z, pT2, Q2, q2, xMin, type, m2Bef, m2r, m2s,
        m2e) ) return false;

  Vec4 pRad(state[iRad].p());
  Vec4 pRec(state[iRec].p());
  Vec4 pEmt(state[iEmt].p());
  Vec4 qpar(q.px()+pRad.px()+pEmt.px(), q.py()+pRad.py()+pEmt.py(), q.pz(),
            q.e());
  double qpar2 = qpar.m2Calc();
  double pT2ijt = pow2(pRad.px()+pEmt.px()) + pow2(pRad.py()+pEmt.py());

  // Calculate derived variables.
  double sij  = (state[iRad].p() + state[iEmt].p()).m2Calc();

  // Now construct the new recoiler momentum in the lab frame.
  Vec4 pRecBef( (pRec - (q*pRec)/qpar2 * qpar)
            * sqrt( (lABC(q2,m2Bef,m2s) - 4.*m2s*pT2ijt)
                   /(lABC(q2,sij,m2s)   - 4.*m2s*pT2ijt))
            + qpar * (q2+m2s-m2Bef)/(2.*qpar2) );

  // Get momentum of radiator my momentum conservation.
  Vec4 pRadBef(-q+pRecBef);

  // Ensure that radiator is on mass-shell
  double errMass = abs(pRadBef.mCalc() - sqrt(m2Bef)) / max( 1.0, pRadBef.e());
  if ( errMass > mTolErr*1e-2 ) {
    double deltam2 = pRadBef.m2Calc() - m2Bef;
    pRadBef.e(sqrtpos(pow2(pRadBef.e()) - deltam2));
  }
  // Ensure that recoiler is on mass-shell
  errMass = abs(pRecBef.mCalc() - sqrt(m2s)) / max( 1.0, pRecBef.e());
  if ( errMass > mTolErr*1e-2 ) {
    double deltam2 = pRecBef.m2Calc() - m2s;
    pRecBef.e(sqrtpos(pow2(pRecBef.e()) - deltam2));
  }

  radBef.p(pRadBef);
  recBef.p(pRecBef);
  radBef.m(sqrtpos(m2Bef));
  recBef.m(sqrtpos(m2s));

  // Done
  return true;
}

//--------------------------------------------------------------------------

// Function to in the input event find a particle with quantum
// numbers matching those of the input particle
// IN  Particle : Particle to be searched for
//     Event    : Event to be searched in
// OUT int      : > 0 : Position of matching particle in event
//                < 0 : No match in event

int DireTimes::FindParticle( const Particle& particle, const Event& event,
  bool checkStatus ) {

  int index = -1;

  for ( int i = int(event.size()) - 1; i > 0; --i )
    if ( event[i].id()         == particle.id()
      && event[i].colType()    == particle.colType()
      && event[i].chargeType() == particle.chargeType()
      && event[i].col()        == particle.col()
      && event[i].acol()       == particle.acol()
      && event[i].charge()     == particle.charge() ) {
      index = i;
      break;
    }

  if ( checkStatus && event[index].status() != particle.status() )
    index = -1;

  return index;
}

//--------------------------------------------------------------------------

double DireTimes::pT2_FF ( const Particle& rad, const Particle& emt,
  const Particle& rec) {
  double sij = 2.*rad.p()*emt.p();
  double sik = 2.*rad.p()*rec.p();
  double sjk = 2.*rec.p()*emt.p();
  return sij*sjk / (sij+sik+sjk);
}

//--------------------------------------------------------------------------

double DireTimes::pT2_FI ( const Particle& rad, const Particle& emt,
  const Particle& rec) {
  double sij =  2.*rad.p()*emt.p();
  double sai = -2.*rec.p()*rad.p();
  double saj = -2.*rec.p()*emt.p();
  //return sij*saj / (sai+saj) * (sij+saj+sai) / (sai+saj) ;
  double t = sij*saj / (sai+saj) * (sij+saj+sai) / (sai+saj) ;
  if (sij+saj+sai < 1e-5 && abs(sij+saj+sai) < 1e-5) t = sij;
  return t;

}

//--------------------------------------------------------------------------

double DireTimes::z_FF ( const Particle& rad, const Particle& emt,
  const Particle& rec) {
  double sij = 2.*rad.p()*emt.p();
  double sik = 2.*rad.p()*rec.p();
  double sjk = 2.*rec.p()*emt.p();
  return (sij + sik) / (sij+sik+sjk);
}

double DireTimes::z_FF_fromVec ( const Vec4& rad, const Vec4& emt,
  const Vec4& rec) {
  double sij = 2.*rad*emt;
  double sik = 2.*rad*rec;
  double sjk = 2.*rec*emt;
  return (sij + sik) / (sij+sik+sjk);
}

//--------------------------------------------------------------------------

double DireTimes::z_FI ( const Particle& rad, const Particle& emt,
  const Particle& rec) {
  double sai = -2.*rec.p()*rad.p();
  double saj = -2.*rec.p()*emt.p();
  return sai / (sai+saj);
}

//--------------------------------------------------------------------------

double DireTimes::m2dip_FF ( const Particle& rad, const Particle& emt,
  const Particle& rec) {
  double sij = 2.*rad.p()*emt.p();
  double sik = 2.*rad.p()*rec.p();
  double sjk = 2.*rec.p()*emt.p();
  return (sij+sik+sjk);
}

//--------------------------------------------------------------------------

double DireTimes::m2dip_FI ( const Particle& rad, const Particle& emt,
  const Particle& rec) {
  double sij =  2.*rad.p()*emt.p();
  double sai = -2.*rec.p()*rad.p();
  double saj = -2.*rec.p()*emt.p();
  return -1.*(sij+saj+sai);
}

//--------------------------------------------------------------------------

// From Pythia version 8.218 onwards.
// Return the evolution variable and splitting information. More comments
// in the header.

map<string, double> DireTimes::getStateVariables (const Event& state,
  int rad, int emt, int rec, string name) {
  map<string,double> ret;

  // Kinematical variables.
  if (rad > 0 && emt > 0 && rec > 0) {

    double pT2 = pT2Times ( state[rad], state[emt], state[rec]);
    double z   = zTimes ( state[rad], state[emt], state[rec]);
    ret.insert(make_pair("t",pT2));
    ret.insert(make_pair("tRS",pT2));
    ret.insert(make_pair("scaleAS",pT2));
    ret.insert(make_pair("scaleEM",pT2));
    ret.insert(make_pair("scalePDF",pT2));
    ret.insert(make_pair("z",z));

    // Book-keeping for particle before emission.
    int radBefID
       = (name != "")
       ? (*splittingsPtr)[name]->radBefID(state[rad].id(), state[emt].id())
       : 0;
    pair<int,int> radBefCols
       = (name != "")
       ? (*splittingsPtr)[name]->radBefCols(state[rad].col(),
             state[rad].acol(), state[emt].col(), state[emt].acol())
       : make_pair(0,0);
    ret.insert(make_pair("radBefID", radBefID));
    ret.insert(make_pair("radBefCol", radBefCols.first));
    ret.insert(make_pair("radBefAcol", radBefCols.second));

    int couplingType
       = (name != "")
       ? (*splittingsPtr)[name]->couplingType(state[rad].id(), state[emt].id())
       : -1;
    double couplingValue
       = (name != "")
       ? (*splittingsPtr)[name]->coupling(z,pT2)
       : -1.0;
    ret.insert(make_pair
               ("scaleForCoupling " + std::to_string(couplingType), pT2));
    ret.insert(make_pair("couplingType",couplingType));
    ret.insert(make_pair("couplingValue",couplingValue));

    double m2dip = m2dipTimes ( state[rad], state[emt], state[rec]);
    ret.insert(make_pair("m2dip", m2dip));

  // Variables defining the PS starting scales.
  } else {

    // In this case, insert only dummy information except for PDF scale.
    ret.insert(make_pair("t",0.));
    ret.insert(make_pair("tRS",0.));
    ret.insert(make_pair("scaleAS",0.));
    ret.insert(make_pair("scaleEM",0.));
    ret.insert(make_pair("z",0.));
    ret.insert(make_pair("radBefID", 0));
    ret.insert(make_pair("radBefCol", 0));
    ret.insert(make_pair("radBefAcol", 0));
    ret.insert(make_pair("scaleForCoupling "+std::to_string(-1),0.));
    ret.insert(make_pair("couplingType",-1));
    ret.insert(make_pair("couplingValue",-1.));

    // Find the shower starting scale.
    vector<DireTimesEnd> dipEnds;

    // Loop through final state of system to find possible dipole ends.
    for (int iRad = 0; iRad < state.size(); ++iRad) {

      if ( !state[iRad].isFinal() ) continue;

      // Find dipole end formed by colour index.
      int colTag = state[iRad].col();
      if (colTag > 0) getQCDdip( iRad,  colTag,  1, state, dipEnds);
      // Find dipole end formed by anticolour index.
      int acolTag = state[iRad].acol();
      if (acolTag > 0) getQCDdip( iRad, acolTag, -1, state, dipEnds);
      // Now find non-QCD dipoles and/or update the existing dipoles.
      getGenDip( -1, 0, iRad, state, false, dipEnds);

    }

    // Get x for both beams.
    int in1(3), in2(4);
    double x1 = state[3].pPos() / state[0].m();
    double x2 = state[4].pNeg() / state[0].m();

    // Store invariant masses of all dipole ends.
    stringstream oss;
    for (int iDip = 0; iDip < int(dipEnds.size()); ++iDip) {
      double m2 = abs(2.*state[dipEnds[iDip].iRadiator].p()
                        *state[dipEnds[iDip].iRecoiler].p());
      if ( dipEnds[iDip].iRecoiler == in1) m2 /= x1;
      if ( dipEnds[iDip].iRecoiler == in2) m2 /= x2;
      oss.str("");
      oss << "scalePDF-" << dipEnds[iDip].iRadiator
           << "-"        << dipEnds[iDip].iRecoiler;
      ret.insert(make_pair(oss.str(),m2));
    }

  }

  return ret;
}

//--------------------------------------------------------------------------

// Compute splitting probability.

// From Pythia version 8.215 onwards.
double DireTimes::getSplittingProb( const Event& state, int iRad,
  int iEmt, int iRec, string name) {

  // Get kernel order.
  int order = atoi( (char*)name.substr( name.find("-",0)+1,
                                        name.size() ).c_str() );
  name=name.substr( 0, name.size()-2);

  // Do nothing if kernel says so, e.g. to avoid infinite loops
  // if the kernel uses the History class.
  if ( splits[name]->splitInfo.extras.find("unitKernel")
    != splits[name]->splitInfo.extras.end() ) return 1.;

  double z     = zTimes(state[iRad], state[iEmt], state[iRec]);
  double pT2   = pT2Times(state[iRad], state[iEmt], state[iRec]);
  double m2D   = (state[iRec].isFinal())
               ? abs( 2.*state[iEmt].p()*state[iRad].p()
                    + 2.*state[iRec].p()*state[iRad].p()
                    + 2.*state[iEmt].p()*state[iRec].p())
               : abs( 2.*state[iEmt].p()*state[iRad].p()
                    - 2.*state[iRec].p()*state[iRad].p()
                    - 2.*state[iEmt].p()*state[iRec].p());

  // Disallow below cut-off.
  if ( pT2cut(state[iEmt].id()) > pT2) return 0.;
  if ( !splits[name]->aboveCutoff( pT2, state[iRad], state[iRec], 0,
        partonSystemsPtr)) return 0.;

  int idRadBef = splits[name]->radBefID(state[iRad].id(), state[iEmt].id());

  double m2Bef = ( abs(idRadBef) < 6 || idRadBef == 21 || idRadBef == 22)
               ? getMass(idRadBef,2)
               : (idRadBef == state[iRad].id())
                  ? getMass(idRadBef,3,state[iRad].mCalc())
                  : getMass(idRadBef,2);
  double m2r   = state[iRad].p().m2Calc();
  double m2e   = state[iEmt].p().m2Calc();
  double m2s   = state[iRec].p().m2Calc();
  int type     = state[iRec].isFinal() ? 1 : -1;

  // Upate type if this is a massive splitting.
  if (type == 1 && (m2Bef > TINYMASS || m2r > TINYMASS || m2s > TINYMASS
    || m2e > TINYMASS) ) type = 2;
  if (type ==-1 && (m2Bef > TINYMASS || m2r > TINYMASS || m2s > TINYMASS
    || m2e > TINYMASS) ) type =-2;

  // Recalculate the kinematicaly available dipole mass.
  int massSign  = (type > 0) ? 1 : -1;
  double q2     = (state[iRec].p() + massSign*state[iRad].p()
                                   + massSign*state[iEmt].p()).m2Calc();
  double Q2     = m2D;

  double kappa2 =  pT2/Q2;
  double xCS    = 1 - kappa2/(1.-z);
  double xNew   = (type > 0) ? 0.0 : 2.*state[iRec].e() / state[0].m();
  double xOld   = (type > 0) ? 0.0 : xCS * xNew;

  // Check phase space constraints.
  bool   hasPDFrec = type < 0 && (state[iRec].colType() != 0
    || (state[iRec].isLepton() && settingsPtr->flag("PDF:lepton")));
  double xMin = (hasPDFrec) ? xOld : 0.;

  if (abs(q2) < 1e-5) {
    q2 = Q2 = (state[iRad].p() + state[iEmt].p()).m2Calc();
    xMin = 0.;
    type = 0;
  }

  if (name.compare("Dire_fsr_qcd_1->21&1") == 0) swap(iRad,iEmt);
  if (name.compare("Dire_fsr_qed_1->22&1") == 0) swap(iRad,iEmt);
  if (name.compare("Dire_fsr_qed_11->22&11") == 0) swap(iRad,iEmt);

  // Disallow below cut-off.
  //if ( pT2cut(state[iEmt].id()) > pT2) return 0.;

  if ( !inAllowedPhasespace( 1, z, pT2, Q2, q2, xMin, type, m2Bef, m2r, m2s,
        m2e) ) return 0.0;

  // Calculate splitting probability.
  double p = 0.;

  // Get phi angle.
  pair<Vec4, Vec4> pTdirection = getTwoPerpendicular(state[iRec].p(),
    state[iRad].p()+state[iEmt].p());
  double px= -pTdirection.first*state[iRad].p();
  double py= -pTdirection.second*state[iRad].p();
  double kT2 = pow2(px)+pow2(py);
  double phi1 = atan2(px/sqrt(kT2), py/sqrt(kT2));
  if (phi1 < 0.) phi1 = 2.*M_PI+phi1;

  pair <Event, pair<int,int> > born(clustered_internal(
    state, iRad, iEmt, iRec, name ));

  int nEmissions = splittingsPtr->nEmissions(name);
  double m2dipBef = abs(2.*born.first[born.second.first].p()
                        *born.first[born.second.second].p());
  splits[name]->splitInfo.save();
  splits[name]->splitInfo.clear();
  splits[name]->splitInfo.storeInfo(name, type, 0, 0, 0,
    born.second.first, born.second.second, born.first,
    state[iEmt].id(), state[iRad].id(),
    nEmissions, m2dipBef, pT2, pT2, z, phi1, m2Bef, m2s,
    (nEmissions == 1 ? m2r : 0.0),(nEmissions == 1 ? m2e : 0.0),
    0.0, 0.0, 0.0, 0.0, xOld, xNew);
  splits[name]->splitInfo.setSiblings(DireSingleColChain());

  // Get splitting probability.
  unordered_map < string, double > kernels;
  // Get complete kernel.
  if (splits[name]->calc(born.first, order) )
    kernels = splits[name]->getKernelVals();
  if ( kernels.find("base") != kernels.end() ) p += kernels["base"];
  // Reset again.
  splits[name]->splitInfo.clear();
  splits[name]->splitInfo.restore();

  // Multiply with 1/pT^2. Note: No additional Jacobian factors, since for our
  // choice of variables, we always have
  // Jacobian_{mass to CS} * Jacobian_{CS to DIRE} * Propagator = 1/pT2
  p *= 1. / pT2;

  // Make splitting probability positive if ME corrections are available.
  bool hasME = pT2 > pT2minMECs && doMEcorrections && weights->hasME(state);
  if (hasME) p = abs(p);

  if (!dryrun && splits[name]->hasMECAft(state, pT2)) p *= KERNEL_HEADROOM;

  double mecover=1.;
  int nFinal = 0;
  for (int i=0; i < state.size(); ++i) if (state[i].isFinal()) nFinal++;
  double xDau = (type>0) ? 1. : xOld;
  if (!dryrun)
    mecover = splits[name]->
      overhead(m2dipBef*xDau, state[iRad].id(), nFinal-1);
  p *= mecover;

  // Note: The additional factor 1/xCS for rescaling the initial flux is NOT
  // included, so that we can apply PDF ratios [x1 f(x1)] / [x0 f(x0) ] later.

  return p;

}


bool DireTimes::allowedSplitting( const Event& state, int iRad, int iEmt) {

  bool isAP = state[iRad].id() < 0;
  int idRad = state[iRad].id();
  int idEmt = state[iEmt].id();

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();

  int colShared = (colRad  > 0 && colRad == acolEmt) ? colRad
                : (acolRad > 0 && colEmt == acolRad) ? colEmt : 0;

  // Only consider final-state emissions.
  if ( state[iRad].status() < 0) return false;

  // Gluon emission is allowed.
  if (idEmt == 21 && colShared > 0)
    return true;

  // Gluon emission is allowed.
  if (idRad == 21 && colShared > 0)
    return true;

  // Gluon branching to quarks is allowed.
  if ( idEmt == -idRad
    && state[iEmt].colType() != 0
    && ( (isAP && acolRad != colEmt) || (!isAP && acolEmt != colRad) ) )
    return true;

  // Photon emission.
  // Photon emission from quarks.
  if ( idEmt == 22 && abs(idRad) < 10)
    return true;
  if ( idRad == 22 && abs(idEmt) < 10)
    return true;

  // Photon emission from charged leptons.
  if ( idEmt == 22 && (abs(idRad) == 11 || abs(idRad) == 13
    || abs(idRad) == 15))
    return true;
  if ( idRad == 22 && (abs(idEmt) == 11 || abs(idEmt) == 13
    || abs(idEmt) == 15))
    return true;

  // Z-boson emission.
  // Z-boson emission from quarks.
  if ( idEmt == 23 && abs(idRad) < 10)
    return true;
  if ( idRad == 23 && abs(idEmt) < 10)
    return true;

  // Z-boson emission from charged leptons.
  if ( idEmt == 22 && (abs(idRad) == 11 || abs(idRad) == 13
    || abs(idRad) == 15))
    return true;
  if ( idRad == 22 && (abs(idEmt) == 11 || abs(idEmt) == 13
    || abs(idEmt) == 15))
    return true;

  // Photon splitting.
  // Photon branching to quarks is allowed.
  if ( idEmt == -idRad && state[iEmt].colType() != 0 && colShared > 0 )
    return true;

  // Photon branching to leptons is allowed.
  if ( idEmt == -idRad && state[iEmt].colType() == 0 )
    return true;

  // W-boson splitting.
  // W-boson branching to quarks is allowed.
  int emtSign = (idEmt>0) ? 1 : -1;
  int radSign = (idRad>0) ? 1 : -1;
  if ( emtSign*( abs(idEmt)+1 ) == -idRad
    && state[iEmt].colType() != 0 && colShared > 0 )
    return true;
  if ( idEmt == -radSign*(abs(idRad)+1)
    && state[iEmt].colType() != 0 && colShared > 0 )
    return true;

  // H-boson splitting.
  // H-boson branching to photons is allowed.
  if ( idEmt == idRad && idRad == 22 )
    return true;

  return false;

}

//--------------------------------------------------------------------------

vector<int> DireTimes::getRecoilers( const Event& state, int iRad, int iEmt,
  string name) {
  // List of recoilers.
  return splits[name]->recPositions(state, iRad, iEmt);
}

//--------------------------------------------------------------------------

Event DireTimes::makeHardEvent( int iSys, const Event& state, bool isProcess) {

  bool hasSystems = !isProcess && partonSystemsPtr->sizeSys() > 0;
  int sizeSys     = (hasSystems) ? partonSystemsPtr->sizeSys() : 1;

  Event event = Event();
  event.clear();
  event.init( "(hard process-modified)", particleDataPtr );

  int in1 = 0;
  for ( int i = state.size()-1; i > 0; --i)
    if ( state[i].mother1() == 1 && state[i].mother2() == 0
      && (!hasSystems || partonSystemsPtr->getSystemOf(i,true) == iSys))
      {in1 = i; break;}
  if (in1 == 0) in1 = partonSystemsPtr->getInA(iSys);

  int in2 = 0;
  for ( int i = state.size()-1; i > 0; --i)
    if ( state[i].mother1() == 2 && state[i].mother2() == 0
      && (!hasSystems || partonSystemsPtr->getSystemOf(i,true) == iSys))
      {in2 = i; break;}
  if (in2 == 0) in2 = partonSystemsPtr->getInB(iSys);

  // Try to find incoming particle in other systems, i.e. if the current
  // system arose from a resonance decay.
  bool resonantIncoming = false;
  if ( in1 == 0 && in2 == 0 ) {
    int iParentInOther = 0;
    int nSys = partonSystemsPtr->sizeAll(iSys);
    for (int iInSys = 0; iInSys < nSys; ++iInSys){
      int iNow = partonSystemsPtr->getAll(iSys,iInSys);
      for (int iOtherSys = 0; iOtherSys < sizeSys; ++iOtherSys){
        if (iOtherSys == iSys) continue;
        int nOtherSys = partonSystemsPtr->sizeAll(iOtherSys);
        for (int iInOtherSys = 0; iInOtherSys < nOtherSys; ++iInOtherSys){
          int iOtherNow = partonSystemsPtr->getAll(iOtherSys,iInOtherSys);
          if (state[iNow].isAncestor(iOtherNow)) {
            iParentInOther = iOtherNow;
          }
        }
      }
    }
    in1 = iParentInOther;
    if (iParentInOther) resonantIncoming = true;
  }

  int i1(-1), i2(-1);
  if ( !hasSystems || partonSystemsPtr->hasInAB(iSys) ) {
    event.append(state[0]);
    i1 = event.append(state[1]);
    event[i1].mothers(0,0);
    i2 = event.append(state[2]);
    event[i2].mothers(0,0);
  }

  // Attach the first incoming particle.
  int inNow(-1);
  if (resonantIncoming) {
    event.append(state[0]);
    if (in1 > 0) {
      // Append one time (as beams)
      int beamA = event.append(state[state[in1].mother1()]);
      event[beamA].mothers(0,0);
      event[beamA].status(-12);
      int beamB = event.append(state[state[in1].mother2()]);
      event[beamB].mothers(0,0);
      event[beamB].status(-12);
      // Append second time as incoming particles.
      i1 = event.append(state[in1]);
      event[i1].mothers(beamA,0);
      event[i1].daughters(0,0);
      event[i1].status(-21);
      event[beamA].daughters(i1,0);
    }
  } else {
    if (in1 > 0) {
      inNow = event.append(state[in1]);
      event[inNow].mothers(i1,0);
      event[inNow].status(-21);
      event[i1].daughters(inNow,0);
      i1= inNow;
    }
    if (in2 > -1) {
      inNow = event.append(state[in2]);
      event[inNow].mothers(i2,0);
      event[inNow].status(-21);
      event[i2].daughters(inNow,0);
      i2 = inNow;
    }
  }

  int sizeOld = event.size();

  for ( int i = 0; i < state.size(); ++i) {
    // Careful when builing the sub-events: A particle that is currently
    // intermediate in one system could be the progenitor of another
    // system, i.e. when resonance decays are present. In this case, the
    // intermediate particle in the current system should be final.
    bool isFin   = state[i].isFinal();
    bool isInSys = (partonSystemsPtr->getSystemOf(i) == iSys);

    if (isFin && (!hasSystems || isInSys) ) {
      int iN = event.append(state[i]);
      event[iN].daughters(0,0);
      event[iN].mothers(max(0,i1),max(0,i2));
      int status = (state[i].statusAbs() == 22) ? state[i].statusAbs() : 23;
      if ( particleDataPtr->isResonance(state[i].id()) ) status = 22;
      event[iN].status(status);
    }
  }

  int iDaughter2 = ((unsigned int)event.size() > (unsigned int)sizeOld + 1) ?
    event.size()-1 : 0;

  // Set daughters of initial particles.
  if (i1 > -1 && event.size() > sizeOld)
    event[i1].daughters(sizeOld, iDaughter2);
  if (i2 > -1 && event.size() > sizeOld)
    event[i2].daughters(sizeOld, iDaughter2);
  return event;

}

//-------------------------------------------------------------------------

// Check colour/flavour correctness of state.

bool DireTimes::validMomentum( const Vec4& p, int id, int status) {

  // Check for NaNs or INFs.
  if (isnan(p) || isinf(p)) return false;

  // Check if particles is on mass shell
  double mNow = (status < 0) ? 0.
              : ((abs(id) < 6) ? getMass(id,2) : getMass(id,1));

  if (status < 0 && useMassiveBeams
    && (abs(id) == 11 || abs(id) == 13 || abs(id) > 900000))
    mNow = getMass(id,1);

  mNow = sqrt(mNow);
  // Do not check on-shell condition for massive intermediate (!)
  // resonances. Assuming all non-SM particles are heavy here!
  if ( particleDataPtr->isResonance(id) || abs(id) > 22) mNow = p.mCalc();
  double errMass = abs(p.mCalc() - mNow) / max( 1.0, p.e());
  if ( errMass > mTolErr ) return false;

  // Check for negative energies.
  if ( p.e() < 0. ) return false;

  // Done
  return true;

}

//-------------------------------------------------------------------------

// Check colour/flavour correctness of state.
bool DireTimes::validEvent( const Event& state, bool isProcess,
  int iSysCheck ) {

  bool validColour  = true;
  bool validCharge  = true;
  bool validMomenta = true;

  bool hasSystems = !isProcess && partonSystemsPtr->sizeSys() > 0;
  int sizeSys     = (hasSystems) ? partonSystemsPtr->sizeSys() : 1;

  // Check for NaNs or INFs.
  for ( int i = 0; i < state.size(); ++i)
    if (isnan(state[i].p()) || isinf(state[i].p())) return false;

  Event event = Event();
  event.clear();
  event.init( "(hard process-modified)", particleDataPtr );

  for (int iSys = 0; iSys < sizeSys; ++iSys) {

    if (iSysCheck >= 0 && iSys != iSysCheck) continue;

    // Done if the state is already broken.
    if (!validColour || !validCharge ) break;

    event.clear();
    event.init( "(hard process-modified)", particleDataPtr );
    event.clear();

    event = makeHardEvent(iSys, state, isProcess);

    // Check if event is coloured
    for ( int i = 0; i < event.size(); ++i)
     // Check colour of quarks
     if ( event[i].isFinal() && event[i].colType() == 1
            // No corresponding anticolour in final state
         && ( FindCol(event[i].col(),vector<int>(1,i),event,1) == 0
            // No corresponding colour in initial state
           && FindCol(event[i].col(),vector<int>(1,i),event,2) == 0 )) {
       validColour = false;
       break;
     // Check anticolour of antiquarks
     } else if ( event[i].isFinal() && event[i].colType() == -1
            // No corresponding colour in final state
         && ( FindCol(event[i].acol(),vector<int>(1,i),event,2) == 0
            // No corresponding anticolour in initial state
           && FindCol(event[i].acol(),vector<int>(1,i),event,1) == 0 )) {
       validColour = false;
       break;
     // No uncontracted colour (anticolour) charge of gluons
     } else if ( event[i].isFinal() && event[i].colType() == 2
            // No corresponding anticolour in final state
         && ( FindCol(event[i].col(),vector<int>(1,i),event,1) == 0
            // No corresponding colour in initial state
           && FindCol(event[i].col(),vector<int>(1,i),event,2) == 0 )
            // No corresponding colour in final state
         && ( FindCol(event[i].acol(),vector<int>(1,i),event,2) == 0
            // No corresponding anticolour in initial state
           && FindCol(event[i].acol(),vector<int>(1,i),event,1) == 0 )) {
       validColour = false;
       break;
     }

    for(int i = 0; i < event.size(); ++i) {
      if ( !event[i].isFinal()
        &&  event[i].status() != -11
        &&  event[i].status() != -12)  {
        if ( event[i].colType() == 1 && event[i].acol()>0) validColour = false;
        if ( event[i].colType() ==-1 && event[i].col() >0) validColour = false;
      }
      if ( event[i].isFinal() ) {
        if ( event[i].colType() == 1 && event[i].acol()>0) validColour = false;
        if ( event[i].colType() ==-1 && event[i].col() >0) validColour = false;
      }
    }

    // Check charge sum in initial and final state
    double initCharge = 0.0;
    for(int i = 0; i < event.size(); ++i)
      if ( !event[i].isFinal()
        &&  event[i].status() != -11
        &&  event[i].status() != -12)
        initCharge += event[i].charge();
    double finalCharge = 0.0;
    for(int i = 0; i < event.size(); ++i)
      if (event[i].isFinal()) finalCharge += event[i].charge();
    if (abs(initCharge-finalCharge) > 1e-12) validCharge = false;

    // Check if particles are on mass shell
    for ( int i = 0; i < event.size(); ++i) {
      if (event[i].statusAbs() < 20) continue;
      validMomenta = validMomenta && validMomentum(event[i].p(),
        event[i].id(), (event[i].isFinal() ? 1 : -1));
    }

    // Check that overall pT is vanishing.
    Vec4 pSum(0.,0.,0.,0.);
    for ( int i = 0; i < event.size(); ++i) {
      //if ( i ==3 || i == 4 )    pSum -= event[i].p();
      if ( event[i].status() == -21
        || event[i].status() == -22 ) pSum -= event[i].p();
      if ( event[i].isFinal() )       pSum += event[i].p();
    }
    if ( abs(pSum.px()) > mTolErr || abs(pSum.py()) > mTolErr)
      validMomenta = false;
    if ( !particleDataPtr->isResonance(event[3].id())
      && event[3].status() == -21
      && (abs(event[3].px()) > mTolErr || abs(event[3].py()) > mTolErr))
      validMomenta = false;
    if ( !particleDataPtr->isResonance(event[4].id())
      && event[4].status() == -21
      && (abs(event[4].px()) > mTolErr || abs(event[4].py()) > mTolErr))
      validMomenta = false;

    // Check for negative energies.
    for ( int i = 0; i < event.size(); ++i)
      if ( (event[i].status() == -21 || event[i].status() == -22
         || event[i].isFinal() ) && event[i].e() < 0. ) validMomenta = false;

  } // Done with loop over systems.

  return (validColour && validCharge && validMomenta);

}
//-------------------------------------------------------------------------

bool DireTimes::validMotherDaughter( const Event& event ) {

  vector<int> noMot;
  vector<int> noDau;
  vector< pair<int,int> > noMotDau;

  // Loop through the event and check that there are beam particles.
  bool hasBeams = false;
  for (int i = 0; i < event.size(); ++i) {
    int status = event[i].status();
    if (abs(status) == 12) hasBeams = true;

    // Check that mother and daughter lists not empty where not expected to.
    vector<int> mList = event[i].motherList();
    vector<int> dList = event[i].daughterList();
    if (mList.size() == 0 && abs(status) != 11 && abs(status) != 12)
      noMot.push_back(i);
    if (dList.size() == 0 && status < 0 && status != -11)
      noDau.push_back(i);

    // Check that the particle appears in the daughters list of each mother.
    for (int j = 0; j < int(mList.size()); ++j) {
      if ( event[mList[j]].daughter1() <= i
        && event[mList[j]].daughter2() >= i ) continue;
      vector<int> dmList = event[mList[j]].daughterList();
      bool foundMatch = false;
      for (int k = 0; k < int(dmList.size()); ++k)
      if (dmList[k] == i) {
        foundMatch = true;
        break;
      }
      if (!hasBeams && mList.size() == 1 && mList[0] == 0) foundMatch = true;
      if (!foundMatch) {
        bool oldPair = false;
        for (int k = 0; k < int(noMotDau.size()); ++k)
        if (noMotDau[k].first == mList[j] && noMotDau[k].second == i) {
          oldPair = true;
          break;
        }
        if (!oldPair) noMotDau.push_back( make_pair( mList[j], i) );
      }
    }

    // Check that the particle appears in the mothers list of each daughter.
    for (int j = 0; j < int(dList.size()); ++j) {
      if ( event[dList[j]].statusAbs() > 80
        && event[dList[j]].statusAbs() < 90
        && event[dList[j]].mother1() <= i
        && event[dList[j]].mother2() >= i) continue;
      vector<int> mdList = event[dList[j]].motherList();
      bool foundMatch = false;
      for (int k = 0; k < int(mdList.size()); ++k)
      if (mdList[k] == i) {
        foundMatch = true;
        break;
      }
      if (!foundMatch) {
        bool oldPair = false;
        for (int k = 0; k < int(noMotDau.size()); ++k)
        if (noMotDau[k].first == i && noMotDau[k].second == dList[j]) {
          oldPair = true;
          break;
        }
        if (!oldPair) noMotDau.push_back( make_pair( i, dList[j]) );
      }
    }
  }

  // Mother-daughter relations not correct if any lists do not match.
  bool valid = true;
  if (noMot.size() > 0 || noDau.size() > 0 || noMotDau.size() > 0)
    valid = false;

  // Done.
  return valid;

}

//--------------------------------------------------------------------------

// Find index of colour partner for input colour.

int DireTimes::FindCol(int col, vector<int> iExc, const Event& event,
    int type, int iSys) {

  int index = 0;

  int inA = 0, inB = 0;
  for (int i=event.size()-1; i > 0; --i) {
    if ( event[i].mother1() == 1 && event[i].status() != -31
      && event[i].status() != -34) { if (inA == 0) inA = i; }
    if ( event[i].mother1() == 2 && event[i].status() != -31
      && event[i].status() != -34) { if (inB == 0) inB = i; }
  }
  if (iSys >= 0) {
    inA = partonSystemsPtr->getInA(iSys);
    inB = partonSystemsPtr->getInB(iSys);
  }
  // Unset if the incoming particles are flagged as outgoing. Instead, try to
  // resort to information stored in 0th event entry.
  if (event[inA].status() > 0) {
    inA = 0;
    if (event[0].daughter1() > 0) inA = event[0].daughter1();
  }
  if (event[inB].status() > 0) {
    inB = 0;
    if (event[0].daughter2() > 0) inB = event[0].daughter2();
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

//--------------------------------------------------------------------------

// Rescatter: If a dipole stretches between two different systems, those
//            systems will no longer locally conserve momentum. These
//            imbalances become problematic when ISR or primordial kT
//            is switched on as these steps involve Lorentz boosts.
//
//            'rescatterPropagateRecoil' tries to fix momentum in all
//            systems by propogating recoil momentum through all
//            intermediate systems. As the momentum transfer is already
//            defined, this can lead to internal lines gaining a
//            virtuality.

// Useful definitions for a pair of integers and a vector of pairs
typedef pair < int, int >  pairInt;
typedef vector < pairInt > vectorPairInt;

//--------------------------------------------------------------------------

// findParentSystems
//  Utility routine to find all parent systems of a given system
//  Returns a vector of pairs of integers with:
//   a) The system index, including the starting system (negative
//      if (b) points to a parent system, positive if (b) points
//      to a daughter system
//   b) The event record index that is the path out of the system
//      (if forwards == false, this is an incoming parton to the
//      system, and is +ve if side A or -ve if side B,
//      if forwards == true, this is an outgoing parton from the
//      system).
//  Returns as empty vector on failure
//  Note: this assumes single rescattering only and therefore only
//        one possible parent system

inline vectorPairInt findParentSystems(const int sys,
  Event& event, PartonSystems* partonSystemsPtr, bool forwards) {

  vectorPairInt parentSystems;
  parentSystems.reserve(10);

  int iSysCur = sys;
  while (true) {
    // Get two incoming partons
    int iInA = partonSystemsPtr->getInA(iSysCur);
    int iInB = partonSystemsPtr->getInB(iSysCur);

    // Check if either of these links to another system
    int iIn = 0;
    if (event[iInA].isRescatteredIncoming()) iIn =  iInA;
    if (event[iInB].isRescatteredIncoming()) iIn = -iInB;

    // Save the current system to the vector
    parentSystems.push_back( pairInt(-iSysCur, iIn) );
    if (iIn == 0) break;

    int iInAbs  = abs(iIn);
    int iMother = event[iInAbs].mother1();
    iSysCur     = partonSystemsPtr->getSystemOf(iMother);
    if (iSysCur == -1) {
      parentSystems.clear();
      break;
    }
  }

  // If forwards is set, change all event record indices to go to daughter
  // systems rather than parent systems
  if (forwards) {
    vectorPairInt::reverse_iterator rit;
    for (rit = parentSystems.rbegin(); rit < (parentSystems.rend() - 1);
         ++rit) {
      pairInt &cur  = *rit;
      pairInt &next = *(rit + 1);
      cur.first     = -cur.first;
      cur.second    = (next.second < 0) ? -event[abs(next.second)].mother1() :
                                           event[abs(next.second)].mother1();
    }
  }

  return parentSystems;
}

//--------------------------------------------------------------------------

// Print the list of dipoles.

void DireTimes::list() const {

  // Header.
  cout << "\n --------  DIRE DireTimes Dipole Listing  ------------------"
       << "--------------------------------------------------------------"
       << "----------\n\n"
       << "   i     rad    rec       pTmax     col    isr"
       << "   sys   sysR            m2          siblings        allowedIds\n"
       << fixed << setprecision(3);

  // Loop over dipole list and print it.
  for (int i = 0; i < int(dipEnd.size()); ++i) {
  cout << scientific << setprecision(4)
     << setw(4) << i << " | "
     << setw(4) << dipEnd[i].iRadiator << " | "
     << setw(4) << dipEnd[i].iRecoiler << " | "
     << setw(11) << dipEnd[i].pTmax  << " | "
     << setw(3) << dipEnd[i].colType  << " | "
     << setw(4) << dipEnd[i].isrType << " | "
     << setw(4) << dipEnd[i].system << " | "
     << setw(4) << dipEnd[i].systemRec << " | "
     << setw(11) << dipEnd[i].m2Dip << " | ";
    ostringstream os;
    os << dipEnd[i].iSiblings.listPos();
    cout << setw(15) << os.str() << " | ";
    os.str("");
    for (int j = 0; j < int(dipEnd[i].allowedEmissions.size()); ++j)
      os << setw(4) << dipEnd[i].allowedEmissions[j];
    cout << setw(15) << os.str() << endl;
  }

  if (dryrun){
    for ( unordered_map<string,DireSplitting*>::const_iterator it
      = splits.begin(); it != splits.end(); ++it ) {
      multimap<double,OverheadInfo> bla = it->second->overhead_map;
      cout << it->first << endl;
      for ( multimap<double, OverheadInfo >::const_iterator itb = bla.begin();
        itb != bla.end(); ++itb )
        cout << "  pT2=" << itb->first << " " << itb->second.list() << endl;
    }
  }

  // Done.
  cout << "\n --------  End DIRE DireTimes Dipole Listing  --------------"
       << "--------------------------------------------------------------"
       << "----------" << endl;

}

//--------------------------------------------------------------------------

// Function to calculate the correct alphaS/2*Pi value, including
// renormalisation scale variations + threshold matching.

double DireTimes::alphasNow( double pT2, double renormMultFacNow, int iSys ) {

  // Get beam for PDF alphaS, if necessary.
  BeamParticle* beam = nullptr;
  if (beamAPtr != nullptr || beamBPtr != nullptr) {
    beam = (beamAPtr != nullptr && particleDataPtr->isHadron(beamAPtr->id()))
         ? beamAPtr
         : (beamBPtr != nullptr && particleDataPtr->isHadron(beamBPtr->id()))
         ? beamBPtr : nullptr;
    if (beam == nullptr && beamAPtr != 0) beam = beamAPtr;
    if (beam == nullptr && beamBPtr != 0) beam = beamBPtr;
  }
  double scale       = pT2*renormMultFacNow;
  scale              = max(scale, pT2colCut);

  // Get alphaS(k*pT^2) and subtractions.
  double asPT2pi      = (usePDFalphas && beam != nullptr)
                      ? beam->alphaS(scale)  / (2.*M_PI)
                      : alphaS.alphaS(scale) / (2.*M_PI);

  // Get kernel order.
  int order = kernelOrder-1;
  // Use simple kernels for showering secondary scatterings.
  bool hasInA = (partonSystemsPtr->getInA(iSys) != 0);
  bool hasInB = (partonSystemsPtr->getInB(iSys) != 0);
  if (iSys != 0 && hasInA && hasInB) order = kernelOrderMPI-1;

  // Now find the necessary thresholds so that alphaS can be matched
  // correctly.
  double m2cNow = m2cPhys;
  if ( !( (scale > m2cNow && pT2 < m2cNow)
       || (scale < m2cNow && pT2 > m2cNow) )) m2cNow = -1.;
  double m2bNow = m2bPhys;
  if ( !( (scale > m2bNow && pT2 < m2bNow)
       || (scale < m2bNow && pT2 > m2bNow) ) ) m2bNow = -1.;
  vector<double> scales;
  scales.push_back(scale);
  scales.push_back(pT2);
  if (m2cNow > 0.) scales.push_back(m2cNow);
  if (m2bNow > 0.) scales.push_back(m2bNow);
  sort( scales.begin(), scales.end());
  if (scale > pT2) reverse(scales.begin(), scales.end());

  double asPT2piCorr  = asPT2pi;
  for ( int i = 1; i< int(scales.size()); ++i) {
    double NF    = getNF( 0.5*(scales[i]+scales[i-1]) );
    double L     = log( scales[i]/scales[i-1] );
    double subt  = 0.;
    if (order > 0) subt += asPT2piCorr * beta0(NF) * L;
    if (order > 2) subt += pow2( asPT2piCorr ) * ( beta1(NF)*L
                                   - pow2(beta0(NF)*L) );
    if (order > 4) subt += pow( asPT2piCorr, 3) * ( beta2(NF)*L
                                   - 2.5 * beta0(NF)*beta1(NF)*L*L
                                   + pow( beta0(NF)*L, 3) );
    asPT2piCorr *= 1.0 - subt;
  }

  // Done.
  return asPT2piCorr;

}

//-------------------------------------------------------------------------

// Function to calculate the correct alphaem/2*Pi value, including
// renormalisation scale variations + threshold matching.

double DireTimes::alphaemNow( double pT2, double renormMultFacNow, int ) {

  double scale       = pT2*renormMultFacNow;
  scale              = max(scale, pT2colCut);

  // Get alphaEM(k*pT^2) and subtractions.
  double aemPT2pi = alphaEM.alphaEM(scale) / (2.*M_PI);

  // Done.
  return aemPT2pi;

}

//-------------------------------------------------------------------------

// Auxiliary function to get number of flavours.

double DireTimes::getNF(double pT2) {

  double NF = 6.;

  BeamParticle* beam = nullptr;
  if (beamAPtr != nullptr || beamBPtr != nullptr) {
    beam = (beamAPtr != nullptr && particleDataPtr->isHadron(beamAPtr->id()))
         ? beamAPtr
         : (beamBPtr != nullptr && particleDataPtr->isHadron(beamBPtr->id()))
         ? beamBPtr : nullptr;
    if (beam == nullptr && beamAPtr != 0) beam = beamAPtr;
    if (beam == nullptr && beamBPtr != 0) beam = beamBPtr;
  }
  // Get current number of flavours.
  if ( !usePDFalphas || beam == nullptr ) {
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

//==========================================================================

} // end namespace Pythia8
