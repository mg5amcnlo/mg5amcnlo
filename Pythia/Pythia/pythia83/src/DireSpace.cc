// DireSpace.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the DireSpace class.

#include "Pythia8/DireSpace.h"
#include "Pythia8/DireTimes.h"
#include "Pythia8/DireHistory.h"

namespace Pythia8 {

//==========================================================================

// The DireSpace class.

//--------------------------------------------------------------------------

// Constants: could be changed here if desired, but normally should not.
// These are of technical nature, as described for each.

// Leftover companion can give PDF > 0 at small Q2 where other PDF's = 0,
// and then one can end in infinite loop of impossible kinematics.
const int    DireSpace::MAXLOOPTINYPDF = 10;

// Minimal allowed c and b quark masses, for flavour thresholds.
const double DireSpace::MCMIN          = 1.2;
const double DireSpace::MBMIN          = 4.0;

// Switch to alternative (but equivalent) backwards evolution for
// g -> Q Qbar (Q = c or b) when below QTHRESHOLD * mQ2.
const double DireSpace::CTHRESHOLD     = 2.0;
const double DireSpace::BTHRESHOLD     = 2.0;

// Renew evaluation of PDF's when the pT2 step is bigger than this
// (in addition to initial scale and c and b thresholds.)
const double DireSpace::EVALPDFSTEP    = 0.1;

// Lower limit on PDF value in order to avoid division by zero.
//const double DireSpace::TINYPDF        = 1e-15;
const double DireSpace::TINYPDF        = 1e-5;

// Lower limit on estimated evolution rate, below which stop.
const double DireSpace::TINYKERNELPDF  = 1e-15;

// Lower limit on pT2, below which branching is rejected.
const double DireSpace::TINYPT2        = 0.25e-6;

// No attempt to do backwards evolution of a heavy (c or b) quark
// if evolution starts at a scale pT2 < HEAVYPT2EVOL * mQ2.
const double DireSpace::HEAVYPT2EVOL   = 1.1;

// No attempt to do backwards evolution of a heavy (c or b) quark
// if evolution starts at a  x > HEAVYXEVOL * x_max, where
// x_max is the largest possible x value for a g -> Q Qbar branching.
const double DireSpace::HEAVYXEVOL     = 0.9;

// When backwards evolution Q -> g + Q creates a heavy quark Q,
// an earlier branching g -> Q + Qbar will restrict kinematics
// to  M_{Q Qbar}^2 > EXTRASPACEQ * 4 m_Q^2. (Smarter to be found??)
const double DireSpace::EXTRASPACEQ    = 2.0;

// Never pick pT so low that alphaS is evaluated too close to Lambda_3.
const double DireSpace::LAMBDA3MARGIN  = 1.1;

// Do not warn for large PDF ratios at small pT2 scales.
const double DireSpace::PT2MINWARN = 1.;

// Cutoff for f_e^e at x < 1 - 10^{-10} to be used in z selection.
// Note: the x_min quantity come from 1 - x_max.
const double DireSpace::LEPTONXMIN     = 1e-10;
const double DireSpace::LEPTONXMAX     = 1. - 1e-10;

// Stop l -> l gamma evolution slightly above m2l.
const double DireSpace::LEPTONPT2MIN   = 1.2;

// Enhancement of l -> l gamma trial rate to compensate imperfect modelling.
const double DireSpace::LEPTONFUDGE    = 10.;

// Overestimation extra factors by branching type
const double DireSpace::HEADROOMQ2G = 1.35;
const double DireSpace::HEADROOMQ2Q = 1.15;
const double DireSpace::HEADROOMG2Q = 1.35;
const double DireSpace::HEADROOMG2G = 1.35;

const double DireSpace::TINYMASS = 1e-3;

const double DireSpace::DPHI_II = 2./3.;
const double DireSpace::DPHI_IF = 2./3.;

// pT2 below which PDF overestimates should be larger.
const double DireSpace::PT2_INCREASE_OVERESTIMATE = 2.;

const double DireSpace::KERNEL_HEADROOM = 1.;

//--------------------------------------------------------------------------

// Initialize alphaStrong, alphaEM and related pTmin parameters.

void DireSpace::init( BeamParticle* beamAPtrIn,
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

  // Store input pointers for future use.
  beamAPtr        = beamAPtrIn;
  beamBPtr        = beamBPtrIn;

  // Main flags to switch on and off branchings.
  doQCDshower     = settingsPtr->flag("SpaceShower:QCDshower");
  doQEDshowerByQ  = settingsPtr->flag("SpaceShower:QEDshowerByQ");
  doQEDshowerByL  = settingsPtr->flag("SpaceShower:QEDshowerByL");
  doDecaysAsShower   = settingsPtr->flag("DireSpace:DecaysAsShower");

  // Matching in pT of hard interaction to shower evolution.
  pTmaxMatch      = settingsPtr->mode("SpaceShower:pTmaxMatch");
  pTdampMatch     = settingsPtr->mode("SpaceShower:pTdampMatch");
  pTmaxFudge      = settingsPtr->parm("SpaceShower:pTmaxFudge");
  pTmaxFudgeMPI   = settingsPtr->parm("SpaceShower:pTmaxFudgeMPI");
  pTdampFudge     = settingsPtr->parm("SpaceShower:pTdampFudge");
  pT2minVariations= pow2(max(0.,settingsPtr->parm("Variations:pTmin")));
  pT2minEnhance   = pow2(max(0.,settingsPtr->parm("Enhance:pTmin")));
  pT2minMECs      = pow2(max(0.,settingsPtr->parm("Dire:pTminMECs")));
  Q2minMECs       = pow2(max(0.,settingsPtr->parm("Dire:QminMECs")));
  nFinalMaxMECs   = settingsPtr->mode("Dire:nFinalMaxMECs");
  suppressLargeMECs = settingsPtr->flag("Dire:suppressLargeMECs");

  // Optionally force emissions to be ordered in rapidity/angle.
  doRapidityOrder = settingsPtr->flag("SpaceShower:rapidityOrder");

  // Charm, bottom and lepton mass thresholds.
  mc              = max( MCMIN, particleDataPtr->m0(4));
  mb              = max( MBMIN, particleDataPtr->m0(5));
  m2c             = pow2(mc);
  m2b             = pow2(mb);

  // Parameters of scale choices (inherited from Pythia).
  renormMultFac     = settingsPtr->parm("SpaceShower:renormMultFac");
  factorMultFac     = settingsPtr->parm("SpaceShower:factorMultFac");
  useFixedFacScale  = settingsPtr->flag("SpaceShower:useFixedFacScale");
  fixedFacScale2    = pow2(settingsPtr->parm("SpaceShower:fixedFacScale"));

  // Parameters of alphaStrong generation.
  alphaSvalue     = settingsPtr->parm("SpaceShower:alphaSvalue");
  alphaSorder     = settingsPtr->mode("SpaceShower:alphaSorder");
  alphaSnfmax     = settingsPtr->mode("StandardModel:alphaSnfmax");
  alphaSuseCMW    = settingsPtr->flag("SpaceShower:alphaSuseCMW");
  alphaS2pi       = 0.5 * alphaSvalue / M_PI;
  asScheme        = settingsPtr->mode("DireSpace:alphasScheme");

  // Set flavour thresholds by default Pythia masses, unless zero.
  double mcpy = particleDataPtr->m0(4);
  double mbpy = particleDataPtr->m0(5);
  double mtpy = particleDataPtr->m0(6);
  if (mcpy > 0.0 && mbpy > 0.0 && mtpy > 0.0)
    alphaS.setThresholds(mcpy, mbpy, mtpy);

  // Initialize alpha_strong generation.
  alphaS.init( alphaSvalue, alphaSorder, alphaSnfmax, alphaSuseCMW);

  // Lambda for 5, 4 and 3 flavours.
  Lambda5flav     = alphaS.Lambda5();
  Lambda4flav     = alphaS.Lambda4();
  Lambda3flav     = alphaS.Lambda3();
  Lambda5flav2    = pow2(Lambda5flav);
  Lambda4flav2    = pow2(Lambda4flav);
  Lambda3flav2    = pow2(Lambda3flav);

  // Regularization of QCD evolution for pT -> 0. Can be taken
  // same as for multiparton interactions, or be set separately.
  useSamePTasMPI  = settingsPtr->flag("SpaceShower:samePTasMPI");
  if (useSamePTasMPI) {
    pT0Ref        = settingsPtr->parm("MultipartonInteractions:pT0Ref");
    ecmRef        = settingsPtr->parm("MultipartonInteractions:ecmRef");
    ecmPow        = settingsPtr->parm("MultipartonInteractions:ecmPow");
    pTmin         = settingsPtr->parm("MultipartonInteractions:pTmin");
  } else {
    pT0Ref        = settingsPtr->parm("SpaceShower:pT0Ref");
    ecmRef        = settingsPtr->parm("SpaceShower:ecmRef");
    ecmPow        = settingsPtr->parm("SpaceShower:ecmPow");
    pTmin         = settingsPtr->parm("SpaceShower:pTmin");
  }

  // Calculate nominal invariant mass of events. Set current pT0 scale to zero.
  sCM = m2( beamAPtr->p(), beamBPtr->p());
  eCM = sqrt(sCM);
  pT0 = 0.;

  // Restrict pTmin to ensure that alpha_s(pTmin^2 + pT_0^2) does not blow up.
  double pTminAbs = sqrtpos(pow2(LAMBDA3MARGIN) * Lambda3flav2 / renormMultFac
                  - pT0*pT0);
  if (pTmin < pTminAbs) {
    pTmin         = pTminAbs;
    ostringstream newPTmin;
    newPTmin << fixed << setprecision(3) << pTmin;
    infoPtr->errorMsg("Warning in DireSpace::init: pTmin too low",
                      ", raised to " + newPTmin.str() );
    infoPtr->setTooLowPTmin(true);
  }

  // Derived parameters of QCD evolution.
  pT20            = pow2(pT0);
  pT2min          = pow2(pTmin);

  m2min           = pT2min;
  mTolErr         = settingsPtr->parm("Check:mTolErr");

  double pT2minQED = pow2(settingsPtr->parm("SpaceShower:pTminChgQ"));
  pT2minQED = max(pT2minQED, pow2(settingsPtr->parm("SpaceShower:pTminChgL")));

  pT2cutSave = create_unordered_map<int,double>
    (21,pT2min)
    (1,pT2min)(-1,pT2min)(2,pT2min)(-2,pT2min)
    (3,pT2min)(-3,pT2min)(4,pT2min)(-4,pT2min)
    (5,pT2min)(-5,pT2min)(6,pT2min)(-6,pT2min)
    (22,pT2minQED)
    (11,pT2minQED)(-11,pT2minQED)(13,pT2minQED)(-13,pT2minQED)
    (15,pT2minQED)(-15,pT2minQED)
    (900032,pT2minQED)(900012,pT2minQED)
    (900040,pT2minQED);

  bool_settings = create_unordered_map<string,bool>
    ("doQEDshowerByL",doQEDshowerByL)
    ("doQEDshowerByQ",doQEDshowerByQ);

  usePDFalphas       = settingsPtr->flag("ShowerPDF:usePDFalphas");
  useSummedPDF       = settingsPtr->flag("ShowerPDF:useSummedPDF");
  usePDF             = settingsPtr->flag("ShowerPDF:usePDF");
  BeamParticle& beam = (particleDataPtr->isHadron(beamAPtr->id())) ?
    *beamAPtr : *beamBPtr;
  alphaS2piOverestimate = (usePDFalphas) ? beam.alphaS(pT2min) * 0.5/M_PI
                        : (alphaSorder > 0) ? alphaS.alphaS(pT2min) * 0.5/M_PI
                                            :  0.5 * 0.5/M_PI;
  usePDFmasses       = settingsPtr->flag("ShowerPDF:usePDFmasses");
  BeamParticle* bb   = ( particleDataPtr->isHadron(beamAPtr->id())) ? beamAPtr
                     : ( particleDataPtr->isHadron(beamBPtr->id())) ?
    beamBPtr : NULL;
  m2cPhys            = (usePDFalphas && bb != NULL)
                     ? pow2(max(0.,bb->mQuarkPDF(4))) : alphaS.muThres2(4);
  m2bPhys            = (usePDFalphas && bb != NULL)
                     ? pow2(max(0.,bb->mQuarkPDF(5))) : alphaS.muThres2(5);
  useSystems         = true;

  // Allow massive incoming particles. Currently not supported by Pythia.
  useMassiveBeams    = false;

  // Create maps of accept/reject weights
  string key = "base";
  rejectProbability.insert( make_pair(key, multimap<double,double>() ));
  acceptProbability.insert( make_pair(key, map<double,double>() ));
  doVariations = settingsPtr->flag("Variations:doVariations");
  splittingSelName="";
  splittingNowName="";

  // Set splitting library, if already exists.
  if (splittingsPtr) splits = splittingsPtr->getSplittings();
  overhead.clear();
  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) overhead.insert(make_pair(it->first,1.));

  nFinalMax          = settingsPtr->mode("DireSpace:nFinalMax");
  useGlobalMapIF     = settingsPtr->flag("DireSpace:useGlobalMapIF");

  forceMassiveMap    = settingsPtr->flag("DireSpace:forceMassiveMap");

  // Mode for higher-order kernels.
  kernelOrder        = settingsPtr->mode("DireSpace:kernelOrder");
  kernelOrderMPI     = settingsPtr->mode("DireSpace:kernelOrderMPI");

  // Various other parameters.
  doMEcorrections    = settingsPtr->flag("Dire:doMECs")
                    || settingsPtr->flag("Dire:doMOPS");
  doMEafterFirst     = settingsPtr->flag("SpaceShower:MEafterFirst");
  doPhiPolAsym       = settingsPtr->flag("SpaceShower:phiPolAsym");
  doPhiIntAsym       = settingsPtr->flag("SpaceShower:phiIntAsym");
  strengthIntAsym    = settingsPtr->parm("SpaceShower:strengthIntAsym");
  nQuarkIn           = settingsPtr->mode("SpaceShower:nQuarkIn");

  // Possibility of two predetermined hard emissions in event.
  doSecondHard       = settingsPtr->flag("SecondHard:generate");

  // Optional dampening at small pT's when large multiplicities.
  enhanceScreening
    = settingsPtr->mode("MultipartonInteractions:enhanceScreening");
  if (!useSamePTasMPI) enhanceScreening = 0;

  // Possibility to allow user veto of emission step.
  hasUserHooks       = (userHooksPtr != 0);
  canVetoEmission    = (userHooksPtr != 0)
                     ? userHooksPtr->canVetoISREmission() : false;

  // Done.
  isInitSave = true;

}

//--------------------------------------------------------------------------

// Initialize bookkeeping of shower variations.

void DireSpace::initVariations() {

  // Create maps of accept/reject weights
  for ( int i=0; i < weights->sizeWeights(); ++i) {
    string key = weights->weightName(i);
    if ( key.compare("base") == 0) continue;
    if ( key.find("fsr") != string::npos) continue;
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

//bool DireSpace::limitPTmax( Event& event, double Q2Fac, double) {
bool DireSpace::limitPTmax( Event& event, double, double) {

  // Find whether to limit pT. Begin by user-set cases.
  bool dopTlimit = false;
  dopTlimit1 = dopTlimit2 = false;
  int nHeavyCol = 0;
  if (pTmaxMatch == 1) dopTlimit = dopTlimit1 = dopTlimit2 = true;

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

// Function to reset generic things (called from DireTimes::prepareGlobal!)

void DireSpace::resetWeights() {

  // Clear accept/reject weights.
  weights->reset();
  for ( unordered_map<string, multimap<double,double> >::iterator
    it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
    it->second.clear();
  for ( unordered_map<string, map<double,double> >::iterator
    it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
    it->second.clear();

  return;
}

//--------------------------------------------------------------------------

// Prepare system for evolution; identify ME.
// Routine may be called after multiparton interactions, for a new subystem.

void DireSpace::prepare( int iSys, Event& event, bool limitPTmaxIn) {

  // Calculate remainder shower weight after MPI.
  if (nMPI < infoPtr->getCounter(23) && iSys == infoPtr->getCounter(23)) {
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

  // Find positions of incoming colliding partons.
  int in1 = getInA(iSys);
  int in2 = getInB(iSys);

  // Rescattered partons cannot radiate.
  bool canRadiate1 = !(event[in1].isRescatteredIncoming());
  bool canRadiate2 = !(event[in2].isRescatteredIncoming());

  // Reset dipole-ends list for first interaction. Also resonances.
  if (iSys == 0) {dipEnd.resize(0); idResFirst  = 0;}
  else if (iSys == 1) idResSecond = 0;

  // Set splitting library.
  splits = splittingsPtr->getSplittings();
  overhead.clear();
  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) overhead.insert(make_pair(it->first,1.));

  // Find matrix element corrections for system.
  int MEtype = 0;

  // In case of DPS overwrite limitPTmaxIn by saved value.
  if (doSecondHard && iSys == 0) limitPTmaxIn = dopTlimit1;
  if (doSecondHard && iSys == 1) limitPTmaxIn = dopTlimit2;

  // Find dipole ends for QCD radiation.
  // Note: colour type can change during evolution, so book also if zero.
  if (doQCDshower) {

    // Find dipole end formed by colour index.
    int colTag = event[in1].col();
    if (canRadiate1 && colTag > 0)  setupQCDdip( iSys, 1,  colTag,  1, event,
                                                 MEtype, limitPTmaxIn);
    // Find dipole end formed by anticolour index.
    int acolTag = event[in1].acol();
    if (canRadiate1 && acolTag > 0) setupQCDdip( iSys, 1, acolTag, -1, event,
                                                 MEtype, limitPTmaxIn);
    // Find dipole end formed by colour index.
    colTag = event[in2].col();
    if (canRadiate2 && colTag > 0)  setupQCDdip( iSys, 2,  colTag,  1, event,
                                                 MEtype, limitPTmaxIn);
    // Find dipole end formed by anticolour index.
    acolTag = event[in2].acol();
    if (canRadiate2 && acolTag > 0) setupQCDdip( iSys, 2, acolTag, -1, event,
                                                 MEtype, limitPTmaxIn);
  }

  // Now find non-QCD dipoles and/or update the existing dipoles.
  getGenDip( iSys, 1, event, limitPTmaxIn, dipEnd);
  getGenDip( iSys, 2, event, limitPTmaxIn, dipEnd);

  // Store the z and pT2 values of the last previous splitting
  // when an event history has already been constructed.
  if (iSys == 0 && infoPtr->hasHistory()) {
    double zNow   = infoPtr->zNowISR();
    double pT2Now = infoPtr->pT2NowISR();
    for (int iDipEnd = 0; iDipEnd < int(dipEnd.size()); ++iDipEnd) {
      dipEnd[iDipEnd].zOld = zNow;
      dipEnd[iDipEnd].pT2Old = pT2Now;
      ++dipEnd[iDipEnd].nBranch;
    }
  }

  // Now update all dipoles.
  updateDipoles(event, iSys);

  // Counter of proposed emissions.
  nProposedPT.clear();
  if ( nProposedPT.find(iSys) == nProposedPT.end() )
    nProposedPT.insert(make_pair(iSys,0));

  splittingSelName="";
  splittingNowName="";
  dipEndSel = 0;

  // Clear weighted shower book-keeping.
  for ( unordered_map<string, multimap<double,double> >::iterator
    it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
    it->second.clear();
  for ( unordered_map<string, map<double,double> >::iterator
    it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
    it->second.clear();

}

//--------------------------------------------------------------------------

void DireSpace::clear() {

  dipEnd.resize(0);
  weights->reset();
  dipEndSel = 0;

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

// Setup a dipole end for a QCD colour charge.

void DireSpace::setupQCDdip( int iSys, int side, int colTag, int colSign,
  const Event& event, int MEtype, bool limitPTmaxIn) {

  // Initial values. Find if allowed to hook up beams.
  int iRad     = (side == 1) ? getInA(iSys) : getInB(iSys);
  int iPartner = 0;
  int sizeAllA = partonSystemsPtr->sizeAll(iSys);
  int sizeOut  = partonSystemsPtr->sizeOut(iSys);
  int sizeAll  = sizeAllA;
  int sizeIn   = sizeAll - sizeOut;
  int sizeInA  = sizeAllA - sizeIn - sizeOut;

  // Colour: other end by same index in final state or opposite in beam.
  if (colSign > 0)
  for (int j = 0; j < sizeAll; ++j) {
    int iRecNow = partonSystemsPtr->getAll(iSys, j + sizeInA);
    if (iRecNow == iRad) continue;
    if ( ( j >= sizeIn && event[iRecNow].col() == colTag
      && event[iRecNow].isFinal() )
      || ( j <  sizeIn && event[iRecNow].acol() == colTag
      && !event[iRecNow].isRescatteredIncoming() ) ) {
      iPartner = iRecNow;
      break;
    }
  }

  // Anticolour: other end by same index in final state or opposite in beam.
  if (colSign < 0)
  for (int j = 0; j < sizeAll; ++j) {
    int iRecNow = partonSystemsPtr->getAll(iSys, j + sizeInA);
    if (iRecNow == iRad) continue;
    if ( ( j >= sizeIn && event[iRecNow].acol()  == colTag
      && event[iRecNow].isFinal() )
      || ( j <  sizeIn && event[iRecNow].col() == colTag
      && !event[iRecNow].isRescatteredIncoming() ) ) {
      iPartner = iRecNow;
      break;
    }
  }

  // Check for failure to locate any recoiler
  if ( iPartner == 0 ) {
    infoPtr->errorMsg("Error in DireSpace::setupQCDdip: "
                      "failed to locate any recoiling partner");
    return;
  }

  // Store dipole colour end(s).
  // Max scale either by parton scale or by half dipole mass.
  double pTmax = event[iRad].scale();
  if (limitPTmaxIn) {
    if (iSys == 0 || (iSys == 1 && doSecondHard)) pTmax *= pTmaxFudge;
    else if (sizeIn > 0) pTmax *= pTmaxFudgeMPI;
  //} else pTmax = 0.5 * m( event[iRad], event[iPartner]);
  } else pTmax = m( event[iRad], event[iPartner]);

  // Force maximal pT to LHEF input value.
  if ( abs(event[iRad].status()) > 20 &&  abs(event[iRad].status()) < 24
    && settingsPtr->flag("Beams:setProductionScalesFromLHEF")
    && event[iRad].scale() > 0.)
    pTmax = event[iRad].scale();

  // Force maximal pT to LHEF scales tag value.
  double mups = infoPtr->getScalesAttribute("mups");
  if ( abs(event[iRad].status()) > 20
    && abs(event[iRad].status()) < 24
    && settingsPtr->flag("Beams:setProductionScalesFromLHEF")
    && !isnan(mups) )
    pTmax = mups;

  int colType  = (event[iRad].id() == 21) ? 2 * colSign : colSign;
  dipEnd.push_back( DireSpaceEnd( iSys, side, iRad, iPartner, pTmax, colType,
                                  0, 0, MEtype));
  dipEnd.back().init(event);

}

//--------------------------------------------------------------------------

// Setup a dipole end for a QCD colour charge.

void DireSpace::getGenDip( int iSys, int side, const Event& event,
  bool limitPTmaxIn, vector<DireSpaceEnd>& dipEnds) {

  // Initial values.
  int iRad     = (iSys > -1) ? ((side == 1) ? getInA(iSys) : getInB(iSys))
               : side;
  int sizeAllA = (iSys > -1) ? partonSystemsPtr->sizeAll(iSys) : event.size();
  int sizeOut  = (iSys > -1) ? partonSystemsPtr->sizeOut(iSys) : event.size();
  int sizeAll  = (iSys > -1) ? sizeAllA : event.size();
  int sizeIn   = (iSys > -1) ? sizeAll - sizeOut : 0;
  int sizeInA  = (iSys > -1) ? sizeAllA - sizeIn - sizeOut : 0;

  for (int i = 0; i < sizeAll; ++i) {
    int iRecNow = (iSys > -1) ?
      partonSystemsPtr->getAll(iSys, i + sizeInA) : i;
    if ( !event[iRecNow].isFinal()
       && event[iRecNow].mother1() != 1
       && event[iRecNow].mother1() != 2) continue;
    // Skip radiator.
    if ( iRecNow == iRad) continue;
    // Skip if dipole already exists, attempt to update the dipole end (a)
    // for the current a-b dipole.
    vector<int> iDip;
    for (int j = 0; j < int(dipEnds.size()); ++j)
      if ( dipEnds[j].iRadiator == iRad && dipEnds[j].iRecoiler == iRecNow )
        iDip.push_back(j);
    if ( int(iDip.size()) > 0) {
      for (int j = 0; j < int(iDip.size()); ++j)
        updateAllowedEmissions(event, &dipEnds[iDip[j]]);
      continue;
    }

    double pTmax = abs(2.*event[iRad].p()*event[iRecNow].p());
    if (limitPTmaxIn) {
      if (iSys == 0 || (iSys == 1 && doSecondHard)) pTmax *= pTmaxFudge;
      else if (sizeIn > 0) pTmax *= pTmaxFudgeMPI;
    } else pTmax = m( event[iRad], event[iRecNow]);

    appendDipole( event, iSys, side, iRad, iRecNow, pTmax, 0, 0, 0, 0, true, 0,
      vector<int>(), vector<double>(), dipEnds);
  }

  // Done.
  return;

}

//--------------------------------------------------------------------------

// Setup a dipole end for a QCD colour charge.

void DireSpace::getQCDdip( int iRad, int colTag, int colSign,
  const Event& event, vector<DireSpaceEnd>& dipEnds) {

  int iPartner = 0;

  // Colour: other end by same index in final state or opposite in beam.
  if (colSign > 0)
  for (int iRecNow = 0; iRecNow < event.size(); ++iRecNow) {
    if (iRecNow == iRad) continue;
    if ( ( event[iRecNow].col()  == colTag &&  event[iRecNow].isFinal() )
      || ( event[iRecNow].acol() == colTag && !event[iRecNow].isFinal() ) ) {
      iPartner = iRecNow;
      break;
    }
  }

  // Anticolour: other end by same index in final state or opposite in beam.
  if (colSign < 0)
  for (int iRecNow = 0; iRecNow < event.size(); ++iRecNow) {
    if (iRecNow == iRad) continue;
    if ( ( event[iRecNow].acol() == colTag &&  event[iRecNow].isFinal() )
      || ( event[iRecNow].col()  == colTag && !event[iRecNow].isFinal() ) ) {
      iPartner = iRecNow;
      break;
    }
  }

  // Store dipole colour end(s).
  // Max scale either by parton scale or by half dipole mass.
  double pTmax = abs(2.*event[iRad].p()*event[iPartner].p());
  int side     = (event[iRad].pz() > 0.) ? 1 : 2;
  int colType  = (event[iRad].id() == 21) ? 2 * colSign : colSign;

  if (iPartner > 0) {
    dipEnds.push_back( DireSpaceEnd( 0, side, iRad, iPartner, pTmax, colType));
    dipEnds.back().init(event);
  }
}

//--------------------------------------------------------------------------

// Function to set up and append a new dipole.

bool DireSpace::appendDipole( const Event& state, int sys, int side,
  int iRad, int iRecNow, double pTmax, int colType, int chgType, int weakType,
  int MEtype, bool normalRecoil, int weakPol, vector<int> iSpectator,
  vector<double> mass, vector<DireSpaceEnd>& dipEnds) {

  // Check and reset color type.
  if (colType == 0 && state[iRad].colType() != 0) {
    vector<int> shared = sharedColor(state[iRad], state[iRecNow]);
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
        if ( dipEnds[j].iRadiator == iRad && dipEnds[j].iRecoiler == iRecNow
          && dipEnds[j].colType == colTypeNow) { found = true; break; }
      }
      // Reset if color tag has not been found.
      if (!found) break;
    }
    colType = colTypeNow;
  }

  // Construct dipole.
  DireSpaceEnd dipNow = DireSpaceEnd( sys, side, iRad, iRecNow, pTmax, colType,
    chgType, weakType, MEtype, normalRecoil, weakPol,
    DireSingleColChain(), iSpectator, mass);

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

vector<int> DireSpace::sharedColor(const Particle& rad, const Particle& rec) {
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

void DireSpace::saveSiblings(const Event& state, int iSys) {

  int sizeAllSys = partonSystemsPtr->sizeSys();
  for (int iSystem=0; iSystem < sizeAllSys; ++iSystem) {

    if (iSys > -1 && iSystem != iSys) continue;

    vector<int> q, qb, g;
    int sizeSystem(partonSystemsPtr->sizeAll(iSystem)), nFinal(0);
    for ( int i = 0; i < sizeSystem; ++i) {

      int iPos = partonSystemsPtr->getAll(iSystem, i);
      if ( state[iPos].isFinal()) nFinal++;

      if (!state[iPos].isFinal()
          && state[iPos].mother1() != 1
          && state[iPos].mother1() != 2) continue;

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
      if (chains.chainOf(g[i]).size() != 0) continue;
      chains.addChain( DireSingleColChain(g[i],state, partonSystemsPtr));
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

// Function to set up and append a new dipole.

void DireSpace::updateDipoles(const Event& state, int iSys) {

  // Update the dipoles, and if necesarry, flag inactive dipoles for removal.
  vector<int> iRemove;
  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip) {
    if (!updateAllowedEmissions(state, &dipEnd[iDip])) iRemove.push_back(iDip);
    dipEnd[iDip].init(state);
  }
  // Now remove inactive dipoles.
  for (int i = iRemove.size()-1; i >= 0; --i) {
    dipEnd[iRemove[i]] = dipEnd.back();
    dipEnd.pop_back();
  }

  // Now go through dipole list and perform rudimentary checks.
  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip) {
    DireSpaceEnd* dip = &dipEnd[iDip];
    int iRad    = dip->iRadiator;
    int iRecNow = dip->iRecoiler;
    // Check and reset color type.
    if (dip->colType == 0 && state[iRad].colType() != 0) {
      vector<int> shared = sharedColor(state[iRad], state[iRecNow]);
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
  }

  saveSiblings(state, iSys);

}

//--------------------------------------------------------------------------

bool DireSpace::updateAllowedEmissions(const Event& state, DireSpaceEnd* dip) {
  // Clear any allowed emissions.
  dip->clearAllowedEmt();
  // Append all possible emissions.
  return appendAllowedEmissions(state, dip);
}

//--------------------------------------------------------------------------

// Function to set up and append a new dipole.

bool DireSpace::appendAllowedEmissions(const Event& state, DireSpaceEnd* dip) {

  // Now loop through all splitting kernels to find which emissions are
  // allowed from the current radiator-recoiler combination.
  bool isAllowed = false;
  int iRad(dip->iRadiator), iRecNow(dip->iRecoiler);
  pair<int,int> iRadRec(make_pair(iRad, iRecNow));
  pair<int,int> iRecRad(make_pair(iRecNow, iRad));

  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) {

    // Check if splitting is allowed.
    bool allowed = it->second->useFastFunctions()
                 ? it->second->canRadiate(state,iRad,iRecNow)
                 : it->second->canRadiate(state,iRadRec,bool_settings);
    if (!allowed) continue;

    // Get emission id.
    vector<int> re = it->second->radAndEmt( state[iRad].id(), dip->colType);

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
        for ( unordered_map<string,DireSplitting*>::iterator itRec =
                splits.begin();
          itRec != splits.end(); ++itRec ) {

          if ( isPartialFractioned ) break;
          bool allowedRec = itRec->second->useFastFunctions()
                 ? itRec->second->canRadiate(state,iRecNow,iRad)
                 : itRec->second->canRadiate(state,iRecRad,bool_settings);
          if (!allowedRec) continue;

          // Get emission id.
          int colTypeRec
            = state[iRecNow].isFinal() ? -dip->colType : dip->colType;
          vector<int> reRec
            = itRec->second->radAndEmt( state[iRecNow].id(), colTypeRec);

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

// Update dipole list after each ISR emission (so not used for resonances).

void DireSpace::update( int iSys, Event& event, bool) {

  // Find positions of incoming colliding partons.
  int in1 = getInA(iSys);
  int in2 = getInB(iSys);

  // Rescattered partons cannot radiate.
  bool canRadiate1 = !(event[in1].isRescatteredIncoming()) && doQCDshower;
  bool canRadiate2 = !(event[in2].isRescatteredIncoming()) && doQCDshower;

  // Find matrix element corrections for system.
  int MEtype = 0;

  dipEnd.resize(0);
  // Find dipole end formed by colour index.
  int colTag = event[in1].col();
  if (canRadiate1 && colTag > 0)  setupQCDdip( iSys, 1,  colTag,  1, event,
                                               MEtype, false);
  // Find dipole end formed by anticolour index.
  int acolTag = event[in1].acol();
  if (canRadiate1 && acolTag > 0) setupQCDdip( iSys, 1, acolTag, -1, event,
                                               MEtype, false);

  // Find dipole end formed by colour index.
  colTag = event[in2].col();
  if (canRadiate2 && colTag > 0)  setupQCDdip( iSys, 2,  colTag,  1, event,
                                               MEtype, false);
  // Find dipole end formed by anticolour index.
  acolTag = event[in2].acol();
  if (canRadiate2 && acolTag > 0) setupQCDdip( iSys, 2, acolTag, -1, event,
                                               MEtype, false);

  // Now find non-QCD dipoles and/or update the existing dipoles.
  getGenDip( iSys, 1, event, false, dipEnd);
  getGenDip( iSys, 2, event, false, dipEnd);

  // Now update all dipoles.
  updateDipoles(event, iSys);

}

//--------------------------------------------------------------------------

// Select next pT in downwards evolution of the existing dipoles.

double DireSpace::newPoint(const Event& inevt) {

  int asOrderSave = alphaSorder;
  double pT2minMECsSave = pT2minMECs;
  alphaSorder = 0.;
  pT2minMECs = 0.;
  double tFreeze = 1.;

  Event event(inevt);

  // Starting values: no radiating dipole found.
  double pT2sel = 0.;
  iDipSel       = 0;
  iSysSel       = 0;
  dipEndSel     = 0;
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

    // Loop over all possible dipole ends.
    for (int iDipEnd = 0; iDipEnd < int(dipEnd.size()); ++iDipEnd) {

      iDipNow        = iDipEnd;
      dipEndNow      = &dipEnd[iDipEnd];
      iSysNow        = dipEndNow->system;

      // Check whether dipole end should be allowed to shower.
      double pT2begDip = m2Max(iDipNow, event);

      if (pT2begDip > pT2sel) {
        double pT2endDip = 0.;

        // Find properties of dipole and radiating dipole end.
        sideA         = ( abs(dipEndNow->side) == 1 );
        bool finalRec = event[dipEndNow->iRecoiler].isFinal();
        BeamParticle& beamNow = (sideA) ? *beamAPtr : *beamBPtr;
        BeamParticle& beamRec = (sideA) ? *beamBPtr : *beamAPtr;
        iNow          = beamNow[iSysNow].iPos();
        iRec          = (finalRec) ? dipEndNow->iRecoiler
                                 : beamRec[iSysNow].iPos();
        idDaughter    = beamNow[iSysNow].id();
        xDaughter     = beamNow[iSysNow].x();
        x1Now         = (sideA) ? xDaughter : beamRec[iSysNow].x();
        x2Now         = (sideA) ? beamRec[iSysNow].x() : xDaughter;

        // Note dipole mass correction when recoiler is a rescatter.
        m2Rec         = (dipEndNow->normalRecoil) ? 0. : event[iRec].m2();
        m2Dip         = abs(2.*event[iNow].p()*event[iRec].p());

        // Dipole properties.
        dipEndNow->m2Dip  = m2Dip;
        // Reset emission properties.
        dipEndNow->pT2         =  0.0;
        dipEndNow->z           = -1.0;
        dipEndNow->phi         = -1.0;
        // Reset properties of 1->3 splittings.
        dipEndNow->sa1         =  0.0;
        dipEndNow->xa          = -1.0;
        dipEndNow->phia1       = -1.0;

        // Now do evolution in pT2, for QCD
        if (pT2begDip > pT2endDip) {

          if ( dipEndNow->canEmit() ) pT2nextQCD( pT2begDip, 0.,
            *dipEndNow, event, 0., tFreeze, true);

          // Update if found larger pT than current maximum.
          if (dipEndNow->pT2 > pT2sel) {
            pT2sel    = dipEndNow->pT2;
            iDipSel   = iDipNow;
            iSysSel   = iSysNow;
            dipEndSel = dipEndNow;
            splittingSelName = splittingNowName;
            splittingSel = splits[splittingSelName];
            splitInfoSel.store(splits[splittingSelName]->splitInfo);
            kernelSel = kernelNow;
            auxSel    = auxNow;
            overSel   = overNow;
            boostSel  = boostNow;
          }
        }
      }
    // End loop over dipole ends.
    }

    if (dipEndSel) {
      bool hasBranched = false;
      if ( event[dipEndSel->iRecoiler].isFinal())
           hasBranched = branch_IF(event, true, &splitInfoSel);
      else hasBranched = branch_II(event, true, &splitInfoSel);
      if (!hasBranched) {
        dipEndSel = 0;
        iDipSel   =-1;
        pT2sel    = 0.;
        splitInfoSel.clear();
        splittingSel = 0;
      }
    }

  }

  alphaSorder = asOrderSave;
  pT2minMECs  = pT2minMECsSave;

  // Unable to produce splitting.
  if (dipEndSel == 0) return 0.;

  // Return nonvanishing value if found pT bigger than already found.
  return sqrt(pT2sel);

}

//--------------------------------------------------------------------------

// Select next pT in downwards evolution of the existing dipoles.

double DireSpace::pTnext( Event& event, double pTbegAll, double pTendAll,
  int nRadIn, bool doTrialIn) {

  direInfoPtr->message(1) << "Next ISR starting from " << pTbegAll << endl;

  // Current cm energy, in case it varies between events.
  sCM           = m2( beamAPtr->p(), beamBPtr->p());
  eCM           = sqrt(sCM);
  pTbegRef      = pTbegAll;

  // Starting values: no radiating dipole found.
  nRad          = nRadIn;
  double pT2sel = pow2(pTendAll);
  iDipSel       = 0;
  iSysSel       = 0;
  dipEndSel     = 0;
  splittingNowName="";
  splittingSelName="";
  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) it->second->splitInfo.clear();
  splitInfoSel.clear();
  kernelSel.clear();
  kernelNow.clear();
  auxSel = overSel = auxNow = overNow = 0.;
  splittingSel = 0;

  // Remember if this is a trial emission.
  doTrialNow    = doTrialIn;

  // Loop over all possible dipole ends.
  for (int iDipEnd = 0; iDipEnd < int(dipEnd.size()); ++iDipEnd) {
    iDipNow        = iDipEnd;
    dipEndNow      = &dipEnd[iDipEnd];
    iSysNow        = dipEndNow->system;
    double pTbegDip = min( pTbegAll, dipEndNow->pTmax );

    // Limit final state multiplicity. For debugging only
    int nfmax = settingsPtr->mode("DireSpace:nFinalMax");
    int nFinal = 0;
    for (int i=0; i < event.size(); ++i)
      if (event[i].isFinal()) nFinal++;
    if (nfmax > -10 && nFinal > nfmax) continue;

    // Check whether dipole end should be allowed to shower.
    double pT2begDip = pow2(pTbegDip);

    if (pT2begDip > pT2sel) {
      double pT2endDip = 0.;

      // Determine lower cut for evolution for QCD
      pT2endDip = max( pT2cutMin(dipEndNow), pTendAll*pTendAll);
      pT2endDip = max(pT2endDip, pT2sel);

      // Find properties of dipole and radiating dipole end.
      sideA         = ( abs(dipEndNow->side) == 1 );
      bool finalRec = event[dipEndNow->iRecoiler].isFinal();
      BeamParticle& beamNow = (sideA) ? *beamAPtr : *beamBPtr;
      BeamParticle& beamRec = (sideA) ? *beamBPtr : *beamAPtr;
      iNow          = beamNow[iSysNow].iPos();
      iRec          = (finalRec) ? dipEndNow->iRecoiler
                                 : beamRec[iSysNow].iPos();
      idDaughter    = beamNow[iSysNow].id();
      xDaughter     = beamNow[iSysNow].x();
      x1Now         = (sideA) ? xDaughter : beamRec[iSysNow].x();
      x2Now         = (sideA) ? beamRec[iSysNow].x() : xDaughter;

      // Note dipole mass correction when recoiler is a rescatter.
      m2Rec         = (dipEndNow->normalRecoil) ? 0. : event[iRec].m2();
      m2Dip         = abs(2.*event[iNow].p()*event[iRec].p());

      // Dipole properties.
      dipEndNow->m2Dip  = m2Dip;
      // Reset emission properties.
      dipEndNow->pT2         =  0.0;
      dipEndNow->z           = -1.0;
      dipEndNow->phi         = -1.0;
      // Reset properties of 1->3 splittings.
      dipEndNow->sa1         =  0.0;
      dipEndNow->xa          = -1.0;
      dipEndNow->phia1       = -1.0;

      // Now do evolution in pT2, for QCD
      if (pT2begDip > pT2endDip) {

        if ( dipEndNow->canEmit() ) pT2nextQCD( pT2begDip, pT2endDip,
          *dipEndNow, event);

        // Update if found larger pT than current maximum.
        if (dipEndNow->pT2 > pT2sel) {
          pT2sel    = dipEndNow->pT2;
          iDipSel   = iDipNow;
          iSysSel   = iSysNow;
          dipEndSel = dipEndNow;
          splittingSelName = splittingNowName;
          splittingSel = splits[splittingSelName];
          splitInfoSel.store(splits[splittingSelName]->splitInfo);
          kernelSel = kernelNow;
          auxSel    = auxNow;
          overSel   = overNow;
          boostSel  = boostNow;
        }
      }
    }
  // End loop over dipole ends.
  }

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

  // Spacelike shower tries to propose state changes last. If no state change
  // occurs and timelike showers have also terminated, finalize the event with
  // e.g. decaying resonances etc.
  if (dipEndSel == 0) {
    for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
      it != splits.end(); ++it ) {
      if (it->second->fsr && it->second->fsr->dipSel == 0) {
        it->second->fsr->finalize(event);
        break;
      }
    }
  }

  // Return nonvanishing value if found pT is bigger than already found.
  return (dipEndSel == 0) ? 0. : sqrt(pT2sel);

}


//--------------------------------------------------------------------------

// Select next pT in downwards evolution of the existing dipoles.

double DireSpace::noEmissionProbability( double pTbegAll, double pTendAll,
  double m2dip, int idA, int type, double s, double x) {

  useSystems = false;

  // Current cm energy, in case it varies between events.
  sCM           = s;
  eCM           = sqrt(s);
  pTbegRef      = pTbegAll;

  double x1 = x;
  double x2 = m2dip/s/x1;

  // Starting values: no radiating dipole found.
  splittingNowName="";
  splittingSelName="";
  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) it->second->splitInfo.clear();

  // Make dummy event with two entries.
  Event state;
  state.init("(dummy event)", particleDataPtr);
  // Setup two dipole ends for each flavor combination.
  Vec4 pA(0., 0., 0.5*sqrt(m2dip), 0.5*sqrt(m2dip)), pB;
  if (type < 0) pB.p(0., 0.,-0.5*sqrt(m2dip), 0.5*sqrt(m2dip));
  if (type > 0) pB.p(0., 0.,0.5*sqrt(m2dip), 0.5*sqrt(m2dip));

  int colA  = 1;
  int acolA = 2;
  if (particleDataPtr->colType(idA) == 1) {colA = 1; acolA = 0;}
  if (particleDataPtr->colType(idA) ==-1) {colA = 0; acolA = 1;}

  // Add recoiler. For 1->3 splitting, attach "dummy" recoiler.
  state.append( 0, 0, 0, 0, 0, 0, 0, 0, pA+pB, 0.0, sqrt(m2dip) );
  state.append( idA, -21, 0, 0, 0, 0, colA, acolA, pA, 0.0, sqrt(m2dip) );

  // Now loop through possible recoilers.
  int idB  = (idA == 21) ? 21
           : ((type < 0) ? -idA : idA);
  vector<int> recids; recids.push_back(idB);
  vector<int> recpos;

  for (unsigned int i = 0; i < recids.size(); ++i) {
    int colB(2), acolB(1);
    if ( type < 0
      && particleDataPtr->colType(idA) == 1
      && particleDataPtr->colType(recids[i])   ==-1) {colB = 0; acolB = colA;}
    if ( type < 0
      && particleDataPtr->colType(idA) ==-1
      && particleDataPtr->colType(recids[i])   == 1) {colB = acolA; acolB = 0;}

    if ( type < 0
      && particleDataPtr->colType(idA) == 2
      && particleDataPtr->colType(recids[i])   ==-1) {colB = 0; acolB = colA;}
    if ( type < 0
      && particleDataPtr->colType(idA) == 2
      && particleDataPtr->colType(recids[i])   == 1) {colB = acolA; acolB = 0;}

    if (type < 0) state.append( recids[i], -21, 0, 0, 0, 0, colB, acolB,
                                pB, 0.0, sqrt(m2dip) );
    if (type > 0) state.append( recids[i],  23, 0, 0, 0, 0, colB, acolB,
                                pB, 0.0, sqrt(m2dip) );
    recpos.push_back(i+1);
  }

  beamAPtr->clear();
  beamBPtr->clear();
  beamAPtr->append( 1, idA, x1);
  beamBPtr->append( 2, idB, x2);
  beamAPtr->xfISR( 0, idA, x1, pTbegAll*pTbegAll);
  int vsc1 = beamAPtr->pickValSeaComp();
  beamBPtr->xfISR( 0, idB, x2, pTbegAll*pTbegAll);
  int vsc2 = beamBPtr->pickValSeaComp();
  infoPtr->setValence( (vsc1 == -3), (vsc2 == -3));

  // Store participating partons as first set in list of all systems.
  partonSystemsPtr->clear();
  partonSystemsPtr->addSys();
  partonSystemsPtr->setInA(0, 1);
  partonSystemsPtr->setInB(0, 2);
  partonSystemsPtr->setSHat( 0, m2dip);
  partonSystemsPtr->setPTHat( 0, pTbegAll);

  // Find positions of incoming colliding partons.
  int in1 = 1;
  vector<DireSpaceEnd> dipEnds;
  int colTag = state[in1].col();
  if (colTag > 0)  getQCDdip( in1,  colTag,  1, state, dipEnds);
  int acolTag = state[in1].acol();
  if (acolTag > 0) getQCDdip( in1, acolTag, -1, state, dipEnds);

  // Set starting scale.
  double startingScale = pTbegAll;

  // Set output.
  double wt            = 1.;

  while ( true ) {

    // Reset process scale so that shower starting scale is correctly set.
    state.scale(startingScale);

    // Get pT before reclustering
    double minScale = pTendAll;

    mergingHooksPtr->setShowerStoppingScale(minScale);

    // If the maximal scale and the minimal scale coincide (as would
    // be the case for the corrected scales of unordered histories),
    // do not generate Sudakov
    if (minScale >= startingScale) break;

    // Get trial shower pT.
    double pTtrial = pTnext( dipEnds, state, startingScale, minScale,
                             m2dip, type, s, x);

    pair<double,double> wtShower
      = weights->getWeight( (pTtrial <= 0.) ? pow2(minScale) : pow2(pTtrial));

    double enhancement = 1.;
    if ( pTtrial > minScale) enhancement
      = weights->getTrialEnhancement( pow2(pTtrial));

    weights->reset();
    weights->clearTrialEnhancements();

    // Done if evolution scale has fallen below minimum
    if ( pTtrial < minScale ) { wt *= wtShower.second; break;}

    // Reset starting scale.
    startingScale = pTtrial;

    if ( pTtrial > minScale) wt *= wtShower.first*wtShower.second
                                 * (1.-1./enhancement);
    if ( wt == 0.) break;
    if ( pTtrial > minScale) continue;

    // Done
    break;

  }

  beamAPtr->clear();
  beamBPtr->clear();
  partonSystemsPtr->clear();

  // Done
  double res = wt;
  return res;

}

double DireSpace::pTnext( vector<DireSpaceEnd> dipEnds, Event event,
  double pTbegAll, double pTendAll, double m2dip, int, double s,
  double x) {

  double x1 = x;
  double x2 = m2dip/s/x1;
  int iSys = 0;

  // Starting values: no radiating dipole found.
  double pT2sel = pow2(pTendAll);
  iDipSel       = 0;
  iSysSel       = 0;
  dipEndSel     = 0;
  splittingNowName="";
  splittingSelName="";
  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) it->second->splitInfo.clear();
  splitInfoSel.clear();
  kernelSel.clear();
  kernelNow.clear();
  auxSel = overSel = auxNow = overNow = 0.;
  splittingSel = 0;


  // Set splitting library.
  splits = splittingsPtr->getSplittings();
  overhead.clear();
  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) overhead.insert(make_pair(it->first,1.));

  // Counter of proposed emissions.
  nProposedPT.clear();
  if ( nProposedPT.find(iSys) == nProposedPT.end() )
    nProposedPT.insert(make_pair(iSys,0));

  splittingSelName="";
  splittingNowName="";
  dipEndSel = 0;

  // Clear weighted shower book-keeping.
  for ( unordered_map<string, multimap<double,double> >::iterator
    it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
    it->second.clear();
  for ( unordered_map<string, map<double,double> >::iterator
    it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
    it->second.clear();

  // Loop over all possible dipole ends.
  for (int iDipEnd = 0; iDipEnd < int(dipEnds.size()); ++iDipEnd) {

    iDipNow        = iDipEnd;
    dipEndNow      = &dipEnds[iDipEnd];
    double pTbegDip = min( pTbegAll, dipEndNow->pTmax );

    // Limit final state multiplicity. For debugging only
    int nFinal = 0;
    for (int i=0; i < event.size(); ++i)
      if (event[i].isFinal()) nFinal++;
    if (nFinalMax > -10 && nFinal > nFinalMax) continue;

    // Check whether dipole end should be allowed to shower.
    double pT2begDip = pow2(pTbegDip);
    double pT2endDip = 0.;
    // Determine lower cut for evolution for QCD
    pT2endDip = max( pT2cutMin(dipEndNow), pTendAll*pTendAll);
    pT2endDip = max(pT2endDip, pT2sel);

    // Find properties of dipole and radiating dipole end.
    sideA         = ( abs(dipEndNow->side) == 1 );
    iNow          = dipEndNow->iRadiator;
    iRec          = dipEndNow->iRecoiler;
    idDaughter    = event[dipEndNow->iRadiator].id();
    xDaughter     = x1;
    x1Now         = (sideA) ? x1 : x2;
    x2Now         = (sideA) ? x2 : x1;
    // Note dipole mass correction when recoiler is a rescatter.
    m2Rec         = (dipEndNow->normalRecoil) ? 0. : event[iRec].m2();
    m2Dip         = abs(2.*event[iNow].p()*event[iRec].p());

    // Dipole properties.
    dipEndNow->m2Dip  = m2Dip;
    // Reset emission properties.
    dipEndNow->pT2         =  0.0;
    dipEndNow->z           = -1.0;
    dipEndNow->phi         = -1.0;
    // Reset properties of 1->3 splittings.
    dipEndNow->sa1         =  0.0;
    dipEndNow->xa          = -1.0;
    dipEndNow->phia1       = -1.0;

    // Now do evolution in pT2, for QCD
    if (pT2begDip > pT2endDip) {

      if ( dipEndNow->canEmit() ) pT2nextQCD( pT2begDip, pT2endDip,
        *dipEndNow, event);

      // Update if found larger pT than current maximum.
      if (dipEndNow->pT2 > pT2sel) {
        pT2sel    = dipEndNow->pT2;
        iDipSel   = iDipNow;
        iSysSel   = iSys;
        dipEndSel = dipEndNow;
        splittingSelName = splittingNowName;
        splittingSel = splits[splittingSelName];
        splitInfoSel.store(splits[splittingSelName]->splitInfo);
        kernelSel = kernelNow;
        auxSel    = auxNow;
        overSel   = overNow;
        boostSel  = boostNow;
      }

    }
  // End loop over dipole ends.
  }

  // Insert additional weights.
  for ( unordered_map<string, multimap<double,double> >::iterator
    itR = rejectProbability.begin(); itR != rejectProbability.end(); ++itR){
    weights->insertWeights(acceptProbability[itR->first], itR->second,
                           itR->first);
  }

  for ( unordered_map<string, multimap<double,double> >::iterator
    it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
    it->second.clear();
  for ( unordered_map<string, map<double,double> >::iterator
    it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
    it->second.clear();

  resetOverheadFactors();

  // Return nonvanishing value if found pT is bigger than already found.
  return (dipEndSel == 0) ? 0. : sqrt(pT2sel);

}

//--------------------------------------------------------------------------

double DireSpace::enhanceOverestimateFurther( string name, int,
  double tOld) {

  if (tOld < pT2minEnhance) return 1.;
  double enhance = weights->enhanceOverestimate(name);
  return enhance;

}

//--------------------------------------------------------------------------

double DireSpace::overheadFactors( string name, int idDau, bool isValence,
  double m2dip, double pT2Old ) {

  double factor = 1.;

  // Additional weight to smooth out valence bump.
  if (isValence && name.find("isr_qcd_1->1&21") != string::npos)
    factor *= log(max(2.71828,16/(pT2Old/m2dip)));

  // Additional enhancement for G->QQ, to smooth out PDF factors.
  if (name.find("isr_qcd_21->1&1") != string::npos)
    factor *= log(max(2.71828,log(max(2.71828,m2dip/pT2Old))
                    + pow(m2dip/pT2Old,3./2.)));

  // Artificial constant increase of overestimate.
  double MARGIN = 1.;
  if (name.find("isr_qcd_1->1&21") != string::npos && !isValence)
    MARGIN = 1.65;
  if (name.find("isr_qcd_1->21&1") != string::npos && !isValence)
    MARGIN = 1.65;
  if (name.find("isr_qcd_21->1&1") != string::npos)
    MARGIN = 1.65;
  if (name.find("isr_qcd_21->21&21a") != string::npos && pT2Old < 2.0)
    MARGIN = 1.25;
  if (name.find("isr_qcd_21->21&21b") != string::npos && pT2Old < 2.0)
    MARGIN = 1.25;

  // For very low cut-offs, do not artificially increase overestimate.
  if (pT2Old < pT2min*1.25) MARGIN = 1.0;

  factor *= MARGIN;

  // Further enhance charm/bottom conversions close to threshold.
  if ( abs(idDau) == 4 && name.find("isr_qcd_21->1&1") != string::npos
    && pT2Old < 2.*m2cPhys) factor *= 1. / max(0.01, abs(pT2Old - m2cPhys));
  if ( abs(idDau) == 5 && name.find("isr_qcd_21->1&1") != string::npos
    && pT2Old < 2.*m2bPhys) factor *= 1. / max(0.01, abs(pT2Old - m2bPhys));

  // Multiply dynamically adjusted overhead factor.
  if ( overhead.find(name) != overhead.end() ) factor *= overhead[name];

  return factor;

}

//--------------------------------------------------------------------------

// Function to generate new user-defined overestimates to evolution.

void DireSpace::getNewOverestimates( int idDau, DireSpaceEnd* dip,
  const Event& state, double tOld, double xDau, double zMinAbs,
  double zMaxAbs, multimap<double,string>& newOverestimates ) {

  // Get beam for correction factors.
  BeamParticle& beam = (sideA) ? *beamAPtr : *beamBPtr;
  bool   isValence   = (usePDF) ? beam[iSysNow].isValence() : false;
  pair<int,int> iRadRec(make_pair(dip->iRadiator, dip->iRecoiler));

  double sum=0.;

  // Loop over splitting names and get overestimates.
  for ( unordered_map<string,DireSplitting*>::iterator it = splits.begin();
    it != splits.end(); ++it ) {

    string name = it->first;

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

    // No 1->3 conversion of heavy quarks below 2*m_q.
    if ( tOld < 4.*m2bPhys && abs(idDau) == 5
      && it->second->nEmissions() == 2) continue;
    else if ( tOld < 4.*m2cPhys && abs(idDau) == 4
      && it->second->nEmissions() == 2) continue;

    // Get kernel order.
    int order = kernelOrder;
    // Use simple kernels for showering secondary scatterings.
    bool hasInA = (getInA(dip->system) != 0);
    bool hasInB = (getInB(dip->system) != 0);
    if (dip->system != 0 && hasInA && hasInB) order = kernelOrderMPI;

    it->second->splitInfo.set_pT2Old  ( tOld );
    it->second->splitInfo.storeRadBef(state[dip->iRadiator]);
    it->second->splitInfo.storeRecBef(state[dip->iRecoiler]);

    // Discard below the cut-off for the splitting.
    if (!it->second->aboveCutoff( tOld, state[dip->iRadiator],
      state[dip->iRecoiler], dip->system, partonSystemsPtr)) continue;

    // Get overestimate (of splitting kernel only)
    double wt = it->second->overestimateInt(zMinAbs, zMaxAbs, tOld,
                                           dip->m2Dip, order);

    // Calculate numerator of PDF ratio, and construct ratio.
    // PDF factors for Q -> GQ.
    double pdfRatio = getPDFOverestimates(idDau, tOld, xDau, it->first,
      false, -1., re[0], re[0]);

    // Include PDF ratio for Q->GQ or G->QQ.
    wt *= pdfRatio;

    // Include artificial enhancements.
    double headRoom =
      overheadFactors(name, idDau, isValence, dip->m2Dip, tOld);
    wt *= headRoom;

    // Now add user-defined enhance factor.
    double enhanceFurther = enhanceOverestimateFurther(name, idDau, tOld);
    wt *= enhanceFurther;

    if (!dryrun && it->second->hasMECBef(state, tOld)) wt *= KERNEL_HEADROOM;
    int nFinal = 0;
    for (int i=0; i < state.size(); ++i) if (state[i].isFinal()) nFinal++;
    if (!dryrun) wt *= it->second->overhead
                   (dip->m2Dip*xDau, state[dip->iRadiator].id(), nFinal);

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

double DireSpace::getPDFOverestimates( int idDau, double tOld, double xDau,
  string name, bool pickMother, double RN, int& idMother, int& idSister) {

  BeamParticle& beam = (sideA) ? *beamAPtr : *beamBPtr;
  DireSplitting* splitNow = splits[name];

  // Get old PDF for PDF weights.
  double PDFscale2 = (useFixedFacScale) ? fixedFacScale2 : factorMultFac*tOld;
  PDFscale2        = max(PDFscale2, pT2min);
  bool inD         = (hasPDF(idDau)) ? beam.insideBounds(xDau, PDFscale2) :
    true;
  double xPDFdaughter = getXPDF( idDau, xDau, PDFscale2, iSysNow, &beam);
  // Make PDF ratio overestimate larger close to threshold.
  if (abs(idDau) == 4 && m2cPhys > 0. && tOld < 4.*m2cPhys) {
    double xPDFthres = getXPDF( idDau, xDau, m2cPhys+0.1, iSysNow, &beam);
    xPDFdaughter     = min(xPDFdaughter, xPDFthres);
  }
  if (abs(idDau) == 5 && m2bPhys > 0. && tOld < 4.*m2bPhys) {
    double xPDFthres = getXPDF( idDau, xDau, m2bPhys, iSysNow, &beam);
    xPDFdaughter     = min(xPDFdaughter, xPDFthres);
  }

  // Calculate numerator of PDF ratio, and construct ratio.
  // PDF factors for Q -> GQ.
  double pdfRatio = 1.;
  if (splitNow->is(splittingsPtr->isrQCD_1_to_21_and_1)) {

    // Get sum of PDFs for all potential quark mothers to a g.
    double xPDFmotherSum = 0.;
    double xPDFmother[21] = {0.};
    for (int iQuark = -nQuarkIn; iQuark <= nQuarkIn; ++iQuark) {
      if (iQuark == 0) {
        xPDFmother[10] = 0.;
      } else {
        xPDFmother[iQuark+10] = getXPDF(iQuark,xDau,PDFscale2,iSysNow,&beam);
        xPDFmotherSum += xPDFmother[iQuark+10];
      }
    }

    if (pickMother) {
      // Pick flavour for Q -> GQ (picked here, as kernel might depend on
      // flavour)
      double temp = xPDFmotherSum * RN;
      idMother = -nQuarkIn - 1;
      do { temp -= xPDFmother[(++idMother) + 10]; }
      while (temp > 0. && idMother < nQuarkIn);
      idSister = idMother;
      pdfRatio = xPDFmother[idMother + 10]/xPDFdaughter;
    } else {
      pdfRatio = xPDFmotherSum/xPDFdaughter;
    }

  // PDF factors for G -> QQ
  } else if (splitNow->is(splittingsPtr->isrQCD_21_to_1_and_1)){

    double xPDFmother = getXPDF(21, xDau, PDFscale2, iSysNow, &beam);
    if ( xPDFmother != 0. && abs(xPDFmother) < tinypdf(xDau) ) {
      int sign   = (xPDFmother >= 0.) ? 1 : -1;
      xPDFmother = sign*tinypdf(xDau);
    }
    pdfRatio = xPDFmother / xPDFdaughter;

  // PDF factors for q --> q' splitting.
  } else if (splitNow->is(splittingsPtr->isrQCD_1_to_2_and_1_and_2)) {

    // Sum over all potential mothers q'.
    multimap<double, pair<int,double> > xPDFmother;
    double xPDFmotherSum = 0.;
    for (int i =-nQuarkIn; i <= nQuarkIn; ++i)
      if (abs(i) != abs(idDau) && i != 0) {
        double temp = getXPDF( i, xDau, PDFscale2, iSysNow, &beam);
        // Make overestimate larger if heavy quark converts to valence quark.
        if (particleDataPtr->isHadron(beam.id()) && (i == 1 || i == 2)) {
          if (abs(idDau) == 4 && m2cPhys > 0. && tOld < 4.*m2cPhys) {
            double xPDFval = getXPDF(i, 0.25, PDFscale2, iSysNow, &beam);
            temp = max(temp, xPDFval);
          }
          if (abs(idDau) == 5 && m2bPhys > 0. && tOld < 4.*m2bPhys) {
            double xPDFval = getXPDF( i, 0.25, PDFscale2, iSysNow, &beam);
            temp = max(temp, xPDFval);
          }
        }
        xPDFmotherSum += temp;
        xPDFmother.insert(make_pair(xPDFmotherSum, make_pair(i, temp) ) );
      }

    if (pickMother) {
      // Pick flavour.
      double R = xPDFmotherSum * RN;
      if (xPDFmother.lower_bound(R) == xPDFmother.end()) {
        idMother = xPDFmother.rbegin()->second.first;
        pdfRatio = xPDFmother.rbegin()->second.second / xPDFdaughter;
      } else {
        idMother = xPDFmother.lower_bound(R)->second.first;
        pdfRatio = xPDFmother.lower_bound(R)->second.second / xPDFdaughter;
      }
      idSister = idMother;
    } else {
      pdfRatio = xPDFmotherSum / xPDFdaughter;
    }

  // PDF factors for q --> qbar splitting.
  } else if (splitNow->is(splittingsPtr->isrQCD_1_to_1_and_1_and_1)) {

    double xPDFmother = getXPDF( -idDau, xDau, PDFscale2, iSysNow, &beam);
    if ( xPDFmother != 0. && abs(xPDFmother) < tinypdf(xDau) ) {
      int sign   = (xPDFmother >= 0.) ? 1 : -1;
      xPDFmother = sign*tinypdf(xDau);
    }
    pdfRatio = xPDFmother / xPDFdaughter;

  } else if ( tOld < PT2_INCREASE_OVERESTIMATE
    && ( splitNow->is(splittingsPtr->isrQCD_21_to_21_and_21a)
      || splitNow->is(splittingsPtr->isrQCD_21_to_21_and_21b))) {
    double xPDFmother = xPDFdaughter;
    int NTSTEPS(3), NXSTEPS(3);
    for (int i=1; i <= NTSTEPS; ++i) {
      double tNew = pT2min + double(i)/double(NTSTEPS)*(max(tOld, pT2min)
                                                        - pT2min);
      for (int j=1; j <= NXSTEPS; ++j) {
        double xNew = xDau + double(j)/double(NXSTEPS)*(0.999999-xDau);
        double xPDFnew = getXPDF( 21, xNew, tNew, iSysNow, &beam);
        xPDFmother = max(xPDFmother, xPDFnew);
      }
    }
    pdfRatio = xPDFmother/xPDFdaughter;

  // All other cases, e.g. for user-defined kernels.
  } else {
    double xPDFmother = getXPDF(idMother,xDau, PDFscale2, iSysNow, &beam);

    int NTSTEPS(3), NXSTEPS(3);
    for (int i=0; i <= NTSTEPS; ++i) {
      double tNew = pT2min + double(i)/double(NTSTEPS)*(max(tOld, pT2min)
                                                        - pT2min);
      for (int j=1; j <= NXSTEPS; ++j) {
        double xNew = xDau + double(j)/double(NXSTEPS)*(0.999999-xDau);
        double xPDFnew = getXPDF(idMother, xNew, tNew, iSysNow, &beam);
        xPDFmother = max(xPDFmother, xPDFnew);
      }
    }
    pdfRatio = xPDFmother/xPDFdaughter;
  }

  if (pdfRatio < 0. || abs(xPDFdaughter) < tinypdf(xDau)) pdfRatio = 0.;
  if (!inD) pdfRatio = 0.;

  return pdfRatio;

}

//--------------------------------------------------------------------------

// Function to generate new user-defined overestimates to evolution.

void DireSpace::getNewSplitting( const Event& state, DireSpaceEnd* dip,
  double tOld, double xDau, double t, double zMinAbs, double zMaxAbs,
  int idDau, string name, bool forceFixedAs, int& idMother, int& idSister,
  double& z, double& wt, unordered_map<string,double>& full, double& over ) {

  BeamParticle& beam = (sideA) ? *beamAPtr : *beamBPtr;
  bool   isValence   = (usePDF) ? beam[iSysNow].isValence() : false;
  // Pointer to splitting for easy/fast access.
  DireSplitting* splitNow = splits[name];

  splitNow->splitInfo.storeRadBef(state[dip->iRadiator]);
  splitNow->splitInfo.storeRecBef(state[dip->iRecoiler]);
  bool canUseSplitInfo = splitNow->canUseForBranching();

  //// Return auxiliary variable, overestimate, mother and sister ids.
  vector<int> re = splitNow->radAndEmt(idDau, dip->colType);
  idMother = re[0];
  idSister = re[1];

  // Reject below cut-off.
  if ( pT2cut(idSister) > t) { wt = over = 0.; full.clear(); return; }

  // Return auxiliary variable, overestimate, mother and sister ids.
  if(z< 0.) z = splitNow->zSplit(zMinAbs, zMaxAbs, dip->m2Dip);
  over        = splitNow->overestimateDiff(z, dip->m2Dip);

  // Discard below the cut-off for the splitting.
  if (!splitNow->aboveCutoff( t, state[dip->iRadiator], state[dip->iRecoiler],
    dip->system, partonSystemsPtr)) { wt = over = 0.; full.clear(); return; }

  // Calculate numerator of PDF ratio, and construct ratio.
  double RNflav = rndmPtr->flat();;
  double pdfRatio = getPDFOverestimates(idDau, tOld, xDau, name, true, RNflav,
    idMother, idSister);

  // Get particle masses.
  double m2Bef = 0.0;
  double m2r   = 0.0;
  double m2s   = 0.0;
  int type     = (state[dip->iRecoiler].isFinal()) ? 1 : -1;
  if (type == 1) {
    m2s = particleDataPtr->isResonance(state[dip->iRecoiler].id())
        ? getMass(state[dip->iRecoiler].id(),3,
                  state[dip->iRecoiler].mCalc())
        : (state[dip->iRecoiler].idAbs() < 6)
        ? getMass(state[dip->iRecoiler].id(),2)
        : getMass(state[dip->iRecoiler].id(),1);
  }

  // Force emission massless for now.
  double m2e   = 0.0;

  // Upate type if this is a massive splitting.
  if (type == 1 && (m2Bef > TINYMASS || m2r > TINYMASS || m2e > TINYMASS
    || m2s > TINYMASS)) type = 2;
  if (type ==-1 && (m2Bef > TINYMASS || m2r > TINYMASS || m2e > TINYMASS
    || m2s > TINYMASS)) type =-2;

  // Adjust the dipole kinematical mass to accomodate masses after branching.
  double m2dipCorr  = dip->m2Dip - m2Bef + m2r + m2e;

  // Set kinematics mapping, as needed to check limits.
  // 1 --> Dire
  // 2 --> Catani-Seymour
  int kinType = splitNow->kinMap();

  dip->z = z;
  dip->pT2 = t;
  // Already pick phi value here, since we may need to construct the
  // momenta to evaluate the splitting probability.
  dip->phi   = 2.*M_PI*rndmPtr->flat();

  dip->mass.clear();

  double m2i = getMass(idMother,2);
  double m2j = getMass(idSister,2);
  bool physical = true;
  // Generate additional variables for 1->3 splitting.
  if ( splitNow->nEmissions() == 2 ) {
    dip->mass.push_back(m2r);
    dip->mass.push_back(m2i);
    dip->mass.push_back(m2j);
    dip->mass.push_back(m2s);
    // Choose xa flat in [z, 1.0]
    zCollNextQCD( dip, dip->z, 1. );
    // Choose sai flat in [0.0, m2Dip]
    physical = virtNextQCD( dip, 0.0, dip->m2Dip);
    // Choose phi flat in [0, 2Pi]
    dip->phia1 = 2.*M_PI*rndmPtr->flat();
  }

  // Set correct variables for 1->3 splitting.
  vector <double> aux;
  if ( splitNow->nEmissions() == 2 ) {
    type       = (state[dip->iRecoiler].isFinal()) ? 2 : -2;
    aux.push_back( dip->m2Dip );
    if (type > 0) aux.push_back( (state[dip->iRadiator].p()
                                 -state[dip->iRecoiler].p()).m2Calc() );
    else          aux.push_back( (state[dip->iRadiator].p()
                                 +state[dip->iRecoiler].p()).m2Calc() );
    aux.push_back(dip->pT2);
    aux.push_back(dip->sa1);
    aux.push_back(dip->z);
    aux.push_back(dip->xa);
    aux.push_back(m2Bef);
    aux.push_back(m2r);
    aux.push_back(m2i);
    aux.push_back(m2j);
    aux.push_back(m2s);
  }

  // Setup splitting information.
  int nEmissions = splitNow->nEmissions();
  splitNow->splitInfo.storeInfo(name, type, dip->system, dip->system,
    dip->side, dip->iRadiator, dip->iRecoiler, state, idSister, idMother,
    nEmissions, m2dipCorr, dip->pT2, dip->pT2Old, dip->z, dip->phi, m2Bef,
    m2s, m2r, (nEmissions == 1 ? m2e : m2i), dip->sa1, dip->xa, dip->phia1,
    m2j, xDau, -1.);

  if (canUseSplitInfo) {
    splitNow->splitInfo.setRadAft(re[0]);
    splitNow->splitInfo.setEmtAft(re[1]);
    if (nEmissions==2) splitNow->splitInfo.setEmtAft2(re[2]);
    splitNow->splitInfo.canUseForBranching(true);
  } else {
    splitNow->splitInfo.setRadAft(idMother);
    splitNow->splitInfo.setEmtAft(idSister);
    if (nEmissions==2) splitNow->splitInfo.setEmtAft2(-idSister);
  }

  // Check phase space limits.
  double zcheck(z), tcheck(t);
  if (kinType==99) {
    zcheck = (type<0)
      ? splitNow->zdire_ii(z, t, m2dipCorr)
      : splitNow->zdire_if(z, t, m2dipCorr);
    tcheck = (type<0)
      ? splitNow->tdire_ii(z, t, m2dipCorr)
      : splitNow->tdire_if(z, t, m2dipCorr);
  }
  if ( !physical || !inAllowedPhasespace( kinType, zcheck, tcheck, m2dipCorr,
          xDau, type, m2Bef, m2r, m2s, m2e, aux ) )
    { wt = over = 0.; full.clear(); return; }

  // Get kernel order.
  int order = kernelOrder;
  // Use simple kernels for showering secondary scatterings.
  bool hasInA = (getInA(dip->system) != 0);
  bool hasInB = (getInB(dip->system) != 0);
  if (dip->system != 0 && hasInA && hasInB) order = kernelOrderMPI;

  // Set splitting colors, if necessary.
  if (canUseSplitInfo) {
    vector< pair<int,int> > cols
      = splitNow->radAndEmtCols( dip->iRadiator, dip->colType, state);
    splitNow->splitInfo.setRadAft(re[0], cols[0].first, cols[0].second);
    splitNow->splitInfo.setEmtAft(re[1], cols[1].first, cols[1].second);
    if (nEmissions==2) splitNow->splitInfo.setEmtAft2(re[2], cols[2].first,
      cols[2].second);
  }

  // Return overestimate.
  over        = splitNow->overestimateDiff(z, dip->m2Dip, order);

  // Get complete kernel.
  if (splitNow->calc( state, order) ) full = splitNow->getKernelVals();

  if (!dryrun && splitNow->hasMECBef(state, tOld)) over *= KERNEL_HEADROOM;
  if (!dryrun && splitNow->hasMECBef(state, dip->pT2))
    for (unordered_map<string,double>::iterator it=full.begin();
    it != full.end(); ++it) it->second *= KERNEL_HEADROOM;

  direInfoPtr->message(1) << __FILE__ << " " << __func__
    << " " << __LINE__ << " : New splitting "
    << setw(15) << name << " at pT="
    << setw(15) << sqrt(dip->pT2) << " z = "
    << setw(15) << dip->z << " prob = "
    << setw(15) << full["base"] << endl;

  // Reweight with coupling factor if necessary.
  double coupl = splitNow->coupling(dip->z, dip->pT2, m2dipCorr, -1.,
      make_pair(state[dip->iRadiator].id(), state[dip->iRadiator].isFinal()),
      make_pair(state[dip->iRecoiler].id(), state[dip->iRecoiler].isFinal()));
  // Retrieve argument of alphaS.
  double scale2 = splits[splittingNowName]->couplingScale2(dip->z, dip->pT2,
    m2dipCorr,
    make_pair (state[dip->iRadiator].id(), state[dip->iRadiator].isFinal()),
    make_pair (state[dip->iRecoiler].id(), state[dip->iRecoiler].isFinal()));
  if (scale2 < 0.) scale2 = dip->pT2;
  double talpha = max(scale2, pT2min);
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
  in.push_back(re[0]);
  for (size_t i=1; i < re.size(); ++i) out.push_back(re[i]);
  bool hasME = dip->pT2 > pT2minMECs && doMEcorrections
    && weights->hasME(in,out);
  if (hasME) for (unordered_map<string,double>::iterator it=full.begin();
    it != full.end(); ++it) it->second = abs(it->second);

  double mecover=1.;
  int nFinal = 0;
  for (int i=0; i < state.size(); ++i) if (state[i].isFinal()) nFinal++;
  if (!dryrun) mecover = splitNow->
                 overhead(dip->m2Dip*xDau, state[dip->iRadiator].id(), nFinal);
  for (unordered_map<string,double>::iterator it=full.begin();
    it != full.end(); ++it) it->second *= mecover;
  over *= mecover;

  // Calculate accept probability.
  wt          = full["base"]/over;

  // Divide out PDF ratio used in overestimate.
  if (pdfRatio != 0.) wt /= pdfRatio;
  else wt = 0.;
  over *= pdfRatio;

  // Divide out artificial enhancements.
  double headRoom = overheadFactors(name, idDau, isValence, dip->m2Dip, tOld);
  wt   /= headRoom;
  over *= headRoom;

  // Ensure positive weight.
  wt = abs(wt);

}

//--------------------------------------------------------------------------

pair<bool, pair<double, double> > DireSpace::getMEC ( const Event& state,
  DireSplitInfo* splitInfo) {

  double MECnum(1.0), MECden(1.0);

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
      makeHardEvent(max(0,splitInfo->system), state, false), false) );
    // Store candidates for the splitting V -> qqbar'
    mergingHooksPtr->storeHardProcessCandidates( newProcess );

    // Calculate number of clustering steps
    int nSteps = mergingHooksPtr->
      getNumberOfClusteringSteps( newProcess, true);
    // Set dummy process scale.
    newProcess.scale(0.0);
    // Generate all histories
    DireHistory myHistory( nSteps, 0.0, newProcess, DireClustering(),
      mergingHooksPtr, (*beamAPtr), (*beamBPtr), particleDataPtr, infoPtr,
      NULL, splits.begin()->second->fsr, splits.begin()->second->isr, weights,
      coupSMPtr, true, true, 1.0, 1.0, 1.0, 1.0, 0);
    // Project histories onto desired branches, e.g. only ordered paths.
    myHistory.projectOntoDesiredHistories();

    MECnum = myHistory.MECnum;
    MECden = myHistory.MECden;

    // Restore to previous mergingHooks setup.
    mergingHooksPtr->init();

  }

  if (abs(MECden) < 1e-15) direInfoPtr->message(1) << __FILE__ << " "
                                                   << __func__
    << " " << __LINE__ << " : Small MEC denominator="
    << MECden << " for numerator=" << MECnum << endl;
  if (abs(MECnum/MECden) > 1e2) {direInfoPtr->message(1) << __FILE__ << " "
                                                         << __func__
    << " " << __LINE__ << " : Large MEC. Denominator="
    << MECden << " Numerator=" << MECnum << " at pT="
    << sqrt(splitInfo->kinematics()->pT2) << " "
    << endl;
  }

  return make_pair(hasME, make_pair(MECnum,MECden));

}

//--------------------------------------------------------------------------

bool DireSpace::applyMEC ( const Event& state, DireSplitInfo* splitInfo,
  vector<Event> auxState) {

  // Get value of ME correction.
  pair<bool, pair<double, double> > mec = getMEC ( state, splitInfo);
  bool hasME    = mec.first;
  double MECnum = mec.second.first;
  double MECden = mec.second.second;
  double MECnumX = mec.second.first;
  double MECdenX = mec.second.second;

  if (!hasME) return false;

  double kernel = kernelSel["base"];
  bool reject   = false;

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
  double oas2    = 0.;
  if (kernelSel.find("base_order_as2") != kernelSel.end() ) {
    oas2 = kernelSel["base_order_as2"];
    kernelSel.erase(kernelSel.find("base_order_as2"));
  }
  double baseNew = ((kernel - oas2) * MECnum/MECden + oas2);

  // Now check if the splitting should be vetoed/accepted given new kernel.
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

  if (suppressLargeMECs)  while (baseNew/auxNew < 5e-2)  auxNew /= 5.;

  // Reset overestimate if necessary.
  if (baseNew/auxNew > 1.) {
    double rescale = baseNew/auxNew * 1.15;
    auxNew *= rescale;
  }
  double wt = baseNew/auxNew;

  // New rejection weight.
  double wvNow = auxNew/overNew * (overNew - baseNew) / (auxNew -  baseNew);

  // New acceptance weight.
  double waNow = auxNew/overNew;

  if (abs(wvNow) > 1e0) {
    direInfoPtr->message(1) << __FILE__ << " " << __func__
    << " " << __LINE__ << " : Large reject weight=" << wvNow
    << "\t for kernel=" << baseNew << " overestimate=" << overNew
    << "\t aux. overestimate=" << auxNew << " at pT2="
    << splitInfo->kinematics()->pT2
    <<  " for " << splittingSelName
    << endl;
  }
  if (abs(waNow) > 1e0) {
    direInfoPtr->message(1) << __FILE__ << " " << __func__
    << " " << __LINE__ << " : Large accept weight=" << waNow
    << "\t for kernel=" << baseNew << " overestimate=" << overNew
    << "\t aux. overestimate=" << auxNew << " at pT2="
    << splitInfo->kinematics()->pT2
    <<  " for " << splittingSelName
    << endl;
    }

  if (wt < rndmPtr->flat()) {

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
    // Loop through and reset weights.
    for (unordered_map<string,double>::iterator it = kernelSel.begin();
         it != kernelSel.end(); ++it) {
      // Get old accept weight.
      double waOld = weights->getAcceptWeight( splitInfo->kinematics()->pT2,
        it->first);
      // Remove previous reject weight and replace accept weight.
      weights->eraseRejectWeight(splitInfo->kinematics()->pT2, it->first);
      weights->resetAcceptWeight(splitInfo->kinematics()->pT2, waOld*waNow,
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

bool DireSpace::inAllowedPhasespace( int kinType, double z, double pT2,
  double m2dip, double xOld, int splitType, double m2RadBef, double m2r,
  double m2s, double m2e, vector<double> aux) {

  double xIncoming = usePDF ? xOld : 0.;

  // splitType == 1 -> Massless IF
  if (splitType == 1) {

    // Calculate CS variables.
    double kappa2 = pT2 / m2dip;
    double xCS = z;
    double uCS = kappa2/(1-z);

    // CS variables directly.
    if (kinType == 2) {
      xCS = z;
      uCS = 0.5*xCS*( 1. - sqrt( 1. - 4.*xCS*kappa2 / pow2( 1- xCS)) );
    }

    // Forbidden emission if outside allowed z range for given pT2.
    if ( xCS < xIncoming || xCS > 1. || uCS < 0. || uCS > 1. ) return false;

  // splitType == 2 -> Massive IF
  } else if (splitType == 2 && aux.size() == 0) {

    double kappa2 = pT2 / m2dip;
    double xCS = z;
    double uCS = kappa2/(1-z);
    // Construct massive phase space boundary for uCS
    double pijpa  = m2dip - m2r - m2e + m2RadBef;
    double mu2Rec = m2s / pijpa * xCS;
    double uCSmax = (1. - xCS) / (1. - xCS + mu2Rec );

    // Forbidden emission if outside allowed z range for given pT2.
    if (xCS < xIncoming || xCS > 1. || uCS < 0. || uCS > uCSmax) return false;

  // splitType == 2 -> Massive 1->3 IF
  } else if (splitType == 2 && aux.size() > 0) {

    // Not correctly set up!
    if ( int(aux.size()) < 11) return false;

    double q2    = aux[1];
    double t     = aux[2];
    double sai   = aux[3];
    double za    = aux[4];
    double xa    = aux[5];
    double m2a   = aux[7];
    double m2i   = aux[8];
    double m2j   = aux[9];
    double m2k   = aux[10];
    double m2ai  = -sai + m2a + m2i;

    // Check that IF-part is possible.
    double uCS   = za*(m2ai-m2a-m2i) / q2;
    double xCS   = uCS + xa - (t*za) / (q2*xa);
    double m2jk   = t/xa + q2*( 1. - xa/za ) - m2ai;

    if ( m2jk < 0. ) return false;

    double mu2Rec = m2jk/(-q2+m2jk) * xCS;
    double uCSmax = (1. - xCS) / (1. - xCS + mu2Rec );

    // Forbidden emission if outside allowed z range for given pT2.
    if (xCS < xIncoming || xCS > 1. || uCS < 0. || uCS > uCSmax) return false;

    // Check that kinematical kT is valid.
    double s_i_jk = (1. - 1./xCS)*(q2 - m2a) + (m2i + m2jk) / xCS;
    double zbar   = (q2-s_i_jk-m2a) / bABC(q2,s_i_jk,m2a)
                   *( uCS - m2a / gABC(q2,s_i_jk,m2a)
                         * (s_i_jk + m2i - m2jk) / (q2 - s_i_jk - m2a));
    double kT2   = zbar*(1.-zbar)*s_i_jk - (1-zbar)*m2i - zbar*m2jk;
    if (kT2 < 0.) return false;

    // Check that FF part is possible
    double zCS = t/xa / ( t/xa - q2*xa/za);
    double yCS = (m2jk - m2k - m2j)
               / (m2jk - m2k - m2j + t/xa - q2*xa/za);

    double q2_2 = m2ai + m2jk + t/xa - q2*xa/za;
    // Calculate derived variables.
    double sij  = yCS * (q2_2 - m2ai) + (1.-yCS)*(m2j+m2k);
    zbar = (q2_2-sij-m2ai) / bABC(q2_2,sij,m2ai)
                * (zCS - m2ai/gABC(q2_2,sij,m2ai)
                       *(sij + m2j - m2k)/(q2_2-sij-m2ai));
    kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2j - zbar*m2k;

    if (kT2 < 0.) return false;

  // splitType ==-1 -> Massless II
  } else if (splitType == -1) {

    // Calculate CS variables.
    double kappa2 = pT2 / m2dip;
    double xCS    = (z*(1-z)- kappa2)/(1-z);
    double vCS    = kappa2/(1-z);

    // CS variables directly.
    if (kinType == 2) {
      xCS = z;
      vCS = 0.5*xCS*( 1. - sqrt( 1. - 4.*xCS*kappa2 / pow2( 1- xCS)) );
    }

    // Forbidden emission if outside allowed z range for given pT2.
    if (xCS < xIncoming || xCS > 1. || vCS < 0. || vCS > 1.) return false;
    if (1.-xCS-vCS < 0.) return false;

  // splitType ==-2 -> Massive II
  } else if (splitType == -2 && aux.size() == 0) {

    // Calculate CS variables.

    double q2 = m2dip + m2s + m2RadBef;
    double m2DipCorr = m2dip - m2RadBef + m2r + m2e;
    double kappa2    = pT2 / m2DipCorr;
    double xCS       = (z*(1-z)- kappa2)/(1-z);
    double vCS       = kappa2/(1-z);

    // Calculate derived variables.
    double sab  = (q2 - m2e)/xCS + (m2r+m2s) * (1-1/xCS);
    double saj  = -vCS*(sab - m2r-m2s) + m2r + m2e;
    double zbar = (sab - m2r - m2s) / bABC(sab,m2r,m2s)
                *( (xCS + vCS)  - m2s / gABC(sab,m2r,m2s)
                       * (saj + m2r - m2e) / (sab - m2r - m2s));
    double kT2  = zbar*(1.-zbar)*m2r - (1-zbar)*saj - zbar*m2e;

    // Disallow kinematically impossible transverse momentum.
    if (kT2 < 0. || isnan(kT2)) return false;

  // splitType ==-2 -> Massive 1->3 II
  } else {

    // Massive II only possible for 1->3 splitting, thus more complicated.

    // Not correctly set up!
    if ( int(aux.size()) < 11) return false;

    // Retrieve necessary variables for 1->3 splitting.
    double q2_1   = aux[1];
    double t      = aux[2];
    double sai    = aux[3];
    double za     = aux[4];
    double xa     = aux[5];
    double m2a    = aux[7];
    double m2i    = aux[8];
    double m2j    = aux[9];
    double m2k    = aux[10];
    double m2ai  = -sai + m2a + m2i;

    if (za < xIncoming || za > 1.) return false;

    // Check "first" step.
    double p2ab = q2_1/za + m2a + m2k;
    double zbar = (p2ab - m2a - m2k) / bABC(p2ab,m2a,m2k)
                *( xa - m2k / gABC(p2ab,m2a,m2k)
                       * (m2ai + m2a - m2i) / (p2ab - m2a - m2k));
    double kT2  = zbar*(1.-zbar)*m2a - (1-zbar)*m2ai - zbar*m2i;

    // Disallow kinematically impossible transverse momentum.
    if (kT2 < 0. || isnan(kT2)) return false;

    // Check "second" step.
    double m2rec = m2ai;
    double m2emt = q2_1;
    double m2rad = m2j;
    double zCS  = t/xa / (q2_1*xa/za + 2.*m2ai);
    double yCS  = 1. / ( 1. + (q2_1*xa/za + 2.*m2ai)
                            / (q2_1*(xa/za - 1.) + m2ai + m2k - m2j));
    double q2_2 = 4.*m2ai + 2.*q2_1*xa/za + m2k;

    // Not possible to find sensible final-final variables.
    if (yCS < 0. || yCS > 1. || zCS < 0. || zCS > 1.) return false;

    // Calculate derived variables.
    double sij  = yCS * (q2_2 - m2rec) + (1.-yCS)*(m2rad+m2emt);
    zbar        = (q2_2-sij-m2rec) / bABC(q2_2,sij,m2rec)
                * (zCS - m2rec/gABC(q2_2,sij,m2rec)
                       *(sij + m2rad - m2emt)/(q2_2-sij-m2rec));
    kT2         = zbar*(1.-zbar)*sij - (1.-zbar)*m2rad - zbar*m2emt;

    // Not possible to construct kinematics if kT2 < 0.0
    if (kT2 < 0. || isnan(kT2)) return false;

  }

  return true;

}

//--------------------------------------------------------------------------

// Function to add user-defined overestimates to old overestimate.

void DireSpace::addNewOverestimates( multimap<double,string> newOverestimates,
  double& oldOverestimate ) {

  // No other tricks necessary at the moment.
  if (!newOverestimates.empty())
    oldOverestimate += newOverestimates.rbegin()->first;

  // Done.

}

//--------------------------------------------------------------------------

// Function to attach the correct alphaS weights to the kernels.

void DireSpace::alphasReweight(double, double talpha, int iSys,
  bool forceFixedAs, double& weight, double& fullWeight, double& overWeight,
  double renormMultFacNow) {

  if (forceFixedAs) {
    overWeight *= alphaS2piOverestimate;
    weight *= alphasNow(pT2min, 1., iSys) / alphaS2piOverestimate;
    fullWeight *= alphasNow(pT2min, 1., iSys);
    return;
  }
  talpha = max(talpha, pT2min);

  double scale       = talpha*renormMultFacNow;
  scale              = max(scale, pT2min);

  // Get current alphaS value.
  double asPT2piCorr  = alphasNow(talpha, renormMultFacNow, iSys);

  // Begin with multiplying alphaS to overestimate.
  double asOver = 1.;
  if (usePDFalphas)        asOver = alphaS2piOverestimate;
  else if (alphaSorder==0) asOver = alphaS2pi;
  else                     asOver = asPT2piCorr;
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

// Evolve a QCD dipole end.

void DireSpace::pT2nextQCD( double pT2begDip, double pT2endDip,
  DireSpaceEnd& dip, Event& event, double pT2endForce, double pT2freeze,
  bool forceBranching) {

  if (event[dip.iRecoiler].isFinal()) {
    pT2nextQCD_IF(pT2begDip, pT2endDip, dip, event, pT2endForce, pT2freeze,
      forceBranching);
  } else {
    pT2nextQCD_II(pT2begDip, pT2endDip, dip, event, pT2endForce, pT2freeze,
      forceBranching);
  }

  // Done
}

//--------------------------------------------------------------------------

// Evolve a QCD dipole end.

bool DireSpace::pT2nextQCD_II( double pT2begDip, double pT2sel,
  DireSpaceEnd& dip, Event& event, double pT2endForce, double pT2freeze,
  bool forceBranching) {

  // Lower cut for evolution. Return if no evolution range.
  double pT2endDip = max( pT2sel, pT2cutMin(&dip));
  if (pT2endForce >= 0.) pT2endDip = pT2endForce;
  if (pT2begDip < pT2endDip) return false;

  // Reset dipole mass.
  int iRadi  = dip.iRadiator;
  int iReco  = dip.iRecoiler;
  m2Dip      = abs(2.*event[iRadi].p()*event[iReco].p());
  dip.m2Dip  = m2Dip;

  // Some properties and kinematical starting values.
  BeamParticle& beam = (sideA) ? *beamAPtr : *beamBPtr;
  double tnow        = pT2begDip;
  double xMaxAbs     = beam.xMax(iSysNow);
  double zMinAbs     = xDaughter;

  if (usePDF && xMaxAbs < 0.) {
    infoPtr->errorMsg("Warning in DireSpace::pT2nextQCD_II: "
    "xMaxAbs negative");
    return false;
  }

  // Variables used inside evolution loop. (Mainly dummy starting values.)
  int    nFlavour       = 3;
  double Lambda2        = Lambda3flav2;
  int    idMother       = 0;
  int    idSister       = 0;
  double znow           = 0.;
  double zMaxAbs        = 0.;
  double xPDFdaughter   = 0.;
  double kernelPDF      = 0.;
  double xMother        = 0.;
  double wt             = 0.;
  double mSister        = 0.;
  double m2Sister       = 0.;
  double pT2corr        = 0.;
  double teval          = pT2begDip;
  bool   needNewPDF     = true;
  bool hasPDFdau        = hasPDF(idDaughter);
  if (!hasPDFdau) zMinAbs = 0.;

  multimap<double,string> newOverestimates;
  unordered_map<string,double> fullWeightsNow;
  double fullWeightNow(0.), overWeightNow(0.), auxWeightNow(0.), daux(0.);

  // Begin evolution loop towards smaller pT values.
  int    loopTinyPDFdau = 0;
  int    nContinue(0), nContinueMax(10000);
  bool   hasTinyPDFdau  = false;
  do {

    wt        = 0.;
    //tnow = (!forceBranching) ? dip.pT2 : pT2begDip;
    tnow = (!forceBranching) ? tnow : pT2begDip;
    znow      = -1.;
    dip.phi   = -1.;
    dip.phia1 = -1.;

    // Force exit if non-Sudakov style forced branching is stuck.
    if (forceBranching && nContinue >= nContinueMax) {
      wt = 0.0; dip.pT2 = tnow = 0.;
      break;
    }

    // Update event weight after one step.
    if ( fullWeightNow != 0. && overWeightNow != 0. ) {
      double enhanceFurther
         = enhanceOverestimateFurther(splittingNowName, idDaughter, teval);
      if (doTrialNow) enhanceFurther = 1.;
      kernelNow = fullWeightsNow;
      auxNow = auxWeightNow;
      overNow = overWeightNow;
      boostNow = enhanceFurther;

      for ( unordered_map<string,double>::iterator it = fullWeightsNow.begin();
        it != fullWeightsNow.end(); ++it ) {

        // No weight bookkeeping for non-variation components of kernel vector.
        if (it->first == "base_order_as2") continue;

        double wv = auxWeightNow/overWeightNow
                 * (overWeightNow- it->second/enhanceFurther)
                 / (auxWeightNow - fullWeightNow);
        if (abs(wv) > 1e0) {
        direInfoPtr->message(1)
        << scientific << setprecision(15)
        << __FILE__ << " " << __func__
        << " " << __LINE__ << " : Large reject weight=" << wv
        << "\t for kernel=" << it->second
        << " " << fullWeightNow << " overestimate=" << overNow
        << "\t aux. overestimate=" << auxNow << " at pT2="
        << tnow
        <<  " for " << splittingNowName << endl;
        }
        rejectProbability[it->first].insert( make_pair(tnow,wv));
      }
    }

    splittingNowName="";
    fullWeightsNow.clear();
    fullWeightNow = overWeightNow = auxWeightNow = 0.;

    // Leave unconverted for now.
    if (abs(idDaughter)==4 && tnow <= m2cPhys) { dip.pT2 = 0.0; return false;}
    if (abs(idDaughter)==5 && tnow <= m2bPhys) { dip.pT2 = 0.0; return false;}

    // Finish evolution if PDF vanishes.
    double tnew = (useFixedFacScale) ? fixedFacScale2 : factorMultFac*tnow;
    tnew        = max(tnew, pT2min);
    bool inNew  = (hasPDFdau) ? beam.insideBounds(xDaughter, tnew) : true;
    if (hasPDFdau && !inNew) { dip.pT2 = 0.0; return false; }

    // Bad sign if repeated looping with small daughter PDF, so fail.
    // (Example: if all PDF's = 0 below Q_0, except for c/b companion.)
    if (hasTinyPDFdau) ++loopTinyPDFdau;
    if (hasPDFdau && loopTinyPDFdau > MAXLOOPTINYPDF) {
      infoPtr->errorMsg("Warning in DireSpace::pT2nextQCD: "
      "small daughter PDF");
      dip.pT2 = 0.0;
      return false;
    }

    // Initialize integrals of splitting kernels and evaluate parton
    // densities at the beginning. Reinitialize after long evolution
    // in pT2 or when crossing c and b flavour thresholds.
    if (needNewPDF
      || tnow < evalpdfstep(event[iRadi].id(), tnow, m2cPhys, m2bPhys)*teval) {

      teval         = tnow;
      hasTinyPDFdau = false;

      newOverestimates.clear();
      kernelPDF = 0.;

      // Determine overestimated z range; switch at c and b masses.
      if (tnow > m2b) {
        nFlavour  = 5;
        Lambda2   = Lambda5flav2;
      } else if (tnow > m2c) {
        nFlavour  = 4;
        Lambda2   = Lambda4flav2;
      } else {
        nFlavour  = 3;
        Lambda2   = Lambda3flav2;
      }

      // A change of renormalization scale expressed by a change of Lambda.
      Lambda2    /= renormMultFac;
      zMinAbs     = (hasPDFdau) ? xDaughter : 0.;
      zMaxAbs     = 1.;

      // Parton density of daughter at current scale.
      pdfScale2    = (useFixedFacScale) ? fixedFacScale2 : factorMultFac*tnow;
      pdfScale2    = max(pdfScale2, pT2min);
      xPDFdaughter = getXPDF(idDaughter, xDaughter, pdfScale2, iSysNow, &beam);
      if ( hasPDFdau && xPDFdaughter != 0.
        && abs(xPDFdaughter) < tinypdf(xDaughter)) {
        int sign      = (xPDFdaughter > 0.) ? 1 : -1;
        xPDFdaughter  = sign*tinypdf(xDaughter);
        hasTinyPDFdau = true;
      }

      // Calculate and add user-defined overestimates.
      getNewOverestimates( idDaughter, &dip, event, teval,
        xDaughter, zMinAbs, zMaxAbs, newOverestimates);
      addNewOverestimates(newOverestimates, kernelPDF);

      // Store pT at which overestimate has been evaluated.
      dip.pT2Old = teval;

      // End evaluation of splitting kernels and parton densities.
      needNewPDF = false;
    }

    if (kernelPDF < TINYKERNELPDF) { dip.pT2 = 0.0; return false; }
    if (newOverestimates.empty())  { dip.pT2 = 0.0; return false; }

    // Generate next evolution scale.
    bool forceFixedAs = (tnow < pT2min);
    tnow = tNextQCD( &dip, kernelPDF, tnow, pT2endDip, pT2freeze,
      (forceBranching ? -1 : 1));
    if (tnow < 0.) {
      wt = dip.pT2 = tnow = 0.;
      double R0 = kernelPDF*rndmPtr->flat();
      if (!newOverestimates.empty()) {
        if (newOverestimates.lower_bound(R0) == newOverestimates.end())
          splittingNowName = newOverestimates.rbegin()->second;
        else
          splittingNowName = newOverestimates.lower_bound(R0)->second;
      }
      break;
    }

    // Abort evolution if below cutoff scale, or below another branching.
    if ( tnow <= pT2endDip) { dip.pT2 = tnow = 0.; break; }

    // Check for pT2 values that prompt special action.
    // If crossed b threshold, continue evolution from this threshold.
    if (nFlavour == 5 && tnow <= m2bPhys) {
      needNewPDF = true;
    // If crossed c threshold, continue evolution from this threshold.
    } else if (nFlavour == 4 && tnow <= m2cPhys) {
      needNewPDF = true;
    }

    // Leave heavy quarks below threshold unconverted for now.
    if (abs(idDaughter) == 5 && tnow <= m2bPhys) {
      dip.pT2 = tnow = 0.; break;
    } else if (abs(idDaughter) == 4 && tnow <= m2cPhys) {
      dip.pT2 = tnow = 0.; break;
    }

    double xMin = xDaughter;
    if (!hasPDFdau) xMin = 0.;

    // Select z value of branching to g, and corrective weight.
    // User-defined splittings.
    double R = kernelPDF*rndmPtr->flat();
    if (!newOverestimates.empty()) {
      if (newOverestimates.lower_bound(R) == newOverestimates.end())
        splittingNowName = newOverestimates.rbegin()->second;
      else
        splittingNowName = newOverestimates.lower_bound(R)->second;
      getNewSplitting( event, &dip, teval, xMin, tnow, zMinAbs,
        zMaxAbs, idDaughter, splittingNowName, forceFixedAs, idMother,
        idSister, znow, wt, fullWeightsNow, overWeightNow);
    }

    // Impossible emission (e.g. if outside allowed z range for given pT2).
    if ( wt == 0.) {
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    fullWeightNow = fullWeightsNow["base"];

    // Disallow gluon -> heavy quark, heavy quark --> light quark and
    // light quark -> heavy quark if pT has fallen below 2*mQuark.
    if ( tnow <= 4.*m2bPhys
      && ( (abs(idDaughter) == 21 && abs(idSister) == 5)
      || (abs(idDaughter) == 5 && splits[splittingNowName]->nEmissions()==2)
      || (abs(idSister) == 5 && splits[splittingNowName]->nEmissions()==2))) {
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    } else if ( tnow <= 4.*m2cPhys
      && ( (abs(idDaughter) == 21 && abs(idSister) == 4)
      || (abs(idDaughter) == 4 && splits[splittingNowName]->nEmissions()==2)
      || (abs(idSister) == 4 && splits[splittingNowName]->nEmissions()==2))) {
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Get particle masses.
    double m2Bef = 0.0;
    double m2r   = 0.0;
    // Force emission massless for now.
    double m2e = 0.0;
    double m2s = 0.;
    double q2  = (event[iRadi].p()+event[iReco].p()).m2Calc();

    // Discard this 1->3 splitting if the pT has fallen below mEmission (since
    // such splittings would not be included in the virtual corrections to the
    // 1->2 kernels. Note that the threshold is pT>mEmission,since alphaS is
    // evaluated at pT, not virtuality sa1).
    if ( splits[splittingNowName]->nEmissions() == 2 )
      if ( (abs(idSister) == 4 && tnow < m2cPhys)
        || (abs(idSister) == 5 && tnow < m2bPhys)) {
      needNewPDF = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Pick remaining variables for 1->3 splitting.
    double m2a(m2r), m2i(getMass(idMother,2)),
    m2j(getMass(-event[iRadi].id(),2)), m2aij(m2Bef), m2k(m2s);

    // Adjust the dipole kinematical mass to accomodate masses after branching.
    double m2DipCorr = m2Dip - m2Bef + m2r + m2e;
    // Calculate CS variables.
    double kappa2    = tnow / m2DipCorr;
    double xCS       = (znow*(1-znow)- kappa2)/(1-znow);

    // Jacobian for 1->3 splittings, in CS variables.
    double jacobian(1.);

    bool canUseSplitInfo = splits[splittingNowName]->canUseForBranching();
    if (canUseSplitInfo) {
      jacobian
        = splits[splittingNowName]->getJacobian(event,partonSystemsPtr);
      unordered_map<string,double> psvars
        = splits[splittingNowName]->
        getPhasespaceVars( event, partonSystemsPtr);
      xMother = psvars["xInAft"];
    } else {
      if ( splits[splittingNowName]->nEmissions() == 2 ) {
        double za    = dip.z;
        double xa    = dip.xa;
        xCS          =  za * (q2 - m2a - m2i - m2j - m2k) / q2;

        // Calculate Jacobian.
        double sab = q2/za + m2a + m2k;
        jacobian = (sab-m2a-m2k) / sqrt(lABC(sab, m2a, m2k) );
        double sai   = dip.sa1;
        double m2ai  = -sai + m2a + m2i;
        double sjq   = q2*xa/za + m2ai + m2k;
        jacobian *= (sjq-m2ai-m2k) / sqrt(lABC(sjq, m2ai, m2k) );
        // Additional factor from massive propagator.
        jacobian *= 1. / (1. - (m2ai + m2j - m2aij) / (dip.pT2/dip.xa)) ;
      }
      xMother = xDaughter/xCS;
    }

    // Evaluation of new daughter and mother PDF's.
    pdfScale2 = (useFixedFacScale) ? fixedFacScale2 : factorMultFac * tnow;
    pdfScale2 = max(pdfScale2, tnow);
    double pdfScale2Old = pdfScale2;
    double pdfScale2New = pdfScale2;
    if (forceBranching) pdfScale2Old = pdfScale2New = infoPtr->Q2Fac();
    bool inD = (hasPDFdau) ? beam.insideBounds(xDaughter, pdfScale2Old) : true;
    bool inM = (hasPDFdau) ? beam.insideBounds(xMother,   pdfScale2New) : true;
    double xPDFdaughterNew = getXPDF( idDaughter, xDaughter, pdfScale2Old,
      iSysNow, &beam, false, znow, m2Dip);
    double xPDFmotherNew   = getXPDF( idMother, xMother, pdfScale2New,
      iSysNow, &beam, false, znow, m2Dip);
    if ( hasPDFdau && xPDFdaughterNew != 0.
      && abs(xPDFdaughterNew) < tinypdf(xDaughter) ) {
      hasTinyPDFdau = true;
      needNewPDF    = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Extreme case of low-scale gluon, for which denominator rapidly
    // approaches zero. In this case, cut off branching probability if
    // daughter PDF fell too rapidly, to avoid large shower
    // weights. (Note: Last resort - would like something more
    // physical here!)
    double xPDFdaughterLow = getXPDF( idDaughter, xDaughter,
      pdfScale2Old*pdfScale2Old/max(teval,pT2min), iSysNow, &beam);
    if ( hasPDFdau && idDaughter == 21
      && ( abs(xPDFdaughterNew/xPDFdaughter) < 1e-4
        || abs(xPDFdaughterLow/xPDFdaughterNew) < 1e-4) ) {
      hasTinyPDFdau = true;
      needNewPDF    = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Set PDF ratio to zero if x-/pT-values are out-of-bounds.
    double pdfRatio = (inD && inM) ? xPDFmotherNew/xPDFdaughterNew : 0.;

    // More last resort.
    if (hasPDFdau && idDaughter == 21 && pdfScale2 == pT2min && pdfRatio>50.)
      pdfRatio = 0.;

    fullWeightNow  *= pdfRatio*jacobian;

    for ( unordered_map<string,double>::iterator it = fullWeightsNow.begin();
      it != fullWeightsNow.end(); ++it )
      it->second   *= pdfRatio*jacobian;

    // Before generating kinematics: Reset sai if the kernel fell on an
    // endpoint contribution.
    if ( splits[splittingNowName]->nEmissions() == 2 )
      dip.sa1 = splits[splittingNowName]->splitInfo.kinematics()->sai;

    if ( fullWeightNow == 0. ) {
      needNewPDF = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Retrieve argument of alphaS.
    double scale2 =  splits[splittingNowName]->couplingScale2 ( dip.z, tnow,
      m2DipCorr,
      make_pair (event[dip.iRadiator].id(), event[dip.iRadiator].isFinal()),
      make_pair (event[dip.iRecoiler].id(), event[dip.iRecoiler].isFinal()));
    if (scale2 < 0.) scale2 = tnow;
    double talpha = max(scale2, pT2min);

    // Reweight to match PDF alpha_s, including corrective terms for
    // renormalisation scale variation.
    alphasReweight(tnow, talpha, dip.system, forceFixedAs, wt, fullWeightNow,
      overWeightNow, renormMultFac);

    // Create muR-variations.
    double asw = 1.;
    alphasReweight(tnow, talpha, dip.system, forceFixedAs, daux, asw, daux,
      renormMultFac);
    fullWeightsNow["base"] *= asw;
    if (fullWeightsNow.find("base_order_as2") != fullWeightsNow.end())
      fullWeightsNow["base_order_as2"] *= asw;
    if (doVariations) {
      if ( splittingNowName.find("qcd") != string::npos
        && settingsPtr->parm("Variations:muRisrDown") != 1.) {
        asw = 1.;
        alphasReweight(tnow, talpha, dip.system, forceFixedAs, daux, asw, daux,
          (tnow > pT2minVariations) ? settingsPtr->parm
                       ("Variations:muRisrDown")*renormMultFac
          : renormMultFac);
        fullWeightsNow["Variations:muRisrDown"] *= asw;
      } else if ( splittingNowName.find("qcd") == string::npos )
        fullWeightsNow["Variations:muRisrDown"] *= asw;
      if ( splittingNowName.find("qcd") != string::npos
        && settingsPtr->parm("Variations:muRisrUp")   != 1.) {
        asw = 1.;
        alphasReweight(tnow, talpha, dip.system, forceFixedAs, daux, asw, daux,
          (tnow > pT2minVariations) ? settingsPtr->parm
                       ("Variations:muRisrUp")*renormMultFac
          : renormMultFac);
        fullWeightsNow["Variations:muRisrUp"] *= asw;
      } else if ( splittingNowName.find("qcd") == string::npos )
        fullWeightsNow["Variations:muRisrUp"] *= asw;

      // PDF variations.
      if (hasPDFdau && settingsPtr->flag("Variations:PDFup") ) {
        int valSea = (beam[iSysNow].isValence()) ? 1 : 0;
        if( beam[iSysNow].isUnmatched() ) valSea = 2;
        beam.calcPDFEnvelope( make_pair(idMother, idDaughter),
          make_pair(xMother, xDaughter), pdfScale2, valSea);
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

    // Set auxiliary weight and ensure that accept probability is positive.
    auxWeightNow = overWeightNow;

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
        << " and z=" << znow << "\t(PDF ratio=" << pdfRatio << ")" << endl;
      double rescale = fullWeightNow/auxWeightNow * 1.15;
      auxWeightNow *= rescale;
      infoPtr->errorMsg("Info in DireSpace::pT2nextQCD_II: Found large "
                        "acceptance weight for " + splittingNowName);
    }

    wt = fullWeightNow/auxWeightNow;

  // Iterate until acceptable pT (or have fallen below pTmin).
  } while (wt < rndmPtr->flat()) ;

  // Not possible to find splitting.
  if ( wt == 0.) { dip.pT2 = 0.0; return false; }

  // Update accepted event weight. No weighted shower for first
  // "pseudo-emission" step in 1->3 splitting.
  if ( fullWeightNow != 0. && overWeightNow != 0. ) {
    double enhanceFurther
       = enhanceOverestimateFurther(splittingNowName, idDaughter, teval);
    if (doTrialNow) {
      weights->addTrialEnhancement(tnow, enhanceFurther);
      enhanceFurther = 1.;
    }
    kernelNow = fullWeightsNow;
    auxNow = auxWeightNow;
    overNow = overWeightNow;
    boostNow = enhanceFurther;
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

  double zStore   = dip.z;
  double xaStore  = dip.xa;
  double pT2Store = dip.pT2;
  double sa1Store = dip.sa1;
  double Q2store  = 0.;

  // Save values for (so far) acceptable branching.
  dipEndNow->store( idDaughter,idMother, idSister,
    x1Now, x2Now, m2Dip, pT2Store, zStore, sa1Store, xaStore, xMother,
    Q2store, mSister, m2Sister, pT2corr, dip.phi, dip.phia1);

  // Done
  return true;

}

//--------------------------------------------------------------------------

// Evolve a QCD dipole end.

bool DireSpace::pT2nextQCD_IF( double pT2begDip, double pT2sel,
  DireSpaceEnd& dip, Event& event, double pT2endForce, double pT2freeze,
  bool forceBranching) {

  // Lower cut for evolution. Return if no evolution range.
  double pT2endDip = max( pT2sel, pT2cutMin(&dip));
  if (pT2endForce >= 0.) pT2endDip = pT2endForce;
  if (pT2begDip < pT2endDip) return false;

  // Reset dipole mass.
  int iRadi  = dip.iRadiator;
  int iReco  = dip.iRecoiler;
  m2Dip      = abs(2.*event[iRadi].p()*event[iReco].p());
  dip.m2Dip  = m2Dip;

  // Some properties and kinematical starting values.
  BeamParticle& beam = (sideA && particleDataPtr->isHadron(beamAPtr->id()))
                     ? *beamAPtr
                     : (particleDataPtr->isHadron(beamBPtr->id()) ? *beamBPtr
                                                      : *beamAPtr );
  double tnow        = pT2begDip;
  double xMaxAbs     = beam.xMax(iSysNow);
  double zMinAbs     = xDaughter;

  // Get momentum of other beam, since this might be needed to calculate
  // the Jacobian.
  int iOther = sideA ? getInB(iSysNow) : getInA(iSysNow);
  Vec4 pOther(event[iOther].p());

  if (usePDF && xMaxAbs < 0.) {
    infoPtr->errorMsg("Warning in DireSpace::pT2nextQCD_IF: "
    "xMaxAbs negative");
    return false;
  }

  // Variables used inside evolution loop. (Mainly dummy starting values.)
  int    nFlavour       = 3;
  double Lambda2        = Lambda3flav2;
  int    idMother       = 0;
  int    idSister       = 0;
  double znow           = 0.;
  double zMaxAbs        = 0.;
  double xPDFdaughter   = 0.;
  double kernelPDF      = 0.;
  double xMother        = 0.;
  double wt             = 0.;
  double mSister        = 0.;
  double m2Sister       = 0.;
  double pT2corr        = 0.;
  double teval          = pT2begDip;
  bool   needNewPDF     = true;
  bool   hasPDFdau      = hasPDF(idDaughter);
  if (!hasPDFdau) zMinAbs = 0.;

  multimap<double,string> newOverestimates;
  unordered_map<string,double> fullWeightsNow;
  double fullWeightNow(0.), overWeightNow(0.), auxWeightNow(0.), daux(0.);

  // Begin evolution loop towards smaller pT values.
  int    loopTinyPDFdau = 0;
  int    nContinue(0), nContinueMax(10000);
  bool   hasTinyPDFdau  = false;
  do {

    wt   = 0.;
    znow = -1.;
    dip.phi   = -1.0;
    dip.phia1 = -1.0;
    tnow = (!forceBranching) ? tnow : pT2begDip;

    // Force exit if non-Sudakov style forced branching is stuck.
    if (forceBranching && nContinue >= nContinueMax) {
      wt = 0.0; dip.pT2 = tnow = 0.;
      break;
    }

    // Update event weight after one step.
    if ( fullWeightNow != 0. && overWeightNow != 0. ) {
      double enhanceFurther
        = enhanceOverestimateFurther(splittingNowName, idDaughter, teval);
      if (doTrialNow) enhanceFurther = 1.;
      kernelNow = fullWeightsNow;
      auxNow = auxWeightNow;
      overNow = overWeightNow;
      boostNow = enhanceFurther;
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

    splittingNowName="";
    fullWeightsNow.clear();
    fullWeightNow = overWeightNow = auxWeightNow = 0.;

    // Leave unconverted for now.
    if (abs(idDaughter)==4 && tnow <= m2cPhys) { dip.pT2 = 0.0; return false;}
    if (abs(idDaughter)==5 && tnow <= m2bPhys) { dip.pT2 = 0.0; return false;}

    // Finish evolution if PDF vanishes.
    double tnew = (useFixedFacScale) ? fixedFacScale2 : factorMultFac*tnow;
    tnew        = max(tnew, pT2min);
    bool inNew  = (hasPDFdau) ? beam.insideBounds(xDaughter, tnew) : true;
    if (!inNew && hasPDFdau) { dip.pT2 = 0.0; return false; }

    // Bad sign if repeated looping with small daughter PDF, so fail.
    // (Example: if all PDF's = 0 below Q_0, except for c/b companion.)
    if (hasTinyPDFdau) ++loopTinyPDFdau;
    if ( hasPDFdau && loopTinyPDFdau > MAXLOOPTINYPDF) {
      infoPtr->errorMsg("Warning in DireSpace::pT2nextQCD_IF: "
      "small daughter PDF");
      dip.pT2 = 0.0;
      return false;
    }

    // Initialize integrals of splitting kernels and evaluate parton
    // densities at the beginning. Reinitialize after long evolution
    // in pT2 or when crossing c and b flavour thresholds.
    if ( needNewPDF
      || tnow < evalpdfstep(event[iRadi].id(), tnow, m2cPhys, m2bPhys)*teval) {
      teval         = tnow;
      hasTinyPDFdau = false;

      newOverestimates.clear();
      kernelPDF = 0.;

      // Determine overestimated z range; switch at c and b masses.
      if (tnow > m2b) {
        nFlavour  = 5;
        Lambda2   = Lambda5flav2;
      } else if (tnow > m2c) {
        nFlavour  = 4;
        Lambda2   = Lambda4flav2;
      } else {
        nFlavour  = 3;
        Lambda2   = Lambda3flav2;
      }

      // A change of renormalization scale expressed by a change of Lambda.
      Lambda2    /= renormMultFac;
      zMinAbs     = (hasPDFdau) ? xDaughter : 0.;
      zMaxAbs     = 1.;

      // Parton density of daughter at current scale.
      pdfScale2    = (useFixedFacScale) ? fixedFacScale2 : factorMultFac*tnow;
      pdfScale2    = max(pdfScale2, pT2min);
      xPDFdaughter = getXPDF(idDaughter, xDaughter, pdfScale2, iSysNow, &beam);
      if ( hasPDFdau && xPDFdaughter != 0.
        && abs(xPDFdaughter) < tinypdf(xDaughter)) {
        int sign      = (xPDFdaughter > 0.) ? 1 : -1;
        xPDFdaughter  = sign*tinypdf(xDaughter);
        hasTinyPDFdau = true;
      }

      // Calculate and add user-defined overestimates.
      getNewOverestimates( idDaughter, &dip, event, teval,
        xDaughter, zMinAbs, zMaxAbs, newOverestimates );
      addNewOverestimates(newOverestimates, kernelPDF);

      // End evaluation of splitting kernels and parton densities.
      needNewPDF = false;

      // Store pT at which overestimate has been evaluated.
      dip.pT2Old = teval;

    }

    if (kernelPDF < TINYKERNELPDF) { dip.pT2 = 0.0; return false; }
    if (newOverestimates.empty()) return false;

    // Generate next evolution scale.
    bool forceFixedAs = (tnow < pT2min);
    tnow  = tNextQCD( &dip, kernelPDF, tnow, pT2endDip, pT2freeze,
      (forceBranching ? -1 : 1));
    if (tnow < 0.) {
      wt = dip.pT2 = tnow = 0.;
      double R0 = kernelPDF*rndmPtr->flat();
      if (!newOverestimates.empty()) {
        if (newOverestimates.lower_bound(R0) == newOverestimates.end())
          splittingNowName = newOverestimates.rbegin()->second;
        else
          splittingNowName = newOverestimates.lower_bound(R0)->second;
      }
      break;
    }

    // Abort evolution if below cutoff scale, or below another branching.
    if (tnow <= pT2endDip) { dip.pT2 = tnow = 0.; break; }

    // Check for pT2 values that prompt special action.
    // If crossed b threshold, continue evolution from this threshold.
    if (nFlavour == 5 && tnow <= m2bPhys) {
      needNewPDF = true;
    // If crossed c threshold, continue evolution from this threshold.
    } else if (nFlavour == 4 && tnow <= m2cPhys) {
      needNewPDF = true;
    }

    // Leave heavy quarks below threshold unconverted for now.
    if        (abs(idDaughter) == 5 && tnow <= m2bPhys) {
      dip.pT2 = tnow = 0.; break;
    } else if (abs(idDaughter) == 4 && tnow <= m2cPhys) {
      dip.pT2 = tnow = 0.; break;
    }

    double xMin = xDaughter;
    if (!hasPDFdau) xMin = 0.;

    // Select z value of branching, and corrective weight.
    double R = kernelPDF*rndmPtr->flat();
    if (!newOverestimates.empty()) {
      if (newOverestimates.lower_bound(R) == newOverestimates.end())
        splittingNowName = newOverestimates.rbegin()->second;
      else
        splittingNowName = newOverestimates.lower_bound(R)->second;
      getNewSplitting( event, &dip, teval, xMin, tnow, zMinAbs,
        zMaxAbs, idDaughter, splittingNowName, forceFixedAs, idMother,
        idSister, znow, wt, fullWeightsNow, overWeightNow);

    }

    // Impossible emission (e.g. if outside allowed z range for given pT2).
    if ( wt == 0.) {
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    fullWeightNow = fullWeightsNow["base"];

    // Disallow gluon -> heavy quark, heavy quark --> light quark and
    // light quark -> heavy quark if pT has fallen below 2*mQuark.
    if ( tnow <= 4.*m2bPhys
      && ( (abs(idDaughter) == 21 && abs(idSister) == 5)
      || (abs(idDaughter) == 5 && splits[splittingNowName]->nEmissions()==2)
      || (abs(idSister) == 5 && splits[splittingNowName]->nEmissions()==2))) {
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    } else if ( tnow <= 4.*m2cPhys
      && ( (abs(idDaughter) == 21 && abs(idSister) == 4)
      || (abs(idDaughter) == 4 && splits[splittingNowName]->nEmissions()==2)
      || (abs(idSister) == 4 && splits[splittingNowName]->nEmissions()==2))) {
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Get particle masses.
    double m2Bef = 0.0;
    double m2r   = 0.0;
    // Emission mass (here only used to calculate argument of alphas)
    double m2e = (abs(idSister)<6) ? getMass(idSister,2) : getMass(idSister,1);
    if (!forceMassiveMap) m2e = 0.0;
    double m2s = particleDataPtr->isResonance(event[dip.iRecoiler].id())
          ? getMass(event[dip.iRecoiler].id(),3,
                    event[dip.iRecoiler].mCalc())
          : (event[dip.iRecoiler].idAbs() < 6)
          ? getMass(event[dip.iRecoiler].id(),2)
          : getMass(event[dip.iRecoiler].id(),1);

    // Discard this 1->3 splitting if the pT has fallen below mEmission (since
    // such splittings would not be included in the virtual corrections to the
    // 1->2 kernels. Note that the threshold is pT>mEmission,since alphaS is
    // evaluated at pT, not virtuality sa1).
    if ( splits[splittingNowName]->nEmissions() == 2 )
      if ( (abs(idSister) == 4 && tnow < m2cPhys)
        || (abs(idSister) == 5 && tnow < m2bPhys)) {
      needNewPDF = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Pick remaining variables for 1->3 splitting.
    double jacobian(1.), m2aij(m2Bef), m2ai(0.), m2a(m2r),
      m2i(getMass(idMother,2)),
    m2j(getMass(-event[iRadi].id(),2)), m2k(m2s);
    m2ai  = -dip.sa1 + m2a + m2i;
    double q2 = (event[iRadi].p()-event[iReco].p()).m2Calc();

    bool canUseSplitInfo = splits[splittingNowName]->canUseForBranching();
    if (canUseSplitInfo) {
      jacobian
        = splits[splittingNowName]->getJacobian(event,partonSystemsPtr);
      unordered_map<string,double> psvars = splits[splittingNowName]->
        getPhasespaceVars( event, partonSystemsPtr);
      xMother = psvars["xInAft"];
    } else {

      // Jacobian for 1->3 splittings, in CS variables.
      if ( splits[splittingNowName]->nEmissions() == 2 ) {
        double m2jk = dip.pT2/dip.xa + q2*( 1. - dip.xa/dip.z) - m2ai;

        // Construnct the new initial state momentum, as needed to
        // calculate the Jacobian.
        double uCS  = dip.z*(m2ai-m2a-m2i)/q2;
        double xCS  = uCS + dip.xa - (dip.pT2*dip.z)/(q2*dip.xa);
        Vec4 q( event[iRadi].p() - event[iReco].p() );
        double sHatBef = (event[iRadi].p() + pOther).m2Calc();
        double sijk    = q2*(1.-1./dip.z) - m2a;

        // sHat after emission depends on the recoil scheme if the incoming
        // particles have non-zero mass.
        // Local scheme.
        double sHatAft(0.);
        if (!useGlobalMapIF) {

          // Get transverse and parallel vectors.
          Vec4 pTk_tilde(event[iReco].p().px(), event[iReco].p().py(), 0., 0.);
          Vec4 qpar( q + pTk_tilde );
          // Calculate derived variables.
          double q2par  = qpar.m2Calc();
          double pT2k   = -pTk_tilde.m2Calc();
          double s_i_jk = (1. - 1./xCS)*(q2 - m2a) + (m2i + m2jk) / xCS;
          // Construct radiator after branching.
          Vec4 pa( ( event[iRadi].p() - 0.5*(q2-m2aij-m2k)/q2par * qpar )
                     * sqrt( (lABC(q2,s_i_jk,m2a) - 4.*m2a*pT2k)
                           / (lABC(q2,m2k,m2aij) - 4.*m2aij*pT2k))
                    + qpar * 0.5 * (q2 + m2a - s_i_jk) / q2par);
          // Now get changed eCM.
          sHatAft = (pa + pOther).m2Calc();

        // Global scheme.
        } else {

          // Construct radiator after branching.
          // Simple massless case.
          Vec4 pa;

          // Get dipole 4-momentum.
          Vec4 pb_tilde(   event[iReco].p() );
          Vec4 pa12_tilde( event[iRadi].p() );
          q.p(pb_tilde-pa12_tilde);

          // Calculate derived variables.
          double zbar = (q2-m2ai-m2jk) / bABC(q2,m2ai,m2jk)
                      *( (xCS - 1)/(xCS-uCS)  - m2jk / gABC(q2,m2ai,m2jk)
                             * (m2ai + m2i - m2a) / (q2 - m2ai - m2jk));
          double kT2  = zbar*(1.-zbar)*m2ai - (1-zbar)*m2i - zbar*m2a;

          // Now construct recoiler in lab frame.
          Vec4 pjk( (pb_tilde - q*pb_tilde/q2*q)
                     *sqrt(lABC(q2,m2ai,m2jk)/lABC(q2,m2aij,m2k))
                   + 0.5*(q2+m2jk-m2ai)/q2*q );

          // Construct left-over dipole momentum by momentum conservation.
          Vec4 pai(-q+pjk);

          // Set up kT vector by using two perpendicular four-vectors.
          pair<Vec4, Vec4> pTvecs = getTwoPerpendicular(pai, pjk);
          Vec4 kTmom( sqrt(kT2)*sin(dip.phi)*pTvecs.first
                    + sqrt(kT2)*cos(dip.phi)*pTvecs.second);

          // Construct new emission momentum.
          Vec4 pi( - zbar *(gABC(q2,m2ai,m2jk)*pai + m2ai*pjk)
                          / bABC(q2,m2ai,m2jk)
                    + ( (1.-zbar)*m2ai + m2i - m2a) / bABC(q2,m2ai,m2jk)
                    * (pjk + m2jk/gABC(q2,m2ai,m2jk)*pai)
                    + kTmom);

          // Contruct radiator momentum by momentum conservation.
          pa.p(-q+pjk+pi);

          // Now get changed eCM.
          sHatAft = (pa + pOther).m2Calc();
        }

        // Now calculate Jacobian.
        double m2Other = pOther.m2Calc();
        double rho_aij = sqrt( lABC(sHatBef, m2a, m2Other)
                              /lABC(sHatAft, m2a, m2Other));
        jacobian = rho_aij / dip.z * (sijk + m2a - q2)
          / sqrt(lABC(sijk, m2a, q2));
        // Additional jacobian for non-competing steps.
        jacobian *= -q2 * dip.xa / dip.z / sqrt(lABC(m2jk, m2ai, q2));
        // Additional factor from massive propagator.
        jacobian *= 1. / (1. - (m2ai + m2j - m2aij) / (dip.pT2/dip.xa)) ;
      }

      // Calculate CS variables.
      double xCS = znow;
      xMother = xDaughter/xCS;
    }

    // Multiply with Jacobian.
    fullWeightNow  *= jacobian;
    for ( unordered_map<string,double>::iterator it = fullWeightsNow.begin();
      it != fullWeightsNow.end(); ++it )
      it->second   *= jacobian;

    // Evaluation of new daughter and mother PDF's.
    double pdfRatio = 1.;
    pdfScale2 = (useFixedFacScale) ? fixedFacScale2 : factorMultFac * tnow;
    pdfScale2 = max(pdfScale2, pT2min);
    double pdfScale2Old = pdfScale2;
    double pdfScale2New = pdfScale2;
    if (forceBranching) pdfScale2Old = pdfScale2New = infoPtr->Q2Fac();
    bool inD = (hasPDFdau) ? beam.insideBounds(xDaughter, pdfScale2Old) : true;
    bool inM = (hasPDFdau) ? beam.insideBounds(xMother,   pdfScale2New) : true;
    double xPDFdaughterNew = getXPDF( idDaughter, xDaughter, pdfScale2Old,
      iSysNow, &beam, true, znow, m2Dip);
    double xPDFmotherNew   = getXPDF( idMother, xMother, pdfScale2New,
      iSysNow, &beam, true, znow, m2Dip);

    if ( hasPDFdau && xPDFdaughterNew != 0.
      && abs(xPDFdaughterNew) < tinypdf(xDaughter) ) {
      hasTinyPDFdau = true;
      needNewPDF = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Extreme case of low-scale gluon, for which denominator rapidly
    // approaches zero. In this case, cut off branching probability if
    // daughter PDF fell too rapidly, to avoid large shower weights.
    // (Note: Last resort - would like something more physical here!)
    double xPDFdaughterLow = getXPDF( idDaughter, xDaughter,
      pdfScale2Old * pdfScale2Old/max(teval,pT2min), iSysNow, &beam);
    if ( hasPDFdau && idDaughter == 21
      && ( abs(xPDFdaughterNew/xPDFdaughter) < 1e-4
        || abs(xPDFdaughterLow/xPDFdaughterNew) < 1e-4) ) {
      hasTinyPDFdau = true;
      needNewPDF    = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Set PDF ratio to zero if x-/pT-values are out-of-bounds.
    pdfRatio = (hasPDFdau) ?
      ((inD && inM) ? xPDFmotherNew/xPDFdaughterNew : 0.) : 1.;

    // More last resort.
    if (idDaughter == 21 && pdfScale2 < 1.01 && pdfRatio > 50.) pdfRatio = 0.;

    fullWeightNow  *= pdfRatio;
    for ( unordered_map<string,double>::iterator it = fullWeightsNow.begin();
      it != fullWeightsNow.end(); ++it )
      it->second   *= pdfRatio;

    // Before generating kinematics: Reset sai if the kernel fell on an
    // endpoint contribution.
    if ( splits[splittingNowName]->nEmissions() == 2 )
      dip.sa1 = splits[splittingNowName]->splitInfo.kinematics()->sai;

    if (fullWeightNow == 0.) {
      needNewPDF = true;
      fullWeightsNow.clear();
      wt = fullWeightNow = overWeightNow = auxWeightNow = 0.;
      nContinue++; continue;
    }

    // Retrieve argument of alphaS.
    double m2DipCorr  = dip.m2Dip - m2Bef + m2r + m2e;
    double scale2 =  splits[splittingNowName]->couplingScale2 (
      dip.z, tnow, m2DipCorr,
      make_pair (event[dip.iRadiator].id(), event[dip.iRadiator].isFinal()),
      make_pair (event[dip.iRecoiler].id(), event[dip.iRecoiler].isFinal()));
    if (scale2 < 0.) scale2 = tnow;
    double talpha = max(scale2, pT2min);

    // Reweight to match PDF alpha_s, including corrective terms for
    // renormalisation scale variation.
    alphasReweight(tnow, talpha, dip.system, forceFixedAs, wt, fullWeightNow,
      overWeightNow, renormMultFac);

    // Create muR-variations.
    double asw = 1.;
    alphasReweight(tnow, talpha, dip.system, forceFixedAs, daux, asw, daux,
      renormMultFac);
    fullWeightsNow["base"] *= asw;
    if (fullWeightsNow.find("base_order_as2") != fullWeightsNow.end())
      fullWeightsNow["base_order_as2"] *= asw;
    if (doVariations) {
      if ( splittingNowName.find("qcd") != string::npos
        && settingsPtr->parm("Variations:muRisrDown") != 1.) {
        asw = 1.;
        alphasReweight(tnow, talpha, dip.system, forceFixedAs, daux, asw, daux,
          (tnow > pT2minVariations) ? settingsPtr->parm
                       ("Variations:muRisrDown")*renormMultFac
          : renormMultFac);
        fullWeightsNow["Variations:muRisrDown"] *= asw;
      } else if ( splittingNowName.find("qcd") == string::npos )
        fullWeightsNow["Variations:muRisrDown"] *= asw;
      if ( splittingNowName.find("qcd") != string::npos
        && settingsPtr->parm("Variations:muRisrUp")   != 1.) {
        asw = 1.;
        alphasReweight(tnow, talpha, dip.system, forceFixedAs, daux, asw, daux,
          (tnow > pT2minVariations) ? settingsPtr->parm
                       ("Variations:muRisrUp")*renormMultFac
          : renormMultFac);
        fullWeightsNow["Variations:muRisrUp"] *= asw;
      } else if ( splittingNowName.find("qcd") == string::npos )
        fullWeightsNow["Variations:muRisrUp"] *= asw;

      // PDF variations.
      if (hasPDFdau && settingsPtr->flag("Variations:PDFup") ) {
        int valSea = (beam[iSysNow].isValence()) ? 1 : 0;
        if( beam[iSysNow].isUnmatched() ) valSea = 2;
        beam.calcPDFEnvelope( make_pair(idMother, idDaughter),
          make_pair(xMother, xDaughter), pdfScale2, valSea);
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

    auxWeightNow = overWeightNow;

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
        << " and z=" << znow << "\t(PDF ratio=" << pdfRatio << ")" << " "
                              << xMother << endl;
      double rescale = fullWeightNow/auxWeightNow * 1.15;
      auxWeightNow *= rescale;
      needNewPDF = true;
      infoPtr->errorMsg("Info in DireSpace::pT2nextQCD_IF: Found large "
                        "acceptance weight for " + splittingNowName);
    }

    wt = fullWeightNow/auxWeightNow;

  // Iterate until acceptable pT (or have fallen below pTmin).
  } while (wt < rndmPtr->flat()) ;

  // Not possible to find splitting.
  if ( wt == 0.) { dip.pT2 = 0.0; return false; }

  // Update accepted event weight. No weighted shower for first
  // "pseudo-emission" step in 1->3 splitting.
  if ( fullWeightNow != 0. && overWeightNow != 0. ) {
    double enhanceFurther
      = enhanceOverestimateFurther(splittingNowName, idDaughter, teval);
    if (doTrialNow) {
      weights->addTrialEnhancement(tnow, enhanceFurther);
      enhanceFurther = 1.;
    }
    kernelNow = fullWeightsNow;
    auxNow = auxWeightNow;
    overNow = overWeightNow;
    boostNow = enhanceFurther;
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

  double zStore   = dip.z;
  double xaStore  = dip.xa;
  double pT2Store = dip.pT2;
  double sa1Store = dip.sa1;
  double Q2store  = 0.;

  // Save values for (so far) acceptable branching.
  dipEndNow->store( idDaughter, idMother, idSister,
    x1Now, x2Now, m2Dip, pT2Store, zStore, sa1Store, xaStore, xMother,
    Q2store, mSister, m2Sister, pT2corr, dip.phi, dip.phia1);

  // Done
  return true;

}

//--------------------------------------------------------------------------

double DireSpace::tNextQCD( DireSpaceEnd*, double overestimateInt,
   double tOld, double tMin, double tFreeze, int algoType) {

  bool forceFixedAs = (tOld < pT2min);
  double asOver     = (usePDFalphas || forceFixedAs)
                      ? alphaS2piOverestimate : alphaS2pi;
  // Return if below cut-off.
  double rnd    = rndmPtr->flat();
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
  double Q2alphaS;
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
      Q2alphaS = renormMultFac * max( tnow+tFreeze,
        pow2(LAMBDA3MARGIN) * Lambda3flav2);
    } while (alphaS.alphaS2OrdCorr(Q2alphaS) < rndmPtr->flat()
      && tnow > tMin);
  }

  // Done.
  return tnow;

}

//--------------------------------------------------------------------------

// Get auxiliary variable for branching of a QCD dipole end.

bool DireSpace::zCollNextQCD( DireSpaceEnd* dip, double zMin, double zMax,
  double, double ) {

  // Choose logarithmically.
  dip->xa = zMax * pow( zMax/zMin, -rndmPtr->flat());

  // Done
  return true;

}

//--------------------------------------------------------------------------

// Evolve a QCD dipole end.

bool DireSpace::virtNextQCD( DireSpaceEnd* dip, double, double,
  double, double) {

  double v   = (dip->z/dip->xa) * rndmPtr->flat();
  double m2j = dip->mass[2];
  dip->sa1 = v / (dip->z/dip->xa-v) * ( dip->pT2/dip->xa - m2j);
  if (abs(dip->z/dip->xa-v) < 1e-10) return false;

  // Done
  return true;

}

//--------------------------------------------------------------------------

// Kinematics of branching.
// Construct mother -> daughter + sister, with recoiler on other side.

bool DireSpace::branch( Event& event) {

  if (abs(dipEndSel->pT2 - pT2cutMin(dipEndSel)) < 1e-10) return false;

  // This function is a wrapper for setting up the branching
  // kinematics.
  bool hasBranched = false;
  if ( event[dipEndSel->iRecoiler].isFinal() )
       hasBranched = branch_IF(event, false, &splitInfoSel);
  else hasBranched = branch_II(event, false, &splitInfoSel);

  // Done.
  return hasBranched;

}

//--------------------------------------------------------------------------

// Kinematics of branching.
// Construct mother -> daughter + sister, with recoiler on other side.

bool DireSpace::branch_II( Event& event, bool trial,
  DireSplitInfo* split ) {

  Event auxevent1 = event;
  Event auxevent2 = event;

  // Side on which branching occured.
  int side          = (!trial) ? abs(dipEndSel->side) : split->side;

  // Read in flavour and colour variables.
  int iDaughter     = (!trial) ? dipEndSel->iRadiator  : split->iRadBef;
  int iRecoiler     = (!trial) ? dipEndSel->iRecoiler  : split->iRecBef;
  int idDaughterNow = (!trial) ? dipEndSel->idDaughter : split->radBef()->id;
  int idMother      = (!trial) ? dipEndSel->idMother   : split->radAft()->id;
  int idSister      = (!trial) ? dipEndSel->idSister   : split->emtAft()->id;
  int colDaughter   = event[iDaughter].col();
  int acolDaughter  = event[iDaughter].acol();
  int colRecBef     = event[iRecoiler].col();
  int acolRecBef    = event[iRecoiler].acol();
  bool colMatch     = (acolRecBef == colDaughter);
  bool acolMatch    = (colRecBef  == acolDaughter);

  // Name of the splitting.
  string name = (!trial) ? splittingSelName : split->splittingSelName;
  int nEmissions = splits[name]->nEmissions();

  if ( nEmissions == 2 ) idSister = -event[iDaughter].id();

  // Read in kinematical variables.
  double pT2        = (!trial) ? dipEndSel->pT2 : split->kinematics()->pT2;
  double z          = (!trial) ? dipEndSel->z   : split->kinematics()->z;
  splits[name]->splitInfo.store(*split);
  unordered_map<string,double> psp(splits[name]->getPhasespaceVars
                                   (event, partonSystemsPtr));
  // Allow splitting kernel to overwrite phase space variables.
  if (split->useForBranching) { pT2 = psp["pT2"]; z = psp["z"]; }

  // Get particle masses.
  double m2Bef = 0.0, m2r = 0.0, m2s = 0.0;
  // Emission.
  double m2ex = (abs(idSister) < 6) ?
    getMass(idSister, 2) : getMass(idSister, 1);
  double m2e  = (!trial) ? m2ex
    : ( (split->kinematics()->m2EmtAft > 0.) ? split->kinematics()->m2EmtAft
                                            : m2ex);
  // Radiator mass.
  if ( useMassiveBeams && (abs(idDaughter) == 11 || abs(idDaughter) == 13
                           || abs(idDaughter) > 900000))
    m2Bef = getMass(idDaughter,1);
  if ( useMassiveBeams && (abs(idMother) == 11 || abs(idMother) == 13
                           || abs(idMother) > 900000))
    m2r   = getMass(idMother,1);
  // Recoiler mass
  if ( useMassiveBeams && (event[iRecoiler].idAbs() == 11
                        || event[iRecoiler].idAbs() == 13
                        || event[iRecoiler].idAbs() > 900000))
    m2s   = getMass(event[iRecoiler].id(),1);
  // Emission mass
  if ( useMassiveBeams && (abs(idSister) == 11 || abs(idSister) == 13
                           || abs(idSister) > 900000))
    m2e   = getMass(idSister,1);

  // Force emission massless by default.
  if (!forceMassiveMap) m2e = 0.0;
  double m2dip     = (!trial) ? dipEndSel->m2Dip : split->kinematics()->m2Dip;
  double m2DipCorr = m2dip - m2Bef + m2r + m2e;
  double kappa2    = pT2 / m2DipCorr;
  double xCS       = (z*(1-z)- kappa2)/(1-z);
  double vCS       = kappa2/(1-z);
  double sai       = (!trial) ? dipEndSel->sa1 : split->kinematics()->sai;
  double xa        = (!trial) ? dipEndSel->xa  : split->kinematics()->xa;
  // Allow splitting kernel to overwrite phase space variables.
  if (split->useForBranching) { sai = psp["sai"]; xa = psp["xa"]; }
  double m2Rad     = m2r;
  double m2Emt     = m2e;

  // Get dipole 4-momentum.
  Vec4 paj_tilde(event[iDaughter].p());
  Vec4 pb_tilde(event[iRecoiler].p());
  Vec4 q(pb_tilde+paj_tilde);
  double q2   = q.m2Calc();

  // Current event and subsystem size.
  int eventSizeOld  = event.size();
  int iSysSelNow    = (!trial) ? iSysSel : split->system;
  int systemSizeOld = partonSystemsPtr->sizeAll(iSysSelNow);

  // Save properties to be restored in case of user-hook veto of emission.
  int beamOff1 = 1 + beamOffset;
  int beamOff2 = 2 + beamOffset;
  int ev1Dau1V = event[beamOff1].daughter1();
  int ev2Dau1V = event[beamOff2].daughter1();
  vector<int> statusV, mother1V, mother2V, daughter1V, daughter2V;

  // Check if the first emission should be checked for removal.
  bool physical      = true;
  bool canMergeFirst = (mergingHooksPtr != 0)
                     ? mergingHooksPtr->canVetoEmission() : false;
  for ( int iCopy = 0; iCopy < systemSizeOld; ++iCopy) {
    int iOldCopy    = partonSystemsPtr->getAll(iSysSelNow, iCopy);
    statusV.push_back( event[iOldCopy].status());
    mother1V.push_back( event[iOldCopy].mother1());
    mother2V.push_back( event[iOldCopy].mother2());
    daughter1V.push_back( event[iOldCopy].daughter1());
    daughter2V.push_back( event[iOldCopy].daughter2());
  }

  // Take copy of existing system, to be given modified kinematics.
  // Incoming negative status. Rescattered also negative, but after copy.
  int iMother(0), iNewRecoiler(0);
  for ( int iCopy = 0; iCopy < systemSizeOld; ++iCopy) {
    int iOldCopy    = partonSystemsPtr->getAll(iSysSelNow, iCopy);
    int statusOld   = event[iOldCopy].status();
    int statusNew   = (iOldCopy == iDaughter
      || iOldCopy == iRecoiler) ? statusOld : 44;
    int iNewCopy    = event.copy(iOldCopy, statusNew);
    auxevent1.copy(iOldCopy, statusNew);
    auxevent2.copy(iOldCopy, statusNew);
    if (iOldCopy == iDaughter) iMother      = iNewCopy;
    if (iOldCopy == iRecoiler) iNewRecoiler = iNewCopy;
    if (statusOld < 0) {
      event[iNewCopy].statusNeg();
      auxevent1[iNewCopy].statusNeg();
      auxevent2[iNewCopy].statusNeg();
    }
  }

  // For 1->3 splitting, intermediate mother is a gluon.
  int idMotherNow = idMother;
  if ( nEmissions == 2 ) idMotherNow = 21;

  // Define colour flow in branching.
  // Default corresponds to f -> f + gamma.
  int colMother     = colDaughter;
  int acolMother    = acolDaughter;
  int colSister     = 0;
  int acolSister    = 0;

  double RN2 = rndmPtr->flat();
  if (idSister == 22 || idSister == 23 || idSister == 25) ;

  // q -> q + g and 50% of g -> g + g; need new colour.
  else if (idSister == 21 && ( (idMotherNow > 0 && idMotherNow < 9)
    || ( idMother == 21 && colMatch && acolMatch && RN2 < 0.5) ) ) {
    colMother       = event.nextColTag();
    colSister       = colMother;
    acolSister      = colDaughter;
  // qbar -> qbar + g and other 50% of g -> g + g; need new colour.
  } else if (idSister == 21 && ( (idMotherNow < 0 && idMotherNow > -9)
    || ( idMotherNow == 21 && colMatch && acolMatch) ) ) {
    acolMother      = event.nextColTag();
    acolSister      = acolMother;
    colSister       = acolDaughter;
  } else if (idMotherNow == 21 && idSister == 21 && colMatch && !acolMatch) {
    colMother       = event.nextColTag();
    acolMother      = acolDaughter;
    colSister       = colMother;
    acolSister      = colDaughter;
  } else if (idMotherNow == 21 && idSister == 21 && !colMatch && acolMatch) {
    colMother       = colDaughter;
    acolMother      = event.nextColTag();
    acolSister      = acolMother;
    colSister       = acolDaughter;
  // q -> g + q.
  } else if (idDaughterNow == 21 && idMotherNow > 0) {
    colMother       = colDaughter;
    acolMother      = 0;
    colSister       = acolDaughter;
  // qbar -> g + qbar
  } else if (idDaughterNow == 21) {
    acolMother      = acolDaughter;
    colMother       = 0;
    acolSister      = colDaughter;
  // g -> q + qbar.
  } else if (idDaughterNow > 0 && idDaughterNow < 9) {
    acolMother      = event.nextColTag();
    acolSister      = acolMother;
  // g -> qbar + q.
  } else if (idDaughterNow < 0 && idDaughterNow > -9) {
    colMother       = event.nextColTag();
    colSister       = colMother;
  }

  // Swap colours if radiated gluon carries momentum fraction z
  int colSave = colSister, acolSave = acolSister;
  if ( splits[name]->is(splittingsPtr->isrQCD_21_to_21_and_21b)) {
    colSister  = acolMother;
    acolSister = colMother;
    colMother  = acolSave;
    acolMother = colSave;
  }

  if (split->useForBranching) {
    idMotherNow= event[iDaughter].id();
    colMother  = event[iDaughter].col();
    acolMother = event[iDaughter].acol();
    colSister  = 0;
    acolSister = 0;
    // Now reset if splitting information is available.
    if (split->radAft()->id   != 0) idMotherNow= split->radAft()->id;
    if (split->emtAft()->id   != 0) idSister   = split->emtAft()->id;
    if (split->radAft()->col  > -1) colMother  = split->radAft()->col;
    if (split->radAft()->acol > -1) acolMother = split->radAft()->acol;
    if (split->emtAft()->col  > -1) colSister  = split->emtAft()->col;
    if (split->emtAft()->acol > -1) acolSister = split->emtAft()->acol;
  }

  int colMother1, acolMother1;
  int colSister1, acolSister1;
  colMother1 = acolMother1 = colSister1 = acolSister1 = 0;
  if ( nEmissions == 2 ) {
    // Daughter color transferred to quark mother "1", sister anti-color
    // transferred to sister "1" color.
    if (idMother*idDaughterNow > 0 && idMother > 0) {
      colMother1  = colDaughter;
      colSister1  = acolSister;
    }
    // Daughter anticolor transferred to antiquark mother "1", sister color
    // transferred to sister "1" anticolor.
    if (idMother*idDaughterNow > 0 && idMother < 0) {
      colMother1  = 0;
      acolMother1 = acolDaughter;
      acolSister1 = colSister;
      colSister1  = 0;
    }
    // Sister color transferred to quark mother "1", daughter anti-color
    // transferred to sister "1" color.
    if (idMother*idDaughterNow < 0 && idMother > 0) {
      colMother1  = colSister;
      acolMother1 = 0;
      acolSister1 = 0;
      colSister1  = acolDaughter;
      // Reset dummy mother colours.
      acolMother = acolDaughter;
      colMother  = colSister;
    }
    // Sister anticolor transferred to antiquark mother "1", daughter color
    // transferred to sister "1" anti-color.
    if (idMother*idDaughterNow < 0 && idMother < 0) {
      colMother1  = 0;
      acolMother1 = acolSister;
      acolSister1 = colDaughter;
      colSister1  = 0;
      // Reset dummy mother colours.
      acolMother = acolSister;
      colMother  = colDaughter;
    }
  }

  // Indices of partons involved. Add new sister. For 1->3 splitting, replace
  // mother by dummy and attach "real" mother later.
  int iMother1  = 0;
  if ( nEmissions == 2 )
    iMother1 = event.append( 0, 0, 0, 0, 0, 0, 0, 0, Vec4(0.,0.,0.,0.), 0.0,
                 sqrt(pT2) );

  int iSister       = event.append( idSister, 43, iMother, 0, 0, 0,
     colSister, acolSister, Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );
  auxevent1.append( idSister, 43, iMother, 0, 0, 0,
     colSister, acolSister, Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );
  auxevent2.append( idSister, 43, iMother, 0, 0, 0,
     colSister, acolSister, Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );

  // Second sister particle for 1->3 splitting.
  int iSister1      = 0;
  if ( nEmissions == 2 )
    iSister1 = event.append( 0, 0, 0, 0, 0, 0, 0, 0, Vec4(0.,0.,0.,0.), 0.0,
                 sqrt(pT2) );

  // References to the partons involved.
  Particle& daughter    = event[iDaughter];
  Particle& mother      = event[iMother];
  Particle& newRecoiler = event[iNewRecoiler];
  Particle& sister      = event[iSister];
  Particle& mother1     = event[iMother1];
  Particle& sister1     = event[iSister1];

  // Replace old by new mother; update new recoiler.
  mother.id( idMotherNow );
  mother.status( -41);
  mother.cols( colMother, acolMother);
  mother.p(0.,0., event[iDaughter].pz()/xCS, event[iDaughter].e()/xCS);
  if (mother.idAbs() == 21 || mother.idAbs() == 22) mother.pol(9);
  newRecoiler.status(-42);
  newRecoiler.p(0.,0., event[iRecoiler].pz(), event[iRecoiler].e());

  // Update mother and daughter pointers; also for beams.
  daughter.mothers( iMother, 0);
  mother.daughters( iSister, iDaughter);
  int iBeam1Dau1 = event[beamOff1].daughter1();
  int iBeam2Dau1 = event[beamOff2].daughter1();
  if (iSysSelNow == 0) {
    event[beamOff1].daughter1( (side == 1) ? iMother : iNewRecoiler );
    event[beamOff2].daughter1( (side == 2) ? iMother : iNewRecoiler );
  }

  auxevent1[iMother].id( idMotherNow );
  auxevent1[iMother].status( -41);
  auxevent1[iMother].cols( colMother, acolMother);
  auxevent1[iMother].p(0.,0., event[iDaughter].pz()/xCS,
                       event[iDaughter].e()/xCS);
  if (auxevent1[iMother].idAbs() == 21 || auxevent1[iMother].idAbs() == 22)
    auxevent1[iMother].pol(9);
  auxevent1[iNewRecoiler].status(-42);
  auxevent1[iNewRecoiler].p(0.,0., auxevent1[iRecoiler].pz(),
                            auxevent1[iRecoiler].e());
  auxevent1[iDaughter].mothers( iMother, 0);
  auxevent1[iMother].daughters( iSister, iDaughter);
  if (iSysSelNow == 0) {
    auxevent1[beamOff1].daughter1( (side == 1) ? iMother : iNewRecoiler );
    auxevent1[beamOff2].daughter1( (side == 2) ? iMother : iNewRecoiler );
  }

  auxevent2[iMother].id( idMotherNow );
  auxevent2[iMother].status( -41);
  auxevent2[iMother].cols( colMother, acolMother);
  auxevent2[iMother].p(0.,0., event[iDaughter].pz()/xCS,
                       event[iDaughter].e()/xCS);
  if (auxevent2[iMother].idAbs() == 21 || auxevent2[iMother].idAbs() == 22)
    auxevent2[iMother].pol(9);
  auxevent2[iNewRecoiler].status(-42);
  auxevent2[iNewRecoiler].p(0.,0., auxevent2[iRecoiler].pz(),
                            auxevent2[iRecoiler].e());
  auxevent2[iDaughter].mothers( iMother, 0);
  auxevent2[iMother].daughters( iSister, iDaughter);
  if (iSysSelNow == 0) {
    auxevent2[beamOff1].daughter1( (side == 1) ? iMother : iNewRecoiler );
    auxevent2[beamOff2].daughter1( (side == 2) ? iMother : iNewRecoiler );
  }

  bool doVeto = false;
  bool printWarnings = (!trial || forceMassiveMap);
  bool doMECreject = false;

  // Regular massive kinematics for 1+1 -> 2+1 splitting
  if ( nEmissions != 2 ) {

    // Calculate derived variables.
    double sab  = (q2 - m2Emt)/xCS + (m2Rad+m2s) * (1-1/xCS);
    double saj  = -vCS*(sab - m2Rad-m2s) + m2Rad + m2Emt;
    double zbar = (sab - m2Rad - m2s) / bABC(sab,m2Rad,m2s)
                *( (xCS + vCS)  - m2s / gABC(sab,m2Rad,m2s)
                       * (saj + m2Rad - m2Emt) / (sab - m2Rad - m2s));
    double kT2  = zbar*(1.-zbar)*m2Rad - (1-zbar)*saj - zbar*m2Emt;

    // Not possible to construct kinematics if kT2 < 0.0
    if (kT2 < 0.) {
      if (printWarnings)
        infoPtr->errorMsg("Warning in DireSpace::branch_II: Reject state "
                          "with kinematically forbidden kT^2.");
      physical = false;
    }

    // Now construct radiator in lab frame.
    Vec4 pRad = (paj_tilde - m2Bef/gABC(q2,m2Bef,m2s)*pb_tilde)
               *sqrt(lABC(sab,m2Rad,m2s)/lABC(q2,m2Bef,m2s))
             + m2Rad / gABC(sab,m2Rad,m2s)*pb_tilde;

    // Set momenta.
    mother.p(pRad);
    newRecoiler.p(pb_tilde);
    mother.m(sqrtpos(m2Rad));
    newRecoiler.m(sqrtpos(m2s));

    auxevent1[iMother].p(pRad);
    auxevent1[iNewRecoiler].p(pb_tilde);
    auxevent1[iMother].m(sqrtpos(m2Rad));
    auxevent1[iNewRecoiler].m(sqrtpos(m2s));

    auxevent2[iMother].p(pRad);
    auxevent2[iNewRecoiler].p(pb_tilde);
    auxevent2[iMother].m(sqrtpos(m2Rad));
    auxevent2[iNewRecoiler].m(sqrtpos(m2s));

    // Store momenta in case momentum construction fails.
    Vec4 kTilde(paj_tilde + pb_tilde);

    Event NewEvent = Event();
    NewEvent.init("(hard process-modified)", particleDataPtr);
    // Copy all unchanged particles to NewEvent
    for (int i = 0; i < event.size(); ++i)
      NewEvent.append( event[i] );

    // Construct dummy overall momentum.
    Vec4 pSum(1e5,1e5,1e5,1e5);
    Vec4 pSumIn( beamAPtr->p() + beamBPtr->p() );

    // Now produce momenta of emitted and final state particles, and ensure
    // good momentum conservation. (More than one try only necessary in rare
    // numerical instabilities.
    int nTries = 0;
    while ( abs(pSum.px()-pSumIn.px()) > mTolErr
         || abs(pSum.py()-pSumIn.py()) > mTolErr ) {

      // Give up after too many tries.
      nTries++;
      if (nTries > 100
       || (nTries > 1 && split->useForBranching)) {
        if (printWarnings)
          infoPtr->errorMsg("Warning in DireSpace::branch_II: Could not "
                               "set up state after branching, thus reject.");
        physical = false; break;
      }

      // Now construct the transverse momentum vector in the dipole CM frame.
      double phi_kt = (!trial) ? 2.*M_PI*rndmPtr->flat()
        : (split->kinematics()->phi < 0. ?
           2.*M_PI*rndmPtr->flat() : split->kinematics()->phi);

      // Allow splitting kernel to overwrite phase space variables.
      if (split->useForBranching) { phi_kt = psp["phi"]; }

      double phi_kt1 = phi_kt+DPHI_II*M_PI;
      if (phi_kt1>2.*M_PI) phi_kt1 -= 2.*M_PI;
      double phi_kt2 = phi_kt-DPHI_II*M_PI;
      if (phi_kt2<0.)      phi_kt2 += 2.*M_PI;
      if (phi_kt1<phi_kt2) swap(phi_kt1, phi_kt2);

      // Set up transverse momentum vector by using two perpendicular
      // four-vectors.
      pair<Vec4, Vec4> pTvecs = getTwoPerpendicular(pRad, pb_tilde);
      Vec4 kTmom( sqrt(kT2)*sin(phi_kt)*pTvecs.first
                + sqrt(kT2)*cos(phi_kt)*pTvecs.second);

      // Construct new emission momentum.
      Vec4 pEmt = (1-zbar) * (gABC(sab,m2Rad,m2s)*pRad - m2Rad*pb_tilde)
                              / bABC(sab,m2Rad,m2s)
                + (m2Emt + kT2) / ((1-zbar)*bABC(sab,m2Rad,m2s))
                * (pb_tilde - m2s/gABC(sab,m2Rad,m2s)*pRad)
                + kTmom;

      // Set all momenta.
      sister.p(pEmt);
      sister.m(sqrtpos(m2Emt));

      kTmom.p( sqrt(kT2)*sin(phi_kt1)*pTvecs.first
             + sqrt(kT2)*cos(phi_kt1)*pTvecs.second);
      pEmt.p(  (1-zbar) * (gABC(sab,m2Rad,m2s)*pRad - m2Rad*pb_tilde)
                              / bABC(sab,m2Rad,m2s)
                + (m2Emt + kT2) / ((1-zbar)*bABC(sab,m2Rad,m2s))
                * (pb_tilde - m2s/gABC(sab,m2Rad,m2s)*pRad)
                + kTmom);
      auxevent1[iSister].p(pEmt);
      auxevent1[iSister].m(sqrtpos(m2Emt));

      kTmom.p( sqrt(kT2)*sin(phi_kt2)*pTvecs.first
             + sqrt(kT2)*cos(phi_kt2)*pTvecs.second);
      pEmt.p(  (1-zbar) * (gABC(sab,m2Rad,m2s)*pRad - m2Rad*pb_tilde)
                              / bABC(sab,m2Rad,m2s)
                + (m2Emt + kT2) / ((1-zbar)*bABC(sab,m2Rad,m2s))
                * (pb_tilde - m2s/gABC(sab,m2Rad,m2s)*pRad)
                + kTmom);
      auxevent2[iSister].p(pEmt);
      auxevent2[iSister].m(sqrtpos(m2Emt));

      // Gather total momentum for subsequent check that total pT vanishes.
      vector<int> iPos(1,iSister);
      pSum = sister.p();

      // Transform all final state momenta to distribute recoil.
      Vec4 k(mother.p() + newRecoiler.p()  - sister.p());
      Vec4 kSum(kTilde + k);
      for ( int i = eventSizeOld + 2; i < eventSizeOld + systemSizeOld; ++i) {
        Vec4 pIn = NewEvent[i].p();
        double kSum2    = kSum.m2Calc();
        double k2       = k.m2Calc();
        double kTildeXp = kTilde*pIn;
        double kSumXp   = kSum*pIn;
        Vec4 res = pIn - kSum * 2.0*( kSumXp / kSum2 )
                       + k * 2.0 *( kTildeXp/k2);
        event[i].p(res);
        auxevent1[i].p(res);
        auxevent2[i].p(res);
        // If Lorentz transformation fails to be accurate enough, set pSum
        // to force another trial.
        if (!validMomentum(event[i].p(), event[i].id(), event[i].status()))
          pSum += event[i].p();
        iPos.push_back(i);
        if (event[i].status() > 0) pSum += event[i].p();
      }

      // Collect remaining final state momenta.
      for (int i = 0; i < event.size(); ++i)
        if ( event[i].isFinal()
          && partonSystemsPtr->getSystemOf(i,true) == iSysSelNow
          && find(iPos.begin(), iPos.end(), i) == iPos.end() )
          pSum += event[i].p();
    }

    // Check momenta.
    if ( !validMomentum( mother.p(), idMother, -1)
      || !validMomentum( sister.p(), idSister,  1)
      || !validMomentum( newRecoiler.p(), event[iNewRecoiler].id(), -1) )
      physical = false;

    doVeto = (( canVetoEmission && userHooksPtr->doVetoISREmission(
                eventSizeOld, event, iSysSelNow))
           || ( canMergeFirst   && mergingHooksPtr->doVetoEmission(event)) );

    double xm = 2.*mother.e() / (beamAPtr->e() + beamBPtr->e());

    // Test that enough beam momentum remains.
    double xAnew = (mother.mother1() == 1)
              ? 2.*mother.e()      / (beamAPtr->e() + beamBPtr->e())
              : 2.*newRecoiler.e() / (beamAPtr->e() + beamBPtr->e());
    double iAold = (mother.mother1() == 1) ? iDaughter : iRecoiler;
    double iAnew = (mother.mother1() == 1) ? iMother : iNewRecoiler;
    double xBnew = (mother.mother1() == 1)
              ? 2.*newRecoiler.e() / (beamAPtr->e() + beamBPtr->e())
              : 2.*mother.e()      / (beamAPtr->e() + beamBPtr->e());
    double iBold = (mother.mother1() == 1) ? iRecoiler : iDaughter;
    double iBnew = (mother.mother1() == 1) ? iNewRecoiler : iMother;
    if ( hasPDF(event[iAnew].id()) && beamAPtr->size() > 0) {
      double xOld = (*beamAPtr)[iSysSelNow].x();
      (*beamAPtr)[iSysSelNow].iPos(iAnew);
      (*beamAPtr)[iSysSelNow].x(xAnew);
      if (beamAPtr->xMax(-1) < 0.0) {
        if (!trial) infoPtr->errorMsg("Warning in DireSpace::branch_II: "
          "used up beam momentum; discard splitting.");
        physical = false;
      }
      // Restore old beams.
      (*beamAPtr)[iSysSelNow].iPos(iAold);
      (*beamAPtr)[iSysSelNow].x(xOld);
    }
    if ( hasPDF(event[iBnew].id()) && beamBPtr->size() > 0) {
      double xOld = (*beamBPtr)[iSysSelNow].x();
      (*beamBPtr)[iSysSelNow].iPos(iBnew);
      (*beamBPtr)[iSysSelNow].x(xBnew);
      if (beamBPtr->xMax(-1) < 0.0) {
        if (!trial) infoPtr->errorMsg("Warning in DireSpace::branch_II: "
          "used up beam momentum; discard splitting.");
        physical = false;
      }
      // Restore old beams.
      (*beamBPtr)[iSysSelNow].iPos(iBold);
      (*beamBPtr)[iSysSelNow].x(xOld);
    }

    // Apply ME correction if necessary.
    bool isHardSystem = partonSystemsPtr->getSystemOf(iDaughter,true) == 0
                     && partonSystemsPtr->getSystemOf(iRecoiler,true) == 0;
    if (isHardSystem && physical && doMEcorrections
      && pT2 > pT2minMECs && checkSIJ(event,Q2minMECs)) {
      int iA      = getInA(iSysSelNow);
      int iB      = getInB(iSysSelNow);
      vector<int> iOut(createvector<int>(0)(0));
      for (int iCopy = 2; iCopy < systemSizeOld; ++iCopy)
        iOut.push_back(partonSystemsPtr->getOut(iSysSelNow, iCopy - 2));
      bool motherHasPlusPz = (event[iMother].pz() > 0.);
      if (motherHasPlusPz) {
        partonSystemsPtr->setInA(iSysSelNow, iMother);
        partonSystemsPtr->setInB(iSysSelNow, iNewRecoiler);
      } else {
        partonSystemsPtr->setInA(iSysSelNow, iNewRecoiler);
        partonSystemsPtr->setInB(iSysSelNow, iMother);
      }
      for (int iCopy = 2; iCopy < systemSizeOld; ++iCopy)
        partonSystemsPtr->setOut(iSysSelNow, iCopy - 2, eventSizeOld + iCopy);
      partonSystemsPtr->addOut(iSysSelNow, iSister);

      if ( nFinalMaxMECs < 0
        || nFinalMaxMECs > partonSystemsPtr->sizeOut(iSysSelNow))
        doMECreject = applyMEC (event, split,
          createvector<Event>(auxevent1)(auxevent2));

      partonSystemsPtr->setInA(iSysSelNow, iA);
      partonSystemsPtr->setInB(iSysSelNow, iB);
      for (int iCopy = 2; iCopy < systemSizeOld; ++iCopy)
        partonSystemsPtr->setOut(iSysSelNow, iCopy - 2, iOut[iCopy]);
      partonSystemsPtr->popBackOut(iSysSelNow);
    }

    // Update dipoles and beams. Note: dipEndSel no longer valid after this.
    if (physical && !doVeto && !trial && !doMECreject) updateAfterII(
      iSysSelNow, side,
      iDipSel, eventSizeOld, systemSizeOld, event, iDaughter, iMother, iSister,
      iNewRecoiler, pT2, xm);

  // Perform 1+1 -> 3 + 1 splitting.
  } else {

    // 1->3 splitting as one II and one FF step.
    double m2a   = 0.0;
    double m2i   = getMass(idMother,2);
    double m2j   = getMass(idSister,2);
    double m2ai  = -sai + m2a + m2i;
    double m2aij = 0.0;
    double m2k   = 0.0;
    q2         = (event[iDaughter].p()
                 +event[iRecoiler].p()).m2Calc();

    // Perform II step.
    double za     = z;

    // Calculate derived variables.
    double p2ab  = q2/za + m2a + m2k;
    double zbar = (p2ab - m2a - m2k) / bABC(p2ab,m2a,m2k)
                *( xa - m2k / gABC(p2ab,m2a,m2k)
                       * (m2ai + m2a - m2i) / (p2ab - m2a - m2k));
    double kT2  = zbar*(1.-zbar)*m2a - (1-zbar)*m2ai - zbar*m2i;

    // Disallow kinematically impossible transverse momentum.
    if (kT2 < 0. || isnan(kT2)) physical = false;

    // Now construct radiator in lab frame.
    Vec4 pa = (paj_tilde - m2aij/gABC(q2,m2aij,m2k)*pb_tilde)
              *sqrt(lABC(p2ab,m2a,m2k)/lABC(q2,m2aij,m2k))
              + m2a / gABC(p2ab,m2a,m2k)*pb_tilde;

    // Set momenta. Mother acts purely as a dummy, since we already have the
    // final incoming momentum. Recoiler unchanged.
    mother.p(pa);
    mother.m(sqrtpos(m2a));
    mother1.p(pa);
    mother1.m(sqrtpos(m2a));
    newRecoiler.p(pb_tilde);

    // Now construct the transverse momentum vector in the dipole CM frame.
    double phi_kt = (!trial)
      ? ((dipEndSel->phi > 0.)
        ? dipEndSel->phi          : 2.*M_PI*rndmPtr->flat())
      : ((split->kinematics()->phi > 0.)
        ? split->kinematics()->phi : 2.*M_PI*rndmPtr->flat());

    // Allow splitting kernel to overwrite phase space variables.
    if (split->useForBranching) { phi_kt = psp["phi"]; }

    // Set up transverse momentum vector by using two perpendicular
    // four-vectors
    Vec4 pijb(q-pa);

    // Set up transverse momentum vector by using two perpendicular
    // four-vectors.
    pair<Vec4, Vec4> pTvecs = getTwoPerpendicular(pa, pijb);
    Vec4 kTmom( sqrt(kT2)*sin(phi_kt)*pTvecs.first
           + sqrt(kT2)*cos(phi_kt)*pTvecs.second);

    // Construct new emission momentum.
    Vec4 pi = (1-zbar) * (gABC(p2ab,m2a,m2k)*pa - m2a*pb_tilde)
                       / bABC(p2ab,m2a,m2k)
              + (m2i + kT2) / ((1-zbar)*bABC(p2ab,m2a,m2k))
                * (pb_tilde - m2k/gABC(p2ab,m2a,m2k)*pa)
              + kTmom;

    // Set all momenta.
    sister1.p(pi);
    sister1.m(sqrtpos(m2i));

    // Perform FF step.
    Vec4 pai(pa-pi);
    Vec4 pRadBef(pai+pb_tilde);
    Vec4 pRecBef(pai);

    double phiFF = (!trial)
      ? ((dipEndSel->phia1 > 0.)
        ? dipEndSel->phia1         : 2.*M_PI*rndmPtr->flat())
      : ((split->kinematics()->phi2 > 0.)
        ? split->kinematics()->phi2 : 2.*M_PI*rndmPtr->flat());

    // Allow splitting kernel to overwrite phase space variables.
    if (split->useForBranching) { phiFF = psp["phi2"]; }

    // Calculate CS variables.
    double m2rec      = m2ai;
    double m2emt      = q2;
    double m2rad      = m2j;
    double zCS = pT2/xa / (q2*xa/za + 2.*m2ai);
    double yCS = 1. / ( 1. + (q2*xa/za + 2.*m2ai)
                           / (q2*(xa/za - 1.) + m2ai + m2k - m2j));

    // Construct FF dipole momentum.
    Vec4 qtilde(q);
    double q2tilde = qtilde.m2Calc();
    q.p(pRadBef + pRecBef);
    q2 = 4.*m2ai + 2.*q2tilde*xa/za + m2k;

    // Calculate derived variables.
    double sij  = yCS * (q2 - m2rec) + (1.-yCS)*(m2rad+m2emt);
    zbar = (q2-sij-m2rec) / bABC(q2,sij,m2rec)
                * (zCS - m2rec/gABC(q2,sij,m2rec)
                       *(sij + m2rad - m2emt)/(q2-sij-m2rec));
    kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2rad - zbar*m2emt;

    // Not possible to construct kinematics if kT2 < 0.0
    if (kT2 < 0.) {
      if (printWarnings)
        infoPtr->errorMsg("Warning in DireSpace::branch_II: Reject state "
                          "with kinematically forbidden kT^2.");
      physical = false;
    }

    // NaN kT2 can happen for a 1->3 splitting in which the g->QQ~ produces
    // massive quarks Q.
    if (physical && (kT2!=kT2 || abs(kT2-kT2) > 1e5) ) {
      if (printWarnings)
        infoPtr->errorMsg("Warning in DireSpace::branch_II: Reject state "
                          "with not-a-number kT^2 for branching " + name);
      physical = false;
    }

    // Construct left-over dipole momentum by momentum conservation.
    Vec4 pRec(pRecBef);
    Vec4 pij(q-pRec);

    // Set up transverse momentum vector by using two perpendicular
    // four-vectors.
    pTvecs = getTwoPerpendicular(pRec, pij);
    kTmom.p( sqrt(kT2)*sin(phiFF)*pTvecs.first
           + sqrt(kT2)*cos(phiFF)*pTvecs.second);

    // Construct new radiator momentum.
    Vec4 pj( zbar * (gABC(q2,sij,m2rec)*pij - sij*pRec) / bABC(q2,sij,m2rec)
              + (m2rad+kT2) / (zbar*bABC(q2,sij,m2rec))
                * (pRec - m2rec/gABC(q2,sij,m2rec)*pij)
              + kTmom);

    Vec4 qnew(q-pRec-pj);

    // Set particle momenta.
    sister.p(pj);
    sister.m(sqrtpos(m2j));

    // Gather total momentum for subsequent check that total pT vanishes.
    vector<int> iPos(1,iSister1);
    iPos.push_back(iSister);
    Vec4 pSum = mother1.p()+newRecoiler.p()-sister1.p()-sister.p();

    // Transform all final state momenta to distribute recoil.
    Vec4 kTilde(qtilde);
    Vec4 k(qnew);
    Vec4 kSum(kTilde + k);
    for ( int i = eventSizeOld + 2; i < eventSizeOld + systemSizeOld; ++i) {
      Vec4 pIn = event[i].p();
      double kSum2    = kSum.m2Calc();
      double k2       = k.m2Calc();
      double kTildeXp = kTilde*pIn;
      double kSumXp   = kSum*pIn;
      Vec4 res = pIn - kSum * 2.0*( kSumXp / kSum2 )
                     + k * 2.0 *( kTildeXp/k2);
      event[i].p(res);
      if (i != iSister && i != iSister1) {
        pSum -= event[i].p();
        iPos.push_back(i);
      }
    }

    // Check momenta.
    if ( !validMomentum( mother1.p(), idMother, -1)
      || !validMomentum( sister.p(),  idSister,  1)
      || !validMomentum( sister1.p(), idMother,  1)
      || !validMomentum( newRecoiler.p(), event[iNewRecoiler].id(), -1))
      physical = false;

    // Check invariants
    if ( false ) {
      Vec4 pk(event[iRecoiler].p());
      pj.p(sister.p());
      pi.p(sister1.p());
      double saix(2.*pa*pi), sakx(2.*pa*pk), sajx(2.*pa*pj), sikx(2.*pi*pk),
             sjkx(2.*pj*pk), sijx(2.*pi*pj);
      double pptt = (sajx-sijx)*(sakx-sikx)/(sakx);
      double ssaaii = saix;
      double zzaa = 2.*event[iDaughter].p()*event[iRecoiler].p()/ ( sakx  );
      double xxaa = (sakx-sikx) / ( sakx );
      if ( physical &&
           (abs(pptt-pT2)/abs(pT2) > 1e-3 || abs(ssaaii-sai)/abs(sai) > 1e-3 ||
            abs(zzaa-za)/abs(za)   > 1e-3 || abs(xxaa-xa)/abs(xa) > 1e-3 )) {
        cout << scientific << setprecision(8);
        cout << "Error in branch_II: Invariant masses after branching do not "
             << "match chosen values." << endl;
        cout << "Chosen:    "
             << " Q2 " << (event[iDaughter].p()+event[iRecoiler].p()).m2Calc()
             << " pT2 " << pT2
             << " sai " << sai
             << " za " << z
             << " xa " << xa << endl;
        cout << "Generated: "
             << " Q2 " << sakx-saix-sajx+sijx-sikx-sjkx+m2a+m2i+m2j+m2k
             << " pT2 " << pptt
             << " sai " << ssaaii
             << " za " << zzaa
             << " xa " << xxaa << endl;
        physical = false;
      }
    }

    // Check that total pT vanishes.
    for (int i = 0; i < event.size(); ++i)
      if ( event[i].isFinal()
        && partonSystemsPtr->getSystemOf(i,true) == iSysSelNow
        && find(iPos.begin(), iPos.end(), i) == iPos.end() )
        pSum -= event[i].p();
    if (abs(pSum.px()) > mTolErr || abs(pSum.py()) > mTolErr )
      physical = false;

    // Test that enough beam momentum remains.
    double xAnew = (mother.mother1() == 1)
              ? 2.*mother1.e()     / (beamAPtr->e() + beamBPtr->e())
              : 2.*newRecoiler.e() / (beamAPtr->e() + beamBPtr->e());
    double iAold = (mother.mother1() == 1) ? iDaughter : iRecoiler;
    double iAnew = (mother.mother1() == 1) ? iMother1 : iNewRecoiler;
    double xBnew = (mother.mother1() == 1)
              ? 2.*newRecoiler.e() / (beamAPtr->e() + beamBPtr->e())
              : 2.*mother1.e()     / (beamAPtr->e() + beamBPtr->e());
    double iBold = (mother.mother1() == 1) ? iRecoiler : iDaughter;
    double iBnew = (mother.mother1() == 1) ? iNewRecoiler : iMother1;
    if ( hasPDF(event[iAnew].id()) && beamAPtr->size() > 0) {
      double xOld = (*beamAPtr)[iSysSelNow].x();
      (*beamAPtr)[iSysSelNow].iPos(iAnew);
      (*beamAPtr)[iSysSelNow].x(xAnew);
      if (beamAPtr->xMax(-1) < 0.0) {
        if (!trial) infoPtr->errorMsg("Warning in DireSpace::branch_II: "
          "used up beam momentum; discard splitting.");
        physical = false;
      }
      // Restore old beams.
      (*beamAPtr)[iSysSelNow].iPos(iAold);
      (*beamAPtr)[iSysSelNow].x(xOld);
    }
    if ( hasPDF(event[iBnew].id()) && beamBPtr->size() > 0) {
      double xOld = (*beamBPtr)[iSysSelNow].x();
      (*beamBPtr)[iSysSelNow].iPos(iBnew);
      (*beamBPtr)[iSysSelNow].x(xBnew);
      if (beamBPtr->xMax(-1) < 0.0) {
        if (!trial) infoPtr->errorMsg("Warning in DireSpace::branch_II: "
          "used up beam momentum; discard splitting.");
        physical = false;
      }
      // Restore old beams.
      (*beamBPtr)[iSysSelNow].iPos(iBold);
      (*beamBPtr)[iSysSelNow].x(xOld);
    }

    // Update dipoles and beams. Note: dipEndSel no longer valid after this.
    double xm = 2.*mother1.e() / (beamAPtr->e() + beamBPtr->e());
    if (physical && !trial) updateAfterII( iSysSelNow, side, iDipSel,
      eventSizeOld, systemSizeOld, event, iDaughter, iMother, iSister,
      iNewRecoiler, pT2, xm);

    // Update flavours, colours, status.
    mother1.id( idMother);
    mother1.status(-41);
    mother1.cols(colMother1, acolMother1);
    mother1.daughters( iSister1, iMother);
    mother1.mothers( mother.mother1(), mother.mother2());
    mother.mothers( iMother1, 0);
    sister1.id(idMother);
    sister1.status(43);
    sister1.mothers(iMother1,0);
    sister1.cols(colSister1, acolSister1);
    sister1.scale(sqrt(pT2));

    if (iSysSelNow == 0) {
      event[beamOff1].daughter1( (side == 1) ? iMother1 : iNewRecoiler );
      event[beamOff2].daughter1( (side == 2) ? iMother1 : iNewRecoiler );
    }

    // Update dipoles and beams. Note: dipEndSel no longer valid after this.
    if (physical && !trial) updateAfterII( iSysSelNow, side, iDipSel, 0, 0,
      event, iMother, iMother1, iSister1, iNewRecoiler, pT2, xm);

  }

  physical = physical && !doVeto;

  // Temporarily set the daughters in the beams to zero, to
  // allow mother-daughter relation checks.
  if (iSysSelNow > 0) {
    if (side == 1) event[beamOff1].daughter1(0);
    if (side == 2) event[beamOff2].daughter1(0);
  }

  // Check if mother-daughter relations are correctly set. Check only
  // possible if no MPI are present.
  if ( physical && !trial && !doMECreject
    && !validMotherDaughter(event)) {
    if (printWarnings)
      infoPtr->errorMsg("Error in DireSpace::branch_II: Mother-daughter "
                        "relations after branching not valid.");
    physical = false;
  }

  // Restore correct daughters in the beams.
  if (iSysSelNow > 0) {
    if (side == 1) event[beamOff1].daughter1(iBeam1Dau1);
    if (side == 2) event[beamOff2].daughter1(iBeam2Dau1);
  }

  // Allow veto of branching. If so restore event record to before emission.
  if ( !physical || doMECreject) {
    event.popBack( event.size() - eventSizeOld);
    if (iSysSelNow == 0) {
      event[beamOff1].daughter1( ev1Dau1V);
      event[beamOff2].daughter1( ev2Dau1V);
    }
    for ( int iCopy = 0; iCopy < systemSizeOld; ++iCopy) {
      int iOldCopy = partonSystemsPtr->getAll(iSysSelNow, iCopy);
      event[iOldCopy].status( statusV[iCopy]);
      event[iOldCopy].mothers( mother1V[iCopy], mother2V[iCopy]);
      event[iOldCopy].daughters( daughter1V[iCopy], daughter2V[iCopy]);
    }

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
      //weights->calcWeight(pT2);
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
  if (trial) split->storePosAfter(
    (nEmissions < 2) ? iMother : iMother1, iNewRecoiler,
    iSister, (nEmissions < 2) ? 0 : iSister1);

  // Set shower weight.
  if (!trial) {

    if (!doTrialNow) {
      weights->calcWeight(pT2);
      weights->reset();
      // Store positions of new soft particles for (FSR) evolution.
      direInfoPtr->updateSoftPos( iDaughter, iSister );
      direInfoPtr->updateSoftPosIfMatch( iRecoiler, iNewRecoiler );
      if (nEmissions > 1) direInfoPtr->addSoftPos( iSister1 );
    }

    // Clear accept/reject weights.
    for ( unordered_map<string, multimap<double,double> >::iterator
      it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
      it->second.clear();
    for ( unordered_map<string, map<double,double> >::iterator
      it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
      it->second.clear();
  }

  // Done without any errors.
  return true;

}

//--------------------------------------------------------------------------

void DireSpace::updateAfterII( int iSysSelNow, int sideNow, int iDipSelNow,
  int eventSizeOldNow, int systemSizeOldNow, Event& event, int iDaughter,
  int iMother, int iSister, int iNewRecoiler, double pT2, double xNew) {

  // Update the number of proposed emissions.
  if (nProposedPT.find(iSysSelNow) != nProposedPT.end())
    ++nProposedPT[iSysSelNow];

  int idMother         = event[iMother].id();
  int idDaughterNow    = event[iDaughter].id();
  bool motherHasPlusPz = (event[iMother].pz() > 0.);

  // Bookkeep shower-induced resonances.
  if ( direInfoPtr->isRes(iDaughter) &&
    event[iMother].id() != event[iDaughter].id() )
    direInfoPtr->removeResPos(iDaughter);
  if ( particleDataPtr->isResonance(event[iMother].id()) ) {
    if ( direInfoPtr->isRes(iDaughter) )
          direInfoPtr->updateResPos(iDaughter,iMother);
  }
  if ( particleDataPtr->isResonance(event[iNewRecoiler].id()) )
    direInfoPtr->addResPos(iNewRecoiler);
  if ( particleDataPtr->isResonance(event[iSister].id()) )
    direInfoPtr->addResPos(iSister);

  // Update list of partons in system; adding newly produced one.
  if (motherHasPlusPz) {
    partonSystemsPtr->setInA(iSysSelNow, iMother);
    partonSystemsPtr->setInB(iSysSelNow, iNewRecoiler);
  } else {
    partonSystemsPtr->setInA(iSysSelNow, iNewRecoiler);
    partonSystemsPtr->setInB(iSysSelNow, iMother);
  }
  for (int iCopy = 2; iCopy < systemSizeOldNow; ++iCopy) {
    int iOut = partonSystemsPtr->getOut(iSysSelNow, iCopy - 2);
    // Remember shower-induced resonances.
    direInfoPtr->updateResPosIfMatch ( iOut, eventSizeOldNow + iCopy);
    direInfoPtr->updateSoftPosIfMatch( iOut, eventSizeOldNow + iCopy);
    partonSystemsPtr->setOut(iSysSelNow, iCopy - 2, eventSizeOldNow + iCopy);
  }
  partonSystemsPtr->addOut(iSysSelNow, iSister);

  // Get new center-of-mass energy
  int iA      = getInA(iSysSelNow);
  int iB      = getInB(iSysSelNow);
  double shat = (event[iA].p() + event[iB].p()).m2Calc();
  partonSystemsPtr->setSHat(iSysSelNow, shat);

  // dipEnd array may have expanded and been moved, so regenerate dipEndSel.
  dipEndSel = &dipEnd[iDipSelNow];

  // Update info on radiating dipole ends (QCD).
  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip)
  if ( dipEnd[iDip].system == iSysSelNow) {
    if (abs(dipEnd[iDip].side) == sideNow) {
      dipEnd[iDip].iRadiator = iMother;
      dipEnd[iDip].iRecoiler = iNewRecoiler;
      if (dipEnd[iDip].colType  != 0)
        dipEnd[iDip].colType = event[iMother].colType();

    // Update info on recoiling dipole ends (QCD or QED).
    } else {
      dipEnd[iDip].iRadiator = iNewRecoiler;
      dipEnd[iDip].iRecoiler = iMother;
      dipEnd[iDip].MEtype = 0;
    }
  }

  // Update info on beam remnants.
  BeamParticle& beamNow = (sideNow == 1) ? *beamAPtr : *beamBPtr;
  beamNow[iSysSelNow].update( iMother, idMother, xNew);
  // Redo choice of companion kind whenever new flavour.
  if (idMother != idDaughterNow) {
    pdfScale2 = (useFixedFacScale) ? fixedFacScale2 : factorMultFac * pT2;
    pdfScale2 = max(pdfScale2, pT2min);
    beamNow.xfISR( iSysSelNow, idMother, xNew, pdfScale2);
    beamNow.pickValSeaComp();
  }
  BeamParticle& beamRec = (sideNow == 1) ? *beamBPtr : *beamAPtr;
  beamRec[iSysSelNow].iPos( iNewRecoiler);

  // Update ISR dipoles.
  update(iSysSelNow,event);

  // Pointer to selected dipole no longer valid after update, thus unset.
  dipEndSel = 0;

  return;
}

//--------------------------------------------------------------------------

// Kinematics of branching.
// Construct mother -> daughter + sister, with recoiler on other side.

bool DireSpace::branch_IF( Event& event, bool trial,
  DireSplitInfo* split ) {

  Event auxevent1 = event;
  Event auxevent2 = event;

  // Side on which branching occured.
  int side          = (!trial) ? abs(dipEndSel->side) : split->side;

  // Read in flavour and colour variables.
  int iDaughter     = (!trial) ? dipEndSel->iRadiator  : split->iRadBef;
  int iRecoiler     = (!trial) ? dipEndSel->iRecoiler  : split->iRecBef;
  int idDaughterNow = (!trial) ? dipEndSel->idDaughter : split->radBef()->id;
  int idMother      = (!trial) ? dipEndSel->idMother   : split->radAft()->id;
  int idSister      = (!trial) ? dipEndSel->idSister   : split->emtAft()->id;
  int colDaughter   = event[iDaughter].col();
  int acolDaughter  = event[iDaughter].acol();
  int colRecBef     = event[iRecoiler].col();
  int acolRecBef    = event[iRecoiler].acol();
  bool colMatch     = (colRecBef  == colDaughter);
  bool acolMatch    = (acolRecBef == acolDaughter);
  int iSysSelNow    = (!trial) ? iSysSel : 0;
  //int iOldOther     = (side==1) ? partonSystemsPtr->getInB(iSysSelNow)
  //                              : partonSystemsPtr->getInA(iSysSelNow);
  int iOldOther     = (side==1) ? getInB(iSysSelNow) : getInA(iSysSelNow);
  string name       = (!trial) ? splittingSelName : split->splittingSelName;
  int nEmissions    = splits[name]->nEmissions();
  if ( nEmissions == 2) idSister = -event[iDaughter].id();

  // Read in kinematical variables.
  double pT2        = (!trial) ? dipEndSel->pT2 : split->kinematics()->pT2;
  double z          = (!trial) ? dipEndSel->z   : split->kinematics()->z;
  splits[name]->splitInfo.store(*split);
  unordered_map<string,double> psp(splits[name]->getPhasespaceVars
                                   (event, partonSystemsPtr));
  // Allow splitting kernel to overwrite phase space variables.
  if (split->useForBranching) { pT2 = psp["pT2"]; z = psp["z"]; }

  // Get particle masses.
  double m2Bef = 0.0;
  double m2r   = 0.0;
  double m2s   = 0.0;
  int type     = (event[iRecoiler].isFinal()) ? 1 : -1;
  if (type == 1) {
    m2s = particleDataPtr->isResonance(event[iRecoiler].id())
        ? getMass(event[iRecoiler].id(),3,
                  event[iRecoiler].mCalc())
        : (event[iRecoiler].idAbs() < 6)
        ? getMass(event[iRecoiler].id(),2)
        : getMass(event[iRecoiler].id(),1);
  }

  // Emission.
  double m2ex = (abs(idSister) < 6) ?
    getMass(idSister, 2) : getMass(idSister, 1);
  double m2e  = (!trial) ? m2ex
    : ( (split->kinematics()->m2EmtAft > 0.) ? split->kinematics()->m2EmtAft
                                            : m2ex);

  // Radiator mass.
  if ( useMassiveBeams && (abs(idDaughter) == 11 || abs(idDaughter) == 13
                           || abs(idDaughter) > 900000))
    m2Bef = getMass(idDaughter,1);
  if ( useMassiveBeams && (abs(idMother) == 11 || abs(idMother) == 13
                           || abs(idMother) > 900000))
    m2r   = getMass(idMother,1);
  // Emission mass
  if ( useMassiveBeams && (abs(idSister) == 11 || abs(idSister) == 13
                           || abs(idSister) > 900000))
    m2e   = getMass(idSister,1);

  // Force emission massless by default.
  if (!forceMassiveMap) m2e = 0.0;
  // Adjust the dipole kinematical mass to accomodate masses after branching.
  double m2dip      = (!trial) ? dipEndSel->m2Dip : split->kinematics()->m2Dip;
  double m2dipCorr  = m2dip - m2Bef + m2r + m2e;
  // Calculate CS variables.
  double xCS = z;
  double uCS = (pT2/m2dipCorr)/(1-z);
  double sai = (!trial) ? dipEndSel->sa1 : split->kinematics()->sai;
  double xa  = (!trial) ? dipEndSel->xa  : split->kinematics()->xa;
  // Allow splitting kernel to overwrite phase space variables.
  if (split->useForBranching) { sai = psp["sai"]; xa = psp["xa"]; }

  // Current event and subsystem size.
  int eventSizeOld  = event.size();
  int systemSizeOld = partonSystemsPtr->sizeAll(iSysSelNow);

  // Save properties to be restored in case of user-hook veto of emission.
  int beamOff1 = 1 + beamOffset;
  int beamOff2 = 2 + beamOffset;
  int ev1Dau1V = event[beamOff1].daughter1();
  int ev2Dau1V = event[beamOff2].daughter1();
  vector<int> statusV, mother1V, mother2V, daughter1V, daughter2V;

  // Check if the first emission shoild be checked for removal
  bool physical      = true;
  bool canMergeFirst = (mergingHooksPtr != 0)
                     ? mergingHooksPtr->canVetoEmission() : false;

  // Keep track of the system's full final for global recoil.
  if (useGlobalMapIF) {
    for ( int iCopy = 0; iCopy < systemSizeOld; ++iCopy) {
      int iOldCopy    = partonSystemsPtr->getAll(iSysSelNow, iCopy);
      statusV.push_back( event[iOldCopy].status());
      mother1V.push_back( event[iOldCopy].mother1());
      mother2V.push_back( event[iOldCopy].mother2());
      daughter1V.push_back( event[iOldCopy].daughter1());
      daughter2V.push_back( event[iOldCopy].daughter2());
    }
  }

  // Less book-keeping for local recoil scheme.
  int iDauStatusV  = event[iDaughter].status();
  int iDauMot1V    = event[iDaughter].mother1();
  int iDauMot2V    = event[iDaughter].mother2();
  int iDauDau1V    = event[iDaughter].daughter1();
  int iDauDau2V    = event[iDaughter].daughter2();
  int iRecStatusV  = event[iRecoiler].status();
  int iRecMot1V    = event[iRecoiler].mother1();
  int iRecMot2V    = event[iRecoiler].mother2();
  int iRecDau1V    = event[iRecoiler].daughter1();
  int iRecDau2V    = event[iRecoiler].daughter2();

  // For global recoil, take copy of existing system, to be given modified
  // kinematics. Incoming negative status.
  int iMother(0), iNewRecoiler(0), iNewOther(0);
  if (useGlobalMapIF) {
    for ( int iCopy = 0; iCopy < systemSizeOld; ++iCopy) {
      int iOldCopy    = partonSystemsPtr->getAll(iSysSelNow, iCopy);
      int statusOld   = event[iOldCopy].status();
      int statusNew   = (iOldCopy == iDaughter
                      || iOldCopy == iOldOther) ? statusOld :
                        (iOldCopy == iRecoiler) ? 48 : 44;
      int iNewCopy    = event.copy(iOldCopy, statusNew);
      if (iOldCopy == iDaughter) iMother      = iNewCopy;
      if (iOldCopy == iRecoiler) iNewRecoiler = iNewCopy;
      if (iOldCopy == iOldOther) iNewOther    = iNewCopy;
      if (statusOld < 0) event[iNewCopy].statusNeg();
    }
  }

  // For 1->3 splitting, intermediate mother is a gluon.
  int idMotherNow = idMother;
  if ( nEmissions == 2) idMotherNow = 21;

  // Define colour flow in branching.
  // Default corresponds to f -> f + gamma.
  int colMother     = colDaughter;
  int acolMother    = acolDaughter;
  int colSister     = 0;
  int acolSister    = 0;

  if (idSister == 22 || idSister == 23 || idSister == 25) ;

  // q -> q + g and 50% of g -> g + g; need new colour.
  else if (idSister == 21 && ( (idMotherNow > 0 && idMotherNow < 9)
    || ( idMotherNow == 21 && colMatch && acolMatch
      && rndmPtr->flat() < 0.5) ) ) {
    colMother       = event.nextColTag();
    colSister       = colMother;
    acolSister      = colDaughter;
  // qbar -> qbar + g and other 50% of g -> g + g; need new colour.
  } else if (idSister == 21 && ( (idMotherNow < 0 && idMotherNow > -9)
    || ( idMotherNow == 21 && colMatch && acolMatch) ) ) {
    acolMother      = event.nextColTag();
    acolSister      = acolMother;
    colSister       = acolDaughter;
  } else if (idMotherNow == 21 && idSister == 21 && colMatch && !acolMatch) {
    colMother       = event.nextColTag();
    acolMother      = acolDaughter;
    colSister       = colMother;
    acolSister      = colDaughter;
  } else if (idMotherNow == 21 && idSister == 21 && !colMatch && acolMatch) {
    colMother       = colDaughter;
    acolMother      = event.nextColTag();
    acolSister      = acolMother;
    colSister       = acolDaughter;
  // q -> g + q.
  } else if (idDaughterNow == 21 && idMotherNow > 0) {
    colMother       = colDaughter;
    acolMother      = 0;
    colSister       = acolDaughter;
  // qbar -> g + qbar
  } else if (idDaughterNow == 21) {
    acolMother      = acolDaughter;
    colMother       = 0;
    acolSister      = colDaughter;
  // g -> q + qbar.
  } else if (idDaughterNow > 0 && idDaughterNow < 9) {
    acolMother      = event.nextColTag();
    acolSister      = acolMother;
  // g -> qbar + q.
  } else if (idDaughterNow < 0 && idDaughterNow > -9) {
    colMother       = event.nextColTag();
    colSister       = colMother;
  }

  // Swap colours if radiated gluon carries momentum fraction z
  int colSave = colSister, acolSave = acolSister;
  if ( splits[name]->is(splittingsPtr->isrQCD_21_to_21_and_21b)) {
    colSister  = acolMother;
    acolSister = colMother;
    colMother  = acolSave;
    acolMother = colSave;
  }

  if (split->useForBranching) {
    idMotherNow= event[iDaughter].id();
    colMother  = event[iDaughter].col();
    acolMother = event[iDaughter].acol();
    colSister  = 0;
    acolSister = 0;
    // Now reset if splitting information is available.
    if (split->radAft()->id   != 0) idMotherNow= split->radAft()->id;
    if (split->emtAft()->id   != 0) idSister   = split->emtAft()->id;
    if (split->radAft()->col  > -1) colMother  = split->radAft()->col;
    if (split->radAft()->acol > -1) acolMother = split->radAft()->acol;
    if (split->emtAft()->col  > -1) colSister  = split->emtAft()->col;
    if (split->emtAft()->acol > -1) acolSister = split->emtAft()->acol;
  }

  int colMother1, acolMother1;
  int colSister1, acolSister1;
  colMother1 = acolMother1 = colSister1 = acolSister1 = 0;
  if ( nEmissions == 2) {
    // Daughter color transferred to quark mother "1", sister anti-color
    // transferred to sister "1" color.
    if (idMother*idDaughterNow > 0 && idMother > 0) {
      colMother1  = colDaughter;
      colSister1  = acolSister;
    }
    // Daughter anticolor transferred to antiquark mother "1", sister color
    // transferred to sister "1" anticolor.
    if (idMother*idDaughterNow > 0 && idMother < 0) {
      colMother1  = 0;
      acolMother1 = acolDaughter;
      acolSister1 = colSister;
      colSister1  = 0;
    }
    // Sister color transferred to quark mother "1", daughter anti-color
    // transferred to sister "1" color.
    if (idMother*idDaughterNow < 0 && idMother > 0) {
      colMother1  = colSister;
      acolMother1 = 0;
      acolSister1 = 0;
      colSister1  = acolDaughter;
      // Reset dummy mother colours.
      acolMother = acolDaughter;
      colMother  = colSister;
    }
    // Sister anticolor transferred to antiquark mother "1", daughter color
    // transferred to sister "1" anti-color.
    if (idMother*idDaughterNow < 0 && idMother < 0) {
      colMother1  = 0;
      acolMother1 = acolSister;
      acolSister1 = colDaughter;
      colSister1  = 0;
      // Reset dummy mother colours.
      acolMother = acolSister;
      colMother  = colDaughter;
    }
  }

  // Add mother. For 1->3 splitting, attach "dummy" mother.
  if (!useGlobalMapIF) {iMother = event.append( idMotherNow, -41,
    event[iDaughter].mother1(), 0, 0, 0, colMother, acolMother,
    Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );
    auxevent1.append( idMotherNow, -41,
    event[iDaughter].mother1(), 0, 0, 0, colMother, acolMother,
    Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );
    auxevent2.append( idMotherNow, -41,
    event[iDaughter].mother1(), 0, 0, 0, colMother, acolMother,
    Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );
  }
  // Add recoiler. For 1->3 splitting, attach "dummy" recoiler.
  if (!useGlobalMapIF) { iNewRecoiler =
    event.append( event[iRecoiler].id(), 48,
    iRecoiler, iRecoiler, 0, 0, event[iRecoiler].col(),
    event[iRecoiler].acol(), Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );
    auxevent1.append( event[iRecoiler].id(), 48,
    iRecoiler, iRecoiler, 0, 0, event[iRecoiler].col(),
    event[iRecoiler].acol(), Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );
    auxevent2.append( event[iRecoiler].id(), 48,
    iRecoiler, iRecoiler, 0, 0, event[iRecoiler].col(),
    event[iRecoiler].acol(), Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );
  }
  // Remember other initial state.
  if (!useGlobalMapIF) iNewOther = iOldOther;
  // For 1->3 splitting, add "real" mother.
  int iMother1  = 0;
  if ( nEmissions == 2)
    iMother1 = event.append( 0, 0, 0, 0, 0, 0, 0, 0, Vec4(0.,0.,0.,0.), 0.0,
    sqrt(pT2) );
  // Add new sister.
  int iSister       = event.append( idSister, 43, iMother, 0, 0, 0,
    colSister, acolSister, Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );
  auxevent1.append( idSister, 43, iMother, 0, 0, 0,
    colSister, acolSister, Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );
  auxevent2.append( idSister, 43, iMother, 0, 0, 0,
    colSister, acolSister, Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );
  // For 1->3 splitting, add "real" recoiler.
  int iNewRecoiler1  = 0;
  if ( nEmissions == 2)
    iNewRecoiler1 = event.append( event[iRecoiler].id(), 48, iNewRecoiler,
    iNewRecoiler, 0, 0, event[iRecoiler].col(), event[iRecoiler].acol(),
    Vec4(0.,0.,0.,0.), 0.0, sqrt(pT2) );
  // Second sister particle for 1->3 splitting.
  int iSister1      = 0;
  if ( nEmissions == 2)
    iSister1 = event.append( 0, 0, 0, 0, 0, 0, 0, 0, Vec4(0.,0.,0.,0.), 0.0,
    sqrt(pT2) );

  // References to the partons involved.
  Particle& daughter     = event[iDaughter];
  Particle& mother       = event[iMother];
  Particle& newRecoiler  = event[iNewRecoiler];
  Particle& sister       = event[iSister];
  Particle& mother1      = event[iMother1];
  Particle& newRecoiler1 = event[iNewRecoiler1];
  Particle& sister1      = event[iSister1];

  // Replace old by new mother; update old recoiler.
  event[iRecoiler].statusNeg();
  event[iRecoiler].daughters( iNewRecoiler, iNewRecoiler);
  auxevent1[iRecoiler].statusNeg();
  auxevent2[iRecoiler].statusNeg();
  auxevent1[iRecoiler].daughters( iNewRecoiler, iNewRecoiler);
  auxevent2[iRecoiler].daughters( iNewRecoiler, iNewRecoiler);
  if (mother.idAbs() == 21 || mother.idAbs() == 22) mother.pol(9);

  // Update mother and daughter pointers; also for beams.
  daughter.mothers( iMother, 0);
  mother.daughters( iSister, iDaughter);
  mother.cols( colMother, acolMother);
  mother.id( idMotherNow );
  int iBeam1Dau1 = event[beamOff1].daughter1();
  int iBeam2Dau1 = event[beamOff2].daughter1();
  if (iSysSelNow == 0) {
    event[beamOff1].daughter1( (side == 1) ? iMother : iNewOther );
    event[beamOff2].daughter1( (side == 2) ? iMother : iNewOther );
  }

  auxevent1[iDaughter].mothers( iMother, 0);
  auxevent1[iMother].daughters( iSister, iDaughter);
  auxevent1[iMother].cols( colMother, acolMother);
  auxevent1[iMother].id( idMotherNow );
  if (iSysSelNow == 0) {
    auxevent1[beamOff1].daughter1( (side == 1) ? iMother : iNewOther );
    auxevent1[beamOff2].daughter1( (side == 2) ? iMother : iNewOther );
  }

  auxevent2[iDaughter].mothers( iMother, 0);
  auxevent2[iMother].daughters( iSister, iDaughter);
  auxevent2[iMother].cols( colMother, acolMother);
  auxevent2[iMother].id( idMotherNow );
  if (iSysSelNow == 0) {
    auxevent2[beamOff1].daughter1( (side == 1) ? iMother : iNewOther );
    auxevent2[beamOff2].daughter1( (side == 2) ? iMother : iNewOther );
  }

  bool doVeto = false;
  bool printWarnings = (!trial || !forceMassiveMap);
  bool doMECreject = false;

  // Regular massive kinematics for 1+1 -> 2+1 splitting
  if ( nEmissions != 2) {

    // Massive kinematics, in two schemes.

    // Local scheme.
    if (!useGlobalMapIF) {

      // Get dipole 4-momentum.
      Vec4 paj_tilde(event[iDaughter].p());
      Vec4 pk_tilde(event[iRecoiler].p());
      Vec4 pTk_tilde(event[iRecoiler].px(),event[iRecoiler].py(),0.,0.);
      Vec4 q(paj_tilde-pk_tilde);
      Vec4 qpar(q+pTk_tilde);

      // Calculate derived variables.
      double q2    = q.m2Calc();
      double q2par = qpar.m2Calc();
      double pT2k  = -pTk_tilde.m2Calc();
      double sjk   = (1. - 1./xCS)*(q2 - m2r) + (m2e + m2s) / xCS;
      double zbar  = (q2-sjk-m2r) / bABC(q2,sjk,m2r)
                    *( uCS - m2r / gABC(q2,sjk,m2r)
                          * (sjk + m2e - m2s) / (q2 - sjk - m2r));
      double kT2   = zbar*(1.-zbar)*sjk - (1-zbar)*m2e - zbar*m2s;

      if (kT2 < 0.) {
        if (printWarnings)
          infoPtr->errorMsg("Warning in DireSpace::branch_IF: Reject state "
                            "with kinematically forbidden kT^2.");
        physical = false;
      }

      // Now construct radiator in lab frame.
      Vec4 pRad( ( paj_tilde - q*paj_tilde/q2par * qpar )
                 * sqrt( (lABC(q2,sjk,m2r) - 4.*m2r*pT2k)
                       / (lABC(q2,m2s,m2Bef) - 4.*m2Bef*pT2k))
                + qpar * 0.5 * (q2 + m2r - sjk) / q2par);

      // Construct dummy overall momentum.
      Vec4 pSum(1e5,1e5,1e5,1e5);
      Vec4 pSumIn( beamAPtr->p() + beamBPtr->p() );

      // Now produce momenta of emitted and recoiling particles, and ensure
      // good momentum conservation. (More than one try only necessary in rare
      // numerical instabilities.
      Vec4 pRec, pEmt;
      Vec4 auxpRec1, auxpEmt1;
      Vec4 auxpRec2, auxpEmt2;
      int nTries = 0;
      while ( abs(pSum.px()-pSumIn.px()) > mTolErr
           || abs(pSum.py()-pSumIn.py()) > mTolErr ) {

        // Give up after too many tries.
        nTries++;
        if (nTries > 100) {
          if (printWarnings)
            infoPtr->errorMsg
              ("Warning in DireSpace::branch_IF: Could not set up"
                              " state after branching, thus reject.");
          physical = false; break;
        }

        // Now construct the transverse momentum vector in the dipole CM frame.
        double phi_kt = (!trial) ? 2.*M_PI*rndmPtr->flat()
          : split->kinematics()->phi < 0.  ?
          2.*M_PI*rndmPtr->flat() : split->kinematics()->phi;

        double phi_kt1 = phi_kt+DPHI_IF*M_PI;
        if (phi_kt1>2.*M_PI) phi_kt1 -= 2.*M_PI;
        double phi_kt2 = phi_kt-DPHI_IF*M_PI;
        if (phi_kt2<0.)      phi_kt2 += 2.*M_PI;
        if (phi_kt1<phi_kt2) swap(phi_kt1, phi_kt2);

        // Allow splitting kernel to overwrite phase space variables.
        if (split->useForBranching) { phi_kt = psp["phi"]; }

        // Construct left-over dipole momentum by momentum conservation.
        Vec4 pjk(-q+pRad);

        // Set up transverse momentum vector by using two
        // perpendicular four-vectors.
        pair<Vec4, Vec4> pTvecs = getTwoPerpendicular(pRad,pjk);
        Vec4 kTmom( sqrt(kT2)*sin(phi_kt)*pTvecs.first
                  + sqrt(kT2)*cos(phi_kt)*pTvecs.second);

        // Construct new emission momentum.
        pEmt.p( zbar * (gABC(q2,sjk,m2r)*pjk + sjk*pRad) / bABC(q2,sjk,m2r)
                  - (m2e + kT2) / (zbar*bABC(q2,sjk,m2r))
                  * (pRad + m2r/gABC(q2,sjk,m2r)*pjk)
                  + kTmom);

        // Construct recoiler momentum by momentum conservation.
        pRec.p(-q+pRad-pEmt);

        kTmom.p( sqrt(kT2)*sin(phi_kt1)*pTvecs.first
                  + sqrt(kT2)*cos(phi_kt1)*pTvecs.second);
        auxpEmt1.p( zbar * (gABC(q2,sjk,m2r)*pjk + sjk*pRad) / bABC(q2,sjk,m2r)
                  - (m2e + kT2) / (zbar*bABC(q2,sjk,m2r))
                  * (pRad + m2r/gABC(q2,sjk,m2r)*pjk)
                  + kTmom);
        auxpRec1.p(-q+pRad-auxpEmt1);

        kTmom.p( sqrt(kT2)*sin(phi_kt2)*pTvecs.first
                  + sqrt(kT2)*cos(phi_kt2)*pTvecs.second);
        auxpEmt2.p( zbar * (gABC(q2,sjk,m2r)*pjk + sjk*pRad) / bABC(q2,sjk,m2r)
                  - (m2e + kT2) / (zbar*bABC(q2,sjk,m2r))
                  * (pRad + m2r/gABC(q2,sjk,m2r)*pjk)
                  + kTmom);
        auxpRec2.p(-q+pRad-auxpEmt2);

        sister.p(pEmt);
        mother.p(pRad);
        newRecoiler.p(pRec);

        auxevent1[iSister].p(auxpEmt1);
        auxevent1[iMother].p(pRad);
        auxevent1[iNewRecoiler].p(auxpRec1);

        auxevent2[iSister].p(auxpEmt2);
        auxevent2[iMother].p(pRad);
        auxevent2[iNewRecoiler].p(auxpRec2);

        // Gather total momentum for subsequent check that total pT vanishes.
        vector<int> iPos(1,iSister);
        pSum  = sister.p();
        iPos.push_back(iNewRecoiler);
        pSum += newRecoiler.p();

        // Collect remaining final state momenta.
        for (int i = 0; i < event.size(); ++i)
          if ( event[i].isFinal()
            && partonSystemsPtr->getSystemOf(i,true) == iSysSelNow
            && find(iPos.begin(), iPos.end(), i) == iPos.end() )
            pSum += event[i].p();
      }

    // Global scheme.
    } else {

      // Get dipole 4-momentum.
      Vec4 paj_tilde(event[iDaughter].p());
      Vec4 pk_tilde(event[iRecoiler].p());
      Vec4 q(pk_tilde-paj_tilde);

      // Calculate derived variables.
      double q2 = q.m2Calc();
      //double saj = uCS/xCS*(q2 - m2s) + (m2r+m2e) * (1-uCS)/xCS;
      double saj = uCS/xCS*(q2 - m2s) + (m2r+m2e) * (1-uCS/xCS);
      double zbar = (q2-saj-m2s) / bABC(q2,saj,m2s)
                  *( (xCS - 1)/(xCS-uCS)  - m2s / gABC(q2,saj,m2s)
                         * (saj + m2e - m2r) / (q2 - saj - m2s));
      double kT2  = zbar*(1.-zbar)*saj - (1-zbar)*m2e - zbar*m2r;

      // Disallow kinematically impossible transverse momentum.
      if (kT2 < 0. || isnan(kT2)) physical = false;

      // Now construct recoiler in lab frame.
      Vec4 pRec( (pk_tilde - q*pk_tilde/q2*q)
                 *sqrt(lABC(q2,saj,m2s)/lABC(q2,m2Bef,m2s))
               + 0.5*(q2+m2s-saj)/q2*q );

      // Construct left-over dipole momentum by momentum conservation.
      Vec4 paj(-q+pRec);

      double phi_kt = (!trial) ? 2.*M_PI*rndmPtr->flat()
        : split->kinematics()->phi < 0.  ?
        2.*M_PI*rndmPtr->flat() : split->kinematics()->phi;

      // Allow splitting kernel to overwrite phase space variables.
      if (split->useForBranching) { phi_kt = psp["phi"]; }

      // Set up transverse momentum vector by using two perpendicular
      // four-vectors.
      pair<Vec4, Vec4> pTvecs = getTwoPerpendicular(paj, pRec);
      Vec4 kTmom( sqrt(kT2)*sin(phi_kt)*pTvecs.first
                + sqrt(kT2)*cos(phi_kt)*pTvecs.second);

      // Construct new emission momentum.
      Vec4 pEmt( - zbar * (gABC(q2,saj,m2s)*paj + saj*pRec) / bABC(q2,saj,m2s)
                + (m2e + kT2) / (zbar*bABC(q2,saj,m2s))
                * (pRec + m2s/gABC(q2,saj,m2s)*paj)
                + kTmom);

      // Contruct radiator momentum by momentum conservation.
      Vec4 pRad(-q+pRec+pEmt);

      // Boost to realign the incoming radiator.
      int iOther  = getInB(iSysSelNow);
      if (side == 2) iOther = getInA(iSysSelNow);

      Vec4 pOther(event[iOther].p());

      // Boost to rest frame of incoming particles A and B.
      RotBstMatrix toABCM;
      if (side == 1) toABCM.toCMframe( pRad, pOther);
      else           toABCM.toCMframe( pOther, pRad);

      // After this, the radiator has vanishing pT.
      pRad.rotbst(toABCM);
      pOther.rotbst(toABCM);

      // Restore the momentum fraction of the incoming particle that
      // did not participate in the splitting.
      RotBstMatrix restoreB;
      restoreB.bst( pOther, event[iOther].p());

      // After this, the inactive beam returns to the correct energy fraction.
      pRad.rotbst(restoreB);
      pOther.rotbst(restoreB);

      // Set all momenta.
      sister.p(pEmt);
      mother.p(pRad);
      newRecoiler.p(pRec);

      // Rotate and boost all final state particles to absorb the pT of the
      // radiator.
      for ( int i = eventSizeOld + 2; i < eventSizeOld + systemSizeOld; ++i) {
        if ( event[i].isFinal()) {
          event[i].rotbst(toABCM);
          event[i].rotbst(restoreB);
        }
      }

      // Transform the emission to the new lab frame.
      sister.rotbst(toABCM);
      sister.rotbst(restoreB);

    }

    // Store masses.
    sister.m(sqrtpos(m2e));
    mother.m(sqrtpos(m2r));
    newRecoiler.m(sqrtpos(m2s));

    auxevent1[iSister].m(sqrtpos(m2e));
    auxevent1[iMother].m(sqrtpos(m2r));
    auxevent1[iNewRecoiler].m(sqrtpos(m2s));

    auxevent2[iSister].m(sqrtpos(m2e));
    auxevent2[iMother].m(sqrtpos(m2r));
    auxevent2[iNewRecoiler].m(sqrtpos(m2s));

    // Check momenta.
    if ( !validMomentum( mother.p(), idMother, -1)
      || !validMomentum( sister.p(), idSister,  1)
      || !validMomentum( newRecoiler.p(), event[iNewRecoiler].id(), 1))
      physical = false;

      // Rotate and boost all final state particles to absorb the pT of the
    doVeto = (( canVetoEmission && userHooksPtr->doVetoISREmission(
                eventSizeOld, event, iSysSelNow))
           || ( canMergeFirst   && mergingHooksPtr->doVetoEmission(event)) );

    double xm = 2.*mother.e() / (beamAPtr->e() + beamBPtr->e());

    // Test that enough beam momentum remains.
    int iOther  = getInB(iSysSelNow);
    if (side == 2) iOther = getInA(iSysSelNow);
    double xAnew = (mother.mother1() == 1)
              ? 2.*mother.e()        / (beamAPtr->e() + beamBPtr->e())
              : 2.*event[iOther].e() / (beamAPtr->e() + beamBPtr->e());
    double iAold = (mother.mother1() == 1) ? iDaughter : iOther;
    double iAnew = (mother.mother1() == 1) ? iMother : iOther;
    double xBnew = (mother.mother1() == 1)
              ? 2.*event[iOther].e() / (beamAPtr->e() + beamBPtr->e())
              : 2.*mother.e()        / (beamAPtr->e() + beamBPtr->e());
    double iBold = (mother.mother1() == 1) ? iOther : iDaughter;
    double iBnew = (mother.mother1() == 1) ? iNewRecoiler : iOther;
    if ( hasPDF(event[iAnew].id()) && beamAPtr->size() > 0) {
      double xOld = (*beamAPtr)[iSysSelNow].x();
      (*beamAPtr)[iSysSelNow].iPos(iAnew);
      (*beamAPtr)[iSysSelNow].x(xAnew);
      if (beamAPtr->xMax(-1) < 0.0) {
        if (!trial) infoPtr->errorMsg("Warning in DireSpace::branch_IF: "
          "used up beam momentum; discard splitting.");
        physical = false;
      }
      // Restore old beams.
      (*beamAPtr)[iSysSelNow].iPos(iAold);
      (*beamAPtr)[iSysSelNow].x(xOld);
    }
    if ( hasPDF(event[iBnew].id()) && beamBPtr->size() > 0) {
      double xOld = (*beamBPtr)[iSysSelNow].x();
      (*beamBPtr)[iSysSelNow].iPos(iBnew);
      (*beamBPtr)[iSysSelNow].x(xBnew);
      if (beamBPtr->xMax(-1) < 0.0) {
        if (!trial) infoPtr->errorMsg("Warning in DireSpace::branch_IF: "
          "used up beam momentum; discard splitting.");
        physical = false;
      }
      // Restore old beams.
      (*beamBPtr)[iSysSelNow].iPos(iBold);
      (*beamBPtr)[iSysSelNow].x(xOld);
    }

    // Apply ME correction if necessary.
    bool isHardSystem = partonSystemsPtr->getSystemOf(iDaughter,true) == 0
                     && partonSystemsPtr->getSystemOf(iRecoiler,true) == 0;
    if (isHardSystem && physical && doMEcorrections
      && pT2 > pT2minMECs && checkSIJ(event,Q2minMECs)) {

      int iA      = getInA(iSysSelNow);
      int iB      = getInB(iSysSelNow);
      // Update and add newly produced particles.
      vector<int> iOut(createvector<int>(0)(0));
      if (useGlobalMapIF) {
        // Add newly produced particle.
        for (int iCopy = 2; iCopy < systemSizeOld; ++iCopy)
          iOut.push_back(partonSystemsPtr->getOut(iSysSel, iCopy - 2));
      }

      bool motherHasPlusPz = (event[iMother].pz() > 0.);
      if (motherHasPlusPz) {
        partonSystemsPtr->setInA(iSysSelNow, iMother);
        partonSystemsPtr->setInB(iSysSelNow, iNewOther);
      } else {
        partonSystemsPtr->setInB(iSysSelNow, iMother);
        partonSystemsPtr->setInA(iSysSelNow, iNewOther);
      }

      // Update and add newly produced particles.
      if (useGlobalMapIF) {
        // Add newly produced particle.
        for (int iCopy = 2; iCopy < systemSizeOld; ++iCopy)
          partonSystemsPtr->setOut(iSysSel, iCopy - 2, eventSizeOld + iCopy);
        partonSystemsPtr->addOut(iSysSelNow, iSister);
        partonSystemsPtr->replace(iSysSelNow, iRecoiler, iNewRecoiler);
      } else {
        // Add newly produced particle.
        partonSystemsPtr->addOut(iSysSelNow, iSister);
        partonSystemsPtr->replace(iSysSelNow, iRecoiler, iNewRecoiler);
      }

      if ( nFinalMaxMECs < 0
        || nFinalMaxMECs > partonSystemsPtr->sizeOut(iSysSelNow))
        doMECreject = applyMEC (event, split,
          createvector<Event>(auxevent1)(auxevent2));

      partonSystemsPtr->setInA(iSysSelNow, iA);
      partonSystemsPtr->setInB(iSysSelNow, iB);
      if (useGlobalMapIF) {
        for (int iCopy = 2; iCopy < systemSizeOld; ++iCopy)
          partonSystemsPtr->setOut(iSysSelNow, iCopy - 2, iOut[iCopy]);
        partonSystemsPtr->replace(iSysSelNow, iNewRecoiler, iRecoiler);
        partonSystemsPtr->popBackOut(iSysSelNow);
      } else {
        partonSystemsPtr->replace(iSysSelNow, iNewRecoiler, iRecoiler);
        partonSystemsPtr->popBackOut(iSysSelNow);
      }
    }

    // Update dipoles and beams. Note: dipEndSel no longer valid after this.
    if (physical && !doVeto && !trial && !doMECreject) updateAfterIF(
      iSysSelNow, side,
      iDipSel, eventSizeOld, systemSizeOld, event, iDaughter, iRecoiler,
      iMother, iSister, iNewRecoiler, iNewOther, pT2, xm);

  // Perform 1+1 -> 3 + 1 splitting.

  } else {

    // Perform 1->3 splitting as two consecutive steps.

    // Save momenta before the splitting.
    Vec4 pa12_tilde(event[iDaughter].p());
    Vec4 pb_tilde(event[iRecoiler].p());

    double za     = z;

    // Massive kinematics, in two schemes.

    // Local scheme.
    if (!useGlobalMapIF) {

      // 1->3 splitting as one IF and one ("recoil-less") FF step.
      // (aij)_tilde (k)_tilde -> (a) (i) (jk) -> (a) (i) (j) (k)
      // Recoiler k shifts mass in first and second step.

      //double m2a = getMass(idMother,2);
      double m2a  = 0.0;
      double m2i  = getMass(idMother,2);
      double m2j  = getMass(idSister,2);
      double m2k  = m2s;
      m2Bef       = 0.0;
      double m2ai = -sai + m2a + m2i;

      // Get dipole 4-momentum.
      Vec4 q( pa12_tilde - pb_tilde );
      double q2   = q.m2Calc();
      double m2jk = pT2/xa + q2*( 1. - xa/za) - m2ai;

      // Perform first IF step.

      // Get transverse and parallel vector.
      Vec4 pTk_tilde( pb_tilde.px(), pb_tilde.py(), 0., 0.);
      Vec4 qpar( q + pTk_tilde );

      uCS = za*(m2ai-m2a-m2i)/q2;
      xCS = uCS + xa - (pT2*za)/(q2*xa);

      // Calculate derived variables.
      double q2par  = qpar.m2Calc();
      double pT2k   = -pTk_tilde.m2Calc();
      double s_i_jk = (1. - 1./xCS)*(q2 - m2a) + (m2i + m2jk) / xCS;
      double zbar   = (q2-s_i_jk-m2a) / bABC(q2,s_i_jk,m2a)
                     *( uCS - m2a / gABC(q2,s_i_jk,m2a)
                           * (s_i_jk + m2i - m2jk) / (q2 - s_i_jk - m2a));
      double kT2   = zbar*(1.-zbar)*s_i_jk - (1-zbar)*m2i - zbar*m2jk;

      // Disallow kinematically impossible transverse momentum.
      if (kT2 < 0. || isnan(kT2)) physical = false;

      // Now construct radiator in lab frame.
      Vec4 pa( ( pa12_tilde - 0.5*(q2-m2Bef-m2k)/q2par * qpar )
                 * sqrt( (lABC(q2,s_i_jk,m2a) - 4.*m2a*pT2k)
                       / (lABC(q2,m2k,m2Bef) - 4.*m2Bef*pT2k))
                + qpar * 0.5 * (q2 + m2a - s_i_jk) / q2par);

      // Construct left-over dipole momentum by momentum conservation.
      Vec4 pijk(-q+pa);

      double phi_kt = (!trial)
        ? ((dipEndSel->phi > 0.)
          ? dipEndSel->phi          : 2.*M_PI*rndmPtr->flat())
        : ((split->kinematics()->phi > 0.)
          ? split->kinematics()->phi : 2.*M_PI*rndmPtr->flat());

      // Allow splitting kernel to overwrite phase space variables.
      if (split->useForBranching) { phi_kt = psp["phi"]; }

      // Set up transverse momentum vector by using two perpendicular
      // four-vectors.
      pair<Vec4, Vec4> pTvecs = getTwoPerpendicular(pa, pijk);
      Vec4 kTmom( sqrt(kT2)*sin(phi_kt)*pTvecs.first
                + sqrt(kT2)*cos(phi_kt)*pTvecs.second);

      // Construct new emission momentum.
      Vec4 pi(zbar*(gABC(q2,s_i_jk,m2a)*pijk + s_i_jk*pa) / bABC(q2,s_i_jk,m2a)
                - ((1.-zbar)*s_i_jk - m2jk + m2i) / bABC(q2,s_i_jk,m2a)
                * (pa + m2a/gABC(q2,s_i_jk,m2a)*pijk)
                + kTmom);

      // Construct recoiler momentum by momentum conservation.
      Vec4 pjk(-q+pa-pi);

      // Set particle momenta.
      // Mother (a) already fixed. No need to introduce dummy intermediate.
      mother.p(pa);
      mother.m(sqrtpos(m2a));
      mother1.p(pa);
      mother1.m(sqrtpos(m2a));

      // Second sister (i) already fixed.
      sister1.p(pi);
      sister1.m(sqrtpos(m2i));

      // Intermediate off-shell recoiler. To be decayed in second step.
      newRecoiler.p(pjk);
      newRecoiler.m(sqrtpos(m2jk));

      // Perform FF step.

      // Set up kinematics as 1->2 decay in pjk rest frame.
      Vec4 pai(pa-pi);

      double phiFF = (!trial)
        ? ((dipEndSel->phia1 > 0.)
          ? dipEndSel->phia1         : 2.*M_PI*rndmPtr->flat())
        : ((split->kinematics()->phi2 > 0.)
          ? split->kinematics()->phi2 : 2.*M_PI*rndmPtr->flat());

      // Allow splitting kernel to overwrite phase space variables.
      if (split->useForBranching) { phiFF = psp["phi2"]; }

      double q2tot = q2;
      // Construct FF dipole momentum.
      q.p(pai - pjk);
      q2 = q.m2Calc();
      // Calculate CS variables.
      double m2Emt      = m2k;
      double m2Rad      = m2j;
      double zCS        = pT2/xa / ( pT2/xa - q2*xa/za);
      double yCS = (m2jk - m2Emt - m2Rad)
                 / (m2jk - m2Emt - m2Rad + 2.*pai*pjk);

      q.p(pai + pjk);
      q2 = q.m2Calc();
      // Calculate derived variables.
      double sij  = yCS * (q2 - m2ai) + (1.-yCS)*(m2Rad+m2Emt);
      zbar = (q2-sij-m2ai) / bABC(q2,sij,m2ai)
                  * (zCS - m2ai/gABC(q2,sij,m2ai)
                         *(sij + m2Rad - m2Emt)/(q2-sij-m2ai));
      kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2Rad - zbar*m2Emt;

      // Not possible to construct kinematics if kT2 < 0.0
      if (kT2 < 0.) {
        if (printWarnings)
          infoPtr->errorMsg("Warning in DireSpace::branch_IF: Reject state "
                            "with kinematically forbidden kT^2.");
        physical = false;
      }

      // NaN kT2 can happen for a 1->3 splitting in which the g->QQ~ produces
      // massive quarks Q.
      if (physical && (kT2!=kT2 || abs(kT2-kT2) > 1e5) ) {
        if (printWarnings)
          infoPtr->errorMsg("Warning in DireSpace::branch_IF: Reject state "
                            "with not-a-number kT^2 for branching " + name);
        physical = false;
      }

      // Construct left-over dipole momentum by momentum conservation.
      Vec4 pij(q-pai);

      // Set up transverse momentum vector by using two perpendicular
      // four-vectors.
      pTvecs = getTwoPerpendicular(pai, pij);
      kTmom.p( sqrt(kT2)*sin(phiFF)*pTvecs.first
             + sqrt(kT2)*cos(phiFF)*pTvecs.second);

      // Construct new radiator momentum.
      Vec4 pj( zbar * (gABC(q2,sij,m2ai)*pij - sij*pai) / bABC(q2,sij,m2ai)
                + (m2Rad+kT2) / (zbar*bABC(q2,sij,m2ai))
                  * (pai - m2ai/gABC(q2,sij,m2ai)*pij)
                + kTmom);

      // Contruct the emission momentum by momentum conservation.
      Vec4 pk(q-pj-pai);

      // Set particle momenta.
      sister.p(pj);
      sister.m(sqrtpos(m2j));
      newRecoiler1.p(pk);
      newRecoiler1.m(sqrtpos(m2k));

      // Check momenta.
      if ( !validMomentum( mother1.p(), idMother, -1)
        || !validMomentum( sister.p(),  idSister,  1)
        || !validMomentum( sister1.p(), idMother,  1)
        || !validMomentum( newRecoiler1.p(), event[iNewRecoiler1].id(), 1))
        physical = false;

      // Check invariants.
      if ( false ) {
        double saix(2.*pa*pi), sakx(2.*pa*pk), sajx(2.*pa*pj), sikx(2.*pi*pk),
               sjkx(2.*pj*pk), sijx(2.*pi*pj);
        double pptt = (sajx-sijx)*(sakx-sikx)/(saix+sajx+sakx);
        double ssaaii = saix;
        double zzaa = -q2tot/ ( saix + sajx + sakx  );
        double xxaa = (sakx-sikx) / ( saix + sajx + sakx );
        if ( physical &&
             (abs(pptt-pT2) > 1e-5 || abs(ssaaii-sai) > 1e-5 ||
              abs(zzaa-za) > 1e-5  || abs(xxaa-xa) > 1e-5) ){
          cout << "Error in branch_IF: Invariant masses after branching do "
               << "not match chosen values." << endl;
          cout << "Chosen:    "
               << " Q2 " << q2tot
               << " pT2 " << pT2
               << " sai " << sai
               << " za " << z
               << " xa " << xa << endl;
          cout << "Generated: "
               << " Q2 " << saix+sajx+sakx-sijx-sikx-sjkx
               << " pT2 " << pptt
               << " sai " << ssaaii
               << " za " << zzaa
               << " xa " << xxaa << endl;
          physical = false;
        }
      }

      // Test that enough beam momentum remains.
      int iOther  = getInB(iSysSelNow);
      if (side == 2) iOther = getInA(iSysSelNow);
      double xAnew = (mother.mother1() == 1)
                ? 2.*mother1.e()     / (beamAPtr->e() + beamBPtr->e())
                : 2.*event[iOther].e() / (beamAPtr->e() + beamBPtr->e());
      double iAold = (mother.mother1() == 1) ? iDaughter : iOther;
      double iAnew = (mother.mother1() == 1) ? iMother1 : iOther;
      double xBnew = (mother.mother1() == 1)
                ? 2.*event[iOther].e() / (beamAPtr->e() + beamBPtr->e())
                : 2.*mother1.e()     / (beamAPtr->e() + beamBPtr->e());
      double iBold = (mother.mother1() == 1) ? iOther : iDaughter;
      double iBnew = (mother.mother1() == 1) ? iOther : iMother1;
      if ( hasPDF(event[iAnew].id()) && beamAPtr->size() > 0) {
        double xOld = (*beamAPtr)[iSysSelNow].x();
        (*beamAPtr)[iSysSelNow].iPos(iAnew);
        (*beamAPtr)[iSysSelNow].x(xAnew);
        if (beamAPtr->xMax(-1) < 0.0) {
          if (!trial) infoPtr->errorMsg("Warning in DireSpace::branch_II: "
            "used up beam momentum; discard splitting.");
          physical = false;
        }
        // Restore old beams.
        (*beamAPtr)[iSysSelNow].iPos(iAold);
        (*beamAPtr)[iSysSelNow].x(xOld);
      }
      if ( hasPDF(event[iBnew].id()) && beamBPtr->size() > 0) {
        double xOld = (*beamBPtr)[iSysSelNow].x();
        (*beamBPtr)[iSysSelNow].iPos(iBnew);
        (*beamBPtr)[iSysSelNow].x(xBnew);
        if (beamBPtr->xMax(-1) < 0.0) {
          if (!trial) infoPtr->errorMsg("Warning in DireSpace::branch_II: "
            "used up beam momentum; discard splitting.");
          physical = false;
        }
        // Restore old beams.
        (*beamBPtr)[iSysSelNow].iPos(iBold);
        (*beamBPtr)[iSysSelNow].x(xOld);
      }

      double xm = 2.*mother1.e() / (beamAPtr->e() + beamBPtr->e());
      // Update dipoles and beams, for first step. Note: dipEndSel no longer
      // valid after this.
      if (physical && !trial) updateAfterIF( iSysSelNow, side, iDipSel,
        eventSizeOld, systemSizeOld, event, iDaughter, iRecoiler, iMother,
        iSister, iNewRecoiler, iNewOther, pT2, xm);

      // Update flavours, colours, status after first step.

      // Exempt intermediate off-shell recoiler from Pythia momentum checks.
      newRecoiler.status(-49);
      newRecoiler.statusNeg();
      newRecoiler.daughters( iNewRecoiler1, iNewRecoiler1);
      mother1.id( idMother);
      mother1.status(-41);
      mother1.cols(colMother1, acolMother1);
      mother1.daughters( iSister1, iMother);
      mother1.mothers( mother.mother1(), mother.mother2());
      mother.mothers( iMother1, 0);
      sister1.id(idMother);
      sister1.status(43);
      sister1.mothers(iMother1,0);
      sister1.cols(colSister1, acolSister1);
      sister1.scale(sqrt(pT2));

      if (iSysSelNow == 0) {
        event[beamOff1].daughter1( (side == 1) ? iMother1 : iNewOther );
        event[beamOff2].daughter1( (side == 2) ? iMother1 : iNewOther );
      }

      // Update dipoles and beams, for second step.
      if (physical && !trial) updateAfterIF( iSysSelNow, side, iDipSel, 0, 0,
        event, iMother, iNewRecoiler, iMother1, iSister1, iNewRecoiler1,
        iNewOther, pT2, xm);

    // Global scheme.
    } else {

      // 1->3 splitting as two consecutive IF steps.
      double m2a  = 0.0;
      double m2i  = getMass(idMother,2);
      double m2j  = getMass(idSister,2);
      double m2k  = m2s;
      double m2ai = -sai + m2a + m2i;
      m2Bef       = 0.0;

      // Perform first IF step.

      // Get dipole 4-momentum.
      Vec4 q(pb_tilde-pa12_tilde);
      double q2  = q.m2Calc();

      double m2jk = pT2/xa + q2*( 1. - xa/za) - m2ai;
      uCS = za*(m2ai-m2a-m2i)/q2;
      xCS = uCS + xa - (pT2*za)/(q2*xa);

      // Calculate derived variables.
      double zbar = (q2-m2ai-m2jk) / bABC(q2,m2ai,m2jk)
                  *( (xCS - 1)/(xCS-uCS)  - m2jk / gABC(q2,m2ai,m2jk)
                         * (m2ai + m2i - m2a) / (q2 - m2ai - m2jk));
      double kT2  = zbar*(1.-zbar)*m2ai - (1-zbar)*m2i - zbar*m2a;

      // Disallow kinematically impossible transverse momentum.
      if (kT2 < 0. || isnan(kT2)) physical = false;

      // Now construct recoiler in lab frame.
      Vec4 pjk( (pb_tilde - q*pb_tilde/q2*q)
                 *sqrt(lABC(q2,m2ai,m2jk)/lABC(q2,m2Bef,m2s))
               + 0.5*(q2+m2jk-m2ai)/q2*q );

      // Construct left-over dipole momentum by momentum conservation.
      Vec4 pai(-q+pjk);

      double phi_kt = (!trial)
        ? ((dipEndSel->phi > 0.)
          ? dipEndSel->phi          : 2.*M_PI*rndmPtr->flat())
        : ((split->kinematics()->phi > 0.)
          ? split->kinematics()->phi : 2.*M_PI*rndmPtr->flat());

      // Allow splitting kernel to overwrite phase space variables.
      if (split->useForBranching) { phi_kt = psp["phi"]; }

      // Set up transverse momentum vector by using two perpendicular
      // four-vectors.
      pair<Vec4, Vec4> pTvecs = getTwoPerpendicular(pai, pjk);
      Vec4 kTmom( sqrt(kT2)*sin(phi_kt)*pTvecs.first
                + sqrt(kT2)*cos(phi_kt)*pTvecs.second);

      // Construct new emission momentum.
      Vec4 pi( - zbar *(gABC(q2,m2ai,m2jk)*pai + m2ai*pjk) / bABC(q2,m2ai,m2jk)
                + ( (1.-zbar)*m2ai + m2i - m2a) / bABC(q2,m2ai,m2jk)
                * (pjk + m2jk/gABC(q2,m2ai,m2jk)*pai)
                + kTmom);

      // Contruct radiator momentum by momentum conservation.
      Vec4 pa(-q+pjk+pi);

      // Boost to realign the incoming radiator.
      int iOther  = getInB(iSysSelNow);
      if (side == 2) iOther = getInA(iSysSelNow);
      Vec4 pOther(event[iOther].p());

      // Boost to rest frame of incoming particles A and B.
      RotBstMatrix toABCM;
      if (side == 1) toABCM.toCMframe( pa, pOther);
      else           toABCM.toCMframe( pOther, pa);

      // After this, the radiator has vanishing pT.
      pa.rotbst(toABCM);
      pOther.rotbst(toABCM);

      // Restore the momentum fraction of the incoming particle that
      // did not participate in the splitting.
      RotBstMatrix restoreB;
      restoreB.bst( pOther, event[iOther].p());

      // After this, the inactive beam returns to the correct energy fraction.
      pa.rotbst(restoreB);
      pOther.rotbst(restoreB);

      // Boost and rotate final-state momenta.
      pi.rotbst(toABCM);
      pi.rotbst(restoreB);
      pjk.rotbst(toABCM);
      pjk.rotbst(restoreB);

      // Set all momenta.
      sister1.p(pi);
      sister1.m(sqrtpos(m2i));
      // Mother (a) already fixed. No need to introduce dummy intermediate.
      mother.p(pa);
      mother.m(sqrtpos(m2a));
      mother1.p(pa);
      mother1.m(sqrtpos(m2a));
      newRecoiler.p(pjk);

      // Rotate and boost all final state particles to absorb the pT of the
      // radiator.
      for ( int i = eventSizeOld + 2; i < eventSizeOld + systemSizeOld; ++i) {
        // Skip sister(i) and intermediate recoiler (jk), since
        // already transformed.
        if ( i == iSister1 || i == iNewRecoiler) continue;
        if ( event[i].isFinal()) {
          event[i].rotbst(toABCM);
          event[i].rotbst(restoreB);
        }
      }

      // Perform FF step.

      // Set up kinematics as 1->2 decay in pjk rest frame.
      pai.p(pa-pi);

      double phiFF = (!trial)
        ? ((dipEndSel->phia1 > 0.)
          ? dipEndSel->phia1         : 2.*M_PI*rndmPtr->flat())
        : ((split->kinematics()->phi2 > 0.)
          ? split->kinematics()->phi2 : 2.*M_PI*rndmPtr->flat());

      // Allow splitting kernel to overwrite phase space variables.
      if (split->useForBranching) { phiFF = psp["phi2"]; }

      double q2tot = q2;
      // Construct FF dipole momentum.
      q.p(pai - pjk);
      q2 = q.m2Calc();
      // Calculate CS variables.
      double m2Emt      = m2k;
      double m2Rad      = m2j;
      double zCS        = pT2/xa / ( pT2/xa - q2*xa/za);
      double yCS = (m2jk - m2Emt - m2Rad)
                 / (m2jk - m2Emt - m2Rad + 2.*pai*pjk);

      q.p(pai + pjk);
      q2 = q.m2Calc();
      // Calculate derived variables.
      double sij  = yCS * (q2 - m2ai) + (1.-yCS)*(m2Rad+m2Emt);
      zbar = (q2-sij-m2ai) / bABC(q2,sij,m2ai)
                  * (zCS - m2ai/gABC(q2,sij,m2ai)
                         *(sij + m2Rad - m2Emt)/(q2-sij-m2ai));
      kT2  = zbar*(1.-zbar)*sij - (1.-zbar)*m2Rad - zbar*m2Emt;

      // Not possible to construct kinematics if kT2 < 0.0
      if (kT2 < 0.) {
        if (printWarnings)
          infoPtr->errorMsg("Warning in DireSpace::branch_IF: Reject state "
                            "with kinematically forbidden kT^2.");
        physical = false;
      }

      // NaN kT2 can happen for a 1->3 splitting in which the g->QQ~ produces
      // massive quarks Q.
      if (physical && (kT2!=kT2 || abs(kT2-kT2) > 1e5) ) {
        if (printWarnings)
          infoPtr->errorMsg("Warning in DireSpace::branch_IF: Reject state "
                            "with not-a-number kT^2 for branching " + name);
        physical = false;
      }

      // Construct left-over dipole momentum by momentum conservation.
      Vec4 pij(q-pai);

      // Set up transverse momentum vector by using two perpendicular
      // four-vectors.
      pTvecs = getTwoPerpendicular(pai, pij);
      kTmom.p( sqrt(kT2)*sin(phiFF)*pTvecs.first
             + sqrt(kT2)*cos(phiFF)*pTvecs.second);

      // Construct new radiator momentum.
      Vec4 pj( zbar * (gABC(q2,sij,m2ai)*pij - sij*pai) / bABC(q2,sij,m2ai)
                + (m2Rad+kT2) / (zbar*bABC(q2,sij,m2ai))
                  * (pai - m2ai/gABC(q2,sij,m2ai)*pij)
                + kTmom);

      // Contruct the emission momentum by momentum conservation.
      Vec4 pk(q-pj-pai);

      // Set particle momenta.
      sister.p(pj);
      sister.m(sqrtpos(m2j));
      newRecoiler1.p(pk);
      newRecoiler1.m(sqrtpos(m2k));

      // Check momenta.
      if ( !validMomentum( mother1.p(), idMother, -1)
        || !validMomentum( sister.p(),  idSister,  1)
        || !validMomentum( sister1.p(), idMother,  1)
        || !validMomentum( newRecoiler1.p(), event[iNewRecoiler1].id(), 1))
        physical = false;

      // Check invariants.
      if ( true ) {
        double saix(2.*pa*pi), sakx(2.*pa*pk), sajx(2.*pa*pj), sikx(2.*pi*pk),
               sjkx(2.*pj*pk), sijx(2.*pi*pj);
        double pptt = (sajx-sijx)*(sakx-sikx)/(saix+sajx+sakx);
        double ssaaii = saix;
        double zzaa = -q2tot/ ( saix + sajx + sakx  );
        double xxaa = (sakx-sikx) / ( saix + sajx + sakx );
        if ( physical &&
             (abs(pptt-pT2) > 1e-5 || abs(ssaaii-sai) > 1e-5 ||
              abs(zzaa-za) > 1e-5  || abs(xxaa-xa) > 1e-5) ){
          cout << "Error in branch_IF: Invariant masses after branching do "
               << "not match chosen values." << endl;
          cout << "Chosen:    "
               << " Q2 " << q2tot
               << " pT2 " << pT2
               << " sai " << sai
               << " za " << z
               << " xa " << xa << endl;
          cout << "Generated: "
               << " Q2 " << saix+sajx+sakx-sijx-sikx-sjkx
               << " pT2 " << pptt
               << " sai " << ssaaii
               << " za " << zzaa
               << " xa " << xxaa << endl;
          physical = false;
        }
      }

      double xm = 2.*mother1.e() / (beamAPtr->e() + beamBPtr->e());
      // Update dipoles and beams, for first step. Note: dipEndSel no longer
      // valid after this.
      if (physical && !trial) updateAfterIF( iSysSelNow, side, iDipSel,
        eventSizeOld, systemSizeOld, event, iDaughter, iRecoiler, iMother,
        iSister, iNewRecoiler, iNewOther, pT2, xm);

      // Update flavours, colours, status after first step.

      // Exempt intermediate off-shell recoiler from Pythia momentum checks.
      newRecoiler.status(-49);
      newRecoiler.statusNeg();
      newRecoiler.daughters( iNewRecoiler1, iNewRecoiler1);
      mother1.id( idMother);
      mother1.status(-41);
      mother1.cols(colMother1, acolMother1);
      mother1.daughters( iSister1, iMother);
      mother1.mothers( mother.mother1(), mother.mother2());
      mother.mothers( iMother1, 0);
      sister1.id(idMother);
      sister1.status(43);
      sister1.mothers(iMother1,0);
      sister1.cols(colSister1, acolSister1);
      sister1.scale(sqrt(pT2));

      if (iSysSelNow == 0) {
        event[beamOff1].daughter1( (side == 1) ? iMother1 : iNewOther );
        event[beamOff2].daughter1( (side == 2) ? iMother1 : iNewOther );
      }

      // Update dipoles and beams, for second step.
      if (physical && !trial) updateAfterIF( iSysSelNow, side, iDipSel, 0, 0,
        event, iMother, iNewRecoiler, iMother1, iSister1, iNewRecoiler1,
        iNewOther, pT2, xm);

    }

    // Done with 1->3 splitting kinematics.

  }

  physical = physical && !doVeto;

  // Temporarily set the daughters in the beams to zero, to
  // allow mother-daughter relation checks.
  if (iSysSelNow > 0) {
    if (side == 1) event[beamOff1].daughter1(0);
    if (side == 2) event[beamOff2].daughter1(0);
  }

  // Check if mother-daughter relations are correctly set. Check only
  // possible if no MPI are present.
  if ( physical && !trial && !doMECreject && !validMotherDaughter(event)) {
    if (printWarnings)
      infoPtr->errorMsg("Error in DireSpace::branch_IF: Mother-daughter "
                        "relations after branching not valid.");
    physical = false;
  }

  // Restore correct daughters in the beams.
  if (iSysSelNow > 0) {
    if (side == 1) event[beamOff1].daughter1(iBeam1Dau1);
    if (side == 2) event[beamOff2].daughter1(iBeam2Dau1);
  }

  // Allow veto of branching. If so restore event record to before emission.
  if ( !physical || doMECreject) {
    event.popBack( event.size() - eventSizeOld);

    if (iSysSelNow == 0) {
      event[beamOff1].daughter1( ev1Dau1V);
      event[beamOff2].daughter1( ev2Dau1V);
    }

    if (useGlobalMapIF) {
      for ( int iCopy = 0; iCopy < systemSizeOld; ++iCopy) {
        int iOldCopy = partonSystemsPtr->getAll(iSysSel, iCopy);
        event[iOldCopy].status( statusV[iCopy]);
        event[iOldCopy].mothers( mother1V[iCopy], mother2V[iCopy]);
        event[iOldCopy].daughters( daughter1V[iCopy], daughter2V[iCopy]);
      }
    } else {
      event[iDaughter].status( iDauStatusV);
      event[iDaughter].mothers(iDauMot1V, iDauMot2V);
      event[iDaughter].daughters(iDauDau1V, iDauDau2V);
      event[iRecoiler].status( iRecStatusV);
      event[iRecoiler].mothers( iRecMot1V, iRecMot2V);
      event[iRecoiler].daughters( iRecDau1V, iRecDau2V);
    }

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
  if (trial) split->storePosAfter(
    (nEmissions < 2) ? iMother : iMother1,
    (nEmissions < 2) ? iNewRecoiler : iNewRecoiler1,
    iSister, (nEmissions < 2) ? 0 : iSister1);

  // Set shower weight.
  if (!trial) {
    if (!doTrialNow) {
      weights->calcWeight(pT2);
      weights->reset();

      // Store positions of new soft particles for (FSR) evolution.
      direInfoPtr->updateSoftPos( iDaughter, iSister );
      direInfoPtr->updateSoftPosIfMatch( iRecoiler, iNewRecoiler );
      if (nEmissions > 1) {
        direInfoPtr->updateSoftPosIfMatch( iNewRecoiler, iNewRecoiler1 );
        direInfoPtr->addSoftPos( iSister1 );
      }
    }

    // Clear accept/reject weights.
    for ( unordered_map<string, multimap<double,double> >::iterator
      it = rejectProbability.begin(); it != rejectProbability.end(); ++it )
      it->second.clear();
    for ( unordered_map<string, map<double,double> >::iterator
      it = acceptProbability.begin(); it != acceptProbability.end(); ++it )
      it->second.clear();
  }

  // Done without any errors.
  return true;

}

//--------------------------------------------------------------------------

void DireSpace::updateAfterIF( int iSysSelNow, int sideNow, int iDipSelNow,
  int eventSizeOldNow, int systemSizeOldNow, Event& event, int iDaughter,
  int iRecoiler, int iMother, int iSister, int iNewRecoiler, int iNewOther,
  double pT2, double xNew) {

  // Update the number of proposed emissions.
  if ( nProposedPT.find(iSysSelNow) != nProposedPT.end() )
    ++nProposedPT[iSysSelNow];

  int idMother         = event[iMother].id();
  int idDaughterNow    = event[iDaughter].id();
  bool motherHasPlusPz = (event[iMother].pz() > 0.);

  // Bookkeep shower-induced resonances.
  if ( direInfoPtr->isRes(iDaughter) &&
    event[iMother].id() != event[iDaughter].id() )
    direInfoPtr->removeResPos(iDaughter);
  if ( particleDataPtr->isResonance(event[iMother].id()) ) {
    if ( direInfoPtr->isRes(iDaughter) )
         direInfoPtr->updateResPos(iDaughter,iMother);
  }
  if ( direInfoPtr->isRes(iRecoiler) )
    direInfoPtr->updateResPos(iRecoiler,iNewRecoiler);
  if ( particleDataPtr->isResonance(event[iSister].id()) )
    direInfoPtr->addResPos(iSister);

  // Update list of partons in system; adding newly produced one.
  if (motherHasPlusPz) {
    partonSystemsPtr->setInA(iSysSelNow, iMother);
    partonSystemsPtr->setInB(iSysSelNow, iNewOther);
  } else {
    partonSystemsPtr->setInB(iSysSelNow, iMother);
    partonSystemsPtr->setInA(iSysSelNow, iNewOther);
  }

  // Update and add newly produced particles.
  if (useGlobalMapIF) {
    // Add newly produced particle.
    for (int iCopy = 2; iCopy < systemSizeOldNow; ++iCopy) {
      int iOut = partonSystemsPtr->getOut(iSysSelNow, iCopy - 2);
      // Remember shower-induced resonances.
      direInfoPtr->updateResPosIfMatch ( iOut, eventSizeOldNow + iCopy);
      direInfoPtr->updateSoftPosIfMatch( iOut, eventSizeOldNow + iCopy);
      partonSystemsPtr->setOut(iSysSelNow, iCopy - 2, eventSizeOldNow + iCopy);
    }
    partonSystemsPtr->addOut(iSysSelNow, iSister);
    partonSystemsPtr->replace(iSysSelNow, iRecoiler, iNewRecoiler);
  } else {
    // Add newly produced particle.
    partonSystemsPtr->addOut(iSysSelNow, iSister);
    partonSystemsPtr->replace(iSysSelNow, iRecoiler, iNewRecoiler);
  }

  // Get new center-of-mass energy
  int iA      = getInA(iSysSelNow);
  int iB      = getInB(iSysSelNow);
  double shat = (event[iA].p() + event[iB].p()).m2Calc();
  partonSystemsPtr->setSHat(iSysSelNow, shat);

  // dipEnd array may have expanded and been moved, so regenerate dipEndSel.
  dipEndSel = &dipEnd[iDipSelNow];

  // Update info on radiating dipole ends (QCD).
  for (int iDip = 0; iDip < int(dipEnd.size()); ++iDip)
  if ( dipEnd[iDip].system == iSysSelNow) {
    if (abs(dipEnd[iDip].side) == sideNow) {
      dipEnd[iDip].iRadiator = iMother;
      dipEnd[iDip].iRecoiler = iNewRecoiler;
      if (dipEnd[iDip].colType  != 0)
        dipEnd[iDip].colType = event[iMother].colType();

    // Update info on recoiling dipole ends (QCD or QED).
    } else {
      dipEnd[iDip].iRadiator = iNewRecoiler;
      dipEnd[iDip].iRecoiler = iMother;
      dipEnd[iDip].MEtype = 0;
    }
  }

  // Update info on beam remnants.
  BeamParticle& beamNow = (sideNow == 1) ? *beamAPtr : *beamBPtr;
  beamNow[iSysSelNow].update( iMother, idMother, xNew);
  // Redo choice of companion kind whenever new flavour.
  if (idMother != idDaughterNow) {
    pdfScale2 = (useFixedFacScale) ? fixedFacScale2 : factorMultFac * pT2;
    pdfScale2 = max(pdfScale2, pT2min);
    beamNow.xfISR( iSysSelNow, idMother, xNew, pdfScale2);
    beamNow.pickValSeaComp();
  }
  BeamParticle& beamRec = (sideNow == 1) ? *beamBPtr : *beamAPtr;
  beamRec[iSysSelNow].iPos( iNewOther);

  // Update ISR dipoles.
  update(iSysSelNow,event);

  // Pointer to selected dipole no longer valid after update, thus unset.
  dipEndSel = 0;

}

//--------------------------------------------------------------------------

pair <Event, pair<int,int> > DireSpace::clustered_internal( const Event& state,
  int iRad, int iEmt, int iRecAft, string name ) {

  // Flags for type of radiation
  int radType = state[iRad].isFinal() ? 1 : -1;
  int recType = state[iRecAft].isFinal() ? 1 : -1;

  // Construct the clustered event
  Event NewEvent = Event();
  NewEvent.init("(hard process-modified)", particleDataPtr);
  NewEvent.clear();
  // Copy all unchanged particles to NewEvent
  for (int i = 0; i < state.size(); ++i) {
    if ( i == iRad || i == iRecAft || i == iEmt ) continue;
    NewEvent.append( state[i] );
  }

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
  Particle RecBefore = Particle( state[iRecAft] );
  RecBefore.setEvtPtr(&NewEvent);
  RecBefore.daughters(0,0);
  // Find flavour of radiator before splitting
  int radID = splits[name]->radBefID(state[iRad].id(), state[iEmt].id());
  int recID = state[iRecAft].id();
  Particle RadBefore = Particle( state[iRad] );
  RadBefore.setEvtPtr(&NewEvent);
  RadBefore.id(radID);
  RadBefore.daughters(0,0);
  // Put dummy values for colours
  RadBefore.cols(RecBefore.acol(),RecBefore.col());

  // Reset status if the reclustered radiator is a resonance.
  if ( particleDataPtr->isResonance(radID) && radType == 1)
    RadBefore.status(state[iRad].status());

  // Put mass for radiator and recoiler
  double radMass = particleDataPtr->m0(radID);
  double recMass = particleDataPtr->m0(recID);
  if (radType == 1 ) RadBefore.m(radMass);
  else RadBefore.m(0.0);
  if (recType == 1 ) RecBefore.m(recMass);
  else RecBefore.m(0.0);

  // Construct momenta and  colours of clustered particles
  bool isClustered = false;
  if ( state[iRecAft].isFinal())
    isClustered = cluster_IF(state,iRad,iEmt,iRecAft,radID,RadBefore,
                             RecBefore,NewEvent);
  else
    isClustered = cluster_II(state,iRad,iEmt,iRecAft,radID,RadBefore,
                             RecBefore,NewEvent);

  // Clustering not possible, e.g. because not in allowed phase space.
  if (!isClustered) { NewEvent.clear(); return make_pair
                                          (NewEvent, make_pair(0,0));}

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

  // Force incoming partons to have "hard event" status
  outState[3].status(-21);
  outState[4].status(-21);

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
    if (NewEvent[i].status() == 22) outState.append( NewEvent[i] );
  // Then start appending partons
  if (!radAppended && RadBefore.statusAbs() == 22)
    radPos = outState.append(RadBefore);
  if (!recAppended)
    recPos = outState.append(RecBefore);
  if (!radAppended && RadBefore.statusAbs() != 22)
    radPos = outState.append(RadBefore);
  // Then partons (not reclustered recoiler)
  for(int i = 0; i < int(NewEvent.size()-1); ++i)
    if ( NewEvent[i].status()  != 22
      && NewEvent[i].colType() != 0
      && NewEvent[i].isFinal()) {
      outState.append( NewEvent[i] );
      // Force partons to have "hard event" status
      int status = particleDataPtr->isResonance(NewEvent[i].id()) ? 22 : 23;
      outState.back().status(status);
      outState.back().mother1(3);
      outState.back().mother2(4);
    }
  // Then the rest
  for(int i = 0; i < int(NewEvent.size()-1); ++i)
    if ( NewEvent[i].status() != 22
      && NewEvent[i].colType() == 0
      && NewEvent[i].isFinal() ) {
      outState.append( NewEvent[i]);
      int status = particleDataPtr->isResonance(NewEvent[i].id()) ? 22 : 23;
      outState.back().status(status);
      outState.back().mother1(3);
      outState.back().mother2(4);
    }

  // Find intermediate and respective daughters
  vector<int> PosIntermediate;
  vector<int> PosDaughter1;
  vector<int> PosDaughter2;
  for(int i=0; i < int(outState.size()); ++i)
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
  // Set daughters and mothers
  for(int i=0; i < int(PosIntermediate.size()); ++i) {
    outState[PosIntermediate[i]].daughters(PosDaughter1[i],PosDaughter2[i]);
    outState[PosDaughter1[i]].mother1(PosIntermediate[i]);
    outState[PosDaughter2[i]].mother1(PosIntermediate[i]);
    outState[PosDaughter1[i]].mother2(0);
    outState[PosDaughter2[i]].mother2(0);
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
        if ( i != iRad && i != iEmt && i != iRecAft
          && state[i].status() == -22
          && state[i].col() == state[iRad].col() )
          iColRes = i;
  } else if ( radType == -1 && state[iRad].colType() == -1) {
      // Find resonance connected to initial anticolour
      for(int i=0; i < int(state.size()); ++i)
        if ( i != iRad && i != iEmt && i != iRecAft
          && state[i].status() == -22
          && state[i].acol() == state[iRad].acol() )
          iColRes = i;
  } else if ( radType == 1 && state[iRad].colType() == 1) {
      // Find resonance connected to final state colour
      for(int i=0; i < int(state.size()); ++i)
        if ( i != iRad && i != iEmt && i != iRecAft
          && state[i].status() == -22
          && state[i].col() == state[iRad].col() )
          iColRes = i;
  } else if ( radType == 1 && state[iRad].colType() == -1) {
      // Find resonance connected to final state anticolour
      for(int i=0; i < int(state.size()); ++i)
        if ( i != iRad && i != iEmt && i != iRecAft
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

  // Now reset the beams, if necessary.
  int iDof1 = outState[3].mother1() == 1 ? 3 : 4;
  int iDof2 = outState[3].mother1() == 2 ? 3 : 4;

  if (outState[1].idAbs() == 11 && outState[iDof1].id() == 22 ) {
    outState[1].id(outState[iDof1].id());
  }
  if (outState[2].idAbs() == 11 && outState[iDof2].id() == 22 ) {
    outState[2].id(outState[iDof2].id());
  }
  if (outState[1].idAbs() == 11 && outState[iDof1].colType() != 0) {
    outState[1].id(2212);
  }
  if (outState[2].idAbs() == 11 && outState[iDof2].colType() != 0) {
    outState[2].id(2212);
  }

  // Now check event.
  for ( int i = 0; i < outState.size(); ++i) {
    if ( outState[i].status() == 23 && particleDataPtr->
         isResonance(outState[i].id())) outState[i].status(22);
  }

  // Check if the state is valid. If not, return empty state.
  if (!validEvent( outState, true )) { outState.clear(); }

  // Done
  return make_pair(outState, make_pair(radPos, recPos));

}

//--------------------------------------------------------------------------

bool DireSpace::cluster_II( const Event& state,
  int iRad, int iEmt, int iRecAft, int idRadBef, Particle& radBef,
  Particle& recBef, Event& partial ) {

  if (false) cout << idRadBef;

  // Calculate CS variables.
  double pT2    = pT2_II(state[iRad], state[iEmt], state[iRecAft]);
  double Q2     = 2.*state[iRad].p()*state[iRecAft].p()
                - 2.*state[iRad].p()*state[iEmt].p()
                - 2.*state[iEmt].p()*state[iRecAft].p();
  double z      = z_II(state[iRad], state[iEmt], state[iRecAft]);

  double kappa2 = pT2 / Q2;
  double xCS    = (z*(1-z)- kappa2)/(1-z);

  // Get particle masses.
  double m2Bef = 0.0, m2r = 0.0;
  double m2e   = state[iEmt].p().m2Calc();
  double m2s   = state[iRecAft].p().m2Calc();

  // Check phase space contraints.
  double xNew = 2.*state[iRad].e()/state[0].m();
  double xOld = xNew*xCS;

  double xMin = (particleDataPtr->colType(idRadBef) != 0) ? xOld : 0.;

  if ( !inAllowedPhasespace( 1, z, pT2, Q2, xMin, -2, m2Bef, m2r, m2s, m2e) ) {
    return false;
  }

  // Set up kinematics.
  Vec4 q(state[iRad].p() - state[iEmt].p() + state[iRecAft].p());
  double q2 = q.m2Calc();
  double sab = (state[iRad].p() + state[iRecAft].p()).m2Calc();

  Vec4 pRad = ( state[iRad].p() - m2r/gABC(sab,m2r,m2s)*state[iRecAft].p())
             *sqrt(lABC(q2,m2Bef,m2s)/lABC(sab,m2r,m2s))
           + m2Bef / gABC(q2,m2Bef,m2s)*state[iRecAft].p();

  radBef.p( pRad );
  recBef.p( state[iRecAft].p() );

  // Set mass of initial recoiler to zero
  radBef.m( 0.0 );
  recBef.m( 0.0 );

  Vec4 kTilde(radBef.p() + recBef.p());
  Vec4 k(state[iRad].p() + state[iRecAft].p()  - state[iEmt].p());
  Vec4 kSum(kTilde + k);
  for ( int i = 0; i < partial.size(); ++i) {
    if ( !partial[i].isFinal() && partial[i].statusAbs() != 22 ) continue;
    Vec4 pIn = partial[i].p();
    double kSum2    = kSum.m2Calc();
    double k2       = k.m2Calc();
    double kXp      = k*pIn;
    double kSumXp   = kSum*pIn;
    Vec4 res = pIn - kSum * 2.0*( kSumXp / kSum2 ) + kTilde * 2.0 *( kXp/k2);
    partial[i].p(res);
  }

  // Done
  return true;

}

//--------------------------------------------------------------------------

bool DireSpace::cluster_IF( const Event& state,
  int iRad, int iEmt, int iRecAft, int idRadBef, Particle& radBef,
  Particle& recBef, Event& partial ) {

  if (false) cout << idRadBef;

  // Calculate CS variables.
  double pT2 = pT2_IF(state[iRad], state[iEmt], state[iRecAft]);
  double z   = z_IF(state[iRad], state[iEmt], state[iRecAft]);
  double xCS = z;
  int side   = (state[iRad].pz() > 0.) ? 1 : -1;

  // Get particle masses.
  double m2Bef = 0.0, m2r = 0.0;
  //double m2e = 0.0;
  double m2e   = state[iEmt].m2Calc();
  double m2s   = state[iRecAft].p().m2Calc();

  // Adjust the dipole kinematical mass to accomodate masses after branching.
  double Q2  = 2.*state[iRad].p()*state[iEmt].p()
             + 2.*state[iRad].p()*state[iRecAft].p()
             - 2.*state[iRecAft].p()*state[iEmt].p();
  double xNew = 2.*state[iRad].e()/state[0].m();
  double xOld = xNew*xCS;

  // Check phase space contraints.
  double xMin = (particleDataPtr->colType(idRadBef) != 0) ? xOld : 0.;

  if ( !inAllowedPhasespace( 1, z, pT2, Q2, xMin, 2, m2Bef, m2r, m2s, m2e) )
    return false;

  Vec4 pRadBef, pRecBef;

  // Massive kinematics, in two schemes.
  if (!useGlobalMapIF) {

    // Get dipole 4-momentum.
    Vec4 paj(state[iRad].p()-state[iEmt].p());
    Vec4 pk(state[iRecAft].p());
    Vec4 pTk(pk.px()+state[iEmt].px(),pk.py()+state[iEmt].py(),0.,0.);
    Vec4 q(paj-pk);
    Vec4 qpar(q+pTk);

    // Calculate derived variables.
    double q2    = q.m2Calc();
    double q2par = qpar.m2Calc();
    double pT2k  = -pTk.m2Calc();
    //double sjk   = 2.*state[iRecAft].p()*state[iEmt].p();
    double sjk   = ( state[iRecAft].p() + state[iEmt].p()).m2Calc();

    // Now construct radiator in lab frame.
    pRadBef = ( state[iRad].p() - q*state[iRad].p()/q2par * qpar )
               * sqrt( (lABC(q2,m2s,m2Bef) - 4.*m2Bef*pT2k)
                     / (lABC(q2,sjk,m2r) - 4.*m2r*pT2k))
              + qpar * 0.5 * (q2 + m2Bef - m2s) / q2par;


    // Contruct recoiler momentum by momentum conservation.
    pRecBef = -q+pRadBef;

    radBef.p(pRadBef);
    recBef.p(pRecBef);
    radBef.m(sqrtpos(m2r));
    recBef.m(sqrtpos(m2s));

  } else {

    // Get dipole 4-momentum.
    Vec4 paj(state[iRad].p()-state[iEmt].p());
    Vec4 pk(state[iRecAft].p());
    Vec4 q(pk-paj);

    // Calculate derived variables.
    double q2 = q.m2Calc();
    double saj = 2.*state[iRad].p()*state[iEmt].p();

    // Now construct recoiler in lab frame.
    pRecBef = (pk - q*pk/q2*q)
               *sqrt(lABC(q2,m2Bef,m2s)/lABC(q2,saj,m2s))
             + 0.5*(q2+m2s-m2Bef)/q2*q;

    // Contruct radiator momentum by momentum conservation.
    pRadBef = -q+pRecBef;

    // Boost to realign the incoming radiator.
    int iOther  = getInB(iSysSel);
    if (side == -1) iOther = getInA(iSysSel);
    Vec4 pOther(state[iOther].p());

    // Boost to rest frame of incoming particles A and B.
    RotBstMatrix toABCM;
    if (side == 1) toABCM.toCMframe( pRadBef, pOther);
    else           toABCM.toCMframe( pOther, pRadBef);

    // After this, the radiator has vanishing pT.
    pRadBef.rotbst(toABCM);
    pOther.rotbst(toABCM);

    // Restore the momentum fraction of the incoming particle that
    // did not participate in the splitting.
    RotBstMatrix restoreB;
    restoreB.bst( pOther, state[iOther].p());

    // After this, the inactive beam returns to the correct energy fraction.
    pRadBef.rotbst(restoreB);
    pOther.rotbst(restoreB);

    // Set all momenta.
    radBef.p(pRadBef);
    recBef.p(pRecBef);
    radBef.m(sqrtpos(m2r));
    recBef.m(sqrtpos(m2s));

    // Rotate and boost all final state particles to absorb the pT of the
    // radiator.
    for ( int i = 0; i < partial.size(); ++i) {
      if ( !partial[i].isFinal() && partial[i].statusAbs() != 22 ) continue;
      partial[i].rotbst(toABCM);
      partial[i].rotbst(restoreB);
    }

  }

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

int DireSpace::FindParticle( const Particle& particle, const Event& event,
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

double DireSpace::pT2_II ( const Particle& rad, const Particle& emt,
  const Particle& rec) {
  double sai = -2.*rad.p()*emt.p();
  double sbi = -2.*rec.p()*emt.p();
  double sab =  2.*rad.p()*rec.p();

  return sai*sbi / sab * ( sai+sbi+sab ) / sab;

}

//--------------------------------------------------------------------------

double DireSpace::pT2_IF ( const Particle& rad, const Particle& emt,
  const Particle& rec) {
  double sai = -2.*rad.p()*emt.p();
  double sik =  2.*rec.p()*emt.p();
  double sak = -2.*rad.p()*rec.p();
  return sai*sik / (sai+sak) * (sai+sik+sak) / (sai+sak);
}

//--------------------------------------------------------------------------

double DireSpace::z_II ( const Particle& rad, const Particle& emt,
  const Particle& rec) {
  double sbi = -2.*rec.p()*emt.p();
  double sab =  2.*rad.p()*rec.p();
  return 1. + sbi/sab;
}

//-------------------------------------------------------------------------

double DireSpace::z_IF ( const Particle& rad, const Particle& emt,
  const Particle& rec) {
  double sai = -2.*rad.p()*emt.p();
  double sik =  2.*rec.p()*emt.p();
  double sak = -2.*rad.p()*rec.p();
  return 1. + sik / (sai+sak);
}

//--------------------------------------------------------------------------

double DireSpace::m2dip_II ( const Particle& rad, const Particle& emt,
  const Particle& rec) {
  double sai = -2.*rad.p()*emt.p();
  double sbi = -2.*rec.p()*emt.p();
  double sab =  2.*rad.p()*rec.p();
  return (sab + sai + sbi);
}

//--------------------------------------------------------------------------

double DireSpace::m2dip_IF ( const Particle& rad, const Particle& emt,
  const Particle& rec) {
  double sai = -2.*rad.p()*emt.p();
  double sik =  2.*rec.p()*emt.p();
  double sak = -2.*rad.p()*rec.p();
  return -1.*(sai+sik+sak);
}

//-------------------------------------------------------------------------

// From Pythia version 8.218 onwards.
// Return the evolution variable and splitting information. See header for
// more comments.

map<string, double> DireSpace::getStateVariables (const Event& state,
  int rad, int emt, int rec, string name) {
  map<string,double> ret;

  // State variables for a shower splitting (radBef,recBef) --> (rad,emt,rec)
  if (rad > 0 && emt > 0 && rec > 0) {
    double pT2 = pT2Space ( state[rad], state[emt], state[rec]);
    double z   = zSpace ( state[rad], state[emt], state[rec]);
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

    double m2dip = m2dipSpace ( state[rad], state[emt], state[rec]);
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
    // Find positions of incoming colliding partons.
    int in1(3), in2(4);
    vector<DireSpaceEnd> dipEnds;
    // Find dipole end formed by colour index.
    int colTag = state[in1].col();
    if (colTag > 0)  getQCDdip( in1,  colTag,  1, state, dipEnds);
    // Find dipole end formed by anticolour index.
    int acolTag = state[in1].acol();
    if (acolTag > 0) getQCDdip( in1, acolTag, -1, state, dipEnds);
    // Find dipole end formed by colour index.
    colTag = state[in2].col();
    if (colTag > 0)  getQCDdip( in2,  colTag,  1, state, dipEnds);
    // Find dipole end formed by anticolour index.
    acolTag = state[in2].acol();
    if (acolTag > 0) getQCDdip( in2, acolTag, -1, state, dipEnds);

    // Now find non-QCD dipoles and/or update the existing dipoles.
    getGenDip( -1, in1, state, false, dipEnds);
    getGenDip( -1, in2, state, false, dipEnds);

    // Get x for both beams.
    double x1 = state[3].pPos() / state[0].m();
    double x2 = state[4].pNeg() / state[0].m();

    // Store invariant masses of all dipole ends.
    stringstream oss;
    for (int iDip = 0; iDip < int(dipEnds.size()); ++iDip) {
      double m2 = abs(2.*state[dipEnds[iDip].iRadiator].p()
                        *state[dipEnds[iDip].iRecoiler].p());
      if ( dipEnds[iDip].iRadiator == in1) m2 /= x1;
      if ( dipEnds[iDip].iRecoiler == in1) m2 /= x1;
      if ( dipEnds[iDip].iRadiator == in2) m2 /= x2;
      if ( dipEnds[iDip].iRecoiler == in2) m2 /= x2;
      oss.str("");
      oss << "scalePDF-" << dipEnds[iDip].iRadiator
           << "-"        << dipEnds[iDip].iRecoiler;
      ret.insert(make_pair(oss.str(),m2));
    }
  }

  return ret;
}

//-------------------------------------------------------------------------

// Compute splitting probability.
// From Pythia version 8.215 onwards.
double DireSpace::getSplittingProb( const Event& state, int iRad,
  int iEmt, int iRecAft, string name) {

  // Get kernel order.
  int order = atoi( (char*)name.substr( name.find("-",0)+1,
                                        name.size() ).c_str() );
  name=name.substr( 0, name.size()-2);

  // Do nothing if kernel says so, e.g. to avoid infinite loops
  // if the kernel uses the History class.
  if ( splits[name]->splitInfo.extras.find("unitKernel")
    != splits[name]->splitInfo.extras.end() ) return 1.;

  double z     = zSpace(state[iRad], state[iEmt], state[iRecAft]);
  double pT2   = pT2Space(state[iRad], state[iEmt], state[iRecAft]);
  double m2D   = (state[iRecAft].isFinal())
               ? abs( 2.*state[iEmt].p()*state[iRad].p()
                     -2.*state[iEmt].p()*state[iRecAft].p()
                     +2.*state[iRad].p()*state[iRecAft].p())
               : abs(-2.*state[iEmt].p()*state[iRad].p()
                     -2.*state[iEmt].p()*state[iRecAft].p()
                     +2.*state[iRad].p()*state[iRecAft].p());
  int idRadBef = splits[name]->radBefID(state[iRad].id(), state[iEmt].id());
  double m2Bef = ( abs(idRadBef) < 6)
               ? getMass(idRadBef,2)
               : (idRadBef == state[iRad].id())
                  ? getMass(idRadBef,3,state[iRad].mCalc())
                  : getMass(idRadBef,2);
  double m2r   = state[iRad].p().m2Calc();
  double m2e   = state[iEmt].p().m2Calc();
  double m2s   = state[iRecAft].p().m2Calc();
  int type     = (state[iRecAft].isFinal()) ? 1 : -1;

  // Disallow below cut-off.
  if ( pT2cut(state[iEmt].id()) > pT2) return 0.;
  if ( !splits[name]->aboveCutoff( pT2, state[iRad], state[iRecAft], 0,
        partonSystemsPtr)) return 0.;

  // Upate type if this is a massive splitting.
  if (type == 1 && (m2Bef > TINYMASS || m2r > TINYMASS || m2e > TINYMASS
    || m2s > TINYMASS)) type = 2;
  if (type ==-1 && (m2Bef > TINYMASS || m2r > TINYMASS || m2e > TINYMASS
    || m2s > TINYMASS)) type =-2;

  // Adjust the dipole kinematical mass to accomodate masses after branching.
  double m2dipCorr  = m2D;
  double kappa2     = pT2 / m2dipCorr;
  double xCS        = (state[iRecAft].isFinal()
                      ? z : (z*(1-z) - kappa2) / (1 -z));
  double xMot = 2.*state[iRad].e()/state[0].m();
  double xDau = xCS * xMot;

  // Check phase space contraints.
  double xMin = (particleDataPtr->colType(idRadBef) != 0) ? xDau : 0.;

  if ( !inAllowedPhasespace( 1, z, pT2, m2dipCorr, xMin, type, m2Bef, m2r, m2s,
    m2e) ) { return 0.0;}

  // Calculate splitting probability.
  double p = 0.;

  // Get phi angle.
  int massSign = type > 0 ? 1 : -1;
  pair<Vec4, Vec4> pTdirection = getTwoPerpendicular(state[iRad].p(),
    massSign*state[iRecAft].p()+state[iEmt].p());
  double px= -pTdirection.first*state[iEmt].p();
  double py= -pTdirection.second*state[iEmt].p();
  double kT2 = pow2(px)+pow2(py);
  double phi1 = atan2(px/sqrt(kT2), py/sqrt(kT2));
  if (phi1 < 0.) phi1 = 2.*M_PI+phi1;

  // Setup splitting information.
  pair <Event, pair<int,int> > born
    (clustered_internal( state, iRad, iEmt, iRecAft, name ));
  int nEmissions = splits[name]->nEmissions();
  double m2dipBef = abs(2.*born.first[born.second.first].p()
                        *born.first[born.second.second].p());
  splits[name]->splitInfo.clear();
  splits[name]->splitInfo.storeInfo(name, type, 0, 0, 0,
    born.second.first, born.second.second,
    born.first, state[iEmt].id(), state[iRad].id(),
    nEmissions, m2dipBef, pT2, pT2, z, phi1, m2Bef, m2s,
    (nEmissions == 1 ? m2r : 0.0),(nEmissions == 1 ? m2e : 0.0),
    0.0, 0.0, 0.0, 0.0, xDau, xMot);

  // Get splitting probability.
  unordered_map < string, double > kernels;
  // Get complete kernel.
  if (splits[name]->calc(born.first, order) ) kernels =
                                                splits[name]->getKernelVals();
  if ( kernels.find("base") != kernels.end() ) p += kernels["base"];
  // Reset again.
  splits[name]->splitInfo.clear();

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
  if (!dryrun) mecover = splits[name]->overhead(m2dipBef*xDau,
                                                state[iRad].id(), nFinal-1);
  p *= mecover;

  // Note: The additional factor 1/xCS for rescaling the initial flux is NOT
  // included, so that we can apply PDF ratios [x1 f(x1)] / [x0 f(x0) ] later.

  return p;

}

//--------------------------------------------------------------------------

bool DireSpace::allowedSplitting( const Event& state, int iRad, int iEmt) {

  bool isAP = state[iEmt].id() < 0;
  int idRad = state[iRad].id();
  int idEmt = state[iEmt].id();

  int colRad  = state[iRad].col();
  int acolRad = state[iRad].acol();
  int colEmt  = state[iEmt].col();
  int acolEmt = state[iEmt].acol();

  int colShared = (colRad  > 0 && colRad  == colEmt ) ? colRad
                : (acolRad > 0 && acolRad == acolEmt) ? acolRad : 0;

  // Only consider initial-state emissions.
  if ( state[iRad].isFinal() ) return false;

  // Gluon emission is allowed.
  if (idEmt == 21 && colShared > 0)
    return true;

  // Q -> G Q is allowed.
  if ( abs(idRad) < 10 && idRad == idEmt && colShared == 0)
    return true;

  // Gluon branching to quarks is allowed.
  if ( idRad == 21 && abs(idEmt) < 10
    && ( (isAP && acolEmt == acolRad) || (!isAP && colEmt == colRad) ) )
    return true;

  // Photon emission is allowed.
  if ( idEmt == 22 && abs(idRad) < 10)
    return true;

  // Photon emission is allowed.
  if ( idEmt == 22 && (abs(idRad) == 11 || abs(idRad) == 13
    || abs(idRad) == 15))
    return true;

  // Q -> A Q is allowed.
  if ( abs(idEmt) < 10 && idRad == idEmt && colShared > 0)
    return true;

  // L -> A L is allowed.
  if ( (abs(idEmt) == 11 || abs(idEmt) == 13 || abs(idEmt) == 15)
    && idRad == idEmt)
    return true;

  // Photon branching to quarks is allowed.
  if ( idRad == 22 && abs(idEmt) < 10 && idEmt == idRad && colShared > 0)
    return true;

  // Photon branching to leptons is allowed.
  if (idRad == 22 && (abs(idEmt) == 11 || abs(idEmt) == 13 || abs(idEmt) == 15)
    && idEmt == idRad)
    return true;

  // Z-boson emission is allowed.
  if ( idEmt == 23 && abs(idRad) < 10)
    return true;

  // Z-boson emission is allowed.
  if ( idEmt == 23 && (abs(idRad) == 11 || abs(idRad) == 13
    || abs(idRad) == 15))
    return true;

  return false;

}

//--------------------------------------------------------------------------

vector<int> DireSpace::getRecoilers( const Event& state, int iRad, int iEmt,
  string name) {
  // List of recoilers.
  return splits[name]->recPositions(state, iRad, iEmt);
}

//-------------------------------------------------------------------------

Event DireSpace::makeHardEvent( int iSys, const Event& state, bool isProcess) {

  bool hasSystems = !isProcess && partonSystemsPtr->sizeSys() > 0;
  int sizeSys     = (hasSystems) ? partonSystemsPtr->sizeSys() : 1;
  Event event     = Event();

  event.clear();
  event.init( "(hard process-modified)", particleDataPtr );
  event.clear();

  int in1 = 0;
  for ( int i = state.size()-1; i > 0; --i)
    if ( state[i].mother1() == 1 && state[i].mother2() == 0
      && (!hasSystems || partonSystemsPtr->getSystemOf(i,true) == iSys))
      {in1 = i; break;}
  if (in1 == 0) in1 = getInA(iSys);
  int in2 = 0;
  for ( int i = state.size()-1; i > 0; --i)
    if ( state[i].mother1() == 2 && state[i].mother2() == 0
      && (!hasSystems || partonSystemsPtr->getSystemOf(i,true) == iSys))
      {in2 = i; break;}
  if (in2 == 0) in2 = getInB(iSys);

  // Try to find incoming particle in other systems, i.e. if the current
  // system arose from a resonance decay.
  bool resonantIncoming = false;
  if ( in1 == 0 && in2 == 0 ) {
    int iParentInOther = 0;
    int nSys = partonSystemsPtr->sizeAll(iSys);
    for (int iInSys = 0; iInSys < nSys; ++iInSys){
      int iiNow = partonSystemsPtr->getAll(iSys,iInSys);
      bool hasOtherParent = false;
      for (int iOtherSys = 0; iOtherSys < sizeSys; ++iOtherSys){
        if (iOtherSys == iSys) continue;
        int nOtherSys = partonSystemsPtr->sizeAll(iOtherSys);
        for (int iInOtherSys = 0; iInOtherSys < nOtherSys; ++iInOtherSys){
          int iOtherNow = partonSystemsPtr->getAll(iOtherSys,iInOtherSys);
          if (state[iiNow].isAncestor(iOtherNow)) {
            iParentInOther = iOtherNow;
            hasOtherParent = true;
            break;
          }
        }
        if (hasOtherParent) break;
      }
      if (hasOtherParent) break;
    }
    in1 = iParentInOther;
    if (iParentInOther) resonantIncoming = true;
  }

  event.append(state[0]);
  event.append(state[1]);
  event[1].daughters(3,0);
  event.append(state[2]);
  event[2].daughters(4,0);

  // Attach the first incoming particle.
  event.append(state[in1]);
  event[3].mothers(1,0);
  if (resonantIncoming) event[3].status(-22);
  else event[3].status(-21);

  // Attach the second incoming particle.
  event.append(state[in2]);
  event[4].mothers(2,0);
  event[4].status(-21);

  for ( int i = 0; i < state.size(); ++i) {
    // Careful when builing the sub-events: A particle that is currently
    // intermediate in one system could be the pirogenitor of another
    // system, i.e. when resonance decays are present. In this case, the
    // intermediate particle in the current system should be final.
    bool isFin   = state[i].isFinal();
    bool isInSys = (partonSystemsPtr->getSystemOf(i) == iSys);

    bool isParentOfOther = false;
    if (!isFin && isInSys) {
      for (int iOtherSys = 0; iOtherSys < sizeSys; ++iOtherSys){
        if (iOtherSys == iSys) continue;
        double nSys = partonSystemsPtr->sizeAll(iOtherSys);
        for (int iInSys = 0; iInSys < nSys; ++iInSys){
          int iiNow = partonSystemsPtr->getAll(iOtherSys,iInSys);
          if (state[iiNow].isAncestor(i)) {isParentOfOther=true; break;}
        }
      }
    }

    if ( (isFin || isParentOfOther) && (!hasSystems || isInSys) ) {
      int iN = event.append(state[i]);
      event[iN].daughters(0,0);
      event[iN].mothers(3,4);
      int status = (state[i].statusAbs() == 22) ? state[i].statusAbs() : 23;
      if ( particleDataPtr->isResonance(state[i].id()) ) status = 22;
      event[iN].status(status);
    }
  }

  // Set daughters of initial particles.
  event[3].daughters(5,event.size()-1);
  event[4].daughters(5,event.size()-1);

  return event;

}

//-------------------------------------------------------------------------

// Check colour/flavour correctness of state.

bool DireSpace::validMomentum( const Vec4& p, int id, int status) {

  // Check for NaNs and INFs.
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
  if ( abs(id) == 6 || abs(id) > 22) mNow = p.mCalc();
  double errMass = abs(p.mCalc() - mNow) / max( 1.0, p.e());
  if ( errMass > mTolErr ) return false;

  // Check for negative energies.
  if ( p.e() < 0. ) return false;

  // Done
  return true;

}

//-------------------------------------------------------------------------

// Check colour/flavour correctness of state.

bool DireSpace::validEvent( const Event& state, bool isProcess ) {

  bool validColour  = true;
  bool validCharge  = true;
  bool validMomenta = true;

  bool hasSystems = !isProcess && partonSystemsPtr->sizeSys() > 0;
  int sizeSys     = (hasSystems) ? partonSystemsPtr->sizeSys() : 1;
  Event e         = Event();

  // Check for NaNs or INFs.
  for ( int i = 0; i < state.size(); ++i)
    if (isnan(state[i].p()) || isinf(state[i].p())) return false;

  for (int iSys = 0; iSys < sizeSys; ++iSys) {

    // Done if the state is already broken.
    if (!validColour || !validCharge ) break;

    e.clear();
    e.init( "(hard process-modified)", particleDataPtr );
    e.clear();
    e = makeHardEvent(iSys, state, isProcess);

    // Check if event is coloured
    for ( int i = 0; i < e.size(); ++i)
     // Check colour of quarks
     if ( e[i].isFinal() && e[i].colType() == 1
            // No corresponding anticolour in final state
         && ( FindCol(e[i].col(),vector<int>(1,i),e,1) == 0
            // No corresponding colour in initial state
           && FindCol(e[i].col(),vector<int>(1,i),e,2) == 0 )) {
       validColour = false;
       break;
     // Check anticolour of antiquarks
     } else if ( e[i].isFinal() && e[i].colType() == -1
            // No corresponding colour in final state
         && ( FindCol(e[i].acol(),vector<int>(1,i),e,2) == 0
            // No corresponding anticolour in initial state
           && FindCol(e[i].acol(),vector<int>(1,i),e,1) == 0 )) {
       validColour = false;
       break;
     // No uncontracted colour (anticolour) charge of gluons
     } else if ( e[i].isFinal() && e[i].colType() == 2
            // No corresponding anticolour in final state
         && ( FindCol(e[i].col(),vector<int>(1,i),e,1) == 0
            // No corresponding colour in initial state
           && FindCol(e[i].col(),vector<int>(1,i),e,2) == 0 )
            // No corresponding colour in final state
         && ( FindCol(e[i].acol(),vector<int>(1,i),e,2) == 0
            // No corresponding anticolour in initial state
           && FindCol(e[i].acol(),vector<int>(1,i),e,1) == 0 )) {
       validColour = false;
       break;
     }

    // Check charge sum in initial and final state
    double initCharge = e[3].charge() + e[4].charge();
    double finalCharge = 0.0;
    for(int i = 0; i < e.size(); ++i)
      if (e[i].isFinal()) finalCharge += e[i].charge();
    if (abs(initCharge-finalCharge) > 1e-12) validCharge = false;

    // Check if particles are on mass shell
    for ( int i = 0; i < e.size(); ++i) {
      if (i==3 || i==4 || e[i].isFinal()) {
        validMomenta = validMomenta
          && validMomentum(e[i].p(), e[i].id(), (e[i].isFinal() ? 1 : -1));
      }
    }

    // Check that overall pT is vanishing.
    Vec4 pSum(0.,0.,0.,0.);
    for ( int i = 0; i < e.size(); ++i) {
      if ( e[i].status() == -21
        || e[i].status() == -22 ) pSum -= e[i].p();
      if ( e[i].isFinal() )       pSum += e[i].p();
    }
    if ( abs(pSum.px()) > mTolErr || abs(pSum.py()) > mTolErr)
      validMomenta = false;
    if ( e[3].status() == -21
      && (abs(e[3].px()) > mTolErr || abs(e[3].py()) > mTolErr))
      validMomenta = false;
    if ( e[4].status() == -21
      && (abs(e[4].px()) > mTolErr || abs(e[4].py()) > mTolErr))
      validMomenta = false;

  } // Done with loop over systems.

  return (validColour && validCharge && validMomenta);

}

//-------------------------------------------------------------------------

bool DireSpace::validMotherDaughter( const Event& event ) {

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

//-------------------------------------------------------------------------

// Find index colour partner for input colour.

int DireSpace::FindCol(int col, vector<int> iExc, const Event& event,
  int type, int iSys) {

  int index = 0;

  int inA = 0, inB = 0;
  for (int i=event.size()-1; i > 0; --i) {
    if ( event[i].mother1() == 1 && event[i].status() != -31
      && event[i].status() != -34) { if (inA == 0) inA = i; }
    if ( event[i].mother1() == 2 && event[i].status() != -31
      && event[i].status() != -34) { if (inB == 0) inB = i; }
  }
  if (iSys >= 0) {inA = getInA(iSys); inB = getInB(iSys);}

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

//-------------------------------------------------------------------------

// Print the list of dipoles.

void DireSpace::list() const {

  // Header.
  cout << "\n --------  DIRE DireSpace Dipole Listing  ------------------"
       << "--------------------------------------------------------------"
       << "----------\n\n"
       << "   i    sys   side    rad    rec         pTmax   col  "
       << "        m2Dip          siblings        allowedIDs\n"
       << fixed << setprecision(3);

  // Loop over dipole list and print it.
  for (int i = 0; i < int(dipEnd.size()); ++i) {
    cout << scientific << setprecision(4)
         << setw(4) << i << " | "
         << setw(4) << dipEnd[i].system << " | "
         << setw(4) << dipEnd[i].side << " | "
         << setw(4) << dipEnd[i].iRadiator << " | "
         << setw(4) << dipEnd[i].iRecoiler << " | "
         << setw(11) << dipEnd[i].pTmax << " | "
         << setw(3) << dipEnd[i].colType << " | "
         << setw(12) << dipEnd[i].m2Dip << " | ";
    ostringstream os;
    os << dipEnd[i].iSiblings.listPos();
    cout << setw(15) << os.str() << " | ";
    os.str("");
    for (int j = 0; j < int(dipEnd[i].allowedEmissions.size()); ++j)
      os << setw(4) << dipEnd[i].allowedEmissions[j];
    cout << setw(15) << os.str() << endl;
  }

  // Done.
  cout << "\n --------  End DIRE DireSpace Dipole Listing  --------------"
       << "--------------------------------------------------------------"
       << "----------" << endl;

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

}

//--------------------------------------------------------------------------

// Function to calculate the correct alphaS/2*Pi value, including
// renormalisation scale variations + threshold matching.

double DireSpace::alphasNow( double pT2, double renormMultFacNow, int iSys ) {

  // Get beam for PDF alphaS, if necessary.
  BeamParticle* beam = (particleDataPtr->isHadron(beamAPtr->id()))
                     ? beamAPtr
                     : (particleDataPtr->isHadron(beamBPtr->id()) ? beamBPtr :
                        NULL );
  if (usePDFalphas && beam == NULL) beam = beamAPtr;
  double scale       = pT2*renormMultFacNow;
  scale              = max(scale, pT2min);

  // Get alphaS(k*pT^2) and subtractions.
  double asPT2pi      = (usePDFalphas && beam != NULL)
                      ? beam->alphaS(scale)  / (2.*M_PI)
                      : alphaS.alphaS(scale) / (2.*M_PI);

  // Get kernel order.
  int order = kernelOrder-1;
  // Use simple kernels for showering secondary scatterings.
  bool hasInA = (getInA(iSys) != 0);
  bool hasInB = (getInB(iSys) != 0);
  if (iSys != 0 && hasInA && hasInB) order = kernelOrderMPI-1;

  // Now find the necessary thresholds so that alphaS can be matched
  // correctly.
  double m2cNow(m2cPhys), m2bNow(m2bPhys);
  if ( !( (scale > m2cNow && pT2 < m2cNow)
       || (scale < m2cNow && pT2 > m2cNow) ) ) m2cNow = -1.;
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

// Auxiliary function to get number of flavours.

double DireSpace::getNF(double pT2) {
  double NF = 6.;
  BeamParticle* beam = (particleDataPtr->isHadron(beamAPtr->id()))
                     ? beamAPtr
                     : (particleDataPtr->isHadron(beamBPtr->id()) ? beamBPtr :
                        NULL );
  // Get current number of flavours.
  if ( !usePDFalphas || beam == NULL ) {
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
