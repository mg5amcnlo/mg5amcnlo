// DireSpace.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for the spacelike initial-state showers.
// DireSpaceEnd: radiating dipole end in ISR.
// DireSpace: handles the showering description.

#ifndef Pythia8_DireSpace_H
#define Pythia8_DireSpace_H

#define DIRE_SPACE_VERSION "2.002"

#include "Pythia8/Basics.h"
#include "Pythia8/SpaceShower.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/Event.h"
#include "Pythia8/Info.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PartonSystems.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StandardModel.h"
#include "Pythia8/UserHooks.h"
#include "Pythia8/MergingHooks.h"
#include "Pythia8/SimpleWeakShowerMEs.h"
#include "Pythia8/DireBasics.h"
#include "Pythia8/DireSplittingLibrary.h"
#include "Pythia8/DireWeightContainer.h"

namespace Pythia8 {

//==========================================================================

// Data on radiating dipole ends, only used inside DireSpace.

class DireSpaceEnd {

public:

  // Constructor.
  DireSpaceEnd( int systemIn = 0, int sideIn = 0, int iRadiatorIn = 0,
    int iRecoilerIn = 0, double pTmaxIn = 0., int colTypeIn = 0,
    int chgTypeIn = 0, int weakTypeIn = 0,  int MEtypeIn = 0,
    bool normalRecoilIn = true, int weakPolIn = 0,
    DireSingleColChain iSiblingsIn = DireSingleColChain(),
    vector<int> iSpectatorIn = vector<int>(),
    vector<double> massIn = vector<double>(),
    vector<int> allowedIn = vector<int>() ) :
    system(systemIn), side(sideIn), iRadiator(iRadiatorIn),
    iRecoiler(iRecoilerIn), pTmax(pTmaxIn), colType(colTypeIn),
    chgType(chgTypeIn), weakType(weakTypeIn), MEtype(MEtypeIn),
    normalRecoil(normalRecoilIn), weakPol(weakPolIn), nBranch(0),
    pT2Old(0.), zOld(0.5), mass(massIn), iSpectator(iSpectatorIn),
    allowedEmissions(allowedIn), iSiblings(iSiblingsIn) {
    idDaughter = idMother = idSister = iFinPol = 0;
    x1  = x2 = m2Dip = pT2 = z = xMo = Q2 = mSister = m2Sister = pT2corr
        = pT2Old = zOld = asymPol = sa1 = xa = pT2start = pT2stop = 0.;
    mRad = m2Rad = mRec = m2Rec = mDip = 0.;
    phi = phia1 = -1.;
  }

  // Explicit copy constructor.
  DireSpaceEnd( const DireSpaceEnd& dip )
    : system(dip.system), side(dip.side), iRadiator(dip.iRadiator),
      iRecoiler(dip.iRecoiler), pTmax(dip.pTmax), colType(dip.colType),
      chgType(dip.chgType), weakType(dip.weakType), MEtype(dip.MEtype),
      normalRecoil(dip.normalRecoil), weakPol(dip.weakPol),
      nBranch(dip.nBranch), idDaughter(dip.idDaughter), idMother(dip.idMother),
      idSister(dip.idSister), iFinPol(dip.iFinPol), x1(dip.x1), x2(dip.x2),
      m2Dip(dip.m2Dip), pT2(dip.pT2), z(dip.z), xMo(dip.xMo), Q2(dip.Q2),
      mSister(dip.mSister), m2Sister(dip.m2Sister), pT2corr(dip.pT2corr),
      pT2Old(dip.pT2Old), zOld(dip.zOld), asymPol(dip.asymPol), phi(dip.phi),
      pT2start(dip.pT2start), pT2stop(dip.pT2stop),
      mRad(dip.mRad), m2Rad(dip.m2Rad), mRec(dip.mRec), m2Rec(dip.m2Rec),
      mDip(dip.mDip), sa1(dip.sa1), xa(dip.xa),
      phia1(dip.phia1), mass(dip.mass), iSpectator(dip.iSpectator),
      allowedEmissions(dip.allowedEmissions), iSiblings(dip.iSiblings) {}

  // Assignment operator.
  DireSpaceEnd & operator=(const DireSpaceEnd &s) { if (this != &s)
    { system = s.system; side = s.side; iRadiator = s.iRadiator;
      iRecoiler = s.iRecoiler; pTmax = s.pTmax; colType = s.colType;
      chgType = s.chgType; weakType = s.weakType; MEtype = s.MEtype;
      normalRecoil = s.normalRecoil; weakPol = s.weakPol;
      nBranch = s.nBranch; idDaughter = s.idDaughter; idMother = s.idMother;
      idSister = s.idSister; iFinPol = s.iFinPol; x1 = s.x1; x2 = s.x2;
      m2Dip = s.m2Dip; pT2 = s.pT2; z = s.z; xMo = s.xMo; Q2 = s.Q2;
      mSister = s.mSister; m2Sister = s.m2Sister; pT2corr = s.pT2corr;
      pT2Old = s.pT2Old; zOld = s.zOld; asymPol = s.asymPol; phi = s.phi;
      pT2start = s.pT2start; pT2stop = s.pT2stop;
      mRad = s.mRad; m2Rad = s.m2Rad; mRec = s.mRec; m2Rec = s.m2Rec;
      mDip = s.mDip; sa1 = s.sa1; xa = s.xa; phia1 = s.phia1; mass = s.mass;
      iSpectator = s.iSpectator; allowedEmissions = s.allowedEmissions;
      iSiblings = s.iSiblings;} return *this; }

  // Store values for trial emission.
  void store( int idDaughterIn, int idMotherIn, int idSisterIn,
    double x1In, double x2In, double m2DipIn, double pT2In, double zIn,
    double sa1In, double xaIn, double xMoIn, double Q2In, double mSisterIn,
    double m2SisterIn, double pT2corrIn, double phiIn = -1.,
    double phia1In = 1.) {
    idDaughter = idDaughterIn; idMother = idMotherIn;
    idSister = idSisterIn; x1 = x1In; x2 = x2In; m2Dip = m2DipIn;
    pT2 = pT2In; z = zIn; sa1 = sa1In; xa = xaIn; xMo = xMoIn; Q2 = Q2In;
    mSister = mSisterIn; m2Sister = m2SisterIn; pT2corr = pT2corrIn;
    mRad = m2Rad = mRec = m2Rec = mDip = 0.;
    phi = phiIn; phia1 = phia1In; }

  // Basic properties related to evolution and matrix element corrections.
  int    system, side, iRadiator, iRecoiler;
  double pTmax;
  int    colType, chgType, weakType, MEtype;
  bool   normalRecoil;
  int    weakPol;

  // Properties specific to current trial emission.
  int    nBranch, idDaughter, idMother, idSister, iFinPol;
  double x1, x2, m2Dip, pT2, z, xMo, Q2, mSister, m2Sister, pT2corr,
         pT2Old, zOld, asymPol, phi, pT2start, pT2stop,
         mRad, m2Rad, mRec, m2Rec, mDip;

  // Properties of 1->3 splitting.
  double sa1, xa, phia1;

  // Stored masses.
  vector<double> mass;

  // Extended list of recoilers.
  vector<int> iSpectator;

  vector<int> allowedEmissions;

  // List of allowed emissions (to avoid double-counting, since one
  // particle can be part of many different dipoles.
  void appendAllowedEmt( int id) {
    if ( find(allowedEmissions.begin(), allowedEmissions.end(), id)
        == allowedEmissions.end() ) { allowedEmissions.push_back(id);}
  }
  void clearAllowedEmt() { allowedEmissions.resize(0); }
  bool canEmit() { return int(allowedEmissions.size() > 0); }

  void init(const Event& state) {
    mRad   = state[iRadiator].m();
    mRec   = state[iRecoiler].m();
    mDip   = sqrt( abs(2. * state[iRadiator].p() * state[iRecoiler].p()));
    m2Rad  = pow2(mRad);
    m2Rec  = pow2(mRec);
    m2Dip  = pow2(mDip);
  }

  void list() const {
    // Header.
    cout << "\n --------  DireSpaceEnd Listing  -------------- \n"
         << "\n    syst  side   rad   rec       pTmax  col  chg   ME rec \n"
         << fixed << setprecision(3);
    cout << setw(6) << system
         << setw(6) << side      << setw(6)  << iRadiator
         << setw(6) << iRecoiler << setw(12) << pTmax
         << setw(5) << colType   << setw(5)  << chgType
         << setw(5) << MEtype    << setw(4)
         << normalRecoil
         << setw(12) << m2Dip;
    for (int j = 0; j < int(allowedEmissions.size()); ++j)
      cout << setw(5) << allowedEmissions[j] << " ";
    cout << endl;
   // Done.
    cout << "\n --------  End DireSpaceEnd Listing  ------------"
         << "-------------------------------------------------------" << endl;
  }

  DireSingleColChain iSiblings;
  void setSiblings(DireSingleColChain s) { clearSiblings(); iSiblings = s; }
  void clearSiblings() { iSiblings.clear(); }

};

//==========================================================================

// The DireSpace class does spacelike showers.

class DireSpace : public SpaceShower {

public:

  // Constructor.
  DireSpace() {
    beamOffset          = 0;
    pTdampFudge         = 0.;
    mergingHooksPtr     = nullptr;
    splittingsPtr       = nullptr;
    weights             = 0;
    direInfoPtr         = nullptr;
    beamAPtr = beamBPtr = nullptr;
    printBanner         = true;
    nWeightsSave        = 0;
    isInitSave          = false;
    nMPI                = 0;
    usePDFalphas        = false;
    usePDF              = true;
    useSystems          = true;
    suppressLargeMECs   = false;
  }

  DireSpace( MergingHooksPtr mergingHooksPtrIn, PartonVertexPtr ) :
      pTdampFudge(0.), mc(0.), mb(0.), m2c(0.), m2b(0.), m2cPhys(0.),
      m2bPhys(0.), renormMultFac(0.), factorMultFac(0.), fixedFacScale2(0.),
      alphaSvalue(0.), alphaS2pi(0.), Lambda3flav(0.), Lambda4flav(0.),
      Lambda5flav(0.), Lambda3flav2(0.), Lambda4flav2(0.), Lambda5flav2(0.),
      pT0Ref(0.), ecmRef(0.), ecmPow(0.), pTmin(0.), sCM(0.), eCM(0.), pT0(0.),
      pT20(0.), pT2min(0.), m2min(0.), mTolErr(0.), pTmaxFudgeMPI(0.),
      strengthIntAsym(0.), pT2minVariations(0.), pT2minEnhance(0.),
      pT2minMECs(0.), Q2minMECs(0.),
      alphaS2piOverestimate(0.), usePDFalphas(false), usePDFmasses(false),
      useSummedPDF(false), usePDF(true), useSystems(true),
      useGlobalMapIF(false), forceMassiveMap(false), useMassiveBeams(false),
      suppressLargeMECs(false) {
      mergingHooksPtr   = mergingHooksPtrIn;
      beamOffset        = 0;
      pTdampFudge       = 0.;
      splittingsPtr     = nullptr;
      weights           = 0;
      direInfoPtr       = nullptr;
      printBanner       = true;
      nWeightsSave      = 0;
      isInitSave        = false;
      nMPI = 0;
      beamAPtr = 0;
      beamBPtr = 0;
  }

  // Destructor.
  virtual ~DireSpace() {}

  // Initialize generation. Possibility to force re-initialization by hand.
  virtual void init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn);

  bool initSplits() {
    if (splittingsPtr) splits = splittingsPtr->getSplittings();
    return (splits.size() > 0);
  }

  // Initialize various pointers.
  // (Separated from rest of init since not virtual.)
  void reinitPtr(Info* infoPtrIn,  MergingHooksPtr mergingHooksPtrIn,
    DireSplittingLibrary* splittingsPtrIn, DireInfo* direInfoPtrIn) {
       infoPtr          = infoPtrIn;
       settingsPtr      = infoPtr->settingsPtr;
       particleDataPtr  = infoPtr->particleDataPtr;
       rndmPtr          = infoPtr->rndmPtr;
       partonSystemsPtr = infoPtr->partonSystemsPtr;
       userHooksPtr     = infoPtr->userHooksPtr;
       mergingHooksPtr  = mergingHooksPtrIn;
       splittingsPtr    = splittingsPtrIn;
       direInfoPtr      = direInfoPtrIn;
  }

  void initVariations();

  // Reset parton shower.
  void clear();

  void setWeightContainerPtr(DireWeightContainer* weightsIn) {
    weights = weightsIn;}

  // Find whether to limit maximum scale of emissions, and whether to dampen.
  virtual bool limitPTmax( Event& event, double Q2Fac = 0.,
    double Q2Ren = 0.);

  // Potential enhancement factor of pTmax scale for hardest emission.
  virtual double enhancePTmax() const {return pTmaxFudge;}

  // Prepare system for evolution; identify ME.
  void resetWeights();
  virtual void prepare( int iSys, Event& event, bool limitPTmaxIn = true);

  // Update dipole list after each FSR emission.
  // Usage: update( iSys, event).
  virtual void update( int , Event&, bool = false);

  // Update dipole list after initial-initial splitting.
  void updateAfterII( int iSysSelNow, int sideNow, int iDipSelNow,
    int eventSizeOldNow, int systemSizeOldNow, Event& event, int iDaughter,
    int iMother, int iSister, int iNewRecoiler, double pT2, double xNew);

  // Update dipole list after initial-initial splitting.
  void updateAfterIF( int iSysSelNow, int sideNow, int iDipSelNow,
    int eventSizeOldNow, int systemSizeOldNow, Event& event, int iDaughter,
    int iRecoiler, int iMother, int iSister, int iNewRecoiler, int iNewOther,
    double pT2, double xNew);

  // Select next pT in downwards evolution.
  virtual double pTnext( Event& event, double pTbegAll, double pTendAll,
    int nRadIn = -1, bool = false);
  double newPoint( const Event& event);

  // Select next pT in downwards evolution, based only on dipole mass and
  // incoming momentum fraction.
  double pTnext( vector<DireSpaceEnd> dipEnds, Event event, double pTbegAll,
    double pTendAll, double m2dip, int type, double s = -1., double x = -1.);
  double noEmissionProbability( double pTbegAll, double pTendAll, double m2dip,
    int id, int type, double s = -1., double x = -1.);

  // Setup branching kinematics.
  virtual bool branch( Event& event);

  bool branch_II( Event& event, bool = false,
    DireSplitInfo* split = nullptr);
  bool branch_IF( Event& event, bool = false,
    DireSplitInfo* split = nullptr);

  // Setup clustering kinematics.
  virtual Event clustered( const Event& state, int iRad, int iEmt, int iRecAft,
    string name) {
    pair <Event, pair<int,int> > reclus
      = clustered_internal(state, iRad, iEmt, iRecAft, name);
    if (reclus.first.size() > 0)
      reclus.first[0].mothers(reclus.second.first,reclus.second.second);
    return reclus.first;
  }
  pair <Event, pair<int,int> > clustered_internal( const Event& state,
    int iRad, int iEmt, int iRecAft, string name);
  bool cluster_II( const Event& state, int iRad,
    int iEmt, int iRecAft, int idRadBef, Particle& radBef, Particle& recBef,
    Event& partialState);
  bool cluster_IF( const Event& state, int iRad,
    int iEmt, int iRecAft, int idRadBef, Particle& radBef, Particle& recBef,
    Event& partialState);

  // Return ordering variable.
  // From Pythia version 8.215 onwards no longer virtual.
  double pT2Space ( const Particle& rad, const Particle& emt,
    const Particle& rec) {
    if (rec.isFinal()) return pT2_IF(rad,emt,rec);
    return pT2_II(rad,emt,rec);
  }

  double pT2_II ( const Particle& rad, const Particle& emt,
    const Particle& rec);
  double pT2_IF ( const Particle& rad, const Particle& emt,
    const Particle& rec);

  // Return auxiliary variable.
  // From Pythia version 8.215 onwards no longer virtual.
  double zSpace ( const Particle& rad, const Particle& emt,
    const Particle& rec) {
    if (rec.isFinal()) return z_IF(rad,emt,rec);
    return z_II(rad,emt,rec);
  }

  double z_II ( const Particle& rad, const Particle& emt,
    const Particle& rec);
  double z_IF ( const Particle& rad, const Particle& emt,
    const Particle& rec);

  double m2dipSpace ( const Particle& rad, const Particle& emt,
    const Particle& rec) {
    if (rec.isFinal()) return m2dip_IF(rad,emt,rec);
    return m2dip_II(rad,emt,rec);
  }
  double m2dip_II ( const Particle& rad, const Particle& emt,
    const Particle& rec);
  double m2dip_IF ( const Particle& rad, const Particle& emt,
    const Particle& rec);

  // From Pythia version 8.218 onwards.
  // Return the evolution variable.
  // Usage: getStateVariables( const Event& event,  int iRad, int iEmt,
  //                   int iRec, string name)
  // Important note:
  // - This map must contain an entry for the shower evolution variable,
  //   specified with key "t".
  // - This map must contain an entry for the shower evolution variable from
  //   which the shower would be restarted after a branching. This entry
  //   must have key "tRS",
  // - This map must contain an entry for the argument of \alpha_s used
  //   for the branching. This entry must have key "scaleAS".
  // - This map must contain an entry for the argument of the PDFs used
  //   for the branching. This entry must have key "scalePDF".
  virtual map<string, double> getStateVariables (const Event& state,
    int rad, int emt, int rec, string name);

  // From Pythia version 8.215 onwards.
  // Check if attempted clustering is handled by timelike shower
  // Usage: isSpacelike( const Event& event,  int iRad, int iEmt,
  //                   int iRec, string name)
  virtual bool isSpacelike(const Event& state, int iRad, int, int, string)
    { return !state[iRad].isFinal(); }

  // From Pythia version 8.215 onwards.
  // Return a string identifier of a splitting.
  // Usage: getSplittingName( const Event& event, int iRad, int iEmt, int iRec)
  virtual vector<string> getSplittingName( const Event& state, int iRad,
    int iEmt,int) { return splittingsPtr->getSplittingName(state,iRad,iEmt); }

  // From Pythia version 8.215 onwards.
  // Return the splitting probability.
  // Usage: getSplittingProb( const Event& event, int iRad, int iEmt, int iRec)
  virtual double getSplittingProb( const Event& state, int iRad,
    int iEmt, int iRecAft, string);

  virtual bool allowedSplitting( const Event& state, int iRad, int iEmt);

  virtual vector<int> getRecoilers( const Event& state, int iRad, int iEmt,
    string name);

  virtual double getCoupling( double mu2Ren, string name) {
    if (splits.find(name) != splits.end())
      return splits[name]->coupling(-1.,mu2Ren, 0., 1.);
    return 1.;
  }

  bool isSymmetric( string name, const Particle* rad, const Particle* emt) {
    if (splits.find(name) != splits.end())
      return splits[name]->isSymmetric(rad,emt);
    return false;
  }

  // Auxiliary function to return the position of a particle.
  // Should go int Event class eventually!
  int FindParticle( const Particle& particle, const Event& event,
    bool checkStatus = true );

  // Print dipole list; for debug mainly.
  virtual void list() const;

  Event makeHardEvent( int iSys, const Event& state, bool isProcess = false );

  // Check that particle has sensible momentum.
  bool validMomentum( const Vec4& p, int id, int status);

  // Check colour/flavour correctness of state.
  bool validEvent( const Event& state, bool isProcess = false );

  // Check that mother-daughter-relations are correctly set.
  bool validMotherDaughter( const Event& state );

  // Find index colour partner for input colour.
  int FindCol(int col, vector<int> iExc, const Event& event, int type,
    int iSys = -1);

  // Pointers to the two incoming beams.
  BeamParticle*  getBeamA () { return beamAPtr; }
  BeamParticle*  getBeamB () { return beamBPtr; }

  // Pointer to Standard Model couplings.
  CoupSM* getCoupSM () { return coupSMPtr; }

  // Function to calculate the correct alphaS/2*Pi value, including
  // renormalisation scale variations + threshold matching.
  double alphasNow( double pT2, double renormMultFacNow = 1., int iSys = 0 );

  bool isInit() { return isInitSave; }

  // Function to calculate the absolute phase-sace boundary for emissions.
  double m2Max (int iDip, const Event& state) {
    if ( state[dipEnd[iDip].iRecoiler].isFinal()
      && state[dipEnd[iDip].iRadiator].isFinal())
      return dipEnd[iDip].m2Dip;
    int iSys = dipEnd[iDip].system;
    int inA = partonSystemsPtr->getInA(iSys);
    int inB = partonSystemsPtr->getInB(iSys);
    double x = 1.;
    int iRad(dipEnd[iDip].iRadiator), iRecNow(dipEnd[iDip].iRecoiler);
    if (hasPDF(state[iRad].id()) && iRad == inA)
      x *= state[inA].pPos() / state[0].m();
    if (hasPDF(state[iRad].id()) && iRad == inB)
      x *= state[inB].pNeg() / state[0].m();
    if (hasPDF(state[iRecNow].id()) && iRecNow == inA)
      x *= state[inA].pPos() / state[0].m();
    if (hasPDF(state[iRecNow].id()) && iRecNow == inB)
      x *= state[inB].pNeg() / state[0].m();
    return dipEnd[iDip].m2Dip/x;
  }

  bool dryrun;

private:

  friend class DireTimes;

  // Number of times the same error message is repeated, unless overridden.
  static const int TIMESTOPRINT;

  // Allow conversion from mb to pb.
  static const double CONVERTMB2PB;

  // Colour factors.
  //static const double CA, CF, TR, NC;
  double CA, CF, TR, NC;

  // Store common beam quantities.
  int    idASave, idBSave;

protected:

  // Store properties to be returned by methods.
  int    iSysSel;
  double pTmaxFudge;

private:

  // Constants: could only be changed in the code itself.
  static const int    MAXLOOPTINYPDF;
  static const double MCMIN, MBMIN, CTHRESHOLD, BTHRESHOLD, EVALPDFSTEP,
         TINYPDF, TINYKERNELPDF, TINYPT2, HEAVYPT2EVOL, HEAVYXEVOL,
         EXTRASPACEQ, LAMBDA3MARGIN, PT2MINWARN, LEPTONXMIN, LEPTONXMAX,
         LEPTONPT2MIN, LEPTONFUDGE, HEADROOMQ2Q, HEADROOMQ2G,
         HEADROOMG2G, HEADROOMG2Q, TINYMASS,
         PT2_INCREASE_OVERESTIMATE, KERNEL_HEADROOM;
  static const double DPHI_II, DPHI_IF;
  static const double G2QQPDFPOW1, G2QQPDFPOW2;

  // Initialization data, normally only set once.
  bool   isInitSave, doQCDshower, doQEDshowerByQ, doQEDshowerByL,
         useSamePTasMPI, doMEcorrections, doMEafterFirst, doPhiPolAsym,
         doPhiIntAsym, doRapidityOrder, useFixedFacScale, doSecondHard,
         canVetoEmission, hasUserHooks, alphaSuseCMW, printBanner, doTrialNow;
  int    pTmaxMatch, pTdampMatch, alphaSorder, alphaSnfmax,
         nQuarkIn, enhanceScreening, nFinalMax, nFinalMaxMECs, kernelOrder,
         kernelOrderMPI, nWeightsSave, nMPI, asScheme;
  double pTdampFudge, mc, mb, m2c, m2b, m2cPhys, m2bPhys, renormMultFac,
         factorMultFac, fixedFacScale2, alphaSvalue, alphaS2pi, Lambda3flav,
         Lambda4flav, Lambda5flav, Lambda3flav2, Lambda4flav2, Lambda5flav2,
         pT0Ref, ecmRef, ecmPow, pTmin, sCM, eCM, pT0, pT20,
         pT2min, m2min, mTolErr, pTmaxFudgeMPI, strengthIntAsym,
         pT2minVariations, pT2minEnhance, pT2minMECs, Q2minMECs;
  double alphaS2piOverestimate;
  bool  usePDFalphas, usePDFmasses, useSummedPDF,  usePDF, useSystems,
        useGlobalMapIF, forceMassiveMap, useMassiveBeams, suppressLargeMECs;

  unordered_map<int,double> pT2cutSave;
  double pT2cut(int id) {
    if (pT2cutSave.find(id) != pT2cutSave.end()) return pT2cutSave[id];
    // Else return maximal value.
    double ret = 0.;
    for ( unordered_map<int,double>::iterator it = pT2cutSave.begin();
      it != pT2cutSave.end(); ++it ) ret = max(ret, it->second);
    return ret;
  }
  double pT2cutMax(DireSpaceEnd* dip) {
    double ret = 0.;
    for (int i=0; i < int(dip->allowedEmissions.size()); ++i)
      ret = max( ret, pT2cut(dip->allowedEmissions[i]));
    return ret;
  }
  double pT2cutMin(DireSpaceEnd* dip) {
    double ret = 1e15;
    for (int i=0; i < int(dip->allowedEmissions.size()); ++i)
      ret = min( ret, pT2cut(dip->allowedEmissions[i]));
    return ret;
  }

  bool doDecaysAsShower;

  // alphaStrong and alphaEM calculations.
  AlphaStrong alphaS;

  // Some current values.
  bool   sideA, dopTlimit1, dopTlimit2, dopTdamp;
  int    iNow, iRec, idDaughter, nRad, idResFirst, idResSecond;
  double xDaughter, x1Now, x2Now, m2Dip, m2Rec, pT2damp, pTbegRef, pdfScale2;

  // List of emissions in different sides in different systems:
  vector<int> nRadA,nRadB;

  // All dipole ends
  vector<DireSpaceEnd> dipEnd;

  // Pointers to the current and hardest (so far) dipole ends.
  int iDipNow, iSysNow;
  DireSpaceEnd* dipEndNow;
  DireSplitInfo splitInfoSel;
  DireSplitting* splittingSel;
  int iDipSel;
  DireSpaceEnd* dipEndSel;
  unordered_map<string,double> kernelSel, kernelNow;
  double auxSel, overSel, boostSel, auxNow, overNow, boostNow;

  void setupQCDdip( int iSys, int side, int colTag, int colSign,
    const Event& event, int MEtype, bool limitPTmaxIn);

  void getGenDip( int iSys, int side, const Event& event,
    bool limitPTmaxIn, vector<DireSpaceEnd>& dipEnds );

  void getQCDdip( int iRad, int colTag, int colSign,
    const Event& event, vector<DireSpaceEnd>& dipEnds);

  // Function to set up and append a new dipole.
  bool appendDipole( const Event& state, int sys, int side,
    int iRad, int iRecNow, double pTmax, int colType,
    int chgType, int weakType, int MEtype, bool normalRecoil,
    int weakPolIn, vector<int> iSpectatorIn, vector<double> massIn,
    vector<DireSpaceEnd>& dipEnds);

  vector<int> sharedColor(const Particle& rad, const Particle& rec);

  // Function to set up and append a new dipole.
  void saveSiblings(const Event& state, int iSys = -1);
  void updateDipoles(const Event& state, int iSys = -1);
  bool updateAllowedEmissions( const Event& state, DireSpaceEnd* dip);
  bool appendAllowedEmissions( const Event& state, DireSpaceEnd* dip);

  // Flag for failure in branch(...) that will force a retry of parton level.
  bool doRestart() const {return false;}
  // Tell if latest scattering was a gamma->qqbar.
  bool wasGamma2qqbar() { return false; }
  // Tell whether ISR has done a weak emission.
  bool getHasWeaklyRadiated() {return false;}
  int system() const { return iSysSel;}

  // Evolve a QCD dipole end.
  void pT2nextQCD( double pT2begDip, double pT2endDip,
    DireSpaceEnd& dip, Event& event, double pT2endForce = -1.,
    double pT2freeze = 0., bool forceBranching = false);
  bool pT2nextQCD_II( double pT2begDip, double pT2endDip,
    DireSpaceEnd& dip, Event& event, double pT2endForce = -1.,
    double pT2freeze = 0., bool forceBranching = false);
  bool pT2nextQCD_IF( double pT2begDip, double pT2endDip,
    DireSpaceEnd& dip, Event& event, double pT2endForce = -1.,
    double pT2freeze = 0., bool forceBranching = false);

  double tNextQCD( DireSpaceEnd*, double overestimateInt,
    double tOld, double tMin, double tFreeze=0., int algoType = 0);
  bool zCollNextQCD( DireSpaceEnd* dip, double zMin, double zMax,
    double tMin = 0., double tMax = 0.);
  bool virtNextQCD( DireSpaceEnd* dip, double tMin, double tMax,
    double zMin =-1., double zMax =-1.);

  // Function to determine how often the integrated overestimate should be
  // recalculated.
  double evalpdfstep(int idRad, double pT2, double m2cp = -1.,
    double m2bp = -1.) {
    double ret = 0.2;
    if (m2cp < 0.) m2cp = particleDataPtr->m0(4);
    if (m2bp < 0.) m2bp = particleDataPtr->m0(5);
    // More steps close to the thresholds.
    if ( abs(idRad) == 4 && pT2 < 1.2*m2cp && pT2 > m2cp) ret = 1.0;
    if ( abs(idRad) == 5 && pT2 < 1.2*m2bp && pT2 > m2bp) ret = 1.0;
    return ret;
  }

  double tinypdf( double x) {
    double xref = 0.01;
    return TINYPDF*log(1-x)/log(1-xref);
  }

  // Function to check if id is part of the incoming hadron state.
  bool hasPDF (int id) {
    if ( !usePDF )                          return false;
    if ( particleDataPtr->colType(id) != 0) return true;
    if ( particleDataPtr->isLepton(id)
      && settingsPtr->flag("PDF:lepton"))   return true;
    return false;
  }

  // Wrapper around PDF calls.
  double getXPDF( int id, double x, double t, int iSys = 0,
    BeamParticle* beam = nullptr, bool finalRec = false, double z = 0.,
    double m2dip = 0.) {
    // Return one if no PDF should be used.
    if (!hasPDF(id)) return 1.0;
    // Else get PDF from beam particle.
    BeamParticle* b = beam;
    if (b == nullptr) {
      if (beamAPtr != nullptr || beamBPtr != nullptr) {
        b = (beamAPtr != nullptr && particleDataPtr->isHadron(beamAPtr->id()))
            ? beamAPtr
          : (beamBPtr != nullptr && particleDataPtr->isHadron(beamBPtr->id()))
            ? beamBPtr : nullptr;
      }
      if (b == nullptr && beamAPtr != nullptr) *beam = *beamAPtr;
      if (b == nullptr && beamBPtr != nullptr) *beam = *beamBPtr;
    }

    double scale2 = t;
    if (asScheme == 2 && z != 0) {
      if (!finalRec) {
        double xcs = (z * (1.-z) - t/m2dip) / (1.-z);
        double vcs = t/m2dip / (1.-z);
        double sab = m2dip/xcs;
        double saj = vcs*sab;
        double sjb = sab-saj-m2dip;
        scale2= abs(saj*sjb/sab);
      } else {
        double xcs = z;
        double ucs = t/m2dip / (1.-z);
        scale2 = (1-xcs)/xcs*ucs/(1-ucs)*m2dip;
      }
    }

    double ret =  (useSummedPDF) ? b->xf(id, x, scale2)
                                 : b->xfISR(iSys,id, x, scale2);
    // Done.
    return ret;
  }

  // Functions to extract beams w/o requiring parton systems pointer.
  int getInA ( int sys, const Event& state = Event() ) {
    if (useSystems) return partonSystemsPtr->getInA(sys);
    int inA = 0;
    for (int i=0; i < state.size(); ++i)
      if (state[i].mother1() == 1) {inA = i; break; }
    return inA;
  }
  int getInB ( int sys, const Event& state = Event() ) {
    if (useSystems) return partonSystemsPtr->getInB(sys);
    int inB = 0;
    for (int i=0; i < state.size(); ++i)
      if (state[i].mother1() == 2) {inB = i; break; }
    return inB;
  }


  DireSplittingLibrary* splittingsPtr;

  // Number of proposed splittings in hard scattering systems.
  unordered_map<int,int> nProposedPT;

  // Return headroom factors for integrated/differential overestimates.
  double overheadFactors( string, int, bool, double, double);
  double enhanceOverestimateFurther( string, int, double );

  // Function to fill map of integrated overestimates.
  void getNewOverestimates( int, DireSpaceEnd*, const Event&, double,
    double, double, double, multimap<double,string>& );

  // Function to fill map of integrated overestimates.
  double getPDFOverestimates( int, double, double, string, bool, double, int&,
    int&);

  // Function to sum all integrated overestimates.
  void addNewOverestimates( multimap<double,string>, double&);

  // Function to attach the correct alphaS weights to the kernels.
  void alphasReweight(double t, double talpha, int iSys, bool forceFixedAs,
    double& weight, double& fullWeight, double& overWeight,
    double renormMultFacNow);

  // Function to evaluate the accept-probability, including picking of z.
  void getNewSplitting( const Event&, DireSpaceEnd*, double, double, double,
    double, double, int, string, bool, int&, int&, double&, double&,
    unordered_map<string,double>&, double&);

  pair<bool, pair<double,double> > getMEC ( const Event& state,
    DireSplitInfo* splitInfo);
  bool applyMEC ( const Event& state, DireSplitInfo* splitInfo,
    vector<Event> auxEvent = vector<Event>() );

  // Get particle masses.
  double getMass(int id, int strategy, double mass = 0.) {
    BeamParticle& beam = ( particleDataPtr->isHadron(beamAPtr->id()) )
                       ? *beamAPtr : *beamBPtr;
    bool usePDFmass = usePDFmasses
      && (toLower(settingsPtr->word("PDF:pSet")).find("lhapdf")
         != string::npos);
    double mRet = 0.;
    // Parton masses.
    if ( particleDataPtr->colType(id) != 0) {
      if (strategy == 1) mRet = particleDataPtr->m0(id);
      if (strategy == 2 &&  usePDFmass) mRet = beam.mQuarkPDF(id);
      if (strategy == 2 && !usePDFmass) mRet = particleDataPtr->m0(id);
      if (strategy == 3) mRet = mass;
      if (mRet < TINYMASS) mRet = 0.;
    // Masses of other particles.
    } else {
      mRet = particleDataPtr->m0(id);
      if (strategy == 3) mRet = mass;
      if (mRet < TINYMASS) mRet = 0.;
    }
    return pow2(max(0.,mRet));
  }

  // Check if variables are in allowed phase space.
  bool inAllowedPhasespace(int kinType, double z, double pT2, double m2dip,
    double xOld, int splitType = 0, double m2RadBef = 0.,
    double m2r = 0.,  double m2s = 0., double m2e = 0.,
    vector<double> aux = vector<double>());

  // Function to attach the correct alphaS weights to the kernels.
  // Auxiliary function to get number of flavours.
  double getNF(double pT2);

  // Auxiliary functions to get beta function coefficients.
  double beta0 (double NF)
    { return 11./6.*CA - 2./3.*NF*TR; }
  double beta1 (double NF)
    { return 17./6.*pow2(CA) - (5./3.*CA+CF)*NF*TR; }
  double beta2 (double NF)
    { return 2857./432.*pow(CA,3)
    + (-1415./216.*pow2(CA) - 205./72.*CA*CF + pow2(CF)/4.) *TR*NF
    + ( 79.*CA + 66.*CF)/108.*pow2(TR*NF); }

  // Identifier of the splitting
  string splittingNowName, splittingSelName;

  // Weighted shower book-keeping.
  unordered_map<string, map<double,double> > acceptProbability;
  unordered_map<string, multimap<double,double> > rejectProbability;

public:

  DireWeightContainer* weights;
  DireInfo* direInfoPtr;
  // List of splitting kernels.
  unordered_map<string, DireSplitting* > splits;

private:

  bool doVariations;

  // Dynamically adjustable overestimate factors.
  unordered_map<string, double > overhead;
  void scaleOverheadFactor(string name, double scale) {
    overhead[name] *= scale;
    return;
  }
  void resetOverheadFactors() {
    for ( unordered_map<string,double>::iterator it = overhead.begin();
      it != overhead.end(); ++it )
      it->second = 1.0;
    return;
  }

  // Map to store some settings, to be passes to splitting kernels.
  unordered_map<string,bool> bool_settings;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_DireSpace_H
