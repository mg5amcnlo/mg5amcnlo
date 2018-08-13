// DireTimes.h is a part of the DIRE plugin to the PYTHIA event generator.
// Copyright (C) 2018 Stefan Prestel.

// Header file for the timelike final-state showers.
// DireTimesEnd: data on a radiating dipole end.
// DireTimes: handles the showering description.

#ifndef Pythia8_DireTimes_H
#define Pythia8_DireTimes_H

#define DIRE_TIMES_VERSION "2.002"

#include "Pythia8/Basics.h"
#include "Pythia8/TimeShower.h"
#include "Pythia8/Pythia.h"
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

#include "Dire/Basics.h"
#include "Dire/SplittingLibrary.h"
#include "Dire/WeightContainer.h"

namespace Pythia8 {

//==========================================================================

// Data on radiating dipole ends; only used inside DireTimes class.

class DireTimesEnd {

public:

  // Constructors.
  DireTimesEnd() : iRadiator(-1), iRecoiler(-1), pTmax(0.), colType(0),
    chgType(0), gamType(0), weakType(0), isrType(0), system(0), systemRec(0),
    MEtype(0), iMEpartner(-1), weakPol(0), isOctetOnium(false),
    isHiddenValley(false), colvType(0), MEmix(0.), MEorder(true),
    MEsplit(true), MEgluinoRec(false), isFlexible(false), idRadAft(0),
    idEmtAft(0), isSoftRad(false), isSoftRec(false) {
    mRad = m2Rad = mRec = m2Rec = mDip = m2Dip = m2DipCorr = pT2 = m2 = z
         = mFlavour = asymPol = flexFactor = phi = 0.;
    sa1  = xa = phia1 = pT2start = pT2stop = pT2Old = 0.;
  }

  DireTimesEnd(int iRadiatorIn, int iRecoilerIn, double pTmaxIn = 0.,
    int colIn = 0, int chgIn = 0, int gamIn = 0, int weakTypeIn = 0,
    int isrIn = 0, int systemIn = 0, int MEtypeIn = 0, int iMEpartnerIn = -1,
    int weakPolIn = 0, bool isOctetOniumIn = false, bool isSoftRadIn = false,
    bool isSoftRecIn = false, bool isHiddenValleyIn = false,
    int colvTypeIn = 0, double MEmixIn = 0.,
    bool MEorderIn = true, bool MEsplitIn = true, bool MEgluinoRecIn = false,
    bool isFlexibleIn = false, int idRadAftIn = 0, int idEmtAftIn = 0,
    vector<int> iSpectatorIn = vector<int>(),
    vector<double> massIn = vector<double>(),
    vector<int> allowedIn = vector<int>() ) :
    iRadiator(iRadiatorIn), iRecoiler(iRecoilerIn), pTmax(pTmaxIn),
    colType(colIn), chgType(chgIn), gamType(gamIn), weakType(weakTypeIn),
    isrType(isrIn), system(systemIn), systemRec(systemIn), MEtype(MEtypeIn),
    iMEpartner(iMEpartnerIn), weakPol(weakPolIn), isOctetOnium(isOctetOniumIn),
    isHiddenValley(isHiddenValleyIn), colvType(colvTypeIn), MEmix(MEmixIn),
    MEorder (MEorderIn), MEsplit(MEsplitIn), MEgluinoRec(MEgluinoRecIn),
    isFlexible(isFlexibleIn), mass(massIn), idRadAft(idRadAftIn),
    idEmtAft(idEmtAftIn), iSpectator(iSpectatorIn),
    allowedEmissions(allowedIn), isSoftRad(isSoftRadIn),
    isSoftRec(isSoftRecIn){ 
    mRad = m2Rad = mRec = m2Rec = mDip = m2Dip = m2DipCorr = pT2 = m2 = z
         = mFlavour = asymPol = flexFactor = phi = 0.;
    sa1  = xa = phia1 = 0.;
    pT2start = pT2stop = pT2Old = 0.;
  }

  DireTimesEnd( const DireTimesEnd& dip )
    : iRadiator(dip.iRadiator), iRecoiler(dip.iRecoiler), pTmax(dip.pTmax),
    colType(dip.colType), chgType(dip.chgType), gamType(dip.gamType),
    weakType(dip.weakType), isrType(dip.isrType), system(dip.system),
    systemRec(dip.systemRec), MEtype(dip.MEtype), iMEpartner(dip.iMEpartner),
    weakPol(dip.weakPol), isOctetOnium(dip.isOctetOnium),
    isHiddenValley(dip.isHiddenValley), colvType(dip.colvType),
    MEmix(dip.MEmix), MEorder(dip.MEorder), MEsplit(dip.MEsplit),
    MEgluinoRec(dip.MEgluinoRec), isFlexible(dip.isFlexible),
    flavour(dip.flavour), iAunt(dip.iAunt),
    mRad(dip.mRad), m2Rad(dip.m2Rad), mRec(dip.mRec), m2Rec(dip.m2Rec),
    mDip(dip.mDip), m2Dip(dip.m2Dip), m2DipCorr(dip.m2DipCorr), pT2(dip.pT2),
    m2(dip.m2), z(dip.z), mFlavour(dip.mFlavour), asymPol(dip.asymPol),
    flexFactor(dip.flexFactor), phi(dip.phi), pT2start(dip.pT2start),
    pT2stop(dip.pT2stop), pT2Old(dip.pT2Old), sa1(dip.sa1), xa(dip.xa),
    phia1(dip.phia1), mass(dip.mass), idRadAft(dip.idRadAft),
    idEmtAft(dip.idEmtAft), iSpectator(dip.iSpectator),
    allowedEmissions(dip.allowedEmissions), isSoftRad(dip.isSoftRad),
    isSoftRec(dip.isSoftRec) {}

  // Basic properties related to dipole and matrix element corrections.
  int    iRadiator, iRecoiler;
  double pTmax;
  int    colType, chgType, gamType, weakType, isrType, system, systemRec,
         MEtype, iMEpartner, weakPol;
  bool   isOctetOnium, isHiddenValley;
  int    colvType;
  double MEmix;
  bool   MEorder, MEsplit, MEgluinoRec, isFlexible;

  // Properties specific to current trial emission.
  int    flavour, iAunt;
  double mRad, m2Rad, mRec, m2Rec, mDip, m2Dip, m2DipCorr,
         pT2, m2, z, mFlavour, asymPol, flexFactor, phi, pT2start, pT2stop,
         pT2Old;

  // Properties of 1->3 splitting.
  double sa1, xa,  phia1;

  // Stored masses.
  vector<double> mass;

  int idRadAft, idEmtAft;

  // Extended list of recoilers.
  vector<int> iSpectator;
  // List of allowed emissions (to avoid double-counting, since one
  // particle can be part of many different dipoles.
  void appendAllowedEmt( int id) {
    if ( find(allowedEmissions.begin(), allowedEmissions.end(), id)
      == allowedEmissions.end() ) allowedEmissions.push_back(id);
  }
  void removeAllowedEmt( int id) {
    if ( find(allowedEmissions.begin(), allowedEmissions.end(), id)
      != allowedEmissions.end() ) allowedEmissions.erase ( 
      remove(allowedEmissions.begin(), allowedEmissions.end(), id),
      allowedEmissions.end());
  }
  void clearAllowedEmt() { allowedEmissions.resize(0); }
  vector<int> allowedEmissions;
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
    cout << "\n --------  Begin DireTimesEnd Listing  ----------------"
         << "------------------------------------------------------- \n \n  "
         << "  rad    rec       pTmax  col  chg  gam weak  isr"
         << "  sys sysR type  MErec    pol    m2      allowedIds\n"
         << fixed << setprecision(3);
    cout << scientific << setprecision(4) 
      << setw(7) << iRadiator
      << setw(7) << iRecoiler   << setw(12)<< pTmax
      << setw(5) << colType     << setw(5) << chgType
      << setw(5) << gamType     << setw(5) << weakType
      << setw(5) << isrType
      << setw(5) << system      << setw(5) << systemRec
      << setw(5) << MEtype      << setw(7) << iMEpartner
      << setw(5) << weakPol
      << setw(12) << m2Dip;
    for (int j = 0; j < int(allowedEmissions.size()); ++j)
      cout << setw(5) << allowedEmissions[j] << " ";
    cout << endl;
   // Done.
    cout << "\n --------  End DireTimesEnd Listing  ------------"
         << "-------------------------------------------------------" << endl;
  }

  bool isSoftRad, isSoftRec;
  void setSoftRad() { isSoftRad = true; }
  void setHardRad() { isSoftRad = false; }
  void setSoftRec() { isSoftRec = true; }
  void setHardRec() { isSoftRec = false; }

};

//==========================================================================

// The DireTimes class does timelike showers.

class DireTimes : public Pythia8::TimeShower {

public:

  // Constructor.
  DireTimes() {}

  DireTimes(Pythia8::Pythia* pythiaPtr) {
    beamOffset = 0;
    infoPtr           = &pythiaPtr->info;
    particleDataPtr   = &pythiaPtr->particleData;
    partonSystemsPtr  = &pythiaPtr->partonSystems;
    rndmPtr           = &pythiaPtr->rndm;
    settingsPtr       = &pythiaPtr->settings;
    userHooksPtr      = 0;
    mergingHooksPtr   = pythiaPtr->mergingHooksPtr;
    splittingsPtr     = 0;
    weights           = 0;
    debugPtr          = 0;
    printBanner       = true;
    isInitSave        = false;
    //dipEnd.reserve(1000000);
  }

  // Destructor.
  virtual ~DireTimes() {}

  // Initialize alphaStrong and related pTmin parameters.
  virtual void init( BeamParticle* beamAPtrIn = 0,
    BeamParticle* beamBPtrIn = 0);

  bool initSplits() {
    if (splittingsPtr) splits = splittingsPtr->getSplittings();
    return (splits.size() > 0);
  }

  // Initialize various pointers.
  // (Separated from rest of init since not virtual.)
  void reinitPtr(Info* infoPtrIn, Settings* settingsPtrIn,
       ParticleData* particleDataPtrIn, Rndm* rndmPtrIn,
       PartonSystems* partonSystemsPtrIn, UserHooks* userHooksPtrIn,
       MergingHooks* mergingHooksPtrIn, SplittingLibrary* splittingsPtrIn,
       DebugInfo* debugPtrIn) {
       infoPtr = infoPtrIn;
       settingsPtr = settingsPtrIn;
       particleDataPtr = particleDataPtrIn;
       rndmPtr = rndmPtrIn;
       partonSystemsPtr = partonSystemsPtrIn;
       userHooksPtr = userHooksPtrIn;
       mergingHooksPtr = mergingHooksPtrIn;
       splittingsPtr = splittingsPtrIn;
       debugPtr      = debugPtrIn;
  }

  void initVariations();

  void setWeightContainerPtr(WeightContainer* weightsIn){weights = weightsIn;}

  // Find whether to limit maximum scale of emissions, and whether to dampen.
  virtual bool limitPTmax( Event& event, double Q2Fac = 0.,
    double Q2Ren = 0.);

  // Potential enhancement factor of pTmax scale for hardest emission.
  virtual double enhancePTmax() { return pTmaxFudge;}

  // Top-level routine to do a full time-like shower in resonance decay.
  virtual int shower( int iBeg, int iEnd, Event& event, double pTmax,
    int nBranchMax = 0);

  // Global recoil: reset counters and store locations of outgoing partons.
  virtual void prepareGlobal( Event&);

  // Prepare system for evolution after each new interaction; identify ME.
  virtual void prepare( int iSys, Event& event, bool limitPTmaxIn = true);

  // Update dipole list after a multiparton interactions rescattering.
  virtual void rescatterUpdate( int iSys, Event& event);

  // Update dipole list after each ISR emission.
  virtual void update( int iSys, Event& event, bool = false);

  // Update dipole list after final-final splitting.
  void updateAfterFF( int iSysSelNow, int iSysSelRec,
    Event& event, int iRadBef, int iRecBef, int iRad, int iEmt, int iRec,
    int flavour, int colType, double pTsel);

  // Update dipole list after final-final splitting.
  void updateAfterFI( int iSysSelNow, int iSysSelRec,
    Event& event, int iRadBef, int iRecBef, int iRad, int iEmt, int iRec,
    int flavour, int colType, double pTsel, double xNew);

  // Select next pT in downwards evolution. Wrapper function inherited from
  // Pythia.
  virtual double pTnext( Event& event, double pTbegAll, double pTendAll,
    bool = false, bool = false);

  // Setup branching kinematics.
  virtual bool branch( Event& event, bool = false);
  bool branch_FF( Event& event, bool = false,
    SplitInfo* split = NULL);
  bool branch_FI( Event& event, bool = false,
    SplitInfo* split = NULL);

  pair < Vec4, Vec4 > decayWithOnshellRec( double zCS, double yCS, double phi,
    double m2Rec, double m2RadAft, double m2EmtAft,
    Vec4 pRadBef, Vec4 pRecBef);
  pair < Vec4, Vec4 > decayWithOffshellRec( double zCS, double yCS, double phi,
    double m2RadBef, double m2RadAft, double m2EmtAft,
    Vec4 pRadBef, Vec4 pRecBef);

  // Tell which system was the last processed one.
  virtual int system() const {return iSysSel;};

  // Setup clustering kinematics.
  virtual Event clustered( const Event& state, int iRad, int iEmt, int iRec,
    string name) {
    return  clustered_internal(state, iRad, iEmt, iRec, name).first; }
  pair <Event, pair<int,int> > clustered_internal( const Event& state,
    int iRad, int iEmt, int iRec, string name);
  bool cluster_FF( const Event& state, int iRad,
    int iEmt, int iRec, int idRadBef, Particle& radBef, Particle& recBef);
  bool cluster_FI( const Event& state, int iRad,
    int iEmt, int iRec, int idRadBef, Particle& radBef, Particle& recBef);

  // From Pythia version 8.215 onwards no longer virtual.
  double pT2Times ( const Particle& rad, const Particle& emt,
    const Particle& rec) {
    if (rec.isFinal()) return pT2_FF(rad,emt,rec);
    return pT2_FI(rad,emt,rec);
  }

  double pT2_FF ( const Particle& rad, const Particle& emt,
    const Particle& rec);
  double pT2_FI ( const Particle& rad, const Particle& emt,
    const Particle& rec);

  // From Pythia version 8.215 onwards no longer virtual.
  double zTimes ( const Particle& rad, const Particle& emt,
    const Particle& rec) {
    if (rec.isFinal()) return z_FF(rad,emt,rec);
    return z_FI(rad,emt,rec);
  }

  double z_FF ( const Particle& rad, const Particle& emt,
    const Particle& rec);
  double z_FI ( const Particle& rad, const Particle& emt,
    const Particle& rec);
  double z_FF_fromVec ( const Vec4& rad, const Vec4& emt, const Vec4& rec);

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
  // Usage: isTimelike( const Event& event,  int iRad, int iEmt, 
  //                   int iRec, string name)
  virtual bool isTimelike(const Event& state, int iRad, int, int, string)
    { return state[iRad].isFinal(); }

  // From Pythia version 8.215 onwards.
  // Return a string identifier of a splitting.
  // Usage: getSplittingName( const Event& event, int iRad, int iEmt, int iRec)
  //string getSplittingName( const Event& state, int iRad, int iEmt , int )
  //  { return splittingsPtr->getSplittingName(state,iRad,iEmt).front(); }
  virtual vector<string> getSplittingName( const Event& state, int iRad,
    int iEmt, int)
    { return splittingsPtr->getSplittingName(state,iRad,iEmt); }

  // From Pythia version 8.215 onwards.
  // Return the splitting probability.
  // Usage: getSplittingProb( const Event& event, int iRad, int iEmt, int iRec)
  virtual double getSplittingProb( const Event& state, int iRad,
    int iEmt, int iRec, string name);

  virtual bool allowedSplitting( const Event& state, int iRad, int iEmt);

  virtual vector<int> getRecoilers( const Event& state, int iRad, int iEmt,
    string name);

  //virtual double getCoupling( const Event& state, double mu2Ren, int iRad,
  //  int iEmt, int iRec, string name) {
  virtual double getCoupling( const Event&, double mu2Ren, int, int, int,
    string name) { 
    if (splits.find(name) != splits.end()) 
      return splits[name]->coupling(mu2Ren);
    return 1.;
  }

  /*// Return Jacobian for final-final phase space factorisation.
  double jacobian_FF(double z, double pT2, double m2dip, double q2,
    double m2RadBef = 0., double m2r = 0., double m2s = 0.,
    double m2e = 0.) {

    // Calculate CS variables.
    double kappa2 = pT2/m2dip;
    double yCS    = kappa2 / (1.-z);

    // Get scaled masses.
    double mu2RadBef = m2RadBef/q2;
    double mu2Rad = m2r/q2;
    double mu2Rec = m2s/q2;
    double mu2Emt = m2e/q2;

    // Get Jacobian.
    double jac1 = (1. - mu2Rad - mu2Rec - mu2Emt)
                / sqrt(lABC(1.,mu2RadBef,mu2Rec));
    double jac2 = 1. + ( mu2Rad + mu2Emt - mu2RadBef)
                     / ( yCS * ( 1. - mu2Rad - mu2Emt - mu2Rec ) );
    return jac1 / jac2 * (1-yCS);

  }

  // Return Jacobian for final-initial phase space factorisation.
  double jacobian_FI(double = 0., double = 0., double = 0., double = 0.,
    double = 0., double = 0., double = 0., double = 0.) { return 1.;}*/

  // Auxiliary function to return the position of a particle.
  // Should go int Event class eventually!
  int FindParticle( const Particle& particle, const Event& event,
    bool checkStatus = true );

  // Print dipole list; for debug mainly.
  virtual void list() const;

  Event makeHardEvent( int iSys, const Event& state, bool isProcess = false );
  //bool hasME(const Event& event);

  // Check that particle has sensible momentum.
  bool validMomentum( const Vec4& p, int id, int status);

  // Check colour/flavour correctness of state.
  bool validEvent( const Event& state, bool isProcess = false );

  // Check that mother-daughter-relations are correctly set.
  bool validMotherDaughter( const Event& state );

  // Find index of colour partner for input colour.
  int FindCol(int col, vector<int> iExclude, const Event& event, int type,
    int iSys = -1);

  // Pointers to the two incoming beams.
  BeamParticle*  getBeamA () { return beamAPtr; }
  BeamParticle*  getBeamB () { return beamBPtr; }

  // Function to calculate the correct alphaS/2*Pi value, including
  // renormalisation scale variations + threshold matching.
  double alphasNow( double pT2, double renormMultFacNow = 1., int iSys = 0 );

  // Function to calculate the correct alphaEM/2*Pi value.
  double alphaemNow( double pT2, double renormMultFacNow = 1., int iSys = 0 );

  bool isInit() { return isInitSave; }

private:

  friend class Splitting;

  // Number of times the same error message is repeated, unless overridden.
  static const int TIMESTOPRINT;

  // Allow conversion from mb to pb.
  static const double CONVERTMB2PB;

  // Colour factors.
  static const double CA, CF, TR, NC;

  // Store common beam quantities.
  int    idASave, idBSave;

protected:

  // Store properties to be returned by methods.
  int    iSysSel;
  double pTmaxFudge, pTLastBranch;

private:

  // Constants: could only be changed in the code itself.
  static const int MAXLOOPTINYPDF;
  static const double MCMIN, MBMIN, SIMPLIFYROOT, XMARGIN, XMARGINCOMB,
         TINYPDF, LARGEM2, THRESHM2, LAMBDA3MARGIN, TINYMASS, TINYOVERESTIMATE,
         PT2_INCREASE_OVERESTIMATE;

  // Initialization data, normally only set once.
  bool   isInitSave, doQCDshower, doQEDshowerByQ, doQEDshowerByL,
         doMEcorrections, doMEafterFirst, doPhiPolAsym,
         doInterleave, allowBeamRecoil, dampenBeamRecoil, recoilToColoured,
         useFixedFacScale, allowRescatter, canVetoEmission, hasUserHooks,
         doSecondHard, alphaSuseCMW, printBanner, doTrialNow;
  int    pTmaxMatch, pTdampMatch, alphaSorder, alphaSnfmax, alphaEMorder,
         nGluonToQuark, nGammaToQuark, nGammaToLepton, nFinalMax,
         nFinalMaxMECs,kernelOrder, kernelOrderMPI, nMPI;
  double pTdampFudge, mc, mb, m2c, m2b, renormMultFac, factorMultFac,
         fixedFacScale2, alphaSvalue, alphaS2pi, Lambda3flav, Lambda4flav,
         Lambda5flav, Lambda3flav2, Lambda4flav2, Lambda5flav2,
         pTcolCutMin, pTcolCut,
         pT2colCut, m2colCut, mTolErr, mZ, gammaZ, thetaW, mW, gammaW,
         pTmaxFudgeMPI, sumCharge2L, sumCharge2Q, sumCharge2Tot,
         pT2minVariations, pT2minMECs, pT2recombine;
  double alphaS2piOverestimate;
  bool usePDFalphas, usePDFmasses, useSummedPDF, usePDF, useSystems, useMassiveBeams;

  map<int,double> pT2cutSave;
  double pT2cut(int id) {
    if (pT2cutSave.find(id) != pT2cutSave.end()) return pT2cutSave[id];
    // Else return maximal value.
    double ret = 0.;
    for ( map<int,double>::iterator it = pT2cutSave.begin();
      it != pT2cutSave.end(); ++it ) ret = max(ret, it->second);
    return ret;
  }
  double pT2cutMax(DireTimesEnd* dip) {
    double ret = 0.;
    for (int i=0; i < int(dip->allowedEmissions.size()); ++i)
      ret = max( ret, pT2cut(dip->allowedEmissions[i]));
    return ret;
  }
  double pT2cutMin(DireTimesEnd* dip) {
    double ret = 1e15;
    for (int i=0; i < int(dip->allowedEmissions.size()); ++i)
      ret = min( ret, pT2cut(dip->allowedEmissions[i]));
    return ret;
  }

  bool doDecaysAsShower;

  // alphaStrong and alphaEM calculations.
  AlphaStrong alphaS;
  AlphaEM     alphaEM;

  // Some current values.
  bool   dopTlimit1, dopTlimit2, dopTdamp;
  double pT2damp, kRad, kEmt, pdfScale2;

  // All dipole ends and a pointer to the selected hardest dipole end.
  vector<DireTimesEnd> dipEnd;
  DireTimesEnd* dipSel;
  SplitInfo splitSel;
  int iDipSel;
  map<string,double> kernelSel, kernelNow;
  double auxSel, overSel, boostSel, auxNow, overNow, boostNow;


  double tinypdf( double x) {
    double xref = 0.01;
    return TINYPDF*log(1-x)/log(1-xref);
  }

  // Evolve a QCD dipole end near heavy quark threshold region.
  // Setup a dipole end, either QCD, QED/photon, weak or Hidden Valley one.
  void setupQCDdip( int iSys, int i, int colTag,  int colSign, Event& event,
    bool isOctetOnium = false, bool limitPTmaxIn = true);
  void getGenDip( int iSys, int i, int iRad, const Event& event,
    bool limitPTmaxIn, vector<DireTimesEnd>& dipEnds );
  void getQCDdip( int iRad, int colTag, int colSign, const Event& event,
    vector<DireTimesEnd>& dipEnds );
  void setupDecayDip( int iSys, int iRad, const Event& event,
    vector<DireTimesEnd>& dipEnds);

  // Function to set up and append a new dipole.
  bool appendDipole( const Event& state, int iRad, int iRec, double pTmax,
    int colType, int chgType, int gamType, int weakType, int isrType, int iSys,
    int MEtype, int iMEpartner, int weakPol, bool isOctetOnium, bool isSoftRad,
    bool isSoftRec, vector<DireTimesEnd>& dipEnds);

  vector<int> sharedColor(const Particle& rad, const Particle& rec);

  // Function to set up and append a new dipole.
  void updateDipoles(const Event& state);
  void checkDipoles(const Event& state);
  bool updateAllowedEmissions( const Event& state, DireTimesEnd* dip);
  bool appendAllowedEmissions( const Event& state, DireTimesEnd* dip);

  // Evolve a QCD dipole end.
  void pT2nextQCD( double pT2begDip, double pT2sel, DireTimesEnd& dip,
    Event& event);
  bool pT2nextQCD_FF( double pT2begDip, double pT2sel, DireTimesEnd& dip,
    const Event& event);
  bool pT2nextQCD_FI( double pT2begDip, double pT2sel, DireTimesEnd& dip,
    const Event& event);
  bool zCollNextQCD( DireTimesEnd* dip, double zMin, double zMax,
    double tMin = 0., double tMax = 0.);
  bool virtNextQCD( DireTimesEnd* dip, double tMin, double tMax,
    double zMin =-1., double zMax =-1.);

  SplittingLibrary* splittingsPtr;

  // Number of proposed splittings in hard scattering systems.
  map<int,int> nProposedPT;

  // Return headroom factors for integrated/differential overestimates.
  double overheadFactors(DireTimesEnd*, const Event&, string, double,
    double, double);
  double enhanceOverestimateFurther( string, int, double );

  // Function to fill map of integrated overestimates.
  void getNewOverestimates( DireTimesEnd*, const Event&, double, double,
    double, double, multimap<double,string>&);

  // Function to sum all integrated overestimates.
  void addNewOverestimates( multimap<double,string>, double&);

  // Function to attach the correct alphaS weights to the kernels.
  void alphasReweight(double t, double talpha, int iSys, double& weight,
    double& fullWeight, double& overWeight, double renormMultFacNow);

  // Function to evaluate the accept-probability, including picking of z.
  void getNewSplitting( const Event&, DireTimesEnd*, double, double, double,
    double, double, int, string, int&, int&, double&, double&, 
    map<string,double>&, double&);

  pair<bool, pair<double,double> > getMEC ( const Event& state, 
    SplitInfo* splitInfo);
  bool applyMEC ( const Event& state, SplitInfo* splitInfo);

  // Get particle masses.
  double getMass(int id, int strategy, double mass = 0.) {
    BeamParticle* beam = NULL;
    if (beamAPtr != NULL || beamBPtr != NULL) {
      beam = (beamAPtr != NULL && particleDataPtr->isHadron(beamAPtr->id()))
           ? beamAPtr
           : (beamBPtr != NULL && particleDataPtr->isHadron(beamBPtr->id()))
              ? beamBPtr : NULL;
    }
    bool usePDFmass = usePDFmasses
      && (toLower(settingsPtr->word("PDF:pSet")).find("lhapdf")
         != string::npos);
    double mRet = 0.;
    // Parton masses.
    if ( particleDataPtr->colType(id) != 0) {
      if (strategy == 1) mRet = particleDataPtr->m0(id);
      if (strategy == 2 &&  usePDFmass && beam != NULL)
        mRet = beam->mQuarkPDF(id);
      if (strategy == 2 && (!usePDFmass || beam == NULL))
        mRet = particleDataPtr->m0(id);
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
    double q2, double xOld, int splitType = 0, double m2RadBef = 0.,
    double m2r = 0., double m2s = 0., double m2e = 0.,
    vector<double> aux = vector<double>());

  /*// Kallen function and derived quantities. Helpers for massive phase
  // space mapping.
  double lABC(double a, double b, double c) { return pow2(a-b-c) - 4.*b*c;}
  double bABC(double a, double b, double c) { 
    double ret = 0.;
    if      ((a-b-c) > 0.) ret = sqrt(lABC(a,b,c));
    else if ((a-b-c) < 0.) ret =-sqrt(lABC(a,b,c));
    else                   ret = 0.;
    return ret; }
  double gABC(double a, double b, double c) { return 0.5*(a-b-c+bABC(a,b,c));}*/

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
  map<string, map<double,double> > acceptProbability;
  map<string, multimap<double,double> > rejectProbability;

  vector<int> softPosSave;

public:

  WeightContainer* weights;
  DebugInfo* debugPtr;

  void addSoftPos( int iPos ) { softPosSave.push_back(iPos); }
  void removeSoftPos( int iPos) { 
    vector<int>::iterator it = find( softPosSave.begin(),
      softPosSave.end(), iPos);
    if ( it != softPosSave.end() ) softPosSave.erase(it);
  }
  vector<int> softPos () { return softPosSave; }
  void clearSoftPos () { softPosSave.clear(); }

private:

  bool doVariations;

  // List of splitting kernels.
  map<string, Splitting* > splits;
  map<string, double > overhead;
  void scaleOverheadFactor(string name, double scale) {
    overhead[name] *= scale;
    return;
  }
  void resetOverheadFactors() {
    for ( map<string,double>::iterator it = overhead.begin();
      it != overhead.end(); ++it )
      it->second = 1.0;
    return;
  }

  double octetOniumColFac;
  bool useLocalRecoilNow;

  // Map to store some settings, to be passes to splitting kernels.
  map<string,bool> bool_settings;

};

//==========================================================================

} // end namespace Pythia8

#endif
