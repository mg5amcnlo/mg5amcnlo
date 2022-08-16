// Info.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains classes that keep track of generic event info.
// Info: contains information on the generation process and errors.
// Info: user interface to make parts of Info publicly available.

#ifndef Pythia8_Info_H
#define Pythia8_Info_H

#include "Pythia8/Basics.h"
#include "Pythia8/LHEF3.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/SharedPointers.h"
#include "Pythia8/Weights.h"

namespace Pythia8 {

// Forward declaration of various classes.
class Settings;
class ParticleData;
class Rndm;
class CoupSM;
class CoupSUSY;
class BeamParticle;
class PartonSystems;
class SigmaTotal;
class HadronWidths;

// Forward declaration of HIInfo class.
class HIInfo;

//==========================================================================

// The Info class contains a mixed bag of information on the event
// generation activity, especially on the current subprocess properties,
// and on the number of errors encountered. This is used by the
// generation machinery.

class Info {

public:

  // Constructors.
  Info() = default;
  Info(bool) : Info() {}

  // Destructor for clean-up.
  ~Info() {
    if (hasOwnEventAttributes && eventAttributes != nullptr)
      delete eventAttributes;
  }

  // Assignment operator gives a shallow copy; no objects pointed to
  // are copied.
  Info & operator=(const Info &) = default;

  // Pointers to other class objects, carried piggyback by Info.

  // Set pointers to other class objects.
  void setPtrs(Settings* settingsPtrIn, ParticleData* particleDataPtrIn,
    Rndm* rndmPtrIn, CoupSM* coupSMPtrIn, CoupSUSY* coupSUSYPtrIn,
    BeamParticle* beamAPtrIn,    BeamParticle* beamBPtrIn,
    BeamParticle* beamPomAPtrIn, BeamParticle*  beamPomBPtrIn,
    BeamParticle* beamGamAPtrIn, BeamParticle*  beamGamBPtrIn,
    BeamParticle* beamVMDAPtrIn, BeamParticle*  beamVMDBPtrIn,
    PartonSystems* partonSystemsPtrIn, SigmaTotal* sigmaTotPtrIn,
    HadronWidths* hadronWidthsPtrIn,
    WeightContainer* weightContainerPtrIn) {
    settingsPtr = settingsPtrIn; particleDataPtr = particleDataPtrIn;
    rndmPtr = rndmPtrIn; coupSMPtr = coupSMPtrIn; coupSUSYPtr = coupSUSYPtrIn;
    beamAPtr    = beamAPtrIn;    beamBPtr    = beamBPtrIn;
    beamPomAPtr = beamPomAPtrIn; beamPomBPtr = beamPomBPtrIn;
    beamGamAPtr = beamGamAPtrIn; beamGamBPtr = beamGamBPtrIn;
    beamVMDAPtr = beamVMDAPtrIn; beamVMDBPtr = beamVMDBPtrIn;
    partonSystemsPtr = partonSystemsPtrIn; sigmaTotPtr = sigmaTotPtrIn;
    hadronWidthsPtr = hadronWidthsPtrIn;
    weightContainerPtr = weightContainerPtrIn; }

  // Pointer to the settings database.
  Settings*      settingsPtr{};

  // Pointer to the particle data table.
  ParticleData*  particleDataPtr{};

  // Pointer to the random number generator.
  Rndm*          rndmPtr{};

  // Pointers to Standard Model and Beyond SM couplings.
  CoupSM*        coupSMPtr{};
  CoupSUSY*      coupSUSYPtr{};

  // Pointers to the two incoming beams and to Pomeron, photon or VMD
  // beam-inside-beam cases.
  BeamParticle*  beamAPtr{};
  BeamParticle*  beamBPtr{};
  BeamParticle*  beamPomAPtr{};
  BeamParticle*  beamPomBPtr{};
  BeamParticle*  beamGamAPtr{};
  BeamParticle*  beamGamBPtr{};
  BeamParticle*  beamVMDAPtr{};
  BeamParticle*  beamVMDBPtr{};

  // Pointer to information on subcollision parton locations.
  PartonSystems* partonSystemsPtr{};

  // Pointer to the total/elastic/diffractive cross sections.
  SigmaTotal*    sigmaTotPtr{};

  // Pointer to the hadron widths data table.
  HadronWidths*  hadronWidthsPtr;

  // Pointer to the UserHooks object set for the run.
  UserHooksPtr   userHooksPtr{};

  // Pointer to information about a HeavyIons run and the current event.
  // (Is NULL if HeavyIons object is inactive.)
  HIInfo*        hiInfo{};

  WeightContainer* weightContainerPtr{};

  // Listing of most available information on current event.
  void   list() const;

  // Beam particles (in rest frame). CM energy of event.
  int    idA()                const {return idASave;}
  int    idB()                const {return idBSave;}
  double pzA()                const {return pzASave;}
  double pzB()                const {return pzBSave;}
  double eA()                 const {return eASave;}
  double eB()                 const {return eBSave;}
  double mA()                 const {return mASave;}
  double mB()                 const {return mBSave;}
  double eCM()                const {return eCMSave;}
  double s()                  const {return sSave;}

  // Warnings from initialization.
  bool   tooLowPTmin()        const {return lowPTmin;}

  // Process name and code, and the number of final-state particles.
  string name()               const {return nameSave;}
  int    code()               const {return codeSave;}
  int    nFinal()             const {return nFinalSave;}

  // Are beam particles resolved, with pdf's? Are they diffractive?
  bool   isResolved()         const {return isRes;}
  bool   isDiffractiveA()     const {return isDiffA;}
  bool   isDiffractiveB()     const {return isDiffB;}
  bool   isDiffractiveC()     const {return isDiffC;}
  bool   isNonDiffractive()   const {return isND;}
  bool   isElastic()          const {return (codeSave == 102);}
  // Retained for backwards compatibility.
  bool   isMinBias()          const {return isND;}

  // Information for Les Houches Accord and reading files.
  bool   isLHA()              const {return isLH;}
  bool   atEndOfFile()        const {return atEOF;}

  // For nondiffractive and Les Houches Accord identify hardest subprocess.
  bool   hasSub(int i = 0)    const {return hasSubSave[i];}
  string nameSub(int i = 0)   const {return nameSubSave[i];}
  int    codeSub(int i = 0)   const {return codeSubSave[i];}
  int    nFinalSub(int i = 0) const {return nFinalSubSave[i];}

  // Incoming parton flavours and x values.
  int    id1(int i = 0)       const {return id1Save[i];}
  int    id2(int i = 0)       const {return id2Save[i];}
  double x1(int i = 0)        const {return x1Save[i];}
  double x2(int i = 0)        const {return x2Save[i];}
  double y(int i = 0)         const {return 0.5 * log( x1Save[i] / x2Save[i]);}
  double tau(int i = 0)       const {return x1Save[i] * x2Save[i];}

  // Hard process flavours, x values, parton densities, couplings, Q2 scales.
  int    id1pdf(int i = 0)    const {return id1pdfSave[i];}
  int    id2pdf(int i = 0)    const {return id2pdfSave[i];}
  double x1pdf(int i = 0)     const {return x1pdfSave[i];}
  double x2pdf(int i = 0)     const {return x2pdfSave[i];}
  double pdf1(int i = 0)      const {return pdf1Save[i];}
  double pdf2(int i = 0)      const {return pdf2Save[i];}
  double QFac(int i = 0)      const {return sqrtpos(Q2FacSave[i]);}
  double Q2Fac(int i = 0)     const {return Q2FacSave[i];}
  bool   isValence1()         const {return isVal1;}
  bool   isValence2()         const {return isVal2;}
  double alphaS(int i = 0)    const {return alphaSSave[i];}
  double alphaEM(int i = 0)   const {return alphaEMSave[i];}
  double QRen(int i = 0)      const {return sqrtpos(Q2RenSave[i]);}
  double Q2Ren(int i = 0)     const {return Q2RenSave[i];}
  double scalup(int i = 0)    const {return scalupSave[i];}

  // Kinematics of photons from lepton beams.
  double xGammaA()            const {return x1GammaSave;}
  double xGammaB()            const {return x2GammaSave;}
  double Q2GammaA()           const {return Q2Gamma1Save;}
  double Q2GammaB()           const {return Q2Gamma2Save;}
  double eCMsub()             const {return eCMsubSave;}
  double thetaScatLepA()      const {return thetaLepton1;}
  double thetaScatLepB()      const {return thetaLepton2;}
  double sHatNew()            const {return sHatNewSave;}
  int    photonMode()         const {return gammaModeEvent;}

  // Information on VMD state inside a photon.
  bool   isVMDstateA()        const {return isVMDstateAEvent;}
  bool   isVMDstateB()        const {return isVMDstateBEvent;}
  int    idVMDA()             const {return idVMDASave;}
  int    idVMDB()             const {return idVMDBSave;}
  double mVMDA()              const {return mVMDASave;}
  double mVMDB()              const {return mVMDBSave;}
  double scaleVMDA()          const {return scaleVMDASave;}
  double scaleVMDB()          const {return scaleVMDBSave;}

  // Mandelstam variables (notation as if subcollision).
  double mHat(int i = 0)      const {return sqrt(sH[i]);}
  double sHat(int i = 0)      const {return sH[i];}
  double tHat(int i = 0)      const {return tH[i];}
  double uHat(int i = 0)      const {return uH[i];}
  double pTHat(int i = 0)     const {return pTH[i];}
  double pT2Hat(int i = 0)    const {return pTH[i] * pTH[i];}
  double m3Hat(int i = 0)     const {return m3H[i];}
  double m4Hat(int i = 0)     const {return m4H[i];}
  double thetaHat(int i = 0)  const {return thetaH[i];}
  double phiHat(int i = 0)    const {return phiH[i];}

  // Weight of current event; normally 1, but used for Les Houches events
  // or when reweighting phase space selection. Conversion from mb to pb
  // for LHA strategy +-4. Uncertainty variations can be accessed by
  // providing an index >= 1 (0 = no variations). Also cumulative sum.
  double weight(int i = 0)    const;
  double weightSum()          const;
  double lhaStrategy()        const {return lhaStrategySave;}

  // Further access to uncertainty weights: number and labels
  int nWeights() const
    { return weightContainerPtr->weightsShowerPtr->getWeightsSize(); }
  string weightLabel(int iWeight) const {
    return weightContainerPtr->weightsShowerPtr->getWeightsName(iWeight);
    }

  int    nWeightGroups() const { return weightContainerPtr->
      weightsShowerPtr->nWeightGroups(); }
  string getGroupName(int iGN) const {
    return weightContainerPtr->weightsShowerPtr->getGroupName(iGN);
  }
  double getGroupWeight(int iGW) const {
    return weightContainerPtr->weightsShowerPtr->getGroupWeight(iGW)
      *weightContainerPtr->weightNominal;
  }

  // Number of times other steps have been carried out.
  int    nISR()               const {return nISRSave;}
  int    nFSRinProc()         const {return nFSRinProcSave;}
  int    nFSRinRes()          const {return nFSRinResSave;}

  // Maximum pT scales for MPI, ISR and FSR (in hard process).
  double pTmaxMPI()           const {return pTmaxMPISave;}
  double pTmaxISR()           const {return pTmaxISRSave;}
  double pTmaxFSR()           const {return pTmaxFSRSave;}

  // Current evolution scale (for UserHooks).
  double pTnow()              const {return pTnowSave;}

  // Impact parameter picture, global information
  double a0MPI()              const {return a0MPISave;}

  // Impact parameter picture, as set by hardest interaction.
  double bMPI()               const {return (bIsSet) ? bMPISave : 1.;}
  double enhanceMPI()         const {return (bIsSet) ? enhanceMPISave : 1.;}
  double enhanceMPIavg()      const {return (bIsSet) ? enhanceMPIavgSave : 1.;}
  double eMPI(int i)          const {return (bIsSet) ? eMPISave[i] : 1.;}
  double bMPIold()            const {return (bIsSet) ? bMPIoldSave : 1.;}
  double enhanceMPIold()      const {return (bIsSet) ? enhanceMPIoldSave : 1.;}
  double enhanceMPIoldavg()   const {return (bIsSet)
                                     ? enhanceMPIoldavgSave : 1.;}

  // Number of multiparton interactions, with code and pT for them.
  int    nMPI()               const {return nMPISave;}
  int    codeMPI(int i)       const {return codeMPISave[i];}
  double pTMPI(int i)         const {return pTMPISave[i];}
  int    iAMPI(int i)         const {return iAMPISave[i];}
  int    iBMPI(int i)         const {return iBMPISave[i];}

  // Cross section estimate, optionally process by process.
  vector<int> codesHard();
  string nameProc(int i = 0)  const {return (i == 0) ? "sum"
    : ( (procNameM.at(i) == "") ? "unknown process" : procNameM.at(i) );}
  long   nTried(int i = 0)    const {return (i == 0) ? nTry : nTryM.at(i);}
  long   nSelected(int i = 0) const {return (i == 0) ? nSel : nSelM.at(i);}
  long   nAccepted(int i = 0) const {return (i == 0) ? nAcc : nAccM.at(i);}
  double sigmaGen(int i = 0)  const {return (i == 0) ? sigGen : sigGenM.at(i);}
  double sigmaErr(int i = 0)  const {return (i == 0) ? sigErr : sigErrM.at(i);}

  // Counters for number of loops in various places.
  int    getCounter( int i)   const {return counters[i];}

  // Set or increase the value stored in a counter.
  void   setCounter( int i, int value = 0) {counters[i]  = value;}
  void   addCounter( int i, int value = 1) {counters[i] += value;}

  // Reset to empty map of error messages.
  void   errorReset() {messages.clear();}

  // Print a message the first few times. Insert in database.
  void   errorMsg(string messageIn, string extraIn = " ",
    bool showAlways = false);

  // Provide total number of errors/aborts/warnings experienced to date.
  int    errorTotalNumber() const;

  // Print statistics on errors/aborts/warnings.
  void   errorStatistics() const;

  // Set initialization warning flag when too low pTmin in ISR/FSR/MPI.
  void   setTooLowPTmin(bool lowPTminIn) {lowPTmin = lowPTminIn;}

  // Set info on valence character of hard collision partons.
  void setValence( bool isVal1In, bool isVal2In) {isVal1 = isVal1In;
    isVal2 = isVal2In;}

  // Set and get some MPI/ISR/FSR properties needed for matching,
  // i.e. mainly of internal relevance.
  void   hasHistory( bool hasHistoryIn) {hasHistorySave = hasHistoryIn;}
  bool   hasHistory() {return hasHistorySave;}
  void   zNowISR( double zNowIn) {zNowISRSave = zNowIn;}
  double zNowISR() {return zNowISRSave;}
  void   pT2NowISR( double pT2NowIn) {pT2NowISRSave = pT2NowIn;}
  double pT2NowISR() {return pT2NowISRSave;}

  // Return merging weight.
  double mergingWeight(int i=0) const {
    return weightContainerPtr->weightsMerging.getWeightsValue(i);}

  // Return the complete NLO weight.
  double mergingWeightNLO(int i=0) const {
    return weightContainerPtr->weightsMerging.getWeightsValue(i); }

  // Return an LHEF header
  string header(const string &key) const {
    if (headers.find(key) == headers.end()) return "";
    else return headers.at(key);
  }

  // Return a list of all header key names
  vector<string> headerKeys() const;

  // Return the number of processes in the LHEF.
  int nProcessesLHEF() const { return int(sigmaLHEFSave.size());}
  // Return the cross section information read from LHEF.
  double sigmaLHEF(int iProcess) const { return sigmaLHEFSave[iProcess];}

  // LHEF3 information: Public for easy access.
  int LHEFversionSave;

  // Save process information as read from init block of LHEF.
  vector<double> sigmaLHEFSave;

  // Contents of the LHAinitrwgt tag
  LHAinitrwgt *initrwgt{};

  // Contents of the LHAgenerator tags.
  vector<LHAgenerator > *generators{};

  // A map of the LHAweightgroup tags, indexed by name.
  map<string,LHAweightgroup > *weightgroups{};

  // A map of the LHAweight tags, indexed by name.
  map<string,LHAweight > *init_weights{};

  // Store current-event Les Houches event tags.
  bool hasOwnEventAttributes{};
  map<string, string > *eventAttributes{};

  // The weights associated with this event, as given by the LHAwgt tags
  map<string,double > *weights_detailed{};

  // The weights associated with this event, as given by the LHAweights tags
  vector<double > *weights_compressed{};

  // Contents of the LHAscales tag.
  LHAscales *scales{};

  // Contents of the LHAweights tag (compressed format)
  LHAweights *weights{};

  // Contents of the LHArwgt tag (detailed format)
  LHArwgt *rwgt{};

  // Vectorized version of LHArwgt tag, for easy and ordered access.
  vector<double> weights_detailed_vector;

  // Value of the unit event weight read from a Les Houches event, necessary
  // if additional weights in an otherwise unweighted input file are in
  // relation to this number.
  double eventWeightLHEF = {};

  // Set the LHEF3 objects read from the init and header blocks.
  void setLHEF3InitInfo();
  void setLHEF3InitInfo( int LHEFversionIn, LHAinitrwgt *initrwgtIn,
    vector<LHAgenerator> *generatorsIn,
    map<string,LHAweightgroup> *weightgroupsIn,
    map<string,LHAweight> *init_weightsIn, string headerBlockIn );
  // Set the LHEF3 objects read from the event block.
  void setLHEF3EventInfo();
  void setLHEF3EventInfo( map<string, string> *eventAttributesIn,
    map<string, double > *weights_detailedIn,
    vector<double > *weights_compressedIn,
    LHAscales *scalesIn, LHAweights *weightsIn,
    LHArwgt *rwgtIn, vector<double> weights_detailed_vecIn,
    vector<string> weights_detailed_name_vecIn,
    string eventCommentsIn, double eventWeightLHEFIn );

  // Retrieve events tag information.
  string getEventAttribute(string key, bool doRemoveWhitespace = false) const;

  // Externally set event tag auxiliary information.
  void setEventAttribute(string key, string value, bool doOverwrite = true) {
    if (eventAttributes == NULL) {
      eventAttributes = new map<string,string>();
      hasOwnEventAttributes = true;
    }
    map<string, string>::iterator it = eventAttributes->find(key);
    if ( it != eventAttributes->end() && !doOverwrite ) return;
    if ( it != eventAttributes->end() ) eventAttributes->erase(it);
    eventAttributes->insert(make_pair(key,value));
  }

  // Retrieve LHEF version
  int LHEFversion() const { return LHEFversionSave; }

  // Retrieve initrwgt tag information.
  unsigned int getInitrwgtSize() const;

  // Retrieve generator tag information.
  unsigned int getGeneratorSize() const;
  string getGeneratorValue(unsigned int n = 0) const;
  string getGeneratorAttribute( unsigned int n, string key,
    bool doRemoveWhitespace = false) const;

  // Retrieve rwgt tag information.
  unsigned int getWeightsDetailedSize() const;
  double getWeightsDetailedValue(string n) const;
  string getWeightsDetailedAttribute(string n, string key,
    bool doRemoveWhitespace = false) const;

  // Retrieve weights tag information.
  unsigned int getWeightsCompressedSize() const;
  double getWeightsCompressedValue(unsigned int n) const;
  string getWeightsCompressedAttribute(string key,
    bool doRemoveWhitespace = false) const;

  // Retrieve scales tag information.
  string getScalesValue(bool doRemoveWhitespace = false) const;
  double getScalesAttribute(string key) const;

  // Retrieve complete header block and event comments
  // Retrieve scales tag information.
  string getHeaderBlock() const { return headerBlock;}
  string getEventComments() const { return eventComments;}

  // Set LHEF headers
  void setHeader(const string &key, const string &val)
    { headers[key] = val; }

  // Set abort in parton level.
  void setAbortPartonLevel(bool abortIn) {abortPartonLevel = abortIn;}
  bool getAbortPartonLevel() const {return abortPartonLevel;}

  // Get information on hard diffractive events.
  bool   hasUnresolvedBeams() const {return hasUnresBeams;}
  bool   hasPomPsystem()      const {return hasPomPsys;}
  bool   isHardDiffractive()  const {return isHardDiffA || isHardDiffB;}
  bool   isHardDiffractiveA() const {return isHardDiffA;}
  bool   isHardDiffractiveB() const {return isHardDiffB;}
  double xPomeronA()          const {return xPomA;}
  double xPomeronB()          const {return xPomB;}
  double tPomeronA()          const {return tPomA;}
  double tPomeronB()          const {return tPomB;}

  // History information needed to setup the weak shower for 2 -> n.
  vector<int> getWeakModes() const {return weakModes;}
  vector<pair<int,int> > getWeakDipoles() const {return weakDipoles;}
  vector<Vec4> getWeakMomenta() const {return weakMomenta;}
  vector<int> getWeak2to2lines() const {return weak2to2lines;}
  void setWeakModes(vector<int> weakModesIn) {weakModes = weakModesIn;}
  void setWeakDipoles(vector<pair<int,int> > weakDipolesIn)
    {weakDipoles = weakDipolesIn;}
  void setWeakMomenta(vector<Vec4> weakMomentaIn)
    {weakMomenta = weakMomentaIn;}
  void setWeak2to2lines(vector<int> weak2to2linesIn)
    {weak2to2lines = weak2to2linesIn;}

  // From here on what used to be the private part of the class.

  // Number of times the same error message is repeated, unless overridden.
  static const int TIMESTOPRINT;

  // Allow conversion from mb to pb.
  static const double CONVERTMB2PB;

  // Store common beam quantities.
  int    idASave{}, idBSave{};
  double pzASave{}, eASave{}, mASave{}, pzBSave{},  eBSave{}, mBSave{},
         eCMSave{}, sSave{};

  // Store initialization information.
  bool   lowPTmin;

  // Store common integrated cross section quantities.
  long   nTry{}, nSel{}, nAcc{};
  double sigGen{}, sigErr{}, wtAccSum{};
  map<int, string> procNameM;
  map<int, long> nTryM, nSelM, nAccM;
  map<int, double> sigGenM, sigErrM;
  int    lhaStrategySave{};

  // Store common MPI information.
  double a0MPISave;

  // Store current-event quantities.
  bool   isRes{}, isDiffA{}, isDiffB{}, isDiffC{}, isND{}, isLH{},
         hasSubSave[4], bIsSet{}, evolIsSet{}, atEOF{}, isVal1{}, isVal2{},
         hasHistorySave{}, abortPartonLevel{}, isHardDiffA{}, isHardDiffB{},
         hasUnresBeams{}, hasPomPsys{};
  int    codeSave{}, codeSubSave[4], nFinalSave{}, nFinalSubSave[4], nTotal{},
         id1Save[4], id2Save[4], id1pdfSave[4], id2pdfSave[4], nMPISave{},
         nISRSave{}, nFSRinProcSave{}, nFSRinResSave{};
  double x1Save[4], x2Save[4], x1pdfSave[4], x2pdfSave[4], pdf1Save[4],
         pdf2Save[4], Q2FacSave[4], alphaEMSave[4], alphaSSave[4],
         Q2RenSave[4], scalupSave[4], sH[4], tH[4], uH[4], pTH[4], m3H[4],
         m4H[4], thetaH[4], phiH[4], bMPISave{}, enhanceMPISave{},
         enhanceMPIavgSave{}, bMPIoldSave{}, enhanceMPIoldSave{},
         enhanceMPIoldavgSave{}, pTmaxMPISave{}, pTmaxISRSave{},
         pTmaxFSRSave{}, pTnowSave{}, zNowISRSave{}, pT2NowISRSave{}, xPomA{},
         xPomB{}, tPomA{}, tPomB{};
  string nameSave{}, nameSubSave[4];
  vector<int>    codeMPISave, iAMPISave, iBMPISave;
  vector<double> pTMPISave, eMPISave;

  // Variables related to photon kinematics.
  bool   isVMDstateAEvent{}, isVMDstateBEvent{};
  int    gammaModeEvent{}, idVMDASave{}, idVMDBSave{};
  double x1GammaSave{}, x2GammaSave{}, Q2Gamma1Save{}, Q2Gamma2Save{},
         eCMsubSave{}, thetaLepton1{}, thetaLepton2{}, sHatNewSave{},
         mVMDASave{}, mVMDBSave{}, scaleVMDASave{}, scaleVMDBSave{};

  // Vector of various loop counters.
  int    counters[50];

  // Map for all error messages.
  map<string, int> messages;

  // Map for LHEF headers.
  map<string, string> headers;

  // Strings for complete header block and event comments.
  string headerBlock{}, eventComments{};

  // Map for plugin libraries.
  map<string, PluginPtr> plugins;

  // Load a plugin library.
  PluginPtr plugin(string nameIn) {
    map<string, PluginPtr>::iterator pluginItr = plugins.find(nameIn);
    if (pluginItr == plugins.end()) {
      PluginPtr pluginPtr = make_shared<Plugin>(nameIn, this);
      plugins[nameIn] = pluginPtr;
      return pluginPtr;
    } else return pluginItr->second;
  };

  // Set info on the two incoming beams: only from Pythia class.
  void setBeamA( int idAin, double pzAin, double eAin, double mAin) {
    idASave = idAin; pzASave = pzAin; eASave = eAin; mASave = mAin;}
  void setBeamB( int idBin, double pzBin, double eBin, double mBin) {
    idBSave = idBin; pzBSave = pzBin; eBSave = eBin; mBSave = mBin;}
  void setECM( double eCMin) {eCMSave = eCMin; sSave = eCMSave * eCMSave;}

  // Set info related to gamma+gamma subcollision.
  void setX1Gamma( double x1GammaIn)     { x1GammaSave    = x1GammaIn;   }
  void setX2Gamma( double x2GammaIn)     { x2GammaSave    = x2GammaIn;   }
  void setQ2Gamma1( double Q2gammaIn)    { Q2Gamma1Save   = Q2gammaIn;   }
  void setQ2Gamma2( double Q2gammaIn)    { Q2Gamma2Save   = Q2gammaIn;   }
  void setTheta1( double theta1In)       { thetaLepton1   = theta1In;    }
  void setTheta2( double theta2In)       { thetaLepton2   = theta2In;    }
  void setECMsub( double eCMsubIn)       { eCMsubSave     = eCMsubIn;    }
  void setsHatNew( double sHatNewIn)     { sHatNewSave    = sHatNewIn;   }
  void setGammaMode( double gammaModeIn) { gammaModeEvent = gammaModeIn; }
  // Set info on VMD state.
  void setVMDstateA(bool isVMDAIn, int idAIn, double mAIn, double scaleAIn)
    {isVMDstateAEvent = isVMDAIn; idVMDASave = idAIn; mVMDASave = mAIn;
    scaleVMDASave = scaleAIn;}
  void setVMDstateB(bool isVMDBIn, int idBIn, double mBIn, double scaleBIn)
    {isVMDstateBEvent = isVMDBIn; idVMDBSave = idBIn; mVMDBSave = mBIn;
    scaleVMDBSave = scaleBIn;}

  // Reset info for current event: only from Pythia class.
  void clear() {
    isRes = isDiffA = isDiffB = isDiffC = isND = isLH = bIsSet
      = evolIsSet = atEOF = isVal1 = isVal2 = hasHistorySave
      = isHardDiffA = isHardDiffB = hasUnresBeams = hasPomPsys = false;
    codeSave = nFinalSave = nTotal = nMPISave = nISRSave = nFSRinProcSave
      = nFSRinResSave = 0;
    bMPISave = enhanceMPISave = enhanceMPIavgSave = bMPIoldSave
      = enhanceMPIoldSave = enhanceMPIoldavgSave = 1.;
    pTmaxMPISave = pTmaxISRSave = pTmaxFSRSave = pTnowSave = zNowISRSave
      = pT2NowISRSave = 0.;
    nameSave = " ";
    for (int i = 0; i < 4; ++i) {
      hasSubSave[i] = false;
      codeSubSave[i] = nFinalSubSave[i] = id1pdfSave[i] = id2pdfSave[i]
        = id1Save[i] = id2Save[i] = 0;
      x1pdfSave[i] = x2pdfSave[i] = pdf1Save[i] = pdf2Save[i]
        = Q2FacSave[i] = alphaEMSave[i] = alphaSSave[i] = Q2RenSave[i]
        = scalupSave[i] = x1Save[i] = x2Save[i] = sH[i] = tH[i] = uH[i]
        = pTH[i] = m3H[i] = m4H[i] = thetaH[i] = phiH[i] = 0.;
      nameSubSave[i] = " ";
    }
    codeMPISave.resize(0); iAMPISave.resize(0); iBMPISave.resize(0);
    pTMPISave.resize(0); eMPISave.resize(0); setHardDiff();
    weightContainerPtr->clear();
  }

  // Reset info arrays only for parton and hadron level.
  int sizeMPIarrays() const { return codeMPISave.size(); }
  void resizeMPIarrays(int newSize) { codeMPISave.resize(newSize);
    iAMPISave.resize(newSize); iBMPISave.resize(newSize);
    pTMPISave.resize(newSize); eMPISave.resize(newSize); }

  // Set info on the (sub)process: from ProcessLevel, ProcessContainer or
  // MultipartonInteractions classes.
  void setType( string nameIn, int codeIn, int nFinalIn,
    bool isNonDiffIn = false, bool isResolvedIn = true,
    bool isDiffractiveAin = false, bool isDiffractiveBin = false,
    bool isDiffractiveCin = false, bool isLHAin = false) {
    nameSave = nameIn; codeSave = codeIn; nFinalSave = nFinalIn;
    isND = isNonDiffIn; isRes = isResolvedIn; isDiffA = isDiffractiveAin;
    isDiffB = isDiffractiveBin; isDiffC = isDiffractiveCin; isLH = isLHAin;
    nTotal = 2 + nFinalSave; bIsSet = false; hasSubSave[0] = false;
    nameSubSave[0] = " "; codeSubSave[0] = 0; nFinalSubSave[0] = 0;
    evolIsSet = false;}
  void setSubType( int iDS, string nameSubIn, int codeSubIn,
    int nFinalSubIn) { hasSubSave[iDS] = true; nameSubSave[iDS] = nameSubIn;
    codeSubSave[iDS] = codeSubIn; nFinalSubSave[iDS] = nFinalSubIn;}
  void setPDFalpha( int iDS, int id1pdfIn, int id2pdfIn, double x1pdfIn,
    double x2pdfIn, double pdf1In, double pdf2In, double Q2FacIn,
    double alphaEMIn, double alphaSIn, double Q2RenIn, double scalupIn) {
    id1pdfSave[iDS] = id1pdfIn; id2pdfSave[iDS] = id2pdfIn;
    x1pdfSave[iDS] = x1pdfIn; x2pdfSave[iDS] = x2pdfIn;
    pdf1Save[iDS] = pdf1In; pdf2Save[iDS] = pdf2In; Q2FacSave[iDS] = Q2FacIn;
    alphaEMSave[iDS] = alphaEMIn; alphaSSave[iDS] = alphaSIn;
    Q2RenSave[iDS] = Q2RenIn; scalupSave[iDS] = scalupIn;}
  void setScalup( int iDS, double scalupIn) {scalupSave[iDS] = scalupIn;}
  void setKin( int iDS, int id1In, int id2In, double x1In, double x2In,
    double sHatIn, double tHatIn, double uHatIn, double pTHatIn,
    double m3HatIn, double m4HatIn, double thetaHatIn, double phiHatIn) {
    id1Save[iDS] = id1In; id2Save[iDS] = id2In; x1Save[iDS] = x1In;
    x2Save[iDS] = x2In; sH[iDS] = sHatIn; tH[iDS] = tHatIn;
    uH[iDS] = uHatIn; pTH[iDS] = pTHatIn; m3H[iDS] = m3HatIn;
    m4H[iDS] = m4HatIn; thetaH[iDS] = thetaHatIn; phiH[iDS] = phiHatIn;}
  void setTypeMPI( int codeMPIIn, double pTMPIIn, int iAMPIIn = 0,
    int iBMPIIn = 0, double eMPIIn = 1.) {codeMPISave.push_back(codeMPIIn);
    pTMPISave.push_back(pTMPIIn); iAMPISave.push_back(iAMPIIn);
    iBMPISave.push_back(iBMPIIn); eMPISave.push_back(eMPIIn); }

  // Set info on cross section: from ProcessLevel (reset in Pythia).
  void sigmaReset() { nTry = nSel = nAcc = 0; sigGen = sigErr = wtAccSum = 0.;
    procNameM.clear(); nTryM.clear(); nSelM.clear(); nAccM.clear();
    sigGenM.clear(); sigErrM.clear();}
  void setSigma( int i, string procNameIn, long nTryIn, long nSelIn,
    long nAccIn, double sigGenIn, double sigErrIn, double wtAccSumIn) {
    if (i == 0) {nTry = nTryIn; nSel = nSelIn; nAcc = nAccIn;
      sigGen = sigGenIn; sigErr = sigErrIn; wtAccSum = wtAccSumIn;}
    else {procNameM[i] = procNameIn; nTryM[i] = nTryIn; nSelM[i] = nSelIn;
      nAccM[i] = nAccIn; sigGenM[i] = sigGenIn; sigErrM[i] = sigErrIn;} }
  void addSigma( int i, long nTryIn, long nSelIn, long nAccIn, double sigGenIn,
    double sigErrIn) { nTryM[i] += nTryIn; nSelM[i] += nSelIn;
    nAccM[i] += nAccIn; sigGenM[i] += sigGenIn;
    sigErrM[i] = sqrtpos(sigErrM[i]*sigErrM[i] + sigErrIn*sigErrIn); }

  // Set info on impact parameter: from PartonLevel.
  void setImpact( double bMPIIn, double enhanceMPIIn, double enhanceMPIavgIn,
    bool bIsSetIn = true, bool pushBack = false) {
    if (pushBack) {bMPIoldSave = bMPISave; enhanceMPIoldSave = enhanceMPISave;
      enhanceMPIoldavgSave = enhanceMPIavgSave;}
    bMPISave = bMPIIn; enhanceMPISave = eMPISave[0] = enhanceMPIIn,
    enhanceMPIavgSave = enhanceMPIavgIn, bIsSet = bIsSetIn;}

  // Set info on pTmax scales and number of evolution steps: from PartonLevel.
  void setPartEvolved( int nMPIIn, int nISRIn) {
    nMPISave = nMPIIn; nISRSave = nISRIn;}
  void setEvolution( double pTmaxMPIIn, double pTmaxISRIn, double pTmaxFSRIn,
    int nMPIIn, int nISRIn, int nFSRinProcIn, int nFSRinResIn) {
    pTmaxMPISave = pTmaxMPIIn; pTmaxISRSave = pTmaxISRIn;
    pTmaxFSRSave = pTmaxFSRIn; nMPISave = nMPIIn; nISRSave = nISRIn;
    nFSRinProcSave = nFSRinProcIn; nFSRinResSave = nFSRinResIn;
    evolIsSet = true;}

  // Set current pT evolution scale for MPI/ISR/FSR; from PartonLevel.
  void setPTnow( double pTnowIn) {pTnowSave = pTnowIn;}

  // Set a0 from MultipartonInteractions.
  void seta0MPI(double a0MPIin) {a0MPISave = a0MPIin;}

  // Set info whether reading of Les Houches Accord file at end.
  void setEndOfFile( bool atEOFin) {atEOF = atEOFin;}

  // Set event weight, either for LHEF3 or for uncertainty bands. If weight
  // has units (lhaStrategy 4 or -4), input in mb
  void setWeight( double weightIn, int lhaStrategyIn) {
    for (int i = 0; i < nWeights(); ++i)
      weightContainerPtr->weightsShowerPtr->setValueByIndex(i,1.);
    // Nominal weight in weightContainer saved in pb for lhaStrategy +-4
    weightContainerPtr->setWeightNominal(
        abs(lhaStrategyIn) == 4 ? CONVERTMB2PB*weightIn : weightIn);
    lhaStrategySave = lhaStrategyIn;}
  //
  // Set info on resolved processes.
  void setIsResolved(bool isResIn) {isRes = isResIn;}

  // Set info on hard diffraction.
  void setHardDiff(bool hasUnresBeamsIn = false, bool hasPomPsysIn = false,
    bool isHardDiffAIn = false, bool isHardDiffBIn = false,
    double xPomAIn = 0., double xPomBIn = 0., double tPomAIn = 0.,
    double tPomBIn = 0.) { hasUnresBeams = hasUnresBeamsIn;
    hasPomPsys = hasPomPsysIn; isHardDiffA = isHardDiffAIn;
    isHardDiffB = isHardDiffBIn;
    xPomA = xPomAIn; xPomB = xPomBIn;
    tPomA = tPomAIn; tPomB = tPomBIn;}

  // Move process information to an another diffractive system.
  void reassignDiffSystem(int iDSold, int iDSnew);

  // Set information in hard diffractive events.
  void setHasUnresolvedBeams(bool hasUnresBeamsIn)
    {hasUnresBeams = hasUnresBeamsIn;}
  void setHasPomPsystem(bool hasPomPsysIn) {hasPomPsys = hasPomPsysIn;}

  // Variables for weak shower setup.
  vector<int> weakModes, weak2to2lines;
  vector<Vec4> weakMomenta;
  vector<pair<int, int> > weakDipoles;

  int numberOfWeights() const {
    return weightContainerPtr->numberOfWeights(); }
  double weightValueByIndex(int key=0) const {
    return weightContainerPtr->weightValueByIndex(key); }
  string weightNameByIndex(int key=0) const {
    return weightContainerPtr->weightNameByIndex(key); }
  vector<double> weightValueVector() const {
    return weightContainerPtr->weightValueVector(); }
  vector<string> weightNameVector() const {
    return weightContainerPtr->weightNameVector(); }

};

//==========================================================================

// Class for loading plugin libraries at run time.

class Plugin {

public:

  // Constructor.
  Plugin(string nameIn = "", Info *infoPtrIn = nullptr);

  // Destructor.
  ~Plugin();

  // Return if the plugin is loaded.
  bool isLoaded() {return libPtr != nullptr;}

  // Symbol from the plugin library.
  typedef void (*Symbol)();

  // Access plugin library symbols.
  Symbol symbol(string symName);

protected:

  // Small routine for error printout, depending on infoPtr existing or not.
  void errorMsg(string errMsg) {
    if (infoPtr != nullptr) infoPtr->errorMsg(errMsg);
    else cout << errMsg << endl;
  }

 private:

  // Pointer to info.
  Info  *infoPtr;
  // The loaded plugin library.
  void  *libPtr;
  // The plugin library name.
  string name;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_Info_H
