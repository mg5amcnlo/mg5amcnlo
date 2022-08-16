// DireWeightContainer.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for Dire QED splittings.

#ifndef Pythia8_DireSplittingsQED_H
#define Pythia8_DireSplittingsQED_H

#define DIRE_SPLITTINGSQED_VERSION "2.002"

#include "Pythia8/Basics.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StandardModel.h"

#include "Pythia8/DireSplittingsQCD.h"

namespace Pythia8 {

//==========================================================================

class DireSplittingQED : public DireSplittingQCD {

public:

  // Constructor and destructor.
  DireSplittingQED(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQCD(idIn,
      softRS,settings,particleData,rndm,beamA,beamB,coupSM,info, direInfo)
    { init(); }
  virtual ~DireSplittingQED() {}

  void init();

  // VARIABLES
  double sumCharge2Tot, sumCharge2L, sumCharge2Q, aem0, enhance, pT2min,
         pT2minL, pT2minQ, pT2minForcePos;
  bool doQEDshowerByQ, doQEDshowerByL, doForcePos;

  AlphaEM     alphaEM;

  // Function to calculate the correct running coupling/2*Pi value, including
  // renormalisation scale variations + threshold matching.
  double aem2Pi ( double pT2, int = 0);

  bool useFastFunctions() { return true; }

  virtual vector <int> radAndEmt(int idDaughter, int)
   { return createvector<int>(motherID(idDaughter))(sisterID(idDaughter)); }
  virtual int nEmissions()  { return 1; }
  virtual bool isPartial()  { return true; }

  virtual int couplingType (int, int) { return 2; }
  virtual double coupling (double = 0., double = 0., double = 0., double = -1,
    pair<int,bool> = pair<int,bool>(), pair<int,bool> = pair<int,bool>()) {
    return (aem0 / (2.*M_PI));
  }
  virtual double couplingScale2 (double = 0., double = 0., double = 0.,
    pair<int,bool> = pair<int,bool>(), pair<int,bool> = pair<int,bool>()) {
    return -1.;
  }

  virtual bool aboveCutoff( double t, const Particle& radBef,
    const Particle& recBef, int iSys, PartonSystems* partonSystemsPtr);

};

//==========================================================================

class Dire_fsr_qed_Q2QA : public DireSplittingQED {

public:

  Dire_fsr_qed_Q2QA(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  vector<pair<int,int> > radAndEmtCols(int iRad, int, Event state) {
    return createvector<pair<int,int> >
      (make_pair(state[iRad].col(),state[iRad].acol()))(make_pair(0, 0));
  }

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  vector <int> recPositions( const Event& state, int iRad, int iEmt);

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

//==========================================================================

class Dire_fsr_qed_Q2AQ : public DireSplittingQED {

public:

  Dire_fsr_qed_Q2AQ(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  vector<pair<int,int> > radAndEmtCols(int iRad, int, Event state) {
    return createvector<pair<int,int> >
      (make_pair(state[iRad].col(),state[iRad].acol()))(make_pair(0, 0));
  }

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  vector <int> recPositions( const Event& state, int iRad, int iEmt);

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

//==========================================================================

class Dire_fsr_qed_L2LA : public DireSplittingQED {

public:

  Dire_fsr_qed_L2LA(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  vector<pair<int,int> > radAndEmtCols(int, int, Event) {
    return createvector<pair<int,int> > (make_pair(0,0))(make_pair(0, 0));
  }

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  vector <int> recPositions( const Event& state, int iRad, int iEmt);

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

//==========================================================================

class Dire_fsr_qed_L2AL : public DireSplittingQED {

public:

  Dire_fsr_qed_L2AL(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  vector<pair<int,int> > radAndEmtCols(int, int, Event) {
    return createvector<pair<int,int> > (make_pair(0,0))(make_pair(0, 0));
  }

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  vector <int> recPositions( const Event& state, int iRad, int iEmt);

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

//==========================================================================

class Dire_fsr_qed_A2FF : public DireSplittingQED {

public:

  int idRadAfterSave;
  double nchSaved;

  Dire_fsr_qed_A2FF(int idRadAfterIn, string idIn, int softRS,
    Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn,
    softRS, settings, particleData, rndm, beamA, beamB, coupSM, info,
    direInfo),
    idRadAfterSave(idRadAfterIn), nchSaved(1) {}
  bool canRadiate ( const Event& state, pair<int,int> ints,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL) {
    return ( state[ints.first].isFinal()
          && state[ints.first].id() == 22
          && state[ints.second].isCharged());
  }
  bool canRadiate ( const Event& state, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL) {
    return ( state[iRadBef].isFinal()
          && state[iRadBef].id() == 22
          && state[iRecBef].isCharged());
  }

  int kinMap () { return 1;};
  bool canUseForBranching() { return true; }
  bool isPartial()  { return false; }

  vector<pair<int,int> > radAndEmtCols(int iRad, int, Event state) {
    vector< pair<int,int> > ret;
    if (state[iRad].id() != 22) return ret;
    ret = createvector<pair<int,int> >(make_pair(0, 0))(make_pair(0, 0));
    if (particleDataPtr->colType(idRadAfterSave) != 0) {
      int sign      = (idRadAfterSave > 0) ? 1 : -1;
      int newCol    = state.nextColTag();
      if (sign> 0) {
        ret[0].first  = newCol;
        ret[0].second = 0;
        ret[1].first  = 0;
        ret[1].second = newCol;
      } else {
        ret[0].first  = 0;
        ret[0].second = newCol;
        ret[1].first  = newCol;
        ret[1].second = 0;
      }
    }
    return ret;
  }

  // Return id of mother after splitting.
  int motherID(int)
    { return idRadAfterSave; }
  int sisterID(int)
    { return -idRadAfterSave; }
  vector <int> radAndEmt(int, int)
    { return createvector<int>(idRadAfterSave)(-idRadAfterSave); }
    //{ return createvector<int>(1)(-1); }
  double gaugeFactor    ( int=0, int=0 )
    { return pow2(particleDataPtr->charge(idRadAfterSave)); }
  double symmetryFactor ( int=0, int=0 )
    { return 1./double(nchSaved); }
  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter) {
    if ( idRadAfter == idRadAfterSave
      && particleDataPtr->isQuark(idRadAfter)
      && particleDataPtr->isQuark(idEmtAfter)) return 22;
    return 0;
  }
  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int, int, int, int) { return make_pair(0,0); }

  // All charged particles are potential recoilers.
  vector <int> recPositions( const Event& state, int iRad, int iEmt) {
    if ( state[iRad].isFinal() || state[iRad].id() != idRadAfterSave
      || state[iEmt].id() != -idRadAfterSave) return vector<int>();
    // Particles to exclude as recoilers.
    vector<int> iExc(createvector<int>(iRad)(iEmt));
    // Find charged particles.
    vector<int> recs;
    for (int i=0; i < state.size(); ++i) {
      if ( find(iExc.begin(), iExc.end(), i) != iExc.end() ) continue;
      if ( state[i].isCharged() ) {
        if (state[i].isFinal())
          recs.push_back(i);
        if (state[i].mother1() == 1 && state[i].mother2() == 0)
          recs.push_back(i);
        if (state[i].mother1() == 2 && state[i].mother2() == 0)
          recs.push_back(i);
      }
    }
    // Done.
    return recs;
  }

  // All charged particles are potential recoilers.
  int set_nCharged( const Event& state) {
    // Find charged particles.
    int nch=0;
    for (int i=0; i < state.size(); ++i) {
      if ( state[i].isCharged() ) {
        if (state[i].isFinal()) nch++;
        if (state[i].mother1() == 1 && state[i].mother2() == 0) nch++;
        if (state[i].mother1() == 2 && state[i].mother2() == 0) nch++;
      }
    }
    // Done.
    nchSaved = nch;
    return nch;
  }

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double /*m2dip*/) {
      return (zMinAbs + rndmPtr->flat() * (zMaxAbs - zMinAbs));
  }

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double /*pT2Old*/, double /*m2dip*/, int /*order*/ = -1) {
    double preFac = symmetryFactor() * gaugeFactor();
    double wt     = 2. *enhance * preFac * 0.5 * ( zMaxAbs - zMinAbs);
    return wt;
  }

  // Return kernel for new splitting.
  double overestimateDiff(double /*z*/, double /*m2dip*/, int /*order*/ = -1) {
    double preFac = symmetryFactor() * gaugeFactor();
    double wt     = 2. *enhance * preFac * 0.5;
    return wt;
  }

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state, int orderNow) {

    // Dummy statement to avoid compiler warnings.
    if (false) cout << state[0].e() << orderNow << endl;

    // Read all splitting variables.
    double z(splitInfo.kinematics()->z), pT2(splitInfo.kinematics()->pT2),
      m2dip(splitInfo.kinematics()->m2Dip),
      //m2RadBef(splitInfo.kinematics()->m2RadBef),
      m2Rad(splitInfo.kinematics()->m2RadAft),
      m2Rec(splitInfo.kinematics()->m2Rec),
      m2Emt(splitInfo.kinematics()->m2EmtAft);
    int splitType(splitInfo.type);

    // Set number of recoilers.
    set_nCharged(state);

    double wt = 0.;
    double preFac = symmetryFactor() * gaugeFactor();
    double kappa2 = pT2/m2dip;
    wt  = preFac
        * (pow(1.-z,2.) + pow(z,2.));

    // Correction for massive splittings.
    bool doMassive = (abs(splitType) == 2);

    if (doMassive) {

      double vijk = 1., pipj = 0.;

      // splitType == 2 -> Massive FF
      if (splitType == 2) {
        // Calculate CS variables.
        double yCS = kappa2 / (1.-z);
        double nu2Rad = m2Rad/m2dip;
        double nu2Emt = m2Emt/m2dip;
        double nu2Rec = m2Rec/m2dip;
        vijk          = pow2(1.-yCS) - 4.*(yCS+nu2Rad+nu2Emt)*nu2Rec;
        vijk          = sqrt(vijk) / (1-yCS);
        pipj          = m2dip * yCS /2.;

      // splitType ==-2 -> Massive FI
      } else if (splitType ==-2) {
        // Calculate CS variables.
        double xCS = 1 - kappa2/(1.-z);
        vijk   = 1.;
        pipj   = m2dip/2. * (1-xCS)/xCS;
      }

      // Reset kernel for massive splittings.
      wt = preFac * 1. / vijk * ( pow2(1.-z) + pow2(z)
                                      + m2Emt / ( pipj + m2Emt) );
    }

    // Multiply with z factor
    if (idRadAfterSave > 0) wt *= z;
    else                    wt *= 1.-z;

    // Trivial map of values, since kernel does not depend on coupling.
    unordered_map<string,double> wts;
    wts.insert( make_pair("base", wt ));
    if (doVariations) {
      // Create muR-variations.
      if (settingsPtr->parm("Variations:muRfsrDown") != 1.)
        wts.insert( make_pair("Variations:muRfsrDown", wt ));
      if (settingsPtr->parm("Variations:muRfsrUp")   != 1.)
        wts.insert( make_pair("Variations:muRfsrUp", wt ));
    }

    // Store kernel values.
    clearKernels();
    for ( unordered_map<string,double>::iterator it = wts.begin();
      it != wts.end(); ++it )
      kernelVals.insert(make_pair( it->first, it->second ));

    return true;
  }

};


//==========================================================================

class Dire_isr_qed_Q2QA : public DireSplittingQED {

public:

  Dire_isr_qed_Q2QA(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  vector<pair<int,int> > radAndEmtCols(int iRad, int, Event state) {
    return createvector<pair<int,int> >
      (make_pair(state[iRad].col(),state[iRad].acol()))(make_pair(0, 0));
  }

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  vector <int> recPositions( const Event& state, int iRad, int iEmt);

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class Dire_isr_qed_A2QQ : public DireSplittingQED {

public:

  Dire_isr_qed_A2QQ(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  vector<pair<int,int> > radAndEmtCols(int iRad, int, Event state) {
    return createvector<pair<int,int> >
      (make_pair(0, 0))(make_pair(state[iRad].acol(),state[iRad].col()));
  }

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class Dire_isr_qed_Q2AQ : public DireSplittingQED {

public:

  Dire_isr_qed_Q2AQ(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  vector<pair<int,int> > radAndEmtCols(int, int colType, Event state) {
    int newCol    = state.nextColTag();
    if (colType > 0) return createvector<pair<int,int> >
      (make_pair(newCol,0))(make_pair(newCol,0));
    return createvector<pair<int,int> >
      (make_pair(0,newCol))(make_pair(0,newCol));
  }

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class Dire_isr_qed_L2LA : public DireSplittingQED {

public:

  Dire_isr_qed_L2LA(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  vector<pair<int,int> > radAndEmtCols(int, int, Event) {
    return createvector<pair<int,int> > (make_pair(0,0))(make_pair(0,0));
  }

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  vector <int> recPositions( const Event& state, int iRad, int iEmt);

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class Dire_isr_qed_A2LL : public DireSplittingQED {

public:

  Dire_isr_qed_A2LL(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  vector<pair<int,int> > radAndEmtCols(int, int, Event) {
    return createvector<pair<int,int> > (make_pair(0,0))(make_pair(0,0));
  }

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class Dire_isr_qed_L2AL : public DireSplittingQED {

public:

  Dire_isr_qed_L2AL(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  vector<pair<int,int> > radAndEmtCols(int, int, Event) {
    return createvector<pair<int,int> > (make_pair(0,0))(make_pair(0,0));
  }

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

//==========================================================================

class Dire_fsr_qed_Q2QA_notPartial : public DireSplittingQED {

public:

  Dire_fsr_qed_Q2QA_notPartial(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);

  int kinMap ();
  bool canUseForBranching() { return true; }
  bool isPartial()  { return false; }

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  vector<pair<int,int> > radAndEmtCols(int iRad, int, Event state) {
    return createvector<pair<int,int> >
      (make_pair(state[iRad].col(),state[iRad].acol()))(make_pair(0, 0));
  }

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  vector <int> recPositions( const Event& state, int iRad, int iEmt);

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

//==========================================================================

class Dire_fsr_qed_L2LA_notPartial : public DireSplittingQED {

public:

  Dire_fsr_qed_L2LA_notPartial(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQED(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);

  int kinMap ();
  bool canUseForBranching() { return true; }
  bool isPartial()  { return false; }

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  vector<pair<int,int> > radAndEmtCols(int, int, Event) {
    return createvector<pair<int,int> > (make_pair(0,0))(make_pair(0,0));
  }

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  vector <int> recPositions( const Event& state, int iRad, int iEmt);

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_DireSplittingsQED_H
