// DireSplittingsEW.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for the Dire electroweak splitings.

#ifndef Pythia8_DireSplittingsEW_H
#define Pythia8_DireSplittingsEW_H

#define DIRE_SPLITTINGSEW_VERSION "2.002"

#include "Pythia8/Basics.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"

#include "Pythia8/DireSplittingsQCD.h"

namespace Pythia8 {

//==========================================================================

class DireSplittingEW : public DireSplittingQCD {

public:

  // Constructor and destructor.
  DireSplittingEW(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingQCD(idIn,
      softRS,settings,particleData,rndm,beamA,beamB,coupSM,info, direInfo)
    { init(); }
  virtual ~DireSplittingEW() {}

  void init();

  // Z0 and W+- properties needed for gamma/Z0 mixing and weak showers.
  double mZ, gammaZ, thetaW, mW, gammaW, aem0, enhance;
  bool doQEDshowerByQ, doQEDshowerByL;

  AlphaEM     alphaEM;

  // Function to calculate the correct running coupling/2*Pi value, including
  // renormalisation scale variations + threshold matching.
  double aem2Pi ( double pT2);

  bool useFastFunctions() { return true; }

  virtual vector <int> radAndEmt(int idDaughter, int)
   { return createvector<int>(motherID(idDaughter))(sisterID(idDaughter)); }
  virtual int nEmissions()  { return 1; }
  virtual bool isPartial()  { return true; }

  virtual bool canUseForBranching() { return true; }

  virtual int couplingType (int, int) { return 2; }
  virtual double coupling (double = 0., double = 0., double = 0., double = -1.,
    pair<int,bool> = pair<int,bool>(), pair<int,bool> = pair<int,bool>()) {
    return (aem0 / (2.*M_PI));
  }
  virtual double couplingScale2 (double = 0., double = 0., double = 0.,
    pair<int,bool> = pair<int,bool>(), pair<int,bool> = pair<int,bool>()) {
    return -1.;
  }

};

//==========================================================================

class Dire_fsr_ew_Q2QZ : public DireSplittingEW {

public:

  Dire_fsr_ew_Q2QZ(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingEW(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int, int,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL)
    {return false;}

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

  double gaugeFactor ( int=0, int=0);
  double symmetryFactor ( int=0, int=0);

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

class Dire_fsr_ew_Q2ZQ : public DireSplittingEW {

public:

  Dire_fsr_ew_Q2ZQ(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingEW(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int, int,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL)
    {return false;}

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

  double gaugeFactor ( int=0, int=0);
  double symmetryFactor ( int=0, int=0);

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

class Dire_fsr_ew_Z2QQ1 : public DireSplittingEW {

public:

  Dire_fsr_ew_Z2QQ1(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingEW(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int, int,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL)
    {return false;}

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

  double gaugeFactor ( int=0, int=0);
  double symmetryFactor ( int=0, int=0);

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

class Dire_fsr_ew_Z2QQ2 : public DireSplittingEW {

public:

  Dire_fsr_ew_Z2QQ2(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingEW(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int, int,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL)
    {return false;}

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

  double gaugeFactor ( int=0, int=0);
  double symmetryFactor ( int=0, int=0);

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

class Dire_fsr_ew_W2QQ1 : public DireSplittingEW {

public:

  Dire_fsr_ew_W2QQ1(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingEW(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int, int,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL)
    {return false;}

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

  double gaugeFactor ( int=0, int=0);
  double symmetryFactor ( int=0, int=0);

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

class Dire_fsr_ew_W2QQ2 : public DireSplittingEW {

public:

  Dire_fsr_ew_W2QQ2(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingEW(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int, int,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL)
    {return false;}

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

  double gaugeFactor ( int=0, int=0);
  double symmetryFactor ( int=0, int=0);

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

class Dire_fsr_ew_H2WW : public DireSplittingEW {

public:

  Dire_fsr_ew_H2WW(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingEW(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int, int,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL)
    {return false;}

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

  double gaugeFactor ( int=0, int=0);
  double symmetryFactor ( int=0, int=0);

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

class Dire_fsr_ew_H2AA : public DireSplittingEW {

public:

  Dire_fsr_ew_H2AA(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingEW(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo) {
    widthToAA = particleDataPtr->particleDataEntryPtr(25)->resWidthChan(
      particleDataPtr->m0(25), 22, 22);
    widthTot = settings->parm("MEM:WidthH");
  }

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool isPartial()  { return false; }

  bool isSymmetric( const Particle* rad, const Particle* emt) {
   if (rad->id() == 22 && emt->id() == 22) return true;
   return false;
  }

  int couplingType (int, int);
  double coupling (double = 0., double = 0., double = 0., double = -1.,
  pair<int,bool> = pair<int,bool>(), pair<int,bool> = pair<int,bool>());

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event& state, int iRad, int iEmt);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  double gaugeFactor ( int=0, int=0);
  double symmetryFactor ( int=0, int=0);

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

  double widthToAA, widthTot;

};

//==========================================================================

class Dire_fsr_ew_H2GG : public DireSplittingEW {

public:

  Dire_fsr_ew_H2GG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingEW(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo) {
    widthToGG = particleDataPtr->particleDataEntryPtr(25)->resWidthChan(
      particleDataPtr->m0(25), 21, 21);
    widthTot = settings->parm("MEM:WidthH");
  }

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int iRadBef, int iRecBef,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool isPartial()  { return false; }

  bool isSymmetric( const Particle* rad, const Particle* emt) {
   if (rad->id() == 21 && emt->id() == 21) return true;
   return false;
  }

  int couplingType (int, int);
  double coupling (double = 0., double = 0., double = 0., double = -1.,
  pair<int,bool> = pair<int,bool>(), pair<int,bool> = pair<int,bool>());

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event& state, int iRad, int iEmt);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  double gaugeFactor ( int=0, int=0);
  double symmetryFactor ( int=0, int=0);

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

  double widthToGG, widthTot;

};

//==========================================================================

class Dire_fsr_ew_W2WA : public DireSplittingEW {

public:

  Dire_fsr_ew_W2WA(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingEW(idIn, softRS, settings, particleData, rndm, beamA, beamB,
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

  vector <int> recPositions( const Event& state, int iRad, int iEmt);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter,
    int colEmtAfter, int acolEmtAfter);

  vector<pair<int,int> > radAndEmtCols(int iRad, int, Event state) {
    vector< pair<int,int> > ret;
    if (state[iRad].idAbs() != 24) return ret;
    ret = createvector<pair<int,int> >(make_pair(0, 0))(make_pair(0, 0));
    return ret;
  }

  double gaugeFactor ( int=0, int=0);
  double symmetryFactor ( int=0, int=0);

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

class Dire_isr_ew_Q2QZ : public DireSplittingEW {

public:

  Dire_isr_ew_Q2QZ(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info, DireInfo* direInfo) :
    DireSplittingEW(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info, direInfo){}

  bool canRadiate ( const Event&, pair<int,int>,
    unordered_map<string,bool> = unordered_map<string,bool>(),
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL);
  bool canRadiate ( const Event&, int, int,
    Settings* = NULL, PartonSystems* = NULL, BeamParticle* = NULL)
    {return false;}

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

  double gaugeFactor ( int=0, int=0);
  double symmetryFactor ( int=0, int=0);

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

} // end namespace Pythia8

#endif // Pythia8_DireSplittingLibrary_H
