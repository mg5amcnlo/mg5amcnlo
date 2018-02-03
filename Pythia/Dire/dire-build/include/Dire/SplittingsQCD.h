
#ifndef Pythia8_SplittingsQCD_H
#define Pythia8_SplittingsQCD_H

#define ZETA3 1.202056903159594
#define DIRE_SPLITTINGSQCD_VERSION "2.002"

#include "Pythia8/Basics.h"
#include "Pythia8/Pythia.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StandardModel.h"

#include "Dire/Splittings.h"

namespace Pythia8 {

//==========================================================================

class SplittingQCD : public Splitting {

public:  

  // Constructor and destructor.
  SplittingQCD(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) : Splitting(idIn,
      softRS, settings, particleData, rndm, beamA, beamB, coupSM, info)
    { init(); }
  virtual ~SplittingQCD() {}

  void init();

  // VARIABLES
  double CA, TR, CF, pTmin, pT2minVariations;
  int NF_qcd_fsr;
  bool usePDFalphas, doVariations;
  double alphaS2pi;

  AlphaStrong alphaS;

  static const double SMALL_TEVOL;

  // AUXILIARY FUNCTIONS
  double getNF(double pT2);
  double GammaQCD2(double NF=5.);
  double GammaQCD3(double NF=5.);
  double betaQCD0(double NF=5.);
  double betaQCD1(double NF=5.);
  double betaQCD2(double NF=5.);

  // Function to calculate the correct running coupling/2*Pi value, including
  // renormalisation scale variations + threshold matching.
  //double as2Pi  ( double pT2, int orderNow = 0, double renormMultFacNow = -1.);
  double as2Pi  ( double pT2, int orderNow = -1, double renormMultFacNow = -1.);

  double softRescaleInt(int order);
  double softRescaleDiff(int order, double pT2, double renormMultFacNow = -1.);

  double polevl(double x,double* coef,int N );
  double  DiLog(double x);

  vector<int> sharedColor(const Event& event, int iRad, int iRec);
  int findCol(int col, vector<int> iExc, const Event&, int type);

  virtual vector <int> radAndEmt(int idDaughter, int)
   { return createvector<int>(motherID(idDaughter))(sisterID(idDaughter)); } 
  virtual bool isPartial()  { return true; }

  virtual int couplingType (int, int) { return 1; }
  virtual double coupling (double z, double pT2, double m2dip,
    pair<int,bool> radBef, pair<int,bool> recBef) {
    double scale2 = couplingScale2 ( z, pT2, m2dip, radBef, recBef);
    if (scale2 < 0.) scale2 = pT2;
    return as2Pi(scale2);
  }
  virtual double couplingScale2 (double z, double pT2, double m2dip,
    pair<int,bool> radBef, pair<int,bool> recBef) {
    if        ( radBef.second &&  recBef.second) {
      if (settingsPtr->mode("DireTimes:alphasScheme") == 0) return pT2;
      return pT2;
    } else if ( radBef.second && !recBef.second) {
      if (settingsPtr->mode("DireTimes:alphasScheme") == 0) return pT2;
      double zcs = z;
      double xcs = m2dip * zcs * (1.-zcs) / (pT2 + m2dip * zcs * (1.-zcs));
      double kt2 = m2dip * (1.-xcs) / xcs * zcs * (1.-zcs);
      return kt2;
    } else if (!radBef.second &&  recBef.second) {
      if (settingsPtr->mode("DireTimes:alphasScheme") == 0) return pT2;
      double xcs = z;
      double ucs = pT2/m2dip / (1.-z);
      double kt2 = m2dip * (1-xcs) / xcs * ucs * (1.-ucs);
      return kt2;
    } else if (!radBef.second && !recBef.second) {
      if (settingsPtr->mode("DireTimes:alphasScheme") == 0) return pT2;
      double xcs = ( z * (1.-z) - pT2/m2dip) / (1.-z);
      double vcs = pT2/m2dip / (1.-z);
      double kt2 = m2dip * vcs * (1.-xcs-vcs) / xcs;
      return kt2;
    }
    return -1.;
  }

  // Functions that allow different ordering variables for emissions.
  // Note: Only works after splitInfo has been properly filled.
  virtual double getJacobian( const Event& = Event(),
    PartonSystems* partonSystems = 0);
  virtual map<string, double> getPhasespaceVars(const Event& = Event(),
    PartonSystems* = 0);

};

//==========================================================================

class fsr_qcd_Q2QGG : public SplittingQCD {

public:  

  fsr_qcd_Q2QGG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

  vector <int> radAndEmt(int idDaughter, int)
    { return createvector<int>(idDaughter)(21)(21);}
  int nEmissions()          { return 2; }
  int kinMap()              { return 2;}
  bool canUseForBranching() { return true; }

  vector<pair<int,int> > radAndEmtCols(int iRad, int colType, Event state); 

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  //vector <int> recPositions( const Event&, int, int);

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

class fsr_qcd_G2GGG : public SplittingQCD {

public:  

  fsr_qcd_G2GGG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

  vector <int> radAndEmt(int,int) { return createvector<int>(21)(21)(21);}
  int nEmissions()                { return 2; }
  int kinMap()                    { return 2;}
  bool canUseForBranching()       { return true; }

  vector<pair<int,int> > radAndEmtCols(int iRad, int colType, Event state);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  //vector <int> recPositions( const Event&, int, int);

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

class fsr_qcd_G2QQG : public SplittingQCD {

public:  

  fsr_qcd_G2QQG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info)
      { nGluonToQuark = settingsPtr->mode("TimeShower:nGluonToQuark"); }

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

  virtual vector <int> radAndEmt(int, int) { return vector<int>(); } 

  int nEmissions()            { return 2; }
  int kinMap()                { return 2;}
  bool canUseForBranching()   { return true; }

  vector<pair<int,int> > radAndEmtCols(int iRad, int colType, Event state);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  //vector <int> recPositions( const Event&, int, int);

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

  int nGluonToQuark;

};

//==========================================================================

class fsr_qcd_G2DDG : public fsr_qcd_G2QQG {

public:  

  fsr_qcd_G2DDG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    fsr_qcd_G2QQG(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info) {}

  vector <int> radAndEmt(int, int colType) { 
    int sign = (colType > 0) ? 1 : -1; 
    int idRadAft = sign * 1;
    return createvector<int>(idRadAft)(-idRadAft)(21);
  }

};

//==========================================================================

class fsr_qcd_G2UUG : public fsr_qcd_G2QQG {

public:  

  fsr_qcd_G2UUG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    fsr_qcd_G2QQG(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info) {}

  vector <int> radAndEmt(int, int colType) { 
    int sign = (colType > 0) ? 1 : -1; 
    int idRadAft = sign * 2;
    return createvector<int>(idRadAft)(-idRadAft)(21);
  }

};

//==========================================================================

class fsr_qcd_G2SSG : public fsr_qcd_G2QQG {

public:  

  fsr_qcd_G2SSG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    fsr_qcd_G2QQG(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info) {}

  vector <int> radAndEmt(int, int colType) { 
    int sign = (colType > 0) ? 1 : -1; 
    int idRadAft = sign * 3;
    return createvector<int>(idRadAft)(-idRadAft)(21);
  }

};

//==========================================================================

class fsr_qcd_G2CCG : public fsr_qcd_G2QQG {

public:  

  fsr_qcd_G2CCG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    fsr_qcd_G2QQG(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info) {}

  vector <int> radAndEmt(int, int colType) { 
    int sign = (colType > 0) ? 1 : -1; 
    int idRadAft = sign * 4;
    return createvector<int>(idRadAft)(-idRadAft)(21);
  }

};
//==========================================================================

class fsr_qcd_G2BBG : public fsr_qcd_G2QQG {

public:  

  fsr_qcd_G2BBG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    fsr_qcd_G2QQG(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info) {}

  vector <int> radAndEmt(int, int colType) { 
    int sign = (colType > 0) ? 1 : -1; 
    int idRadAft = sign * 5;
    return createvector<int>(idRadAft)(-idRadAft)(21);
  }

};

//==========================================================================

class fsr_qcd_G2TTG : public fsr_qcd_G2QQG {

public:  

  fsr_qcd_G2TTG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    fsr_qcd_G2QQG(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info) {}

  vector <int> radAndEmt(int, int colType) { 
    int sign = (colType > 0) ? 1 : -1; 
    int idRadAft = sign * 6;
    return createvector<int>(idRadAft)(-idRadAft)(21);
  }

};

//==========================================================================

class fsr_qcd_Q2QG : public SplittingQCD {

public:  

  fsr_qcd_Q2QG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class fsr_qcd_Q2GQ : public SplittingQCD {

public:  

  fsr_qcd_Q2GQ(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class fsr_qcd_G2GG1 : public SplittingQCD {

public:  

  fsr_qcd_G2GG1(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class fsr_qcd_G2GG2 : public SplittingQCD {

public:  

  fsr_qcd_G2GG2(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class fsr_qcd_G2QQ1 : public SplittingQCD {

public:  

  fsr_qcd_G2QQ1(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }
  bool isPartial() { return false; }
  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class fsr_qcd_G2QQ2 : public SplittingQCD {

public:  

  fsr_qcd_G2QQ2(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }
  bool isPartial() { return false; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class fsr_qcd_Q2qQqbarDist : public SplittingQCD {

public:  

  fsr_qcd_Q2qQqbarDist(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 2; }
  bool isPartial() { return false; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  //vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class fsr_qcd_Q2QbarQQId : public SplittingQCD {

public:  

  fsr_qcd_Q2QbarQQId(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 2; }
  bool isPartial() { return false; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  //vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class isr_qcd_Q2QG : public SplittingQCD {

public:  

  isr_qcd_Q2QG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class isr_qcd_G2GG1 : public SplittingQCD {

public:  

  isr_qcd_G2GG1(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class isr_qcd_G2GG2 : public SplittingQCD {

public:  

  isr_qcd_G2GG2(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class isr_qcd_G2QQ : public SplittingQCD {

public:  

  isr_qcd_G2QQ(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }
  bool isPartial() { return false; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class isr_qcd_Q2GQ : public SplittingQCD {

public:  

  isr_qcd_Q2GQ(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }
  bool isPartial() { return false; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

// Class inheriting from SplittingQCD class.
class isr_qcd_Q2qQqbarDist : public SplittingQCD {

public:  

  isr_qcd_Q2qQqbarDist(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 2; }
  bool isPartial() { return false; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  //vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class isr_qcd_Q2QbarQQId : public SplittingQCD {

public:  

  isr_qcd_Q2QbarQQId(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 2; }
  bool isPartial() { return false; }

  int kinMap ();

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  //vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class fsr_qcd_Q2QG_notPartial : public SplittingQCD {

public:  

  fsr_qcd_Q2QG_notPartial(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }

  int kinMap ();
  bool canUseForBranching() { return true; }
  bool isPartial()  { return false; }
  vector<pair<int,int> > radAndEmtCols(int iRad, int, Event state);

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class fsr_qcd_G2GG_notPartial : public SplittingQCD {

public:  

  fsr_qcd_G2GG_notPartial(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }

  int kinMap ();
  bool canUseForBranching() { return true; }
  bool isPartial()  { return false; }
  vector<pair<int,int> > radAndEmtCols(int iRad, int, Event state);

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

class fsr_qcd_G2QQ_notPartial : public SplittingQCD {

public:  

  fsr_qcd_G2QQ_notPartial(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB,
      coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);
  int nEmissions() { return 1; }

  int kinMap ();
  bool canUseForBranching() { return true; }
  bool isPartial()  { return false; }
  vector<pair<int,int> > radAndEmtCols(int iRad, int, Event state);

  // Return id of mother after splitting.
  int motherID(int idDaughter);

  // Return id of emission.
  int sisterID(int idDaughter);

  // Return id of recombined radiator (before splitting!)
  int radBefID(int idRadAfter, int idEmtAfter);

  vector <int> recPositions( const Event&, int, int);

  // Return colours of recombined radiator (before splitting!)
  pair<int,int> radBefCols(int colRadAfter, int acolRadAfter, 
    int colEmtAfter, int acolEmtAfter);

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

} // end namespace Pythia8

#endif
