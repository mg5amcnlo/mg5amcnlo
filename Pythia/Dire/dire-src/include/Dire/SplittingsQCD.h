
#ifndef Pythia8_SplittingsQCD_H
#define Pythia8_SplittingsQCD_H

#define ZETA3 1.202056903159594
#define DIRE_SPLITTINGSQCD_VERSION "2.000"

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
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    Splitting(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info)
    { init(); }
  virtual ~SplittingQCD() {}

  void init();

  // VARIABLES
  double CA, TR, CF, pTmin;
  int NF_qcd_fsr;
  bool usePDFalphas, doVariations;
  double alphaS2pi;

  AlphaStrong alphaS;

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

};

//==========================================================================

class fsr_qcd_Q2QG : public SplittingQCD {

public:  

  fsr_qcd_Q2QG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int splitType = 0,
    double m2RadBef = 0., double m2Rad = 0., double m2Rec = 0.,
    double m2Emt = 0., const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>() );

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class fsr_qcd_Q2GQ : public SplittingQCD {

public:  

  fsr_qcd_Q2GQ(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int splitType = 0,
    double m2RadBef = 0., double m2Rad = 0., double m2Rec = 0.,
    double m2Emt = 0., const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class fsr_qcd_G2GG1 : public SplittingQCD {

public:  

  fsr_qcd_G2GG1(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int splitType = 0,
    double m2RadBef = 0., double m2Rad = 0., double m2Rec = 0.,
    double m2Emt = 0., const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class fsr_qcd_G2GG2 : public SplittingQCD {

public:  

  fsr_qcd_G2GG2(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int splitType = 0,
    double m2RadBef = 0., double m2Rad = 0., double m2Rec = 0.,
    double m2Emt = 0., const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class fsr_qcd_G2QQ1 : public SplittingQCD {

public:  

  fsr_qcd_G2QQ1(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int splitType = 0,
    double m2RadBef = 0., double m2Rad = 0., double m2Rec = 0.,
    double m2Emt = 0., const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class fsr_qcd_G2QQ2 : public SplittingQCD {

public:  

  fsr_qcd_G2QQ2(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int splitType = 0,
    double m2RadBef = 0., double m2Rad = 0., double m2Rec = 0.,
    double m2Emt = 0., const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class fsr_qcd_Q2qQqbarDist : public SplittingQCD {

public:  

  fsr_qcd_Q2qQqbarDist(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int splitType = 0,
    double m2RadBef = 0., double m2Rad = 0., double m2Rec = 0.,
    double m2Emt = 0., const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class fsr_qcd_Q2QbarQQId : public SplittingQCD {

public:  

  fsr_qcd_Q2QbarQQId(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int splitType = 0,
    double m2RadBef = 0., double m2Rad = 0., double m2Rec = 0.,
    double m2Emt = 0., const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class isr_qcd_Q2QG : public SplittingQCD {

public:  

  isr_qcd_Q2QG(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int, double, double, 
    double, double, const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class isr_qcd_G2GG1 : public SplittingQCD {

public:  

  isr_qcd_G2GG1(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int splitType = 0,
    double m2RadBef = 0., double m2Rad = 0., double m2Rec = 0.,
    double m2Emt = 0., const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class isr_qcd_G2GG2 : public SplittingQCD {

public:  

  isr_qcd_G2GG2(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int splitType = 0, 
    double m2RadBef = 0., double m2Rad = 0., double m2Rec = 0.,
    double m2Emt = 0., const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class isr_qcd_G2QQ : public SplittingQCD {

public:  

  isr_qcd_G2QQ(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int, double, double, 
    double, double, const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class isr_qcd_Q2GQ : public SplittingQCD {

public:  

  isr_qcd_Q2GQ(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int splitType = 0,
    double m2RadBef = 0., double m2Rad = 0., double m2Rec = 0.,
    double m2Emt = 0., const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

// Class inheriting from SplittingQCD class.
class isr_qcd_Q2qQqbarDist : public SplittingQCD {

public:  

  isr_qcd_Q2qQqbarDist(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  //// Return kernel for new splitting.
  //double kernel(double z, double pT2, double m2dip, int, double, double, 
  //  double, double, const Event& state = Event(), int order = -1,
  //  map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

class isr_qcd_Q2QbarQQId : public SplittingQCD {

public:  

  isr_qcd_Q2QbarQQId(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSM, Info* info) :
    SplittingQCD(idIn, softRS, settings, particleData, rndm, beamA, beamB, coupSM, info){}

  bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL);

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

  double gaugeFactor ( int=0, int=0 );
  double symmetryFactor ( int=0, int=0 );

  // Pick z for new splitting.
  double zSplit(double zMinAbs, double zMaxAbs, double m2dip);

  // New overestimates, z-integrated versions.
  double overestimateInt(double zMinAbs,double zMaxAbs,
    double pT2Old, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double overestimateDiff(double z, double m2dip, int order = -1);

  // Return kernel for new splitting.
  double kernel(double z, double pT2, double m2dip, int splitType = 0,
    double m2RadBef = 0., double m2Rad = 0., double m2Rec = 0.,
    double m2Emt = 0., const Event& state = Event(), int order = -1,
    map<string,double> aux= map<string,double>());

  // Functions to calculate the kernel from SplitInfo information.
  bool calc(const Event& state = Event(), int order = -1);

};

} // end namespace Pythia8

#endif
