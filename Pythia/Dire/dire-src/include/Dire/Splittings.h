
#ifndef Pythia8_Splittings_H
#define Pythia8_Splittings_H

#define DIRE_SPLITTINGS_VERSION "2.000"

#include "Pythia8/Basics.h"
#include "Pythia8/Pythia.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StandardModel.h"
#include "Pythia8/Info.h"
#include "Dire/SplitInfo.h"
#include "Dire/Basics.h"

namespace Pythia8 {

class DireSpace;
class DireTimes;
 
//==========================================================================

class Splitting {

public:  

  // Constructor and destructor.
  Splitting(string idIn, int softRS, Settings* settings,
    ParticleData* particleData, Rndm* rndm, BeamParticle* beamA,
    BeamParticle* beamB, CoupSM* coupSMPtrIn, Info* infoPtrIn) :
      id(idIn), correctionOrder(softRS), settingsPtr(settings),
      particleDataPtr(particleData), rndmPtr(rndm), beamAPtr(beamA),
      beamBPtr(beamB),  coupSMPtr(coupSMPtrIn), infoPtr(infoPtrIn)
    { init(); splitInfo.storeName(name()); }
  virtual ~Splitting() {}

  void init();

public:

  double renormMultFac;

  string id;
  int correctionOrder;
  Settings* settingsPtr;
  ParticleData* particleDataPtr;
  Rndm* rndmPtr;
  BeamParticle* beamAPtr;
  BeamParticle* beamBPtr;
  CoupSM* coupSMPtr;
  Info* infoPtr;

  map<string,double> kernelVals;

  string name () {return id;}

  virtual bool canRadiate ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL)
    {return false;}

  /*virtual bool canCluster ( const Event&, map<string,int>,
    map<string,bool> = map<string,bool>(), Settings* = NULL,
    PartonSystems* = NULL, BeamParticle* = NULL)
    {return false;}*/

  virtual int kinMap () {return 1;}

  // Return id of mother after splitting.
  virtual int motherID(int) {return 0;}

  // Return id of emission.
  virtual int sisterID(int) {return 0;}

  // Return id of recombined radiator (before splitting!)
  virtual int radBefID(int, int) {return 0;}

  // Return colours of recombined radiator (before splitting!)
  virtual pair<int,int> radBefCols(int, int, int, int)
    {return make_pair(0,0);}

  // Return color factor for splitting.
  virtual double gaugeFactor (int, int) {return 1.;}

  // Return symmetry factor for splitting.
  virtual double symmetryFactor (int, int) {return 1.;}

  // Pick z for new splitting.
  virtual double zSplit(double, double, double) {return 0.5;}

  // New overestimates, z-integrated versions.
  virtual double overestimateInt(double, double, double, double, int = -1)
   { return 0.;}

  // Return kernel for new splitting.
  virtual double overestimateDiff(double, double, int = -1) {return 1.;}

  // Return kernel for new splitting.
  virtual double kernel(double, double, double, int = 0, double = 0.,
    double = 0., double = 0., double = 0., const Event& = Event(), int = -1,
     map<string,double> = map<string,double>())
    { return 0.; }

  // Functions to store and retrieve all the variants of the kernel.
  virtual double getKernel(string = "");
  virtual map<string,double> getKernelVals() { return kernelVals; }
  virtual void   clearKernels()         { kernelVals.clear(); }

  SplitInfo splitInfo;
  /*virtual pair<int,int> radAft_and_emtAft(int,int,double)
    { return make_pair(0,0); }*/

  // Functions to calculate the kernel from SplitInfo information.
  virtual bool calc(const Event& = Event(), int = -1) { return false; }

  DireSpace* isr;
  DireTimes* fsr;
  void setTimesPtr(DireTimes* fsrIn) { fsr=fsrIn;}
  void setSpacePtr(DireSpace* isrIn) { isr=isrIn;}

};

} // end namespace Pythia8

#endif
