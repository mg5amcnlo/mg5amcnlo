// MyMergingHooks.h is a part of the PYTHIA event generator.
// Copyright (C) 2016 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file is written by Stefan Prestel.
// Header file to allow user access to program at different stages.
// MyHardProcess: Container class for the hard process to be merged. Holds the
//              bookkeeping of particles not be be reclustered
// MyMergingHooks: Steering class for matrix element merging. Some functions can
//               be redefined in a derived class to have access to the merging

#ifndef Pythia8_MyMergingHooks_H
#define Pythia8_MyMergingHooks_H

#include "Pythia8/Basics.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/Event.h"
#include "Pythia8/Info.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PartonSystems.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"


namespace Pythia8 {

class PartonLevel;

//==========================================================================

// Declaration of hard process class
// This class holds information on the desired hard 2->2 process
// for the merging.
// This class is a container class for History class use.

class MyHardProcess : public HardProcess {

public:


  // Constructor with core process input
  void initOnProcess( string process, ParticleData* particleData);

  // Function to translate the process string (in MG/ME notation)
  void translateProcessString( string process);

  // Function to check whether the sets of candidates Pos1, Pos2, together
  // with the proposed candidate iPos give an allowed hard process state
  bool allowCandidates(int iPos, vector<int> Pos1, vector<int> Pos2,
    const Event& event);
  // Function to identify the hard subprocess in the current event
  void storeCandidates( const Event& event, string process);
  // Function to check if the particle event[iPos] matches any of
  // the stored outgoing particles of the hard subprocess
  bool matchesAnyOutgoing(int iPos, const Event& event);
  // Function to check if instead of the particle event[iCandidate], another
  // particle could serve as part of the hard process. Assumes that iCandidate
  // is already stored as part of the hard process.
  bool findOtherCandidates(int iPos, const Event& event, bool doReplace);
  // Function to exchange a stored hard process candidate with another choice.
  bool exchangeCandidates( vector<int> candidates1, vector<int> candidates2,
    map<int,int> further1, map<int,int> further2);

};

//==========================================================================

// MyMergingHooks is base class for user input to the merging procedure.

class MyMergingHooks : public MergingHooks {

public:

  // Function to return the number of clustering steps for the current event
  virtual int getNumberOfClusteringSteps(const Event& event,
    bool resetNjetMax = false);

  //----------------------------------------------------------------------//
  // Functions used as default merging scales
  //----------------------------------------------------------------------//

  // Function to check if the input particle is a light jet, i.e. should be
  // checked against the merging scale defintion.
  bool checkAgainstCut( const Particle& particle);
  // Function to return the value of the merging scale function in the
  // current event.
  double tmsNow( const Event& event );
  // Find the minimal Lund pT between coloured partons in the event
  //double scalems( const Event& event, bool withColour);
  double scalems( const Event& event);

  //----------------------------------------------------------------------//
  // Functions to steer shower evolution (public to allow for PS plugin)
  //----------------------------------------------------------------------//

  // Flag to indicate trial shower usage.
  void doIgnoreEmissions( bool doIgnoreIn ) {
    doIgnoreEmissionsSave = doIgnoreIn;
  }
  // Function to allow not counting a trial emission.
  bool canVetoEmission() { return !doIgnoreEmissionsSave; }
  // Function to check if emission should be rejected.
  bool doVetoEmission( const Event& );

  void init();

protected:

  //----------------------------------------------------------------------//
  //----------------------------------------------------------------------//
  // Functions to steer merging code
  //----------------------------------------------------------------------//

  // Flag to indicate if events should be vetoed.
  void doIgnoreStep( bool doIgnoreIn ) { doIgnoreStepSave = doIgnoreIn; }
  // Function to allow event veto.
  bool canVetoStep() { return !doIgnoreStepSave; }

public:

  // Function to check event veto.
  virtual bool doVetoStep( const Event& process, const Event& event,
    bool doResonance = false );

  // Set starting scales
  virtual bool setShowerStartingScales( bool isTrial, bool doMergeFirstEmm,
    double& pTscaleIn, const Event& event,
    double& pTmaxFSRIn, bool& limitPTmaxFSRin,
    double& pTmaxISRIn, bool& limitPTmaxISRin,
    double& pTmaxMPIIn, bool& limitPTmaxMPIin );


  bool doMOPSSave;
  bool doMOPS() { return doMOPSSave; }

protected:

  //----------------------------------------------------------------------//
  // Functions for internal merging scale definions
  //----------------------------------------------------------------------//

  // Function to compute "pythia pT separation" from Particle input
  double rhoPythia(const Event& event, int rad, int emt, int rec,
    int ShowerType);

  // Function to find a colour (anticolour) index in the input event,
  // used to find colour-connected recoilers
  int findColour(int col, int iExclude1, int iExclude2,
    const Event& event, int type, bool isHardIn);

  //----------------------------------------------------------------------//
  // Functions for weight management
  //----------------------------------------------------------------------//

  // Function to get the CKKW-L weight for the current event
  double getWeightNLO() { return (weightCKKWLSave - weightFIRSTSave);}
  // Return CKKW-L weight.
  double getWeightCKKWL() { return weightCKKWLSave; }
  // Return O(\alpha_s) weight.
  double getWeightFIRST() { return weightFIRSTSave; }
  // Set CKKW-L weight.
  void setWeightCKKWL( double weightIn){
    weightCKKWLSave = weightIn;
    if ( !includeWGTinXSEC() ) infoPtr->setWeightCKKWL(weightIn); }
  // Set O(\alpha_s) weight.
  void setWeightFIRST( double weightIn){
    weightFIRSTSave = weightIn;
    infoPtr->setWeightFIRST(weightIn); }

//  void setEmtIDs(vector<int> in) { allowedEmtID = in; }
//  void clearEmtIDs() { allowedEmtID.clear(); }
//  vector<int> getEmtIDs() { return allowedEmtID; }
//  vector<int> allowedEmtID;

  // Function to allow effective gg -> EW boson couplings.
  bool hasEffectiveG2EW() {
    if ( getProcessString().compare("pp>h") == 0 ) return true;
    return false; }

  // Function to allow effective gg -> EW boson couplings.
  bool allowEffectiveVertex( vector<int> in, vector<int> out) {
    if ( getProcessString().compare("ta+ta->jj") == 0 
      || getProcessString().compare("ta-ta+>jj") == 0 ) {
      int nInFermions(0), nOutFermions(0), nOutBosons(0);
      for (int i=0; i < int(in.size()); ++i)
        if (abs(in[i])<20) nInFermions++;
      for (int i=0; i < int(out.size()); ++i) {
        if (abs(out[i])<20) nOutFermions++;
        if (abs(out[i])>20) nOutBosons++;
      }
      return (nInFermions%2==0 && nOutFermions%2==0);
    }

    int nInG(0), nOutWp(0), nOutWm(0), nOutH(0);
    for (int i=0; i < int(in.size()); ++i)
        if (in[i]==21) nInG++;
    for (int i=0; i < int(out.size()); ++i) {
      if (out[i] == 24) nOutWp++;
      if (out[i] ==-24) nOutWm++;
      if (out[i] == 25) nOutH++;
    }

    if (nInG==2 && nOutWp+nOutWp > 0 && nOutWp-nOutWm == 0) return true;
    if (nInG==2 && nOutH > 0) return true;

    return false;
  }

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_MyMergingHooks_H
