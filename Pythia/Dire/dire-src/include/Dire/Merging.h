// Merging.h is a part of the PYTHIA event generator.
// Merging.h is a part of the DIRE plugin to the PYTHIA event generator.
// Copyright (C) 2018 Stefan Prestel.

#ifndef Pythia8_MyMerging_H
#define Pythia8_MyMerging_H

#include "Pythia8/Basics.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/Event.h"
#include "Pythia8/Info.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PartonLevel.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StandardModel.h"
#include "Pythia8/Merging.h"
#include "Pythia8/MergingHooks.h"
#include "Pythia8/LesHouches.h"

#include "Dire/History.h"
#include "Dire/MergingHooks.h"
#include "Dire/WeightContainer.h"

namespace Pythia8 {

class DireSpace;
class DireTimes;

//==========================================================================

// Merging is a wrapper class for the interface of matrix element merging and
// Pythia8.

class MyMerging : public Merging {

public:

  // Constructor.
  MyMerging() { settingsPtr = 0; infoPtr = 0; particleDataPtr = 0;
    rndmPtr = 0; beamAPtr = 0; beamBPtr = 0; trialPartonLevelPtr = 0;
    mergingHooksPtr = 0; myHistory = 0; myLHEF3Ptr = 0; fsr = 0; isr = 0; }

  void setWeightsPtr( WeightContainer* wgtsIn ) { psweights = wgtsIn; } 
  void setShowerPtrs( DireTimes* timesPtr, DireSpace* spacePtr) {
    fsr = timesPtr; isr = spacePtr; }

  DireTimes* fsr;
  DireSpace* isr;

  // Destructor.
  ~MyMerging(){ if (myLHEF3Ptr) delete myLHEF3Ptr; }

  // Initialisation function for internal use inside Pythia source code
  virtual void init();

  // Function to print statistics.
  virtual void statistics();

  //----------------------------------------------------------------------//
  // Functions that implement matrix element merging.
  //----------------------------------------------------------------------//

  // Function to steer different merging prescriptions.
  virtual int mergeProcess( Event& process);

  // Return CKKW-L weight.
  void getSudakovs( double & wt )  const { wt = sudakovs;  return; }
  void getASratios( double & wt )  const { wt = asRatios;  return; }
  void getPDFratios( double & wt ) const { wt = pdfRatios; return; }

  void getSudakovExp( int order, double & wt )  const {
    wt = 0.;
    if (order >= 0 && order < int(sudakovsExp.size()))
      wt = sudakovsExp[order];
    return;
  }
  void getASratioExp( int order, double & wt )  const {
    wt = 0.;
    if (order >= 0 && order < int(asRatiosExp.size()))
      wt = asRatiosExp[order];
    return;
  }
  void getPDFratioExp( int order, double & wt ) const {
    wt = 0.;
    if (order >= 0 && order <= int(pdfRatiosExp.size()))
      wt = pdfRatiosExp[order];
    return;
  }

  void nRealSubtractions (int & nsub) {
    nsub = int(subtractions.size());
    return;
  }
  void getRealSubtraction (const int& iSub, const int& charSize,
    double& val, char * psppoint ) {
    val = subtractions[iSub].first;
    myLHEF3Ptr->setEventPtr(&subtractions[iSub].second);
    myLHEF3Ptr->setEvent();
    string helper = myLHEF3Ptr->getEventString();
    // Right-pad string with whitespace.
    helper.insert(helper.end(), charSize - helper.size(), ' ');
    std::copy(helper.begin(), helper.end(),psppoint);
    psppoint[helper.size()] = '\0';
    return;
  }

  void clearInfos() {
    stoppingScalesSave.clear();
    startingScalesSave.clear();
  }
  void storeInfos();

  vector<double> getStoppingScales() { 
    return stoppingScalesSave;
  }
  vector<double> getStartingScales() { 
    return startingScalesSave;
  }
  vector<double> stoppingScalesSave;
  vector<double> startingScalesSave;

protected:

  //----------------------------------------------------------------------//
  // The members
  //----------------------------------------------------------------------//

  // Make Pythia class friend
  friend class Pythia;

  // Function to perform CKKW-L merging on the event.
  int mergeProcessCKKWL( Event& process);

  // Function to perform UMEPS merging on the event.
  int mergeProcessUMEPS( Event& process);

  // Function to perform NL3 NLO merging on the event.
  int mergeProcessNL3( Event& process);

  // Function to perform UNLOPS merging on the event.
  int mergeProcessUNLOPS( Event& process);

  // Function to apply the merging scale cut on an input event.
  bool cutOnProcess( Event& process);

  // Function to perform CKKW-L merging on the event.
  int calculate( Event& process);

  MyHistory* myHistory;

  bool   generateHistories( const Event& process);
  double getPathIndex( bool useAll = false);
  int    calculateWeights( double RNpath, bool useAll = false);
  int    getStartingConditions( double RNpath, Event& process );

  void   setSudakovs( double wt )  { sudakovs = wt;  return; }
  void   setASratios( double wt )  { asRatios = wt;  return; }
  void   setPDFratios( double wt ) { pdfRatios = wt; return; }

  void setSudakovExp( vector<double> wts ) {
    // Clear previous results.
    sudakovsExp.clear();
    // Store coefficients of Sudakov expansion.
    sudakovsExp.insert(sudakovsExp.end(), wts.begin(), wts.end());
    return;
  }
  void setASratioExp( vector<double> wts ) {
    // Clear previous results.
    asRatiosExp.clear();
    // Store coefficients of Sudakov expansion.
    asRatiosExp.insert(asRatiosExp.end(), wts.begin(), wts.end());
    return;
  }
  void setPDFratiosExp( vector<double> wts ) {
    // Clear previous results.
    pdfRatiosExp.clear();
    // Store coefficients of Sudakov expansion.
    pdfRatiosExp.insert(pdfRatiosExp.end(), wts.begin(), wts.end());
    return;
  }

  void clearSubtractions() { subtractions.clear(); }
  void appendSubtraction( double wt, const Event& event ) {
    subtractions.push_back( make_pair(wt, event) );
    return;
  }
  bool calculateSubtractions();

  double sudakovs, asRatios, pdfRatios;
  vector<double> sudakovsExp, asRatiosExp, pdfRatiosExp;
  vector<pair<double,Event> > subtractions;

  // Create and open file for LHEF 3.0 output.
  LHEF3FromPythia8* myLHEF3Ptr;

  WeightContainer* psweights;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_MyMerging_H
