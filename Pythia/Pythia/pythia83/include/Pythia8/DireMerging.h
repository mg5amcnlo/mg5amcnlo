// DireMerging.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for merging with Dire.

#ifndef Pythia8_DireMerging_H
#define Pythia8_DireMerging_H

#define DIRE_MERGING_VERSION "2.002"

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

#include "Pythia8/DireHistory.h"
#include "Pythia8/DireMergingHooks.h"
#include "Pythia8/DireWeightContainer.h"

namespace Pythia8 {

class DireSpace;
class DireTimes;

//==========================================================================

// Merging is a wrapper class for the interface of matrix element merging and
// Pythia8.

class DireMerging : public Merging {

public:

  // Constructor.
  DireMerging() : totalProbSave(createvector<double>(0.)(0.)(0.)),
    sudakovs(1.), asRatios(1.), pdfRatios(1.), psweights(0), first(true) {
    vector<double> tmp(createvector<double>(0.)(0.)(0.));
    signalProbSave.insert(make_pair("higgs",tmp));
    bkgrndProbSave.insert(make_pair("higgs",tmp));
    signalProbSave.insert(make_pair("higgs-subt",tmp));
    bkgrndProbSave.insert(make_pair("higgs-subt",tmp));
    signalProbSave.insert(make_pair("higgs-nosud",tmp));
    bkgrndProbSave.insert(make_pair("higgs-nosud",tmp));
    signalProbSave.insert(make_pair("qed",tmp));
    bkgrndProbSave.insert(make_pair("qed",tmp));
    signalProbSave.insert(make_pair("qcd",tmp));
    bkgrndProbSave.insert(make_pair("qcd",tmp));
    settingsPtr = 0; infoPtr = 0; particleDataPtr = 0; rndmPtr = 0;
    beamAPtr = 0; beamBPtr = 0; trialPartonLevelPtr = 0;
    mergingHooksPtr = 0; myHistory = 0; fsr = 0; isr = 0;
    direInfoPtr = 0; sum_time_1 = sum_time_2 = 0.; sum_paths = 0;
    enforceCutOnLHE = doMOPS = applyTMSCut = doMerging
      //= doMcAtNloDelta
      = allowReject = doMECs = doMEM = doGenerateSubtractions
      = doGenerateMergingWeights = doExitAfterMerging
      = allowIncompleteReal = false;
    usePDF = true;
    nQuarksMerge = 5;
  }

  void setWeightsPtr( DireWeightContainer* wgtsIn ) { psweights = wgtsIn; }
  void setShowerPtrs( shared_ptr<DireTimes> timesPtr,
    shared_ptr<DireSpace> spacePtr) {fsr = timesPtr; isr = spacePtr; }

  void initPtrs( DireWeightContainer* wgtsIn, shared_ptr<DireTimes> timesPtr,
    shared_ptr<DireSpace> spacePtr, DireInfo* direInfoIn) {
    psweights = wgtsIn;
    fsr = timesPtr;
    isr = spacePtr;
    direInfoPtr = direInfoIn;
  }

  shared_ptr<DireTimes> fsr;
  shared_ptr<DireSpace> isr;
  DireInfo* direInfoPtr;

  // Destructor.
  ~DireMerging(){
    if (myHistory) delete myHistory;
  }

  // Initialisation function for internal use inside Pythia source code
  virtual void init();
  void reset();

  // Function to print statistics.
  virtual void statistics();

  // Functions that implement matrix element merging.

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

  void clearInfos() {
    stoppingScalesSave.clear();
    startingScalesSave.clear();
    mDipSave.clear();
    radSave.clear();
    emtSave.clear();
    recSave.clear();
  }
  void storeInfos();

  vector<double> getStoppingScales() {
    return stoppingScalesSave;
  }
  vector<double> getStartingScales() {
    return startingScalesSave;
  }
  void getStoppingInfo(double scales [100][100], double masses [100][100]);
  vector<double> stoppingScalesSave, startingScalesSave, mDipSave;
  vector<int>    radSave, emtSave, recSave;

  double generateSingleSudakov ( double pTbegAll,
   double pTendAll, double m2dip, int idA, int type, double s = -1.,
   double x = -1.);

  vector<double> getSignalProb(string key) { return signalProbSave[key]; }
  vector<double> getBkgrndProb(string key) { return bkgrndProbSave[key]; }
  vector<double> getTotalProb() { return totalProbSave; }
  vector<double> totalProbSave;
  map<string, vector<double> > signalProbSave, bkgrndProbSave;
  void clearClassifier() {
    for ( map<string, vector<double> >::iterator it = signalProbSave.begin();
      it != signalProbSave.end(); ++it) for (size_t i=0; i<it->second.size();
        ++i) it->second[i]=0.;
    for ( map<string, vector<double> >::iterator it = bkgrndProbSave.begin();
      it != bkgrndProbSave.end(); ++it) for (size_t i=0; i<it->second.size();
        ++i) it->second[i]=0.;
    for (size_t i=0; i<totalProbSave.size(); ++i) totalProbSave[i]=0.;
  }

protected:

  // The members.
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

  DireHistory* myHistory;

  bool   generateHistories( const Event& process, bool orderedOnly = true);
  void   tagHistories();

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

  bool generateUnorderedPoint(Event& process);

  double sudakovs, asRatios, pdfRatios;
  vector<double> sudakovsExp, asRatiosExp, pdfRatiosExp;
  vector<pair<double,Event> > subtractions;

  DireWeightContainer* psweights;

  double sum_time_1, sum_time_2;
  int sum_paths;

  bool enforceCutOnLHE, doMOPS, applyTMSCut, doMerging,
       usePDF, allowReject, doMECs, doMEM, doGenerateSubtractions,
       doGenerateMergingWeights, doExitAfterMerging, allowIncompleteReal;
  int nQuarksMerge;

  bool first;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_DireMerging_H
