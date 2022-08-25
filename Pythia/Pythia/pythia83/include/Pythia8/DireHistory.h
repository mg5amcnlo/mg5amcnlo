// DireHistory.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for Dire history classes.

#ifndef Pythia8_DireHistory_H
#define Pythia8_DireHistory_H

#include "Pythia8/Basics.h"
#include "Pythia8/BeamParticle.h"
#include "Pythia8/Event.h"
#include "Pythia8/Info.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PartonLevel.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"
#include "Pythia8/StandardModel.h"
#include "Pythia8/MergingHooks.h"
#include "Pythia8/SimpleWeakShowerMEs.h"
#include "Pythia8/DireWeightContainer.h"

namespace Pythia8 {

class DireTimes;
class DireSpace;

//==========================================================================

// Declaration of Clustering class.
// This class holds information about one radiator, recoiler, emitted system.
// This class is a container class for MyHistory class use.

class DireClustering {

public:

  int radPos() const { return emittor; }
  int emtPos() const { return emitted; }
  int recPos() const { return recoiler; }
  const Particle* rad() { return radSave; }
  const Particle* emt() { return emtSave; }
  const Particle* rec() { return recSave; }

  // The emitted parton location.
  int emitted;
  // The emittor parton
  int emittor;
  // The recoiler parton.
  int recoiler;
  // The colour connected recoiler (Can be different for ISR)
  int partner;
  // The scale associated with this clustering.
  double pTscale;

  const Particle* radSave;
  const Particle* emtSave;
  const Particle* recSave;

  // The flavour of the radiator prior to the emission.
  int flavRadBef;
  // Spin of the radiator before the splitting.
  int spinRadBef;
  // The radiator before the splitting.
  int radBef;
  // The recoiler before the splitting.
  int recBef;

  // Name of the splitting.
  string splitName;
  string name() const { return splitName;}

  // Default constructor
  DireClustering(){
    emitted    = 0;
    emittor    = 0;
    recoiler   = 0;
    partner    = 0;
    pTscale    = 0.;
    radSave    = 0;
    recSave    = 0;
    emtSave    = 0;
    flavRadBef = 0;
    spinRadBef = 9;
    recBef     = 0;
    radBef     = 0;
    splitName  = "";
  }

  // Default destructor
  ~DireClustering(){}

  // Copy constructor
  DireClustering( const DireClustering& inSystem ){
    emitted    = inSystem.emitted;
    emittor    = inSystem.emittor;
    recoiler   = inSystem.recoiler;
    partner    = inSystem.partner;
    pTscale    = inSystem.pTscale;
    flavRadBef = inSystem.flavRadBef;
    spinRadBef = inSystem.spinRadBef;
    radBef     = inSystem.radBef;
    recBef     = inSystem.recBef;
    radSave    = inSystem.radSave;
    emtSave    = inSystem.emtSave;
    recSave    = inSystem.recSave;
    splitName  = inSystem.splitName;
  }

  // Assignment operator.
  DireClustering & operator=(const DireClustering& c) { if (this != &c)
    { emitted = c.emitted; emittor = c.emittor; recoiler = c.recoiler;
      partner = c.partner; pTscale = c.pTscale; flavRadBef = c.flavRadBef;
      spinRadBef = c.spinRadBef; radBef = c.radBef; recBef = c.recBef;
      radSave = c.radSave; emtSave = c.emtSave; recSave = c.recSave;
      splitName  = c.splitName;} return *this; }

  // Constructor with input
  DireClustering( int emtPosIn, int radPosIn, int recPosIn, int partnerIn,
    double pTscaleIn, const Particle* radIn, const Particle* emtIn,
    const Particle* recIn, string splitNameIn,
    int flavRadBefIn = 0, int spinRadBefIn = 9,
    int radBefIn = 0, int recBefIn = 0)
    : emitted(emtPosIn), emittor(radPosIn), recoiler(recPosIn),
      partner(partnerIn), pTscale(pTscaleIn), radSave(radIn), emtSave(emtIn),
      recSave(recIn), flavRadBef(flavRadBefIn), spinRadBef(spinRadBefIn),
      radBef(radBefIn), recBef(recBefIn), splitName(splitNameIn) {}

  // Function to return pythia pT scale of clustering
  double pT() const { return pTscale; }

  // Function to return the dipole mass for this clustering.
  double mass() const {
    double sik = 2.*radSave->p()*recSave->p();
    double sij = 2.*radSave->p()*emtSave->p();
    double sjk = 2.*emtSave->p()*recSave->p();

    double m2=-1.;
    if      ( radSave->isFinal() &&  recSave->isFinal()) m2 = sik+sij+sjk;
    else if ( radSave->isFinal() && !recSave->isFinal()) m2 =-sik+sij-sjk;
    else if (!radSave->isFinal() &&  recSave->isFinal()) m2 =-sik-sij+sjk;
    else if (!radSave->isFinal() && !recSave->isFinal()) m2 = sik-sij-sjk;
    return sqrt(m2);
  }

  // print for debug
  void list() const;

};

//==========================================================================

// Declaration of MyHistory class
//
// A MyHistory object represents an event in a given step in the CKKW-L
// clustering procedure. It defines a tree-like recursive structure,
// where the root node represents the state with n jets as given by
// the matrix element generator, and is characterized by the member
// variable mother being null. The leaves on the tree corresponds to a
// fully clustered paths where the original n-jets has been clustered
// down to the Born-level state. Also states which cannot be clustered
// down to the Born-level are possible - these will be called
// incomplete. The leaves are characterized by the vector of children
// being empty.

class DireHistory {

public:

  // The only constructor. Default arguments are used when creating
  // the initial history node. The \a depth is the maximum number of
  // clusterings requested. \a scalein is the scale at which the \a
  // statein was clustered (should be set to the merging scale for the
  // initial history node. \a beamAIn and beamBIn are needed to
  // calcutate PDF ratios, \a particleDataIn to have access to the
  // correct masses of particles. If \a isOrdered is true, the previous
  // clusterings has been ordered. \a is the PDF ratio for this
  // clustering (=1 for FSR clusterings). \a probin is the accumulated
  // probabilities for the previous clusterings, and \ mothin is the
  // previous history node (null for the initial node).
  DireHistory( int depthIn,
           double scalein,
           Event statein,
           DireClustering c,
           MergingHooksPtr mergingHooksPtrIn,
           BeamParticle beamAIn,
           BeamParticle beamBIn,
           ParticleData* particleDataPtrIn,
           Info* infoPtrIn,
           PartonLevel* showersIn,
           shared_ptr<DireTimes> fsrIn,
           shared_ptr<DireSpace> isrIn,
           DireWeightContainer* psweightsIn,
           CoupSM* coupSMPtrIn,
           bool isOrdered,
           bool isAllowed,
           double clusterProbIn,
           double clusterCouplIn,
           double prodOfProbsIn,
           double prodOfProbsFullIn,
           DireHistory * mothin);

  // The destructor deletes each child.
  ~DireHistory()
    { for ( int i = 0, N = children.size(); i < N; ++i ) delete children[i]; }

  // Function to project paths onto desired paths.
  bool projectOntoDesiredHistories();

  // For CKKW-L, NL3 and UMEPS:
  // In the initial history node, select one of the paths according to
  // the probabilities. This function should be called for the initial
  // history node.
  // IN  trialShower*    : Previously initialised trialShower object,
  //                       to perform trial showering and as
  //                       repository of pointers to initialise alphaS
  //     PartonSystems* : PartonSystems object needed to initialise
  //                      shower objects
  // OUT double         : (Sukadov) , (alpha_S ratios) , (PDF ratios)
  double weightTREE(PartonLevel* trial, AlphaStrong * asFSR,
    AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN);

  double weightMOPS(PartonLevel* trial, AlphaStrong * as, AlphaEM * aem,
    double RN);
  vector<double> weightMEM(PartonLevel* trial, AlphaStrong* as, AlphaEM* aem,
    double RN);
  double weightMEC() { return MECnum/MECden; }

  // For default NL3:
  // Return weight of virtual correction and subtractive for NL3 merging
  double weightLOOP(PartonLevel* trial, double RN);
  // Return O(\alpha_s)-term of CKKWL-weight for NL3 merging
  double weightFIRST(PartonLevel* trial, AlphaStrong* asFSR,
    AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN,
    Rndm* rndmPtr);


  // For UMEPS:
  double weight_UMEPS_TREE(PartonLevel* trial, AlphaStrong * asFSR,
    AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN);
  double weight_UMEPS_SUBT(PartonLevel* trial, AlphaStrong * asFSR,
    AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN);


  // For unitary NL3:
  double weight_UNLOPS_TREE(PartonLevel* trial, AlphaStrong * asFSR,
    AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN,
    int depthIn = -1);
  double weight_UNLOPS_SUBT(PartonLevel* trial, AlphaStrong * asFSR,
    AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN,
    int depthIn = -1);
  double weight_UNLOPS_LOOP(PartonLevel* trial, AlphaStrong * asFSR,
     AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN,
     int depthIn = -1);
  double weight_UNLOPS_SUBTNLO(PartonLevel* trial, AlphaStrong * asFSR,
    AlphaStrong * asISR, AlphaEM * aemFSR, AlphaEM * aemISR, double RN,
    int depthIn = -1);
  double weight_UNLOPS_CORRECTION( int order, PartonLevel* trial,
    AlphaStrong* asFSR, AlphaStrong * asISR, AlphaEM * aemFSR,
    AlphaEM * aemISR, double RN, Rndm* rndmPtr );

  // Function to check if any allowed histories were found
  bool foundAllowedHistories() {
    return (children.size() > 0 && foundAllowedPath); }
  // Function to check if any ordered histories were found
  bool foundOrderedHistories() {
    return (children.size() > 0 && foundOrderedPath); }
  // Function to check if any ordered histories were found
  bool foundCompleteHistories() {
    return (children.size() > 0 && foundCompletePath); }

  // Function to set the state with complete scales for evolution
  void getStartingConditions( const double RN, Event& outState );
  // Function to get the state with complete scales for evolution
  bool getClusteredEvent( const double RN, int nSteps, Event& outState);
  // Function to get the first reclustered state above the merging scale.
  bool getFirstClusteredEventAboveTMS( const double RN, int nDesired,
    Event& process, int & nPerformed, bool updateProcess = true );
  // Function to return the depth of the history (i.e. the number of
  // reclustered splittings)
  int nClusterings();

  // Function to get the lowest multiplicity reclustered event
  Event lowestMultProc( const double RN) {
    // Return lowest multiplicity state
    return (select(RN)->state);
  }

  // Calculate and return pdf ratio
  double getPDFratio( int side, bool forSudakov, bool useHardPDF,
                      int flavNum, double xNum, double muNum,
                      int flavDen, double xDen, double muDen);

  // Function to print the history that would be chosen from the random number
  // RN. Mainly for debugging.
  void printHistory( const double RN );
  // Function to print the states in a history, starting from the hard process.
  // Mainly for debugging.
  void printStates();

  // Make Pythia class friend
  friend class Pythia;
  // Make Merging class friend
  friend class DireMerging;
  friend class DireTimes;
  friend class DireSpace;

private:

  // Number of trial emission to use for calculating the average number of
  // emissions
  static const int NTRIAL;

  // Define size of windows around the charm and bottom thresholds. This
  // parameter is needed to define the flavour threshold region.
  static const double MCWINDOW, MBWINDOW, PROBMAXFAC;

  // Function to set all scales in the sequence of states. This is a
  // wrapper routine for setScales and setEventScales methods
  void setScalesInHistory();

  // Function to find the index (in the mother histories) of the
  // child history, thus providing a way access the path from both
  // initial history (mother == 0) and final history (all children == 0)
  // IN vector<int> : The index of each child in the children vector
  //                  of the current history node will be saved in
  //                  this vector
  // NO OUTPUT
  void findPath(vector<int>& out);

  // Functions to set the  parton production scales and enforce
  // ordering on the scales of the respective clusterings stored in
  // the MyHistory node:
  // Method will start from lowest multiplicity state and move to
  // higher states, setting the production scales the shower would
  // have used.
  // When arriving at the highest multiplicity, the method will switch
  // and go back in direction of lower states to check and enforce
  // ordering for unordered histories.
  // IN vector<int> : Vector of positions of the chosen child
  //                  in the mother history to allow to move
  //                  in direction initial->final along path
  //    bool        : True: Move in direction low->high
  //                       multiplicity and set production scales
  //                  False: Move in direction high->low
  //                       multiplicity and check and enforce
  //                       ordering
  // NO OUTPUT
  void setScales( vector<int> index, bool forward);

  // Function to find a particle in all higher multiplicity events
  // along the history path and set its production scale to the input
  // scale
  // IN  int iPart       : Parton in refEvent to be checked / rescaled
  //     Event& refEvent : Reference event for iPart
  //     double scale    : Scale to be set as production scale for
  //                       unchanged copies of iPart in subsequent steps
  void scaleCopies(int iPart, const Event& refEvent, double rho);

  // Function to set the OVERALL EVENT SCALES [=state.scale()] to
  // the scale of the last clustering
  // NO INPUT
  // NO OUTPUT
  void setEventScales();

  // Function to print information on the reconstructed scales in one path.
  // For debug only.
  void printScales() {
    int type = (!mother) ? 0
             : ( clusterIn.rad()->isFinal() &&  clusterIn.rec()->isFinal()) ? 2
             : ( clusterIn.rad()->isFinal() && !clusterIn.rec()->isFinal()) ? 1
             : (!clusterIn.rad()->isFinal() &&  clusterIn.rec()->isFinal()) ?
               -1 : -2;
    cout << scientific << setprecision(6);
    cout << " size " << state.size() << " scale " << scale
    << " clusterIn " << clusterIn.pT() << " state.scale() " << state.scale()
    << " scaleEffective " << scaleEffective;
    if (type==-2) cout << "\t\t" << " II splitting emt="
      << clusterIn.emt()->id() << endl;
    if (type==-1) cout << "\t\t" << " IF splitting emt="
      << clusterIn.emt()->id() << endl;
    if (type== 1) cout << "\t\t" << " FI splitting emt="
      << clusterIn.emt()->id() << endl;
    if (type== 2) cout << "\t\t" << " FF splitting emt="
      << clusterIn.emt()->id() << endl;
    if ( mother ) mother->printScales();
  }

  // Function to project paths onto desired paths.
  bool trimHistories();
  // Function to tag history for removal.
  void remove(){ doInclude = false; }
  // Function to return flag of allowed histories to choose from.
  bool keep(){ return doInclude; }
  // Function implementing checks on a paths, for deciding if the path should
  // be considered valid or not.
  bool keepHistory();
  // Function to check if a path is ordered in evolution pT.
  bool isOrderedPath( double maxscale);

  bool hasScalesAboveCutoff();

  bool followsInputPath();

  // Function to check if all reconstucted states in a path pass the merging
  // scale cut.
  bool allIntermediateAboveRhoMS( double rhoms, bool good = true );
  // Function to check if any ordered paths were found (and kept).
  bool foundAnyOrderedPaths();

  // Functions to return the event with nSteps additional partons
  // INPUT  int   : Number of splittings in the event,
  //                as counted from core 2->2 process
  // OUTPUT Event : event with nSteps additional partons
  Event clusteredState( int nSteps);

  // Function to choose a path from all paths in the tree
  // according to their splitting probabilities
  // IN double    : Random number
  // OUT MyHistory* : Leaf of history path chosen
  DireHistory * select(double rnd);

  // For a full path, find the weight calculated from the ratio of
  // couplings, the no-emission probabilities, and possible PDF
  // ratios. This function should only be called for the last history
  // node of a full path.
  // IN  TimeShower : Already initialised shower object to be used as
  //                  trial shower
  //     double     : alpha_s value used in ME calculation
  //     double     : Maximal mass scale of the problem (e.g. E_CM)
  //     AlphaStrong: Initialised shower alpha_s object for FSR alpha_s
  //                  ratio calculation
  //     AlphaStrong: Initialised shower alpha_s object for ISR alpha_s
  //                  ratio calculation (can be different from previous)
  double weight(PartonLevel* trial, double as0, double aem0,
    double maxscale, double pdfScale, AlphaStrong * asFSR, AlphaStrong * asISR,
    AlphaEM * aemFSR, AlphaEM * aemISR, double& asWeight, double& aemWeight,
    double& pdfWeight);

  // Function to return the \alpha_s-ratio part of the CKKWL weight.
  double weightALPHAS( double as0, AlphaStrong * asFSR,
    AlphaStrong * asISR,  int njetMin = -1 , int njetMax = -1 );
  // Function to return the \alpha_em-ratio part of the CKKWL weight.
  double weightALPHAEM( double aem0, AlphaEM * aemFSR,
    AlphaEM * aemISR, int njetMin = -1, int njetMax = -1 );
  // Function to return the coupling ratio part of the CKKWL weight.
  vector<double> weightCouplings();
  vector<double> weightCouplingsDenominator();
  // Function to return the PDF-ratio part of the CKKWL weight.
  double weightPDFs( double maxscale, double pdfScale, int njetMin = -1,
    int njetMax = -1 );
  // Function to return the no-emission probability part of the CKKWL weight.
  double weightEmissions( PartonLevel* trial, int type, int njetMin,
    int njetMax, double maxscale );
  vector<double> weightEmissionsVec( PartonLevel* trial, int type,
    int njetMin, int njetMax, double maxscale );

  // Function to generate the O(\alpha_s)-term of the CKKWL-weight
  double weightFirst(PartonLevel* trial, double as0, double muR,
    double maxscale, AlphaStrong * asFSR, AlphaStrong * asISR, Rndm* rndmPtr );

  // Function to generate the O(\alpha_s)-term of the \alpha_s-ratios
  // appearing in the CKKWL-weight.
  double weightFirstALPHAS( double as0, double muR, AlphaStrong * asFSR,
    AlphaStrong * asISR);
  // Function to generate the O(\alpha_em)-term of the \alpha_em-ratios
  // appearing in the CKKWL-weight.
  double weightFirstALPHAEM( double aem0, double muR, AlphaEM * aemFSR,
    AlphaEM * aemISR);
  // Function to generate the O(\alpha_s)-term of the PDF-ratios
  // appearing in the CKKWL-weight.
  double weightFirstPDFs( double as0, double maxscale, double pdfScale,
    Rndm* rndmPtr );
  // Function to generate the O(\alpha_s)-term of the no-emission
  // probabilities appearing in the CKKWL-weight.
  double weightFirstEmissions(PartonLevel* trial, double as0, double maxscale,
    AlphaStrong * asFSR, AlphaStrong * asISR, bool fixpdf, bool fixas );

  // Function to return the default factorisation scale of the hard process.
  double hardFacScale(const Event& event);
  // Function to return the default renormalisation scale of the hard process.
  double hardRenScale(const Event& event);

  double hardProcessScale( const Event& event);

  double hardStartScale(const Event& event);

  // Perform a trial shower using the \a pythia object between
  // maxscale down to this scale and return the corresponding Sudakov
  // form factor.
  // IN  trialShower : Shower object used as trial shower
  //     double     : Maximum scale for trial shower branching
  // OUT  0.0       : trial shower emission outside allowed pT range
  //      1.0       : trial shower successful (any emission was below
  //                  the minimal scale )
  vector<double> doTrialShower(PartonLevel* trial, int type, double maxscale,
    double minscale = 0.);

  // Function to bookkeep the indices of weights generated in countEmissions
  bool updateind(vector<int> & ind, int i, int N);

  // Function to count number of emissions between two scales for NLO merging
  vector<double> countEmissions(PartonLevel* trial, double maxscale,
    double minscale, int showerType, double as0, AlphaStrong * asFSR,
    AlphaStrong * asISR, int N, bool fixpdf, bool fixas);

  // Function to integrate PDF ratios between two scales over x and t,
  // where the PDFs are always evaluated at the lower t-integration limit
  double monteCarloPDFratios(int flav, double x, double maxScale,
           double minScale, double pdfScale, double asME, Rndm* rndmPtr);

  // Default: Check if a ordered (and complete) path has been found in
  // the initial node, in which case we will no longer be interested in
  // any unordered paths.
  bool onlyOrderedPaths();

  // Check if an allowed (according to user-criterion) path has been found in
  // the initial node, in which case we will no longer be interested in
  // any forbidden paths.
  bool onlyAllowedPaths();

  // When a full path has been found, register it with the initial
  // history node.
  // IN  MyHistory : MyHistory to be registered as path
  //     bool    : Specifying if clusterings so far were ordered
  //     bool    : Specifying if path is complete down to 2->2 process
  // OUT true if MyHistory object forms a plausible path (eg prob>0 ...)
  bool registerPath(DireHistory & l, bool isOrdered,
         bool isAllowed, bool isComplete);

  // For one given state, find all possible clusterings.
  // IN  Event : state to be investigated
  // OUT vector of all (rad,rec,emt) systems in the state
  vector<DireClustering> getAllClusterings( const Event& event);
  vector<DireClustering> getClusterings( int emt, int rad, const Event& event);
  // Function to attach (spin-dependent duplicates of) a clustering.
  void attachClusterings (vector<DireClustering>& clus, int iEmt, int iRad,
    int iRec, int iPartner, double pT, string name, const Event& event);

  // Calculate and return the probability of a clustering.
  // IN  Clustering : rad,rec,emt - System for which the splitting
  //                  probability should be calcuated
  // OUT splitting probability
  pair <double,double> getProb(const DireClustering & SystemIn);

  // Set up the beams (fill the beam particles with the correct
  // current incoming particles) to allow calculation of splitting
  // probability.
  // For interleaved evolution, set assignments dividing PDFs into
  // sea and valence content. This assignment is, until a history path
  // is chosen and a first trial shower performed, not fully correct
  // (since content is chosen form too high x and too low scale). The
  // assignment used for reweighting will be corrected after trial
  // showering
  void setupBeams();

  // Calculate the PDF ratio used in the argument of the no-emission
  // probability.
  double pdfForSudakov();

  // Calculate the hard process matrix element to include in the selection
  // probability.
  double hardProcessME( const Event& event);
  double hardProcessCouplings( const Event& event, int order = 0,
    double renormMultFac = 1., AlphaStrong* alphaS = NULL,
    AlphaEM* alphaEM = NULL, bool fillCouplCounters = false,
    bool with2pi = true);

  // Perform the clustering of the current state and return the
  // clustered state.
  // IN Clustering : rad,rec,emt triple to be clustered to two partons
  // OUT clustered state
  Event cluster( DireClustering & inSystem);

  // Function to get the flavour of the radiator before the splitting
  // for clustering
  // IN  int   : Position of the radiator after the splitting, in the event
  //     int   : Position of the emitted after the splitting, in the event
  //     Event : Reference event
  // OUT int   : Flavour of the radiator before the splitting
  int getRadBeforeFlav(const int radAfter, const int emtAfter,
        const Event& event);

  // Function to get the spin of the radiator before the splitting
  // IN int  : Spin of the radiator after the splitting
  //    int  : Spin of the emitted after the splitting
  // OUT int : Spin of the radiator before the splitting
  int getRadBeforeSpin(const int radAfter, const int emtAfter,
        const int spinRadAfter, const int spinEmtAfter,
        const Event& event);

  // Function to get the colour of the radiator before the splitting
  // for clustering
  // IN  int   : Position of the radiator after the splitting, in the event
  //     int   : Position of the emitted after the splitting, in the event
  //     Event : Reference event
  // OUT int   : Colour of the radiator before the splitting
  int getRadBeforeCol(const int radAfter, const int emtAfter,
        const Event& event);

  // Function to get the anticolour of the radiator before the splitting
  // for clustering
  // IN  int   : Position of the radiator after the splitting, in the event
  //     int   : Position of the emitted after the splitting, in the event
  //     Event : Reference event
  // OUT int   : Anticolour of the radiator before the splitting
  int getRadBeforeAcol(const int radAfter, const int emtAfter,
        const Event& event);

  // Function to get the parton connected to in by a colour line
  // IN  int   : Position of parton for which partner should be found
  //     Event : Reference event
  // OUT int   : If a colour line connects the "in" parton with another
  //             parton, return the Position of the partner, else return 0
  int getColPartner(const int in, const Event& event);

  // Function to get the parton connected to in by an anticolour line
  // IN  int   : Position of parton for which partner should be found
  //     Event : Reference event
  // OUT int   : If an anticolour line connects the "in" parton with another
  //             parton, return the Position of the partner, else return 0
  int getAcolPartner(const int in, const Event& event);

  // Function to get the list of partons connected to the particle
  // formed by reclusterinf emt and rad by colour and anticolour lines
  // IN  int          : Position of radiator in the clustering
  // IN  int          : Position of emitted in the clustering
  //     Event        : Reference event
  // OUT vector<int>  : List of positions of all partons that are connected
  //                    to the parton that will be formed
  //                    by clustering emt and rad.
  vector<int> getReclusteredPartners(const int rad, const int emt,
    const Event& event);

  // Function to extract a chain of colour-connected partons in
  // the event
  // IN     int          : Type of parton from which to start extracting a
  //                       parton chain. If the starting point is a quark
  //                       i.e. flavType = 1, a chain of partons that are
  //                       consecutively connected by colour-lines will be
  //                       extracted. If the starting point is an antiquark
  //                       i.e. flavType =-1, a chain of partons that are
  //                       consecutively connected by anticolour-lines
  //                       will be extracted.
  // IN      int         : Position of the parton from which a
  //                       colour-connected chain should be derived
  // IN      Event       : Refernence event
  // IN/OUT  vector<int> : Partons that should be excluded from the search.
  // OUT     vector<int> : Positions of partons along the chain
  // OUT     bool        : Found singlet / did not find singlet
  bool getColSinglet(const int flavType, const int iParton,
    const Event& event, vector<int>& exclude,
    vector<int>& colSinglet);

  // Function to check that a set of partons forms a colour singlet
  // IN  Event       : Reference event
  // IN  vector<int> : Positions of the partons in the set
  // OUT bool        : Is a colour singlet / is not
  bool isColSinglet( const Event& event, vector<int> system);
  // Function to check that a set of partons forms a flavour singlet
  // IN  Event       : Reference event
  // IN  vector<int> : Positions of the partons in the set
  // IN  int         : Flavour of all the quarks in the set, if
  //                   all quarks in a set should have a fixed flavour
  // OUT bool        : Is a flavour singlet / is not
  bool isFlavSinglet( const Event& event,
    vector<int> system, int flav=0);

  // Function to properly colour-connect the radiator to the rest of
  // the event, as needed during clustering
  // IN  Particle& : Particle to be connected
  //     Particle  : Recoiler forming a dipole with Radiator
  //     Event     : event to which Radiator shall be appended
  // OUT true               : Radiator could be connected to the event
  //     false              : Radiator could not be connected to the
  //                          event or the resulting event was
  //                          non-valid
  bool connectRadiator( Particle& radiator, const int radType,
                        const Particle& recoiler, const int recType,
                        const Event& event );

  // Function to find a colour (anticolour) index in the input event
  // IN  int col       : Colour tag to be investigated
  //     int iExclude1 : Identifier of first particle to be excluded
  //                     from search
  //     int iExclude2 : Identifier of second particle to be excluded
  //                     from  search
  //     Event event   : event to be searched for colour tag
  //     int type      : Tag to define if col should be counted as
  //                      colour (type = 1) [->find anti-colour index
  //                                         contracted with col]
  //                      anticolour (type = 2) [->find colour index
  //                                         contracted with col]
  // OUT int           : Position of particle in event record
  //                     contraced with col [0 if col is free tag]
  int FindCol(int col, int iExclude1, int iExclude2,
              const Event& event, int type, bool isHardIn);

  // Function to in the input event find a particle with quantum
  // numbers matching those of the input particle
  // IN  Particle : Particle to be searched for
  //     Event    : Event to be searched in
  // OUT int      : > 0 : Position of matching particle in event
  //                < 0 : No match in event
  int FindParticle( const Particle& particle, const Event& event,
    bool checkStatus = true );

  // Function to check if rad,emt,rec triple is allowed for clustering
  // IN int rad,emt,rec,partner : Positions (in event record) of the three
  //                      particles considered for clustering, and the
  //                      correct colour-connected recoiler (=partner)
  //    Event event     : Reference event
  bool allowedClustering( int rad, int emt, int rec, int partner,
    string name, const Event& event );

  bool hasConnections(int arrSize, int nIncIDs[], int nOutIDs[]);
  bool canConnectFlavs( map<int,int> nIncIDs, map<int,int> nOutIDs);

  // Function to check if rad,emt,rec triple is results in
  // colour singlet radBefore+recBefore
  // IN int rad,emt,rec : Positions (in event record) of the three
  //                      particles considered for clustering
  //    Event event     : Reference event
  bool isSinglett( int rad, int emt, int rec, const Event& event );

  // Function to check if event is sensibly constructed: Meaning
  // that all colour indices are contracted and that the charge in
  // initial and final states matches
  // IN  event : event to be checked
  // OUT TRUE  : event is properly construced
  //     FALSE : event not valid
  bool validEvent( const Event& event );

  // Function to check whether two clusterings are identical, used
  // for finding the history path in the mother -> children direction
  bool equalClustering( DireClustering clus1 , DireClustering clus2 );

  // Chose dummy scale for event construction. By default, choose
  //     sHat     for 2->Boson(->2)+ n partons processes and
  //     M_Boson  for 2->Boson(->)             processes
  double choseHardScale( const Event& event ) const;

  // If the state has an incoming hadron return the flavour of the
  // parton entering the hard interaction. Otherwise return 0
  int getCurrentFlav(const int) const;

   // If the state has an incoming hadron return the x-value for the
   // parton entering the hard interaction. Otherwise return 0.
  double getCurrentX(const int) const;

  double getCurrentZ(const int rad, const int rec, const int emt,
    int idRadBef = 0) const;

  // Function to compute "pythia pT separation" from Particle input
  double pTLund(const Event& event, int radAfterBranch, int emtAfterBranch,
    int recAfterBranch, string name);

  // Function to return the position of the initial line before (or after)
  // a single (!) splitting.
  int posChangedIncoming(const Event& event, bool before);

  vector<int> getSplittingPos(const Event& event, int type);

  // Function to give back the ratio of PDFs and PDF * splitting kernels
  // needed to convert a splitting at scale pdfScale, chosen with running
  // PDFs, to a splitting chosen with PDFs at a fixed scale mu. As needed to
  // properly count emissions.
  double pdfFactor( const Event& process, const Event& event, const int type,
    double pdfScale, double mu );

  // Function giving the product of splitting kernels and PDFs so that the
  // resulting flavour is given by flav. This is used as a helper routine
  // to dgauss
  double integrand(int flav, double x, double scaleInt, double z);

  // Function providing a list of possible new flavours after a w emssion
  // from the input flavour.
  vector<int> posFlavCKM(int flav);

  // Check if the new flavour structure is possible.
  // If clusType is 1 final clustering is assumed, otherwise initial clustering
  // is assumed.
  bool checkFlavour(vector<int>& flavCounts, int flavRad, int flavRadBef,
    int clusType);

  // Check if an event reclustered into a 2 -> 2 dijet.
  // (Only enabled if W reclustering is used).
  bool isQCD2to2(const Event& event);

  // Check if an event reclustered into a 2 -> 1 Drell-Yan.
  // (Only enabled if W reclustering is used).
  bool isEW2to1(const Event& event);

  // Check if an event reclustered into massless 2 -> 2.
  bool isMassless2to2(const Event& event);
  // Check if an event reclustered into DIS 2 -> 2.
  bool isDIS2to2(const Event& event);

  // Function to allow effective gg -> EW boson couplings.
  bool mayHaveEffectiveVertex(string process, vector<int> in, vector<int> out);

  // Set selected child indices.
  void setSelectedChild();

  void setGoodChildren();
  void setGoodSisters();
  void setProbabilities();
  void printMECS();

  void tagPath(DireHistory* leaf);
  void multiplyMEsToPath(DireHistory* leaf);
  //int tag() { return tagSave; }
  bool hasTag(string key) {
    if(find(tagSave.begin(), tagSave.end(), key) != tagSave.end())
      return true;
    return false;
  }

  void setEffectiveScales();

  // Functions to retrieve scale information from external showers.
  double getShowerPluginScale(const Event& event, int rad, int emt, int rec,
    string name, string key, double scalePythia);

  pair<int,double> getCoupling(const Event& event, int rad, int emt,
    int rec, string name);

  void setCouplingOrderCount(DireHistory* leaf,
    map<string,int> count = map<string,int>());

public:

  // The state of the event correponding to this step in the
  // reconstruction.
  Event state;

  // Index for generation.
  int generation;

private:

  // The previous step from which this step has been clustered. If
  // null, this is the initial step with the n-jet state generated by
  // the matrix element.
  DireHistory * mother;

  // The different steps corresponding to possible clusterings of this
  // state.
  vector<DireHistory *> children;
  vector<DireHistory *> goodSisters;

  // After a path is selected, store the child index.
  int selectedChild;

  // The different paths which have been reconstructed indexed with
  // the (incremental) corresponding probability. This map is empty
  // unless this is the initial step (mother == 0).
  map<double,DireHistory *> paths;

  // The sum of the probabilities of the full paths. This is zero
  // unless this is the initial step (mother == 0).
  double sumpath;

public:

  // The different allowed paths after projection, indexed with
  // the (incremental) corresponding probability. This map is empty
  // unless this is the initial step (mother == 0).
  map<double,DireHistory *> goodBranches, badBranches;

private:

  // The sum of the probabilities of allowed paths after projection. This is
  // zero unless this is the initial step (mother == 0).
  double sumGoodBranches, sumBadBranches;

  // This is set true if an ordered (and complete) path has been found
  // and inserted in paths.
  bool foundOrderedPath;

  // This is set true if an allowed (according to a user criterion) path has
  // been found and inserted in paths.
  bool foundAllowedPath;

  // This is set true if a complete (with the required number of
  // clusterings) path has been found and inserted in paths.
  bool foundCompletePath;

  bool foundOrderedChildren;

  // The scale of this step, corresponding to clustering which
  // constructed the corresponding state (or the merging scale in case
  // mother == 0).
  double scale, scaleEffective, couplEffective;

  bool allowedOrderingPath;

  // Flag indicating if a clustering in the construction of all histories is
  // the next clustering demanded by inout clusterings in LesHouches 2.0
  // accord.
  bool nextInInput;

  // The probability associated with this step and the previous steps.
  double clusterProb, clusterCoupl, prodOfProbs, prodOfProbsFull;

  // The partons and scale of the last clustering.
  DireClustering clusterIn;
  int iReclusteredOld, iReclusteredNew;

  // Flag to include the path amongst allowed paths.
  bool doInclude;

  bool hasMEweight;
  double MECnum, MECden, MECcontrib;

  vector<int> goodChildren;

  // Pointer to MergingHooks object to get all the settings.
  MergingHooksPtr mergingHooksPtr;

   // The default constructor is private.
  DireHistory() {}

  // The copy-constructor is private.
  DireHistory(const DireHistory &) {}

  // The assignment operator is private.
  DireHistory & operator=(const DireHistory &) {
    return *this;
  }

  // BeamParticle to get access to PDFs
  BeamParticle beamA;
  BeamParticle beamB;
  // ParticleData needed to initialise the shower AND to get the
  // correct masses of partons needed in calculation of probability
  ParticleData* particleDataPtr;

  // Info object to have access to all information read from LHE file
  Info* infoPtr;

  // Class for calculation weak shower ME.
  SimpleWeakShowerMEs weakShowerMEs;

  // Pointer to showers, to simplify external clusterings.
  PartonLevel* showers;

  shared_ptr<DireTimes> fsr;
  shared_ptr<DireSpace> isr;

  // Pointer to standard model couplings.
  CoupSM* coupSMPtr;

  DireWeightContainer* psweights;

  int nStepsMax;
  bool doSingleLegSudakovs;

  vector<string> tagSave;

  double probMaxSave;
  double probMax() {
    if (mother) return mother->probMax();
    return probMaxSave;
  }
  void updateProbMax(double probIn, bool isComplete = false) {
    if (mother) mother->updateProbMax(probIn, isComplete);
    if (!isComplete && !foundCompletePath) return;
    if (abs(probIn) > probMaxSave) probMaxSave = probIn;
  }

  int depth, minDepthSave;
  int minDepth() {
    if ( mother ) return mother->minDepth();
    return minDepthSave;
  }
  void updateMinDepth(int depthIn) {
    if ( mother ) return mother->updateMinDepth(depthIn);
    minDepthSave = (minDepthSave>0) ? min(minDepthSave,depthIn) : depthIn;
  }

  map<string,int> couplingPowCount;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_DireHistory_H
