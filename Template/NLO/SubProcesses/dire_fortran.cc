#include <iostream>
#include "Pythia8/Pythia.h"
#include "Dire/Dire.h"
#include "fstream"

using namespace std;
using namespace Pythia8;

// let us add a new class that inherits fomr LHAupFortran
class MyLHAupFortran4dire : public LHAupFortran_aMCatNLO {
  public:

  MyLHAupFortran4dire(){
    initialised = false;
  }

  //the common blocks should be alredy filled at the fortran level
  //so simply return true
  bool fillHepRup(){
    initialised = true;
    return true;
  }
  bool fillHepEup(){
    return true;
  }
  
  bool is_initialised(){
    return initialised;
  }

  private:
  bool initialised;
};

class PrintFirstEmission4dire : public UserHooks {

public:

  PrintFirstEmission4dire(LHA3FromPythia8* lhawriterPtrIn)
    : lhawriterPtr(lhawriterPtrIn), inputEvent(50000) {
    doRemoveDecayProducts=true;
    inputEvent.init("(hard process-modified)", particleDataPtr);
    resonances.resize(0);
  }

  bool canVetoISREmission() { return true; }
  bool canVetoFSREmission() { return true; }

  bool doVetoISREmission(int, const Event& event, int iSys) {
    nISR++;
    if (nISR + nFSR > 1) return false;
    // Reattach resonance decay products and write event.
    Event outEvent = makeHardEvent(iSys, event, true);
    reattachResonanceDecays(outEvent);
    lhawriterPtr->setEventPtr(&outEvent);
    lhawriterPtr->setEvent();
    // Done.
    return false;
  }

  bool doVetoFSREmission(int, const Event& event, int iSys) {
    nISR++;
    if (nISR + nFSR > 1) return false;
    // Reattach resonance decay products and write event.
    Event outEvent = makeHardEvent(iSys, event, true);
    reattachResonanceDecays(outEvent);
    lhawriterPtr->setEventPtr(&outEvent);
    lhawriterPtr->setEvent();
    // Done.
    return false;
  }

  bool canVetoProcessLevel() { return true; }
  bool doVetoProcessLevel(Event& process) {
    // Initailize and store resonance decay products.
    nISR = nFSR = 0;
    lhawriterPtr->setProcessPtr(&process);
    bareEvent(process,true);
    return false;
  }

  LHA3FromPythia8* lhawriterPtr;

  Event inputEvent;
  vector< pair<int,int> > resonances;
  bool doRemoveDecayProducts;

  int nISR, nFSR;

  //--------------------------------------------------------------------------
  Event makeHardEvent( int iSys, const Event& state, bool isProcess) {

    bool hasSystems = !isProcess && partonSystemsPtr->sizeSys() > 0;
    int sizeSys     = (hasSystems) ? partonSystemsPtr->sizeSys() : 1;

    Event event = Event();
    event.clear();
    event.init( "(hard process-modified)", particleDataPtr );

    int in1 = 0;
    for ( int i = state.size()-1; i > 0; --i)
      if ( state[i].mother1() == 1 && state[i].mother2() == 0
        && (!hasSystems || partonSystemsPtr->getSystemOf(i,true) == iSys))
        {in1 = i; break;}
    if (in1 == 0) in1 = partonSystemsPtr->getInA(iSys);

    int in2 = 0;
    for ( int i = state.size()-1; i > 0; --i)
      if ( state[i].mother1() == 2 && state[i].mother2() == 0
        && (!hasSystems || partonSystemsPtr->getSystemOf(i,true) == iSys))
        {in2 = i; break;}
    if (in2 == 0) in2 = partonSystemsPtr->getInB(iSys);

    // Try to find incoming particle in other systems, i.e. if the current
    // system arose from a resonance decay.
    bool resonantIncoming = false;
    if ( in1 == 0 && in2 == 0 ) {
      int iParentInOther = 0;
      int nSys = partonSystemsPtr->sizeAll(iSys);
      for (int iInSys = 0; iInSys < nSys; ++iInSys){
        int iNow = partonSystemsPtr->getAll(iSys,iInSys);
        for (int iOtherSys = 0; iOtherSys < sizeSys; ++iOtherSys){
          if (iOtherSys == iSys) continue;
          int nOtherSys = partonSystemsPtr->sizeAll(iOtherSys);
          for (int iInOtherSys = 0; iInOtherSys < nOtherSys; ++iInOtherSys){
            int iOtherNow = partonSystemsPtr->getAll(iOtherSys,iInOtherSys);
            if (state[iNow].isAncestor(iOtherNow)) {
              iParentInOther = iOtherNow;
            }
          }
        }
      }
      in1 = iParentInOther;
      if (iParentInOther) resonantIncoming = true;
    } 

    event.append(state[0]);
    event.append(state[1]);
    event[1].daughters(3,0);
    event.append(state[2]);
    event[2].daughters(4,0);

    // Attach the first incoming particle.
    event.append(state[in1]);
    event[3].mothers(1,0);
    if (resonantIncoming) event[3].status(-22);
    else event[3].status(-21);

    // Attach the second incoming particle.
    event.append(state[in2]);
    event[4].mothers(2,0);
    event[4].status(-21);

    for ( int i = 0; i < state.size(); ++i) {
      // Careful when builing the sub-events: A particle that is currently
      // intermediate in one system could be the pirogenitor of another
      // system, i.e. when resonance decays are present. In this case, the
      // intermediate particle in the current system should be final. 
      bool isFin   = state[i].isFinal();
      bool isInSys = (partonSystemsPtr->getSystemOf(i) == iSys);

      bool isParentOfOther = false;
      if (!isFin && isInSys) {
        for (int iOtherSys = 0; iOtherSys < sizeSys; ++iOtherSys){
          if (iOtherSys == iSys) continue;
          double nSys = partonSystemsPtr->sizeAll(iOtherSys);
          for (int iInSys = 0; iInSys < nSys; ++iInSys){
            int iNow = partonSystemsPtr->getAll(iOtherSys,iInSys);
            if (state[iNow].isAncestor(i)) {isParentOfOther=true; break;}
          }
        }
      }

      if ( (isFin || isParentOfOther) && (!hasSystems || isInSys) ) {
      //if ( state[i].isFinal() 
      //  && (!hasSystems || partonSystemsPtr->getSystemOf(i) == iSys)) {
        int iN = event.append(state[i]);
        event[iN].daughters(0,0);
        event[iN].mothers(3,4);
        event[iN].status(23);
      }
    }

    // Set daughters of initial particles.
    event[3].daughters(5,event.size()-1);
    event[4].daughters(5,event.size()-1);
    return event;

  }


  //--------------------------------------------------------------------------
  // Return event stripped off decay products.
  Event bareEvent(const Event& inputEventIn,
    bool storeInputEvent ) {

    // Find and detach decay products.
    Event newProcess = Event();
    newProcess.init("(hard process-modified)", particleDataPtr);

    // If desired, store input event.
    if ( storeInputEvent ) {
      resonances.resize(0);
      inputEvent.clear();
      inputEvent = inputEventIn;
    }

    // Now remove decay products.
    if ( doRemoveDecayProducts ) {

      // Add the beam and initial partons to the event record.
      for (int i = 0; i < inputEventIn.size(); ++ i) {
        if ( inputEventIn[i].mother1() > 4
          || inputEventIn[i].statusAbs() == 22
          || inputEventIn[i].statusAbs() == 23)
          break;
        newProcess.append(inputEventIn[i]);
      }

      // Add the intermediate particles to the event record.
      for (int i = 0; i < inputEventIn.size(); ++ i) {
        if (inputEventIn[i].mother1() > 4) break;
        if ( inputEventIn[i].status() == -22) {
          int j = newProcess.append(inputEventIn[i]);
          newProcess[j].statusPos();
          if ( storeInputEvent ) resonances.push_back( make_pair(j, i) );
          newProcess[j].daughters(0, 0);
        }
      }

      // Add remaining outgoing particles to the event record.
      for (int i = 0; i < inputEventIn.size(); ++ i) {
        if (inputEventIn[i].mother1() > 4) break;
        if ( inputEventIn[i].statusAbs() != 11
          && inputEventIn[i].statusAbs() != 12
          && inputEventIn[i].statusAbs() != 21
          && inputEventIn[i].statusAbs() != 22)
          newProcess.append(inputEventIn[i]);
      }

      // Update event colour tag to maximum in whole process.
      int maxColTag = 0;
      for (int i = 0; i < inputEventIn.size(); ++ i) {
        if ( inputEventIn[i].col() > maxColTag )
          maxColTag = inputEventIn[i].col();
        if ( inputEventIn[i].acol() > maxColTag )
          maxColTag = inputEventIn[i].acol();
      }
      newProcess.initColTag(maxColTag);

      // Copy junctions from process to newProcess.
      for (int i = 0; i < inputEventIn.sizeJunction(); ++i)
        newProcess.appendJunction( inputEventIn.getJunction(i));

      newProcess.saveSize();
      newProcess.saveJunctionSize();

    } else {
      newProcess = inputEventIn;
    }

    // Remember scale
    newProcess.scale( inputEventIn.scale() );

    // Done
    return newProcess;

  }

  //--------------------------------------------------------------------------
  // Write event with decay products attached to argument. Only possible if an
  // input event with decay producs had been stored before.
  bool reattachResonanceDecays(Event& process ) {

    // Now reattach the decay products.
    if ( doRemoveDecayProducts && inputEvent.size() > 0 ) {

    int sizeBef = process.size();
    // Vector of resonances for which the decay products were already attached.
    vector<int> iAftChecked;
    // Reset daughters and status of intermediate particles.
    for ( int i = 0; i < int(resonances.size()); ++i ) {
      for (int j = 0; j < sizeBef; ++j ) {
        if ( j != resonances[i].first ) continue;

        int iOldDaughter1 = inputEvent[resonances[i].second].daughter1();
        int iOldDaughter2 = inputEvent[resonances[i].second].daughter2();

        // Get momenta in case of reclustering.
        int iHardMother      = resonances[i].second;
        Particle& hardMother = inputEvent[iHardMother];
        // Find current mother copy (after clustering).
        int iAftMother       = 0;
        for ( int k = 0; k < process.size(); ++k )
          if ( process[k].id() == inputEvent[resonances[i].second].id() ) {
            // Only attempt if decays of this resonance were not attached.
            bool checked = false;
            for ( int l = 0; l < int(iAftChecked.size()); ++l )
              if ( k == iAftChecked[l] )
                checked = true;
            if ( !checked ) {
              iAftChecked.push_back(k);
              iAftMother = k;
              break;
            }
          }

        Particle& aftMother  = process[iAftMother];

        // Resonance can have been moved by clustering,
        // so prepare to update colour and momentum information for system.
        int colBef  = hardMother.col();
        int acolBef = hardMother.acol();
        int colAft  = aftMother.col();
        int acolAft = aftMother.acol();
        RotBstMatrix M;
        M.bst( hardMother.p(), aftMother.p());

        // Attach resonance decay products.
        int iNewDaughter1 = 0;
        int iNewDaughter2 = 0;
        for ( int k = iOldDaughter1; k <= iOldDaughter2; ++k ) {
          if ( k == iOldDaughter1 )
            iNewDaughter1 = process.append(inputEvent[k] );
          else
            iNewDaughter2 = process.append(inputEvent[k] );
          process.back().statusPos();
          Particle& now = process.back();
          // Update colour and momentum information.
          if (now.col()  != 0 && now.col()  == colBef ) now.col(colAft);
          if (now.acol() != 0 && now.acol() == acolBef) now.acol(acolAft);
          now.rotbst( M);
          // Update vertex information.
          if (now.hasVertex()) now.vProd( aftMother.vDec() );
          // Update mothers.
          now.mothers(iAftMother,0);
        }

        process[iAftMother].daughters( iNewDaughter1, iNewDaughter2 );
        process[iAftMother].statusNeg();

        // Loop through event and attach remaining decays
        int iDec = 0;
        do {
          if ( process[iDec].isFinal() && process[iDec].canDecay()
            && process[iDec].mayDecay() && process[iDec].isResonance() ) {

            int iD1 = process[iDec].daughter1();
            int iD2 = process[iDec].daughter2();

            // Done if no daughters exist.
            if ( iD1 == 0 || iD2 == 0 ) continue;

            // Attach daughters.
            int iNewDaughter12 = 0;
            int iNewDaughter22 = 0;
            for ( int k = iD1; k <= iD2; ++k ) {
              if ( k == iD1 )
                iNewDaughter12 = process.append(inputEvent[k] );
              else
                iNewDaughter22 = process.append(inputEvent[k] );
              process.back().statusPos();
              Particle& now = process.back();
              // Update colour and momentum information.
              if (now.col() != 0 && now.col() == colBef ) now.col(colAft);
              if (now.acol()!= 0 && now.acol()== acolBef) now.acol(acolAft);
              now.rotbst( M);
              // Update vertex information.
              if (now.hasVertex()) now.vProd( process[iDec].vDec() );
              // Update mothers.
              now.mothers(iDec,0);
            }

            // Modify mother status and daughters.
            process[iDec].status(-22);
            process[iDec].daughters(iNewDaughter12, iNewDaughter22);

          // End of loop over all entries.
          }
        } while (++iDec < process.size());
      } // End loop over process entries.
    } // End loop over resonances.

    // Update event colour tag to maximum in whole process.
    int maxColTag = 0;
    for (int i = 0; i < process.size(); ++ i) {
      if (process[i].col() > maxColTag) maxColTag = process[i].col();
      if (process[i].acol() > maxColTag) maxColTag = process[i].acol();
    }
    process.initColTag(maxColTag);

    }

    // Done.
    return (doRemoveDecayProducts) ? inputEvent.size() > 0 : true;

  }

};

extern "C" { 

  // set up a global instance of pythia8 and dire
  Pythia pythia4dire;
  Dire dire;

  // set up a global instance of LHAup
  MyLHAupFortran4dire lhareader4dire;
  LHA3FromPythia8 lhawriter4dire(&pythia4dire.event, &pythia4dire.settings,
    &pythia4dire.info, &pythia4dire.particleData);
  PrintFirstEmission4dire printFirstEmission4dire(&lhawriter4dire); 

  // Allow Pythia to use Dire merging classes. 
  MyMerging* merging           = new MyMerging();
  MyHardProcess* hardProcess   = new MyHardProcess();
  MyMergingHooks* mergingHooks = new MyMergingHooks();

  // a counter for the number of event
  int iEvent4dire = 0;

  // an initialisation function
  void dire_init_(char input[500]) {
    string cmdFilePath(input);
    // Remove whitespaces
    while(cmdFilePath.find(" ", 0) != string::npos)
      cmdFilePath.erase(cmdFilePath.begin()+cmdFilePath.find(" ",0));
    if (cmdFilePath!="" && !(fopen(cmdFilePath.c_str(), "r"))) {
      cout<<"Pythia8 input file '"<<cmdFilePath<<"' not found."<<endl;
      abort();
    }
    lhareader4dire.setInit();
    // Example of a user hook for storing in the out stream the event after the first emission.
    pythia4dire.setUserHooksPtr(&printFirstEmission4dire);
    bool cmdFileEmpty = (cmdFilePath == "");
    if (!cmdFileEmpty) {
      cout<<"Initialising Pythia8 from cmd file '"<<cmdFilePath<<"'"<<endl;		
      pythia4dire.readFile(cmdFilePath.c_str());
    } else {
      cout<<"Using default initialization of Pythia8."<<endl;
      pythia4dire.readString("Beams:frameType=5");
      pythia4dire.readString("Check:epTolErr=1.0000000000e-02");
      cmdFilePath = "blub.cmnd";
      int syscall = system(("touch "+cmdFilePath).c_str());
      if (syscall == -1) cout << "Warning: Could not use system call in file"
        << __FILE__ << " at line " << __LINE__ << endl;
      syscall = system((" echo ShowerPDF:usePDFalphas    = off >> "+cmdFilePath).c_str());
      if (syscall == -1) cout << "Warning: Could not use system call in file"
        << __FILE__ << " at line " << __LINE__ << endl;
      syscall = system((" echo ShowerPDF:usePDFmasses    = off >> "+cmdFilePath).c_str());
      syscall = system((" echo DireSpace:ForceMassiveMap = on >> "+cmdFilePath).c_str());
      if (syscall == -1) cout << "Warning: Could not use system call in file"
        << __FILE__ << " at line " << __LINE__ << endl;
    }
    pythia4dire.setLHAupPtr(& lhareader4dire);
    dire.init(pythia4dire, cmdFilePath.c_str());
//    if (cmdFileEmpty) system("rm -f "+cmdFilePath.c_str());
    // Flag that Pythia8 intiialisation has been performed.
    pythia_control_.is_pythia_active = 1;
  }

  // an initialisation function
  void dire_init_default_() {

    lhareader4dire.setInit();
    // Example of a user hook for storing in the out stream the event after the first emission.
    pythia4dire.setUserHooksPtr(&printFirstEmission4dire);

    mergingHooks->setHardProcessPtr(hardProcess);
    pythia4dire.setMergingHooksPtr(mergingHooks);
    pythia4dire.setMergingPtr(merging);

    cout<<"Using default initialization of Pythia8."<<endl;
    pythia4dire.readString("Beams:frameType                 = 5");
    pythia4dire.readString("Check:epTolErr                  = 1.000000e-02");
    pythia4dire.readString("merging:doptlundmerging         = on");
    pythia4dire.readString("merging:process                 = pp>LEPTONS,NEUTRINOS");
    pythia4dire.readString("merging:tms                     = -1.0");
    pythia4dire.readString("merging:includeWeightInXSection = off");
    pythia4dire.readString("merging:njetmax                 = 1000");
    pythia4dire.readString("merging:applyveto               = off");
    pythia4dire.readString("Merging:useShowerPlugin         = on");
    pythia4dire.readString("PartonLevel:MPI                 = off");
    pythia4dire.readString("Print:quiet = on");
    pythia4dire.readString("Merging:nRequested = 0");

    pythia4dire.setLHAupPtr(&lhareader4dire);
    dire.initSettings(pythia4dire);

    //pythia4dire.readString("Dire:doMECs                     = on");
    //pythia4dire.readString("Dire:MG5card                    = param_card_sm.dat");
    pythia4dire.readString("Merging:useShowerPlugin         = on");
    pythia4dire.readString("Dire:doMerging                  = on");
    pythia4dire.readString("Dire:doExitAfterMerging         = on");
    pythia4dire.readString("Check:abortIfVeto               = on");
    pythia4dire.readString("Merging:mayRemoveDecayProducts  = on");
    pythia4dire.readString("Dire:doGenerateMergingWeights   = on");
    pythia4dire.readString("Dire:doGenerateSubtractions     = on");
    pythia4dire.readString("Dire:doMcAtNloDelta             = on");
    pythia4dire.readString("1:m0 = 0.0");
    pythia4dire.readString("2:m0 = 0.0");
    pythia4dire.readString("3:m0 = 0.0");
    pythia4dire.readString("4:m0 = 0.0");

//    double boost = 10000.;
    double boost = 1.5;
    pythia4dire.settings.parm("Enhance:fsr_qcd_1->1&21_CS",    boost);
    pythia4dire.settings.parm("Enhance:fsr_qcd_1->1&21_CS",    boost);
    pythia4dire.settings.parm("Enhance:fsr_qcd_1->21&1_CS",    boost);
    pythia4dire.settings.parm("Enhance:fsr_qcd_21->21&21a_CS", boost);
    pythia4dire.settings.parm("Enhance:fsr_qcd_21->21&21b_CS", boost);
    pythia4dire.settings.parm("Enhance:fsr_qcd_21->1&1a_CS",   boost);
    pythia4dire.settings.parm("Enhance:fsr_qcd_21->1&1b_CS",   boost);
    pythia4dire.settings.parm("Enhance:isr_qcd_1->1&21_CS",    boost);
    pythia4dire.settings.parm("Enhance:isr_qcd_21->1&1_CS",    boost);
    pythia4dire.settings.parm("Enhance:isr_qcd_21->21&21a_CS", boost);
    pythia4dire.settings.parm("Enhance:isr_qcd_21->21&21b_CS", boost);
    pythia4dire.settings.parm("Enhance:isr_qcd_1->21&1_CS",    boost);

    pythia4dire.readString("Enhance:fsr_qcd_1->2&1&2_CS = 1.0");
    pythia4dire.readString("Enhance:fsr_qcd_1->1&1&1_CS = 1.0");
    pythia4dire.readString("Enhance:isr_qcd_1->2&1&2_CS = 1.0");
    pythia4dire.readString("Enhance:isr_qcd_1->1&1&1_CS = 1.0");


    /*pythia4dire.readString("Enhance:fsr_qcd_1->1&21_CS = 10.0");
    pythia4dire.readString("Enhance:fsr_qcd_1->21&1_CS = 10.0");
    pythia4dire.readString("Enhance:fsr_qcd_21->21&21a_CS = 10.0");
    pythia4dire.readString("Enhance:fsr_qcd_21->21&21b_CS = 10.0");
    pythia4dire.readString("Enhance:fsr_qcd_21->1&1a_CS = 10.0");
    pythia4dire.readString("Enhance:fsr_qcd_21->1&1b_CS = 10.0");
    pythia4dire.readString("Enhance:fsr_qcd_1->2&1&2_CS = 1.0");
    pythia4dire.readString("Enhance:fsr_qcd_1->1&1&1_CS = 1.0");
    pythia4dire.readString("Enhance:isr_qcd_1->1&21_CS = 10.0");
    pythia4dire.readString("Enhance:isr_qcd_21->1&1_CS = 10.0");
    pythia4dire.readString("Enhance:isr_qcd_21->21&21a_CS = 10.0");
    pythia4dire.readString("Enhance:isr_qcd_21->21&21b_CS = 10.0");
    pythia4dire.readString("Enhance:isr_qcd_1->21&1_CS = 10.0");
    pythia4dire.readString("Enhance:isr_qcd_1->2&1&2_CS = 1.0");
    pythia4dire.readString("Enhance:isr_qcd_1->1&1&1_CS = 1.0");*/

    dire.init(pythia4dire,"", -999, &printFirstEmission4dire);

    // Transfer initialized shower weights pointer to merging class. 
    merging->setWeightsPtr(dire.weightsPtr);
    merging->setShowerPtrs(dire.timesPtr, dire.spacePtr);

    // Flag that Pythia8 intiialisation has been performed.
    pythia_control_.is_pythia_active = 1;
  }

  // a function to shower and analyse events
  void dire_setevent_() {
    if (!lhareader4dire.is_initialised()) {
      lhareader4dire.setInit();
      pythia4dire.init();
    }
    // This should set the LHA event using fortran common blocks
    lhareader4dire.setEvent();
  }

  // a function to shower and analyse events
  void dire_next_() {
    if (!lhareader4dire.is_initialised()) {
      lhareader4dire.setInit();
      pythia4dire.init();
    }
//    pythia4dire.settings.listAll();
    pythia4dire.next();
	
    ++iEvent4dire;
  }

  void dire_get_mergingweight_( double& w ) {
    w = pythia4dire.info.mergingWeightNLO();
  }

  // This should set the LHA event using fortran common blocks
  // a function to close everything
  void dire_stat_() {
    pythia4dire.stat();
  }

  void dire_get_sudakov_stopping_scales_( double scales [1000] ) {
//    vector<double> sca(mergingHooks->stoppingScales());
    vector<double> sca(merging->getStoppingScales());
    for (int i=0; i < sca.size(); ++i)
      scales[i] = sca[i];
    for (int i=sca.size(); i < 1000; ++i)
      scales[i] = -1.0;

  }

}

