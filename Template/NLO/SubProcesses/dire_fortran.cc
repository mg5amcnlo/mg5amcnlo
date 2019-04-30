#include <iostream>
#include "Pythia8/Pythia.h"
#include "Dire/Dire.h"
#include "fstream"
#include <ctime>

using namespace std;
using namespace Pythia8;

extern "C" { 

  // set up a global instance of pythia8 and dire
  Pythia pythia4dire;
  Dire dire;

  // set up a global instance of LHAup
  //MyLHAupFortran4dire lhareader4dire(&pythia4dire.settings);
  MyLHAupFortran lhareader4dire(&pythia4dire.settings);
  LHA3FromPythia8 lhawriter4dire(&pythia4dire.event, &pythia4dire.settings,
    &pythia4dire.info, &pythia4dire.particleData);
  PrintFirstEmission printFirstEmission4dire(&lhawriter4dire); 

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
    // Flag that Pythia8 intiialisation has been performed.
    pythia_control_.is_pythia_active = 1;
  }

  // an initialisation function
  void dire_init_default_(int& idIn1, int& idIn2, int outIDs [10], double masses[26] ) {
    lhareader4dire.setInit();
    // Example of a user hook for storing in the out stream the event after the first emission.
    pythia4dire.setUserHooksPtr(&printFirstEmission4dire);

    mergingHooks->setHardProcessPtr(hardProcess);
    pythia4dire.setMergingHooksPtr(mergingHooks);
    pythia4dire.setMergingPtr(merging);

    // Reconstruct the process string.
    string processString = "";
    // Set incoming particles.
    if (idIn1 == 2212) processString += "p";
    if (idIn1 == 11)   processString += "e-";
    if (idIn1 ==-11)   processString += "e+";
    if (idIn2 == 2212) processString += "p";
    if (idIn2 == 11)   processString += "e-";
    if (idIn2 ==-11)   processString += "e+";
    processString += ">";
    // Set outgoing particles.
    bool foundOutgoing = false;
    for (int i=0; i < 10; ++i) {
      if (outIDs[i]==0) continue;
      if (outIDs[i]==2212) {
        processString += "j";
      } else {
        ostringstream proc;
        proc << "{" << pythia4dire.particleData.name(outIDs[i]) << "," << outIDs[i] << "}";
        processString += proc.str();
      }
    } 

    // Initialize masses.
    for (int i=1; i <= 25; ++i){
      if (masses[i]<0.) continue;
      stringstream s;
      // Need a non-zero muon mass to get correct Higgs width. Otherwise gets a NAN. Need to be fixed later.
      if (i==13) continue;
      s << i << ":m0 =" << masses[i];
      pythia4dire.readString(s.str());
    }

    cout<<"Using default initialization of Pythia8."<<endl;
    pythia4dire.readString("Beams:frameType                 = 5");
    pythia4dire.readString("Check:epTolErr                  = 1.000000e-02");
    pythia4dire.readString("merging:doptlundmerging         = on");
    pythia4dire.settings.word("Merging:Process", processString);
    pythia4dire.readString("merging:tms                     = -1.0");
    pythia4dire.readString("merging:includeWeightInXSection = off");
    pythia4dire.readString("merging:njetmax                 = 1000");
    pythia4dire.readString("merging:applyveto               = off");
    pythia4dire.readString("Merging:useShowerPlugin         = on");
    pythia4dire.readString("PartonLevel:MPI                 = off");
    pythia4dire.readString("Print:quiet = on");
    pythia4dire.readString("Merging:nRequested = 0");
    pythia4dire.readString("Beams:setProductionScalesFromLHEF = off");

    pythia4dire.setLHAupPtr(&lhareader4dire);
    merging->setLHAPtr(&lhawriter4dire);
    dire.initSettings(pythia4dire);

    pythia4dire.readString("Merging:useShowerPlugin         = on");
    pythia4dire.readString("Dire:doMerging                  = on");
    pythia4dire.readString("Dire:doExitAfterMerging         = on");
    pythia4dire.readString("Check:abortIfVeto               = on");
    pythia4dire.readString("Merging:mayRemoveDecayProducts  = on");
    pythia4dire.readString("Dire:doMcAtNloDelta             = on");
    pythia4dire.readString("Dire:doAuxMergingInfo           = off");

    // Disallow Pythia to overwrite parts of Les Houches input.
    pythia4dire.readString("LesHouches:setQuarkMass = 0");
    pythia4dire.readString("LesHouches:setLeptonMass = 0");
    pythia4dire.readString("LesHouches:mRecalculate = -1.0");
    pythia4dire.readString("LesHouches:matchInOut = off");

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
    pythia4dire.next();
    ++iEvent4dire;
  }

  void dire_get_stopping_info_( double scales [100][100],
    double mass [100][100] ) {
    merging->getStoppingInfo(scales, mass);
  }

  void dire_get_dead_zones_( bool dzone [100][100] ) {
    merging->getDeadzones( dzone);
  }

  void dire_clear_() { merging->clear(); }

}

