#include <iostream>
#include "Pythia8/Pythia.h"
#include "Dire/Dire.h"
#include "fstream"

using namespace std;
using namespace Pythia8;

extern "C" { 

  // set up a global instance of pythia8 and dire
  Pythia pythia4dire;
  Dire dire;

  // Allow Pythia to use Dire merging classes. 
  MyMerging* merging           = new MyMerging();
  MyHardProcess* hardProcess   = new MyHardProcess();
  MyMergingHooks* mergingHooks = new MyMergingHooks();

  // a counter for the number of event
  int iEvent4dire = 0;

  // an initialisation function
  void dire_init_(double masses[21]) {

    mergingHooks->setHardProcessPtr(hardProcess);
    pythia4dire.setMergingHooksPtr(mergingHooks);
    pythia4dire.setMergingPtr(merging);

    for (int i=0; i < 21; ++i){
      stringstream s;

      if ( i+1 == 7 
        || i+1 == 8 
        || i+1 == 9 
        || i+1 == 10 
        || i+1 == 17 
        || i+1 == 18 
        || i+1 == 19 
        || i+1 == 20) continue; 

      s << i+1 << ":m0 =" << masses[i];
      pythia4dire.readString(s.str());
    }

    cout<<"Using default initialization of Pythia8."<<endl;
    pythia4dire.readString("HardQCD:all = on");
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
    pythia4dire.readString("Beams:setProductionScalesFromLHEF = off");

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
    pythia4dire.readString("Dire:doSingleLegSudakovs        = on");
    pythia4dire.readString("1:m0 = 0.0");
    pythia4dire.readString("2:m0 = 0.0");
    pythia4dire.readString("3:m0 = 0.0");
    pythia4dire.readString("4:m0 = 0.0");

    double boost = 10.;
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

    pythia4dire.readString("ShowerPDF:usePDF = off");

    dire.init(pythia4dire);

    // Transfer initialized shower weights pointer to merging class. 
    merging->setWeightsPtr(dire.weightsPtr);
    merging->setShowerPtrs(dire.timesPtr, dire.spacePtr);

    // Perform a single step to check initialization.
    pythia4dire.next();
  }

  void dire_get_sudakov_stopping_scales_( double scales [1000] ) {
    vector<double> sca(merging->getStoppingScales());
    for (int i=0; i < sca.size(); ++i)
      scales[i] = sca[i];
    for (int i=sca.size(); i < 1000; ++i)
      scales[i] = -1.0;

  }

  void dire_get_stopping_info_( double scales [100][100],
    double mass [100][100] ) {
    merging->getStoppingInfo(scales, mass);
  }

  void dire_get_no_emission_prob_( double& noemProb, double& startingScale,
    double& stoppingScale, double& mDipole, int& id, int& type ) {
    noemProb = merging->generateSingleSudakov ( startingScale,
      stoppingScale, pow(mDipole,2) , id, type, 7000., 0.1);
  }


}

