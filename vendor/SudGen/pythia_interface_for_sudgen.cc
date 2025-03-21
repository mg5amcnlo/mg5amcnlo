#include <iostream>
#include "Pythia8/Pythia.h"
#include "fstream"

using namespace std;
using namespace Pythia8;

extern "C" { 

  // set up a global instance of pythia8 and dire
  Pythia pythia;

  // an initialisation function
  void pythia_init_(double masses[21]) {

    for (int i=0; i < 21; ++i){
      stringstream s;

      if ( i+1 == 6
        || i+1 == 7 
        || i+1 == 8 
        || i+1 == 9 
        || i+1 == 10 
        || i+1 == 17 
        || i+1 == 18 
        || i+1 == 19 
        || i+1 == 20) continue; 

      s << i+1 << ":m0 =" << masses[i];
      pythia.readString(s.str());
    }

    cout<<"Using default initialization of Pythia8."<<endl;
    pythia.readString("HardQCD:all = on");
    pythia.readString("Check:epTolErr                  = 1.000000e-02");
    pythia.readString("merging:doptlundmerging         = on");
    pythia.readString("merging:process                 = pp>jj");
    pythia.readString("merging:tms                     = -1.0");
    pythia.readString("merging:includeWeightInXSection = off");
    pythia.readString("merging:njetmax                 = 1000");
    pythia.readString("merging:applyveto               = off");
    pythia.readString("Merging:useShowerPlugin         = on");
    pythia.readString("PartonLevel:MPI                 = off");
    pythia.readString("Print:quiet = on");
    pythia.readString("Merging:nRequested = 0");
    pythia.readString("Beams:setProductionScalesFromLHEF = off");
    pythia.readString("TimeShower:QEDshowerByQ = off");
    pythia.readString("TimeShower:QEDshowerByL = off");
    pythia.readString("TimeShower:QEDshowerByOther = off");

    pythia.settings.addParm("Dire:Sudakov:Min",0.0,false,false,0.0,1e0);

    //pythia.readString("Dire:doMECs                     = on");
    //pythia.readString("Dire:MG5card                    = param_card_sm.dat");
    pythia.readString("Check:abortIfVeto               = on");
    pythia.readString("Merging:mayRemoveDecayProducts  = on");
    pythia.readString("1:m0 = 0.0");
    pythia.readString("2:m0 = 0.0");
    pythia.readString("3:m0 = 0.0");
    pythia.readString("4:m0 = 0.0");
    pythia.settings.forceParm("Spaceshower:pt0ref", 0.0);
    pythia.readString("TimeShower:MEcorrections = off");
    pythia.readString("TimeShower:PhiPolAsym = off");
    pythia.readString("TimeShower:PhiPolAsymHard = off");
    pythia.readString("SpaceShower:MEcorrections = off");
    pythia.readString("SpaceShower:PhiPolAsym = off");
    pythia.readString("SpaceShower:PhiPolAsymHard = off");
    pythia.readString("SpaceShower:PhiIntAsym = off");
    pythia.readString("SpaceShower:RapidityOrder = off");

    // Change number of incoming quarks in the PDFs used in ISR.
    //pythia.readString("SpaceShower:nQuarkin = 3");
    pythia.readString("SpaceShower:nQuarkin = 5");
    pythia.readString("SpaceShower:alphaSvalue = 0.130");
    //pythia.readString("SpaceShower:alphaSvalue = 0.139386");
    pythia.readString("SpaceShower:pdfMode = 2");

    pythia.init();
    // Perform a single step to check initialization.
    pythia.next();
  }

  void pythia_get_no_emission_prob_( double& noemProb, double& startingScale,
    double& stoppingScale, double& mDipole, int& id, int& type, int& seed,
    double& min_sudakov) {
    // Set random seed.`
    pythia.readString("Random:setSeed = on");
    pythia.settings.mode("Random:seed", seed);
    pythia.rndm.init(seed);
    // Set cut-off for Sudakov.
    pythia.settings.parm("Dire:Sudakov:Min", min_sudakov);
    noemProb = pythia.mergingPtr->generateSingleSudakov ( startingScale,
      stoppingScale, pow(mDipole,2) , id, type, pow2(7000.), 0.1);
  }


  void pythia_get_no_emission_prob_x_( double& noemProb, double& startingScale,
    double& stoppingScale, double& mDipole, int& id, int& type, int& seed,
    double& min_sudakov, double& x) {
    // Set random seed.`
    pythia.readString("Random:setSeed = on");
    pythia.settings.mode("Random:seed", seed);
    pythia.rndm.init(seed);
    // Set cut-off for Sudakov.
    pythia.settings.parm("Dire:Sudakov:Min", min_sudakov);
    noemProb = pythia.mergingPtr->generateSingleSudakov ( startingScale,
      stoppingScale, pow(mDipole,2) , id, type, pow2(7000.), x);
  }


}

