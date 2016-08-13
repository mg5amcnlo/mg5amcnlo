#include "Pythia8/Pythia.h"
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <sstream>

// Small values used to cut off momenta
const double TINY      = 1e-15;
const double TINYMASS  = 1e-8;

bool initialization_done = false;

Pythia8::Pythia pythia;

extern "C" {
//Fortran interface
  void py8_bias_weight_(const double &   eCM,
						const int &      Pythia8BeamA,
						const int &      Pythia8BeamB,
		                const char *     Pythia8EvtRecord, 						
		                const double *   p, 
						const int &      nParticles,
						const double &   MurScale,
						const double &   AlphaQCD,
						const double &   AlphaQED,
						const int *      Pythia8ID,
						const int *      Pythia8MotherOne,
						const int *      Pythia8MotherTwo,
						const int *      Pythia8ColorOne,
						const int *      Pythia8ColorTwo,
						const int *      Pythia8Status,
						const int *      Pythia8Helicities,
						double &         OutputBiasWeight    ) 
  {
	if (!initialization_done) {
		pythia.settings.word("Beams:LHEF","/Users/valentin/Documents/Work/MG5/bias/PROC_sm_19/SubProcesses/P1_lvl_qq/G1/dummy.lhe");
		pythia.settings.mode("Beams:frameType",4);
		pythia.init();
		initialization_done = true;
	}

// Example of some Pythia calls
/*
	Pythia8::History history(0,0.,Pythia8::Event(),Pythia8::Clustering(), NULL, Pythia8::BeamParticle(),Pythia8::BeamParticle(),NULL,NULL,NULL,NULL,false,false,false,false,0.,NULL);
	pythia.next();
	pythia.event.list();
*/

	std::string EvtStr(Pythia8EvtRecord);
	EvtStr = EvtStr.substr(0,EvtStr.find("</event>")+8);
 	std::stringstream EvtSS(EvtStr);
	std::cout<<"---> Event transmitted to Pythiaa8 <---\n"<<EvtSS.str()<<std::endl;

// Example of usage of the already-parsed information
/*
    for (int i=0; i<nParticles*5; i=i+5) {
      // Store information in a State.
      double pup0 = (abs(p[i+0]) < TINY) ? 0.: p[i+0];
      double pup1 = (abs(p[i+1]) < TINY) ? 0.: p[i+1];
      double pup2 = (abs(p[i+2]) < TINY) ? 0.: p[i+2];
      double pup3 = (abs(p[i+3]) < TINY) ? 0.: p[i+3];
      double pup4 = (abs(p[i+4]) < TINYMASS) ? 0.: p[i+4];
      // color information not known
      int col1 = Pythia8ColorOne[i/5];
      int col2 = Pythia8ColorTwo[i/5];

//    Do something to currEvent 

	}

//    delete currEvent;
*/

	OutputBiasWeight = 1.0;
  }
}
