#include "Pythia8/Pythia.h"
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <sstream>
#include "param_card.h"
#include "run_card.h"
#include "proc_card.h"

bool initialization_done = false;

Pythia8::Pythia pythia;
// Stringstream of the events in which MG5aMC will write and PY8 read.
std::stringstream EvtSS;

void generate_header_SS(std::stringstream & headerSS,
						const int &      Pythia8BeamA,
						const int &      Pythia8BeamB,
						const double &   PythiaBeamEnergyA,
						const double &   PythiaBeamEnergyB) {
	headerSS<<"<LesHouchesEvents version=\"3.0\">\n";
	headerSS<<"<header>\n";
	// Writing proc_card.dat in the banner
	headerSS<<"<MG5ProcCard>\n";
	headerSS<<ProcCard<<"\n";	
	headerSS<<"</MG5ProcCard>\n";
	// Writing run_card.dat in the banner	
	headerSS<<"<MGRunCard>\n";
	headerSS<<RunCard<<"\n";	
	headerSS<<"</MGRunCard>\n";
	// Writing param_card.dat in the banner	
	headerSS<<"<slha>\n";
	headerSS<<ParamCard<<"\n";	
	headerSS<<"</slha>\n";
	headerSS<<"</header>\n";
	// Writing the init block (only beam related information is relevant here)	
	headerSS<<"<init>\n";
	headerSS<<Pythia8BeamA<<" "<<Pythia8BeamB<<" ";
	headerSS<<std::scientific<<PythiaBeamEnergyA<<" "<<std::scientific<<PythiaBeamEnergyB<<" ";
	// We fill in the rest of the init information with blanks
	headerSS<<"0 0 -1 -1 -3 1\n";
	headerSS<<"-1.0 0.0 1.0 1\n";
	headerSS<<"<generator name='MadGraph5_aMC@NLO'>please cite 1405.0301</generator>\n";
	headerSS<<"</init>\n";
//	headerSS<<"</LesHouchesEvents>\n";
	return; 
}

extern "C" {
//Fortran interface
  void py8_bias_weight_(const double &   eCM,
						const int &      Pythia8BeamA,
						const double &   PythiaBeamEnergyA,
						const int &      Pythia8BeamB,
						const double &   PythiaBeamEnergyB,
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
//     /!\ WARNING this example is tailored for p p > z > e+ e- j for now.

		pythia.readString("Merging:doKtMerging=on");
		pythia.readString("Merging:Process=pp>e+e-");
		pythia.readString("Merging:TMS=30.0");
		pythia.readString("Merging:nJetMax=2");
		pythia.readString("Merging:applyVeto=off");
		pythia.readString("Beams:readLHEFheaders=on");
		pythia.readString("Merging:includeWeightInXsection=off");

		pythia.readString("hadronlevel:all=off");
		pythia.readString("partonlevel:all=off");
		pythia.readString("SpaceShower:QEDshowerByL=off");
		pythia.readString("SpaceShower:QEDshowerByQ=off");
		pythia.readString("ProcessLevel:resonanceDecays=off");
		pythia.readString("BeamRemnants:primordialKT=off");
		pythia.readString("TimeShower:QEDshowerByQ=off");
		pythia.readString("TimeShower:QEDshowerByL=off");
		pythia.readString("partonlevel:mpi=off");
		pythia.readString("PartonLevel:FSRinResonances=off");
		pythia.readString("PartonLevel:Remnants=off");
		pythia.readString("Check:event=off");
		
		pythia.settings.mode("Beams:frameType",4);		
		std::stringstream headerSS;
		generate_header_SS(headerSS, Pythia8BeamA,Pythia8BeamB,PythiaBeamEnergyA,PythiaBeamEnergyB);
		EvtSS.str(headerSS.str());
		pythia.init(&EvtSS, &headerSS);
		initialization_done = true;
	}

	std::string EvtStr(Pythia8EvtRecord);
	EvtStr = EvtStr.substr(0,EvtStr.find("</event>")+8);
	// Restart the buffer, clearing EOF state
	EvtSS.clear();
	// Set the stringstream buffer to contain the event only
	EvtSS.str(EvtStr);

//	std::cout<<"---> Event transmitted to Pythiaa8 <---\n"<<EvtSS.str()<<std::endl;

	// Compute the merging weight. All other shower processsing has been disabled in the initialisation above.
	pythia.next();	
//	pythia.event.list();
    // This is the CKKW Sudakov merging weight from the trial shower.
	std::cout<<pythia.info.mergingWeight()<<std::endl;
//	exit(0);

	OutputBiasWeight = pythia.info.mergingWeight();
  }
}
