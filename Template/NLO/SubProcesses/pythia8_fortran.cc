#include <iostream>
#include "Pythia8/Pythia.h"
#include "LHAFortran_aMCatNLO.h"
#include "fstream"

using namespace std;
using namespace Pythia8;

// let us add a new class that inherits fomr LHAupFortran
class MyLHAupFortran : public LHAupFortran_aMCatNLO {
  public:

  MyLHAupFortran(){
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

class PrintFirstEmission : public UserHooks {

public:

  PrintFirstEmission(LHA3FromPythia8* lhawriterPtrIn)
    : lhawriterPtr(lhawriterPtrIn) {}

  bool canVetoISREmission() { return true; }
  bool canVetoFSREmission() { return true; }

  bool doVetoISREmission(int, const Event& event, int iSys) {
    int nfinal(0);
	for (int i=0; i < event.size(); ++i)
	  if (event[i].isFinal() ) nfinal++;
	if (nfinal != 2 ) return false;
	lhawriterPtr->setEventPtr(&event);
	lhawriterPtr->setEvent();
	return false;
  }

  bool doVetoFSREmission(int, const Event& event, int iSys) {
    return false;
  }

  bool canVetoProcessLevel() { return true; }
  bool doVetoProcessLevel(Event& process) {
    lhawriterPtr->setProcessPtr(&process);
	return false;
  }

  LHA3FromPythia8* lhawriterPtr;

};


extern "C" { 

  // set up a global instance of pytia8
  Pythia pythia;
  // set up a global instance of LHAup
  MyLHAupFortran lhareader;
  LHA3FromPythia8 lhawriter(&pythia.event, &pythia.settings, &pythia.info,
    &pythia.particleData);

  PrintFirstEmission printFirstEmission(&lhawriter); 

  //pythia.setUserHooksPtr(printFirstEmission);
  //Pythia pythia2;
  //pythia2.stat();

  // a counter for the number of event
  int iEvent = 0;

  // an initialisation function
  void pythia_init_(char input[500]) {
	string cmdFilePath(input);
    // Remove whitespaces
    while(cmdFilePath.find(" ", 0) != string::npos)
      cmdFilePath.erase(cmdFilePath.begin()+cmdFilePath.find(" ",0));
    if (cmdFilePath!="" && !(fopen(cmdFilePath.c_str(), "r"))) {
		cout<<"Pythia8 input file '"<<cmdFilePath<<"' not found."<<endl;
		abort();
    }
    lhareader.setInit();
	// Example of a user hook for storing in the out stream the event after the first emission.
    pythia.setUserHooksPtr(&printFirstEmission);
	if (cmdFilePath!="") {
       cout<<"Initialising Pythia8 from cmd file '"<<cmdFilePath<<"'"<<endl;		
       pythia.readFile(cmdFilePath.c_str());
	} else {
	   cout<<"Using default initialization of Pythia8."<<endl;
	   pythia.readString("Beams:frameType=5");
	   pythia.readString("Check:epTolErr=1.0000000000e-02");
	}
    pythia.setLHAupPtr(& lhareader);
    pythia.init();
	// Flag that Pythia8 intiialisation has been performed.
	pythia_control_.is_pythia_active = 1;
  }

  // a function to shower and analyse events
  void pythia_setevent_() {
    if (!lhareader.is_initialised()) {
      lhareader.setInit();
      pythia.init();
    }
    //This should set the LHA event using fortran common blocks
    lhareader.setEvent();
  }

  // a function to shower and analyse events
  void pythia_next_() {
    if (!lhareader.is_initialised()) {
      lhareader.setInit();
      pythia.init();
    }
//    pythia.settings.listAll();
    pythia.next();
	
    ++iEvent;
  }

    //This should set the LHA event using fortran common blocks
  //a function to close everything
  void pythia_stat_() {
    pythia.stat();
  }

}

