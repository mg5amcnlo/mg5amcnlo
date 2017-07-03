
#include "Pythia8/Pythia.h"
#include <iostream>
#include <string>

Pythia8::Pythia pythia;

bool initComplete=false;

// Init function
void init(int idBeamA, int idBeamB, double eCM, std::string process ) {
  //if (process.compare("") != 0 ) {
  if ( true ) {
    pythia.settings.flag(process, true);
    pythia.settings.mode("Beams:idA", idBeamA);
    pythia.settings.mode("Beams:idB", idBeamB);
    pythia.settings.parm("Beams:eCM", eCM);
    pythia.init();
  } else {
    std::ifstream ifs;
    ifs.open("wbj_lhef3.lhe");
    std::istream* is = (std::istream*) &ifs;
    std::ifstream ifs2;
    ifs2.open("wbj_lhef3.lhe");
    std::istream* isHead = (std::istream*) &ifs2;
    pythia.init(is,isHead);
  }
  initComplete=true;
  return;
}

void next() {

  if (!initComplete) return;

  pythia.next();
  pythia.event.list();

}

int main() {

  init(11,-11,100.,"WeakSingleBoson:ffbar2gmZ");

  next();
  next();
  next();
  next();

  return 0;

}
