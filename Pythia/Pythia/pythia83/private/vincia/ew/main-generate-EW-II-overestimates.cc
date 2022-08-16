#include "Pythia8/Pythia.h"
#include "Pythia8/VinciaEW.h"

using namespace Pythia8;

int main() {
  // For II, there is no need to do NLP

  Pythia pythia;
  // Enable a process so that pythia doesn't complain
  pythia.readString("HardQCD:qqbar2gg = on");
  pythia.init();

  Info* infoPtr = const_cast<Info*>(&pythia.info); 

  // Make a VinciaCommon object
  VinciaCommon vinCom;
  vinCom.initPtr(infoPtr);
  vinCom.init();

  // Load the shower
  VinciaEW ewsh;
  ewsh.initPtr(infoPtr, &vinCom);
  ewsh.load();
  ewsh.init();

  // Load configurations
  ifstream infInitial("EWoverestimates/configsII.dat");
  ofstream outfInitial("EWoverestimates/branchingsII.xml");

  int idI, idi, idj, polI;

  while(!infInitial.eof()) {
      // Read data
      infInitial >> idI;
      infInitial >> idi;
      infInitial >> idj;
      infInitial >> polI;

      // Compute couplings
      double v = ewsh.ampCalc.vMap.at(make_pair(abs(idI), abs(idj)));
      double a = ewsh.ampCalc.aMap.at(make_pair(abs(idI), abs(idj)));
      double vMin = v - polI*a;
      double vPls = v + polI*a;

      // f scales with vMin, fbar scales with vPls
      double c0 = idI > 0 ? 4*pow2(vMin) : 4*pow2(vPls);
      double c1 = 0;
      double c2 = 0;
      double c3 = 0;

      // Write to XML
      if (c0 > 1E-2) {
          outfInitial << "<EWbranchingInitial idI=\"" << idI << "\" idi=\"" << idi << "\" idj=\"" << idj << "\" polI=\"" << polI 
          << "\" c0=\"" << c0 << "\" c1=\"" << c1 << "\" c2=\"" << c2 << "\" c3=\"" << c3 << "\" >" << endl;
          outfInitial << "</EWbranchingInitial>" << endl;
      }
  }


}