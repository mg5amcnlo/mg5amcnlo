#include "Pythia8/Pythia.h"
#include "Pythia8/VinciaEW.h"

using namespace Pythia8;

int main() {
  // Number of data points
  int nData = 100000;

  Pythia pythia;
  // Enable a process so that pythia doesn't complain
  pythia.readString("HardQCD:qqbar2gg = on");
  pythia.init();

  Info* infoPtr = const_cast<Info*>(&pythia.info); 

  // Make a VinciaCommon object to load the EW shower with
  VinciaCommon vinCom;
  vinCom.initPtr(infoPtr);
  vinCom.init();

  // Load the shower
  VinciaEW ewsh;
  ewsh.initPtr(infoPtr, &vinCom);
  ewsh.load();
  ewsh.init();

  // Vincia treats some quarks as massless
  int nMassless = pythia.settings.mode("Vincia:nFlavZeroMass");

  vector<string> configFileNames = {"EWoverestimates/configsFF.dat", "EWoverestimates/configsFFres.dat"};
  vector<string> branchingsFileNames = {"EWoverestimates/branchingsFF.xml", "EWoverestimates/branchingsFFres.xml"};

  for (unsigned int iName=0; iName<configFileNames.size(); iName++) {
    // Open files
    ifstream infFinal(configFileNames[iName]);
    ofstream outfFinal(branchingsFileNames[iName]);

    int idI, idi, idj, polI;

    while(!infFinal.eof()) {
      // Phase space point output
      ofstream outfData("EWoverestimates/data.csv");
      // Add header
      outfData << "val,c0,c1,c2,c3" << endl;

      // Read data
      infFinal >> idI;
      infFinal >> idi;
      infFinal >> idj;
      infFinal >> polI;

      // Print the branching
      cout << "Branching: (" << idI << ", " << polI << ") -> " << idi << ", " << idj << endl;

      // Get particle masses from particleDataPtr
      double mI = abs(idI) <= nMassless ? 0 : infoPtr->particleDataPtr->m0(idI);
      double mi = abs(idi) <= nMassless ? 0 : infoPtr->particleDataPtr->m0(idi);
      double mj = abs(idj) <= nMassless ? 0 : infoPtr->particleDataPtr->m0(idj);

      double mI2 = pow2(mI);
      double mi2 = pow2(mi);
      double mj2 = pow2(mj);

      // Antenna invariant mass min and max
      double mIK2Min = max(1., pow2(mi + mj));
      double mIK2Max = 1E20;

      double QT2Min = max(1., mi2 + mj2 - mI2);

      // Now generate random branchings in randomly generated antennae
      int iData = 0;
      int nTriesAllowed = 1000;
      while(iData < nData) {
        if (iData!=0 && iData%(nData/10)==0) {cout << iData << endl;}

        // Start from a random antenna mass, giving preference to smaller values
        double mIK2 = mIK2Min*pow(mIK2Max/mIK2Min, infoPtr->rndmPtr->flat());

        // Next, we generate a branching in this antenna

        // Determine the overestimated phase space boundaries
        double zMin = 0.5*(1. - sqrt(1. - 4.*QT2Min/mIK2));
        double zMax = 0.5*(1. + sqrt(1. - 4.*QT2Min/mIK2));
        double QT2Max = mIK2;
        
        // Try to generate branching in antenna
        double QT2, Q2, z, sij, sjk, sik;
        int nTries = 0;
        while(true) {  
          // Sample a value of the antenna pT
          QT2 = QT2Min * pow(QT2Max/QT2Min, infoPtr->rndmPtr->flat());

          // z is (sij + mj2)/mIK2
          z = zMin + (zMax - zMin)*infoPtr->rndmPtr->flat();
          Q2 = QT2/z;
          sij = Q2 - mi2 - mj2 + mI2;
          sjk = z*mIK2 - mj2;
          sik = mIK2 - sij - sjk - mi2 - mj2;

          // Check if inside phase space
          double G = sij*sjk*sik - mi2*pow2(sjk) - mj2*pow2(sik);
          if (sij > 0 && sjk > 0 && sik > 0 && G > 0) {break;}

          nTries++;
          if (nTries > nTriesAllowed) {break;}
        }
        if (nTries > nTriesAllowed) {continue;}

        // We found a point inside phase space
        // Now compute the antenna function and the overestimate
        // Note that we multiply by a factor Q2 to ensure dimensionlessness
        double xi = (sij + sik + mi2)/mIK2;
        double xj = (sij + sjk + mj2)/mIK2;

        // Overestimate factors
        double c1Fac = 1;
        double c2Fac = 1./xi;
        double c3Fac = 1./xj;
        double c4Fac = mI2/Q2;

        // Antenna function
        double ant = 0;
        vector<AntWrapper> ants = ewsh.ampCalc.antFuncFF(Q2, 0, xi, xj, idI, idi, idj, mI, mi, mj, polI);
        for (int i=0; i<(int)ants.size(); i++) {
            ant += ants[i].val*Q2;
        }

        // Write the results to file
        outfData << ant << "," << c1Fac << "," << c2Fac << "," << c3Fac << "," << c4Fac << endl;
      
        iData++;
      }
      outfData.close();

      // Run python script calling PuLP to find overestimates
      system("EWoverestimates/runLinearProgramming.sh");

      // Read result
      ifstream infConstants("EWoverestimates/consts.dat");

      if (infConstants) {
        // Read constants from file
        double c0, c1, c2, c3;
        infConstants >> c0;
        infConstants >> c1;
        infConstants >> c2;
        infConstants >> c3;

        // Filter out small constants
        double cMax = c0;
        if (c1 > cMax) {cMax = c1;}
        if (c2 > cMax) {cMax = c2;}
        if (c3 > cMax) {cMax = c3;}

        // We drop constants lower than 1% of the highest
        if (c0/cMax < 1E-2) {c0 = 0;}
        if (c1/cMax < 1E-2) {c1 = 0;}
        if (c2/cMax < 1E-2) {c2 = 0;}
        if (c3/cMax < 1E-2) {c3 = 0;}

        // Write to XML
        outfFinal << "<EWbranchingFinal idI=\"" << idI << "\" idi=\"" << idi << "\" idj=\"" << idj << "\" polI=\"" << polI 
        << "\" c0=\"" << c0 << "\" c1=\"" << c1 << "\" c2=\"" << c2 << "\" c3=\"" << c3 << "\" >" << endl;
        outfFinal << "</EWbranchingFinal>" << endl;
      }
    }
  }
}