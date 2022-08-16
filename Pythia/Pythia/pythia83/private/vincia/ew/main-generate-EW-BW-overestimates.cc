#include "Pythia8/Pythia.h"
#include "Pythia8/VinciaEW.h"
using namespace Pythia8;

int main() {
    // ----------------------------------------------------------------
    // These are the id and pol to determine the overestimate for
    int id = 24;
    int pol = -1;
    // ----------------------------------------------------------------

    Pythia pythia;
    pythia.readString("HardQCD:qqbar2gg = on");
    pythia.init();

    Info* infoPtr = const_cast<Info*>(&pythia.info);    

    // Make a VinciaCommon object
    VinciaCommon vinCom;
    vinCom.initPtr(infoPtr);
    vinCom.init();

    VinciaEW ewsh;
    ewsh.initPtr(infoPtr, &vinCom);
    ewsh.load();
    ewsh.init();


    double m0 = infoPtr->particleDataPtr->m0(id);
    double m02 = pow2(m0);
    double width0 = ewsh.ampCalc.getTotalWidth(id, m0, pol); 
    cout << "Total width " << width0 << endl;

    // Consts
    double c0Best = 5;
    double c1Best = 5;
    double c2Best = 0.015;
    double c3Best = 2;

    // ---------------- Breit-Wigner optimizer ---------------------
    int nEvents = 100000;
    int nTries = 10000;

    // Minima and maxima of coefficients to scan
    double c0Min = 1, c0Max = 3;
    double c1Min = 1, c1Max = 3;
    double c2Min = 0.001, c2Max = 0.05;
    double c3Min = 1.01, c3Max = 3;

    cout << id << " " << m0 << " " << width0 << endl;

    // Mass test values - the new proposal is tested here to see if the overestimate ever exceeds the BW
    vector<double> mTest = {m0, m0+width0, m0-width0, width0, 10*m0, 100*m0, 1000*m0};

    // Test metrics
    double sumBest = 0;

    for (int iTries = 0; iTries < nTries; iTries++) {
        if (iTries != 0 && iTries%(nTries/10) == 0) {
            cout << iTries << endl;
        }
        vector<double> w;
        bool failed = false;

        // Sample some c0-c3
        double c0 = c0Min + (c0Max - c0Min)*infoPtr->rndmPtr->flat();
        double c1 = c1Min + (c1Max - c1Min)*infoPtr->rndmPtr->flat();
        double c2 = c2Min + (c2Max - c2Min)*infoPtr->rndmPtr->flat();
        double c3 = c3Min + (c3Max - c3Min)*infoPtr->rndmPtr->flat();

        // First test these coefficients against the test masses
        for (int iTest = 0; iTest<(int)mTest.size(); iTest++) {
            double m2Test = pow2(mTest[iTest]);

            // This is the general form of the overestimate
            double BWover = c0*width0*m0/( pow2(m2Test - m02) + pow2(c1)*m02*pow2(width0) );
            BWover += m2Test/m02 > c3 ? c2*m0/pow(m2Test - m02, 3./2.) : 0;

            // Compute the real Breit-Wigner
            double BWreal = ewsh.ampCalc.getBreitWigner(id, mTest[iTest], pol);

            if (BWover < BWreal) {
                failed = true;
                break;
            }
        }

        // Check if we failed
        if (failed) {continue;}

        // Once the initial test is passed, sample a bunch of points
        // from the overestimate and compute the accept probs
        // Eventually, the configuration with the highest average accept probability is kept.
        for (int i=0; i<nEvents; i++) {
            double nBW = c0/c1 * (M_PI/2. + atan(m0/c1/width0));
            double np  = 2.*c2/sqrt(c3 - 1);

            // Select one of the probs.
            double m2;
            if (infoPtr->rndmPtr->flat() < nBW/(nBW + np)) {
                m2 = m02 + c1*m0*width0*tan( c1/c0 * infoPtr->rndmPtr->flat() * nBW - atan(m0/c1/width0));
            }

            else {
                m2 = m02*( pow2(2*c2*sqrt(c3 - 1) / (2*c2 - infoPtr->rndmPtr->flat()*np*sqrt(c3 - 1))) + 1);
            }

            // Evaluate probabilities
            double BWover = c0*width0*m0/( pow2(m2 - m02) + pow2(c1)*m02*pow2(width0) );
            BWover += m2/m02 > c3 ? c2*m0/pow(m2 - m02, 3./2.) : 0;
    
            double BWreal = ewsh.ampCalc.getBreitWigner(id, sqrt(m2), pol);

            // If overestimation failed, drop these coefficients
            if (BWover < BWreal) {
                failed = true;
                break;
            }

            // Otherwise store the accept prob
            w.push_back(BWreal/BWover);

        }

        // Again check if we failed
        if (failed) {continue;}

        // We found a valid overestimate
        // Now see if it is the best one

        // Compute the sum of the accept probabilities as a metric
        double sum = 0, sqSum = 0;
        for (int i=0; i<nEvents; i++) {
            sum += w[i];
            sqSum += pow2(w[i]);
        }

        if (sum > sumBest) {
            c0Best = c0;
            c1Best = c1;
            c2Best = c2;
            c3Best = c3;
            sumBest = sum;
        }
    }

    // -------------------------------------------------------------

    // Perform some final checks to make absolutely sure the overestimate is valid
    int nChecks = 1000000;
    double mMaxChecks = 10000;
    for (int i=0; i<nChecks; i++) {
        double m = ((double)i)/((double)nChecks)*mMaxChecks;
        double m2 = pow2(m);
        double BWover = c0Best*width0*m0/( pow2(m2 - m02) + pow2(c1Best)*m02*pow2(width0) );
        if (abs(id) == 6) {
            BWover += m2/m02 > c3Best ? c2Best/pow(m2 - m02, 1./2.)/m0 : 0;
        }
        else {
            BWover += m2/m02 > c3Best ? c2Best*m0/pow(m2 - m02, 3./2.) : 0;
        }
        double BWreal = ewsh.ampCalc.getBreitWigner(id, m, pol);
        if (BWover < BWreal) {
            cout << "Overestimate incorrect " << m << " " << BWreal/BWover << endl;
            break;
        }
    }
    
    cout << "{" << c0Best << ", " << c1Best << ", " << c2Best << ", " << c3Best << "}" << endl;
    cout << " Average sum = " << sumBest/(double)nEvents << endl;
}