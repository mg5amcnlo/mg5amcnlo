// test203.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Test of Vincia's EW shower, including Born polarisation with MG5 MEs,
// On and off the Z pole (ie with Z as stable intermediate particle or
// with Z as unstable particle), and for ee > WW, pp > WW, and pp > tt. 

// Authors: Peter Skands <peter.skands@monash.edu>

// Keywords: Vincia; electron-positron;

// Include Pythia8 header(s) and namespace.
#include "Pythia8/Pythia.h"
#include "Pythia8/Vincia.h"
using namespace Pythia8;

// Write a histogram.
void write(ofstream& ofs, const Hist& hist) {
  ofs << "\n# " << hist.getTitle() << endl; hist.table(ofs);}

// Main Program
int main() {

  //************************************************************************

  // Number of events and number of aborts to accept before stopping.
  int    nEvent      = 1000;
  int    nAbort      = 2;
  int    nVerboseBeg = -1;
  int    nVerboseEnd = -1;
  
  // Create test output.
  ofstream ofs("test203.dat");
    
  // Loop over CM energies, process, and collider.
  // iTest = 0 : ee > Z > leptons @ 20 GeV
  // iTest = 1 : ee > Z > leptons @ 91.2 GeV
  // iTest = 2 : ee > Z > leptons @ 1000 GeV
  // iTest = 3 : ee > WW > leptons @ 1000 GeV
  // iTest = 4 : pp > WW > leptons @ 10000 GeV
  // iTest = 5 : pp > tt > leptons @ 100000 GeV
  int iTestBeg = 0;
  int iTestEnd = 5;
  for (int iTest=iTestBeg; iTest<=iTestEnd; ++iTest) {

    //**********************************************************************
    // Define Pythia 8 generator
    
    Pythia pythia;
    
    //**********************************************************************

    // Shorthands
    Event& event = pythia.event;
    Settings& settings = pythia.settings;

    // Define settings common to all runs. 
    // We will print the event record ourselves (with helicities)
    pythia.readString("Next:numberShowEvent  = 0");
    // ee beams
    pythia.readString("Beams:idA  =  11");
    pythia.readString("Beams:idB  = -11");
    pythia.readString("Next:numberCount = 1000");

    // Select leptonic Z and W decays (including neutrinos since we are
    // testing the weak shower).
    pythia.readString("23:onMode  = off");
    pythia.readString("23:onIfAny = 11 12 13 14");
    pythia.readString("24:onMode  = off");
    pythia.readString("24:onIfAny = 11 12 13 14");
    // Force incoming leptons to have x=1.
    pythia.readString("PDF:lepton = off");
    // VINCIA settings
    pythia.readString("PartonShowers:model   = 2");
    pythia.readString("Vincia:helicityShower = on"); 
    pythia.readString("Vincia:ewMode         = 3");  
    pythia.readString("Vincia:verbose        = 0");

    // CM energy and process. 
    double eCM = 20.0;
    string process = "WeakSingleBoson:ffbar2gmZ = on";
    string mg5lib  = "Vincia:mePlugin = procs_ew_sm-ckm";
    if (iTest == 0) { }
    else if (iTest == 1) eCM = 91.2;
    else if (iTest == 2) eCM = 1000.0;
    else if (iTest == 3) {
      // WW at 1000 GeV.
      eCM = 1000.0;
      process = "WeakDoubleBoson:ffbar2WW = on";
    }
    else if (iTest == 4) {
      // pp > WW at 10 TeV.
      eCM = 10000.0;
      pythia.readString("Beams:idA  = 2212");
      pythia.readString("Beams:idB  = 2212");
      process = "WeakDoubleBoson:ffbar2WW = on";
      // Use artificially low alphaS to speed things up. 
      pythia.readString("Vincia:alphaSvalue = 0.01");
      pythia.readString("MultipartonInteractions:alphaSvalue = 0.01");
      // Don't include pi0 -> gamma gamma to keep things neat.
      pythia.readString("111:mayDecay = off");
    }
    else if (iTest == 5) {
      // qqbar > ttbar 100 TeV (no MPI, no hadronisation).
      eCM = 100000.0;
      pythia.readString("Beams:idA  = 2212");
      pythia.readString("Beams:idB  = 2212");
      process = "Top:qqbar2ttbar = on";
      pythia.readString("HadronLevel:all = off");
      pythia.readString("PartonLevel:MPI = off");
      mg5lib = "Vincia:mePlugin = procs_top_sm";
    }
    settings.parm("Beams:eCM", eCM);
    pythia.readString(process);
    pythia.readString(mg5lib);

    // Initialize
    if(!pythia.init()) { return EXIT_FAILURE; }
    
    // Define counters and PYTHIA histograms.
    double nGamSum   = 0.0;
    double nWeakSum  = 0.0;
    double nFinalSum = 0.0;
    Hist histNFinal("nFinal", 100, -0.5, 99.5);
    Hist histNGam("nPhotons", 20, -0.5, 19.5);
    Hist histNWeak("nWeakBosons", 10, -0.5, 9.5);
    
    //************************************************************************
    
    // EVENT GENERATION LOOP.
    // Generation, event-by-event printout, analysis, and histogramming.
  
    // Counter for negative-weight events
    double weight=1.0;
    double sumWeights = 0.0;
    
    // Begin event loop
    int iAbort = 0;
    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

      // Verbose output for a sub-range of events,
      // eg for debugging rare occurrences.      
      if (iEvent == nVerboseBeg) {
        shared_ptr<Vincia> vinciaPtr =
          dynamic_pointer_cast<Vincia> (pythia.getShowerModelPtr());
        vinciaPtr->setVerbose(3);
      }
      else if (iEvent == nVerboseEnd) {
        shared_ptr<Vincia> vinciaPtr =
          dynamic_pointer_cast<Vincia> (pythia.getShowerModelPtr());
        vinciaPtr->setVerbose(2);
      }        
      
      bool aborted = !pythia.next();
      if(aborted){
        event.list();
        if (++iAbort < nAbort){
          continue;
        }
        cout << " Event generation aborted prematurely, owing to error!\n";
        cout<< "Event number was : "<<iEvent<<endl;
        break;
      }
      
      // Check for weights
      weight = pythia.info.weight();
      sumWeights += weight;
      
      // Print event with helicities
      if (iEvent == 0) event.list(true);
      
      // Count FS final-state particles, weak bosons, and photons.
      double nFinal = 0;
      double nWeak  = 0;
      double nGam   = 0;
      for (int i=5;i<event.size();i++) {
        // Count up final-state charged hadrons
        if (event[i].isFinal()) {
          ++nFinal;
          // Final-state photons that are not from hadron decays
          if (event[i].id() == 22 && event[i].status() < 90) ++nGam;
        }
        // Weak bosons (not counting hard process)
        else if (event[i].idAbs() == 23 || event[i].idAbs() == 24) {
          // Find weak bosons that were radiator or emitter.
          if (event[i].status() != -51) continue;
          nWeak += 0.5;
        }
      }
      histNWeak.fill(nWeak,weight);
      histNFinal.fill(nFinal,weight);
      histNGam.fill(nGam,weight);
      nGamSum   += nGam * weight;      
      nWeakSum  += nWeak * weight;      
      nFinalSum += nFinal * weight;      
    
    }
    
    //**********************************************************************
    
    // POST-RUN FINALIZATION    
    // Normalization.
    double normFac = 1./sumWeights;
    
    // Print a few histograms.
    ofs << "---------------------------------------------------------------\n";
    ofs << "test " << iTest << "\n";
    ofs << "---------------------------------------------------------------\n";
    write(ofs, histNWeak);
    write(ofs, histNGam);
    write(ofs, histNFinal);
    
    // Print out end-of-run information.
    pythia.stat();
    ofs<<endl;
    ofs<<fixed;
    ofs<<" <nFinal>   = "<<num2str(nFinalSum * normFac)<<endl;
    ofs<< " <nPhotons> = "<<num2str(nGamSum * normFac)<<endl;
    ofs<< " <nZW>      = "<<num2str(nWeakSum * normFac)<<endl;
    ofs<<endl;

    cout<<" iTest "<<iTest<<" <nFinal>   = "
        <<num2str(nFinalSum * normFac)<<endl;
    cout<<" iTest "<<iTest<<" <nGamShow> = "
        <<num2str(nGamSum * normFac)<<endl;
    cout<<" iTest "<<iTest<<" <nZW>      = "
        <<num2str(nWeakSum * normFac)<<endl;
    cout<<endl;

  }
  
  // Done.
  return 0;
}
