// main46.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Author: Christian T Preuss <christian.preuss@monash.edu>

// Keywords: HDF5 file; lheh5; hepmc;

// This program (main46.cc) illustrates how a HDF5 event file can be
// used by Pythia8. See main44.cc and main45.cc for how to use LHE
// files instead. Example usage is:
//     ./main46 main46.cmnd ttbar.hdf5 main46.hepmc

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/LHAHDF5.h"
#ifndef HEPMC2
#include "Pythia8Plugins/HepMC3.h"
#else
#include "Pythia8Plugins/HepMC2.h"
#endif

using namespace Pythia8;

//==========================================================================

// Example main programm to illustrate simple HDF5 usage.

int main(int argc, char* argv[]) {

  // Input sanity check
  if (argc < 4) {
    cout << "ERROR: Not enough arguments provided" << endl << endl
         << "Usage:\n\t" << argv[0]
         << "  COMMAND.cmnd INPUT.hdf5 OUTPUT.hepmc [OFFSET]"
         << endl << endl;
    return EXIT_FAILURE;
  }

  // Check whether input file exists.
  string cmndFile = argv[1];
  ifstream isCmnd(cmndFile);
  if (!isCmnd) {
    cerr << " File " << cmndFile << " was not found. \n"
         << " Program stopped! " << endl;
    return EXIT_FAILURE;
  }

  // Check whether event file exists.
  string hdf5File = argv[2];
  ifstream isH5(hdf5File);
  if (!isH5) {
    cerr << " File " << hdf5File << " was not found. \n"
         << " Program stopped! " << endl;
    return EXIT_FAILURE;
  }

  // HepMC file.
  string hepMCFile = argv[3];

  // Optionally: skip events.
  size_t eventOffset = (argc > 4) ? atoi(argv[4]) : 0;

  // PYTHIA.
  Pythia pythia;

  // Settings.
  pythia.readFile(cmndFile);
  pythia.settings.mode("Beams:frameType", 5);

  // Shorthands.
  int nEvents  = pythia.settings.mode("Main:numberOfEvents");
  int nAbort   = pythia.mode("Main:timesAllowErrors");

  // HDF5.
  HighFive::File file(hdf5File, HighFive::File::ReadOnly);

  // Create an LHAup object that can access relevant information in pythia.
  size_t readSize    = size_t(nEvents);
  string version     = pythia.settings.word("LHAHDF5:version");
  shared_ptr<LHAupH5> lhaUpPtr =
    make_shared<LHAupH5>(&file, eventOffset, readSize, version);

  // HepMC.
  Pythia8::Pythia8ToHepMC toHepMC(hepMCFile);
  toHepMC.set_print_inconsistency(false);

  // Hand Pythia the external reader.
  pythia.setLHAupPtr(lhaUpPtr);

  // Initialise.
  if (!pythia.init()) {
    cout << " Failed to initialise Pythia. Program stopped." << endl;
    return EXIT_FAILURE;
  }

  // Abort for too many errors.
  int  iAbort  = 0;
  bool doAbort = false;

  // Cross section and error.
  cout << "Start generating events.\n";
  double sigmaSample(0.), errorSample(0.);

  // Get the inclusive x-section by summing over all process x-sections.
  double xs = 0.;
  for (int i=0; i < pythia.info.nProcessesLHEF(); ++i)
    xs += pythia.info.sigmaLHEF(i);

  // Loop over events.
  while(pythia.info.nSelected() < nEvents) {
    // Generate next event.
    if( !pythia.next() ) {
      ++iAbort;
      if ( pythia.info.atEndOfFile() ) break;
      else if (iAbort > nAbort) {
        cout <<  " Aborting event generation after "
             << iAbort << " failed events." << endl;
        break;
      } else continue;
    }

    // Get event weight(s).
    double evtweight = pythia.info.weight();

    // Do not print zero-weight events.
    if ( evtweight == 0. ) continue;

    // Fill HepMC event.
    toHepMC.writeNextEvent(pythia);
    sigmaSample += evtweight;
    errorSample += pow2(evtweight);

  }

  // print cross section, errors
  pythia.stat();

  // Finalise cross section.
  double norm = 1./double(1.e9*lhaUpPtr->getTrials());
  if (abs(pythia.info.lhaStrategy()) == 3) norm *= xs;
  sigmaSample *= norm;
  errorSample = sqrt(errorSample)*norm;

  cout << " sigma = (" << scientific << setprecision(8)
       << sigmaSample << "  +-  " << errorSample << ") mb\n";

  // Done
  return 0;

}
