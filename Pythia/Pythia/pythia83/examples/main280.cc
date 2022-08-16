// main280.cc is a part of PYTHIA 8.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Author: Christian T Preuss <christian.preuss@monash.edu>

// Keywords: merging; leading order; CKKW-L; hepmc; vincia;

// This is a test program to run Pythia merging and  write HepMC2 files,
// written for Vincia sector merging, see the Sector Merging page in
// the online manual.
// It also redirects Pythia output to a log file, so that
// the terminal output is significantly reduced.
// An example command is
//     ./main280 main280.cmnd vbf_h_ output.hepmc
// where main280.cmnd supplies the commands and vbf_h_ specifies the event
// file path that will be auto-completed to the respective jet multiplicity
// (e.g. vbf_h_1.lhe for the 1-jet sample). The last argument specifies the
// output filename in the HepMC2 format.

// Standard packages.
#include <chrono>
#include <ctime>
#include <fstream>

// PYTHIA
#include "Pythia8/Pythia.h"
#include "Pythia8/Vincia.h"

// HepMC
#ifndef HEPMC2
#include "Pythia8Plugins/HepMC3.h"
#else
#include "Pythia8Plugins/HepMC2.h"
#endif

// Conversion factor mb --> pb.
#define MB_TO_PB 1.e9

using namespace Pythia8;
typedef std::chrono::system_clock Clock;
typedef std::chrono::high_resolution_clock HighResClock;

// Little helper function for execution time as nicely formatted string.
string getTimeAsString(double seconds) {
  stringstream ss;
  if (seconds >= 3600.*24.) {
    int days = seconds / (3600*24);
    ss << days << "d ";
    seconds -= days*(3600.*24.);
  }
  if (seconds >= 3600.) {
    int hours = seconds / 3600;
    ss << hours << "h ";
    seconds -= hours*3600.;
  }
  if (seconds >= 60.) {
    int mins = seconds / 60;
    ss << mins << "m ";
    seconds -= mins*60.;
  }
  ss << int(seconds) << "s";
  return ss.str();
}

// Main Program
int main(int argc, char* argv[]) {

  // Check number of command-line arguments.
  if (argc != 4) {
    cerr << " Unexpected number of command-line arguments. \n"
         << " Please specify:\n"
         << "   1) command file\n"
         << "   2) event file path\n"
         << "   3) output file\n"
         << " \n E.g. run \n"
         << "   ./main280 main280.cmnd w+_production_lhc_ output.hepmc\n"
         << " \n Program stopped! " << endl;
    return 1;
  }

  // Check whether input file exists.
  string cmndFile = argv[1];
  ifstream is(cmndFile);
  if (!is) {
    cerr << " File " << cmndFile << " was not found. \n"
         << " Program stopped! " << endl;
    return 1;
  }

  string evtPath = argv[2];
  string hepMCFile = argv[3];

  //------------------------------------------------------------------------
  // PYTHIA
  //------------------------------------------------------------------------

  Pythia pythia;

  // Add mode to generate only sample for specific jet number.
  pythia.settings.addMode("Merging:nJetReq", -1, true, false, -1, 100);

  // Add flag to decide whether to check all event files exist.
  pythia.settings.addFlag("Merging:checkFilesExist", false);

  // Read user settings from file.
  pythia.readFile(cmndFile);

  //------------------------------------------------------------------------

  // Settings.
  int  nEvent          = pythia.settings.mode("Main:numberOfEvents");
  int  nCount          = pythia.settings.mode("Next:numberCount");
  int  nErr            = pythia.settings.mode("Main:timesAllowErrors");
  int  nMerge          = pythia.settings.mode("Merging:nJetMax");
  int  nReq            = pythia.settings.mode("Merging:nJetReq");
  bool vinciaOn        = (pythia.settings.mode("PartonShowers:model") == 2);
  bool checkFilesExist = pythia.settings.flag("Merging:checkFilesExist");
  bool fsrOn           = pythia.settings.flag("PartonLevel:FSR");
  bool isrOn           = pythia.settings.flag("PartonLevel:ISR");
  bool mpiOn           = pythia.settings.flag("PartonLevel:MPI");
  bool hadLvlOn        = pythia.settings.flag("HadronLevel:all");

  //------------------------------------------------------------------------
  // HepMC
  //------------------------------------------------------------------------

  Pythia8::Pythia8ToHepMC toHepMC(hepMCFile);
  toHepMC.set_print_inconsistency(false);
  toHepMC.set_store_pdf(false);
  toHepMC.set_store_proc(false);
  toHepMC.set_store_xsec(false);

  //------------------------------------------------------------------------

  // Quickly check whether all event files exist.
  if (checkFilesExist) {
    bool allFilesExist = true;
    vector<string> missingFiles;
    for (int i(0); i<=nMerge; ++i) {
      string evtFile = evtPath+std::to_string(i)+".lhe";
      ifstream isNow(evtFile);
      if (!isNow) {
        allFilesExist = false;
        missingFiles.push_back(evtFile);
      }
    }
    if (!allFilesExist) {
      cerr << " Error: the following event files are missing:" << endl;
      for (auto& f : missingFiles) cerr << "  " << f << endl;
      cerr << " Please check the event file path you specified." << endl;
      return EXIT_FAILURE;
    }
  }

  //------------------------------------------------------------------------

  // Redirect cout to log file.

  auto now = Clock::now();
  std::time_t now_c = Clock::to_time_t(now);
  struct tm *parts = std::localtime(&now_c);
  string year  = std::to_string(1900 + parts->tm_year);
  string month = std::to_string(1 + parts->tm_mon);
  string day   = std::to_string(parts->tm_mday);
  string hour  = std::to_string(parts->tm_hour);
  string min   = std::to_string(parts->tm_min);
  string sec   = std::to_string(parts->tm_sec);
  string fname = "main280-"+year+"_"+month+"_"+day+"-"
    +hour+"_"+min+"_"+sec+".log";

  std::ofstream outstream(fname);
  std::streambuf* filebuf = outstream.rdbuf();
  std::streambuf* coutbuf = cout.rdbuf();
  cout.rdbuf(filebuf);

  //------------------------------------------------------------------------

  // Cross section estimation run.
  pythia.settings.flag("Merging:doXSectionEstimate", true);

  // Estimates.
  map<int, double> xSecEst;
  map<int, int> nSelected;
  map<int, int> nAccepted;
  map<int, int> lhaStrategy;

  // Switch hadron-level and shower off.
  pythia.settings.flag("PartonLevel:FSR", false);
  pythia.settings.flag("PartonLevel:ISR", false);
  pythia.settings.flag("HadronLevel:all", false);
  pythia.settings.flag("PartonLevel:MPI", false);
  pythia.settings.mode("Next:numberCount", nEvent);

  // Loop over all samples to merge.
  std::clog << "\n *--------  Merging Info  -----------------------------*\n"
            << " |                                                     |\n"
            << " | Starting cross section estimation for up to "
            << setw(1) << nMerge << " jets  |\n"
            << " |                                                     |\n"
            << " *-----------------------------------------------------*\n\n";
  int nDotCntr = nEvent / 40;
  if (nDotCntr == 0) nDotCntr = 1;
  for (int iMerge(nMerge); iMerge >= 0; --iMerge) {

    if (nReq >= 0 && iMerge != nReq) continue;

    // Read input for this subrun and initialise.
    string lheFileNow = evtPath+std::to_string(iMerge)+".lhe";
    pythia.settings.mode("Beams:frameType", 4);
    pythia.settings.word("Beams:LHEF", lheFileNow);
    if(!pythia.init()) {
      cerr << " Pythia failed initialisation in xSec estimation run "
           << iMerge << "." << endl;
      cerr << "\n Check the log file " << fname << " for details" << endl;
      return EXIT_FAILURE;
    }

    // Event loop.
    std::clog << " Estimating cross section for "
         << iMerge << "-jet sample";
    for(int iEvt(0); iEvt<nEvent; ++iEvt){
      if (iEvt % nDotCntr == 0) {
        std::clog << ".";
      }
      // Generate next event.
      if (!pythia.next()) {
        if (pythia.info.atEndOfFile()) break;
        else continue;
      }
    }
    std::clog << endl;

    pythia.stat();

    // Save estimates.
    xSecEst[iMerge] = pythia.info.sigmaGen();
    nSelected[iMerge] = pythia.info.nSelected();
    nAccepted[iMerge] = pythia.info.nAccepted();
    lhaStrategy[iMerge] =  pythia.info.lhaStrategy();
  }

  // Restore settings.
  pythia.settings.flag("Merging:doXSectionEstimate", false);
  pythia.settings.flag("PartonLevel:FSR", fsrOn);
  pythia.settings.flag("PartonLevel:ISR", isrOn);
  pythia.settings.flag("HadronLevel:all", hadLvlOn);
  pythia.settings.flag("PartonLevel:MPI", mpiOn);
  pythia.settings.mode("Next:numberCount", nCount);

  //------------------------------------------------------------------------

  // Merged total cross section.
  double  sigmaTot = 0.;
  double  errorTot = 0.;
  map<int, double> sigmaSample;
  map<int, double> errorSample;

  // Loop over all samples to merge.
  for (int iMerge(nMerge); iMerge >= 0; --iMerge) {

    if (nReq >= 0 && iMerge != nReq) continue;

    // Initialise cross sections.
    sigmaSample[iMerge] = 0.;
    errorSample[iMerge] = 0.;

    // Get normalisation for HepMC output.
    double normhepmc = (abs(lhaStrategy[iMerge]) == 4) ?
      1./(MB_TO_PB * nSelected[iMerge])
      : xSecEst[iMerge]/nAccepted[iMerge];

    // Read input file for this subrun and initialise.
    string lheFileNow = evtPath+std::to_string(iMerge)+".lhe";
    pythia.settings.mode("Beams:frameType", 4);
    pythia.settings.word("Beams:LHEF", lheFileNow);
    if(!pythia.init()) {
      cerr << " Pythia failed initialisation in run " << iMerge << ".\n"
           << "\n Check the log file " << fname << " for details.\n";
      return EXIT_FAILURE;
    }

    // Debugging.
    ShowerModelPtr showerPtr = pythia.getShowerModelPtr();
    shared_ptr<Vincia> vinciaPtr = dynamic_pointer_cast<Vincia>(showerPtr);
    if(vinciaOn && vinciaPtr==nullptr){
      cerr << "Couldn't fetch Vincia pointer in run "
           << nMerge-iMerge+1 << ".\n";
      return EXIT_FAILURE;
    }

    // Event loop.
    int iErr = 0;
    bool breakMerging = false;
    std::clog << "\n *--------  Merging Info  -----------------------------*\n"
              << " |                                                     |\n"
              << " | Starting event loop for ";
    if (iMerge > 0) std::clog << setw(1) << iMerge << "-jet sample";
    else std::clog << "Born sample ";
    std::clog << "                |\n"
              << " |                                                     |\n"
              << " *-----------------------------------------------------*\n";
    auto evtGenStart = HighResClock::now();
    for(int iEvt(0); iEvt<nEvent; ++iEvt){
      if (iEvt != 0 && iEvt % nCount == 0) {
        auto stopNow = HighResClock::now();
        double timeElapsed
          = std::chrono::duration_cast<std::chrono::seconds>
          (stopNow - evtGenStart).count();
        double timeLeft = timeElapsed * double(nEvent-iEvt) / double(iEvt);
        std::clog << endl << " " << iEvt
                  << " events generated  ("
                  << getTimeAsString(timeElapsed) << " elapsed / "
                  << getTimeAsString(timeLeft) << " left)" << endl;
      }

      // Generate next event.
      // Break out of event loop if at end of LHE file
      // or if too many errors appeared.
      if (!pythia.next()) {
        ++iErr;
        if (pythia.info.atEndOfFile()) break;
        if (iErr >= nErr) break;
        else continue;
      }

      // Get CKKW-L weight of current event.
      double evtweight = pythia.info.weight();
      double weight    = pythia.info.mergingWeight();
      evtweight *= weight;

      // Skip zero-weight events.
      if (evtweight == 0.) continue;

      // Add event weight to total cross section.
      sigmaTot += evtweight*normhepmc;
      errorTot += pow2(evtweight*normhepmc);
      sigmaSample[iMerge] += evtweight*normhepmc;
      errorSample[iMerge] += pow2(evtweight*normhepmc);

      // Write HepMC event and set cross section.
      toHepMC.writeNextEvent(pythia);
    }

    // Statistics (printed to terminal).
    cout.rdbuf(coutbuf);
    pythia.stat();
    cout.rdbuf(filebuf);

    if (breakMerging) break;
  }
  errorTot = sqrt(errorTot);

  // Restore cout.
  cout.rdbuf(coutbuf);

  // Print cross section information.
  cout << "\n *--------  XSec Summary  -----------------------------*\n"
       << " |                                                     |\n"
       << " | Exclusive cross sections:                           |\n";
  for (auto it=sigmaSample.begin(); it!=sigmaSample.end(); ++it) {
    if (it->first==0) cout << " |      Born:  ";
    else cout << " |     " << it->first << "-jet:  ";
    cout << setw(8) << scientific
         << setprecision(6) << it->second << " +- "
         << setw(8) << errorSample.at(it->first) << " mb         |\n";
  }
  cout << " |                                                     |\n"
       << " |- - - - - - - - - - - - - - - - - - - - - - - - - - -|\n"
       << " |                                                     |\n"
       << " | Inclusive cross sections:                           |\n";
  for (auto it=xSecEst.begin(); it!=xSecEst.end(); ++it) {
    if (it->first==0) cout << " |      Born:  ";
    else cout << " |     " << it->first << "-jet:  ";
    cout << setw(8) << scientific
         << setprecision(6) << it->second
         << " mb                         |\n";
  }
  cout << " |                                                     |\n"
       << " | CKKW-L merged inclusive cross section:              |\n"
       << " |             " << setw(8) << scientific << setprecision(6)
       << sigmaTot << " +- " << setw(8) << errorTot << " mb         |\n"
       << " |                                                     |\n"
       << " *-----------------------------------------------------*\n\n"
       << "\n Detailed PYTHIA output has been written to the log file "
       << fname << "\n";

  // Done.
  return 0;
}
