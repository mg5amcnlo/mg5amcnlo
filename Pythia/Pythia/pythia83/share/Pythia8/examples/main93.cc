// main93.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Christian Bierlich <christian.bierlich@thep.lu.se>.

// Keywords: analysis; hepmc; command file; command line option; root; rivet;
// tuning;

// Streamlined event generation with possibility to output ROOT files,
// output HepMC files and run RIVET analyses, all by specifying output modes
// in a cmnd file, where also the event generator settings are specified.
// The example is run with command line options, run ./main93 -h to see a
// full list. See ROOT Usage for information about ROOT output, RIVET Usage
// for information about RIVET and HepMC Interface for information about HepMC.

#include "Pythia8/Pythia.h"
#include "Pythia8/HeavyIons.h"
#include "Pythia8Plugins/Pythia8Rivet.h"
#include "main93.h"
#include <chrono>
#ifdef PY8ROOT
#include "TTree.h"
#include "TFile.h"
#endif

using namespace Pythia8;

// Helper class to parse command line options.
class InputParser {

public:

  InputParser (int &argc, char **argv) {
    for (int i = 1; i < argc; ++i) arglist.push_back(string(argv[i]));
  }

  const string& getOption(const string &opt) const {
    vector<string>::const_iterator itr = find(arglist.begin(),
      arglist.end(), opt);
    if (itr != arglist.end() && ++itr != arglist.end()) return *itr;
    return "";
  }

  bool hasOption(const string &opt) const {
    return find(arglist.begin(), arglist.end(), opt) != arglist.end();
  }

private:
  vector<string> arglist;
};


int main(int argc, char* argv[]) {

  // Parser object for command line input.
  InputParser ip(argc, argv);

  // Print help text and exit.
  if(ip.hasOption("-h") || ip.hasOption("--help")) {
    cout << "Usage: Run Pythia with cmnd file input, and get Rivet, HepMC or\n"
      "standard Pythia output.\n" << endl;
    cout << "Examples:\n\n\t ./main93 [options] \n\n\t or\n\n\t ./main93 -c "
      "main93.cmnd -n 1000 -o myoutput\n" << endl;
    cout << "Options:\n"
      "\t -h, --help\n\t\t Show this help message and exit.\n"
      "\t -c CMND-FILE\n\t\t Use this user-written command file.\n"
      "\t -c2 CMND-FILE2\n\t\t Use a second cmnd file, loaded after "
       "the first.\n"
      "\t \t Useful for eg. tuning studies.\n"
      "\t -s SEED \n\t\t Specify seed for the random number generator.\n"
      "\t -o OUT \n\t\t Specify output prefix. Rivet histograms becomes "
      "OUT.yoda.\n"
      "\t -n NEVENTS\n\t\t Number of events.\n"
      "\t -l \n\t\t Silence the splash screen.\n"
      "\t -t \n\t\t Time event generation.\n"
      "\t -v \n\t\t Print the Pythia version number and exit.\n"
        << endl;
     cout << "Additional options in cmnd file:\n"
       "A few extra commands can be added to the cmnd file, compared "
       "to normal.\n"
       "\t Main:runRivet = on \n\t\tRun Rivet analyses (requires a\n"
       "\t\tworking installation of Rivet, linked to main93).\n"
       "\t Main:analyses = ANALYSIS1,ANALYSIS2,...\n "
       "\t\tA comma separated list of desired Rivet analyses to be run.\n"
       "\t\tAnalyses can be post-fixed with Rivet analysis parameters:\n"
        "\t\tANALYSIS:parm->value:parm2->value2 etc.\n"
       "\t Main:rivetRunName = STRING \n\t\tAdd an optional run name to "
       "the Rivet analysis.\n"
       "\t Main:rivetIgnoreBeams = on\n\t\tIgnore beams in Rivet. \n"
       "\t Main:rivetDumpPeriod = NUMBER\n\t\tDump Rivet histograms "
       "to file evert NUMBER of events.\n"
       "\t Main:rivetDumpFile = STRING\n\t\t Specify alternative "
       "name for Rivet dump file. Default = OUT.\n"
       "\t Main:writeHepMC = on \n\t\tWrite HepMC output (requires "
       "a linked installation of HepMC).\n"
       "\t Main:writeRoot = on \n\t\tWrite a root tree defined in the "
       "main93.h header file.\n\t\tRequires a working installation of Root, "
       "linked to Pythia.\n"
       "\t Main:outputLog = on\n\t\tRedirect output to a logfile. Default is "
       "OUT prefix, ie. pythia.log.\n"
         << endl;
    return 0;
  }

  // Print version number and exit.
  if(ip.hasOption("-v") || ip.hasOption("--version")) {
    cout << "PYTHIA version: " << PYTHIA_VERSION << endl;
    return 0;
  }

  string cmndfile = "";
  // Input command file.
  if(ip.hasOption("-c")) {
    cmndfile = ip.getOption("-c");
    if(cmndfile.find(".cmnd") == string::npos &&
        cmndfile.find(".dat") == string::npos) {
      cout << "Please provide a valid .cmnd file as "
      "argument to the -c option." << endl;
      return 1;
    }
  }
  else {
    cout << "You must provide a command file to produce output.\n"
            "Use option -h to show all command line options." << endl;
    return 1;
  }

  string cmndfile2 = "";
  // Optional secondary input command file.
  if(ip.hasOption("-c2")) {
    cmndfile2 = ip.getOption("-c2");
    if(cmndfile2.find(".cmnd") == string::npos &&
        cmndfile2.find(".dat") == string::npos) {
      cout << "Please provide a valid .cmnd file as argument "
      "to the -c2 option." << endl;
      return 1;
    }
  }

  string seed = "-1";
  // Optional seed from command line.
  if(ip.hasOption("-s")) seed = ip.getOption("-s");

  string out = "";
  // Set individual output prefix.
  if(ip.hasOption("-o")) out = ip.getOption("-o");

  bool takeTime = false;
  if (ip.hasOption("-t")) takeTime = true;

  int nev = -1;
  // Command line number of event, overrides the one set in input .cmnd file.
  if(ip.hasOption("-n")) nev = stoi(ip.getOption("-n"));

  // Catch the splash screen in a buffer.
  stringstream splashBuf;
  std::streambuf* sBuf = cout.rdbuf();
  cout.rdbuf(splashBuf.rdbuf());
  // The Pythia object.
  Pythia pythia;
  // Direct cout back.
  cout.rdbuf(sBuf);

  // UserHooks wrapper
  auto userHooksWrapper = make_shared<UserHooksWrapper>();
  userHooksWrapper->additionalSettings(&pythia.settings);
  pythia.setUserHooksPtr(userHooksWrapper);
  // Some extra parameters.
  pythia.settings.addFlag("Main:writeHepMC",false);
  pythia.settings.addFlag("Main:writeRoot",false);
  pythia.settings.addFlag("Main:runRivet",false);
  pythia.settings.addFlag("Main:rivetIgnoreBeams",false);
  pythia.settings.addMode("Main:rivetDumpPeriod",-1, true, false, -1, 0);
  pythia.settings.addWord("Main:rivetDumpFile", "");
  pythia.settings.addFlag("Main:outputLog",false);
  pythia.settings.addWVec("Main:analyses",vector<string>());
  pythia.settings.addWVec("Main:preload",vector<string>());
  pythia.settings.addWord("Main:rivetRunName","");
  // Read input from external file.
  pythia.readFile(cmndfile);

  if(cmndfile2 != "") pythia.readFile(cmndfile2);

  // Set seed after reading input
  if(seed != "-1") {
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = "+seed);
  }

  // Read the extra parameters.
  int nEvent = pythia.mode("Main:numberOfEvents");;
  if(nev > -1) nEvent = nev;
  const bool hepmc = pythia.flag("Main:writeHepMC");
  const bool root = pythia.flag("Main:writeRoot");
  const bool runRivet = pythia.flag("Main:runRivet");
  const bool ignoreBeams = pythia.flag("Main:rivetIgnoreBeams");
  const bool doLog = pythia.flag("Main:outputLog");
  const string rivetrName = pythia.settings.word("Main:rivetRunName");
  const vector<string> rAnalyses = pythia.settings.wvec("Main:analyses");
  const vector<string> rPreload = pythia.settings.wvec("Main:preload");
  const int rivetDump = pythia.settings.mode("Main:rivetDumpPeriod");
  const string rivetDumpName = pythia.settings.word("Main:rivetDumpFile");
  int nError = pythia.mode("Main:timesAllowErrors");
  const bool countErrors = (nError > 0 ? true : false);
  // HepMC conversion object.
  Pythia8ToHepMC ToHepMC;
  if (hepmc)
    ToHepMC.setNewFile((out == "" ? "pythia.hepmc" : out + ".hepmc"));
  // Rivet initialization.
  Pythia8Rivet rivet(pythia,(out == "" ? "Rivet.yoda" : out + ".yoda"));
  rivet.ignoreBeams(ignoreBeams);
  rivet.dump(rivetDump, rivetDumpName);
  for(int i = 0, N = rAnalyses.size(); i < N; ++i){
    string analysis = rAnalyses[i];
    size_t pos = analysis.find(":");
    // Simple case, no analysis parameters.
    if(pos == string::npos)
      rivet.addAnalysis(analysis);
    else {
      string an = analysis.substr(0,pos);
      analysis.erase(0, pos + 1);
      vector<string> pKeys;
      vector<string> pVals;
      while (analysis.find("->") != string::npos) {
        pos = analysis.find(":");
        string par = analysis.substr(0,pos);
        size_t pos2 = par.find("->");
        if (pos2 == string::npos){
           cout << "Error in main93: malformed parameter " << par << endl;
        }
        string pKey = par.substr(0,pos2);
        string pVal = par.substr(pos2+2,par.length());
        pKeys.push_back(pKey);
        pVals.push_back(pVal);
        analysis.erase(0,par.length()+1);
      }
      for (int j = 0, N = pKeys.size(); j < N; ++j)
        an += ":"+pKeys[j]+"="+pVals[j];
      rivet.addAnalysis(an);
    }
  }
  for(int i = 0, N = rPreload.size(); i < N; ++i)
    rivet.addPreload(rPreload[i]);
  rivet.addRunName(rivetrName);
  // Root initialization
  #ifdef PY8ROOT
  TFile* file;
  RootEvent* re;
  TTree* tree;
  #endif
  if (root) {
   // First test if root is available on system.
   #ifndef PY8ROOT
        cout << "Option Main::writeRoot = on requires a working,\n"
                "linked Root installation." << endl;
        return 1;
   #else
   string op = (out == "" ? "pythia.root" : out + ".root");
   file = TFile::Open(op.c_str(),"recreate" );
   re = new RootEvent();
   tree = new TTree("t","Pythia8 event tree");
   tree->Branch("events",&re);
   #endif
  }

  // Logfile initialization.
  ofstream logBuf;
  std::streambuf* oldCout;
  if(doLog) {
    oldCout = cout.rdbuf(logBuf.rdbuf());
    logBuf.open((out == "" ? "pythia.log" : out + ".log"));
  }
  // Option to trash the splash screen.
  ostream cnull(NULL);
  if(ip.hasOption("-l")) cnull << splashBuf.str();
  else cout << splashBuf.str();
  // Initialise Pythia.
  pythia.init();
  // Make a sanity check of initialized Rivet analyses
  if (!runRivet && rAnalyses.size() > 0 )
    cout << "Warning in main93: Rivet analyses initialized, but runRivet "
         << "set to off." << endl;
  // Loop over events.
  auto startAllEvents = std::chrono::high_resolution_clock::now();
  for ( int iEvent = 0; iEvent < nEvent; ++iEvent ) {
    auto startThisEvent = std::chrono::high_resolution_clock::now();
    if ( !pythia.next() ) {
      if (countErrors && --nError < 0) {
        pythia.stat();
        cout << " \n *-------  PYTHIA STOPPED!  -----------------------*"
             << endl;
        cout << " | Event generation failed due to too many errors. |" << endl;
        cout << " *-------------------------------------------------*" << endl;
        return 1;
      }
      continue;
    }
    auto stopThisEvent = std::chrono::high_resolution_clock::now();
    auto eventTime = std::chrono::duration_cast<std::chrono::milliseconds>
      (stopThisEvent - startThisEvent);
    double tt = eventTime.count();
    if (runRivet) {
      if (takeTime) rivet.addAttribute("EventTime", tt);
      rivet();
    }
    if (hepmc) {
      ToHepMC.writeNextEvent(pythia);
    }

    #ifdef PY8ROOT
    if (root) {
      // If we want to write a root file, the event must be skimmed here.
      vector<RootTrack> rts;
      for(int i = 0; i < pythia.event.size(); ++i) {
        RootTrack t;
        Particle& p = pythia.event[i];
        // Any particle cuts and further track definitions should
        // be implemented in the RootTrack class by the user.
        if (t.init(p)) rts.push_back(t);
      }
      bool fillTree = re->init(&pythia.info);
      re->tracks = rts;
      if(fillTree) tree->Fill();
    }
    #endif
    }
  pythia.stat();
  #ifdef PY8ROOT
  if (root) {
   tree->Print();
   tree->Write();
   delete file;
   delete re;
  }
  #endif
  auto stopAllEvents = std::chrono::high_resolution_clock::now();
  auto durationAll = std::chrono::duration_cast<std::chrono::milliseconds>
    (stopAllEvents - startAllEvents);
  if (takeTime) {
    cout << " \n *-------  Generation time  -----------------------*\n";
    cout << " | Event generation, analysis and writing to files  |\n";
    cout << " | took: " << double(durationAll.count()) << " ms or " <<
      double(durationAll.count())/double(nEvent) << " ms per event     |\n";
    cout << " *-------------------------------------------------*\n";
  }
  // Put cout back in its place.
  if (doLog) cout.rdbuf(oldCout);
  return 0;
}
