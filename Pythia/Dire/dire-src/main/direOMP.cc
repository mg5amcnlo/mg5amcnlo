
// DIRE includes.
#include "Dire/Dire.h"

// Pythia includes.
#include "Pythia8/Pythia.h"
#ifdef HEPMC2
#include "Pythia8Plugins/HepMC2.h"
#endif

// OpenMP includes.
#ifdef OPENMP
#include <omp.h>
#endif

using namespace Pythia8;

//==========================================================================

// The following functions analyze a scattering event and save the event in
// an output format that can be converted into a postscript figure using the
// "graphviz" program. Written by N. Fischer, 2017

string py_status(int stAbs) {
  string status    = "";
       if (stAbs > 20 && stAbs <  30) status = "hardProcess";
  else if (stAbs > 30 && stAbs <  40) status = "MPI";
  else if (stAbs > 40 && stAbs <  50) status = "ISR";
  else if (stAbs > 50 && stAbs <  60) status = "FSR";
  else if (stAbs > 60 && stAbs <  70) status = "beamRemnants";
  else if (stAbs > 70 && stAbs <  80) status = "hadronizationPrep";
  else if (stAbs > 80 && stAbs <  90) status = "hadronization";
  else if (stAbs > 90 && stAbs < 110) status = "decays";
  else                                status = "default";
  return status;
}

void makeArrow(map< pair<string,string>, string >* arrows,
  string identParent, string identChild) {
  pair<string,string> key = make_pair(identParent,identChild);
  string value = "  " + identParent + " -> " + identChild
    + " [weight=2,label=\" \"];";
  arrows->insert( pair< pair<string,string>, string>(key, value) );
}

void printEvent(Event& evt, string fileName = "event") {

  bool simplifyHadronization = true;
  bool addLegend             = true;
  map<string, pair<string,string> > colMap;
  colMap["default"]           = make_pair("white","black");
  colMap["hardProcess"]       = make_pair("red","black");
  colMap["MPI"]               = make_pair("lightsalmon","black");
  colMap["ISR"]               = make_pair("lightseagreen","black");
  colMap["FSR"]               = make_pair("limegreen","black");
  colMap["beamRemnants"]      = make_pair("mediumpurple","black");
  colMap["hadronizationPrep"] = make_pair("blue","black");
  colMap["hadronization"]     = make_pair("blue","black");
  colMap["decays"]            = make_pair("lightskyblue","black");

  map<string,string> blobs;
  map< pair<string,string>, string > arrows;
  vector< vector<int> > hadronGroups;
  vector< vector<int> > hadronParents;
  
  for (int i=1; i<(int)evt.size(); i++) {
    // Identifier of that particle.
    string ident     = "F" + STRING(10000+i);
    // Name that will appear in graph.
    string label     = STRING(evt[i].id()) + " (" + evt[i].name() + ")";
    // Find particle group for colors.
    string status    = py_status(evt[i].statusAbs());
    // Skip hadrons and decay products for simplified output.
    if (simplifyHadronization && 
      (status == "decays" || status == "hadronization") ) continue;
    // Special treatment of hadronization particles for simplified output.
    bool checkDaughters = simplifyHadronization;
    if (status != "hadronizationPrep" && status != "beamRemnants")
        checkDaughters = false;
    // Check that daughters are are part of hadronization
    if (checkDaughters) {
      vector<int> daus = evt[i].daughterList();
      for (int j=0; j<(int)daus.size(); j++)
        if (py_status(evt[daus[j]].statusAbs()) != "hadronization")
          checkDaughters = false;
    }
    if (checkDaughters) {
      vector<int> daus = evt[i].daughterList();
      // Check if other particles in preparation has same daughter list.
      bool foundSameDaus = false;
      for (int j=0; j<(int)hadronGroups.size(); j++) {
        if (daus.size() == hadronGroups[j].size()) {
          foundSameDaus = true;
          for (int k=0; k<(int)hadronGroups[j].size(); k++)
            if (daus[k] != hadronGroups[j][k]) foundSameDaus = false;
          if (foundSameDaus) {
            hadronParents[j].push_back(i);
            break;
          }
        }
      }
      if (!foundSameDaus) {
        hadronGroups.push_back(daus);
        vector<int> parents; parents.push_back(i);
        hadronParents.push_back(parents);
      }
      if (status == "hadronizationPrep") continue;
    }
    // Setup the graph for the particle.
    pair<string,string> colors = colMap[status];
    string fillcolor = colors.first, fontcolor = colors.second;
    blobs[ident] = "  " + ident + " [shape=box,style=filled,fillcolor=\""
      + fillcolor + "\",fontcolor=\"" + fontcolor + "\",label=\""
      + label + "\"];";
    // Setup arrow to mother(s).
    int mot1 = evt[i].mother1(), mot2 = evt[i].mother2();
    if ( i > 3 && (mot1 == 0 || mot2 == 0) ) 
      makeArrow(&arrows, "F"+STRING(10000+max(mot1,mot2)), ident);
    // Setup arrow to daughter(s).
    if (!checkDaughters) {
      vector<int> daus = evt[i].daughterList();
      for (int j=0; j<(int)daus.size(); j++)
        makeArrow(&arrows, ident, "F"+STRING(10000+daus[j]));
    }
  }

  // Add the hadron groups for simplified output.
  map< pair<string,string>, string > arrowsSav = arrows;
  for (int i=0; i<(int)hadronGroups.size(); i++) {
    // Identifier of that group.
    string ident     = "G" + STRING(10000+i);
    pair<string,string> colors = colMap["hadronization"];
    string fillcolor = colors.first, fontcolor = colors.second;
    string line      = "  " + ident + " [shape=none,\n     label = <<"
      "table border=\"0\" cellspacing=\"0\">\n";
    for (int j=0; j<(int)hadronGroups[i].size(); j++) {
      // Name that will appear in graph.
      string label = STRING(evt[hadronGroups[i][j]].id()) + " ("
        + evt[hadronGroups[i][j]].name() + ")";
      line += ( "               <tr><td port=\"port" + STRING(j)
        + "\" border=\"1\" bgcolor=\"" + fillcolor + "\"><font color=\""
        + fontcolor + "\">" + label + "</font></td></tr>\n" );
    }
    line += "             </table>> ];";
    // Add the group to the graph.
    blobs[ident] = line;
    // Add an arrow from each parent to the group.
    for (int j=0; j<(int)hadronParents[i].size(); j++) {
      // Identifier of that parent.
      string identParent = "F"+STRING(10000+hadronParents[i][j]);
      // List of particles to be erased.
      vector<string> toErase;
      toErase.push_back(identParent);
      // Check if parent is beam remnant.
      bool parentIsBR = (py_status(evt[hadronParents[i][j]].statusAbs()) ==
        "beamRemnants");
      if (parentIsBR) {
        makeArrow(&arrows, identParent, ident);
      } else {
        int nrGP1 = evt[hadronParents[i][j]].mother1();
        int nrGP2 = evt[hadronParents[i][j]].mother2();
        if (nrGP1 > 0) {
          // Trace back one more generation if double hadronization prep.
          if (py_status(evt[nrGP1].statusAbs()) == "hadronizationPrep") {
            toErase.push_back("F"+STRING(10000+nrGP1));
            int nrGGP1 = evt[nrGP1].mother1();
            int nrGGP2 = evt[nrGP1].mother2();
            if (nrGGP1 > 0) makeArrow(&arrows, "F"+STRING(10000+nrGGP1), ident);
            if (nrGGP2 > 0 && nrGGP2 != nrGGP1)
              makeArrow(&arrows, "F"+STRING(10000+nrGGP2), ident);
          } else makeArrow(&arrows, "F"+STRING(10000+nrGP1), ident);
        }
        if (nrGP2 > 0 && nrGP2 != nrGP1) {
          // Trace back one more generation if double hadronization prep.
          if (py_status(evt[nrGP2].statusAbs()) == "hadronizationPrep") {
            toErase.push_back("F"+STRING(10000+nrGP2));
            int nrGGP1 = evt[nrGP2].mother1();
            int nrGGP2 = evt[nrGP2].mother2();
            if (nrGGP1 > 0) makeArrow(&arrows, "F"+STRING(10000+nrGGP1), ident);
            if (nrGGP2 > 0 && nrGGP2 != nrGGP1)
              makeArrow(&arrows, "F"+STRING(10000+nrGGP2), ident);
          } else makeArrow(&arrows, "F"+STRING(10000+nrGP2), ident);
        }
        // Erase any parents that might be left in the graph.
        for (int iToE=0; iToE<(int)toErase.size(); iToE++)
          if (blobs.find(toErase[iToE]) != blobs.end())
            blobs.erase(toErase[iToE]);
        for (map< pair<string,string>, string >::iterator k=arrowsSav.begin();
          k!=arrowsSav.end(); k++) {
          for (int iToE=0; iToE<(int)toErase.size(); iToE++) {
            if (k->first.second == toErase[iToE]) 
              arrows.erase(k->first);
          }
        }
      }
    }
  }

  // Write output.
  ofstream outfile;
  outfile.open((char*)(fileName+".dot").c_str());
  outfile << "digraph \"event\" {" << endl
          << "  rankdir=LR;" << endl;
  for (map<string,string>::iterator iBlob=blobs.begin(); iBlob!=blobs.end();
    iBlob++) outfile << iBlob->second << endl;
  for (map< pair<string,string>, string >::iterator iArrow=arrows.begin();
    iArrow!=arrows.end(); iArrow++) outfile << iArrow->second << endl;
  // Add a legend, skip default.
  if (addLegend) {
    outfile << "  { rank = source;" << endl
            << "    Legend [shape=none, margin=0, label=<<table border=\"0\""
            << " cellspacing=\"0\">" << endl
            << "     <tr><td port=\"0\" border=\"1\"><b>Legend</b></td></tr>" << endl;
    int count = 1;
    for (map<string, pair<string,string> >::iterator iLeg=colMap.begin();
      iLeg!=colMap.end(); iLeg++) {
      if (iLeg->first == "default") continue;
      if (iLeg->first == "hadronizationPrep") continue;
      if (simplifyHadronization && iLeg->first == "decays") continue;
      string fillcolor = iLeg->second.first;
      string fontcolor = iLeg->second.second;
      outfile << "     <tr><td port=\"port" << count << "\" border=\"1\" "
              << "bgcolor=\"" << fillcolor << "\"><font color=\"" << fontcolor
              << "\">" << iLeg->first << "</font></td></tr>" << endl;
      count++;
    }
    outfile << "    </table>" << endl << "   >];" << endl << "  }" << endl;
  }
  outfile << "}" << endl;
  outfile.close();

  cout << "\n\nPrinted one event to output file " << fileName + ".dot\n";
  if (system(NULL)) {
    if (system("which dot > /dev/null 2>&1") == 0) {
      cout << "Producing .ps figure by using the 'dot' command." << endl;
      string command =  "dot -Tps " + fileName + ".dot -o " + fileName+".ps"; 
      if (system(command.c_str()) == 0)
        cout << "Stored event visualization in file " << fileName+".ps" << endl;
      else
        cout << "Failed to store event visualization in file." << endl;
    }
  } else {
    cout << "You can now produce a .ps figure by using the 'dot' command:\n\n"
      << "dot -Tps " << fileName << ".dot -o " << fileName << ".ps" << "\n\n";
    cout << "Note: 'dot' is part of the 'graphviz' package.\n"
      << "You might want to install this package to produce the .ps event"
      << endl << endl;
  }

}

//==========================================================================
// An example Dire main program.

int main( int argc, char* argv[] ){

  // Get command-line arguments
  vector<string> arguments;
  for (int i = 0; i < argc; ++i) { 
    arguments.push_back(string(argv[i]));
    if (arguments.back() == "--visualize_event")
      arguments.push_back(" ");
  }

  // Print help.
  for (int i = 0; i < int(arguments.size()); ++i) {
    if ( arguments[i] == "--help" || arguments[i] == "-h") {
      cout << "\n"
        << "Simple standardized executable for the Pythia+Dire event "
        << "generator.\n\n" 
        << "Usage:\n\n"
        << "dire [option] <optionValue> [option] <optionValue> ...\n\n"
        << "Examples:\n\n"
        << "dire --nevents 50 --setting \"WeakSingleBoson:ffbar2gmZ = on\"\n"
        << "dire --input main/lep.cmnd --hepmc_output myfile.hepmc\n\n"
        << "Options:\n\n"
        << "  --visualize_event       :"
        << " Saves one event for visualization of event generation steps.\n"
        << "  --nevents N             :"
        << " Generate N events (overwrites default value and\n"
        << "                           "
        << " number of events in input settings file).\n"
        << "  --nthreads N            :"
        << " Use N threads, takes effect only if Dire was configured\n"
        << "                            with OpenMP\n"
        << "  --input FILENAME        :"
        << " Use file FILENAME to read & use Pythia settings.\n" 
        << "                            Multiple input files are allowed.\n" 
        << "  --hepmc_output FILENAME :"
        << " Store generated events in HepMC file FILENAME.\n" 
        << "  --setting VALUE         :"
        << " Use the Pythia/Dire setting VALUE for event generation, e.g.\n"
        << "                            --setting Beams:eCM=100.0\n"
        << "                            --setting \"Beams:idA = -11\"\n"
        << "                            --setting \"PartonLevel:MPI = off\"\n"
        << "                           "
        << " possible Pythia/Dire settings can be found in the\n"
        << "                            respective online manuals\n\n"
        << "More documentation can be found on dire.gitlab.io\n" << endl;
      return 0;
    }
  }

  // Parse command-line arguments
  // input file
  vector<string>::iterator it
     = std::find(arguments.begin(),arguments.end(),"--input");
  string input  = (it != arguments.end()) ? *(it+1) : "";
  // output hepmc file
  it = std::find(arguments.begin(),arguments.end(),"--hepmc_output");
  string hepmc_output = (it != arguments.end()) ? *(it+1) : "";
  // number of events to generate
  it = std::find(arguments.begin(),arguments.end(),"--nevents");
  int nevents = (it != arguments.end()) ? atoi((*(it+1)).c_str()) : -1;
#ifdef OPENMP
  // number of threads
  it = std::find(arguments.begin(),arguments.end(),"--nthreads");
  int nThreads = (it != arguments.end()) ? atoi((*(it+1)).c_str()) : 1;
#endif
  // visualize_event flag
  it = std::find(arguments.begin(),arguments.end(),"--visualize_event");
  bool visualize_event     = (it != arguments.end());
  string visualize_output  = (input == "") ? "event" : "event-" + input;
  replace(visualize_output.begin(), visualize_output.end(), '/', '-');

  vector<Pythia*> pythiaPtr;
  vector<Dire*> direPtr;
#ifdef MG5MES
  // Allow Pythia to use Dire merging classes. 
  vector<MyMerging*> mergingPtr;
  vector<MyHardProcess*> hardProcessPtr;
  vector<MyMergingHooks*> mergingHooksPtr;
#endif

  // Read input files.
  vector<string> input_file;
  for (int i = 0; i < int(arguments.size()); ++i)
    if (arguments[i] == "--input" && i+1 <= int(arguments.size())-1)
      input_file.push_back(arguments[i+1]);
  if (input_file.size() < 1) input_file.push_back("");

  // For several settings files as input, check that they use
  // a different process.
  if (input_file.size() > 1) {
    bool sameProcess = false;
    for (int i = 1; i < int(input_file.size()); ++i)
      if (input_file[i] == input_file[i-1]) sameProcess = true;
    if (sameProcess)
      cout << "WARNING: several input files with the same name\n"
        << " found; this will lead to a wrong cross section!\n";
  }

  std::streambuf *old = cout.rdbuf();
  for (int i = 0; i < int(input_file.size()); ++i) {
    stringstream ss;
    // Redirect output so that Pythia banner will not be printed twice.
    if(i>0) cout.rdbuf (ss.rdbuf());
    pythiaPtr.push_back( new Pythia());
    // Restore print-out.
    cout.rdbuf (old);
    direPtr.push_back( new Dire());
#ifdef MG5MES
    mergingPtr.push_back( new MyMerging());
    hardProcessPtr.push_back( new MyHardProcess());
    mergingHooksPtr.push_back( new MyMergingHooks());
    mergingHooksPtr.back()->setHardProcessPtr( hardProcessPtr.back() );
#endif
  }

#ifdef OPENMP
  old = cout.rdbuf();
  int nThreadsMax = omp_get_max_threads();
  int nThreadsReq = min(nThreads,nThreadsMax);
  // Divide a single Pythia object into several, in case of multiple threads.
  int nPythiaOrg = (int)pythiaPtr.size();
  if (nThreadsReq > 1 && nPythiaOrg == 1) {
    for (int i = 1; i < nThreadsReq; ++i) {
      input_file.push_back(input_file.front());
      stringstream ss;
      // Redirect output so that Pythia banner will not be printed twice.
      cout.rdbuf (ss.rdbuf());
      pythiaPtr.push_back( new Pythia());
      direPtr.push_back( new Dire());
#ifdef MG5MES
      mergingPtr.push_back( new MyMerging());
      hardProcessPtr.push_back( new MyHardProcess());
      mergingHooksPtr.push_back( new MyMergingHooks());
      mergingHooksPtr.back()->setHardProcessPtr( hardProcessPtr.back() );
#endif
    }
  }
  // Restore print-out.
  cout.rdbuf (old);

  bool doParallel = true;
  int  nPythiaAct = (int)pythiaPtr.size();
  // The number of Pythia objects exceeds the number of available threads.
  if (nPythiaAct > nThreadsReq) {
    cout << "WARNING: The number of requested Pythia instances exceeds the\n"
      << " number of available threads! No parallelization will be done!\n";
    nThreadsReq = 1;
    doParallel  = false;
  }
#endif

  // Read command line settings.
  for (int i = 0; i < int(arguments.size()); ++i) {
    if (arguments[i] == "--setting" && i+1 <= int(arguments.size())-1) {
      string setting = arguments[i+1];
      replace(setting.begin(), setting.end(), '"', ' ');      

      // Skip Dire settings at this stage.
      if (setting.find("Dire") != string::npos) continue;

      for (int j = 0; j < int(pythiaPtr.size()); ++j)
        pythiaPtr[j]->readString(setting);

    }
  }

#ifdef MG5MES
  // Allow Pythia to use Dire merging classes. 
  for (int i = 0; i < int(pythiaPtr.size()); ++i) {
    pythiaPtr[i]->setMergingHooksPtr(mergingHooksPtr[i]);
    pythiaPtr[i]->setMergingPtr(mergingPtr[i]);
  }
#endif

  // Allow Pythia to use Dire merging classes. 
  for (int i = 0; i < int(direPtr.size()); ++i)
    direPtr[i]->init(*pythiaPtr[i], input_file[i].c_str());

  // Read command line settings again and overwrite file settings.
  for (int i = 0; i < int(arguments.size()); ++i) {
    if (arguments[i] == "--setting" && i+1 <= int(arguments.size())-1) {
      string setting = arguments[i+1];
      replace(setting.begin(), setting.end(), '"', ' ');      
      for (int j = 0; j < int(pythiaPtr.size()); ++j)
        pythiaPtr[j]->readString(setting);
    }
  }

#ifdef OPENMP
  // A divided single Pythia run does not work with Les Houches events.
  if (nThreadsReq > 1 && nPythiaOrg == 1 &&
    pythiaPtr.front()->mode("Beams:frameType") > 3) {
    cout << "WARNING: can not divide the run into subruns as the\n"
      << " same hard events from the Les Houches file would be\n"
      << " used multiple times!\n";
    // Clean-up.
    nThreadsReq = 1;
    for (int i = 1; i < int(pythiaPtr.size()); ++i) {
      delete pythiaPtr[i];
      delete direPtr[i];
#ifdef MG5MES
      delete mergingPtr[i];
      delete hardProcessPtr[i];
      delete mergingHooksPtr[i];
#endif
    }
  }
  // Add random seeds for parallelization of a single Pythia run.
  bool splitSingleRun = false;
  if (nThreadsReq > 1 && nPythiaOrg == 1) {
    splitSingleRun = true;
    int randomOffset = 100;
    for (int j = 0; j < int(pythiaPtr.size()); ++j) {
      pythiaPtr[j]->readString("Random:setSeed = on");
      pythiaPtr[j]->readString("Random:seed = "+STRING(randomOffset+j));
    }
  }
#endif

#ifdef MG5MES
  // Transfer initialized shower weights pointer to merging class.
  for (int i = 0; i < int(direPtr.size()); ++i) {
    mergingPtr[i]->setWeightsPtr(direPtr[i]->weightsPtr);
    mergingPtr[i]->setShowerPtrs(direPtr[i]->timesPtr, direPtr[i]->spacePtr);
  }
#endif

  int nEventEst = pythiaPtr.front()->settings.mode("Main:numberOfEvents");
  if (nevents > 0) nEventEst = nevents;

  int nEventEstEach = nEventEst;
#ifdef OPENMP
  // Number of events per thread.
  if (nThreadsReq > 1) {
    while (nEventEst%nThreadsReq != 0) nEventEst++;
    nEventEstEach = nEventEst/nThreadsReq;
  }
#endif

  // Switch off all showering and MPI when estimating the cross section,
  // and re-initialise (unfortunately).
  bool fsr = pythiaPtr.front()->flag("PartonLevel:FSR");
  bool isr = pythiaPtr.front()->flag("PartonLevel:ISR");
  bool mpi = pythiaPtr.front()->flag("PartonLevel:MPI");
  bool had = pythiaPtr.front()->flag("HadronLevel:all");
  bool rem = pythiaPtr.front()->flag("PartonLevel:Remnants");
  bool chk = pythiaPtr.front()->flag("Check:Event");
  if (!visualize_event) {
    for (int i = 0; i < int(pythiaPtr.size()); ++i) {
      pythiaPtr[i]->settings.flag("PartonLevel:FSR",false);
      pythiaPtr[i]->settings.flag("PartonLevel:ISR",false);
      pythiaPtr[i]->settings.flag("PartonLevel:MPI",false);
      pythiaPtr[i]->settings.flag("HadronLevel:all",false);
      pythiaPtr[i]->settings.flag("PartonLevel:Remnants",false);
      pythiaPtr[i]->settings.flag("Check:Event",false);
    }
  }

  // Force PhaseSpace:pTHatMinDiverge to something very small to not bias DIS.
  for (int i = 0; i < int(pythiaPtr.size()); ++i)
    pythiaPtr[i]->settings.forceParm("PhaseSpace:pTHatMinDiverge",1e-6);

  for (int i = 0; i < int(pythiaPtr.size()); ++i)
    pythiaPtr[i]->init();

  // Cross section estimate run.
  vector<double> nash, sumsh;
  for (int i = 0; i < int(pythiaPtr.size()); ++i) {
    nash.push_back(0.);
    sumsh.push_back(0.);
  }

  // Save a single event for event generation visualization.
  if (visualize_event) {
    while ( !pythiaPtr.front()->next() )
      if ( pythiaPtr.front()->info.atEndOfFile() ) break;
    printEvent(pythiaPtr.front()->event, visualize_output);
    cout << "\nCreated event visualization. Exiting event generation.\n"<<endl; 
    // Clean-up.
    for (int i = 0; i < int(pythiaPtr.size()); ++i) {
      delete pythiaPtr[i];
      delete direPtr[i];
#ifdef MG5MES
      delete mergingPtr[i];
      delete hardProcessPtr[i];
      delete mergingHooksPtr[i];
#endif
    }
    return 0;
  }

#ifdef OPENMP
#pragma omp parallel num_threads(nThreadsReq)
{
  for (int i = 0; i < nEventEstEach; ++i) {
    vector<int> pythiaIDs;
    // If parallelization can not be done, loop through all
    // Pythia objects.
    if (!doParallel)
      for (int j = 0; j < int(pythiaPtr.size()); ++j)
        pythiaIDs.push_back(j);
    else pythiaIDs.push_back(omp_get_thread_num());
    for (int id = 0; id < int(pythiaIDs.size()); ++id) {
      int j = pythiaIDs[id];
      if ( !pythiaPtr[j]->next() ) {
        if ( pythiaPtr[j]->info.atEndOfFile() ) break;
        else continue;
      }
      sumsh[j] += pythiaPtr[j]->info.weight();
      map <string,string> eventAttributes;
      if (pythiaPtr[j]->info.eventAttributes)
        eventAttributes = *(pythiaPtr[j]->info.eventAttributes);
      string trials = (eventAttributes.find("trials") != eventAttributes.end())
                    ?  eventAttributes["trials"] : "";
      if (trials != "") nash[j] += atof(trials.c_str());
    }
  }
}
#pragma omp barrier
#else
  for (int i = 0; i < nEventEstEach; ++i) {
    for (int j = 0; j < int(pythiaPtr.size()); ++j) {
      if ( !pythiaPtr[j]->next() ) {
        if ( pythiaPtr[j]->info.atEndOfFile() ) break;
        else continue;
      }
      sumsh[j] += pythiaPtr[j]->info.weight();
      map <string,string> eventAttributes;
      if (pythiaPtr[j]->info.eventAttributes)
        eventAttributes = *(pythiaPtr[j]->info.eventAttributes);
      string trials = (eventAttributes.find("trials") != eventAttributes.end())
                    ?  eventAttributes["trials"] : "";
      if (trials != "") nash[j] += atof(trials.c_str());
    }
  }
#endif

  // Store estimated cross sections.
  vector<double> na, xss;
  old = cout.rdbuf();
  for (int i = 0; i < int(pythiaPtr.size()); ++i) {
    // Redirect output so that Pythia banner will not be printed twice.
    stringstream ss;
    cout.rdbuf (ss.rdbuf());
    pythiaPtr[i]->stat();
    na.push_back(pythiaPtr[i]->info.nAccepted());
    xss.push_back(pythiaPtr[i]->info.sigmaGen());
  }
  // Restore print-out.
  cout.rdbuf (old);

#ifdef HEPMC2
  bool printHepMC = !(hepmc_output == "");
  // Interface for conversion from Pythia8::Event to HepMC one. 
  HepMC::Pythia8ToHepMC ToHepMC;
  // Specify file where HepMC events will be stored.
  HepMC::IO_GenEvent ascii_io(hepmc_output, std::ios::out);
  // Switch off warnings for parton-level events.
  ToHepMC.set_print_inconsistency(false);
  ToHepMC.set_free_parton_exception(false);
  // Do not store cross section information, as this will be done manually.
  ToHepMC.set_store_pdf(false);
  ToHepMC.set_store_proc(false);
  ToHepMC.set_store_xsec(false);
  vector< HepMC::IO_GenEvent* > ascii_io_var;
#endif

  // Cross section and weights:
  // Total for all runs combined: main.
  double sigmaTotAll(0.);
  // Total for all runs combined: variations.
  vector<double> sigmaTotVarAll;
  // Individual run: main.
  vector<double> sigmaInc, sigmaTot, sumwt, sumwtsq;
  // Individual run: variations.
  vector<vector<double> > sigmaTotVar;
  // Check variations.
  bool doVariationsAll(true);
  for (int i = 0; i < int(pythiaPtr.size()); ++i)
    if ( ! pythiaPtr.front()->settings.flag("Variations:doVariations") )
      doVariationsAll = false;
  if ( doVariationsAll ) {
    //for (int iwt=0; iwt<direPtr.front()->weightsPtr->sizeWeights(); ++iwt) {
    for (int iwt=0; iwt < 3; ++iwt) {
#ifdef HEPMC2
      if (printHepMC) {
        //string hepmc_var = hepmc_output + "-"
        //  + direPtr.front()->weightsPtr->weightName(iwt);
        //std::replace(hepmc_var.begin(), hepmc_var.end(),' ', '_');
        //std::replace(hepmc_var.begin(), hepmc_var.end(),':', '_');
        string hepmc_var = hepmc_output + "-" + STRING(iwt);
        ascii_io_var.push_back( new HepMC::IO_GenEvent(hepmc_var, std::ios::out));
      }
#endif
      sigmaTotVarAll.push_back(0.);
    }
  }
  for (int i = 0; i < int(pythiaPtr.size()); ++i) {
    sigmaTotVar.push_back(vector<double>());
    if ( doVariationsAll ) {
      //for (int iwt=0; iwt<direPtr.front()->weightsPtr->sizeWeights(); ++iwt)
      for (int iwt=0; iwt < 3; ++iwt)
        sigmaTotVar.back().push_back(0.);
    }
    sigmaInc.push_back(0.);
    sigmaTot.push_back(0.);
    sumwt.push_back(0.);
    sumwtsq.push_back(0.);
  }

  int nEvent = pythiaPtr.front()->settings.mode("Main:numberOfEvents");
  if (nevents > 0) nEvent = nevents;

  int nEventEach = nEvent;
#ifdef OPENMP
  // Number of events per thread.
  if (nThreadsReq > 1) {
    while (nEvent%nThreadsReq != 0) nEvent++;
    nEventEach = nEvent/nThreadsReq;
  }
#endif

  cout << endl << endl << endl;
  cout << "Start generating events" << endl;

  // Switch showering and multiple interaction back on.
  for (int i = 0; i < int(pythiaPtr.size()); ++i) {
    pythiaPtr[i]->settings.flag("PartonLevel:FSR",fsr);
    pythiaPtr[i]->settings.flag("PartonLevel:ISR",isr);
    pythiaPtr[i]->settings.flag("HadronLevel:all",had);
    pythiaPtr[i]->settings.flag("PartonLevel:MPI",mpi);
    pythiaPtr[i]->settings.flag("PartonLevel:Remnants",rem);
    pythiaPtr[i]->settings.flag("Check:Event",chk);
  }

  // Reinitialize Pythia for event generation runs.
  for (int i = 0; i < int(pythiaPtr.size()); ++i)
    pythiaPtr[i]->init();

  // Event generation run.
  double wmax =-1e15;
  double wmin = 1e15;

#ifdef OPENMP
#pragma omp parallel num_threads(nThreadsReq)
{
  for (int i = 0; i < nEventEach; ++i) {
    vector<int> pythiaIDs;
    // If parallelization can not be done, loop through all
    // Pythia objects.
    if (!doParallel)
      for (int j = 0; j < int(pythiaPtr.size()); ++j)
        pythiaIDs.push_back(j);
    else pythiaIDs.push_back(omp_get_thread_num());
    for (int id = 0; id < int(pythiaIDs.size()); ++id) {
      int j = pythiaIDs[id];
      if ( !pythiaPtr[j]->next() ) {
        if ( pythiaPtr[j]->info.atEndOfFile() ) break;
        else continue;
      }

      // Get event weight(s).
      double evtweight = pythiaPtr[j]->info.weight();

      // Do not print zero-weight events.
      if ( evtweight == 0. ) continue;

      // Retrieve the shower weight.
      direPtr[j]->weightsPtr->calcWeight(0.);
      direPtr[j]->weightsPtr->reset();
      double pswt = direPtr[j]->weightsPtr->getShowerWeight();

      if (abs(pswt) > 1e3) {
        cout << scientific << setprecision(8)
          << "Warning in DIRE main program dire.cc: Large shower weight wt="
          << pswt << endl;
        if (abs(pswt) > 1e4) { 
          cout << "Warning in DIRE main program dire.cc: Shower weight larger"
            << " than 10000. Discard event with rare shower weight fluctuation."
            << endl;
          evtweight = 0.;
        }
        // Print diagnostic output.
        direPtr[j]->debugInfo.print(1);
        evtweight = 0.;
      }

      // Do not print zero-weight events.
      if ( evtweight == 0. ) continue;

      // Now retrieve additional shower weights, and combine these
      // into muR-up and muR-down variations.
      vector<double> pswts;
      //if (pythia.settings.flag("Variations:doVariations")) { 
      //  for (int iwt=0; iwt < dire.weightsPtr->sizeWeights(); ++iwt) {
      //    string key = dire.weightsPtr->weightName(iwt);
      //    pswts.push_back(dire.weightsPtr->getShowerWeight(key));
      //  }
      //}

      if (pythiaPtr[j]->settings.flag("Variations:doVariations")) { 
        pswts.push_back(direPtr[j]->weightsPtr->getShowerWeight("base"));
        bool hasupvar(false), hasdownvar(false);
        double uvar(1.), dvar(1.);
        // Get ISR variations.
        if ( pythiaPtr[j]->settings.flag("PartonLevel:ISR")) {
          if ( pythiaPtr[j]->settings.parm("Variations:muRisrUp") != 1.) {
            hasupvar=true;
            uvar *= direPtr[j]->weightsPtr->getShowerWeight("Variations:muRisrUp");
          }
          if ( pythiaPtr[j]->settings.parm("Variations:muRisrDown") != 1.) {
            hasdownvar=true;
            dvar *= direPtr[j]->weightsPtr->getShowerWeight("Variations:muRisrDown");
          }
        }
        // Get FSR variations.
        if ( pythiaPtr[j]->settings.flag("PartonLevel:FSR")) {
          if ( pythiaPtr[j]->settings.parm("Variations:muRfsrUp") != 1.) {
            hasupvar=true;
            uvar *= direPtr[j]->weightsPtr->getShowerWeight("Variations:muRfsrUp");
          }
          if ( pythiaPtr[j]->settings.parm("Variations:muRfsrDown") != 1.) {
            hasdownvar=true;
            dvar *= direPtr[j]->weightsPtr->getShowerWeight("Variations:muRfsrDown");
          }
        }
        if (hasupvar && abs(uvar) < 1e3)   pswts.push_back(uvar);
        else            pswts.push_back(0.0);
        if (hasdownvar && abs(dvar) < 1e3) pswts.push_back(dvar);
        else            pswts.push_back(0.0);
      }

#pragma omp critical
{
      wmin = min(wmin,pswt);
      wmax = max(wmax,pswt);

      sumwt[j]   += pswt;
      sumwtsq[j] += pow2(pswt);

      double normhepmc = xss[j]/na[j];

      // Weighted events with additional number of trial events to consider.
      if ( pythiaPtr[j]->info.lhaStrategy() != 0
        && pythiaPtr[j]->info.lhaStrategy() != 3
        && nash[j] > 0)
        normhepmc = 1. / (1e9*nash[j]);
      // Weighted events.
      else if ( pythiaPtr[j]->info.lhaStrategy() != 0
        && pythiaPtr[j]->info.lhaStrategy() != 3
        && nash[j] == 0)
        normhepmc = 1. / (1e9*na[j]);

      if (pythiaPtr[j]->settings.flag("PhaseSpace:bias2Selection"))
        normhepmc = xs / (sumsh[j]);

      if (pythiaPtr[j]->event.size() > 3) {

        double w = evtweight*pswt*normhepmc;
        // Add the weight of the current event to the cross section.
        double divide = (splitSingleRun ? double(nThreadsReq) : 1.0);
        sigmaTotAll += w/divide;
        sigmaInc[j] += evtweight*normhepmc;
        sigmaTot[j] += w;
#ifdef HEPMC2
        if (printHepMC) {
          // Construct new empty HepMC event.
          HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
          // Set event weight
          hepmcevt->weights().push_back(w);
          // Fill HepMC event
          ToHepMC.fill_next_event( *pythiaPtr[j], hepmcevt );
          // Report cross section to hepmc
          HepMC::GenCrossSection xsec;
          xsec.set_cross_section( sigmaTotAll*1e9,
            pythiaPtr[j]->info.sigmaErr()*1e9 );
          hepmcevt->set_cross_section( xsec );
          // Write the HepMC event to file. Done with it.
          ascii_io << hepmcevt;
          delete hepmcevt;
        }
#endif

        // Write additional HepMC events.
        for (int iwt=0; iwt < int(pswts.size()); ++iwt) {
          double wVar = evtweight*pswts[iwt]*normhepmc;
          // Add the weight of the current event to the cross section.
          double divideVar = (splitSingleRun ? double(nThreadsReq) : 1.0);
          sigmaTotVarAll[iwt] += wVar/divideVar;
          sigmaTotVar[j][iwt] += wVar;
#ifdef HEPMC2
          if (printHepMC) {
            HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
            // Set event weight
            hepmcevt->weights().push_back(wVar);
            // Fill HepMC event
            ToHepMC.fill_next_event( *pythiaPtr[j], hepmcevt );
            // Report cross section to hepmc
            HepMC::GenCrossSection xsec;
            xsec.set_cross_section( sigmaTotVarAll[iwt]*1e9,
              pythiaPtr[j]->info.sigmaErr()*1e9 );
            hepmcevt->set_cross_section( xsec );
            // Write the HepMC event to file. Done with it.
            *ascii_io_var[iwt] << hepmcevt;
            delete hepmcevt;
          }
#endif
        }
}
      }

    }
  }
}
#pragma omp barrier
#else
  for (int i = 0; i < nEventEach; ++i) {
    for (int j = 0; j < int(pythiaPtr.size()); ++j) {
      if ( !pythiaPtr[j]->next() ) {
        if ( pythiaPtr[j]->info.atEndOfFile() ) break;
        else continue;
      }

      // Get event weight(s).
      double evtweight = pythiaPtr[j]->info.weight();

      // Do not print zero-weight events.
      if ( evtweight == 0. ) continue;

      // Retrieve the shower weight.
      direPtr[j]->weightsPtr->calcWeight(0.);
      direPtr[j]->weightsPtr->reset();
      double pswt = direPtr[j]->weightsPtr->getShowerWeight();

      if (abs(pswt) > 1e3) {
        cout << scientific << setprecision(8)
          << "Warning in DIRE main program dire.cc: Large shower weight wt="
          << pswt << endl;
        if (abs(pswt) > 1e4) { 
          cout << "Warning in DIRE main program dire.cc: Shower weight larger"
            << " than 10000. Discard event with rare shower weight fluctuation."
            << endl;
          evtweight = 0.;
        }
        // Print diagnostic output.
        direPtr[j]->debugInfo.print(1);
        evtweight = 0.;
      }

      // Do not print zero-weight events.
      if ( evtweight == 0. ) continue;

      // Now retrieve additional shower weights, and combine these
      // into muR-up and muR-down variations.
      vector<double> pswts;
      //if (pythia.settings.flag("Variations:doVariations")) { 
      //  for (int iwt=0; iwt < dire.weightsPtr->sizeWeights(); ++iwt) {
      //    string key = dire.weightsPtr->weightName(iwt);
      //    pswts.push_back(dire.weightsPtr->getShowerWeight(key));
      //  }
      //}

      if (pythiaPtr[j]->settings.flag("Variations:doVariations")) { 
        pswts.push_back(direPtr[j]->weightsPtr->getShowerWeight("base"));
        bool hasupvar(false), hasdownvar(false);
        double uvar(1.), dvar(1.);
        // Get ISR variations.
        if ( pythiaPtr[j]->settings.flag("PartonLevel:ISR")) {
          if ( pythiaPtr[j]->settings.parm("Variations:muRisrUp") != 1.) {
            hasupvar=true;
            uvar *= direPtr[j]->weightsPtr->getShowerWeight("Variations:muRisrUp");
          }
          if ( pythiaPtr[j]->settings.parm("Variations:muRisrDown") != 1.) {
            hasdownvar=true;
            dvar *= direPtr[j]->weightsPtr->getShowerWeight("Variations:muRisrDown");
          }
        }
        // Get FSR variations.
        if ( pythiaPtr[j]->settings.flag("PartonLevel:FSR")) {
          if ( pythiaPtr[j]->settings.parm("Variations:muRfsrUp") != 1.) {
            hasupvar=true;
            uvar *= direPtr[j]->weightsPtr->getShowerWeight("Variations:muRfsrUp");
          }
          if ( pythiaPtr[j]->settings.parm("Variations:muRfsrDown") != 1.) {
            hasdownvar=true;
            dvar *= direPtr[j]->weightsPtr->getShowerWeight("Variations:muRfsrDown");
          }
        }
        if (hasupvar && abs(uvar) < 1e3)   pswts.push_back(uvar);
        else            pswts.push_back(0.0);
        if (hasdownvar && abs(dvar) < 1e3) pswts.push_back(dvar);
        else            pswts.push_back(0.0);
      }

      wmin = min(wmin,pswt);
      wmax = max(wmax,pswt);

      sumwt[j]   += pswt;
      sumwtsq[j] += pow2(pswt);

      double normhepmc = xss[j]/na[j];

      // Weighted events with additional number of trial events to consider.
      if ( pythiaPtr[j]->info.lhaStrategy() != 0
        && pythiaPtr[j]->info.lhaStrategy() != 3
        && nash[j] > 0)
        normhepmc = 1. / (1e9*nash[j]);
      // Weighted events.
      else if ( pythiaPtr[j]->info.lhaStrategy() != 0
        && pythiaPtr[j]->info.lhaStrategy() != 3
        && nash[j] == 0)
        normhepmc = 1. / (1e9*na[j]);

      if (pythiaPtr[j]->settings.flag("PhaseSpace:bias2Selection"))
        normhepmc = 1. / (1e9*na[j]);

      if (pythiaPtr[j]->event.size() > 3) {

        double w = evtweight*pswt*normhepmc;
        // Add the weight of the current event to the cross section.
        sigmaTotAll += w;
        sigmaInc[j] += evtweight*normhepmc;
        sigmaTot[j] += w;
#ifdef HEPMC2
        if (printHepMC) {
          // Construct new empty HepMC event.
          HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
          // Set event weight
          hepmcevt->weights().push_back(w);
          // Fill HepMC event
          ToHepMC.fill_next_event( *pythiaPtr[j], hepmcevt );
          // Report cross section to hepmc
          HepMC::GenCrossSection xsec;
          xsec.set_cross_section( sigmaTotAll*1e9,
            pythiaPtr[j]->info.sigmaErr()*1e9 );
          hepmcevt->set_cross_section( xsec );
          // Write the HepMC event to file. Done with it.
          ascii_io << hepmcevt;
          delete hepmcevt;
        }
#endif

        // Write additional HepMC events.
        for (int iwt=0; iwt < int(pswts.size()); ++iwt) {
          double wVar = evtweight*pswts[iwt]*normhepmc;
          // Add the weight of the current event to the cross section.
          sigmaTotVarAll[iwt] += wVar;
          sigmaTotVar[j][iwt] += wVar;
#ifdef HEPMC2
          if (printHepMC) {
            HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
            // Set event weight
            hepmcevt->weights().push_back(wVar);
            // Fill HepMC event
            ToHepMC.fill_next_event( *pythiaPtr[j], hepmcevt );
            // Report cross section to hepmc
            HepMC::GenCrossSection xsec;
            xsec.set_cross_section( sigmaTotVarAll[iwt]*1e9,
              pythiaPtr[j]->info.sigmaErr()*1e9 );
            hepmcevt->set_cross_section( xsec );
            // Write the HepMC event to file. Done with it.
            *ascii_io_var[iwt] << hepmcevt;
            delete hepmcevt;
          }
#endif
        }
      }

    }
  }
#endif

  // print cross section, errors
  cout << "\nSummary of individual runs:\n";
  for (int i = 0; i < int(pythiaPtr.size()); ++i) {
    cout << "\nSummary of runs #" << i << " :\n";
    pythiaPtr[i]->stat();
    int nAc = pythiaPtr[i]->info.nAccepted();
    cout << scientific << setprecision(6)
      << "\n\t Mean shower weight        = " << sumwt[i]/double(nAc) << "\n"
      << "\t Variance of shower weight = "
      << sqrt(1/double(nAc)*(sumwtsq[i] - pow(sumwt[i],2)/double(nAc)))
      << endl;
    cout << "Inclusive cross section    : " << sigmaInc[i] << "\n";
    cout << "Cross section after shower : " << sigmaTot[i] << "\n";
  }
  cout << "\nCombination of runs:\n"
    << scientific << setprecision(6)
    << "\t Minimal shower weight     = " << wmin << "\n"
    << "\t Maximal shower weight     = " << wmax << "\n"
    << "Total cross section after shower: " << sigmaTotAll << "\n";
  
#ifdef HEPMC2
  // Clean-up.
  if ( pythiaPtr.front()->settings.flag("Variations:doVariations") ) { 
    //for (int iwt=0; iwt < dire.weightsPtr->sizeWeights(); ++iwt) {
    for (int iwt=0; iwt < 3; ++iwt) {
      delete ascii_io_var[iwt];
    }
  }
#endif

  // Clean-up.
  for (int i = 0; i < int(pythiaPtr.size()); ++i) {
    delete pythiaPtr[i];
    delete direPtr[i];
#ifdef MG5MES
    delete mergingPtr[i];
    delete hardProcessPtr[i];
    delete mergingHooksPtr[i];
#endif
  }

  // Done.
  return 0;

}
