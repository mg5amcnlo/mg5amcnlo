
// This main program is intended for tutorials only. The goal is to generate a
// single showered and hadronized event, and save this event to an output file 
// that can be converted into a postscript figure using the "graphviz" program.
// This might help visualize the output of event generators. 
// direEvent.cc (C) N. Fischer, 2017

// DIRE includes.
#include "Dire/Dire.h"

// Pythia includes.
#include "Pythia8/Pythia.h"

using namespace Pythia8;

// Funtions to convert a Pythia event to graphviz output.

string nr2st(int nr) {
  return static_cast<ostringstream*>( &(ostringstream() << nr) )->str();
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
    string ident     = "F" + nr2st(10000+i);
    // Name that will appear in graph.
    string label     = nr2st(evt[i].id()) + " (" + evt[i].name() + ")";
    // Find particle group for colors.
    int    stAbs     = evt[i].statusAbs();
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
    // Skip hadrons and decay products for simplified output.
    if (simplifyHadronization && 
      (status == "decays" || status == "hadronization") ) continue;
    // Special treatment of hadronization particles for simplified output.
    if (simplifyHadronization && status == "hadronizationPrep") {
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
      continue;
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
      makeArrow(&arrows, "F"+nr2st(10000+max(mot1,mot2)), ident);
    // Setup arrow to daughter(s).
    vector<int> daus = evt[i].daughterList();
    for (int j=0; j<(int)daus.size(); j++)
      makeArrow(&arrows, ident, "F"+nr2st(10000+daus[j]));
  }

  // Add the hadron groups for simplified output.
  map< pair<string,string>, string > arrowsSav = arrows;
  for (int i=0; i<(int)hadronGroups.size(); i++) {
    // Identifier of that group.
    string ident     = "G" + nr2st(10000+i);
    pair<string,string> colors = colMap["hadronization"];
    string fillcolor = colors.first, fontcolor = colors.second;
    string line      = "  " + ident + " [shape=none,\n     label = <<"
      "table border=\"0\" cellspacing=\"0\">\n";
    for (int j=0; j<(int)hadronGroups[i].size(); j++) {
      // Name that will appear in graph.
      string label = nr2st(evt[hadronGroups[i][j]].id()) + " ("
        + evt[hadronGroups[i][j]].name() + ")";
      line += ( "               <tr><td port=\"port" + nr2st(j)
        + "\" border=\"1\" bgcolor=\"" + fillcolor + "\"><font color=\""
        + fontcolor + "\">" + label + "</font></td></tr>\n" );
    }
    line += "             </table>> ];";
    // Add the group to the graph.
    blobs[ident] = line;
    // Add an arrow from each parent to the group.
    for (int j=0; j<(int)hadronParents[i].size(); j++) {
      // Identifier of that parent.
      int    nrGrandParent1 = evt[hadronParents[i][j]].mother1();
      int    nrGrandParent2 = evt[hadronParents[i][j]].mother2();
      string identParent;
      if (nrGrandParent1 > 0) {
        // Trace back one more generation if double hadronization prep.
        if (evt[nrGrandParent2].statusAbs() > 70) {
          nrGrandParent1 = evt[nrGrandParent2].mother1();
          nrGrandParent2 = evt[nrGrandParent2].mother2();
          if (nrGrandParent1 > 0)
            makeArrow(&arrows, "F"+nr2st(10000+nrGrandParent1), ident);
          if (nrGrandParent2 > 0)
            makeArrow(&arrows, "F"+nr2st(10000+nrGrandParent2), ident);
        } else makeArrow(&arrows, "F"+nr2st(10000+nrGrandParent1), ident);
      }
      if (nrGrandParent2 > 0) {
        // Trace back one more generation if double hadronization prep.
        if (evt[nrGrandParent2].statusAbs() > 70) {
          nrGrandParent1 = evt[nrGrandParent2].mother1();
          nrGrandParent2 = evt[nrGrandParent2].mother2();
          if (nrGrandParent1 > 0)
            makeArrow(&arrows, "F"+nr2st(10000+nrGrandParent1), ident);
          if (nrGrandParent2 > 0)
            makeArrow(&arrows, "F"+nr2st(10000+nrGrandParent2), ident);
        } else makeArrow(&arrows, "F"+nr2st(10000+nrGrandParent2), ident);
      }
      // Erase any parents that might be left in the graph.
      identParent = "F"+nr2st(10000+hadronParents[i][j]);
      blobs.erase(identParent);
      for (map< pair<string,string>, string >::iterator k=arrowsSav.begin();
        k!=arrowsSav.end(); k++) if (k->first.second == identParent) {
        arrows.erase(k->first);
      }
    }
  }

  // Write output.
  ofstream outfile;
  outfile.open((char*)(fileName+".dot").c_str(), ios_base::out);
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
  cout << "You can now produce a .ps figure by using the 'dot' command:\n\n"
       << "dot -Tps " << fileName << ".dot -o " << fileName << ".ps" << "\n\n";
  cout << "Note: 'dot' is part of the 'graphviz' package.\n"
       << "You might want to install this package to produce the .ps event"
       << endl << endl;

}

//==========================================================================

// Pythia+Dire main program to print one event.

int main( int nargs, char* argv[]  ){

  Pythia pythia;
  // Create and initialize DIRE shower plugin.
  Dire dire;
  if (nargs > 1) dire.init(pythia, argv[1]);
  else           dire.init(pythia, "lep.cmnd");
  string outputfile = (nargs > 2) ? string(argv[2]) : "event";

  // Generate and print one event.
  pythia.next();
  printEvent(pythia.event, outputfile);

  // Done
  return 0;

}
