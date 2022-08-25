// Visualisation.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// The following functions analyze a scattering event and save the event in
// an output format that can be converted into a postscript figure using the
// "graphviz" program.
// Author: Nadine Fischer (primary), Stefan Prestel (porting)

#ifndef Pythia8_Visualisation_H
#define Pythia8_Visualisation_H

namespace Pythia8 {

//==========================================================================

// The functions below assist the main function, printEvent, in producing
// an event visualisation, by converting status codes and defining arrows.

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

//--------------------------------------------------------------------------

void makeArrow(map< pair<string,string>, string >* arrows,
  string identParent, string identChild) {
  pair<string,string> key = make_pair(identParent,identChild);
  string value = "  " + identParent + " -> " + identChild
    + " [weight=2,label=\" \"];";
  arrows->insert( pair< pair<string,string>, string>(key, value) );
}

//--------------------------------------------------------------------------

// The main visualisation function. Note that this is really only a schematic
// visualisation. In particular, the space-time structure of the partonic
// evolution and the space-time position of string breakups are only handled
// schematically.

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
    string ident     = "F" + std::to_string(10000+i);
    // Name that will appear in graph.
    string label    = std::to_string(evt[i].id()) + " (" + evt[i].name() + ")";
    // Find particle group for colors.
    string status   = py_status(evt[i].statusAbs());
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
      makeArrow(&arrows, "F"+std::to_string(10000+max(mot1,mot2)), ident);
    // Setup arrow to daughter(s).
    if (!checkDaughters) {
      vector<int> daus = evt[i].daughterList();
      for (int j=0; j<(int)daus.size(); j++)
        makeArrow(&arrows, ident, "F"+std::to_string(10000+daus[j]));
    }
  }

  // Add the hadron groups for simplified output.
  map< pair<string,string>, string > arrowsSav = arrows;
  for (int i=0; i<(int)hadronGroups.size(); i++) {
    // Identifier of that group.
    string ident     = "G" + std::to_string(10000+i);
    pair<string,string> colors = colMap["hadronization"];
    string fillcolor = colors.first, fontcolor = colors.second;
    string line      = "  " + ident + " [shape=none,\n     label = <<"
      "table border=\"0\" cellspacing=\"0\">\n";
    for (int j=0; j<(int)hadronGroups[i].size(); j++) {
      // Name that will appear in graph.
      string label = std::to_string(evt[hadronGroups[i][j]].id()) + " ("
        + evt[hadronGroups[i][j]].name() + ")";
      line += ( "               <tr><td port=\"port" + std::to_string(j)
        + "\" border=\"1\" bgcolor=\"" + fillcolor + "\"><font color=\""
        + fontcolor + "\">" + label + "</font></td></tr>\n" );
    }
    line += "             </table>> ];";
    // Add the group to the graph.
    blobs[ident] = line;
    // Add an arrow from each parent to the group.
    for (int j=0; j<(int)hadronParents[i].size(); j++) {
      // Identifier of that parent.
      string identParent = "F"+std::to_string(10000+hadronParents[i][j]);
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
            toErase.push_back("F"+std::to_string(10000+nrGP1));
            int nrGGP1 = evt[nrGP1].mother1();
            int nrGGP2 = evt[nrGP1].mother2();
            if (nrGGP1 > 0)
              makeArrow(&arrows, "F"+std::to_string(10000+nrGGP1), ident);
            if (nrGGP2 > 0 && nrGGP2 != nrGGP1)
              makeArrow(&arrows, "F"+std::to_string(10000+nrGGP2), ident);
          } else makeArrow(&arrows, "F"+std::to_string(10000+nrGP1), ident);
        }
        if (nrGP2 > 0 && nrGP2 != nrGP1) {
          // Trace back one more generation if double hadronization prep.
          if (py_status(evt[nrGP2].statusAbs()) == "hadronizationPrep") {
            toErase.push_back("F"+std::to_string(10000+nrGP2));
            int nrGGP1 = evt[nrGP2].mother1();
            int nrGGP2 = evt[nrGP2].mother2();
            if (nrGGP1 > 0)
              makeArrow(&arrows, "F"+std::to_string(10000+nrGGP1), ident);
            if (nrGGP2 > 0 && nrGGP2 != nrGGP1)
              makeArrow(&arrows, "F"+std::to_string(10000+nrGGP2), ident);
          } else makeArrow(&arrows, "F"+std::to_string(10000+nrGP2), ident);
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
            << "     <tr><td port=\"0\" border=\"1\"><b>Legend</b></td></tr>"
            << endl;
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
        cout << "Stored event visualization in file " << fileName+".ps"
             << endl;
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

} // end namespace Pythia8

#endif  // end Pythia8_Visualisation_H
