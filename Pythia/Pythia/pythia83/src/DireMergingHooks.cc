// DireMergingHooks.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for Dire merging classes.

#include "Pythia8/PartonLevel.h"
#include "Pythia8/DireMergingHooks.h"

namespace Pythia8 {

//==========================================================================

// The DireHardProcess class.

//--------------------------------------------------------------------------

// Declaration of hard process class
// This class holds information on the desired hard 2->2 process to be merged
// This class is a container class for History class use.

// Initialisation on the process string

void DireHardProcess::initOnProcess( string process,
  ParticleData* particleData) {
  state.init("(hard process)", particleData);
  translateProcessString(process);
}

//--------------------------------------------------------------------------

// Function to translate a string specitying the core process into the
// internal notation
// Currently, the input string has to be in MadEvent notation

void DireHardProcess::translateProcessString( string process){

  vector <int> incom;
  vector <int> inter;
  vector <int> outgo;
  // Particle identifiers, ordered in such a way that e.g. the "u"
  // in a mu is not mistaken for an u quark
  int inParticleNumbers[] = {
        // Leptons
        -11,11,-12,12,-13,13,-14,14,-15,15,-16,16,
        // Jet container
        2212,2212,0,0,0,0,
        // Quarks
        -1,1,-2,2,-3,3,-4,4,-5,5,-6,6};
  string inParticleNamesMG[] =  {
        // Leptons
        "e+","e-","ve~","ve","mu+","mu-","vm~","vm","ta+","ta-","vt~","vt",
        // Jet container
        "p~","p","l+","l-","vl~","vl",
        // Quarks
        "d~","d","u~","u","s~","s","c~","c","b~","b","t~","t"};

  // Declare intermediate particle identifiers
  int interParticleNumbers[] = {
         // Electroweak gauge bosons
         22,23,-24,24,25,2400,
         // All squarks
        -1000001,1000001,-1000002,1000002,-1000003,1000003,-1000004,1000004,
        -1000005,1000005,-1000006,1000006,-2000001,2000001,-2000002,2000002,
        -2000003,2000003,-2000004,2000004,-2000005,2000005,-2000006,2000006,
         // Top quarks
         -6,6,
         // Dummy index as back-up
         0};
  // Declare names of intermediate particles
  string interParticleNamesMG[] = {
        // Electroweak gauge bosons
        "a","z","w-","w+","h","W",
         // All squarks
         "dl~","dl","ul~","ul","sl~","sl","cl~","cl","b1~","b1","t1~","t1",
         "dr~","dr","ur~","ur","sr~","sr","cr~","cr","b2~","b2","t2~","t2",
         // Top quarks
         "t~","t",
        // Dummy index as back-up
        "xx"};

  // Declare final state particle identifiers
  int outParticleNumbers[] = {
        // Leptons
        -11,11,-12,12,-13,13,-14,14,-15,15,-16,16,
        // Jet container and lepton containers
        2212,2212,0,0,0,0,1200,1100,5000,
        // Containers for inclusive handling for weak bosons and jets
         10000022,10000023,10000024,10000025,10002212,
        // All squarks
        -1000001,1000001,-1000002,1000002,-1000003,1000003,-1000004,1000004,
        -1000005,1000005,-1000006,1000006,-2000001,2000001,-2000002,2000002,
        -2000003,2000003,-2000004,2000004,-2000005,2000005,-2000006,2000006,
        // Quarks
        -1,1,-2,2,-3,3,-4,4,-5,5,-6,6,
        // SM uncoloured bosons
        22,23,-24,24,25,2400,
        // Neutralino in SUSY
        1000022};
  // Declare names of final state particles
  string outParticleNamesMG[] =  {
        // Leptons
        "e+","e-","ve~","ve","mu+","mu-","vm~","vm","ta+","ta-","vt~","vt",
        // Jet container and lepton containers
        "j~","j","l+","l-","vl~","vl","NEUTRINOS","LEPTONS","BQUARKS",
        // Containers for inclusive handling for weak bosons and jets
        "Ainc","Zinc","Winc", "Hinc", "Jinc",
        // All squarks
        "dl~","dl","ul~","ul","sl~","sl","cl~","cl","b1~","b1","t1~","t1",
        "dr~","dr","ur~","ur","sr~","sr","cr~","cr","b2~","b2","t2~","t2",
        // Quarks
        "d~","d","u~","u","s~","s","c~","c","b~","b","t~","t",
        // SM uncoloured bosons
        "a","z","w-","w+","h","W",
        // Neutralino in SUSY
        "n1"};

  // Declare size of particle name arrays
  int nIn   = 30;
  int nInt  = 33;
  int nOut  = 69;

  // Start mapping user-defined particles onto particle ids.
  string fullProc = process;

  // Find user-defined hard process content
  // Count number of user particles
  int nUserParticles = 0;
  for(int n = fullProc.find("{", 0); n != int(string::npos);
          n = fullProc.find("{", n)) {
    nUserParticles++;
    n++;
  }
  // Cut user-defined particles from remaining process
  vector <string> userParticleStrings;
  for(int i =0; i < nUserParticles;++i) {
    int n = fullProc.find("{", 0);
    userParticleStrings.push_back(fullProc.substr(0,n));
    fullProc = fullProc.substr(n+1,fullProc.size());
  }
  // Cut remaining process string from rest
  if (nUserParticles > 0)
    userParticleStrings.push_back(
      fullProc.substr( 0, fullProc.find("}",0) ) );
  // Remove curly brackets and whitespace
  for(int i =0; i < int(userParticleStrings.size());++i) {
    while(userParticleStrings[i].find("{", 0) != string::npos)
      userParticleStrings[i].erase(userParticleStrings[i].begin()
                                  +userParticleStrings[i].find("{", 0));
    while(userParticleStrings[i].find("}", 0) != string::npos)
      userParticleStrings[i].erase(userParticleStrings[i].begin()
                                  +userParticleStrings[i].find("}", 0));
    while(userParticleStrings[i].find(" ", 0) != string::npos)
      userParticleStrings[i].erase(userParticleStrings[i].begin()
                                  +userParticleStrings[i].find(" ", 0));
  }

  // Convert particle numbers in user particle to integers
  vector<int>userParticleNumbers;
  if ( int(userParticleStrings.size()) > 1) {
    for( int i = 1; i < int(userParticleStrings.size()); ++i) {
      userParticleNumbers.push_back(
        atoi((char*)userParticleStrings[i].substr(
          userParticleStrings[i].find(",",0)+1,
          userParticleStrings[i].size()).c_str() ) );
    }
  }
  // Save remaining process string
  if (nUserParticles > 0)
    userParticleStrings.push_back(
      fullProc.substr(
        fullProc.find("}",0)+1, fullProc.size() ) );
  // Remove curly brackets and whitespace
  for( int i = 0; i < int(userParticleStrings.size()); ++i) {
    while(userParticleStrings[i].find("{", 0) != string::npos)
      userParticleStrings[i].erase(userParticleStrings[i].begin()
                                  +userParticleStrings[i].find("{", 0));
    while(userParticleStrings[i].find("}", 0) != string::npos)
      userParticleStrings[i].erase(userParticleStrings[i].begin()
                                  +userParticleStrings[i].find("}", 0));
    while(userParticleStrings[i].find(" ", 0) != string::npos)
      userParticleStrings[i].erase(userParticleStrings[i].begin()
                                  +userParticleStrings[i].find(" ", 0));
  }

  // Start mapping residual process string onto particle IDs.
  // Declare leftover process after user-defined particles have been converted
  string residualProc;
  if ( int(userParticleStrings.size()) > 1 )
    residualProc = userParticleStrings.front() + userParticleStrings.back();
  else
    residualProc = fullProc;

  // Remove comma separation
  while(residualProc.find(",", 0) != string::npos)
    residualProc.erase(residualProc.begin()+residualProc.find(",",0));

  // Count number of resonances
  int appearances = 0;
  for(int n = residualProc.find("(", 0); n != int(string::npos);
          n = residualProc.find("(", n)) {
    appearances++;
    n++;
  }

  // Cut string in incoming, resonance+decay and outgoing pieces
  vector <string> pieces;
  for(int i =0; i < appearances;++i) {
    int n = residualProc.find("(", 0);
    pieces.push_back(residualProc.substr(0,n));
    residualProc = residualProc.substr(n+1,residualProc.size());
  }
  // Cut last resonance from rest
  if (appearances > 0) {
    pieces.push_back( residualProc.substr(0,residualProc.find(")",0)) );
    pieces.push_back( residualProc.substr(
      residualProc.find(")",0)+1, residualProc.size()) );
  }

  // If the string was not cut into pieces, i.e. no resonance was
  // required, cut string using '>' as delimiter
  if (pieces.empty() ){
    appearances = 0;
    for(int n = residualProc.find(">", 0); n != int(string::npos);
            n = residualProc.find(">", n)) {
      appearances++;
      n++;
    }

    // Cut string in incoming and outgoing pieces
    for(int i =0; i < appearances;++i) {
      int n = residualProc.find(">", 0);
      pieces.push_back(residualProc.substr(0,n));
      residualProc = residualProc.substr(n+1,residualProc.size());
    }

    if (appearances == 1) pieces.push_back(residualProc);
    if (appearances > 1) {
      pieces.push_back( residualProc.substr(0,residualProc.find(">",0)) );
      pieces.push_back( residualProc.substr(
        residualProc.find(">",0)+1, residualProc.size()) );
    }
  }

  // Get incoming particles
  for(int i=0; i < nIn; ++i) {
    for(int n = pieces[0].find(inParticleNamesMG[i], 0);
           n != int(string::npos);
           n = pieces[0].find(inParticleNamesMG[i], n)) {
      incom.push_back(inParticleNumbers[i]);
      pieces[0].erase(pieces[0].begin()+n,
                      pieces[0].begin()+n+inParticleNamesMG[i].size());
      n=0;
    }
  }

  // Check intermediate resonances and decay products
  for(int i =1; i < int(pieces.size()); ++i){
    // Seperate strings into intermediate and outgoing, if not already done
    int k = pieces[i].find(">", 0);

    string intermediate = (pieces[i].find(">", 0) != string::npos) ?
                           pieces[i].substr(0,k) : "";
    string outgoing = (pieces[i].find(">", 0) != string::npos) ?
                       pieces[i].substr(k+1,pieces[i].size()) : pieces[i];

    // Get intermediate particles
    for(int j=0; j < nInt; ++j) {
      for(int n = intermediate.find(interParticleNamesMG[j], 0);
             n != int(string::npos);
             n = intermediate.find(interParticleNamesMG[j], n)) {
        inter.push_back(interParticleNumbers[j]);
        intermediate.erase(intermediate.begin()+n,
                    intermediate.begin()+n+interParticleNamesMG[j].size());
        n=0;
      }
    }

    // Get outgoing particles
    for(int j=0; j < nOut; ++j) {
      for(int n = outgoing.find(outParticleNamesMG[j], 0);
             n != int(string::npos);
             n = outgoing.find(outParticleNamesMG[j], n)) {
        outgo.push_back(outParticleNumbers[j]);
        outgoing.erase(outgoing.begin()+n,
                       outgoing.begin()+n+outParticleNamesMG[j].size());
        n=0;
      }
    }

    // For arbitrary or non-existing intermediate, remember zero for each
    // two outgoing particles, without bosons.
    if (inter.empty()) {

      // For final state bosons, bookkeep the final state boson as
      // intermediate as well.
      int nBosons = 0;
      for(int l=0; l < int(outgo.size()); ++l)
        if ( (abs(outgo[l]) > 20 && abs(outgo[l]) <= 25) || outgo[l] == 2400)
          nBosons++;

      int nZeros = (outgo.size() - nBosons)/2;
      for(int l=0; l < nZeros; ++l)
        inter.push_back(0);
    }

    // For final state bosons, bookkeep the final state boson as
    // intermediate as well.
    for(int l=0; l < int(outgo.size()); ++l)
      if ( (abs(outgo[l]) > 20 && abs(outgo[l]) <= 25) || outgo[l] == 2400)
        inter.push_back(outgo[l]);

  }

  // Now store incoming, intermediate and outgoing
  // Set intermediate tags
  for(int i=0; i < int(inter.size()); ++i)
    hardIntermediate.push_back(inter[i]);

  // Set the incoming particle tags
  if (incom.size() != 2)
    cout << "Only two incoming particles allowed" << endl;
  else {
    hardIncoming1 = incom[0];
    hardIncoming2 = incom[1];
  }

  // Now store final particle identifiers
  // Start with user-defined particles.
  for( int i = 0; i < int(userParticleNumbers.size()); ++i)
    if (userParticleNumbers[i] > 0) {
      hardOutgoing2.push_back( userParticleNumbers[i]);
      hardIntermediate.push_back(0);
      // For non-existing intermediate, remember zero.
    } else if (userParticleNumbers[i] < 0) {
      hardOutgoing1.push_back( userParticleNumbers[i]);
      // For non-existing intermediate, remember zero.
      hardIntermediate.push_back(0);
    }

  // Push back particles / antiparticles
  for(int i=0; i < int(outgo.size()); ++i)
    if (outgo[i] > 0
      && outgo[i] != 2212
      && outgo[i] != 5000
      && outgo[i] != 1100
      && outgo[i] != 1200
      && outgo[i] != 2400
      && outgo[i] != 1000022
      && outgo[i] < 10000000)
      hardOutgoing2.push_back( outgo[i]);
    else if (outgo[i] < 0)
      hardOutgoing1.push_back( outgo[i]);

  // Save final state W-boson container as particle
  for(int i=0; i < int(outgo.size()); ++i)
    if ( outgo[i] == 2400)
      hardOutgoing2.push_back( outgo[i]);

  // Push back jets, distribute evenly among particles / antiparticles
  // Push back majorana particles, distribute evenly
  int iNow = 0;
  for(int i=0; i < int(outgo.size()); ++i)
    if ( (outgo[i] == 2212
      || outgo[i] == 5000
      || outgo[i] == 1200
      || outgo[i] == 1000022
      || outgo[i] > 10000000)
      && iNow%2 == 0 ){
      hardOutgoing2.push_back( outgo[i]);
      iNow++;
    } else if ( (outgo[i] == 2212
             || outgo[i] == 5000
             || outgo[i] == 1100
             || outgo[i] == 1000022
             || outgo[i] > 10000000)
             && iNow%2 == 1 ){
      hardOutgoing1.push_back( outgo[i]);
      iNow++;
    }

  // Done
}

//--------------------------------------------------------------------------

// Function to check if the candidates stored in Pos1 and Pos2, together with
// a proposed candidate iPos are allowed.

bool DireHardProcess::allowCandidates(int iPos, vector<int> Pos1,
  vector<int> Pos2, const Event& event){

  bool allowed = true;

  // Find colour-partner of new candidate
  int type = (event[iPos].col() > 0) ? 1 : (event[iPos].acol() > 0) ? -1 : 0;

  if (type == 0) return true;

  if (type == 1){
    int col = event[iPos].col();
    int iPartner = 0;
    for(int i=0; i < int(event.size()); ++i)
      if ( i != iPos
        && (( event[i].isFinal() && event[i].acol() == col)
          ||( event[i].status() == -21 && event[i].col() == col) ))
      iPartner = i;

    vector<int> partners;
    for(int i=0; i < int(event.size()); ++i)
      for(int j=0; j < int(Pos1.size()); ++j)
        if ( Pos1[j] != 0 && i != Pos1[j] && event[Pos1[j]].colType() != 0
        && (( event[i].isFinal() && event[i].col() == event[Pos1[j]].acol())
          ||( event[i].status() == -21
           && event[i].acol() == event[Pos1[j]].acol()) ))
         partners.push_back(i);

    // Never allow equal initial partners!
    if (event[iPartner].status() == -21){
      for(int i=0; i < int(partners.size()); ++i)
        if ( partners[i] == iPartner)
          allowed = false;
    }

  } else {
    int col = event[iPos].acol();
    int iPartner = 0;
    for(int i=0; i < int(event.size()); ++i)
      if ( i != iPos
        && (( event[i].isFinal() && event[i].col()  == col)
          ||(!event[i].isFinal() && event[i].acol() == col) ))
      iPartner = i;

    vector<int> partners;
    for(int i=0; i < int(event.size()); ++i)
      for(int j=0; j < int(Pos2.size()); ++j)
        if ( Pos2[j] != 0 && i != Pos2[j] && event[Pos2[j]].colType() != 0
        && (( event[i].isFinal() && event[i].acol() == event[Pos2[j]].col())
          ||( event[i].status() == -21
           && event[i].col() == event[Pos2[j]].col()) ))
         partners.push_back(i);

    // Never allow equal initial partners!
    if (event[iPartner].status() == -21){
      for(int i=0; i < int(partners.size()); ++i){
        if ( partners[i] == iPartner)
          allowed = false;
      }
    }

  }

  return allowed;

}

//--------------------------------------------------------------------------

// Function to identify the hard subprocess in the current event

void DireHardProcess::storeCandidates( const Event& event, string process){

  // Store the reference event
  state.clear();
  state = event;

  // Local copy of intermediate bosons
  vector<int> intermediates;
  for(int i =0; i < int(hardIntermediate.size());++i)
    intermediates.push_back( hardIntermediate[i]);

  // Local copy of outpoing partons
  vector<int> outgoing1;
  for(int i =0; i < int(hardOutgoing1.size());++i)
    outgoing1.push_back( hardOutgoing1[i]);
  vector<int> outgoing2;
  for(int i =0; i < int(hardOutgoing2.size());++i)
    outgoing2.push_back( hardOutgoing2[i]);

  // Clear positions of intermediate and outgoing particles
  PosIntermediate.resize(0);
  PosOutgoing1.resize(0);
  PosOutgoing2.resize(0);
  for(int i =0; i < int(hardIntermediate.size());++i)
    PosIntermediate.push_back(0);
  for(int i =0; i < int(hardOutgoing1.size());++i)
    PosOutgoing1.push_back(0);
  for(int i =0; i < int(hardOutgoing2.size());++i)
    PosOutgoing2.push_back(0);

  // For QCD dijet or e+e- > jets hard process, do not store any candidates,
  // as to not discriminate clusterings
  if (  process.compare("pp>jj") == 0
    || process.compare("e+e->jj") == 0
    || process.compare("e+e->(z>jj)") == 0 ){
    for(int i =0; i < int(hardOutgoing1.size());++i)
      PosOutgoing1[i] = 0;
    for(int i =0; i < int(hardOutgoing2.size());++i)
      PosOutgoing2[i] = 0;
    // Done
    return;
  }

  // For inclusive merging, do not store any candidates,
  // as to not discriminate clusterings
  bool isInclusive = true;
  for(int i =0; i < int(hardOutgoing1.size());++i)
    if (hardOutgoing1[i] < 10000000) isInclusive = false;
  for(int i =0; i < int(hardOutgoing2.size());++i)
    if (hardOutgoing2[i] < 10000000) isInclusive = false;
  if ( isInclusive ){
    for(int i =0; i < int(hardOutgoing1.size());++i)
      PosOutgoing1[i] = 0;
    for(int i =0; i < int(hardOutgoing2.size());++i)
      PosOutgoing2[i] = 0;
    // Done
    return;
  }

  // Initialise vector of particles that were already identified as
  // hard process particles
  vector<int> iPosChecked;

  // If the hard process is specified only by containers, then add all
  // particles matching with the containers to the hard process.
  bool hasOnlyContainers = true;
  for(int i =0; i < int(hardOutgoing1.size());++i)
    if (  hardOutgoing1[i] != 1100
      && hardOutgoing1[i] != 1200
      && hardOutgoing1[i] != 5000)
      hasOnlyContainers = false;
  for(int i =0; i < int(hardOutgoing2.size());++i)
    if (  hardOutgoing2[i] != 1100
      && hardOutgoing2[i] != 1200
      && hardOutgoing2[i] != 5000)
      hasOnlyContainers = false;

  if (hasOnlyContainers){

    PosOutgoing1.resize(0);
    PosOutgoing2.resize(0);

    // Try to find all unmatched hard process leptons.
    // Loop through event to find outgoing lepton
    for(int i=0; i < int(event.size()); ++i){

      // Skip non-final particles
      if ( !event[i].isFinal() ) continue;

      // Skip all particles that have already been identified
      bool skip = false;
      for(int k=0; k < int(iPosChecked.size()); ++k){
        if (i == iPosChecked[k])
          skip = true;
      }
      if (skip) continue;

      for(int j=0; j < int(outgoing2.size()); ++j){

        // If the particle matches an outgoing neutrino, save it
        if ( outgoing2[j] == 1100
          && ( event[i].idAbs() == 11
            || event[i].idAbs() == 13
            || event[i].idAbs() == 15) ){
          PosOutgoing2.push_back(i);
          iPosChecked.push_back(i);
        }

        // If the particle matches an outgoing lepton, save it
        if ( outgoing2[j] == 1200
          && ( event[i].idAbs() == 12
            || event[i].idAbs() == 14
            || event[i].idAbs() == 16) ){
          PosOutgoing2.push_back(i);
          iPosChecked.push_back(i);
        }

        // If the particle matches an outgoing b-quark, save it
        if ( outgoing2[j] == 5000 && event[i].idAbs() == 5 ){
          PosOutgoing2.push_back(i);
          iPosChecked.push_back(i);
        }

      }

      // Skip all particles that have already been identified
      skip = false;
      for(int k=0; k < int(iPosChecked.size()); ++k){
        if (i == iPosChecked[k])
          skip = true;
      }
      if (skip) continue;

      for(int j=0; j < int(outgoing1.size()); ++j){

        // If the particle matches an outgoing neutrino, save it
        if ( outgoing1[j] == 1100
          && ( event[i].idAbs() == 11
            || event[i].idAbs() == 13
            || event[i].idAbs() == 15) ){
          PosOutgoing1.push_back(i);
          iPosChecked.push_back(i);
        }

        // If the particle matches an outgoing lepton, save it
        if ( outgoing1[j] == 1200
          && ( event[i].idAbs() == 12
            || event[i].idAbs() == 14
            || event[i].idAbs() == 16) ){
          PosOutgoing1.push_back(i);
          iPosChecked.push_back(i);
        }

        // If the particle matches an outgoing b-quark, save it
        if ( outgoing1[j] == 5000 && event[i].idAbs() == 5 ){
          PosOutgoing1.push_back(i);
          iPosChecked.push_back(i);
        }

      }
    }

    // Done
    return;
  }

  // Now begin finding candidates when not only containers are used.

  // First try to find final state bosons
  for(int i=0; i < int(intermediates.size()); ++i){

    // Do nothing if the intermediate boson is absent
    if (intermediates[i] == 0) continue;

    // Do nothing if this boson does not match any final state boson
    bool matchesFinalBoson = false;
    for(int j =0; j< int(outgoing1.size()); ++j){
      if ( intermediates[i] == outgoing1[j] )
        matchesFinalBoson = true;
    }
    for(int j =0; j< int(outgoing2.size()); ++j){
      if ( intermediates[i] == outgoing2[j] )
        matchesFinalBoson = true;
    }
    if (!matchesFinalBoson) continue;

    // Loop through event
    for(int j=0; j < int(event.size()); ++j) {

      // Skip all particles that have already been identified
      bool skip = false;
      for(int m=0; m < int(iPosChecked.size()); ++m)
        if (j == iPosChecked[m]) skip = true;
      if (skip) continue;

      // If the particle has a requested intermediate id, check if
      // if is a final state boson
      if ( (event[j].id() == intermediates[i])
        ||(event[j].idAbs() == 24 && intermediates[i] == 2400) ) {

        PosIntermediate[i] = j;
        intermediates[i] = 0;
        // Be careful only to replace one index at a time!
        bool indexSet = false;

        for(int k=0; k < int(outgoing1.size()); ++k) {
          if (event[j].id() == outgoing1[k] && !indexSet){
            PosOutgoing1[k] = j;
            outgoing1[k] = 99;
            indexSet = true;
          }
        }

        for(int k=0; k < int(outgoing2.size()); ++k) {
          if (event[j].id() == outgoing2[k] && !indexSet){
            PosOutgoing2[k] = j;
            outgoing2[k] = 99;
            indexSet = true;
          }
        }

        // Check for W-boson container
        for(int k=0; k < int(outgoing2.size()); ++k) {
          if (event[j].idAbs() == 24 && outgoing2[k] == 2400 && !indexSet ){
            PosOutgoing2[k] = j;
            outgoing2[k] = 99;
            indexSet = true;
          }
        }

        iPosChecked.push_back(j);

      }
    }
  }

  // Second try to find particles coupled to intermediate bosons
  for(int i=0; i < int(intermediates.size()); ++i){

    // Do nothing if the intermediate boson is absent
    if (intermediates[i] == 0) continue;

    // Loop through event
    for(int j=0; j < int(event.size()); ++j) {
      // If the particle has a requested intermediate id, check if
      // daughters are hard process particles
      if ( (event[j].id() == intermediates[i])
        ||(event[j].idAbs() == 24 && intermediates[i] == 2400) ) {
        // If this particle is a potential intermediate
        PosIntermediate[i] = j;
        intermediates[i] = 0;
        // If id's of daughters are good, store position
        int iPos1 = event[j].daughter1();
        int iPos2 = event[j].daughter2();

        // Loop through daughters to check if these contain some hard
        // outgoing particles
        for( int k=iPos1; k <= iPos2; ++k){
          int id = event[k].id();

          // Skip all particles that have already been identified
          bool skip = false;
          for(int m=0; m < int(iPosChecked.size()); ++m)
            if (k == iPosChecked[m]) skip = true;
          if (skip) continue;

          // Check if daughter is hard outgoing particle
          for(int l=0; l < int(outgoing2.size()); ++l)
            if ( outgoing2[l] != 99 ){
                // Found particle id
              if (id == outgoing2[l]
                // Found jet
                || (id > 0 && abs(id) < 10 && outgoing2[l] == 2212) ){
                // Store position
                PosOutgoing2[l] = k;
                // Remove the matched particle from the list
                outgoing2[l] = 99;
                iPosChecked.push_back(k);
                break;
              }

            }

          // Check if daughter is hard outgoing antiparticle
          for(int l=0; l < int(outgoing1.size()); ++l)
            if ( outgoing1[l] != 99 ){
                // Found particle id
              if (id == outgoing1[l]
                // Found jet
                || (id < 0 && abs(id) < 10 && outgoing1[l] == 2212) ){
                // Store position
                PosOutgoing1[l] = k;
                // Remove the matched particle from the list
                outgoing1[l] = 99;
                iPosChecked.push_back(k);
                break;
            }

          }

        } // End loop through daughters
      } // End if ids match
    } // End loop through event
  } // End loop though requested intermediates

  // If all outgoing particles were found, done
  bool done = true;
  for(int i=0; i < int(outgoing1.size()); ++i)
    if (outgoing1[i] != 99)
      done = false;
  for(int i=0; i < int(outgoing2.size()); ++i)
    if (outgoing2[i] != 99)
      done = false;
  // Return
  if (done) return;

  // Leptons not associated with resonance are allowed.
  // Try to find all unmatched hard process leptons.
  // Loop through event to find outgoing lepton
  for(int i=0; i < int(event.size()); ++i){
    // Skip non-final particles and final partons
    if ( !event[i].isFinal() || event[i].colType() != 0)
      continue;
    // Skip all particles that have already been identified
    bool skip = false;
    for(int k=0; k < int(iPosChecked.size()); ++k){
      if (i == iPosChecked[k])
        skip = true;
    }
    if (skip) continue;

    // Check if any hard outgoing leptons remain
    for(int j=0; j < int(outgoing2.size()); ++j){
      // Do nothing if this particle has already be found,
      // or if this particle is a jet or quark
      if (  outgoing2[j] == 99
        || outgoing2[j] == 2212
        || abs(outgoing2[j]) < 10)
        continue;

      // If the particle matches an outgoing lepton, save it
      if (  event[i].id() == outgoing2[j] ){
        PosOutgoing2[j] = i;
        outgoing2[j] = 99;
        iPosChecked.push_back(i);
      }
    }

    // Check if any hard outgoing antileptons remain
    for(int j=0; j < int(outgoing1.size()); ++j){
      // Do nothing if this particle has already be found,
      // or if this particle is a jet or quark
      if (  outgoing1[j] == 99
        || outgoing1[j] == 2212
        || abs(outgoing1[j]) < 10)
        continue;

      // If the particle matches an outgoing lepton, save it
      if (event[i].id() == outgoing1[j] ){
        PosOutgoing1[j] = i;
        outgoing1[j] = 99;
        iPosChecked.push_back(i);
      }
    }
  }

  multimap<int,int> out2copy;
  for(int i=0; i < int(event.size()); ++i)
    for(int j=0; j < int(outgoing2.size()); ++j)
      // Do nothing if this particle has already be found,
      // or if this particle is a jet.
      if ( outgoing2[j] != 99
        && outgoing2[j] != 2212
        && ( abs(outgoing2[j]) < 10
          || (abs(outgoing2[j]) > 1000000 && abs(outgoing2[j]) < 1000010)
          || (abs(outgoing2[j]) > 2000000 && abs(outgoing2[j]) < 2000010)
          || abs(outgoing2[j]) == 1000021 )
        && event[i].isFinal()
        && event[i].id() == outgoing2[j] ){
        out2copy.insert(make_pair(j, i));
      }

  multimap<int,int> out1copy;
  for(int i=0; i < int(event.size()); ++i)
    for(int j=0; j < int(outgoing1.size()); ++j)
      // Do nothing if this particle has already be found,
      // or if this particle is a jet.
      if ( outgoing1[j] != 99
        && outgoing1[j] != 2212
        && ( abs(outgoing1[j]) < 10
          || (abs(outgoing1[j]) > 1000000 && abs(outgoing1[j]) < 1000010)
          || (abs(outgoing1[j]) > 2000000 && abs(outgoing1[j]) < 2000010)
          || abs(outgoing1[j]) == 1000021 )
        && event[i].isFinal()
        && event[i].id() == outgoing1[j] ){
        out1copy.insert(make_pair(j, i));
      }

  if ( out1copy.size() >  out2copy.size()){

    // In case the index of the multimap is filled twice, make sure not to
    // arbitrarily overwrite set values.
    vector<int> indexWasSet;
    for ( multimap<int, int>::iterator it = out2copy.begin();
      it != out2copy.end(); ++it ) {
      if ( allowCandidates(it->second, PosOutgoing1, PosOutgoing2, event) ){

        // Skip all particles that have already been identified
        bool skip = false;
        for(int k=0; k < int(iPosChecked.size()); ++k)
          if (it->second == iPosChecked[k]) skip = true;
        // Skip all indices that have already been identified
        for(int k=0; k < int(indexWasSet.size()); ++k)
          if (it->first == indexWasSet[k]) skip = true;
        if (skip) continue;

        // Save parton
        PosOutgoing2[it->first] = it->second;
        // remove entry form lists
        outgoing2[it->first] = 99;
        iPosChecked.push_back(it->second);
        indexWasSet.push_back(it->first);
      }
    }

    indexWasSet.resize(0);
    for ( multimap<int, int>::iterator it = out1copy.begin();
      it != out1copy.end(); ++it ) {
      if ( allowCandidates(it->second, PosOutgoing1, PosOutgoing2, event) ){

        // Skip all particles that have already been identified
        bool skip = false;
        for(int k=0; k < int(iPosChecked.size()); ++k)
          if (it->second == iPosChecked[k]) skip = true;
        // Skip all indices that have already been identified
        for(int k=0; k < int(indexWasSet.size()); ++k)
          if (it->first == indexWasSet[k]) skip = true;
        if (skip) continue;

        // Save parton
        PosOutgoing1[it->first] = it->second;
        // remove entry form lists
        outgoing1[it->first] = 99;
        iPosChecked.push_back(it->second);
        indexWasSet.push_back(it->first);
      }
    }

  } else {

    // In case the index of the multimap is filled twice, make sure not to
    // arbitraryly overwrite set values.
    vector<int> indexWasSet;
    for ( multimap<int, int>::iterator it = out1copy.begin();
      it != out1copy.end(); ++it ) {
      if ( allowCandidates(it->second, PosOutgoing1, PosOutgoing2, event) ){

        // Skip all particles that have already been identified
        bool skip = false;
        for(int k=0; k < int(iPosChecked.size()); ++k)
          if (it->second == iPosChecked[k]) skip = true;
        // Skip all indices that have already been identified
        for(int k=0; k < int(indexWasSet.size()); ++k)
          if (it->first == indexWasSet[k]) skip = true;
        if (skip) continue;

        // Save parton
        PosOutgoing1[it->first] = it->second;
        // remove entry form lists
        outgoing1[it->first] = 99;
        iPosChecked.push_back(it->second);
        indexWasSet.push_back(it->first);
      }
    }

    indexWasSet.resize(0);
    for ( multimap<int, int>::iterator it = out2copy.begin();
      it != out2copy.end(); ++it ) {
      if ( allowCandidates(it->second, PosOutgoing1, PosOutgoing2, event) ){

        // Skip all particles that have already been identified
        bool skip = false;
        for(int k=0; k < int(iPosChecked.size()); ++k)
          if (it->second == iPosChecked[k]) skip = true;
        // Skip all indices that have already been identified
        for(int k=0; k < int(indexWasSet.size()); ++k)
          if (it->first == indexWasSet[k]) skip = true;
        if (skip) continue;

        // Save parton
        PosOutgoing2[it->first] = it->second;
        // remove entry form lists
        outgoing2[it->first] = 99;
        iPosChecked.push_back(it->second);
        indexWasSet.push_back(it->first);
      }
    }
  }

  // It sometimes happens that MadEvent does not put a
  // heavy coloured resonance into the LHE file, even if requested.
  // This means that the decay products of this resonance need to be
  // found separately.
  // Loop through event to find hard process (anti)quarks
  for(int i=0; i < int(event.size()); ++i){

    // Skip non-final particles and final partons
    if ( !event[i].isFinal() || event[i].colType() == 0)
      continue;

    // Skip all particles that have already been identified
    bool skip = false;
    for(int k=0; k < int(iPosChecked.size()); ++k){
      if (i == iPosChecked[k])
        skip = true;
    }
    if (skip) continue;

    // Check if any hard outgoing quarks remain
    for(int j=0; j < int(outgoing2.size()); ++j){
      // Do nothing if this particle has already be found,
      // or if this particle is a jet, lepton container or lepton

      if (  outgoing2[j] == 99
        || outgoing2[j] == 2212
        || (abs(outgoing2[j]) > 10 && abs(outgoing2[j]) < 20)
        || outgoing2[j] == 1100
        || outgoing2[j] == 1200
        || outgoing2[j] == 2400 )
        continue;

      // If the particle matches an outgoing quark, save it
      if (event[i].id() == outgoing2[j]){
        // Save parton
        PosOutgoing2[j] = i;
        // remove entry form lists
        outgoing2[j] = 99;
        iPosChecked.push_back(i);
        break;
      }
    }

    // Check if any hard outgoing antiquarks remain
    for(int j=0; j < int(outgoing1.size()); ++j){
      // Do nothing if this particle has already be found,
      // or if this particle is a jet, lepton container or lepton
      if (  outgoing1[j] == 99
        || outgoing1[j] == 2212
        || (abs(outgoing1[j]) > 10 && abs(outgoing1[j]) < 20)
        || outgoing1[j] == 1100
        || outgoing1[j] == 1200
        || outgoing1[j] == 2400 )
        continue;
      // If the particle matches an outgoing antiquark, save it
      if (event[i].id() == outgoing1[j]){
        // Save parton
        PosOutgoing1[j] = i;
        // Remove parton from list
        outgoing1[j] = 99;
        iPosChecked.push_back(i);
        break;
      }
    }
  }

  // Done
}

//--------------------------------------------------------------------------

// Function to check if the particle event[iPos] matches any of
// the stored outgoing particles of the hard subprocess

bool DireHardProcess::matchesAnyOutgoing(int iPos, const Event& event){

  // Match quantum numbers of any first outgoing candidate
  bool matchQN1 = false;
  // Match quantum numbers of any second outgoing candidate
  bool matchQN2 = false;
  // Match parton in the hard process,
  // or parton from decay of electroweak boson in hard process,
  // or parton from decay of electroweak boson from decay of top
  bool matchHP = false;

  // Check outgoing candidates
  for(int i=0; i < int(PosOutgoing1.size()); ++i)
    // Compare particle properties
    if ( event[iPos].id()         == state[PosOutgoing1[i]].id()
     && event[iPos].colType()    == state[PosOutgoing1[i]].colType()
     && event[iPos].chargeType() == state[PosOutgoing1[i]].chargeType()
     && ( ( event[iPos].col() > 0
         && event[iPos].col() == state[PosOutgoing1[i]].col())
       || ( event[iPos].acol() > 0
         && event[iPos].acol() == state[PosOutgoing1[i]].acol()))
     && event[iPos].charge()     == state[PosOutgoing1[i]].charge() )
      matchQN1 = true;

  // Check outgoing candidates
  for(int i=0; i < int(PosOutgoing2.size()); ++i)
    // Compare particle properties
    if ( event[iPos].id()         == state[PosOutgoing2[i]].id()
     && event[iPos].colType()    == state[PosOutgoing2[i]].colType()
     && event[iPos].chargeType() == state[PosOutgoing2[i]].chargeType()
     && ( ( event[iPos].col() > 0
         && event[iPos].col() == state[PosOutgoing2[i]].col())
       || ( event[iPos].acol() > 0
         && event[iPos].acol() == state[PosOutgoing2[i]].acol()))
     && event[iPos].charge()     == state[PosOutgoing2[i]].charge() )
      matchQN2 = true;

  // Check if maps to hard process:
  // Check that particle is in hard process
  if ( event[iPos].mother1()*event[iPos].mother2() == 12
      // Or particle has taken recoil from first splitting
      || (  event[iPos].status() == 44
         && event[event[iPos].mother1()].mother1()
           *event[event[iPos].mother1()].mother2() == 12 )
      || (  event[iPos].status() == 48
         && event[event[iPos].mother1()].mother1()
           *event[event[iPos].mother1()].mother2() == 12 )
      // Or particle has on-shell resonace as mother
      || (  event[iPos].status() == 23
         && event[event[iPos].mother1()].mother1()
           *event[event[iPos].mother1()].mother2() == 12 )
      // Or particle has on-shell resonace as mother,
      // which again has and on-shell resonance as mother
      || (  event[iPos].status() == 23
         && event[event[iPos].mother1()].status() == -22
         && event[event[event[iPos].mother1()].mother1()].status() == -22
         && event[event[event[iPos].mother1()].mother1()].mother1()
           *event[event[event[iPos].mother1()].mother1()].mother2() == 12 ) )
      matchHP = true;

  // Done
  return ( matchHP && (matchQN1 || matchQN2) );

}


//--------------------------------------------------------------------------

// Function to check if instead of the particle event[iCandidate], another
// particle could serve as part of the hard process. Assumes that iCandidate
// is already stored as part of the hard process.

bool DireHardProcess::findOtherCandidates(int iPos, const Event& event,
    bool doReplace){

  // Return value
  bool foundCopy = false;

  // Save stored candidates' properties.
  int id  = event[iPos].id();
  int col = event[iPos].col();
  int acl = event[iPos].acol();

  // If the particle's mother is an identified intermediate resonance,
  // then do not attempt any replacement.
  int iMoth1 = event[iPos].mother1();
  int iMoth2 = event[iPos].mother2();
  if ( iMoth1 > 0 && iMoth2 == 0 ) {
    bool hasIdentifiedMother = false;
    for(int i=0; i < int(PosIntermediate.size()); ++i)
      // Compare particle properties
      if ( event[iMoth1].id()         == state[PosIntermediate[i]].id()
        && event[iMoth1].colType()    == state[PosIntermediate[i]].colType()
        && event[iMoth1].chargeType() == state[PosIntermediate[i]].chargeType()
        && event[iMoth1].col()        == state[PosIntermediate[i]].col()
        && event[iMoth1].acol()       == state[PosIntermediate[i]].acol()
        && event[iMoth1].charge()     == state[PosIntermediate[i]].charge() )
         hasIdentifiedMother = true;
    if(hasIdentifiedMother && event[iMoth1].id() != id) return false;
  }

  // Find candidate amongst the already stored ME process candidates.
  vector<int> candidates1;
  vector<int> candidates2;
  // Check outgoing candidates
  for(int i=0; i < int(PosOutgoing1.size()); ++i)
    // Compare particle properties
    if ( id  == state[PosOutgoing1[i]].id()
      && col == state[PosOutgoing1[i]].col()
      && acl == state[PosOutgoing1[i]].acol() )
      candidates1.push_back(i);
  // Check outgoing candidates
  for(int i=0; i < int(PosOutgoing2.size()); ++i)
    // Compare particle properties
    if ( id  == state[PosOutgoing2[i]].id()
      && col == state[PosOutgoing2[i]].col()
      && acl == state[PosOutgoing2[i]].acol() )
      candidates2.push_back(i);

  // If more / less than one stored candidate for iPos has been found, exit.
  if ( candidates1.size() + candidates2.size() != 1 ) return false;

  // Now check for other allowed candidates.
  unordered_map<int,int> further1;
  for(int i=0; i < int(state.size()); ++i)
    for(int j=0; j < int(PosOutgoing1.size()); ++j)
      // Do nothing if this particle has already be found,
      // or if this particle is a jet, lepton container or lepton
      if ( state[i].isFinal()
        && i != PosOutgoing1[j]
        && state[PosOutgoing1[j]].id() == id
        && state[i].id() == id ){
        // Declare vector of already existing candiates.
        vector<int> newPosOutgoing1;
        for(int k=0; k < int(PosOutgoing1.size()); ++k)
          if ( k != j ) newPosOutgoing1.push_back( PosOutgoing1[k] );
        // If allowed, remember replacment parton.
        if ( allowCandidates(i, newPosOutgoing1, PosOutgoing2, state) )
          further1.insert(make_pair(j, i));
      }

  // Now check for other allowed candidates.
  unordered_map<int,int> further2;
  for(int i=0; i < int(state.size()); ++i)
    for(int j=0; j < int(PosOutgoing2.size()); ++j)
      // Do nothing if this particle has already be found,
      // or if this particle is a jet, lepton container or lepton
      if ( state[i].isFinal()
        && i != PosOutgoing2[j]
        && state[PosOutgoing2[j]].id() == id
        && state[i].id() == id ){
        // Declare vector of already existing candidates.
        vector<int> newPosOutgoing2;
        for(int k=0; k < int(PosOutgoing2.size()); ++k)
          if ( k != j ) newPosOutgoing2.push_back( PosOutgoing2[k] );
        // If allowed, remember replacment parton.
        if ( allowCandidates(i, PosOutgoing1, newPosOutgoing2, state) )
          further2.insert(make_pair(j, i));
      }

  // Remove all hard process particles that would be counted twice.
  unordered_map<int,int>::iterator it2 = further2.begin();
  while(it2 != further2.end()) {
    bool remove = false;
    for(int j=0; j < int(PosOutgoing2.size()); ++j)
      if (it2->second == PosOutgoing2[j] ) remove = true;
    if ( remove ) further2.erase(it2++);
    else ++it2;
  }
  unordered_map<int,int>::iterator it1 = further1.begin();
  while(it1 != further1.end()) {
    bool remove = false;
    for(int j=0; j < int(PosOutgoing1.size()); ++j)
      if (it1->second == PosOutgoing1[j] ) remove = true;
    if ( remove ) further1.erase(it1++);
    else ++it1;
  }

  // Decide of a replacment candidate has been found.
  foundCopy = (doReplace)
            ? exchangeCandidates(candidates1, candidates2, further1, further2)
            : (further1.size() + further2.size() > 0);

  // Done
  return foundCopy;

}

//--------------------------------------------------------------------------

// Function to exchange hard process candidates.

bool DireHardProcess::exchangeCandidates( vector<int> candidates1,
    vector<int> candidates2, unordered_map<int,int> further1,
    unordered_map<int,int> further2) {

  int nOld1 = candidates1.size();
  int nOld2 = candidates2.size();
  int nNew1 = further1.size();
  int nNew2 = further2.size();
  bool exchanged = false;
  // Replace, if one-to-one correspondence exists.
  if ( nOld1 == 1 && nOld2 == 0 && nNew1 == 1 && nNew2 == 0){
    PosOutgoing1[further1.begin()->first] = further1.begin()->second;
    exchanged = true;
  } else if ( nOld1 == 0 && nOld2 == 1 && nNew1 == 0 && nNew2 == 1){
    PosOutgoing2[further2.begin()->first] = further2.begin()->second;
    exchanged = true;
  // Else simply swap with the first candidate.
  } else if ( nNew1 >  1 && nNew2 == 0 ) {
    PosOutgoing1[further1.begin()->first] = further1.begin()->second;
    exchanged = true;
  } else if ( nNew1 == 0 && nNew2 >  0 ) {
    PosOutgoing2[further2.begin()->first] = further2.begin()->second;
    exchanged = true;
  }

  // Done
  return exchanged;

}

//==========================================================================

// The DireMergingHooks class.

//--------------------------------------------------------------------------

// Initialise DireMergingHooks class

void DireMergingHooks::init(){

  // Abuse init to store and restore state of MergingHooks.
  if (isInit)   { store();   isInit = false; isStored = true;  return;}
  if (isStored) { restore(); isInit = true;  isStored = false; return;}

  // Get core process from user input. Return if no process was selected.
  processSave           = settingsPtr->word("Merging:Process");
  if (processSave == "void") return;

  // Save pointers
  showers               = 0;

  // Initialise AlphaS objects for reweighting
  double alphaSvalueFSR = settingsPtr->parm("TimeShower:alphaSvalue");
  int    alphaSorderFSR = settingsPtr->mode("TimeShower:alphaSorder");
  int    alphaSnfmax    = settingsPtr->mode("StandardModel:alphaSnfmax");
  int    alphaSuseCMWFSR= settingsPtr->flag("TimeShower:alphaSuseCMW");
  AlphaS_FSRSave.init(alphaSvalueFSR, alphaSorderFSR, alphaSnfmax,
    alphaSuseCMWFSR);
  double alphaSvalueISR = settingsPtr->parm("SpaceShower:alphaSvalue");
  int    alphaSorderISR = settingsPtr->mode("SpaceShower:alphaSorder");
  int    alphaSuseCMWISR= settingsPtr->flag("SpaceShower:alphaSuseCMW");
  AlphaS_ISRSave.init(alphaSvalueISR, alphaSorderISR, alphaSnfmax,
    alphaSuseCMWISR);

  // Initialise AlphaEM objects for reweighting
  int    alphaEMFSRorder = settingsPtr->mode("TimeShower:alphaEMorder");
  AlphaEM_FSRSave.init(alphaEMFSRorder, settingsPtr);
  int    alphaEMISRorder = settingsPtr->mode("SpaceShower:alphaEMorder");
  AlphaEM_ISRSave.init(alphaEMISRorder, settingsPtr);

  // Initialise merging switches
  doUserMergingSave      = settingsPtr->flag("Merging:doUserMerging");
  // Initialise automated MadGraph kT merging
  doMGMergingSave        = settingsPtr->flag("Merging:doMGMerging");
  // Initialise kT merging
  doKTMergingSave        = settingsPtr->flag("Merging:doKTMerging");
  // Initialise evolution-pT merging
  doPTLundMergingSave    = settingsPtr->flag("Merging:doPTLundMerging");
  // Initialise \Delta_R_{ij}, pT_i Q_{ij} merging
  doCutBasedMergingSave  = settingsPtr->flag("Merging:doCutBasedMerging");
  // Initialise exact definition of kT
  ktTypeSave             = settingsPtr->mode("Merging:ktType");

  // Initialise NL3 switches.
  doNL3TreeSave          = settingsPtr->flag("Merging:doNL3Tree");
  doNL3LoopSave          = settingsPtr->flag("Merging:doNL3Loop");
  doNL3SubtSave          = settingsPtr->flag("Merging:doNL3Subt");
  bool doNL3             = doNL3TreeSave || doNL3LoopSave || doNL3SubtSave;

  // Initialise UNLOPS switches.
  doUNLOPSTreeSave      =  settingsPtr->flag("Merging:doUNLOPSTree");
  doUNLOPSLoopSave      =  settingsPtr->flag("Merging:doUNLOPSLoop");
  doUNLOPSSubtSave      =  settingsPtr->flag("Merging:doUNLOPSSubt");
  doUNLOPSSubtNLOSave   =  settingsPtr->flag("Merging:doUNLOPSSubtNLO");
  bool doUNLOPS         = doUNLOPSTreeSave || doUNLOPSLoopSave
                       || doUNLOPSSubtSave || doUNLOPSSubtNLOSave;

  // Initialise UMEPS switches
  doUMEPSTreeSave      =  settingsPtr->flag("Merging:doUMEPSTree");
  doUMEPSSubtSave      =  settingsPtr->flag("Merging:doUMEPSSubt");
  nReclusterSave       =  settingsPtr->mode("Merging:nRecluster");
  nQuarksMergeSave     =  settingsPtr->mode("Merging:nQuarksMerge");
  nRequestedSave       =  settingsPtr->mode("Merging:nRequested");
  bool doUMEPS         =  doUMEPSTreeSave || doUMEPSSubtSave;

  // Flag to only do phase space cut.
  doEstimateXSection   =  settingsPtr->flag("Merging:doXSectionEstimate");

  doMOPSSave           = settingsPtr->flag("Dire:doMOPS");
  doMEMSave            = settingsPtr->flag("Dire:doMEM");

  // Flag to check if merging weight should directly be included in the cross
  // section.
  includeWGTinXSECSave = settingsPtr->flag("Merging:includeWeightInXsection");

  // Flag to check if CKKW-L event veto should be applied.
  applyVeto            =  settingsPtr->flag("Merging:applyVeto");

  // Clear hard process
  hardProcess->clear();

  // Initialise input event.
  inputEvent.init("(hard process)", particleDataPtr);

  doRemoveDecayProducts = settingsPtr->flag("Merging:mayRemoveDecayProducts");

  // Initialise the hard process
  if ( doMGMergingSave )
    hardProcess->initOnLHEF(lheInputFile, particleDataPtr);
  else
    hardProcess->initOnProcess(processSave, particleDataPtr);

  // Parameters for reconstruction of evolution scales
  includeMassiveSave        = settingsPtr->flag("Merging:includeMassive");
  enforceStrongOrderingSave = settingsPtr->flag
    ("Merging:enforceStrongOrdering");
  scaleSeparationFactorSave = settingsPtr->parm
    ("Merging:scaleSeparationFactor");
  orderInRapiditySave       = settingsPtr->flag("Merging:orderInRapidity");

  // Parameters for choosing history probabilistically
  nonJoinedNormSave     = settingsPtr->parm("Merging:nonJoinedNorm");
  fsrInRecNormSave      = settingsPtr->parm("Merging:fsrInRecNorm");
  pickByFullPSave       = settingsPtr->flag("Merging:pickByFullP");
  pickByPoPT2Save       = settingsPtr->flag("Merging:pickByPoPT2");
  includeRedundantSave  = settingsPtr->flag("Merging:includeRedundant");

  // Parameters for scale choices
  unorderedScalePrescipSave   =
    settingsPtr->mode("Merging:unorderedScalePrescrip");
  unorderedASscalePrescipSave =
    settingsPtr->mode("Merging:unorderedASscalePrescrip");
  unorderedPDFscalePrescipSave =
    settingsPtr->mode("Merging:unorderedPDFscalePrescrip");
  incompleteScalePrescipSave  =
    settingsPtr->mode("Merging:incompleteScalePrescrip");

  // Parameter for allowing swapping of one colour index while reclustering
  allowColourShufflingSave  =
    settingsPtr->flag("Merging:allowColourShuffling");

  // Parameters to allow setting hard process scales to default (dynamical)
  // Pythia values.
  resetHardQRenSave     =  settingsPtr->flag("Merging:usePythiaQRenHard");
  resetHardQFacSave     =  settingsPtr->flag("Merging:usePythiaQFacHard");

  // Parameters for choosing history by sum(|pT|)
  pickBySumPTSave       = settingsPtr->flag("Merging:pickBySumPT");
  herwigAcollFSRSave    = settingsPtr->parm("Merging:aCollFSR");
  herwigAcollISRSave    = settingsPtr->parm("Merging:aCollISR");

  // Information on the shower cut-off scale
  pT0ISRSave            = settingsPtr->parm("SpaceShower:pT0Ref");
  pTcutSave             = settingsPtr->parm("SpaceShower:pTmin");
  pTcutSave             = max(pTcutSave,pT0ISRSave);

  // Initialise CKKWL weight
  weightCKKWLSave = {1.};
  weightFIRSTSave = {0.};
  nMinMPISave = 100;
  muMISave = -1.;

  // Initialise merging scale
  tmsValueSave = 0.;
  tmsListSave.resize(0);

  kFactor0jSave         = settingsPtr->parm("Merging:kFactor0j");
  kFactor1jSave         = settingsPtr->parm("Merging:kFactor1j");
  kFactor2jSave         = settingsPtr->parm("Merging:kFactor2j");

  muFSave               = settingsPtr->parm("Merging:muFac");
  muRSave               = settingsPtr->parm("Merging:muRen");
  muFinMESave           = settingsPtr->parm("Merging:muFacInME");
  muRinMESave           = settingsPtr->parm("Merging:muRenInME");

  doWeakClusteringSave  = settingsPtr->flag("Merging:allowWeakClustering");
  doSQCDClusteringSave  = settingsPtr->flag("Merging:allowSQCDClustering");
  DparameterSave        = settingsPtr->parm("Merging:Dparameter");

  // Save merging scale on maximal number of jets
  if (  doKTMergingSave || doUserMergingSave || doPTLundMergingSave
    || doUMEPS ) {
    // Read merging scale (defined in kT) from input parameter.
    tmsValueSave    = settingsPtr->parm("Merging:TMS");
    nJetMaxSave     = settingsPtr->mode("Merging:nJetMax");
    nJetMaxNLOSave  = -1;
  } else if (doMGMergingSave) {
    // Read merging scale (defined in kT) from LHE file.
    tmsValueSave    = hardProcess->tms;
    nJetMaxSave     = settingsPtr->mode("Merging:nJetMax");
    nJetMaxNLOSave  = -1;
  } else if (doCutBasedMergingSave) {

    // Save list of cuts defining the merging scale.
    nJetMaxSave     = settingsPtr->mode("Merging:nJetMax");
    nJetMaxNLOSave  = -1;
    // Write tms cut values to list of cut values,
    // ordered by DeltaR_{ij}, pT_{i}, Q_{ij}.
    tmsListSave.resize(0);
    double drms     = settingsPtr->parm("Merging:dRijMS");
    double ptms     = settingsPtr->parm("Merging:pTiMS");
    double qms      = settingsPtr->parm("Merging:QijMS");
    tmsListSave.push_back(drms);
    tmsListSave.push_back(ptms);
    tmsListSave.push_back(qms);

  }

  // Read additional settingsPtr->for NLO merging methods.
  if ( doNL3 || doUNLOPS || doEstimateXSection ) {
    tmsValueSave    = settingsPtr->parm("Merging:TMS");
    nJetMaxSave     = settingsPtr->mode("Merging:nJetMax");
    nJetMaxNLOSave  = settingsPtr->mode("Merging:nJetMaxNLO");
  }

  // Internal Pythia cross section should not include NLO merging weights.
  if ( doNL3 || doUNLOPS ) includeWGTinXSECSave = false;

  hasJetMaxLocal  = false;
  nJetMaxLocal    = nJetMaxSave;
  nJetMaxNLOLocal = nJetMaxNLOSave;

  // Check if external shower plugin should be used.
  useShowerPluginSave = settingsPtr->flag("Merging:useShowerPlugin");

  bool writeBanner =  doKTMergingSave || doMGMergingSave
                   || doUserMergingSave
                   || doNL3 || doUNLOPS || doUMEPS
                   || doPTLundMergingSave || doCutBasedMergingSave;

  isInit = true;

  if (!writeBanner) return;

  // Write banner.
  cout << "\n *------------------ MEPS Merging Initialization  ---------------"
       << "---*";
  cout << "\n |                                                               "
       << "   |\n";
  if ( doKTMergingSave || doMGMergingSave || doUserMergingSave
    || doPTLundMergingSave || doCutBasedMergingSave )
    cout << " | CKKW-L merge                                                  "
         << "   |\n"
         << " |"<< setw(34) << processSave << "  with up to"
         << setw(3) << nJetMaxSave << " additional jets |\n";
  else if ( doNL3 )
    cout << " | NL3 merge                                                     "
         << "   |\n"
         << " |" << setw(31) << processSave << " with jets up to"
         << setw(3) << nJetMaxNLOSave << " correct to NLO |\n"
         << " | and up to" << setw(3) << nJetMaxSave
         << " additional jets included by CKKW-L merging at LO    |\n";
  else if ( doUNLOPS )
    cout << " | UNLOPS merge                                                  "
         << "   |\n"
         << " |" << setw(31) << processSave << " with jets up to"
         << setw(3)<< nJetMaxNLOSave << " correct to NLO |\n"
         << " | and up to" << setw(3) << nJetMaxSave
         << " additional jets included by UMEPS merging at LO     |\n";
  else if ( doUMEPS )
    cout << " | UMEPS merge                                                   "
         << "   |\n"
         << " |" << setw(34) << processSave << "  with up to"
         << setw(3) << nJetMaxSave << " additional jets |\n";

  if ( doKTMergingSave )
    cout << " | Merging scale is defined in kT, with value ktMS = "
         << tmsValueSave << " GeV";
  else if ( doMGMergingSave )
    cout << " | Perform automanted MG/ME merging \n"
         << " | Merging scale is defined in kT, with value ktMS = "
       << setw(6) << fixed << setprecision(1) << tmsValueSave << " GeV |";
  else if ( doUserMergingSave )
    cout << " | Merging scale is defined by the user, with value tMS = "
         << setw(6) << fixed << setprecision(1) << tmsValueSave << "     |";
  else if ( doPTLundMergingSave )
    cout << " | Merging scale is defined by Lund pT, with value tMS = "
         << setw(6) << fixed << setprecision(1) << tmsValueSave << " GeV |";
  else if ( doCutBasedMergingSave )
    cout << " | Merging scale is defined by combination of Delta R_{ij}, pT_i "
         << "   |\n"
         << " | and Q_{ij} cut, with values                                   "
         << "   |\n"
         << " | Delta R_{ij,min} = "
         << setw(7) << scientific << setprecision(2) << tmsListSave[0]
         << "                                      |\n"
         << " | pT_{i,min}       = "
         << setw(6) << fixed << setprecision(1) << tmsListSave[1]
         << " GeV                                    |\n"
         << " | Q_{ij,min}       = "
         << setw(6) << fixed << setprecision(1) << tmsListSave[2]
         << " GeV                                    |";
  else if ( doNL3TreeSave )
    cout << " | Generate tree-level O(alpha_s)-subtracted events              "
         << "   |\n"
         << " | Merging scale is defined by Lund pT, with value tMS = "
         << setw(6) << fixed << setprecision(1) << tmsValueSave << " GeV |";
  else if ( doNL3LoopSave )
    cout << " | Generate virtual correction unit-weight events                "
         << "   |\n"
         << " | Merging scale is defined by Lund pT, with value tMS = "
       << setw(6) << fixed << setprecision(1) << tmsValueSave << " GeV |";
  else if ( doNL3SubtSave )
    cout << " | Generate reclustered tree-level events                        "
         << "   |\n"
         << " | Merging scale is defined by Lund pT, with value tMS = "
         << setw(6) << fixed << setprecision(1) << tmsValueSave << " GeV |";
  else if ( doUNLOPSTreeSave )
    cout << " | Generate tree-level O(alpha_s)-subtracted events              "
         << "   |\n"
         << " | Merging scale is defined by Lund pT, with value tMS = "
         << setw(6) << fixed << setprecision(1) << tmsValueSave << " GeV |";
  else if ( doUNLOPSLoopSave )
    cout << " | Generate virtual correction unit-weight events                "
         << "   |\n"
         << " | Merging scale is defined by Lund pT, with value tMS = "
         << setw(6) << fixed << setprecision(1) << tmsValueSave << " GeV |";
  else if ( doUNLOPSSubtSave )
    cout << " | Generate reclustered tree-level events                        "
         << "   |\n"
         << " | Merging scale is defined by Lund pT, with value tMS = "
         << setw(6) << fixed << setprecision(1) << tmsValueSave << " GeV |";
  else if ( doUNLOPSSubtNLOSave )
    cout << " | Generate reclustered loop-level events                        "
         << "   |\n"
         << " | Merging scale is defined by Lund pT, with value tMS = "
         << setw(6) << fixed << setprecision(1) << tmsValueSave << " GeV |";
  else if ( doUMEPSTreeSave )
    cout << " | Generate tree-level events                                    "
         << "   |\n"
         << " | Merging scale is defined by Lund pT, with value tMS = "
         << setw(6) << fixed << setprecision(1) << tmsValueSave << " GeV |";
  else if ( doUMEPSSubtSave )
    cout << " | Generate reclustered tree-level events                        "
         << "   |\n"
         << " | Merging scale is defined by Lund pT, with value tMS = "
         << setw(6) << fixed << setprecision(1) << tmsValueSave << " GeV |";

  cout << "\n |                                                               "
       << "   |";
  cout << "\n *-------------- END MEPS Merging Initialization  ---------------"
       << "---*\n\n";

}

//--------------------------------------------------------------------------


void DireMergingHooks::store() {

  hardProcessStore.hardIncoming1    = hardProcess->hardIncoming1;
  hardProcessStore.hardIncoming2    = hardProcess->hardIncoming2;
  hardProcessStore.hardOutgoing1    = hardProcess->hardOutgoing1;
  hardProcessStore.hardOutgoing2    = hardProcess->hardOutgoing2;
  hardProcessStore.hardIntermediate = hardProcess->hardIntermediate;
  hardProcessStore.state            = hardProcess->state;
  hardProcessStore.PosOutgoing1     = hardProcess->PosOutgoing1;
  hardProcessStore.PosOutgoing2     = hardProcess->PosOutgoing2;
  hardProcessStore.PosIntermediate  = hardProcess->PosIntermediate;
  hardProcessStore.tms              = hardProcess->tms;

  nReclusterStore                   = nReclusterSave;
  nRequestedStore                   = nRequestedSave;
  pT0ISRStore                       = pT0ISRSave;
  pTcutStore                        = pTcutSave;
  inputEventStore                   = inputEvent;
  resonancesStore                   = resonances;
  muMIStore                         = muMISave;
  tmsValueStore                     = tmsValueSave;
  tmsValueNowStore                  = tmsValueNow;
  DparameterStore                   = DparameterSave;
  nJetMaxStore                      = nJetMaxSave;
  nJetMaxNLOStore                   = nJetMaxNLOSave;
  doOrderHistoriesStore             = doOrderHistoriesSave;
  muFStore                          = muFSave;
  muRStore                          = muRSave;
  muFinMEStore                      = muFinMESave;
  muRinMEStore                      = muRinMESave;
  doIgnoreEmissionsStore            = doIgnoreEmissionsSave;
  doIgnoreStepStore                 = doIgnoreStepSave;
  pTstore                           = pTsave;
  nMinMPIStore                      = nMinMPISave;
  nJetMaxLocalStore                 = nJetMaxLocal;
  nJetMaxNLOLocalStore              = nJetMaxNLOLocal;
  hasJetMaxLocalStore               = hasJetMaxLocal;
  nHardNowStore                     = nHardNowSave;
  nJetNowStore                      = nJetNowSave;
  tmsHardNowStore                   = tmsHardNowSave;
  tmsNowStore                       = tmsNowSave;

}

//--------------------------------------------------------------------------

void DireMergingHooks::restore() {

  hardProcess->hardIncoming1    = hardProcessStore.hardIncoming1;
  hardProcess->hardIncoming2    = hardProcessStore.hardIncoming2;
  hardProcess->hardOutgoing1    = hardProcessStore.hardOutgoing1;
  hardProcess->hardOutgoing2    = hardProcessStore.hardOutgoing2;
  hardProcess->hardIntermediate = hardProcessStore.hardIntermediate;
  hardProcess->state            = hardProcessStore.state;
  hardProcess->PosOutgoing1     = hardProcessStore.PosOutgoing1;
  hardProcess->PosOutgoing2     = hardProcessStore.PosOutgoing2;
  hardProcess->PosIntermediate  = hardProcessStore.PosIntermediate;
  hardProcess->tms              = hardProcessStore.tms;

  nReclusterSave                = nReclusterStore;
  nRequestedSave                = nRequestedStore;
  pT0ISRSave                    = pT0ISRStore;
  pTcutSave                     = pTcutStore;
  inputEvent                    = inputEventStore;
  resonances                    = resonancesStore;
  muMISave                      = muMIStore;
  tmsValueSave                  = tmsValueStore;
  tmsValueNow                   = tmsValueNowStore;
  DparameterSave                = DparameterStore;
  nJetMaxSave                   = nJetMaxStore;
  nJetMaxNLOSave                = nJetMaxNLOStore;
  doOrderHistoriesSave          = doOrderHistoriesStore;
  muFSave                       = muFStore;
  muRSave                       = muRStore;
  muFinMESave                   = muFinMEStore;
  muRinMESave                   = muRinMEStore;
  doIgnoreEmissionsSave         = doIgnoreEmissionsStore;
  doIgnoreStepSave              = doIgnoreStepStore;
  pTsave                        = pTstore;
  nMinMPISave                   = nMinMPIStore;
  nJetMaxLocal                  = nJetMaxLocalStore;
  nJetMaxNLOLocal               = nJetMaxNLOLocalStore;
  hasJetMaxLocal                = hasJetMaxLocalStore;
  nHardNowSave                  = nHardNowStore;
  nJetNowSave                   = nJetNowStore;
  tmsHardNowSave                = tmsHardNowStore;
  tmsNowSave                    = tmsNowStore;

}

//--------------------------------------------------------------------------

// Function to check if emission should be rejected.

bool DireMergingHooks::doVetoEmission( const Event& event) {

  // Do nothing in trial showers, or after first step.
  if ( doIgnoreEmissionsSave ) return false;

  // Do nothing in CKKW-L
  if (  doUserMerging() || doMGMerging() || doKTMerging()
    ||  doPTLundMerging() || doCutBasedMerging() )
     return false;

  if ( doMOPS() ) return false;

  // For NLO merging, count and veto emissions above the merging scale
  bool veto = false;
  // Get number of clustering steps
  int nSteps  = getNumberOfClusteringSteps(event);
  // Get merging scale in current event
  double tnow = tmsNow( event);

  // Get maximal number of additional jets
  int nJetMax = nMaxJets();
  // Always remove emissions above the merging scale for
  // samples containing reclusterings!
  if ( nRecluster() > 0 ) nSteps = max(1, min(nJetMax-2, 1));
  // Check veto condition
  if ( nSteps - 1 < nJetMax && nSteps >= 1 && tnow > tms() && tms() > 0. )
    veto = true;

  // Do not veto if state already includes MPI.
  if ( infoPtr->nMPI() > 1 ) veto = false;

  // When performing NL3 merging of tree-level events, reset the
  // CKKWL weight.
  if ( veto && doNL3Tree() ) setWeightCKKWL({0.});

  // If the emission is allowed, do not check any further emissions
  if ( !veto ) doIgnoreEmissionsSave = true;

  // Done
  return veto;

}

//--------------------------------------------------------------------------

// Function to check if emission should be rejected.

bool DireMergingHooks::doVetoStep( const Event& process, const Event& event,
  bool doResonance ) {

  // Do nothing in trial showers, or after first step.
  if ( doIgnoreStepSave && !doResonance ) return false;

  // Do nothing in UMEPS or UNLOPS
  if ( doUMEPSTree() || doUMEPSSubt() || doUMEPSMerging() || doUNLOPSTree()
    || doUNLOPSLoop() || doUNLOPSSubt() || doUNLOPSSubtNLO()
    || doUNLOPSMerging() )
    return false;

  if ( doMOPS() ) return false;

  // Get number of clustering steps. If necessary, remove resonance
  // decay products first.
  int nSteps = 0;
  if ( getProcessString().find("inc") != string::npos )
    nSteps = getNumberOfClusteringSteps( bareEvent( process, false) );
  else nSteps  = (doResonance) ? getNumberOfClusteringSteps(process)
         : getNumberOfClusteringSteps( bareEvent( process, false) );

  // Get maximal number of additional jets.
  int nJetMax = nMaxJets();
  // Get merging scale in current event.
  double tnow = tmsNow( event );

  // Do not print zero-weight events.
  // For non-resonant showers, simply check veto. If event should indeed be
  // vetoed, save the current pT and weights in case the veto needs to be
  // revoked.
  if ( !doResonance ) {

    // Store pT to check if veto needs to be revoked later.
    pTsave = infoPtr->pTnow();
    if ( nRecluster() == 1) nSteps--;

    //// Store veto inputs to perform veto at a later stage.
    //if (!applyVeto) {
    //  setEventVetoInfo(nSteps, tnow);
    //  return false;
    //}

    // Check merging veto condition.
    bool veto = false;
    if ( nSteps > nMaxJetsNLO() && nSteps < nJetMax && tnow > tms()
      && tms() > 0. ) {
      // Set weight to zero if event should be vetoed.
      weightCKKWL1Save = {0.};
      // Save weight before veto, in case veto needs to be revoked.
      weightCKKWL2Save = getWeightCKKWL();
      // Reset stored weights.
      if ( !includeWGTinXSEC() ) setWeightCKKWL({0.});
      if (  includeWGTinXSEC() ) infoPtr->weightContainerPtr->
        setWeightNominal(0.);
      veto = true;
    }

    // Store veto inputs to perform veto at a later stage.
    if (!applyVeto) {
      setEventVetoInfo(nSteps, tnow);
      return false;
    }

    // Done
    return veto;

  // For resonant showers, check if any previous veto should be revoked.
  // This means we treat showers off resonance decay products identical to
  // MPI: If a hard resonance emission has been produced, the event should
  // have been kept. Following this reasoning, it might be necessary to revoke
  // any previous veto.
  } else {

    // Initialise switch to revoke vetoing.
    bool revokeVeto = false;

    // Nothing to check if pTsave was not stored, i.e. no emission to
    // possibly veto was recorded.
    // Only allow revoking the veto for diboson processes, with resonant
    // electroweak bosons
    bool check =  (nHardInLeptons() == 0)&& (nHardOutLeptons() == 2)
               && (nHardOutPartons() == 2);

    // For current purpose only!!!
    check = false;

    // For hadronic resonance decays at hadron colliders, do not veto
    // events with a hard emission of the resonance decay products,
    // since such states are not included in the matrix element
    if ( pTsave > 0. && check ) {

      // Check how many resonance decay systems are allowed
      int nResNow = nResInCurrent();

      // Find systems really containing only emission off a resonance
      // decay
      vector<int>goodSys;
      // Resonance decay systems are considered last, thus at the end of
      // the list
      int sysSize = partonSystemsPtr->sizeSys();
      for ( int i=0; i < nResNow; ++i ) {
        if ( partonSystemsPtr->sizeOut(sysSize - 1 - i) == 3 )
          goodSys.push_back(sysSize - 1 - i);
      }

      // Check the members of the resonance decay systems
      for ( int i=0; i < int(goodSys.size()); ++i ) {

        // Save the (three) members of the resonance decay system
        int iMem1 = partonSystemsPtr->getOut(goodSys[i],0);
        int iMem2 = partonSystemsPtr->getOut(goodSys[i],1);
        int iMem3 = partonSystemsPtr->getOut(goodSys[i],2);

        // Now find emitted gluon or gamma
        int iEmtGlue = ((event[iMem1].id() == 21) ? iMem1
                     : ((event[iMem2].id() == 21) ? iMem2
                       : ((event[iMem3].id() == 21) ? iMem3: 0)));
        int iEmtGamm = ((event[iMem1].id() == 22) ? iMem1
                     : ((event[iMem2].id() == 22) ? iMem2
                       : ((event[iMem3].id() == 22) ? iMem3: 0)));
        // Per system, only one emission
        int iEmt = (iEmtGlue != 0) ? iEmtGlue : iEmtGamm;

        int iRad = 0;
        int iRec = 0;
        if ( iEmt == iMem1 ) {
          iRad = (event[iMem2].mother1() != event[iMem2].mother2())
               ? iMem2 : iMem3;
          iRec = (event[iMem3].mother1() == event[iMem3].mother2())
               ? iMem3 : iMem2;
        } else if ( iEmt == iMem2 ) {
          iRad = (event[iMem1].mother1() != event[iMem1].mother2())
               ? iMem1 : iMem3;
          iRec = (event[iMem3].mother1() == event[iMem3].mother2())
               ? iMem3 : iMem1;
        } else {
          iRad = (event[iMem1].mother1() != event[iMem1].mother2())
               ? iMem1 : iMem2;
          iRec = (event[iMem2].mother1() == event[iMem2].mother2())
               ? iMem2 : iMem1;
        }

        double pTres = rhoPythia(event, iRad, iEmt, iRec, 1);

        // Revoke previous veto of last emission if a splitting of the
        // resonance produced a harder parton, i.e. we are inside the
        // PS region
        if ( pTres > pTsave ) {
          revokeVeto = true;
        // Do nothing (i.e. allow other first emission veto) for soft
        // splitting
        } else {
          revokeVeto = false;
        }
      // Done with one system
      }
    // Done with all systems
    }

    // Check veto condition
    bool veto = false;
    if ( revokeVeto && check ) {
      setWeightCKKWL(weightCKKWL2Save);
    } else if ( check ) {
      setWeightCKKWL(weightCKKWL1Save);
      if ( weightCKKWL1Save[0] == 0. ) veto = true;
    }

    // Check veto condition.
    if ( !check && nSteps > nMaxJetsNLO() && nSteps < nJetMax && tnow > tms()
      && tms() > 0.){
      // Set stored weights to zero.
      if ( !includeWGTinXSEC() ) setWeightCKKWL({0.});
      if (  includeWGTinXSEC() ) infoPtr->weightContainerPtr->
        setWeightNominal(0.);
      // Now allow veto.
      veto = true;
    }

    // If the emission is allowed, do not check any further emissions
    if ( !veto || !doIgnoreStepSave ) doIgnoreStepSave = true;

    // Done
    return veto;

  }

  // Done
  return false;

}

//--------------------------------------------------------------------------

// Function to return the number of clustering steps for the current event

int DireMergingHooks::getNumberOfClusteringSteps(const Event& event,
  bool resetJetMax ){

  // Count the number of final state partons
  int nFinalPartons = 0;
  for ( int i=0; i < event.size(); ++i)
    if ( event[i].isFinal()
      && isInHard( i, event)
      && (event[i].isQuark() || event[i].isGluon()) )
      nFinalPartons++;

  // Count the number of final state leptons
  int nFinalLeptons = 0;
  for( int i=0; i < event.size(); ++i)
    if ( event[i].isFinal() && isInHard( i, event) && event[i].isLepton())
      nFinalLeptons++;

  // Add neutralinos to number of leptons
  for( int i=0; i < event.size(); ++i)
    if ( event[i].isFinal() && isInHard( i, event)
       && event[i].idAbs() == 1000022)
      nFinalLeptons++;

  // Add sleptons to number of leptons
  for( int i=0; i < event.size(); ++i)
    if ( event[i].isFinal() && isInHard( i, event)
       && (event[i].idAbs() == 1000011
        || event[i].idAbs() == 2000011
        || event[i].idAbs() == 1000013
        || event[i].idAbs() == 2000013
        || event[i].idAbs() == 1000015
        || event[i].idAbs() == 2000015) )
      nFinalLeptons++;

  // Count the number of final state electroweak bosons
  int nFinalBosons = 0;
  for( int i=0; i < event.size(); ++i)
    if ( event[i].isFinal() && isInHard( i, event)
      && ( event[i].idAbs() == 22
        || event[i].idAbs() == 23
        || event[i].idAbs() == 24
        || event[i].idAbs() == 25 ) )
      nFinalBosons++;

  // Save sum of all final state particles
  int nFinal = nFinalPartons + nFinalLeptons
             + 2*(nFinalBosons - nHardOutBosons() );

  // Return the difference to the core process outgoing particles
  int nsteps = nFinal - nHardOutPartons() - nHardOutLeptons();

  nsteps =  nFinalPartons     + nFinalLeptons     + nFinalBosons
         - (nHardOutPartons() + nHardOutLeptons() + nHardOutBosons());

  // For inclusive handling, the number of reclustering steps
  // can be different within a single sample.
  if ( getProcessString().find("inc") != string::npos ) {

    // Final particle counters
    int njInc = 0, naInc = 0, nzInc = 0, nwInc =0, nhInc = 0;
    for (int i=0; i < event.size(); ++i){
      if ( event[i].isFinal() && event[i].colType() != 0 ) njInc++;
      if ( getProcessString().find("Ainc") != string::npos
        && event[i].isFinal() && event[i].idAbs() == 22 )  naInc++;
      if ( getProcessString().find("Zinc") != string::npos
        && event[i].isFinal() && event[i].idAbs() == 23 )  nzInc++;
      if ( getProcessString().find("Winc") != string::npos
        && event[i].isFinal() && event[i].idAbs() == 24 )  nwInc++;
      if ( getProcessString().find("Hinc") != string::npos
        && event[i].isFinal() && event[i].idAbs() == 25 )  nhInc++;
    }

    // Set steps for QCD or QCD+QED events: Need at least two
    // massless particles at lowest multiplicity.
    if (nzInc == 0 && nwInc == 0 && nhInc == 0 && njInc+naInc > 1) {
      nsteps = naInc + njInc - 2;
      if (resetJetMax) {
        hasJetMaxLocal = true;
        nJetMaxLocal   = nJetMaxSave - 2;
        nRequestedSave = nsteps;
      }
    }

    // Set steps for events containing heavy bosons. Need at least one
    // massive particle at lowest multiplicity.
    if ( nzInc > 0 || nwInc > 0 || nhInc > 0) {
      nsteps = njInc + naInc + nzInc + nwInc + nhInc - 1;
      if (resetJetMax) {
        hasJetMaxLocal = true;
        nJetMaxLocal   = nJetMaxSave - 1;
        nRequestedSave = nsteps;
      }
    }

  } // dynamical handling of steps

  // Return the difference to the core process outgoing particles
  return nsteps;

}

//--------------------------------------------------------------------------

// Function to set the correct starting scales of the shower.
// Note: 2 -> 2 QCD systems can be produced by MPI. Hence, there is an
// overlap between MPI and "hard" 2 -> 2 QCD systems which needs to be
// removed by no-MPI probabilities. This means that for any "hard" 2 -> 2 QCD
// system, multiparton interactions should start at the maximal scale
// of multiple interactions. The same argument holds for any "hard" process
// that overlaps with MPI.

bool DireMergingHooks::setShowerStartingScales( bool isTrial,
  bool doMergeFirstEmm, double& pTscaleIn, const Event& event,
  double& pTmaxFSRIn, bool& limitPTmaxFSRIn,
  double& pTmaxISRIn, bool& limitPTmaxISRIn,
  double& pTmaxMPIIn, bool& limitPTmaxMPIIn ) {

  // Local copies of power/wimpy shower booleans and scales.
  bool   limitPTmaxFSR = limitPTmaxFSRIn;
  bool   limitPTmaxISR = limitPTmaxISRIn;
  bool   limitPTmaxMPI = limitPTmaxMPIIn;
  double pTmaxFSR      = pTmaxFSRIn;
  double pTmaxISR      = pTmaxISRIn;
  double pTmaxMPI      = pTmaxMPIIn;
  double pTscale       = pTscaleIn;

  // Merging of EW+QCD showers with matrix elements: Ensure that
  // 1. any event with more than one final state particle will be showered
  //    from the reconstructed transverse momentum of the last emission,
  //    even if the factorisation scale is low.
  // 2. the shower starting scale for events with no emission is given by
  //    the (user-defined) choice.
  bool isInclusive = ( getProcessString().find("inc") != string::npos );

  // Check if the process only contains two outgoing partons. If so, then
  // this process could also have been produced by MPI. Thus, the MPI starting
  // scale would need to be set accordingly to correctly attach a
  // "no-MPI-probability" to multi-jet events. ("Hard" MPI are included
  // by not restricting MPI when showering the lowest-multiplicity sample.)
  double pT2to2 = 0;
  int nFinalPartons = 0, nInitialPartons = 0, nFinalOther = 0;
  for ( int i = 0; i < event.size(); ++i ) {
    if ( (event[i].mother1() == 1 || event[i].mother1() == 2 )
      && (event[i].idAbs()   < 6  || event[i].id()      == 21) )
      nInitialPartons++;
    if (event[i].isFinal() && (event[i].idAbs() < 6 || event[i].id() == 21)) {
        nFinalPartons++;
        pT2to2 = event[i].pT();
    } else if ( event[i].isFinal() ) nFinalOther++;
  }
  bool is2to2QCD     = ( nFinalPartons == 2 && nInitialPartons == 2
                      && nFinalOther   == 0 );
  bool hasMPIoverlap = is2to2QCD;
  bool is2to1        = ( nFinalPartons == 0 );

  double scale   = event.scale();

  // SET THE STARTING SCALES FOR TRIAL SHOWERS.
  if ( isTrial ) {

    // Reset shower and MPI scales.
    pTmaxISR = pTmaxFSR = pTmaxMPI = scale;

    // Reset to minimal scale for wimpy showers. Keep scales for EW+QCD
    // merging.
    if ( limitPTmaxISR && !isInclusive ) pTmaxISR = min(scale,muF());
    if ( limitPTmaxFSR && !isInclusive ) pTmaxFSR = min(scale,muF());
    if ( limitPTmaxMPI && !isInclusive ) pTmaxMPI = min(scale,muF());

    // For EW+QCD merging, apply wimpy shower only to 2->1 processes.
    if ( limitPTmaxISR && isInclusive && is2to1 ) pTmaxISR = min(scale,muF());
    if ( limitPTmaxFSR && isInclusive && is2to1 ) pTmaxFSR = min(scale,muF());
    if ( limitPTmaxMPI && isInclusive && is2to1 ) pTmaxMPI = min(scale,muF());

    // For pure QCD set the PS starting scales to the pT of the dijet system.
    if (is2to2QCD) {
      pTmaxFSR = pT2to2;
      pTmaxISR = pT2to2;
    }

    // If necessary, set the MPI starting scale to the collider energy.
    if ( hasMPIoverlap ) pTmaxMPI = infoPtr->eCM();

    // Reset phase space limitation flags
    if ( pTscale < infoPtr->eCM() ) {
      limitPTmaxISR = limitPTmaxFSR = limitPTmaxMPI = true;
      // If necessary, set the MPI starting scale to the collider energy.
      if ( hasMPIoverlap ) limitPTmaxMPI = false;
    }

  }

  // SET THE STARTING SCALES FOR REGULAR SHOWERS.
  if ( doMergeFirstEmm ) {

    // Remember if this is a "regular" shower off a reclustered event.
    bool doRecluster = doUMEPSSubt() || doNL3Subt() || doUNLOPSSubt()
                    || doUNLOPSSubtNLO();

    // Reset shower and MPI scales.
    pTmaxISR = pTmaxFSR = pTmaxMPI = scale;

    // Reset to minimal scale for wimpy showers. Keep scales for EW+QCD
    // merging.
    if ( limitPTmaxISR && !isInclusive ) pTmaxISR = min(scale,muF());
    if ( limitPTmaxFSR && !isInclusive ) pTmaxFSR = min(scale,muF());
    if ( limitPTmaxMPI && !isInclusive ) pTmaxMPI = min(scale,muF());

    // For EW+QCD merging, apply wimpy shower only to 2->1 processes.
    if ( limitPTmaxISR && isInclusive && is2to1 ) pTmaxISR = min(scale,muF());
    if ( limitPTmaxFSR && isInclusive && is2to1 ) pTmaxFSR = min(scale,muF());
    if ( limitPTmaxMPI && isInclusive && is2to1 ) pTmaxMPI = min(scale,muF());

    // For pure QCD set the PS starting scales to the pT of the dijet system.
    if (is2to2QCD) {
      pTmaxFSR = pT2to2;
      pTmaxISR = pT2to2;
    }

    // If necessary, set the MPI starting scale to the collider energy.
    if ( hasMPIoverlap && !doRecluster ) {
      pTmaxMPI = infoPtr->eCM();
      limitPTmaxMPI = false;
    }

    // For reclustered events, no-MPI-probability between "pTmaxMPI" and
    // "scale" already included in the event weight.
    if ( doRecluster ) {
      pTmaxMPI      = muMI();
      limitPTmaxMPI = true;
    }
  }

  // Reset power/wimpy shower switches iand scales if necessary.
  limitPTmaxFSRIn = limitPTmaxFSR;
  limitPTmaxISRIn = limitPTmaxISR;
  limitPTmaxMPIIn = limitPTmaxMPI;
  pTmaxFSRIn      = pTmaxFSR;
  pTmaxISRIn      = pTmaxISR;
  pTmaxMPIIn      = pTmaxMPI;
  pTscaleIn       = pTscale;

  // Done
  return true;

}

//--------------------------------------------------------------------------

// Function to return the value of the merging scale function in the current
// event.

double DireMergingHooks::tmsNow( const Event& event ) {

  // Get merging scale in current event.
  double tnow = 0.;
  //tnow = scalems(event, false);
  tnow = scalems(event);
  // Return merging scale value. Done
  return tnow;
}

//--------------------------------------------------------------------------

// Function to check if the properties of the input particle should be
// checked against the cut-based merging scale defintion.

bool DireMergingHooks::checkAgainstCut( const Particle& particle){

  // Do not check uncoloured particles.
  if (particle.colType() == 0) return false;
  // By default, use u-, d-, c-, s- and b-quarks.
  if ( particle.idAbs() != 21 && particle.idAbs() > nQuarksMergeSave )
    return false;
  // Done
  return true;

}

//--------------------------------------------------------------------------

// Find the minimal Lund pT between coloured partons in the input
// event. If doPTLundMerging = true, this function will be used as a merging
// scale definition.

double DireMergingHooks::scalems( const Event& event){

  // Only check first emission.
  if (!isFirstEmission(event)) return 0.;

  // Find all electroweak decayed bosons in the state.
  vector<int> ewResonancePos;
  ewResonancePos.clear();
  for (int i=0; i < event.size(); ++i)
    if ( abs(event[i].status()) == 22
      && ( event[i].idAbs() == 22
        || event[i].idAbs() == 23
        || event[i].idAbs() == 24
        || event[i].idAbs() == 25
        || event[i].idAbs() == 6 ) )
      ewResonancePos.push_back(i);

  // Declare final parton vectors
  vector <int> FinalPartPos;
  FinalPartPos.clear();
  // Search inEvent record for final state partons.
  // Exclude decay products of ew resonance.
  for (int i=0; i < event.size(); ++i){

    if ( event[i].isFinal()
      && isInHard( i, event )
      && event[i].colType() != 0
      && checkAgainstCut(event[i]) ){
      bool isDecayProduct = false;
      for(int j=0; j < int(ewResonancePos.size()); ++j)
        if ( event[i].isAncestor( ewResonancePos[j]) )
          isDecayProduct = true;
      // Except for e+e- -> jets, do not check radiation in resonance decays.
      if ( !isDecayProduct
        || getProcessString().compare("e+e->jj") == 0
        || getProcessString().compare("e+e->(z>jj)") == 0 )
        FinalPartPos.push_back(i);
    }

    // Include photons into the tms definition for "weak+QCD merging".
    if ( getProcessString().find("Ainc") != string::npos
      && event[i].isFinal() && event[i].idAbs() == 22 )
      FinalPartPos.push_back(i);
    // Include Z-bosons into the tms definition for "weak+QCD merging".
    if ( getProcessString().find("Zinc") != string::npos
      && event[i].isFinal() && event[i].idAbs() == 23 )
      FinalPartPos.push_back(i);
    // Include W-bosons into the tms definition for "weak+QCD merging".
    if ( getProcessString().find("Winc") != string::npos
      && event[i].isFinal() && event[i].idAbs() == 24 )
      FinalPartPos.push_back(i);
  }

  // Get index of first incoming
  int in1 = 0;
  for (int i=0; i < event.size(); ++i)
    if (abs(event[i].status()) == 41 ){
      in1 = i;
      break;
    }

  // Get index of second incoming
  int in2 = 0;
  for (int i=0; i < event.size(); ++i)
    if (abs(event[i].status()) == 42 ){
      in2 = i;
      break;
    }

  // If no incoming of the cascade are found, try incoming
  if (in1 == 0 || in2 == 0){
    // Find current incoming partons
    for(int i=3; i < int(event.size()); ++i){
      if ( !isInHard( i, event ) ) continue;
      if (event[i].mother1() == 1) in1 = i;
      if (event[i].mother1() == 2) in2 = i;
    }
  }

  int nInitialPartons(0), nFinalOther(0);
  for ( int i = 0; i < event.size(); ++i ) {
    if ( (event[i].mother1() == 1 || event[i].mother1() == 2 )
      && (event[i].idAbs()   < 6  || event[i].id()      == 21) )
      nInitialPartons++;
    if (event[i].isFinal() && event[i].idAbs() >= 6 && event[i].id() != 21)
      nFinalOther++;
  }
  bool is2to2QCD = ( int(FinalPartPos.size()) == 2 && nInitialPartons == 2
                  && nFinalOther   == 0 );

  // For pure QCD set the cut to the pT of the dijet system.
  if (is2to2QCD) {
    double pt12 = min(event[FinalPartPos[0]].pT(),
                      event[FinalPartPos[1]].pT());
    return pt12;
  }

  // No cut if only massive particles in final state.
  int nLight(0);
  for(int i=0; i < int(FinalPartPos.size()); ++i)
    if (  event[FinalPartPos[i]].idAbs() <= 5
      ||  event[FinalPartPos[i]].idAbs() == 21
      ||  event[FinalPartPos[i]].idAbs() == 22) nLight++;
  if (nLight == 0) return 0.;


  // Find minimal pythia pt in event
  double ptmin = event[0].e();
  for(int i=0; i < int(FinalPartPos.size()); ++i){

    double pt12  = ptmin;

    // Compute II separation of i-emission and first incoming as radiator
    if (event[in1].colType() != 0) {
      double temp = rhoPythia( event, in1, FinalPartPos[i], in2, 0);
      pt12 = min(pt12, temp);
    }

    // Compute II separation of i-emission and second incoming as radiator
    if ( event[in2].colType() != 0) {
      double temp = rhoPythia( event, in2, FinalPartPos[i], in1, 0);
      pt12 = min(pt12, temp);
    }

    // Compute all IF separations of i-emission and first incoming as radiator
    if ( event[in1].colType() != 0 ) {
      for(int j=0; j < int(FinalPartPos.size()); ++j) {
        // Allow both initial partons as recoiler
        if ( i != j ){
          // Check with first initial as recoiler
          double temp = rhoPythia(event,in1,FinalPartPos[i],FinalPartPos[j],0);
          pt12 = min(pt12, temp);
        }
      }
    }

    // Compute all IF separations of i-emission and second incoming as radiator
    if ( event[in2].colType() != 0 ) {
      for(int j=0; j < int(FinalPartPos.size()); ++j) {
        // Allow both initial partons as recoiler
        if ( i != j ){
          // Check with first initial as recoiler
          double temp = rhoPythia(event,in2,FinalPartPos[i],FinalPartPos[j],0);
          pt12 = min(pt12, temp);
        }
      }
    }

    // Compute all FF separations between final state partons.
    for(int j=0; j < int(FinalPartPos.size()); ++j) {
      for(int k=0; k < int(FinalPartPos.size()); ++k) {
        // Allow any parton as recoiler
        if ( (i != j) && (i != k) && (j != k) ){
          double temp = rhoPythia( event, FinalPartPos[i], FinalPartPos[j],
                            FinalPartPos[k], 0);
          pt12 = min(pt12, temp);
          temp        = rhoPythia( event, FinalPartPos[j], FinalPartPos[i],
                            FinalPartPos[k], 0);
          pt12 = min(pt12, temp);
        }
      }
    }

    // Compute pythia FSR separation between two jets, with eith initial as
    // recoiler.
    if ( event[in1].colType() != 0 ) {
      for(int j=0; j < int(FinalPartPos.size()); ++j) {
        // Allow both initial partons as recoiler
        if ( i != j ){
          // Check with first initial as recoiler
          double temp = rhoPythia( event, FinalPartPos[i],FinalPartPos[j],
                                   in1, 0);
          pt12 = min(pt12, temp);
        }
      }
    }

    // Compute pythia FSR separation between two jets, with eith initial as
    // recoiler.
    if ( event[in2].colType() != 0) {
      for(int j=0; j < int(FinalPartPos.size()); ++j) {
        // Allow both initial partons as recoiler
        if ( i != j ){
          // Check with second initial as recoiler
          double temp = rhoPythia( event, FinalPartPos[i],FinalPartPos[j],
                                   in2, 0);
          pt12 = min(pt12, temp);
        }
      }
    }

    // Reset minimal y separation
    ptmin = min(ptmin,pt12);
  }

  return ptmin;

}

//--------------------------------------------------------------------------

// Function to compute "pythia pT separation" from Particle input, as a helper
// for rhoms(...).

double DireMergingHooks::rhoPythia(const Event& event, int rad, int emt,
  int rec, int){

  // Use external shower for merging.
  // Ask showers for evolution variable.
  map<string,double> stateVars;
  double ptret = event[0].m();
  bool isFSR = showers->timesPtr->isTimelike(event, rad, emt, rec, "");
  if (isFSR) {
    vector<string> name = showers->timesPtr->getSplittingName
      (event, rad, emt, rec);
    for (int i=0; i < int(name.size()); ++i) {
      stateVars = showers->timesPtr->getStateVariables
        (event, rad, emt, rec, name[i]);
      double pttemp = ptret;
      if (stateVars.size() > 0 && stateVars.find("t") != stateVars.end())
        pttemp = sqrt(stateVars["t"]);
      ptret = min(ptret,pttemp);
    }
  } else {
    vector<string> name = showers->spacePtr->getSplittingName
      (event, rad, emt, rec);
    for (int i=0; i < int(name.size()); ++i) {
      stateVars = showers->spacePtr->getStateVariables
        (event, rad, emt, rec, name[i]);
      double pttemp = ptret;
      if (stateVars.size() > 0 && stateVars.find("t") != stateVars.end())
        pttemp = sqrt(stateVars["t"]);
      ptret = min(ptret,pttemp);
    }
  }
  return ptret;

}

//--------------------------------------------------------------------------

// Function to find a colour (anticolour) index in the input event.
// Helper for rhoms
// IN  int col       : Colour tag to be investigated
//     int iExclude1 : Identifier of first particle to be excluded
//                     from search
//     int iExclude2 : Identifier of second particle to be excluded
//                     from  search
//     Event event   : event to be searched for colour tag
//     int type      : Tag to define if col should be counted as
//                      colour (type = 1) [->find anti-colour index
//                                         contracted with col]
//                      anticolour (type = 2) [->find colour index
//                                         contracted with col]
// OUT int           : Position of particle in event record
//                     contraced with col [0 if col is free tag]

int DireMergingHooks::findColour(int col, int iExclude1, int iExclude2,
      const Event& event, int type, bool isHardIn){

  bool isHard = isHardIn;
  int index = 0;

  if (isHard){
    // Search event record for matching colour & anticolour
    for(int n = 0; n < event.size(); ++n) {
      if ( n != iExclude1 && n != iExclude2
        && event[n].colType() != 0
        &&(   event[n].status() > 0          // Check outgoing
           || event[n].status() == -21) ) {  // Check incoming
         if ( event[n].acol() == col ) {
          index = -n;
          break;
        }
        if ( event[n].col()  == col ){
          index =  n;
          break;
        }
      }
    }
  } else {

    // Search event record for matching colour & anticolour
    for(int n = 0; n < event.size(); ++n) {
      if (  n != iExclude1 && n != iExclude2
        && event[n].colType() != 0
        &&(   event[n].status() == 43        // Check outgoing from ISR
           || event[n].status() == 51        // Check outgoing from FSR
           || event[n].status() == 52        // Check outgoing from FSR
           || event[n].status() == -41       // first initial
           || event[n].status() == -42) ) {  // second initial
        if ( event[n].acol() == col ) {
          index = -n;
          break;
        }
        if ( event[n].col()  == col ){
          index =  n;
          break;
        }
      }
    }
  }
  // if no matching colour / anticolour has been found, return false
  if ( type == 1 && index < 0) return abs(index);
  if ( type == 2 && index > 0) return abs(index);

  return 0;
}

//==========================================================================

} // end namespace Pythia8
