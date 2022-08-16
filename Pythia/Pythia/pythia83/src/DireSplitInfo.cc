// DireSplitInfo.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the Dire classes
// use as storage containers for splitting information.

#include "Pythia8/DireSplitInfo.h"

namespace Pythia8 {

//==========================================================================

// Definition of DireSplitKinematics members.

void DireSplitKinematics::store ( const DireSplitKinematics& k) {
  m2Dip     = k.m2Dip;
  pT2       = k.pT2;
  pT2Old    = k.pT2Old;
  z         = k.z;
  phi       = k.phi;
  sai       = k.sai;
  xa        = k.xa;
  phi2      = k.phi2;
  m2RadBef  = k.m2RadBef;
  m2Rec     = k.m2Rec;
  m2RadAft  = k.m2RadAft;
  m2EmtAft  = k.m2EmtAft;
  m2EmtAft2 = k.m2EmtAft2;
  xBef      = k.xBef;
  xAft      = k.xAft;
}

void DireSplitKinematics::list() {
  cout << "List DireSplitKinematics:"
       << scientific << setprecision(3) << "\n"
       << " m2Dip = " << m2Dip << "\n"
       << " pT2 = "   << pT2 << "\t"
       << " z = "     << z << "\t"
       << " phi = "   << phi << "\n"
       << " sai = "   << sai << "\t"
       << " xa = "    << xa << "\t"
       << " phi2 = "  << phi2 << "\n"
       << " m2RadBef = "   << m2RadBef << " "
       << " m2Rec = "      << m2Rec << " "
       << " m2RadAft = "   << m2RadAft << " "
       << " m2EmtAft = "   << m2EmtAft << " "
       << " m2EmtAft2t = " << m2EmtAft2 << "\n";
}

//==========================================================================

// Definition of DireSplitInfo members.

void DireSplitInfo::init(const Event& state) {
  if (iRadBef>0) particleSave.push_back(DireSplitParticle(state[iRadBef]));
  else           particleSave.push_back(DireSplitParticle());
  if (iRecBef>0) particleSave.push_back(DireSplitParticle(state[iRecBef]));
  else           particleSave.push_back(DireSplitParticle());
  if (iRadAft>0) particleSave.push_back(DireSplitParticle(state[iRadAft]));
  else           particleSave.push_back(DireSplitParticle());
  if (iRecAft>0) particleSave.push_back(DireSplitParticle(state[iRecAft]));
  else           particleSave.push_back(DireSplitParticle());
  if (iEmtAft>0) particleSave.push_back(DireSplitParticle(state[iEmtAft]));
  else           particleSave.push_back(DireSplitParticle());
  if (iEmtAft2>0)particleSave.push_back(DireSplitParticle(state[iEmtAft2]));
  else           particleSave.push_back(DireSplitParticle());
}

void DireSplitInfo::store (const DireSplitInfo& s) {
  clear();
  kinSave.clear();
  particleSave.resize(0);
  extras.clear();
  iRadBef = s.iRadBef;
  iRecBef = s.iRecBef;
  iRadAft = s.iRadAft;
  iRecAft = s.iRecAft;
  iEmtAft = s.iEmtAft;
  iEmtAft2 = s.iEmtAft2;
  for (int i=0; i < int(s.particleSave.size()); ++i)
    particleSave.push_back(s.particleSave[i]);
  kinSave.store(s.kinSave);
  side = s.side;
  type = s.type;
  system = s.system;
  systemRec = s.systemRec;
  splittingSelName = s.splittingSelName;
  for ( unordered_map<string,double>::const_iterator it = s.extras.begin();
    it != s.extras.end(); ++it )
    extras.insert(make_pair(it->first,it->second));
  useForBranching = s.useForBranching;
  terminateEvolution = s.terminateEvolution;
  iSiblings       = s.iSiblings;
}

void DireSplitInfo::save () {
  kinSaveStore = kinSave;
  particleSaveStore = particleSave;
  extrasStore = extras;
  iRadBefStore = iRadBef;
  iRecBefStore = iRecBef;
  iRadAftStore = iRadAft;
  iRecAftStore = iRecAft;
  iEmtAftStore = iEmtAft;
  iEmtAft2Store = iEmtAft2;
  sideStore = side;
  typeStore = type;
  systemStore = system;
  systemRecStore = systemRec;
  splittingSelNameStore = splittingSelName;
  useForBranchingStore = useForBranching;
  terminateEvolutionStore = terminateEvolution;
  iSiblingsStore       = iSiblings;
}

void DireSplitInfo::restore () {
  kinSave = kinSaveStore;
  particleSave = particleSaveStore;
  extras = extrasStore;
  iRadBef = iRadBefStore;
  iRecBef = iRecBefStore;
  iRadAft = iRadAftStore;
  iRecAft = iRecAftStore;
  iEmtAft = iEmtAftStore;
  iEmtAft2 = iEmtAft2Store;
  side = sideStore;
  type = typeStore;
  system = systemStore;
  systemRec = systemRecStore;
  splittingSelName = splittingSelNameStore;
  useForBranching = useForBranchingStore;
  terminateEvolution = terminateEvolutionStore;
  iSiblings       = iSiblingsStore;
}

void DireSplitInfo::storeInfo(string name, int typeIn, int systemIn,
  int systemRecIn,
  int sideIn, int iPosRadBef, int iPosRecBef,
  const Event& state, int idEmtAft,
  int idRadAft, int nEmissions, double m2Dip, double pT2, double pT2Old,
  double z, double phi, double m2Bef, double m2s, double m2r, double m2i,
  double sa1, double xa, double phia1, double m2j, double xBef, double xAft) {
  clear();
  storeName(name);
  storeType(typeIn);
  storeSystem(systemIn);
  storeSystemRec(systemRecIn);
  storeSide(sideIn);
  storeRadRecBefPos(iPosRadBef, iPosRecBef);
  storeRadBef(state[iPosRadBef]);
  storeRecBef(state[iPosRecBef]);
  setEmtAft(idEmtAft);
  setRadAft(idRadAft);
  if (nEmissions == 2) set2to4kin( m2Dip, pT2, z, phi, sa1, xa, phia1,
    m2Bef, m2s, m2r, m2i, m2j);
  else set2to3kin( m2Dip, pT2, z, phi, m2Bef, m2s, m2r, m2i);
  storeExtras(
    unordered_map<string,double>(create_unordered_map<string,double>
    ("iRadBef",iPosRadBef)("iRecBef",iPosRecBef)("idRadAft",idRadAft)) );
  set_pT2Old(pT2Old);
  set_xBef(xBef);
  set_xAft(xAft);
}

void DireSplitInfo::storePosAfter( int iRadAftIn, int iRecAftIn, int iEmtAftIn,
  int iEmtAft2In) {
  iRadAft = iRadAftIn;
  iRecAft = iRecAftIn;
  iEmtAft = iEmtAftIn;
  iEmtAft2 = iEmtAft2In;
}

void DireSplitInfo::clear() {
  iRadBef = iRecBef = iRadAft = iRecAft = iEmtAft = iEmtAft2 = side
    = type = system = systemRec = 0;
  splittingSelName = "";
  useForBranching = terminateEvolution = false;
  for (int i= 0; i < int(particleSave.size()); ++i) particleSave[i].clear();
  kinSave.clear();
  extras.clear();
}

void DireSplitInfo::list() {
  cout << "List DireSplitInfo: "
       << " name = " << splittingSelName << "\n"
       << " [ id(radBef)= " << radBef()->id
       << " id(recBef)= "   << recBef()->id << " ] --> "
       << " { id(radAft)= " << radAft()->id
       << " id(emtAft)= "   << emtAft()->id
       << " id(emtAft2)= "  << emtAft2()->id
       << " id(recAft)= "   << recAft()->id
       << " } \n";
  kinSave.list();
  cout << "\n";
}

//==========================================================================

// Definition of color chain members.

DireSingleColChain::DireSingleColChain(int iPos, const Event& state,
  PartonSystems* partonSysPtr) {

  int colSign = (iPos > 0) ? 1 : -1;
  iPos = abs(iPos);
  int type        = state[iPos].colType();
  int iStart      = iPos;
  int iSys        = partonSysPtr->getSystemOf(iPos,true);
  int sizeSystem  = partonSysPtr->sizeAll(iSys);
  if (!state[iPos].isFinal() || colSign < 0) type *= -1;
  addToChain(iPos, state);

  do {

    int icol      = colEnd();
    if (type < 0) icol = acolEnd();

    bool foundRad = false;
    for ( int i = 0; i < sizeSystem; ++i) {
      int j = partonSysPtr->getAll(iSys, i);
      if ( j == iPos || state[j].colType() == 0) continue;
      if (!state[j].isFinal()
        && state[j].mother1() != 1
        && state[j].mother1() != 2) continue;
      int jcol = state[j].col();
      int jacl = state[j].acol();
      if ( type < 0) swap(jcol,jacl);
      if ( !state[j].isFinal()) swap(jcol,jacl);
      if ( jacl == icol ) {
        iPos = j;
        foundRad = true;
        break;
      }
    }
    // Found next color index in evolving system.
    if (foundRad) addToChain( iPos, state);
    // Try to use find next color index in system of mother (if
    // different), and then exit.
    else {
      bool foundMotRad = false;

      // Try to find incoming particle in other systems, i.e. if the current
      // system arose from a resonance decay.
      int sizeSys     = partonSysPtr->sizeSys();
      int in1=0;
      int iParentInOther = 0;
      int nSys = partonSysPtr->sizeAll(iSys);
      for (int iInSys = 0; iInSys < nSys; ++iInSys){
        int iNow = partonSysPtr->getAll(iSys,iInSys);
        for (int iOtherSys = 0; iOtherSys < sizeSys; ++iOtherSys){
          if (iOtherSys == iSys) continue;
          int nOtherSys = partonSysPtr->sizeAll(iOtherSys);
          for (int iInOtherSys = 0; iInOtherSys < nOtherSys; ++iInOtherSys){
            int iOtherNow = partonSysPtr->getAll(iOtherSys,iInOtherSys);
            if (state[iNow].isAncestor(iOtherNow)) {
              iParentInOther = iOtherNow;
            }
          }
        }
      }
      in1 = iParentInOther;

      int jcol = state[in1].col();
      int jacl = state[in1].acol();
      if ( !state[in1].isFinal() || type < 0) swap(jcol,jacl);
      if ( !state[in1].isFinal() && type < 0) swap(jcol,jacl);
      if ( jacl == icol ) {
        iPos = in1;
        foundMotRad = true;
      }

      if (foundMotRad) { addToChain( iPos, state); break;}
    }

  } while ( abs(state[iPosEnd()].colType()) != 1 && iPosEnd() != iStart );
  if (iPosEnd() == iStart) chain.pop_back();

}

void DireSingleColChain::addToChain(const int iPos, const Event& state){
  int col = state[iPos].col();
  int acl = state[iPos].acol();
  original_chain.push_back( make_pair(iPos, make_pair(col, acl)));
  if ( !state[iPos].isFinal()) swap(col,acl);
  chain.push_back( make_pair(iPos, make_pair(col, acl)));
}

bool DireSingleColChain::isInChain( int iPos) {
  for (int i=0; i< size(); ++i)
    if (chain[i].first == iPos) return true;
  return false;
}

int DireSingleColChain::posInChain( int iPos) {
  for (int i=0; i< size(); ++i)
    if (chain[i].first == iPos) return i;
  return -1;
}

bool DireSingleColChain::colInChain( int col) {
  for (int i=0; i< size(); ++i)
    if ( chain[i].second.first  == col
      || chain[i].second.second == col) return true;
  return false;
}

DireSingleColChain DireSingleColChain::chainFromCol(int iPos, int col,
  int nSteps, const Event& state) {
  DireSingleColChain ret;
  int iSteps = 0;
  int iPosInChain = posInChain(iPos);

  // For gluon, just match both colors.
  if (state[iPos].id() == 21) {

    if (iPosInChain == 0) {

      ret.addToChain(chain[iPosInChain].first, state);
      if ( iPosInChain+1 < size() && chain[iPosInChain+1].first > 0
        && !ret.isInChain(chain[iPosInChain+1].first))
        ret.addToChain(chain[iPosInChain+1].first, state);
      if ( iPosInChain+2 < size() && chain[iPosInChain+2].first > 0
        && !ret.isInChain(chain[iPosInChain+2].first))
        ret.addToChain(chain[iPosInChain+2].first, state);

    } else if (iPosInChain == size()-1) {

      if ( iPosInChain-2 >= 0
        && chain[iPosInChain-2].first > 0
        && !ret.isInChain(chain[iPosInChain-2].first))
        ret.addToChain(chain[iPosInChain-2].first, state);
      if ( iPosInChain-1 >= 0 && iPosInChain-1 < size()
        && chain[iPosInChain-1].first > 0
        && !ret.isInChain(chain[iPosInChain-1].first))
        ret.addToChain(chain[iPosInChain-1].first, state);
      ret.addToChain(chain[iPosInChain].first, state);

    } else {

      if ( iPosInChain-1 >= 0 && iPosInChain-1 < size()
        && chain[iPosInChain-1].first > 0
        && !ret.isInChain(chain[iPosInChain-1].first))
        ret.addToChain(chain[iPosInChain-1].first, state);
      if ( iPosInChain   >= 0 && iPosInChain   < size()
        && chain[iPosInChain].first   > 0
        && !ret.isInChain(chain[iPosInChain].first))
        ret.addToChain(chain[iPosInChain].first, state);
      if ( iPosInChain+1 < size() && chain[iPosInChain+1].first > 0
        && !ret.isInChain(chain[iPosInChain+1].first))
        ret.addToChain(chain[iPosInChain+1].first, state);
    }
    return ret;
  }

  // Loop through, find color and attach subsequent particles to chain.
  for (int i=0; i< size(); ++i) {
    if ( iSteps == 0 && size() - 1 - i > nSteps
      && chain[i].second.first  != col
      && chain[i].second.second != col) continue;
    iSteps++;
    if (chain[i].first > 0 && !ret.isInChain(chain[i].first))
      ret.addToChain(chain[i].first, state);
    if (iSteps > nSteps) break;
  }

  // Done
  return ret;
}

string DireSingleColChain::listPos() const {
  ostringstream os;
  for (int i=0; i< size(); ++i) os << " "  << chain[i].first;
  return os.str();
}

// List functions by N. Fischer.
void DireSingleColChain::print() const {
  int i    = 0;
  int length = size();
  int max  = length;

  // first line: positions
  for (i=0; i<length; i++) {
    cout << setw(i == 0 ? 5 : 10) << chain[i].first;
  }
  cout << endl;
  // second line: color-anticolor connections (horizontal)
  i   = 0;
  max = (length%2 == 0 ? length : length-1);
  while (i < max) {
    if (i == 0) cout << "  ";
    if (i < max-1) cout << (i%2 == 0 ? " _____________" : "      ");
    i++;
  }
  cout << endl;
  // third line: color-anticolor connections (vertical)
  i = 0;
  while (i < max) {
    if (i == 0) cout << "  ";
    cout << "|";
    if (i < max-1) cout << (i%2 == 0 ? "             " : "     ");
    i++;
  }
  cout << endl;
  // fourth line: colors and anticolors
  for (i=0; i<length; i++) {
    cout << setw(4) << chain[i].second.first;
    cout << setw(4) << chain[i].second.second;
    cout << "  ";
  }
  cout << endl;
  // fifth line: color-anticolor connections (vertical & horizontal)
  i   = 0;
  max = (length%2 == 0 ? length-2 : length-1);
  while (i < max) {
    if (i == 0) cout << "            ";
    cout << "|";
    if (i < max-1) cout << (i%2 == 0 ? "_____________" : "     ");
    i++;
  }
  cout << endl;
  // sixth line: if first gluon is connected to last gluon
  max = 10*(length-1)-5;
  if (chain[0].second.second == chain[length-1].second.first
    && chain[0].second.second != 0) {
    cout << "      |";
    for (i=0; i<max; i++) cout << "_";
    cout << "|";
  }
  cout << endl;
}

// List functions by N. Fischer.
void DireSingleColChain::list () const {
  if (size() > 0) cout << " ";
  for (int i=0; i<size(); i++) {
    cout << "[" << chain[i].second.second << "]";
    cout << " " << chain[i].first << " ";
    cout << "(" << chain[i].second.first << ")";
    if (i < size()-1) cout << " --- ";
  }
  cout << endl;
}

// List functions by N. Fischer.
string DireSingleColChain::list2 () const {
  ostringstream os;
  if (size() > 0) os << " ";
  for (int i=0; i<size(); i++) {
    os << "[" << chain[i].second.second << "]";
    os << " " << chain[i].first << " ";
    os << "(" << chain[i].second.first << ")";
    if (i < size()-1) os << " --- ";
  }
  return os.str();
}

//==========================================================================

// Definition of DireColChain members

DireSingleColChain DireColChains::chainOf (int iPos) {
  for (int i=0; i< size(); ++i)
    if ( chains[i].isInChain(iPos) ) return chains[i];
  return DireSingleColChain();
}

DireSingleColChain DireColChains::chainFromCol (int iPos, int col, int nSteps,
  const Event& state) {
  for (int i=0; i< size(); ++i) {
    if ( chains[i].colInChain(col) ) {
      return chains[i].chainFromCol(iPos, col, nSteps, state);}
  }
  return DireSingleColChain();
}

int DireColChains::check(int iSys, const Event& state,
  PartonSystems* partonSysPtr) {

  int sizeSystem = partonSysPtr->sizeAll(iSys);
  int nFinal     = 0;
  for ( int i = 0; i < sizeSystem; ++i) {
     int j = partonSysPtr->getAll(iSys, i);
     if (!state[j].isFinal())      continue;
     nFinal++;
     if ( state[j].colType() == 0) continue;
     if ( chainOf(j).size() < 2)   return j;
  }

  for ( int i = 0; i < sizeSystem; ++i) {
    int j = partonSysPtr->getAll(iSys, i);
    if ( state[j].colType() == 0)                            continue;
    if ( state[j].mother1() != 1 && state[j].mother1() != 2) continue;
    if ( nFinal > 0 && chainOf(j).size() < 2)                return j;
  }

  // Done.
  return -1;

}

void DireColChains::list() {
  cout << "\n --------- Begin DIRE Color Chain Listing  -----------------"
       << "--------------------------------------------------------------"
       << "----------" << endl << endl;

  for (int i=0; i < size(); ++i){
    cout << " Chain " << setw(4) << i << "\n" << endl;
    chains[i].print();
    if (i < size()-1)
    cout << " **********************************************************"
         << "***********************************************************"
         << "**************" << endl;
  }
  // Done.
  cout << " ----------  End DIRE Color Chain Listing  -----------------"
       << "--------------------------------------------------------------"
       << "----------" << endl;

}

//==========================================================================

} // end namespace Pythia8
