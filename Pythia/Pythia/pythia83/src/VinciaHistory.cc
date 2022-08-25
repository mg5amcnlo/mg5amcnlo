// VinciaHistory.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the VinciaHistory class
// and auxiliary classes.

// Include the Vincia headers.
#include "Pythia8/VinciaHistory.h"

namespace Pythia8 {

using namespace VinciaConstants;

//==========================================================================

// The ColourFlow class.

//--------------------------------------------------------------------------

// Method to initialise neutral(/plus/minus) chains.

void ColourFlow::addChain(int charge, int flavStart, int flavEnd,
  bool hasInitialIn) {

  if (abs(charge)<2) {
    ++nChains;
    int iChain=nChains-1;

    // Save info about this chain.
    chainStartToFlav[iChain]=flavStart;
    chainEndToFlav[iChain]=flavEnd;
    hasInitial[iChain]=hasInitialIn;
    chainToCharge[iChain]=charge;
    bool FC = ( abs(flavStart) != abs(flavEnd) ) ? true : false;

    // Index (to look up pseudochain).
    // Base Index = 2^iChain.
    int baseIndex = pow(2,iChain);

    // ChargeIndex = 0,1,2,3.
    int chargeIndex = getChargeIndex(charge,FC);
    int indexSav = baseIndex*4 + chargeIndex;

    // Create a new pseudochain with a single entry.
    PseudoChain newps;
    newps.chainlist.push_back(iChain);
    newps.hasInitial = hasInitialIn;
    newps.charge = charge;
    newps.flavStart = flavStart;
    newps.flavEnd = flavEnd;
    newps.index = baseIndex;
    newps.cindex = chargeIndex;

    // Save this pseudochain using chargeIndex and baseIndex.
    pseudochains[indexSav] = vector<PseudoChain>(1,newps);
    // Increment counter.
    countChainsByChargeIndex[chargeIndex]++;
    // Save id.
    pseudochainIDs.push_back(getID(newps));

    // Keep track of indices.
    vector<int> usedIndices(1,indexSav);

    // Insert in all possible ways into existing pseudochains.
    // (But skip this step if gluon loop).
    if (abs(flavStart) != 21 && abs(flavEnd) != 21) {

      // Loop over all existing pseudochains.
      for (int index=1; index<baseIndex; ++index) {
        int chainsIndexNew = baseIndex + index;

        // Insert in all possible ways.
        vector<int> newIndices;
        for(int icharge=0; icharge<4; ++icharge) {
          int oldIndex = index*4+icharge;
          addChain(oldIndex,chainsIndexNew,iChain,newIndices);
        }

        // Did we create any new pseudochains?
        if (newIndices.size()!=0) {

          // Save as used.
          usedIndices.insert(usedIndices.end(),newIndices.begin(),
            newIndices.end());

          // Check which other chains were in this pseudochain.
          int checkChain=0;
          int chainsIndexOld = index;
          // Convert index to binary.
          while (chainsIndexOld>0) {
            // Odd at this power?
            // ==> contains ichain
            if (chainsIndexOld%2==1) {
              chainToIndices[checkChain].insert(
                chainToIndices[checkChain].end(),
                newIndices.begin(),newIndices.end());
            }
            // Go to next power.
            checkChain++;
            chainsIndexOld = (chainsIndexOld-chainsIndexOld%2)/2;
          }
        }
      }// End loop over existing pseudochains.
    }

    // For flavour-changing chains, try to make new connections.
    if (abs(flavStart) != abs(flavEnd) && iChain > 1
      && usedIndices.size() > 1) {
      // Loop over all existing chains.
      for (int iChainOld(0); iChainOld<iChain; ++iChainOld) {
        int baseIndexOld = pow(2, iChainOld);
        // Insert into new pseudochains in all possible ways.
        vector<int> newIndices;
        for (int iNew(1); iNew<(int)usedIndices.size(); ++iNew) {
          int oldIndex = usedIndices.at(iNew);
          // Don't add a chain to a pseudochain that already contains it.
          bool skip = false;
          for (PseudoChain& psc : pseudochains[oldIndex]) {
            auto it = find(psc.chainlist.begin(),
              psc.chainlist.end(), iChainOld);
            if (it != psc.chainlist.end()) {
              skip = true;
              break;
            }
          }
          if (skip) continue;
          int baseIndexNew = pseudochains[oldIndex].at(0).index + baseIndexOld;
          addChain(oldIndex,baseIndexNew,iChainOld,newIndices);
        }

        // Did we create any new pseudochains?
        if (newIndices.size()!=0) {
          // Save as used.
          usedIndices.insert(usedIndices.end(),newIndices.begin(),
            newIndices.end());

          // Check which other chains were in this pseudochain.
          int checkChain=0;
          int chainsIndexOld = baseIndexOld;
          // Convert index to binary.
          while (chainsIndexOld>0) {
            // Odd at this power?
            // ==> contains ichain
            if (chainsIndexOld%2==1) {
              chainToIndices[checkChain].insert(
                chainToIndices[checkChain].end(),
                newIndices.begin(),newIndices.end());
            }
            // Go to next power.
            checkChain++;
            chainsIndexOld = (chainsIndexOld-chainsIndexOld%2)/2;
          }
        }
      }// End loop over existing pseudochains.
    }

    // Save all pseudochain indices which this chain is in.
    chainToIndices[iChain] = usedIndices;
  }
  // Done.
}

//--------------------------------------------------------------------------

// Select a chains.

void ColourFlow::selectResChains(int index,int iorder,int id) {

  // Fetch the selected pseudochain.
  if (pseudochains.find(index)==pseudochains.end()) return;
  else if (iorder >= int(pseudochains[index].size())) return;

  // Save selection.
  resChains[id].push_back(pseudochains[index].at(iorder));

  // Update counters.
  countResByChargeIndex[index%4]--;
  nRes--;

  // Remove selected chains from list of available ones.
  selectPseudochain(resChains[id].back().chainlist);

}

//--------------------------------------------------------------------------

// Add beam chains.

void ColourFlow::selectBeamChains(int index,int iorder) {

  // Fetch the selected pseudochain.
  if (pseudochains.find(index)==pseudochains.end()) return;
  else if (iorder>= int(pseudochains[index].size())) return;

  // Save selection.
  beamChains.push_back(pseudochains[index].at(iorder));

  // Remove selected chains from list of available ones.
  selectPseudochain(beamChains.back().chainlist);

}

//--------------------------------------------------------------------------

// Initialise from hard process information.

bool ColourFlow::initHard(map<int, map<int,int> >& countRes,
  shared_ptr<VinciaMergingHooks> vinMergingHooksPtr) {

  if (!vinMergingHooksPtr->hasSetColourStructure()) return false;

  // Get hadronically decaying resonances.
  vector<int> resPlusHard = vinMergingHooksPtr->getResPlusHad();
  vector<int> resMinusHard = vinMergingHooksPtr->getResMinusHad();
  vector<int> resNeutralFCHard = vinMergingHooksPtr->getResNeutralFCHad();
  vector<int> resNeutralFNHard = vinMergingHooksPtr->getResNeutralFNHad();

  // Count each type of resonance (need to know how many identical)
  // and initialise resChains map.
  addResonances(resPlusHard,countRes,1,true);
  addResonances(resMinusHard,countRes,-1,true);
  addResonances(resNeutralFCHard,countRes,0,true);
  addResonances(resNeutralFNHard,countRes,0,false);
  nBeamChainsMin = vinMergingHooksPtr->getNChainsMin();
  nBeamChainsMax = vinMergingHooksPtr->getNChainsMax();
  return true;

}

//--------------------------------------------------------------------------

// Check we have enough chains.

bool ColourFlow::checkChains(int cIndex) {
  if (!checkChains()) return false;
  int nchains = countChainsByChargeIndex[cIndex];
  int nres = countResByChargeIndex[cIndex];
  if ( nres > nchains ) return false;
  else return true;
}

//--------------------------------------------------------------------------

// Check we have enough chains.

bool ColourFlow::checkChains() {
  // How many still left to assign?
  int nMinNow = max(0,int(nBeamChainsMin-beamChains.size()));
  nMinNow+=nRes;
  // Do we have enough?
  if ( nMinNow > int(chainToIndices.size()) ) return false;
  else return true;
}

//--------------------------------------------------------------------------

// Return number of unassigned chains.

int ColourFlow::getNChainsLeft() {return int(chainToIndices.size());}

//--------------------------------------------------------------------------

// Return maximum length next pseudochain selection.

int ColourFlow::maxLength() {
  // How many beam chains still to select.
  int nBeamsLeft = nBeamChainsMin-beamChains.size();
  int nLeft = nBeamsLeft+nRes-1;
  return int(chainToIndices.size())-nLeft;
}

//--------------------------------------------------------------------------

// Return minimum length next pseudochain selection.

int ColourFlow::minLength() {
  if (nRes==0 && (nBeamChainsMax-beamChains.size())==1 ) {
    return int(chainToIndices.size());
  }
  else if (nBeamChainsMax==0 && nRes==1) return int(chainToIndices.size());
  else return 1;
}

//--------------------------------------------------------------------------

// Print the colour flow.

void ColourFlow::print(bool printpsch) {

  string tab =" ";
  cout<<endl
      <<tab<<"------  Colour Flow Summary  --------------------------------"
      <<"------------------------------------------------------------------"
      <<endl;
  cout<<endl<<tab<<tab<<"Unassigned chains:"<<endl;
  int countNFC = countChainsByChargeIndex[getChargeIndex(0,true)];
  int countNFN = countChainsByChargeIndex[getChargeIndex(0,false)];
  int countPlus = countChainsByChargeIndex[getChargeIndex(1,true)];
  int countMinus  = countChainsByChargeIndex[getChargeIndex(-1,true)];
  int sum = countNFC + countNFN + countPlus + countMinus;
  cout<< tab << "  Total chains: " << chainToIndices.size() << "\n"
      << tab << "  Total pseudochains: " << sum << "\n"
      << tab << "  Neutral FC pseudochains: " << countNFC << "\n"
      << tab << "  Neutral FN pseudochains: " << countNFN << "\n"
      << tab << "  Positive charge pseudochains: " << countPlus << "\n"
      << tab << "  Negative charge pseudochains: " << countMinus << "\n";
  if (printpsch) {
    cout << tab << "  All pseudochains: \n";
    for(auto it = pseudochains.begin(); it!= pseudochains.end(); ++it) {
      cout<<tab<<"    Index = "<< it->first;
      auto kit = it->second.begin();
      auto kitEnd = it->second.end();
      cout<< " charge = "<< kit->charge
          << " hasInitial = "<< kit->hasInitial
          << " nOrderings = "<< it->second.size()
          << " Chains: ";
      for( ; kit!=kitEnd; ++kit) {
        cout << "(";
        auto jit=kit->chainlist.begin();
        auto jitEnd=kit->chainlist.end();
        for( ; jit!=jitEnd; ++jit) cout<<" "<< *jit;
        cout << " ) ";
      }
      cout << "\n";
    }
  }
  cout << endl << tab << tab << "Unassigned resonances: " << nRes << endl;
  cout << endl << tab << tab << "Assigned chains:" << endl;
  int nResChains = 0;
  stringstream ss;
  for(auto itRes = resChains.begin(); itRes != resChains.end(); ++itRes) {
    int nIdentical((itRes->second).size()), nChainsNow(0);
    for(auto psch = itRes->second.begin(); psch != itRes->second.end();
        ++psch) nChainsNow += psch->chainlist.size();
    nResChains += nChainsNow;
    cout << tab << "  ID: " << itRes->first
         << ": # identical = " << nIdentical
         << " # chains = " << nChainsNow<<endl;
  }
  cout << tab << "  Total resonance chains: " << nResChains << endl;
  int nBeamChains = 0;
  for(auto itBeam = beamChains.begin(); itBeam != beamChains.end(); ++itBeam)
    nBeamChains += itBeam->chainlist.size();
  cout << tab << "  Beam chains: " << nBeamChains << "\n"
       << tab << "  Total: " << nResChains + nBeamChains << "\n\n"
       << tab << "------------------------------------------------------------"
       << "-----------------------------------------------------------------"
       << "\n\n";

}

//--------------------------------------------------------------------------

// Add a chain.

void ColourFlow::addChain(int oldIndex, int chainsIndex, int iChain,
  vector<int>& newIndices) {

  if (pseudochains.find(oldIndex) != pseudochains.end()) {
    // Retrieve information about the new chain.
    int flavBeg = chainStartToFlav[iChain];
    int flavEnd = chainEndToFlav[iChain];
    int charge = chainToCharge[iChain];
    bool FC =  ( abs(flavBeg) != abs(flavEnd) ) ? true : false;

    for ( PseudoChain& pschain : pseudochains[oldIndex] ) {
      // Check the net charge of hypothetical chain.
      int chargeNew = pschain.charge + charge;
      if (abs(chargeNew) > 1) continue;

      // Don't try to add to gluon loops.
      if (abs(pschain.flavEnd) == 21
        || abs(pschain.flavStart) == 21) continue;

      // Loop over all possible non-cyclic insert positions.
      size_t last = pschain.chainlist.size();
      // Tentatively allow to insert in position 0 when one of the chains
      // is FC. Might be vetoed below.
      bool thisFC = abs(pschain.flavEnd) != abs(pschain.flavStart);
      bool bothFC = (FC && thisFC);
      size_t first = (FC || thisFC) ? 0 : 1;
      for (size_t pos = first; pos <= last; ++pos) {
        // Do the flavours match? If so we can insert.
        bool matchLeft = true;
        bool matchRight = true;
        if (pos>0) {
          // Fetch flavour at end of previous chain.
          int chainPrev = pschain.chainlist[pos-1];
          int flavPrev = chainEndToFlav[chainPrev];
          matchLeft = (flavPrev == -flavBeg) ? true : false;
        }
        if (pos != last) {
          // Fetch flavour at start of next chain.
          int chainNext = pschain.chainlist[pos];
          int flavNext = chainStartToFlav[chainNext];
          matchRight = (flavNext == -flavEnd) ? true : false;
          // Veto insertion at 0 here if new pseudochain has cyclic symmetry.
          if (pos == 0 && matchRight && !bothFC) {
            int pscFlavEnd = pschain.flavEnd;
            if (pscFlavEnd == -flavBeg) matchRight = false;
          }
        }
        // TODO check whether rotations are needed for FC matches.
        // Should be dealt with in top-level addChain() now.
        if (matchLeft && matchRight) {
          // Make a new pseudochain.
          PseudoChain newps = pschain;
          vector<int>::iterator insert= newps.chainlist.begin()+pos;

          newps.chainlist.insert(insert,iChain);
          // Save whether any chains contain an initial parton.
          newps.hasInitial = pschain.hasInitial || hasInitial[iChain];
          // Update charge.
          newps.charge = chargeNew;
          // Update end flavours.
          newps.flavStart = chainStartToFlav[newps.chainlist.front()];
          newps.flavEnd = chainEndToFlav[newps.chainlist.back()];

          // Get ID of new pseudochain.
          int newpsID = getID(newps);

          // Check if we found this already.
          auto itCheck = find(pseudochainIDs.begin(), pseudochainIDs.end(),
            newpsID);
          if (itCheck == pseudochainIDs.end()) {
            // Save ID.
            pseudochainIDs.push_back(newpsID);

            // Set indices.
            bool FCnew = !( abs(newps.flavEnd) == abs(newps.flavStart));
            int chargeIndex = getChargeIndex(chargeNew,FCnew);
            int indexNow = chainsIndex*4+chargeIndex;
            newps.cindex = chargeIndex;
            newps.index = chainsIndex;

            // Save any new indices.
            if (find(newIndices.begin(),newIndices.end(),indexNow)
              ==newIndices.end()) {
              newIndices.push_back(indexNow);
            }

            // Save pseudochain.
            if (pseudochains.find(indexNow)==pseudochains.end()) {
              pseudochains[indexNow] = vector<PseudoChain>();
            }
            pseudochains[indexNow].push_back(newps);
            // Increment counter.
            countChainsByChargeIndex[chargeIndex]++;
          }
        }
      }// End loop over insert positions in current pseudochain.

    } // Loop over all pseudochains with oldIndex.
  }
}

//--------------------------------------------------------------------------

// Remove all chains in a pseudochain from list of unassigned chains.

void ColourFlow::selectPseudochain(vector<int>& psch) {
  for(auto it = psch.begin() ; it!= psch.end(); ++it) selectChain(*it);}

//--------------------------------------------------------------------------

// Remove iChain from list of unassigned chains.

void ColourFlow::selectChain(int iChain) {
  if (chainToIndices.find(iChain)!= chainToIndices.end()) {
    // Remove any pseudochains that this chain is involved in.
    vector<int>::iterator it = chainToIndices[iChain].begin();
    for( ; it!= chainToIndices[iChain].end(); ++it) {
      if (pseudochains.find(*it)!=pseudochains.end()) {
        int nRm = int(pseudochains[*it].size());
        // Update counters.
        int chargeIndex = *it%4;
        countChainsByChargeIndex[chargeIndex] -= nRm;
        // Erase.
        pseudochains.erase(*it);
      }
    }
    // Now erase the entry in chainToIndices.
    chainToIndices.erase(iChain);
  }
}

//--------------------------------------------------------------------------

// Method to intialise resChains.
// Set a map to count instances of each id.

void ColourFlow::addResonances(vector<int>& idsIn,
  map<int, map<int,int> > & idCounter, int charge, bool fc) {

  // Fetch charge index and initialise.
  int cIndex = getChargeIndex(charge, fc);
  if (idCounter.find(cIndex)==idCounter.end()) {
    idCounter[cIndex]=map<int,int>();
  }

  // Loop over idsIn.
  for(unsigned int iid=0; iid<idsIn.size(); iid++) {
    int idNow = idsIn.at(iid);
    // Count how many times we have seen this id.
    if (idCounter[cIndex].find(idNow)==idCounter[cIndex].end()) {
      idCounter[cIndex][idNow]=1;
    }
    else {
      idCounter[cIndex][idNow]++;
    }
    // Increment internal counters.
    countResByChargeIndex[cIndex]++;
    nRes++;
    // Initialise map.
    if (resChains.find(idNow) == resChains.end()) {
      resChains[idNow] = vector<PseudoChain>();
    }
  }
}

//--------------------------------------------------------------------------

// Return the charge index.

int ColourFlow::getChargeIndex(int charge, bool fc) {
  if (charge==0 && !fc) { return 0;}
  else if (charge==0 && fc) { return 1;}
  else if (charge==-1) { return 2;}
  else { return 3;}
}

//==========================================================================

// The HistoryNode class.

//--------------------------------------------------------------------------

// Return how many clusterings there are.

int HistoryNode::getNClusterings(shared_ptr<VinciaMergingHooks>
  vinMergingHooksPtr, Info* infoPtr, int verboseIn) {

  // Find all possible clusterings.
  setClusterList(vinMergingHooksPtr,infoPtr, verboseIn);

  // Return the number we found.
  return clusterList.size();

}

//--------------------------------------------------------------------------

// Find all possible clusterings for current event.

void HistoryNode::setClusterList(shared_ptr<VinciaMergingHooks>
  vinMergingHooksPtr, Info* infoPtr, int verboseIn) {
  if (verboseIn >= DEBUG) printOut(__METHOD_NAME__,"begin", dashLen);

  if (!isInitPtr) {
    string msg = ": pointers were not initialised";
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    return;
  }

  // Print state to be clustered.
  if (verboseIn >= DEBUG) {
    string msg = "Setting cluster list for event:";
    printOut(__METHOD_NAME__, msg);
    state.list();
  }

  // We can only have a single clusterable chain for resonances.
  if (hasRes && clusterableChains.size() > 1) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__,
      ": More than one clusterable chain in resonance decay.");
    return;
  }

  // For hard process in VBF, we need at least two (beam) chains.
  bool doVBF = vinMergingHooksPtr->doMergeInVBF();
  if (!hasRes && doVBF && clusterableChains.size() < 2) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+
      ": Less than two quark lines in VBF process.");
    return;
  }

  // Loop over all chains.
  for (int iChain(0); iChain<(int)clusterableChains.size(); ++iChain) {
    // For VBF, need one quark pair per line.
    int nMinQQbarNow = doVBF ? 1 : nMinQQbar;

    // Fetch current chain and check whether we want to consider it.
    vector<int> clusChain = clusterableChains.at(iChain);

    // If this chain has less than three particles, nothing more to be done.
    if (clusChain.size() < 3) continue;

    // Find candidate clusterings and qqbar pairs.
    vector<VinciaClustering> candidates;
    int nQ(0), nQbar(0);

    // Count quarks and antiquarks in this chain.
    for (int iPtcl(0); iPtcl<(int)clusChain.size(); ++iPtcl) {
      if (state[clusChain.at(iPtcl)].isQuark()) {
        int colType = state[clusChain.at(iPtcl)].colType();
        if (!state[clusChain.at(iPtcl)].isFinal()) colType *=-1;
        if (colType==1) ++nQ;
        else ++nQbar;
      }
    }
    if (nQ!=nQbar) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+
        ": Number of quarks / antiquarks does not match");
      return;
    }
    int nQQbarNow = nQ;
    if (nQQbarNow<nMinQQbarNow) {
      string msg = "";
      if (verboseIn >= DEBUG)
        msg = "Expected " + to_string(nMinQQbarNow) + ", have "
          + to_string(nQQbarNow);
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+
        ": Not enough quarks in colour chains", msg, true);
      return;
    }

    // Loop over particles in current chain.
    for (int iPtcl(0); iPtcl<(int)clusChain.size(); ++iPtcl) {
      // Fetch children of this clustering.
      int child1 = (iPtcl == 0 ? clusChain.back() : clusChain[iPtcl-1]);
      int child2 = clusChain[iPtcl];
      int child3 = (iPtcl == (int)(clusChain.size()-1) ?
        clusChain.front() : clusChain[iPtcl+1]);

      if (verboseIn >= DEBUG) {
        stringstream msg;
        msg << "Candidate clustering: "
            << child1 <<" "<< child2<<" "<<child3;
        printOut(__METHOD_NAME__, msg.str());
      }

      // Don't cluster emissions into the initial state...
      if (state[child2].isFinal()) {
        // Set information about children (masses and invariants).
        VinciaClustering thisClus;
        thisClus.setChildren(state, child1, child2, child3);
        // Make sure we only add sensible clusterings to the list.
        if (vinComPtr->isValidClustering(thisClus, state, verboseIn)) {
          candidates.push_back(thisClus);
          if (verboseIn >= DEBUG) {
            stringstream msg;
            msg << "Added clustering to candidate list.";
            printOut(__METHOD_NAME__, msg.str());
          }
        }
      }
      // ...but include colour partners of initial-state quarks in GXconv...
      else if (state[child2].isQuark()) {
        // Fetch colour connection.
        bool colCon12
          = vinComPtr->colourConnected(state[child1],state[child2]);
        bool colCon23
          = vinComPtr->colourConnected(state[child2],state[child3]);
        if (state[child2].id() == state[child1].id()) {
          if (state[child1].isFinal()) {
            // Check colour connection:
            // 1 and 2 must not be connected, 2 and 3 must be connected.
            if (!colCon12 && colCon23) {
              // Convention is that j is always in the final state,
              // so we swap the order of child1 and child2.
              VinciaClustering thisClus;
              thisClus.setChildren(state, child2, child1, child3);
              // Make sure we only add sensible clusterings to the list.
              if (vinComPtr->isValidClustering(thisClus, state, verboseIn)) {
                candidates.push_back(thisClus);
                if (verboseIn >= DEBUG) {
                  stringstream msg;
                  msg << "Added clustering to candidate list.";
                  printOut(__METHOD_NAME__, msg.str());
                }
              }
            }
          }
        }
        if (state[child2].id() == state[child3].id()) {
          if (state[child3].isFinal()) {
            // Check colour connection:
            // 2 and 3 must not be connected, 1 and 2 must be connected.
            if (!colCon23 && colCon12) {
              // Convention is that j is always in the final state,
              // so we swap the order of child2 and child3.
              VinciaClustering thisClus;
              thisClus.setChildren(state, child1, child3, child2);
              // Make sure we only add sensible clusterings to the list.
              if (vinComPtr->isValidClustering(thisClus, state, verboseIn)) {
                candidates.push_back(thisClus);
                if (verboseIn >= DEBUG) {
                  stringstream msg;
                  msg << "Added clustering to candidate list.";
                  printOut(__METHOD_NAME__, msg.str());
                }
              }
            }
          }
        }
      }
      // ...and colour partners of initial-state gluons in QXsplit.
      else if (state[child2].isGluon()) {
        // If both colour-connected partners are quarks, we either encountered
        // it already or will encounter it later.
        if (state[child1].isQuark() && !state[child3].isQuark()) {
          // Convention is that j is always in the final state,
          // so we swap the order of child1 and child2.
          VinciaClustering thisClus;
          thisClus.setChildren(state, child2, child1, child3);
          // Make sure we only add sensible clusterings to the list.
          if (vinComPtr->isValidClustering(thisClus, state, verboseIn)) {
            candidates.push_back(thisClus);
            if (verboseIn >= DEBUG) {
              stringstream msg;
              msg << "Added clustering to candidate list.";
              printOut(__METHOD_NAME__, msg.str());
            }
          }
        }
        else if (!state[child1].isQuark() && state[child3].isQuark()) {
          // Convention is that j is always in the final state,
          // so we swap the order of child3 and child2.
          VinciaClustering thisClus;
          thisClus.setChildren(state, child1, child3, child2);
          // Make sure we only add sensible clusterings to the list.
          if (vinComPtr->isValidClustering(thisClus, state, verboseIn)) {
            candidates.push_back(thisClus);
            if (verboseIn >= DEBUG) {
              stringstream msg;
              msg << "Added clustering to candidate list.";
              printOut(__METHOD_NAME__, msg.str());
            }
          }
        }
      }
    }

    // Now loop over candidate clusterings.
    for (auto itClus = candidates.begin();
      itClus!=candidates.end(); ++itClus) {

      int child1 = itClus->child1;
      int child2 = itClus->child2;
      int child3 = itClus->child3;

      if (verboseIn >= DEBUG) {
        stringstream msg;
        msg << "Considering clustering: "
            << child1 <<" "<< child2<<" "<<child3;
        printOut(__METHOD_NAME__,msg.str());
      }

      // Skip clusterings of resonances.
      if (state[child2].isResonance()) {
        if (verboseIn >= DEBUG)
          printOut(__METHOD_NAME__,"Skipping resonance clustering.");
        continue;
      }

      // Find all antennae that can produce this post-branching state.
      vector<VinciaClustering> clusterings;
      clusterings = vinComPtr->findAntennae(state, child1, child2, child3);
      if (clusterings.size() == 0) {
        infoPtr->errorMsg("Error in "+__METHOD_NAME__+
          ": No antenna found. Clustering will be skipped.");
        continue;
      }

      // Loop over all histories for this post-branching state
      // and save corresponding clusterings.
      for (int iHist(0); iHist<(int)clusterings.size(); ++iHist) {
        // Fetch information about antenna function.
        bool isFSR = clusterings.at(iHist).isFSR;
        enum AntFunType antFunType = clusterings.at(iHist).antFunType;

        // Check whether we are allowed to do this clustering.
        bool quarkClustering = false;
        if (isFSR) {
          // Is FF on?
          if (clusterings.at(iHist).isFF() &&
            !vinMergingHooksPtr->canClusFF()) {
            if (verboseIn >= DEBUG) {
              printOut(__METHOD_NAME__,
                "Skipping FF clustering (turned off in shower)");
            }
            continue;
          }

          // Is RF on?
          if (clusterings.at(iHist).isRF() &&
            !vinMergingHooksPtr->canClusRF()) {
            if (verboseIn >= NORMAL) {
              printOut(__METHOD_NAME__,
                "Skipping RF clustering (turned off in shower)");
            }
            continue;
          }
          // For now warn if we do RF clusterings.
          if (clusterings.at(iHist).isRF()) {
            infoPtr->errorMsg("Warning in "+__METHOD_NAME__
              +": Performing unvalidated resonance-final clustering");
          }

          if (antFunType == GXsplitFF) quarkClustering = true;
          AntennaFunction* antPtr= antSetFSRptr->getAntFunPtr(antFunType);
          if (antPtr==nullptr) {
            if (verboseIn >= NORMAL) {
              stringstream msg;
              msg << " (antFunType = " << antFunType << ")";
              infoPtr->errorMsg("Error in "+__METHOD_NAME__+": Non-existent "
                "antenna", msg.str());
            }
            continue;
          }
          clusterings.at(iHist).kMapType = antPtr->kineMap();
        } else {
          // Is II on?
          if (clusterings.at(iHist).isII() &&
            !vinMergingHooksPtr->canClusII()) {
            if (verboseIn >= DEBUG) {
              printOut(__METHOD_NAME__,
                "Skipping II clustering (turned off in shower).");
            }
            continue;
          }

          // Is IF on?
          if (clusterings.at(iHist).isIF() &&
            !vinMergingHooksPtr->canClusIF()) {
            if (verboseIn >= DEBUG) {
              printOut(__METHOD_NAME__,
                "Skipping IF clustering (turned off in shower).");
            }
            continue;
          }

          if (antFunType == XGsplitIF || antFunType == GXconvIF ||
              antFunType == GXconvII) quarkClustering = true;
        }
        // Check if we have enough quarks left.
        if (quarkClustering && nQQbarNow == nMinQQbarNow) {
          if (verboseIn >= DEBUG)
            printOut(__METHOD_NAME__,"Skipping quark clustering");
          continue;
        }

        // Initialise vectors of invariants and masses.
        if (!clusterings.at(iHist).initInvariantAndMassVecs()) {
          if (verboseIn >= DEBUG) {
            stringstream msg;
            msg << "No phase space left for clustering."
                << " Will be skipped.";
            printOut(__METHOD_NAME__, msg.str());
          }
          continue;
        }

        // Calculate sector resolution variable for this clustering.
        if (calcResolution(clusterings.at(iHist)) < 0.) {
          if (verboseIn >= NORMAL) {
            stringstream msg;
            msg << "Sector resolution is negative."
                << " Will ignore clustering!";
            printOut(__METHOD_NAME__, msg.str()+" ("
              +num2str(clusterings.at(iHist).Q2res)+")");
          }
          continue;
        }

        if (verboseIn >= DEBUG) {
          stringstream ss;
          int idMother1 = clusterings.at(iHist).idMoth1;
          int idMother2 = clusterings.at(iHist).idMoth2;
          ss << "Found viable clustering {" << clusterings.at(iHist).child1
             << " " << clusterings.at(iHist).child2 << " "
             << clusterings.at(iHist).child3 << "} to "<< idMother1
             << " " << idMother2 << " (" << clusterings.at(iHist).getAntName()
             << ")" <<" with Qres = "<< sqrt(clusterings.at(iHist).Q2res)
             << " GeV";
          printOut(__METHOD_NAME__,ss.str());
        }

        // If nothing went wrong, save in cluster list.
        clusterList[clusterings.at(iHist).Q2res] = clusterings.at(iHist);
      }
    }
  }

  if (verboseIn >= DEBUG) printOut(__METHOD_NAME__,"end", dashLen);
  return;

}

//--------------------------------------------------------------------------

// Do the clustering in cluster list that corresponds to the maximal
// resolution criterion.

bool HistoryNode::cluster(HistoryNode& nodeNext,
  Info* infoPtrIn, int verboseIn) {

  // Check for clusterings.
  if (clusterList.empty()) {
    if (verboseIn >= REPORT) {
      string msg = ": No clusterings found";
      infoPtrIn->errorMsg("Error in "+__METHOD_NAME__ + msg);
    }
    return false;
  }

  // 1) Fetch the VinciaClustering with minimal resolution criterion.
  VinciaClustering clusWin = (clusterList.begin())->second;

  // 2) Perform the clustering on event.
  Event clusEvent;
  vector< vector<int> > clusChains;
  if (!doClustering(clusWin,clusEvent,clusChains,infoPtrIn,verboseIn)) {
    if (verboseIn >= REPORT) {
      string msg = ": Clustering could not be done.";
      infoPtrIn->errorMsg("Error in "+__METHOD_NAME__ + msg);
    }
    return false;
  }

  // Find the evolution scale.
  double q2Evol = calcEvolScale(clusWin);
  if (q2Evol < 0.) {
    if (verboseIn >= REPORT) {
      string msg = ": Evolution variable is negative.";
      infoPtrIn->errorMsg("Error in "+__METHOD_NAME__ + msg,
        "("+num2str(q2Evol)+")");
    }
    return false;
  }

  // Save.
  nodeNext.state = clusEvent;
  nodeNext.clusterableChains = clusChains;
  nodeNext.lastClustering = clusWin;
  nodeNext.setEvolScale(sqrt(q2Evol));
  nodeNext.initPtr(vinComPtr,resPtr,antSetFSRptr);
  nodeNext.hasRes = hasRes;
  nodeNext.iRes = iRes;
  nodeNext.idRes = idRes;
  nodeNext.nMinQQbar = nMinQQbar;

  // Done.
  return true;
}

//--------------------------------------------------------------------------

// Method to perform clustering on state, given a VinciaClustering object.
//   In: clustering to perform.
//   Out: clustered event, clustered colour chains.
// Return success(true) or failure(false).

bool HistoryNode::doClustering(VinciaClustering& clus, Event& clusEvent,
  vector< vector<int> >& clusChains, Info* infoPtr, int verboseIn) {

  if (!isInitPtr) {
    if (verboseIn >= NORMAL) {
      string msg = ": pointers were not initialised";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    }
    return false;
  }

  // Note the changed particles.
  int iaEvt = clus.child1;
  int ijEvt = clus.child2;
  int ibEvt = clus.child3;

  if (verboseIn >= DEBUG) {
    stringstream ss;
    ss << "Clustering " << iaEvt << ", " << ijEvt << ", " << ibEvt;
    printOut(__METHOD_NAME__,ss.str());
  }

  vector<Particle> pClustered;
  if (!vinComPtr->clus3to2(clus,state,pClustered)) {
    if (verboseIn >= REPORT) {
      string msg = ": failed to cluster particles";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    }
    return false;
  }

  // Set new event.
  map<int,int> oldToNewIndices;
  clusEvent=state;
  clusEvent.reset();

  // Copy across system.
  clusEvent[0]=state[0];

  int inA=state[1].daughter1();
  int inB=state[2].daughter1();

  // Map from system to outStart and outEnd.
  map<int, pair<int,int>> sysToOut;
  // Always have system 0.
  sysToOut[0] = pair<int,int>(0,0);

  int iOffset = 3;
  for (int iPart=1; iPart!=state.size(); ++iPart) {
    Particle p;
    if (iPart==ijEvt) {
      ++iOffset;
      continue;
    }
    else if (iPart == 1 || iPart == 2)
      p = state[iPart];
    else
      p = pClustered.at(iPart-iOffset);

    // Add to event.
    oldToNewIndices[iPart] = clusEvent.append(p);

    // Update daughter indices for beam system.
    if (p.mother1()==inA && p.mother2()==inB) {
      if (sysToOut[0].first == 0) {
        sysToOut[0].first = oldToNewIndices[iPart];
      }
      sysToOut[0].second = oldToNewIndices[iPart];
    }

    // Initialise new daughter indices for each resonance.
    if (p.isResonance()) {
      sysToOut[oldToNewIndices[iPart]] = pair<int,int>(0,0);
    }
  }

  // Update daughter indices for resonances.
  if (sysToOut.size() > 1) {
    // Particle 6 is the first that can have a resonance as parent.
    for (int iPart(6); iPart<clusEvent.size(); ++iPart) {
      int iMotherOld = clusEvent[iPart].mother1();
      int iMotherNew = oldToNewIndices[iMotherOld];
      if (clusEvent[iMotherNew].isResonance()) {
        if (sysToOut[iMotherNew].first == 0) {
          sysToOut[iMotherNew].first = iPart;
        }
        sysToOut[iMotherNew].second = iPart;
      }
    }
  }
  if (verboseIn >= DEBUG) {
    printOut(__METHOD_NAME__, "Found " + num2str((int)sysToOut.size(),2)
      + " systems:");
    auto it = sysToOut.begin();
    for ( ; it != sysToOut.end(); ++it)
      cout << "\tSystem " << it->first << ": "
           << it->second.first << ", " << it->second.second << endl;
  }

  // Sanity checks.
  bool pass = true;
  if (clusEvent.size() != state.size()-1) {
    if (verboseIn >= REPORT) {
      string msg = ": Wrong number of particles in clustered event.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    }
    pass = false;
  } else if (oldToNewIndices[inA] != inA || oldToNewIndices[inB] != inB) {
    if (verboseIn >= REPORT) {
      string msg = ": Initial state particle changed position in";
      msg+= " clustered event.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    }
    pass = false;
  } else if (sysToOut.size() < 1) {
    if (verboseIn >= REPORT) {
      string msg = ": No parton systems found in clustered event.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    }
    pass = false;
  } else if (sysToOut[0].first == 0 || sysToOut[0].second == 0) {
    if (verboseIn >= REPORT) {
      string msg = ": No outgoing particles found in clustered event.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    }
    pass = false;
  }

  if (!pass) {
    if (verboseIn >= DEBUG) clusEvent.list();
    return false;
  }

  // Update daughters.
  for (int iPart(1); iPart<clusEvent.size(); ++iPart) {
    if (iPart == inA || iPart == inB) {
      clusEvent[iPart].daughters(sysToOut[0].first,sysToOut[0].second);
    } else if (sysToOut.find(iPart) != sysToOut.end()) {
      clusEvent[iPart].daughters(sysToOut[iPart].first,sysToOut[iPart].second);
    } else {
      int iDtr1Old = clusEvent[iPart].daughter1();
      int iDtr1New = iDtr1Old == 0 ? 0 : oldToNewIndices[iDtr1Old];
      int iDtr2Old = clusEvent[iPart].daughter2();
      int iDtr2New = iDtr2Old == 0 ? 0 : oldToNewIndices[iDtr2Old];
      clusEvent[iPart].daughters(iDtr1New, iDtr2New);
    }
  }

  // Update colour chains.
  clusChains.clear();
  for (unsigned int iChain =0; iChain<clusterableChains.size(); ++iChain) {
    vector<int> newChain;
    for (unsigned int iPos=0; iPos< clusterableChains.at(iChain).size();
         ++iPos) {
      int iPart = clusterableChains.at(iChain).at(iPos);
      if (iPart == ijEvt) {
        continue;
      }
      if (oldToNewIndices.find(iPart)==oldToNewIndices.end()) {
        if (verboseIn >= REPORT) {
          string msg = ": Could not update clustered colour chains";
          infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
        }
        return false;
      }
      int iNew = oldToNewIndices[iPart];
      newChain.push_back(iNew);
    }
    clusChains.push_back(newChain);
  }

  if (verboseIn >= DEBUG) {
    printOut(__METHOD_NAME__,"Clustered event:");
    clusEvent.list();
  }

  return true;
}

//==========================================================================

// The VinciaHistory class.

//--------------------------------------------------------------------------

// Constructor.

VinciaHistory::VinciaHistory(Event &stateIn,
  BeamParticle* beamAPtrIn,  BeamParticle* beamBPtrIn,
  MergingHooksPtr mergingHooksPtrIn,
  PartonLevel* trialPartonLevelPtrIn,
  ParticleData* particleDataPtrIn,
  Info* infoPtrIn) {

  aborted = false;

  // Save copies of pointers to Pythia objects.
  trialPartonLevel = trialPartonLevelPtrIn;
  particleDataPtr  = particleDataPtrIn;
  infoPtr          = infoPtrIn;

  // Cast input pointers as vincia objects.
  vinMergingHooksPtr = dynamic_pointer_cast<VinciaMergingHooks>
    (mergingHooksPtrIn);
  fsrShowerPtr = dynamic_pointer_cast<VinciaFSR>
    (trialPartonLevelPtrIn->timesPtr);
  isrShowerPtr = dynamic_pointer_cast<VinciaISR>
    (trialPartonLevelPtrIn->spacePtr);

  if (vinMergingHooksPtr == nullptr || fsrShowerPtr == nullptr ||
    isrShowerPtr == nullptr) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+
      "Could not create history. Is Vincia on?");
    return;
  }

  // Fetch AntennaSetFSR, MECs, VinciaCommon, and Resolution pointers.
  antSetFSRptr = fsrShowerPtr->antSetPtr;
  mecsPtr      = fsrShowerPtr->mecsPtr;
  vinComPtr    = fsrShowerPtr->vinComPtr;
  resPtr       = fsrShowerPtr->resolutionPtr;

  // Get verbosity.
  verbose = vinMergingHooksPtr->getVerbose();

  // Save event.
  state = stateIn;

  // Copy beams.
  beamA = (state[3].pz() > 0.) ? *beamAPtrIn : *beamBPtrIn;
  beamB = (state[4].pz() > 0.) ? *beamAPtrIn : *beamBPtrIn;

  // Set the merging scale (cutoff to ME generator).
  qms = vinMergingHooksPtr->tmsCut();

  // Set whether the merging scale is set in terms of the evolution variable.
  msIsEvolVar = true;
  if ( vinMergingHooksPtr->doKTMerging() || vinMergingHooksPtr->doMGMerging()
    || vinMergingHooksPtr->doCutBasedMerging()
    || vinMergingHooksPtr->doPTLundMerging() ) {
    msIsEvolVar = false;
  }

  // Set the maximum multiplicites of matrix-element generator.
  nMax = vinMergingHooksPtr->nMaxJets();
  nMaxRes = vinMergingHooksPtr->nMaxJetsRes();

  // Possible new hard process information.
  hasNewProcessSav = false;
  newProcess = Event();
  newProcessScale = 0.;

  // Find all histories, but only save best.
  findBestHistory();

}

//--------------------------------------------------------------------------

// (Sudakov) weight from performing trial shower.

double VinciaHistory::getWeightCKKWL() {

  if (!foundValidHistory) {
    string msg = ": Couldn't find valid history. Abort.";
    if (verbose >= QUIET)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    return 0.;
  }

  // Initialise weight.
  double wCKKWL = 1.;

  // Communicate to Vincia that we are doing a trial shower.
  fsrShowerPtr->setIsTrialShower(true);
  isrShowerPtr->setIsTrialShower(true);

  // Loop over systems.
  auto itHistory = historyBest.begin();
  auto itEndHistory = historyBest.end();
  for( ; itHistory!= itEndHistory; ++itHistory) {

    int iSys = itHistory->first;
    vector<HistoryNode>& history = itHistory->second;
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Performing a trial shower in system " << iSys
         << " with "<< history.size() << " nodes.";
      printOut(__METHOD_NAME__, ss.str());
    }

    // Fetch shorthands for ME and Born node.
    HistoryNode* bornNodePtr = &history.back();
    HistoryNode* meNodePtr = &history.front();

    // Communicate to Vincia whether this is a resonance system trial.
    bool isResSys = (iSys == 0 ) ? false : true;
    fsrShowerPtr->setIsTrialShowerRes(isResSys);
    isrShowerPtr->setIsTrialShowerRes(isResSys);

    // Set correct Born state for trial shower.
    // Note: Born is last entry in history.
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__,
        "Born state that will be saved for the trial shower:");
      history.back().state.list();
    }
    if (iSys == 0)
      isrShowerPtr->saveBornForTrialShower(bornNodePtr->state);
    fsrShowerPtr->saveBornForTrialShower(bornNodePtr->state);

    // Get starting scale for trial shower.
    double qStart = getStartScale(bornNodePtr->state, isResSys);

    // Loop over nodes in history starting with Born.
    // NB: last shower step starting from S_n will be performed in main shower.
    for (int iNode(history.size()-1); iNode > 0; --iNode) {
      HistoryNode& nodeNow = history.at(iNode);

      // Get end scale for trial shower.
      double qEnd = nodeNow.getEvolNow();
      if (verbose >= DEBUG) {
        stringstream ss;
        ss << "qStart = " << qStart<< ", qEnd = " << qEnd;
        printOut(__METHOD_NAME__,ss.str());
      }
      if (qEnd >= qStart) {
        if (verbose >= DEBUG) {
          string msg = "Skipping trial shower for unordered clustering.";
          printOut(__METHOD_NAME__,msg);
        }
        // Replace starting scale for next node.
        qStart = qEnd;
        continue;
      }

      // Communicate end scale to merging hooks.
      vinMergingHooksPtr->setShowerStoppingScale(qEnd);

      // Perform a trial branching, get scale.
      double qTrial = qNextTrial(qStart, nodeNow.state);

      // Check for abort.
      if (aborted) {
        if (verbose >= DEBUG)
          printOut(__METHOD_NAME__,"Aborting trial shower.");
        return 0.;
      }

      // Was it above the current scale?
      if (qTrial > qEnd) {
        if (verbose >= DEBUG) {
          stringstream ss;
          ss << "Trial shower generated an emission with scale "
             << qTrial << " above clustering scale.";
          printOut(__METHOD_NAME__,ss.str());
        }

        // Was this a new topology?
        if (hasNewProcessSav) {
          if (verbose >= DEBUG) {
            string msg = "Found a new topology: don't veto.";
            printOut(__METHOD_NAME__,msg);
          }
          return wCKKWL;
        }
        else {
          // Veto this event.
          return 0.;
        }
      }

      // Calculate PDF and alphaS ratios with next scale.
      double pT2now  = pow2(qStart);
      if (iNode == int(history.size()-1) && isrShowerPtr->pTmaxMatch != 1)
        pT2now = pow2(vinMergingHooksPtr->muFinME());
      double pT2next = pow2(qEnd);
      double wPDF    = calcPDFRatio(&nodeNow, pT2now, pT2next);
      double wAlphaS = calcAlphaSRatio(nodeNow);
      wCKKWL *= wPDF * wAlphaS;
      if (verbose >= DEBUG) {
        stringstream ss;
        ss << "PDF weight wPDF = " << wPDF
           << ", alphaS weight wAlphaS = " << wAlphaS;
        printOut(__METHOD_NAME__, ss.str());
        string msg = "Pass";
        if (iNode > 1) msg += " to next node";
        printOut(__METHOD_NAME__, msg);
      }

      // Replace starting scale for next node.
      qStart = qEnd;

    }// Finished loop over clustered states.

    // PDF ratio with factorisation scale.
    double mu2F = pow2(vinMergingHooksPtr->muFinME());
    double wPDF = 1.;
    if (history.size() > 1 || isrShowerPtr->pTmaxMatch == 1)
      wPDF = calcPDFRatio(meNodePtr, pow2(qStart), mu2F);
    wCKKWL *= wPDF;
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "PDF weight wPDF(mu2F) = " << wPDF;
      printOut(__METHOD_NAME__, ss.str());
      ss.str("");
      ss << "Saving the restart scale: "<< qStart;
      printOut(__METHOD_NAME__,ss.str());
    }
    meNodePtr->state.scale(qStart);
    // For resonance systems also tell merging hooks.
    if (isResSys)
      vinMergingHooksPtr->setScaleRes(meNodePtr->iRes, qStart);

  }// Finished loop over systems.

  // Reset changed values to factory.
  fsrShowerPtr->setIsTrialShower(false);
  isrShowerPtr->setIsTrialShower(false);
  fsrShowerPtr->setIsTrialShowerRes(false);
  isrShowerPtr->setIsTrialShowerRes(false);

  // Got to here - event passes!
  if (verbose >= DEBUG)
    printOut(__METHOD_NAME__, "wCKKWL = " + num2str(wCKKWL));
  return wCKKWL;
}

//--------------------------------------------------------------------------

// Return how many clusterings there are in our history
// (= number of additional emissions relative to Born config).

int VinciaHistory::getNClusterSteps() {

  int nClus = 0;
  auto itSys = historyBest.begin();
  auto itEnd = historyBest.end();

  // Loop over systems.
  for( ; itSys!=itEnd; ++itSys) {
    int nClusSys = itSys->second.size() -1;
    nClus+= nClusSys;
  }

  return nClus;
}

//--------------------------------------------------------------------------

// Return the scale at which to restart the parton shower.

double VinciaHistory::getRestartScale() {

  // Scale of new (MPI) emission.
  if (hasNewProcessSav && newProcessScale > 0.) {
    return newProcessScale;
  }
  // Return the reconstructed scale of the last node.
  else {
    // Note: scale of first node gets set correctly in getStartScale().
    double qRestart = 2.*state[0].e();
    // Loop over systems and find lowest restart scale.
    //TODO only use system 0 unless we use an actual resonance shower?
    auto itSys = historyBest.begin();
    auto itSysEnd = historyBest.end();
    for ( ; itSys != itSysEnd; ++itSys) {
      double qNow = itSys->second.front().state.scale();
      if (qNow > 0.)
        qRestart = min(qRestart, qNow);
    }
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Shower restart scale: " << qRestart;
      printOut(__METHOD_NAME__, ss.str());
    }

    if (qRestart < 2.*state[0].e())
      return qRestart;
  }

  // Merging scale. Should only be the last resort.
  if (verbose >= REPORT) {
    stringstream ss;
    ss << "Warning in " << __METHOD_NAME__
       << ": No restart scale found. Using merging scale.";
    infoPtr->errorMsg(ss.str(), "("+num2str(qms,6)+")");
  }
  return qms;

}


//--------------------------------------------------------------------------

// Considers all viable colour orderings and selects the best
// (has largest PS approx to ME2).

void VinciaHistory::findBestHistory() {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"begin", dashLen);
  bool foundIncompleteHistory = false;
  foundValidHistory = false;
  failedMSCut = false;
  ME2guessBest = -NANO;

  // How many colour orderings are there compatible with the Born Topology?
  // NB this is the slow part - grows as (nqqbar!).
  unsigned int nPerms = countPerms();

  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Found "<< nPerms << " colour orderings.";
    printOut(__METHOD_NAME__,ss.str());
  }

  // Something went wrong.
  if (nPerms==0) {
    if (verbose >= REPORT) {
      printOut(__METHOD_NAME__," Warning: no permutations found!");
      state.list();
    }
    return;
  }

  // Loop over all viable colour orderings and find the parton shower
  // history.
  for (unsigned int iPerm=0; iPerm<nPerms; ++iPerm) {
    // Debug printout.
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__,"Constructing history for colour flow:");
      cout << "   Beam chains:";
      for (const auto& bc : colPerms.at(iPerm).beamChains) {
        cout << " (";
        for (const int idx : bc.chainlist) cout << " " << idx;
        cout << " )";
      }
      cout << endl;
    }

    // Find the parton shower history for this permutation.
    std::tuple<bool, double, HistoryNodes> hPerm =
      findHistoryPerm(colPerms.at(iPerm));

    // For errors we have cleared the event record.
    if (std::get<2>(hPerm).size() == 0) {
      if (verbose >= NORMAL) {
        stringstream ss;
        ss << "Warning: history could not be constructed.";
        printOut(__METHOD_NAME__,ss.str());
      }
      continue;
    }

    // Check if incomplete.
    bool isIncomplete = std::get<0>(hPerm);

    // Get the PS approx to the matrix element for this history.
    double ME2guessNow = std::get<1>(hPerm);
    if (ME2guessNow <= 0. || std::isnan(ME2guessNow) ) {
      if (verbose >= NORMAL) {
        stringstream ss;
        ss << "Warning: history has "
           << (std::isnan(ME2guessNow) ? "NaN" : "negative") << " weight.";
        printOut(__METHOD_NAME__,ss.str());
      }
      continue;
    }

    // Check merging scale cut.
    if (!checkMergingCut(std::get<2>(hPerm))) {
      if (verbose >= DEBUG) {
        stringstream ss;
        ss << "History failed merging scale cut.";
        printOut(__METHOD_NAME__,ss.str());
      }
      failedMSCut = true;
      continue;
    }

    // Decide whether to save the current history.
    bool saveHistory = !foundValidHistory;
    // Always try to replace incomplete histories.
    if (foundIncompleteHistory && !isIncomplete) saveHistory = true;
    // Want to select the most singular choice as best so far.
    if (ME2guessNow > ME2guessBest) {
      // Never replace complete histories by incomplete ones.
      if (!foundIncompleteHistory && !isIncomplete) saveHistory = true;
      if (foundIncompleteHistory) saveHistory = true;
    }

    if (saveHistory) {
      // Save this choice.
      foundValidHistory = true;
      failedMSCut = false;
      foundIncompleteHistory = isIncomplete;
      historyBest = std::get<2>(hPerm);
      ME2guessBest = ME2guessNow;
      if (verbose >= DEBUG) {
        stringstream ss;
        ss<<"Saving history with weight: "<< ME2guessBest;
        printOut(__METHOD_NAME__,ss.str());
      }
    } else if (verbose >= DEBUG)
      printOut(__METHOD_NAME__,"Discarding history in favour of saved one.");

  }// End loop over all viable colour orderings.

  if (!foundValidHistory && verbose >= DEBUG) {
    printOut(__METHOD_NAME__,"Did not find any valid history");
    return;
  }

  // Done.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss<<"Best history has weight: "<< ME2guessBest;
    printOut(__METHOD_NAME__,ss.str());
    printOut(__METHOD_NAME__,"end", dashLen);
  }
}

//--------------------------------------------------------------------------

// Find all possible orderings of colour chains compatible with
// the Born-level process, and return number.

unsigned int VinciaHistory::countPerms() {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"begin", dashLen);

  // 1.) Find all colour-connected chains of gluons for input event.
  if (!getColChains()) {
    // Something went wrong.
    return 0;
  }

  // 2.) Analyse charge and flavours of the chain ends.
  int chargeSum = 0;
  int nColChains = (int)colChainsSav.size();
  ColourFlow colFlowOrig;
  // Add colour chains.
  for (int ichain=0; ichain<nColChains; ++ichain) {
    int idFront = state[colChainsSav.at(ichain).front()].id();
    int chargeFront = state[colChainsSav.at(ichain).front()].chargeType();
    if (!state[colChainsSav.at(ichain).front()].isFinal()) {
      idFront*= -1;
      chargeFront*=-1;
    }

    int idBack = state[colChainsSav.at(ichain).back()].id();
    int chargeBack = state[colChainsSav.at(ichain).back()].chargeType();
    if (!state[colChainsSav.at(ichain).back()].isFinal()) {
      idBack *= -1;
      chargeBack *= -1;
    }

    int charge = (chargeFront + chargeBack)/3;
    chargeSum += charge;

    // Save.
    colFlowOrig.addChain(charge,idFront,idBack,chainHasInitial[ichain]);
  }

  // Check if sum of beam chain charges balances with charge
  // of leptons and undecayed resonances.
  int nResPlusUndc  = vinMergingHooksPtr->getNResPlusUndecayed();
  int nResMinusUndc = vinMergingHooksPtr->getNResMinusUndecayed();
  int lepChargeSum  = 0;
  vector<HardProcessParticle*> leptons = vinMergingHooksPtr->getLeptons();
  for (auto lep : leptons) lepChargeSum += lep->chargeType();
  int resChargeSum = lepChargeSum + nResPlusUndc - nResMinusUndc;
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "Charge sums: ");
    cout << "     chains: " << num2str(chargeSum,1) << endl;
    cout << "    leptons: " << num2str(lepChargeSum,1) << endl;
    cout << " resonances: "
         << num2str(nResPlusUndc-nResMinusUndc,1) << endl;
  }
  if (chargeSum + resChargeSum != 0) {
    if (verbose >= NORMAL) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+
        ": Chain charges do not balance (with resonances).");
    }
    return 0;
  }

  // 3.) Look up the colour structure of the hard process to intialise.
  map<int, map<int,int> > countRes;
  if (!colFlowOrig.initHard(countRes,vinMergingHooksPtr)) {
    if (verbose >= NORMAL) {
      string msg = ": Failed to extract colour structure";
      msg+=" from hard process";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    }
    return 0;
  }

  // Print in DEBUG mode.
  if (verbose >= DEBUG) colFlowOrig.print(true);

  // 4.) Find all permutations compatible with the hard process.
  // Intialise vector of permutations.
  colPerms = vector<ColourFlow>(1,colFlowOrig);

  // Assign chains to resonances in all possible ways.
  if (!assignResChains(countRes,colPerms)) {
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__,"Failed to assign resonance chains");
    }
    return 0;
  }

  // Assign chains to the beams.
  if (!assignBeamChains(colPerms)) {
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__,"Failed to assign beam chains");
    }
    return 0;
  }

  // Done.
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"end", dashLen);

  // Return the number of viable permutations we found.
  return colPerms.size();

 }

//--------------------------------------------------------------------------

// Find colour-connected chains of partons in the event.

bool VinciaHistory::getColChains() {

  map<int,int> indexToAcol;
  map<int,int> colToIndex;

  // Count qqbar pairs (with everything mapped to outgoing).
  int nQ=0;
  int nQbar=0;
  vector<int> qbarList;

  // Fetch incoming partons' colours.
  vector<int> incoming;
  incoming.push_back(state[1].daughter1());
  incoming.push_back(state[2].daughter1());

  auto iIn = incoming.begin();
  for( ; iIn!=incoming.end(); ++iIn) {
    // Check this is indeed incoming, and that we haven't seen before.
    if (!state[*iIn].isFinal() && indexToAcol.find(*iIn)==indexToAcol.end()) {
      // Treat incoming col as acol.
      if (state[*iIn].col()!=0) {
        indexToAcol[*iIn]=state[*iIn].col();
      }
      if (state[*iIn].acol()!=0) {
        colToIndex[state[*iIn].acol()]=*iIn;
      }
      // Is it a quark?
      if (state[*iIn].isQuark()) {
        // Treat incoming quark as antiquark.
        if (state[*iIn].id()>0) {
          nQbar++;
          qbarList.push_back(*iIn);
        }
        else {
          nQ++;
        }
      }
    }
  }

  // Fetch outgoing partons' colours.
  for(int iPart=3; iPart<state.size(); iPart++) {
    if (state[iPart].isFinal()) {
      if (state[iPart].acol()!=0) {
        indexToAcol[iPart]=state[iPart].acol();
      }
      if (state[iPart].col()!=0) {
        colToIndex[state[iPart].col()]=iPart;
      }
      if (state[iPart].isQuark()) {
        if (state[iPart].id()>0) {
          nQ++;
        }
        else {
          nQbar++;
          qbarList.push_back(iPart);
        }
      }
    }
  }


  if (nQ!=nQbar) {
    // Error message.
    if (verbose >QUIET) {
      string msg = ": Number of quarks and antiquarks do not match.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    }
    return false;
  }

  // Store all colour connected chains (open and closed).
  vector< vector<int> > colchains;

  // Start a new colchain.
  vector<int> colchainNow;

  bool chainNowHasInitial=false;

  int nGluonLoops=0;

  while (indexToAcol.size()!=0) {

    // Find first in chain.

    // Initially start with first (probably incoming)...
    auto it=indexToAcol.begin();
    int indexNow = it->first;

    // ... but if there are any quark pairs left, start here.
    if (qbarList.size()!=0) {
      indexNow=qbarList.back();
      qbarList.pop_back();
    }
    else {
      nGluonLoops++;
    }

    while (indexToAcol.find(indexNow)!=indexToAcol.end()) {

      // Save to chain.
      colchainNow.push_back(indexNow);

      if (indexNow == incoming[0] || indexNow == incoming[1]) {
        chainNowHasInitial=true;
      }

      // Get anticolour line of this index.
      int colNow = indexToAcol[indexNow];

      // Delete entry from list of available anticolours.
      indexToAcol.erase(indexNow);

      // Fetch Pythia index for particle sharing colour line.
      indexNow = colToIndex[colNow];

    }
    if (state[indexNow].isQuark()) {
      colchainNow.push_back(indexNow);
      if (indexNow == incoming[0] || indexNow == incoming[1]) {
        chainNowHasInitial=true;
      }
    }

    // Finished: either hit a quark or a gluon we already erased.

    // Save chain.
    colchains.push_back(colchainNow);

    // Save if chain has initial partons.
    chainHasInitial[colchains.size()-1]=chainNowHasInitial;

    // Save if chain is the decay of a resonance.
    if (!chainNowHasInitial) {
      int moth = state[colchainNow.front()].mother1();
      if (state[moth].isResonance()) {
        // Save ID of resonance.
        int idRes = state[moth].id();
        if (resIDToIndices.find(idRes)==resIDToIndices.end()) {
          resIDToIndices[idRes]=vector<int>();
        }
        if (resIndexToChains.find(moth)==resIndexToChains.end()) {
          resIDToIndices[idRes].push_back(moth);
          resIndexToChains[moth]=vector<int>();
        }
        resIndexToChains[moth].push_back(colchains.size()-1);
      }
    }

    // Clear working chain.
    colchainNow.clear();

  }

  if (int(colchains.size())!=nGluonLoops+nQ) {
    // Error message.
    if (verbose >= NORMAL) {
      string msg = ": Incorrect number of colour chains.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    }
    return false;
  }

  // Save as initial permutation.
  colChainsSav = colchains;
  nQSave = nQ;
  nGluonLoopsSave = nGluonLoops;

  if (verbose >= DEBUG) {
    printChains();
  }

  return true;

}

//--------------------------------------------------------------------------

// Print the chains.

void VinciaHistory::printChains() {

  string tab ="  ";
  int nChains = (int)colChainsSav.size();
  cout << "\n --------- Colour Chain Summary -------------------------------\n"
       << tab << "Found " << nChains << " colour "
       << (nChains>1 ? "chains." : "chain.") << endl;
  tab= "     ";
  for(unsigned int ichain = 0; ichain < colChainsSav.size(); ++ichain) {
    cout << tab << "Chain " << ichain << ":";
    for(unsigned int j = 0; j < colChainsSav.at(ichain).size(); ++j)
      cout << " " << colChainsSav.at(ichain).at(j);
    cout << endl;
  }
  cout << " --------------------------------------------------------------\n";

}

//--------------------------------------------------------------------------

// Select a (single) chain for every resonance in all possible ways
// ensuring we don't double count for cases with several identical.

bool VinciaHistory::assignResChains(map<int, map<int,int>>& idCounter,
  vector<ColourFlow>& flowsSoFar) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"begin", dashLen);

  if (flowsSoFar.empty()) {
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__,"Empty flow vector");
    }
    return false;
  }

  // Select specific resonances in event record.
  if (!assignResFromEvent(idCounter,flowsSoFar)) {
    if (verbose >= DEBUG) {
      string msg = "Could not assign resonances found in event.";
      printOut(__METHOD_NAME__,msg);
    }
    return false;
  }

  // Now select resonances which were not directly in event record.

  // Outside loop over charge index.
  auto chargeIt = idCounter.begin();
  for( ; chargeIt!= idCounter.end() ; ++chargeIt) {

    int cIndex = chargeIt->first;

    // Outside loop over all resonance ids.
    auto idIt = chargeIt->second.begin();
    auto idItStop = chargeIt->second.end();
    for( ; idIt!=idItStop; ++idIt) {
      // ID of this resonance.
      int id = idIt->first;
      // How many identical copies?
      int nCopies = idIt->second;

      // Loop over each res to select.
      for(int iCopy=0; iCopy<nCopies; ++iCopy) {
        // Make assignment for this copy in all possible ways.
        if (!assignNext(flowsSoFar,true,id,cIndex)) {
          if (verbose >= DEBUG) {
            stringstream ss;
            ss<<"Could not assign copy "<< iCopy+1<< "/"
              << nCopies<< " of resonance "<< id;
            printOut(__METHOD_NAME__,ss.str());
          }
          return false;
        }
      }// Finished loop over identical copies.
    }// Finished loop over IDs (each with nCopies) to select.
  }// Finished loop over charge index.

  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"end", dashLen);

  // Done!
  if (flowsSoFar.empty()) return false;
  else return true;
}

//--------------------------------------------------------------------------

// Make specific selections from resonances identified from the event.

bool VinciaHistory::assignResFromEvent(map<int, map<int,int> >& idCounter,
  vector< ColourFlow >& flowsSoFar) {

  // Outside loop over charge index.
  auto chargeIt = idCounter.begin();
  for( ; chargeIt!= idCounter.end() ; ++chargeIt) {

    int cIndex = chargeIt->first;
    map<int,int> idsLeft;

    // Outside loop over all resonance ids.
    auto idIt = chargeIt->second.begin();
    auto idItStop = chargeIt->second.end();
    for( ; idIt!=idItStop; ++idIt) {
      // ID of this resonance.
      int id = idIt->first;
      // How many identical copies?
      int nCopies = idIt->second;
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__,"Found resonance " + num2str(id,2) + " with "
          + num2str(nCopies,2) + (nCopies > 1 ? " copies." : " copy."));

      // Are there any in the event record?
      if (resIDToIndices.find(id)!=resIDToIndices.end()) {
        int nInRecord = int(resIDToIndices[id].size());
        if (nInRecord>nCopies) {
          if (verbose >= DEBUG) {
            stringstream ss;
            ss<<"Number of resonances of type "<<id
              <<" in event is incompatible with hard process"
              <<" ("<<nInRecord<<").";
            printOut(__METHOD_NAME__,ss.str());
          }
          return false;
        }

        // Loop over indices.
        for(int iIndex=0; iIndex<nInRecord; iIndex++) {
          int index = resIDToIndices[id].at(iIndex);
          // Fetch the chains for this index.
          vector<int> chains = resIndexToChains[index];
          // Make selection.
          if (!assignThis(flowsSoFar,id,cIndex,chains)) {
            return false;
          }
          nCopies--;
        }// End loop over resonance indices.
      }// End loop over ids found in event record.

      // Are there any left?
      if (nCopies > 0) {
        idsLeft[id]=nCopies;
      }
    }// Finished loop over IDs.

    // Save leftover.
    chargeIt->second = idsLeft;

  }// End loop over charge indices.
  return true;

}

//--------------------------------------------------------------------------

// Make a single selection in all possible ways.

bool VinciaHistory::assignNext(vector<ColourFlow>& flowsSoFar, bool isRes,
  int id, int cIndexIn) {

  if (flowsSoFar.empty()) {
    if (verbose >= DEBUG) printOut(__METHOD_NAME__,"Empty flow vector");
    return false;
  }

  // Copy flows so far.
  vector<ColourFlow> flowsSav = flowsSoFar;

  // Clear original.
  flowsSoFar.clear();

  // Loop over flows so far.
  for(int iFlow=0; iFlow<int(flowsSav.size());++iFlow) {

    ColourFlow flowNow = flowsSav.at(iFlow);

    // Check if we have enough chains left to make selection.
    bool goodFlow=true;
    if (isRes) goodFlow = flowNow.checkChains(cIndexIn);
    else goodFlow = flowNow.checkChains();
    if (!goodFlow) {
      if (verbose >= DEBUG) {
        printOut(__METHOD_NAME__,"Skipping bad flow");
      }
      // Not enough - skip.
      continue;
    }

    // Fetch the previous index selection of this type of resonance.
    int lastIndex = -1;
    if (isRes) {
      if (flowNow.resChains[id].size()>0) {
        int chainIndex = flowNow.resChains[id].back().index;
        lastIndex = chainIndex*4+cIndexIn;
      }
    } else if (flowNow.beamChains.size()!=0) {
      int chainIndex = flowNow.beamChains.back().index;
      int chargeIndexLast = flowNow.beamChains.back().cindex;
      lastIndex = chainIndex*4+chargeIndexLast;
    }

    // Ensure ordered - next index must exceed last.
    auto itpseudochain = flowNow.pseudochains.upper_bound(lastIndex);

    // Loop over all available pseudochains for this selection.
    for( ; itpseudochain!= flowNow.pseudochains.end(); ++itpseudochain) {
      int index = itpseudochain->first;

      // Extra conditions for resonances.
      if (isRes) {
        // Skip wrong charge.
        int cindex = index%4;
        if (cindex != cIndexIn) continue;

        // Skip if initial.
        if (itpseudochain->second.front().hasInitial) continue;
      }
      else if (lastIndex>0) {
        // For beam selections, list of pseudochains should be
        // "unmergeable" - otherwise we'll hit same config twice.
        int frontID = itpseudochain->second.front().flavStart;
        int lastID = flowNow.beamChains.back().flavEnd;
        // Check if we want to do VBF.
        bool doVBF = vinMergingHooksPtr->doMergeInVBF();
        if (!doVBF && abs(frontID)==abs(lastID)) continue;
      }

      // Skip if pseudochain is too long/too short.
      int minLength=flowNow.minLength();
      int maxLength=flowNow.maxLength();
      auto next=itpseudochain;
      next++;

      int length = (itpseudochain->second.front().chainlist).size();
      if (length > maxLength || length < minLength) continue;

      // This selection is OK!

      // Loop over all orderings for this pseudochain.
      int nOrderings = itpseudochain->second.size();
      for(int iorder=0; iorder<nOrderings; ++iorder) {

        // Make a new copy.
        ColourFlow newFlow = flowNow;

        // Make the selection.
        if (isRes) {
          newFlow.selectResChains(index,iorder,id);
        }
        else {
          newFlow.selectBeamChains(index,iorder);
        }

        // Save.
        flowsSoFar.push_back(newFlow);
      }// Finished loop over orderings of these chains.
    }// Finished loop over this selection of chains.
  }// Finished loop over all prior selections for flows.

  // Done.
  if (flowsSoFar.empty()) return false;
  else return true;

}

//--------------------------------------------------------------------------

// Make a specific selection for a resonance.

bool VinciaHistory::assignThis(vector< ColourFlow > &flowsSoFar, int id,
  int cIndex, vector<int>& chains) {

  if (flowsSoFar.empty()) return false;

  // Calculate the pseudochain index.
  int assignIndex=cIndex;
  for(auto itchain = chains.begin() ; itchain!=chains.end(); ++itchain) {
    int baseIndex = pow(2,*itchain);
    assignIndex+=4*baseIndex;
  }

  // Copy flows so far.
  vector< ColourFlow > flowsSav=flowsSoFar;

  // Clear original.
  flowsSoFar.clear();

  // Loop over flows so far.
  for(int iFlow=0; iFlow<int(flowsSav.size());++iFlow) {

    ColourFlow flowNow = flowsSav.at(iFlow);

    // Fetch the requested pseudochain.
    auto itpseudochain=flowNow.pseudochains.find(assignIndex);
    if (itpseudochain==flowNow.pseudochains.end()) {
      if (verbose >= NORMAL) {
        stringstream ss;
        ss<<assignIndex;
        infoPtr->errorMsg("Error in "+__METHOD_NAME__+": could not "
          "find requested pseudochain ",ss.str());
      }
      return false;
    }
    if (verbose >= DEBUG) {
      stringstream ss;
      ss << "Assigned pseudochain " << assignIndex
         << " to resonance id " << id;
      printOut(__METHOD_NAME__,ss.str());
    }

    // Loop over all orderings for this pseudochain.
    int nOrderings = itpseudochain->second.size();
    for(int iorder=0; iorder<nOrderings; ++iorder) {

      // Make a new copy.
      ColourFlow newFlow = flowNow;

      // Make the selection.
      newFlow.selectResChains(assignIndex,iorder,id);

      // Save.
      flowsSoFar.push_back(newFlow);
    }// Finished loop over orderings of these chains.

  }// Finished loop over all prior selections for flows.

  // Done.
  if (flowsSoFar.empty()) return false;
  else return true;

}

//--------------------------------------------------------------------------

// Assign all remaining chains in all possible orderings to beams.

bool VinciaHistory::assignBeamChains(vector<ColourFlow>& flowsSoFar) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"begin", dashLen);

  if (flowsSoFar.empty()) {
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__,"Empty flow vector");
    }
    return false;
  }

  int nMin = vinMergingHooksPtr->getNChainsMin();
  if (nMin>0) {
    // Firstly, make nMin selections.
    for(int iSelect=0; iSelect<nMin; ++iSelect) {
      assignNext(flowsSoFar);
    }

    // Create empty vector for completed flows.
    vector<ColourFlow> completed;
    while (flowsSoFar.size()>0) {

      // Sort out finished / unfinished.
      vector<ColourFlow> unfinished;
      for(int iFlow=0; iFlow!=int(flowsSoFar.size()); ++iFlow) {
        // Get number of chains remaining.
        if (flowsSoFar.at(iFlow).getNChainsLeft()==0) {
          // Nothing left to do: save and continue.
          completed.push_back(flowsSoFar.at(iFlow));
        }
        else {
          unfinished.push_back(flowsSoFar.at(iFlow));
        }
      }

      // Make one more selection in all unfinished flows.
      if (unfinished.size()>0) {
        assignNext(unfinished);
      }

      // Copy across.
      flowsSoFar.clear();
      flowsSoFar = unfinished;
    }

    // All flows are complete.
    // Just copy back to flowsSoFar to return.
    flowsSoFar=completed;
  }
  if (verbose >= DEBUG) printOut(__METHOD_NAME__,"end", dashLen);

  // Done!
  if (flowsSoFar.empty()) return false;
  else return true;

}

//--------------------------------------------------------------------------

// Is this (completed) flow compatible with the hard process?

bool VinciaHistory::check(ColourFlow & /*flow*/) {return true;}

//--------------------------------------------------------------------------

// Construct history for a given colour permutation.

std::tuple<bool,double,HistoryNodes> VinciaHistory::findHistoryPerm(
  ColourFlow& flow) {

  // Initialise the matrix element weight.
  double ME2guess = 1.0;
  bool   isIncomplete = false;

  // First check this is a valid colour flow.
  if (!check(flow))
    return make_tuple(isIncomplete,0.,HistoryNodes());

  // Initialise history nodes for each system.
  HistoryNodes sysToHistory = initHistoryNodes(flow);
  if (sysToHistory.size() == 0)
    return make_tuple(isIncomplete,0.,sysToHistory);

  // Now loop over systems and find histories.
  for(auto itHistory = sysToHistory.begin(); itHistory != sysToHistory.end();
      ++itHistory) {

    int iSys = itHistory->first;
    bool isResSys = (iSys == 0 ) ? false : true;

    vector<HistoryNode>& history = itHistory->second;
    bool foundIncomplete = false;

    // Check if we hit the Born configuration.
    while (!foundIncomplete && !isBorn(history.back(), isResSys)) {

      // Check if we found any valid clusterings.
      int nClusterings =
        history.back().getNClusterings(vinMergingHooksPtr,infoPtr,verbose);
      if (nClusterings <= 0) {
        if (verbose >= DEBUG)
          printOut("VinciaHistory::findHistoryPerm()",
            "Couldn't find any clusterings.");
        // This is an incomplete history.
        foundIncomplete = true;
        continue;
      }

      if (verbose >= DEBUG)
        printOut("VinciaHistory::findHistoryPerm()","Found "
          + num2str(nClusterings,3) + " clusterings.");

      // Perform clustering that corresponds to minimal sector resolution.
      HistoryNode next;
      if (!history.back().cluster(next, infoPtr, verbose)) {
        infoPtr->errorMsg("Error in VinciaHistory::findHistoryPerm",
          ": Could not perform clustering");
        return make_tuple(isIncomplete,0.,HistoryNodes());
      }

      // Save.
      history.push_back(std::move(next));

      // Iterate until we hit Born topology (compare to hard process).
    }

    if (verbose >= DEBUG) {
      stringstream ss;
      if (!foundIncomplete) ss << "Reached Born topology in system " << iSys;
      else ss << "Found incomplete history in system " << iSys;
      printOut("VinciaHistory::findHistoryPerm()",ss.str());
    }

    // Check if incomplete.
    if (foundIncomplete) isIncomplete = true;

    // Fetch the parton shower weight.
    double ME2guessSys = calcME2guess(history, isResSys);
    ME2guess *= ME2guessSys;

    // Stop if non-positive or nan weight.
    if (ME2guessSys<=0. || std::isnan(ME2guessSys) ) {
      if (verbose >= DEBUG) {
        stringstream ss;
        ss << "ME2 guess = " << ME2guessSys << " in system " << iSys;
        printOut("VinciaHistory::findHistoryPerm() ",ss.str());
      }
      return make_tuple(isIncomplete,ME2guess,HistoryNodes());
    }

  }// End loop over systems.

  // Return the sequences of history nodes with corresponding PS weight.
  return make_tuple(isIncomplete,ME2guess,sysToHistory);

}

//--------------------------------------------------------------------------

// Check to see if history failed merging scale cut.

bool VinciaHistory::checkMergingCut(HistoryNodes& history) {
  // If we have set the merging scale in terms of the evolution variable,
  // only check last state for each system.
  if (msIsEvolVar) {
    // Loop over systems and check last node.
    for(auto itSys = history.begin() ; itSys != history.end(); ++itSys) {
      HistoryNode& lastNode = itSys->second.back();
      double qmsNow = lastNode.getEvolNow();
      // Failed.
      if (qmsNow < qms) return false;
    }
  }
  // If the merging variable is different to the evolution variable,
  // make sure each intermediate state is above the merging scale.
  else {
    // Loop over systems and check last node.
    for(auto itSys = history.begin(); itSys != history.end(); ++itSys) {
      // Loop over the history of this system.
      vector<HistoryNode> historyNow = itSys->second;
      for (auto itHistory = historyNow.begin(); itHistory != historyNow.end();
           ++itHistory) {
        // Failed.
        if (!vinMergingHooksPtr->isAboveMS(itHistory->state)) return false;
      }
    }
  }

  // Passed.
  return true;

}


//--------------------------------------------------------------------------

// Initialise history nodes for each system.

HistoryNodes VinciaHistory::initHistoryNodes(ColourFlow& flow) {

  if (verbose >= DEBUG)
    printOut("VinciaHistory::initHistoryNodes",
      "Initialising start nodes for this history");

  // Create empty list of histories nodes.
  HistoryNodes histories;

  // Create a map of systems to resonance id.
  map<int, int> sysToRes;

  // Match up systems to colour chains for this colour flow.
  map<int, vector< vector<int> > > systems = getSystems(flow,sysToRes);
  if (systems.size()==0) {
    return histories;
  }
  bool hasBeam = (systems.find(0)!=systems.end()) ? true : false;

  // Create a new event.
  Event newState = state;
  newState.reset();
  newState[0]=state[0];

  // Copy across system and beams.
  for (int i=1; i<=2; i++) {
    newState.append(state[i]);
  }
  // Copy accross incoming.
  int inA = state[1].daughter1();
  int inB = state[2].daughter1();
  int inAnew = newState.append(state[inA]);
  int inBnew = newState.append(state[inB]);

  int outStart = newState.size();

  // Add beam chain members to event.
  vector< vector<int> > beamChains;
  if (hasBeam) {
    beamChains = systems[0];
    auto itBeamChain= systems[0].begin();
    auto itEnd= systems[0].end();
    for( ; itBeamChain!=itEnd; ++itBeamChain) {
      size_t posChain = itBeamChain - systems[0].begin();
      auto itPart=itBeamChain->begin();
      auto itPartEnd=itBeamChain->end();
      for( ; itPart!=itPartEnd; ++itPart) {
        // Skip incoming.
        if (*itPart == inA || *itPart == inB) {
          continue;
        }
        size_t pos = itPart - itBeamChain->begin();
        // Add to event.
        int iNew = newState.append(state[*itPart]);
        newState[iNew].mothers(inAnew,inBnew);
        // Save new indices.
        beamChains[posChain].at(pos) = iNew;
      }
    }
  }

  // Add any leptonically-decaying singlet resonances.
  map<int, vector<int> > leptons;
  for(int iPart=3; iPart<int(state.size()); ++iPart) {
    if (state[iPart].isResonance()) {
      // Skip coloured.
      if (state[iPart].colType() != 0) continue;
      // Check daughters.
      int d1 = state[iPart].daughter1();
      int d2 = state[iPart].daughter2();
      if ( (d1!= 0 && !state[d1].isLepton() ) ||
          (d2!= 0 && !state[d2].isLepton() ) ) {
        continue;
      }
      int iNew = newState.append(state[iPart]);
      newState[iNew].statusPos();
      newState[iNew].mothers(inAnew,inBnew);
      newState[iNew].daughters(0,0);
      leptons[iNew]=state[iPart].daughterList();
    }
  }

  // Add any hadronically-decaying resonances.
  map<int, int> iResToSys;
  auto itSys = sysToRes.begin();
  auto itSysEnd = sysToRes.end();
  for( ; itSys!=itSysEnd; ++itSys) {
    int iSys = itSys->first;
    int idRes = itSys->second;
    // Get momentum of resonance.
    Vec4 pRes(0.,0.,0.,0.);
    vector<int> resChain = systems[iSys].front();
    auto itChain = resChain.begin();
    auto itChainEnd = resChain.end();
    for( ; itChain!=itChainEnd; ++itChain) {
      pRes+= state[*itChain].p();
    }
    int iNew = newState.append(
      Particle(idRes,22,inAnew,inBnew,0,0,0,0,pRes,pRes.mCalc(),0.,9) );
    iResToSys[iNew]=iSys;
  }

  // Add any leptons that are attached to the beams.
  if (hasBeam) {
    for(int iPart(5); iPart<int(state.size()); ++iPart) {
      if (!state[iPart].isLepton()) continue;
      if (state[iPart].mother1() == 3 && state[iPart].mother2() == 4) {
        int iNew = newState.append(state[iPart]);
        newState[iNew].statusPos();
        newState[iNew].mothers(inAnew,inBnew);
        newState[iNew].daughters(0,0);
      }
    }
  }
  int outEnd = newState.size()-1;

  // Update daughters of incoming.
  newState[inAnew].daughters(outStart,outEnd);
  newState[inBnew].daughters(outStart,outEnd);

  // Add leptons that are attached to resonances.
  auto itlepton = leptons.begin();
  for(; itlepton!=leptons.end(); ++itlepton) {
    if (itlepton->second.size() == 0) continue;
    int iRes = itlepton->first;
    int d1=newState.size();
    auto it=itlepton->second.begin();
    for( ; it!=itlepton->second.end(); ++it) {
      int iNew = newState.append(state[*it]);
      newState[iNew].mothers(iRes,0);
    }
    int d2 = newState.size()-1;
    newState[iRes].daughters(d1,d2);
    newState[iRes].statusNeg();
  }

  // Create history for hard scattering.
  HistoryNode beamNode(newState,beamChains,qms);
  beamNode.initPtr(vinComPtr,resPtr,antSetFSRptr);
  beamNode.nMinQQbar = vinMergingHooksPtr->getNQPairs() ;

  if (hasBeam) histories[0] = vector<HistoryNode>(1,beamNode);

  // Create history for each hadronically decaying resonance.
  auto itRes = iResToSys.begin();
  auto itResEnd = iResToSys.end();
  for( ; itRes!=itResEnd; ++itRes) {

    // Make a new node.
    HistoryNode resNode = beamNode;

    // Fetch index of resonance and system.
    int iRes = itRes->first;
    int iSys = itRes->second;

    // Fetch decay chain.
    vector<int> daughters;
    auto itChain = systems[iSys].front().begin();
    auto itChainEnd = systems[iSys].front().end();
    for( ; itChain!=itChainEnd; ++itChain) {
      // Add this daughter.
      int iNew = resNode.state.append(state[*itChain]);
      // Update mother info.
      resNode.state[iNew].mothers(iRes,0);
      daughters.push_back(iNew);
    }

    // Update resonance.
    resNode.state[iRes].statusNeg();
    resNode.state[iRes].daughters(daughters.front(),daughters.back());

    // Set node colour chains.
    resNode.clusterableChains.clear();
    resNode.clusterableChains.push_back(daughters);

    // Store resonance info.
    resNode.hasRes = true;
    resNode.iRes = iRes;
    resNode.idRes = resNode.state[iRes].id();

    // Minimum number of qqbar pairs.
    resNode.nMinQQbar = 1;

    // Add to histories.
    histories[iSys] = vector<HistoryNode>(1,resNode);

  }

  return histories;
}

//--------------------------------------------------------------------------

// Translate abstract book-keeping of colourflow into systems of particles.

map<int,vector< vector<int> > > VinciaHistory::getSystems( ColourFlow& flow,
    map<int, int> & sysToRes) {

  map<int, vector< vector<int> > > systems;
  sysToRes.clear();

  // Start with beam.
  int iSys=0;
  int nBeamChains = int(flow.beamChains.size());
  // Create entry in systems.
  if (nBeamChains>0)
    systems[iSys] = vector< vector<int> >(nBeamChains,vector<int>());
  // Loop over pseudochains.
  for( int iChain=0; iChain<nBeamChains; ++iChain) {
    PseudoChain & pschainNow = flow.beamChains.at(iChain);
    // Convert pseudochain into vector of int.

    // Loop over all chains in pschain.
    for(int jChain=0; jChain<int(pschainNow.chainlist.size()); ++jChain) {
      // Get colour chain.
      int chainIndex = pschainNow.chainlist.at(jChain);
      vector<int> chain = colChainsSav.at(chainIndex);
      // Insert.
      auto pos = (systems[iSys][iChain]).end();
      (systems[iSys][iChain]).insert(pos, chain.begin(),chain.end());
    }
  }

  // Deal with resonances.
  auto itRes = flow.resChains.begin();
  auto itResEnd = flow.resChains.end();
  for( ; itRes!= itResEnd; ++itRes) {
    int idRes = itRes->first;
    int nCopies = int(itRes->second.size());

    // Loop over copies.
    for(int iCopy=0; iCopy!=nCopies; ++iCopy) {

      // Get next system number.
      iSys++;

      // Save id.
      sysToRes[iSys]=idRes;

      // Create entry in systems.
      systems[iSys]= vector< vector<int> >(1,vector<int>());

      // Fetch pseudo chain.
      PseudoChain& pschainNow = itRes->second.at(iCopy);

      // Loop over all chains in pseudo chain.
      for(int jChain=0; jChain<int(pschainNow.chainlist.size()); ++jChain) {
        // Get colour chain.
        int chainIndex = pschainNow.chainlist.at(jChain);
        vector<int> chain = colChainsSav.at(chainIndex);
        // Insert.
        auto pos = (systems[iSys][0]).end();
        (systems[iSys][0]).insert(pos, chain.begin(),chain.end());
      }
    }
  }

  if (verbose >= DEBUG) {
    stringstream ss;
    ss<< "Found "<<systems.size()<<" systems.";
    printOut("VinciaHistory::getSystems()",ss.str());
  }
  return systems;

}

//--------------------------------------------------------------------------

// Decide if state is the Born topology.

bool VinciaHistory::isBorn(const HistoryNode& nodeIn, bool isRes) {

  // Count chains.
  int nChains = nodeIn.clusterableChains.size();

  // Check resonance chain.
  if (isRes) {

    // Should only be a single chain: something went wrong.
    if (nChains > 1) {
      return false;
    }
    // Should be a single qqbar pair.
    if ( nodeIn.clusterableChains.back().size() > 2 ) {
      return false;
    }
  } else {
    // Check beam chains.
    if (nChains > vinMergingHooksPtr->getNChainsMax()) {
      return false;
    }

    // Count partons.
    int nPartons=0;
    for(int ichain=0; ichain<nChains; ++ichain) {
      nPartons+= nodeIn.clusterableChains.at(ichain).size();
    }
    if (nPartons > vinMergingHooksPtr->getNPartons()) {
      return false;
    }
  }

  // Got to here: passed all checks.
  return true;

}

//--------------------------------------------------------------------------

// Initialise beams for node.

bool VinciaHistory::setupBeams(const HistoryNode* node, double scale2) {

  // Require sensible event.
  if (node->state.size() < 4) return false;

  // Nothing to do for leptonic beams.
  if (node->colTypeA() == 0 && node->colTypeB() == 0) return true;

  // Check that beam A has positive momentum along z.
  bool beamAisPos = (node->state[3].pz() > 0.);

  // Set beam parameters.
  int iPos = beamAisPos ? 3 : 4;
  int iNeg = beamAisPos ? 4 : 3;
  int idPos = beamAisPos ? node->idA() : node->idB();
  int idNeg = beamAisPos ? node->idB() : node->idA();
  double xPos = beamAisPos ? node->xA() : node->xB();
  double xNeg = beamAisPos ? node->xB() : node->xA();

  //TOD: save companions?

  // Fill beams.
  beamA.clear();
  beamB.clear();
  beamA.append(iPos, idPos, xPos);
  beamB.append(iNeg, idNeg, xNeg);

  // Store whether resolved parton is valence or sea.
  // Note: Needs call to xfModified first.
  beamA.xfISR(0, idPos, xPos, scale2);
  beamB.xfISR(0, idNeg, xNeg, scale2);
  beamA.pickValSeaComp();
  beamB.pickValSeaComp();

  // All good: return.
  return true;
}

//--------------------------------------------------------------------------

// Calculate criterion for testing whether to keep history.

double VinciaHistory::calcME2guess(vector<HistoryNode>& history, bool isRes) {

  // Calculate product of sector antennae * ME Born.

  // Start with Born.
  vector<HistoryNode>::reverse_iterator itNode = history.rbegin();

  double ME2sector = calcME2Born(*itNode,isRes);

  // Iterate over all nodes in history except first.
  for( ; itNode!=history.rend()-1; ++itNode) {

    // Fetch the clustering that lead to this state.
    VinciaClustering & clusNow = itNode->lastClustering;

    // Calculate the sector antenna function.
    double antNow = calcAntFun(clusNow);
    ME2sector *= antNow;
  }

  return ME2sector;
}

//--------------------------------------------------------------------------

// Calculate ME for Born-level event if we have it.

double VinciaHistory::calcME2Born(const HistoryNode & bornNode, bool isRes) {

  const Event& born = bornNode.state;
  vector<Particle> parts;

  int nIn = 0;
  if (isRes) {
    // Fetch particles.
    vector<int> children = bornNode.clusterableChains.back();
    int iRes = born[children.front()].mother1();
    parts.push_back(born[iRes]);
    auto itChild = children.begin();
    for( ; itChild!= children.end(); ++itChild) {
      parts.push_back(born[*itChild]);
    }
    nIn = 1;
  } else {
    for(int i=3; i!=int(born.size()); ++i) {
      parts.push_back(born[i]);
      if (!born[i].isFinal()) ++nIn;
    }
    if (nIn > 2) {
      if (verbose >= DEBUG) {
        printOut(__METHOD_NAME__,
          "Too many incoming particles in Born, returning 1.");
      }
      return 1.;
    }
  }

  if (mecsPtr->meAvailable(parts)) {
    double ME2Born = mecsPtr->getME2(parts, nIn);
    if (ME2Born > 0. && !std::isnan(ME2Born)) {
      if (verbose >= DEBUG) {
        stringstream ss;
        ss<<"Born ME2 = "<< ME2Born;
        printOut(__METHOD_NAME__,ss.str());
      }
      return ME2Born;
    } else {
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__,"Couldn't calculate Born ME2, returning 1.");
    }
  } else {
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__,"Born ME2 not available, returning 1.");
  }

  return 1.;
}

//--------------------------------------------------------------------------

// Calculate the antenna function for a given clustering.

double VinciaHistory::calcAntFun(const VinciaClustering& clusNow) {

  // Fetch correct antfunptr.
  AntennaFunction* antFunPtr;
  if (clusNow.isFSR)
    antFunPtr = fsrShowerPtr->getAntFunPtr(clusNow.antFunType);
  else
    antFunPtr = isrShowerPtr->getAntFunPtr(clusNow.antFunType);

  if (!antFunPtr) {
    stringstream ss;
    ss << "(" << "antFunType = " << clusNow.antFunType << ")";
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": Could not fetch antenna.",
      ss.str());
    return -1.;
  }

  // Pass invariants and masses to evaluate antFun.
  double antFun = antFunPtr->antFun(clusNow.invariants,
    clusNow.massesChildren);
  return antFun;

}

//--------------------------------------------------------------------------

// Calculate the PDF ratio to multiply the CKKW-L weight.

double VinciaHistory::calcPDFRatio(const HistoryNode* nodeNow,
  double pT2now, double pT2next) {

  // Nothing to do for leptonic beams.
  if (nodeNow->colTypeA() == 0 && nodeNow->colTypeB() == 0) return 1.;

  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Calculating PDF ratio between pTnow = " << sqrt(pT2now)
       << " and pTnext = " << sqrt(pT2next) << " for state:";
    printOut(__METHOD_NAME__, ss.str());
    nodeNow->state.list();
  }

  // Set beams for current node and calculate PDFs.
  setupBeams(nodeNow, pT2now);
  //  cout << __METHOD_NAME__ << "calculating PDFs now." << endl;
  double xfAnow = nodeNow->colTypeA() != 0 ?
    beamA.xfISR(0, nodeNow->idA(), nodeNow->xA(), pT2now) : 1.;
  double xfBnow = nodeNow->colTypeB() != 0 ?
    beamB.xfISR(0, nodeNow->idB(), nodeNow->xB(), pT2now) : 1.;

  double xfAnext = nodeNow->colTypeA() != 0 ?
    beamA.xfISR(0, nodeNow->idA(), nodeNow->xA(), pT2next) : 1.;
  double xfBnext = nodeNow->colTypeB() != 0 ?
    beamB.xfISR(0, nodeNow->idB(), nodeNow->xB(), pT2next) : 1.;
  if (xfAnext != 0 && xfAnext < 0.1*NANO) xfAnext = 0.1*NANO;
  if (xfBnext != 0 && xfBnext < 0.1*NANO) xfBnext = 0.1*NANO;

  // Calculate PDF ratios.
  double RpdfA = xfAnow/xfAnext;
  double RpdfB = xfBnow/xfBnext;

  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "xfAnow = " << xfAnow << ", xfAnext = "
       << xfAnext << ", RpdfA = " << RpdfA;
    ss << ", xfBnow = " << xfBnow << ", xfBnext = "
       << xfBnext << ", RpdfB = " << RpdfB;
    printOut(__METHOD_NAME__, ss.str());
  }

  return RpdfA*RpdfB;

}

//--------------------------------------------------------------------------

// Calculate the alphaS ratio to multiply the CKKW-L weight.

double VinciaHistory::calcAlphaSRatio(const HistoryNode& node) {

  // Fetch alphaS as used in matrix element.
  double aSME = infoPtr->alphaS();

  // Calculate alphaS as used in shower branching.
  enum AntFunType antFunType = node.lastClustering.antFunType;
  double aSshower = 0.;
  double pT2next = pow2(node.getEvolNow());
  if (node.lastClustering.isFSR) {
    double kMu2 = 0.;
    double mu2 = 0.;
    // AlphaS for gluon splittings.
    if (antFunType == GXsplitFF || antFunType == XGsplitRF) {
      // Set kFactor as in shower.
      kMu2 = fsrShowerPtr->aSkMu2Split;
      mu2 = max(fsrShowerPtr->mu2min,
        fsrShowerPtr->mu2freeze + kMu2 * pT2next);
      aSshower = fsrShowerPtr->aSsplitPtr->alphaS(mu2);
    } else {
      // Set kFactor as in shower
      kMu2 = fsrShowerPtr->aSkMu2Emit;
      mu2 = max(fsrShowerPtr->mu2min,
        fsrShowerPtr->mu2freeze + kMu2 * pT2next);
      aSshower = fsrShowerPtr->aSemitPtr->alphaS(mu2);
    }
    // Limit maximum alphaS.
    aSshower = min(aSshower, fsrShowerPtr->alphaSmax);
  } else {
    // Set kFactor as in shower.
    double kMu2 = isrShowerPtr->aSkMu2EmitI;
    if (antFunType == XGsplitIF)
      kMu2 = isrShowerPtr->aSkMu2SplitF;
    else if (antFunType == QXsplitIF || antFunType == QXsplitII)
      kMu2 = isrShowerPtr->aSkMu2SplitI;
    else if (antFunType == GXconvIF || antFunType == GXconvII)
      kMu2 = isrShowerPtr->aSkMu2Conv;
    double mu2 = max(isrShowerPtr->mu2min,
      isrShowerPtr->mu2freeze + kMu2 * pT2next);
    aSshower = isrShowerPtr->alphaSptr->alphaS(mu2);
    // Limit maximum alphaS.
    aSshower = min(aSshower, isrShowerPtr->alphaSmax);
  }

  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "At scale pT = " << sqrt(pT2next)
       << ": alphaS(shower) = " << aSshower
       << ", alphaS(ME) = " << aSME;
    printOut(__METHOD_NAME__, ss.str());
  }

  return aSshower / aSME;

}

//--------------------------------------------------------------------------

// Determine the starting scale.

double VinciaHistory::getStartScale(Event& event, bool isRes) {

  double mSys = 0.;

  // For hard event set start scale as in main shower.
  if (!isRes) {
    // If pTmaxMatch = 2: always start at eCM.
    if (fsrShowerPtr->pTmaxMatch == 2)
      mSys = (event[1].p()+event[2].p()).mCalc();
    // If pTmaxMatch = 1: always start at QF (modulo kFudge).
    else if (fsrShowerPtr->pTmaxMatch == 1)
      mSys = sqrt(fsrShowerPtr->pT2maxFudge * infoPtr->Q2Fac());
    // Else check if this event has final-state jets or photons.
    else {
      bool hasRad = false;
      for (int i(5); i<event.size(); ++i) {
        if (!event[i].isFinal()) continue;
        int idAbs = event[i].idAbs();
        if (idAbs <= 5 || idAbs == 21 || idAbs == 22) hasRad = true;
        if (idAbs == 6 && fsrShowerPtr->nGluonToQuark == 6) hasRad = true;
        if (hasRad) break;
      }
      // If no QCD/QED partons detected, allow to go to phase-space maximum.
      if (hasRad) mSys = sqrt(fsrShowerPtr->pT2maxFudge * infoPtr->Q2Fac());
      else mSys = (event[1].p()+event[2].p()).mCalc();;
    }
  }
  // Otherwise find the resonance that has decayed (should only be one).
  else {
    for(int iPart=0; iPart<int(event.size()); ++iPart) {
      if (!event[iPart].isFinal() && event[iPart].isResonance() ) {
        mSys = event[iPart].mCalc();
      }
    }
  }

  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Setting start scale = " << mSys;
    printOut(__METHOD_NAME__, ss.str());
  }
  return mSys;

}

//--------------------------------------------------------------------------

// Perform a trial branching and return scale.

double VinciaHistory::qNextTrial(double qStart, Event& evtIn) {

  if (qStart<=0.) {
    return 0.;
  }

  if (verbose >= DEBUG) {
    stringstream ss;
    ss<<"Doing a trial shower starting from "<< qStart;
    printOut(__METHOD_NAME__,ss.str());
  }

  // Reset trialShower object.
  trialPartonLevel->resetTrial();

  // Construct event to be showered.
  Event evtOut = Event();
  evtOut.init("(hard process - modified)", particleDataPtr);
  evtOut.clear();

  // Reset process scale so that shower starting scale is correctly set.
  evtIn.scale(qStart);

  // Perform trial shower emission.
  if (!trialPartonLevel->next(evtIn,evtOut)) {
    aborted = true;
    return 0.;
  }

  // Get trial shower pT and type.
  double qtrial = trialPartonLevel->pTLastInShower();
  int typeTrial = trialPartonLevel->typeLastInShower();

  // Check if this was MPI.
  if ( typeTrial==1 ) {
    hasNewProcessSav = true;
    newProcess = evtOut;
    newProcessScale = qtrial;
    int sizeOld = int(evtIn.size());
    for(int iNew=sizeOld; iNew<newProcess.size(); ++iNew) {
      if (newProcess[iNew].statusAbs()==31) {
        newProcess[iNew].statusCode(21);
      }
      else if (newProcess[iNew].statusAbs()==33) {
        newProcess[iNew].statusCode(23);
      }
    }
  }
  return qtrial;

}

//==========================================================================

} // end namespace Pythia8
