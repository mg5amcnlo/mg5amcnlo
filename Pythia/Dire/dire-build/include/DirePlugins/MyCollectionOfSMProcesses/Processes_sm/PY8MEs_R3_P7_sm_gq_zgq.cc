//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.5.3, 2017-03-09
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "PY8MEs_R3_P7_sm_gq_zgq.h"
#include "HelAmps_sm.h"

using namespace Pythia8_sm; 

namespace PY8MEs_namespace 
{
//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: g u > z g u WEIGHTED<=4 @3
// Process: g c > z g c WEIGHTED<=4 @3
// Process: g d > z g d WEIGHTED<=4 @3
// Process: g s > z g s WEIGHTED<=4 @3
// Process: g u~ > z g u~ WEIGHTED<=4 @3
// Process: g c~ > z g c~ WEIGHTED<=4 @3
// Process: g d~ > z g d~ WEIGHTED<=4 @3
// Process: g s~ > z g s~ WEIGHTED<=4 @3

// Exception class
class PY8MEs_R3_P7_sm_gq_zgqException : public exception
{
  virtual const char * what() const throw()
  {
    return "Exception in class 'PY8MEs_R3_P7_sm_gq_zgq'."; 
  }
}
PY8MEs_R3_P7_sm_gq_zgq_exception; 

// Required s-channel initialization
int PY8MEs_R3_P7_sm_gq_zgq::req_s_channels[nreq_s_channels] = {}; 

int PY8MEs_R3_P7_sm_gq_zgq::helicities[ncomb][nexternal] = {{-1, -1, -1, -1,
    -1}, {-1, -1, -1, -1, 1}, {-1, -1, -1, 1, -1}, {-1, -1, -1, 1, 1}, {-1, -1,
    0, -1, -1}, {-1, -1, 0, -1, 1}, {-1, -1, 0, 1, -1}, {-1, -1, 0, 1, 1}, {-1,
    -1, 1, -1, -1}, {-1, -1, 1, -1, 1}, {-1, -1, 1, 1, -1}, {-1, -1, 1, 1, 1},
    {-1, 1, -1, -1, -1}, {-1, 1, -1, -1, 1}, {-1, 1, -1, 1, -1}, {-1, 1, -1, 1,
    1}, {-1, 1, 0, -1, -1}, {-1, 1, 0, -1, 1}, {-1, 1, 0, 1, -1}, {-1, 1, 0, 1,
    1}, {-1, 1, 1, -1, -1}, {-1, 1, 1, -1, 1}, {-1, 1, 1, 1, -1}, {-1, 1, 1, 1,
    1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, 1, -1}, {1, -1,
    -1, 1, 1}, {1, -1, 0, -1, -1}, {1, -1, 0, -1, 1}, {1, -1, 0, 1, -1}, {1,
    -1, 0, 1, 1}, {1, -1, 1, -1, -1}, {1, -1, 1, -1, 1}, {1, -1, 1, 1, -1}, {1,
    -1, 1, 1, 1}, {1, 1, -1, -1, -1}, {1, 1, -1, -1, 1}, {1, 1, -1, 1, -1}, {1,
    1, -1, 1, 1}, {1, 1, 0, -1, -1}, {1, 1, 0, -1, 1}, {1, 1, 0, 1, -1}, {1, 1,
    0, 1, 1}, {1, 1, 1, -1, -1}, {1, 1, 1, -1, 1}, {1, 1, 1, 1, -1}, {1, 1, 1,
    1, 1}};

//--------------------------------------------------------------------------
// Color config initialization
void PY8MEs_R3_P7_sm_gq_zgq::initColorConfigs() 
{
  color_configs = vector < vec_vec_int > (); 

  // Color flows of process Process: g u > z g u WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp00[2 * nexternal] = {1, 2, 3, 0, 0, 0, 3, 2, 1, 0}; 
  color_configs[0].push_back(vec_int(jamp00, jamp00 + (2 * nexternal))); 
  // JAMP #1
  int jamp01[2 * nexternal] = {3, 2, 2, 0, 0, 0, 3, 1, 1, 0}; 
  color_configs[0].push_back(vec_int(jamp01, jamp01 + (2 * nexternal))); 

  // Color flows of process Process: g d > z g d WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp10[2 * nexternal] = {1, 2, 3, 0, 0, 0, 3, 2, 1, 0}; 
  color_configs[1].push_back(vec_int(jamp10, jamp10 + (2 * nexternal))); 
  // JAMP #1
  int jamp11[2 * nexternal] = {3, 2, 2, 0, 0, 0, 3, 1, 1, 0}; 
  color_configs[1].push_back(vec_int(jamp11, jamp11 + (2 * nexternal))); 

  // Color flows of process Process: g u~ > z g u~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp20[2 * nexternal] = {1, 2, 0, 1, 0, 0, 3, 2, 0, 3}; 
  color_configs[2].push_back(vec_int(jamp20, jamp20 + (2 * nexternal))); 
  // JAMP #1
  int jamp21[2 * nexternal] = {3, 2, 0, 1, 0, 0, 3, 1, 0, 2}; 
  color_configs[2].push_back(vec_int(jamp21, jamp21 + (2 * nexternal))); 

  // Color flows of process Process: g d~ > z g d~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp30[2 * nexternal] = {1, 2, 0, 1, 0, 0, 3, 2, 0, 3}; 
  color_configs[3].push_back(vec_int(jamp30, jamp30 + (2 * nexternal))); 
  // JAMP #1
  int jamp31[2 * nexternal] = {3, 2, 0, 1, 0, 0, 3, 1, 0, 2}; 
  color_configs[3].push_back(vec_int(jamp31, jamp31 + (2 * nexternal))); 
}

//--------------------------------------------------------------------------
// Destructor.
PY8MEs_R3_P7_sm_gq_zgq::~PY8MEs_R3_P7_sm_gq_zgq() 
{
  for(int i = 0; i < nexternal; i++ )
  {
    delete[] p[i]; 
    p[i] = NULL; 
  }
}

//--------------------------------------------------------------------------
// Return the list of possible helicity configurations
vector < vec_int > PY8MEs_R3_P7_sm_gq_zgq::getHelicityConfigs(vector<int>
    permutation)
{
  vector<int> chosenPerm; 
  if (permutation.size() == 0)
  {
    chosenPerm = perm; 
  }
  else
  {
    chosenPerm = permutation; 
  }
  vector < vec_int > res(ncomb, vector<int> (nexternal, 0)); 
  for (int ihel = 0; ihel < ncomb; ihel++ )
  {
    for(int j = 0; j < nexternal; j++ )
    {
      res[ihel][chosenPerm[j]] = helicities[ihel][j]; 
    }
  }
  return res; 
}

//--------------------------------------------------------------------------
// Return the list of possible color configurations
vector < vec_int > PY8MEs_R3_P7_sm_gq_zgq::getColorConfigs(int specify_proc_ID,
    vector<int> permutation)
{
  int chosenProcID = -1; 
  if (specify_proc_ID == -1)
  {
    chosenProcID = proc_ID; 
  }
  else
  {
    chosenProcID = specify_proc_ID; 
  }
  vector<int> chosenPerm; 
  if (permutation.size() == 0)
  {
    chosenPerm = perm; 
  }
  else
  {
    chosenPerm = permutation; 
  }
  vector < vec_int > res(color_configs[chosenProcID].size(), vector<int>
      (nexternal * 2, 0));
  for (int icol = 0; icol < color_configs[chosenProcID].size(); icol++ )
  {
    for(int j = 0; j < (2 * nexternal); j++ )
    {
      res[icol][chosenPerm[j/2] * 2 + j%2] =
          color_configs[chosenProcID][icol][j];
    }
  }
  return res; 
}


//--------------------------------------------------------------------------
// Implements the map Helicity ID -> Helicity Config
vector<int> PY8MEs_R3_P7_sm_gq_zgq::getHelicityConfigForID(int hel_ID,
    vector<int> permutation)
{
  if (hel_ID < 0 || hel_ID >= ncomb)
  {
    cout <<  "Error in function 'getHelicityConfigForID' of class" << 
    " 'PY8MEs_R3_P7_sm_gq_zgq': Specified helicity ID '" << 
    hel_ID <<  "' cannot be found." << endl; 
    throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
  }
  vector<int> chosenPerm; 
  if (permutation.size() == 0)
  {
    chosenPerm = perm; 
  }
  else
  {
    chosenPerm = permutation; 
  }
  vector<int> res(nexternal, 0); 
  for (int j = 0; j < nexternal; j++ )
  {
    res[chosenPerm[j]] = helicities[hel_ID][j]; 
  }
  return res; 
}

//--------------------------------------------------------------------------
// Implements the map Helicity Config -> Helicity ID
int PY8MEs_R3_P7_sm_gq_zgq::getHelicityIDForConfig(vector<int> hel_config,
    vector<int> permutation)
{
  vector<int> chosenPerm; 
  if (permutation.size() == 0)
  {
    chosenPerm = perm; 
  }
  else
  {
    chosenPerm = permutation; 
  }
  if (permutation.size() == 0) {
    chosenPerm = invert_mapping(perm); 
  } else {
    chosenPerm = invert_mapping(permutation);
  }
  int user_ihel = -1; 
  if (hel_config.size() > 0)
  {
    bool found = false; 
    for(int i = 0; i < ncomb; i++ )
    {
      found = true; 
      for (int j = 0; j < nexternal; j++ )
      {
        if (helicities[i][chosenPerm[j]] != hel_config[j])
        {
          found = false; 
          break; 
        }
      }
      if ( !found)
        continue; 
      else
      {
        user_ihel = i; 
        break; 
      }
    }
    if (user_ihel == -1)
    {
      cout <<  "Error in function 'getHelicityIDForConfig' of class" << 
      " 'PY8MEs_R3_P7_sm_gq_zgq': Specified helicity" << 
      " configuration cannot be found." << endl; 
      throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
    }
  }
  return user_ihel; 
}


//--------------------------------------------------------------------------
// Implements the map Color ID -> Color Config
vector<int> PY8MEs_R3_P7_sm_gq_zgq::getColorConfigForID(int color_ID, int
    specify_proc_ID, vector<int> permutation)
{
  int chosenProcID = -1; 
  if (specify_proc_ID == -1)
  {
    chosenProcID = proc_ID; 
  }
  else
  {
    chosenProcID = specify_proc_ID; 
  }
  if (color_ID < 0 || color_ID >= int(color_configs[chosenProcID].size()))
  {
    cout <<  "Error in function 'getColorConfigForID' of class" << 
    " 'PY8MEs_R3_P7_sm_gq_zgq': Specified color ID '" << 
    color_ID <<  "' cannot be found." << endl; 
    throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
  }
  vector<int> chosenPerm; 
  if (permutation.size() == 0)
  {
    chosenPerm = perm; 
  }
  else
  {
    chosenPerm = permutation; 
  }
  vector<int> res(color_configs[chosenProcID][color_ID].size(), 0); 
  for (int j = 0; j < (2 * nexternal); j++ )
  {
    res[chosenPerm[j/2] * 2 + j%2] = color_configs[chosenProcID][color_ID][j]; 
  }
  return res; 
}

//--------------------------------------------------------------------------
// Implements the map Color Config -> Color ID
int PY8MEs_R3_P7_sm_gq_zgq::getColorIDForConfig(vector<int> color_config, int
    specify_proc_ID, vector<int> permutation)
{
  int chosenProcID = -1; 
  if (specify_proc_ID == -1)
  {
    chosenProcID = proc_ID; 
  }
  else
  {
    chosenProcID = specify_proc_ID; 
  }
  vector<int> chosenPerm; 
  if (permutation.size() == 0)
  {
    chosenPerm = perm; 
  }
  else
  {
    chosenPerm = permutation; 
  }
  if (permutation.size() == 0) {
    chosenPerm = invert_mapping(perm); 
  } else {
    chosenPerm = invert_mapping(permutation);
  }
  // Find which color configuration is asked for
  // -1 indicates one wants to sum over all color configurations
  int user_icol = -1; 
  if (color_config.size() > 0)
  {
    bool found = false; 
    for(int i = 0; i < color_configs[chosenProcID].size(); i++ )
    {
      found = true; 
      for (int j = 0; j < (nexternal * 2); j++ )
      {

        // If colorless then make sure it matches
        // The little arithmetics in the color index is just
        // the permutation applies on the particle list which is
        // twice smaller since each particle can have two color indices.
        if (color_config[j] == 0)
        {
          if (color_configs[chosenProcID][i][chosenPerm[j/2] * 2 + j%2] != 0)
          {
            found = false; 
            break; 
          }
          // Otherwise check that the color linked position matches
        }
        else
        {
          int color_linked_pos = -1; 
          // Find the other end of the line in the user color config
          for (int k = 0; j < (nexternal * 2); k++ )
          {
            if (k == j)
              continue; 
            if (color_config[j] == color_config[k])
            {
              color_linked_pos = k; 
              break; 
            }
          }
          if (color_linked_pos == -1)
          {
            cout <<  "Error in function 'getColorIDForConfig' of class" << 
            " 'PY8MEs_R3_P7_sm_gq_zgq': A color line could " << 
            " not be closed." << endl;
            return -2; 
            //throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
          }
          // Now check whether the color line matches
          if (color_configs[chosenProcID][i][chosenPerm[j/2] * 2 + j%2] !=
              color_configs[chosenProcID][i][chosenPerm[color_linked_pos/2] * 2
              + color_linked_pos%2])
          {
            found = false; 
            break; 
          }
        }
      }
      if ( !found)
        continue; 
      else
      {
        user_icol = i; 
        break; 
      }
    }

    if (user_icol == -1)
    {
      cout <<  "Error in function 'getColorIDForConfig' of class" << 
      " 'PY8MEs_R3_P7_sm_gq_zgq': Specified color" << 
      " configuration cannot be found." << endl;
      return -2; 
      //throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
    }
  }
  return user_icol; 
}

//--------------------------------------------------------------------------
// Returns all result previously computed in SigmaKin
vector < vec_double > PY8MEs_R3_P7_sm_gq_zgq::getAllResults(int
    specify_proc_ID)
{
  int chosenProcID = -1; 
  if (specify_proc_ID == -1)
  {
    chosenProcID = proc_ID; 
  }
  else
  {
    chosenProcID = specify_proc_ID; 
  }
  return all_results[chosenProcID]; 
}

//--------------------------------------------------------------------------
// Returns a result previously computed in SigmaKin for a specific helicity
// and color ID. -1 means avg and summed over that characteristic.
double PY8MEs_R3_P7_sm_gq_zgq::getResult(int helicity_ID, int color_ID, int
    specify_proc_ID)
{
  if (helicity_ID < - 1 || helicity_ID >= ncomb)
  {
    cout <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R3_P7_sm_gq_zgq': Specified helicity ID '" << 
    helicity_ID <<  "' configuration cannot be found." << endl; 
    throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
  }
  int chosenProcID = -1; 
  if (specify_proc_ID == -1)
  {
    chosenProcID = proc_ID; 
  }
  else
  {
    chosenProcID = specify_proc_ID; 
  }
  if (color_ID < - 1 || color_ID >= int(color_configs[chosenProcID].size()))
  {
    cout <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R3_P7_sm_gq_zgq': Specified color ID '" << 
    color_ID <<  "' configuration cannot be found." << endl; 
    throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
  }
  return all_results[chosenProcID][helicity_ID + 1][color_ID + 1]; 
}

//--------------------------------------------------------------------------
// Check for the availability of the requested process and if available,
// If available, this returns the corresponding permutation and Proc_ID to use.
// If not available, this returns a negative Proc_ID.
pair < vector<int> , int > PY8MEs_R3_P7_sm_gq_zgq::static_getPY8ME(vector<int>
    initial_pdgs, vector<int> final_pdgs, set<int> schannels)
{

  // Not available return value
  pair < vector<int> , int > NA(vector<int> (), -1); 

  // Check if s-channel requirements match
  if (nreq_s_channels > 0)
  {
    std::set<int> s_channel_proc(req_s_channels, req_s_channels +
        nreq_s_channels);
    if (schannels != s_channel_proc)
      return NA; 
  }
  else
  {
    if (schannels.size() != 0)
      return NA; 
  }

  // Check number of final state particles
  if (final_pdgs.size() != (nexternal - ninitial))
    return NA; 

  // Check number of initial state particles
  if (initial_pdgs.size() != ninitial)
    return NA; 

  // List of processes available in this class
  const int nprocs = 16; 
  const int proc_IDS[nprocs] = {0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3,
      3};
  const int in_pdgs[nprocs][ninitial] = {{21, 2}, {21, 4}, {21, 1}, {21, 3},
      {21, -2}, {21, -4}, {21, -1}, {21, -3}, {2, 21}, {4, 21}, {1, 21}, {3,
      21}, {-2, 21}, {-4, 21}, {-1, 21}, {-3, 21}};
  const int out_pdgs[nprocs][nexternal - ninitial] = {{23, 21, 2}, {23, 21, 4},
      {23, 21, 1}, {23, 21, 3}, {23, 21, -2}, {23, 21, -4}, {23, 21, -1}, {23,
      21, -3}, {23, 21, 2}, {23, 21, 4}, {23, 21, 1}, {23, 21, 3}, {23, 21,
      -2}, {23, 21, -4}, {23, 21, -1}, {23, 21, -3}};

  bool in_pdgs_used[ninitial]; 
  bool out_pdgs_used[nexternal - ninitial]; 
  for(int i = 0; i < nprocs; i++ )
  {
    int permutations[nexternal]; 

    // Reinitialize initial state look-up variables
    for(int j = 0; j < ninitial; j++ )
    {
      in_pdgs_used[j] = false; 
      permutations[j] = -1; 
    }
    // Look for initial state matches
    for(int j = 0; j < ninitial; j++ )
    {
      for(int k = 0; k < ninitial; k++ )
      {
        // Make sure it has not been used already
        if (in_pdgs_used[k])
          continue; 
        if (initial_pdgs[k] == in_pdgs[i][j])
        {
          permutations[j] = k; 
          in_pdgs_used[k] = true; 
          break; 
        }
      }
      // If no match found for this particular initial state,
      // proceed with the next process
      if (permutations[j] == -1)
        break; 
    }
    // Proceed with next process if not match found
    if (permutations[ninitial - 1] == -1)
      continue; 

    // Reinitialize final state look-up variables
    for(int j = 0; j < (nexternal - ninitial); j++ )
    {
      out_pdgs_used[j] = false; 
      permutations[ninitial + j] = -1; 
    }
    // Look for final state matches
    for(int j = 0; j < (nexternal - ninitial); j++ )
    {
      for(int k = 0; k < (nexternal - ninitial); k++ )
      {
        // Make sure it has not been used already
        if (out_pdgs_used[k])
          continue; 
        if (final_pdgs[k] == out_pdgs[i][j])
        {
          permutations[ninitial + j] = ninitial + k; 
          out_pdgs_used[k] = true; 
          break; 
        }
      }
      // If no match found for this particular initial state,
      // proceed with the next process
      if (permutations[ninitial + j] == -1)
        break; 
    }
    // Proceed with next process if not match found
    if (permutations[nexternal - 1] == -1)
      continue; 

    // Return process found
    return pair < vector<int> , int > (vector<int> (permutations, permutations
        + nexternal), proc_IDS[i]);
  }

  // No process found
  return NA; 
}

//--------------------------------------------------------------------------
// Set momenta
void PY8MEs_R3_P7_sm_gq_zgq::setMomenta(vector < vec_double > momenta_picked)
{
  if (momenta_picked.size() != nexternal)
  {
    cout <<  "Error in function 'setMomenta' of class" << 
    " 'PY8MEs_R3_P7_sm_gq_zgq': Incorrect number of" << 
    " momenta specified." << endl; 
    throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
  }
  for(int i = 0; i < nexternal; i++ )
  {
    if (momenta_picked[i].size() != 4)
    {
      cout <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R3_P7_sm_gq_zgq': Incorrect number of" << 
      " momenta components specified." << endl; 
      throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
    }
    if (momenta_picked[i][0] < 0.0)
    {
      cout <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R3_P7_sm_gq_zgq': A momentum was specified" << 
      " with negative energy. Check conventions." << endl; 
      throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
    }
    for (int j = 0; j < 4; j++ )
    {
      p[i][j] = momenta_picked[i][j]; 
    }
  }
}

//--------------------------------------------------------------------------
// Set color configuration to use. An empty vector means sum over all.
void PY8MEs_R3_P7_sm_gq_zgq::setColors(vector<int> colors_picked)
{
  if (colors_picked.size() == 0)
  {
    user_colors = vector<int> (); 
    return; 
  }
  if (colors_picked.size() != (2 * nexternal))
  {
    cout <<  "Error in function 'setColors' of class" << 
    " 'PY8MEs_R3_P7_sm_gq_zgq': Incorrect number" << 
    " of colors specified." << endl; 
    throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
  }
  user_colors = vector<int> ((2 * nexternal), 0); 
  for(int i = 0; i < (2 * nexternal); i++ )
  {
    user_colors[i] = colors_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the helicity configuration to use. Am empty vector means sum over all.
void PY8MEs_R3_P7_sm_gq_zgq::setHelicities(vector<int> helicities_picked) 
{
  if (helicities_picked.size() != nexternal)
  {
    if (helicities_picked.size() == 0)
    {
      user_helicities = vector<int> (); 
      return; 
    }
    cout <<  "Error in function 'setHelicities' of class" << 
    " 'PY8MEs_R3_P7_sm_gq_zgq': Incorrect number" << 
    " of helicities specified." << endl; 
    throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
  }
  user_helicities = vector<int> (nexternal, 0); 
  for(int i = 0; i < nexternal; i++ )
  {
    user_helicities[i] = helicities_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the permutation to use (will apply to momenta, colors and helicities)
void PY8MEs_R3_P7_sm_gq_zgq::setPermutation(vector<int> perm_picked) 
{
  if (perm_picked.size() != nexternal)
  {
    cout <<  "Error in function 'setPermutations' of class" << 
    " 'PY8MEs_R3_P7_sm_gq_zgq': Incorrect number" << 
    " of permutations specified." << endl; 
    throw PY8MEs_R3_P7_sm_gq_zgq_exception; 
  }
  for(int i = 0; i < nexternal; i++ )
  {
    perm[i] = perm_picked[i]; 
  }
}

// Set the proc_ID to use
void PY8MEs_R3_P7_sm_gq_zgq::setProcID(int procID_picked) 
{
  proc_ID = procID_picked; 
}

//--------------------------------------------------------------------------
// Initialize process.

void PY8MEs_R3_P7_sm_gq_zgq::initProc() 
{
  // Initialize vectors.
  perm = vector<int> (nexternal, 0); 
  user_colors = vector<int> (2 * nexternal, 0); 
  user_helicities = vector<int> (nexternal, 0); 
  p = vector < double * > (); 
  for (int i = 0; i < nexternal; i++ )
  {
    p.push_back(new double[4]); 
  }
  initColorConfigs(); 
  // Instantiate the model class and set parameters that stay fixed during run
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->mdl_MZ); 
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->ZERO); 
  jamp2 = vector < vec_double > (4); 
  jamp2[0] = vector<double> (2, 0.); 
  jamp2[1] = vector<double> (2, 0.); 
  jamp2[2] = vector<double> (2, 0.); 
  jamp2[3] = vector<double> (2, 0.); 
  all_results = vector < vec_vec_double > (4); 
  // The first entry is always the color or helicity avg/summed matrix element.
  all_results[0] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[1] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[2] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[3] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
}

//--------------------------------------------------------------------------
// Evaluate the squared matrix element.

double PY8MEs_R3_P7_sm_gq_zgq::sigmaKin() 
{
  // Set the parameters which change event by event
  pars->setDependentParameters(); 
  pars->setDependentCouplings(); 
  // Reset color flows
  for(int i = 0; i < 2; i++ )
    jamp2[0][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[1][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[2][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[3][i] = 0.; 

  // Local variables and constants
  const int max_tries = 10; 
  vector < vec_bool > goodhel(nprocesses, vec_bool(ncomb, false)); 
  vec_int ntry(nprocesses, 0); 
  double t = 0.; 
  double result = 0.; 

  // Denominators: spins, colors and identical particles
  const int denom_colors[nprocesses] = {24, 24, 24, 24}; 
  const int denom_hels[nprocesses] = {4, 4, 4, 4}; 
  const int denom_iden[nprocesses] = {1, 1, 1, 1}; 

  if (ntry[proc_ID] <= max_tries)
    ntry[proc_ID] = ntry[proc_ID] + 1; 

  // Find which helicity configuration is asked for
  // -1 indicates one wants to sum over helicities
  int user_ihel = getHelicityIDForConfig(user_helicities); 

  // Find which color configuration is asked for
  // -1 indicates one wants to sum over all color configurations
  int user_icol = getColorIDForConfig(user_colors); 

if ( user_icol == -2) return -1.;

  // Reset the list of results that will be recomputed here
  // Starts with -1 which are the summed results
  for (int ihel = -1; ihel + 1 < all_results[proc_ID].size(); ihel++ )
  {
    // Only if it is the helicity picked
    if (user_ihel != -1 && ihel != user_ihel)
      continue; 
    for (int icolor = -1; icolor + 1 < all_results[proc_ID][ihel + 1].size();
        icolor++ )
    {
      // Only if color picked
      if (user_icol != -1 && icolor != user_icol)
        continue; 
      all_results[proc_ID][ihel + 1][icolor + 1] = 0.; 
    }
  }

  // Calculate the matrix element for all helicities
  // unless already detected as vanishing
  for(int ihel = 0; ihel < ncomb; ihel++ )
  {
    // Skip helicity if already detected as vanishing
    if ((ntry[proc_ID] >= max_tries) && !goodhel[proc_ID][ihel])
      continue; 

    // Also skip helicity if user asks for a specific one
    if ((ntry[proc_ID] >= max_tries) && user_ihel != -1 && user_ihel != ihel)
      continue; 

    calculate_wavefunctions(helicities[ihel]); 

    // Reset locally computed color flows
    for(int i = 0; i < 2; i++ )
      jamp2[0][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[1][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[2][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[3][i] = 0.; 

    if (proc_ID == 0)
      t = matrix_3_gu_zgu(); 
    if (proc_ID == 1)
      t = matrix_3_gd_zgd(); 
    if (proc_ID == 2)
      t = matrix_3_gux_zgux(); 
    if (proc_ID == 3)
      t = matrix_3_gdx_zgdx(); 

    // Store which helicities give non-zero result
    if ((ntry[proc_ID] < max_tries) && t != 0. && !goodhel[proc_ID][ihel])
      goodhel[proc_ID][ihel] = true; 

    // Aggregate results
    if (user_ihel == -1 || user_ihel == ihel)
    {
      if (user_icol == -1)
      {
        result = result + t; 
        if (user_ihel == -1)
        {
          all_results[proc_ID][0][0] += t; 
          for (int i = 0; i < jamp2[proc_ID].size(); i++ )
          {
            all_results[proc_ID][0][i + 1] += jamp2[proc_ID][i]; 
          }
        }
        all_results[proc_ID][ihel + 1][0] += t; 
        for (int i = 0; i < jamp2[proc_ID].size(); i++ )
        {
          all_results[proc_ID][ihel + 1][i + 1] += jamp2[proc_ID][i]; 
        }
      }
      else
      {
        result = result + jamp2[proc_ID][user_icol]; 
        if (user_ihel == -1)
        {
          all_results[proc_ID][0][user_icol + 1] += jamp2[proc_ID][user_icol]; 
        }
        all_results[proc_ID][ihel + 1][user_icol + 1] +=
            jamp2[proc_ID][user_icol];
      }
    }

  }

  // Normalize results with the identical particle factor
  result = result/denom_iden[proc_ID]; 
  // Starts with -1 which are the summed results
  for (int ihel = -1; ihel + 1 < all_results[proc_ID].size(); ihel++ )
  {
    // Only if it is the helicity picked
    if (user_ihel != -1 && ihel != user_ihel)
      continue; 
    for (int icolor = -1; icolor + 1 < all_results[proc_ID][ihel + 1].size();
        icolor++ )
    {
      // Only if color picked
      if (user_icol != -1 && icolor != user_icol)
        continue; 
      all_results[proc_ID][ihel + 1][icolor + 1] /= denom_iden[proc_ID]; 
    }
  }


  // Normalize when when summing+averaging over helicity configurations
  if (user_ihel == -1)
  {
    result /= denom_hels[proc_ID]; 
    if (user_icol == -1)
    {
      all_results[proc_ID][0][0] /= denom_hels[proc_ID]; 
      for (int i = 0; i < jamp2[proc_ID].size(); i++ )
      {
        all_results[proc_ID][0][i + 1] /= denom_hels[proc_ID]; 
      }
    }
    else
    {
      all_results[proc_ID][0][user_icol + 1] /= denom_hels[proc_ID]; 
    }
  }

  // Normalize when summing+averaging over color configurations
  if (user_icol == -1)
  {
    result /= denom_colors[proc_ID]; 
    if (user_ihel == -1)
    {
      all_results[proc_ID][0][0] /= denom_colors[proc_ID]; 
      for (int i = 0; i < ncomb; i++ )
      {
        all_results[proc_ID][i + 1][0] /= denom_colors[proc_ID]; 
      }
    }
    else
    {
      all_results[proc_ID][user_ihel + 1][0] /= denom_colors[proc_ID]; 
    }
  }

  // Finally return it
  return result; 

}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

void PY8MEs_R3_P7_sm_gq_zgq::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  // Calculate all wavefunctions
  vxxxxx(p[perm[0]], mME[0], hel[0], -1, w[0]); 
  ixxxxx(p[perm[1]], mME[1], hel[1], +1, w[1]); 
  vxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  vxxxxx(p[perm[3]], mME[3], hel[3], +1, w[3]); 
  oxxxxx(p[perm[4]], mME[4], hel[4], +1, w[4]); 
  VVV1P0_1(w[0], w[3], pars->GC_10, pars->ZERO, pars->ZERO, w[5]); 
  FFV2_5_1(w[4], w[2], pars->GC_51, pars->GC_58, pars->ZERO, pars->ZERO, w[6]); 
  FFV2_5_2(w[1], w[2], pars->GC_51, pars->GC_58, pars->ZERO, pars->ZERO, w[7]); 
  FFV1_1(w[4], w[0], pars->GC_11, pars->ZERO, pars->ZERO, w[8]); 
  FFV1_2(w[1], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[9]); 
  FFV1_2(w[1], w[0], pars->GC_11, pars->ZERO, pars->ZERO, w[10]); 
  FFV1_1(w[4], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[11]); 
  FFV2_3_1(w[4], w[2], pars->GC_50, pars->GC_58, pars->ZERO, pars->ZERO,
      w[12]);
  FFV2_3_2(w[1], w[2], pars->GC_50, pars->GC_58, pars->ZERO, pars->ZERO,
      w[13]);
  oxxxxx(p[perm[1]], mME[1], hel[1], -1, w[14]); 
  ixxxxx(p[perm[4]], mME[4], hel[4], -1, w[15]); 
  FFV2_5_1(w[14], w[2], pars->GC_51, pars->GC_58, pars->ZERO, pars->ZERO,
      w[16]);
  FFV2_5_2(w[15], w[2], pars->GC_51, pars->GC_58, pars->ZERO, pars->ZERO,
      w[17]);
  FFV1_1(w[14], w[0], pars->GC_11, pars->ZERO, pars->ZERO, w[18]); 
  FFV1_2(w[15], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[19]); 
  FFV1_2(w[15], w[0], pars->GC_11, pars->ZERO, pars->ZERO, w[20]); 
  FFV1_1(w[14], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[21]); 
  FFV2_3_1(w[14], w[2], pars->GC_50, pars->GC_58, pars->ZERO, pars->ZERO,
      w[22]);
  FFV2_3_2(w[15], w[2], pars->GC_50, pars->GC_58, pars->ZERO, pars->ZERO,
      w[23]);

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[1], w[6], w[5], pars->GC_11, amp[0]); 
  FFV1_0(w[7], w[4], w[5], pars->GC_11, amp[1]); 
  FFV2_5_0(w[9], w[8], w[2], pars->GC_51, pars->GC_58, amp[2]); 
  FFV1_0(w[7], w[8], w[3], pars->GC_11, amp[3]); 
  FFV2_5_0(w[10], w[11], w[2], pars->GC_51, pars->GC_58, amp[4]); 
  FFV1_0(w[10], w[6], w[3], pars->GC_11, amp[5]); 
  FFV1_0(w[7], w[11], w[0], pars->GC_11, amp[6]); 
  FFV1_0(w[9], w[6], w[0], pars->GC_11, amp[7]); 
  FFV1_0(w[1], w[12], w[5], pars->GC_11, amp[8]); 
  FFV1_0(w[13], w[4], w[5], pars->GC_11, amp[9]); 
  FFV2_3_0(w[9], w[8], w[2], pars->GC_50, pars->GC_58, amp[10]); 
  FFV1_0(w[13], w[8], w[3], pars->GC_11, amp[11]); 
  FFV2_3_0(w[10], w[11], w[2], pars->GC_50, pars->GC_58, amp[12]); 
  FFV1_0(w[10], w[12], w[3], pars->GC_11, amp[13]); 
  FFV1_0(w[13], w[11], w[0], pars->GC_11, amp[14]); 
  FFV1_0(w[9], w[12], w[0], pars->GC_11, amp[15]); 
  FFV1_0(w[15], w[16], w[5], pars->GC_11, amp[16]); 
  FFV1_0(w[17], w[14], w[5], pars->GC_11, amp[17]); 
  FFV2_5_0(w[19], w[18], w[2], pars->GC_51, pars->GC_58, amp[18]); 
  FFV1_0(w[17], w[18], w[3], pars->GC_11, amp[19]); 
  FFV2_5_0(w[20], w[21], w[2], pars->GC_51, pars->GC_58, amp[20]); 
  FFV1_0(w[20], w[16], w[3], pars->GC_11, amp[21]); 
  FFV1_0(w[17], w[21], w[0], pars->GC_11, amp[22]); 
  FFV1_0(w[19], w[16], w[0], pars->GC_11, amp[23]); 
  FFV1_0(w[15], w[22], w[5], pars->GC_11, amp[24]); 
  FFV1_0(w[23], w[14], w[5], pars->GC_11, amp[25]); 
  FFV2_3_0(w[19], w[18], w[2], pars->GC_50, pars->GC_58, amp[26]); 
  FFV1_0(w[23], w[18], w[3], pars->GC_11, amp[27]); 
  FFV2_3_0(w[20], w[21], w[2], pars->GC_50, pars->GC_58, amp[28]); 
  FFV1_0(w[20], w[22], w[3], pars->GC_11, amp[29]); 
  FFV1_0(w[23], w[21], w[0], pars->GC_11, amp[30]); 
  FFV1_0(w[19], w[22], w[0], pars->GC_11, amp[31]); 


}
double PY8MEs_R3_P7_sm_gq_zgq::matrix_3_gu_zgu() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 8; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {3, 3}; 
  double cf[ncolor][ncolor] = {{16, -2}, {-2, 16}}; 

  // Calculate color flows
  jamp[0] = -std::complex<double> (0, 1) * amp[0] - std::complex<double> (0, 1)
      * amp[1] + amp[2] + amp[3] + amp[7];
  jamp[1] = +std::complex<double> (0, 1) * amp[0] + std::complex<double> (0, 1)
      * amp[1] + amp[4] + amp[5] + amp[6];

  // Sum and square the color flows to get the matrix element
  double matrix = 0; 
  for(i = 0; i < ncolor; i++ )
  {
    ztemp = 0.; 
    for(j = 0; j < ncolor; j++ )
      ztemp = ztemp + cf[i][j] * jamp[j]; 
    matrix = matrix + real(ztemp * conj(jamp[i]))/denom[i]; 
  }

  // Store the leading color flows for choice of color
  for(i = 0; i < ncolor; i++ )
    jamp2[0][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P7_sm_gq_zgq::matrix_3_gd_zgd() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 8; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {3, 3}; 
  double cf[ncolor][ncolor] = {{16, -2}, {-2, 16}}; 

  // Calculate color flows
  jamp[0] = -std::complex<double> (0, 1) * amp[8] - std::complex<double> (0, 1)
      * amp[9] + amp[10] + amp[11] + amp[15];
  jamp[1] = +std::complex<double> (0, 1) * amp[8] + std::complex<double> (0, 1)
      * amp[9] + amp[12] + amp[13] + amp[14];

  // Sum and square the color flows to get the matrix element
  double matrix = 0; 
  for(i = 0; i < ncolor; i++ )
  {
    ztemp = 0.; 
    for(j = 0; j < ncolor; j++ )
      ztemp = ztemp + cf[i][j] * jamp[j]; 
    matrix = matrix + real(ztemp * conj(jamp[i]))/denom[i]; 
  }

  // Store the leading color flows for choice of color
  for(i = 0; i < ncolor; i++ )
    jamp2[1][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P7_sm_gq_zgq::matrix_3_gux_zgux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 8; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {3, 3}; 
  double cf[ncolor][ncolor] = {{16, -2}, {-2, 16}}; 

  // Calculate color flows
  jamp[0] = +std::complex<double> (0, 1) * amp[16] + std::complex<double> (0,
      1) * amp[17] - amp[18] - amp[19] - amp[23];
  jamp[1] = -std::complex<double> (0, 1) * amp[16] - std::complex<double> (0,
      1) * amp[17] - amp[20] - amp[21] - amp[22];

  // Sum and square the color flows to get the matrix element
  double matrix = 0; 
  for(i = 0; i < ncolor; i++ )
  {
    ztemp = 0.; 
    for(j = 0; j < ncolor; j++ )
      ztemp = ztemp + cf[i][j] * jamp[j]; 
    matrix = matrix + real(ztemp * conj(jamp[i]))/denom[i]; 
  }

  // Store the leading color flows for choice of color
  for(i = 0; i < ncolor; i++ )
    jamp2[2][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P7_sm_gq_zgq::matrix_3_gdx_zgdx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 8; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {3, 3}; 
  double cf[ncolor][ncolor] = {{16, -2}, {-2, 16}}; 

  // Calculate color flows
  jamp[0] = +std::complex<double> (0, 1) * amp[24] + std::complex<double> (0,
      1) * amp[25] - amp[26] - amp[27] - amp[31];
  jamp[1] = -std::complex<double> (0, 1) * amp[24] - std::complex<double> (0,
      1) * amp[25] - amp[28] - amp[29] - amp[30];

  // Sum and square the color flows to get the matrix element
  double matrix = 0; 
  for(i = 0; i < ncolor; i++ )
  {
    ztemp = 0.; 
    for(j = 0; j < ncolor; j++ )
      ztemp = ztemp + cf[i][j] * jamp[j]; 
    matrix = matrix + real(ztemp * conj(jamp[i]))/denom[i]; 
  }

  // Store the leading color flows for choice of color
  for(i = 0; i < ncolor; i++ )
    jamp2[3][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}


}  // end namespace PY8MEs_namespace

