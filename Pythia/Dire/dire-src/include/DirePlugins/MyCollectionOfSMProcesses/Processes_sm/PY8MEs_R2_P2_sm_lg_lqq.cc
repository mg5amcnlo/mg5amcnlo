//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "PY8MEs_R2_P2_sm_lg_lqq.h"
#include "HelAmps_sm.h"

using namespace Pythia8_sm; 

namespace PY8MEs_namespace 
{
//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: e- g > e- u u~ WEIGHTED<=5 @2
// Process: e- g > e- c c~ WEIGHTED<=5 @2
// Process: mu- g > mu- u u~ WEIGHTED<=5 @2
// Process: mu- g > mu- c c~ WEIGHTED<=5 @2
// Process: e- g > e- d d~ WEIGHTED<=5 @2
// Process: e- g > e- s s~ WEIGHTED<=5 @2
// Process: mu- g > mu- d d~ WEIGHTED<=5 @2
// Process: mu- g > mu- s s~ WEIGHTED<=5 @2
// Process: e+ g > e+ u u~ WEIGHTED<=5 @2
// Process: e+ g > e+ c c~ WEIGHTED<=5 @2
// Process: mu+ g > mu+ u u~ WEIGHTED<=5 @2
// Process: mu+ g > mu+ c c~ WEIGHTED<=5 @2
// Process: e+ g > e+ d d~ WEIGHTED<=5 @2
// Process: e+ g > e+ s s~ WEIGHTED<=5 @2
// Process: mu+ g > mu+ d d~ WEIGHTED<=5 @2
// Process: mu+ g > mu+ s s~ WEIGHTED<=5 @2
// Process: e- g > ve d u~ WEIGHTED<=5 @2
// Process: e- g > ve s c~ WEIGHTED<=5 @2
// Process: mu- g > vm d u~ WEIGHTED<=5 @2
// Process: mu- g > vm s c~ WEIGHTED<=5 @2
// Process: ve g > e- u d~ WEIGHTED<=5 @2
// Process: ve g > e- c s~ WEIGHTED<=5 @2
// Process: vm g > mu- u d~ WEIGHTED<=5 @2
// Process: vm g > mu- c s~ WEIGHTED<=5 @2
// Process: ve g > ve u u~ WEIGHTED<=5 @2
// Process: ve g > ve c c~ WEIGHTED<=5 @2
// Process: vm g > vm u u~ WEIGHTED<=5 @2
// Process: vm g > vm c c~ WEIGHTED<=5 @2
// Process: vt g > vt u u~ WEIGHTED<=5 @2
// Process: vt g > vt c c~ WEIGHTED<=5 @2
// Process: ve g > ve d d~ WEIGHTED<=5 @2
// Process: ve g > ve s s~ WEIGHTED<=5 @2
// Process: vm g > vm d d~ WEIGHTED<=5 @2
// Process: vm g > vm s s~ WEIGHTED<=5 @2
// Process: vt g > vt d d~ WEIGHTED<=5 @2
// Process: vt g > vt s s~ WEIGHTED<=5 @2
// Process: e+ g > ve~ u d~ WEIGHTED<=5 @2
// Process: e+ g > ve~ c s~ WEIGHTED<=5 @2
// Process: mu+ g > vm~ u d~ WEIGHTED<=5 @2
// Process: mu+ g > vm~ c s~ WEIGHTED<=5 @2
// Process: ve~ g > e+ d u~ WEIGHTED<=5 @2
// Process: ve~ g > e+ s c~ WEIGHTED<=5 @2
// Process: vm~ g > mu+ d u~ WEIGHTED<=5 @2
// Process: vm~ g > mu+ s c~ WEIGHTED<=5 @2
// Process: ve~ g > ve~ u u~ WEIGHTED<=5 @2
// Process: ve~ g > ve~ c c~ WEIGHTED<=5 @2
// Process: vm~ g > vm~ u u~ WEIGHTED<=5 @2
// Process: vm~ g > vm~ c c~ WEIGHTED<=5 @2
// Process: vt~ g > vt~ u u~ WEIGHTED<=5 @2
// Process: vt~ g > vt~ c c~ WEIGHTED<=5 @2
// Process: ve~ g > ve~ d d~ WEIGHTED<=5 @2
// Process: ve~ g > ve~ s s~ WEIGHTED<=5 @2
// Process: vm~ g > vm~ d d~ WEIGHTED<=5 @2
// Process: vm~ g > vm~ s s~ WEIGHTED<=5 @2
// Process: vt~ g > vt~ d d~ WEIGHTED<=5 @2
// Process: vt~ g > vt~ s s~ WEIGHTED<=5 @2

// Exception class
class PY8MEs_R2_P2_sm_lg_lqqException : public exception
{
  virtual const char * what() const throw()
  {
    return "Exception in class 'PY8MEs_R2_P2_sm_lg_lqq'."; 
  }
}
PY8MEs_R2_P2_sm_lg_lqq_exception; 

int PY8MEs_R2_P2_sm_lg_lqq::helicities[ncomb][nexternal] = {{-1, -1, -1, -1,
    -1}, {-1, -1, -1, -1, 1}, {-1, -1, -1, 1, -1}, {-1, -1, -1, 1, 1}, {-1, -1,
    1, -1, -1}, {-1, -1, 1, -1, 1}, {-1, -1, 1, 1, -1}, {-1, -1, 1, 1, 1}, {-1,
    1, -1, -1, -1}, {-1, 1, -1, -1, 1}, {-1, 1, -1, 1, -1}, {-1, 1, -1, 1, 1},
    {-1, 1, 1, -1, -1}, {-1, 1, 1, -1, 1}, {-1, 1, 1, 1, -1}, {-1, 1, 1, 1, 1},
    {1, -1, -1, -1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, 1, -1}, {1, -1, -1, 1,
    1}, {1, -1, 1, -1, -1}, {1, -1, 1, -1, 1}, {1, -1, 1, 1, -1}, {1, -1, 1, 1,
    1}, {1, 1, -1, -1, -1}, {1, 1, -1, -1, 1}, {1, 1, -1, 1, -1}, {1, 1, -1, 1,
    1}, {1, 1, 1, -1, -1}, {1, 1, 1, -1, 1}, {1, 1, 1, 1, -1}, {1, 1, 1, 1, 1}};

//--------------------------------------------------------------------------
// Color config initialization
void PY8MEs_R2_P2_sm_lg_lqq::initColorConfigs() 
{
  color_configs = vector < vec_vec_int > (); 
  jamp_nc_relative_power = vector < vec_int > (); 

  // Color flows of process Process: e- g > e- u u~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp00[2 * nexternal] = {0, 0, 1, 2, 0, 0, 1, 0, 0, 2}; 
  color_configs[0].push_back(vec_int(jamp00, jamp00 + (2 * nexternal))); 
  jamp_nc_relative_power[0].push_back(0); 

  // Color flows of process Process: e- g > e- d d~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp10[2 * nexternal] = {0, 0, 1, 2, 0, 0, 1, 0, 0, 2}; 
  color_configs[1].push_back(vec_int(jamp10, jamp10 + (2 * nexternal))); 
  jamp_nc_relative_power[1].push_back(0); 

  // Color flows of process Process: e+ g > e+ u u~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp20[2 * nexternal] = {0, 0, 1, 2, 0, 0, 1, 0, 0, 2}; 
  color_configs[2].push_back(vec_int(jamp20, jamp20 + (2 * nexternal))); 
  jamp_nc_relative_power[2].push_back(0); 

  // Color flows of process Process: e+ g > e+ d d~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp30[2 * nexternal] = {0, 0, 1, 2, 0, 0, 1, 0, 0, 2}; 
  color_configs[3].push_back(vec_int(jamp30, jamp30 + (2 * nexternal))); 
  jamp_nc_relative_power[3].push_back(0); 

  // Color flows of process Process: e- g > ve d u~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp40[2 * nexternal] = {0, 0, 1, 2, 0, 0, 1, 0, 0, 2}; 
  color_configs[4].push_back(vec_int(jamp40, jamp40 + (2 * nexternal))); 
  jamp_nc_relative_power[4].push_back(0); 

  // Color flows of process Process: ve g > ve u u~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp50[2 * nexternal] = {0, 0, 1, 2, 0, 0, 1, 0, 0, 2}; 
  color_configs[5].push_back(vec_int(jamp50, jamp50 + (2 * nexternal))); 
  jamp_nc_relative_power[5].push_back(0); 

  // Color flows of process Process: ve g > ve d d~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp60[2 * nexternal] = {0, 0, 1, 2, 0, 0, 1, 0, 0, 2}; 
  color_configs[6].push_back(vec_int(jamp60, jamp60 + (2 * nexternal))); 
  jamp_nc_relative_power[6].push_back(0); 

  // Color flows of process Process: e+ g > ve~ u d~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp70[2 * nexternal] = {0, 0, 1, 2, 0, 0, 1, 0, 0, 2}; 
  color_configs[7].push_back(vec_int(jamp70, jamp70 + (2 * nexternal))); 
  jamp_nc_relative_power[7].push_back(0); 

  // Color flows of process Process: ve~ g > ve~ u u~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp80[2 * nexternal] = {0, 0, 1, 2, 0, 0, 1, 0, 0, 2}; 
  color_configs[8].push_back(vec_int(jamp80, jamp80 + (2 * nexternal))); 
  jamp_nc_relative_power[8].push_back(0); 

  // Color flows of process Process: ve~ g > ve~ d d~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp90[2 * nexternal] = {0, 0, 1, 2, 0, 0, 1, 0, 0, 2}; 
  color_configs[9].push_back(vec_int(jamp90, jamp90 + (2 * nexternal))); 
  jamp_nc_relative_power[9].push_back(0); 
}

//--------------------------------------------------------------------------
// Destructor.
PY8MEs_R2_P2_sm_lg_lqq::~PY8MEs_R2_P2_sm_lg_lqq() 
{
  for(int i = 0; i < nexternal; i++ )
  {
    delete[] p[i]; 
    p[i] = NULL; 
  }
}

//--------------------------------------------------------------------------
// Invert the permutation mapping
vector<int> PY8MEs_R2_P2_sm_lg_lqq::invert_mapping(vector<int> mapping) 
{
  vector<int> inverted_mapping; 
  for (int i = 0; i < mapping.size(); i++ )
  {
    for (int j = 0; j < mapping.size(); j++ )
    {
      if (mapping[j] == i)
      {
        inverted_mapping.push_back(j); 
        break; 
      }
    }
  }
  return inverted_mapping; 
}

//--------------------------------------------------------------------------
// Return the list of possible helicity configurations
vector < vec_int > PY8MEs_R2_P2_sm_lg_lqq::getHelicityConfigs(vector<int>
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
vector < vec_int > PY8MEs_R2_P2_sm_lg_lqq::getColorConfigs(int specify_proc_ID,
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
// Get JAMP relative N_c power
int PY8MEs_R2_P2_sm_lg_lqq::getColorFlowRelativeNCPower(int color_flow_ID, int
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
  return jamp_nc_relative_power[chosenProcID][color_flow_ID]; 
}

//--------------------------------------------------------------------------
// Implements the map Helicity ID -> Helicity Config
vector<int> PY8MEs_R2_P2_sm_lg_lqq::getHelicityConfigForID(int hel_ID,
    vector<int> permutation)
{
  if (hel_ID < 0 || hel_ID >= ncomb)
  {
    cerr <<  "Error in function 'getHelicityConfigForID' of class" << 
    " 'PY8MEs_R2_P2_sm_lg_lqq': Specified helicity ID '" << 
    hel_ID <<  "' cannot be found." << endl; 
    throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
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
int PY8MEs_R2_P2_sm_lg_lqq::getHelicityIDForConfig(vector<int> hel_config,
    vector<int> permutation)
{
  vector<int> chosenPerm; 
  if (permutation.size() == 0)
  {
    chosenPerm = invert_mapping(perm); 
  }
  else
  {
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
      cerr <<  "Error in function 'getHelicityIDForConfig' of class" << 
      " 'PY8MEs_R2_P2_sm_lg_lqq': Specified helicity" << 
      " configuration cannot be found." << endl; 
      throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
    }
  }
  return user_ihel; 
}


//--------------------------------------------------------------------------
// Implements the map Color ID -> Color Config
vector<int> PY8MEs_R2_P2_sm_lg_lqq::getColorConfigForID(int color_ID, int
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
    cerr <<  "Error in function 'getColorConfigForID' of class" << 
    " 'PY8MEs_R2_P2_sm_lg_lqq': Specified color ID '" << 
    color_ID <<  "' cannot be found." << endl; 
    throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
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
int PY8MEs_R2_P2_sm_lg_lqq::getColorIDForConfig(vector<int> color_config, int
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
    chosenPerm = invert_mapping(perm); 
  }
  else
  {
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
            cerr <<  "Error in function 'getColorIDForConfig' of class" << 
            " 'PY8MEs_R2_P2_sm_lg_lqq': A color line could " << 
            " not be closed." << endl; 
            throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
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
      cerr <<  "Error in function 'getColorIDForConfig' of class" << 
      " 'PY8MEs_R2_P2_sm_lg_lqq': Specified color" << 
      " configuration cannot be found." << endl; 
      throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
    }
  }
  return user_icol; 
}

//--------------------------------------------------------------------------
// Returns all result previously computed in SigmaKin
vector < vec_double > PY8MEs_R2_P2_sm_lg_lqq::getAllResults(int
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
double PY8MEs_R2_P2_sm_lg_lqq::getResult(int helicity_ID, int color_ID, int
    specify_proc_ID)
{
  if (helicity_ID < - 1 || helicity_ID >= ncomb)
  {
    cerr <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R2_P2_sm_lg_lqq': Specified helicity ID '" << 
    helicity_ID <<  "' configuration cannot be found." << endl; 
    throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
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
    cerr <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R2_P2_sm_lg_lqq': Specified color ID '" << 
    color_ID <<  "' configuration cannot be found." << endl; 
    throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
  }
  return all_results[chosenProcID][helicity_ID + 1][color_ID + 1]; 
}

//--------------------------------------------------------------------------
// Check for the availability of the requested process and if available,
// If available, this returns the corresponding permutation and Proc_ID to use.
// If not available, this returns a negative Proc_ID.
pair < vector<int> , int > PY8MEs_R2_P2_sm_lg_lqq::static_getPY8ME(vector<int>
    initial_pdgs, vector<int> final_pdgs, set<int> schannels)
{

  // Not available return value
  pair < vector<int> , int > NA(vector<int> (), -1); 

  // Check if s-channel requirements match
  if (nreq_s_channels > 0)
  {
    std::set<int> s_channel_proc;
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
  const int nprocs = 56; 
  const int proc_IDS[nprocs] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
      4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
      7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9};
  const int in_pdgs[nprocs][ninitial] = {{11, 21}, {11, 21}, {13, 21}, {13,
      21}, {11, 21}, {11, 21}, {13, 21}, {13, 21}, {-11, 21}, {-11, 21}, {-13,
      21}, {-13, 21}, {-11, 21}, {-11, 21}, {-13, 21}, {-13, 21}, {11, 21},
      {11, 21}, {13, 21}, {13, 21}, {12, 21}, {12, 21}, {14, 21}, {14, 21},
      {12, 21}, {12, 21}, {14, 21}, {14, 21}, {16, 21}, {16, 21}, {12, 21},
      {12, 21}, {14, 21}, {14, 21}, {16, 21}, {16, 21}, {-11, 21}, {-11, 21},
      {-13, 21}, {-13, 21}, {-12, 21}, {-12, 21}, {-14, 21}, {-14, 21}, {-12,
      21}, {-12, 21}, {-14, 21}, {-14, 21}, {-16, 21}, {-16, 21}, {-12, 21},
      {-12, 21}, {-14, 21}, {-14, 21}, {-16, 21}, {-16, 21}};
  const int out_pdgs[nprocs][nexternal - ninitial] = {{11, 2, -2}, {11, 4, -4},
      {13, 2, -2}, {13, 4, -4}, {11, 1, -1}, {11, 3, -3}, {13, 1, -1}, {13, 3,
      -3}, {-11, 2, -2}, {-11, 4, -4}, {-13, 2, -2}, {-13, 4, -4}, {-11, 1,
      -1}, {-11, 3, -3}, {-13, 1, -1}, {-13, 3, -3}, {12, 1, -2}, {12, 3, -4},
      {14, 1, -2}, {14, 3, -4}, {11, 2, -1}, {11, 4, -3}, {13, 2, -1}, {13, 4,
      -3}, {12, 2, -2}, {12, 4, -4}, {14, 2, -2}, {14, 4, -4}, {16, 2, -2},
      {16, 4, -4}, {12, 1, -1}, {12, 3, -3}, {14, 1, -1}, {14, 3, -3}, {16, 1,
      -1}, {16, 3, -3}, {-12, 2, -1}, {-12, 4, -3}, {-14, 2, -1}, {-14, 4, -3},
      {-11, 1, -2}, {-11, 3, -4}, {-13, 1, -2}, {-13, 3, -4}, {-12, 2, -2},
      {-12, 4, -4}, {-14, 2, -2}, {-14, 4, -4}, {-16, 2, -2}, {-16, 4, -4},
      {-12, 1, -1}, {-12, 3, -3}, {-14, 1, -1}, {-14, 3, -3}, {-16, 1, -1},
      {-16, 3, -3}};

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
void PY8MEs_R2_P2_sm_lg_lqq::setMomenta(vector < vec_double > momenta_picked)
{
  if (momenta_picked.size() != nexternal)
  {
    cerr <<  "Error in function 'setMomenta' of class" << 
    " 'PY8MEs_R2_P2_sm_lg_lqq': Incorrect number of" << 
    " momenta specified." << endl; 
    throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
  }
  for(int i = 0; i < nexternal; i++ )
  {
    if (momenta_picked[i].size() != 4)
    {
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R2_P2_sm_lg_lqq': Incorrect number of" << 
      " momenta components specified." << endl; 
      throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
    }
    if (momenta_picked[i][0] < 0.0)
    {
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R2_P2_sm_lg_lqq': A momentum was specified" << 
      " with negative energy. Check conventions." << endl; 
      throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
    }
    for (int j = 0; j < 4; j++ )
    {
      p[i][j] = momenta_picked[i][j]; 
    }
  }
}

//--------------------------------------------------------------------------
// Set color configuration to use. An empty vector means sum over all.
void PY8MEs_R2_P2_sm_lg_lqq::setColors(vector<int> colors_picked)
{
  if (colors_picked.size() == 0)
  {
    user_colors = vector<int> (); 
    return; 
  }
  if (colors_picked.size() != (2 * nexternal))
  {
    cerr <<  "Error in function 'setColors' of class" << 
    " 'PY8MEs_R2_P2_sm_lg_lqq': Incorrect number" << 
    " of colors specified." << endl; 
    throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
  }
  user_colors = vector<int> ((2 * nexternal), 0); 
  for(int i = 0; i < (2 * nexternal); i++ )
  {
    user_colors[i] = colors_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the helicity configuration to use. Am empty vector means sum over all.
void PY8MEs_R2_P2_sm_lg_lqq::setHelicities(vector<int> helicities_picked) 
{
  if (helicities_picked.size() != nexternal)
  {
    if (helicities_picked.size() == 0)
    {
      user_helicities = vector<int> (); 
      return; 
    }
    cerr <<  "Error in function 'setHelicities' of class" << 
    " 'PY8MEs_R2_P2_sm_lg_lqq': Incorrect number" << 
    " of helicities specified." << endl; 
    throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
  }
  user_helicities = vector<int> (nexternal, 0); 
  for(int i = 0; i < nexternal; i++ )
  {
    user_helicities[i] = helicities_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the permutation to use (will apply to momenta, colors and helicities)
void PY8MEs_R2_P2_sm_lg_lqq::setPermutation(vector<int> perm_picked) 
{
  if (perm_picked.size() != nexternal)
  {
    cerr <<  "Error in function 'setPermutations' of class" << 
    " 'PY8MEs_R2_P2_sm_lg_lqq': Incorrect number" << 
    " of permutations specified." << endl; 
    throw PY8MEs_R2_P2_sm_lg_lqq_exception; 
  }
  for(int i = 0; i < nexternal; i++ )
  {
    perm[i] = perm_picked[i]; 
  }
}

// Set the proc_ID to use
void PY8MEs_R2_P2_sm_lg_lqq::setProcID(int procID_picked) 
{
  proc_ID = procID_picked; 
}

//--------------------------------------------------------------------------
// Initialize process.

void PY8MEs_R2_P2_sm_lg_lqq::initProc() 
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
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->ZERO); 
  jamp2 = vector < vec_double > (10); 
  jamp2[0] = vector<double> (1, 0.); 
  jamp2[1] = vector<double> (1, 0.); 
  jamp2[2] = vector<double> (1, 0.); 
  jamp2[3] = vector<double> (1, 0.); 
  jamp2[4] = vector<double> (1, 0.); 
  jamp2[5] = vector<double> (1, 0.); 
  jamp2[6] = vector<double> (1, 0.); 
  jamp2[7] = vector<double> (1, 0.); 
  jamp2[8] = vector<double> (1, 0.); 
  jamp2[9] = vector<double> (1, 0.); 
  all_results = vector < vec_vec_double > (10); 
  // The first entry is always the color or helicity avg/summed matrix element.
  all_results[0] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[1] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[2] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[3] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[4] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[5] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[6] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[7] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[8] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[9] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
}

//--------------------------------------------------------------------------
// Evaluate the squared matrix element.

double PY8MEs_R2_P2_sm_lg_lqq::sigmaKin() 
{
  // Set the parameters which change event by event
  pars->setDependentParameters(); 
  pars->setDependentCouplings(); 
  // Reset color flows
  for(int i = 0; i < 1; i++ )
    jamp2[0][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[1][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[2][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[3][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[4][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[5][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[6][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[7][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[8][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[9][i] = 0.; 

  // Local variables and constants
  const int max_tries = 10; 
  vector < vec_bool > goodhel(nprocesses, vec_bool(ncomb, false)); 
  vec_int ntry(nprocesses, 0); 
  double t = 0.; 
  double result = 0.; 

  // Denominators: spins, colors and identical particles
  const int denom_colors[nprocesses] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8}; 
  const int denom_hels[nprocesses] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4}; 
  const int denom_iden[nprocesses] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; 

  if (ntry[proc_ID] <= max_tries)
    ntry[proc_ID] = ntry[proc_ID] + 1; 

  // Find which helicity configuration is asked for
  // -1 indicates one wants to sum over helicities
  int user_ihel = getHelicityIDForConfig(user_helicities); 

  // Find which color configuration is asked for
  // -1 indicates one wants to sum over all color configurations
  int user_icol = getColorIDForConfig(user_colors); 

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
    for(int i = 0; i < 1; i++ )
      jamp2[0][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[1][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[2][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[3][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[4][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[5][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[6][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[7][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[8][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[9][i] = 0.; 

    if (proc_ID == 0)
      t = matrix_2_emg_emuux(); 
    if (proc_ID == 1)
      t = matrix_2_emg_emddx(); 
    if (proc_ID == 2)
      t = matrix_2_epg_epuux(); 
    if (proc_ID == 3)
      t = matrix_2_epg_epddx(); 
    if (proc_ID == 4)
      t = matrix_2_emg_vedux(); 
    if (proc_ID == 5)
      t = matrix_2_veg_veuux(); 
    if (proc_ID == 6)
      t = matrix_2_veg_veddx(); 
    if (proc_ID == 7)
      t = matrix_2_epg_vexudx(); 
    if (proc_ID == 8)
      t = matrix_2_vexg_vexuux(); 
    if (proc_ID == 9)
      t = matrix_2_vexg_vexddx(); 

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
  if (include_symmetry_factors)
  {
    result = result/denom_iden[proc_ID]; 
  }
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
      if (include_symmetry_factors)
      {
        all_results[proc_ID][ihel + 1][icolor + 1] /= denom_iden[proc_ID]; 
      }
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

void PY8MEs_R2_P2_sm_lg_lqq::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  // Calculate all wavefunctions
  ixxxxx(p[perm[0]], mME[0], hel[0], +1, w[0]); 
  vxxxxx(p[perm[1]], mME[1], hel[1], -1, w[1]); 
  oxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  oxxxxx(p[perm[3]], mME[3], hel[3], +1, w[3]); 
  ixxxxx(p[perm[4]], mME[4], hel[4], -1, w[4]); 
  FFV1P0_3(w[0], w[2], pars->GC_3, pars->ZERO, pars->ZERO, w[5]); 
  FFV1_1(w[3], w[1], pars->GC_11, pars->ZERO, pars->ZERO, w[6]); 
  FFV2_4_3(w[0], w[2], pars->GC_50, pars->GC_59, pars->mdl_MZ, pars->mdl_WZ,
      w[7]);
  FFV1_2(w[4], w[1], pars->GC_11, pars->ZERO, pars->ZERO, w[8]); 
  oxxxxx(p[perm[0]], mME[0], hel[0], -1, w[9]); 
  ixxxxx(p[perm[2]], mME[2], hel[2], -1, w[10]); 
  FFV1P0_3(w[10], w[9], pars->GC_3, pars->ZERO, pars->ZERO, w[11]); 
  FFV2_4_3(w[10], w[9], pars->GC_50, pars->GC_59, pars->mdl_MZ, pars->mdl_WZ,
      w[12]);
  FFV2_3(w[0], w[2], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[13]); 
  FFV2_3(w[0], w[2], pars->GC_62, pars->mdl_MZ, pars->mdl_WZ, w[14]); 
  FFV2_3(w[10], w[9], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[15]); 
  FFV2_3(w[10], w[9], pars->GC_62, pars->mdl_MZ, pars->mdl_WZ, w[16]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[4], w[6], w[5], pars->GC_2, amp[0]); 
  FFV2_5_0(w[4], w[6], w[7], pars->GC_51, pars->GC_58, amp[1]); 
  FFV1_0(w[8], w[3], w[5], pars->GC_2, amp[2]); 
  FFV2_5_0(w[8], w[3], w[7], pars->GC_51, pars->GC_58, amp[3]); 
  FFV1_0(w[4], w[6], w[5], pars->GC_1, amp[4]); 
  FFV2_3_0(w[4], w[6], w[7], pars->GC_50, pars->GC_58, amp[5]); 
  FFV1_0(w[8], w[3], w[5], pars->GC_1, amp[6]); 
  FFV2_3_0(w[8], w[3], w[7], pars->GC_50, pars->GC_58, amp[7]); 
  FFV1_0(w[4], w[6], w[11], pars->GC_2, amp[8]); 
  FFV2_5_0(w[4], w[6], w[12], pars->GC_51, pars->GC_58, amp[9]); 
  FFV1_0(w[8], w[3], w[11], pars->GC_2, amp[10]); 
  FFV2_5_0(w[8], w[3], w[12], pars->GC_51, pars->GC_58, amp[11]); 
  FFV1_0(w[4], w[6], w[11], pars->GC_1, amp[12]); 
  FFV2_3_0(w[4], w[6], w[12], pars->GC_50, pars->GC_58, amp[13]); 
  FFV1_0(w[8], w[3], w[11], pars->GC_1, amp[14]); 
  FFV2_3_0(w[8], w[3], w[12], pars->GC_50, pars->GC_58, amp[15]); 
  FFV2_0(w[4], w[6], w[13], pars->GC_100, amp[16]); 
  FFV2_0(w[8], w[3], w[13], pars->GC_100, amp[17]); 
  FFV2_5_0(w[4], w[6], w[14], pars->GC_51, pars->GC_58, amp[18]); 
  FFV2_5_0(w[8], w[3], w[14], pars->GC_51, pars->GC_58, amp[19]); 
  FFV2_3_0(w[4], w[6], w[14], pars->GC_50, pars->GC_58, amp[20]); 
  FFV2_3_0(w[8], w[3], w[14], pars->GC_50, pars->GC_58, amp[21]); 
  FFV2_0(w[4], w[6], w[15], pars->GC_100, amp[22]); 
  FFV2_0(w[8], w[3], w[15], pars->GC_100, amp[23]); 
  FFV2_5_0(w[4], w[6], w[16], pars->GC_51, pars->GC_58, amp[24]); 
  FFV2_5_0(w[8], w[3], w[16], pars->GC_51, pars->GC_58, amp[25]); 
  FFV2_3_0(w[4], w[6], w[16], pars->GC_50, pars->GC_58, amp[26]); 
  FFV2_3_0(w[8], w[3], w[16], pars->GC_50, pars->GC_58, amp[27]); 


}
double PY8MEs_R2_P2_sm_lg_lqq::matrix_2_emg_emuux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 1; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{4}}; 

  // Calculate color flows
  jamp[0] = -amp[0] - amp[1] - amp[2] - amp[3]; 

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
    jamp2[0][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P2_sm_lg_lqq::matrix_2_emg_emddx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 1; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{4}}; 

  // Calculate color flows
  jamp[0] = -amp[4] - amp[5] - amp[6] - amp[7]; 

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
    jamp2[1][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P2_sm_lg_lqq::matrix_2_epg_epuux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 1; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{4}}; 

  // Calculate color flows
  jamp[0] = +amp[8] + amp[9] + amp[10] + amp[11]; 

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
    jamp2[2][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P2_sm_lg_lqq::matrix_2_epg_epddx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 1; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{4}}; 

  // Calculate color flows
  jamp[0] = +amp[12] + amp[13] + amp[14] + amp[15]; 

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
    jamp2[3][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P2_sm_lg_lqq::matrix_2_emg_vedux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 2; 
  const int ncolor = 1; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{4}}; 

  // Calculate color flows
  jamp[0] = -amp[16] - amp[17]; 

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
    jamp2[4][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P2_sm_lg_lqq::matrix_2_veg_veuux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 2; 
  const int ncolor = 1; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{4}}; 

  // Calculate color flows
  jamp[0] = -amp[18] - amp[19]; 

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
    jamp2[5][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P2_sm_lg_lqq::matrix_2_veg_veddx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 2; 
  const int ncolor = 1; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{4}}; 

  // Calculate color flows
  jamp[0] = -amp[20] - amp[21]; 

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
    jamp2[6][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P2_sm_lg_lqq::matrix_2_epg_vexudx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 2; 
  const int ncolor = 1; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{4}}; 

  // Calculate color flows
  jamp[0] = +amp[22] + amp[23]; 

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
    jamp2[7][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P2_sm_lg_lqq::matrix_2_vexg_vexuux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 2; 
  const int ncolor = 1; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{4}}; 

  // Calculate color flows
  jamp[0] = +amp[24] + amp[25]; 

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
    jamp2[8][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P2_sm_lg_lqq::matrix_2_vexg_vexddx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 2; 
  const int ncolor = 1; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{4}}; 

  // Calculate color flows
  jamp[0] = +amp[26] + amp[27]; 

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
    jamp2[9][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}


}  // end namespace PY8MEs_namespace

