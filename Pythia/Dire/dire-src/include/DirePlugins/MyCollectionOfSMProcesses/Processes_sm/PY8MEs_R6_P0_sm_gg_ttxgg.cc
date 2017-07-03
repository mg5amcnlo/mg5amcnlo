//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.5.3, 2017-03-09
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "PY8MEs_R6_P0_sm_gg_ttxgg.h"
#include "HelAmps_sm.h"

using namespace Pythia8_sm; 

namespace PY8MEs_namespace 
{
//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: g g > t t~ g g WEIGHTED<=4 @6

// Exception class
class PY8MEs_R6_P0_sm_gg_ttxggException : public exception
{
  virtual const char * what() const throw()
  {
    return "Exception in class 'PY8MEs_R6_P0_sm_gg_ttxgg'."; 
  }
}
PY8MEs_R6_P0_sm_gg_ttxgg_exception; 

// Required s-channel initialization
int PY8MEs_R6_P0_sm_gg_ttxgg::req_s_channels[nreq_s_channels] = {}; 

int PY8MEs_R6_P0_sm_gg_ttxgg::helicities[ncomb][nexternal] = {{-1, -1, -1, -1,
    -1, -1}, {-1, -1, -1, -1, -1, 1}, {-1, -1, -1, -1, 1, -1}, {-1, -1, -1, -1,
    1, 1}, {-1, -1, -1, 1, -1, -1}, {-1, -1, -1, 1, -1, 1}, {-1, -1, -1, 1, 1,
    -1}, {-1, -1, -1, 1, 1, 1}, {-1, -1, 1, -1, -1, -1}, {-1, -1, 1, -1, -1,
    1}, {-1, -1, 1, -1, 1, -1}, {-1, -1, 1, -1, 1, 1}, {-1, -1, 1, 1, -1, -1},
    {-1, -1, 1, 1, -1, 1}, {-1, -1, 1, 1, 1, -1}, {-1, -1, 1, 1, 1, 1}, {-1, 1,
    -1, -1, -1, -1}, {-1, 1, -1, -1, -1, 1}, {-1, 1, -1, -1, 1, -1}, {-1, 1,
    -1, -1, 1, 1}, {-1, 1, -1, 1, -1, -1}, {-1, 1, -1, 1, -1, 1}, {-1, 1, -1,
    1, 1, -1}, {-1, 1, -1, 1, 1, 1}, {-1, 1, 1, -1, -1, -1}, {-1, 1, 1, -1, -1,
    1}, {-1, 1, 1, -1, 1, -1}, {-1, 1, 1, -1, 1, 1}, {-1, 1, 1, 1, -1, -1},
    {-1, 1, 1, 1, -1, 1}, {-1, 1, 1, 1, 1, -1}, {-1, 1, 1, 1, 1, 1}, {1, -1,
    -1, -1, -1, -1}, {1, -1, -1, -1, -1, 1}, {1, -1, -1, -1, 1, -1}, {1, -1,
    -1, -1, 1, 1}, {1, -1, -1, 1, -1, -1}, {1, -1, -1, 1, -1, 1}, {1, -1, -1,
    1, 1, -1}, {1, -1, -1, 1, 1, 1}, {1, -1, 1, -1, -1, -1}, {1, -1, 1, -1, -1,
    1}, {1, -1, 1, -1, 1, -1}, {1, -1, 1, -1, 1, 1}, {1, -1, 1, 1, -1, -1}, {1,
    -1, 1, 1, -1, 1}, {1, -1, 1, 1, 1, -1}, {1, -1, 1, 1, 1, 1}, {1, 1, -1, -1,
    -1, -1}, {1, 1, -1, -1, -1, 1}, {1, 1, -1, -1, 1, -1}, {1, 1, -1, -1, 1,
    1}, {1, 1, -1, 1, -1, -1}, {1, 1, -1, 1, -1, 1}, {1, 1, -1, 1, 1, -1}, {1,
    1, -1, 1, 1, 1}, {1, 1, 1, -1, -1, -1}, {1, 1, 1, -1, -1, 1}, {1, 1, 1, -1,
    1, -1}, {1, 1, 1, -1, 1, 1}, {1, 1, 1, 1, -1, -1}, {1, 1, 1, 1, -1, 1}, {1,
    1, 1, 1, 1, -1}, {1, 1, 1, 1, 1, 1}};

//--------------------------------------------------------------------------
// Color config initialization
void PY8MEs_R6_P0_sm_gg_ttxgg::initColorConfigs() 
{
  color_configs = vector < vec_vec_int > (); 

  // Color flows of process Process: g g > t t~ g g WEIGHTED<=4 @6
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp00[2 * nexternal] = {1, 2, 2, 3, 1, 0, 0, 5, 4, 3, 5, 4}; 
  color_configs[0].push_back(vec_int(jamp00, jamp00 + (2 * nexternal))); 
  // JAMP #1
  int jamp01[2 * nexternal] = {1, 2, 2, 3, 1, 0, 0, 4, 4, 5, 5, 3}; 
  color_configs[0].push_back(vec_int(jamp01, jamp01 + (2 * nexternal))); 
  // JAMP #2
  int jamp02[2 * nexternal] = {1, 2, 4, 3, 1, 0, 0, 5, 4, 2, 5, 3}; 
  color_configs[0].push_back(vec_int(jamp02, jamp02 + (2 * nexternal))); 
  // JAMP #3
  int jamp03[2 * nexternal] = {1, 2, 5, 3, 1, 0, 0, 3, 4, 2, 5, 4}; 
  color_configs[0].push_back(vec_int(jamp03, jamp03 + (2 * nexternal))); 
  // JAMP #4
  int jamp04[2 * nexternal] = {1, 2, 5, 3, 1, 0, 0, 4, 4, 3, 5, 2}; 
  color_configs[0].push_back(vec_int(jamp04, jamp04 + (2 * nexternal))); 
  // JAMP #5
  int jamp05[2 * nexternal] = {1, 2, 4, 3, 1, 0, 0, 3, 4, 5, 5, 2}; 
  color_configs[0].push_back(vec_int(jamp05, jamp05 + (2 * nexternal))); 
  // JAMP #6
  int jamp06[2 * nexternal] = {3, 2, 1, 3, 1, 0, 0, 5, 4, 2, 5, 4}; 
  color_configs[0].push_back(vec_int(jamp06, jamp06 + (2 * nexternal))); 
  // JAMP #7
  int jamp07[2 * nexternal] = {3, 2, 1, 3, 1, 0, 0, 4, 4, 5, 5, 2}; 
  color_configs[0].push_back(vec_int(jamp07, jamp07 + (2 * nexternal))); 
  // JAMP #8
  int jamp08[2 * nexternal] = {4, 2, 1, 3, 1, 0, 0, 5, 4, 3, 5, 2}; 
  color_configs[0].push_back(vec_int(jamp08, jamp08 + (2 * nexternal))); 
  // JAMP #9
  int jamp09[2 * nexternal] = {5, 2, 1, 3, 1, 0, 0, 2, 4, 3, 5, 4}; 
  color_configs[0].push_back(vec_int(jamp09, jamp09 + (2 * nexternal))); 
  // JAMP #10
  int jamp010[2 * nexternal] = {5, 2, 1, 3, 1, 0, 0, 4, 4, 2, 5, 3}; 
  color_configs[0].push_back(vec_int(jamp010, jamp010 + (2 * nexternal))); 
  // JAMP #11
  int jamp011[2 * nexternal] = {4, 2, 1, 3, 1, 0, 0, 2, 4, 5, 5, 3}; 
  color_configs[0].push_back(vec_int(jamp011, jamp011 + (2 * nexternal))); 
  // JAMP #12
  int jamp012[2 * nexternal] = {4, 2, 2, 3, 1, 0, 0, 5, 4, 1, 5, 3}; 
  color_configs[0].push_back(vec_int(jamp012, jamp012 + (2 * nexternal))); 
  // JAMP #13
  int jamp013[2 * nexternal] = {4, 2, 5, 3, 1, 0, 0, 3, 4, 1, 5, 2}; 
  color_configs[0].push_back(vec_int(jamp013, jamp013 + (2 * nexternal))); 
  // JAMP #14
  int jamp014[2 * nexternal] = {3, 2, 4, 3, 1, 0, 0, 5, 4, 1, 5, 2}; 
  color_configs[0].push_back(vec_int(jamp014, jamp014 + (2 * nexternal))); 
  // JAMP #15
  int jamp015[2 * nexternal] = {5, 2, 4, 3, 1, 0, 0, 2, 4, 1, 5, 3}; 
  color_configs[0].push_back(vec_int(jamp015, jamp015 + (2 * nexternal))); 
  // JAMP #16
  int jamp016[2 * nexternal] = {5, 2, 2, 3, 1, 0, 0, 3, 4, 1, 5, 4}; 
  color_configs[0].push_back(vec_int(jamp016, jamp016 + (2 * nexternal))); 
  // JAMP #17
  int jamp017[2 * nexternal] = {3, 2, 5, 3, 1, 0, 0, 2, 4, 1, 5, 4}; 
  color_configs[0].push_back(vec_int(jamp017, jamp017 + (2 * nexternal))); 
  // JAMP #18
  int jamp018[2 * nexternal] = {5, 2, 2, 3, 1, 0, 0, 4, 4, 3, 5, 1}; 
  color_configs[0].push_back(vec_int(jamp018, jamp018 + (2 * nexternal))); 
  // JAMP #19
  int jamp019[2 * nexternal] = {5, 2, 4, 3, 1, 0, 0, 3, 4, 2, 5, 1}; 
  color_configs[0].push_back(vec_int(jamp019, jamp019 + (2 * nexternal))); 
  // JAMP #20
  int jamp020[2 * nexternal] = {3, 2, 5, 3, 1, 0, 0, 4, 4, 2, 5, 1}; 
  color_configs[0].push_back(vec_int(jamp020, jamp020 + (2 * nexternal))); 
  // JAMP #21
  int jamp021[2 * nexternal] = {4, 2, 5, 3, 1, 0, 0, 2, 4, 3, 5, 1}; 
  color_configs[0].push_back(vec_int(jamp021, jamp021 + (2 * nexternal))); 
  // JAMP #22
  int jamp022[2 * nexternal] = {4, 2, 2, 3, 1, 0, 0, 3, 4, 5, 5, 1}; 
  color_configs[0].push_back(vec_int(jamp022, jamp022 + (2 * nexternal))); 
  // JAMP #23
  int jamp023[2 * nexternal] = {3, 2, 4, 3, 1, 0, 0, 2, 4, 5, 5, 1}; 
  color_configs[0].push_back(vec_int(jamp023, jamp023 + (2 * nexternal))); 
}

//--------------------------------------------------------------------------
// Destructor.
PY8MEs_R6_P0_sm_gg_ttxgg::~PY8MEs_R6_P0_sm_gg_ttxgg() 
{
  for(int i = 0; i < nexternal; i++ )
  {
    delete[] p[i]; 
    p[i] = NULL; 
  }
}

//--------------------------------------------------------------------------
// Return the list of possible helicity configurations
vector < vec_int > PY8MEs_R6_P0_sm_gg_ttxgg::getHelicityConfigs(vector<int>
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
vector < vec_int > PY8MEs_R6_P0_sm_gg_ttxgg::getColorConfigs(int
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
vector<int> PY8MEs_R6_P0_sm_gg_ttxgg::getHelicityConfigForID(int hel_ID,
    vector<int> permutation)
{
  if (hel_ID < 0 || hel_ID >= ncomb)
  {
    cout <<  "Error in function 'getHelicityConfigForID' of class" << 
    " 'PY8MEs_R6_P0_sm_gg_ttxgg': Specified helicity ID '" << 
    hel_ID <<  "' cannot be found." << endl; 
    throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
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
int PY8MEs_R6_P0_sm_gg_ttxgg::getHelicityIDForConfig(vector<int> hel_config,
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
      " 'PY8MEs_R6_P0_sm_gg_ttxgg': Specified helicity" << 
      " configuration cannot be found." << endl; 
      throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
    }
  }
  return user_ihel; 
}


//--------------------------------------------------------------------------
// Implements the map Color ID -> Color Config
vector<int> PY8MEs_R6_P0_sm_gg_ttxgg::getColorConfigForID(int color_ID, int
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
    " 'PY8MEs_R6_P0_sm_gg_ttxgg': Specified color ID '" << 
    color_ID <<  "' cannot be found." << endl; 
    throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
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
int PY8MEs_R6_P0_sm_gg_ttxgg::getColorIDForConfig(vector<int> color_config, int
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

      //cout << "col config " << i << endl;
      //for(int j = 0; j < color_configs[chosenProcID][i].size(); j++ )
      //  cout << color_configs[chosenProcID][i][j] << " ";
      //cout << endl;


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
            " 'PY8MEs_R6_P0_sm_gg_ttxgg': A color line could " << 
            " not be closed." << endl;
            return -2; 
            //throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
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
      " 'PY8MEs_R6_P0_sm_gg_ttxgg': Specified color" << 
      " configuration cannot be found." << endl; 
      return -2;
      //throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
    }
  }
  return user_icol; 
}

//--------------------------------------------------------------------------
// Returns all result previously computed in SigmaKin
vector < vec_double > PY8MEs_R6_P0_sm_gg_ttxgg::getAllResults(int
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
double PY8MEs_R6_P0_sm_gg_ttxgg::getResult(int helicity_ID, int color_ID, int
    specify_proc_ID)
{
  if (helicity_ID < - 1 || helicity_ID >= ncomb)
  {
    cout <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R6_P0_sm_gg_ttxgg': Specified helicity ID '" << 
    helicity_ID <<  "' configuration cannot be found." << endl; 
    throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
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
    " 'PY8MEs_R6_P0_sm_gg_ttxgg': Specified color ID '" << 
    color_ID <<  "' configuration cannot be found." << endl; 
    throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
  }
  return all_results[chosenProcID][helicity_ID + 1][color_ID + 1]; 
}

//--------------------------------------------------------------------------
// Check for the availability of the requested process and if available,
// If available, this returns the corresponding permutation and Proc_ID to use.
// If not available, this returns a negative Proc_ID.
pair < vector<int> , int > PY8MEs_R6_P0_sm_gg_ttxgg::static_getPY8ME(vector<int> initial_pdgs, vector<int> final_pdgs, set<int> schannels) 
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
  const int nprocs = 1; 
  const int proc_IDS[nprocs] = {0}; 
  const int in_pdgs[nprocs][ninitial] = {{21, 21}}; 
  const int out_pdgs[nprocs][nexternal - ninitial] = {{6, -6, 21, 21}}; 

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
void PY8MEs_R6_P0_sm_gg_ttxgg::setMomenta(vector < vec_double > momenta_picked)
{
  if (momenta_picked.size() != nexternal)
  {
    cout <<  "Error in function 'setMomenta' of class" << 
    " 'PY8MEs_R6_P0_sm_gg_ttxgg': Incorrect number of" << 
    " momenta specified." << endl; 
    throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
  }
  for(int i = 0; i < nexternal; i++ )
  {
    if (momenta_picked[i].size() != 4)
    {
      cout <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R6_P0_sm_gg_ttxgg': Incorrect number of" << 
      " momenta components specified." << endl; 
      throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
    }
    if (momenta_picked[i][0] < 0.0)
    {
      cout <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R6_P0_sm_gg_ttxgg': A momentum was specified" << 
      " with negative energy. Check conventions." << endl; 
      throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
    }
    for (int j = 0; j < 4; j++ )
    {
      p[i][j] = momenta_picked[i][j]; 
    }
  }
}

//--------------------------------------------------------------------------
// Set color configuration to use. An empty vector means sum over all.
void PY8MEs_R6_P0_sm_gg_ttxgg::setColors(vector<int> colors_picked)
{
  if (colors_picked.size() == 0)
  {
    user_colors = vector<int> (); 
    return; 
  }
  if (colors_picked.size() != (2 * nexternal))
  {
    cout <<  "Error in function 'setColors' of class" << 
    " 'PY8MEs_R6_P0_sm_gg_ttxgg': Incorrect number" << 
    " of colors specified." << endl; 
    throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
  }
  user_colors = vector<int> ((2 * nexternal), 0); 
  for(int i = 0; i < (2 * nexternal); i++ )
  {
    user_colors[i] = colors_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the helicity configuration to use. Am empty vector means sum over all.
void PY8MEs_R6_P0_sm_gg_ttxgg::setHelicities(vector<int> helicities_picked) 
{
  if (helicities_picked.size() != nexternal)
  {
    if (helicities_picked.size() == 0)
    {
      user_helicities = vector<int> (); 
      return; 
    }
    cout <<  "Error in function 'setHelicities' of class" << 
    " 'PY8MEs_R6_P0_sm_gg_ttxgg': Incorrect number" << 
    " of helicities specified." << endl; 
    throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
  }
  user_helicities = vector<int> (nexternal, 0); 
  for(int i = 0; i < nexternal; i++ )
  {
    user_helicities[i] = helicities_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the permutation to use (will apply to momenta, colors and helicities)
void PY8MEs_R6_P0_sm_gg_ttxgg::setPermutation(vector<int> perm_picked) 
{
  if (perm_picked.size() != nexternal)
  {
    cout <<  "Error in function 'setPermutations' of class" << 
    " 'PY8MEs_R6_P0_sm_gg_ttxgg': Incorrect number" << 
    " of permutations specified." << endl; 
    throw PY8MEs_R6_P0_sm_gg_ttxgg_exception; 
  }
  for(int i = 0; i < nexternal; i++ )
  {
    perm[i] = perm_picked[i]; 
  }
}

// Set the proc_ID to use
void PY8MEs_R6_P0_sm_gg_ttxgg::setProcID(int procID_picked) 
{
  proc_ID = procID_picked; 
}

//--------------------------------------------------------------------------
// Initialize process.

void PY8MEs_R6_P0_sm_gg_ttxgg::initProc() 
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
  mME.push_back(pars->mdl_MT); 
  mME.push_back(pars->mdl_MT); 
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->ZERO); 
  jamp2 = vector < vec_double > (1); 
  jamp2[0] = vector<double> (24, 0.); 
  all_results = vector < vec_vec_double > (1); 
  // The first entry is always the color or helicity avg/summed matrix element.
  all_results[0] = vector < vec_double > (ncomb + 1, vector<double> (24 + 1,
      0.));
}

//--------------------------------------------------------------------------
// Evaluate the squared matrix element.

double PY8MEs_R6_P0_sm_gg_ttxgg::sigmaKin() 
{
  // Set the parameters which change event by event
  pars->setDependentParameters(); 
  pars->setDependentCouplings(); 
  // Reset color flows
  for(int i = 0; i < 24; i++ )
    jamp2[0][i] = 0.; 

  // Local variables and constants
  const int max_tries = 10; 
  vector < vec_bool > goodhel(nprocesses, vec_bool(ncomb, false)); 
  vec_int ntry(nprocesses, 0); 
  double t = 0.; 
  double result = 0.; 

  // Denominators: spins, colors and identical particles
  const int denom_colors[nprocesses] = {64}; 
  const int denom_hels[nprocesses] = {4}; 
  const int denom_iden[nprocesses] = {2}; 

  if (ntry[proc_ID] <= max_tries)
    ntry[proc_ID] = ntry[proc_ID] + 1; 

  // Find which helicity configuration is asked for
  // -1 indicates one wants to sum over helicities
  int user_ihel = getHelicityIDForConfig(user_helicities); 

  // Find which color configuration is asked for
  // -1 indicates one wants to sum over all color configurations
  int user_icol = getColorIDForConfig(user_colors); 

if (user_icol == -2) return -1.;

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
    for(int i = 0; i < 24; i++ )
      jamp2[0][i] = 0.; 

    if (proc_ID == 0)
      t = matrix_6_gg_ttxgg(); 

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

void PY8MEs_R6_P0_sm_gg_ttxgg::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  // Calculate all wavefunctions
  vxxxxx(p[perm[0]], mME[0], hel[0], -1, w[0]); 
  vxxxxx(p[perm[1]], mME[1], hel[1], -1, w[1]); 
  oxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  ixxxxx(p[perm[3]], mME[3], hel[3], -1, w[3]); 
  vxxxxx(p[perm[4]], mME[4], hel[4], +1, w[4]); 
  vxxxxx(p[perm[5]], mME[5], hel[5], +1, w[5]); 
  VVV1P0_1(w[0], w[1], pars->GC_10, pars->ZERO, pars->ZERO, w[6]); 
  FFV1P0_3(w[3], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[7]); 
  VVV1P0_1(w[6], w[4], pars->GC_10, pars->ZERO, pars->ZERO, w[8]); 
  VVV1P0_1(w[6], w[5], pars->GC_10, pars->ZERO, pars->ZERO, w[9]); 
  VVV1P0_1(w[4], w[5], pars->GC_10, pars->ZERO, pars->ZERO, w[10]); 
  FFV1_1(w[2], w[4], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[11]); 
  FFV1_2(w[3], w[6], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[12]); 
  FFV1_2(w[3], w[5], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[13]); 
  FFV1_1(w[2], w[5], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[14]); 
  FFV1_2(w[3], w[4], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[15]); 
  FFV1_1(w[2], w[6], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[16]); 
  FFV1_1(w[2], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[17]); 
  FFV1_2(w[3], w[1], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[18]); 
  FFV1_1(w[17], w[4], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[19]); 
  FFV1_1(w[17], w[5], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[20]); 
  VVV1P0_1(w[1], w[4], pars->GC_10, pars->ZERO, pars->ZERO, w[21]); 
  FFV1P0_3(w[3], w[17], pars->GC_11, pars->ZERO, pars->ZERO, w[22]); 
  VVV1P0_1(w[1], w[5], pars->GC_10, pars->ZERO, pars->ZERO, w[23]); 
  FFV1_1(w[17], w[1], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[24]); 
  VVVV1P0_1(w[1], w[4], w[5], pars->GC_12, pars->ZERO, pars->ZERO, w[25]); 
  VVVV3P0_1(w[1], w[4], w[5], pars->GC_12, pars->ZERO, pars->ZERO, w[26]); 
  VVVV4P0_1(w[1], w[4], w[5], pars->GC_12, pars->ZERO, pars->ZERO, w[27]); 
  FFV1_2(w[3], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[28]); 
  FFV1_1(w[2], w[1], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[29]); 
  FFV1_2(w[28], w[4], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[30]); 
  FFV1_2(w[28], w[5], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[31]); 
  FFV1P0_3(w[28], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[32]); 
  FFV1_2(w[28], w[1], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[33]); 
  VVV1P0_1(w[0], w[4], pars->GC_10, pars->ZERO, pars->ZERO, w[34]); 
  FFV1_2(w[3], w[34], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[35]); 
  VVV1P0_1(w[34], w[5], pars->GC_10, pars->ZERO, pars->ZERO, w[36]); 
  FFV1_1(w[2], w[34], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[37]); 
  VVV1P0_1(w[34], w[1], pars->GC_10, pars->ZERO, pars->ZERO, w[38]); 
  VVV1P0_1(w[0], w[5], pars->GC_10, pars->ZERO, pars->ZERO, w[39]); 
  FFV1_2(w[3], w[39], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[40]); 
  VVV1P0_1(w[39], w[4], pars->GC_10, pars->ZERO, pars->ZERO, w[41]); 
  FFV1_1(w[2], w[39], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[42]); 
  VVV1P0_1(w[39], w[1], pars->GC_10, pars->ZERO, pars->ZERO, w[43]); 
  FFV1_1(w[29], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[44]); 
  FFV1_2(w[15], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[45]); 
  FFV1_2(w[13], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[46]); 
  VVV1P0_1(w[0], w[10], pars->GC_10, pars->ZERO, pars->ZERO, w[47]); 
  FFV1_2(w[18], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[48]); 
  FFV1_1(w[11], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[49]); 
  FFV1_1(w[14], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[50]); 
  VVV1P0_1(w[0], w[21], pars->GC_10, pars->ZERO, pars->ZERO, w[51]); 
  VVV1P0_1(w[0], w[7], pars->GC_10, pars->ZERO, pars->ZERO, w[52]); 
  VVV1P0_1(w[0], w[23], pars->GC_10, pars->ZERO, pars->ZERO, w[53]); 
  VVVV1P0_1(w[0], w[1], w[4], pars->GC_12, pars->ZERO, pars->ZERO, w[54]); 
  VVVV3P0_1(w[0], w[1], w[4], pars->GC_12, pars->ZERO, pars->ZERO, w[55]); 
  VVVV4P0_1(w[0], w[1], w[4], pars->GC_12, pars->ZERO, pars->ZERO, w[56]); 
  VVVV1P0_1(w[0], w[1], w[5], pars->GC_12, pars->ZERO, pars->ZERO, w[57]); 
  VVVV3P0_1(w[0], w[1], w[5], pars->GC_12, pars->ZERO, pars->ZERO, w[58]); 
  VVVV4P0_1(w[0], w[1], w[5], pars->GC_12, pars->ZERO, pars->ZERO, w[59]); 
  VVVV1P0_1(w[0], w[4], w[5], pars->GC_12, pars->ZERO, pars->ZERO, w[60]); 
  VVVV3P0_1(w[0], w[4], w[5], pars->GC_12, pars->ZERO, pars->ZERO, w[61]); 
  VVVV4P0_1(w[0], w[4], w[5], pars->GC_12, pars->ZERO, pars->ZERO, w[62]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  VVVV1_0(w[6], w[7], w[4], w[5], pars->GC_12, amp[0]); 
  VVVV3_0(w[6], w[7], w[4], w[5], pars->GC_12, amp[1]); 
  VVVV4_0(w[6], w[7], w[4], w[5], pars->GC_12, amp[2]); 
  VVV1_0(w[7], w[5], w[8], pars->GC_10, amp[3]); 
  VVV1_0(w[7], w[4], w[9], pars->GC_10, amp[4]); 
  VVV1_0(w[6], w[7], w[10], pars->GC_10, amp[5]); 
  FFV1_0(w[12], w[11], w[5], pars->GC_11, amp[6]); 
  FFV1_0(w[3], w[11], w[9], pars->GC_11, amp[7]); 
  FFV1_0(w[13], w[11], w[6], pars->GC_11, amp[8]); 
  FFV1_0(w[12], w[14], w[4], pars->GC_11, amp[9]); 
  FFV1_0(w[3], w[14], w[8], pars->GC_11, amp[10]); 
  FFV1_0(w[15], w[14], w[6], pars->GC_11, amp[11]); 
  FFV1_0(w[15], w[16], w[5], pars->GC_11, amp[12]); 
  FFV1_0(w[15], w[2], w[9], pars->GC_11, amp[13]); 
  FFV1_0(w[13], w[16], w[4], pars->GC_11, amp[14]); 
  FFV1_0(w[13], w[2], w[8], pars->GC_11, amp[15]); 
  FFV1_0(w[3], w[16], w[10], pars->GC_11, amp[16]); 
  FFV1_0(w[12], w[2], w[10], pars->GC_11, amp[17]); 
  FFV1_0(w[18], w[19], w[5], pars->GC_11, amp[18]); 
  FFV1_0(w[18], w[20], w[4], pars->GC_11, amp[19]); 
  FFV1_0(w[18], w[17], w[10], pars->GC_11, amp[20]); 
  VVV1_0(w[21], w[5], w[22], pars->GC_10, amp[21]); 
  FFV1_0(w[3], w[20], w[21], pars->GC_11, amp[22]); 
  FFV1_0(w[13], w[17], w[21], pars->GC_11, amp[23]); 
  VVV1_0(w[23], w[4], w[22], pars->GC_10, amp[24]); 
  FFV1_0(w[3], w[19], w[23], pars->GC_11, amp[25]); 
  FFV1_0(w[15], w[17], w[23], pars->GC_11, amp[26]); 
  FFV1_0(w[15], w[24], w[5], pars->GC_11, amp[27]); 
  FFV1_0(w[15], w[20], w[1], pars->GC_11, amp[28]); 
  FFV1_0(w[13], w[24], w[4], pars->GC_11, amp[29]); 
  FFV1_0(w[13], w[19], w[1], pars->GC_11, amp[30]); 
  FFV1_0(w[3], w[24], w[10], pars->GC_11, amp[31]); 
  VVV1_0(w[1], w[10], w[22], pars->GC_10, amp[32]); 
  FFV1_0(w[3], w[17], w[25], pars->GC_11, amp[33]); 
  FFV1_0(w[3], w[17], w[26], pars->GC_11, amp[34]); 
  FFV1_0(w[3], w[17], w[27], pars->GC_11, amp[35]); 
  FFV1_0(w[30], w[29], w[5], pars->GC_11, amp[36]); 
  FFV1_0(w[31], w[29], w[4], pars->GC_11, amp[37]); 
  FFV1_0(w[28], w[29], w[10], pars->GC_11, amp[38]); 
  VVV1_0(w[21], w[5], w[32], pars->GC_10, amp[39]); 
  FFV1_0(w[31], w[2], w[21], pars->GC_11, amp[40]); 
  FFV1_0(w[28], w[14], w[21], pars->GC_11, amp[41]); 
  VVV1_0(w[23], w[4], w[32], pars->GC_10, amp[42]); 
  FFV1_0(w[30], w[2], w[23], pars->GC_11, amp[43]); 
  FFV1_0(w[28], w[11], w[23], pars->GC_11, amp[44]); 
  FFV1_0(w[33], w[11], w[5], pars->GC_11, amp[45]); 
  FFV1_0(w[31], w[11], w[1], pars->GC_11, amp[46]); 
  FFV1_0(w[33], w[14], w[4], pars->GC_11, amp[47]); 
  FFV1_0(w[30], w[14], w[1], pars->GC_11, amp[48]); 
  FFV1_0(w[33], w[2], w[10], pars->GC_11, amp[49]); 
  VVV1_0(w[1], w[10], w[32], pars->GC_10, amp[50]); 
  FFV1_0(w[28], w[2], w[25], pars->GC_11, amp[51]); 
  FFV1_0(w[28], w[2], w[26], pars->GC_11, amp[52]); 
  FFV1_0(w[28], w[2], w[27], pars->GC_11, amp[53]); 
  FFV1_0(w[35], w[29], w[5], pars->GC_11, amp[54]); 
  FFV1_0(w[3], w[29], w[36], pars->GC_11, amp[55]); 
  FFV1_0(w[13], w[29], w[34], pars->GC_11, amp[56]); 
  FFV1_0(w[18], w[37], w[5], pars->GC_11, amp[57]); 
  FFV1_0(w[18], w[2], w[36], pars->GC_11, amp[58]); 
  FFV1_0(w[18], w[14], w[34], pars->GC_11, amp[59]); 
  FFV1_0(w[3], w[37], w[23], pars->GC_11, amp[60]); 
  FFV1_0(w[35], w[2], w[23], pars->GC_11, amp[61]); 
  VVV1_0(w[34], w[23], w[7], pars->GC_10, amp[62]); 
  VVVV1_0(w[34], w[1], w[7], w[5], pars->GC_12, amp[63]); 
  VVVV3_0(w[34], w[1], w[7], w[5], pars->GC_12, amp[64]); 
  VVVV4_0(w[34], w[1], w[7], w[5], pars->GC_12, amp[65]); 
  VVV1_0(w[7], w[5], w[38], pars->GC_10, amp[66]); 
  VVV1_0(w[1], w[7], w[36], pars->GC_10, amp[67]); 
  FFV1_0(w[3], w[14], w[38], pars->GC_11, amp[68]); 
  FFV1_0(w[35], w[14], w[1], pars->GC_11, amp[69]); 
  FFV1_0(w[13], w[2], w[38], pars->GC_11, amp[70]); 
  FFV1_0(w[13], w[37], w[1], pars->GC_11, amp[71]); 
  FFV1_0(w[40], w[29], w[4], pars->GC_11, amp[72]); 
  FFV1_0(w[3], w[29], w[41], pars->GC_11, amp[73]); 
  FFV1_0(w[15], w[29], w[39], pars->GC_11, amp[74]); 
  FFV1_0(w[18], w[42], w[4], pars->GC_11, amp[75]); 
  FFV1_0(w[18], w[2], w[41], pars->GC_11, amp[76]); 
  FFV1_0(w[18], w[11], w[39], pars->GC_11, amp[77]); 
  FFV1_0(w[3], w[42], w[21], pars->GC_11, amp[78]); 
  FFV1_0(w[40], w[2], w[21], pars->GC_11, amp[79]); 
  VVV1_0(w[39], w[21], w[7], pars->GC_10, amp[80]); 
  VVVV1_0(w[39], w[1], w[7], w[4], pars->GC_12, amp[81]); 
  VVVV3_0(w[39], w[1], w[7], w[4], pars->GC_12, amp[82]); 
  VVVV4_0(w[39], w[1], w[7], w[4], pars->GC_12, amp[83]); 
  VVV1_0(w[7], w[4], w[43], pars->GC_10, amp[84]); 
  VVV1_0(w[1], w[7], w[41], pars->GC_10, amp[85]); 
  FFV1_0(w[3], w[11], w[43], pars->GC_11, amp[86]); 
  FFV1_0(w[40], w[11], w[1], pars->GC_11, amp[87]); 
  FFV1_0(w[15], w[2], w[43], pars->GC_11, amp[88]); 
  FFV1_0(w[15], w[42], w[1], pars->GC_11, amp[89]); 
  FFV1_0(w[15], w[44], w[5], pars->GC_11, amp[90]); 
  FFV1_0(w[45], w[29], w[5], pars->GC_11, amp[91]); 
  FFV1_0(w[13], w[44], w[4], pars->GC_11, amp[92]); 
  FFV1_0(w[46], w[29], w[4], pars->GC_11, amp[93]); 
  FFV1_0(w[3], w[44], w[10], pars->GC_11, amp[94]); 
  FFV1_0(w[3], w[29], w[47], pars->GC_11, amp[95]); 
  FFV1_0(w[48], w[11], w[5], pars->GC_11, amp[96]); 
  FFV1_0(w[18], w[49], w[5], pars->GC_11, amp[97]); 
  FFV1_0(w[48], w[14], w[4], pars->GC_11, amp[98]); 
  FFV1_0(w[18], w[50], w[4], pars->GC_11, amp[99]); 
  FFV1_0(w[48], w[2], w[10], pars->GC_11, amp[100]); 
  FFV1_0(w[18], w[2], w[47], pars->GC_11, amp[101]); 
  VVVV1_0(w[0], w[21], w[7], w[5], pars->GC_12, amp[102]); 
  VVVV3_0(w[0], w[21], w[7], w[5], pars->GC_12, amp[103]); 
  VVVV4_0(w[0], w[21], w[7], w[5], pars->GC_12, amp[104]); 
  VVV1_0(w[7], w[5], w[51], pars->GC_10, amp[105]); 
  VVV1_0(w[21], w[5], w[52], pars->GC_10, amp[106]); 
  FFV1_0(w[3], w[14], w[51], pars->GC_11, amp[107]); 
  FFV1_0(w[3], w[50], w[21], pars->GC_11, amp[108]); 
  FFV1_0(w[13], w[2], w[51], pars->GC_11, amp[109]); 
  FFV1_0(w[46], w[2], w[21], pars->GC_11, amp[110]); 
  VVVV1_0(w[0], w[23], w[7], w[4], pars->GC_12, amp[111]); 
  VVVV3_0(w[0], w[23], w[7], w[4], pars->GC_12, amp[112]); 
  VVVV4_0(w[0], w[23], w[7], w[4], pars->GC_12, amp[113]); 
  VVV1_0(w[7], w[4], w[53], pars->GC_10, amp[114]); 
  VVV1_0(w[23], w[4], w[52], pars->GC_10, amp[115]); 
  FFV1_0(w[3], w[11], w[53], pars->GC_11, amp[116]); 
  FFV1_0(w[3], w[49], w[23], pars->GC_11, amp[117]); 
  FFV1_0(w[15], w[2], w[53], pars->GC_11, amp[118]); 
  FFV1_0(w[45], w[2], w[23], pars->GC_11, amp[119]); 
  VVVV1_0(w[0], w[1], w[7], w[10], pars->GC_12, amp[120]); 
  VVVV3_0(w[0], w[1], w[7], w[10], pars->GC_12, amp[121]); 
  VVVV4_0(w[0], w[1], w[7], w[10], pars->GC_12, amp[122]); 
  VVV1_0(w[1], w[10], w[52], pars->GC_10, amp[123]); 
  VVV1_0(w[1], w[7], w[47], pars->GC_10, amp[124]); 
  FFV1_0(w[13], w[49], w[1], pars->GC_11, amp[125]); 
  FFV1_0(w[46], w[11], w[1], pars->GC_11, amp[126]); 
  FFV1_0(w[15], w[50], w[1], pars->GC_11, amp[127]); 
  FFV1_0(w[45], w[14], w[1], pars->GC_11, amp[128]); 
  VVV1_0(w[54], w[7], w[5], pars->GC_10, amp[129]); 
  VVV1_0(w[55], w[7], w[5], pars->GC_10, amp[130]); 
  VVV1_0(w[56], w[7], w[5], pars->GC_10, amp[131]); 
  FFV1_0(w[3], w[14], w[54], pars->GC_11, amp[132]); 
  FFV1_0(w[3], w[14], w[55], pars->GC_11, amp[133]); 
  FFV1_0(w[3], w[14], w[56], pars->GC_11, amp[134]); 
  FFV1_0(w[13], w[2], w[54], pars->GC_11, amp[135]); 
  FFV1_0(w[13], w[2], w[55], pars->GC_11, amp[136]); 
  FFV1_0(w[13], w[2], w[56], pars->GC_11, amp[137]); 
  VVV1_0(w[57], w[7], w[4], pars->GC_10, amp[138]); 
  VVV1_0(w[58], w[7], w[4], pars->GC_10, amp[139]); 
  VVV1_0(w[59], w[7], w[4], pars->GC_10, amp[140]); 
  FFV1_0(w[3], w[11], w[57], pars->GC_11, amp[141]); 
  FFV1_0(w[3], w[11], w[58], pars->GC_11, amp[142]); 
  FFV1_0(w[3], w[11], w[59], pars->GC_11, amp[143]); 
  FFV1_0(w[15], w[2], w[57], pars->GC_11, amp[144]); 
  FFV1_0(w[15], w[2], w[58], pars->GC_11, amp[145]); 
  FFV1_0(w[15], w[2], w[59], pars->GC_11, amp[146]); 
  FFV1_0(w[3], w[29], w[60], pars->GC_11, amp[147]); 
  FFV1_0(w[3], w[29], w[61], pars->GC_11, amp[148]); 
  FFV1_0(w[3], w[29], w[62], pars->GC_11, amp[149]); 
  FFV1_0(w[18], w[2], w[60], pars->GC_11, amp[150]); 
  FFV1_0(w[18], w[2], w[61], pars->GC_11, amp[151]); 
  FFV1_0(w[18], w[2], w[62], pars->GC_11, amp[152]); 
  VVV1_0(w[60], w[1], w[7], pars->GC_10, amp[153]); 
  VVV1_0(w[61], w[1], w[7], pars->GC_10, amp[154]); 
  VVV1_0(w[62], w[1], w[7], pars->GC_10, amp[155]); 
  VVV1_0(w[0], w[25], w[7], pars->GC_10, amp[156]); 
  VVV1_0(w[0], w[26], w[7], pars->GC_10, amp[157]); 
  VVV1_0(w[0], w[27], w[7], pars->GC_10, amp[158]); 


}
double PY8MEs_R6_P0_sm_gg_ttxgg::matrix_6_gg_ttxgg() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 159; 
  const int ncolor = 24; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
      54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54};
  double cf[ncolor][ncolor] = {{512, -64, -64, 8, 8, 80, -64, 8, 8, -1, -1,
      -10, 8, -1, 80, -10, 71, 62, -1, -10, -10, 62, 62, -28}, {-64, 512, 8,
      80, -64, 8, 8, -64, -1, -10, 8, -1, -1, -10, -10, 62, 62, -28, 8, -1, 80,
      -10, 71, 62}, {-64, 8, 512, -64, 80, 8, 8, -1, 80, -10, 71, 62, -64, 8,
      8, -1, -1, -10, -10, -1, 62, -28, -10, 62}, {8, 80, -64, 512, 8, -64, -1,
      -10, -10, 62, 62, -28, 8, -64, -1, -10, 8, -1, -1, 8, 71, 62, 80, -10},
      {8, -64, 80, 8, 512, -64, -1, 8, 71, 62, 80, -10, -10, -1, 62, -28, -10,
      62, -64, 8, 8, -1, -1, -10}, {80, 8, 8, -64, -64, 512, -10, -1, 62, -28,
      -10, 62, -1, 8, 71, 62, 80, -10, 8, -64, -1, -10, 8, -1}, {-64, 8, 8, -1,
      -1, -10, 512, -64, -64, 8, 8, 80, 80, -10, 8, -1, 62, 71, -10, 62, -1,
      -10, -28, 62}, {8, -64, -1, -10, 8, -1, -64, 512, 8, 80, -64, 8, -10, 62,
      -1, -10, -28, 62, 80, -10, 8, -1, 62, 71}, {8, -1, 80, -10, 71, 62, -64,
      8, 512, -64, 80, 8, 8, -1, -64, 8, -10, -1, 62, -28, -10, -1, 62, -10},
      {-1, -10, -10, 62, 62, -28, 8, 80, -64, 512, 8, -64, -1, -10, 8, -64, -1,
      8, 71, 62, -1, 8, -10, 80}, {-1, 8, 71, 62, 80, -10, 8, -64, 80, 8, 512,
      -64, 62, -28, -10, -1, 62, -10, 8, -1, -64, 8, -10, -1}, {-10, -1, 62,
      -28, -10, 62, 80, 8, 8, -64, -64, 512, 71, 62, -1, 8, -10, 80, -1, -10,
      8, -64, -1, 8}, {8, -1, -64, 8, -10, -1, 80, -10, 8, -1, 62, 71, 512,
      -64, -64, 8, 8, 80, 62, -10, -28, 62, -1, -10}, {-1, -10, 8, -64, -1, 8,
      -10, 62, -1, -10, -28, 62, -64, 512, 8, 80, -64, 8, -10, 80, 62, 71, 8,
      -1}, {80, -10, 8, -1, 62, 71, 8, -1, -64, 8, -10, -1, -64, 8, 512, -64,
      80, 8, -28, 62, 62, -10, -10, -1}, {-10, 62, -1, -10, -28, 62, -1, -10,
      8, -64, -1, 8, 8, 80, -64, 512, 8, -64, 62, 71, -10, 80, -1, 8}, {71, 62,
      -1, 8, -10, 80, 62, -28, -10, -1, 62, -10, 8, -64, 80, 8, 512, -64, -1,
      8, -10, -1, -64, 8}, {62, -28, -10, -1, 62, -10, 71, 62, -1, 8, -10, 80,
      80, 8, 8, -64, -64, 512, -10, -1, -1, 8, 8, -64}, {-1, 8, -10, -1, -64,
      8, -10, 80, 62, 71, 8, -1, 62, -10, -28, 62, -1, -10, 512, -64, -64, 8,
      8, 80}, {-10, -1, -1, 8, 8, -64, 62, -10, -28, 62, -1, -10, -10, 80, 62,
      71, 8, -1, -64, 512, 8, 80, -64, 8}, {-10, 80, 62, 71, 8, -1, -1, 8, -10,
      -1, -64, 8, -28, 62, 62, -10, -10, -1, -64, 8, 512, -64, 80, 8}, {62,
      -10, -28, 62, -1, -10, -10, -1, -1, 8, 8, -64, 62, 71, -10, 80, -1, 8, 8,
      80, -64, 512, 8, -64}, {62, 71, -10, 80, -1, 8, -28, 62, 62, -10, -10,
      -1, -1, 8, -10, -1, -64, 8, 8, -64, 80, 8, 512, -64}, {-28, 62, 62, -10,
      -10, -1, 62, 71, -10, 80, -1, 8, -10, -1, -1, 8, 8, -64, 80, 8, 8, -64,
      -64, 512}};

  // Calculate color flows
  jamp[0] = +std::complex<double> (0, 1) * amp[0] + std::complex<double> (0, 1)
      * amp[1] + std::complex<double> (0, 1) * amp[3] + std::complex<double>
      (0, 1) * amp[5] + std::complex<double> (0, 1) * amp[14] + amp[15] +
      amp[16] + amp[21] + std::complex<double> (0, 1) * amp[23] - amp[29] +
      std::complex<double> (0, 1) * amp[31] + amp[32] - amp[35] + amp[33] +
      std::complex<double> (0, 1) * amp[102] + std::complex<double> (0, 1) *
      amp[103] + std::complex<double> (0, 1) * amp[105] + std::complex<double>
      (0, 1) * amp[106] + amp[109] + std::complex<double> (0, 1) * amp[120] +
      std::complex<double> (0, 1) * amp[121] + std::complex<double> (0, 1) *
      amp[123] + std::complex<double> (0, 1) * amp[129] - std::complex<double>
      (0, 1) * amp[131] - amp[137] + amp[135] - std::complex<double> (0, 1) *
      amp[156] + std::complex<double> (0, 1) * amp[158];
  jamp[1] = -std::complex<double> (0, 1) * amp[0] + std::complex<double> (0, 1)
      * amp[2] + std::complex<double> (0, 1) * amp[4] - std::complex<double>
      (0, 1) * amp[5] + std::complex<double> (0, 1) * amp[12] + amp[13] -
      amp[16] + amp[24] + std::complex<double> (0, 1) * amp[26] - amp[27] -
      std::complex<double> (0, 1) * amp[31] - amp[32] - amp[34] - amp[33] +
      std::complex<double> (0, 1) * amp[111] + std::complex<double> (0, 1) *
      amp[112] + std::complex<double> (0, 1) * amp[114] + std::complex<double>
      (0, 1) * amp[115] + amp[118] - std::complex<double> (0, 1) * amp[120] -
      std::complex<double> (0, 1) * amp[121] - std::complex<double> (0, 1) *
      amp[123] + std::complex<double> (0, 1) * amp[138] - std::complex<double>
      (0, 1) * amp[140] - amp[146] + amp[144] + std::complex<double> (0, 1) *
      amp[157] + std::complex<double> (0, 1) * amp[156];
  jamp[2] = -amp[21] - std::complex<double> (0, 1) * amp[23] - amp[24] +
      std::complex<double> (0, 1) * amp[25] - amp[30] + amp[35] + amp[34] +
      amp[60] - std::complex<double> (0, 1) * amp[62] + std::complex<double>
      (0, 1) * amp[63] + std::complex<double> (0, 1) * amp[64] +
      std::complex<double> (0, 1) * amp[66] + amp[70] + std::complex<double>
      (0, 1) * amp[71] - std::complex<double> (0, 1) * amp[102] -
      std::complex<double> (0, 1) * amp[103] - std::complex<double> (0, 1) *
      amp[105] - std::complex<double> (0, 1) * amp[106] - amp[109] -
      std::complex<double> (0, 1) * amp[112] - std::complex<double> (0, 1) *
      amp[113] - std::complex<double> (0, 1) * amp[115] - std::complex<double>
      (0, 1) * amp[130] - std::complex<double> (0, 1) * amp[129] - amp[136] -
      amp[135] - std::complex<double> (0, 1) * amp[157] - std::complex<double>
      (0, 1) * amp[158];
  jamp[3] = -amp[18] + std::complex<double> (0, 1) * amp[20] + amp[24] -
      std::complex<double> (0, 1) * amp[25] - amp[32] - amp[34] - amp[33] +
      std::complex<double> (0, 1) * amp[57] + amp[58] - amp[60] +
      std::complex<double> (0, 1) * amp[62] - std::complex<double> (0, 1) *
      amp[64] - std::complex<double> (0, 1) * amp[65] - std::complex<double>
      (0, 1) * amp[67] + amp[101] + std::complex<double> (0, 1) * amp[112] +
      std::complex<double> (0, 1) * amp[113] + std::complex<double> (0, 1) *
      amp[115] - std::complex<double> (0, 1) * amp[121] - std::complex<double>
      (0, 1) * amp[122] - std::complex<double> (0, 1) * amp[123] -
      std::complex<double> (0, 1) * amp[124] - amp[152] + amp[150] -
      std::complex<double> (0, 1) * amp[153] + std::complex<double> (0, 1) *
      amp[155] + std::complex<double> (0, 1) * amp[157] + std::complex<double>
      (0, 1) * amp[156];
  jamp[4] = -amp[21] + std::complex<double> (0, 1) * amp[22] - amp[24] -
      std::complex<double> (0, 1) * amp[26] - amp[28] + amp[35] + amp[34] +
      amp[78] - std::complex<double> (0, 1) * amp[80] + std::complex<double>
      (0, 1) * amp[81] + std::complex<double> (0, 1) * amp[82] +
      std::complex<double> (0, 1) * amp[84] + amp[88] + std::complex<double>
      (0, 1) * amp[89] - std::complex<double> (0, 1) * amp[103] -
      std::complex<double> (0, 1) * amp[104] - std::complex<double> (0, 1) *
      amp[106] - std::complex<double> (0, 1) * amp[111] - std::complex<double>
      (0, 1) * amp[112] - std::complex<double> (0, 1) * amp[114] -
      std::complex<double> (0, 1) * amp[115] - amp[118] - std::complex<double>
      (0, 1) * amp[139] - std::complex<double> (0, 1) * amp[138] - amp[145] -
      amp[144] - std::complex<double> (0, 1) * amp[157] - std::complex<double>
      (0, 1) * amp[158];
  jamp[5] = -amp[19] - std::complex<double> (0, 1) * amp[20] + amp[21] -
      std::complex<double> (0, 1) * amp[22] + amp[32] - amp[35] + amp[33] +
      std::complex<double> (0, 1) * amp[75] + amp[76] - amp[78] +
      std::complex<double> (0, 1) * amp[80] - std::complex<double> (0, 1) *
      amp[82] - std::complex<double> (0, 1) * amp[83] - std::complex<double>
      (0, 1) * amp[85] - amp[101] + std::complex<double> (0, 1) * amp[103] +
      std::complex<double> (0, 1) * amp[104] + std::complex<double> (0, 1) *
      amp[106] + std::complex<double> (0, 1) * amp[121] + std::complex<double>
      (0, 1) * amp[122] + std::complex<double> (0, 1) * amp[123] +
      std::complex<double> (0, 1) * amp[124] - amp[151] - amp[150] +
      std::complex<double> (0, 1) * amp[154] + std::complex<double> (0, 1) *
      amp[153] - std::complex<double> (0, 1) * amp[156] + std::complex<double>
      (0, 1) * amp[158];
  jamp[6] = -std::complex<double> (0, 1) * amp[0] - std::complex<double> (0, 1)
      * amp[1] - std::complex<double> (0, 1) * amp[3] - std::complex<double>
      (0, 1) * amp[5] - std::complex<double> (0, 1) * amp[14] - amp[15] -
      amp[16] + amp[55] + std::complex<double> (0, 1) * amp[56] -
      std::complex<double> (0, 1) * amp[63] + std::complex<double> (0, 1) *
      amp[65] - std::complex<double> (0, 1) * amp[66] + std::complex<double>
      (0, 1) * amp[67] - amp[70] - amp[92] + std::complex<double> (0, 1) *
      amp[94] + amp[95] - std::complex<double> (0, 1) * amp[120] +
      std::complex<double> (0, 1) * amp[122] + std::complex<double> (0, 1) *
      amp[124] + std::complex<double> (0, 1) * amp[130] + std::complex<double>
      (0, 1) * amp[131] + amp[137] + amp[136] - amp[149] + amp[147] +
      std::complex<double> (0, 1) * amp[153] - std::complex<double> (0, 1) *
      amp[155];
  jamp[7] = +std::complex<double> (0, 1) * amp[0] - std::complex<double> (0, 1)
      * amp[2] - std::complex<double> (0, 1) * amp[4] + std::complex<double>
      (0, 1) * amp[5] - std::complex<double> (0, 1) * amp[12] - amp[13] +
      amp[16] + amp[73] + std::complex<double> (0, 1) * amp[74] -
      std::complex<double> (0, 1) * amp[81] + std::complex<double> (0, 1) *
      amp[83] - std::complex<double> (0, 1) * amp[84] + std::complex<double>
      (0, 1) * amp[85] - amp[88] - amp[90] - std::complex<double> (0, 1) *
      amp[94] - amp[95] + std::complex<double> (0, 1) * amp[120] -
      std::complex<double> (0, 1) * amp[122] - std::complex<double> (0, 1) *
      amp[124] + std::complex<double> (0, 1) * amp[139] + std::complex<double>
      (0, 1) * amp[140] + amp[146] + amp[145] - amp[148] - amp[147] -
      std::complex<double> (0, 1) * amp[154] - std::complex<double> (0, 1) *
      amp[153];
  jamp[8] = -amp[55] - std::complex<double> (0, 1) * amp[56] +
      std::complex<double> (0, 1) * amp[63] - std::complex<double> (0, 1) *
      amp[65] + std::complex<double> (0, 1) * amp[66] - std::complex<double>
      (0, 1) * amp[67] + amp[70] + std::complex<double> (0, 1) * amp[72] -
      amp[73] + amp[79] + std::complex<double> (0, 1) * amp[80] -
      std::complex<double> (0, 1) * amp[82] - std::complex<double> (0, 1) *
      amp[83] - std::complex<double> (0, 1) * amp[85] - amp[93] -
      std::complex<double> (0, 1) * amp[102] + std::complex<double> (0, 1) *
      amp[104] - std::complex<double> (0, 1) * amp[105] - amp[109] +
      std::complex<double> (0, 1) * amp[110] - std::complex<double> (0, 1) *
      amp[130] - std::complex<double> (0, 1) * amp[129] - amp[136] - amp[135] +
      amp[149] + amp[148] + std::complex<double> (0, 1) * amp[154] +
      std::complex<double> (0, 1) * amp[155];
  jamp[9] = -amp[37] + std::complex<double> (0, 1) * amp[38] + amp[39] +
      std::complex<double> (0, 1) * amp[40] + amp[50] - amp[53] + amp[51] -
      std::complex<double> (0, 1) * amp[72] + amp[73] - amp[79] -
      std::complex<double> (0, 1) * amp[80] + std::complex<double> (0, 1) *
      amp[82] + std::complex<double> (0, 1) * amp[83] + std::complex<double>
      (0, 1) * amp[85] - amp[95] - std::complex<double> (0, 1) * amp[103] -
      std::complex<double> (0, 1) * amp[104] - std::complex<double> (0, 1) *
      amp[106] - std::complex<double> (0, 1) * amp[121] - std::complex<double>
      (0, 1) * amp[122] - std::complex<double> (0, 1) * amp[123] -
      std::complex<double> (0, 1) * amp[124] - amp[148] - amp[147] -
      std::complex<double> (0, 1) * amp[154] - std::complex<double> (0, 1) *
      amp[153] + std::complex<double> (0, 1) * amp[156] - std::complex<double>
      (0, 1) * amp[158];
  jamp[10] = +std::complex<double> (0, 1) * amp[54] - amp[55] + amp[61] +
      std::complex<double> (0, 1) * amp[62] - std::complex<double> (0, 1) *
      amp[64] - std::complex<double> (0, 1) * amp[65] - std::complex<double>
      (0, 1) * amp[67] - amp[73] - std::complex<double> (0, 1) * amp[74] +
      std::complex<double> (0, 1) * amp[81] - std::complex<double> (0, 1) *
      amp[83] + std::complex<double> (0, 1) * amp[84] - std::complex<double>
      (0, 1) * amp[85] + amp[88] - amp[91] - std::complex<double> (0, 1) *
      amp[111] + std::complex<double> (0, 1) * amp[113] - std::complex<double>
      (0, 1) * amp[114] - amp[118] + std::complex<double> (0, 1) * amp[119] -
      std::complex<double> (0, 1) * amp[139] - std::complex<double> (0, 1) *
      amp[138] - amp[145] - amp[144] + amp[149] + amp[148] +
      std::complex<double> (0, 1) * amp[154] + std::complex<double> (0, 1) *
      amp[155];
  jamp[11] = -amp[36] - std::complex<double> (0, 1) * amp[38] + amp[42] +
      std::complex<double> (0, 1) * amp[43] - amp[50] - amp[52] - amp[51] -
      std::complex<double> (0, 1) * amp[54] + amp[55] - amp[61] -
      std::complex<double> (0, 1) * amp[62] + std::complex<double> (0, 1) *
      amp[64] + std::complex<double> (0, 1) * amp[65] + std::complex<double>
      (0, 1) * amp[67] + amp[95] - std::complex<double> (0, 1) * amp[112] -
      std::complex<double> (0, 1) * amp[113] - std::complex<double> (0, 1) *
      amp[115] + std::complex<double> (0, 1) * amp[121] + std::complex<double>
      (0, 1) * amp[122] + std::complex<double> (0, 1) * amp[123] +
      std::complex<double> (0, 1) * amp[124] - amp[149] + amp[147] +
      std::complex<double> (0, 1) * amp[153] - std::complex<double> (0, 1) *
      amp[155] - std::complex<double> (0, 1) * amp[157] - std::complex<double>
      (0, 1) * amp[156];
  jamp[12] = -std::complex<double> (0, 1) * amp[1] - std::complex<double> (0,
      1) * amp[2] - std::complex<double> (0, 1) * amp[3] - std::complex<double>
      (0, 1) * amp[4] + amp[7] + std::complex<double> (0, 1) * amp[8] - amp[15]
      - amp[60] + std::complex<double> (0, 1) * amp[62] - std::complex<double>
      (0, 1) * amp[63] - std::complex<double> (0, 1) * amp[64] -
      std::complex<double> (0, 1) * amp[66] - amp[70] - std::complex<double>
      (0, 1) * amp[71] - std::complex<double> (0, 1) * amp[111] +
      std::complex<double> (0, 1) * amp[113] - std::complex<double> (0, 1) *
      amp[114] + amp[116] + std::complex<double> (0, 1) * amp[117] - amp[125] +
      std::complex<double> (0, 1) * amp[130] + std::complex<double> (0, 1) *
      amp[131] + amp[137] + amp[136] - std::complex<double> (0, 1) * amp[138] +
      std::complex<double> (0, 1) * amp[140] - amp[143] + amp[141];
  jamp[13] = -std::complex<double> (0, 1) * amp[57] - amp[58] + amp[60] -
      std::complex<double> (0, 1) * amp[62] + std::complex<double> (0, 1) *
      amp[64] + std::complex<double> (0, 1) * amp[65] + std::complex<double>
      (0, 1) * amp[67] - amp[76] + std::complex<double> (0, 1) * amp[77] -
      std::complex<double> (0, 1) * amp[81] + std::complex<double> (0, 1) *
      amp[83] - std::complex<double> (0, 1) * amp[84] + std::complex<double>
      (0, 1) * amp[85] + amp[86] - amp[97] + std::complex<double> (0, 1) *
      amp[111] - std::complex<double> (0, 1) * amp[113] + std::complex<double>
      (0, 1) * amp[114] - amp[116] - std::complex<double> (0, 1) * amp[117] +
      std::complex<double> (0, 1) * amp[139] + std::complex<double> (0, 1) *
      amp[138] - amp[142] - amp[141] + amp[152] + amp[151] -
      std::complex<double> (0, 1) * amp[154] - std::complex<double> (0, 1) *
      amp[155];
  jamp[14] = +std::complex<double> (0, 1) * amp[1] + std::complex<double> (0,
      1) * amp[2] + std::complex<double> (0, 1) * amp[3] + std::complex<double>
      (0, 1) * amp[4] - amp[7] - std::complex<double> (0, 1) * amp[8] + amp[15]
      - amp[79] - std::complex<double> (0, 1) * amp[80] + std::complex<double>
      (0, 1) * amp[81] + std::complex<double> (0, 1) * amp[82] +
      std::complex<double> (0, 1) * amp[84] - amp[86] + std::complex<double>
      (0, 1) * amp[87] + std::complex<double> (0, 1) * amp[102] -
      std::complex<double> (0, 1) * amp[104] + std::complex<double> (0, 1) *
      amp[105] + amp[109] - std::complex<double> (0, 1) * amp[110] - amp[126] +
      std::complex<double> (0, 1) * amp[129] - std::complex<double> (0, 1) *
      amp[131] - amp[137] + amp[135] - std::complex<double> (0, 1) * amp[139] -
      std::complex<double> (0, 1) * amp[140] + amp[143] + amp[142];
  jamp[15] = -amp[39] - std::complex<double> (0, 1) * amp[40] - amp[42] +
      std::complex<double> (0, 1) * amp[44] - amp[46] + amp[53] + amp[52] +
      amp[79] + std::complex<double> (0, 1) * amp[80] - std::complex<double>
      (0, 1) * amp[81] - std::complex<double> (0, 1) * amp[82] -
      std::complex<double> (0, 1) * amp[84] + amp[86] - std::complex<double>
      (0, 1) * amp[87] + std::complex<double> (0, 1) * amp[103] +
      std::complex<double> (0, 1) * amp[104] + std::complex<double> (0, 1) *
      amp[106] + std::complex<double> (0, 1) * amp[111] + std::complex<double>
      (0, 1) * amp[112] + std::complex<double> (0, 1) * amp[114] +
      std::complex<double> (0, 1) * amp[115] - amp[116] + std::complex<double>
      (0, 1) * amp[139] + std::complex<double> (0, 1) * amp[138] - amp[142] -
      amp[141] + std::complex<double> (0, 1) * amp[157] + std::complex<double>
      (0, 1) * amp[158];
  jamp[16] = -std::complex<double> (0, 1) * amp[0] + std::complex<double> (0,
      1) * amp[2] + std::complex<double> (0, 1) * amp[4] - std::complex<double>
      (0, 1) * amp[5] + std::complex<double> (0, 1) * amp[6] - amp[7] + amp[17]
      + amp[76] - std::complex<double> (0, 1) * amp[77] + std::complex<double>
      (0, 1) * amp[81] - std::complex<double> (0, 1) * amp[83] +
      std::complex<double> (0, 1) * amp[84] - std::complex<double> (0, 1) *
      amp[85] - amp[86] - amp[96] + std::complex<double> (0, 1) * amp[100] -
      amp[101] - std::complex<double> (0, 1) * amp[120] + std::complex<double>
      (0, 1) * amp[122] + std::complex<double> (0, 1) * amp[124] -
      std::complex<double> (0, 1) * amp[139] - std::complex<double> (0, 1) *
      amp[140] + amp[143] + amp[142] - amp[151] - amp[150] +
      std::complex<double> (0, 1) * amp[154] + std::complex<double> (0, 1) *
      amp[153];
  jamp[17] = +std::complex<double> (0, 1) * amp[0] - std::complex<double> (0,
      1) * amp[2] - std::complex<double> (0, 1) * amp[4] + std::complex<double>
      (0, 1) * amp[5] - std::complex<double> (0, 1) * amp[6] + amp[7] - amp[17]
      + amp[42] - std::complex<double> (0, 1) * amp[44] - amp[45] +
      std::complex<double> (0, 1) * amp[49] - amp[50] - amp[52] - amp[51] -
      std::complex<double> (0, 1) * amp[111] - std::complex<double> (0, 1) *
      amp[112] - std::complex<double> (0, 1) * amp[114] - std::complex<double>
      (0, 1) * amp[115] + amp[116] + std::complex<double> (0, 1) * amp[120] +
      std::complex<double> (0, 1) * amp[121] + std::complex<double> (0, 1) *
      amp[123] - std::complex<double> (0, 1) * amp[138] + std::complex<double>
      (0, 1) * amp[140] - amp[143] + amp[141] - std::complex<double> (0, 1) *
      amp[157] - std::complex<double> (0, 1) * amp[156];
  jamp[18] = -std::complex<double> (0, 1) * amp[1] - std::complex<double> (0,
      1) * amp[2] - std::complex<double> (0, 1) * amp[3] - std::complex<double>
      (0, 1) * amp[4] + amp[10] + std::complex<double> (0, 1) * amp[11] -
      amp[13] - amp[78] + std::complex<double> (0, 1) * amp[80] -
      std::complex<double> (0, 1) * amp[81] - std::complex<double> (0, 1) *
      amp[82] - std::complex<double> (0, 1) * amp[84] - amp[88] -
      std::complex<double> (0, 1) * amp[89] - std::complex<double> (0, 1) *
      amp[102] + std::complex<double> (0, 1) * amp[104] - std::complex<double>
      (0, 1) * amp[105] + amp[107] + std::complex<double> (0, 1) * amp[108] -
      amp[127] - std::complex<double> (0, 1) * amp[129] + std::complex<double>
      (0, 1) * amp[131] - amp[134] + amp[132] + std::complex<double> (0, 1) *
      amp[139] + std::complex<double> (0, 1) * amp[140] + amp[146] + amp[145];
  jamp[19] = -amp[58] + std::complex<double> (0, 1) * amp[59] -
      std::complex<double> (0, 1) * amp[63] + std::complex<double> (0, 1) *
      amp[65] - std::complex<double> (0, 1) * amp[66] + std::complex<double>
      (0, 1) * amp[67] + amp[68] - std::complex<double> (0, 1) * amp[75] -
      amp[76] + amp[78] - std::complex<double> (0, 1) * amp[80] +
      std::complex<double> (0, 1) * amp[82] + std::complex<double> (0, 1) *
      amp[83] + std::complex<double> (0, 1) * amp[85] - amp[99] +
      std::complex<double> (0, 1) * amp[102] - std::complex<double> (0, 1) *
      amp[104] + std::complex<double> (0, 1) * amp[105] - amp[107] -
      std::complex<double> (0, 1) * amp[108] + std::complex<double> (0, 1) *
      amp[130] + std::complex<double> (0, 1) * amp[129] - amp[133] - amp[132] +
      amp[152] + amp[151] - std::complex<double> (0, 1) * amp[154] -
      std::complex<double> (0, 1) * amp[155];
  jamp[20] = +std::complex<double> (0, 1) * amp[1] + std::complex<double> (0,
      1) * amp[2] + std::complex<double> (0, 1) * amp[3] + std::complex<double>
      (0, 1) * amp[4] - amp[10] - std::complex<double> (0, 1) * amp[11] +
      amp[13] - amp[61] - std::complex<double> (0, 1) * amp[62] +
      std::complex<double> (0, 1) * amp[63] + std::complex<double> (0, 1) *
      amp[64] + std::complex<double> (0, 1) * amp[66] - amp[68] +
      std::complex<double> (0, 1) * amp[69] + std::complex<double> (0, 1) *
      amp[111] - std::complex<double> (0, 1) * amp[113] + std::complex<double>
      (0, 1) * amp[114] + amp[118] - std::complex<double> (0, 1) * amp[119] -
      amp[128] - std::complex<double> (0, 1) * amp[130] - std::complex<double>
      (0, 1) * amp[131] + amp[134] + amp[133] + std::complex<double> (0, 1) *
      amp[138] - std::complex<double> (0, 1) * amp[140] - amp[146] + amp[144];
  jamp[21] = -amp[39] + std::complex<double> (0, 1) * amp[41] - amp[42] -
      std::complex<double> (0, 1) * amp[43] - amp[48] + amp[53] + amp[52] +
      amp[61] + std::complex<double> (0, 1) * amp[62] - std::complex<double>
      (0, 1) * amp[63] - std::complex<double> (0, 1) * amp[64] -
      std::complex<double> (0, 1) * amp[66] + amp[68] - std::complex<double>
      (0, 1) * amp[69] + std::complex<double> (0, 1) * amp[102] +
      std::complex<double> (0, 1) * amp[103] + std::complex<double> (0, 1) *
      amp[105] + std::complex<double> (0, 1) * amp[106] - amp[107] +
      std::complex<double> (0, 1) * amp[112] + std::complex<double> (0, 1) *
      amp[113] + std::complex<double> (0, 1) * amp[115] + std::complex<double>
      (0, 1) * amp[130] + std::complex<double> (0, 1) * amp[129] - amp[133] -
      amp[132] + std::complex<double> (0, 1) * amp[157] + std::complex<double>
      (0, 1) * amp[158];
  jamp[22] = +std::complex<double> (0, 1) * amp[0] + std::complex<double> (0,
      1) * amp[1] + std::complex<double> (0, 1) * amp[3] + std::complex<double>
      (0, 1) * amp[5] + std::complex<double> (0, 1) * amp[9] - amp[10] -
      amp[17] + amp[58] - std::complex<double> (0, 1) * amp[59] +
      std::complex<double> (0, 1) * amp[63] - std::complex<double> (0, 1) *
      amp[65] + std::complex<double> (0, 1) * amp[66] - std::complex<double>
      (0, 1) * amp[67] - amp[68] - amp[98] - std::complex<double> (0, 1) *
      amp[100] + amp[101] + std::complex<double> (0, 1) * amp[120] -
      std::complex<double> (0, 1) * amp[122] - std::complex<double> (0, 1) *
      amp[124] - std::complex<double> (0, 1) * amp[130] - std::complex<double>
      (0, 1) * amp[131] + amp[134] + amp[133] - amp[152] + amp[150] -
      std::complex<double> (0, 1) * amp[153] + std::complex<double> (0, 1) *
      amp[155];
  jamp[23] = -std::complex<double> (0, 1) * amp[0] - std::complex<double> (0,
      1) * amp[1] - std::complex<double> (0, 1) * amp[3] - std::complex<double>
      (0, 1) * amp[5] - std::complex<double> (0, 1) * amp[9] + amp[10] +
      amp[17] + amp[39] - std::complex<double> (0, 1) * amp[41] - amp[47] -
      std::complex<double> (0, 1) * amp[49] + amp[50] - amp[53] + amp[51] -
      std::complex<double> (0, 1) * amp[102] - std::complex<double> (0, 1) *
      amp[103] - std::complex<double> (0, 1) * amp[105] - std::complex<double>
      (0, 1) * amp[106] + amp[107] - std::complex<double> (0, 1) * amp[120] -
      std::complex<double> (0, 1) * amp[121] - std::complex<double> (0, 1) *
      amp[123] - std::complex<double> (0, 1) * amp[129] + std::complex<double>
      (0, 1) * amp[131] - amp[134] + amp[132] + std::complex<double> (0, 1) *
      amp[156] - std::complex<double> (0, 1) * amp[158];

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


}  // end namespace PY8MEs_namespace

