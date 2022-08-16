//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "PY8MEs_R130_P0_sm_gg_ggg.h"
#include "HelAmps_sm.h"

using namespace Pythia8_sm; 

namespace PY8MEs_namespace 
{
//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: g g > g g g WEIGHTED<=3 @130

// Exception class
class PY8MEs_R130_P0_sm_gg_gggException : public exception
{
  virtual const char * what() const throw()
  {
    return "Exception in class 'PY8MEs_R130_P0_sm_gg_ggg'."; 
  }
}
PY8MEs_R130_P0_sm_gg_ggg_exception; 

std::set<int> PY8MEs_R130_P0_sm_gg_ggg::s_channel_proc = std::set<int>
    (createset<int> ());

int PY8MEs_R130_P0_sm_gg_ggg::helicities[ncomb][nexternal] = {{-1, -1, -1, -1,
    -1}, {-1, -1, -1, -1, 1}, {-1, -1, -1, 1, -1}, {-1, -1, -1, 1, 1}, {-1, -1,
    1, -1, -1}, {-1, -1, 1, -1, 1}, {-1, -1, 1, 1, -1}, {-1, -1, 1, 1, 1}, {-1,
    1, -1, -1, -1}, {-1, 1, -1, -1, 1}, {-1, 1, -1, 1, -1}, {-1, 1, -1, 1, 1},
    {-1, 1, 1, -1, -1}, {-1, 1, 1, -1, 1}, {-1, 1, 1, 1, -1}, {-1, 1, 1, 1, 1},
    {1, -1, -1, -1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, 1, -1}, {1, -1, -1, 1,
    1}, {1, -1, 1, -1, -1}, {1, -1, 1, -1, 1}, {1, -1, 1, 1, -1}, {1, -1, 1, 1,
    1}, {1, 1, -1, -1, -1}, {1, 1, -1, -1, 1}, {1, 1, -1, 1, -1}, {1, 1, -1, 1,
    1}, {1, 1, 1, -1, -1}, {1, 1, 1, -1, 1}, {1, 1, 1, 1, -1}, {1, 1, 1, 1, 1}};

// Normalization factors the various processes
// Denominators: spins, colors and identical particles
int PY8MEs_R130_P0_sm_gg_ggg::denom_colors[nprocesses] = {64}; 
int PY8MEs_R130_P0_sm_gg_ggg::denom_hels[nprocesses] = {4}; 
int PY8MEs_R130_P0_sm_gg_ggg::denom_iden[nprocesses] = {6}; 

//--------------------------------------------------------------------------
// Color config initialization
void PY8MEs_R130_P0_sm_gg_ggg::initColorConfigs() 
{
  color_configs = vector < vec_vec_int > (); 
  jamp_nc_relative_power = vector < vec_int > (); 

  // Color flows of process Process: g g > g g g WEIGHTED<=3 @130
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[0].push_back(vec_int(createvector<int>
      (5)(1)(1)(2)(3)(2)(4)(3)(5)(4)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #1
  color_configs[0].push_back(vec_int(createvector<int>
      (4)(1)(1)(2)(3)(2)(4)(5)(5)(3)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #2
  color_configs[0].push_back(vec_int(createvector<int>
      (5)(1)(1)(2)(3)(4)(4)(2)(5)(3)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #3
  color_configs[0].push_back(vec_int(createvector<int>
      (3)(1)(1)(2)(3)(5)(4)(2)(5)(4)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #4
  color_configs[0].push_back(vec_int(createvector<int>
      (4)(1)(1)(2)(3)(5)(4)(3)(5)(2)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #5
  color_configs[0].push_back(vec_int(createvector<int>
      (3)(1)(1)(2)(3)(4)(4)(5)(5)(2)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #6
  color_configs[0].push_back(vec_int(createvector<int>
      (5)(1)(3)(2)(3)(1)(4)(2)(5)(4)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #7
  color_configs[0].push_back(vec_int(createvector<int>
      (4)(1)(3)(2)(3)(1)(4)(5)(5)(2)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #8
  color_configs[0].push_back(vec_int(createvector<int>
      (5)(1)(4)(2)(3)(1)(4)(3)(5)(2)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #9
  color_configs[0].push_back(vec_int(createvector<int>
      (2)(1)(5)(2)(3)(1)(4)(3)(5)(4)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #10
  color_configs[0].push_back(vec_int(createvector<int>
      (4)(1)(5)(2)(3)(1)(4)(2)(5)(3)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #11
  color_configs[0].push_back(vec_int(createvector<int>
      (2)(1)(4)(2)(3)(1)(4)(5)(5)(3)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #12
  color_configs[0].push_back(vec_int(createvector<int>
      (5)(1)(4)(2)(3)(2)(4)(1)(5)(3)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #13
  color_configs[0].push_back(vec_int(createvector<int>
      (3)(1)(4)(2)(3)(5)(4)(1)(5)(2)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #14
  color_configs[0].push_back(vec_int(createvector<int>
      (5)(1)(3)(2)(3)(4)(4)(1)(5)(2)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #15
  color_configs[0].push_back(vec_int(createvector<int>
      (2)(1)(5)(2)(3)(4)(4)(1)(5)(3)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #16
  color_configs[0].push_back(vec_int(createvector<int>
      (3)(1)(5)(2)(3)(2)(4)(1)(5)(4)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #17
  color_configs[0].push_back(vec_int(createvector<int>
      (2)(1)(3)(2)(3)(5)(4)(1)(5)(4)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #18
  color_configs[0].push_back(vec_int(createvector<int>
      (4)(1)(5)(2)(3)(2)(4)(3)(5)(1)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #19
  color_configs[0].push_back(vec_int(createvector<int>
      (3)(1)(5)(2)(3)(4)(4)(2)(5)(1)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #20
  color_configs[0].push_back(vec_int(createvector<int>
      (4)(1)(3)(2)(3)(5)(4)(2)(5)(1)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #21
  color_configs[0].push_back(vec_int(createvector<int>
      (2)(1)(4)(2)(3)(5)(4)(3)(5)(1)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #22
  color_configs[0].push_back(vec_int(createvector<int>
      (3)(1)(4)(2)(3)(2)(4)(5)(5)(1)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #23
  color_configs[0].push_back(vec_int(createvector<int>
      (2)(1)(3)(2)(3)(4)(4)(5)(5)(1)));
  jamp_nc_relative_power[0].push_back(0); 
}

//--------------------------------------------------------------------------
// Destructor.
PY8MEs_R130_P0_sm_gg_ggg::~PY8MEs_R130_P0_sm_gg_ggg() 
{
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    delete[] p[i]; 
    p[i] = NULL; 
  }
}

//--------------------------------------------------------------------------
// Invert the permutation mapping
vector<int> PY8MEs_R130_P0_sm_gg_ggg::invert_mapping(vector<int> mapping) 
{
  vector<int> inverted_mapping; 
  for (unsigned int i = 0; i < mapping.size(); i++ )
  {
    for (unsigned int j = 0; j < mapping.size(); j++ )
    {
      if (mapping[j] == ((int)i))
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
vector < vec_int > PY8MEs_R130_P0_sm_gg_ggg::getHelicityConfigs(vector<int>
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
  for (unsigned int ihel = 0; ihel < ncomb; ihel++ )
  {
    for(unsigned int j = 0; j < nexternal; j++ )
    {
      res[ihel][chosenPerm[j]] = helicities[ihel][j]; 
    }
  }
  return res; 
}

//--------------------------------------------------------------------------
// Return the list of possible color configurations
vector < vec_int > PY8MEs_R130_P0_sm_gg_ggg::getColorConfigs(int
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
  for (unsigned int icol = 0; icol < color_configs[chosenProcID].size(); icol++
      )
  {
    for(unsigned int j = 0; j < (2 * nexternal); j++ )
    {
      res[icol][chosenPerm[j/2] * 2 + j%2] =
          color_configs[chosenProcID][icol][j];
    }
  }
  return res; 
}

//--------------------------------------------------------------------------
// Get JAMP relative N_c power
int PY8MEs_R130_P0_sm_gg_ggg::getColorFlowRelativeNCPower(int color_flow_ID,
    int specify_proc_ID)
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
vector<int> PY8MEs_R130_P0_sm_gg_ggg::getHelicityConfigForID(int hel_ID,
    vector<int> permutation)
{
  if (hel_ID < 0 || hel_ID >= ncomb)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'getHelicityConfigForID' of class" << 
    " 'PY8MEs_R130_P0_sm_gg_ggg': Specified helicity ID '" << 
    hel_ID <<  "' cannot be found." << endl; 
    #endif
    throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
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
  for (unsigned int j = 0; j < nexternal; j++ )
  {
    res[chosenPerm[j]] = helicities[hel_ID][j]; 
  }
  return res; 
}

//--------------------------------------------------------------------------
// Implements the map Helicity Config -> Helicity ID
int PY8MEs_R130_P0_sm_gg_ggg::getHelicityIDForConfig(vector<int> hel_config,
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
    for(unsigned int i = 0; i < ncomb; i++ )
    {
      found = true; 
      for (unsigned int j = 0; j < nexternal; j++ )
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
      #ifdef DEBUG
      cerr <<  "Error in function 'getHelicityIDForConfig' of class" << 
      " 'PY8MEs_R130_P0_sm_gg_ggg': Specified helicity" << 
      " configuration cannot be found." << endl; 
      #endif
      throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
    }
  }
  return user_ihel; 
}


//--------------------------------------------------------------------------
// Implements the map Color ID -> Color Config
vector<int> PY8MEs_R130_P0_sm_gg_ggg::getColorConfigForID(int color_ID, int
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
    #ifdef DEBUG
    cerr <<  "Error in function 'getColorConfigForID' of class" << 
    " 'PY8MEs_R130_P0_sm_gg_ggg': Specified color ID '" << 
    color_ID <<  "' cannot be found." << endl; 
    #endif
    throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
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
  for (unsigned int j = 0; j < (2 * nexternal); j++ )
  {
    res[chosenPerm[j/2] * 2 + j%2] = color_configs[chosenProcID][color_ID][j]; 
  }
  return res; 
}

//--------------------------------------------------------------------------
// Implements the map Color Config -> Color ID
int PY8MEs_R130_P0_sm_gg_ggg::getColorIDForConfig(vector<int> color_config, int
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
    for(unsigned int i = 0; i < color_configs[chosenProcID].size(); i++ )
    {
      found = true; 
      for (unsigned int j = 0; j < (nexternal * 2); j++ )
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
          for (unsigned int k = 0; k < (nexternal * 2); k++ )
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
            #ifdef DEBUG
            cerr <<  "Error in function 'getColorIDForConfig' of class" << 
            " 'PY8MEs_R130_P0_sm_gg_ggg': A color line could " << 
            " not be closed." << endl; 
            #endif
            throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
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
      #ifdef DEBUG
      cerr <<  "Error in function 'getColorIDForConfig' of class" << 
      " 'PY8MEs_R130_P0_sm_gg_ggg': Specified color" << 
      " configuration cannot be found." << endl; 
      #endif
      throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
    }
  }
  return user_icol; 
}

//--------------------------------------------------------------------------
// Returns all result previously computed in SigmaKin
vector < vec_double > PY8MEs_R130_P0_sm_gg_ggg::getAllResults(int
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
double PY8MEs_R130_P0_sm_gg_ggg::getResult(int helicity_ID, int color_ID, int
    specify_proc_ID)
{
  if (helicity_ID < - 1 || helicity_ID >= ncomb)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R130_P0_sm_gg_ggg': Specified helicity ID '" << 
    helicity_ID <<  "' configuration cannot be found." << endl; 
    #endif
    throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
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
    #ifdef DEBUG
    cerr <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R130_P0_sm_gg_ggg': Specified color ID '" << 
    color_ID <<  "' configuration cannot be found." << endl; 
    #endif
    throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
  }
  return all_results[chosenProcID][helicity_ID + 1][color_ID + 1]; 
}

//--------------------------------------------------------------------------
// Check for the availability of the requested process and if available,
// If available, this returns the corresponding permutation and Proc_ID to use.
// If not available, this returns a negative Proc_ID.
pair < vector<int> , int > PY8MEs_R130_P0_sm_gg_ggg::static_getPY8ME(vector<int> initial_pdgs, vector<int> final_pdgs, set<int> schannels) 
{

  // Not available return value
  pair < vector<int> , int > NA(vector<int> (), -1); 

  // Check if s-channel requirements match
  if (nreq_s_channels > 0)
  {
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
  const int out_pdgs[nprocs][nexternal - ninitial] = {{21, 21, 21}}; 

  bool in_pdgs_used[ninitial]; 
  bool out_pdgs_used[nexternal - ninitial]; 
  for(unsigned int i = 0; i < nprocs; i++ )
  {
    int permutations[nexternal]; 

    // Reinitialize initial state look-up variables
    for(unsigned int j = 0; j < ninitial; j++ )
    {
      in_pdgs_used[j] = false; 
      permutations[j] = -1; 
    }
    // Look for initial state matches
    for(unsigned int j = 0; j < ninitial; j++ )
    {
      for(unsigned int k = 0; k < ninitial; k++ )
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
    for(unsigned int j = 0; j < (nexternal - ninitial); j++ )
    {
      out_pdgs_used[j] = false; 
      permutations[ninitial + j] = -1; 
    }
    // Look for final state matches
    for(unsigned int j = 0; j < (nexternal - ninitial); j++ )
    {
      for(unsigned int k = 0; k < (nexternal - ninitial); k++ )
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
void PY8MEs_R130_P0_sm_gg_ggg::setMomenta(vector < vec_double > momenta_picked)
{
  if (momenta_picked.size() != nexternal)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setMomenta' of class" << 
    " 'PY8MEs_R130_P0_sm_gg_ggg': Incorrect number of" << 
    " momenta specified." << endl; 
    #endif
    throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
  }
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    if (momenta_picked[i].size() != 4)
    {
      #ifdef DEBUG
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R130_P0_sm_gg_ggg': Incorrect number of" << 
      " momenta components specified." << endl; 
      #endif
      throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
    }
    if (momenta_picked[i][0] < 0.0)
    {
      #ifdef DEBUG
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R130_P0_sm_gg_ggg': A momentum was specified" << 
      " with negative energy. Check conventions." << endl; 
      #endif
      throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
    }
    for (unsigned int j = 0; j < 4; j++ )
    {
      p[i][j] = momenta_picked[i][j]; 
    }
  }
}

//--------------------------------------------------------------------------
// Set color configuration to use. An empty vector means sum over all.
void PY8MEs_R130_P0_sm_gg_ggg::setColors(vector<int> colors_picked)
{
  if (colors_picked.size() == 0)
  {
    user_colors = vector<int> (); 
    return; 
  }
  if (colors_picked.size() != (2 * nexternal))
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setColors' of class" << 
    " 'PY8MEs_R130_P0_sm_gg_ggg': Incorrect number" << 
    " of colors specified." << endl; 
    #endif
    throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
  }
  user_colors = vector<int> ((2 * nexternal), 0); 
  for(unsigned int i = 0; i < (2 * nexternal); i++ )
  {
    user_colors[i] = colors_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the helicity configuration to use. Am empty vector means sum over all.
void PY8MEs_R130_P0_sm_gg_ggg::setHelicities(vector<int> helicities_picked) 
{
  if (helicities_picked.size() != nexternal)
  {
    if (helicities_picked.size() == 0)
    {
      user_helicities = vector<int> (); 
      return; 
    }
    #ifdef DEBUG
    cerr <<  "Error in function 'setHelicities' of class" << 
    " 'PY8MEs_R130_P0_sm_gg_ggg': Incorrect number" << 
    " of helicities specified." << endl; 
    #endif
    throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
  }
  user_helicities = vector<int> (nexternal, 0); 
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    user_helicities[i] = helicities_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the permutation to use (will apply to momenta, colors and helicities)
void PY8MEs_R130_P0_sm_gg_ggg::setPermutation(vector<int> perm_picked) 
{
  if (perm_picked.size() != nexternal)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setPermutations' of class" << 
    " 'PY8MEs_R130_P0_sm_gg_ggg': Incorrect number" << 
    " of permutations specified." << endl; 
    #endif
    throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
  }
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    perm[i] = perm_picked[i]; 
  }
}

// Set the proc_ID to use
void PY8MEs_R130_P0_sm_gg_ggg::setProcID(int procID_picked) 
{
  proc_ID = procID_picked; 
}

//--------------------------------------------------------------------------
// Initialize process.

void PY8MEs_R130_P0_sm_gg_ggg::initProc() 
{

  // Initialize flags
  include_symmetry_factors = true; 
  include_helicity_averaging_factors = true; 
  include_color_averaging_factors = true; 

  // Initialize vectors.
  perm = vector<int> (nexternal, 0); 
  user_colors = vector<int> (2 * nexternal, 0); 
  user_helicities = vector<int> (nexternal, 0); 
  p = vector < double * > (); 
  for (unsigned int i = 0; i < nexternal; i++ )
  {
    p.push_back(new double[4]); 
  }
  initColorConfigs(); 
  // Synchronize local variables dependent on the model with the active model.
  mME = vector<double> (nexternal, 0.); 
  syncProcModelParams(); 
  jamp2 = vector < vec_double > (1); 
  jamp2[0] = vector<double> (24, 0.); 
  all_results = vector < vec_vec_double > (1); 
  // The first entry is always the color or helicity avg/summed matrix element.
  all_results[0] = vector < vec_double > (ncomb + 1, vector<double> (24 + 1,
      0.));
}

// Synchronize local variables of the process that depend on the model
// parameters
void PY8MEs_R130_P0_sm_gg_ggg::syncProcModelParams() 
{

  // Instantiate the model class and set parameters that stay fixed during run
  mME[0] = pars->ZERO; 
  mME[1] = pars->ZERO; 
  mME[2] = pars->ZERO; 
  mME[3] = pars->ZERO; 
  mME[4] = pars->ZERO; 
}

//--------------------------------------------------------------------------
// Setter allowing to force particular values for the external masses
void PY8MEs_R130_P0_sm_gg_ggg::setMasses(vec_double external_masses) 
{

  if (external_masses.size() != mME.size())
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setMasses' of class" << 
    " 'PY8MEs_R130_P0_sm_gg_ggg': Incorrect number of" << 
    " masses specified." << endl; 
    #endif
    throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
  }
  for (unsigned int j = 0; j < mME.size(); j++ )
  {
    mME[j] = external_masses[perm[j]]; 
  }
}

//--------------------------------------------------------------------------
// Getter accessing external masses with the correct ordering
vector<double> PY8MEs_R130_P0_sm_gg_ggg::getMasses() 
{

  vec_double external_masses; 
  vector<int> invertedPerm; 
  invertedPerm = invert_mapping(perm); 
  for (unsigned int i = 0; i < mME.size(); i++ )
  {
    external_masses.push_back(mME[invertedPerm[i]]); 
  }
  return external_masses; 

}


// Set all values of the external masses to float(-mode) where mode can be
// 0 : Mass taken from the model
// 1 : Mass taken from p_i^2 if not massless to begin with
// 2 : Mass always taken from p_i^2.
void PY8MEs_R130_P0_sm_gg_ggg::setExternalMassesMode(int mode) 
{
  if (mode != 0 && mode != 1 && mode != 2)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setExternalMassesMode' of class" << 
    " 'PY8MEs_R130_P0_sm_gg_ggg': Incorrect mode selected :" << mode << 
    ". It must be either 0, 1 or 2" << endl; 
    #endif
    throw PY8MEs_R130_P0_sm_gg_ggg_exception; 
  }
  if (mode == 0)
  {
    syncProcModelParams(); 
  }
  else if (mode == 1)
  {
    for (unsigned int j = 0; j < mME.size(); j++ )
    {
      if (mME[j] != pars->ZERO)
      {
        mME[j] = -1.0; 
      }
    }
  }
  else if (mode == 2)
  {
    for (unsigned int j = 0; j < mME.size(); j++ )
    {
      mME[j] = -1.0; 
    }
  }
}

//--------------------------------------------------------------------------
// Evaluate the squared matrix element.

double PY8MEs_R130_P0_sm_gg_ggg::sigmaKin() 
{
  // Set the parameters which change event by event
  pars->setDependentParameters(); 
  pars->setDependentCouplings(); 
  // Reset color flows
  for(int i = 0; i < 24; i++ )
    jamp2[0][i] = 0.; 

  // Save previous values of mME
  vector<double> saved_mME(mME.size(), 0.0); 
  for (unsigned int i = 0; i < mME.size(); i++ )
  {
    if (mME[i] < 0.0)
    {
      saved_mME[i] = mME[i]; 
      mME[i] = sqrt(abs(pow(p[perm[i]][0], 2) - 
      (pow(p[perm[i]][1], 2) + pow(p[perm[i]][2], 2) + pow(p[perm[i]][3],
          2))));
    }
  }

  // Local variables and constants
  const int max_tries = 10; 
  vector < vec_bool > goodhel(nprocesses, vec_bool(ncomb, false)); 
  vec_int ntry(nprocesses, 0); 
  double t = 0.; 
  double result = 0.; 

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
  for (int ihel = -1; ihel + 1 < ((int)all_results[proc_ID].size()); ihel++ )
  {
    // Only if it is the helicity picked
    if (user_ihel != -1 && ihel != user_ihel)
      continue; 
    for (int icolor = -1; icolor + 1 < ((int)all_results[proc_ID][ihel +
        1].size()); icolor++ )
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
      t = matrix_13gg_ggg(); 

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
          for (unsigned int i = 0; i < jamp2[proc_ID].size(); i++ )
          {
            all_results[proc_ID][0][i + 1] += jamp2[proc_ID][i]; 
          }
        }
        all_results[proc_ID][ihel + 1][0] += t; 
        for (unsigned int i = 0; i < jamp2[proc_ID].size(); i++ )
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
  for (int ihel = -1; ihel + 1 < ((int)all_results[proc_ID].size()); ihel++ )
  {
    // Only if it is the helicity picked
    if (user_ihel != -1 && ihel != user_ihel)
      continue; 
    for (int icolor = -1; icolor + 1 < ((int)all_results[proc_ID][ihel +
        1].size()); icolor++ )
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
  if (user_ihel == -1 && include_helicity_averaging_factors)
  {
    result /= denom_hels[proc_ID]; 
    if (user_icol == -1)
    {
      all_results[proc_ID][0][0] /= denom_hels[proc_ID]; 
      for (unsigned int i = 0; i < jamp2[proc_ID].size(); i++ )
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
  if (user_icol == -1 && include_color_averaging_factors)
  {
    result /= denom_colors[proc_ID]; 
    if (user_ihel == -1)
    {
      all_results[proc_ID][0][0] /= denom_colors[proc_ID]; 
      for (unsigned int i = 0; i < ncomb; i++ )
      {
        all_results[proc_ID][i + 1][0] /= denom_colors[proc_ID]; 
      }
    }
    else
    {
      all_results[proc_ID][user_ihel + 1][0] /= denom_colors[proc_ID]; 
    }
  }

  // Reinstate previous values of mME
  for (unsigned int i = 0; i < mME.size(); i++ )
  {
    if (saved_mME[i] < 0.0)
    {
      mME[i] = saved_mME[i]; 
    }
  }

  // Finally return it
  return result; 
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

void PY8MEs_R130_P0_sm_gg_ggg::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  // Calculate all wavefunctions
  vxxxxx(p[perm[0]], mME[0], hel[0], -1, w[0]); 
  vxxxxx(p[perm[1]], mME[1], hel[1], -1, w[1]); 
  vxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  vxxxxx(p[perm[3]], mME[3], hel[3], +1, w[3]); 
  vxxxxx(p[perm[4]], mME[4], hel[4], +1, w[4]); 
  VVV1P0_1(w[0], w[1], pars->GC_10, pars->ZERO, pars->ZERO, w[5]); 
  VVV1P0_1(w[2], w[3], pars->GC_10, pars->ZERO, pars->ZERO, w[6]); 
  VVV1P0_1(w[2], w[4], pars->GC_10, pars->ZERO, pars->ZERO, w[7]); 
  VVV1P0_1(w[3], w[4], pars->GC_10, pars->ZERO, pars->ZERO, w[8]); 
  VVV1P0_1(w[0], w[2], pars->GC_10, pars->ZERO, pars->ZERO, w[9]); 
  VVV1P0_1(w[1], w[3], pars->GC_10, pars->ZERO, pars->ZERO, w[10]); 
  VVV1P0_1(w[1], w[4], pars->GC_10, pars->ZERO, pars->ZERO, w[11]); 
  VVV1P0_1(w[0], w[3], pars->GC_10, pars->ZERO, pars->ZERO, w[12]); 
  VVV1P0_1(w[1], w[2], pars->GC_10, pars->ZERO, pars->ZERO, w[13]); 
  VVV1P0_1(w[0], w[4], pars->GC_10, pars->ZERO, pars->ZERO, w[14]); 
  VVVV1P0_1(w[0], w[1], w[2], pars->GC_12, pars->ZERO, pars->ZERO, w[15]); 
  VVVV3P0_1(w[0], w[1], w[2], pars->GC_12, pars->ZERO, pars->ZERO, w[16]); 
  VVVV4P0_1(w[0], w[1], w[2], pars->GC_12, pars->ZERO, pars->ZERO, w[17]); 
  VVVV1P0_1(w[0], w[1], w[3], pars->GC_12, pars->ZERO, pars->ZERO, w[18]); 
  VVVV3P0_1(w[0], w[1], w[3], pars->GC_12, pars->ZERO, pars->ZERO, w[19]); 
  VVVV4P0_1(w[0], w[1], w[3], pars->GC_12, pars->ZERO, pars->ZERO, w[20]); 
  VVVV1P0_1(w[0], w[1], w[4], pars->GC_12, pars->ZERO, pars->ZERO, w[21]); 
  VVVV3P0_1(w[0], w[1], w[4], pars->GC_12, pars->ZERO, pars->ZERO, w[22]); 
  VVVV4P0_1(w[0], w[1], w[4], pars->GC_12, pars->ZERO, pars->ZERO, w[23]); 
  VVVV1P0_1(w[0], w[2], w[3], pars->GC_12, pars->ZERO, pars->ZERO, w[24]); 
  VVVV3P0_1(w[0], w[2], w[3], pars->GC_12, pars->ZERO, pars->ZERO, w[25]); 
  VVVV4P0_1(w[0], w[2], w[3], pars->GC_12, pars->ZERO, pars->ZERO, w[26]); 
  VVVV1P0_1(w[0], w[2], w[4], pars->GC_12, pars->ZERO, pars->ZERO, w[27]); 
  VVVV3P0_1(w[0], w[2], w[4], pars->GC_12, pars->ZERO, pars->ZERO, w[28]); 
  VVVV4P0_1(w[0], w[2], w[4], pars->GC_12, pars->ZERO, pars->ZERO, w[29]); 
  VVVV1P0_1(w[0], w[3], w[4], pars->GC_12, pars->ZERO, pars->ZERO, w[30]); 
  VVVV3P0_1(w[0], w[3], w[4], pars->GC_12, pars->ZERO, pars->ZERO, w[31]); 
  VVVV4P0_1(w[0], w[3], w[4], pars->GC_12, pars->ZERO, pars->ZERO, w[32]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  VVV1_0(w[5], w[6], w[4], pars->GC_10, amp[0]); 
  VVV1_0(w[5], w[7], w[3], pars->GC_10, amp[1]); 
  VVV1_0(w[5], w[2], w[8], pars->GC_10, amp[2]); 
  VVVV1_0(w[2], w[3], w[4], w[5], pars->GC_12, amp[3]); 
  VVVV3_0(w[2], w[3], w[4], w[5], pars->GC_12, amp[4]); 
  VVVV4_0(w[2], w[3], w[4], w[5], pars->GC_12, amp[5]); 
  VVV1_0(w[9], w[10], w[4], pars->GC_10, amp[6]); 
  VVV1_0(w[9], w[11], w[3], pars->GC_10, amp[7]); 
  VVV1_0(w[9], w[1], w[8], pars->GC_10, amp[8]); 
  VVVV1_0(w[1], w[3], w[4], w[9], pars->GC_12, amp[9]); 
  VVVV3_0(w[1], w[3], w[4], w[9], pars->GC_12, amp[10]); 
  VVVV4_0(w[1], w[3], w[4], w[9], pars->GC_12, amp[11]); 
  VVV1_0(w[12], w[13], w[4], pars->GC_10, amp[12]); 
  VVV1_0(w[12], w[11], w[2], pars->GC_10, amp[13]); 
  VVV1_0(w[12], w[1], w[7], pars->GC_10, amp[14]); 
  VVVV1_0(w[1], w[2], w[4], w[12], pars->GC_12, amp[15]); 
  VVVV3_0(w[1], w[2], w[4], w[12], pars->GC_12, amp[16]); 
  VVVV4_0(w[1], w[2], w[4], w[12], pars->GC_12, amp[17]); 
  VVV1_0(w[14], w[13], w[3], pars->GC_10, amp[18]); 
  VVV1_0(w[14], w[10], w[2], pars->GC_10, amp[19]); 
  VVV1_0(w[14], w[1], w[6], pars->GC_10, amp[20]); 
  VVVV1_0(w[1], w[2], w[3], w[14], pars->GC_12, amp[21]); 
  VVVV3_0(w[1], w[2], w[3], w[14], pars->GC_12, amp[22]); 
  VVVV4_0(w[1], w[2], w[3], w[14], pars->GC_12, amp[23]); 
  VVV1_0(w[0], w[13], w[8], pars->GC_10, amp[24]); 
  VVV1_0(w[0], w[10], w[7], pars->GC_10, amp[25]); 
  VVV1_0(w[0], w[11], w[6], pars->GC_10, amp[26]); 
  VVV1_0(w[3], w[4], w[15], pars->GC_10, amp[27]); 
  VVV1_0(w[3], w[4], w[16], pars->GC_10, amp[28]); 
  VVV1_0(w[3], w[4], w[17], pars->GC_10, amp[29]); 
  VVV1_0(w[2], w[4], w[18], pars->GC_10, amp[30]); 
  VVV1_0(w[2], w[4], w[19], pars->GC_10, amp[31]); 
  VVV1_0(w[2], w[4], w[20], pars->GC_10, amp[32]); 
  VVV1_0(w[2], w[3], w[21], pars->GC_10, amp[33]); 
  VVV1_0(w[2], w[3], w[22], pars->GC_10, amp[34]); 
  VVV1_0(w[2], w[3], w[23], pars->GC_10, amp[35]); 
  VVV1_0(w[1], w[4], w[24], pars->GC_10, amp[36]); 
  VVV1_0(w[1], w[4], w[25], pars->GC_10, amp[37]); 
  VVV1_0(w[1], w[4], w[26], pars->GC_10, amp[38]); 
  VVV1_0(w[1], w[3], w[27], pars->GC_10, amp[39]); 
  VVV1_0(w[1], w[3], w[28], pars->GC_10, amp[40]); 
  VVV1_0(w[1], w[3], w[29], pars->GC_10, amp[41]); 
  VVV1_0(w[1], w[2], w[30], pars->GC_10, amp[42]); 
  VVV1_0(w[1], w[2], w[31], pars->GC_10, amp[43]); 
  VVV1_0(w[1], w[2], w[32], pars->GC_10, amp[44]); 


}
double PY8MEs_R130_P0_sm_gg_ggg::matrix_13gg_ggg() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 45;
  const int ncolor = 24; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {108, 108, 108, 108, 108, 108, 108, 108,
      108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
      108, 108};
  static const double cf[ncolor][ncolor] = {{455, -58, -58, 14, 14, 68, -58,
      -4, 14, -58, 5, -4, 14, 5, 68, -4, 14, 68, -58, -4, -4, 68, 68, -40},
      {-58, 455, 14, 68, -58, 14, -4, -58, 5, -4, 14, -58, -58, -4, -4, 68, 68,
      -40, 14, 5, 68, -4, 14, 68}, {-58, 14, 455, -58, 68, 14, 14, 5, 68, -4,
      14, 68, -58, -4, 14, -58, 5, -4, -4, -58, 68, -40, -4, 68}, {14, 68, -58,
      455, 14, -58, -58, -4, -4, 68, 68, -40, -4, -58, 5, -4, 14, -58, 5, 14,
      14, 68, 68, -4}, {14, -58, 68, 14, 455, -58, 5, 14, 14, 68, 68, -4, -4,
      -58, 68, -40, -4, 68, -58, -4, 14, -58, 5, -4}, {68, 14, 14, -58, -58,
      455, -4, -58, 68, -40, -4, 68, 5, 14, 14, 68, 68, -4, -4, -58, 5, -4, 14,
      -58}, {-58, -4, 14, -58, 5, -4, 455, -58, -58, 14, 14, 68, 68, -4, 14, 5,
      68, 14, -4, 68, -58, -4, -40, 68}, {-4, -58, 5, -4, 14, -58, -58, 455,
      14, 68, -58, 14, -4, 68, -58, -4, -40, 68, 68, -4, 14, 5, 68, 14}, {14,
      5, 68, -4, 14, 68, -58, 14, 455, -58, 68, 14, 14, -58, -58, -4, -4, 5,
      68, -40, -4, -58, 68, -4}, {-58, -4, -4, 68, 68, -40, 14, 68, -58, 455,
      14, -58, 5, -4, -4, -58, -58, 14, 14, 68, 5, 14, -4, 68}, {5, 14, 14, 68,
      68, -4, 14, -58, 68, 14, 455, -58, 68, -40, -4, -58, 68, -4, 14, -58,
      -58, -4, -4, 5}, {-4, -58, 68, -40, -4, 68, 68, 14, 14, -58, -58, 455,
      14, 68, 5, 14, -4, 68, 5, -4, -4, -58, -58, 14}, {14, -58, -58, -4, -4,
      5, 68, -4, 14, 5, 68, 14, 455, -58, -58, 14, 14, 68, 68, -4, -40, 68,
      -58, -4}, {5, -4, -4, -58, -58, 14, -4, 68, -58, -4, -40, 68, -58, 455,
      14, 68, -58, 14, -4, 68, 68, 14, 14, 5}, {68, -4, 14, 5, 68, 14, 14, -58,
      -58, -4, -4, 5, -58, 14, 455, -58, 68, 14, -40, 68, 68, -4, -4, -58},
      {-4, 68, -58, -4, -40, 68, 5, -4, -4, -58, -58, 14, 14, 68, -58, 455, 14,
      -58, 68, 14, -4, 68, 5, 14}, {14, 68, 5, 14, -4, 68, 68, -40, -4, -58,
      68, -4, 14, -58, 68, 14, 455, -58, -58, 14, -4, 5, -58, -4}, {68, -40,
      -4, -58, 68, -4, 14, 68, 5, 14, -4, 68, 68, 14, 14, -58, -58, 455, -4, 5,
      -58, 14, -4, -58}, {-58, 14, -4, 5, -58, -4, -4, 68, 68, 14, 14, 5, 68,
      -4, -40, 68, -58, -4, 455, -58, -58, 14, 14, 68}, {-4, 5, -58, 14, -4,
      -58, 68, -4, -40, 68, -58, -4, -4, 68, 68, 14, 14, 5, -58, 455, 14, 68,
      -58, 14}, {-4, 68, 68, 14, 14, 5, -58, 14, -4, 5, -58, -4, -40, 68, 68,
      -4, -4, -58, -58, 14, 455, -58, 68, 14}, {68, -4, -40, 68, -58, -4, -4,
      5, -58, 14, -4, -58, 68, 14, -4, 68, 5, 14, 14, 68, -58, 455, 14, -58},
      {68, 14, -4, 68, 5, 14, -40, 68, 68, -4, -4, -58, -58, 14, -4, 5, -58,
      -4, 14, -58, 68, 14, 455, -58}, {-40, 68, 68, -4, -4, -58, 68, 14, -4,
      68, 5, 14, -4, 5, -58, 14, -4, -58, 68, 14, 14, -58, -58, 455}};

  // Calculate color flows
  jamp[0] = +2. * (+Complex<double> (0, 1) * amp[0] + Complex<double> (0, 1) *
      amp[2] + Complex<double> (0, 1) * amp[3] - Complex<double> (0, 1) *
      amp[5] - Complex<double> (0, 1) * amp[18] - Complex<double> (0, 1) *
      amp[20] - Complex<double> (0, 1) * amp[21] + Complex<double> (0, 1) *
      amp[23] + Complex<double> (0, 1) * amp[24] - Complex<double> (0, 1) *
      amp[29] + Complex<double> (0, 1) * amp[27] + Complex<double> (0, 1) *
      amp[35] + Complex<double> (0, 1) * amp[34] - Complex<double> (0, 1) *
      amp[43] - Complex<double> (0, 1) * amp[42]);
  jamp[1] = +2. * (+Complex<double> (0, 1) * amp[1] - Complex<double> (0, 1) *
      amp[2] + Complex<double> (0, 1) * amp[4] + Complex<double> (0, 1) *
      amp[5] - Complex<double> (0, 1) * amp[12] - Complex<double> (0, 1) *
      amp[14] - Complex<double> (0, 1) * amp[15] + Complex<double> (0, 1) *
      amp[17] - Complex<double> (0, 1) * amp[24] + Complex<double> (0, 1) *
      amp[29] - Complex<double> (0, 1) * amp[27] + Complex<double> (0, 1) *
      amp[32] + Complex<double> (0, 1) * amp[31] - Complex<double> (0, 1) *
      amp[44] + Complex<double> (0, 1) * amp[42]);
  jamp[2] = +2. * (-Complex<double> (0, 1) * amp[0] - Complex<double> (0, 1) *
      amp[1] - Complex<double> (0, 1) * amp[4] - Complex<double> (0, 1) *
      amp[3] - Complex<double> (0, 1) * amp[19] + Complex<double> (0, 1) *
      amp[20] - Complex<double> (0, 1) * amp[22] - Complex<double> (0, 1) *
      amp[23] + Complex<double> (0, 1) * amp[25] - Complex<double> (0, 1) *
      amp[32] + Complex<double> (0, 1) * amp[30] - Complex<double> (0, 1) *
      amp[35] - Complex<double> (0, 1) * amp[34] - Complex<double> (0, 1) *
      amp[40] - Complex<double> (0, 1) * amp[39]);
  jamp[3] = +2. * (+Complex<double> (0, 1) * amp[1] - Complex<double> (0, 1) *
      amp[2] + Complex<double> (0, 1) * amp[4] + Complex<double> (0, 1) *
      amp[5] - Complex<double> (0, 1) * amp[6] - Complex<double> (0, 1) *
      amp[8] - Complex<double> (0, 1) * amp[9] + Complex<double> (0, 1) *
      amp[11] - Complex<double> (0, 1) * amp[25] + Complex<double> (0, 1) *
      amp[29] + Complex<double> (0, 1) * amp[28] + Complex<double> (0, 1) *
      amp[32] - Complex<double> (0, 1) * amp[30] - Complex<double> (0, 1) *
      amp[41] + Complex<double> (0, 1) * amp[39]);
  jamp[4] = +2. * (-Complex<double> (0, 1) * amp[0] - Complex<double> (0, 1) *
      amp[1] - Complex<double> (0, 1) * amp[4] - Complex<double> (0, 1) *
      amp[3] - Complex<double> (0, 1) * amp[13] + Complex<double> (0, 1) *
      amp[14] - Complex<double> (0, 1) * amp[16] - Complex<double> (0, 1) *
      amp[17] + Complex<double> (0, 1) * amp[26] - Complex<double> (0, 1) *
      amp[32] - Complex<double> (0, 1) * amp[31] - Complex<double> (0, 1) *
      amp[35] + Complex<double> (0, 1) * amp[33] - Complex<double> (0, 1) *
      amp[37] - Complex<double> (0, 1) * amp[36]);
  jamp[5] = +2. * (+Complex<double> (0, 1) * amp[0] + Complex<double> (0, 1) *
      amp[2] + Complex<double> (0, 1) * amp[3] - Complex<double> (0, 1) *
      amp[5] - Complex<double> (0, 1) * amp[7] + Complex<double> (0, 1) *
      amp[8] - Complex<double> (0, 1) * amp[10] - Complex<double> (0, 1) *
      amp[11] - Complex<double> (0, 1) * amp[26] - Complex<double> (0, 1) *
      amp[29] - Complex<double> (0, 1) * amp[28] + Complex<double> (0, 1) *
      amp[35] - Complex<double> (0, 1) * amp[33] - Complex<double> (0, 1) *
      amp[38] + Complex<double> (0, 1) * amp[36]);
  jamp[6] = +2. * (+Complex<double> (0, 1) * amp[6] + Complex<double> (0, 1) *
      amp[8] + Complex<double> (0, 1) * amp[9] - Complex<double> (0, 1) *
      amp[11] + Complex<double> (0, 1) * amp[18] + Complex<double> (0, 1) *
      amp[19] + Complex<double> (0, 1) * amp[22] + Complex<double> (0, 1) *
      amp[21] - Complex<double> (0, 1) * amp[24] - Complex<double> (0, 1) *
      amp[28] - Complex<double> (0, 1) * amp[27] + Complex<double> (0, 1) *
      amp[41] + Complex<double> (0, 1) * amp[40] + Complex<double> (0, 1) *
      amp[43] + Complex<double> (0, 1) * amp[42]);
  jamp[7] = +2. * (+Complex<double> (0, 1) * amp[7] - Complex<double> (0, 1) *
      amp[8] + Complex<double> (0, 1) * amp[10] + Complex<double> (0, 1) *
      amp[11] + Complex<double> (0, 1) * amp[12] + Complex<double> (0, 1) *
      amp[13] + Complex<double> (0, 1) * amp[16] + Complex<double> (0, 1) *
      amp[15] + Complex<double> (0, 1) * amp[24] + Complex<double> (0, 1) *
      amp[28] + Complex<double> (0, 1) * amp[27] + Complex<double> (0, 1) *
      amp[38] + Complex<double> (0, 1) * amp[37] + Complex<double> (0, 1) *
      amp[44] - Complex<double> (0, 1) * amp[42]);
  jamp[8] = +2. * (-Complex<double> (0, 1) * amp[6] - Complex<double> (0, 1) *
      amp[7] - Complex<double> (0, 1) * amp[10] - Complex<double> (0, 1) *
      amp[9] - Complex<double> (0, 1) * amp[19] + Complex<double> (0, 1) *
      amp[20] - Complex<double> (0, 1) * amp[22] - Complex<double> (0, 1) *
      amp[23] - Complex<double> (0, 1) * amp[26] - Complex<double> (0, 1) *
      amp[34] - Complex<double> (0, 1) * amp[33] - Complex<double> (0, 1) *
      amp[38] + Complex<double> (0, 1) * amp[36] - Complex<double> (0, 1) *
      amp[41] - Complex<double> (0, 1) * amp[40]);
  jamp[9] = +2. * (-Complex<double> (0, 1) * amp[0] - Complex<double> (0, 1) *
      amp[2] - Complex<double> (0, 1) * amp[3] + Complex<double> (0, 1) *
      amp[5] + Complex<double> (0, 1) * amp[7] - Complex<double> (0, 1) *
      amp[8] + Complex<double> (0, 1) * amp[10] + Complex<double> (0, 1) *
      amp[11] + Complex<double> (0, 1) * amp[26] + Complex<double> (0, 1) *
      amp[29] + Complex<double> (0, 1) * amp[28] - Complex<double> (0, 1) *
      amp[35] + Complex<double> (0, 1) * amp[33] + Complex<double> (0, 1) *
      amp[38] - Complex<double> (0, 1) * amp[36]);
  jamp[10] = +2. * (-Complex<double> (0, 1) * amp[6] - Complex<double> (0, 1) *
      amp[7] - Complex<double> (0, 1) * amp[10] - Complex<double> (0, 1) *
      amp[9] - Complex<double> (0, 1) * amp[13] + Complex<double> (0, 1) *
      amp[14] - Complex<double> (0, 1) * amp[16] - Complex<double> (0, 1) *
      amp[17] - Complex<double> (0, 1) * amp[25] - Complex<double> (0, 1) *
      amp[31] - Complex<double> (0, 1) * amp[30] - Complex<double> (0, 1) *
      amp[38] - Complex<double> (0, 1) * amp[37] - Complex<double> (0, 1) *
      amp[41] + Complex<double> (0, 1) * amp[39]);
  jamp[11] = +2. * (-Complex<double> (0, 1) * amp[1] + Complex<double> (0, 1) *
      amp[2] - Complex<double> (0, 1) * amp[4] - Complex<double> (0, 1) *
      amp[5] + Complex<double> (0, 1) * amp[6] + Complex<double> (0, 1) *
      amp[8] + Complex<double> (0, 1) * amp[9] - Complex<double> (0, 1) *
      amp[11] + Complex<double> (0, 1) * amp[25] - Complex<double> (0, 1) *
      amp[29] - Complex<double> (0, 1) * amp[28] - Complex<double> (0, 1) *
      amp[32] + Complex<double> (0, 1) * amp[30] + Complex<double> (0, 1) *
      amp[41] - Complex<double> (0, 1) * amp[39]);
  jamp[12] = +2. * (+Complex<double> (0, 1) * amp[12] + Complex<double> (0, 1)
      * amp[14] + Complex<double> (0, 1) * amp[15] - Complex<double> (0, 1) *
      amp[17] + Complex<double> (0, 1) * amp[18] + Complex<double> (0, 1) *
      amp[19] + Complex<double> (0, 1) * amp[22] + Complex<double> (0, 1) *
      amp[21] - Complex<double> (0, 1) * amp[25] - Complex<double> (0, 1) *
      amp[31] - Complex<double> (0, 1) * amp[30] + Complex<double> (0, 1) *
      amp[40] + Complex<double> (0, 1) * amp[39] + Complex<double> (0, 1) *
      amp[44] + Complex<double> (0, 1) * amp[43]);
  jamp[13] = +2. * (+Complex<double> (0, 1) * amp[6] + Complex<double> (0, 1) *
      amp[7] + Complex<double> (0, 1) * amp[10] + Complex<double> (0, 1) *
      amp[9] + Complex<double> (0, 1) * amp[13] - Complex<double> (0, 1) *
      amp[14] + Complex<double> (0, 1) * amp[16] + Complex<double> (0, 1) *
      amp[17] + Complex<double> (0, 1) * amp[25] + Complex<double> (0, 1) *
      amp[31] + Complex<double> (0, 1) * amp[30] + Complex<double> (0, 1) *
      amp[38] + Complex<double> (0, 1) * amp[37] + Complex<double> (0, 1) *
      amp[41] - Complex<double> (0, 1) * amp[39]);
  jamp[14] = +2. * (-Complex<double> (0, 1) * amp[12] - Complex<double> (0, 1)
      * amp[13] - Complex<double> (0, 1) * amp[16] - Complex<double> (0, 1) *
      amp[15] - Complex<double> (0, 1) * amp[18] - Complex<double> (0, 1) *
      amp[20] - Complex<double> (0, 1) * amp[21] + Complex<double> (0, 1) *
      amp[23] + Complex<double> (0, 1) * amp[26] + Complex<double> (0, 1) *
      amp[34] + Complex<double> (0, 1) * amp[33] - Complex<double> (0, 1) *
      amp[37] - Complex<double> (0, 1) * amp[36] - Complex<double> (0, 1) *
      amp[44] - Complex<double> (0, 1) * amp[43]);
  jamp[15] = +2. * (+Complex<double> (0, 1) * amp[0] + Complex<double> (0, 1) *
      amp[1] + Complex<double> (0, 1) * amp[4] + Complex<double> (0, 1) *
      amp[3] + Complex<double> (0, 1) * amp[13] - Complex<double> (0, 1) *
      amp[14] + Complex<double> (0, 1) * amp[16] + Complex<double> (0, 1) *
      amp[17] - Complex<double> (0, 1) * amp[26] + Complex<double> (0, 1) *
      amp[32] + Complex<double> (0, 1) * amp[31] + Complex<double> (0, 1) *
      amp[35] - Complex<double> (0, 1) * amp[33] + Complex<double> (0, 1) *
      amp[37] + Complex<double> (0, 1) * amp[36]);
  jamp[16] = +2. * (-Complex<double> (0, 1) * amp[7] + Complex<double> (0, 1) *
      amp[8] - Complex<double> (0, 1) * amp[10] - Complex<double> (0, 1) *
      amp[11] - Complex<double> (0, 1) * amp[12] - Complex<double> (0, 1) *
      amp[13] - Complex<double> (0, 1) * amp[16] - Complex<double> (0, 1) *
      amp[15] - Complex<double> (0, 1) * amp[24] - Complex<double> (0, 1) *
      amp[28] - Complex<double> (0, 1) * amp[27] - Complex<double> (0, 1) *
      amp[38] - Complex<double> (0, 1) * amp[37] - Complex<double> (0, 1) *
      amp[44] + Complex<double> (0, 1) * amp[42]);
  jamp[17] = +2. * (-Complex<double> (0, 1) * amp[1] + Complex<double> (0, 1) *
      amp[2] - Complex<double> (0, 1) * amp[4] - Complex<double> (0, 1) *
      amp[5] + Complex<double> (0, 1) * amp[12] + Complex<double> (0, 1) *
      amp[14] + Complex<double> (0, 1) * amp[15] - Complex<double> (0, 1) *
      amp[17] + Complex<double> (0, 1) * amp[24] - Complex<double> (0, 1) *
      amp[29] + Complex<double> (0, 1) * amp[27] - Complex<double> (0, 1) *
      amp[32] - Complex<double> (0, 1) * amp[31] + Complex<double> (0, 1) *
      amp[44] - Complex<double> (0, 1) * amp[42]);
  jamp[18] = +2. * (+Complex<double> (0, 1) * amp[12] + Complex<double> (0, 1)
      * amp[13] + Complex<double> (0, 1) * amp[16] + Complex<double> (0, 1) *
      amp[15] + Complex<double> (0, 1) * amp[18] + Complex<double> (0, 1) *
      amp[20] + Complex<double> (0, 1) * amp[21] - Complex<double> (0, 1) *
      amp[23] - Complex<double> (0, 1) * amp[26] - Complex<double> (0, 1) *
      amp[34] - Complex<double> (0, 1) * amp[33] + Complex<double> (0, 1) *
      amp[37] + Complex<double> (0, 1) * amp[36] + Complex<double> (0, 1) *
      amp[44] + Complex<double> (0, 1) * amp[43]);
  jamp[19] = +2. * (+Complex<double> (0, 1) * amp[6] + Complex<double> (0, 1) *
      amp[7] + Complex<double> (0, 1) * amp[10] + Complex<double> (0, 1) *
      amp[9] + Complex<double> (0, 1) * amp[19] - Complex<double> (0, 1) *
      amp[20] + Complex<double> (0, 1) * amp[22] + Complex<double> (0, 1) *
      amp[23] + Complex<double> (0, 1) * amp[26] + Complex<double> (0, 1) *
      amp[34] + Complex<double> (0, 1) * amp[33] + Complex<double> (0, 1) *
      amp[38] - Complex<double> (0, 1) * amp[36] + Complex<double> (0, 1) *
      amp[41] + Complex<double> (0, 1) * amp[40]);
  jamp[20] = +2. * (-Complex<double> (0, 1) * amp[12] - Complex<double> (0, 1)
      * amp[14] - Complex<double> (0, 1) * amp[15] + Complex<double> (0, 1) *
      amp[17] - Complex<double> (0, 1) * amp[18] - Complex<double> (0, 1) *
      amp[19] - Complex<double> (0, 1) * amp[22] - Complex<double> (0, 1) *
      amp[21] + Complex<double> (0, 1) * amp[25] + Complex<double> (0, 1) *
      amp[31] + Complex<double> (0, 1) * amp[30] - Complex<double> (0, 1) *
      amp[40] - Complex<double> (0, 1) * amp[39] - Complex<double> (0, 1) *
      amp[44] - Complex<double> (0, 1) * amp[43]);
  jamp[21] = +2. * (+Complex<double> (0, 1) * amp[0] + Complex<double> (0, 1) *
      amp[1] + Complex<double> (0, 1) * amp[4] + Complex<double> (0, 1) *
      amp[3] + Complex<double> (0, 1) * amp[19] - Complex<double> (0, 1) *
      amp[20] + Complex<double> (0, 1) * amp[22] + Complex<double> (0, 1) *
      amp[23] - Complex<double> (0, 1) * amp[25] + Complex<double> (0, 1) *
      amp[32] - Complex<double> (0, 1) * amp[30] + Complex<double> (0, 1) *
      amp[35] + Complex<double> (0, 1) * amp[34] + Complex<double> (0, 1) *
      amp[40] + Complex<double> (0, 1) * amp[39]);
  jamp[22] = +2. * (-Complex<double> (0, 1) * amp[6] - Complex<double> (0, 1) *
      amp[8] - Complex<double> (0, 1) * amp[9] + Complex<double> (0, 1) *
      amp[11] - Complex<double> (0, 1) * amp[18] - Complex<double> (0, 1) *
      amp[19] - Complex<double> (0, 1) * amp[22] - Complex<double> (0, 1) *
      amp[21] + Complex<double> (0, 1) * amp[24] + Complex<double> (0, 1) *
      amp[28] + Complex<double> (0, 1) * amp[27] - Complex<double> (0, 1) *
      amp[41] - Complex<double> (0, 1) * amp[40] - Complex<double> (0, 1) *
      amp[43] - Complex<double> (0, 1) * amp[42]);
  jamp[23] = +2. * (-Complex<double> (0, 1) * amp[0] - Complex<double> (0, 1) *
      amp[2] - Complex<double> (0, 1) * amp[3] + Complex<double> (0, 1) *
      amp[5] + Complex<double> (0, 1) * amp[18] + Complex<double> (0, 1) *
      amp[20] + Complex<double> (0, 1) * amp[21] - Complex<double> (0, 1) *
      amp[23] - Complex<double> (0, 1) * amp[24] + Complex<double> (0, 1) *
      amp[29] - Complex<double> (0, 1) * amp[27] - Complex<double> (0, 1) *
      amp[35] - Complex<double> (0, 1) * amp[34] + Complex<double> (0, 1) *
      amp[43] + Complex<double> (0, 1) * amp[42]);

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
    jamp2[0][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}


}  // end namespace PY8MEs_namespace

