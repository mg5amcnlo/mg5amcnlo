//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "PY8MEs_R130_P6_sm_qq_gqq.h"
#include "HelAmps_sm.h"

using namespace Pythia8_sm; 

namespace PY8MEs_namespace 
{
//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u > g u u WEIGHTED<=3 @130
// Process: c c > g c c WEIGHTED<=3 @130
// Process: d d > g d d WEIGHTED<=3 @130
// Process: s s > g s s WEIGHTED<=3 @130
// Process: u u~ > g u u~ WEIGHTED<=3 @130
// Process: c c~ > g c c~ WEIGHTED<=3 @130
// Process: d d~ > g d d~ WEIGHTED<=3 @130
// Process: s s~ > g s s~ WEIGHTED<=3 @130
// Process: u~ u~ > g u~ u~ WEIGHTED<=3 @130
// Process: c~ c~ > g c~ c~ WEIGHTED<=3 @130
// Process: d~ d~ > g d~ d~ WEIGHTED<=3 @130
// Process: s~ s~ > g s~ s~ WEIGHTED<=3 @130
// Process: u c > g u c WEIGHTED<=3 @130
// Process: u d > g u d WEIGHTED<=3 @130
// Process: u s > g u s WEIGHTED<=3 @130
// Process: c d > g c d WEIGHTED<=3 @130
// Process: c s > g c s WEIGHTED<=3 @130
// Process: d s > g d s WEIGHTED<=3 @130
// Process: u u~ > g c c~ WEIGHTED<=3 @130
// Process: u u~ > g d d~ WEIGHTED<=3 @130
// Process: u u~ > g s s~ WEIGHTED<=3 @130
// Process: c c~ > g u u~ WEIGHTED<=3 @130
// Process: c c~ > g d d~ WEIGHTED<=3 @130
// Process: c c~ > g s s~ WEIGHTED<=3 @130
// Process: d d~ > g u u~ WEIGHTED<=3 @130
// Process: d d~ > g c c~ WEIGHTED<=3 @130
// Process: d d~ > g s s~ WEIGHTED<=3 @130
// Process: s s~ > g u u~ WEIGHTED<=3 @130
// Process: s s~ > g c c~ WEIGHTED<=3 @130
// Process: s s~ > g d d~ WEIGHTED<=3 @130
// Process: u c~ > g u c~ WEIGHTED<=3 @130
// Process: u d~ > g u d~ WEIGHTED<=3 @130
// Process: u s~ > g u s~ WEIGHTED<=3 @130
// Process: c u~ > g c u~ WEIGHTED<=3 @130
// Process: c d~ > g c d~ WEIGHTED<=3 @130
// Process: c s~ > g c s~ WEIGHTED<=3 @130
// Process: d u~ > g d u~ WEIGHTED<=3 @130
// Process: d c~ > g d c~ WEIGHTED<=3 @130
// Process: d s~ > g d s~ WEIGHTED<=3 @130
// Process: s u~ > g s u~ WEIGHTED<=3 @130
// Process: s c~ > g s c~ WEIGHTED<=3 @130
// Process: s d~ > g s d~ WEIGHTED<=3 @130
// Process: u~ c~ > g u~ c~ WEIGHTED<=3 @130
// Process: u~ d~ > g u~ d~ WEIGHTED<=3 @130
// Process: u~ s~ > g u~ s~ WEIGHTED<=3 @130
// Process: c~ d~ > g c~ d~ WEIGHTED<=3 @130
// Process: c~ s~ > g c~ s~ WEIGHTED<=3 @130
// Process: d~ s~ > g d~ s~ WEIGHTED<=3 @130

// Exception class
class PY8MEs_R130_P6_sm_qq_gqqException : public exception
{
  virtual const char * what() const throw()
  {
    return "Exception in class 'PY8MEs_R130_P6_sm_qq_gqq'."; 
  }
}
PY8MEs_R130_P6_sm_qq_gqq_exception; 

std::set<int> PY8MEs_R130_P6_sm_qq_gqq::s_channel_proc = std::set<int>
    (createset<int> ());

int PY8MEs_R130_P6_sm_qq_gqq::helicities[ncomb][nexternal] = {{-1, -1, -1, -1,
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
int PY8MEs_R130_P6_sm_qq_gqq::denom_colors[nprocesses] = {9, 9, 9, 9, 9, 9, 9}; 
int PY8MEs_R130_P6_sm_qq_gqq::denom_hels[nprocesses] = {4, 4, 4, 4, 4, 4, 4}; 
int PY8MEs_R130_P6_sm_qq_gqq::denom_iden[nprocesses] = {2, 1, 2, 1, 1, 1, 1}; 

//--------------------------------------------------------------------------
// Color config initialization
void PY8MEs_R130_P6_sm_qq_gqq::initColorConfigs() 
{
  color_configs = vector < vec_vec_int > (); 
  jamp_nc_relative_power = vector < vec_int > (); 

  // Color flows of process Process: u u > g u u WEIGHTED<=3 @130
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[0].push_back(vec_int(createvector<int>
      (3)(0)(2)(0)(3)(1)(1)(0)(2)(0)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #1
  color_configs[0].push_back(vec_int(createvector<int>
      (2)(0)(3)(0)(3)(1)(1)(0)(2)(0)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #2
  color_configs[0].push_back(vec_int(createvector<int>
      (3)(0)(1)(0)(3)(2)(1)(0)(2)(0)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #3
  color_configs[0].push_back(vec_int(createvector<int>
      (1)(0)(3)(0)(3)(2)(1)(0)(2)(0)));
  jamp_nc_relative_power[0].push_back(0); 

  // Color flows of process Process: u u~ > g u u~ WEIGHTED<=3 @130
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[1].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(3)(2)(2)(0)(0)(3)));
  jamp_nc_relative_power[1].push_back(0); 
  // JAMP #1
  color_configs[1].push_back(vec_int(createvector<int>
      (3)(0)(0)(1)(3)(2)(2)(0)(0)(1)));
  jamp_nc_relative_power[1].push_back(0); 
  // JAMP #2
  color_configs[1].push_back(vec_int(createvector<int>
      (3)(0)(0)(1)(3)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[1].push_back(0); 
  // JAMP #3
  color_configs[1].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(3)(1)(2)(0)(0)(3)));
  jamp_nc_relative_power[1].push_back(0); 

  // Color flows of process Process: u~ u~ > g u~ u~ WEIGHTED<=3 @130
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[2].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(3)(2)(0)(1)(0)(3)));
  jamp_nc_relative_power[2].push_back(0); 
  // JAMP #1
  color_configs[2].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(3)(2)(0)(3)(0)(1)));
  jamp_nc_relative_power[2].push_back(0); 
  // JAMP #2
  color_configs[2].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(3)(1)(0)(2)(0)(3)));
  jamp_nc_relative_power[2].push_back(0); 
  // JAMP #3
  color_configs[2].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(3)(1)(0)(3)(0)(2)));
  jamp_nc_relative_power[2].push_back(0); 

  // Color flows of process Process: u c > g u c WEIGHTED<=3 @130
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[3].push_back(vec_int(createvector<int>
      (3)(0)(2)(0)(3)(1)(1)(0)(2)(0)));
  jamp_nc_relative_power[3].push_back(-1); 
  // JAMP #1
  color_configs[3].push_back(vec_int(createvector<int>
      (2)(0)(3)(0)(3)(1)(1)(0)(2)(0)));
  jamp_nc_relative_power[3].push_back(0); 
  // JAMP #2
  color_configs[3].push_back(vec_int(createvector<int>
      (3)(0)(1)(0)(3)(2)(1)(0)(2)(0)));
  jamp_nc_relative_power[3].push_back(0); 
  // JAMP #3
  color_configs[3].push_back(vec_int(createvector<int>
      (1)(0)(3)(0)(3)(2)(1)(0)(2)(0)));
  jamp_nc_relative_power[3].push_back(-1); 

  // Color flows of process Process: u u~ > g c c~ WEIGHTED<=3 @130
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[4].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(3)(2)(2)(0)(0)(3)));
  jamp_nc_relative_power[4].push_back(-1); 
  // JAMP #1
  color_configs[4].push_back(vec_int(createvector<int>
      (3)(0)(0)(1)(3)(2)(2)(0)(0)(1)));
  jamp_nc_relative_power[4].push_back(0); 
  // JAMP #2
  color_configs[4].push_back(vec_int(createvector<int>
      (3)(0)(0)(1)(3)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[4].push_back(-1); 
  // JAMP #3
  color_configs[4].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(3)(1)(2)(0)(0)(3)));
  jamp_nc_relative_power[4].push_back(0); 

  // Color flows of process Process: u c~ > g u c~ WEIGHTED<=3 @130
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[5].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(3)(2)(2)(0)(0)(3)));
  jamp_nc_relative_power[5].push_back(0); 
  // JAMP #1
  color_configs[5].push_back(vec_int(createvector<int>
      (3)(0)(0)(1)(3)(2)(2)(0)(0)(1)));
  jamp_nc_relative_power[5].push_back(-1); 
  // JAMP #2
  color_configs[5].push_back(vec_int(createvector<int>
      (3)(0)(0)(1)(3)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[5].push_back(0); 
  // JAMP #3
  color_configs[5].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(3)(1)(2)(0)(0)(3)));
  jamp_nc_relative_power[5].push_back(-1); 

  // Color flows of process Process: u~ c~ > g u~ c~ WEIGHTED<=3 @130
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[6].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(3)(2)(0)(1)(0)(3)));
  jamp_nc_relative_power[6].push_back(-1); 
  // JAMP #1
  color_configs[6].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(3)(2)(0)(3)(0)(1)));
  jamp_nc_relative_power[6].push_back(0); 
  // JAMP #2
  color_configs[6].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(3)(1)(0)(2)(0)(3)));
  jamp_nc_relative_power[6].push_back(0); 
  // JAMP #3
  color_configs[6].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(3)(1)(0)(3)(0)(2)));
  jamp_nc_relative_power[6].push_back(-1); 
}

//--------------------------------------------------------------------------
// Destructor.
PY8MEs_R130_P6_sm_qq_gqq::~PY8MEs_R130_P6_sm_qq_gqq() 
{
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    delete[] p[i]; 
    p[i] = NULL; 
  }
}

//--------------------------------------------------------------------------
// Invert the permutation mapping
vector<int> PY8MEs_R130_P6_sm_qq_gqq::invert_mapping(vector<int> mapping) 
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
vector < vec_int > PY8MEs_R130_P6_sm_qq_gqq::getHelicityConfigs(vector<int>
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
vector < vec_int > PY8MEs_R130_P6_sm_qq_gqq::getColorConfigs(int
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
int PY8MEs_R130_P6_sm_qq_gqq::getColorFlowRelativeNCPower(int color_flow_ID,
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
vector<int> PY8MEs_R130_P6_sm_qq_gqq::getHelicityConfigForID(int hel_ID,
    vector<int> permutation)
{
  if (hel_ID < 0 || hel_ID >= ncomb)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'getHelicityConfigForID' of class" << 
    " 'PY8MEs_R130_P6_sm_qq_gqq': Specified helicity ID '" << 
    hel_ID <<  "' cannot be found." << endl; 
    #endif
    throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
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
int PY8MEs_R130_P6_sm_qq_gqq::getHelicityIDForConfig(vector<int> hel_config,
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
      " 'PY8MEs_R130_P6_sm_qq_gqq': Specified helicity" << 
      " configuration cannot be found." << endl; 
      #endif
      throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
    }
  }
  return user_ihel; 
}


//--------------------------------------------------------------------------
// Implements the map Color ID -> Color Config
vector<int> PY8MEs_R130_P6_sm_qq_gqq::getColorConfigForID(int color_ID, int
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
    " 'PY8MEs_R130_P6_sm_qq_gqq': Specified color ID '" << 
    color_ID <<  "' cannot be found." << endl; 
    #endif
    throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
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
int PY8MEs_R130_P6_sm_qq_gqq::getColorIDForConfig(vector<int> color_config, int
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
            " 'PY8MEs_R130_P6_sm_qq_gqq': A color line could " << 
            " not be closed." << endl; 
            #endif
            throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
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
      " 'PY8MEs_R130_P6_sm_qq_gqq': Specified color" << 
      " configuration cannot be found." << endl; 
      #endif
      throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
    }
  }
  return user_icol; 
}

//--------------------------------------------------------------------------
// Returns all result previously computed in SigmaKin
vector < vec_double > PY8MEs_R130_P6_sm_qq_gqq::getAllResults(int
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
double PY8MEs_R130_P6_sm_qq_gqq::getResult(int helicity_ID, int color_ID, int
    specify_proc_ID)
{
  if (helicity_ID < - 1 || helicity_ID >= ncomb)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R130_P6_sm_qq_gqq': Specified helicity ID '" << 
    helicity_ID <<  "' configuration cannot be found." << endl; 
    #endif
    throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
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
    " 'PY8MEs_R130_P6_sm_qq_gqq': Specified color ID '" << 
    color_ID <<  "' configuration cannot be found." << endl; 
    #endif
    throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
  }
  return all_results[chosenProcID][helicity_ID + 1][color_ID + 1]; 
}

//--------------------------------------------------------------------------
// Check for the availability of the requested process and if available,
// If available, this returns the corresponding permutation and Proc_ID to use.
// If not available, this returns a negative Proc_ID.
pair < vector<int> , int > PY8MEs_R130_P6_sm_qq_gqq::static_getPY8ME(vector<int> initial_pdgs, vector<int> final_pdgs, set<int> schannels) 
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
  const int nprocs = 88; 
  const int proc_IDS[nprocs] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
      3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      5, 5, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6};
  const int in_pdgs[nprocs][ninitial] = {{2, 2}, {4, 4}, {1, 1}, {3, 3}, {2,
      -2}, {4, -4}, {1, -1}, {3, -3}, {-2, -2}, {-4, -4}, {-1, -1}, {-3, -3},
      {2, 4}, {2, 1}, {2, 3}, {4, 1}, {4, 3}, {1, 3}, {2, -2}, {2, -2}, {2,
      -2}, {4, -4}, {4, -4}, {4, -4}, {1, -1}, {1, -1}, {1, -1}, {3, -3}, {3,
      -3}, {3, -3}, {2, -4}, {2, -1}, {2, -3}, {4, -2}, {4, -1}, {4, -3}, {1,
      -2}, {1, -4}, {1, -3}, {3, -2}, {3, -4}, {3, -1}, {-2, -4}, {-2, -1},
      {-2, -3}, {-4, -1}, {-4, -3}, {-1, -3}, {-2, 2}, {-4, 4}, {-1, 1}, {-3,
      3}, {4, 2}, {1, 2}, {3, 2}, {1, 4}, {3, 4}, {3, 1}, {-2, 2}, {-2, 2},
      {-2, 2}, {-4, 4}, {-4, 4}, {-4, 4}, {-1, 1}, {-1, 1}, {-1, 1}, {-3, 3},
      {-3, 3}, {-3, 3}, {-4, 2}, {-1, 2}, {-3, 2}, {-2, 4}, {-1, 4}, {-3, 4},
      {-2, 1}, {-4, 1}, {-3, 1}, {-2, 3}, {-4, 3}, {-1, 3}, {-4, -2}, {-1, -2},
      {-3, -2}, {-1, -4}, {-3, -4}, {-3, -1}};
  const int out_pdgs[nprocs][nexternal - ninitial] = {{21, 2, 2}, {21, 4, 4},
      {21, 1, 1}, {21, 3, 3}, {21, 2, -2}, {21, 4, -4}, {21, 1, -1}, {21, 3,
      -3}, {21, -2, -2}, {21, -4, -4}, {21, -1, -1}, {21, -3, -3}, {21, 2, 4},
      {21, 2, 1}, {21, 2, 3}, {21, 4, 1}, {21, 4, 3}, {21, 1, 3}, {21, 4, -4},
      {21, 1, -1}, {21, 3, -3}, {21, 2, -2}, {21, 1, -1}, {21, 3, -3}, {21, 2,
      -2}, {21, 4, -4}, {21, 3, -3}, {21, 2, -2}, {21, 4, -4}, {21, 1, -1},
      {21, 2, -4}, {21, 2, -1}, {21, 2, -3}, {21, 4, -2}, {21, 4, -1}, {21, 4,
      -3}, {21, 1, -2}, {21, 1, -4}, {21, 1, -3}, {21, 3, -2}, {21, 3, -4},
      {21, 3, -1}, {21, -2, -4}, {21, -2, -1}, {21, -2, -3}, {21, -4, -1}, {21,
      -4, -3}, {21, -1, -3}, {21, 2, -2}, {21, 4, -4}, {21, 1, -1}, {21, 3,
      -3}, {21, 2, 4}, {21, 2, 1}, {21, 2, 3}, {21, 4, 1}, {21, 4, 3}, {21, 1,
      3}, {21, 4, -4}, {21, 1, -1}, {21, 3, -3}, {21, 2, -2}, {21, 1, -1}, {21,
      3, -3}, {21, 2, -2}, {21, 4, -4}, {21, 3, -3}, {21, 2, -2}, {21, 4, -4},
      {21, 1, -1}, {21, 2, -4}, {21, 2, -1}, {21, 2, -3}, {21, 4, -2}, {21, 4,
      -1}, {21, 4, -3}, {21, 1, -2}, {21, 1, -4}, {21, 1, -3}, {21, 3, -2},
      {21, 3, -4}, {21, 3, -1}, {21, -2, -4}, {21, -2, -1}, {21, -2, -3}, {21,
      -4, -1}, {21, -4, -3}, {21, -1, -3}};

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
void PY8MEs_R130_P6_sm_qq_gqq::setMomenta(vector < vec_double > momenta_picked)
{
  if (momenta_picked.size() != nexternal)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setMomenta' of class" << 
    " 'PY8MEs_R130_P6_sm_qq_gqq': Incorrect number of" << 
    " momenta specified." << endl; 
    #endif
    throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
  }
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    if (momenta_picked[i].size() != 4)
    {
      #ifdef DEBUG
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R130_P6_sm_qq_gqq': Incorrect number of" << 
      " momenta components specified." << endl; 
      #endif
      throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
    }
    if (momenta_picked[i][0] < 0.0)
    {
      #ifdef DEBUG
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R130_P6_sm_qq_gqq': A momentum was specified" << 
      " with negative energy. Check conventions." << endl; 
      #endif
      throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
    }
    for (unsigned int j = 0; j < 4; j++ )
    {
      p[i][j] = momenta_picked[i][j]; 
    }
  }
}

//--------------------------------------------------------------------------
// Set color configuration to use. An empty vector means sum over all.
void PY8MEs_R130_P6_sm_qq_gqq::setColors(vector<int> colors_picked)
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
    " 'PY8MEs_R130_P6_sm_qq_gqq': Incorrect number" << 
    " of colors specified." << endl; 
    #endif
    throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
  }
  user_colors = vector<int> ((2 * nexternal), 0); 
  for(unsigned int i = 0; i < (2 * nexternal); i++ )
  {
    user_colors[i] = colors_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the helicity configuration to use. Am empty vector means sum over all.
void PY8MEs_R130_P6_sm_qq_gqq::setHelicities(vector<int> helicities_picked) 
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
    " 'PY8MEs_R130_P6_sm_qq_gqq': Incorrect number" << 
    " of helicities specified." << endl; 
    #endif
    throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
  }
  user_helicities = vector<int> (nexternal, 0); 
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    user_helicities[i] = helicities_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the permutation to use (will apply to momenta, colors and helicities)
void PY8MEs_R130_P6_sm_qq_gqq::setPermutation(vector<int> perm_picked) 
{
  if (perm_picked.size() != nexternal)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setPermutations' of class" << 
    " 'PY8MEs_R130_P6_sm_qq_gqq': Incorrect number" << 
    " of permutations specified." << endl; 
    #endif
    throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
  }
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    perm[i] = perm_picked[i]; 
  }
}

// Set the proc_ID to use
void PY8MEs_R130_P6_sm_qq_gqq::setProcID(int procID_picked) 
{
  proc_ID = procID_picked; 
}

//--------------------------------------------------------------------------
// Initialize process.

void PY8MEs_R130_P6_sm_qq_gqq::initProc() 
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
  jamp2 = vector < vec_double > (7); 
  jamp2[0] = vector<double> (4, 0.); 
  jamp2[1] = vector<double> (4, 0.); 
  jamp2[2] = vector<double> (4, 0.); 
  jamp2[3] = vector<double> (4, 0.); 
  jamp2[4] = vector<double> (4, 0.); 
  jamp2[5] = vector<double> (4, 0.); 
  jamp2[6] = vector<double> (4, 0.); 
  all_results = vector < vec_vec_double > (7); 
  // The first entry is always the color or helicity avg/summed matrix element.
  all_results[0] = vector < vec_double > (ncomb + 1, vector<double> (4 + 1,
      0.));
  all_results[1] = vector < vec_double > (ncomb + 1, vector<double> (4 + 1,
      0.));
  all_results[2] = vector < vec_double > (ncomb + 1, vector<double> (4 + 1,
      0.));
  all_results[3] = vector < vec_double > (ncomb + 1, vector<double> (4 + 1,
      0.));
  all_results[4] = vector < vec_double > (ncomb + 1, vector<double> (4 + 1,
      0.));
  all_results[5] = vector < vec_double > (ncomb + 1, vector<double> (4 + 1,
      0.));
  all_results[6] = vector < vec_double > (ncomb + 1, vector<double> (4 + 1,
      0.));
}

// Synchronize local variables of the process that depend on the model
// parameters
void PY8MEs_R130_P6_sm_qq_gqq::syncProcModelParams() 
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
void PY8MEs_R130_P6_sm_qq_gqq::setMasses(vec_double external_masses) 
{

  if (external_masses.size() != mME.size())
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setMasses' of class" << 
    " 'PY8MEs_R130_P6_sm_qq_gqq': Incorrect number of" << 
    " masses specified." << endl; 
    #endif
    throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
  }
  for (unsigned int j = 0; j < mME.size(); j++ )
  {
    mME[j] = external_masses[perm[j]]; 
  }
}

//--------------------------------------------------------------------------
// Getter accessing external masses with the correct ordering
vector<double> PY8MEs_R130_P6_sm_qq_gqq::getMasses() 
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
void PY8MEs_R130_P6_sm_qq_gqq::setExternalMassesMode(int mode) 
{
  if (mode != 0 && mode != 1 && mode != 2)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setExternalMassesMode' of class" << 
    " 'PY8MEs_R130_P6_sm_qq_gqq': Incorrect mode selected :" << mode << 
    ". It must be either 0, 1 or 2" << endl; 
    #endif
    throw PY8MEs_R130_P6_sm_qq_gqq_exception; 
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

double PY8MEs_R130_P6_sm_qq_gqq::sigmaKin() 
{
  // Set the parameters which change event by event
  pars->setDependentParameters(); 
  pars->setDependentCouplings(); 
  // Reset color flows
  for(int i = 0; i < 4; i++ )
    jamp2[0][i] = 0.; 
  for(int i = 0; i < 4; i++ )
    jamp2[1][i] = 0.; 
  for(int i = 0; i < 4; i++ )
    jamp2[2][i] = 0.; 
  for(int i = 0; i < 4; i++ )
    jamp2[3][i] = 0.; 
  for(int i = 0; i < 4; i++ )
    jamp2[4][i] = 0.; 
  for(int i = 0; i < 4; i++ )
    jamp2[5][i] = 0.; 
  for(int i = 0; i < 4; i++ )
    jamp2[6][i] = 0.; 

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
    for(int i = 0; i < 4; i++ )
      jamp2[0][i] = 0.; 
    for(int i = 0; i < 4; i++ )
      jamp2[1][i] = 0.; 
    for(int i = 0; i < 4; i++ )
      jamp2[2][i] = 0.; 
    for(int i = 0; i < 4; i++ )
      jamp2[3][i] = 0.; 
    for(int i = 0; i < 4; i++ )
      jamp2[4][i] = 0.; 
    for(int i = 0; i < 4; i++ )
      jamp2[5][i] = 0.; 
    for(int i = 0; i < 4; i++ )
      jamp2[6][i] = 0.; 

    if (proc_ID == 0)
      t = matrix_13uu_guu(); 
    if (proc_ID == 1)
      t = matrix_13uux_guux(); 
    if (proc_ID == 2)
      t = matrix_13uxux_guxux(); 
    if (proc_ID == 3)
      t = matrix_13uc_guc(); 
    if (proc_ID == 4)
      t = matrix_13uux_gccx(); 
    if (proc_ID == 5)
      t = matrix_13ucx_gucx(); 
    if (proc_ID == 6)
      t = matrix_13uxcx_guxcx(); 

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

void PY8MEs_R130_P6_sm_qq_gqq::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  // Calculate all wavefunctions
  ixxxxx(p[perm[0]], mME[0], hel[0], +1, w[0]); 
  ixxxxx(p[perm[1]], mME[1], hel[1], +1, w[1]); 
  vxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  oxxxxx(p[perm[3]], mME[3], hel[3], +1, w[3]); 
  oxxxxx(p[perm[4]], mME[4], hel[4], +1, w[4]); 
  FFV1_2(w[0], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[5]); 
  FFV1P0_3(w[1], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[6]); 
  FFV1P0_3(w[1], w[4], pars->GC_11, pars->ZERO, pars->ZERO, w[7]); 
  FFV1_1(w[3], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[8]); 
  FFV1P0_3(w[0], w[4], pars->GC_11, pars->ZERO, pars->ZERO, w[9]); 
  FFV1_1(w[4], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[10]); 
  FFV1P0_3(w[0], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[11]); 
  FFV1_2(w[1], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[12]); 
  oxxxxx(p[perm[1]], mME[1], hel[1], -1, w[13]); 
  ixxxxx(p[perm[4]], mME[4], hel[4], -1, w[14]); 
  FFV1P0_3(w[14], w[13], pars->GC_11, pars->ZERO, pars->ZERO, w[15]); 
  FFV1P0_3(w[14], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[16]); 
  FFV1_1(w[13], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[17]); 
  FFV1P0_3(w[0], w[13], pars->GC_11, pars->ZERO, pars->ZERO, w[18]); 
  FFV1_2(w[14], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[19]); 
  oxxxxx(p[perm[0]], mME[0], hel[0], -1, w[20]); 
  ixxxxx(p[perm[3]], mME[3], hel[3], -1, w[21]); 
  FFV1_2(w[21], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[22]); 
  FFV1P0_3(w[14], w[20], pars->GC_11, pars->ZERO, pars->ZERO, w[23]); 
  FFV1_1(w[20], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[24]); 
  FFV1P0_3(w[21], w[13], pars->GC_11, pars->ZERO, pars->ZERO, w[25]); 
  FFV1P0_3(w[21], w[20], pars->GC_11, pars->ZERO, pars->ZERO, w[26]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[5], w[4], w[6], pars->GC_11, amp[0]); 
  FFV1_0(w[5], w[3], w[7], pars->GC_11, amp[1]); 
  FFV1_0(w[1], w[8], w[9], pars->GC_11, amp[2]); 
  FFV1_0(w[0], w[8], w[7], pars->GC_11, amp[3]); 
  FFV1_0(w[1], w[10], w[11], pars->GC_11, amp[4]); 
  FFV1_0(w[0], w[10], w[6], pars->GC_11, amp[5]); 
  FFV1_0(w[12], w[4], w[11], pars->GC_11, amp[6]); 
  FFV1_0(w[12], w[3], w[9], pars->GC_11, amp[7]); 
  VVV1_0(w[2], w[11], w[7], pars->GC_10, amp[8]); 
  VVV1_0(w[2], w[9], w[6], pars->GC_10, amp[9]); 
  FFV1_0(w[5], w[3], w[15], pars->GC_11, amp[10]); 
  FFV1_0(w[5], w[13], w[16], pars->GC_11, amp[11]); 
  FFV1_0(w[14], w[17], w[11], pars->GC_11, amp[12]); 
  FFV1_0(w[0], w[17], w[16], pars->GC_11, amp[13]); 
  FFV1_0(w[14], w[8], w[18], pars->GC_11, amp[14]); 
  FFV1_0(w[0], w[8], w[15], pars->GC_11, amp[15]); 
  FFV1_0(w[19], w[3], w[18], pars->GC_11, amp[16]); 
  FFV1_0(w[19], w[13], w[11], pars->GC_11, amp[17]); 
  VVV1_0(w[2], w[18], w[16], pars->GC_10, amp[18]); 
  VVV1_0(w[2], w[11], w[15], pars->GC_10, amp[19]); 
  FFV1_0(w[22], w[13], w[23], pars->GC_11, amp[20]); 
  FFV1_0(w[22], w[20], w[15], pars->GC_11, amp[21]); 
  FFV1_0(w[14], w[24], w[25], pars->GC_11, amp[22]); 
  FFV1_0(w[21], w[24], w[15], pars->GC_11, amp[23]); 
  FFV1_0(w[14], w[17], w[26], pars->GC_11, amp[24]); 
  FFV1_0(w[21], w[17], w[23], pars->GC_11, amp[25]); 
  FFV1_0(w[19], w[13], w[26], pars->GC_11, amp[26]); 
  FFV1_0(w[19], w[20], w[25], pars->GC_11, amp[27]); 
  VVV1_0(w[2], w[26], w[15], pars->GC_10, amp[28]); 
  VVV1_0(w[2], w[25], w[23], pars->GC_10, amp[29]); 
  FFV1_0(w[5], w[3], w[7], pars->GC_11, amp[30]); 
  FFV1_0(w[0], w[8], w[7], pars->GC_11, amp[31]); 
  FFV1_0(w[1], w[10], w[11], pars->GC_11, amp[32]); 
  FFV1_0(w[12], w[4], w[11], pars->GC_11, amp[33]); 
  VVV1_0(w[2], w[11], w[7], pars->GC_10, amp[34]); 
  FFV1_0(w[5], w[13], w[16], pars->GC_11, amp[35]); 
  FFV1_0(w[0], w[17], w[16], pars->GC_11, amp[36]); 
  FFV1_0(w[14], w[8], w[18], pars->GC_11, amp[37]); 
  FFV1_0(w[19], w[3], w[18], pars->GC_11, amp[38]); 
  VVV1_0(w[2], w[18], w[16], pars->GC_10, amp[39]); 
  FFV1_0(w[5], w[3], w[15], pars->GC_11, amp[40]); 
  FFV1_0(w[0], w[8], w[15], pars->GC_11, amp[41]); 
  FFV1_0(w[14], w[17], w[11], pars->GC_11, amp[42]); 
  FFV1_0(w[19], w[13], w[11], pars->GC_11, amp[43]); 
  VVV1_0(w[2], w[11], w[15], pars->GC_10, amp[44]); 
  FFV1_0(w[22], w[20], w[15], pars->GC_11, amp[45]); 
  FFV1_0(w[21], w[24], w[15], pars->GC_11, amp[46]); 
  FFV1_0(w[14], w[17], w[26], pars->GC_11, amp[47]); 
  FFV1_0(w[19], w[13], w[26], pars->GC_11, amp[48]); 
  VVV1_0(w[2], w[26], w[15], pars->GC_10, amp[49]); 


}
double PY8MEs_R130_P6_sm_qq_gqq::matrix_13uu_guu() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 10;
  const int ncolor = 4; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 4, 4, 0}, {4, 12, 0, 4}, {4,
      0, 12, 4}, {0, 4, 4, 12}};

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[0] + 1./3. * amp[1] + amp[2] + 1./3. * amp[3] -
      Complex<double> (0, 1) * amp[9]);
  jamp[1] = +1./2. * (-1./3. * amp[2] - amp[3] - amp[6] - 1./3. * amp[7] -
      Complex<double> (0, 1) * amp[8]);
  jamp[2] = +1./2. * (-1./3. * amp[0] - amp[1] - amp[4] - 1./3. * amp[5] +
      Complex<double> (0, 1) * amp[8]);
  jamp[3] = +1./2. * (+1./3. * amp[4] + amp[5] + 1./3. * amp[6] + amp[7] +
      Complex<double> (0, 1) * amp[9]);

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

double PY8MEs_R130_P6_sm_qq_gqq::matrix_13uux_guux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 10;
  const int ncolor = 4; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 4, 0, 4}, {4, 12, 4, 0}, {0,
      4, 12, 4}, {4, 0, 4, 12}};

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[14] + amp[15] + 1./3. * amp[16] + amp[17] +
      Complex<double> (0, 1) * amp[19]);
  jamp[1] = +1./2. * (-1./3. * amp[10] - amp[11] - amp[14] - 1./3. * amp[15] +
      Complex<double> (0, 1) * amp[18]);
  jamp[2] = +1./2. * (+amp[10] + 1./3. * amp[11] + amp[12] + 1./3. * amp[13] -
      Complex<double> (0, 1) * amp[19]);
  jamp[3] = +1./2. * (-1./3. * amp[12] - amp[13] - amp[16] - 1./3. * amp[17] -
      Complex<double> (0, 1) * amp[18]);

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
    jamp2[1][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R130_P6_sm_qq_gqq::matrix_13uxux_guxux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 10;
  const int ncolor = 4; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 4, 4, 0}, {4, 12, 0, 4}, {4,
      0, 12, 4}, {0, 4, 4, 12}};

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[24] + amp[25] + 1./3. * amp[26] + amp[27] +
      Complex<double> (0, 1) * amp[29]);
  jamp[1] = +1./2. * (-1./3. * amp[20] - amp[21] - amp[24] - 1./3. * amp[25] +
      Complex<double> (0, 1) * amp[28]);
  jamp[2] = +1./2. * (-1./3. * amp[22] - amp[23] - amp[26] - 1./3. * amp[27] -
      Complex<double> (0, 1) * amp[28]);
  jamp[3] = +1./2. * (+amp[20] + 1./3. * amp[21] + amp[22] + 1./3. * amp[23] -
      Complex<double> (0, 1) * amp[29]);

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
    jamp2[2][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R130_P6_sm_qq_gqq::matrix_13uc_guc() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 5;
  const int ncolor = 4; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 4, 4, 0}, {4, 12, 0, 4}, {4,
      0, 12, 4}, {0, 4, 4, 12}};

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[30] + 1./3. * amp[31]); 
  jamp[1] = +1./2. * (-amp[31] - amp[33] - Complex<double> (0, 1) * amp[34]); 
  jamp[2] = +1./2. * (-amp[30] - amp[32] + Complex<double> (0, 1) * amp[34]); 
  jamp[3] = +1./2. * (+1./3. * amp[32] + 1./3. * amp[33]); 

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
    jamp2[3][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R130_P6_sm_qq_gqq::matrix_13uux_gccx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 5;
  const int ncolor = 4; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 4, 0, 4}, {4, 12, 4, 0}, {0,
      4, 12, 4}, {4, 0, 4, 12}};

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[37] + 1./3. * amp[38]); 
  jamp[1] = +1./2. * (-amp[35] - amp[37] + Complex<double> (0, 1) * amp[39]); 
  jamp[2] = +1./2. * (+1./3. * amp[35] + 1./3. * amp[36]); 
  jamp[3] = +1./2. * (-amp[36] - amp[38] - Complex<double> (0, 1) * amp[39]); 

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
    jamp2[4][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R130_P6_sm_qq_gqq::matrix_13ucx_gucx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 5;
  const int ncolor = 4; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 4, 0, 4}, {4, 12, 4, 0}, {0,
      4, 12, 4}, {4, 0, 4, 12}};

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[41] + amp[43] + Complex<double> (0, 1) * amp[44]); 
  jamp[1] = +1./2. * (-1./3. * amp[40] - 1./3. * amp[41]); 
  jamp[2] = +1./2. * (+amp[40] + amp[42] - Complex<double> (0, 1) * amp[44]); 
  jamp[3] = +1./2. * (-1./3. * amp[42] - 1./3. * amp[43]); 

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
    jamp2[5][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R130_P6_sm_qq_gqq::matrix_13uxcx_guxcx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 5;
  const int ncolor = 4; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 4, 4, 0}, {4, 12, 0, 4}, {4,
      0, 12, 4}, {0, 4, 4, 12}};

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[47] + 1./3. * amp[48]); 
  jamp[1] = +1./2. * (-amp[45] - amp[47] + Complex<double> (0, 1) * amp[49]); 
  jamp[2] = +1./2. * (-amp[46] - amp[48] - Complex<double> (0, 1) * amp[49]); 
  jamp[3] = +1./2. * (+1./3. * amp[45] + 1./3. * amp[46]); 

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
    jamp2[6][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}


}  // end namespace PY8MEs_namespace

