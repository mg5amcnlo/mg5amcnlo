//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "PY8MEs_R2_P4_sm_lq_lgq.h"
#include "HelAmps_sm.h"

using namespace Pythia8_sm; 

namespace PY8MEs_namespace 
{
//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: e- u > e- g u WEIGHTED<=5 @2
// Process: e- c > e- g c WEIGHTED<=5 @2
// Process: mu- u > mu- g u WEIGHTED<=5 @2
// Process: mu- c > mu- g c WEIGHTED<=5 @2
// Process: e- d > e- g d WEIGHTED<=5 @2
// Process: e- s > e- g s WEIGHTED<=5 @2
// Process: mu- d > mu- g d WEIGHTED<=5 @2
// Process: mu- s > mu- g s WEIGHTED<=5 @2
// Process: e- u~ > e- g u~ WEIGHTED<=5 @2
// Process: e- c~ > e- g c~ WEIGHTED<=5 @2
// Process: mu- u~ > mu- g u~ WEIGHTED<=5 @2
// Process: mu- c~ > mu- g c~ WEIGHTED<=5 @2
// Process: e- d~ > e- g d~ WEIGHTED<=5 @2
// Process: e- s~ > e- g s~ WEIGHTED<=5 @2
// Process: mu- d~ > mu- g d~ WEIGHTED<=5 @2
// Process: mu- s~ > mu- g s~ WEIGHTED<=5 @2
// Process: e+ u > e+ g u WEIGHTED<=5 @2
// Process: e+ c > e+ g c WEIGHTED<=5 @2
// Process: mu+ u > mu+ g u WEIGHTED<=5 @2
// Process: mu+ c > mu+ g c WEIGHTED<=5 @2
// Process: e+ d > e+ g d WEIGHTED<=5 @2
// Process: e+ s > e+ g s WEIGHTED<=5 @2
// Process: mu+ d > mu+ g d WEIGHTED<=5 @2
// Process: mu+ s > mu+ g s WEIGHTED<=5 @2
// Process: e+ u~ > e+ g u~ WEIGHTED<=5 @2
// Process: e+ c~ > e+ g c~ WEIGHTED<=5 @2
// Process: mu+ u~ > mu+ g u~ WEIGHTED<=5 @2
// Process: mu+ c~ > mu+ g c~ WEIGHTED<=5 @2
// Process: e+ d~ > e+ g d~ WEIGHTED<=5 @2
// Process: e+ s~ > e+ g s~ WEIGHTED<=5 @2
// Process: mu+ d~ > mu+ g d~ WEIGHTED<=5 @2
// Process: mu+ s~ > mu+ g s~ WEIGHTED<=5 @2
// Process: e- u > ve g d WEIGHTED<=5 @2
// Process: e- c > ve g s WEIGHTED<=5 @2
// Process: mu- u > vm g d WEIGHTED<=5 @2
// Process: mu- c > vm g s WEIGHTED<=5 @2
// Process: ve d > e- g u WEIGHTED<=5 @2
// Process: ve s > e- g c WEIGHTED<=5 @2
// Process: vm d > mu- g u WEIGHTED<=5 @2
// Process: vm s > mu- g c WEIGHTED<=5 @2
// Process: e- d~ > ve g u~ WEIGHTED<=5 @2
// Process: e- s~ > ve g c~ WEIGHTED<=5 @2
// Process: mu- d~ > vm g u~ WEIGHTED<=5 @2
// Process: mu- s~ > vm g c~ WEIGHTED<=5 @2
// Process: ve u~ > e- g d~ WEIGHTED<=5 @2
// Process: ve c~ > e- g s~ WEIGHTED<=5 @2
// Process: vm u~ > mu- g d~ WEIGHTED<=5 @2
// Process: vm c~ > mu- g s~ WEIGHTED<=5 @2
// Process: ve u > ve g u WEIGHTED<=5 @2
// Process: ve c > ve g c WEIGHTED<=5 @2
// Process: vm u > vm g u WEIGHTED<=5 @2
// Process: vm c > vm g c WEIGHTED<=5 @2
// Process: vt u > vt g u WEIGHTED<=5 @2
// Process: vt c > vt g c WEIGHTED<=5 @2
// Process: ve d > ve g d WEIGHTED<=5 @2
// Process: ve s > ve g s WEIGHTED<=5 @2
// Process: vm d > vm g d WEIGHTED<=5 @2
// Process: vm s > vm g s WEIGHTED<=5 @2
// Process: vt d > vt g d WEIGHTED<=5 @2
// Process: vt s > vt g s WEIGHTED<=5 @2
// Process: ve u~ > ve g u~ WEIGHTED<=5 @2
// Process: ve c~ > ve g c~ WEIGHTED<=5 @2
// Process: vm u~ > vm g u~ WEIGHTED<=5 @2
// Process: vm c~ > vm g c~ WEIGHTED<=5 @2
// Process: vt u~ > vt g u~ WEIGHTED<=5 @2
// Process: vt c~ > vt g c~ WEIGHTED<=5 @2
// Process: ve d~ > ve g d~ WEIGHTED<=5 @2
// Process: ve s~ > ve g s~ WEIGHTED<=5 @2
// Process: vm d~ > vm g d~ WEIGHTED<=5 @2
// Process: vm s~ > vm g s~ WEIGHTED<=5 @2
// Process: vt d~ > vt g d~ WEIGHTED<=5 @2
// Process: vt s~ > vt g s~ WEIGHTED<=5 @2
// Process: e+ d > ve~ g u WEIGHTED<=5 @2
// Process: e+ s > ve~ g c WEIGHTED<=5 @2
// Process: mu+ d > vm~ g u WEIGHTED<=5 @2
// Process: mu+ s > vm~ g c WEIGHTED<=5 @2
// Process: ve~ u > e+ g d WEIGHTED<=5 @2
// Process: ve~ c > e+ g s WEIGHTED<=5 @2
// Process: vm~ u > mu+ g d WEIGHTED<=5 @2
// Process: vm~ c > mu+ g s WEIGHTED<=5 @2
// Process: e+ u~ > ve~ g d~ WEIGHTED<=5 @2
// Process: e+ c~ > ve~ g s~ WEIGHTED<=5 @2
// Process: mu+ u~ > vm~ g d~ WEIGHTED<=5 @2
// Process: mu+ c~ > vm~ g s~ WEIGHTED<=5 @2
// Process: ve~ d~ > e+ g u~ WEIGHTED<=5 @2
// Process: ve~ s~ > e+ g c~ WEIGHTED<=5 @2
// Process: vm~ d~ > mu+ g u~ WEIGHTED<=5 @2
// Process: vm~ s~ > mu+ g c~ WEIGHTED<=5 @2
// Process: ve~ u > ve~ g u WEIGHTED<=5 @2
// Process: ve~ c > ve~ g c WEIGHTED<=5 @2
// Process: vm~ u > vm~ g u WEIGHTED<=5 @2
// Process: vm~ c > vm~ g c WEIGHTED<=5 @2
// Process: vt~ u > vt~ g u WEIGHTED<=5 @2
// Process: vt~ c > vt~ g c WEIGHTED<=5 @2
// Process: ve~ d > ve~ g d WEIGHTED<=5 @2
// Process: ve~ s > ve~ g s WEIGHTED<=5 @2
// Process: vm~ d > vm~ g d WEIGHTED<=5 @2
// Process: vm~ s > vm~ g s WEIGHTED<=5 @2
// Process: vt~ d > vt~ g d WEIGHTED<=5 @2
// Process: vt~ s > vt~ g s WEIGHTED<=5 @2
// Process: ve~ u~ > ve~ g u~ WEIGHTED<=5 @2
// Process: ve~ c~ > ve~ g c~ WEIGHTED<=5 @2
// Process: vm~ u~ > vm~ g u~ WEIGHTED<=5 @2
// Process: vm~ c~ > vm~ g c~ WEIGHTED<=5 @2
// Process: vt~ u~ > vt~ g u~ WEIGHTED<=5 @2
// Process: vt~ c~ > vt~ g c~ WEIGHTED<=5 @2
// Process: ve~ d~ > ve~ g d~ WEIGHTED<=5 @2
// Process: ve~ s~ > ve~ g s~ WEIGHTED<=5 @2
// Process: vm~ d~ > vm~ g d~ WEIGHTED<=5 @2
// Process: vm~ s~ > vm~ g s~ WEIGHTED<=5 @2
// Process: vt~ d~ > vt~ g d~ WEIGHTED<=5 @2
// Process: vt~ s~ > vt~ g s~ WEIGHTED<=5 @2

// Exception class
class PY8MEs_R2_P4_sm_lq_lgqException : public exception
{
  virtual const char * what() const throw()
  {
    return "Exception in class 'PY8MEs_R2_P4_sm_lq_lgq'."; 
  }
}
PY8MEs_R2_P4_sm_lq_lgq_exception; 

int PY8MEs_R2_P4_sm_lq_lgq::helicities[ncomb][nexternal] = {{-1, -1, -1, -1,
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
void PY8MEs_R2_P4_sm_lq_lgq::initColorConfigs() 
{
  color_configs = vector < vec_vec_int > (); 
  jamp_nc_relative_power = vector < vec_int > (); 

  // Color flows of process Process: e- u > e- g u WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp00[2 * nexternal] = {0, 0, 2, 0, 0, 0, 2, 1, 1, 0}; 
  color_configs[0].push_back(vec_int(jamp00, jamp00 + (2 * nexternal))); 
  jamp_nc_relative_power[0].push_back(0); 

  // Color flows of process Process: e- d > e- g d WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp10[2 * nexternal] = {0, 0, 2, 0, 0, 0, 2, 1, 1, 0}; 
  color_configs[1].push_back(vec_int(jamp10, jamp10 + (2 * nexternal))); 
  jamp_nc_relative_power[1].push_back(0); 

  // Color flows of process Process: e- u~ > e- g u~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp20[2 * nexternal] = {0, 0, 0, 1, 0, 0, 2, 1, 0, 2}; 
  color_configs[2].push_back(vec_int(jamp20, jamp20 + (2 * nexternal))); 
  jamp_nc_relative_power[2].push_back(0); 

  // Color flows of process Process: e- d~ > e- g d~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp30[2 * nexternal] = {0, 0, 0, 1, 0, 0, 2, 1, 0, 2}; 
  color_configs[3].push_back(vec_int(jamp30, jamp30 + (2 * nexternal))); 
  jamp_nc_relative_power[3].push_back(0); 

  // Color flows of process Process: e+ u > e+ g u WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp40[2 * nexternal] = {0, 0, 2, 0, 0, 0, 2, 1, 1, 0}; 
  color_configs[4].push_back(vec_int(jamp40, jamp40 + (2 * nexternal))); 
  jamp_nc_relative_power[4].push_back(0); 

  // Color flows of process Process: e+ d > e+ g d WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp50[2 * nexternal] = {0, 0, 2, 0, 0, 0, 2, 1, 1, 0}; 
  color_configs[5].push_back(vec_int(jamp50, jamp50 + (2 * nexternal))); 
  jamp_nc_relative_power[5].push_back(0); 

  // Color flows of process Process: e+ u~ > e+ g u~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp60[2 * nexternal] = {0, 0, 0, 1, 0, 0, 2, 1, 0, 2}; 
  color_configs[6].push_back(vec_int(jamp60, jamp60 + (2 * nexternal))); 
  jamp_nc_relative_power[6].push_back(0); 

  // Color flows of process Process: e+ d~ > e+ g d~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp70[2 * nexternal] = {0, 0, 0, 1, 0, 0, 2, 1, 0, 2}; 
  color_configs[7].push_back(vec_int(jamp70, jamp70 + (2 * nexternal))); 
  jamp_nc_relative_power[7].push_back(0); 

  // Color flows of process Process: e- u > ve g d WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp80[2 * nexternal] = {0, 0, 2, 0, 0, 0, 2, 1, 1, 0}; 
  color_configs[8].push_back(vec_int(jamp80, jamp80 + (2 * nexternal))); 
  jamp_nc_relative_power[8].push_back(0); 

  // Color flows of process Process: e- d~ > ve g u~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp90[2 * nexternal] = {0, 0, 0, 1, 0, 0, 2, 1, 0, 2}; 
  color_configs[9].push_back(vec_int(jamp90, jamp90 + (2 * nexternal))); 
  jamp_nc_relative_power[9].push_back(0); 

  // Color flows of process Process: ve u > ve g u WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp100[2 * nexternal] = {0, 0, 2, 0, 0, 0, 2, 1, 1, 0}; 
  color_configs[10].push_back(vec_int(jamp100, jamp100 + (2 * nexternal))); 
  jamp_nc_relative_power[10].push_back(0); 

  // Color flows of process Process: ve d > ve g d WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp110[2 * nexternal] = {0, 0, 2, 0, 0, 0, 2, 1, 1, 0}; 
  color_configs[11].push_back(vec_int(jamp110, jamp110 + (2 * nexternal))); 
  jamp_nc_relative_power[11].push_back(0); 

  // Color flows of process Process: ve u~ > ve g u~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp120[2 * nexternal] = {0, 0, 0, 1, 0, 0, 2, 1, 0, 2}; 
  color_configs[12].push_back(vec_int(jamp120, jamp120 + (2 * nexternal))); 
  jamp_nc_relative_power[12].push_back(0); 

  // Color flows of process Process: ve d~ > ve g d~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp130[2 * nexternal] = {0, 0, 0, 1, 0, 0, 2, 1, 0, 2}; 
  color_configs[13].push_back(vec_int(jamp130, jamp130 + (2 * nexternal))); 
  jamp_nc_relative_power[13].push_back(0); 

  // Color flows of process Process: e+ d > ve~ g u WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp140[2 * nexternal] = {0, 0, 2, 0, 0, 0, 2, 1, 1, 0}; 
  color_configs[14].push_back(vec_int(jamp140, jamp140 + (2 * nexternal))); 
  jamp_nc_relative_power[14].push_back(0); 

  // Color flows of process Process: e+ u~ > ve~ g d~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp150[2 * nexternal] = {0, 0, 0, 1, 0, 0, 2, 1, 0, 2}; 
  color_configs[15].push_back(vec_int(jamp150, jamp150 + (2 * nexternal))); 
  jamp_nc_relative_power[15].push_back(0); 

  // Color flows of process Process: ve~ u > ve~ g u WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp160[2 * nexternal] = {0, 0, 2, 0, 0, 0, 2, 1, 1, 0}; 
  color_configs[16].push_back(vec_int(jamp160, jamp160 + (2 * nexternal))); 
  jamp_nc_relative_power[16].push_back(0); 

  // Color flows of process Process: ve~ d > ve~ g d WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp170[2 * nexternal] = {0, 0, 2, 0, 0, 0, 2, 1, 1, 0}; 
  color_configs[17].push_back(vec_int(jamp170, jamp170 + (2 * nexternal))); 
  jamp_nc_relative_power[17].push_back(0); 

  // Color flows of process Process: ve~ u~ > ve~ g u~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp180[2 * nexternal] = {0, 0, 0, 1, 0, 0, 2, 1, 0, 2}; 
  color_configs[18].push_back(vec_int(jamp180, jamp180 + (2 * nexternal))); 
  jamp_nc_relative_power[18].push_back(0); 

  // Color flows of process Process: ve~ d~ > ve~ g d~ WEIGHTED<=5 @2
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  static const int jamp190[2 * nexternal] = {0, 0, 0, 1, 0, 0, 2, 1, 0, 2}; 
  color_configs[19].push_back(vec_int(jamp190, jamp190 + (2 * nexternal))); 
  jamp_nc_relative_power[19].push_back(0); 
}

//--------------------------------------------------------------------------
// Destructor.
PY8MEs_R2_P4_sm_lq_lgq::~PY8MEs_R2_P4_sm_lq_lgq() 
{
  for(int i = 0; i < nexternal; i++ )
  {
    delete[] p[i]; 
    p[i] = NULL; 
  }
}

//--------------------------------------------------------------------------
// Invert the permutation mapping
vector<int> PY8MEs_R2_P4_sm_lq_lgq::invert_mapping(vector<int> mapping) 
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
vector < vec_int > PY8MEs_R2_P4_sm_lq_lgq::getHelicityConfigs(vector<int>
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
vector < vec_int > PY8MEs_R2_P4_sm_lq_lgq::getColorConfigs(int specify_proc_ID,
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
int PY8MEs_R2_P4_sm_lq_lgq::getColorFlowRelativeNCPower(int color_flow_ID, int
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
vector<int> PY8MEs_R2_P4_sm_lq_lgq::getHelicityConfigForID(int hel_ID,
    vector<int> permutation)
{
  if (hel_ID < 0 || hel_ID >= ncomb)
  {
    cerr <<  "Error in function 'getHelicityConfigForID' of class" << 
    " 'PY8MEs_R2_P4_sm_lq_lgq': Specified helicity ID '" << 
    hel_ID <<  "' cannot be found." << endl; 
    throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
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
int PY8MEs_R2_P4_sm_lq_lgq::getHelicityIDForConfig(vector<int> hel_config,
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
      " 'PY8MEs_R2_P4_sm_lq_lgq': Specified helicity" << 
      " configuration cannot be found." << endl; 
      throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
    }
  }
  return user_ihel; 
}


//--------------------------------------------------------------------------
// Implements the map Color ID -> Color Config
vector<int> PY8MEs_R2_P4_sm_lq_lgq::getColorConfigForID(int color_ID, int
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
    " 'PY8MEs_R2_P4_sm_lq_lgq': Specified color ID '" << 
    color_ID <<  "' cannot be found." << endl; 
    throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
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
int PY8MEs_R2_P4_sm_lq_lgq::getColorIDForConfig(vector<int> color_config, int
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
            " 'PY8MEs_R2_P4_sm_lq_lgq': A color line could " << 
            " not be closed." << endl; 
            throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
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
      " 'PY8MEs_R2_P4_sm_lq_lgq': Specified color" << 
      " configuration cannot be found." << endl; 
      throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
    }
  }
  return user_icol; 
}

//--------------------------------------------------------------------------
// Returns all result previously computed in SigmaKin
vector < vec_double > PY8MEs_R2_P4_sm_lq_lgq::getAllResults(int
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
double PY8MEs_R2_P4_sm_lq_lgq::getResult(int helicity_ID, int color_ID, int
    specify_proc_ID)
{
  if (helicity_ID < - 1 || helicity_ID >= ncomb)
  {
    cerr <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R2_P4_sm_lq_lgq': Specified helicity ID '" << 
    helicity_ID <<  "' configuration cannot be found." << endl; 
    throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
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
    " 'PY8MEs_R2_P4_sm_lq_lgq': Specified color ID '" << 
    color_ID <<  "' configuration cannot be found." << endl; 
    throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
  }
  return all_results[chosenProcID][helicity_ID + 1][color_ID + 1]; 
}

//--------------------------------------------------------------------------
// Check for the availability of the requested process and if available,
// If available, this returns the corresponding permutation and Proc_ID to use.
// If not available, this returns a negative Proc_ID.
pair < vector<int> , int > PY8MEs_R2_P4_sm_lq_lgq::static_getPY8ME(vector<int>
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
  const int nprocs = 112; 
  const int proc_IDS[nprocs] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
      4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,
      9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,
      12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14,
      14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17,
      17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19};
  const int in_pdgs[nprocs][ninitial] = {{11, 2}, {11, 4}, {13, 2}, {13, 4},
      {11, 1}, {11, 3}, {13, 1}, {13, 3}, {11, -2}, {11, -4}, {13, -2}, {13,
      -4}, {11, -1}, {11, -3}, {13, -1}, {13, -3}, {-11, 2}, {-11, 4}, {-13,
      2}, {-13, 4}, {-11, 1}, {-11, 3}, {-13, 1}, {-13, 3}, {-11, -2}, {-11,
      -4}, {-13, -2}, {-13, -4}, {-11, -1}, {-11, -3}, {-13, -1}, {-13, -3},
      {11, 2}, {11, 4}, {13, 2}, {13, 4}, {12, 1}, {12, 3}, {14, 1}, {14, 3},
      {11, -1}, {11, -3}, {13, -1}, {13, -3}, {12, -2}, {12, -4}, {14, -2},
      {14, -4}, {12, 2}, {12, 4}, {14, 2}, {14, 4}, {16, 2}, {16, 4}, {12, 1},
      {12, 3}, {14, 1}, {14, 3}, {16, 1}, {16, 3}, {12, -2}, {12, -4}, {14,
      -2}, {14, -4}, {16, -2}, {16, -4}, {12, -1}, {12, -3}, {14, -1}, {14,
      -3}, {16, -1}, {16, -3}, {-11, 1}, {-11, 3}, {-13, 1}, {-13, 3}, {-12,
      2}, {-12, 4}, {-14, 2}, {-14, 4}, {-11, -2}, {-11, -4}, {-13, -2}, {-13,
      -4}, {-12, -1}, {-12, -3}, {-14, -1}, {-14, -3}, {-12, 2}, {-12, 4},
      {-14, 2}, {-14, 4}, {-16, 2}, {-16, 4}, {-12, 1}, {-12, 3}, {-14, 1},
      {-14, 3}, {-16, 1}, {-16, 3}, {-12, -2}, {-12, -4}, {-14, -2}, {-14, -4},
      {-16, -2}, {-16, -4}, {-12, -1}, {-12, -3}, {-14, -1}, {-14, -3}, {-16,
      -1}, {-16, -3}};
  const int out_pdgs[nprocs][nexternal - ninitial] = {{11, 21, 2}, {11, 21, 4},
      {13, 21, 2}, {13, 21, 4}, {11, 21, 1}, {11, 21, 3}, {13, 21, 1}, {13, 21,
      3}, {11, 21, -2}, {11, 21, -4}, {13, 21, -2}, {13, 21, -4}, {11, 21, -1},
      {11, 21, -3}, {13, 21, -1}, {13, 21, -3}, {-11, 21, 2}, {-11, 21, 4},
      {-13, 21, 2}, {-13, 21, 4}, {-11, 21, 1}, {-11, 21, 3}, {-13, 21, 1},
      {-13, 21, 3}, {-11, 21, -2}, {-11, 21, -4}, {-13, 21, -2}, {-13, 21, -4},
      {-11, 21, -1}, {-11, 21, -3}, {-13, 21, -1}, {-13, 21, -3}, {12, 21, 1},
      {12, 21, 3}, {14, 21, 1}, {14, 21, 3}, {11, 21, 2}, {11, 21, 4}, {13, 21,
      2}, {13, 21, 4}, {12, 21, -2}, {12, 21, -4}, {14, 21, -2}, {14, 21, -4},
      {11, 21, -1}, {11, 21, -3}, {13, 21, -1}, {13, 21, -3}, {12, 21, 2}, {12,
      21, 4}, {14, 21, 2}, {14, 21, 4}, {16, 21, 2}, {16, 21, 4}, {12, 21, 1},
      {12, 21, 3}, {14, 21, 1}, {14, 21, 3}, {16, 21, 1}, {16, 21, 3}, {12, 21,
      -2}, {12, 21, -4}, {14, 21, -2}, {14, 21, -4}, {16, 21, -2}, {16, 21,
      -4}, {12, 21, -1}, {12, 21, -3}, {14, 21, -1}, {14, 21, -3}, {16, 21,
      -1}, {16, 21, -3}, {-12, 21, 2}, {-12, 21, 4}, {-14, 21, 2}, {-14, 21,
      4}, {-11, 21, 1}, {-11, 21, 3}, {-13, 21, 1}, {-13, 21, 3}, {-12, 21,
      -1}, {-12, 21, -3}, {-14, 21, -1}, {-14, 21, -3}, {-11, 21, -2}, {-11,
      21, -4}, {-13, 21, -2}, {-13, 21, -4}, {-12, 21, 2}, {-12, 21, 4}, {-14,
      21, 2}, {-14, 21, 4}, {-16, 21, 2}, {-16, 21, 4}, {-12, 21, 1}, {-12, 21,
      3}, {-14, 21, 1}, {-14, 21, 3}, {-16, 21, 1}, {-16, 21, 3}, {-12, 21,
      -2}, {-12, 21, -4}, {-14, 21, -2}, {-14, 21, -4}, {-16, 21, -2}, {-16,
      21, -4}, {-12, 21, -1}, {-12, 21, -3}, {-14, 21, -1}, {-14, 21, -3},
      {-16, 21, -1}, {-16, 21, -3}};

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
void PY8MEs_R2_P4_sm_lq_lgq::setMomenta(vector < vec_double > momenta_picked)
{
  if (momenta_picked.size() != nexternal)
  {
    cerr <<  "Error in function 'setMomenta' of class" << 
    " 'PY8MEs_R2_P4_sm_lq_lgq': Incorrect number of" << 
    " momenta specified." << endl; 
    throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
  }
  for(int i = 0; i < nexternal; i++ )
  {
    if (momenta_picked[i].size() != 4)
    {
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R2_P4_sm_lq_lgq': Incorrect number of" << 
      " momenta components specified." << endl; 
      throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
    }
    if (momenta_picked[i][0] < 0.0)
    {
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R2_P4_sm_lq_lgq': A momentum was specified" << 
      " with negative energy. Check conventions." << endl; 
      throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
    }
    for (int j = 0; j < 4; j++ )
    {
      p[i][j] = momenta_picked[i][j]; 
    }
  }
}

//--------------------------------------------------------------------------
// Set color configuration to use. An empty vector means sum over all.
void PY8MEs_R2_P4_sm_lq_lgq::setColors(vector<int> colors_picked)
{
  if (colors_picked.size() == 0)
  {
    user_colors = vector<int> (); 
    return; 
  }
  if (colors_picked.size() != (2 * nexternal))
  {
    cerr <<  "Error in function 'setColors' of class" << 
    " 'PY8MEs_R2_P4_sm_lq_lgq': Incorrect number" << 
    " of colors specified." << endl; 
    throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
  }
  user_colors = vector<int> ((2 * nexternal), 0); 
  for(int i = 0; i < (2 * nexternal); i++ )
  {
    user_colors[i] = colors_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the helicity configuration to use. Am empty vector means sum over all.
void PY8MEs_R2_P4_sm_lq_lgq::setHelicities(vector<int> helicities_picked) 
{
  if (helicities_picked.size() != nexternal)
  {
    if (helicities_picked.size() == 0)
    {
      user_helicities = vector<int> (); 
      return; 
    }
    cerr <<  "Error in function 'setHelicities' of class" << 
    " 'PY8MEs_R2_P4_sm_lq_lgq': Incorrect number" << 
    " of helicities specified." << endl; 
    throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
  }
  user_helicities = vector<int> (nexternal, 0); 
  for(int i = 0; i < nexternal; i++ )
  {
    user_helicities[i] = helicities_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the permutation to use (will apply to momenta, colors and helicities)
void PY8MEs_R2_P4_sm_lq_lgq::setPermutation(vector<int> perm_picked) 
{
  if (perm_picked.size() != nexternal)
  {
    cerr <<  "Error in function 'setPermutations' of class" << 
    " 'PY8MEs_R2_P4_sm_lq_lgq': Incorrect number" << 
    " of permutations specified." << endl; 
    throw PY8MEs_R2_P4_sm_lq_lgq_exception; 
  }
  for(int i = 0; i < nexternal; i++ )
  {
    perm[i] = perm_picked[i]; 
  }
}

// Set the proc_ID to use
void PY8MEs_R2_P4_sm_lq_lgq::setProcID(int procID_picked) 
{
  proc_ID = procID_picked; 
}

//--------------------------------------------------------------------------
// Initialize process.

void PY8MEs_R2_P4_sm_lq_lgq::initProc() 
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
  jamp2 = vector < vec_double > (20); 
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
  jamp2[10] = vector<double> (1, 0.); 
  jamp2[11] = vector<double> (1, 0.); 
  jamp2[12] = vector<double> (1, 0.); 
  jamp2[13] = vector<double> (1, 0.); 
  jamp2[14] = vector<double> (1, 0.); 
  jamp2[15] = vector<double> (1, 0.); 
  jamp2[16] = vector<double> (1, 0.); 
  jamp2[17] = vector<double> (1, 0.); 
  jamp2[18] = vector<double> (1, 0.); 
  jamp2[19] = vector<double> (1, 0.); 
  all_results = vector < vec_vec_double > (20); 
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
  all_results[10] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[11] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[12] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[13] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[14] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[15] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[16] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[17] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[18] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[19] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
}

//--------------------------------------------------------------------------
// Evaluate the squared matrix element.

double PY8MEs_R2_P4_sm_lq_lgq::sigmaKin() 
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
  for(int i = 0; i < 1; i++ )
    jamp2[10][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[11][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[12][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[13][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[14][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[15][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[16][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[17][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[18][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[19][i] = 0.; 

  //pars->printIndependentCouplings();
  //pars->printDependentCouplings();
  //pars->printIndependentParameters();
  //pars->printDependentParameters();

  // Local variables and constants
  const int max_tries = 10; 
  vector < vec_bool > goodhel(nprocesses, vec_bool(ncomb, false)); 
  vec_int ntry(nprocesses, 0); 
  double t = 0.; 
  double result = 0.; 

  // Denominators: spins, colors and identical particles
  const int denom_colors[nprocesses] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3};
  const int denom_hels[nprocesses] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4};
  const int denom_iden[nprocesses] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1};

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
    for(int i = 0; i < 1; i++ )
      jamp2[10][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[11][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[12][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[13][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[14][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[15][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[16][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[17][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[18][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[19][i] = 0.; 

    if (proc_ID == 0)
      t = matrix_2_emu_emgu(); 
    if (proc_ID == 1)
      t = matrix_2_emd_emgd(); 
    if (proc_ID == 2)
      t = matrix_2_emux_emgux(); 
    if (proc_ID == 3)
      t = matrix_2_emdx_emgdx(); 
    if (proc_ID == 4)
      t = matrix_2_epu_epgu(); 
    if (proc_ID == 5)
      t = matrix_2_epd_epgd(); 
    if (proc_ID == 6)
      t = matrix_2_epux_epgux(); 
    if (proc_ID == 7)
      t = matrix_2_epdx_epgdx(); 
    if (proc_ID == 8)
      t = matrix_2_emu_vegd(); 
    if (proc_ID == 9)
      t = matrix_2_emdx_vegux(); 
    if (proc_ID == 10)
      t = matrix_2_veu_vegu(); 
    if (proc_ID == 11)
      t = matrix_2_ved_vegd(); 
    if (proc_ID == 12)
      t = matrix_2_veux_vegux(); 
    if (proc_ID == 13)
      t = matrix_2_vedx_vegdx(); 
    if (proc_ID == 14)
      t = matrix_2_epd_vexgu(); 
    if (proc_ID == 15)
      t = matrix_2_epux_vexgdx(); 
    if (proc_ID == 16)
      t = matrix_2_vexu_vexgu(); 
    if (proc_ID == 17)
      t = matrix_2_vexd_vexgd(); 
    if (proc_ID == 18)
      t = matrix_2_vexux_vexgux(); 
    if (proc_ID == 19)
      t = matrix_2_vexdx_vexgdx(); 

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

void PY8MEs_R2_P4_sm_lq_lgq::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  // Calculate all wavefunctions
  ixxxxx(p[perm[0]], mME[0], hel[0], +1, w[0]); 
  ixxxxx(p[perm[1]], mME[1], hel[1], +1, w[1]); 
  oxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  vxxxxx(p[perm[3]], mME[3], hel[3], +1, w[3]); 
  oxxxxx(p[perm[4]], mME[4], hel[4], +1, w[4]); 
  FFV1P0_3(w[0], w[2], pars->GC_3, pars->ZERO, pars->ZERO, w[5]); 
  FFV1_1(w[4], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[6]); 
  FFV2_4_3(w[0], w[2], pars->GC_50, pars->GC_59, pars->mdl_MZ, pars->mdl_WZ,
      w[7]);
  FFV1_2(w[1], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[8]); 
  oxxxxx(p[perm[1]], mME[1], hel[1], -1, w[9]); 
  ixxxxx(p[perm[4]], mME[4], hel[4], -1, w[10]); 
  FFV1_1(w[9], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[11]); 
  FFV1_2(w[10], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[12]); 
  oxxxxx(p[perm[0]], mME[0], hel[0], -1, w[13]); 
  ixxxxx(p[perm[2]], mME[2], hel[2], -1, w[14]); 
  FFV1P0_3(w[14], w[13], pars->GC_3, pars->ZERO, pars->ZERO, w[15]); 
  FFV2_4_3(w[14], w[13], pars->GC_50, pars->GC_59, pars->mdl_MZ, pars->mdl_WZ,
      w[16]);
  FFV2_3(w[0], w[2], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[17]); 
  FFV2_3(w[0], w[2], pars->GC_62, pars->mdl_MZ, pars->mdl_WZ, w[18]); 
  FFV2_3(w[14], w[13], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[19]); 
  FFV2_3(w[14], w[13], pars->GC_62, pars->mdl_MZ, pars->mdl_WZ, w[20]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[1], w[6], w[5], pars->GC_2, amp[0]); 
  FFV2_5_0(w[1], w[6], w[7], pars->GC_51, pars->GC_58, amp[1]); 
  FFV1_0(w[8], w[4], w[5], pars->GC_2, amp[2]); 
  FFV2_5_0(w[8], w[4], w[7], pars->GC_51, pars->GC_58, amp[3]); 
  FFV1_0(w[1], w[6], w[5], pars->GC_1, amp[4]); 
  FFV2_3_0(w[1], w[6], w[7], pars->GC_50, pars->GC_58, amp[5]); 
  FFV1_0(w[8], w[4], w[5], pars->GC_1, amp[6]); 
  FFV2_3_0(w[8], w[4], w[7], pars->GC_50, pars->GC_58, amp[7]); 
  FFV1_0(w[10], w[11], w[5], pars->GC_2, amp[8]); 
  FFV2_5_0(w[10], w[11], w[7], pars->GC_51, pars->GC_58, amp[9]); 
  FFV1_0(w[12], w[9], w[5], pars->GC_2, amp[10]); 
  FFV2_5_0(w[12], w[9], w[7], pars->GC_51, pars->GC_58, amp[11]); 
  FFV1_0(w[10], w[11], w[5], pars->GC_1, amp[12]); 
  FFV2_3_0(w[10], w[11], w[7], pars->GC_50, pars->GC_58, amp[13]); 
  FFV1_0(w[12], w[9], w[5], pars->GC_1, amp[14]); 
  FFV2_3_0(w[12], w[9], w[7], pars->GC_50, pars->GC_58, amp[15]); 
  FFV1_0(w[1], w[6], w[15], pars->GC_2, amp[16]); 
  FFV2_5_0(w[1], w[6], w[16], pars->GC_51, pars->GC_58, amp[17]); 
  FFV1_0(w[8], w[4], w[15], pars->GC_2, amp[18]); 
  FFV2_5_0(w[8], w[4], w[16], pars->GC_51, pars->GC_58, amp[19]); 
  FFV1_0(w[1], w[6], w[15], pars->GC_1, amp[20]); 
  FFV2_3_0(w[1], w[6], w[16], pars->GC_50, pars->GC_58, amp[21]); 
  FFV1_0(w[8], w[4], w[15], pars->GC_1, amp[22]); 
  FFV2_3_0(w[8], w[4], w[16], pars->GC_50, pars->GC_58, amp[23]); 
  FFV1_0(w[10], w[11], w[15], pars->GC_2, amp[24]); 
  FFV2_5_0(w[10], w[11], w[16], pars->GC_51, pars->GC_58, amp[25]); 
  FFV1_0(w[12], w[9], w[15], pars->GC_2, amp[26]); 
  FFV2_5_0(w[12], w[9], w[16], pars->GC_51, pars->GC_58, amp[27]); 
  FFV1_0(w[10], w[11], w[15], pars->GC_1, amp[28]); 
  FFV2_3_0(w[10], w[11], w[16], pars->GC_50, pars->GC_58, amp[29]); 
  FFV1_0(w[12], w[9], w[15], pars->GC_1, amp[30]); 
  FFV2_3_0(w[12], w[9], w[16], pars->GC_50, pars->GC_58, amp[31]); 
  FFV2_0(w[1], w[6], w[17], pars->GC_100, amp[32]); 
  FFV2_0(w[8], w[4], w[17], pars->GC_100, amp[33]); 
  FFV2_0(w[10], w[11], w[17], pars->GC_100, amp[34]); 
  FFV2_0(w[12], w[9], w[17], pars->GC_100, amp[35]); 
  FFV2_5_0(w[1], w[6], w[18], pars->GC_51, pars->GC_58, amp[36]); 
  FFV2_5_0(w[8], w[4], w[18], pars->GC_51, pars->GC_58, amp[37]); 
  FFV2_3_0(w[1], w[6], w[18], pars->GC_50, pars->GC_58, amp[38]); 
  FFV2_3_0(w[8], w[4], w[18], pars->GC_50, pars->GC_58, amp[39]); 
  FFV2_5_0(w[10], w[11], w[18], pars->GC_51, pars->GC_58, amp[40]); 
  FFV2_5_0(w[12], w[9], w[18], pars->GC_51, pars->GC_58, amp[41]); 
  FFV2_3_0(w[10], w[11], w[18], pars->GC_50, pars->GC_58, amp[42]); 
  FFV2_3_0(w[12], w[9], w[18], pars->GC_50, pars->GC_58, amp[43]); 
  FFV2_0(w[1], w[6], w[19], pars->GC_100, amp[44]); 
  FFV2_0(w[8], w[4], w[19], pars->GC_100, amp[45]); 
  FFV2_0(w[10], w[11], w[19], pars->GC_100, amp[46]); 
  FFV2_0(w[12], w[9], w[19], pars->GC_100, amp[47]); 
  FFV2_5_0(w[1], w[6], w[20], pars->GC_51, pars->GC_58, amp[48]); 
  FFV2_5_0(w[8], w[4], w[20], pars->GC_51, pars->GC_58, amp[49]); 
  FFV2_3_0(w[1], w[6], w[20], pars->GC_50, pars->GC_58, amp[50]); 
  FFV2_3_0(w[8], w[4], w[20], pars->GC_50, pars->GC_58, amp[51]); 
  FFV2_5_0(w[10], w[11], w[20], pars->GC_51, pars->GC_58, amp[52]); 
  FFV2_5_0(w[12], w[9], w[20], pars->GC_51, pars->GC_58, amp[53]); 
  FFV2_3_0(w[10], w[11], w[20], pars->GC_50, pars->GC_58, amp[54]); 
  FFV2_3_0(w[12], w[9], w[20], pars->GC_50, pars->GC_58, amp[55]); 


}
double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_emu_emgu() 
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

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_emd_emgd() 
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

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_emux_emgux() 
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

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_emdx_emgdx() 
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

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_epu_epgu() 
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
  jamp[0] = +amp[16] + amp[17] + amp[18] + amp[19]; 

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

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_epd_epgd() 
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
  jamp[0] = +amp[20] + amp[21] + amp[22] + amp[23]; 

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

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_epux_epgux() 
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
  jamp[0] = -amp[24] - amp[25] - amp[26] - amp[27]; 

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

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_epdx_epgdx() 
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
  jamp[0] = -amp[28] - amp[29] - amp[30] - amp[31]; 

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

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_emu_vegd() 
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
  jamp[0] = -amp[32] - amp[33]; 

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

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_emdx_vegux() 
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
  jamp[0] = +amp[34] + amp[35]; 

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

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_veu_vegu() 
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
  jamp[0] = -amp[36] - amp[37]; 

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
    jamp2[10][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_ved_vegd() 
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
  jamp[0] = -amp[38] - amp[39]; 

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
    jamp2[11][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_veux_vegux() 
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
  jamp[0] = +amp[40] + amp[41]; 

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
    jamp2[12][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_vedx_vegdx() 
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
  jamp[0] = +amp[42] + amp[43]; 

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
    jamp2[13][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_epd_vexgu() 
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
  jamp[0] = +amp[44] + amp[45]; 

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
    jamp2[14][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_epux_vexgdx() 
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
  jamp[0] = -amp[46] - amp[47]; 

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
    jamp2[15][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_vexu_vexgu() 
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
  jamp[0] = +amp[48] + amp[49]; 

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
    jamp2[16][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_vexd_vexgd() 
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
  jamp[0] = +amp[50] + amp[51]; 

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
    jamp2[17][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_vexux_vexgux() 
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
  jamp[0] = -amp[52] - amp[53]; 

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
    jamp2[18][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}

double PY8MEs_R2_P4_sm_lq_lgq::matrix_2_vexdx_vexgdx() 
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
  jamp[0] = -amp[54] - amp[55]; 

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
    jamp2[19][i] += real(jamp[i] * conj(jamp[i])) * cf[i][i]; 

  return matrix; 
}


}  // end namespace PY8MEs_namespace

