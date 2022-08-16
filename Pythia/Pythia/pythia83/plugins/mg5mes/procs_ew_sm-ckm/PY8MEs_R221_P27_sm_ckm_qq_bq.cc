//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "PY8MEs_R221_P27_sm_ckm_qq_bq.h"
#include "HelAmps_sm_ckm.h"

using namespace Pythia8_sm_ckm; 

namespace PY8MEs_namespace 
{
//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u d > b u QCD=0 @221
// Process: u s > b c QCD=0 @221
// Process: u d > b c QCD=0 @221
// Process: u s > b u QCD=0 @221
// Process: u u~ > b d~ QCD=0 @221
// Process: u c~ > b s~ QCD=0 @221
// Process: u u~ > b s~ QCD=0 @221
// Process: u c~ > b d~ QCD=0 @221
// Process: c d > b u QCD=0 @221
// Process: c s > b c QCD=0 @221
// Process: c d > b c QCD=0 @221
// Process: c s > b u QCD=0 @221
// Process: c u~ > b d~ QCD=0 @221
// Process: c c~ > b s~ QCD=0 @221
// Process: c u~ > b s~ QCD=0 @221
// Process: c c~ > b d~ QCD=0 @221
// Process: d u~ > b u~ QCD=0 @221
// Process: s c~ > b u~ QCD=0 @221
// Process: d u~ > b c~ QCD=0 @221
// Process: s c~ > b c~ QCD=0 @221
// Process: d c~ > b u~ QCD=0 @221
// Process: d c~ > b c~ QCD=0 @221
// Process: s u~ > b u~ QCD=0 @221
// Process: s u~ > b c~ QCD=0 @221
// Process: u u~ > b~ d QCD=0 @221
// Process: c u~ > b~ s QCD=0 @221
// Process: u u~ > b~ s QCD=0 @221
// Process: u c~ > b~ d QCD=0 @221
// Process: c c~ > b~ s QCD=0 @221
// Process: u c~ > b~ s QCD=0 @221
// Process: u d~ > b~ u QCD=0 @221
// Process: c s~ > b~ u QCD=0 @221
// Process: u d~ > b~ c QCD=0 @221
// Process: c s~ > b~ c QCD=0 @221
// Process: u s~ > b~ u QCD=0 @221
// Process: u s~ > b~ c QCD=0 @221
// Process: c u~ > b~ d QCD=0 @221
// Process: c c~ > b~ d QCD=0 @221
// Process: c d~ > b~ u QCD=0 @221
// Process: c d~ > b~ c QCD=0 @221
// Process: u~ d~ > b~ u~ QCD=0 @221
// Process: u~ s~ > b~ c~ QCD=0 @221
// Process: u~ d~ > b~ c~ QCD=0 @221
// Process: u~ s~ > b~ u~ QCD=0 @221
// Process: c~ d~ > b~ u~ QCD=0 @221
// Process: c~ s~ > b~ c~ QCD=0 @221
// Process: c~ d~ > b~ c~ QCD=0 @221
// Process: c~ s~ > b~ u~ QCD=0 @221

// Exception class
class PY8MEs_R221_P27_sm_ckm_qq_bqException : public exception
{
  virtual const char * what() const throw()
  {
    return "Exception in class 'PY8MEs_R221_P27_sm_ckm_qq_bq'."; 
  }
}
PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 

std::set<int> PY8MEs_R221_P27_sm_ckm_qq_bq::s_channel_proc = std::set<int>
    (createset<int> ());

int PY8MEs_R221_P27_sm_ckm_qq_bq::helicities[ncomb][nexternal] = {{-1, -1, -1,
    -1}, {-1, -1, -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1},
    {-1, 1, -1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, 1}, {1, -1, -1, -1}, {1, -1, -1,
    1}, {1, -1, 1, -1}, {1, -1, 1, 1}, {1, 1, -1, -1}, {1, 1, -1, 1}, {1, 1, 1,
    -1}, {1, 1, 1, 1}};

// Normalization factors the various processes
// Denominators: spins, colors and identical particles
int PY8MEs_R221_P27_sm_ckm_qq_bq::denom_colors[nprocesses] = {9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9};
int PY8MEs_R221_P27_sm_ckm_qq_bq::denom_hels[nprocesses] = {4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4};
int PY8MEs_R221_P27_sm_ckm_qq_bq::denom_iden[nprocesses] = {1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1};

//--------------------------------------------------------------------------
// Color config initialization
void PY8MEs_R221_P27_sm_ckm_qq_bq::initColorConfigs() 
{
  color_configs = vector < vec_vec_int > (); 
  jamp_nc_relative_power = vector < vec_int > (); 

  // Color flows of process Process: u d > b u QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[0].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[0].push_back(0); 

  // Color flows of process Process: u d > b c QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[1].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[1].push_back(0); 

  // Color flows of process Process: u s > b u QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[2].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[2].push_back(0); 

  // Color flows of process Process: u u~ > b d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[3].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[3].push_back(0); 

  // Color flows of process Process: u u~ > b s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[4].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[4].push_back(0); 

  // Color flows of process Process: u c~ > b d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[5].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[5].push_back(0); 

  // Color flows of process Process: c d > b u QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[6].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[6].push_back(0); 

  // Color flows of process Process: c d > b c QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[7].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[7].push_back(0); 

  // Color flows of process Process: c s > b u QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[8].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[8].push_back(0); 

  // Color flows of process Process: c u~ > b d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[9].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[9].push_back(0); 

  // Color flows of process Process: c u~ > b s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[10].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[10].push_back(0); 

  // Color flows of process Process: c c~ > b d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[11].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[11].push_back(0); 

  // Color flows of process Process: d u~ > b u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[12].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[12].push_back(0); 

  // Color flows of process Process: d u~ > b c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[13].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[13].push_back(0); 

  // Color flows of process Process: d c~ > b u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[14].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[14].push_back(0); 

  // Color flows of process Process: d c~ > b c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[15].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[15].push_back(0); 

  // Color flows of process Process: s u~ > b u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[16].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[16].push_back(0); 

  // Color flows of process Process: s u~ > b c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[17].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[17].push_back(0); 

  // Color flows of process Process: u u~ > b~ d QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[18].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(1)(2)(0)));
  jamp_nc_relative_power[18].push_back(0); 

  // Color flows of process Process: u u~ > b~ s QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[19].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(1)(2)(0)));
  jamp_nc_relative_power[19].push_back(0); 

  // Color flows of process Process: u c~ > b~ d QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[20].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(1)(2)(0)));
  jamp_nc_relative_power[20].push_back(0); 

  // Color flows of process Process: u c~ > b~ s QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[21].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(1)(2)(0)));
  jamp_nc_relative_power[21].push_back(0); 

  // Color flows of process Process: u d~ > b~ u QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[22].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(2)(2)(0)));
  jamp_nc_relative_power[22].push_back(0); 

  // Color flows of process Process: u d~ > b~ c QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[23].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(2)(2)(0)));
  jamp_nc_relative_power[23].push_back(0); 

  // Color flows of process Process: u s~ > b~ u QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[24].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(2)(2)(0)));
  jamp_nc_relative_power[24].push_back(0); 

  // Color flows of process Process: u s~ > b~ c QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[25].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(2)(2)(0)));
  jamp_nc_relative_power[25].push_back(0); 

  // Color flows of process Process: c u~ > b~ d QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[26].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(1)(2)(0)));
  jamp_nc_relative_power[26].push_back(0); 

  // Color flows of process Process: c c~ > b~ d QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[27].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(1)(2)(0)));
  jamp_nc_relative_power[27].push_back(0); 

  // Color flows of process Process: c d~ > b~ u QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[28].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(2)(2)(0)));
  jamp_nc_relative_power[28].push_back(0); 

  // Color flows of process Process: c d~ > b~ c QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[29].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(2)(2)(0)));
  jamp_nc_relative_power[29].push_back(0); 

  // Color flows of process Process: u~ d~ > b~ u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[30].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[30].push_back(0); 

  // Color flows of process Process: u~ d~ > b~ c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[31].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[31].push_back(0); 

  // Color flows of process Process: u~ s~ > b~ u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[32].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[32].push_back(0); 

  // Color flows of process Process: c~ d~ > b~ u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[33].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[33].push_back(0); 

  // Color flows of process Process: c~ d~ > b~ c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[34].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[34].push_back(0); 

  // Color flows of process Process: c~ s~ > b~ u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[35].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[35].push_back(0); 
}

//--------------------------------------------------------------------------
// Destructor.
PY8MEs_R221_P27_sm_ckm_qq_bq::~PY8MEs_R221_P27_sm_ckm_qq_bq() 
{
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    delete[] p[i]; 
    p[i] = NULL; 
  }
}

//--------------------------------------------------------------------------
// Invert the permutation mapping
vector<int> PY8MEs_R221_P27_sm_ckm_qq_bq::invert_mapping(vector<int> mapping) 
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
vector < vec_int > PY8MEs_R221_P27_sm_ckm_qq_bq::getHelicityConfigs(vector<int>
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
vector < vec_int > PY8MEs_R221_P27_sm_ckm_qq_bq::getColorConfigs(int
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
int PY8MEs_R221_P27_sm_ckm_qq_bq::getColorFlowRelativeNCPower(int
    color_flow_ID, int specify_proc_ID)
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
vector<int> PY8MEs_R221_P27_sm_ckm_qq_bq::getHelicityConfigForID(int hel_ID,
    vector<int> permutation)
{
  if (hel_ID < 0 || hel_ID >= ncomb)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'getHelicityConfigForID' of class" << 
    " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Specified helicity ID '" << 
    hel_ID <<  "' cannot be found." << endl; 
    #endif
    throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
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
int PY8MEs_R221_P27_sm_ckm_qq_bq::getHelicityIDForConfig(vector<int>
    hel_config, vector<int> permutation)
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
      " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Specified helicity" << 
      " configuration cannot be found." << endl; 
      #endif
      throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
    }
  }
  return user_ihel; 
}


//--------------------------------------------------------------------------
// Implements the map Color ID -> Color Config
vector<int> PY8MEs_R221_P27_sm_ckm_qq_bq::getColorConfigForID(int color_ID, int
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
    " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Specified color ID '" << 
    color_ID <<  "' cannot be found." << endl; 
    #endif
    throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
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
int PY8MEs_R221_P27_sm_ckm_qq_bq::getColorIDForConfig(vector<int> color_config,
    int specify_proc_ID, vector<int> permutation)
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
            " 'PY8MEs_R221_P27_sm_ckm_qq_bq': A color line could " << 
            " not be closed." << endl; 
            #endif
            throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
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
      " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Specified color" << 
      " configuration cannot be found." << endl; 
      #endif
      throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
    }
  }
  return user_icol; 
}

//--------------------------------------------------------------------------
// Returns all result previously computed in SigmaKin
vector < vec_double > PY8MEs_R221_P27_sm_ckm_qq_bq::getAllResults(int
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
double PY8MEs_R221_P27_sm_ckm_qq_bq::getResult(int helicity_ID, int color_ID,
    int specify_proc_ID)
{
  if (helicity_ID < - 1 || helicity_ID >= ncomb)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Specified helicity ID '" << 
    helicity_ID <<  "' configuration cannot be found." << endl; 
    #endif
    throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
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
    " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Specified color ID '" << 
    color_ID <<  "' configuration cannot be found." << endl; 
    #endif
    throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
  }
  return all_results[chosenProcID][helicity_ID + 1][color_ID + 1]; 
}

//--------------------------------------------------------------------------
// Check for the availability of the requested process and if available,
// If available, this returns the corresponding permutation and Proc_ID to use.
// If not available, this returns a negative Proc_ID.
pair < vector<int> , int > PY8MEs_R221_P27_sm_ckm_qq_bq::static_getPY8ME(vector<int> initial_pdgs, vector<int> final_pdgs, set<int> schannels) 
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
  const int nprocs = 96; 
  const int proc_IDS[nprocs] = {0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10,
      11, 12, 12, 13, 13, 14, 15, 16, 17, 18, 18, 19, 20, 20, 21, 22, 22, 23,
      23, 24, 25, 26, 27, 28, 29, 30, 30, 31, 32, 33, 33, 34, 35, 0, 0, 1, 2,
      3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 12, 13, 13, 14, 15, 16, 17, 18,
      18, 19, 20, 20, 21, 22, 22, 23, 23, 24, 25, 26, 27, 28, 29, 30, 30, 31,
      32, 33, 33, 34, 35};
  const int in_pdgs[nprocs][ninitial] = {{2, 1}, {2, 3}, {2, 1}, {2, 3}, {2,
      -2}, {2, -4}, {2, -2}, {2, -4}, {4, 1}, {4, 3}, {4, 1}, {4, 3}, {4, -2},
      {4, -4}, {4, -2}, {4, -4}, {1, -2}, {3, -4}, {1, -2}, {3, -4}, {1, -4},
      {1, -4}, {3, -2}, {3, -2}, {2, -2}, {4, -2}, {2, -2}, {2, -4}, {4, -4},
      {2, -4}, {2, -1}, {4, -3}, {2, -1}, {4, -3}, {2, -3}, {2, -3}, {4, -2},
      {4, -4}, {4, -1}, {4, -1}, {-2, -1}, {-2, -3}, {-2, -1}, {-2, -3}, {-4,
      -1}, {-4, -3}, {-4, -1}, {-4, -3}, {1, 2}, {3, 2}, {1, 2}, {3, 2}, {-2,
      2}, {-4, 2}, {-2, 2}, {-4, 2}, {1, 4}, {3, 4}, {1, 4}, {3, 4}, {-2, 4},
      {-4, 4}, {-2, 4}, {-4, 4}, {-2, 1}, {-4, 3}, {-2, 1}, {-4, 3}, {-4, 1},
      {-4, 1}, {-2, 3}, {-2, 3}, {-2, 2}, {-2, 4}, {-2, 2}, {-4, 2}, {-4, 4},
      {-4, 2}, {-1, 2}, {-3, 4}, {-1, 2}, {-3, 4}, {-3, 2}, {-3, 2}, {-2, 4},
      {-4, 4}, {-1, 4}, {-1, 4}, {-1, -2}, {-3, -2}, {-1, -2}, {-3, -2}, {-1,
      -4}, {-3, -4}, {-1, -4}, {-3, -4}};
  const int out_pdgs[nprocs][nexternal - ninitial] = {{5, 2}, {5, 4}, {5, 4},
      {5, 2}, {5, -1}, {5, -3}, {5, -3}, {5, -1}, {5, 2}, {5, 4}, {5, 4}, {5,
      2}, {5, -1}, {5, -3}, {5, -3}, {5, -1}, {5, -2}, {5, -2}, {5, -4}, {5,
      -4}, {5, -2}, {5, -4}, {5, -2}, {5, -4}, {-5, 1}, {-5, 3}, {-5, 3}, {-5,
      1}, {-5, 3}, {-5, 3}, {-5, 2}, {-5, 2}, {-5, 4}, {-5, 4}, {-5, 2}, {-5,
      4}, {-5, 1}, {-5, 1}, {-5, 2}, {-5, 4}, {-5, -2}, {-5, -4}, {-5, -4},
      {-5, -2}, {-5, -2}, {-5, -4}, {-5, -4}, {-5, -2}, {5, 2}, {5, 4}, {5, 4},
      {5, 2}, {5, -1}, {5, -3}, {5, -3}, {5, -1}, {5, 2}, {5, 4}, {5, 4}, {5,
      2}, {5, -1}, {5, -3}, {5, -3}, {5, -1}, {5, -2}, {5, -2}, {5, -4}, {5,
      -4}, {5, -2}, {5, -4}, {5, -2}, {5, -4}, {-5, 1}, {-5, 3}, {-5, 3}, {-5,
      1}, {-5, 3}, {-5, 3}, {-5, 2}, {-5, 2}, {-5, 4}, {-5, 4}, {-5, 2}, {-5,
      4}, {-5, 1}, {-5, 1}, {-5, 2}, {-5, 4}, {-5, -2}, {-5, -4}, {-5, -4},
      {-5, -2}, {-5, -2}, {-5, -4}, {-5, -4}, {-5, -2}};

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
void PY8MEs_R221_P27_sm_ckm_qq_bq::setMomenta(vector < vec_double >
    momenta_picked)
{
  if (momenta_picked.size() != nexternal)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setMomenta' of class" << 
    " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Incorrect number of" << 
    " momenta specified." << endl; 
    #endif
    throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
  }
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    if (momenta_picked[i].size() != 4)
    {
      #ifdef DEBUG
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Incorrect number of" << 
      " momenta components specified." << endl; 
      #endif
      throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
    }
    if (momenta_picked[i][0] < 0.0)
    {
      #ifdef DEBUG
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R221_P27_sm_ckm_qq_bq': A momentum was specified" << 
      " with negative energy. Check conventions." << endl; 
      #endif
      throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
    }
    for (unsigned int j = 0; j < 4; j++ )
    {
      p[i][j] = momenta_picked[i][j]; 
    }
  }
}

//--------------------------------------------------------------------------
// Set color configuration to use. An empty vector means sum over all.
void PY8MEs_R221_P27_sm_ckm_qq_bq::setColors(vector<int> colors_picked)
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
    " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Incorrect number" << 
    " of colors specified." << endl; 
    #endif
    throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
  }
  user_colors = vector<int> ((2 * nexternal), 0); 
  for(unsigned int i = 0; i < (2 * nexternal); i++ )
  {
    user_colors[i] = colors_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the helicity configuration to use. Am empty vector means sum over all.
void PY8MEs_R221_P27_sm_ckm_qq_bq::setHelicities(vector<int> helicities_picked) 
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
    " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Incorrect number" << 
    " of helicities specified." << endl; 
    #endif
    throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
  }
  user_helicities = vector<int> (nexternal, 0); 
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    user_helicities[i] = helicities_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the permutation to use (will apply to momenta, colors and helicities)
void PY8MEs_R221_P27_sm_ckm_qq_bq::setPermutation(vector<int> perm_picked) 
{
  if (perm_picked.size() != nexternal)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setPermutations' of class" << 
    " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Incorrect number" << 
    " of permutations specified." << endl; 
    #endif
    throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
  }
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    perm[i] = perm_picked[i]; 
  }
}

// Set the proc_ID to use
void PY8MEs_R221_P27_sm_ckm_qq_bq::setProcID(int procID_picked) 
{
  proc_ID = procID_picked; 
}

//--------------------------------------------------------------------------
// Initialize process.

void PY8MEs_R221_P27_sm_ckm_qq_bq::initProc() 
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
  jamp2 = vector < vec_double > (36); 
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
  jamp2[20] = vector<double> (1, 0.); 
  jamp2[21] = vector<double> (1, 0.); 
  jamp2[22] = vector<double> (1, 0.); 
  jamp2[23] = vector<double> (1, 0.); 
  jamp2[24] = vector<double> (1, 0.); 
  jamp2[25] = vector<double> (1, 0.); 
  jamp2[26] = vector<double> (1, 0.); 
  jamp2[27] = vector<double> (1, 0.); 
  jamp2[28] = vector<double> (1, 0.); 
  jamp2[29] = vector<double> (1, 0.); 
  jamp2[30] = vector<double> (1, 0.); 
  jamp2[31] = vector<double> (1, 0.); 
  jamp2[32] = vector<double> (1, 0.); 
  jamp2[33] = vector<double> (1, 0.); 
  jamp2[34] = vector<double> (1, 0.); 
  jamp2[35] = vector<double> (1, 0.); 
  all_results = vector < vec_vec_double > (36); 
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
  all_results[20] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[21] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[22] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[23] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[24] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[25] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[26] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[27] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[28] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[29] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[30] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[31] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[32] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[33] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[34] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[35] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
}

// Synchronize local variables of the process that depend on the model
// parameters
void PY8MEs_R221_P27_sm_ckm_qq_bq::syncProcModelParams() 
{

  // Instantiate the model class and set parameters that stay fixed during run
  mME[0] = pars->ZERO; 
  mME[1] = pars->ZERO; 
  mME[2] = pars->mdl_MB; 
  mME[3] = pars->ZERO; 
}

//--------------------------------------------------------------------------
// Setter allowing to force particular values for the external masses
void PY8MEs_R221_P27_sm_ckm_qq_bq::setMasses(vec_double external_masses) 
{

  if (external_masses.size() != mME.size())
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setMasses' of class" << 
    " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Incorrect number of" << 
    " masses specified." << endl; 
    #endif
    throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
  }
  for (unsigned int j = 0; j < mME.size(); j++ )
  {
    mME[j] = external_masses[perm[j]]; 
  }
}

//--------------------------------------------------------------------------
// Getter accessing external masses with the correct ordering
vector<double> PY8MEs_R221_P27_sm_ckm_qq_bq::getMasses() 
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
void PY8MEs_R221_P27_sm_ckm_qq_bq::setExternalMassesMode(int mode) 
{
  if (mode != 0 && mode != 1 && mode != 2)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setExternalMassesMode' of class" << 
    " 'PY8MEs_R221_P27_sm_ckm_qq_bq': Incorrect mode selected :" << mode << 
    ". It must be either 0, 1 or 2" << endl; 
    #endif
    throw PY8MEs_R221_P27_sm_ckm_qq_bq_exception; 
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

double PY8MEs_R221_P27_sm_ckm_qq_bq::sigmaKin() 
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
  for(int i = 0; i < 1; i++ )
    jamp2[20][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[21][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[22][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[23][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[24][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[25][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[26][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[27][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[28][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[29][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[30][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[31][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[32][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[33][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[34][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[35][i] = 0.; 

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
    for(int i = 0; i < 1; i++ )
      jamp2[20][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[21][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[22][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[23][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[24][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[25][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[26][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[27][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[28][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[29][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[30][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[31][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[32][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[33][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[34][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[35][i] = 0.; 

    if (proc_ID == 0)
      t = matrix_221_ud_bu(); 
    if (proc_ID == 1)
      t = matrix_221_ud_bc(); 
    if (proc_ID == 2)
      t = matrix_221_us_bu(); 
    if (proc_ID == 3)
      t = matrix_221_uux_bdx(); 
    if (proc_ID == 4)
      t = matrix_221_uux_bsx(); 
    if (proc_ID == 5)
      t = matrix_221_ucx_bdx(); 
    if (proc_ID == 6)
      t = matrix_221_cd_bu(); 
    if (proc_ID == 7)
      t = matrix_221_cd_bc(); 
    if (proc_ID == 8)
      t = matrix_221_cs_bu(); 
    if (proc_ID == 9)
      t = matrix_221_cux_bdx(); 
    if (proc_ID == 10)
      t = matrix_221_cux_bsx(); 
    if (proc_ID == 11)
      t = matrix_221_ccx_bdx(); 
    if (proc_ID == 12)
      t = matrix_221_dux_bux(); 
    if (proc_ID == 13)
      t = matrix_221_dux_bcx(); 
    if (proc_ID == 14)
      t = matrix_221_dcx_bux(); 
    if (proc_ID == 15)
      t = matrix_221_dcx_bcx(); 
    if (proc_ID == 16)
      t = matrix_221_sux_bux(); 
    if (proc_ID == 17)
      t = matrix_221_sux_bcx(); 
    if (proc_ID == 18)
      t = matrix_221_uux_bxd(); 
    if (proc_ID == 19)
      t = matrix_221_uux_bxs(); 
    if (proc_ID == 20)
      t = matrix_221_ucx_bxd(); 
    if (proc_ID == 21)
      t = matrix_221_ucx_bxs(); 
    if (proc_ID == 22)
      t = matrix_221_udx_bxu(); 
    if (proc_ID == 23)
      t = matrix_221_udx_bxc(); 
    if (proc_ID == 24)
      t = matrix_221_usx_bxu(); 
    if (proc_ID == 25)
      t = matrix_221_usx_bxc(); 
    if (proc_ID == 26)
      t = matrix_221_cux_bxd(); 
    if (proc_ID == 27)
      t = matrix_221_ccx_bxd(); 
    if (proc_ID == 28)
      t = matrix_221_cdx_bxu(); 
    if (proc_ID == 29)
      t = matrix_221_cdx_bxc(); 
    if (proc_ID == 30)
      t = matrix_221_uxdx_bxux(); 
    if (proc_ID == 31)
      t = matrix_221_uxdx_bxcx(); 
    if (proc_ID == 32)
      t = matrix_221_uxsx_bxux(); 
    if (proc_ID == 33)
      t = matrix_221_cxdx_bxux(); 
    if (proc_ID == 34)
      t = matrix_221_cxdx_bxcx(); 
    if (proc_ID == 35)
      t = matrix_221_cxsx_bxux(); 

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

void PY8MEs_R221_P27_sm_ckm_qq_bq::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  // Calculate all wavefunctions
  ixxxxx(p[perm[0]], mME[0], hel[0], +1, w[0]); 
  ixxxxx(p[perm[1]], mME[1], hel[1], +1, w[1]); 
  oxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  oxxxxx(p[perm[3]], mME[3], hel[3], +1, w[3]); 
  FFV2_3(w[0], w[2], pars->GC_102, pars->mdl_MW, pars->mdl_WW, w[4]); 
  oxxxxx(p[perm[1]], mME[1], hel[1], -1, w[5]); 
  ixxxxx(p[perm[3]], mME[3], hel[3], -1, w[6]); 
  FFV2_3(w[0], w[2], pars->GC_46, pars->mdl_MW, pars->mdl_WW, w[7]); 
  FFV2_3(w[6], w[2], pars->GC_102, pars->mdl_MW, pars->mdl_WW, w[8]); 
  FFV2_3(w[6], w[2], pars->GC_46, pars->mdl_MW, pars->mdl_WW, w[9]); 
  ixxxxx(p[perm[2]], mME[2], hel[2], -1, w[10]); 
  FFV2_3(w[0], w[3], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[11]); 
  FFV2_3(w[0], w[3], pars->GC_101, pars->mdl_MW, pars->mdl_WW, w[12]); 
  FFV2_3(w[0], w[5], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[13]); 
  FFV2_3(w[0], w[5], pars->GC_101, pars->mdl_MW, pars->mdl_WW, w[14]); 
  FFV2_3(w[0], w[3], pars->GC_44, pars->mdl_MW, pars->mdl_WW, w[15]); 
  FFV2_3(w[0], w[5], pars->GC_44, pars->mdl_MW, pars->mdl_WW, w[16]); 
  oxxxxx(p[perm[0]], mME[0], hel[0], -1, w[17]); 
  FFV2_3(w[6], w[5], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[18]); 
  FFV2_3(w[6], w[5], pars->GC_44, pars->mdl_MW, pars->mdl_WW, w[19]); 
  FFV2_3(w[6], w[5], pars->GC_101, pars->mdl_MW, pars->mdl_WW, w[20]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV2_0(w[1], w[3], w[4], pars->GC_100, amp[0]); 
  FFV2_0(w[1], w[3], w[4], pars->GC_44, amp[1]); 
  FFV2_0(w[1], w[3], w[4], pars->GC_101, amp[2]); 
  FFV2_0(w[6], w[5], w[4], pars->GC_100, amp[3]); 
  FFV2_0(w[6], w[5], w[4], pars->GC_101, amp[4]); 
  FFV2_0(w[6], w[5], w[4], pars->GC_44, amp[5]); 
  FFV2_0(w[1], w[3], w[7], pars->GC_100, amp[6]); 
  FFV2_0(w[1], w[3], w[7], pars->GC_44, amp[7]); 
  FFV2_0(w[1], w[3], w[7], pars->GC_101, amp[8]); 
  FFV2_0(w[6], w[5], w[7], pars->GC_100, amp[9]); 
  FFV2_0(w[6], w[5], w[7], pars->GC_101, amp[10]); 
  FFV2_0(w[6], w[5], w[7], pars->GC_44, amp[11]); 
  FFV2_0(w[0], w[5], w[8], pars->GC_100, amp[12]); 
  FFV2_0(w[0], w[5], w[9], pars->GC_100, amp[13]); 
  FFV2_0(w[0], w[5], w[8], pars->GC_44, amp[14]); 
  FFV2_0(w[0], w[5], w[9], pars->GC_44, amp[15]); 
  FFV2_0(w[0], w[5], w[8], pars->GC_101, amp[16]); 
  FFV2_0(w[0], w[5], w[9], pars->GC_101, amp[17]); 
  FFV2_0(w[10], w[5], w[11], pars->GC_43, amp[18]); 
  FFV2_0(w[10], w[5], w[12], pars->GC_43, amp[19]); 
  FFV2_0(w[10], w[5], w[11], pars->GC_46, amp[20]); 
  FFV2_0(w[10], w[5], w[12], pars->GC_46, amp[21]); 
  FFV2_0(w[10], w[3], w[13], pars->GC_43, amp[22]); 
  FFV2_0(w[10], w[3], w[13], pars->GC_46, amp[23]); 
  FFV2_0(w[10], w[3], w[14], pars->GC_43, amp[24]); 
  FFV2_0(w[10], w[3], w[14], pars->GC_46, amp[25]); 
  FFV2_0(w[10], w[5], w[15], pars->GC_43, amp[26]); 
  FFV2_0(w[10], w[5], w[15], pars->GC_46, amp[27]); 
  FFV2_0(w[10], w[3], w[16], pars->GC_43, amp[28]); 
  FFV2_0(w[10], w[3], w[16], pars->GC_46, amp[29]); 
  FFV2_0(w[10], w[17], w[18], pars->GC_43, amp[30]); 
  FFV2_0(w[10], w[17], w[19], pars->GC_43, amp[31]); 
  FFV2_0(w[10], w[17], w[20], pars->GC_43, amp[32]); 
  FFV2_0(w[10], w[17], w[18], pars->GC_46, amp[33]); 
  FFV2_0(w[10], w[17], w[19], pars->GC_46, amp[34]); 
  FFV2_0(w[10], w[17], w[20], pars->GC_46, amp[35]); 


}
double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_ud_bu() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[0]; 

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

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_ud_bc() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[1]; 

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

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_us_bu() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[2]; 

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

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_uux_bdx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[3]; 

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

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_uux_bsx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[4]; 

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

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_ucx_bdx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[5]; 

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

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_cd_bu() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[6]; 

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

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_cd_bc() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[7]; 

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
    jamp2[7][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_cs_bu() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[8]; 

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
    jamp2[8][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_cux_bdx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[9]; 

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
    jamp2[9][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_cux_bsx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[10]; 

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
    jamp2[10][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_ccx_bdx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[11]; 

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
    jamp2[11][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_dux_bux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[12]; 

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
    jamp2[12][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_dux_bcx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[13]; 

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
    jamp2[13][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_dcx_bux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[14]; 

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
    jamp2[14][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_dcx_bcx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[15]; 

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
    jamp2[15][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_sux_bux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[16]; 

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
    jamp2[16][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_sux_bcx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[17]; 

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
    jamp2[17][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_uux_bxd() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[18]; 

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
    jamp2[18][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_uux_bxs() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[19]; 

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
    jamp2[19][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_ucx_bxd() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[20]; 

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
    jamp2[20][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_ucx_bxs() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[21]; 

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
    jamp2[21][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_udx_bxu() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[22]; 

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
    jamp2[22][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_udx_bxc() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[23]; 

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
    jamp2[23][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_usx_bxu() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[24]; 

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
    jamp2[24][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_usx_bxc() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[25]; 

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
    jamp2[25][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_cux_bxd() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[26]; 

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
    jamp2[26][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_ccx_bxd() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[27]; 

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
    jamp2[27][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_cdx_bxu() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[28]; 

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
    jamp2[28][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_cdx_bxc() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[29]; 

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
    jamp2[29][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_uxdx_bxux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[30]; 

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
    jamp2[30][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_uxdx_bxcx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[31]; 

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
    jamp2[31][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_uxsx_bxux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[32]; 

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
    jamp2[32][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_cxdx_bxux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[33]; 

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
    jamp2[33][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_cxdx_bxcx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[34]; 

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
    jamp2[34][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P27_sm_ckm_qq_bq::matrix_221_cxsx_bxux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 1;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[35]; 

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
    jamp2[35][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}


}  // end namespace PY8MEs_namespace

