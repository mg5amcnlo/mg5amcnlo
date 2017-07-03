//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.5.3, 2017-03-09
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "PY8MEs_R6_P5_sm_qq_ttxqq.h"
#include "HelAmps_sm.h"

using namespace Pythia8_sm; 

namespace PY8MEs_namespace 
{
//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u > t t~ u u WEIGHTED<=4 @6
// Process: c c > t t~ c c WEIGHTED<=4 @6
// Process: d d > t t~ d d WEIGHTED<=4 @6
// Process: s s > t t~ s s WEIGHTED<=4 @6
// Process: u u~ > t t~ u u~ WEIGHTED<=4 @6
// Process: c c~ > t t~ c c~ WEIGHTED<=4 @6
// Process: d d~ > t t~ d d~ WEIGHTED<=4 @6
// Process: s s~ > t t~ s s~ WEIGHTED<=4 @6
// Process: u~ u~ > t t~ u~ u~ WEIGHTED<=4 @6
// Process: c~ c~ > t t~ c~ c~ WEIGHTED<=4 @6
// Process: d~ d~ > t t~ d~ d~ WEIGHTED<=4 @6
// Process: s~ s~ > t t~ s~ s~ WEIGHTED<=4 @6
// Process: u c > t t~ u c WEIGHTED<=4 @6
// Process: u d > t t~ u d WEIGHTED<=4 @6
// Process: u s > t t~ u s WEIGHTED<=4 @6
// Process: c d > t t~ c d WEIGHTED<=4 @6
// Process: c s > t t~ c s WEIGHTED<=4 @6
// Process: d s > t t~ d s WEIGHTED<=4 @6
// Process: u u~ > t t~ c c~ WEIGHTED<=4 @6
// Process: u u~ > t t~ d d~ WEIGHTED<=4 @6
// Process: u u~ > t t~ s s~ WEIGHTED<=4 @6
// Process: c c~ > t t~ u u~ WEIGHTED<=4 @6
// Process: c c~ > t t~ d d~ WEIGHTED<=4 @6
// Process: c c~ > t t~ s s~ WEIGHTED<=4 @6
// Process: d d~ > t t~ u u~ WEIGHTED<=4 @6
// Process: d d~ > t t~ c c~ WEIGHTED<=4 @6
// Process: d d~ > t t~ s s~ WEIGHTED<=4 @6
// Process: s s~ > t t~ u u~ WEIGHTED<=4 @6
// Process: s s~ > t t~ c c~ WEIGHTED<=4 @6
// Process: s s~ > t t~ d d~ WEIGHTED<=4 @6
// Process: u c~ > t t~ u c~ WEIGHTED<=4 @6
// Process: u d~ > t t~ u d~ WEIGHTED<=4 @6
// Process: u s~ > t t~ u s~ WEIGHTED<=4 @6
// Process: c u~ > t t~ c u~ WEIGHTED<=4 @6
// Process: c d~ > t t~ c d~ WEIGHTED<=4 @6
// Process: c s~ > t t~ c s~ WEIGHTED<=4 @6
// Process: d u~ > t t~ d u~ WEIGHTED<=4 @6
// Process: d c~ > t t~ d c~ WEIGHTED<=4 @6
// Process: d s~ > t t~ d s~ WEIGHTED<=4 @6
// Process: s u~ > t t~ s u~ WEIGHTED<=4 @6
// Process: s c~ > t t~ s c~ WEIGHTED<=4 @6
// Process: s d~ > t t~ s d~ WEIGHTED<=4 @6
// Process: u~ c~ > t t~ u~ c~ WEIGHTED<=4 @6
// Process: u~ d~ > t t~ u~ d~ WEIGHTED<=4 @6
// Process: u~ s~ > t t~ u~ s~ WEIGHTED<=4 @6
// Process: c~ d~ > t t~ c~ d~ WEIGHTED<=4 @6
// Process: c~ s~ > t t~ c~ s~ WEIGHTED<=4 @6
// Process: d~ s~ > t t~ d~ s~ WEIGHTED<=4 @6

// Exception class
class PY8MEs_R6_P5_sm_qq_ttxqqException : public exception
{
  virtual const char * what() const throw()
  {
    return "Exception in class 'PY8MEs_R6_P5_sm_qq_ttxqq'."; 
  }
}
PY8MEs_R6_P5_sm_qq_ttxqq_exception; 

// Required s-channel initialization
int PY8MEs_R6_P5_sm_qq_ttxqq::req_s_channels[nreq_s_channels] = {}; 

int PY8MEs_R6_P5_sm_qq_ttxqq::helicities[ncomb][nexternal] = {{-1, -1, -1, -1,
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
void PY8MEs_R6_P5_sm_qq_ttxqq::initColorConfigs() 
{
  color_configs = vector < vec_vec_int > (); 

  // Color flows of process Process: u u > t t~ u u WEIGHTED<=4 @6
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp00[2 * nexternal] = {1, 0, 2, 0, 1, 0, 0, 3, 2, 0, 3, 0}; 
  color_configs[0].push_back(vec_int(jamp00, jamp00 + (2 * nexternal))); 
  // JAMP #1
  int jamp01[2 * nexternal] = {1, 0, 3, 0, 1, 0, 0, 2, 2, 0, 3, 0}; 
  color_configs[0].push_back(vec_int(jamp01, jamp01 + (2 * nexternal))); 
  // JAMP #2
  int jamp02[2 * nexternal] = {2, 0, 1, 0, 1, 0, 0, 3, 2, 0, 3, 0}; 
  color_configs[0].push_back(vec_int(jamp02, jamp02 + (2 * nexternal))); 
  // JAMP #3
  int jamp03[2 * nexternal] = {3, 0, 1, 0, 1, 0, 0, 2, 2, 0, 3, 0}; 
  color_configs[0].push_back(vec_int(jamp03, jamp03 + (2 * nexternal))); 
  // JAMP #4
  int jamp04[2 * nexternal] = {2, 0, 3, 0, 1, 0, 0, 1, 2, 0, 3, 0}; 
  color_configs[0].push_back(vec_int(jamp04, jamp04 + (2 * nexternal))); 
  // JAMP #5
  int jamp05[2 * nexternal] = {3, 0, 2, 0, 1, 0, 0, 1, 2, 0, 3, 0}; 
  color_configs[0].push_back(vec_int(jamp05, jamp05 + (2 * nexternal))); 

  // Color flows of process Process: u u~ > t t~ u u~ WEIGHTED<=4 @6
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp10[2 * nexternal] = {1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3}; 
  color_configs[1].push_back(vec_int(jamp10, jamp10 + (2 * nexternal))); 
  // JAMP #1
  int jamp11[2 * nexternal] = {1, 0, 0, 1, 2, 0, 0, 3, 3, 0, 0, 2}; 
  color_configs[1].push_back(vec_int(jamp11, jamp11 + (2 * nexternal))); 
  // JAMP #2
  int jamp12[2 * nexternal] = {2, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 3}; 
  color_configs[1].push_back(vec_int(jamp12, jamp12 + (2 * nexternal))); 
  // JAMP #3
  int jamp13[2 * nexternal] = {3, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 2}; 
  color_configs[1].push_back(vec_int(jamp13, jamp13 + (2 * nexternal))); 
  // JAMP #4
  int jamp14[2 * nexternal] = {2, 0, 0, 1, 2, 0, 0, 3, 3, 0, 0, 1}; 
  color_configs[1].push_back(vec_int(jamp14, jamp14 + (2 * nexternal))); 
  // JAMP #5
  int jamp15[2 * nexternal] = {3, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 1}; 
  color_configs[1].push_back(vec_int(jamp15, jamp15 + (2 * nexternal))); 

  // Color flows of process Process: u~ u~ > t t~ u~ u~ WEIGHTED<=4 @6
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp20[2 * nexternal] = {0, 1, 0, 2, 3, 0, 0, 1, 0, 2, 0, 3}; 
  color_configs[2].push_back(vec_int(jamp20, jamp20 + (2 * nexternal))); 
  // JAMP #1
  int jamp21[2 * nexternal] = {0, 1, 0, 2, 3, 0, 0, 1, 0, 3, 0, 2}; 
  color_configs[2].push_back(vec_int(jamp21, jamp21 + (2 * nexternal))); 
  // JAMP #2
  int jamp22[2 * nexternal] = {0, 1, 0, 2, 3, 0, 0, 2, 0, 1, 0, 3}; 
  color_configs[2].push_back(vec_int(jamp22, jamp22 + (2 * nexternal))); 
  // JAMP #3
  int jamp23[2 * nexternal] = {0, 1, 0, 2, 3, 0, 0, 3, 0, 1, 0, 2}; 
  color_configs[2].push_back(vec_int(jamp23, jamp23 + (2 * nexternal))); 
  // JAMP #4
  int jamp24[2 * nexternal] = {0, 1, 0, 2, 3, 0, 0, 2, 0, 3, 0, 1}; 
  color_configs[2].push_back(vec_int(jamp24, jamp24 + (2 * nexternal))); 
  // JAMP #5
  int jamp25[2 * nexternal] = {0, 1, 0, 2, 3, 0, 0, 3, 0, 2, 0, 1}; 
  color_configs[2].push_back(vec_int(jamp25, jamp25 + (2 * nexternal))); 

  // Color flows of process Process: u c > t t~ u c WEIGHTED<=4 @6
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp30[2 * nexternal] = {1, 0, 2, 0, 1, 0, 0, 3, 2, 0, 3, 0}; 
  color_configs[3].push_back(vec_int(jamp30, jamp30 + (2 * nexternal))); 
  // JAMP #1
  int jamp31[2 * nexternal] = {1, 0, 3, 0, 1, 0, 0, 2, 2, 0, 3, 0}; 
  color_configs[3].push_back(vec_int(jamp31, jamp31 + (2 * nexternal))); 
  // JAMP #2
  int jamp32[2 * nexternal] = {2, 0, 1, 0, 1, 0, 0, 3, 2, 0, 3, 0}; 
  color_configs[3].push_back(vec_int(jamp32, jamp32 + (2 * nexternal))); 
  // JAMP #3
  int jamp33[2 * nexternal] = {3, 0, 1, 0, 1, 0, 0, 2, 2, 0, 3, 0}; 
  color_configs[3].push_back(vec_int(jamp33, jamp33 + (2 * nexternal))); 
  // JAMP #4
  int jamp34[2 * nexternal] = {2, 0, 3, 0, 1, 0, 0, 1, 2, 0, 3, 0}; 
  color_configs[3].push_back(vec_int(jamp34, jamp34 + (2 * nexternal))); 
  // JAMP #5
  int jamp35[2 * nexternal] = {3, 0, 2, 0, 1, 0, 0, 1, 2, 0, 3, 0}; 
  color_configs[3].push_back(vec_int(jamp35, jamp35 + (2 * nexternal))); 

  // Color flows of process Process: u u~ > t t~ c c~ WEIGHTED<=4 @6
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp40[2 * nexternal] = {1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3}; 
  color_configs[4].push_back(vec_int(jamp40, jamp40 + (2 * nexternal))); 
  // JAMP #1
  int jamp41[2 * nexternal] = {1, 0, 0, 1, 2, 0, 0, 3, 3, 0, 0, 2}; 
  color_configs[4].push_back(vec_int(jamp41, jamp41 + (2 * nexternal))); 
  // JAMP #2
  int jamp42[2 * nexternal] = {2, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 3}; 
  color_configs[4].push_back(vec_int(jamp42, jamp42 + (2 * nexternal))); 
  // JAMP #3
  int jamp43[2 * nexternal] = {3, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 2}; 
  color_configs[4].push_back(vec_int(jamp43, jamp43 + (2 * nexternal))); 
  // JAMP #4
  int jamp44[2 * nexternal] = {2, 0, 0, 1, 2, 0, 0, 3, 3, 0, 0, 1}; 
  color_configs[4].push_back(vec_int(jamp44, jamp44 + (2 * nexternal))); 
  // JAMP #5
  int jamp45[2 * nexternal] = {3, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 1}; 
  color_configs[4].push_back(vec_int(jamp45, jamp45 + (2 * nexternal))); 

  // Color flows of process Process: u c~ > t t~ u c~ WEIGHTED<=4 @6
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp50[2 * nexternal] = {1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3}; 
  color_configs[5].push_back(vec_int(jamp50, jamp50 + (2 * nexternal))); 
  // JAMP #1
  int jamp51[2 * nexternal] = {1, 0, 0, 1, 2, 0, 0, 3, 3, 0, 0, 2}; 
  color_configs[5].push_back(vec_int(jamp51, jamp51 + (2 * nexternal))); 
  // JAMP #2
  int jamp52[2 * nexternal] = {2, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 3}; 
  color_configs[5].push_back(vec_int(jamp52, jamp52 + (2 * nexternal))); 
  // JAMP #3
  int jamp53[2 * nexternal] = {3, 0, 0, 1, 2, 0, 0, 1, 3, 0, 0, 2}; 
  color_configs[5].push_back(vec_int(jamp53, jamp53 + (2 * nexternal))); 
  // JAMP #4
  int jamp54[2 * nexternal] = {2, 0, 0, 1, 2, 0, 0, 3, 3, 0, 0, 1}; 
  color_configs[5].push_back(vec_int(jamp54, jamp54 + (2 * nexternal))); 
  // JAMP #5
  int jamp55[2 * nexternal] = {3, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 1}; 
  color_configs[5].push_back(vec_int(jamp55, jamp55 + (2 * nexternal))); 

  // Color flows of process Process: u~ c~ > t t~ u~ c~ WEIGHTED<=4 @6
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp60[2 * nexternal] = {0, 1, 0, 2, 3, 0, 0, 1, 0, 2, 0, 3}; 
  color_configs[6].push_back(vec_int(jamp60, jamp60 + (2 * nexternal))); 
  // JAMP #1
  int jamp61[2 * nexternal] = {0, 1, 0, 2, 3, 0, 0, 1, 0, 3, 0, 2}; 
  color_configs[6].push_back(vec_int(jamp61, jamp61 + (2 * nexternal))); 
  // JAMP #2
  int jamp62[2 * nexternal] = {0, 1, 0, 2, 3, 0, 0, 2, 0, 1, 0, 3}; 
  color_configs[6].push_back(vec_int(jamp62, jamp62 + (2 * nexternal))); 
  // JAMP #3
  int jamp63[2 * nexternal] = {0, 1, 0, 2, 3, 0, 0, 3, 0, 1, 0, 2}; 
  color_configs[6].push_back(vec_int(jamp63, jamp63 + (2 * nexternal))); 
  // JAMP #4
  int jamp64[2 * nexternal] = {0, 1, 0, 2, 3, 0, 0, 2, 0, 3, 0, 1}; 
  color_configs[6].push_back(vec_int(jamp64, jamp64 + (2 * nexternal))); 
  // JAMP #5
  int jamp65[2 * nexternal] = {0, 1, 0, 2, 3, 0, 0, 3, 0, 2, 0, 1}; 
  color_configs[6].push_back(vec_int(jamp65, jamp65 + (2 * nexternal))); 
}

//--------------------------------------------------------------------------
// Destructor.
PY8MEs_R6_P5_sm_qq_ttxqq::~PY8MEs_R6_P5_sm_qq_ttxqq() 
{
  for(int i = 0; i < nexternal; i++ )
  {
    delete[] p[i]; 
    p[i] = NULL; 
  }
}

//--------------------------------------------------------------------------
// Return the list of possible helicity configurations
vector < vec_int > PY8MEs_R6_P5_sm_qq_ttxqq::getHelicityConfigs(vector<int>
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
vector < vec_int > PY8MEs_R6_P5_sm_qq_ttxqq::getColorConfigs(int
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
vector<int> PY8MEs_R6_P5_sm_qq_ttxqq::getHelicityConfigForID(int hel_ID,
    vector<int> permutation)
{
  if (hel_ID < 0 || hel_ID >= ncomb)
  {
    cout <<  "Error in function 'getHelicityConfigForID' of class" << 
    " 'PY8MEs_R6_P5_sm_qq_ttxqq': Specified helicity ID '" << 
    hel_ID <<  "' cannot be found." << endl; 
    throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
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
int PY8MEs_R6_P5_sm_qq_ttxqq::getHelicityIDForConfig(vector<int> hel_config,
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
      " 'PY8MEs_R6_P5_sm_qq_ttxqq': Specified helicity" << 
      " configuration cannot be found." << endl; 
      throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
    }
  }
  return user_ihel; 
}


//--------------------------------------------------------------------------
// Implements the map Color ID -> Color Config
vector<int> PY8MEs_R6_P5_sm_qq_ttxqq::getColorConfigForID(int color_ID, int
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
    " 'PY8MEs_R6_P5_sm_qq_ttxqq': Specified color ID '" << 
    color_ID <<  "' cannot be found." << endl; 
    throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
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
int PY8MEs_R6_P5_sm_qq_ttxqq::getColorIDForConfig(vector<int> color_config, int
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
            " 'PY8MEs_R6_P5_sm_qq_ttxqq': A color line could " << 
            " not be closed." << endl;
            return -2; 
            //throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
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
      " 'PY8MEs_R6_P5_sm_qq_ttxqq': Specified color" << 
      " configuration cannot be found." << endl;  
      return -2;
      //throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
    }
  }
  return user_icol; 
}

//--------------------------------------------------------------------------
// Returns all result previously computed in SigmaKin
vector < vec_double > PY8MEs_R6_P5_sm_qq_ttxqq::getAllResults(int
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
double PY8MEs_R6_P5_sm_qq_ttxqq::getResult(int helicity_ID, int color_ID, int
    specify_proc_ID)
{
  if (helicity_ID < - 1 || helicity_ID >= ncomb)
  {
    cout <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R6_P5_sm_qq_ttxqq': Specified helicity ID '" << 
    helicity_ID <<  "' configuration cannot be found." << endl; 
    throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
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
    " 'PY8MEs_R6_P5_sm_qq_ttxqq': Specified color ID '" << 
    color_ID <<  "' configuration cannot be found." << endl; 
    throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
  }
  return all_results[chosenProcID][helicity_ID + 1][color_ID + 1]; 
}

//--------------------------------------------------------------------------
// Check for the availability of the requested process and if available,
// If available, this returns the corresponding permutation and Proc_ID to use.
// If not available, this returns a negative Proc_ID.
pair < vector<int> , int > PY8MEs_R6_P5_sm_qq_ttxqq::static_getPY8ME(vector<int> initial_pdgs, vector<int> final_pdgs, set<int> schannels) 
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
  const int out_pdgs[nprocs][nexternal - ninitial] = {{6, -6, 2, 2}, {6, -6, 4,
      4}, {6, -6, 1, 1}, {6, -6, 3, 3}, {6, -6, 2, -2}, {6, -6, 4, -4}, {6, -6,
      1, -1}, {6, -6, 3, -3}, {6, -6, -2, -2}, {6, -6, -4, -4}, {6, -6, -1,
      -1}, {6, -6, -3, -3}, {6, -6, 2, 4}, {6, -6, 2, 1}, {6, -6, 2, 3}, {6,
      -6, 4, 1}, {6, -6, 4, 3}, {6, -6, 1, 3}, {6, -6, 4, -4}, {6, -6, 1, -1},
      {6, -6, 3, -3}, {6, -6, 2, -2}, {6, -6, 1, -1}, {6, -6, 3, -3}, {6, -6,
      2, -2}, {6, -6, 4, -4}, {6, -6, 3, -3}, {6, -6, 2, -2}, {6, -6, 4, -4},
      {6, -6, 1, -1}, {6, -6, 2, -4}, {6, -6, 2, -1}, {6, -6, 2, -3}, {6, -6,
      4, -2}, {6, -6, 4, -1}, {6, -6, 4, -3}, {6, -6, 1, -2}, {6, -6, 1, -4},
      {6, -6, 1, -3}, {6, -6, 3, -2}, {6, -6, 3, -4}, {6, -6, 3, -1}, {6, -6,
      -2, -4}, {6, -6, -2, -1}, {6, -6, -2, -3}, {6, -6, -4, -1}, {6, -6, -4,
      -3}, {6, -6, -1, -3}, {6, -6, 2, -2}, {6, -6, 4, -4}, {6, -6, 1, -1}, {6,
      -6, 3, -3}, {6, -6, 2, 4}, {6, -6, 2, 1}, {6, -6, 2, 3}, {6, -6, 4, 1},
      {6, -6, 4, 3}, {6, -6, 1, 3}, {6, -6, 4, -4}, {6, -6, 1, -1}, {6, -6, 3,
      -3}, {6, -6, 2, -2}, {6, -6, 1, -1}, {6, -6, 3, -3}, {6, -6, 2, -2}, {6,
      -6, 4, -4}, {6, -6, 3, -3}, {6, -6, 2, -2}, {6, -6, 4, -4}, {6, -6, 1,
      -1}, {6, -6, 2, -4}, {6, -6, 2, -1}, {6, -6, 2, -3}, {6, -6, 4, -2}, {6,
      -6, 4, -1}, {6, -6, 4, -3}, {6, -6, 1, -2}, {6, -6, 1, -4}, {6, -6, 1,
      -3}, {6, -6, 3, -2}, {6, -6, 3, -4}, {6, -6, 3, -1}, {6, -6, -2, -4}, {6,
      -6, -2, -1}, {6, -6, -2, -3}, {6, -6, -4, -1}, {6, -6, -4, -3}, {6, -6,
      -1, -3}};

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
void PY8MEs_R6_P5_sm_qq_ttxqq::setMomenta(vector < vec_double > momenta_picked)
{
  if (momenta_picked.size() != nexternal)
  {
    cout <<  "Error in function 'setMomenta' of class" << 
    " 'PY8MEs_R6_P5_sm_qq_ttxqq': Incorrect number of" << 
    " momenta specified." << endl; 
    throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
  }
  for(int i = 0; i < nexternal; i++ )
  {
    if (momenta_picked[i].size() != 4)
    {
      cout <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R6_P5_sm_qq_ttxqq': Incorrect number of" << 
      " momenta components specified." << endl; 
      throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
    }
    if (momenta_picked[i][0] < 0.0)
    {
      cout <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R6_P5_sm_qq_ttxqq': A momentum was specified" << 
      " with negative energy. Check conventions." << endl; 
      throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
    }
    for (int j = 0; j < 4; j++ )
    {
      p[i][j] = momenta_picked[i][j]; 
    }
  }
}

//--------------------------------------------------------------------------
// Set color configuration to use. An empty vector means sum over all.
void PY8MEs_R6_P5_sm_qq_ttxqq::setColors(vector<int> colors_picked)
{
  if (colors_picked.size() == 0)
  {
    user_colors = vector<int> (); 
    return; 
  }
  if (colors_picked.size() != (2 * nexternal))
  {
    cout <<  "Error in function 'setColors' of class" << 
    " 'PY8MEs_R6_P5_sm_qq_ttxqq': Incorrect number" << 
    " of colors specified." << endl; 
    throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
  }
  user_colors = vector<int> ((2 * nexternal), 0); 
  for(int i = 0; i < (2 * nexternal); i++ )
  {
    user_colors[i] = colors_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the helicity configuration to use. Am empty vector means sum over all.
void PY8MEs_R6_P5_sm_qq_ttxqq::setHelicities(vector<int> helicities_picked) 
{
  if (helicities_picked.size() != nexternal)
  {
    if (helicities_picked.size() == 0)
    {
      user_helicities = vector<int> (); 
      return; 
    }
    cout <<  "Error in function 'setHelicities' of class" << 
    " 'PY8MEs_R6_P5_sm_qq_ttxqq': Incorrect number" << 
    " of helicities specified." << endl; 
    throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
  }
  user_helicities = vector<int> (nexternal, 0); 
  for(int i = 0; i < nexternal; i++ )
  {
    user_helicities[i] = helicities_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the permutation to use (will apply to momenta, colors and helicities)
void PY8MEs_R6_P5_sm_qq_ttxqq::setPermutation(vector<int> perm_picked) 
{
  if (perm_picked.size() != nexternal)
  {
    cout <<  "Error in function 'setPermutations' of class" << 
    " 'PY8MEs_R6_P5_sm_qq_ttxqq': Incorrect number" << 
    " of permutations specified." << endl; 
    throw PY8MEs_R6_P5_sm_qq_ttxqq_exception; 
  }
  for(int i = 0; i < nexternal; i++ )
  {
    perm[i] = perm_picked[i]; 
  }
}

// Set the proc_ID to use
void PY8MEs_R6_P5_sm_qq_ttxqq::setProcID(int procID_picked) 
{
  proc_ID = procID_picked; 
}

//--------------------------------------------------------------------------
// Initialize process.

void PY8MEs_R6_P5_sm_qq_ttxqq::initProc() 
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
  jamp2 = vector < vec_double > (7); 
  jamp2[0] = vector<double> (6, 0.); 
  jamp2[1] = vector<double> (6, 0.); 
  jamp2[2] = vector<double> (6, 0.); 
  jamp2[3] = vector<double> (6, 0.); 
  jamp2[4] = vector<double> (6, 0.); 
  jamp2[5] = vector<double> (6, 0.); 
  jamp2[6] = vector<double> (6, 0.); 
  all_results = vector < vec_vec_double > (7); 
  // The first entry is always the color or helicity avg/summed matrix element.
  all_results[0] = vector < vec_double > (ncomb + 1, vector<double> (6 + 1,
      0.));
  all_results[1] = vector < vec_double > (ncomb + 1, vector<double> (6 + 1,
      0.));
  all_results[2] = vector < vec_double > (ncomb + 1, vector<double> (6 + 1,
      0.));
  all_results[3] = vector < vec_double > (ncomb + 1, vector<double> (6 + 1,
      0.));
  all_results[4] = vector < vec_double > (ncomb + 1, vector<double> (6 + 1,
      0.));
  all_results[5] = vector < vec_double > (ncomb + 1, vector<double> (6 + 1,
      0.));
  all_results[6] = vector < vec_double > (ncomb + 1, vector<double> (6 + 1,
      0.));
}

//--------------------------------------------------------------------------
// Evaluate the squared matrix element.

double PY8MEs_R6_P5_sm_qq_ttxqq::sigmaKin() 
{
  // Set the parameters which change event by event
  pars->setDependentParameters(); 
  pars->setDependentCouplings(); 
  // Reset color flows
  for(int i = 0; i < 6; i++ )
    jamp2[0][i] = 0.; 
  for(int i = 0; i < 6; i++ )
    jamp2[1][i] = 0.; 
  for(int i = 0; i < 6; i++ )
    jamp2[2][i] = 0.; 
  for(int i = 0; i < 6; i++ )
    jamp2[3][i] = 0.; 
  for(int i = 0; i < 6; i++ )
    jamp2[4][i] = 0.; 
  for(int i = 0; i < 6; i++ )
    jamp2[5][i] = 0.; 
  for(int i = 0; i < 6; i++ )
    jamp2[6][i] = 0.; 

  // Local variables and constants
  const int max_tries = 10; 
  vector < vec_bool > goodhel(nprocesses, vec_bool(ncomb, false)); 
  vec_int ntry(nprocesses, 0); 
  double t = 0.; 
  double result = 0.; 

  // Denominators: spins, colors and identical particles
  const int denom_colors[nprocesses] = {9, 9, 9, 9, 9, 9, 9}; 
  const int denom_hels[nprocesses] = {4, 4, 4, 4, 4, 4, 4}; 
  const int denom_iden[nprocesses] = {2, 1, 2, 1, 1, 1, 1}; 

  if (ntry[proc_ID] <= max_tries)
    ntry[proc_ID] = ntry[proc_ID] + 1; 

  // Find which helicity configuration is asked for
  // -1 indicates one wants to sum over helicities
  int user_ihel = getHelicityIDForConfig(user_helicities); 

  // Find which color configuration is asked for
  // -1 indicates one wants to sum over all color configurations
  int user_icol = getColorIDForConfig(user_colors); 
if(user_icol==-2) return -1.;
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
    for(int i = 0; i < 6; i++ )
      jamp2[0][i] = 0.; 
    for(int i = 0; i < 6; i++ )
      jamp2[1][i] = 0.; 
    for(int i = 0; i < 6; i++ )
      jamp2[2][i] = 0.; 
    for(int i = 0; i < 6; i++ )
      jamp2[3][i] = 0.; 
    for(int i = 0; i < 6; i++ )
      jamp2[4][i] = 0.; 
    for(int i = 0; i < 6; i++ )
      jamp2[5][i] = 0.; 
    for(int i = 0; i < 6; i++ )
      jamp2[6][i] = 0.; 

    if (proc_ID == 0)
      t = matrix_6_uu_ttxuu(); 
    if (proc_ID == 1)
      t = matrix_6_uux_ttxuux(); 
    if (proc_ID == 2)
      t = matrix_6_uxux_ttxuxux(); 
    if (proc_ID == 3)
      t = matrix_6_uc_ttxuc(); 
    if (proc_ID == 4)
      t = matrix_6_uux_ttxccx(); 
    if (proc_ID == 5)
      t = matrix_6_ucx_ttxucx(); 
    if (proc_ID == 6)
      t = matrix_6_uxcx_ttxuxcx(); 

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

void PY8MEs_R6_P5_sm_qq_ttxqq::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  // Calculate all wavefunctions
  ixxxxx(p[perm[0]], mME[0], hel[0], +1, w[0]); 
  ixxxxx(p[perm[1]], mME[1], hel[1], +1, w[1]); 
  oxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  ixxxxx(p[perm[3]], mME[3], hel[3], -1, w[3]); 
  oxxxxx(p[perm[4]], mME[4], hel[4], +1, w[4]); 
  oxxxxx(p[perm[5]], mME[5], hel[5], +1, w[5]); 
  FFV1P0_3(w[0], w[4], pars->GC_11, pars->ZERO, pars->ZERO, w[6]); 
  FFV1P0_3(w[1], w[5], pars->GC_11, pars->ZERO, pars->ZERO, w[7]); 
  FFV1_1(w[2], w[6], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[8]); 
  FFV1_2(w[3], w[6], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[9]); 
  FFV1P0_3(w[3], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[10]); 
  FFV1_2(w[1], w[6], pars->GC_11, pars->ZERO, pars->ZERO, w[11]); 
  FFV1_1(w[5], w[6], pars->GC_11, pars->ZERO, pars->ZERO, w[12]); 
  FFV1P0_3(w[0], w[5], pars->GC_11, pars->ZERO, pars->ZERO, w[13]); 
  FFV1P0_3(w[1], w[4], pars->GC_11, pars->ZERO, pars->ZERO, w[14]); 
  FFV1_1(w[2], w[13], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[15]); 
  FFV1_2(w[3], w[13], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[16]); 
  FFV1_2(w[1], w[13], pars->GC_11, pars->ZERO, pars->ZERO, w[17]); 
  FFV1_1(w[4], w[13], pars->GC_11, pars->ZERO, pars->ZERO, w[18]); 
  FFV1_2(w[0], w[14], pars->GC_11, pars->ZERO, pars->ZERO, w[19]); 
  FFV1_2(w[0], w[10], pars->GC_11, pars->ZERO, pars->ZERO, w[20]); 
  FFV1_2(w[0], w[7], pars->GC_11, pars->ZERO, pars->ZERO, w[21]); 
  oxxxxx(p[perm[1]], mME[1], hel[1], -1, w[22]); 
  ixxxxx(p[perm[5]], mME[5], hel[5], -1, w[23]); 
  FFV1P0_3(w[0], w[22], pars->GC_11, pars->ZERO, pars->ZERO, w[24]); 
  FFV1P0_3(w[23], w[4], pars->GC_11, pars->ZERO, pars->ZERO, w[25]); 
  FFV1_1(w[2], w[24], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[26]); 
  FFV1_2(w[3], w[24], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[27]); 
  FFV1_2(w[23], w[24], pars->GC_11, pars->ZERO, pars->ZERO, w[28]); 
  FFV1_1(w[4], w[24], pars->GC_11, pars->ZERO, pars->ZERO, w[29]); 
  FFV1P0_3(w[23], w[22], pars->GC_11, pars->ZERO, pars->ZERO, w[30]); 
  FFV1_2(w[23], w[6], pars->GC_11, pars->ZERO, pars->ZERO, w[31]); 
  FFV1_1(w[22], w[6], pars->GC_11, pars->ZERO, pars->ZERO, w[32]); 
  FFV1_2(w[0], w[30], pars->GC_11, pars->ZERO, pars->ZERO, w[33]); 
  FFV1_2(w[0], w[25], pars->GC_11, pars->ZERO, pars->ZERO, w[34]); 
  oxxxxx(p[perm[0]], mME[0], hel[0], -1, w[35]); 
  ixxxxx(p[perm[4]], mME[4], hel[4], -1, w[36]); 
  FFV1P0_3(w[36], w[35], pars->GC_11, pars->ZERO, pars->ZERO, w[37]); 
  FFV1_1(w[2], w[37], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[38]); 
  FFV1_2(w[3], w[37], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[39]); 
  FFV1_2(w[23], w[37], pars->GC_11, pars->ZERO, pars->ZERO, w[40]); 
  FFV1_1(w[22], w[37], pars->GC_11, pars->ZERO, pars->ZERO, w[41]); 
  FFV1P0_3(w[36], w[22], pars->GC_11, pars->ZERO, pars->ZERO, w[42]); 
  FFV1P0_3(w[23], w[35], pars->GC_11, pars->ZERO, pars->ZERO, w[43]); 
  FFV1_1(w[2], w[42], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[44]); 
  FFV1_2(w[3], w[42], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[45]); 
  FFV1_2(w[23], w[42], pars->GC_11, pars->ZERO, pars->ZERO, w[46]); 
  FFV1_1(w[35], w[42], pars->GC_11, pars->ZERO, pars->ZERO, w[47]); 
  FFV1_2(w[36], w[43], pars->GC_11, pars->ZERO, pars->ZERO, w[48]); 
  FFV1_2(w[36], w[10], pars->GC_11, pars->ZERO, pars->ZERO, w[49]); 
  FFV1_2(w[36], w[30], pars->GC_11, pars->ZERO, pars->ZERO, w[50]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[3], w[8], w[7], pars->GC_11, amp[0]); 
  FFV1_0(w[9], w[2], w[7], pars->GC_11, amp[1]); 
  VVV1_0(w[6], w[7], w[10], pars->GC_10, amp[2]); 
  FFV1_0(w[11], w[5], w[10], pars->GC_11, amp[3]); 
  FFV1_0(w[1], w[12], w[10], pars->GC_11, amp[4]); 
  FFV1_0(w[3], w[15], w[14], pars->GC_11, amp[5]); 
  FFV1_0(w[16], w[2], w[14], pars->GC_11, amp[6]); 
  VVV1_0(w[13], w[14], w[10], pars->GC_10, amp[7]); 
  FFV1_0(w[17], w[4], w[10], pars->GC_11, amp[8]); 
  FFV1_0(w[1], w[18], w[10], pars->GC_11, amp[9]); 
  FFV1_0(w[19], w[5], w[10], pars->GC_11, amp[10]); 
  FFV1_0(w[20], w[5], w[14], pars->GC_11, amp[11]); 
  FFV1_0(w[21], w[4], w[10], pars->GC_11, amp[12]); 
  FFV1_0(w[20], w[4], w[7], pars->GC_11, amp[13]); 
  FFV1_0(w[3], w[26], w[25], pars->GC_11, amp[14]); 
  FFV1_0(w[27], w[2], w[25], pars->GC_11, amp[15]); 
  VVV1_0(w[24], w[25], w[10], pars->GC_10, amp[16]); 
  FFV1_0(w[28], w[4], w[10], pars->GC_11, amp[17]); 
  FFV1_0(w[23], w[29], w[10], pars->GC_11, amp[18]); 
  FFV1_0(w[3], w[8], w[30], pars->GC_11, amp[19]); 
  FFV1_0(w[9], w[2], w[30], pars->GC_11, amp[20]); 
  VVV1_0(w[6], w[30], w[10], pars->GC_10, amp[21]); 
  FFV1_0(w[31], w[22], w[10], pars->GC_11, amp[22]); 
  FFV1_0(w[23], w[32], w[10], pars->GC_11, amp[23]); 
  FFV1_0(w[33], w[4], w[10], pars->GC_11, amp[24]); 
  FFV1_0(w[20], w[4], w[30], pars->GC_11, amp[25]); 
  FFV1_0(w[34], w[22], w[10], pars->GC_11, amp[26]); 
  FFV1_0(w[20], w[22], w[25], pars->GC_11, amp[27]); 
  FFV1_0(w[3], w[38], w[30], pars->GC_11, amp[28]); 
  FFV1_0(w[39], w[2], w[30], pars->GC_11, amp[29]); 
  VVV1_0(w[37], w[30], w[10], pars->GC_10, amp[30]); 
  FFV1_0(w[40], w[22], w[10], pars->GC_11, amp[31]); 
  FFV1_0(w[23], w[41], w[10], pars->GC_11, amp[32]); 
  FFV1_0(w[3], w[44], w[43], pars->GC_11, amp[33]); 
  FFV1_0(w[45], w[2], w[43], pars->GC_11, amp[34]); 
  VVV1_0(w[42], w[43], w[10], pars->GC_10, amp[35]); 
  FFV1_0(w[46], w[35], w[10], pars->GC_11, amp[36]); 
  FFV1_0(w[23], w[47], w[10], pars->GC_11, amp[37]); 
  FFV1_0(w[48], w[22], w[10], pars->GC_11, amp[38]); 
  FFV1_0(w[49], w[22], w[43], pars->GC_11, amp[39]); 
  FFV1_0(w[50], w[35], w[10], pars->GC_11, amp[40]); 
  FFV1_0(w[49], w[35], w[30], pars->GC_11, amp[41]); 
  FFV1_0(w[21], w[4], w[10], pars->GC_11, amp[42]); 
  FFV1_0(w[20], w[4], w[7], pars->GC_11, amp[43]); 
  FFV1_0(w[3], w[26], w[25], pars->GC_11, amp[44]); 
  FFV1_0(w[27], w[2], w[25], pars->GC_11, amp[45]); 
  VVV1_0(w[24], w[25], w[10], pars->GC_10, amp[46]); 
  FFV1_0(w[28], w[4], w[10], pars->GC_11, amp[47]); 
  FFV1_0(w[23], w[29], w[10], pars->GC_11, amp[48]); 
  FFV1_0(w[34], w[22], w[10], pars->GC_11, amp[49]); 
  FFV1_0(w[20], w[22], w[25], pars->GC_11, amp[50]); 
  FFV1_0(w[3], w[8], w[30], pars->GC_11, amp[51]); 
  FFV1_0(w[9], w[2], w[30], pars->GC_11, amp[52]); 
  VVV1_0(w[6], w[30], w[10], pars->GC_10, amp[53]); 
  FFV1_0(w[31], w[22], w[10], pars->GC_11, amp[54]); 
  FFV1_0(w[23], w[32], w[10], pars->GC_11, amp[55]); 
  FFV1_0(w[33], w[4], w[10], pars->GC_11, amp[56]); 
  FFV1_0(w[20], w[4], w[30], pars->GC_11, amp[57]); 
  FFV1_0(w[3], w[38], w[30], pars->GC_11, amp[58]); 
  FFV1_0(w[39], w[2], w[30], pars->GC_11, amp[59]); 
  VVV1_0(w[37], w[30], w[10], pars->GC_10, amp[60]); 
  FFV1_0(w[40], w[22], w[10], pars->GC_11, amp[61]); 
  FFV1_0(w[23], w[41], w[10], pars->GC_11, amp[62]); 
  FFV1_0(w[50], w[35], w[10], pars->GC_11, amp[63]); 
  FFV1_0(w[49], w[35], w[30], pars->GC_11, amp[64]); 


}
double PY8MEs_R6_P5_sm_qq_ttxqq::matrix_6_uu_ttxuu() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 14; 
  const int ncolor = 6; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1, 1, 1, 1, 1}; 
  double cf[ncolor][ncolor] = {{27, 9, 9, 3, 3, 9}, {9, 27, 3, 9, 9, 3}, {9, 3,
      27, 9, 9, 3}, {3, 9, 9, 27, 3, 9}, {3, 9, 9, 3, 27, 9}, {9, 3, 3, 9, 9,
      27}};

  // Calculate color flows
  jamp[0] = +1./4. * (+amp[0] - std::complex<double> (0, 1) * amp[2] + amp[3] +
      1./3. * amp[5] + 1./3. * amp[6] + 1./3. * amp[10] + 1./3. * amp[11] +
      amp[13]);
  jamp[1] = +1./4. * (-1./3. * amp[0] - 1./3. * amp[1] - amp[5] +
      std::complex<double> (0, 1) * amp[7] - amp[8] - amp[11] - 1./3. * amp[12]
      - 1./3. * amp[13]);
  jamp[2] = +1./4. * (-1./3. * amp[0] - 1./3. * amp[1] - 1./3. * amp[3] - 1./3.
      * amp[4] - amp[6] - std::complex<double> (0, 1) * amp[7] - amp[9] -
      amp[10]);
  jamp[3] = +1./4. * (+amp[1] + std::complex<double> (0, 1) * amp[2] + amp[4] +
      1./3. * amp[5] + 1./3. * amp[6] + 1./3. * amp[8] + 1./3. * amp[9] +
      amp[12]);
  jamp[4] = +1./4. * (+1./9. * amp[0] + 1./9. * amp[1] + 1./9. * amp[3] + 1./9.
      * amp[4] + 1./3. * amp[8] + 1./3. * amp[9] + 1./3. * amp[10] + 1./3. *
      amp[11] + 1./9. * amp[12] + 1./9. * amp[13]);
  jamp[5] = +1./4. * (-1./3. * amp[3] - 1./3. * amp[4] - 1./9. * amp[5] - 1./9.
      * amp[6] - 1./9. * amp[8] - 1./9. * amp[9] - 1./9. * amp[10] - 1./9. *
      amp[11] - 1./3. * amp[12] - 1./3. * amp[13]);

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

double PY8MEs_R6_P5_sm_qq_ttxqq::matrix_6_uux_ttxuux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 14; 
  const int ncolor = 6; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1, 1, 1, 1, 1}; 
  double cf[ncolor][ncolor] = {{27, 9, 9, 3, 3, 9}, {9, 27, 3, 9, 9, 3}, {9, 3,
      27, 9, 9, 3}, {3, 9, 9, 27, 3, 9}, {3, 9, 9, 3, 27, 9}, {9, 3, 3, 9, 9,
      27}};

  // Calculate color flows
  jamp[0] = +1./4. * (+1./9. * amp[14] + 1./9. * amp[15] + 1./9. * amp[17] +
      1./9. * amp[18] + 1./3. * amp[22] + 1./3. * amp[23] + 1./3. * amp[24] +
      1./3. * amp[25] + 1./9. * amp[26] + 1./9. * amp[27]);
  jamp[1] = +1./4. * (-1./3. * amp[14] - 1./3. * amp[15] - 1./3. * amp[17] -
      1./3. * amp[18] - amp[20] - std::complex<double> (0, 1) * amp[21] -
      amp[23] - amp[24]);
  jamp[2] = +1./4. * (-1./3. * amp[14] - 1./3. * amp[15] - amp[19] +
      std::complex<double> (0, 1) * amp[21] - amp[22] - amp[25] - 1./3. *
      amp[26] - 1./3. * amp[27]);
  jamp[3] = +1./4. * (+amp[15] + std::complex<double> (0, 1) * amp[16] +
      amp[18] + 1./3. * amp[19] + 1./3. * amp[20] + 1./3. * amp[22] + 1./3. *
      amp[23] + amp[26]);
  jamp[4] = +1./4. * (+amp[14] - std::complex<double> (0, 1) * amp[16] +
      amp[17] + 1./3. * amp[19] + 1./3. * amp[20] + 1./3. * amp[24] + 1./3. *
      amp[25] + amp[27]);
  jamp[5] = +1./4. * (-1./3. * amp[17] - 1./3. * amp[18] - 1./9. * amp[19] -
      1./9. * amp[20] - 1./9. * amp[22] - 1./9. * amp[23] - 1./9. * amp[24] -
      1./9. * amp[25] - 1./3. * amp[26] - 1./3. * amp[27]);

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

double PY8MEs_R6_P5_sm_qq_ttxqq::matrix_6_uxux_ttxuxux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 14; 
  const int ncolor = 6; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1, 1, 1, 1, 1}; 
  double cf[ncolor][ncolor] = {{27, 9, 9, 3, 3, 9}, {9, 27, 3, 9, 9, 3}, {9, 3,
      27, 9, 9, 3}, {3, 9, 9, 27, 3, 9}, {3, 9, 9, 3, 27, 9}, {9, 3, 3, 9, 9,
      27}};

  // Calculate color flows
  jamp[0] = +1./4. * (+amp[29] + std::complex<double> (0, 1) * amp[30] +
      amp[32] + 1./3. * amp[33] + 1./3. * amp[34] + 1./3. * amp[36] + 1./3. *
      amp[37] + amp[40]);
  jamp[1] = +1./4. * (-1./3. * amp[28] - 1./3. * amp[29] - amp[33] +
      std::complex<double> (0, 1) * amp[35] - amp[36] - amp[39] - 1./3. *
      amp[40] - 1./3. * amp[41]);
  jamp[2] = +1./4. * (-1./3. * amp[28] - 1./3. * amp[29] - 1./3. * amp[31] -
      1./3. * amp[32] - amp[34] - std::complex<double> (0, 1) * amp[35] -
      amp[37] - amp[38]);
  jamp[3] = +1./4. * (+1./9. * amp[28] + 1./9. * amp[29] + 1./9. * amp[31] +
      1./9. * amp[32] + 1./3. * amp[36] + 1./3. * amp[37] + 1./3. * amp[38] +
      1./3. * amp[39] + 1./9. * amp[40] + 1./9. * amp[41]);
  jamp[4] = +1./4. * (+amp[28] - std::complex<double> (0, 1) * amp[30] +
      amp[31] + 1./3. * amp[33] + 1./3. * amp[34] + 1./3. * amp[38] + 1./3. *
      amp[39] + amp[41]);
  jamp[5] = +1./4. * (-1./3. * amp[31] - 1./3. * amp[32] - 1./9. * amp[33] -
      1./9. * amp[34] - 1./9. * amp[36] - 1./9. * amp[37] - 1./9. * amp[38] -
      1./9. * amp[39] - 1./3. * amp[40] - 1./3. * amp[41]);

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

double PY8MEs_R6_P5_sm_qq_ttxqq::matrix_6_uc_ttxuc() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 7; 
  const int ncolor = 6; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1, 1, 1, 1, 1}; 
  double cf[ncolor][ncolor] = {{27, 9, 9, 3, 3, 9}, {9, 27, 3, 9, 9, 3}, {9, 3,
      27, 9, 9, 3}, {3, 9, 9, 27, 3, 9}, {3, 9, 9, 3, 27, 9}, {9, 3, 3, 9, 9,
      27}};

  // Calculate color flows
  jamp[0] = +1./4. * (+amp[0] - std::complex<double> (0, 1) * amp[2] + amp[3] +
      amp[43]);
  jamp[1] = +1./4. * (-1./3. * amp[0] - 1./3. * amp[1] - 1./3. * amp[42] -
      1./3. * amp[43]);
  jamp[2] = +1./4. * (-1./3. * amp[0] - 1./3. * amp[1] - 1./3. * amp[3] - 1./3.
      * amp[4]);
  jamp[3] = +1./4. * (+amp[1] + std::complex<double> (0, 1) * amp[2] + amp[4] +
      amp[42]);
  jamp[4] = +1./4. * (+1./9. * amp[0] + 1./9. * amp[1] + 1./9. * amp[3] + 1./9.
      * amp[4] + 1./9. * amp[42] + 1./9. * amp[43]);
  jamp[5] = +1./4. * (-1./3. * amp[3] - 1./3. * amp[4] - 1./3. * amp[42] -
      1./3. * amp[43]);

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

double PY8MEs_R6_P5_sm_qq_ttxqq::matrix_6_uux_ttxccx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 7; 
  const int ncolor = 6; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1, 1, 1, 1, 1}; 
  double cf[ncolor][ncolor] = {{27, 9, 9, 3, 3, 9}, {9, 27, 3, 9, 9, 3}, {9, 3,
      27, 9, 9, 3}, {3, 9, 9, 27, 3, 9}, {3, 9, 9, 3, 27, 9}, {9, 3, 3, 9, 9,
      27}};

  // Calculate color flows
  jamp[0] = +1./4. * (+1./9. * amp[44] + 1./9. * amp[45] + 1./9. * amp[47] +
      1./9. * amp[48] + 1./9. * amp[49] + 1./9. * amp[50]);
  jamp[1] = +1./4. * (-1./3. * amp[44] - 1./3. * amp[45] - 1./3. * amp[47] -
      1./3. * amp[48]);
  jamp[2] = +1./4. * (-1./3. * amp[44] - 1./3. * amp[45] - 1./3. * amp[49] -
      1./3. * amp[50]);
  jamp[3] = +1./4. * (+amp[45] + std::complex<double> (0, 1) * amp[46] +
      amp[48] + amp[49]);
  jamp[4] = +1./4. * (+amp[44] - std::complex<double> (0, 1) * amp[46] +
      amp[47] + amp[50]);
  jamp[5] = +1./4. * (-1./3. * amp[47] - 1./3. * amp[48] - 1./3. * amp[49] -
      1./3. * amp[50]);

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
    jamp2[4][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R6_P5_sm_qq_ttxqq::matrix_6_ucx_ttxucx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 7; 
  const int ncolor = 6; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1, 1, 1, 1, 1}; 
  double cf[ncolor][ncolor] = {{27, 9, 9, 3, 3, 9}, {9, 27, 3, 9, 9, 3}, {9, 3,
      27, 9, 9, 3}, {3, 9, 9, 27, 3, 9}, {3, 9, 9, 3, 27, 9}, {9, 3, 3, 9, 9,
      27}};

  // Calculate color flows
  jamp[0] = +1./4. * (+1./3. * amp[54] + 1./3. * amp[55] + 1./3. * amp[56] +
      1./3. * amp[57]);
  jamp[1] = +1./4. * (-amp[52] - std::complex<double> (0, 1) * amp[53] -
      amp[55] - amp[56]);
  jamp[2] = +1./4. * (-amp[51] + std::complex<double> (0, 1) * amp[53] -
      amp[54] - amp[57]);
  jamp[3] = +1./4. * (+1./3. * amp[51] + 1./3. * amp[52] + 1./3. * amp[54] +
      1./3. * amp[55]);
  jamp[4] = +1./4. * (+1./3. * amp[51] + 1./3. * amp[52] + 1./3. * amp[56] +
      1./3. * amp[57]);
  jamp[5] = +1./4. * (-1./9. * amp[51] - 1./9. * amp[52] - 1./9. * amp[54] -
      1./9. * amp[55] - 1./9. * amp[56] - 1./9. * amp[57]);

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
    jamp2[5][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R6_P5_sm_qq_ttxqq::matrix_6_uxcx_ttxuxcx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 7; 
  const int ncolor = 6; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1, 1, 1, 1, 1}; 
  double cf[ncolor][ncolor] = {{27, 9, 9, 3, 3, 9}, {9, 27, 3, 9, 9, 3}, {9, 3,
      27, 9, 9, 3}, {3, 9, 9, 27, 3, 9}, {3, 9, 9, 3, 27, 9}, {9, 3, 3, 9, 9,
      27}};

  // Calculate color flows
  jamp[0] = +1./4. * (+amp[59] + std::complex<double> (0, 1) * amp[60] +
      amp[62] + amp[63]);
  jamp[1] = +1./4. * (-1./3. * amp[58] - 1./3. * amp[59] - 1./3. * amp[63] -
      1./3. * amp[64]);
  jamp[2] = +1./4. * (-1./3. * amp[58] - 1./3. * amp[59] - 1./3. * amp[61] -
      1./3. * amp[62]);
  jamp[3] = +1./4. * (+1./9. * amp[58] + 1./9. * amp[59] + 1./9. * amp[61] +
      1./9. * amp[62] + 1./9. * amp[63] + 1./9. * amp[64]);
  jamp[4] = +1./4. * (+amp[58] - std::complex<double> (0, 1) * amp[60] +
      amp[61] + amp[64]);
  jamp[5] = +1./4. * (-1./3. * amp[61] - 1./3. * amp[62] - 1./3. * amp[63] -
      1./3. * amp[64]);

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
    jamp2[6][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}


}  // end namespace PY8MEs_namespace

