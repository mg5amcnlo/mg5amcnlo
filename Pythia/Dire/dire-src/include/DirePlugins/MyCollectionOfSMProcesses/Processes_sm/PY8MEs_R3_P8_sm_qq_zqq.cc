//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.5.3, 2017-03-09
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "PY8MEs_R3_P8_sm_qq_zqq.h"
#include "HelAmps_sm.h"

using namespace Pythia8_sm; 

namespace PY8MEs_namespace 
{
//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u > z u u WEIGHTED<=4 @3
// Process: c c > z c c WEIGHTED<=4 @3
// Process: u u~ > z u u~ WEIGHTED<=4 @3
// Process: c c~ > z c c~ WEIGHTED<=4 @3
// Process: d d > z d d WEIGHTED<=4 @3
// Process: s s > z s s WEIGHTED<=4 @3
// Process: d d~ > z d d~ WEIGHTED<=4 @3
// Process: s s~ > z s s~ WEIGHTED<=4 @3
// Process: u~ u~ > z u~ u~ WEIGHTED<=4 @3
// Process: c~ c~ > z c~ c~ WEIGHTED<=4 @3
// Process: d~ d~ > z d~ d~ WEIGHTED<=4 @3
// Process: s~ s~ > z s~ s~ WEIGHTED<=4 @3
// Process: u c > z u c WEIGHTED<=4 @3
// Process: u d > z u d WEIGHTED<=4 @3
// Process: u s > z u s WEIGHTED<=4 @3
// Process: c d > z c d WEIGHTED<=4 @3
// Process: c s > z c s WEIGHTED<=4 @3
// Process: u u~ > z c c~ WEIGHTED<=4 @3
// Process: c c~ > z u u~ WEIGHTED<=4 @3
// Process: u u~ > z d d~ WEIGHTED<=4 @3
// Process: u u~ > z s s~ WEIGHTED<=4 @3
// Process: c c~ > z d d~ WEIGHTED<=4 @3
// Process: c c~ > z s s~ WEIGHTED<=4 @3
// Process: u c~ > z u c~ WEIGHTED<=4 @3
// Process: c u~ > z c u~ WEIGHTED<=4 @3
// Process: u d~ > z u d~ WEIGHTED<=4 @3
// Process: u s~ > z u s~ WEIGHTED<=4 @3
// Process: c d~ > z c d~ WEIGHTED<=4 @3
// Process: c s~ > z c s~ WEIGHTED<=4 @3
// Process: d s > z d s WEIGHTED<=4 @3
// Process: d u~ > z d u~ WEIGHTED<=4 @3
// Process: d c~ > z d c~ WEIGHTED<=4 @3
// Process: s u~ > z s u~ WEIGHTED<=4 @3
// Process: s c~ > z s c~ WEIGHTED<=4 @3
// Process: d d~ > z u u~ WEIGHTED<=4 @3
// Process: d d~ > z c c~ WEIGHTED<=4 @3
// Process: s s~ > z u u~ WEIGHTED<=4 @3
// Process: s s~ > z c c~ WEIGHTED<=4 @3
// Process: d d~ > z s s~ WEIGHTED<=4 @3
// Process: s s~ > z d d~ WEIGHTED<=4 @3
// Process: d s~ > z d s~ WEIGHTED<=4 @3
// Process: s d~ > z s d~ WEIGHTED<=4 @3
// Process: u~ c~ > z u~ c~ WEIGHTED<=4 @3
// Process: u~ d~ > z u~ d~ WEIGHTED<=4 @3
// Process: u~ s~ > z u~ s~ WEIGHTED<=4 @3
// Process: c~ d~ > z c~ d~ WEIGHTED<=4 @3
// Process: c~ s~ > z c~ s~ WEIGHTED<=4 @3
// Process: d~ s~ > z d~ s~ WEIGHTED<=4 @3

// Exception class
class PY8MEs_R3_P8_sm_qq_zqqException : public exception
{
  virtual const char * what() const throw()
  {
    return "Exception in class 'PY8MEs_R3_P8_sm_qq_zqq'."; 
  }
}
PY8MEs_R3_P8_sm_qq_zqq_exception; 

// Required s-channel initialization
//int PY8MEs_R3_P8_sm_qq_zqq::req_s_channels[nreq_s_channels] = {}; 

int PY8MEs_R3_P8_sm_qq_zqq::helicities[ncomb][nexternal] = {{-1, -1, -1, -1,
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
void PY8MEs_R3_P8_sm_qq_zqq::initColorConfigs() 
{
  color_configs = vector < vec_vec_int > (); 

  // Color flows of process Process: u u > z u u WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp00[2 * nexternal] = {1, 0, 2, 0, 0, 0, 1, 0, 2, 0}; 
  color_configs[0].push_back(vec_int(jamp00, jamp00 + (2 * nexternal))); 
  // JAMP #1
  int jamp01[2 * nexternal] = {2, 0, 1, 0, 0, 0, 1, 0, 2, 0}; 
  color_configs[0].push_back(vec_int(jamp01, jamp01 + (2 * nexternal))); 

  // Color flows of process Process: u u~ > z u u~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp10[2 * nexternal] = {1, 0, 0, 1, 0, 0, 2, 0, 0, 2}; 
  color_configs[1].push_back(vec_int(jamp10, jamp10 + (2 * nexternal))); 
  // JAMP #1
  int jamp11[2 * nexternal] = {2, 0, 0, 1, 0, 0, 2, 0, 0, 1}; 
  color_configs[1].push_back(vec_int(jamp11, jamp11 + (2 * nexternal))); 

  // Color flows of process Process: d d > z d d WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp20[2 * nexternal] = {1, 0, 2, 0, 0, 0, 1, 0, 2, 0}; 
  color_configs[2].push_back(vec_int(jamp20, jamp20 + (2 * nexternal))); 
  // JAMP #1
  int jamp21[2 * nexternal] = {2, 0, 1, 0, 0, 0, 1, 0, 2, 0}; 
  color_configs[2].push_back(vec_int(jamp21, jamp21 + (2 * nexternal))); 

  // Color flows of process Process: d d~ > z d d~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp30[2 * nexternal] = {1, 0, 0, 1, 0, 0, 2, 0, 0, 2}; 
  color_configs[3].push_back(vec_int(jamp30, jamp30 + (2 * nexternal))); 
  // JAMP #1
  int jamp31[2 * nexternal] = {2, 0, 0, 1, 0, 0, 2, 0, 0, 1}; 
  color_configs[3].push_back(vec_int(jamp31, jamp31 + (2 * nexternal))); 

  // Color flows of process Process: u~ u~ > z u~ u~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp40[2 * nexternal] = {0, 1, 0, 2, 0, 0, 0, 1, 0, 2}; 
  color_configs[4].push_back(vec_int(jamp40, jamp40 + (2 * nexternal))); 
  // JAMP #1
  int jamp41[2 * nexternal] = {0, 1, 0, 2, 0, 0, 0, 2, 0, 1}; 
  color_configs[4].push_back(vec_int(jamp41, jamp41 + (2 * nexternal))); 

  // Color flows of process Process: d~ d~ > z d~ d~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp50[2 * nexternal] = {0, 1, 0, 2, 0, 0, 0, 1, 0, 2}; 
  color_configs[5].push_back(vec_int(jamp50, jamp50 + (2 * nexternal))); 
  // JAMP #1
  int jamp51[2 * nexternal] = {0, 1, 0, 2, 0, 0, 0, 2, 0, 1}; 
  color_configs[5].push_back(vec_int(jamp51, jamp51 + (2 * nexternal))); 

  // Color flows of process Process: u c > z u c WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp60[2 * nexternal] = {1, 0, 2, 0, 0, 0, 1, 0, 2, 0}; 
  color_configs[6].push_back(vec_int(jamp60, jamp60 + (2 * nexternal))); 
  // JAMP #1
  int jamp61[2 * nexternal] = {2, 0, 1, 0, 0, 0, 1, 0, 2, 0}; 
  color_configs[6].push_back(vec_int(jamp61, jamp61 + (2 * nexternal))); 

  // Color flows of process Process: u d > z u d WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp70[2 * nexternal] = {1, 0, 2, 0, 0, 0, 1, 0, 2, 0}; 
  color_configs[7].push_back(vec_int(jamp70, jamp70 + (2 * nexternal))); 
  // JAMP #1
  int jamp71[2 * nexternal] = {2, 0, 1, 0, 0, 0, 1, 0, 2, 0}; 
  color_configs[7].push_back(vec_int(jamp71, jamp71 + (2 * nexternal))); 

  // Color flows of process Process: u u~ > z c c~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp80[2 * nexternal] = {1, 0, 0, 1, 0, 0, 2, 0, 0, 2}; 
  color_configs[8].push_back(vec_int(jamp80, jamp80 + (2 * nexternal))); 
  // JAMP #1
  int jamp81[2 * nexternal] = {2, 0, 0, 1, 0, 0, 2, 0, 0, 1}; 
  color_configs[8].push_back(vec_int(jamp81, jamp81 + (2 * nexternal))); 

  // Color flows of process Process: u u~ > z d d~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp90[2 * nexternal] = {1, 0, 0, 1, 0, 0, 2, 0, 0, 2}; 
  color_configs[9].push_back(vec_int(jamp90, jamp90 + (2 * nexternal))); 
  // JAMP #1
  int jamp91[2 * nexternal] = {2, 0, 0, 1, 0, 0, 2, 0, 0, 1}; 
  color_configs[9].push_back(vec_int(jamp91, jamp91 + (2 * nexternal))); 

  // Color flows of process Process: u c~ > z u c~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp100[2 * nexternal] = {1, 0, 0, 1, 0, 0, 2, 0, 0, 2}; 
  color_configs[10].push_back(vec_int(jamp100, jamp100 + (2 * nexternal))); 
  // JAMP #1
  int jamp101[2 * nexternal] = {2, 0, 0, 1, 0, 0, 2, 0, 0, 1}; 
  color_configs[10].push_back(vec_int(jamp101, jamp101 + (2 * nexternal))); 

  // Color flows of process Process: u d~ > z u d~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp110[2 * nexternal] = {1, 0, 0, 1, 0, 0, 2, 0, 0, 2}; 
  color_configs[11].push_back(vec_int(jamp110, jamp110 + (2 * nexternal))); 
  // JAMP #1
  int jamp111[2 * nexternal] = {2, 0, 0, 1, 0, 0, 2, 0, 0, 1}; 
  color_configs[11].push_back(vec_int(jamp111, jamp111 + (2 * nexternal))); 

  // Color flows of process Process: d s > z d s WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp120[2 * nexternal] = {1, 0, 2, 0, 0, 0, 1, 0, 2, 0}; 
  color_configs[12].push_back(vec_int(jamp120, jamp120 + (2 * nexternal))); 
  // JAMP #1
  int jamp121[2 * nexternal] = {2, 0, 1, 0, 0, 0, 1, 0, 2, 0}; 
  color_configs[12].push_back(vec_int(jamp121, jamp121 + (2 * nexternal))); 

  // Color flows of process Process: d u~ > z d u~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp130[2 * nexternal] = {1, 0, 0, 1, 0, 0, 2, 0, 0, 2}; 
  color_configs[13].push_back(vec_int(jamp130, jamp130 + (2 * nexternal))); 
  // JAMP #1
  int jamp131[2 * nexternal] = {2, 0, 0, 1, 0, 0, 2, 0, 0, 1}; 
  color_configs[13].push_back(vec_int(jamp131, jamp131 + (2 * nexternal))); 

  // Color flows of process Process: d d~ > z u u~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp140[2 * nexternal] = {1, 0, 0, 1, 0, 0, 2, 0, 0, 2}; 
  color_configs[14].push_back(vec_int(jamp140, jamp140 + (2 * nexternal))); 
  // JAMP #1
  int jamp141[2 * nexternal] = {2, 0, 0, 1, 0, 0, 2, 0, 0, 1}; 
  color_configs[14].push_back(vec_int(jamp141, jamp141 + (2 * nexternal))); 

  // Color flows of process Process: d d~ > z s s~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp150[2 * nexternal] = {1, 0, 0, 1, 0, 0, 2, 0, 0, 2}; 
  color_configs[15].push_back(vec_int(jamp150, jamp150 + (2 * nexternal))); 
  // JAMP #1
  int jamp151[2 * nexternal] = {2, 0, 0, 1, 0, 0, 2, 0, 0, 1}; 
  color_configs[15].push_back(vec_int(jamp151, jamp151 + (2 * nexternal))); 

  // Color flows of process Process: d s~ > z d s~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp160[2 * nexternal] = {1, 0, 0, 1, 0, 0, 2, 0, 0, 2}; 
  color_configs[16].push_back(vec_int(jamp160, jamp160 + (2 * nexternal))); 
  // JAMP #1
  int jamp161[2 * nexternal] = {2, 0, 0, 1, 0, 0, 2, 0, 0, 1}; 
  color_configs[16].push_back(vec_int(jamp161, jamp161 + (2 * nexternal))); 

  // Color flows of process Process: u~ c~ > z u~ c~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp170[2 * nexternal] = {0, 1, 0, 2, 0, 0, 0, 1, 0, 2}; 
  color_configs[17].push_back(vec_int(jamp170, jamp170 + (2 * nexternal))); 
  // JAMP #1
  int jamp171[2 * nexternal] = {0, 1, 0, 2, 0, 0, 0, 2, 0, 1}; 
  color_configs[17].push_back(vec_int(jamp171, jamp171 + (2 * nexternal))); 

  // Color flows of process Process: u~ d~ > z u~ d~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp180[2 * nexternal] = {0, 1, 0, 2, 0, 0, 0, 1, 0, 2}; 
  color_configs[18].push_back(vec_int(jamp180, jamp180 + (2 * nexternal))); 
  // JAMP #1
  int jamp181[2 * nexternal] = {0, 1, 0, 2, 0, 0, 0, 2, 0, 1}; 
  color_configs[18].push_back(vec_int(jamp181, jamp181 + (2 * nexternal))); 

  // Color flows of process Process: d~ s~ > z d~ s~ WEIGHTED<=4 @3
  color_configs.push_back(vec_vec_int()); 
  // JAMP #0
  int jamp190[2 * nexternal] = {0, 1, 0, 2, 0, 0, 0, 1, 0, 2}; 
  color_configs[19].push_back(vec_int(jamp190, jamp190 + (2 * nexternal))); 
  // JAMP #1
  int jamp191[2 * nexternal] = {0, 1, 0, 2, 0, 0, 0, 2, 0, 1}; 
  color_configs[19].push_back(vec_int(jamp191, jamp191 + (2 * nexternal))); 
}

//--------------------------------------------------------------------------
// Destructor.
PY8MEs_R3_P8_sm_qq_zqq::~PY8MEs_R3_P8_sm_qq_zqq() 
{
  for(int i = 0; i < nexternal; i++ )
  {
    delete[] p[i]; 
    p[i] = NULL; 
  }
}

//--------------------------------------------------------------------------
// Return the list of possible helicity configurations
vector < vec_int > PY8MEs_R3_P8_sm_qq_zqq::getHelicityConfigs(vector<int>
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
vector < vec_int > PY8MEs_R3_P8_sm_qq_zqq::getColorConfigs(int specify_proc_ID,
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
vector<int> PY8MEs_R3_P8_sm_qq_zqq::getHelicityConfigForID(int hel_ID,
    vector<int> permutation)
{
  if (hel_ID < 0 || hel_ID >= ncomb)
  {
    cout <<  "Error in function 'getHelicityConfigForID' of class" << 
    " 'PY8MEs_R3_P8_sm_qq_zqq': Specified helicity ID '" << 
    hel_ID <<  "' cannot be found." << endl; 
    throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
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
int PY8MEs_R3_P8_sm_qq_zqq::getHelicityIDForConfig(vector<int> hel_config,
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
      " 'PY8MEs_R3_P8_sm_qq_zqq': Specified helicity" << 
      " configuration cannot be found." << endl; 
      throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
    }
  }
  return user_ihel; 
}


//--------------------------------------------------------------------------
// Implements the map Color ID -> Color Config
vector<int> PY8MEs_R3_P8_sm_qq_zqq::getColorConfigForID(int color_ID, int
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
    " 'PY8MEs_R3_P8_sm_qq_zqq': Specified color ID '" << 
    color_ID <<  "' cannot be found." << endl; 
    throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
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
int PY8MEs_R3_P8_sm_qq_zqq::getColorIDForConfig(vector<int> color_config, int
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
          for (int k = 0; k < (nexternal * 2); k++ )
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
            " 'PY8MEs_R3_P8_sm_qq_zqq': A color line could " << 
            " not be closed." << endl;
            return -2; 
            //throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
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
      " 'PY8MEs_R3_P8_sm_qq_zqq': Specified color" << 
      " configuration cannot be found." << endl;
      return -2; 
      //throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
    }
  }
  return user_icol; 
}

//--------------------------------------------------------------------------
// Returns all result previously computed in SigmaKin
vector < vec_double > PY8MEs_R3_P8_sm_qq_zqq::getAllResults(int
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
double PY8MEs_R3_P8_sm_qq_zqq::getResult(int helicity_ID, int color_ID, int
    specify_proc_ID)
{
  if (helicity_ID < - 1 || helicity_ID >= ncomb)
  {
    cout <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R3_P8_sm_qq_zqq': Specified helicity ID '" << 
    helicity_ID <<  "' configuration cannot be found." << endl; 
    throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
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
    " 'PY8MEs_R3_P8_sm_qq_zqq': Specified color ID '" << 
    color_ID <<  "' configuration cannot be found." << endl; 
    throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
  }
  return all_results[chosenProcID][helicity_ID + 1][color_ID + 1]; 
}

//--------------------------------------------------------------------------
// Check for the availability of the requested process and if available,
// If available, this returns the corresponding permutation and Proc_ID to use.
// If not available, this returns a negative Proc_ID.
pair < vector<int> , int > PY8MEs_R3_P8_sm_qq_zqq::static_getPY8ME(vector<int>
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
  const int nprocs = 88; 
  const int proc_IDS[nprocs] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 7,
      7, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11, 11, 12, 13, 13, 13, 13, 14, 14,
      14, 14, 15, 15, 16, 16, 17, 18, 18, 18, 18, 19, 1, 1, 3, 3, 6, 7, 7, 7,
      7, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11, 11, 12, 13, 13, 13, 13, 14, 14,
      14, 14, 15, 15, 16, 16, 17, 18, 18, 18, 18, 19};
  const int in_pdgs[nprocs][ninitial] = {{2, 2}, {4, 4}, {2, -2}, {4, -4}, {1,
      1}, {3, 3}, {1, -1}, {3, -3}, {-2, -2}, {-4, -4}, {-1, -1}, {-3, -3}, {2,
      4}, {2, 1}, {2, 3}, {4, 1}, {4, 3}, {2, -2}, {4, -4}, {2, -2}, {2, -2},
      {4, -4}, {4, -4}, {2, -4}, {4, -2}, {2, -1}, {2, -3}, {4, -1}, {4, -3},
      {1, 3}, {1, -2}, {1, -4}, {3, -2}, {3, -4}, {1, -1}, {1, -1}, {3, -3},
      {3, -3}, {1, -1}, {3, -3}, {1, -3}, {3, -1}, {-2, -4}, {-2, -1}, {-2,
      -3}, {-4, -1}, {-4, -3}, {-1, -3}, {-2, 2}, {-4, 4}, {-1, 1}, {-3, 3},
      {4, 2}, {1, 2}, {3, 2}, {1, 4}, {3, 4}, {-2, 2}, {-4, 4}, {-2, 2}, {-2,
      2}, {-4, 4}, {-4, 4}, {-4, 2}, {-2, 4}, {-1, 2}, {-3, 2}, {-1, 4}, {-3,
      4}, {3, 1}, {-2, 1}, {-4, 1}, {-2, 3}, {-4, 3}, {-1, 1}, {-1, 1}, {-3,
      3}, {-3, 3}, {-1, 1}, {-3, 3}, {-3, 1}, {-1, 3}, {-4, -2}, {-1, -2}, {-3,
      -2}, {-1, -4}, {-3, -4}, {-3, -1}};
  const int out_pdgs[nprocs][nexternal - ninitial] = {{23, 2, 2}, {23, 4, 4},
      {23, 2, -2}, {23, 4, -4}, {23, 1, 1}, {23, 3, 3}, {23, 1, -1}, {23, 3,
      -3}, {23, -2, -2}, {23, -4, -4}, {23, -1, -1}, {23, -3, -3}, {23, 2, 4},
      {23, 2, 1}, {23, 2, 3}, {23, 4, 1}, {23, 4, 3}, {23, 4, -4}, {23, 2, -2},
      {23, 1, -1}, {23, 3, -3}, {23, 1, -1}, {23, 3, -3}, {23, 2, -4}, {23, 4,
      -2}, {23, 2, -1}, {23, 2, -3}, {23, 4, -1}, {23, 4, -3}, {23, 1, 3}, {23,
      1, -2}, {23, 1, -4}, {23, 3, -2}, {23, 3, -4}, {23, 2, -2}, {23, 4, -4},
      {23, 2, -2}, {23, 4, -4}, {23, 3, -3}, {23, 1, -1}, {23, 1, -3}, {23, 3,
      -1}, {23, -2, -4}, {23, -2, -1}, {23, -2, -3}, {23, -4, -1}, {23, -4,
      -3}, {23, -1, -3}, {23, 2, -2}, {23, 4, -4}, {23, 1, -1}, {23, 3, -3},
      {23, 2, 4}, {23, 2, 1}, {23, 2, 3}, {23, 4, 1}, {23, 4, 3}, {23, 4, -4},
      {23, 2, -2}, {23, 1, -1}, {23, 3, -3}, {23, 1, -1}, {23, 3, -3}, {23, 2,
      -4}, {23, 4, -2}, {23, 2, -1}, {23, 2, -3}, {23, 4, -1}, {23, 4, -3},
      {23, 1, 3}, {23, 1, -2}, {23, 1, -4}, {23, 3, -2}, {23, 3, -4}, {23, 2,
      -2}, {23, 4, -4}, {23, 2, -2}, {23, 4, -4}, {23, 3, -3}, {23, 1, -1},
      {23, 1, -3}, {23, 3, -1}, {23, -2, -4}, {23, -2, -1}, {23, -2, -3}, {23,
      -4, -1}, {23, -4, -3}, {23, -1, -3}};

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
void PY8MEs_R3_P8_sm_qq_zqq::setMomenta(vector < vec_double > momenta_picked)
{
  if (momenta_picked.size() != nexternal)
  {
    cout <<  "Error in function 'setMomenta' of class" << 
    " 'PY8MEs_R3_P8_sm_qq_zqq': Incorrect number of" << 
    " momenta specified." << endl; 
    throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
  }
  for(int i = 0; i < nexternal; i++ )
  {
    if (momenta_picked[i].size() != 4)
    {
      cout <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R3_P8_sm_qq_zqq': Incorrect number of" << 
      " momenta components specified." << endl; 
      throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
    }
    if (momenta_picked[i][0] < 0.0)
    {
      cout <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R3_P8_sm_qq_zqq': A momentum was specified" << 
      " with negative energy. Check conventions." << endl; 
      throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
    }
    for (int j = 0; j < 4; j++ )
    {
      p[i][j] = momenta_picked[i][j]; 
    }
  }
}

//--------------------------------------------------------------------------
// Set color configuration to use. An empty vector means sum over all.
void PY8MEs_R3_P8_sm_qq_zqq::setColors(vector<int> colors_picked)
{
  if (colors_picked.size() == 0)
  {
    user_colors = vector<int> (); 
    return; 
  }
  if (colors_picked.size() != (2 * nexternal))
  {
    cout <<  "Error in function 'setColors' of class" << 
    " 'PY8MEs_R3_P8_sm_qq_zqq': Incorrect number" << 
    " of colors specified." << endl; 
    throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
  }
  user_colors = vector<int> ((2 * nexternal), 0); 
  for(int i = 0; i < (2 * nexternal); i++ )
  {
    user_colors[i] = colors_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the helicity configuration to use. Am empty vector means sum over all.
void PY8MEs_R3_P8_sm_qq_zqq::setHelicities(vector<int> helicities_picked) 
{
  if (helicities_picked.size() != nexternal)
  {
    if (helicities_picked.size() == 0)
    {
      user_helicities = vector<int> (); 
      return; 
    }
    cout <<  "Error in function 'setHelicities' of class" << 
    " 'PY8MEs_R3_P8_sm_qq_zqq': Incorrect number" << 
    " of helicities specified." << endl; 
    throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
  }
  user_helicities = vector<int> (nexternal, 0); 
  for(int i = 0; i < nexternal; i++ )
  {
    user_helicities[i] = helicities_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the permutation to use (will apply to momenta, colors and helicities)
void PY8MEs_R3_P8_sm_qq_zqq::setPermutation(vector<int> perm_picked) 
{
  if (perm_picked.size() != nexternal)
  {
    cout <<  "Error in function 'setPermutations' of class" << 
    " 'PY8MEs_R3_P8_sm_qq_zqq': Incorrect number" << 
    " of permutations specified." << endl; 
    throw PY8MEs_R3_P8_sm_qq_zqq_exception; 
  }
  for(int i = 0; i < nexternal; i++ )
  {
    perm[i] = perm_picked[i]; 
  }
}

// Set the proc_ID to use
void PY8MEs_R3_P8_sm_qq_zqq::setProcID(int procID_picked) 
{
  proc_ID = procID_picked; 
}

//--------------------------------------------------------------------------
// Initialize process.

void PY8MEs_R3_P8_sm_qq_zqq::initProc() 
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
  jamp2 = vector < vec_double > (20); 
  jamp2[0] = vector<double> (2, 0.); 
  jamp2[1] = vector<double> (2, 0.); 
  jamp2[2] = vector<double> (2, 0.); 
  jamp2[3] = vector<double> (2, 0.); 
  jamp2[4] = vector<double> (2, 0.); 
  jamp2[5] = vector<double> (2, 0.); 
  jamp2[6] = vector<double> (2, 0.); 
  jamp2[7] = vector<double> (2, 0.); 
  jamp2[8] = vector<double> (2, 0.); 
  jamp2[9] = vector<double> (2, 0.); 
  jamp2[10] = vector<double> (2, 0.); 
  jamp2[11] = vector<double> (2, 0.); 
  jamp2[12] = vector<double> (2, 0.); 
  jamp2[13] = vector<double> (2, 0.); 
  jamp2[14] = vector<double> (2, 0.); 
  jamp2[15] = vector<double> (2, 0.); 
  jamp2[16] = vector<double> (2, 0.); 
  jamp2[17] = vector<double> (2, 0.); 
  jamp2[18] = vector<double> (2, 0.); 
  jamp2[19] = vector<double> (2, 0.); 
  all_results = vector < vec_vec_double > (20); 
  // The first entry is always the color or helicity avg/summed matrix element.
  all_results[0] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[1] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[2] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[3] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[4] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[5] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[6] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[7] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[8] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[9] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[10] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[11] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[12] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[13] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[14] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[15] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[16] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[17] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[18] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[19] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
}

//--------------------------------------------------------------------------
// Evaluate the squared matrix element.

double PY8MEs_R3_P8_sm_qq_zqq::sigmaKin() 
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
  for(int i = 0; i < 2; i++ )
    jamp2[4][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[5][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[6][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[7][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[8][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[9][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[10][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[11][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[12][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[13][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[14][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[15][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[16][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[17][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[18][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[19][i] = 0.; 

  // Local variables and constants
  const int max_tries = 10; 
  vector < vec_bool > goodhel(nprocesses, vec_bool(ncomb, false)); 
  vec_int ntry(nprocesses, 0); 
  double t = 0.; 
  double result = 0.; 

  // Denominators: spins, colors and identical particles
  const int denom_colors[nprocesses] = {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
      9, 9, 9, 9, 9, 9, 9};
  const int denom_hels[nprocesses] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4};
  const int denom_iden[nprocesses] = {2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1};

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
    for(int i = 0; i < 2; i++ )
      jamp2[4][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[5][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[6][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[7][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[8][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[9][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[10][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[11][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[12][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[13][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[14][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[15][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[16][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[17][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[18][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[19][i] = 0.; 

    if (proc_ID == 0)
      t = matrix_3_uu_zuu(); 
    if (proc_ID == 1)
      t = matrix_3_uux_zuux(); 
    if (proc_ID == 2)
      t = matrix_3_dd_zdd(); 
    if (proc_ID == 3)
      t = matrix_3_ddx_zddx(); 
    if (proc_ID == 4)
      t = matrix_3_uxux_zuxux(); 
    if (proc_ID == 5)
      t = matrix_3_dxdx_zdxdx(); 
    if (proc_ID == 6)
      t = matrix_3_uc_zuc(); 
    if (proc_ID == 7)
      t = matrix_3_ud_zud(); 
    if (proc_ID == 8)
      t = matrix_3_uux_zccx(); 
    if (proc_ID == 9)
      t = matrix_3_uux_zddx(); 
    if (proc_ID == 10)
      t = matrix_3_ucx_zucx(); 
    if (proc_ID == 11)
      t = matrix_3_udx_zudx(); 
    if (proc_ID == 12)
      t = matrix_3_ds_zds(); 
    if (proc_ID == 13)
      t = matrix_3_dux_zdux(); 
    if (proc_ID == 14)
      t = matrix_3_ddx_zuux(); 
    if (proc_ID == 15)
      t = matrix_3_ddx_zssx(); 
    if (proc_ID == 16)
      t = matrix_3_dsx_zdsx(); 
    if (proc_ID == 17)
      t = matrix_3_uxcx_zuxcx(); 
    if (proc_ID == 18)
      t = matrix_3_uxdx_zuxdx(); 
    if (proc_ID == 19)
      t = matrix_3_dxsx_zdxsx(); 

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

void PY8MEs_R3_P8_sm_qq_zqq::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  // Calculate all wavefunctions
  ixxxxx(p[perm[0]], mME[0], hel[0], +1, w[0]); 
  ixxxxx(p[perm[1]], mME[1], hel[1], +1, w[1]); 
  vxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  oxxxxx(p[perm[3]], mME[3], hel[3], +1, w[3]); 
  oxxxxx(p[perm[4]], mME[4], hel[4], +1, w[4]); 
  FFV2_5_2(w[0], w[2], pars->GC_51, pars->GC_58, pars->ZERO, pars->ZERO, w[5]); 
  FFV1P0_3(w[1], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[6]); 
  FFV1P0_3(w[1], w[4], pars->GC_11, pars->ZERO, pars->ZERO, w[7]); 
  FFV1P0_3(w[0], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[8]); 
  FFV2_5_2(w[1], w[2], pars->GC_51, pars->GC_58, pars->ZERO, pars->ZERO, w[9]); 
  FFV2_5_1(w[4], w[2], pars->GC_51, pars->GC_58, pars->ZERO, pars->ZERO,
      w[10]);
  FFV1P0_3(w[0], w[4], pars->GC_11, pars->ZERO, pars->ZERO, w[11]); 
  FFV2_5_1(w[3], w[2], pars->GC_51, pars->GC_58, pars->ZERO, pars->ZERO,
      w[12]);
  oxxxxx(p[perm[1]], mME[1], hel[1], -1, w[13]); 
  ixxxxx(p[perm[4]], mME[4], hel[4], -1, w[14]); 
  FFV1P0_3(w[14], w[13], pars->GC_11, pars->ZERO, pars->ZERO, w[15]); 
  FFV1P0_3(w[14], w[3], pars->GC_11, pars->ZERO, pars->ZERO, w[16]); 
  FFV1P0_3(w[0], w[13], pars->GC_11, pars->ZERO, pars->ZERO, w[17]); 
  FFV2_5_2(w[14], w[2], pars->GC_51, pars->GC_58, pars->ZERO, pars->ZERO,
      w[18]);
  FFV2_5_1(w[13], w[2], pars->GC_51, pars->GC_58, pars->ZERO, pars->ZERO,
      w[19]);
  FFV2_3_2(w[0], w[2], pars->GC_50, pars->GC_58, pars->ZERO, pars->ZERO,
      w[20]);
  FFV2_3_2(w[1], w[2], pars->GC_50, pars->GC_58, pars->ZERO, pars->ZERO,
      w[21]);
  FFV2_3_1(w[4], w[2], pars->GC_50, pars->GC_58, pars->ZERO, pars->ZERO,
      w[22]);
  FFV2_3_1(w[3], w[2], pars->GC_50, pars->GC_58, pars->ZERO, pars->ZERO,
      w[23]);
  FFV2_3_2(w[14], w[2], pars->GC_50, pars->GC_58, pars->ZERO, pars->ZERO,
      w[24]);
  FFV2_3_1(w[13], w[2], pars->GC_50, pars->GC_58, pars->ZERO, pars->ZERO,
      w[25]);
  oxxxxx(p[perm[0]], mME[0], hel[0], -1, w[26]); 
  ixxxxx(p[perm[3]], mME[3], hel[3], -1, w[27]); 
  FFV2_5_2(w[27], w[2], pars->GC_51, pars->GC_58, pars->ZERO, pars->ZERO,
      w[28]);
  FFV1P0_3(w[14], w[26], pars->GC_11, pars->ZERO, pars->ZERO, w[29]); 
  FFV1P0_3(w[27], w[26], pars->GC_11, pars->ZERO, pars->ZERO, w[30]); 
  FFV1P0_3(w[27], w[13], pars->GC_11, pars->ZERO, pars->ZERO, w[31]); 
  FFV2_5_1(w[26], w[2], pars->GC_51, pars->GC_58, pars->ZERO, pars->ZERO,
      w[32]);
  FFV2_3_2(w[27], w[2], pars->GC_50, pars->GC_58, pars->ZERO, pars->ZERO,
      w[33]);
  FFV2_3_1(w[26], w[2], pars->GC_50, pars->GC_58, pars->ZERO, pars->ZERO,
      w[34]);

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[5], w[4], w[6], pars->GC_11, amp[0]); 
  FFV1_0(w[5], w[3], w[7], pars->GC_11, amp[1]); 
  FFV1_0(w[9], w[4], w[8], pars->GC_11, amp[2]); 
  FFV1_0(w[1], w[10], w[8], pars->GC_11, amp[3]); 
  FFV1_0(w[9], w[3], w[11], pars->GC_11, amp[4]); 
  FFV1_0(w[1], w[12], w[11], pars->GC_11, amp[5]); 
  FFV1_0(w[0], w[10], w[6], pars->GC_11, amp[6]); 
  FFV1_0(w[0], w[12], w[7], pars->GC_11, amp[7]); 
  FFV1_0(w[5], w[3], w[15], pars->GC_11, amp[8]); 
  FFV1_0(w[5], w[13], w[16], pars->GC_11, amp[9]); 
  FFV1_0(w[18], w[3], w[17], pars->GC_11, amp[10]); 
  FFV1_0(w[14], w[12], w[17], pars->GC_11, amp[11]); 
  FFV1_0(w[18], w[13], w[8], pars->GC_11, amp[12]); 
  FFV1_0(w[14], w[19], w[8], pars->GC_11, amp[13]); 
  FFV1_0(w[0], w[12], w[15], pars->GC_11, amp[14]); 
  FFV1_0(w[0], w[19], w[16], pars->GC_11, amp[15]); 
  FFV1_0(w[20], w[4], w[6], pars->GC_11, amp[16]); 
  FFV1_0(w[20], w[3], w[7], pars->GC_11, amp[17]); 
  FFV1_0(w[21], w[4], w[8], pars->GC_11, amp[18]); 
  FFV1_0(w[1], w[22], w[8], pars->GC_11, amp[19]); 
  FFV1_0(w[21], w[3], w[11], pars->GC_11, amp[20]); 
  FFV1_0(w[1], w[23], w[11], pars->GC_11, amp[21]); 
  FFV1_0(w[0], w[22], w[6], pars->GC_11, amp[22]); 
  FFV1_0(w[0], w[23], w[7], pars->GC_11, amp[23]); 
  FFV1_0(w[20], w[3], w[15], pars->GC_11, amp[24]); 
  FFV1_0(w[20], w[13], w[16], pars->GC_11, amp[25]); 
  FFV1_0(w[24], w[3], w[17], pars->GC_11, amp[26]); 
  FFV1_0(w[14], w[23], w[17], pars->GC_11, amp[27]); 
  FFV1_0(w[24], w[13], w[8], pars->GC_11, amp[28]); 
  FFV1_0(w[14], w[25], w[8], pars->GC_11, amp[29]); 
  FFV1_0(w[0], w[23], w[15], pars->GC_11, amp[30]); 
  FFV1_0(w[0], w[25], w[16], pars->GC_11, amp[31]); 
  FFV1_0(w[28], w[13], w[29], pars->GC_11, amp[32]); 
  FFV1_0(w[28], w[26], w[15], pars->GC_11, amp[33]); 
  FFV1_0(w[18], w[13], w[30], pars->GC_11, amp[34]); 
  FFV1_0(w[14], w[19], w[30], pars->GC_11, amp[35]); 
  FFV1_0(w[18], w[26], w[31], pars->GC_11, amp[36]); 
  FFV1_0(w[14], w[32], w[31], pars->GC_11, amp[37]); 
  FFV1_0(w[27], w[19], w[29], pars->GC_11, amp[38]); 
  FFV1_0(w[27], w[32], w[15], pars->GC_11, amp[39]); 
  FFV1_0(w[33], w[13], w[29], pars->GC_11, amp[40]); 
  FFV1_0(w[33], w[26], w[15], pars->GC_11, amp[41]); 
  FFV1_0(w[24], w[13], w[30], pars->GC_11, amp[42]); 
  FFV1_0(w[14], w[25], w[30], pars->GC_11, amp[43]); 
  FFV1_0(w[24], w[26], w[31], pars->GC_11, amp[44]); 
  FFV1_0(w[14], w[34], w[31], pars->GC_11, amp[45]); 
  FFV1_0(w[27], w[25], w[29], pars->GC_11, amp[46]); 
  FFV1_0(w[27], w[34], w[15], pars->GC_11, amp[47]); 
  FFV1_0(w[5], w[3], w[7], pars->GC_11, amp[48]); 
  FFV1_0(w[9], w[4], w[8], pars->GC_11, amp[49]); 
  FFV1_0(w[1], w[10], w[8], pars->GC_11, amp[50]); 
  FFV1_0(w[0], w[12], w[7], pars->GC_11, amp[51]); 
  FFV1_0(w[5], w[3], w[7], pars->GC_11, amp[52]); 
  FFV1_0(w[21], w[4], w[8], pars->GC_11, amp[53]); 
  FFV1_0(w[1], w[22], w[8], pars->GC_11, amp[54]); 
  FFV1_0(w[0], w[12], w[7], pars->GC_11, amp[55]); 
  FFV1_0(w[5], w[13], w[16], pars->GC_11, amp[56]); 
  FFV1_0(w[18], w[3], w[17], pars->GC_11, amp[57]); 
  FFV1_0(w[14], w[12], w[17], pars->GC_11, amp[58]); 
  FFV1_0(w[0], w[19], w[16], pars->GC_11, amp[59]); 
  FFV1_0(w[5], w[13], w[16], pars->GC_11, amp[60]); 
  FFV1_0(w[24], w[3], w[17], pars->GC_11, amp[61]); 
  FFV1_0(w[14], w[23], w[17], pars->GC_11, amp[62]); 
  FFV1_0(w[0], w[19], w[16], pars->GC_11, amp[63]); 
  FFV1_0(w[5], w[3], w[15], pars->GC_11, amp[64]); 
  FFV1_0(w[18], w[13], w[8], pars->GC_11, amp[65]); 
  FFV1_0(w[14], w[19], w[8], pars->GC_11, amp[66]); 
  FFV1_0(w[0], w[12], w[15], pars->GC_11, amp[67]); 
  FFV1_0(w[5], w[3], w[15], pars->GC_11, amp[68]); 
  FFV1_0(w[24], w[13], w[8], pars->GC_11, amp[69]); 
  FFV1_0(w[14], w[25], w[8], pars->GC_11, amp[70]); 
  FFV1_0(w[0], w[12], w[15], pars->GC_11, amp[71]); 
  FFV1_0(w[20], w[3], w[7], pars->GC_11, amp[72]); 
  FFV1_0(w[21], w[4], w[8], pars->GC_11, amp[73]); 
  FFV1_0(w[1], w[22], w[8], pars->GC_11, amp[74]); 
  FFV1_0(w[0], w[23], w[7], pars->GC_11, amp[75]); 
  FFV1_0(w[18], w[13], w[8], pars->GC_11, amp[76]); 
  FFV1_0(w[20], w[3], w[15], pars->GC_11, amp[77]); 
  FFV1_0(w[0], w[23], w[15], pars->GC_11, amp[78]); 
  FFV1_0(w[14], w[19], w[8], pars->GC_11, amp[79]); 
  FFV1_0(w[18], w[3], w[17], pars->GC_11, amp[80]); 
  FFV1_0(w[20], w[13], w[16], pars->GC_11, amp[81]); 
  FFV1_0(w[0], w[25], w[16], pars->GC_11, amp[82]); 
  FFV1_0(w[14], w[12], w[17], pars->GC_11, amp[83]); 
  FFV1_0(w[20], w[13], w[16], pars->GC_11, amp[84]); 
  FFV1_0(w[24], w[3], w[17], pars->GC_11, amp[85]); 
  FFV1_0(w[14], w[23], w[17], pars->GC_11, amp[86]); 
  FFV1_0(w[0], w[25], w[16], pars->GC_11, amp[87]); 
  FFV1_0(w[20], w[3], w[15], pars->GC_11, amp[88]); 
  FFV1_0(w[24], w[13], w[8], pars->GC_11, amp[89]); 
  FFV1_0(w[14], w[25], w[8], pars->GC_11, amp[90]); 
  FFV1_0(w[0], w[23], w[15], pars->GC_11, amp[91]); 
  FFV1_0(w[28], w[26], w[15], pars->GC_11, amp[92]); 
  FFV1_0(w[18], w[13], w[30], pars->GC_11, amp[93]); 
  FFV1_0(w[14], w[19], w[30], pars->GC_11, amp[94]); 
  FFV1_0(w[27], w[32], w[15], pars->GC_11, amp[95]); 
  FFV1_0(w[28], w[26], w[15], pars->GC_11, amp[96]); 
  FFV1_0(w[24], w[13], w[30], pars->GC_11, amp[97]); 
  FFV1_0(w[14], w[25], w[30], pars->GC_11, amp[98]); 
  FFV1_0(w[27], w[32], w[15], pars->GC_11, amp[99]); 
  FFV1_0(w[33], w[26], w[15], pars->GC_11, amp[100]); 
  FFV1_0(w[24], w[13], w[30], pars->GC_11, amp[101]); 
  FFV1_0(w[14], w[25], w[30], pars->GC_11, amp[102]); 
  FFV1_0(w[27], w[34], w[15], pars->GC_11, amp[103]); 


}
double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_uu_zuu() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 8; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[0] + 1./3. * amp[1] + 1./3. * amp[2] + 1./3. *
      amp[3] + amp[4] + amp[5] + amp[6] + 1./3. * amp[7]);
  jamp[1] = +1./2. * (-1./3. * amp[0] - amp[1] - amp[2] - amp[3] - 1./3. *
      amp[4] - 1./3. * amp[5] - 1./3. * amp[6] - amp[7]);

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

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_uux_zuux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 8; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[8] + 1./3. * amp[9] + 1./3. * amp[10] + 1./3. *
      amp[11] + amp[12] + amp[13] + amp[14] + 1./3. * amp[15]);
  jamp[1] = +1./2. * (-1./3. * amp[8] - amp[9] - amp[10] - amp[11] - 1./3. *
      amp[12] - 1./3. * amp[13] - 1./3. * amp[14] - amp[15]);

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

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_dd_zdd() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 8; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[16] + 1./3. * amp[17] + 1./3. * amp[18] + 1./3. *
      amp[19] + amp[20] + amp[21] + amp[22] + 1./3. * amp[23]);
  jamp[1] = +1./2. * (-1./3. * amp[16] - amp[17] - amp[18] - amp[19] - 1./3. *
      amp[20] - 1./3. * amp[21] - 1./3. * amp[22] - amp[23]);

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

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_ddx_zddx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 8; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[24] + 1./3. * amp[25] + 1./3. * amp[26] + 1./3. *
      amp[27] + amp[28] + amp[29] + amp[30] + 1./3. * amp[31]);
  jamp[1] = +1./2. * (-1./3. * amp[24] - amp[25] - amp[26] - amp[27] - 1./3. *
      amp[28] - 1./3. * amp[29] - 1./3. * amp[30] - amp[31]);

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

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_uxux_zuxux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 8; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[32] + 1./3. * amp[33] + 1./3. * amp[34] + 1./3. *
      amp[35] + amp[36] + amp[37] + amp[38] + 1./3. * amp[39]);
  jamp[1] = +1./2. * (-1./3. * amp[32] - amp[33] - amp[34] - amp[35] - 1./3. *
      amp[36] - 1./3. * amp[37] - 1./3. * amp[38] - amp[39]);

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

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_dxdx_zdxdx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 8; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[40] + 1./3. * amp[41] + 1./3. * amp[42] + 1./3. *
      amp[43] + amp[44] + amp[45] + amp[46] + 1./3. * amp[47]);
  jamp[1] = +1./2. * (-1./3. * amp[40] - amp[41] - amp[42] - amp[43] - 1./3. *
      amp[44] - 1./3. * amp[45] - 1./3. * amp[46] - amp[47]);

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

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_uc_zuc() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[48] + 1./3. * amp[49] + 1./3. * amp[50] +
      1./3. * amp[51]);
  jamp[1] = +1./2. * (-amp[48] - amp[49] - amp[50] - amp[51]); 

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

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_ud_zud() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[52] + 1./3. * amp[53] + 1./3. * amp[54] +
      1./3. * amp[55]);
  jamp[1] = +1./2. * (-amp[52] - amp[53] - amp[54] - amp[55]); 

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
    jamp2[7][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_uux_zccx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[56] + 1./3. * amp[57] + 1./3. * amp[58] +
      1./3. * amp[59]);
  jamp[1] = +1./2. * (-amp[56] - amp[57] - amp[58] - amp[59]); 

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
    jamp2[8][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_uux_zddx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[60] + 1./3. * amp[61] + 1./3. * amp[62] +
      1./3. * amp[63]);
  jamp[1] = +1./2. * (-amp[60] - amp[61] - amp[62] - amp[63]); 

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
    jamp2[9][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_ucx_zucx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[64] + amp[65] + amp[66] + amp[67]); 
  jamp[1] = +1./2. * (-1./3. * amp[64] - 1./3. * amp[65] - 1./3. * amp[66] -
      1./3. * amp[67]);

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
    jamp2[10][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_udx_zudx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[68] + amp[69] + amp[70] + amp[71]); 
  jamp[1] = +1./2. * (-1./3. * amp[68] - 1./3. * amp[69] - 1./3. * amp[70] -
      1./3. * amp[71]);

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
    jamp2[11][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_ds_zds() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[72] + 1./3. * amp[73] + 1./3. * amp[74] +
      1./3. * amp[75]);
  jamp[1] = +1./2. * (-amp[72] - amp[73] - amp[74] - amp[75]); 

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
    jamp2[12][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_dux_zdux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[76] + amp[77] + amp[78] + amp[79]); 
  jamp[1] = +1./2. * (-1./3. * amp[76] - 1./3. * amp[77] - 1./3. * amp[78] -
      1./3. * amp[79]);

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
    jamp2[13][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_ddx_zuux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[80] + 1./3. * amp[81] + 1./3. * amp[82] +
      1./3. * amp[83]);
  jamp[1] = +1./2. * (-amp[80] - amp[81] - amp[82] - amp[83]); 

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
    jamp2[14][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_ddx_zssx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[84] + 1./3. * amp[85] + 1./3. * amp[86] +
      1./3. * amp[87]);
  jamp[1] = +1./2. * (-amp[84] - amp[85] - amp[86] - amp[87]); 

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
    jamp2[15][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_dsx_zdsx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[88] + amp[89] + amp[90] + amp[91]); 
  jamp[1] = +1./2. * (-1./3. * amp[88] - 1./3. * amp[89] - 1./3. * amp[90] -
      1./3. * amp[91]);

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
    jamp2[16][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_uxcx_zuxcx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[92] + 1./3. * amp[93] + 1./3. * amp[94] +
      1./3. * amp[95]);
  jamp[1] = +1./2. * (-amp[92] - amp[93] - amp[94] - amp[95]); 

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
    jamp2[17][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_uxdx_zuxdx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[96] + 1./3. * amp[97] + 1./3. * amp[98] +
      1./3. * amp[99]);
  jamp[1] = +1./2. * (-amp[96] - amp[97] - amp[98] - amp[99]); 

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
    jamp2[18][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}

double PY8MEs_R3_P8_sm_qq_zqq::matrix_3_dxsx_zdxsx() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 4; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  double denom[ncolor] = {1, 1}; 
  double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = +1./2. * (+1./3. * amp[100] + 1./3. * amp[101] + 1./3. * amp[102] +
      1./3. * amp[103]);
  jamp[1] = +1./2. * (-amp[100] - amp[101] - amp[102] - amp[103]); 

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
    jamp2[19][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}


}  // end namespace PY8MEs_namespace

