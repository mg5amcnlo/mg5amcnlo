//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "PY8MEs_R221_P2_sm_ckm_qq_qq.h"
#include "HelAmps_sm_ckm.h"

using namespace Pythia8_sm_ckm; 

namespace PY8MEs_namespace 
{
//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u > u u QCD=0 @221
// Process: c c > c c QCD=0 @221
// Process: u u~ > u u~ QCD=0 @221
// Process: c c~ > c c~ QCD=0 @221
// Process: d d > d d QCD=0 @221
// Process: s s > s s QCD=0 @221
// Process: d d~ > d d~ QCD=0 @221
// Process: s s~ > s s~ QCD=0 @221
// Process: u~ u~ > u~ u~ QCD=0 @221
// Process: c~ c~ > c~ c~ QCD=0 @221
// Process: d~ d~ > d~ d~ QCD=0 @221
// Process: s~ s~ > s~ s~ QCD=0 @221
// Process: u d > u d QCD=0 @221
// Process: c s > c s QCD=0 @221
// Process: u s > u s QCD=0 @221
// Process: u u~ > d d~ QCD=0 @221
// Process: c c~ > s s~ QCD=0 @221
// Process: u u~ > s s~ QCD=0 @221
// Process: u d~ > u d~ QCD=0 @221
// Process: c s~ > c s~ QCD=0 @221
// Process: u s~ > u s~ QCD=0 @221
// Process: c d > c d QCD=0 @221
// Process: c c~ > d d~ QCD=0 @221
// Process: c d~ > c d~ QCD=0 @221
// Process: d u~ > d u~ QCD=0 @221
// Process: s c~ > s c~ QCD=0 @221
// Process: d c~ > d c~ QCD=0 @221
// Process: d d~ > u u~ QCD=0 @221
// Process: s s~ > c c~ QCD=0 @221
// Process: d d~ > c c~ QCD=0 @221
// Process: s u~ > s u~ QCD=0 @221
// Process: s s~ > u u~ QCD=0 @221
// Process: u~ d~ > u~ d~ QCD=0 @221
// Process: c~ s~ > c~ s~ QCD=0 @221
// Process: u~ s~ > u~ s~ QCD=0 @221
// Process: c~ d~ > c~ d~ QCD=0 @221
// Process: u c > u c QCD=0 @221
// Process: u u~ > c c~ QCD=0 @221
// Process: c c~ > u u~ QCD=0 @221
// Process: u c~ > u c~ QCD=0 @221
// Process: c u~ > c u~ QCD=0 @221
// Process: d s > d s QCD=0 @221
// Process: d d~ > s s~ QCD=0 @221
// Process: s s~ > d d~ QCD=0 @221
// Process: d s~ > d s~ QCD=0 @221
// Process: s d~ > s d~ QCD=0 @221
// Process: u~ c~ > u~ c~ QCD=0 @221
// Process: d~ s~ > d~ s~ QCD=0 @221
// Process: u d > u s QCD=0 @221
// Process: u s > c s QCD=0 @221
// Process: u d > c d QCD=0 @221
// Process: c d > c s QCD=0 @221
// Process: u d > c s QCD=0 @221
// Process: u s > u d QCD=0 @221
// Process: c s > u s QCD=0 @221
// Process: u s > c d QCD=0 @221
// Process: c d > u s QCD=0 @221
// Process: u u~ > d s~ QCD=0 @221
// Process: c u~ > s s~ QCD=0 @221
// Process: d s~ > u u~ QCD=0 @221
// Process: s s~ > c u~ QCD=0 @221
// Process: u u~ > s d~ QCD=0 @221
// Process: u c~ > s s~ QCD=0 @221
// Process: s d~ > u u~ QCD=0 @221
// Process: s s~ > u c~ QCD=0 @221
// Process: u c~ > d d~ QCD=0 @221
// Process: c c~ > s d~ QCD=0 @221
// Process: d d~ > u c~ QCD=0 @221
// Process: s d~ > c c~ QCD=0 @221
// Process: u c~ > d s~ QCD=0 @221
// Process: c u~ > s d~ QCD=0 @221
// Process: d s~ > u c~ QCD=0 @221
// Process: s d~ > c u~ QCD=0 @221
// Process: u c~ > s d~ QCD=0 @221
// Process: s d~ > u c~ QCD=0 @221
// Process: u d~ > u s~ QCD=0 @221
// Process: c s~ > u s~ QCD=0 @221
// Process: u d~ > c d~ QCD=0 @221
// Process: c s~ > c d~ QCD=0 @221
// Process: u d~ > c s~ QCD=0 @221
// Process: c s~ > u d~ QCD=0 @221
// Process: u s~ > u d~ QCD=0 @221
// Process: u s~ > c s~ QCD=0 @221
// Process: u s~ > c d~ QCD=0 @221
// Process: c d > u d QCD=0 @221
// Process: c s > c d QCD=0 @221
// Process: c s > u d QCD=0 @221
// Process: c u~ > d d~ QCD=0 @221
// Process: c c~ > d s~ QCD=0 @221
// Process: d d~ > c u~ QCD=0 @221
// Process: d s~ > c c~ QCD=0 @221
// Process: c u~ > d s~ QCD=0 @221
// Process: d s~ > c u~ QCD=0 @221
// Process: c d~ > u d~ QCD=0 @221
// Process: c d~ > c s~ QCD=0 @221
// Process: c d~ > u s~ QCD=0 @221
// Process: d u~ > d c~ QCD=0 @221
// Process: s c~ > d c~ QCD=0 @221
// Process: d u~ > s u~ QCD=0 @221
// Process: s c~ > s u~ QCD=0 @221
// Process: d u~ > s c~ QCD=0 @221
// Process: s c~ > d u~ QCD=0 @221
// Process: d c~ > d u~ QCD=0 @221
// Process: d c~ > s c~ QCD=0 @221
// Process: d c~ > s u~ QCD=0 @221
// Process: s u~ > d u~ QCD=0 @221
// Process: s u~ > s c~ QCD=0 @221
// Process: s u~ > d c~ QCD=0 @221
// Process: u~ d~ > u~ s~ QCD=0 @221
// Process: u~ s~ > c~ s~ QCD=0 @221
// Process: u~ d~ > c~ d~ QCD=0 @221
// Process: c~ d~ > c~ s~ QCD=0 @221
// Process: u~ d~ > c~ s~ QCD=0 @221
// Process: u~ s~ > u~ d~ QCD=0 @221
// Process: c~ s~ > u~ s~ QCD=0 @221
// Process: u~ s~ > c~ d~ QCD=0 @221
// Process: c~ d~ > u~ s~ QCD=0 @221
// Process: c~ d~ > u~ d~ QCD=0 @221
// Process: c~ s~ > c~ d~ QCD=0 @221
// Process: c~ s~ > u~ d~ QCD=0 @221

// Exception class
class PY8MEs_R221_P2_sm_ckm_qq_qqException : public exception
{
  virtual const char * what() const throw()
  {
    return "Exception in class 'PY8MEs_R221_P2_sm_ckm_qq_qq'."; 
  }
}
PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 

std::set<int> PY8MEs_R221_P2_sm_ckm_qq_qq::s_channel_proc = std::set<int>
    (createset<int> ());

int PY8MEs_R221_P2_sm_ckm_qq_qq::helicities[ncomb][nexternal] = {{-1, -1, -1,
    -1}, {-1, -1, -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1},
    {-1, 1, -1, 1}, {-1, 1, 1, -1}, {-1, 1, 1, 1}, {1, -1, -1, -1}, {1, -1, -1,
    1}, {1, -1, 1, -1}, {1, -1, 1, 1}, {1, 1, -1, -1}, {1, 1, -1, 1}, {1, 1, 1,
    -1}, {1, 1, 1, 1}};

// Normalization factors the various processes
// Denominators: spins, colors and identical particles
int PY8MEs_R221_P2_sm_ckm_qq_qq::denom_colors[nprocesses] = {9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
int PY8MEs_R221_P2_sm_ckm_qq_qq::denom_hels[nprocesses] = {4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
int PY8MEs_R221_P2_sm_ckm_qq_qq::denom_iden[nprocesses] = {2, 1, 2, 1, 2, 2, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

//--------------------------------------------------------------------------
// Color config initialization
void PY8MEs_R221_P2_sm_ckm_qq_qq::initColorConfigs() 
{
  color_configs = vector < vec_vec_int > (); 
  jamp_nc_relative_power = vector < vec_int > (); 

  // Color flows of process Process: u u > u u QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[0].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #1
  color_configs[0].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[0].push_back(0); 

  // Color flows of process Process: u u~ > u u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[1].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[1].push_back(0); 
  // JAMP #1
  color_configs[1].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[1].push_back(0); 

  // Color flows of process Process: d d > d d QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[2].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[2].push_back(0); 
  // JAMP #1
  color_configs[2].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[2].push_back(0); 

  // Color flows of process Process: d d~ > d d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[3].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[3].push_back(0); 
  // JAMP #1
  color_configs[3].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[3].push_back(0); 

  // Color flows of process Process: u~ u~ > u~ u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[4].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[4].push_back(0); 
  // JAMP #1
  color_configs[4].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(2)(0)(1)));
  jamp_nc_relative_power[4].push_back(0); 

  // Color flows of process Process: d~ d~ > d~ d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[5].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[5].push_back(0); 
  // JAMP #1
  color_configs[5].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(2)(0)(1)));
  jamp_nc_relative_power[5].push_back(0); 

  // Color flows of process Process: u d > u d QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[6].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[6].push_back(0); 
  // JAMP #1
  color_configs[6].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[6].push_back(0); 

  // Color flows of process Process: u s > u s QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[7].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[7].push_back(0); 
  // JAMP #1
  color_configs[7].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[7].push_back(0); 

  // Color flows of process Process: u u~ > d d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[8].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[8].push_back(0); 
  // JAMP #1
  color_configs[8].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[8].push_back(0); 

  // Color flows of process Process: u u~ > s s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[9].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[9].push_back(0); 
  // JAMP #1
  color_configs[9].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[9].push_back(0); 

  // Color flows of process Process: u d~ > u d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[10].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[10].push_back(0); 
  // JAMP #1
  color_configs[10].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[10].push_back(0); 

  // Color flows of process Process: u s~ > u s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[11].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[11].push_back(0); 
  // JAMP #1
  color_configs[11].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[11].push_back(0); 

  // Color flows of process Process: c d > c d QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[12].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[12].push_back(0); 
  // JAMP #1
  color_configs[12].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[12].push_back(0); 

  // Color flows of process Process: c c~ > d d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[13].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[13].push_back(0); 
  // JAMP #1
  color_configs[13].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[13].push_back(0); 

  // Color flows of process Process: c d~ > c d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[14].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[14].push_back(0); 
  // JAMP #1
  color_configs[14].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[14].push_back(0); 

  // Color flows of process Process: d u~ > d u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[15].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[15].push_back(0); 
  // JAMP #1
  color_configs[15].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[15].push_back(0); 

  // Color flows of process Process: d c~ > d c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[16].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[16].push_back(0); 
  // JAMP #1
  color_configs[16].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[16].push_back(0); 

  // Color flows of process Process: d d~ > u u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[17].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[17].push_back(0); 
  // JAMP #1
  color_configs[17].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[17].push_back(0); 

  // Color flows of process Process: d d~ > c c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[18].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[18].push_back(0); 
  // JAMP #1
  color_configs[18].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[18].push_back(0); 

  // Color flows of process Process: s u~ > s u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[19].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[19].push_back(0); 
  // JAMP #1
  color_configs[19].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[19].push_back(0); 

  // Color flows of process Process: s s~ > u u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[20].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[20].push_back(0); 
  // JAMP #1
  color_configs[20].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[20].push_back(0); 

  // Color flows of process Process: u~ d~ > u~ d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[21].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[21].push_back(0); 
  // JAMP #1
  color_configs[21].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(2)(0)(1)));
  jamp_nc_relative_power[21].push_back(0); 

  // Color flows of process Process: u~ s~ > u~ s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[22].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[22].push_back(0); 
  // JAMP #1
  color_configs[22].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(2)(0)(1)));
  jamp_nc_relative_power[22].push_back(0); 

  // Color flows of process Process: c~ d~ > c~ d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[23].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[23].push_back(0); 
  // JAMP #1
  color_configs[23].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(2)(0)(1)));
  jamp_nc_relative_power[23].push_back(0); 

  // Color flows of process Process: u c > u c QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[24].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[24].push_back(0); 

  // Color flows of process Process: u u~ > c c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[25].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[25].push_back(0); 

  // Color flows of process Process: u c~ > u c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[26].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[26].push_back(0); 

  // Color flows of process Process: d s > d s QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[27].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[27].push_back(0); 

  // Color flows of process Process: d d~ > s s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[28].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[28].push_back(0); 

  // Color flows of process Process: d s~ > d s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[29].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[29].push_back(0); 

  // Color flows of process Process: u~ c~ > u~ c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[30].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[30].push_back(0); 

  // Color flows of process Process: d~ s~ > d~ s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[31].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(1)(0)(2)));
  jamp_nc_relative_power[31].push_back(0); 

  // Color flows of process Process: u d > u s QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[32].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[32].push_back(0); 

  // Color flows of process Process: u d > c d QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[33].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[33].push_back(0); 

  // Color flows of process Process: u d > c s QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[34].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[34].push_back(0); 

  // Color flows of process Process: u s > u d QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[35].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[35].push_back(0); 

  // Color flows of process Process: u s > c d QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[36].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[36].push_back(0); 

  // Color flows of process Process: u u~ > d s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[37].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[37].push_back(0); 

  // Color flows of process Process: u u~ > s d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[38].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[38].push_back(0); 

  // Color flows of process Process: u c~ > d d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[39].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[39].push_back(0); 

  // Color flows of process Process: u c~ > d s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[40].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[40].push_back(0); 

  // Color flows of process Process: u c~ > s d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[41].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[41].push_back(0); 

  // Color flows of process Process: u d~ > u s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[42].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[42].push_back(0); 

  // Color flows of process Process: u d~ > c d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[43].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[43].push_back(0); 

  // Color flows of process Process: u d~ > c s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[44].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[44].push_back(0); 

  // Color flows of process Process: u s~ > u d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[45].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[45].push_back(0); 

  // Color flows of process Process: u s~ > c d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[46].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[46].push_back(0); 

  // Color flows of process Process: c d > u d QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[47].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[47].push_back(0); 

  // Color flows of process Process: c s > u d QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[48].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[48].push_back(0); 

  // Color flows of process Process: c u~ > d d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[49].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[49].push_back(0); 

  // Color flows of process Process: c u~ > d s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[50].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(2)(0)(0)(1)));
  jamp_nc_relative_power[50].push_back(0); 

  // Color flows of process Process: c d~ > u d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[51].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[51].push_back(0); 

  // Color flows of process Process: c d~ > u s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[52].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[52].push_back(0); 

  // Color flows of process Process: d u~ > d c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[53].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[53].push_back(0); 

  // Color flows of process Process: d u~ > s u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[54].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[54].push_back(0); 

  // Color flows of process Process: d u~ > s c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[55].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[55].push_back(0); 

  // Color flows of process Process: d c~ > d u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[56].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[56].push_back(0); 

  // Color flows of process Process: d c~ > s u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[57].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[57].push_back(0); 

  // Color flows of process Process: s u~ > d u~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[58].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[58].push_back(0); 

  // Color flows of process Process: s u~ > d c~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[59].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(2)(0)(0)(2)));
  jamp_nc_relative_power[59].push_back(0); 

  // Color flows of process Process: u~ d~ > u~ s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[60].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(2)(0)(1)));
  jamp_nc_relative_power[60].push_back(0); 

  // Color flows of process Process: u~ d~ > c~ d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[61].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(2)(0)(1)));
  jamp_nc_relative_power[61].push_back(0); 

  // Color flows of process Process: u~ d~ > c~ s~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[62].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(2)(0)(1)));
  jamp_nc_relative_power[62].push_back(0); 

  // Color flows of process Process: u~ s~ > u~ d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[63].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(2)(0)(1)));
  jamp_nc_relative_power[63].push_back(0); 

  // Color flows of process Process: u~ s~ > c~ d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[64].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(2)(0)(1)));
  jamp_nc_relative_power[64].push_back(0); 

  // Color flows of process Process: c~ d~ > u~ d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[65].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(2)(0)(1)));
  jamp_nc_relative_power[65].push_back(0); 

  // Color flows of process Process: c~ s~ > u~ d~ QCD=0 @221
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[66].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(2)(0)(1)));
  jamp_nc_relative_power[66].push_back(0); 
}

//--------------------------------------------------------------------------
// Destructor.
PY8MEs_R221_P2_sm_ckm_qq_qq::~PY8MEs_R221_P2_sm_ckm_qq_qq() 
{
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    delete[] p[i]; 
    p[i] = NULL; 
  }
}

//--------------------------------------------------------------------------
// Invert the permutation mapping
vector<int> PY8MEs_R221_P2_sm_ckm_qq_qq::invert_mapping(vector<int> mapping) 
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
vector < vec_int > PY8MEs_R221_P2_sm_ckm_qq_qq::getHelicityConfigs(vector<int>
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
vector < vec_int > PY8MEs_R221_P2_sm_ckm_qq_qq::getColorConfigs(int
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
int PY8MEs_R221_P2_sm_ckm_qq_qq::getColorFlowRelativeNCPower(int color_flow_ID,
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
vector<int> PY8MEs_R221_P2_sm_ckm_qq_qq::getHelicityConfigForID(int hel_ID,
    vector<int> permutation)
{
  if (hel_ID < 0 || hel_ID >= ncomb)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'getHelicityConfigForID' of class" << 
    " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Specified helicity ID '" << 
    hel_ID <<  "' cannot be found." << endl; 
    #endif
    throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
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
int PY8MEs_R221_P2_sm_ckm_qq_qq::getHelicityIDForConfig(vector<int> hel_config,
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
      " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Specified helicity" << 
      " configuration cannot be found." << endl; 
      #endif
      throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
    }
  }
  return user_ihel; 
}


//--------------------------------------------------------------------------
// Implements the map Color ID -> Color Config
vector<int> PY8MEs_R221_P2_sm_ckm_qq_qq::getColorConfigForID(int color_ID, int
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
    " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Specified color ID '" << 
    color_ID <<  "' cannot be found." << endl; 
    #endif
    throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
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
int PY8MEs_R221_P2_sm_ckm_qq_qq::getColorIDForConfig(vector<int> color_config,
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
            " 'PY8MEs_R221_P2_sm_ckm_qq_qq': A color line could " << 
            " not be closed." << endl; 
            #endif
            throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
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
      " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Specified color" << 
      " configuration cannot be found." << endl; 
      #endif
      throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
    }
  }
  return user_icol; 
}

//--------------------------------------------------------------------------
// Returns all result previously computed in SigmaKin
vector < vec_double > PY8MEs_R221_P2_sm_ckm_qq_qq::getAllResults(int
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
double PY8MEs_R221_P2_sm_ckm_qq_qq::getResult(int helicity_ID, int color_ID,
    int specify_proc_ID)
{
  if (helicity_ID < - 1 || helicity_ID >= ncomb)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Specified helicity ID '" << 
    helicity_ID <<  "' configuration cannot be found." << endl; 
    #endif
    throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
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
    " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Specified color ID '" << 
    color_ID <<  "' configuration cannot be found." << endl; 
    #endif
    throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
  }
  return all_results[chosenProcID][helicity_ID + 1][color_ID + 1]; 
}

//--------------------------------------------------------------------------
// Check for the availability of the requested process and if available,
// If available, this returns the corresponding permutation and Proc_ID to use.
// If not available, this returns a negative Proc_ID.
pair < vector<int> , int > PY8MEs_R221_P2_sm_ckm_qq_qq::static_getPY8ME(vector<int> initial_pdgs, vector<int> final_pdgs, set<int> schannels) 
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
  const int nprocs = 232; 
  const int proc_IDS[nprocs] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8,
      8, 9, 10, 10, 11, 12, 13, 14, 15, 15, 16, 17, 17, 18, 19, 20, 21, 21, 22,
      23, 24, 25, 25, 26, 26, 27, 28, 28, 29, 29, 30, 31, 32, 32, 33, 33, 34,
      35, 35, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40,
      40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 47, 47, 48, 49, 49,
      49, 49, 50, 50, 51, 51, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 58, 58,
      59, 60, 60, 61, 61, 62, 63, 63, 64, 64, 65, 65, 66, 1, 1, 3, 3, 6, 6, 7,
      8, 8, 9, 10, 10, 11, 12, 13, 14, 15, 15, 16, 17, 17, 18, 19, 20, 21, 21,
      22, 23, 24, 25, 25, 26, 26, 27, 28, 28, 29, 29, 30, 31, 32, 32, 33, 33,
      34, 35, 35, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40,
      40, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 47, 47, 48, 49,
      49, 49, 49, 50, 50, 51, 51, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 58,
      58, 59, 60, 60, 61, 61, 62, 63, 63, 64, 64, 65, 65, 66};
  const int in_pdgs[nprocs][ninitial] = {{2, 2}, {4, 4}, {2, -2}, {4, -4}, {1,
      1}, {3, 3}, {1, -1}, {3, -3}, {-2, -2}, {-4, -4}, {-1, -1}, {-3, -3}, {2,
      1}, {4, 3}, {2, 3}, {2, -2}, {4, -4}, {2, -2}, {2, -1}, {4, -3}, {2, -3},
      {4, 1}, {4, -4}, {4, -1}, {1, -2}, {3, -4}, {1, -4}, {1, -1}, {3, -3},
      {1, -1}, {3, -2}, {3, -3}, {-2, -1}, {-4, -3}, {-2, -3}, {-4, -1}, {2,
      4}, {2, -2}, {4, -4}, {2, -4}, {4, -2}, {1, 3}, {1, -1}, {3, -3}, {1,
      -3}, {3, -1}, {-2, -4}, {-1, -3}, {2, 1}, {2, 3}, {2, 1}, {4, 1}, {2, 1},
      {2, 3}, {4, 3}, {2, 3}, {4, 1}, {2, -2}, {4, -2}, {1, -3}, {3, -3}, {2,
      -2}, {2, -4}, {3, -1}, {3, -3}, {2, -4}, {4, -4}, {1, -1}, {3, -1}, {2,
      -4}, {4, -2}, {1, -3}, {3, -1}, {2, -4}, {3, -1}, {2, -1}, {4, -3}, {2,
      -1}, {4, -3}, {2, -1}, {4, -3}, {2, -3}, {2, -3}, {2, -3}, {4, 1}, {4,
      3}, {4, 3}, {4, -2}, {4, -4}, {1, -1}, {1, -3}, {4, -2}, {1, -3}, {4,
      -1}, {4, -1}, {4, -1}, {1, -2}, {3, -4}, {1, -2}, {3, -4}, {1, -2}, {3,
      -4}, {1, -4}, {1, -4}, {1, -4}, {3, -2}, {3, -2}, {3, -2}, {-2, -1}, {-2,
      -3}, {-2, -1}, {-4, -1}, {-2, -1}, {-2, -3}, {-4, -3}, {-2, -3}, {-4,
      -1}, {-4, -1}, {-4, -3}, {-4, -3}, {-2, 2}, {-4, 4}, {-1, 1}, {-3, 3},
      {1, 2}, {3, 4}, {3, 2}, {-2, 2}, {-4, 4}, {-2, 2}, {-1, 2}, {-3, 4}, {-3,
      2}, {1, 4}, {-4, 4}, {-1, 4}, {-2, 1}, {-4, 3}, {-4, 1}, {-1, 1}, {-3,
      3}, {-1, 1}, {-2, 3}, {-3, 3}, {-1, -2}, {-3, -4}, {-3, -2}, {-1, -4},
      {4, 2}, {-2, 2}, {-4, 4}, {-4, 2}, {-2, 4}, {3, 1}, {-1, 1}, {-3, 3},
      {-3, 1}, {-1, 3}, {-4, -2}, {-3, -1}, {1, 2}, {3, 2}, {1, 2}, {1, 4}, {1,
      2}, {3, 2}, {3, 4}, {3, 2}, {1, 4}, {-2, 2}, {-2, 4}, {-3, 1}, {-3, 3},
      {-2, 2}, {-4, 2}, {-1, 3}, {-3, 3}, {-4, 2}, {-4, 4}, {-1, 1}, {-1, 3},
      {-4, 2}, {-2, 4}, {-3, 1}, {-1, 3}, {-4, 2}, {-1, 3}, {-1, 2}, {-3, 4},
      {-1, 2}, {-3, 4}, {-1, 2}, {-3, 4}, {-3, 2}, {-3, 2}, {-3, 2}, {1, 4},
      {3, 4}, {3, 4}, {-2, 4}, {-4, 4}, {-1, 1}, {-3, 1}, {-2, 4}, {-3, 1},
      {-1, 4}, {-1, 4}, {-1, 4}, {-2, 1}, {-4, 3}, {-2, 1}, {-4, 3}, {-2, 1},
      {-4, 3}, {-4, 1}, {-4, 1}, {-4, 1}, {-2, 3}, {-2, 3}, {-2, 3}, {-1, -2},
      {-3, -2}, {-1, -2}, {-1, -4}, {-1, -2}, {-3, -2}, {-3, -4}, {-3, -2},
      {-1, -4}, {-1, -4}, {-3, -4}, {-3, -4}};
  const int out_pdgs[nprocs][nexternal - ninitial] = {{2, 2}, {4, 4}, {2, -2},
      {4, -4}, {1, 1}, {3, 3}, {1, -1}, {3, -3}, {-2, -2}, {-4, -4}, {-1, -1},
      {-3, -3}, {2, 1}, {4, 3}, {2, 3}, {1, -1}, {3, -3}, {3, -3}, {2, -1}, {4,
      -3}, {2, -3}, {4, 1}, {1, -1}, {4, -1}, {1, -2}, {3, -4}, {1, -4}, {2,
      -2}, {4, -4}, {4, -4}, {3, -2}, {2, -2}, {-2, -1}, {-4, -3}, {-2, -3},
      {-4, -1}, {2, 4}, {4, -4}, {2, -2}, {2, -4}, {4, -2}, {1, 3}, {3, -3},
      {1, -1}, {1, -3}, {3, -1}, {-2, -4}, {-1, -3}, {2, 3}, {4, 3}, {4, 1},
      {4, 3}, {4, 3}, {2, 1}, {2, 3}, {4, 1}, {2, 3}, {1, -3}, {3, -3}, {2,
      -2}, {4, -2}, {3, -1}, {3, -3}, {2, -2}, {2, -4}, {1, -1}, {3, -1}, {2,
      -4}, {4, -4}, {1, -3}, {3, -1}, {2, -4}, {4, -2}, {3, -1}, {2, -4}, {2,
      -3}, {2, -3}, {4, -1}, {4, -1}, {4, -3}, {2, -1}, {2, -1}, {4, -3}, {4,
      -1}, {2, 1}, {4, 1}, {2, 1}, {1, -1}, {1, -3}, {4, -2}, {4, -4}, {1, -3},
      {4, -2}, {2, -1}, {4, -3}, {2, -3}, {1, -4}, {1, -4}, {3, -2}, {3, -2},
      {3, -4}, {1, -2}, {1, -2}, {3, -4}, {3, -2}, {1, -2}, {3, -4}, {1, -4},
      {-2, -3}, {-4, -3}, {-4, -1}, {-4, -3}, {-4, -3}, {-2, -1}, {-2, -3},
      {-4, -1}, {-2, -3}, {-2, -1}, {-4, -1}, {-2, -1}, {2, -2}, {4, -4}, {1,
      -1}, {3, -3}, {2, 1}, {4, 3}, {2, 3}, {1, -1}, {3, -3}, {3, -3}, {2, -1},
      {4, -3}, {2, -3}, {4, 1}, {1, -1}, {4, -1}, {1, -2}, {3, -4}, {1, -4},
      {2, -2}, {4, -4}, {4, -4}, {3, -2}, {2, -2}, {-2, -1}, {-4, -3}, {-2,
      -3}, {-4, -1}, {2, 4}, {4, -4}, {2, -2}, {2, -4}, {4, -2}, {1, 3}, {3,
      -3}, {1, -1}, {1, -3}, {3, -1}, {-2, -4}, {-1, -3}, {2, 3}, {4, 3}, {4,
      1}, {4, 3}, {4, 3}, {2, 1}, {2, 3}, {4, 1}, {2, 3}, {1, -3}, {3, -3}, {2,
      -2}, {4, -2}, {3, -1}, {3, -3}, {2, -2}, {2, -4}, {1, -1}, {3, -1}, {2,
      -4}, {4, -4}, {1, -3}, {3, -1}, {2, -4}, {4, -2}, {3, -1}, {2, -4}, {2,
      -3}, {2, -3}, {4, -1}, {4, -1}, {4, -3}, {2, -1}, {2, -1}, {4, -3}, {4,
      -1}, {2, 1}, {4, 1}, {2, 1}, {1, -1}, {1, -3}, {4, -2}, {4, -4}, {1, -3},
      {4, -2}, {2, -1}, {4, -3}, {2, -3}, {1, -4}, {1, -4}, {3, -2}, {3, -2},
      {3, -4}, {1, -2}, {1, -2}, {3, -4}, {3, -2}, {1, -2}, {3, -4}, {1, -4},
      {-2, -3}, {-4, -3}, {-4, -1}, {-4, -3}, {-4, -3}, {-2, -1}, {-2, -3},
      {-4, -1}, {-2, -3}, {-2, -1}, {-4, -1}, {-2, -1}};

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
void PY8MEs_R221_P2_sm_ckm_qq_qq::setMomenta(vector < vec_double >
    momenta_picked)
{
  if (momenta_picked.size() != nexternal)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setMomenta' of class" << 
    " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Incorrect number of" << 
    " momenta specified." << endl; 
    #endif
    throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
  }
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    if (momenta_picked[i].size() != 4)
    {
      #ifdef DEBUG
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Incorrect number of" << 
      " momenta components specified." << endl; 
      #endif
      throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
    }
    if (momenta_picked[i][0] < 0.0)
    {
      #ifdef DEBUG
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R221_P2_sm_ckm_qq_qq': A momentum was specified" << 
      " with negative energy. Check conventions." << endl; 
      #endif
      throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
    }
    for (unsigned int j = 0; j < 4; j++ )
    {
      p[i][j] = momenta_picked[i][j]; 
    }
  }
}

//--------------------------------------------------------------------------
// Set color configuration to use. An empty vector means sum over all.
void PY8MEs_R221_P2_sm_ckm_qq_qq::setColors(vector<int> colors_picked)
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
    " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Incorrect number" << 
    " of colors specified." << endl; 
    #endif
    throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
  }
  user_colors = vector<int> ((2 * nexternal), 0); 
  for(unsigned int i = 0; i < (2 * nexternal); i++ )
  {
    user_colors[i] = colors_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the helicity configuration to use. Am empty vector means sum over all.
void PY8MEs_R221_P2_sm_ckm_qq_qq::setHelicities(vector<int> helicities_picked) 
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
    " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Incorrect number" << 
    " of helicities specified." << endl; 
    #endif
    throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
  }
  user_helicities = vector<int> (nexternal, 0); 
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    user_helicities[i] = helicities_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the permutation to use (will apply to momenta, colors and helicities)
void PY8MEs_R221_P2_sm_ckm_qq_qq::setPermutation(vector<int> perm_picked) 
{
  if (perm_picked.size() != nexternal)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setPermutations' of class" << 
    " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Incorrect number" << 
    " of permutations specified." << endl; 
    #endif
    throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
  }
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    perm[i] = perm_picked[i]; 
  }
}

// Set the proc_ID to use
void PY8MEs_R221_P2_sm_ckm_qq_qq::setProcID(int procID_picked) 
{
  proc_ID = procID_picked; 
}

//--------------------------------------------------------------------------
// Initialize process.

void PY8MEs_R221_P2_sm_ckm_qq_qq::initProc() 
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
  jamp2 = vector < vec_double > (67); 
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
  jamp2[20] = vector<double> (2, 0.); 
  jamp2[21] = vector<double> (2, 0.); 
  jamp2[22] = vector<double> (2, 0.); 
  jamp2[23] = vector<double> (2, 0.); 
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
  jamp2[36] = vector<double> (1, 0.); 
  jamp2[37] = vector<double> (1, 0.); 
  jamp2[38] = vector<double> (1, 0.); 
  jamp2[39] = vector<double> (1, 0.); 
  jamp2[40] = vector<double> (1, 0.); 
  jamp2[41] = vector<double> (1, 0.); 
  jamp2[42] = vector<double> (1, 0.); 
  jamp2[43] = vector<double> (1, 0.); 
  jamp2[44] = vector<double> (1, 0.); 
  jamp2[45] = vector<double> (1, 0.); 
  jamp2[46] = vector<double> (1, 0.); 
  jamp2[47] = vector<double> (1, 0.); 
  jamp2[48] = vector<double> (1, 0.); 
  jamp2[49] = vector<double> (1, 0.); 
  jamp2[50] = vector<double> (1, 0.); 
  jamp2[51] = vector<double> (1, 0.); 
  jamp2[52] = vector<double> (1, 0.); 
  jamp2[53] = vector<double> (1, 0.); 
  jamp2[54] = vector<double> (1, 0.); 
  jamp2[55] = vector<double> (1, 0.); 
  jamp2[56] = vector<double> (1, 0.); 
  jamp2[57] = vector<double> (1, 0.); 
  jamp2[58] = vector<double> (1, 0.); 
  jamp2[59] = vector<double> (1, 0.); 
  jamp2[60] = vector<double> (1, 0.); 
  jamp2[61] = vector<double> (1, 0.); 
  jamp2[62] = vector<double> (1, 0.); 
  jamp2[63] = vector<double> (1, 0.); 
  jamp2[64] = vector<double> (1, 0.); 
  jamp2[65] = vector<double> (1, 0.); 
  jamp2[66] = vector<double> (1, 0.); 
  all_results = vector < vec_vec_double > (67); 
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
  all_results[20] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[21] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[22] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
      0.));
  all_results[23] = vector < vec_double > (ncomb + 1, vector<double> (2 + 1,
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
  all_results[36] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[37] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[38] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[39] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[40] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[41] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[42] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[43] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[44] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[45] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[46] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[47] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[48] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[49] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[50] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[51] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[52] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[53] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[54] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[55] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[56] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[57] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[58] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[59] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[60] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[61] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[62] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[63] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[64] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[65] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
  all_results[66] = vector < vec_double > (ncomb + 1, vector<double> (1 + 1,
      0.));
}

// Synchronize local variables of the process that depend on the model
// parameters
void PY8MEs_R221_P2_sm_ckm_qq_qq::syncProcModelParams() 
{

  // Instantiate the model class and set parameters that stay fixed during run
  mME[0] = pars->ZERO; 
  mME[1] = pars->ZERO; 
  mME[2] = pars->ZERO; 
  mME[3] = pars->ZERO; 
}

//--------------------------------------------------------------------------
// Setter allowing to force particular values for the external masses
void PY8MEs_R221_P2_sm_ckm_qq_qq::setMasses(vec_double external_masses) 
{

  if (external_masses.size() != mME.size())
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setMasses' of class" << 
    " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Incorrect number of" << 
    " masses specified." << endl; 
    #endif
    throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
  }
  for (unsigned int j = 0; j < mME.size(); j++ )
  {
    mME[j] = external_masses[perm[j]]; 
  }
}

//--------------------------------------------------------------------------
// Getter accessing external masses with the correct ordering
vector<double> PY8MEs_R221_P2_sm_ckm_qq_qq::getMasses() 
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
void PY8MEs_R221_P2_sm_ckm_qq_qq::setExternalMassesMode(int mode) 
{
  if (mode != 0 && mode != 1 && mode != 2)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setExternalMassesMode' of class" << 
    " 'PY8MEs_R221_P2_sm_ckm_qq_qq': Incorrect mode selected :" << mode << 
    ". It must be either 0, 1 or 2" << endl; 
    #endif
    throw PY8MEs_R221_P2_sm_ckm_qq_qq_exception; 
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

double PY8MEs_R221_P2_sm_ckm_qq_qq::sigmaKin() 
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
  for(int i = 0; i < 2; i++ )
    jamp2[20][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[21][i] = 0.; 
  for(int i = 0; i < 2; i++ )
    jamp2[22][i] = 0.; 
  for(int i = 0; i < 2; i++ )
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
  for(int i = 0; i < 1; i++ )
    jamp2[36][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[37][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[38][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[39][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[40][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[41][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[42][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[43][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[44][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[45][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[46][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[47][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[48][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[49][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[50][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[51][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[52][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[53][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[54][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[55][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[56][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[57][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[58][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[59][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[60][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[61][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[62][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[63][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[64][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[65][i] = 0.; 
  for(int i = 0; i < 1; i++ )
    jamp2[66][i] = 0.; 

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
    for(int i = 0; i < 2; i++ )
      jamp2[20][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[21][i] = 0.; 
    for(int i = 0; i < 2; i++ )
      jamp2[22][i] = 0.; 
    for(int i = 0; i < 2; i++ )
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
    for(int i = 0; i < 1; i++ )
      jamp2[36][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[37][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[38][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[39][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[40][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[41][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[42][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[43][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[44][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[45][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[46][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[47][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[48][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[49][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[50][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[51][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[52][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[53][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[54][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[55][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[56][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[57][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[58][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[59][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[60][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[61][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[62][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[63][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[64][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[65][i] = 0.; 
    for(int i = 0; i < 1; i++ )
      jamp2[66][i] = 0.; 

    if (proc_ID == 0)
      t = matrix_221_uu_uu(); 
    if (proc_ID == 1)
      t = matrix_221_uux_uux(); 
    if (proc_ID == 2)
      t = matrix_221_dd_dd(); 
    if (proc_ID == 3)
      t = matrix_221_ddx_ddx(); 
    if (proc_ID == 4)
      t = matrix_221_uxux_uxux(); 
    if (proc_ID == 5)
      t = matrix_221_dxdx_dxdx(); 
    if (proc_ID == 6)
      t = matrix_221_ud_ud(); 
    if (proc_ID == 7)
      t = matrix_221_us_us(); 
    if (proc_ID == 8)
      t = matrix_221_uux_ddx(); 
    if (proc_ID == 9)
      t = matrix_221_uux_ssx(); 
    if (proc_ID == 10)
      t = matrix_221_udx_udx(); 
    if (proc_ID == 11)
      t = matrix_221_usx_usx(); 
    if (proc_ID == 12)
      t = matrix_221_cd_cd(); 
    if (proc_ID == 13)
      t = matrix_221_ccx_ddx(); 
    if (proc_ID == 14)
      t = matrix_221_cdx_cdx(); 
    if (proc_ID == 15)
      t = matrix_221_dux_dux(); 
    if (proc_ID == 16)
      t = matrix_221_dcx_dcx(); 
    if (proc_ID == 17)
      t = matrix_221_ddx_uux(); 
    if (proc_ID == 18)
      t = matrix_221_ddx_ccx(); 
    if (proc_ID == 19)
      t = matrix_221_sux_sux(); 
    if (proc_ID == 20)
      t = matrix_221_ssx_uux(); 
    if (proc_ID == 21)
      t = matrix_221_uxdx_uxdx(); 
    if (proc_ID == 22)
      t = matrix_221_uxsx_uxsx(); 
    if (proc_ID == 23)
      t = matrix_221_cxdx_cxdx(); 
    if (proc_ID == 24)
      t = matrix_221_uc_uc(); 
    if (proc_ID == 25)
      t = matrix_221_uux_ccx(); 
    if (proc_ID == 26)
      t = matrix_221_ucx_ucx(); 
    if (proc_ID == 27)
      t = matrix_221_ds_ds(); 
    if (proc_ID == 28)
      t = matrix_221_ddx_ssx(); 
    if (proc_ID == 29)
      t = matrix_221_dsx_dsx(); 
    if (proc_ID == 30)
      t = matrix_221_uxcx_uxcx(); 
    if (proc_ID == 31)
      t = matrix_221_dxsx_dxsx(); 
    if (proc_ID == 32)
      t = matrix_221_ud_us(); 
    if (proc_ID == 33)
      t = matrix_221_ud_cd(); 
    if (proc_ID == 34)
      t = matrix_221_ud_cs(); 
    if (proc_ID == 35)
      t = matrix_221_us_ud(); 
    if (proc_ID == 36)
      t = matrix_221_us_cd(); 
    if (proc_ID == 37)
      t = matrix_221_uux_dsx(); 
    if (proc_ID == 38)
      t = matrix_221_uux_sdx(); 
    if (proc_ID == 39)
      t = matrix_221_ucx_ddx(); 
    if (proc_ID == 40)
      t = matrix_221_ucx_dsx(); 
    if (proc_ID == 41)
      t = matrix_221_ucx_sdx(); 
    if (proc_ID == 42)
      t = matrix_221_udx_usx(); 
    if (proc_ID == 43)
      t = matrix_221_udx_cdx(); 
    if (proc_ID == 44)
      t = matrix_221_udx_csx(); 
    if (proc_ID == 45)
      t = matrix_221_usx_udx(); 
    if (proc_ID == 46)
      t = matrix_221_usx_cdx(); 
    if (proc_ID == 47)
      t = matrix_221_cd_ud(); 
    if (proc_ID == 48)
      t = matrix_221_cs_ud(); 
    if (proc_ID == 49)
      t = matrix_221_cux_ddx(); 
    if (proc_ID == 50)
      t = matrix_221_cux_dsx(); 
    if (proc_ID == 51)
      t = matrix_221_cdx_udx(); 
    if (proc_ID == 52)
      t = matrix_221_cdx_usx(); 
    if (proc_ID == 53)
      t = matrix_221_dux_dcx(); 
    if (proc_ID == 54)
      t = matrix_221_dux_sux(); 
    if (proc_ID == 55)
      t = matrix_221_dux_scx(); 
    if (proc_ID == 56)
      t = matrix_221_dcx_dux(); 
    if (proc_ID == 57)
      t = matrix_221_dcx_sux(); 
    if (proc_ID == 58)
      t = matrix_221_sux_dux(); 
    if (proc_ID == 59)
      t = matrix_221_sux_dcx(); 
    if (proc_ID == 60)
      t = matrix_221_uxdx_uxsx(); 
    if (proc_ID == 61)
      t = matrix_221_uxdx_cxdx(); 
    if (proc_ID == 62)
      t = matrix_221_uxdx_cxsx(); 
    if (proc_ID == 63)
      t = matrix_221_uxsx_uxdx(); 
    if (proc_ID == 64)
      t = matrix_221_uxsx_cxdx(); 
    if (proc_ID == 65)
      t = matrix_221_cxdx_uxdx(); 
    if (proc_ID == 66)
      t = matrix_221_cxsx_uxdx(); 

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

void PY8MEs_R221_P2_sm_ckm_qq_qq::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  // Calculate all wavefunctions
  ixxxxx(p[perm[0]], mME[0], hel[0], +1, w[0]); 
  ixxxxx(p[perm[1]], mME[1], hel[1], +1, w[1]); 
  oxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  oxxxxx(p[perm[3]], mME[3], hel[3], +1, w[3]); 
  FFV1P0_3(w[0], w[2], pars->GC_2, pars->ZERO, pars->ZERO, w[4]); 
  FFV2_5_3(w[0], w[2], pars->GC_51, pars->GC_58, pars->mdl_MZ, pars->mdl_WZ,
      w[5]);
  FFV1P0_3(w[0], w[3], pars->GC_2, pars->ZERO, pars->ZERO, w[6]); 
  FFV2_5_3(w[0], w[3], pars->GC_51, pars->GC_58, pars->mdl_MZ, pars->mdl_WZ,
      w[7]);
  oxxxxx(p[perm[1]], mME[1], hel[1], -1, w[8]); 
  ixxxxx(p[perm[3]], mME[3], hel[3], -1, w[9]); 
  FFV1P0_3(w[0], w[8], pars->GC_2, pars->ZERO, pars->ZERO, w[10]); 
  FFV2_5_3(w[0], w[8], pars->GC_51, pars->GC_58, pars->mdl_MZ, pars->mdl_WZ,
      w[11]);
  FFV1P0_3(w[0], w[2], pars->GC_1, pars->ZERO, pars->ZERO, w[12]); 
  FFV2_3_3(w[0], w[2], pars->GC_50, pars->GC_58, pars->mdl_MZ, pars->mdl_WZ,
      w[13]);
  FFV1P0_3(w[0], w[3], pars->GC_1, pars->ZERO, pars->ZERO, w[14]); 
  FFV2_3_3(w[0], w[3], pars->GC_50, pars->GC_58, pars->mdl_MZ, pars->mdl_WZ,
      w[15]);
  FFV1P0_3(w[0], w[8], pars->GC_1, pars->ZERO, pars->ZERO, w[16]); 
  FFV2_3_3(w[0], w[8], pars->GC_50, pars->GC_58, pars->mdl_MZ, pars->mdl_WZ,
      w[17]);
  oxxxxx(p[perm[0]], mME[0], hel[0], -1, w[18]); 
  ixxxxx(p[perm[2]], mME[2], hel[2], -1, w[19]); 
  FFV1P0_3(w[19], w[18], pars->GC_2, pars->ZERO, pars->ZERO, w[20]); 
  FFV2_5_3(w[19], w[18], pars->GC_51, pars->GC_58, pars->mdl_MZ, pars->mdl_WZ,
      w[21]);
  FFV1P0_3(w[19], w[8], pars->GC_2, pars->ZERO, pars->ZERO, w[22]); 
  FFV2_5_3(w[19], w[8], pars->GC_51, pars->GC_58, pars->mdl_MZ, pars->mdl_WZ,
      w[23]);
  FFV1P0_3(w[19], w[18], pars->GC_1, pars->ZERO, pars->ZERO, w[24]); 
  FFV2_3_3(w[19], w[18], pars->GC_50, pars->GC_58, pars->mdl_MZ, pars->mdl_WZ,
      w[25]);
  FFV1P0_3(w[19], w[8], pars->GC_1, pars->ZERO, pars->ZERO, w[26]); 
  FFV2_3_3(w[19], w[8], pars->GC_50, pars->GC_58, pars->mdl_MZ, pars->mdl_WZ,
      w[27]);
  FFV2_3(w[0], w[3], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[28]); 
  FFV2_3(w[0], w[3], pars->GC_101, pars->mdl_MW, pars->mdl_WW, w[29]); 
  FFV2_3(w[0], w[2], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[30]); 
  FFV2_3(w[0], w[2], pars->GC_101, pars->mdl_MW, pars->mdl_WW, w[31]); 
  FFV2_3(w[0], w[8], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[32]); 
  FFV2_3(w[0], w[8], pars->GC_101, pars->mdl_MW, pars->mdl_WW, w[33]); 
  FFV2_3(w[0], w[3], pars->GC_44, pars->mdl_MW, pars->mdl_WW, w[34]); 
  FFV2_3(w[0], w[2], pars->GC_44, pars->mdl_MW, pars->mdl_WW, w[35]); 
  FFV2_3(w[0], w[8], pars->GC_44, pars->mdl_MW, pars->mdl_WW, w[36]); 
  FFV1P0_3(w[9], w[8], pars->GC_2, pars->ZERO, pars->ZERO, w[37]); 
  FFV2_5_3(w[9], w[8], pars->GC_51, pars->GC_58, pars->mdl_MZ, pars->mdl_WZ,
      w[38]);
  FFV2_3(w[9], w[2], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[39]); 
  FFV2_3(w[9], w[2], pars->GC_44, pars->mdl_MW, pars->mdl_WW, w[40]); 
  FFV1P0_3(w[9], w[2], pars->GC_2, pars->ZERO, pars->ZERO, w[41]); 
  FFV2_5_3(w[9], w[2], pars->GC_51, pars->GC_58, pars->mdl_MZ, pars->mdl_WZ,
      w[42]);
  FFV2_3(w[9], w[8], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[43]); 
  FFV2_3(w[9], w[8], pars->GC_44, pars->mdl_MW, pars->mdl_WW, w[44]); 
  FFV2_3(w[9], w[2], pars->GC_101, pars->mdl_MW, pars->mdl_WW, w[45]); 
  FFV2_3(w[9], w[8], pars->GC_101, pars->mdl_MW, pars->mdl_WW, w[46]); 
  FFV2_3(w[19], w[8], pars->GC_100, pars->mdl_MW, pars->mdl_WW, w[47]); 
  FFV2_3(w[19], w[8], pars->GC_101, pars->mdl_MW, pars->mdl_WW, w[48]); 
  FFV2_3(w[19], w[8], pars->GC_44, pars->mdl_MW, pars->mdl_WW, w[49]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[1], w[3], w[4], pars->GC_2, amp[0]); 
  FFV2_5_0(w[1], w[3], w[5], pars->GC_51, pars->GC_58, amp[1]); 
  FFV1_0(w[1], w[2], w[6], pars->GC_2, amp[2]); 
  FFV2_5_0(w[1], w[2], w[7], pars->GC_51, pars->GC_58, amp[3]); 
  FFV1_0(w[9], w[2], w[10], pars->GC_2, amp[4]); 
  FFV2_5_0(w[9], w[2], w[11], pars->GC_51, pars->GC_58, amp[5]); 
  FFV1_0(w[9], w[8], w[4], pars->GC_2, amp[6]); 
  FFV2_5_0(w[9], w[8], w[5], pars->GC_51, pars->GC_58, amp[7]); 
  FFV1_0(w[1], w[3], w[12], pars->GC_1, amp[8]); 
  FFV2_3_0(w[1], w[3], w[13], pars->GC_50, pars->GC_58, amp[9]); 
  FFV1_0(w[1], w[2], w[14], pars->GC_1, amp[10]); 
  FFV2_3_0(w[1], w[2], w[15], pars->GC_50, pars->GC_58, amp[11]); 
  FFV1_0(w[9], w[2], w[16], pars->GC_1, amp[12]); 
  FFV2_3_0(w[9], w[2], w[17], pars->GC_50, pars->GC_58, amp[13]); 
  FFV1_0(w[9], w[8], w[12], pars->GC_1, amp[14]); 
  FFV2_3_0(w[9], w[8], w[13], pars->GC_50, pars->GC_58, amp[15]); 
  FFV1_0(w[9], w[8], w[20], pars->GC_2, amp[16]); 
  FFV2_5_0(w[9], w[8], w[21], pars->GC_51, pars->GC_58, amp[17]); 
  FFV1_0(w[9], w[18], w[22], pars->GC_2, amp[18]); 
  FFV2_5_0(w[9], w[18], w[23], pars->GC_51, pars->GC_58, amp[19]); 
  FFV1_0(w[9], w[8], w[24], pars->GC_1, amp[20]); 
  FFV2_3_0(w[9], w[8], w[25], pars->GC_50, pars->GC_58, amp[21]); 
  FFV1_0(w[9], w[18], w[26], pars->GC_1, amp[22]); 
  FFV2_3_0(w[9], w[18], w[27], pars->GC_50, pars->GC_58, amp[23]); 
  FFV1_0(w[1], w[3], w[4], pars->GC_1, amp[24]); 
  FFV2_3_0(w[1], w[3], w[5], pars->GC_50, pars->GC_58, amp[25]); 
  FFV2_0(w[1], w[2], w[28], pars->GC_100, amp[26]); 
  FFV1_0(w[1], w[3], w[4], pars->GC_1, amp[27]); 
  FFV2_3_0(w[1], w[3], w[5], pars->GC_50, pars->GC_58, amp[28]); 
  FFV2_0(w[1], w[2], w[29], pars->GC_101, amp[29]); 
  FFV1_0(w[9], w[2], w[10], pars->GC_1, amp[30]); 
  FFV2_3_0(w[9], w[2], w[11], pars->GC_50, pars->GC_58, amp[31]); 
  FFV2_0(w[9], w[8], w[30], pars->GC_100, amp[32]); 
  FFV1_0(w[9], w[2], w[10], pars->GC_1, amp[33]); 
  FFV2_3_0(w[9], w[2], w[11], pars->GC_50, pars->GC_58, amp[34]); 
  FFV2_0(w[9], w[8], w[31], pars->GC_101, amp[35]); 
  FFV1_0(w[9], w[8], w[4], pars->GC_1, amp[36]); 
  FFV2_3_0(w[9], w[8], w[5], pars->GC_50, pars->GC_58, amp[37]); 
  FFV2_0(w[9], w[2], w[32], pars->GC_100, amp[38]); 
  FFV1_0(w[9], w[8], w[4], pars->GC_1, amp[39]); 
  FFV2_3_0(w[9], w[8], w[5], pars->GC_50, pars->GC_58, amp[40]); 
  FFV2_0(w[9], w[2], w[33], pars->GC_101, amp[41]); 
  FFV1_0(w[1], w[3], w[4], pars->GC_1, amp[42]); 
  FFV2_3_0(w[1], w[3], w[5], pars->GC_50, pars->GC_58, amp[43]); 
  FFV2_0(w[1], w[2], w[34], pars->GC_44, amp[44]); 
  FFV1_0(w[9], w[2], w[10], pars->GC_1, amp[45]); 
  FFV2_3_0(w[9], w[2], w[11], pars->GC_50, pars->GC_58, amp[46]); 
  FFV2_0(w[9], w[8], w[35], pars->GC_44, amp[47]); 
  FFV1_0(w[9], w[8], w[4], pars->GC_1, amp[48]); 
  FFV2_3_0(w[9], w[8], w[5], pars->GC_50, pars->GC_58, amp[49]); 
  FFV2_0(w[9], w[2], w[36], pars->GC_44, amp[50]); 
  FFV1_0(w[0], w[2], w[37], pars->GC_1, amp[51]); 
  FFV2_3_0(w[0], w[2], w[38], pars->GC_50, pars->GC_58, amp[52]); 
  FFV2_0(w[0], w[8], w[39], pars->GC_100, amp[53]); 
  FFV1_0(w[0], w[2], w[37], pars->GC_1, amp[54]); 
  FFV2_3_0(w[0], w[2], w[38], pars->GC_50, pars->GC_58, amp[55]); 
  FFV2_0(w[0], w[8], w[40], pars->GC_44, amp[56]); 
  FFV1_0(w[0], w[8], w[41], pars->GC_1, amp[57]); 
  FFV2_3_0(w[0], w[8], w[42], pars->GC_50, pars->GC_58, amp[58]); 
  FFV2_0(w[0], w[2], w[43], pars->GC_100, amp[59]); 
  FFV1_0(w[0], w[8], w[41], pars->GC_1, amp[60]); 
  FFV2_3_0(w[0], w[8], w[42], pars->GC_50, pars->GC_58, amp[61]); 
  FFV2_0(w[0], w[2], w[44], pars->GC_44, amp[62]); 
  FFV1_0(w[0], w[2], w[37], pars->GC_1, amp[63]); 
  FFV2_3_0(w[0], w[2], w[38], pars->GC_50, pars->GC_58, amp[64]); 
  FFV2_0(w[0], w[8], w[45], pars->GC_101, amp[65]); 
  FFV1_0(w[0], w[8], w[41], pars->GC_1, amp[66]); 
  FFV2_3_0(w[0], w[8], w[42], pars->GC_50, pars->GC_58, amp[67]); 
  FFV2_0(w[0], w[2], w[46], pars->GC_101, amp[68]); 
  FFV1_0(w[9], w[8], w[20], pars->GC_1, amp[69]); 
  FFV2_3_0(w[9], w[8], w[21], pars->GC_50, pars->GC_58, amp[70]); 
  FFV2_0(w[9], w[18], w[47], pars->GC_100, amp[71]); 
  FFV1_0(w[9], w[8], w[20], pars->GC_1, amp[72]); 
  FFV2_3_0(w[9], w[8], w[21], pars->GC_50, pars->GC_58, amp[73]); 
  FFV2_0(w[9], w[18], w[48], pars->GC_101, amp[74]); 
  FFV1_0(w[9], w[8], w[20], pars->GC_1, amp[75]); 
  FFV2_3_0(w[9], w[8], w[21], pars->GC_50, pars->GC_58, amp[76]); 
  FFV2_0(w[9], w[18], w[49], pars->GC_44, amp[77]); 
  FFV1_0(w[9], w[2], w[10], pars->GC_2, amp[78]); 
  FFV2_5_0(w[9], w[2], w[11], pars->GC_51, pars->GC_58, amp[79]); 
  FFV1_0(w[9], w[8], w[4], pars->GC_2, amp[80]); 
  FFV2_5_0(w[9], w[8], w[5], pars->GC_51, pars->GC_58, amp[81]); 
  FFV1_0(w[1], w[3], w[12], pars->GC_1, amp[82]); 
  FFV2_3_0(w[1], w[3], w[13], pars->GC_50, pars->GC_58, amp[83]); 
  FFV1_0(w[9], w[2], w[16], pars->GC_1, amp[84]); 
  FFV2_3_0(w[9], w[2], w[17], pars->GC_50, pars->GC_58, amp[85]); 
  FFV1_0(w[9], w[8], w[12], pars->GC_1, amp[86]); 
  FFV2_3_0(w[9], w[8], w[13], pars->GC_50, pars->GC_58, amp[87]); 
  FFV1_0(w[9], w[8], w[20], pars->GC_2, amp[88]); 
  FFV2_5_0(w[9], w[8], w[21], pars->GC_51, pars->GC_58, amp[89]); 
  FFV1_0(w[9], w[8], w[24], pars->GC_1, amp[90]); 
  FFV2_3_0(w[9], w[8], w[25], pars->GC_50, pars->GC_58, amp[91]); 
  FFV2_0(w[1], w[2], w[29], pars->GC_100, amp[92]); 
  FFV2_0(w[1], w[2], w[28], pars->GC_44, amp[93]); 
  FFV2_0(w[1], w[2], w[29], pars->GC_44, amp[94]); 
  FFV2_0(w[1], w[2], w[28], pars->GC_101, amp[95]); 
  FFV2_0(w[1], w[2], w[28], pars->GC_100, amp[96]); 
  FFV2_0(w[9], w[8], w[30], pars->GC_101, amp[97]); 
  FFV2_0(w[9], w[8], w[31], pars->GC_100, amp[98]); 
  FFV2_0(w[9], w[8], w[30], pars->GC_44, amp[99]); 
  FFV2_0(w[9], w[8], w[30], pars->GC_100, amp[100]); 
  FFV2_0(w[9], w[8], w[31], pars->GC_44, amp[101]); 
  FFV2_0(w[9], w[2], w[32], pars->GC_101, amp[102]); 
  FFV2_0(w[9], w[2], w[32], pars->GC_44, amp[103]); 
  FFV2_0(w[9], w[2], w[32], pars->GC_100, amp[104]); 
  FFV2_0(w[9], w[2], w[33], pars->GC_100, amp[105]); 
  FFV2_0(w[9], w[2], w[33], pars->GC_44, amp[106]); 
  FFV2_0(w[1], w[2], w[34], pars->GC_100, amp[107]); 
  FFV2_0(w[1], w[2], w[34], pars->GC_101, amp[108]); 
  FFV2_0(w[9], w[8], w[35], pars->GC_100, amp[109]); 
  FFV2_0(w[9], w[8], w[35], pars->GC_101, amp[110]); 
  FFV2_0(w[9], w[2], w[36], pars->GC_100, amp[111]); 
  FFV2_0(w[9], w[2], w[36], pars->GC_101, amp[112]); 
  FFV2_0(w[0], w[8], w[40], pars->GC_100, amp[113]); 
  FFV2_0(w[0], w[8], w[45], pars->GC_100, amp[114]); 
  FFV2_0(w[0], w[8], w[39], pars->GC_100, amp[115]); 
  FFV2_0(w[0], w[8], w[39], pars->GC_44, amp[116]); 
  FFV2_0(w[0], w[8], w[45], pars->GC_44, amp[117]); 
  FFV2_0(w[0], w[8], w[39], pars->GC_101, amp[118]); 
  FFV2_0(w[0], w[8], w[40], pars->GC_101, amp[119]); 
  FFV2_0(w[9], w[18], w[47], pars->GC_101, amp[120]); 
  FFV2_0(w[9], w[18], w[49], pars->GC_100, amp[121]); 
  FFV2_0(w[9], w[18], w[49], pars->GC_101, amp[122]); 
  FFV2_0(w[9], w[18], w[48], pars->GC_100, amp[123]); 
  FFV2_0(w[9], w[18], w[47], pars->GC_100, amp[124]); 
  FFV2_0(w[9], w[18], w[47], pars->GC_44, amp[125]); 
  FFV2_0(w[9], w[18], w[48], pars->GC_44, amp[126]); 


}
double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uu_uu() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 4;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[0] - amp[1]; 
  jamp[1] = +amp[2] + amp[3]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uux_uux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 4;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[4] - amp[5]; 
  jamp[1] = +amp[6] + amp[7]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_dd_dd() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 4;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[8] - amp[9]; 
  jamp[1] = +amp[10] + amp[11]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ddx_ddx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 4;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[12] - amp[13]; 
  jamp[1] = +amp[14] + amp[15]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uxux_uxux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 4;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[16] - amp[17]; 
  jamp[1] = +amp[18] + amp[19]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_dxdx_dxdx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 4;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[20] - amp[21]; 
  jamp[1] = +amp[22] + amp[23]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ud_ud() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[24] - amp[25]; 
  jamp[1] = +amp[26]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_us_us() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[27] - amp[28]; 
  jamp[1] = +amp[29]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uux_ddx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[30] - amp[31]; 
  jamp[1] = +amp[32]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uux_ssx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[33] - amp[34]; 
  jamp[1] = +amp[35]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_udx_udx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[38]; 
  jamp[1] = +amp[36] + amp[37]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_usx_usx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[41]; 
  jamp[1] = +amp[39] + amp[40]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_cd_cd() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[42] - amp[43]; 
  jamp[1] = +amp[44]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ccx_ddx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[45] - amp[46]; 
  jamp[1] = +amp[47]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_cdx_cdx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[50]; 
  jamp[1] = +amp[48] + amp[49]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_dux_dux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[53]; 
  jamp[1] = +amp[51] + amp[52]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_dcx_dcx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[56]; 
  jamp[1] = +amp[54] + amp[55]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ddx_uux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[57] - amp[58]; 
  jamp[1] = +amp[59]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ddx_ccx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[60] - amp[61]; 
  jamp[1] = +amp[62]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_sux_sux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[65]; 
  jamp[1] = +amp[63] + amp[64]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ssx_uux() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[66] - amp[67]; 
  jamp[1] = +amp[68]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uxdx_uxdx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[69] - amp[70]; 
  jamp[1] = +amp[71]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uxsx_uxsx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[72] - amp[73]; 
  jamp[1] = +amp[74]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_cxdx_cxdx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 3;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[75] - amp[76]; 
  jamp[1] = +amp[77]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uc_uc() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[0] - amp[1]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uux_ccx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[78] - amp[79]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ucx_ucx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[80] + amp[81]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ds_ds() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[82] - amp[83]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ddx_ssx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[84] - amp[85]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_dsx_dsx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = +amp[86] + amp[87]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uxcx_uxcx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[88] - amp[89]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_dxsx_dxsx() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 1; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{9}}; 

  // Calculate color flows
  jamp[0] = -amp[90] - amp[91]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ud_us() 
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
  jamp[0] = +amp[92]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ud_cd() 
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
  jamp[0] = +amp[93]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ud_cs() 
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
  jamp[0] = +amp[94]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_us_ud() 
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
  jamp[0] = +amp[95]; 

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

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_us_cd() 
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
  jamp[0] = +amp[96]; 

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
    jamp2[36][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uux_dsx() 
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
  jamp[0] = +amp[97]; 

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
    jamp2[37][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uux_sdx() 
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
  jamp[0] = +amp[98]; 

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
    jamp2[38][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ucx_ddx() 
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
  jamp[0] = +amp[99]; 

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
    jamp2[39][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ucx_dsx() 
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
  jamp[0] = +amp[100]; 

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
    jamp2[40][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_ucx_sdx() 
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
  jamp[0] = +amp[101]; 

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
    jamp2[41][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_udx_usx() 
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
  jamp[0] = -amp[102]; 

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
    jamp2[42][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_udx_cdx() 
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
  jamp[0] = -amp[103]; 

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
    jamp2[43][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_udx_csx() 
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
  jamp[0] = -amp[104]; 

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
    jamp2[44][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_usx_udx() 
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
  jamp[0] = -amp[105]; 

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
    jamp2[45][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_usx_cdx() 
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
  jamp[0] = -amp[106]; 

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
    jamp2[46][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_cd_ud() 
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
  jamp[0] = +amp[107]; 

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
    jamp2[47][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_cs_ud() 
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
  jamp[0] = +amp[108]; 

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
    jamp2[48][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_cux_ddx() 
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
  jamp[0] = +amp[109]; 

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
    jamp2[49][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_cux_dsx() 
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
  jamp[0] = +amp[110]; 

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
    jamp2[50][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_cdx_udx() 
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
  jamp[0] = -amp[111]; 

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
    jamp2[51][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_cdx_usx() 
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
  jamp[0] = -amp[112]; 

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
    jamp2[52][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_dux_dcx() 
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
  jamp[0] = -amp[113]; 

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
    jamp2[53][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_dux_sux() 
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
  jamp[0] = -amp[114]; 

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
    jamp2[54][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_dux_scx() 
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
  jamp[0] = -amp[115]; 

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
    jamp2[55][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_dcx_dux() 
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
  jamp[0] = -amp[116]; 

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
    jamp2[56][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_dcx_sux() 
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
  jamp[0] = -amp[117]; 

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
    jamp2[57][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_sux_dux() 
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
  jamp[0] = -amp[118]; 

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
    jamp2[58][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_sux_dcx() 
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
  jamp[0] = -amp[119]; 

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
    jamp2[59][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uxdx_uxsx() 
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
  jamp[0] = +amp[120]; 

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
    jamp2[60][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uxdx_cxdx() 
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
  jamp[0] = +amp[121]; 

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
    jamp2[61][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uxdx_cxsx() 
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
  jamp[0] = +amp[122]; 

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
    jamp2[62][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uxsx_uxdx() 
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
  jamp[0] = +amp[123]; 

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
    jamp2[63][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_uxsx_cxdx() 
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
  jamp[0] = +amp[124]; 

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
    jamp2[64][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_cxdx_uxdx() 
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
  jamp[0] = +amp[125]; 

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
    jamp2[65][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R221_P2_sm_ckm_qq_qq::matrix_221_cxsx_uxdx() 
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
  jamp[0] = +amp[126]; 

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
    jamp2[66][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}


}  // end namespace PY8MEs_namespace

