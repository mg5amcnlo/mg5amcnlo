//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "PY8MEs_R906_P4_heft_ckm_qq_hqq.h"
#include "HelAmps_heft_ckm.h"

using namespace Pythia8_heft_ckm; 

namespace PY8MEs_namespace 
{
//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u > h u u HIG=0 HIW=0 QCD=0 / b @906
// Process: c c > h c c HIG=0 HIW=0 QCD=0 / b @906
// Process: u d > h u d HIG=0 HIW=0 QCD=0 / b @906
// Process: c s > h c s HIG=0 HIW=0 QCD=0 / b @906
// Process: u s > h u s HIG=0 HIW=0 QCD=0 / b @906
// Process: u u~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c c~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u u~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c c~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u u~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u d~ > h u d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c s~ > h c s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u s~ > h u s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c d > h c d HIG=0 HIW=0 QCD=0 / b @906
// Process: c c~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c d~ > h c d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d > h d d HIG=0 HIW=0 QCD=0 / b @906
// Process: s s > h s s HIG=0 HIW=0 QCD=0 / b @906
// Process: d u~ > h d u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s c~ > h s c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d c~ > h d c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s s~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s s~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s u~ > h s u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s s~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ u~ > h u~ u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ c~ > h c~ c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ d~ > h u~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ s~ > h c~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ s~ > h u~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ d~ > h c~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d~ d~ > h d~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s~ s~ > h s~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u c > h u c HIG=0 HIW=0 QCD=0 / b @906
// Process: u d > h u s HIG=0 HIW=0 QCD=0 / b @906
// Process: u s > h c s HIG=0 HIW=0 QCD=0 / b @906
// Process: u d > h c d HIG=0 HIW=0 QCD=0 / b @906
// Process: c d > h c s HIG=0 HIW=0 QCD=0 / b @906
// Process: u d > h c s HIG=0 HIW=0 QCD=0 / b @906
// Process: u s > h u d HIG=0 HIW=0 QCD=0 / b @906
// Process: c s > h u s HIG=0 HIW=0 QCD=0 / b @906
// Process: u s > h c d HIG=0 HIW=0 QCD=0 / b @906
// Process: c d > h u s HIG=0 HIW=0 QCD=0 / b @906
// Process: u u~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c c~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u u~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c u~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d s~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s s~ > h c u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u u~ > h s d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u c~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s d~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s s~ > h u c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u c~ > h u c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c u~ > h c u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u c~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c c~ > h s d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d~ > h u c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s d~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u c~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c u~ > h s d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d s~ > h u c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s d~ > h c u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u c~ > h s d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s d~ > h u c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u d~ > h u s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c s~ > h u s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u d~ > h c d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c s~ > h c d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u d~ > h c s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c s~ > h u d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u s~ > h u d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u s~ > h c s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u s~ > h c d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c d > h u d HIG=0 HIW=0 QCD=0 / b @906
// Process: c s > h c d HIG=0 HIW=0 QCD=0 / b @906
// Process: c s > h u d HIG=0 HIW=0 QCD=0 / b @906
// Process: c u~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c c~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d~ > h c u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d s~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c u~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d s~ > h c u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c d~ > h u d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c d~ > h c s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c d~ > h u s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d s > h d s HIG=0 HIW=0 QCD=0 / b @906
// Process: d u~ > h d c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s c~ > h d c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d u~ > h s u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s c~ > h s u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d u~ > h s c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s c~ > h d u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d c~ > h d u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d c~ > h s c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d c~ > h s u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s s~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d s~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s d~ > h s d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s u~ > h d u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s u~ > h s c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s u~ > h d c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ c~ > h u~ c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ d~ > h u~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ s~ > h c~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ d~ > h c~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ d~ > h c~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ d~ > h c~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ s~ > h u~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ s~ > h u~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ s~ > h c~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ d~ > h u~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ d~ > h u~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ s~ > h c~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ s~ > h u~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d~ s~ > h d~ s~ HIG=0 HIW=0 QCD=0 / b @906

// Exception class
class PY8MEs_R906_P4_heft_ckm_qq_hqqException : public exception
{
  virtual const char * what() const throw()
  {
    return "Exception in class 'PY8MEs_R906_P4_heft_ckm_qq_hqq'."; 
  }
}
PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 

std::set<int> PY8MEs_R906_P4_heft_ckm_qq_hqq::s_channel_proc = std::set<int>
    (createset<int> ());

int PY8MEs_R906_P4_heft_ckm_qq_hqq::helicities[ncomb][nexternal] = {{-1, -1, 0,
    -1, -1}, {-1, -1, 0, -1, 1}, {-1, -1, 0, 1, -1}, {-1, -1, 0, 1, 1}, {-1, 1,
    0, -1, -1}, {-1, 1, 0, -1, 1}, {-1, 1, 0, 1, -1}, {-1, 1, 0, 1, 1}, {1, -1,
    0, -1, -1}, {1, -1, 0, -1, 1}, {1, -1, 0, 1, -1}, {1, -1, 0, 1, 1}, {1, 1,
    0, -1, -1}, {1, 1, 0, -1, 1}, {1, 1, 0, 1, -1}, {1, 1, 0, 1, 1}};

// Normalization factors the various processes
// Denominators: spins, colors and identical particles
int PY8MEs_R906_P4_heft_ckm_qq_hqq::denom_colors[nprocesses] = {9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
int PY8MEs_R906_P4_heft_ckm_qq_hqq::denom_hels[nprocesses] = {4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
int PY8MEs_R906_P4_heft_ckm_qq_hqq::denom_iden[nprocesses] = {2, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

//--------------------------------------------------------------------------
// Color config initialization
void PY8MEs_R906_P4_heft_ckm_qq_hqq::initColorConfigs() 
{
  color_configs = vector < vec_vec_int > (); 
  jamp_nc_relative_power = vector < vec_int > (); 

  // Color flows of process Process: u u > h u u HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[0].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[0].push_back(0); 
  // JAMP #1
  color_configs[0].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[0].push_back(0); 

  // Color flows of process Process: u d > h u d HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[1].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[1].push_back(0); 
  // JAMP #1
  color_configs[1].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[1].push_back(0); 

  // Color flows of process Process: u s > h u s HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[2].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[2].push_back(0); 
  // JAMP #1
  color_configs[2].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[2].push_back(0); 

  // Color flows of process Process: u u~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[3].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[3].push_back(0); 
  // JAMP #1
  color_configs[3].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[3].push_back(0); 

  // Color flows of process Process: u u~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[4].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[4].push_back(0); 
  // JAMP #1
  color_configs[4].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[4].push_back(0); 

  // Color flows of process Process: u u~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[5].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[5].push_back(0); 
  // JAMP #1
  color_configs[5].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[5].push_back(0); 

  // Color flows of process Process: u d~ > h u d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[6].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[6].push_back(0); 
  // JAMP #1
  color_configs[6].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[6].push_back(0); 

  // Color flows of process Process: u s~ > h u s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[7].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[7].push_back(0); 
  // JAMP #1
  color_configs[7].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[7].push_back(0); 

  // Color flows of process Process: c d > h c d HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[8].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[8].push_back(0); 
  // JAMP #1
  color_configs[8].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[8].push_back(0); 

  // Color flows of process Process: c c~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[9].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[9].push_back(0); 
  // JAMP #1
  color_configs[9].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[9].push_back(0); 

  // Color flows of process Process: c d~ > h c d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[10].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[10].push_back(0); 
  // JAMP #1
  color_configs[10].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[10].push_back(0); 

  // Color flows of process Process: d d > h d d HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[11].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[11].push_back(0); 
  // JAMP #1
  color_configs[11].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[11].push_back(0); 

  // Color flows of process Process: d u~ > h d u~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[12].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[12].push_back(0); 
  // JAMP #1
  color_configs[12].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[12].push_back(0); 

  // Color flows of process Process: d c~ > h d c~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[13].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[13].push_back(0); 
  // JAMP #1
  color_configs[13].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[13].push_back(0); 

  // Color flows of process Process: d d~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[14].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[14].push_back(0); 
  // JAMP #1
  color_configs[14].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[14].push_back(0); 

  // Color flows of process Process: d d~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[15].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[15].push_back(0); 
  // JAMP #1
  color_configs[15].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[15].push_back(0); 

  // Color flows of process Process: d d~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[16].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[16].push_back(0); 
  // JAMP #1
  color_configs[16].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[16].push_back(0); 

  // Color flows of process Process: s u~ > h s u~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[17].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[17].push_back(0); 
  // JAMP #1
  color_configs[17].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[17].push_back(0); 

  // Color flows of process Process: s s~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[18].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[18].push_back(0); 
  // JAMP #1
  color_configs[18].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[18].push_back(0); 

  // Color flows of process Process: u~ u~ > h u~ u~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[19].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(1)(0)(2)));
  jamp_nc_relative_power[19].push_back(0); 
  // JAMP #1
  color_configs[19].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(2)(0)(1)));
  jamp_nc_relative_power[19].push_back(0); 

  // Color flows of process Process: u~ d~ > h u~ d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[20].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(1)(0)(2)));
  jamp_nc_relative_power[20].push_back(0); 
  // JAMP #1
  color_configs[20].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(2)(0)(1)));
  jamp_nc_relative_power[20].push_back(0); 

  // Color flows of process Process: u~ s~ > h u~ s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[21].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(1)(0)(2)));
  jamp_nc_relative_power[21].push_back(0); 
  // JAMP #1
  color_configs[21].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(2)(0)(1)));
  jamp_nc_relative_power[21].push_back(0); 

  // Color flows of process Process: c~ d~ > h c~ d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[22].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(1)(0)(2)));
  jamp_nc_relative_power[22].push_back(0); 
  // JAMP #1
  color_configs[22].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(2)(0)(1)));
  jamp_nc_relative_power[22].push_back(0); 

  // Color flows of process Process: d~ d~ > h d~ d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[23].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(1)(0)(2)));
  jamp_nc_relative_power[23].push_back(0); 
  // JAMP #1
  color_configs[23].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(2)(0)(1)));
  jamp_nc_relative_power[23].push_back(0); 

  // Color flows of process Process: u c > h u c HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[24].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[24].push_back(0); 

  // Color flows of process Process: u d > h u s HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[25].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[25].push_back(0); 

  // Color flows of process Process: u d > h c d HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[26].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[26].push_back(0); 

  // Color flows of process Process: u d > h c s HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[27].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[27].push_back(0); 

  // Color flows of process Process: u s > h u d HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[28].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[28].push_back(0); 

  // Color flows of process Process: u s > h c d HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[29].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[29].push_back(0); 

  // Color flows of process Process: u u~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[30].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[30].push_back(0); 

  // Color flows of process Process: u u~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[31].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[31].push_back(0); 

  // Color flows of process Process: u u~ > h s d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[32].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[32].push_back(0); 

  // Color flows of process Process: u c~ > h u c~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[33].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[33].push_back(0); 

  // Color flows of process Process: u c~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[34].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[34].push_back(0); 

  // Color flows of process Process: u c~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[35].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[35].push_back(0); 

  // Color flows of process Process: u c~ > h s d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[36].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[36].push_back(0); 

  // Color flows of process Process: u d~ > h u s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[37].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[37].push_back(0); 

  // Color flows of process Process: u d~ > h c d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[38].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[38].push_back(0); 

  // Color flows of process Process: u d~ > h c s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[39].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[39].push_back(0); 

  // Color flows of process Process: u s~ > h u d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[40].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[40].push_back(0); 

  // Color flows of process Process: u s~ > h c d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[41].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[41].push_back(0); 

  // Color flows of process Process: c d > h u d HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[42].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[42].push_back(0); 

  // Color flows of process Process: c s > h u d HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[43].push_back(vec_int(createvector<int>
      (2)(0)(1)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[43].push_back(0); 

  // Color flows of process Process: c u~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[44].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[44].push_back(0); 

  // Color flows of process Process: c u~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[45].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[45].push_back(0); 

  // Color flows of process Process: c d~ > h u d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[46].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[46].push_back(0); 

  // Color flows of process Process: c d~ > h u s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[47].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[47].push_back(0); 

  // Color flows of process Process: d s > h d s HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[48].push_back(vec_int(createvector<int>
      (1)(0)(2)(0)(0)(0)(1)(0)(2)(0)));
  jamp_nc_relative_power[48].push_back(0); 

  // Color flows of process Process: d u~ > h d c~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[49].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[49].push_back(0); 

  // Color flows of process Process: d u~ > h s u~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[50].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[50].push_back(0); 

  // Color flows of process Process: d u~ > h s c~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[51].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[51].push_back(0); 

  // Color flows of process Process: d c~ > h d u~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[52].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[52].push_back(0); 

  // Color flows of process Process: d c~ > h s u~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[53].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[53].push_back(0); 

  // Color flows of process Process: d d~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[54].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[54].push_back(0); 

  // Color flows of process Process: d s~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[55].push_back(vec_int(createvector<int>
      (2)(0)(0)(1)(0)(0)(2)(0)(0)(1)));
  jamp_nc_relative_power[55].push_back(0); 

  // Color flows of process Process: s u~ > h d u~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[56].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[56].push_back(0); 

  // Color flows of process Process: s u~ > h d c~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[57].push_back(vec_int(createvector<int>
      (1)(0)(0)(1)(0)(0)(2)(0)(0)(2)));
  jamp_nc_relative_power[57].push_back(0); 

  // Color flows of process Process: u~ c~ > h u~ c~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[58].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(1)(0)(2)));
  jamp_nc_relative_power[58].push_back(0); 

  // Color flows of process Process: u~ d~ > h u~ s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[59].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(2)(0)(1)));
  jamp_nc_relative_power[59].push_back(0); 

  // Color flows of process Process: u~ d~ > h c~ d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[60].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(2)(0)(1)));
  jamp_nc_relative_power[60].push_back(0); 

  // Color flows of process Process: u~ d~ > h c~ s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[61].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(2)(0)(1)));
  jamp_nc_relative_power[61].push_back(0); 

  // Color flows of process Process: u~ s~ > h u~ d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[62].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(2)(0)(1)));
  jamp_nc_relative_power[62].push_back(0); 

  // Color flows of process Process: u~ s~ > h c~ d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[63].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(2)(0)(1)));
  jamp_nc_relative_power[63].push_back(0); 

  // Color flows of process Process: c~ d~ > h u~ d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[64].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(2)(0)(1)));
  jamp_nc_relative_power[64].push_back(0); 

  // Color flows of process Process: c~ s~ > h u~ d~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[65].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(2)(0)(1)));
  jamp_nc_relative_power[65].push_back(0); 

  // Color flows of process Process: d~ s~ > h d~ s~ HIG=0 HIW=0 QCD=0 / b @906
  color_configs.push_back(vec_vec_int()); 
  jamp_nc_relative_power.push_back(vec_int()); 
  // JAMP #0
  color_configs[66].push_back(vec_int(createvector<int>
      (0)(1)(0)(2)(0)(0)(0)(1)(0)(2)));
  jamp_nc_relative_power[66].push_back(0); 
}

//--------------------------------------------------------------------------
// Destructor.
PY8MEs_R906_P4_heft_ckm_qq_hqq::~PY8MEs_R906_P4_heft_ckm_qq_hqq() 
{
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    delete[] p[i]; 
    p[i] = NULL; 
  }
}

//--------------------------------------------------------------------------
// Invert the permutation mapping
vector<int> PY8MEs_R906_P4_heft_ckm_qq_hqq::invert_mapping(vector<int> mapping) 
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
vector < vec_int > PY8MEs_R906_P4_heft_ckm_qq_hqq::getHelicityConfigs(vector<int> permutation) 
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
vector < vec_int > PY8MEs_R906_P4_heft_ckm_qq_hqq::getColorConfigs(int
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
int PY8MEs_R906_P4_heft_ckm_qq_hqq::getColorFlowRelativeNCPower(int
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
vector<int> PY8MEs_R906_P4_heft_ckm_qq_hqq::getHelicityConfigForID(int hel_ID,
    vector<int> permutation)
{
  if (hel_ID < 0 || hel_ID >= ncomb)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'getHelicityConfigForID' of class" << 
    " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Specified helicity ID '" << 
    hel_ID <<  "' cannot be found." << endl; 
    #endif
    throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
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
int PY8MEs_R906_P4_heft_ckm_qq_hqq::getHelicityIDForConfig(vector<int>
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
      " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Specified helicity" << 
      " configuration cannot be found." << endl; 
      #endif
      throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
    }
  }
  return user_ihel; 
}


//--------------------------------------------------------------------------
// Implements the map Color ID -> Color Config
vector<int> PY8MEs_R906_P4_heft_ckm_qq_hqq::getColorConfigForID(int color_ID,
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
  if (color_ID < 0 || color_ID >= int(color_configs[chosenProcID].size()))
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'getColorConfigForID' of class" << 
    " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Specified color ID '" << 
    color_ID <<  "' cannot be found." << endl; 
    #endif
    throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
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
int PY8MEs_R906_P4_heft_ckm_qq_hqq::getColorIDForConfig(vector<int>
    color_config, int specify_proc_ID, vector<int> permutation)
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
            " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': A color line could " << 
            " not be closed." << endl; 
            #endif
            throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
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
      " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Specified color" << 
      " configuration cannot be found." << endl; 
      #endif
      throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
    }
  }
  return user_icol; 
}

//--------------------------------------------------------------------------
// Returns all result previously computed in SigmaKin
vector < vec_double > PY8MEs_R906_P4_heft_ckm_qq_hqq::getAllResults(int
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
double PY8MEs_R906_P4_heft_ckm_qq_hqq::getResult(int helicity_ID, int color_ID,
    int specify_proc_ID)
{
  if (helicity_ID < - 1 || helicity_ID >= ncomb)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'getResult' of class" << 
    " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Specified helicity ID '" << 
    helicity_ID <<  "' configuration cannot be found." << endl; 
    #endif
    throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
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
    " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Specified color ID '" << 
    color_ID <<  "' configuration cannot be found." << endl; 
    #endif
    throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
  }
  return all_results[chosenProcID][helicity_ID + 1][color_ID + 1]; 
}

//--------------------------------------------------------------------------
// Check for the availability of the requested process and if available,
// If available, this returns the corresponding permutation and Proc_ID to use.
// If not available, this returns a negative Proc_ID.
pair < vector<int> , int > PY8MEs_R906_P4_heft_ckm_qq_hqq::static_getPY8ME(vector<int> initial_pdgs, vector<int> final_pdgs, set<int> schannels) 
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
  const int proc_IDS[nprocs] = {0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8, 9,
      10, 11, 11, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18, 19, 19, 20, 20, 21,
      22, 23, 23, 24, 25, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 31,
      31, 32, 32, 32, 32, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 37,
      37, 38, 38, 39, 39, 40, 40, 41, 42, 42, 43, 44, 44, 44, 44, 45, 45, 46,
      46, 47, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 54, 54, 55, 55, 56, 56,
      57, 58, 59, 59, 60, 60, 61, 62, 62, 63, 63, 64, 64, 65, 66, 1, 1, 2, 3,
      3, 4, 4, 5, 6, 6, 7, 8, 9, 10, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18,
      20, 20, 21, 22, 24, 25, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31,
      31, 31, 32, 32, 32, 32, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36,
      37, 37, 38, 38, 39, 39, 40, 40, 41, 42, 42, 43, 44, 44, 44, 44, 45, 45,
      46, 46, 47, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 54, 54, 55, 55, 56,
      56, 57, 58, 59, 59, 60, 60, 61, 62, 62, 63, 63, 64, 64, 65, 66};
  const int in_pdgs[nprocs][ninitial] = {{2, 2}, {4, 4}, {2, 1}, {4, 3}, {2,
      3}, {2, -2}, {4, -4}, {2, -2}, {4, -4}, {2, -2}, {2, -1}, {4, -3}, {2,
      -3}, {4, 1}, {4, -4}, {4, -1}, {1, 1}, {3, 3}, {1, -2}, {3, -4}, {1, -4},
      {1, -1}, {3, -3}, {1, -1}, {1, -1}, {3, -3}, {3, -2}, {3, -3}, {-2, -2},
      {-4, -4}, {-2, -1}, {-4, -3}, {-2, -3}, {-4, -1}, {-1, -1}, {-3, -3}, {2,
      4}, {2, 1}, {2, 3}, {2, 1}, {4, 1}, {2, 1}, {2, 3}, {4, 3}, {2, 3}, {4,
      1}, {2, -2}, {4, -4}, {2, -2}, {4, -2}, {1, -3}, {3, -3}, {2, -2}, {2,
      -4}, {3, -1}, {3, -3}, {2, -4}, {4, -2}, {2, -4}, {4, -4}, {1, -1}, {3,
      -1}, {2, -4}, {4, -2}, {1, -3}, {3, -1}, {2, -4}, {3, -1}, {2, -1}, {4,
      -3}, {2, -1}, {4, -3}, {2, -1}, {4, -3}, {2, -3}, {2, -3}, {2, -3}, {4,
      1}, {4, 3}, {4, 3}, {4, -2}, {4, -4}, {1, -1}, {1, -3}, {4, -2}, {1, -3},
      {4, -1}, {4, -1}, {4, -1}, {1, 3}, {1, -2}, {3, -4}, {1, -2}, {3, -4},
      {1, -2}, {3, -4}, {1, -4}, {1, -4}, {1, -4}, {1, -1}, {3, -3}, {1, -3},
      {3, -1}, {3, -2}, {3, -2}, {3, -2}, {-2, -4}, {-2, -1}, {-2, -3}, {-2,
      -1}, {-4, -1}, {-2, -1}, {-2, -3}, {-4, -3}, {-2, -3}, {-4, -1}, {-4,
      -1}, {-4, -3}, {-4, -3}, {-1, -3}, {1, 2}, {3, 4}, {3, 2}, {-2, 2}, {-4,
      4}, {-2, 2}, {-4, 4}, {-2, 2}, {-1, 2}, {-3, 4}, {-3, 2}, {1, 4}, {-4,
      4}, {-1, 4}, {-2, 1}, {-4, 3}, {-4, 1}, {-1, 1}, {-3, 3}, {-1, 1}, {-1,
      1}, {-3, 3}, {-2, 3}, {-3, 3}, {-1, -2}, {-3, -4}, {-3, -2}, {-1, -4},
      {4, 2}, {1, 2}, {3, 2}, {1, 2}, {1, 4}, {1, 2}, {3, 2}, {3, 4}, {3, 2},
      {1, 4}, {-2, 2}, {-4, 4}, {-2, 2}, {-2, 4}, {-3, 1}, {-3, 3}, {-2, 2},
      {-4, 2}, {-1, 3}, {-3, 3}, {-4, 2}, {-2, 4}, {-4, 2}, {-4, 4}, {-1, 1},
      {-1, 3}, {-4, 2}, {-2, 4}, {-3, 1}, {-1, 3}, {-4, 2}, {-1, 3}, {-1, 2},
      {-3, 4}, {-1, 2}, {-3, 4}, {-1, 2}, {-3, 4}, {-3, 2}, {-3, 2}, {-3, 2},
      {1, 4}, {3, 4}, {3, 4}, {-2, 4}, {-4, 4}, {-1, 1}, {-3, 1}, {-2, 4}, {-3,
      1}, {-1, 4}, {-1, 4}, {-1, 4}, {3, 1}, {-2, 1}, {-4, 3}, {-2, 1}, {-4,
      3}, {-2, 1}, {-4, 3}, {-4, 1}, {-4, 1}, {-4, 1}, {-1, 1}, {-3, 3}, {-3,
      1}, {-1, 3}, {-2, 3}, {-2, 3}, {-2, 3}, {-4, -2}, {-1, -2}, {-3, -2},
      {-1, -2}, {-1, -4}, {-1, -2}, {-3, -2}, {-3, -4}, {-3, -2}, {-1, -4},
      {-1, -4}, {-3, -4}, {-3, -4}, {-3, -1}};
  const int out_pdgs[nprocs][nexternal - ninitial] = {{25, 2, 2}, {25, 4, 4},
      {25, 2, 1}, {25, 4, 3}, {25, 2, 3}, {25, 2, -2}, {25, 4, -4}, {25, 1,
      -1}, {25, 3, -3}, {25, 3, -3}, {25, 2, -1}, {25, 4, -3}, {25, 2, -3},
      {25, 4, 1}, {25, 1, -1}, {25, 4, -1}, {25, 1, 1}, {25, 3, 3}, {25, 1,
      -2}, {25, 3, -4}, {25, 1, -4}, {25, 2, -2}, {25, 4, -4}, {25, 4, -4},
      {25, 1, -1}, {25, 3, -3}, {25, 3, -2}, {25, 2, -2}, {25, -2, -2}, {25,
      -4, -4}, {25, -2, -1}, {25, -4, -3}, {25, -2, -3}, {25, -4, -1}, {25, -1,
      -1}, {25, -3, -3}, {25, 2, 4}, {25, 2, 3}, {25, 4, 3}, {25, 4, 1}, {25,
      4, 3}, {25, 4, 3}, {25, 2, 1}, {25, 2, 3}, {25, 4, 1}, {25, 2, 3}, {25,
      4, -4}, {25, 2, -2}, {25, 1, -3}, {25, 3, -3}, {25, 2, -2}, {25, 4, -2},
      {25, 3, -1}, {25, 3, -3}, {25, 2, -2}, {25, 2, -4}, {25, 2, -4}, {25, 4,
      -2}, {25, 1, -1}, {25, 3, -1}, {25, 2, -4}, {25, 4, -4}, {25, 1, -3},
      {25, 3, -1}, {25, 2, -4}, {25, 4, -2}, {25, 3, -1}, {25, 2, -4}, {25, 2,
      -3}, {25, 2, -3}, {25, 4, -1}, {25, 4, -1}, {25, 4, -3}, {25, 2, -1},
      {25, 2, -1}, {25, 4, -3}, {25, 4, -1}, {25, 2, 1}, {25, 4, 1}, {25, 2,
      1}, {25, 1, -1}, {25, 1, -3}, {25, 4, -2}, {25, 4, -4}, {25, 1, -3}, {25,
      4, -2}, {25, 2, -1}, {25, 4, -3}, {25, 2, -3}, {25, 1, 3}, {25, 1, -4},
      {25, 1, -4}, {25, 3, -2}, {25, 3, -2}, {25, 3, -4}, {25, 1, -2}, {25, 1,
      -2}, {25, 3, -4}, {25, 3, -2}, {25, 3, -3}, {25, 1, -1}, {25, 1, -3},
      {25, 3, -1}, {25, 1, -2}, {25, 3, -4}, {25, 1, -4}, {25, -2, -4}, {25,
      -2, -3}, {25, -4, -3}, {25, -4, -1}, {25, -4, -3}, {25, -4, -3}, {25, -2,
      -1}, {25, -2, -3}, {25, -4, -1}, {25, -2, -3}, {25, -2, -1}, {25, -4,
      -1}, {25, -2, -1}, {25, -1, -3}, {25, 2, 1}, {25, 4, 3}, {25, 2, 3}, {25,
      2, -2}, {25, 4, -4}, {25, 1, -1}, {25, 3, -3}, {25, 3, -3}, {25, 2, -1},
      {25, 4, -3}, {25, 2, -3}, {25, 4, 1}, {25, 1, -1}, {25, 4, -1}, {25, 1,
      -2}, {25, 3, -4}, {25, 1, -4}, {25, 2, -2}, {25, 4, -4}, {25, 4, -4},
      {25, 1, -1}, {25, 3, -3}, {25, 3, -2}, {25, 2, -2}, {25, -2, -1}, {25,
      -4, -3}, {25, -2, -3}, {25, -4, -1}, {25, 2, 4}, {25, 2, 3}, {25, 4, 3},
      {25, 4, 1}, {25, 4, 3}, {25, 4, 3}, {25, 2, 1}, {25, 2, 3}, {25, 4, 1},
      {25, 2, 3}, {25, 4, -4}, {25, 2, -2}, {25, 1, -3}, {25, 3, -3}, {25, 2,
      -2}, {25, 4, -2}, {25, 3, -1}, {25, 3, -3}, {25, 2, -2}, {25, 2, -4},
      {25, 2, -4}, {25, 4, -2}, {25, 1, -1}, {25, 3, -1}, {25, 2, -4}, {25, 4,
      -4}, {25, 1, -3}, {25, 3, -1}, {25, 2, -4}, {25, 4, -2}, {25, 3, -1},
      {25, 2, -4}, {25, 2, -3}, {25, 2, -3}, {25, 4, -1}, {25, 4, -1}, {25, 4,
      -3}, {25, 2, -1}, {25, 2, -1}, {25, 4, -3}, {25, 4, -1}, {25, 2, 1}, {25,
      4, 1}, {25, 2, 1}, {25, 1, -1}, {25, 1, -3}, {25, 4, -2}, {25, 4, -4},
      {25, 1, -3}, {25, 4, -2}, {25, 2, -1}, {25, 4, -3}, {25, 2, -3}, {25, 1,
      3}, {25, 1, -4}, {25, 1, -4}, {25, 3, -2}, {25, 3, -2}, {25, 3, -4}, {25,
      1, -2}, {25, 1, -2}, {25, 3, -4}, {25, 3, -2}, {25, 3, -3}, {25, 1, -1},
      {25, 1, -3}, {25, 3, -1}, {25, 1, -2}, {25, 3, -4}, {25, 1, -4}, {25, -2,
      -4}, {25, -2, -3}, {25, -4, -3}, {25, -4, -1}, {25, -4, -3}, {25, -4,
      -3}, {25, -2, -1}, {25, -2, -3}, {25, -4, -1}, {25, -2, -3}, {25, -2,
      -1}, {25, -4, -1}, {25, -2, -1}, {25, -1, -3}};

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
void PY8MEs_R906_P4_heft_ckm_qq_hqq::setMomenta(vector < vec_double >
    momenta_picked)
{
  if (momenta_picked.size() != nexternal)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setMomenta' of class" << 
    " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Incorrect number of" << 
    " momenta specified." << endl; 
    #endif
    throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
  }
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    if (momenta_picked[i].size() != 4)
    {
      #ifdef DEBUG
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Incorrect number of" << 
      " momenta components specified." << endl; 
      #endif
      throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
    }
    if (momenta_picked[i][0] < 0.0)
    {
      #ifdef DEBUG
      cerr <<  "Error in function 'setMomenta' of class" << 
      " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': A momentum was specified" << 
      " with negative energy. Check conventions." << endl; 
      #endif
      throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
    }
    for (unsigned int j = 0; j < 4; j++ )
    {
      p[i][j] = momenta_picked[i][j]; 
    }
  }
}

//--------------------------------------------------------------------------
// Set color configuration to use. An empty vector means sum over all.
void PY8MEs_R906_P4_heft_ckm_qq_hqq::setColors(vector<int> colors_picked)
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
    " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Incorrect number" << 
    " of colors specified." << endl; 
    #endif
    throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
  }
  user_colors = vector<int> ((2 * nexternal), 0); 
  for(unsigned int i = 0; i < (2 * nexternal); i++ )
  {
    user_colors[i] = colors_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the helicity configuration to use. Am empty vector means sum over all.
void PY8MEs_R906_P4_heft_ckm_qq_hqq::setHelicities(vector<int>
    helicities_picked)
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
    " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Incorrect number" << 
    " of helicities specified." << endl; 
    #endif
    throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
  }
  user_helicities = vector<int> (nexternal, 0); 
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    user_helicities[i] = helicities_picked[i]; 
  }
}

//--------------------------------------------------------------------------
// Set the permutation to use (will apply to momenta, colors and helicities)
void PY8MEs_R906_P4_heft_ckm_qq_hqq::setPermutation(vector<int> perm_picked) 
{
  if (perm_picked.size() != nexternal)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setPermutations' of class" << 
    " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Incorrect number" << 
    " of permutations specified." << endl; 
    #endif
    throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
  }
  for(unsigned int i = 0; i < nexternal; i++ )
  {
    perm[i] = perm_picked[i]; 
  }
}

// Set the proc_ID to use
void PY8MEs_R906_P4_heft_ckm_qq_hqq::setProcID(int procID_picked) 
{
  proc_ID = procID_picked; 
}

//--------------------------------------------------------------------------
// Initialize process.

void PY8MEs_R906_P4_heft_ckm_qq_hqq::initProc() 
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
void PY8MEs_R906_P4_heft_ckm_qq_hqq::syncProcModelParams() 
{

  // Instantiate the model class and set parameters that stay fixed during run
  mME[0] = pars->ZERO; 
  mME[1] = pars->ZERO; 
  mME[2] = pars->mdl_MH; 
  mME[3] = pars->ZERO; 
  mME[4] = pars->ZERO; 
}

//--------------------------------------------------------------------------
// Setter allowing to force particular values for the external masses
void PY8MEs_R906_P4_heft_ckm_qq_hqq::setMasses(vec_double external_masses) 
{

  if (external_masses.size() != mME.size())
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setMasses' of class" << 
    " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Incorrect number of" << 
    " masses specified." << endl; 
    #endif
    throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
  }
  for (unsigned int j = 0; j < mME.size(); j++ )
  {
    mME[j] = external_masses[perm[j]]; 
  }
}

//--------------------------------------------------------------------------
// Getter accessing external masses with the correct ordering
vector<double> PY8MEs_R906_P4_heft_ckm_qq_hqq::getMasses() 
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
void PY8MEs_R906_P4_heft_ckm_qq_hqq::setExternalMassesMode(int mode) 
{
  if (mode != 0 && mode != 1 && mode != 2)
  {
    #ifdef DEBUG
    cerr <<  "Error in function 'setExternalMassesMode' of class" << 
    " 'PY8MEs_R906_P4_heft_ckm_qq_hqq': Incorrect mode selected :" << mode << 
    ". It must be either 0, 1 or 2" << endl; 
    #endif
    throw PY8MEs_R906_P4_heft_ckm_qq_hqq_exception; 
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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::sigmaKin() 
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
      t = matrix_906_uu_huu_no_b(); 
    if (proc_ID == 1)
      t = matrix_906_ud_hud_no_b(); 
    if (proc_ID == 2)
      t = matrix_906_us_hus_no_b(); 
    if (proc_ID == 3)
      t = matrix_906_uux_huux_no_b(); 
    if (proc_ID == 4)
      t = matrix_906_uux_hddx_no_b(); 
    if (proc_ID == 5)
      t = matrix_906_uux_hssx_no_b(); 
    if (proc_ID == 6)
      t = matrix_906_udx_hudx_no_b(); 
    if (proc_ID == 7)
      t = matrix_906_usx_husx_no_b(); 
    if (proc_ID == 8)
      t = matrix_906_cd_hcd_no_b(); 
    if (proc_ID == 9)
      t = matrix_906_ccx_hddx_no_b(); 
    if (proc_ID == 10)
      t = matrix_906_cdx_hcdx_no_b(); 
    if (proc_ID == 11)
      t = matrix_906_dd_hdd_no_b(); 
    if (proc_ID == 12)
      t = matrix_906_dux_hdux_no_b(); 
    if (proc_ID == 13)
      t = matrix_906_dcx_hdcx_no_b(); 
    if (proc_ID == 14)
      t = matrix_906_ddx_huux_no_b(); 
    if (proc_ID == 15)
      t = matrix_906_ddx_hccx_no_b(); 
    if (proc_ID == 16)
      t = matrix_906_ddx_hddx_no_b(); 
    if (proc_ID == 17)
      t = matrix_906_sux_hsux_no_b(); 
    if (proc_ID == 18)
      t = matrix_906_ssx_huux_no_b(); 
    if (proc_ID == 19)
      t = matrix_906_uxux_huxux_no_b(); 
    if (proc_ID == 20)
      t = matrix_906_uxdx_huxdx_no_b(); 
    if (proc_ID == 21)
      t = matrix_906_uxsx_huxsx_no_b(); 
    if (proc_ID == 22)
      t = matrix_906_cxdx_hcxdx_no_b(); 
    if (proc_ID == 23)
      t = matrix_906_dxdx_hdxdx_no_b(); 
    if (proc_ID == 24)
      t = matrix_906_uc_huc_no_b(); 
    if (proc_ID == 25)
      t = matrix_906_ud_hus_no_b(); 
    if (proc_ID == 26)
      t = matrix_906_ud_hcd_no_b(); 
    if (proc_ID == 27)
      t = matrix_906_ud_hcs_no_b(); 
    if (proc_ID == 28)
      t = matrix_906_us_hud_no_b(); 
    if (proc_ID == 29)
      t = matrix_906_us_hcd_no_b(); 
    if (proc_ID == 30)
      t = matrix_906_uux_hccx_no_b(); 
    if (proc_ID == 31)
      t = matrix_906_uux_hdsx_no_b(); 
    if (proc_ID == 32)
      t = matrix_906_uux_hsdx_no_b(); 
    if (proc_ID == 33)
      t = matrix_906_ucx_hucx_no_b(); 
    if (proc_ID == 34)
      t = matrix_906_ucx_hddx_no_b(); 
    if (proc_ID == 35)
      t = matrix_906_ucx_hdsx_no_b(); 
    if (proc_ID == 36)
      t = matrix_906_ucx_hsdx_no_b(); 
    if (proc_ID == 37)
      t = matrix_906_udx_husx_no_b(); 
    if (proc_ID == 38)
      t = matrix_906_udx_hcdx_no_b(); 
    if (proc_ID == 39)
      t = matrix_906_udx_hcsx_no_b(); 
    if (proc_ID == 40)
      t = matrix_906_usx_hudx_no_b(); 
    if (proc_ID == 41)
      t = matrix_906_usx_hcdx_no_b(); 
    if (proc_ID == 42)
      t = matrix_906_cd_hud_no_b(); 
    if (proc_ID == 43)
      t = matrix_906_cs_hud_no_b(); 
    if (proc_ID == 44)
      t = matrix_906_cux_hddx_no_b(); 
    if (proc_ID == 45)
      t = matrix_906_cux_hdsx_no_b(); 
    if (proc_ID == 46)
      t = matrix_906_cdx_hudx_no_b(); 
    if (proc_ID == 47)
      t = matrix_906_cdx_husx_no_b(); 
    if (proc_ID == 48)
      t = matrix_906_ds_hds_no_b(); 
    if (proc_ID == 49)
      t = matrix_906_dux_hdcx_no_b(); 
    if (proc_ID == 50)
      t = matrix_906_dux_hsux_no_b(); 
    if (proc_ID == 51)
      t = matrix_906_dux_hscx_no_b(); 
    if (proc_ID == 52)
      t = matrix_906_dcx_hdux_no_b(); 
    if (proc_ID == 53)
      t = matrix_906_dcx_hsux_no_b(); 
    if (proc_ID == 54)
      t = matrix_906_ddx_hssx_no_b(); 
    if (proc_ID == 55)
      t = matrix_906_dsx_hdsx_no_b(); 
    if (proc_ID == 56)
      t = matrix_906_sux_hdux_no_b(); 
    if (proc_ID == 57)
      t = matrix_906_sux_hdcx_no_b(); 
    if (proc_ID == 58)
      t = matrix_906_uxcx_huxcx_no_b(); 
    if (proc_ID == 59)
      t = matrix_906_uxdx_huxsx_no_b(); 
    if (proc_ID == 60)
      t = matrix_906_uxdx_hcxdx_no_b(); 
    if (proc_ID == 61)
      t = matrix_906_uxdx_hcxsx_no_b(); 
    if (proc_ID == 62)
      t = matrix_906_uxsx_huxdx_no_b(); 
    if (proc_ID == 63)
      t = matrix_906_uxsx_hcxdx_no_b(); 
    if (proc_ID == 64)
      t = matrix_906_cxdx_huxdx_no_b(); 
    if (proc_ID == 65)
      t = matrix_906_cxsx_huxdx_no_b(); 
    if (proc_ID == 66)
      t = matrix_906_dxsx_hdxsx_no_b(); 

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

void PY8MEs_R906_P4_heft_ckm_qq_hqq::calculate_wavefunctions(const int hel[])
{
  // Calculate wavefunctions for all processes
  // Calculate all wavefunctions
  ixxxxx(p[perm[0]], mME[0], hel[0], +1, w[0]); 
  ixxxxx(p[perm[1]], mME[1], hel[1], +1, w[1]); 
  sxxxxx(p[perm[2]], +1, w[2]); 
  oxxxxx(p[perm[3]], mME[3], hel[3], +1, w[3]); 
  oxxxxx(p[perm[4]], mME[4], hel[4], +1, w[4]); 
  FFV2_5_3(w[0], w[3], pars->GC_41, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[5]);
  FFV2_5_3(w[1], w[4], pars->GC_41, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[6]);
  FFV2_5_3(w[0], w[4], pars->GC_41, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[7]);
  FFV2_5_3(w[1], w[3], pars->GC_41, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[8]);
  FFV2_3_3(w[1], w[4], pars->GC_40, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[9]);
  FFV2_3(w[0], w[4], pars->GC_31, pars->mdl_MW, pars->mdl_WW, w[10]); 
  FFV2_3(w[1], w[3], pars->GC_31, pars->mdl_MW, pars->mdl_WW, w[11]); 
  FFV2_3(w[0], w[4], pars->GC_32, pars->mdl_MW, pars->mdl_WW, w[12]); 
  FFV2_3(w[1], w[3], pars->GC_32, pars->mdl_MW, pars->mdl_WW, w[13]); 
  oxxxxx(p[perm[1]], mME[1], hel[1], -1, w[14]); 
  ixxxxx(p[perm[4]], mME[4], hel[4], -1, w[15]); 
  FFV2_5_3(w[0], w[14], pars->GC_41, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[16]);
  FFV2_5_3(w[15], w[3], pars->GC_41, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[17]);
  FFV2_5_3(w[15], w[14], pars->GC_41, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[18]);
  FFV2_3_3(w[15], w[3], pars->GC_40, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[19]);
  FFV2_3(w[0], w[3], pars->GC_31, pars->mdl_MW, pars->mdl_WW, w[20]); 
  FFV2_3(w[15], w[14], pars->GC_31, pars->mdl_MW, pars->mdl_WW, w[21]); 
  FFV2_3(w[0], w[3], pars->GC_32, pars->mdl_MW, pars->mdl_WW, w[22]); 
  FFV2_3(w[15], w[14], pars->GC_32, pars->mdl_MW, pars->mdl_WW, w[23]); 
  FFV2_3_3(w[15], w[14], pars->GC_40, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[24]);
  FFV2_3(w[0], w[14], pars->GC_31, pars->mdl_MW, pars->mdl_WW, w[25]); 
  FFV2_3(w[15], w[3], pars->GC_31, pars->mdl_MW, pars->mdl_WW, w[26]); 
  FFV2_3(w[0], w[14], pars->GC_32, pars->mdl_MW, pars->mdl_WW, w[27]); 
  FFV2_3(w[15], w[3], pars->GC_32, pars->mdl_MW, pars->mdl_WW, w[28]); 
  FFV2_3(w[0], w[4], pars->GC_34, pars->mdl_MW, pars->mdl_WW, w[29]); 
  FFV2_3(w[1], w[3], pars->GC_34, pars->mdl_MW, pars->mdl_WW, w[30]); 
  FFV2_3(w[0], w[3], pars->GC_34, pars->mdl_MW, pars->mdl_WW, w[31]); 
  FFV2_3(w[15], w[14], pars->GC_34, pars->mdl_MW, pars->mdl_WW, w[32]); 
  FFV2_3(w[0], w[14], pars->GC_34, pars->mdl_MW, pars->mdl_WW, w[33]); 
  FFV2_3(w[15], w[3], pars->GC_34, pars->mdl_MW, pars->mdl_WW, w[34]); 
  FFV2_3_3(w[0], w[3], pars->GC_40, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[35]);
  FFV2_3_3(w[0], w[4], pars->GC_40, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[36]);
  FFV2_3_3(w[1], w[3], pars->GC_40, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[37]);
  FFV2_3_3(w[0], w[14], pars->GC_40, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[38]);
  oxxxxx(p[perm[0]], mME[0], hel[0], -1, w[39]); 
  ixxxxx(p[perm[3]], mME[3], hel[3], -1, w[40]); 
  FFV2_5_3(w[40], w[39], pars->GC_41, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[41]);
  FFV2_5_3(w[40], w[14], pars->GC_41, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[42]);
  FFV2_5_3(w[15], w[39], pars->GC_41, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[43]);
  FFV2_3(w[40], w[14], pars->GC_31, pars->mdl_MW, pars->mdl_WW, w[44]); 
  FFV2_3(w[15], w[39], pars->GC_31, pars->mdl_MW, pars->mdl_WW, w[45]); 
  FFV2_3(w[40], w[14], pars->GC_32, pars->mdl_MW, pars->mdl_WW, w[46]); 
  FFV2_3(w[15], w[39], pars->GC_32, pars->mdl_MW, pars->mdl_WW, w[47]); 
  FFV2_3(w[40], w[14], pars->GC_34, pars->mdl_MW, pars->mdl_WW, w[48]); 
  FFV2_3(w[15], w[39], pars->GC_34, pars->mdl_MW, pars->mdl_WW, w[49]); 
  FFV2_3_3(w[40], w[39], pars->GC_40, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[50]);
  FFV2_3_3(w[40], w[14], pars->GC_40, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[51]);
  FFV2_3_3(w[15], w[39], pars->GC_40, pars->GC_53, pars->mdl_MZ, pars->mdl_WZ,
      w[52]);

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  VVS2_0(w[5], w[6], w[2], pars->GC_73, amp[0]); 
  VVS2_0(w[7], w[8], w[2], pars->GC_73, amp[1]); 
  VVS2_0(w[5], w[9], w[2], pars->GC_73, amp[2]); 
  VVS2_0(w[10], w[11], w[2], pars->GC_70, amp[3]); 
  VVS2_0(w[5], w[9], w[2], pars->GC_73, amp[4]); 
  VVS2_0(w[12], w[13], w[2], pars->GC_70, amp[5]); 
  VVS2_0(w[16], w[17], w[2], pars->GC_73, amp[6]); 
  VVS2_0(w[5], w[18], w[2], pars->GC_73, amp[7]); 
  VVS2_0(w[16], w[19], w[2], pars->GC_73, amp[8]); 
  VVS2_0(w[20], w[21], w[2], pars->GC_70, amp[9]); 
  VVS2_0(w[16], w[19], w[2], pars->GC_73, amp[10]); 
  VVS2_0(w[22], w[23], w[2], pars->GC_70, amp[11]); 
  VVS2_0(w[5], w[24], w[2], pars->GC_73, amp[12]); 
  VVS2_0(w[25], w[26], w[2], pars->GC_70, amp[13]); 
  VVS2_0(w[5], w[24], w[2], pars->GC_73, amp[14]); 
  VVS2_0(w[27], w[28], w[2], pars->GC_70, amp[15]); 
  VVS2_0(w[5], w[9], w[2], pars->GC_73, amp[16]); 
  VVS2_0(w[29], w[30], w[2], pars->GC_70, amp[17]); 
  VVS2_0(w[16], w[19], w[2], pars->GC_73, amp[18]); 
  VVS2_0(w[31], w[32], w[2], pars->GC_70, amp[19]); 
  VVS2_0(w[5], w[24], w[2], pars->GC_73, amp[20]); 
  VVS2_0(w[33], w[34], w[2], pars->GC_70, amp[21]); 
  VVS2_0(w[35], w[9], w[2], pars->GC_73, amp[22]); 
  VVS2_0(w[36], w[37], w[2], pars->GC_73, amp[23]); 
  VVS2_0(w[18], w[35], w[2], pars->GC_73, amp[24]); 
  VVS2_0(w[26], w[25], w[2], pars->GC_70, amp[25]); 
  VVS2_0(w[18], w[35], w[2], pars->GC_73, amp[26]); 
  VVS2_0(w[34], w[33], w[2], pars->GC_70, amp[27]); 
  VVS2_0(w[17], w[38], w[2], pars->GC_73, amp[28]); 
  VVS2_0(w[21], w[20], w[2], pars->GC_70, amp[29]); 
  VVS2_0(w[17], w[38], w[2], pars->GC_73, amp[30]); 
  VVS2_0(w[32], w[31], w[2], pars->GC_70, amp[31]); 
  VVS2_0(w[38], w[19], w[2], pars->GC_73, amp[32]); 
  VVS2_0(w[35], w[24], w[2], pars->GC_73, amp[33]); 
  VVS2_0(w[18], w[35], w[2], pars->GC_73, amp[34]); 
  VVS2_0(w[28], w[27], w[2], pars->GC_70, amp[35]); 
  VVS2_0(w[17], w[38], w[2], pars->GC_73, amp[36]); 
  VVS2_0(w[23], w[22], w[2], pars->GC_70, amp[37]); 
  VVS2_0(w[41], w[18], w[2], pars->GC_73, amp[38]); 
  VVS2_0(w[42], w[43], w[2], pars->GC_73, amp[39]); 
  VVS2_0(w[41], w[24], w[2], pars->GC_73, amp[40]); 
  VVS2_0(w[44], w[45], w[2], pars->GC_70, amp[41]); 
  VVS2_0(w[41], w[24], w[2], pars->GC_73, amp[42]); 
  VVS2_0(w[46], w[47], w[2], pars->GC_70, amp[43]); 
  VVS2_0(w[41], w[24], w[2], pars->GC_73, amp[44]); 
  VVS2_0(w[48], w[49], w[2], pars->GC_70, amp[45]); 
  VVS2_0(w[50], w[24], w[2], pars->GC_73, amp[46]); 
  VVS2_0(w[51], w[52], w[2], pars->GC_73, amp[47]); 
  VVS2_0(w[12], w[11], w[2], pars->GC_70, amp[48]); 
  VVS2_0(w[10], w[30], w[2], pars->GC_70, amp[49]); 
  VVS2_0(w[12], w[30], w[2], pars->GC_70, amp[50]); 
  VVS2_0(w[10], w[13], w[2], pars->GC_70, amp[51]); 
  VVS2_0(w[10], w[11], w[2], pars->GC_70, amp[52]); 
  VVS2_0(w[16], w[17], w[2], pars->GC_73, amp[53]); 
  VVS2_0(w[20], w[23], w[2], pars->GC_70, amp[54]); 
  VVS2_0(w[22], w[21], w[2], pars->GC_70, amp[55]); 
  VVS2_0(w[5], w[18], w[2], pars->GC_73, amp[56]); 
  VVS2_0(w[20], w[32], w[2], pars->GC_70, amp[57]); 
  VVS2_0(w[20], w[21], w[2], pars->GC_70, amp[58]); 
  VVS2_0(w[22], w[32], w[2], pars->GC_70, amp[59]); 
  VVS2_0(w[25], w[28], w[2], pars->GC_70, amp[60]); 
  VVS2_0(w[25], w[34], w[2], pars->GC_70, amp[61]); 
  VVS2_0(w[25], w[26], w[2], pars->GC_70, amp[62]); 
  VVS2_0(w[27], w[26], w[2], pars->GC_70, amp[63]); 
  VVS2_0(w[27], w[34], w[2], pars->GC_70, amp[64]); 
  VVS2_0(w[29], w[11], w[2], pars->GC_70, amp[65]); 
  VVS2_0(w[29], w[13], w[2], pars->GC_70, amp[66]); 
  VVS2_0(w[31], w[21], w[2], pars->GC_70, amp[67]); 
  VVS2_0(w[31], w[23], w[2], pars->GC_70, amp[68]); 
  VVS2_0(w[33], w[26], w[2], pars->GC_70, amp[69]); 
  VVS2_0(w[33], w[28], w[2], pars->GC_70, amp[70]); 
  VVS2_0(w[35], w[9], w[2], pars->GC_73, amp[71]); 
  VVS2_0(w[34], w[25], w[2], pars->GC_70, amp[72]); 
  VVS2_0(w[28], w[25], w[2], pars->GC_70, amp[73]); 
  VVS2_0(w[26], w[25], w[2], pars->GC_70, amp[74]); 
  VVS2_0(w[26], w[33], w[2], pars->GC_70, amp[75]); 
  VVS2_0(w[28], w[33], w[2], pars->GC_70, amp[76]); 
  VVS2_0(w[38], w[19], w[2], pars->GC_73, amp[77]); 
  VVS2_0(w[35], w[24], w[2], pars->GC_73, amp[78]); 
  VVS2_0(w[26], w[27], w[2], pars->GC_70, amp[79]); 
  VVS2_0(w[34], w[27], w[2], pars->GC_70, amp[80]); 
  VVS2_0(w[41], w[18], w[2], pars->GC_73, amp[81]); 
  VVS2_0(w[44], w[47], w[2], pars->GC_70, amp[82]); 
  VVS2_0(w[48], w[45], w[2], pars->GC_70, amp[83]); 
  VVS2_0(w[48], w[47], w[2], pars->GC_70, amp[84]); 
  VVS2_0(w[46], w[45], w[2], pars->GC_70, amp[85]); 
  VVS2_0(w[44], w[45], w[2], pars->GC_70, amp[86]); 
  VVS2_0(w[44], w[49], w[2], pars->GC_70, amp[87]); 
  VVS2_0(w[46], w[49], w[2], pars->GC_70, amp[88]); 
  VVS2_0(w[50], w[24], w[2], pars->GC_73, amp[89]); 


}
double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uu_huu_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[0]; 
  jamp[1] = +amp[1]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ud_hud_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[2]; 
  jamp[1] = +amp[3]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_us_hus_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[4]; 
  jamp[1] = +amp[5]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uux_huux_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[6]; 
  jamp[1] = +amp[7]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uux_hddx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[8]; 
  jamp[1] = +amp[9]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uux_hssx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[10]; 
  jamp[1] = +amp[11]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_udx_hudx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[13]; 
  jamp[1] = +amp[12]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_usx_husx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[15]; 
  jamp[1] = +amp[14]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_cd_hcd_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[16]; 
  jamp[1] = +amp[17]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ccx_hddx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[18]; 
  jamp[1] = +amp[19]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_cdx_hcdx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[21]; 
  jamp[1] = +amp[20]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_dd_hdd_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[22]; 
  jamp[1] = +amp[23]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_dux_hdux_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[25]; 
  jamp[1] = +amp[24]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_dcx_hdcx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[27]; 
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
    jamp2[13][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ddx_huux_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[28]; 
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
    jamp2[14][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ddx_hccx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[30]; 
  jamp[1] = +amp[31]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ddx_hddx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[32]; 
  jamp[1] = +amp[33]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_sux_hsux_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[35]; 
  jamp[1] = +amp[34]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ssx_huux_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[36]; 
  jamp[1] = +amp[37]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uxux_huxux_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[38]; 
  jamp[1] = +amp[39]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uxdx_huxdx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[40]; 
  jamp[1] = +amp[41]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uxsx_huxsx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[42]; 
  jamp[1] = +amp[43]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_cxdx_hcxdx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[44]; 
  jamp[1] = +amp[45]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_dxdx_hdxdx_no_b() 
{
  int i, j; 
  // Local variables
  // const int ngraphs = 2;
  const int ncolor = 2; 
  Complex<double> ztemp; 
  Complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1}; 
  static const double cf[ncolor][ncolor] = {{9, 3}, {3, 9}}; 

  // Calculate color flows
  jamp[0] = -amp[46]; 
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
    jamp2[23][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uc_huc_no_b() 
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
    jamp2[24][i] += real(jamp[i] * conj(jamp[i])) * (cf[i][i]/denom[i]); 

  return matrix; 
}

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ud_hus_no_b() 
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
  jamp[0] = +amp[48]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ud_hcd_no_b() 
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
  jamp[0] = +amp[49]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ud_hcs_no_b() 
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
  jamp[0] = +amp[50]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_us_hud_no_b() 
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
  jamp[0] = +amp[51]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_us_hcd_no_b() 
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
  jamp[0] = +amp[52]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uux_hccx_no_b() 
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
  jamp[0] = -amp[53]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uux_hdsx_no_b() 
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
  jamp[0] = +amp[54]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uux_hsdx_no_b() 
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
  jamp[0] = +amp[55]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ucx_hucx_no_b() 
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
  jamp[0] = +amp[56]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ucx_hddx_no_b() 
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
  jamp[0] = +amp[57]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ucx_hdsx_no_b() 
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
  jamp[0] = +amp[58]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ucx_hsdx_no_b() 
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
  jamp[0] = +amp[59]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_udx_husx_no_b() 
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
  jamp[0] = -amp[60]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_udx_hcdx_no_b() 
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
  jamp[0] = -amp[61]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_udx_hcsx_no_b() 
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
  jamp[0] = -amp[62]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_usx_hudx_no_b() 
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
  jamp[0] = -amp[63]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_usx_hcdx_no_b() 
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
  jamp[0] = -amp[64]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_cd_hud_no_b() 
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
  jamp[0] = +amp[65]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_cs_hud_no_b() 
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
  jamp[0] = +amp[66]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_cux_hddx_no_b() 
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
  jamp[0] = +amp[67]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_cux_hdsx_no_b() 
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
  jamp[0] = +amp[68]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_cdx_hudx_no_b() 
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
  jamp[0] = -amp[69]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_cdx_husx_no_b() 
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
  jamp[0] = -amp[70]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ds_hds_no_b() 
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
  jamp[0] = -amp[71]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_dux_hdcx_no_b() 
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
  jamp[0] = -amp[72]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_dux_hsux_no_b() 
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
  jamp[0] = -amp[73]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_dux_hscx_no_b() 
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
  jamp[0] = -amp[74]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_dcx_hdux_no_b() 
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
  jamp[0] = -amp[75]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_dcx_hsux_no_b() 
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
  jamp[0] = -amp[76]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_ddx_hssx_no_b() 
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
  jamp[0] = -amp[77]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_dsx_hdsx_no_b() 
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
  jamp[0] = +amp[78]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_sux_hdux_no_b() 
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
  jamp[0] = -amp[79]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_sux_hdcx_no_b() 
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
  jamp[0] = -amp[80]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uxcx_huxcx_no_b() 
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
  jamp[0] = -amp[81]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uxdx_huxsx_no_b() 
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
  jamp[0] = +amp[82]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uxdx_hcxdx_no_b() 
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
  jamp[0] = +amp[83]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uxdx_hcxsx_no_b() 
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
  jamp[0] = +amp[84]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uxsx_huxdx_no_b() 
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
  jamp[0] = +amp[85]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_uxsx_hcxdx_no_b() 
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
  jamp[0] = +amp[86]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_cxdx_huxdx_no_b() 
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
  jamp[0] = +amp[87]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_cxsx_huxdx_no_b() 
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
  jamp[0] = +amp[88]; 

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

double PY8MEs_R906_P4_heft_ckm_qq_hqq::matrix_906_dxsx_hdxsx_no_b() 
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
  jamp[0] = -amp[89]; 

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

