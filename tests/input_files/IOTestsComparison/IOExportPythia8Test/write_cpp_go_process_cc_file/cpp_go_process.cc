//==========================================================================
// This file has been automatically generated for C++ Standalone by
// MadGraph5_aMC@NLO v. %(version)s, %(date)s
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "CPPProcess.h"
#include "HelAmps_sm.h"

using namespace MG5_sm; 

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: u u~ > go go

//--------------------------------------------------------------------------
// Initialize process.

void CPPProcess::initProc(string param_card_name) 
{
  // Instantiate the model class and set parameters that stay fixed during run
  pars = Parameters_sm::getInstance(); 
  SLHAReader slha(param_card_name); 
  pars->setIndependentParameters(slha); 
  pars->setIndependentCouplings(); 
  pars->printIndependentParameters(); 
  pars->printIndependentCouplings(); 
  // Set external particle masses for this matrix element
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->MGO); 
  mME.push_back(pars->MGO); 
  jamp2[0] = new double[2]; 
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour.

void CPPProcess::sigmaKin(int * flavor) 
{
  // Set the parameters which change event by event
  pars->setDependentParameters(); 
  pars->setDependentCouplings(); 
  static bool firsttime = true; 
  if (firsttime)
  {
    pars->printDependentParameters(); 
    pars->printDependentCouplings(); 
    firsttime = false; 
  }

  // Reset color flows
  for(int i = 0; i < 2; i++ )
    jamp2[0][i] = 0.; 

  // Local variables and constants
  const int ncomb = 16; 
  const int nflav = 1; 
  static const int sk_flav_table[nflav][4] = {{0, 0, 0, 0}}; 
  static bool goodhel[nflav][ncomb] = {}; 
  static int ntry[nflav] = {}, sum_hel[nflav] = {}, ngood[nflav] = {}; 
  static int igood[nflav][ncomb]; 
  static int jhel[nflav]; 
  std::complex<double> * * wfs; 
  double t[nprocesses]; 
  // Helicities for the process
  static const int helicities[ncomb][nexternal] = {{-1, -1, -1, -1}, {-1, -1,
      -1, 1}, {-1, -1, 1, -1}, {-1, -1, 1, 1}, {-1, 1, -1, -1}, {-1, 1, -1, 1},
      {-1, 1, 1, -1}, {-1, 1, 1, 1}, {1, -1, -1, -1}, {1, -1, -1, 1}, {1, -1,
      1, -1}, {1, -1, 1, 1}, {1, 1, -1, -1}, {1, 1, -1, 1}, {1, 1, 1, -1}, {1,
      1, 1, 1}};
  // Denominators: spins, colors and identical particles
  const int denominators[nprocesses] = {72, 72}; 

  int flav_idx = -1; 
  for (int fi = 0; fi < nflav; ++ fi)
  {
    bool fmatch = true; 
    for (int fj = 0; fj < 4; ++ fj)
    {
      if (flavor[fj] != sk_flav_table[fi][fj])
      {
        fmatch = false; break; 
      }
    }
    if (fmatch)
    {
      flav_idx = fi; break; 
    }
  }
  if (flav_idx < 0)
  {
    for (int i = 0; i < nprocesses; i++ )
      matrix_element[i] = 0.; 
    return; 
  }
  ntry[flav_idx] = ntry[flav_idx] + 1; 

  // Reset the matrix elements
  for(int i = 0; i < nprocesses; i++ )
  {
    matrix_element[i] = 0.; 
  }
  // Define permutation
  int perm[nexternal]; 
  for(int i = 0; i < nexternal; i++ )
  {
    perm[i] = i; 
  }

  if (sum_hel[flav_idx] == 0 || ntry[flav_idx] < 10)
  {
    // Calculate the matrix element for all helicities
    for(int ihel = 0; ihel < ncomb; ihel++ )
    {
      if (goodhel[flav_idx][ihel] || ntry[flav_idx] < 2)
      {
        calculate_wavefunctions(perm, helicities[ihel], flavor); 
        t[0] = matrix_uux_gogo(); 
        // Mirror initial state momenta for mirror process
        perm[0] = 1; 
        perm[1] = 0; 
        int flv_tmp = flavor[0]; 
        flavor[0] = flavor[1]; 
        flavor[1] = flv_tmp; 
        // Calculate wavefunctions
        calculate_wavefunctions(perm, helicities[ihel], flavor); 
        // Mirror back
        perm[0] = 0; 
        perm[1] = 1; 
        flavor[1] = flavor[0]; 
        flavor[0] = flv_tmp; 
        // Calculate matrix elements
        t[1] = matrix_uux_gogo(); 
        double tsum = 0; 
        for(int iproc = 0; iproc < nprocesses; iproc++ )
        {
          matrix_element[iproc] += t[iproc]; 
          tsum += t[iproc]; 
        }
        // Store which helicities give non-zero result
        if (tsum != 0. && !goodhel[flav_idx][ihel])
        {
          goodhel[flav_idx][ihel] = true; 
          ngood[flav_idx]++; 
          igood[flav_idx][ngood[flav_idx]] = ihel; 
        }
      }
    }
    jhel[flav_idx] = 0; 
    sum_hel[flav_idx] = min(sum_hel[flav_idx], ngood[flav_idx]); 
  }
  else
  {
    // Only use the "good" helicities
    for(int j = 0; j < sum_hel[flav_idx]; j++ )
    {
      jhel[flav_idx]++; 
      if (jhel[flav_idx] >= ngood[flav_idx])
        jhel[flav_idx] = 0; 
      double hwgt = double(ngood[flav_idx])/double(sum_hel[flav_idx]); 
      int ihel = igood[flav_idx][jhel[flav_idx]]; 
      calculate_wavefunctions(perm, helicities[ihel], flavor); 
      t[0] = matrix_uux_gogo(); 
      // Mirror initial state momenta for mirror process
      perm[0] = 1; 
      perm[1] = 0; 
      int flv_tmp = flavor[0]; 
      flavor[0] = flavor[1]; 
      flavor[1] = flv_tmp; 
      // Calculate wavefunctions
      calculate_wavefunctions(perm, helicities[ihel], flavor); 
      // Mirror back
      perm[0] = 0; 
      perm[1] = 1; 
      flavor[1] = flavor[0]; 
      flavor[0] = flv_tmp; 
      // Calculate matrix elements
      t[1] = matrix_uux_gogo(); 
      for(int iproc = 0; iproc < nprocesses; iproc++ )
      {
        matrix_element[iproc] += t[iproc] * hwgt; 
      }
    }
  }

  for (int i = 0; i < nprocesses; i++ )
    matrix_element[i] = matrix_element[i] * broken_sym(flavor)/denominators[i]; 



}

//--------------------------------------------------------------------------
// Evaluate |M|^2, including incoming flavour dependence.

double CPPProcess::sigmaHat() 
{
  // Select between the different processes
  if(id1 == -2 && id2 == 2)
  {
    // Add matrix elements for processes with beams (-2, 2)
    return matrix_element[1]; 
  }
  else if(id1 == 2 && id2 == -2)
  {
    // Add matrix elements for processes with beams (2, -2)
    return matrix_element[0]; 
  }
  else
  {
    // Return 0 if not correct initial state assignment
    return 0.; 
  }
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

void CPPProcess::calculate_wavefunctions(const int perm[], const int hel[],
    const int flavor[])
{
  // Calculate wavefunctions for all processes
  int i, j; 


  double BWCUTOFF = 15; 
  // Calculate all wavefunctions

  ixxxxx(p[perm[0]], mME[0], hel[0], +1, flavor[0], w[0]); 
  oxxxxx(p[perm[1]], mME[1], hel[1], -1, flavor[1], w[1]); 
  ixxxxx(p[perm[2]], mME[2], hel[2], -1, flavor[2], w[2]); 
  oxxxxx(p[perm[3]], mME[3], hel[3], +1, flavor[3], w[3]); 
  FFV1_3(w[0], w[1], pars->GC_10, pars->ZERO, pars->ZERO, w[4]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[2], w[3], w[4], pars->GC_8, amp[0]); 

}
double CPPProcess::matrix_uux_gogo() 
{

  // Local variables
  const int ngraphs = 1; 
  const int ncolor = 2; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const int denom = 3; 
  static const int cf[ncolor * (ncolor + 1)/2] = {16, -4, 16}; 

  // Calculate color flows
  jamp[0] = -std::complex<double> (0, 1) * amp[0]; 
  jamp[1] = +std::complex<double> (0, 1) * amp[0]; 

  // Sum and square the color flows to get the matrix element
  double matrix = 0; 
  int cf_index = 0; 
  for(int i = 0; i < ncolor; i++ )
  {
    ztemp = 0.; 
    for(int j = i; j < ncolor; j++ , cf_index++ )
    {
      ztemp = ztemp + static_cast<double> (cf[cf_index]) * jamp[j]; 
    }
    matrix = matrix + real(ztemp * conj(jamp[i])); 
  }
  matrix = matrix/denom; 

  // Store the leading color flows for choice of color
  for(int i = 0; i < ncolor; i++ )
    jamp2[0][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}


//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

int CPPProcess::broken_sym(int * flavor)
{
  const int n_components = 1; 
  const int n_entries = 2; 
  const int comp_beg[n_components] = {1}; 
  const int comp_end[n_components] = {2}; 
  const int comp_old[n_components] = {2}; 
  const int pid_list[n_entries] = {1000021, 1000021}; 
  const int block_start[n_entries] = {3, 4}; 
  const int block_len[n_entries] = {1, 1}; 
  int pid_work[n_entries]; 
  for (int i = 0; i < n_entries; i++ )
    pid_work[i] = pid_list[i]; 

  int total_factor = 1; 
  for (int icomp = 0; icomp < n_components; icomp++ )
  {
    int old_factor = comp_old[icomp]; 
    if (comp_old[icomp] > 1)
    {
      for (int i = comp_beg[icomp] - 1; i < comp_end[icomp]; i++ )
      {
        if (pid_work[i] == 0)
          continue; 
        int n_tot = 1; 
        for (int j = i + 1; j < comp_end[icomp]; j++ )
        {
          if (pid_work[i] != pid_work[j])
            continue; 
          bool same_block = (block_len[i] == block_len[j]); 
          for (int k = 0; same_block && k < block_len[i]; k++ )
          {
            if (flavor[block_start[i] - 1 + k] != flavor[block_start[j] - 1 +
                k])
              same_block = false; 
          }
          if (same_block)
          {
            pid_work[j] = 0; 
            n_tot = n_tot + 1; 
            old_factor = old_factor/n_tot; 
          }
        }
      }
    }
    total_factor = total_factor * old_factor; 
  }
  return total_factor; 
}


