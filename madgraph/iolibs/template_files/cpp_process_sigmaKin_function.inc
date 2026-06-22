// Local variables and constants
const int ncomb = %(ncomb)d;
%(cpp_goodhel_decl)s
std::complex<double> **wfs;
double t[nprocesses];
// Helicities for the process
%(helicity_matrix)s
// Denominators: spins, colors and identical particles
const int denominators[nprocesses] = {%(den_factors)s};

%(cpp_flav_idx_compute)sntry[flav_idx]=ntry[flav_idx]+1;

// Reset the matrix elements
for(int i = 0; i < nprocesses; i++){
    matrix_element[i] = 0.;
}
// Define permutation
int perm[nexternal];
for(int i = 0; i < nexternal; i++){
  perm[i]=i;
}

if (sum_hel[flav_idx] == 0 || ntry[flav_idx] < 10){
// Calculate the matrix element for all helicities
  for(int ihel = 0; ihel < ncomb; ihel ++){
    if (goodhel[flav_idx][ihel] || ntry[flav_idx] < 2){
      calculate_wavefunctions(perm, helicities[ihel], flavor);
      %(get_matrix_t_lines)s
%(get_mirror_matrix_lines)s
      double tsum = 0;
      for(int iproc = 0;iproc < nprocesses; iproc++){
         matrix_element[iproc]+=t[iproc];
         tsum += t[iproc];
      }
      // Store which helicities give non-zero result
      if (tsum != 0. && !goodhel[flav_idx][ihel]){
	goodhel[flav_idx][ihel]=true;
	ngood[flav_idx] ++;
	igood[flav_idx][ngood[flav_idx]] = ihel;
      }
    }
  }
  jhel[flav_idx] = 0;
  sum_hel[flav_idx]=min(sum_hel[flav_idx], ngood[flav_idx]);
}
else              
{
// Only use the "good" helicities
  for(int j=0; j < sum_hel[flav_idx]; j++){
    jhel[flav_idx]++;
    if (jhel[flav_idx] >= ngood[flav_idx]) jhel[flav_idx]=0;
    double hwgt = double(ngood[flav_idx])/double(sum_hel[flav_idx]);
    int ihel = igood[flav_idx][jhel[flav_idx]];
    calculate_wavefunctions(perm, helicities[ihel], flavor);
    %(get_matrix_t_lines)s
%(get_mirror_matrix_lines)s
    for(int iproc = 0;iproc < nprocesses; iproc++){
      matrix_element[iproc]+=t[iproc]*hwgt;
    }
  }
}

for (int i=0;i < nprocesses; i++)
    matrix_element[i] = matrix_element[i] * broken_sym(flavor) / denominators[i];


