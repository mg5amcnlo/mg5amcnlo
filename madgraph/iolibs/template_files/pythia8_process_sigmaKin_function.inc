// Local variables and constants
const int ncomb = %(ncomb)d;
static bool goodhel[ncomb] = {ncomb * false};
static int ntry = 0, sum_hel = 0, ngood = 0;
static int igood[ncomb];
static int jhel;
double t[nprocesses];
// Helicities for the process
%(helicity_matrix)s
// Denominators: spins, colors and identical particles
const int denominators[nprocesses] = {%(den_factors)s};

ntry=ntry+1;

// Reset the matrix elements
for(int i = 0; i < nprocesses; i++){
    matrix_element[i] = 0.;
    t[i] = 0.;
}

// Define permutation
int perm[nexternal];
for(int i = 0; i < nexternal; i++){
  perm[i]=i;
}

// For now, call setupForME() here
id1 = %(id1)d;
id2 = %(id2)d;
if(!setupForME()){
    return;
}

if (sum_hel == 0 || ntry < 10){
// Calculate the matrix element for all helicities
  for(int ihel = 0; ihel < ncomb; ihel ++){
    if (goodhel[ihel] || ntry < 2){
      calculate_wavefunctions(perm, helicities[ihel]);
      %(get_matrix_t_lines)s
%(get_mirror_matrix_lines)s
      double tsum = 0;
      for(int iproc = 0;iproc < nprocesses; iproc++){
         matrix_element[iproc]+=t[iproc];
         tsum += t[iproc];
      }
      // Store which helicities give non-zero result
      if (tsum != 0. && !goodhel[ihel]){
	goodhel[ihel]=true;
	ngood ++;
	igood[ngood] = ihel;
      }
    }
  }
  jhel = 0;
  sum_hel=min(sum_hel, ngood);
}
else              
{
// Only use the "good" helicities
  for(int j=0; j < sum_hel; j++){
    jhel++;
    if (jhel >= ngood) jhel=0;
    double hwgt = double(ngood)/double(sum_hel);
    int ihel = igood[jhel];
    calculate_wavefunctions(perm, helicities[ihel]);
    %(get_matrix_t_lines)s
%(get_mirror_matrix_lines)s
    for(int iproc = 0;iproc < nprocesses; iproc++){
      matrix_element[iproc]+=t[iproc]*hwgt;
    }
  }
}

for (int i=0;i < nprocesses; i++)
    matrix_element[i] /= denominators[i];


