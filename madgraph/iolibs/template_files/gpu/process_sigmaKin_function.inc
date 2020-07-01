  // for (int xx = 0; xx < 384; ++xx) {
  int nprocesses = %(nproc)i;
  int dim = blockIdx.x * blockDim.x + threadIdx.x;

  char *devPtr = (char *)tp.ptr;
  size_t dpt = tp.pitch;
  size_t slicePitch = dpt * 4;

  char *dps = devPtr + dim * slicePitch;
  double *matrix_element = (double *)((char *)meDevPtr + dim * mePitch);

  thrust::complex<double> amp[%(nb_amp)i];


// Local variables and constants
const int ncomb = %(ncomb)d;
//static bool goodhel[ncomb] = {ncomb * false};
//static int ntry = 0, sum_hel = 0, ngood = 0;
//static int igood[ncomb];
static int jhel;
std::complex<double> **wfs;
double t[%(nproc)i];
// Helicities for the process
//%(helicity_matrix)s
// Denominators: spins, colors and identical particles
const int denominators[nprocesses] = {%(den_factors)s};


// Reset the matrix elements
for(int i = 0; i < nprocesses; i++){
    matrix_element[i] = 0.;
}
// Define permutation
//int perm[nexternal];
//for(int i = 0; i < nexternal; i++){
//  perm[i]=i;
//}


for (int ihel = 0; ihel < ncomb; ihel++) {
    calculate_wavefunctions(ihel, dps, dpt, amp);
     %(get_matrix_t_lines)s
}


for (int i=0;i < nprocesses; ++i){
    matrix_element[i] /= denominators[i];
}

