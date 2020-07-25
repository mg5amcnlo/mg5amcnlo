  // for (int xx = 0; xx < 384; ++xx) {
  const int nprocesses = %(nproc)i;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  //char *devPtr = (char *)tp.ptr;
  //size_t dpt = tp.pitch;
  //size_t slicePitch = dpt * %(nexternal)i;

  //char *dps = devPtr + dim * slicePitch;
  double matrix_element[nprocesses];

  thrust::complex<double> amp[1];// was %(nb_amp)i];


   Double local_m[%(nexternal)i][3];
   int DIM = blockDim.x * gridDim.x;
   //for (int i=0; i<20;i++){
   //     printf(" %%f ", allmomenta[i]);
   //}
   //printf("\n");
   //printf("DIM is %%i/%%i\n", tid, DIM);
   for (int i=0; i<%(nexternal)i;i++){
          for (int j=0; j<3;j++){	
                local_m[i][j] =  allmomenta[i*3*DIM+j*DIM+tid];
                //printf(" %%f ", local_m[i][j]);
          }
          //printf("\n");
  }


// Local variables and constants
const int ncomb = %(ncomb)d;
//static bool goodhel[ncomb] = {ncomb * false};
//static int ntry = 0, sum_hel = 0, ngood = 0;
//static int igood[ncomb];
//static int jhel;
//std::complex<double> **wfs;
//double t[%(nproc)i];
// Helicities for the process
//%(helicity_matrix)s
// Denominators: spins, colors and identical particles
const int denominators[%(nproc)i] = {%(den_factors)s};


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
    calculate_wavefunctions(ihel, local_m,  matrix_element[0]);
}


for (int i=0;i < nprocesses; ++i){
    matrix_element[i] /= denominators[i];
}
for (int i=0;i < nprocesses; ++i){
output[i*nprocesses+tid] = matrix_element[i];
//printf("output %%i %%i %%i %%f", tid, i, i*nprocesses+tid,  output[i*nprocesses+tid]);

}

