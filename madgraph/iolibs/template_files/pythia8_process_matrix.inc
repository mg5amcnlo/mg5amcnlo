// Local variables
const int nwavefuncs = %(nwavefuncs)d, ngraphs = %(ngraphs)d;
const double zero = 0., ZERO = 0.;
int i,j;
complex ztemp;
complex amp[ngraphs], jamp[ncolor];
complex w[nwavefuncs][18];
// The color matrix;
%(color_matrix_lines)s

// Calculate all amplitudes
%(amplitude_calls)s

// Calculate color flows
%(jamp_lines)s

// Sum and square the color flows to get the matrix element
double matrix = 0;
for(i=0;i < ncolor; i++){
  ztemp = 0.;
  for(j = 1; j < ncolor; j++)
    ztemp = ztemp + cf[i][j]*jamp[j];
  matrix = matrix+real(ztemp*conj(jamp[i]))/denom[i];
}

// Store the leading color flows for choice of color
for(i=0;i < ncolor; i++)
    jamp2[i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
