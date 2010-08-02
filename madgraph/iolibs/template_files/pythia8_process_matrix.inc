// Local variables
const int nwavefuncs = %(nwavefuncs)d, ngraphs = %(ngraphs)d;
int i,j;
complex ztemp;
complex amp[ngraphs], jamp[ncolor];
complex w[nwavefuncs][18];
// The color matrix;
%(color_matrix_lines)s

// Convert Pythia 4-vectors to double[]
double p[nexternal][4];
for(i=0;i < nexternal;i++){
    p[i][0] = pME[i].e();
    p[i][1] = pME[i].px();
    p[i][2] = pME[i].py();
    p[i][3] = pME[i].pz();
}

// Calculate all amplitudes
%(amplitude_calls)s

// Calculate color flows
%(jamp_lines)s

// Sum and square the color flows to get the matrix element
double matrix = 0;
for(i=0;i < ncolor; i++){
  ztemp = 0.;
  for(j = 0; j < ncolor; j++)
    ztemp = ztemp + cf[i][j]*jamp[j];
  matrix = matrix+real(ztemp*conj(jamp[i]))/denom[i];
}

// Store the leading color flows for choice of color
for(i=0;i < ncolor; i++)
    jamp2[i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
