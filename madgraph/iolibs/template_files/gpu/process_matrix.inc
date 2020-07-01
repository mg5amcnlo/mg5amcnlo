//double %(process_class_name)s::matrix_%(proc_name)s(%(matrix_args)s) { 
%(all_wavefunction_calls)s
// Local variables

  // const int ngraphs = 2;
  const int ncolor = 1;
  thrust::complex<double> ztemp;
  thrust::complex<double> jamp[ncolor];
  // The color matrix;
%(color_matrix_lines)s

// Calculate color flows
%(jamp_lines)s

// Sum and square the color flows to get the matrix element
matrix = 0;
for(i=0;i < ncolor; i++){
  ztemp = 0.;
  for(j = 0; j < ncolor; j++)
    ztemp = ztemp + cf[i][j]*jamp[j];
  matrix = matrix+(ztemp*conj(jamp[i])).real()/denom[i];
}

// Store the leading color flows for choice of color
//for(i=0;i < ncolor; i++)
//    jamp2[%(proc_number)d][i] += real(jamp[i]*conj(jamp[i]));
    
}
