double %(process_class_name)s::matrix_%(proc_name)s(%(matrix_args)s) { 
%(all_wavefunction_calls)s
// Local variables
const int ngraphs = %(ngraphs)d;
const int ncolor = %(ncolor)d;
std::complex<double> ztemp;
std::complex<double> jamp[ncolor];
// The color matrix;
%(color_matrix_lines)s

// Calculate color flows
%(jamp_lines)s

// Sum and square the color flows to get the matrix element
double matrix = 0;
int cf_index = 0;
for(int i=0;i < ncolor; i++){
  ztemp = 0.;
  for(int j = i; j < ncolor; j++, cf_index++){
    ztemp = ztemp + static_cast<double>(cf[cf_index])*jamp[j];
   }
  matrix = matrix+real(ztemp*conj(jamp[i]));
}
 matrix = matrix/denom;

// Store the leading color flows for choice of color
for(int i=0;i < ncolor; i++)
    jamp2[%(proc_number)d][i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
}
