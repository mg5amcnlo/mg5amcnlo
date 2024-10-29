//double %(process_class_name)s::matrix_%(proc_name)s(%(matrix_args)s) { 


%(all_wavefunction_calls)s
// Local variables

  // The color matrix;
%(color_matrix_lines)s


// Sum and square the color flows to get the matrix element
for(int icol=0; icol < ncolor; icol++){
  cxtype ztemp = cxmake( 0, 0 );
  for(int jcol = 0; jcol < ncolor; jcol++)
    ztemp = ztemp + cf[icol][jcol]*jamp[jcol];
  meHelSum = meHelSum+ cxreal(ztemp*conj(jamp[icol]))/denom[icol];
}

// Store the leading color flows for choice of color
//for(i=0;i < ncolor; i++)
//    jamp2[%(proc_number)d][i] += real(jamp[i]*conj(jamp[i]));

    mgDebug( 1, __FUNCTION__ );
    return;
}
