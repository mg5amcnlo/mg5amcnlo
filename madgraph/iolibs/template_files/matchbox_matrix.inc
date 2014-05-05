# 1 "/Users/omatt/Downloads/cpp_process_matrix.inc"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/Users/omatt/Downloads/cpp_process_matrix.inc"


std::complex<double> %(process_class_name)s::jamp(int i) {
  const int ncolor = %(ncolor)d;
  std::complex<double> jamp[ncolor];
%(jamp_lines)s
return jamp[i];
}

int %(process_class_name)s::colorstring(int i,int j) {
%(color_sting_lines)s
return res[i][j];
}


int %(process_class_name)s::NCol() {
const int ncolor = %(ncolor)d;
return ncolor;
}

