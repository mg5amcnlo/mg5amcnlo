#include "LHAPDF/LHAPDF.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
int main(){LHAPDF::setVerbosity(0);auto p=LHAPDF::mkPDF("NNPDF40_nlo_as_01180_nf_4",0);double q,a,err=0;int n=0;while(std::cin>>q>>a){err=std::max(err,std::abs(a/p->alphasQ(q)-1));n++;}std::cout<<std::setprecision(17)<<"{\"points\":"<<n<<",\"max_relative_alpha_error\":"<<err<<"}\n";delete p;return err<1.e-6?0:1;}
