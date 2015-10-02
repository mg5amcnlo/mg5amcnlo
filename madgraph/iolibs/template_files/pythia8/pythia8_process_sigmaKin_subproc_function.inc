void %(process_class_name)s::sigmaKin_%(proc_name)s() { 

// Local variables and constants
const int ncomb = %(ncomb)d;
static bool goodhel[ncomb] = {ncomb * false};
static int ntry = 0, sum_hel = 0, ngood = 0;
static int igood[ncomb];
static int jhel;
std::complex<double> **wfs;
double t;
// Helicities for the process
%(helicity_matrix)s
// Denominator: spins, colors and identical particles
const int denominator = %(den_factor)d;

ntry=ntry+1;

// Reset the matrix element
matrix_element[%(proc_number)d] = 0.;

if (sum_hel == 0 || ntry < 10){
// Calculate the matrix element for all helicities
  for(int ihel = 0; ihel < ncomb; ihel ++){
    if (goodhel[ihel] || ntry < 2){
      t=matrix_%(proc_name)s(helicities[ihel]);            
      matrix_element[%(proc_number)d]+=t;
      // Store which helicities give non-zero result
      if (t != 0. && !goodhel[ihel]){
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
    t=matrix_%(proc_name)s(helicities[ihel]);
    matrix_element[%(proc_number)d] += t*hwgt;
  }
}

matrix_element[%(proc_number)d] /= denominator;

}
