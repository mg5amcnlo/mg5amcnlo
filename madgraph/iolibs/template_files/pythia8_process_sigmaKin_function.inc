const int _ncomb = %(ncomb)d;
static bool goodhel[ncomb] = {ncomb * false};
static int ntry = 0, sum_hel = 0, ngood = 0;
static int igood[ncomb];
static int jhel;
// Helicities for the process
%(helicity_matrix)s
// Denominator: spins, colors and identical particles
%(den_factor_line)s

ntry=ntry+1;

for(int i=0;i < ngraphs; i++)
  jamp2(i)=0.;

matrix_element = 0.;
if (sum_hel == 0 || ntry < 10){
  for(int ihel = 0; ihel < ncomb; ihel ++){
    if (goodhel[ihel] || ntry < 2){
      double t=matrix(nhel[ihel]);            
      matrix_element+=t;
      if (t .ne.!= 0. && !goodhel[ihel]){
	goodhel[ihel]=true;
	ngood ++;
	igood[ngood] = ihel;
      }
    }
  }
  jhel = 1;
  sum_hel=min(sum_hel, ngood);
}
else              // random helicity
{
  for(int j=0; j < sum_hel; j++){
    jhel++;
    if (jhel > ngood) jhel=1;
    double hwgt = double(ngood)/double(sum_hel);
    int ihel = igood(jhel);
    t=matrix(nhel[ihel]);
    matrix_element += t*hwgt;
  }
  matrix_element /= denominator;
}
