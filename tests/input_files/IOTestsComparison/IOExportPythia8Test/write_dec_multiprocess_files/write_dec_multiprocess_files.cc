//==========================================================================
// Class member functions for calculating the matrix elements for
# Process: g d > z d WEIGHTED<=3
# *   Decay: z > d d~ WEIGHTED<=2
# Process: g s > z s WEIGHTED<=3
# *   Decay: z > d d~ WEIGHTED<=2
# Process: g d > z d WEIGHTED<=3
# *   Decay: z > u u~ WEIGHTED<=2
# Process: g s > z s WEIGHTED<=3
# *   Decay: z > u u~ WEIGHTED<=2
# Process: g d > z d WEIGHTED<=3
# *   Decay: z > s s~ WEIGHTED<=2
# Process: g s > z s WEIGHTED<=3
# *   Decay: z > s s~ WEIGHTED<=2
# Process: g u > z u WEIGHTED<=3
# *   Decay: z > d d~ WEIGHTED<=2
# Process: g u > z u WEIGHTED<=3
# *   Decay: z > s s~ WEIGHTED<=2
# Process: g u > z u WEIGHTED<=3
# *   Decay: z > u u~ WEIGHTED<=2
# Process: g d~ > z d~ WEIGHTED<=3
# *   Decay: z > d d~ WEIGHTED<=2
# Process: g s~ > z s~ WEIGHTED<=3
# *   Decay: z > d d~ WEIGHTED<=2
# Process: g d~ > z d~ WEIGHTED<=3
# *   Decay: z > u u~ WEIGHTED<=2
# Process: g s~ > z s~ WEIGHTED<=3
# *   Decay: z > u u~ WEIGHTED<=2
# Process: g d~ > z d~ WEIGHTED<=3
# *   Decay: z > s s~ WEIGHTED<=2
# Process: g s~ > z s~ WEIGHTED<=3
# *   Decay: z > s s~ WEIGHTED<=2
# Process: g u~ > z u~ WEIGHTED<=3
# *   Decay: z > d d~ WEIGHTED<=2
# Process: g u~ > z u~ WEIGHTED<=3
# *   Decay: z > s s~ WEIGHTED<=2
# Process: g u~ > z u~ WEIGHTED<=3
# *   Decay: z > u u~ WEIGHTED<=2

//--------------------------------------------------------------------------
// Initialize process. 
  
void Sigma_sm_gd_ddxd::initProc() {
// Instantiate the model class and set parameters that stay fixed during run
    pars = Parameters_sm::getInstance();
    pars->setIndependentParameters(particleDataPtr, coupSUSYPtr, slhaPtr);
    pars->setIndependentCouplings();
    // Set massive/massless matrix elements for c/b/mu/tau
mcME = 0.;
mbME = 0.;
mmuME = 0.;
mtauME = 0.;
jamp2[0] = new double[1];
jamp2[1] = new double[1];
jamp2[2] = new double[1];
jamp2[3] = new double[1];
jamp2[4] = new double[1];
jamp2[5] = new double[1];
jamp2[6] = new double[1];
jamp2[7] = new double[1];
jamp2[8] = new double[1];
jamp2[9] = new double[1];
} 

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour. 

void Sigma_sm_gd_ddxd::sigmaKin() { 
    // Set the parameters which change event by event
    pars->setDependentParameters(particleDataPtr, coupSUSYPtr, slhaPtr, alpS);
    pars->setDependentCouplings();
    // Reset color flows
    for(int i=0;i < 1; i++)
            jamp2[0][i]=0.;
for(int i=0;i < 1; i++)
            jamp2[1][i]=0.;
for(int i=0;i < 1; i++)
            jamp2[2][i]=0.;
for(int i=0;i < 1; i++)
            jamp2[3][i]=0.;
for(int i=0;i < 1; i++)
            jamp2[4][i]=0.;
for(int i=0;i < 1; i++)
            jamp2[5][i]=0.;
for(int i=0;i < 1; i++)
            jamp2[6][i]=0.;
for(int i=0;i < 1; i++)
            jamp2[7][i]=0.;
for(int i=0;i < 1; i++)
            jamp2[8][i]=0.;
for(int i=0;i < 1; i++)
            jamp2[9][i]=0.;

    // Local variables and constants
const int ncomb = 32;
static bool goodhel[ncomb] = {ncomb * false};
static int ntry = 0, sum_hel = 0, ngood = 0;
static int igood[ncomb];
static int jhel;
double t[nprocesses];
// Helicities for the process
static const int helicities[ncomb][nexternal] = {{-1,-1,-1,-1,-1},{-1,-1,-1,-1,1},{-1,-1,-1,1,-1},{-1,-1,-1,1,1},{-1,-1,1,-1,-1},{-1,-1,1,-1,1},{-1,-1,1,1,-1},{-1,-1,1,1,1},{-1,1,-1,-1,-1},{-1,1,-1,-1,1},{-1,1,-1,1,-1},{-1,1,-1,1,1},{-1,1,1,-1,-1},{-1,1,1,-1,1},{-1,1,1,1,-1},{-1,1,1,1,1},{1,-1,-1,-1,-1},{1,-1,-1,-1,1},{1,-1,-1,1,-1},{1,-1,-1,1,1},{1,-1,1,-1,-1},{1,-1,1,-1,1},{1,-1,1,1,-1},{1,-1,1,1,1},{1,1,-1,-1,-1},{1,1,-1,-1,1},{1,1,-1,1,-1},{1,1,-1,1,1},{1,1,1,-1,-1},{1,1,1,-1,1},{1,1,1,1,-1},{1,1,1,1,1}};
// Denominators: spins, colors and identical particles
const int denominators[nprocesses] = {96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96};

ntry=ntry+1;

// Reset the matrix elements
for(int i = 0; i < nprocesses; i++){
    matrix_element[i] = 0.;
    t[i] = 0.;
}

// Define permutation
int perm[nexternal];
for(int i = 0; i < nexternal; i++){
  perm[i]=i;
}

// For now, call setupForME() here
id1 = 21;
id2 = 1;
if(!setupForME()){
    return;
}

if (sum_hel == 0 || ntry < 10){
// Calculate the matrix element for all helicities
  for(int ihel = 0; ihel < ncomb; ihel ++){
    if (goodhel[ihel] || ntry < 2){
      calculate_wavefunctions(perm, helicities[ihel]);
      t[0]=matrix_gd_zd_z_ddx();
t[1]=matrix_gd_zd_z_uux();
t[2]=matrix_gd_zd_z_ssx();
t[3]=matrix_gu_zu_z_ddx();
t[4]=matrix_gu_zu_z_uux();
t[5]=matrix_gdx_zdx_z_ddx();
t[6]=matrix_gdx_zdx_z_uux();
t[7]=matrix_gdx_zdx_z_ssx();
t[8]=matrix_gux_zux_z_ddx();
t[9]=matrix_gux_zux_z_uux();
             // Mirror initial state momenta for mirror process
                perm[0]=1;
                perm[1]=0;
                // Calculate wavefunctions
                calculate_wavefunctions(perm, helicities[ihel]);
                // Mirror back
                perm[0]=0;
                perm[1]=1;
                // Calculate matrix elements
                t[10]=matrix_gd_zd_z_ddx();
t[11]=matrix_gd_zd_z_uux();
t[12]=matrix_gd_zd_z_ssx();
t[13]=matrix_gu_zu_z_ddx();
t[14]=matrix_gu_zu_z_uux();
t[15]=matrix_gdx_zdx_z_ddx();
t[16]=matrix_gdx_zdx_z_uux();
t[17]=matrix_gdx_zdx_z_ssx();
t[18]=matrix_gux_zux_z_ddx();
t[19]=matrix_gux_zux_z_uux();
      double tsum = 0;
      for(int iproc = 0;iproc < nprocesses; iproc++){
         matrix_element[iproc]+=t[iproc];
         tsum += t[iproc];
      }
      // Store which helicities give non-zero result
      if (tsum != 0. && !goodhel[ihel]){
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
    calculate_wavefunctions(perm, helicities[ihel]);
    t[0]=matrix_gd_zd_z_ddx();
t[1]=matrix_gd_zd_z_uux();
t[2]=matrix_gd_zd_z_ssx();
t[3]=matrix_gu_zu_z_ddx();
t[4]=matrix_gu_zu_z_uux();
t[5]=matrix_gdx_zdx_z_ddx();
t[6]=matrix_gdx_zdx_z_uux();
t[7]=matrix_gdx_zdx_z_ssx();
t[8]=matrix_gux_zux_z_ddx();
t[9]=matrix_gux_zux_z_uux();
             // Mirror initial state momenta for mirror process
                perm[0]=1;
                perm[1]=0;
                // Calculate wavefunctions
                calculate_wavefunctions(perm, helicities[ihel]);
                // Mirror back
                perm[0]=0;
                perm[1]=1;
                // Calculate matrix elements
                t[10]=matrix_gd_zd_z_ddx();
t[11]=matrix_gd_zd_z_uux();
t[12]=matrix_gd_zd_z_ssx();
t[13]=matrix_gu_zu_z_ddx();
t[14]=matrix_gu_zu_z_uux();
t[15]=matrix_gdx_zdx_z_ddx();
t[16]=matrix_gdx_zdx_z_uux();
t[17]=matrix_gdx_zdx_z_ssx();
t[18]=matrix_gux_zux_z_ddx();
t[19]=matrix_gux_zux_z_uux();
    for(int iproc = 0;iproc < nprocesses; iproc++){
      matrix_element[iproc]+=t[iproc]*hwgt;
    }
  }
}

for (int i=0;i < nprocesses; i++)
    matrix_element[i] /= denominators[i];



}

//--------------------------------------------------------------------------
// Evaluate |M|^2, including incoming flavour dependence. 

double Sigma_sm_gd_ddxd::sigmaHat() {  
    // Select between the different processes
if(id1 == -3 && id2 == 21){
// Add matrix elements for processes with beams (-3, 21)
return matrix_element[15]+matrix_element[16]+matrix_element[17];
}
else if(id1 == -2 && id2 == 21){
// Add matrix elements for processes with beams (-2, 21)
return matrix_element[18]*2+matrix_element[19];
}
else if(id1 == -1 && id2 == 21){
// Add matrix elements for processes with beams (-1, 21)
return matrix_element[15]+matrix_element[16]+matrix_element[17];
}
else if(id1 == 1 && id2 == 21){
// Add matrix elements for processes with beams (1, 21)
return matrix_element[10]+matrix_element[11]+matrix_element[12];
}
else if(id1 == 2 && id2 == 21){
// Add matrix elements for processes with beams (2, 21)
return matrix_element[13]*2+matrix_element[14];
}
else if(id1 == 3 && id2 == 21){
// Add matrix elements for processes with beams (3, 21)
return matrix_element[10]+matrix_element[11]+matrix_element[12];
}
else if(id1 == 21 && id2 == -3){
// Add matrix elements for processes with beams (21, -3)
return matrix_element[5]+matrix_element[6]+matrix_element[7];
}
else if(id1 == 21 && id2 == -2){
// Add matrix elements for processes with beams (21, -2)
return matrix_element[8]*2+matrix_element[9];
}
else if(id1 == 21 && id2 == -1){
// Add matrix elements for processes with beams (21, -1)
return matrix_element[5]+matrix_element[6]+matrix_element[7];
}
else if(id1 == 21 && id2 == 1){
// Add matrix elements for processes with beams (21, 1)
return matrix_element[0]+matrix_element[1]+matrix_element[2];
}
else if(id1 == 21 && id2 == 2){
// Add matrix elements for processes with beams (21, 2)
return matrix_element[3]*2+matrix_element[4];
}
else if(id1 == 21 && id2 == 3){
// Add matrix elements for processes with beams (21, 3)
return matrix_element[0]+matrix_element[1]+matrix_element[2];
}
else {
// Return 0 if not correct initial state assignment
 return 0.;}
}

//--------------------------------------------------------------------------
// Select identity, colour and anticolour.

void Sigma_sm_gd_ddxd::setIdColAcol() {
    if(id1 == -3 && id2 == 21){
// Pick one of the flavor combinations 
int flavors[3][3] = {{3,-3,-3},{2,-2,-3},{1,-1,-3}};
vector<double> probs;
double sum = matrix_element[17]+matrix_element[16]+matrix_element[15];
probs.push_back(matrix_element[17]/sum);
probs.push_back(matrix_element[16]/sum);
probs.push_back(matrix_element[15]/sum);
int choice = rndmPtr->pick(probs);
id3 = flavors[choice][0];
id4 = flavors[choice][1];
id5 = flavors[choice][2];
}
else if(id1 == -2 && id2 == 21){
// Pick one of the flavor combinations 
int flavors[3][3] = {{3,-3,-2},{1,-1,-2},{2,-2,-2}};
vector<double> probs;
double sum = matrix_element[18]+matrix_element[18]+matrix_element[19];
probs.push_back(matrix_element[18]/sum);
probs.push_back(matrix_element[18]/sum);
probs.push_back(matrix_element[19]/sum);
int choice = rndmPtr->pick(probs);
id3 = flavors[choice][0];
id4 = flavors[choice][1];
id5 = flavors[choice][2];
}
else if(id1 == -1 && id2 == 21){
// Pick one of the flavor combinations 
int flavors[3][3] = {{3,-3,-1},{2,-2,-1},{1,-1,-1}};
vector<double> probs;
double sum = matrix_element[17]+matrix_element[16]+matrix_element[15];
probs.push_back(matrix_element[17]/sum);
probs.push_back(matrix_element[16]/sum);
probs.push_back(matrix_element[15]/sum);
int choice = rndmPtr->pick(probs);
id3 = flavors[choice][0];
id4 = flavors[choice][1];
id5 = flavors[choice][2];
}
else if(id1 == 1 && id2 == 21){
// Pick one of the flavor combinations 
int flavors[3][3] = {{2,-2,1},{3,-3,1},{1,-1,1}};
vector<double> probs;
double sum = matrix_element[11]+matrix_element[12]+matrix_element[10];
probs.push_back(matrix_element[11]/sum);
probs.push_back(matrix_element[12]/sum);
probs.push_back(matrix_element[10]/sum);
int choice = rndmPtr->pick(probs);
id3 = flavors[choice][0];
id4 = flavors[choice][1];
id5 = flavors[choice][2];
}
else if(id1 == 2 && id2 == 21){
// Pick one of the flavor combinations 
int flavors[3][3] = {{3,-3,2},{1,-1,2},{2,-2,2}};
vector<double> probs;
double sum = matrix_element[13]+matrix_element[13]+matrix_element[14];
probs.push_back(matrix_element[13]/sum);
probs.push_back(matrix_element[13]/sum);
probs.push_back(matrix_element[14]/sum);
int choice = rndmPtr->pick(probs);
id3 = flavors[choice][0];
id4 = flavors[choice][1];
id5 = flavors[choice][2];
}
else if(id1 == 3 && id2 == 21){
// Pick one of the flavor combinations 
int flavors[3][3] = {{2,-2,3},{1,-1,3},{3,-3,3}};
vector<double> probs;
double sum = matrix_element[11]+matrix_element[10]+matrix_element[12];
probs.push_back(matrix_element[11]/sum);
probs.push_back(matrix_element[10]/sum);
probs.push_back(matrix_element[12]/sum);
int choice = rndmPtr->pick(probs);
id3 = flavors[choice][0];
id4 = flavors[choice][1];
id5 = flavors[choice][2];
}
else if(id1 == 21 && id2 == -3){
// Pick one of the flavor combinations (3, -3, -3), (2, -2, -3), (1, -1, -3)
int flavors[3][3] = {{3,-3,-3},{2,-2,-3},{1,-1,-3}};
vector<double> probs;
double sum = matrix_element[7]+matrix_element[6]+matrix_element[5];
probs.push_back(matrix_element[7]/sum);
probs.push_back(matrix_element[6]/sum);
probs.push_back(matrix_element[5]/sum);
int choice = rndmPtr->pick(probs);
id3 = flavors[choice][0];
id4 = flavors[choice][1];
id5 = flavors[choice][2];
}
else if(id1 == 21 && id2 == -2){
// Pick one of the flavor combinations (3, -3, -2), (1, -1, -2), (2, -2, -2)
int flavors[3][3] = {{3,-3,-2},{1,-1,-2},{2,-2,-2}};
vector<double> probs;
double sum = matrix_element[8]+matrix_element[8]+matrix_element[9];
probs.push_back(matrix_element[8]/sum);
probs.push_back(matrix_element[8]/sum);
probs.push_back(matrix_element[9]/sum);
int choice = rndmPtr->pick(probs);
id3 = flavors[choice][0];
id4 = flavors[choice][1];
id5 = flavors[choice][2];
}
else if(id1 == 21 && id2 == -1){
// Pick one of the flavor combinations (3, -3, -1), (2, -2, -1), (1, -1, -1)
int flavors[3][3] = {{3,-3,-1},{2,-2,-1},{1,-1,-1}};
vector<double> probs;
double sum = matrix_element[7]+matrix_element[6]+matrix_element[5];
probs.push_back(matrix_element[7]/sum);
probs.push_back(matrix_element[6]/sum);
probs.push_back(matrix_element[5]/sum);
int choice = rndmPtr->pick(probs);
id3 = flavors[choice][0];
id4 = flavors[choice][1];
id5 = flavors[choice][2];
}
else if(id1 == 21 && id2 == 1){
// Pick one of the flavor combinations (2, -2, 1), (3, -3, 1), (1, -1, 1)
int flavors[3][3] = {{2,-2,1},{3,-3,1},{1,-1,1}};
vector<double> probs;
double sum = matrix_element[1]+matrix_element[2]+matrix_element[0];
probs.push_back(matrix_element[1]/sum);
probs.push_back(matrix_element[2]/sum);
probs.push_back(matrix_element[0]/sum);
int choice = rndmPtr->pick(probs);
id3 = flavors[choice][0];
id4 = flavors[choice][1];
id5 = flavors[choice][2];
}
else if(id1 == 21 && id2 == 2){
// Pick one of the flavor combinations (3, -3, 2), (1, -1, 2), (2, -2, 2)
int flavors[3][3] = {{3,-3,2},{1,-1,2},{2,-2,2}};
vector<double> probs;
double sum = matrix_element[3]+matrix_element[3]+matrix_element[4];
probs.push_back(matrix_element[3]/sum);
probs.push_back(matrix_element[3]/sum);
probs.push_back(matrix_element[4]/sum);
int choice = rndmPtr->pick(probs);
id3 = flavors[choice][0];
id4 = flavors[choice][1];
id5 = flavors[choice][2];
}
else if(id1 == 21 && id2 == 3){
// Pick one of the flavor combinations (2, -2, 3), (1, -1, 3), (3, -3, 3)
int flavors[3][3] = {{2,-2,3},{1,-1,3},{3,-3,3}};
vector<double> probs;
double sum = matrix_element[1]+matrix_element[0]+matrix_element[2];
probs.push_back(matrix_element[1]/sum);
probs.push_back(matrix_element[0]/sum);
probs.push_back(matrix_element[2]/sum);
int choice = rndmPtr->pick(probs);
id3 = flavors[choice][0];
id4 = flavors[choice][1];
id5 = flavors[choice][2];
}
setId(id1,id2,id3,id4,id5);
// Pick color flow
int ncolor[10] = {1,1,1,1,1,1,1,1,1,1};
if((id1 == 21&&id2 == 1&&id3 == 1&&id4 == -1&&id5 == 1)||(id1 == 21&&id2 == 3&&id3 == 1&&id4 == -1&&id5 == 3)){
vector<double> probs;
                  double sum = jamp2[0][0];
                  for(int i=0;i<ncolor[0];i++)
                  probs.push_back(jamp2[0][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{2,3,3,0,1,0,0,1,2,0}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 21&&id2 == 1&&id3 == 2&&id4 == -2&&id5 == 1)||(id1 == 21&&id2 == 3&&id3 == 2&&id4 == -2&&id5 == 3)){
vector<double> probs;
                  double sum = jamp2[1][0];
                  for(int i=0;i<ncolor[1];i++)
                  probs.push_back(jamp2[1][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{2,3,3,0,1,0,0,1,2,0}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 21&&id2 == 1&&id3 == 3&&id4 == -3&&id5 == 1)||(id1 == 21&&id2 == 3&&id3 == 3&&id4 == -3&&id5 == 3)){
vector<double> probs;
                  double sum = jamp2[2][0];
                  for(int i=0;i<ncolor[2];i++)
                  probs.push_back(jamp2[2][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{2,3,3,0,1,0,0,1,2,0}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 21&&id2 == 2&&id3 == 1&&id4 == -1&&id5 == 2)||(id1 == 21&&id2 == 2&&id3 == 3&&id4 == -3&&id5 == 2)){
vector<double> probs;
                  double sum = jamp2[3][0];
                  for(int i=0;i<ncolor[3];i++)
                  probs.push_back(jamp2[3][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{2,3,3,0,1,0,0,1,2,0}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 21&&id2 == 2&&id3 == 2&&id4 == -2&&id5 == 2)){
vector<double> probs;
                  double sum = jamp2[4][0];
                  for(int i=0;i<ncolor[4];i++)
                  probs.push_back(jamp2[4][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{2,3,3,0,1,0,0,1,2,0}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 21&&id2 == -1&&id3 == 1&&id4 == -1&&id5 == -1)||(id1 == 21&&id2 == -3&&id3 == 1&&id4 == -1&&id5 == -3)){
vector<double> probs;
                  double sum = jamp2[5][0];
                  for(int i=0;i<ncolor[5];i++)
                  probs.push_back(jamp2[5][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{1,3,0,1,2,0,0,2,0,3}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 21&&id2 == -1&&id3 == 2&&id4 == -2&&id5 == -1)||(id1 == 21&&id2 == -3&&id3 == 2&&id4 == -2&&id5 == -3)){
vector<double> probs;
                  double sum = jamp2[6][0];
                  for(int i=0;i<ncolor[6];i++)
                  probs.push_back(jamp2[6][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{1,3,0,1,2,0,0,2,0,3}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 21&&id2 == -1&&id3 == 3&&id4 == -3&&id5 == -1)||(id1 == 21&&id2 == -3&&id3 == 3&&id4 == -3&&id5 == -3)){
vector<double> probs;
                  double sum = jamp2[7][0];
                  for(int i=0;i<ncolor[7];i++)
                  probs.push_back(jamp2[7][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{1,3,0,1,2,0,0,2,0,3}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 21&&id2 == -2&&id3 == 1&&id4 == -1&&id5 == -2)||(id1 == 21&&id2 == -2&&id3 == 3&&id4 == -3&&id5 == -2)){
vector<double> probs;
                  double sum = jamp2[8][0];
                  for(int i=0;i<ncolor[8];i++)
                  probs.push_back(jamp2[8][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{1,3,0,1,2,0,0,2,0,3}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 21&&id2 == -2&&id3 == 2&&id4 == -2&&id5 == -2)){
vector<double> probs;
                  double sum = jamp2[9][0];
                  for(int i=0;i<ncolor[9];i++)
                  probs.push_back(jamp2[9][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{1,3,0,1,2,0,0,2,0,3}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 1&&id2 == 21&&id3 == 1&&id4 == -1&&id5 == 1)||(id1 == 3&&id2 == 21&&id3 == 1&&id4 == -1&&id5 == 3)){
vector<double> probs;
                  double sum = jamp2[0][0];
                  for(int i=0;i<ncolor[0];i++)
                  probs.push_back(jamp2[0][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{3,0,2,3,1,0,0,1,2,0}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 1&&id2 == 21&&id3 == 2&&id4 == -2&&id5 == 1)||(id1 == 3&&id2 == 21&&id3 == 2&&id4 == -2&&id5 == 3)){
vector<double> probs;
                  double sum = jamp2[1][0];
                  for(int i=0;i<ncolor[1];i++)
                  probs.push_back(jamp2[1][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{3,0,2,3,1,0,0,1,2,0}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 1&&id2 == 21&&id3 == 3&&id4 == -3&&id5 == 1)||(id1 == 3&&id2 == 21&&id3 == 3&&id4 == -3&&id5 == 3)){
vector<double> probs;
                  double sum = jamp2[2][0];
                  for(int i=0;i<ncolor[2];i++)
                  probs.push_back(jamp2[2][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{3,0,2,3,1,0,0,1,2,0}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 2&&id2 == 21&&id3 == 1&&id4 == -1&&id5 == 2)||(id1 == 2&&id2 == 21&&id3 == 3&&id4 == -3&&id5 == 2)){
vector<double> probs;
                  double sum = jamp2[3][0];
                  for(int i=0;i<ncolor[3];i++)
                  probs.push_back(jamp2[3][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{3,0,2,3,1,0,0,1,2,0}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == 2&&id2 == 21&&id3 == 2&&id4 == -2&&id5 == 2)){
vector<double> probs;
                  double sum = jamp2[4][0];
                  for(int i=0;i<ncolor[4];i++)
                  probs.push_back(jamp2[4][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{3,0,2,3,1,0,0,1,2,0}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == -1&&id2 == 21&&id3 == 1&&id4 == -1&&id5 == -1)||(id1 == -3&&id2 == 21&&id3 == 1&&id4 == -1&&id5 == -3)){
vector<double> probs;
                  double sum = jamp2[5][0];
                  for(int i=0;i<ncolor[5];i++)
                  probs.push_back(jamp2[5][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{0,1,1,3,2,0,0,2,0,3}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == -1&&id2 == 21&&id3 == 2&&id4 == -2&&id5 == -1)||(id1 == -3&&id2 == 21&&id3 == 2&&id4 == -2&&id5 == -3)){
vector<double> probs;
                  double sum = jamp2[6][0];
                  for(int i=0;i<ncolor[6];i++)
                  probs.push_back(jamp2[6][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{0,1,1,3,2,0,0,2,0,3}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == -1&&id2 == 21&&id3 == 3&&id4 == -3&&id5 == -1)||(id1 == -3&&id2 == 21&&id3 == 3&&id4 == -3&&id5 == -3)){
vector<double> probs;
                  double sum = jamp2[7][0];
                  for(int i=0;i<ncolor[7];i++)
                  probs.push_back(jamp2[7][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{0,1,1,3,2,0,0,2,0,3}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == -2&&id2 == 21&&id3 == 1&&id4 == -1&&id5 == -2)||(id1 == -2&&id2 == 21&&id3 == 3&&id4 == -3&&id5 == -2)){
vector<double> probs;
                  double sum = jamp2[8][0];
                  for(int i=0;i<ncolor[8];i++)
                  probs.push_back(jamp2[8][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{0,1,1,3,2,0,0,2,0,3}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
else if((id1 == -2&&id2 == 21&&id3 == 2&&id4 == -2&&id5 == -2)){
vector<double> probs;
                  double sum = jamp2[9][0];
                  for(int i=0;i<ncolor[9];i++)
                  probs.push_back(jamp2[9][i]/sum);
                  int ic = rndmPtr->pick(probs);
static int colors[1][10] = {{0,1,1,3,2,0,0,2,0,3}};
setColAcol(colors[ic][0],colors[ic][1],colors[ic][2],colors[ic][3],colors[ic][4],colors[ic][5],colors[ic][6],colors[ic][7],colors[ic][8],colors[ic][9]);
}
}

//--------------------------------------------------------------------------
// Evaluate weight for angles of decay products in process 

double Sigma_sm_gd_ddxd::weightDecay(Event& process, int iResBeg, int iResEnd) {
    // Just use isotropic decay (default)
return 1.;
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

void Sigma_sm_gd_ddxd::calculate_wavefunctions(const int perm[], const int hel[]){
// Calculate wavefunctions for all processes
double p[nexternal][4];
int i;
 
// Convert Pythia 4-vectors to double[]
for(i=0;i < nexternal;i++){
    p[i][0] = pME[i].e();
    p[i][1] = pME[i].px();
    p[i][2] = pME[i].py();
    p[i][3] = pME[i].pz();
}

// Calculate all wavefunctions
vxxxxx(p[perm[0]],mME[0],hel[0],-1,w[0]);
ixxxxx(p[perm[1]],mME[1],hel[1],+1,w[1]);
oxxxxx(p[perm[2]],mME[2],hel[2],+1,w[2]);
ixxxxx(p[perm[3]],mME[3],hel[3],-1,w[3]);
FFV1_3(w[3],w[2],pars->GDZ2,pars->MZ,pars->WZ,w[4]);
oxxxxx(p[perm[4]],mME[4],hel[4],+1,w[5]);
FFV1_2(w[1],w[0],pars->GQQ,pars->zero,pars->zero,w[6]);
FFV1_1(w[5],w[0],pars->GQQ,pars->zero,pars->zero,w[7]);
FFV1_2_3(w[3],w[2],pars->GUZ1,pars->GUZ2,pars->MZ,pars->WZ,w[8]);
oxxxxx(p[perm[1]],mME[1],hel[1],-1,w[9]);
ixxxxx(p[perm[4]],mME[4],hel[4],-1,w[10]);
FFV1_2(w[10],w[0],pars->GQQ,pars->zero,pars->zero,w[11]);
FFV1_1(w[9],w[0],pars->GQQ,pars->zero,pars->zero,w[12]);

// Calculate all amplitudes
# Amplitude(s) for diagram number 0
FFV1_0(w[6],w[5],w[4],pars->GDZ2,amp[0]);
FFV1_0(w[1],w[7],w[4],pars->GDZ2,amp[1]);
FFV1_0(w[6],w[5],w[8],pars->GDZ2,amp[2]);
FFV1_0(w[1],w[7],w[8],pars->GDZ2,amp[3]);
FFV1_2_0(w[6],w[5],w[4],pars->GUZ1,pars->GUZ2,amp[4]);
FFV1_2_0(w[1],w[7],w[4],pars->GUZ1,pars->GUZ2,amp[5]);
FFV1_2_0(w[6],w[5],w[8],pars->GUZ1,pars->GUZ2,amp[6]);
FFV1_2_0(w[1],w[7],w[8],pars->GUZ1,pars->GUZ2,amp[7]);
FFV1_0(w[11],w[9],w[4],pars->GDZ2,amp[8]);
FFV1_0(w[10],w[12],w[4],pars->GDZ2,amp[9]);
FFV1_0(w[11],w[9],w[8],pars->GDZ2,amp[10]);
FFV1_0(w[10],w[12],w[8],pars->GDZ2,amp[11]);
FFV1_0(w[11],w[9],w[4],pars->GDZ2,amp[12]);
FFV1_0(w[10],w[12],w[4],pars->GDZ2,amp[13]);
FFV1_2_0(w[11],w[9],w[4],pars->GUZ1,pars->GUZ2,amp[14]);
FFV1_2_0(w[10],w[12],w[4],pars->GUZ1,pars->GUZ2,amp[15]);
FFV1_2_0(w[11],w[9],w[8],pars->GUZ1,pars->GUZ2,amp[16]);
FFV1_2_0(w[10],w[12],w[8],pars->GUZ1,pars->GUZ2,amp[17]);


}
double Sigma_sm_gd_ddxd::matrix_gd_zd_z_ddx() { 

// Local variables
const int ngraphs = 2;
const int ncolor = 1;
std::complex<double> ztemp;
std::complex<double> jamp[ncolor];
// The color matrix;
static const int denom = 1;
static const int cf[ncolor*(ncolor+1)/2] = {12};

// Calculate color flows
jamp[0]=-amp[0]-amp[1];

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
    jamp2[0][i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
}

double Sigma_sm_gd_ddxd::matrix_gd_zd_z_uux() { 

// Local variables
const int ngraphs = 2;
const int ncolor = 1;
std::complex<double> ztemp;
std::complex<double> jamp[ncolor];
// The color matrix;
static const int denom = 1;
static const int cf[ncolor*(ncolor+1)/2] = {12};

// Calculate color flows
jamp[0]=-amp[2]-amp[3];

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
    jamp2[1][i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
}

double Sigma_sm_gd_ddxd::matrix_gd_zd_z_ssx() { 

// Local variables
const int ngraphs = 2;
const int ncolor = 1;
std::complex<double> ztemp;
std::complex<double> jamp[ncolor];
// The color matrix;
static const int denom = 1;
static const int cf[ncolor*(ncolor+1)/2] = {12};

// Calculate color flows
jamp[0]=-amp[0]-amp[1];

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
    jamp2[2][i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
}

double Sigma_sm_gd_ddxd::matrix_gu_zu_z_ddx() { 

// Local variables
const int ngraphs = 2;
const int ncolor = 1;
std::complex<double> ztemp;
std::complex<double> jamp[ncolor];
// The color matrix;
static const int denom = 1;
static const int cf[ncolor*(ncolor+1)/2] = {12};

// Calculate color flows
jamp[0]=-amp[4]-amp[5];

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
    jamp2[3][i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
}

double Sigma_sm_gd_ddxd::matrix_gu_zu_z_uux() { 

// Local variables
const int ngraphs = 2;
const int ncolor = 1;
std::complex<double> ztemp;
std::complex<double> jamp[ncolor];
// The color matrix;
static const int denom = 1;
static const int cf[ncolor*(ncolor+1)/2] = {12};

// Calculate color flows
jamp[0]=-amp[6]-amp[7];

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
    jamp2[4][i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
}

double Sigma_sm_gd_ddxd::matrix_gdx_zdx_z_ddx() { 

// Local variables
const int ngraphs = 2;
const int ncolor = 1;
std::complex<double> ztemp;
std::complex<double> jamp[ncolor];
// The color matrix;
static const int denom = 1;
static const int cf[ncolor*(ncolor+1)/2] = {12};

// Calculate color flows
jamp[0]=+amp[8]+amp[9];

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
    jamp2[5][i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
}

double Sigma_sm_gd_ddxd::matrix_gdx_zdx_z_uux() { 

// Local variables
const int ngraphs = 2;
const int ncolor = 1;
std::complex<double> ztemp;
std::complex<double> jamp[ncolor];
// The color matrix;
static const int denom = 1;
static const int cf[ncolor*(ncolor+1)/2] = {12};

// Calculate color flows
jamp[0]=+amp[10]+amp[11];

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
    jamp2[6][i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
}

double Sigma_sm_gd_ddxd::matrix_gdx_zdx_z_ssx() { 

// Local variables
const int ngraphs = 2;
const int ncolor = 1;
std::complex<double> ztemp;
std::complex<double> jamp[ncolor];
// The color matrix;
static const int denom = 1;
static const int cf[ncolor*(ncolor+1)/2] = {12};

// Calculate color flows
jamp[0]=+amp[12]+amp[13];

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
    jamp2[7][i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
}

double Sigma_sm_gd_ddxd::matrix_gux_zux_z_ddx() { 

// Local variables
const int ngraphs = 2;
const int ncolor = 1;
std::complex<double> ztemp;
std::complex<double> jamp[ncolor];
// The color matrix;
static const int denom = 1;
static const int cf[ncolor*(ncolor+1)/2] = {12};

// Calculate color flows
jamp[0]=+amp[14]+amp[15];

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
    jamp2[8][i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
}

double Sigma_sm_gd_ddxd::matrix_gux_zux_z_uux() { 

// Local variables
const int ngraphs = 2;
const int ncolor = 1;
std::complex<double> ztemp;
std::complex<double> jamp[ncolor];
// The color matrix;
static const int denom = 1;
static const int cf[ncolor*(ncolor+1)/2] = {12};

// Calculate color flows
jamp[0]=+amp[16]+amp[17];

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
    jamp2[9][i] += real(jamp[i]*conj(jamp[i]));
    
return matrix;
}

