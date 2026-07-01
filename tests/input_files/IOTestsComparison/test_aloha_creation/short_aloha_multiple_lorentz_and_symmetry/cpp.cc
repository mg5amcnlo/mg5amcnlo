#include "VVS1_1.h"

 void VVS1_1(ALOHAOBJ V2, ALOHAOBJ S3, std::complex<double> COUP, double M1, double W1,ALOHAOBJ  & V1)
{
static std::complex<double> cI = std::complex<double>(0.,1.);
 double  OM1;
 double  P1[4];
 std::complex<double>  TMP0;
 std::complex<double>  denom;
    OM1 = 0.;
    if (M1 != 0.)
 OM1=1./(M1*M1);
    V1.p[0] = +V2.p[0]+S3.p[0];
    V1.p[1] = +V2.p[1]+S3.p[1];
    V1.p[2] = +V2.p[2]+S3.p[2];
    V1.p[3] = +V2.p[3]+S3.p[3];
P1[0] = -V1.p[0];
P1[1] = -V1.p[1];
P1[2] = -V1.p[2];
P1[3] = -V1.p[3];
 TMP0 = (V2.W[0]*P1[0]-V2.W[1]*P1[1]-V2.W[2]*P1[2]-V2.W[3]*P1[3]);
    denom = COUP/((P1[0]*P1[0])-(P1[1]*P1[1])-(P1[2]*P1[2])-(P1[3]*P1[3]) - M1 * (M1 -cI* W1));
    V1.W[0]= denom*S3.W[0]*(-cI*(V2.W[0])+cI*(P1[0]*OM1*TMP0));
    V1.W[1]= denom*S3.W[0]*(-cI*(V2.W[1])+cI*(P1[1]*OM1*TMP0));
    V1.W[2]= denom*S3.W[0]*(-cI*(V2.W[2])+cI*(P1[2]*OM1*TMP0));
    V1.W[3]= denom*S3.W[0]*(-cI*(V2.W[3])+cI*(P1[3]*OM1*TMP0));
}

 void VVS1_2(ALOHAOBJ V2, ALOHAOBJ S3, std::complex<double> COUP, double M1, double W1,ALOHAOBJ  & V1)
{

 VVS1_1(V2,S3,COUP,M1,W1,V1);
}
 void VVS1_2_1(ALOHAOBJ V2, ALOHAOBJ S3, std::complex<double> COUP1, std::complex<double> COUP2, double M1, double W1,ALOHAOBJ  & V1)
{
 ALOHAOBJ  Vtmp;
 int  i;
    VVS1_1(V2,S3,COUP1,M1,W1,V1);
    VVS2_1(V2,S3,COUP2,M1,W1,Vtmp);
 i= 0;
while (i < 4)
{
 V1.W[i] = V1.W[i] + Vtmp.W[i];
 i++;
}
}
 void VVS1_2_2(ALOHAOBJ V2, ALOHAOBJ S3, std::complex<double> COUP1, std::complex<double> COUP2, double M1, double W1,ALOHAOBJ  & V1)
{
 ALOHAOBJ  Vtmp;
 int  i;
    VVS1_1(V2,S3,COUP1,M1,W1,V1);
    VVS2_1(V2,S3,COUP2,M1,W1,Vtmp);
 i= 0;
while (i < 4)
{
 V1.W[i] = V1.W[i] + Vtmp.W[i];
 i++;
}
}
