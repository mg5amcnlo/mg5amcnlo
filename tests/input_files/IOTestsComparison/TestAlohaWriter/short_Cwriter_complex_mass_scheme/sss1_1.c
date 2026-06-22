#include "SSS1_1.h"

 void SSS1_1(ALOHAOBJ S2, ALOHAOBJ S3, std::complex<double> COUP, std::complex<double> M1,ALOHAOBJ  & S1)
{
static std::complex<double> cI = std::complex<double>(0.,1.);
 double  P1[4];
 std::complex<double>  denom;
    S1.p[0] = +S2.p[0]+S3.p[0];
    S1.p[1] = +S2.p[1]+S3.p[1];
    S1.p[2] = +S2.p[2]+S3.p[2];
    S1.p[3] = +S2.p[3]+S3.p[3];
P1[0] = -S1.p[0];
P1[1] = -S1.p[1];
P1[2] = -S1.p[2];
P1[3] = -S1.p[3];
    denom = COUP/((P1[0]*P1[0])-(P1[1]*P1[1])-(P1[2]*P1[2])-(P1[3]*P1[3]) - (M1*M1));
    S1.W[0]= denom*cI * S3.W[0]*S2.W[0];
}

 void SSS1_2(ALOHAOBJ S2, ALOHAOBJ S3, std::complex<double> COUP, std::complex<double> M1,ALOHAOBJ  & S1)
{

 SSS1_1(S2,S3,COUP,M1,S1);
}
 void SSS1_3(ALOHAOBJ S2, ALOHAOBJ S3, std::complex<double> COUP, std::complex<double> M1,ALOHAOBJ  & S1)
{

 SSS1_1(S2,S3,COUP,M1,S1);
}
