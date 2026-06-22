import cmath
import wavefunctions
def VVS1_1(V2,S3,COUP,M1,W1):
    OM1 = 0.0
    if (M1): OM1=1.0/M1**2
    V1 = wavefunctions.WaveFunction(size=6)
    V1.momenta[0] = +V2.momenta[0]+S3.momenta[0]
    V1.momenta[1] = +V2.momenta[1]+S3.momenta[1]
    V1.momenta[2] = +V2.momenta[2]+S3.momenta[2]
    V1.momenta[3] = +V2.momenta[3]+S3.momenta[3]
    P1 = [-V1.momenta[j] for j in range(4)]
    TMP0 = (V2.W[0]*P1[0]-V2.W[1]*P1[1]-V2.W[2]*P1[2]-V2.W[3]*P1[3])
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    V1.W[0]= denom*S3.W[0]*(-1j*(V2.W[0])+1j*(P1[0]*OM1*TMP0))
    V1.W[1]= denom*S3.W[0]*(-1j*(V2.W[1])+1j*(P1[1]*OM1*TMP0))
    V1.W[2]= denom*S3.W[0]*(-1j*(V2.W[2])+1j*(P1[2]*OM1*TMP0))
    V1.W[3]= denom*S3.W[0]*(-1j*(V2.W[3])+1j*(P1[3]*OM1*TMP0))
    return V1


import cmath
import wavefunctions
def VVS1_2(V2,S3,COUP,M1,W1):

    return VVS1_1(V2,S3,COUP,M1,W1)
import cmath
import wavefunctions
def VVS1_2_1(V2,S3,COUP1,COUP2,M1,W1):
    V1 = VVS1_1(V2,S3,COUP1,M1,W1)
    tmp = VVS2_1(V2,S3,COUP2,M1,W1)
    for i in range(4):
        V1.W[i] += tmp.W[i]
    return V1

import cmath
import wavefunctions
def VVS1_2_2(V2,S3,COUP1,COUP2,M1,W1):
    V1 = VVS1_1(V2,S3,COUP1,M1,W1)
    tmp = VVS2_1(V2,S3,COUP2,M1,W1)
    for i in range(4):
        V1.W[i] += tmp.W[i]
    return V1

