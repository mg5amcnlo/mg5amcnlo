import cmath
import wavefunctions
def RFSC1_0(F2,R1,S3,COUP):
    flv_index1 = F1.flavor
    flv_index2 = F2.flavor
    if flv_index1 != -1 and flv_index2 != -1 and flv_index1 != flv_index2:
        return 0j
    TMP0 = (F2.W[2]*(-R1.W[0]+R1.W[5]+R1.W[12]+1j*(R1.W[9]))-F2.W[3]*(R1.W[1]+R1.W[13]-R1.W[4]+1j*(R1.W[8])))
    vertex = COUP*-1j * TMP0*S3.W[0]
    return vertex


