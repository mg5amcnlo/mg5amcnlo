import cmath
import wavefunctions
def RFSC1_1(R1,S3,COUP,M2,W2):
    F2 = wavefunctions.WaveFunction(size=6)
    F2.momenta[0] = +R1.momenta[0]+S3.momenta[0]
    F2.momenta[1] = +R1.momenta[1]+S3.momenta[1]
    F2.momenta[2] = +R1.momenta[2]+S3.momenta[2]
    F2.momenta[3] = +R1.momenta[3]+S3.momenta[3]
    P2 = [-F2.momenta[j] for j in range(4)]
    denom = COUP/(P2[0]**2-P2[1]**2-P2[2]**2-P2[3]**2 - M2 * (M2 -1j* W2))
    F2.W[0]= denom*1j * S3.W[0]*(P2[0]*(-1)*(-R1.W[0]+R1.W[5]+R1.W[12]+1j*(R1.W[9]))+(P2[1]*(R1.W[1]+R1.W[13]-R1.W[4]+1j*(R1.W[8]))+(P2[2]*(+1j*(R1.W[1]+R1.W[13])-1j*(R1.W[4])-R1.W[8])-P2[3]*(-R1.W[0]+R1.W[5]+R1.W[12]+1j*(R1.W[9])))))
    F2.W[1]= denom*1j * S3.W[0]*(P2[0]*(R1.W[1]+R1.W[13]-R1.W[4]+1j*(R1.W[8]))+(P2[1]*(-1)*(-R1.W[0]+R1.W[5]+R1.W[12]+1j*(R1.W[9]))+(P2[2]*(-1j*(R1.W[0])+1j*(R1.W[5]+R1.W[12])-R1.W[9])-P2[3]*(R1.W[1]+R1.W[13]-R1.W[4]+1j*(R1.W[8])))))
    F2.W[2]= denom*1j * M2*S3.W[0]*(-R1.W[0]+R1.W[5]+R1.W[12]+1j*(R1.W[9]))
    F2.W[3]= denom*-1j * M2*S3.W[0]*(R1.W[1]+R1.W[13]-R1.W[4]+1j*(R1.W[8]))
    return F2


