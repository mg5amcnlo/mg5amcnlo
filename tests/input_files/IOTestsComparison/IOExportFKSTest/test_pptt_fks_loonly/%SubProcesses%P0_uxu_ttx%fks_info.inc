      INTEGER IPOS, JPOS
      INTEGER FKS_I_D(1), FKS_J_D(1)
      INTEGER FKS_J_FROM_I_D(1, NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE_D(1, NEXTERNAL), PDG_TYPE_D(1, NEXTERNAL)
      REAL*8 PARTICLE_CHARGE_D(1, NEXTERNAL)

      DATA FKS_I_D / 5 /
      DATA FKS_J_D / 4 /

      DATA (FKS_J_FROM_I_D(1, 5, JPOS), JPOS = 0, 1)  / 1, 4 /

C     
C     Particle type:
C     octet = 8, triplet = 3, singlet = 1
      DATA (PARTICLE_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / -3, 3, 3, 
     $ -3, 8 /

C     
C     Particle type according to PDG:
C     
      DATA (PDG_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / -2, 2, 6, -6, 
     $ -21 /

C     
C     Particle charge:
C     charge is set 0. with QCD corrections, which is irrelevant
      DATA (PARTICLE_CHARGE_D(1, IPOS), IPOS=1, NEXTERNAL) /  
     $  0.000000000000000D0,   0.000000000000000D0,  
     $  0.000000000000000D0,   0.000000000000000D0,  
     $  0.000000000000000D0 /

