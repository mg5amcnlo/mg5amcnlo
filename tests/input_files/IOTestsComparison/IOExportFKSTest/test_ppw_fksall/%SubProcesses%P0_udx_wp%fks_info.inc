      INTEGER IPOS, JPOS
      INTEGER FKS_I_D(4), FKS_J_D(4)
      INTEGER EXTRA_CNT_D(4), ISPLITORDER_BORN_D(4),
     $  ISPLITORDER_CNT_D(4)
      INTEGER FKS_J_FROM_I_D(4, NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE_D(4, NEXTERNAL), PDG_TYPE_D(4, NEXTERNAL)
      REAL*8 PARTICLE_CHARGE_D(4, NEXTERNAL)
      LOGICAL SPLIT_TYPE_D(4, 2)
      LOGICAL NEED_COLOR_LINKS_D(4), NEED_CHARGE_LINKS_D(4)

      DATA FKS_I_D / 4, 4, 4, 4 /
      DATA FKS_J_D / 1, 2, 1, 2 /

      DATA NEED_COLOR_LINKS_D / .TRUE., .TRUE., .FALSE., .FALSE. /
      DATA NEED_CHARGE_LINKS_D / .FALSE., .FALSE., .FALSE., .FALSE. /

      DATA (FKS_J_FROM_I_D(1, 4, JPOS), JPOS = 0, 2)  / 2, 1, 2 /

      DATA (FKS_J_FROM_I_D(2, 4, JPOS), JPOS = 0, 2)  / 2, 1, 2 /

      DATA (FKS_J_FROM_I_D(3, 4, JPOS), JPOS = 0, 1)  / 1, 1 /

      DATA (FKS_J_FROM_I_D(4, 4, JPOS), JPOS = 0, 1)  / 1, 2 /


C     Split type: the order of elements is QCD, QED
      DATA (SPLIT_TYPE_D (1, IPOS), IPOS=1, 2) / .TRUE., .FALSE. /
      DATA (SPLIT_TYPE_D (2, IPOS), IPOS=1, 2) / .TRUE., .FALSE. /
      DATA (SPLIT_TYPE_D (3, IPOS), IPOS=1, 2) / .TRUE., .FALSE. /
      DATA (SPLIT_TYPE_D (4, IPOS), IPOS=1, 2) / .TRUE., .FALSE. /

C     Extra cnt: if the real emission needs an extra cnt (g/a >
C      qqbar), then
C     this is the id of the extra cnt. Otherwise it should be set to 0.
C     If not 0, store also the position in the splitorders array of
C      the 
C     coupling corresponding to the splitting type
      DATA EXTRA_CNT_D / 0, 0, 0, 0 /
      DATA ISPLITORDER_BORN_D / 0, 0, 0, 0 /
      DATA ISPLITORDER_CNT_D / 0, 0, 0, 0 /

C     
C     Particle type:
C     octet = 8, triplet = 3, singlet = 1
      DATA (PARTICLE_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / 3, -3, 1, 8
     $  /
      DATA (PARTICLE_TYPE_D(2, IPOS), IPOS=1, NEXTERNAL) / 3, -3, 1, 8
     $  /
      DATA (PARTICLE_TYPE_D(3, IPOS), IPOS=1, NEXTERNAL) / 8, -3, 1, 
     $ -3 /
      DATA (PARTICLE_TYPE_D(4, IPOS), IPOS=1, NEXTERNAL) / 3, 8, 1, 3 /

C     
C     Particle type according to PDG:
C     
      DATA (PDG_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / 2, -1, 24, 21 /
      DATA (PDG_TYPE_D(2, IPOS), IPOS=1, NEXTERNAL) / 2, -1, 24, 21 /
      DATA (PDG_TYPE_D(3, IPOS), IPOS=1, NEXTERNAL) / 21, -1, 24, -2 /
      DATA (PDG_TYPE_D(4, IPOS), IPOS=1, NEXTERNAL) / 2, 21, 24, 1 /

C     
C     Particle charge:
C     charge is set 0. with QCD corrections, which is irrelevant
      DATA (PARTICLE_CHARGE_D(1, IPOS), IPOS=1, NEXTERNAL) /  
     $  0.666666666666667D0,   0.333333333333333D0,  
     $  1.000000000000000D0,   0.000000000000000D0 /
      DATA (PARTICLE_CHARGE_D(2, IPOS), IPOS=1, NEXTERNAL) /  
     $  0.666666666666667D0,   0.333333333333333D0,  
     $  1.000000000000000D0,   0.000000000000000D0 /
      DATA (PARTICLE_CHARGE_D(3, IPOS), IPOS=1, NEXTERNAL) /  
     $  0.000000000000000D0,   0.333333333333333D0,  
     $  1.000000000000000D0,  -0.666666666666667D0 /
      DATA (PARTICLE_CHARGE_D(4, IPOS), IPOS=1, NEXTERNAL) /  
     $  0.666666666666667D0,   0.000000000000000D0,  
     $  1.000000000000000D0,  -0.333333333333333D0 /

