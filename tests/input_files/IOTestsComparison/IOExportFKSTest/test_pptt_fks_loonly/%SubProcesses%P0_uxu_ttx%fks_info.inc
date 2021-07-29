      INTEGER IPOS, JPOS
      INTEGER FKS_I_D(1), FKS_J_D(1)
      INTEGER EXTRA_CNT_D(1), ISPLITORDER_BORN_D(1),
     $  ISPLITORDER_CNT_D(1)
      INTEGER FKS_J_FROM_I_D(1, NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE_D(1, NEXTERNAL), PDG_TYPE_D(1, NEXTERNAL)
      REAL*8 PARTICLE_CHARGE_D(1, NEXTERNAL)
      LOGICAL SPLIT_TYPE_D(1, 2)
      LOGICAL NEED_COLOR_LINKS_D(1), NEED_CHARGE_LINKS_D(1)

      DATA FKS_I_D / 5 /
      DATA FKS_J_D / 4 /

      DATA NEED_COLOR_LINKS_D / .TRUE. /
      DATA NEED_CHARGE_LINKS_D / .TRUE. /

      DATA (FKS_J_FROM_I_D(1, 5, JPOS), JPOS = 0, 1)  / 1, 4 /

C     Split type: the order of elements is QCD, QED
      DATA (SPLIT_TYPE_D (1, IPOS), IPOS=1, 2) / .FALSE., .FALSE. /

C     Extra cnt: if the real emission needs an extra cnt (g/a >
C      qqbar), then
C     this is the id of the extra cnt. Otherwise it should be set to 0.
C     If not 0, store also the position in the splitorders array of
C      the 
C     coupling corresponding to the splitting type
      DATA EXTRA_CNT_D / 0 /
      DATA ISPLITORDER_BORN_D / 0 /
      DATA ISPLITORDER_CNT_D / 0 /

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
     $ -0.666666666666667D0,   0.666666666666667D0,  
     $  0.666666666666667D0,  -0.666666666666667D0,  
     $  0.000000000000000D0 /

