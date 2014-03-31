      INTEGER IPOS, JPOS
      INTEGER FKS_I_D(6), FKS_J_D(6)
      INTEGER FKS_J_FROM_I_D(6, NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE_D(6, NEXTERNAL), PDG_TYPE_D(6, NEXTERNAL)

      DATA FKS_I_D / 5, 5, 5, 5, 5, 5 /
      DATA FKS_J_D / 1, 2, 3, 4, 1, 2 /

      DATA (FKS_J_FROM_I_D(1, 5, JPOS), JPOS = 0, 4)  / 4, 1, 2, 3, 4 /

      DATA (FKS_J_FROM_I_D(2, 5, JPOS), JPOS = 0, 4)  / 4, 1, 2, 3, 4 /

      DATA (FKS_J_FROM_I_D(3, 5, JPOS), JPOS = 0, 4)  / 4, 1, 2, 3, 4 /

      DATA (FKS_J_FROM_I_D(4, 5, JPOS), JPOS = 0, 4)  / 4, 1, 2, 3, 4 /

      DATA (FKS_J_FROM_I_D(5, 5, JPOS), JPOS = 0, 2)  / 2, 1, 2 /

      DATA (FKS_J_FROM_I_D(6, 5, JPOS), JPOS = 0, 2)  / 2, 1, 2 /


C     
C     Particle type:
C     octet = 8, triplet = 3, singlet = 1
      DATA (PARTICLE_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / -3, 3, 3, 
     $ -3, 8 /
      DATA (PARTICLE_TYPE_D(2, IPOS), IPOS=1, NEXTERNAL) / -3, 3, 3, 
     $ -3, 8 /
      DATA (PARTICLE_TYPE_D(3, IPOS), IPOS=1, NEXTERNAL) / -3, 3, 3, 
     $ -3, 8 /
      DATA (PARTICLE_TYPE_D(4, IPOS), IPOS=1, NEXTERNAL) / -3, 3, 3, 
     $ -3, 8 /
      DATA (PARTICLE_TYPE_D(5, IPOS), IPOS=1, NEXTERNAL) / 8, 3, 3, 
     $ -3, 3 /
      DATA (PARTICLE_TYPE_D(6, IPOS), IPOS=1, NEXTERNAL) / -3, 8, 3, 
     $ -3, -3 /

C     
C     Particle type according to PDG:
C     
      DATA (PDG_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / -2, 2, 6, 
     $ -6, 21 /
      DATA (PDG_TYPE_D(2, IPOS), IPOS=1, NEXTERNAL) / -2, 2, 6, 
     $ -6, 21 /
      DATA (PDG_TYPE_D(3, IPOS), IPOS=1, NEXTERNAL) / -2, 2, 6, 
     $ -6, 21 /
      DATA (PDG_TYPE_D(4, IPOS), IPOS=1, NEXTERNAL) / -2, 2, 6, 
     $ -6, 21 /
      DATA (PDG_TYPE_D(5, IPOS), IPOS=1, NEXTERNAL) / 21, 2, 6, -6, 2 /
      DATA (PDG_TYPE_D(6, IPOS), IPOS=1, NEXTERNAL) / -2, 21, 6, -6, 
     $ -2 /


