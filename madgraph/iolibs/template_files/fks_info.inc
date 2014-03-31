      INTEGER IPOS, JPOS
      INTEGER FKS_I_D(%(nconfs)d), FKS_J_D(%(nconfs)d)
      INTEGER FKS_J_FROM_I_D(%(nconfs)d, NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE_D(%(nconfs)d, NEXTERNAL), PDG_TYPE_D(%(nconfs)d, NEXTERNAL)
      REAL*8 PARTICLE_CHARGE_D(%(nconfs)d, NEXTERNAL)
      LOGICAL SPLIT_TYPE_D(%(nconfs)d, %(nsplitorders)d)
      
data fks_i_D / %(fks_i_values)s /
data fks_j_D / %(fks_j_values)s /

%(fks_j_from_i_lines)s

C     Split type: the order of elements is %(splitorders_name)s
%(split_type_lines)s

C     
C     Particle type:
C     octet = 8, triplet = 3, singlet = 1
%(col_lines)s

C     
C     Particle type according to PDG:
C     
%(pdg_lines)s

C
C     Particle charge:
C     charge is set 0. with QCD corrections, which is irrelevant
%(charge_lines)s
