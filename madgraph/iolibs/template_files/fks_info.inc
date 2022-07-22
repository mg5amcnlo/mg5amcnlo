      INTEGER IPOS, JPOS
      INTEGER FKS_I_D(%(nconfs)d), FKS_J_D(%(nconfs)d)
      INTEGER EXTRA_CNT_D(%(nconfs)d), ISPLITORDER_BORN_D(%(nconfs)d), ISPLITORDER_CNT_D(%(nconfs)d)
      INTEGER FKS_J_FROM_I_D(%(nconfs)d, NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE_D(%(nconfs)d, NEXTERNAL), PDG_TYPE_D(%(nconfs)d, NEXTERNAL)
      LOGICAL PARTICLE_TAG_D(%(nconfs)d, NEXTERNAL)
      REAL*8 PARTICLE_CHARGE_D(%(nconfs)d, NEXTERNAL)
      LOGICAL SPLIT_TYPE_D(%(nconfs)d, %(nsplitorders)d)
      LOGICAL NEED_COLOR_LINKS_D(%(nconfs)d), NEED_CHARGE_LINKS_D(%(nconfs)d)
      
data fks_i_D / %(fks_i_values)s /
data fks_j_D / %(fks_j_values)s /

data need_color_links_d / %(need_color_links)s /
data need_charge_links_d / %(need_charge_links)s /

%(fks_j_from_i_lines)s

C     Split type: the order of elements is %(splitorders_name)s
%(split_type_lines)s

C     Extra cnt: if the real emission needs an extra cnt (g/a > qqbar), then
C     this is the id of the extra cnt. Otherwise it should be set to 0.
C     If not 0, store also the position in the splitorders array of the 
C     coupling corresponding to the splitting type
data extra_cnt_D / %(extra_cnt_values)s /
data isplitorder_born_D / %(isplitorder_born_values)s /
data isplitorder_cnt_D / %(isplitorder_cnt_values)s /

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

C     
C     Tagged particles:
C     
%(tag_lines)s

