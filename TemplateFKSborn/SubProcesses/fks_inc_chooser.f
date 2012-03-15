      subroutine fks_inc_chooser()
c For a given nFKSprocess, it fills the c_fks_inc common block with the
c fks.inc information
      implicit none
      include 'nexternal.inc'
      include 'fks_info.inc'
      integer i,j
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_typ,pdg_type
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
c
      i_fks=fks_i_D(nFKSprocess)
      j_fks=fks_j_D(nFKSprocess)
      do i=1,nexternal
         if (i.eq.fks_i) then
            do j=0,fks_j_from_i_D(nFKSprocess,i,0)
               fks_j_from_i(i,j)=fks_j_from_i_D(nFKSprocess,i,j)
            enddo
         endif
         particle_type(i)=particle_type_D(nFKSprocess,i)
         pdg_type(i)=pdg_type_D(nFKSprocess,i)
      enddo
c
      return
      end

