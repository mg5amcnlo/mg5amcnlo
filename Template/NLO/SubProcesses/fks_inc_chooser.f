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
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer i_type,j_type,m_type
      integer particle_type_born(nexternal-1)
      common /c_particle_type_born/particle_type_born
c
      i_fks=fks_i_D(nFKSprocess)
      j_fks=fks_j_D(nFKSprocess)
      do i=1,nexternal
         if (fks_j_from_i_D(nFKSprocess,i,0).ge.0 .and.
     &        fks_j_from_i_D(nFKSprocess,i,0).le.nexternal) then
            do j=0,fks_j_from_i_D(nFKSprocess,i,0)
               fks_j_from_i(i,j)=fks_j_from_i_D(nFKSprocess,i,j)
            enddo
         else
            write (*,*) 'ERROR in fks_inc_chooser'
            stop
         endif
         particle_type(i)=particle_type_D(nFKSprocess,i)
         pdg_type(i)=pdg_type_D(nFKSprocess,i)
      enddo
      do i=1,nexternal
         if (i.lt.min(i_fks,j_fks)) then
            particle_type_born(i)=particle_type(i)
         elseif (i.gt.max(i_fks,j_fks)) then
            particle_type_born(i-1)=particle_type(i)
         elseif (i.eq.min(i_fks,j_fks)) then
            i_type=particle_type(i_fks)
            j_type=particle_type(j_fks)
            if (abs(i_type).eq.abs(j_type)) then
               m_type=8
               if ( (j_fks.le.nincoming .and.
     &              abs(i_type).eq.3 .and. j_type.ne.i_type) .or.
     &              (j_fks.gt.nincoming .and.
     &              abs(i_type).eq.3 .and. j_type.ne.-i_type)) then
                  write(*,*)'Flavour mismatch #1 in fks_inc_chooser',
     &                 i_fks,j_fks,i_type,j_type
                  stop
               endif
            elseif(abs(i_type).eq.3 .and. j_type.eq.8)then
               if(j_fks.le.nincoming)then
                  m_type=-i_type
               else
                  write (*,*) 'Error in fks_inc_chooser: (i,j)=(q,g)'
                  stop
               endif
            elseif(i_type.eq.8 .and. abs(j_type).eq.3)then
               if (j_fks.le.nincoming) then
                  m_type=j_type
               else
                  m_type=j_type
               endif
            elseif(i_type.eq.8.and.j_type.eq.1.and.pdg_type(i_fks).eq.-21)then
            ! dirty trick for LOonly processes without colored legs
               m_type=j_type
            else
               write(*,*)'Flavour mismatch #2 in fks_inc_chooser',
     &              i_type,j_type,m_type
               stop
            endif
            particle_type_born(i)=m_type
         elseif (i.ne.max(i_fks,j_fks)) then
            particle_type_born(i)=particle_type(i)
         endif
      enddo
      return
      end

