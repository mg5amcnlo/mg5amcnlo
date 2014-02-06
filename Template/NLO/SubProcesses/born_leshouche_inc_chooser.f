      subroutine born_leshouche_inc_chooser()
c For a given nborn, it fills the c_leshouche_inc common block with the
c leshouche.inc information
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'born_leshouche_info.inc'
      integer i,j,k
      INTEGER NBORN
      COMMON/C_NBORN/NBORN
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow)
      common /c_born_leshouche_inc/idup,mothup,icolup
c
      if (maxprocb_used.gt.maxproc) then
         write (*,*) 
     &    'ERROR in born_leshouche_inc_chooser: increase maxproc',
     &        maxproc,maxprocb_used
         stop
      endif
      if (maxflowb_used.gt.maxflow) then
         write (*,*) 
     &    'ERROR in born_leshouche_inc_chooser: increase maxflow',
     &        maxflow,maxflowb_used
         stop
      endif
      do j=1,maxprocb_used
         do i=1,nexternal-1
            IDUP(i,j)=IDUP_B(nborn,i,j)
            MOTHUP(1,i,j)=MOTHUP_B(nborn,1,i,j)
            MOTHUP(2,i,j)=MOTHUP_B(nborn,2,i,j)
         enddo
      enddo
c
      do j=1,maxflowb_used
         do i=1,nexternal-1
            ICOLUP(1,i,j)=ICOLUP_B(nborn,1,i,j)
            ICOLUP(2,i,j)=ICOLUP_B(nborn,2,i,j)
         enddo
      enddo
c
      return
      end

