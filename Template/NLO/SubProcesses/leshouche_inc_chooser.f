      subroutine leshouche_inc_chooser()
c For a given nFKSprocess, it fills the c_leshouche_inc common block with the
c leshouche.inc information
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'leshouche_info.inc'
      integer i,j,k
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow)
      common /c_leshouche_inc/idup,mothup,icolup
c
      if (maxproc_used.gt.maxproc) then
         write (*,*) 'ERROR in leshouche_inc_chooser: increase maxproc',
     &        maxproc,maxproc_used
         stop
      endif
      if (maxflow_used.gt.maxflow) then
         write (*,*) 'ERROR in leshouche_inc_chooser: increase maxflow',
     &        maxflow,maxflow_used
         stop
      endif
      do j=1,maxproc_used
         do i=1,nexternal
            IDUP(i,j)=IDUP_D(nFKSprocess,i,j)
            MOTHUP(1,i,j)=MOTHUP_D(nFKSprocess,1,i,j)
            MOTHUP(2,i,j)=MOTHUP_D(nFKSprocess,2,i,j)
         enddo
      enddo
c
      do j=1,maxflow_used
         do i=1,nexternal
            ICOLUP(1,i,j)=ICOLUP_D(nFKSprocess,1,i,j)
            ICOLUP(2,i,j)=ICOLUP_D(nFKSprocess,2,i,j)
         enddo
      enddo
c
      return
      end

