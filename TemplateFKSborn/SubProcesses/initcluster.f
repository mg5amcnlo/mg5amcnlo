      subroutine initcluster()

      implicit none

      include 'message.inc'
      include 'run.inc'
      include 'nexternal.inc'
      include 'cluster.inc'
C
C     SPECIAL CUTS
C
      REAL*8 XPTJ,XPTB,XPTA,XPTL
      REAL*8 XETAMIN,XQCUT,deltaeta
      COMMON /TO_SPECXPT/XPTJ,XPTB,XPTA,XPTL,XETAMIN,XQCUT,deltaeta

      integer i,j
      logical filmap, cluster
      external filmap, cluster

c     
c     check whether y_cut is used -> set scale to y_cut*S
c

c      if (ickkw.le.0) return
      if (ickkw.le.0.and.xqcut.le.0d0) return

      if(ickkw.eq.2.and.xqcut.le.0d0)then
        write(*,*)'Must set qcut > 0 for ickkw = 2'
        write(*,*)'Exiting...'
        stop
      endif

c      if(xqcut.gt.0d0)then
      if(ickkw.eq.2)then
        scale = xqcut
        q2fact(1) = scale**2    ! fact scale**2 for pdf1
        q2fact(2) = scale**2    ! fact scale**2 for pdf2
        fixed_ren_scale=.true.
        fixed_fac_scale=.true.
      endif
c   
c     initialize clustering map
c         
      if (.not.filmap()) then
        write(*,*) 'cuts.f: cluster map initialization failed'
        stop
      endif
      if (btest(mlevel,3)) then
        do i=1,ishft(1,nexternal+1)
          write(*,*) 'prop ',i,' in'
          do j=1,id_cl(i,0)
            write(*,*) '  graph ',id_cl(i,j)
          enddo
        enddo
        write(*,*)'ok'
      endif
      igscl(0)=0
       
      RETURN
      END

