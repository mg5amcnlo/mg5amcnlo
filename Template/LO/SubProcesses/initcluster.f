      subroutine initcluster()

      implicit none

      include 'message.inc'
      include 'maxparticles.inc'
      include 'run.inc'
      include 'nexternal.inc'
      include 'maxamps.inc'
      include 'cluster.inc'
C
C     SPECIAL CUTS
C
      real*8 xptj,xptb,xpta,xptl,xmtc
      real*8 xetamin,xqcut,deltaeta
      common /to_specxpt/xptj,xptb,xpta,xptl,xmtc,xetamin,xqcut,deltaeta

      integer i,j,iproc
      logical filmap, cluster
      external filmap, cluster

c     
c     check whether y_cut is used -> set scale to y_cut*S
c

c      if (ickkw.le.0) return
      if (ickkw.le.0.and.xqcut.le.0d0.and.fixed_ren_scale.and.fixed_fac_scale) return

c      if(ickkw.eq.2.and.xqcut.le.0d0)then
c        write(*,*)'Must set qcut > 0 for ickkw = 2'
c        write(*,*)'Exiting...'
c        stop
c      endif

c      if(xqcut.gt.0d0)then
c      if(ickkw.eq.2)then
c        scale = xqcut
c        q2fact(1) = scale**2    ! fact scale**2 for pdf1
c        q2fact(2) = scale**2    ! fact scale**2 for pdf2
c        fixed_ren_scale=.true.
c        fixed_fac_scale=.true.
c      endif
c   
c     initialize clustering map
c         
      if (.not.filmap()) then
        write(*,*) 'cuts.f: cluster map initialization failed'
        stop
      endif
      if (btest(mlevel,3)) then
        do iproc=1,maxsproc
           write(*,*)'for proc ',iproc
           do i=1,n_max_cl
              write(*,*) 'prop ',i,' in'
              do j=1,id_cl(iproc,i,0)
                 write(*,*) '  graph ',id_cl(iproc,i,j)
              enddo
           enddo
           write(*,*)'ok'
        enddo
      endif
      igraphs(0)=0
       
      RETURN
      END

