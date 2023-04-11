      subroutine idenparts(iden_part,itree,sprop,forcebw,prwidth)
c
c     Keep track of identical particles to map radiation processes
c     (i.e., not use BW for such processes).
c     Only consider particles that are present in final state as
c     radiating, since need to correctly map conflicting BWs for decays.
c
c     Constants
c
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'maxamps.inc'
      include 'nexternal.inc'
c
c     Arguments
c
      integer iden_part(-nexternal+1:nexternal)
      integer itree(2,-max_branch:-1),iconfig
      integer sprop(maxsproc,-max_branch:-1)  ! Propagator id
      integer forcebw(-max_branch:-1) ! Forced BW, for identical particle conflicts
      double precision prwidth(-nexternal:0)  !Propagator width
c
c     local
c
      integer i,j,it,isp
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'
      integer ipdg(-nexternal+1:nexternal)
c
c     Global
c
      include 'coupl.inc'                     !Mass and width info
      double precision stot
      common/to_stot/stot

      do i=1,nexternal
         ipdg(i) = idup(i,1,1)
      enddo
      do i=-(nexternal-1),nexternal
         iden_part(i)=0
      enddo

      i=1
      do while (i .lt. nexternal-2 .and. itree(1,-i) .ne. 1)
C        Find first non-zero sprop
         do j=1,maxsproc
            if(sprop(j,-i).ne.0) then
               isp=sprop(j,-i)
               exit
            endif
         enddo
         ipdg(-i)=isp
         if (prwidth(-i) .gt. 0d0) then
            if(ipdg(-i).eq.ipdg(itree(1,-i)).and.itree(1,-i).gt.0.or.
     $         ipdg(-i).eq.ipdg(itree(2,-i)).and.itree(2,-i).gt.0) then
               iden_part(-i) = ipdg(-i)
            else if(ipdg(-i).eq.ipdg(itree(1,-i)).and.
     $              iden_part(itree(1,-i)).ne.0.or.
     $         ipdg(-i).eq.ipdg(itree(2,-i)).and.
     $              iden_part(itree(2,-i)).ne.0) then
               iden_part(-i) = ipdg(-i)
            endif
         endif
         i=i+1
      enddo
      end
