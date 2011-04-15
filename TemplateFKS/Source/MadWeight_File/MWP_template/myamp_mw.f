      logical function cut_bw(p)
c*****************************************************************************
c     Approximates matrix element by propagators
c*****************************************************************************
      implicit none
c
c     Constants
c     
      include 'genps.inc'
      include 'nexternal.inc'
      double precision   zero
      parameter (zero = 0d0)
      include 'run.inc'
c
c     Arguments
c
      double precision p(0:3,nexternal)
c
c     Local
c
      double precision xp(0:3,-nexternal:nexternal)
      double precision mpole(-nexternal:0),shat,tsgn
      integer i,j,iconfig

      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      logical first_time, onshell
      double precision xmass
      integer nbw

      integer ida(2),idenpart
c
c     Global
c
      integer    maxflow
      parameter (maxflow=999)
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      common/to_sprop/sprop,tprid
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config

      integer        lbw(0:nexternal)  !Use of B.W.
      common /to_BW/ lbw

      logical             OnBW(-nexternal:0)     !Set if event is on B.W.
      common/to_BWEvents/ OnBW
      
      include 'coupl.inc'

      integer idup(nexternal,maxproc)
      integer mothup(2,nexternal,maxproc)
      integer icolup(2,nexternal,maxflow)
      include 'leshouche.inc'

      logical gForceBW(-max_branch:-1,lmaxconfigs)  ! Forced BW
      include 'decayBW.inc'
c
c     External
c
      double precision dot

      save pmass,pwidth,pow
      data first_time /.true./
c-----
c  Begin Code
c-----      
      cut_bw = .false.    !Default is we passed the cut
      iconfig = this_config
      if (first_time) then
         include 'props.inc'
         nbw = 0
         tsgn = 1d0
         do i=-1,-(nexternal-3),-1
            if (iforest(1,i,iconfig) .eq. 1) tsgn=-1d0
            if (pwidth(i,iconfig) .gt. 0 .and. tsgn .eq. 1d0) then
               nbw=nbw+1
               if (lbw(nbw) .eq. 1) then
                  write(*,*) 'Requiring BW ',i,nbw
               elseif(lbw(nbw) .eq. 2) then
                  write(*,*) 'Excluding BW ',i,nbw
               else
                  write(*,*) 'No cut BW ',i,nbw
               endif
            endif
         enddo
         first_time=.false.
      endif

      do i=1,nexternal
         mpole(-i)=0d0
         do j=0,3
            xp(j,i)=p(j,i)
         enddo
      enddo
      nbw = 0
      tsgn    = +1d0
      do i=-1,-(nexternal-3),-1              !Loop over propagators
         onbw(i) = .false.
         if (iforest(1,i,iconfig) .eq. 1) tsgn=-1d0
         do j=0,3
            xp(j,i) = xp(j,iforest(1,i,iconfig))
     $           +tsgn*xp(j,iforest(2,i,iconfig))
         enddo
         if (tsgn .gt. 0d0 .and. pwidth(i,iconfig) .gt. 0d0 ) then !This is B.W.
            nbw = nbw+1
c            write(*,*) 'Checking BW',nbw
            xmass = sqrt(dot(xp(0,i),xp(0,i)))
c            write(*,*) 'xmass',xmass,pmass(i,iconfig)
            onshell = (abs(xmass - pmass(i,iconfig)) .lt.
     $           bwcutoff*pwidth(i,iconfig))

c
c           Here we set if the BW is "on-shell" for LesHouches
c
            if(onshell)then
c           Only allow onshell if no "decay" to identical particle
              OnBW(i) = .true.
              idenpart=0
              do j=1,2
                ida(j)=iforest(j,i,iconfig)
                if((ida(j).lt.0.and.sprop(i,iconfig).eq.sprop(ida(j),iconfig))
     $             .or.(ida(j).gt.0.and.sprop(i,iconfig).eq.IDUP(ida(j),1)))
     $             idenpart=ida(j)
              enddo
c           Always remove if daughter final-state
              if(idenpart.gt.0) then
                OnBW(i)=.false.
c           Else remove if daughter forced to be onshell
              elseif(idenpart.lt.0.and.gForceBW(idenpart, iconfig)) then
                OnBW(i)=.false.
c           Else remove daughter if forced to be onshell
              elseif(idenpart.lt.0.and.gForceBW(i, iconfig)) then
                OnBW(idenpart)=.false.
c           Else remove either this resonance or daughter, which is closer to mass shell
              elseif(idenpart.lt.0.and.abs(xmass-pmass(i,iconfig)).gt.
     $             abs(sqrt(dot(xp(0,idenpart),xp(0,idenpart)))-
     $             pmass(i,iconfig))) then
                OnBW(i)=.false.
c           Else remove OnBW for daughter
              else if(idenpart.lt.0) then
                OnBW(idenpart)=.false.
              endif
            endif
            if (onshell .and. (lbw(nbw).eq. 2) ) cut_bw=.true.
            if (.not. onshell .and. (lbw(nbw).eq. 1)) cut_bw=.true.
c            write(*,*) nbw,xmass,onshell,lbw(nbw),cut_bw
         endif

      enddo
      end

