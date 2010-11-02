      double precision function testamp(p)
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
c
c     Arguments
c
      double precision p(0:3,nexternal)
c      integer iconfig
c
c     Local
c
      double precision xp(0:3,-nexternal:nexternal)
      double precision mpole(-nexternal:0),shat,tsgn
      integer i,j,iconfig

      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      logical first_time
c
c     Global
c
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config
      
      include 'coupl.inc'
c
c     External
c
      double precision dot

      save pmass,pwidth,pow
      data first_time /.true./
c-----
c  Begin Code
c-----      
      iconfig = this_config
      if (first_time) then
c         include 'props.inc'
         first_time=.false.
      endif

      do i=1,nexternal
         mpole(-i)=0d0
         do j=0,3
            xp(j,i)=p(j,i)
         enddo
      enddo
c      mpole(-3) = 174**2
c      shat = dot(p(0,1),p(0,2))/(1800)**2
      shat = dot(p(0,1),p(0,2))/(10)**2
c      shat = 1d0
      testamp = 1d0
      tsgn    = +1d0
      do i=-1,-(nexternal-3),-1              !Find all the propagotors
         if (iforest(1,i,iconfig) .eq. 1) tsgn=-1d0
         do j=0,3
            xp(j,i) = xp(j,iforest(1,i,iconfig))
     $           +tsgn*xp(j,iforest(2,i,iconfig))
         enddo
         if (pwidth(i,iconfig) .ne. 0d0 .and. .false.) then
            testamp=testamp/((dot(xp(0,i),xp(0,i))
     $                        -pmass(i,iconfig)**2)**2
     $         -(pmass(i,iconfig)*pwidth(i,iconfig))**2)
         else
            testamp = testamp/((dot(xp(0,i),xp(0,i)) -
     $                          pmass(i,iconfig)**2)
     $                          **(pow(i,iconfig)))
         endif
        testamp=testamp*shat**(pow(i,iconfig))
c        write(*,*) i,iconfig,pow(i,iconfig),pmass(i,iconfig)
      enddo
c      testamp = 1d0/dot(xp(0,-1),xp(0,-1))
      testamp=abs(testamp)
c      testamp = 1d0
      end

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
      include 'maxamps.inc'
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

      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
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
     $             .or.(ida(j).gt.0.and.sprop(i,iconfig).eq.IDUP(ida(j),1,1)))
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


      subroutine set_peaks
c*****************************************************************************
c     Attempts to determine peaks for this configuration
c*****************************************************************************
      implicit none
c
c     Constants
c     
      include 'genps.inc'
      include 'nexternal.inc'
      double precision   zero
      parameter (zero = 0d0)
c
c     Arguments
c
c
c     Local
c
      double precision  xm(-nexternal:nexternal)
      double precision  xe(-nexternal:nexternal)
      double precision tsgn, xo, a
      double precision x1,x2,xk(nexternal)
      double precision dr,mtot,etot,stot,xqfact
      integer i, iconfig, l1, l2, j, nt, nbw

      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)

c
c     Global
c
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest

      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config

      real*8         emass(nexternal)
      common/to_mass/emass

      include 'run.inc'

      double precision etmin(nincoming+1:nexternal),etamax(nincoming+1:nexternal)
      double precision emin(nincoming+1:nexternal)
      double precision                    r2min(nincoming+1:nexternal,nincoming+1:nexternal)
      double precision s_min(nexternal,nexternal)
      common/to_cuts/  etmin, emin, etamax, r2min, s_min

      double precision xqcutij(nexternal,nexternal),xqcuti(nexternal)
      common/to_xqcuts/xqcutij,xqcuti

      double precision      spole(maxinvar),swidth(maxinvar),bwjac
      common/to_brietwigner/spole          ,swidth          ,bwjac

      integer        lbw(0:nexternal)  !Use of B.W.
      common /to_BW/ lbw
      
      include 'coupl.inc'

c
c     External
c

c-----
c  Begin Code
c-----      
      include 'props.inc'
      stot = 4d0*ebeam(1)*ebeam(2)
c      etmin = 10
      nt = 0
      iconfig = this_config
      mtot = 0d0
      etot = 0d0   !Total energy needed
      xqfact=1d0
      if(ickkw.eq.2.or.ktscheme.eq.2) xqfact=0.3d0
      do i=nincoming+1,nexternal  !assumes 2 incoming
         xm(i)=emass(i)
c-fax
         xe(i)=max(emass(i),max(etmin(i),0d0))
         xe(i)=max(xe(i),max(emin(i),0d0))
c-JA 1/2009: Set grid also based on xqcut
         xe(i)=max(xe(i),xqfact*xqcuti(i))
         xk(i)= 0d0
         etot = etot+xe(i)
         mtot=mtot+xm(i)         
      enddo
      tsgn    = +1d0
      nbw = 0
      do i=-1,-(nexternal-3),-1              !Find all the propagotors
         if (iforest(1,i,iconfig) .eq. 1) tsgn=-1d0
         if (tsgn .eq. 1d0) then                         !s channel
            xm(i) = xm(iforest(1,i,iconfig))+xm(iforest(2,i,iconfig))
            xe(i) = xe(iforest(1,i,iconfig))+xe(iforest(2,i,iconfig))
            mtot = mtot - xm(i)
            etot = etot - xe(i)
            if (iforest(1,i,iconfig) .gt. 0
     &           .and. iforest(2,i,iconfig) .gt. 0) then
c-JA 1/2009: Set deltaR cuts here together with s_min cuts
              l1 = iforest(1,i,iconfig)
              l2 = iforest(2,i,iconfig)
              xm(i)=max(xm(i),sqrt(max(s_min(l1,l2),0d0)))
              dr = max(r2min(l1,l2)*dabs(r2min(l1,l2)),0d0)*0.8d0
              xm(i)=max(xm(i),
     &           sqrt(max(etmin(l2),0d0)*max(etmin(l1),0d0)*dr))
c-JA 1/2009: Set grid also based on xqcut
              xm(i)=max(xm(i),sqrt(max(xqcutij(l1,l2),0d0)))
              xe(i)=max(xe(i),xm(i))
            endif
c            write(*,*) 'iconfig,i',iconfig,i
c            write(*,*) pwidth(i,iconfig),pmass(i,iconfig)
            if (pwidth(i,iconfig) .gt. 0) nbw=nbw+1
            if (pwidth(i,iconfig) .gt. 0 .and. lbw(nbw) .le. 1) then         !B.W.
c               nbw = nbw +1

               if (i .eq. -(nexternal-(nincoming+1))) then  !This is s-hat
                  j = 3*(nexternal-2)-4+1    !set i to ndim+1
c-----
c tjs 11/2008 if require BW then force even if worried about energy
c----
                  if(pmass(i,iconfig).ge.xe(i) .or. lbw(nbw).eq.1) then
                     write(*,*) 'Setting PDF BW',j,nbw,pmass(i,iconfig)
                     spole(j)=pmass(i,iconfig)*pmass(i,iconfig)/stot
                     swidth(j) = pwidth(i,iconfig)*pmass(i,iconfig)/stot
                     xm(i) = pmass(i,iconfig)
                  else
                     spole(j)=0d0
                     swidth(j) = 0d0
                  endif
               else
                  write(*,*) 'Setting BW',i,nbw,pmass(i,iconfig)
                  spole(-i)=pmass(i,iconfig)*pmass(i,iconfig)/stot
                  swidth(-i) = pwidth(i,iconfig)*pmass(i,iconfig)/stot
                  xm(i) = pmass(i,iconfig)
c RF & TJS, should start from final state particle masses, not only at resonance.
c Therefore remove the next line.
c                  xe(i) = max(xe(i),xm(i))
               endif
            else                                  !1/x^pow
c-JA 1/2009: Comment out this whole section, since it only sets (wrong) xm(i)
c               if (xm(i) - pmass(i,iconfig) .le. 0d0) then !Can hit pole
cc                  write(*,*) 'Setting new min',i,xm(i),pmass(i,iconfig)
c                  l1 = iforest(1,i,iconfig)                  !need dr cut
c                  l2 = iforest(2,i,iconfig)
c                  if (l2 .lt. l1) then
c                     j = l1
c                     l1 = l2
c                     l2 = j
c                  endif
c                  dr = 0
cc-fax
c                  if (l1 .gt. 0) 
c     &  dr = max(r2min(l2,l1)*dabs(r2min(l2,l1)),0d0) !dr only for external
cc                  write(*,*) 'using r2min',l2,l1,sqrt(dr)
c                  dr = dr*.8d0                        !0.8 to hit peak hard
c                  xo = 0.5d0*pmass(i,iconfig)**2      !0.5 to hit peak hard
cc-fax
c                  if (dr .gt. 0d0) 
c     &            xo = max(xo,max(etmin(l2),0d0)*max(etmin(l1),0d0)*dr)
cc                  write(*,*) 'Got dr',i,l1,l2,dr
cc------
cctjs 11/2008  if explicitly missing pole, don't want to include mass in xm
cc-----
c                 if (pwidth(i,iconfig) .le. 0) then
cc                     write(*,*) "Including mass",i,pmass(i,iconfig)
c                    xo = xo+pmass(i,iconfig)**2
c                 else
cc                     write(*,*) "Skipping mass", i,pmass(i,iconfig),sqrt(xo)
c                 endif
cc                  write(*,*) 'Setting xm',i,xm(i),sqrt(xo)
c                  xm(i) = sqrt(xo)    !Reset xm to realistic minimum
c                  xo = xo/stot
cc                  xo = sqrt(pmass(i,iconfig)**2+ pmass(i,iconfig)**2)
cc-fax
cc                  xo = pmass(i,iconfig)+max(etmin,0d0)
c               else
c                  write(*,*) 'Using xm',i,xm(i)
c                  xo = xm(i)**2/stot
c               endif
              xo = xm(i)**2/stot
              a=pmass(i,iconfig)**2/stot
c               call setgrid(-i,xo,a,pow(i,iconfig))
c               write(*,*) 'Enter minimum for ',-i, xo
c               read(*,*) xo


               if (pwidth(i,iconfig) .eq. 0) call setgrid(-i,xo,a,1)
               if (pwidth(i,iconfig) .gt. 0) then
                  write(*,*) 'Using flat grid for BW',i,nbw,
     $                 pmass(i,iconfig)
               endif
            endif
c            xe(i) = max(xm(i),xe(i))               
            etot = etot+xe(i)
            mtot=mtot+max(xm(i),xm(i))
c            write(*,*) 'New mtot',i,mtot,xm(i)
         else                                        !t channel
c
c     Check closest to p1
c
            nt = nt+1
            l2 = iforest(2,i,iconfig) !need dr cut
            x1 = 0            
c-fax
c-JA 1/2009: Set grid also based on xqcut
            if (l2 .gt. 0) x1 = max(etmin(l2),max(xqfact*xqcuti(l2),0d0))
            x1 = max(x1, xm(l2)/1d0)
            if (nt .gt. 1) x1 = max(x1,xk(nt-1))
            xk(nt)=x1
c            write(*,*) 'Using 1',l2,x1

c
c     Check closest to p2
c
            j = i-1
            l2 = iforest(2,j,iconfig)
            x2 = 0
c-JA 1/2009: Set grid also based on xqcut
            if (l2 .gt. 0) x2 = max(etmin(l2),max(xqfact*xqcuti(l2),0d0))
c            if (l2 .gt. 0) x2 = max(etmin(l2),0d0)
            x2 = max(x2, xm(l2)/1d0)
c            if (nt .gt. 1) x2 = max(x2,xk(nt-1))
            
c            write(*,*) 'Using 2',l2,x2

            xo = min(x1,x2)

            xo = xo*xo/stot
            a=-pmass(i,iconfig)**2/stot
c            call setgrid(-i,xo,a,pow(i,iconfig))

c               write(*,*) 'Enter minimum for ',-i, xo
c               read(*,*) xo

             if (i .ne. -1 .or. .true.) call setgrid(-i,xo,a,1)
         endif
      enddo
      if (abs(lpp(1)) .eq. 1 .or. abs(lpp(2)) .eq. 1) then
         write(*,*) 'etot',etot,nexternal
         xo = etot**2/stot
         i = 3*(nexternal-2) - 4 + 1
c-----------------------
c     tjs  4/29/2008 use analytic transform for s-hat
c-----------------------
         if (swidth(i) .eq. 0d0) then
            swidth(i) = xo
            spole(i)= -2.0d0    ! 1/s pole
            write(*,*) "Transforming s_hat 1/s ",i,xo
         else
            write(*,*) "Transforming s_hat BW ",spole(i),swidth(i)
         endif
c-----------------------
         if (swidth(i) .eq. 0d0) then
            call setgrid(i,xo,0d0,1)
         endif
      endif

      i=-8
c      write(*,*) 'Enter minimum for ',-i, xo
c      read(*,*) xo      
c      if (xo .gt. 0)      call setgrid(-i,xo,a,1)

      i=-10
c      write(*,*) 'Enter minimum for ',-i, xo
c      read(*,*) xo      
c      if (xo .gt. 0) call setgrid(-i,xo,a,1)

      end



c      subroutine find_matches(iconfig,isym)
cc*****************************************************************************
cc     Finds all of the matches to see what gains may be possible
cc*****************************************************************************
c      implicit none
cc
cc     Constants
cc     
c      include 'genps.inc'
c      double precision   zero
c      parameter (zero = 0d0)
cc
cc     Arguments
cc
c      integer iconfig,isym(0:10)
cc
cc     Local
cc
c      integer i,j, nc, imatch
c      double precision tprop(3,nexternal),xprop(3,nexternal)
cc
cc     Global
cc
c      integer iforest(2,-max_branch:-1,lmaxconfigs)
c      common/to_forest/ iforest
c
c      integer            mapconfig(0:lmaxconfigs), this_config
c      common/to_mconfigs/mapconfig, this_config
c
cc      include 'coupl.inc'
cc      include 'configs.inc'
cc
cc     External
cc
c
cc-----
cc  Begin Code
cc-----      
cc
cc     First get peaks for configuration interested in
cc
c      if (.false.) then
c         isym(0)=1
c         isym(1)=iconfig
c      else
c         call get_peaks(iconfig,tprop)
c         isym(0)=0
c         do i=1,mapconfig(0)
c            call get_peaks(i,xprop)
cc        call sort_prop(xprop)
c            call match_peak(tprop,xprop,imatch)
c            if (imatch .eq. 1) then
c               write(*,*) 'Found Match',iconfig,i
c               isym(0)=isym(0)+1
c               isym(isym(0)) = i
c            endif
c         enddo
c      endif
c      write(*,'(20i4)') (isym(i),i=0,isym(0))
c      end
c
c      subroutine find_all_matches()
cc*****************************************************************************
cc     Finds all of the matches to see what gains may be possible
cc*****************************************************************************
c      implicit none
cc
cc     Constants
cc     
c      include 'genps.inc'
c      double precision   zero
c      parameter (zero = 0d0)
cc
cc     Arguments
cc
c      double precision tprop(3,nexternal),xprop(3,nexternal)
c      integer iconfig
cc
cc     Local
cc
c      integer i,j, nc, imatch
c      logical gm(lmaxconfigs)
cc
cc     Global
cc
c      integer iforest(2,-max_branch:-1,lmaxconfigs)
c      common/to_forest/ iforest
c
c      integer            mapconfig(0:lmaxconfigs), this_config
c      common/to_mconfigs/mapconfig, this_config
c
cc      include 'coupl.inc'
cc      include 'configs.inc'
cc
cc     External
cc
c
cc-----
cc  Begin Code
cc-----      
c      nc = 0
c      do i=1,mapconfig(0)
c         gm(i)=.false.
c      enddo
c      do j=1,mapconfig(0)
c         if (.not. gm(j)) then
c            nc=nc+1
cc            write(*,*) 'Need config ',j
c            call get_peaks(j,tprop)
cc            call sort_prop(tprop)
c            write(*,'(i4,4e12.4)') j,(tprop(1,i), i=1,4)
c            do i=j+1,mapconfig(0)
c               call get_peaks(i,xprop)
cc               call sort_prop(xprop)
c               call match_peak(tprop,xprop,imatch)
c               if (imatch .eq. 1) then
c                  write(*,*) 'Found Match',j,i
c                  gm(i)=.true.
c               endif
c            enddo
c         endif
c      enddo
c      write(*,*) 'Found matches',mapconfig(0),nc
c      stop
c      end

c      subroutine sort_prop(xprop)
cc*****************************************************************************
cc     Sort props in order from min to max based on 1st component only
cc*****************************************************************************
c      implicit none
cc
cc     Constants
cc     
c      include 'genps.inc'
c      double precision   zero
c      parameter (zero = 0d0)
cc
cc     Arguments
cc
c      double precision xprop(3,nexternal)
cc
cc     Local
cc
c      integer i,j,imin
c      double precision temp(3,nexternal),xmin
c      logical used(nexternal)
cc
cc     Global
cc
cc
cc     External
cc
cc-----
cc  Begin Code
cc-----      
c      do i=1,nexternal-3
c         used(i)=.false.
c      enddo
c      do j=1,nexternal-3
c         do i=1,nexternal-3
c            xmin = 2d0
c            if (.not. used(i) .and. xprop(1,i) .lt. xmin) then
c               xmin = xprop(1,i)
c               imin = i
c            endif
c         enddo
c         do i=1,3
c            temp(i,j)=xprop(i,imin)
c         enddo
c         used(i)=.true.
c      enddo
c      do i=1,nexternal-3
c         do j=1,3
c            xprop(j,i)=temp(j,i)
c         enddo
c      enddo
c      end
c
c
c      subroutine match_peak(tprop,xprop,imatch)
cc*****************************************************************************
cc     Determines if two sets of peaks are equivalent
cc*****************************************************************************
c      implicit none
cc
cc     Constants
cc     
c      include 'genps.inc'
c      double precision   zero
c      parameter (zero = 0d0)
cc
cc     Arguments
cc
c      double precision xprop(3,nexternal),tprop(3,nexternal)
c      integer imatch
cc
cc     Local
cc
c      integer i,j
cc
cc     Global
cc
cc
cc     External
cc
cc-----
cc  Begin Code
cc-----      
c      imatch = 1                     !By default assume match
c      do i=1,nexternal-3
c         do j=1,3
c            if (tprop(j,i) .ne. xprop(j,i)) imatch=0
c         enddo
c      enddo
c      end

c      subroutine get_peaks(iconfig,xt)
cc*****************************************************************************
cc     Attempts to determine peaks for this configuration
cc*****************************************************************************
c      implicit none
cc
cc     Constants
cc     
c      include 'genps.inc'
c      double precision   zero
c      parameter (zero = 0d0)
cc
cc     Arguments
cc
c      double precision xt(3,nexternal)
c      integer iconfig
c
cc
cc     Local
cc
c      double precision  xm(-nexternal:nexternal)
c      double precision tsgn, xo, a
c      double precision x1,x2
c      double precision dr,mtot, stot
c      integer i, l1, l2, j
c
c      double precision pmass(-nexternal:0,lmaxconfigs)
c      double precision pwidth(-nexternal:0,lmaxconfigs)
c      integer pow(-nexternal:0,lmaxconfigs)
c
c      integer imatch(0:lmaxconfigs)
c
cc
cc     Global
cc
c      integer iforest(2,-max_branch:-1,lmaxconfigs)
c      common/to_forest/ iforest
c
c      integer            mapconfig(0:lmaxconfigs), this_config
c      common/to_mconfigs/mapconfig, this_config
c
c      real*8         emass(nexternal)
c      common/to_mass/emass
c
c      include 'run.inc'
c
c      double precision etmin(nincoming+1:nexternal),etamax(nincoming+1:nexternal)
c      double precision emin(nincoming+1:nexternal)
c      double precision                    r2min(nincoming+1:nexternal,nincoming+1:nexternal)
c      double precision s_min(nexternal,nexternal)
c      common/to_cuts/  etmin, emin, etamax, r2min, s_min
c
c      double precision      spole(maxinvar),swidth(maxinvar),bwjac
c      common/to_brietwigner/spole          ,swidth          ,bwjac
c      
c      include 'coupl.inc'
cc      include 'props.inc'
c
cc
cc     External
cc
c
cc-----
cc  Begin Code
cc-----      
c      stot = 4d0*ebeam(1)*ebeam(2)
cc      iconfig = this_config
c      mtot = 0d0
c      do i=1,nexternal
c         xm(i)=emass(i)
c         mtot=mtot+xm(i)
c      enddo
c      tsgn    = +1d0
c      do i=-1,-(nexternal-3),-1              !Find all the propagotors
c         if (iforest(1,i,iconfig) .eq. 1) tsgn=-1d0
c         if (tsgn .eq. 1d0) then                         !s channel
c            xm(i) = xm(iforest(1,i,iconfig))+xm(iforest(2,i,iconfig))
c            mtot = mtot - xm(i)
c            if (pwidth(i,iconfig) .gt. 0) then    !B.W.
c               write(*,*) 'Setting BW',i,pmass(i,iconfig)
c               spole(-i)=pmass(i,iconfig)*pmass(i,iconfig)/stot
c               swidth(-i) = pwidth(i,iconfig)*pmass(i,iconfig)/stot
c               xm(i) = pmass(i,iconfig)
c               xt(1,-i) = spole(-i)
c               xt(2,-i) = swidth(-i)
c               xt(3,-i) = 2
c            else                                  !1/x^pow
c               if (xm(i) - pmass(i,iconfig) .le. 0d0) then !Can hit pole
cc                  write(*,*) 'Setting new min',i,xm(i),pmass(i,iconfig)
c                  l1 = iforest(1,i,iconfig)                  !need dr cut
c                  l2 = iforest(2,i,iconfig)
c                  if (l2 .lt. l1) then
c                     j = l1
c                     l1 = l2
c                     l2 = j
c                  endif
c                  dr = 0
cc-fax
c                  if (l1 .gt. 0) 
c     &  dr = max(r2min(l2,l1)*dabs(r2min(l2,l1)),0d0) !dr only for external
c                  dr = dr*.8d0                        !0.8 to hit peak hard
c                  xo = 0.5d0*pmass(i,iconfig)**2      !0.5 to hit peak hard
cc-fax
c                  if (dr .gt. 0d0) 
c     &            xo = max(xo,max(etmin(l2),0d0)*max(etmin(l1),0d0)*dr)
cc                  write(*,*) 'Got dr',i,l1,l2,dr
c                  xo = xo+pmass(i,iconfig)**2
cc                  write(*,*) 'Setting xm',i,xm(i),sqrt(xo)
c                  xm(i) = sqrt(xo)    !Reset xm to realistic minimum
c                  xo = xo/stot
cc                  xo = sqrt(pmass(i,iconfig)**2+ pmass(i,iconfig)**2)
cc-fax
cc                  xo = pmass(i,iconfig)+max(etmin,0d0)
c               else
cc                  write(*,*) 'Using xm',i,xm(i)
c                  xo = xm(i)**2/stot
c               endif
c               a=pmass(i,iconfig)**2/stot
cc               call setgrid(-i,xo,a,pow(i,iconfig))
cc               call setgrid(-i,xo,a,1)
c               xt(1,-i) = xo
c               xt(2,-i) = a
c               xt(3,-i) = 1
c            endif
c            mtot=mtot+xm(i)
cc            write(*,*) 'New mtot',i,mtot,xm(i)
c         else                                        !t channel
cc
cc     Check closest to p1
cc
c            l2 = iforest(2,i,iconfig) !need dr cut
c            x1 = 0
cc-fax
c            if (l2 .gt. 0) x1 = max(etmin(l2),0d0)
c            x1 = max(x1, xm(l2)/2d0)
cc            write(*,*) 'Using 1',l2,x1
c
cc
cc     Check closest to p2
cc
c            j = i-1
c            l2 = iforest(2,j,iconfig)
c            x2 = 0
cc-fax
c            if (l2 .gt. 0) x2 = max(etmin(l2),0d0)
c            x2 = max(x2, xm(l2)/2d0)
c            
cc            write(*,*) 'Using 2',l2,x2
c
c            xo = min(x1,x2)
c
c            xo = xo*xo/stot
c            a=-pmass(i,iconfig)**2/stot
cc            call setgrid(-i,xo,a,pow(i,iconfig))
cc            call setgrid(-i,xo,a,1)
c               xt(1,-i) = xo
c               xt(2,-i) = a
c               xt(3,-i) = 1
c         endif
c      enddo
cc---------------------
cc     tjs routine for x-hat
cc------------------------
c      if (abs(lpp(1)) .eq. 1 .or. abs(lpp(2)) .eq. 1) then
c         write(*,*) "setting s_hat",mtot,sqrt(stot)
c         xo = mtot**2/stot
c         i = 3*(nexternal-2) - 4 + 1
cc         call setgrid(i,xo,0d0,1)
c      endif
c      end


c       subroutine writeamp(p)
cc
cc     Constants
cc     
c      include 'genps.inc'
cc
cc     Arguments
cc
c      double precision p(0:3,nexternal)
cc
cc     Local
cc
c      double precision xp(0:3,-nexternal:nexternal)
c      integer i,j,iconfig
cc
cc     Global
cc
c      integer iforest(2,-max_branch:-1,lmaxconfigs)
c      common/to_forest/ iforest
c
c      integer            mapconfig(0:lmaxconfigs), this_config
c      common/to_mconfigs/mapconfig, this_config
c
cc
cc     External
cc
c      double precision dot
cc-----
cc  Begin Code
cc-----
c      iconfig = this_config
c      do i=1,nexternal
c         do j=0,3
c            xp(j,i)=p(j,i)
c         enddo
c      enddo
c      shat = dot(p(0,1),p(0,2))
c      testamp = 1d0
c      tsgn    = +1d0
c      do i=-1,-(nexternal-3),-1              !Find all the propagotors
c         if (iforest(1,i,iconfig) .eq. 1) tsgn=-1d0
c         do j=0,3
c            xp(j,i) = xp(j,iforest(1,i,iconfig))
c     $           +tsgn*xp(j,iforest(2,i,iconfig))
c         enddo
cc         testamp=testamp*shat/(dot(xp(0,i),xp(0,i)))**2
cc         testamp=testamp**2
c      enddo
c      if(nexternal.gt.3)
c     $   write(*,'(A,4e15.5)') 'V',(dot(xp(0,-i),xp(0,-i)),i=1,nexternal-3)
cc      testamp=abs(testamp)
c      end
c
c
c      subroutine histamp(p,dwgt)
cc
cc     Constants
cc     
c      include 'genps.inc'
cc
cc     Arguments
cc
c      double precision p(0:3,nexternal),dwgt
cc
cc     Local
cc
c      double precision xp(0:3,-nexternal:nexternal)
c      integer i,j,iconfig
c      real wgt
cc
cc     Global
cc
c      integer iforest(2,-max_branch:-1,lmaxconfigs)
c      common/to_forest/ iforest
c
c      integer            mapconfig(0:lmaxconfigs), this_config
c      common/to_mconfigs/mapconfig, this_config
c
cc
cc     External
cc
c      double precision dot
cc-----
cc  Begin Code
cc-----
c      wgt = dwgt
c      iconfig = this_config
c      do i=1,nexternal
c         do j=0,3
c            xp(j,i)=p(j,i)
c         enddo
c      enddo
c      shat = dot(p(0,1),p(0,2))
c      testamp = 1d0
c      tsgn    = +1d0
c      do i=-1,-(nexternal-3),-1              !Find all the propagotors
c         if (iforest(1,i,iconfig) .eq. 1) tsgn=-1d0
c         do j=0,3
c            xp(j,i) = xp(j,iforest(1,i,iconfig))
c     $           +tsgn*xp(j,iforest(2,i,iconfig))
c         enddo
c      enddo
c      do i=1,nexternal-3
cc         write(*,*) sqrt(abs(dot(xp(0,-i),xp(0,-i)))),wgt
c         call hfill(10+i,real(sqrt(abs(dot(xp(0,-i),xp(0,-i))))),0.,wgt)
c      enddo
c      end
c
c
c      subroutine check_limits(p,xlimit, iconfig)
cc*************************************************************************
cc     Checks limits on all of the functions being integrated
cc*************************************************************************
c      implicit none
cc
cc     Constants
cc
c      include 'genps.inc'
cc
cc     Arguments
cc
c      double precision p(0:3,nexternal), xlimit(2,nexternal)
c      integer iconfig
cc
cc     Local
cc
c      double precision xp(0:3,-nexternal:nexternal), tsgn
c      double precision sm
c      integer ik(4)
c      integer i,j, k1,k2
cc
cc     Global
cc
c      integer iforest(2,-max_branch:-1,lmaxconfigs)
c      common/to_forest/ iforest
c      integer            mapconfig(0:lmaxconfigs), this_config
c      common/to_mconfigs/mapconfig, this_config
cc
cc     External
cc
c      double precision dot
c      external dot
c
cc      data ik/1,2,3,0/
cc-----
cc  Begin Code
cc-----
cc
cc     Transform from rambo(1:4) format to helas (0:3)
cc     
c      do i=1,nexternal
c         do j=0,3
c            xp(j,i)=p(j,i)
c         enddo
c      enddo
cc
cc     Now build propagators
cc      
c      tsgn=+1d0
c      do i=-1,-(nexternal-3),-1
c         if (iforest(1,i,iconfig) .eq. 1) tsgn=-1d0
c         k1=iforest(1,i,iconfig)
c         k2=iforest(2,i,iconfig)
c         do j=0,3
c            xp(j,i)=xp(j,k1)+tsgn*xp(j,k2)
c         enddo
c         sm = tsgn*dot(xp(0,i),xp(0,i))
c         if (sm .lt. xlimit(1,-i)) then
c            xlimit(1,-i)=sm
cc            write(*,*) 'New limit',-i,sm
c         endif
c         if (sm .gt. xlimit(2,-i)) then
c            xlimit(2,-i)=sm
cc            write(*,*) 'New limit',-i,sm
c         endif
c      enddo
c      end
c
