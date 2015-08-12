      double precision function testamp(p)
c*****************************************************************************
c     Approximates matrix element by propagators
c*****************************************************************************
      implicit none
c
c     Constants
c     
      include 'genps.inc'
      include 'maxconfigs.inc'
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

      double precision prmass(-nexternal:0,lmaxconfigs)
      double precision prwidth(-nexternal:0,lmaxconfigs)
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

      save prmass,prwidth,pow
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
         if (prwidth(i,iconfig) .ne. 0d0 .and. .false.) then
            testamp=testamp/((dot(xp(0,i),xp(0,i))
     $                        -prmass(i,iconfig)**2)**2
     $         -(prmass(i,iconfig)*prwidth(i,iconfig))**2)
         else
            testamp = testamp/((dot(xp(0,i),xp(0,i)) -
     $                          prmass(i,iconfig)**2)
     $                          **(pow(i,iconfig)))
         endif
        testamp=testamp*shat**(pow(i,iconfig))
c        write(*,*) i,iconfig,pow(i,iconfig),prmass(i,iconfig)
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
      include 'maxconfigs.inc'
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
      integer i,j,iconfig,iproc

      double precision prmass(-nexternal:0,lmaxconfigs)
      double precision prwidth(-nexternal:0,lmaxconfigs)
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
      integer sprop(maxsproc,-max_branch:-1,lmaxconfigs)
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

      integer gForceBW(-max_branch:-1,lmaxconfigs)  ! Forced BW
      include 'decayBW.inc'
c
c     External
c
      double precision dot

      save prmass,prwidth,pow
      data first_time /.true./
c-----
c  Begin Code
c-----      
      cut_bw = .false.    !Default is we passed the cut
      iconfig = this_config
      if (first_time) then
         include 'props.inc'
         nbw = 0
         do i=-1,-(nexternal-3),-1
            if (iforest(1,i,iconfig) .eq. 1 .or. prwidth(i,iconfig).le.0) then
              cycle
            endif
            nbw=nbw+1
            if (lbw(nbw) .eq. 1) then
               write(*,*) 'Requiring BW ',i,nbw
            elseif(lbw(nbw) .eq. 2) then
               write(*,*) 'Excluding BW ',i,nbw
            else
               write(*,*) 'No cut BW ',i,nbw
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
c     Find non-zero process number
      do iproc=1,maxsproc
         if(sprop(iproc,-1,iconfig).ne.0) goto 10
      enddo
 10   continue
c     If no non-zero sprop, set iproc to 1
      if(iproc.gt.maxsproc) iproc=1
c     Start loop over propagators
      do i=-1,-(nexternal-3),-1
         onbw(i) = .false.
         if (iforest(1,i,iconfig) .eq. 1) tsgn=-1d0
         do j=0,3
            xp(j,i) = xp(j,iforest(1,i,iconfig))
     $           +tsgn*xp(j,iforest(2,i,iconfig))
         enddo
         if (tsgn .lt. 0d0) cycle
         if (prwidth(i,iconfig) .gt. 0d0 ) then !This is B.W.
            nbw=nbw+1
c            write(*,*) 'Checking BW',nbw
            xmass = sqrt(dot(xp(0,i),xp(0,i)))
c            write(*,*) 'xmass',xmass,prmass(i,iconfig)
c
c           Here we set if the BW is "on-shell" for LesHouches
c
            onshell = (abs(xmass - prmass(i,iconfig)) .lt.
     $           bwcutoff*prwidth(i,iconfig))
            if(onshell)then
c     Remove on-shell forbidden s-channels (gForceBW=2) (JA 2/10/11)
              if(gForceBW(i,iconfig).eq.2) then
                 cut_bw = .true.
                 return               
              endif
c           Only allow OnBW if no "decay" to identical particle
              OnBW(i) = .true.
              idenpart=0
              do j=1,2
                ida(j)=iforest(j,i,iconfig)
                if(ida(j).lt.0) then
                   if(sprop(iproc,i,iconfig).eq.sprop(iproc,ida(j),iconfig))
     $                  idenpart=ida(j)
                elseif (ida(j).gt.0) then
                   if(sprop(iproc,i,iconfig).eq.IDUP(ida(j),1,iproc))
     $                  idenpart=ida(j)
                endif
              enddo
c           Always remove if daughter final-state
              if(idenpart.gt.0) then
                 OnBW(i)=.false.
c           Else remove if daughter forced to be onshell
              elseif(idenpart.lt.0)then
                 if(gForceBW(idenpart, iconfig).eq.1) then
                    OnBW(i)=.false.
c           Else remove daughter if forced to be onshell
                 elseif(gForceBW(i, iconfig).eq.1) then
                    OnBW(idenpart)=.false.
c           Else remove either this resonance or daughter, which is closer to mass shell
                 elseif(abs(xmass-prmass(i,iconfig)).gt.
     $                   abs(sqrt(dot(xp(0,idenpart),xp(0,idenpart)))-
     $                   prmass(i,iconfig))) then
                    OnBW(i)=.false.
c           Else remove OnBW for daughter
                 else
                    OnBW(idenpart)=.false.
                 endif
              endif
            else if (gForceBW(i, iconfig).eq.1) then ! .not. onshell
c             Check if we are supposed to cut forced bw (JA 4/8/11)
              cut_bw = .true.
c              write(*,*) 'cut_bw: ',i,gForceBW(i,iconfig),OnBW(i),cut_bw
              return
            endif
c
c     Here we set onshell for phase space integration (JA 4/8/11)
c     For decay-chain syntax use BWcutoff here too (22/12/14)
            if (gForceBW(i, iconfig).eq.1) then
               onshell = (abs(xmass - prmass(i,iconfig)) .lt.
     $           bwcutoff*prwidth(i,iconfig))
            else
               onshell = (abs(xmass - prmass(i,iconfig)) .lt.
     $           5d0*prwidth(i,iconfig))
            endif

            if (onshell .and. (lbw(nbw).eq. 2) .or.
     $          .not. onshell .and. (lbw(nbw).eq. 1)) then
               cut_bw=.true.
               if (gForceBW(i, iconfig).eq.1) then
                  return
               endif
c               write(*,*) 'cut_bw: ',nbw,xmass,onshell,lbw(nbw),cut_bw
            endif
         endif
c         write(*,*) 'final cut_bw: ',nbw,lbw(nbw),xmass,onshell,OnBW(i),cut_bw
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
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'maxamps.inc'
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
      double precision bwcut_for_PS(-nexternal:0)
      double precision tsgn, xo, a
      double precision x1,x2,xk(nexternal)
      double precision dr,mtot,etot,xqfact
      double precision spmass
      integer i, iconfig, l1, l2, j, nt, nbw, iproc, k
      integer iden_part(-nexternal+1:nexternal)

      double precision prmass(-nexternal:0,lmaxconfigs)
      double precision prwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)

      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'

      integer gForceBW(-max_branch:-1,lmaxconfigs)  ! Forced BW
      include 'decayBW.inc'

c
c     Global
c
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest

      integer sprop(maxsproc,-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      common/to_sprop/sprop,tprid

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

      double precision stot,m1,m2
      common/to_stot/stot,m1,m2

      include 'coupl.inc'
      include 'cuts.inc'
C
C     SPECIAL CUTS
C
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_L(NEXTERNAL)
      LOGICAL  IS_A_B(NEXTERNAL),IS_A_A(NEXTERNAL),IS_A_ONIUM(NEXTERNAL)
      LOGICAL  IS_A_NU(NEXTERNAL),IS_HEAVY(NEXTERNAL), DO_CUTS(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_A,IS_A_L,IS_A_B,IS_A_NU,IS_HEAVY,
     . IS_A_ONIUM,DO_CUTS
      integer njet



c
c     External
c

c-----
c  Begin Code
c-----      
      include 'props.inc'
c      etmin = 10
      nt = 0
      iconfig = this_config
      mtot = 0d0
      etot = 0d0   !Total energy needed
      spmass = 0d0 !Keep track of BW masses for shat
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
      spmass=mtot
      tsgn    = +1d0
c     Reset variables
      nbw = 0
      do i=1,nexternal-2
         spole(i)=0
         swidth(i)=0
      enddo
c     Find non-zero process number
      do iproc=1,maxsproc
         if(sprop(iproc,-1,iconfig).ne.0) goto 10
      enddo
 10   continue
c     If no non-zero sprop, set iproc to 1
      if(iproc.ge.maxsproc.and.sprop(maxsproc,-1,iconfig).eq.0)
     $     iproc=1

c     Look for identical particles to map radiation processes
      call idenparts(iden_part, iforest(1,-max_branch,iconfig),
     $     sprop(1,-max_branch,iconfig), gForceBW(-max_branch,iconfig),
     $     prwidth(-nexternal,iconfig))

c     Start loop over propagators
      do i=-1,-(nexternal-3),-1
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
              xm(i)=max(xm(i),max(xqcutij(l1,l2),0d0))
            endif
c            write(*,*) 'iconfig,i',iconfig,i
c            write(*,*) prwidth(i,iconfig),prmass(i,iconfig)
            if (prwidth(i,iconfig) .gt. 0 ) then
               nbw=nbw+1
c              JA 6/8/2011 Set xe(i) for resonances
               if (gforcebw(i,iconfig).eq.1) then
                  xm(i) = max(xm(i), prmass(i,iconfig)-bwcutoff*prwidth(i,iconfig))
                  bwcut_for_PS(i) = bwcutoff
               else if (lbw(nbw).eq.1) then
                  xm(i) = max(xm(i), prmass(i,iconfig)-5d0*prwidth(i,iconfig))
                  bwcut_for_PS(i) = 5d0
               else
                  bwcut_for_PS(i) = 5d0
               endif
            endif
            xe(i)=max(xe(i),xm(i))
c     Check for impossible onshell configurations
c     Either: required onshell and daughter masses too large
c     Or: forced and daughter masses too large
c     Or: required offshell and forced
            if(prwidth(i,iconfig) .gt. 0.and.
     $         (lbw(nbw).eq.1.and.
     $          (prmass(i,iconfig)+bwcut_for_PS(i)*prwidth(i,iconfig).lt.xm(i)
     $           .or.prmass(i,iconfig)-bwcut_for_PS(i)*prwidth(i,iconfig).gt.dsqrt(stot))
     $          .or.gforcebw(i,iconfig).eq.1.and.
     $              prmass(i,iconfig)+bwcutoff*prwidth(i,iconfig).lt.xm(i)
     $          .or.lbw(nbw).eq.2.and.gforcebw(i,iconfig).eq.1))
     $        then
c     Write results.dat and quit
               call write_null_results()
               stop
            endif
            if (prwidth(i,iconfig) .gt. 0 .and. lbw(nbw) .le. 1) then         !B.W.
               if (i .eq. -(nexternal-(nincoming+1))) then  !This is s-hat
                  j = 3*(nexternal-2)-4+1    !set i to ndim+1
c-----
c tjs 11/2008 if require BW then force even if worried about energy
c JA 8/2011 don't use BW if mass is > CM energy
c----
                  if(prmass(i,iconfig).ge.xm(i).and.iden_part(i).eq.0.and.
     $                 prmass(i,iconfig).lt.sqrt(stot)
     $                 .or. lbw(nbw).eq.1) then
                     write(*,*) 'Setting PDF BW',j,nbw,prmass(i,iconfig)
                     spole(j)=prmass(i,iconfig)*prmass(i,iconfig)/stot
                     swidth(j) = prwidth(i,iconfig)*prmass(i,iconfig)/stot
                  endif
               else if((prmass(i,iconfig)+bwcut_for_PS(i)*prwidth(i,iconfig)).ge.xm(i)
     $                  .and. iden_part(i).eq.0 .or. lbw(nbw).eq.1) then
c              JA 02/13 Only allow BW if xm below M+5*Gamma
                  write(*,*) 'Setting BW',i,nbw,prmass(i,iconfig)
                  spole(-i)=prmass(i,iconfig)*prmass(i,iconfig)/stot
                  swidth(-i) = prwidth(i,iconfig)*prmass(i,iconfig)/stot
               endif
c     JA 4/1/2011 Set grid in case there is no BW (radiation process)
               if (swidth(-i) .eq. 0d0 .and.
     $              i.ne.-(nexternal-(nincoming+1)))then
                  a=prmass(i,iconfig)**2/stot
                  xo = min(xm(i)**2/stot, 1-1d-8)
                  if (xo.eq.0d0) xo=1d0/stot
                  call setgrid(-i,xo,a,1)
               endif
c     Set spmass for BWs
               if (swidth(-i) .ne. 0d0)
     $              spmass=spmass-xm(i) +
     $              max(xm(i),prmass(i,iconfig)-bwcut_for_PS(i)*prwidth(i,iconfig))
            else                                  !1/x^pow
              a=prmass(i,iconfig)**2/stot
c     JA 4/1/2011 always set grid
              xo = min(xm(i)**2/stot, 1-1d-8)

c     OM 7/27/2013 use MMJJ in order to set the mass in a appropriate way
              if (xo.eq.0d0.and.MMJJ.gt.0d0) then
                 njet = 0
                 do k =1,2
                    if (iforest(k,i,iconfig).gt.0)then
                      if (is_a_j(iforest(k,i,iconfig))) njet = njet + 1
                    endif
                 enddo
                 if (njet.eq.1) then
                    xo = (MMJJ/1d2)**2/stot
                 else if (njet.eq.2) then
                    xo = (MMJJ * 0.8)**2/stot
                 endif
              endif
              if (xo.eq.0d0) xo=1d0/stot
c              if (prwidth(i, iconfig) .eq. 0d0.or.iden_part(i).gt.0) then 
              call setgrid(-i,xo,a,1)
c              else 
c                 write(*,*) 'Using flat grid for BW',i,nbw,
c     $                prmass(i,iconfig)
c              endif
            endif
            etot = etot+xe(i)
            mtot=mtot+xm(i)
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
            x1 = max(x1, xe(l2)/1d0)
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
            x2 = max(x2, xe(l2)/1d0)
c            if (nt .gt. 1) x2 = max(x2,xk(nt-1))
            
c            write(*,*) 'Using 2',l2,x2

            xo = min(x1,x2)

c           Use 1/10000 of sqrt(s) as minimum, to always get integration
            xo = xo*xo/stot
            if (xo.eq.0d0)then
               xo=1d0/stot
               write(*,*) 'Warning: No cutoff for shat integral found'
               write(*,*) '         Minimum set to ', xo
            endif
            a=-prmass(i,iconfig)**2/stot
c            call setgrid(-i,xo,a,pow(i,iconfig))

c               write(*,*) 'Enter minimum for ',-i, xo
c               read(*,*) xo
             if (i .ne. -1 .or. .true.) call setgrid(-i,xo,a,1)
         endif
      enddo
c     Perform setting for shat (PDF BW or 1/s)
      if (abs(lpp(1)) .eq. 1 .or. abs(lpp(2)) .eq. 1) then
c     Set minimum based on: 1) required energy 2) resonances 3) 1/10000 of sqrt(s)
         i = max(1,3*(nexternal-2) - 4 + 1)
         xo = max(min(etot**2/stot, 1d0-1d-8),1d0/stot)
c        Take into account special cuts
         xo = max(xo, xptj*dabs(xptj)/stot)
         xo = max(xo, xptb*dabs(xptb)/stot)
         xo = max(xo, xpta*dabs(xpta)/stot)
         xo = max(xo, xptl*dabs(xptl)/stot)
         xo = max(xo, xmtc*dabs(xmtc)/stot)
         xo = max(xo, htjmin**2/stot)
         xo = max(xo, ptj1min**2/stot)
         xo = max(xo, (2*ptj2min)**2/stot)
         xo = max(xo, (3*ptj3min)**2/stot)
         xo = max(xo, (4*ptj4min)**2/stot)
         xo = max(xo, ht2min**2/stot)
         xo = max(xo, ht3min**2/stot)
         xo = max(xo, ht4min**2/stot)
         xo = max(xo, misset**2/stot)
         xo = max(xo, ptllmin**2/stot)
         xo = max(xo, ptl1min**2/stot)
         xo = max(xo, (2*ptl2min)**2/stot)
         xo = max(xo, (3*ptl3min)**2/stot)
         xo = max(xo, (4*ptl4min)**2/stot)
         xo = max(xo, mmnl**2/stot)
         if (mmjj.ne.0d0) then
            njet = 0
            do k=nincoming+1,nexternal
               if (is_a_j(k)) njet = njet + 1
            enddo
            xo = max(xo, njet*(njet -1)/2d0*mmjj**2/stot)
         endif

c     Include mass scale from BWs
         xo = max(xo, spmass**2/stot)
         if (swidth(i).eq.0.and.xo.eq.1d0/stot) then
            write(*,*) 'Warning: No minimum found for integration'
            write(*,*) '         Setting minimum to ',1d0/stot
         endif
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

      subroutine write_null_results()
      implicit none

      write(*,*),'Impossible BW configuration'
      open(unit=66,file='results.dat',status='unknown')
      write(66,'(3e12.5,2i9,i5,i9,4e10.3)')0.,0.,0.,0,0,1,0,0.,0.,0.,0.
      write(66,'(i4,5e15.5)') 1,0.,0.,0.,0.,0.
      close(66)
      end
