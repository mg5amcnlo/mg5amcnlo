      subroutine addmothers(ip,jpart,pb,isym,jsym,rscale,aqcd,aqed,buff,
     $                      npart,numproc)

      implicit none
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'cluster.inc'
      include 'message.inc'
      include 'run.inc'
      include 'maxamps.inc'

      integer jpart(7,-nexternal+3:2*nexternal-3),npart,ip,numproc
      double precision pb(0:4,-nexternal+3:2*nexternal-3)
      double precision rscale,aqcd,aqed,targetamp(maxflow)
      character*140 buff

      integer isym(nexternal,99), jsym
      integer i,j,k,ida(2),ns,nres,ires,icl,ito2,idenpart,nc,ic
      integer ito(-nexternal+3:nexternal),iseed,maxcolor
      integer icolalt(2,-nexternal+3:2*nexternal-3)
      double precision qicl(-nexternal+3:2*nexternal-3), factpm
      double precision xtarget
      data iseed/0/

      double precision ZERO
      parameter (ZERO=0d0)
      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      logical first_time
      save pmass,pwidth,pow
      data first_time /.true./

      Double Precision amp2(maxamps), jamp2(0:maxflow)
      common/to_amps/  amp2,       jamp2

      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig
      integer idmap(-nexternal:nexternal),icmp

      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      common/to_sprop/sprop,tprid
      integer            mapconfig(0:lmaxconfigs), iconfig
      common/to_mconfigs/mapconfig, iconfig

      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'

      include 'coloramps.inc'
      
      logical             OnBW(-nexternal:0)     !Set if event is on B.W.
      common/to_BWEvents/ OnBW

C     iproc has the present process number
      integer iproc
      common/to_iproc/iproc
      data iproc/1/

c      integer ncols,ncolflow(maxamps),ncolalt(maxamps),icorg
c      common/to_colstats/ncols,ncolflow,ncolalt,icorg

      double precision pt
      real ran1
      external pt,ran1

      if (first_time) then
         include 'props.inc'
         first_time=.false.
      endif

      npart = nexternal
      buff = ' '
c   
c   Choose the config (diagram) which was actually used to produce the event
c   
c   ...unless the diagram is passed in igscl(1); then use that diagram
c      if (igscl(0).ne.0) then
c        if (btest(mlevel,3)) then
c          write(*,*)'unwgt.f: write out diagram ',igscl(1)
c        endif
c        iconfig=igscl(1)
c      endif
      
c
c    Choose a color flow which is certain to work with the propagator
c    structure of the chosen diagram and use that as an alternative
c   

      nc = int(jamp2(0))
      maxcolor=0
      if(nc.gt.0)then
      if(icolamp(1,iconfig,iproc)) then
        targetamp(1)=jamp2(1)
c        print *,'Color flow 1 allowed for config ',iconfig
      else
        targetamp(1)=0d0
      endif
      do ic =2,nc
        if(icolamp(ic,iconfig,iproc))then
          targetamp(ic) = jamp2(ic)+targetamp(ic-1)
c          print *,'Color flow ',ic,' allowed for config ',iconfig
        else
          targetamp(ic)=targetamp(ic-1)
        endif
      enddo
      xtarget=ran1(iseed)*targetamp(nc)

      ic = 1
      do while (targetamp(ic) .lt. xtarget .and. ic .lt. nc)
         ic=ic+1
      enddo
c      print *,'Chose color flow ',ic

      if(targetamp(nc).eq.0) ic=0
      do i=1,nexternal
         if(ic.gt.0) then
            icolalt(1,isym(i,jsym))=icolup(1,i,ic,numproc)
            icolalt(2,isym(i,jsym))=icolup(2,i,ic,numproc)
            if (icolup(1,i,ic, numproc).gt.maxcolor) maxcolor=icolup(1,i,ic, numproc)
            if (icolup(2,i,ic, numproc).gt.maxcolor) maxcolor=icolup(2,i,ic, numproc)
         else
            icolalt(1,i)=jpart(4,i)
            icolalt(2,i)=jpart(5,i)
            if (jpart(4,i).gt.maxcolor) maxcolor=jpart(4,i)
            if (jpart(5,i).gt.maxcolor) maxcolor=jpart(5,i)
         endif
      enddo

      else ! nc.gt.0

      do i=1,nexternal
         icolalt(1,i)=0
         icolalt(2,i)=0
      enddo

      endif ! nc.gt.0

c     
c     Get mother information from chosen graph
c     

c     First check number of resonant s-channel propagators
        ns=0
        nres=0

c     Loop over propagators to find mother-daughter information
        do i=-1,-nexternal+3,-1
c       Daughters
          ida(1)=iforest(1,i,iconfig)
          ida(2)=iforest(2,i,iconfig)
          do j=1,2
            if(ida(j).gt.0) ida(j)=isym(ida(j),jsym)
          enddo
c       Decide s- or t-channel
          if(iabs(sprop(i,iconfig)).gt.0) then ! s-channel propagator
            jpart(1,i)=sprop(i,iconfig)
            ns=ns+1
          else
c         Don't care about t-channel propagators
            goto 100
          endif
c       Set status codes for propagator
c          if((igscl(0).ne.0.and.
c     $       (iabs(jpart(1,i)).gt.5.and.iabs(jpart(1,i)).lt.11).or.
c     $       (iabs(jpart(1,i)).gt.16.and.iabs(jpart(1,i)).ne.21)).or.
c     $       (igscl(0).eq.0.and.OnBW(i))) then 
          if(OnBW(i)) then 
c         Resonance whose mass should be preserved
            jpart(6,i)=2
            nres=nres+1
          else
c         Propagator for documentation only - not included
            jpart(6,i)=3
          endif
c       Calculate momentum (p1+p2 for s-channel, p2-p1 for t-channel)
          do j=0,3
            pb(j,i)=pb(j,ida(1))+pb(j,ida(2))
          enddo
          pb(4,i)=sqrt(max(0d0,pb(0,i)**2-pb(1,i)**2-pb(2,i)**2-pb(3,i)**2))
c          if(jpart(6,i).eq.2.and.
c     $       abs(pb(4,i)-pmass(i,iconfig)).gt.5d0*pwidth(i,iconfig)) then
c            jpart(6,i)=3
c            nres=nres-1
c          endif
c     Remove propagator if "decay" to same particle (i.e. radiation of g/gamma/higgs)
          if(jpart(6,i).eq.2 .and. (jpart(1,i).eq.jpart(1,ida(1)).or.
     $         jpart(1,i).eq.jpart(1,ida(2)))) then
             if(jpart(1,i).eq.jpart(1,ida(1))) idenpart=ida(1)
             if(jpart(1,i).eq.jpart(1,ida(2))) idenpart=ida(2)
c     Always remove if daughter final-state or already removed for this reason
             if(idenpart.gt.0.or.jpart(6,idenpart).eq.4) then
                jpart(6,i)=4
                nres=nres-1
c     Else remove either this resonance or daughter, which is closer to mass shell
             elseif(abs(pb(4,i)-pmass(i,iconfig)).gt.
     $               abs(pb(4,idenpart)-pmass(i,iconfig))) then
                jpart(6,i)=4
                nres=nres-1
             else if(jpart(6,idenpart).eq.2) then
                jpart(6,idenpart)=4
                nres=nres-1
             endif
          endif
c       Set color info for all s-channels
c       Fist set "safe" color info
          if(icolalt(1,ida(1))+icolalt(1,ida(2))-
     $       icolalt(2,ida(1))-icolalt(2,ida(2)).eq.0) then ! color singlet
            icolalt(1,i) = 0
            icolalt(2,i) = 0            
          elseif(icolalt(1,ida(1))-icolalt(2,ida(2)).eq.0) then
            icolalt(1,i) = icolalt(1,ida(2))
            icolalt(2,i) = icolalt(2,ida(1))
          else if(icolalt(1,ida(2))-icolalt(2,ida(1)).eq.0) then
            icolalt(1,i) = icolalt(1,ida(1))
            icolalt(2,i) = icolalt(2,ida(2))
          else if(icolalt(1,ida(2)).gt.0.and.icolalt(1,ida(1)).gt.0.and.
     $           icolalt(2,ida(2)).le.0.and.icolalt(2,ida(1)).le.0) then         ! sextet
            maxcolor=maxcolor+1
            icolalt(1,i) = maxcolor
            icolalt(2,i) = 0
          else if(icolalt(2,ida(2)).gt.0.and.icolalt(2,ida(1)).gt.0.and.
     $           icolalt(1,ida(2)).le.0.and.icolalt(1,ida(1)).le.0) then         ! antisextet
            maxcolor=maxcolor+1
            icolalt(1,i) = 0
            icolalt(2,i) = maxcolor
          else if(jpart(6,i).ge.3) then ! Don't need to match
            icolalt(1,i) = icolalt(1,ida(1))+icolalt(1,ida(2))
            icolalt(2,i) = icolalt(2,ida(1))+icolalt(2,ida(2))
          else
c         Erraneous color assignment for propagator - set color to 0
            icolalt(1,i) = 0
            icolalt(2,i) = 0
          endif
c         Set tentative mothers
          jpart(2,i) = 1
          jpart(3,i) = 2
c         Set mother info for daughters
          do j=1,2
            jpart(2,ida(j)) = i
            jpart(3,ida(j)) = i
          enddo
c       Just zero helicity info for intermediate states
          jpart(7,i) = 0
        enddo                   ! do i
 100    continue

c    Remove non-resonant mothers, set position of particles
        ires=0
        do i=-ns,nexternal
          jpart(4,i)=icolalt(1,i)
          jpart(5,i)=icolalt(2,i)
          if(i.eq.1.or.i.eq.2) then 
            ito(i)=i            ! initial state particle
          else if(i.ge.3) then 
            ito(i)=i+nres       ! final state particle
          else if(i.le.-1.and.jpart(6,i).eq.2) then
            ires=ires+1
            ito(i)=2+ires       ! s-channel resonances
          else 
            ito(i)=0
            if(i.eq.0) cycle
          endif
          if(jpart(2,i).lt.0.and.jpart(6,jpart(2,i)).ne.2) then
            jpart(2,i)=jpart(2,jpart(2,i))
            jpart(3,i)=jpart(3,jpart(3,i))
          endif
        enddo
        
c
c    Shift particles to right place and set mothers of particles
c
        do i=nexternal,-ns,-1
          if(ito(i).le.0) cycle
          do j=1,7
            jpart(j,ito(i))=jpart(j,i)
          enddo
          if(jpart(2,ito(i)).lt.0) then
            jpart(2,ito(i))=ito(jpart(2,ito(i)))
            jpart(3,ito(i))=ito(jpart(3,ito(i)))
          endif
          do j=0,4
            pb(j,ito(i))=pb(j,i)
          enddo
        enddo

        if(ickkw.gt.0)
     $       write(buff,'(a1,9e15.7)') '#',(ptclus(i),i=3,min(nexternal,11))
        npart = nexternal+nres

      return
      end
