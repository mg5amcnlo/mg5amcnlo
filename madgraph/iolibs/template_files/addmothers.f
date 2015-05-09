      subroutine addmothers(ip,jpart,pb,isym,jsym,rscale,aqcd,aqed,buff,
     $                      npart,numproc,flip)

      implicit none
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'maxamps.inc'
      include 'cluster.inc'
      include 'message.inc'
      include 'run.inc'

      integer jpart(7,-nexternal+3:2*nexternal-3),npart,ip,numproc
      double precision pb(0:4,-nexternal+3:2*nexternal-3)
      double precision rscale,aqcd,aqed,targetamp(maxflow)
      character*1000 buff
      character*20 cform
      logical flip ! If .true., initial state is mirrored

      integer isym(nexternal,99), jsym
      integer i,j,k,ida(2),ns,nres,ires,icl,ito2,idenpart,nc,ic
      integer mo_color,da_color(2),itmp
      integer ito(-nexternal+3:nexternal),iseed,maxcolor,maxorg
      integer icolalt(2,-nexternal+2:2*nexternal-3)
      double precision qicl(-nexternal+3:2*nexternal-3), factpm
      double precision xtarget
      data iseed/0/
      integer lconfig,idij(-nexternal+2:nexternal)

      integer diag_number
      common/to_diag_number/diag_number

c     Variables for combination of color indices (including multipart. vert)
      integer maxcolmp
      parameter(maxcolmp=20)
      integer ncolmp,icolmp(2,maxcolmp),is_colors(2,nincoming)

      double precision ZERO
      parameter (ZERO=0d0)
      double precision prmass(-nexternal:0,lmaxconfigs)
      double precision prwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      logical first_time,tchannel
      save prmass,prwidth,pow
      data first_time /.true./

      Double Precision amp2(maxamps), jamp2(0:maxflow)
      common/to_amps/  amp2,       jamp2

      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig
      integer idmap(-nexternal:nexternal),icmp

      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest
      integer sprop(maxsproc,-max_branch:-1,lmaxconfigs)
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
      CHARACTER temp*600,temp0*7,integ*1,float*18
      CHARACTER(LEN=45*nexternal) ptclusstring

C     iproc has the present process number
      integer imirror, iproc
      common/to_mirror/imirror, iproc
      data iproc/1/

c      integer ncols,ncolflow(maxamps),ncolalt(maxamps),icorg
c      common/to_colstats/ncols,ncolflow,ncolalt,icorg

      double precision pt
      integer get_color,elim_indices,set_colmp,fix_tchannel_color,combid
      real ran1
      external pt,ran1,get_color,elim_indices,set_colmp,fix_tchannel_color

      if (first_time) then
         include 'props.inc'
         first_time=.false.
      endif

      npart = nexternal
      buff = ' '

      do i=-nexternal+2,nexternal
         icolalt(1,i)=0
         icolalt(2,i)=0
      enddo

c   
c   Choose the config (diagram) which was actually used to produce the event
c   
c   ...unless the diagram is passed in igraphs(1); then use that diagram
      lconfig=iconfig
      if (ickkw.gt.0) then
         if (btest(mlevel,3)) then
            write(*,*)'unwgt.f: write out diagram ',igraphs(1)
         endif
         lconfig=igraphs(1)
      endif
      
c
c    Choose a color flow which is certain to work with the propagator
c    structure of the chosen diagram and use that as an alternative
c   

      nc = int(jamp2(0))
      maxcolor=0
      if(nc.gt.0)then
      if(icolamp(1,%(iconfig)s,iproc)) then
        targetamp(1)=jamp2(1)
c        print *,'Color flow 1 allowed for config ',lconfig
      else
        targetamp(1)=0d0
      endif
      do ic =2,nc
        if(icolamp(ic,%(iconfig)s,iproc))then
          targetamp(ic) = jamp2(ic)+targetamp(ic-1)
c          print *,'Color flow ',ic,' allowed for config ',lconfig,targetamp(ic)
        else
          targetamp(ic)=targetamp(ic-1)
        endif
      enddo
c     ensure that at least one leading color is different of zero if not allow
c     all subleading color. 
      if (targetamp(nc).eq.0)then
       targetamp(1)=jamp2(1)
       do ic =2,nc
           targetamp(ic) = jamp2(ic)+targetamp(ic-1)
       enddo
      endif


      xtarget=ran1(iseed)*targetamp(nc)

      ic = 1
      do while (targetamp(ic) .lt. xtarget .and. ic .lt. nc)
         ic=ic+1
      enddo
      if(targetamp(nc).eq.0) ic=0
c      print *,'Chose color flow ',ic
      do i=1,nexternal
         if(ic.gt.0) then
            icolalt(1,isym(i,jsym))=icolup(1,i,ic,numproc)
            icolalt(2,isym(i,jsym))=icolup(2,i,ic,numproc)
c            write(*,*) i, icolalt(1,isym(i,jsym)), icolalt(2,isym(i,jsym))
            if (abs(icolup(1,i,ic, numproc)).gt.maxcolor) maxcolor=icolup(1,i,ic, numproc)
            if (abs(icolup(2,i,ic, numproc)).gt.maxcolor) maxcolor=icolup(2,i,ic, numproc)
         endif
      enddo
      else ! nc.gt.0

      do i=1,nexternal
         icolalt(1,i)=0
         icolalt(2,i)=0
      enddo

      endif ! nc.gt.0

c     Store original maxcolor to know if we have epsilon vertices
        maxorg=maxcolor
c     Keep track of IS colors that go through to final state
c     (since we shouldn't replace pop-up indices with those)
        do i=1,nincoming
           do j=1,2
              is_colors(j,i)=0
              do k=3,nexternal
                 if (iabs(icolalt(j,i)).eq.iabs(icolalt(j,k))) then
c                This color is going through to FS
                    is_colors(j,i)=iabs(icolalt(j,i))
                 endif
              enddo
           enddo
        enddo
c     
c     Get mother information from chosen graph
c     

c     Set idij for external particles (needed to keep track of BWs)
        if(ickkw.gt.0) then
           do i=1,nexternal
              idij(i)=ishft(1,i-1)
           enddo
        endif

c     First check number of resonant s-channel propagators
        ns=0
        nres=0
        tchannel=.false.
c     Ensure that mother-daughter information starts from 0
        do i=-nexternal+3,0
           jpart(2,i) = 0
           jpart(3,i) = 0
        enddo
 

c     Loop over propagators to find mother-daughter information
        do i=-1,-nexternal+2,-1
c       Daughters
          if(i.gt.-nexternal+2)then
             ida(1)=iforest(1,i,lconfig)
             ida(2)=iforest(2,i,lconfig)
             do j=1,2
                if(ida(j).gt.0) ida(j)=isym(ida(j),jsym)
             enddo
c            Set idij (needed to keep track of BWs)
             if(ickkw.gt.0) idij(i)=combid(idij(ida(1)),idij(ida(2)))
          endif
c       Decide s- or t-channel
          if(i.gt.-nexternal+2.and.
     $         iabs(sprop(numproc,i,lconfig)).gt.0) then ! s-channel propagator
            jpart(1,i)=sprop(numproc,i,lconfig)
            ns=ns+1
          else if(nres.gt.0.and.maxcolor.gt.maxorg) then
c         For t-channel propagators, just check that the colors are ok
             if(i.eq.-nexternal+2) then
c            This is the final t-channel, combining with leg 2
                mo_color=0
                if(.not.tchannel)then
c                  There are no previous t-channels, so this is a combination of
c                  2, 1 and the last s-channel
                   ida(1)=1
                   ida(2)=i+1
                else
c                  The daughter info is in iforest
                   ida(1)=iforest(1,i,lconfig)
                   ida(2)=iforest(2,i,lconfig)
                endif
c            Reverse colors of t-channels to get right color ordering
                ncolmp=0
                ncolmp=set_colmp(ncolmp,icolmp,2,jpart,
     $               iforest(1,-max_branch,lconfig),icolalt,
     $               icolalt(2,2),icolalt(1,2))
             else
                jpart(1,i)=tprid(i,lconfig)
                mo_color=get_color(jpart(1,i))
                ncolmp=0
             endif
             if(mo_color.gt.1.and.
     $            mo_color.ne.3.and.mo_color.ne.8)then
                da_color(1)=get_color(jpart(1,ida(1)))
                da_color(2)=get_color(jpart(1,ida(2)))
                call write_error(da_color(1), da_color(2), mo_color)
             endif
c            Set icolmp for daughters
             ncolmp=set_colmp(ncolmp,icolmp,ida(2),jpart,
     $            iforest(1,-max_branch,lconfig),icolalt,
     $            icolalt(1,ida(2)),icolalt(2,ida(2)))
c            Reverse colors of t-channels to get right color ordering
             ncolmp=set_colmp(ncolmp,icolmp,ida(1),jpart,
     $            iforest(1,-max_branch,lconfig),icolalt,
     $            icolalt(2,ida(1)),icolalt(1,ida(1)))
c            Fix t-channel color
c             print *,'t-channel: ',i,ida(1),ida(2),mo_color
c             print *,'colors: ',((icolmp(j,k),j=1,2),k=1,ncolmp)
             maxcolor=fix_tchannel_color(mo_color,maxcolor,
     $                        ncolmp,icolmp,i,icolalt,is_colors)
             tchannel=.true.
             cycle
          else
            goto 100             
          endif
c       Set status codes for propagator
c          if((igscl(0).ne.0.and.
c     $       (iabs(jpart(1,i)).gt.5.and.iabs(jpart(1,i)).lt.11).or.
c     $       (iabs(jpart(1,i)).gt.16.and.iabs(jpart(1,i)).ne.21)).or.
c     $       (igscl(0).eq.0.and.OnBW(i))) then 
          if(ickkw.eq.0.and.OnBW(i))then
c         Resonance whose mass should be preserved
            jpart(6,i)=2
            nres=nres+1
          else if (ickkw.gt.0) then
             if(isbw(idij(i))) then 
c         Resonance whose mass should be preserved
                jpart(6,i)=2
                nres=nres+1
             else
                jpart(6,i)=3
             endif
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
c     $       abs(pb(4,i)-prmass(i,lconfig)).gt.5d0*prwidth(i,lconfig)) then
c            jpart(6,i)=3
c            nres=nres-1
c          endif
c       Set color info for all s-channels
          mo_color = get_color(jpart(1,i))
c     If inside multipart. vertex (indicated by color 2) cycle
c         Set tentative mothers
          jpart(2,i) = 1
          jpart(3,i) = 2
c         Set mother info for daughters
          do j=1,2
            jpart(2,ida(j)) = i
            jpart(3,ida(j)) = i
          enddo
          if(mo_color.eq.2) cycle
c     Reset list of color indices
          ncolmp=0
c     Add new color indices to list of color indices
          do j=1,2
             ncolmp=set_colmp(ncolmp,icolmp,ida(j),jpart,
     $            iforest(1,-max_branch,lconfig),icolalt,
     $            icolalt(1,ida(j)),icolalt(2,ida(j)))
          enddo
c          print *,'s-channel: ',i,mo_color,ida(1),ida(2)
c          print *,'colors: ',((icolmp(j,k),j=1,2),k=1,ncolmp)
          if(mo_color.eq.1) then ! color singlet
             maxcolor=elim_indices(0,0,ncolmp,icolmp,i,icolalt,
     $            is_colors,maxcolor)
          elseif(mo_color.eq.-3) then ! color anti-triplet
             maxcolor=elim_indices(0,1,ncolmp,icolmp,i,icolalt,
     $            is_colors,maxcolor)
          elseif(mo_color.eq.3) then ! color triplet
             maxcolor=elim_indices(1,0,ncolmp,icolmp,i,icolalt,
     $            is_colors,maxcolor)
          elseif(mo_color.eq.-6) then ! color anti-sextet
             maxcolor=elim_indices(0,2,ncolmp,icolmp,i,icolalt,
     $            is_colors,maxcolor)
          elseif(mo_color.eq.6) then ! color sextet
             maxcolor=elim_indices(2,0,ncolmp,icolmp,i,icolalt,
     $            is_colors,maxcolor)
          elseif(mo_color.eq.8) then ! color octet
             maxcolor=elim_indices(1,1,ncolmp,icolmp,i,icolalt,
     $            is_colors,maxcolor)
          else ! 2 indicates multipart. vertex
             da_color(1) = get_color(jpart(1,ida(1)))
             da_color(2) = get_color(jpart(1,ida(2)))
             call write_error(da_color(1), da_color(2), mo_color)
          endif
c       Just zero helicity info for intermediate states
          jpart(7,i) = 0
        enddo                   ! do i
 100    continue
        call check_pure_internal_flow(icolalt,jpart, maxcolor)


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
c
c     Set correct mother number for clustering info
c
        if (icluster(1,1).ne.0) then
           do i=1,nexternal-2
              if(icluster(4,i).gt.0)then
                 icluster(4,i)=ito(icluster(4,i))
              else
                 icluster(4,i)=-1
              endif
              if(icluster(3,i).eq.0)then
                 icluster(3,i)=-1
              endif
              if(ito(icluster(1,i)).gt.0)
     $             icluster(1,i)=ito(icluster(1,i))
              if(ito(icluster(2,i)).gt.0)
     $             icluster(2,i)=ito(icluster(2,i))
              if(flip)then
                 if(icluster(1,i).le.2)
     $             icluster(1,i)=3-icluster(1,i)
                 if(icluster(2,i).le.2)
     $             icluster(2,i)=3-icluster(2,i)
                 if(icluster(3,i).ge.1.and.icluster(3,i).le.2)
     $             icluster(3,i)=3-icluster(3,i)
              endif
           enddo
        endif

        if (flip) then
c       Need to flip initial state color, since might be overwritten
           do i=1,7
              j=jpart(i,1)
              jpart(i,1)=jpart(i,2)
              jpart(i,2)=j
           enddo
        endif

        if(ickkw.gt.0) then
            if (lhe_version.lt.3d0) then
              write(cform,'(a4,i2,a6)') '(a1,',max(nexternal,10),'e15.7)'
              write(buff,cform) '#',(ptclus(i),i=3,nexternal)
           else if(nexternal.gt.2)then
              temp0='<scales '
              temp=''	
              do i=3,nexternal
                 integ=''
                 float=''
                 write(integ,'(i1)') i-2
                 Write(float,'(f16.5)') ptclus(i)
                 temp=trim(temp)//' pt_clust_'//integ//'="'//trim(adjustl(float))//'"'
              enddo
              ptclusstring=trim(adjustl(temp0//trim(temp)//'></scales>'))
c             write(*,*)'WRITING THE ptclusscale:',trim(adjustl(ptclusstring))
              write(buff,'(a)') trim(adjustl(ptclusstring))
           endif
        endif

        npart = nexternal+nres

      return
      end

c     *************************************
      subroutine write_error(ida1,ida2,imo)
c     *************************************
      implicit none
      integer ida1,ida2,imo

      open(unit=26,file='../../../error',status='unknown',err=999)
      if (ida1.eq.1000)then
         write(26,*) 'Error: too many particles in multipart. vertex,',
     $        ' please increase maxcolmp in addmothers.f'
         write(*,*) 'Error: too many particles in multipart. vertex,',
     $        ' please increase maxcolmp in addmothers.f'
         stop
      endif
      if (ida1.eq.1001)then
         write(26,*) 'Error: failed to reduce to color indices: ',ida2,imo
         write(*,*) 'Error: failed to reduce to color indices: ',ida2,imo
         stop
      endif
      write(26,*) 'Error: Color combination ',ida1,ida2,
     $     '->',imo,' not implemented in addmothers.f'
      write(*,*) 'Error: Color combination ',ida1,ida2,
     $     '->',imo,' not implemented in addmothers.f'
      stop

 999  write(*,*) 'error'
      end

c     *********************************************************************
      function set_colmp(ncolmp,icolmp,npart,jpart,forest,icol,icol1,icol2)
c     *********************************************************************
      implicit none
      integer maxcolmp
      parameter(maxcolmp=20)
      include 'nexternal.inc'
      include 'genps.inc'
c     Arguments
      integer set_colmp
      integer ncolmp,icolmp(2,*),npart,icol1,icol2
      integer icol(2,-nexternal+2:2*nexternal-3)
      integer jpart(7,-nexternal+3:2*nexternal-3)
      integer forest(2,-max_branch:-1)
c     Local
      integer da_color(2),itmp,ida(2),icolor,ipart,i,j
      integer get_color,set1colmp

      set_colmp=ncolmp
      icolor=get_color(jpart(1,npart))
      if(icolor.eq.1) return
      if(icolor.eq.2) then
c     Multiparticle vertex - need to go through daughters and collect all colors
         ipart=npart
        do while(icolor.eq.2)
          ida(1)=forest(1,ipart)
          ida(2)=forest(2,ipart)
          da_color(1)=get_color(jpart(1,ida(1)))
          da_color(2)=get_color(jpart(1,ida(2)))
c          print *,'iforest: ',ipart,ida(1),ida(2),da_color(1),da_color(2)
          if(da_color(1).ne.2.and.da_color(2).lt.da_color(1).or.
     $         da_color(2).eq.2)then
c            Order daughters according to color, but always color 2 first
             itmp=ida(1)
             ida(1)=ida(2)
             ida(2)=itmp
             itmp=da_color(1)
             da_color(1)=da_color(2)
             da_color(2)=itmp
          endif
          do i=1,2
             if(da_color(i).ne.1.and.da_color(i).ne.2)then
                ncolmp=set1colmp(ncolmp,icolmp,icol(1,ida(i)),
     $               icol(2,ida(i)))
             endif
          enddo
          icolor=da_color(1)
          ipart=ida(1)
        enddo
      else
         ncolmp=set1colmp(ncolmp,icolmp,icol1,icol2)
      endif
      set_colmp=ncolmp
      return
      end

c     ******************************************************
      function set1colmp(ncolmp,icolmp,icol1,icol2)
c     ******************************************************
      implicit none
c     Arguments
      integer maxcolmp
      parameter(maxcolmp=20)
      integer set1colmp
      integer ncolmp,icolmp(2,*),icol1,icol2,i,j

c      print *,'icol1,icol2: ',icol1,icol2

      ncolmp=ncolmp+1
      icolmp(1,ncolmp)=icol1
      icolmp(2,ncolmp)=icol2
c     Avoid color sextet-type negative indices
      if(icolmp(1,ncolmp).lt.0)then
         ncolmp=ncolmp+1
         icolmp(2,ncolmp)=-icolmp(1,ncolmp-1)
         icolmp(1,ncolmp-1)=0
         icolmp(1,ncolmp)=0
      elseif(icolmp(2,ncolmp).lt.0)then
         ncolmp=ncolmp+1
         icolmp(1,ncolmp)=-icolmp(2,ncolmp-1)
         icolmp(2,ncolmp-1)=0
         icolmp(2,ncolmp)=0
      endif
c      print *,'icolmp: ',((icolmp(i,j),i=1,2),j=1,ncolmp)
      if(ncolmp.gt.maxcolmp)
     $     call write_error(1000,ncolmp,maxcolmp)
      set1colmp=ncolmp
      return
      end

c********************************************************************
      function fix_tchannel_color(mo_color,maxcolor,ncolmp,icolmp,ires,
     $                            icol,is_colors)
c********************************************************************
c     Successively eliminate identical pairwise color indices from the
c     icolmp list, until only (max) one triplet and one antitriplet remains
c

      implicit none
      include 'nexternal.inc'
      integer fix_tchannel_color
      integer mo_color,maxcolor,ncolmp,icolmp(2,*)
      integer ires,icol(2,-nexternal+2:2*nexternal-3)
      integer is_colors(2,nincoming)
      integer i,j,i3,i3bar,max3,max3bar,min3,min3bar,maxcol,mincol
      integer count

c     Successively eliminate color indices in pairs until only the wanted
c     indices remain
      do i=1,ncolmp
         do j=1,ncolmp
            if(icolmp(1,i).ne.0.and.icolmp(1,i).eq.icolmp(2,j)) then
               icolmp(1,i)=0
               icolmp(2,j)=0
            endif
         enddo
      enddo
      
      i3=0
      i3bar=0
      icol(1,ires)=0
      icol(2,ires)=0
      do i=1,ncolmp
         if(icolmp(1,i).gt.0)then
            i3=i3+1
c           color for t-channels needs to be reversed
            if(i3.eq.1) icol(2,ires)=icolmp(1,i)
            if(i3.eq.2) icol(1,ires)=-icolmp(1,i)
         endif
         if(icolmp(2,i).gt.0)then
            i3bar=i3bar+1
c           color for t-channels needs to be reversed
            if(i3bar.eq.1) icol(1,ires)=icolmp(2,i)
            if(i3bar.eq.2) icol(2,ires)=-icolmp(2,i)
         endif
      enddo

      if(mo_color.eq.0)then
         icol(1,ires)=0
         icol(2,ires)=0
      endif

      fix_tchannel_color=maxcolor
      if(mo_color.le.1.and.i3.eq.0.and.i3bar.eq.0) return
      if(mo_color.eq.3.and.(i3.eq.1.and.i3bar.eq.0
     $     .or.i3bar.eq.1.and.i3.eq.0)) return
      if(mo_color.eq.8.and.i3.eq.1.and.i3bar.eq.1) return

c     Make sure that max and min don't come from the same octet
      call find_max_min(icolmp,ncolmp,max3,min3,max3bar,min3bar,
     $     i3,i3bar,is_colors)
c      print *,'After finding: ',ncolmp,((icolmp(j,i),j=1,2),i=1,ncolmp)
c      print *,'mo_color = ',mo_color

      if(mo_color.le.1.and.i3-i3bar.eq.2.or.
     $   mo_color.le.1.and.i3bar-i3.eq.2.or.
     $   mo_color.le.1.and.i3.eq.1.and.i3bar.eq.1) then
c     Replace the maximum index with the minimum one everywhere
         maxcol=max(max3,max3bar)
         if(maxcol.eq.max3) then
            mincol=min3bar
         else
            mincol=min3
         endif
         do i=ires+1,-1
            do j=1,2
               if(icol(j,i).eq.maxcol)
     $              icol(j,i)=mincol
            enddo
         enddo
c         print *,'Replaced ',maxcol,' by ',mincol
      elseif(mo_color.le.1.and.i3.eq.2.and.i3bar.eq.2) then
c     Ensure that max > min
         if(max3bar.lt.min3)then
            i=min3
            min3=max3bar
            max3bar=i
         endif
         if(max3.lt.min3bar)then
            i=min3bar
            min3bar=max3
            max3=i
         endif
c     Replace the maximum indices with the minimum ones everywhere
         do i=ires+1,-1
            do j=1,2
               if(icol(j,i).eq.max3bar)
     $              icol(j,i)=min3
               if(icol(j,i).eq.max3)
     $              icol(j,i)=min3bar
            enddo
         enddo
c         print *,'Replaced ',max3bar,' by ',min3,' and ',max3,' by ',min3bar
      elseif(mo_color.le.1.and.mod(i3,3).eq.0.and.mod(i3bar,3).eq.0)then
c     This is epsilon index - do nothing
         continue
      else if(mo_color.eq.3.and.mod(i3-i3bar,3).eq.2) then
c     This is an epsilon index
         maxcolor=maxcolor+1
         icol(1,ires)=maxcolor
         icol(2,ires)=0
c         print *,'Set mother color for ',ires,' to ',(icol(j,ires),j=1,2)
      else if(mo_color.eq.3.and.mod(i3bar-i3,3).eq.2) then
c     This is an epsilon index
         maxcolor=maxcolor+1
         icol(1,ires)=0
         icol(2,ires)=maxcolor
c         print *,'Set mother color for ',ires,' to ',(icol(j,ires),j=1,2)
      else if(mo_color.eq.3.and.(i3-i3bar.eq.1.or.i3bar-i3.eq.1).or.
     $        mo_color.eq.8.and.i3.eq.2.and.i3bar.eq.2) then
c     Replace the maximum index with the minimum one everywhere
c     (we don't know if we should replace i3 with i3bar or vice versa)
c     Actually we know if one of the index is repeated (we do not want to replace that one)
         maxcol=max(max3,max3bar)
         if(maxcol.eq.max3) then
            mincol=min3bar
         else
            mincol=min3
         endif
         do i=ires+1,-1
            do j=1,2
               if(icol(j,i).eq.maxcol)
     $              icol(j,i)=mincol
            enddo
         enddo

         if (mincol.eq.0.and.mo_color.eq.3) then
c            situation like (possible if they are epsilon in the gluon decay
c            (503,0)----------+ggggggggggggg (509,508)
c                             |
c                             |(x,y)
c            in this case maxcol=509 and mincol=0
c            The correct solution in this case is:
c            (503,0)----------+ggggggggggggg (503,508)
c                             |
c                             |(0,508)
            if(icolmp(2,1).eq.0)then
               maxcol = icolmp(1,2)
               mincol = icolmp(1,1)
               icol(1,ires) = 0
               icol(2,ires) = icolmp(2,2)
            elseif(icolmp(1,1).eq.0)then
               maxcol = icolmp(2,2)
               mincol = icolmp(2,1)
               icol(1,ires) = icolmp(1,2)
               icol(2,ires) = 0 
            elseif(icolmp(2,2).eq.0)then
               maxcol = icolmp(1,1)
               mincol = icolmp(1,2)
               icol(1,ires) = 0
               icol(2,ires) = icolmp(2,1)
            elseif(icolmp(1,2).eq.0)then
               maxcol = icolmp(2,1)
               mincol = icolmp(2,2)
               icol(1,ires) = icolmp(1,1)
               icol(2,ires) = 0
            else
               !should not happen
               write(*,*) "weird color combination in addmothers.f"
               write(*,*) icolmp(1,1), icolmp(2,1), icolmp(1,2), icolmp(2,2)
               call write_error(1001,0,0) 
            endif
c           now maxcol=509 and mincol=503 so replace all occurence of 509-> 503
c            print *,'Replaced ',maxcol,' by ',mincol
            do i=ires+1,nexternal
               do j=1,2
                  if(icol(j,i).eq.maxcol)
     $                 icol(j,i)=mincol
               enddo
            enddo
         else
c        standard case
c     First check that mincol is not a going trough index. If it is it 
C     should not be assign to one of the temporary index
            count=0
            do i=1,nexternal
               do j=1,2
                  if (icol(j,i).eq.mincol) count = count +1
               enddo
            enddo

            if(count.eq.2)then
c     we do not want to use that index pass to the other one
               if (mincol.eq.min3)then
                  mincol = min3bar
                  maxcol = max3
               else
                  mincol = min3
                  maxcol = max3bar
               endif
            endif

c     Fix the color for ires (remember 3<->3bar for t-channels)
            icol(1,ires)=0
            icol(2,ires)=0
c         print *,'Replaced ',maxcol,' by ',mincol
            if(max3.eq.maxcol)then
               if(i3-i3bar.ge.0) icol(2,ires)=min3
               if(i3bar-i3.ge.0) icol(1,ires)=max3bar
            else
               if(i3-i3bar.ge.0) icol(2,ires)=max3
               if(i3bar-i3.ge.0) icol(1,ires)=min3bar
            endif
         endif
c     print *,'Set mother color for ',ires,' to ',(icol(j,ires),j=1,2)
      else
c     Don't know how to deal with this
         call write_error(i3,i3bar,mo_color)
      endif
      fix_tchannel_color=maxcolor
      
      return
      end

c*******************************************************************
      function elim_indices(n3,n3bar,ncolmp,icolmp,ires,icol,
     $     is_colors,maxcolor)
c*******************************************************************
c     Successively eliminate identical pairwise color indices from the
c     icolmp list, until only the wanted indices remain
c     n3 gives the number of triplet indices, n3bar number of antitriplets
c     n3=1 for triplet, n3bar=1 for antitriplet, 
c     (n3,n3bar)=(1,1) for octet,
c     n3=2 for sextet, n3bar=2 for antisextet 
c     If there are epsilon^{ijk} or epsilonbar color couplings, we
c     need to introduce new index based on maxcolor.
c

      implicit none
      include 'nexternal.inc'
      integer elim_indices
      integer n3,n3bar,ncolmp,icolmp(2,*),maxcolor
      integer ires,icol(2,-nexternal+2:2*nexternal-3)
      integer is_colors(2,nincoming)
      integer i,j,i3,i3bar

c     Successively eliminate color indices in pairs until only the wanted
c     indices remain
      do i=1,ncolmp
         do j=1,ncolmp
            if(icolmp(1,i).ne.0.and.icolmp(1,i).eq.icolmp(2,j)) then
               icolmp(1,i)=0
               icolmp(2,j)=0
            endif
         enddo
      enddo
      
      i3=0
      i3bar=0
      icol(1,ires)=0
      icol(2,ires)=0
      do i=1,ncolmp
         if(icolmp(1,i).gt.0)then
            i3=i3+1
            if(i3.eq.1) icol(1,ires)=icolmp(1,i)
            if(i3.eq.2) icol(2,ires)=-icolmp(1,i)
         endif
         if(icolmp(2,i).gt.0)then
            i3bar=i3bar+1
            if(i3bar.eq.1) icol(2,ires)=icolmp(2,i)
            if(i3bar.eq.2) icol(1,ires)=-icolmp(2,i)
         endif
      enddo

c      print *,'i3,n3,i3bar,n3bar: ',i3,n3,i3bar,n3bar
c      print *,'icol(1,ires),icol(2,ires): ',icol(1,ires),icol(2,ires)

      if(n3bar.le.1.and.n3.eq.0) icol(1,ires)=0
      if(n3.le.1.and.n3bar.eq.0) icol(2,ires)=0

      if(i3.ne.n3.or.i3bar.ne.n3bar) then
         if(n3.gt.0.and.n3bar.eq.0.and.mod(i3bar+n3,3).eq.0.and.i3.eq.0)then
c        This is an epsilon index interaction
c            write(*,*) i3, n3, i3bar, n3bar, ires
            maxcolor=maxcolor+1
            icol(1,ires)=maxcolor
            if(n3.eq.2)then
               maxcolor=maxcolor+1
               icol(2,ires)=-maxcolor
            endif
         elseif(n3bar.gt.0.and.n3.eq.0.and.mod(i3+n3bar,3).eq.0.and.i3bar.eq.0)then
c        This is an epsilonbar index interaction
c            write(*,*) i3, n3, i3bar, n3bar, ires
            maxcolor=maxcolor+1
            icol(2,ires)=maxcolor
            if(n3.eq.2)then
               maxcolor=maxcolor+1
               icol(1,ires)=-maxcolor
            endif
         elseif(n3.gt.0.and.n3bar.eq.0.and.i3-i3bar.eq.n3.or.
     $          n3bar.gt.0.and.n3.eq.0.and.i3bar-i3.eq.n3bar.or.
     $          n3.eq.1.and.n3bar.eq.1.and.i3-i3bar.eq.0.or.
     $          n3.eq.0.and.n3bar.eq.0.and.i3-i3bar.eq.0.or.
     $      n3bar.gt.0.and.n3.eq.0.and.mod(i3+n3bar,3).eq.0.and.i3bar.ne.0.or.
     $      n3.gt.0.and.n3bar.eq.0.and.mod(i3bar+n3,3).eq.0.and.i3.ne.0)then
c        We have a previous epsilon which gives the wrong pop-up index
            call fix_s_color_indices(n3,n3bar,i3,i3bar,ncolmp,icolmp,
     $           ires,icol,is_colors)
         else
c           Don't know how to deal with this
            call write_error(1001,n3,n3bar)
         endif
      endif

      elim_indices=maxcolor
      
      return
      end

c*******************************************************************
      subroutine fix_s_color_indices(n3,n3bar,i3,i3bar,ncolmp,icolmp,
     $                             ires,icol,is_colors)
c*******************************************************************
c
c     Fix color flow if some particle has got the wrong pop-up color
c     due to epsilon-ijk vertices
c

      implicit none
      include 'nexternal.inc'
      integer n3,n3bar,ncolmp,icolmp(2,*),maxcolor
      integer ires,icol(2,-nexternal+2:2*nexternal-3)
      integer is_colors(2,nincoming)
      integer i,j,i3,i3bar
      integer max_n3,max_n3bar,min_n3,min_n3bar,maxcol,mincol

      icol(1,ires)=0
      icol(2,ires)=0
      
c      print *,'Colors: ',ncolmp,((icolmp(j,i),j=1,2),i=1,ncolmp)
c      print *,'n3,n3bar,i3,i3bar: ',n3,n3bar,i3,i3bar

c     Make sure that max and min don't come from the same octet
      call find_max_min(icolmp,ncolmp,max_n3,min_n3,
     $                   max_n3bar,min_n3bar,i3,i3bar,is_colors)
c      print *,'max3,min3bar,min3,max3bar: ',max_n3,min_n3bar,min_n3,max_n3bar

      if(n3.eq.1.and.n3bar.eq.0.and.i3-i3bar.eq.n3.or.
     $   n3bar.eq.1.and.n3.eq.0.and.i3bar-i3.eq.n3bar.or.
     $   n3bar.eq.1.and.n3.eq.1.and.i3bar-i3.eq.0.or.
     $   n3bar.eq.0.and.n3.eq.0.and.i3bar-i3.eq.0.or.
     $      n3bar.gt.0.and.n3.eq.0.and.mod(i3+n3bar,3).eq.0.and.i3bar.ne.0.or.
     $      n3.gt.0.and.n3bar.eq.0.and.mod(i3bar+n3,3).eq.0.and.i3.ne.0)then

      if ((i3.eq.2.or.i3bar.eq.2).and.(n3bar+n3.eq.1))then
c     Special case:
c     -------------------- (0,503)
c              g
c              g
c              g (504,505)
c
c    need to correct to 
c    -------------------------  (0,503)
c    (0,505)   g
c              g
c              g (503,505)
         if (i3.eq.2) then
            icol(1,ires) = max(icolmp(1,1), icolmp(1,2))
            icol(2,ires) = 0
            maxcol = max(icolmp(2,1), icolmp(2,2))
            mincol = min(icolmp(1,1), icolmp(1,2))
c           replace maxcol by mincol
         elseif (i3bar.eq.2) then
            icol(1,ires) = 0
            icol(2,ires) = max(icolmp(2,1), icolmp(2,2))
            maxcol = max(icolmp(1,1), icolmp(1,2))
            mincol = min(icolmp(2,1), icolmp(2,2))
         endif
c         write(*,*) "replace ", maxcol,"by",mincol
            do i=ires+1,nexternal
               do j=1,2
                  if(icol(j,i).eq.maxcol)
     $                 icol(j,i)=mincol
               enddo
            enddo            
        

      else
c     Replace the highest 3bar-index with the lowest 3-index,
c     or vice versa


         maxcol=max(max_n3,max_n3bar)
         if(maxcol.eq.max_n3) then
            mincol=min_n3bar
         else
            mincol=min_n3
         endif
         do i=ires,-1
            do j=1,2
               if(icol(j,i).eq.maxcol)
     $              icol(j,i)=mincol
            enddo
         enddo
c         print *,'Replaced ',maxcol,' with ',mincol
         if(max_n3.eq.maxcol)then
            if(n3.eq.1) icol(1,ires)=min_n3
            if(n3bar.eq.1) icol(2,ires)=max_n3bar
         else
            if(n3.eq.1) icol(1,ires)=max_n3
            if(n3bar.eq.1) icol(2,ires)=min_n3bar
         endif
c         print *,'Set mother color for ',ires,' to ',(icol(j,ires),j=1,2)
      endif
      else
c     Don't know how to deal with this
         call write_error(1001,n3,n3bar)
      endif       
      return
      end


      subroutine check_pure_internal_flow(icol,jpart, maxcolor)

      implicit none 
      include 'nexternal.inc'
      integer jpart(7,-nexternal+3:2*nexternal-3)
      integer icol(2,-nexternal+2:2*nexternal-3)
      integer maxcolor
      integer i,j,k,l
      logical found

c      do i=-nexternal+3,nexternal
c         write(*,*) i, icol(1,i), icol(2,i),(jpart(j,i) , j=1,3)
c      enddo
      do i=-nexternal+3,-1
         if (jpart(2,i).eq.0.or.jpart(3,i).eq.0) goto 20 ! not define mother -> continue
         if (icol(1,i).eq.1000.or.icol(2,i).eq.1000) goto 20 ! special color value -> continue
         do k = 1,2
            found=.false.
            do j=1,nexternal
               if(icol(k,i).eq.icol(1,j).or.icol(k,i).eq.icol(2,j))then
                  found=.true.
                  goto 10       ! break
               endif
            enddo
 10         continue
            if (.not.found)then
               call correct_external_flow_epsilon(icol, jpart, maxcolor,
     &              icol(k,i))
            endif
         enddo
 20      continue
      enddo
      return 
      end



      subroutine correct_external_flow_epsilon(icol, jpart, maxcolor, mincol)
c
c     for avoiding double epsilon problem
c
      implicit none
      include 'nexternal.inc'
      integer jpart(7,-nexternal+3:2*nexternal-3)
      integer maxcolor
      integer icol(2,-nexternal+2:2*nexternal-3)
      integer i,j,i3
      integer mincol ! the potential propagator linked to the two epsilon.
      integer k,l
      integer potential_index(2)
      integer epsilon_index(4)
      integer mothers(2*nexternal-3)
      logical to_change

C        In presence of two epsilon_ijk  connected by a flow we need to ensure that the 
C        the index of the non summed indices do not repeat each other
         l=0
         do i=-nexternal+3,2*nexternal-3
            if (icol(1,i).eq.mincol.or.icol(2,i).eq.mincol)then
               potential_index(1)=0
c               write(*,*) "particle",i,"has color index", mincol
               k=0 !index to see how many child we found so far
               do j=-nexternal+3,2*nexternal-3
                  if (jpart(2,j).eq.i.or.jpart(3,j).eq.i)then
c                     write(*,*) "find the child", j
                     if (icol(1,j).eq.mincol.or.icol(2,j).eq.mincol)then
                        potential_index(1)=0
c                        write(*,*) "the color", mincol, 
c     &       "is pass to one of the children ->no epsilon at this stage"
c                       the color flow is pass to a child so no need to do anything for this part/junction                        
                        goto 10 ! break
                     elseif(icol(1,j).ne.0) then
c             write(*,*) "child has not colour", mincol, "add", icol(1,j)
                        k = k+1
                        potential_index(k) = icol(1,j)
                        mothers(1) = i
                     elseif(icol(2,j).ne.0)then
c             write(*,*) "child has not colour", mincol, "add", icol(2,j)
                        k = k+1
                        potential_index(k) = icol(2,j)
                        mothers(1) = i 
                     else
                        call write_error(1001,0,0) 
                     endif
                  endif
               enddo
 10            continue
c              store the index of the final junction for this color 
c               write(*,*) "found", potential_index
               if (potential_index(1).ne.0) then
                  l = l+1
                  epsilon_index(l) = potential_index(1)
                  l = l+1
                  epsilon_index(l) = potential_index(2)
               endif
            endif
         enddo
C        Remove the duplication index if any 
         mothers(2) = 0
c        check the mother of mothers1 and change the color index
c        epsilon_index(3) -> maxcolor+1, epsilon_index(4) -> maxcolor+2          
c        then add info on mothers to recursively change
c        Firs check if we have to change  
         to_change = .false.
         do i=3,4
            do j=1,2
               if (epsilon_index(i).eq.epsilon_index(j))then
C     `         The index is duplicated need to change one
                  to_change = .true. 
               endif
            enddo
         enddo
         if (epsilon_index(4).eq.0) to_change = .false.
         if (to_change)then
         k=1
         do i =1, 2*nexternal-3
            if (mothers(i).eq.0)then 
               goto 20 !break
            endif
            do j=mothers(i)+1,2*nexternal-3
               if (jpart(2,j).eq.mothers(i).or.jpart(3,j).eq.mothers(i))then
                  if (icol(1,j).eq.epsilon_index(3))then
                     icol(1,j) = maxcolor + 1
                     k = k+1
                     mothers(k) = j
                     mothers(k+1) = 0
                  elseif (icol(2,j).eq.epsilon_index(3))then
                     icol(2,j) = maxcolor + 1
                     k = k+1
                     mothers(k) = j
                     mothers(k+1) = 0
                  elseif (icol(1,j).eq.epsilon_index(4))then
                     icol(1,j) = maxcolor + 2
                     k = k+1
                     mothers(k) = j
                     mothers(k+1) = 0
                  elseif (icol(2,j).eq.epsilon_index(4))then
                     icol(2,j) = maxcolor + 2
                     k = k+1
                     mothers(k) = j
                     mothers(k+1) = 0
                  endif
               endif
            enddo
         enddo
 20      continue
         maxcolor = maxcolor +2
      endif
      end

c*******************************************************************************
      subroutine find_max_min(icolmp,ncolmp,max3,min3,max3bar,min3bar,
     $     i3,i3bar,is_colors)
c*******************************************************************************
      implicit none
      include 'nexternal.inc'
      integer ncolmp,icolmp(2,*)
      integer is_colors(2,nincoming)
      integer i,j,k,max3,max3bar,min3,min3bar,i3,i3bar,i3now,i3barnow
      integer allpairs(2,nexternal),npairs,maxcol,mincol

      i3now=i3
      i3barnow=i3bar

c     Create all possible pairs (3,3bar) that
c     1. come from different octets, 2. are different, 
c     3. don't contain any color lines passing through the event
      npairs = 0
      do 20 i=1,ncolmp
         if(icolmp(1,i).eq.0) goto 20
         do k=1,nincoming
            if(icolmp(1,i).eq.is_colors(1,k)) goto 20
         enddo
         do 10 j=1,ncolmp
            if(i.eq.j.or.icolmp(2,j).eq.0.or.
     $           icolmp(1,i).eq.icolmp(2,j)) goto 10
            do k=1,nincoming
               if(icolmp(2,j).eq.is_colors(2,k)) goto 10
            enddo
            npairs=npairs+1
            allpairs(1,npairs)=icolmp(1,i)
            allpairs(2,npairs)=icolmp(2,j)
 10      enddo
 20   enddo

c      print *,'is_colors: ',((is_colors(i,j),i=1,2),j=1,nincoming)
c      print *,'pairs: ',((allpairs(i,j),i=1,2),j=1,npairs)

c     Find the pairs with maximum 3 and 3bar
      min3=1000
      min3bar=1000
      max3=0
      max3bar=0
      do i=1,npairs
         if(allpairs(1,i).gt.max3.and.
     $      (allpairs(2,i).lt.max3bar.or.
     $       allpairs(1,i).gt.allpairs(2,i)))then
            max3=allpairs(1,i)
            min3bar=allpairs(2,i)
         else if(allpairs(2,i).gt.max3bar.and.
     $           (allpairs(1,i).lt.max3.or.
     $            allpairs(2,i).gt.allpairs(1,i)))then
            max3bar=allpairs(2,i)
            min3=allpairs(1,i)
         endif
      enddo

c     Find "maximum" pairs with minimum 3 and 3bar
      do i=1,npairs
         if(allpairs(1,i).eq.max3.and.
     $        allpairs(2,i).lt.min3bar.and.
     $        allpairs(2,i).ne.max3bar)
     $        min3bar=allpairs(2,i)
         if(allpairs(2,i).eq.max3bar.and.
     $        allpairs(1,i).lt.min3.and.
     $        allpairs(1,i).ne.max3)
     $        min3=allpairs(1,i)
      enddo

c     Check that the pair are indeed different. Might not be the case if
c     The process contains some epsilon_ijk somewhere else.
      if (i3bar.gt.1.and.i3.gt.1)then
      if (max3bar.eq.min3bar)then
c        try to change min3bar
         min3bar=10000
         do i=1,npairs
c          search a new pair but with a different index!
           if(allpairs(1,i).eq.max3.and.
     $        allpairs(2,i).lt.min3bar.and.
     $        allpairs(2,i).ne.max3bar)then
              min3bar=allpairs(2,i)
              endif
         enddo
c        check if we found a new one. If not try to change the other index (max3bar)
         if (min3bar.eq.10000)then
            min3bar = max3bar
            max3bar = 0
c           search a new pair but with a different index!
            do i=1,npairs
               if(allpairs(1,i).eq.min3.and.
     $              allpairs(2,i).gt.max3bar.and.
     $              allpairs(2,i).ne.min3bar) then
                  max3bar = allpairs(2,i)
               endif
            enddo
c           This should not happen.
            if (max3bar.eq.0)then
               write(*,*) "colorflow problem"
               call write_error(1001,0,0) 
            endif
         endif
      endif
c     Doing the same but for the color index.
      if (max3.eq.min3)then
c        try to change min3
         min3=10000
         do i=1,npairs
           if(allpairs(2,i).eq.max3bar.and.
     $        allpairs(1,i).lt.min3.and.
     $        allpairs(1,i).ne.max3)then
              min3=allpairs(1,i)
              endif
         enddo
         if (min3.eq.10000)then
            min3 = max3 ! restore value
            max3 = 0
c         try to change max3
            do i=1,npairs
               if(allpairs(2,i).eq.min3bar.and.
     $              allpairs(1,i).gt.max3.and.
     $              allpairs(1,i).ne.min3) then
                  max3 = allpairs(1,i)
               endif
            enddo
            if (max3.eq.0)then
               write(*,*) "colorflow problem"
               stop
            endif
         endif
      endif
      endif

      if (max3.gt.0.and.max3bar.gt.0) then
c       We have found our two pairs, so we're done
         return
      endif

      if(max3.gt.0.or.max3bar.gt.0)then
         i3now=i3now-1
         i3barnow=i3barnow-1
      endif

c     Find pair for non-maximum (where we allow all colors)
      maxcol=max(max3,max3bar)
      if(maxcol.eq.max3) then
         mincol=min3bar
      else
         mincol=min3
      endif

      npairs=0
      do i=1,ncolmp
         if(icolmp(1,i).eq.0.and.i3now.gt.0) cycle
         if(icolmp(1,i).eq.maxcol.or.icolmp(1,i).eq.mincol)
     $        cycle
         do j=1,ncolmp
            if(icolmp(2,j).eq.0.and.i3barnow.gt.0) cycle
            if(i.eq.j.or.icolmp(1,i).eq.icolmp(2,j)) cycle
            if(icolmp(2,j).eq.maxcol.or.icolmp(2,j).eq.mincol)
     $           cycle
            npairs=npairs+1
            allpairs(1,npairs)=icolmp(1,i)
            allpairs(2,npairs)=icolmp(2,j)
         enddo
      enddo
      if(npairs.ge.1)then
         if(maxcol.eq.max3)then
            min3=allpairs(1,1)
            max3bar=allpairs(2,1)
         else
            max3=allpairs(1,1)
            min3bar=allpairs(2,1)            
         endif
      endif
      
c      print *,'allpairs: ',((allpairs(i,j),i=1,2),j=1,npairs)

      end
