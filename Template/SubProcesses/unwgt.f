      subroutine Zoom_Event(wgt,p)
C**************************************************************************
C     Determines if region needs to be investigated in case of large
c     weight events.
C**************************************************************************
      IMPLICIT NONE
c
c     Constant
c
      integer    max_zoom
      parameter (max_zoom=2000)
      include 'genps.inc'
      include 'nexternal.inc'

c
c     Arguments
c
      double precision wgt, p(0:3,nexternal)
c
c     Local
c
      double precision xstore(2),gstore,qstore(2)
      double precision trunc_wgt, xsum, wstore,pstore(0:3,nexternal)
      integer ix, i,j

C     
C     GLOBAL
C
      double precision twgt, maxwgt,swgt(maxevents)
      integer                             lun, nw
      common/to_unwgt/twgt, maxwgt, swgt, lun, nw
      integer nzoom
      double precision  tx(1:3,maxinvar)
      common/to_xpoints/tx, nzoom
      double precision xzoomfact
      common/to_zoom/  xzoomfact
      include 'run.inc'
      include 'coupl.inc'
c
c     DATA
c
      data trunc_wgt /-1d0/
      data xsum/0d0/
      data wstore /0d0/
      save ix, pstore, wstore, xstore, gstore, qstore
c-----
c  Begin Code
c-----
      if (trunc_Wgt .lt. 0d0 .and. twgt .gt. 0d0) then
         write(*,*) 'Selecting zoom level', twgt*500, wgt
      endif
      if (twgt .lt. 0d0) then
         write(*,*) 'Resetting zoom iteration', twgt
         twgt = -twgt
         trunc_wgt = twgt * 500d0
      endif
      if (nw .eq. 0) then
         trunc_wgt = twgt * 500d0
      endif
      trunc_wgt=max(trunc_wgt, twgt*500d0)
      if (nzoom .eq. 0 .and. trunc_wgt .gt. 0d0 ) then
         if (wgt .gt. trunc_wgt) then
            write(*,*) 'Zooming on large event ',wgt / trunc_wgt
            wstore=wgt
            do i=1,nexternal
               do j=0,3
                  pstore(j,i) = p(j,i)
               enddo
            enddo
            do i=1,2
               xstore(i)=xbk(i)
               qstore(i)=q2fact(i)
            enddo
            gstore=g
            xsum = wgt
            nzoom = max_zoom
            wgt=0d0
            ix = 1
         endif
      elseif (trunc_wgt .gt. 0 .and. wgt .gt. 0d0) then
         xsum = xsum + wgt
         if (nzoom .gt. 1) wgt = 0d0
         ix = ix + 1
      endif
      if (xsum .ne. 0d0 .and. nzoom .le. 1) then
         if (wgt .gt. 0d0) then
c            xzoomfact = xsum/real(max_zoom) / wgt !Store avg wgt
            xzoomfact = wstore / wgt  !Store large wgt
         else
            xzoomfact = -xsum/real(max_zoom)
         endif
         wgt = max(xsum/real(max_zoom),trunc_wgt)  !Don't make smaller then truncated wgt
         do i=1,nexternal
            do j=0,3
               p(j,i) = pstore(j,i)
            enddo
         enddo
         do i=1,2
            xbk(i)=xstore(i)
            q2fact(i)=qstore(i)
         enddo
         g=gstore
         write(*,'(a,2e15.3,2f15.3, i8)') 'Stored wgt ',
     $            wgt/trunc_wgt, wstore, wstore/wgt, real(ix)/max_zoom, ix
         trunc_wgt = max(trunc_wgt, wgt)
         xsum = 0d0
         nzoom = 0
      endif
      end

      subroutine clear_events
c-------------------------------------------------------------------
c     delete all events thus far, start from scratch
c------------------------------------------------------------------
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
c
c     Global
c
      integer iseed, nover, nstore
C     
C     GLOBAL
C
      double precision twgt, maxwgt,swgt(maxevents)
      integer                             lun, nw
      common/to_unwgt/twgt, maxwgt, swgt, lun, nw

c-----
c  Begin Code
c-----
c      write(*,*) 'storing Events'
      call store_events
      rewind(lun)
      nw = 0
      maxwgt = 0d0
      end

      SUBROUTINE unwgt(px,wgt,numproc)
C**************************************************************************
C     Determines if event should be written based on its weight
C**************************************************************************
      IMPLICIT NONE
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
c
c     Arguments
c
      double precision px(0:3,nexternal),wgt
      integer numproc
c
c     Local
c
      integer idum, i,j
      double precision uwgt,yran,fudge, p(0:3,nexternal), xwgt
C     
C     GLOBAL
C
      double precision twgt, maxwgt,swgt(maxevents)
      integer                             lun, nw
      common/to_unwgt/twgt, maxwgt, swgt, lun, nw

      double precision    matrix
      common/to_maxmatrix/matrix

      logical               zooming
      common /to_zoomchoice/zooming

c
c     External
c
      real xran1
      external xran1
c
c     Data
c
      data idum/-1/
      data yran/1d0/
      data fudge/10d0/
C-----
C  BEGIN CODE
C-----
      if (twgt .ge. 0d0) then
         do i=1,nexternal
            do j=0,3
               p(j,i)=px(j,i)
            enddo
         enddo
         xwgt = wgt
         if (zooming) call zoom_event(xwgt,P)
         if (xwgt .eq. 0d0) return
         yran = xran1(idum)
         if (xwgt .gt. twgt*fudge*yran) then
            uwgt = max(xwgt,twgt*fudge)
            if (twgt .gt. 0) uwgt=uwgt/twgt/fudge
c            call write_event(p,uwgt)
c            write(29,'(2e15.5)') matrix,wgt
c $B$ S-COMMENT_C $B$
            call write_leshouche(p,uwgt,numproc)
         elseif (xwgt .gt. 0d0 .and. nw .lt. 5) then
            call write_leshouche(p,wgt/twgt*1d-6,numproc)
c $E$ S-COMMENT_C $E$
         endif
         maxwgt=max(maxwgt,xwgt)
      endif
      end

      subroutine store_events()
C**************************************************************************
C     Takes events from scratch file (lun) and writes them to a permanent
c     file  events.dat
C**************************************************************************
      IMPLICIT NONE
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
      double precision trunc_max
      parameter       (trunc_max = 0.01)  !Maximum % cross section to truncate
c
c     Arguments
c
c
c     Local
c
      integer i, lunw, ic(7,2*nexternal-3), n, j
      logical done
      double precision wgt,p(0:4,2*nexternal-3)
      double precision xsec,xerr,xscale,xtot
      double precision xsum, xover
      double precision target_wgt,orig_Wgt(maxevents)
      logical store_event(maxevents)
      integer iseed, nover, nstore
      double precision scale,aqcd,aqed
      integer ievent
      character*140 buff
C     
C     GLOBAL
C
      double precision twgt, maxwgt,swgt(maxevents)
      integer                             lun, nw
      common/to_unwgt/twgt, maxwgt, swgt, lun, nw

      integer                   neventswritten
      common /to_eventswritten/ neventswritten
c      save neventswritten

      integer ngroup
      common/to_group/ngroup

c
c     external
c
      real xran1

      data iseed/0/
      data neventswritten/0/
C-----
C  BEGIN CODE
C-----
c
c     First scale all of the events to the total cross section
c
      if (nw .le. 0) return
      call sample_result(xsec,xerr)
      if (xsec .le. 0) return   !Fix by TS 12/3/2010
      xtot=0
      call dsort(nw, swgt)
      do i=1,nw
         xtot=xtot+swgt(i)
      enddo
c
c     Determine minimum target weight given truncation parameter
c
      xsum = 0d0
      i = nw
      do while (xsum-swgt(i)*(nw-i) .lt. xtot*trunc_max .and. i .gt. 2)
         xsum = xsum + swgt(i)
         i = i-1
      enddo
      if (i .lt. nw) i=i+1
      target_wgt = swgt(i)
c
c     Select events which will be written
c
      xsum = 0d0
      nstore = 0
      rewind(lun)
      done = .false. 
      do i=1,nw
         if (.not. done) then
            call read_event(lun,P,wgt,n,ic,ievent,scale,aqcd,aqed,buff,done)
         else
            wgt = 0d0
         endif
         if (wgt .gt. target_wgt*xran1(iseed)) then
            xsum=xsum+max(wgt,target_Wgt)
            store_event(i)=.true.
            nstore=nstore+1
         else
            store_event(i) = .false.
         endif
      enddo
      xscale = xsec/xsum
      target_wgt = target_wgt*xscale
      rewind(lun)
      if (nstore .le. neventswritten) then
         write(*,*) 'No improvement in events',nstore, neventswritten
         return
      endif
      lunw = 25
      open(unit = lunw, file='events.lhe', status='unknown')
      done = .false.
      i=0      
      xtot = 0
      xover = 0
      nover = 0
      do j=1,nw
         if (.not. done) then
            call read_event(lun,P,wgt,n,ic,ievent,scale,aqcd,aqed,buff,done)
         else
            write(*,*) 'Error done early',j,nw
         endif
         if (store_event(j) .and. .not. done) then
            wgt=wgt*xscale
            wgt = max(wgt, target_wgt)
            if (wgt .gt. target_wgt) then
               xover = xover + wgt - target_wgt
               nover = nover+1
            endif
            xtot = xtot + wgt
            i=i+1
            call write_Event(lunw,p,wgt,n,ic,ngroup,scale,aqcd,aqed,buff)
         endif
      enddo
      write(*,*) 'Found ',nw,' events.'
      write(*,*) 'Wrote ',i ,' events.'
      write(*,*) 'Correct xsec ',xsec
      write(*,*) 'Event xsec ', xtot
      write(*,*) 'Events wgts > 1: ', nover
      write(*,*) '% Cross section > 1: ',xover, xover/xtot*100.
      neventswritten = i
 99   close(lunw)
c      close(lun)
      end

      SUBROUTINE write_leshouche(p,wgt,numproc)
C**************************************************************************
C     Writes out information for event
C**************************************************************************
      IMPLICIT NONE
c
c     Constants
c
      double precision zero
      parameter       (ZERO = 0d0)
      include 'genps.inc'
      include 'nexternal.inc'
      include 'maxamps.inc'
      include 'message.inc'
      include 'cluster.inc'
      include 'run.inc'
c
c     Arguments
c
      double precision p(0:3,nexternal),wgt
      integer numproc
c
c     Local
c
      integer i,j,k
      double precision sum_wgt,sum_wgt2, xtarget,targetamp(maxflow)
      integer ip, np, ic, nc, jpart(7,-nexternal+3:2*nexternal-3)
      integer ida(2),ito(-nexternal+3:nexternal),ns,nres,ires,icloop
      integer iseed
      double precision pboost(0:3),pb(0:4,-nexternal+3:2*nexternal-3),eta
      double precision ptcltmp(nexternal), pdum(0:3)

      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)

      integer isym(nexternal,99), nsym, jsym

      double precision sscale,aaqcd,aaqed
      integer ievent,npart
      logical flip

      real ran1
      external ran1

      character*140 buff

C     
C     GLOBAL
C
      double precision twgt, maxwgt,swgt(maxevents)
      integer                             lun, nw
      common/to_unwgt/twgt, maxwgt, swgt, lun, nw

      integer              IPROC 
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SubProc/ PD, IPROC

      Double Precision amp2(maxamps), jamp2(0:maxflow)
      common/to_amps/  amp2,       jamp2

      character*101       hel_buf
      common/to_helicity/hel_buf

      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig

      integer ngroup
      common/to_group/ngroup

c
c     Data
c
      include 'leshouche.inc'
      data iseed /0/

      double precision pmass(nexternal)
      common/to_mass/  pmass

c      integer ncols,ncolflow(maxamps),ncolalt(maxamps)
c      common/to_colstats/ncols,ncolflow,ncolalt,ic
c      data ncolflow/maxamps*0/
c      data ncolalt/maxamps*0/

      include 'symswap.inc'
      include 'coupl.inc'
C-----
C  BEGIN CODE
C-----
      
      if (nw .ge. maxevents) return
      nw = nw+1
      swgt(nw)=wgt
      sum_wgt=sum_wgt+wgt
      sum_wgt2=sum_wgt2+wgt**2
c
c     First choose a process  iproc comes set to the number of processes
c
      if(ickkw.gt.0)then
        ip = iprocset
      else
        np = iproc
        xtarget=ran1(iseed)*pd(np)
        ip = 1
        do while (pd(ip) .lt. xtarget .and. ip .lt. np)
          ip=ip+1
        enddo
      endif
      
c
c     Now choose a color flow
c
      nc = jamp2(0)
c      ncols=jamp2(0)
      if(nc.gt.0)then
        targetamp(1)=jamp2(1)
        do ic =2,nc
          targetamp(ic) = jamp2(ic)+targetamp(ic-1)
        enddo
        xtarget=ran1(iseed)*targetamp(nc)
        ic = 1
        do while (targetamp(ic) .lt. xtarget .and. ic .lt. nc)
          ic=ic+1
        enddo
c        ncolflow(ic)=ncolflow(ic)+1
      endif
c
c     In case of identical particles symmetry, choose assignment
c
      xtarget = ran1(iseed)*nsym
      jsym = 1
      do while (xtarget .gt. jsym .and. jsym .lt. nsym)
         jsym = jsym+1
      enddo
c
c     Fill jpart color and particle info
c
      do i=1,nexternal
         jpart(1,isym(i,jsym)) = idup(i,ip,numproc)
         jpart(2,isym(i,jsym)) = mothup(1,i)
         jpart(3,isym(i,jsym)) = mothup(2,i)
         jpart(4,isym(i,jsym)) = icolup(1,i,ic,numproc)
         jpart(5,isym(i,jsym)) = icolup(2,i,ic,numproc)
         jpart(6,isym(i,jsym)) = 1
      enddo
      do i=1,nincoming
         jpart(6,isym(i,jsym))=-1
      enddo

c   Set helicities
c      write(*,*) 'Getting helicity',hel_buf(1:50)
      read(hel_buf,'(20i5)') (jpart(7,isym(i, jsym)),i=1,nexternal)
c      write(*,*) 'ihel',jpart(7,1),jpart(7,2)

c   Fix ordering of ptclus
      do i=1,nexternal
        ptcltmp(isym(i,jsym)) = ptclus(i)
      enddo
      do i=1,nexternal
        ptclus(i) = ptcltmp(i)
      enddo

c     Check if we have flipped particle 1 and 2, and flip back
      flip = .false.
      if (p(3,1).lt.0) then
         do j=0,3
            pdum(j)=p(j,1)
            p(j,1)=p(j,2)
            p(j,2)=pdum(j)
         enddo
         flip = .true.
      endif

c
c     Boost momentum to lab frame
c
      pboost(0)=1d0
      pboost(1)=0d0
      pboost(2)=0d0
      pboost(3)=0d0
      if (xbk(2)*xbk(1) .gt. 0d0) then
         eta = sqrt(xbk(1)*ebeam(1)/(xbk(2)*ebeam(2)))
         pboost(0)=p(0,1)*(eta + 1d0/eta)
         pboost(3)=p(0,1)*(eta - 1d0/eta)
      else
         write(*,*) 'Warning bad x1 or x2 in write_leshouche',
     $           xbk(1),xbk(2)
      endif
      do j=1,nexternal
         call boostx(p(0,j),pboost,pb(0,isym(j,jsym)))
c      Add mass information in pb(4)
         pb(4,isym(j,jsym))=pmass(j)
      enddo

c
c     Add info on resonant mothers
c
      call addmothers(ip,jpart,pb,isym,jsym,sscale,aaqcd,aaqed,buff,
     $                npart,numproc)

c     Need to flip after addmothers, since color might get overwritten
      if (flip) then
         do i=1,7
            j=jpart(i,1)
            jpart(i,1)=jpart(i,2)
            jpart(i,2)=j
         enddo
         ptcltmp(1)=ptclus(1)
         ptclus(1)=ptclus(2)
         ptclus(2)=ptcltmp(1)
      endif

c
c     Write events to lun
c
c      write(*,*) 'Writing event'
      if(q2fact(1).gt.0.and.q2fact(2).gt.0)then
         sscale = (q2fact(1)*q2fact(2))**0.25
      else if(q2fact(1).gt.0)then
         sscale = sqrt(q2fact(1))
      else if(q2fact(2).gt.0)then
         sscale = sqrt(q2fact(2))
      else
         sscale = 0d0
      endif
      aaqcd = g*g/4d0/3.1415926d0
      aaqed = gal(1)*gal(1)/4d0/3.1415926d0

      if (btest(mlevel,3)) then
        write(*,*)' write_leshouche: SCALUP to: ',sscale
      endif
      
      
      call write_event(lun,pb(0,1),wgt,npart,jpart(1,1),ngroup,
     &   sscale,aaqcd,aaqed,buff)
      if(btest(mlevel,1))
     &   call write_event(6,pb(0,1),wgt,npart,jpart(1,1),ngroup,
     &   sscale,aaqcd,aaqed,buff)

      end
      
      integer function n_unwgted()
c************************************************************************
c     Determines the number of unweighted events which have been written
c************************************************************************
      implicit none
c
c     Parameter
c
      include 'genps.inc'
      include 'nexternal.inc'
c
c     Local
c
      integer i
      double precision xtot, sum
C     
C     GLOBAL
C
      double precision twgt, maxwgt,swgt(maxevents)
      integer                             lun, nw
      common/to_unwgt/twgt, maxwgt, swgt, lun, nw
c-----
c  Begin Code
c-----

c      write(*,*) 'Sorting ',nw
      if (nw .gt. 1) call dsort(nw,swgt)
      sum = 0d0
      do i=1,nw
         sum=sum+swgt(i)
      enddo
      xtot = 0d0
      i = nw
      do while (xtot .lt. sum/100d0 .and. i .gt. 2)    !Allow for 1% accuracy
         xtot = xtot + swgt(i)
         i=i-1
      enddo
      if (i .lt. nw) i = i+1
c      write(*,*) 'Found ',nw,' events'
c      write(*,*) 'Integrated weight',sum
c      write(*,*) 'Maximum wgt',swgt(nw), swgt(i)
c      write(*,*) 'Average wgt', sum/nw
c      write(*,*) 'Unweight Efficiency', (sum/nw)/swgt(i)
      n_unwgted = sum/swgt(i)
c      write(*,*) 'Number unweighted ',sum/swgt(i), nw
      if (nw .ge. maxevents) n_unwgted = -sum/swgt(i)
      end


      subroutine dsort(n,ra)
      integer n
      double precision ra(n)

      l=n/2+1
      ir=n
10    continue
        if(l.gt.1)then
          l=l-1
          rra=ra(l)
        else
          rra=ra(ir)
          ra(ir)=ra(1)
          ir=ir-1
          if(ir.eq.1)then
            ra(1)=rra
            return
          endif
        endif
        i=l
        j=l+l
20      if(j.le.ir)then
          if(j.lt.ir)then
            if(ra(j).lt.ra(j+1))j=j+1
          endif
          if(rra.lt.ra(j))then
            ra(i)=ra(j)
            i=j
            j=j+j
          else
            j=ir+1
          endif
        go to 20
        endif
        ra(i)=rra
      go to 10
      end
