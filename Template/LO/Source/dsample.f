      subroutine sample_full(ndim,ncall,itmax,itmin,dsig,ninvar,nconfigs)
c**************************************************************************
c     Driver for sample which does complete integration
c     This is done in double precision, and should be told the
c     number of possible phasespace choices.
c     Arguments:
c     ndim       Number of dimensions for integral(number or random #'s/point)
c     ncall      Number of times to evaluate the function/iteration
c     itmax      Max number of iterations
c     itmin      Min number of iterations
c     ninvar     Number of invarients to keep grids on (s,t,u, s',t' etc)
c     nconfigs   Number of different pole configurations 
c     dsig       Function to be integrated
c**************************************************************************
      implicit none
      include 'genps.inc'
c
c Arguments
c
      integer ndim,ncall,itmax,itmin,ninvar,nconfigs
      external         dsig
      double precision dsig
c
c Local
c
      double precision x(maxinvar),wgt,p(4*maxdim/3+14)
      double precision tdem, chi2, dum
      integer ievent,kevent,nwrite,iter,nun,luntmp,itsum
      integer jmax,i,j,ipole
      integer itmax_adjust
c
c     External
c
      integer  n_unwgted
      external n_unwgted
c
c Global
c
      integer                                      nsteps
      character*40          result_file,where_file
      common /sample_status/result_file,where_file,nsteps
      double precision fx
      common /to_fx/   fx

      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig

      double precision     xmean(99),xsigma(99),xwmax(99),xeff(99), xrmean(99)
      common/to_iterations/xmean,    xsigma,    xwmax,    xeff,     xrmean

      double precision    accur
      common /to_accuracy/accur

      double precision twgt, maxwgt,swgt(maxevents)
      integer                             lun, nw, itminx
      common/to_unwgt/twgt, maxwgt, swgt, lun, nw, itminx

      integer nzoom
      double precision  tx(1:3,maxinvar)
      common/to_xpoints/tx, nzoom

      double precision xzoomfact
      common/to_zoom/  xzoomfact

      double precision tmean, trmean, tsigma
      integer             dim, events, itm, kn, cur_it, invar, configs
      common /sample_common/
     .     tmean, trmean, tsigma, dim, events, itm, kn, cur_it, invar, configs

      integer           use_cut
      common /to_weight/use_cut

      integer              icor
      common/to_correlated/icor

      integer                   neventswritten
      common /to_eventswritten/ neventswritten

c
c     External
c
      logical pass_point
      integer NEXTUNOPEN
c
c     Data
c
      data result_file,where_file,nsteps/'SAMPLE','WHERE.AMI',100/
      data accur/-1d0/
      data mincfig /1/
      data maxcfig /1/
      data twgt/-1d0/              !Dont write out events
      data lun/27/                 !Unit number for events
      data maxwgt/0d0/
      data nw/0/                   !Number of events written
      

c-----
c Begin Code
c-----
      ievent = 0
      kevent = 0
      nzoom = 0
      xzoomfact = 1d0
      itminx = itmin
      if (nsteps .lt. 1) nsteps=1
      nwrite = itmax*ncall/nsteps

      call sample_init(ndim,ncall,itmax,ninvar,nconfigs)
      call graph_init
      do i=1,itmax
         xmean(i)=0d0
         xsigma(i)=0d0
      enddo
c      mincfig=1
c      maxcfig=nconfigs
      wgt = 0d0
c
c     Main Integration Loop
c
      iter = 1
      do while(iter .le. itmax)
c
c     Get integration point
c
         call sample_get_config(wgt,iter,ipole)
         if (iter .le. itmax) then
            ievent=ievent+1
            call x_to_f_arg(ndim,ipole,mincfig,maxcfig,ninvar,wgt,x,p)
            if (pass_point(p)) then
               fx = dsig(p,wgt,0) !Evaluate function
               wgt = wgt*fx
               if (wgt .ne. 0d0) call graph_point(p,wgt) !Update graphs
            else
               fx =0d0
               wgt=0d0
            endif
            call sample_put_point(wgt,x(1),iter,ipole,itmin) !Store result
         endif
         if (wgt .ne. 0d0) kevent=kevent+1    
c
c     Write out progress/histograms
c
         if (kevent .ge. nwrite) then
            nwrite = nwrite+ncall*itmax/nsteps
            nwrite = min(nwrite,ncall*itmax)
            call graph_store
         endif
 99   enddo
c
c     All done
c
      tmean = 0d0
      trmean = 0d0
      tsigma = 0d0
      tdem = 0d0
      open(unit=66,file='results.dat',status='unknown')
      i=1
      do while(xmean(i) .ne. 0 .and. i .lt. cur_it)
         i=i+1
      enddo
      cur_it = i
c     Use the last 3 iterations or cur_it-1 if cur_it-1 >= itmin but < 3
      itsum = min(max(itmin,cur_it-1),3)
      i = cur_it - itsum
      if (i .gt. 0) then
      tmean = 0d0
      trmean = 0d0
      tsigma = 0d0
      tdem = 0d0
      do while (xmean(i) .ne. 0 .and. i .lt. cur_it)
         tmean = tmean+xmean(i)*xmean(i)**2/xsigma(i)**2
         trmean = trmean+xrmean(i)*xmean(i)**2/xsigma(i)**2
         tdem = tdem+xmean(i)**2/xsigma(i)**2
         tsigma = tsigma + xmean(i)**2/ xsigma(i)**2
         i=i+1
      enddo
      tmean = tmean/tsigma
      trmean = trmean/tsigma
      tsigma= tmean/sqrt(tsigma)
c      nun = n_unwgted()

      nun = neventswritten

      chi2 = 0d0
      do i = cur_it-itsum,cur_it-1
         chi2 = chi2+(xmean(i)-tmean)**2/xsigma(i)**2
      enddo
      chi2 = chi2/2d0   !Since using only last 3, n-1=2
      write(*,'(a)') '-------------------------------------------------'
      write(*,'(a)') '---------------------------'
      write(*,'(a,i3,a,e12.4)') ' Results Last ',itsum,
     $     ' iters: Integral = ',trmean
      write(*,'(21x,a,e12.4)') 'Abs integral = ',tmean
      write(*,'(26x,a,e12.4)') 'Std dev = ',tsigma
      write(*,'(18x,a,f12.4)') 'Chi**2 per DoF. =',chi2
      write(*,'(a)') '-------------------------------------------------'
      write(*,'(a)') '---------------------------'

      if (nun .lt. 0) nun=-nun   !Case when wrote maximun number allowed
      if (chi2 .gt. 1) tsigma=tsigma*sqrt(chi2)
c     JA 02/2011 Added twgt to results.dat to allow event generation in
c     first iteration for gridpack runs
      if (icor .eq. 0) then
         write(66,'(3e12.5,2i9,i5,i9,e10.3,e12.5,e13.5)')tmean,tsigma,0.0,
     &     kevent, nw, cur_it-1, nun, nun/max(tmean,1d-99), twgt, trmean
      else
         write(66,'(3e12.5,2i9,i5,i9,e10.3,e12.5,e13.5)')tmean,0.0,tsigma,
     &     kevent, nw, cur_it-1, nun, nun/max(tmean,1d-99), twgt, trmean
      endif
c      do i=1,cur_it-1
      do i=cur_it-itsum,cur_it-1
         write(66,'(i4,5e15.5)') i,xmean(i),xsigma(i),xeff(i),xwmax(i),xrmean(i)
      enddo
      flush(66)
      close(66, status='KEEP')
      else
         open(unit=66,file='results.dat',status='unknown')
         write(66,'(3e12.5,2i9,i5,i9,3e10.3)')0.,0.,0.,kevent,nw,
     &     1,0,0.,0.,0.
         write(66,'(i4,5e15.5)') 1,0.,0.,0.,0.,0.
         flush(66)
         close(66, status='KEEP')

      endif
c
c     Now let's check to see if we got all of the events we needed
c     if not, will give it another try with 5 iterations to set
c     the grid, and 4 more to try and get the appropriate number of 
c     unweighted events.
c
      write(*,*) "Status",accur, cur_it, itmax
      if (accur .ge. 0d0 .or. cur_it .gt. itmax+3) then
        return
      endif
c     Check for neventswritten and chi2 (JA 8/17/11 lumi*mean xsec)
      if (neventswritten .gt. -accur*tmean .and. chi2 .lt. 10d0) then
         write(*,*) "We found enough events",neventswritten, -accur*tmean
         return
      endif
      
c
c     Need to start from scratch. This is clunky but I'll just
c     remove the grid, so we are clean
c
      write(*,*) "Trying w/ fresh grid"
      open(unit=25,file='ftn25',status='unknown',err=102)
      write(25,*) ' '
 102  close(25)

c
c     First few iterations will allow the grid to adjust
c
c
c     Reset counters
c
      ievent = 0
      kevent = 0
      nzoom = 0
      xzoomfact = 1d0

      ncall = ncall*4 ! / 2**(itmax-2)
      write(*,*) "Starting w/ ncall = ", ncall
      itmax = 8
      call sample_init(ndim,ncall,itmax,ninvar,nconfigs)
      do i=1,itmax
         xmean(i)=0d0
         xsigma(i)=0d0
      enddo
      wgt = 0d0
      call clear_events
      call set_peaks
c
c     Main Integration Loop
c
      iter = 1
c      itmax = 8
      itmax_adjust = 5
      use_cut = 2  !Start adjusting grid
      do while(iter .le. itmax)
         if (iter .gt. itmax_adjust .and. use_cut .ne. 0) then
            use_cut=0           !Fix grid
            write(*,*) 'Fixing grid'
         endif
c
c     Get integration point
c
         call sample_get_config(wgt,iter,ipole)
         if (iter .le. itmax) then
            ievent=ievent+1
            call x_to_f_arg(ndim,ipole,mincfig,maxcfig,ninvar,wgt,x,p)
            if (pass_point(p)) then
               xzoomfact = 1d0
               fx = dsig(p,wgt,0) !Evaluate function
               if (xzoomfact .gt. 0d0) then
                  wgt = wgt*fx*xzoomfact
               else
                  wgt = -xzoomfact
               endif
               if (wgt .gt. 0d0) call graph_point(p,wgt) !Update graphs
            else
               fx =0d0
               wgt=0d0
            endif
            if (nzoom .le. 0) then
               call sample_put_point(wgt,x(1),iter,ipole,itmin) !Store result
            else
               nzoom = nzoom -1
               ievent=ievent-1
            endif
         endif
         if (wgt .gt. 0d0) kevent=kevent+1    
199   enddo
c
c     All done
c
      open(unit=66,file='results.dat',status='unknown')
      i=1
      do while(xmean(i) .ne. 0 .and. i .lt. cur_it)
         i=i+1
      enddo
      cur_it = i
c     Use the last 3 iterations or cur_it-1 if cur_it-1 >= itmin
      itsum = min(max(itmin,cur_it-1),3)
      i = cur_it - itsum
      if (i .gt. 0) then
      tmean = 0d0
      trmean = 0d0
      tsigma = 0d0
      tdem = 0d0
      do while (xmean(i) .ne. 0 .and. i .lt. cur_it)
         tmean = tmean+xmean(i)*xmean(i)**2/xsigma(i)**2
         trmean = trmean+xrmean(i)*xmean(i)**2/xsigma(i)**2
         tdem = tdem+xmean(i)**2/xsigma(i)**2
         tsigma = tsigma + xmean(i)**2/ xsigma(i)**2
         i=i+1
      enddo
      tmean = tmean/tsigma
      trmean = trmean/tsigma
      tsigma= tmean/sqrt(tsigma)
c      nun = n_unwgted()
c
c     tjs 8/7/2007
c
      nun = neventswritten

      chi2 = 0d0
      do i = cur_it-itsum,cur_it-1
         chi2 = chi2+(xmean(i)-tmean)**2/xsigma(i)**2
      enddo
      chi2 = chi2/2d0   !Since using only last 3, n-1=2
      write(*,'(a)') '-------------------------------------------------'
      write(*,'(a)') '---------------------------'
      write(*,'(a,i3,a,e12.4)') ' Results Last ',itsum,
     $     ' iters: Integral = ',trmean
      write(*,'(21x,a,e12.4)') 'Abs integral = ',tmean
      write(*,'(25x,a,e12.4)') 'Std dev = ',tsigma
      write(*,'(17x,a,f12.4)') 'Chi**2 per DoF. =',chi2
      write(*,'(a)') '-------------------------------------------------'
      write(*,'(a)') '---------------------------'

      if (nun .lt. 0) nun=-nun   !Case when wrote maximun number allowed
      if (chi2 .gt. 1) tsigma=tsigma*sqrt(chi2)
c     JA 02/2011 Added twgt to results.dat to allow event generation in
c     first iteration for gridpack runs
      if (icor .eq. 0) then
         write(66,'(3e12.5,2i9,i5,i9,e10.3,e12.5,e13.5)')tmean,tsigma,0.0,
     &     kevent, nw, cur_it-1, nun, nun/max(tmean,1d-99), twgt,trmean
      else
         write(66,'(3e12.5,2i9,i5,i9,e10.3,e12.5,e13.5)')tmean,0.0,tsigma,
     &     kevent, nw, cur_it-1, nun, nun/max(tmean,1d-99), twgt,trmean
      endif
c      do i=1,cur_it-1
      do i=cur_it-itsum,cur_it-1
         write(66,'(i4,5e15.5)') i,xmean(i),xsigma(i),xeff(i),xwmax(i),xrmean(i)
      enddo
      flush(66)
      close(66, status='KEEP')
      else
         open(unit=66,file='results.dat',status='unknown')
         write(66,'(3e12.5,2i9,i5,i9,3e10.3)')0.,0.,0.,kevent,nw,
     &     1,0,0.,0.,0.
         write(66,'(i4,5e15.5)') 1,0.,0.,0.,0.,0.
         flush(66)
         close(66, status='KEEP')

      endif      

      end

      subroutine sample_writehtm()
c***********************************************************************
c     Writes out results of run in html format
c***********************************************************************
      implicit none
c
c     Constants
c
      character*(*) htmfile
      parameter (htmfile='results.html')
      integer    lun
      parameter (lun=26)
c
c     Local
c
      character*4 cpref
      double precision scale
      integer i
c
c     Global
c
      double precision     xmean(99),xsigma(99),xwmax(99),xeff(99), xrmean(99)
      common/to_iterations/xmean,    xsigma,    xwmax,    xeff,     xrmean

c-----
c  Begin Code
c-----
      return
c
c     Here we determine the appropriate units. Assuming the results 
c     were written in picobarns
c
      if (xmean(1) .ge. 1e4) then         !Use nano barns
         scale=1d-3
         cpref='(nb)'
      elseif (xmean(1) .ge. 1e1) then     !Use pico barns
         scale=1d0
         cpref='(pb)'
      else                               !Use fempto
         scale=1d+3
         cpref='(fb)'
      endif
      open(unit=lun,file=htmfile,status='unknown',err=999)      
      write(lun,50) '<head><title>Results_head</title></head>'
      write(lun,50) '<body><h2>Results for Process</h2>'
      write(lun,50) '<table border>'
      write(lun,50) '<Caption> Caption Results'
      write(lun,49) '<tr><th>Iteration</th>'
      write(lun,48)'<th>Cross Sect',cpref,'</th><th>Error',cpref,'</th>' 
      write(lun,49) '<th>Events (K)</th><th>Eff</th>'
      write(lun,50) '<th>Wrote</th><th>Unwgt</th></tr>'

c      write(lun,60) '<tr><th>AVG</th><th>',xtot*scale
c     $     ,'</th><th>',errtot*scale,'</th><th align=right>',
c     $     ntot/1000,'</th><th align=right>',teff,'</th></tr>'
      i=1
      do while(xmean(i) .gt. 0d0)
         write(lun,'(a)') '<tr>'
         write(lun,45) '<td align=right>',i,'</tr>'
         write(lun,46) '<td align=right>',xmean(i)*scale,'</td>'
         write(lun,46) '<td align=right>',xsigma(i)*scale,'</td>'
         write(lun,46) '<td align=right>',xeff(i)*scale,'</td>'
         write(lun,'(a)') '</tr>'
         i=i+1
      enddo
      write(lun,50) '</table></body>'
 999  close(lun)
 45   format(a,i4,a)
 46   format(a,f12.3,a)
 48   format(a,a,a,a)
 49   format(a)
 50   format(a)
      end



      subroutine sample_init(p1, p2, p3, p4, p5)
c************************************************************************
c     Initialize grid and random number generators
c************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'maxconfigs.inc'
c
c     Arguments
c
      integer p1, p2, p3, p4, p5

c
c     Local
c
      integer i, j
c
c     Global
c
      integer                                      nsteps
      character*40          result_file,where_file
      common /sample_status/result_file,where_file,nsteps

      double precision tmean, trmean, tsigma
      integer             dim, events, itm, kn, cur_it, invar, configs
      common /sample_common/
     .     tmean, trmean, tsigma, dim, events, itm, kn, cur_it, invar, configs

      double precision   grid(2, ng, 0:maxinvar)
      common /data_grid/ grid
      integer           Minvar(maxdim,lmaxconfigs)
      common /to_invar/ Minvar
      double precision   psect(maxconfigs),alpha(maxconfigs)
      common/to_mconfig2/psect          ,alpha
      logical first_time
      common/to_first/first_time
      integer           use_cut
      common /to_weight/use_cut
      integer           ituple
      common /to_random/ituple

      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file

      integer              icor
      common/to_correlated/icor

      logical               zooming
      common /to_zoomchoice/zooming

      data use_cut/2/            !Grid: 0=fixed , 1=standard, 2=non-zero
      data ituple/1/             !1=htuple, 2=sobel 
      data Minvar(1,1)/-1/       !No special variable mapping

c-----
c  Begin Code
c-----
      icor = 0
      If (use_cut .eq. 0) then
         icor = 1          !Assume correlated unless grid read
         print*,'Keeping grid fixed.'
      elseif(use_cut .eq. 1) then
         print*,'Using standard SAMPLE grid deformation.'
      elseif(use_cut .eq. 2) then
         print*,'Using non-zero grid deformation.'
      elseif(use_cut .eq. 3) then
         print*,'Using fluctuation for grid deformation.'
      elseif(use_cut .eq. 4) then
         print*,'Generating unweighted event shape.'
      elseif(use_cut .eq. 5) then
         print*,'Using constant plus linear grid deformation.'
      elseif(use_cut .eq. 6) then
         print*,'Using power law grid deformation.'
      else
         print*,'Using unknown grid deformation:',use_cut
      endif
c      open(unit=22,file=result_file,status='unknown')
c      write(22,*) 'Sample Status ',p2,p3,nsteps
c      close(22)
c      open(unit=22,file=where_file,status='unknown')
c      write(22,*) 'Sample Progress ',p2,p3,nsteps
c      close(22)

      dim      = p1
      events   = p2
      itm     = p3
      invar    = p4
      configs  = p5
      first_time = .true.

      if (dim .gt. maxdim) then
         write(*,*) 'Too many dimensions requested from Sample()'
         stop
      endif
c      if (dim .gt. invar) then
c         write(*,*) 'Too many dimensions dim > invar',dim,invar
c         stop
c      endif
      if (p4 .gt. maxinvar) then
         write(*,*) 'Too many invarients requested from Sample()',p4
         stop
      endif
      if (p5 .gt. maxconfigs) then
         write(*,*) 'Too many configs requested from Sample()',p5
         stop
      endif

      write(*,'(i3,a,i7,a,i3,a,i3,a,i3,a)') dim, ' dimensions', events,
     &     ' events',p4,' invarients',itm, ' iterations',
     &     p5,' config(s),  (0.99)'

      if (ituple .eq. 1) then
         print*,'Using h-tuple random number sequence.'
      elseif (ituple .eq. 2) then
         print*,'Using Sobel quasi-random number sequence.'
         write(*,*) 'Sorry cant use sobel'
         stop
c         call isobel(dim)
      else
         print*,'Unknown random number generator',ituple
      endif
c
c     See if need mapping between dimensions in different configurations
c     (ie using s,t,u type invarients)
c
      if (Minvar(1,1) .eq. -1) then
         print*,'No invarient mapping defined, using 1 to 1.'
         do i=1,configs
            do j=1,dim
               Minvar(j,i) = j+(i-1)*dim
            enddo
         enddo
      endif
c
c     Reset counters
c
      tmean = 0d0
      trmean = 0d0
      tsigma = 0d0
      kn = 0
      cur_it = 1
      do j=1,ng
         grid(2,j,0) = xgmin+(xgmax-xgmin)*j/dble(ng)
      enddo
c
c     Try to read grid from file
c
      flat_grid=.true.
      open(unit=25,file='ftn25',status='unknown',err=102)
      read(25,fmt='(4f20.17)', err=101, end=101)
     .     ((grid(2,i,j),i=1,ng),j=1,maxinvar)
c      write(*,*) 'Got the grid OK, now getting alpha'
c      read(25,fmt='(4f20.17)',err=101,end=101)(alpha(i),i=1,maxconfigs)
      write(*,*) 'Grid read from file'
      flat_grid=.false.
      close(25)
c
c     Determine weighting for each configuration
c      
      if (.not. flat_grid) icor = 0 !0 = not correlated 
      zooming = (.not. flat_grid .and. use_cut .eq. 0) !only zoom if grid already adjusted and not changing more
c
c   tjs 5/22/07 turn off zooming
c
      zooming = .false.
      if (configs .eq. 1) then
         do i=1,maxconfigs
            alpha(i) = 1
         enddo
      else
         write(*,*) 'Using uniform alpha',alpha(1)
c         tot=0d0
c         do i=1,configs
c            tot=tot+alpha(i)
c         enddo
         do i=1,maxconfigs
            if(i .le. configs) then
               alpha(i)=1d0/dble(configs)
            else
               alpha(i)=0d0
            endif
         enddo
      endif
      goto 103
 101  close(25)
c      write(*,*) 'Tried reading it',i,j
 102  write(*,*) 'Error opening grid'
c
c     Unable to read grid, using uniform grid and equal points in
c     each configuration
c
      write(*,*) 'Using Uniform Grid!'
      do j = 1, maxinvar
         do i = 1, ng
            grid(2, i, j) = xgmin+ (xgmax-xgmin)*(i / dble(ng))**1
         end do
      end do
      do j=1,maxconfigs
         if (j .le. configs) then
            alpha(j)=1d0/dble(configs)
         else
            alpha(j)=0d0
         endif
      enddo
      write(*,*) 'Using uniform alpha',alpha(1)
c      write(*,*) 'Forwarding random number generator'

 103  write(*,*) 'Grid defined OK'
      end

      subroutine setgrid(j,xo,a,itype)
c*************************************************************************
c     Presets the grid for a 1/(x-a)^itype distribution down to xo
c*************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
c
c     Arguments
c
      integer j, itype                !grid number
      double precision  xo            !minimum value
      double precision  a             !offset for peak
c
c     Local
c
      integer i,k
      integer ngu, ngd
c
c     Global
c
      double precision   grid(2, ng, 0:maxinvar)
      common /data_grid/ grid

      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file

c----- 
c  Begin Code
c-----
      if (flat_grid) then
         if (itype.gt.1) then
            write(*,'(a,i4,2e15.5,i4)') 'Setting grid',j,xo,a,itype
            if (a .ge. xo) then
               write(*,*) 'Can not integrate over singularity'
               write(*,*) 'Set grid',j,xo,a
               return
            endif
         else
            write(*,'(a,i4,1e15.5,i4)') 'Setting grid',j,xo,itype            
         endif
c     grid(2,1,j) = xo
         grid(2,ng,j)=xgmax
         if (itype .eq. 1) then
c
c     We'll use most for the peak, but save some for going down
c
            ngu = ng *0.9
            ngd = ng-ngu

            do i=1,ngu-1
c-------------------
c     tjs 6/30/2009; tjs & ja 2/25/2011
c     New form for setgrid
c-------------------
c               grid(2,i+ngd,j)=((1d0-a)/(xo-a))**(1d0-dble(i)/dble(ngu))
c               grid(2,i+ngd,j)=1d0/grid(2,i+ngd,j)+a
c               grid(2,i+ngd,j) = xo + ((dble(i)+xo-a)/(dble(ngu)+xo-a))**2
               grid(2,i+ngd,j) = xo**(1-dble(i)/dble(ngu))

            enddo
c
c     Now lets go down the other side
c
            grid(2,ngd,j) =  xo
            do i=1,ngd-1
c               grid(2,i,j) = ((1d0-a)/(xo-a))**(1d0-dble(i)/dble(ngd))
               grid(2,ngd-i,j) = xo-(grid(2,ngd+i,j)-xo)
               if (grid(2,ngd-i,j) .lt. -1d0) then
                  write(*,*) 'Error grid set too low',grid(2,ngd-i,j)
                  do k=1,ng
                     write(*,*) k,grid(2,k,j)
                  enddo
                  stop
               endif
            enddo
c
c     tjs, ja 2/25/11
c     Make sure sample all the way down to zero only if minimum positive
c     
            if (grid(2,1,j) .gt. 0) grid(2,1,j) = 0d0
c            write(*,*) "Adjusted bin 1 to zero"

         elseif (itype .eq. 2) then
            do i=2,ng-1
               grid(2,i,j)=(1d0/(xo-a))*(1d0-dble(i)/dble(ng))+
     $              (dble(i)/dble(ng))*(1d0/(1d0-a))
               grid(2,i,j)=1d0/grid(2,i,j)+a
            enddo         
         else
            write(*,*) 'No modification in setgrid',itype
         endif
         do i=1,ng
c             write(*,*) j,i,grid(2,i,j)
         enddo
         call sample_write_g(j,'_0')
      else
         write(*,*) 'No modification is setgrid, grid read from file'
      endif
      end

      subroutine sample_get_config(wgt, iteration, iconfig)
c************************************************************************
c     
c     INPUTS:
c
c     OUTPUTS:   wgt       == 1/nevents*niterations
c                iteration == Current iteration
c                iconfig   == configuration to use
c
c************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
c
c     Arguments
c
      double precision wgt
      integer iteration, iconfig
c
c     Local
c
      integer idum
      real xrnd
      double precision tot
c
c     External
c
      real ran1
c
c     Global
c
      double precision tmean, trmean, tsigma
      integer             dim, events, itm, kn, cur_it, invar, configs
      common /sample_common/
     .     tmean, trmean, tsigma, dim, events, itm, kn, cur_it, invar, configs
      double precision   psect(maxconfigs),alpha(maxconfigs)
      common/to_mconfig2/psect            ,alpha
      data idum/0/

      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig

c-----
c  Begin Code
c-----
      iteration = cur_it
      if (cur_it .gt. itm) then
         wgt = -1d0
      else
         wgt = 1d0 / (dble(events) * dble(itm))
c
c     Choose configuration
c
         if (configs .gt. 1) then
            xrnd = ran1(idum)
            iconfig=1
            tot = alpha(iconfig)
            do while (tot .lt. xrnd .and. iconfig .lt. configs)
               iconfig=iconfig+1
               tot = tot+alpha(iconfig)
            enddo
         else
            iconfig=mincfig
         endif
      endif
      end

      subroutine sample_get_x(wgt, x, j, ipole, xmin, xmax)
c************************************************************************
c     Returns maxdim random numbers between 0 and 1, and the wgt
c     associated with this set of points, and the iteration number
c     This routine chooses the point within the range specified by
c     xmin and xmax for dimension j in configuration ipole
c************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'maxconfigs.inc'
c
c     Arguments
c
      double precision wgt, x, xmin, xmax
      integer j, ipole
c
c     Local
c
      integer  im, ip,ij,icount,it_warned
      double precision xbin_min,xbin_max,ddum(maxdim),xo,y
c
c     External
c
      double precision xbin
      external         xbin
c
c     Global
c
      double precision tmean, trmean, tsigma
      integer             dim, events, itm, kn, cur_it, invar, configs
      common /sample_common/
     .     tmean, trmean, tsigma, dim, events, itm, kn, cur_it, invar, configs

      double precision    grid(2, ng, 0:maxinvar)
      common /data_grid/ grid
      integer           Minvar(maxdim,lmaxconfigs)
      common /to_invar/ Minvar

      integer           ituple
      common /to_random/ituple

      double precision      spole(maxinvar),swidth(maxinvar),bwjac
      common/to_brietwigner/spole        ,swidth        ,bwjac

      integer nzoom
      double precision  tx(1:3,maxinvar)
      common/to_xpoints/tx, nzoom

      data ddum/maxdim*0d0/
      data icount/0/
      data it_warned/0/

      integer            lastbin(maxdim)
      common /to_lastbin/lastbin

c-----
c  Begin Code
c-----
      if (it_warned .ne. cur_it) then
         icount=0
         it_warned = cur_it
      endif
      if (ituple .eq. 2) then   !Sobel generator
         print*,'Sorry Sobel generator disabled'
         stop
c         call sobel(ddum)
c         write(*,'(7f11.5)')(ddum(j)*real(ng),j=1,dim)
      endif
      if (ituple .eq. 1) then
c         write(*,*) 'Getting variable',ipole,j,minvar(j,ipole)
         xbin_min = xbin(xmin,minvar(j,ipole))
         xbin_max = xbin(xmax,minvar(j,ipole))
         if (xbin_min .gt. xbin_max-1) then
c            write(*,'(a,4e15.4)') 'Bad limits',xbin_min,xbin_max,
c     &           xmin,xmax
c            xbin_max=xbin_min+1d-10
            xbin_max = xbin(xmax,minvar(j,ipole))
            xbin_min = min(xbin(xmin,minvar(j,ipole)), xbin_max)
         endif
c
c     Line which allows us to keep choosing same x
c
c         if (swidth(j) .ge. 0) then
         if (nzoom .le. 0) then
            call ntuple(ddum(j), xbin_min,xbin_max, j, ipole)
         else
c            write(*,*) 'Reusing num',j,nzoom,tx(2,j)

            call ntuple(ddum(j),max(xbin_min,dble(int(tx(2,j)))),
     $           min(xbin_max,dble(int(tx(2,j))+1)),j,ipole)

            if(max(xbin_min,dble(int(tx(2,j)))).gt.
     $           min(xbin_max,dble(int(tx(2,j))+1))) then
c               write(*,*) 'not good'
            endif

c            write(*,'(2i6,4e15.5)') nzoom,j,ddum(j),tx(2,j),
c     $           max(xbin_min,dble(int(tx(2,j)))),
c     $           min(xbin_max,dble(int(tx(2,j))+1))

c            ddum(j) = tx(2,j)                 !Use last value


         endif
         tx(1,j) = xbin_min
         tx(2,j) = ddum(j)
         tx(3,j) = xbin_max
      elseif (ituple .eq. 2) then
         if (ipole .gt. 1) then
            print*,'Sorry Sobel not configured for multi-pole.'
            stop
         endif
         ddum(j)=ddum(j)*dble(ng)
      else
         print*,'Error unknown random number generator.',ituple
         stop
      endif

      im = ddum(j)
      if (im.ge.ng)then
         im = ng -1
         ddum(j) = ng
      endif
      if (im.le.0) im = 1
      ip = im + 1
      ij = Minvar(j,ipole)
c------
c     tjs 3/5/2011  save bin used to avoid looking up when storing wgt
c------
      lastbin(j) = ip
c
c     New method of choosing x from bins
c
      if (ip .eq. 1) then         !This is in the first bin
         xo = grid(2, ip, ij)-xgmin
         x = grid(2, ip, ij) - xo * (dble(ip) - ddum(j))
      else           
         xo = grid(2, ip, ij)-grid(2,im,ij)
         x = grid(2, ip, ij) - xo * (dble(ip) - ddum(j))
      endif
c
c     Now we transform x if there is a B.W., S, or T  pole
c
      if (ij .gt. 0) then
c         write(*,*) "pole, width",ij,spole(ij),swidth(ij)
         if (swidth(ij) .gt. 0d0) then
c            write(*,*) 'Tranpole called',ij,swidth(ij)
            y = x                             !Takes uniform y and returns
            call transpole(spole(ij),swidth(ij),y,x,wgt) !x on BW pole or 1/x 
         endif
      endif
c
c     Simple checks to see if we got the right point note 1e-3 corresponds
c     to the fact that the grids are required to be separated by 1e-14. Since
c     double precision is about 18 digits, we expect things to agree to
c     3 digit accuracy.
c
      if (abs(ddum(j)-xbin(x,ij))/(ddum(j)+1d-22) .gt. 1e-3) then
         if (icount .lt. 5) then
            write(*,'(a,i4,2e14.6,1e12.4)')
     &           'Warning xbin not returning correct x', ij,
     &           ddum(j),xbin(x,ij),xo
         elseif (icount .eq. 5) then
            write(*,'(a,a)')'Warning xbin still not working well. ',
     &           'Last message this iteration.'
         endif
         icount=icount+1
      endif
      if (x .lt. xmin .or. x .gt. xmax) then
c         write(*,'(a,4i4,2f24.16,1e10.2)') 'Bad x',ij,int(xbin_min),ip,
c     &        int(xbin_max),xmin,x,xmax-xmin
      endif

      wgt = wgt * xo * dble(xbin_max-xbin_min)
c      print*,'Returning x',ij,ipole,j,x
      end

      subroutine sample_get_wgt(wgt, x, j, ipole, xmin, xmax)
c************************************************************************
c     Returns the wgt for a point x in grid j of configuration
c     ipole between xmin and xmax
c************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'maxconfigs.inc'
c
c     Arguments
c
      double precision wgt, x, xmin, xmax
      integer j, ipole
c
c     Local
c
      integer  im, ip,ij
      double precision xbin_min,xbin_max,xbin2
      double precision xo
c
c     External
c
      double precision xbin
      external         xbin
c
c     Global
c
      double precision tmean, trmean, tsigma
      integer             dim, events, itm, kn, cur_it, invar, configs
      common /sample_common/
     .     tmean, trmean, tsigma, dim, events, itm, kn, cur_it, invar, configs

      double precision    grid(2, ng, 0:maxinvar)
      common /data_grid/ grid
      integer           Minvar(maxdim,lmaxconfigs)
      common /to_invar/ Minvar
      integer           ituple
      common /to_random/ituple
      double precision      spole(maxinvar),swidth(maxinvar),bwjac
      common/to_brietwigner/spole        ,swidth        ,bwjac

c-----
c  Begin Code
c-----
      if (xmin .gt. x) then
         if (xmin-x .lt. 1d-13) then
            x=xmin
         else
            write(*,'(a,2i4,4e10.4)') 'Error x out of range in get_wgt',
     $           j,minvar(j,ipole),xmin,x,xmax,x-xmin
            return
         endif
      endif
      if (xmax .lt. x) then
         if (x-xmax .lt. 1d-13) then
            x=xmax
         else
            write(*,'(a,2i4,4f8.4)') 'Error x out of range in get_wgt',
     $           j,minvar(j,ipole),xmin,x,xmax,x-xmin
            return
         endif
      endif
      if (ituple .eq. 1) then
         xbin_min = xbin(xmin,minvar(j,ipole))
         xbin_max = xbin(xmax,minvar(j,ipole))
         xbin2    = xbin(x,minvar(j,ipole))  !This must be last one for bwjac
         if (xbin_min .gt. xbin_max) then
            write(*,'(a,2e15.3,i6,2e15.3)') 'Error xbinmin>xbinmax'
     &           ,xbin_min,
     &           xbin_max,minvar(j,ipole),xmin,xmax
         endif
      else
         print*,'Error unknown random number generator.',ituple
         stop
      endif
      im = xbin2
      ip = im + 1
      ij = Minvar(j,ipole)
c
c     New method for finding bin
c
      if (ip .eq. 1) then
         xo=grid(2,ip,ij)-xgmin
      else
         xo=grid(2,ip,ij)-grid(2,im,ij)
      endif
      wgt = wgt * xo * dble(xbin_max-xbin_min)*bwjac
      if (wgt .le. 0d0) then
c         write(*,'(a,3i4,2f6.1,3e15.3)') 'Error wgt<0',j,ij,ip,
c     &        xbin_min,xbin_max,xo,xmin,xmax
c         write(*,'(2e25.15)') grid(2, ip, ij),grid(2, im, ij)
c         write(*,'(a,5e15.5)') 'Wgt',wgt,xo,
c     &        dble(xbin_max-xbin_min),bwjac
      endif
      end

      subroutine sample_result(mean, rmean, sigma, itmin)
      implicit none
      double precision mean, rmean, sigma
      integer i,cur_it,itmin,itsum
      double precision tsigma,tmean,trmean,tsig,tdem

      double precision     xmean(99),xsigma(99),xwmax(99),xeff(99), xrmean(99)
      common/to_iterations/xmean,    xsigma,    xwmax,    xeff,     xrmean


      i=1
      do while(xmean(i) .ne. 0 .and. i .lt. 99)
         i=i+1
      enddo
      cur_it = i
c     Use the last 3 iterations or cur_it-1 if cur_it-1 >= itmin
      itsum = min(max(itmin,cur_it-1),3)
      i = cur_it - itsum
      tmean = 0d0
      trmean = 0d0
      tsigma = 0d0
      if (i .gt. 0) then
      tdem = 0d0
      do while (xmean(i) .ne. 0 .and. i .lt. cur_it)
         tmean = tmean+xmean(i)*xmean(i)**2/xsigma(i)**2
         trmean = trmean+xrmean(i)*xmean(i)**2/xsigma(i)**2
         tdem = tdem+xmean(i)**2/xsigma(i)**2
         tsigma = tsigma + xmean(i)**2/ xsigma(i)**2
         i=i+1
      enddo
      tmean = tmean/tsigma
      trmean = trmean/tsigma
      tsigma= tmean/sqrt(tsigma)
      endif

      mean = tmean
      rmean = trmean
      sigma = tsigma

      end


      subroutine sample_put_point(wgt, point, iteration,ipole)
c**************************************************************************
c     Given point(maxinvar),wgt and iteration, updates the grid.
c     If at the end of an iteration, reforms the grid as necessary
c     and outputs current results
c**************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      integer    max_events
      parameter (max_events=500000) !Maximum # events before get non_zero
c
c     Arguments
c
      integer iteration,ipole
      double precision wgt, point(maxinvar)
c
c     Local
c
      integer i, j, k, knt, non_zero, nun,itsum
      double precision vol,xnmin,xnmax,tot,xdum,tmp1,chi2tmp
      double precision rc, dr, xo, xn, x(maxinvar), dum(ng-1)
      save vol,knt
      double precision  chi2
      save chi2, non_zero
      double precision wmax1,ddumb
      save wmax1
      double precision twgt1,xchi2,xxmean,tmeant,tsigmat
      integer iavg,navg
      save twgt1,iavg,navg
c
c     External
c
      double precision binwidth,xbin,dsig
      logical rebin
      integer n_unwgted
      external binwidth,xbin,dsig,rebin,n_unwgted
c
c     Global
c
      double precision    accur
      common /to_accuracy/accur

      double precision     xmean(99),xsigma(99),xwmax(99),xeff(99), xrmean(99)
      common/to_iterations/xmean,    xsigma,    xwmax,    xeff,     xrmean

      double precision mean,rmean,sigma
      common/to_result/mean,rmean,sigma

      double precision grid2(0:ng,maxinvar)
      integer               inon_zero(ng,maxinvar)
      common/to_grid2/grid2,inon_zero

      double precision tmean, trmean, tsigma
      integer             dim, events, itm, kn, cur_it, invar, configs
      common /sample_common/
     .     tmean, trmean, tsigma, dim, events, itm, kn, cur_it, invar, configs

      double precision    grid(2, ng, 0:maxinvar)
      common /data_grid/ grid
      integer                                      nsteps
      character*40          result_file,where_file
      common /sample_status/result_file,where_file,nsteps
      logical         first_time
      common/to_first/first_time
      integer           use_cut
      common /to_weight/use_cut
      double precision   xmin(maxinvar),xmax(maxinvar)
      common /to_extreme/xmin        ,xmax
      double precision reliable(ng,maxdim)
      common /to_error/reliable

      double precision twgt, maxwgt,swgt(maxevents)
      integer                             lun, nw, itmin
      common/to_unwgt/twgt, maxwgt, swgt, lun, nw, itmin


      real*8             wmax                 !This is redundant
      common/to_unweight/wmax

      double precision fx
      common /to_fx/   fx
      double precision   prb(maxconfigs,maxpoints,maxplace)
      double precision   fprb(maxinvar,maxpoints,maxplace)
      integer                      jpnt,jplace
      common/to_mconfig1/prb ,fprb,jpnt,jplace
      double precision   psect(maxconfigs),alpha(maxconfigs)
      common/to_mconfig2/psect          ,alpha
      double precision      spole(maxinvar),swidth(maxinvar),bwjac
      common/to_brietwigner/spole        ,swidth        ,bwjac
      
      integer                   neventswritten
      common /to_eventswritten/ neventswritten

      integer            lastbin(maxdim)
      common /to_lastbin/lastbin

      data prb/maxprb*1d0/
      data fprb/maxfprb*1d0/
      data jpnt,jplace /1,1/

c-----
c  Begin Code
c-----
      if (first_time) then
         first_time = .false.
         twgt1 = 0d0       !
         iavg = 0         !Vars for averging to increase err estimate
         navg = 1      !
         wmax1= 99d99
         wmax = -1d0
         mean = 0d0
         rmean = 0d0
         sigma = 0d0
         chi2 = 0d0
         non_zero = 0
         vol = 1d0 / dble(events * itm)
         knt = events

         do i=1,maxconfigs
            psect(i)=0d0
         enddo
         do i=1,invar
            xmin(i) = xgmax
            xmax(i) = xgmin
            do j=1,ng
               inon_zero(j,i)=0
               grid(1,j,i)   =0d0
               grid2(j,i)    =0d0
            enddo
         enddo
      endif

      if (iteration .eq. cur_it) then
         kn = kn + 1
         if (.true.) then       !Average points to increase error estimate
            twgt1=twgt1+dabs(wgt)     !This doesn't change anything should remove
            iavg = iavg+1
            if (iavg .ge. navg) then
               sigma=sigma+twgt1**2
               iavg = 0
               twgt1=0d0
            endif
         else
            sigma = sigma + wgt**2
         endif
         if (wgt .ne. 0.) then
            if (dabs(wgt)*itm*events .gt. wmax) then
               wmax=dabs(wgt)*itm*events
            endif
            non_zero = non_zero + 1
            mean = mean + dabs(wgt)
            rmean = rmean + wgt
            if (.true. ) then
c               psect(ipole)=psect(ipole)+wgt*wgt/alpha(ipole)  !Ohl 
c               psect(ipole)=1d0                 !Not doing multi_config
            else
               tot = 0d0
               do i=1,configs
                  tot=tot+prb(i,jpnt,jplace)*alpha(i)
               enddo
               do i=1,configs
                  if (tot .gt. 0d0) then !Pittau hep-ph/9405257
                     psect(i)=psect(i)+wgt*wgt*prb(i,jpnt,jplace)/tot
                  else
                     psect(i)=psect(i)+wgt*wgt*alpha(i) !prb not set....
                  endif
               enddo
            endif
c            write(123,'(2i6,1e15.5)') 1,1,wgt
c            write(123,'(5e15.9)') (fprb(i,jpnt,jplace),i=1,invar) 
c            write(123,'(5e15.9)') (prb(i,jpnt,jplace),i=1,configs) 
            do j = 1, invar
c               i = int(xbin(point(j),j))+1    
c--------------
c     tjs 3/5/2011  use stored value for last bin
c--------------
               i = lastbin(j)
c               write(*,*) 'bin choice',j,i,lastbin(j)
               if (i .gt. ng) then
                  print*,'error i>ng',i,j,ng,point(j)
                  i=ng
               endif
               grid(1, i, j) = grid(1, i, j) + abs(wgt)
               grid2(i, j) = grid2(i, j) + wgt**2
c
c     Lines below are for multiconfiguration
c
c               grid(1, i, j) = grid(1, i, j) +
c     &                          (abs(wgt)**2)*fprb(j,jpnt,jplace)
c               grid2(i, j) = grid2(i, j) + wgt**4*fprb(j,jpnt,jplace)
               if (abs(wgt) .gt. 0) inon_zero(i,j) = inon_zero(i,j)+1
c
c     Here we need to look out for point(j) which has been transformed
c     for Briet-Wigner pole
c
               if (j .gt. 0) then
                  if (swidth(j) .gt. 0d0) then
                     ddumb=0d0
                     call untranspole(spole(j),swidth(j),
     &                    point(j),point(j),ddumb)
                     if (point(j) .lt. 0d0) then
                        print*,'Warning point<0',j,point(j)
                     endif
                  endif
               endif
               if (abs(wgt) .gt. 0) xmin(j)=min(xmin(j),point(j))
               if (abs(wgt) .gt. 0) xmax(j)=max(xmax(j),point(j))
               if (xmin(j) .lt. xgmin) then
                  print*,'Warning xmin<0',j,xmin(j),point(j)
               endif
               xmin(j)=max(xmin(j),xgmin)
            end do
         endif
c
c     Now if done with an iteration, print out stats, rebin, reset
c         
c         if (kn .eq. events) then
         if (kn .ge. max_events .and. non_zero .lt. 5) then
            call none_pass(max_events)
         endif
         if (non_zero .eq. events .or. (kn .gt. 200*events .and.
     $        non_zero .gt. 5)) then
            mean=mean*dble(events)/dble(non_zero)
            rmean=rmean*dble(events)/dble(non_zero)
            twgt1=twgt1*dble(events)/dble(non_zero)
            sigma=sigma+twgt1**2    !This line for averaging over points
            if (non_zero .eq. 0) then
               write(*,*) 'Error no points passed the cuts.'
               write(*,*) 'Try running with more points or looser cuts.'
               stop
            endif
c            mean = mean * itm                 !Used if don't have non_zero
            if (.true.) then
               mean = mean * itm *dble(non_zero)/dble(kn)
               rmean = rmean * itm *dble(non_zero)/dble(kn)
               knt = kn
            endif
c
c     Need to fix this if averaging over navg events
c
c        write(*,*) (sigma/vol/vol-knt*mean*mean)/dble(knt-1)/dble(knt),
c     &        (sigma/vol/vol-knt*mean*mean*navg)/dble(knt-1)/ dble(knt)

            if (.true.) then
c               vol = 1d0/(knt*itm)
               sigma = (sigma/vol/vol-non_zero*mean*mean*navg)  !knt replaced by non_zero
     .              / dble(knt-1) / dble(knt)
            else
               sigma = (sigma/vol/vol - knt*mean*mean)
     .              / dble(knt-1) / dble(knt)
            endif

            tmean = tmean + mean * (mean**2 / sigma)
            trmean = trmean + rmean * (mean**2 / sigma)
            tsigma = tsigma + mean**2 / sigma
            chi2 = chi2 + mean**2 * (mean**2 / sigma)
            sigma = sqrt(abs(sigma))

            if (cur_it .lt. 100) then
               xmean(cur_it) = mean
               xrmean(cur_it) = rmean
               xsigma(cur_it) = sigma
               xwmax(cur_it)= wmax*dble(non_zero)/dble(kn)
               xeff(cur_it)= sigma*sqrt(dble(non_zero))/mean
c               call sample_writehtm()
            endif
            write(*,222) 'Iteration',cur_it,'Mean: ',rmean,
     &           ' Abs mean: ',mean, '  Fluctuation: ',sigma,
     &           wmax*(dble(non_zero)/dble(kn)),
     &           dble(non_zero)/dble(kn)*100.,'%'
 222        format(a10,I3,3x,a6,e10.4,a11,e10.4,a16,e10.3,e12.3,3x,f5.1,a1)

            write(*,223) cur_it, rmean, mean,' +- ', sigma,
     &           sigma*sqrt(dble(non_zero))/mean
 223        format( i3,3x,2e11.4,a,e10.4,f10.2)
            tot=0d0
            do i=1,configs
               tot=tot+psect(i)
            enddo
            if (configs .gt. 1)
     &           write(*,'(8f10.5)') (psect(i)/tot, i=1,configs)
c
c     Now set things up for generating unweighted events
c
            if (twgt .eq. -2d0) then               
               twgt = mean *kn/ (dble(itm)*dble(events)*dble(events))
c
c     now scale twgt, in case have large fluctuations
c

c               twgt = twgt * max(1d0, xeff(cur_it))

c
c     For small number of events only write about 1% of events
c
c               if (events .le. 2500) then
c                  twgt = mean *kn*100 /
c     $                 (dble(itm)*dble(events)*dble(events)) 
c               endif
c               twgt = max(twgt, maxwgt/10d0)
               write(*,*) 'Writing out events',twgt, xeff(cur_it)
c               write(*,*) mean, kn, itm, events
            endif
c
c     This tells it to write out a file for unweighted events
c
c            if(wmax*(dble(non_zero)/dble(kn)) .lt. wmax1) then
            if(sigma/(mean+1d-99) .lt. wmax1 .and. use_cut .ne. 0) then
c               wmax1 = wmax*(dble(non_zero)/dble(kn))
               wmax1 = sigma/(mean+1d-99)
c               open(26, file='ftn99',status='unknown')
c               write(26,fmt='(4f20.17)')
c     $              ((grid(2,i,j),i=1,ng),j=1,maxinvar)
c               write(26,fmt='(4f20.17)') (alpha(i),i=1,maxconfigs)
c               close(26)
            endif
            tot=0d0
            if (use_cut .ne. 0) then
c              write(*,*) 'Keeping alpha fixed'
               if (configs .gt. 1) then
                  do i=1,configs
                     alpha(i)=alpha(i)*sqrt(sqrt(psect(i))) !Pittau
                     tot = tot+alpha(i)
                     psect(i)=0d0
                  enddo
                  do i=1,configs
                     alpha(i)=alpha(i)/tot
                  enddo
                  write(*,'(A)') 'Configs:'
                  write(*,'(8f10.5)') (alpha(i),i=1,configs)
               endif
            endif
c            open(unit=22,file=result_file,status='old',access='append',
c     &           err=23)
c            write(22,222) 'Iteration',cur_it,'Mean: ',mean,
c     &           '  Fluctuation: ',sigma,
c     &           wmax*(dble(non_zero)/dble(kn)),
c     &           dble(non_zero)/dble(kn)*100.,'%'
c            close(22)
c------
c    Here we will double the number of events requested for the next run
c-----
 23         events = 2 * events
            vol = 1d0/dble(events*itm)
            knt = events
            twgt = mean / (dble(itm)*dble(events))
c            write(*,*) 'New number of events',events,twgt

            mean = 0d0
            rmean = 0d0
            sigma = 0d0
            cur_it = cur_it + 1
            kn = 0
            wmax = -1d0
c
c     do so adjusting of weights according to number of events in bin
c
            do j=1,invar
               do i = 1, ng
                  if (use_cut .ne. 2 .and.
     &                use_cut .ne. 3 .and. use_cut .ne. 5)
     $                 inon_zero(i,j) = 0
                  if (use_cut .eq. 3) grid(1,i,j)=grid2(i,j)
                  if (inon_zero(i,j) .ne. 0) then
                     grid(1,i,j) = grid(1,i,j)
     &                 *dble(min((real(non_zero)/real(inon_zero(i,j))),
     $                    10000.))
                     grid2(i,j) = grid2(i,j)
     &                 *dble(min((real(non_zero)/real(inon_zero(i,j))),
     $                    10000.))**2
                     if (real(non_zero)/real(inon_zero(i,j))
     &                    .gt. 100000) then
c                        if (j .eq. 1) then
                           print*,'Exceeded boost',j,i,
     &                          real(non_zero)/real(inon_zero(i,j))
c                        endif
                     endif
                     inon_zero(i,j) = 0
                  endif
                  if (use_cut .eq. 4)
     &                 reliable(i,j)=dsqrt(grid2(i,j))/grid(1,i,j)
               enddo
            enddo
            if (use_cut .eq. 4) then
               use_cut=0
            endif
            do j = 1, invar
               k=1
c
c              special routines to deal with xmin cutoff
c
               do while(grid(1,k,j) .le. 0d0 .and. k+1 .lt. ng)
                  k=k+1
               enddo

c               if (j .eq. 1) then
c                  open(unit=22,file='x1.dat',status='unknown')
c                  do i=1,ng
c                     write(22,'(i6,2e20.8)') i,grid(1,i,j),
c     $                    dsqrt(grid2(i,j))
c                  enddo
c                  close(22)
c               endif

               x(j)=0d0
               do i=1,ng
                  x(j)=x(j)+grid(1,i,j)
               enddo

               call average_grid(j,k,grid,grid2,x)

c               if (j .eq. 1 .and. .true.) then
c               open(unit=22,file='x1avg.dat',status='unknown')
c               do i=1,ng
c                  write(22,'(i6,2e20.8)') i,grid(1,i,1),
c     $                 dsqrt(grid2(i,1))
c               enddo
c               close(22)
c               endif

c
c     Now take logs to help the rebinning converge quicker
c
               rc = 0d0
               do i= k, ng
                  xo = (1.0d-14) + grid(1, i, j) / x(j)
                  grid(1, i, j) = ((xo - 1d0) / log(xo))**1.5 !this is 1.5
                  rc = rc + grid(1, i, j)
c                  write(*,*) i,rc
               end do      
               rc = rc / dble(ng)
               k = 0
               xn = xgmin
               dr = 0d0
               i = 0
c
c     Special lines to deal with xmin .ne. 0 cutoffs
c               
c
c     These assume one endpoints are xgmin and xgmax
c     
c

               xnmin = xgmin              !Endpoints for grid usually 0d0
               xnmax = xgmax              !Endpoint for grid usually 1d0
               if (xmin(j)-xgmin .gt. (grid(2,2,j)-grid(2,1,j)))then
                  xnmin = xmin(j)-(grid(2,2,j)-grid(2,1,j))/5d0
                  i = 1
                  dum(i)= xnmin
                  xn = xnmin
                  rc = rc * dble(ng)/dble(ng-i)
               endif
               dum(ng-1) = -1d0
               if (xgmax-xmax(j).gt.(grid(2,ng-1,j)-grid(2,ng-2,j)))then
                  xnmax = xmax(j)+(grid(2,ng-1,j)-grid(2,ng-2,j))/5d0
                  dum(ng-1)= xnmax
                  rc = rc * dble(ng-i)/dble(ng-i-1)
c                  print*,'xmax',j,xmax(j),dum(ng-1)
               endif
               
 25            k = k + 1
               dr = dr + grid(1, k, j)
               xo = xn
               xn = max(grid(2, k, j),xnmin)
               xn = min(xn,xnmax)
 26            if (rc .gt. dr) goto 25

               i = i + 1
               dr = dr - rc
               dum(i) = xn - (xn - xo) * dr / grid(1, k, j)
c
c     Put in check for 0 width bin NEED TO FIX THIS
c
               if (dum(ng-1) .eq. -1) then
                  if (i .lt. ng - 1 ) goto 26
               else
                  if (i .lt. ng - 2 ) goto 26
               endif
c
c     Here is another fix for 0 width bins
c
               do i=1,ng-2
                  if (dum(i+1)-dum(i) .le. 1d-14) then
c                     write(*,'(a,2i4,2f24.17,1e10.3)') 'Bin too small',
c     &                    j,i,dum(i),dum(i+1),dum(i+1)-dum(i)
                     dum(i+1)=dum(i)+1d-14
                     if (dum(i+1) .gt. xgmax) then
                        write(*,*) 'Error in rebin',i,dum(i),dum(i+1)
                     endif
                  endif
               enddo
c
c     Now reset counters and set new grid as necessary
c
               do i = 1, ng - 1
                  grid(1, i, j) = 0d0
                  grid2(i,j) = 0d0
                  if (use_cut .ne. 0 .and. j .gt. 0)
     $                 grid(2, i, j) = dum(i)
               end do
               grid(1, ng, J) = 0d0
               grid(2, ng, J) = xgmax
               grid2(ng,j)  = 0d0
               non_zero = 0

               call sample_write_g(j,'_1')

            end do
c            write(*,*) (irebin(j),j=1,dim)
c            open(unit=26,file='grid.dat',status='unknown')
c            do j=1,maxinvar
c               do i=1,ng
c                  write(26,*) grid(2,i,j),j,i
c               enddo
c            enddo
c            close(26)

c     Update weights in dsig (needed for subprocess group mode)
            xdum=dsig(0,0,2)
c
c     Add test to see if we have achieved desired accuracy 
c     Allow minimum itmin iterations
c
            if (tsigma .gt. 0d0 .and. cur_it .gt. itmin .and. accur .gt. 0d0) then

               xxmean = tmean/tsigma
               xchi2 = (chi2/xxmean/xxmean-tsigma)/dble(cur_it-2)               
               write(*,'(a,4f8.3)') ' Accuracy: ',sqrt(xchi2/tsigma),
     &              accur,1/sqrt(tsigma),xchi2
c               write(*,*) 'We got it',1d0/sqrt(tsigma), accur
c               if (1d0/sqrt(tsigma) .lt. accur) then
               if (sqrt(xchi2/tsigma) .lt. accur) then
                  write(*,*) 'Finished due to accuracy ',sqrt(xchi2/tsigma), accur
                  tmean = tmean / tsigma
                  trmean = trmean / tsigma
                  if (cur_it .gt. 2) then
                     chi2 = (chi2/tmean/tmean-tsigma)/dble(cur_it-2)
                  else
                     chi2=0d0
                  endif
                  tsigma = tmean / sqrt(tsigma)
                  write(*, 80) real(tmean), real(tsigma), real(trmean), real(chi2)
                  if (use_cut .ne. 0) then
                  open(26, file='ftn26',status='unknown')
                  write(26,fmt='(4f20.17)')
     $                 ((grid(2,i,j),i=1,ng),j=1,maxinvar)
                  write(26,fmt='(4f20.17)') (alpha(i),i=1,maxconfigs)
                  close(26)
                  endif
                  call sample_writehtm()
c                  open(unit=22,file=result_file,status='old',
c     $                 access='append',err=122)
c                  write(22, 80) real(tmean), real(tsigma), real(chi2)
c 122              close(22)
                  tsigma = tsigma*sqrt(chi2)  !This gives the 68% confidence cross section
                  call store_events
                  cur_it = itm+2
                  return
               endif
            endif                  
c
c New check to see if we need to keep integrating this one or not.
c
            if (cur_it .gt. itmin .and. accur .lt. 0d0) then  !Check luminocity
c
c             Lets get the actual number instead 
c             tjs 5/22/2007
c
c               nun = n_unwgted()
c               write(*,*) 'Estimated events',nun, accur
               call store_events
               nun = neventswritten
c               tmp1 = tmean / tsigma
c               chi2tmp = (chi2/tmp1/tmp1-tsigma)/dble(cur_it-2)
c     Calculate chi2 for last few iterations (ja 03/11)
               tmeant = 0d0
               tsigmat = 0d0
c     Use the last 3 iterations or cur_it-1 if cur_it-1 >= itmin but < 3
               itsum = min(max(itmin,cur_it-1),3)
               do i=cur_it-itsum,cur_it-1
                  tmeant = tmeant+xmean(i)*xmean(i)**2/xsigma(i)**2
                  tsigmat = tsigmat + xmean(i)**2/ xsigma(i)**2
               enddo
               tmeant = tmeant/tsigmat
               chi2tmp = 0d0
               do i = cur_it-itsum,cur_it-1
                  chi2tmp = chi2tmp+(xmean(i)-tmeant)**2/xsigma(i)**2
               enddo
               chi2tmp = chi2tmp/2d0  !Since using only last 3, n-1=2
c     JA 8/17/2011 Redefined -accur as lumi, so nevents is -accur*cross section
               write(*,*) "Checking number of events",-accur*tmeant,nun,' chi2: ',chi2tmp
c     Check nun and chi2 (ja 03/11)
               if (nun .gt. -accur*tmeant .and. chi2tmp .lt. 10d0)then   
                  tmean = tmean / tsigma
                  if (cur_it .gt. 2) then
                     chi2 = (chi2/tmean/tmean-tsigma)/dble(cur_it-2)
                  else
                     chi2=0d0
                  endif
                  tsigma = tmean / sqrt(tsigma)
                  write(*, 80) real(tmean), real(tsigma), real(chi2)
                  if (use_cut .ne. 0) then
                  open(26, file='ftn26',status='unknown')
                  write(26,fmt='(4f20.17)')
     $                 ((grid(2,i,j),i=1,ng),j=1,maxinvar)
                  write(26,fmt='(4f20.17)') (alpha(i),i=1,maxconfigs)
                  close(26)
                  endif
                  call sample_writehtm()

c                  open(unit=22,file=result_file,status='old',
c     $                 access='append',err=129)
c                  write(22, 80) real(tmean), real(tsigma), real(chi2)
c 129              close(22)
                  tsigma = tsigma*sqrt(chi2) !This gives the 68% confidence cross section
                  cur_it = itm+20
                  return
               endif
            endif                     


            if (cur_it .gt. itm) then               
               call store_events
               tmean = tmean / tsigma
               trmean = trmean / tsigma
               chi2 = (chi2 / tmean / tmean - tsigma) / dble(itm - 1)
               tsigma = tmean / sqrt(tsigma)
               write(*, 80) real(tmean), real(tsigma), real(trmean), real(chi2)
 80            format(/1X,79(1H-)/1X,23HAccumulated results:   ,
     .              10HIntegral =,e12.4/24X,10HStd dev  =,e12.4
     .              /23X,11HCross sec =,e12.4/
     .              13X,21HChi**2 per DoF.     =,f12.4/1X,79(1H-))
               if (use_cut .ne. 0) then
               open(26, file='ftn26',status='unknown')
               write(26,fmt='(4f20.17)')
     $              ((grid(2,i,j),i=1,ng),j=1,maxinvar)
               write(26,fmt='(4f20.17)') (alpha(i),i=1,maxconfigs)
               close(26)
               endif
               call sample_writehtm()
c               open(unit=22,file=result_file,status='old',
c     $              access='append',err=123)
c               write(22, 80) real(tmean), real(tsigma), real(chi2)
c 123           close(22)
               tsigma = tsigma*sqrt(chi2) !This gives the 68% confidence cross section
            else
c
c              Starting new iteration, should clean out stored events 
c              and start fresh
c
c                  nun = n_unwgted()
c                  write(*,*) 'Estimated unweighted events ', nun
              call clear_Events
            endif
         endif
      else
      endif
      end

      subroutine none_pass(max_events)
c*************************************************************************
c      Special break to handle case where no events are passing cuts
c     We'll set the cross section to zero here.
c*************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
c
c     Arguments
c
      integer max_events
c
c     Global
c
      integer                                      nsteps
      character*40          result_file,where_file
      common /sample_status/result_file,where_file,nsteps

c----
c  Begin Code
c----
      write(*,*) 'No points passed cuts!'
      write(*,*) 'Loosen cuts or increase max_events',max_events
c      open(unit=22,file=result_file,status='old',access='append',
c     &           err=23)
c      write(22,222) 'Iteration',0,'Mean: ',0d0,
c     &     '  Fluctuation: ',0d0,
c     &     0d0,
c     &     0d0,'%'
c 23   close(22)
 222  format(a10,I3,3x,a6,e10.4,a16,e10.3,e12.3,3x,f5.1,a1)

      open(unit=66,file='results.dat',status='unknown')
      write(66,'(3e12.5,2i9,i5,i9,3e10.3)')0.,0.,0.,0,0,
     &     0,1,0.,0.,0.
      write(66,'(i4,5e15.5)') 1,0.,0.,0.,0.,0.
      flush(66)
      close(66, status='KEEP')

c     Remove file events.lhe (otherwise event combination gets screwed up)
      write(*,*) 'Deleting file events.lhe'
      open(unit=67,file='events.lhe',status='unknown')
      write(67,*)
      close(67)

      stop
      end
            
      subroutine average_grid(j,k,grid,grid2,x)
c**************************************************************************
c     Special routine to deal with averaging over the grid bins
c     This routine starts averaging at bin k rather than bin 1 so that
c     one can accommodate cutoffs.  With k=1 this should give the
c     standard sample/vegas/bases averaging results.
c
c     Also stops averaging when reaches maximum value
c
c**************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
c
c     Arguments
c
      integer j,k
      double precision grid(2,ng,0:maxinvar),grid2(0:ng,maxinvar)
      double precision x(maxinvar)
c
c     Local
c
      integer i,kmax
      double precision xo,xn
c-----
c  Begin Code
c-----
      kmax=k
      do i=k+1,ng-1
         if (grid(1,i,j) .gt. 0d0) kmax=i
      enddo
      xo = grid(1,k,j)
      xn = grid(1,k+1,j)
      grid(1,k,j) = (xo+xn)/2d0
      x(j) = grid(1,k,j)
c      do i=k+1,ng-1                      !Original without kmax stuff
      do i=k+1,kmax-1
         grid(1, i, j) = xo + xn
         xo = xn
         xn = grid(1, i+1, j)
         grid(1, i, j) = (grid(1, i, j) + xn) / 3d0
         x(j) = x(j) + grid(1, i, j)
      end do
c      grid(1, ng, j) = (xn + xo) / 2d0  !Original without kmax stuff
      grid(1, kmax, j) = (xn + xo) / 2d0
      x(j) = x(j) + grid(1, ng, j)
      end

      double precision function xbin(y,j)
c**************************************************************************
c     Subroutine to determine which value y  will map to give you the
c     value of x when put through grid j.  That is what random number
c     do you need to be given to get the value x out of grid j and will be
c     between 0 < x < ng.
c**************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      double precision tol
      parameter       (tol=1d-12)
c
c     Arguments
c      
      double precision y
      integer j
c
c     Local
c
      integer i,jl,ju
      double precision x,xo
c
c     Global
c
      double precision    grid(2, ng, 0:maxinvar)
      common /data_grid/ grid
      double precision      spole(maxinvar),swidth(maxinvar),bwjac
      common/to_brietwigner/spole        ,swidth        ,bwjac
c
c     Data
c
      data spole,swidth/maxinvar*0d0,maxinvar*0d0/
c-----
c  Begin Code
c-----
      bwjac = 1d0
      if (j .gt. 0) then
         if (swidth(j) .gt. 0d0) then
            call  untranspole(spole(j),swidth(j),x,y,bwjac)
         else
            x=y
         endif
      else
         x=y
      endif
      if (x .eq. xgmax) then
         i=ng
         xbin = dble(ng)
      elseif (x .eq. xgmin) then
         xbin=0d0
      elseif(x .le. grid(2,1,j)) then
         i=1
         xo = grid(2,i,j)-xgmin
         xbin = dble(i)+(x-grid(2,i,j))/xo
      else
         jl = 1
         ju = ng
         do while (ju-jl .gt. 1)                    !Binary search
            i = (ju-jl)/2+jl
            if (grid(2,i,j) .le. x) then
               jl=i
            else
               ju=i
            endif
         enddo
         i=ju
         xo = grid(2,i,j)-grid(2,i-1,j)
         xbin = dble(i)+(x-grid(2,i,j))/xo
      endif
c      jbin=i
c      x = 
c      if (x+tol .gt. grid(2,i,j) .and. i .ne. ng) then
c         write(*,'(a,2e23.16,e9.2)') 'Warning in DSAMPLE:JBIN ',
c     &                x,grid(2,i,j),tol
c         x=2d0*grid(2,i,j)-x
c         jbin=i+1
c      endif
      end


      subroutine sample_write_g(idim,cpost)
c**************************************************************************
c     Writes out grid in function form for dimension i with extension cpost
c     
c**************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
c
c     Arguments
c
      integer idim
      character*(*) cpost
c
c     Local
c
      character*60 fname
      integer i
      double precision xo,yo
c
c     Global
c
      double precision   grid(2, ng, 0:maxinvar)
      common /data_grid/ grid

c-----
c  Begin Code
c-----
      return
      if (idim .lt. 1 .or. idim .gt.maxinvar) then
         write(*,*) 'Error invalid dimension in sample_write_f',idim
         return
      endif
      if (idim .lt. 10) then
         write(fname,'(a,i1,a,a)') 'g_',idim,cpost,'.dat'
      elseif (idim .lt. 100) then
         write(fname,'(a,i2,a,a)') 'g_',idim,cpost,'.dat'
      endif
      open(unit=21,file=fname,status='unknown',err=99)
      do i=1,ng-1
         xo = (grid(2,i,idim)+grid(2,i+1,idim))/2d0
         yo =1d0/(-grid(2,i,idim)+grid(2,i+1,idim))
         write(21,*) xo,yo
      enddo
      close(21)
      return
 99   write(*,*) 'Error opening file ',fname
      end

      function ran1(idum)
      dimension r(97)
      parameter (m1=259200,ia1=7141,ic1=54773,rm1=3.8580247e-6)
      parameter (m2=134456,ia2=8121,ic2=28411,rm2=7.4373773e-6)
      parameter (m3=243000,ia3=4561,ic3=51349)
      data iff /0/
      save r, ix1, ix2, ix3
      if (idum.lt.0.or.iff.eq.0) then
        iff=1
        ix1=mod(ic1-idum,m1)
        ix1=mod(ia1*ix1+ic1,m1)
        ix2=mod(ix1,m2)
        ix1=mod(ia1*ix1+ic1,m1)
        ix3=mod(ix1,m3)
        do 11 j=1,97
          ix1=mod(ia1*ix1+ic1,m1)
          ix2=mod(ia2*ix2+ic2,m2)
          r(j)=(float(ix1)+float(ix2)*rm2)*rm1
11      continue
        idum=1
      endif
      ix1=mod(ia1*ix1+ic1,m1)
      ix2=mod(ia2*ix2+ic2,m2)
      ix3=mod(ia3*ix3+ic3,m3)
      j=1+(97*ix3)/m3
      if(j.gt.97.or.j.lt.1) stop
      ran1=r(j)
      r(j)=(float(ix1)+float(ix2)*rm2)*rm1
      return
      end



