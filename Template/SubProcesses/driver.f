      Program DRIVER
c**************************************************************************
c     This is the driver for the whole calulation
c**************************************************************************
      implicit none
C
C     CONSTANTS
C
      double precision zero
      parameter       (ZERO = 0d0)
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      INTEGER    ITMAX,   NCALL
C
C     LOCAL
C
      integer i,ninvar,nconfigs,j,l,l1,l2,ndim
      double precision dsig,tot,mean,sigma
      integer npoints,lunsud
      double precision x,y,jac,s1,s2,xmin
      external dsig
      character*130 buf
      integer NextUnopen
      external NextUnopen
c
c     Global
c
      integer                                      nsteps
      character*40          result_file,where_file
      common /sample_status/result_file,where_file,nsteps
      integer           Minvar(maxdim,lmaxconfigs)
      common /to_invar/ Minvar
      real*8          dsigtot(10)
      common/to_dsig/ dsigtot
      integer ngroup
      common/to_group/ngroup
      data ngroup/0/
cc
      include 'run.inc'
      
      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig


      double precision twgt, maxwgt,swgt(maxevents)
      integer                             lun, nw
      common/to_unwgt/twgt, maxwgt, swgt, lun, nw

c--masses
      double precision pmass(nexternal)
      common/to_mass/  pmass
      double precision qmass(2)
      common/to_qmass/  qmass

c     $B$ new_def $E$  this is a tag for MadWeigth, Don't edit this line

c      double precision xsec,xerr
c      integer ncols,ncolflow(maxamps),ncolalt(maxamps),ic
c      common/to_colstats/ncols,ncolflow,ncolalt,ic

      include 'coupl.inc'

C-----
C  BEGIN CODE
C-----  
c
c     Read process number
c
      open (unit=lun+1,file='../dname.mg',status='unknown',err=11)
      read (lun+1,'(a130)',err=11,end=11) buf
      l1=index(buf,'P')
      l2=index(buf,'_')
      if(l1.ne.0.and.l2.ne.0.and.l1.lt.l2-1)
     $     read(buf(l1+1:l2-1),*,err=11) ngroup
 11   print *,'Process in group number ',ngroup

      lun = 27
      twgt = -2d0            !determine wgt after first iteration
      open(unit=lun,status='scratch')
      nsteps=2
      call setrun                !Sets up run parameters
c     $B$ setpara $B$ ! this is a tag for MadWeight. Don't edit this line
      call setpara('param_card.dat',.true.)   !Sets up couplings and masses
c     $E$ setpara $E$ ! this is a tag for MadWeight. Don't edit this line
      include 'pmass.inc'        !Sets up particle masses
      include 'qmass.inc'        !Sets up particle masses inside onium state
      call setcuts               !Sets up cuts 
      call printout              !Prints out a summary of paramaters
      call run_printout          !Prints out a summary of the run settings
      nconfigs = 1

c   If CKKW-type matching, read IS Sudakov grid
      if(ickkw.eq.2 .and. (lpp(1).ne.0.or.lpp(2).ne.0))then
        lunsud=NextUnopen()
        open(unit=lunsud,file=issgridfile,status='old',ERR=20)
        goto 40
 20     issgridfile='lib/'//issgridfile
        do i=1,5
          open(unit=lunsud,file=issgridfile,status='old',ERR=30)          
          exit
 30       issgridfile='../'//issgridfile
          if(i.eq.5)then
            print *,'ERROR: No Sudakov grid file found in lib with ickkw=2'
            stop
          endif
        enddo
        print *,'Reading Sudakov grid file ',issgridfile
 40     call readgrid(lunsud)
        print *,'Done reading IS Sudakovs'
      endif
        
      if(ickkw.eq.2)then
        hmult=.false.
        if(ngroup.ge.nhmult) hmult=.true.
        if(hmult)then
          print *,'Running CKKW as highest mult sample'
        else
          print *,'Running CKKW as lower mult sample'
        endif
      endif

c     
c     Get user input
c
      write(*,*) "getting user params"
      call get_user_params(ncall,itmax,mincfig)
      maxcfig=mincfig
      minvar(1,1) = 0              !This tells it to map things invarients
      write(*,*) 'Attempting mappinvarients',nconfigs,nexternal
      call map_invarients(minvar,nconfigs,ninvar,mincfig,maxcfig,nexternal,nincoming)
      write(*,*) "Completed mapping",nexternal
      ndim = 3*(nexternal-2)-4
      if (abs(lpp(1)) .ge. 1) ndim=ndim+1
      if (abs(lpp(2)) .ge. 1) ndim=ndim+1
      ninvar = ndim
      do j=mincfig,maxcfig
         if (abs(lpp(1)) .ge. 1 .and. abs(lpp(1)) .ge. 1) then
            minvar(ndim-1,j)=ninvar-1
            minvar(ndim,j) = ninvar
         elseif (abs(lpp(1)) .ge. 1 .or. abs(lpp(1)) .ge. 1) then
            minvar(ndim,j) = ninvar
         endif
      enddo
      write(*,*) "about to integrate ", ndim,ncall,itmax,ninvar,nconfigs
      call sample_full(ndim,ncall,itmax,dsig,ninvar,nconfigs)
c
c     Now write out events to permanent file
c
      if (twgt .gt. 0d0) maxwgt=maxwgt/twgt
      write(lun,'(a,f20.5)') 'Summary', maxwgt
      
c      call store_events

c      write(*,'(a34,20I7)'),'Color flows originally chosen:   ',
c     &     (ncolflow(i),i=1,ncols)
c      write(*,'(a34,20I7)'),'Color flows according to diagram:',
c     &     (ncolalt(i),i=1,ncols)
c
c      call sample_result(xsec,xerr)
c      write(*,*) 'Final xsec: ',xsec

      rewind(lun)

      close(lun)
      end

c     $B$ get_user_params $B$ ! tag for MadWeight
c     change this routine to read the input in a file
c
      subroutine get_user_params(ncall,itmax,iconfig)
c**********************************************************************
c     Routine to get user specified parameters for run
c**********************************************************************
      implicit none
c
c     Constants
c
      include 'nexternal.inc'
c
c     Arguments
c
      integer ncall,itmax,iconfig, jconfig
c
c     Local
c
      integer i, j
      double precision dconfig
c
c     Global
c
      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel
      double precision    accur
      common /to_accuracy/accur
      integer           use_cut
      common /to_weight/use_cut

      integer        lbw(0:nexternal)  !Use of B.W.
      common /to_BW/ lbw

c-----
c  Begin Code
c-----
      write(*,'(a)') 'Enter number of events and iterations: '
      read(*,*) ncall,itmax
      write(*,*) 'Number of events and iterations ',ncall,itmax
      write(*,'(a)') 'Enter desired fractional accuracy: '
      read(*,*) accur
      write(*,*) 'Desired fractional accuracy: ',accur

      write(*,'(a)') 'Enter 0 for fixed, 2 for adjustable grid: '
      read(*,*) use_cut
      if (use_cut .lt. 0 .or. use_cut .gt. 2) then
         write(*,*) 'Bad choice, using 2',use_cut
         use_cut = 2
      endif

      write(*,10) 'Suppress amplitude (0 no, 1 yes)? '
      read(*,*) i
      if (i .eq. 1) then
         multi_channel = .true.
         write(*,*) 'Using suppressed amplitude.'
      else
         multi_channel = .false.
         write(*,*) 'Using full amplitude.'
      endif

      write(*,10) 'Exact helicity sum (0 yes, n = number/event)? '
      read(*,*) i
      if (i .eq. 0) then
         isum_hel = 0
         write(*,*) 'Explicitly summing over helicities'
      else
         isum_hel= i
         write(*,*) 'Summing over',i,' helicities/event'
      endif

      write(*,10) 'Enter Configuration Number: '
      read(*,*) dconfig
      iconfig = int(dconfig)
      write(*,12) 'Running Configuration Number: ',iconfig
c
c     Here I want to set up with B.W. we map and which we don't
c
      dconfig = dconfig-iconfig
      if (dconfig .eq. 0) then
         write(*,*) 'Not subdividing B.W.'
         lbw(0)=0
      else
         lbw(0)=1
         jconfig=dconfig*1000.1
         write(*,*) 'Using dconfig=',jconfig
         call DeCode(jconfig,lbw(1),3,nexternal)
         write(*,*) 'BW Setting ', (lbw(j),j=1,nexternal-2)
c         do i=nexternal-3,0,-1
c            if (jconfig .ge. 2**i) then
c               lbw(i+1)=1
c               jconfig=jconfig-2**i
c            else
c               lbw(i+1)=0
c            endif 
c            write(*,*) i+1, lbw(i+1)
c         enddo
      endif
 10   format( a)
 12   format( a,i4)
      end
c     $E$ get_user_params $E$ ! tag for MadWeight
c     change this routine to read the input in a file
c
















