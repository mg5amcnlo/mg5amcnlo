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
      data HEL_PICKED/-1/
      data hel_jacobian/1.0d0/
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      INTEGER    ITMAX, ITMIN, NCALL
C
C     LOCAL
C
      integer i,ninvar,nconfigs,j,l,l1,l2,ndim,idum
      double precision dsig,tot,mean,sigma,xdum
      integer npoints,lunsud
      double precision x,y,jac,s1,s2,xmin
      external dsig
      character*130 buf
      integer NextUnopen
      external NextUnopen
      double precision t_before
c
c     Global
c
      integer                                      nsteps
      character*40          result_file,where_file
      common /sample_status/result_file,where_file,nsteps
      integer           Minvar(maxdim,lmaxconfigs)
      common /to_invar/ Minvar
      integer ngroup
      common/to_group/ngroup
      data ngroup/0/

      DOUBLE PRECISION CUMULATED_TIMING
      COMMON/GENERAL_STATS/CUMULATED_TIMING

c
c     PARAM_CARD
c
      character*30 param_card_name
      common/to_param_card_name/param_card_name
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
      call cpu_time(t_before)
      CUMULATED_TIMING = t_before
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

c     Read weight from results.dat if present, to allow event generation
c     in first iteration for gridpacks
      open (unit=lun+1,file='results.dat',status='unknown',err=13)
      read (lun+1,'(a130)',err=12,end=12) buf
      close (lun+1)
      read(buf,'(3e12.5,2i9,i5,i9,e10.3,e12.5)',err=13) xdum,xdum,xdum,
     $     idum,idum,idum,idum,xdum,twgt
      goto 14
 12   close (lun+1)
 13   twgt = -2d0               !determine wgt after first iteration
 14   continue
      lun = 27

      open(unit=lun,status='scratch')
      nsteps=2
      param_card_name = '%(param_card_name)s'
      call setrun                !Sets up run parameters
      call setpara(param_card_name %(secondparam)s)   !Sets up couplings and masses
      include 'pmass.inc'        !Sets up particle masses
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
      call init_good_hel()
      call get_user_params(ncall,itmax,itmin,mincfig)
      maxcfig=mincfig
      minvar(1,1) = 0              !This tells it to map things invarients
      write(*,*) 'Attempting mappinvarients',nconfigs,nexternal
      call map_invarients(minvar,nconfigs,ninvar,mincfig,maxcfig,nexternal,nincoming)
      write(*,*) "Completed mapping",nexternal
      ndim = 3*(nexternal-nincoming)-4
      if (nincoming.gt.1.and.abs(lpp(1)) .ge. 1) ndim=ndim+1
      if (nincoming.gt.1.and.abs(lpp(2)) .ge. 1) ndim=ndim+1
      ninvar = ndim
      do j=mincfig,maxcfig
         if (abs(lpp(1)) .ge. 1 .and. abs(lpp(1)) .ge. 1) then
            if(ndim.gt.1) minvar(ndim-1,j)=ninvar-1
            minvar(ndim,j) = ninvar
         elseif (abs(lpp(1)) .ge. 1 .or. abs(lpp(1)) .ge. 1) then
            minvar(ndim,j) = ninvar
         endif
      enddo
      write(*,*) "about to integrate ", ndim,ncall,itmax,itmin,ninvar,nconfigs
      call sample_full(ndim,ncall,itmax,itmin,dsig,ninvar,nconfigs)

c
c     Now write out events to permanent file
c
      if (twgt .gt. 0d0) maxwgt=maxwgt/twgt
      write(lun,'(a,f20.5)') 'Summary', maxwgt
      

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
      subroutine get_user_params(ncall,itmax,itmin,iconfig)
c**********************************************************************
c     Routine to get user specified parameters for run
c**********************************************************************
      use DiscreteSampler

      implicit none
c
c     Constants
c
      include 'nexternal.inc'
      include 'maxparticles.inc'
      integer NCOMB
      parameter (NCOMB=%(ncomb)i)
c
c     Arguments
c
      integer ncall,itmax,itmin,iconfig, diag_number
      common/to_diag_number/diag_number
c
c     Local
c
      integer i, j, jconfig, ncode
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
      write(*,'(a)') 'Enter number of events and max and min iterations: '
      read(*,*) ncall,itmax,itmin
      write(*,*) 'Number of events and iterations ',ncall,itmax,itmin
      write(*,'(a)') 'Enter desired fractional accuracy: '
      read(*,*) accur
      write(*,*) 'Desired fractional accuracy: ',accur

      write(*,'(a)') 'Enter 0 for fixed, 2 for adjustable grid: '
      read(*,*) use_cut
      if (use_cut .lt. 0 .or. use_cut .gt. 2) then
         if (use_cut.ne.-2) then
            write(*,*) 'Bad choice, using 2',use_cut
            use_cut = 2
         else if (use_cut.eq.-2)then
            itmax= 1
            itmin=1
         endif

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
         write(*,*) 'Monte-Carlo over helicities'
c        initialize the discrete sampler module
         call DS_register_dimension('Helicity',NCOMB)
c        Also set the minimum number of points for which each helicity
c        should be probed before the grid is used for sampling.
C        Typically 10 * n_matrix<i>
         call DS_set_min_points(%(hel_init_points)d,'Helicity')
      endif

      write(*,10) 'Enter Configuration Number: '
      read(*,*) dconfig
c     ncode is number of digits needed for the BW code
      ncode=int(dlog10(3d0)*(max_particles-3))+1
      iconfig = int(dconfig*(1+10**(-ncode)))
      write(*,12) 'Running Configuration Number: ',iconfig
      diag_number = iconfig
c
c     Here I want to set up with B.W. we map and which we don't
c
      dconfig = dconfig-iconfig
      if (dconfig .eq. 0) then
         write(*,*) 'Not subdividing B.W.'
         lbw(0)=0
      else
         lbw(0)=1
         jconfig=dconfig*(10**ncode + 0.1)
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
















