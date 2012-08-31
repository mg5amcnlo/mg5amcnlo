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
      INTEGER    ITMAX,   NCALL

      common/citmax/itmax,ncall
C
C     LOCAL
C
      integer i,ninvar,nconfigs,j,l,l1,l2,ndim
      double precision dsig,tot,mean,sigma
      integer npoints
      double precision x,y,jac,s1,s2,xmin
      external dsig
      character*130 buf
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
      include 'coupl.inc'
      
      integer           iconfig
      common/to_configs/iconfig


      double precision twgt, maxwgt,swgt(maxevents)
      integer                             lun, nw
      common/to_unwgt/twgt, maxwgt, swgt, lun, nw

c Vegas stuff
      integer ipole
      common/tosigint/ndim,ipole

      real*8 sigint,resA,errA,resS,errS,chi2a
      external sigint

      integer irestart
      character * 70 idstring
      logical savegrid

      external initplot

c For tests
      real*8 fksmaxwgt,xisave,ysave
      common/cfksmaxwgt/fksmaxwgt,xisave,ysave

c For MINT:
      include "mint.inc"
      real * 8 xgrid(0:nintervals,ndimmax),xint,ymax(nintervals,ndimmax)
      integer imode,ixi_i,iphi_i,iy_ij
      integer ifold(ndimmax) 
      common /cifold/ifold
      integer ifold_energy,ifold_phi,ifold_yij
      common /cifoldnumbers/ifold_energy,ifold_phi,ifold_yij
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      character*10 dum


c     $B$ new_def $E$  this is a tag for MadWeigth, Don't edit this line

c      double precision xsec,xerr
c      integer ncols,ncolflow(maxamps),ncolalt(maxamps),ic
c      common/to_colstats/ncols,ncolflow,ncolalt,ic
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
      call setpara('param_card.dat')   !Sets up couplings and masses
      call setcuts               !Sets up cuts & particle masses
      call printout              !Prints out a summary of paramaters
      call run_printout          !Prints out a summary of the run settings
      nconfigs = 1
c     
c     Get user input
c
      write(*,*) "getting user params"
      call get_user_params(ncall,itmax,iconfig,imode,ixi_i,iphi_i,iy_ij)
      call setfksfactor(iconfig)
c$$$      maxcfig=mincfig
c$$$      ipole=mincfig
c$$$      minvar(1,1) = 0              !This tells it to map things invarients
c$$$      write(*,*) 'Attempting mappinvarients',nconfigs,nexternal
c$$$      call map_invarients(minvar,nconfigs,ninvar,mincfig,maxcfig,nexternal,nincoming)
c$$$      write(*,*) "Completed mapping",nexternal
      ndim = 3*(nexternal-2)-4
      if (abs(lpp(1)) .ge. 1) ndim=ndim+1
      if (abs(lpp(2)) .ge. 1) ndim=ndim+1
c$$$      ninvar = ndim
c$$$      do j=mincfig,maxcfig
c$$$         if (abs(lpp(1)) .ge. 1 .and. abs(lpp(1)) .ge. 1) then
c$$$            minvar(ndim-1,j)=ninvar-1
c$$$            minvar(ndim,j) = ninvar
c$$$         elseif (abs(lpp(1)) .ge. 1 .or. abs(lpp(1)) .ge. 1) then
c$$$            minvar(ndim,j) = ninvar
c$$$         endif
c$$$      enddo

c Don't proceed if muF1#muF2 (we need to work out the relevant formulae
c at the NLO)
      if( ( fixed_fac_scale .and.
     #       (muF1_over_ref*muF1_ref_fixed) .ne.
     #       (muF2_over_ref*muF2_ref_fixed) ) .or.
     #    ( (.not.fixed_fac_scale) .and.
     #      muF1_over_ref.ne.muF2_over_ref ) )then
        write(*,*)'NLO computations require muF1=muF2'
        stop
      endif

      write(*,*) "about to integrate ", ndim,ncall,itmax,ninvar,nconfigs

      unwgt=.false.

      call addfil(dum)
      if (imode.eq.0) then
         do j=0,nintervals
            do i=1,ndimmax
              xgrid(j,i)=0.d0
            enddo
         enddo
         open(unit=99,file='WARMUP.top',status='unknown')
         call initplot
         write (*,*) 'imode is ',0
         call mint(sigint,ndim,ncall,itmax,imode,
     &        xgrid,xint,ymax,resA,errA,resS,errS)
         write(*,*)'Final result [ABS]:',resA,' +/-',errA
         write(*,*)'Final result:',resS,' +/-',errS
      
c to Save grids:
         open (unit=12, file='mint_grids',status='unknown')
         do j=0,nintervals
            write (12,*) (xgrid(j,i),i=1,ndim)
         enddo
         write (12,*) xint
         write (12,*) ifold_energy,ifold_phi,ifold_yij
         close (12)

      elseif(imode.eq.1) then
      open(unit=99,file='MADatNLO.top',status='unknown')
      call initplot

c to restore grids:
         open (unit=12, file='mint_grids',status='old')
         do j=0,nintervals
            read (12,*) (xgrid(j,i),i=1,ndim)
         enddo
         read (12,*) xint
         read (12,*) ifold_energy,ifold_phi,ifold_yij
         close (12)

c Prepare the MINT folding
         do j=1,ndimmax
            if (j.le.ndim) then
               ifold(j)=1
            else
               ifold(j)=0
            endif
         enddo
         ifold(ifold_energy)=ixi_i
         ifold(ifold_phi)=iphi_i
         ifold(ifold_yij)=iy_ij
         
         write (*,*) 'imode is ',1
         call mint(sigint,ndim,ncall,itmax,imode,
     &        xgrid,xint,ymax,resA,errA,resS,errS)
         write(*,*)'Final result [ABS]:',resA,' +/-',errA
         write(*,*)'Final result:',resS,' +/-',errS
      endif
         
      call mclear
      call topout
      close(99)

      end


      function sigint(xx,w,ifl,f_abs)
c From dsample_fks
      implicit none
      integer ndim,ipole
      common/tosigint/ndim,ipole
      integer           iconfig
      common/to_configs/iconfig
      integer i
      integer ifl
      integer fold
      common /cfl/fold
      real*8 sigint,xx(ndim),w
      integer ione
      parameter (ione=1)
      double precision wgt,dsig,f_abs
      include 'nexternal.inc'
      double precision x(99),p(0:3,nexternal)
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      double precision result
      save result
c
      do i=1,99
        if(i.le.ndim)then
          x(i)=xx(i)
        else
          x(i)=0.d0
        endif
      enddo
      wgt=1.d0
      fold=ifl
      if (ifl.eq.0)then
         call generate_momenta(ndim,iconfig,wgt,x,p)
         result = w*dsig(p,wgt,w)
         sigint = result
         f_abs = abs(result)
      elseif(ifl.eq.1) then
         call generate_momenta(ndim,iconfig,wgt,x,p)
         result = result+w*dsig(p,wgt,w)
         sigint = result
         f_abs = abs(result)
      elseif(ifl.eq.2) then
         sigint = result
         f_abs = abs(result)
         evtsgn = sign(1d0,result)
      endif
      return
      end


      subroutine get_user_params(ncall,itmax,iconfig,
     &     imode,ixi_i,iphi_i,iy_ij)
c**********************************************************************
c     Routine to get user specified parameters for run
c**********************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
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
      double precision    accuracy
      common /to_accuracy/accuracy
      integer           use_cut
      common /to_weight/use_cut

      integer        lbw(0:nexternal)  !Use of B.W.
      common /to_BW/ lbw

      character*5 abrvinput
      character*4 abrv
      common /to_abrv/ abrv

      logical nbody
      common/cnbody/nbody
      integer           iconfig
      common/to_configs/iconfig
c
c To convert diagram number to configuration
c
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer mapconfig(0:lmaxconfigs)
      include 'born_conf.inc'
c
c MINT stuff
c
      integer imode,ixi_i,iphi_i,iy_ij

      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
c-----
c  Begin Code
c-----
      mint=.true.
      usexinteg=.false.
      unwgt=.false.
      write(*,'(a)') 'Enter number of events and iterations: '
      read(*,*) ncall,itmax
      write(*,*) 'Number of events and iterations ',ncall,itmax
      write(*,'(a)') 'Enter desired fractional accuracy: '
      read(*,*) accuracy
      write(*,*) 'Desired fractional accuracy: ',accuracy

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
      do i=1,mapconfig(0)
         if (iconfig.eq.mapconfig(i)) then
            iconfig=i
            exit
         endif
      enddo
      write(*,12) 'Running Configuration Number: ',iconfig

      write (*,'(a)') 'Enter running mode for MINT:'
      write (*,'(a)') '(0) to set-up grids, (1) to integrate'
      read (*,*) imode
      write (*,*) imode

      write (*,'(a)') 'Set the three folding parameters for MINT'
      write (*,'(a)') 'xi_i, phi_i, y_ij'
      read (*,*) ixi_i,iphi_i,iy_ij
      write (*,*)ixi_i,iphi_i,iy_ij

      abrvinput='     '
      write (*,*) "'all ', 'born', 'real', 'virt', 'novi' or 'grid'?"
      write (*,*) "Enter 'born0' or 'virt0' to perform"
      write (*,*) " a pure n-body integration (no S functions)"
      read(5,*) abrvinput
      if(abrvinput(5:5).eq.'0')then
        nbody=.true.
      else
        nbody=.false.
      endif
      abrv=abrvinput(1:4)
      if(nbody.and.abrv.ne.'born'.and.abrv(1:2).ne.'vi'
     &     .and. abrv.ne.'grid')then
        write(*,*)'Error in driver: inconsistent input',abrvinput
        stop
      endif

      write (*,*) "doing the ",abrv," of this channel"
      if(nbody)then
        write (*,*) "integration Born/virtual with Sfunction=1"
      else
        write (*,*) "Normal integration (Sfunction != 1)"
      endif
c
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
      endif
 10   format( a)
 12   format( a,i4)
      end
c     $E$ get_user_params $E$ ! tag for MadWeight
c     change this routine to read the input in a file
c
