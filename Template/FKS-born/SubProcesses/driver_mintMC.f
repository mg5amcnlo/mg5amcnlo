      Program DRIVER
c**************************************************************************
c     This is the driver for the whole calculation
c**************************************************************************
      implicit none
C
C     CONSTANTS
C
      double precision zero
      parameter       (ZERO = 0d0)
      include 'nexternal.inc'
      include 'genps.inc'
      INTEGER    ITMAX,   NCALL

      common/citmax/itmax,ncall
C
C     LOCAL
C
      integer i,ninvar,nconfigs,j,l,l1,l2,ndim,nevts
      double precision tot,mean,sigma,res_abs
      integer npoints
      double precision y,jac,s1,s2,xmin
      character*130 buf,string

      integer lunlhe
      parameter (lunlhe=98)
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

      real*8 sigintH,sigintS,sigintF,resA,errA,resS,errS,chi2a
      external sigintH,sigintS,sigintF

      integer irestart
      character * 70 idstring
      logical savegrid

      external initplot
c Set plotKin=.true. to plot H and S event kinematics (integration steps)
c Set plotEv=.true. to use events for plotting (unweighting phase)
      logical plotEv,plotKin
      common/cEvKinplot/plotEv,plotKin

c For tests
      real*8 fksmaxwgt,xisave,ysave
      common/cfksmaxwgt/fksmaxwgt,xisave,ysave

      integer itotalpoints
      common/ctotalpoints/itotalpoints

      integer ivirtpoints,ivirtpointsExcept
      double precision  virtmax,virtmin,virtsum
      common/cvirt3test/virtmax,virtmin,virtsum,ivirtpoints,
     &     ivirtpointsExcept
      double precision total_wgt_sum,total_wgt_sum_max,
     &                 total_wgt_sum_min
      common/csum_of_wgts/total_wgt_sum,total_wgt_sum_max,
     &                 total_wgt_sum_min

      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file

c For MINT:
      include "mint.inc"
      real * 8 xgrid(0:nintervals,ndimmax),xint,ymax(nintervals,ndimmax)
      real * 8 x(ndimmax)
      integer ixi_i,iphi_i,iy_ij
      integer ifold(ndimmax) 
      common /cifold/ifold
      integer ifold_energy,ifold_phi,ifold_yij
      common /cifoldnumbers/ifold_energy,ifold_phi,ifold_yij
      logical putonshell
      integer imode
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt

      logical SHsep
      logical Hevents
      common/SHevents/Hevents
      character*10 dum

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
      call get_user_params(ncall,itmax,iconfig,imode,
     &     ixi_i,iphi_i,iy_ij,SHsep)
      if(imode.eq.0)then
        flat_grid=.true.
      else
        flat_grid=.false.
      endif
c$$$      call setfksfactor(iconfig)
      ndim = 3*(nexternal-2)-4
      if (abs(lpp(1)) .ge. 1) ndim=ndim+1
      if (abs(lpp(2)) .ge. 1) ndim=ndim+1

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

      itotalpoints=0
      ivirtpoints=0
      ivirtpointsExcept=0
      total_wgt_sum=0d0
      total_wgt_sum_max=0d0
      total_wgt_sum_min=0d0

      unwgt=.false.

c Plots
      plotEv=.false.
      plotKin=.false.

      call addfil(dum)
      if (imode.eq.-1.or.imode.eq.0) then
         if(imode.eq.0)then
            do j=0,nintervals
               do i=1,ndimmax
                  xgrid(j,i)=0.d0
               enddo
            enddo
         else
c to restore grids:
            open (unit=12, file='preset_mint_grids',status='old')
            do j=0,nintervals
               read (12,*) (xgrid(j,i),i=1,ndim)
            enddo
            read (12,*) xint
            read (12,*) ifold_energy,ifold_phi,ifold_yij
            close (12)
         endif
c
         if(plotKin)then
            open(unit=99,file='WARMUP.top',status='unknown')
            call initplot
         endif
c
         write (*,*) 'imode is ',imode
         if (Hevents.and.SHsep) then
            call mint(sigintH,ndim,ncall,itmax,imode,
     &           xgrid,xint,ymax,resA,errA,resS,errS)
         elseif(.not.SHsep) then
            call mint(sigintF,ndim,ncall,itmax,imode,
     &           xgrid,xint,ymax,resA,errA,resS,errS)
         else
            call mint(sigintS,ndim,ncall,itmax,imode,
     &           xgrid,xint,ymax,resA,errA,resS,errS)
         endif
         open(unit=58,file='res_0',status='unknown')
         write(58,*)'Final result [ABS]:',resA,' +/-',errA
         write(58,*)'Final result:',resS,' +/-',errS
         close(58)
         write(*,*)'Final result [ABS]:',resA,' +/-',errA
         write(*,*)'Final result:',resS,' +/-',errS
      
c to save grids:
         open (unit=12, file='mint_grids',status='unknown')
         do j=0,nintervals
            write (12,*) (xgrid(j,i),i=1,ndim)
         enddo
         write (12,*) xint
         write (12,*) ifold_energy,ifold_phi,ifold_yij
         close (12)

      elseif(imode.eq.1) then
         open (unit=99,file='nevts',status='old',err=999)
         read (99,*) nevts
         close(99)
         if(nevts.eq.0) then
            write (*,*)
     &           'No events needed for this channel...skipping it'
            stop
         endif
c
         if(plotKin)then
            open(unit=99,file='MADatNLO.top',status='unknown')
            call initplot
         endif

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
         
         write (*,*) 'imode is ',imode
         if (Hevents.and.SHsep) then
            call mint(sigintH,ndim,ncall,itmax,imode,
     &           xgrid,xint,ymax,resA,errA,resS,errS)
         elseif (.not.SHsep) then
            call mint(sigintF,ndim,ncall,itmax,imode,
     &           xgrid,xint,ymax,resA,errA,resS,errS)
         else
            call mint(sigintS,ndim,ncall,itmax,imode,
     &           xgrid,xint,ymax,resA,errA,resS,errS)
         endif
         open(unit=58,file='res_1',status='unknown')
         write(58,*)'Final result [ABS]:',resA,' +/-',errA
         write(58,*)'Final result:',resS,' +/-',errS
         close(58)

         write(*,*)'Final result [ABS]:',resA,' +/-',errA
         write(*,*)'Final result:',resS,' +/-',errS

c to save grids:
         open (unit=12, file='mint_grids_NLO',status='unknown')
         write (12,*) (xgrid(0,i),i=1,ndim)
         do j=1,nintervals
            write (12,*) (xgrid(j,i),i=1,ndim)
            write (12,*) (ymax(j,i),i=1,ndim)
         enddo
         write (12,*) (ifold(i),i=1,ndim)
         write (12,*) resS,errS
         close (12)


c Event generation
      elseif(imode.eq.2) then

c Mass-shell stuff. This is MC-dependent
         call fill_MC_mshell()
         putonshell=.true.
c$$$         putonshell=.false.

         unwgt=.true.
         open (unit=99,file='nevts',status='old',err=999)
         read (99,*) nevts
         close(99)
         if(nevts.eq.0) then
            write (*,*)
     &           'No events needed for this channel...skipping it'
            stop
         endif
         ncall=nevts ! Update ncall with the number found in 'nevts'
         
c to restore grids:
         open (unit=12, file='mint_grids_NLO',status='unknown')
         read (12,*) (xgrid(0,i),i=1,ndim)
         do j=1,nintervals
            read (12,*) (xgrid(j,i),i=1,ndim)
            read (12,*) (ymax(j,i),i=1,ndim)
         enddo
         read (12,*) (ifold(i),i=1,ndim)
         read (12,*) resS,errS
         close (12)

         if(plotEv)open(unit=99,file='hard-events.top',status='unknown')
         open(unit=lunlhe,file='events.lhe',status='unknown')

         call write_header_init(lunlhe,ncall,resS,errS)
         if(plotEv)call initplot
         open(unit=58,file='res_1',status='old')
         read(58,'(a)')string
         read(string(index(string,'[ABS]:')+6:index(string,'+/-')-1),*)
     &        res_abs
         close(58)

         write (*,*) 'imode is ',imode
         if (Hevents.and.SHsep) then
            imode=0 
            call gen(sigintH,ndim,xgrid,ymax,imode,x) 
            imode=1 
            do j=1,ncall
               call gen(sigintH,ndim,xgrid,ymax,imode,x) 
               call finalize_event(x,res_abs,lunlhe,plotEv,putonshell)
            enddo 
            imode=3 
            call gen(sigintH,ndim,xgrid,ymax,imode,x) 
         elseif (.not.SHsep) then
            imode=0 
            call gen(sigintF,ndim,xgrid,ymax,imode,x) 
            imode=1 
            do j=1,ncall
               call gen(sigintF,ndim,xgrid,ymax,imode,x) 
               call finalize_event(x,res_abs,lunlhe,plotEv,putonshell)
            enddo 
            imode=3 
            call gen(sigintF,ndim,xgrid,ymax,imode,x) 
         else
            imode=0 
            call gen(sigintS,ndim,xgrid,ymax,imode,x) 
            imode=1 
            do j=1,ncall
               call gen(sigintS,ndim,xgrid,ymax,imode,x) 
               call finalize_event(x,res_abs,lunlhe,plotEv,putonshell)
            enddo 
            imode=3 
            call gen(sigintS,ndim,xgrid,ymax,imode,x) 
         endif
         write (*,*) 'Generation efficiency:',x(1)

         write (lunlhe,'(a)') "</LesHouchesEvents>"
         close(lunlhe)
      endif

      write(*,*)'Maximum weight found:',fksmaxwgt
      write(*,*)'Found for:',xisave,ysave



      write (*,*) ''
      write (*,*) '----------------------------------------------------'
      write (*,*) 'Total points tried:                   ',
     &     ncall*itmax
      write (*,*) 'Total points passing generation cuts: ',
     &     itotalpoints
      write (*,*) 'Efficiency of events passing cuts:    ',
     &     dble(itotalpoints)/dble(ncall*itmax)
      write (*,*) '----------------------------------------------------'
      write (*,*) ''
      write (*,*) ''
      write (*,*) '----------------------------------------------------'
      write (*,*) 'number of except PS points:',ivirtpointsExcept,
     &     'out of',ivirtpoints,'points'
      write (*,*) '   treatment of exceptional PS points:'
      write (*,*) '      maximum approximation:',
     &     total_wgt_sum + dsqrt(total_wgt_sum_max)/dble(ncall*itmax)
      write (*,*) '      minimum approximation:',
     &     total_wgt_sum - dsqrt(total_wgt_sum_min)/dble(ncall*itmax)
      write (*,*) '      taking the max/min average:',
     &     total_wgt_sum/dble(ncall*itmax)
      write (*,*) '----------------------------------------------------'
      write (*,*) ''



      if(plotEv.or.plotKin)then
        call mclear
        call topout
        close(99)
      endif

      return
 999  write (*,*) 'nevts file not found'
      stop
      end

      subroutine get_user_params(ncall,itmax,iconfig,
     &     imode,ixi_i,iphi_i,iy_ij,SHsep)
c**********************************************************************
c     Routine to get user specified parameters for run
c**********************************************************************
      implicit none
c
c     Constants
c
      include 'nexternal.inc'
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
      double precision    accur
      common /to_accuracy/accur
      integer           use_cut
      common /to_weight/use_cut

      integer        lbw(0:nexternal)  !Use of B.W.
      common /to_BW/ lbw

      character*5 abrvinput
      character*4 abrv
      common /to_abrv/ abrv

      logical nbodyonly
      common/cnbodyonly/nbodyonly
c
c To convert diagram number to configuration
c
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer mapconfig(0:lmaxconfigs)
      include 'born_conf.inc'
c
c MC counterterm stuff
c
c alsf and besf are the parameters that control gfunsoft
      double precision alsf,besf
      common/cgfunsfp/alsf,besf
c alazi and beazi are the parameters that control gfunazi
      double precision alazi,beazi
      common/cgfunazi/alazi,beazi
      
      logical SHsep
      logical Hevents
      common/SHevents/Hevents
c
c MINT stuff
c
      integer imode,ixi_i,iphi_i,iy_ij

      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint

      character*10 MonteCarlo
      common/cMonteCarloType/MonteCarlo
c-----
c  Begin Code
c-----
      mint=.true.
      usexinteg=.false.
      write(*,'(a)') 'Enter number of events and iterations: '
      read(*,*) ncall,itmax
      write(*,*) 'Number of events and iterations ',ncall,itmax

      write(*,*)'Enter alpha, beta for G_soft'
      write(*,*)'  Enter alpha<0 to set G_soft=1 (no ME soft)'
      read(*,*)alsf,besf
      write (*,*) 'for G_soft: alpha=',alsf,', beta=',besf 

      write(*,*)'Enter alpha, beta for G_azi'
      write(*,*)'  Enter alpha>0 to set G_azi=0 (no azi corr)'
      read(*,*)alazi,beazi
      write (*,*) 'for G_azi: alpha=',alazi,', beta=',beazi

      write (*,*) "H-events (0), or S-events (1)"
      read (*,*) i
      if (i.eq.0) then
         Hevents=.true.
         write (*,*) 'Doing the H-events'
         SHsep=.true.
      elseif (i.eq.1) then
         Hevents=.false.
         write (*,*) 'Doing the S-events'
         SHsep=.true.
      elseif (i.eq.2) then
         Hevents=.true.
         write (*,*) 'Doing the S and H events together'
         SHsep=.false.
      endif

c These should be ignored (but kept for 'historical reasons')      
      accur=0
      use_cut=2


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
      write (*,'(a)') '0 to set-up grids, 1 to integrate,'//
     &     ' 2 to generate events'
      read (*,*) imode
      write (*,*) 'MINT running mode:',imode
      if (imode.eq.2)then
         write (*,*) 'Generating events, doing only one iteration'
         itmax=1
      endif

      write (*,'(a)') 'Set the three folding parameters for MINT'
      write (*,'(a)') 'xi_i, phi_i, y_ij'
      read (*,*) ixi_i,iphi_i,iy_ij
      write (*,*)ixi_i,iphi_i,iy_ij


      write (*,'(a)') 'Monte Carlo for showering. Possible options are'
      write (*,'(a)') 'HERWIG6, HERWIGPP, PYTHIA6Q, PYTHIA6PT, PYTHIA8'
      read (*,*) MonteCarlo
      write (*,*) MonteCarlo

      if(MonteCarlo.ne.'HERWIG6' .and.MonteCarlo.ne.'HERWIGPP' .and.
     #   MonteCarlo.ne.'PYTHIA6Q'.and.MonteCarlo.ne.'PYTHIA6PT'.and.
     #   MonteCarlo.ne.'PYTHIA8')then
        write(*,*)'Take it easy. This MC is not implemented'
        stop
      endif

      abrvinput='     '
      write (*,*) "'all ', 'born', 'real', 'virt', 'novi' or 'grid'?"
      write (*,*) "Enter 'born0' or 'virt0' to perform"
      write (*,*) " a pure n-body integration (no S functions)"
      read(*,*) abrvinput
      if(abrvinput(5:5).eq.'0')then
        nbodyonly=.true.
      else
        nbodyonly=.false.
      endif
      abrv=abrvinput(1:4)
      if(nbodyonly.and.abrv.ne.'born'.and.abrv(1:2).ne.'vi'
     &     .and. abrv.ne.'grid')then
        write(*,*)'Error in driver: inconsistent input',abrvinput
        stop
      endif

      write (*,*) "doing the ",abrv," of this channel"
      if(nbodyonly)then
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

