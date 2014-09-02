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
      include 'nexternal.inc'
      INTEGER    ITMAX,   NCALL
      common/citmax/itmax,ncall
C
C     LOCAL
C
      integer i,j,l,l1,l2,ndim
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

      real*8 sigint,res,err,chi2a
      external sigint

      integer irestart
      character * 70 idstring
      logical savegrid

      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file

      external initplot_dummy

      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint

      double precision average_virtual,virtual_fraction
      common/c_avg_virt/average_virtual,virtual_fraction

c For tests
      real*8 fksmaxwgt,xisave,ysave
      common/cfksmaxwgt/fksmaxwgt,xisave,ysave

      integer itotalpoints
      common/ctotalpoints/itotalpoints

      integer i_momcmp_count
      double precision xratmax
      common/ccheckcnt/i_momcmp_count,xratmax

c For tests of virtuals
      double precision vobmax,vobmin
      common/cvirt0test/vobmax,vobmin
      double precision vNsumw,vAsumw,vSsumw,vNsumf,vAsumf,vSsumf
      common/cvirt1test/vNsumw,vAsumw,vSsumw,vNsumf,vAsumf,vSsumf
      integer nvtozero
      logical doVirtTest
      common/cvirt2test/nvtozero,doVirtTest
      integer ivirtpoints,ivirtpointsExcept
      double precision  virtmax,virtmin,virtsum
      common/cvirt3test/virtmax,virtmin,virtsum,ivirtpoints,
     &     ivirtpointsExcept
      double precision total_wgt_sum,total_wgt_sum_max,
     &                 total_wgt_sum_min
      common/csum_of_wgts/total_wgt_sum,total_wgt_sum_max,
     &                 total_wgt_sum_min

      integer n_mp, n_disc
c statistics for MadLoop      
      integer ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1(0:9)
      common/ups_stats/ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1

c general MadFKS parameters
      include "FKSParams.inc"

C-----
C  BEGIN CODE
C-----
c
c     Read general MadFKS parameters
c
      call FKSParamReader(paramFileName,.TRUE.,.FALSE.)
      average_virtual=0d0
      virtual_fraction=Virt_fraction
c
c     Read process number
c
      ntot=0
      nsun=0
      nsps=0
      nups=0
      neps=0
      n100=0
      nddp=0
      nqdp=0
      nini=0
      n10=0
      do i=0,9
        n1(i)=0
      enddo

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
      call setcuts               !Sets up cuts and particle masses
      call printout              !Prints out a summary of paramaters
      call run_printout          !Prints out a summary of the run settings
      call initcluster
c     
c     Get user input
c
      write(*,*) "getting user params"
      call get_user_params(ncall,itmax,iconfig,irestart,idstring
     &     ,savegrid)
      if(irestart.eq.1)then
        flat_grid=.true.
      else
        flat_grid=.false.
      endif
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

      write(*,*) "about to integrate ", ndim,ncall,itmax,iconfig

      if(doVirtTest)then
        vobmax=-1.d8
        vobmin=1.d8
        vNsumw=0.d0
        vAsumw=0.d0
        vSsumw=0.d0
        vNsumf=0.d0
        vAsumf=0.d0
        vSsumf=0.d0
        nvtozero=0
        virtmax=-1d99
        virtmin=1d99
        virtsum=0d0
      endif
      itotalpoints=0
      ivirtpoints=0
      ivirtpointsExcept=0
      total_wgt_sum=0d0
      total_wgt_sum_max=0d0
      total_wgt_sum_min=0d0

      i_momcmp_count=0
      xratmax=0.d0

c Setup for parton-level NLO reweighting
      if(do_rwgt_scale.or.do_rwgt_pdf) call setup_fill_rwgt_NLOplot()
      call initplot
      if(savegrid)then
         call integrate(initplot_dummy,sigint,idstring,itmax,irestart
     $        ,ndim,ncall,res,err,chi2a,savegrid)
         usexinteg=.false.
      else
         call xinteg(sigint,ndim,itmax,ncall,res,err)
         usexinteg=.true.
      endif

      write (*,*) ''
      write (*,*) '----------------------------------------------------'
      write(*,*)'Final result:',res,'+/-',err
      write(*,*)'Maximum weight found:',fksmaxwgt
      write(*,*)'Found for:',xisave,ysave
      write (*,*) '----------------------------------------------------'
      write (*,*) ''

      open(unit=58,file='results.dat',status='unknown')
      write(58,*)res, err, 0d0, 0, 0, 0, 0, 0d0 ,0d0, res
      close(58)

      if(doVirtTest)then
        write(*,*)'  '
        write(*,*)'Statistics for virtuals'
        write(*,*)'max[V/(as/(2*pi)B)]:',vobmax
        write(*,*)'min[V/(as/(2*pi)B)]:',vobmin
        if(vNsumw.ne.0.d0)then
          vAsumw=vAsumw/vNsumw
          vSsumw=vSsumw/vNsumw
          write(*,*)'Weighted:'
          write(*,*)'  average=',vAsumw
          if(vSsumw.lt.(vAsumw**2*0.9999d0))then
            write(*,*)'Error in sigma',vSsumw,vAsumw
          else
            write(*,*)'  std dev=',sqrt(abs(vSsumw-vAsumw**2))
          endif
        else
          write(*,*)'Sum of weights [virt_w] is zero'
        endif
c
        if(vNsumf.ne.0.d0)then
          vAsumf=vAsumf/vNsumf
          vSsumf=vSsumf/vNsumf
          write(*,*)'Flat:'
          write(*,*)'  average=',vAsumf
          if(vSsumf.lt.(vAsumf**2*0.9999d0))then
            write(*,*)'Error in sigma',vSsumf,vAsumf
          else
            write(*,*)'  std dev=',sqrt(abs(vSsumf-vAsumf**2))
          endif
        else
          write(*,*)'Sum of weights [virt_f] is zero'
        endif
c
        if(nvtozero.ne.0)then
          write(*,*)
     &          '# of points (passing cuts) with Born=0 and virt=0:',
     &          nvtozero
        endif
        write (*,*) 'virtual weights directly from BinothLHA.f:'
        if (ivirtpoints.ne.0) then
           write (*,*) 'max(virtual/Born/ao2pi)= ',virtmax
           write (*,*) 'min(virtual/Born/ao2pi)= ',virtmin
           write (*,*) 'avg(virtual/Born/ao2pi)= ',
     &          virtsum/dble(ivirtpoints)
        endif
      endif

      write (*,*) ''
      write (*,*) '----------------------------------------------------'
      if (irestart.eq.1 .or. irestart.eq.3) then
         write (*,*) 'Total points tried:                   ',
     &        ncall*itmax
         write (*,*) 'Total points passing generation cuts: ',
     &        itotalpoints
         write (*,*) 'Efficiency of events passing cuts:    ',
     &        dble(itotalpoints)/dble(ncall*itmax)
      else
         write (*,*)
     &       'Run has been restarted, next line is only for current run'
         write (*,*) 'Total points passing cuts: ',itotalpoints
      endif
      write (*,*) '----------------------------------------------------'
      write (*,*) ''
      write (*,*) ''
      write (*,*) '----------------------------------------------------'

      if (ntot.ne.0) then
         write(*,*) "Satistics from MadLoop:"
         write(*,*)
     &        "  Total points tried:                              ",ntot
         write(*,*)
     &        "  Stability unknown:                               ",nsun
         write(*,*)
     &        "  Stable PS point:                                 ",nsps
         write(*,*)
     &        "  Unstable PS point (and rescued):                 ",nups
         write(*,*)
     &        "  Exceptional PS point (unstable and not rescued): ",neps
         write(*,*)
     &        "  Double precision used:                           ",nddp
         write(*,*)
     &        "  Quadruple precision used:                        ",nqdp
         write(*,*)
     &        "  Initialization phase-space points:               ",nini
         write(*,*)
     &        "  Unknown return code (100):                       ",n100
         write(*,*)
     &        "  Unknown return code (10):                        ",n10
         write(*,*)
     &        "  Unit return code distribution (1):               "
         do j=0,9
           if (n1(j).ne.0) then
              write(*,*) "#Unit ",j," = ",n1(j)
           endif
         enddo
      endif

      if(savegrid)call initplot_dummy
      call topout

      if(i_momcmp_count.ne.0)then
        write(*,*)'     '
        write(*,*)'WARNING: genps_fks code 555555'
        write(*,*)i_momcmp_count,xratmax
      endif

      end

      subroutine initplot_dummy
      return
      end

      function sigint(xx,peso)
c From dsample_fks
      implicit none
      include 'nexternal.inc'
      real*8 sigint,peso,xx(58)
      integer ione
      parameter (ione=1)
      integer ndim
      common/tosigint/ndim
      integer           iconfig
      common/to_configs/iconfig
      integer i
      double precision wgt,dsig,ran2,rnd
      external ran2
      double precision x(99),p(0:3,nexternal)
      include 'nFKSconfigs.inc'
      include 'reweight_all.inc'
      integer nfksprocess_all
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      character*4 abrv
      common /to_abrv/ abrv
      logical nbody
      common/cnbody/nbody
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      logical sum,firsttime
      parameter (sum=.false.)
      data firsttime /.true./
      logical foundB(2)
      integer nFKSprocessBorn(2)
      save nFKSprocessBorn,foundB
      double precision vol,sigintR
      integer itotalpoints
      common/ctotalpoints/itotalpoints
      logical fillh
      integer mc_hel,ihel
      double precision volh
      common/mc_int2/volh,mc_hel,ihel,fillh
c
      do i=1,99
         if (abrv.eq.'grid'.or.abrv.eq.'born'.or.abrv(1:2).eq.'vi')
     &        then
            if(i.le.ndim-3)then
               x(i)=xx(i)
            elseif(i.le.ndim) then
               x(i)=ran2()      ! Choose them flat when not including real-emision
            else
               x(i)=0.d0
            endif
         else
            if(i.le.ndim)then
               x(i)=xx(i)
            else
               x(i)=0.d0
            endif
         endif
      enddo

      sigint=0d0

c Find the nFKSprocess for which we compute the Born-like contributions
      if (firsttime) then
         firsttime=.false.
         foundB(1)=.false.
         foundB(2)=.false.
         do nFKSprocess=1,fks_configs
            call fks_inc_chooser()
            if (particle_type(i_fks).eq.8) then
               if (j_fks.le.nincoming) then
                  foundB(1)=.true.
                  nFKSprocessBorn(1)=nFKSprocess
               else
                  foundB(2)=.true.
                  nFKSprocessBorn(2)=nFKSprocess
               endif
            endif
         enddo
         write (*,*) 'Total number of FKS directories is', fks_configs
         write (*,*) 'For the Born we use nFKSprocesses  #',
     &        nFKSprocessBorn
      endif
         
c
c Compute the Born-like contributions with nbody=.true.
c THIS CAN BE OPTIMIZED
c
      call get_MC_integer(1,fks_configs,nFKSprocess,vol)
      nFKSprocess_all=nFKSprocess
      call fks_inc_chooser()
      if (j_fks.le.nincoming) then
         if (.not.foundB(1)) then
            write(*,*) 'Trying to generate Born momenta with '/
     &           /'initial state j_fks, but there is no '/
     &           /'configuration with i_fks a gluon and j_fks '/
     &           /'initial state'
            stop
         endif
         nFKSprocess=nFKSprocessBorn(1)
      else
         if (.not.foundB(2)) then
            write(*,*) 'Trying to generate Born momenta with '/
     &           /'final state j_fks, but there is no configuration'/
     &           /' with i_fks a gluon and j_fks final state'
            stop
         endif
         nFKSprocess=nFKSprocessBorn(2)
      endif
      nbody=.true.
      fillh=.false.  ! this is set to true in BinothLHA if doing MC over helicities
      nFKSprocess_used=nFKSprocess
      nFKSprocess_used_Born=nFKSprocess
      call fks_inc_chooser()
      call leshouche_inc_chooser()
      call setcuts
      call setfksfactor(iconfig)
      wgt=1d0
      call generate_momenta(ndim,iconfig,wgt,x,p)
      sigint = sigint+dsig(p,wgt,peso)
      if (mc_hel.ne.0 .and. fillh) then
c Fill the importance sampling array
         call fill_MC_integer(2,ihel,abs(sigint*peso*volh))
      endif
c
c Compute the subtracted real-emission corrections either as an explicit
c sum or a Monte Carlo sum.
c      
      if (abrv.ne.'born' .and. abrv.ne.'grid' .and.
     &     abrv(1:2).ne.'vi') then
         nbody=.false.
         if (sum) then
c THIS CAN BE OPTIMIZED
            do nFKSprocess=1,fks_configs
               nFKSprocess_used=nFKSprocess
               call fks_inc_chooser()
               call leshouche_inc_chooser()
               call setcuts
               call setfksfactor(iconfig)
               wgt=1d0
               call generate_momenta(ndim,iconfig,wgt,x,p)
               sigint = sigint+dsig(p,wgt,peso)
            enddo
         else                   ! Monte Carlo over nFKSprocess
            nFKSprocess=nFKSprocess_all
            nFKSprocess_used=nFKSprocess
c THIS CAN BE OPTIMIZED
            call fks_inc_chooser()
            call leshouche_inc_chooser()
            call setcuts
            call setfksfactor(iconfig)
c     The variable 'vol' is the size of the cell for the MC over
c     nFKSprocess. Need to divide by it here to correctly take into
c     account this Jacobian
            wgt=1d0/vol
            call generate_momenta(ndim,iconfig,wgt,x,p)
            sigintR = dsig(p,wgt,peso)
            call fill_MC_integer(1,nFKSprocess,abs(sigintR)*peso*vol)
            sigint = sigint+ sigintR
         endif
      endif
      if (sigint.ne.0d0)itotalpoints=itotalpoints+1
      return
      end

c
      subroutine get_user_params(ncall,itmax,iconfig,
     #                           irestart,idstring,savegrid)
c**********************************************************************
c     Routine to get user specified parameters for run
c**********************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
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
      logical fillh
      integer mc_hel,ihel
      double precision volh
      common/mc_int2/volh,mc_hel,ihel,fillh
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

      integer nvtozero
      logical doVirtTest
      common/cvirt2test/nvtozero,doVirtTest
c
c To convert diagram number to configuration
c
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer mapconfig(0:lmaxconfigs)
      include 'born_conf.inc'
c
c Vegas stuff
c
      integer irestart,itmp
      character * 70 idstring
      logical savegrid

      character * 80 runstr
      common/runstr/runstr
      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt

c-----
c  Begin Code
c-----
      doVirtTest=.true.
      mint=.false.
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
         mc_hel = 0
         write(*,*) 'Explicitly summing over helicities for virt'
      else
         mc_hel= i
         write(*,*) 'Summing over',i,' helicities/event for virt'
      endif
      isum_hel=0

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
c
c Enter parameters that control Vegas grids
c
      write(*,*)'enter id string for this run'
      read(*,*) idstring
      runstr=idstring
      write(*,*)'enter 1 if you want restart files'
      read (*,*) itmp
      if(itmp.eq.1) then
         savegrid = .true.
      else
         savegrid = .false.
      endif
      write(*,*)'enter 0 to exclude, 1 for new run, 2 to restart'
      read(5,*)irestart

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
c Options are way too many: make sure we understand all of them
      if ( abrv.ne.'all '.and.abrv.ne.'born'.and.abrv.ne.'real'.and.
     &     abrv.ne.'virt'.and.abrv.ne.'novi'.and.abrv.ne.'grid'.and.
     &     abrv.ne.'viSC'.and.abrv.ne.'viLC'.and.abrv.ne.'novA'.and.
     &     abrv.ne.'novB'.and.abrv.ne.'viSA'.and.abrv.ne.'viSB') then
        write(*,*)'Error in input: abrv is:',abrv
        stop
      endif
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

      doVirtTest=doVirtTest.and.abrv(1:2).eq.'vi'
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
c
