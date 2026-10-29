      Program DRIVER
c**************************************************************************
c     This is the driver for the whole calculation
c**************************************************************************
      use extra_weights
      use mint_module
      use FKSParams
      use process_module
      use scale_module
      use weight_lines, only: rejected_born_flow_points,
     $     rejected_mc_kinematics_points,rejected_nonfinite_points
      implicit none
C
C     CONSTANTS
C
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      integer ncall_virt,ncall_novi
      character*4 abrv
      common /to_abrv/ abrv
C
C     LOCAL
C
      integer i,j,nndim,nevts,p_label

      integer lunlhe
      parameter (lunlhe=98)
c
c     Global
c
cc
      include 'run.inc'
      include 'coupl.inc'
c Vegas stuff
      common/tosigint/nndim

      real*8 sigintF
      external sigintF

      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file

      double precision xratmax
      integer i_momcmp_count
      common/ccheckcnt/xratmax,i_momcmp_count

      double precision virtual_over_born
      common/c_vob/virtual_over_born
      include 'orders.inc'

      double precision weight,event_weight,inv_bias
      character*7 event_norm
      common /event_normalisation/event_norm
      integer ixi_i,iphi_i,iy_ij,vn
      logical putonshell
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      double precision ran2,x(ndimmax)
      external ran2
      
      integer ifile,ievents
      double precision inter,absint,uncer
      common /to_write_header_init/inter,absint,uncer,ifile,ievents

      logical SHsep
      logical Hevents
      common/SHevents/Hevents
      integer iFKS_picked
c statistics for MadLoop      
      integer ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1(0:9)
      common/ups_stats/ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1

c timing statistics
      double precision job_started,job_finished,tTot
      double precision res_abs,err_abs
c general MadFKS parameters
      integer ifold_picked
      double precision x_save(ndimmax,max_fold)
      common /c_vegas_x_fold/x_save,ifold_picked
      include 'born_nhel.inc'
C-----
C  BEGIN CODE
C-----  
c Write the process PID in the log.txt files (i.e., to the screen)
      write (*,*) getpid()

      call timing_reset()
      call cpu_time(job_started)
      fixed_order=.false.
      nlo_ps=.true.
      if (nincoming.ne.2) then
         write (*,*) 'Decay processes not supported for'/
     &        /' event generation'
         stop 1
      endif

c     Read general MadFKS parameters
c
      call FKSParamReader(paramFileName,.TRUE.,.FALSE.)
      min_virt_fraction_mint=min_virt_fraction
      do i=0,n_ave_virt
         average_virtual(i,1)=0d0
      enddo
      virtual_fraction(1)=virt_fraction
      n_ord_virt=amp_split_size
      
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

      call setrun                !Sets up run parameters
      if(macnlops)then
         call macnlops_validate_profile
c Both supported source laws keep their own sign classification. Do not
c silently replace sampled native components by a global history sum.
         if(MCExplicitKLSum.eqv.MCSubtractionAtFixedFlow)then
            write(*,*) 'MacNLOPS requires either sampled native ',
     $           'histories at fixed flow, or the complete history/flow sum'
            stop 1
         endif
         if(MCExplicitKLSum)then
            write(*,*) 'MacNLOPS source: complete native H-history ',
     $           'and Born-colour sum'
         else
            if(macnlops_resolved_source)
     $         error stop 'Legacy resolved source needs global H summation'
            write(*,*) 'MacNLOPS source: sampled native H components ',
     $           'at fixed Born colour flow'
         endif
         write(*,*) 'MacNLOPS source FKSExplicitSum = ',FKSExplicitSum
      endif
      call configure_fxfx_sudakov(ickkw,nlo_ps,shower_mc,.true.)
      call print_fks_isr_mapping(nlo_ps,shower_mc)
      call setpara('param_card.dat')   !Sets up couplings and masses
      call setcuts               !Sets up cuts and particle masses
      call printout              !Prints out a summary of paramaters
      call run_printout          !Prints out a summary of the run settings
      call fill_configurations_common
      call check_amp_split 
c     
c     Get user input
c
      write(*,*) "getting user params"
      call get_user_params(ncalls0,itmax,
     &     ixi_i,iphi_i,iy_ij,SHsep,nevts,p_label,event_weight)
c Only do the reweighting when actually generating the events
      if (imode.eq.2) then
         doreweight=do_rwgt_scale.or.do_rwgt_pdf.or.store_rwgt_info
      else
         doreweight=.false.
         do_rwgt_scale=.false.
         do_rwgt_pdf=.false.
      endif
      if (abrv(1:4).eq.'virt') then
         only_virt=.true.
      else
         only_virt=.false.
      endif

      if(imode.eq.0)then
        flat_grid=.true.
      else
        flat_grid=.false.
      endif
      ndim = 3*(nexternal-nincoming)-4
      if (abs(lpp(1)) .ge. 1) ndim=ndim+1
      if (abs(lpp(2)) .ge. 1) ndim=ndim+1
      nndim=ndim
      call born_spread_configure(born_spreading.and..not.only_virt
     $     .and.(abrv.eq.'all'.or.abrv.eq.'novi'),nexternal,
     $     nincoming,fks_configs,ndim)
      if (born_spread_active.and.imode.gt.0)
     $     call born_spread_load_table
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
      write(*,*) "about to integrate ", ndim,ncalls0,itmax,iconfig
      i_momcmp_count=0
      xratmax=0.d0
      unwgt=.false.
c     Prepare the MINT folding
      do j=1,ndimmax
         if (j.le.ndim) then
            ifold(j)=1
         else
            ifold(j)=0
         endif
      enddo
      ifold_energy=ndim-2
      ifold_yij=ndim-1
      ifold_phi=ndim
      ifold(ifold_energy)=ixi_i
      ifold(ifold_phi)=iphi_i
      ifold(ifold_yij)=iy_ij

      ! initialise the global, but process dependent, information in the process module.
      call init_process_module_global(shower_mc,abrv,nexternal,nincoming
     $     ,mcatnlo_delta,ebeam(1)+ebeam(2),max_bcol,maxflow,ickkw)
      ! Also put all the n-body process dependent stuff here. It does
      ! not depend on PS point or FKS config, so all global information.
      call init_process_module_nbody_wrapper()
      call init_scale_module(nexternal,shower_scale_factor,fks_configs
     $     ,product(ifold(1:ndim)))
         
      
c*************************************************************
c     setting of the grids
c*************************************************************
      if (imode.eq.-1.or.imode.eq.0) then
         write (*,*) 'imode is ',imode
         call mint(sigintF)
         call deallocate_weight_lines
         open(unit=58,file='results.dat',status='unknown')
         write(58,*) ans(1,1),unc(2,1),0d0,0,0,0,0,0d0,0d0,ans(2,1)
         close(58)
c*************************************************************
c     computation of upper bounding envelope
c*************************************************************
      elseif(imode.eq.1) then
         write (*,*) 'imode is ',imode
         call mint(sigintF)
         call deallocate_weight_lines
         open(unit=58,file='results.dat',status='unknown')
         write(58,*) ans(1,1)+ans(5,1),unc(2,1),0d0,0,0,0,0,0d0,0d0
     $        ,ans(2,1) 
         close(58)
c*************************************************************
c     event generation
c*************************************************************
      elseif(imode.eq.2) then
c MacNLOPS evaluates the hard density at the LHE coordinates. Preserve
c those coordinates; the ordinary shower-mass reshuffle changes them.
         call fill_MC_mshell()
         putonshell=.true.
         if (ickkw.eq.-1) putonshell=.false.
         unwgt=.true.
         write(*,*) 'Generating ', nevts, ' events'
         if(nevts.eq.0) then
            write (*,*)
     &           'No events needed for this channel...skipping it'
            stop
         endif
         ncalls0=nevts ! Update ncall with the number found in 'nevts'

c     to restore grids:

         call read_grids_from_file

c determine how many events for the virtual and how many for the no-virt
         ncall_virt=int(ans(5,1)/(ans(1,1)+ans(5,1)) * ncalls0)
         ncall_novi=ncalls0-ncall_virt

         write (*,*) "Generating virt :: novi approx.",ncall_virt
     $        ,ncall_novi

         open(unit=lunlhe,file='events.lhe',status='unknown')

c fill the information for the write_header_init common block
         ifile=lunlhe
         ievents=ncalls0
         inter=ans(2,1)
         absint=ans(1,1)+ans(5,1)
         uncer=unc(2,1)
         weight=event_weight

         if (abrv(1:3).ne.'all' .and. abrv(1:4).ne.'born' .and.
     $        abrv(1:4).ne.'virt') then
            write (*,*) 'CANNOT GENERATE EVENTS FOR ABRV',abrv
            stop 1
         endif

         write (*,*) 'imode is ',imode
         vn=-1
         call gen(sigintF,0,vn,x)
         do j=1,ncalls0
            if (abrv(1:4).eq.'born') then
               vn=3
               call gen(sigintF,1,vn,x)
            else
               if (ran2().lt.ans(5,1)/(ans(1,1)+ans(5,1)) .or. only_virt) then
                  abrv='virt'
                  if (only_virt) then
                     vn=2
                     call gen(sigintF,1,vn,x)
                  else
                     vn=1
                     call gen(sigintF,1,vn,x)
                  endif
               else
                  abrv='novi'
                  vn=2
                  call gen(sigintF,1,vn,x)
               endif
            endif
c Randomly pick the contribution that will be written in the event file
            call pick_unweight_contr(iFKS_picked,ifold_picked)
            call update_fks_dir(iFKS_picked)
            if (event_norm(1:4).eq.'bias') then
               call include_inverse_bias_wgt(inv_bias)
               weight=event_weight*inv_bias
            endif
            call fill_rwgt_lines
            call finalize_event(x_save(1,ifold_picked),weight,lunlhe
     $           ,putonshell,p_label)
         enddo
         call deallocate_weight_lines
         vn=-1
         call gen(sigintF,3,vn,x) ! print counters generation efficiencies
         write (lunlhe,'(a)') "</LesHouchesEvents>"
         close(lunlhe)
      endif

      if(rejected_born_flow_points.gt.0)write(*,*)
     $     'Total phase-space points rejected for invalid Born flows:',
     $     rejected_born_flow_points
      if(rejected_mc_kinematics_points.gt.0)write(*,*)
     $     'Total phase-space points rejected for invalid MC kinematics:',
     $     rejected_mc_kinematics_points
      if(rejected_nonfinite_points.gt.0)write(*,*)
     $     'Total phase-space points rejected for nonfinite weights:',
     $     rejected_nonfinite_points
      if(i_momcmp_count.ne.0)then
        write(*,*)'     '
        write(*,*)'WARNING: genps_fks code 555555'
        write(*,*)i_momcmp_count,xratmax
      endif

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

      call cpu_time(job_finished)
      tTot=job_finished-job_started
      call timing_report(tTot)

      open (unit=12, file='res.dat',status='unknown')
      if (imode.eq.0) then
         res_abs=ans(1,1)
         err_abs=unc(1,1)
      else
         res_abs=ans(1,1)+ans(5,1)
         err_abs=sqrt(unc(1,1)**2+unc(5,1)**2)
      endif
c Trailing fields: the S-event part and the Born cross section.
      write (12,*)res_abs,err_abs,ans(2,1),unc(2,1),itmax,ncalls0,tTot
     $     ,ans(int_s_events,1),unc(int_s_events,1)
     $     ,ans(int_born_xsec,1),unc(int_born_xsec,1)
      close(12)

      return
      end
