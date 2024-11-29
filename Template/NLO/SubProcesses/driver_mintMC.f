      Program DRIVER
c**************************************************************************
c     This is the driver for the whole calculation
c**************************************************************************
      use extra_weights
      use mint_module
      use FKSParams
      implicit none
C
C     CONSTANTS
C
      double precision zero
      parameter       (ZERO = 0d0)
      include 'nexternal.inc'
      include 'genps.inc'
      integer ncall_virt,ncall_novi
      character*4 abrv
      common /to_abrv/ abrv
C
C     LOCAL
C
      integer i,j,k,l,l1,l2,nndim,nevts

      integer lunlhe
      parameter (lunlhe=98)
c
c     Global
c
cc
      include 'run.inc'
      include 'coupl.inc'
c
c     Properly initialize PY8 controls
c
      include 'pythia8_control.inc'
      include 'pythia8_control_setup.inc'
c Vegas stuff
      common/tosigint/nndim

      real*8 sigintF
      external sigintF

      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file

      integer i_momcmp_count
      double precision xratmax
      common/ccheckcnt/i_momcmp_count,xratmax

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
      character*10 dum
      integer iFKS_picked
c statistics for MadLoop      
      integer ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1(0:9)
      common/ups_stats/ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1

c timing statistics
      include "timing_variables.inc"
      real*4 tOther, tTot
c general MadFKS parameters
      integer ifold_picked
      double precision x_save(ndimmax,max_fold)
      common /c_vegas_x_fold/x_save,ifold_picked
      double precision deravg,derstd,dermax,xi_i_fks_ev_der_max
     &     ,y_ij_fks_ev_der_max
      integer ntot_granny,derntot,ncase(0:6)
      common /c_granny_counters/ ntot_granny,ncase,derntot,deravg,derstd
     &     ,dermax,xi_i_fks_ev_der_max,y_ij_fks_ev_der_max
      integer                     n_MC_subt_diverge
      common/counter_subt_diverge/n_MC_subt_diverge
C-----
C  BEGIN CODE
C-----  
c Write the process PID in the log.txt files (i.e., to the screen)
      write (*,*) getpid()

      call cpu_time(tBefore)
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
      n_MC_subt_diverge=0
      ntot_granny=0
      derntot=0
      do i=0,6
         ncase(i)=0
      enddo
      
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
     &     ixi_i,iphi_i,iy_ij,SHsep)
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
c Don''t proceed if muF1#muF2 (we need to work out the relevant formulae
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
c Mass-shell stuff. This is MC-dependent
         call fill_MC_mshell()
         putonshell=.true.
         if (ickkw.eq.-1) putonshell=.false.
         unwgt=.true.
         open (unit=99,file='nevts',status='old',err=999)
         if (event_norm(1:4).ne.'bias') then
            read (99,*) nevts
         else
            read (99,*) nevts,event_weight
         endif
         close(99)
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

         if (event_norm(1:4).ne.'bias') then
            weight=(ans(1,1)+ans(5,1))/ncalls0
         else
            weight=event_weight
         endif

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
            call fill_rwgt_lines
            if (event_norm(1:4).eq.'bias') then
               call include_inverse_bias_wgt(inv_bias)
               weight=event_weight*inv_bias
            endif
            call finalize_event(x_save(1,ifold_picked),weight,lunlhe
     $           ,putonshell)
         enddo
         call deallocate_weight_lines
         vn=-1
         call gen(sigintF,3,vn,x) ! print counters generation efficiencies
         write (lunlhe,'(a)') "</LesHouchesEvents>"
         close(lunlhe)
      endif

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

      write (*,*) 'counters for the granny resonances'
      write (*,*) 'ntot     ',ntot_granny
      if (ntot_granny.gt.0) then
         do i=0,6
            write (*,*) '% icase ',i,' : ',ncase(i)/dble(ntot_granny)
         enddo
         write (*,*) 'average,std dev. and max of derivative:',deravg
     &        ,sqrt(abs(derstd-deravg**2)),dermax
         write (*,*)
     &        'and xi_i_fks and y_ij_fks corresponding to max of der.',
     &        xi_i_fks_ev_der_max,y_ij_fks_ev_der_max
      endif
      write (*,*) 'counter for the diverging MC subtraction',n_MC_subt_diverge
      call cpu_time(tAfter)
      tTot = tAfter-tBefore
      tOther = tTot - (tBorn+tGenPS+tReal+tCount+tIS+tFxFx+tf_nb+tf_all
     $     +t_as+tr_s+tr_pdf+t_plot+t_cuts+t_MC_subt+t_isum+t_p_unw
     $     +t_write+t_coupl)
      write(*,*) 'Time spent in Born : ',tBorn
      write(*,*) 'Time spent in PS_Generation : ',tGenPS
      write(*,*) 'Time spent in Reals_evaluation: ',tReal
      write(*,*) 'Time spent in MCsubtraction : ',t_MC_subt
      write(*,*) 'Time spent in Counter_terms : ',tCount
      write(*,*) 'Time spent in Integrated_CT : ',tIS-tOLP
      write(*,*) 'Time spent in Virtuals : ',tOLP      
      write(*,*) 'Time spent in FxFx_cluster : ',tFxFx
      write(*,*) 'Time spent in Nbody_prefactor : ',tf_nb
      write(*,*) 'Time spent in N1body_prefactor : ',tf_all
      write(*,*) 'Time spent in Adding_alphas_pdf : ',t_as
      write(*,*) 'Time spent in Reweight_scale : ',tr_s
      write(*,*) 'Time spent in Reweight_pdf : ',tr_pdf
      write(*,*) 'Time spent in Filling_plots : ',t_plot
      write(*,*) 'Time spent in Applying_cuts : ',t_cuts
      write(*,*) 'Time spent in Sum_ident_contr : ',t_isum
      write(*,*) 'Time spent in Pick_unwgt : ',t_p_unw
      write(*,*) 'Time spent in Write_events : ',t_write
      write(*,*) 'Time spent in AlphaS_dependencies : ',t_coupl
      write(*,*) 'Time spent in Other_tasks : ',tOther
      write(*,*) 'Time spent in Total : ',tTot

      open (unit=12, file='res.dat',status='unknown')
      if (imode.eq.0) then
         write (12,*)ans(1,1),unc(1,1),ans(2,1),unc(2,1),itmax,ncalls0,tTot
      else
         write (12,*)ans(1,1)+ans(5,1),sqrt(unc(1,1)**2+unc(5,1)**2),ans(2,1)
     $        ,unc(2,1),itmax,ncalls0,tTot
      endif
      close(12)

      return
 999  write (*,*) 'nevts file not found'
      stop
      end
