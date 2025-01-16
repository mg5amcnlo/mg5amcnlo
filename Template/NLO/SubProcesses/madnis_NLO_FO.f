************************************************************************
      subroutine madnis_nlo_initialise()
************************************************************************
*     initialize phase-space etc, at the beginning of the run
************************************************************************
      use extra_weights
      use mint_module
      use FKSParams
      implicit none
C
C     LOCAL
C
      integer i,j,k,l,l1,l2,kchan
      character*130 buf
c timing statistics
      include "timing_variables.inc"
      include 'orders.inc'
      include 'run.inc'
      include 'nexternal.inc'

c stats for granny_is_res
      double precision deravg,derstd,dermax,xi_i_fks_ev_der_max
     &     ,y_ij_fks_ev_der_max
      integer ntot_granny,derntot,ncase(0:6)
      common /c_granny_counters/ ntot_granny,ncase,derntot,deravg,derstd
     &     ,dermax,xi_i_fks_ev_der_max,y_ij_fks_ev_der_max
c PineAPPL
      logical pineappl
      common /for_pineappl/ pineappl
c statistics for MadLoop      
      integer ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1(0:9)
      common/ups_stats/ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1
c     Vegas stuff
      integer         nndim
      common/tosigint/nndim

      character*4      abrv
      common /to_abrv/ abrv
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file
      integer i_momcmp_count
      double precision xratmax
      common/ccheckcnt/i_momcmp_count,xratmax
      logical useitmax
      common/cuseitmax/useitmax
      character*10 dum

C-----
C  BEGIN CODE
C-----
c Write the process PID in the log.txt files (i.e., to the screen)

      write (*,*) getpid()
      
      useitmax=.false. ! to be overwritten in open_output_files.f if need be
c
c     Setup the timing variable
c
      call cpu_time(tBefore)
      fixed_order=.true.
      nlo_ps=.false.

c     Read general MadFKS parameters
c
      call FKSParamReader(paramFileName,.TRUE.,.FALSE.)
      min_virt_fraction_mint=min_virt_fraction
      do kchan=1,maxchannels
         do i=0,n_ave_virt
            average_virtual(i,kchan)=0d0
         enddo
         virtual_fraction(kchan)=max(virt_fraction,min_virt_fraction)
      enddo
      n_ord_virt=amp_split_size
c
c     Read process number
c
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
      call get_user_params(ncalls0,itmax,imode)
      if(imode.eq.0)then
        flat_grid=.true.
      else
        flat_grid=.false.
      endif
      ndim = 3*(nexternal-nincoming)-4
      if (abs(lpp(1)) .ge. 1) ndim=ndim+1
      if (abs(lpp(2)) .ge. 1) ndim=ndim+1
      nndim=ndim
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
      write(*,*) "about to integrate ", ndim,ncalls0,itmax
c PineAPPL
      if (imode.eq.0) pineappl=.False. ! overwrite when starting completely fresh
      if(pineappl) then
         write(6,*) "Initializing PineAPPL ..."
c     Set flavor map, starting from all possible
c     parton lumi configurations defined in initial_states_map.dat
         call setup_flavourmap
c     Fill the number of combined matrix elements for given initial state luminosity
         call find_iproc_map
         write(6,*) "   ... done."
      endif
      if (abrv(1:4).eq.'virt') then
         only_virt=.true.
      else
         only_virt=.false.
      endif
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
c      
      i_momcmp_count=0
      xratmax=0.d0
      unwgt=.false.
      call addfil(dum)

      if (imode.eq.-1.or.imode.eq.0) then
         if(imode.eq.0)then
c Don't safe the reweight information when just setting up the grids.
            doreweight=.false.
            do_rwgt_scale=.false.
            do_rwgt_pdf=.false.
         else
            doreweight=do_rwgt_scale.or.do_rwgt_pdf.or.store_rwgt_info
         endif
c
         write (*,*) 'imode is ',imode

         if (ickkw.eq.-1) then
            min_virt_fraction=1d0
            do kchan=1,nchans
               virtual_fraction(kchan)=1d0
            enddo
         endif
c
      else
         write (*,*) 'Unknown imode',imode
         stop
      endif

      return
      end

************************************************************************
      subroutine madnis_get_channel(ichan_out)
************************************************************************
*     This is a subroutine that returns the the used channel integrat
*
*     OUTPUTS: ichan_out == used channel of integrtation
************************************************************************
      use mint_module
! picks and integration channel and returns it.
! Wraps functions inside mint_module
      implicit none
      integer ichan_out
      call get_channel_public(ichan_out)
      return
      end


      subroutine madnis_get_nchans(nchan_out)
      use mint_module
! picks and integration channel and returns it.
! Wraps functions inside mint_module
      implicit none
      integer nchan_out
      call get_nchans(nchan_out)
      return
      end


      subroutine madnis_set_channel(ichan_in, vol_in)
      use mint_module
! set ichan and the associated volume
! Wraps functions inside mint_module
      implicit none
      integer ichan_in
      double precision vol_in
      call set_channel(ichan_in, vol_in)
      return
      end


************************************************************************
      subroutine madnis_nlo_terminate()
************************************************************************
*     The termination routines, to be called at the end of the run
*     > probably not important for plain madnis
************************************************************************
c timing statistics
      use mint_module
      implicit none
      integer kchan
      include "timing_variables.inc"
      real*4 tOther, tTot

      call topout
      call deallocate_weight_lines

      call cpu_time(tAfter)
      tTot = tAfter-tBefore
      tOther = tTot - (tBorn+tGenPS+tReal+tCount+tIS+tFxFx+tf_nb+tf_all
     &     +t_as+tr_s+tr_pdf+t_plot+t_cuts+t_MC_subt+t_isum+t_p_unw
     $     +t_write+t_ewsud+t_coupl)
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
      write(*,*) 'Time spent in EW_sudakov : ',t_ewsud
      write(*,*) 'Time spent in AlphaS_dependencies : ',t_coupl
      write(*,*) 'Time spent in Other_tasks : ',tOther
      write(*,*) 'Time spent in Total : ',tTot

      open (unit=12, file='res.dat',status='unknown')
      do kchan=0,nchans
         write (12,*)ans(1,kchan),unc(1,kchan),ans(2,kchan),unc(2,kchan)
     $        ,itmax,ncalls0,tTot
      enddo
      close(12)

      return
      end

************************************************************************
      subroutine madnis_nlo_evaluate(xx,vegas_wgt,ifl,f)
************************************************************************
*     The evaluation of the integrand, essentially wrapping around
*     sigint
*
*     INPUTS:  xx        == random numbers
*              vegas_wgt == vegas weight (or madnis)
*              ifl       == choose channel config
*     OUTPUTS: f         == weights of the integral -> relevant is f(2)
*                           [check fks_singular.f for details]
************************************************************************
      use mint_module
      implicit none
      double precision xx(ndimmax),vegas_wgt,f(nintegrals)
      integer ifl

      call sigint(xx,vegas_wgt,ifl,f)
      return
      end


