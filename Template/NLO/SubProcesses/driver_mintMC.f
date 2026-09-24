      Program DRIVER
c**************************************************************************
c     This is the driver for the whole calculation
c**************************************************************************
      use extra_weights
      use mint_module
      use FKSParams
      use process_module
      use scale_module
      implicit none
C
C     CONSTANTS
C
      double precision zero
      parameter       (ZERO = 0d0)
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      integer ncall_virt,ncall_novi
      character*4 abrv
      common /to_abrv/ abrv
C
C     LOCAL
C
      integer i,j,k,l,l1,l2,nndim,nevts,p_label

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
      common /c_granny_counters/ deravg,derstd,dermax,xi_i_fks_ev_der_max
     &     ,y_ij_fks_ev_der_max,ntot_granny,derntot,ncase
      integer                     n_MC_subt_diverge
      common/counter_subt_diverge/n_MC_subt_diverge
      include 'leshouche_decl.inc'
      include 'born_nhel.inc'
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
     $     ,mcatnlo_delta,ebeam(1)+ebeam(2),max_bcol,maxflow_used,ickkw)
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
c Mass-shell stuff. This is MC-dependent
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


      block data timing
c timing statistics
      include "timing_variables.inc"
      data tOLP/0.0/
      data tGenPS/0.0/
      data tBorn/0.0/
      data tIS/0.0/
      data tReal/0.0/
      data tCount/0.0/
      data tFxFx/0.0/
      data tf_nb/0.0/
      data tf_all/0.0/
      data t_as/0.0/
      data tr_s/0.0/
      data tr_pdf/0.0/
      data t_plot/0.0/
      data t_cuts/0.0/
      data t_MC_subt/0.0/
      data t_isum/0.0/
      data t_p_unw/0.0/
      data t_write/0.0/
      data t_coupl/0.0/
      end


      subroutine get_user_params(ncall,nitmax,
     &     ixi_i,iphi_i,iy_ij,SHsep,nevts,p_label,event_weight)
c**********************************************************************
c     Routine to get user specified parameters for run
c**********************************************************************
      use mint_module
      implicit none
c
c     Constants
c
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'run.inc'
c
c     Arguments
c
      integer ncall,nitmax
c
c     Local
c
      integer i, j
      double precision dconfig
c
c     Global
c
      integer             ini_fin_fks
      common/fks_channels/ini_fin_fks
      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel
      logical fillh
      integer mc_hel,ihel
      double precision volh
      common/mc_int2/volh,mc_hel,ihel,fillh
      integer           use_cut
      common /to_weight/use_cut

      integer        lbw(0:nexternal)  !Use of B.W.
      common /to_BW/ lbw

      character*5 abrvinput
      character*4 abrv
      common /to_abrv/ abrv

      logical nbody
      common/cnbody/nbody
c
c To convert diagram number to configuration
c
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
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

      character*7 event_norm
      common /event_normalisation/event_norm
c Les Houches init block (for the <init> info)
      integer maxpup
      parameter(maxpup=100)
      integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
      double precision ebmup,xsecup,xerrup,xmaxup
      common /heprup/ idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &     idwtup,nprup,xsecup(maxpup),xerrup(maxpup),
     &     xmaxup(maxpup),lprup(maxpup)
      double precision dum1,dum2,dum3
      integer p_label,nevts,nevents
      double precision event_weight
c
c MINT stuff
c
      integer ixi_i,iphi_i,iy_ij

c-----
c  Begin Code
c-----
      write(*,'(a)') 'Enter number of events and iterations: '
      read(*,*) ncall,nitmax
      write(*,*) 'Number of events and iterations ',ncall,nitmax

      write(*,'(a)') 'Enter desired fractional accuracy: '
      read(*,*) accuracy
      write(*,*) 'Desired fractional accuracy: ',accuracy

      write(*,*)'Enter alpha, beta for G_soft'
      write(*,*)'  Enter alpha<0 to set G_soft=1 (no ME soft)'
      read(*,*)alsf,besf
      write (*,*) 'for G_soft: alpha=',alsf,', beta=',besf 

      write(*,*)'Enter alpha, beta for G_azi'
      write(*,*)'  Enter alpha>0 to set G_azi=0 (no azi corr)'
      read(*,*)alazi,beazi
      write (*,*) 'for G_azi: alpha=',alazi,', beta=',beazi
      i=2
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
      use_cut=2


      write(*,*) 'Suppress amplitude (0 no, 1 yes)? '
      read(*,*) i
      if (i .eq. 1) then
         multi_channel = .true.
         write(*,*) 'Using suppressed amplitude.'
      else
         multi_channel = .false.
         write(*,*) 'Using full amplitude.'
      endif

      write(*,*) 'Exact helicity sum (0 yes, n = number/event)? '
      read(*,*) i
      if (nincoming.eq.1) then
         write (*,*) 'Sum over helicities in the virtuals'/
     $        /' for decay process'
         mc_hel=0
      elseif (i.eq.0) then
         mc_hel=0
         write (*,*) 'Explicitly summing over helicities'/
     $        /' for the virtuals'
      else
         mc_hel=1
         write(*,*) 'Do MC over helicities for the virtuals'
      endif
      isum_hel = 0

      write(*,'(a)') 'Enter Configuration Number: '
      read(*,*) dconfig
      iconfig = int(dconfig)
      if ( nint(dconfig*10) - iconfig*10 .eq.0 ) then
         ini_fin_fks=0
      elseif ( nint(dconfig*10) -iconfig*10 .eq.1 ) then
         ini_fin_fks=1
      elseif ( nint(dconfig*10) -iconfig*10 .eq.2 ) then
         ini_fin_fks=2
      else
         write (*,*) 'ERROR: invalid configuration number',dconfig
         stop 1
      endif
      do i=1,mapconfig(0,0)
         if (iconfig.eq.mapconfig(i,0)) then
            iconfig=i
            exit
         endif
      enddo
      write(*,*) 'Running Configuration Number: ',iconfig,ini_fin_fks
      nchans=1
      iconfigs(1)=iconfig
      wgt_mult=1d0

      write (*,'(a)') 'Enter running mode for MINT:'
      write (*,'(a)') '0 to set-up grids, 1 to integrate,'//
     &     ' 2 to generate events'
      read (*,*) imode
      write (*,*) 'MINT running mode:',imode
      if (imode.eq.2)then
         write (*,*) 'Generating events, doing only one iteration'
         nitmax=1
      endif

      write (*,'(a)') 'Set the three folding parameters for MINT'
      write (*,'(a)') 'xi_i, y_ij, phi_i'
      read (*,*) ixi_i,iy_ij,iphi_i
      write (*,*)ixi_i,iy_ij,iphi_i


      abrvinput='     '
      write (*,*) "'all ', 'born', 'real', 'virt', 'novi' or 'grid'?"
      write (*,*) "Enter 'born0' or 'virt0' to perform"
      write (*,*) " a pure n-body integration (no S functions)"
      read(*,*) abrvinput
      if(abrvinput(5:5).eq.'0')then
         write (*,*) 'This option is no longer supported:',abrvinput
         stop
        nbody=.true.
      else
        nbody=.false.
      endif
      abrv=abrvinput(1:4)
      if (fks_integrated.eq.1) then
         if (pdg_type_d(1,fks_i_d(1)).eq.-21) then
            write (*,*) 'Process generated with [LOonly=QCD]. '/
     $           /'Setting abrv to "born".'
            abrv='born'
c$$$            if (ickkw.eq.3) then
c$$$               write (*,*) 'FxFx merging not possible with'/
c$$$     $              /' [LOonly=QCD] processes'
c$$$               stop 1
c$$$            endif
         endif
      endif
      if(nbody.and.abrv.ne.'born'.and.abrv.ne.'virt'
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
      lbw(0)=0

      if (imode.eq.2) then
         read(*,*) p_label,nevts
         read(*,*) nevents,event_weight,dum1,dum2,dum3
         read(*,*) NPRUP
         do i=1,NPRUP
            read(*,*)LPRUP(i),dum1,dum2,XSECUP(i),XERRUP(i)
         enddo
      endif
      if (event_norm(1:5).eq.'unity'.or.event_norm(1:3).eq.'sum') then
         IDWTUP=-3
         if (event_norm(1:5).eq.'unity') then
            XMAXUP(1:NPRUP)=1d0
         elseif(event_norm(1:3).eq.'sum') then
            XMAXUP(1:NPRUP)=XMAXUP(1:NPRUP)/nevents
         endif
      else
         IDWTUP=-4
         if (event_norm(1:4).eq.'bias') then
            XMAXUP(1:NPRUP)=-1d0
         else
            XMAXUP(1:NPRUP)=event_weight
         endif
      endif
      if (event_norm(1:5).eq.'unity') then
         event_weight=1d0
      elseif(event_norm(1:3).eq.'sum') then
         event_weight=event_weight/dble(nevents)
      endif
      end



      function sigintF(xx,vegas_wgt,ifl,f)
      use mc_native_context, only: mc_begin_real_point,mc_end_real_point
      use weight_lines
      use mint_module
      use kinematics_module
      use process_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      include 'orders.inc'
      include 'fks_info.inc'
      include 'mc_histories.inc'
      logical firsttime,passcuts,passcuts_nbody,passcuts_n1body
      integer i,j,ifl,proc_map(0:fks_configs,0:fks_configs)
     $     ,nFKS_picked_nbody,nFKS_in,nFKS_out,izero,ione,itwo,mohdr
     $     ,iFKS,sum,partner_picked(fks_configs),first_native_H
      save partner_picked
      double precision xx(ndimmax),vegas_wgt,f(nintegrals),jac,p(0:3
     $     ,nexternal),rwgt,vol,sig,x_local(99),MC_int_wgt,vol1,probne
     $     ,sigintF,n1body_wgt,p_lab(0:3
     $     ,nexternal) ,p_cms(0:3,nexternal),jacPS
      save vol1,proc_map
      integer             ini_fin_fks
      common/fks_channels/ini_fin_fks
      external passcuts
      parameter (izero=0,ione=1,itwo=2,mohdr=-100)
      data firsttime/.true./
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      integer     fold,ifold_counter
      common /cfl/fold,ifold_counter
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer              MCcntcalled
      common/c_MCcntcalled/MCcntcalled
      double precision virtual_over_born
      common /c_vob/   virtual_over_born
      logical       nbody
      common/cnbody/nbody
      integer         nndim
      common/tosigint/nndim
      character*4      abrv
      common /to_abrv/ abrv
      double precision p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $     ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real
      integer ifold_picked
      double precision x_save(ndimmax,max_fold)
      common /c_vegas_x_fold/x_save,ifold_picked
      integer icolup_s(2,nexternal-1),icolup_h(2,nexternal)
      common /colour_connections/ icolup_s,icolup_h
      integer fks_father
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision wgt1
      double precision born_flow_factor
! The same flow is used throughout the folds, so keep its draw probability.
      save born_flow_factor
c
      if (new_point .and. ifl.ne.2) then
         pass_cuts_check=.false.
      endif
      sigintF=0d0
c Find the nFKSprocess for which we compute the Born-like contributions
      if (firsttime) then
         firsttime=.false.
c Determines the proc_map that sets which FKS configuration can be
c summed explicitly and which by MC-ing.
         call setup_proc_map(sum,proc_map,ini_fin_fks)
c For the S-events, we can combine processes when they give identical
c processes at the Born. Make sure we check that we get indeed identical
c IRPOC's
         call find_iproc_map()
c For FxFx or UNLOPS matching with pythia8, set the correct attributes
c for the <event> tag in the LHEF file. "npNLO" are the number of Born
c partons in this multiplicity when running the code at NLO accuracy
c ("npLO" is -1 in that case). When running LO only, invert "npLO" and
c "npNLO".
         call setup_event_attributes
      endif

      if (ifl.eq.0) then
         ifold_counter=1
      elseif(ifl.eq.1) then
         ifold_counter=ifold_counter+1
      endif

      fold=ifl
      if (ifl.eq.0 .or. ifl.eq.1) then
         if (ifl.eq.0) then
            icontr=0
            virt_wgt_mint(0:amp_split_size)=0d0
            born_wgt_mint(0:amp_split_size)=0d0
            virtual_over_born=0d0
            born_flow_picked=0
         endif
         MCcntcalled=0
         wgt_me_real=0d0
         wgt_me_born=0d0
         if (ickkw.eq.3) call set_FxFx_scale(0,p)
         call update_vegas_x(xx,x_local)
         do i=1,nndim
            x_save(i,ifold_counter)=x_local(i)
         enddo
         call born_spread_set_point(x_local(ndim-2)**2,
     $        x_local(ndim-1)**2)
         born_spread_bin_fold(ifold_counter)=born_spread_current_bin
         born_spread_sector_fold(ifold_counter)=1
         if (ifl.eq.0)
     &        call get_MC_integer(1,proc_map(0,0),proc_map(0,1),vol1)

c The nbody contributions
         if (abrv.eq.'real') goto 11
         nbody=.true.
         calculatedBorn=.false.
c Pick the first one because that's the one with the soft singularity
         nFKS_picked_nbody=proc_map(proc_map(0,1),1)
         if (sum.eq.0) then
c For sum=0, determine nFKSprocess so that the soft limit gives a non-zero Born
            nFKS_in=nFKS_picked_nbody
            call get_born_nFKSprocess(nFKS_in,nFKS_out)
            nFKS_picked_nbody=nFKS_out
         endif
         call update_fks_dir(nFKS_picked_nbody)
c Keep the Born sector even when later real-emission maps change nFKS.
         born_spread_sector_fold(ifold_counter)=nFKS_picked_nbody
         if (ini_fin_fks.eq.0) then
            jac=1d0
         else
            jac=0.5d0
         endif
c Also the Born needs to be included in the Importance Sampling over the
c FKS configurations (for the shower scale) (multiply by
c 1/proc_map(0,0)*vol1)
         jac=jac/(proc_map(0,0)*vol1)
         call generate_momenta(nndim,iconfig,jac,x_local,p,p_lab,p_cms)
         if (p_born(0,1).lt.0d0) goto 12
         call compute_prefactors_nbody(vegas_wgt)
         call set_cms_stuff(izero)
         if (ickkw.eq.3) call set_FxFx_scale(1,p1_cnt(0,1,0))
         passcuts_nbody=passcuts(p1_cnt(0,1,0),rwgt)
            
         if (passcuts_nbody) then
            pass_cuts_check=.true.
            call set_born_spread_point(x_local(ndim-2),
     $           x_local(ndim-1),ifold_counter)
            call set_alphaS(p1_cnt(0,1,0))
            call include_multichannel_enhance(1)
            if (abrv.eq.'born') then
               ! Doing only the Born contribution.
               call compute_born
               if (ifl.eq.0) call get_born_flow(born_flow_picked
     $              ,born_flow_factor)
               call Bornonly_shower_scale(p_born,born_flow_picked)
               emsca_S(nFKS_picked_nbody,ifold_counter,1:ndelS,1:ndelS)
     $              =get_random_shower_dipole_scale()
            elseif (abrv(1:2).eq.'vi') then
               ! Doing only the Virtual contribution (could be because
               ! we are generating a virtual event).
               call compute_nbody_noborn
               if (ifl.eq.0) call get_born_flow(born_flow_picked
     $              ,born_flow_factor)
               call compute_shower_scale_nbody(p_born,born_flow_picked)
               emsca_S(nFKS_picked_nbody,ifold_counter,1:ndelS,1:ndelS)
     $              =get_random_shower_dipole_scale()
            else
               ! Normal: all contributions included. Determine the
               ! shower scale when looping over FKS configurations.
               call compute_born
               if (abrv.ne.'bovi') call compute_ewsudakov
               call compute_nbody_noborn
               ! only for ifl==0, since we want the same flow for each fold.
               if (ifl.eq.0) call get_born_flow(born_flow_picked
     $              ,born_flow_factor)
               ! We need to fill emsca_S(iFKS_born) with a value that
               ! will be used if we are in the dead-zone. If we are not
               ! in the dead-zone, this will not be used (or
               ! overwritten).
               call compute_shower_scale_nbody(p_born,born_flow_picked)
               emsca_S(nFKS_picked_nbody,ifold_counter,1:ndelS,1:ndelS)
     $              =get_random_shower_dipole_scale()
            endif
         elseif (ifl.eq.0) then
            call sborn_native(p_born,wgt1)
            call get_born_flow(born_flow_picked,born_flow_factor)
! give it a negative value so that we can keep track of the fact that
! this was obtained with momenta that do not pass the cuts.
            born_flow_picked=-born_flow_picked
         endif
         
 11      continue
c The n+1-body contributions (including counter terms)
         if (abrv.eq.'born'.or.abrv(1:2).eq.'vi') goto 12
c Set calculated Born to zero to prevent numerical inaccuracies: not
c always exactly the same momenta in computation of Born when computed
c for different nFKSprocess.
         if(sum.eq.0) calculatedBorn=.false.
         nbody=.false.
         do i=1,proc_map(proc_map(0,1),0)
            wgt_me_real=0d0
            wgt_me_born=0d0
            iFKS=proc_map(proc_map(0,1),i)
            call update_fks_dir(iFKS)
            if (born_flow_picked.gt.0) then
!     Consider all flows for the shower scale assignment (with
!     assignements only needed for the dipoles where the fks-mother is
!     one end of the dipole line)
               fks_father=min(i_fks,j_fks)
               call compute_shower_scale_nbody(p_born,-fks_father) 
!     assign emsca_S: we know flow (from driver_mintMC) and the
!     father. Therefore the partner is fixed (except when father is a gluon
!     (then there is a two-fold ambiguity)). Determine the partner:

!     Determine the partner of the father (keep it the same for each fold)
               if (ifl.eq.0) call determine_partner(born_flow_picked
     $              ,partner_picked(iFKS))
!     The shower scale to be used in the event file (if it's an S-event and
!     fks_picked will be iFKS):
               if (.not.mcatnlo_delta) then
                  emsca_S(iFKS,ifold_counter,1:ndelS,1:ndelS)
     $                 =shower_scale_nbody(fks_father
     $                 ,partner_picked(iFKS))
               else
                  emsca_S(iFKS,ifold_counter,1:ndelS,1:ndelS)
     $                 =shower_scale_nbody(1:ndelS,1:ndelS)
               endif
            endif
               
            probne=1d0
            gfactsf=1.d0
            gfactcl=1.d0
            MCcntcalled=0
            icolup_s(1,1)=-1    ! set colour connection to -1: i.e., complete_xmcsubt has not been called
! Apply the outer-sector sampling weight to the counterevents as well.
            jacPS=1d0/vol1
            call generate_momenta(nndim,iconfig,jacPS,x_local,p,p_lab
     $           ,p_cms)
            jac=jacPS
            jacPS=jacPS*vol1
c Every contribution has to have a viable set of Born momenta (even if
c counter-event momenta do not exist).
            if (p_born(0,1).lt.0d0) cycle

! fill the valid_dipole array and fill the H-event shower scale array            
            call init_process_module_n1body_wrapper(born_flow_picked)
            call compute_shower_scale_n1body(p,i_fks,j_fks)
! The shower scale to be used in the event file (if it's an H-event and
! fks_picked will be iFKS). If the emission is hard, we take the dipole
! scale as upper boundary for the next emissions. On the other hand, to
! make sure that (soft/collinear) real-emission still "cancels the
! divergence in the virtual" these configurations should have a shower
! scale related to the underlying S-event.
            if (born_flow_picked.gt.0) then
               ! TODO: maybe we should also go here when the n-body does not pass the cuts. CHECK.
               if (.not.mcatnlo_delta) then
                  emsca_H(iFKS,ifold_counter,1:ndelH,1:ndelH)
     $                 =max(shower_scale_n1body(i_fks,j_fks),
     $                 shower_scale_nbody(fks_father
     $                 ,partner_picked(iFKS)))
               else
! in the case of MC@NLO-delta, an H-event contribution is by definition
! 'hard', and we should use the corresponding dipole scale for
! subsequent showering.
                  emsca_H(iFKS,ifold_counter,1:ndelH,1:ndelH)
     $                 =shower_scale_n1body(1:ndelH,1:ndelH)
               endif
            else ! we have no Born, so take the n+1-body dipole(s) as
                 ! starting scale.
               if (.not.mcatnlo_delta) then
                  emsca_H(iFKS,ifold_counter,1:ndelH,1:ndelH)
     $                 =shower_scale_n1body(i_fks,j_fks)
               else
                  emsca_H(iFKS,ifold_counter,1:ndelH,1:ndelH)
     $                 =shower_scale_n1body(1:ndelH,1:ndelH)
               endif
            endif
c Compute the n1-body prefactors
            call compute_prefactors_n1body(vegas_wgt,jac)
! This flow was drawn with q_c=p_c at the common outer Born point.
            call include_born_flow_weight(born_flow_factor,
     $           born_flow_factor)
c Include the FxFx Sudakovs into the prefactors
            if (ickkw.eq.3) then
               call set_FxFx_scale(0,p) ! reset the FxFx scales
               call set_cms_stuff(izero)
               call set_FxFx_scale(2,p1_cnt(0,1,0))
               call set_cms_stuff(mohdr)
               if (p(0,1).gt.0d0) call set_FxFx_scale(3,p)
            endif
c check if event or counter-event passes cuts
            call set_cms_stuff(izero)
            if (ickkw.eq.3) call set_FxFx_scale(-2,p1_cnt(0,1,0))
            passcuts_nbody=passcuts(p1_cnt(0,1,0),rwgt)
            if (passcuts_nbody .and. (born_flow_picked .le.0)) then
               write (*,*) 'something funny is going on: passing cuts,'
     $              //' but no born_flow assigned.'
               write (*,*) born_flow_picked
               stop 1
            endif
            
            call set_cms_stuff(mohdr)
            if (ickkw.eq.3) call set_FxFx_scale(-3,p)
            passcuts_n1body=passcuts(p,rwgt)
! Another native Born projection can pass cuts even if this one does
! not. The complete H sum must still be evaluated in that case.
            if (.not.(passcuts_nbody.or.passcuts_n1body) .and.
     $           (ickkw.eq.4 .or. abrv.eq.'real')) cycle
            first_native_H=icontr+1
! Share only full real amplitudes at this physical point. Each native
! history retains its own counterevents, scales and regulated factors.
            call mc_begin_real_point(p)
! The complete H sum below replaces the outer H records. Suppress them
! here, preserving the original path if no real point can be repartitioned.
            mc_S_only=ickkw.ne.4.and.abrv.ne.'real'.and.
     $           p(0,1).gt.0d0.and.jacPS.gt.0d0.and.MC_HIST_COUNT.gt.0
            call compute_native_NLOPS_weights(p,p_lab,p_cms,jacPS,
     $           passcuts_nbody,passcuts_n1body,probne)
            mc_S_only=.false.
            if (ickkw.ne.4 .and. abrv.ne.'real') then
               call repartition_MC_H(first_native_H,x_local,p,p_lab,
     $              p_cms,jacPS,vegas_wgt,1d0/vol1,born_flow_factor)
            endif
            call mc_end_real_point()
         enddo
         call apply_born_spread_weight(ifold_counter)
 12      continue
      elseif(ifl.eq.2) then
         if (ifold_counter .ne.
     $       ifold(ifold_energy)*ifold(ifold_yij)*ifold(ifold_phi)) then
            write (*,*) "ERROR in folding parameters (driver_mintMC.f)"
     $           ,ifold_counter,ifold_energy,ifold_yij,ifold_phi
            write (*,*) ifold(:)
            stop 1
         endif
c Special check: in rare cases there can be S-event contributions,
c without a single FKS configuration that contains a soft singularity
c passing cuts (this only happens if n-body configuration does not pass
c the cuts and DELTA (from complete_xmcsubt) is not equal to 1). Need to
c add a bogus contribution corresponding to an FKS configuration that
c contains a soft singularity to make sure that the code continues
c correctly.
         ! TODO: HOW CAN THIS HAPPEN? delta can only be not equal to one if n-body passes the cuts...
         call special_check_SoftSing(proc_map(proc_map(0,1),1))
c Include PDFs and alpha_S and reweight to include the uncertainties
         call include_PDF_and_alphas
c Include the weight from the bias_function
         call include_bias_wgt
c Sum the contributions that can be summed before taking the ABS value
         call sum_identical_contributions
         call fill_mint_function_NLOPS(f,n1body_wgt)
         call fill_MC_integer(1,proc_map(0,1),n1body_wgt*vol1)
      endif

      return
      end

      subroutine repartition_MC_H(first_native,x_outer,p,p_lab,p_cms,
     $     jacPS,vegas_wgt,sampling_wgt,born_flow_factor)
! At a fixed real point form Hhat_a = S_a sum_b P_b (S_b R - M_b).
! The ordinary S records have already been made and are not changed.
! Each M_b includes its native G replacement, luminosities and Born map.
! Each inner history samples its OWN colour flow and includes 1/q_b,c.
! The outer event colour is only the event owner, not an inner proposal.
! Thus the colour-sampled summand is
! P_b,c*(p_b,c*S_b*R-M_b,c)/q_b,c, with M_b,c already flow-weighted.
      use mc_native_context, only: native_metadata,set_native_history,
     $     native_mapping
      use weight_lines, only: icontr,H_event,wgt,event_nFKS,momenta,
     $     momenta_m,y_bst,need_match,mc_H_only
      use mint_module, only: ndim,iconfig
      use process_module, only: ndelH
      use kinematics_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'genps.inc'
      include 'run.inc'
      include 'fks_symmetry.inc'
      include 'mc_histories.inc'
      integer first_native,last_native,first_alt,owner,iFKS,ii,jj,ihist,
     $     ict,flow_save,called_save,owner_match(nexternal),
     $     outer_config,native_config
      double precision x_outer(99),p(0:3,nexternal),
     $     p_lab(0:3,nexternal),p_cms(0:3,nexternal),jacPS,vegas_wgt,
     $     sampling_wgt,born_flow_factor,outer_measure,native_measure,
     $     sector_weight,factor,xx(99),jac_native,xbjrk_born(2),
     $     p_flipped(0:3,nexternal),pn(0:3,nexternal),
     $     pn_lab(0:3,nexternal),pn_cms(0:3,nexternal),probne_native,
     $     flow_factor_native,rwgt,outer_boost,gfun_save(3),
     $     outer_channel,mc_outer_channel_weight
      double precision nbody_scales_save(nexternal-1,nexternal-1,3),
     $     n1body_scales_save(nexternal,nexternal),
     $     emsca_save(fks_configs,ndelH,ndelH)
      logical cuts_born,cuts_real,passcuts,native_valid
      double precision fks_Sij
      external fks_Sij,passcuts
      double precision born_weight,replay_tolerance
      external mc_outer_channel_weight
      integer nFKSprocess,i_fks,j_fks
      common/c_nFKSprocess/nFKSprocess
      common/fks_indices/i_fks,j_fks
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision pmass(nexternal)
      common/to_mass/pmass
      double precision xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3),
     $     p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision xinorm_ev
      common/cxinormev/xinorm_ev
      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     $     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     $     fkssymmetryfactorDeg,ngluons,nquarks,nphotons
      double precision p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2),
     $     pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     $     sqrtshat,shat
      integer fold,ifold_counter,MCcntcalled
      common/cfl/fold,ifold_counter
      common/c_MCcntcalled/MCcntcalled
      integer need_matching_S(nexternal),need_matching_H(nexternal),
     $     need_matching_cuts(nexternal),matching_save(nexternal,3)
      common/c_need_matching/need_matching_S,need_matching_H,
     $     need_matching_cuts
      integer icolup_s(2,nexternal-1),icolup_h(2,nexternal),
     $     colours_s_save(2,nexternal-1),colours_h_save(2,nexternal)
      common/colour_connections/icolup_s,icolup_h
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

      if (p(0,1).le.0d0 .or. jacPS.le.0d0) return
      if (MC_HIST_COUNT.eq.0) return
      call set_cms_stuff(-100)
      owner=nFKSprocess
      if (.not.MC_HIST_COMPLETE(owner)) then
         write (*,*) 'Incomplete MC history sum for FKS sector',owner
         write (*,*) 'Required global histories are unresolved or',
     $        ' ambiguous; see Source/BornSupport/registry.json'
         stop 1
      endif
      outer_config=iconfig
      outer_channel=mc_outer_channel_weight()
      outer_boost=ybst_til_tolab
      owner_match=need_matching_H
      matching_save(:,1)=need_matching_S
      matching_save(:,2)=need_matching_H
      matching_save(:,3)=need_matching_cuts
      sector_weight=fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
      last_native=icontr
      do ict=first_native,last_native
         if (H_event(ict)) wgt(:,ict)=0d0
      enddo
      if (sector_weight.eq.0d0) return

! K_a is the full outer real measure, including its sampling probability
! and orbit multiplicity, and the outer matrix-element channel weight.
      outer_measure=xinorm_ev*xi_i_fks_ev*jacPS*vegas_wgt*
     $     sampling_wgt*fkssymmetryfactor*outer_channel
      if (outer_measure.le.0d0) return
      if (fkssymmetryfactor.ne.
     $     dble(FKS_FAC_I_D(owner)*FKS_FAC_J_D(owner))) then
         write (*,*) 'Incorrect outer MC H orbit factor',owner
         stop 1
      endif
      if (MC_HIST_OWN(owner).lt.MC_HIST_FIRST(owner) .or.
     $     MC_HIST_OWN(owner).gt.MC_HIST_LAST(owner)) then
         write (*,*) 'Missing native entry in MC history table',owner
         stop 1
      endif
      gfun_save=[gfactsf,gfactcl,gfactazi]
      flow_save=born_flow_picked
      called_save=MCcntcalled
      colours_s_save=icolup_s
      colours_h_save=icolup_h
      nbody_scales_save(:,:,1)=shower_scale_nbody
      nbody_scales_save(:,:,2)=shower_scale_nbody_min
      nbody_scales_save(:,:,3)=shower_scale_nbody_max
      n1body_scales_save=shower_scale_n1body
      emsca_save=emsca_H(:,ifold_counter,:,:)
      mc_H_only=.true.
      native_mapping=.true.

! The exporter supplies unique ordered histories, including both gg
! orientations, with native Born/order identities and checked label maps.
! Every row has unit multiplicity. Recompute the owner too, using the
! same measure conversion and channel-free density as all other rows.
      do ihist=MC_HIST_FIRST(owner),MC_HIST_LAST(owner)
         iFKS=MC_HIST_NATIVE(ihist)
         ii=MC_HIST_I(ihist)
         jj=MC_HIST_J(ihist)
         iconfig=1
         call update_fks_dir(iFKS)
         call set_native_history(ihist)
         call init_process_module_nbody_wrapper()
         call update_coltype_and_charge(iFKS,i_fks,j_fks)
         if (fkssymmetryfactor.ne.
     $        dble(FKS_FAC_I_D(iFKS)*FKS_FAC_J_D(iFKS)) .or.
     $        MC_HIST_PERM(i_fks,ihist).ne.ii .or.
     $        MC_HIST_PERM(j_fks,ihist).ne.jj) then
            write (*,*) 'Incorrect native MC H orbit',owner,ihist
            stop 1
         endif
         call apply_momentum_permutation(MC_HIST_PERM(:,ihist),
     $        p_lab,p_flipped)
! Select the first native mapping with an invertible physical point.
! The outer channel index has no meaning in a different Born topology.
         native_valid=.false.
         do native_config=1,native_metadata%configurations(0)
            iconfig=native_config
            xx=0d0
            jac_native=1d0
            call generate_lab_momenta_inverse(ndim,iconfig,
     $           jac_native,xx,p_flipped,xbjrk_born)
            if(jac_native.le.0d0)cycle

! Inversion alone does not fill the native FKS counterevents. Replay
! the forward map to obtain those points AND their limit measures.
            calculatedBorn=.false.
            jac_native=1d0
            call generate_momenta(ndim,iconfig,jac_native,xx,
     $           pn,pn_lab,pn_cms)
            if (jac_native.le.0d0 .or. pn(0,1).le.0d0 .or.
     $           p_born(0,1).le.0d0)cycle
! The inverse azimuth is ill conditioned when the two FKS daughters
! are antipodal. Roundoff can then shift the replay by a few parts in
! 10**6 even though both maps describe the same physical point.
            replay_tolerance=1d-7
            if (xx(ndim-1).gt.1d0-1d-10)
     $           replay_tolerance=1d-5
            if (maxval(abs(pn_lab-p_flipped)).gt.
     $           replay_tolerance*max(1d0,
     $           maxval(abs(p_flipped))))cycle
            native_valid=.true.
            exit
         enddo
         if (.not.native_valid) then
            write (*,*) 'No native MC H mapping passes inversion',
     $           ' and forward momentum checks',
     $           owner,iFKS,ii,jj
            stop 1
         endif
         native_measure=xinorm_ev*xi_i_fks_ev*jac_native*
     $        fkssymmetryfactor
         if (native_measure.le.0d0) then
            write (*,*) 'Invalid native MC H measure',
     $           owner,iFKS,ii,jj,native_measure
            stop 1
         endif

! Divide out the native real measure of the COMPLETE generated H weight
! and insert K_a. This is not an extra Jacobian on a raw MC density:
! its native K_b cancels exactly. The counter/real measure ratios inside
! the G replacement, however, must be retained. The inner orbit factor
! cancels too, since the labelled histories are explicitly enumerated.
         factor=sector_weight*outer_measure/native_measure
         MCcntcalled=0
         call fill_kinematics_module(pn_cms,i_fks,j_fks,
     $        xi_i_fks_ev,y_ij_fks_ev,pmass(j_fks),.false.)
         call compute_prefactors_n1body(1d0,jac_native)
         if (ickkw.eq.3) then
            call set_FxFx_scale(0,pn)
            call set_cms_stuff(0)
            call set_FxFx_scale(2,p1_cnt(0,1,0))
            call set_cms_stuff(-100)
            call set_FxFx_scale(3,pn)
         endif
         call set_cms_stuff(0)
         if (ickkw.eq.3) call set_FxFx_scale(-2,p1_cnt(0,1,0))
! Sample in this history's own Born basis. Reusing the outer label
! would require a flow map and support at a different Born point.
! q_b,c=p_b,c here; no additional outer 1/q_a,c belongs on this term.
         call set_alphaS(p1_cnt(0,1,0))
         calculatedBorn=.false.
         call sborn_native(p_born,born_weight)
         call get_born_flow(born_flow_picked,flow_factor_native)
         calculatedBorn=.false.
         call include_born_flow_weight(flow_factor_native,
     $        flow_factor_native)
         call init_process_module_n1body_wrapper(born_flow_picked)
         call compute_shower_scale_nbody(p_born,-fksfather)
         call compute_shower_scale_n1body(pn,i_fks,j_fks)
         cuts_born=passcuts(p1_cnt(0,1,0),rwgt)
         call set_cms_stuff(-100)
         if (ickkw.eq.3) call set_FxFx_scale(-3,pn)
         cuts_real=passcuts(pn,rwgt)
         first_alt=icontr+1
         call compute_native_NLOPS_weights(pn,pn_lab,pn_cms,
     $        jac_native,cuts_born,cuts_real,probne_native)
         do ict=first_alt,icontr
            if (.not.H_event(ict)) then
               write (*,*) 'S event entered the inner MC H sum',ihist
               stop 1
            endif
            wgt(:,ict)=wgt(:,ict)*factor
! Keep BOTH native momentum sets for ME reweighting, expressed in the
! outer frame. Only event kinematics and shower ownership are outer.
            event_nFKS(ict)=owner
            call boost_n1_to_lab(momenta_m(:,:,1,ict),pn_cms,
     $           y_bst(ict)-outer_boost)
            momenta_m(:,:,1,ict)=pn_cms
            call boost_n1_to_lab(momenta_m(:,:,2,ict),pn_cms,
     $           y_bst(ict)-outer_boost)
            momenta_m(:,:,2,ict)=pn_cms
            momenta(:,:,ict)=p
            y_bst(ict)=outer_boost
            need_match(:,ict)=owner_match
         enddo
      enddo
! Replaying the saved OUTER random numbers restores all FKS event and
! counterevent COMMON blocks, including Born/spin and Bjorken data.
      call set_native_history(0)
      native_mapping=.false.
      iconfig=outer_config
      call update_fks_dir(owner)
      call init_process_module_nbody_wrapper()
      call update_coltype_and_charge(owner,i_fks,j_fks)
      calculatedBorn=.false.
      jac_native=sampling_wgt
      call generate_momenta(ndim,iconfig,jac_native,x_outer,
     $     pn,pn_lab,pn_cms)
      if (jac_native.le.0d0 .or. p_born(0,1).le.0d0) then
         write (*,*) 'Could not restore outer FKS point after MC H sum'
         stop 1
      endif
      born_flow_picked=flow_save
      call init_process_module_n1body_wrapper(born_flow_picked)
      shower_scale_nbody=nbody_scales_save(:,:,1)
      shower_scale_nbody_min=nbody_scales_save(:,:,2)
      shower_scale_nbody_max=nbody_scales_save(:,:,3)
      shower_scale_n1body=n1body_scales_save
      emsca_H(:,ifold_counter,:,:)=emsca_save
      icolup_s=colours_s_save
      icolup_h=colours_h_save
      MCcntcalled=called_save
      call fill_kinematics_module(p_cms,i_fks,j_fks,
     $     xi_i_fks_ev,y_ij_fks_ev,pmass(j_fks),.false.)
      gfactsf=gfun_save(1)
      gfactcl=gfun_save(2)
      gfactazi=gfun_save(3)
      call compute_prefactors_n1body(vegas_wgt,jac_native)
      call include_born_flow_weight(born_flow_factor,born_flow_factor)
      if (ickkw.eq.3) then
         call set_FxFx_scale(0,p)
         call set_cms_stuff(0)
         call set_FxFx_scale(2,p1_cnt(0,1,0))
         call set_cms_stuff(-100)
         call set_FxFx_scale(3,p)
      endif
      call set_cms_stuff(-100)
      call set_alphaS(p)
      calculatedBorn=.false.
      need_matching_S=matching_save(:,1)
      need_matching_H=matching_save(:,2)
      need_matching_cuts=matching_save(:,3)
      mc_H_only=.false.
      end

      subroutine init_process_module_nbody_wrapper()
      use process_module
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'born_nhel.inc'
      integer iFKS,colour(1:nexternal-1),i,j,k,get_color
      external get_color
      double precision mass(1:nexternal-1),get_mass_from_id
      external get_mass_from_id
      logical valid_dipole(1:nexternal-1,1:nexternal-1,1:max_bcol)
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      include 'born_leshouche.inc'

      do i=1,nexternal-1
         mass(i)=get_mass_from_id(idup(i,1))
         colour(i)=get_color(idup(i,1))
      enddo
      valid_dipole=.false.
      do k=1,max_bcol
         do j=1,nexternal-1
            if (icolup(1,j,k).eq.0 .and. icolup(2,j,k).eq.0) cycle
            do i=1,nexternal-1
               if (i.eq.j) cycle
               if (icolup(1,i,k).eq.0 .and. icolup(2,i,k).eq.0) cycle
               if ( (abs(icolup(1,i,k)).eq.abs(icolup(1,j,k)).and.icolup(1,i,k).ne.0) .or.
     &              (abs(icolup(1,i,k)).eq.abs(icolup(2,j,k)).and.icolup(1,i,k).ne.0) .or.
     &              (abs(icolup(2,i,k)).eq.abs(icolup(1,j,k)).and.icolup(2,i,k).ne.0) .or.
     &              (abs(icolup(2,i,k)).eq.abs(icolup(2,j,k)).and.icolup(2,i,k).ne.0) ) then
                  valid_dipole(i,j,k)=.true.
               endif
            enddo
         enddo
      enddo
      call init_process_module_nbody(nexternal-1,mass,colour
     $     ,max_bcol,valid_dipole)
      
      end

      
      subroutine init_process_module_n1body_wrapper(bornflow)
      use process_module
      use weight_lines, only: mc_H_only
      use scale_module, only: event_colour_H
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      integer iFKS,colour(1:nexternal),i,j,k,get_color,bornflow
      double precision mass(1:nexternal),get_mass_from_id
      external get_color
      external get_mass_from_id
      logical valid_dipole(1:nexternal,1:nexternal)
      integer icolup(1:2,1:nexternal)
      integer jpart(7,-nexternal+3:2*nexternal-3)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     dummy(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,dummy,niprocs
      integer nFKSprocess,fold,ifold_counter
      common/c_nFKSprocess/nFKSprocess
      common/cfl/fold,ifold_counter

      if (bornflow.ne.0) then
         ! take ABS because bornflow is negative if n-body did not pass the cuts
! Keep the same colour-insertion variate throughout the native H sum
! and its outer-state restoration; do not redraw the event's colour.
         call fill_icolor_H(abs(bornflow),jpart,.not.mc_H_only)
      else
         write (*,*) 'Born-flow not set in n1body_wrapper'
         stop 1
      endif
      do i=1,nexternal
         ICOLUP(1,i)=jpart(4,i)
         ICOLUP(2,i)=jpart(5,i)
      enddo
! Keep the outer colour assignment with its sector and fold. Inner
! histories and their restoration must not overwrite event ownership.
      if (.not.mc_H_only) then
         event_colour_H(:,:,nFKSprocess,ifold_counter)=ICOLUP
      endif
      
      do i=1,nexternal
         mass(i)=get_mass_from_id(idup(i,1))
         colour(i)=get_color(idup(i,1))
      enddo
      valid_dipole=.false.
      do j=1,nexternal
         if (icolup(1,j).eq.0 .and. icolup(2,j).eq.0) cycle
         do i=1,nexternal
            if (i.eq.j) cycle
            if (icolup(1,i).eq.0 .and. icolup(2,i).eq.0) cycle
            if ( (abs(icolup(1,i)).eq.abs(icolup(1,j)).and.icolup(1,i).ne.0) .or.
     &           (abs(icolup(1,i)).eq.abs(icolup(2,j)).and.icolup(1,i).ne.0) .or.
     &           (abs(icolup(2,i)).eq.abs(icolup(1,j)).and.icolup(2,i).ne.0) .or.
     &           (abs(icolup(2,i)).eq.abs(icolup(2,j)).and.icolup(2,i).ne.0) ) then
               valid_dipole(i,j)=.true.
            endif
         enddo
      enddo
      
      call init_process_module_n1body(nexternal,mass,colour
     $     ,maxflow,valid_dipole)
      
      end
      

      subroutine setup_proc_map(sum,proc_map,ini_fin_fks)
c Determines the proc_map that sets which FKS configuration can be
c summed explicitly and which by MC-ing.
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      double precision lum,dlum
      external dlum
      logical found_ini1,found_ini2,found_fnl
      integer proc_map(0:fks_configs,0:fks_configs)
     $     ,j_fks_proc(fks_configs),i_fks_pdg_proc(fks_configs)
     $     ,j_fks_pdg_proc(fks_configs),i,sum,j,ini_fin_fks
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links
      sum=3
      if (ickkw.eq.4) then
         sum=0
         write (*,*)'Using ickkw=4, include only 1 FKS dir per'/
     $        /' Born PS point (sum=0)'
      endif
      do nFKSprocess=1,fks_integrated
         call fks_inc_chooser()
c Set Bjorken x's to some random value before calling the dlum() function
         xbk(1)=0.5d0
         xbk(2)=0.5d0
         lum=dlum()  ! updates IPROC
      enddo
      write (*,*) 'Total number of FKS directories is', fks_configs
c For sum over identical FKS pairs, need to find the identical structures
      if (sum.eq.3) then
c MC over FKS pairs that have soft singularity
         proc_map(0,0)=0
         do i=1,fks_integrated
            proc_map(i,0)=0
            i_fks_pdg_proc(i)=0
            j_fks_pdg_proc(i)=0
            j_fks_proc(i)=0
         enddo
c First find all the nFKSprocesses that have a soft singularity and put
c them in the process map
         do nFKSprocess=1,fks_integrated
            call fks_inc_chooser()
            if (ini_fin_fks.eq.1 .and. j_fks.le.nincoming) cycle
            if (ini_fin_fks.eq.2 .and. j_fks.gt.nincoming) cycle
            if (need_color_links.or.need_charge_links) then
               proc_map(0,0)=proc_map(0,0)+1
               proc_map(proc_map(0,0),0)=proc_map(proc_map(0,0),0)+1
               proc_map(proc_map(0,0),proc_map(proc_map(0,0),0))
     $              =nFKSprocess
               i_fks_pdg_proc(proc_map(0,0))=PDG_type(i_fks)
               j_fks_pdg_proc(proc_map(0,0))=PDG_type(j_fks)
               j_fks_proc(proc_map(0,0))=j_fks
            endif
         enddo
c Check to make sure that there is at most two initial and one final
c state all gluon
         found_ini1=.false.
         found_ini2=.false.
         found_fnl=.false.
         do i=1,proc_map(0,0)
            if ((abs(i_fks_pdg_proc(i)).eq.21.or.i_fks_pdg_proc(i).eq.22)
     &       .and. j_fks_proc(i).eq.1 .and. .not.found_ini1) then
               found_ini1=.true.
            elseif ((abs(i_fks_pdg_proc(i)).eq.21.or.i_fks_pdg_proc(i).eq.22)
     $       .and. j_fks_proc(i).eq.1.and. found_ini1) then
               write (*,*)'Initial state 1 g->gg already'/
     $              /' found in driver_mintMC'
               write (*,*) i_fks_pdg_proc
               write (*,*) j_fks_pdg_proc
               write (*,*) j_fks_proc
               stop
            elseif ((abs(i_fks_pdg_proc(i)).eq.21.or.i_fks_pdg_proc(i).eq.22)
     $       .and. j_fks_proc(i).eq.2.and. .not.found_ini2) then
               found_ini2=.true.
            elseif ((abs(i_fks_pdg_proc(i)).eq.21.or.i_fks_pdg_proc(i).eq.22)
     $       .and. j_fks_proc(i).eq.2.and. found_ini2) then
               write (*,*)'Initial state 2 g->gg already'/
     $              /' found in driver_mintMC'
               write (*,*) i_fks_pdg_proc
               write (*,*) j_fks_pdg_proc
               write (*,*) j_fks_proc
               stop
            elseif (abs(i_fks_pdg_proc(i)).eq.21 .and.
     $              j_fks_pdg_proc(i).eq.21 .and.
     $              j_fks_proc(i).gt.nincoming .and. .not.found_fnl)
     $              then
               found_fnl=.true.
            elseif (abs(i_fks_pdg_proc(i)).eq.21 .and.
     $              j_fks_pdg_proc(i).eq.21 .and.
     $              j_fks_proc(i).gt.nincoming .and. found_fnl) then
               write (*,*)
     &              'Final state g->gg already found in driver_mintMC'
               write (*,*) i_fks_pdg_proc
               write (*,*) j_fks_pdg_proc
               write (*,*) j_fks_proc
               stop
            endif
         enddo
c Loop again, and identify the nFKSprocesses that do not have a soft
c singularity and put them together with the corresponding gluon to
c gluons splitting
         do nFKSprocess=1,fks_integrated
            call fks_inc_chooser()
            if (ini_fin_fks.eq.1 .and. j_fks.le.nincoming) cycle
            if (ini_fin_fks.eq.2 .and. j_fks.gt.nincoming) cycle
            if (.not.(need_color_links.or.need_charge_links)) then
               if (j_fks.eq.1 .and. found_ini1) then
                  do i=1,proc_map(0,0)
                     if ((abs(i_fks_pdg_proc(i)).eq.21.or.i_fks_pdg_proc(i).eq.22)
     $                    .and. j_fks_proc(i).eq.1) then
                        proc_map(i,0)=proc_map(i,0)+1
                        proc_map(i,proc_map(i,0))=nFKSprocess
                        exit
                     endif
                  enddo
               elseif (j_fks.eq.2 .and. found_ini2) then
                  do i=1,proc_map(0,0)
                     if ((abs(i_fks_pdg_proc(i)).eq.21.or.i_fks_pdg_proc(i).eq.22)
     $                   .and. j_fks_proc(i).eq.2) then
                        proc_map(i,0)=proc_map(i,0)+1
                        proc_map(i,proc_map(i,0))=nFKSprocess
                        exit
                     endif
                  enddo
               elseif (j_fks.gt.nincoming .and. found_fnl) then
                  do i=1,proc_map(0,0)
                     if (abs(i_fks_pdg_proc(i)).eq.21 .and.
     $                    j_fks_pdg_proc(i).eq.21.and.
     $                    j_fks_proc(i).gt.nincoming) then
                        proc_map(i,0)=proc_map(i,0)+1
                        proc_map(i,proc_map(i,0))=nFKSprocess
                        exit
                     endif
                  enddo
               else
                  write (*,*) 'Driver_mintMC: inconsistent process'
                  write (*,*) 'This process has nFKSprocesses'/
     $                 /' without soft singularities, but not a'/
     $                 /' corresponding g->gg splitting that has a'/
     $                 /' soft singularity.',found_ini1,found_ini2
     $                 ,found_fnl
                  do i=1,proc_map(0,0)
                     write (*,*) i,'-->',proc_map(i,0),':',
     &                    (proc_map(i,j),j=1,proc_map(i,0))
                  enddo
                  stop
               endif
            endif
         enddo
      elseif (sum.eq.0 .and. ickkw.eq.4) then
c MC over FKS directories (1 FKS directory per nbody PS point)
         proc_map(0,0)=fks_integrated
         do i=1,fks_integrated
            proc_map(i,0)=1
            proc_map(i,1)=i
         enddo
      else
         write (*,*) 'sum not known in driver_mintMC.f',sum
         stop
      endif
      write (*,*) 'FKS process map (sum=',sum,') :'
      do i=1,proc_map(0,0)
         write (*,*) i,'-->',proc_map(i,0),':',
     &        (proc_map(i,j),j=1,proc_map(i,0))
      enddo
      return
      end
c


      subroutine setup_event_attributes
c For FxFx or UNLOPS matching with pythia8, set the correct attributes
c for the <event> tag in the LHEF file. "npNLO" are the number of Born
c partons in this multiplicity when running the code at NLO accuracy
c ("npLO" is -1 in that case). When running LO only, invert "npLO" and
c "npNLO".
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      integer i
      integer                 nattr,npNLO,npLO
      common/event_attributes/nattr,npNLO,npLO
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      character*4      abrv
      common /to_abrv/ abrv
      if ((shower_mc.eq.'PYTHIA8' .or. shower_mc.eq.'HERWIGPP') .and.
     $     (ickkw.eq.3.or.ickkw.eq.4))then
         nattr=2
         nFKSprocess=1          ! just pick one
         call fks_inc_chooser()
         call leshouche_inc_chooser()
         npNLO=0
         npLO=-1
         do i=nincoming+1,nexternal
c     include all quarks (except top quark) and the gluon.
            if(abs(idup(i,1)).le.5 .or. abs(idup(i,1)).eq.21)
     &           npNLO=npNLO+1
         enddo
         npNLO=npNLO-1
         if (npNLO.gt.99) then
            write (*,*) 'Too many partons',npNLO
            stop
         endif
         if (abrv.eq.'born') then
            npLO=npNLO
            npNLO=-1
         endif
      else
         nattr=0
      endif
      return
      end


      subroutine update_vegas_x(xx,x)
      use mint_module
      implicit none
      integer i
      double precision xx(ndimmax),x(99),ran2
      external ran2
      integer         nndim
      common/tosigint/nndim
      character*4      abrv
      common /to_abrv/ abrv
      do i=1,99
         if (abrv.eq.'born') then
            if(i.le.nndim-3)then
               x(i)=xx(i)
            elseif(i.le.nndim) then
               x(i)=ran2()      ! Choose them flat when not including real-emision
            else
               x(i)=0.d0
            endif
         else
            if(i.le.nndim)then
               x(i)=xx(i)
            else
               x(i)=0.d0
            endif
         endif
      enddo
      return
      end



      subroutine get_born_nFKSprocess(nFKS_in,nFKS_out)
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      integer nFKS_in,nFKS_out,iFKS,iiFKS,nFKSprocessBorn(fks_configs)
      logical firsttime
      data firsttime /.true./
      save nFKSprocessBorn
c
      if (firsttime) then
         firsttime=.false.
         do iFKS=1,fks_integrated
            nFKSprocessBorn(iFKS)=0
            if ( need_color_links_D(iFKS) .or. 
     &           need_charge_links_D(iFKS) )then
               nFKSprocessBorn(iFKS)=iFKS
            endif
            if (nFKSprocessBorn(iFKS).eq.0) then
c     try to find the process that has the same j_fks but with i_fks a
c     gluon
               do iiFKS=1,fks_integrated
                  if ( (need_color_links_D(iiFKS) .or.
     &                  need_charge_links_D(iiFKS)) .and.
     &                 fks_j_D(iFKS).eq.fks_j_D(iiFKS) ) then
                     nFKSprocessBorn(iFKS)=iiFKS
                     exit
                  endif
               enddo
            endif
c     try to find the process that has the j_fks initial state if
c     current j_fks is initial state (and similar for final state j_fks)
            if (nFKSprocessBorn(iFKS).eq.0) then
               do iiFKS=1,fks_integrated
                  if ( need_color_links_D(iiFKS) .or.
     &                 need_charge_links_D(iiFKS) ) then
                     if ( fks_j_D(iiFKS).le.nincoming .and.
     &                    fks_j_D(iFKS).le.nincoming ) then
                        nFKSprocessBorn(iFKS)=iiFKS
                        exit
                     elseif ( fks_j_D(iiFKS).gt.nincoming .and.
     &                        fks_j_D(iFKS).gt.nincoming ) then
                        nFKSprocessBorn(iFKS)=iiFKS
                        exit
                     endif
                  endif
               enddo
            endif
c     If still not found, just pick any one that has a soft singularity
            if (nFKSprocessBorn(iFKS).eq.0) then
               do iiFKS=1,fks_integrated
                  if ( need_color_links_D(iiFKS) .or.
     &                 need_charge_links_D(iiFKS) ) then
                     nFKSprocessBorn(iFKS)=iiFKS
                  endif
               enddo
            endif
c     if there are no soft singularities at all, just do something trivial
            if (nFKSprocessBorn(iFKS).eq.0) then
               nFKSprocessBorn(iFKS)=iFKS
            endif
         enddo
         write (*,*) 'Total number of FKS directories is', fks_configs
         write (*,*) 'For the Born we use nFKSprocesses:'
         write (*,*)  nFKSprocessBorn
      endif
      if (nFKSprocessBorn(nFKS_in).eq.0) then
         write(*,*) 'Could not find the correct map to Born '/
     &        /'FKS configuration for the NLO FKS '/
     &        /'configuration', nFKS_in
         stop 1
      else
         nFKS_out=nFKSprocessBorn(nFKS_in)
      endif
      return
      end


      subroutine special_check_SoftSing(isoft)
      use weight_lines
      implicit none
      include 'nexternal.inc'
      include 'fks_info.inc'
      integer i_soft,isoft,i
      logical found_S
      i_soft=0
      found_S=.false.
      do i=1,icontr
         if (H_event(i)) then
            cycle
         else
            found_S=.true.
         endif
         if (abs(pdg_type_d(nFKS(i),fks_i_d(nFKS(i)))).eq.21) then
            i_soft=i
            exit
         endif
      enddo
      if (found_S .and. i_soft.eq.0) then
         ! add an artificial contribution
         call update_fks_dir(isoft)
         call add_wgt(2,1d-199,0d0,0d0)
      endif
      return
      end
