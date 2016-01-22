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
      include 'reweight.inc'
      INTEGER    ITMAX,   NCALL

      common/citmax/itmax,ncall
C
C     LOCAL
C
      integer i,j,l,l1,l2,ndim
      character*130 buf
c
c     Global
c
      integer                                      nsteps
      character*40          result_file,where_file
      common /sample_status/result_file,where_file,nsteps
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
      common/tosigint/ndim

      real*8 sigint
      external sigint

      integer irestart
      logical savegrid

      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file

      integer i_momcmp_count
      double precision xratmax
      common/ccheckcnt/i_momcmp_count,xratmax

      integer n_mp, n_disc
c For MINT:
      include "mint.inc"
      real* 8 xgrid(0:nintervals,ndimmax),ymax(nintervals,ndimmax)
     $     ,ymax_virt,ans(nintegrals),unc(nintegrals),chi2(nintegrals)
     $     ,x(ndimmax),itmax_fl
      integer ixi_i,iphi_i,iy_ij,vn
      integer ifold(ndimmax) 
      common /cifold/ifold
      integer ifold_energy,ifold_phi,ifold_yij
      common /cifoldnumbers/ifold_energy,ifold_phi,ifold_yij
      logical putonshell
      integer imode,dummy
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      integer nvirt(nintervals_virt,ndimmax),nvirt_acc(nintervals_virt
     $     ,ndimmax)
      double precision ave_virt(nintervals_virt,ndimmax)
     $     ,ave_virt_acc(nintervals_virt,ndimmax)
     $     ,ave_born_acc(nintervals_virt ,ndimmax)
      common/c_ave_virt/ave_virt,ave_virt_acc,ave_born_acc,nvirt
     $     ,nvirt_acc

      logical SHsep
      logical Hevents
      common/SHevents/Hevents
      character*10 dum
c statistics for MadLoop      
      integer ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1(0:9)
      common/ups_stats/ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1

      double precision virtual_over_born
      common/c_vob/virtual_over_born
      double precision average_virtual,virtual_fraction
      common/c_avg_virt/average_virtual,virtual_fraction

c timing statistics
      include "timing_variables.inc"
      real*4 tOther, tTot

c general MadFKS parameters
      include "FKSParams.inc"

c applgrid
      integer iappl
      common /for_applgrid/ iappl

      logical              fixed_order,nlo_ps
      common /c_fnlo_nlops/fixed_order,nlo_ps


C-----
C  BEGIN CODE
C-----  
c
c     Setup the timing variable
c
      call cpu_time(tBefore)
      fixed_order=.true.
      nlo_ps=.false.

c     Read general MadFKS parameters
c
      call FKSParamReader(paramFileName,.TRUE.,.FALSE.)
      average_virtual=0d0
      virtual_fraction=max(virt_fraction,min_virt_fraction)
      
      
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
      call get_user_params(ncall,itmax,iconfig,imode)
      if(imode.eq.0)then
        flat_grid=.true.
      else
        flat_grid=.false.
      endif
      ndim = 3*(nexternal-nincoming)-4
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
c APPLgrid
      if (imode.eq.0) iappl=0 ! overwrite when starting completely fresh
      if(iappl.ne.0) then
         write(6,*) "Initializing aMCfast ..."
c     Set flavor map, starting from all possible
c     parton lumi configurations defined in initial_states_map.dat
         call setup_flavourmap
c     Fill the number of combined matrix elements for given initial state luminosity
         call find_iproc_map
         write(6,*) "   ... done."
      endif
      i_momcmp_count=0
      xratmax=0.d0
      unwgt=.false.
      call addfil(dum)
      if (imode.eq.-1.or.imode.eq.0) then
         if(imode.eq.0)then
c Don't safe the reweight information when just setting up the grids.
            doreweight=.false.
            do j=0,nintervals
               do i=1,ndimmax
                  xgrid(j,i)=0.d0
               enddo
            enddo
         else
            doreweight=do_rwgt_scale.or.do_rwgt_pdf
c to restore grids:
            open (unit=12, file='mint_grids',status='old')
            do j=0,nintervals
               read (12,*) (xgrid(j,i),i=1,ndim)
            enddo
            do j=1,nintervals_virt
               read (12,*) (ave_virt(j,i),i=1,ndim)
            enddo
            read (12,*) ans(1),unc(1),dummy,dummy
            read (12,*) virtual_fraction,average_virtual
            close (12)
            write (*,*) "Update iterations and points to",itmax,ncall
         endif
c
         write (*,*) 'imode is ',imode

         if (ickkw.eq.-1) then
            min_virt_fraction=1d0
            virtual_fraction=1d0
         endif
c
c Setup for parton-level NLO reweighting
         if(do_rwgt_scale.or.do_rwgt_pdf) call setup_fill_rwgt_NLOplot()
         call mint(sigint,ndim,ncall,itmax,imode,xgrid,ymax,ymax_virt
     $        ,ans,unc,chi2)
         call topout
         write(*,*)'Final result [ABS]:',ans(1),' +/-',unc(1)
         write(*,*)'Final result:',ans(2),' +/-',unc(2)
         write(*,*)'chi**2 per D.o.F.:',chi2(1)
         open(unit=58,file='results.dat',status='unknown')
         write(58,*) ans(1),unc(2),0d0,0,0,0,0,0d0,0d0,ans(2)
         close(58)
c
c to save grids:
         open (unit=12, file='mint_grids',status='unknown')
         do j=0,nintervals
            write (12,*) (xgrid(j,i),i=1,ndim)
         enddo
         do j=1,nintervals_virt
            write (12,*) (ave_virt(j,i),i=1,ndim)
         enddo
         write (12,*) ans(1),unc(1),ncall,itmax
         write (12,*) virtual_fraction,average_virtual
         close (12)
      else
         write (*,*) 'Unknown imode',imode
         stop
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

      call cpu_time(tAfter)
      tTot = tAfter-tBefore
      tOther = tTot - (tBorn+tGenPS+tReal+tCount+tIS+tFxFx+tf_nb+tf_all
     &     +t_as+tr_s+tr_pdf+t_plot+t_cuts+t_MC_subt+t_isum+t_p_unw
     $     +t_write)
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
      write(*,*) 'Time spent in Other_tasks : ',tOther
      write(*,*) 'Time spent in Total : ',tTot

      open (unit=12, file='res.dat',status='unknown')
      write (12,*)ans(1),unc(1),ans(2),unc(2),itmax,ncall,tTot
      close(12)


      if(i_momcmp_count.ne.0)then
        write(*,*)'     '
        write(*,*)'WARNING: genps_fks code 555555'
        write(*,*)i_momcmp_count,xratmax
      endif

      end


      block data timing
c timing statistics
      include "timing_variables.inc"
      data tOLP/0.0/
      data tFastJet/0.0/
      data tPDF/0.0/
      data tDSigI/0.0/
      data tDSigR/0.0/
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
      end


      double precision function sigint(xx,vegas_wgt,ifl,f)
      implicit none
      include 'nexternal.inc'
      include 'mint.inc'
      include 'nFKSconfigs.inc'
      include 'c_weight.inc'
      include 'reweight.inc'
      include 'run.inc'
      double precision xx(ndimmax),vegas_wgt,f(nintegrals),jac,p(0:3
     $     ,nexternal),rwgt,vol,sig,x(99),MC_int_wgt
      integer ifl,nFKS_born,nFKS_picked,iFKS,nFKS_min
     $     ,nFKS_max,izero,ione,itwo,mohdr
      parameter (izero=0,ione=1,itwo=2,mohdr=-100)
      logical passcuts,passcuts_nbody,passcuts_n1body,sum
      external passcuts
      data sum /.false./
      integer         ndim
      common/tosigint/ndim
      logical       nbody
      common/cnbody/nbody
      integer           iconfig
      common/to_configs/iconfig
      double precision p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $     ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      double precision           virt_wgt_mint,born_wgt_mint
      common /virt_born_wgt_mint/virt_wgt_mint,born_wgt_mint
      double precision virtual_over_born
      common/c_vob/virtual_over_born
      logical                calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      character*4      abrv
      common /to_abrv/ abrv
      integer iappl
      common /for_applgrid/ iappl
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real
      if (ifl.ne.0) then
         write (*,*) 'ERROR ifl not equal to zero in sigint',ifl
         stop 1
      endif
      if (iappl.ne.0 .and. sum) then
         write (*,*) 'WARNING: applgrid only possible '/
     &        /'with MC over FKS directories',iappl,sum
         write (*,*) 'Switching to MC over FKS directories'
         sum=.false.
      endif
      sigint=0d0
      icontr=0
      virt_wgt_mint=0d0
      born_wgt_mint=0d0
      virtual_over_born=0d0
      wgt_me_born=0d0
      wgt_me_real=0d0
      if (ickkw.eq.-1) H1_factor_virt=0d0
      if (ickkw.eq.3) call set_FxFx_scale(0,p)
      call update_vegas_x(xx,x)
      call get_MC_integer(1,fks_configs,nFKS_picked,vol)

c The nbody contributions
      if (abrv.eq.'real') goto 11
      nbody=.true.
      calculatedBorn=.false.
      call get_born_nFKSprocess(nFKS_picked,nFKS_born)
      call update_fks_dir(nFKS_born,iconfig)
      jac=1d0
      call generate_momenta(ndim,iconfig,jac,x,p)
      if (p_born(0,1).lt.0d0) goto 12
      call compute_prefactors_nbody(vegas_wgt)
      call set_cms_stuff(izero)
      passcuts_nbody=passcuts(p1_cnt(0,1,0),rwgt)
      if (passcuts_nbody) then
         if (ickkw.eq.3) call set_FxFx_scale(1,p1_cnt(0,1,0))
         call set_alphaS(p1_cnt(0,1,0))
         if (abrv(1:2).ne.'vi') then
            call compute_born
         endif
         if (abrv.ne.'born') then
            call compute_nbody_noborn
         endif
      endif

 11   continue
c The n+1-body contributions (including counter terms)
      if (abrv.eq.'born'.or.abrv(1:2).eq.'vi') goto 12
      nbody=.false.
      if (sum) then
         nFKS_min=1
         nFKS_max=fks_configs
         MC_int_wgt=1d0
      else
         nFKS_min=nFKS_picked
         nFKS_max=nFKS_picked
         MC_int_wgt=1d0/vol
      endif
      do iFKS=nFKS_min,nFKS_max
         wgt_me_born=0d0
         wgt_me_real=0d0
         jac=MC_int_wgt
         call update_fks_dir(iFKS,iconfig)
         call generate_momenta(ndim,iconfig,jac,x,p)
         if (p_born(0,1).lt.0d0) cycle
         call compute_prefactors_n1body(vegas_wgt,jac)
         call set_cms_stuff(izero)
         passcuts_nbody =passcuts(p1_cnt(0,1,0),rwgt)
         call set_cms_stuff(mohdr)
         passcuts_n1body=passcuts(p,rwgt)
         if (passcuts_nbody .and. abrv.ne.'real') then
            call set_cms_stuff(izero)
            if (ickkw.eq.3) call set_FxFx_scale(2,p1_cnt(0,1,0))
            call set_alphaS(p1_cnt(0,1,0))
            call compute_soft_counter_term(0d0)
            call set_cms_stuff(ione)
            call compute_collinear_counter_term(0d0)
            call set_cms_stuff(itwo)
            call compute_soft_collinear_counter_term(0d0)
         endif
         if (passcuts_n1body) then
            call set_cms_stuff(mohdr)
            if (ickkw.eq.3) call set_FxFx_scale(3,p)
            call set_alphaS(p)
            call compute_real_emission(p,1d0)
         endif
      enddo
      
 12   continue
c Include PDFs and alpha_S and reweight to include the uncertainties
      if (ickkw.eq.-1) call include_veto_multiplier
      call include_PDF_and_alphas
      if (doreweight) then
         if (do_rwgt_scale .and. ickkw.ne.-1) call reweight_scale
         if (do_rwgt_scale .and. ickkw.eq.-1) call reweight_scale_NNLL
         if (do_rwgt_pdf) call reweight_pdf
      endif
      
      if (iappl.ne.0) then
         if (sum) then
            write (*,*) 'ERROR: applgrid only possible '/
     &           /'with MC over FKS directories',iappl,sum
            stop 1
         endif
         call fill_applgrid_weights(vegas_wgt)
      endif

c Importance sampling for FKS configurations
      if (sum) then
         call get_wgt_nbody(sig)
         call fill_MC_integer(1,nFKS_picked,abs(sig))
      else
         call get_wgt_no_nbody(sig)
         call fill_MC_integer(1,nFKS_picked,abs(sig)*vol)
      endif

c Finalize PS point
      call fill_plots
      call fill_mint_function(f)
      return
      end

      subroutine update_fks_dir(nFKS,iconfig)
      implicit none
      integer nFKS,iconfig
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      nFKSprocess=nFKS
      call fks_inc_chooser()
      call leshouche_inc_chooser()
      call setcuts
      call setfksfactor(iconfig,.false.)
      return
      end
      
      subroutine get_born_nFKSprocess(nFKS_in,nFKS_out)
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      integer nFKS_in,nFKS_out,iFKS,nFKSprocessBorn(2)
      logical firsttime,foundB(2)
      data firsttime /.true./
      save nFKSprocessBorn,foundB
      if (firsttime) then
         firsttime=.false.
         foundB(1)=.false.
         foundB(2)=.false.
         do iFKS=1,fks_configs
            if (particle_type_D(iFKS,fks_i_D(iFKS)).eq.8) then
               if (fks_j_D(iFKS).le.nincoming) then
                  foundB(1)=.true.
                  nFKSprocessBorn(1)=iFKS
               else
                  foundB(2)=.true.
                  nFKSprocessBorn(2)=iFKS
               endif
            endif
         enddo
         write (*,*) 'Total number of FKS directories is', fks_configs
         write (*,*) 'For the Born we use nFKSprocesses  #',
     $        nFKSprocessBorn
      endif
      if (fks_j_D(nFKS_in).le.nincoming) then
         if (.not.foundB(1)) then
            write(*,*) 'Trying to generate Born momenta with '/
     &           /'initial state j_fks, but there is no '/
     &           /'configuration with i_fks a gluon and j_fks '/
     &           /'initial state'
            stop 1
         endif
         nFKS_out=nFKSprocessBorn(1)
      else
         if (.not.foundB(2)) then
            write(*,*) 'Trying to generate Born momenta with '/
     &           /'final state j_fks, but there is no configuration'/
     &           /' with i_fks a gluon and j_fks final state'
            stop 1
         endif
         nFKS_out=nFKSprocessBorn(2)
      endif
      return
      end

      subroutine update_vegas_x(xx,x)
      implicit none
      include 'mint.inc'
      integer i
      double precision xx(ndimmax),x(99),ran2
      external ran2
      integer ndim
      common/tosigint/ndim
      character*4 abrv
      common /to_abrv/ abrv
      do i=1,99
         if (abrv.eq.'born'.or.abrv(1:2).eq.'vi') then
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
      return
      end

c
      subroutine get_user_params(ncall,itmax,iconfig,irestart)
c**********************************************************************
c     Routine to get user specified parameters for run
c**********************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'run.inc'
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

      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      logical fillh
      integer mc_hel,ihel
      double precision volh
      common/mc_int2/volh,mc_hel,ihel,fillh

      logical done
      character*100 buffer
c-----
c  Begin Code
c-----
      mint=.true.
      unwgt=.false.
      open (unit=83,file='input_app.txt',status='old')
      done=.false.
      do while (.not. done)
         read(83,'(a)',err=222,end=222) buffer
         if (buffer(1:7).eq.'NPOINTS') then
            buffer=buffer(10:100)
            read(buffer,*) ncall
            write (*,*) 'Number of phase-space points per iteration:',ncall
         elseif(buffer(1:11).eq.'NITERATIONS') then
            read(buffer(14:),*) itmax
            write (*,*) 'Maximum number of iterations is:',itmax
         elseif(buffer(1:8).eq.'ACCURACY') then
            read(buffer(11:),*) accuracy
            write (*,*) 'Desired accuracy is:',accuracy
         elseif(buffer(1:10).eq.'ADAPT_GRID') then
            read(buffer(13:),*) use_cut
            write (*,*) 'Using adaptive grids:',use_cut
         elseif(buffer(1:12).eq.'MULTICHANNEL') then
            read(buffer(15:),*) i
            if (i.eq.1) then
               multi_channel=.true.
               write (*,*) 'Using Multi-channel integration'
            else
               multi_channel=.false.
               write (*,*) 'Not using Multi-channel integration'
            endif
         elseif(buffer(1:12).eq.'SUM_HELICITY') then
            read(buffer(15:),*) i
            if (nincoming.eq.1) then
               write (*,*) 'Sum over helicities in the virtuals'/
     $              /' for decay process'
               mc_hel=0
            elseif (i.eq.0) then
               mc_hel=0
               write (*,*) 'Explicitly summing over helicities'/
     $              /' for the virtuals'
            else
               mc_hel=1
               write(*,*) 'Do MC over helicities for the virtuals'
            endif
            isum_hel=0
         elseif(buffer(1:7).eq.'CHANNEL') then
            read(buffer(10:),*) dconfig
            iconfig = int(dconfig)
            do i=1,mapconfig(0)
               if (iconfig.eq.mapconfig(i)) then
                  iconfig=i
                  exit
               endif
            enddo
            write(*,12) 'Running Configuration Number: ',iconfig
         elseif(buffer(1:5).eq.'SPLIT') then
            read(buffer(8:),*) i
            write (*,*) 'Splitting channel:',i
         elseif(buffer(1:8).eq.'RUN_MODE') then
            read(buffer(11:),*) abrvinput
            if(abrvinput(5:5).eq.'0')then
               nbody=.true.
            else
               nbody=.false.
            endif
            abrv=abrvinput(1:4)
            write (*,*) "doing the ",abrv," of this channel"
            if(nbody)then
               write (*,*) "integration Born/virtual with Sfunction=1"
            else
               write (*,*) "Normal integration (Sfunction != 1)"
            endif
         elseif(buffer(1:7).eq.'RESTART') then
            read(buffer(10:),*) irestart
            if (irestart.eq.0) then
               write (*,*) 'RESTART: Fresh run'
            elseif(irestart.eq.-1) then
               write (*,*) 'RESTART: Use old grids, but refil plots'
            elseif(irestart.eq.1) then
               write (*,*) 'RESTART: continue with existing run'
            else
               write (*,*) 'RESTART:',irestart
            endif
         endif
         cycle
 222     done=.true.
      enddo
      close(83)

      if (fks_configs.eq.1) then
         if (pdg_type_d(1,fks_i_d(1)).eq.-21) then
            write (*,*) 'Process generated with [LOonly=QCD]. '/
     $           /'Setting abrv to "born".'
            abrv='born'
            if (ickkw.eq.3) then
               write (*,*) 'FxFx merging not possible with'/
     $              /' [LOonly=QCD] processes'
               stop 1
            endif
         endif
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
c
