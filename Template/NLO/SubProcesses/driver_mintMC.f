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
      integer ncall_virt,ncall_novi
      character*4 abrv
      common /to_abrv/ abrv
C
C     LOCAL
C
      integer i,j,l,l1,l2,ndim,nevts

      integer lunlhe
      parameter (lunlhe=98)
c
c     Global
c
cc
      include 'run.inc'
      include 'coupl.inc'
      
      integer           iconfig
      common/to_configs/iconfig

c Vegas stuff
      common/tosigint/ndim

      real*8 sigintF
      external sigintF

      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file

      integer i_momcmp_count
      double precision xratmax
      common/ccheckcnt/i_momcmp_count,xratmax

      double precision virtual_over_born
      common/c_vob/virtual_over_born
      double precision average_virtual,virtual_fraction
      common/c_avg_virt/average_virtual,virtual_fraction

      double precision weight
c For MINT:
      include "mint.inc"
      real* 8 xgrid(0:nintervals,ndimmax),ymax(nintervals,ndimmax)
     $     ,ymax_virt,ans(nintegrals),unc(nintegrals),chi2(nintegrals)
     $     ,x(ndimmax)
      integer ixi_i,iphi_i,iy_ij,vn
      integer ifold(ndimmax) 
      common /cifold/ifold
      integer ifold_energy,ifold_phi,ifold_yij
      common /cifoldnumbers/ifold_energy,ifold_phi,ifold_yij
      logical putonshell
      logical only_virt
      integer imode
      common /c_imode/imode,only_virt
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
      double precision ran2
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
      include "FKSParams.inc"
      logical              fixed_order,nlo_ps
      common /c_fnlo_nlops/fixed_order,nlo_ps

C-----
C  BEGIN CODE
C-----  
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
      average_virtual=0d0
      virtual_fraction=virt_fraction

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
      call initcluster
c     
c     Get user input
c
      write(*,*) "getting user params"
      call get_user_params(ncall,itmax,iconfig,imode,
     &     ixi_i,iphi_i,iy_ij,SHsep)
c Only do the reweighting when actually generating the events
      if (imode.eq.2) then
         doreweight=do_rwgt_scale.or.do_rwgt_pdf
      else
         doreweight=.false.
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
      write(*,*) "about to integrate ", ndim,ncall,itmax,iconfig
      i_momcmp_count=0
      xratmax=0.d0
      unwgt=.false.

c*************************************************************
c     setting of the grids
c*************************************************************
      if (imode.eq.-1.or.imode.eq.0) then
         if(imode.eq.0)then
c initialize grids
            do j=0,nintervals
               do i=1,ndimmax
                  xgrid(j,i)=0.d0
               enddo
            enddo
         else
c to restore grids:
            open (unit=12, file='mint_grids',status='old')
            do j=0,nintervals
               read (12,*) (xgrid(j,i),i=1,ndim)
            enddo
            do j=1,nintervals_virt
               read (12,*) (ave_virt(j,i),i=1,ndim)
            enddo
            read (12,*) (ans(i),i=1,nintegrals)
            read (12,*) ifold_energy,ifold_phi,ifold_yij
            read (12,*) virtual_fraction,average_virtual
            close (12)
         endif
c
         write (*,*) 'imode is ',imode
         call mint(sigintF,ndim,ncall,itmax,imode,xgrid,ymax,ymax_virt
     $        ,ans,unc,chi2)
         open(unit=58,file='res_0',status='unknown')
         write(58,*)'Final result [ABS]:',ans(1),' +/-',unc(1)
         write(58,*)'Final result:',ans(2),' +/-',unc(2)
         close(58)
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
         write (12,*) (ans(i),i=1,nintegrals)
         write (12,*) ifold_energy,ifold_phi,ifold_yij
         write (12,*) virtual_fraction,average_virtual
         close (12)

c*************************************************************
c     computation of upper bounding envelope
c*************************************************************
      elseif(imode.eq.1) then
c to restore grids:
         open (unit=12, file='mint_grids',status='old')
         do j=0,nintervals
            read (12,*) (xgrid(j,i),i=1,ndim)
         enddo
         do j=1,nintervals_virt
            read (12,*) (ave_virt(j,i),i=1,ndim)
         enddo
         read (12,*) (ans(i),i=1,nintegrals)
         read (12,*) ifold_energy,ifold_phi,ifold_yij
         read (12,*) virtual_fraction,average_virtual
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
         call mint(sigintF,ndim,ncall,itmax,imode,xgrid,ymax,ymax_virt
     $        ,ans,unc,chi2)
         
c If integrating the virtuals alone, we include the virtuals in
c ans(1). Therefore, no need to have them in ans(5) and we have to set
c them to zero.
         if (only_virt) then
            ans(3)=0d0 ! virtual Xsec
            ans(5)=0d0 ! ABS virtual Xsec
         endif

         open(unit=58,file='res_1',status='unknown')
         write(58,*)'Final result [ABS]:',ans(1)+ans(5),' +/-'
     $        ,sqrt(unc(1)**2+unc(5)**2)
         write(58,*)'Final result:',ans(2),' +/-',unc(2)
         close(58)
         write(*,*)'Final result [ABS]:',ans(1)+ans(5),' +/-'
     $        ,sqrt(unc(1)**2+unc(5)**2)
         write(*,*)'Final result:',ans(2),' +/-',unc(2)
         write(*,*)'chi**2 per D.o.F.:',chi2(1)
c write the results.dat file 
         open(unit=58,file='results.dat',status='unknown')
         write(58,*)ans(1)+ans(5), unc(2), 0d0, 0, 0, 0, 0, 0d0 ,0d0, ans(2) 
         close(58)

c to save grids:
         open (unit=12, file='mint_grids',status='unknown')
         write (12,*) (xgrid(0,i),i=1,ndim)
         do j=1,nintervals
            write (12,*) (xgrid(j,i),i=1,ndim)
            write (12,*) (ymax(j,i),i=1,ndim)
         enddo
         do j=1,nintervals_virt
            write (12,*) (ave_virt(j,i),i=1,ndim)
         enddo
         write (12,*) ymax_virt
         write (12,*) (ifold(i),i=1,ndim)
         write (12,*) (ans(i),i=1,nintegrals)
         write (12,*) (unc(i),i=1,nintegrals)
         write (12,*) virtual_fraction,average_virtual
         close (12)

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
         read (99,*) nevts
         close(99)
         write(*,*) 'Generating ', nevts, ' events'
         if(nevts.eq.0) then
            write (*,*)
     &           'No events needed for this channel...skipping it'
            stop
         endif
         ncall=nevts ! Update ncall with the number found in 'nevts'

c to restore grids:
         open (unit=12, file='mint_grids',status='unknown')
         read (12,*) (xgrid(0,i),i=1,ndim)
         do j=1,nintervals
            read (12,*) (xgrid(j,i),i=1,ndim)
            read (12,*) (ymax(j,i),i=1,ndim)
         enddo
         do j=1,nintervals_virt
            read (12,*) (ave_virt(j,i),i=1,ndim)
         enddo
         read (12,*) ymax_virt
         read (12,*) (ifold(i),i=1,ndim)
         read (12,*) (ans(i),i=1,nintegrals)
         read (12,*) (unc(i),i=1,nintegrals)
         read (12,*) virtual_fraction,average_virtual
         close (12)

c determine how many events for the virtual and how many for the no-virt
         ncall_virt=int(ans(5)/(ans(1)+ans(5)) * ncall)
         ncall_novi=ncall-ncall_virt

         write (*,*) "Generating virt :: novi approx.",ncall_virt
     $        ,ncall_novi

         open(unit=lunlhe,file='events.lhe',status='unknown')

c fill the information for the write_header_init common block
         ifile=lunlhe
         ievents=ncall
         inter=ans(2)
         absint=ans(1)+ans(5)
         uncer=unc(2)

         weight=(ans(1)+ans(5))/ncall

         if (abrv(1:3).ne.'all' .and. abrv(1:4).ne.'born' .and.
     $        abrv(1:4).ne.'virt') then
            write (*,*) 'CANNOT GENERATE EVENTS FOR ABRV',abrv
            stop 1
         endif

         write (*,*) 'imode is ',imode
         vn=-1
         call gen(sigintF,ndim,xgrid,ymax,ymax_virt,0,x,vn)
         do j=1,ncall
            if (abrv(1:4).eq.'born') then
               vn=3
               call gen(sigintF,ndim,xgrid,ymax,ymax_virt,1,x,vn)
            else
               if (ran2().lt.ans(5)/(ans(1)+ans(5)) .or. only_virt) then
                  abrv='virt'
                  if (only_virt) then
                     vn=2
                     call gen(sigintF,ndim,xgrid,ymax,ymax_virt,1,x,vn)
                  else
                     vn=1
                     call gen(sigintF,ndim,xgrid,ymax,ymax_virt,1,x,vn)
                  endif
               else
                  abrv='novi'
                  vn=2
                  call gen(sigintF,ndim,xgrid,ymax,ymax_virt,1,x,vn)
               endif
            endif
c Randomly pick the contribution that will be written in the event file
            call pick_unweight_contr(iFKS_picked)
            call update_fks_dir(iFKS_picked,iconfig)
            call fill_rwgt_lines
            call finalize_event(x,weight,lunlhe,putonshell)
         enddo
         vn=-1
         call gen(sigintF,ndim,xgrid,ymax,ymax_virt,3,x,vn)
         write (*,*) 'Generation efficiencies:',x(1),x(4)
c Uncomment the next to lines to print the integral from the PS points
c trown during event generation. This corresponds only to the cross
c section if these points are thrown flat, so not using the xmmm() stuff
c in mint.
c         write (*,*) 'Integral from novi points computed',x(2),x(3)
c         write (*,*) 'Integral from virt points computed',x(5),x(6)
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

      call cpu_time(tAfter)
      tTot = tAfter-tBefore
      tOther = tTot - (tBorn+tGenPS+tReal+tCount+tIS+tFxFx+tf_nb+tf_all
     $     +t_as+tr_s+tr_pdf+t_plot+t_cuts+t_MC_subt+t_isum+t_p_unw
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
      if (imode.eq.0) then
         write (12,*)ans(1),unc(1),ans(2),unc(2),itmax,ncall,tTot
      else
         write (12,*)ans(1)+ans(5),sqrt(unc(1)**2+unc(5)**2),ans(2)
     $        ,unc(2),itmax,ncall,tTot
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
      include 'mint.inc'
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

c-----
c  Begin Code
c-----
      mint=.true.
      usexinteg=.false.
      write(*,'(a)') 'Enter number of events and iterations: '
      read(*,*) ncall,itmax
      write(*,*) 'Number of events and iterations ',ncall,itmax

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






      function sigintF(xx,vegas_wgt,ifl,f)
c From dsample_fks
      implicit none
      include 'mint.inc'
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'c_weight.inc'
      include 'run.inc'
      logical firsttime,passcuts,passcuts_nbody,passcuts_n1body
      integer i,ifl,proc_map(0:fks_configs,0:fks_configs)
     $     ,nFKS_picked_nbody,nFKS_in,nFKS_out,izero,ione,itwo,mohdr
     $     ,iFKS,sum
      double precision xx(ndimmax),vegas_wgt,f(nintegrals),jac,p(0:3
     $     ,nexternal),rwgt,vol,sig,x(99),MC_int_wgt,vol1,probne,gfactsf
     $     ,gfactcl,replace_MC_subt,sudakov_damp,sigintF,n1body_wgt
      external passcuts
      parameter (izero=0,ione=1,itwo=2,mohdr=-100)
      data firsttime/.true./
      integer           iconfig
      common/to_configs/iconfig
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      integer     fold
      common /cfl/fold
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      logical              MCcntcalled
      common/c_MCcntcalled/MCcntcalled
      double precision           virt_wgt_mint,born_wgt_mint
      common /virt_born_wgt_mint/virt_wgt_mint,born_wgt_mint
      double precision virtual_over_born
      common /c_vob/   virtual_over_born
      logical       nbody
      common/cnbody/nbody
      integer         ndim
      common/tosigint/ndim
      character*4      abrv
      common /to_abrv/ abrv
      double precision p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $     ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      logical               only_virt
      integer         imode
      common /c_imode/imode,only_virt
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real
      sigintF=0d0
c Find the nFKSprocess for which we compute the Born-like contributions
      if (firsttime) then
         firsttime=.false.
c Determines the proc_map that sets which FKS configuration can be
c summed explicitly and which by MC-ing.
         call setup_proc_map(sum,proc_map)
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

      fold=ifl
      if (ifl.eq.0) then
         icontr=0
         virt_wgt_mint=0d0
         born_wgt_mint=0d0
         virtual_over_born=0d0
         MCcntcalled=.false.
         wgt_me_real=0d0
         wgt_me_born=0d0
         if (ickkw.eq.3) call set_FxFx_scale(0,p)
         call update_vegas_x(xx,x)
         call get_MC_integer(1,proc_map(0,0),proc_map(0,1),vol1)

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
         call update_fks_dir(nFKS_picked_nbody,iconfig)
         jac=1d0
         call generate_momenta(ndim,iconfig,jac,x,p)
         if (p_born(0,1).lt.0d0) goto 12
         call compute_prefactors_nbody(vegas_wgt)
         call set_cms_stuff(izero)
         call set_shower_scale_noshape(p,nFKS_picked_nbody*2-1)
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
c Update the shower starting scale. This might be updated again below if
c the nFKSprocess is the same.
         call include_shape_in_shower_scale(p,nFKS_picked_nbody)
            
         
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
            call update_fks_dir(iFKS,iconfig)
            jac=1d0/vol1
            probne=1d0
            gfactsf=1.d0
            gfactcl=1.d0
            MCcntcalled=.false.
            call generate_momenta(ndim,iconfig,jac,x,p)
c Every contribution has to have a viable set of Born momenta (even if
c counter-event momenta do not exist).
            if (p_born(0,1).lt.0d0) cycle
c check if event or counter-event passes cuts
            call set_cms_stuff(izero)
            passcuts_nbody=passcuts(p1_cnt(0,1,0),rwgt)
            call set_cms_stuff(mohdr)
            passcuts_n1body=passcuts(p,rwgt)
            if (.not. (passcuts_nbody.or.passcuts_n1body)) cycle
c Set the shower scales            
            call set_cms_stuff(izero)
            call set_shower_scale_noshape(p,iFKS*2-1)
            call set_cms_stuff(mohdr)
            call set_shower_scale_noshape(p,iFKS*2)
c Compute the n1-body prefactors
            call compute_prefactors_n1body(vegas_wgt,jac)
c Include the FxFx Sudakov terms in the prefactors:
c   CP : counter-event kinematics passes cuts
c   EP : event kinematics passes cuts
c   CE : counter-event kinematics exists
c   EE : event kinematics exists
c   CC : compute FxFx for counter-events kinematics
c   EC : compute FxFx for event kinematics
c
c     CP  EP  CE  EE | CC  EC
c     X   X   X   X  | X   X
c     X       X   X  | X   X
c         X   X   X  |     X
c     X       X      | X   X
c         X       X  |     X
c
            if (ickkw.eq.3) then
               call set_FxFx_scale(0,p)
               if (passcuts_nbody .and. abrv.ne.'real') then
                  call set_cms_stuff(izero)
                  call set_FxFx_scale(2,p1_cnt(0,1,0))
               endif
               if (p(0,1).gt.0d0) then
                  call set_cms_stuff(mohdr)
                  call set_FxFx_scale(3,p)
               endif
            endif               
            if (passcuts_nbody .and. abrv.ne.'real') then
c Include the MonteCarlo subtraction terms
               if (ickkw.ne.4) then
                  call set_cms_stuff(mohdr)
                  if (ickkw.eq.3) call set_FxFx_scale(-3,p)
                  call set_alphaS(p)
                  call compute_MC_subt_term(p,gfactsf,gfactcl,probne)
               else
c For UNLOPS all real-emission contributions need to be added to the
c S-events. Do this by setting probne to 0. For UNLOPS, no MC counter
c events are called, so this will remain 0.
                  probne=0d0
               endif
c Include the FKS counter terms. When close to the soft or collinear
c limits, the MC subtraction terms should be replaced by the FKS
c ones. This is set via the gfactsf, gfactcl and probne functions (set
c by the call to compute_MC_subt_term) through the 'replace_MC_subt'.
               call set_cms_stuff(izero)
               if (ickkw.eq.3) call set_FxFx_scale(-2,p1_cnt(0,1,0))
               call set_alphaS(p1_cnt(0,1,0))
               replace_MC_subt=(1d0-gfactsf)*probne
               call compute_soft_counter_term(replace_MC_subt)
               call set_cms_stuff(ione)
               replace_MC_subt=(1d0-gfactcl)*(1d0-gfactsf)*probne
               call compute_collinear_counter_term(replace_MC_subt)
               call set_cms_stuff(itwo)
               replace_MC_subt=(1d0-gfactcl)*(1d0-gfactsf)*probne
               call compute_soft_collinear_counter_term(replace_MC_subt)
            endif
c Include the real-emission contribution.
            if (passcuts_n1body) then
               call set_cms_stuff(mohdr)
               if (ickkw.eq.3) call set_FxFx_scale(-3,p)
               call set_alphaS(p)
               sudakov_damp=probne
               call compute_real_emission(p,sudakov_damp)
            endif
c Update the shower starting scale with the shape from the MC
c subtraction terms.
            call include_shape_in_shower_scale(p,iFKS)
         enddo
 12      continue

c Include PDFs and alpha_S and reweight to include the uncertainties
         call include_PDF_and_alphas
c Sum the contributions that can be summed before taking the ABS value
         call sum_identical_contributions
c Update the shower starting scale for the S-events after we have
c determined which contributions are identical.
         call update_shower_scale_Sevents
         call fill_mint_function_NLOPS(f,n1body_wgt)
         call fill_MC_integer(1,proc_map(0,1),n1body_wgt*vol1)
      elseif(ifl.eq.1) then
         write (*,*) 'Folding not implemented'
         stop 1
      elseif(ifl.eq.2) then
         call fill_mint_function_NLOPS(f,n1body_wgt)
      endif
      return
      end


      subroutine setup_proc_map(sum,proc_map)
c Determines the proc_map that sets which FKS configuration can be
c summed explicitly and which by MC-ing.
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'reweight_all.inc'
      include 'nFKSconfigs.inc'
      double precision lum,dlum
      external dlum
      logical found_ini1,found_ini2,found_fnl
      integer proc_map(0:fks_configs,0:fks_configs)
     $     ,j_fks_proc(fks_configs),i_fks_pdg_proc(fks_configs)
     $     ,j_fks_pdg_proc(fks_configs),i,sum,j
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
      sum=3
      if (ickkw.eq.4) then
         sum=0
         write (*,*)'Using ickkw=4, include only 1 FKS dir per'/
     $        /' Born PS point (sum=0)'
      endif
      maxproc_save=0
      do nFKSprocess=1,fks_configs
         call fks_inc_chooser()
c Set Bjorken x's to some random value before calling the dlum() function
         xbk(1)=0.5d0
         xbk(2)=0.5d0
         lum=dlum()  ! updates IPROC
         maxproc_save=max(maxproc_save,IPROC)
         if (doreweight) then
            call reweight_settozero()
            call reweight_settozero_all(nFKSprocess*2,.true.)
            call reweight_settozero_all(nFKSprocess*2-1,.true.)
         endif
      enddo
      write (*,*) 'Total number of FKS directories is', fks_configs
c For sum over identical FKS pairs, need to find the identical structures
      if (sum.eq.3) then
c MC over FKS pairs that have soft singularity
         proc_map(0,0)=0
         do i=1,fks_configs
            proc_map(i,0)=0
            i_fks_pdg_proc(i)=0
            j_fks_pdg_proc(i)=0
            j_fks_proc(i)=0
         enddo
c First find all the nFKSprocesses that have a soft singularity and put
c them in the process map
         do nFKSprocess=1,fks_configs
            call fks_inc_chooser()
            if (abs(PDG_type(i_fks)).eq.21) then
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
            if (abs(i_fks_pdg_proc(i)).eq.21 .and. j_fks_proc(i).eq.1
     $           .and. .not.found_ini1) then
               found_ini1=.true.
            elseif (abs(i_fks_pdg_proc(i)).eq.21 .and.
     $              j_fks_proc(i).eq.1.and. found_ini1) then
               write (*,*)'Initial state 1 g->gg already'/
     $              /' found in driver_mintMC'
               write (*,*) i_fks_pdg_proc
               write (*,*) j_fks_pdg_proc
               write (*,*) j_fks_proc
               stop
            elseif (abs(i_fks_pdg_proc(i)).eq.21 .and.
     $              j_fks_proc(i).eq.2.and. .not.found_ini2) then
               found_ini2=.true.
            elseif (abs(i_fks_pdg_proc(i)).eq.21 .and.
     $              j_fks_proc(i).eq.2.and. found_ini2) then
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
         do nFKSprocess=1,fks_configs
            call fks_inc_chooser()
            if (abs(PDG_type(i_fks)).ne.21) then
               if (j_fks.eq.1 .and. found_ini1) then
                  do i=1,proc_map(0,0)
                     if (abs(i_fks_pdg_proc(i)).eq.21 .and.
     $                    j_fks_proc(i).eq.1) then
                        proc_map(i,0)=proc_map(i,0)+1
                        proc_map(i,proc_map(i,0))=nFKSprocess
                        exit
                     endif
                  enddo
               elseif (j_fks.eq.2 .and. found_ini2) then
                  do i=1,proc_map(0,0)
                     if (abs(i_fks_pdg_proc(i)).eq.21 .and.
     $                    j_fks_proc(i).eq.2) then
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
         proc_map(0,0)=fks_configs
         do i=1,fks_configs
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
      integer    maxflow
      parameter (maxflow=999)
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


      subroutine update_fks_dir(nFKS,iconfig)
      implicit none
      integer nFKS,iconfig
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      nFKSprocess=nFKS
      call fks_inc_chooser()
      call leshouche_inc_chooser()
      call setcuts
      call setfksfactor(iconfig,.true.)
      return
      end

      subroutine update_vegas_x(xx,x)
      implicit none
      include 'mint.inc'
      integer i
      double precision xx(ndimmax),x(99),ran2
      external ran2
      integer         ndim
      common/tosigint/ndim
      character*4 abrv
      common /to_abrv/ abrv
      do i=1,99
         if (abrv.eq.'born') then
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
