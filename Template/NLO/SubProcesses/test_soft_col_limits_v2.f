      program test_soft_col_limits
      implicit none
      integer ilim,nsofttests,ncolltests,fks_loop_min
     $     ,fks_loop_max,fks_loop,bs_min,bs_max
      double precision xi_i_fks_fix_input,y_ij_fks_fix_input
      
      call read_input_file(ilim,nsofttests,ncolltests,fks_loop_min
     $     ,fks_loop_max,xi_i_fks_fix_input,y_ij_fks_fix_input)

      call init_test_limits(ilim)
      
      do fks_loop=fks_loop_min,fks_loop_max
         call init_new_loop(fks_loop,bs_min,bs_max)
         do iconfig=bs_min,bs_max
            call init_iconfig_loop(ilim)

            call test_soft_limit(ilim,iconfig,nsofttests,xi_i_fks_fix_input
     $           ,y_ij_fks_fix_input)
            
         enddo
      enddo
      end

      subroutine test_soft_limits(ilim,iconfig,nsofttests,xi_i_fks_fix_input
     $     ,y_ij_fks_fix_input)
      implicit none
      imax=10
      Hevents=.true.
      softtest=.true.
      colltest=.false.
      nerr(:)=0
      imax=10
      do j=1,nsofttests
         xi_i_fks_fix=xi_i_fks_fix_input
         y_ij_fks_fix=y_ij_fks_fix_input
         call generate_valid_momenta(ndim,iconfig,wgt,x,p)
         do i=1,imax
            if (softtest) xi_i_fks_fix=0.1d0**i
            if (colltest) y_ij_fks_fix=1-0.1d0**i
            call compute_towards_limit(ilim,iconfig,x
     $           ,towards_amp_split(1,i),towards_wgt_PS(i),towards_p(0,1
     $           ,i))
         enddo
         
         call compute_in_the_limit(ilim,xi_i_fks_fix_input
     $     ,y_ij_fks_fix_input)

      enddo      
      end

      subroutine compute_towards_limit(ilim,iconfig,x,amp,wgt_PS,xp)
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      integer ilim,iamp,iconfig
      double precision amp(amp_split_size),wgt_PS,wgt,x(99),p(0:3
     $     ,nexternal),xp(0:3,nexternal+1)
      logical                calculatedBorn
      common/ccalculatedBorn/calculatedBorn

      wgt=1d0
      call generate_momenta(ndim,iconfig,wgt,x,p)
      if (ilim.eq.2) then
         calculatedBorn=.false.
         call set_cms_stuff(-100)
         call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
      else
         write (*,*) 'to implement'
      endif

      ! save amplitudes (and PS weight) towards limit
      do iamp=1,amp_split_size
         if (ilim.eq.2) then
            amp(iamp) = amp_split(iamp)*wgt
         else
            amp(iamp) = amp_split_mc(iamp)*wgt
         endif
      enddo
      wgt_PS = wgt

      ! save momenta
      xp(0:3,1:nexternal)=p(0:3,1:nexternal)
      xp(0:3,nexternal+1)=p_i_fks_ev(0:3)
      
      end
      
      
      
      subroutine compute_in_the_limit(ilim,xi_i_fks_fix_input
     $     ,y_ij_fks_fix_input)
      implicit none
      integer ilim
      double precision xi_i_fks_fix_input,y_ij_fks_fix_input
      xi_i_fks_fix=xi_i_fks_fix_input
      y_ij_fks_fix=y_ij_fks_fix_input
      wgt=1d0
      call generate_momenta(ndim,iconfig,wgt,x,p)
      if (ilim.eq.2) then
         calculatedBorn=.false.
         if (softtest) then
            call set_cms_stuff(0)
            call sreal(p1_cnt(0,1,0),zero,y_ij_fks_ev,fx)
         elseif(colltest) then
            call set_cms_stuff(1)
            call sreal(p1_cnt(0,1,1),xi_i_fks_ev,one,fx)
         endif
      else
         write (*,*) 'to implement'
      endif
      
      ! save amplitudes (and PS weight) in the limit
      do iamp=1,amp_split_size
         if (ilim.eq.2) then
            limit_split(iamp) = amp_split(iamp)*wgt
         else
            limit_split(iamp) = amp_split_mc(iamp)*wgt
         endif
         limit_PS_split(iamp) = wgt
      enddo

! save momenta
      if (softtest) then
         lxp(0:3,1:nexternal)=p1_cnt(0:3,1:nexternal,0)
         lxp(0:3,nexternal+1)=p_i_fks_cnt(0:3,0)
      elseif (colltest) then
         lxp(0:3,1:nexternal)=p1_cnt(0:3,1:nexternal,1)
         lxp(0:3,nexternal+1)=p_i_fks_cnt(0:3,1)
      endif
      
      end



      subroutine generate_valid_momenta(ndim,iconfig,wgt,x,p)
      use mint_module
      implicit none
      include 'nexternal.inc'
      integer ndim,iconfig
      double precision wgt,x(99),p(0:3,nexternal)
      integer jj,ntry
      double precision ran2
      external ran2
      logical                calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      do jj=1,ndim
         x(jj)=ran2()
      enddo
      new_point=.true.
      wgt=1d0
      call generate_momenta(ndim,iconfig,wgt,x,p)
      calculatedBorn=.false.
      do while (( wgt.lt.0 .or. p(0,1).le.0d0 .or. p_born(0,1).le.0d0
     &     ) .and. ntry .lt. 1000)
         do jj=1,ndim
            x(jj)=ran2()
         enddo
         new_point=.true.
         wgt=1d0
         call generate_momenta(ndim,iconfig,wgt,x,p)
         calculatedBorn=.false.
         ntry=ntry+1
      enddo
      if (ntry.ge.1000) then
         write (*,*) 'No points passed cuts...'
         write (12,*) 'ERROR: no points passed cuts...'/
     $        /' Cannot perform ME tests properly for config',iconfig
         cycle
      endif
      end
      
      subroutine init_iconfig_loop(ilim)
      use mint_module
      implicit none
      include 'nexternal.inc'
      integer ilim
      double precision x(99),wgt,p(0:3,nexternal)
      double complex wgt1(2)
      logical        softtest,colltest
      common/sctests/softtest,colltest
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      ichan=1
      iconfigs(1)=iconfig
      if (ilim.eq.2) then
         call setfksfactor(.false.)
      else
         call setfksfactor(.true.)
      endif
      call setcuts

      softtest=.false.
      colltest=.false.

      call generate_valid_momenta(ndim,iconfig,wgt,x,p)
      
      call sborn(p_born,wgt1)
      write (*,*) ''
      write (*,*) ''
      write (*,*) ''
      end

      subroutine init_new_loop(fks_loop,bs_min,bs_max)
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      integer iconfig_in,bs_min,bs_max
      integer         nndim
      common/tosigint/nndim
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      
      nFKSprocess=fks_loop
      call fks_inc_chooser()
      call leshouche_inc_chooser()
      write (*,*) ''
      write (*,*) '================================================='
      write (*,*) ''
      write (*,*) 'NEW FKS CONFIGURATION:'
      write (*,*) 'FKS configuration number is ',nFKSprocess
      write (*,*) 'FKS partons are: i=',i_fks,'  j=',j_fks
      write (*,*) 'with PDGs:       i=',PDG_type(i_fks),'  j='
     $     ,PDG_type(j_fks)
c
      ndim = 3*(nexternal-nincoming)-4
      if (abs(lpp(1)).ge.1) ndim=ndim+1
      if (abs(lpp(2)).ge.1) ndim=ndim+1
      nndim=ndim

      call set_ebeam()

!     update shat
      call init_process_module_global(shower_mc,'all ',nexternal
     $     ,nincoming,mcatnlo_delta,ebeam(1)+ebeam(2),max_bcol,maxflow
     $     ,ickkw)
      
      write(*,*)'  '
      write(*,*)'  '
      write(*,*)"Enter graph number (iconfig), "
     &     //"'0' loops over all graphs, '-1' takes the first non-zero"
      read(*,*) iconfig_in
      
      if (iconfig_in.eq.0) then
         bs_min=1
         bs_max=mapconfig(0)
      elseif (iconfig_in.eq.-1) then
         bs_min=1
         bs_max=1
      else
         bs_min=iconfig_in
         bs_max=iconfig_in
      endif
      
      end

      subroutine set_ebeam()
      implicit none
      include 'nexternal.inc'
      include 'cuts.inc'
      include 'run.inc'
      integer i,k
      double precision totmass,pmass(nexternal)
      LOGICAL IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
      call setcuts              !Sets up cuts 
c When doing hadron-hadron collision reduce the effect collision energy.
c Note that tests are always performed at fixed energy with Bjorken x=1.
      totmass = 0.0d0
      include 'pmass.inc'       ! make sure to set the masses after the model has been included
      do i=nincoming+1,nexternal-1
         if (is_a_j(i)) then
            totmass = totmass + max(ptj,pmass(i))
         elseif ((is_a_lp(i).or.is_a_lm(i))) then
            totmass = totmass + max(mll/2d0,mll_sf/2d0,ptl,pmass(i))
         elseif (is_a_ph(i)) then
            totmass = totmass + ptgmin
         else
            if (any(mxxmin(i,i+1:nexternal-1).gt.0d0)) then
               do k=i+1,nexternal-1
                  if (mxxmin(i,k).gt.0d0) then
                     totmass = totmass + mxxmin(i,k)
                  endif
               enddo
            elseif (etmin(i).gt.0d0) then
               totmass=totmass+max(etmin(i),pmass(i))
            else
               totmass = totmass + pmass(i)
           endif
         endif
      enddo
      if (lpp(1).ne.0) ebeam(1)=max(ebeam(1)/20d0,totmass)
      if (lpp(2).ne.0) ebeam(2)=max(ebeam(2)/20d0,totmass)
      end

      

      subroutine init_test_limits(ilim)
      use mint_module
      use process_module
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'run.inc'
      include 'born_nhel.inc'
      include 'genps.inc'
      integer ilim
c-----
      if (fks_configs.eq.1) then
         if (pdg_type_d(1,fks_i_d(1)).eq.-21) then
            write (*,*) 'Process generated with [LOonly=QCD]. '/
     $           /'No tests to do.'
            return
         endif
      endif
      if (ilim.eq.2) then
         nlo_ps=.false.
         fixed_order=.true.
      else
         nlo_ps=.true.
         fixed_order=.false.
      endif
      call setrun               !Sets up run parameters
      call setpara('param_card.dat') !Sets up couplings and masses
      call fill_configurations_common
      ! initialise the global, but process dependent, information in the process module.
      call init_process_module_global(shower_mc,'all ',nexternal
     $     ,nincoming,mcatnlo_delta,ebeam(1)+ebeam(2),max_bcol,maxflow
     $     ,ickkw)
      ! Also put all the n-body process dependent stuff here. It does
      ! not depend on PS point or FKS config, so all global information.
      call init_process_module_nbody_wrapper()
      call init_scale_module(nexternal,shower_scale_factor,fks_configs
     $     ,1)
      end


      subroutine read_input_file(ilim,nsofttests,ncolltests,fks_loop_min
     $     ,fks_loop_max,xi_i_fks_fix_input,y_ij_fks_fix_input)
      implicit none
      integer ilim,nsofttests,ncolltests,fks_conf_number,fks_loop_min
     $     ,fks_loop_max
c$$$      character*10           MonteCarlo
c$$$      common/cMonteCarloType/MonteCarlo
      double precision alsf,besf
      common /cgfunsfp/alsf,besf
      double precision alazi,beazi
      common /cgfunazi/alazi,beazi
      double precision xi_i_fks_fix_input,y_ij_fks_fix_input
      write(*,*) 'Enter 0 to compute MC/MC(limit)'
      write(*,*) '      1 to compute MC/ME(limit)'
      write(*,*) '      2 to compute ME/ME(limit)'
      read (*,*) ilim
      if (ilim.ne.0 .and. ilim.ne.1 .and. ilim.ne.2) then
         write (*,*) 'ERROR: not a valid choice'
         stop 1
      endif
      if (ilim.eq.0 .or. ilim.eq.1) then
c$$$         write(*,*) 'Enter the Monte Carlo name: possible choices are'
c$$$         write(*,*) 'HERWIG6, HERWIGPP, PYTHIA6Q, PYTHIA6PT, PYTHIA8'
c$$$         read (*,*) MonteCarlo
c$$$         if ( MonteCarlo(1:7).ne.'HERWIG6'.and.
c$$$     &        MonteCarlo(1:8).ne.'HERWIGPP'.and.
c$$$     $        MonteCarlo(1:8).ne.'PYTHIA6Q'.and.
c$$$     &        MonteCarlo(1:9).ne.'PYTHIA6PT'.and.
c$$$     $        MonteCarlo(1:7).ne.'PYTHIA8' )then
c$$$            write(*,*)'Wrong name ',MonteCarlo,' during the tests'
c$$$            stop 1
c$$$         endif
         write(*,*) 'Enter alpha, beta for G_soft'
         write(*,*) '  Enter alpha<0 to set G_soft=1 (no ME soft)'
         read (*,*) alsf,besf
         write(*,*) 'Enter alpha, beta for G_azi'
         write(*,*) '  Enter alpha>0 to set G_azi=0 (no azi corr)'
         read (*,*) alazi,beazi
      endif
      write(*,*) 'Enter xi_i, y_ij to be used in coll/soft tests'
      write(*,*) ' Enter -2 to generate them randomly'
      read (*,*) xi_i_fks_fix_input,y_ij_fks_fix_input

      write(*,*) 'Enter number of tests for soft and collinear limits'
      read (*,*) nsofttests,ncolltests
      
      write (*,*) 'Give FKS configuration number ("0" loops over all)'
      read (*,*) fks_conf_number

      if (fks_conf_number.eq.0) then
         fks_loop_min=1
         fks_loop_max=fks_configs
      else
         fks_loop_min=fks_conf_number
         fks_loop_max=fks_conf_number
      endif
      
      end
     
