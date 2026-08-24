      program test_soft_col_limits
      use mint_module
      implicit none
      integer ilim,nsofttests,ncolltests,fks_loop_min
     $     ,fks_loop_max,fks_loop,bs_min,bs_max,nstep,ntests
      double precision xi_i_fks_fix_input,y_ij_fks_fix_input,mass_jfks
      logical        softtest,colltest
      common/sctests/softtest,colltest
      
      call read_input_file(ilim,nsofttests,ncolltests,fks_loop_min
     $     ,fks_loop_max,xi_i_fks_fix_input,y_ij_fks_fix_input)

      call init_test_limits(ilim,nstep)
      
      do fks_loop=fks_loop_min,fks_loop_max
         call init_new_loop(fks_loop,bs_min,bs_max,mass_jfks)
         do iconfig=bs_min,bs_max
            call init_iconfig_loop(ilim)
            softtest=.true.
            colltest=.false.
            ntests=nsofttests
            call test_limits(ilim,ntests,xi_i_fks_fix_input
     $           ,y_ij_fks_fix_input,nstep,mass_jfks)
            if (mass_jfks.gt.0d0) then
               write (*,*) 'No collinear test for massive j_fks'
               cycle
            endif
            softtest=.false.
            colltest=.true.
            ntests=ncolltests
            call test_limits(ilim,ntests,xi_i_fks_fix_input
     $           ,y_ij_fks_fix_input,nstep,mass_jfks)
         enddo
      enddo
      end

      subroutine test_limits(ilim,ntests,xi_i_fks_fix_input
     $     ,y_ij_fks_fix_input,nstep,mass_jfks)
      use mint_module
      use scale_module
      use kinematics_module
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      integer nstep,ntests,i,ilim,jtest,partner_picked
      double precision xi_i_fks_fix_input,y_ij_fks_fix_input,wgt,xx(99)
     $     ,p(0:3,nexternal),towards_amp_split(1:amp_split_size,1:nstep)
     $     ,towards_wgt_PS(1:nstep),towards_p(0:3,nexternal+1,1:nstep)
     $     ,limit_amp_split(1:amp_split_size),limit_wgt_PS,limit_p(0:3
     $     ,nexternal+1),born_flow_factor,mass_jfks
      double complex wgt1(2)
      double precision xi_i_fks_fix,y_ij_fks_fix
      common /cxiyfix/ xi_i_fks_fix,y_ij_fks_fix
      logical        softtest,colltest
      common/sctests/softtest,colltest
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      do jtest=1,ntests
         if (colltest) then
            xi_i_fks_fix=xi_i_fks_fix_input
            y_ij_fks_fix=0.9d0
         elseif (softtest) then
            xi_i_fks_fix=0.1d0
            y_ij_fks_fix=y_ij_fks_fix_input
         endif
         call generate_valid_momenta(wgt,xx,p)

         if (ilim.eq.1) then
            call sborn(p_born,wgt1)
            call fill_father_and_ileg(i_fks,j_fks,mass_jfks)
            call get_born_flow(born_flow_picked,born_flow_factor)
            call determine_partner(born_flow_picked,partner_picked)
            call init_process_module_n1body_wrapper(born_flow_picked)
            call compute_shower_scale_nbody(p_born,born_flow_picked)
         endif
         
         do i=1,nstep
            if (softtest) xi_i_fks_fix=0.1d0**i
            if (colltest) y_ij_fks_fix=1-0.1d0**i
            call compute_towards_limit(ilim,xx,born_flow_factor
     $           ,towards_amp_split(1,i),towards_wgt_PS(i),towards_p(0,1
     $           ,i))
         enddo
         ! reset xi and y to input values
         if (colltest) then
            xi_i_fks_fix=xi_i_fks_fix_input
            y_ij_fks_fix=1.0d0
         elseif (softtest) then
            xi_i_fks_fix=0.0d0
            y_ij_fks_fix=y_ij_fks_fix_input
         endif
         call compute_in_the_limit(ilim,xx,born_flow_factor
     $        ,limit_amp_split,limit_wgt_PS,limit_p(0,1))
         call check_limit_and_print_result(nstep,towards_amp_split
     $        ,towards_wgt_PS,towards_p,limit_amp_split,limit_wgt_PS
     $        ,limit_p,ntests,jtest)
      enddo
      if (ntests.gt.10) then
         call print_summary(iconfig,ntests)
      endif
      end

      subroutine print_summary(iconfig,ntests)
      implicit none
      include 'orders.inc'
      double precision max_fail
      parameter       (max_fail=0.3d0)
      integer iamp,i,orders(nsplitorders),ntests ,iconfig
      double precision fail_frac
      character*9 print_string
      logical        softtest,colltest
      common/sctests/softtest,colltest
      integer nerr(0:amp_split_size)
      common /c_nerr/nerr
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      if (softtest) then
         print_string='     Soft'
      elseif (colltest) then
         print_string='Collinear'
      endif
      write(*,*)print_string//' tests done for (Born) config',iconfig
      write(*,*)'Failures:',nerr
      do iamp = 0, amp_split_size
         if (iamp.gt.0.and.iamp.le.amp_split_size_born) cycle
         fail_frac= nerr(iamp)/dble(ntests)
         if (iamp.ne.0) then
            write(*,fmt="(a,i3,a)",advance="no")'Split-order',iamp,': '
            call amp_split_pos_to_orders(iamp,orders)
            do i = 1, nsplitorders
               write(*,fmt="(a,a,i3,a)",advance="no") ordernames(i), ':'
     $              ,orders(i),'; '
            enddo
         else
            write(*,fmt="(a)", advance="no")'Sum of all orders: '
         endif
         if (fail_frac.lt.max_fail) then
            write(*,401) print_string,nFKSprocess,fail_frac
         else
            write(*,402) print_string,nFKSprocess,fail_frac
         endif
      enddo
 401  format(a9,'test ',i2,' PASSED. Fraction of failures: ', f4.2) 
 402  format(a9,'test ',i2,' FAILED. Fraction of failures: ', f4.2) 
      end

      subroutine check_limit_and_print_result(nstep,towards_amp_split
     $        ,towards_wgt_PS,towards_p,limit_amp_split,limit_wgt_PS
     $        ,limit_p,ntests,jtest)
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      integer nstep,ntests,jtest
      double precision towards_amp_split(1:amp_split_size,1:nstep)
     $     ,towards_wgt_PS(1:nstep),towards_p(0:3,nexternal+1,1:nstep)
     $     ,limit_amp_split(1:amp_split_size),limit_wgt_PS,limit_p(0:3
     $     ,nexternal+1)
      if (ntests.le.10) then
         call check_and_print_to_screen(nstep,towards_amp_split
     $        ,towards_wgt_PS,towards_p,limit_amp_split,limit_wgt_PS
     $        ,limit_p,jtest)
      else
         call check_limits(nstep,towards_amp_split
     $        ,towards_wgt_PS,towards_p,limit_amp_split,limit_wgt_PS
     $        ,limit_p,jtest)
      endif
      end

      subroutine check_limits(nstep,towards_amp_split
     $        ,towards_wgt_PS,towards_p,limit_amp_split,limit_wgt_PS
     $        ,limit_p,jtest)
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      integer nstep,jtest,iamp,iflag,iret
      double precision towards_amp_split(1:amp_split_size,1:nstep)
     $     ,towards_wgt_PS(1:nstep),towards_p(0:3,nexternal+1,1:nstep)
     $     ,limit_amp_split(1:amp_split_size),limit_wgt_PS,limit_p(0:3
     $     ,nexternal+1),amp(1:nstep),limit
      logical        softtest,colltest
      common/sctests/softtest,colltest
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer nerr(0:amp_split_size)
      common /c_nerr/nerr
      if (softtest) then
         iflag=0
      endif
      if (colltest) then
         iflag=1
      endif
      
! check limit summed over orders
      amp(1:nstep)=sum(towards_amp_split(1:amp_split_size,1:nstep),dim
     $     =1)
      limit=sum(limit_amp_split(1:amp_split_size),dim=1)

      call checkres(amp(1:nstep) ,limit,towards_wgt_PS(1:nstep)
     $     ,limit_wgt_PS,towards_p,limit_p,iflag,nstep,jtest,nexternal
     $     ,i_fks,j_fks,iret)
      nerr(0)=nerr(0)+iret
! check limit order-by-order
      do iamp=1, amp_split_size
         if (towards_amp_split(iamp ,1).ne.0d0 .or.
     $        limit_amp_split(iamp).ne.0d0) then
            call checkres(towards_amp_split(iamp,1:nstep)
     $           ,limit_amp_split(iamp),towards_wgt_PS(1:nstep)
     $           ,limit_wgt_PS,towards_p,limit_p,iflag,nstep,jtest
     $           ,nexternal,i_fks,j_fks,iret)
            nerr(iamp)=nerr(iamp)+iret
         endif
      enddo
      
      end

      subroutine check_and_print_to_screen(nstep,towards_amp_split
     $        ,towards_wgt_PS,towards_p,limit_amp_split,limit_wgt_PS
     $        ,limit_p,jtest)
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      integer nstep,iret,jtest,iflag,iamp,i,k,l,orders(nsplitorders)
      double precision towards_amp_split(1:amp_split_size,1:nstep)
     $     ,towards_wgt_PS(1:nstep),towards_p(0:3,nexternal+1,1:nstep)
     $     ,limit_amp_split(1:amp_split_size),limit_wgt_PS,limit_p(0:3
     $     ,nexternal+1),amp(1:nstep),limit
      logical        softtest,colltest
      common/sctests/softtest,colltest
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      if (softtest) then
         write (*,*) 'Soft limit:'
         iflag=0
      endif
      if (colltest) then
         write (*,*) 'Collinear limit:'
         iflag=1
      endif
! print amplitudes summed over orders:
      amp(1:nstep)=sum(towards_amp_split(1:amp_split_size,1:nstep),dim
     $     =1)
      limit=sum(limit_amp_split(1:amp_split_size),dim=1)
      do i=1,nstep
         call xprintout(6,amp(i),limit)
      enddo
! print amplitude order-by-order, and check that they approach limit correctly.
      do iamp=1, amp_split_size
         if (limit_amp_split(iamp).ne.0d0 .or. towards_amp_split(iamp
     $        ,1).ne.0d0) then
            write(*,*) '   Split-order', iamp
            call amp_split_pos_to_orders(iamp,orders)
            do i=1,nsplitorders
               write(*,*) '      ',ordernames(i), ':',orders(i)
            enddo
            do i=1,nstep
               call xprintout(6,towards_amp_split(iamp,i)
     $              ,limit_amp_split(iamp))
            enddo
            call checkres(towards_amp_split(iamp,1:nstep)
     $           ,limit_amp_split(iamp),towards_wgt_PS(1:nstep)
     $           ,limit_wgt_PS ,towards_p,limit_p,iflag,nstep,jtest
     $           ,nexternal,i_fks,j_fks,iret)
            write(*,*) 'RETURN CODE', iret
         endif
      enddo
c dump momenta in a fort.80 file
      write(80,*)'  '
      write(80,*)'****************************'
      write(80,*)'  '
      do k=1,nexternal+1
         write(80,*)''
         write(80,*)'part:',k
         do l=0,3
            write(80,*)'comp:',l
            do i=1,nstep
               call xprintout(80,towards_p(l,k,i),limit_p(l,k))
            enddo
         enddo
      enddo
      end
      
      subroutine compute_towards_limit(ilim,x,born_flow_factor,amp
     $     ,wgt_PS,xp)
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      integer ilim,iamp,idum,nFKSprocess_save,iFKS
      double precision wgt,x(99),born_flow_factor,p(0:3,nexternal),fx
     $     ,amp(amp_split_size),wgt_PS,xp(0:3,nexternal+1),p_lab(0:3
     $     ,nexternal) ,p_cms(0:3,nexternal)
      logical                calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      double precision xbjrk_ev(2),xbjrk_cnt(2,-2:2)
      common/cbjorkenx/xbjrk_ev,xbjrk_cnt
      logical soft_limit_is_zero
      common /c_soft_limit_is_zero/soft_limit_is_zero
      logical        softtest,colltest
      common/sctests/softtest,colltest

      integer i
      wgt=1d0
      call generate_momenta(ndim,iconfig,wgt,x,p,p_lab,p_cms)
      
      calculatedBorn=.false.
      call set_cms_stuff(-100)
      if (ilim.eq.2) then
         call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
      elseif (ilim.eq.1) then
         amp=0d0
         call fks_inc_chooser()
         call update_coltype_and_charge(nFKSprocess,i_fks,j_fks)
         call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
         do iamp=1,amp_split_size
            amp(iamp) = amp(iamp)+amp_split(iamp)*born_flow_factor
         enddo
         call fks_inc_chooser()
         call update_coltype_and_charge(nFKSprocess,i_fks,j_fks)
         call compute_MC_subt_term_test(p,p_cms,p_lab,wgt
     $        ,born_flow_factor)
         do iamp=1,amp_split_size
            if (.not.(soft_limit_is_zero .and. softtest)) then
               if (amp_split(iamp).ne.0d0) then
                  amp(iamp) = amp(iamp)/amp_split(iamp)
               else
                  amp(iamp) = 1d0
               endif
            else
               amp(iamp)=amp_split(iamp)
            endif
         enddo
      else
         write (*,*) 'to implement (MC/MC)'
      endif

      wgt_PS = wgt

      ! save amplitudes (and PS weight) towards limit
      if (ilim.eq.2) then
         do iamp=1,amp_split_size
            amp(iamp) = amp_split(iamp)*wgt
         enddo
      endif
      ! save momenta
      xp(0:3,1:nexternal)=p(0:3,1:nexternal)
      xp(0:3,nexternal+1)=p_i_fks_ev(0:3)
      
      end
      
      
      
      subroutine compute_in_the_limit(ilim,x,born_flow_factor
     $     ,limit_split,limit_PS_wgt,lxp)
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision zero,    one
      parameter       (zero=0d0,one=1d0)
      integer ilim,iamp,idum
      double precision wgt,x(99),p(0:3,nexternal),fx,born_flow_factor
     $     ,limit_split(amp_split_size),limit_PS_wgt,lxp(0:3,nexternal
     $     +1),p_lab(0:3,nexternal) ,p_cms(0:3,nexternal)
      logical                calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      logical        softtest,colltest
      common/sctests/softtest,colltest
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision   xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      logical soft_limit_is_zero
      common /c_soft_limit_is_zero/soft_limit_is_zero
      wgt=1d0
      call generate_momenta(ndim,iconfig,wgt,x,p,p_lab,p_cms)
      
      calculatedBorn=.false.
      if (softtest) then
         call set_cms_stuff(0)
         if (ilim.eq.2) then
            call sreal(p1_cnt(0,1,0),zero,y_ij_fks_ev,fx)
         endif
      elseif(colltest) then
         call set_cms_stuff(1)
         if (ilim.eq.2) then
            call sreal(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,fx)
         endif
      endif
      
      if (softtest) then
         limit_PS_wgt = jac_cnt(0)
      elseif (colltest) then
         limit_PS_wgt = jac_cnt(1)
      endif
      
! save amplitudes (and PS weight) in the limit
      if (ilim.eq.2) then
         do iamp=1,amp_split_size
            limit_split(iamp) = amp_split(iamp)*limit_PS_wgt
         enddo
      elseif (ilim.eq.1) then
         do iamp=1,amp_split_size
            if (.not.(soft_limit_is_zero .and. softtest)) then
               limit_split(iamp) = 1d0
            else
               limit_split(iamp) = 0d0
            endif
         enddo
      endif
      
! save momenta
      if (softtest) then
         lxp(0:3,1:nexternal)=p1_cnt(0:3,1:nexternal,0)
         lxp(0:3,nexternal+1)=p_i_fks_cnt(0:3,0)
      elseif (colltest) then
         lxp(0:3,1:nexternal)=p1_cnt(0:3,1:nexternal,1)
         lxp(0:3,nexternal+1)=p_i_fks_cnt(0:3,1)
      endif
      
      end



      subroutine compute_MC_subt_term_test(p,p_cms,p_lab,wgt
     $     ,born_flow_factor)
      use mint_module
      use kinematics_module
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      logical include_gfun
      integer iFKS,k_fks,l_fks,n_connect,iconnect,iamp,nFKSprocess_save
     $     ,ii,jj,i
      double precision p(0:3,nexternal),xi,y,z(2),born_flow_factor
     $     ,amp_split_gfunc(amp_split_size),dummy
     $     ,amp_split_xmcxsec(amp_split_size,2),p_cms(0:3,nexternal)
     $     ,p_lab(0:3,nexternal) ,xx(99),wgt,jac,mass,p_cms_flipped(0:3
     $     ,nexternal),p_lab_flipped(0:3,nexternal),p_flipped(0:3
     $     ,nexternal)
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision amp_split_mc(1:amp_split_size)
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      logical                calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      ! use local amp_split_mc, since, compute_MCsubtraction_kl will overwrite amp_split:
      amp_split_mc(1:amp_split_size)=0d0
      include_gfun=.true. ! to set gfactsf. 
      xi=get_xi_from_p(i_fks,j_fks,p_cms)
      y=get_yij_from_p(i_fks,j_fks,p_cms)
      call compute_MCsubtraction_kl(i_fks,j_fks,xi,y,p
     $     ,p_cms,p_born,include_gfun,z,n_connect
     $     ,amp_split_xmcxsec)
      
      ! include_gfun will be false here if in dead zone.
      do iconnect=1,n_connect
         amp_split_mc(1:amp_split_size) =
     $        amp_split_mc(1:amp_split_size) +
     $        amp_split_xmcxsec(1:amp_split_size,iconnect)
      enddo
      amp_split_gfunc=0d0
      if (include_gfun) then
         call compute_MCsubtraction_from_gfun_test(xi,y,amp_split_gfunc)
         amp_split_mc(1:amp_split_size) = amp_split_mc(1:amp_split_size)
     $        + amp_split_gfunc(1:amp_split_size) * born_flow_factor
      endif
      include_gfun=.false.

      
      nFKSprocess_save=nFKSprocess

      
      do iFKS=1,fks_configs
         nFKSprocess=iFKS
         ! only include the ones compatible with the real-emission process
         if (any(pdg_type_d(iFKS,:).ne.pdg_type_d(nFKSprocess_save,:)))
     $        cycle
         ! This sets i_fks and j_fks to correspond to the ones in
         ! nFKSprocess (which here is iFKS).
         call fks_inc_chooser()
         call update_coltype_and_charge(nFKSprocess,i_fks,j_fks)
         
!     1. include do-loop over identical particless for i-fks and j-fks
!     2. flip all momenta (p, p_lab and p_cms) among the possible i-fks and j-fks
!     3. do NOT update i-fks and j-fks.
         do ii=3,nexternal
            if (pdg_type_d(nFKSprocess_save,ii).ne.
     &           pdg_type_d(nFKSprocess_save,i_fks)) cycle
            do jj=1,nexternal
               if (ii.eq.jj) cycle
               if (j_fks.le.nincoming .and. j_fks.ne.jj) cycle
               if (jj.le.nincoming .and. j_fks.ne.jj) cycle
               if (pdg_type_d(nFKSprocess_save,jj).ne.
     &              pdg_type_d(nFKSprocess_save,j_fks)) cycle
               if (pdg_type_d(nFKSprocess_save,ii).eq.
     $              pdg_type_d(nFKSprocess_save,jj) .and.
     $              ii.lt.jj) cycle
               if ( nFKSprocess.eq.nFKSprocess_save .and. 
     &              ii.eq.i_fks .and. jj.eq.j_fks) cycle ! this is already included above

               call flip_momenta(i_fks,ii,j_fks,jj,p,p_flipped)
               call flip_momenta(i_fks,ii,j_fks,jj,p_cms,p_cms_flipped)
               call flip_momenta(i_fks,ii,j_fks,jj,p_lab,p_lab_flipped)
               
!     compute kinematic variables
               xi=get_xi_from_p(i_fks,j_fks,p_cms_flipped)
               y=get_yij_from_p(i_fks,j_fks,p_cms_flipped)
               
! call the inverse phase-space. This will update the Born
! momenta, and the corresponding phase-space jacobian for the
! n+1-body. Note: if the random numbers are not generated flat
! (they are flat here), also the jacobian from importance
! sampling should be included.
               jac=1d0
!     inputs are: ndim,iconfig,p
!     outputs are: xx,jac (also updates pborn common block)
               call generate_lab_momenta_inverse(ndim,iconfig,jac,xx
     $              ,p_lab_flipped)
               if (jac.le.0d0) cycle
               CalculatedBorn=.false.
               ! include_gfun must be .false., because we do not want to
               ! update gfactsf
               call compute_MCsubtraction_kl(i_fks,j_fks,xi,y,p_flipped
     $              ,p_cms_flipped,p_born,include_gfun,z,n_connect
     $              ,amp_split_xmcxsec)
               do iconnect=1,n_connect
                  amp_split_mc(1:amp_split_size) =
     $                 amp_split_mc(1:amp_split_size) +
     $                 amp_split_xmcxsec(1:amp_split_size,iconnect) *
     $                 jac/wgt
               enddo
            enddo
         enddo
      enddo
      nFKSprocess=nFKSprocess_save
      call fks_inc_chooser()
      call update_coltype_and_charge(nFKSprocess,i_fks,j_fks)
      xi=get_xi_from_p(i_fks,j_fks,p_cms) ! these correspond to ij, not kl
      y=get_yij_from_p(i_fks,j_fks,p_cms)
      amp_split=amp_split_mc*xi**2*(1d0-y) ! re-remove the 1/xi^2 and 1/(1-y) factors; they depend on 'ij', not 'kl'
      end

      subroutine compute_MCsubtraction_from_gfun_test(xi,y,amp_split_gfunc)
      use kinematics_module
      implicit none
      include "nexternal.inc"
      include 'orders.inc'
      double precision zero,one
      parameter (zero=0d0,one=1d0)
      integer izero,ione,itwo
      parameter (izero=0,ione=1,itwo=2)
      double precision xi,y
      integer iFKS
      double precision amp_split_gfunc(amp_split_size)
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision dum,amp_split_s(amp_split_size),
     $     amp_split_c(amp_split_size),amp_split_sc(amp_split_size)
      amp_split_gfunc(1:amp_split_size) = 0d0
      call set_cms_stuff(izero)
      call sreal(p1_cnt(0,1,0),zero,y,dum)
      amp_split_s(1:amp_split_size) = amp_split(1:amp_split_size)
      call set_cms_stuff(ione)
      call sreal(p1_cnt(0,1,1),xi,one,dum)
      amp_split_c(1:amp_split_size) = amp_split(1:amp_split_size)
      call set_cms_stuff(itwo)
      call sreal(p1_cnt(0,1,2),zero,one,dum)
      amp_split_sc(1:amp_split_size) = amp_split(1:amp_split_size)
      amp_split_gfunc(1:amp_split_size) = (1d0-gfactsf)
     $     *( amp_split_s(1:amp_split_size) + (1d0-gfactcl)
     $      *(amp_split_c(1:amp_split_size)
     $        -amp_split_sc(1:amp_split_size)) )
     $     /(xi**2*(1d0-y)) ! re-instate 1/xi^2 and 1/(1-y); they should
                            ! not depend on 'kl', but rather on 'ij'
      return
      end

      subroutine generate_valid_momenta(wgt,x,p)
      use mint_module
      implicit none
      include 'nexternal.inc'
      double precision wgt,x(99),p(0:3,nexternal),p_lab(0:3,nexternal)
     $     ,p_cms(0:3,nexternal)
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
      call generate_momenta(ndim,iconfig,wgt,x,p,p_lab,p_cms)
      calculatedBorn=.false.
      do while (( wgt.lt.0 .or. p(0,1).le.0d0 .or. p_born(0,1).le.0d0
     &     ) .and. ntry .lt. 1000)
         do jj=1,ndim
            x(jj)=ran2()
         enddo
         new_point=.true.
         wgt=1d0
         call generate_momenta(ndim,iconfig,wgt,x,p,p_lab,p_cms)
         calculatedBorn=.false.
         ntry=ntry+1
      enddo
      if (ntry.ge.1000) then
         write (*,*) 'No valid phase-space points...'
         write (12,*) 'ERROR: no valid phase-space points...'/
     $        /' Cannot perform ME tests properly for config',iconfig
         stop 1
      endif
      end
      
      subroutine init_iconfig_loop(ilim)
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      integer ilim
      double precision x(99),wgt,p(0:3,nexternal)
      double complex wgt1(2)
      logical        softtest,colltest
      common/sctests/softtest,colltest
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      integer nerr(0:amp_split_size)
      common /c_nerr/nerr
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

      call generate_valid_momenta(wgt,x,p)
      
      call sborn(p_born,wgt1)
      write (*,*) ''
      write (*,*) ''
      write (*,*) ''
      nerr(0:amp_split_size)=0
      end

      subroutine init_new_loop(fks_loop,bs_min,bs_max,mass_jfks)
      use process_module
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      include 'born_nhel.inc'
      include 'born_maxamps.inc'
      include 'born_conf.inc'
      include 'coupl.inc'
      include 'leshouche_decl.inc'
      include 'fks_info.inc'
      double precision ZERO,    one
      parameter       (ZERO=0d0,one=1d0)
      integer iconfig_in,bs_min,bs_max,fks_loop,ifks
      double precision mass_jfks
      double precision pmass(nexternal)
      integer         nndim
      common/tosigint/nndim
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      logical soft_limit_is_zero
      common /c_soft_limit_is_zero/soft_limit_is_zero
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
      
      include 'pmass.inc' ! this is filled by setcuts (which is in set_ebeam())
      mass_jfks=pmass(j_fks)

!     update shat
      call init_process_module_global(shower_mc,'all ',nexternal
     $     ,nincoming,mcatnlo_delta,ebeam(1)+ebeam(2),max_bcol
     $     ,maxflow_used,ickkw)
      
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

      ! Check if soft-limit diverges
      if (need_color_links_D(nFKSprocess) .or.
     $     need_charge_links_D(nFKSprocess)) then
         soft_limit_is_zero=.False.
      else
         soft_limit_is_zero=.True.
      endif
      end

      subroutine set_ebeam()
      implicit none
      include 'nexternal.inc'
      include 'cuts.inc'
      include 'run.inc'
      include 'coupl.inc'
      double precision ZERO,    one
      parameter       (ZERO=0d0,one=1d0)
      integer i,k
      double precision totmass,pmass(nexternal)
      LOGICAL IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
      double precision etmin(nincoming+1:nexternal-1)
      double precision etmax(nincoming+1:nexternal-1)
      double precision mxxmin(nincoming+1:nexternal-1,nincoming+1:nexternal-1)
      common /to_cuts/etmin,etmax, mxxmin
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
      if (lpp(1).ne.0) ebeam(1)=max(ebeam(1)/20d0,totmass*2d0)
      if (lpp(2).ne.0) ebeam(2)=max(ebeam(2)/20d0,totmass*2d0)
      end

      

      subroutine init_test_limits(ilim,nstep)
      use mint_module
      use process_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'run.inc'
      include 'born_nhel.inc'
      include 'genps.inc'
      integer ilim,nstep
      logical         Hevents
      common/SHevents/Hevents
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
      Hevents=.true.
      nstep=10  ! take 10 steps towards the limit
      end


      subroutine read_input_file(ilim,nsofttests,ncolltests,fks_loop_min
     $     ,fks_loop_max,xi_i_fks_fix_input,y_ij_fks_fix_input)
      implicit none
      include 'nFKSconfigs.inc'
      integer ilim,nsofttests,ncolltests,fks_conf_number,fks_loop_min
     $     ,fks_loop_max
c$$$      character*10           MonteCarlo
c$$$      common/cMonteCarloType/MonteCarlo
      double precision alsf,besf
      common /cgfunsfp/alsf,besf
      double precision alazi,beazi
      common /cgfunazi/alazi,beazi
      double precision xi_i_fks_fix_input,y_ij_fks_fix_input
      write(*,*) 'Enter 0 to compute MC/MC(limit) (no longer available)'
      write(*,*) '      1 to compute MC/ME(limit)'
      write(*,*) '      2 to compute ME/ME(limit)'
      read (*,*) ilim
      if (ilim.ne.0 .and. ilim.ne.1 .and. ilim.ne.2) then
         write (*,*) 'ERROR: not a valid choice'
         stop 1
      endif
      if (ilim.eq.0) then
         write (*,*) 'ERROR: MC/MC(limit) is no longer available. '//
     &        '(This test was kind of irrelevant anyway)'
         stop 1
      endif
      if (ilim.eq.1) then
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
      ! TODO : remove this option
      write(*,*) 'Enter xi_i, y_ij to be used in coll/soft tests'
      write(*,*) ' Enter -2 to generate them randomly'
      read (*,*) xi_i_fks_fix_input,y_ij_fks_fix_input

      if (xi_i_fks_fix_input.ne.-2d0 .or. y_ij_fks_fix_input.ne.-2d0)
     $     then
         write (*,*) 'Cannot use fixed inputs'
         stop 1
      endif
      

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
      integer idup(nexternal,maxproc)
      integer mothup(2,nexternal,maxproc)
      integer icolup(2,nexternal,max_bcol)
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

      call fill_icolor_H(bornflow,jpart,.true.)
      do i=1,nexternal
        ICOLUP(1,i)=jpart(4,i)
        ICOLUP(2,i)=jpart(5,i)
      enddo
      
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
      
