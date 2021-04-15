      program symmetry
c*****************************************************************************
c     Given identical particles, and the configurations. This program identifies
c     identical configurations and specifies which ones can be skipped
c*****************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'      
      include 'nexternal.inc'
      include '../../Source/run_config.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'run.inc'
      include 'cuts.inc'
      include 'mint.inc'
      
      double precision ZERO,one
      parameter       (ZERO = 0d0)
      parameter       (one = 1d0)
      integer   maxswitch
      parameter(maxswitch=99)
c
c     Local
c
      integer itree(2,-max_branch:-1)
      integer imatch
      integer i,j, k, n, nsym,l,ii,jj
      double precision diff,xi_i_fks
      double precision pmass(nexternal)

      integer biforest(2,-max_branch:-1,lmaxconfigs)
      integer fksmother,fksgrandmother,fksaunt,compare
      integer fksconfiguration,mapbconf(0:lmaxconfigs)
      integer r2b(lmaxconfigs),b2r(lmaxconfigs)
      logical searchforgranny,is_beta_cms,is_granny_sch,topdown,non_prop
      integer nbranch,ns_channel,nt_channel
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      double precision fxl(15),wfxl(15),limit(15),wlimit(15)
      double precision lxp(15,0:3,nexternal+1),xp(15,0:3,nexternal+1)
      double precision fks_Sij
      double precision check,tolerance,zh,h_damp
      parameter (tolerance=1.d-4)
      integer kk,ll,bs,bs_min,bs_max,iconfig_in

      integer nsofttests,ncolltests,nerr,imax,iflag,iret
c
c     Local for generating amps
c
      double precision p(0:3,99), wgt, x(99), fx
      double complex wgt1(2)
      double precision p1(0:3,99),xx(maxinvar)
      integer ninvar, ndim,  minconfig, maxconfig
      common/tosigint/ndim
      integer ncall,itmax,nconfigs,ntry, ngraphs
      integer ic(nexternal,maxswitch), jc(12),nswitch
      double precision saveamp(maxamps)
      integer nmatch, ibase
      logical mtc, even

      double precision totmass

      double precision xi_i_fks_fix_save,y_ij_fks_fix_save
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c
c     Global
c
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2
      include 'coupl.inc'

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt

      logical rotategranny
      common/crotategranny/rotategranny

      logical softtest,colltest
      common/sctests/softtest,colltest
      
      logical xexternal
      common /toxexternal/ xexternal

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m
      double precision particle_charge(nexternal), particle_charge_born(nexternal-1)
      common /c_charges/particle_charge
      common /c_charges_born/particle_charge_born

c
c     External
c
      logical check_swap
      double precision dsig,ran2
      external dsig,ran2
      external check_swap, fks_Sij

c define here the maximum fraction of failures to consider the test
c   passed
      double precision max_fail, fail_frac
      parameter (max_fail=0.3d0)

c helicity stuff
      integer          isum_hel
      logical                    multi_channel
      common/to_matrix/isum_hel, multi_channel

      integer fks_conf_number,fks_loop_min,fks_loop_max,fks_loop
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS

C split orders stuff
      include 'orders.inc'
      integer iamp
      integer orders(nsplitorders)
      double precision fxl_split(15,amp_split_size),wfxl_split(15,amp_split_size)
      double precision limit_split(15,amp_split_size), wlimit_split(15,amp_split_size)

c born configuration stuff
      include 'born_ngraphs.inc'
      include 'born_conf.inc'
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL  IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
      
      logical new_point
      common /c_new_point/new_point
c      integer icomp
c-----
c  Begin Code
c-----
      if (fks_configs.eq.1) then
         if (pdg_type_d(1,fks_i_d(1)).eq.-21) then
            write (*,*) 'Process generated with [LOonly=QCD]. '/
     $           /'No tests to do.'
            return
         endif
      endif

      write(*,*)'Enter xi_i, y_ij to be used in coll/soft tests'
      write(*,*)' Enter -2 to generate them randomly'
      read(*,*)xi_i_fks_fix_save,y_ij_fks_fix_save

      write(*,*)'Enter number of tests for soft and collinear limits'
      read(*,*)nsofttests,ncolltests

      write(*,*)'Sum over helicity (0), or random helicity (1)'
      read(*,*) isum_hel

      call setrun                !Sets up run parameters
      call setpara('param_card.dat')   !Sets up couplings and masses
      call setcuts               !Sets up cuts 

c When doing hadron-hadron collision reduce the effect collision energy.
c Note that tests are always performed at fixed energy with Bjorken x=1.
      totmass = 0.0d0
      include 'pmass.inc' ! make sure to set the masses after the model has been included
      do i=nincoming+1,nexternal
         if (is_a_j(i) .and. i.ne.nexternal) then
            totmass = totmass + max(ptj,pmass(i))
         elseif ((is_a_lp(i).or.is_a_lm(i)) .and. i.ne.nexternal) then
            totmass = totmass + max(mll/2d0,mll_sf/2d0,ptl,pmass(i))
         else
            totmass = totmass + pmass(i)
         endif
      enddo
      if (lpp(1).ne.0) ebeam(1)=max(ebeam(1)/20d0,totmass)
      if (lpp(2).ne.0) ebeam(2)=max(ebeam(2)/20d0,totmass)
c

      write (*,*) 'Give FKS configuration number ("0" loops over all)'
      read (*,*) fks_conf_number

      if (fks_conf_number.eq.0) then
         fks_loop_min=1
         fks_loop_max=fks_configs
      else
         fks_loop_min=fks_conf_number
         fks_loop_max=fks_conf_number
      endif

      do fks_loop=fks_loop_min,fks_loop_max
         nFKSprocess=fks_loop
         write (*,*) ''
         write (*,*) '================================================='
         write (*,*) ''
         write (*,*) 'NEW FKS CONFIGURATION:'

         call fks_inc_chooser()
         call leshouche_inc_chooser()
         write (*,*) 'FKS configuration number is ',nFKSprocess
         write (*,*) 'FKS partons are: i=',i_fks,'  j=',j_fks
         write (*,*) 'with PDGs:       i=',PDG_type(i_fks),'  j='
     $        ,PDG_type(j_fks)

c
      ndim = 55
      ncall = 10000
      itmax = 10
      ninvar = 35
      nconfigs = 1

c Set color types of i_fks, j_fks and fks_mother.
      i_type=particle_type(i_fks)
      j_type=particle_type(j_fks)
      ch_i=particle_charge(i_fks)
      ch_j=particle_charge(j_fks)
      call get_mother_col_charge(i_type,ch_i,j_type,ch_j,m_type,ch_m) 


c     
c     Get momentum configuration
c

c Set xexternal to true to use the x's from external vegas in the
c x_to_f_arg subroutine
      xexternal=.true.
      
      write(*,*)'  '
      write(*,*)'  '
      write(*,*)'Enter graph number (iconfig), '
     &     //"'0' loops over all graphs"
      read(*,*)iconfig_in
      
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

      do iconfig=bs_min,bs_max  ! Born configurations
         ichan=1
         iconfigs(1)=iconfig
      call setcuts
      call setfksfactor(.false.)
      wgt=1d0
      ntry=1

      softtest=.false.
      colltest=.false.

      do jj=1,ndim
         x(jj)=ran2()
      enddo
      new_point=.true.
      call generate_momenta(ndim,iconfig,wgt,x,p)
      calculatedBorn=.false.
      do while (( wgt.lt.0 .or. p(0,1).le.0d0 .or. p_born(0,1).le.0d0
     &           ) .and. ntry .lt. 1000)
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
         write (12,*) 'ERROR: no points passed cuts...'
     &        //' Cannot perform ME tests properly for config',iconfig
         exit
      endif

      call sborn(p_born,wgt1)
      
      write (*,*) ''
      write (*,*) ''
      write (*,*) ''

      softtest=.true.
      colltest=.false.
      nerr=0
      imax=14
      do j=1,nsofttests
      call get_helicity(i_fks,j_fks)
         do iamp=1,amp_split_size
            do i = 1,imax
               fxl_split(i,iamp) = 0d0
               wfxl_split(i,iamp) = 0d0
               limit_split(i,iamp) = 0d0
               wlimit_split(i,iamp) = 0d0
            enddo
         enddo

         if(nsofttests.le.10)then
           write (*,*) ' '
           write (*,*) ' '
         endif

         y_ij_fks_fix=y_ij_fks_fix_save
         xi_i_fks_fix=0.1d0
         ntry=1
         wgt=1d0
         do jj=1,ndim
            x(jj)=ran2()
         enddo
         new_point=.true.
         call generate_momenta(ndim,iconfig,wgt,x,p)
         do while (( wgt.lt.0 .or. p(0,1).le.0d0) .and. ntry.lt.1000)
            wgt=1d0
            do jj=1,ndim
               x(jj)=ran2()
            enddo
            new_point=.true.
            call generate_momenta(ndim,iconfig,wgt,x,p)
            ntry=ntry+1
         enddo
         if(nsofttests.le.10)write (*,*) 'ntry',ntry
         do i=1,imax
            wgt=1d0
            call generate_momenta(ndim,iconfig,wgt,x,p)
            calculatedBorn=.false.
            call set_cms_stuff(0)
            call sreal(p1_cnt(0,1,0),zero,y_ij_fks_ev,fx)
            fxl(i)=fx*wgt
            wfxl(i)=jac_cnt(0)
! keep track of the separate splitorders
            do iamp=1,amp_split_size
               fxl_split(i,iamp) = amp_split(iamp)*jac_cnt(0)
               wfxl_split(i,iamp)=jac_cnt(0)
            enddo
            calculatedBorn=.false.
            call set_cms_stuff(-100)
            call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
            limit(i)=fx*wgt
            wlimit(i)=wgt
         ! keep track of the separate splitorders
            do iamp=1,amp_split_size
               limit_split(i,iamp) = amp_split(iamp)*wgt
               wlimit_split(i,iamp) = wgt
            enddo
            do k=1,nexternal
               do l=0,3
                  xp(i,l,k)=p(l,k)
                  lxp(i,l,k)=p1_cnt(l,k,0)
               enddo
            enddo
            do l=0,3
               xp(i,l,nexternal+1)=p_i_fks_ev(l)
               lxp(i,l,nexternal+1)=p_i_fks_cnt(l,0)
            enddo
            xi_i_fks_fix=xi_i_fks_fix/10d0
         enddo

         if(nsofttests.le.10)then
           write (*,*) 'Soft limit:'
           write (*,*) '   Sum of all contributions:'
           do i=1,imax
              call xprintout(6,limit(i),fxl(i))
           enddo
           ! check the contributions coming from each splitorders
           ! only look at the non vanishing ones
           do iamp=1, amp_split_size
             if (limit_split(1,iamp).ne.0d0.or.fxl_split(1,iamp).ne.0d0) then
               write(*,*) '   Split-order', iamp
               call amp_split_pos_to_orders(iamp,orders)
               do i = 1, nsplitorders
                  write(*,*) '      ',ordernames(i), ':', orders(i)
               enddo
               do i=1,imax
                  call xprintout(6,limit_split(i,iamp),fxl_split(i,iamp))
               enddo
               call checkres2(limit_split(1,iamp),fxl_split(1,iamp),
     &                   wlimit_split(1,iamp),wfxl_split(1,iamp),xp,lxp,
     &                   iflag,imax,j,nexternal,i_fks,j_fks,iret)
               write(*,*) 'RETURN CODE', iret
             endif
           enddo
           
c
           write(80,*)'  '
           write(80,*)'****************************'
           write(80,*)'  '
           do k=1,nexternal+1
              write(80,*)''
              write(80,*)'part:',k
              do l=0,3
                 write(80,*)'comp:',l
                 do i=1,10
                    call xprintout(80,xp(i,l,k),lxp(i,l,k))
                 enddo
              enddo
           enddo
        else
           iflag=0
           call checkres2(limit,fxl,wlimit,wfxl,xp,lxp,
     &                   iflag,imax,j,nexternal,i_fks,j_fks,iret)
           nerr=nerr+iret
           ! check the contributions coming from each splitorders
           ! only look at the non vanishing ones
           do iamp=1, amp_split_size
             if (limit_split(1,iamp).ne.0d0.or.fxl_split(1,iamp).ne.0d0) then
               call checkres2(limit_split(1,iamp),fxl_split(1,iamp),
     &                   wlimit_split(1,iamp),wfxl_split(1,iamp),xp,lxp,
     &                   iflag,imax,j,nexternal,i_fks,j_fks,iret)
               nerr=nerr+iret
             endif
           enddo
        endif

      enddo
      if(nsofttests.gt.10)then
         write(*,*)'Soft tests done for (Born) config',iconfig
         write(*,*)'Failures:',nerr
         fail_frac= nerr/dble(nsofttests)
         if (fail_frac.lt.max_fail) then
             write(*,401) nFKSprocess, fail_frac
         else
             write(*,402) nFKSprocess, fail_frac
         endif
      endif

      write (*,*) ''
      write (*,*) ''
      write (*,*) ''
      
      include 'pmass.inc'

      if (pmass(j_fks).ne.0d0) then
         write (*,*) 'No collinear test for massive j_fks'
         goto 123
      endif

      softtest=.false.
      colltest=.true.

c Set rotategranny=.true. to align grandmother along the z axis, when 
c grandmother is not the c.m. system (if granny=cms, this rotation coincides
c with the identity, and the following is harmless).
c WARNING: the setting of rotategranny changes the definition of xij_aor
c in genps_fks_test.f
      rotategranny=.false.

      nerr=0
      imax=14
      do j=1,ncolltests
         call get_helicity(i_fks,j_fks)
         do iamp=1,amp_split_size
            do i = 1,imax
               fxl_split(i,iamp) = 0d0
               wfxl_split(i,iamp) = 0d0
               limit_split(i,iamp) = 0d0
               wlimit_split(i,iamp) = 0d0
            enddo
         enddo

         if(ncolltests.le.10)then
            write (*,*) ' '
            write (*,*) ' '
         endif

         y_ij_fks_fix=0.9d0
         xi_i_fks_fix=xi_i_fks_fix_save
         ntry=1
         wgt=1d0
         do jj=1,ndim
            x(jj)=ran2()
         enddo
         new_point=.true.
         call generate_momenta(ndim,iconfig,wgt,x,p)
         do while (( wgt.lt.0 .or. p(0,1).le.0d0) .and. ntry.lt.1000)
            wgt=1d0
            do jj=1,ndim
               x(jj)=ran2()
            enddo
            new_point=.true.
            call generate_momenta(ndim,iconfig,wgt,x,p)
            ntry=ntry+1
         enddo
         if(ncolltests.le.10)write (*,*) 'ntry',ntry
         do i=1,imax
            y_ij_fks_fix=1-0.1d0**i
            wgt=1d0
            call generate_momenta(ndim,iconfig,wgt,x,p)
            calculatedBorn=.false.
            call set_cms_stuff(1)
            call sreal(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,fx) 
            fxl(i)=fx*jac_cnt(1)
         ! keep track of the separate splitorders
            do iamp=1,amp_split_size
               fxl_split(i,iamp) = amp_split(iamp)*jac_cnt(1)
               wfxl_split(i,iamp) = jac_cnt(1)
            enddo
            wfxl(i)=jac_cnt(1)
            calculatedBorn=.false.
            call set_cms_stuff(-100)
            call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
            limit(i)=fx*wgt
            wlimit(i)=wgt
            ! keep track of the separate splitorders
            do iamp=1,amp_split_size
              limit_split(i,iamp) = amp_split(iamp)*wgt
              wlimit_split(i,iamp) = wgt
            enddo
            do k=1,nexternal
               do l=0,3
                  lxp(i,l,k)=p1_cnt(l,k,1)
                  xp(i,l,k)=p(l,k)
               enddo
            enddo
            do l=0,3
               lxp(i,l,nexternal+1)=p_i_fks_cnt(l,1)
               xp(i,l,nexternal+1)=p_i_fks_ev(l)
            enddo
         enddo
         if(ncolltests.le.10)then
            write (*,*) 'Collinear limit:'
           write (*,*) '   Sum of all contributions:'
            do i=1,imax
               call xprintout(6,limit(i),fxl(i))
            enddo
           ! check the contributions coming from each splitorders
           ! only look at the non vanishing ones
           do iamp=1, amp_split_size
             if (limit_split(1,iamp).ne.0d0.or.fxl_split(1,iamp).ne.0d0) then
               write(*,*) '   Split-order', iamp
               call amp_split_pos_to_orders(iamp,orders)
               do i = 1, nsplitorders
                  write(*,*) '      ',ordernames(i), ':', orders(i)
               enddo
               do i=1,imax
                  call xprintout(6,limit_split(i,iamp),fxl_split(i,iamp))
               enddo
               call checkres2(limit_split(1,iamp),fxl_split(1,iamp),
     &                   wlimit_split(1,iamp),wfxl_split(1,iamp),xp,lxp,
     &                   iflag,imax,j,nexternal,i_fks,j_fks,iret)
               write(*,*) 'RETURN CODE', iret
             endif
           enddo
c
            write(80,*)'  '
            write(80,*)'****************************'
            write(80,*)'  '
            do k=1,nexternal+1
               write(80,*)''
               write(80,*)'part:',k
               do l=0,3
                  write(80,*)'comp:',l
                  do i=1,10
                     call xprintout(80,xp(i,l,k),lxp(i,l,k))
                  enddo
               enddo
            enddo
         else
            iflag=1
           call checkres2(limit,fxl,wlimit,wfxl,xp,lxp,
     &                   iflag,imax,j,nexternal,i_fks,j_fks,iret)
            nerr=nerr+iret
           ! check the contributions coming from each splitorders
           ! only look at the non vanishing ones
           do iamp=1, amp_split_size
             if (limit_split(1,iamp).ne.0d0.or.fxl_split(1,iamp).ne.0d0) then
               call checkres2(limit_split(1,iamp),fxl_split(1,iamp),
     &                   wlimit_split(1,iamp),wfxl_split(1,iamp),xp,lxp,
     &                   iflag,imax,j,nexternal,i_fks,j_fks,iret)
               nerr=nerr+iret
             endif
           enddo
         endif
      enddo
      if(ncolltests.gt.10)then
         write(*,*)'Collinear tests done for (Born) config', iconfig
         write(*,*)'Failures:',nerr
         fail_frac= nerr/dble(ncolltests)
         if (fail_frac.lt.max_fail) then
             write(*,501) nFKSprocess, fail_frac
         else
             write(*,502) nFKSprocess, fail_frac
         endif
      endif

 123  continue

      enddo                     ! Loop over Born configurations
      enddo                     ! Loop over nFKSprocess


      return
 401  format('     Soft test ',i2,' PASSED. Fraction of failures: ',
     & f4.2) 
 402  format('     Soft test ',I2,' FAILED. Fraction of failures: ',
     & f4.2) 
 501  format('Collinear test ',i2,' PASSED. Fraction of failures: ',
     & f4.2) 
 502  format('Collinear test ',I2,' FAILED. Fraction of failures: ',
     & f4.2) 
      end

c
c
c Dummy routines
c
c
      subroutine clear_events()
      end
      subroutine initplot
      end
      subroutine store_events()
      end
      integer function n_unwgted()
      n_unwgted = 1
      end

      subroutine outfun(pp,www)
      implicit none
      include 'nexternal.inc'
      real*8 pp(0:3,nexternal),www
c
      write(*,*)'This routine should not be called here'
      stop
      end
