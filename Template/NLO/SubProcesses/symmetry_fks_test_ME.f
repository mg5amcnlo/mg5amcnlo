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
      
      double precision ZERO,one
      parameter       (ZERO = 0d0)
      parameter       (one = 1d0)
      integer   maxswitch
      parameter(maxswitch=99)
c
c     Local
c
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer mapconfig(0:lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer itree(2,-max_branch:-1)
      integer imatch
      integer i,j, k, n, nsym,l,ii,jj
      double precision diff,xi_i_fks
c$$$      double precision pmass(-max_branch:-1,lmaxconfigs)   !Propagotor mass
      double precision pmass(nexternal)
      double precision pwidth(-max_branch:-1,lmaxconfigs)  !Propagotor width
      integer pow(-max_branch:-1,lmaxconfigs)

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
      double precision fxl,limit(15),wlimit(15)
      double precision lxp(0:3,nexternal+1),xp(15,0:3,nexternal+1)
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
      integer ninvar, ndim, iconfig, minconfig, maxconfig
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
      common/cparticle_types/i_type,j_type,m_type

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
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL  IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
      
c      integer icomp
c
c     DATA
c
      integer tprid(-max_branch:-1,lmaxconfigs)
      include 'born_conf.inc'
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
      if (abs(i_type).eq.abs(j_type)) then
         m_type=8
         if ( (j_fks.le.nincoming .and.
     &        abs(i_type).eq.3 .and. j_type.ne.i_type) .or.
     &        (j_fks.gt.nincoming .and.
     &        abs(i_type).eq.3 .and. j_type.ne.-i_type)) then
            write(*,*)'Flavour mismatch #1 in setfksfactor',
     &           i_fks,j_fks,i_type,j_type
            stop
         endif
      elseif(abs(i_type).eq.3 .and. j_type.eq.8)then
         if(j_fks.le.nincoming)then
            m_type=-i_type
         else
            write (*,*) 'Error in setfksfactor: (i,j)=(q,g)'
            stop
         endif
      elseif(i_type.eq.8 .and. abs(j_type).eq.3)then
         if (j_fks.le.nincoming) then
            m_type=j_type
         else
            m_type=j_type
         endif
      else
         write(*,*)'Flavour mismatch #2 in setfksfactor',
     &        i_type,j_type,m_type
         stop
      endif


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

      do iconfig=bs_min,bs_max       ! Born configurations
      call setcuts
      call setfksfactor(iconfig,.false.)
      wgt=1d0
      ntry=1

      softtest=.false.
      colltest=.false.

      do jj=1,ndim
         x(jj)=ran2()
      enddo
      call generate_momenta(ndim,iconfig,wgt,x,p)
      calculatedBorn=.false.
      do while (( wgt.lt.0 .or. p(0,1).le.0d0 .or. p_born(0,1).le.0d0
     &           ) .and. ntry .lt. 1000)
         do jj=1,ndim
            x(jj)=ran2()
         enddo
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
      imax=10
      do j=1,nsofttests
      call get_helicity(i_fks,j_fks)

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
         call generate_momenta(ndim,iconfig,wgt,x,p)
         do while (( wgt.lt.0 .or. p(0,1).le.0d0) .and. ntry.lt.1000)
            wgt=1d0
            do jj=1,ndim
               x(jj)=ran2()
            enddo
            call generate_momenta(ndim,iconfig,wgt,x,p)
            ntry=ntry+1
         enddo
         if(nsofttests.le.10)write (*,*) 'ntry',ntry
         calculatedBorn=.false.
         call set_cms_stuff(0)
         call sreal(p1_cnt(0,1,0),zero,y_ij_fks_ev,fxl) 
         fxl=fxl*jac_cnt(0)

         call set_cms_stuff(-100)
         call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
         limit(1)=fx*wgt
         wlimit(1)=wgt

         do k=1,nexternal
            do l=0,3
               lxp(l,k)=p1_cnt(l,k,0)
               xp(1,l,k)=p(l,k)
            enddo
         enddo
         do l=0,3
            lxp(l,nexternal+1)=p_i_fks_cnt(l,0)
            xp(1,l,nexternal+1)=p_i_fks_ev(l)
         enddo

         do i=2,imax
            xi_i_fks_fix=xi_i_fks_fix/10d0
            wgt=1d0
            call generate_momenta(ndim,iconfig,wgt,x,p)
            calculatedBorn=.false.
            call set_cms_stuff(-100)
            call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
            limit(i)=fx*wgt
            wlimit(i)=wgt
            do k=1,nexternal
               do l=0,3
                  xp(i,l,k)=p(l,k)
               enddo
            enddo
            do l=0,3
               xp(i,l,nexternal+1)=p_i_fks_ev(l)
            enddo
         enddo

         if(nsofttests.le.10)then
           write (*,*) 'Soft limit:'
           do i=1,imax
              call xprintout(6,limit(i),fxl)
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
                    call xprintout(80,xp(i,l,k),lxp(l,k))
                 enddo
              enddo
           enddo
        else
           iflag=0
           call checkres(limit,fxl,wlimit,jac_cnt(0),xp,lxp,
     &                   iflag,imax,j,nexternal,i_fks,j_fks,iret)
           nerr=nerr+iret
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
      imax=10
      do j=1,ncolltests
         call get_helicity(i_fks,j_fks)

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
         call generate_momenta(ndim,iconfig,wgt,x,p)
         do while (( wgt.lt.0 .or. p(0,1).le.0d0) .and. ntry.lt.1000)
            wgt=1d0
            do jj=1,ndim
               x(jj)=ran2()
            enddo
            call generate_momenta(ndim,iconfig,wgt,x,p)
            ntry=ntry+1
         enddo
         if(ncolltests.le.10)write (*,*) 'ntry',ntry
         calculatedBorn=.false.
         call set_cms_stuff(1)
         call sreal(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,fxl) 
         fxl=fxl*jac_cnt(1)

         call set_cms_stuff(-100)
         call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
         limit(1)=fx*wgt
         wlimit(1)=wgt

         do k=1,nexternal
            do l=0,3
               lxp(l,k)=p1_cnt(l,k,1)
               xp(1,l,k)=p(l,k)
            enddo
         enddo
         do l=0,3
            lxp(l,nexternal+1)=p_i_fks_cnt(l,1)
            xp(1,l,nexternal+1)=p_i_fks_ev(l)
         enddo

         do i=2,imax
            y_ij_fks_fix=1-0.1d0**i
            wgt=1d0
            call generate_momenta(ndim,iconfig,wgt,x,p)
            calculatedBorn=.false.
            call set_cms_stuff(-100)
            call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
            limit(i)=fx*wgt
            wlimit(i)=wgt
            do k=1,nexternal
               do l=0,3
                  xp(i,l,k)=p(l,k)
               enddo
            enddo
            do l=0,3
               xp(i,l,nexternal+1)=p_i_fks_ev(l)
            enddo
         enddo
         if(ncolltests.le.10)then
            write (*,*) 'Collinear limit:'
            do i=1,imax
               call xprintout(6,limit(i),fxl)
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
                     call xprintout(80,xp(i,l,k),lxp(l,k))
                  enddo
               enddo
            enddo
         else
            iflag=1
            call checkres(limit,fxl,wlimit,jac_cnt(1),xp,lxp,
     &                    iflag,imax,j,nexternal,i_fks,j_fks,iret)
            nerr=nerr+iret
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
