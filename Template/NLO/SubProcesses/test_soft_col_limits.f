      program test_soft_col_limits
c*****************************************************************************
c     Given identical particles, and the configurations. This program identifies
c     identical configurations and specifies which ones can be skipped
c*****************************************************************************
      implicit none
      include 'genps.inc'      
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'run.inc'
      include 'cuts.inc'
      include 'mint.inc'
      include 'coupl.inc'
      integer mapconfig(0:lmaxconfigs),iforest(2,-max_branch:-1
     $     ,lmaxconfigs),sprop(-max_branch:-1,lmaxconfigs),tprid(
     $     -max_branch:-1,lmaxconfigs)
      include 'born_conf.inc' ! needed for mapconfig
      double precision ZERO,    one
      parameter       (ZERO=0d0,one=1d0)
      double precision max_fail
      parameter       (max_fail=0.3d0)
      integer i,j,k,n,l,jj,bs_min,bs_max,iconfig_in,nsofttests
     $     ,ncolltests,nerr,imax,iflag,iret,ntry,fks_conf_number
     $     ,fks_loop_min,fks_loop_max,fks_loop,ilim
      double precision fxl,limit(15),wlimit(15),lxp(0:3,nexternal+1)
     $     ,xp(15,0:3,nexternal+1),p(0:3,nexternal),wgt,x(99),fx,totmass
     $     ,xi_i_fks_fix_save,y_ij_fks_fix_save,fail_frac
     $     ,pmass(nexternal)
      double complex wgt1(2)
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer         ndim
      common/tosigint/ndim
      double precision xi_i_fks_fix,y_ij_fks_fix
      common /cxiyfix/ xi_i_fks_fix,y_ij_fks_fix
      logical                calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision   xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt
      logical        softtest,colltest
      common/sctests/softtest,colltest
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      LOGICAL IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
      logical             new_point
      common /c_new_point/new_point
      double precision alsf,besf
      common /cgfunsfp/alsf,besf
      double precision alazi,beazi
      common /cgfunazi/alazi,beazi
      logical         Hevents
      common/SHevents/Hevents
      character*10           MonteCarlo
      common/cMonteCarloType/MonteCarlo
      double precision shower_S_scale(fks_configs*2)
     $     ,shower_H_scale(fks_configs*2),ref_H_scale(fks_configs*2)
     $     ,pt_hardness
      common /cshowerscale2/shower_S_scale,shower_H_scale,ref_H_scale
     &     ,pt_hardness
      double precision ran2
      external         ran2
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
      write(*,*) 'Enter 0 to compute MC/MC(limit)'
      write(*,*) '      1 to compute MC/ME(limit)'
      write(*,*) '      2 to compute ME/ME(limit)'
      read (*,*) ilim

      if (ilim.ne.0 .and. ilim.ne.1 .and. ilim.ne.2) then
         write (*,*) 'ERROR: not a valid choice'
         stop 1
      endif

      if (ilim.eq.0 .or. ilim.eq.1) then
         write(*,*) 'Enter the Monte Carlo name: possible choices are'
         write(*,*) 'HERWIG6, HERWIGPP, PYTHIA6Q, PYTHIA6PT, PYTHIA8'
         read (*,*) MonteCarlo
         if(MonteCarlo.ne.'HERWIG6'.and.MonteCarlo.ne.'HERWIGPP'.and.
     &        MonteCarlo.ne.'PYTHIA6Q'.and.MonteCarlo.ne.'PYTHIA6PT'.and.
     &        MonteCarlo.ne.'PYTHIA8')then
            write(*,*)'Wrong name ',MonteCarlo,' during the tests'
            stop
         endif

         write(*,*) 'Enter alpha, beta for G_soft'
         write(*,*) '  Enter alpha<0 to set G_soft=1 (no ME soft)'
         read (*,*) alsf,besf
         
         write(*,*) 'Enter alpha, beta for G_azi'
         write(*,*) '  Enter alpha>0 to set G_azi=0 (no azi corr)'
         read (*,*) alazi,beazi
      endif

      write(*,*) 'Enter xi_i, y_ij to be used in coll/soft tests'
      write(*,*) ' Enter -2 to generate them randomly'
      read (*,*) xi_i_fks_fix_save,y_ij_fks_fix_save

      write(*,*) 'Enter number of tests for soft and collinear limits'
      read (*,*) nsofttests,ncolltests

      
      if (ilim.eq.0 .or. ilim.eq.1) then
         write(*,*) '  '
         write(*,*) '  '
         write(*,*) '**************************************************'
         write(*,*) '**************************************************'
         write(*,*) '            Testing limits for ',MonteCarlo
         write(*,*) '**************************************************'
         write(*,*) '**************************************************'
         write(*,*) '  '
         write(*,*) '  '
      endif

      call setrun               !Sets up run parameters
      call setpara('param_card.dat') !Sets up couplings and masses
      call setcuts              !Sets up cuts 

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
      
      write(*,*)'  '
      write(*,*)'  '
      write(*,*)"Enter graph number (iconfig), "
     &     //"'0' loops over all graphs"
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

      do iconfig=bs_min,bs_max  ! Born configurations
         ichan=1
         iconfigs(1)=iconfig
         if (ilim.eq.2) then
            call setfksfactor(.false.)
         else
            call setfksfactor(.true.)
         endif
         call setcuts
         ntry=1

         softtest=.false.
         colltest=.false.

         do jj=1,ndim
            x(jj)=ran2()
         enddo
         new_point=.true.
         wgt=1d0
         call generate_momenta(ndim,iconfig,wgt,x,p)
         calculatedBorn=.false.
         do while (( wgt.lt.0 .or. p(0,1).le.0d0 .or. p_born(0,1).le.0d0
     &        ) .and. ntry .lt. 1000)
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
     $           /' Cannot perform ME tests properly for config',iconfig
            cycle
         endif
         call sborn(p_born,wgt1)
      
         write (*,*) ''
         write (*,*) ''
         write (*,*) ''

         Hevents=.true.
         softtest=.true.
         colltest=.false.
         nerr=0
         imax=10
         do j=1,nsofttests
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
            if (ilim.eq.2) then
               calculatedBorn=.false.
               call set_cms_stuff(0)
               call sreal(p1_cnt(0,1,0),zero,y_ij_fks_ev,fxl) 
            else
c Set xi_i_fks to zero, to correctly generate the collinear momenta for the
c configurations close to the soft-collinear limit
               xi_i_fks_fix=0.d0
               wgt=1d0
               call generate_momenta(ndim,iconfig,wgt,x,p)
               calculatedBorn=.false.
               call set_cms_stuff(0)
               calculatedBorn=.false.
c Initialise shower_S_scale to a large value, not to get spurious dead zones
               shower_S_scale=1d10*ebeam(1)
               if(ilim.eq.0)then
                  call xmcsubt_wrap(p1_cnt(0,1,0),zero,y_ij_fks_ev,fxl)
               else
                  call sreal(p1_cnt(0,1,0),zero,y_ij_fks_ev,fxl) 
               endif
            endif
            fxl=fxl*jac_cnt(0)
            if (ilim.eq.2) then
               call set_cms_stuff(-100)
               call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
            else
c Now generate the momenta for the original xi_i_fks=0.1, slightly shifted,
c because otherwise fresh random will be used...
               xi_i_fks_fix=0.100001d0
               wgt=1d0
               call generate_momenta(ndim,iconfig,wgt,x,p)
               calculatedBorn=.false.
               call set_cms_stuff(-100)
               call xmcsubt_wrap(p,xi_i_fks_ev,y_ij_fks_ev,fx)
            endif
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
               if (ilim.eq.2) then
                  call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
               else
                  call xmcsubt_wrap(p,xi_i_fks_ev,y_ij_fks_ev,fx)
               endif
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
     &              iflag,imax,j,nexternal,i_fks,j_fks,iret)
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

         if (pmass(j_fks).ne.0d0) then
            write (*,*) 'No collinear test for massive j_fks'
            goto 123
         endif
         
         softtest=.false.
         colltest=.true.

         nerr=0
         imax=10
         do j=1,ncolltests
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
            calculatedBorn=.false.
            call set_cms_stuff(1)
            if(ilim.eq.0)then
               call xmcsubt_wrap(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,fxl)
            else
               call sreal(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,fxl) 
            endif
            fxl=fxl*jac_cnt(1)

            call set_cms_stuff(-100)
            if (ilim.eq.2) then
               call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
            else
               call xmcsubt_wrap(p,xi_i_fks_ev,y_ij_fks_ev,fx)
            endif
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
               if (ilim.eq.2) then
                  call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
               else
                  call xmcsubt_wrap(p,xi_i_fks_ev,y_ij_fks_ev,fx)
               endif
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
     &              iflag,imax,j,nexternal,i_fks,j_fks,iret)
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
         
 123     continue
         
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
      subroutine initplot
      end
      subroutine outfun(pp,www)
      implicit none
      include 'nexternal.inc'
      real*8 pp(0:3,nexternal),www
      write(*,*)'This routine should not be called here'
      stop
      end
