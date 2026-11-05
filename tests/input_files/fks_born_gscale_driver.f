      program test_born_gscale
c Regression driver for the gscale path of the FKS Born (SBORN with
c check_born_amps): at fixed momenta, the Born at a strong coupling g2
c obtained by rescaling the amplitudes cached at g1 must agree with a
c fresh evaluation at g2, for every split-order piece (amp_split), the
c counterterm pieces (ans_cnt, amp_split_cnt: Born tilde), AMP2/JAMP2
c and every colour link. Four modes, per FKS configuration and point:
c   0 fast path : the result cache is valid, it is rescaled as well
c   1 loop path : the cache belongs to another FKS configuration, so
c                 the colour loop runs on the rescaled amplitudes
c   2 loop path : a single-helicity call invalidated the result cache
c   3 refusal   : a coupling of QCD order 0 (QED_COUPLING) changed
c                 together with g, the cache must be discarded
c The test inserts the counter NGSCALE_TEST at the born_gscale call of
c born.f, so that a silent fallback to a fresh evaluation is caught.
c Placeholders filled by the test: QED_COUPLING, NLINKS_VALUE.
      use mint_module
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'run.inc'
      include 'cuts.inc'
      include 'coupl.inc'
      include 'born_conf.inc'
      include 'orders.inc'
      integer jj,ntry,ipt,npts,mode,ifks,nfks_save,nval,nexp
      integer ncount(0:3)
      double precision x(99),p(0:3,nexternal),wgt,g1,g2,wb
      double precision ref(5000),tst(5000),rf3(5000),worst(0:3)
      double precision shift3
      double complex coup_save
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      logical check_born_amps, born_checks_amps
      common/c_born_amps_check/check_born_amps,born_checks_amps
      integer ngscale_test
      common/c_gscale_test/ngscale_test
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      double precision pb(0:3,nexternal-1)
      integer         nndim
      common/tosigint/nndim
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      double precision ran2
      external         ran2
      include 'pythia8_control.inc'
      include 'pythia8_control_setup.inc'

      call setrun
      call setpara('param_card.dat')
      call fill_configurations_common
      g1=g
      g2=g1*1.0731d0
      npts=2
      nexp=0
      worst(:)=0d0
      ncount(:)=0
      shift3=1d99
      do ifks=1,fks_configs
       nFKSprocess=ifks
       call fks_inc_chooser()
       call leshouche_inc_chooser()
       ndim = 3*(nexternal-nincoming)-4
       if (abs(lpp(1)).ge.1) ndim=ndim+1
       if (abs(lpp(2)).ge.1) ndim=ndim+1
       nndim=ndim
       iconfig=1
       ichan=1
       iconfigs(1)=iconfig
       call setfksfactor(.false.)
       call setcuts
       do ipt=1,npts
        ntry=0
        wgt=-1d0
        p(0,1)=0d0
        p_born(0,1)=0d0
        do while ((wgt.le.0d0 .or. p(0,1).le.0d0 .or.
     &       p_born(0,1).le.0d0) .and. ntry.lt.1000)
          do jj=1,ndim
            x(jj)=ran2()
          enddo
          new_point=.true.
          wgt=1d0
          call generate_momenta(ndim,iconfig,wgt,x,p)
          ntry=ntry+1
        enddo
        if (ntry.ge.1000) stop 'no phase-space point'
        pb(:,:)=p_born(:,:)
        nexp=nexp+1
c reference: fresh evaluation at g2
        call set_g(g2)
        calculatedBorn=.false.
        call collect(pb,ref,nval)
        do mode=0,3
c fresh evaluation at g1: the cache then holds the g1 amplitudes
          call set_g(g1)
          nfks_save=nFKSprocess
          if (mode.eq.1) then
            nFKSprocess=mod(nfks_save,fks_configs)+1
            call fks_inc_chooser()
            call leshouche_inc_chooser()
          endif
          calculatedBorn=.false.
          call sborn(pb,wb)
          if (mode.eq.1) then
            nFKSprocess=nfks_save
            call fks_inc_chooser()
            call leshouche_inc_chooser()
          endif
          if (mode.eq.2) call touch_onehel(pb)
          call set_g(g2)
          if (mode.eq.3) then
            coup_save=QED_COUPLING
            QED_COUPLING=QED_COUPLING*1.1d0
          endif
c same momenta, other couplings: request the check (as set_alphaS
c and the fold reset of driver_mintMC do)
          check_born_amps=.true.
          ngscale_test=0
          call collect(pb,tst,nval)
          ncount(mode)=ncount(mode)+ngscale_test
          if (mode.eq.3) then
            calculatedBorn=.false.
            call collect(pb,rf3,nval)
            call compare(rf3,tst,nval,worst(mode))
            shift3=min(shift3,abs(rf3(1)-ref(1))/abs(ref(1)))
            QED_COUPLING=coup_save
          else
            call compare(ref,tst,nval,worst(mode))
          endif
        enddo
       enddo
      enddo
      write(*,'(a,i8)') ' GSCALE NPOINTS ',nexp
      write(*,'(a,i8)') ' GSCALE NVALUES ',nval
      do mode=0,3
        write(*,'(a,i2,e12.4,i8)') ' GSCALE MODE ',mode,worst(mode),
     &       ncount(mode)
      enddo
      write(*,'(a,e12.4)') ' GSCALE REFUSAL_SHIFT ',shift3
      end


      subroutine set_g(gx)
      implicit none
      include 'coupl.inc'
      double precision gx
      g=gx
      call update_as_param()
      end


      subroutine touch_onehel(pb)
c a single-helicity Born call (as for the EW Sudakov): it invalidates
c the result cache of SBORN but keeps the cached amplitudes
      implicit none
      include 'nexternal.inc'
      double precision pb(0:3,nexternal-1),w
      integer nhel(nexternal-1),i
      do i=1,nexternal-1
        nhel(i)=-1
      enddo
      call sborn_onehel(pb,nhel,1,w)
      end


      subroutine collect(pb,v,n)
c the Born, its split-order pieces, the counterterm pieces, AMP2/JAMP2
c and the colour links, in one flat array
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'born_maxamps.inc'
      double precision pb(0:3,nexternal-1),v(*),wb,w
      integer n,i,j,k,m
      double precision amp2(maxamps),jamp2(0:maxflow)
      common/to_amps/  amp2,       jamp2
      complex*16 ans_cnt(2,nsplitorders)
      common /c_born_cnt/ ans_cnt
      call sborn(pb,wb)
      n=1
      v(n)=wb
      do i=1,amp_split_size
        n=n+1
        v(n)=amp_split(i)
      enddo
      do k=1,nsplitorders
        do j=1,2
          n=n+1
          v(n)=dble(ans_cnt(j,k))
          n=n+1
          v(n)=dimag(ans_cnt(j,k))
          do i=1,amp_split_size
            n=n+1
            v(n)=dble(amp_split_cnt(i,j,k))
            n=n+1
            v(n)=dimag(amp_split_cnt(i,j,k))
          enddo
        enddo
      enddo
      do i=1,maxamps
        n=n+1
        v(n)=amp2(i)
      enddo
      do i=1,maxflow
        n=n+1
        v(n)=jamp2(i)
      enddo
      do m=1,NLINKS_VALUE
        call sb_sf_link(pb,m,w)
        n=n+1
        v(n)=w
      enddo
      end


      subroutine compare(a,b,n,worst)
c largest difference, relative to the entry or to 1d-10 of the largest
c entry (the colour links cancel between flows)
      implicit none
      integer n,i
      double precision a(n),b(n),worst,amax,d
      amax=0d0
      do i=1,n
        amax=max(amax,abs(a(i)))
      enddo
      do i=1,n
        d=abs(a(i)-b(i))/max(abs(a(i)),1d-10*amax,1d-300)
        worst=max(worst,d)
      enddo
      end
