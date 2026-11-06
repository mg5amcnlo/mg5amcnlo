      program test_born_gscale
c Regression driver for native FKS Born rescaling: at fixed momenta,
c the Born at a strong coupling g2
c obtained by rescaling the amplitudes cached at g1 must agree with a
c fresh evaluation at g2, for every split-order piece (amp_split), the
c counterterm pieces (ans_cnt, amp_split_cnt: Born tilde), AMP2/JAMP2
c and every colour link. Four modes, per FKS configuration and point:
c   0 same sector: rescale the cached amplitudes and contract them
c   1 new sector: the cache belongs to another FKS configuration, so
c                 the colour loop runs on the rescaled amplitudes
c   2 one helicity: a single-helicity query precedes the coupling change
c   3 refusal   : a coupling of QCD order 0 (QED_COUPLING) changed
c                 together with g, the cache must be discarded
c Test-only private-provider instrumentation counts rescalings and HELAS
c calls, and provides a one-shot cache reset for truly fresh references.
c Placeholders: QED_COUPLING, NLINKS_VALUE, LINK_M_VALUES, LINK_N_VALUES.
      use mint_module
      use fks_phase_space, only: generate_momenta
      use fks_phase_space_data, only: p_born
      use mc_born_types, only: born_test_fresh,born_test_scaled,
     &                        born_test_helas
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
      integer ncount(0:3),nhelas(0:3)
      double precision x(99),p(0:3,nexternal),wgt,g1,g2,wb
      double precision p_lab(0:3,nexternal),p_cms(0:3,nexternal)
      double precision ref(5000),tst(5000),rf3(5000),worst(0:3)
      double precision shift3
      double complex coup_save
      double precision pb(0:3,nexternal-1)
      integer         nndim
      common/tosigint/nndim
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      double precision ran2
      external         ran2
      call setrun
      call setpara('param_card.dat')
      call fill_configurations_common
      g1=g
      g2=g1*1.0731d0
      npts=2
      nexp=0
      worst(:)=0d0
      ncount(:)=0
      nhelas(:)=0
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
          call generate_momenta(ndim,iconfig,wgt,x,p,p_lab,p_cms)
          ntry=ntry+1
        enddo
        if (ntry.ge.1000) stop 'no phase-space point'
        pb(:,:)=p_born(:,:)
        nexp=nexp+1
c reference: fresh evaluation at g2
        call set_g(g2)
        born_test_fresh=.true.
        born_test_helas=0
        call collect(pb,ref,nval)
        if(born_test_helas.le.0)error stop 11
        do mode=0,3
c fresh evaluation at g1: the cache then holds the g1 amplitudes
          call set_g(g1)
          nfks_save=nFKSprocess
          if (mode.eq.1) then
            nFKSprocess=mod(nfks_save,fks_configs)+1
            call fks_inc_chooser()
            call leshouche_inc_chooser()
          endif
          born_test_fresh=.true.
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
            call mc_born_model_changed()
          endif
c Native requests compare the configuration ID and dynamic values.
          born_test_scaled=0
          born_test_helas=0
          call collect(pb,tst,nval)
          ncount(mode)=ncount(mode)+born_test_scaled
          nhelas(mode)=nhelas(mode)+born_test_helas
          if (mode.eq.3) then
            born_test_fresh=.true.
            born_test_helas=0
            call collect(pb,rf3,nval)
            if(born_test_helas.le.0)error stop 12
            call compare(rf3,tst,nval,worst(mode))
            shift3=min(shift3,abs(rf3(1)-ref(1))/abs(ref(1)))
            QED_COUPLING=coup_save
            call mc_born_model_changed()
          else
            call compare(ref,tst,nval,worst(mode))
          endif
        enddo
       enddo
      enddo
      write(*,'(a,i8)') ' GSCALE NPOINTS ',nexp
      write(*,'(a,i8)') ' GSCALE NVALUES ',nval
      do mode=0,3
        write(*,'(a,i2,e12.4,2i8)') ' GSCALE MODE ',mode,worst(mode),
     &       ncount(mode),nhelas(mode)
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
c A single-helicity Born query, as for the EW Sudakov.
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
      integer link_m(NLINKS_VALUE),link_n(NLINKS_VALUE)
      data link_m/LINK_M_VALUES/
      data link_n/LINK_N_VALUES/
      logical need_color_links,need_charge_links
      common/c_need_links/need_color_links,need_charge_links
      double precision amp_split_soft(amp_split_size)
      common/to_amp_split_soft/amp_split_soft
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
        need_color_links=.true.
        need_charge_links=.false.
        call sborn_sf(pb,link_m(m),link_n(m),w)
        n=n+1
        v(n)=w
        do i=1,amp_split_size
          n=n+1
          v(n)=amp_split_soft(i)
        enddo
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
