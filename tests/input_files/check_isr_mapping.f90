! Global ISR mapping and FKS finite integrals using production routines.
subroutine check_isr_mapping(mode)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mc_native_context, only: native_mapping
  use FKSParams, only: FKSISRMapping
  use fks_phase_space_data, only: pb => p_born,pbl => p_born_l,pbe => p_born_ev, &
       bound_born => tau_Born_lower_bound,bound_res => tau_lower_bound_resonance, &
       bound_tau => tau_lower_bound,pc => p1_cnt,jc => jac_cnt, &
       xi => xi_i_fks_ev,y => y_ij_fks_ev,xic => xi_i_fks_cnt, &
       xb => xbjrk_ev,xbc => xbjrk_cnt,xmax => xiimax_ev,xmaxc => xiimax_cnt
  use fks_phase_space, only: generate_FKS_kinematics,invert_fks_radiation, &
       compute_flux,boost_born_momenta_noevpr
  use fks_phase_space_helpers, only: boost_n1_to_lab,boost_n1_to_its_cms
  use fks_radiation_maps, only: generate_momenta_initial
  implicit none
  include 'genps.inc'
  include 'nexternal.inc'
  include 'run.inc'
  include 'fks_powers.inc'
  character(len=*),intent(in) :: mode
  double precision,parameter :: pi=3.1415926535897932385d0
  double precision,parameter :: xicuts(3)=[.1d0,.5d0,.9d0],deltacuts(3)=[.2d0,1d0,1.8d0]
  double precision :: pmass(nexternal)
  common/to_mass/pmass
  integer :: ifks,jfks
  common/fks_indices/ifks,jfks
  double precision :: omx(2)
  common/to_ee_omx1/omx
  logical :: nbody,evpr,softtest,colltest
  logical :: fixed_order,nlo_ps,symmetric_map
  common/c_fnlo_nlops/fixed_order,nlo_ps
  common/cnbody/nbody
  common/to_use_evpr/evpr
  common/sctests/softtest,colltest
  double precision :: xi_fix,y_fix
  common/cxiyfix/xi_fix,y_fix
  double precision :: delta_used,xicut_used,xiScut_used,xiBSVcut_used
  common/cdelta_used/delta_used
  common/cxicut_used/xicut_used
  common/cxiScut_used/xiScut_used,xiBSVcut_used
  double precision :: symmetry,symmetry_b,symmetry_d
  integer :: ngluons,nquarks(-6:6),nphotons
  common/numberofparticles/symmetry,symmetry_b,symmetry_d,ngluons,nquarks,nphotons
  double precision :: fr,fs,fc,fdc,fsc,fdsc(4)
  common/factor_n1body/fr,fs,fc,fdc,fsc,fdsc
  double precision :: x(99),mb(nexternal-1)
  double precision :: p(0:3,nexternal),plab(0:3,nexternal),pcm(0:3,nexternal),pno(0:3,nexternal)
  double precision :: reference(0:3,nexternal),born(0:3,nexternal-1),bn(0:3,nexternal-1)
  double precision :: invborn(0:3,-max_branch:nexternal-1),rad(3),bx(2),bxinv(2)
  double precision :: stot,sb,mass,tb,yb,tauinv,yinv,jac,jinv,psinv,sreal,rapidity
  double precision :: energy,momentum,phi,z,h,eta,normalization,total,expect,weight,integrand
  double precision :: nodes(64),weights(64),split(3),lo,hi,xx,tmp_xmax,tmp_norm,tmp_xi,tmp_y
  double precision :: tmp_hat,tmp_ps,tmp_jac,tmp_s,tmp_sqrt,tmp_tau,tmp_ycm,tmp_xb(2),tmp_pi(0:3)
  double precision :: radii(4),angles(4),fractions(2,4),saved_jac
  double precision :: max_replay,max_inverse,expected_xb(2),omega,expected_jac
  integer :: which,side,ir,ia,native,k,segment,cut,kind
  logical :: pass

  ebeam=[6500d0,4000d0]
  lpp=1
  stot=4d0*product(ebeam)
  nbody=.false.
  fixed_order=.true.
  nlo_ps=.false.
  symmetric_map=mode.eq.'isr_symmetric'.or.mode.eq.'isr_symmetric_fks'.or.mode.eq.'isr_symmetric_bounds'
  FKSISRMapping=2
  if(symmetric_map)FKSISRMapping=1
  if(mode.eq.'isr_automatic')FKSISRMapping=0
  evpr=.true.
  softtest=.false.
  colltest=.false.
  omx=0d0
  bound_born=0d0
  bound_res=0d0
  bound_tau=0d0
  symmetry=1d0
  symmetry_b=1d0
  symmetry_d=1d0
  ngluons=0
  nquarks=0
  nphotons=0
  xiBSVcut_used=1d0
  ifks=5
  fractions(:,1)=[.15d0,.23d0]
  fractions(:,2)=[.76d0,.95d0]
  fractions(:,3)=[.012d0,.97d0]
  fractions(:,4)=[1d0-1d-8,.85d0]
  radii=[.01d0,.4d0,.85d0,.999d0]
  angles=[.001d0,.3d0,.7d0,.999d0]
  max_replay=0d0
  max_inverse=0d0

  if(mode.eq.'isr_symmetric_bounds')then
    call check_symmetric_domains()
    return
  endif

  if(mode.eq.'isr_fks'.or.mode.eq.'isr_symmetric_fks')then
    native_mapping=.true. ! no sampling offsets in the analytic integrals
    call gauss_legendre(nodes,weights)
    do side=1,2
      jfks=side
      do which=1,2
        bx=fractions(:,which)
        call set_born()
        h=1d0-bx(side)
        do cut=1,3
          xicut_used=xicuts(cut)
          delta_used=deltacuts(cut)
          xiScut_used=.5d0
          do kind=1,2
            h=1d0-bx(side)
            if(symmetric_map.and.kind.eq.1)h=symmetric_bound(bx,.5d0,side)
            ! Split at the numerical subtraction cutoff so the quadrature
            ! sees smooth integrands on each interval.
            split=[0d0,sqrt(min(1d0,xiScut_used/h)),1d0]
            if(kind.eq.2)split=[0d0,sqrt(deltaS/2d0),1d0]
            total=0d0
            do segment=1,2
              lo=split(segment)
              hi=split(segment+1)
              if(hi.eq.lo)cycle
              do k=1,size(nodes)
                xx=(lo+hi+(hi-lo)*nodes(k))/2d0
                weight=weights(k)*(hi-lo)/2d0
                x=0d0
                x(1:3)=[xx,.5d0,.31d0]
                if(kind.eq.2)x(1:3)=[.6d0,xx,.31d0]
                call generate()
                call compute_prefactors_n1body(1d0,jac)
                z=1d0-xi
                eta=1d0-y
                if(kind.eq.1)then
                  ! F(xi)=1+xi+xi^2, no collinear singularity. z cancels
                  ! the real/soft flux ratio in the production measure.
                  integrand=fr*z*eta*(1d0+xi+xi**2)
                  if(xi.lt.xiScut_used)integrand=integrand-fs*eta
                  normalization=2d0*pi*(4d0*x(2))/(2d0*(4d0*pi)**3*(2d0*pi)**2)
                else
                  ! F(eta)=1+eta+eta^2, no soft singularity.
                  integrand=fr*z*xi*(1d0+eta+eta**2)
                  if(eta.lt.deltaS)integrand=integrand-fc*(1d0-xic(1))*xic(1)
                  normalization=h*2d0*pi*(2d0*x(1))/(2d0*(4d0*pi)**3*(2d0*pi)**2)
                  if(symmetric_map)then
                    ! Divide out the smooth angle-dependent energy range
                    ! in the test function, including its collinear limit.
                    ! The remaining plus distribution has a known integral.
                    integrand=fr*z*xi/xmax*(1d0+eta+eta**2)
                    if(eta.lt.deltaS)integrand=integrand-fc*(1d0-xic(1))*xic(1)/xmaxc(1)
                    normalization=normalization/h
                  endif
                endif
                total=total+weight*integrand/normalization
              enddo
            enddo
            if(kind.eq.1)then
              total=total+log(xicut_used)
              expect=h+h*h/2d0+log(h)
            else
              total=total+log(delta_used)
              expect=4d0+log(2d0)
            endif
            if(abs(total-expect).gt.2d-9)then
              write(*,*) 'FKS finite master mismatch',side,which,cut,kind,total,expect
              error stop 'ISR FKS finite endpoint integral'
            endif
          enddo
          if(.not.symmetric_map)call check_mixed_integral(h)
        enddo
      enddo
    enddo
    return
  endif

  do native=0,1
    native_mapping=native.eq.1
    do side=1,2
      jfks=side
      do which=1,size(fractions,2)
        bx=fractions(:,which)
        call set_born()
        do ir=1,size(radii)
          do ia=1,size(angles)
            x=0d0
            x(1:3)=[radii(ir),angles(ia),.31d0]
            call generate()
            z=1d0-xi
            phi=2d0*pi*x(3)
            if(symmetric_map)then
              omega=sqrt((2d0-xi*(1d0+(3-2*side)*y))/(2d0-xi*(1d0-(3-2*side)*y)))
              expected_xb=bx*[1d0/omega,omega]/sqrt(z)
              if(maxval(abs(xb-expected_xb)).gt.1d-13)error stop 'symmetric ISR fractions'
              if(abs(xmax-symmetric_bound(bx,y,side)).gt.5d-13)then
                write(*,*) 'ISR boundary mismatch',native,side,which,ir,ia,xmax,symmetric_bound(bx,y,side)
                error stop 'symmetric ISR physical boundary'
              endif
              if(abs(xmaxc(1)-(1d0-bx(side))).gt.1d-13) &
                   error stop 'symmetric ISR collinear boundary'
              call symmetric_reference(born,xi,y,phi,side,reference)
            else
              if(abs(xmax-(1d0-bx(side))).gt.1d-14.or. &
                   maxval(abs(xmaxc(0:2)-xmax)).gt.1d-14)error stop 'ISR endpoint is angle dependent'
              if(abs(xb(side)*z-bx(side)).gt.1d-13.or. &
                   xb(3-side).ne.bx(3-side))error stop 'ISR changed spectator fraction'
              call reference_map(born,xi,y,phi,side,reference)
            endif
            if(maxval(abs(xbc(:,0)-bx)).gt.1d-14)error stop 'ISR changed soft fractions'
            max_replay=max(max_replay,maxval(abs(reference-p))/mass)
            if(maxval(abs(reference-p))/mass.gt.5d-10)then
              write(*,*) 'ISR recoil mismatch',native,side,which,ir,ia,xi,y
              write(*,*) 'Relative differences', (reference-p)/mass
              error stop 'ISR independent recoil mismatch'
            endif
            if(maxval(abs(pc(:,3:4,0)-born(:,3:4))).gt.1d-12*mass.or. &
                 maxval(abs(pc(:,3:4,1)-born(:,3:4))).gt.1d-12*mass.or. &
                 maxval(abs(pc(:,3:4,2)-born(:,3:4))).gt.1d-12*mass) &
                 error stop 'ISR singular projection changes the hard Born'
            if(maxval(abs(pc(:,side,1)-pc(:,ifks,1)-born(:,side))).gt.1d-11*mass) &
                 error stop 'ISR collinear incoming Born momentum'
            saved_jac=jac
            ! Independent phase-space factorization, including the flux,
            ! azimuth and sampled energy/angle derivatives. FKS inserts
            ! the omitted xi and xi-range factors when assembling weights.
            expected_jac=2d0*pi*8d0*x(1)*x(2)/(2d0*(4d0*pi)**3*(2d0*pi)**2*z)
            if(native.eq.0)expected_jac=expected_jac*(1d0-1d-6)*(1d0-5d-7)
            if(abs(jac/expected_jac-1d0).gt.2d-12)error stop 'ISR factorized measure'
            call boost_n1_to_lab(p,plab,-yb)
            jinv=1d0
            psinv=1d0
            call invert_fks_radiation(rad,jinv,psinv,stot,tauinv,yinv,bxinv,plab,invborn)
            max_inverse=max(max_inverse,maxval(abs(invborn(:,1:4)-born))/mass)
            if(jinv.le.0d0.or.maxval(abs(invborn(:,1:4)-born))/mass.gt.1d-8.or. &
                 maxval(abs(bxinv-bx)).gt.1d-10.or.abs(yinv-yb).gt.1d-10) &
                 error stop 'ISR inverse Born projection'
            ! Very small 1-x loses relative precision when inferred from
            ! real energies; the momenta and fractions remain accurate.
            if(which.ne.4)then
              if(maxval(abs(rad-x(1:3))).gt.2d-8)error stop 'ISR inverse radiation coordinates'
              sreal=sb/z
              call compute_flux(sreal,sqrt(sreal),0d0,0d0,psinv,jinv)
              if(abs(jinv/saved_jac-1d0).gt.2d-8) &
                   error stop 'ISR inverse measure'
            endif
            ! The asymmetric lepton chart shares its event-projection
            ! recoil. The historical symmetric lepton chart uses a
            ! single boost along the recoil, with a different orientation.
            bn=born
            call boost_n1_to_its_cms(p,pcm,rapidity)
            pno=pcm
            call boost_born_momenta_noevpr(bn,pno,xi,y,phi,ifks,jfks,sb/z,sb)
            if(symmetric_map)then
              reference=pcm
              do k=3,4
                reference(:,k)=born(:,k)
                call boost(reference(:,k),-pcm(1:3,ifks)/(sum(pcm(0,1:2))-pcm(0,ifks)))
              enddo
              if(maxval(abs(pno-reference))/mass.gt.1d-10)error stop 'symmetric lepton chart recoil'
            else
              if(maxval(abs(pno-pcm))/mass.gt.1d-10)error stop 'ISR charts have different recoil'
            endif
          enddo
        enddo
      enddo
      ! Above-Born real thresholds in the asymmetric map restrict xi, never y. Exercise
      ! both an allowed interval and a genuinely empty physical domain.
      bx=[.2d0,.4d0]
      call set_born()
      if(.not.symmetric_map)then
      bound_born=tb*1.1d0
      bound_res=bound_born
      bound_tau=bound_born
      x=0d0
      x(1:3)=[.6d0,.99d0,.4d0]
      call generate()
      if(xi.lt.1d0-tb/bound_tau)error stop 'ISR lost real mass threshold'
      bound_born=tb/bx(side)*1.1d0
      bound_res=bound_born
      bound_tau=bound_born
      tmp_jac=1d0
      tmp_ps=1d0
      p(:,1:4)=born
      call generate_momenta_initial(-100,ifks,jfks,bx,tb,yb,0d0,sb,.3d0,p,x, &
           tmp_s,stot,tmp_sqrt,tmp_tau,tmp_ycm,tmp_xb,tmp_pi,tmp_xmax,tmp_norm, &
           tmp_xi,tmp_y,tmp_hat,tmp_ps,tmp_jac,pass)
      if(pass.or.tmp_jac.ge.0d0)error stop 'ISR accepted an empty radiation interval'
      endif
      ! Exactly soft no-event-projection recoil must not divide by xi.
      bound_born=0d0
      bound_res=0d0
      bound_tau=0d0
      bn=born
      pno=0d0
      call boost_born_momenta_noevpr(bn,pno,0d0,.2d0,.7d0,ifks,jfks,sb,sb)
      if(.not.all(ieee_is_finite(pno)).or.maxval(abs(pno(:,1:4)-born)).gt.1d-12*mass) &
           error stop 'ISR no-event-projection soft endpoint'
      ! Both exact singular projections must retain the hard Born momenta.
      softtest=.true.
      xi_fix=0d0
      call generate()
      if(maxval(abs(p(:,1:4)-born)).gt.1d-12*mass.or.any(p(:,ifks).ne.0d0)) &
           error stop 'ISR exact soft endpoint'
      softtest=.false.
      colltest=.true.
      y_fix=1d0
      call generate()
      if(maxval(abs(p(:,3:4)-born(:,3:4))).gt.1d-12*mass) &
           error stop 'ISR exact collinear endpoint'
      if(maxval(abs(p(:,side)-p(:,ifks)-born(:,side))).gt.1d-12*mass) &
           error stop 'ISR exact collinear beam projection'
      colltest=.false.
      ! Dressed-lepton sampling can resolve 1-x even when x rounds to 1.
      bx(side)=1d0
      omx(side)=1d-18
      call set_born()
      call generate()
      if(symmetric_map)then
        if(abs(xmax/(2d0*omx(side)/(1d0+y))-1d0).gt.1d-10.or.xi.le.0d0) &
             error stop 'symmetric ISR lost lepton endpoint precision'
      else
        if(xmax.ne.omx(side).or.xi.le.0d0)error stop 'ISR lost lepton endpoint precision'
      endif
      omx=0d0
    enddo
  enddo
  write(*,*) 'ISR recoil and inverse maximum relative errors',max_replay,max_inverse
contains
  subroutine check_symmetric_domains()
    double precision :: selected_angles(3),minimum
    selected_angles=[1d-4,.5d0,.9999d0]
    do native=0,1
      native_mapping=native.eq.1
      do side=1,2
        jfks=side
        bx=[.2d0,.4d0]
        call set_born()
        ! Requiring a high real invariant mass restricts both ends of y.
        ! Neither collinear endpoint is physically accessible here.
        bound_tau=.8d0
        minimum=1d0-tb/bound_tau
        do ia=1,size(selected_angles)
          x(1:3)=[.6d0,selected_angles(ia),.31d0]
          pb=born
          pbl=born
          pbe=born
          call generate_FKS_kinematics(x(1:3),nbody,1d0,1d0,stot,sb,mass,tb,yb,0d0, &
               bx,mb,jac,p,pass)
          if(.not.pass.or.jac.le.0d0.or..not.all(ieee_is_finite(p))) &
               error stop 'symmetric restricted angular domain rejected'
          if(xi.lt.minimum.or.any(xb.gt.1d0).or.abs(y).ge.1d0) &
               error stop 'symmetric real point outside restricted domain'
          if(any(jc(1:2).ge.0d0).or.any(pc(0,1,1:2).ge.0d0)) &
               error stop 'inaccessible symmetric collinear endpoints retained'
          ! Inversion must use the same restricted angular range and
          ! nonzero energy minimum as the forward generation.
          saved_jac=jac
          sreal=sb/(1d0-xi)
          call boost_n1_to_lab(p,plab,-yb)
          jinv=1d0
          psinv=1d0
          call invert_fks_radiation(rad,jinv,psinv,stot,tauinv,yinv,bxinv,plab,invborn)
          if(jinv.le.0d0.or.maxval(abs(rad-x(1:3))).gt.1d-7.or. &
               maxval(abs(invborn(:,1:4)-born))/mass.gt.1d-10) &
               error stop 'symmetric restricted-domain inverse'
          call compute_flux(sreal,sqrt(sreal),0d0,0d0,psinv,jinv)
          if(abs(jinv/saved_jac-1d0).gt.1d-6)error stop 'restricted-domain inverse measure'
        enddo
        ! No real point can satisfy a threshold above the hadronic energy.
        bound_tau=1.1d0
        call generate_slot(-100)
        if(pass.or.tmp_jac.ge.0d0)error stop 'symmetric empty angular domain accepted'
        ! A saturated emitting fraction has no finite-angle radiation
        ! support; the completely exhausted two-beam point must reject too.
        bound_tau=0d0
        bx(side)=1d0
        call set_born()
        x(1:3)=[.6d0,.5d0,.31d0]
        call generate_slot(-100)
        if(pass.or.tmp_jac.ge.0d0)error stop 'symmetric zero radiation domain accepted'
        bx=1d0
        call set_born()
        call generate_slot(-100)
        if(pass.or.tmp_jac.ge.0d0)error stop 'symmetric exhausted beams accepted'
      enddo
    enddo
  end subroutine

  subroutine generate_slot(slot)
    integer,intent(in) :: slot
    tmp_jac=1d0
    tmp_ps=1d0
    p=0d0
    p(:,1:4)=born
    call generate_momenta_initial(slot,ifks,jfks,bx,tb,yb,0d0,sb,2d0*pi*x(3),p,x, &
         tmp_s,stot,tmp_sqrt,tmp_tau,tmp_ycm,tmp_xb,tmp_pi,tmp_xmax,tmp_norm, &
         tmp_xi,tmp_y,tmp_hat,tmp_ps,tmp_jac,pass)
  end subroutine

  double precision function symmetric_bound(bx,y,side) result(bound)
    ! Locate the physical support directly from the incoming fractions;
    ! this does not reuse the production analytic quadratic solution.
    double precision,intent(in) :: bx(2),y
    integer,intent(in) :: side
    integer,parameter :: quad=selected_real_kind(30)
    real(kind=quad) :: lower,upper,trial,omega,fractions(2),qbx(2),qy
    integer :: iteration
    qbx=real(bx,quad)
    qy=real(y,quad)*(3-2*side)
    lower=0.0_quad
    upper=1.0_quad
    do iteration=1,110
      trial=(lower+upper)/2.0_quad
      omega=sqrt((2.0_quad-trial*(1.0_quad+qy))/(2.0_quad-trial*(1.0_quad-qy)))
      fractions=qbx*[1.0_quad/omega,omega]/sqrt(1.0_quad-trial)
      if(maxval(fractions).gt.1.0_quad)then
        upper=trial
      else
        lower=trial
      endif
    enddo
    bound=real((lower+upper)/2.0_quad,8)
  end function

  subroutine symmetric_reference(born,xi,y,phi,side,out)
    ! The historical symmetric map is a transverse boost of the hard
    ! Born system. Build radiation in the real CM and boost it into the
    ! frame where the hard system has no longitudinal momentum.
    double precision,intent(in) :: born(0:3,4),xi,y,phi
    integer,intent(in) :: side
    double precision,intent(out) :: out(0:3,5)
    double precision :: mass,energy,cosine,sine,beta(3),kt,hard_energy
    integer :: i
    mass=2d0*born(0,1)
    energy=mass/(2d0*sqrt(1d0-xi))
    cosine=(3-2*side)*y
    sine=sqrt((1d0-cosine)*(1d0+cosine))
    kt=xi*energy*sine
    hard_energy=sqrt(mass**2+kt**2)
    beta=-kt/hard_energy*[cos(phi),sin(phi),0d0]
    out(:,1:4)=born
    do i=3,4
      call boost(out(:,i),beta)
    enddo
    out(:,1)=[energy,0d0,0d0,energy]
    out(:,2)=[energy,0d0,0d0,-energy]
    out(:,5)=xi*energy*[1d0,sine*cos(phi),sine*sin(phi),cosine]
    beta=[0d0,0d0,xi*cosine/(2d0-xi)]
    call boost(out(:,1),beta)
    call boost(out(:,2),beta)
    call boost(out(:,5),beta)
  end subroutine

  subroutine check_mixed_integral(h)
    ! Both endpoint distributions act on F(xi,eta)=(1+xi+xi^2)*(1+eta+eta^2).
    ! In particular, test the finite mixed logarithm in f_sc.
    double precision,intent(in) :: h
    double precision :: sx(3),sy(3),tx,ty,wx,wy,result,term,norm,expected
    double precision :: gx,gy
    integer :: ix,iy,kx,ky
    sx=[0d0,sqrt(min(1d0,xiScut_used/h)),1d0]
    sy=[0d0,sqrt(deltaS/2d0),1d0]
    norm=2d0*pi/(2d0*(4d0*pi)**3*(2d0*pi)**2)
    result=0d0
    do ix=1,2
      if(sx(ix+1).eq.sx(ix))cycle
      do iy=1,2
        do kx=1,size(nodes)
          tx=(sx(ix+1)+sx(ix)+(sx(ix+1)-sx(ix))*nodes(kx))/2d0
          wx=weights(kx)*(sx(ix+1)-sx(ix))/2d0
          do ky=1,size(nodes)
            ty=(sy(iy+1)+sy(iy)+(sy(iy+1)-sy(iy))*nodes(ky))/2d0
            wy=weights(ky)*(sy(iy+1)-sy(iy))/2d0
            x=0d0
            x(1:3)=[tx,ty,.31d0]
            call generate()
            call compute_prefactors_n1body(1d0,jac)
            gx=1d0+xi+xi**2
            gy=1d0+(1d0-y)+(1d0-y)**2
            term=fr*(1d0-xi)*gx*gy
            if(xi.lt.xiScut_used)term=term-fs*gy
            if(1d0-y.lt.deltaS)then
              term=term-fc*(1d0-xic(1))*gx
              if(xic(1).lt.xiScut_used)term=term+fsc
            endif
            result=result+wx*wy*term/norm
          enddo
        enddo
      enddo
    enddo
    expected=(h+h*h/2d0+log(h/xicut_used))*(4d0+log(2d0/delta_used))
    if(abs(result-expected).gt.2d-8)then
      write(*,*) 'FKS mixed master mismatch',side,which,cut,result,expected
      error stop 'ISR FKS finite soft-collinear integral'
    endif
  end subroutine

  subroutine set_born()
    tb=product(bx)
    yb=log(bx(1)/bx(2))/2d0
    sb=tb*stot
    mass=sqrt(sb)
    pmass=0d0
    pmass(3:4)=[.1d0,.2d0]*mass
    mb=pmass(1:4)
    energy=(sb+mb(3)**2-mb(4)**2)/(2d0*mass)
    momentum=sqrt(energy**2-mb(3)**2)
    born(:,1)=[mass/2d0,0d0,0d0,mass/2d0]
    born(:,2)=[mass/2d0,0d0,0d0,-mass/2d0]
    born(:,3)=[energy,momentum*.3d0,momentum*.4d0,momentum*sqrt(.75d0)]
    born(:,4)=[mass-energy,-born(1:3,3)]
  end subroutine

  subroutine generate()
    pb=born
    pbl=born
    pbe=born
    call generate_FKS_kinematics(x(1:3),nbody,1d0,1d0,stot,sb,mass,tb,yb,0d0, &
         bx,mb,jac,p,pass)
    if(jac.le.0d0.or..not.pass.or..not.all(ieee_is_finite(p)))error stop 'ISR generation failed'
    if(any(jc(0:2).le.0d0))error stop 'ISR counterevents missing'
  end subroutine

  subroutine reference_map(born,xi,y,phi,side,out)
    ! Replay the boost/rotation construction, independently of the
    ! production light-cone equations. No shower kernels are tested.
    double precision,intent(in) :: born(0:3,4),xi,y,phi
    integer,intent(in) :: side
    double precision,intent(out) :: out(0:3,5)
    double precision :: z,mass,q2,er,kt,kz,sign,ph,theta,beta(3)
    double precision :: mo(0:3),rec(0:3),sis(0:3),sumnew(0:3),v(0:3),test(0:3)
    integer :: i
    sign=dble(3-2*side)
    z=1d0-xi
    mass=2d0*born(0,1)
    q2=mass**2/z*xi*(1d0-y)/2d0
    er=(mass**2+q2)/(2d0*mass)
    kt=sqrt(max(0d0,q2-z*(mass**2+q2)*q2/mass**2))*mass**2/(z*(mass**2+q2))
    kz=sign*mass/2d0*((mass**2-q2)/(z*(mass**2+q2))+q2/mass**2)
    mo=[sqrt(kt**2+kz**2),kt,0d0,kz]
    rec=[er,0d0,0d0,-sign*er]
    sumnew=mo+rec
    sis=sumnew-[mass,0d0,0d0,0d0]
    beta=-sumnew(1:3)/sumnew(0)
    test=mo
    call boost(test,beta)
    theta=atan2(test(1),test(3))
    ph=phi
    if(side.eq.2)then
      theta=theta+pi
    endif
    do i=1,5
      if(i.le.4)then
        v=born(:,i)
        if(i.eq.1)v=mo
        if(i.eq.2)v=rec
        if(i.ge.3)call rotation(v,0d0,-ph)
      else
        v=sis
      endif
      call boost(v,beta)
      call rotation(v,-theta,ph)
      call boost(v,[0d0,0d0,sign*(1d0-z)/(1d0+z)])
      out(:,i)=v
    enddo
    if(side.eq.2)then
      v=out(:,1)
      out(:,1)=out(:,2)
      out(:,2)=v
    endif
  end subroutine

  subroutine boost(p,beta)
    double precision,intent(inout) :: p(0:3)
    double precision,intent(in) :: beta(3)
    double precision :: gamma,bp,out(0:3)
    gamma=1d0/sqrt(1d0-sum(beta**2))
    bp=sum(beta*p(1:3))
    out(0)=gamma*(p(0)+bp)
    out(1:3)=p(1:3)+(gamma*p(0)+gamma**2/(gamma+1d0)*bp)*beta
    p=out
  end subroutine

  subroutine rotation(p,theta,phi)
    double precision,intent(inout) :: p(0:3)
    double precision,intent(in) :: theta,phi
    double precision :: v(3)
    v=[cos(theta)*p(1)+sin(theta)*p(3),p(2),-sin(theta)*p(1)+cos(theta)*p(3)]
    p(1:3)=[cos(phi)*v(1)-sin(phi)*v(2),sin(phi)*v(1)+cos(phi)*v(2),v(3)]
  end subroutine

  subroutine gauss_legendre(nodes,weights)
    double precision,intent(out) :: nodes(:),weights(:)
    double precision :: root,previous,p1,p2,p3,derivative
    integer :: i,j,n
    n=size(nodes)
    do i=1,(n+1)/2
      root=cos(pi*(i-.25d0)/(n+.5d0))
      do
        p1=1d0
        p2=0d0
        do j=1,n
          p3=p2
          p2=p1
          p1=((2*j-1)*root*p2-(j-1)*p3)/j
        enddo
        derivative=n*(root*p1-p2)/(root**2-1d0)
        previous=root
        root=previous-p1/derivative
        if(abs(root-previous).lt.2d-15)exit
      enddo
      nodes(i)=-root
      nodes(n+1-i)=root
      weights(i)=2d0/((1d0-root**2)*derivative**2)
      weights(n+1-i)=weights(i)
    enddo
  end subroutine
end subroutine
