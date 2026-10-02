program check_resonance_recoil
  use fks_phase_space_data, only: spin => xij_aor
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mc_native_context, only: native_mapping
  use fks_phase_space, only: generate_FKS_kinematics,generate_native_momenta
  use fks_phase_space_helpers, only: getangles
  implicit none
  include 'nexternal.inc'
  double precision, parameter :: pi=3.1415926535897932d0, mr=200d0
  double precision :: born(0:3,nexternal),p(0:3,nexternal),again(0:3,nexternal)
  double precision :: projected(0:3,nexternal-1),q(0:3),qmap(0:3),m2,ma,mj
  double precision :: masses(nexternal),rnd(3),inv(3),jac,ps,jinv,psinv,j2,ps2
  double precision :: xi,y,xihat,xmax,xnorm,phat(0:3),phat0(0:3),r(0:3),r0(0:3)
  double precision :: xs(6)=[1d-4,0.04d0,0.4d0,0.8d0,0.96d0,0.999d0]
  double precision :: ys(4)=[0.02d0,0.45d0,0.85d0,0.995d0]
  double precision :: boosts(3)=[0d0,1d0,7d0],qrest(0:3),pj(0:3),pr(0:3)
  double precision :: cth,sth,cph,sph,th,phi,err,mom,t,volume,reference,lam
  double precision :: gx(128),gw(128),sx(512),sw(512),upper
  double precision :: soft_scale,coll_scale,angular_scale,mismatch_log,scale0(4),scale1(4)
  double precision :: total(0:3),testboost(0:3),totalb(0:3),qb(0:3),pjb(0:3),phatb(0:3)
  complex*16 :: spin_save,expected,z
  logical :: mask(nexternal),aunts(nexternal),pass,softtest,colltest
  common/sctests/softtest,colltest
  double precision :: xifix,yfix
  common/cxiyfix/xifix,yfix
  character(len=32) :: mode
  integer :: native,kind,masscase,ibst,perm,ir,iy,ifks,jfks,isign,i,j,k,ib,nminus,nplus,ic

  call get_command_argument(1,mode)
  softtest=.false.
  colltest=.false.
  nminus=0
  nplus=0
  if(mode.eq.'soft_scales')call gauss(512,sx,sw)
  if(mode.eq.'volume')then
    native_mapping=.true.
    call gauss(128,gx,gw)
    call gauss(512,sx,sw)
    do masscase=0,1
      mj=50d0*masscase
      do kind=1,3
        ma=80d0
        if(kind.eq.1)ma=0d0
        call fixture(mj,ma,kind,1d0,1)
        volume=0d0
        do ir=1,size(gx)
          do iy=1,size(gx)
            rnd=[gx(ir),gx(iy),0.31d0]
            call forward(-100,p,jac,ps)
            if(.not.pass)error stop 'phase space rejected in volume integral'
            volume=volume+gw(ir)*gw(iy)*jac*ps*xi*xnorm
          enddo
        enddo
        ! Lorentz-invariant recursion Phi3 = integral dt/(2*pi)
        ! Phi2(M;sqrt(t),ma) Phi2(sqrt(t);mj,0), divided by Phi2(M;mj,ma).
        reference=0d0
        upper=(mr-ma)**2
        lam=sqrt((mr**2-(mj+ma)**2)*(mr**2-(mj-ma)**2))
        do i=1,size(sx)
          t=mj**2+(upper-mj**2)*sx(i)
          reference=reference+sw(i)*sqrt(max(0d0, &
              (mr**2-t-ma**2)**2-4d0*t*ma**2))*(t-mj**2)/t
        enddo
        reference=reference*(upper-mj**2)/(16d0*pi**2*lam)
        if(abs(volume/reference-1d0).gt.2d-4)then
          write(*,*)'volume mismatch',mj,ma,volume,reference,volume/reference
          error stop 'local phase-space measure'
        endif
      enddo
    enddo
  elseif(mode.eq.'invalid')then
    native_mapping=.false.
    mj=0d0
    call fixture(mj,80d0,2,1d0,1)
    rnd=[0.3d0,0.4d0,0.5d0]
    mask(1)=.true.
    call forward(-100,p,jac,ps)
    if(pass.or.jac.ge.0d0)error stop 'incoming recoil accepted'
    mask(1)=.false.
    mask(ifks)=.false.
    call forward(-100,p,jac,ps)
    if(pass.or.jac.ge.0d0)error stop 'missing radiation accepted'
    mask=.false.
    mask(ifks)=.true.
    mask(jfks)=.true.
    call forward(-100,p,jac,ps)
    if(pass.or.jac.ge.0d0)error stop 'missing aunt accepted'
  else
    do native=0,1
      native_mapping=native.eq.1
      do masscase=0,1
        mj=50d0*masscase
        if(mode.eq.'spin'.and.masscase.ne.0)cycle
        do kind=1,3
          ma=80d0
          if(kind.eq.1)ma=0d0
          do ibst=1,size(boosts)
            do perm=1,2
              call fixture(mj,ma,kind,boosts(ibst),perm)
              do ir=1,size(xs)
                do iy=1,size(ys)
                  rnd=[xs(ir),ys(iy),0.317d0]
                  if(mode.eq.'spin')then
                    colltest=.true.
                    yfix=1d0-1d-7
                  endif
                  call forward(-100,p,jac,ps)
                  if(.not.pass)then
                    write(*,*)'rejected',native,kind,mj,ibst,perm,rnd
                    error stop 'valid local map rejected'
                  endif
                  call invariants(p)
                  if(isign.eq.1)nplus=nplus+1
                  if(isign.eq.-1)nminus=nminus+1
                  spin_save=spin
                  if(mode.eq.'production')then
                    call production_roundtrip()
                  elseif(mode.eq.'inverse')then
                    jinv=1d0
                    psinv=1d0
                    call invert_momenta_resonance_final(p,mask,ifks,jfks,mj, &
                         projected,inv,jinv,psinv,qmap,m2,pass)
                    if(.not.pass)then
                      write(*,*)'inverse rejected',native,kind,mj,ibst,perm,rnd,jinv
                      error stop 'local inverse rejected'
                    endif
                    if(maxval(abs(inv-rnd)).gt.2d-7)then
                      write(*,*)'coordinate mismatch',native,kind,mj,ibst,perm,rnd,inv
                      error stop 'local inverse coordinates'
                    endif
                    ib=0
                    do i=1,nexternal
                      if(i.eq.ifks)cycle
                      ib=ib+1
                      if(maxval(abs(projected(:,ib)-born(:,i))).gt.2d-7*mr) &
                           error stop 'local Born projection'
                      if(.not.mask(i).and.any(projected(:,ib).ne.born(:,i))) &
                           error stop 'inverse moved non-descendant'
                    enddo
                    if(abs(jinv/jac-1d0).gt.2d-6.or.abs(psinv/ps-1d0).gt.2d-6)then
                      write(*,*)'measure mismatch',native,kind,mj,ibst,perm,rnd,jinv/jac,psinv/ps
                      error stop 'forward/inverse measures'
                    endif
                    if(mj.eq.0d0.and.abs(spin-spin_save).gt.1d-8) &
                         error stop 'forward/inverse helicity phase'
                  elseif(mode.eq.'limits')then
                    call forward(0,again,j2,ps2)
                    if(.not.pass)error stop 'soft counterevent rejected'
                    if(maxval(abs(again-born)).gt.1d-9*mr)error stop 'soft Born momenta'
                    if(abs(ps2/(mr**2/(4d0*pi)**3)-1d0).gt.1d-8.and.native.eq.0) &
                         error stop 'soft measure'
                    if(abs(mdot(phat,phat)).gt.1d-8*mr**2)error stop 'soft direction not null'
                    if(abs(2d0*mdot(phat,q)/mr**2-1d0).gt.1d-10) &
                         error stop 'soft direction normalization'
                    if(mj.eq.0d0)then
                      call forward(1,again,j2,ps2)
                      if(.not.pass)error stop 'collinear counterevent rejected'
                      call invariants(again)
                      if(maxval(abs(again(:,ifks)+again(:,jfks)-born(:,jfks))).gt.1d-9*mr) &
                           error stop 'collinear parent'
                      do i=1,nexternal
                        if(aunts(i).and.maxval(abs(again(:,i)-born(:,i))).gt.1d-9*mr) &
                             error stop 'collinear aunt'
                      enddo
                      call forward(2,again,j2,ps2)
                      if(.not.pass.or.maxval(abs(again-born)).gt.1d-9*mr) &
                           error stop 'soft collinear Born momenta'
                    endif
                  elseif(mode.eq.'soft_scales')then
                    call forward(0,again,j2,ps2)
                    if(.not.pass)error stop 'soft scale fixture'
                    total=sum(born(:,1:nincoming),dim=2)
                    call resonance_subtraction_scales(total,q,born(:,jfks),phat, &
                         soft_scale,coll_scale,angular_scale,mismatch_log,pass)
                    if(.not.pass)error stop 'soft scale conversion'
                    if(abs(soft_scale*2d0*phat(0)/total(0)-1d0).gt.1d-10) &
                         error stop 'soft scale normalization'
                    err=mdot(born(:,jfks),q)/(mr*born(0,jfks))
                    if(abs(coll_scale/(err*total(0)/mr)-1d0).gt.1d-10.or. &
                         abs(angular_scale*err**2-1d0).gt.1d-10)error stop 'collinear scales'
                    ! Independently integrate the dampers in the soft energy.
                    reference=0d0
                    do i=1,size(sx)
                      t=sx(i)/(1d0-sx(i))
                      reference=reference+sw(i)*(exp(-soft_scale*t)-exp(-coll_scale*t))/ &
                           (sx(i)*(1d0-sx(i)))
                    enddo
                    if(abs(reference-mismatch_log).gt.1d-9)error stop 'soft radial integral'
                    if(mj.eq.0d0)then
                      reference=qterm(born(0,jfks),total(0)**2,0.13d0,0.4d0)
                      t=qterm(mdot(born(:,jfks),q)/mr,mr**2, &
                           0.13d0*coll_scale,0.4d0*angular_scale)
                      if(abs(t-reference).gt.1d-9)error stop 'integrated collinear conversion'
                      ! Its collinear soft mismatch vanishes before integration.
                      r=born(:,jfks)*mr**2/(2d0*mdot(born(:,jfks),q))
                      call resonance_subtraction_scales(total,q,born(:,jfks),r, &
                           scale1(1),scale1(2),scale1(3),scale1(4),pass)
                      if(.not.pass.or.abs(scale1(4)).gt.1d-12)error stop 'collinear soft mismatch'
                    endif
                    scale0=[soft_scale,coll_scale,angular_scale,mismatch_log]
                    testboost=[sqrt(1d0+0.2d0**2+0.3d0**2),0.2d0,-0.3d0,0d0]
                    call boostx(total,testboost,totalb)
                    call boostx(q,testboost,qb)
                    call boostx(born(:,jfks),testboost,pjb)
                    call boostx(phat,testboost,phatb)
                    call resonance_subtraction_scales(totalb,qb,pjb,phatb, &
                         scale1(1),scale1(2),scale1(3),scale1(4),pass)
                    if(.not.pass.or.maxval(abs(scale1-scale0)).gt.1d-9) &
                         error stop 'soft scales not covariant'
                  elseif(mode.eq.'spin')then
                    ! Independently read the transverse relative direction
                    ! from a finite, nearly collinear physical pair.
                    pj=p(:,ifks)+p(:,jfks)
                    call getangles(pj,th,cth,sth,phi,cph,sph)
                    call trp_rotate_invar(p(:,ifks),pr,cth,sth,cph,sph)
                    z=dcmplx(cph,sph)*dcmplx(pr(1),pr(2))
                    expected=-z*z/(pr(1)**2+pr(2)**2)
                    if(abs(spin_save-expected).gt.8d-4)then
                      write(*,*)'spin mismatch',native,kind,ibst,rnd,spin_save,expected
                      error stop 'boosted spin phase'
                    endif
                    colltest=.false.
                  elseif(mode.ne.'invariants')then
                    error stop 'unknown check mode'
                  endif
                enddo
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo
    if(mode.ne.'spin'.and.(nminus.eq.0.or.nplus.eq.0))error stop 'massive branch coverage'
  endif
  write(*,*)'PASS '//trim(mode)
contains
  subroutine production_roundtrip()
  use fks_phase_space_data,only: resq => resonance_momentum,resmass2 => resonance_mass2, &
       active => resonance_recoil,members => resonance_members,nocnt => nocntevents,pb => p_born, &
       pbl => p_born_l,pbe => p_born_ev,branch => isolsign,pc => p1_cnt,&
       jc => jac_cnt,bound_born => tau_Born_lower_bound,bound_res => tau_lower_bound_resonance, &
       bound_tau => tau_lower_bound
    use fks_phase_space_helpers, only: boost_n1_to_lab
    implicit none
    include 'genps.inc'
    include 'run.inc'
  logical :: nbody,evpr
    common/cnbody/nbody
    common/to_use_evpr/evpr
  double precision :: pmass(nexternal)
    common/to_mass/pmass
  integer :: ifks_active,jfks_active,config
    common/fks_indices/ifks_active,jfks_active
    common/to_mconfigs/config
  double precision :: jc_save(-2:2)
  double precision :: omx(2)
    common/to_ee_omx1/omx
    double precision :: xgen(99),mb(nexternal-1), &
         out(0:3,nexternal),lab(0:3,nexternal),outlab(0:3,nexternal),outcms(0:3,nexternal), &
         born_save(0:3,nexternal-1),sqrts,s,stot,taub,yb,yhat,xb(2),j0,ps0,jout,jnew,flux,jexpected
    double precision :: shower(0:3,nexternal),kn,knbar,kn0,smass2,born_energy,shower_total(0:3)
    integer :: k,b
    logical :: valid
    active=.true.
    members=mask
    ifks_active=ifks
    jfks_active=jfks
    pmass=masses
    sqrts=sum(born(0,1:nincoming))
    s=sqrts**2
    if(nincoming.eq.1)pmass(1)=sqrts
    b=0
    do k=1,nexternal
      if(k.eq.ifks)cycle
      b=b+1
      pb(:,b)=born(:,k)
      mb(b)=pmass(k)
    enddo
    born_save=pb
    pbl=pb
    pbe=pb
    ebeam=6500d0
    lpp=1
    stot=4d0*product(ebeam)
    if(nincoming.eq.1)stot=s
    taub=s/stot
    yb=0d0
    yhat=0d0
    xb=sqrt(taub)
    nbody=.false.
    evpr=.true.
    bound_born=0d0
    bound_res=0d0
    bound_tau=0d0
    omx=0d0
    xgen=0d0
    xgen(1:3)=rnd
    j0=7d0
    ps0=3d0
    call generate_FKS_kinematics(xgen(1:3),nbody,j0,ps0,stot,s,sqrts,taub,yb,yhat, &
         xb,mb,jout,out,valid)
    if(.not.valid.or.jout.le.0d0)error stop 'production map rejected'
    if(maxval(abs(out-p)).gt.2d-7*mr)error stop 'production map differs from local map'
    flux=1d0/(2d0*s)
    if(nincoming.eq.1)flux=1d0/(2d0*sqrts)
    flux=flux/(2d0*acos(-1d0))**(3*(nexternal-nincoming)-7)
    jexpected=7d0*3d0*jac*ps*flux
    if(abs(jout/jexpected-1d0).gt.2d-7)error stop 'local map used resonance flux'
    if(maxval(abs(resq-q)).gt.2d-8*mr)error stop 'production resonance context'
    call resonance_shower_frame(out,ifks,jfks,shower,kn,knbar,kn0,smass2)
    born_energy=(mr**2+mj**2-ma**2)/(2d0*mr)
    if(abs(knbar-sqrt(born_energy**2-mj**2)).gt.2d-8*mr)error stop 'shower Born emitter'
    if(abs(smass2/mr**2-1d0).gt.2d-8)error stop 'shower resonance scale'
    shower_total=sum(shower(:,nincoming+1:nexternal),dim=2)
    if(maxval(abs(shower_total-[mr,0d0,0d0,0d0])).gt.2d-8*mr)error stop 'shower local total'
    if(.not.nocnt)then
      if(maxval(abs(pc(:,:,0)-born)).gt.2d-8*mr)error stop 'production soft Born'
      if(maxval(abs(pb-born_save)).gt.2d-8*mr)error stop 'production Born changed'
    endif
    jc_save=jc
    if(nincoming.eq.2.and.native_mapping)then
      ! The full native evaluator must recover the same local Born and
      ! counter/real measures at a fixed physical lab event.
      call boost_n1_to_lab(out,lab,-yb)
      pb=-31d0
      pbl=-32d0
      pbe=-33d0
      call generate_native_momenta(lab,out,outlab,outcms,jnew,valid)
      if(.not.valid)error stop 'native local projection rejected'
      if(maxval(abs(pb-born_save)).gt.2d-6*mr)error stop 'native local Born differs'
      if(abs(jnew*21d0/jout-1d0).gt.2d-5)error stop 'native local real measure'
      do k=-2,2
        if(jc_save(k).gt.0d0)then
          if(jc(k).le.0d0.or.abs(jc(k)*21d0/jc_save(k)-1d0).gt.2d-5) &
               error stop 'native local counterevent measure'
        endif
      enddo
    endif
    active=.false.
  end subroutine

  double precision function qterm(energy,s,xicut,delta)
    double precision, intent(in) :: energy,s,xicut,delta
    double precision, parameter :: c=4d0/3d0,gamma=2d0,gammap=4.5d0,qes2=121d0**2
    qterm=gammap-log(s*delta/(2d0*qes2))*(gamma-2d0*c*log(2d0*energy/(xicut*sqrt(s)))) &
         +2d0*c*(log(2d0*energy/sqrt(s))**2-log(xicut)**2)-2d0*gamma*log(2d0*energy/sqrt(s))
  end function

  double precision function mdot(a,b)
    double precision, intent(in) :: a(0:3),b(0:3)
    mdot=a(0)*b(0)-sum(a(1:3)*b(1:3))
  end function

  subroutine fixture(mj,ma,kind,b,perm)
    double precision, intent(in) :: mj,ma,b
    integer, intent(in) :: kind,perm
    double precision :: ej,u,rest(0:3,nexternal),aunt(0:3),daughter(0:3),tmp(0:3),energy
    integer :: j0,a1,a2,outside,i
    j0=nincoming+1
    a1=j0+1
    a2=j0+2
    outside=j0+3
    ifks=nexternal
    jfks=j0
    mask=.false.
    aunts=.false.
    masses=0d0
    rest=0d0
    born=0d0
    ej=(mr**2+mj**2-ma**2)/(2d0*mr)
    u=sqrt(ej**2-mj**2)
    rest(:,j0)=[ej,0.3d0*u,0.4d0*u,sqrt(0.75d0)*u]
    aunt=[mr-ej,-rest(1:3,j0)]
    if(kind.eq.3)then
      daughter=[ma/2d0,0d0,ma/2d0,0d0]
      call boostx(daughter,aunt,rest(:,a1))
      daughter(1:3)=-daughter(1:3)
      call boostx(daughter,aunt,rest(:,a2))
      mask(a2)=.true.
      aunts(a2)=.true.
    else
      rest(:,a1)=aunt
      masses(a1)=ma
    endif
    masses(j0)=mj
    mask(j0)=.true.
    mask(a1)=.true.
    aunts(a1)=.true.
    q(1:3)=b*[90d0,-60d0,110d0]
    q(0)=sqrt(mr**2+sum(q(1:3)**2))
    do i=nincoming+1,nexternal-1
      if(mask(i))call boostx(rest(:,i),q,born(:,i))
    enddo
    if(kind.ne.3)born(:,a2)=[40d0,0d0,40d0,0d0]
    born(1:3,outside)=-q(1:3)-born(1:3,a2)*merge(0d0,1d0,kind.eq.3)
    born(0,outside)=sqrt(sum(born(1:3,outside)**2))
    energy=sum(born(0,nincoming+1:nexternal))
    if(nincoming.eq.2)then
      born(:,1)=[energy/2d0,0d0,0d0,energy/2d0]
      born(:,2)=[energy/2d0,0d0,0d0,-energy/2d0]
    else
      born(:,1)=[energy,0d0,0d0,0d0]
    endif
    mask(ifks)=.true.
    if(perm.eq.2)then
      tmp=born(:,ifks)
      born(:,ifks)=born(:,jfks)
      born(:,jfks)=tmp
      masses(ifks)=mj
      masses(jfks)=0d0
      i=ifks
      ifks=jfks
      jfks=i
    endif
  end subroutine

  subroutine forward(ic,p,jac,ps)
    integer, intent(in) :: ic
    double precision, intent(out) :: p(0:3,nexternal),jac,ps
    p=born
    jac=2d0*pi
    ps=1d0
    call generate_momenta_resonance_final(ic,isign,ifks,jfks,mj,mask, &
         rnd(1:2),2d0*pi*rnd(3),p,xmax,xnorm,xi,y,xihat,phat,jac,ps,qmap,m2,pass)
  end subroutine

  subroutine invariants(p)
    double precision, intent(in) :: p(0:3,nexternal)
    double precision :: total(0:3),res(0:3),aunt(0:3),auntborn(0:3),tol
    integer :: i,j
    if(.not.all(ieee_is_finite(p)).or..not.ieee_is_finite(jac*ps))error stop 'nonfinite local map'
    total=sum(p(:,1:nincoming),dim=2)-sum(p(:,nincoming+1:nexternal),dim=2)
    tol=2d-10*maxval(born(0,:))
    if(maxval(abs(total)).gt.tol)error stop 'four momentum conservation'
    res=0d0
    aunt=0d0
    auntborn=0d0
    do i=1,nexternal
      if(mask(i))res=res+p(:,i)
      if(.not.mask(i).and.any(p(:,i).ne.born(:,i)))error stop 'non-descendant moved'
      if(i.le.nincoming)cycle
      if(abs(mdot(p(:,i),p(:,i))-masses(i)**2).gt.tol*max(1d0,p(0,i))) &
           error stop 'external mass shell'
      if(aunts(i))then
        aunt=aunt+p(:,i)
        auntborn=auntborn+born(:,i)
        do j=i+1,nexternal
          if(aunts(j).and.abs(mdot(p(:,i),p(:,j))-mdot(born(:,i),born(:,j))).gt.tol*mr) &
               error stop 'aunt internal invariant'
        enddo
      endif
    enddo
    if(maxval(abs(res-q)).gt.tol)error stop 'resonance four momentum'
    if(abs(mdot(res,res)-mr**2).gt.tol*q(0))error stop 'resonance mass'
    if(abs(mdot(aunt,aunt)-mdot(auntborn,auntborn)).gt.tol*q(0))error stop 'aunt mass'
  end subroutine

  subroutine gauss(n,x,w)
    integer, intent(in) :: n
    double precision, intent(out) :: x(n),w(n)
    double precision :: z,zold,p1,p2,p3,pp
    integer :: i,j
    do i=1,(n+1)/2
      z=cos(pi*(i-0.25d0)/(n+0.5d0))
      do
        p1=1d0
        p2=0d0
        do j=1,n
          p3=p2
          p2=p1
          p1=((2*j-1)*z*p2-(j-1)*p3)/j
        enddo
        pp=n*(z*p1-p2)/(z*z-1d0)
        zold=z
        z=zold-p1/pp
        if(abs(z-zold).lt.2d-15)exit
      enddo
      x(i)=(1d0-z)/2d0
      x(n+1-i)=(1d0+z)/2d0
      w(i)=1d0/((1d0-z*z)*pp*pp)
      w(n+1-i)=w(i)
    enddo
  end subroutine
end program
