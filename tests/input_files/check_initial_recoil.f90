program check_initial_recoil
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mc_native_context, only: native_mapping
  implicit none
  include 'nexternal.inc'
  double precision, parameter :: pi=3.1415926535897932d0
  double precision :: born(0:3,nexternal),p(0:3,nexternal),again(0:3,nexternal)
  double precision :: projected(0:3,nexternal-1),q(0:3),qmap(0:3),m2,mj,xbar,xback
  double precision :: rnd(3),inv(3),jac,ps,jinv,psinv,j2,ps2,ratio,ratio0
  double precision :: xi,y,xihat,xmax,xnorm,phat(0:3),total(0:3),residual(0:3)
  double precision :: xs(5)=[0.002d0,0.1d0,0.55d0,0.85d0,0.98d0]
  double precision :: ys(4)=[0.03d0,0.35d0,0.75d0,0.97d0]
  double precision :: fractions(3)=[0.1d0,0.6d0,0.97d0]
  double precision :: gx(160),gw(160),volume,reference,r,ss,cs,angs,mlog,t
  double precision :: xifix,yfix
  common/cxiyfix/xifix,yfix
  complex*16 :: spin,spinsave
  common/cxij_aor/spin
  logical :: pass,softtest,colltest
  common/sctests/softtest,colltest
  character(len=32) :: mode
  integer :: native,masscase,beam,ib,perm,ir,iy,ifks,jfks,isign,i,j,nminus

  call get_command_argument(1,mode)
  softtest=.false.
  colltest=.false.
  nminus=0
  if(mode.eq.'corners')then
    native_mapping=.false.
    do beam=1,2
      do masscase=0,1
        mj=150d0*masscase
        do ib=1,2
          xbar=1d-6
          if(ib.eq.2)xbar=1d0-1d-8
          do perm=1,2
            call fixture(perm)
            do ir=1,3
              if(ir.eq.1)rnd=[0.2d0,0.3d0,0.137d0]
              if(ir.eq.2)rnd=[0.8d0,0.9d0,0.137d0]
              if(ir.eq.3)rnd=[0.99d0,0.999d0,0.137d0]
              call forward(-100,p,jac,ps)
              if(.not.pass)error stop 'FI beam corner map rejected'
              call invariants(p)
              ratio0=ratio
              jinv=1d0
              psinv=1d0
              call invert_momenta_initial_recoil(p,beam,xbar*ratio0,ifks,jfks,mj, &
                   projected,inv,jinv,psinv,qmap,m2,xback,pass)
              if(.not.pass.or.maxval(abs(inv-rnd)).gt.2d-5.or.abs(xback/xbar-1d0).gt.2d-6)then
                write(*,*)'beam corner inverse',mj,beam,xbar,rnd,inv,xback
                error stop 'FI beam corner inverse coordinates'
              endif
            enddo
          enddo
        enddo
      enddo
    enddo
  elseif(mode.eq.'minimal')then
    native_mapping=.true.
    mj=1000d0
    xbar=0.35d0
    do beam=1,2
      do perm=1,2
        call fixture(perm)
        rnd=[0.4d0,0.6d0,0.317d0]
        call forward(-100,p,jac,ps)
        if(.not.pass)error stop 'minimal FI map rejected'
        call invariants(p)
        call production_roundtrip()
      enddo
    enddo
  elseif(mode.eq.'volume')then
    call gauss(size(gx),gx,gw)
    native_mapping=.true.
    do masscase=0,1
      mj=80d0*masscase
      do beam=1,2
        xbar=0.35d0
        call fixture(1)
        volume=0d0
        do ir=1,size(gx)
          do iy=1,size(gx)
            ! Resolve the antiparallel daughter angle quadratically,
            ! including the corner where the reservoir is exhausted.
            rnd=[gx(ir),1d0-gx(iy)**2,0.31d0]
            call forward(-100,p,jac,ps)
            if(.not.pass)error stop 'volume point rejected'
            volume=volume+gw(ir)*gw(iy)*2d0*gx(iy)*jac*ps*xi*xnorm
          enddo
        enddo
        ! At fixed Born spectators, integrating over the incoming fraction
        ! gives dPhi(k)/(1-P.k/(P.p)). In the reservoir rest frame the
        ! independent integration variables have 0<E<(M^2-m^2)/(2M)
        ! and -1<cos(theta)<1. Its exact integral is the expression below.
        r=mj**2/m2
        reference=1d0-r
        if(r.gt.0d0)reference=reference+r*log(r)
        reference=reference*m2/(16d0*pi**2)
        if(abs(volume/reference-1d0).gt.5d-4)then
          write(*,*)'volume',beam,mj,volume,reference,volume/reference
          error stop 'FI hadronic measure volume'
        endif
      enddo
    enddo
  elseif(mode.eq.'invalid')then
    native_mapping=.false.
    mj=0d0
    beam=1
    xbar=0.4d0
    call fixture(1)
    rnd=[0.3d0,0.4d0,0.5d0]
    beam=3
    call forward(-100,p,jac,ps)
    if(pass.or.jac.ge.0d0)error stop 'final leg accepted as incoming recoil'
    beam=1
    xbar=1d0
    call forward(-100,p,jac,ps)
    if(pass.or.jac.ge.0d0)error stop 'exhausted beam accepted'
    xbar=0d0
    call forward(-100,p,jac,ps)
    if(pass.or.jac.ge.0d0)error stop 'zero Born fraction accepted'
    xbar=0.4d0
    born(:,beam)=[500d0,0d0,0d0,400d0]
    call forward(-100,p,jac,ps)
    if(pass.or.jac.ge.0d0)error stop 'massive incoming recoil accepted'
  else
    do native=0,1
      native_mapping=native.eq.1
      do masscase=0,1
        mj=80d0*masscase
        do beam=1,2
          do ib=1,size(fractions)
            xbar=fractions(ib)
            do perm=1,2
              call fixture(perm)
              do ir=1,size(xs)
                do iy=1,size(ys)
                  rnd=[xs(ir),ys(iy),0.317d0]
                  call forward(-100,p,jac,ps)
                  if(.not.pass)then
                    write(*,*)'rejected',native,mj,beam,xbar,rnd
                    error stop 'valid FI map rejected'
                  endif
                  if(isign.eq.-1)nminus=nminus+1
                  call invariants(p)
                  spinsave=spin
                  ratio0=ratio
                  if(mode.eq.'production')then
                    call production_roundtrip()
                  elseif(mode.eq.'inverse')then
                    jinv=1d0
                    psinv=1d0
                    call invert_momenta_initial_recoil(p,beam,xbar*ratio0,ifks,jfks,mj, &
                         projected,inv,jinv,psinv,qmap,m2,xback,pass)
                    if(.not.pass)error stop 'FI inverse rejected'
                    if(maxval(abs(inv-rnd)).gt.2d-6.or.abs(xback/xbar-1d0).gt.2d-8)then
                      write(*,*)'inverse',native,mj,beam,xbar,rnd,inv,xback
                      error stop 'FI inverse coordinates'
                    endif
                    j=0
                    do i=1,nexternal
                      if(i.eq.ifks)cycle
                      j=j+1
                      if(maxval(abs(projected(:,j)-born(:,i))).gt.3d-5) &
                           error stop 'FI inverse Born momenta'
                    enddo
                    if(abs(jinv/jac-1d0).gt.5d-5.or.abs(psinv/ps-1d0).gt.5d-5)then
                      write(*,*)'measures',native,mj,beam,xbar,jinv/jac,psinv/ps
                      error stop 'FI inverse radiation measure'
                    endif
                    if(mj.eq.0d0.and.abs(spin-spinsave).gt.1d-7)error stop 'FI inverse spin phase'
                  elseif(mode.eq.'limits'.or.mode.eq.'endpoints')then
                    if(isign.eq.-1)cycle
                    call forward(0,again,j2,ps2)
                    if(.not.pass.or.maxval(abs(again-born)).gt.2d-6)error stop 'FI soft projection'
                    if(abs(ratio-1d0).gt.1d-10)error stop 'FI soft beam fraction'
                    if(native.eq.0.and.abs(ps2/(m2/(4d0*pi)**3)-1d0).gt.1d-8) &
                         error stop 'FI soft radiation measure'
                    if(abs(2d0*mdot(phat,qmap)/m2-1d0).gt.1d-8)error stop 'FI soft normalization'
                    if(mode.eq.'endpoints')then
                      total=born(:,1)+born(:,2)
                      call resonance_subtraction_scales(total,qmap,born(:,jfks),phat, &
                           ss,cs,angs,mlog,pass)
                      if(.not.pass)error stop 'FI subtraction scales'
                      if(mj.eq.0d0)then
                        reference=qterm(born(0,jfks),mdot(total,total),0.13d0,0.4d0)
                        t=qterm(mdot(qmap,born(:,jfks))/sqrt(m2),m2,0.13d0*cs,0.4d0*angs)
                        if(abs(t-reference).gt.2d-8)error stop 'FI integrated collinear term'
                      endif
                      ! Exercise the production prefactors once for each fixture.
                      if(native.eq.0.and.ir.eq.2.and.iy.eq.2)call endpoint_prefactors()
                    endif
                    if(mj.eq.0d0)then
                      call forward(1,again,j2,ps2)
                      if(.not.pass)error stop 'FI collinear map'
                      call invariants(again)
                      if(abs(ratio-1d0).gt.1d-10)error stop 'FI collinear beam fraction'
                      if(maxval(abs(again(:,ifks)+again(:,jfks)-born(:,jfks))).gt.2d-6) &
                           error stop 'FI collinear Born parent'
                      if(abs(ps2/(m2/(4d0*pi)**3*(1d0-xi))-1d0).gt.1d-7) &
                           error stop 'FI collinear radiation measure'
                      call forward(2,again,j2,ps2)
                      if(.not.pass.or.maxval(abs(again-born)).gt.2d-6)error stop 'FI soft collinear projection'
                    endif
                  endif
                enddo
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo
    if(nminus.eq.0)error stop 'massive second branch untested'
  endif
  write(*,*)'PASS '//trim(mode)

contains

  subroutine production_roundtrip()
    use process_module, only: next_n1,nincoming_mod
    use kinematics_module, only: boost_n1_to_lab
    implicit none
    include 'genps.inc'
    include 'run.inc'
    logical :: active,members(nexternal),nbody,nocnt,evpr,valid
    double precision :: resq(0:3),resmass2
    common/c_resonance_recoil/resq,resmass2,active,members
    common/cnbody/nbody
    common/cnocntevents/nocnt
    common/to_use_evpr/evpr
    integer :: recoil_leg
    common/c_initial_recoil/recoil_leg
    double precision :: pmass(nexternal),pb(0:3,nexternal-1),pbl(0:3,nexternal-1),pbe(0:3,nexternal-1)
    common/to_mass/pmass
    common/pborn/pb
    common/pborn_l/pbl
    common/pborn_ev/pbe
    integer :: ifks_active,jfks_active,config,branch
    common/fks_indices/ifks_active,jfks_active
    common/to_mconfigs/config
    common/c_isolsign/branch
    double precision :: pc(0:3,nexternal,-2:2),wc(-2:2),psc(-2:2),jc(-2:2),jc_save(-2:2)
    common/counterevnts/pc,wc,psc,jc
    double precision :: xev(2),xcnt(2,-2:2),sev,rootsev,scnt(-2:2),rootscnt(-2:2)
    common/cbjorkenx/xev,xcnt
    common/parton_cms_ev/rootsev,sev
    common/parton_cms_cnt/rootscnt,scnt
    double precision :: tauev,yev,taucnt(-2:2),ycnt(-2:2)
    common/cbjrk12_ev/tauev,yev
    common/cbjrk12_cnt/taucnt,ycnt
    double precision :: bounds(3),omx(2)
    common/ctau_lower_bound/bounds
    common/to_ee_omx1/omx
    double precision :: xgen(99),massarr(-max_branch:max_particles),mb(nexternal-1), &
         out(0:3,nexternal),lab(0:3,nexternal),outlab(0:3,nexternal),outcms(0:3,nexternal), &
         born_save(0:3,nexternal-1),sqrts,s,stot,taub,yb,yhat,xb(2),j0,ps0,jout,jnew,flux,jexpected, &
         jb,psb,ratio_save,xx(3),jinv0,psinv0,pbinv(0:3,-max_branch:nexternal-1),xbback(2),tauback,yback
    integer :: k,b,ic

    next_n1=nexternal
    nincoming_mod=nincoming
    active=.true.
    members=.false.
    recoil_leg=beam
    ifks_active=ifks
    jfks_active=jfks
    pmass=0d0
    pmass(jfks)=mj
    if(nexternal.gt.4)pmass(4)=200d0
    sqrts=1000d0
    s=sqrts**2
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
    xb=0.21d0
    xb(beam)=xbar
    stot=s/product(xb)
    ebeam=sqrt(stot)/2d0
    lpp=1
    taub=product(xb)
    yb=log(xb(1)/xb(2))/2d0
    yhat=yb/(-0.5d0*log(taub))
    nbody=.false.
    evpr=.true.
    bounds=0d0
    omx=0d0
    xgen=0d0
    xgen(1:3)=rnd
    j0=7d0
    ps0=3d0
    massarr=0d0
    ratio_save=ratio
    call generate_FKS_kinematics(xgen,3,j0,ps0,stot,s,sqrts,taub,yb,yhat, &
         xb,massarr,mb,jout,out,valid)
    if(.not.valid.or.jout.le.0d0)error stop 'production FI map rejected'
    if(maxval(abs(out-p)).gt.3d-5)error stop 'production FI map differs from direct map'
    if(maxval(abs(pbe-born_save)).gt.3d-5)error stop 'FI real partition projection differs from common Born'
    flux=1d0/(2d0*s*ratio_save*(2d0*pi)**(3*(nexternal-nincoming)-7))
    jexpected=7d0*3d0*jac*ps*flux
    if(abs(jout/jexpected-1d0).gt.3d-7)error stop 'FI real flux uses Born energy'
    if(abs(sev/(s*ratio_save)-1d0).gt.1d-10)error stop 'FI real invariant'
    if(abs(xev(beam)/(xbar*ratio_save)-1d0).gt.1d-10)error stop 'FI real PDF fraction'
    if(abs(xev(3-beam)/xb(3-beam)-1d0).gt.1d-10)error stop 'FI spectator PDF fraction'
    if(abs(yev-yb-sign(0.5d0,1.5d0-beam)*log(ratio_save)).gt.1d-10) &
         error stop 'FI real rapidity'
    if(.not.nocnt)then
      if(maxval(abs(pc(:,:,0)-born)).gt.3d-5)error stop 'FI production soft Born'
      if(maxval(abs(pb-born_save)).gt.3d-5)error stop 'FI production changed Born'
      do ic=0,2
        if(jc(ic).le.0d0)cycle
        if(maxval(abs(xcnt(:,ic)-xb)).gt.1d-10.or.abs(scnt(ic)/s-1d0).gt.1d-10) &
             error stop 'FI counterevent PDF or invariant'
        call forward(ic,again,jb,psb)
        flux=1d0/(2d0*s*(2d0*pi)**(3*(nexternal-nincoming)-7))
        if(abs(jc(ic)/(21d0*jb*psb*flux)-1d0).gt.3d-7) &
             error stop 'FI counterevent measure or flux'
      enddo
    endif
    jc_save=jc
    call boost_n1_to_lab(out,lab,-yb)
    jinv0=1d0
    psinv0=1d0
    call invert_fks_radiation(xx,jinv0,psinv0,stot,tauback,yback,xbback,lab,pbinv)
    if(jinv0.le.0d0.or.maxval(abs(xx-rnd)).gt.2d-6)then
      write(*,*)'production inverse',native,mj,beam,xbar,rnd,xx,jinv0
      error stop 'FI production inverse coordinates'
    endif
    if(abs(yback-yb).gt.2d-8.or.maxval(abs(xbback-xb)).gt.2d-8) &
         error stop 'FI production inverse beam fractions'
    if(maxval(abs(pbinv(:,1:nexternal-1)-born_save)).gt.3d-5)error stop 'FI inverse Born CM'
    if(native_mapping)then
      pb=-31d0
      pbl=-32d0
      pbe=-33d0
      call generate_native_momenta(lab,out,outlab,outcms,jnew,valid)
      if(.not.valid)error stop 'FI native replay rejected'
      if(maxval(abs(pb-born_save)).gt.3d-5)error stop 'FI native replay Born CM'
      if(abs(jnew*21d0/jout-1d0).gt.5d-5)error stop 'FI native replay real measure'
      do k=-2,2
        if(jc_save(k).gt.0d0)then
          if(jc(k).le.0d0.or.abs(jc(k)*21d0/jc_save(k)-1d0).gt.5d-5) &
               error stop 'FI native replay counterevent measure'
        endif
      enddo
    endif
    active=.false.
    recoil_leg=0
  end subroutine

  subroutine fixture(perm)
    integer, intent(in) :: perm
    double precision :: energy,momentum
    ifks=nexternal
    jfks=3
    if(perm.eq.2)then
      ifks=3
      jfks=nexternal
    endif
    born=0d0
    born(:,1)=[500d0,0d0,0d0,500d0]
    born(:,2)=[500d0,0d0,0d0,-500d0]
    if(nexternal.gt.4)then
      energy=(1000d0**2+mj**2-200d0**2)/2000d0
      momentum=sqrt(energy**2-mj**2)
      born(:,jfks)=[energy,momentum*sqrt(1d0-0.3d0**2)*cos(0.6d0), &
                          momentum*sqrt(1d0-0.3d0**2)*sin(0.6d0),momentum*0.3d0]
      born(:,4)=born(:,1)+born(:,2)-born(:,jfks)
    else
      born(:,jfks)=born(:,1)+born(:,2)
    endif
    q=born(:,jfks)+(1d0/xbar-1d0)*born(:,beam)
  end subroutine

  subroutine forward(ic,pout,jout,psout)
    integer, intent(in) :: ic
    double precision, intent(out) :: pout(0:3,nexternal),jout,psout
    pout=born
    jout=2d0*pi
    psout=1d0
    call generate_momenta_initial_recoil(ic,isign,ifks,jfks,mj,beam,xbar, &
         rnd(1:2),2d0*pi*rnd(3),pout,xmax,xnorm,xi,y,xihat,phat,jout,psout,qmap,m2,ratio,pass)
  end subroutine

  subroutine invariants(pout)
    double precision, intent(in) :: pout(0:3,nexternal)
    double precision :: scale,shell,mass
    integer :: leg
    if(.not.all(ieee_is_finite(pout)).or..not.ieee_is_finite(jac*ps))error stop 'nonfinite FI map'
    scale=maxval(pout(0,:))
    residual=sum(pout(:,1:2),dim=2)-sum(pout(:,3:nexternal),dim=2)
    if(maxval(abs(residual)).gt.3d-8*scale)error stop 'FI four momentum conservation'
    do leg=1,nexternal
      if(leg.ne.ifks.and.leg.ne.jfks.and.leg.ne.beam)then
        if(any(pout(:,leg).ne.born(:,leg)))error stop 'FI spectator changed'
      endif
      mass=0d0
      if(leg.eq.jfks)mass=mj
      if(leg.eq.4.and.nexternal.gt.4)mass=200d0
      shell=mdot(pout(:,leg),pout(:,leg))
      if(abs(shell-mass**2).gt.4d-8*scale**2)error stop 'FI mass shell'
    enddo
    if(maxval(abs(pout(:,beam)-ratio*born(:,beam))).gt.1d-8*scale)error stop 'FI beam direction'
    if(ratio.lt.1d0-1d-10.or.ratio*xbar.gt.1d0+1d-10)error stop 'FI beam fraction bounds'
  end subroutine

  subroutine endpoint_prefactors()
    double precision :: cmq(0:3),cmm2
    logical :: local,mask(nexternal)
    common/c_resonance_recoil/cmq,cmm2,local,mask
    double precision :: cnt(0:3,nexternal,-2:2),unused(-2:2),pscnt(-2:2),jcnt(-2:2)
    common/counterevnts/cnt,unused,pscnt,jcnt
    double precision :: xiev,yev,pev(0:3),pcnt(0:3,-2:2),xicnt(-2:2),normev,maxev
    common/fksvariables/xiev,yev,pev,pcnt
    common/cxiifkscnt/xicnt
    common/cxinormev/normev
    common/cxiimaxev/maxev
    double precision :: maxcnt(-2:2),normcnt(-2:2)
    common/cxiimaxcnt/maxcnt
    common/cxinormcnt/normcnt
    double precision :: delta,cut,scut,bsvcut,boostlab,boostcm,sqrts,s
    common/cdelta_used/delta
    common/cxicut_used/cut
    common/cxiScut_used/scut,bsvcut
    common/parton_cms_stuff/boostlab,boostcm,sqrts,s
    double precision :: sym,symborn,symdeg
    integer :: ng,nquarks(-6:6),ngamma,ci,cj
    common/numberofparticles/sym,symborn,symdeg,ng,nquarks,ngamma
    common/fks_indices/ci,cj
    logical :: nocounter
    common/cnocntevents/nocounter
    double precision :: fr,fs,fc,fdc,fsc,fdsc(4),masses(nexternal)
    common/factor_n1body/fr,fs,fc,fdc,fsc,fdsc
    common/to_mass/masses
    double precision :: mcs,mch,mccs,mcch,mcscs,mcsch,mcrs,mcrh
    common/factor_n1body_NLOPS/mcs,mch,mccs,mcch,mcscs,mcsch,mcrs,mcrh
    double precision :: before(3),mcbefore(6),expect,cap,a,vegas,je,pe,logs,logc,loga
    integer :: ic
    ci=ifks
    cj=jfks
    nocounter=.false.
    sym=1.3d0
    symborn=0.9d0
    symdeg=sym
    cut=0.13d0
    delta=0.4d0
    scut=0.31d0
    bsvcut=0.7d0
    vegas=0.8d0
    masses=0d0
    masses(jfks)=mj
    s=1d6
    sqrts=1d3
    call forward(-100,p,je,pe)
    normev=xnorm
    maxev=xmax
    xiev=xi
    yev=y
    pev=phat
    do ic=0,2
      if(mj.ne.0d0.and.ic.ne.0)cycle
      call forward(ic,cnt(:,:,ic),jcnt(ic),pscnt(ic))
      jcnt(ic)=jcnt(ic)*pscnt(ic)
      normcnt(ic)=xnorm
      maxcnt(ic)=xmax
      xicnt(ic)=xi
      pcnt(:,ic)=phat
    enddo
    cmq=qmap
    cmm2=m2
    local=.false.
    call compute_prefactors_n1body(vegas,je*pe)
    before=[fs,fc,fsc]
    mcbefore=[mcs,mch,mccs,mcch,mcscs,mcsch]
    local=.true.
    call compute_prefactors_n1body(vegas,je*pe)
    call resonance_subtraction_scales(born(:,1)+born(:,2),cmq,born(:,jfks),pcnt(:,0), &
         ss,cs,angs,mlog,pass)
    logs=log(ss)
    logc=log(cs)
    loga=log(angs)
    expect=jcnt(0)*sym*vegas*normev/min(maxev,scut)*logs/(1d0-yev)
    if(abs(fs-before(1)-expect).gt.2d-10*max(1d0,abs(fs)))error stop 'FI soft endpoint mismatch'
    if(maxval(abs([mcs,mch,mccs,mcch,mcscs,mcsch]-mcbefore)).gt.1d-10) &
         error stop 'FI endpoint changed MC G replacement'
    if(mj.eq.0d0)then
      a=normcnt(1)
      cap=min(maxcnt(1),scut)
      expect=jcnt(1)*sym*vegas*a/xicnt(1)*loga
      if(abs(fc-before(2)-expect).gt.2d-10*max(1d0,abs(fc)))error stop 'FI collinear endpoint mismatch'
      expect=a/xicnt(1)*loga+a/cap*logc/(1d0-yev) &
           +a/cap*(logc*log(delta)+log(cut/cap)*loga+logc*loga)
      expect=expect*jcnt(2)*sym*vegas
      if(abs(fsc-before(3)-expect).gt.2d-10*max(1d0,abs(fsc))) &
           error stop 'FI soft collinear endpoint mismatch'
    endif
  end subroutine

  double precision function mdot(a,b)
    double precision, intent(in) :: a(0:3),b(0:3)
    mdot=a(0)*b(0)-sum(a(1:3)*b(1:3))
  end function

  double precision function qterm(energy,s,cut,delta)
    double precision, intent(in) :: energy,s,cut,delta
    double precision, parameter :: c=4d0/3d0,gamma=2d0,gammap=4.5d0,qes2=121d0**2
    qterm=gammap-log(s*delta/(2d0*qes2))*(gamma-2d0*c*log(2d0*energy/(cut*sqrt(s)))) &
         +2d0*c*(log(2d0*energy/sqrt(s))**2-log(cut)**2)-2d0*gamma*log(2d0*energy/sqrt(s))
  end function

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

subroutine set_cms_stuff(ic)
  implicit none
  integer, intent(in) :: ic
  ! The prefactor fixture holds the physical Born CM fixed.
end subroutine
