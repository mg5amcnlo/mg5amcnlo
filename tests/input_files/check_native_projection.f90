! Check the production native projection independently of Born chart sampling.
subroutine check_ee_soft_recoil_projection()
  use fks_phase_space_data
  use fks_phase_space, only: generate_native_momenta
  use mc_native_context, only: native_mapping
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  include 'nexternal.inc'
  include 'run.inc'
  double precision :: pmass(nexternal)
  common/to_mass/pmass
  integer :: ifks,jfks,config
  common/fks_indices/ifks,jfks
  common/to_mconfigs/config
  logical :: nbody,evpr,fixed_order,nlo_ps
  common/cnbody/nbody
  common/to_use_evpr/evpr
  common/c_fnlo_nlops/fixed_order,nlo_ps
  double precision :: omx(2)
  common/to_ee_omx1/omx
  double precision :: original(0:3,5),lab(0:3,5),out(0:3,5),outlab(0:3,5),cms(0:3,5),jac,error
  logical :: pass
  integer :: side

  ! e+e- -> u ubar g, AmpliCol Born-spreading calibration: hard
  ! antiparallel daughters leave a spectator with E=5.88e-5 GeV.
  ! Encoding their opening angle as sqrt((1-y)/2) rounded away enough
  ! of 1+y to rotate the reconstructed hard momenta by 1.6 percent.
  original(:,1)=[500d0,0d0,0d0,500d0]
  original(:,2)=[500d0,0d0,0d0,-500d0]
  original(:,3)=[4.99999971120579517d2,3.91755149232678434d2, &
       1.17192260134123487d2,-2.87739201943214880d2]
  original(:,4)=[5.88200116507517839d-5,-7.83233714307370300d-6, &
       5.65384072435636868d-5,1.42076307614650772d-5]
  original(:,5)=[4.99999970059408838d2,-3.91755141400341188d2, &
       -1.17192316672530623d2,2.87739187735584096d2]
  native_mapping=.true.
  fixed_order=.false.
  nlo_ps=.true.
  ebeam=500d0
  lpp=0
  pmass=0d0
  omx=0d0
  nbody=.false.
  evpr=.true.
  ifks=5
  jfks=3
  config=1
  tau_Born_lower_bound=0d0
  tau_lower_bound=0d0
  tau_lower_bound_resonance=0d0
  initial_recoil_leg=0
  resonance_recoil=.false.
  do side=1,2
     lab=original
     if(side.eq.2)then
        lab(:,3)=original(:,5)
        lab(:,5)=original(:,3)
     endif
     call generate_native_momenta(lab,out,outlab,cms,jac,pass)
     error=maxval(abs(outlab-lab))/maxval(abs(lab))
     if(.not.pass.or..not.ieee_is_finite(jac).or.jac.le.0d0.or.error.gt.1d-6)then
        write(*,*) 'soft recoil projection error',side,error
        error stop 'massless native soft-recoil projection failed'
     endif
  enddo
end subroutine

subroutine check_onshell_isr_boost()
  use fks_phase_space_helpers, only: boost_isr_recoil
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  double precision :: born(0:3),realp(0:3),recovered(0:3),mass2,xi,y,phi,tolerance
  integer :: idir,mass_case,orientation,power,angle

  phi=0.61d0
  do idir=-1,1,2
    do mass_case=0,1
      mass2=0.04d0*mass_case
      do orientation=-1,1,2
        born=[1d0,3d-5,4d-5,orientation*sqrt(1d0-mass2-25d-10)]
        ! The unchanged forward map forms E-|pz|. Its input roundoff
        ! is amplified by E^2/(m^2+pt^2) on the spectator side; the
        ! inverse must recover that point while preserving its mass.
        tolerance=max(5d-9,8d0*epsilon(1d0)/(mass2+sum(born(1:2)**2)))
        do power=1,11,2
          xi=1d0-10d0**(-power)
          do angle=-1,1
            y=0.9d0*angle
            call boost_isr_recoil(born,realp,xi,y,phi,idir,.false.)
            call boost_isr_recoil(realp,recovered,xi,y,phi,idir,.true.,mass2)
            if(.not.all(ieee_is_finite(recovered)))error stop 'nonfinite on-shell ISR boost'
            if(maxval(abs(recovered-born)).gt.tolerance)then
              write(*,*) 'ISR boost round trip',idir,mass_case,orientation,power,angle
              write(*,*) 'Born/recovered',born,recovered
              error stop 'on-shell ISR boost round trip'
            endif
            if(abs(recovered(0)**2-sum(recovered(1:3)**2)-mass2).gt.1d-12) &
                 error stop 'on-shell ISR boost mass'
            ! Production boosts permit input/output aliasing.
            call boost_isr_recoil(realp,realp,xi,y,phi,idir,.true.,mass2)
            if(any(realp.ne.recovered))error stop 'on-shell ISR boost aliasing'
          enddo
        enddo
        call boost_isr_recoil(born,recovered,0d0,0.3d0,phi,idir,.true.,mass2)
        if(any(recovered.ne.born))error stop 'soft ISR boost is not identity'
        call boost_isr_recoil(born,recovered,0.9d0,1d0,phi,idir,.true.,mass2)
        if(any(recovered.ne.born))error stop 'collinear ISR boost is not identity'
      enddo
    enddo
    ! The exact spectator-beam null ray has p+=0 and m=pt=0.
    born=[1d0,0d0,0d0,-dble(idir)]
    xi=1d0-1d-10
    call boost_isr_recoil(born,realp,xi,0.3d0,phi,idir,.false.)
    call boost_isr_recoil(realp,recovered,xi,0.3d0,phi,idir,.true.,0d0)
    if(.not.all(ieee_is_finite(recovered)).or.maxval(abs(recovered-born)).gt.1d-14) &
         error stop 'spectator null ray ISR inverse'
  enddo
end subroutine

subroutine check_dijet_isr_boundary()
  use fks_phase_space_data
  use fks_phase_space, only: generate_native_momenta
  use mc_native_context, only: native_mapping
  use FKSParams, only: FKSISRMapping
  use mcatnlo_delta_scales, only: pythia8_starting_scales,delta_ok
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  include 'nexternal.inc'
  include 'run.inc'
  double precision :: pmass(nexternal)
  common/to_mass/pmass
  integer :: ifks,jfks,config
  common/fks_indices/ifks,jfks
  common/to_mconfigs/config
  logical :: nbody,evpr,fixed_order,nlo_ps
  common/cnbody/nbody
  common/to_use_evpr/evpr
  common/c_fnlo_nlops/fixed_order,nlo_ps
  double precision :: omx(2)
  common/to_ee_omx1/omx
  double precision :: original(0:3,5),lab(0:3,5),out(0:3,5),outlab(0:3,5),cms(0:3,5),jac
  double precision :: scales(4,4),mass2,residual(0:3)
  logical :: pass,connected(4,4)
  integer :: side,i,status

  ! u d > u d, seed 482701: the native beam-2 ISR projection used to
  ! produce p_4^2=-2.66e-9 and (p_4+p_2)^2=-2.03e-9 GeV^2. This
  ! tripped the Pythia8 starting-scale guard, despite physical real input.
  original(:,1)=[169.60230028028022d0,0d0,0d0,169.60230028028022d0]
  original(:,2)=[2.3379478331029611d0,0d0,0d0,-2.3379478331029611d0]
  original(:,3)=[4.5693370933264097d-3,4.1505332515925218d-5, &
       3.9912357784991324d-5,4.5689742594499762d-3]
  original(:,4)=[4.5145970111160088d0,0.50417273752395886d0, &
       4.4852320381793174d0,-0.10044693726073906d0]
  original(:,5)=[167.42108176517380d0,-0.50421424285647387d0, &
       -4.4852719505370953d0,167.36023041017850d0]
  native_mapping=.true.
  fixed_order=.false.
  nlo_ps=.true.
  FKSISRMapping=2
  ebeam=6500d0
  lpp=1
  pmass=0d0
  omx=0d0
  nbody=.false.
  evpr=.true.
  ifks=5
  config=1
  tau_Born_lower_bound=0d0
  tau_lower_bound=0d0
  tau_lower_bound_resonance=0d0
  initial_recoil_leg=0
  resonance_recoil=.false.
  connected=.true.
  do i=1,4
    connected(i,i)=.false.
  enddo
  do side=1,2
    jfks=side
    lab=original
    if(side.eq.1)then
      lab(:,1)=original(:,2)
      lab(:,2)=original(:,1)
      lab(3,:)=-lab(3,:)
    endif
    call generate_native_momenta(lab,out,outlab,cms,jac,pass)
    if(.not.pass.or..not.ieee_is_finite(jac).or.jac.le.0d0) &
         error stop 'dijet native ISR projection failed'
    do i=1,4
      mass2=p_born(0,i)**2-sum(p_born(1:3,i)**2)
      if(abs(mass2).gt.1d-12*p_born(0,i)**2) &
           error stop 'dijet ISR inverse lost the Born mass shell'
    enddo
    residual=sum(p_born(:,1:2),dim=2)-sum(p_born(:,3:4),dim=2)
    if(maxval(abs(residual)).gt.1d-7*maxval(abs(p_born))) &
         error stop 'dijet Born momentum conservation'
    if(maxval(abs(outlab-lab)).gt.1d-7*maxval(abs(lab))) &
         error stop 'dijet ISR projection changed the real event'
    call pythia8_starting_scales(4,p_born,pmass(1:4),connected, &
         4.7682856448980768d0,scales,status)
    if(status.ne.delta_ok.or..not.all(ieee_is_finite(scales))) &
         error stop 'dijet Pythia8 starting scales failed'
  enddo
end subroutine

subroutine check_native_projection()
  use fks_phase_space_data,only: pb => p_born,pbl => p_born_l,pbe => p_born_ev,isign => isolsign, &
       bound_born => tau_Born_lower_bound,bound_res => tau_lower_bound_resonance, &
       bound_tau => tau_lower_bound,nocnt => nocntevents,pc => p1_cnt,&
       jc => jac_cnt,xi => xi_i_fks_ev,y => y_ij_fks_ev,pi_ev => p_i_fks_ev,pi_cnt => p_i_fks_cnt, &
       xi_cnt => xi_i_fks_cnt,xih => xi_i_hat_ev,xih_cnt => xi_i_hat_cnt,xb => xbjrk_ev,xbc => xbjrk_cnt, &
       tau => tau_ev,ycm => ycm_ev,tauc => tau_cnt,ycmc => ycm_cnt,sqrts_ev => sqrtshat_ev, &
       s_ev => shat_ev,sqrts_cnt => sqrtshat_cnt,s_cnt => shat_cnt,xmax => xiimax_ev,xmaxc => xiimax_cnt, &
       xnorm => xinorm_ev,xnormc => xinorm_cnt,kn => veckn_ev,knbar => veckbarn_ev,kn0 => xp0jfks, &
       spin => xij_aor
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mc_native_context, only: native_mapping
  use fks_phase_space, only: generate_FKS_kinematics,generate_native_momenta
  use fks_phase_space_helpers, only: boost_n1_to_lab
  implicit none
  include 'genps.inc'
  include 'nexternal.inc'
  include 'run.inc'
  double precision :: pmass(nexternal)
  common/to_mass/pmass
  integer :: ifks,jfks,config
  common/fks_indices/ifks,jfks
  common/to_mconfigs/config
  double precision :: omx(2)
  common/to_ee_omx1/omx
  logical :: nbody,evpr,fixed_order,nlo_ps
  common/cnbody/nbody
  common/to_use_evpr/evpr
  common/c_fnlo_nlops/fixed_order,nlo_ps
  double precision :: x(99),mb(nexternal-1),born(0:3,nexternal-1)
  double precision :: p(0:3,nexternal),lab(0:3,nexternal),out(0:3,nexternal),outlab(0:3,nexternal),cms(0:3,nexternal)
  double precision :: stot,sborn,sqrtborn,taub,yb,yhat,xbb(2),j0,ps0,jac,jnew,u,e3,e4
  double precision :: reference(256),actual(256),ratio(-2:2),cntref(0:3,nexternal,-2:2)
  double precision :: bounds_poison(3),omx_poison(2)
  double precision,parameter :: radii(4)=[0.02d0,0.45d0,0.9d0,0.999d0]
  double precision,parameter :: angles(3)=[0.15d0,0.6d0,0.95d0]
  logical :: pass,valid(-2:2),nocnt_ref
  integer :: round,history,h,ir,ia,k,n,isign_ref,negative_count

  native_mapping=.true.
  fixed_order=.false.
  nlo_ps=.true.
  ebeam=[6500d0,4000d0] ! asymmetric physical beams; input is symmetric hadron frame
  lpp=1
  stot=4d0*product(ebeam)
  sqrtborn=1000d0
  sborn=sqrtborn**2
  taub=sborn/stot
  yb=0.37d0
  yhat=yb/(-0.5d0*log(taub))
  xbb=sqrt(taub)*[exp(yb),exp(-yb)]
  ifks=5
  negative_count=0
  do round=1,2
    do history=1,4
      h=history
      if(round.eq.2)h=5-history
      jfks=min(h,4)
      if(h.eq.3)jfks=4
      pmass=0d0
      pmass(3)=80d0
      if(h.eq.4)pmass(4)=173d0
      mb=pmass(1:4)
      e3=(sborn+mb(3)**2-mb(4)**2)/(2d0*sqrtborn)
      e4=sqrtborn-e3
      u=sqrt(e3**2-mb(3)**2)
      born(:,1)=[sqrtborn/2d0,0d0,0d0,sqrtborn/2d0]
      born(:,2)=[sqrtborn/2d0,0d0,0d0,-sqrtborn/2d0]
      born(:,3)=[e3,u*0.3d0,u*0.4d0,u*sqrt(0.75d0)]
      born(:,4)=[e4,-born(1:3,3)]
      do ir=1,size(radii)
        do ia=1,size(angles)
          x=0d0
          x(1:3)=[radii(ir),angles(ia),0.31d0]
          bound_born=sum(mb(3:4))**2/stot
          bound_res=sum(mb(3:4))**2/stot
          bound_tau=sum(mb(3:4))**2/stot
          omx=0d0
          nbody=.false.
          evpr=.true.
          pb=born
          pbl=born
          pbe=born
          pc=0d0
          pc(0,1,:)=-1d0
          kn=0d0
          knbar=0d0
          kn0=0d0
          ! Arbitrary nonunit Born factors must cancel from all ratios.
          j0=7d0
          ps0=3d0
          call generate_FKS_kinematics(x(1:3),nbody,j0,ps0,stot,sborn,sqrtborn,taub,yb,yhat, &
               xbb,mb,jac,p,pass)
          if(jac.le.0d0)error stop 'invalid reference radiation point'
          call boost_n1_to_lab(p,lab,-yb)
          valid=jc.gt.0d0
          ratio=jc/jac
          cntref=pc
          isign_ref=isign
          nocnt_ref=nocnt
          if(isign.eq.-1)negative_count=negative_count+1
          call snapshot(reference,n)
          ! Deliberately poison state left by a different history/fold.
          bounds_poison=[0.8d0,0.9d0,0.95d0]
          omx_poison=[0.2d0,0.3d0]
          bound_born=bounds_poison(1)
          bound_res=bounds_poison(2)
          bound_tau=bounds_poison(3)
          omx=omx_poison
          nbody=.true.
          evpr=.false.
          pb=-999d0
          pbl=-998d0
          pbe=-997d0
          pc=777d0
          jc=777d0
          spin=(99d0,99d0)
          pi_cnt=999d0
          config=999
          call generate_native_momenta(lab,out,outlab,cms,jnew,pass)
          if(.not.pass)error stop 'native radiation projection failed'
          if(abs(jnew*21d0/jac-1d0).gt.1d-8)error stop 'Born measure did not cancel'
          if(maxval(abs(outlab-lab)).gt.1d-7)error stop 'real point changed'
          if(maxval(abs(pb-born)).gt.1d-8.or.any(pb.ne.pbl).or.any(pb.ne.pbe)) &
               error stop 'Born arrays not populated'
          if(any([bound_born,bound_res,bound_tau].ne.bounds_poison).or.any(omx.ne.omx_poison).or. &
               .not.nbody)error stop 'input controls leaked'
          if(.not.evpr.or.config.ne.1)error stop 'native evaluator controls not installed'
          if(any(valid.neqv.(jc.gt.0d0)).or.(nocnt.neqv.nocnt_ref).or.isign.ne.isign_ref) &
               error stop 'counterevent validity changed'
          do k=-2,2
            if(valid(k))then
              if(abs(jc(k)/jnew-ratio(k)).gt.1d-8*max(1d0,abs(ratio(k)))) &
                   error stop 'counterevent measure ratio changed'
              if(maxval(abs(pc(:,:,k)-cntref(:,:,k))).gt.1d-8)error stop 'counterevent changed'
            else
              if(pc(0,1,k).ge.0d0)error stop 'stale inactive counterevent'
            endif
          enddo
          call snapshot(actual,n)
          if(.not.all(ieee_is_finite(actual(1:n))))error stop 'nonfinite native phase-space state'
          if(any(abs(actual(1:n)-reference(1:n)).gt.1d-8*max(1d0,abs(reference(1:n))))) &
               error stop 'native phase-space state differs'
        enddo
      enddo
    enddo
  enddo
  if(negative_count.eq.0)error stop 'massive second solution not covered'
  lab(0,1)=-1d0
  call generate_native_momenta(lab,out,outlab,cms,jnew,pass)
  if(pass.or.jnew.ge.0d0.or.out(0,1).ge.0d0)error stop 'invalid input accepted'
  if(any([bound_born,bound_res,bound_tau].ne.bounds_poison).or.any(omx.ne.omx_poison).or. &
       .not.nbody)error stop 'invalid input changed controls'
contains
  subroutine snapshot(v,n)
    double precision,intent(out) :: v(256)
    integer,intent(out) :: n
    integer :: k
    v=0d0
    v(1:20)=[xi,y,pi_ev,xih,xb,tau,ycm,sqrts_ev,s_ev,xmax,xnorm, &
         real(spin,kind=8),aimag(spin),kn,knbar,kn0]
    n=20
    do k=-2,2
      if(jc(k).le.0d0)cycle
      v(n+1:n+14)=[pi_cnt(:,k),xi_cnt(k),xih_cnt(k),xbc(:,k),tauc(k),ycmc(k), &
           sqrts_cnt(k),s_cnt(k),xmaxc(k),xnormc(k)]
      n=n+14
    enddo
  end subroutine
end subroutine

! W+jet FxFx failures with seeds 33 and 34, including Born spreading.
! These ISR points approach xi=1, y=-1 in the native massless FSR map.
! Combine the leptons into one spectator to fit the five-leg fixture.
subroutine check_wjet_boundary_projection()
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mc_native_context, only: native_mapping
  use fks_phase_space, only: generate_native_momenta
  implicit none
  include 'nexternal.inc'
  include 'run.inc'
  double precision :: pmass(nexternal)
  common/to_mass/pmass
  integer :: ifks,jfks,icase
  common/fks_indices/ifks,jfks
  logical :: fixed_order,nlo_ps,pass
  common/c_fnlo_nlops/fixed_order,nlo_ps
  double precision :: lab(0:3,nexternal),out(0:3,nexternal),outlab(0:3,nexternal),cms(0:3,nexternal),jac
  double precision, parameter :: max_error(3)=[2d-7,3d-6,1d-5]

  native_mapping=.true.
  fixed_order=.false.
  nlo_ps=.true.
  ebeam=6500d0
  lpp=1
  ifks=5
  jfks=4
  do icase=1,3
     select case(icase)
     case(1)
        lab(:,1)=[4.97378848813441863d1,0d0,0d0,4.97378848813441863d1]
        lab(:,2)=[5.78442699296585943d2,0d0,0d0,-5.78442699296585943d2]
        lab(:,3)=[7.79721787019871271d-1,1.41351812862392456d-1,2.47042628431951433d-2,-7.66404220729106300d-1] &
             +[3.49521894148706238d1,6.74629986065035858d0,2.77473019055295526d-2,-3.42949298464129342d1]
        lab(:,4)=[4.89929574265278063d2,9.48042725279744474d1,7.08861844937025976d-1,-4.80668945496875779d2]
        lab(:,5)=[1.02519098714858615d2,-1.01691924201487225d2,-7.61313409685750675d-1,-1.29745348553210249d1]
     case(2)
        lab(:,1)=[2.79685316665763025d2,0d0,0d0,2.79685316665763025d2]
        lab(:,2)=[1.57595997453356313d3,0d0,0d0,-1.57595997453356313d3]
        lab(:,3)=[7.94702108421625439d1,9.05590640269074765d-1,1.61699534918292165d0,-7.94485974887583950d1] &
             +[6.08235144134407051d2,3.11191205504165591d0,1.32552602632878944d1,-6.08082728449770343d2]
        lab(:,4)=[8.87331714243827491d2,5.15862952689793719d0,1.93499583581557992d1,-8.87105708895975567d2]
        lab(:,5)=[2.80608221958734418d2,-9.17613222220866831d0,-3.42222139706266120d1,2.78362376986898823d2]
     case(3)
        lab(:,1)=[2.37359568100654741d2,0d0,0d0,2.37359568100654741d2]
        lab(:,2)=[7.23277069457887194d1,0d0,0d0,-7.23277069457887194d1]
        lab(:,3)=[1.28512259200941870d1,-5.22294127586781842d-1,-3.39813855243778218d-1,-1.28361109000106950d1] &
             +[5.11000910848290175d1,-1.74546024693960611d0,-8.38989360004154139d-1,-5.10633799729118607d1]
        lab(:,4)=[8.39693291711085799d0,-3.03041822428878171d-1,-1.57287808907320997d-1,-8.38998859435982602d0]
        lab(:,5)=[2.37339025127945661d2,2.57079619695526640d0,1.33609102415525349d0,2.37321340618612169d2]
     end select
     pmass=0d0
     pmass(3)=sqrt(lab(0,3)**2-sum(lab(1:3,3)**2))
     call generate_native_momenta(lab,out,outlab,cms,jac,pass)
     if(.not.pass.or..not.ieee_is_finite(jac).or.jac.le.0d0) &
          error stop 'near-boundary W+jet native projection rejected'
     if(.not.all(ieee_is_finite(outlab)))error stop 'nonfinite W+jet reconstruction'
     if(maxval(abs(outlab-lab)).gt.max_error(icase)*maxval(abs(lab))) &
          error stop 'near-boundary W+jet reconstruction changed'
  enddo
  ! A larger inconsistency must still fail, leaving invalid output sentinels.
  lab(0,5)=lab(0,5)+1d-5
  call generate_native_momenta(lab,out,outlab,cms,jac,pass)
  if(pass.or.jac.ge.0d0.or.out(0,1).ge.0d0) &
       error stop 'inconsistent W+jet native projection accepted'
end subroutine

! Seed 33 of the single-top acceptance test: an ISR event is projected
! onto the massive top's FSR history. The boosted massless spectator has
! a mass squared of about -2e-6 GeV^2 at shat=1.17e8 GeV^2 from roundoff.
subroutine check_singletop_projection()
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_get_flag,ieee_set_flag,ieee_invalid
  use mc_native_context, only: native_mapping
  use fks_phase_space, only: generate_native_momenta
  use fks_phase_space_helpers, only: get_recoil
  implicit none
  include 'nexternal.inc'
  include 'run.inc'
  double precision :: pmass(nexternal)
  common/to_mass/pmass
  integer :: ifks,jfks
  common/fks_indices/ifks,jfks
  logical :: fixed_order,nlo_ps
  common/c_fnlo_nlops/fixed_order,nlo_ps
  double precision :: lab(0:3,nexternal),out(0:3,nexternal),outlab(0:3,nexternal),cms(0:3,nexternal),jac
  double precision :: born(0:3,nexternal-1),recoil_mass2
  logical :: pass,invalid

  native_mapping=.true.
  fixed_order=.false.
  nlo_ps=.true.
  ebeam=6500d0
  lpp=1
  pmass=0d0
  pmass(3)=173d0
  ifks=5
  lab(:,1)=[4.90336305984129922d3,0d0,0d0,4.90336305984129922d3]
  lab(:,2)=[5.98832878508703834d3,0d0,0d0,-5.98832878508703834d3]
  lab(:,3)=[1.05213686096379342d3,8.22413110017194981d2,5.03813650535259967d2,3.83238119457055632d2]
  lab(:,4)=[4.14225519962702037d3,2.91528792504187959d3,2.39945608080732018d3,1.70352134392874473d3]
  lab(:,5)=[5.69729978433733322d3,-3.73770103505907400d3,-2.90326973134258014d3,-3.17172518863134883d3]
  do jfks=1,4
     call ieee_set_flag(ieee_invalid,.false.)
     call generate_native_momenta(lab,out,outlab,cms,jac,pass)
     call ieee_get_flag(ieee_invalid,invalid)
     if(invalid)error stop 'invalid arithmetic in single-top projection'
     if(.not.pass.or..not.ieee_is_finite(jac).or.jac.le.0d0) &
          error stop 'single-top native recoil projection failed'
     if(maxval(abs(outlab-lab)).gt.1d-7*maxval(abs(lab))) &
          error stop 'single-top native recoil changed real point'
  enddo
  ! A small physical timelike recoil must survive, while a spacelike
  ! recoil larger than boost roundoff must still reject the point.
  born=0d0
  born(:,1)=[10000d0,0d0,0d0,10000d0]
  born(:,2)=[10000d0,0d0,0d0,-10000d0]
  born(:,3)=[12000d0-5d-10,8000d0,0d0,0d0]
  born(:,4)=born(:,1)+born(:,2)-born(:,3)
  call get_recoil(born,3,4d8,recoil_mass2,pass)
  if(.not.pass.or.recoil_mass2.le.0d0)error stop 'small positive recoil mass lost'
  born(0,3)=12000d0+1d-4
  born(:,4)=born(:,1)+born(:,2)-born(:,3)
  call get_recoil(born,3,4d8,recoil_mass2,pass)
  if(pass)error stop 'unphysical spacelike recoil accepted'
end subroutine
