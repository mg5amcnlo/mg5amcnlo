module controlled_connections
  implicit none
  integer :: connection_count=1
  logical :: angular_live(2)=.false.
  double precision :: kernel_g(2)
end module controlled_connections

program check_mc_dead_zones
  use mc_counterterms, only: compute_MCsubtraction_kl,get_dead_zone, &
       ileg,fksfather,xm12,xm22,w1,w2,shat_n1,gfactsf,gfactcl,gfactazi
  use process_module
  use fks_phase_space_helpers, only: dot,sumdot
  use scale_module
  use herwig7_scales
  use controlled_connections
  use FKSParams, only: Pythia8MMaxGamma
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  implicit none
  double precision :: pb(0:3,5), p(0:3,6), pmass(6), raw(1,2), zout(2)
  double precision :: alsf,besf,alazi,beazi,xi,y,q,cap,z,xis,weight
  double precision :: shifts(3),coeff(3),a(3),gs(3),partner_mass,basepx
  double precision :: angular(5,5),effective(5,5)
  double precision :: ch_i,ch_j,ch_m
  integer :: i_type,j_type,m_type,j_pdg
  common /cparticle_types/ ch_i,ch_j,ch_m,i_type,j_type,m_type,j_pdg
  integer :: k,j,ncon,i,status,imass
  integer :: lpp(2)
  double precision :: ebeam(2),xbk(2),q2fact(2)
  common /to_collider/ ebeam,xbk,q2fact,lpp
  logical :: include_gfun,live,invalid,divzero,connected(5,5)
  logical :: softtest,colltest
  character(32) :: name
  common /to_mass/pmass
  common /cgfunsfp/alsf,besf
  common /cgfunazi/alazi,beazi
  common /sctests/softtest,colltest

  call get_command_argument(1,name)
  softtest=.false.
  colltest=.false.
  next_n=5
  next_n1=6
  nincoming_mod=2
  allocate(mass_n(5),shower_scale_nbody_min(5,5),shower_scale_nbody_max(5,5))
  mass_n=0d0
  pmass=0d0
  born_flow_picked=1
  alsf=1d0
  besf=-0.1d0
  alazi=-1d0
  beazi=-0.1d0
  shower_scale_nbody_min=100d0
  shower_scale_nbody_max=1000d0
  shower_mc_mod='PYTHIA8'
  pb=0d0

  select case (trim(name))
  case ('herwig')
    shower_mc_mod='HERWIG7'
    connected=.true.
    do i=1,5
      connected(i,i)=.false.
    enddo
    do imass=1,2
      if (imass.eq.2) mass_n(3:4)=[173d0,80d0]
      pb(:,1)=[500d0,0d0,0d0,500d0]
      pb(:,2)=[500d0,0d0,0d0,-500d0]
      pb(:,3)=[sqrt(300d0**2+mass_n(3)**2),300d0,0d0,0d0]
      pb(:,4)=[sqrt(300d0**2+mass_n(4)**2),-300d0,0d0,0d0]
      pb(:,5)=[10d0,0d0,10d0,0d0]
      call herwig7_starting_scales(5,pb,mass_n,connected,10000d0,effective,status,angular)
      call require(status.eq.hw7_ok,'Herwig scale status')
      shower_scale_nbody_max=10000d0
      do i=1,4
        fksfather=i
        ileg=i
        xm12=mass_n(i)**2
        xm22=0d0
        if (i.gt.2) ileg=merge(3,4,mass_n(i).gt.0d0)
        do j=1,4
          if (i.eq.j) cycle
          do k=1,2
            xis=angular(i,j)**2*merge(1d0-1d-8,1d0+1d-8,k.eq.1)
            call get_dead_zone(0.5d0,xis,pb,0d0,j,live,weight)
            call require(live.eqv.(k.eq.1),'Herwig module agrees with production angular boundary')
          enddo
        enddo
      enddo
      ! Soft wide-angle radiation may have qtilde > SCALUP and pT < SCALUP.
      fksfather=3
      ileg=merge(3,4,mass_n(3).gt.0d0)
      xm12=mass_n(3)**2
      xis=0.9d0*angular(3,4)**2
      z=0.99d0
      q=(1d0-z)*sqrt(z*z*xis-xm12)
      shower_scale_nbody_max=2d0*q
      call get_dead_zone(z,xis,pb,q,4,live,weight)
      call require(live,'Herwig keeps soft wide-angle radiation above scalar angular scale')
      shower_scale_nbody_max=q/2d0
      call get_dead_zone(z,xis,pb,q,4,live,weight)
      call require(.not.live,'Herwig enforces independent scalar pT veto')
    enddo

  case ('qed_isr_channels')
    pb(:,1)=[500d0,0d0,0d0,500d0]
    pb(:,2)=[500d0,0d0,0d0,-500d0]
    fksfather=1
    ileg=1
    shat_n1=1d6
    z=0.5d0
    xis=1d0
    q=1d0
    lpp=[1,1]
    j_type=3
    ch_j=2d0/3d0
    j_pdg=2
    m_type=3
    ch_m=ch_j
    i_type=1
    ch_i=0d0
    call get_dead_zone(z,xis,pb,q,2,live,weight,2)
    call require(live,'hadron quark photon radiation is supported')
    m_type=1
    ch_m=0d0
    i_type=3
    ch_i=ch_j
    call get_dead_zone(z,xis,pb,q,2,live,weight,2)
    call require(live,'hadron Born photon evolves backwards to a quark')
    j_pdg=6
    call get_dead_zone(z,xis,pb,q,2,live,weight,2)
    call require(.not.live,'Born photon has no incoming top mother')
    j_pdg=2
    lpp=[0,0]
    call get_dead_zone(z,xis,pb,q,2,live,weight,2)
    call require(.not.live,'lepton beam Born photon has no backward quark evolution')
    lpp=[1,1]
    j_type=1
    ch_j=-1d0
    j_pdg=11
    i_type=1
    ch_i=-1d0
    call get_dead_zone(z,xis,pb,q,2,live,weight,2)
    call require(.not.live,'hadron Born photon has no backward lepton evolution')
    j_pdg=22
    ch_j=0d0
    m_type=3
    ch_m=2d0/3d0
    i_type=3
    ch_i=-ch_m
    call get_dead_zone(z,xis,pb,q,2,live,weight,2)
    call require(.not.live,'proton photon PDF cannot produce a backward photon mother')
    call get_dead_zone(z,xis,pb,q,2,live,weight)
    call require(live,'QCD support is independent of QED channel restrictions')
    lpp=[0,0]
    call get_dead_zone(z,xis,pb,q,2,live,weight,2)
    call require(.not.live,'lepton beam cannot produce a backward photon mother')
    j_pdg=11
    ch_j=-1d0
    m_type=1
    ch_m=-1d0
    i_type=1
    ch_i=0d0
    call get_dead_zone(z,xis,pb,q,2,live,weight,2)
    call require(live,'ordinary lepton ISR photon radiation remains supported')
    lpp=[2,2]
    call get_dead_zone(z,xis,pb,q,2,live,weight,2)
    call require(.not.live,'two UPC beams disable ISR as in launch script')
    lpp=[2,1]
    j_pdg=2
    j_type=3
    ch_j=2d0/3d0
    m_type=1
    ch_m=0d0
    i_type=3
    ch_i=ch_j
    call get_dead_zone(z,xis,pb,q,2,live,weight,2)
    call require(live,'one UPC beam does not disable ISR in launch script')
    ! Restrict the active incoming side, independently of the opposite beam.
    lpp=[0,1]
    fksfather=2
    ileg=2
    call get_dead_zone(z,xis,pb,q,1,live,weight,2)
    call require(live,'incoming side two uses its own physical beam type')

  case ('qed_conversion')
    pb(:,1)=[500d0,0d0,0d0,500d0]
    pb(:,2)=[500d0,0d0,0d0,-500d0]
    pb(:,3)=[400d0,400d0,0d0,0d0]
    pb(:,4)=[300d0,-200d0,sqrt(50000d0),0d0]
    pb(:,5)=[300d0,-200d0,-sqrt(50000d0),0d0]
    fksfather=3
    ileg=4
    shat_n1=1d6
    xm12=2d5
    xm22=0d0
    w1=0d0
    w2=100d0
    z=0.5d0
    xis=w2*z*(1d0-z)
    q=sqrt(xis)
    m_type=1
    ch_m=0d0
    Pythia8MMaxGamma=10d0
    call get_dead_zone(z,xis,pb,q,4,live,weight)
    call require(live,'default QCD support ignores photon conversion bound')
    call get_dead_zone(z,xis,pb,q,4,live,weight,2)
    call require(.not.live,'QED conversion excludes exact upper mass bound')
    ch_m=-1d0
    call get_dead_zone(z,xis,pb,q,4,live,weight,2)
    call require(live,'charged lepton radiation has no conversion mass bound')
    ch_m=0d0
    Pythia8MMaxGamma=11d0
    call get_dead_zone(z,xis,pb,q,4,live,weight,2)
    call require(live,'QED conversion uses configurable upper mass')
    w2=1d-24
    xis=w2*z*(1d0-z)
    q=sqrt(xis)
    call get_dead_zone(z,xis,pb,q,4,live,weight,2)
    call require(live,'QED conversion subtraction retains the collinear limit')
    ! Single massless spectator invariants in generated QED points can
    ! be slightly spacelike. Keep their singular support without an FPE.
    xm12=-1.3d-10
    call ieee_set_flag(ieee_invalid,.false.)
    call ieee_set_flag(ieee_divide_by_zero,.false.)
    call get_dead_zone(z,xis,pb,q,4,live,weight,2)
    call ieee_get_flag(ieee_invalid,invalid)
    call ieee_get_flag(ieee_divide_by_zero,divzero)
    call require(live,'massless global recoil roundoff keeps the collinear limit')
    call require(.not.invalid.and..not.divzero,'massless global recoil has no FPE')
    xm12=-1d-5
    call get_dead_zone(z,xis,pb,q,4,live,weight,2)
    call require(.not.live,'materially spacelike global recoil is rejected')
    xm12=2d5
    w2=100d0
    xis=w2*z*(1d0-z)
    q=sqrt(xis)
    pb(:,4)=[100d0,100d0,0d0,0d0]
    pb(:,5)=[500d0,-500d0,0d0,0d0]
    call get_dead_zone(z,xis,pb,q,4,live,weight,2)
    call require(.not.live,'QED conversion still obeys its local dipole scale cap')

  case ('massless','massive')
    partner_mass=0d0
    if (name.eq.'massive') partner_mass=173d0
    mass_n(4)=partner_mass
    pb(:,3)=[500d0,500d0,0d0,0d0]
    pb(:,4)=[sqrt(500d0**2+partner_mass**2),400d0,300d0,0d0]
    pb(:,5)=[100d0*sqrt(90d0),-900d0,-300d0,0d0]
    pb(0,1)=sum(pb(0,3:5))/2d0
    pb(0,2)=pb(0,1)
    pb(3,1)=pb(0,1)
    pb(3,2)=-pb(0,2)
    fksfather=3
    ileg=4
    shat_n1=(2d0*pb(0,1))**2
    xm12=sumdot(pb(:,4),pb(:,5),1d0)
    xm22=0d0
    w1=0d0
    cap=(sqrt(partner_mass**2+2d0*dot(pb(:,3),pb(:,4)))-partner_mass)/2d0
    shifts=[0d0,-1d-12,1d-12]
    basepx=pb(1,4)
    do k=1,3
      pb(1,4)=basepx+shifts(k)
      do j=1,2
        q=cap*merge(0.8d0,1.2d0,j.eq.1)
        z=0.5d0
        xis=q*q
        w2=4d0*xis
        call ieee_set_flag(ieee_invalid,.false.)
        call ieee_set_flag(ieee_divide_by_zero,.false.)
        call get_dead_zone(z,xis,pb,q,4,live,weight)
        call ieee_get_flag(ieee_invalid,invalid)
        call ieee_get_flag(ieee_divide_by_zero,divzero)
        call require(.not.invalid.and..not.divzero,'finite dipole boundary')
        call require(live.eqv.(j.eq.1),'local dipole cap survives roundoff')
      enddo
    enddo

  case ('soft_transition')
    ! Dead angular support must retain a smooth soft replacement.
    do k=1,3
      call evaluate(0.05d0+(k-2)*1d-8,-0.5d0,.true.)
      coeff(k)=1d0-gfactsf
      call require(all(raw.eq.0d0),'dead raw kernel')
    enddo
    call require(all(coeff.gt.0.49d0),'retain G on both sides of xi=0.05')
    call require(maxval(coeff)-minval(coeff).lt.1d-6,'continuous soft taper')
    call evaluate(0.1d0,-0.5d0,.true.)
    call require(gfactsf.eq.1d0,'natural end of G support')

  case ('scale_endpoint')
    angular_live=.true.
    ! For these ISR kinematics q=1000*xi*sqrt(2*(1-y)).
    q=75d0*sqrt(3d0)
    shower_scale_nbody_min=0.1d0*q
    do k=1,3
      shower_scale_nbody_max=q*(1d0+(2-k)*1d-6)
      call evaluate(0.075d0,-0.5d0,.true.)
      coeff(k)=1d0-gfactsf
      a(k)=sum(raw)
    enddo
    call require(coeff(1).ge.0d0.and.coeff(1).lt.2d-12,'G tapers to zero')
    call require(all(coeff(2:3).eq.0d0),'G vanishes at and above scale cap')
    call require(a(1).ge.0d0.and.a(1).lt.1d-9,'raw term tapers to zero')
    call require(all(a(2:3).eq.0d0),'raw term vanishes at scale cap')

  case ('two_connections')
    connection_count=2
    angular_live=[.true.,.false.]
    q=75d0*sqrt(3d0)
    ! First connection has D=0; the angular-dead second has D=1.
    shower_scale_nbody_min=2d0*q
    shower_scale_nbody_max=3d0*q
    shower_scale_nbody_min(1,2)=0.1d0*q
    shower_scale_nbody_max(1,2)=q
    call evaluate(0.075d0,-0.5d0,.true.)
    call require(abs((1d0-gfactsf)-0.05d0).lt.1d-14,'equal partner probabilities')
    call require(all(raw.eq.0d0),'no raw term from dead connections')
    ! At D=1/2, the raw kernel must still see the original G=0.9.
    angular_live=.true.
    shower_scale_nbody_min=0.5d0*q
    shower_scale_nbody_max=1.5d0*q
    call evaluate(0.075d0,-0.5d0,.true.)
    call require(all(abs(kernel_g-0.9d0).lt.1d-14),'G update follows raw kernels')
    call require(abs(sum(raw)*0.075d0**2*1.5d0-0.9d0).lt.1d-14, &
                 'raw damping applied once')

  case ('limits')
    ! Angular-dead soft radiation still has its complete replacement.
    call evaluate(1d-6,-0.5d0,.true.)
    call require(gfactsf.eq.0d0,'soft wide-angle limit')
    call evaluate(1d-6,1d0-1d-8,.true.)
    call require(gfactsf.eq.0d0.and.gfactcl.eq.0d0,'soft-collinear limit')
    ! A history that does not request G must not alter its shared values.
    gfactsf=0.25d0
    gfactcl=0.5d0
    gfactazi=0.75d0
    gs=[gfactsf,gfactcl,gfactazi]
    call evaluate(0.075d0,-0.5d0,.false.)
    call require(all(gs.eq.[gfactsf,gfactcl,gfactazi]),'disabled G is untouched')
  case default
    stop 2
  end select
  print *, 'PASS '//trim(name)

contains
  subroutine evaluate(xi_in,y_in,with_g)
    double precision, intent(in) :: xi_in,y_in
    logical, intent(in) :: with_g
    ! Only beams and the emitted momentum enter the ISR invariants.
    xi=xi_in
    y=y_in
    p=0d0
    p(:,1)=[1000d0,0d0,0d0,1000d0]
    p(:,2)=[1000d0,0d0,0d0,-1000d0]
    p(:,6)=1000d0*xi*[1d0,sqrt(1d0-y*y),0d0,y]
    pb(:,1:2)=p(:,1:2)
    include_gfun=with_g
    call compute_MCsubtraction_kl(6,1,xi,y,p,p,pb,include_gfun,zout,ncon,raw)
  end subroutine evaluate

  subroutine require(condition,message)
    logical, intent(in) :: condition
    character(*), intent(in) :: message
    if (.not.condition) then
      print *, 'FAIL '//trim(name)//': '//message
      stop 1
    endif
  end subroutine require
end program check_mc_dead_zones

subroutine find_color_connectors(iflow,iparticle,n_connect,i_connect)
  use controlled_connections
  implicit none
  integer :: iflow,iparticle,n_connect,i_connect(2)
  n_connect=connection_count
  i_connect=[2,3]
end subroutine find_color_connectors

! The support fixture supplies controlled Born amplitudes and connections.
subroutine prepare_MCsubtraction_born(p,xi,y,p_born,kernel_index,born_weights,born_spin_weights)
  implicit none
  double precision :: p(0:3,6),xi,y,p_born(0:3,5)
  double precision :: born_weights(1),born_spin_weights(1)
  integer :: kernel_index
  kernel_index=1
  born_weights=0d0
  born_spin_weights=0d0
end subroutine prepare_MCsubtraction_born

subroutine xmcsubt_connection(p_born,i_connect,qmc,damping,include_gfun,kernel_index, &
                             kernel_limits,born_weights,born_spin_weights,live,z,amp)
  use mc_counterterms, only: mc_kernel_limits,fksfather,gfactsf
  use controlled_connections
  use scale_module
  implicit none
  double precision :: p_born(0:3,5),qmc,damping,z,amp(1),g
  double precision :: born_weights(1),born_spin_weights(1)
  type(mc_kernel_limits) :: kernel_limits
  integer :: i_connect,kernel_index,j
  logical :: include_gfun,live
  j=i_connect-1
  kernel_g(j)=gfactsf
  live=angular_live(j).and.qmc.le.shower_scale_nbody_max(fksfather,i_connect)
  z=0.5d0
  amp=0d0
  g=1d0
  if (include_gfun) g=gfactsf
  if (live) amp=g*damping
end subroutine xmcsubt_connection
