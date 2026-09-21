module controlled_connections
  implicit none
  integer :: connection_count=1
  logical :: angular_live(2)=.false.
  double precision :: kernel_g(2)
end module controlled_connections

program check_mc_dead_zones
  use process_module
  use kinematics_module
  use scale_module
  use controlled_connections
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  implicit none
  double precision :: pb(0:3,5), p(0:3,6), pmass(6), raw(1,2), zout(2)
  double precision :: alsf,besf,alazi,beazi,xi,y,q,cap,z,xis,weight
  double precision :: shifts(3),coeff(3),a(3),gs(3),partner_mass,basepx
  integer :: k,j,ncon
  logical :: include_gfun,live,invalid,divzero
  character(32) :: name
  common /to_mass/pmass
  common /cgfunsfp/alsf,besf
  common /cgfunazi/alazi,beazi

  call get_command_argument(1,name)
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
    shower_scale_nbody_min(2,1)=0.1d0*q
    shower_scale_nbody_max(2,1)=q
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

subroutine xmcsubt_connection(p,xi,y,p_born,i_connect,include_gfun,live,z,amp)
  use controlled_connections
  use kinematics_module
  use scale_module
  implicit none
  double precision :: p(0:3,6),xi,y,p_born(0:3,5),z,amp(1),g
  double precision, external :: compute_damping_weight
  integer :: i_connect,j
  logical :: include_gfun,live
  j=i_connect-1
  kernel_g(j)=gfactsf
  live=angular_live(j).and.get_qMC(xi,y).le.shower_scale_nbody_max(i_connect,fksfather)
  z=0.5d0
  amp=0d0
  g=1d0
  if (include_gfun) g=gfactsf
  if (live) amp=g*compute_damping_weight(i_connect,xi,y)
end subroutine xmcsubt_connection
