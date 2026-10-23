! Scalar first-branch factors for Pythia 8.318's simple QCD showers.
!
! The ISR routine describes ordinary global initial-initial backwards
! evolution with no MEC, azimuthal bias, pT0 screening, rescattering,
! enhancement or uncertainty weights.  Alpha_s, PDFs, the inverse shower
! map/Jacobian, colour factors and starting-scale averaging stay with
! the caller.  A supported zero means genuinely absent shower support.
! An unsupported status MUST NOT be interpreted as a zero kernel.
!
! Near a c/b threshold Pythia calls pT2nearThreshold: Q->Qg is absent and
! the only possible branching is a forced, normalized g->QQbar conversion
! rather than alpha_s P(z)/t. No scalar correction to the massless
! production kernel represents that conversion distribution. Similarly a
! massive emitted sister requires its massive map and Jacobian, not only
! the mass correction to an AP coefficient. The FSR routine supplies the
! physical post-rescaling cutoff and g->qqbar species support, on top of
! the caller's existing scalar global-recoil map and AP density.
module mc_pythia8_kernel
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: py8_isr_ordinary_weight, py8_isr_mass_floor
  public :: py8_fsr_ordinary_weight
  public :: py8_isr_ok, py8_isr_bad_input, py8_isr_flavour
  public :: py8_isr_threshold, py8_isr_massive_map

  integer, parameter :: py8_isr_ok=0, py8_isr_bad_input=1
  integer, parameter :: py8_isr_flavour=2, py8_isr_threshold=3
  integer, parameter :: py8_isr_massive_map=4
  double precision, parameter :: tiny_pt2=0.25d-6
  double precision, parameter :: heavy_x_evol=0.9d0
  double precision, parameter :: heavy_threshold=2d0
  double precision, parameter :: extra_pair_space=8d0

contains

  pure double precision function py8_isr_mass_floor(id,mass_charm,mass_bottom) result(mass)
    integer, intent(in) :: id
    double precision, intent(in) :: mass_charm,mass_bottom
    mass=0d0
    if (abs(id)==4) mass=max(1.2d0,mass_charm)
    if (abs(id)==5) mass=max(4d0,mass_bottom)
  end function py8_isr_mass_floor

  subroutine py8_fsr_ordinary_weight(id_parent,id_first,id_second,z,pt2, &
      mass_parent,mass_first,mass_second,ptmin,n_gluon_to_quark,weight,status)
    ! Additional accepted-branch support for SimpleTimeShower::branch.
    ! The caller supplies the existing global-recoil map, scalar AP
    ! coefficient and recoilDeadCone factor. No MECs/azimuthal bias.
    ! z may refer to either daughter: all factors below are symmetric.
    integer, intent(in) :: id_parent,id_first,id_second,n_gluon_to_quark
    double precision, intent(in) :: z,pt2,mass_parent,mass_first,mass_second,ptmin
    double precision, intent(out) :: weight
    integer, intent(out) :: status
    double precision :: pair2,rescale
    integer :: parent,first,second

    weight=0d0
    status=py8_isr_bad_input
    if (.not.all(ieee_is_finite([z,pt2,mass_parent,mass_first,mass_second,ptmin]))) return
    if (z<=0d0.or.z>=1d0.or.pt2<=0d0) return
    if (min(mass_parent,mass_first,mass_second,ptmin)<0d0) return
    if (n_gluon_to_quark<0.or.n_gluon_to_quark>5) return
    parent=abs(id_parent)
    first=abs(id_first)
    second=abs(id_second)
    status=py8_isr_flavour

    if (parent>=1.and.parent<=6.and. &
        ((id_first==id_parent.and.second==21).or. &
         (id_second==id_parent.and.first==21))) then
      ! The pre-rescaling virtual pair is m_parent^2+t/[z(1-z)].
      ! Pythia rejects g emission on the RESCALED evolution pT,
      ! even if the generated t lies above pTcolCut^2.
      pair2=mass_parent**2+pt2/(z*(1d0-z))
      rescale=(pt2/(z*(1d0-z)))/pair2
      status=py8_isr_ok
      if (sqrt(pt2)*rescale<ptmin) return
    else if (parent==21.and.first==21.and.second==21) then
      status=py8_isr_ok
      if (sqrt(pt2)<ptmin) return
    else if (parent==21.and.first>=1.and.first<=6.and.id_second==-id_first) then
      status=py8_isr_ok
      ! The flavour draw never produces top; nGluonToQuark <= 5.
      if (first>n_gluon_to_quark) return
      ! A massive pair has beta-weighted z and a different daughter
      ! map. Do not multiply a massless inverse-map measure by beta.
      if (mass_first>0d0.or.mass_second>0d0) then
        status=py8_isr_massive_map
        return
      endif
      ! Unlike gluon emission, a qqbar conversion has no secondary
      ! pTorig cutoff in branch(). pT2nextQCD enforces t>=pTcolCut^2.
      if (sqrt(pt2)<ptmin) return
    else
      return
    endif
    weight=1d0
  end subroutine py8_fsr_ordinary_weight

  subroutine py8_isr_ordinary_weight(id_daughter,id_mother,id_sister,z,pt2, &
      shat_born,x_daughter,x_max,mass_charm,mass_bottom,mass_sister, &
      massive_sister_map,weight,status)
    integer, intent(in) :: id_daughter,id_mother,id_sister
    double precision, intent(in) :: z,pt2,shat_born,x_daughter,x_max
    double precision, intent(in) :: mass_charm,mass_bottom,mass_sister
    logical, intent(in) :: massive_sister_map
    double precision, intent(out) :: weight
    integer, intent(out) :: status
    double precision :: mass2,zmax,mratio,q2,pt2corr,pair_pt2corr
    integer :: daughter,mother,sister,channel

    weight=0d0
    status=py8_isr_bad_input
    if (.not.all(ieee_is_finite([z,pt2,shat_born,x_daughter,x_max, &
        mass_charm,mass_bottom,mass_sister]))) return
    if (z<=0d0.or.z>=1d0.or.pt2<=0d0.or.shat_born<=0d0) return
    if (x_daughter<=0d0.or.x_max<=0d0.or.x_max>1d0) return
    if (min(mass_charm,mass_bottom,mass_sister)<0d0) return

    daughter=abs(id_daughter)
    mother=abs(id_mother)
    sister=abs(id_sister)
    status=py8_isr_flavour
    channel=0
    if (daughter==21.and.mother==21.and.sister==21) then
      channel=1 ! g -> g (+ g)
    else if (daughter>=1.and.daughter<=5.and.id_mother==id_daughter.and.sister==21) then
      channel=2 ! q -> q (+ g)
    else if (daughter>=1.and.daughter<=5.and.mother==21.and.id_sister==-id_daughter) then
      channel=3 ! g -> q (+ qbar)
    else if (daughter==21.and.mother>=1.and.mother<=5.and.id_sister==id_mother) then
      channel=4 ! q -> g (+ q)
    endif
    if (channel==0) return

    status=py8_isr_ok
    ! Beam extraction bound. Heavy threshold conversions must be detected
    ! before testing ordinary virtuality: they use Q2=t/(1-z)-mQ2.
    if (x_daughter/z>x_max) return
    mass2=py8_isr_mass_floor(id_daughter,mass_charm,mass_bottom)**2
    if (mass2>0d0) then
      mratio=sqrt(mass2/shat_born)
      if (mratio>=1d0) return
      zmax=(1d0-mratio)/(1d0+mratio*(1d0-mratio))
      if (zmax<=0d0) return
      if (x_daughter>heavy_x_evol*zmax*x_max.or.z>zmax) return
      if (pt2<heavy_threshold*mass2) then
        ! The heavy-daughter evolution switches to pT2nearThreshold
        ! BEFORE selecting an ordinary channel. That routine can only
        ! make g -> Q (+ Qbar); Q -> Q (+ g) is genuinely absent here,
        ! irrespective of the starting scale (which can reduce the
        ! threshold window but cannot add an ordinary emission in it).
        if (channel==2) return
        ! The global-II virtuality and emission measure also change in
        ! pT2nearThreshold. Do not return a misleading zero or AP ratio.
        status=py8_isr_threshold
        return
      endif
    endif

    ! Global-II transverse-momentum reality bound for ordinary evolution.
    q2=pt2/(1d0-z)
    pt2corr=q2-z*(shat_born+q2)*(q2+mass_sister**2)/shat_born
    if (pt2corr<tiny_pt2) return

    if (channel==4.and.(mother==4.or.mother==5)) then
      ! A backwards Q -> g Q must leave room for an earlier QQbar pair.
      pair_pt2corr=q2-z*(shat_born+q2)* &
        (q2+extra_pair_space*mass_sister**2)/shat_born
      if (pair_pt2corr<tiny_pt2) return
    endif

    if (mass_sister>0d0.and..not.massive_sister_map) then
      status=py8_isr_massive_map
      return
    endif

    weight=1d0
    if (channel==2.and.mass2>0d0) then
      weight=1d0-2d0*z*(1d0-z)**2*mass2/(pt2*(1d0+z*z))
    else if (channel==3.and.mass2>0d0) then
      weight=1d0+2d0*z*(1d0-z)*mass2/(pt2*(z*z+(1d0-z)**2))
    endif
    ! Negative accepted trial weights have zero probability in Pythia.
    weight=max(0d0,weight)
  end subroutine py8_isr_ordinary_weight

end module mc_pythia8_kernel
