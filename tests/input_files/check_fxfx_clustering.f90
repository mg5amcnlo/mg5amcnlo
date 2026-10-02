program check_fxfx_clustering
  implicit none
  integer :: lpp(2)
  double precision :: bwcutoff,d
  common /test_run/ bwcutoff,lpp
  common /to_dj/ d
  character(len=32) :: mode
  call get_command_argument(1,mode)
  lpp=1
  bwcutoff=15d0
  d=1d0
  select case (mode)
  case ('resonance')
     call check_candidate(.true.)
  case ('nonresonance')
     call check_candidate(.false.)
  case ('initial_state')
     call check_initial_state()
  case ('zero','wrapper')
     call check_zero(mode.eq.'wrapper')
  case ('qcd_core')
     call check_qcd_core()
  case default
     error stop 'unknown clustering test'
  end select
  print *, 'PASS '//trim(mode)
contains
  subroutine close(actual,expected,label)
    double precision, intent(in) :: actual,expected
    character(len=*), intent(in) :: label
    if (.not.(abs(actual-expected).le.1d-10*max(1d0,abs(expected)))) then
       print *, label,actual,expected
       error stop 'unexpected clustering value'
    endif
  end subroutine

  subroutine check_candidate(resonance_first)
    logical, intent(in) :: resonance_first
    integer :: ctype(63),particle_type(6),imap(6),bw(2,0:3),lst(6,1)
    integer :: i,win,iwin,jwin
    double precision :: p(0:4,6),scale,p_inter(0:4,0:2),expected_mass
    logical :: is_bw,valid(1)
    p=0d0
    p(:,1)=[145d0,0d0,0d0,145d0,0d0]
    p(:,2)=[145d0,0d0,0d0,-145d0,0d0]
    p(:,3)=[45d0,27d0,0d0,36d0,0d0]
    p(:,4)=[45d0,-27d0,0d0,-36d0,0d0]
    p(:,5)=[100d0,100d0,0d0,0d0,0d0]
    p(:,6)=[100d0,-100d0,0d0,0d0,0d0]
    do i=1,6
       imap(i)=ishft(1,i-1)
    enddo
    ! The first pair wins in both fixtures. The later pair has the opposite
    ! resonance status, so using the last candidate's flag fails either way.
    ctype=0
    bw=0
    bw(1,0)=1
    if (resonance_first) then
       ctype(12)=16
       ctype(48)=1
       particle_type=[2,2,8,8,1,1]
       bw(:,1)=[12,23]
       expected_mass=90d0
    else
       ctype(12)=1
       ctype(48)=16
       particle_type=[2,2,1,1,8,8]
       bw(:,1)=[48,23]
       expected_mass=0d0
    endif
    lst(:,1)=[12,48,13,51,15,50]
    valid=.true.
    is_bw=.not.resonance_first
    call cluster_one_step(6,p,imap,3,1,valid,lst,bw,iwin,jwin,win,scale,is_bw,ctype,particle_type)
    if (win.ne.12.or.iwin.ne.4.or.jwin.ne.3) error stop 'wrong winning candidate'
    if (is_bw.neqv.resonance_first) error stop 'resonance flag belongs to losing candidate'
    if (resonance_first) call close(scale,90d0,'resonance clustering scale')
    call update_momenta(6,p,iwin,jwin,p_inter,is_bw,particle_type,ctype(win))
    call close(p(4,jwin),expected_mass,'stored clustering mass')
    call close(p_inter(4,0),expected_mass,'intermediate clustering mass')
    call close(p(0,jwin),90d0,'combined energy')
    if (particle_type(jwin).ne.ctype(win)) error stop 'combined particle type'
  end subroutine

  subroutine check_initial_state()
    integer :: bw(2,0:1),cl(0:2)
    double precision :: p(0:4),beam(0:4),scale,cluster_scale
    logical :: is_bw
    external cluster_scale
    bw=0
    cl=1
    p=[5d0,3d0,0d0,4d0,0d0]
    beam=[100d0,0d0,0d0,100d0,0d0]
    is_bw=.true.
    scale=cluster_scale(bw,1,1,5,p,beam,cl,is_bw)
    if (is_bw) error stop 'initial-state clustering retained resonance flag'
    call close(scale,3d0,'initial-state scale')
  end subroutine

  subroutine check_zero(wrapper)
    use fks_phase_space_data, only: p_born
    use mc_native_context, only: native_epoch
    use weight_lines, only: pdg_uborn
    logical, intent(in) :: wrapper
    integer :: ipdg(3),cpdg(0:2,0:0,2),tree(2,-1:-1),ij(0),iord(0:0)
    integer :: conf,empty(0,2),sp(0),ctype(7),process,ib,repeat_call
    integer :: nren,matching(4),this_config,nFKSprocess
    double precision :: p(0:3,3),scales(0:0),mass(0),width(0),m,rapidity
    double precision :: sudakov,expanded,ren(0:4),fac
    double precision :: pmass(-4:0,2,0:1),pwidth(-4:0,2,0:1)
    integer :: iforest(2,-2:-1,2,0:1),sprop(-2:-1,2,0:1)
    integer :: tprid(-2:-1,2,0:1),mapconfig(0:2,0:1)
    common /c_configurations/pmass,pwidth,iforest,sprop,tprid,mapconfig
    common /to_mconfigs/this_config
    common /c_nFKSprocess/nFKSprocess
    pmass=0d0
    pwidth=0d0
    iforest=0
    sprop=0
    tprid=0
    mapconfig=0
    mapconfig(0,0)=2
    this_config=2
    nFKSprocess=1
    tree(:,-1)=[1,2]
    iforest(:,-1,1,0)=tree(:,-1)
    iforest(:,-1,2,0)=tree(:,-1)
    do process=1,3
       select case (process)
       case (1)
          ipdg=[2,-2,23]
          m=90d0
          lpp=1
       case (2)
          ipdg=[21,21,25]
          m=125d0
          lpp=1
       case (3)
          ipdg=[11,-11,23]
          m=90d0
          lpp=0
       end select
       do ib=0,1
          ! Test longitudinal invariance for hadronic initial states. The
          ! lepton-collider metric uses energies in its centre-of-mass frame.
          if (process.eq.3.and.ib.eq.1) cycle
          rapidity=0.7d0*ib
          p(:,1)=[m/2d0*exp(rapidity),0d0,0d0,m/2d0*exp(rapidity)]
          p(:,2)=[m/2d0*exp(-rapidity),0d0,0d0,-m/2d0*exp(-rapidity)]
          p(:,3)=p(:,1)+p(:,2)
          if (wrapper) then
             native_epoch=native_epoch+1
             p_born=p
             pdg_uborn=0
             pdg_uborn(1:3,0)=ipdg
             do repeat_call=1,2
                ! The second call exercises the cached topology tables.
                call cluster_and_reweight(-1,sudakov,expanded,nren,ren,fac,matching,.false.)
                if (nren.ne.0) error stop '2->1 has a QCD branching scale'
                call close(ren(0),m,'2->1 renormalisation scale')
                call close(fac,m,'2->1 factorisation scale')
                call close(sudakov,1d0,'2->1 Sudakov')
                call close(expanded,0d0,'2->1 expanded Sudakov')
                if (matching(3).ne.0) error stop 'colour singlet requires matching'
             enddo
          else
             ctype=0
             cpdg=-777
             iord=-777
             call cluster(3,p,2,0,empty,cpdg,tree,ipdg,mass,width,2,sp,conf,scales,ij,iord,ctype)
             if (conf.ne.2) error stop '2->1 lost configuration selection'
             if (iord(0).ne.0) error stop '2->1 core ordering'
             if (any(cpdg(:,0,conf).ne.[ipdg(3),ipdg(1),ipdg(2)])) error stop '2->1 core PDGs'
             call close(scales(0),m,'2->1 core scale')
          endif
       enddo
    enddo
  end subroutine

  subroutine check_qcd_core()
    integer :: ipdg(4),cpdg(0:2,0:2,1),tree(2,-2:-1),ij(1),iord(0:1)
    integer :: conf,lst(2,1),sp(-1:-1),tp(-1:-1),ctype(15)
    double precision :: p(0:3,4),scales(0:1),mass(-1:-1),width(-1:-1)
    ipdg=[2,-2,21,21]
    p(:,1)=[50d0,0d0,0d0,50d0]
    p(:,2)=[50d0,0d0,0d0,-50d0]
    p(:,3)=[50d0,30d0,0d0,40d0]
    p(:,4)=[50d0,-30d0,0d0,-40d0]
    tree(:,-1)=[3,4]
    tree(:,-2)=[1,2]
    sp=21
    tp=0
    mass=0d0
    width=0d0
    ctype=0
    call iforest_to_list(4,2,1,tree,sp,tp,width,ipdg,lst,cpdg,ctype)
    call cluster(4,p,1,1,lst,cpdg,tree,ipdg,mass,width,1,sp,conf,scales,ij,iord,ctype)
    if (conf.ne.1.or.ij(1).ne.12.or.iord(1).ne.1) error stop 'QCD core clustering'
    if (any(cpdg(:,0,1).ne.[21,2,-2])) error stop 'QCD core PDGs'
    call close(scales(0),30d0,'QCD core transverse scale')
    call close(scales(1),30d0,'last QCD transverse scale')
  end subroutine
end program

! Model/export interfaces are controlled data. All kinematic calculations and
! clustering/reweighting routines are the production implementations.
integer function get_color(id)
  implicit none
  integer :: id
  get_color=1
  if (abs(id).le.6.and.id.ne.0) get_color=3
  if (id.eq.21) get_color=8
end function

integer function get_spin(id)
  implicit none
  integer :: id
  get_spin=2
  if (id.eq.21.or.id.eq.23) get_spin=3
  if (id.eq.25) get_spin=1
end function

double precision function get_mass_from_id(id)
  implicit none
  integer :: id
  get_mass_from_id=0d0
  if (id.eq.23) get_mass_from_id=90d0
  if (id.eq.25) get_mass_from_id=125d0
end function

subroutine set_pdg(dummy,iproc)
  implicit none
  integer :: dummy,iproc
  if (dummy.ne.0.or.iproc.ne.1) error stop 'unexpected process lookup'
end subroutine

subroutine QCDsudakov(q0,q2,q1,next,itype,mass,exponent,expanded,for_mcatnlo_scale)
  implicit none
  integer :: next,itype(0:next)
  double precision :: q0,q2,q1,mass(next),exponent,expanded
  logical :: for_mcatnlo_scale
  error stop '2->1 process must not request a branching Sudakov'
end subroutine
