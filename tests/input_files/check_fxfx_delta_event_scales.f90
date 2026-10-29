module captured_lhe_scales
  implicit none
  double precision :: scalar,directed(500,500)
  integer :: event_size
end module

program check_fxfx_delta_event_scales
  use process_module
  use scale_module
  use captured_lhe_scales
  implicit none
  character(32) :: mode
  logical :: delta
  integer :: i,j,next,npart,ickkw,jpart(7,-2:7),icolalt(2,-2:7)
  double precision :: pb(0:4,-2:7),scales(-2:7,-2:7),original(5,5),expected
  integer :: need_matching(5),nfksprocess
  integer :: iSorH_lhe,ifks_lhe(1),jfks_lhe(1),fksfather_lhe(1),ipartner_lhe(1)
  double precision :: scale1_lhe(1),scale2_lhe(1),g,gal(2),mu_r
  character(1000) :: ptclusstring
  character(18) :: number
  common/test_ickkw/ickkw
  common/test_couplings/g,gal,mu_r
  common/c_need_matching_to_write/need_matching
  common/c_nFKSprocess/nfksprocess
  common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
  common/cto_LHE2/scale1_lhe,scale2_lhe
  common/c_ptclusstring/ptclusstring

  call get_command_argument(1,mode)
  delta=trim(mode).eq.'delta'
  call init_process_module_global('PYTHIA8   ','all ',5,2,delta,13000d0,1,1,3)
  call init_scale_module(5,1d0,1,1)
  ickkw=3
  nfksprocess=1
  ifks_lhe=5
  jfks_lhe=1
  fksfather_lhe=1
  ipartner_lhe=2
  scale1_lhe=1d0
  scale2_lhe=1d0
  g=1d0
  gal=1d0
  mu_r=91d0
  showerscaleS_hard=29d0
  showerscaleH_hard=53d0
  do iSorH_lhe=1,2
    next=3+iSorH_lhe
    expected=showerscaleS_hard
    if(iSorH_lhe.eq.2)expected=showerscaleH_hard
    do i=1,5
      do j=1,5
        original(i,j)=dble(10*i+j)/10d0
        if(i.eq.j)original(i,j)=-1d0
      enddo
    enddo
    if (delta) then
      showerscaleS=original(1:4,1:4)
      showerscaleH=original
    else
      showerscaleS=expected
      showerscaleH=expected
    endif
    scales=-1d0
    if(iSorH_lhe.eq.1)then
      scales(1:ndelS,1:ndelS)=showerscaleS
    else
      scales(1:ndelH,1:ndelH)=showerscaleH
    endif
    pb=0d0
    jpart=0
    icolalt=0
    need_matching=1
    need_matching(4)=-1
    do i=1,next
      pb(0,i)=100d0+i
      jpart(1,i)=i
      jpart(2:3,i)=1
      jpart(6,i)=1
      if(i.le.2)jpart(6,i)=-1
    enddo
    ! One resonance is inserted at output slot 3, moving its daughters.
    jpart(1,-1)=23
    jpart(2,-1)=1
    jpart(3,-1)=2
    jpart(6,-1)=2
    jpart(2:3,3:4)=-1
    call finish_event_metadata(next,1,1,jpart,icolalt,pb,scales,npart)
    call require(npart.eq.next+1,'one resonance in the event')
    call require(jpart(1,3).eq.23.and.all(jpart(2:3,4:5).eq.3),'resonance mother remapping')
    call write_events_lhe(pb(0,1),1d0,jpart(1,1),npart,99,ickkw,scales,1)
    call require(event_size.eq.npart,'LHE event size')
    call require(scalar.eq.expected,'SCALUP uses selected scalar, never a matrix diagonal')
    write(number,'(f16.5)') scalar
    call require(index(ptclusstring,'pt_clust_4="'//trim(adjustl(number))//'"').gt.0, &
         'required matching parton has SCALUP marker after remapping')
    write(number,'(f16.5)') scalar-0.001d0
    call require(index(ptclusstring,'pt_clust_5="'//trim(adjustl(number))//'"').gt.0, &
         'excluded matching parton has offset marker after remapping')
    call require(index(ptclusstring,'pt_clust_3=').eq.0,'resonance has no matching marker')
    call require(index(ptclusstring,'fxfxdelta=').eq.0,'both modes use ordinary FxFx markers')
    call require(index(ptclusstring,'fxfxmatch').eq.0,'no custom matching flags')
    call require(all(directed.eq.-1d0),'FxFx showers through SCALUP alone')
  enddo
  print *, 'PASS '//trim(mode)
contains
  subroutine require(condition,label)
    logical, intent(in) :: condition
    character(*), intent(in) :: label
    if(condition)return
    print *, 'FAIL: '//label
    stop 1
  end subroutine
end program

subroutine write_header_init
end subroutine

subroutine write_lhef_event(lun,nup,idprup,wgt,scalup,aqed,aqcd, &
     idup,istup,mothup,icolup,pup,vtimup,spinup,buff,scalup_a)
  use captured_lhe_scales
  implicit none
  integer :: lun,nup,idprup,idup(500),istup(500),mothup(2,500),icolup(2,500)
  double precision :: wgt,scalup,aqed,aqcd,pup(5,500),vtimup(500),spinup(500),scalup_a(500,500)
  character(*) :: buff
  scalar=scalup
  directed=scalup_a
  event_size=nup
end subroutine
