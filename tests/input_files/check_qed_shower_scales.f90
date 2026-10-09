program check_qed_shower_scales
  use process_module
  use scale_module
  implicit none
  double precision :: born(0:3,4),realp(0:3,5),masses(4),charges(4)
  double precision :: saved_hard
  logical :: colour_dipoles(4,4,2),real_colour_dipoles(5,5)
  integer :: pdg(4),partner,flow

  call init_process_module_global('PYTHIA8   ','all ',5,2,.false.,100d0,2,1,0)
  call init_scale_module(5,1d0,2,2)
  pdg=[11,-11,13,-13]
  charges=[-1d0,1d0,-1d0,1d0]
  masses=0d0
  colour_dipoles=.false.
  call init_process_module_nbody(4,masses,[1,1,1,1],2,colour_dipoles,pdg,charges)
  born(:,1)=[50d0,0d0,0d0,50d0]
  born(:,2)=[50d0,0d0,0d0,-50d0]
  born(:,3)=[50d0,50d0,0d0,0d0]
  born(:,4)=[50d0,-50d0,0d0,0d0]
  qed_matching=.true.
  do flow=1,2
    call compute_shower_scale_nbody(born,flow,0d0)
    call require(abs(shower_scale_hard-55d0).lt.1d-12,'scalar S scale')
    call require(abs(shower_scale_nbody(1,2)-55d0).lt.1d-12,'lepton ISR scale')
    call require(abs(shower_scale_nbody(3,4)-50d0).lt.1d-12,'lepton FSR dipole cap')
    call require(shower_scale_nbody(3,1).lt.0d0,'directed recoil mask')
    call require(shower_scale_nbody_min(3,4).eq.10d0,'uncapped lower damping bound')
    call require(shower_scale_nbody_max(3,4).eq.100d0,'uncapped upper damping bound')
    call determine_partner(flow,3,partner)
    call require(partner.eq.4,'partner independent of colour flow')
  enddo
  call save_shower_scale_nbody(1,1,3,partner)
  saved_hard=emsca_S_hard(1,1)

  ! A photon Born state also has a directed recoil scale for conversion.
  pdg(3:4)=22
  charges(3:4)=0d0
  call init_process_module_nbody(4,masses,[1,1,1,1],2,colour_dipoles,pdg,charges)
  call compute_shower_scale_nbody(born,-3,0d0)
  call determine_partner(1,3,partner)
  call require(partner.eq.1,'photon charge-weighted beam recoil')
  call require(shower_scale_nbody(3,partner).gt.0d0,'photon conversion scale')

  ! A real photon remains colourless; QED H scales must not inherit -1.
  realp(:,1:2)=born(:,1:2)
  realp(:,3)=[40d0,-40d0,0d0,0d0]
  realp(:,4)=[30d0,20d0,sqrt(500d0),0d0]
  realp(:,5)=[30d0,20d0,-sqrt(500d0),0d0]
  real_colour_dipoles=.false.
  call init_process_module_n1body(5,[0d0,0d0,0d0,0d0,0d0],[1,1,1,1,1], &
       1,real_colour_dipoles,[11,-11,13,-13,22],[-1d0,1d0,-1d0,1d0,0d0])
  call compute_shower_scale_n1body(realp,5,3,0d0)
  call require(all(pack(shower_scale_n1body,qed_dipole_n1).gt.0d0),'real QED dipole scales')
  call require(shower_scale_n1body(3,5).gt.0d0,'emitter-photon H scale')
  call require(all(.not.valid_dipole_n1),'QED leaves event colour dipoles unchanged')
  call require(emsca_S_hard(1,1).eq.saved_hard,'H evaluation preserves saved S scale')

  ! Both QED daughters use mu+ as recoiler. The fermion invariant is
  ! smaller and must not be skipped after finding the photon connection.
  realp(:,3)=[32d0,-23d0,sqrt(495d0),0d0]
  realp(:,4)=[28d0,-17d0,-sqrt(495d0),0d0]
  realp(:,5)=[40d0,40d0,0d0,0d0]
  call compute_shower_scale_n1body(realp,5,3,0d0)
  call require(qed_dipole_n1(3,4).and.qed_dipole_n1(5,4),'shared daughter recoiler')
  call require(abs(shower_scale_n1body(3,5)-sqrt(2000d0)).lt.1d-12, &
       'common H scale includes both daughter invariants')

  ! QCD uses its own mask after changing correction order, with no QED leak.
  qed_matching=.false.
  call Bornonly_shower_scale(born,1)
  call require(all(shower_scale_nbody.lt.0d0),'QCD does not use stale QED dipoles')

  ! Charged W resonances must reach the same S/H plumbing as leptons.
  call init_process_module_global('PYTHIA8   ','all ',5,2,.false.,200d0,2,1,0)
  qed_matching=.true.
  pdg=[11,-11,24,-24]
  charges=[-1d0,1d0,1d0,-1d0]
  masses=[0d0,0d0,80d0,80d0]
  call init_process_module_nbody(4,masses,[1,1,1,1],2,colour_dipoles,pdg,charges)
  born(:,1)=[100d0,0d0,0d0,100d0]
  born(:,2)=[100d0,0d0,0d0,-100d0]
  born(:,3)=[100d0,60d0,0d0,0d0]
  born(:,4)=[100d0,-60d0,0d0,0d0]
  call compute_shower_scale_nbody(born,1,80d0)
  call determine_partner(1,3,partner)
  call require(partner.eq.4,'W-pair reference partner')
  call require(abs(shower_scale_nbody(3,4)-sqrt(2000d0)).lt.1d-12, &
       'massive W S dipole cap')
  call require(abs(shower_scale_nbody(4,3)-sqrt(2000d0)).lt.1d-12, &
       'charge-conjugate W S dipole cap')
  realp(:,1:2)=born(:,1:2)
  realp(:,3)=[90d0,-10d0,40d0,0d0]
  realp(:,4)=[90d0,-10d0,-40d0,0d0]
  realp(:,5)=[20d0,20d0,0d0,0d0]
  call init_process_module_n1body(5,[0d0,0d0,80d0,80d0,0d0],[1,1,1,1,1], &
       1,real_colour_dipoles,[11,-11,24,-24,22],[-1d0,1d0,1d0,-1d0,0d0])
  call compute_shower_scale_n1body(realp,5,3,80d0)
  call require(qed_dipole_n1(3,4).and.qed_dipole_n1(4,3),'real W recoil routing')
  call require(all(pack(shower_scale_n1body,qed_dipole_n1).gt.0d0),'real W QED dipole scales')
  call require(abs(shower_scale_n1body(3,5)-sqrt(4000d0)).lt.1d-12, &
       'W-photon H scale')
  call require(all(.not.valid_dipole_n1),'W radiation leaves event colour dipoles unchanged')
  print *, 'PASS QED S/H scales and colour-flow independence'
contains
  subroutine require(condition,message)
    logical,intent(in)::condition
    character(*),intent(in)::message
    if (.not.condition) then
      print *,message
      stop 1
    endif
  end subroutine
end program

double precision function ran2()
  ran2=0.5d0
end function

subroutine cluster_and_reweight(iproc,a,b,nscale,scales,fac,matching,for_shower)
  use process_module,only:qed_matching
  implicit none
  integer :: iproc,nscale,matching(*)
  double precision :: a,b,scales(0:*),fac(*)
  logical :: for_shower
  if (qed_matching) stop 'QED must not require QCD jet clustering'
  nscale=0
  scales(0)=100d0
end subroutine
