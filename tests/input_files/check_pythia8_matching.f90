! Exercise production radiation functions, kernels and support. Input momenta
! and phase-space references are constructed independently by the Python test.
program check_pythia8_matching
  use fks_phase_space_data,only: bound_born => tau_Born_lower_bound, &
       bound_res => tau_lower_bound_resonance,bound_tau => tau_lower_bound,vk => veckn_ev, &
       vkb => veckbarn_ev,ve => xp0jfks,ylab => ybst_til_tolab,ycms => ybst_til_tocm,roots => sqrtshat, &
       shat
  use process_module, only: next_n1, nincoming_mod, mass_n, shower_mc_mod
  use kinematics_module
  use scale_module
  use fks_phase_space, only: invert_fks_radiation
  implicit none
  character(len=16) mode, shower
  integer ios, i, kind, np, ifks, jfks
  common /fks_indices/ ifks, jfks
  double precision pmass(6),omx(2)
  common /to_mass/ pmass
  common /to_ee_omx1/ omx
  logical softtest, colltest, zone(2)
  common /sctests/ softtest, colltest
  double precision ch_i, ch_j, ch_m
  integer i_type, j_type, m_type, j_pdg
  common /cparticle_types/ ch_i, ch_j, ch_m, i_type, j_type, m_type, j_pdg
  integer fks_j_from_i(6,0:6), particle_type(6), pdg_type(6)
  common /c_fks_inc/ fks_j_from_i, particle_type, pdg_type
  double precision g, alsf, besf, alazi, beazi
  double complex gal(2)
  common /test_couplings/ g, gal
  common /cgfunsfp/ alsf, besf
  common /cgfunazi/ alazi, beazi
  double precision p(0:3,6), pb(0:3,-8:5), m, mr, rad, delta, k
  double precision rt(3), jac, ps, tau, yb, xb(2), z, t, jz, f, hij
  double precision kernel(2), azimuth(2), damping(2), qmc, weight, max1, max2, e0sq
  double precision zpy8, xipy8, xjacpy8, xfact_ileg12, xfact_ileg3, xfact_ileg4
  double precision fks_hij, compute_damping_weight, py8_gluon_recoil_weight
  external zpy8, xipy8, xjacpy8, xfact_ileg12, xfact_ileg3, xfact_ileg4, fks_hij
  external compute_damping_weight, py8_gluon_recoil_weight

  call get_command_argument(1,mode)
  next_n1=6
  nincoming_mod=2
  allocate(mass_n(5))
  mass_n=0d0
  bound_born=1d-12
  bound_res=1d-12
  bound_tau=1d-12
  omx=0d0
  pmass=0d0
  softtest=.false.
  colltest=.false.
  ifks=6
  jfks=3
  g=1d0
  gal=cmplx(1d0,0d0,kind=8)
  alsf=1d0
  besf=-0.1d0
  alazi=-1d0
  beazi=-0.1d0
  ylab=0d0
  ycms=0d0
  shat=1d6
  roots=1d3
  fks_j_from_i=0
  pdg_type=0
  particle_type=8
  shower_mc_mod='PYTHIA8'

  do
    if (mode.eq.'weight') then
      read(*,*,iostat=ios) z, shat_n1, mr, k
      if (ios.ne.0) exit
      write(*,'(ES25.16)') py8_gluon_recoil_weight(z,shat_n1,mr,k)
      cycle
    elseif (mode.eq.'radiation') then
      read(*,*,iostat=ios) ileg, m, mr, rad, delta, k
      if (ios.ne.0) exit
      shat_n1=1d6
      x=1d0-rad
      yij=1d0-delta
      rad=1d0-x
      delta=1d0-yij
      kn=k
      kn0=sqrt(k*k+m*m)
      knbar=sqrt((shat_n1+m*m-mr*mr)**2/(4d0*shat_n1)-m*m)
      vk=kn
      vkb=knbar
      ve=kn0
      if (ileg.eq.3) then
        xm12=m*m
        xm22=mr*mr
        w1=sqrt(shat_n1)*rad*(m*m/(kn0+kn)+delta*kn)
        w2=shat_n1*rad-w1
      else
        xm12=mr*mr
        xm22=0d0
        w2=sqrt(shat_n1)*rad*kn*delta
        w1=shat_n1*rad-w2
        xij=2d0*kn/sqrt(shat_n1)
      endif
      betas=1d0+(xm12-xm22)/shat_n1
      betad=sqrt((1d0-(xm12-xm22)/shat_n1)**2-4d0*xm22/shat_n1)
      z=zpy8()
      t=xipy8(z)
      jz=xjacpy8(z)
      if (ileg.le.2) then
        f=xfact_ileg12(1)
      elseif (ileg.eq.3) then
        f=xfact_ileg3(1)
      else
        f=xfact_ileg4(1)
      endif
      write(*,'(5ES25.16)') z,t,jz,f,get_qmc(rad,yij)**2
      cycle
    elseif (mode.ne.'event'.and.mode.ne.'measure') then
      stop 1
    endif

    read(*,*,iostat=ios) m, kind, shower, max1, max2
    if (ios.ne.0) exit
    do i=1,6
      read(*,*) p(:,i)
    enddo
    shower_mc_mod=shower
    pmass=0d0
    pmass(3)=m
    mass_n=0d0
    mass_n(3)=m
    jac=1d0
    ps=1d0
    call invert_fks_radiation(rt,jac,ps,4d6,tau,yb,xb,p,pb)
    if (jac.le.0d0) stop 2
    rad=get_xi_from_p(6,3,p)
    delta=get_yij_from_p(6,3,p)
    vk=rho(p(:,3))
    vkb=rho(pb(:,3))
    ve=p(0,3)
    call fill_kinematics_module(p,6,3,rad,delta,m,.true.)
    if (mode.eq.'measure') then
      shower_scale_nbody_max=1000d0
      qmc=get_qmc(rad,delta)
      f=xfact_ileg3(1)
      do i=1,5
        if (i.eq.3) cycle
        e0sq=dot(pb(:,3),pb(:,i))
        call get_shower_variables(e0sq,z,t,jz)
        call get_dead_zone(z,t,pb(:,1:5),qmc,i,zone(1),weight)
        write(*,'(9ES25.16)') z,t,jz,f,rad,delta,e0sq,qmc,merge(1d0,0d0,zone(1))
      enddo
      cycle
    endif
    ch_i=0d0
    ch_j=0d0
    ch_m=0d0
    i_type=8
    j_type=8
    m_type=8
    j_pdg=21
    np=2
    if (kind.eq.2.or.kind.eq.3) then
      j_type=3
      m_type=3
      j_pdg=1
      if (kind.eq.3) j_pdg=1000001
      np=1
    elseif (kind.eq.4) then
      j_pdg=1000021
    elseif (kind.eq.5) then
      i_type=3
      j_type=-3
    endif
    particle_type(3)=j_type
    particle_type(6)=i_type
    z=zpy8()
    t=xipy8(z)
    jz=xjacpy8(z)
    if (ileg.eq.3) then
      f=xfact_ileg3(np)
    else
      f=xfact_ileg4(np)
    endif
    call limits(rad,delta)
    ! A one-mother/MEC context is outside the audited hard-system contract.
    if (kind.eq.6) nincoming_mod=1
    call compute_splitting_kernels(kernel,azimuth,z,t,jz)
    nincoming_mod=2
    hij=fks_hij(p,6,3)
    shower_scale_nbody_min=0d0
    shower_scale_nbody_max=1000d0
    shower_scale_nbody_max(3,1)=max1
    shower_scale_nbody_max(3,2)=max2
    qmc=get_qmc(rad,delta)
    do i=1,2
      call get_dead_zone(z,t,pb(:,1:5),qmc,i,zone(i),weight)
      damping(i)=compute_damping_weight(i,rad,delta)
    enddo
    mr=xm12
    if (ileg.eq.3) mr=xm22
    write(*,'(19ES25.16)') z,t,jz,f,rad,delta,kernel(1),azimuth(1), &
      kernel(2),azimuth(2),hij,merge(1d0,0d0,zone(1)), &
      merge(1d0,0d0,zone(2)),damping,mr,gfactsf,gfactcl,gfactazi
  enddo
end program
