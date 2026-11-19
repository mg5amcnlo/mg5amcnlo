program check_pythia_local_matching
  use mc_counterterms
  use fks_phase_space_data
  use fks_phase_space, only: fks_born_point,fks_phase_space_point,generate_fks_radiation
  use fks_phase_space_helpers, only: dot
  use mc_native_context, only: native_mapping
  use FKSParams
  use scale_module
  implicit none
  include 'coupl.inc'
  integer i,j,a,f,kind,nconnect,ifks,jfks,particle(5),pdg(5),partners(5,0:5)
  common/fks_indices/ifks,jfks
  common/c_fks_inc/partners,particle,pdg
  logical aorg(5),charged(5),softtest,colltest,fixed_order,nlo_ps,pass
  common/c_is_aorg/aorg
  common/c_is_charged/charged
  common/sctests/softtest,colltest
  common/c_fnlo_nlops/fixed_order,nlo_ps
  double precision pmass(5),p(0:3,5),mass,zout(2),weights(1,2),xborn(2),lum, &
       v,smin,smax
  common/to_mass/pmass
  double precision ch_i,ch_j,ch_m
  integer i_type,j_type,m_type,j_pdg
  common/cparticle_types/ch_i,ch_j,ch_m,i_type,j_type,m_type,j_pdg
  double precision ebeam(2),xbk(2)
  integer lpp(2)
  common/test_run/ebeam,xbk,lpp
  double precision rad(3),s,k,e,theta,phi,raw,soft,total_raw,total_soft, &
       n(0:3),current,allcurrent,y,fks_Sij,soft_prefactor,reference(0:3),energy_error
  external fks_Sij
  double precision fr,fs,fc,fdc,fsc,fdsc(4),mc_factors(8),delta_used,xicut_used, &
       xiScut_used,xiBSVcut_used,symmetry,symmetry_b,symmetry_d
  integer ngluons,nquarks(-6:6),nphotons
  common/factor_n1body/fr,fs,fc,fdc,fsc,fdsc
  common/factor_n1body_NLOPS/mc_factors
  common/cdelta_used/delta_used
  common/cxicut_used/xicut_used
  common/cxiScut_used/xiScut_used,xiBSVcut_used
  common/numberofparticles/symmetry,symmetry_b,symmetry_d,ngluons,nquarks,nphotons
  type(fks_born_point) born
  type(fks_phase_space_point) point
  character(24) mode,kernel_choice,profile_choice
  call get_command_argument(1,mode)
  call get_command_argument(2,kernel_choice)
  call get_command_argument(3,profile_choice)
  if(len_trim(profile_choice).gt.0)read(profile_choice,*)PythiaLocalGProfile
  FKSPythiaLocalRecoil=1
  PythiaLocalCollinearPatch=kernel_choice.eq.'patch'.or.kernel_choice.eq.'2patch'
  PythiaLocalSoftPatch=kernel_choice.eq.'2patch'
  FKSExplicitSum=.true.
  MCExplicitKLSum=.true.
  g=1d0
  gal=0d0
  ch_i=0d0
  ch_j=0d0
  ch_m=0d0
  m_type=3
  j_pdg=1
  softtest=.false.
  colltest=.false.
  fixed_order=.false.
  nlo_ps=.true.
  native_mapping=.false.
  pmass=0d0
  shower_scale_nbody_min=1d8
  shower_scale_nbody_max=2d8
  ifks=5
  lpp=1
  if(mode.eq.'profile')then
     read(*,*)PythiaLocalGWidth,v
     write(*,'(es26.17e3)')pythia_local_gprofile(v)
  elseif(mode.eq.'parameters')then
     call FKSParamReader('FKS_params.dat',.false.,.true.)
     j=get_fks_isr_mapping(.true.,'PYTHIA8')
     write(*,*)'PROFILE',PythiaLocalGProfile
  elseif(mode.eq.'density'.or.mode.eq.'matched_density'.or.mode.eq.'matched_factors')then
     read(*,*)a,f,jfks,kind,mass,xborn
     if(mode.ne.'density')then
        read(*,*)PythiaLocalGWidth,smin,smax
        shower_scale_nbody_min=smin
        shower_scale_nbody_max=smax
     endif
     do i=1,5
        read(*,*)p(:,i)
     enddo
     pmass(f)=mass
     pythia_recoil_pair=[a,f]
     i_type=8
     j_type=3
     if(kind.eq.2)then
        i_type=3
        j_type=8
     endif
     call prepare_pythia_local_kinematics(p,ifks,jfks)
     y=sum(p(1:3,jfks)*p(1:3,5))/ &
          sqrt(sum(p(1:3,jfks)**2)*sum(p(1:3,5)**2))
     call compute_pythia_local_counterterm([1d0],zout,nconnect,weights, &
          mode.ne.'density',y)
     call get_mc_lum(jfks,zout(1),.37d0,lum,xborn)
     if(mode.eq.'matched_factors')then
        write(*,'(14es26.17e3)')weights(1,1),local_z,local_q2,local_d, &
             local_zeta,local_w,local_pt2,local_chi,xbk,lum,gfactsf,gfactcl,gfactazi
     else
        write(*,'(12es26.17e3)')weights(1,1),local_z,local_q2,local_d, &
             local_zeta,local_w,local_pt2,local_chi,xbk,lum,gfactsf
     endif
  elseif(mode.eq.'soft'.or.mode.eq.'common_soft')then
     read(*,*)mass,rad
     s=1d6
     e=sqrt(s)/2d0
     k=(s-mass**2)/(2d0*sqrt(s))
     theta=.6d0
     phi=.4d0
     born%valid=.true.
     born%stot=s/.024d0
     born%shat=s
     born%sqrtshat=sqrt(s)
     born%tau=.024d0
     born%xbjrk=[.12d0,.20d0]
     born%ycm=log(.12d0/.20d0)/2d0
     born%mass=[0d0,0d0,mass,0d0]
     born%p(:,1)=[e,0d0,0d0,e]
     born%p(:,2)=[e,0d0,0d0,-e]
     born%p(:,3)=[(s+mass**2)/(2d0*sqrt(s)),k*sin(theta)*cos(phi), &
          k*sin(theta)*sin(phi),k*cos(theta)]
     born%p(:,4)=[k,-born%p(1:3,3)]
     ebeam=sqrt(born%stot)/2d0
     pmass(3)=mass
     delta_used=.7d0
     xicut_used=.35d0
     xiScut_used=.5d0
     xiBSVcut_used=1d0
     symmetry=1d0
     symmetry_b=1d0
     symmetry_d=1d0
     ngluons=1
     nquarks=0
     nphotons=0
     partners=0
     partners(5,0)=4
     partners(5,1:4)=[1,2,3,4]
     particle=[3,3,3,3,8]
     pdg=[5,2,6,1,21]
     aorg=.false.
     aorg(5)=.true.
     charged=.false.
     i_type=8
     j_type=3
     total_raw=0d0
     total_soft=0d0
     energy_error=0d0
     reference=0d0
     do j=1,4
        jfks=j
        a=1
        f=3
        if(j.eq.2.or.j.eq.4)then
           a=2
           f=4
        endif
        pythia_recoil_pair=[a,f]
        initial_recoil_leg=0
        resonance_recoil=.false.
        resonance_members=.false.
        if(j.gt.2)then
           initial_recoil_leg=a
           resonance_recoil=.true.
           resonance_members(j)=.true.
           resonance_members(5)=.true.
        endif
        raw=1d0
        call generate_fks_radiation(born,rad,.false.,raw,point,pass)
        if(.not.pass)error stop 'correlated radiation rejected'
        energy_error=max(energy_error,abs(point%p(0,5)/(rad(1)**2*sqrt(s)/2d0)-1d0))
        n=p_i_fks_cnt(:,0)/p_i_fks_cnt(0,0)
        if(j.eq.1)reference=n
        if(maxval(abs(n-reference)).gt.1d-10)error stop 'different soft directions'
        call prepare_pythia_local_kinematics(point%p,ifks,jfks)
        call compute_pythia_local_counterterm([1d0],zout,nconnect,weights, &
             mode.eq.'common_soft',y_ij_fks_ev)
        if(mode.eq.'common_soft'.and.gfactcl.ne.1d0) &
             error stop 'soft completion check needs wide angles'
        call set_cms_stuff(-100)
        call compute_prefactors_n1body(1d0,raw)
        total_raw=total_raw+weights(1,1)*xi_i_fks_ev**2* &
             (1d0-y_ij_fks_ev)*mc_factors(7)
        call set_cms_stuff(0)
        allcurrent=0d0
        n=p_i_fks_cnt(:,0)
        do i=1,2
           f=i+2
           current=dot(p1_cnt(:,i,0),p1_cnt(:,f,0))/ &
                (dot(p1_cnt(:,i,0),n)*dot(p1_cnt(:,f,0),n))- &
                pmass(f)**2/(2d0*dot(p1_cnt(:,f,0),n)**2)
           allcurrent=allcurrent+current
        enddo
        y=y_ij_fks_ev
        soft=2d0*g*g*(4d0/3d0)*allcurrent*(1d0-y)* &
             fks_Sij(p1_cnt(:,:,0),5,j,0d0,y)
        soft_prefactor=xinorm_ev/xi_i_fks_ev/(1d0-y)
        if(j.le.2)soft_prefactor=xinorm_ev/pythia_isr_coordinate/(1d0-y)
        total_soft=total_soft+soft*soft_prefactor*jac_cnt(0)
        if(mode.eq.'common_soft')total_raw=total_raw+ &
             (1d0-gfactsf)*soft*soft_prefactor*jac_cnt(0)
     enddo
     write(*,'(4es26.17e3)')total_raw,total_soft,(total_raw-total_soft)/total_soft,energy_error
  else
     error stop 'unknown mode'
  endif
end program
