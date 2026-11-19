program check_pythia_local_recoil
  use fks_radiation_maps, only: generate_momenta_pythia_initial, invert_momenta_pythia_initial
  use fks_phase_space_helpers, only: dot,boost_n1_to_lab
  use fks_phase_space_data
  use fks_phase_space, only: generate_FKS_kinematics,generate_native_momenta, &
       fks_phase_space_point,capture_fks_phase_space,restore_fks_phase_space
  use mc_native_context, only: native_mapping
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  integer beam,partner,ifks,jfks,i,slot,n,ix,iy,iz,emitted
  common/fks_indices/ifks,jfks
  logical softtest,colltest,pass,use_evpr
  common/sctests/softtest,colltest
  common/to_use_evpr/use_evpr
  double precision pmass(5)
  common/to_mass/pmass
  double precision ebeam(2),xbk(2)
  integer lpp(2)
  common/test_run/ebeam,xbk,lpp
  double precision p(0:3,5),born(0:3,5),born4(0:3,4),pb(0:3,4),xbar(2),rad(3),mass,theta,phi0, &
       e,k,s,sroot,tau0,ycm0,xbj(2),phat(0:3),xim,xnorm,xi,y,xihat,ps,jac, &
       xa,w,mb(4),plab(0:3,5),pcm(0:3,5),pout(0:3,5),poutlab(0:3,5),jagain, &
       q(0:3),m2,r,reference,volume,saved_w,saved_scale,xi_fix,y_fix
  common/cxiyfix/xi_fix,y_fix
  double precision,allocatable:: nodes(:),weights(:)
  double precision fr,fs,fc,fdc,fsc,fdsc(4),saved_coll(7),delta_used,xicut_used, &
       xiScut_used,xiBSVcut_used,symmetry,symmetry_b,symmetry_d,observed,scale
  integer ngluons,nquarks(-6:6),nphotons
  common/factor_n1body/fr,fs,fc,fdc,fsc,fdsc
  common/cdelta_used/delta_used
  common/cxicut_used/xicut_used
  common/cxiScut_used/xiScut_used,xiBSVcut_used
  common/numberofparticles/symmetry,symmetry_b,symmetry_d,ngluons,nquarks,nphotons
  double precision,parameter:: pi=3.1415926535897932384626433832795d0
  type(fks_phase_space_point) snapshot
  character(32) mode,argument
  call get_command_argument(1,mode)
  read(*,*)beam,partner,mass,rad
  ifks=5
  call get_command_argument(2,argument)
  if(len_trim(argument).gt.0)read(argument,*)ifks
  jfks=beam
  native_mapping=.true.
  softtest=.false.
  colltest=.false.
  use_evpr=.true.
  s=1d6
  sroot=sqrt(s)
  e=sroot/2d0
  k=(s-mass**2)/(2d0*sroot)
  theta=.6d0
  phi0=.4d0
  born=0d0
  born(:,1)=[e,0d0,0d0,e]
  born(:,2)=[e,0d0,0d0,-e]
  born(:,3)=[(s+mass**2)/(2d0*sroot),k*sin(theta)*cos(phi0), &
       k*sin(theta)*sin(phi0),k*cos(theta)]
  born(:,4)=[k,-born(1:3,3)]
  born4=born(:,1:4)
  if(ifks.ne.5)then
     born=0d0
     do i=1,4
        emitted=i
        if(i.ge.ifks)emitted=i+1
        born(:,emitted)=born4(:,i)
     enddo
     if(partner.ge.ifks)partner=partner+1
  endif
  xbar=[.12d0,.20d0]
  tau0=product(xbar)
  ycm0=log(xbar(1)/xbar(2))/2d0
  ebeam=sroot/(2d0*sqrt(tau0))
  lpp=1
  pmass=0d0
  emitted=3
  if(emitted.ge.ifks)emitted=emitted+1
  pmass(emitted)=mass
  tau_Born_lower_bound=mass**2/(4d0*product(ebeam))
  tau_lower_bound=tau_Born_lower_bound
  tau_lower_bound_resonance=tau_Born_lower_bound
  pythia_recoil_pair=[beam,partner]
  initial_recoil_leg=0
  resonance_recoil=.false.
  if(mode.eq.'native_point')then
     ebeam=6500d0
     do i=1,5
        read(*,*)plab(:,i)
     enddo
     call generate_native_momenta(plab,pout,poutlab,pcm,jagain,pass)
     if(.not.pass)error stop 'native corner inverse'
     if(maxval(abs(poutlab-plab)).gt.1d-4*maxval(abs(plab)))error stop 'native corner replay'
     write(*,*)'PASS native point'
  elseif(mode.eq.'invalid')then
     p=born
     p(1,partner)=ieee_value(0d0,ieee_quiet_nan)
     jac=1d0
     ps=1d0
     call generate_momenta_pythia_initial(-100,ifks,beam,partner,xbar,tau0,ycm0,s, &
          2d0*pi*rad(3),p,rad(1:2),shat,sqrtshat,tau_ev,ycm_ev,xbj,phat, &
          xim,xnorm,xi,y,xihat,ps,jac,pass)
     if(pass.or.jac.ge.0d0)error stop 'accepted nonfinite Born point'
     rad(1)=2d0
     call forward(-100)
     if(pass.or.jac.ge.0d0)error stop 'accepted invalid beam fraction'
     write(*,*)'PASS invalid'
  elseif(mode.eq.'volume')then
     read(*,*)n
     allocate(nodes(n),weights(n))
     do i=1,n
        read(*,*)nodes(i),weights(i)
     enddo
     volume=0d0
     do ix=1,n
        do iy=1,n
           do iz=1,n
              rad=[nodes(ix),nodes(iy),nodes(iz)]
              call forward(-100)
              if(.not.pass)error stop 'volume map rejected'
              volume=volume+weights(ix)*weights(iy)*weights(iz)* &
                   jac*ps*xi*xnorm*2d0*pi
           enddo
        enddo
     enddo
     q=born(:,partner)+(1d0-xbar(beam))/xbar(beam)*born(:,beam)
     m2=dot(q,q)
     r=max(0d0,dot(born(:,partner),born(:,partner)))/m2
     reference=1d0-r
     if(r.gt.0d0)reference=reference+r*log(r)
     reference=reference*m2/(16d0*pi**2)
     write(*,'(2es26.17e3)')volume,reference
  elseif(mode.eq.'production'.or.mode.eq.'endpoints')then
     p_born_l=born4
     p_born=p_born_l
     mb=[0d0,0d0,mass,0d0]
     jac=1d0
     call generate_FKS_kinematics(rad,.false.,1d0,1d0,4d0*product(ebeam), &
          s,sroot,tau0,ycm0,xbar,mb,jac,p,pass)
     if(.not.pass.or.jac.le.0d0)error stop 'production generation'
     do slot=0,2
        if(jac_cnt(slot).le.0d0)error stop 'missing local ISR counterevent'
     enddo
     if(maxval(abs(p1_cnt(:,:,0)-born)).gt.1d-9)error stop 'soft projection'
     if(maxval(abs(p1_cnt(:,partner,1)-born(:,partner))).gt.1d-9)error stop 'collinear partner moved'
     if(mode.eq.'endpoints')then
        delta_used=.7d0
        xicut_used=.35d0
        xiScut_used=.5d0
        xiBSVcut_used=1d0
        symmetry=1d0
        symmetry_b=1d0
        symmetry_d=1d0
        ngluons=0
        nquarks=0
        nphotons=0
        call set_cms_stuff(-100)
        call compute_prefactors_n1body(1d0,jac)
        scale=dot(born(:,partner),p_i_fks_cnt(:,0))/dot(born(:,partner),born(:,beam))
        w=pythia_isr_coordinate
        observed=log(xicut_used/min(xiimax_ev,xiScut_used))- &
             min(xiimax_ev,xiScut_used)*(fs*(1d0-y_ij_fks_ev)/ &
             (jac_cnt(0)*xinorm_ev)-1d0/w)
        ! This is the finite log(a) left by the dimensionally regulated
        ! energy integral. It must not disappear with the pole subtraction.
        if(abs(observed+log(scale)).gt.2d-10)error stop 'local soft finite conversion'
        saved_coll=[fc,fdc,fsc,fdsc]
        pythia_recoil_pair=0
        call generate_FKS_kinematics(rad,.false.,1d0,1d0,4d0*product(ebeam), &
             s,sroot,tau0,ycm0,xbar,mb,jac,p,pass)
        if(.not.pass)error stop 'reference ISR generation'
        call set_cms_stuff(-100)
        call compute_prefactors_n1body(1d0,jac)
        if(maxval(abs(saved_coll-[fc,fdc,fsc,fdsc]))/maxval(abs(saved_coll)).gt.1d-10) &
             error stop 'collinear/PDF prefactors changed'
        write(*,'(2es26.17e3)')observed,scale
        stop
     endif
     saved_w=pythia_isr_coordinate
     saved_scale=pythia_isr_soft_scale
     call capture_fks_phase_space(snapshot)
     pythia_recoil_pair=0
     pythia_isr_coordinate=-5d0
     pythia_isr_soft_scale=-7d0
     call restore_fks_phase_space(snapshot)
     if(any(pythia_recoil_pair.ne.[beam,partner]))error stop 'pair restore'
     if(pythia_isr_coordinate.ne.saved_w.or.pythia_isr_soft_scale.ne.saved_scale) &
          error stop 'soft coordinate restore'
     call boost_n1_to_lab(p,plab,-ycm0)
     call generate_native_momenta(plab,pout,poutlab,pcm,jagain,pass)
     if(.not.pass)error stop 'local native inverse'
     if(maxval(abs(p_born_l-born4)).gt.1d-7)error stop 'native Born projection'
     write(*,*)'PASS production'
  else
     call forward(-100)
     if(.not.pass)error stop 'local map rejected'
     call invert_momenta_pythia_initial(p,ifks,beam,partner,pmass(partner), &
          xbj(beam),pb,xa,w,pass)
     if(.not.pass)error stop 'local inverse rejected'
     if(maxval(abs(pb-born4)).gt.2d-7)error stop 'inverse momenta'
     if(abs(xa-xbar(beam)).gt.1d-11)error stop 'inverse fraction'
     write(*,'(8es26.17e3)')shat,xi,pythia_isr_coordinate,y,ps,jac,pythia_isr_soft_scale,xa
     do i=1,5
        write(*,'(4es26.17e3)')born(:,i)
     enddo
     do i=1,5
        write(*,'(4es26.17e3)')p(:,i)
     enddo
  endif
contains
  subroutine forward(counter)
    integer,intent(in)::counter
    p=born
    jac=1d0
    ps=1d0
    call generate_momenta_pythia_initial(counter,ifks,beam,partner,xbar,tau0,ycm0,s, &
         2d0*pi*rad(3),p,rad(1:2),shat,sqrtshat,tau_ev,ycm_ev,xbj,phat, &
         xim,xnorm,xi,y,xihat,ps,jac,pass)
  end subroutine
end program
