! Check the production native projection independently of Born chart sampling.
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
