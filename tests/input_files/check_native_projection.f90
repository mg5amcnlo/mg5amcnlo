! Check the production native projection independently of Born chart sampling.
subroutine check_native_projection()
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mc_native_context, only: native_mapping
  use kinematics_module, only: boost_n1_to_lab
  implicit none
  include 'genps.inc'
  include 'nexternal.inc'
  include 'run.inc'
  double precision :: pmass(nexternal),pb(0:3,nexternal-1),pbl(0:3,nexternal-1),pbe(0:3,nexternal-1)
  common/to_mass/pmass
  common/pborn/pb
  common/pborn_l/pbl
  common/pborn_ev/pbe
  integer :: ifks,jfks,config,isign
  common/fks_indices/ifks,jfks
  common/to_mconfigs/config
  common/c_isolsign/isign
  double precision :: bounds(3),omx(2)
  common/ctau_lower_bound/bounds
  common/to_ee_omx1/omx
  logical :: nbody,only_event,skip_event,evpr,nocnt,fixed_order,nlo_ps
  common/cnbody/nbody
  common/c_skip_only_event_phsp/only_event,skip_event
  common/to_use_evpr/evpr
  common/cnocntevents/nocnt
  common/c_fnlo_nlops/fixed_order,nlo_ps
  double precision :: pc(0:3,nexternal,-2:2),wc(-2:2),psc(-2:2),jc(-2:2)
  common/counterevnts/pc,wc,psc,jc
  double precision :: xi,y,pi_ev(0:3),pi_cnt(0:3,-2:2),xi_cnt(-2:2),xih,xih_cnt(-2:2)
  common/fksvariables/xi,y,pi_ev,pi_cnt
  common/cxiifkscnt/xi_cnt
  common/cxi_i_hat/xih,xih_cnt
  double precision :: xb(2),xbc(2,-2:2),tau,ycm,tauc(-2:2),ycmc(-2:2)
  common/cbjorkenx/xb,xbc
  common/cbjrk12_ev/tau,ycm
  common/cbjrk12_cnt/tauc,ycmc
  double precision :: sqrts_ev,s_ev,sqrts_cnt(-2:2),s_cnt(-2:2)
  common/parton_cms_ev/sqrts_ev,s_ev
  common/parton_cms_cnt/sqrts_cnt,s_cnt
  double precision :: xmax,xmaxc(-2:2),xnorm,xnormc(-2:2),kn,knbar,kn0
  common/cxiimaxev/xmax
  common/cxiimaxcnt/xmaxc
  common/cxinormev/xnorm
  common/cxinormcnt/xnormc
  common/cgenps_fks/kn,knbar,kn0
  double complex :: spin
  common/cxij_aor/spin
  double precision :: x(99),m(-max_branch:max_particles),mb(nexternal-1),born(0:3,nexternal-1)
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
          bounds=sum(mb(3:4))**2/stot
          omx=0d0
          nbody=.false.
          only_event=.false.
          skip_event=.false.
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
          m=0d0
          call generate_FKS_kinematics(x,3,j0,ps0,stot,sborn,sqrtborn,taub,yb,yhat, &
               xbb,.false.,m,mb,jac,p,pass)
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
          bounds=bounds_poison
          omx=omx_poison
          nbody=.true.
          only_event=.true.
          skip_event=.true.
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
               error stop 'Born COMMON arrays not populated'
          if(any(bounds.ne.bounds_poison).or.any(omx.ne.omx_poison).or. &
               .not.nbody.or..not.only_event.or..not.skip_event)error stop 'input controls leaked'
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
          if(.not.all(ieee_is_finite(actual(1:n))))error stop 'nonfinite native COMMON state'
          if(any(abs(actual(1:n)-reference(1:n)).gt.1d-8*max(1d0,abs(reference(1:n))))) &
               error stop 'native COMMON state differs'
        enddo
      enddo
    enddo
  enddo
  if(negative_count.eq.0)error stop 'massive second solution not covered'
  lab(0,1)=-1d0
  call generate_native_momenta(lab,out,outlab,cms,jnew,pass)
  if(pass.or.jnew.ge.0d0.or.out(0,1).ge.0d0)error stop 'invalid input accepted'
  if(any(bounds.ne.bounds_poison).or.any(omx.ne.omx_poison).or. &
       .not.nbody.or..not.only_event.or..not.skip_event)error stop 'invalid input changed controls'
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
