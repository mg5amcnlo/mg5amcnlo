! Test the production phase-space routines without matrix elements or PDFs.
program check_momentum_maps
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mc_native_context, only: native_mapping
  use process_module, only: next_n1,nincoming_mod
  use kinematics_module, only: boost_n1_to_its_cms,boost_n1_to_lab, &
       get_xi_from_p,get_yij_from_p,get_phi_from_p,fill_father_and_ileg
  implicit none
  include 'genps.inc'
  include 'nexternal.inc'
  character(len=16) :: mode
  double precision, parameter :: pi=3.1415926535897932d0
  double precision :: p(0:3,5),plab(0:3,5),pcm(0:3,5),out(0:3,5)
  double precision :: born(0:3,4),invborn(0:3,-max_branch:4)
  double precision :: rnd(3),inv(3),jac,jacinv,pswgt,pswgtinv,jacout
  double precision :: shat,sqrtshat,mass,mrec,energy,momentum,phi,xi,yij
  double precision :: xiimax,xinorm,xihat,rat_xi,pifks(0:3),rapidity,error
  double precision :: tau_cnt(-2:2),ycm_cnt(-2:2)
  common /cbjrk12_cnt/tau_cnt,ycm_cnt
  logical :: softtest,colltest,pass
  common /sctests/softtest,colltest
  integer :: isign,i,j,k,nplus,nminus,nchecked,father
  integer :: nratios=0
  double precision, parameter :: xs(7)=[1d-4,0.01d0,0.25d0,0.6d0,0.9d0,0.97d0,0.999d0]
  double precision, parameter :: ys(4)=[0.2d0,0.7d0,0.9d0,0.999d0]
  double precision, parameter :: energies(3)=[400d0,1000d0,4000d0]

  call get_command_argument(1,mode)
  next_n1=5
  nincoming_mod=2
  softtest=.false.
  colltest=.false.
  if (mode.eq.'boost') then
     do i=-12,12,3
        p=0d0
        p(0,1)=6000d0
        p(3,1)=p(0,1)
        p(0,2)=6000d0*10d0**i
        p(3,2)=-p(0,2)
        call boost_n1_to_its_cms(p,pcm,rapidity)
        if (.not.all(ieee_is_finite(pcm))) error stop 'nonfinite CMS boost'
        energy=sqrt(p(0,1)*p(0,2))
        if (maxval(abs(pcm(0,1:2)-energy))/energy.gt.1d-12) &
             error stop 'asymmetric beam CMS energy'
        call boost_n1_to_lab(pcm,plab,-rapidity)
        if (.not.all(ieee_is_finite(plab))) error stop 'nonfinite lab boost'
        if (maxval(abs(plab-p))/maxval(abs(p)).gt.1d-12) &
             error stop 'asymmetric beam round trip'
     enddo
     write(*,*) 'PASS boost'
     stop
  endif

  if (mode.eq.'massive_recoil'.or.mode.eq.'massive_branch'.or.index(mode,'massless_recoil').eq.1) then
     ! A ttbar real point with an almost stationary spectator. Replaying
     ! another history used to lose precision in |p_i+p_j| and its angle.
     native_mapping=.true.
     mass=173d0
     mrec=mass
     father=4
     plab(:,1)=[322.68530330568893d0,0d0,0d0,322.68530330568893d0]
     plab(:,2)=[214.20779879452144d0,0d0,0d0,-214.20779879452144d0]
     plab(:,3)=[176.64351850683025d0,-2.4450242582330782d-3, &
          -1.3603843060490267d-3,35.692192740767325d0]
     plab(:,4)=[214.43585363999250d0,27.680095453052886d0, &
          123.64130031556655d0,0.61359457138139817d0]
     plab(:,5)=[145.81372995338756d0,-27.677650428794667d0, &
          -123.63993993126064d0,72.171717199018786d0]
     if(mode.eq.'massive_branch')then
        plab(:,1)=[5894.5874484213145d0,0d0,0d0,5894.5874484213145d0]
        plab(:,2)=[4791.5059074978099d0,0d0,0d0,-4791.5059074978099d0]
        plab(:,3)=[3311.3888974023553d0,-812.64805238353540d0, &
             1532.9301421210912d0,2815.1546586597055d0]
        plab(:,4)=[2483.2449406725818d0,-635.73005954441101d0, &
             1240.3339852491772d0,2047.9246401670955d0]
        plab(:,5)=[4891.4595178441878d0,1448.3781119279461d0, &
             -2773.2641273702679d0,-3759.9977579032957d0]
     endif
     if(mode.eq.'massless_recoil')then
        ! Single-top integration point with a very soft massless spectator.
        father=3
        mrec=0d0
        plab(:,1)=[2.0109761849155021d2,0d0,0d0,2.0109761849155021d2]
        plab(:,2)=[9.9166854874691154d1,0d0,0d0,-9.9166854874691154d1]
        plab(:,3)=[1.9313968527374286d2,6.7680639002317534d1, &
             -4.2916875719676455d1,3.0844949511066687d1]
        plab(:,4)=[4.2885354705666972d-3,-2.6739189613040220d-3, &
             1.7834986716283069d-3,2.8391597063659743d-3]
        plab(:,5)=[1.0712049955702797d2,-6.7677965083356256d1, &
             4.2915092221004826d1,7.1082974946086026d1]
     endif
     if(mode.eq.'massless_recoil2')then
        ! A second point exposes amplification of the boost-direction error.
        father=3
        mrec=0d0
        plab(:,1)=[2.07429859882268346d2,0d0,0d0,2.07429859882268346d2]
        plab(:,2)=[7.94418779006372375d1,0d0,0d0,-7.94418779006372375d1]
        plab(:,3)=[1.86992207498333840d2,1.51729011936139528d1, &
             5.29980240994808440d1,4.46998677405158844d1]
        plab(:,4)=[9.10000786468497642d-4,2.39940238696784245d-5, &
             -5.24617190079674773d-4,7.43171621713114845d-4]
        plab(:,5)=[9.98786202837852954d1,-1.51729251876378264d1, &
             -5.29974994822907561d1,8.32873710694935028d1]
     endif
     call fill_father_and_ileg(5,father,mass)
     call boost_n1_to_its_cms(plab,pcm,rapidity)
     shat=4d0*pcm(0,1)*pcm(0,2)
     sqrtshat=sqrt(shat)
     xi=get_xi_from_p(5,father,pcm)
     yij=get_yij_from_p(5,father,pcm)
     phi=get_phi_from_p(5,father,pcm)
     jacinv=1d0
     pswgtinv=1d0
     call generate_momenta_massive_final_inverse(pcm,xi,yij,phi, &
          invborn,inv,jacinv,pswgtinv,shat,sqrtshat,5,father,mass)
     if(.not.ieee_is_finite(jacinv).or.jacinv.le.0d0) &
          error stop 'could not invert stationary recoil'
     p(:,1:4)=invborn(:,1:4)
     jac=2d0*pi
     pswgt=1d0
     rat_xi=0d0
     isign=1
     call generate_momenta_massive_final(-100,isign,.false.,rat_xi, &
          5,father,invborn(:,father),shat,sqrtshat,mass,inv,mrec**2,p,phi, &
          xiimax,xinorm,xi,yij,xihat,pifks,jac,pswgt,pass)
     call boost_n1_to_lab(p,out,-rapidity)
     error=maxval(abs(out-plab))/maxval(abs(plab))
     if(.not.pass.or..not.all(ieee_is_finite(out)).or.error.gt.1d-7)then
        write(*,*)'stationary recoil error',error
        error stop 'stationary recoil round trip'
     endif
     write(*,*)'PASS ',trim(mode)
     stop
  endif

  native_mapping=index(mode,'native_').eq.1
  if (mode.eq.'massive'.or.mode.eq.'native_massive') then
     mass=173d0
     mrec=173d0
  elseif (mode.eq.'massless'.or.mode.eq.'native_massless') then
     mass=0d0
     mrec=80.419d0
  else
     error stop 'unknown test'
  endif
  call fill_father_and_ileg(5,3,mass)
  nplus=0
  nminus=0
  nchecked=0
  do k=1,size(energies)
     sqrtshat=energies(k)
     shat=sqrtshat**2
     born=0d0
     born(0,1:2)=sqrtshat/2d0
     born(3,1)=sqrtshat/2d0
     born(3,2)=-sqrtshat/2d0
     energy=(shat+mass**2-mrec**2)/(2d0*sqrtshat)
     momentum=sqrt(energy**2-mass**2)
     born(:,3)=[energy,momentum*sqrt(0.91d0)*cos(0.7d0), &
          momentum*sqrt(0.91d0)*sin(0.7d0),momentum*0.3d0]
     born(:,4)=born(:,1)+born(:,2)-born(:,3)
     do i=1,size(xs)
        do j=1,size(ys)
           rnd=[xs(i),ys(j),0.137d0+0.1d0*mod(i+j,7)]
           p=0d0
           p(:,1:4)=born
           phi=2d0*pi*rnd(3)
           jac=2d0*pi
           pswgt=1d0
           rat_xi=0d0
           isign=1
           if (mass.gt.0d0) then
              call generate_momenta_massive_final(-100,isign,.false.,rat_xi, &
                   5,3,born(:,3),shat,sqrtshat,mass,rnd,mrec**2,p,phi,xiimax, &
                   xinorm,xi,yij,xihat,pifks,jac,pswgt,pass)
           else
              call generate_momenta_massless_final(-100,5,3,born(:,3), &
                   shat,sqrtshat,rnd,mrec**2,p,phi,xiimax,xinorm,xi,yij, &
                   xihat,pifks,jac,pswgt,pass)
           endif
           if (.not.pass.or.jac.le.0d0) error stop 'invalid forward test point'
           if (isign.eq.1) nplus=nplus+1
           if (isign.eq.-1) nminus=nminus+1

           ! Match the production call sequence: the minus solution has no
           ! soft counterevent to refresh the Born rapidity used for lab boosts.
           rapidity=0.25d0+0.3d0*mod(i+j,7)
           ycm_cnt(0)=-0.75d0
           call fill_FKS_commons(-100,1d0,rapidity,rapidity,shat,sqrtshat, &
                [0.2d0,0.1d0],xiimax,xinorm,xi,xihat,pifks,yij,p,out, &
                jac,jacout,mass,5,3)
           if (isign.eq.1) then
              call fill_FKS_commons(0,1d0,rapidity,rapidity,shat,sqrtshat, &
                   [0.2d0,0.1d0],xiimax,xinorm,xi,xihat,pifks,yij,p,out, &
                   jac,jacout,mass,5,3)
           endif
           call boost_n1_to_lab(p,plab,-ycm_cnt(0))
           if (abs(plab(0,1)/(sqrtshat/2d0*exp(rapidity))-1d0).gt.1d-12) &
                error stop 'stale rapidity without counterevent'
           call boost_n1_to_its_cms(plab,pcm,rapidity)
           xi=get_xi_from_p(5,3,pcm)
           yij=get_yij_from_p(5,3,pcm)
           phi=get_phi_from_p(5,3,pcm)
           jacinv=1d0
           pswgtinv=1d0
           if (mass.gt.0d0) then
              call generate_momenta_massive_final_inverse(pcm,xi,yij,phi, &
                   invborn,inv,jacinv,pswgtinv,shat,sqrtshat,5,3,mass)
           else
              call generate_momenta_massless_final_inverse(pcm,xi,yij,phi, &
                   invborn,inv,jacinv,pswgtinv,shat,sqrtshat,5,3)
           endif
           if (jacinv.le.0d0) error stop 'could not invert generated point'
           if (.not.all(ieee_is_finite(inv)).or. &
                .not.all(ieee_is_finite(invborn(:,1:4)))) &
                error stop 'nonfinite inverse map'
           error=maxval(abs(inv-rnd))
           if (error.gt.1d-7.or.error.ne.error) then
              write(*,*) 'random numbers',rnd,inv,error
              error stop 'inverse returned a different event'
           endif
           if (maxval(abs(invborn(:,1:4)-born))/sqrtshat.gt.1d-9) &
                error stop 'inverse Born recoil'
           if (abs(jacinv/jac-1d0).gt.1d-6) &
                error stop 'inverse Jacobian or spurious branch multiplicity'
           if(abs(pswgtinv/pswgt-1d0).gt.1d-6) &
                error stop 'inverse phase-space measure'
           if(native_mapping.and.mass.gt.0d0.and.isign.eq.1) &
                call check_counter_ratio(pcm,born,rnd,jac*pswgt,nratios)
           nchecked=nchecked+1
        enddo
     enddo
  enddo
  if (mode.eq.'massive'.and.nminus.eq.0) error stop 'minus solution not covered'
  if(mode.eq.'native_massive'.and.nratios.eq.0)error stop 'no counterevent ratios tested'
  write(*,*) 'PASS ',trim(mode),nchecked,nplus,nminus
contains
  subroutine check_counter_ratio(real_p,born_p,native_x,native_measure,checked)
    ! Compare the physical soft-counterevent/real measure ratio with the
    ! original xi parameterization, independently of either sampling Jacobian.
    double precision,intent(in) :: real_p(0:3,5),born_p(0:3,4),native_x(3),native_measure
    integer,intent(inout) :: checked
    double precision :: legacy_x(3),pb(0:3,-max_branch:4),q(0:3,5),scaled(0:3)
    double precision :: cj,cp,lj,lp,ratio,oldratio,local_xi,local_y,local_phi
    double precision :: xmax,xnorm,xhat,rat
    integer :: sign_branch
    logical :: good
    local_xi=get_xi_from_p(5,3,real_p)
    local_y=get_yij_from_p(5,3,real_p)
    local_phi=get_phi_from_p(5,3,real_p)
    native_mapping=.false.
    lj=1d0
    lp=1d0
    call generate_momenta_massive_final_inverse(real_p,local_xi,local_y,local_phi, &
         pb,legacy_x,lj,lp,shat,sqrtshat,5,3,mass)
    native_mapping=.true.
    if(lj.le.0d0)return ! The legacy sampling cutoff excludes some native points.
    q(:,1:4)=born_p
    cj=2d0*pi
    cp=1d0
    rat=0d0
    sign_branch=1
    call generate_momenta_massive_final(0,sign_branch,.false.,rat,5,3,born_p(:,3), &
         shat,sqrtshat,mass,native_x,mrec**2,q,local_phi,xmax,xnorm,local_xi,local_y, &
         xhat,scaled,cj,cp,good)
    if(.not.good)error stop 'invalid native soft counterevent'
    ratio=cj*cp/native_measure
    native_mapping=.false.
    q(:,1:4)=born_p
    cj=2d0*pi
    cp=1d0
    call generate_momenta_massive_final(0,sign_branch,.false.,rat,5,3,born_p(:,3), &
         shat,sqrtshat,mass,legacy_x,mrec**2,q,local_phi,xmax,xnorm,local_xi,local_y, &
         xhat,scaled,cj,cp,good)
    native_mapping=.true.
    if(.not.good)error stop 'invalid reference soft counterevent'
    oldratio=cj*cp/(lj*lp)
    if(abs(ratio-oldratio).gt.1d-6*max(abs(ratio),abs(oldratio)))then
       write(*,*)'counterevent ratios',ratio,oldratio
       error stop 'native counterevent measure conversion'
    endif
    checked=checked+1
  end subroutine
end program check_momentum_maps
