! Test the production phase-space routines without matrix elements or PDFs.
program check_momentum_maps
  use fks_phase_space_data,only: saved_xi => xi_i_fks_ev,saved_y => y_ij_fks_ev, &
       saved_pi => p_i_fks_ev,saved_pi_cnt => p_i_fks_cnt,tau_cnt,ycm_cnt
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mc_native_context, only: native_mapping
  use fks_phase_space, only: fill_fks_point_data
  use fks_radiation_maps, only: generate_momenta_massless_final, &
       generate_momenta_massless_final_inverse,generate_momenta_massive_final, &
       generate_momenta_massive_final_inverse
  use fks_phase_space_helpers, only: yminmax,gentcms,boost_n1_to_its_cms,boost_n1_to_lab, &
       get_xi_from_p,get_yij_from_p,get_phi_from_p
  implicit none
  include 'genps.inc'
  include 'nexternal.inc'
  character(len=32) :: mode,fixture
  double precision, parameter :: pi=3.1415926535897932d0
  double precision :: p(0:3,5),plab(0:3,5),pcm(0:3,5),out(0:3,5)
  double precision :: born(0:3,4),invborn(0:3,-max_branch:4)
  double precision :: rnd(3),inv(3),jac,jacinv,pswgt,pswgtinv,jacout
  double precision :: shat,sqrtshat,mass,mrec,energy,momentum,phi,xi,yij
  double precision :: xiimax,xinorm,xihat,pifks(0:3),rapidity,error
  double precision :: transfer,tmin,tmax,remainder(0:3)
  logical :: softtest,colltest,pass
  common /sctests/softtest,colltest
  integer :: isign,i,j,k,nplus,nminus,nchecked,father
  integer :: nratios=0
  double precision, parameter :: xs(7)=[1d-4,0.01d0,0.25d0,0.6d0,0.9d0,0.97d0,0.999d0]
  double precision, parameter :: ys(4)=[0.2d0,0.7d0,0.9d0,0.999d0]
  double precision, parameter :: energies(3)=[400d0,1000d0,4000d0]

  call get_command_argument(1,mode)
  softtest=.false.
  colltest=.false.
  if (mode.eq.'isr'.or.mode.eq.'isr_fks'.or. &
       mode.eq.'isr_symmetric'.or.mode.eq.'isr_automatic'.or.mode.eq.'isr_symmetric_fks'.or. &
       mode.eq.'isr_symmetric_bounds'.or.mode.eq.'isr_lower_bound') then
     call check_isr_mapping(trim(mode))
     write(*,*) 'PASS ',trim(mode)
     stop
  endif
  if (mode.eq.'native_projection') then
     call check_native_projection()
     write(*,*) 'PASS native_projection'
     stop
  endif
  if (mode.eq.'massive_boundary') then
     call check_massive_boundary()
     write(*,*) 'PASS massive_boundary'
     stop
  endif
  if (mode.eq.'massive_precision') then
     call check_massive_precision()
     write(*,*) 'PASS massive_precision'
     stop
  endif
  if (mode.eq.'fsr_cutoff_jacobians') then
     call check_fsr_cutoff_jacobians()
     write(*,*) 'PASS fsr_cutoff_jacobians'
     stop
  endif
  if (mode.eq.'near_axis_angles') then
     call check_near_axis_angles()
     write(*,*) 'PASS near_axis_angles'
     stop
  endif
  if (mode.eq.'dijet_isr_boundary') then
     call check_dijet_isr_boundary()
     write(*,*) 'PASS dijet_isr_boundary'
     stop
  endif
  if (mode.eq.'isr_onshell_boost') then
     call check_onshell_isr_boost()
     write(*,*) 'PASS isr_onshell_boost'
     stop
  endif
  if (mode.eq.'wjet_boundary') then
     call check_wjet_boundary_projection()
     write(*,*) 'PASS wjet_boundary'
     stop
  endif
  if (mode.eq.'singletop_recoil') then
     call check_singletop_projection()
     write(*,*) 'PASS singletop_recoil'
     stop
  endif
  if (mode.eq.'soft_direction') then
     ! Finite real legs cannot borrow the cached direction of another
     ! history, including when the soft leg is the sister rather than i.
     native_mapping=.true.
     do k=5,10
        p=0d0
        p(:,4)=[10d0**(-k),0d0,10d0**(-k),0d0]
        p(:,5)=[20d0,20d0,0d0,0d0]
        saved_pi_cnt=0d0
        saved_pi_cnt(:,0)=[1d0,1d0,0d0,0d0]
        if(abs(get_yij_from_p(5,4,p,saved_pi_cnt(:,0))).gt.1d-12.or. &
             abs(get_yij_from_p(4,5,p,saved_pi_cnt(:,0))).gt.1d-12) &
             error stop 'finite soft momentum replaced by cached direction'
     enddo
     ! Exactly soft counterevents still need their scaled direction.
     p(:,5)=0d0
     if(abs(get_yij_from_p(5,4,p,saved_pi_cnt(:,0))).gt.1d-12) &
          error stop 'zero-energy soft direction lost'
     write(*,*)'PASS ',trim(mode)
     stop
  endif
  if (mode.eq.'wjet_soft_history') then
     ! W+jet, seed 12004: after an ISR history, the cached soft direction
     ! was parallel to leg 4. Reusing it for the physical soft gluon 5
     ! made the next FSR inverse collinear, with a zero Jacobian.
     native_mapping=.true.
     plab(:,1)=[6493.4774065510182d0,0d0,0d0,6493.4774065510182d0]
     plab(:,2)=[0.53658527893563424d0,0d0,0d0,-0.53658527893563424d0]
     plab(:,3)=[4139.1636210648694d0,27.817799808130822d0, &
          10.112078278955879d0,4138.2764747995252d0]
     plab(:,4)=[2354.8503602114188d0,-27.817800038912463d0, &
          -10.112078140338355d0,2354.6643359223262d0]
     plab(:,5)=[1.0553665566093150d-5,2.3078164005879552d-7, &
          -1.3861752450787524d-7,1.0550231367010456d-5]
     call boost_n1_to_its_cms(plab,pcm,rapidity)
     sqrtshat=pcm(0,1)+pcm(0,2)
     shat=sqrtshat**2
     mrec=sqrt(pcm(0,3)**2-sum(pcm(1:3,3)**2))
     do k=1,2
        p=pcm
        if(k.eq.2)then
           p(:,4)=pcm(:,5)
           p(:,5)=pcm(:,4)
        endif
        ! Make the stale direction parallel to the hard daughter in
        ! both orderings. Neither finite physical leg may use it.
        saved_pi_cnt=0d0
        saved_pi_cnt(:,0)=pcm(:,4)
        xi=get_xi_from_p(5,p)
        yij=get_yij_from_p(5,4,p,saved_pi_cnt(:,0))
        phi=get_phi_from_p(5,4,p)
        if(abs(yij).gt.0.99d0)error stop 'soft history became collinear'
        jacinv=1d0
        pswgtinv=1d0
        call generate_momenta_massless_final_inverse(p,xi,yij,phi, &
             invborn,inv,jacinv,pswgtinv,shat,sqrtshat,5,4)
        if(.not.ieee_is_finite(jacinv).or.jacinv.le.0d0) &
             error stop 'soft history has no finite native inverse'
        out(:,1:4)=invborn(:,1:4)
        jac=2d0*pi
        pswgt=1d0
        call generate_momenta_massless_final(-100,5,4,invborn(:,4), &
             shat,sqrtshat,inv,mrec**2,out,phi,xiimax,xinorm,xi,yij,xihat,pifks,jac,pswgt,pass)
        if(.not.pass.or..not.ieee_is_finite(jac).or.jac.le.0d0.or. &
             .not.all(ieee_is_finite(out)))error stop 'invalid soft history replay'
        call boost_n1_to_lab(out,plab,-rapidity)
        call boost_n1_to_lab(p,out,-rapidity)
        error=maxval(abs(plab-out))/maxval(abs(out))
        if(error.gt.1d-7)error stop 'soft history lab momentum round trip'
     enddo
     write(*,*)'PASS ',trim(mode)
     stop
  endif
  if (mode.eq.'soft_counterevent') then
     mass=173d0
     mrec=mass
     sqrtshat=1000d0
     shat=sqrtshat**2
     energy=sqrtshat/2d0
     momentum=sqrt(energy**2-mass**2)
     born(:,1)=[energy,0d0,0d0,energy]
     born(:,2)=[energy,0d0,0d0,-energy]
     born(:,3)=[energy,momentum*0.3d0,momentum*0.4d0,momentum*sqrt(0.75d0)]
     born(:,4)=[energy,-born(1:3,3)]
     do k=0,1
        native_mapping=k.eq.1
        do i=1,size(xs)
           do j=1,size(ys)
              rnd=[xs(i),ys(j),0.27934974545675351d0]
              p(:,1:4)=born
              phi=2d0*pi*rnd(3)
              jac=2d0*pi
              pswgt=1d0
              isign=1
              call generate_momenta_massive_final(-100,isign, &
                   5,3,born(:,3),shat,sqrtshat,mass,rnd,mrec**2,p,phi, &
                   xiimax,xinorm,xi,yij,xihat,pifks,jac,pswgt,pass)
              if(.not.pass)error stop 'invalid real point before counterevent'
              p(:,1:4)=born
              jac=2d0*pi
              pswgt=1d0
              call generate_momenta_massive_final(0,isign, &
                   5,3,born(:,3),shat,sqrtshat,mass,rnd,mrec**2,p,phi, &
                   xiimax,xinorm,xi,yij,xihat,pifks,jac,pswgt,pass)
              if(.not.pass.or..not.all(ieee_is_finite(p)).or. &
                   .not.all(ieee_is_finite(pifks)).or..not.ieee_is_finite(jac*pswgt)) &
                   error stop 'invalid massive soft counterevent'
              if(maxval(abs(p(:,1:4)-born))/sqrtshat.gt.1d-12.or. &
                   maxval(abs(p(:,5))).ne.0d0)error stop 'counterevent differs from Born'
              ! At zero emission energy, the scaled emission vector still
              ! carries the opening angle relative to the Born mother.
              error=abs(sum(pifks(1:3)*born(1:3,3))/(pifks(0)*momentum)-yij)
              if(error.gt.1d-12)error stop 'soft counterevent lost emission angle'
           enddo
        enddo
     enddo
     write(*,*)'PASS ',trim(mode)
     stop
  endif
  if (mode.eq.'massive_born_measure') then
     call check_massive_born_measure()
     write(*,*)'PASS ',trim(mode)
     stop
  endif
  if (mode.eq.'soft_recoil_inverse') then
     ! Independent three-body massless kinematics: the soft spectator is
     ! transverse to one hard daughter. Boosting this recoil back to Born
     ! kinematics must give E=sqrt(s)/2 along its original direction.
     ! Inferring its energy by subtracting the hard pair from the beams
     ! loses relative precision as its energy decreases.
     mass=0d0
     mrec=0d0
     sqrtshat=200d0
     shat=sqrtshat**2
     do k=0,1
        native_mapping=k.eq.1
        do i=3,9
           momentum=10d0**(-i)
           energy=(shat-2d0*sqrtshat*momentum)/(2d0*(sqrtshat-momentum))
           p(:,1)=[sqrtshat/2d0,0d0,0d0,sqrtshat/2d0]
           p(:,2)=[sqrtshat/2d0,0d0,0d0,-sqrtshat/2d0]
           p(:,3)=[energy,energy,0d0,0d0]
           p(:,4)=[momentum,0d0,momentum,0d0]
           p(:,5)=[sqrtshat-momentum-energy,-energy,-momentum,0d0]
           xi=get_xi_from_p(5,p)
           yij=get_yij_from_p(5,3,p,saved_pi_cnt(:,0))
           phi=get_phi_from_p(5,3,p)
           jacinv=1d0
           pswgtinv=1d0
           call generate_momenta_massless_final_inverse(p,xi,yij,phi, &
                invborn,inv,jacinv,pswgtinv,shat,sqrtshat,5,3)
           if (.not.ieee_is_finite(jacinv).or.jacinv.le.0d0.or. &
                .not.all(ieee_is_finite(invborn(:,1:4)))) &
                error stop 'invalid soft massless recoil inverse'
           error=maxval(abs(invborn(:,4)-[sqrtshat/2d0,0d0,sqrtshat/2d0,0d0]))/sqrtshat
           if(error.gt.1d-9)then
              write(*,*)'soft recoil Born error',native_mapping,momentum,error
              error stop 'soft recoil inverse boost'
           endif
        enddo
     enddo
     write(*,*)'PASS ',trim(mode)
     stop
  endif
  if (mode.eq.'outer_angles') then
     native_mapping=.false.
     mass=173d0
     mrec=mass
     do k=1,size(energies)
        sqrtshat=energies(k)
        shat=sqrtshat**2
        energy=sqrtshat/2d0
        momentum=sqrt(energy**2-mass**2)
        born(:,1)=[energy,0d0,0d0,energy]
        born(:,2)=[energy,0d0,0d0,-energy]
        born(:,3)=[energy,momentum*0.3d0,momentum*0.4d0,momentum*sqrt(0.75d0)]
        born(:,4)=[energy,-born(1:3,3)]
        do i=3,7
           do j=1,3
              ! As xi approaches its endpoint from below, the massive
              ! sister is almost at rest and cos(theta_i) rounds to one.
              ! The sampled opening angle still fixes its direction.
              rnd=[1d0-10d0**(-i),0.15d0*j,0.27934974545675351d0]
              p(:,1:4)=born
              phi=2d0*pi*rnd(3)
              jac=2d0*pi
              pswgt=1d0
              isign=1
              call generate_momenta_massive_final(-100,isign, &
                   5,3,born(:,3),shat,sqrtshat,mass,rnd,mrec**2,p,phi, &
                   xiimax,xinorm,xi,yij,xihat,pifks,jac,pswgt,pass)
              if (.not.pass.or..not.all(ieee_is_finite(p))) &
                   error stop 'invalid soft massive sister point'
              error=abs(sum(p(1:3,3)*p(1:3,5))/ &
                   sqrt(sum(p(1:3,3)**2)*sum(p(1:3,5)**2))-yij)
              if(error.gt.1d-7)then
                 write(*,*)'soft massive sister angular error',sqrtshat,rnd,error
                 error stop 'lost soft massive sister angle'
              endif
           enddo
        enddo
     enddo
     write(*,*)'PASS ',trim(mode)
     stop
  endif
  if (mode.eq.'soft_daughter') then
     ! W+jet production (seed 12004) exposed a soft sister of a hard FKS
     ! parton: recovering sin(theta) from 1-cos(theta)**2 rounded both
     ! directions onto each other and made the native inverse Jacobian zero.
     native_mapping=.false.
     mass=0d0
     mrec=80.419002445756163d0
     sqrtshat=105.09663678328764d0
     shat=sqrtshat**2
     momentum=(shat-mrec**2)/(2d0*sqrtshat)
     born=0d0
     born(:,1)=[sqrtshat/2d0,0d0,0d0,sqrtshat/2d0]
     born(:,2)=[sqrtshat/2d0,0d0,0d0,-sqrtshat/2d0]
     born(:,3)=[momentum,momentum*0.3d0,momentum*0.4d0,momentum*sqrt(0.75d0)]
     born(:,4)=born(:,1)+born(:,2)-born(:,3)
     rnd=[0.99999976678858993d0,9.1845446031599692d-3,0.27934974545675351d0]
     phi=2d0*pi*rnd(3)
     jac=2d0*pi
     pswgt=1d0
     p(:,1:4)=born
     call generate_momenta_massless_final(-100,5,3,born(:,3), &
          shat,sqrtshat,rnd,mrec**2,p,phi,xiimax,xinorm,xi,yij,xihat,pifks,jac,pswgt,pass)
     if(.not.pass) error stop 'invalid soft-daughter point'
     error=abs(p(0,3)**2-sum(p(1:3,3)**2))/p(0,3)**2
     if(error.gt.1d-8) then
        write(*,*) 'soft daughter relative mass-shell error',error
        error stop 'soft daughter is off shell'
     endif
     native_mapping=.true.
     xi=get_xi_from_p(5,p)
     yij=get_yij_from_p(5,3,p,saved_pi_cnt(:,0))
     phi=get_phi_from_p(5,3,p)
     jacinv=1d0
     pswgtinv=1d0
     call generate_momenta_massless_final_inverse(p,xi,yij,phi, &
          invborn,inv,jacinv,pswgtinv,shat,sqrtshat,5,3)
     if(.not.ieee_is_finite(jacinv).or.jacinv.le.0d0) &
          error stop 'soft daughter has no native inverse'
     out(:,1:4)=invborn(:,1:4)
     jac=2d0*pi
     pswgt=1d0
     call generate_momenta_massless_final(-100,5,3,invborn(:,3), &
          shat,sqrtshat,inv,mrec**2,out,phi,xiimax,xinorm,xi,yij,xihat,pifks,jac,pswgt,pass)
     error=maxval(abs(out-p))/maxval(abs(p))
     if(.not.pass.or..not.all(ieee_is_finite(out)).or.error.gt.1d-7) &
          error stop 'soft daughter native round trip'
     write(*,*) 'PASS soft_daughter'
     stop
  endif
  if (mode.eq.'born_threshold') then
     ! Native single-top histories can project onto a Born configuration
     ! with a soft massless spectator. The expanded Kallen polynomial
     ! rounded to zero here, excluding physical t values.
     mass=173d0
     do i=-7,1
        momentum=10d0**i
        energy=(sqrt(mass**2+momentum**2)+momentum)/2d0
        born(:,1)=[energy,0d0,0d0,energy]
        born(:,2)=[energy,0d0,0d0,-energy]
        born(:,3)=[sqrt(mass**2+momentum**2), &
             momentum*0.3d0,momentum*0.4d0,momentum*sqrt(0.75d0)]
        born(:,4)=[momentum,-born(1:3,3)]
        jac=1d0
        transfer=(born(0,1)-born(0,3))**2-sum((born(1:3,1)-born(1:3,3))**2)
        phi=atan2(born(2,3),born(1,3))
        call yminmax((2d0*energy)**2,mass**2,0d0,0d0,0d0,tmin,tmax)
        if (tmin.ge.tmax.or.transfer.lt.tmin.or.transfer.gt.tmax) &
             error stop 'physical Born momentum outside t-channel bounds'
        call gentcms(born(:,1),born(:,2),transfer,phi,mass,0d0, &
             out(:,3),remainder,jac)
        if (jac.le.0d0.or.maxval(abs(out(:,3)-born(:,3))).gt.1d-10) &
             error stop 'near-threshold Born round trip'
     enddo
     write(*,*) 'PASS born_threshold'
     stop
  endif
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

  fixture=mode
  if(index(mode,'outer_').eq.1)fixture=mode(7:)
  if (fixture.eq.'massive_recoil'.or.fixture.eq.'massive_branch'.or.index(fixture,'massless_recoil').eq.1) then
     ! A ttbar real point with an almost stationary spectator. Replaying
     ! another history used to lose precision in |p_i+p_j| and its angle.
     native_mapping=index(mode,'outer_').ne.1
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
     if(fixture.eq.'massive_branch')then
        plab(:,1)=[5894.5874484213145d0,0d0,0d0,5894.5874484213145d0]
        plab(:,2)=[4791.5059074978099d0,0d0,0d0,-4791.5059074978099d0]
        plab(:,3)=[3311.3888974023553d0,-812.64805238353540d0, &
             1532.9301421210912d0,2815.1546586597055d0]
        plab(:,4)=[2483.2449406725818d0,-635.73005954441101d0, &
             1240.3339852491772d0,2047.9246401670955d0]
        plab(:,5)=[4891.4595178441878d0,1448.3781119279461d0, &
             -2773.2641273702679d0,-3759.9977579032957d0]
     endif
     if(fixture.eq.'massless_recoil')then
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
     if(fixture.eq.'massless_recoil2')then
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
     call boost_n1_to_its_cms(plab,pcm,rapidity)
     shat=4d0*pcm(0,1)*pcm(0,2)
     sqrtshat=sqrt(shat)
     xi=get_xi_from_p(5,pcm)
     yij=get_yij_from_p(5,father,pcm,saved_pi_cnt(:,0))
     phi=get_phi_from_p(5,father,pcm)
     jacinv=1d0
     pswgtinv=1d0
     call generate_momenta_massive_final_inverse(pcm,xi,yij,phi, &
          invborn,inv,jacinv,pswgtinv,shat,sqrtshat,5,father,mass)
     if(.not.ieee_is_finite(jacinv).or.jacinv.le.0d0) &
          error stop 'could not invert stationary recoil'
     if(index(mode,'outer_').eq.1.and.index(fixture,'massless_recoil').eq.1)then
        ! The outer xi coordinate becomes ill-conditioned here; compare
        ! only the coordinate-independent Born reconstruction with the
        ! native inverse. Native forward round trips are checked separately.
        born=invborn(:,1:4)
        native_mapping=.true.
        jacinv=1d0
        pswgtinv=1d0
        call generate_momenta_massive_final_inverse(pcm,xi,yij,phi, &
             invborn,inv,jacinv,pswgtinv,shat,sqrtshat,5,father,mass)
        if(jacinv.le.0d0.or.maxval(abs(invborn(:,1:4)-born))/sqrtshat.gt.1d-9) &
             error stop 'outer and native Born inverse disagree'
        write(*,*)'PASS ',trim(mode)
        stop
     endif
     p(:,1:4)=invborn(:,1:4)
     jac=2d0*pi
     pswgt=1d0
     isign=1
     call generate_momenta_massive_final(-100,isign, &
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
           isign=1
           if (mass.gt.0d0) then
              call generate_momenta_massive_final(-100,isign, &
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
           call fill_fks_point_data(-100,1d0,rapidity,rapidity,shat,sqrtshat, &
                [0.2d0,0.1d0],xiimax,xinorm,xi,xihat,pifks,yij,p,out, &
                jac,jacout)
           if (isign.eq.1) then
              call fill_fks_point_data(0,1d0,rapidity,rapidity,shat,sqrtshat, &
                   [0.2d0,0.1d0],xiimax,xinorm,xi,xihat,pifks,yij,p,out, &
                   jac,jacout)
           endif
           call boost_n1_to_lab(p,plab,-ycm_cnt(0))
           if (abs(plab(0,1)/(sqrtshat/2d0*exp(rapidity))-1d0).gt.1d-12) &
                error stop 'stale rapidity without counterevent'
           call boost_n1_to_its_cms(plab,pcm,rapidity)
           xi=get_xi_from_p(5,pcm)
           yij=get_yij_from_p(5,3,pcm,saved_pi_cnt(:,0))
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
  subroutine check_massive_precision()
    use fks_phase_space_helpers, only: get_massive_fsr_bounds
    integer,parameter :: qp=selected_real_kind(30)
    double precision,parameter :: openings(3)=[-.805d0,.4d0,-.98d0]
    double precision :: xim,xibm,ximax,a,b,c,delta,ratio,gap,sq,coordinate,distance,observed
    double precision :: conditioned,path_weight,physical_weight
    real(qp) :: wq,mq,rq,mjq,mrq,yq,lambdaq,ximq,sqq,uq,aq,xmaxq,xnormq
    real(qp) :: xq,ratioq,gq,d1q,dmq,xiq,dq,eiq,hq,bq,cq,kq,ejq,reference,expected
    integer :: configuration,opening_index,sample,point,slot
    native_mapping=.false.
    mass=173d0
    do configuration=1,4
      sqrtshat=1000d0
      mrec=173d0
      if(configuration.eq.2.or.configuration.eq.4)mrec=0d0
      if(configuration.eq.3)sqrtshat=(mass+mrec)*(1d0+1d-8)
      if(configuration.eq.4)sqrtshat=mass*(1d0+1d-6)
      shat=sqrtshat**2
      momentum=sqrt((sqrtshat-mass-mrec)*(sqrtshat+mass+mrec)* &
           (sqrtshat-mass+mrec)*(sqrtshat+mass-mrec))/(2d0*sqrtshat)
      energy=hypot(momentum,mass)
      born(:,1)=[sqrtshat/2d0,0d0,0d0,sqrtshat/2d0]
      born(:,2)=[sqrtshat/2d0,0d0,0d0,-sqrtshat/2d0]
      born(:,3)=[energy,.3d0*momentum,.4d0*momentum,sqrt(.75d0)*momentum]
      born(:,4)=[hypot(momentum,mrec),-born(1:3,3)]
      wq=real(sqrtshat,qp)
      mq=real(mass,qp)
      rq=real(mrec,qp)
      mjq=mq/wq
      mrq=rq/wq
      lambdaq=((wq-mq-rq)*(wq+mq+rq)*(wq-mq+rq)*(wq+mq-rq))/wq**4
      ximq=(wq-mq-rq)*(wq-mq+rq)/(wq*(wq-mq))
      do opening_index=1,size(openings)
        yij=openings(opening_index)
        call get_massive_fsr_bounds(shat,sqrtshat,mass,mrec**2,yij, &
             xim,xibm,ximax,a,b,c,delta,xiimax,xinorm,gap,sq)
        yq=real(yij,qp)
        aq=1-mjq*mjq*(1-yq*yq)
        sqq=mjq*sqrt(4*mrq*mrq+(1-yq*yq)*lambdaq)
        uq=(mjq*ximq*yq)**2/(sqq+sqrt(sqq*sqq+aq*(mjq*ximq*yq)**2))
        xmaxq=ximq
        xnormq=ximq
        if(yij.lt.0d0)then
          xmaxq=ximq+uq
          xnormq=ximq+2*uq
        endif
        if(abs(real(xim,qp)/ximq-1).gt.3d-13.or. &
             abs(real(xiimax,qp)/xmaxq-1).gt.3d-13.or. &
             abs(real(xinorm,qp)/xnormq-1).gt.3d-13)then
          write(*,*) 'bounds precision',configuration,opening_index,xim,ximq,xiimax,xmaxq,xinorm,xnormq
          error stop 'massive threshold bounds precision'
        endif
        ratio=xiimax/xinorm
        do sample=1,8
          coordinate=.3d0*ratio
          if(sample.eq.2)coordinate=.8d0*ratio
          if(sample.eq.3)coordinate=ratio*(1d0-1d-6)
          if(sample.eq.4)coordinate=ratio*(1d0-1d-10)
          if(sample.eq.5)coordinate=ratio+(1d0-ratio)*1d-6
          if(sample.eq.6)coordinate=ratio+(1d0-ratio)*1d-10
          if(sample.eq.7)coordinate=1d0-1d-6
          if(sample.eq.8)coordinate=1d0-1d-10
          if(coordinate.eq.ratio.or.(yij.ge.0d0.and.sample.ge.5.and.sample.le.6))cycle
          rnd=[coordinate,sqrt(((1d0-openings(opening_index))/2d0-5d-7)/(1d0-5d-7)),.317d0]
          p=0d0
          p(:,1:4)=born
          jac=2d0*pi
          pswgt=1d0
          call generate_momenta_massive_final(-100,isign,5,3,born(:,3),shat,sqrtshat,mass,rnd,mrec**2,p, &
               2d0*pi*rnd(3),xiimax,xinorm,xi,yij,xihat,pifks,jac,pswgt,pass)
          if(.not.pass.or..not.all(ieee_is_finite(p)).or..not.ieee_is_finite(pswgt)) &
               error stop 'nonfinite massive precision point'
          call get_massive_fsr_bounds(shat,sqrtshat,mass,mrec**2,yij, &
               xim,xibm,ximax,a,b,c,delta,xiimax,xinorm,gap,sq)
          ratio=xiimax/xinorm
          observed=sqrt(sum(p(1:3,3)**2))
          ! Resolve the sampled distance from the nearest physical endpoint
          ! in quad precision. Rounding xi itself to double can erase that
          ! distance, so do not demand relative endpoint digits from it.
          yq=real(yij,qp)
          aq=1-mjq*mjq*(1-yq*yq)
          sqq=mjq*sqrt(4*mrq*mrq+(1-yq*yq)*lambdaq)
          uq=(mjq*ximq*yq)**2/(sqq+sqrt(sqq*sqq+aq*(mjq*ximq*yq)**2))
          xq=real(coordinate,qp)
          ratioq=real(ratio,qp)
          if(isign.eq.1)then
            gq=real(xiimax,qp)*(1-real(1d-6,qp))*(1-(xq/ratioq)**2)
            dmq=gq
            if(yij.lt.0d0)dmq=dmq-uq
          elseif(coordinate-ratio.le.1d0-coordinate)then
            dmq=real(xinorm,qp)*(xq-ratioq)-uq
          else
            dmq=-real(xinorm,qp)*(1-xq)
          endif
          xiq=ximq-dmq
          eiq=xiq*wq/2
          hq=(wq*wq+mq*mq-rq*rq)/2-wq*eiq
          aq=(wq-eiq)**2-(eiq*yq)**2
          bq=2*hq*eiq*yq
          ! Factor the constant coefficient near the stationary endpoint;
          ! the otherwise independent energy-conservation quadratic would
          ! itself lose reference digits by subtracting two O(W**4) terms.
          cq=-(wq-mq)*wq*dmq/2*((wq-eiq)*mq+hq)
          dq=bq*bq-4*aq*cq
          if(dq.le.0)error stop 'nonphysical endpoint-distance reference'
          kq=(-bq+isign*sqrt(dq))/(2*aq)
          ejq=sqrt(kq*kq+mq*mq)
          expected=2*wq*wq/(4*acos(-1._qp))**3*(kq/real(momentum,qp))/ &
               abs(2-xiq*(1-ejq*yq/kq))
          path_weight=real(expected,8)
          if(abs(real(observed,qp)-kq).gt.3d-12*max(abs(kq),wq*1d-24))then
            write(*,*) 'path root',configuration,opening_index,sample,coordinate,ratio,observed,kq
            write(*,*) 'path geometry',xim,ximq,gap,uq,sq,sqq,xiimax,xinorm,dmq,xi,xiq
            error stop 'massive sampled-path root precision'
          endif
          if(abs(pswgt-path_weight).gt.3d-12*max(path_weight,shat*1d-24))then
            write(*,*) 'path precision',configuration,opening_index,sample,coordinate,ratio,pswgt,path_weight
            error stop 'massive sampled-path phase-space precision'
          endif
          ! An independent, dimensionful energy-conservation quadratic checks
          ! the physical root and units. Its density is ill-conditioned near
          ! the junction and the stationary daughter, so account for rounding
          ! of the returned xi instead of demanding meaningless relative digits.
          eiq=real(xi,qp)*wq/2
          yq=real(yij,qp)
          aq=(wq-eiq)**2-(eiq*yq)**2
          bq=2*((wq*wq+mq*mq-rq*rq)/2-wq*eiq)*eiq*yq
          cq=(wq-eiq)**2*mq*mq-((wq*wq+mq*mq-rq*rq)/2-wq*eiq)**2
          dq=bq*bq-4*aq*cq
          distance=min(abs(coordinate-ratio),1d0-coordinate)
          conditioned=4d-12+128d0*epsilon(1d0)/max(distance,1d-15)
          if(dq.gt.0)then
            kq=(-bq+isign*sqrt(dq))/(2*aq)
            ejq=sqrt(kq*kq+mq*mq)
            if(abs(real(observed,qp)-kq).gt.conditioned*max(abs(kq),wq*1d-14))then
              write(*,*) 'physical root',configuration,opening_index,sample,observed,kq,conditioned
              error stop 'massive physical root precision'
            endif
            reference=2*wq*wq/(4*acos(-1._qp))**3*(kq/real(momentum,qp))/ &
                 abs(2-real(xi,qp)*(1-ejq*yq/kq))
            physical_weight=real(reference,8)
            if(abs(pswgt-physical_weight).gt.conditioned*max(physical_weight,shat*1d-24))then
              write(*,*) 'physical phase',configuration,opening_index,sample,pswgt,physical_weight,conditioned
              error stop 'massive physical phase-space precision'
            endif
          elseif(distance.gt.1d-12)then
            error stop 'massive precision point outside physical support'
          endif
          remainder=sum(p(:,1:2),dim=2)-sum(p(:,3:5),dim=2)
          if(maxval(abs(remainder))/sqrtshat.gt.2d-12)then
            write(*,*) 'precision conservation',configuration,opening_index,sample,rnd,remainder/sqrtshat
            error stop 'massive precision momentum conservation'
          endif
        enddo
        if(configuration.eq.1.and.openings(opening_index).lt.0d0)then
          ! The exactly coalescent real density is singular, but its soft
          ! endpoint remains a valid independent Born configuration.
          rnd=[ratio,sqrt(((1d0-openings(opening_index))/2d0-5d-7)/(1d0-5d-7)),.317d0]
          do point=1,2
            slot=-100
            if(point.eq.2)slot=0
            p=0d0
            p(:,1:4)=born
            jac=2d0*pi
            pswgt=1d0
            call generate_momenta_massive_final(slot,isign,5,3,born(:,3),shat,sqrtshat,mass,rnd,mrec**2,p, &
                 2d0*pi*rnd(3),xiimax,xinorm,xi,yij,xihat,pifks,jac,pswgt,pass)
            if(slot.eq.-100)then
              if(pass.or.jac.ne.-105d0)error stop 'singular massive junction retained'
            else
              if(.not.pass.or.jac.le.0d0.or..not.all(ieee_is_finite(p)).or. &
                   .not.ieee_is_finite(pswgt))error stop 'junction lost its soft endpoint'
            endif
          enddo
        endif
      enddo
    enddo
  end subroutine check_massive_precision

  subroutine check_near_axis_angles()
    use fks_phase_space_helpers, only: getangles
    double precision,parameter :: scales(3)=[1d0,1d-200,1d200]
    double precision :: vector(0:3),rotated(0:3),restored(0:3),axis(0:3)
    double precision :: theta,ctheta,stheta,azimuth,cazimuth,sazimuth
    double precision :: small,norm,direction(3),ux,uy,tol
    integer :: pole,power,quadrant,scale_index
    tol=16d0*epsilon(1d0)
    ! Keep nonzero transverse components even when cos(theta) rounds to +/-1.
    ! Their known 3:4 ratio also checks the azimuth in every quadrant.
    do pole=-1,1,2
      do power=2,12
        small=10d0**(-power)
        norm=sqrt(1d0+small**2)
        do quadrant=0,3
          ux=.6d0
          uy=.8d0
          if(btest(quadrant,0))ux=-ux
          if(btest(quadrant,1))uy=-uy
          direction=[small*ux,small*uy,dble(pole)]/norm
          do scale_index=1,size(scales)
            vector=[2d0,small*ux,small*uy,dble(pole)]*scales(scale_index)
            call getangles(vector,theta,ctheta,stheta,azimuth,cazimuth,sazimuth)
            if(.not.all(ieee_is_finite([theta,ctheta,stheta,azimuth,cazimuth,sazimuth]))) &
                 error stop 'nonfinite near-axis angles'
            if(abs(stheta/(small/norm)-1d0).gt.tol.or. &
                 abs(ctheta-direction(3)).gt.tol)error stop 'near-axis polar direction lost'
            if(abs(cazimuth-ux).gt.tol.or.abs(sazimuth-uy).gt.tol.or. &
                 abs(azimuth-atan2(uy,ux)).gt.tol)error stop 'near-axis azimuth lost'
            if(abs(theta-atan2(small,dble(pole))).gt.tol)error stop 'near-axis polar angle'
            axis=[1d0,0d0,0d0,1d0]
            call rotate_invar(axis,rotated,ctheta,stheta,cazimuth,sazimuth)
            if(maxval(abs(rotated(1:2)/direction(1:2)-1d0)).gt.tol.or. &
                 abs(rotated(3)-direction(3)).gt.tol)error stop 'near-axis rotation lost direction'
            vector=[2d0,.31d0,-.27d0,.63d0]
            call rotate_invar(vector,rotated,ctheta,stheta,cazimuth,sazimuth)
            call trp_rotate_invar(rotated,restored,ctheta,stheta,cazimuth,sazimuth)
            if(maxval(abs(restored-vector)).gt.tol)error stop 'near-axis inverse rotation'
          enddo
          ! The FSR map must retain the Born emitter's transverse momentum,
          ! not merely pass an angle-reconstruction unit test.
          native_mapping=.false.
          sqrtshat=1000d0
          shat=sqrtshat**2
          mrec=173d0
          energy=(shat-mrec**2)/(2d0*sqrtshat)
          born(:,1)=[sqrtshat/2d0,0d0,0d0,sqrtshat/2d0]
          born(:,2)=[sqrtshat/2d0,0d0,0d0,-sqrtshat/2d0]
          born(:,3)=[energy,energy*direction]
          born(:,4)=born(:,1)+born(:,2)-born(:,3)
          p=0d0
          p(:,1:4)=born
          rnd=[.63d0,.42d0,.317d0]
          jac=2d0*pi
          pswgt=1d0
          call generate_momenta_massless_final(-100,5,3,born(:,3),shat,sqrtshat,rnd,mrec**2,p, &
               2d0*pi*rnd(3),xiimax,xinorm,xi,yij,xihat,pifks,jac,pswgt,pass)
          if(.not.pass.or..not.all(ieee_is_finite(p)))error stop 'near-axis FSR map rejected'
          remainder=sum(p(:,1:2),dim=2)-sum(p(:,3:5),dim=2)
          if(maxval(abs(remainder))/sqrtshat.gt.tol)error stop 'near-axis map momentum conservation'
        enddo
      enddo
      vector=[1d0,0d0,0d0,dble(pole)]
      call getangles(vector,theta,ctheta,stheta,azimuth,cazimuth,sazimuth)
      if(.not.all(ieee_is_finite([theta,ctheta,stheta,azimuth,cazimuth,sazimuth]))) &
           error stop 'nonfinite exact-axis angles'
      if(ctheta.ne.dble(pole).or.stheta.ne.0d0.or.abs(theta-acos(dble(pole))).gt.tol.or. &
           azimuth.ne.0d0.or.cazimuth.ne.1d0.or.sazimuth.ne.0d0)error stop 'exact-axis angle convention'
    enddo
    vector=0d0
    call getangles(vector,theta,ctheta,stheta,azimuth,cazimuth,sazimuth)
    if(.not.all(ieee_is_finite([theta,ctheta,stheta,azimuth,cazimuth,sazimuth]))) &
         error stop 'nonfinite zero-vector angles'
    if(theta.ne.0d0.or.ctheta.ne.1d0.or.stheta.ne.0d0.or.azimuth.ne.0d0.or. &
         cazimuth.ne.1d0.or.sazimuth.ne.0d0)error stop 'zero-vector angle convention'
  end subroutine check_near_axis_angles

  subroutine check_fsr_cutoff_jacobians()
    double precision,parameter :: step=1d-5
    double precision :: center(3),shifted(3),j0,jwork,coordinate0,coordinate_plus,coordinate_minus
    double precision :: angle0,angle_plus,angle_minus,radial_derivative,angular_derivative,expected,ratio
    double precision :: xi_fix,y_fix
    common/cxiyfix/xi_fix,y_fix
    integer :: mapping,mass_case,branch,slot
    sqrtshat=1000d0
    shat=sqrtshat**2
    do mapping=0,1
      native_mapping=mapping.eq.1
      do mass_case=0,1
        mass=173d0*mass_case
        mrec=173d0
        energy=(shat+mass**2-mrec**2)/(2d0*sqrtshat)
        momentum=sqrt(energy**2-mass**2)
        born(:,1)=[sqrtshat/2d0,0d0,0d0,sqrtshat/2d0]
        born(:,2)=[sqrtshat/2d0,0d0,0d0,-sqrtshat/2d0]
        born(:,3)=[energy,0.3d0*momentum,0.4d0*momentum,sqrt(0.75d0)*momentum]
        born(:,4)=born(:,1)+born(:,2)-born(:,3)
        do branch=1,1+mass_case
          center=[.45d0,sqrt(((1d0+.805d0)/2d0-5d-7)/(1d0-5d-7)),.317d0]
          call cutoff_point(-100,center,j0,coordinate0,angle0)
          ratio=xiimax/xinorm
          if(branch.eq.2)then
            center(1)=(1d0+ratio)/2d0
            if(native_mapping)center(1)=.99d0
          endif
          shifted=center
          shifted(1)=center(1)+step
          call cutoff_point(-100,shifted,jwork,coordinate_plus,angle0)
          shifted(1)=center(1)-step
          call cutoff_point(-100,shifted,jwork,coordinate_minus,angle0)
          radial_derivative=(coordinate_plus-coordinate_minus)/(2d0*step)
          shifted=center
          shifted(2)=center(2)+step
          call cutoff_point(-100,shifted,jwork,coordinate0,angle_plus)
          shifted(2)=center(2)-step
          call cutoff_point(-100,shifted,jwork,coordinate0,angle_minus)
          angular_derivative=(angle_plus-angle_minus)/(2d0*step)
          expected=2d0*pi*abs(radial_derivative*angular_derivative)
          call cutoff_point(-100,center,j0,coordinate0,angle0)
          if(abs(j0/expected-1d0).gt.2d-8)error stop 'FSR Jacobian differs from sampled-coordinate derivative'
          if(mass_case.eq.1.and.mapping.eq.0.and.isign.ne.3-2*branch) &
               error stop 'FSR Jacobian fixture missed massive branch'
          invborn=0d0
          jacinv=1d0
          pswgtinv=1d0
          if(mass_case.eq.0)then
            call generate_momenta_massless_final_inverse(p,xi,yij,2d0*pi*center(3), &
                 invborn,inv,jacinv,pswgtinv,shat,sqrtshat,5,3)
          else
            call generate_momenta_massive_final_inverse(p,xi,yij,2d0*pi*center(3), &
                 invborn,inv,jacinv,pswgtinv,shat,sqrtshat,5,3,mass)
          endif
          if(jacinv.le.0d0.or.maxval(abs(inv-center)).gt.1d-9.or. &
               abs(jacinv/expected-1d0).gt.2d-8)error stop 'FSR inverse cutoff Jacobian'
          ! A projected endpoint retains the sampling measure of its real
          ! coordinate; massive second solutions have no soft endpoint.
          if(branch.eq.1)then
            do slot=-2,2
              if(mass_case.eq.1.and.slot.ne.0)cycle
              call cutoff_point(slot,center,jwork,coordinate0,angle0)
              if(abs(jwork/j0-1d0).gt.1d-12)error stop 'FSR endpoint cutoff Jacobian'
            enddo
          endif
        enddo
        if(mapping.ne.0)cycle
        ! Diagnostic limit tests turn the cutoffs off and omit the Jacobian
        ! for their fixed coordinate, independently of the endpoint slot.
        softtest=.true.
        xi_fix=.02d0
        call cutoff_point(-100,center,jwork,coordinate0,angle0)
        expected=2d0*pi*4d0*center(2)
        if(abs(jwork/expected-1d0).gt.1d-12)error stop 'FSR soft-test Jacobian'
        softtest=.false.
        if(mass_case.ne.0)cycle
        colltest=.true.
        y_fix=.2d0
        call cutoff_point(-100,center,jwork,coordinate0,angle0)
        expected=2d0*pi*2d0*center(1)
        if(abs(jwork/expected-1d0).gt.1d-12)error stop 'FSR collinear-test Jacobian'
        colltest=.false.
      enddo
    enddo
    native_mapping=.false.
  end subroutine

  subroutine cutoff_point(slot,coordinates,jacobian,radial,angle)
    integer,intent(in) :: slot
    double precision,intent(in) :: coordinates(3)
    double precision,intent(out) :: jacobian,radial,angle
    p=0d0
    p(:,1:4)=born
    jacobian=2d0*pi
    pswgt=1d0
    if(mass.eq.0d0)then
      call generate_momenta_massless_final(slot,5,3,born(:,3),shat,sqrtshat,coordinates,mrec**2,p, &
           2d0*pi*coordinates(3),xiimax,xinorm,xi,angle,xihat,pifks,jacobian,pswgt,pass)
    else
      call generate_momenta_massive_final(slot,isign,5,3,born(:,3),shat,sqrtshat,mass,coordinates,mrec**2,p, &
           2d0*pi*coordinates(3),xiimax,xinorm,xi,angle,xihat,pifks,jacobian,pswgt,pass)
    endif
    if(.not.pass.or.jacobian.le.0d0)error stop 'invalid FSR cutoff-Jacobian fixture'
    yij=angle
    radial=xi/xinorm
    if(native_mapping.and.mass.gt.0d0)radial=1d0-sqrt(sum(p(1:3,3)**2))/momentum
  end subroutine

  subroutine check_massive_boundary()
    use fks_phase_space_helpers, only: get_massive_fsr_bounds
    integer,parameter :: qp=selected_real_kind(30)
    double precision,parameter :: angles(3)=[-0.805d0,-0.4d0,-0.98d0]
    double precision :: ratio,xim,xibm,ximax,a,b,c,d,offset,expected_xi
    double precision :: emitted_xi,opening,azimuth,tolerance
    real(qp) :: wq,eiq,mq,mrq,yq,aq,bq,cq,dq,uq,ejq,ubq,weightq
    integer :: recoil_case,angle_case,power,side,expected_sign
    native_mapping=.false.
    mass=173d0
    sqrtshat=1000d0
    shat=sqrtshat**2
    do recoil_case=0,1
      mrec=173d0*recoil_case
      energy=(shat+mass**2-mrec**2)/(2d0*sqrtshat)
      momentum=sqrt(energy**2-mass**2)
      born(:,1)=[sqrtshat/2d0,0d0,0d0,sqrtshat/2d0]
      born(:,2)=[sqrtshat/2d0,0d0,0d0,-sqrtshat/2d0]
      born(:,3)=[energy,0.3d0*momentum,0.4d0*momentum,sqrt(0.75d0)*momentum]
      born(:,4)=born(:,1)+born(:,2)-born(:,3)
      do angle_case=1,size(angles)
        opening=angles(angle_case)
        call get_massive_fsr_bounds(shat,sqrtshat,mass,mrec**2,opening, &
             xim,xibm,ximax,a,b,c,d,xiimax,xinorm)
        ratio=xiimax/xinorm
        if(ratio.le.0d0.or.ratio.ge.1d0)error stop 'boundary fixture lacks two solutions'
        do power=12,4,-1
          offset=10d0**(-power)
          do side=-1,1,2
            opening=angles(angle_case)
            rnd=[ratio+side*offset,sqrt(((1d0-opening)/2d0-5d-7)/(1d0-5d-7)),0.317d0]
            phi=2d0*pi*rnd(3)
            p=0d0
            p(:,1:4)=born
            jac=2d0*pi
            pswgt=1d0
            call generate_momenta_massive_final(-100,isign,5,3,born(:,3), &
                 shat,sqrtshat,mass,rnd,mrec**2,p,phi,xiimax,xinorm,xi,yij, &
                 xihat,pifks,jac,pswgt,pass)
            expected_sign=-side
            if(.not.pass.or.isign.ne.expected_sign)error stop 'wrong massive boundary branch'
            ! The old map exceeded xi_max in a finite band and left a gap
            ! on the other branch; neither is a rounding-sized effect.
            if(xi.gt.xiimax+32d0*epsilon(xiimax).or.xi.lt.0d0) &
                 error stop 'massive boundary outside physical support'
            if(side.lt.0)then
              expected_xi=xiimax*(1d-6+(1d0-1d-6)*(rnd(1)/ratio)**2)
            else
              expected_xi=2d0*xiimax-xinorm*rnd(1)
            endif
            if(abs(xi-expected_xi).gt.32d0*epsilon(xi))error stop 'massive branches leave a gap'
            if(.not.all(ieee_is_finite(p)).or..not.ieee_is_finite(jac*pswgt)) &
                 error stop 'nonfinite point beside massive branch boundary'
            ! Independently solve energy conservation with higher precision:
            ! (W-E_i)*sqrt(u**2+m**2)+E_i*y*u=(W**2+m**2-M**2)/2-W*E_i.
            wq=real(sqrtshat,qp)
            eiq=real(xi,qp)*wq/2
            mq=real(mass,qp)
            mrq=real(mrec,qp)
            yq=real(yij,qp)
            aq=(wq-eiq)**2-(eiq*yq)**2
            bq=2*((wq*wq+mq*mq-mrq*mrq)/2-wq*eiq)*eiq*yq
            cq=(wq-eiq)**2*mq*mq-((wq*wq+mq*mq-mrq*mrq)/2-wq*eiq)**2
            dq=bq*bq-4*aq*cq
            if(dq.le.0)error stop 'nonphysical massive boundary quadratic'
            uq=(-bq+expected_sign*sqrt(dq))/(2*aq)
            ejq=sqrt(uq*uq+mq*mq)
            ubq=real(momentum,qp)
            weightq=2*wq*wq/(4*acos(-1._qp))**3*(uq/ubq)/ &
                 abs(2-real(xi,qp)*(1-ejq*yq/uq))
            ! Resolving a turning point amplifies floating-point errors by
            ! roughly epsilon/|x-ratio|; do not require a finite density at
            ! the exactly singular point itself.
            tolerance=2d-8+32d0*epsilon(xi)/offset
            if(abs(pswgt/real(weightq,8)-1d0).gt.tolerance) &
                 error stop 'massive boundary phase-space weight'
            emitted_xi=get_xi_from_p(5,p)
            opening=get_yij_from_p(5,3,p,saved_pi_cnt(:,0))
            azimuth=get_phi_from_p(5,3,p)
            jacinv=1d0
            pswgtinv=1d0
            call generate_momenta_massive_final_inverse(p,emitted_xi,opening,azimuth, &
                 invborn,inv,jacinv,pswgtinv,shat,sqrtshat,5,3,mass)
            if(jacinv.le.0d0.or.maxval(abs(inv-rnd)).gt.2d-8) &
                 error stop 'massive boundary inverse coordinates'
            if(maxval(abs(invborn(:,1:4)-born))/sqrtshat.gt.2d-9) &
                 error stop 'massive boundary inverse Born'
            if(abs(jacinv/jac-1d0).gt.2d-8.or. &
                 abs(pswgtinv/pswgt-1d0).gt.tolerance)error stop 'massive boundary inverse measure'
          enddo
        enddo
      enddo
    enddo
  end subroutine check_massive_boundary

  subroutine check_massive_born_measure()
    ! Integrate the actual endpoint measure over both massive branches.
    ! Removing the analytic soft radiation volume must leave one Born
    ! measure for each emitter, including when their branch boundaries differ.
    use fks_phase_space_data, only: p1_cnt,jac_cnt, &
         xiimax_ev,xinorm_ev,nocntevents,isolsign
    use fks_radiation_maps, only: fsr_born_cutoff_factor
    integer,parameter :: nx=16,ny=16
    double precision,parameter :: recoils(4)=[4.7d0,300d0,900d0,173d0]
    double precision :: mb(4),pmass(5)
    common/to_mass/pmass
    integer :: i_fks,j_fks
    common/fks_indices/i_fks,j_fks
    double precision :: rad(3),bounds(2),ratio,total,w,dx
    double precision :: soft_volume,born_flux
    integer :: sample,emitter,ix,iy,branch,minus_count
    logical :: supported(2)

    native_mapping=.false.
    sqrtshat=1000d0
    shat=sqrtshat**2
    i_fks=5
    soft_volume=shat/(16d0*pi**2)
    ! For a 2->3 call with unit Born sampling factors, compute_flux supplies
    ! 1/(2*s*(2*pi)**2); it is common to the real and soft configurations.
    born_flux=1d0/(2d0*shat*(2d0*pi)**2)
    minus_count=0
    do sample=1,size(recoils)
       mass=4.7d0
       if(sample.eq.4)mass=173d0
       mrec=recoils(sample)
       energy=(shat+mass**2-mrec**2)/(2d0*sqrtshat)
       momentum=sqrt(energy**2-mass**2)
       born(:,1)=[sqrtshat/2d0,0d0,0d0,sqrtshat/2d0]
       born(:,2)=[sqrtshat/2d0,0d0,0d0,-sqrtshat/2d0]
       born(:,3)=[energy,0.3d0*momentum,0.4d0*momentum,sqrt(0.75d0)*momentum]
       born(:,4)=born(:,1)+born(:,2)-born(:,3)
       mb=[0d0,0d0,mass,mrec]
       pmass=[mb,0d0]
       do emitter=1,2
          j_fks=emitter+2
          total=0d0
          do iy=1,ny
             rad=[0.1d0,(iy-0.5d0)/ny,0.137d0]
             call generate_massive_born_point(rad,born,mb)
             ratio=xiimax_ev/xinorm_ev
             do branch=1,2
                dx=ratio/nx
                if(branch.eq.2)dx=(1d0-ratio)/nx
                if(dx.eq.0d0)cycle
                do ix=1,nx
                   rad(1)=(ix-0.5d0)*dx
                   if(branch.eq.2)rad(1)=ratio+rad(1)
                   call generate_massive_born_point(rad,born,mb)
                   if(isolsign.eq.-1)then
                      minus_count=minus_count+1
                      if(.not.nocntevents.or.any(jac_cnt(0:2).gt.0d0).or. &
                           any(p1_cnt(0,1,0:2).gt.0d0)) &
                           error stop 'second branch acquired a soft counterevent'
                      w=0d0
                   else
                      if(nocntevents.or.jac_cnt(0).le.0d0) &
                           error stop 'first branch lost its soft counterevent'
                      if(maxval(abs(p1_cnt(:,1:4,0)-born)).gt.1d-9.or. &
                           any(p1_cnt(:,5,0).ne.0d0))error stop 'endpoint changed Born point'
                      ! Use the same sampled auxiliary volume as the Born
                      ! prefactor, keeping its integral exactly normalized.
                      w=jac_cnt(0)/born_flux*xinorm_ev/ &
                           (xiimax_ev*soft_volume*fsr_born_cutoff_factor())
                   endif
                   total=total+w*dx/ny
                enddo
             enddo
          enddo
          if(abs(total-1d0).gt.1d-10)then
             write(*,*)'Born normalization',sample,emitter,total
             error stop 'massive Born measure is not normalized'
          endif
       enddo
       ! Keep exactly the same Born point and radiation coordinates. The
       ! unequal-mass pair has different branches for its two emitters.
       if(sample.eq.2)then
          rad=[0.1d0,0.999d0,0.137d0]
          do emitter=1,2
             j_fks=emitter+2
             call generate_massive_born_point(rad,born,mb)
             bounds(emitter)=xiimax_ev/xinorm_ev
          enddo
          if(abs(bounds(1)-bounds(2)).lt.1d-3)error stop 'fixture has no branch difference'
          rad(1)=sum(bounds)/2d0
          do emitter=1,2
             j_fks=emitter+2
             call generate_massive_born_point(rad,born,mb)
             supported(emitter)=jac_cnt(0).gt.0d0
          enddo
          if(supported(1).eqv.supported(2))error stop 'sector-dependent branch support lost'
       endif
    enddo
    if(minus_count.eq.0)error stop 'normalization never tested the second branch'
  end subroutine check_massive_born_measure

  subroutine generate_massive_born_point(rad,born_point,mb)
    use fks_phase_space, only: generate_FKS_kinematics
    use fks_phase_space_data, only: p_born,p_born_l
    double precision,intent(in) :: rad(3),born_point(0:3,4),mb(4)
    double precision :: jac0,q(0:3,5),xb(2)
    logical :: good
    p_born=born_point
    p_born_l=born_point
    jac0=1d0
    xb=1d0
    good=.true.
    call generate_FKS_kinematics(rad,.true.,jac0,1d0,shat,shat,sqrtshat, &
         1d0,0d0,xb,mb,jac,q,good)
    if(.not.good.or.jac.le.0d0.or..not.all(ieee_is_finite(q))) &
         error stop 'invalid massive radiation point'
    if(maxval(abs(p_born-born_point)).ne.0d0)error stop 'radiation changed shared Born input'
  end subroutine

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
    local_xi=get_xi_from_p(5,real_p)
    local_y=get_yij_from_p(5,3,real_p,saved_pi_cnt(:,0))
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
    call generate_momenta_massive_final(0,sign_branch,5,3,born_p(:,3), &
         shat,sqrtshat,mass,native_x,mrec**2,q,local_phi,xmax,xnorm,local_xi,local_y, &
         xhat,scaled,cj,cp,good)
    if(.not.good)error stop 'invalid native soft counterevent'
    ratio=cj*cp/native_measure
    native_mapping=.false.
    q(:,1:4)=born_p
    cj=2d0*pi
    cp=1d0
    call generate_momenta_massive_final(0,sign_branch,5,3,born_p(:,3), &
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
