! Test the production phase-space routines without matrix elements or PDFs.
program check_momentum_maps
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
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
  integer :: isign,i,j,k,nplus,nminus,nchecked
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

  if (mode.eq.'massive') then
     mass=173d0
     mrec=173d0
  elseif (mode.eq.'massless') then
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
           nchecked=nchecked+1
        enddo
     enddo
  enddo
  if (mode.eq.'massive'.and.nminus.eq.0) error stop 'minus solution not covered'
  write(*,*) 'PASS ',trim(mode),nchecked,nplus,nminus
end program check_momentum_maps
