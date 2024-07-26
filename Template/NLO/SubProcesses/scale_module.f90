module scale_module
  use process_module
  use kinematics_module
  implicit none
!  include 'nFKSconfigs.inc'
  double precision,public,allocatable,dimension(:,:) :: shower_scale_nbody, &
       shower_scale_nbody_max,shower_scale_nbody_min
!  double precision,public :: emsca(nFKSconfigs)
  double precision,public :: emsca(1000),SCALUP
  integer,public :: flow_picked,partner_picked
  double precision,private :: global_ref_scale,shower_scale_factor

  double precision,private,parameter :: frac_low=0.1d0,frac_upp=1.0d0
  double precision,private,parameter :: scaleMClow=10d0,scaleMCdelta=20d0
  double precision,private,parameter :: scaleMCcut=3d0

  public :: compute_shower_scale_nbody,init_scale_module,Bornonly_shower_scale,&
            get_random_shower_dipole_scale,get_born_flow,determine_partner
  private

contains
  
  subroutine init_scale_module(nexternal,shower_scale_factor_in)
    implicit none
    integer :: nexternal
    double precision :: shower_scale_factor_in
    if (.not.allocated(shower_scale_nbody)) &
         allocate(shower_scale_nbody(nexternal-1,nexternal-1))
    if (.not.allocated(shower_scale_nbody_max)) &
         allocate(shower_scale_nbody_max(nexternal-1,nexternal-1))
    if (.not.allocated(shower_scale_nbody_min)) &
         allocate(shower_scale_nbody_min(nexternal-1,nexternal-1))
    shower_scale_factor=shower_scale_factor_in
  end subroutine init_scale_module
    
  subroutine compute_shower_scale_nbody(p,flow_picked)
    implicit none
    integer :: i,j,flow_picked,fks_father
    double precision,dimension(0:3,next_n) :: p
    double precision :: ref_scale,scalemin,scalemax,rrnd
    double precision, external :: ran2
    ! loop over dipoles
    call get_global_ref_scale(next_n,p)
    if (flow_picked .lt. 0) then
       fks_father=-flow_picked
       do_i : do i=1,next_n
          if (i.eq.fks_father) cycle
          do j=1,max_flows_n
             if (.not. valid_dipole_n(i,fks_father,j)) cycle do_i
          enddo
!!$          if (.not. any(valid_dipole_n(i,fks_father,1:max_flows_n))) cycle
          ref_scale=get_ref_scale_dipole(next_n,p,i,fks_father)
          call get_scaleminmax(ref_scale,scalemin,scalemax)
          rrnd=ran2()
          rrnd=damping_inv(rrnd,1d0)
          scalemin=max(scalemin,scaleMCcut)
          scalemax=max(scalemax,scalemin+scaleMCdelta)
          shower_scale_nbody(i,fks_father)=scalemin+rrnd*(scalemax-scalemin)
          shower_scale_nbody_min(i,fks_father)=scalemin
          shower_scale_nbody_max(i,fks_father)=scalemax
          ! symmetrize the matrix:
          shower_scale_nbody(fks_father,i)=shower_scale_nbody(i,fks_father)
          shower_scale_nbody_min(fks_father,i)=shower_scale_nbody_min(i,fks_father)
          shower_scale_nbody_max(fks_father,i)=shower_scale_nbody_max(i,fks_father)
       enddo do_i
    elseif (flow_picked.gt.0) then
       do i=1,next_n
          do j=1,next_n
             if (valid_dipole_n(i,j,flow_picked)) then
                ref_scale=get_ref_scale_dipole(next_n,p,i,j)
                call get_scaleminmax(ref_scale,scalemin,scalemax)
                ! this breaks backward compatibility. In earlier versions, the
                ! shower_scale_nbody was constrained by the ptresc (which
                ! depended on the n+1-body and was used to decide if in life or
                ! dead zone). Also, now we randomize for each dipole separately
                ! also for non-delta.
                rrnd=ran2()
                rrnd=damping_inv(rrnd,1d0)
                scalemin=max(scalemin,scaleMCcut)
                scalemax=max(scalemax,scalemin+scaleMCdelta)
                shower_scale_nbody(i,j)=scalemin+rrnd*(scalemax-scalemin)
                shower_scale_nbody_min(i,j)=scalemin
                shower_scale_nbody_max(i,j)=scalemax
             else
                shower_scale_nbody(i,j)=-1d0
                shower_scale_nbody_min(i,j)=-1d0
                shower_scale_nbody_max(i,j)=-1d0
             endif
          enddo
       enddo
    else
       write (*,*) 'flow_picked is zero in compute_shower_scale_nbody',flow_picked
       stop 1
    endif
  end subroutine compute_shower_scale_nbody

  subroutine compute_shower_scale_n1body(p)
    implicit none
    double precision,dimension(0:3,next_n1) :: p
    call get_global_ref_scale(next_n1,p)
    do i=1,next_n1
       do j=1,next_n1
          if (valid_dipole_n1(i,j)) then
             ref_scale=get_ref_scale_dipole(next_n1,p,i,j)
             call get_scaleminmax(ref_scale,scalemin,scalemax)
             shower_scale_n1body(i,j)=scalemax
          else
             shower_scale_n1body(i,j)=-1d0
          endif
       enddo
    enddo
  end subroutine compute_shower_scale_n1body
  
  subroutine Bornonly_shower_scale(p,flow_picked)
    implicit none
    integer :: i,j,flow_picked
    double precision,dimension(0:3,next_n) :: p
    call get_global_ref_scale(next_n,p)
    do i=1,next_n
       do j=1,next_n
          if (valid_dipole_n(i,j,flow_picked)) then
             shower_scale_nbody(i,j)=get_ref_scale_dipole(next_n,p,i,j)
          else
             shower_scale_nbody(i,j)=-1d0
          endif
       enddo
    enddo
    shower_scale_nbody_min=-1d0
    shower_scale_nbody_max=-1d0
  end subroutine Bornonly_shower_scale

  subroutine get_scaleminmax(ref_scale,scalemin,scalemax)
    implicit none
    double precision :: ref_scale,scalemin,scalemax
    scalemin=max(shower_scale_factor*frac_low*ref_scale,scaleMClow)
    scalemax=max(shower_scale_factor*frac_upp*ref_scale, &
         scalemin+scaleMCdelta)
    scalemax=min(scalemax,collider_energy)
    scalemin=min(scalemin,scalemax)
    if(abrv_mod.ne.'born' .and. shower_mc_mod(1:7).eq.'PYTHIA6' .and. &
         ileg.eq.3)then
! TODO: Shower scale depends on xm12: This is the mass^2 of j_fks. Hence, this
! introduces FKS info into subtraction terms.
       scalemin=max(scalemin,sqrt(xm12))
       scalemax=max(scalemin,scalemax)
    endif
  end subroutine get_scaleminmax
           
  double precision function damping_fun(x,alpha)
    implicit none
    double precision :: x,alpha
    if(x.lt.0d0.or.x.gt.1d0)then
       write(*,*)'Fatal error in damping_fun'
       stop
    endif
    damping_fun=x**(2*alpha)/(x**(2*alpha)+(1-x)**(2*alpha))
  end function damping_fun

  double precision function damping_inv(r,alpha)
! Inverse of the damping function, implemented only for alpha=1 for the moment
    implicit none
    double precision :: r,alpha
    if(r.lt.0d0.or.r.gt.1d0.or.alpha.ne.1d0)then
       write(*,*)'Fatal error in damping_inv'
       stop
    endif
    damping_inv=sqrt(r)/(sqrt(r)+sqrt(1d0-r))
  end function damping_inv

  
  double precision function get_ref_scale_dipole(n,p,i,j)
    implicit none
    integer :: i,j
    double precision,dimension(0:3,n) :: p
    get_ref_scale_dipole=min(sqrt(max(0d0,sumdot(p(0,i),p(0,j),1d0))) &
         ,global_ref_scale)
  end function get_ref_scale_dipole
  
  
  subroutine get_global_ref_scale(n,p)
    ! this is the global reference shower scale (i.e., without damping),
    ! i.e. HT/2 for non-delta (no longer used), and shat reduced by kT of
    ! splitting, or ET of massive in case of delta (now for both delta and
    ! non-delta).
    implicit none
    double precision,dimension(0:3,n) :: p,pQCD
    integer :: i,j,NN
 ! start from s-hat      
       global_ref_scale=sqrt(2d0*dot(p(0,1),p(0,2)))
       NN=0
       do j=nincoming_mod+1,n
          if (abs(colour_n(j)).ne.1 .and. mass_n(j).eq.0d0) then
             NN=NN+1
             do i=0,3
                pQCD(i,NN)=p(i,j)
             enddo
          elseif (abs(colour_n(j)).ne.1 .and. abs(mass_n(j)).ne.0d0) then
             !     reduce by ET of massive QCD particles
             global_ref_scale=min(global_ref_scale,sqrt((p(0,j)+p(3,j))*(p(0,j)-p(3,j))))
          elseif (abs(colour_n(j)).ne.1 .and. abs(mass_n(j)).eq.0d0) then
             write (*,*) 'Error in assign_ref_scale(): colored' &
                  //' massless particle that does not enter jets'
             stop 1
          endif
       enddo
       ! reduce by kT-cluster scale of massless QCD partons
       if (NN.eq.1) then
          global_ref_scale=min(global_ref_scale,pt(pQCD(0,1)))
       elseif (NN.ge.2) then
          do i=1,NN
             do j=i+1,NN
                global_ref_scale=min(global_ref_scale,min(pt(pQCD(0,i)),pt(pQCD(0,j))) &
                                                      *deltaR(pQCD(0,i),pQCD(0,j)))
             enddo
             global_ref_scale=min(global_ref_scale,pt(pQCD(0,i)))
          enddo
       endif
  end subroutine get_global_ref_scale

  double precision function get_random_shower_dipole_scale()
    implicit none
    integer :: n_scales,i,j,iscale
    integer,dimension(next_n**2,2) :: dip
    double precision,external :: ran2
    n_scales=0
    do i=1,next_n
       do j=1,next_n
          if (shower_scale_nbody(i,j).gt.0d0) then
             n_scales=n_scales+1
             dip(n_scales,1)=i
             dip(n_scales,2)=j
          endif
       enddo
    enddo
    iscale=int(ran2()*n_scales)+1
    get_random_shower_dipole_scale=shower_scale_nbody(dip(iscale,1),dip(iscale,2))
  end function get_random_shower_dipole_scale
      
  subroutine determine_partner(flow_picked,partner_picked)
    use process_module
    use kinematics_module
    implicit none
    integer :: ndip(0:next_n),i,flow_picked,partner_picked
    double precision,external :: ran2
    ndip(0)=0
    do i=1,next_n
       if (valid_dipole_n(i,fksfather,flow_picked)) then
          ndip(0)=ndip(0)+1
          ndip(ndip(0))=i
       endif
    enddo
    if (ndip(0).eq.1) then
       partner_picked=ndip(1)
    elseif (ndip(0).eq.2) then
       if (ran2().lt.0.5d0) then
          partner_picked=ndip(1)
       else
          partner_picked=ndip(2)
       endif
    else
       write (*,*) 'Inconsistent dipoles',ndip
       stop 1
    endif
  end subroutine determine_partner

  subroutine get_born_flow(flow_picked)
    ! This assumes that the Born matrix elements are called. This is
    ! always the case if either the compute_born or the virtual
    ! (through bornsoftvirtual) are evaluated.
    implicit none
    include 'genps.inc'
    include "born_nhel.inc"
    integer :: flow_picked,i
    double precision :: sumborn,target,sum
    double precision,external :: ran2
    double Precision :: amp2(ngraphs),jamp2(0:ncolor)
    common/to_amps/  amp2         ,jamp2
    logical :: is_leading_cflow(max_bcol)
    integer :: num_leading_cflows
    common/c_leading_cflows/is_leading_cflow,num_leading_cflows
    ! sumborn is the sum of the leading colour flow contributions to the Born.
    sumborn=0.d0
    do i=1,max_bcol
       if(is_leading_cflow(i)) sumborn=sumborn+jamp2(i)
    enddo
    target=ran2()*sumborn
    sum=0d0
    do i=1,max_bcol
       if (.not.is_leading_cflow(i)) cycle
       sum=sum+jamp2(i)
       if(sum.gt.target) then
          flow_picked=i
          return
       endif
    enddo
    write (*,*) 'Error #1 in get_born_flow',sum,target,i
    stop 1
  end subroutine get_born_flow
  
end module scale_module
