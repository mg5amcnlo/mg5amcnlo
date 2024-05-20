module scale_module
  use kinematics_module
  use process_module
  implicit none
  double precision,public,allocatable,dimension(:,:) :: shower_scale_nbody, &
       shower_scale_nbody_nodamp
  double precision,private :: global_ref_scale,shower_scale_factor

  double precision,private,parameter :: frac_low=0.1d0,frac_upp=1.0d0
  double precision,private,parameter :: scaleMClow=10d0,scaleMCdelta=20d0
  double precision,private,parameter :: scaleMCcut=3d0

  public :: compute_shower_scale_nbody,init_scale_module,Bornonly_shower_scale
  private

contains
  subroutine init_scale_module(nexternal,shower_scale_factor_in)
    implicit none
    integer :: nexternal
    double precision :: shower_scale_factor_in
    if (.not.allocated(shower_scale_nbody)) &
         allocate(shower_scale_nbody(nexternal-1,nexternal-1))
    shower_scale_factor=shower_scale_factor_in
  end subroutine init_scale_module
    
  subroutine compute_shower_scale_nbody(p,flow_picked)
    implicit none
    integer :: i,j,flow_picked
    double precision,dimension(0:3,next_n) :: p
    double precision :: ref_scale,scalemin,scalemax,rrnd
    double precision, external :: ran2
    ! loop over dipoles
    call get_global_ref_scale(p)
    if (flow_picked .lt. 0) then
       fks_father=-flow_picked
       do i=1,next_n
          if (i.eq.fks_father) cycle
          if (.not. any(valid_dipole_n(i,fks_father,1:max_flows_n))) cycle
          ref_scale=get_ref_scale_dipole(p,i,fks_father)
          call get_scaleminmax(ref_scale,scalemin,scalemax)
          rrnd=ran2()
          rrnd=damping_inv(rrnd,1d0)
          shower_scale_nbody(i,fks_father)=max(scalemin+rrnd*(scalemax-scalemin),scaleMCcut)
          shower_scale_nbody_nodamp(i,fks_father)=max(scalemax,scaleMCcut)
       enddo
    elseif (flow_picked.gt.0) then
       do i=1,next_n
          do j=1,next_n
             if (valid_dipole_n(i,j,flow_picked)) then
                ref_scale=get_ref_scale_dipole(p,i,j)
                call get_scaleminmax(ref_scale,scalemin,scalemax)
                ! this breaks backward compatibility. In earlier versions, the
                ! shower_scale_nbody was constrained by the ptresc (which
                ! depended on the n+1-body and was used to decide if in life or
                ! dead zone). Also, now we randomize for each dipole separately
                ! also for non-delta.
                rrnd=ran2()
                rrnd=damping_inv(rrnd,1d0)
                shower_scale_nbody(i,j)=max(scalemin+rrnd*(scalemax-scalemin),scaleMCcut)
                shower_scale_nbody_nodamp(i,j)=max(scalemax,scaleMCcut)
             else
                shower_scale_nbody(i,j)=-1d0
                shower_scale_nbody_nodamp(i,j)=-1d0
             endif
          enddo
       enddo
    else
       write (*,*) 'flow_picked is zero in compute_shower_scale_nbody',flow_picked
       stop 1
    endif
  end subroutine compute_shower_scale_nbody

  subroutine Bornonly_shower_scale(p,flow_picked)
    implicit none
    integer :: i,j,flow_picked
    double precision,dimension(0:3,next_n) :: p
    call get_global_ref_scale(p)
    do i=1,next_n
       do j=1,next_n
          if (valid_dipole_n(i,j,flow_picked)) then
             shower_scale_nbody(i,j)=get_ref_scale_dipole(p,i,j)
             shower_scale_nbody_nodamp(i,j)=shower_scale_nbody(i,j)
          else
             shower_scale_nbody(i,j)=-1d0
             shower_scale_nbody_nodamp(i,j)=-1d0
          endif
       enddo
    enddo
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

  
  double precision function get_ref_scale_dipole(p,i,j)
    implicit none
    integer :: i,j
    double precision,dimension(0:3,next_n) :: p
!!$    if (.not.mcatnlo_delta_mod) then
!!$       get_ref_scale_dipole=global_ref_scale
!!$    else
       get_ref_scale_dipole=min(sqrt(max(0d0,sumdot(p(0,i),p(0,j),1d0))) &
                                ,global_ref_scale)
!!$    endif
  end function get_ref_scale_dipole
  
  
  subroutine get_global_ref_scale(p)
    ! this is the global reference shower scale (i.e., without damping),
    ! i.e. HT/2 for non-delta (no longer used), and shat reduced by kT of
    ! splitting, or ET of massive in case of delta (now for both delta and
    ! non-delta).
    implicit none
    double precision,dimension(0:3,next_n) :: p,pQCD
    integer :: i,j,NN
!!$    if (.not.mcatnlo_delta_mod) then
!!$       ! Sum of final-state transverse masses
!!$       global_ref_scale=0d0
!!$       do i=3,next_n
!!$          global_ref_scale=global_ref_scale+ &
!!$               dsqrt(max(0d0,(p(0,i)+p(3,i))*(p(0,i)-p(3,i))))
!!$       enddo
!!$       global_ref_scale=global_ref_scale/2d0
!!$    else
 ! start from s-hat      
       global_ref_scale=sqrt(2d0*dot(p(0,1),p(0,2)))
       NN=0
       do j=nincoming_mod+1,next_n
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
!!$    endif
  end subroutine get_global_ref_scale

  
end module scale_module
