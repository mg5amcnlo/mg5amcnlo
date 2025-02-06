module scale_module
  use process_module
  use kinematics_module
  implicit none
  double precision,public,allocatable,dimension(:,:) :: shower_scale_nbody, &
       shower_scale_nbody_max,shower_scale_nbody_min&
       &,shower_scale_n1body,showerscaleS,showerscaleH
  double precision,public,allocatable,dimension(:,:,:,:) :: emsca_S&
       &,emsca_H
!  double precision,public :: SCALUP
  double precision,private :: global_ref_scale,shower_scale_factor
  double precision,private,parameter :: frac_low=0.1d0,frac_upp=1.0d0
  double precision,private,parameter :: scaleMClow=0d0,scaleMCdelta=3d0
  double precision,private,parameter :: scaleMCcut=3d0
  integer,public :: born_flow_picked
  public :: compute_shower_scale_nbody,compute_shower_scale_n1body, &
       init_scale_module,Bornonly_shower_scale,get_random_shower_dipole_scale, &
       get_born_flow,determine_partner
  private
contains
  
  subroutine init_scale_module(nexternal,shower_scale_factor_in,nfks,nfold)
    implicit none
    integer :: nexternal,nfks,nfold
    double precision :: shower_scale_factor_in
    if (.not.allocated(shower_scale_nbody)) &
         allocate(shower_scale_nbody(nexternal-1,nexternal-1))
    if (.not.allocated(shower_scale_nbody_max)) &
         allocate(shower_scale_nbody_max(nexternal-1,nexternal-1))
    if (.not.allocated(shower_scale_nbody_min)) &
         allocate(shower_scale_nbody_min(nexternal-1,nexternal-1))
    if (.not.allocated(shower_scale_n1body)) &
         allocate(shower_scale_n1body(nexternal,nexternal))
    if (.not.allocated(emsca_S)) &
         allocate(emsca_S(nfks,nfold,ndelS,ndelS))
    if (.not.allocated(emsca_H)) &
         allocate(emsca_H(nfks,nfold,ndelH,ndelH))
    if (.not.allocated(showerscaleS)) &
         allocate(showerscaleS(ndelS,ndelS))
    if (.not.allocated(showerscaleH)) &
         allocate(showerscaleH(ndelH,ndelH))
    shower_scale_factor=shower_scale_factor_in
  end subroutine init_scale_module
    
  subroutine compute_shower_scale_nbody(p,flow_picked)
    implicit none
    integer :: i,j,flow_picked,iflow_min,iflow_max
    double precision,dimension(0:3,next_n) :: p
    double precision :: ref_scale,scalemin,scalemax,rrnd
    double precision, external :: ran2
    shower_scale_nbody=-1d0
    shower_scale_nbody_min=-1d0
    shower_scale_nbody_max=-1d0
    call get_global_ref_scale(next_n,p)
    if (ickkw_mod.eq.3) then
       ! For FxFx, the scale should be the smallest clustering scale as
       ! returned by the clustering routine. This is the global_ref_scale
       shower_scale_nbody=shower_scale_factor*global_ref_scale
       shower_scale_nbody_min=shower_scale_factor*global_ref_scale
       shower_scale_nbody_max=shower_scale_factor*global_ref_scale+scaleMCdelta
       return
    endif
    if (flow_picked.gt.0) then
       iflow_min=flow_picked
       iflow_max=flow_picked
    else
       ! check valid_dipole for any possible flow
       iflow_min=1
       iflow_max=max_flows_n
    endif
    do i=1,next_n-1
       do j=i+1,next_n
          if (.not. any(valid_dipole_n(i,j,iflow_min:iflow_max))) cycle
          ref_scale=get_ref_scale_dipole(next_n,p,i,j)
          call get_scaleminmax(ref_scale,scalemin,scalemax)
          rrnd=ran2()
          rrnd=damping_inv(rrnd,1d0)
          scalemin=max(scalemin,scaleMCcut)
          scalemax=max(scalemax,scalemin+scaleMCdelta)
          shower_scale_nbody(i,j)=scalemin+rrnd*(scalemax-scalemin)
          shower_scale_nbody_min(i,j)=scalemin
          shower_scale_nbody_max(i,j)=scalemax
          ! symmetrize the matrix:
          shower_scale_nbody(j,i)=shower_scale_nbody(i,j)
          shower_scale_nbody_min(j,i)=shower_scale_nbody_min(i,j)
          shower_scale_nbody_max(j,i)=shower_scale_nbody_max(i,j)
       enddo
    enddo
  end subroutine compute_shower_scale_nbody

  subroutine compute_shower_scale_n1body(p,i_fks,j_fks)
    implicit none
    double precision,dimension(0:3,next_n1) :: p
    integer i,j,ii,i_fks,j_fks
    double precision ref_scale,scalemin,scalemax
    call get_global_ref_scale(next_n1,p)
    do i=1,next_n1
       do j=1,next_n1
          if (valid_dipole_n1(i,j)) then
             ref_scale=get_ref_scale_dipole(next_n1,p,i,j)
             call get_scaleminmax(ref_scale,scalemin,scalemax)
             scalemax=max(scalemax,scaleMCcut)
             shower_scale_n1body(i,j)=scalemax
          elseif ((i.eq.i_fks .and. j.eq.j_fks) .or. (j.eq.i_fks .and. i.eq.j_fks)) then
             ! find the partner of i_fks and j_fks
             ref_scale=99d99
             do ii=1,next_n1
                if (valid_dipole_n1(ii,i_fks)) then
                   ref_scale=min(get_ref_scale_dipole(next_n1,p,ii,i_fks),ref_scale)
                elseif( valid_dipole_n1(ii,j_fks)) then
                   ref_scale=min(get_ref_scale_dipole(next_n1,p,ii,j_fks),ref_scale)
                endif
             enddo
             call get_scaleminmax(ref_scale,scalemin,scalemax)
             scalemax=max(scalemax,scaleMCcut)
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
    if (ickkw_mod.eq.3) then
       ! For FxFx, the scale should be the smallest clustering scale as
       ! returned by the clustering routine. This is the global_ref_scale
       shower_scale_nbody=shower_scale_factor*global_ref_scale
       shower_scale_nbody_min=-1d0
       shower_scale_nbody_max=-1d0
       return
    endif
    do i=1,next_n
       do j=1,next_n
          if (valid_dipole_n(i,j,flow_picked)) then
             shower_scale_nbody(i,j)=max(get_ref_scale_dipole(next_n,p,i,j),scaleMCcut)
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
! WARNING: Shower scale depends on xm12: This is the mass^2 of j_fks. Hence,
! this introduces FKS info into Pythia6 subtraction terms.
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
    integer :: i,j,n
    double precision,dimension(0:3,n) :: p
    get_ref_scale_dipole=min(sqrt(max(0d0,sumdot(p(0,i),p(0,j),1d0))) &
         ,global_ref_scale)
  end function get_ref_scale_dipole
  
  integer function colour(n,i)
    implicit none
    integer :: n,i
    if (n.eq.next_n) then
       colour=colour_n(i)
    elseif (n.eq.next_n1) then
       colour=colour_n1(i)
    endif
  end function colour

  double precision function mass(n,i)
    implicit none
    integer :: n,i
    if (n.eq.next_n) then
       mass=mass_n(i)
    elseif (n.eq.next_n1) then
       mass=mass_n1(i)
    endif
  end function mass

  subroutine get_global_ref_scale(n,p)
    ! this is the global reference shower scale (i.e., without damping),
    ! i.e. the smallest scale returned by the clustering routine.
    implicit none
    integer :: n
    double precision,dimension(0:3,n) :: p,pQCD
    integer :: i,j,NN,iproc
    integer :: nFxFx_ren_scales
    double precision,dimension(0:next_n1) :: FxFx_ren_scales
    double precision,dimension(2) :: FxFx_fac_scale
    integer,dimension(next_n1) :: need_matching
    double precision :: dummy1,dummy2
    logical,parameter :: for_mcatnlo_scale=.true.
    INTEGER              NFKSPROCESS
    COMMON/C_NFKSPROCESS/NFKSPROCESS
    if (n.eq.next_n1) then
       iproc=nFKSprocess
    else
       iproc=0
    endif
    call cluster_and_reweight(iproc,dummy1 &
            ,dummy2,nFxFx_ren_scales,FxFx_ren_scales(0) &
            ,fxfx_fac_scale(1),need_matching,for_mcatnlo_scale)
    global_ref_scale=minval(FxFx_ren_scales(0:nFxFx_ren_scales))
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
