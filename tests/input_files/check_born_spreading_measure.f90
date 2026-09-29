! Integrate a fitted, nonconstant table against the actual FKS soft-map
! Jacobians at fixed Born kinematics, including the massive-emitter support.
program check_born_measure
  use mint_module
  implicit none
  integer, parameter :: nexternal=5,nincoming=2
  double precision, parameter :: pi=3.1415926535897932385d0
  double precision :: pmass(nexternal),xinorm_ev,xiimax_ev
  integer :: i_fks,j_fks
  common /to_mass/pmass
  common /fks_indices/i_fks,j_fks
  common /cxinormev/xinorm_ev
  common /cxiimaxev/xiimax_ev
  logical :: softtest,colltest
  common /sctests/softtest,colltest
  double precision :: tau_born_lower_bound,tau_lower_bound_resonance,tau_lower_bound
  common /ctau_lower_bound/tau_born_lower_bound,tau_lower_bound_resonance,tau_lower_bound
  double precision :: omx_ee(2)
  common /to_ee_omx1/omx_ee
  integer :: ix,iy,kind,energy,isign
  double precision :: b(1),c(1),base,spread,weight,x(2),lo,hi,dy,ratio
  double precision :: xp(0:3,nexternal),pborn(0:3),p_i(0:3),mass,shat,sqrts,shat_born
  double precision :: xjac,xpswgt,y,xi,xi_hat,tau,ycm,xbj(2),xborn(2),rat
  character(len=16) :: arg
  logical :: pass

  call born_spread_configure(.true.,nexternal,nincoming,4,7)
  call get_command_argument(1,arg)
  if (trim(arg).eq.'load') then
     call born_spread_load_table
     stop
  endif
  born_spread_phase=1
  do iy=1,born_spread_ny
     do ix=1,born_spread_nxi
        call born_spread_set_point((ix-0.5d0)/born_spread_nxi,(iy-0.5d0)/born_spread_ny)
        b=1d0
        c=-4d0
        ! Populate backward angles, where the massive map has two solutions.
        if (ix.gt.born_spread_nxi/2.or.iy.le.born_spread_ny/2) then
           b=-1d0
           c=-1d0
        endif
        call born_spread_observe_sample(b,c,1)
     enddo
  enddo
  born_spread_phase=0
  call solve_born_spreading_table
  if (maxval(born_spread_factor)-minval(born_spread_factor).lt.1d0) stop 1
  call born_spread_write_table
  born_spread_factor=0d0
  call born_spread_load_table
  softtest=.false.
  colltest=.false.
  omx_ee=0d0
  tau_born_lower_bound=0d0
  tau_lower_bound_resonance=0d0
  tau_lower_bound=0d0
  i_fks=5
  do energy=1,2
     sqrts=370d0
     if (energy.eq.2) sqrts=1000d0
     shat=sqrts**2
     do kind=1,4
        j_fks=min(kind,3)
        mass=0d0
        if (kind.eq.4) mass=173d0
        pmass=0d0
        pmass(3:4)=mass
        base=0d0
        spread=0d0
        do iy=1,born_spread_ny
           lo=sqrt(dble(iy-1)/born_spread_ny)
           hi=sqrt(dble(iy)/born_spread_ny)
           x(2)=(lo+hi)/2d0
           dy=hi-lo
           x(1)=0.1d0
           call radiation_map
           ratio=1d0
           if (kind.eq.4) ratio=xiimax_ev/xinorm_ev
           do ix=1,born_spread_nxi
              lo=ratio*sqrt(dble(ix-1)/born_spread_nxi)
              hi=ratio*sqrt(dble(ix)/born_spread_nxi)
              x(1)=(lo+hi)/2d0
              call radiation_map
              ! compute_prefactors_nbody's radiation factor, including phi.
              weight=2d0*pi*xjac*xpswgt*xinorm_ev/xiimax_ev*16d0*pi**2/shat
              weight=weight*(hi-lo)*dy
              call set_born_spread_point(x(1),x(2),2)
              ! Restore the saved fold after a later map changed the bin.
              call born_spread_set_point(0.99d0,0.99d0)
              born_spread_current_bin=born_spread_bin_fold(2)
              base=base+weight
              spread=spread+weight*born_spread_get_factor()
           enddo
        enddo
        if (abs(base-1d0).gt.2d-6) stop 2 ! FKS endpoint offsets
        if (abs(spread/base-1d0).gt.1d-11) then
           write(*,*) 'FAIL Born radiation normalization',kind,sqrts,base,spread
           stop 3
        endif
     enddo
  enddo
  write(*,*) 'PASS Born radiation normalization'

contains
  subroutine radiation_map
    xp=0d0
    xp(0,1:4)=sqrts/2d0
    xp(3,1)=sqrts/2d0
    xp(3,2)=-sqrts/2d0
    xp(3,3)=sqrt(shat/4d0-mass**2)
    xp(3,4)=-xp(3,3)
    pborn=xp(:,3)
    xjac=1d0
    xpswgt=1d0
    if (kind.le.2) then
       xborn=(/0.2d0,0.3d0/)
       shat_born=shat
       call generate_momenta_initial(0,i_fks,j_fks,xborn,product(xborn), &
            0.5d0*log(xborn(1)/xborn(2)),0d0,shat_born,0.3d0,xp,x,shat, &
            shat_born/product(xborn),sqrts,tau,ycm,xbj,p_i,xiimax_ev,xinorm_ev, &
            xi,y,xi_hat,xpswgt,xjac,pass)
    elseif (kind.eq.3) then
       call generate_momenta_massless_final(0,i_fks,j_fks,pborn,shat,sqrts, &
            x,mass**2,xp,0.3d0,xiimax_ev,xinorm_ev,xi,y,xi_hat,p_i,xjac,xpswgt,pass)
    else
       call generate_momenta_massive_final(0,isign,.false.,rat,i_fks,j_fks, &
            pborn,shat,sqrts,mass,x,mass**2,xp,0.3d0,xiimax_ev,xinorm_ev, &
            xi,y,xi_hat,p_i,xjac,xpswgt,pass)
    endif
    if (.not.pass) stop 4
  end subroutine radiation_map
end program check_born_measure

double precision function ran2()
  ! No sampling is needed for this deterministic quadrature.
  stop 99
end function ran2
