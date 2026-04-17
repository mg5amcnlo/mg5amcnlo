module spherical_harmonic
  implicit none
  private
  public::SphericalHarmonicY,SpinorSphericalHarmonicY

  integer,parameter ::dp=kind(1.0d0)
  complex(dp),parameter ::I_c=(0.0_dp,1.0_dp)
  real(dp),parameter :: PI = 3.14159265358979323846264338327950288_dp

contains

  !----------------------------------------------------------------
  !> Compute spinor spherical harmonic Y_{kappa m}(theta, phi)
  !> kappa : non-zero integer (...,-3,-2,-1,1,2,3,...)
  !> m     : real(dp) in [-j, j] (it must be half integers)
  !> theta : polar angle in radians (0..pi)
  !> phi   : azimuthal angle in radians (0..2*pi)
  !> j=abs(kappa)-0.5, l=kappa if kappa > 0; -1-kappa if kappa < 0
  !> Returns a complex(dp) array with two elements
  !----------------------------------------------------------------
  function SpinorSphericalHarmonicY(kappa,m,theta,phi) result(Y)
    implicit none
    integer,intent(in)::kappa
    real(dp),intent(in)::m,theta,phi
    complex(dp),dimension(2)::Y
    real(dp)::j,rl,pref
    integer::l,mm1,mm2
    real(dp),parameter::tiny=1d-6
    complex(dp)::Y1,Y2
    if(kappa.eq.0)then
       print *, 'SpinorSphericalHarmonicY error: kappa = 0'
       stop
    endif
    if(abs(2*m-int(2*m)).gt.tiny)then
       print *, 'SpinorSphericalHarmonicY error: m is an not half integer'
       stop
    endif
    Y(1:2)=0
    if(kappa.lt.0)then
       ! aligned spin: j=l+1/2, kappa=-(j+1/2)
       j=-real(kappa,dp)-0.5_dp
       if(m.gt.(j+tiny).or.m.lt.-(j+tiny))then
          print *, 'SpinorSphericalHarmonicY error: |m|>j #1'
          stop
       endif
       l=-kappa-1
       rl=real(l,dp)
       pref=1.0_dp/sqrt(2.0_dp*rl+1.0_dp)

       mm1=int(m-0.5_dp)
       if(mm1.eq.-l-1)then
          Y(1)=0
       else
          Y1=SphericalHarmonicY(l,mm1,theta,phi)
          Y(1)=pref*sqrt(rl+m+0.5_dp)*Y1
       endif
       
       mm2=int(m+0.5_dp)
       if(mm2.eq.l+1)then
          Y(2)=0
       else
          Y2=SphericalHarmonicY(l,mm2,theta,phi)
          Y(2)=pref*sqrt(rl-m+0.5_dp)*Y2
       endif
    else
       ! anti-aligned spin: j=l-1/2, kappa=+(j+1/2)
       j=real(kappa,dp)-0.5_dp
       if(m.gt.(j+tiny).or.m.lt.-(j+tiny))then
          print *, 'SpinorSphericalHarmonicY error: |m|>j #2'
          stop
       endif
       l=kappa
       rl=real(l,dp)
       pref=1.0_dp/sqrt(2.0_dp*rl+1.0_dp)
       
       mm1=int(m-0.5_dp)
       Y1=SphericalHarmonicY(l,mm1,theta,phi)
       Y(1)=-pref*sqrt(rl-m+0.5_dp)*Y1
       
       mm2=int(m+0.5_dp)
       Y2=SphericalHarmonicY(l,mm2,theta,phi)
       Y(2)=pref*sqrt(rl+m+0.5_dp)*Y2
    endif
    return
  end function SpinorSphericalHarmonicY

  !----------------------------------------------------------------
  !> Compute normalized complex spherical harmonic Y_l^m(theta,phi)
  !> l     : non-negative integer (0,1,2,...)
  !> m     : integer in [-l, l]
  !> theta : polar angle in radians (0..pi)
  !> phi   : azimuthal angle in radians (0..2*pi)
  !> Returns complex(dp) value of Y_lm
  !> Conventions: uses Condon-Shortley phase in associated Legendre P_l^m.
  !> It is same as the Mathematica function SphericalHarmonicY[l,m,theta,phi]
  !----------------------------------------------------------------
  function SphericalHarmonicY(l, m, theta, phi) result(Y)
    implicit none
    integer, intent(in) :: l, m
    real(dp), intent(in) :: theta, phi
    complex(dp) :: Y
    
    integer::mm,abs_m
    real(dp)::x,P,norm,fac_ratio
    complex(dp)::phase

    if (l.lt.0) then
      print *, 'SphericalHarmonicY error: l must be >= 0'
      stop
    end if
    if (abs(m).gt.l) then
      print *, 'SphericalHarmonicY error: |m| > l'
      stop
    end if

    x = cos(theta)
    mm = abs(m)
    P=associated_legendreP(l,mm,x)
    fac_ratio=factorial_ratio(l,mm)
    norm=sqrt((2.0_dp*l+1.0_dp)/(4.0_dp*PI)*fac_ratio)
    phase=exp(I_c*real(mm,dp)*phi)
    Y=norm*P*phase
    if(m.lt.0)then
       ! Use relation Y_{l,-m} = (-1)^m conj(Y_{l,m})
       if(mod(mm,2).eq.0)then
          Y=conjg(Y)
       else
          Y=-conjg(Y)
       end if
    endif
    
    return
  end function SphericalHarmonicY

  !----------------------------------------------------------------
  !> associated_legendreP(l,m,x)
  !> Computes P_l^m(x) for 0 <= m <= l using stable recurrence.
  !> Includes Condon-Shortley factor (-1)^m.
  !> Same as the Mathematica function LegendreP[l,m,x] with m>=0
  !----------------------------------------------------------------
  function associated_legendreP(l, m, x) result(P_lm)
    implicit none
    integer,intent(in)::l,m
    real(dp),intent(in)::x
    real(dp)::P_lm
    integer::i
    real(dp)::pmm,pmmp1,pll,somx2
    
    if (m.lt.0.or.m.gt.l) then
       print *, 'associated_legendreP error: invalid m'
       stop
    end if
    
    ! Compute P_m^m(x)
    if(m.eq.0)then
       pmm=1.0_dp
    else
       somx2=sqrt(1.0_dp-x*x)
       pmm = 1.0_dp
       do i = 1, m
          pmm=pmm*(-(2.0_dp*i-1.0_dp)*somx2)  ! includes (-1)^m factor
       end do
    end if
    
    if(l.eq.m)then
       P_lm = pmm
       return
    end if
    
    ! Compute P_{m+1}^m(x)
    pmmp1=x*(2*m+1)*pmm
    if(l.eq.m+1)then
       P_lm=pmmp1
       return
    end if
    
    ! Upward recurrence for l > m+1
    pll=0.0_dp
    do i = m + 2, l
       pll = ((2*i-1)*x*pmmp1-(i+m-1)*pmm)/(i-m)
       pmm = pmmp1
       pmmp1 = pll
    end do
    
    P_lm = pmmp1
    return
  end function associated_legendreP

  !----------------------------------------------------------------
  !> factorial_ratio(l,m) returns (l-m)!/(l+m)! as a real(dp)
  !> computed as product to avoid overflow where possible.
  !----------------------------------------------------------------
  function factorial_ratio(l, m) result(r)
    implicit none
    integer,intent(in)::l,m
    real(dp)::r
    integer::k
    integer::numerator_start,numerator_end
    ! For m >= 0, (l-m)!/(l+m)! = 1 / [ (l+m)*(l+m-1)*...*(l-m+1) ]
    if (m.lt.0.or.m.gt.l) then
       print *, 'factorial_ratio error: invalid m'
       stop
    end if

    if (m.eq.0) then
       r = 1.0_dp
       return
    end if

    r = 1.0_dp
    do k = l-m+1, l+m
       r = r/real(k,dp)
    end do
  end function factorial_ratio

end module spherical_harmonic

