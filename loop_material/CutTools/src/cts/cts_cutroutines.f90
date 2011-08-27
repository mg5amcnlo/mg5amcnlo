  subroutine ctsxcut(rootsvalue,muscalein,number_propagators, &
                     numdummy,mpnumdummy,rnk,p,m2,amp,ar1,stable,forcemp)
  use loopfunctions
  use coefficients
  implicit none
  logical, intent(in) :: forcemp
  integer, intent(in) :: number_propagators
  external numdummy,mpnumdummy
  integer, intent(in) :: rnk
  include 'cts_dpr.h'
   , intent(in) :: rootsvalue,muscalein
  include 'cts_dpr.h'
   , intent(in), dimension(0:3,0:(number_propagators-1)) :: p
  include 'cts_dpc.h'
   , intent(in), dimension(0:(number_propagators-1)) :: m2
  include 'cts_dpc.h'
   , dimension (0:2), intent(out) :: amp
  include 'cts_dpc.h' 
   :: amp0
  include 'cts_dpc.h'
   , intent(out) :: ar1
  include 'cts_dpc.h'
   , dimension (0:2) :: amp1
  include 'cts_mpc.h'
   , dimension (0:2) :: mp_amp,mp_amp1
  include 'cts_mpc.h' 
   :: mp_amp0
  include 'cts_mpr.h' 
   :: mlt_prec
  include 'cts_dpr.h' 
   :: dbl_prec
  type(propagator), dimension(0:(number_propagators-1)) :: dn
  type(mp_propagator), dimension(0:number_propagators-1) :: mp_dn 
  integer :: i,j,ib,k,dmr
  logical, intent(out) :: stable
  logical :: passed
  if (number_propagators.gt.maxden) then
   stop 'increase maxden in cts_combinatorics.f90'
  endif
  if (muscalein.ne.muscale) then
     if (scaloop.eq.2) then
!
! set the scale in avh
!
        muscale= muscalein
        call avh_olo_mu_set(muscale)
     elseif (scaloop.eq.3) then
        muscale= muscalein
        call avh_olo_mu_set(muscale)
!
! set the scale in qcdloop
!
        musq= muscalein**2
     else
        stop 'value of scaloop not allowed'
     endif
  endif
  dmr =  number_propagators-rnk
  if (dmr.eq.-1) then
    if (number_propagators.gt.4) then
      print*,'dmr=',dmr,' not implemented yet with',&
     ' number_propagators=',number_propagators
      stop
    endif 
  endif
!
! set the internal scale of CutTools (the result should not depend on that)
!
  roots= rootsvalue
!
!!!!!!!!!!!!!!!!!!!!
!                  !
! double precision !
!                  !
!!!!!!!!!!!!!!!!!!!!
!
  stable=.true.
!
! define and load the dp propagators
!
  do i= 0,(number_propagators-1)
   dn(i)%p =  p(:,i)
   dn(i)%m2=  m2(i)
  enddo
  call load_denominators(dn,number_propagators)
!
! compute the loop functions (in double precision only)
!
  call getloop(number_propagators)
!
! get the coefficients in double precision
!
  call get_coefficients(dbl_prec,numdummy,number_propagators,dmr &
                       ,roots)
!
! compute the cc part of amp and amp1 
!
  call computeampcc
  call computeamp1cc
!
! save the cc result for the finite part
!
  amp0   = amp(0)
!
! add R1 before performing the test
!
  amp(0) = amp(0)+save_rat1
  amp1(0)= amp1(0)+rat1
!
! perform the test in double precision
!
  call dptest(passed)
!
! the cc part and the R1 piece in double precision:
!
  amp(0)= amp0
  ar1   = save_rat1
  if (forcemp) goto 8
  if (passed) then
    return
  else
    if (.not.mpflag) goto 9
  endif
!
!!!!!!!!!!!!!!!!!!!!
!                  !
! multiprecision   !
!                  !
!!!!!!!!!!!!!!!!!!!!
!
8 n_mp= n_mp+1
!
! define and load the mp propagators
!
  do j= 0,number_propagators-1
    do k= 0,3; mp_dn(j)%p(k)= dn(j)%p(k); enddo
               mp_dn(j)%m2  = dn(j)%m2
  enddo
  call load_denominators(mp_dn,number_propagators)
!
! get the coefficients in multiprecision
!
  call get_coefficients(mlt_prec,mpnumdummy,number_propagators,dmr &
                       ,roots)
!
! compute the cc part of mp_amp and mp_amp1 
!
  call mpcomputeampcc
  call mpcomputeamp1cc
!
! save the cc result for the finite part
!
  mp_amp0  = mp_amp(0)
!
! add R1 before performing the test
!
  mp_amp(0) = mp_amp(0)+save_mp_rat1
  mp_amp1(0)= mp_amp1(0)+mp_rat1
!
! perform the test in multiple precision
!
  call mptest(passed)
!
! the cc part and the R1 piece in multiprecision:
!
  do k= 0,2; amp(k)= mp_amp(k); enddo
  amp(0)= mp_amp0
  ar1   = save_mp_rat1
  if (.not.passed) goto 9
  return
9 n_disc= n_disc+1
  stable=.false.
  contains
!
  subroutine computeampcc
!
!   compute the cc part of the amplitude 
!   in double precision
!
    amp = 0.d0
    if (number_propagators.ge.4) then
     do i= 1,nbn4(number_propagators)
      ib= mbn4(number_propagators,i) 
      do k= 0,2; amp(k)= amp(k)+save_dcoeff(0,ib)*dloopfun(k,ib); enddo
     enddo
    endif
    if (number_propagators.ge.3) then
     do i= 1,nbn3(number_propagators)
      ib= mbn3(number_propagators,i)
      do k= 0,2; amp(k)= amp(k)+save_ccoeff(0,ib)*cloopfun(k,ib); enddo
     enddo
    endif
    if (number_propagators.ge.2) then
     do i= 1,nbn2(number_propagators)
      ib= mbn2(number_propagators,i)
      do k= 0,2
       amp(k)= amp(k)+ save_bcoeff(0,ib)               *bloopfun(k,ib)  &
                     +(save_bcoeff(3,ib)*vveck1(ib))   *b1loopfun(k,ib) &
                     +(save_bcoeff(6,ib)*vveck1(ib)**2)*b11loopfun(k,ib) 
      enddo
     enddo
    endif
    if (number_propagators.ge.1) then
     do i= 1,nbn1(number_propagators)
      ib= mbn1(number_propagators,i)
      do k= 0,2; amp(k)= amp(k)+save_acoeff(0,ib)*aloopfun(k,ib); enddo
     enddo
    endif 
  end subroutine computeampcc
!
  subroutine computeamp1cc
!
!   compute the cc part of the second determination of the amplitude 
!   in double precision
!
    amp1= 0.d0
    if (number_propagators.ge.4) then
     do i= 1,nbn4(number_propagators)
      ib= mbn4(number_propagators,i) 
      do k= 0,2; amp1(k)= amp1(k)+save_dcoeff(0,ib)*dloopfun(k,ib); enddo
     enddo
    endif
    if (number_propagators.ge.3) then
     do i= 1,nbn3(number_propagators)
      ib= mbn3(number_propagators,i)
      do k= 0,2; amp1(k)= amp1(k)+ccoeff(0,ib)*cloopfun(k,ib); enddo
     enddo
    endif
    if (number_propagators.ge.2) then
     do i= 1,nbn2(number_propagators)
      ib= mbn2(number_propagators,i)
      do k= 0,2
       amp1(k)= amp1(k)+ bcoeff(0,ib)               *bloopfun(k,ib)  &
                       +(bcoeff(3,ib)*vveck1(ib))   *b1loopfun(k,ib) &
                       +(bcoeff(6,ib)*vveck1(ib)**2)*b11loopfun(k,ib) 
      enddo
     enddo
    endif
    if (number_propagators.ge.1) then
     do i= 1,nbn1(number_propagators)
      ib= mbn1(number_propagators,i)
      do k= 0,2; amp1(k)= amp1(k)+acoeff(0,ib)*aloopfun(k,ib); enddo
     enddo
    endif
  end subroutine computeamp1cc
!
  subroutine mpcomputeampcc
!
!   compute the cc part of the amplitude 
!   in multiprecision
!
    do k= 0,2; mp_amp(k)= 0.d0; enddo
    if (number_propagators.ge.4) then
     do i= 1,nbn4(number_propagators)
      ib= mbn4(number_propagators,i) 
      do k= 0,2; mp_amp(k)= mp_amp(k)+save_mp_dcoeff(0,ib)*dloopfun(k,ib); enddo
     enddo
    endif
    if (number_propagators.ge.3) then
     do i= 1,nbn3(number_propagators)
      ib= mbn3(number_propagators,i)
      do k= 0,2; mp_amp(k)= mp_amp(k)+save_mp_ccoeff(0,ib)*cloopfun(k,ib); enddo
     enddo
    endif
    if (number_propagators.ge.2) then
     do i= 1,nbn2(number_propagators)
      ib= mbn2(number_propagators,i)
      do k= 0,2
       mp_amp(k)= mp_amp(k)+ save_mp_bcoeff(0,ib)            *bloopfun(k,ib)  &
                     +(save_mp_bcoeff(3,ib)*mp_vveck1(ib))   *b1loopfun(k,ib) &
                     +(save_mp_bcoeff(6,ib)*mp_vveck1(ib)**2)*b11loopfun(k,ib) 
      enddo
     enddo
    endif
    if (number_propagators.ge.1) then
     do i= 1,nbn1(number_propagators)
      ib= mbn1(number_propagators,i)
      do k= 0,2; mp_amp(k)= mp_amp(k)+save_mp_acoeff(0,ib)*aloopfun(k,ib); enddo
     enddo
    endif
  end subroutine mpcomputeampcc
!
  subroutine mpcomputeamp1cc
!
!   compute the cc part of the second determination of the amplitude 
!   in multiprecision
!
    do k= 0,2; mp_amp1(k)= 0.d0; enddo
    if (number_propagators.ge.4) then
     do i= 1,nbn4(number_propagators)
      ib= mbn4(number_propagators,i) 
      do k= 0,2; mp_amp1(k)= mp_amp1(k)+save_mp_dcoeff(0,ib)*dloopfun(k,ib); enddo
     enddo
    endif
    if (number_propagators.ge.3) then
     do i= 1,nbn3(number_propagators)
      ib= mbn3(number_propagators,i)
      do k= 0,2; mp_amp1(k)= mp_amp1(k)+mp_ccoeff(0,ib)*cloopfun(k,ib); enddo
     enddo
    endif
    if (number_propagators.ge.2) then
     do i= 1,nbn2(number_propagators)
      ib= mbn2(number_propagators,i)
      do k= 0,2
       mp_amp1(k)= mp_amp1(k)+ mp_bcoeff(0,ib)            *bloopfun(k,ib)  &
                       +(mp_bcoeff(3,ib)*mp_vveck1(ib))   *b1loopfun(k,ib) &
                       +(mp_bcoeff(6,ib)*mp_vveck1(ib)**2)*b11loopfun(k,ib) 
      enddo
     enddo
    endif
    if (number_propagators.ge.1) then
     do i= 1,nbn1(number_propagators)
      ib= mbn1(number_propagators,i)
      do k= 0,2; mp_amp1(k)= mp_amp1(k)+mp_acoeff(0,ib)*aloopfun(k,ib); enddo
     enddo
    endif
  end subroutine mpcomputeamp1cc
!
  subroutine dptest(passed)
    logical, intent(out) :: passed
    include 'cts_dpr.h' 
     :: prec 
    passed=.true.
! comment
!     prec= abs(amp1(0)-amp(0))/max(tiny(prec),abs(amp1(0)))
!     print*,'           '
!     print*,' amp(0)+R_1=',amp(0) 
!     print*,'amp1(0)+R_1=',amp1(0) 
!     print*,'prec,limit=',prec,limit     
!     print*,'           '
! comment
    if(abs(amp1(0)-amp(0)).gt.limit*abs(amp1(0))) passed=.false.
    if (.not.stablen) then
     passed=.false.
! comment
!     print*,'   '
!     print*,'Instable Numerator found in double precision!   '
!     print*,'stablen=',stablen
!     print*,'   '
! comment
    endif
  end subroutine dptest
!
  subroutine mptest(passed)
    logical, intent(out) :: passed
    include 'cts_mpr.h' 
     :: aus1,aus2     
    include 'cts_mpr.h' 
     :: mp_prec 
    passed=.true.
! comment
!     mp_prec= max(mp_tiny(mp_prec),abs(mp_amp1(0)))
!     mp_prec= abs(mp_amp1(0)-mp_amp(0))/mp_prec
!     print*,'           '
!     aus= mp_amp(0)
!     print*,' mp_amp(0)+R_1=',aus 
!     aus= mp_amp1(0)
!     print*,'mp_amp1(0)+R_1=',aus
!     aus= mp_prec
!     print*,'prec,limit    =',real(aus),limit     
!     print*,'           '
! comment
    aus1= abs(mp_amp1(0)-mp_amp(0))
    aus2= limit*abs(mp_amp1(0))
    if (aus1.gt.aus2) passed=.false.
    if (.not.stablen) then
      passed=.false.
! comment
!     print*,'   '
!     print*,'Instable Numerator found in multiprecision!   '
!     print*,'stablen=',stablen
!     print*,'   '
! comment
    endif
  end subroutine mptest
  end subroutine ctsxcut
