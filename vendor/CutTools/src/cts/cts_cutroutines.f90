  subroutine ctsxcut(imode,rootsvalue,muscalein,number_propagators, &
                     numdummy,mpnumdummy,rnk,p,m2,amp,ampcc,ampr1,stable)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                            !
! imode:|  actions performed by ctsxcut:                                     !
!       |                                                                    !
!   0   | (dp_dir,dp_inv)-> dp_Atest -> stable -> (only if stable=.false.) ->!
!       | (mp_dir,mp_inv)-> mp_Atest -> stable                               ! 
!   1   | (dp_dir)       -> dp_Ntest -> stable                               !
!   2   | (dp_inv)       -> dp_Ntest -> stable                               !
!   3   | (dp_dir,dp_inv)-> dp_Atest -> stable                               !
!   4   | (mp_dir)       -> mp_Ntest -> stable                               !  
!   5   | (mp_inv)       -> mp_Ntest -> stable                               ! 
!   6   | (mp_dir,mp_inv)-> mp_Atest -> stable                               !
!                                                                            !
! Legenda:                                                                   !
!                                                                            !
! dp_dir    = compute amp in double precision with normal   propagator order !
! dp_inv    = compute amp in double precision with reversed propagator order !
! mp_dir    = compute amp in multi  precision with normal   propagator order !
! mp_inv    = compute amp in multi  precision with reversed propagator order !
! dp_Atest  = perform the A=A test in double precision                       !
! mp_Atest  = perform the A=A test in multi  precision                       !
! dp_Ntest  = perform the N=N test in double precision                       !
! mp_Ntest  = perform the N=N test in multi  precision                       !
! -> stable = set stable=.true. or stable=.false.                            !
!             according to the outcome of the test                           !
!                                                                            !
! Tests:                                                                     !
!                                                                            !
! -The N=N test is a test on the reconstructed OPP integrand performed       !
!  by comparing original and reconstacted integrands at an arbirtary value   !
!  of the integration momentum.                                              !
!                                                                            ! 
! -The A=A test checks the 2 amplitudes obtained with dir and inv orders.    !
!                                                                            !
! Notes:                                                                     !
!                                                                            ! 
! a) imode= 0 is recommended, unless you really know what you are doing.     !
!                                                                            !
! b) When two determinations of amp are available, that one with more        !
!    accurate recounstructed numerator (coming from the N=N test) is used.   !
!                                                                            !  
! c) When running in multi precision with scaloop= 3 (qcdloop), the loop     !
!    functions are computed in double precision only. A full multi           !
!    precision result can only be obtained with scaloop= 2 (OneLoop).        !
!                                                                            !  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  include 'cts_mprec.h'
  use scale
  use combinatorics
  use mbnvalues
  use loopfunctions
  use mp_loopfunctions
  use coefficients
  use avh_olo
  use dimensions
  use denominators 
  use maxnumden
  use def_propagator                                       
  use def_mp_propagator                                       
  implicit none
  integer, intent(in) :: imode
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
   , intent(out) :: amp(0:2),ampcc,ampr1
  include 'cts_dpc.h'
   :: amp1(0:2),amp1cc,amp1r1
  include 'cts_mpc.h'
   :: mp_amp(0:2),mp_ampcc,mp_ampr1
  include 'cts_mpc.h'
   :: mp_amp1(0:2),mp_amp1cc,mp_amp1r1
  include 'cts_mpr.h' 
   :: mlt_prec
  include 'cts_dpr.h' 
   :: dbl_prec
  type(propagator), dimension(0:(number_propagators-1)) :: dn
  type(mp_propagator), dimension(0:number_propagators-1) :: mp_dn 
  integer :: i,j,ib,k,dmr,ierr
  logical, intent(out) :: stable
  logical :: stablen1,stablen2
  logical :: passeddp,passedmp
  logical :: dir_stable,mpcomputation
  logical :: firsttime=.true.
  include 'cts_dpr.h' 
   :: precstablen1,precstablen2
  save firsttime
  if (number_propagators.gt.maxden) then
   stop 'increase maxden in cts_combinatorics.f90'
  endif
  if ((imode.lt.0).or.(imode.gt.6)) then
   stop 'wrong input value of imode in ctsxcut'
  endif
!
! Allocating local arrays
!
  ierr= -1
  if (number_propagators.ge.1) then 
   dmns_1= nbn1(number_propagators)
   allocate  (bbn1(number_propagators,dmns_1), stat=ierr)
   if (ierr.ne.0) STOP "Not enough memory to allocate bbn1"
   bbn1= 0
  endif
  if (number_propagators.ge.2) then 
   dmns_2= nbn2(number_propagators)
   allocate  (bbn2(number_propagators,dmns_2), stat=ierr)
   if (ierr.ne.0) STOP "Not enough memory to allocate bbn2"
   bbn2= 0
  endif
  if (number_propagators.ge.3) then 
   dmns_3= nbn3(number_propagators)
   allocate  (bbn3(number_propagators,dmns_3), stat=ierr)
   if (ierr.ne.0) STOP "Not enough memory to allocate bbn3"
   bbn3= 0
  endif
  if (number_propagators.ge.4) then 
   dmns_4= nbn4(number_propagators)
   allocate  (bbn4(number_propagators,dmns_4), stat=ierr)
   if (ierr.ne.0) STOP "Not enough memory to allocate bbn4"
   bbn4= 0
  endif
!
  do i= 1,number_propagators  
   if (number_propagators.ge.1) then
    do j= 1,dmns_1; bbn1(i,j)= bn1(number_propagators,i,j); enddo
   endif
   if (number_propagators.ge.2) then
    do j= 1,dmns_2; bbn2(i,j)= bn2(number_propagators,i,j); enddo
   endif
   if (number_propagators.ge.3) then
    do j= 1,dmns_3; bbn3(i,j)= bn3(number_propagators,i,j); enddo
   endif
   if (number_propagators.ge.4) then
    do j= 1,dmns_4; bbn4(i,j)= bn4(number_propagators,i,j); enddo
   endif
  enddo
!
! count the number of calls to ctsxcut
!
  n_tot= n_tot+1
!
! set the internal scale of CutTools (the result should not depend on that)
!
  roots  = rootsvalue
  muscale= muscalein
  if (scaloop.eq.2) then
    call olo_scale(muscale)
  elseif (scaloop.eq.3) then
!
!   set the scale in qcdloop
!
    musq= muscale**2
!
!   some 2-point OneLoop 2-point function is also used
!
    call olo_scale(muscale)
  else
    stop 'value of scaloop not allowed'
  endif
!
  dmr =  number_propagators-rnk
!
! comment
!  if (dmr.eq.-1) then
!    if (number_propagators.gt.4) then
!      print*,'dmr=',dmr,' not implemented yet with',&
!     ' number_propagators=',number_propagators
!      stop
!    endif 
!  endif
! comment
!
  stable  =.true.
  passeddp=.true.
  passedmp=.true.
  mpcomputation=.false.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                               !
! double precision normal order !
!           (dir)               !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  if ((imode.eq.0).or.(imode.eq.1).or.(imode.eq.3)) then
!
!   define and load the dp propagators
!
    do i= 0,(number_propagators-1)
     dn(i)%p =  p(:,i) 
     dn(i)%m2=  m2(i)
    enddo
    call load_denominators(dn,number_propagators)
!
!   compute the loop functions (in double precision only)
!
    call getloop(number_propagators)
!
!   get the coefficients in double precision
!
    call get_coefficients(dbl_prec,numdummy,number_propagators,dmr &
                         ,roots)
!
!   compute the amplitude, the CC part and R1 
!
    call computeamp(amp,ampcc,ampr1)
!
    stablen1= stablen
    if ((imode.eq.1).and.(.not.stablen1)) call tag_as_unstable
    precstablen1= precstablen 
  endif
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                !
! double precision inverse order !
!           (inv)                !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  if ((imode.eq.0).or.(imode.eq.2).or.(imode.eq.3)) then
!
!   define and load the dp propagators with inverse order 
!
    do i= 0,(number_propagators-1)
     k= (number_propagators-1)-i
     dn(i)%p =  p(:,k)
     dn(i)%m2=  m2(k)
    enddo
    call load_denominators(dn,number_propagators)
!
!   compute the loop functions (in double precision only)
!
    call getloop(number_propagators)
!
!   get the coefficients in double precision
!
    call get_coefficients(dbl_prec,numdummy,number_propagators,dmr &
                         ,roots)
!
!   compute the amplitude, the CC part and R1 
!
    call computeamp(amp1,amp1cc,amp1r1)
!
    stablen2= stablen
    if ((imode.eq.2).and.(.not.stablen2)) call tag_as_unstable
    precstablen2= precstablen 
  endif
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                      !
! perform the test in double precision !
!                                      !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  if ((imode.eq.0).or.(imode.eq.3)) then
    call dptest(passeddp)
    if ((imode.eq.3).and.(.not.passeddp)) call tag_as_unstable
  endif 
!
! The cases when multiprecision is called:  
!
  if (((imode.eq.0).and.(.not.passeddp)).or. &
      (imode.eq.4).or.(imode.eq.5).or.(imode.eq.6)) then
    mpcomputation=.true.
    n_mp= n_mp+1
    if (scaloop.eq.2) then
      call olo_scale(muscale)
    else
      if (firsttime) then
       firsttime=.false.
       print*,'  '
       print*,'WARNING: multiprecision unavailable for one-loop scalar functions' 
       print*,'with the chosen value of scaloop=',scaloop
       print*,'Only the CutTools part is computed in multiprecision.'
       print*,'  '
      endif
    endif
  endif
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                             !
! multiprecision normal order !
!         (dir)               !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  if (((imode.eq.0).and.(.not.passeddp)).or.(imode.eq.4).or.(imode.eq.6)) then
!
!   define and load the mp propagators
!
    do i= 0,(number_propagators-1)
     dn(i)%p =  p(:,i)
     dn(i)%m2=  m2(i)
    enddo
    call load_denominators(dn,number_propagators)
!
    do j= 0,number_propagators-1
      do k= 0,3; mp_dn(j)%p(k)= dn(j)%p(k); enddo
                 mp_dn(j)%m2  = dn(j)%m2
    enddo
    call load_denominators(mp_dn,number_propagators)
!
!   compute the loop functions in multiprecision 
!
    call get_mp_loop(number_propagators)
!
!   get the coefficients in multiprecision
!
    call get_coefficients(mlt_prec,mpnumdummy,number_propagators,dmr &
                         ,roots)
!
!   compute the amplitude, the CC part and R1 
!
    call mpcomputeamp(mp_amp,mp_ampcc,mp_ampr1)
!
    stablen1= stablen
    if ((imode.eq.4).and.(.not.stablen1)) call tag_as_unstable
    precstablen1= precstablen 
  endif
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                              !
! multiprecision inverse order !
!         (inv)                !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  if (((imode.eq.0).and.(.not.passeddp)).or.(imode.eq.5).or.(imode.eq.6)) then
!
!   define and load the mp propagators with inverse order
!
    do i= 0,(number_propagators-1)
     k= (number_propagators-1)-i
     dn(i)%p =  p(:,k)
     dn(i)%m2=  m2(k)
    enddo
    call load_denominators(dn,number_propagators)
!
    do j= 0,number_propagators-1
      do k= 0,3; mp_dn(j)%p(k)= dn(j)%p(k); enddo
                 mp_dn(j)%m2  = dn(j)%m2
    enddo
    call load_denominators(mp_dn,number_propagators)
!
!   compute the loop functions in multiprecision
!
    call get_mp_loop(number_propagators)
!
!   get the coefficients in multiprecision
!
    call get_coefficients(mlt_prec,mpnumdummy,number_propagators,dmr &
                         ,roots)
!
!   compute the amplitude, the CC part and R1 
!
    call mpcomputeamp(mp_amp1,mp_amp1cc,mp_amp1r1)
!
    stablen2= stablen
    if ((imode.eq.5).and.(.not.stablen2)) call tag_as_unstable
    precstablen2= precstablen 
  endif
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                        !
! perform the test in multiple precision !
!                                        ! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
  if (((imode.eq.0).and.(.not.passeddp)).or.(imode.eq.6)) then
    call mptest(passedmp)
    if (.not.passedmp) call tag_as_unstable
  endif
!
! The final output according to all possible cases:
!
  if     (imode.eq.0) then
    if ((stable).and.(.not.mpcomputation)) then
      if (dir_stable) then
       call from_dir_dp
      else
       call from_inv_dp
      endif 
    else
      if (dir_stable) then
       call from_dir_mp
      else
       call from_inv_mp
      endif 
    endif
  elseif (imode.eq.1) then    
    call from_dir_dp
  elseif (imode.eq.2) then
    call from_inv_dp
  elseif (imode.eq.3) then
    if (dir_stable) then
     call from_dir_dp
    else
     call from_inv_dp
    endif 
  elseif (imode.eq.4) then
    call from_dir_mp
  elseif (imode.eq.5) then
    call from_inv_mp
  elseif (imode.eq.6) then
    if (dir_stable) then
      call from_dir_mp
    else
      call from_inv_mp
    endif 
  else
   stop 'wrong value of imode in ctsxcut'
  endif
!
! Deallocating local arrays
!
  ierr= -1
  if (number_propagators.ge.1) then 
   deallocate  (bbn1, stat=ierr)
   if (ierr.ne.0) STOP "bbn1 NOT deallocated"
  endif
  if (number_propagators.ge.2) then 
   deallocate  (bbn2, stat=ierr)
   if (ierr.ne.0) STOP "bbn2 NOT deallocated"
  endif
  if (number_propagators.ge.3) then 
   deallocate  (bbn3, stat=ierr)
   if (ierr.ne.0) STOP "bbn3 NOT deallocated"
  endif
  if (number_propagators.ge.4) then 
   deallocate  (bbn4, stat=ierr)
   if (ierr.ne.0) STOP "bbn4 NOT deallocated"
  endif
  contains
!
  subroutine tag_as_unstable
    stable=.false.
    n_unst= n_unst+1
  end subroutine tag_as_unstable
!
  subroutine from_dir_dp
  end subroutine from_dir_dp
!
  subroutine from_inv_dp
    amp   = amp1
    ampcc = amp1cc
    ampr1 = amp1r1
  end subroutine from_inv_dp
!
  subroutine from_dir_mp
    do k= 0,2
      amp(k)= mp_amp(k)
    enddo
    ampcc = mp_ampcc
    ampr1 = mp_ampr1
  end subroutine from_dir_mp
!
  subroutine from_inv_mp
    do k= 0,2
     amp(k)   = mp_amp1(k)
    enddo
    ampcc = mp_amp1cc
    ampr1 = mp_amp1r1
  end subroutine from_inv_mp
!
  subroutine computeamp(camp,campcc,campr1)
!
!   compute the cc and r1 parts of the amplitude 
!   in double precision
!
    include 'cts_dpc.h'
     , intent(out) :: camp(0:2),campcc,campr1
    camp = 0.d0
    if (number_propagators.ge.4) then
     do ib= 1,dmns_4
      do k= 0,2; camp(k)= camp(k)+save_dcoeff(0,ib)*dloopfun(k,ib); enddo
     enddo
    endif
    if (number_propagators.ge.3) then
     do ib= 1,dmns_3
      do k= 0,2; camp(k)= camp(k)+save_ccoeff(0,ib)*cloopfun(k,ib); enddo
     enddo
    endif
    if (number_propagators.ge.2) then
     do ib= 1,dmns_2
      do k= 0,2
       camp(k)= camp(k)+ save_bcoeff(0,ib)               *bloopfun(k,ib)  &
                       +(save_bcoeff(3,ib)*vveck1(ib))   *b1loopfun(k,ib) &
                       +(save_bcoeff(6,ib)*vveck1(ib)**2)*b11loopfun(k,ib) 
      enddo
     enddo
    endif
    if (number_propagators.ge.1) then
     do ib= 1,dmns_1
      do k= 0,2; camp(k)= camp(k)+save_acoeff(0,ib)*aloopfun(k,ib); enddo
     enddo
    endif 
    campcc= camp(0)
    campr1= save_rat1
    camp(0)= camp(0)+campr1
  end subroutine computeamp
!
  subroutine mpcomputeamp(mp_camp,mp_campcc,mp_campr1)
!
!   compute the cc and r1 parts of the amplitude 
!   in multiprecision
!
    include 'cts_mpc.h'
     , intent(out) :: mp_camp(0:2),mp_campcc,mp_campr1
    do k= 0,2; mp_camp(k)= 0.d0; enddo
    if (number_propagators.ge.4) then
     do ib= 1,dmns_4
      do k= 0,2
        mp_camp(k)= mp_camp(k)+save_mp_dcoeff(0,ib)*mp_dloopfun(k,ib)
      enddo
     enddo
    endif
    if (number_propagators.ge.3) then
     do ib= 1,dmns_3
      do k= 0,2
       mp_camp(k)= mp_camp(k)+save_mp_ccoeff(0,ib)*mp_cloopfun(k,ib)
      enddo
     enddo
    endif
    if (number_propagators.ge.2) then
     do ib= 1,dmns_2
      do k= 0,2
       mp_camp(k)= mp_camp(k)+ save_mp_bcoeff(0,ib)    *mp_bloopfun(k,ib)  &
               +(save_mp_bcoeff(3,ib)*mp_vveck1(ib))   *mp_b1loopfun(k,ib) &
               +(save_mp_bcoeff(6,ib)*mp_vveck1(ib)**2)*mp_b11loopfun(k,ib) 
      enddo
     enddo
    endif
    if (number_propagators.ge.1) then
     do ib= 1,dmns_1
      do k= 0,2
       mp_camp(k)= mp_camp(k)+save_mp_acoeff(0,ib)*mp_aloopfun(k,ib)
      enddo
     enddo
    endif
    mp_campcc = mp_camp(0)
    mp_campr1 = save_mp_rat1
    mp_camp(0)= mp_camp(0)+mp_campr1
  end subroutine mpcomputeamp
!
  subroutine dptest(passed)
    logical, intent(out) :: passed
    include 'cts_dpr.h' 
     :: prec 
    passed=.true.
    dir_stable=.true.
    if (precstablen2.lt.precstablen1) dir_stable=.false.
! comment
!     prec= abs(amp1(0)-amp(0))/max(my_tiny(prec),abs(amp1(0)))
!     print*,'           '
!     print*,'amp(0)      =',amp(0) 
!     print*,'amp1(0)     =',amp1(0) 
!     print*,'prec,limit  =',prec,limit     
!     print*,'precstablen1=',precstablen1 
!     print*,'precstablen2=',precstablen2 
!     print*,'           '
! comment
    if(abs(amp1(0)-amp(0)).gt.limit*abs(amp1(0))) passed=.false.
    if ((.not.stablen1).or.(.not.stablen2)) then
     passed=.false.
! comment
!     print*,'   '
!     print*,'Instable Numerator found in double precision!   '
!     print*,'stablen1,stablen2=',stablen1,stablen2
!     print*,'precstablen1     =',precstablen1 
!     print*,'precstablen2     =',precstablen2 
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
    dir_stable=.true.
    if (precstablen2.lt.precstablen1) dir_stable=.false.
! comment
!     mp_prec= max(my_tiny(mp_prec),abs(mp_amp1(0)))
!     mp_prec= abs(mp_amp1(0)-mp_amp(0))/mp_prec
!     print*,'           '
!     aus= mp_amp(0)
!     print*,'mp_amp(0)   =',aus 
!     aus= mp_amp1(0)
!     print*,'mp_amp1(0)  =',aus
!     aus= mp_prec
!     print*,'prec,limit  =',real(aus),limit      
!     print*,'precstablen1=',precstablen1 
!     print*,'precstablen2=',precstablen2 
!     print*,'           '
! comment
    aus1= abs(mp_amp1(0)-mp_amp(0))
    aus2= limit*abs(mp_amp1(0))
    if (aus1.gt.aus2) passed=.false.
    if ((.not.stablen1).or.(.not.stablen2)) then
      passed=.false.
! comment
!     print*,'   '
!     print*,'Instable Numerator found in multiprecision!   '
!     print*,'stablen1,stablen2=',stablen1,stablen2
!     print*,'precstablen1     =',precstablen1 
!     print*,'precstablen2     =',precstablen2 
!     print*,'   '
! comment
    endif
  end subroutine mptest
  end subroutine ctsxcut
