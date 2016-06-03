  module countdigits 
   include 'cts_mprec.h'
   implicit none
   private 
   public :: ctscountdigits
   contains
!
   subroutine ctscountdigits(mcountd)
   include 'cts_mpr.h'
    :: x
   integer :: i,digit
   integer, intent(out) :: mcountd
   x      = 1.d0
   x      = x/3.d0
   mcountd= 0; digit  = 3
   do while(digit.eq.3)
    digit  = nint(10*x-x)
    mcountd= mcountd+1
    x      = 10*x-3
   enddo
   mcountd  = mcountd-1
   end subroutine ctscountdigits
  end module countdigits
!
  subroutine ctsinit(limitvalue,scaloopin,ext_num_for_r1in)
   use combinatorics
   use scale
   use countdigits
   use avh_olo
   use avh_olo_units
   include 'cts_mprec.h'
   implicit none 
   integer, intent(in) :: scaloopin
   include 'cts_dpr.h' 
    , intent(in) :: limitvalue
   integer :: idig,ncountd
   include 'cts_dpr.h' 
    :: thrsin
   include 'cts_mpr.h' 
    :: one
   logical, intent(in) :: ext_num_for_r1in
   include 'cts_mpinit.h'
   limit= limitvalue ! limit of precision below which the mp routines activate
   call ctscountdigits(ncountd)
   write (*,*) ' '
   write (*,'(a72)') '------------------------------------------------------------------------'
   write (*,'(a72)') '|              You are using CutTools - Version 1.9.3                  |'  
   write (*,'(a72)') '|              Authors: G. Ossola, C. Papadopoulos, R. Pittau          |' 
   write (*,'(a72)') '|              Published in JHEP 0803:042,2008                         |'
   write (*,'(a72)') '|              http://www.ugr.es/~pittau/CutTools                      |'
   write (*,'(a72)') '|                                                                      |'
   if (ncountd.gt.50) then
    write(*,'(a72)') '|              Internal mproutines detected                            |'
   endif
   write(*,'(a28,i4.2,a40)')'|              Compiler with',ncountd,'  significant digits detetected        |'
   write (*,'(a72)') '---------------------------------------------------------------------- '
   write (*,*) '  '
!
!  Too large values of ncountd are not supported by avh
!
   ncountd= min(34,ncountd)
!
   call load_combinatorics
!
!  Allocate all the needed variables
!
   call allocating
!
!  Initilaization of the scalar 1-loop libraries:
!
!  scaloop= 1 -> looptools 1-loop scalar functions (not implemented yet)
!  scaloop= 2 -> avh       1-loop scalar functions (massive with complex masses)
!  scaloop= 3 -> qcdloop   1-loop scalar functions (Ellis and Zanderighi)
!
   scaloop= scaloopin 
   ext_num_for_r1= ext_num_for_r1in
   call set_unit('all',-1)
   if    (scaloop.eq.2) then
!                               
!    avh initialization:
!
     thrsin= 1.d-6 
     call olo_precision(ncountd)
     call olo_onshell(thrsin) 
   elseif(scaloop.eq.3) then
!                               
!    qcdloop initialization:
!
     call qlinit 
!
!    also OneLOop is used for rank 1 and 2 2-point functions:
!
     thrsin= 1.d-6 
     call olo_precision(ncountd) 
     call olo_onshell(thrsin) 
   else
    stop 'value of scaloop not allowed'
   endif
   contains
!
   subroutine allocating
    use dimensions
    use denominators
    use coefficients
    use loopfunctions
    use mp_loopfunctions
    call load_dimensions
    call dp_allocate_den
    call dp_allocate_arrays(dmns)
    call mp_allocate_den
    call mp_allocate_arrays(dmns)
    call allocate_loopfun(dmns)
    call allocate_mp_loopfun(dmns)
   end subroutine allocating
  end subroutine ctsinit
!
  subroutine ctsstatistics(discarded)
  use scale
  implicit none 
  logical, intent(out) :: discarded
  write(*,*) 'n_tot =',n_tot  ! total n. of points
  write(*,*) 'n_mp  =',n_mp   ! n. of points evaluated in mult. prec.
  write(*,*) 'n_unst=',n_unst ! n. of unstable points
  if (n_unst.ne.0) then
   discarded=.true.
  else
   discarded=.false.
  endif
  end subroutine ctsstatistics



 
