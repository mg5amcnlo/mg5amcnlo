  module countdigits 
   include 'cts_mprec.h'
   implicit none
   private 
   public :: ctscountdigits,radix,digits
   private :: mp_radix,mp_digits
!
   interface radix
    procedure mp_radix
   end interface
!
   interface digits
    procedure mp_digits
   end interface
   contains
!
   subroutine ctscountdigits(mcountd)
   implicit none
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
!
   function mp_radix(arg)
   type(mp_real), intent(in) :: arg
   integer :: mp_radix
   mp_radix= 2
   end function
!
   function mp_digits(arg)
   type(mp_real), intent(in) :: arg
   integer :: mp_digits
   mp_digits= 64 
   end function
  end module countdigits
!
  subroutine ctsinit(limitvalue,scaloopin)
   use combinatorics
   use scale
   use countdigits
   include 'cts_mprec.h'
   implicit none 
   integer, intent(in) :: scaloopin
   include 'cts_dpr.h' 
    :: limitvalue
   integer :: idig,ncountd
   include 'cts_dpr.h' 
    :: thrsin
   include 'cts_mpr.h' 
    :: one
   limit= limitvalue ! limit of precision below which the mp routines activate
   idig = 64
   call mpinit(idig)          ! set the max    n. of digits for the mp routines
   call mpsetprec (idig)      ! set the actual n. of digits for the mp routines
   call mpsetoutputprec(idig) ! set the max n.of digits in the mp output
   call ctscountdigits(ncountd)
   write (*,*) ' '
   write (*,'(a72)') '------------------------------------------------------------------------'
   write (*,'(a72)') '|              You are using CutTools - Version 1.7.0                  |'  
   write (*,'(a72)') '|              Authors: G. Ossola, C. Papadopoulos, R. Pittau          |' 
   write (*,'(a72)') '|              Published in JHEP 0803:042,2008                         |'
   write (*,'(a72)') '|              http://www.ugr.es/~pittau/CutTools                      |'
   write (*,'(a72)') '|                                                                      |'
   if (ncountd.gt.50) then
    ncountd = 32
    write(*,'(a72)') '|              Internal mproutines detected in CutTools                |'
    write(*,'(a72)') '------------------------------------------------------------------------'
    write(*,*) '  '
   else
    one= 1.d0
    ncountd = int(digits(one)*log(radix(one)*one)/log(one*10))
    write(*,'(a19,i4.2,a49)')'|     Compiler with',ncountd,'  significant digits detetected in CutTools.    |'
    write (*,'(a72)') '---------------------------------------------------------------------- '
    write (*,*) '  '
   endif
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
   if    (scaloop.eq.2) then
!                               
!    avh initialization:
!
     thrsin= 1.d-6 
     call avh_olo_onshell(thrsin) 
     call avh_olo_mp_onshell(thrsin) 
   elseif(scaloop.eq.3) then
!                               
!    qcdloop initialization:
!
     call qlinit 
!
!    also OneLOop is used for rank 1 and 2 2-point functions:
!
     thrsin= 1.d-6 
     call avh_olo_onshell(thrsin) 
   else
    stop 'value of scaloop not allowed'
   endif
   contains
!
   subroutine allocating
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
  logical, intent(out) :: discarded
  write(*,*) 'n_tot =',n_tot  ! total n. of points
  write(*,*) 'n_mp  =',n_mp   ! n. of points evaluated in mult. prec.
  write(*,*) 'n_disc=',n_disc ! n. of discarded points
  if (n_disc.ne.0) then
   discarded=.true.
  else
   discarded=.false.
  endif
  end subroutine ctsstatistics



 
