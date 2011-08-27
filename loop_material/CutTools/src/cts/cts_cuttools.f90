  subroutine ctsinit(rootsvalue,limitvalue,idigvalue,scaloopin,muscalein,thrsin)
   use combinatorics
   use scale
   include 'cts_mprec.h'
   implicit none 
   include 'cts_dpr.h'
    , intent(in) :: rootsvalue,limitvalue
   integer, intent(in) :: idigvalue,scaloopin
   integer :: idig
   include 'cts_dpr.h' 
    :: muscalein,thrsin
   write (*,*) ' '
   write (*,*) '----------------------------------------------------- '
   write (*,*) '|  You are using CutTools - Version 1.6.5           | '      
   write (*,*) '|  Authors: G. Ossola, C. Papadopoulos, R. Pittau   | ' 
   write (*,*) '|  Published in JHEP 0803:042,2008                  | '
   write (*,*) '|  http://www.ugr.es/~pittau/CutTools               | '
   write (*,*) '----------------------------------------------------- '
   write (*,*) '  '
   call load_combinatorics
!
!  Allocate all the needed variables
!
   call allocating
   scaloop= scaloopin
!
!  scaloop= 1 -> looptools 1-loop scalar functions (not implemented yet)
!  scaloop= 2 -> avh       1-loop scalar functions (massive with complex masses)
!  scaloop= 3 -> qcdloop   1-loop scalar functions (Ellis and Zanderighi)
!
   if (scaloop.eq.2) then
!                               
!  avh initializations:
!
    muscale= muscalein
    call avh_olo_mu_set(muscale) 
    call avh_olo_onshell(thrsin) 
   elseif (scaloop.eq.3) then
!
!  qcdloop initializations:
!
    muscale= muscalein
    call avh_olo_mu_set(muscale)
    call avh_olo_onshell(thrsin)
    musq= muscalein**2
    call qlinit 
   else
    stop 'value of scaloop not allowed'
   endif
!
   roots= rootsvalue
   limit= limitvalue  ! limit of precision below which the mp routines activate
   if (idigvalue.eq.0) then
     idig  = 64
     mpflag=.false. 
   else
     idig= idigvalue
     mpflag=.true.
   endif 
   call mpinit(idig)          ! set the max    n. of digits for the mp routines
   call mpsetprec (idig)      ! set the actual n. of digits for the mp routines
   call mpsetoutputprec(idig) ! set the max n.of digits in the mp output
   contains
!
   subroutine allocating
    use coefficients
    use loopfunctions
    call load_dimensions
    call dp_allocate_den
    call dp_allocate_arrays(dmns)
    call mp_allocate_den
    call mp_allocate_arrays(dmns)
    call allocate_loopfun(dmns)
   end subroutine allocating
  end subroutine ctsinit
!
  subroutine ctsstatistics(nvalue_mp,nvalue_disc)
  use scale
  implicit none 
  integer, intent(out) :: nvalue_mp, nvalue_disc
  nvalue_mp  = n_mp
  nvalue_disc= n_disc
  end subroutine ctsstatistics

