   idig = 64
   call mpinit(idig)          ! set the max    n. of digits for the mp routines
   call mpsetprec (idig)      ! set the actual n. of digits for the mp routines
   call mpsetoutputprec(idig) ! set the max n.of digits in the mp output