	program t3
c
c	Example program for mcfio. Simple case. Read the file produced by t1
c
	implicit none
#include "mcfio.inc"
	integer ostr_all, ostr_sel, istr_old, nopen, istr(10), length
	integer inum, iblki(5)
	integer numblocks, blkids(3)
	integer i, irun, ievt, istore, itrig
	integer NUMTRIES
	PARAMETER (NUMTRIES = 50)
	integer xdr_stdhep_multi
	external xdr_stdhep_multi
	
c
c	Initialization phase.
c
	call mcfio_Init
	istr_old = mcfio_OpenReadDirect('test1.dat')
	
	irun = 0
	istore = 0
	ievt = 0
	itrig = 255
 	if (
     &    mcfio_SpecificEvent(istr_old, ievt, istore, irun, itrig)
     &      .eq. -1) then 
          print *, ' Magic event not found, stop '
          stop
        end if
        call mcfio_InfoEventInt(istr_old, MCFIO_EVENTNUMBER, inum)
        print *, ' Event number for magic trigger ', inum
        if (mcfio_block(istr_old, MCFIO_STDHEPM, xdr_stdhep_multi)
     &        .eq. -1) then 
             print *, ' Error reading STDHEP block '
             STOP
          end if
        call study_stdhep
        itrig = 0 
        istore = 3
 	if (
     &    mcfio_NextSpecificEvent(istr_old, ievt, istore, irun, itrig)
     &      .eq. -1) then 
          print *, ' First event of magic spill not found, stop '
          stop
        end if
        call mcfio_InfoEventInt(istr_old, MCFIO_EVENTNUMBER, inum)
        print *, ' Event number for 1st evt from magic spill', inum
c
	call mcfio_Close(istr_old)
	stop
	end
	subroutine study_stdhep
	implicit none
	integer i, io, nn
#include "stdhep.inc"
	
	print  *, ' Number of tracks ', nhep
	nn = nhep
	if (nn .gt. 3 ) nn = 3
	do i = 1, nn
	 print *, ' Pz, Py, Pz, ', (phep(io,i), io = 1,3)
	 print *, '   Mass ', phep(5,i)
	 print *, ' Y position, Z position ', vhep(2,i), vhep(3,i)
	end do
	return
	end 	
