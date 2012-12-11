	program t7
c
c	Example program for mcfio. Simple case. Read the file produced by t1
c
	implicit none
#include "mcfio.inc"
	integer istr_old,  length, nevts
	integer inum, iblki(5)
	character *60 filename
	character *80 title
	character *255 answer
	integer numblocks, blkids(3)
	integer i, ifile, ievt, istore, irun
	
c
c	Initialization phase.
c
	call mcfio_Init
	
	
	do ifile = 1,4
	 print *, ' =================== New File ================='
	 print *, ' '
	 if (ifile .eq. 1) then 
	   filename = 'test6_offl_even.dat'
c	   filename = 'test6_offl_odd.dat'
	 else if (ifile .eq. 2) then 
	   filename = 'test6_offl_odd.dat'
	 else if (ifile .eq. 3) then 
	   filename = 'test6_stdhep_even.dat'
	 else if (ifile .eq. 4) then 
	   filename = 'test6_stdhep_odd.dat'
	 end if 
	 print *, ' File  ', filename
	 istr_old = mcfio_OpenReadDirect(filename)
	 print *, ' Stream index is now ', istr_old
	call mcfio_InfoStreamChar(
     &               istr_old, MCFIO_TITLE, answer, length)
	print *, ' TITLE ', answer(1:length)
c
c	=============================================================
c
	call mcfio_InfoStreamInt(istr_old, MCFIO_NUMEVTS, nevts)
	print *, ' Number of events in this file ', nevts
        do i = 1, nevts
          if (mcfio_NextEvent(istr_old) .eq. -1) then 
             print *,'  Error reading next event, Stop '
             stop
          end if
          call mcfio_InfoEventInt(istr_old, MCFIO_RUNNUMBER, irun)
          call mcfio_InfoEventInt(istr_old, MCFIO_STORENUMBER, istore)
          call mcfio_InfoEventInt(istr_old, MCFIO_EVENTNUMBER, ievt)
          print *, 'ev info ', ievt, istore, irun
        end do
c
c	=============================================================
c
	print *, ' Closing Stream ',  istr_old
	call mcfio_Close(istr_old)
	end do
	stop
	end
