	program t2
c
c	Example program for mcfio. Simple case. Read the file produced by t1
c
	implicit none
#include "mcfio.inc"
	integer ostr_all, ostr_sel, istr_old, nopen, istr(10), length
	integer inum, iblki(5)
	character *60 filename
	character *80 title
	character *255 answer
	integer numblocks, blkids(3)
	integer i, numevts_real
	integer NUMTRIES
	PARAMETER (NUMTRIES = 5000000)
	integer MCFIO_CRAPPER 
	parameter (MCFIO_CRAPPER = 4562468)
	integer xdr_stdhep_multi
	external xdr_stdhep_multi
	
c
c	Initialization phase.
c
	call mcfio_Init
c	call mcfio_PrintDictionary()
	
c	istr_old = mcfio_OpenReadDirect('/home/simulation/pltmp/chi_test.dat')
	istr_old = mcfio_OpenReadDirect(
     &      	           'test1.dat')
	nopen =  mcfio_InfoNumStream(istr, 10)
	
	print *, ' Number of stream opened ', nopen
	print *, ' Stream index ' , istr(1)
	
	call mcfio_InfoStreamChar(
     &               istr(1), MCFIO_CREATIONDATE, answer, length)
	print *, ' Creation Date ', answer(1:length)
	
	call mcfio_InfoStreamChar(
     &               istr(1), MCFIO_TITLE, answer, length)
	print *, ' TITLE ', answer(1:length)
	
	print *, ' Expect an error message now.. '
	call mcfio_InfoStreamChar(
     &               istr(1), MCFIO_CRAPPER, answer, length)
	print *, ' No answer expected. ', answer(1:length)
	
	call mcfio_InfoStreamChar(
     &               istr(1), MCFIO_COMMENT, answer, length)
	print *, ' COMMENT ', answer(1:length)
	
	call mcfio_InfoStreamInt(istr(1), MCFIO_NUMEVTS, inum)
	print *,' Number of events ', inum
	numevts_real = inum
	
	call mcfio_InfoStreamInt(istr(1), MCFIO_NUMBLOCKS, inum)
	print *,' Maximum number of Blocks per event ', inum
	
	call mcfio_InfoStreamInt(istr(1), MCFIO_BLOCKIDS, iblki)
	print *,'Block id for the last block ', iblki(inum)
	
c
c	=============================================================
c
	call mcfio_InfoStreamInt(istr(1), MCFIO_NUMWORDS, inum)
	print *,' Number of words read ', inum
        do i = 1, numevts_real + 10
          if (mcfio_NextEvent(istr_old) .ne. MCFIO_RUNNING) then 
             print *,'  Error reading next event '
             call mcfio_InfoStreamInt(istr_old, MCFIO_STATUS, inum)
             if (inum .eq. MCFIO_EOF ) then
                 print *, ' Reached EOF, Close and Stop '
	         call mcfio_Close(istr_old)
	       stop
             else 
               print *, ' Unrecognized Status, stat = ',  inum
             end if 
             stop
          else 
            print *, ' O.K., running... '
          end if
          if (i .eq. 1) then 
            call mcfio_InfoEventInt(istr_old, MCFIO_RUNNUMBER, inum)
            print *, ' Run number ', inum
          end if
          call mcfio_InfoEventInt(istr_old, MCFIO_EVENTNUMBER, inum)
          print *, ' Event number ', inum
          call mcfio_InfoEventInt(istr_old, MCFIO_STORENUMBER, inum)
          print *, ' Store number ', inum
          call mcfio_InfoEventInt(istr_old, MCFIO_TRIGGERMASK, inum)
          if (inum .eq. 255) print *, ' Special lucky event'
            
          if (mcfio_block(istr_old, MCFIO_STDHEPM, xdr_stdhep_multi)
     &        .eq. -1) then 
             print *, ' Error reading STDHEP block '
             STOP
          end if
c          if (i .lt. 30) call study_stdhep
        end do
c
c	=============================================================
c
	call mcfio_InfoStreamInt(istr(1), MCFIO_NUMWORDS, inum)
	print *,' Number of words read ', inum
	
	call mcfio_InfoStreamInt(istr(1), MCFIO_EFFICIENCY, inum)
	print *,' Efficiency ', inum
	call mcfio_Close(istr_old)
	stop
	end
	subroutine study_stdhep
	implicit none
	integer i, io, nn
#include "stdhep.inc"
	
	print  *, ' Number of tracks ', nhep
	nn = nhep
	if (nn .gt. 2) nn = 2
	do i = 1,nn
	 print *, ' Pz, Py, Pz, ', (phep(io,i), io = 1,3)
	 print *, '   Mass ', phep(5,i)
	 print *, ' Y position, Z position ', vhep(2,i), vhep(3,i)
	end do
	
	print *, ' NMULTI = ' , NMULTI
	return
	end 	
