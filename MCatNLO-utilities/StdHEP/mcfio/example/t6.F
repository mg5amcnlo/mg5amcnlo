	program t6
c
c	Example program for mcfio.  Read a file, and split into 2 different 
c	files, depending on store number. Thus, 5 files are opened in 
c	this example. 
c
	implicit none
#include "mcfio.inc"
#include "stdhep.inc"
#include "offline_track.inc"
	integer ostr_stdhep(2), ostr_offl(2), istr_old, inum
	character *60 filename
	character *80 title
	integer numblocks, blkids(3), nevts
	integer i, ivalue, istore, irun, ievtn
	integer xdr_stdhep_multi
	external xdr_stdhep_multi
        integer xdr_mcfast_track_offline 
        external xdr_mcfast_track_offline
	
	
c
c	Initialization phase.
c
	call mcfio_Init
	
	filename = 'test4.dat'
	istr_old = mcfio_OpenReadDirect('test4.dat')
	
	numblocks = 1
	blkids(1) = MCFIO_STDHEPM
	filename = 'test6_stdhep_even.dat'
	title = ' Second Simple test, Stdhep run, even stores.  '
	ostr_stdhep(1) = mcfio_OpenWriteDirect(filename,
     &	 title, %val(0), 1000, blkids, numblocks)
	filename = 'test6_stdhep_odd.dat'
	title = ' Second Simple test, Stdhep run, odd stores.  '
	ostr_stdhep(2) = mcfio_OpenWriteDirect(filename, 
     &     title, %val(0),  1000, blkids, numblocks)
     
	blkids(1) = MCFIO_OFFTRACKARRAYS
	filename = 'test6_offl_even.dat'
	title = ' Second Simple test, Offline Tracks run, even stores.'
	ostr_offl(1) = mcfio_OpenWriteDirect(filename,
     &       title, %val(0),  1000, blkids, numblocks)
	filename = 'test6_offl_odd.dat'
	title = ' Second Simple test, Offline Tracks run, odd stores.'
	ostr_offl(2) = mcfio_OpenWriteDirect(filename, title,
     &                   ' Not so hot ! ', 
     &             1000, blkids, numblocks)

c
c	===============================================================
c
	call mcfio_InfoStreamInt(istr_old, MCFIO_NUMEVTS, nevts)
	print *,' Number of events on file test4.dat ', nevts
	
	do i = 1, nevts
          if (mcfio_NextEvent(istr_old) .eq. -1) then 
             print *,'  Error reading next event, Stop '
             stop
          end if
          call mcfio_InfoEventInt(istr_old, MCFIO_EVENTNUMBER, ievtn)
          call mcfio_InfoEventInt(istr_old, MCFIO_STORENUMBER, istore)
          call mcfio_InfoEventInt(istr_old, MCFIO_RUNNUMBER, irun)
          if (mcfio_block(istr_old, MCFIO_STDHEPM, xdr_stdhep_multi)
     &        .eq. -1) then 
             print *, ' Error reading STDHEP block '
             STOP
          end if
          if (mcfio_block(istr_old, MCFIO_OFFTRACKARRAYS,
     &         xdr_mcfast_track_offline).eq. -1) then 
             print *, ' Error reading OFFTRACKARRAYS  block '
             STOP
          end if
          
	  if (mod(istore,2) .eq. 0) then 
            if (mcfio_block(ostr_stdhep(1),
     &                 MCFIO_STDHEPM, xdr_stdhep_multi) .eq. -1) then 
             print *, ' Error writing STDHEP block '
             STOP
            end if
            if (mcfio_block(ostr_offl(1), MCFIO_OFFTRACKARRAYS,
     &         xdr_mcfast_track_offline).eq. -1) then 
             print *, ' Error writing OFFTRACKARRAYS  block '
             STOP
            end if
c
c	Transfer the Event Information.
c             
            call mcfio_SetEventInfo(ostr_stdhep(1),
     &                              MCFIO_EVENTNUMBER, ievtn)
            call mcfio_SetEventInfo(ostr_stdhep(1), 
     &                              MCFIO_STORENUMBER, istore)
            call mcfio_SetEventInfo(ostr_stdhep(1), 
     &                              MCFIO_RUNNUMBER, irun)
            if (mcfio_NextEvent(ostr_stdhep(1)) .eq. -1) then 
             print *,'  Error writing  next event, Stop, stream =  ',
     &         ostr_stdhep(1)         
             stop
            end if
            call mcfio_SetEventInfo(ostr_offl(1),
     &                              MCFIO_EVENTNUMBER, ievtn)
            call mcfio_SetEventInfo(ostr_offl(1), 
     &                              MCFIO_STORENUMBER, istore)
            call mcfio_SetEventInfo(ostr_offl(1), 
     &                              MCFIO_RUNNUMBER, irun)
            if (mcfio_NextEvent(ostr_offl(1)) .eq. -1) then 
             print *,'  Error writing  next event, Stop ',
     &         ostr_offl(1)         
             stop
            end if

	else 
            if (mcfio_block(ostr_stdhep(2),
     &                 MCFIO_STDHEPM, xdr_stdhep_multi) .eq. -1) then 
             print *, ' Error writing STDHEP block '
             STOP
            end if
            if (mcfio_block(ostr_offl(2), MCFIO_OFFTRACKARRAYS,
     &         xdr_mcfast_track_offline).eq. -1) then 
             print *, ' Error writing OFFTRACKARRAYS  block '
             STOP
            end if
            call mcfio_SetEventInfo(ostr_stdhep(2),
     &                              MCFIO_EVENTNUMBER, ievtn)
            call mcfio_SetEventInfo(ostr_stdhep(2), 
     &                              MCFIO_STORENUMBER, istore)
            call mcfio_SetEventInfo(ostr_stdhep(2), 
     &                              MCFIO_RUNNUMBER, irun)
            if (mcfio_NextEvent(ostr_stdhep(2)) .eq. -1) then 
             print *,'  Error writing  next event, Stop, stream =  ',
     &         ostr_stdhep(2)         
             stop
            end if
            call mcfio_SetEventInfo(ostr_offl(2),
     &                              MCFIO_EVENTNUMBER, ievtn)
            call mcfio_SetEventInfo(ostr_offl(2), 
     &                              MCFIO_STORENUMBER, istore)
            call mcfio_SetEventInfo(ostr_offl(2), 
     &                              MCFIO_RUNNUMBER, irun)
            if (mcfio_NextEvent(ostr_offl(2)) .eq. -1) then 
             print *,'  Error writing  next event, Stop ',
     &         ostr_offl(2)         
             stop
            end if
	  end if
	 end do
c
c	=================================================================
c
	call mcfio_close(istr_old)
	do i = 1,2
          call mcfio_InfoStreamInt(ostr_stdhep(i),
     &                MCFIO_NUMWORDS, inum)
          print *,
     &    ' NUMBER OF 4-BYTES WORDS WRITTEN on Stream',ostr_stdhep(i),
     &          ' is = ', inum
     
	  call mcfio_close(ostr_stdhep(i))
          call mcfio_InfoStreamInt(ostr_offl(i),
     &                MCFIO_NUMWORDS, inum)
          print *,
     &    ' NUMBER OF 4-BYTES WORDS WRITTEN on Stream',ostr_offl(i),
     &          ' is = ', inum
	  call mcfio_close(ostr_offl(i))
	end do
	stop
	end
	
	 
	 
	 
	  
	  
	
	
	  
	   
