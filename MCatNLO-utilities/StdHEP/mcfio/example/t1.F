	program t1
c
c	Example program for mcfio. Simple case.
c
	implicit none
#include "mcfio.inc"
#include "stdhep.inc"
	integer ostr_all, ostr_sel, istr_old
	character *60 filename
	character *80 title
	integer numblocks, blkids(3)
	integer i, ivalue
	integer NUMTRIES
	PARAMETER (NUMTRIES = 50)
	integer xdr_stdhep_multi
	external xdr_stdhep_multi
	
c
c	Initialization phase.
c
	call mcfio_Init
	filename = 'test1.dat'
	title = ' First Simple test '
	numblocks = 3
	blkids(1) = MCFIO_STDHEPM
	blkids(2) = MCFIO_OFFTRACKARRAYS
	blkids(3) = MCFIO_OFFTRACKSTRUCT
	ostr_all = mcfio_OpenWriteDirect(filename, title, %val(0), 
     &             NUMTRIES, blkids, numblocks)
        if (ostr_all .eq. -1) then 
           print *, ' Can t open output file, give up '
           stop
        end if
        
        ivalue = 456
        call mcfio_SetEventInfo(ostr_all, MCFIO_RUNNUMBER, ivalue) 
        do i = 1, NUMTRIES
          call fill_stdhep(i)
          if (mcfio_block(ostr_all, MCFIO_STDHEPM, xdr_stdhep_multi)
     &        .eq. -1) then 
             print *, ' Error writing STDHEP block '
             STOP
          end if
          ivalue = i/12
          call mcfio_SetEventInfo(ostr_all, MCFIO_STORENUMBER, ivalue) 
          if (i .ne. 13) then 
            ivalue = 0
          else
            ivalue =255
            call study_stdhep
          end if
          call mcfio_SetEventInfo(ostr_all, MCFIO_TRIGGERMASK, ivalue) 
          if (mcfio_NextEvent(ostr_all) .eq. -1) then 
             print *,'  Error writing next event, Stop '
             stop
          end if
        end do
        call mcfio_close(ostr_all)
        stop
        end 
        subroutine fill_stdhep(inum)
c
c	Moronic routine to generate pseudo data.
c
        implicit none 
#include "stdhep.inc"
	integer inum
	real rand
	integer i, j, nn, iseed
	save iseed
	data iseed/66678756/
	double precision vec(4), masses(5)
	save masses
	data masses/0.1345, 0.6783, 1.245, 1.45, 5.678/
	
	nmulti = 1
	jmulti(1) = 1
	nn = mod(inum, 100) * ifix(5. * rand())
	if (nn .gt. NMXHEP) nn = 10 
	nevhep = inum
	nhep = nn
	do i = 1, nn
	  phep(5,i) = masses(ifix(1. + 3.*rand()))
	  phep(1,i) = i*0.1
	  phep(2,i) = i*0.2
	  phep(3,i) = i*0.3
	  vec(4) = 0.
	  do j = 1,3
	    vec(j) = 1.d0*phep(j,i)
	    vec(4) = vec(4) + vec(j)**2
	  end do
	  phep(4,i) = sngl(dsqrt(vec(4) +
     &                    (1.d0*phep(5,i))*(1.d0*phep(5,i))))
          vhep(1,i) = rand()
          vhep(2,i) = vhep(1,i)
          vhep(3,i) = 0.
          isthep(i) = i
          jmohep(1,i) = 1
          jmohep(2,i) = 3
          jdahep(1,i) = 0
          jdahep(2,i) = 4
        end do
        return
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
          
	    
	  
	
