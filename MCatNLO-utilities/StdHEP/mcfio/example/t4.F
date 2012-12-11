	program t4
c
c	Example program for mcfio.  Case with two different blocks.
c		Writing a file.
c
	implicit none
#include "mcfio.inc"
	integer ostr_all, ostr_sel, istr_old, inum
	character *60 filename
	character *80 title
	integer numblocks, blkids(3)
	integer i, ivalue
	integer NUMTRIES
	PARAMETER (NUMTRIES = 226)
	integer xdr_stdhep_multi
	external xdr_stdhep_multi
        integer xdr_mcfast_track_offlslow
        external xdr_mcfast_track_offlslow
       integer xdr_mcfast_track_offline 
       external xdr_mcfast_track_offline
	
	
c
c	Initialization phase.
c
	call mcfio_Init
	filename = 'test4.dat'
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
          if (mcfio_block(ostr_all, 
     &         MCFIO_OFFTRACKARRAYS, xdr_mcfast_track_offline)
     &        .eq. -1) then 
             print *, ' Error writing OFFTRACKARRAYS, block '
             STOP
          end if
c          if (mcfio_block(ostr_all, 
c     &         MCFIO_OFFTRACKSTRUCT, xdr_mcfast_track_offlslow)
c     &        .eq. -1) then 
c             print *, ' Error writing OFFTRACKSTRUCT, block '
c             STOP
c          end if
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
        call mcfio_InfoStreamInt(ostr_all, MCFIO_EFFICIENCY, inum)
        print *,' Efficiency ', inum
        call mcfio_close(ostr_all)
        stop
        end 
        subroutine fill_stdhep(inum)
c
c	Moronic routine to generate pseudo data. And copy it to the 
c	Offline structure. 
c
        implicit none 
#include "stdhep.inc"
#include "offline_track.inc"
	integer inum
#if AIX
	real rand
#else
	real ran
	external ran
#endif
	integer i, j, k, nn, iseed
	save iseed
	data iseed/66678756/
	double precision vec(4), masses(5)
	save masses
	data masses/0.1345, 0.6783, 1.245, 1.45, 5.678/
	
	
#if AIX
	nn = mod(inum, 100) * 5 * ifix(5. * rand())
#else
	nn = mod(inum, 100) * 5 * ifix(5. * ran(iseed))
#endif
	if (nn .gt. trk_off_max) nn = trk_off_max -1
	nevhep = inum
	nhep = nn
	trk_off_num = nn
	do i = 1, nn
#if AIX
	  phep(5,i) = masses(ifix(1. + 3.*rand()))
#else
	  phep(5,i) = masses(ifix(1. + 3.*ran(iseed)))
#endif
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
#if AIX
          vhep(1,i) = rand()
#else
          vhep(1,i) = ran(iseed)
#endif
          vhep(2,i) = vhep(1,i)
          vhep(3,i) = 0.
          isthep(i) = i
          jmohep(1,i) = 1
          jmohep(2,i) = 3
          jdahep(1,i) = 0
          jdahep(2,i) = 4
          trk_off(i).w.px = phep(1,i)
          trk_off(i).w.py = phep(2,i)
          trk_off(i).w.pz = phep(3,i)
          trk_off(i).w.E = phep(4,i)
          trk_off(i).mass = phep(5,i)
          trk_off(i).w.x = vhep(1,i)
          trk_off(i).w.y = vhep(2,i)
          trk_off(i).w.z = vhep(3,i)
          trk_off(i).w.pt =
     &        dsqrt(trk_off(i).w.px**2 + trk_off(i).w.py**2)
          trk_off(i).w.p = dsqrt(vec(4))
          trk_off(i).w.q = 1.0 
          do j = 1, 7
           do k = 1,7
            trk_off(i).cov(j,k) = 0.01*j + 0.1*k
           end do
          end do
          trk_off(i).chisq = 2.0
          trk_off(i).nhit = 50
          trk_off(i).nhit_sil = 10
          trk_off(i).nhit_stereo = 25
          trk_off(i).icov = 1
          trk_off(i).hep = i
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
          
	    
	  
	
