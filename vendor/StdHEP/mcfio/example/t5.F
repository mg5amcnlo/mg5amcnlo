	program t5
c
c	Example program for mcfio. Simple case. Read the file produced by t4
c
	implicit none
#include "mcfio.inc"
	integer ostr_all, ostr_sel, istr_old, nopen, istr(10), length
	integer inum, iblki(5)
	integer numblocks, blkids(3)
	integer i, irun, ievt, istore, itrig
	integer xdr_mcfast_track_offlslow
	external xdr_mcfast_track_offlslow
        integer xdr_mcfast_track_offline 
        external xdr_mcfast_track_offline
	
c
c	Initialization phase.
c
	call mcfio_Init
	istr_old = mcfio_OpenReadDirect('test4.dat')
	
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
c        if (mcfio_block(istr_old, 
c     &         MCFIO_OFFTRACKSTRUCT, xdr_mcfast_track_offlslow)
c     &        .eq. -1) then 
c             print *, ' Error reading OFFTRACKSTRUCT block '
c             STOP
c          end if
        if (mcfio_block(istr_old, 
     &         MCFIO_OFFTRACKARRAYS, xdr_mcfast_track_offline)
     &        .eq. -1) then 
             print *, ' Error reading OFFTRACKARRAYS block '
             STOP
        end if
        call study_offtrk
c
	call mcfio_Close(istr_old)
	stop
	end
	subroutine study_offtrk
	implicit none
	integer i, io, nn
#include "offline_track.inc"
	
	print  *, ' Number of tracks ', trk_off_num
	nn = trk_off_num
	if (nn .gt. 3 ) nn = 3
	do i = 1, nn
	 print *, ' Px ', trk_off(i).w.px
	 print *, ' Py ', trk_off(i).w.py
	 print *, ' Pz ', trk_off(i).w.pz
	 print *, ' Pt ', trk_off(i).w.pt
	 print *, ' P ', trk_off(i).w.p
	 print *, ' E ', trk_off(i).w.E
	 print *, '   Mass ', trk_off(i).mass
	 print *, ' Y position, Z position ',
     &                 trk_off(i).w.y, trk_off(i).w.z
         print *, ' Number of Silicon hits ', trk_off(i).nhit_sil 
	end do
	return
	end 	
