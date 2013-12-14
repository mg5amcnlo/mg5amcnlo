	program tN2
c
c	Example program for mcfio. Simple case. Read the file produced by t1
c
	implicit none
#include "mcfio.inc"
	integer ostr_all, ostr_sel, istr_old, nopen, length
	integer inum, iblki(5)
	character *255 filename
	character *80 title
	character *255 answer
	integer MAX_NUMNTUPLE
	parameter (MAX_NUMNTUPLE = 10)
	integer numblocks, numNTuples, blkids(MAX_NUMNTUPLE)
	integer i, j, numevts_real, ntu22, ntu23, nn,ll
	integer xdr_stdhep_multi
	external xdr_stdhep_multi
#include "glob22.inc"
#include "glo23.inc"
	
c
c	Initialization phase.
c
	call mcfio_Init
	call mcfio_SetForSaveDecoding(.False.)
c	call mcfio_PrintDictionary()
	
c	istr_old = mcfio_OpenReadDirect('testN1_l.dat')
	istr_old = mcfio_OpenReadMapped('testN1_l.dat')
c	filename = '/home/focus/scr1/pltmp/mcfio_mcfio/testN1_l.dat'
c	istr_old = mcfio_OpenReadDirect(filename)
	call mcfio_InfoStreamChar(
     &               istr_old, MCFIO_CREATIONDATE, answer, length)
	print *, ' Creation Date ', answer(1:length)
	
	call mcfio_InfoStreamInt(istr_old, MCFIO_NUMEVTS, inum)
	print *,' Number of events ', inum
	numevts_real = inum
c
c	Look for the Ntuple content from the file header.
c
	call mcfio_InfoStreamInt(istr_old, MCFIO_NUMNTUPLES, numNTuples)
	print *, ' Number of Ntuples for this stream ',  numNTuples
c
c	Inquire about identifier list for the stream.
c
	if (numNTuples .gt. MAX_NUMNTUPLE) then 
	   print *, ' Too many NTuple on this stream, gave up'
	   stop
	end if
	nn = mcfio_GetNTupleIds(istr_old, blkids, MAX_NUMNTUPLE)
	if (nn .gt. MAX_NUMNTUPLE) then 
	   print *, ' Too many NTuple on this stream, gave up'
	   stop
	end if
c
c	We are interested only in the structure with the uid 22
c
        ntu22 = -1
        ntu23 = -1
	do i = 1, numNTuples
	    print *, ' Ntuple Sequential number ',i 
	    ll =  mcfio_GetNtupleCategory(blkids(i), answer)
	    print *, '        Category  ', answer(1:ll)
	    ll =  mcfio_GetNtupleTitle(blkids(i), answer)
	    print *, '        Title  ', answer(1:ll)
	    ll =  mcfio_GetNtupleName(blkids(i), answer)
	    print *, '        Template Name ', answer(1:ll)
	    if (mcfio_GetNTupleUID(blkids(i)) .eq. 22) ntu22 = blkids(i)
	    if (mcfio_GetNTupleUID(blkids(i)) .eq. 23) ntu23 = blkids(i)
	end do
c
	If (ntu22 .eq. -1) then 
	    print *, ' Can not find Ntuple number 22 '
	    stop
	end if 	     		
        do i = 1, numevts_real
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
          end if 
          if (mcfio_NTupleMult(istr_old, ntu22, glob_c.version)
     &           .eq. -1) then 
            print *, 'Problem decoding Multiplicty from Ntuple glob22'
            stop
          end if
          if (glob_c.nglob .gt. 5) then
           if (mcfio_NTupleSubStruct(istr_old, ntu22,
     &                             5, glob_c.version)
     &           .eq. -1) then 
              print *, 'Problem decoding SubVariable from Ntuple glob22'
              stop
             end if
                if (ifix(glob_c.glob_var(5).v7_now2).ne.(i*10 +5))
     &          print *, ' v7_now2 is wrong '
             
             
c              if (mcfio_NTuple(istr_old, ntu22, glob_c.version)
c     &           .eq. -1) then 
c            print *, 'Problem decoding Multiplicty from Ntuple glob22'
c              stop
c              end if
c             call check_glob22(i)
c              if (mcfio_NTuple(istr_old, ntu23, version_glo23)
c     &           .eq. -1) then 
c            print *, 'Problem decoding Multiplicty from Ntuple glob23'
c              stop
c              end if
          end if          
        end do
        call mcfio_close(istr_old)
        stop 
        end 
        subroutine check_glob22(ievt)
        implicit none
        integer i,ievt
#include "glob22.inc"
	
        if (ievt .eq. 1) print *, ' Version ', glob_c.version
        print *, ' --- Dump for evt ', ievt       
        print *, ' Nglob ', glob_c.nglob
        print *, ' MissingEt, Something ',
     &       glob_c.Missinget, glob_c.something
        do i = 1,glob_c.nglob
           if (ifix(glob_c.glob_var(i).v7_now2) .ne. (ievt*10 +i))
     &           print *,'v7_now is wrong'
           if (glob_c.glob_var(i).v7_now_last .ne. i*1.d0) 
     &         print *,'v7_now_last is wrong'
        end do
        print *, ' ------ ' 
        return
        end
        
