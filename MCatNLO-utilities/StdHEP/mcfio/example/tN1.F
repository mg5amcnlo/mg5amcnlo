	program tN1
c
c	Example program for mcfio. Simple case for Ntuple
c
	implicit none
#include "mcfio.inc"
	integer ostr_all, ostr_sel, istr_old
	character *60 filename
	character *80 title
	integer numblocks, blkids(3)
	integer i, ivalue
	integer NUMTRIES
	PARAMETER (NUMTRIES = 50)
#include "glob22.inc"
#include "glo23.inc"
	
c
c	Initialization phase.
c
	call mcfio_Init
c	filename = '/home/focus/scr1/pltmp/mcfio_mcfio/testN1_l.dat'
        filename = 'testN1_l.dat'
	title = ' First Simple test '
	numblocks = 3
	ostr_all = mcfio_OpenWriteDirect(filename, title, %val(0), 
     &             NUMTRIES, blkids, 0)
        if (ostr_all .eq. -1) then 
           print *, ' Can t open output file, give up '
           stop
        end if
        blkids(1) = mcfio_DeclareNtuple(22, 'Global Super_22', 'Top',
     &                                 ostr_all, 'glob22.db') 
     
        blkids(2) = mcfio_DeclareNtuple(23, 'Global Super_23', 'Top',
     &                                 ostr_all, 'glo23.db') 
     
        blkids(3) = mcfio_DeclareNtuple(1, 
     &                                 'Global Super_Charm', 'Charming',
     &                                 ostr_all, 'glob22.db')
        if (mcfio_EndDeclNTuples(ostr_all) .ne. 3) then 
            print *, 'Problem with stream initialisation '
            stop
        end if
c
c	Set the version token.  We must do this..
c
	glob_c.version = '1.10'
	version_glo23 = '1.00'        
        ivalue = 456
        call mcfio_SetEventInfo(ostr_all, MCFIO_RUNNUMBER, ivalue) 
        do i = 1, NUMTRIES
          call fill_glob22(i)
          if (mcfio_NTuple(ostr_all,blkids(1), glob_c.version)
     &           .eq. -1) then 
            print *, 'Problem with Encoding glob22 '
            stop
          end if
          call fill_glo23(i)
          if (mcfio_NTuple(ostr_all,blkids(2), version_glo23)
     &           .eq. -1) then 
            print *, 'Problem with Encoding glo23 '
            stop
          end if
          if (mcfio_NextEvent(ostr_all) .eq. -1) then 
             print *,'  Error writing next event, Stop '
             stop
          end if
          print *,' Written evt number ',i
        end do
        call mcfio_close(ostr_all)
        stop
        end 
        subroutine fill_glob22(ievt)
        implicit none
c
c	Dummy fill for now.. 
c
#include "glob22.inc"
	integer ievt, i,j, k
	
	glob_c.eventnumber(1) = mod(ievt,64)
c	No_way = ievt+99
	glob_c.Missinget = 578.9 - ievt*1.
	glob_c.something = 123.5 +ievt*7.
	glob_c.nglob = mod(ievt, 10)
	if (ievt .eq. 1) glob_c.nglob = 12
	if (glob_c.nglob .eq. 0) return
	do i=1, glob_c.nglob
	   glob_c.glob_var(i).missingEtphi = 'Not too much '
	   if (i .gt. 4)
     &      glob_c.glob_var(i).missingEtphi = 'Quite a bit'
	   do j = 1,80
	      glob_c.glob_var(i).v6_new(j) = j + mod(ievt,2)
	   end do
	   do j= 1,6
	      do k = 1,5
	          glob_c.glob_var(i).v5_new(k,j) = 0.0
	      end do
	   end do
	  glob_c.glob_var(i).v7_now2 = ievt*10 + i
	  glob_c.glob_var(i).missingetnomuon = ievt*100. + i*10.
	  glob_c.glob_var(i).v7_now = ievt/10. + i
	  glob_c.glob_var(i).v7_now_last = i*1.d0
	end do
	return
	end 
	
        subroutine fill_glo23(ievt)
        implicit none
c
c	Dummy fill for now.. 
c
#include "glo23.inc"
c
	integer ievt, i,j,k,iseed
	save iseed
	data iseed /5683873/
#if AIX
	real rand
#else
	real ran
	external ran
#endif
	
#if AIX
	ngl23 = ifix(ngl23_max*rand()/10.)
#else
	ngl23 = ifix(ngl23_max*ran(iseed)/10.)
#endif
	eventnumber = ievt
#if AIX
	triggerbit1(1) = .false.
	triggerbit1(2) = .false.
#else
	triggerbit1(1) = 0
	triggerbit1(2) = 0
#endif
	runnumber = 456
	No_way = 0
	missingEt = 789.
	something = 0.
	do i = 1, ngl23
	  do j= 1,5
	   do k = 1,8
	    Amtrack(j,k,i) = 1.0 *(j + k + i)
	   end do
	  end do
	   
	  do j= 1,10
	   do k = 1,10
	    v5_new(j,k,i) = 0.
	   end do
	  end do
	  v7_now2(i) = 0.
	  missingetnomuon(i) = 0.
	end do
	return 
	end 
	
	   
	       
	 
	    
	  
	
