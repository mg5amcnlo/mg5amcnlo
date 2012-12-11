
      subroutine parsecsv_2004
C
C   This routine reads a line from the new, complete, 
C   mass and width table supplied by PDG.
C   This routine works with the 2004 table:
C   FORMAT (BN, 2(E11.0, 1X, 2(E9.0, 1X)), A3, 1X, A1, 1X, A4, 1X, 3(A1, 1X), A7, 1X, A4, 1X, 2(A1, 1X), A18, 1X, A)
C
C
      implicit none
C
#include "stdlun.inc"
#include "stdcsv.inc"
C
      integer i,frac
      real top,bottom
C
c set everything to zero, just in case
      mval = 0.
      wval = 0.
      mpos = 0.
      mneg = 0.
      wpos = 0.
      wneg = 0.
      chg = 0.
      pdgid = 0
      name = ' '

c make sure this is not an old style file or a comment
      if (cline(1:1).eq.'M' .or. cline(1:1).eq.'W' 
     &    .or. cline(1:1).eq.'*') then
	   return
      endif

c      write(lnhout,202) cline
      read(cline(1:11), fmt = '( BN,E11.0)') mval
      read(cline(13:21), fmt = '( BN,E9.0)') mpos
      read(cline(23:31), fmt = '( BN,E9.0)') mneg
c      write(lnhout,101) mval, mpos, mneg
      read(cline(33:43), fmt = '( BN,E11.0)') wval
      read(cline(45:53), fmt = '( BN,E9.0)') wpos
      read(cline(55:63), fmt = '( BN,E9.0)') wneg
c      write(lnhout,102) wval, wpos, wneg
      ispin = cline(65:67)
      gpar = cline(69:69)
      jspin = cline(71:74)
      ppar = cline(76:76)
      cpar = cline(78:78)
      antip = cline(80:80)
      read(cline(82:88), fmt = '( BN,I7)') pdgid
      cchg = cline(90:93)
      rank = cline(95:95)
      status = cline(97:97)
      name = cline(99:115)
      quarklist = cline(117:132)

C translate charge
      frac=0 
      do i=1,4
        if(cchg(i:i).eq.'/') frac=i
      enddo
      if( frac.eq.0) then
          read(cchg,fmt='(BN,E4.0)') chg
      else
	  read(cchg(1:frac-1),fmt='(BN,E4.0)') top
	  read(cchg(frac+1:4),fmt='(BN,E4.0)') bottom
	  if(bottom.lt.1)then
	      write(lnhout,501) cchg, top, bottom
	  else
	      chg = top/bottom
	  endif
      endif

      return

 101  format(' mass: ',F15.8,1X,G11.5,1X,G11.5)
 102  format('width: ',F15.8,1X,G11.5,1X,G11.5)
 202  format('reading : ',A130)
 501  format('PARSECSV ERROR: failed to properly parse ',A4,
     &       ' got ',F5.1,' over ',F5.1)
      end
