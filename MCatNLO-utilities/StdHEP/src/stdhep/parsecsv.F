
      subroutine parsecsv
C
C   This routine reads a line from the new, complete, 
C   mass and width table supplied by PDG.
C   This routine works with the 2006 table:
C   FORMAT (BN, E14.0, 1X, 2(E7.0, 1X), E12.0, 1X, 2(E7.0, 1X), A3, 1X, A1, 1X, A4, 1X, 3(A1, 1X), A7, 1X, A4, 1X, 2(A1, 1X), A17, 1X, A)
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
      chg = 0
      pdgid = 0
      name = ' '

c make sure this is not an old style file or a comment
      if (cline(1:1).eq.'M' .or. cline(1:1).eq.'W' 
     &    .or. cline(1:1).eq.'*') then
	   return
      endif

c      write(lnhout,202) cline
      read(cline(1:14), fmt = '( BN,E14.0)') mval
      read(cline(16:22), fmt = '( BN,E7.0)') mpos
      read(cline(24:30), fmt = '( BN,E7.0)') mneg
c      write(lnhout,101) mval, mpos, mneg
      read(cline(32:43), fmt = '( BN,E12.0)') wval
      read(cline(45:51), fmt = '( BN,E7.0)') wpos
      read(cline(53:59), fmt = '( BN,E7.0)') wneg
c      write(lnhout,102) wval, wpos, wneg
      ispin = cline(61:63)
      gpar = cline(65:65)
      jspin = cline(67:70)
      ppar = cline(72:72)
      cpar = cline(74:74)
      antip = cline(76:76)
c      write(lnhout,103) ispin, gpar, jspin, ppar, antip
      read(cline(78:84), fmt = '( BN,I7)') pdgid
c      write(lnhout,104) pdgid
      cchg = cline(86:89)
      rank = cline(91:91)
      status = cline(93:93)
      name = cline(95:111)
      quarklist = cline(113:128)
c      write(lnhout,105) cchg, rank, status, name, quarklist

C translate charge
      frac=0 
      do i=1,4
        if(cchg(i:i).eq.'/') frac=i
      enddo
      if( frac.eq.0) then
          if(cchg.eq.'  ++')then
	     chg = 6
	  elseif(cchg.eq.'  --')then
	     chg = -6
	  elseif(cchg.eq.'   +')then
	     chg = 3
	  elseif(cchg.eq.'   -')then
	     chg = -3
	  elseif(cchg.eq.'   0')then
	     chg = 0
	  else
 	   write(lnhout,502) cchg
	  endif
      else
	  read(cchg(1:frac-1),fmt='(BN,E4.0)') top
	  read(cchg(frac+1:4),fmt='(BN,E4.0)') bottom
	  if(bottom.ne.3)then
	      write(lnhout,501) cchg, top, bottom
	  else
	      chg = top
	  endif
      endif

      return

c 101  format(' mass: ',F15.8,1X,G11.5,1X,G11.5)
c 102  format('width: ',F15.8,1X,G11.5,1X,G11.5)
c 103  format('spin/parity: ',5A5)
c 104  format('PDG ID: ',I10)
c 105  format('name, etc: ',3A5,A20,A30)
 202  format('reading : ',A130)
 501  format('PARSECSV ERROR: failed to properly parse ',A4,
     &       ' got ',F5.1,' over ',F5.1)
 502  format('PARSECSV ERROR: failed to properly parse ',A4)
      end
