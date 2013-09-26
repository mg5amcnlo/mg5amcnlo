      subroutine HEPLST(MLST)

C...Purpose: to list an event from the standard common block.
C
C... MLST = 1   -- print all parameters except vertex
C... MLST = 2   -- print all parameters, including vertex

#include "stdhep.inc"
#include "stdlun.inc"

      character CHAP*20,CHAC*20,CHDL(4)*4
      data CHDL/' ','()','!!','<>'/
      save CHDL

C...Write header
      LMX=18
      if(MLST.EQ.1)then
        write(lnhout,1100) NEVHEP
      elseif(MLST.EQ.2)then
        write(lnhout,1300) NEVHEP
      else
        write(lnhout,1500) NEVHEP,MLST
        MLST = 1
      endif

      do 120 I=1,NHEP

C...Get particle name, pad it and check it is not too long.
        CHAP = CHDL(1)
        if(IDHEP(I).NE.0) call HEPNAM(IDHEP(I),CHAP)
        LEN=0
        do 100 LEM=1,18
  100     if(CHAP(LEM:LEM).NE.' ') LEN=LEM
        if(ISTHEP(I).EQ.1) then
          CHAC=CHAP
          if(LEN.GT.LMX) CHAC(LMX:LMX)='?'
        else
          ICHR = 3
          if(ISTHEP(I).EQ.2) ICHR = 2
C... This is for isajet partons
          if(ISTHEP(I).EQ.21 .OR. ISTHEP(I).EQ.22) ICHR = 4
          if(LEN.EQ.0) then
            CHAC=CHDL(ICHR)(1:2)//' '
          else
            CHAC=CHDL(ICHR)(1:1)//CHAP(1:MIN(LEN,LMX-2))//
     &      CHDL(ICHR)(2:2)//' '
            if(LEN+2.GT.LMX) CHAC(LMX:LMX)='?'
          endif
        endif

C...Write data for particle/jet.
        if(MLST.EQ.1)then
          write(lnhout,1200) I,CHAC,ISTHEP(I),IDHEP(I),
     1       (JMOHEP(J1,I),J1=1,2),(JDAHEP(J2,I),J2=1,2),
     2       (PHEP(J3,I),J3=1,5)
        elseif(MLST.EQ.2)then
          write(lnhout,1400) I,CHAC,ISTHEP(I),IDHEP(I),
     1       (JMOHEP(J1,I),J1=1,2),(JDAHEP(J2,I),J2=1,2),
     2       (PHEP(J3,I),J3=1,5),(VHEP(J,I),J=1,4)
        endif

  120 CONTINUE

      return
C...Format statements for output on unit lnhout:
 1100 format(///18X,'Event listing (HEP format)',12X,'Event: ',I8
     1   //4X,'I',2X,'particle/jet',5X,'ISTHEP',6X,'IDHEP',5X,'JMOHEP',
     2   6X,'JDAHEP',4X,'PHEP(1,I)',3X,'PHEP(2,I)',3X,'PHEP(3,I)',
     3   3X,'PHEP(4,I)',3X,'PHEP(5,I)')
 1200 format(1X,I4,2X,A18,2X,I3,1X,I10,4(2X,I4),5F12.5)
 1300 format(///18X,'Event listing (HEP format with vertices)',
     1   12X,'Event: ',I8
     1   //4X,'I',2X,'particle/jet',5X,'ISTHEP',6X,'IDHEP',5X,'JMOHEP',
     2   6X,'JDAHEP',4X,'PHEP(1,I)',3X,'PHEP(2,I)',3X,'PHEP(3,I)',
     3   3X,'PHEP(4,I)',3X,'PHEP(5,I)'
     4   /65X,'VHEP(1,I)',3X,'VHEP(2,I)',3X,'VHEP(3,I)',3X,'VHEP(4,I)'/)
 1400 format(1X,I4,2X,A18,2X,I3,1X,I10,4(2X,I4),5F12.5/58X,4(F12.3))
 1500 format(///18X,'Event listing (HEP format)',12X,'Event: ',I8
     *   /20X,' Output type changed from unsupported',I5,' to 1'
     1   //4X,'I',2X,'particle/jet',5X,'ISTHEP',4X,'IDHEP',5X,'JMOHEP',
     2   6X,'JDAHEP',4X,'PHEP(1,I)',3X,'PHEP(2,I)',3X,'PHEP(3,I)',
     3   3X,'PHEP(4,I)',3X,'PHEP(5,I)')

      end
