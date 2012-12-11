
      character*(*) function stdexpname(IT,LMX)

C...Purpose: get the padded particle name
C
C      IT = index to HEPEVT common block
C      LMX = lenght of character variable to be filled
C

#include "stdhep.inc"

      integer LMX,IP,IT

      character CHAP*20,CHAC*20,CHDL(4)*4
      data CHDL/' ','()','!!','<>'/
      save CHDL

C...Get particle name, pad it and check it is not too long.
      CHAP = CHDL(1)
      IP = IDHEP(IT)
      if(IP.NE.0) call HEPNAM(IP,CHAP)
      LEN=0
      do 100 LEM=1,18
 100    if(CHAP(LEM:LEM).NE.' ') LEN=LEM
      if(ISTHEP(IT).EQ.1) then
        CHAC=CHAP
        if(LEN.GT.LMX) CHAC(LMX:LMX)='?'
      else
        ICHR = 3
        if(ISTHEP(IT).EQ.2) ICHR = 2
C... This is for isajet partons
        if(ISTHEP(IT).EQ.21 .OR. ISTHEP(IT).EQ.22) ICHR = 4
        if(LEN.EQ.0) then
          CHAC=CHDL(ICHR)(1:2)//' '
        else
          CHAC=CHDL(ICHR)(1:1)//CHAP(1:MIN(LEN,LMX-2))//
     &      CHDL(ICHR)(2:2)//' '
          if(LEN+2.GT.LMX) CHAC(LMX:LMX)='?'
        endif
      endif
      stdexpname = CHAC
      return
      end
