      subroutine stdmsjet
C
C    compare particle mass for QQ and jetset/pythia
C
#include "stdlun.inc"
#include "qqpars.inc"
#include "qqprop.inc"

      integer ID1,IT1,ITT1
      integer qqtran
      character*16 CNAM1,CNAM2
      character*20 HNAM1

C...special cases
      write(lnhout,1001)
      do 100 I=NPMNQQ,NPMXQQ
        ID1=I
        CNAM1 = PNAME(ID1)
        IT1=qqtran(ID1,1)
        ITT1=qqtran(IT1,2)
        call HEPNAM(IT1,HNAM1)
        CNAM2 = ' '
        if(ITT1.GT.-20) CNAM2 = PNAME(ITT1)
        write(lnhout,1101) ID1,CNAM1,IT1,HNAM1,ITT1,CNAM2
 100  CONTINUE
      return
 1001 format(//10X,'Particles'
     1       /5X,'IQQ  QNAME',14X,'ISTD  HPNAM',15X,'IQQ  QNAME')
 1101 format(1X,I7,2X,A16,1X,I7,2X,A16,1X,I7,2X,A16)
      end
