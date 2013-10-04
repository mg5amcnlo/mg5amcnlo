      PROGRAM TST
C
C      test program
C      read the PDG mass and width list
C
#include "stdlun.inc"

C initialize HEP logical units
      lnhwrt=0
      lnhrd=0
      lnhdcy=30
      lnhout=22
      lnhpdf=0
      lnhdmp=0
      OPEN(UNIT=LNHOUT,FILE='readPDG.lpt',STATUS='NEW')
C
      call stdversn
C
      CALL PDGRDTB
C  print with stdhep numbers
      CALL PDGPRTB(1,LNHOUT)
C
      CLOSE(UNIT=LNHOUT)
      STOP
      END

