      PROGRAM TST
C
C      test program
C      list standard particle definitions
C
#include "stdlun.inc"

C initialize HEP logical units
      lnhwrt=0
      lnhrd=0
      lnhdcy=30
      lnhout=22
      lnhpdf=0
      lnhdmp=0
      OPEN(UNIT=LNHOUT,FILE='listStdHep.lpt',STATUS='NEW')
C
      call stdversn
      CALL STDLST
C
      CLOSE(UNIT=LNHOUT)
      STOP
      END
