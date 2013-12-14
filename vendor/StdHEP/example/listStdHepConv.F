      PROGRAM TST
C
C      test program
C      list various conversion options
C
#include "stdlun.inc"

C initialize HEP logical units
      lnhwrt=0
      lnhrd=0
      lnhdcy=30
      lnhout=22
      lnhpdf=0
      lnhdmp=0
      OPEN(UNIT=LNHOUT,FILE='listStdHepConv.lpt',STATUS='NEW')
C
      call stdversn
C
      call std3to4lst
c
      call cnv98to00lst
c
      call cnv2000to2004lst
c
      call cnv2004to2006lst
c
      CLOSE(UNIT=LNHOUT)
      STOP
      END

