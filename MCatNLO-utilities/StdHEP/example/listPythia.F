      PROGRAM TSTJET
C
C     list jetset particle definitions
C
#include "pydat1.inc"
#include "stdlun.inc"

      INTEGER NEVT,I,NPRNT,LOK
C
      NEVT=100
      NPRNT=1
C initialize HEP logical units
      lnhwrt=23
      lnhrd=0
      lnhdcy=0
      lnhout=22
      lnhpdf=0
      lnhdmp=0
      MSTU(11)=LNHOUT
      OPEN(UNIT=LNHOUT,FILE='listPythia.lpt',STATUS='NEW')
C
C... 
      write(lnhout,1001)
      call pylist(11)
      write(lnhout,1002)
      call pylist(12)
c     write(lnhout,1003)
c     call pylist(13)
C
C
c...close output file
      CLOSE(UNIT=LNHOUT)
      STOP
 1001 format(' ----------------------- pylist(11) -------------------')
 1002 format(' ----------------------- pylist(12) -------------------')
 1003 format(' ----------------------- pylist(13) -------------------')
      END

