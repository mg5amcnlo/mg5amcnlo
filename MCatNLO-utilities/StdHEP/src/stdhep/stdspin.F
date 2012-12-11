
      subroutine STDSPIN(IT,JSP)

C...Get the J-spin of this particle
C
C      IT = index to HEPEVT common block
C      For particle ID, +/- IJKLM
C            KQJ = M = 2*Jspin + 1
C      JSP = Jspin
C
      IMPLICIT NONE

      integer IT,KQ,KQA,KQJ
      real JSP

#include "stdhep.inc"

      KQ=IDHEP(IT)
      KQA=IABS(KQ)
      KQJ=MOD(KQA,10)
      JSP = (FLOAT(KQJ) - 1.)/2.
      return
      end
