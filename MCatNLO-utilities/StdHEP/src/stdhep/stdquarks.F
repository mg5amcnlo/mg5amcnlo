      subroutine stdquarks(it,kq1,kq2,kq3,kql,kqj,kqr,kqx)

C...Get constituent quarks of this particle
C
C      IT = index to HEPEVT common block
C      For particle ID, +/- HIJKLMN
C            KQX = H = flag tentative and special assignments
C            KQR = I = radial excitations
C            KQL = J = spin and orbital angular momentum
C            KQ3 = K = quark
C            KQ2 = L = quark
C            KQ1 = M = quark
C            KQJ = N = 2*Jspin + 1

      implicit none

      integer it,kq,kqa,kqx,kq3,kq2,kq1,kqj,kql,kqr,irt

#include "stdhep.inc"

      kq=idhep(it)
      kqa=iabs(kq)
      kqx=mod(kqa/1000000,10)
      kqr=mod(kqa/100000,10)
      kql=mod(kqa/10000,10)
      kq3=mod(kqa/1000,10)
      kq2=mod(kqa/100,10)
      kq1=mod(kqa/10,10)
      kqj=mod(kqa,10)
      irt = mod(kqa,10000)
C... set quarks to zero if not meson, baryon, or diquark
      if(kqa.le.100 .or. kqa.ge.10000000) then
        kq1=0
        kq2=0
        kq3=0
      elseif(kqx.gt.0 .and. irt.le.100) then
        kq1=0
        kq2=0
        kq3=0
      endif
      return
      end
