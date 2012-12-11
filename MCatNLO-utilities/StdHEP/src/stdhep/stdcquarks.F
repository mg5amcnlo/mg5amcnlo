
      subroutine stdcquarks(it,kq1,kq2,kq3,kql,kqj,kqr,kqx)

C...Get signed constituent quarks of this particle
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

      call stdquarks(it,kq1,kq2,kq3,kql,kqj,kqr,kqx)
      kq=idhep(it)
      kqa=iabs(kq)
      irt = mod(kqa,10000)

C...Simple cases: direct translation or special codes.
      if(kqa.le.100 .or. kqa.ge.10000000) then
      elseif(kqx.gt.0 .and. irt.le.100) then
      elseif(kqj.eq.0) then
C... KS, KL, pomeron and undefined
        if(kqa.eq.130 .or. kqa.eq.310)then
          kq1 = sign(kq1,kq)
          kq2 = -sign(kq2,kq)
	else
C... illegal
          kq1=0
          kq2=0
          kq3=0
        endif         

C...Construction from quark content for heavy meson, diquark, baryon.
      elseif(kq3.eq.0) then
        if(kq2.eq.3 .or. kq2.eq.5)then
C...Strange and beauty mesons.
          kq1 = sign(kq1,kq)
          kq2 = -sign(kq2,kq)
        else
C...All other mesons
          kq1 = -sign(kq1,kq)
          kq2 = sign(kq2,kq)
        endif
      elseif(kq1.eq.0) then
C...Diquarks.
        kq2 = sign(kq2,kq)
        kq3 = sign(kq3,kq)
      else
C...Baryons
        kq1 = sign(kq1,kq)
        kq2 = sign(kq2,kq)
        kq3 = sign(kq3,kq)
      endif

      return
      end
