      subroutine amcatnlo_fastjetppgenkt_timed(pQCD,NN,rfj,sycut,palg,
     &pjet,njet,jet)

      include 'nexternal.inc'

c     arguments
      double precision pQCD(0:3,nexternal),PJET(0:3,nexternal)
      double precision rfj,sycut,palg
      integer NN,JET(nexternal),njet

c     timing statistics
      double precision tbefore, tAfter
      double precision tTot, tOLP, tFastJet, tPDF
      common/timings/tTot, tOLP, tFastJet, tPDF

      call cpu_time(tBefore)
      call amcatnlo_fastjetppgenkt(pQCD,NN,rfj,sycut,palg,
     &pjet,njet,jet)
      call cpu_time(tAfter)
      tFastJet = tFastJet + (tAfter-tBefore)
      end
