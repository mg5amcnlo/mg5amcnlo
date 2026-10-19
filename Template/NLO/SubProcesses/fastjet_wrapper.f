      subroutine amcatnlo_fastjetppgenkt_timed(pQCD,NN,rfj,sycut,palg,
     &pjet,njet,jet)

      include 'nexternal.inc'

c     arguments
      double precision pQCD(0:3,nexternal),PJET(0:3,nexternal)
      double precision rfj,sycut,palg
      integer NN,JET(nexternal),njet

c     timing statistics
      include "timing_variables.inc"

      if(timing_sampled)call timing_enter(timing_cuts)
      call amcatnlo_fastjetppgenkt(pQCD,NN,rfj,sycut,palg,
     &pjet,njet,jet)
      if(timing_sampled)call timing_leave(timing_cuts)
      return
      end


      subroutine amcatnlo_fastjetppgenkt_etamax_timed(
     & pQCD,NN,rfj,sycut,etamax,palg,pjet,njet,jet)

      include 'nexternal.inc'

c     arguments
      double precision pQCD(0:3,nexternal),PJET(0:3,nexternal)
      double precision rfj,sycut,palg,etamax
      integer NN,JET(nexternal),njet

c     timing statistics
      include "timing_variables.inc"

      if(timing_sampled)call timing_enter(timing_cuts)
      call amcatnlo_fastjetppgenkt_etamax(
     & pQCD,NN,rfj,sycut,etamax,palg,pjet,njet,jet)
      if(timing_sampled)call timing_leave(timing_cuts)
      return
      end
