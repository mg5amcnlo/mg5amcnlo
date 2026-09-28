      logical function dummy_cuts(p,istatus,ipdg)
c Validation cuts for u b > e+ ve b d at NLO QCD (massless bottom).
c Require at least two resolved jets, including one with nonzero net
c bottom flavour. Inclusive jet multiplicity alone admits a beam-collinear
c bottom spectator with two hard light jets in the crossed real channels.
c A clustered b bbar pair has zero net flavour. This also excludes the
c collinear photon-splitting contribution from this validation observable.
c Use only with the explicitly flavour-labelled single-top process; its
c generated subprocesses do not merge bottom and light-quark flavours.
      implicit none
      include 'nexternal.inc'
      include 'cuts.inc'
      double precision p(0:4,nexternal),pq(0:3,nexternal),
     $     pj(0:3,nexternal)
      integer istatus(nexternal),ipdg(nexternal),jet(nexternal),
     $     flavour(nexternal),jet_flavour(nexternal),nq,njet,i
      dummy_cuts=.false.
      nq=0
      do i=nincoming+1,nexternal
         if(istatus(i).ne.1.or.p(0,i).le.0d0)cycle
         if(abs(ipdg(i)).gt.5.and.ipdg(i).ne.21)cycle
         nq=nq+1
         pq(:,nq)=p(0:3,i)
         flavour(nq)=0
         if(abs(ipdg(i)).eq.5)flavour(nq)=sign(1,ipdg(i))
      enddo
      if(nq.lt.2)return
      call amcatnlo_fastjetppgenkt_etamax_timed(pq,nq,jetradius,
     $     ptj,etaj,jetalgo,pj,njet,jet)
      if(njet.lt.2)return
      jet_flavour=0
      do i=1,nq
         if(jet(i).gt.0.and.jet(i).le.njet)
     $        jet_flavour(jet(i))=jet_flavour(jet(i))+flavour(i)
      enddo
      dummy_cuts=any(jet_flavour(1:njet).ne.0)
      end
