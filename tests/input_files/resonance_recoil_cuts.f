      logical function dummy_cuts(p,istatus,ipdg)
c Validation cuts for u b > e+ ve b d at NLO QCD (massless bottom).
c Require separate resolved jets with positive net bottom and down
c flavour. For this restricted Born process, the crossed real channels
c also contain beam-collinear limits with other underlying Born flavours:
c g b > d b e+ ve ubar with beam-collinear d, and
c u g > d b e+ ve bbar with beam-collinear b. Requiring just two jets and
c any bottom tag does not remove these unsubtracted singularities.
c A clustered b bbar pair has zero net flavour, excluding the collinear
c photon-splitting contribution as well.
c Use only with the explicitly flavour-labelled single-top process; its
c generated subprocesses do not merge bottom and light-quark flavours.
      implicit none
      include 'nexternal.inc'
      include 'cuts.inc'
      double precision p(0:4,nexternal),pq(0:3,nexternal),
     $     pj(0:3,nexternal)
      integer istatus(nexternal),ipdg(nexternal),jet(nexternal),
     $     flavour(nexternal,2),jet_flavour(nexternal,2),nq,njet,i,j
      dummy_cuts=.false.
      nq=0
      do i=nincoming+1,nexternal
         if(istatus(i).ne.1.or.p(0,i).le.0d0)cycle
         if(abs(ipdg(i)).gt.5.and.ipdg(i).ne.21)cycle
         nq=nq+1
         pq(:,nq)=p(0:3,i)
         flavour(nq,:)=0
         if(abs(ipdg(i)).eq.5)flavour(nq,1)=sign(1,ipdg(i))
         if(abs(ipdg(i)).eq.1)flavour(nq,2)=sign(1,ipdg(i))
      enddo
      if(nq.lt.2)return
      call amcatnlo_fastjetppgenkt_etamax_timed(pq,nq,jetradius,
     $     ptj,etaj,jetalgo,pj,njet,jet)
      if(njet.lt.2)return
      jet_flavour=0
      do i=1,nq
         if(jet(i).gt.0.and.jet(i).le.njet)
     $        jet_flavour(jet(i),:)=jet_flavour(jet(i),:)+flavour(i,:)
      enddo
      do i=1,njet
         if(jet_flavour(i,1).le.0)cycle
         do j=1,njet
            if(i.eq.j)cycle
            if(jet_flavour(j,2).gt.0)dummy_cuts=.true.
         enddo
      enddo
      end
