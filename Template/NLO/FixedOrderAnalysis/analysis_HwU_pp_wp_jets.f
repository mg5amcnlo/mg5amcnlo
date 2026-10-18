c Simple fixed-order analysis for p p > e+ ve + jets [QCD].
c Select with FO_ANALYSIS_FORMAT = HwU and
c FO_ANALYSE = analysis_HwU_pp_wp_jets.o in FO_analyse_card.dat.
c
c W truth momentum is p[e+] + p[ve]; no neutrino reconstruction.
c Jets: anti-kt, R=0.4, pT > 20 GeV, |eta| < 4.5, ordered by pT.
c Book four jet ranks and four DJR transitions for every process.
c DJR n->n+1: sqrt[exclusive_dmerge[n]] from kt with R=1,
c using all final-state light QCD partons, without a pT cut.
c Each spectrum has linear bins in GeV and bins in log10[x/GeV].
c Linear range: 0--1000 GeV; log10 range: -3--3.
c Additional W and jet pT spectra: 1 GeV bins from 0 to 100 GeV.
c NLO includes all contributions; LO includes only ibody=3.
c No extra event selection is applied beyond the generation cuts.
c
      subroutine analysis_begin(nwgt,weights_info)
      implicit none
      include 'nexternal.inc'
      integer nwgt,nmax,nplots,i,j,l,k
      parameter (nmax=4,nplots=4+5*nmax)
      character*(*) weights_info(*)
      character*6 cc(2)
      character*6 rank,previous
      character*12 transition
      data cc/'|T@NLO','|T@LO '/

      call HwU_inithist(nwgt,weights_info)
      do i=1,2
         l=(i-1)*nplots
         call HwU_book(l+1,'total rate '//cc(i),1,0d0,1d0)
         call HwU_book(l+2,'pT W+ [GeV] '//cc(i),100,0d0,1000d0)
         call HwU_book(l+3,'log10 pT W+/GeV '//cc(i),60,-3d0,3d0)
         call HwU_book(l+4+4*nmax,'pT W+ [GeV] 1 GeV bins '//cc(i),
     &        100,0d0,100d0)
         do j=1,nmax
            write(rank,'(i6)') j
            rank=adjustl(rank)
            write(previous,'(i6)') j-1
            previous=adjustl(previous)
            transition=trim(previous)//'-'//trim(rank)
            k=l+3+2*(j-1)
            call HwU_book(k+1,'pT j'//trim(rank)//' [GeV] '//cc(i),
     &           100,0d0,1000d0)
            call HwU_book(k+2,'log10 pT j'//trim(rank)//'/GeV '//cc(i),
     &           60,-3d0,3d0)
            call HwU_book(l+4+4*nmax+j,'pT j'//trim(rank)//
     &           ' [GeV] 1 GeV bins '//cc(i),100,0d0,100d0)
            k=l+3+2*nmax+2*(j-1)
            call HwU_book(k+1,'DJR '//trim(transition)//' [GeV] '//
     &           cc(i),100,0d0,1000d0)
            call HwU_book(k+2,'log10 DJR '//trim(transition)//'/GeV '//
     &           cc(i),60,-3d0,3d0)
         enddo
      enddo
      end


      subroutine analysis_end(dummy)
      implicit none
      double precision dummy
      call HwU_write_file
      end


      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
      implicit none
      include 'nexternal.inc'
      integer nmax,nplots
      parameter (nmax=4,nplots=4+5*nmax)
      integer istatus(nexternal),ipdg(nexternal),ibody
      double precision p(0:4,nexternal),wgts(*)
      double precision pqcd(0:3,nexternal),pjet(0:3,nexternal)
      double precision pw(0:3),ptw,ptjet(nmax),djr(nmax)
      double precision rjet,ptmin,etamax,rdjr,amcatnlo_fastjetdmerge
      parameter (rjet=0.4d0,ptmin=20d0,etamax=4.5d0,rdjr=1d0)
      external amcatnlo_fastjetdmerge
      integer nqcd,njet,jet(nexternal),ilep,inu,i,j,k,l

      if (wgts(1).eq.0d0) return
      ilep=0
      inu=0
      nqcd=0
      do j=nincoming+1,nexternal
c Skip the zero momentum used for an absent n-body parton.
         if (istatus(j).ne.1.or.p(0,j).le.0d0) cycle
         if (ipdg(j).eq.-11) ilep=j
         if (ipdg(j).eq.12) inu=j
         if ((abs(ipdg(j)).ge.1.and.abs(ipdg(j)).le.5).or.
     &        ipdg(j).eq.21) then
            nqcd=nqcd+1
            pqcd(:,nqcd)=p(0:3,j)
         endif
      enddo
      if (ilep.eq.0.or.inu.eq.0) then
         write(*,*) 'W+ jets analysis requires final-state e+ and ve'
         stop 1
      endif
      pw=p(0:3,ilep)+p(0:3,inu)
      ptw=sqrt(pw(1)**2+pw(2)**2)

      njet=0
      djr=0d0
      if (nqcd.gt.0) then
         call amcatnlo_fastjetppgenkt_etamax(pqcd,nqcd,rjet,ptmin,
     &        etamax,-1d0,pjet,njet,jet)
         njet=min(njet,nmax)
         do j=1,njet
            ptjet(j)=sqrt(pjet(1,j)**2+pjet(2,j)**2)
         enddo
c Recluster with kt; query this history before another FastJet call.
         call amcatnlo_fastjetppgenkt(pqcd,nqcd,rdjr,0d0,1d0,
     &        pjet,k,jet)
         do j=1,min(nqcd,nmax)
            djr(j)=sqrt(max(0d0,amcatnlo_fastjetdmerge(j-1)))
         enddo
      endif

      do i=1,2
         if (i.eq.2.and.ibody.ne.3) cycle
         l=(i-1)*nplots
         call HwU_fill(l+1,0.5d0,wgts)
         call HwU_fill(l+2,ptw,wgts)
         if (ptw.gt.0d0) call HwU_fill(l+3,log10(ptw),wgts)
         call HwU_fill(l+4+4*nmax,ptw,wgts)
         do j=1,njet
            k=l+3+2*(j-1)
            call HwU_fill(k+1,ptjet(j),wgts)
            call HwU_fill(k+2,log10(ptjet(j)),wgts)
            call HwU_fill(l+4+4*nmax+j,ptjet(j),wgts)
         enddo
         do j=1,nmax
            k=l+3+2*nmax+2*(j-1)
c Missing transitions belong at zero in the linear spectrum, so its
c first bin includes the unresolved real/counterevent cancellation.
c Zero cannot enter a logarithmic spectrum.
            call HwU_fill(k+1,djr(j),wgts)
            if (djr(j).gt.0d0)
     &           call HwU_fill(k+2,log10(djr(j)),wgts)
         enddo
      enddo
      end
