c Parton-shower analysis for p p > e+ ve + jets, without hadronisation.
c Shower card settings:
c   hadronize = F
c   ANALYSE = HwU.o py8an_HwU_pp_wp_jets.o myfastjetfortran.o
c   EXTRALIBS = fastjet stdc++
c MG5 selects full FastJet or its bundled fjcore automatically.
c
c Same observables and bins as analysis_HwU_pp_wp_jets.f, with |T@PS.
c W truth: last W+ copy, or final e+ + ve if no W+ is in the record.
c Jets: final quarks 1--5 and gluons, anti-kt R=0.4, pT > 20 GeV,
c |eta| < 4.5, ordered by pT. Keep four jet ranks.
c DJRs: sqrt[exclusive_dmerge[n]], n=0--3, kt R=1, all these partons
c without pT or eta cuts. Missing transitions have scale zero.
c Linear spectra: 0--1000 GeV; log10[x/GeV]: -3--3.
c Additional W and jet pT spectra: 1 GeV bins from 0 to 100 GeV.
c IMPORTANT: HEPEVT and this shower FastJet wrapper both use
c [px,py,pz,E], unlike the energy-first fixed-order wrapper.
c
      subroutine RCLOS()
      end


      subroutine PYABEG(nnn,wwwi)
      use HwU_wgts_info_len
      include 'HEPMC.INC'
      include 'reweight0.inc'
      integer nnn,nmax,j,k,i,nwgt_analysis,nev_analysis
      parameter (nmax=4)
      common /c_wp_jets_analysis/ nwgt_analysis,nev_analysis
      character*(wgts_info_len) wwwi(*)
      character*(wgts_info_len) weights_info(max_weight_shower)
      character*5 cc
      parameter (cc='|T@PS')
      character*6 rank,previous
      character*12 transition
      logical pineappl
      common /for_pineappl/ pineappl

      if (nnn.lt.0.or.nnn.gt.max_weight_shower) then
         write(*,*) 'Invalid weight count in W+ jets PYABEG',nnn
         stop 1
      endif
      nwgt_analysis=max(1,nnn)
      nev_analysis=0
      weights_info(1)='central value'
      do i=1,nnn
         weights_info(i)=wwwi(i)
      enddo
      pineappl=.false.
      call HwU_inithist(nwgt_analysis,weights_info)
      call set_error_estimation(0)
      call HwU_book(1,'total rate '//cc,1,0d0,1d0)
      call HwU_book(2,'pT W+ [GeV] '//cc,100,0d0,1000d0)
      call HwU_book(3,'log10 pT W+/GeV '//cc,60,-3d0,3d0)
      call HwU_book(4+4*nmax,'pT W+ [GeV] 1 GeV bins '//cc,
     &     100,0d0,100d0)
      do j=1,nmax
         write(rank,'(i6)') j
         rank=adjustl(rank)
         write(previous,'(i6)') j-1
         previous=adjustl(previous)
         transition=trim(previous)//'-'//trim(rank)
         k=3+2*(j-1)
         call HwU_book(k+1,'pT j'//trim(rank)//' [GeV] '//cc,
     &        100,0d0,1000d0)
         call HwU_book(k+2,'log10 pT j'//trim(rank)//'/GeV '//cc,
     &        60,-3d0,3d0)
         call HwU_book(4+4*nmax+j,'pT j'//trim(rank)//
     &        ' [GeV] 1 GeV bins '//cc,100,0d0,100d0)
         k=3+2*nmax+2*(j-1)
         call HwU_book(k+1,'DJR '//trim(transition)//' [GeV] '//cc,
     &        100,0d0,1000d0)
         call HwU_book(k+2,'log10 DJR '//trim(transition)//'/GeV '//
     &        cc,60,-3d0,3d0)
      enddo
      end


      subroutine PYAEND(ievttot)
      implicit none
      double precision ievttot
      integer nwgt_analysis,nev_analysis
      common /c_wp_jets_analysis/ nwgt_analysis,nev_analysis

      if (nev_analysis.eq.0) return
      call finalize_histograms(nev_analysis)
      open(unit=99,file='MADatNLO.HwU',status='replace')
      call HwU_output(99,ievttot)
      close(99)
      end


      subroutine PYANAL(nnn,xww)
      include 'HEPMC.INC'
      include 'reweight0.inc'
      integer nnn,nmax,nwgt_analysis,nev_analysis
      parameter (nmax=4)
      common /c_wp_jets_analysis/ nwgt_analysis,nev_analysis
      double precision xww(*),www(max_weight_shower)
      double precision pqcd(4,nmxhep),pjet(4,nmxhep),pw(4)
      double precision ptw,ptjet(nmax),djr(nmax),fastjetdmerge
      double precision rjet,ptmin,etamax,rdjr
      parameter (rjet=0.4d0,ptmin=20d0,etamax=4.5d0,rdjr=1d0)
      external fastjetdmerge
      integer i,j,k,iw,ilep,inu,nqcd,njet,jet(nmxhep)
      logical lastcopy

      nev_analysis=nev_analysis+1
      if (nnn.eq.0) then
         www(1)=evwgt
      else
         if (xww(1).eq.0d0) return
         do i=1,nwgt_analysis
            www(i)=evwgt*(xww(i)/xww(1))
         enddo
      endif
      if (www(1).eq.0d0) return

      iw=0
      ilep=0
      inu=0
      nqcd=0
      do i=1,nhep
c Prefer the W after shower recoil, before its decay. Ignore W copies
c that still have another W+ among their daughters.
         if (idhep(i).eq.24) then
            lastcopy=.true.
            do j=max(1,jdahep(1,i)),min(nhep,jdahep(2,i))
               if (idhep(j).eq.24) lastcopy=.false.
            enddo
            if (lastcopy) iw=i
         endif
         if (isthep(i).ne.1.or.phep(4,i).le.0d0) cycle
         if (idhep(i).eq.-11) ilep=i
         if (idhep(i).eq.12) inu=i
         if ((abs(idhep(i)).ge.1.and.abs(idhep(i)).le.5).or.
     &        idhep(i).eq.21) then
            nqcd=nqcd+1
c Copy px,py,pz,E directly; do not use the fixed-order ordering.
            pqcd(:,nqcd)=phep(1:4,i)
         endif
      enddo
      if (iw.ne.0) then
         pw=phep(1:4,iw)
      elseif (ilep.ne.0.and.inu.ne.0) then
         pw=phep(1:4,ilep)+phep(1:4,inu)
      else
         write(*,*) 'W+ jets PYANAL requires W+ or final e+ and ve'
         stop 1
      endif
      ptw=sqrt(pw(1)**2+pw(2)**2)

      njet=0
      djr=0d0
      if (nqcd.gt.0) then
         call fastjetppgenkt_etamax(pqcd,nqcd,rjet,ptmin,etamax,
     &        -1d0,pjet,njet,jet)
         njet=min(njet,nmax)
         do j=1,njet
            ptjet(j)=sqrt(pjet(1,j)**2+pjet(2,j)**2)
         enddo
c Query the kt history immediately, before another FastJet call.
         call fastjetppgenkt(pqcd,nqcd,rdjr,0d0,1d0,pjet,k,jet)
         do j=1,min(nqcd,nmax)
            djr(j)=sqrt(max(0d0,fastjetdmerge(j-1)))
         enddo
      endif

      call HwU_fill(1,0.5d0,www)
      call HwU_fill(2,ptw,www)
      if (ptw.gt.0d0) call HwU_fill(3,log10(ptw),www)
      call HwU_fill(4+4*nmax,ptw,www)
      do j=1,njet
         k=3+2*(j-1)
         call HwU_fill(k+1,ptjet(j),www)
         call HwU_fill(k+2,log10(ptjet(j)),www)
         call HwU_fill(4+4*nmax+j,ptjet(j),www)
      enddo
      do j=1,nmax
         k=3+2*nmax+2*(j-1)
         call HwU_fill(k+1,djr(j),www)
         if (djr(j).gt.0d0) call HwU_fill(k+2,log10(djr(j)),www)
      enddo
c Shower events are independent; accumulate once per analysed event.
      call HwU_add_points
      end
