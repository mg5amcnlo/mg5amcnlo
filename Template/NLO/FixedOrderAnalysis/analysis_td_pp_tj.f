c
c Example analysis for "p p > t j [QCD]" process.
c Example analysis for "p p > t j $$ w+ w- [QCD]" process.
c Example analysis for "p p > w+ > t j  [QCD]" process.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(nwgt,weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      integer j,kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
      character*6 cc(2)
      data cc/'|T@NLO','|T@LO '/
      include 'dbook.inc'
      call inihist
      nwgt_analysis=nwgt
      if (nwgt_analysis*48.gt.nplots/4) then
         write (*,*) 'error in analysis_begin: '/
     &        /'too many histograms, increase NPLOTS to',
     &        nwgt_analysis*48*4
         stop 1
      endif
      do j=1,2
      do kk=1,nwgt_analysis
      l=(kk-1)*48+(j-1)*24
      call bookup(l+ 1,'t pt        '//cc(j)//weights_info(kk)
     &     ,5d0,0d0,200d0)
      call bookup(l+ 2,'t log pt    '//cc(j)//weights_info(kk)
     &     ,0.05d0,0d0,5d0)
      call bookup(l+ 3,'t y         '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
      call bookup(l+ 4,'t eta       '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
c
      call bookup(l+ 5,'j1 pt       '//cc(j)//weights_info(kk)
     &     ,5d0,0d0,200d0)
      call bookup(l+ 6,'j1 log pt   '//cc(j)//weights_info(kk)
     &     ,0.05d0,0d0,5d0)
      call bookup(l+ 7,'j1 y        '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
      call bookup(l+ 8,'j1 eta      '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
c
      call bookup(l+ 9,'j2 pt       '//cc(j)//weights_info(kk)
     &     ,5d0,0d0,200d0)
      call bookup(l+10,'j2 log pt   '//cc(j)//weights_info(kk)
     &     ,0.05d0,0d0,5d0)
      call bookup(l+11,'j2 y        '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
      call bookup(l+12,'j2 eta      '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
c
      call bookup(l+13,'bj1 pt      '//cc(j)//weights_info(kk)
     &     ,5d0,0d0,200d0)
      call bookup(l+14,'bj1 log pt  '//cc(j)//weights_info(kk)
     &     ,0.05d0,0d0,5d0)
      call bookup(l+15,'bj1 y       '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
      call bookup(l+16,'bj1 eta     '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
c
      call bookup(l+17,'bj2 pt      '//cc(j)//weights_info(kk)
     &     ,5d0,0d0,200d0)
      call bookup(l+18,'bj2 log pt  '//cc(j)//weights_info(kk)
     &     ,0.05d0,0d0,5d0)
      call bookup(l+19,'bj2 y       '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
      call bookup(l+20,'bj2 eta     '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
c
      call bookup(l+21,'syst pt     '//cc(j)//weights_info(kk)
     &     ,5d0,0d0,200d0)
      call bookup(l+22,'syst log pt '//cc(j)//weights_info(kk)
     &     ,0.05d0,0d0,5d0)
      call bookup(l+23,'syst y      '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
      call bookup(l+24,'syst eta    '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
c
      enddo
      enddo
      return
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_end(xnorm)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      character*14 ytit
      double precision xnorm
      integer i
      integer kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
      include 'dbook.inc'
      call open_topdrawer_file
      call mclear
      do i=1,NPLOTS
         call mopera(i,'+',i,i,xnorm,0.d0)
         call mfinal(i)
      enddo
      ytit='sigma per bin '
      do i=1,2
      do kk=1,nwgt_analysis
      l=(kk-1)*48+(i-1)*24
      call multitop(l+ 1,2,3,'t pt',' ','LOG')
      call multitop(l+ 2,2,3,'t log pt',' ','LOG')
      call multitop(l+ 3,2,3,'t y',' ','LOG')
      call multitop(l+ 4,2,3,'t eta',' ','LOG')
c
      call multitop(l+ 5,2,3,'j1 pt',' ','LOG')
      call multitop(l+ 6,2,3,'j1 log pt',' ','LOG')
      call multitop(l+ 7,2,3,'j1 y',' ','LOG')
      call multitop(l+ 8,2,3,'j1 eta',' ','LOG')
c
      call multitop(l+ 9,2,3,'j2 pt',' ','LOG')
      call multitop(l+10,2,3,'j2 log pt',' ','LOG')
      call multitop(l+11,2,3,'j2 y',' ','LOG')
      call multitop(l+12,2,3,'j2 eta',' ','LOG')
c
      call multitop(l+13,2,3,'bj1 pt',' ','LOG')
      call multitop(l+14,2,3,'bj1 log pt',' ','LOG')
      call multitop(l+15,2,3,'bj1 y',' ','LOG')
      call multitop(l+16,2,3,'bj1 eta',' ','LOG')
c
      call multitop(l+17,2,3,'bj2 pt',' ','LOG')
      call multitop(l+18,2,3,'bj2 log pt',' ','LOG')
      call multitop(l+19,2,3,'bj2 y',' ','LOG')
      call multitop(l+20,2,3,'bj2 eta',' ','LOG')
c
      call multitop(l+21,2,3,'syst pt',' ','LOG')
      call multitop(l+22,2,3,'syst log pt',' ','LOG')
      call multitop(l+23,2,3,'syst y',' ','LOG')
      call multitop(l+24,2,3,'syst eta',' ','LOG')
c
      enddo
      enddo
      call close_topdrawer_file
      return                
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      include 'nexternal.inc'
      include 'cuts.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      double precision wgts(*)
      integer ibody,mu,count_bj,count_j
      double precision wgt,var
      integer i,j,kk,l,nwgt_analysis,itop
      common/c_analysis/nwgt_analysis
      double precision www,pQCD(0:3,nexternal),palg,rfj,sycut,pjet(0:3
     $     ,nexternal),ptj1,yj1,etaj1,ptbj1,ybj1,etabj1,ptj2,yj2,etaj2
     $     ,ptbj2,ybj2,etabj2,p_top(0:3),pttop,ytop,etatop,psyst(0:3)
     $     ,ptsyst,ysyst,etasyst,p_bjet(0:3,nexternal)
      integer nQCD,nb,is_b(nexternal),njet,jet(nexternal),nbjet
      logical is_b_jet(nexternal)
      double precision getpt,getrapidity,getpseudorap
      external getpt,getrapidity,getpseudorap
      double precision pi
      parameter (pi=3.14159265358979312d0)
      if (nexternal.ne.5) then

         write (*,*) 'error #1 in analysis_fill: '/
     &        /'only for process "p p > t j [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(1)).le.5 .or. ipdg(1).eq.21)) then
         write (*,*) 'error #2 in analysis_fill: '/
     &        /'only for process "p p > t j [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(2)).le.5 .or. ipdg(2).eq.21)) then
         write (*,*) 'error #3 in analysis_fill: '/
     &        /'only for process "p p > t j [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(4)).le.5 .or. ipdg(4).eq.21)) then
         write (*,*) 'error #4 in analysis_fill: '/
     &        /'only for process "p p > t j [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(5)).le.5 .or. ipdg(5).eq.21)) then
         write (*,*) 'error #4 in analysis_fill: '/
     &        /'only for process "p p > t j [QCD]"'
         stop 1
      endif
      if (ipdg(3).ne.6) then
         write (*,*) 'error #5 in analysis_fill: '/
     &        /'only for process "p p > t j [QCD]"'
         stop 1
      endif

      nQCD=0
      nb=0
      do i=1,nexternal
         if(abs(ipdg(i)).le.5.or.ipdg(i).eq.21) then
            nQCD=nQCD+1
            do j=0,3
               pQCD(j,nQCD)=p(j,i)
            enddo
         endif
         if (abs(ipdg(i)).eq.5) then
            nb=nb+1
            is_b(nb)=nQCD
         endif
         if(abs(ipdg(i)).eq.6)itop=i
      enddo
      if(nb.ne.1)then
         write(*,*)'not one b',nb
         stop
      endif
            
c Define jet clustering parameters (from cuts.inc via the run_card.dat)
      palg=JETALGO              ! jet algorithm: 1.0=kt, 0.0=C/A, -1.0 = anti-kt
      rfj=JETRADIUS             ! the radius parameter
      sycut=PTJ                 ! minimum transverse momentum

c******************************************************************************
c     call FASTJET to get all the jets
c
c     INPUT:
c     input momenta:               pQCD(0:3,nexternal), energy is 0th component
c     number of input momenta:     nQCD
c     radius parameter:            rfj
c     minumum jet pt:              sycut
c     jet algorithm:               palg, 1.0=kt, 0.0=C/A, -1.0 = anti-kt
c
c     OUTPUT:
c     jet momenta:                             pjet(0:3,nexternal), E is 0th cmpnt
c     the number of jets (with pt > SYCUT):    njet
c     the jet for a given particle 'i':        jet(i),   note that this is
c     the particle in pQCD, which doesn't necessarily correspond to the particle
c     label in the process
c
      call amcatnlo_fastjetppgenkt(pQCD,nQCD,rfj,sycut,palg,pjet,njet,jet)
c
c******************************************************************************
      if(njet.gt.2)then
         write(*,*)'more than two jets',njet
         stop
      endif

C b-jet
      do i=1,njet
         is_b_jet(i)=.false.
         if(jet(is_b(nb)).eq.i)is_b_jet(i)=.true.
      enddo

      do i=0,3
         p_top(i)=p(i,itop)
      enddo
      pttop = getpt(p_top)
      ytop  = getrapidity(p_top)
      etatop= getpseudorap(p_top)

      count_j=0
      count_bj=0
      do i=1,njet
         if(.not.is_b_jet(i))then
            count_j=count_j+1
            if(count_j.eq.1)then
               ptj1 = getpt(pjet(0,i))
               yj1  = getrapidity(pjet(0,i))
               etaj1= getpseudorap(pjet(0,i))
               do mu=0,3
                  psyst(mu)=p_top(mu)+pjet(mu,i)
               enddo
               ptsyst = getpt(psyst)
               ysyst  = getrapidity(psyst)
               etasyst= getpseudorap(psyst)
            elseif(count_j.eq.2)then
               ptj2 = getpt(pjet(0,i))
               yj2  = getrapidity(pjet(0,i))
               etaj2= getpseudorap(pjet(0,i))
            endif
         elseif(is_b_jet(i))then
            count_bj=count_bj+1
            if (count_bj.eq.1) then
               ptbj1 = getpt(pjet(0,i))
               ybj1  = getrapidity(pjet(0,i))
               etabj1= getpseudorap(pjet(0,i))
            elseif (count_bj.eq.2) then
               ptbj2 = getpt(pjet(0,i))
               ybj2  = getrapidity(pjet(0,i))
               etabj2= getpseudorap(pjet(0,i))
            endif
         endif
      enddo
      nbjet=count_bj
c fill the histograms
      do i=1,2
         do kk=1,nwgt_analysis
            www=wgts(kk)
            l=(kk-1)*48+(i-1)*24
            if (ibody.ne.3 .and.i.eq.2) cycle
            call mfill(l+1,pttop,www)
            if(pttop.gt.0d0) call mfill(l+2,log10(pttop),www)
            call mfill(l+3,ytop,www)
            call mfill(l+4,etatop,www)
            if(njet.ge.1)then
               call mfill(l+5,ptj1,www)
               if (ptj1.gt.0d0) call mfill(l+6,log10(ptj1),www)
               call mfill(l+7,yj1,www)
               call mfill(l+8,etaj1,www)
               call mfill(l+21,ptsyst,www)
               if(ptsyst.gt.0d0) call mfill(l+22,log10(ptsyst),www)
               call mfill(l+23,ysyst,www)
               call mfill(l+24,etasyst,www)
            endif
            if(njet.ge.2)then
               call mfill(l+9,ptj2,www)
               if(ptj2.gt.0d0) call mfill(l+10,log10(ptj2),www)
               call mfill(l+11,yj2,www)
               call mfill(l+12,etaj2,www)
            endif
            if(nbjet.ge.1)then
               call mfill(l+13,ptbj1,www)
               if (ptbj1.gt.0d0) call mfill(l+14,log10(ptbj1),www)
               call mfill(l+15,ybj1,www)
               call mfill(l+16,etabj1,www)
            endif
            if(nbjet.ge.2)then
               call mfill(l+17,ptbj2,www)
               if (ptbj2.gt.0d0) call mfill(l+18,log10(ptbj2),www)
               call mfill(l+19,ybj2,www)
               call mfill(l+20,etabj2,www)
            endif
         enddo
      enddo
      
 999  return      
      end


      function getrapidity(p)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y,p(0:3)
      parameter (tiny=1.d-5)
c
      en=p(0)
      pl=p(3)
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
        if( (xplus/xminus).gt.tiny )then
          y=0.5d0*log( xplus/xminus )
        else
          y=sign(1.d0,pl)*1.d8
        endif
      else
        y=sign(1.d0,pl)*1.d8
      endif
      getrapidity=y
      return
      end


      function getpseudorap(p)
      implicit none
      real*8 getpseudorap,en,ptx,pty,pl,tiny,pt,eta,th,p(0:3)
      parameter (tiny=1.d-5)
c
      en=p(0)
      ptx=p(1)
      pty=p(2)
      pl=p(3)
      pt=sqrt(ptx**2+pty**2)
      if(pt.lt.tiny.and.abs(pl).lt.tiny)then
        eta=sign(1.d0,pl)*1.d8
      else
        th=atan2(pt,pl)
        eta=-log(tan(th/2.d0))
      endif
      getpseudorap=eta
      return
      end

      function getpt(p)
      implicit none
      real*8 getpt,p(0:3)
      getpt=dsqrt(p(1)**2+p(2)**2)
      return
      end

