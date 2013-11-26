c
c Example analysis for "p p > h [QCD]" process.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(nwgt,weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      integer kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
      include 'dbook.inc'
      call inihist
      nwgt_analysis=nwgt
      if (nwgt_analysis*40.gt.nplots/4) then
         write (*,*) 'error in analysis_begin: '/
     &        /'too many histograms, increase NPLOTS to',
     &        nwgt_analysis*40*4
         stop 1
      endif
      do kk=1,nwgt_analysis
      l=(kk-1)*40
      call bookup(l+1,'Higgs pT '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call bookup(l+2,'Higgs pT '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call bookup(l+3,'Higgs log[pT] '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)
      call bookup(l+4,'Higgs pT,|y_H|<2 '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call bookup(l+5,'Higgs pT,|y_H|<2 '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call bookup(l+6,'Higgs log[pT],|y_H|<2 '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)

      call bookup(l+7,'j1 pT '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call bookup(l+8,'j1 pT '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call bookup(l+9,'j1 log[pT] '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)
      call bookup(l+10,'j1 pT,|y_j1|<2 '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call bookup(l+11,'j1 pT,|y_j1|<2 '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call bookup(l+12,'j1 log[pT],|y_j1|<2 '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)

      call bookup(l+13,'Inc j pT '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call bookup(l+14,'Inc j pT '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call bookup(l+15,'Inc j log[pT] '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)
      call bookup(l+16,'Inc j pT,|y_Ij|<2 '//weights_info(kk)
     $     ,2.d0,0.d0,2.d2)
      call bookup(l+17,'Inc j pT,|y_Ij|<2 '//weights_info(kk)
     $     ,5.d0,0.d0,5.d2)
      call bookup(l+18,'Inc j log[pT],|y_Ij|<2'//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)

      call bookup(l+19,'Higgs y '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call bookup(l+20,'Higgs y,pT_H>10GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)
      call bookup(l+21,'Higgs y,pT_H>30GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)
      call bookup(l+22,'Higgs y,pT_H>50GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)
      call bookup(l+23,'Higgs y,pT_H>70GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)
      call bookup(l+24,'Higgs y,pt_H>90GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)

      call bookup(l+25,'j1 y '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call bookup(l+26,'j1 y,pT_j1>10GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call bookup(l+27,'j1 y,pT_j1>30GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call bookup(l+28,'j1 y,pT_j1>50GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call bookup(l+29,'j1 y,pT_j1>70GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call bookup(l+30,'j1 y,pT_j1>90GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)

      call bookup(l+31,'H-j1 y '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call bookup(l+32,'H-j1 y,pT_j1>10GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call bookup(l+33,'H-j1 y,pT_j1>30GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call bookup(l+34,'H-j1 y,pT_j1>50GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call bookup(l+35,'H-j1 y,pT_j1>70GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call bookup(l+36,'H-j1 y,pT_j1>90GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      
      call bookup(l+37,'njets '//weights_info(kk)
     $     ,1.d0,-0.5d0,10.5d0)
      call bookup(l+38,'njets,|y_j|<2.5 '//weights_info(kk)
     $     ,1.d0,-0.5d0,10.5d0)
      call bookup(l+39,'xsec '//weights_info(kk)
     $     ,1.d0,-0.5d0,2.5d0)

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
      do kk=1,nwgt_analysis
         l=(kk-1)*40
         call multitop(l+1,3,2,'Higgs pT (GeV)',' ','LOG')
         call multitop(l+2,3,2,'Higgs pT (GeV)',' ','LOG')
         call multitop(l+3,3,2,'Higgs log(pT/GeV)',' ','LOG')
         call multitop(l+4,3,2,'Higgs pT (GeV)',' ','LOG')
         call multitop(l+5,3,2,'Higgs pT (GeV)',' ','LOG')
         call multitop(l+6,3,2,'Higgs log(pT/GeV)',' ','LOG')
c     
         call multitop(l+7,3,2,'j1 pT (GeV)',' ','LOG')
         call multitop(l+8,3,2,'j1 pT (GeV)',' ','LOG')
         call multitop(l+9,3,2,'j1 log(pT/GeV)',' ','LOG')
         call multitop(l+10,3,2,'j1 pT (GeV)',' ','LOG')
         call multitop(l+11,3,2,'j1 pT (GeV)',' ','LOG')
         call multitop(l+12,3,2,'j1 log(pT/GeV)',' ','LOG')
c
         call multitop(l+13,3,2,'Inc j pT (GeV)',' ','LOG')
         call multitop(l+14,3,2,'Inc j pT (GeV)',' ','LOG')
         call multitop(l+15,3,2,'Inc j log(pT/GeV)',' ','LOG')
         call multitop(l+16,3,2,'Inc j pT (GeV)',' ','LOG')
         call multitop(l+17,3,2,'Inc j pT (GeV)',' ','LOG')
         call multitop(l+18,3,2,'Inc j log(pT/GeV)',' ','LOG')
c
         call multitop(l+19,3,2,'Higgs y',' ','LOG')
         call multitop(l+20,3,2,'Higgs y',' ','LOG')
         call multitop(l+21,3,2,'Higgs y',' ','LOG')
         call multitop(l+22,3,2,'Higgs y',' ','LOG')
         call multitop(l+23,3,2,'Higgs y',' ','LOG')
         call multitop(l+24,3,2,'Higgs y',' ','LOG')
c     
         call multitop(l+25,3,2,'j1 y',' ','LOG')
         call multitop(l+26,3,2,'j1 y',' ','LOG')
         call multitop(l+27,3,2,'j1 y',' ','LOG')
         call multitop(l+28,3,2,'j1 y',' ','LOG')
         call multitop(l+29,3,2,'j1 y',' ','LOG')
         call multitop(l+30,3,2,'j1 y',' ','LOG')
c
         call multitop(l+31,3,2,'H-j1 y',' ','LOG')
         call multitop(l+32,3,2,'H-j1 y',' ','LOG')
         call multitop(l+33,3,2,'H-j1 y',' ','LOG')
         call multitop(l+34,3,2,'H-j1 y',' ','LOG')
         call multitop(l+35,3,2,'H-j1 y',' ','LOG')
         call multitop(l+36,3,2,'H-j1 y',' ','LOG')

         call multitop(l+37,3,2,'njets',' ','LOG')
         call multitop(l+38,3,2,'njets',' ','LOG')
         call multitop(l+39,3,2,'xsec',' ','LOG')
      enddo
      call close_topdrawer_file
      return                
      end



cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      include 'nexternal.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      double precision wgts(*)
      integer ibody
      double precision wgt,var
      integer i,kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
      double precision www,pph(0:3),ppj1(0:3),pth,yh,ptj1,yj1,y_central
     $     ,ptcut,njdble,njcdble
      integer njet,njet_central,nj
      double precision getrapidity
      external getrapidity
      if (nexternal.ne.4) then
         write (*,*) 'error #1 in analysis_fill: '/
     &        /'only for process "p p > h [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(1)).le.5 .or. ipdg(1).eq.21)) then
         write (*,*) 'error #2 in analysis_fill: '/
     &        /'only for process "p p > h [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(2)).le.5 .or. ipdg(2).eq.21)) then
         write (*,*) 'error #3 in analysis_fill: '/
     &        /'only for process "p p > h [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(4)).le.5 .or. ipdg(4).eq.21)) then
         write (*,*) 'error #4 in analysis_fill: '/
     &        /'only for process "p p > h [QCD]"'
         stop 1
      endif
      if (ipdg(3).ne.25) then
         write (*,*) 'error #5 in analysis_fill: '/
     &        /'only for process "p p > h [QCD]"'
         stop 1
      endif
c
      do i=0,3
         pph(i)=p(i,3)
         ppj1(i)=p(i,4)
      enddo
c Higgs variables
      pth=sqrt(pph(1)**2+pph(2)**2)
      yh=getrapidity(pph(0),pph(3))
c hardest jet variables
      ptj1=sqrt(ppj1(1)**2+ppj1(2)**2)
      yj1=getrapidity(ppj1(0),ppj1(3))
      njet=0
      njet_central=0
 8    y_central=2.5d0
      ptcut=1d1
      if(ptj1.gt.ptcut)njet=1
      if(ptj1.gt.ptcut.and.abs(yj1).le.y_central)njet_central=1
      njdble=dble(njet)
      njcdble=dble(njet_central)
C
      do kk=1,nwgt_analysis
      www=wgts(kk)
      l=(kk-1)*40
      call mfill(l+1,pth,WWW)
      call mfill(l+2,pth,WWW)
      if(pth.gt.0.d0)call mfill(l+3,log10(pth),WWW)
      if(abs(yh).le.2.d0)then
         call mfill(l+4,pth,WWW)
         call mfill(l+5,pth,WWW)
         if(pth.gt.0.d0)call mfill(l+6,log10(pth),WWW)
      endif
c
      if(njet.ge.1)then
         call mfill(l+7,ptj1,WWW)
         call mfill(l+8,ptj1,WWW)
         if(ptj1.gt.0.d0)call mfill(l+9,log10(ptj1),WWW)
         if(abs(yj1).le.2.d0)then
            call mfill(l+10,ptj1,WWW)
            call mfill(l+11,ptj1,WWW)
            if(ptj1.gt.0.d0)call mfill(l+12,log10(ptj1),WWW)
         endif
c
         do nj=1,njet
            call mfill(l+13,ptj1,WWW)
            call mfill(l+14,ptj1,WWW)
            if(ptj1.gt.0.d0)call mfill(l+15,log10(ptj1),WWW)
            if(abs(yj1).le.2.d0)then
               call mfill(l+16,ptj1,WWW)
               call mfill(l+17,ptj1,WWW)
               if(ptj1.gt.0d0)call mfill(l+18,log10(ptj1),WWW)
            endif
         enddo
      endif
c
      call mfill(l+19,yh,WWW)
      if(pth.ge.10.d0) call mfill(l+20,yh,WWW)
      if(pth.ge.30.d0) call mfill(l+21,yh,WWW)
      if(pth.ge.50.d0) call mfill(l+22,yh,WWW)
      if(pth.ge.70.d0) call mfill(l+23,yh,WWW)
      if(pth.ge.90.d0) call mfill(l+24,yh,WWW)  
c     
      if(njet.ge.1)then
         call mfill(l+25,yj1,WWW)
         if(ptj1.ge.10.d0) call mfill(l+26,yj1,WWW)
         if(ptj1.ge.30.d0) call mfill(l+27,yj1,WWW)
         if(ptj1.ge.50.d0) call mfill(l+28,yj1,WWW)
         if(ptj1.ge.70.d0) call mfill(l+29,yj1,WWW)
         if(ptj1.ge.90.d0) call mfill(l+30,yj1,WWW)
c     
         call mfill(l+31,yh-yj1,WWW)
         if(ptj1.ge.10.d0) call mfill(l+32,yh-yj1,WWW)
         if(ptj1.ge.30.d0) call mfill(l+33,yh-yj1,WWW)
         if(ptj1.ge.50.d0) call mfill(l+34,yh-yj1,WWW)
         if(ptj1.ge.70.d0) call mfill(l+35,yh-yj1,WWW)
         if(ptj1.ge.90.d0) call mfill(l+36,yh-yj1,WWW)
      endif
c
      call mfill(l+37,njdble,WWW)
      call mfill(l+38,njcdble,WWW)
      call mfill(l+39,1d0,WWW)
c      
      enddo
C
 999  return      
      end



      function getrapidity(en,pl)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-8)
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
         if( (xplus/xminus).gt.tiny.and.(xminus/xplus).gt.tiny)then
            y=0.5d0*log( xplus/xminus  )
         else
            y=sign(1.d0,pl)*1.d8
         endif
      else 
         y=sign(1.d0,pl)*1.d8
      endif
      getrapidity=y
      return
      end


      function getpseudorap(en,ptx,pty,pl)
      implicit none
      real*8 getpseudorap,en,ptx,pty,pl,tiny,pt,eta,th
      parameter (tiny=1.d-5)
c
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
