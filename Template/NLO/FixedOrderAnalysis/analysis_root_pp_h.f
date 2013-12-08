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
      call open_root_file()
      nwgt_analysis=nwgt
      do kk=1,nwgt_analysis
      l=(kk-1)*40
      call rbook(l+1,'Higgs pT '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call rbook(l+2,'Higgs pT '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call rbook(l+3,'Higgs log[pT] '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)
      call rbook(l+4,'Higgs pT,|y_H|<2 '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call rbook(l+5,'Higgs pT,|y_H|<2 '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call rbook(l+6,'Higgs log[pT],|y_H|<2 '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)

      call rbook(l+7,'j1 pT '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call rbook(l+8,'j1 pT '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call rbook(l+9,'j1 log[pT] '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)
      call rbook(l+10,'j1 pT,|y_j1|<2 '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call rbook(l+11,'j1 pT,|y_j1|<2 '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call rbook(l+12,'j1 log[pT],|y_j1|<2 '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)

      call rbook(l+13,'Inc j pT '//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call rbook(l+14,'Inc j pT '//weights_info(kk)
     $     ,5.d0,0.d0,500.d0)
      call rbook(l+15,'Inc j log[pT] '//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)
      call rbook(l+16,'Inc j pT,|y_Ij|<2 '//weights_info(kk)
     $     ,2.d0,0.d0,2.d2)
      call rbook(l+17,'Inc j pT,|y_Ij|<2 '//weights_info(kk)
     $     ,5.d0,0.d0,5.d2)
      call rbook(l+18,'Inc j log[pT],|y_Ij|<2'//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)

      call rbook(l+19,'Higgs y '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call rbook(l+20,'Higgs y,pT_H>10GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)
      call rbook(l+21,'Higgs y,pT_H>30GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)
      call rbook(l+22,'Higgs y,pT_H>50GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)
      call rbook(l+23,'Higgs y,pT_H>70GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)
      call rbook(l+24,'Higgs y,pt_H>90GeV '//weights_info(kk)
     $     ,0.12d0,-6.d0,6.d0)

      call rbook(l+25,'j1 y '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call rbook(l+26,'j1 y,pT_j1>10GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call rbook(l+27,'j1 y,pT_j1>30GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call rbook(l+28,'j1 y,pT_j1>50GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call rbook(l+29,'j1 y,pT_j1>70GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call rbook(l+30,'j1 y,pT_j1>90GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)

      call rbook(l+31,'H-j1 y '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call rbook(l+32,'H-j1 y,pT_j1>10GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call rbook(l+33,'H-j1 y,pT_j1>30GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call rbook(l+34,'H-j1 y,pT_j1>50GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call rbook(l+35,'H-j1 y,pT_j1>70GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      call rbook(l+36,'H-j1 y,pT_j1>90GeV '//weights_info(kk)
     $     ,0.2d0,-6.d0,6.d0)
      
      call rbook(l+37,'njets '//weights_info(kk)
     $     ,1.d0,-0.5d0,10.5d0)
      call rbook(l+38,'njets,|y_j|<2.5 '//weights_info(kk)
     $     ,1.d0,-0.5d0,10.5d0)
      call rbook(l+39,'xsec '//weights_info(kk)
     $     ,1.d0,-0.5d0,2.5d0)

      enddo
      return
      end



cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_end(xnorm)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      double precision xnorm
      integer i,jj
      integer kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
c Do not touch the following lines. These lines make sure that the
c histograms will have the correct overall normalisation: cross section
c (in pb) per bin.
      do kk=1,nwgt_analysis
         l=(kk-1)*40
         do jj=1,40
            call ropera(l+jj,'+',l+jj,l+jj,xnorm,0.d0)
         enddo
      enddo
      call close_root_file
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
      call rfill(l+1,pth,WWW)
      call rfill(l+2,pth,WWW)
      if(pth.gt.0.d0)call rfill(l+3,log10(pth),WWW)
      if(abs(yh).le.2.d0)then
         call rfill(l+4,pth,WWW)
         call rfill(l+5,pth,WWW)
         if(pth.gt.0.d0)call rfill(l+6,log10(pth),WWW)
      endif
c
      if(njet.ge.1)then
         call rfill(l+7,ptj1,WWW)
         call rfill(l+8,ptj1,WWW)
         if(ptj1.gt.0.d0)call rfill(l+9,log10(ptj1),WWW)
         if(abs(yj1).le.2.d0)then
            call rfill(l+10,ptj1,WWW)
            call rfill(l+11,ptj1,WWW)
            if(ptj1.gt.0.d0)call rfill(l+12,log10(ptj1),WWW)
         endif
c
         do nj=1,njet
            call rfill(l+13,ptj1,WWW)
            call rfill(l+14,ptj1,WWW)
            if(ptj1.gt.0.d0)call rfill(l+15,log10(ptj1),WWW)
            if(abs(yj1).le.2.d0)then
               call rfill(l+16,ptj1,WWW)
               call rfill(l+17,ptj1,WWW)
               if(ptj1.gt.0d0)call rfill(l+18,log10(ptj1),WWW)
            endif
         enddo
      endif
c
      call rfill(l+19,yh,WWW)
      if(pth.ge.10.d0) call rfill(l+20,yh,WWW)
      if(pth.ge.30.d0) call rfill(l+21,yh,WWW)
      if(pth.ge.50.d0) call rfill(l+22,yh,WWW)
      if(pth.ge.70.d0) call rfill(l+23,yh,WWW)
      if(pth.ge.90.d0) call rfill(l+24,yh,WWW)  
c     
      if(njet.ge.1)then
         call rfill(l+25,yj1,WWW)
         if(ptj1.ge.10.d0) call rfill(l+26,yj1,WWW)
         if(ptj1.ge.30.d0) call rfill(l+27,yj1,WWW)
         if(ptj1.ge.50.d0) call rfill(l+28,yj1,WWW)
         if(ptj1.ge.70.d0) call rfill(l+29,yj1,WWW)
         if(ptj1.ge.90.d0) call rfill(l+30,yj1,WWW)
c     
         call rfill(l+31,yh-yj1,WWW)
         if(ptj1.ge.10.d0) call rfill(l+32,yh-yj1,WWW)
         if(ptj1.ge.30.d0) call rfill(l+33,yh-yj1,WWW)
         if(ptj1.ge.50.d0) call rfill(l+34,yh-yj1,WWW)
         if(ptj1.ge.70.d0) call rfill(l+35,yh-yj1,WWW)
         if(ptj1.ge.90.d0) call rfill(l+36,yh-yj1,WWW)
      endif
c
      call rfill(l+37,njdble,WWW)
      call rfill(l+38,njcdble,WWW)
      call rfill(l+39,1d0,WWW)
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
