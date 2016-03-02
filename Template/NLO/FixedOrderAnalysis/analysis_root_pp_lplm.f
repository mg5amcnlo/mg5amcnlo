c
c Example analysis for "p p > l+ l- [QCD]" process.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(nwgt,weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      integer j,kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
      character*5 cc(2)
      data cc/'     ','cuts '/
      real * 8 bin,xmi,xms,pi
      parameter (pi=3.14159265358979312d0)
      call open_root_file()
      nwgt_analysis=nwgt
      xmi=50.d0
      xms=130.d0
      bin=0.8d0
      do kk=1,nwgt_analysis
      do j=1,2
      l=(kk-1)*42+(j-1)*21
      call rbook(l+ 1,'V pt      '//cc(j)//weights_info(kk)
     &     ,2.d0,0.d0,200.d0)
      call rbook(l+ 2,'V pt      '//cc(j)//weights_info(kk)
     &     ,10.d0,0.d0,1000.d0)
      call rbook(l+ 3,'V log[pt] '//cc(j)//weights_info(kk)
     &     ,0.05d0,0.1d0,5.d0)
      call rbook(l+ 4,'V y       '//cc(j)//weights_info(kk)
     &     ,0.2d0,-9.d0,9.d0)
      call rbook(l+ 5,'V eta     '//cc(j)//weights_info(kk)
     &     ,0.2d0,-9.d0,9.d0)
      call rbook(l+ 6,'mV        '//cc(j)//weights_info(kk)
     &     ,bin,xmi,xms)
c
      call rbook(l+ 7,'lm pt      '//cc(j)//weights_info(kk)
     &     ,2.d0,0.d0,200.d0)
      call rbook(l+ 8,'lm pt      '//cc(j)//weights_info(kk)
     &     ,10.d0,0.d0,1000.d0)
      call rbook(l+ 9,'lm log[pt] '//cc(j)//weights_info(kk)
     &     ,0.05d0,0.1d0,5.d0)
      call rbook(l+10,'lm eta     '//cc(j)//weights_info(kk)
     &     ,0.2d0,-9.d0,9.d0)
      call rbook(l+11,'lp pt      '//cc(j)//weights_info(kk)
     &     ,2.d0,0.d0,200.d0)
      call rbook(l+12,'lp pt      '//cc(j)//weights_info(kk)
     &     ,10.d0,0.d0,1000.d0)
      call rbook(l+13,'lp log[pt] '//cc(j)//weights_info(kk)
     &     ,0.05d0,0.1d0,5.d0)
      call rbook(l+14,'lp eta     '//cc(j)//weights_info(kk)
     &     ,0.2d0,-9.d0,9.d0)
c
      call rbook(l+15,'lmlp delta eta     '//cc(j)//weights_info(kk)
     $     ,0.2d0,-9.d0,9.d0)
      call rbook(l+16,'lmlp azimt         '//cc(j)//weights_info(kk)
     $     ,pi/20.d0,0.d0,pi)
      call rbook(l+17,'lmlp log[pi-azimt] '//cc(j)//weights_info(kk)
     $     ,0.05d0,-4.d0,0.1d0)
      call rbook(l+18,'lmlp inv m         '//cc(j)//weights_info(kk)
     $     ,bin,xmi,xms)
      call rbook(l+19,'lmlp pt            '//cc(j)//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call rbook(l+20,'lmlp log[pt]       '//cc(j)//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)
c
      call rbook(l+21,'total'//cc(j)//weights_info(kk),1.d0,-1.d0,1.d0)
      enddo
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
      do i=1,2
      l=(kk-1)*42+(i-1)*21
      do jj=1,21
         call ropera(l+jj,'+',l+jj,l+jj,xnorm,0.d0)
      enddo
      enddo
      enddo
c
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
      double precision www,ppl(0:3),pplb(0:3),ppv(0:3),ycut,xmv,ptv,yv
     $     ,etav,ptl,yl,etal,ptlb,ylb,etalb,ptpair,azi,azinorm,xmll
     $     ,detallb
      double precision getrapidity,getinvm,getpseudorap,getdelphi
      external getrapidity,getinvm,getpseudorap,getdelphi
      double precision pi
      parameter (pi=3.14159265358979312d0)
      if (nexternal.ne.5) then
         write (*,*) 'error #1 in analysis_fill: '/
     &        /'only for process "p p > l+ l- [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(1)).le.5 .or. ipdg(1).eq.21)) then
         write (*,*) 'error #2 in analysis_fill: '/
     &        /'only for process "p p > l+ l- [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(2)).le.5 .or. ipdg(2).eq.21)) then
         write (*,*) 'error #3 in analysis_fill: '/
     &        /'only for process "p p > l+ l- [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(5)).le.5 .or. ipdg(5).eq.21)) then
         write (*,*) 'error #4 in analysis_fill: '/
     &        /'only for process "p p > l+ l- [QCD]"'
         stop 1
      endif
      if (ipdg(3).ne.-11.and.ipdg(3).ne.-13.and.ipdg(3).ne.-15) then
         write (*,*) 'error #5 in analysis_fill: '/
     &        /'only for process "p p > l+ l- [QCD]"'
         stop 1
      endif
      if (ipdg(4).ne.11.and.ipdg(4).ne.13.and.ipdg(4).ne.15) then
         write (*,*) 'error #6 in analysis_fill: '/
     &        /'only for process "p p > l+ l- [QCD]"'
         stop 1
      endif

      DO i=0,3
        ppl(i)=p(i,4)
        pplb(i)=p(i,3)
        ppv(i)=ppl(i)+pplb(i)
      ENDDO

C FILL THE HISTOS
      YCUT=2.5D0
C Variables of the vector boson
      xmv=getinvm(ppv(0),ppv(1),ppv(2),ppv(3))
      ptv=sqrt(ppv(1)**2+ppv(2)**2)
      yv=getrapidity(ppv(0),ppv(3))
      etav=getpseudorap(ppv(0),ppv(1),ppv(2),ppv(3))
C Variables of the leptons
      ptl=sqrt(ppl(1)**2+ppl(2)**2)
      yl=getrapidity(ppl(0),ppl(3))
      etal=getpseudorap(ppl(0),ppl(1),ppl(2),ppl(3))
c
      ptlb=sqrt(pplb(1)**2+pplb(2)**2)
      ylb=getrapidity(pplb(0),pplb(3))
      etalb=getpseudorap(pplb(0),pplb(1),pplb(2),pplb(3))
c
      ptpair=ptv
      azi=getdelphi(ppl(1),ppl(2),pplb(1),pplb(2))
      azinorm=(pi-azi)/pi
      xmll=xmv
      detallb=etal-etalb
c
      do kk=1,nwgt_analysis
      www=wgts(kk)
      l=(kk-1)*42
      call rfill(l+1,(ptv),(WWW))
      call rfill(l+2,(ptv),(WWW))
      if(ptv.gt.0.d0)call rfill(l+3,(log10(ptv)),(WWW))
      call rfill(l+4,(yv),(WWW))
      call rfill(l+5,(etav),(WWW))
      call rfill(l+6,(xmv),(WWW))
c
      call rfill(l+7,(ptl),(WWW))
      call rfill(l+8,(ptl),(WWW))
      if(ptl.gt.0.d0)call rfill(l+9,(log10(ptl)),(WWW))
      call rfill(l+10,(etal),(WWW))
      call rfill(l+11,(ptlb),(WWW))
      call rfill(l+12,(ptlb),(WWW))
      if(ptlb.gt.0.d0)call rfill(l+13,(log10(ptlb)),(WWW))
      call rfill(l+14,(etalb),(WWW))
c
      call rfill(l+15,(detallb),(WWW))
      call rfill(l+16,(azi),(WWW))
      if(azinorm.gt.0.d0)
     #  call rfill(l+17,(log10(azinorm)),(WWW))
      call rfill(l+18,(xmll),(WWW))
      call rfill(l+19,(ptpair),(WWW))
      if(ptpair.gt.0)call rfill(l+20,(log10(ptpair)),(WWW))
      call rfill(l+21,(0d0),(WWW))
c
      l=l+21

      if(abs(etav).lt.ycut)then
        call rfill(l+1,(ptv),(WWW))
        call rfill(l+2,(ptv),(WWW))
        if(ptv.gt.0.d0)call rfill(l+3,(log10(ptv)),(WWW))
      endif
      if(ptv.gt.20.d0)then
        call rfill(l+4,(yv),(WWW))
        call rfill(l+5,(etav),(WWW))
      endif
      if(abs(etav).lt.ycut.and.ptv.gt.20.d0)then
         call rfill(l+6,(xmv),(WWW))
         call rfill(l+21,(0d0),(WWW))
      endif
c
      if(abs(etal).lt.ycut)then
        call rfill(l+7,(ptl),(WWW))
        call rfill(l+8,(ptl),(WWW))
        if(ptl.gt.0.d0)call rfill(l+9,(log10(ptl)),(WWW))
      endif
      if(ptl.gt.20.d0)call rfill(l+10,(etal),(WWW))
      if(abs(etalb).lt.ycut)then
        call rfill(l+11,(ptlb),(WWW))
        call rfill(l+12,(ptlb),(WWW))
        if(ptlb.gt.0.d0)call rfill(l+13,(log10(ptlb)),(WWW))
      endif
      if(ptlb.gt.20.d0)call rfill(l+14,(etalb),(WWW))
c
      if( abs(etal).lt.ycut.and.abs(etalb).lt.ycut .and.
     #    ptl.gt.20.d0.and.ptlb.gt.20.d0)then
        call rfill(l+15,(detallb),(WWW))
        call rfill(l+16,(azi),(WWW))
        if(azinorm.gt.0.d0)
     #    call rfill(l+17,(log10(azinorm)),(WWW))
        call rfill(l+18,(xmll),(WWW))
        call rfill(l+19,(ptpair),(WWW))
        if(ptpair.gt.0) 
     #    call rfill(l+20,(log10(ptpair)),(WWW))
      endif

      enddo

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


      function getinvm(en,ptx,pty,pl)
      implicit none
      real*8 getinvm,en,ptx,pty,pl,tiny,tmp
      parameter (tiny=1.d-5)
c
      tmp=en**2-ptx**2-pty**2-pl**2
      if(tmp.gt.0.d0)then
        tmp=sqrt(tmp)
      elseif(tmp.gt.-tiny)then
        tmp=0.d0
      else
        write(*,*)'Attempt to compute a negative mass'
        stop
      endif
      getinvm=tmp
      return
      end


      function getdelphi(ptx1,pty1,ptx2,pty2)
      implicit none
      real*8 getdelphi,ptx1,pty1,ptx2,pty2,tiny,pt1,pt2,tmp
      parameter (tiny=1.d-5)
c
      pt1=sqrt(ptx1**2+pty1**2)
      pt2=sqrt(ptx2**2+pty2**2)
      if(pt1.ne.0.d0.and.pt2.ne.0.d0)then
        tmp=ptx1*ptx2+pty1*pty2
        tmp=tmp/(pt1*pt2)
        if(abs(tmp).gt.1.d0+tiny)then
          write(*,*)'Cosine larger than 1'
          stop
        elseif(abs(tmp).ge.1.d0)then
          tmp=sign(1.d0,tmp)
        endif
        tmp=acos(tmp)
      else
        tmp=1.d8
      endif
      getdelphi=tmp
      return
      end
