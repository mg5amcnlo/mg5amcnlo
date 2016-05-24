c
c Example analysis for "p p > l+ l- [QCD]" process.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(nwgt,weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      integer j,kk,l
      character*5 cc(2)
      data cc/'     ','cuts '/
      real * 8 xmi,xms,pi
      parameter (pi=3.14159265358979312d0)
      call HwU_inithist(nwgt,weights_info)
      xmi=50.d0
      xms=130.d0
      do j=1,2
      l=(j-1)*21
      call HwU_book(l+ 1,'V pt      '//cc(j) ,100,0.d0,200.d0)
      call HwU_book(l+ 2,'V pt      '//cc(j) ,100,0.d0,1000.d0)
      call HwU_book(l+ 3,'V log[pt] '//cc(j) ,98,0.1d0,5.d0)
      call HwU_book(l+ 4,'V y       '//cc(j) ,90,-9.d0,9.d0)
      call HwU_book(l+ 5,'V eta     '//cc(j) ,90,-9.d0,9.d0)
      call HwU_book(l+ 6,'mV        '//cc(j) ,100,xmi,xms)
c
      call HwU_book(l+ 7,'lm pt      '//cc(j) ,100,0.d0,200.d0)
      call HwU_book(l+ 8,'lm pt      '//cc(j) ,100,0.d0,1000.d0)
      call HwU_book(l+ 9,'lm log[pt] '//cc(j) ,98,0.1d0,5.d0)
      call HwU_book(l+10,'lm eta     '//cc(j) ,90,-9.d0,9.d0)
      call HwU_book(l+11,'lp pt      '//cc(j) ,100,0.d0,200.d0)
      call HwU_book(l+12,'lp pt      '//cc(j) ,100,0.d0,1000.d0)
      call HwU_book(l+13,'lp log[pt] '//cc(j) ,98,0.1d0,5.d0)
      call HwU_book(l+14,'lp eta     '//cc(j) ,90,-9.d0,9.d0)
c
      call HwU_book(l+15,'lmlp delta eta     '//cc(j) ,90,-9.d0,9.d0)
      call HwU_book(l+16,'lmlp azimt         '//cc(j) ,20,0.d0,pi)
      call HwU_book(l+17,'lmlp log[pi-azimt] '//cc(j) ,82,-4.d0,0.1d0)
      call HwU_book(l+18,'lmlp inv m         '//cc(j) ,100,xmi,xms)
      call HwU_book(l+19,'lmlp pt            '//cc(j) ,100,0.d0,200.d0)
      call HwU_book(l+20,'lmlp log[pt]       '//cc(j) ,98,0.1d0,5.d0)
c
      call HwU_book(l+21,'total'//cc(j),2,-1.d0,1.d0)
      enddo
      return
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_end(dummy)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      double precision dummy
      call HwU_write_file
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
      l=0
      call HwU_fill(l+1,(ptv),(WGTS))
      call HwU_fill(l+2,(ptv),(WGTS))
      if(ptv.gt.0.d0)call HwU_fill(l+3,(log10(ptv)),(WGTS))
      call HwU_fill(l+4,(yv),(WGTS))
      call HwU_fill(l+5,(etav),(WGTS))
      call HwU_fill(l+6,(xmv),(WGTS))
c
      call HwU_fill(l+7,(ptl),(WGTS))
      call HwU_fill(l+8,(ptl),(WGTS))
      if(ptl.gt.0.d0)call HwU_fill(l+9,(log10(ptl)),(WGTS))
      call HwU_fill(l+10,(etal),(WGTS))
      call HwU_fill(l+11,(ptlb),(WGTS))
      call HwU_fill(l+12,(ptlb),(WGTS))
      if(ptlb.gt.0.d0)call HwU_fill(l+13,(log10(ptlb)),(WGTS))
      call HwU_fill(l+14,(etalb),(WGTS))
c
      call HwU_fill(l+15,(detallb),(WGTS))
      call HwU_fill(l+16,(azi),(WGTS))
      if(azinorm.gt.0.d0) call HwU_fill(l+17,(log10(azinorm)),(WGTS))
      call HwU_fill(l+18,(xmll),(WGTS))
      call HwU_fill(l+19,(ptpair),(WGTS))
      if(ptpair.gt.0)call HwU_fill(l+20,(log10(ptpair)),(WGTS))
      call HwU_fill(l+21,(0d0),(WGTS))
c
      l=l+21

      if(abs(etav).lt.ycut)then
        call HwU_fill(l+1,(ptv),(WGTS))
        call HwU_fill(l+2,(ptv),(WGTS))
        if(ptv.gt.0.d0)call HwU_fill(l+3,(log10(ptv)),(WGTS))
      endif
      if(ptv.gt.20.d0)then
        call HwU_fill(l+4,(yv),(WGTS))
        call HwU_fill(l+5,(etav),(WGTS))
      endif
      if(abs(etav).lt.ycut.and.ptv.gt.20.d0)then
         call HwU_fill(l+6,(xmv),(WGTS))
         call HwU_fill(l+21,(0d0),(WGTS))
      endif
c
      if(abs(etal).lt.ycut)then
        call HwU_fill(l+7,(ptl),(WGTS))
        call HwU_fill(l+8,(ptl),(WGTS))
        if(ptl.gt.0.d0)call HwU_fill(l+9,(log10(ptl)),(WGTS))
      endif
      if(ptl.gt.20.d0)call HwU_fill(l+10,(etal),(WGTS))
      if(abs(etalb).lt.ycut)then
        call HwU_fill(l+11,(ptlb),(WGTS))
        call HwU_fill(l+12,(ptlb),(WGTS))
        if(ptlb.gt.0.d0)call HwU_fill(l+13,(log10(ptlb)),(WGTS))
      endif
      if(ptlb.gt.20.d0)call HwU_fill(l+14,(etalb),(WGTS))
c
      if( abs(etal).lt.ycut.and.abs(etalb).lt.ycut .and.
     &     ptl.gt.20.d0.and.ptlb.gt.20.d0)then
        call HwU_fill(l+15,(detallb),(WGTS))
        call HwU_fill(l+16,(azi),(WGTS))
        if(azinorm.gt.0.d0) call HwU_fill(l+17,(log10(azinorm)),(WGTS))
        call HwU_fill(l+18,(xmll),(WGTS))
        call HwU_fill(l+19,(ptpair),(WGTS))
        if(ptpair.gt.0) call HwU_fill(l+20,(log10(ptpair)),(WGTS))
      endif

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
