c
c Example analysis for "p p > t t~ [QCD]" process.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(nwgt,weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      integer i,kk,l
      character*5 cc(2)
      data cc/'     ','cuts '/
      double precision pi
      PARAMETER (PI=3.14159265358979312D0)
      call HwU_inithist(nwgt,weights_info)
      do i=1,2
        l=(i-1)*20
        call HwU_book(l+ 1,'tt pt            '//cc(i),50,0.d0,100.d0)
        call HwU_book(l+ 2,'tt log[pt]       '//cc(i),98,0.1d0,5.d0)
        call HwU_book(l+ 3,'tt inv m         '//cc(i),60,300.d0,1000.d0)
        call HwU_book(l+ 4,'tt azimt         '//cc(i),20,0.d0,pi)
        call HwU_book(l+ 5,'tt del R         '//cc(i),60,0.d0,3*pi)
        call HwU_book(l+ 6,'tb pt            '//cc(i),100,0.d0,500.d0)
        call HwU_book(l+ 7,'tb log[pt]       '//cc(i),98,0.1d0,5.d0)
        call HwU_book(l+ 8,'t pt             '//cc(i),100,0.d0,500.d0)
        call HwU_book(l+ 9,'t log[pt]        '//cc(i),98,0.1d0,5.d0)
        call HwU_book(l+10,'tt delta eta     '//cc(i),40,-4.d0,4.d0)
        call HwU_book(l+11,'y_tt             '//cc(i),80,-4.d0,4.d0)
        call HwU_book(l+12,'delta y          '//cc(i),40,-4.d0,4.d0)
        call HwU_book(l+13,'tt azimt zoomin  '//cc(i),20,2*pi/3,pi)
        call HwU_book(l+14,'tt del R zoomin  '//cc(i),40,2*pi/3,4*pi/3)
        call HwU_book(l+15,'y_tb             '//cc(i),80,-4.d0,4.d0)
        call HwU_book(l+16,'y_t              '//cc(i),80,-4.d0,4.d0)
        call HwU_book(l+17,'tt log[pi-azimt] '//cc(i),82,-4.d0,0.1d0)
        call HwU_book(l+18,'tt pt zoomout    '//cc(i),96,80.d0,2000.d0)
       call HwU_book(l+19,'tb pt zoomout    '//cc(i),100,400.d0,2400.d0)
       call HwU_book(l+20,'t pt  zoomout    '//cc(i),100,400.d0,2400.d0)
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
      integer i,kk,l
      double precision www,xptq(0:3),xptb(0:3),yptqtb(0:3),ycut,ptcut
     $     ,ptq1,ptq2,ptg,yq1,yq2,etaq1,etaq2,azi,azinorm,qqm,dr,yqq
      logical siq1flag,siq2flag,ddflag
      double precision getrapidity,getpseudorap,getinvm,getdelphi
      external getrapidity,getpseudorap,getinvm,getdelphi
      double precision pi
      PARAMETER (PI=3.14159265358979312D0)
      if (nexternal.ne.5) then
         write (*,*) 'error #1 in analysis_fill: '/
     &        /'only for process "p p > t t~ [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(1)).le.5 .or. ipdg(1).eq.21)) then
         write (*,*) 'error #2 in analysis_fill: '/
     &        /'only for process "p p > t t~ [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(2)).le.5 .or. ipdg(2).eq.21)) then
         write (*,*) 'error #3 in analysis_fill: '/
     &        /'only for process "p p > t t~ [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(5)).le.5 .or. ipdg(5).eq.21)) then
         write (*,*) 'error #4 in analysis_fill: '/
     &        /'only for process "p p > t t~ [QCD]"'
         stop 1
      endif
      if (ipdg(3).ne.6) then
         write (*,*) 'error #5 in analysis_fill: '/
     &        /'only for process "p p > t t~ [QCD]"'
         stop 1
      endif
      if (ipdg(4).ne.-6) then
         write (*,*) 'error #6 in analysis_fill: '/
     &        /'only for process "p p > t t~ [QCD]"'
         stop 1
      endif
c
      do i=0,3
        xptq(i)=p(i,3)
        xptb(i)=p(i,4)
        yptqtb(i)=xptq(i)+xptb(i)
      enddo
C FILL THE HISTOS
      YCUT=2.5D0
      PTCUT=30.D0
C
      ptq1 = dsqrt(xptq(1)**2+xptq(2)**2)
      ptq2 = dsqrt(xptb(1)**2+xptb(2)**2)
      ptg = dsqrt(yptqtb(1)**2+yptqtb(2)**2)
      yq1=getrapidity(xptq(0),xptq(3))
      yq2=getrapidity(xptb(0),xptb(3))
      etaq1=getpseudorap(xptq(0),xptq(1),xptq(2),xptq(3))
      etaq2=getpseudorap(xptb(0),xptb(1),xptb(2),xptb(3))
      azi=getdelphi(xptq(1),xptq(2),xptb(1),xptb(2))
      azinorm = (pi-azi)/pi
      qqm=getinvm(yptqtb(0),yptqtb(1),yptqtb(2),yptqtb(3))
      dr  = dsqrt(azi**2+(etaq1-etaq2)**2)
      yqq=getrapidity(yptqtb(0),yptqtb(3))
c-------------------------------------------------------------
      siq1flag=ptq1.gt.ptcut.and.abs(yq1).lt.ycut
      siq2flag=ptq2.gt.ptcut.and.abs(yq2).lt.ycut
      ddflag=siq1flag.and.siq2flag
c-------------------------------------------------------------
      l=0
      call HwU_fill(l+1,ptg,WGTS)
      call HwU_fill(l+18,ptg,WGTS)
      if(ptg.gt.0) call HwU_fill(l+2,log10(ptg),WGTS)
      call HwU_fill(l+3,qqm,WGTS)
      call HwU_fill(l+4,azi,WGTS)
      call HwU_fill(l+13,azi,WGTS)
      if(azinorm.gt.0)call HwU_fill(l+17,log10(azinorm),WGTS)
      call HwU_fill(l+5,dr,WGTS)
      call HwU_fill(l+14,dr,WGTS)
      call HwU_fill(l+10,etaq1-etaq2,WGTS)
      call HwU_fill(l+11,yqq,WGTS)
      call HwU_fill(l+12,yq1-yq2,WGTS)
      call HwU_fill(l+6,ptq2,WGTS)
      call HwU_fill(l+19,ptq2,WGTS)
      if(ptq2.gt.0) call HwU_fill(l+7,log10(ptq2),WGTS)
      call HwU_fill(l+15,yq2,WGTS)
      call HwU_fill(l+8,ptq1,WGTS)
      call HwU_fill(l+20,ptq1,WGTS)
      if(ptq1.gt.0) call HwU_fill(l+9,log10(ptq1),WGTS)
      call HwU_fill(l+16,yq1,WGTS)
c
c***************************************************** with cuts
c
      l=l+20
c
      if(ddflag)then
         call HwU_fill(l+1,ptg,WGTS)
         call HwU_fill(l+18,ptg,WGTS)
         if(ptg.gt.0) call HwU_fill(l+2,log10(ptg),WGTS)
         call HwU_fill(l+3,qqm,WGTS)
         call HwU_fill(l+4,azi,WGTS)
         call HwU_fill(l+13,azi,WGTS)
         if(azinorm.gt.0) call HwU_fill(l+17,log10(azinorm),WGTS)
         call HwU_fill(l+5,dr,WGTS)
         call HwU_fill(l+14,dr,WGTS)
         call HwU_fill(l+10,etaq1-etaq2,WGTS)
         call HwU_fill(l+11,yqq,WGTS)
         call HwU_fill(l+12,yq1-yq2,WGTS)
      endif
      if(abs(yq2).lt.ycut)then
         call HwU_fill(l+6,ptq2,WGTS)
         call HwU_fill(l+19,ptq2,WGTS)
         if(ptq2.gt.0) call HwU_fill(l+7,log10(ptq2),WGTS)
      endif
      if(ptq2.gt.ptcut)call HwU_fill(l+15,yq2,WGTS)
      if(abs(yq1).lt.ycut)then
         call HwU_fill(l+8,ptq1,WGTS)
         call HwU_fill(l+20,ptq1,WGTS)
         if(ptq1.gt.0) call HwU_fill(l+9,log10(ptq1),WGTS)
      endif
      if(ptq1.gt.ptcut)call HwU_fill(l+16,yq1,WGTS)
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
