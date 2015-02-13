c
c Example analysis for "p p > w+ [QCD]" process.
c Example analysis for "p p > w- [QCD]" process.
c Example analysis for "p p > z [QCD]" process.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(nwgt,weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      integer j,kk,l
      character*6 cc(2)
      data cc/'|T@NLO','|T@LO '/
      real * 8 xmi,xms
      call HwU_inithist(nwgt,weights_info)
      xmi=40.d0
      xms=120.d0
      do j=1,2
         l=(j-1)*5
         call HwU_book(l+ 1,'V pt     '//cc(j),100,0.d0,200.d0)
         call HwU_book(l+ 2,'V log pt '//cc(j),100,0.d0,5.d0)
         call HwU_book(l+ 3,'V y      '//cc(j), 78,-9.d0,9.d0)
         call HwU_book(l+ 4,'V eta    '//cc(j), 78,-9.d0,9.d0)
         call HwU_book(l+ 5,'mV       '//cc(j), 80,xmi,xms)
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
      double precision www,pv(0:3),xmv,ptv,yv,etav
      double precision getrapidity,getpseudorap,dot
      external getrapidity,getpseudorap,dot
      if (nexternal.ne.4) then
         write (*,*) 'error #1 in analysis_fill: '/
     &        /'only for process "p p > V [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(1)).le.5 .or. ipdg(1).eq.21)) then
         write (*,*) 'error #2 in analysis_fill: '/
     &        /'only for process "p p > V [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(2)).le.5 .or. ipdg(2).eq.21)) then
         write (*,*) 'error #3 in analysis_fill: '/
     &        /'only for process "p p > V [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(4)).le.5 .or. ipdg(4).eq.21)) then
         write (*,*) 'error #4 in analysis_fill: '/
     &        /'only for process "p p > V [QCD]"'
         stop 1
      endif
      if (abs(ipdg(3)).ne.24.and.ipdg(3).ne.23) then
         write (*,*) 'error #5 in analysis_fill: '/
     &        /'only for process "p p > V [QCD]"'
         stop 1
      endif
C
      do i=0,3
        pv(i)=p(i,3)
      enddo
      xmv=sqrt(max(dot(pv,pv),0d0))
      ptv=sqrt(max(pv(1)**2+pv(2)**2,0d0))
      yv=getrapidity(pv(0),pv(3))
      etav=getpseudorap(pv(0),pv(1),pv(2),pv(3))
C
      do i=1,2
         l=(i-1)*5
         if (ibody.ne.3 .and.i.eq.2) cycle
         call HwU_fill(l+1,ptv,WGTS)
         if(ptv.gt.0) call HwU_fill(l+2,log10(ptv),WGTS)
         call HwU_fill(l+3,yv,WGTS)
         call HwU_fill(l+4,etav,WGTS)
         call HwU_fill(l+5,xmv,WGTS)
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
