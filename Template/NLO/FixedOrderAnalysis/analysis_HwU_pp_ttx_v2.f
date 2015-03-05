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
      character*6 cc(2)
      data cc/'|T@NLO','|T@LO '/
      call HwU_inithist(nwgt,weights_info)
      do i=1,2
         l=(i-1)*8
         call HwU_book(l+ 1,'total rate    '//cc(i),  5,0.5d0,5.5d0)
         call HwU_book(l+ 2,'t rap         '//cc(i), 50,-5d0,5d0)
         call HwU_book(l+ 3,'tx rap        '//cc(i), 50,-5d0,5d0)
         call HwU_book(l+ 4,'t-tx pair rap '//cc(i), 60,-3d0,3d0)
         call HwU_book(l+ 5,'m t-tx        '//cc(i),100,0d0,1000d0)
         call HwU_book(l+ 6,'pt t          '//cc(i),100,0d0,400d0)
         call HwU_book(l+ 7,'pt tx         '//cc(i),100,0d0,400d0)
         call HwU_book(l+ 8,'pt t-tx       '//cc(i),100,0d0,200d0)
      enddo
      return
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_end(dummy)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      character*14 ytit
      double precision dummy
      integer i
      integer kk,l
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
      double precision pttx(0:3),www,mtt,pt_t,pt_tx,pt_ttx,yt,ytx,yttx
      double precision getrapidity,dot
      external getrapidity,dot
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
      do i=0,3
        pttx(i)=p(i,3)+p(i,4)
      enddo
      mtt    = dsqrt(dot(pttx, pttx))
      pt_t   = dsqrt(p(1,3)**2 + p(2,3)**2)
      pt_tx  = dsqrt(p(1,4)**2 + p(2,4)**2)
      pt_ttx = dsqrt((p(1,3)+p(1,4))**2 + (p(2,3)+p(2,4))**2)
      yt  = getrapidity(p(0,3), p(3,3))
      ytx = getrapidity(p(0,4), p(3,4))
      yttx= getrapidity(pttx(0), pttx(3))
      var=1.d0
      do i=1,2
         l=(i-1)*8
         if (ibody.ne.3 .and.i.eq.2) cycle
         call HwU_fill(l+1,var,wgts)
         call HwU_fill(l+2,yt,wgts)
         call HwU_fill(l+3,ytx,wgts)
         call HwU_fill(l+4,yttx,wgts)
         call HwU_fill(l+5,mtt,wgts)
         call HwU_fill(l+6,pt_t,wgts)
         call HwU_fill(l+7,pt_tx,wgts)
         call HwU_fill(l+8,pt_ttx,wgts)
      enddo
c
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
