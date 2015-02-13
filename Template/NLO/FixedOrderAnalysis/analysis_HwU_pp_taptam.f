c
c Example analysis for "p p > ta+ ta- [QCD]" process.
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
         l=(i-1)*4
         call HwU_book(l+1,'total rate  '//cc(i), 5,0.5d0,5.5d0)
         call HwU_book(l+2,'ta+ta- mass '//cc(i), 40,0d0,200d0)
         call HwU_book(l+3,'ta+ta- rap  '//cc(i), 40,-5d0,5d0)
         call HwU_book(l+4,'ta+ta- pt   '//cc(i), 20,0d0,400d0)
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
      double precision ph(0:3),yh,xmh,pth,www
      double precision getrapidity,dot
      external getrapidity,dot
      if (nexternal.ne.5) then
         write (*,*) 'error #1 in analysis_fill: '/
     &        /'only for process "p p > ta+ ta- [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(1)).le.5 .or. ipdg(1).eq.21)) then
         write (*,*) 'error #2 in analysis_fill: '/
     &        /'only for process "p p > ta+ ta- [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(2)).le.5 .or. ipdg(2).eq.21)) then
         write (*,*) 'error #3 in analysis_fill: '/
     &        /'only for process "p p > ta+ ta- [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(5)).le.5 .or. ipdg(5).eq.21)) then
         write (*,*) 'error #4 in analysis_fill: '/
     &        /'only for process "p p > ta+ ta- [QCD]"'
         stop 1
      endif
      if (ipdg(3).ne.-15) then
         write (*,*) 'error #5 in analysis_fill: '/
     &        /'only for process "p p > ta+ ta- [QCD]"'
         stop 1
      endif
      if (ipdg(4).ne.15) then
         write (*,*) 'error #6 in analysis_fill: '/
     &        /'only for process "p p > ta+ ta- [QCD]"'
         stop 1
      endif
      do i=0,3
        ph(i)=p(i,3)+p(i,4)
      enddo
      xmh = sqrt(max(dot(ph,ph),0d0))
      yh  = getrapidity(ph(0),ph(3))
      pth = sqrt(max(ph(1)**2+ph(2)**2,0d0))
      var = 1.d0
      do i=1,2
         l=(i-1)*4
         if (ibody.ne.3 .and.i.eq.2) cycle
         call HwU_fill(l+1,var,wgts)
         call HwU_fill(l+2,xmh,wgts)
         call HwU_fill(l+3,yh,wgts)
         call HwU_fill(l+4,pth,wgts)
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

