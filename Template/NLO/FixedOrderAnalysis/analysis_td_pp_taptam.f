c
c Example analysis for "p p > ta+ ta- [QCD]" process.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(nwgt,weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      integer i,kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
      character*6 cc(2)
      data cc/'|T@NLO','|T@LO '/
      include 'dbook.inc'
      call inihist
      nwgt_analysis=nwgt
      if (nwgt_analysis*8.gt.nplots/4) then
         write (*,*) 'error in analysis_begin: '/
     &        /'too many histograms, increase NPLOTS to',
     &        nwgt_analysis*8*4
         stop 1
      endif
      do i=1,2
      do kk=1,nwgt_analysis
        l=(kk-1)*8+(i-1)*4
        call bookup(l+1,'total rate  '//cc(i)//weights_info(kk),
     &       1.0d0,0.5d0,5.5d0)
        call bookup(l+2,'ta+ta- mass '//cc(i)//weights_info(kk),
     &       5d0,0d0,200d0)
        call bookup(l+3,'ta+ta- rap  '//cc(i)//weights_info(kk),
     &       0.25d0,-5d0,5d0)
        call bookup(l+4,'ta+ta- pt   '//cc(i)//weights_info(kk),
     &       20d0,0d0,400d0)
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
         l=(kk-1)*8+(i-1)*4
         call multitop(l+1,3,2,'total rate  ',ytit,'LIN')
         call multitop(l+2,3,2,'ta+ ta- mass',ytit,'LOG')
         call multitop(l+3,3,2,'ta+ ta- rap ',ytit,'LOG')
         call multitop(l+4,3,2,'ta+ ta- pt  ',ytit,'LOG')
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
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      double precision wgts(*)
      integer ibody
      double precision wgt,var
      integer i,kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
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
         do kk=1,nwgt_analysis
            www=wgts(kk)
            l=(kk-1)*8+(i-1)*4
            if (ibody.ne.3 .and.i.eq.2) cycle
            call mfill(l+1,var,www)
            call mfill(l+2,xmh,www)
            call mfill(l+3,yh,www)
            call mfill(l+4,pth,www)
         enddo
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

