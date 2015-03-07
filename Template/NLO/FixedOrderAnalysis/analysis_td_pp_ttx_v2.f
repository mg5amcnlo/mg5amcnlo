c
c Example analysis for "p p > t t~ [QCD]" process.
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
      if (nwgt_analysis*16.gt.nplots/4) then
         write (*,*) 'error in analysis_begin: '/
     &        /'too many histograms, increase NPLOTS to',
     &        nwgt_analysis*16*4
         stop 1
      endif
      do i=1,2
      do kk=1,nwgt_analysis
        l=(kk-1)*16+(i-1)*8
        call bookup(l+ 1,'total rate    '//cc(i)//weights_info(kk),
     &       1.0d0,0.5d0,5.5d0)
        call bookup(l+ 2,'t rap         '//cc(i)//weights_info(kk),
     &       0.2d0,-5d0,5d0)
        call bookup(l+ 3,'tx rap        '//cc(i)//weights_info(kk),
     &       0.2d0,-5d0,5d0)
        call bookup(l+ 4,'t-tx pair rap '//cc(i)//weights_info(kk),
     &       0.1d0,-3d0,3d0)
        call bookup(l+ 5,'m t-tx        '//cc(i)//weights_info(kk),
     &       10d0,0d0,1000d0)
        call bookup(l+ 6,'pt t          '//cc(i)//weights_info(kk),
     &       4d0,0d0,400d0)
        call bookup(l+ 7,'pt tx         '//cc(i)//weights_info(kk),
     &       4d0,0d0,400d0)
        call bookup(l+ 8,'pt t-tx       '//cc(i)//weights_info(kk),
     &       2d0,0d0,200d0)
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
         l=(kk-1)*16+(i-1)*8
        call multitop(l+ 1,3,2,'total rate   ',ytit,'LIN')
        call multitop(l+ 2,3,2,'t rap        ',ytit,'LOG')
        call multitop(l+ 3,3,2,'tx rap       ',ytit,'LOG')
        call multitop(l+ 4,3,2,'t-tx pair rap',ytit,'LOG')
        call multitop(l+ 5,3,2,'m t-tx       ',ytit,'LOG')
        call multitop(l+ 6,3,2,'pt t         ',ytit,'LOG')
        call multitop(l+ 7,3,2,'pt tx        ',ytit,'LOG')
        call multitop(l+ 8,3,2,'pt t-tx      ',ytit,'LOG')
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
         do kk=1,nwgt_analysis
         www=wgts(kk)
            l=(kk-1)*16+(i-1)*8
            if (ibody.ne.3 .and.i.eq.2) cycle
            call mfill(l+1,var,www)
            call mfill(l+2,yt,www)
            call mfill(l+3,ytx,www)
            call mfill(l+4,yttx,www)
            call mfill(l+5,mtt,www)
            call mfill(l+6,pt_t,www)
            call mfill(l+7,pt_tx,www)
            call mfill(l+8,pt_ttx,www)
         enddo
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
