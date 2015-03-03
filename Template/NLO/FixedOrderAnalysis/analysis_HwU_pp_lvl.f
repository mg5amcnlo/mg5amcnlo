c
c Example analysis for "p p > e+ ve [QCD]" process.
c Example analysis for "p p > e- ve~ [QCD]" process.
c Example analysis for "p p > mu+ vm [QCD]" process.
c Example analysis for "p p > mu- vm~ [QCD]" process.
c Example analysis for "p p > ta+ vt [QCD]" process.
c Example analysis for "p p > ta- vt~ [QCD]" process.
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
        call HwU_book(l+1,'total rate '//cc(i), 5,0.5d0,5.5d0)
        call HwU_book(l+2,'lep rapidity '//cc(i), 20,-5d0,5d0)
        call HwU_book(l+3,'lep pt '//cc(i), 20,0d0,200d0)
        call HwU_book(l+4,'et miss '//cc(i), 20,0d0,200d0)
        call HwU_book(l+5,'trans. mass '//cc(i), 40,0d0,200d0)
        call HwU_book(l+6,'w rapidity '//cc(i), 20,-5d0,5d0)
        call HwU_book(l+7,'w pt '//cc(i), 20,0d0,200d0)
        call HwU_book(l+8,'cphi[l,vl] '//cc(i), 40,-1d0,1d0)
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
      double precision pw(0:3),pe(0:3),pn(0:3),ye,yw,pte,etmiss,mtr,ptw,cphi,www
      double precision getrapidity
      external getrapidity
      if (nexternal.ne.5) then
         write (*,*) 'error #1 in analysis_fill: '/
     &        /'only for process "p p > l vl [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(1)).le.4 .or. ipdg(1).eq.21)) then
         write (*,*) 'error #2 in analysis_fill: '/
     &        /'only for process "p p > l vl [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(2)).le.4 .or. ipdg(2).eq.21)) then
         write (*,*) 'error #3 in analysis_fill: '/
     &        /'only for process "p p > l vl [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(5)).le.4 .or. ipdg(5).eq.21)) then
         write (*,*) 'error #4 in analysis_fill: '/
     &        /'only for process "p p > l vl [QCD]"'
         stop 1
      endif
      if( (abs(abs(ipdg(3))-abs(ipdg(4))).ne.1) .or.
     &    (sign(1d0,dble(ipdg(3))).eq.sign(1d0,dble(ipdg(4)))) .or.
     &    (abs(ipdg(3)).le.10.or.abs(ipdg(3)).ge.16) .or.
     &    (abs(ipdg(4)).le.10.or.abs(ipdg(4)).ge.16) )then
         write(*,*)'analysis not suited for this process',ipdg(3),ipdg(4)
      endif
      do i=0,3
         if(abs(ipdg(3)).eq.11.or.abs(ipdg(3)).eq.13.or.abs(ipdg(3)).eq.15)then
            pe(i)=p(i,3)
            pn(i)=p(i,4)
         else
            pe(i)=p(i,4)
            pn(i)=p(i,3)
         endif
        pw(i)=pe(i)+pn(i)
      enddo
      ye     = getrapidity(pe(0), pe(3))
      yw     = getrapidity(pw(0), pw(3))
      pte    = dsqrt(pe(1)**2 + pe(2)**2)
      ptw    = dsqrt(pw(1)**2 + pw(2)**2)
      etmiss = dsqrt(pn(1)**2 + pn(2)**2)
      mtr    = dsqrt(2d0*pte*etmiss-2d0*pe(1)*pn(1)-2d0*pe(2)*pn(2))
      cphi   = (pe(1)*pn(1)+pe(2)*pn(2))/pte/etmiss
      var    = 1.d0
      do i=1,2
         l=(i-1)*8
         if (ibody.ne.3 .and.i.eq.2) cycle
         call HwU_fill(l+1,var,wgts)
         call HwU_fill(l+2,ye,wgts)
         call HwU_fill(l+3,pte,wgts)
         call HwU_fill(l+4,etmiss,wgts)
         call HwU_fill(l+5,mtr,wgts)
         call HwU_fill(l+6,yw,wgts)
         call HwU_fill(l+7,ptw,wgts)
         call HwU_fill(l+8,cphi,wgts)
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
