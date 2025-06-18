cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(nwgt,weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      integer i,l
      character*6 cc(2)
      double precision pi
      PARAMETER (PI=3.14159265358979312D0)
      data cc/'|T@NLO', '|T@LO '/
      call HwU_inithist(nwgt,weights_info)
       do i=1,2
         l=(i-1)*11
         call HwU_book(l+ 1,'total rate       '//cc(i),  5,0.5d0,5.5d0)
         call HwU_book(l+ 2,'b rap           '//cc(i), 40,-10d0,10d0)
         call HwU_book(l+ 3,'bx rap          '//cc(i), 100,-10d0,10d0)
         call HwU_book(l+ 11,'z              '//cc(i), 300,-3d0,3d0)
         call HwU_book(l+ 5,'m b-bx          '//cc(i),40,0d0,500d0)
         call HwU_book(l+ 6,'pt b            '//cc(i),12,0d0,60d0)
         call HwU_book(l+ 7,'pt bx           '//cc(i),12,1d0,61d0)
         call HwU_book(l+ 8,'avg p_t(b)       '//cc(i),12,0.5d0,60.5d0)
         call HwU_book(l+ 9,'ETA bx         '//cc(i),40,-10d0,10d0)
         call HwU_book(l+ 10,'ETA b         '//cc(i),100,-10d0,10d0)
         call HwU_book(l+ 4,'DELTAPHI       '//cc(i),20,-pi,+pi)
      enddo
      return
      end
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_end(dummy)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
c      character*14 ytit
      double precision dummy
c      integer i
c      integer kk,l
      call HwU_write_file
      return                
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      double precision wgts(*)
      integer ibody
      double precision wgt,var,W,z,P_Qp,P_ap
      integer i,kk,l
      double precision pbbx(0:3),www,mbb,pt_b,pt_bx,pt_bbx,yb,ybx,ybbx,ETA_b,ETA_bx,DELTAPHI
      double precision getrapidity,getpseudorap,getinvm,getdelphi
      external getrapidity,getpseudorap,getinvm,getdelphi
      if (nexternal.ne.5) then
         write (*,*) 'error #1 in analysis_fill: '/
     &        /'only for process "a p > b b~ [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(1)).eq.22)) then
         write (*,*) 'error #2 in analysis_fill: '/
     &        /'only for process "a p > b b~ [QCD]"'
         stop 1
      endif
      if (.not. (abs(ipdg(2)).eq.21.or.abs(ipdg(2)).le.4)) then
         write (*,*) 'error #3 in analysis_fill: '/
     &        /'only for process "a p > b b~ [QCD]"'
         stop 1
      endif
      if (.not. (ipdg(5).eq.22 .or. (abs(ipdg(5)).le. 4 .or. ipdg(5).eq.21))) then
         write (*,*) 'error #4 in analysis_fill: '/
     &        /'only for process "a p > b b~ [QCD]"'
         stop 1
      endif
      if (ipdg(3).ne. 5) then
         write (*,*) 'error #5 in analysis_fill: '/
     &        /'only for process "a p > b b~ [QCD]"'
         stop 1
      endif
      if (ipdg(4).ne.-5) then
         write (*,*) 'error #6 in analysis_fill: '/
     &        /'only for process "a p > b b~ [QCD]"'
         stop 1
      endif
      do i=0,3
        pbbx(i)=p(i,3)+p(i,4)
      enddo
      mbb    = getinvm(pbbx(0),pbbx(1),pbbx(2),pbbx(3)) ! invariant mass of bb~
      pt_b   = dsqrt(p(1,3)**2 + p(2,3)**2)
      pt_bx  = dsqrt(p(1,4)**2 + p(2,4)**2)
      pt_bbx = dsqrt((pt_b**2 + pt_bx**2)/2)
      yb  = getrapidity(p(0,3), p(3,3))
      ybx = getrapidity(p(0,4), p(3,4))
      ybbx= getrapidity(pbbx(0), pbbx(3))
      ETA_b= getpseudorap(p(0,3),p(1,3),p(2,3),p(3,3))
      ETA_bx= getpseudorap(p(0,4),p(1,4),p(2,4),p(3,4))
      DELTAPHI = getdelphi(p(1,3),p(2,3),p(1,4),p(2,4))
      W = dsqrt(4*p(0,1)*ebeam(2))
      P_Qp = p(0,3)*p(0,2) - p(1,3)*p(1,2)- p(2,3)*p(2,2) - p(3,3)*p(3,2)
      P_ap = p(0,1)*p(0,2) - p(1,1)*p(1,2)- p(2,1)*p(2,2) - p(3,1)*p(3,2)
      z = P_Qp/P_ap
      var=1.d0
      do i=1,2
         l=(i-1)*11
c         if (W .lt. 16 ) cycle   !HERA cuts
c         if (W .gt. 207) cycle
c         if(abs(ETA_b).ge.2) cycle
c         if(abs(ETA_bx).ge.2) cycle
c         if(abs(pt_b).le.1) cycle
         if (ibody.ne.3 .and.i.eq.2) cycle
         call HwU_fill(l+1,var,wgts)
         call HwU_fill(l+2,yb,wgts)
         call HwU_fill(l+3,ybx,wgts)
         call HwU_fill(l+11,z,wgts)
         call HwU_fill(l+5,mbb,wgts)
         call HwU_fill(l+6,pt_b,wgts)
         call HwU_fill(l+7,pt_bx,wgts)
         call HwU_fill(l+8,pt_bbx,wgts)
         call HwU_fill(l+9,ETA_bx,wgts)
         call HwU_fill(l+10,ETA_b,wgts)
         call HwU_fill(l+4,DELTAPHI,wgts)
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
      
      
      function getpseudorap(en,ptx,pty,pl)
      implicit none
      real*8 getpseudorap,en,ptx,pty,pl,tiny,pt,eta,th
      parameter (tiny=1.d-5)
c
      pt=sqrt(ptx**2+pty**2)
      if(pt.lt.tiny.and.abs(pl).lt.tiny)then
        eta=sign(1.d0,pl)*1.d8
      else
c       th=atan2(pt,pl)
       th=dacos(pl/dsqrt(ptx**2+pty**2+pl**2))
       eta=-log(tan(th/2.d0))
      endif
      getpseudorap=eta
      return
      end
c      Calculation of Invariant masss      
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
c     Deltaphi calculation      
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
