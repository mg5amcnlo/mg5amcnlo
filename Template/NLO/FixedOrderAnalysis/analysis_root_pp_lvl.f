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
      integer i,kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
      character*6 cc(2)
      data cc/'|T@NLO','|T@LO '/
      call open_root_file()
      nwgt_analysis=nwgt
      do i=1,2
      do kk=1,nwgt_analysis
        l=(kk-1)*16+(i-1)*8
        call rbook(l+1,'total rate '//cc(i)//weights_info(kk),
     &       1.0d0,0.5d0,5.5d0)
        call rbook(l+2,'lep rapidity '//cc(i)//weights_info(kk),
     &       0.5d0,-5d0,5d0)
        call rbook(l+3,'lep pt '//cc(i)//weights_info(kk),
     &       10d0,0d0,200d0)
        call rbook(l+4,'et miss '//cc(i)//weights_info(kk),
     &       10d0,0d0,200d0)
        call rbook(l+5,'trans. mass '//cc(i)//weights_info(kk),
     &       5d0,0d0,200d0)
        call rbook(l+6,'w rapidity '//cc(i)//weights_info(kk),
     &       0.5d0,-5d0,5d0)
        call rbook(l+7,'w pt '//cc(i)//weights_info(kk),
     &       10d00,0d0,200d0)
        call rbook(l+8,'cphi[l,vl] '//cc(i)//weights_info(kk),
     &       0.05d0,-1d0,1d0)
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
      do i=1,2
      do kk=1,nwgt_analysis
        l=(kk-1)*16+(i-1)*8
        do jj=1,8
          call ropera(l+jj,'+',l+jj,l+jj,xnorm,0.d0)
        enddo
      enddo
      enddo
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
         do kk=1,nwgt_analysis
            www=wgts(kk)
            l=(kk-1)*16+(i-1)*8
            if (ibody.ne.3 .and.i.eq.2) cycle
            call rfill(l+1,var,www)
            call rfill(l+2,ye,www)
            call rfill(l+3,pte,www)
            call rfill(l+4,etmiss,www)
            call rfill(l+5,mtr,www)
            call rfill(l+6,yw,www)
            call rfill(l+7,ptw,www)
            call rfill(l+8,cphi,www)
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
