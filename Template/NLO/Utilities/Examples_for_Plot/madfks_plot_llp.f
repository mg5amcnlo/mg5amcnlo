c
c
c Plotting routines
c
c
      subroutine initplot
      implicit none
c Book histograms in this routine. Use mbook or bookup. The entries
c of these routines are real*8
      include 'run.inc'
      real * 8 xm0,gam,xmlow,xmupp,bin
      real * 8 xmi,xms,pi
      PARAMETER (PI=3.14159265358979312D0)
      integer j,k,jpr
      character*5 cc(2)
      data cc/'     ',' cuts'/
c
      call inihist
      do j=1,2
      k=(j-1)*50
c
      xmi=50.d0
      xms=130.d0
      bin=0.8d0
      call bookup(k+ 1,'V pt'//cc(j),2.d0,0.d0,200.d0)
      call bookup(k+ 2,'V pt'//cc(j),10.d0,0.d0,1000.d0)
      call bookup(k+ 3,'V log[pt]'//cc(j),0.05d0,0.1d0,5.d0)
      call bookup(k+ 4,'V y'//cc(j),0.2d0,-9.d0,9.d0)
      call bookup(k+ 5,'V eta'//cc(j),0.2d0,-9.d0,9.d0)
      call bookup(k+ 6,'mV'//cc(j),(bin),xmi,xms)
c
      call bookup(k+ 7,'l pt'//cc(j),2.d0,0.d0,200.d0)
      call bookup(k+ 8,'l pt'//cc(j),10.d0,0.d0,1000.d0)
      call bookup(k+ 9,'l log[pt]'//cc(j),0.05d0,0.1d0,5.d0)
      call bookup(k+10,'l eta'//cc(j),0.2d0,-9.d0,9.d0)
      call bookup(k+11,'lb pt'//cc(j),2.d0,0.d0,200.d0)
      call bookup(k+12,'lb pt'//cc(j),10.d0,0.d0,1000.d0)
      call bookup(k+13,'lb log[pt]'//cc(j),0.05d0,0.1d0,5.d0)
      call bookup(k+14,'lb eta'//cc(j),0.2d0,-9.d0,9.d0)
c
      call bookup(k+15,'llb delta eta'//cc(j),0.2d0,-9.d0,9.d0)
      call bookup(k+16,'llb azimt'//cc(j),pi/20.d0,0.d0,pi)
      call bookup(k+17,'llb log[pi-azimt]'//cc(j),0.05d0,-4.d0,0.1d0)
      call bookup(k+18,'llb inv m'//cc(j),(bin),xmi,xms)
      call bookup(k+19,'llb pt'//cc(j),2.d0,0.d0,200.d0)
      call bookup(k+20,'llb log[pt]'//cc(j),0.05d0,0.1d0,5.d0)
c
      call bookup(k+21,'total'//cc(j),1.d0,-1.d0,1.d0)
      enddo
      return
      end


      subroutine topout
      implicit none
      character*14 ytit
      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint
      integer itmax,ncall
      common/citmax/itmax,ncall
      real*8 xnorm1,xnorm2
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      integer i,kk,j,k
      include 'dbook.inc'
c
      if (unwgt) then
         ytit='events per bin'
      else
         ytit='sigma per bin '
      endif
      xnorm1=1.d0/float(itmax)
      xnorm2=1.d0/float(ncall*itmax)
      do i=1,NPLOTS
        if(usexinteg.and..not.mint) then
           call mopera(i,'+',i,i,xnorm1,0.d0)
        elseif(mint) then
           call mopera(i,'+',i,i,xnorm2,0.d0)
        endif
        call mfinal(i)
      enddo

C
      do j=1,2
      k=(j-1)*50
      call multitop(k+ 1,3,2,'V pt',' ','LOG')
      call multitop(k+ 2,3,2,'V pt',' ','LOG')
      call multitop(k+ 3,3,2,'V log[pt]',' ','LOG')
      call multitop(k+ 4,3,2,'V y',' ','LOG')
      call multitop(k+ 5,3,2,'V eta',' ','LOG')
      call multitop(k+ 6,3,2,'mV',' ','LOG')
      enddo
c
      do j=1,2
      k=(j-1)*50
      call multitop(k+ 7,3,2,'l pt',' ','LOG')
      call multitop(k+ 8,3,2,'l pt',' ','LOG')
      call multitop(k+ 9,3,2,'l log[pt]',' ','LOG')
      call multitop(k+10,3,2,'l eta',' ','LOG')
      call multitop(k+11,3,2,'l pt',' ','LOG')
      call multitop(k+12,3,2,'l pt',' ','LOG')
      call multitop(k+13,3,2,'l log[pt]',' ','LOG')
      call multitop(k+14,3,2,'l eta',' ','LOG')
c
      call multitop(k+15,3,2,'llb deta',' ','LOG')
      call multitop(k+16,3,2,'llb azi',' ','LOG')
      call multitop(k+17,3,2,'llb azi',' ','LOG')
      call multitop(k+18,3,2,'llb inv m',' ','LOG')
      call multitop(k+19,3,2,'llb pt',' ','LOG')
      call multitop(k+20,3,2,'llb pt',' ','LOG')
c
      call multitop(k+21,3,2,'total',' ','LOG')
      enddo
c
      return                
      end


      subroutine outfun(pp,ybst_til_tolab,www,itype)
C
C *WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING*
C
C In MadFKS, the momenta PP given in input to this function are in the
C reduced parton c.m. frame. If need be, boost them to the lab frame.
C The rapidity of this boost is
C
C       YBST_TIL_TOLAB
C
C also given in input
C
C This is the rapidity that enters in the arguments of the sinh() and
C cosh() of the boost, in such a way that
C       ylab = ycm - ybst_til_tolab
C where ylab is the rapidity in the lab frame and ycm the rapidity
C in the center-of-momentum frame.
C
C *WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING*
      implicit none
      include 'nexternal.inc'
      real*8 pp(0:3,nexternal),ybst_til_tolab,www
      integer itype
      real*8 var
      real*8 ppevs(0:3,nexternal)

      real*8 getrapidity,getpseudorap,chybst,shybst,chybstmo
      real*8 xd(1:3)
      data (xd(i),i=1,3)/0,0,1/
      real*8 pplab(0:3,nexternal)
      double precision ppcl(4,nexternal),y(nexternal)
      double precision pjet(0:3,nexternal)
      double precision cthjet(nexternal)
      integer nn,njet,nsub,jet(nexternal)
      real*8 emax,getcth,cpar,dpar,thrust,dot,shat
      integer i,j,kk,imax

      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL  IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
c masses
      double precision pmass(nexternal)
      double precision azi,azinorm,bwcutoff,detallb,etal,etalb,etav,ptl,ptlb,
     &ptpair,ptv,wgamma,wmass,xmll,xmv,yl,ylb,yv,getinvm,getdelphi,pi,ppv(0:3),
     &ppl(0:3),pplb(0:3)
      PARAMETER (PI=3.14159265358979312D0)
      logical flag

      common/to_mass/pmass
      double precision ppkt(0:3,nexternal),ecut,djet
      double precision ycut, palg
      integer ipartjet(nexternal)
c
      kk=0
      if(itype.eq.11.or.itype.eq.12)then
        continue
      elseif(itype.eq.20)then
        return
      else
        write(*,*)'Error in outfun: unknown itype',itype
        stop
      endif

      chybst=cosh(ybst_til_tolab)
      shybst=sinh(ybst_til_tolab)
      chybstmo=chybst-1.d0
      do i=3,nexternal
        call boostwdir2(chybst,shybst,chybstmo,xd,
     #                  pp(0,i),pplab(0,i))
      enddo

      DO i=0,3
        ppl(i)=pplab(i,4)
        pplb(i)=pplab(i,3)
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
      azi=getdelphi(ppl(1),pplb(1),ppl(2),pplb(2))
      azinorm=(pi-azi)/pi
      xmll=xmv
      detallb=etal-etalb
c
      kk=0
      wmass=80.419d0
      wgamma=2.046d0
      bwcutoff=15.d0
      flag=(xmv.ge.wmass-wgamma*bwcutoff.and.
     &      xmv.le.wmass+wgamma*bwcutoff)
      if(flag)then
      call mfill(kk+1,(ptv),(WWW))
      call mfill(kk+2,(ptv),(WWW))
      if(ptv.gt.0.d0)call mfill(kk+3,(log10(ptv)),(WWW))
      call mfill(kk+4,(yv),(WWW))
      call mfill(kk+5,(etav),(WWW))
      call mfill(kk+6,(xmv),(WWW))
c
      call mfill(kk+7,(ptl),(WWW))
      call mfill(kk+8,(ptl),(WWW))
      if(ptl.gt.0.d0)call mfill(kk+9,(log10(ptl)),(WWW))
      call mfill(kk+10,(etal),(WWW))
      call mfill(kk+11,(ptlb),(WWW))
      call mfill(kk+12,(ptlb),(WWW))
      if(ptlb.gt.0.d0)call mfill(kk+13,(log10(ptlb)),(WWW))
      call mfill(kk+14,(etalb),(WWW))
c
      call mfill(kk+15,(detallb),(WWW))
      call mfill(kk+16,(azi),(WWW))
      if(azinorm.gt.0.d0)
     #  call mfill(kk+17,(log10(azinorm)),(WWW))
      call mfill(kk+18,(xmll),(WWW))
      call mfill(kk+19,(ptpair),(WWW))
      if(ptpair.gt.0)call mfill(kk+20,(log10(ptpair)),(WWW))
      call mfill(kk+21,(0d0),(WWW))
c
      kk=50
      if(abs(etav).lt.ycut)then
        call mfill(kk+1,(ptv),(WWW))
        call mfill(kk+2,(ptv),(WWW))
        if(ptv.gt.0.d0)call mfill(kk+3,(log10(ptv)),(WWW))
      endif
      if(ptv.gt.20.d0)then
        call mfill(kk+4,(yv),(WWW))
        call mfill(kk+5,(etav),(WWW))
      endif
      if(abs(etav).lt.ycut.and.ptv.gt.20.d0)then
         call mfill(kk+6,(xmv),(WWW))
         call mfill(kk+21,(0d0),(WWW))
      endif
c
      if(abs(etal).lt.ycut)then
        call mfill(kk+7,(ptl),(WWW))
        call mfill(kk+8,(ptl),(WWW))
        if(ptl.gt.0.d0)call mfill(kk+9,(log10(ptl)),(WWW))
      endif
      if(ptl.gt.20.d0)call mfill(kk+10,(etal),(WWW))
      if(abs(etalb).lt.ycut)then
        call mfill(kk+11,(ptlb),(WWW))
        call mfill(kk+12,(ptlb),(WWW))
        if(ptlb.gt.0.d0)call mfill(kk+13,(log10(ptlb)),(WWW))
      endif
      if(ptlb.gt.20.d0)call mfill(kk+14,(etalb),(WWW))
c
      if( abs(etal).lt.ycut.and.abs(etalb).lt.ycut .and.
     #    ptl.gt.20.d0.and.ptlb.gt.20.d0)then
        call mfill(kk+15,(detallb),(WWW))
        call mfill(kk+16,(azi),(WWW))
        if(azinorm.gt.0.d0)
     #    call mfill(kk+17,(log10(azinorm)),(WWW))
        call mfill(kk+18,(xmll),(WWW))
        call mfill(kk+19,(ptpair),(WWW))
        if(ptpair.gt.0) 
     #    call mfill(kk+20,(log10(ptpair)),(WWW))
      endif
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
