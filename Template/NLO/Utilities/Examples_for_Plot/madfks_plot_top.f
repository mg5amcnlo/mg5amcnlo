c
c
c Plotting routines
c
c
      subroutine initplot
      implicit none
c Book histograms in this routine. Use mbook or bookup. The entries
c of these routines are real*8
      double precision emax,ebin,etamin,etamax,etabin
      integer i,kk,j,k
      include 'run.inc'
      double precision pi
      PARAMETER (PI=3.14159265358979312D0)
      character*5 cc(2)
      data cc/'     ',' cuts'/
c
      call inihist
      do j=1,2
        k=(j-1)*50
        call bookup(k+ 1,'tt pt'//cc(j),2.d0,0.d0,100.d0)
        call bookup(k+ 2,'tt log[pt]'//cc(j),0.05d0,0.1d0,5.d0)
        call bookup(k+ 3,'tt inv m'//cc(j),10.d0,300.d0,1000.d0)
        call bookup(k+ 4,'tt azimt'//cc(j),pi/20.d0,0.d0,pi)
        call bookup(k+ 5,'tt del R'//cc(j),pi/20.d0,0.d0,3*pi)
        call bookup(k+ 6,'tb pt'//cc(j),5.d0,0.d0,500.d0)
        call bookup(k+ 7,'tb log[pt]'//cc(j),0.05d0,0.1d0,5.d0)
        call bookup(k+ 8,'t pt'//cc(j),5.d0,0.d0,500.d0)
        call bookup(k+ 9,'t log[pt]'//cc(j),0.05d0,0.1d0,5.d0)
        call bookup(k+10,'tt delta eta'//cc(j),0.2d0,-4.d0,4.d0)
        call bookup(k+11,'y_tt'//cc(j),0.1d0,-4.d0,4.d0)
        call bookup(k+12,'delta y'//cc(j),0.2d0,-4.d0,4.d0)
        call bookup(k+13,'tt azimt'//cc(j),pi/60.d0,2*pi/3,pi)
        call bookup(k+14,'tt del R'//cc(j),pi/60.d0,2*pi/3,4*pi/3)
        call bookup(k+15,'y_tb'//cc(j),0.1d0,-4.d0,4.d0)
        call bookup(k+16,'y_t'//cc(j),0.1d0,-4.d0,4.d0)
        call bookup(k+17,'tt log[pi-azimt]'//cc(j),0.05d0,-4.d0,0.1d0)
      enddo
      do j=1,2
        k=(j-1)*50
        call bookup(k+18,'tt pt'//cc(j),20.d0,80.d0,2000.d0)
        call bookup(k+19,'tb pt'//cc(j),20.d0,400.d0,2400.d0)
        call bookup(k+20,'t pt'//cc(j),20.d0,400.d0,2400.d0)
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
c
      if (unwgt) then
         ytit='events per bin'
      else
         ytit='sigma per bin '
      endif
      xnorm1=1.d0/float(itmax)
      xnorm2=1.d0/float(ncall*itmax)
      do i=1,500
        if(usexinteg.and..not.mint) then
           call mopera(i,'+',i,i,xnorm1,0.d0)
        elseif(mint) then
           call mopera(i,'+',i,i,xnorm2,0.d0)
        endif
        call mfinal(i)
      enddo

      do j=1,2
        k=(j-1)*50
        call multitop(k+ 1,2,3,'tt pt',' ','LOG')
        call multitop(k+ 2,2,3,'tt log[pt]',' ','LOG')
        call multitop(k+ 3,2,3,'tt inv m',' ','LOG')
        call multitop(k+ 4,2,3,'tt azimt',' ','LOG')
        call multitop(k+ 5,2,3,'tt del R',' ','LOG')
        call multitop(k+ 6,2,3,'tb pt',' ','LOG')
        call multitop(k+ 7,2,3,'tb log[pt]',' ','LOG')
        call multitop(k+ 8,2,3,'t pt',' ','LOG')
        call multitop(k+ 9,2,3,'t log[pt]',' ','LOG')
        call multitop(k+10,2,3,'tt Delta eta',' ','LOG')
        call multitop(k+11,2,3,'y_tt',' ','LOG')
        call multitop(k+12,2,3,'tt Delta y',' ','LOG')
        call multitop(k+13,2,3,'tt azimt',' ','LOG')
        call multitop(k+14,2,3,'tt del R',' ','LOG')
        call multitop(k+15,2,3,'tb y',' ','LOG')
        call multitop(k+16,2,3,'t y',' ','LOG')
        call multitop(k+17,2,3,'tt log[pi-azimt]',' ','LOG')
      enddo
      do j=1,2
        k=(j-1)*50
        call multitop(k+18,2,3,'tt pt',' ','LOG')
        call multitop(k+19,2,3,'tb pt',' ','LOG')
        call multitop(k+20,2,3,'t pt',' ','LOG')
      enddo
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
      double precision djet,ecut,ycut
      double precision ppcl(4,nexternal),y(nexternal)
      double precision pjet(4,nexternal)
      double precision cthjet(nexternal)
      integer nn,njet,nsub,jet(nexternal)
      real*8 emax,getcth,cpar,dpar,thrust,dot,shat,getrapidity
      real*8 threedot,getpseudorap,getdelphi,getinvm
      integer i,j,kk,imax

      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM
c masses
      double precision pmass(nexternal)
      common/to_mass/pmass
      real *8 pplab(0:3, nexternal)
      double precision shybst, chybst, chybstmo
      real*8 xd(1:3)
      data (xd(i), i=1,3) /0,0,1/
      real*8 azi,azinorm,dr,etaq1,etaq2,ptcut,ptg,ptq1,ptq2,qqm,yq1,yq2,
     &yqq,xptq(0:3),xptb(0:3),yptqtb(0:3)
      logical ddflag,siq1flag,siq2flag
      double precision pi
      PARAMETER (PI=3.14159265358979312D0)

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
c
      chybst=cosh(ybst_til_tolab)
      shybst=sinh(ybst_til_tolab)
      chybstmo= chybst-1.d0
      do i=3,nexternal
      call boostwdir2(chybst, shybst, chybstmo,xd,pp(0,i), pplab(0,i))
      enddo
      do i=0,3
        xptq(i)=pplab(i,3)
        xptb(i)=pplab(i,4)
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
      call mfill(1,(ptg),(WWW))
      call mfill(18,(ptg),(WWW))
      if(ptg.gt.0) call mfill(2,(log10(ptg)),(WWW))
      call mfill(3,(qqm),(WWW))
      call mfill(4,(azi),(WWW))
      call mfill(13,(azi),(WWW))
      if(azinorm.gt.0)call mfill(17,(log10(azinorm)),(WWW))
      call mfill(5,(dr),(WWW))
      call mfill(14,(dr),(WWW))
      call mfill(10,(etaq1-etaq2),(WWW))
      call mfill(11,(yqq),(WWW))
      call mfill(12,(yq1-yq2),(WWW))
      call mfill(6,(ptq2),(WWW))
      call mfill(19,(ptq2),(WWW))
      if(ptq2.gt.0) call mfill(7,(log10(ptq2)),(WWW))
      call mfill(15,(yq2),(WWW))
      call mfill(8,(ptq1),(WWW))
      call mfill(20,(ptq1),(WWW))
      if(ptq1.gt.0) call mfill(9,(log10(ptq1)),(WWW))
      call mfill(16,(yq1),(WWW))
c
c***************************************************** with cuts
c
      kk=50
      if(ddflag)then
        call mfill(kk+1,(ptg),(WWW))
        call mfill(kk+18,(ptg),(WWW))
        if(ptg.gt.0) call mfill(kk+2,(log10(ptg)),(WWW))
        call mfill(kk+3,(qqm),(WWW))
        call mfill(kk+4,(azi),(WWW))
        call mfill(kk+13,(azi),(WWW))
        if(azinorm.gt.0) 
     #    call mfill(kk+17,(log10(azinorm)),(WWW))
        call mfill(kk+5,(dr),(WWW))
        call mfill(kk+14,(dr),(WWW))
        call mfill(kk+10,(etaq1-etaq2),(WWW))
        call mfill(kk+11,(yqq),(WWW))
        call mfill(kk+12,(yq1-yq2),(WWW))
      endif
      if(abs(yq2).lt.ycut)then
        call mfill(kk+6,(ptq2),(WWW))
        call mfill(kk+19,(ptq2),(WWW))
        if(ptq2.gt.0) call mfill(kk+7,(log10(ptq2)),(WWW))
      endif
      if(ptq2.gt.ptcut)call mfill(kk+15,(yq2),(WWW))
      if(abs(yq1).lt.ycut)then
        call mfill(kk+8,(ptq1),(WWW))
        call mfill(kk+20,(ptq1),(WWW))
        if(ptq1.gt.0) call mfill(kk+9,(log10(ptq1)),(WWW))
      endif
      if(ptq1.gt.ptcut)call mfill(kk+16,(yq1),(WWW))
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
