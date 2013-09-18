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
      integer i,kk,j
      include 'run.inc'
      real * 8 xmh0
      real * 8 xmhi,xmhs
      character*5 cc(2)
      data cc/'     ',' cuts'/
c
      xmh0=125d0
      xmhi=(xmh0-25d0)
      xmhs=(xmh0+25d0)
      call inihist
      do j=1,1
      kk=(j-1)*50
      call bookup(kk+1,'Higgs pT'//cc(j),2.d0,0.d0,200.d0)
      call bookup(kk+2,'Higgs pT'//cc(j),5.d0,0.d0,500.d0)
      call bookup(kk+3,'Higgs log(pT)'//cc(j),0.05d0,0.1d0,5.d0)
      call bookup(kk+4,'Higgs pT,  |y_H| < 2'//cc(j),2.d0,0.d0,200.d0)
      call bookup(kk+5,'Higgs pT,  |y_H| < 2'//cc(j),5.d0,0.d0,500.d0)
      call bookup(kk+6,'Higgs log(pT),  |y_H| < 2'//cc(j),
     #                                           0.05d0,0.1d0,5.d0)

      call bookup(kk+7,'H jet pT'//cc(j),2.d0,0.d0,200.d0)
      call bookup(kk+8,'H jet pT'//cc(j),5.d0,0.d0,500.d0)
      call bookup(kk+9,'H jet log(pT)'//cc(j),0.05d0,0.1d0,5.d0)
      call bookup(kk+10,'H jet pT,  |y_Hj| < 2'//cc(j),2.d0,0.d0,200.d0)
      call bookup(kk+11,'H jet pT,  |y_Hj| < 2'//cc(j),5.d0,0.d0,500.d0)
      call bookup(kk+12,'H jet log(pT),  |y_Hj| < 2'//cc(j),
     #                                             0.05d0,0.1d0,5.d0)

      call bookup(kk+13,'Inc jet pT'//cc(j),2.d0,0.d0,200.d0)
      call bookup(kk+14,'Inc jet pT'//cc(j),5.d0,0.d0,500.d0)
      call bookup(kk+15,'Inc jet log(pT)'//cc(j),0.05d0,0.1d0,5.d0)
      call bookup(kk+16,'Inc jet pT,  |y_Ij| < 2'//cc(j),2.d0,0.d0,2.d2)
      call bookup(kk+17,'Inc jet pT,  |y_Ij| < 2'//cc(j),5.d0,0.d0,5.d2)
      call bookup(kk+18,'Inc jet log(pT),  |y_Ij| < 2'//cc(j),
     #                                               0.05d0,0.1d0,5.d0)

      call bookup(kk+19,'Higgs y',0.2d0,-6.d0,6.d0)
      call bookup(kk+20,'Higgs y,  pT_H > 10 GeV',0.12d0,-6.d0,6.d0)
      call bookup(kk+21,'Higgs y,  pT_H > 30 GeV',0.12d0,-6.d0,6.d0)
      call bookup(kk+22,'Higgs y,  pT_H > 50 GeV',0.12d0,-6.d0,6.d0)
      call bookup(kk+23,'Higgs y,  pT_H > 70 GeV',0.12d0,-6.d0,6.d0)
      call bookup(kk+24,'Higgs y,  pt_H > 90 GeV',0.12d0,-6.d0,6.d0)

      call bookup(kk+25,'H jet y',0.2d0,-6.d0,6.d0)
      call bookup(kk+26,'H jet y,  pT_Hj > 10 GeV',0.2d0,-6.d0,6.d0)
      call bookup(kk+27,'H jet y,  pT_Hj > 30 GeV',0.2d0,-6.d0,6.d0)
      call bookup(kk+28,'H jet y,  pT_Hj > 50 GeV',0.2d0,-6.d0,6.d0)
      call bookup(kk+29,'H jet y,  pT_Hj > 70 GeV',0.2d0,-6.d0,6.d0)
      call bookup(kk+30,'H jet y,  pT_Hj > 90 GeV',0.2d0,-6.d0,6.d0)

      call bookup(kk+31,'H-Hj y',0.2d0,-6.d0,6.d0)
      call bookup(kk+32,'H-Hj y,  pT_Hj > 10 GeV',0.2d0,-6.d0,6.d0)
      call bookup(kk+33,'H-Hj y,  pT_Hj > 30 GeV',0.2d0,-6.d0,6.d0)
      call bookup(kk+34,'H-Hj y,  pT_Hj > 50 GeV',0.2d0,-6.d0,6.d0)
      call bookup(kk+35,'H-Hj y,  pT_Hj > 70 GeV',0.2d0,-6.d0,6.d0)
      call bookup(kk+36,'H-Hj y,  pT_Hj > 90 GeV',0.2d0,-6.d0,6.d0)

      call bookup(kk+37,'njets',1.d0,-0.5d0,10.5d0)
      call bookup(kk+38,'njets, |y_j| < 2.5 GeV',1.d0,-0.5d0,10.5d0)

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

      do j=1,1
      k=(j-1)*50
      call multitop(k+1,3,2,'Higgs pT (GeV)',' ','LOG')
      call multitop(k+2,3,2,'Higgs pT (GeV)',' ','LOG')
      call multitop(k+3,3,2,'Higgs log(pT/GeV)',' ','LOG')
      call multitop(k+4,3,2,'Higgs pT (GeV)',' ','LOG')
      call multitop(k+5,3,2,'Higgs pT (GeV)',' ','LOG')
      call multitop(k+6,3,2,'Higgs log(pT/GeV)',' ','LOG')
c
      call multitop(k+7,3,2,'H jet pT (GeV)',' ','LOG')
      call multitop(k+8,3,2,'H jet pT (GeV)',' ','LOG')
      call multitop(k+9,3,2,'H jet log(pT/GeV)',' ','LOG')
      call multitop(k+10,3,2,'H jet pT (GeV)',' ','LOG')
      call multitop(k+11,3,2,'H jet pT (GeV)',' ','LOG')
      call multitop(k+12,3,2,'H jet log(pT/GeV)',' ','LOG')
c
      call multitop(k+13,3,2,'Inc jet pT (GeV)',' ','LOG')
      call multitop(k+14,3,2,'Inc jet pT (GeV)',' ','LOG')
      call multitop(k+15,3,2,'Inc jet log(pT/GeV)',' ','LOG')
      call multitop(k+16,3,2,'Inc jet pT (GeV)',' ','LOG')
      call multitop(k+17,3,2,'Inc jet pT (GeV)',' ','LOG')
      call multitop(k+18,3,2,'Inc jet log(pT/GeV)',' ','LOG')
c
      call multitop(k+19,3,2,'Higgs y',' ','LOG')
      call multitop(k+20,3,2,'Higgs y',' ','LOG')
      call multitop(k+21,3,2,'Higgs y',' ','LOG')
      call multitop(k+22,3,2,'Higgs y',' ','LOG')
      call multitop(k+23,3,2,'Higgs y',' ','LOG')
      call multitop(k+24,3,2,'Higgs y',' ','LOG')
c     
      call multitop(k+25,3,2,'H jet y',' ','LOG')
      call multitop(k+26,3,2,'H jet y',' ','LOG')
      call multitop(k+27,3,2,'H jet y',' ','LOG')
      call multitop(k+28,3,2,'H jet y',' ','LOG')
      call multitop(k+29,3,2,'H jet y',' ','LOG')
      call multitop(k+30,3,2,'H jet y',' ','LOG')
c
      call multitop(k+31,3,2,'H-Hj y',' ','LOG')
      call multitop(k+32,3,2,'H-Hj y',' ','LOG')
      call multitop(k+33,3,2,'H-Hj y',' ','LOG')
      call multitop(k+34,3,2,'H-Hj y',' ','LOG')
      call multitop(k+35,3,2,'H-Hj y',' ','LOG')
      call multitop(k+36,3,2,'H-Hj y',' ','LOG')

      call multitop(k+37,3,2,'njets',' ','LOG')
      call multitop(k+38,3,2,'njets',' ','LOG')

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
      real*8 emax,getcth,cpar,dpar,thrust,dot,shat, getrapidity
      real*8 threedot,getpseudorap
      integer i,j,kk,imax,nj,njet_central

      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM
c masses
      double precision pmass(nexternal)
      common/to_mass/pmass
      real *8 pplab(0:3, nexternal)
      double precision shybst, chybst, chybstmo
      real*8 xd(1:3)
      data (xd(i), i=1,3) /0,0,1/
      real*8 njdble,njcdble,ptcut,pth,ptj1,y_central,yh,yj1,pph(0:3),ppj1(0:3)
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
      do i=1,nexternal
      call boostwdir2(chybst, shybst, chybstmo,xd,pp(0,i), pplab(0,i))
      enddo
      do i=0,3
        pph(i)=pplab(i,3)
        ppj1(i)=pplab(i,1)+pplab(i,2)-pplab(i,3)
      enddo
c Higgs variables
      pth=sqrt(pph(1)**2+pph(2)**2)
      yh=getrapidity(pph(0),pph(3))
c hardest jet variables
      ptj1=sqrt(ppj1(1)**2+ppj1(2)**2)
      yj1=getrapidity(ppj1(0),ppj1(3))
      njet=0
      njet_central=0
      y_central=2.5d0
      ptcut=1d1
      if(ptj1.gt.ptcut)njet=1
      if(ptj1.gt.ptcut.and.abs(yj1).le.y_central)njet_central=1
      njdble=dble(njet)
      njcdble=dble(njet_central)

C
      call mfill(kk+1,(pth),(WWW))
      call mfill(kk+2,(pth),(WWW))
      if(pth.gt.0.d0)call mfill(kk+3,(log10(pth)),(WWW))
      if(abs(yh).le.2.d0)then
         call mfill(kk+4,(pth),(WWW))
         call mfill(kk+5,(pth),(WWW))
         if(pth.gt.0.d0)call mfill(kk+6,(log10(pth)),(WWW))
      endif
c
      if(njet.ge.1)then
      call mfill(kk+7,(ptj1),(WWW))
      call mfill(kk+8,(ptj1),(WWW))
      if(ptj1.gt.0.d0)call mfill(kk+9,(log10(ptj1)),(WWW))
      if(abs(yj1).le.2.d0)then
         call mfill(kk+10,(ptj1),(WWW))
         call mfill(kk+11,(ptj1),(WWW))
         if(ptj1.gt.0.d0)call mfill(kk+12,(log10(ptj1)),
     &                                          (WWW))
      endif
c
      do nj=1,njet
         call mfill(kk+13,(ptj1),(WWW))
         call mfill(kk+14,(ptj1),(WWW))
         if(ptj1.gt.0.d0)call mfill(kk+15,(log10(ptj1)),
     &                                                (WWW))
         if(abs(yj1).le.2.d0)then
            call mfill(kk+16,(ptj1),(WWW))
            call mfill(kk+17,(ptj1),(WWW))
            if(ptj1.gt.0d0)call mfill(kk+18,(log10(ptj1)),
     &                                                   (WWW))
         endif
      enddo
      endif
c
      call mfill(kk+19,(yh),(WWW))
      if(pth.ge.10.d0) call mfill(kk+20,(yh),(WWW))
      if(pth.ge.30.d0) call mfill(kk+21,(yh),(WWW))
      if(pth.ge.50.d0) call mfill(kk+22,(yh),(WWW))
      if(pth.ge.70.d0) call mfill(kk+23,(yh),(WWW))
      if(pth.ge.90.d0) call mfill(kk+24,(yh),(WWW))  
c     
      if(njet.ge.1)then
      call mfill(kk+25,(yj1),(WWW))
      if(ptj1.ge.10.d0) call mfill(kk+26,(yj1),(WWW))
      if(ptj1.ge.30.d0) call mfill(kk+27,(yj1),(WWW))
      if(ptj1.ge.50.d0) call mfill(kk+28,(yj1),(WWW))
      if(ptj1.ge.70.d0) call mfill(kk+29,(yj1),(WWW))
      if(ptj1.ge.90.d0) call mfill(kk+30,(yj1),(WWW))
c
      call mfill(kk+31,(yh-yj1),(WWW))
      if(ptj1.ge.10.d0) call mfill(kk+32,(yh-yj1),(WWW))
      if(ptj1.ge.30.d0) call mfill(kk+33,(yh-yj1),(WWW))
      if(ptj1.ge.50.d0) call mfill(kk+34,(yh-yj1),(WWW))
      if(ptj1.ge.70.d0) call mfill(kk+35,(yh-yj1),(WWW))
      if(ptj1.ge.90.d0) call mfill(kk+36,(yh-yj1),(WWW))
      endif
c
      call mfill(kk+37,(njdble),(WWW))
      call mfill(kk+38,(njcdble),(WWW))

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
