
c
c
c Plotting routines
c
c
      subroutine initplot
c Book histograms in this routine. Use mbook or bookup. The entries
c of these routines are real*8
      implicit none
      include "coupl.inc"
      real * 8 pi,xmi,xms
      integer mxbin
      parameter (pi=3.14160D0, mxbin=100)
      real * 8 xm0,bin,xmlow,xmupp,gah,gammaX,xm02,xmlow2,xmupp2
      integer j,k
      character*5 cc(5)
      data cc/'     ',' cut1',' cut2',' cut3',' cut4'/
c 
      gammaX=300d0
      xm02=wmass**2
      gah=wwidth
      xmlow2=max(wmass-gammaX*wwidth,0d0)**2
      xmupp2=(wmass+gammaX*wwidth)**2

      xm0=sqrt(xm02)
      if(gah.eq.0)then
        bin=0.5d0
        xmi=xm0-24.75d0
        xms=xm0+25.25d0
      else
        xmlow=sqrt(xmlow2)
        xmupp=sqrt(xmupp2)
        bin=(xmupp-xmlow)/100.d0
        xmi=xm0-(49*bin+bin/2)
        xms=xm0+(50*bin+bin/2)
      endif
      call inihist 
c
      do j=1,3
      k=(j-1)*8
c
      call bookup(k+ 1,'e pt'//cc(j),2.d0,0.d0,200.d0)
      call bookup(k+ 2,'e eta'//cc(j),0.25d0,-9.d0,9.d0)
      call bookup(k+ 3,'nu pt'//cc(j),2.d0,0.d0,200.d0)
      call bookup(k+ 4,'nu eta'//cc(j),0.25d0,-9.d0,9.d0)
c
      call bookup(k+ 5,'DCe pt'//cc(j),2.d0,0.d0,200.d0)
      call bookup(k+ 6,'DCe eta'//cc(j),0.25d0,-9.d0,9.d0)
      call bookup(k+ 7,'DCnu pt'//cc(j),2.d0,0.d0,200.d0)
      call bookup(k+ 8,'DCnu eta'//cc(j),0.25d0,-9.d0,9.d0)
c
      enddo
c
      do j=1,5
      k=24+(j-1)*3
c
      call bookup(k+ 1,'W pt'//cc(j),2.d0,0.d0,200.d0)
      call bookup(k+ 2,'W y'//cc(j),0.25d0,-9.d0,9.d0)
      call bookup(k+ 3,'mW'//cc(j),bin,xmi,xms)
c
      enddo
      return
      end



      subroutine topout
      logical usexinteg
      common/cusexinteg/usexinteg
      integer itmax
      common/citmax/itmax
      real xnorm
c
      xnorm=1.d0/float(itmax)
      do i=1,200
        if(usexinteg)call mopera(i,'+',i,i,xnorm,0.d0)
        call mfinal(i)
      enddo
c
      do j=1,5
      k=24+(j-1)*3
      call multitop(k+ 1,3,2,'W pt',' ','LOG')
      call multitop(k+ 2,3,2,'W y',' ','LOG')
      call multitop(k+ 3,3,2,'mW',' ','LOG')
      enddo                
c
      do j=1,3
      k=(j-1)*8
c                          
      call multitop(k+ 1,3,2,'e pt',' ','LOG')
      call multitop(k+ 2,3,2,'e eta',' ','LOG')
      call multitop(k+ 3,3,2,'nu pt',' ','LOG')
      call multitop(k+ 4,3,2,'nu eta',' ','LOG')
c                        
      call multitop(k+ 5,3,2,'DCe pt',' ','LOG')
      call multitop(k+ 6,3,2,'DCe eta',' ','LOG')
      call multitop(k+ 7,3,2,'DCnu pt',' ','LOG')
      call multitop(k+ 8,3,2,'DCnu eta',' ','LOG')
c
      enddo
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
      real*8 emax,getcth,cpar,dpar,thrust,dot,shat
      integer i,j,kk,imax

      real*8 rndec(10)
      common/crndec/rndec

      real*8 getrapidity,getpseudorap,etaemin(2),etaemax(2),ptemin(2),
     # xmv,pte,etae,ptnu,etanu,ptdce,etadce,ptdcnu,wt,ptv,yv,
     # etadcnu,chybst,shybst,chybstmo,getinvm
      real*8 xd(1:3)
      data (xd(i),i=1,3)/0,0,1/
      real*8 pplab(0:3,nexternal)
      real*8 ppv(5),ppe(5),ppnu(5),ppdce(5),ppdcnu(5)
      logical flag

      LOGICAL  IS_A_J(NEXTERNAL),IS_A_L(NEXTERNAL)
      LOGICAL  IS_A_B(NEXTERNAL),IS_A_A(NEXTERNAL)
      LOGICAL  IS_A_NU(NEXTERNAL),IS_HEAVY(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_A,IS_A_L,IS_A_B,IS_A_NU,IS_HEAVY
c masses
      double precision pmass(nexternal)
      common/to_mass/pmass
c
      if(itype.eq.11.or.itype.eq.12)then
        kk=0
      elseif(itype.eq.20)then
        return
      else
        write(*,*)'Error in outfun: unknown itype',itype
        stop
      endif
c

      
c$$$      if(sqrt(sh).lt.10000)then
c$$$        etaemin(1)=0.d0
c$$$        etaemax(1)=1.d0
c$$$        ptemin(1)=20.d0
c$$$        etaemin(2)=1.d0
c$$$        etaemax(2)=2.5d0
c$$$        ptemin(2)=20.d0
c$$$      else
        etaemin(1)=0.d0
        etaemax(1)=2.5d0
        ptemin(1)=20.d0

        etaemin(2)=0.d0
        etaemax(2)=2.5d0
        ptemin(2)=40.d0
c$$$      endif
c
      chybst=cosh(ybst_til_tolab)
      shybst=sinh(ybst_til_tolab)
      chybstmo=chybst-1.d0
      do i=3,nexternal
        call boostwdir2(chybst,shybst,chybstmo,xd,
     #                  pp(0,i),pplab(0,i))
      enddo

      do i=1,4
         ppe(i)=pplab(mod(i,4),3)
         ppnu(i)=pplab(mod(i,4),4)
         ppv(i)=ppe(i)+ppnu(i)
      enddo
      ppv(5)=getinvm(ppv(4),ppv(1),ppv(2),ppv(3))
      ppe(5)=0.d0
      ppnu(5)=0.d0
C Include leptons resulting from isotropic W decay
      ppdce(5)=0.d0
      ppdcnu(5)=0.d0
      call pdecay(rndec(1),rndec(2),ppv,ppdce,ppdcnu,wt)
C Variables of the vector boson
      xmv=ppv(5)
      ptv=sqrt(ppv(1)**2+ppv(2)**2)
      yv=getrapidity(ppv(4),ppv(3))
C Variables of the leptons
      pte=sqrt(ppe(1)**2+ppe(2)**2)
      etae= getpseudorap(ppe(4),ppe(1),ppe(2),ppe(3))
      ptnu=sqrt(ppnu(1)**2+ppnu(2)**2)
      etanu= getpseudorap(ppnu(4),ppnu(1),ppnu(2),ppnu(3))
C Variables of the leptons coming from W isotropic decay
      ptdce=sqrt(ppdce(1)**2+ppdce(2)**2)
      etadce= getpseudorap(ppdce(4),ppdce(1),ppdce(2),ppdce(3))
      ptdcnu=sqrt(ppdcnu(1)**2+ppdcnu(2)**2)
      etadcnu= getpseudorap(ppdcnu(4),ppdcnu(1),ppdcnu(2),ppdcnu(3))
c
      do j=1,5
        kk=24+(j-1)*3
        flag=.false.
        if(j.eq.1)then
          flag=.true.
        elseif(j.le.3)then
          if( ptnu.ge.20.d0 .and.
     &        pte.ge.ptemin(j-1) .and.
     &        abs(etae).ge.etaemin(j-1) .and.
     &        abs(etae).le.etaemax(j-1) )flag=.true.
        elseif(j.le.5)then
          if( ptdcnu.ge.20.d0 .and.
     &        ptdce.ge.ptemin(j-3) .and.
     &        abs(etadce).ge.etaemin(j-3) .and.
     &        abs(etadce).le.etaemax(j-3) )flag=.true.
        endif
        if(flag)then
          call mfill(kk+1,ptv,www)
          call mfill(kk+2,yv,www)
          call mfill(kk+3,xmv,www)
        endif
      enddo
C
      kk=0
      call mfill(kk+ 1,pte,www)
      call mfill(kk+ 2,etae,www)
      call mfill(kk+ 3,ptnu,www)
      call mfill(kk+ 4,etanu,www)
      call mfill(kk+ 5,ptdce,www)
      call mfill(kk+ 6,etadce,www)
      call mfill(kk+ 7,ptdcnu,www)
      call mfill(kk+ 8,etadcnu,www)
C
      do j=2,3
        kk=(j-1)*8
        if( ptnu.ge.20.d0 .and.
     &      abs(etae).ge.etaemin(j-1) .and.
     &      abs(etae).le.etaemax(j-1) )
     &        call mfill(kk+1,pte,www)
        if( ptnu.ge.20.d0 .and.
     &      pte.ge.ptemin(j-1) )
     &        call mfill(kk+2,etae,www)
        if( pte.ge.ptemin(j-1) .and.
     &      abs(etae).ge.etaemin(j-1) .and.
     &      abs(etae).le.etaemax(j-1) )then
              call mfill(kk+3,ptnu,www)
              if(ptnu.ge.20.d0)
     &          call mfill(kk+4,etanu,www)
        endif
C
        if( ptdcnu.ge.20.d0 .and.
     &      abs(etadce).ge.etaemin(j-1) .and.
     &      abs(etadce).le.etaemax(j-1) )
     &        call mfill(kk+5,ptdce,www)
        if( ptdcnu.ge.20.d0 .and.
     &      ptdce.ge.ptemin(j-1) )
     &        call mfill(kk+6,etadce,www)
        if( ptdce.ge.ptemin(j-1) .and.
     &      abs(etadce).ge.etaemin(j-1) .and.
     &      abs(etadce).le.etaemax(j-1) )then
              call mfill(kk+7,ptdcnu,www)
              if(ptdcnu.ge.20.d0)
     &          call mfill(kk+8,etadcnu,www)
        endif
      enddo
      return
      end




      function getrapidity(en,pl)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-8)
c
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
        if( (xplus/xminus).gt.tiny.and.(xminus/xplus).gt.tiny )then
          y=0.5d0*log( xplus/xminus )
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


      function getdr(en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2)
      implicit none
      real*8 getdr,en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2,deta,dphi,
     # getpseudorap,getdelphi
c
      deta=getpseudorap(en1,ptx1,pty1,pl1)-
     #     getpseudorap(en2,ptx2,pty2,pl2)
      dphi=getdelphi(ptx1,pty1,ptx2,pty2)
      getdr=sqrt(dphi**2+deta**2)
      return
      end


      function getdry(en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2)
      implicit none
      real*8 getdry,en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2,deta,dphi,
     # getrapidity,getdelphi
c
      deta=getrapidity(en1,pl1)-
     #     getrapidity(en2,pl2)
      dphi=getdelphi(ptx1,pty1,ptx2,pty2)
      getdry=sqrt(dphi**2+deta**2)
      return
      end


C Here x1 and x2 are generated externally -- needed to avoid instabilities
C due to counterevents
      SUBROUTINE PDECAY(X1,X2,P,Q1,Q2,WT)
C Decays a particle with momentum P into two particles with momenta
C Q1 and Q2; these momenta are understood in the lab frame. WT is the 
c phase space density beta_cm.
C The decay is spherically symmetric in the decay C_of_M frame
C Written by MLM, modified by SF
      implicit none
      double precision pi,twopi
      PARAMETER(PI=3.14159,TWOPI=2.*PI)
      double precision q2e,qp,ctheta,stheta,phi,qplab,qplon,qptr,pmod,
     $     ptr,wt,bet,gam
      double precision P(5),Q1(5),Q2(5),V(3),U(3),pm,q1m,q2m
      double precision x1,x2
      integer i
c
      PM=P(5)
      Q1M=Q1(5)
      Q2M=Q2(5)
      Q2E=(PM**2-Q1M**2+Q2M**2)/(2.*PM)
      QP=SQRT(MAX(Q2E**2-Q2M**2,0.d0))
      CTHETA=2.0*X1-1.
      STHETA=SQRT(1.-CTHETA**2)
      PHI=TWOPI*X2
      QPLON=QP*CTHETA
      QPTR=QP*STHETA
C The transverse component of P defined here need not coincide with that
c of the calling routine -- it's purely conventional
      PMOD=SQRT(P(1)**2+P(2)**2+P(3)**2)
      PTR=SQRT(P(2)**2+P(3)**2)                              
C U and V are 3-vectors of modulus 1, perpendicular to each other and to P;
C therefore, they are invariant under the boost along the longitudinal
C component of P
      IF(PTR.LT.1.E-4) THEN
        V(1)=0.
        V(2)=1.
        V(3)=0.
        U(1)=0.
        U(2)=0.
        U(3)=1.
      ELSE
        V(1)=0.
        V(2)=P(3)/PTR
        V(3)=-P(2)/PTR
        U(1)=PTR/PMOD
        U(2)=-P(1)*P(2)/PTR/PMOD
        U(3)=-P(1)*P(3)/PTR/PMOD
      ENDIF
      GAM=P(4)/PM
      BET=PMOD/P(4)
      QPLAB=GAM*(QPLON+BET*Q2E)
      DO I=1,3
      Q2(I)=QPLAB*P(I)/PMOD+QPTR*(V(I)*SIN(PHI)+U(I)*COS(PHI))
      Q1(I)=P(I)-Q2(I)
      END DO
      Q2(4)=GAM*(Q2E+BET*QPLON)
      Q1(4)=P(4)-Q2(4)
      WT=2.*QP/PM
      END            


      FUNCTION RANDAXX(SEED)
*     -----------------
* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
* Use seed = 1 as first value.
*
      IMPLICIT INTEGER(A-Z)
      DOUBLE PRECISION MINV,RANDAXX
      SAVE
      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
      PARAMETER(MINV=0.46566128752458d-09)
      HI = SEED/Q
      LO = MOD(SEED,Q)
      SEED = A*LO - R*HI
      IF(SEED.LE.0) SEED = SEED + M
      RANDAXX = SEED*MINV
      END


