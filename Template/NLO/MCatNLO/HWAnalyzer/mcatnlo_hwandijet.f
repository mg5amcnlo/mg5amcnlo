C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE HWABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      real*4 pi
      parameter (pi=3.14160E0)
      integer j,k
c
      call inihist
c
      call mbook( 1,'total rate  ',1.0e0,0.5e0,5.5e0)
      call mbook( 2,'pt 1st      ',10.0e0,0e0,1000e0)
      call mbook( 3,'eta 1st     ',0.2e0,0e0,6e0)
      call mbook( 4,'rap 1st     ',0.2e0,0e0,6e0)
      call mbook( 5,'pt 2nd      ',10.0e0,0e0,1000e0)
      call mbook( 6,'eta 2nd     ',0.2e0,0e0,6e0)
      call mbook( 7,'rap 2nd     ',0.2e0,0e0,6e0)
      call mbook( 8,'pt pair     ',10.0e0,0e0,1000e0)
      call mbook( 9,'log[pt] pair',0.1e0,0e0,5e0)
      call mbook(10,'eta pair    ',0.2e0,0e0,6e0)
      call mbook(11,'rap pair    ',0.2e0,0e0,6e0)
      call mbook(12,'delta Rjj   ',0.1e0,0e0,5e0)
      call mbook(13,'delta phijj ',pi/50e0,0e0,pi)
      call mbook(14,'mjj         ',10e0,0e0,1000e0)
      call mbook(15,'bthr 1st    ',10.0e0,0e0,1000e0)
      call mbook(16,'bthr 2st    ',10.0e0,0e0,1000e0)
      call mbook(17,'bthr incl   ',10.0e0,0e0,1000e0)
      call mbook(18,'pt incl     ',10.0e0,0e0,1000e0)
      call mbook(19,'eta incl    ',0.2e0,0e0,6e0)
      call mbook(20,'rap incl    ',0.2e0,0e0,6e0)

 999  END

C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      REAL*8 XNORM
      INTEGER I,J,K
      OPEN(UNIT=99,NAME='HERW.top',STATUS='UNKNOWN')
C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB, SINCE THE HERWIG 
C WEIGHT IS IN NB, AND CORRESPONDS TO THE AVERAGE CROSS SECTION
      XNORM=1.D3/DFLOAT(NEVHEP)
      DO I=1,100              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+100)
        CALL MOPERA(I+100,'F',I+100,I+100,SNGL(XNORM),0.E0)
 	CALL MFINAL3(I+100)             
      ENDDO                          
C
      call multitop(100+ 1,99,3,2,'total rate  ',' ','LIN')
      call multitop(100+ 2,99,3,2,'pt 1st      ',' ','LOG')
      call multitop(100+ 3,99,3,2,'eta 1st     ',' ','LIN')
      call multitop(100+ 4,99,3,2,'rap 1st     ',' ','LIN')
      call multitop(100+ 5,99,3,2,'pt 2nd      ',' ','LOG')
      call multitop(100+ 6,99,3,2,'eta 2nd     ',' ','LIN')
      call multitop(100+ 7,99,3,2,'rap 2nd     ',' ','LIN')
      call multitop(100+ 8,99,3,2,'pt pair     ',' ','LOG')
      call multitop(100+ 9,99,3,2,'log[pt] pair',' ','LOG')
      call multitop(100+10,99,3,2,'eta pair    ',' ','LIN')
      call multitop(100+11,99,3,2,'rap pair    ',' ','LIN')
      call multitop(100+12,99,3,2,'delta Rjj   ',' ','LOG')
      call multitop(100+13,99,3,2,'delta phijj ',' ','LOG')
      call multitop(100+14,99,3,2,'mjj         ',' ','LOG')
      call multitop(100+15,99,3,2,'bthr 1st    ',' ','LOG')
      call multitop(100+16,99,3,2,'bthr 2st    ',' ','LOG')
      call multitop(100+17,99,3,2,'bthr incl   ',' ','LOG')
      call multitop(100+18,99,3,2,'pt incl     ',' ','LOG')
      call multitop(100+19,99,3,2,'eta incl    ',' ','LOG')
      call multitop(100+20,99,3,2,'rap incl    ',' ','LOG')
      
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      DOUBLE PRECISION HWVDOT,PSUM(4),PJJ(4)
      INTEGER ICHSUM,ICHINI,IHEP,IST,ID,IJ,IE(30),INU(30),J,NE,NNU,
     &     NN,I
      LOGICAL DIDSOF
      double precision PE(5),PNU(5),pw(5),getptv,getpseudorapv,ptenu,
     &     drenu,getdrv,dphienu,getdelphiv,getptv4,getpseudorapv4,
     &     getinvmv, getinvmv4,getdelphiv4,getdrv4,pte,etae,ptnu,etanu,
     &     bthrs,ptjj,etajj,rapjj,getrapidityv4
c jet stuff
      INTEGER NMAX
      PARAMETER (NMAX=2000)
      INTEGER NJET,JET(NMAX),IPOS(NMAX)
      DOUBLE PRECISION PALG,RFJ,SYCUT,PP(4,NMAX),PJET(4,NMAX),
     # PTJET(NMAX),ETAJET(NMAX),RAPJET(NMAX)
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
      INTEGER KK,J1,J2
      
c
      IF (IERROR.NE.0) RETURN
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,4)).EQ.SIGN(1.D0,PHEP(3,5)))THEN
        CALL HWWARN('HWANAL',111)
        GOTO 999
      ENDIF
      WWW0=EVWGT
      CALL HWVSUM(4,PHEP(1,1),PHEP(1,2),PSUM)
      CALL HWVSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      ICHINI=ICHRG(IDHW(1))+ICHRG(IDHW(2))
      DIDSOF=.FALSE.
      nn=0
      DO 100 IHEP=1,NHEP
        IF (IDHW(IHEP).EQ.16) DIDSOF=.TRUE.
        IF (ISTHEP(IHEP).EQ.1) THEN
          CALL HWVSUM(4,PHEP(1,IHEP),PSUM,PSUM)
          ICHSUM=ICHSUM+ICHRG(IDHW(IHEP))
        ENDIF
        IST=ISTHEP(IHEP)      
        ID=IDHEP(IHEP)
        IF(ABS(ID).GT.100.AND.IST.EQ.1) THEN
           NN=NN+1
           IF (NN.GT.NMAX) STOP 'Too many particles [hadrons]!'
           IPOS(NN)=IHEP
           DO I=1,4
              PP(I,NN)=PHEP(I,IHEP)
           ENDDO
        ENDIF
  100 CONTINUE
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (HWVDOT(3,PSUM,PSUM).GT.1.E-4*PHEP(4,1)**2) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',112)
         GOTO 999
      ENDIF

C---CLUSTER THE EVENT
      palg=1.d0
      rfj=0.7d0
      sycut=40d0
      call fastjetppgenkt(pp,nn,rfj,sycut,palg,pjet,njet,jet)
c need at least 2 jets      
      if (njet.lt.2) return
c Check order in pt
      do i=1,njet
         ptjet(i)=getptv4(pjet(1,i))
         if(i.gt.1)then
            if (ptjet(i).gt.ptjet(i-1)) then
               write (*,*) "Error in fastjet: "//
     &              "jets should be ordered in pt"
               CALL HWWARN('HWANAL',501)
            endif
         endif
         etajet(i)=getpseudorapv4(pjet(1,i))
         rapjet(i)=getrapidityv4(pjet(1,i))
      enddo
      do i=1,4
         pjj(i)=pjet(i,1)+pjet(i,2)
      enddo
      ptjj=getptv4(pjj)
      etajj=getpseudorapv4(pjj)
      rapjj=getrapidityv4(pjj)

      call mfill( 1,sngl(1d0),sngl(www0))
      call mfill( 2,sngl(ptjet(1)),sngl(www0))
      call mfill( 3,sngl(abs(etajet(1))),sngl(www0))
      call mfill( 4,sngl(abs(rapjet(1))),sngl(www0))
      call mfill( 5,sngl(ptjet(2)),sngl(www0))
      call mfill( 6,sngl(abs(etajet(2))),sngl(www0))
      call mfill( 7,sngl(abs(rapjet(2))),sngl(www0))
      call mfill( 8,sngl(ptjj),sngl(www0))
      if (ptjj.gt.1.0d0)
     &     call mfill( 9,sngl(log10(ptjj)),sngl(www0))
      call mfill(10,sngl(abs(etajj)),sngl(www0))
      call mfill(11,sngl(abs(rapjj)),sngl(www0))
      call mfill(12,sngl(getdrv4(pjet(1,1),pjet(1,2))),sngl(www0))
      call mfill(13,sngl(getdelphiv4(pjet(1,1),pjet(1,2))),sngl(www0))
      call mfill(14,sngl(getinvmv4(pjj)),sngl(www0))
      bthrs=ptjet(1)*exp(-abs(etajet(1)))
      call mfill(15,sngl(bthrs),sngl(www0))
      bthrs=ptjet(2)*exp(-abs(etajet(2)))
      call mfill(16,sngl(bthrs),sngl(www0))
      do i=1,njet
        bthrs=ptjet(i)*exp(-abs(etajet(i)))
        call mfill(17,sngl(abs(bthrs)),sngl(www0))
        call mfill(18,sngl(abs(ptjet(i))),sngl(www0))
        call mfill(19,sngl(abs(etajet(i))),sngl(www0))
        call mfill(20,sngl(abs(rapjet(i))),sngl(www0))
      enddo

 999  END


      FUNCTION RANDA(SEED)
*     -----------------
* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
* Use seed = 1 as first value.
*
      IMPLICIT INTEGER(A-Z)
      DOUBLE PRECISION MINV,RANDA
      SAVE
      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
      PARAMETER(MINV=0.46566128752458d-09)
      HI = SEED/Q
      LO = MOD(SEED,Q)
      SEED = A*LO - R*HI
      IF(SEED.LE.0) SEED = SEED + M
      RANDA = SEED*MINV
      END




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


      function getptv(p)
      implicit none
      real*8 getptv,p(5)
c
      getptv=sqrt(p(1)**2+p(2)**2)
      return
      end


      function getpseudorapv(p)
      implicit none
      real*8 getpseudorapv,p(5)
      real*8 getpseudorap
c
      getpseudorapv=getpseudorap(p(4),p(1),p(2),p(3))
      return
      end


      function getrapidityv(p)
      implicit none
      real*8 getrapidityv,p(5)
      real*8 getrapidity
c
      getrapidityv=getrapidity(p(4),p(3))
      return
      end


      function getdrv(p1,p2)
      implicit none
      real*8 getdrv,p1(5),p2(5)
      real*8 getdr
c
      getdrv=getdr(p1(4),p1(1),p1(2),p1(3),
     #             p2(4),p2(1),p2(2),p2(3))
      return
      end


      function getinvmv(p)
      implicit none
      real*8 getinvmv,p(5)
      real*8 getinvm
c
      getinvmv=getinvm(p(4),p(1),p(2),p(3))
      return
      end


      function getdelphiv(p1,p2)
      implicit none
      real*8 getdelphiv,p1(5),p2(5)
      real*8 getdelphi
c
      getdelphiv=getdelphi(p1(1),p1(2),
     #                     p2(1),p2(2))
      return
      end


      function getptv4(p)
      implicit none
      real*8 getptv4,p(4)
c
      getptv4=sqrt(p(1)**2+p(2)**2)
      return
      end


      function getpseudorapv4(p)
      implicit none
      real*8 getpseudorapv4,p(4)
      real*8 getpseudorap
c
      getpseudorapv4=getpseudorap(p(4),p(1),p(2),p(3))
      return
      end


      function getrapidityv4(p)
      implicit none
      real*8 getrapidityv4,p(4)
      real*8 getrapidity
c
      getrapidityv4=getrapidity(p(4),p(3))
      return
      end


      function getdrv4(p1,p2)
      implicit none
      real*8 getdrv4,p1(4),p2(4)
      real*8 getdr
c
      getdrv4=getdr(p1(4),p1(1),p1(2),p1(3),
     #              p2(4),p2(1),p2(2),p2(3))
      return
      end


      function getinvmv4(p)
      implicit none
      real*8 getinvmv4,p(4)
      real*8 getinvm
c
      getinvmv4=getinvm(p(4),p(1),p(2),p(3))
      return
      end


      function getdelphiv4(p1,p2)
      implicit none
      real*8 getdelphiv4,p1(4),p2(4)
      real*8 getdelphi
c
      getdelphiv4=getdelphi(p1(1),p1(2),
     #                      p2(1),p2(2))
      return
      end


      function getcosv4(q1,q2)
      implicit none
      real*8 getcosv4,q1(4),q2(4)
      real*8 xnorm1,xnorm2,tmp
c
      if(q1(4).lt.0.d0.or.q2(4).lt.0.d0)then
        getcosv4=-1.d10
        return
      endif
      xnorm1=sqrt(q1(1)**2+q1(2)**2+q1(3)**2)
      xnorm2=sqrt(q2(1)**2+q2(2)**2+q2(3)**2)
      if(xnorm1.lt.1.d-6.or.xnorm2.lt.1.d-6)then
        tmp=-1.d10
      else
        tmp=q1(1)*q2(1)+q1(2)*q2(2)+q1(3)*q2(3)
        tmp=tmp/(xnorm1*xnorm2)
        if(abs(tmp).gt.1.d0.and.abs(tmp).le.1.001d0)then
          tmp=sign(1.d0,tmp)
        elseif(abs(tmp).gt.1.001d0)then
          write(*,*)'Error in getcosv4',tmp
          stop
        endif
      endif
      getcosv4=tmp
      return
      end


      subroutine getperpenv4(q1,q2,qperp)
c Normal to the plane defined by \vec{q1},\vec{q2}
      implicit none
      real*8 q1(4),q2(4),qperp(4)
      real*8 xnorm1,xnorm2
      integer i
c
      xnorm1=sqrt(q1(1)**2+q1(2)**2+q1(3)**2)
      xnorm2=sqrt(q2(1)**2+q2(2)**2+q2(3)**2)
      if(xnorm1.lt.1.d-6.or.xnorm2.lt.1.d-6)then
        do i=1,4
          qperp(i)=-1.d10
        enddo
      else
        qperp(1)=q1(2)*q2(3)-q1(3)*q2(2)
        qperp(2)=q1(3)*q2(1)-q1(1)*q2(3)
        qperp(3)=q1(1)*q2(2)-q1(2)*q2(1)
        do i=1,3
          qperp(i)=qperp(i)/(xnorm1*xnorm2)
        enddo
        qperp(4)=1.d0
      endif
      return
      end
