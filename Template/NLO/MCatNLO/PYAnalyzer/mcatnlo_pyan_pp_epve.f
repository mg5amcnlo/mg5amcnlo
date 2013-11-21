c
c Example analysis for "p p > e+ ve [QCD]" process.
c
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      implicit none
      include 'reweight0.inc'
      integer j,kk,l,i
      character*5 cc(2)
      data cc/'     ','     '/
      integer nwgt,max_weight,nwgt_analysis
      common/cnwgt/nwgt
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      character*15 weights_info(max_weight)
      common/cwgtsinfo/weights_info
c
      call inihist
      nwgt_analysis=nwgt
      do i=1,1
      do kk=1,nwgt_analysis
        l=(kk-1)*16+(i-1)*8
        call mbook(l+1,'total rate  '//weights_info(kk)//cc(i),
     &       1.0d0,0.5d0,5.5d0)
        call mbook(l+2,'e rapidity  '//weights_info(kk)//cc(i),
     &       0.5d0,-5d0,5d0)
        call mbook(l+3,'e pt        '//weights_info(kk)//cc(i),
     &       10d0,0d0,200d0)
        call mbook(l+4,'et miss     '//weights_info(kk)//cc(i),
     &       10d0,0d0,200d0)
        call mbook(l+5,'trans. mass '//weights_info(kk)//cc(i),
     &       5d0,0d0,200d0)
        call mbook(l+6,'w rapidity  '//weights_info(kk)//cc(i),
     &       0.5d0,-5d0,5d0)
        call mbook(l+7,'w pt        '//weights_info(kk)//cc(i),
     &       10d00,0d0,200d0)
        call mbook(l+8,'cphi[e,ve]  '//weights_info(kk)//cc(i),
     &       0.05d0,-1d0,1d0)
      enddo
      enddo
 999  END


C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVT)
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      REAL*8 XNORM
      INTEGER I,J,KK,l,nwgt_analysis
      integer NPL
      parameter(NPL=15000)
      common/c_analysis/nwgt_analysis
      OPEN(UNIT=99,FILE='PYTLL.TOP',STATUS='UNKNOWN')
      XNORM=1.D0/IEVT
      DO I=1,NPL              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+NPL)
        CALL MOPERA(I+NPL,'F',I+NPL,I+NPL,(XNORM),0.D0)
 	CALL MFINAL3(I+NPL)             
      ENDDO                          
C
      do i=1,1
      do kk=1,nwgt_analysis
         l=(kk-1)*16+(i-1)*8
         call multitop(NPL+l+1,NPL-1,3,2,'total rate ',' ','LIN')
         call multitop(NPL+l+2,NPL-1,3,2,'e rapidity ',' ','LIN')
         call multitop(NPL+l+3,NPL-1,3,2,'e pt       ',' ','LOG')
         call multitop(NPL+l+4,NPL-1,3,2,'et miss    ',' ','LOG')
         call multitop(NPL+l+5,NPL-1,3,2,'trans. mass',' ','LOG')
         call multitop(NPL+l+6,NPL-1,3,2,'w rapidity ',' ','LIN')
         call multitop(NPL+l+7,NPL-1,3,2,'w pt       ',' ','LOG')
         call multitop(NPL+l+8,NPL-1,3,2,'cphi[e,ve] ',' ','LOG')
      enddo
      enddo
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE PYANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      implicit double precision(a-h, o-z)
      implicit integer(i-n)
      include 'reweight0.inc'
      DOUBLE PRECISION HWVDOT,PSUM(4),PPV(5),PTW,YW,YE,PPL(5),PPLB(5),
     & PTE,PLL,PTLB,PLLB,var,mtr,etmiss,cphi
      INTEGER ICHSUM,ICHINI,IHEP,IV,IFV,IST,ID,IJ,ID1,JPR,IDENT,
     #  ILL,ILLB,IHRD
      integer pychge
      double precision p1(4),p2(4),pihep(4)
      external pydata
      common/pyjets/n,npad,k(4000,5),p(4000,5),v(4000,5)
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
      common/pydat2/kchg(500,4),pmas(500,4),parf(2000),vckm(4,4)
      common/pydat3/mdcy(500,3),mdme(8000,2),brat(8000),kfdp(8000,5)
      common/pysubs/msel,mselpd,msub(500),kfin(2,-40:40),ckin(200)
      common/pypars/mstp(200),parp(200),msti(200),pari(200)
      LOGICAL DIDSOF,TEST1,TEST2,TEST3,TEST4,TEST5,TEST6,TEST7,flag
      REAL*8 PI,getrapidity
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0,TINY
      INTEGER KK,I,L
      DATA TINY/.1D-5/
      integer nwgt_analysis,max_weight
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      double precision ww(max_weight),www(max_weight)
      common/cww/ww
      DOUBLE PRECISION EVWEIGHT
      COMMON/CEVWEIGHT/EVWEIGHT
      INTEGER IFAIL
      COMMON/CIFAIL/IFAIL
      SAVE INOBOSON,INOLEPTON,INOLEPTONB
c
      IF(IFAIL.EQ.1)RETURN
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
         STOP
      ENDIF
      IDENT=24
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,P(3,3)).EQ.SIGN(1.D0,P(4,3)))THEN
         WRITE(*,*)'WARNING 111 IN PYANAL'
         GOTO 999
      ENDIF
      DO I=1,nwgt_analysis
         WWW(I)=EVWEIGHT*ww(i)/ww(1)
      ENDDO
      DO I=1,4
         P1(I)=P(1,I)
         P2(I)=P(2,I)
      ENDDO
      CALL VVSUM(4,P1,P2,PSUM)
      CALL VSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      KF1=K(1,2)
      KF2=K(2,2)
      ICHINI=PYCHGE(KF1)+PYCHGE(KF2)
      IFV =0
      IFL =0
      IFLB=0
      IV0 =10000
      IL0 =10000
      ILB0=10000
C
      DO 100 IHEP=1,N
        DO J=1,4
          PIHEP(J)=P(IHEP,J)
        ENDDO
        IST =K(IHEP,1)
        ID1 =K(IHEP,2)
        IORI=K(IHEP,3)
        IF (IST.LE.10) THEN
          CALL VVSUM(4,PIHEP,PSUM,PSUM)
          ICHSUM=ICHSUM+PYCHGE(ID1)
        ENDIF
        TEST1=IORI.EQ.0
        TEST2=ID1.EQ.IDENT
        TEST3=ID1.GT.0.AND.ABS(ID1).GE.11.AND.ABS(ID1).LE.16
        TEST4=ID1.LT.0.AND.ABS(ID1).GE.11.AND.ABS(ID1).LE.16
        IF(TEST1.AND.TEST2)IV0=IHEP
        TEST5=IORI.EQ.IV0
        IF(TEST5)THEN
           IF(TEST2)THEN
              IV1=IHEP
              IFV=IFV+1
           ELSEIF(TEST3)THEN
              IL0 =IHEP
           ELSEIF(TEST4)THEN
              ILB0=IHEP
           ENDIF
        ENDIF
        TEST6=IORI.EQ.IL0
        TEST7=IORI.EQ.ILB0
        IF(TEST6.AND.TEST3)THEN
           IL =IHEP
           IFL=IFL+1
        ENDIF
        IF(TEST7.AND.TEST4)THEN
           ILB=IHEP
           IFLB=IFLB+1
        ENDIF
 100  CONTINUE
C
      DO IJ=1,5
        PPV(IJ) =P(IV1,IJ)
        PPL(IJ) =P(IL, IJ)
        PPLB(IJ)=P(ILB,IJ)
      ENDDO
C CHECK MULTIPLICITIES
      IF(IFV.EQ.0) THEN
         INOBOSON=INOBOSON+1
         WRITE(*,*)'WARNING IN PYANAL: NO WEAK BOSON ',INOBOSON
         GOTO 999
      ELSEIF(IFV.GT.1) THEN
         WRITE(*,*)'WARNING IN PYANAL: TOO MANY WEAK BOSONS ',IFV
         GOTO 999
      ENDIF
      IF(IFL.EQ.0) THEN
         INOLEPTON=INOLEPTON+1
         WRITE(*,*)'WARNING IN PYANAL: NO LEPTON ',INOLEPTON
         GOTO 999
      ELSEIF(IFL.GT.1) THEN
         WRITE(*,*)'WARNING IN PYANAL: TOO MANY LEPTONS ',IFL
         GOTO 999
      ENDIF
      IF(IFLB.EQ.0) THEN
         INOLEPTONB=INOLEPTONB+1
         WRITE(*,*)'WARNING IN PYANAL: NO ANTILEPTON ',INOLEPTONB
         GOTO 999
      ELSEIF(IFLB.GT.1) THEN
         WRITE(*,*)'WARNING IN PYANAL: TOO MANY ANTILEPTONS ',IFLB
         GOTO 999
      ENDIF
C CHECK THAT THE LEPTONS ARE FINAL-STATE LEPTONS
      IF( (K(IL,1).NE.1).OR.K(ILB,1).NE.1 ) THEN
         WRITE(*,*)'WARNING 505 IN PYANAL'
         GOTO 999
      ENDIF
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (VDOT(3,PSUM,PSUM).GT.1.E-4*P(1,4)**2) THEN
         WRITE(*,*)'WARNING 112 IN PYANAL'
         GOTO 999
      ENDIF
      IF (ICHSUM.NE.ICHINI) THEN
         WRITE(*,*)'WARNING 113 IN PYANAL'
         GOTO 999
      ENDIF
      ye     = getrapidity(pplb(4), pplb(3))
      yw     = getrapidity(ppv(4), ppv(3))
      pte    = dsqrt(pplb(1)**2 + pplb(2)**2)
      ptw    = dsqrt(ppv(1)**2+ppv(2)**2)
      etmiss = dsqrt(ppl(1)**2 + ppl(2)**2)
      mtr    = dsqrt(2d0*pte*etmiss-2d0*ppl(1)*pplb(1)-2d0*ppl(2)*pplb(2))
      cphi   = (ppl(1)*pplb(1)+ppl(2)*pplb(2))/pte/etmiss
      var    = 1.d0
      do i=1,1
         do kk=1,nwgt_analysis
            l=(kk-1)*16+(i-1)*8
            call mfill(l+1,var,www(kk))
            call mfill(l+2,ye,www(kk))
            call mfill(l+3,pte,www(kk))
            call mfill(l+4,etmiss,www(kk))
            call mfill(l+5,mtr,www(kk))
            call mfill(l+6,yw,www(kk))
            call mfill(l+7,ptw,www(kk))
            call mfill(l+8,cphi,www(kk))
         enddo
      enddo
 999  END
      

C-----------------------------------------------------------------------
      SUBROUTINE VVSUM(N,P,Q,R)
C-----------------------------------------------------------------------
C    VECTOR SUM
C-----------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER N,I
      DOUBLE PRECISION P(N),Q(N),R(N)
      DO 10 I=1,N
   10 R(I)=P(I)+Q(I)
      END



C-----------------------------------------------------------------------
      SUBROUTINE VSCA(N,C,P,Q)
C-----------------------------------------------------------------------
C     VECTOR TIMES SCALAR
C-----------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER N,I
      DOUBLE PRECISION C,P(N),Q(N)
      DO 10 I=1,N
   10 Q(I)=C*P(I)
      END



C-----------------------------------------------------------------------
      FUNCTION VDOT(N,P,Q)
C-----------------------------------------------------------------------
C     VECTOR DOT PRODUCT
C-----------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER N,I
      DOUBLE PRECISION VDOT,PQ,P(N),Q(N)
      PQ=0.
      DO 10 I=1,N
   10 PQ=PQ+P(I)*Q(I)
      VDOT=PQ
      END

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
