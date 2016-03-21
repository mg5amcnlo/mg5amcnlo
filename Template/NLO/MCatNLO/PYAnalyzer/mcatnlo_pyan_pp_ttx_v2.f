c
c Example analysis for "p p > t t~ [QCD]" process.
c
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG
C     USER''S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      implicit none
      include 'reweight0.inc'
      REAL*8 pi
      integer j,kk,l,i
      PARAMETER (PI=3.14159265358979312D0)
      character*5 cc(2)
      data cc/'     ','Born '/
      integer nwgt,max_weight,nwgt_analysis
      common/cnwgt/nwgt
      common/c_analysis/nwgt_analysis
      character*50 weights_info(max_weight_shower)
      common/cwgtsinfo/weights_info
c
      call inihist
      nwgt_analysis=nwgt
      do i=1,1
      do kk=1,nwgt_analysis
        l=(kk-1)*16+(i-1)*8
        call mbook(l+ 1,'total rate    '//cc(i)//weights_info(kk),
     &       1.0d0,0.5d0,5.5d0)
        call mbook(l+ 2,'t rap         '//cc(i)//weights_info(kk),
     &       0.2d0,-5d0,5d0)
        call mbook(l+ 3,'tx rap        '//cc(i)//weights_info(kk),
     &       0.2d0,-5d0,5d0)
        call mbook(l+ 4,'t-tx pair rap '//cc(i)//weights_info(kk),
     &       0.1d0,-3d0,3d0)
        call mbook(l+ 5,'m t-tx        '//cc(i)//weights_info(kk),
     &       10d0,0d0,1000d0)
        call mbook(l+ 6,'pt t          '//cc(i)//weights_info(kk),
     &       4d0,0d0,400d0)
        call mbook(l+ 7,'pt tx         '//cc(i)//weights_info(kk),
     &       4d0,0d0,400d0)
        call mbook(l+ 8,'pt t-tx       '//cc(i)//weights_info(kk),
     &       2d0,0d0,200d0)
      enddo
      enddo
      END

C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVT)
C     USER''S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      REAL*8 XNORM
      INTEGER I,J,KK,l,nwgt_analysis
      integer NPL
      parameter(NPL=15000)
      common/c_analysis/nwgt_analysis
      OPEN(UNIT=99,FILE='PYTQQ.TOP',STATUS='UNKNOWN')
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
        call multitop(NPL+l+ 1,NPL-1,3,2,'total rate   ',' ','LIN')
        call multitop(NPL+l+ 2,NPL-1,3,2,'t rap        ',' ','LOG')
        call multitop(NPL+l+ 3,NPL-1,3,2,'tx rap       ',' ','LOG')
        call multitop(NPL+l+ 4,NPL-1,3,2,'t-tx pair rap',' ','LOG')
        call multitop(NPL+l+ 5,NPL-1,3,2,'m t-tx       ',' ','LOG')
        call multitop(NPL+l+ 6,NPL-1,3,2,'pt t         ',' ','LOG')
        call multitop(NPL+l+ 7,NPL-1,3,2,'pt tx        ',' ','LOG')
        call multitop(NPL+l+ 8,NPL-1,3,2,'pt t-tx      ',' ','LOG')
      enddo
      enddo
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE PYANAL
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      implicit double precision(a-h, o-z)
      implicit integer(i-n)
      include 'reweight0.inc'
      DOUBLE PRECISION PSUM(4)
      INTEGER ICHSUM,ICHINI,IHEP
      LOGICAL flcuts,siq1flag,siq2flag,ddflag
      INTEGER ID1,IST,IQ1,IQ2,IT1,IT2,ILP,INU,IBQ,ILM,INB,IBB,IJ
      DOUBLE PRECISION YCUT,PTCUT,ptlp,ylp,getrapidity,ptnu,ynu,
     # ptbq,ybq,ptlm,ylm,ptnb,ynb,ptbb,ybb,ptbqbb,dphibqbb,
     # getdelphi,xmbqbb,getinvm,ptlplm,dphilplm,xmlplm,ptbqlm,
     # dphibqlm,xmbqlm,ptbblp,dphibblp,xmbblp,ptbqnb,dphibqnb,
     # xmbqnb,ptbbnu,dphibbnu,xmbbnu,ptq1,ptq2,ptg,yq1,yq2,
     # etaq1,getpseudorap,etaq2,azi,azinorm,qqm,dr,yqq
      DOUBLE PRECISION XPTQ(5),XPTB(5),XPLP(5),XPNU(5),XPBQ(5),XPLM(5),
     # XPNB(5),XPBB(5),p1(4),p2(4),pihep(4),p_t(0:3),p_tx(0:3),pttx(0:3),
     # mtt,pt_t,pt_tx,pt_ttx,yt,ytx,yttx,var
      DOUBLE PRECISION YPBQBB(4),YPLPLM(4),YPBQLM(4),YPBBLP(4),
     # YPBQNB(4),YPBBNU(4),YPTQTB(4)
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
      INTEGER KK,IVLEP1,IVLEP2,i,l
      COMMON/VVLIN/IVLEP1,IVLEP2
      integer nwgt_analysis,max_weight
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      double precision ww(max_weight),www(max_weight)
      common/cww/ww
c
      integer pychge
      external pydata

      common/pyjets/n,npad,k(4000,5),p(4000,5),v(4000,5)
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
      common/pydat2/kchg(500,4),pmas(500,4),parf(2000),vckm(4,4)
      common/pydat3/mdcy(500,3),mdme(8000,2),brat(8000),kfdp(8000,5)
      common/pysubs/msel,mselpd,msub(500),kfin(2,-40:40),ckin(200)
      common/pypars/mstp(200),parp(200),msti(200),pari(200)

      DOUBLE PRECISION EVWEIGHT
      COMMON/CEVWEIGHT/EVWEIGHT
      INTEGER IFAIL
      COMMON/CIFAIL/IFAIL

      IF(IFAIL.EQ.1)RETURN
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
         STOP
      ENDIF
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT''S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,P(3,3)).EQ.SIGN(1.D0,P(4,3)))THEN
         WRITE(*,*)'WARNING 111 IN PYANAL'
        GOTO 999
      ENDIF
      DO I=1,nwgt_analysis
         WWW(I)=EVWEIGHT*ww(i)/ww(1)
      ENDDO
      do i=1,4
         p1(i)=0.d0
         p2(i)=0.d0
         p1(i)=p(1,i)
         p2(i)=p(2,i)
      enddo
      CALL VVSUM(4,P1,P2,PSUM)
      CALL VSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      kf1=k(1,2)
      kf2=k(2,2)
      ICHINI=pychge(kf1)+pychge(kf2)
      IQ1=0
      IQ2=0
      DO 100 IHEP=1,N
        do j=1,4
          pihep(j)=0.d0
          pihep(j)=p(ihep,j)
        enddo
        IST=K(IHEP,1)      
        ID1=K(IHEP,2)
        IORI=K(IHEP,3)
C UNCOMMENT THE FOLLOWING WHEN REMOVING THE CHECK ON MOMENTUM 
C        IF(IQ1*IQ2.EQ.1) GOTO 11
        IF (IST.LE.10) THEN
          CALL VVSUM(4,PIHEP,PSUM,PSUM)
          ICHSUM=ICHSUM+PYCHGE(ID1)
        ENDIF
        IF(ID1.EQ.6)THEN
C FOUND A TOP; KEEP ONLY THE FIRST ON RECORD
          IQ1=IQ1+1
          IT1=IHEP
        ELSEIF(ID1.EQ.-6)THEN
C FOUND AN ANTITOP; KEEP ONLY THE FIRST ON RECORD
          IQ2=IQ2+1
          IT2=IHEP
        ENDIF
  100 CONTINUE
      IF(IQ1*IQ2.EQ.0.AND.IFAIL.EQ.0)THEN
         WRITE(*,*)'ERROR 501 IN PYANAL'
         STOP
      ENDIF
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (VDOT(3,PSUM,PSUM).GT.1.E-4*P(1,4)**2) THEN
         WRITE(*,*)'WARNING 112 IN PYANAL'
      GOTO 999
      ENDIF
      IF(IFAIL.NE.1)THEN
         IF (ICHSUM.NE.ICHINI) THEN
            WRITE(*,*)'ERROR 113 IN PYANAL'
            STOP
         ENDIF
      ENDIF
C FILL THE FOUR-MOMENTA
      DO IJ=1,5
        XPTQ(IJ)=P(IT1,IJ)
        XPTB(IJ)=P(IT2,IJ)
      ENDDO
      DO IJ=1,4
         p_t(MOD(IJ,4))=XPTQ(IJ)
         p_tx(MOD(IJ,4))=XPTB(IJ)
         pttx(MOD(IJ,4))=XPTQ(IJ)+XPTB(IJ)
      ENDDO
      mtt    = getinvm(pttx(0),pttx(1),pttx(2),pttx(3))
      pt_t   = dsqrt(p_t(1)**2 + p_t(2)**2)
      pt_tx  = dsqrt(p_tx(1)**2 + p_tx(2)**2)
      pt_ttx = dsqrt(pttx(1)**2 + pttx(2)**2)
      yt  = getrapidity(p_t(0), p_t(3))
      ytx = getrapidity(p_tx(0), p_tx(3))
      yttx= getrapidity(pttx(0), pttx(3))
      var=1.d0
      do i=1,1
         do kk=1,nwgt_analysis
            l=(kk-1)*16+(i-1)*8
            call mfill(l+1,var,www(kk))
            call mfill(l+2,yt,www(kk))
            call mfill(l+3,ytx,www(kk))
            call mfill(l+4,yttx,www(kk))
            call mfill(l+5,mtt,www(kk))
            call mfill(l+6,pt_t,www(kk))
            call mfill(l+7,pt_tx,www(kk))
            call mfill(l+8,pt_ttx,www(kk))
         enddo
      enddo
c
 999  return
      end


      function getrapidity(en,pl)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-8)
c
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
        if( (xplus/xminus).gt.tiny )then
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

