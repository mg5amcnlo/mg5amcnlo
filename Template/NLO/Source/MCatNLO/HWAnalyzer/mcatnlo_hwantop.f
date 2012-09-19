C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE HWABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      REAL*4 pi
      parameter (pi=3.14160E0)
      integer j,k,ivlep1,ivlep2,idec
      common/vvlin/ivlep1,ivlep2
      character*5 cc(2)
      data cc/'     ',' cuts'/
c
c$$$      if(ivlep1.eq.7)then
c$$$        if(ivlep2.ne.7)call hwwarn('HWABEG',501)
c$$$        idec=1
c$$$      else
c$$$        if(ivlep1.gt.3.or.ivlep2.gt.3)call hwwarn('HWABEG',501)
c$$$        idec=0
c$$$      endif
c
      idec=1
      call inihist
      if(idec.eq.0)then
c Spin correlations are included
        do j=1,2
        k=(j-1)*50
          call mbook(k+ 1,'b pt     '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+ 2,'bbar pt  '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+ 3,'l+ pt    '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+ 4,'l- pt    '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+ 5,'nu pt    '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+ 6,'nubar pt '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+ 7,'b y      '//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+ 8,'bbar y   '//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+ 9,'l+ y     '//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+10,'l- y     '//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+11,'nu y     '//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+12,'nubar y  '//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+13,'bb pt    '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+14,'bb dphi  '//cc(j),pi/20.e0,0.e0,pi)
          call mbook(k+15,'bb m     '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+16,'ll pt    '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+17,'ll dphi  '//cc(j),pi/20.e0,0.e0,pi)
          call mbook(k+18,'ll m     '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+19,'bl- pt   '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+20,'bl- dphi '//cc(j),pi/20.e0,0.e0,pi)
          call mbook(k+21,'bl- m    '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+22,'bbl+ pt  '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+23,'bbl+ dphi'//cc(j),pi/20.e0,0.e0,pi)
          call mbook(k+24,'bbl+ m   '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+25,'bnub pt  '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+26,'bnub dphi'//cc(j),pi/20.e0,0.e0,pi)
          call mbook(k+27,'bnub m   '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+28,'bbnu pt  '//cc(j),2.e0,0.e0,200.e0)
          call mbook(k+29,'bbnu dphi'//cc(j),pi/20.e0,0.e0,pi)
          call mbook(k+30,'bbnu m   '//cc(j),2.e0,0.e0,200.e0)
        enddo
      elseif(idec.eq.1)then
c Spin correlations are not included
        do j=1,2
        k=(j-1)*50
          call mbook(k+ 1,'tt pt'//cc(j),2.e0,0.e0,100.e0)
          call mbook(k+ 2,'tt log[pt]'//cc(j),0.05e0,0.1e0,5.e0)
          call mbook(k+ 3,'tt inv m'//cc(j),10.e0,300.e0,1000.e0)
          call mbook(k+ 4,'tt azimt'//cc(j),pi/20.e0,0.e0,pi)
          call mbook(k+ 5,'tt del R'//cc(j),pi/20.e0,0.e0,3*pi)
          call mbook(k+ 6,'tb pt'//cc(j),5.e0,0.e0,500.e0)
          call mbook(k+ 7,'tb log[pt]'//cc(j),0.05e0,0.1e0,5.e0)
          call mbook(k+ 8,'t pt'//cc(j),5.e0,0.e0,500.e0)
          call mbook(k+ 9,'t log[pt]'//cc(j),0.05e0,0.1e0,5.e0)
          call mbook(k+10,'tt delta eta'//cc(j),0.2e0,-4.e0,4.e0)
          call mbook(k+11,'y_tt'//cc(j),0.1e0,-4.e0,4.e0)
          call mbook(k+12,'delta y'//cc(j),0.2e0,-4.e0,4.e0)
          call mbook(k+13,'tt azimt'//cc(j),pi/60.e0,2*pi/3,pi)
          call mbook(k+14,'tt del R'//cc(j),pi/60.e0,2*pi/3,4*pi/3)
          call mbook(k+15,'y_tb'//cc(j),0.1e0,-4.e0,4.e0)
          call mbook(k+16,'y_t'//cc(j),0.1e0,-4.e0,4.e0)
          call mbook(k+17,'tt log[pi-azimt]'//cc(j),0.05e0,-4.e0,0.1e0)
        enddo
        do j=1,2
        k=(j-1)*50
          call mbook(k+18,'tt pt'//cc(j),20.e0,80.e0,2000.e0)
          call mbook(k+19,'tb pt'//cc(j),20.e0,400.e0,2400.e0)
          call mbook(k+20,'t pt'//cc(j),20.e0,400.e0,2400.e0)
        enddo
      else
        call hwwarn('HWABEG',502)
      endif
      END


C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      REAL*8 XNORM
      INTEGER I,J,K,IVLEP1,IVLEP2,IDEC
      COMMON/VVLIN/IVLEP1,IVLEP2
      OPEN(UNIT=99,NAME='HERQQ.TOP',STATUS='UNKNOWN')
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
c$$$      if(ivlep1.eq.7)then
c$$$        if(ivlep2.ne.7)call hwwarn('HWAEND',501)
c$$$        idec=1
c$$$      else
c$$$        if(ivlep1.gt.3.or.ivlep2.gt.3)call hwwarn('HWAEND',501)
c$$$        idec=0
c$$$      endif
      idec=1
c
      if(idec.eq.0)then
        do j=1,2
        k=(j-1)*50
          call multitop(100+k+ 1,99,2,3,'b pt     ',' ','LOG')
          call multitop(100+k+ 2,99,2,3,'bbar pt  ',' ','LOG')
          call multitop(100+k+ 3,99,2,3,'l+ pt    ',' ','LOG')
          call multitop(100+k+ 4,99,2,3,'l- pt    ',' ','LOG')
          call multitop(100+k+ 5,99,2,3,'nu pt    ',' ','LOG')
          call multitop(100+k+ 6,99,2,3,'nubar pt ',' ','LOG')
          call multitop(100+k+ 7,99,2,3,'b y      ',' ','LOG')
          call multitop(100+k+ 8,99,2,3,'bbar y   ',' ','LOG')
          call multitop(100+k+ 9,99,2,3,'l+ y     ',' ','LOG')
          call multitop(100+k+10,99,2,3,'l- y     ',' ','LOG')
          call multitop(100+k+11,99,2,3,'nu y     ',' ','LOG')
          call multitop(100+k+12,99,2,3,'nubar y  ',' ','LOG')
          call multitop(100+k+13,99,2,3,'bb pt    ',' ','LOG')
          call multitop(100+k+14,99,2,3,'bb dphi  ',' ','LOG')
          call multitop(100+k+15,99,2,3,'bb m     ',' ','LOG')
          call multitop(100+k+16,99,2,3,'ll pt    ',' ','LOG')
          call multitop(100+k+17,99,2,3,'ll dphi  ',' ','LOG')
          call multitop(100+k+18,99,2,3,'ll m     ',' ','LOG')
          call multitop(100+k+19,99,2,3,'bl- pt   ',' ','LOG')
          call multitop(100+k+20,99,2,3,'bl- dphi ',' ','LOG')
          call multitop(100+k+21,99,2,3,'bl- m    ',' ','LOG')
          call multitop(100+k+22,99,2,3,'bbl+ pt  ',' ','LOG')
          call multitop(100+k+23,99,2,3,'bbl+ dphi',' ','LOG')
          call multitop(100+k+24,99,2,3,'bbl+ m   ',' ','LOG')
          call multitop(100+k+25,99,2,3,'bnub pt  ',' ','LOG')
          call multitop(100+k+26,99,2,3,'bnub dphi',' ','LOG')
          call multitop(100+k+27,99,2,3,'bnub m   ',' ','LOG')
          call multitop(100+k+28,99,2,3,'bbnu pt  ',' ','LOG')
          call multitop(100+k+29,99,2,3,'bbnu dphi',' ','LOG')
          call multitop(100+k+30,99,2,3,'bbnu m   ',' ','LOG')
        enddo
      elseif(idec.eq.1)then
        do j=1,2
        k=(j-1)*50
          call multitop(100+k+ 1,99,2,3,'tt pt',' ','LOG')
          call multitop(100+k+ 2,99,2,3,'tt log[pt]',' ','LOG')
          call multitop(100+k+ 3,99,2,3,'tt inv m',' ','LOG')
          call multitop(100+k+ 4,99,2,3,'tt azimt',' ','LOG')
          call multitop(100+k+ 5,99,2,3,'tt del R',' ','LOG')
          call multitop(100+k+ 6,99,2,3,'tb pt',' ','LOG')
          call multitop(100+k+ 7,99,2,3,'tb log[pt]',' ','LOG')
          call multitop(100+k+ 8,99,2,3,'t pt',' ','LOG')
          call multitop(100+k+ 9,99,2,3,'t log[pt]',' ','LOG')
          call multitop(100+k+10,99,2,3,'tt Delta eta',' ','LOG')
          call multitop(100+k+11,99,2,3,'y_tt',' ','LOG')
          call multitop(100+k+12,99,2,3,'tt Delta y',' ','LOG')
          call multitop(100+k+13,99,2,3,'tt azimt',' ','LOG')
          call multitop(100+k+14,99,2,3,'tt del R',' ','LOG')
          call multitop(100+k+15,99,2,3,'tb y',' ','LOG')
          call multitop(100+k+16,99,2,3,'t y',' ','LOG')
          call multitop(100+k+17,99,2,3,'tt log[pi-azimt]',' ','LOG')
        enddo
        do j=1,2
        k=(j-1)*50
          call multitop(100+k+18,99,2,3,'tt pt',' ','LOG')
          call multitop(100+k+19,99,2,3,'tb pt',' ','LOG')
          call multitop(100+k+20,99,2,3,'t pt',' ','LOG')
        enddo
      else
        call hwwarn('HWAEND',502)
      endif
c
      CLOSE(99)
      END


C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      DOUBLE PRECISION HWVDOT,PSUM(4)
      INTEGER ICHSUM,ICHINI,IHEP
      LOGICAL DIDSOF,flcuts,siq1flag,siq2flag,ddflag
      INTEGER ID,ID1,IST,IQ1,IQ2,IT1,IT2,ILP,INU,IBQ,ILM,INB,IBB,IJ
      DOUBLE PRECISION YCUT,PTCUT,ptlp,ylp,getrapidity,ptnu,ynu,
     # ptbq,ybq,ptlm,ylm,ptnb,ynb,ptbb,ybb,ptbqbb,dphibqbb,
     # getdelphi,xmbqbb,getinvm,ptlplm,dphilplm,xmlplm,ptbqlm,
     # dphibqlm,xmbqlm,ptbblp,dphibblp,xmbblp,ptbqnb,dphibqnb,
     # xmbqnb,ptbbnu,dphibbnu,xmbbnu,ptq1,ptq2,ptg,yq1,yq2,
     # etaq1,getpseudorap,etaq2,azi,azinorm,qqm,dr,yqq
      DOUBLE PRECISION XPTQ(5),XPTB(5),XPLP(5),XPNU(5),XPBQ(5),XPLM(5),
     # XPNB(5),XPBB(5)
      DOUBLE PRECISION YPBQBB(4),YPLPLM(4),YPBQLM(4),YPBBLP(4),
     # YPBQNB(4),YPBBNU(4),YPTQTB(4)
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
      INTEGER KK,IVLEP1,IVLEP2,IDEC
      COMMON/VVLIN/IVLEP1,IVLEP2
c

      IF (IERROR.NE.0) RETURN
c
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
      IQ1=0
      IQ2=0
c$$$      IF(IVLEP1.EQ.7)THEN
c$$$        IDEC=1
c$$$      ELSE
c$$$        IDEC=0
c$$$      ENDIF
      idec=1
      DO 100 IHEP=1,NHEP
C UNCOMMENT THE FOLLOWING WHEN REMOVING THE CHECK ON MOMENTUM 
C        IF(IQ1*IQ2.EQ.1) GOTO 11
        IF (IDHW(IHEP).EQ.16) DIDSOF=.TRUE.
        IF (ISTHEP(IHEP).EQ.1) THEN
          CALL HWVSUM(4,PHEP(1,IHEP),PSUM,PSUM)
          ICHSUM=ICHSUM+ICHRG(IDHW(IHEP))
        ENDIF
        IST=ISTHEP(IHEP)      
        ID=IDHW(IHEP)
        ID1=IDHEP(IHEP)
        IF(IST.EQ.155.AND.ID1.EQ.6)THEN
C FOUND A TOP; KEEP ONLY THE FIRST ON RECORD
          IQ1=IQ1+1
          IF(IQ1.EQ.1)IT1=IHEP
        ELSEIF(IST.EQ.155.AND.ID1.EQ.-6)THEN
C FOUND AN ANTITOP; KEEP ONLY THE FIRST ON RECORD
          IQ2=IQ2+1
          IF(IQ2.EQ.1)IT2=IHEP
        ENDIF
  100 CONTINUE
      IF(IQ1*IQ2.EQ.0.AND.IERROR.EQ.0)CALL HWWARN('HWANAL',501)
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (HWVDOT(3,PSUM,PSUM).GT.1.E-4*PHEP(4,1)**2) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',112)
         GOTO 999
      ENDIF
      IF (ICHSUM.NE.ICHINI) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',113)
         GOTO 999
      ENDIF
C FIND THE DECAY PRODUCTS IF SPIN CORRELATIONS ARE INCLUDED
      IF(IDEC.EQ.0)THEN
        ILP=JDAHEP(1,JDAHEP(1,JDAHEP(1,JDAHEP(1,JDAHEP(1,IT1)))))
        INU=JDAHEP(1,JDAHEP(2,JDAHEP(1,JDAHEP(1,JDAHEP(1,IT1)))))
        IBQ=0
        IF(JDAHEP(2,IT1)-JDAHEP(1,IT1).EQ.1)THEN
          DO IJ=JDAHEP(1,JDAHEP(1,JDAHEP(2,IT1))),
     #          JDAHEP(2,JDAHEP(1,JDAHEP(2,IT1)))
            IF(ISTHEP(IJ).EQ.2.AND.IDHEP(IJ).EQ.5)IBQ=IJ
          ENDDO
        ELSEIF(JDAHEP(2,IT1)-JDAHEP(1,IT1).EQ.2)THEN
          DO IJ=JDAHEP(1,JDAHEP(1,JDAHEP(2,IT1)-1)),
     #          JDAHEP(2,JDAHEP(1,JDAHEP(2,IT1)-1))
            IF(ISTHEP(IJ).EQ.2.AND.IDHEP(IJ).EQ.5)IBQ=IJ
          ENDDO
        ELSE
          CALL HWUEPR
          CALL HWWARN('HWANAL',502)
        ENDIF
        ILM=JDAHEP(1,JDAHEP(1,JDAHEP(1,JDAHEP(1,JDAHEP(1,IT2)))))
        INB=JDAHEP(1,JDAHEP(2,JDAHEP(1,JDAHEP(1,JDAHEP(1,IT2)))))
        IBB=0
        IF(JDAHEP(2,IT2)-JDAHEP(1,IT2).EQ.1)THEN
          DO IJ=JDAHEP(1,JDAHEP(1,JDAHEP(2,IT2))),
     #          JDAHEP(2,JDAHEP(1,JDAHEP(2,IT2)))
            IF(ISTHEP(IJ).EQ.2.AND.IDHEP(IJ).EQ.-5)IBB=IJ
          ENDDO
        ELSEIF(JDAHEP(2,IT2)-JDAHEP(1,IT2).EQ.2)THEN
          DO IJ=JDAHEP(1,JDAHEP(1,JDAHEP(2,IT2)-1)),
     #          JDAHEP(2,JDAHEP(1,JDAHEP(2,IT2)-1))
            IF(ISTHEP(IJ).EQ.2.AND.IDHEP(IJ).EQ.-5)IBB=IJ
          ENDDO
        ELSE
          CALL HWUEPR
          CALL HWWARN('HWANAL',503)
        ENDIF
C CHECK THAT THE DECAY PRODUCTS ARE WHAT THEY ARE SUPPOSED TO BE
        IF( ( (IDHEP(ILP).NE.-11.AND.IDHEP(ILP).NE.-13.AND.
     #         IDHEP(ILP).NE.-15).OR.
     #        (ISTHEP(ILP).NE.1.AND.ISTHEP(ILP).NE.195) ) .OR.
     #      ( (IDHEP(INU).NE.12.AND.IDHEP(INU).NE.14.AND.
     #         IDHEP(INU).NE.16).OR.
     #        (ISTHEP(INU).NE.1.AND.ISTHEP(INU).NE.195) ) .OR.
     #      ( IDHEP(IBQ).NE.5 .OR. ISTHEP(IBQ).NE.2 ) .OR.
     #      ( (IDHEP(ILM).NE.11.AND.IDHEP(ILM).NE.13.AND.
     #         IDHEP(ILM).NE.15).OR.
     #        (ISTHEP(ILM).NE.1.AND.ISTHEP(ILM).NE.195) ) .OR.
     #      ( (IDHEP(INB).NE.-12.AND.IDHEP(INB).NE.-14.AND.
     #         IDHEP(INB).NE.-16).OR.
     #        (ISTHEP(INB).NE.1.AND.ISTHEP(INB).NE.195) ) .OR.
     #      ( IDHEP(IBB).NE.-5 .OR. ISTHEP(IBB).NE.2 ) )THEN
          CALL HWUEPR
          CALL HWWARN('HWANAL',504)
        ENDIF
      ENDIF
C FILL THE FOUR-MOMENTA
      DO IJ=1,5
        XPTQ(IJ)=PHEP(IJ,IT1)
        XPTB(IJ)=PHEP(IJ,IT2)
        IF(IDEC.EQ.0)THEN
          XPLP(IJ)=PHEP(IJ,ILP)
          XPNU(IJ)=PHEP(IJ,INU)
          XPBQ(IJ)=PHEP(IJ,IBQ)
          XPLM(IJ)=PHEP(IJ,ILM)
          XPNB(IJ)=PHEP(IJ,INB)
          XPBB(IJ)=PHEP(IJ,IBB)
        ENDIF
      ENDDO
      IF(IDEC.EQ.0)THEN
        DO IJ=1,4
          YPBQBB(IJ)=XPBQ(IJ)+XPBB(IJ)
          YPLPLM(IJ)=XPLP(IJ)+XPLM(IJ)
          YPBQLM(IJ)=XPBQ(IJ)+XPLM(IJ)
          YPBBLP(IJ)=XPBB(IJ)+XPLP(IJ)
          YPBQNB(IJ)=XPBQ(IJ)+XPNB(IJ)
          YPBBNU(IJ)=XPBB(IJ)+XPNU(IJ)
        ENDDO
      ELSE
        DO IJ=1,4
          YPTQTB(IJ)=XPTQ(IJ)+XPTB(IJ)
        ENDDO
      ENDIF
C FILL THE HISTOS
      IF(PBEAM1.GT.2500)THEN
        YCUT=2.5D0
        PTCUT=30.D0
      ELSE
        YCUT=1.0D0
        PTCUT=15.D0
      ENDIF
c
C
      IF(IDEC.EQ.0)THEN
C
      ptlp=sqrt(xplp(1)**2+xplp(2)**2)
      ylp=getrapidity(xplp(4),xplp(3))
      ptnu=sqrt(xpnu(1)**2+xpnu(2)**2)
      ynu=getrapidity(xpnu(4),xpnu(3))
      ptbq=sqrt(xpbq(1)**2+xpbq(2)**2)
      ybq=getrapidity(xpbq(4),xpbq(3))
      ptlm=sqrt(xplm(1)**2+xplm(2)**2)
      ylm=getrapidity(xplm(4),xplm(3))
      ptnb=sqrt(xpnb(1)**2+xpnb(2)**2)
      ynb=getrapidity(xpnb(4),xpnb(3))
      ptbb=sqrt(xpbb(1)**2+xpbb(2)**2)
      ybb=getrapidity(xpbb(4),xpbb(3))
c
      ptbqbb=sqrt(ypbqbb(1)**2+ypbqbb(2)**2)
      dphibqbb=getdelphi(xpbq(1),xpbq(2),xpbb(1),xpbb(2))
      xmbqbb=getinvm(ypbqbb(4),ypbqbb(1),ypbqbb(2),ypbqbb(3))
      ptlplm=sqrt(yplplm(1)**2+yplplm(2)**2)
      dphilplm=getdelphi(xplp(1),xplp(2),xplm(1),xplm(2))
      xmlplm=getinvm(yplplm(4),yplplm(1),yplplm(2),yplplm(3))
      ptbqlm=sqrt(ypbqlm(1)**2+ypbqlm(2)**2)
      dphibqlm=getdelphi(xpbq(1),xpbq(2),xplm(1),xplm(2))
      xmbqlm=getinvm(ypbqlm(4),ypbqlm(1),ypbqlm(2),ypbqlm(3))
      ptbblp=sqrt(ypbblp(1)**2+ypbblp(2)**2)
      dphibblp=getdelphi(xpbb(1),xpbb(2),xplp(1),xplp(2))
      xmbblp=getinvm(ypbblp(4),ypbblp(1),ypbblp(2),ypbblp(3))
      ptbqnb=sqrt(ypbqnb(1)**2+ypbqnb(2)**2)
      dphibqnb=getdelphi(xpbq(1),xpbq(2),xpnb(1),xpnb(2))
      xmbqnb=getinvm(ypbqnb(4),ypbqnb(1),ypbqnb(2),ypbqnb(3))
      ptbbnu=sqrt(ypbbnu(1)**2+ypbbnu(2)**2)
      dphibbnu=getdelphi(xpbb(1),xpbb(2),xpnu(1),xpnu(2))
      xmbbnu=getinvm(ypbbnu(4),ypbbnu(1),ypbbnu(2),ypbbnu(3))
c
      flcuts=abs(ybq).le.ycut.and.ptbq.ge.ptcut .and.
     #       abs(ybb).le.ycut.and.ptbb.ge.ptcut .and.
     #       abs(ylp).le.ycut.and.ptlp.ge.ptcut .and.
     #       abs(ylm).le.ycut.and.ptlm.ge.ptcut
C
C WITHOUT CUTS
C
      kk=0
      call mfill(kk+ 1,sngl(ptbq),sngl(WWW0))
      call mfill(kk+ 2,sngl(ptbb),sngl(WWW0))
      call mfill(kk+ 3,sngl(ptlp),sngl(WWW0))
      call mfill(kk+ 4,sngl(ptlm),sngl(WWW0))
      call mfill(kk+ 5,sngl(ptnu),sngl(WWW0))
      call mfill(kk+ 6,sngl(ptnb),sngl(WWW0))
c
      call mfill(kk+ 7,sngl(ybq),sngl(WWW0))
      call mfill(kk+ 8,sngl(ybb),sngl(WWW0))
      call mfill(kk+ 9,sngl(ylp),sngl(WWW0))
      call mfill(kk+10,sngl(ylm),sngl(WWW0))
      call mfill(kk+11,sngl(ynu),sngl(WWW0))
      call mfill(kk+12,sngl(ynb),sngl(WWW0))
c
      call mfill(kk+13,sngl(ptbqbb),sngl(WWW0))
      call mfill(kk+14,sngl(dphibqbb),sngl(WWW0))
      call mfill(kk+15,sngl(xmbqbb),sngl(WWW0))
      call mfill(kk+16,sngl(ptlplm),sngl(WWW0))
      call mfill(kk+17,sngl(dphilplm),sngl(WWW0))
      call mfill(kk+18,sngl(xmlplm),sngl(WWW0))
      call mfill(kk+19,sngl(ptbqlm),sngl(WWW0))
      call mfill(kk+20,sngl(dphibqlm),sngl(WWW0))
      call mfill(kk+21,sngl(xmbqlm),sngl(WWW0))
      call mfill(kk+22,sngl(ptbblp),sngl(WWW0))
      call mfill(kk+23,sngl(dphibblp),sngl(WWW0))
      call mfill(kk+24,sngl(xmbblp),sngl(WWW0))
      call mfill(kk+25,sngl(ptbqnb),sngl(WWW0))
      call mfill(kk+26,sngl(dphibqnb),sngl(WWW0))
      call mfill(kk+27,sngl(xmbqnb),sngl(WWW0))
      call mfill(kk+28,sngl(ptbbnu),sngl(WWW0))
      call mfill(kk+29,sngl(dphibbnu),sngl(WWW0))
      call mfill(kk+30,sngl(xmbbnu),sngl(WWW0))
c
C
C WITH CUTS
C
      kk=50
      if(flcuts)then
        call mfill(kk+ 1,sngl(ptbq),sngl(WWW0))
        call mfill(kk+ 2,sngl(ptbb),sngl(WWW0))
        call mfill(kk+ 3,sngl(ptlp),sngl(WWW0))
        call mfill(kk+ 4,sngl(ptlm),sngl(WWW0))
        call mfill(kk+ 5,sngl(ptnu),sngl(WWW0))
        call mfill(kk+ 6,sngl(ptnb),sngl(WWW0))
c
        call mfill(kk+ 7,sngl(ybq),sngl(WWW0))
        call mfill(kk+ 8,sngl(ybb),sngl(WWW0))
        call mfill(kk+ 9,sngl(ylp),sngl(WWW0))
        call mfill(kk+10,sngl(ylm),sngl(WWW0))
        call mfill(kk+11,sngl(ynu),sngl(WWW0))
        call mfill(kk+12,sngl(ynb),sngl(WWW0))
c
        call mfill(kk+13,sngl(ptbqbb),sngl(WWW0))
        call mfill(kk+14,sngl(dphibqbb),sngl(WWW0))
        call mfill(kk+15,sngl(xmbqbb),sngl(WWW0))
        call mfill(kk+16,sngl(ptlplm),sngl(WWW0))
        call mfill(kk+17,sngl(dphilplm),sngl(WWW0))
        call mfill(kk+18,sngl(xmlplm),sngl(WWW0))
        call mfill(kk+19,sngl(ptbqlm),sngl(WWW0))
        call mfill(kk+20,sngl(dphibqlm),sngl(WWW0))
        call mfill(kk+21,sngl(xmbqlm),sngl(WWW0))
        call mfill(kk+22,sngl(ptbblp),sngl(WWW0))
        call mfill(kk+23,sngl(dphibblp),sngl(WWW0))
        call mfill(kk+24,sngl(xmbblp),sngl(WWW0))
        call mfill(kk+25,sngl(ptbqnb),sngl(WWW0))
        call mfill(kk+26,sngl(dphibqnb),sngl(WWW0))
        call mfill(kk+27,sngl(xmbqnb),sngl(WWW0))
        call mfill(kk+28,sngl(ptbbnu),sngl(WWW0))
        call mfill(kk+29,sngl(dphibbnu),sngl(WWW0))
        call mfill(kk+30,sngl(xmbbnu),sngl(WWW0))
c
      endif
C
      ELSEIF(IDEC.EQ.1)THEN
C
      ptq1 = dsqrt(xptq(1)**2+xptq(2)**2)
      ptq2 = dsqrt(xptb(1)**2+xptb(2)**2)
      ptg = dsqrt(yptqtb(1)**2+yptqtb(2)**2)
c  Q,Qb rapidities
      yq1=getrapidity(xptq(4),xptq(3))
      yq2=getrapidity(xptb(4),xptb(3))
c  Q,Qb pseudorapidities
      etaq1=getpseudorap(xptq(4),xptq(1),xptq(2),xptq(3))
      etaq2=getpseudorap(xptb(4),xptb(1),xptb(2),xptb(3))
c  azimuth difference
      azi=getdelphi(xptq(1),xptq(2),xptb(1),xptb(2))
      azinorm = (pi-azi)/pi
c  QQ pair mass
      qqm=getinvm(yptqtb(4),yptqtb(1),yptqtb(2),yptqtb(3))
c  QQ pair delta R
      dr  = dsqrt(azi**2+(etaq1-etaq2)**2)
c  QQ pair rapidity
      yqq=getrapidity(yptqtb(4),yptqtb(3))
c-------------------------------------------------------------
      siq1flag=ptq1.gt.ptcut.and.abs(yq1).lt.ycut
      siq2flag=ptq2.gt.ptcut.and.abs(yq2).lt.ycut
      ddflag=siq1flag.and.siq2flag
c-------------------------------------------------------------
c QQ pt
      call mfill(1,sngl(ptg),sngl(WWW0))
      call mfill(18,sngl(ptg),sngl(WWW0))
c QQ log(pt)
      if(ptg.gt.0) call mfill(2,sngl(log10(ptg)),sngl(WWW0))
c QQ invar mass
      call mfill(3,sngl(qqm),sngl(WWW0))
c QQ azimuthal difference
      call mfill(4,sngl(azi),sngl(WWW0))
      call mfill(13,sngl(azi),sngl(WWW0))
      if(azinorm.gt.0) 
     #  call mfill(17,sngl(log10(azinorm)),sngl(WWW0))
c QQ delta R
      call mfill(5,sngl(dr),sngl(WWW0))
      call mfill(14,sngl(dr),sngl(WWW0))
c QQ delta eta
      call mfill(10,sngl(etaq1-etaq2),sngl(WWW0))
c y_QQ
      call mfill(11,sngl(yqq),sngl(WWW0))
c QQ delta y
      call mfill(12,sngl(yq1-yq2),sngl(WWW0))
c Qb pt
      call mfill(6,sngl(ptq2),sngl(WWW0))
      call mfill(19,sngl(ptq2),sngl(WWW0))
c Qb log(pt)
      call mfill(7,sngl(log10(ptq2)),sngl(WWW0))
c Qb y
      call mfill(15,sngl(yq2),sngl(WWW0))
c Q pt
      call mfill(8,sngl(ptq1),sngl(WWW0))
      call mfill(20,sngl(ptq1),sngl(WWW0))
c Q log(pt)
      call mfill(9,sngl(log10(ptq1)),sngl(WWW0))
c Q y
      call mfill(16,sngl(yq1),sngl(WWW0))
c***************************************************** with cuts
      kk=50
      if(ddflag)then
c QQ pt
        call mfill(kk+1,sngl(ptg),sngl(WWW0))
        call mfill(kk+18,sngl(ptg),sngl(WWW0))
c QQ log(pt)
        if(ptg.gt.0) call mfill(kk+2,sngl(log10(ptg)),sngl(WWW0))
c QQ invar mass
        call mfill(kk+3,sngl(qqm),sngl(WWW0))
c QQ azimuthal difference
        call mfill(kk+4,sngl(azi),sngl(WWW0))
        call mfill(kk+13,sngl(azi),sngl(WWW0))
        if(azinorm.gt.0) 
     #    call mfill(kk+17,sngl(log10(azinorm)),sngl(WWW0))
c QQ delta R
        call mfill(kk+5,sngl(dr),sngl(WWW0))
        call mfill(kk+14,sngl(dr),sngl(WWW0))
c QQ delta eta
        call mfill(kk+10,sngl(etaq1-etaq2),sngl(WWW0))
c y_QQ
        call mfill(kk+11,sngl(yqq),sngl(WWW0))
c QQ delta y
        call mfill(kk+12,sngl(yq1-yq2),sngl(WWW0))
      endif
      if(abs(yq2).lt.ycut)then
c Qb pt
        call mfill(kk+6,sngl(ptq2),sngl(WWW0))
        call mfill(kk+19,sngl(ptq2),sngl(WWW0))
c Qb log(pt)
        call mfill(kk+7,sngl(log10(ptq2)),sngl(WWW0))
      endif
c Qb y
      if(ptq2.gt.ptcut)call mfill(kk+15,sngl(yq2),sngl(WWW0))
      if(abs(yq1).lt.ycut)then
c Q pt
        call mfill(kk+8,sngl(ptq1),sngl(WWW0))
        call mfill(kk+20,sngl(ptq1),sngl(WWW0))
c Q log(pt)
        call mfill(kk+9,sngl(log10(ptq1)),sngl(WWW0))
      endif
c Q y
      if(ptq1.gt.ptcut)call mfill(kk+16,sngl(yq1),sngl(WWW0))
C
      ENDIF
C
 999  RETURN
      END


      function getrapidity(en,pl)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-5)
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
