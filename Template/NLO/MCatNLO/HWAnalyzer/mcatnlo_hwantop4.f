c Identical to mcatnlo_hwantop.f, except for a few jet observables added
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
      if(ivlep1.eq.7)then
        if(ivlep2.ne.7)call hwwarn('HWABEG',501)
        idec=1
      else
        if(ivlep1.gt.3.or.ivlep2.gt.3)call hwwarn('HWABEG',501)
        idec=0
      endif
c
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
c
        do j=1,2
        k=(j-1)*50
          call mbook(k+21,'j1 pt'//cc(j),5.e0,0.e0,500.e0)
          call mbook(k+22,'j1 y'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+23,'j1 y, pt>20'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+24,'j1 y, pt>50'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+25,'j1 y, pt>100'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+26,'j1 y, pt>150'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+27,'j1 y, pt>300'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+28,'j1 eta'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+29,'j1 eta, pt>20'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+30,'j1 eta, pt>50'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+31,'j1 eta, pt>100'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+32,'j1 eta, pt>150'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+33,'j1 eta, pt>300'//cc(j),0.2e0,-5.e0,5.e0)
        enddo
        do j=1,2
        k=(j-1)*50
          call mbook(k+34,'j2 pt'//cc(j),5.e0,0.e0,500.e0)
          call mbook(k+35,'j2 y'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+36,'j2 y, pt>20'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+37,'j2 y, pt>50'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+38,'j2 y, pt>100'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+39,'j2 y, pt>150'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+40,'j2 y, pt>300'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+41,'j2 eta'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+42,'j2 eta, pt>20'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+43,'j2 eta, pt>50'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+44,'j2 eta, pt>100'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+45,'j2 eta, pt>150'//cc(j),0.2e0,-5.e0,5.e0)
          call mbook(k+46,'j2 eta, pt>300'//cc(j),0.2e0,-5.e0,5.e0)
        enddo
        do j=1,1
        k=(j-1)*50
          call mbook(k+47,'# of j1'//cc(j),1.e0,-0.5e0,10.5e0)
          call mbook(k+48,'# of j2'//cc(j),1.e0,-0.5e0,10.5e0)
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
      if(ivlep1.eq.7)then
        if(ivlep2.ne.7)call hwwarn('HWAEND',501)
        idec=1
      else
        if(ivlep1.gt.3.or.ivlep2.gt.3)call hwwarn('HWAEND',501)
        idec=0
      endif
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
c
        do j=1,2
        k=(j-1)*50
          call multitop(100+k+21,99,2,3,'j1 pt',' ','LOG')
          call multitop(100+k+22,99,2,3,'j1 y',' ','LOG')
          call multitop(100+k+23,99,2,3,'j1 y, pt>20',' ','LOG')
          call multitop(100+k+24,99,2,3,'j1 y, pt>50',' ','LOG')
          call multitop(100+k+25,99,2,3,'j1 y, pt>100',' ','LOG')
          call multitop(100+k+26,99,2,3,'j1 y, pt>150',' ','LOG')
          call multitop(100+k+27,99,2,3,'j1 y, pt>300',' ','LOG')
          call multitop(100+k+28,99,2,3,'j1 eta',' ','LOG')
          call multitop(100+k+29,99,2,3,'j1 eta, pt>20',' ','LOG')
          call multitop(100+k+30,99,2,3,'j1 eta, pt>50',' ','LOG')
          call multitop(100+k+31,99,2,3,'j1 eta, pt>100',' ','LOG')
          call multitop(100+k+32,99,2,3,'j1 eta, pt>150',' ','LOG')
          call multitop(100+k+33,99,2,3,'j1 eta, pt>300',' ','LOG')
        enddo
        do j=1,2
        k=(j-1)*50
          call multitop(100+k+34,99,2,3,'j2 pt',' ','LOG')
          call multitop(100+k+35,99,2,3,'j2 y',' ','LOG')
          call multitop(100+k+36,99,2,3,'j2 y, pt>20',' ','LOG')
          call multitop(100+k+37,99,2,3,'j2 y, pt>50',' ','LOG')
          call multitop(100+k+38,99,2,3,'j2 y, pt>100',' ','LOG')
          call multitop(100+k+39,99,2,3,'j2 y, pt>150',' ','LOG')
          call multitop(100+k+40,99,2,3,'j2 y, pt>300',' ','LOG')
          call multitop(100+k+41,99,2,3,'j2 eta',' ','LOG')
          call multitop(100+k+42,99,2,3,'j2 eta, pt>20',' ','LOG')
          call multitop(100+k+43,99,2,3,'j2 eta, pt>50',' ','LOG')
          call multitop(100+k+44,99,2,3,'j2 eta, pt>100',' ','LOG')
          call multitop(100+k+45,99,2,3,'j2 eta, pt>150',' ','LOG')
          call multitop(100+k+46,99,2,3,'j2 eta, pt>300',' ','LOG')
        enddo
        do j=1,1
        k=(j-1)*50
          call multitop(100+k+47,99,3,2,'# of j1',' ','LOG')
          call multitop(100+k+48,99,3,2,'# of j2',' ','LOG')
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
      LOGICAL DIDSOF,FLAG,flcuts,siq1flag,siq2flag,ddflag
      INTEGER ID,ID1,IST,IQ1,IQ2,IT1,IT2,ILP,INU,IBQ,ILM,INB,IBB,IJ,
     # NN,NN1,NMAX,I,JET,NJET,NSUB,IJMAX,IJET,NJMAX,JETNO,NCJET,NCY,
     # NCPHI,ICUT,IWP,IWM,IFLAG1,IFLAG2,IMO,IMOCLU,ICOUNT,IB,JB(30),
     # JB1(30),IHADR,ICLU,NJET1,NJET2,IBJET,JJ
      PARAMETER (NMAX=2000)
c WARNING: the parameters NJMAX, NCY, NCPHI are used by getjet, and set
c also there locally. Make sure the values in the analysis code coincide
c with those in getjet.f. The same applies to HEPEVT parameters
      PARAMETER (NJMAX=500)
      PARAMETER (NCY=100)
      PARAMETER (NCPHI=60)
      DOUBLE PRECISION YCUT,PTCUT,ptlp,ylp,getrapidity,ptnu,ynu,
     # ptbq,ybq,ptlm,ylm,ptnb,ynb,ptbb,ybb,ptbqbb,dphibqbb,
     # getdelphi,xmbqbb,getinvm,ptlplm,dphilplm,xmlplm,ptbqlm,
     # dphibqlm,xmbqlm,ptbblp,dphibblp,xmbblp,ptbqnb,dphibqnb,
     # xmbqnb,ptbbnu,dphibbnu,xmbbnu,ptq1,ptq2,ptg,yq1,yq2,
     # etaq1,getpseudorap,etaq2,azi,azinorm,qqm,dr,yqq,ptjcut(5)
      DOUBLE PRECISION XPTQ(5),XPTB(5),XPLP(5),XPNU(5),XPBQ(5),XPLM(5),
     # XPNB(5),XPBB(5)
      DOUBLE PRECISION YPBQBB(4),YPLPLM(4),YPBQLM(4),YPBBLP(4),
     # YPBQNB(4),YPBBNU(4),YPTQTB(4)
      DOUBLE PRECISION PP,PP1,ECUT,Y,PJET,ET1MAX,PTJET,PTCALC,PHJ1,
     # ptj1,yj1,etaj1,PHJ2,ptj2,yj2,etaj2,PCJET,ETJET,DELY,DELPHI,ET,
     # CTHCAL,STHCAL,CPHCAL,SPHCAL,YCMIN,YCMAX,PB
      DIMENSION JET(NMAX),Y(NMAX),PP(4,NMAX),PP1(4,NMAX),PJET(4,NMAX),
     # PHJ1(4),PHJ2(4),PB(4,NMAX),JJ(NMAX)
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
      COMMON/GETCOM_A/PCJET(4,NJMAX),ETJET(NJMAX),JETNO(NCY,NCPHI),NCJET
      COMMON/CALOR_A/DELY,DELPHI,ET(NCY,NCPHI),
     $CTHCAL(NCY),STHCAL(NCY),CPHCAL(NCPHI),SPHCAL(NCPHI),YCMIN,YCMAX
      INTEGER KK,IVLEP1,IVLEP2,IDEC
      COMMON/VVLIN/IVLEP1,IVLEP2
      data ptjcut/20.d0,50.d0,100.d0,150.d0,300.d0/
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
c NN is the number of final-state hadrons to be cluster into kt jets;
c their four momenta are PP(1-4,NN)
      NN=0
c NN1 is the number of final-state hadrons to be cluster into jets; such
c hadrons must not have a parent related to top decay; their four momenta 
c are PP1(1-4,NN1)
      NN1=0
c IB is the number of stable final-state b hadrons; their four momenta 
c are PB(1-4,NN). Positions in event record are JB(IB); positions in
c particles to be clustered into jets is JB1(IB)
      IB=0
      IF(IVLEP1.EQ.7)THEN
        IDEC=1
      ELSE
        IDEC=0
      ENDIF
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
C FIND FINAL STATE HADRONS, TO BE CLUSTERED INTO JETS
        ELSEIF(IST.EQ.1.AND.ABS(ID1).GT.100)THEN
          NN=NN+1
          IF (NN.GT.NMAX)CALL HWWARN('HWANAL',505)
          DO I=1,4
            PP(I,NN)=PHEP(I,IHEP)
          ENDDO
          JJ(NN)=IHEP
          IF(IHADR(ID1).EQ.5)THEN
C FOUND A B-FLAVOURED HADRON
            IB=IB+1
            JB(IB)=IHEP
            JB1(IB)=NN
            DO I=1,4
              PB(I,IB)=PHEP(I,IHEP)
            ENDDO
          ENDIF
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
      IF(IDEC.EQ.0)THEN
C FIND THE DECAY PRODUCTS IF SPIN CORRELATIONS ARE INCLUDED
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
      ELSE
C FIND THE DECAY PRODUCTS IF SPIN CORRELATIONS ARE NOT INCLUDED
        IWP=JDAHEP(1,IT1)
        IBQ=JDAHEP(2,IT1)
        IF(JDAHEP(2,IT1)-JDAHEP(1,IT1).EQ.2)THEN
          IF(IDHEP(JDAHEP(1,IT1)+1).EQ.5)IBQ=JDAHEP(1,IT1)+1
        ENDIF
        IWM=JDAHEP(1,IT2)
        IBB=JDAHEP(2,IT2)
        IF(JDAHEP(2,IT2)-JDAHEP(1,IT2).EQ.2)THEN
          IF(IDHEP(JDAHEP(1,IT2)+1).EQ.-5)IBB=JDAHEP(1,IT2)+1
        ENDIF
C CHECK THAT THE DECAY PRODUCTS ARE WHAT THEY ARE SUPPOSED TO BE
        IF( ( IDHEP(IWP).NE.24 .OR. ISTHEP(IWP).NE.123 ) .OR.
     #      ( IDHEP(IBQ).NE.5 .OR. ISTHEP(IBQ).NE.124 ) .OR.
     #      ( IDHEP(IWM).NE.-24 .OR. ISTHEP(IWM).NE.123 ) .OR.
     #      ( IDHEP(IBB).NE.-5 .OR. ISTHEP(IBB).NE.124 ) )THEN
          CALL HWUEPR
          WRITE(*,*)IT1,IWP,IBQ,IT2,IWM,IBB
          CALL HWWARN('HWANAL',504)
        ENDIF
C CHECK THAT Ws DECAY INTO E NU_E: ELECTRONS AND NEUTRINOS MAY ALSO BE
C EXCHANGED IN THE EVENT RECORDS, IT DEPENDS ON THE CALL TO HWMODK
        ILP=JDAHEP(2,JDAHEP(1,IWP))
        INU=JDAHEP(1,JDAHEP(1,IWP))
        ILM=JDAHEP(2,JDAHEP(1,IWM))
        INB=JDAHEP(1,JDAHEP(1,IWM))
        IF( IDHEP(ILP).NE.-11 .OR. IDHEP(INU).NE.12 .OR.
     #      IDHEP(ILM).NE.11 .OR. IDHEP(INB).NE.-12 )THEN
          CALL HWUEPR
          WRITE(*,*)IWP,ILP,INU
          WRITE(*,*)IWM,ILM,INB
          CALL HWWARN('HWANAL',507)
        ENDIF
C FIND FINAL STATE HADRONS, TO BE CLUSTERED INTO JETS; EXCLUDE PARTONS
C FROM TOP DECAY PRODUCTS
        DO IHEP=4,NHEP
          IST=ISTHEP(IHEP)      
          ID1=IDHEP(IHEP)
          IF(IST.EQ.1.AND.ABS(ID1).GT.100)THEN
            IFLAG1=1
            IFLAG2=1
            ICOUNT=0
            ICLU=0
            IMO=IHEP
            DOWHILE(ICLU.EQ.0)
              ICOUNT=ICOUNT+1
              IF(ICOUNT.GT.NHEP)CALL HWWARN('HWANAL',506)
              IMO=JMOHEP(1,IMO)
              IF(IDHEP(IMO).EQ.91)ICLU=1
            ENDDO
            IMOCLU=IMO
C FOUND A CLUSTER, FOLLOW ITS FIRST MOTHER
            ICOUNT=0
            FLAG=.TRUE.
            DOWHILE(FLAG)
              ICOUNT=ICOUNT+1
              IF(ICOUNT.GT.NHEP)CALL HWWARN('HWANAL',508)
              IMO=JMOHEP(1,IMO)
              IF( ISTHEP(IMO).GE.121.AND.ISTHEP(IMO).LE.124 .AND.
     #            IMO.LE.9 )THEN
                FLAG=.FALSE.
              ELSEIF( IMO.EQ.IBQ .OR. IMO.EQ.IBB )THEN
                FLAG=.FALSE.
                IFLAG1=-1
              ENDIF
            ENDDO
C FOUND A CLUSTER, FOLLOW ITS SECOND MOTHER
            IMO=IMOCLU
            ICOUNT=0
            FLAG=.TRUE.
            DOWHILE(FLAG)
              ICOUNT=ICOUNT+1
              IF(ICOUNT.GT.NHEP)CALL HWWARN('HWANAL',509)
              IF(ICOUNT.EQ.1)THEN
                IMO=JMOHEP(2,IMO)
              ELSE
                IMO=JMOHEP(1,IMO)
              ENDIF
              IF( ISTHEP(IMO).GE.121.AND.ISTHEP(IMO).LE.124 .AND.
     #            IMO.LE.9 )THEN
                FLAG=.FALSE.
              ELSEIF( IMO.EQ.IBQ .OR. IMO.EQ.IBB )THEN
                FLAG=.FALSE.
                IFLAG2=-1
              ENDIF
            ENDDO
C KEEP ONLY HADRONS NOT COMING FROM B'S EMERGING FROM TOPS
            IF(IFLAG1.EQ.1.AND.IFLAG2.EQ.1)THEN
              NN1=NN1+1
              IF (NN1.GT.NMAX)CALL HWWARN('HWANAL',510)
              DO I=1,4
                PP1(I,NN1)=PHEP(I,IHEP)
              ENDDO
            ENDIF
          ENDIF
        ENDDO
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
      IF(PBEAM1.GT.5000)THEN
        YCUT=2.5D0
        PTCUT=30.D0
      ELSE
        YCUT=1.0D0
        PTCUT=15.D0
      ENDIF
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
c
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
c-------------------------------------------------------------
c
c Jet clustering: hadrons from b partons from top decays are excluded
c
C---CLUSTER THE EVENT (DEFINING ECUT AS 1, SO Y IS MEASURED IN GEV**2)
C   USING THE COVARIANT E-SCHEME
C KTCLUS clusters according to the usual kt-clustering scheme, as
c proposed in Nucl.Phys.B406(1993)187. This clustering is the same
c as that adopted by Ellis&Soper; the call to KTCLUS corresponds to
c Ellis&Soper with D=1; for other values of D, call KTCLUR instead,
c with R==D (with minor differences, see the comments in the package)
      NJET=0
      IF(NN1.LT.1)GOTO 123
      ECUT=1
      CALL KTCLUS(5,PP1,NN1,ECUT,Y,*123)
C---RECONSTRUCT THE MOMENTA OF THE JETS USING THE E-SCHEME (IE SIMPLE 
C   VECTOR ADDITION); only those jets with d>YCUT are kept (as opposed
c   to a list of all jets one would get by calling KTINCL). When ECUT=1,
c   YCUT has units GeV^2
      YCUT=100.d0
      CALL KTRECO(1,PP1,NN1,ECUT,YCUT,YCUT,PJET,JET,NJET,NSUB,*123)
C It appears that, when using the above value of YCUT, the jets
c reconstructed by KTRECO are NOT ordered in such a way that the
c hardest jet is the NJETth (i.e. the last of the list). To be on
c the safe side, we order the jets by hand -- the hardness is defined 
c by means of the transverse energy.
c In this analysis, we keep only the hardest jet
      IJMAX=0
      ET1MAX=0.d0
      DO IJET=1,NJET
        PTJET=PTCALC(PJET(1,IJET))
        IF(PTJET.GT.ET1MAX)THEN
          IJMAX=IJET
          ET1MAX=PTJET
        ENDIF
      ENDDO
      if(njet.ne.0)then
        do i=1,4
          phj1(i)=pjet(i,ijmax)
        enddo
        ptj1=sqrt(phj1(1)**2+phj1(2)**2)
        yj1=getrapidity(phj1(4),phj1(3))
        etaj1=getpseudorap(phj1(4),phj1(1),phj1(2),phj1(3))
      endif
c
c jet observables
c
 123  continue
      call mfill(47,float(njet),sngl(WWW0))
      if(njet.ne.0)then
        call mfill(21,sngl(ptj1),sngl(WWW0))
        call mfill(22,sngl(yj1),sngl(WWW0))
        call mfill(28,sngl(etaj1),sngl(WWW0))
        icut=0
        do i=1,5
          if(ptj1.gt.ptjcut(i))icut=i
        enddo
        do i=1,icut
          call mfill(22+i,sngl(yj1),sngl(WWW0))
          call mfill(28+i,sngl(etaj1),sngl(WWW0))
        enddo
      endif
c
c***************************************************** with cuts
      kk=50
c
c jet observables
c
      if(ddflag)then
        if(njet.ne.0)then
          call mfill(kk+21,sngl(ptj1),sngl(WWW0))
          call mfill(kk+22,sngl(yj1),sngl(WWW0))
          call mfill(kk+28,sngl(etaj1),sngl(WWW0))
          icut=0
          do i=1,5
            if(ptj1.gt.ptjcut(i))icut=i
          enddo
          do i=1,icut
            call mfill(kk+22+i,sngl(yj1),sngl(WWW0))
            call mfill(kk+28+i,sngl(etaj1),sngl(WWW0))
          enddo
        endif
      endif
C
c-------------------------------------------------------------
c
c Jet clustering: all hadrons
c
C---CLUSTER THE EVENT (DEFINING ECUT AS 1, SO Y IS MEASURED IN GEV**2)
C   USING THE COVARIANT E-SCHEME
C KTCLUS clusters according to the usual kt-clustering scheme, as
c proposed in Nucl.Phys.B406(1993)187. This clustering is the same
c as that adopted by Ellis&Soper; the call to KTCLUS corresponds to
c Ellis&Soper with D=1; for other values of D, call KTCLUR instead,
c with R==D (with minor differences, see the comments in the package)
      NJET=0
      NJET2=0
      IF(NN.LT.1)GOTO 124
      ECUT=1
      CALL KTCLUS(5,PP,NN,ECUT,Y,*124)
C---RECONSTRUCT THE MOMENTA OF THE JETS USING THE E-SCHEME (IE SIMPLE 
C   VECTOR ADDITION); only those jets with d>YCUT are kept (as opposed
c   to a list of all jets one would get by calling KTINCL). When ECUT=1,
c   YCUT has units GeV^2
      YCUT=100.d0
      CALL KTRECO(1,PP,NN,ECUT,YCUT,YCUT,PJET,JET,NJET,NSUB,*124)
C It appears that, when using the above value of YCUT, the jets
c reconstructed by KTRECO are NOT ordered in such a way that the
c hardest jet is the NJETth (i.e. the last of the list). To be on
c the safe side, we order the jets by hand -- the hardness is defined 
c by means of the transverse energy.
c In this analysis, we keep only the hardest jet
      CALL KTWICH(ECUT,YCUT,JET,NJET1,*124)
      IF(NJET1.NE.NJET)THEN
        CALL HWUEPR
        CALL HWWARN('HWANAL',511)
      ENDIF
c Find the hardest jet, but keeping only those which are not b-jets
      IJMAX=0
      ET1MAX=0.d0
      DO IJET=1,NJET
        PTJET=PTCALC(PJET(1,IJET))
        IF(PTJET.GT.ET1MAX)THEN
          IBJET=0
          DO I=1,IB
            IF(IJET.EQ.JET(JB1(I)))IBJET=1
          ENDDO
          IF(IBJET.EQ.0)THEN
            NJET2=NJET2+1
            IJMAX=IJET
            ET1MAX=PTJET
          ENDIF
        ENDIF
      ENDDO
      if(njet2.ne.0)then
        do i=1,4
          phj2(i)=pjet(i,ijmax)
        enddo
        ptj2=sqrt(phj2(1)**2+phj2(2)**2)
        yj2=getrapidity(phj2(4),phj2(3))
        etaj2=getpseudorap(phj2(4),phj2(1),phj2(2),phj2(3))
      endif
c
c jet observables
c
 124  continue
      call mfill(48,float(njet2),sngl(WWW0))
      if(njet2.ne.0)then
        call mfill(34,sngl(ptj2),sngl(WWW0))
        call mfill(35,sngl(yj2),sngl(WWW0))
        call mfill(41,sngl(etaj2),sngl(WWW0))
        icut=0
        do i=1,5
          if(ptj2.gt.ptjcut(i))icut=i
        enddo
        do i=1,icut
          call mfill(35+i,sngl(yj2),sngl(WWW0))
          call mfill(41+i,sngl(etaj2),sngl(WWW0))
        enddo
      endif
c
c***************************************************** with cuts
      kk=50
c
c jet observables
c
      if(ddflag)then
        if(njet2.ne.0)then
          call mfill(kk+34,sngl(ptj2),sngl(WWW0))
          call mfill(kk+35,sngl(yj2),sngl(WWW0))
          call mfill(kk+41,sngl(etaj2),sngl(WWW0))
          icut=0
          do i=1,5
            if(ptj2.gt.ptjcut(i))icut=i
          enddo
          do i=1,icut
            call mfill(kk+35+i,sngl(yj2),sngl(WWW0))
            call mfill(kk+41+i,sngl(etaj2),sngl(WWW0))
          enddo
        endif
      endif
C
      ENDIF
C
 999  RETURN
      END


      FUNCTION IHADR(ID)
c Returns the PDG code of the heavier quark in the hadron of PDG code ID
      IMPLICIT NONE
      INTEGER IHADR,ID,ID1
C
      IF(ID.NE.0)THEN
        ID1=ABS(ID)
        IF(ID1.GT.10000)ID1=ID1-1000*INT(ID1/1000)
        IHADR=ID1/(10**INT(LOG10(DFLOAT(ID1))))
      ELSE
        IHADR=0
      ENDIF
      RETURN
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


      FUNCTION PTCALC(P)
      IMPLICIT NONE
      DOUBLE PRECISION PTCALC,P(4),PTSQ
      PTSQ=P(1)**2+P(2)**2
      IF (PTSQ.EQ.0D0) THEN
         PTCALC=0D0
      ELSE
         PTCALC=SQRT(PTSQ)
      ENDIF
      END
