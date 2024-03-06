
      DOUBLE PRECISION FUNCTION
     .                    SQSUSY_HDEC(IHIGGS,ISQ,II,JJ,QQ,KONSH,KDTH)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMPLEX*16 C03_HDEC
      DOUBLE PRECISION LAMB_HDEC,MIJ,MIJ0
      DIMENSION XGLBB(2,2),XGHBB(2,2),XGCTB(2,2)
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/HMASS_HDEC/AMSM,AMA,AMHL,AMHH,AMCH,AMAR
      COMMON/SQNLO_HDEC/AMSB(2),STHB,CTHB,GLBB(2,2),GHBB(2,2),GABB,
     .                  AMST(2),STHT,CTHT,GLTT(2,2),GHTT(2,2),GATT
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/SFER1ST_HDEC/AMQL1,AMUR1,AMDR1,AMEL1,AMER1
      COMMON/BREAK_HDEC/AMEL,AMER,AMSQ,AMUR,AMDR,AL,AU0,AD0,AMU,AM20
      COMMON/TRILINEAR_HDEC/AU,AD
      COMMON/BREAKSCALE_HDEC/SUSYSCALE
      COMMON/SQUARKHIGGS_HDEC/THEB,AMG,IONSH,IDTH
      COMMON/ALS_HDEC/XLAMBDA0,AMC0,AMB0,AMT0,N00
      XKAP(AM1,AM2,AM3) = LAMB_HDEC(AM2**2/AM1**2,AM3**2/AM1**2)*AM1**2
      BETS(AM1,AM2,AM3) = DSQRT(1-4*AM1*AM2/(AM3**2-(AM1-AM2)**2))
      XS(AM1,AM2,AM3) = (1-BETS(AM1,AM2,AM3))/(1+BETS(AM1,AM2,AM3))
      BB1(P2,AM1,AM2,XMU)=((AM1**2-AM2**2)*B02_HDEC(0.D0,AM1,AM2,XMU**2)
     .           -(P2+AM1**2-AM2**2)*B02_HDEC(P2,AM1,AM2,XMU**2))/2/P2
      CC0(AM1,AM2,AM3,XMU) =
     . (DLOG(XS(AM1,AM2,AM3))*(DLOG(XMU**2/AM1/AM2)
     . + DLOG(XS(AM1,AM2,AM3))/2
     . - 2*DLOG(1-XS(AM1,AM2,AM3)**2)) - 2*ZETA2 - DLOG(AM1/AM2)**2/2
     . - SP_HDEC(XS(AM1,AM2,AM3)**2)-SP_HDEC(1+AM2/AM1*XS(AM1,AM2,AM3))
     . - SP_HDEC(1+AM1/AM2*XS(AM1,AM2,AM3)))/XKAP(AM1,AM2,AM3)
      PMSQ1(QQ,AM,AMG,ALS) = CF*ALS/PI*(AMG**2*DLOG(QQ**2/AMG**2)
     . + AM**2/2*DLOG(AMG**2/AM**2) + AM**2/2 + 3*AMG**2/2
     . + (AMG**2-AM**2)**2/2/AM**2*DLOG(DABS(AMG**2-AM**2)/AMG**2))
      PMSQ10(QQ,AM,ALS) = CF*ALS/PI*(AM**2*DLOG(QQ**2/AM**2)
     . + AM**2/2 + 3*AM**2/2)
      ALS_SUSY(X,XLB,B0,B1)=12.D0*PI/(B0*DLOG(X**2/XLB**2))
     .          *(1.D0-B1*DLOG(DLOG(X**2/XLB**2))
     .           /DLOG(X**2/XLB**2))
c     write(6,*)ISQ
c     write(6,*)'H-masses:  ',AMHL,AMHH,AMA,DSIN(A),DCOS(A)
c     write(6,*)'sq-masses: ',AMST(1),AMST(2),AMSB(1),AMSB(2)
c     write(6,*)'scale:     ',QQ
c     write(6,*)'h: stops ',AMZ**2*GLTT(1,1),AMZ**2*GLTT(2,2),
c    .                      AMZ**2*GLTT(1,2)
c     write(6,*)'H: stops ',AMZ**2*GHTT(1,1),AMZ**2*GHTT(2,2),
c    .                      AMZ**2*GHTT(1,2)
c     write(6,*)'A: stops ',AMZ**2*GATT
c     write(6,*)'h: sbot  ',AMZ**2*GLBB(1,1),AMZ**2*GLBB(2,2),
c    .                      AMZ**2*GLBB(1,2)
c     write(6,*)'H: sbot  ',AMZ**2*GHBB(1,1),AMZ**2*GHBB(2,2),
c    .                      AMZ**2*GHBB(1,2)
c     write(6,*)'A: sbot  ',AMZ**2*GABB
c     write(6,*)
c     write(6,*)'sin/cos(thet_t):  ',STHT,CTHT
c     write(6,*)'sin/cos(thet_b):  ',STHB,CTHB
c     write(6,*)'thet_t/b:',DATAN(STHT/CTHT),DATAN(STHB/CTHB)
c     write(6,*)
      CF = 4.D0/3.D0
      CA = 3
      PI = 4*DATAN(1.D0)
      TB = DTAN(B)
      ZETA2 = PI**2/6
      XMU   = QQ
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     XMU   = SUSYSCALE
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      IONSH = KONSH
      IDTH = KDTH
c     CALL SQMBAPP_HDEC(QQ)
      DELTA_B = 2*AMG*AMU*DTAN(B)*T_HDEC(AMSB(1),AMSB(2),AMG)
c     DELTA_T = 2*AMG*AMU/DTAN(B)*T_HDEC(AMST(1),AMST(2),AMG)
      DELTA_T = 0
      DELSUSY = 1
c     AMSUSY = DSQRT(2*AMSQ**2+AMUR**2+AMDR**2)/2
      AMSUSY = SUSYSCALE
      QQ0 = AMSUSY
      ALS = ALPHAS_HDEC(QQ0,3)
      SW2=1.D0-AMW**2/AMZ**2
C UP SQUARKS: 
      AMSQ0 = DSQRT(AMQL1**2+(0.5D0-2.D0/3.D0*SW2)*AMZ**2*DCOS(2.D0*B))
      IF(DABS(AMG).NE.AMSQ0)THEN
       AMU1L = DSQRT(AMSQ0**2 + PMSQ1(QQ0,AMSQ0,AMG,ALS))
      ELSE
       AMU1L = DSQRT(AMSQ0**2 + PMSQ10(QQ0,AMSQ0,ALS))
      ENDIF
      AMSQ0 = DSQRT(AMUR1**2+2.D0/3.D0*SW2*AMZ**2*DCOS(2.D0*B))
      IF(DABS(AMG).NE.AMSQ0)THEN
       AMU1R = DSQRT(AMSQ0**2 + PMSQ1(QQ0,AMSQ0,AMG,ALS))
      ELSE
       AMU1R = DSQRT(AMSQ0**2 + PMSQ10(QQ0,AMSQ0,ALS))
      ENDIF
C DOWN SQUARKS
      AMSQ0 = DSQRT(AMQL1**2+(-0.5D0+1.D0/3.D0*SW2)*AMZ**2*DCOS(2.D0*B))
      IF(DABS(AMG).NE.AMSQ0)THEN
       AMD1L = DSQRT(AMSQ0**2 + PMSQ1(QQ0,AMSQ0,AMG,ALS))
      ELSE
       AMD1L = DSQRT(AMSQ0**2 + PMSQ10(QQ0,AMSQ0,ALS))
      ENDIF
      AMSQ0 = DSQRT(AMDR1**2-1.D0/3.D0*SW2*AMZ**2*DCOS(2.D0*B))
      IF(DABS(AMG).NE.AMSQ0)THEN
       AMD1R = DSQRT(AMSQ0**2 + PMSQ1(QQ0,AMSQ0,AMG,ALS))
      ELSE
       AMD1R = DSQRT(AMSQ0**2 + PMSQ10(QQ0,AMSQ0,ALS))
      ENDIF
      EPS = 0
      FFB = AMB*2*STHB*CTHB/AMG
      FFT = AMT*2*STHT*CTHT/AMG
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     FFB = 0
c     FFT = 0
c     aa1 = BB1(AMG**2,AMT,AMST(2),QQ0)
c     aa2 =-BB1(AMG**2,AMST(2),AMT,QQ0)
c    .     -B02_HDEC(AMG**2,AMST(2),AMT,QQ0**2)
c     write(6,*)'ratio = ',aa1,aa2,aa1/aa2
c     write(6,*)'q0  = ',qq0
c     write(6,*)'mg  = ',amg
c     write(6,*)'mb,mt = ',amb,amt
c     write(6,*)'msb = ',amsb
c     write(6,*)'mst = ',amst
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      AM3 = AMG*(1-ALPHAS_HDEC(DABS(AMG),3)/4/PI
     .         *(4*CA+3*CA*DLOG(QQ0**2/AMG**2)
     .          + BB1(AMG**2,EPS,AMU1L,QQ0)
     .          + BB1(AMG**2,EPS,AMU1R,QQ0)
     .          + BB1(AMG**2,EPS,AMD1L,QQ0)
     .          + BB1(AMG**2,EPS,AMD1R,QQ0)
     .          + BB1(AMG**2,AMS,AMD1L,QQ0)
     .          + BB1(AMG**2,AMS,AMD1R,QQ0)
     .          + BB1(AMG**2,AMC,AMU1L,QQ0)
     .          + BB1(AMG**2,AMC,AMU1R,QQ0)
     .          + BB1(AMG**2,AMB,AMSB(1),QQ0)
     .          + BB1(AMG**2,AMB,AMSB(2),QQ0)
     .          + BB1(AMG**2,AMT,AMST(1),QQ0)
     .          + BB1(AMG**2,AMT,AMST(2),QQ0)
     .          + FFB*(B02_HDEC(AMG**2,AMB,AMSB(1),QQ0**2)
     .                -B02_HDEC(AMG**2,AMB,AMSB(2),QQ0**2))
     .          + FFT*(B02_HDEC(AMG**2,AMT,AMST(1),QQ0**2)
     .                -B02_HDEC(AMG**2,AMT,AMST(2),QQ0**2))
     .          ))
      ALS_SUSY0= ALPHAS_HDEC(QQ0,3)*(1+ALPHAS_HDEC(QQ0,3)/PI
     .         * (DLOG(QQ0**2/AMT**2)/6 + DLOG(QQ0**2/AMG**2)/2
     .         + (2*(DLOG(QQ0**2/AMU1L**2)+DLOG(QQ0**2/AMU1R**2)
     .              +DLOG(QQ0**2/AMD1L**2)+DLOG(QQ0**2/AMD1R**2))
     .           +DLOG(QQ0**2/AMSB(1)**2)+DLOG(QQ0**2/AMSB(2)**2)
     .           +DLOG(QQ0**2/AMST(1)**2)+DLOG(QQ0**2/AMST(2)**2))/24))
c     write(6,*)'params: ',ALS_SUSY0,ALPHAS_HDEC(QQ0,3),AMT,AMG,
c    .  AMU1L,AMU1R,AMD1L,AMD1R,AMSB(1),AMSB(2),AMST(1),AMST(2)
      ACC = 1.D-10
      XLB_SUSY = XITSUSY_HDEC(QQ0,ALS_SUSY0,ACC)
      B0 = 9
      B1 =-14.D0/9
      ALS0 = ALS_SUSY(QQ0,XLB_SUSY,B0,B1)
      ALS1 = ALS_SUSY(QQ,XLB_SUSY,B0,B1)
      AU = AU0 + AM3*(-16.D0/9*(ALS1/ALS0-1)*(1-7*ALS0/12/PI)
     .               -16*ALS0/27/PI*(ALS1**2/ALS0**2-1))
      AD = AD0 + AM3*(-16.D0/9*(ALS1/ALS0-1)*(1-7*ALS0/12/PI)
     .               -16*ALS0/27/PI*(ALS1**2/ALS0**2-1))
c     AU1= AU0 + CF*ALPHAS_HDEC(QQ,3)/PI*AM3*DLOG(QQ**2/QQ0**2)
c     AD1= AD0 + CF*ALPHAS_HDEC(QQ,3)/PI*AM3*DLOG(QQ**2/QQ0**2)
c     write(6,*)'scale = ',QQ
c     write(6,*)'m_s, m_c, eps =   ',AMS,AMC,EPS
c     write(6,*)'alpha_s (M_Z) =   ',ALPHAS_HDEC(AMZ,3)
c     write(6,*)'alpha_s (Q_0) =   ',ALPHAS_HDEC(QQ0,3)
c     write(6,*)'alphas iteration: ',ALS_SUSY0,ALS0,ALS0/ALS_SUSY0
c     write(6,*)'Lambda iteration: ',XLB_SUSY,XLAMBDA0
c     write(6,*)'MUL, MUR: ',AMU1L,AMU1R
c     write(6,*)'MDL, MDR: ',AMD1L,AMD1R
c     write(6,*)'Mg, M3: ',AMG,AM3
c     write(6,*)'alphas in A_q: ',ALS1,ALS0,QQ,QQ0
c     write(6,*)'A_t, A_b: ',AU,AD
c     write(6,*)'      LL: ',AU1,AD1
c     write(6,*)'      LO: ',AU0,AD0
      CALL SQMBAPP_HDEC(QQ)
      IF(ISQ.EQ.1)THEN
       AQSC  = QQ0
       AMQSC = AQSC
       ALSSC = (AMST(1)+AMST(2)+DABS(AMG))/3
       AMQ = AMT
       IF(IONSH.EQ.1)THEN
        RMQ = AMT
       ELSE
        RMQ =RUNM_HDEC(QQ,6)
       ENDIF
c      write(6,*)'tmass:   ',RMQ,RUNM_HDEC(QQ,6)
       AM1 = AMST(1)
       AM2 = AMST(2)
       STH = STHT
       CTH = CTHT
       S2T = 2*STH*CTH
       C2T = CTH**2-STH**2
       DELTA_Q = DELTA_T
       RQ = 1/DTAN(B)
       AQQ = AU
       IF(IHIGGS.EQ.1)THEN
        AMH = AMHL
        GQ = GLT
        GLX = 1/AMZ**2
        G11 = GLTT(1,1)/GLX
        G12 = GLTT(1,2)/GLX
        G21 = GLTT(2,1)/GLX
        G22 = GLTT(2,2)/GLX
        GLR = G12*(CTH**2-STH**2)+(G11-G22)*STH*CTH
       ELSEIF(IHIGGS.EQ.2)THEN
        AMH = AMHH
        GQ = GHT
        GLX = 1/AMZ**2
        G11 = GHTT(1,1)/GLX
        G12 = GHTT(1,2)/GLX
        G21 = GHTT(2,1)/GLX
        G22 = GHTT(2,2)/GLX
        GLR = G12*(CTH**2-STH**2)+(G11-G22)*STH*CTH
       ELSEIF(IHIGGS.EQ.3)THEN
        AMH = AMA
        GQ = GAT
        GLX = 1/AMZ**2
        G11 = 0
        G12 = GATT/GLX
        G21 =-GATT/GLX
        G22 = 0
        GLR = G12
       ENDIF
      ELSE
       AQSC  = QQ0
       AMQSC = AQSC
       ALSSC = (AMSB(1)+AMSB(2)+DABS(AMG))/3
       AMQ = AMB
       IF(IONSH.EQ.1)THEN
        RMQ = AMB
       ELSE
        RMQ =RUNM_HDEC(QQ,5)/(1+CF/4*ALPHAS_HDEC(ALSSC,3)/PI*DELTA_B)
       ENDIF
c      write(6,*)'bmass:   ',RMQ,RUNM_HDEC(QQ,5)
       AM1 = AMSB(1)
       AM2 = AMSB(2)
       STH = STHB
       CTH = CTHB
       S2T = 2*STH*CTH
       C2T = CTH**2-STH**2
       DELTA_Q = DELTA_B
       RQ = DTAN(B)
       AQQ = AD
       IF(IHIGGS.EQ.1)THEN
        AMH = AMHL
        GQ = GLB
        GLX = 1/AMZ**2
        G11 = GLBB(1,1)/GLX
        G12 = GLBB(1,2)/GLX
        G21 = GLBB(2,1)/GLX
        G22 = GLBB(2,2)/GLX
        GLR = G12*(CTH**2-STH**2)+(G11-G22)*STH*CTH
       ELSEIF(IHIGGS.EQ.2)THEN
        AMH = AMHH
        GQ = GHB
        GLX = 1/AMZ**2
        G11 = GHBB(1,1)/GLX
        G12 = GHBB(1,2)/GLX
        G21 = GHBB(2,1)/GLX
        G22 = GHBB(2,2)/GLX
        GLR = G12*(CTH**2-STH**2)+(G11-G22)*STH*CTH
       ELSEIF(IHIGGS.EQ.3)THEN
        AMH = AMA
        GQ = GAB
        GLX = 1/AMZ**2
        G11 = 0
        G12 = GABB/GLX
        G21 =-GABB/GLX
        G22 = 0
        GLR = G12
       ENDIF
      ENDIF
      BTGQ = (B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
     .       +B02_HDEC(AM2**2,AMG,RMQ,XMU**2))/2
      B1GQ = B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
      B2GQ = B02_HDEC(AM2**2,AMG,RMQ,XMU**2)
      B11 = B02_HDEC(AMH**2,AM1,AM1,XMU**2)
      B12 = B02_HDEC(AMH**2,AM1,AM2,XMU**2)
      B22 = B02_HDEC(AMH**2,AM2,AM2,XMU**2)
      IF(II.EQ.1.AND.JJ.EQ.1)THEN
       GLO = G11
       AMI = AM1
       AMJ = AM1
       SGI = -1
       SGJ = -1
       DIJ = 1
       SIJ = S2T
       TIJ = C2T**2*G11*B11 + S2T**2*G22*B22 - 2*S2T*C2T*G12*B12
      ELSEIF(II.EQ.1.AND.JJ.EQ.2)THEN
       GLO = G12
       AMI = AM1
       AMJ = AM2
       SGI = -1
       SGJ = 1
       DIJ = 0
       SIJ = C2T
       TIJ = -S2T*C2T*(G11*B11-G22*B22) - (C2T**2-S2T**2)*G12*B12
      ELSEIF(II.EQ.2.AND.JJ.EQ.1)THEN
       GLO = G21
       AMI = AM2
       AMJ = AM1
       SGI = 1
       SGJ = -1
       DIJ = 0
       SIJ = C2T
       TIJ = -S2T*C2T*(G11*B11-G22*B22) - (C2T**2-S2T**2)*G12*B12
      ELSEIF(II.EQ.2.AND.JJ.EQ.2)THEN
       GLO = G22
       AMI = AM2
       AMJ = AM2
       SGI = 1
       SGJ = 1
       DIJ = 1
       SIJ = -S2T
       TIJ = S2T**2*G11*B11 + C2T**2*G22*B22 + 2*S2T*C2T*G12*B12
      ENDIF
      MIJ = RMQ*GQ*DIJ + GLR/RMQ*SIJ
      MIJ0= RMQ*GQ*DIJ
      AIJ = RMQ/2*GQ*SIJ
      C0IJ  = CC0(AMI,AMJ,AMH,XMU)
      CGIJ  = DREAL(C03_HDEC(AMI**2,AMH**2,AMJ**2,AMG,RMQ,RMQ))
      BI0I = B02_HDEC(AMI**2,0.D0,AMI,XMU**2)
      BJ0J = B02_HDEC(AMJ**2,0.D0,AMJ,XMU**2)
      BHIJ = B02_HDEC(AMH**2,AMI,AMJ,XMU**2)
      BIGQ = B02_HDEC(AMI**2,AMG,RMQ,XMU**2)
      BJGQ = B02_HDEC(AMJ**2,AMG,RMQ,XMU**2)
      BHQQ = B02_HDEC(AMH**2,RMQ,RMQ,XMU**2)
      BPI0I = BP02_HDEC(AMI**2,0.D0,AMI,XMU**2)
      BPJ0J = BP02_HDEC(AMJ**2,0.D0,AMJ,XMU**2)
      BPIGQ = BP02_HDEC(AMI**2,AMG,RMQ,XMU**2)
      BPJGQ = BP02_HDEC(AMJ**2,AMG,RMQ,XMU**2)
      AI = AMI**2*(1+DLOG(XMU**2/AMI**2))
      AJ = AMJ**2*(1+DLOG(XMU**2/AMJ**2))
      B101 = B02_HDEC(AM1**2,0.D0,AM1,XMU**2)
      B202 = B02_HDEC(AM2**2,0.D0,AM2,XMU**2)
      BQ0Q = B02_HDEC(RMQ**2,0.D0,RMQ,XMU**2)
      BQG1 = B02_HDEC(RMQ**2,AMG,AM1,XMU**2)
      BQG2 = B02_HDEC(RMQ**2,AMG,AM2,XMU**2)
      B1QG1 = BB1(RMQ**2,AMG,AM1,XMU)
      B1QG2 = BB1(RMQ**2,AMG,AM2,XMU)
      BIGQ = B02_HDEC(AMI**2,AMG,RMQ,XMU**2)
      BJGQ = B02_HDEC(AMJ**2,AMG,RMQ,XMU**2)
      A1 = AM1**2*(1+DLOG(XMU**2/AM1**2))
      A2 = AM2**2*(1+DLOG(XMU**2/AM2**2))
      AQ = RMQ**2*(1+DLOG(XMU**2/RMQ**2))
      AG = AMG**2*(1+DLOG(XMU**2/AMG**2))
      DZII = 2*BI0I-2*BIGQ+4*AMI**2*BPI0I
     .     + 2*(AMG**2+RMQ**2-AMI**2+SGI*2*AMG*RMQ*S2T)*BPIGQ
      DZJJ = 2*BJ0J-2*BJGQ+4*AMJ**2*BPJ0J
     .     + 2*(AMG**2+RMQ**2-AMJ**2+SGJ*2*AMG*RMQ*S2T)*BPJGQ
      DM12 = (1+C2T**2)*A1+S2T**2*A2-2*AG-2*AQ-4*AM1**2*B101
     .     - 2*(AMG**2+RMQ**2-AM1**2-2*AMG*RMQ*S2T)*B1GQ
      DM22 = (1+C2T**2)*A2+S2T**2*A1-2*AG-2*AQ-4*AM2**2*B202
     .     - 2*(AMG**2+RMQ**2-AM2**2+2*AMG*RMQ*S2T)*B2GQ
      IF(IONSH.EQ.1)THEN
       DMQ = RMQ*(AQ/RMQ**2-1+2*BQ0Q+B1QG1+B1QG2
     .     + AMG*2*(AQQ-AMU*RQ)/(AM1**2-AM2**2)*(BQG1-BQG2) + DELSUSY)
       DMQ0= DMQ
       DTH = (S2T*C2T*(A2-A1)+4*AMG*RMQ*C2T*BTGQ)/(AM1**2-AM2**2)
       DTH1 = (S2T*C2T*(A2-A1)+4*AMG*RMQ*C2T*B1GQ)/(AM1**2-AM2**2) - DTH
       DTH2 = (S2T*C2T*(A2-A1)+4*AMG*RMQ*C2T*B2GQ)/(AM1**2-AM2**2) - DTH
       DAQ = (AQQ-AMU*RQ)*(2*DTH*C2T/S2T + DMQ/RMQ
     .                    + (DM12-DM22)/(AM1**2-AM2**2))
       DLMQ = 0
       DLMQ0= DLMQ
       DLAQ = 0
      ELSE
       DMQ = RMQ*(3*DLOG(XMU**2/AMQSC**2)+B1QG1+B1QG2
     .     + AMG*2*(AQQ-AMU*RQ)/(AM1**2-AM2**2)*(BQG1-BQG2)
     .     + DELSUSY - DELTA_Q)
       DMQ0= DMQ
       DAQ = 4 * AMG*DLOG(XMU**2/AQSC**2)
       DLMQ = RMQ*3*DLOG(AMQSC**2/QQ**2)
       DLMQ0= DLMQ
       DLAQ = 4 * AMG*DLOG(AQSC**2/QQ**2)
       IF(IDTH.EQ.1)THEN
        DTH = (S2T*C2T*(A2-A1)+4*AMG*RMQ*C2T*BTGQ)/(AM1**2-AM2**2)
       ELSEIF(IDTH.EQ.21)THEN
        DTH = (S2T*C2T*(A2-A1)+4*AMG*RMQ*C2T*B1GQ)/(AM1**2-AM2**2)
       ELSEIF(IDTH.EQ.22)THEN
        DTH = (S2T*C2T*(A2-A1)+4*AMG*RMQ*C2T*B2GQ)/(AM1**2-AM2**2)
       ELSE
        DTH = S2T/2/C2T*(-DMQ/RMQ + DAQ/(AQQ-AMU*RQ)
     .      - (DM12-DM22)/(AM1**2-AM2**2))
       ENDIF
       DTH1 = (S2T*C2T*(A2-A1)+4*AMG*RMQ*C2T*B1GQ)/(AM1**2-AM2**2) - DTH
       DTH2 = (S2T*C2T*(A2-A1)+4*AMG*RMQ*C2T*B2GQ)/(AM1**2-AM2**2) - DTH
       IF(II.EQ.1.AND.JJ.EQ.1)THEN
        XIJ = 2*G12*DTH1
       ELSEIF(II.EQ.1.AND.JJ.EQ.2)THEN
        XIJ = G22*DTH1 - G11*DTH2
       ELSEIF(II.EQ.2.AND.JJ.EQ.1)THEN
        XIJ = G22*DTH1 - G11*DTH2
       ELSEIF(II.EQ.2.AND.JJ.EQ.2)THEN
        XIJ = -2*G12*DTH2
       ENDIF
      ENDIF
      IF(IHIGGS.EQ.3)THEN
       C1 = BI0I + BJ0J - BHIJ - 2*(AMH**2-AMI**2-AMJ**2)*C0IJ
       C2 = GQ/GLR * (RMQ*AMG*(AMH**2*CGIJ-BIGQ-BJGQ)
     .              - SGI*S2T*RMQ*RMQ*(BIGQ-BJGQ+(AMI**2-AMJ**2)*CGIJ))
       C3 = B12
       C4 = DZII/2 + DZJJ/2 + RMQ/2*GQ/GLR*DAQ - DMQ/RMQ
     .    + RMQ/2*GQ/GLR*DLAQ - DLMQ/RMQ
      ELSE
       C1 = BI0I + BJ0J - BHIJ - 2*(AMH**2-AMI**2-AMJ**2)*C0IJ
       C2 = GQ/GLO * (DIJ*RMQ**2*(BIGQ+BJGQ+2*BHQQ
     .               +(2*AMG**2+2*RMQ**2-AMI**2-AMJ**2)*CGIJ)
     .   +SIJ*RMQ*AMG*((AMH**2-4*RMQ**2)*CGIJ-BIGQ-BJGQ))
       C3 = -TIJ/GLO
       C4 = DZII/2 + DZJJ/2 + (-MIJ*DMQ-MIJ0*DMQ0+AIJ*DAQ-XIJ)/GLO
     .    + (-MIJ*DLMQ-MIJ0*DLMQ0+AIJ*DLAQ)/GLO
      ENDIF
      CVIRT = (C1 + C2 + C3 + C4)/2
C--REAL CORRECTIONS
      RHOI = AMI**2/AMH**2
      RHOJ = AMJ**2/AMH**2
      BET0 = LAMB_HDEC(RHOI,RHOJ)
      X0 = (1-RHOI-RHOJ-BET0)/(1-RHOI-RHOJ+BET0)
      X1 = (1-RHOI+RHOJ-BET0)/(1-RHOI+RHOJ+BET0)
      X2 = (1+RHOI-RHOJ-BET0)/(1+RHOI-RHOJ+BET0)
      CREAL = (1-RHOI-RHOJ)/BET0*(DLOG(X0)/2*DLOG(XMU**2/AMH**2)
     .      + DLOG(X0)*DLOG(DSQRT(RHOI*RHOJ)/BET0**2) + DLOG(X0)**2/2
     .      - DLOG(X1)**2/4 - DLOG(X2)**2/4 + 2*SP_HDEC(1-1/X0)
     .      - SP_HDEC(1-X1) - SP_HDEC(1-X2))
     .      + DLOG(XMU**2/AMH**2) + 2*DLOG(DSQRT(RHOI*RHOJ)/BET0**2)
     .      + 4 - (1+RHOI+RHOJ)/2/BET0*DLOG(X0) + RHOI/BET0*DLOG(X1)
     .      + RHOJ/BET0*DLOG(X2)
      ALP = ALPHAS_HDEC(QQ,3)/PI
      SQSUSY_HDEC = CF*ALP*(CVIRT + CREAL)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      RETURN
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     IF(IHIGGS.EQ.2)THEN
c      write(9,*)
c      write(9,*)II,JJ,QQ,AM1,AM2
c      write(9,*)
c      write(9,*)CF*C1/2,CF*C2/2,CF*C3/2,CF*C4/2,CF*CREAL,
c    .           CF*(CVIRT+CREAL)
c      write(9,*)
c      write(9,*)CF/2*(DZII/2+DZJJ/2),
c    .CF/2*(-MIJ*DMQ-MIJ0*DMQ0-MIJ*DLMQ-MIJ0*DLMQ0)/GLO,
c    .CF/2*(AIJ*DAQ+AIJ*DLAQ)/GLO,CF/2*(-XIJ)/GLO,-CF/4*(DMQ+DLMQ),RMQ
c      write(9,*)2*(MIJ+MIJ0)/GLO
c      write(9,*)-CF/4*RMQ*3*DLOG(XMU**2/QQ**2)
c    .         ,-CF/4*RMQ*DELSUSY
c    .         ,CF/4*RMQ*DELTA_Q
c    .         ,-CF/4*RMQ*(B1QG1+B1QG2+AMG*2*(AQQ-AMU*RQ)
c    .                     /(AM1**2-AM2**2)*(BQG1-BQG2))
c      write(9,*)
c     ENDIF
c     RETURN
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      APPTHRESH = CF/2*ALP*PI**2/BET0 * 4*AMI*AMJ/(AMI+AMJ)**2
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      AMS = (AMI+AMJ+DABS(AMG))/3
      IF(ISQ.EQ.1)THEN
       DELTAQ = AMG/DABS(AMG)*(AU-AMU/DTAN(B))/AMS/4
      ELSE
       DELTAQ = AMG/DABS(AMG)*AD/AMS/4
      ENDIF
      IF(IHIGGS.EQ.3)THEN
       DELTA = 0
       AIJ = RMQ/2*GQ
       MIJ = GLR/RMQ
       MIJ0= 0
      ELSE
       IF(II.EQ.JJ)THEN
        THIJ = 2*G12
       ELSE
        THIJ = G11 + G22
       ENDIF
       DELTA = (RMQ/GLO*(MIJ+MIJ0)-1)*(DLOG(AMH**2/AMS**2)-2)
     .       + GQ*RMQ**2/GLO*DIJ*DLOG(AMH**2/AMS**2)
     .       + THIJ/GLO*C2T*RMQ/AMS*(1.D0-DLOG(AMS**2/RMQ**2)/2)
       IF(ISQ.EQ.1)THEN
        DELTA = DELTA + GQ*RMQ**2/GLO*DIJ*(-PI/DSQRT(3.D0))
     .        + (1-RMQ/GLO*(MIJ+MIJ0))*(5*pi/3/DSQRT(3.D0)-1
     .              +(SGI+SGJ)/2*(1-2*PI/3/DSQRT(3.D0)))
     .        - THIJ/GLO*C2T*RMQ/AMS*(1.D0-DLOG(AMS**2/RMQ**2)/2)
     .        + THIJ/GLO*C2T*RMQ/AMS*(1-2*PI/3/DSQRT(3.D0))
       ENDIF
      ENDIF
      APPLARGE = CF*ALP*(2*RMQ/GLO*(MIJ+MIJ0)
     .                 *(3*DLOG(QQ**2/AMH**2)/4-DLOG(AMH**2/AMS**2)/4
     .                  +7.D0/4+DELTAQ)
     .                  +2*AMG/GLO*AIJ*(-DLOG(QQ**2/AMS**2)
     .                    +DLOG(AMH**2/AMS**2)**2/4+ZETA2/2-2)+DELTA)
      IF(ISQ.EQ.1)THEN
       APPLARGE = APPLARGE
     . + CF*ALP*(2*RMQ/GLO*(MIJ+MIJ0)*(7*PI/12/DSQRT(3.D0)
     .                            +(SGI+SGJ)*(0.25D0-PI/6/DSQRT(3.D0))
     .                            +(1-2*PI/3/DSQRT(3.D0))*DELTAQ)
     .          +2*AMG/GLO*AIJ*(-5*ZETA2/3.D0+PI/DSQRT(3.D0)))
      ENDIF
      RAT1 = APPTHRESH/SQSUSY_HDEC
      RAT2 = APPLARGE/SQSUSY_HDEC
      IF(ISQ.EQ.1)THEN
       IF(IHIGGS.EQ.3)THEN
       if(ii.eq.1.and.jj.eq.1)write(411,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       if(ii.eq.1.and.jj.eq.2)write(412,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       if(ii.eq.2.and.jj.eq.1)write(421,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       if(ii.eq.2.and.jj.eq.2)write(422,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       ELSE
       if(ii.eq.1.and.jj.eq.1)write(211,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       if(ii.eq.1.and.jj.eq.2)write(212,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       if(ii.eq.2.and.jj.eq.1)write(221,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       if(ii.eq.2.and.jj.eq.2)write(222,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       ENDIF
      ELSE
       IF(IHIGGS.EQ.3)THEN
       if(ii.eq.1.and.jj.eq.1)write(311,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       if(ii.eq.1.and.jj.eq.2)write(312,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       if(ii.eq.2.and.jj.eq.1)write(321,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       if(ii.eq.2.and.jj.eq.2)write(322,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       ELSE
       if(ii.eq.1.and.jj.eq.1)write(111,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       if(ii.eq.1.and.jj.eq.2)write(112,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       if(ii.eq.2.and.jj.eq.1)write(121,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       if(ii.eq.2.and.jj.eq.2)write(122,"('3(1X,G12.4)')")AMH,RAT1,RAT2
       ENDIF
      ENDIF
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      IF(IHIGGS.EQ.3)THEN
       IF(ISQ.EQ.1)write(8,*)'Stops: M_A = ',AMH,II,JJ,G12
       IF(ISQ.NE.1)write(8,*)'Sbottoms: M_A = ',AMH,II,JJ,G12
      ELSE
       IF(ISQ.EQ.1)write(8,*)'Stops: M_H = ',AMH,II,JJ,G11,G12/G11
       IF(ISQ.NE.1)write(8,*)'Sbottoms: M_H = ',AMH,II,JJ,G11,G12/G11
      ENDIF
      write(8,"('A8,3(1X,G12.4)')")'Bi0i: ',BI0I,2+DLOG(XMU**2/AMI**2),
     .                                    (2+DLOG(XMU**2/AMI**2))/BI0I
      write(8,"('A8,3(1X,G12.4)')")'B''i0i: ',BPI0I,
     .                           -(2+DLOG(XMU**2/AMI**2))/2/AMI**2,
     .                           -(2+DLOG(XMU**2/AMI**2))/2/AMI**2/BPI0I
      write(8,"('A8,3(1X,G12.4)')")'BHij: ',BHIJ,2+DLOG(XMU**2/AMH**2),
     .                                    (2+DLOG(XMU**2/AMH**2))/BHIJ
      IF(ISQ.EQ.1)THEN
      write(8,"('A8,3(1X,G12.4)')")'Bigq: ',BIGQ,
     .                    2+DLOG(XMU**2/AMG**2)-PI/DSQRT(3.D0),
     .                   (2+DLOG(XMU**2/AMG**2)-PI/DSQRT(3.D0))/BIGQ
      write(8,"('A8,3(1X,G12.4)')")'B''igq: ',BPIGQ,
     .                           -(1-2*PI/3/DSQRT(3.D0))/AMI**2,
     .                           -(1-2*PI/3/DSQRT(3.D0))/AMI**2/BPIGQ
      ELSE
      write(8,"('A8,3(1X,G12.4)')")'Bigq: ',BIGQ,2+DLOG(XMU**2/AMG**2),
     .                                    (2+DLOG(XMU**2/AMG**2))/BIGQ
      write(8,"('A8,3(1X,G12.4)')")'B''igq: ',BPIGQ,
     .                           -(2+DLOG(RMQ**2/AMI**2))/2/AMI**2,
     .                           -(2+DLOG(RMQ**2/AMI**2))/2/AMI**2/BPIGQ
      ENDIF
      write(8,"('A8,3(1X,G12.4)')")'BHqq: ',BHQQ,2+DLOG(XMU**2/AMH**2),
     .                                    (2+DLOG(XMU**2/AMH**2))/BHQQ
      write(8,"('A8,3(1X,G12.4)')")'C0ij: ',C0IJ,
     .    (DLOG(AMI**2/AMH**2)*DLOG(XMU**2/AMH**2)-4*ZETA2
     .    -DLOG(AMI**2/AMH**2)**2/2)/AMH**2,
     .    (DLOG(AMI**2/AMH**2)*DLOG(XMU**2/AMH**2)-4*ZETA2
     .    -DLOG(AMI**2/AMH**2)**2/2)/AMH**2/C0IJ
      IF(ISQ.EQ.1)THEN
      write(8,"('A8,3(1X,G12.4)')")'CGij: ',CGIJ,
     .    (DLOG(AMI**2/AMH**2)**2/2-7*ZETA2/3)/AMH**2,
     .    (DLOG(AMI**2/AMH**2)**2/2-7*ZETA2/3)/AMH**2/CGIJ
      ELSE
      write(8,"('A8,3(1X,G12.4)')")'CGij: ',CGIJ,
     .    (DLOG(AMI**2/AMH**2)**2/2+ZETA2)/AMH**2,
     .    (DLOG(AMI**2/AMH**2)**2/2+ZETA2)/AMH**2/CGIJ
      ENDIF
      XZII = (-DLOG(XMU**2/AMH**2)+DLOG(AMS**2/AMH**2)-2)/2
      XMQ  = RMQ*(-3*DLOG(XMU**2/QQ**2)/4+DLOG(XMU**2/AMH**2)/4
     .           - DLOG(AMS**2/AMH**2)/4-1.D0/4+DELTAQ)
      XAQ  = AMS*DLOG(XMU**2/QQ**2)
      XTH0 = 2*C2T*RMQ/AMS*(1+DLOG(RMQ**2/AMS**2)/2)
      IF(ISQ.EQ.1)THEN
       XZII = XZII +(-1+5*PI/3/DSQRT(3.D0)+SGI*(1-2*PI/3/DSQRT(3.D0)))/2
       XMQ  = XMQ + RMQ*(1/2.D0-PI/4/DSQRT(3.D0)
     .                  +(1-2*PI/3/DSQRT(3.D0))*(AQQ-AMU*RQ)/4/AMS
     .                  )
       XTH0 = 2*C2T*RMQ/AMS*(1-2*PI/3/DSQRT(3.D0))
      ENDIF
      IF(II.EQ.1.AND.JJ.EQ.1)THEN
       SGI = 1
       SGJ = 1
       DIJ = 1
       SIJ = S2T
       TIJ = C2T**2*G11 + S2T**2*G22 - 2*S2T*C2T*G12
       YIJ = 2*G12
      ELSEIF(II.EQ.1.AND.JJ.EQ.2)THEN
       SGI = 1
       SGJ =-1
       DIJ = 0
       SIJ = C2T
       TIJ = -S2T*C2T*(G11-G22) - (C2T**2-S2T**2)*G12
       YIJ = G11 + G22
      ELSEIF(II.EQ.2.AND.JJ.EQ.1)THEN
       SGI =-1
       SGJ = 1
       DIJ = 0
       SIJ = C2T
       TIJ = -S2T*C2T*(G11-G22) - (C2T**2-S2T**2)*G12
       YIJ = G11 + G22
      ELSEIF(II.EQ.2.AND.JJ.EQ.2)THEN
       SGI =-1
       SGJ =-1
       DIJ = 1
       SIJ = -S2T
       TIJ = S2T**2*G11 + C2T**2*G22 + 2*S2T*C2T*G12
       YIJ = 2*G12
      ENDIF
      XTH1 = -XTH0
      XTH2 =  XTH0
      write(8,"('A8,3(1X,G12.4)')")'Zii: ',DZII/4,XZII,4*XZII/DZII
      write(8,"('A8,3(1X,G12.4)')")'mq: ',-(DMQ+DLMQ)/4,XMQ,
     .                                  -4*XMQ/(DMQ+DLMQ)
      write(8,"('A8,3(1X,G12.4)')")'Aq:  ',(DAQ+DLAQ)/4,XAQ
c    .                                  ,4*XAQ/(DAQ+DLAQ)
      write(8,"('A8,3(1X,G12.4)')")'th1: ',DTH1,XTH1,XTH1/DTH1
      write(8,"('A8,3(1X,G12.4)')")'th2: ',DTH2,XTH2,XTH2/DTH2
      write(8,"('A8,3(1X,G12.4)')")'dth: ',XIJ/GLO,-YIJ*XTH0/GLO,
     .                                   -YIJ*XTH0/XIJ
      write(8,"('A8,3(1X,G12.4)')")'dth: ',XIJ/GLO,-THIJ*XTH0/GLO,
     .                                   -THIJ*XTH0/XIJ
      X1 = DLOG(XMU**2/AMH**2)*(1-2*DLOG(AMS**2/AMH**2))
     .   + DLOG(AMS**2/AMH**2)**2-2*DLOG(AMS**2/AMH**2)+8*ZETA2+2
      IF(IHIGGS.EQ.3)THEN
       X2 = GQ/GLO*(SGI*AMS*RMQ*(-2*DLOG(XMU**2/AMH**2)
     .        +2*DLOG(AMS**2/AMH**2)+DLOG(AMS**2/AMH**2)**2/2+ZETA2-4))
       X3 = 2+DLOG(XMU**2/AMH**2)
       X4 = -DLOG(XMU**2/AMH**2)-3*DLOG(XMU**2/QQ**2)-5
     .    + DLOG(AMS**2/AMH**2)+4*DELTAQ
     .    + SGI*RMQ*AMS*GQ/GLO*2*DLOG(XMU**2/QQ**2)
       IF(ISQ.EQ.1)THEN
        X2 = X2 + SGI*RMQ*AMS*GQ/GLO*(-10*ZETA2/3+2*PI/DSQRT(3.D0))
        X4 = X4 - 2+10*PI/3/DSQRT(3.D0)+(SGI+SGJ)*(1-2*PI/3/DSQRT(3.D0))
     .          + 4*(1/2.D0-PI/4/DSQRT(3.D0)
     .              +(1-2*PI/3/DSQRT(3.D0))*(AQQ-AMU*RQ)/4/AMS)
       ENDIF
      ELSE
       X2 = GQ/GLO*(DIJ*RMQ**2*(4*DLOG(XMU**2/AMH**2)
     .                         -2*DLOG(AMS**2/AMH**2)+8)
     .             +SIJ*AMS*RMQ*(-2*DLOG(XMU**2/AMH**2)
     .        +2*DLOG(AMS**2/AMH**2)+DLOG(AMS**2/AMH**2)**2/2+ZETA2-4))
       X3 = -TIJ/GLO*(2+DLOG(XMU**2/AMH**2))
       X4 = -2*DLOG(XMU**2/AMH**2)+2*DLOG(AMS**2/AMH**2)-4
     .    + RMQ*AMS*GQ/GLO*SIJ*2*DLOG(XMU**2/QQ**2)
     .    + (GQ*DIJ*2*RMQ**2+SIJ*GLR)/GLO*(-3*DLOG(XMU**2/QQ**2)
     .      +DLOG(XMU**2/AMH**2)-DLOG(AMS**2/AMH**2)-1+4*DELTAQ)
     .    + YIJ*XTH0/GLO
       IF(ISQ.EQ.1)THEN
        X2 = X2 + GQ/GLO*(DIJ*RMQ**2*(-2*PI/DSQRT(3.D0))
     .          + SIJ*RMQ*AMS*(-10*ZETA2/3+2*PI/DSQRT(3.D0)))
        X4 = X4 - 2+10*PI/3/DSQRT(3.D0)+(SGI+SGJ)*(1-2*PI/3/DSQRT(3.D0))
     .          + 4*(1/2.D0-PI/4/DSQRT(3.D0)
     .              +(1-2*PI/3/DSQRT(3.D0))*(AQQ-AMU*RQ)/4/AMS)
       ENDIF
      ENDIF
      XR = DLOG(XMU**2/AMH**2)*(2+2*DLOG(AMS**2/AMH**2))
     .   - DLOG(AMS**2/AMH**2)**2+2*DLOG(AMS**2/AMH**2)-8*ZETA2+8
      XX = (X1+X2+X3+X4+XR)/2
      write(8,"('A8,3(1X,G12.4)')")'C1: ',C1/2,X1/2,X1/C1
      write(8,"('A8,3(1X,G12.4)')")'C2: ',C2/2,X2/2,X2/C2
      write(8,"('A8,3(1X,G12.4)')")'C3: ',C3/2,X3/2,X3/C3
      write(8,"('A8,3(1X,G12.4)')")'C4: ',C4/2,X4/2,X4/C4
      write(8,"('A8,3(1X,G12.4)')")'Creal: ',CREAL,XR/2,XR/2/CREAL
      write(8,"('A8,4(1X,G12.4)')")'tot: ',SQSUSY_HDEC,CF*ALP*XX,
     .                  CF*ALP*XX/SQSUSY_HDEC,CF*ALP*XX/APPLARGE
c     BIII = B02_HDEC(AMI**2,AMI,AMI,XMU**2)
c     BTTT = 2-PI/DSQRT(3.D0)+DLOG(XMU**2/AMI**2)
c     write(8,"('A8,3(1X,G12.4)')")'test: ',BIII,BTTT,BTTT/BIII
c     BPIII = BP02_HDEC(AMI**2,AMI,AMI,XMU**2)
c     BPTTT = (-1+2*PI/3/DSQRT(3.D0))/AMI**2
c     write(8,"('A8,3(1X,G12.4)')")'testp: ',BPIII,BPTTT,BPTTT/BPIII
      write(8,*)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      RETURN
      END

      DOUBLE PRECISION FUNCTION XITSUSY_HDEC(Q,ALS,ACC)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      ALS2(X,XLB,B0,B1)=12.D0*PI/(B0*DLOG(X**2/XLB**2))
     .              *(1.D0-B1*DLOG(DLOG(X**2/XLB**2))
     .              /DLOG(X**2/XLB**2))
      XIT(A,B,X)=A/2.D0*(1D0+DSQRT(1D0-4D0*B*DLOG(X)))
      PI=4.D0*DATAN(1.D0)
      B0 = 9
      B1 =-14.D0/9
      AA = 12*PI/B0
      BB = B1/AA
      A = AA/ALS
      B = BB*ALS
      XLB1 = Q*DEXP(-6*PI/B0/ALS)
      II=0
1     II=II+1
      XLB = XLB1
      X=DLOG(Q**2/XLB**2)
      XX=XIT(A,B,X)
      XLB1=Q*DEXP(-XX/2.D0)
      Y0=ALS2(Q,XLB,B0,B1)
      Y1=ALS2(Q,XLB1,B0,B1)
      DY=DABS(Y1-Y0)/Y0
      IF(DY.GE.ACC) GOTO 1
      XITSUSY_HDEC=XLB1
      RETURN
      END
 
      SUBROUTINE SQMBAPP_HDEC(QSQ)
      IMPLICIT DOUBLE PRECISION (A-H,K-Z)
      COMMON/PARAM_HDEC/GF,ALPH,AMTAU,AMMUON,MZ,MW
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/COUP_HDEC/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      COMMON/QTBNLO_HDEC/MT,MB
      COMMON/SQNLO_HDEC/AMSB(2),STHB,CTHB,GLBB(2,2),GHBB(2,2),GABB,
     .                  AMST(2),STHT,CTHT,GLTT(2,2),GHTT(2,2),GATT
      COMMON/BREAK_HDEC/AMEL,AMER,AMSQ,AMUR,AMDR,AL,AT0,AB0,MU,AM20
      COMMON/TRILINEAR_HDEC/AT,AB
      COMMON/BREAKSCALE_HDEC/SUSYSCALE
      COMMON/SQUARKHIGGS_HDEC/THEB,AMG,IONSH,IDTH
      PI = 4*DATAN(1.D0)
      CF = 4/3.D0
      SW2=1.D0-MW**2/MZ**2
      TB=DTAN(B)
      SB=STHB
      CB=CTHB
      ST=STHT
      CT=CTHT
      DELTA_B = 2*AMG*MU*DTAN(B)*T_HDEC(AMSB(1),AMSB(2),AMG)
c     DELTA_T = 2*AMG*MU/DTAN(B)*T_HDEC(AMST(1),AMST(2),AMG)
      DELTA_T = 0
      ALSSCB = (AMSB(1)+AMSB(2)+DABS(AMG))/3
      ALSSCT = (AMST(1)+AMST(2)+DABS(AMG))/3
c     AMSUSY = DSQRT(2*AMSQ**2+AMUR**2+AMDR**2)/2
      AMSUSY = SUSYSCALE
      QQ0 = AMSUSY
c     AT = AT0 + CF*ALPHAS_HDEC(QSQ,3)/PI*AMG*DLOG(QSQ**2/QQ0**2)
c     AB = AB0 + CF*ALPHAS_HDEC(QSQ,3)/PI*AMG*DLOG(QSQ**2/QQ0**2)
      IF(IONSH.EQ.1)THEN
       MB = AMB
       MB0= AMB
       MT = AMT
      ELSE
       MB = RUNM_HDEC(QSQ,5)/(1+CF/4*ALPHAS_HDEC(ALSSCB,3)/PI*DELTA_B)
       MB0= MB
       MT = RUNM_HDEC(QSQ,6)/(1+CF/4*ALPHAS_HDEC(ALSSCT,3)/PI*DELTA_T)
      ENDIF
c     write(6,*)'tmass: ',MT,RUNM_HDEC(QSQ,6)
c     write(6,*)'bmass: ',MB,RUNM_HDEC(QSQ,5)
c     write(6,*)'scale: ',QSQ,QQ0
c     write(6,*)'At0,Ab0: ',AT0,AB0
c     write(6,*)'At,Ab:   ',AT,AB
c     write(6,*)'g_1-4:   ',GHT,DCOS(B+A),GLT,GHT
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     MB = AMB
c     MB0= AMB
c     MT = AMT
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C--EFFECTIVE HIGGS COUPLINGS
C
C LIGHT CP--EVEN HIGGS COUPLINGS TO SBOTTOMS
C
      GLBB(1,1)=-DSIN(B+A)*(-0.5D0*CB**2+1.D0/3.D0*SW2*(CB**2-SB**2))
     .    + MB*MB0/MZ**2*GLB + MB*SB*CB/MZ**2*(AB*GLB-MU*GHB)
      GLBB(2,2)=-DSIN(B+A)*(-0.5D0*SB**2-1.D0/3.D0*SW2*(CB**2-SB**2))
     .    + MB*MB0/MZ**2*GLB - MB*SB*CB/MZ**2*(AB*GLB-MU*GHB)
      GLBB(1,2)=-2*DSIN(B+A)*SB*CB*(-1.D0/3.D0*SW2+0.25D0)
     .    + MB*(CB**2-SB**2)/2.D0/MZ**2*(AB*GLB-MU*GHB)
      GLBB(2,1)=-2*DSIN(B+A)*SB*CB*(-1.D0/3.D0*SW2+0.25D0)
     .    + MB*(CB**2-SB**2)/2.D0/MZ**2*(AB*GLB-MU*GHB)
C
C HEAVY CP--EVEN HIGGS COUPLINGS TO SBOTTOMS
C
      GHBB(1,1)= DCOS(B+A)*(-0.5D0*CB**2+1.D0/3.D0*SW2*(CB**2-SB**2))
     .    + MB*MB0/MZ**2*GHB + MB*SB*CB/MZ**2*(AB*GHB+MU*GLB)
      GHBB(2,2)= DCOS(B+A)*(-0.5D0*SB**2-1.D0/3.D0*SW2*(CB**2-SB**2))
     .    + MB*MB0/MZ**2*GHB - MB*SB*CB/MZ**2*(AB*GHB+MU*GLB)
      GHBB(1,2)=2*DCOS(B+A)*SB*CB*(-1.D0/3.D0*SW2+0.25D0)
     .    + MB*(CB**2-SB**2)/2.D0/MZ**2*(AB*GHB+MU*GLB)
      GHBB(2,1)=2*DCOS(B+A)*SB*CB*(-1.D0/3.D0*SW2+0.25D0)
     .    + MB*(CB**2-SB**2)/2.D0/MZ**2*(AB*GHB+MU*GLB)
C
C PSEUDOSCALAR COUPLINGS 
C
      GABB=MB/2.D0/MZ**2*(MU+AB*GAB)
C
C LIGHT CP--EVEN HIGGS COUPLINGS TO STOPS
C 
      GLTT(1,1)=-DSIN(B+A)*(0.5D0*CT**2-2.D0/3.D0*SW2*(CT**2-ST**2)) 
     .    + MT**2/MZ**2*GLT + MT*ST*CT/MZ**2*(AT*GLT+MU*GHT)
      GLTT(2,2)=-DSIN(B+A)*(0.5D0*ST**2+2.D0/3.D0*SW2*(CT**2-ST**2))
     .    + MT**2/MZ**2*GLT - MT*ST*CT/MZ**2*(AT*GLT+MU*GHT)
      GLTT(1,2)=-2*DSIN(B+A)*ST*CT*(2.D0/3.D0*SW2-0.25D0)
     .    + MT*(CT**2-ST**2)/2.D0/MZ**2*(AT*GLT+MU*GHT) 
      GLTT(2,1)=-2*DSIN(B+A)*ST*CT*(2.D0/3.D0*SW2-0.25D0)
     .    + MT*(CT**2-ST**2)/2.D0/MZ**2*(AT*GLT+MU*GHT) 
C
C HEAVY CP--EVEN HIGGS COUPLINGS TO STOPS
C
      GHTT(1,1)=DCOS(B+A)*(0.5D0*CT**2-2.D0/3.D0*SW2*(CT**2-ST**2)) 
     .    + MT**2/MZ**2*GHT + MT*ST*CT/MZ**2*(AT*GHT-MU*GLT)
      GHTT(2,2)= DCOS(B+A)*(0.5D0*ST**2+2.D0/3.D0*SW2*(CT**2-ST**2))
     .    + MT**2/MZ**2*GHT - MT*ST*CT/MZ**2*(AT*GHT-MU*GLT)
      GHTT(1,2)=2*DCOS(B+A)*ST*CT*(2.D0/3.D0*SW2-0.25D0)
     .    + MT*(CT**2-ST**2)/2.D0/MZ**2*(AT*GHT-MU*GLT) 
      GHTT(2,1)=2*DCOS(B+A)*ST*CT*(2.D0/3.D0*SW2-0.25D0)
     .    + MT*(CT**2-ST**2)/2.D0/MZ**2*(AT*GHT-MU*GLT) 
C
C PSEUDOSCALAR COUPLINGS 
C
      GATT=MT/2.D0/MZ**2*(MU+AT*GAT) 

      RETURN
      END

C   ********************************************************************
C     SUBROUTINE FOR SFERMION MASSES, MIXING AND COUPLINGS AT NLO (!!)
C   ********************************************************************

      SUBROUTINE SFERMHO_HDEC(AMSQ,AMUR,AMDR,AU,AD,AMU,TB,AMW,AMZ)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT
      COMMON/SQNLO_HDEC/AMSB(2),STHB,CTHB,GLBB(2,2),GHBB(2,2),GABB,
     .                  AMST(2),STHT,CTHT,GLTT(2,2),GHTT(2,2),GATT
      COMMON/SQUARKHIGGS_HDEC/THEB,AMG,IONSH,IDTH
      COMMON/BREAKSCALE_HDEC/SUSYSCALE
      PI = 4*DATAN(1.D0)
      CF = 4/3.D0
      SW2=1.D0-AMW**2/AMZ**2
c     QQ = DSQRT(2*AMSQ**2+AMUR**2+AMDR**2)/2
      QQ = SUSYSCALE
C
C STOP MASSES/MIXING
C
c     write(6,*)AMSQ,AMUR,AMDR,AU,AD,AMU,TB,AMW,AMZ
c     write(6,*)'mt(mt), mb(mt): ',runm_hdec(amt,6),runm_hdec(amt,5)
c     write(6,*)
c     write(6,*)'STOPS:'
      ISO = 1
      IONSH = 0
      IDTH  = 1
      IF(IONSH.EQ.1)THEN
       RMT = AMT
      ELSE
       RMT = RUNM_HDEC(QQ,6)
      ENDIF
c     CALL SFERMIT_HDEC(ISO,QQ,RMT,AMG,AMSQ,AMUR,AU,AMU,TB,
c    .                  AMW,AMZ,AMST(1),AMST(2),RMSQ,RMUR,THET,IERR)
c     IF(IERR.NE.0)THEN
       FAC = 1
       CALL SFERMIT0_HDEC(ISO,QQ,RMT,AMG,AMSQ,AMUR,AU,AMU,TB,
     .                    AMW,AMZ,AMST(1),AMST(2),RMSQ,RMUR,THET,FAC)
c     ENDIF
      CTHT= DCOS(THET)
      STHT= DSIN(THET) 

      ALP = CF*ALPHAS_HDEC(QQ,3)/4/PI
c     DELTA_Q = 2*AMG*AMU/TB*T_HDEC(AMST(1),AMST(2),AMG)
      DELTA_Q = 0
      AI3 = 0.5D0
      EQ  = 2/3.D0
      DL = (AI3-EQ*SW2)*AMZ**2*(1-TB**2)/(1+TB**2)
      DR = EQ*SW2*AMZ**2*(1-TB**2)/(1+TB**2)
      AMSQL2 = RMSQ**2 + DL
      AMSQR2 = RMUR**2 + DR
      AMLR   = AU - AMU/TB
      DEL    = (AMSQL2-AMSQR2)**2+4*AMT**2*AMLR**2
      AMSQ12 = AMT**2 + (AMSQL2+AMSQR2-DSQRT(DEL))/2
      AMSQ22 = AMT**2 + (AMSQL2+AMSQR2+DSQRT(DEL))/2
      AM1=DSQRT(AMSQ12)
      AM2=DSQRT(AMSQ22)
      IF(AMSQL2.EQ.AMSQR2) THEN
       THET = PI/4
      ELSE
       THET=0.5D0*DATAN(2.D0*AMT*AMLR / (AMSQL2-AMSQR2) )
       IF(AMSQL2.GT.AMSQR2) THET = THET + PI/2
      ENDIF
C
C SBOTTOM MASSES/MIXING
C
c     write(6,*)'SBOTTOMS:'
      ISO = 2
      IONSH = 0
      IDTH  = 1
      IF(IONSH.EQ.1)THEN
       RMB = AMB
      ELSE
       RMB = RUNM_HDEC(QQ,5)
      ENDIF
c     CALL SFERMIT_HDEC(ISO,QQ,RMB,AMG,AMSQ,AMDR,AD,AMU,TB,
c    .                  AMW,AMZ,AMSB(1),AMSB(2),RMSQ,RMDR,THEB,IERR)
c     IF(IERR.NE.0)THEN
       FAC = 1
c      DO I=1,10
c       write(6,*)'i = ',i
c       CALL SFERMIT0_HDEC(ISO,QQ,RMB,AMG,AMSQ,AMDR,AD,AMU,TB,
c    .                     AMW,AMZ,AMSB(1),AMSB(2),RMSQ,RMDR,THEB,FAC)
c      ENDDO
       CALL SFERMIT1_HDEC(ISO,QQ,RMB,AMG,AMSQ,AMDR,AD,AMU,TB,
     .                    AMW,AMZ,AMSB(1),AMSB(2),RMSQ,RMDR,THEB)
c     ENDIF
      CTHB= DCOS(THEB)
      STHB= DSIN(THEB)

      CALL SQMBAPP_HDEC(QQ)
      RETURN
      END 

      SUBROUTINE SFERMIT_HDEC(ISO,QQ,RMQ0,AMG,AML,AMR,AQ,AMU,TB,
     .                        AMW,AMZ,AM1,AM2,RML,RMR,THET,IERR)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      COMMON/SQUARKHIGGS_HDEC/THEB,XMG,IONSH,IDTH
      BB1(P2,AM1,AM2,XMU)=((AM1**2-AM2**2)*B02_HDEC(0.D0,AM1,AM2,XMU**2)
     .           -(P2+AM1**2-AM2**2)*B02_HDEC(P2,AM1,AM2,XMU**2))/2/P2
      ACC = 1.D-12
      DELSUSY = 1
      PI = 4*DATAN(1.D0)
      SW2=1.D0-AMW**2/AMZ**2
      XMU = QQ
      CF = 4/3.D0
      ALP = CF*ALPHAS_HDEC(QQ,3)/4/PI
c************************************
c--factor C_F/4 extracted!!!
c************************************
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      FAC = 0.50D0
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      IF(ISO.EQ.1)THEN
       AI3 = 0.5D0
       EQ  = 2/3.D0
       RQ = 1/TB
      ELSE
       AI3 =-0.5D0
       EQ  =-1/3.D0
       RQ = TB
      ENDIF
      DL = (AI3-EQ*SW2)*AMZ**2*(1-TB**2)/(1+TB**2)
      DR = EQ*SW2*AMZ**2*(1-TB**2)/(1+TB**2)
      RML0 = AML
      RMR0 = AMR
C--INITIALIZATION
      AMSQL20= AML**2 + DL
      AMSQR20= AMR**2 + DR
      AMLR   = AQ - AMU*RQ
      DEL0   = (AMSQL20-AMSQR20)**2+4*RMQ0**2*AMLR**2
      AMSQ12 = RMQ0**2 + (AMSQL20+AMSQR20-DSQRT(DEL0))/2
      AMSQ22 = RMQ0**2 + (AMSQL20+AMSQR20+DSQRT(DEL0))/2
      AM10=DSQRT(AMSQ12)
      AM20=DSQRT(AMSQ22)
      IF(AMSQL20.EQ.AMSQR20) THEN
       THET = PI/4
      ELSE
       THET=0.5D0*DATAN(2.D0*RMQ0*AMLR / (AMSQL20-AMSQR20) )
       IF(AMSQL20.GT.AMSQR20) THET = THET + PI/2
      ENDIF
      CT= DCOS(THET)
      ST= DSIN(THET) 
      C2T0= CT**2-ST**2
      S2T0= 2*ST*CT
      C2T = C2T0
      S2T = S2T0
      DIF1 = 0
      DIF2 = 0
      DIF10= AM10**2
      DIF20= AM20**2
      RM1 = AM10
      RM2 = AM20
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'Init: m10,m20,theta/pi,ml,mr'
c     write(6,*)'Init1: c2t0,s2t0'
c     write(6,*)'Iteration0: i,m1,m2,theta/pi,ml,mr'
c     write(6,*)'Iteration: i,m1,m2,theta/pi,ml,mr'
c     write(6,*)'Check: I,m1,m2,theta/pi'
c     write(6,*)'Check0: I,del1,del2,ierr'
c     write(6,*)'Check1: I,c2t,s2t'
c     write(6,*)'Check2: I,c2t0,s2t0'
c     write(6,*)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'Init:    ',AM10,AM20,THET/PI,AML,AMR
c     write(6,*)'Init1:   ',(AMSQL20-AMSQR20)/(AM10**2-AM20**2),
c    .                      2*RMQ0*AMLR/(AM10**2-AM20**2)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      III = 0
100   RML = RML0
      RMR = RMR0
      AM1 = RM1
      AM2 = RM2
      DELTA_Q = 0
      IF(ISO.EQ.2) DELTA_Q = 2*AMG*AMU*RQ*T_HDEC(AM1,AM2,AMG)
      RMQ = RMQ0
      IF(IONSH.EQ.0)THEN
       RMQ = RMQ0/(1+ALP*DELTA_Q)
      ENDIF
      DEL0   = (AMSQL20-AMSQR20)**2+4*RMQ**2*AMLR**2
      III = III+1
      A1 = AM1**2*(1+DLOG(XMU**2/AM1**2))
      A2 = AM2**2*(1+DLOG(XMU**2/AM2**2))
      AQQ= RMQ**2*(1+DLOG(XMU**2/RMQ**2))
      AG = AMG**2*(1+DLOG(XMU**2/AMG**2))
      BQ0Q = B02_HDEC(RMQ**2,0.D0,RMQ,XMU**2)
      B101 = B02_HDEC(AM1**2,0.D0,AM1,XMU**2)
      B202 = B02_HDEC(AM2**2,0.D0,AM2,XMU**2)
      B1GQ = B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
      B2GQ = B02_HDEC(AM2**2,AMG,RMQ,XMU**2)
      BQG1 = B02_HDEC(RMQ**2,AMG,AM1,XMU**2)
      BQG2 = B02_HDEC(RMQ**2,AMG,AM2,XMU**2)
      B1QG1 = BB1(RMQ**2,AMG,AM1,XMU)
      B1QG2 = BB1(RMQ**2,AMG,AM2,XMU)
      IF(IONSH.EQ.1)THEN
       DMQ = -(AQQ/RMQ**2-1+2*BQ0Q+B1QG1+B1QG2
     .       + AMG*2*(AQ-AMU*RQ)/(AM1**2-AM2**2)*(BQG1-BQG2)
     .       + DELSUSY)
      ELSE
       DMQ = -(3*DLOG(XMU**2/QQ**2)+B1QG1+B1QG2
     .       + AMG*2*(AQ-AMU*RQ)/(AM1**2-AM2**2)*(BQG1-BQG2)
     .       + DELSUSY) + DELTA_Q
      ENDIF
      DM12 = (1+C2T**2)*A1+S2T**2*A2-2*AG-2*AQQ-4*AM1**2*B101
     .     - 2*(AMG**2+RMQ**2-AM1**2-2*AMG*RMQ*S2T)*B1GQ
      DM22 = (1+C2T**2)*A2+S2T**2*A1-2*AG-2*AQQ-4*AM2**2*B202
     .     - 2*(AMG**2+RMQ**2-AM2**2+2*AMG*RMQ*S2T)*B2GQ
      RM12 = -4*(AMG**2-AMG*RMQ*S2T) * DLOG(XMU**2/QQ**2)
     .     + (2*RMQ**2-(AM2**2-AM1**2)*S2T**2/2)*DMQ
      RM22 = -4*(AMG**2+AMG*RMQ*S2T) * DLOG(XMU**2/QQ**2)
     .     + (2*RMQ**2+(AM2**2-AM1**2)*S2T**2/2)*DMQ
      DMQ  = ALP*DMQ
      DM12 = ALP*DM12
      DM22 = ALP*DM22
      RM12 = ALP*RM12
      RM22 = ALP*RM22
      DEL1 = -DM12 + RM12 - DIF1
      DEL2 = -DM22 + RM22 - DIF2
      AMSQL2 = RML**2 + DL
      AMSQR2 = RMR**2 + DR
      DEL    = (AMSQL2-AMSQR2)**2+4*RMQ**2*AMLR**2
      AMSQ12 = RMQ**2 + (AMSQL2+AMSQR2-DSQRT(DEL))/2 + DEL1
      AMSQ22 = RMQ**2 + (AMSQL2+AMSQR2+DSQRT(DEL))/2 + DEL2
      RM1=DSQRT(AMSQ12)
      RM2=DSQRT(AMSQ22)
      IF(AMSQL2.EQ.AMSQR2) THEN
       THET = PI/4
      ELSE
       THET=0.5D0*DATAN(2.D0*RMQ*AMLR / (AMSQL2-AMSQR2) )
       IF(AMSQL2.GT.AMSQR2) THET = THET + PI/2
      ENDIF
      CT= DCOS(THET)
      ST= DSIN(THET) 
      C2T = CT**2-ST**2
      S2T = 2*ST*CT
      CT2 = CT**2
      ST2 = ST**2
      RML0 = DSQRT(RM1**2*CT2 + RM2**2*ST2 - RMQ**2 - DL)
      RMR0 = DSQRT(RM1**2*ST2 + RM2**2*CT2 - RMQ**2 - DR)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      RML0 = RML0 + FAC*(RML-RML0)
      RMR0 = RMR0 + FAC*(RMR-RMR0)
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      DIF1 = (AMSQL2+AMSQR2-DSQRT(DEL))/2
     .     - (AMSQL20+AMSQR20-DSQRT(DEL0))/2
      DIF2 = (AMSQL2+AMSQR2+DSQRT(DEL))/2
     .     - (AMSQL20+AMSQR20+DSQRT(DEL0))/2
      DIF10 = DIF1
      DIF20 = DIF2
      DELL = DABS((RML0-RML)/RML)
      DELR = DABS((RMR0-RMR)/RMR)
      DELT1 = DABS((RM1-AM1)/AM1)
      DELT2 = DABS((RM2-AM2)/AM2)
      XMSQL2 = RML0**2 + DL
      XMSQR2 = RMR0**2 + DR
      DEL    = (XMSQL2-XMSQR2)**2+4*RMQ**2*AMLR**2
      XMSQ12 = RMQ**2 + (XMSQL2+XMSQR2-DSQRT(DEL))/2
      XMSQ22 = RMQ**2 + (XMSQL2+XMSQR2+DSQRT(DEL))/2
      XM1=DSQRT(XMSQ12)
      XM2=DSQRT(XMSQ22)
      IF(XMSQL2.EQ.XMSQR2) THEN
       THET0 = PI/4
      ELSE
       THET0=0.5D0*DATAN(2.D0*RMQ*AMLR / (XMSQL2-XMSQR2) )
       IF(XMSQL2.GT.XMSQR2) THET0 = THET0 + PI/2
      ENDIF
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     IF(DELL.GT.ACC.OR.DELR.GT.ACC) GOTO 100
      IF(DELT1.GT.ACC.OR.DELT2.GT.ACC) GOTO 100
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      RML = RML0
      RMR = RMR0
      AM1 = RM1
      AM2 = RM2
c     write(6,*)'Iteration0:',III,AM1,AM2,THET/PI,RML,RMR
      DTH = 0
      IF(IDTH.NE.0)THEN
       THET0 = THET
       THET1 = THET
       ST = DSIN(THET1)
       CT = DCOS(THET1)
       C2T0 = CT**2-ST**2
       S2T0 = 2*ST*CT
       IF(IDTH.EQ.1)THEN
        BTGQ = (B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
     .         +B02_HDEC(AM2**2,AMG,RMQ,XMU**2))/2
       ELSEIF(IDTH.EQ.21)THEN
        BTGQ = B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
       ELSEIF(IDTH.EQ.22)THEN
        BTGQ = B02_HDEC(AM2**2,AMG,RMQ,XMU**2)
       ENDIF
       DTH = (S2T*C2T*(A2-A1)+4*AMG*RMQ*C2T*BTGQ)/(AM1**2-AM2**2)
       DAQ = 4 * AMG*DLOG(XMU**2/QQ**2)
       DTH0= S2T/2/C2T*(DMQ + DAQ/(AQQ-AMU*RQ)
     .     - (DM12-DM22)/(AM1**2-AM2**2))
       DTH = ALP*DTH
       DTH0= ALP*DTH0
       THET = THET0 + DTH0 - DTH
       DTH = THET-THET0
      ENDIF
c     write(6,*)'Iteration: ',III,AM1,AM2,THET/PI,RML,RMR
      XMSQL2 = RML**2 + DL
      XMSQR2 = RMR**2 + DR
      DEL    = (XMSQL2-XMSQR2)**2+4*RMQ**2*AMLR**2
      XMSQ12 = RMQ**2 + (XMSQL2+XMSQR2-DSQRT(DEL))/2
      XMSQ22 = RMQ**2 + (XMSQL2+XMSQR2+DSQRT(DEL))/2
      XM1=DSQRT(XMSQ12)
      XM2=DSQRT(XMSQ22)
      IF(XMSQL2.EQ.XMSQR2) THEN
       THET = PI/4
      ELSE
       THET=0.5D0*DATAN(2.D0*RMQ*AMLR / (XMSQL2-XMSQR2) )
       IF(XMSQL2.GT.XMSQR2) THET = THET + PI/2
      ENDIF
      IF(IDTH.NE.0)THEN
       THET = THET + DTH
      ENDIF
      DELT1 = DABS((AM1-XM1)/AM1)
      DELT2 = DABS((AM2-XM2)/AM2)
      DDS = DABS(2*RMQ*AMLR/(AM1**2-AM2**2))-1
      DDC = DABS((AMSQL2-AMSQR2)/(AM1**2-AM2**2))-1
      IERR = 0
      IF(DDS.GT.0.D0.OR.DDC.GT.0.D0) IERR=1
c     write(6,*)'Check:     ',III,XM1,XM2,THET/PI,
c    .                        DEL1/XM1**2,DEL2/XM2**2
c     write(6,*)'Check0:    ',III,DEL1,DEL2,IERR
c     write(6,*)'Check1:    ',III,DCOS(2*THET),DSIN(2*THET)
      AMSQL2 = RML**2 + DL
      AMSQR2 = RMR**2 + DR
c     write(6,*)'Check2:    ',III,(AMSQL2-AMSQR2)/(AM1**2-AM2**2),
c    .                            2.D0*RMQ*AMLR/(AM1**2-AM2**2)
c     write(6,*)
      RETURN
      END 

      SUBROUTINE SFERMIT0_HDEC(ISO,QQ,RMQ0,AMG,AML,AMR,AQ,AMU,TB,
     .                        AMW,AMZ,AM1,AM2,RML,RMR,THET,FACT)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      COMMON/SQUARKHIGGS_HDEC/THEB,XMG,IONSH,IDTH
      BB1(P2,AM1,AM2,XMU)=((AM1**2-AM2**2)*B02_HDEC(0.D0,AM1,AM2,XMU**2)
     .           -(P2+AM1**2-AM2**2)*B02_HDEC(P2,AM1,AM2,XMU**2))/2/P2
      ACC = 1.D-10
      DELSUSY = 1
      PI = 4*DATAN(1.D0)
      SW2=1.D0-AMW**2/AMZ**2
      XMU = QQ
      CF = 4/3.D0
      ALP = CF*ALPHAS_HDEC(QQ,3)/4/PI
c************************************
c--factor C_F/4 extracted!!!
c************************************
      IF(ISO.EQ.1)THEN
       AI3 = 0.5D0
       EQ  = 2/3.D0
       RQ = 1/TB
      ELSE
       AI3 =-0.5D0
       EQ  =-1/3.D0
       RQ = TB
      ENDIF
      DL = (AI3-EQ*SW2)*AMZ**2*(1-TB**2)/(1+TB**2)
      DR = EQ*SW2*AMZ**2*(1-TB**2)/(1+TB**2)
      RML0 = AML
      RMR0 = AMR
C--INITIALIZATION
      AMSQL20= AML**2 + DL
      AMSQR20= AMR**2 + DR
      AMLR   = AQ - AMU*RQ
      XMQ = RMQ0*FACT
      DEL0   = (AMSQL20-AMSQR20)**2+4*XMQ**2*AMLR**2
      AMSQ12 = XMQ**2 + (AMSQL20+AMSQR20-DSQRT(DEL0))/2
      AMSQ22 = XMQ**2 + (AMSQL20+AMSQR20+DSQRT(DEL0))/2
      AM10=DSQRT(AMSQ12)
      AM20=DSQRT(AMSQ22)
      IF(AMSQL20.EQ.AMSQR20) THEN
       THET = PI/4
      ELSE
       THET=0.5D0*DATAN(2.D0*RMQ0*AMLR / (AMSQL20-AMSQR20) )
       IF(AMSQL20.GT.AMSQR20) THET = THET + PI/2
      ENDIF
      CT= DCOS(THET)
      ST= DSIN(THET) 
      C2T0= CT**2-ST**2
      S2T0= 2*ST*CT
      C2T = C2T0
      S2T = S2T0
      DIF1 = 0
      DIF2 = 0
      DIF10= AM10**2
      DIF20= AM20**2
      RM1 = AM10
      RM2 = AM20
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'Init: m10,m20,theta/pi,ml,mr'
c     write(6,*)'Init1: c2t0,s2t0'
c     write(6,*)'Iteration0: i,m1,m2,theta/pi,ml,mr'
c     write(6,*)'Iteration: i,m1,m2,theta/pi,ml,mr'
c     write(6,*)'Check: I,m1,m2,theta/pi'
c     write(6,*)'Check0: I,del1,del2,ierr'
c     write(6,*)'Check1: I,c2t,s2t'
c     write(6,*)'Check2: I,c2t0,s2t0'
c     write(6,*)
c     write(6,*)'without iteration'
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     write(6,*)'FACT =   ',FACT
c     write(6,*)'Init:    ',AM10,AM20,THET/PI,AML,AMR
c     write(6,*)'Init1:   ',(AMSQL20-AMSQR20)/(AM10**2-AM20**2),
c    .                      2*RMQ0*AMLR/(AM10**2-AM20**2)
      III = 0
100   RML = RML0
      RMR = RMR0
      AM1 = RM1
      AM2 = RM2
      QS = (AM1+AM2+DABS(AMG))/3
      ALPS = CF*ALPHAS_HDEC(QS,3)/4/PI
      DELTA_Q = 0
      IF(ISO.EQ.2) DELTA_Q = 2*AMG*AMU*RQ*T_HDEC(AM1,AM2,AMG)
      RMQ = RMQ0
      IF(IONSH.EQ.0)THEN
       RMQ = RMQ0/(1+ALPS*DELTA_Q)
c      RMQ = RMQ0
       IF(ISO.EQ.2) THEN
c       write(6,*)'MQ0/eff: ',RMQ0,RMQ,RMQ/RMQ0
c       write(6,*)'MQ0/eff,M1/2,Mg,als: ',RMQ0,RMQ,AM1,AM2,AMG,
c    .            ALPHAS_HDEC(QS,3)
c       write(6,*)'Delta_mb: ',T_HDEC(AM1,AM2,AMG),ALPS*DELTA_Q
        AMSQL20= RML**2 + DL
        AMSQR20= RMR**2 + DR
        DEL0   = (AMSQL20-AMSQR20)**2+4*RMQ**2*AMLR**2
        AMSQ12 = RMQ**2 + (AMSQL20+AMSQR20-DSQRT(DEL0))/2
        AMSQ22 = RMQ**2 + (AMSQL20+AMSQR20+DSQRT(DEL0))/2
        AM1=DSQRT(AMSQ12)
        AM2=DSQRT(AMSQ22)
        DELTA_Q = 2*AMG*AMU*RQ*T_HDEC(AM1,AM2,AMG)
        IF(AMSQL20.EQ.AMSQR20) THEN
         THET = PI/4
        ELSE
         THET=0.5D0*DATAN(2.D0*RMQ*AMLR / (AMSQL20-AMSQR20) )
         IF(AMSQL20.GT.AMSQR20) THET = THET + PI/2
        ENDIF
        CT= DCOS(THET)
        ST= DSIN(THET) 
        C2T= CT**2-ST**2
        S2T= 2*ST*CT
c       write(6,*)'delmb0: ',alp*delta_q,am1,am2
       ENDIF
      ENDIF
      DEL0   = (AMSQL20-AMSQR20)**2+4*RMQ**2*AMLR**2
      III = III+1
      A1 = AM1**2*(1+DLOG(XMU**2/AM1**2))
      A2 = AM2**2*(1+DLOG(XMU**2/AM2**2))
      AQQ= RMQ**2*(1+DLOG(XMU**2/RMQ**2))
      AG = AMG**2*(1+DLOG(XMU**2/AMG**2))
      BQ0Q = B02_HDEC(RMQ**2,0.D0,RMQ,XMU**2)
      B101 = B02_HDEC(AM1**2,0.D0,AM1,XMU**2)
      B202 = B02_HDEC(AM2**2,0.D0,AM2,XMU**2)
      B1GQ = B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
      B2GQ = B02_HDEC(AM2**2,AMG,RMQ,XMU**2)
      BQG1 = B02_HDEC(RMQ**2,AMG,AM1,XMU**2)
      BQG2 = B02_HDEC(RMQ**2,AMG,AM2,XMU**2)
      B1QG1 = BB1(RMQ**2,AMG,AM1,XMU)
      B1QG2 = BB1(RMQ**2,AMG,AM2,XMU)
      IF(IONSH.EQ.1)THEN
       DMQ = -(AQQ/RMQ**2-1+2*BQ0Q+B1QG1+B1QG2
     .       + AMG*2*(AQ-AMU*RQ)/(AM1**2-AM2**2)*(BQG1-BQG2)
     .       + DELSUSY)
      ELSE
       DMQ = -(3*DLOG(XMU**2/QQ**2)+B1QG1+B1QG2
     .       + AMG*2*(AQ-AMU*RQ)/(AM1**2-AM2**2)*(BQG1-BQG2)
     .       + DELSUSY) + DELTA_Q
c      write(6,*)'delmb1: ',alp*delta_q
      ENDIF
      DM12 = (1+C2T**2)*A1+S2T**2*A2-2*AG-2*AQQ-4*AM1**2*B101
     .     - 2*(AMG**2+RMQ**2-AM1**2-2*AMG*RMQ*S2T)*B1GQ
      DM22 = (1+C2T**2)*A2+S2T**2*A1-2*AG-2*AQQ-4*AM2**2*B202
     .     - 2*(AMG**2+RMQ**2-AM2**2+2*AMG*RMQ*S2T)*B2GQ
      RM12 = -4*(AMG**2-AMG*RMQ*S2T) * DLOG(XMU**2/QQ**2)
     .     + (2*RMQ**2-(AM2**2-AM1**2)*S2T**2/2)*DMQ
      RM22 = -4*(AMG**2+AMG*RMQ*S2T) * DLOG(XMU**2/QQ**2)
     .     + (2*RMQ**2+(AM2**2-AM1**2)*S2T**2/2)*DMQ
      DMQ  = ALP*DMQ
      DM12 = ALP*DM12
      DM22 = ALP*DM22
      RM12 = ALP*RM12
      RM22 = ALP*RM22
      DEL1 = -DM12 + RM12 - DIF1
      DEL2 = -DM22 + RM22 - DIF2
c     write(6,*)'mass shift: ',DEL1,DEL2,-DM12,RM12,-DM22,RM22
c     write(6,*)'mass shift: ',DMQ,XMU
c     write(6,*)'mass shift: ',
c    .       -(3*DLOG(XMU**2/QQ**2)+DELSUSY)*ALP,
c    .       ALP*DELTA_Q,
c    .       -(B1QG1+B1QG2)*ALP,
c    .       -(AMG*2*(AQ-AMU*RQ)/(AM1**2-AM2**2)*(BQG1-BQG2))*ALP
c     write(6,*)'mass shift: ',B1QG1,B1QG2
c     xx1 = B02_HDEC(RMQ**2,AMG,AM1,XMU**2)
c     xx2 = B02_HDEC(RMQ**2,AMG,AM2,XMU**2)
c     x01 = B02_HDEC(0.D0,AMG,AM1,XMU**2)
c     x02 = B02_HDEC(0.D0,AMG,AM2,XMU**2)
c     xp1 = BP02_HDEC(0.D0,AMG,AM1,XMU**2)
c     xp2 = BP02_HDEC(0.D0,AMG,AM2,XMU**2)
c     zz1 = ((AMG**2-AM1**2)*X01-(RMQ**2+AMG**2-AM1**2)*XX1)/2/RMQ**2
c     zz2 = ((AMG**2-AM2**2)*X02-(RMQ**2+AMG**2-AM2**2)*XX2)/2/RMQ**2
c     zp1 = -X01/2 - (AMG**2-AM1**2)/2*xp1
c     zp2 = -X02/2 - (AMG**2-AM2**2)/2*xp2
c     write(6,*)'B0 funct:   ',RMQ,AMG,AM1,AM2,XMU
c     write(6,*)'B0 funct:   ',XX1,XX2,X01,X02,zz1,zz2,zp1,zp2
      AMSQL2 = RML**2 + DL
      AMSQR2 = RMR**2 + DR
      DEL    = (AMSQL2-AMSQR2)**2+4*RMQ**2*AMLR**2
      AMSQ12 = RMQ**2 + (AMSQL2+AMSQR2-DSQRT(DEL))/2 + DEL1
      AMSQ22 = RMQ**2 + (AMSQL2+AMSQR2+DSQRT(DEL))/2 + DEL2
c     write(6,*)'Interim: '
c    . ,'III,DEL1,DEL2,DMQ,DM12,DM22,RM12,RM22,ALPS,DELTA_Q,RMQ/RMQ0'
c     write(6,*)'Interim: ',III,DEL1,DEL2,DMQ,DM12,DM22,RM12,RM22
c    .                     ,ALPS,DELTA_Q,RMQ/RMQ0
c     write(6,*)'Interim2: RMQ,AM1,AM2,S2T'
c     write(6,*)'Interim2:',RMQ,AM1,AM2,S2T
      RM1=DSQRT(AMSQ12)
      RM2=DSQRT(AMSQ22)
      IF(AMSQL2.EQ.AMSQR2) THEN
       THET = PI/4
      ELSE
       THET=0.5D0*DATAN(2.D0*RMQ*AMLR / (AMSQL2-AMSQR2) )
       IF(AMSQL2.GT.AMSQR2) THET = THET + PI/2
      ENDIF
      CT= DCOS(THET)
      ST= DSIN(THET) 
      C2T = CT**2-ST**2
      S2T = 2*ST*CT
      CT2 = CT**2
      ST2 = ST**2
      RML0 = DSQRT(RM1**2*CT2 + RM2**2*ST2 - RMQ**2 - DL)
      RMR0 = DSQRT(RM1**2*ST2 + RM2**2*CT2 - RMQ**2 - DR)
c     write(6,*)'st^2, ct^2:  ',ST2,CT2
c     write(6,*)'ML/R:        ',dsqrt(RML0**2+DL),dsqrt(RMR0**2+DR)
c     write(6,*)'MQ:          ',RMQ
c     write(6,*)'M1/2 tree:   ',DSQRT(RM1**2-DEL1),DSQRT(RM2**2-DEL2)
c     write(6,*)'M1/2 before: ',RM1,RM2
      DIF1 = (AMSQL2+AMSQR2-DSQRT(DEL))/2
     .     - (AMSQL20+AMSQR20-DSQRT(DEL0))/2
      DIF2 = (AMSQL2+AMSQR2+DSQRT(DEL))/2
     .     - (AMSQL20+AMSQR20+DSQRT(DEL0))/2
      DIF10 = DIF1
      DIF20 = DIF2
      DELL = DABS((RML0-RML)/RML)
      DELR = DABS((RMR0-RMR)/RMR)
      DELT1 = DABS((RM1-AM1)/AM1)
      DELT2 = DABS((RM2-AM2)/AM2)
      XMSQL2 = RML0**2 + DL
      XMSQR2 = RMR0**2 + DR
      DEL    = (XMSQL2-XMSQR2)**2+4*RMQ**2*AMLR**2
      XMSQ12 = RMQ**2 + (XMSQL2+XMSQR2-DSQRT(DEL))/2
      XMSQ22 = RMQ**2 + (XMSQL2+XMSQR2+DSQRT(DEL))/2
      XM1=DSQRT(XMSQ12)
      XM2=DSQRT(XMSQ22)
c     write(6,*)'M1/2 after:  ',XM1,XM2
      IF(XMSQL2.EQ.XMSQR2) THEN
       THET0 = PI/4
      ELSE
       THET0=0.5D0*DATAN(2.D0*RMQ*AMLR / (XMSQL2-XMSQR2) )
       IF(XMSQL2.GT.XMSQR2) THET0 = THET0 + PI/2
      ENDIF
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c     IF(DELL.GT.ACC.OR.DELR.GT.ACC) GOTO 100
c     IF(DELT1.GT.ACC.OR.DELT2.GT.ACC) GOTO 100
c>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      RML = RML0
      RMR = RMR0
      AM1 = XM1
      AM2 = XM2
c     write(6,*)'Iteration0:',III,AM1,AM2,THET/PI,RML,RMR
      DTH = 0
      IF(IDTH.NE.0)THEN
       THET  = THET0
       THET1 = THET0
       ST = DSIN(THET1)
       CT = DCOS(THET1)
       C2T = CT**2-ST**2
       S2T = 2*ST*CT
       A1 = AM1**2*(1+DLOG(XMU**2/AM1**2))
       A2 = AM2**2*(1+DLOG(XMU**2/AM2**2))
       AQQ= RMQ**2*(1+DLOG(XMU**2/RMQ**2))
       AG = AMG**2*(1+DLOG(XMU**2/AMG**2))
       B101 = B02_HDEC(AM1**2,0.D0,AM1,XMU**2)
       B202 = B02_HDEC(AM2**2,0.D0,AM2,XMU**2)
       B1GQ = B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
       B2GQ = B02_HDEC(AM2**2,AMG,RMQ,XMU**2)
       BQG1 = B02_HDEC(RMQ**2,AMG,AM1,XMU**2)
       BQG2 = B02_HDEC(RMQ**2,AMG,AM2,XMU**2)
       B1QG1 = BB1(RMQ**2,AMG,AM1,XMU)
       B1QG2 = BB1(RMQ**2,AMG,AM2,XMU)
       DMQ = -(3*DLOG(XMU**2/QQ**2)+B1QG1+B1QG2
     .       + AMG*2*(AQ-AMU*RQ)/(AM1**2-AM2**2)*(BQG1-BQG2)
     .       + DELSUSY) + DELTA_Q
       DM12 = (1+C2T**2)*A1+S2T**2*A2-2*AG-2*AQQ-4*AM1**2*B101
     .      - 2*(AMG**2+RMQ**2-AM1**2-2*AMG*RMQ*S2T)*B1GQ
       DM22 = (1+C2T**2)*A2+S2T**2*A1-2*AG-2*AQQ-4*AM2**2*B202
     .      - 2*(AMG**2+RMQ**2-AM2**2+2*AMG*RMQ*S2T)*B2GQ
       DAQ = 4 * AMG*DLOG(XMU**2/QQ**2)
       IF(IDTH.EQ.1)THEN
        BTGQ = (B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
     .         +B02_HDEC(AM2**2,AMG,RMQ,XMU**2))/2
       ELSEIF(IDTH.EQ.21)THEN
        BTGQ = B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
       ELSEIF(IDTH.EQ.22)THEN
        BTGQ = B02_HDEC(AM2**2,AMG,RMQ,XMU**2)
       ENDIF
       DTH = (S2T*C2T*(A2-A1)+4*AMG*RMQ*C2T*BTGQ)/(AM1**2-AM2**2)
       DTH0= S2T/2/C2T*(DMQ + DAQ/(AQQ-AMU*RQ)
     .     - (DM12-DM22)/(AM1**2-AM2**2))
       DTH = ALP*DTH
       DTH0= ALP*DTH0
       THET = THET0 + DTH0 - DTH
c      write(6,*)'angleshift: ',dth0,dth
       DTH = THET-THET0
c      write(6,*)'angleshift: ',dth,xmu

c      write(6,*)'stop_ang0:',DCOS(THET0),DSIN(THET0)
c      write(6,*)'stop_ang1:',DCOS(THET),DSIN(THET)
c      write(6,*)'stop_ang0:',THET0/PI,(THET-THET0)/THET0
c      write(6,*)'stop_ang1:',THET/PI,DTH0/PI,(DTH0-DTH)/PI
c      write(6,*)'stop_mass:',AM1,AM2
c      write(6,*)'stop_ang: ',S2T,C2T,S2T/2/C2T,DMQ,DAQ/(AQQ-AMU*RQ),
c    .                        -(DM12-DM22)/(AM1**2-AM2**2)
      ENDIF
c     write(6,*)'Iteration: ',III,AM1,AM2,THET/PI,RML,RMR
c     write(6,*)'Input:     ',ISO,QQ,RMQ0,AMG,AML,AMR,AQ,AMU,TB,
c    .                        AMW,AMZ
      QS = (AM1+AM2+DABS(AMG))/3
      ALPS = CF*ALPHAS_HDEC(QS,3)/4/PI
      IF(ISO.EQ.2) DELTA_Q = 2*AMG*AMU*RQ*T_HDEC(AM1,AM2,AMG)
      RMQ = RMQ0
      IF(IONSH.EQ.0)THEN
       RMQ = RMQ0/(1+ALPS*DELTA_Q)
      ENDIF
      XMSQL2 = RML**2 + DL
      XMSQR2 = RMR**2 + DR
      DEL    = (XMSQL2-XMSQR2)**2+4*RMQ**2*AMLR**2
      XMSQ12 = RMQ**2 + (XMSQL2+XMSQR2-DSQRT(DEL))/2
      XMSQ22 = RMQ**2 + (XMSQL2+XMSQR2+DSQRT(DEL))/2
      XM1=DSQRT(XMSQ12)
      XM2=DSQRT(XMSQ22)
      IF(XMSQL2.EQ.XMSQR2) THEN
       THET = PI/4
      ELSE
       THET=0.5D0*DATAN(2.D0*RMQ*AMLR / (XMSQL2-XMSQR2) )
       IF(XMSQL2.GT.XMSQR2) THET = THET + PI/2
      ENDIF
      IF(IDTH.NE.0)THEN
       THET = THET + DTH
      ENDIF
      AM1 = XM1
      AM2 = XM2
c     write(6,*)'M1/2 after:  ',XM1,XM2
c     write(6,*)'Check:     ',III,AM1,AM2,THET/PI,
c    .                        DEL1/XM1**2,DEL2/XM2**2
c     write(6,*)'Check0:    ',III,DEL1,DEL2,IERR
c     write(6,*)'Check1:    ',III,DCOS(2*THET),DSIN(2*THET)
c     AMSQL2 = RML**2 + DL
c     AMSQR2 = RMR**2 + DR
c     write(6,*)'Check2:    ',III,(AMSQL2-AMSQR2)/(AM1**2-AM2**2),
c    .                            2.D0*RMQ*AMLR/(AM1**2-AM2**2)
c     write(6,*)'mixing0:   ',DSIN(THET-DTH),DCOS(THET-DTH)
c     write(6,*)'mixing:    ',DSIN(THET),DCOS(THET),DTH/THET
c     write(6,*)
c     DELTA_Q = 2*AMG*AMU*RQ*T_HDEC(AM1,AM2,AMG)
      RMQ1 = RMQ0/(1+ALPS*DELTA_Q)
      FACT = RMQ1/RMQ0
      RETURN
      END 

      SUBROUTINE SFERMIT1_HDEC(ISO,QQ,RMQ0,AMG,AML,AMR,AQ,AMU,TB,
     .                        AMW,AMZ,AM1,AM2,RML,RMR,THET)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      COMMON/SQUARKHIGGS_HDEC/THEB,XMG,IONSH,IDTH
      BB1(P2,AM1,AM2,XMU)=((AM1**2-AM2**2)*B02_HDEC(0.D0,AM1,AM2,XMU**2)
     .           -(P2+AM1**2-AM2**2)*B02_HDEC(P2,AM1,AM2,XMU**2))/2/P2
      ACC = 1.D-8
      DELSUSY = 1
      PI = 4*DATAN(1.D0)
      SW2=1.D0-AMW**2/AMZ**2
      XMU = QQ
      CF = 4/3.D0
      ALP = CF*ALPHAS_HDEC(QQ,3)/4/PI
c************************************
c--factor C_F/4 extracted!!!
c************************************
      IF(ISO.EQ.1)THEN
       AI3 = 0.5D0
       EQ  = 2/3.D0
       RQ = 1/TB
      ELSE
       AI3 =-0.5D0
       EQ  =-1/3.D0
       RQ = TB
      ENDIF
      DL = (AI3-EQ*SW2)*AMZ**2*(1-TB**2)/(1+TB**2)
      DR = EQ*SW2*AMZ**2*(1-TB**2)/(1+TB**2)
      RML0 = AML
      RMR0 = AMR
C--INITIALIZATION
      AMSQL20= AML**2 + DL
      AMSQR20= AMR**2 + DR
      AMLR   = AQ - AMU*RQ
      RMQ = RMQ0
      DEL0   = (AMSQL20-AMSQR20)**2+4*RMQ**2*AMLR**2
      AMSQ12 = RMQ**2 + (AMSQL20+AMSQR20-DSQRT(DEL0))/2
      AMSQ22 = RMQ**2 + (AMSQL20+AMSQR20+DSQRT(DEL0))/2
      AM10=DSQRT(AMSQ12)
      AM20=DSQRT(AMSQ22)
      IF(AMSQL20.EQ.AMSQR20) THEN
       THET = PI/4
      ELSE
       THET=0.5D0*DATAN(2.D0*RMQ*AMLR / (AMSQL20-AMSQR20) )
       IF(AMSQL20.GT.AMSQR20) THET = THET + PI/2
      ENDIF
      CT= DCOS(THET)
      ST= DSIN(THET) 
      C2T= CT**2-ST**2
      S2T= 2*ST*CT
      AM1 = AM10
      AM2 = AM20
      RML = RML0
      RMR = RMR0
c     write(6,*)
c     write(6,*)'Init1:   '
c     write(6,*)'Init:    ',AM1,AM2,THET/PI,AML,AMR
      III = 0
c--begin iteration
c100   III = III+1
      do iii=1,10
      QS = (AM1+AM2+DABS(AMG))/3
      ALPS = CF*ALPHAS_HDEC(QS,3)/4/PI
      DELTA_Q = 2*AMG*AMU*RQ*T_HDEC(AM1,AM2,AMG)
      RMQ = RMQ0/(1+ALPS*DELTA_Q)
      AMSQL20= RML**2 + DL
      AMSQR20= RMR**2 + DR
      DEL0   = (AMSQL20-AMSQR20)**2+4*RMQ**2*AMLR**2
      AMSQ12 = RMQ**2 + (AMSQL20+AMSQR20-DSQRT(DEL0))/2
      AMSQ22 = RMQ**2 + (AMSQL20+AMSQR20+DSQRT(DEL0))/2
      AM1=DSQRT(AMSQ12)
      AM2=DSQRT(AMSQ22)
      IF(AMSQL20.EQ.AMSQR20) THEN
       THET = PI/4
      ELSE
       THET=0.5D0*DATAN(2.D0*RMQ*AMLR / (AMSQL20-AMSQR20) )
       IF(AMSQL20.GT.AMSQR20) THET = THET + PI/2
      ENDIF
      CT= DCOS(THET)
      ST= DSIN(THET) 
      C2T= CT**2-ST**2
      S2T= 2*ST*CT
      DELTA_Q = 2*AMG*AMU*RQ*T_HDEC(AM1,AM2,AMG)
      A1 = AM1**2*(1+DLOG(XMU**2/AM1**2))
      A2 = AM2**2*(1+DLOG(XMU**2/AM2**2))
      AQQ= RMQ**2*(1+DLOG(XMU**2/RMQ**2))
      AG = AMG**2*(1+DLOG(XMU**2/AMG**2))
      BQ0Q = B02_HDEC(RMQ**2,0.D0,RMQ,XMU**2)
      B101 = B02_HDEC(AM1**2,0.D0,AM1,XMU**2)
      B202 = B02_HDEC(AM2**2,0.D0,AM2,XMU**2)
      B1GQ = B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
      B2GQ = B02_HDEC(AM2**2,AMG,RMQ,XMU**2)
      BQG1 = B02_HDEC(RMQ**2,AMG,AM1,XMU**2)
      BQG2 = B02_HDEC(RMQ**2,AMG,AM2,XMU**2)
      B1QG1 = BB1(RMQ**2,AMG,AM1,XMU)
      B1QG2 = BB1(RMQ**2,AMG,AM2,XMU)
      DMQ = -(3*DLOG(XMU**2/QQ**2)+B1QG1+B1QG2
     .      + AMG*2*(AQ-AMU*RQ)/(AM1**2-AM2**2)*(BQG1-BQG2)
     .      + DELSUSY) + DELTA_Q
      DM12 = (1+C2T**2)*A1+S2T**2*A2-2*AG-2*AQQ-4*AM1**2*B101
     .     - 2*(AMG**2+RMQ**2-AM1**2-2*AMG*RMQ*S2T)*B1GQ
      DM22 = (1+C2T**2)*A2+S2T**2*A1-2*AG-2*AQQ-4*AM2**2*B202
     .     - 2*(AMG**2+RMQ**2-AM2**2+2*AMG*RMQ*S2T)*B2GQ
      RM12 = -4*(AMG**2-AMG*RMQ*S2T) * DLOG(XMU**2/QQ**2)
     .     + (2*RMQ**2-(AM2**2-AM1**2)*S2T**2/2)*DMQ
      RM22 = -4*(AMG**2+AMG*RMQ*S2T) * DLOG(XMU**2/QQ**2)
     .     + (2*RMQ**2+(AM2**2-AM1**2)*S2T**2/2)*DMQ
      DMQ  = ALP*DMQ
      DM12 = ALP*DM12
      DM22 = ALP*DM22
      RM12 = ALP*RM12
      RM22 = ALP*RM22
      DEL1 = -DM12 + RM12
      DEL2 = -DM22 + RM22
      AMSQL2 = RML**2 + DL
      AMSQR2 = RMR**2 + DR
      DEL    = (AMSQL2-AMSQR2)**2+4*RMQ**2*AMLR**2
      AMSQ12 = RMQ**2 + (AMSQL2+AMSQR2-DSQRT(DEL))/2 + DEL1
      AMSQ22 = RMQ**2 + (AMSQL2+AMSQR2+DSQRT(DEL))/2 + DEL2
      RM1=DSQRT(AMSQ12)
      RM2=DSQRT(AMSQ22)
      IF(AMSQL2.EQ.AMSQR2) THEN
       THET = PI/4
      ELSE
       THET=0.5D0*DATAN(2.D0*RMQ*AMLR / (AMSQL2-AMSQR2) )
       IF(AMSQL2.GT.AMSQR2) THET = THET + PI/2
      ENDIF
      CT= DCOS(THET)
      ST= DSIN(THET) 
      C2T = CT**2-ST**2
      S2T = 2*ST*CT
      CT2 = CT**2
      ST2 = ST**2
      RML0 = DSQRT(RM1**2*CT2 + RM2**2*ST2 - RMQ**2 - DL)
      RMR0 = DSQRT(RM1**2*ST2 + RM2**2*CT2 - RMQ**2 - DR)
      DELL = DABS((RML0-RML)/RML)
      DELR = DABS((RMR0-RMR)/RMR)
c     IF(DELL.GT.ACC.OR.DELR.GT.ACC) GOTO 100
c     QS = (AM1+AM2+AMG)/3
c     ALPS = CF*ALPHAS_HDEC(QS,3)/4/PI
c     DELTA_Q = 2*AMG*AMU*RQ*T_HDEC(AM1,AM2,AMG)
c     RMQ = RMQ0/(1+ALPS*DELTA_Q)
      AMSQL2 = RML0**2 + DL
      AMSQR2 = RMR0**2 + DR
      DEL    = (AMSQL2-AMSQR2)**2+4*RMQ**2*AMLR**2
      AMSQ12 = RMQ**2 + (AMSQL2+AMSQR2-DSQRT(DEL))/2
      AMSQ22 = RMQ**2 + (AMSQL2+AMSQR2+DSQRT(DEL))/2
      AM1=DSQRT(AMSQ12)
      AM2=DSQRT(AMSQ22)
      IF(AMSQL2.EQ.AMSQR2) THEN
       THET = PI/4
      ELSE
       THET=0.5D0*DATAN(2.D0*RMQ*AMLR / (AMSQL2-AMSQR2) )
       IF(AMSQL2.GT.AMSQR2) THET = THET + PI/2
      ENDIF
c     write(6,*)'Iteration: ',III,RM1,RM2,RMQ/RMQ0
c     write(6,*)'Check:     ',III,AM1,AM2,RML,RMR
      enddo
c--end iteration
      QS = (AM1+AM2+DABS(AMG))/3
      ALPS = CF*ALPHAS_HDEC(QS,3)/4/PI
      DELTA_Q = 2*AMG*AMU*RQ*T_HDEC(AM1,AM2,AMG)
      DTH = 0
      IF(IDTH.NE.0)THEN
       THET0 = THET
       THET1 = THET
       ST = DSIN(THET1)
       CT = DCOS(THET1)
       C2T = CT**2-ST**2
       S2T = 2*ST*CT
       A1 = AM1**2*(1+DLOG(XMU**2/AM1**2))
       A2 = AM2**2*(1+DLOG(XMU**2/AM2**2))
       AQQ= RMQ**2*(1+DLOG(XMU**2/RMQ**2))
       AG = AMG**2*(1+DLOG(XMU**2/AMG**2))
       B101 = B02_HDEC(AM1**2,0.D0,AM1,XMU**2)
       B202 = B02_HDEC(AM2**2,0.D0,AM2,XMU**2)
       B1GQ = B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
       B2GQ = B02_HDEC(AM2**2,AMG,RMQ,XMU**2)
       BQG1 = B02_HDEC(RMQ**2,AMG,AM1,XMU**2)
       BQG2 = B02_HDEC(RMQ**2,AMG,AM2,XMU**2)
       B1QG1 = BB1(RMQ**2,AMG,AM1,XMU)
       B1QG2 = BB1(RMQ**2,AMG,AM2,XMU)
       IF(IDTH.EQ.1)THEN
        BTGQ = (B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
     .         +B02_HDEC(AM2**2,AMG,RMQ,XMU**2))/2
       ELSEIF(IDTH.EQ.21)THEN
        BTGQ = B02_HDEC(AM1**2,AMG,RMQ,XMU**2)
       ELSEIF(IDTH.EQ.22)THEN
        BTGQ = B02_HDEC(AM2**2,AMG,RMQ,XMU**2)
       ENDIF
       DMQ = -(3*DLOG(XMU**2/QQ**2)+B1QG1+B1QG2
     .       + AMG*2*(AQ-AMU*RQ)/(AM1**2-AM2**2)*(BQG1-BQG2)
     .       + DELSUSY) + DELTA_Q
       DM12 = (1+C2T**2)*A1+S2T**2*A2-2*AG-2*AQQ-4*AM1**2*B101
     .      - 2*(AMG**2+RMQ**2-AM1**2-2*AMG*RMQ*S2T)*B1GQ
       DM22 = (1+C2T**2)*A2+S2T**2*A1-2*AG-2*AQQ-4*AM2**2*B202
     .      - 2*(AMG**2+RMQ**2-AM2**2+2*AMG*RMQ*S2T)*B2GQ
       DAQ = 4 * AMG*DLOG(XMU**2/QQ**2)
       DTH = (S2T*C2T*(A2-A1)+4*AMG*RMQ*C2T*BTGQ)/(AM1**2-AM2**2)
       DTH0= S2T/2/C2T*(DMQ + DAQ/(AQQ-AMU*RQ)
     .     - (DM12-DM22)/(AM1**2-AM2**2))
       DTH = ALP*DTH
       DTH0= ALP*DTH0
       THET = THET0 + DTH0 - DTH
       DTH = THET-THET0
      ENDIF
c     write(6,*)'Iteration: ',III,AM1,AM2,THET/PI,THET0/PI,RML,RMR
c     write(6,*)
      RETURN
      END 
