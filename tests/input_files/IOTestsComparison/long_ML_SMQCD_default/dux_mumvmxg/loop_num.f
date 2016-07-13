C     THE CORE SUBROUTINE CALLED BY CUTTOOLS WHICH CONTAINS THE HELAS
C      CALLS BUILDING THE LOOP

      SUBROUTINE ML5_0_LOOPNUM(Q,RES)
C     
C     CONSTANTS 
C     
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=2)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=39)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=10)
      INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=4)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C     
C     ARGUMENTS 
C     
      COMPLEX*16 Q(0:3)
      COMPLEX*16 RES
C     
C     LOCAL VARIABLES 
C     
      COMPLEX*16 CFTOT
      COMPLEX*16 BUFF
      INTEGER I,H
C     
C     GLOBAL VARIABLES
C     
      INTEGER WE(NEXTERNAL)
      INTEGER ID, SYMFACT, MULTIPLIER, AMPLNUM
      COMMON/ML5_0_LOOP/WE,ID,SYMFACT,MULTIPLIER,AMPLNUM

      LOGICAL GOODHEL(NCOMB)
      LOGICAL GOODAMP(NLOOPAMPS,NCOMB)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL

      INTEGER NTRY
      LOGICAL CHECKPHASE,HELDOUBLECHECKED
      REAL*8 REF
      COMMON/ML5_0_INIT/NTRY,CHECKPHASE,HELDOUBLECHECKED,REF

      INTEGER CF_D(NLOOPAMPS,NBORNAMPS)
      INTEGER CF_N(NLOOPAMPS,NBORNAMPS)
      COMMON/ML5_0_CF/CF_D,CF_N

      COMPLEX*16 AMP(NBORNAMPS,NCOMB)
      COMMON/ML5_0_AMPS/AMP
      COMPLEX*16 W(20,NWAVEFUNCS,NCOMB)
      COMMON/ML5_0_WFCTS/W

      INTEGER HELPICKED
      COMMON/ML5_0_HELCHOICE/HELPICKED

      RES=(0.0D0,0.0D0)

      DO H=1,NCOMB
        IF (((HELPICKED.EQ.-1).OR.(HELPICKED.EQ.H)).AND.((CHECKPHASE.OR
     $..NOT.HELDOUBLECHECKED).OR.(GOODHEL(H).AND.GOODAMP(AMPLNUM,H))))
     $    THEN
          CALL ML5_0_LOOPNUMHEL(-Q,BUFF,H)
          DO I=1,NBORNAMPS
            CFTOT=DCMPLX(CF_N(AMPLNUM,I)/DBLE(ABS(CF_D(AMPLNUM,I)))
     $       ,0.0D0)
            IF(CF_D(AMPLNUM,I).LT.0) CFTOT=CFTOT*IMAG1
            RES=RES+CFTOT*BUFF*DCONJG(AMP(I,H))
          ENDDO
        ENDIF
      ENDDO
      RES=(RES*MULTIPLIER)/SYMFACT

      END

      SUBROUTINE ML5_0_LOOPNUMHEL(Q,RES,H)
C     
C     CONSTANTS 
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=4)
      INTEGER    NMAXLOOPWFS
      PARAMETER (NMAXLOOPWFS=(NEXTERNAL+2))
      REAL*8     ZERO
      PARAMETER (ZERO=0.D0)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=10)
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=2)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=39)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     
C     ARGUMENTS 
C     
      COMPLEX*16 Q(0:3)
      COMPLEX*16 RES
      INTEGER H
C     
C     LOCAL VARIABLES 
C     
      COMPLEX*16 BUFF(4)
      COMPLEX*16 WL(20,NMAXLOOPWFS)
      INTEGER I
C     
C     GLOBAL VARIABLES
C     
      COMPLEX*16 LC(MAXLCOUPLINGS)
      COMPLEX*16 ML(NEXTERNAL+2)
      COMMON/ML5_0_DP_LOOP/LC,ML

      INTEGER WE(NEXTERNAL)
      INTEGER ID, SYMFACT,MULTIPLIER,AMPLNUM
      COMMON/ML5_0_LOOP/WE,ID,SYMFACT,MULTIPLIER,AMPLNUM

      COMPLEX*16 AMP(NBORNAMPS,NCOMB)
      COMMON/ML5_0_AMPS/AMP
      COMPLEX*16 W(20,NWAVEFUNCS,NCOMB)
      COMMON/ML5_0_WFCTS/W

C     ----------
C     BEGIN CODE
C     ----------
      RES=(0.D0,0.D0)
      IF (ID.EQ.1) THEN
C       Loop diagram number 3 (might be others, just an example)
        DO I=1,4
          CALL LCUT_F(Q(0),I,WL(1,2))
          CALL FFV1LP0_3(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3))
          CALL FFV1L_1(W(1,WE(2),H),WL(1,3),LC(2),ML(4),ZERO,WL(1,4))
          BUFF(I)=WL(I+4,4)
        ENDDO
        CALL CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.2) THEN
C       Loop diagram number 4 (might be others, just an example)
        DO I=1,4
          CALL LCUT_V(Q(0),I,WL(1,2))
          CALL FFV1L_1(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3))
          CALL FFV2L_1(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1,4))
          CALL FFV1LP0_3(W(1,WE(3),H),WL(1,4),LC(3),ML(5),ZERO,WL(1,5))
          BUFF(I)=WL(I+4,5)
        ENDDO
        CALL CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.3) THEN
C       Loop diagram number 5 (might be others, just an example)
        DO I=1,4
          CALL LCUT_V(Q(0),I,WL(1,2))
          CALL FFV1L_2(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3))
          CALL FFV2L_2(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1,4))
          CALL FFV1LP0_3(WL(1,4),W(1,WE(3),H),LC(3),ML(5),ZERO,WL(1,5))
          BUFF(I)=WL(I+4,5)
        ENDDO
        CALL CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.4) THEN
C       Loop diagram number 6 (might be others, just an example)
        DO I=1,4
          CALL LCUT_F(Q(0),I,WL(1,2))
          CALL FFV1LP0_3(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3))
          CALL FFV1L_1(W(1,WE(2),H),WL(1,3),LC(2),ML(4),ZERO,WL(1,4))
          CALL FFV1L_1(WL(1,4),W(1,WE(3),H),LC(3),ML(5),ZERO,WL(1,5))
          CALL FFV2L_1(WL(1,5),W(1,WE(4),H),LC(4),ML(6),ZERO,WL(1,6))
          BUFF(I)=WL(I+4,6)
        ENDDO
        CALL CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.5) THEN
C       Loop diagram number 7 (might be others, just an example)
        DO I=1,4
          CALL LCUT_F(Q(0),I,WL(1,2))
          CALL FFV1LP0_3(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3))
          CALL VVV1LP0_1(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1,4))
          CALL FFV1L_1(W(1,WE(3),H),WL(1,4),LC(3),ML(5),ZERO,WL(1,5))
          BUFF(I)=WL(I+4,5)
        ENDDO
        CALL CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.6) THEN
C       Loop diagram number 8 (might be others, just an example)
        DO I=1,4
          CALL LCUT_F(Q(0),I,WL(1,2))
          CALL FFV1LP0_3(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3))
          CALL VVV1LP0_1(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1,4))
          CALL FFV1L_1(W(1,WE(3),H),WL(1,4),LC(3),ML(5),ZERO,WL(1,5))
          CALL FFV2L_1(WL(1,5),W(1,WE(4),H),LC(4),ML(6),ZERO,WL(1,6))
          BUFF(I)=WL(I+4,6)
        ENDDO
        CALL CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.7) THEN
C       Loop diagram number 9 (might be others, just an example)
        DO I=1,4
          CALL LCUT_F(Q(0),I,WL(1,2))
          CALL FFV1LP0_3(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3))
          CALL FFV1L_1(W(1,WE(2),H),WL(1,3),LC(2),ML(4),ZERO,WL(1,4))
          CALL FFV2L_1(WL(1,4),W(1,WE(3),H),LC(3),ML(5),ZERO,WL(1,5))
          CALL FFV1L_1(WL(1,5),W(1,WE(4),H),LC(4),ML(6),ZERO,WL(1,6))
          BUFF(I)=WL(I+4,6)
        ENDDO
        CALL CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.8) THEN
C       Loop diagram number 10 (might be others, just an example)
        DO I=1,4
          CALL LCUT_V(Q(0),I,WL(1,2))
          CALL FFV1L_2(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3))
          CALL FFV1L_2(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1,4))
          CALL FFV1LP0_3(WL(1,4),W(1,WE(3),H),LC(3),ML(5),ZERO,WL(1,5))
          BUFF(I)=WL(I+4,5)
        ENDDO
        CALL CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.9) THEN
C       Loop diagram number 11 (might be others, just an example)
        DO I=1,4
          CALL LCUT_AF(Q(0),I,WL(1,2))
          CALL FFV1LP0_3(WL(1,2),W(1,WE(1),H),LC(1),ML(3),ZERO,WL(1,3))
          CALL FFV1L_2(W(1,WE(2),H),WL(1,3),LC(2),ML(4),ZERO,WL(1,4))
          BUFF(I)=WL(I+4,4)
        ENDDO
        CALL CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.10) THEN
C       Loop diagram number 12 (might be others, just an example)
        DO I=1,4
          CALL LCUT_AF(Q(0),I,WL(1,2))
          CALL FFV1LP0_3(WL(1,2),W(1,WE(1),H),LC(1),ML(3),ZERO,WL(1,3))
          CALL VVV1LP0_1(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1,4))
          CALL FFV1L_2(W(1,WE(3),H),WL(1,4),LC(3),ML(5),ZERO,WL(1,5))
          BUFF(I)=WL(I+4,5)
        ENDDO
        CALL CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.11) THEN
C       Loop diagram number 13 (might be others, just an example)
        DO I=1,4
          CALL LCUT_V(Q(0),I,WL(1,2))
          CALL FFV1L_1(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3))
          CALL FFV1L_1(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1,4))
          CALL FFV1LP0_3(W(1,WE(3),H),WL(1,4),LC(3),ML(5),ZERO,WL(1,5))
          BUFF(I)=WL(I+4,5)
        ENDDO
        CALL CLOSE_4(BUFF(1),RES)
      ENDIF
      END

      SUBROUTINE ML5_0_MPLOOPNUM(Q,RES)

      INCLUDE 'cts_mprec.h'
      IMPLICIT NONE
C     
C     CONSTANTS 
C     
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=2)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=39)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=10)
      INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=4)
      COMPLEX*32 IMAG1
      PARAMETER (IMAG1=(0E0_16,1E0_16))
C     
C     ARGUMENTS 
C     
      INCLUDE 'cts_mpc.h'                                             
     $ , INTENT(IN), DIMENSION(0:3) :: Q
      INCLUDE 'cts_mpc.h'                                             
     $ , INTENT(OUT) :: RES
C     
C     LOCAL VARIABLES 
C     
      COMPLEX*32 QPRES
      COMPLEX*32 QPQ(0:3)
      REAL*16 QPP(0:3,NEXTERNAL)
      INTEGER I,J,H
      COMPLEX*32 CFTOT
      COMPLEX*32 BUFF
C     
C     GLOBAL VARIABLES
C     
      LOGICAL MP_DONE
      COMMON/ML5_0_MP_DONE/MP_DONE

      REAL*16 MP_PS(0:3,NEXTERNAL),MP_P(0:3,NEXTERNAL)
      COMMON/ML5_0_MP_PSPOINT/MP_PS,MP_P

      REAL*8 LSCALE
      INTEGER CTMODE
      COMMON/ML5_0_CT/LSCALE,CTMODE

      INTEGER WE(NEXTERNAL)
      INTEGER ID, SYMFACT,MULTIPLIER,AMPLNUM
      COMMON/ML5_0_LOOP/WE,ID,SYMFACT,MULTIPLIER,AMPLNUM

      LOGICAL GOODHEL(NCOMB)
      LOGICAL GOODAMP(NLOOPAMPS,NCOMB)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL

      INTEGER NTRY
      LOGICAL CHECKPHASE,HELDOUBLECHECKED
      REAL*8 REF
      COMMON/ML5_0_INIT/NTRY,CHECKPHASE,HELDOUBLECHECKED,REF

      INTEGER CF_D(NLOOPAMPS,NBORNAMPS)
      INTEGER CF_N(NLOOPAMPS,NBORNAMPS)
      COMMON/ML5_0_CF/CF_D,CF_N

      COMPLEX*32 AMP(NBORNAMPS,NCOMB)
      COMMON/ML5_0_MP_AMPS/AMP
      COMPLEX*32 W(20,NWAVEFUNCS,NCOMB)
      COMMON/ML5_0_MP_WFS/W

      INTEGER HELPICKED
      COMMON/ML5_0_HELCHOICE/HELPICKED
C     ----------
C     BEGIN CODE
C     ----------
      DO I=0,3
        QPQ(I) = Q(I)
      ENDDO
      QPRES=(0.0E0_16,0.0E0_16)

      IF(.NOT.MP_DONE.AND.CTMODE.EQ.0) THEN
C       This is just to compute the wfs in quad prec
        CALL ML5_0_MP_BORN_AMPS_AND_WFS(MP_P)
        MP_DONE=.TRUE.
      ENDIF

      DO H=1,NCOMB
        IF (((HELPICKED.EQ.-1).OR.(HELPICKED.EQ.H)).AND.((CHECKPHASE.OR
     $..NOT.HELDOUBLECHECKED).OR.(GOODHEL(H).AND.GOODAMP(AMPLNUM,H))))
     $    THEN
          CALL ML5_0_MPLOOPNUMHEL(-QPQ,BUFF,H)
          DO I=1,NBORNAMPS
            CFTOT=CMPLX(CF_N(AMPLNUM,I)/(1.0E0_16*ABS(CF_D(AMPLNUM,I)))
     $       ,0.0E0_16,KIND=16)
            IF(CF_D(AMPLNUM,I).LT.0) CFTOT=CFTOT*IMAG1
            QPRES=QPRES+CFTOT*BUFF*CONJG(AMP(I,H))
          ENDDO
        ENDIF
      ENDDO
      QPRES=(QPRES*MULTIPLIER)/SYMFACT

      RES=QPRES
      END

      SUBROUTINE ML5_0_MPLOOPNUMHEL(Q,RES,H)
C     
C     CONSTANTS 
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    MAXLCOUPLINGS
      PARAMETER (MAXLCOUPLINGS=4)
      INTEGER    NMAXLOOPWFS
      PARAMETER (NMAXLOOPWFS=(NEXTERNAL+2))
      REAL*16     ZERO
      PARAMETER (ZERO=0E0_16)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=10)
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=2)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=39)
      INTEGER    NCOMB
      PARAMETER (NCOMB=32)
C     
C     ARGUMENTS 
C     
      COMPLEX*32 Q(0:3)
      COMPLEX*32 RES
      INTEGER H
C     
C     LOCAL VARIABLES 
C     
      COMPLEX*32 BUFF(4)
      COMPLEX*32 WL(20,NMAXLOOPWFS)
      INTEGER I
C     
C     GLOBAL VARIABLES
C     
      COMPLEX*32 LC(MAXLCOUPLINGS)
      COMPLEX*32 ML(NEXTERNAL+2)
      COMMON/ML5_0_MP_LOOP/LC,ML

      INTEGER WE(NEXTERNAL)
      INTEGER ID, SYMFACT,MULTIPLIER,AMPLNUM
      COMMON/ML5_0_LOOP/WE,ID,SYMFACT,MULTIPLIER,AMPLNUM

      COMPLEX*32 AMP(NBORNAMPS,NCOMB)
      COMMON/ML5_0_MP_AMPS/AMP
      COMPLEX*32 W(20,NWAVEFUNCS,NCOMB)
      COMMON/ML5_0_MP_WFS/W
C     ----------
C     BEGIN CODE
C     ----------
      RES=(0E0_16,0E0_16)
      IF (ID.EQ.1) THEN
C       Loop diagram number 3 (might be others, just an example)
        DO I=1,4
          CALL MP_LCUT_F(Q(0),I,WL(1,2))
          CALL MP_FFV1LP0_3(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1
     $     ,3))
          CALL MP_FFV1L_1(W(1,WE(2),H),WL(1,3),LC(2),ML(4),ZERO,WL(1,4)
     $     )
          BUFF(I)=WL(I+4,4)
        ENDDO
        CALL MP_CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.2) THEN
C       Loop diagram number 4 (might be others, just an example)
        DO I=1,4
          CALL MP_LCUT_V(Q(0),I,WL(1,2))
          CALL MP_FFV1L_1(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3)
     $     )
          CALL MP_FFV2L_1(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1,4)
     $     )
          CALL MP_FFV1LP0_3(W(1,WE(3),H),WL(1,4),LC(3),ML(5),ZERO,WL(1
     $     ,5))
          BUFF(I)=WL(I+4,5)
        ENDDO
        CALL MP_CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.3) THEN
C       Loop diagram number 5 (might be others, just an example)
        DO I=1,4
          CALL MP_LCUT_V(Q(0),I,WL(1,2))
          CALL MP_FFV1L_2(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3)
     $     )
          CALL MP_FFV2L_2(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1,4)
     $     )
          CALL MP_FFV1LP0_3(WL(1,4),W(1,WE(3),H),LC(3),ML(5),ZERO,WL(1
     $     ,5))
          BUFF(I)=WL(I+4,5)
        ENDDO
        CALL MP_CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.4) THEN
C       Loop diagram number 6 (might be others, just an example)
        DO I=1,4
          CALL MP_LCUT_F(Q(0),I,WL(1,2))
          CALL MP_FFV1LP0_3(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1
     $     ,3))
          CALL MP_FFV1L_1(W(1,WE(2),H),WL(1,3),LC(2),ML(4),ZERO,WL(1,4)
     $     )
          CALL MP_FFV1L_1(WL(1,4),W(1,WE(3),H),LC(3),ML(5),ZERO,WL(1,5)
     $     )
          CALL MP_FFV2L_1(WL(1,5),W(1,WE(4),H),LC(4),ML(6),ZERO,WL(1,6)
     $     )
          BUFF(I)=WL(I+4,6)
        ENDDO
        CALL MP_CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.5) THEN
C       Loop diagram number 7 (might be others, just an example)
        DO I=1,4
          CALL MP_LCUT_F(Q(0),I,WL(1,2))
          CALL MP_FFV1LP0_3(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1
     $     ,3))
          CALL MP_VVV1LP0_1(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1
     $     ,4))
          CALL MP_FFV1L_1(W(1,WE(3),H),WL(1,4),LC(3),ML(5),ZERO,WL(1,5)
     $     )
          BUFF(I)=WL(I+4,5)
        ENDDO
        CALL MP_CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.6) THEN
C       Loop diagram number 8 (might be others, just an example)
        DO I=1,4
          CALL MP_LCUT_F(Q(0),I,WL(1,2))
          CALL MP_FFV1LP0_3(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1
     $     ,3))
          CALL MP_VVV1LP0_1(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1
     $     ,4))
          CALL MP_FFV1L_1(W(1,WE(3),H),WL(1,4),LC(3),ML(5),ZERO,WL(1,5)
     $     )
          CALL MP_FFV2L_1(WL(1,5),W(1,WE(4),H),LC(4),ML(6),ZERO,WL(1,6)
     $     )
          BUFF(I)=WL(I+4,6)
        ENDDO
        CALL MP_CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.7) THEN
C       Loop diagram number 9 (might be others, just an example)
        DO I=1,4
          CALL MP_LCUT_F(Q(0),I,WL(1,2))
          CALL MP_FFV1LP0_3(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1
     $     ,3))
          CALL MP_FFV1L_1(W(1,WE(2),H),WL(1,3),LC(2),ML(4),ZERO,WL(1,4)
     $     )
          CALL MP_FFV2L_1(WL(1,4),W(1,WE(3),H),LC(3),ML(5),ZERO,WL(1,5)
     $     )
          CALL MP_FFV1L_1(WL(1,5),W(1,WE(4),H),LC(4),ML(6),ZERO,WL(1,6)
     $     )
          BUFF(I)=WL(I+4,6)
        ENDDO
        CALL MP_CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.8) THEN
C       Loop diagram number 10 (might be others, just an example)
        DO I=1,4
          CALL MP_LCUT_V(Q(0),I,WL(1,2))
          CALL MP_FFV1L_2(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3)
     $     )
          CALL MP_FFV1L_2(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1,4)
     $     )
          CALL MP_FFV1LP0_3(WL(1,4),W(1,WE(3),H),LC(3),ML(5),ZERO,WL(1
     $     ,5))
          BUFF(I)=WL(I+4,5)
        ENDDO
        CALL MP_CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.9) THEN
C       Loop diagram number 11 (might be others, just an example)
        DO I=1,4
          CALL MP_LCUT_AF(Q(0),I,WL(1,2))
          CALL MP_FFV1LP0_3(WL(1,2),W(1,WE(1),H),LC(1),ML(3),ZERO,WL(1
     $     ,3))
          CALL MP_FFV1L_2(W(1,WE(2),H),WL(1,3),LC(2),ML(4),ZERO,WL(1,4)
     $     )
          BUFF(I)=WL(I+4,4)
        ENDDO
        CALL MP_CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.10) THEN
C       Loop diagram number 12 (might be others, just an example)
        DO I=1,4
          CALL MP_LCUT_AF(Q(0),I,WL(1,2))
          CALL MP_FFV1LP0_3(WL(1,2),W(1,WE(1),H),LC(1),ML(3),ZERO,WL(1
     $     ,3))
          CALL MP_VVV1LP0_1(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1
     $     ,4))
          CALL MP_FFV1L_2(W(1,WE(3),H),WL(1,4),LC(3),ML(5),ZERO,WL(1,5)
     $     )
          BUFF(I)=WL(I+4,5)
        ENDDO
        CALL MP_CLOSE_4(BUFF(1),RES)
      ELSEIF (ID.EQ.11) THEN
C       Loop diagram number 13 (might be others, just an example)
        DO I=1,4
          CALL MP_LCUT_V(Q(0),I,WL(1,2))
          CALL MP_FFV1L_1(W(1,WE(1),H),WL(1,2),LC(1),ML(3),ZERO,WL(1,3)
     $     )
          CALL MP_FFV1L_1(WL(1,3),W(1,WE(2),H),LC(2),ML(4),ZERO,WL(1,4)
     $     )
          CALL MP_FFV1LP0_3(W(1,WE(3),H),WL(1,4),LC(3),ML(5),ZERO,WL(1
     $     ,5))
          BUFF(I)=WL(I+4,5)
        ENDDO
        CALL MP_CLOSE_4(BUFF(1),RES)
      ENDIF
      END

      SUBROUTINE ML5_0_MPLOOPNUM_DUMMY(Q,RES)
C     
C     ARGUMENTS 
C     
      INCLUDE 'cts_mprec.h'
      INCLUDE 'cts_mpc.h'                                             
     $ , INTENT(IN), DIMENSION(0:3) :: Q
      INCLUDE 'cts_mpc.h'                                             
     $ , INTENT(OUT) :: RES
C     
C     LOCAL VARIABLES 
C     
      COMPLEX*16 DRES
      COMPLEX*16 DQ(0:3)
      INTEGER I
C     ----------
C     BEGIN CODE
C     ----------
      DO I=0,3
        DQ(I) = Q(I)
      ENDDO

      CALL ML5_0_LOOPNUM(DQ,DRES)
      RES=DRES

      END

