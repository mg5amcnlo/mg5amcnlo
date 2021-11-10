      SUBROUTINE ML5_0_HELAS_CALLS_AMPB_1(P,NHEL,H,IC)
C     
C     Modules
C     
      USE ML5_0_POLYNOMIAL_CONSTANTS
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      INTEGER    NCOMB
      PARAMETER (NCOMB=16)
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=3)
      INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=44, NLOOPGROUPS=28, NCTAMPS=85)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=129)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=10,NLOOPWAVEFUNCS=93)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      REAL*16     MP__ZERO
      PARAMETER (MP__ZERO=0.0E0_16)
C     These are constants related to the split orders
      INTEGER    NSO, NSQUAREDSO, NAMPSO
      PARAMETER (NSO=0, NSQUAREDSO=0, NAMPSO=0)
C     
C     ARGUMENTS
C     
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER H
C     
C     LOCAL VARIABLES
C     
      INTEGER I,J,K
      COMPLEX*16 COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)

      LOGICAL DUMMYFALSE
      DATA DUMMYFALSE/.FALSE./
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'coupl.inc'
      INCLUDE 'mp_coupl.inc'

      INTEGER HELOFFSET
      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      LOGICAL CHECKPHASE
      LOGICAL HELDOUBLECHECKED
      COMMON/ML5_0_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER SQSO_TARGET
      COMMON/ML5_0_SOCHOICE/SQSO_TARGET

      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE
     $ ,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE
     $ ,CTCALL_REQ_SO_DONE,FILTER_SO
      COMMON/ML5_0_SO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE
     $ ,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE
     $ ,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

      INTEGER I_SO
      COMMON/ML5_0_I_SO/I_SO
      INTEGER I_LIB
      COMMON/ML5_0_I_LIB/I_LIB

      COMPLEX*16 AMP(NBORNAMPS)
      COMMON/ML5_0_AMPS/AMP
      COMPLEX*16 W(20,NWAVEFUNCS)
      COMMON/ML5_0_W/W

      COMPLEX*16 WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,
     $ -1:NLOOPWAVEFUNCS)
      COMPLEX*16 PL(0:3,-1:NLOOPWAVEFUNCS)
      COMMON/ML5_0_WL/WL,PL

      COMPLEX*16 AMPL(3,NCTAMPS)
      COMMON/ML5_0_AMPL/AMPL

C     
C     ----------
C     BEGIN CODE
C     ----------

C     The target squared split order contribution is already reached
C      if true.
      IF (FILTER_SO.AND.CT_REQ_SO_DONE) THEN
        GOTO 1001
      ENDIF

      CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
      CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
      CALL OXXXXX(P(0,3),MDL_MT,NHEL(3),+1*IC(3),W(1,3))
      CALL IXXXXX(P(0,4),MDL_MT,NHEL(4),-1*IC(4),W(1,4))
      CALL VVV1P0_1(W(1,1),W(1,2),GC_4,ZERO,ZERO,W(1,5))
C     Amplitude(s) for born diagram with ID 1
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),GC_5,AMP(1))
      CALL FFV1_1(W(1,3),W(1,1),GC_5,MDL_MT,MDL_WT,W(1,6))
C     Amplitude(s) for born diagram with ID 2
      CALL FFV1_0(W(1,4),W(1,6),W(1,2),GC_5,AMP(2))
      CALL FFV1_2(W(1,4),W(1,1),GC_5,MDL_MT,MDL_WT,W(1,7))
C     Amplitude(s) for born diagram with ID 3
      CALL FFV1_0(W(1,7),W(1,3),W(1,2),GC_5,AMP(3))
      CALL FFV1P0_3(W(1,4),W(1,3),GC_5,ZERO,ZERO,W(1,8))
C     Counter-term amplitude(s) for loop diagram number 4
      CALL R2_GG_1_R2_GG_2_0(W(1,5),W(1,8),R2_GGG_1,R2_GGG_2,AMPL(1,1))
C     Counter-term amplitude(s) for loop diagram number 5
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),R2_GQQ,AMPL(1,2))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQB_1EPS,AMPL(2,3))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQB_1EPS,AMPL(2,4))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQB_1EPS,AMPL(2,5))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQB_1EPS,AMPL(2,6))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQB_1EPS,AMPL(2,7))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQB_1EPS,AMPL(2,8))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQG_1EPS,AMPL(2,9))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQB,AMPL(1,10))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQT,AMPL(1,11))
      CALL FFV1_2(W(1,4),W(1,2),GC_5,MDL_MT,MDL_WT,W(1,9))
C     Counter-term amplitude(s) for loop diagram number 7
      CALL R2_QQ_1_R2_QQ_2_0(W(1,9),W(1,6),R2_QQQ,R2_QQT,AMPL(1,12))
      CALL R2_QQ_2_0(W(1,9),W(1,6),UV_TMASS_1EPS,AMPL(2,13))
      CALL R2_QQ_2_0(W(1,9),W(1,6),UV_TMASS,AMPL(1,14))
C     Counter-term amplitude(s) for loop diagram number 8
      CALL FFV1_0(W(1,4),W(1,6),W(1,2),R2_GQQ,AMPL(1,15))
      CALL FFV1_0(W(1,4),W(1,6),W(1,2),UV_GQQB_1EPS,AMPL(2,16))
      CALL FFV1_0(W(1,4),W(1,6),W(1,2),UV_GQQB_1EPS,AMPL(2,17))
      CALL FFV1_0(W(1,4),W(1,6),W(1,2),UV_GQQB_1EPS,AMPL(2,18))
      CALL FFV1_0(W(1,4),W(1,6),W(1,2),UV_GQQB_1EPS,AMPL(2,19))
      CALL FFV1_0(W(1,4),W(1,6),W(1,2),UV_GQQB_1EPS,AMPL(2,20))
      CALL FFV1_0(W(1,4),W(1,6),W(1,2),UV_GQQB_1EPS,AMPL(2,21))
      CALL FFV1_0(W(1,4),W(1,6),W(1,2),UV_GQQG_1EPS,AMPL(2,22))
      CALL FFV1_0(W(1,4),W(1,6),W(1,2),UV_GQQB,AMPL(1,23))
      CALL FFV1_0(W(1,4),W(1,6),W(1,2),UV_GQQT,AMPL(1,24))
      CALL FFV1_1(W(1,3),W(1,2),GC_5,MDL_MT,MDL_WT,W(1,10))
C     Counter-term amplitude(s) for loop diagram number 10
      CALL R2_QQ_1_R2_QQ_2_0(W(1,7),W(1,10),R2_QQQ,R2_QQT,AMPL(1,25))
      CALL R2_QQ_2_0(W(1,7),W(1,10),UV_TMASS_1EPS,AMPL(2,26))
      CALL R2_QQ_2_0(W(1,7),W(1,10),UV_TMASS,AMPL(1,27))
C     Counter-term amplitude(s) for loop diagram number 11
      CALL FFV1_0(W(1,7),W(1,3),W(1,2),R2_GQQ,AMPL(1,28))
      CALL FFV1_0(W(1,7),W(1,3),W(1,2),UV_GQQB_1EPS,AMPL(2,29))
      CALL FFV1_0(W(1,7),W(1,3),W(1,2),UV_GQQB_1EPS,AMPL(2,30))
      CALL FFV1_0(W(1,7),W(1,3),W(1,2),UV_GQQB_1EPS,AMPL(2,31))
      CALL FFV1_0(W(1,7),W(1,3),W(1,2),UV_GQQB_1EPS,AMPL(2,32))
      CALL FFV1_0(W(1,7),W(1,3),W(1,2),UV_GQQB_1EPS,AMPL(2,33))
      CALL FFV1_0(W(1,7),W(1,3),W(1,2),UV_GQQB_1EPS,AMPL(2,34))
      CALL FFV1_0(W(1,7),W(1,3),W(1,2),UV_GQQG_1EPS,AMPL(2,35))
      CALL FFV1_0(W(1,7),W(1,3),W(1,2),UV_GQQB,AMPL(1,36))
      CALL FFV1_0(W(1,7),W(1,3),W(1,2),UV_GQQT,AMPL(1,37))
C     Counter-term amplitude(s) for loop diagram number 13
      CALL FFV1_0(W(1,4),W(1,10),W(1,1),R2_GQQ,AMPL(1,38))
      CALL FFV1_0(W(1,4),W(1,10),W(1,1),UV_GQQB_1EPS,AMPL(2,39))
      CALL FFV1_0(W(1,4),W(1,10),W(1,1),UV_GQQB_1EPS,AMPL(2,40))
      CALL FFV1_0(W(1,4),W(1,10),W(1,1),UV_GQQB_1EPS,AMPL(2,41))
      CALL FFV1_0(W(1,4),W(1,10),W(1,1),UV_GQQB_1EPS,AMPL(2,42))
      CALL FFV1_0(W(1,4),W(1,10),W(1,1),UV_GQQB_1EPS,AMPL(2,43))
      CALL FFV1_0(W(1,4),W(1,10),W(1,1),UV_GQQB_1EPS,AMPL(2,44))
      CALL FFV1_0(W(1,4),W(1,10),W(1,1),UV_GQQG_1EPS,AMPL(2,45))
      CALL FFV1_0(W(1,4),W(1,10),W(1,1),UV_GQQB,AMPL(1,46))
      CALL FFV1_0(W(1,4),W(1,10),W(1,1),UV_GQQT,AMPL(1,47))
C     Counter-term amplitude(s) for loop diagram number 14
      CALL FFV1_0(W(1,9),W(1,3),W(1,1),R2_GQQ,AMPL(1,48))
      CALL FFV1_0(W(1,9),W(1,3),W(1,1),UV_GQQB_1EPS,AMPL(2,49))
      CALL FFV1_0(W(1,9),W(1,3),W(1,1),UV_GQQB_1EPS,AMPL(2,50))
      CALL FFV1_0(W(1,9),W(1,3),W(1,1),UV_GQQB_1EPS,AMPL(2,51))
      CALL FFV1_0(W(1,9),W(1,3),W(1,1),UV_GQQB_1EPS,AMPL(2,52))
      CALL FFV1_0(W(1,9),W(1,3),W(1,1),UV_GQQB_1EPS,AMPL(2,53))
      CALL FFV1_0(W(1,9),W(1,3),W(1,1),UV_GQQB_1EPS,AMPL(2,54))
      CALL FFV1_0(W(1,9),W(1,3),W(1,1),UV_GQQG_1EPS,AMPL(2,55))
      CALL FFV1_0(W(1,9),W(1,3),W(1,1),UV_GQQB,AMPL(1,56))
      CALL FFV1_0(W(1,9),W(1,3),W(1,1),UV_GQQT,AMPL(1,57))
C     Counter-term amplitude(s) for loop diagram number 17
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),R2_3GG,AMPL(1,58))
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),UV_3GB_1EPS,AMPL(2,59))
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),UV_3GB_1EPS,AMPL(2,60))
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),UV_3GB_1EPS,AMPL(2,61))
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),UV_3GB_1EPS,AMPL(2,62))
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),UV_3GB_1EPS,AMPL(2,63))
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),UV_3GB_1EPS,AMPL(2,64))
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),UV_3GG_1EPS,AMPL(2,65))
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),UV_3GB,AMPL(1,66))
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),UV_3GT,AMPL(1,67))
C     Counter-term amplitude(s) for loop diagram number 31
      CALL R2_GG_1_0(W(1,5),W(1,8),R2_GGQ,AMPL(1,68))
      CALL R2_GG_1_0(W(1,5),W(1,8),R2_GGQ,AMPL(1,69))
      CALL R2_GG_1_0(W(1,5),W(1,8),R2_GGQ,AMPL(1,70))
      CALL R2_GG_1_0(W(1,5),W(1,8),R2_GGQ,AMPL(1,71))
C     Counter-term amplitude(s) for loop diagram number 32
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),R2_3GQ,AMPL(1,72))
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),R2_3GQ,AMPL(1,73))
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),R2_3GQ,AMPL(1,74))
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),R2_3GQ,AMPL(1,75))
C     Counter-term amplitude(s) for loop diagram number 34
      CALL R2_GG_1_R2_GG_3_0(W(1,5),W(1,8),R2_GGQ,R2_GGB,AMPL(1,76))
C     Counter-term amplitude(s) for loop diagram number 35
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),R2_3GQ,AMPL(1,77))
C     Counter-term amplitude(s) for loop diagram number 37
      CALL R2_GG_1_R2_GG_3_0(W(1,5),W(1,8),R2_GGQ,R2_GGT,AMPL(1,78))
C     Counter-term amplitude(s) for loop diagram number 38
      CALL VVV1_0(W(1,1),W(1,2),W(1,8),R2_3GQ,AMPL(1,79))

      GOTO 1001
 2000 CONTINUE
      CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

