      SUBROUTINE ML5_0_HELAS_CALLS_AMPB_1(P,NHEL,H,IC)
C     
C     Modules
C     
      USE ML5_0_POLYNOMIAL_CONSTANTS
      USE ALOHA_OBJECT
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
      PARAMETER (NLOOPS=44, NLOOPGROUPS=20, NCTAMPS=81)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=125)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=10,NLOOPWAVEFUNCS=92)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      REAL*16     MP__ZERO
      PARAMETER (MP__ZERO=0.0E0_16)
C     These are constants related to the split orders
      INTEGER    NSO, NSQUAREDSO, NAMPSO
      PARAMETER (NSO=1, NSQUAREDSO=1, NAMPSO=2)
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
      INTEGER FLAVOR(NEXTERNAL)
      DATA FLAVOR /NEXTERNAL*1/
      COMPLEX*16 COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)

      LOGICAL DUMMYFALSE
      DATA DUMMYFALSE/.FALSE./
C     
C     GLOBAL VARIABLES
C     

      INCLUDE 'coupl.inc'
      INTEGER WARDINT
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
      TYPE(ALOHA) W(NWAVEFUNCS)
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

      CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1,W(1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1, FLAVOR(2),W(2))
      CALL VXXXXX(P(0,3),ZERO,NHEL(3),+1,W(3))
      CALL OXXXXX(P(0,4),ZERO,NHEL(4),+1, FLAVOR(4),W(4))
      CALL FFV1_2(W(2),W(1),GC_5,ZERO,ZERO,W(5))
C     Amplitude(s) for born diagram with ID 1
      CALL FFV1_0(W(5),W(4),W(3),GC_5,AMP(1))
      CALL VVV1P0_1(W(1),W(3),GC_4,ZERO,ZERO,W(6))
C     Amplitude(s) for born diagram with ID 2
      CALL FFV1_0(W(2),W(4),W(6),GC_5,AMP(2))
      CALL FFV1_1(W(4),W(1),GC_5,ZERO,ZERO,W(7))
C     Amplitude(s) for born diagram with ID 3
      CALL FFV1_0(W(2),W(7),W(3),GC_5,AMP(3))
      CALL FFV1_1(W(4),W(3),GC_5,ZERO,ZERO,W(8))
C     Counter-term amplitude(s) for loop diagram number 4
      CALL R2_QQ_1_0(W(5),W(8),R2_QQQ,AMPL(1,1))
C     Counter-term amplitude(s) for loop diagram number 5
      CALL FFV1_0(W(5),W(4),W(3),R2_GQQ,AMPL(1,2))
      CALL FFV1_0(W(5),W(4),W(3),UV_GQQB_1EPS,AMPL(2,3))
      CALL FFV1_0(W(5),W(4),W(3),UV_GQQB_1EPS,AMPL(2,4))
      CALL FFV1_0(W(5),W(4),W(3),UV_GQQB_1EPS,AMPL(2,5))
      CALL FFV1_0(W(5),W(4),W(3),UV_GQQB_1EPS,AMPL(2,6))
      CALL FFV1_0(W(5),W(4),W(3),UV_GQQB_1EPS,AMPL(2,7))
      CALL FFV1_0(W(5),W(4),W(3),UV_GQQB_1EPS,AMPL(2,8))
      CALL FFV1_0(W(5),W(4),W(3),UV_GQQG_1EPS,AMPL(2,9))
      CALL FFV1_0(W(5),W(4),W(3),UV_GQQB,AMPL(1,10))
      CALL FFV1_0(W(5),W(4),W(3),UV_GQQT,AMPL(1,11))
      CALL FFV1P0_3(W(2),W(4),GC_5,ZERO,ZERO,W(9))
C     Counter-term amplitude(s) for loop diagram number 7
      CALL R2_GG_1_R2_GG_2_0(W(6),W(9),R2_GGG_1,R2_GGG_2,AMPL(1,12))
C     Counter-term amplitude(s) for loop diagram number 8
      CALL FFV1_0(W(2),W(4),W(6),R2_GQQ,AMPL(1,13))
      CALL FFV1_0(W(2),W(4),W(6),UV_GQQB_1EPS,AMPL(2,14))
      CALL FFV1_0(W(2),W(4),W(6),UV_GQQB_1EPS,AMPL(2,15))
      CALL FFV1_0(W(2),W(4),W(6),UV_GQQB_1EPS,AMPL(2,16))
      CALL FFV1_0(W(2),W(4),W(6),UV_GQQB_1EPS,AMPL(2,17))
      CALL FFV1_0(W(2),W(4),W(6),UV_GQQB_1EPS,AMPL(2,18))
      CALL FFV1_0(W(2),W(4),W(6),UV_GQQB_1EPS,AMPL(2,19))
      CALL FFV1_0(W(2),W(4),W(6),UV_GQQG_1EPS,AMPL(2,20))
      CALL FFV1_0(W(2),W(4),W(6),UV_GQQB,AMPL(1,21))
      CALL FFV1_0(W(2),W(4),W(6),UV_GQQT,AMPL(1,22))
      CALL FFV1_2(W(2),W(3),GC_5,ZERO,ZERO,W(10))
C     Counter-term amplitude(s) for loop diagram number 10
      CALL R2_QQ_1_0(W(10),W(7),R2_QQQ,AMPL(1,23))
C     Counter-term amplitude(s) for loop diagram number 11
      CALL FFV1_0(W(2),W(7),W(3),R2_GQQ,AMPL(1,24))
      CALL FFV1_0(W(2),W(7),W(3),UV_GQQB_1EPS,AMPL(2,25))
      CALL FFV1_0(W(2),W(7),W(3),UV_GQQB_1EPS,AMPL(2,26))
      CALL FFV1_0(W(2),W(7),W(3),UV_GQQB_1EPS,AMPL(2,27))
      CALL FFV1_0(W(2),W(7),W(3),UV_GQQB_1EPS,AMPL(2,28))
      CALL FFV1_0(W(2),W(7),W(3),UV_GQQB_1EPS,AMPL(2,29))
      CALL FFV1_0(W(2),W(7),W(3),UV_GQQB_1EPS,AMPL(2,30))
      CALL FFV1_0(W(2),W(7),W(3),UV_GQQG_1EPS,AMPL(2,31))
      CALL FFV1_0(W(2),W(7),W(3),UV_GQQB,AMPL(1,32))
      CALL FFV1_0(W(2),W(7),W(3),UV_GQQT,AMPL(1,33))
C     Counter-term amplitude(s) for loop diagram number 13
      CALL FFV1_0(W(10),W(4),W(1),R2_GQQ,AMPL(1,34))
      CALL FFV1_0(W(10),W(4),W(1),UV_GQQB_1EPS,AMPL(2,35))
      CALL FFV1_0(W(10),W(4),W(1),UV_GQQB_1EPS,AMPL(2,36))
      CALL FFV1_0(W(10),W(4),W(1),UV_GQQB_1EPS,AMPL(2,37))
      CALL FFV1_0(W(10),W(4),W(1),UV_GQQB_1EPS,AMPL(2,38))
      CALL FFV1_0(W(10),W(4),W(1),UV_GQQB_1EPS,AMPL(2,39))
      CALL FFV1_0(W(10),W(4),W(1),UV_GQQB_1EPS,AMPL(2,40))
      CALL FFV1_0(W(10),W(4),W(1),UV_GQQG_1EPS,AMPL(2,41))
      CALL FFV1_0(W(10),W(4),W(1),UV_GQQB,AMPL(1,42))
      CALL FFV1_0(W(10),W(4),W(1),UV_GQQT,AMPL(1,43))
C     Counter-term amplitude(s) for loop diagram number 14
      CALL VVV1_0(W(1),W(3),W(9),R2_3GG,AMPL(1,44))
      CALL VVV1_0(W(1),W(3),W(9),UV_3GB_1EPS,AMPL(2,45))
      CALL VVV1_0(W(1),W(3),W(9),UV_3GB_1EPS,AMPL(2,46))
      CALL VVV1_0(W(1),W(3),W(9),UV_3GB_1EPS,AMPL(2,47))
      CALL VVV1_0(W(1),W(3),W(9),UV_3GB_1EPS,AMPL(2,48))
      CALL VVV1_0(W(1),W(3),W(9),UV_3GB_1EPS,AMPL(2,49))
      CALL VVV1_0(W(1),W(3),W(9),UV_3GB_1EPS,AMPL(2,50))
      CALL VVV1_0(W(1),W(3),W(9),UV_3GG_1EPS,AMPL(2,51))
      CALL VVV1_0(W(1),W(3),W(9),UV_3GB,AMPL(1,52))
      CALL VVV1_0(W(1),W(3),W(9),UV_3GT,AMPL(1,53))
C     Counter-term amplitude(s) for loop diagram number 18
      CALL FFV1_0(W(2),W(8),W(1),R2_GQQ,AMPL(1,54))
      CALL FFV1_0(W(2),W(8),W(1),UV_GQQB_1EPS,AMPL(2,55))
      CALL FFV1_0(W(2),W(8),W(1),UV_GQQB_1EPS,AMPL(2,56))
      CALL FFV1_0(W(2),W(8),W(1),UV_GQQB_1EPS,AMPL(2,57))
      CALL FFV1_0(W(2),W(8),W(1),UV_GQQB_1EPS,AMPL(2,58))
      CALL FFV1_0(W(2),W(8),W(1),UV_GQQB_1EPS,AMPL(2,59))
      CALL FFV1_0(W(2),W(8),W(1),UV_GQQB_1EPS,AMPL(2,60))
      CALL FFV1_0(W(2),W(8),W(1),UV_GQQG_1EPS,AMPL(2,61))
      CALL FFV1_0(W(2),W(8),W(1),UV_GQQB,AMPL(1,62))
      CALL FFV1_0(W(2),W(8),W(1),UV_GQQT,AMPL(1,63))
C     Counter-term amplitude(s) for loop diagram number 31
      CALL R2_GG_1_0(W(6),W(9),R2_GGQ,AMPL(1,64))
      CALL R2_GG_1_0(W(6),W(9),R2_GGQ,AMPL(1,65))
      CALL R2_GG_1_0(W(6),W(9),R2_GGQ,AMPL(1,66))
      CALL R2_GG_1_0(W(6),W(9),R2_GGQ,AMPL(1,67))
C     Counter-term amplitude(s) for loop diagram number 32
      CALL VVV1_0(W(1),W(3),W(9),R2_3GQ,AMPL(1,68))
      CALL VVV1_0(W(1),W(3),W(9),R2_3GQ,AMPL(1,69))
      CALL VVV1_0(W(1),W(3),W(9),R2_3GQ,AMPL(1,70))
      CALL VVV1_0(W(1),W(3),W(9),R2_3GQ,AMPL(1,71))
C     Counter-term amplitude(s) for loop diagram number 34
      CALL R2_GG_1_R2_GG_3_0(W(6),W(9),R2_GGQ,R2_GGB,AMPL(1,72))
C     Counter-term amplitude(s) for loop diagram number 35
      CALL VVV1_0(W(1),W(3),W(9),R2_3GQ,AMPL(1,73))
C     Counter-term amplitude(s) for loop diagram number 37
      CALL R2_GG_1_R2_GG_3_0(W(6),W(9),R2_GGQ,R2_GGT,AMPL(1,74))
C     Counter-term amplitude(s) for loop diagram number 38
      CALL VVV1_0(W(1),W(3),W(9),R2_3GQ,AMPL(1,75))
C     At this point, all CT amps needed for (QCD=6), i.e. of split
C      order ID=1, are computed.
      IF(FILTER_SO.AND.SQSO_TARGET.EQ.1) GOTO 2000

      GOTO 1001
 2000 CONTINUE
      CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

