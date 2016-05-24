      SUBROUTINE MG5_1_HELAS_CALLS_AMPB_1(P,NHEL,H,IC)
C     
C     Modules
C     
      USE MG5_1_POLYNOMIAL_CONSTANTS
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
      PARAMETER (NBORNAMPS=2)
      INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=22, NLOOPGROUPS=13, NCTAMPS=54)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=76)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=8,NLOOPWAVEFUNCS=50)
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
      COMMON/MG5_1_FILTERS/GOODAMP,GOODHEL,HELOFFSET

      LOGICAL CHECKPHASE
      LOGICAL HELDOUBLECHECKED
      COMMON/MG5_1_INIT/CHECKPHASE, HELDOUBLECHECKED

      INTEGER SQSO_TARGET
      COMMON/MG5_1_SOCHOICE/SQSO_TARGET

      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE
     $ ,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE
     $ ,CTCALL_REQ_SO_DONE,FILTER_SO
      COMMON/MG5_1_SO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE
     $ ,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE
     $ ,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

      INTEGER I_SO
      COMMON/MG5_1_I_SO/I_SO
      INTEGER I_LIB
      COMMON/MG5_1_I_LIB/I_LIB

      COMPLEX*16 AMP(NBORNAMPS)
      COMMON/MG5_1_AMPS/AMP
      COMPLEX*16 W(20,NWAVEFUNCS)
      COMMON/MG5_1_W/W

      COMPLEX*16 WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE
     $ ,0:NLOOPWAVEFUNCS)
      COMPLEX*16 PL(0:3,0:NLOOPWAVEFUNCS)
      COMMON/MG5_1_WL/WL,PL

      COMPLEX*16 AMPL(3,NCTAMPS)
      COMMON/MG5_1_AMPL/AMPL

C     
C     ----------
C     BEGIN CODE
C     ----------

C     The target squared split order contribution is already reached
C      if true.
      IF (FILTER_SO.AND.CT_REQ_SO_DONE) THEN
        GOTO 1001
      ENDIF

      CALL IXXXXX(P(0,1),ZERO,NHEL(1),+1*IC(1),W(1,1))
      CALL OXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
      CALL OXXXXX(P(0,3),ZERO,NHEL(3),+1*IC(3),W(1,3))
      CALL IXXXXX(P(0,4),ZERO,NHEL(4),-1*IC(4),W(1,4))
      CALL FFV1P0_3(W(1,1),W(1,2),GC_5,ZERO,ZERO,W(1,5))
C     Amplitude(s) for born diagram with ID 1
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),GC_5,AMP(1))
      CALL FFV1P0_3(W(1,1),W(1,3),GC_5,ZERO,ZERO,W(1,6))
C     Amplitude(s) for born diagram with ID 2
      CALL FFV1_0(W(1,4),W(1,2),W(1,6),GC_5,AMP(2))
      CALL FFV1P0_3(W(1,4),W(1,3),GC_5,ZERO,ZERO,W(1,7))
C     Counter-term amplitude(s) for loop diagram number 3
      CALL R2_GG_1_0(W(1,5),W(1,7),R2_GGQ,AMPL(1,1))
      CALL R2_GG_1_0(W(1,5),W(1,7),R2_GGQ,AMPL(1,2))
      CALL R2_GG_1_0(W(1,5),W(1,7),R2_GGQ,AMPL(1,3))
      CALL R2_GG_1_0(W(1,5),W(1,7),R2_GGQ,AMPL(1,4))
      CALL FFV1P0_3(W(1,4),W(1,2),GC_5,ZERO,ZERO,W(1,8))
C     Counter-term amplitude(s) for loop diagram number 4
      CALL R2_GG_1_0(W(1,6),W(1,8),R2_GGQ,AMPL(1,5))
      CALL R2_GG_1_0(W(1,6),W(1,8),R2_GGQ,AMPL(1,6))
      CALL R2_GG_1_0(W(1,6),W(1,8),R2_GGQ,AMPL(1,7))
      CALL R2_GG_1_0(W(1,6),W(1,8),R2_GGQ,AMPL(1,8))
C     Counter-term amplitude(s) for loop diagram number 5
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQQ_1EPS,AMPL(2,9))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQQ_1EPS,AMPL(2,10))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQQ_1EPS,AMPL(2,11))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQQ_1EPS,AMPL(2,12))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQB,AMPL(1,13))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQQ_1EPS,AMPL(2,14))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQT,AMPL(1,15))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQQ_1EPS,AMPL(2,16))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),UV_GQQG_1EPS,AMPL(2,17))
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),R2_GQQ,AMPL(1,18))
C     Counter-term amplitude(s) for loop diagram number 7
      CALL FFV1_0(W(1,4),W(1,2),W(1,6),UV_GQQQ_1EPS,AMPL(2,19))
      CALL FFV1_0(W(1,4),W(1,2),W(1,6),UV_GQQQ_1EPS,AMPL(2,20))
      CALL FFV1_0(W(1,4),W(1,2),W(1,6),UV_GQQQ_1EPS,AMPL(2,21))
      CALL FFV1_0(W(1,4),W(1,2),W(1,6),UV_GQQQ_1EPS,AMPL(2,22))
      CALL FFV1_0(W(1,4),W(1,2),W(1,6),UV_GQQB,AMPL(1,23))
      CALL FFV1_0(W(1,4),W(1,2),W(1,6),UV_GQQQ_1EPS,AMPL(2,24))
      CALL FFV1_0(W(1,4),W(1,2),W(1,6),UV_GQQT,AMPL(1,25))
      CALL FFV1_0(W(1,4),W(1,2),W(1,6),UV_GQQQ_1EPS,AMPL(2,26))
      CALL FFV1_0(W(1,4),W(1,2),W(1,6),UV_GQQG_1EPS,AMPL(2,27))
      CALL FFV1_0(W(1,4),W(1,2),W(1,6),R2_GQQ,AMPL(1,28))
C     Counter-term amplitude(s) for loop diagram number 9
      CALL FFV1_0(W(1,1),W(1,3),W(1,8),UV_GQQQ_1EPS,AMPL(2,29))
      CALL FFV1_0(W(1,1),W(1,3),W(1,8),UV_GQQQ_1EPS,AMPL(2,30))
      CALL FFV1_0(W(1,1),W(1,3),W(1,8),UV_GQQQ_1EPS,AMPL(2,31))
      CALL FFV1_0(W(1,1),W(1,3),W(1,8),UV_GQQQ_1EPS,AMPL(2,32))
      CALL FFV1_0(W(1,1),W(1,3),W(1,8),UV_GQQB,AMPL(1,33))
      CALL FFV1_0(W(1,1),W(1,3),W(1,8),UV_GQQQ_1EPS,AMPL(2,34))
      CALL FFV1_0(W(1,1),W(1,3),W(1,8),UV_GQQT,AMPL(1,35))
      CALL FFV1_0(W(1,1),W(1,3),W(1,8),UV_GQQQ_1EPS,AMPL(2,36))
      CALL FFV1_0(W(1,1),W(1,3),W(1,8),UV_GQQG_1EPS,AMPL(2,37))
      CALL FFV1_0(W(1,1),W(1,3),W(1,8),R2_GQQ,AMPL(1,38))
C     Counter-term amplitude(s) for loop diagram number 13
      CALL FFV1_0(W(1,1),W(1,2),W(1,7),UV_GQQQ_1EPS,AMPL(2,39))
      CALL FFV1_0(W(1,1),W(1,2),W(1,7),UV_GQQQ_1EPS,AMPL(2,40))
      CALL FFV1_0(W(1,1),W(1,2),W(1,7),UV_GQQQ_1EPS,AMPL(2,41))
      CALL FFV1_0(W(1,1),W(1,2),W(1,7),UV_GQQQ_1EPS,AMPL(2,42))
      CALL FFV1_0(W(1,1),W(1,2),W(1,7),UV_GQQB,AMPL(1,43))
      CALL FFV1_0(W(1,1),W(1,2),W(1,7),UV_GQQQ_1EPS,AMPL(2,44))
      CALL FFV1_0(W(1,1),W(1,2),W(1,7),UV_GQQT,AMPL(1,45))
      CALL FFV1_0(W(1,1),W(1,2),W(1,7),UV_GQQQ_1EPS,AMPL(2,46))
      CALL FFV1_0(W(1,1),W(1,2),W(1,7),UV_GQQG_1EPS,AMPL(2,47))
      CALL FFV1_0(W(1,1),W(1,2),W(1,7),R2_GQQ,AMPL(1,48))
C     Counter-term amplitude(s) for loop diagram number 17
      CALL R2_GG_1_R2_GG_3_0(W(1,5),W(1,7),R2_GGQ,R2_GGB,AMPL(1,49))
C     Counter-term amplitude(s) for loop diagram number 18
      CALL R2_GG_1_R2_GG_3_0(W(1,6),W(1,8),R2_GGQ,R2_GGB,AMPL(1,50))
C     Counter-term amplitude(s) for loop diagram number 19
      CALL R2_GG_1_R2_GG_3_0(W(1,5),W(1,7),R2_GGQ,R2_GGT,AMPL(1,51))
C     Counter-term amplitude(s) for loop diagram number 20
      CALL R2_GG_1_R2_GG_3_0(W(1,6),W(1,8),R2_GGQ,R2_GGT,AMPL(1,52))
C     Counter-term amplitude(s) for loop diagram number 21
      CALL R2_GG_1_R2_GG_2_0(W(1,5),W(1,7),R2_GGG_1,R2_GGG_2,AMPL(1,53)
     $ )
C     Counter-term amplitude(s) for loop diagram number 22
      CALL R2_GG_1_R2_GG_2_0(W(1,6),W(1,8),R2_GGG_1,R2_GGG_2,AMPL(1,54)
     $ )
C     At this point, all CT amps needed for (QCD=6), i.e. of split
C      order ID=1, are computed.
      IF(FILTER_SO.AND.SQSO_TARGET.EQ.1) GOTO 2000

      GOTO 1001
 2000 CONTINUE
      CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

