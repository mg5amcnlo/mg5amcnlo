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
      PARAMETER (NEXTERNAL=5)
      INTEGER    NCOMB
      PARAMETER (NCOMB=48)
      INTEGER NBORNAMPS
      PARAMETER (NBORNAMPS=8)
      INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=144, NLOOPGROUPS=77, NCTAMPS=252)
      INTEGER    NLOOPAMPS
      PARAMETER (NLOOPAMPS=396)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=28,NLOOPWAVEFUNCS=267)
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
      CALL VXXXXX(P(0,3),MDL_MW,NHEL(3),+1*IC(3),W(1,3))
      CALL OXXXXX(P(0,4),MDL_MT,NHEL(4),+1*IC(4),W(1,4))
      CALL IXXXXX(P(0,5),MDL_MB,NHEL(5),-1*IC(5),W(1,5))
      CALL VVV1P0_1(W(1,1),W(1,2),GC_4,ZERO,ZERO,W(1,6))
      CALL FFV2_1(W(1,4),W(1,3),GC_11,MDL_MB,ZERO,W(1,7))
C     Amplitude(s) for born diagram with ID 1
      CALL FFV1_0(W(1,5),W(1,7),W(1,6),GC_5,AMP(1))
      CALL FFV2_2(W(1,5),W(1,3),GC_11,MDL_MT,MDL_WT,W(1,8))
C     Amplitude(s) for born diagram with ID 2
      CALL FFV1_0(W(1,8),W(1,4),W(1,6),GC_5,AMP(2))
      CALL FFV1_1(W(1,4),W(1,1),GC_5,MDL_MT,MDL_WT,W(1,9))
      CALL FFV1_2(W(1,5),W(1,2),GC_5,MDL_MB,ZERO,W(1,10))
C     Amplitude(s) for born diagram with ID 3
      CALL FFV2_0(W(1,10),W(1,9),W(1,3),GC_11,AMP(3))
C     Amplitude(s) for born diagram with ID 4
      CALL FFV1_0(W(1,8),W(1,9),W(1,2),GC_5,AMP(4))
      CALL FFV1_2(W(1,5),W(1,1),GC_5,MDL_MB,ZERO,W(1,11))
      CALL FFV1_1(W(1,4),W(1,2),GC_5,MDL_MT,MDL_WT,W(1,12))
C     Amplitude(s) for born diagram with ID 5
      CALL FFV2_0(W(1,11),W(1,12),W(1,3),GC_11,AMP(5))
C     Amplitude(s) for born diagram with ID 6
      CALL FFV1_0(W(1,11),W(1,7),W(1,2),GC_5,AMP(6))
C     Amplitude(s) for born diagram with ID 7
      CALL FFV1_0(W(1,8),W(1,12),W(1,1),GC_5,AMP(7))
C     Amplitude(s) for born diagram with ID 8
      CALL FFV1_0(W(1,10),W(1,7),W(1,1),GC_5,AMP(8))
      CALL FFV1_2(W(1,5),W(1,6),GC_5,MDL_MB,ZERO,W(1,13))
C     Counter-term amplitude(s) for loop diagram number 9
      CALL R2_QQ_1_R2_QQ_2_0(W(1,13),W(1,7),R2_QQQ,R2_QQB,AMPL(1,1))
      CALL R2_QQ_2_0(W(1,13),W(1,7),UV_BMASS_1EPS,AMPL(2,2))
      CALL R2_QQ_2_0(W(1,13),W(1,7),UV_BMASS,AMPL(1,3))
      CALL FFV1P0_3(W(1,5),W(1,7),GC_5,ZERO,ZERO,W(1,14))
C     Counter-term amplitude(s) for loop diagram number 10
      CALL R2_GG_1_R2_GG_2_0(W(1,6),W(1,14),R2_GGG_1,R2_GGG_2,AMPL(1,4)
     $ )
C     Counter-term amplitude(s) for loop diagram number 11
      CALL FFV1_0(W(1,5),W(1,7),W(1,6),R2_GQQ,AMPL(1,5))
      CALL FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQB_1EPS,AMPL(2,6))
      CALL FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQB_1EPS,AMPL(2,7))
      CALL FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQB_1EPS,AMPL(2,8))
      CALL FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQB_1EPS,AMPL(2,9))
      CALL FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQB_1EPS,AMPL(2,10))
      CALL FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQB_1EPS,AMPL(2,11))
      CALL FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQG_1EPS,AMPL(2,12))
      CALL FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQB,AMPL(1,13))
      CALL FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQT,AMPL(1,14))
      CALL FFV1_1(W(1,4),W(1,6),GC_5,MDL_MT,MDL_WT,W(1,15))
C     Counter-term amplitude(s) for loop diagram number 13
      CALL R2_QQ_1_R2_QQ_2_0(W(1,8),W(1,15),R2_QQQ,R2_QQT,AMPL(1,15))
      CALL R2_QQ_2_0(W(1,8),W(1,15),UV_TMASS_1EPS,AMPL(2,16))
      CALL R2_QQ_2_0(W(1,8),W(1,15),UV_TMASS,AMPL(1,17))
      CALL FFV1P0_3(W(1,8),W(1,4),GC_5,ZERO,ZERO,W(1,16))
C     Counter-term amplitude(s) for loop diagram number 14
      CALL R2_GG_1_R2_GG_2_0(W(1,6),W(1,16),R2_GGG_1,R2_GGG_2,AMPL(1
     $ ,18))
C     Counter-term amplitude(s) for loop diagram number 15
      CALL FFV1_0(W(1,8),W(1,4),W(1,6),R2_GQQ,AMPL(1,19))
      CALL FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQB_1EPS,AMPL(2,20))
      CALL FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQB_1EPS,AMPL(2,21))
      CALL FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQB_1EPS,AMPL(2,22))
      CALL FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQB_1EPS,AMPL(2,23))
      CALL FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQB_1EPS,AMPL(2,24))
      CALL FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQB_1EPS,AMPL(2,25))
      CALL FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQG_1EPS,AMPL(2,26))
      CALL FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQB,AMPL(1,27))
      CALL FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQT,AMPL(1,28))
C     Counter-term amplitude(s) for loop diagram number 17
      CALL FFV2_0(W(1,13),W(1,4),W(1,3),R2_BXTW,AMPL(1,29))
C     Counter-term amplitude(s) for loop diagram number 21
      CALL FFV2_0(W(1,5),W(1,15),W(1,3),R2_BXTW,AMPL(1,30))
      CALL FFV2_1(W(1,9),W(1,3),GC_11,MDL_MB,ZERO,W(1,17))
C     Counter-term amplitude(s) for loop diagram number 22
      CALL R2_QQ_1_R2_QQ_2_0(W(1,10),W(1,17),R2_QQQ,R2_QQB,AMPL(1,31))
      CALL R2_QQ_2_0(W(1,10),W(1,17),UV_BMASS_1EPS,AMPL(2,32))
      CALL R2_QQ_2_0(W(1,10),W(1,17),UV_BMASS,AMPL(1,33))
      CALL FFV2_2(W(1,10),W(1,3),GC_11,MDL_MT,MDL_WT,W(1,18))
C     Counter-term amplitude(s) for loop diagram number 23
      CALL R2_QQ_1_R2_QQ_2_0(W(1,18),W(1,9),R2_QQQ,R2_QQT,AMPL(1,34))
      CALL R2_QQ_2_0(W(1,18),W(1,9),UV_TMASS_1EPS,AMPL(2,35))
      CALL R2_QQ_2_0(W(1,18),W(1,9),UV_TMASS,AMPL(1,36))
C     Counter-term amplitude(s) for loop diagram number 24
      CALL FFV2_0(W(1,10),W(1,9),W(1,3),R2_BXTW,AMPL(1,37))
C     Counter-term amplitude(s) for loop diagram number 25
      CALL FFV1_0(W(1,5),W(1,17),W(1,2),R2_GQQ,AMPL(1,38))
      CALL FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQB_1EPS,AMPL(2,39))
      CALL FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQB_1EPS,AMPL(2,40))
      CALL FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQB_1EPS,AMPL(2,41))
      CALL FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQB_1EPS,AMPL(2,42))
      CALL FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQB_1EPS,AMPL(2,43))
      CALL FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQB_1EPS,AMPL(2,44))
      CALL FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQG_1EPS,AMPL(2,45))
      CALL FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQB,AMPL(1,46))
      CALL FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQT,AMPL(1,47))
C     Counter-term amplitude(s) for loop diagram number 27
      CALL FFV1_0(W(1,8),W(1,9),W(1,2),R2_GQQ,AMPL(1,48))
      CALL FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQB_1EPS,AMPL(2,49))
      CALL FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQB_1EPS,AMPL(2,50))
      CALL FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQB_1EPS,AMPL(2,51))
      CALL FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQB_1EPS,AMPL(2,52))
      CALL FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQB_1EPS,AMPL(2,53))
      CALL FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQB_1EPS,AMPL(2,54))
      CALL FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQG_1EPS,AMPL(2,55))
      CALL FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQB,AMPL(1,56))
      CALL FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQT,AMPL(1,57))
      CALL FFV1_1(W(1,9),W(1,2),GC_5,MDL_MT,MDL_WT,W(1,19))
C     Counter-term amplitude(s) for loop diagram number 28
      CALL R2_QQ_1_R2_QQ_2_0(W(1,8),W(1,19),R2_QQQ,R2_QQT,AMPL(1,58))
      CALL R2_QQ_2_0(W(1,8),W(1,19),UV_TMASS_1EPS,AMPL(2,59))
      CALL R2_QQ_2_0(W(1,8),W(1,19),UV_TMASS,AMPL(1,60))
      CALL FFV1_2(W(1,8),W(1,2),GC_5,MDL_MT,MDL_WT,W(1,20))
C     Counter-term amplitude(s) for loop diagram number 29
      CALL R2_QQ_1_R2_QQ_2_0(W(1,20),W(1,9),R2_QQQ,R2_QQT,AMPL(1,61))
      CALL R2_QQ_2_0(W(1,20),W(1,9),UV_TMASS_1EPS,AMPL(2,62))
      CALL R2_QQ_2_0(W(1,20),W(1,9),UV_TMASS,AMPL(1,63))
C     Counter-term amplitude(s) for loop diagram number 31
      CALL FFV2_0(W(1,5),W(1,19),W(1,3),R2_BXTW,AMPL(1,64))
      CALL FFV2_2(W(1,11),W(1,3),GC_11,MDL_MT,MDL_WT,W(1,21))
C     Counter-term amplitude(s) for loop diagram number 35
      CALL R2_QQ_1_R2_QQ_2_0(W(1,21),W(1,12),R2_QQQ,R2_QQT,AMPL(1,65))
      CALL R2_QQ_2_0(W(1,21),W(1,12),UV_TMASS_1EPS,AMPL(2,66))
      CALL R2_QQ_2_0(W(1,21),W(1,12),UV_TMASS,AMPL(1,67))
      CALL FFV2_1(W(1,12),W(1,3),GC_11,MDL_MB,ZERO,W(1,22))
C     Counter-term amplitude(s) for loop diagram number 36
      CALL R2_QQ_1_R2_QQ_2_0(W(1,11),W(1,22),R2_QQQ,R2_QQB,AMPL(1,68))
      CALL R2_QQ_2_0(W(1,11),W(1,22),UV_BMASS_1EPS,AMPL(2,69))
      CALL R2_QQ_2_0(W(1,11),W(1,22),UV_BMASS,AMPL(1,70))
C     Counter-term amplitude(s) for loop diagram number 37
      CALL FFV2_0(W(1,11),W(1,12),W(1,3),R2_BXTW,AMPL(1,71))
C     Counter-term amplitude(s) for loop diagram number 38
      CALL FFV1_0(W(1,21),W(1,4),W(1,2),R2_GQQ,AMPL(1,72))
      CALL FFV1_0(W(1,21),W(1,4),W(1,2),UV_GQQB_1EPS,AMPL(2,73))
      CALL FFV1_0(W(1,21),W(1,4),W(1,2),UV_GQQB_1EPS,AMPL(2,74))
      CALL FFV1_0(W(1,21),W(1,4),W(1,2),UV_GQQB_1EPS,AMPL(2,75))
      CALL FFV1_0(W(1,21),W(1,4),W(1,2),UV_GQQB_1EPS,AMPL(2,76))
      CALL FFV1_0(W(1,21),W(1,4),W(1,2),UV_GQQB_1EPS,AMPL(2,77))
      CALL FFV1_0(W(1,21),W(1,4),W(1,2),UV_GQQB_1EPS,AMPL(2,78))
      CALL FFV1_0(W(1,21),W(1,4),W(1,2),UV_GQQG_1EPS,AMPL(2,79))
      CALL FFV1_0(W(1,21),W(1,4),W(1,2),UV_GQQB,AMPL(1,80))
      CALL FFV1_0(W(1,21),W(1,4),W(1,2),UV_GQQT,AMPL(1,81))
C     Counter-term amplitude(s) for loop diagram number 40
      CALL FFV1_0(W(1,11),W(1,7),W(1,2),R2_GQQ,AMPL(1,82))
      CALL FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQB_1EPS,AMPL(2,83))
      CALL FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQB_1EPS,AMPL(2,84))
      CALL FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQB_1EPS,AMPL(2,85))
      CALL FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQB_1EPS,AMPL(2,86))
      CALL FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQB_1EPS,AMPL(2,87))
      CALL FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQB_1EPS,AMPL(2,88))
      CALL FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQG_1EPS,AMPL(2,89))
      CALL FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQB,AMPL(1,90))
      CALL FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQT,AMPL(1,91))
      CALL FFV1_2(W(1,11),W(1,2),GC_5,MDL_MB,ZERO,W(1,23))
C     Counter-term amplitude(s) for loop diagram number 41
      CALL R2_QQ_1_R2_QQ_2_0(W(1,23),W(1,7),R2_QQQ,R2_QQB,AMPL(1,92))
      CALL R2_QQ_2_0(W(1,23),W(1,7),UV_BMASS_1EPS,AMPL(2,93))
      CALL R2_QQ_2_0(W(1,23),W(1,7),UV_BMASS,AMPL(1,94))
      CALL FFV1_1(W(1,7),W(1,2),GC_5,MDL_MB,ZERO,W(1,24))
C     Counter-term amplitude(s) for loop diagram number 42
      CALL R2_QQ_1_R2_QQ_2_0(W(1,11),W(1,24),R2_QQQ,R2_QQB,AMPL(1,95))
      CALL R2_QQ_2_0(W(1,11),W(1,24),UV_BMASS_1EPS,AMPL(2,96))
      CALL R2_QQ_2_0(W(1,11),W(1,24),UV_BMASS,AMPL(1,97))
C     Counter-term amplitude(s) for loop diagram number 44
      CALL FFV2_0(W(1,23),W(1,4),W(1,3),R2_BXTW,AMPL(1,98))
C     Counter-term amplitude(s) for loop diagram number 48
      CALL FFV1_0(W(1,5),W(1,22),W(1,1),R2_GQQ,AMPL(1,99))
      CALL FFV1_0(W(1,5),W(1,22),W(1,1),UV_GQQB_1EPS,AMPL(2,100))
      CALL FFV1_0(W(1,5),W(1,22),W(1,1),UV_GQQB_1EPS,AMPL(2,101))
      CALL FFV1_0(W(1,5),W(1,22),W(1,1),UV_GQQB_1EPS,AMPL(2,102))
      CALL FFV1_0(W(1,5),W(1,22),W(1,1),UV_GQQB_1EPS,AMPL(2,103))
      CALL FFV1_0(W(1,5),W(1,22),W(1,1),UV_GQQB_1EPS,AMPL(2,104))
      CALL FFV1_0(W(1,5),W(1,22),W(1,1),UV_GQQB_1EPS,AMPL(2,105))
      CALL FFV1_0(W(1,5),W(1,22),W(1,1),UV_GQQG_1EPS,AMPL(2,106))
      CALL FFV1_0(W(1,5),W(1,22),W(1,1),UV_GQQB,AMPL(1,107))
      CALL FFV1_0(W(1,5),W(1,22),W(1,1),UV_GQQT,AMPL(1,108))
C     Counter-term amplitude(s) for loop diagram number 50
      CALL FFV1_0(W(1,8),W(1,12),W(1,1),R2_GQQ,AMPL(1,109))
      CALL FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQB_1EPS,AMPL(2,110))
      CALL FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQB_1EPS,AMPL(2,111))
      CALL FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQB_1EPS,AMPL(2,112))
      CALL FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQB_1EPS,AMPL(2,113))
      CALL FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQB_1EPS,AMPL(2,114))
      CALL FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQB_1EPS,AMPL(2,115))
      CALL FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQG_1EPS,AMPL(2,116))
      CALL FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQB,AMPL(1,117))
      CALL FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQT,AMPL(1,118))
C     Counter-term amplitude(s) for loop diagram number 51
      CALL FFV1_0(W(1,18),W(1,4),W(1,1),R2_GQQ,AMPL(1,119))
      CALL FFV1_0(W(1,18),W(1,4),W(1,1),UV_GQQB_1EPS,AMPL(2,120))
      CALL FFV1_0(W(1,18),W(1,4),W(1,1),UV_GQQB_1EPS,AMPL(2,121))
      CALL FFV1_0(W(1,18),W(1,4),W(1,1),UV_GQQB_1EPS,AMPL(2,122))
      CALL FFV1_0(W(1,18),W(1,4),W(1,1),UV_GQQB_1EPS,AMPL(2,123))
      CALL FFV1_0(W(1,18),W(1,4),W(1,1),UV_GQQB_1EPS,AMPL(2,124))
      CALL FFV1_0(W(1,18),W(1,4),W(1,1),UV_GQQB_1EPS,AMPL(2,125))
      CALL FFV1_0(W(1,18),W(1,4),W(1,1),UV_GQQG_1EPS,AMPL(2,126))
      CALL FFV1_0(W(1,18),W(1,4),W(1,1),UV_GQQB,AMPL(1,127))
      CALL FFV1_0(W(1,18),W(1,4),W(1,1),UV_GQQT,AMPL(1,128))
C     Counter-term amplitude(s) for loop diagram number 53
      CALL FFV1_0(W(1,10),W(1,7),W(1,1),R2_GQQ,AMPL(1,129))
      CALL FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQB_1EPS,AMPL(2,130))
      CALL FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQB_1EPS,AMPL(2,131))
      CALL FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQB_1EPS,AMPL(2,132))
      CALL FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQB_1EPS,AMPL(2,133))
      CALL FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQB_1EPS,AMPL(2,134))
      CALL FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQB_1EPS,AMPL(2,135))
      CALL FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQG_1EPS,AMPL(2,136))
      CALL FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQB,AMPL(1,137))
      CALL FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQT,AMPL(1,138))
C     Counter-term amplitude(s) for loop diagram number 56
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GG,AMPL(1,139))
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB_1EPS,AMPL(2,140))
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB_1EPS,AMPL(2,141))
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB_1EPS,AMPL(2,142))
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB_1EPS,AMPL(2,143))
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB_1EPS,AMPL(2,144))
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB_1EPS,AMPL(2,145))
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GG_1EPS,AMPL(2,146))
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB,AMPL(1,147))
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GT,AMPL(1,148))
C     Counter-term amplitude(s) for loop diagram number 59
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),R2_3GG,AMPL(1,149))
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),UV_3GB_1EPS,AMPL(2,150))
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),UV_3GB_1EPS,AMPL(2,151))
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),UV_3GB_1EPS,AMPL(2,152))
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),UV_3GB_1EPS,AMPL(2,153))
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),UV_3GB_1EPS,AMPL(2,154))
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),UV_3GB_1EPS,AMPL(2,155))
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),UV_3GG_1EPS,AMPL(2,156))
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),UV_3GB,AMPL(1,157))
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),UV_3GT,AMPL(1,158))
C     Counter-term amplitude(s) for loop diagram number 62
      CALL FFV1_0(W(1,5),W(1,24),W(1,1),R2_GQQ,AMPL(1,159))
      CALL FFV1_0(W(1,5),W(1,24),W(1,1),UV_GQQB_1EPS,AMPL(2,160))
      CALL FFV1_0(W(1,5),W(1,24),W(1,1),UV_GQQB_1EPS,AMPL(2,161))
      CALL FFV1_0(W(1,5),W(1,24),W(1,1),UV_GQQB_1EPS,AMPL(2,162))
      CALL FFV1_0(W(1,5),W(1,24),W(1,1),UV_GQQB_1EPS,AMPL(2,163))
      CALL FFV1_0(W(1,5),W(1,24),W(1,1),UV_GQQB_1EPS,AMPL(2,164))
      CALL FFV1_0(W(1,5),W(1,24),W(1,1),UV_GQQB_1EPS,AMPL(2,165))
      CALL FFV1_0(W(1,5),W(1,24),W(1,1),UV_GQQG_1EPS,AMPL(2,166))
      CALL FFV1_0(W(1,5),W(1,24),W(1,1),UV_GQQB,AMPL(1,167))
      CALL FFV1_0(W(1,5),W(1,24),W(1,1),UV_GQQT,AMPL(1,168))
C     Counter-term amplitude(s) for loop diagram number 65
      CALL FFV1_0(W(1,20),W(1,4),W(1,1),R2_GQQ,AMPL(1,169))
      CALL FFV1_0(W(1,20),W(1,4),W(1,1),UV_GQQB_1EPS,AMPL(2,170))
      CALL FFV1_0(W(1,20),W(1,4),W(1,1),UV_GQQB_1EPS,AMPL(2,171))
      CALL FFV1_0(W(1,20),W(1,4),W(1,1),UV_GQQB_1EPS,AMPL(2,172))
      CALL FFV1_0(W(1,20),W(1,4),W(1,1),UV_GQQB_1EPS,AMPL(2,173))
      CALL FFV1_0(W(1,20),W(1,4),W(1,1),UV_GQQB_1EPS,AMPL(2,174))
      CALL FFV1_0(W(1,20),W(1,4),W(1,1),UV_GQQB_1EPS,AMPL(2,175))
      CALL FFV1_0(W(1,20),W(1,4),W(1,1),UV_GQQG_1EPS,AMPL(2,176))
      CALL FFV1_0(W(1,20),W(1,4),W(1,1),UV_GQQB,AMPL(1,177))
      CALL FFV1_0(W(1,20),W(1,4),W(1,1),UV_GQQT,AMPL(1,178))
      CALL FFV1_1(W(1,12),W(1,1),GC_5,MDL_MT,MDL_WT,W(1,25))
C     Counter-term amplitude(s) for loop diagram number 70
      CALL R2_QQ_1_R2_QQ_2_0(W(1,8),W(1,25),R2_QQQ,R2_QQT,AMPL(1,179))
      CALL R2_QQ_2_0(W(1,8),W(1,25),UV_TMASS_1EPS,AMPL(2,180))
      CALL R2_QQ_2_0(W(1,8),W(1,25),UV_TMASS,AMPL(1,181))
      CALL FFV1_2(W(1,8),W(1,1),GC_5,MDL_MT,MDL_WT,W(1,26))
C     Counter-term amplitude(s) for loop diagram number 71
      CALL R2_QQ_1_R2_QQ_2_0(W(1,26),W(1,12),R2_QQQ,R2_QQT,AMPL(1,182))
      CALL R2_QQ_2_0(W(1,26),W(1,12),UV_TMASS_1EPS,AMPL(2,183))
      CALL R2_QQ_2_0(W(1,26),W(1,12),UV_TMASS,AMPL(1,184))
C     Counter-term amplitude(s) for loop diagram number 73
      CALL FFV2_0(W(1,5),W(1,25),W(1,3),R2_BXTW,AMPL(1,185))
      CALL FFV1_2(W(1,10),W(1,1),GC_5,MDL_MB,ZERO,W(1,27))
C     Counter-term amplitude(s) for loop diagram number 77
      CALL R2_QQ_1_R2_QQ_2_0(W(1,27),W(1,7),R2_QQQ,R2_QQB,AMPL(1,186))
      CALL R2_QQ_2_0(W(1,27),W(1,7),UV_BMASS_1EPS,AMPL(2,187))
      CALL R2_QQ_2_0(W(1,27),W(1,7),UV_BMASS,AMPL(1,188))
      CALL FFV1_1(W(1,7),W(1,1),GC_5,MDL_MB,ZERO,W(1,28))
C     Counter-term amplitude(s) for loop diagram number 78
      CALL R2_QQ_1_R2_QQ_2_0(W(1,10),W(1,28),R2_QQQ,R2_QQB,AMPL(1,189))
      CALL R2_QQ_2_0(W(1,10),W(1,28),UV_BMASS_1EPS,AMPL(2,190))
      CALL R2_QQ_2_0(W(1,10),W(1,28),UV_BMASS,AMPL(1,191))
C     Counter-term amplitude(s) for loop diagram number 80
      CALL FFV2_0(W(1,27),W(1,4),W(1,3),R2_BXTW,AMPL(1,192))
C     Counter-term amplitude(s) for loop diagram number 84
      CALL FFV1_0(W(1,5),W(1,28),W(1,2),R2_GQQ,AMPL(1,193))
      CALL FFV1_0(W(1,5),W(1,28),W(1,2),UV_GQQB_1EPS,AMPL(2,194))
      CALL FFV1_0(W(1,5),W(1,28),W(1,2),UV_GQQB_1EPS,AMPL(2,195))
      CALL FFV1_0(W(1,5),W(1,28),W(1,2),UV_GQQB_1EPS,AMPL(2,196))
      CALL FFV1_0(W(1,5),W(1,28),W(1,2),UV_GQQB_1EPS,AMPL(2,197))
      CALL FFV1_0(W(1,5),W(1,28),W(1,2),UV_GQQB_1EPS,AMPL(2,198))
      CALL FFV1_0(W(1,5),W(1,28),W(1,2),UV_GQQB_1EPS,AMPL(2,199))
      CALL FFV1_0(W(1,5),W(1,28),W(1,2),UV_GQQG_1EPS,AMPL(2,200))
      CALL FFV1_0(W(1,5),W(1,28),W(1,2),UV_GQQB,AMPL(1,201))
      CALL FFV1_0(W(1,5),W(1,28),W(1,2),UV_GQQT,AMPL(1,202))
C     Counter-term amplitude(s) for loop diagram number 87
      CALL FFV1_0(W(1,26),W(1,4),W(1,2),R2_GQQ,AMPL(1,203))
      CALL FFV1_0(W(1,26),W(1,4),W(1,2),UV_GQQB_1EPS,AMPL(2,204))
      CALL FFV1_0(W(1,26),W(1,4),W(1,2),UV_GQQB_1EPS,AMPL(2,205))
      CALL FFV1_0(W(1,26),W(1,4),W(1,2),UV_GQQB_1EPS,AMPL(2,206))
      CALL FFV1_0(W(1,26),W(1,4),W(1,2),UV_GQQB_1EPS,AMPL(2,207))
      CALL FFV1_0(W(1,26),W(1,4),W(1,2),UV_GQQB_1EPS,AMPL(2,208))
      CALL FFV1_0(W(1,26),W(1,4),W(1,2),UV_GQQB_1EPS,AMPL(2,209))
      CALL FFV1_0(W(1,26),W(1,4),W(1,2),UV_GQQG_1EPS,AMPL(2,210))
      CALL FFV1_0(W(1,26),W(1,4),W(1,2),UV_GQQB,AMPL(1,211))
      CALL FFV1_0(W(1,26),W(1,4),W(1,2),UV_GQQT,AMPL(1,212))
C     Counter-term amplitude(s) for loop diagram number 117
      CALL R2_GG_1_0(W(1,6),W(1,14),R2_GGQ,AMPL(1,213))
      CALL R2_GG_1_0(W(1,6),W(1,14),R2_GGQ,AMPL(1,214))
      CALL R2_GG_1_0(W(1,6),W(1,14),R2_GGQ,AMPL(1,215))
      CALL R2_GG_1_0(W(1,6),W(1,14),R2_GGQ,AMPL(1,216))
C     Counter-term amplitude(s) for loop diagram number 118
      CALL R2_GG_1_0(W(1,6),W(1,16),R2_GGQ,AMPL(1,217))
      CALL R2_GG_1_0(W(1,6),W(1,16),R2_GGQ,AMPL(1,218))
      CALL R2_GG_1_0(W(1,6),W(1,16),R2_GGQ,AMPL(1,219))
      CALL R2_GG_1_0(W(1,6),W(1,16),R2_GGQ,AMPL(1,220))
C     Counter-term amplitude(s) for loop diagram number 119
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GQ,AMPL(1,221))
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GQ,AMPL(1,222))
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GQ,AMPL(1,223))
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GQ,AMPL(1,224))
C     Counter-term amplitude(s) for loop diagram number 120
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),R2_3GQ,AMPL(1,225))
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),R2_3GQ,AMPL(1,226))
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),R2_3GQ,AMPL(1,227))
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),R2_3GQ,AMPL(1,228))
C     Counter-term amplitude(s) for loop diagram number 123
      CALL R2_GG_1_R2_GG_3_0(W(1,6),W(1,14),R2_GGQ,R2_GGB,AMPL(1,229))
C     Counter-term amplitude(s) for loop diagram number 124
      CALL R2_GG_1_R2_GG_3_0(W(1,6),W(1,16),R2_GGQ,R2_GGB,AMPL(1,230))
C     Counter-term amplitude(s) for loop diagram number 125
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GQ,AMPL(1,231))
C     Counter-term amplitude(s) for loop diagram number 126
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),R2_3GQ,AMPL(1,232))
C     Counter-term amplitude(s) for loop diagram number 129
      CALL R2_GG_1_R2_GG_3_0(W(1,6),W(1,14),R2_GGQ,R2_GGT,AMPL(1,233))
C     Counter-term amplitude(s) for loop diagram number 130
      CALL R2_GG_1_R2_GG_3_0(W(1,6),W(1,16),R2_GGQ,R2_GGT,AMPL(1,234))
C     Counter-term amplitude(s) for loop diagram number 131
      CALL VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GQ,AMPL(1,235))
C     Counter-term amplitude(s) for loop diagram number 132
      CALL VVV1_0(W(1,1),W(1,2),W(1,16),R2_3GQ,AMPL(1,236))

      GOTO 1001
 2000 CONTINUE
      CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

