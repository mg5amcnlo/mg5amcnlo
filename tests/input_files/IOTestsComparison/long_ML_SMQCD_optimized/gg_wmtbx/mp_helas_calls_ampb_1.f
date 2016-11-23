      SUBROUTINE ML5_0_MP_HELAS_CALLS_AMPB_1(P,NHEL,H,IC)
C     
      USE ML5_0_POLYNOMIAL_CONSTANTS
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
      REAL*16     ZERO
      PARAMETER (ZERO=0.0E0_16)
      COMPLEX*32     IZERO
      PARAMETER (IZERO=CMPLX(0.0E0_16,0.0E0_16,KIND=16))
C     These are constants related to the split orders
      INTEGER    NSO, NSQUAREDSO, NAMPSO
      PARAMETER (NSO=0, NSQUAREDSO=0, NAMPSO=0)
C     
C     ARGUMENTS
C     
      REAL*16 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER H
C     
C     LOCAL VARIABLES
C     
      INTEGER I,J,K
      COMPLEX*32 COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'mp_coupl_same_name.inc'

      INTEGER GOODHEL(NCOMB)
      LOGICAL GOODAMP(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_FILTERS/GOODAMP,GOODHEL

      INTEGER SQSO_TARGET
      COMMON/ML5_0_SOCHOICE/SQSO_TARGET

      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE
     $ ,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE
     $ ,CTCALL_REQ_SO_DONE,FILTER_SO
      COMMON/ML5_0_SO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE
     $ ,CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE
     $ ,MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO

      COMPLEX*32 AMP(NBORNAMPS)
      COMMON/ML5_0_MP_AMPS/AMP
      COMPLEX*32 W(20,NWAVEFUNCS)
      COMMON/ML5_0_MP_W/W

      COMPLEX*32 WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE
     $ ,0:NLOOPWAVEFUNCS)
      COMPLEX*32 PL(0:3,0:NLOOPWAVEFUNCS)
      COMMON/ML5_0_MP_WL/WL,PL

      COMPLEX*32 AMPL(3,NCTAMPS)
      COMMON/ML5_0_MP_AMPL/AMPL

C     
C     ----------
C     BEGIN CODE
C     ----------

C     The target squared split order contribution is already reached
C      if true.
      IF (FILTER_SO.AND.MP_CT_REQ_SO_DONE) THEN
        GOTO 1001
      ENDIF

      CALL MP_VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
      CALL MP_VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
      CALL MP_VXXXXX(P(0,3),MDL_MW,NHEL(3),+1*IC(3),W(1,3))
      CALL MP_OXXXXX(P(0,4),MDL_MT,NHEL(4),+1*IC(4),W(1,4))
      CALL MP_IXXXXX(P(0,5),MDL_MB,NHEL(5),-1*IC(5),W(1,5))
      CALL MP_VVV1P0_1(W(1,1),W(1,2),GC_4,ZERO,ZERO,W(1,6))
      CALL MP_FFV2_1(W(1,4),W(1,3),GC_47,MDL_MB,ZERO,W(1,7))
C     Amplitude(s) for born diagram with ID 1
      CALL MP_FFV1_0(W(1,5),W(1,7),W(1,6),GC_5,AMP(1))
      CALL MP_FFV2_2(W(1,5),W(1,3),GC_47,MDL_MT,MDL_WT,W(1,8))
C     Amplitude(s) for born diagram with ID 2
      CALL MP_FFV1_0(W(1,8),W(1,4),W(1,6),GC_5,AMP(2))
      CALL MP_FFV1_1(W(1,4),W(1,1),GC_5,MDL_MT,MDL_WT,W(1,9))
      CALL MP_FFV1_2(W(1,5),W(1,2),GC_5,MDL_MB,ZERO,W(1,10))
C     Amplitude(s) for born diagram with ID 3
      CALL MP_FFV2_0(W(1,10),W(1,9),W(1,3),GC_47,AMP(3))
C     Amplitude(s) for born diagram with ID 4
      CALL MP_FFV1_0(W(1,8),W(1,9),W(1,2),GC_5,AMP(4))
      CALL MP_FFV1_2(W(1,5),W(1,1),GC_5,MDL_MB,ZERO,W(1,11))
      CALL MP_FFV1_1(W(1,4),W(1,2),GC_5,MDL_MT,MDL_WT,W(1,12))
C     Amplitude(s) for born diagram with ID 5
      CALL MP_FFV2_0(W(1,11),W(1,12),W(1,3),GC_47,AMP(5))
C     Amplitude(s) for born diagram with ID 6
      CALL MP_FFV1_0(W(1,11),W(1,7),W(1,2),GC_5,AMP(6))
C     Amplitude(s) for born diagram with ID 7
      CALL MP_FFV1_0(W(1,8),W(1,12),W(1,1),GC_5,AMP(7))
C     Amplitude(s) for born diagram with ID 8
      CALL MP_FFV1_0(W(1,10),W(1,7),W(1,1),GC_5,AMP(8))
      CALL MP_FFV1P0_3(W(1,5),W(1,7),GC_5,ZERO,ZERO,W(1,13))
C     Counter-term amplitude(s) for loop diagram number 9
      CALL MP_R2_GG_1_0(W(1,6),W(1,13),R2_GGQ,AMPL(1,1))
      CALL MP_R2_GG_1_0(W(1,6),W(1,13),R2_GGQ,AMPL(1,2))
      CALL MP_R2_GG_1_0(W(1,6),W(1,13),R2_GGQ,AMPL(1,3))
      CALL MP_R2_GG_1_0(W(1,6),W(1,13),R2_GGQ,AMPL(1,4))
      CALL MP_FFV1P0_3(W(1,8),W(1,4),GC_5,ZERO,ZERO,W(1,14))
C     Counter-term amplitude(s) for loop diagram number 10
      CALL MP_R2_GG_1_0(W(1,6),W(1,14),R2_GGQ,AMPL(1,5))
      CALL MP_R2_GG_1_0(W(1,6),W(1,14),R2_GGQ,AMPL(1,6))
      CALL MP_R2_GG_1_0(W(1,6),W(1,14),R2_GGQ,AMPL(1,7))
      CALL MP_R2_GG_1_0(W(1,6),W(1,14),R2_GGQ,AMPL(1,8))
C     Counter-term amplitude(s) for loop diagram number 11
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),UV_3GB_1EPS,AMPL(2,9))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),UV_3GB_1EPS,AMPL(2,10))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),UV_3GB_1EPS,AMPL(2,11))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),UV_3GB_1EPS,AMPL(2,12))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),UV_3GB,AMPL(1,13))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),UV_3GB_1EPS,AMPL(2,14))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),UV_3GT,AMPL(1,15))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),UV_3GB_1EPS,AMPL(2,16))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),UV_3GG_1EPS,AMPL(2,17))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),R2_3GQ,AMPL(1,18))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),R2_3GQ,AMPL(1,19))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),R2_3GQ,AMPL(1,20))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),R2_3GQ,AMPL(1,21))
C     Counter-term amplitude(s) for loop diagram number 12
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB_1EPS,AMPL(2,22))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB_1EPS,AMPL(2,23))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB_1EPS,AMPL(2,24))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB_1EPS,AMPL(2,25))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB,AMPL(1,26))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB_1EPS,AMPL(2,27))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GT,AMPL(1,28))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GB_1EPS,AMPL(2,29))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),UV_3GG_1EPS,AMPL(2,30))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GQ,AMPL(1,31))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GQ,AMPL(1,32))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GQ,AMPL(1,33))
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GQ,AMPL(1,34))
      CALL MP_FFV1_2(W(1,5),W(1,6),GC_5,MDL_MB,ZERO,W(1,15))
C     Counter-term amplitude(s) for loop diagram number 15
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,15),W(1,7),R2_QQQ,R2_QQB,AMPL(1,35)
     $ )
      CALL MP_R2_QQ_2_0(W(1,15),W(1,7),UV_BMASS,AMPL(1,36))
      CALL MP_R2_QQ_2_0(W(1,15),W(1,7),UV_BMASS_1EPS,AMPL(2,37))
C     Counter-term amplitude(s) for loop diagram number 16
      CALL MP_R2_GG_1_R2_GG_3_0(W(1,6),W(1,13),R2_GGQ,R2_GGB,AMPL(1,38)
     $ )
C     Counter-term amplitude(s) for loop diagram number 17
      CALL MP_FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQQ_1EPS,AMPL(2,39))
      CALL MP_FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQQ_1EPS,AMPL(2,40))
      CALL MP_FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQQ_1EPS,AMPL(2,41))
      CALL MP_FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQQ_1EPS,AMPL(2,42))
      CALL MP_FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQB,AMPL(1,43))
      CALL MP_FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQQ_1EPS,AMPL(2,44))
      CALL MP_FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQT,AMPL(1,45))
      CALL MP_FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQQ_1EPS,AMPL(2,46))
      CALL MP_FFV1_0(W(1,5),W(1,7),W(1,6),UV_GQQG_1EPS,AMPL(2,47))
      CALL MP_FFV1_0(W(1,5),W(1,7),W(1,6),R2_GQQ,AMPL(1,48))
C     Counter-term amplitude(s) for loop diagram number 19
      CALL MP_R2_GG_1_R2_GG_3_0(W(1,6),W(1,14),R2_GGQ,R2_GGB,AMPL(1,49)
     $ )
C     Counter-term amplitude(s) for loop diagram number 20
      CALL MP_FFV2_0(W(1,15),W(1,4),W(1,3),R2_SXCW,AMPL(1,50))
      CALL MP_FFV1_1(W(1,4),W(1,6),GC_5,MDL_MT,MDL_WT,W(1,16))
C     Counter-term amplitude(s) for loop diagram number 23
      CALL MP_FFV2_0(W(1,5),W(1,16),W(1,3),R2_SXCW,AMPL(1,51))
      CALL MP_FFV2_1(W(1,9),W(1,3),GC_47,MDL_MB,ZERO,W(1,17))
C     Counter-term amplitude(s) for loop diagram number 25
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,10),W(1,17),R2_QQQ,R2_QQB,AMPL(1
     $ ,52))
      CALL MP_R2_QQ_2_0(W(1,10),W(1,17),UV_BMASS,AMPL(1,53))
      CALL MP_R2_QQ_2_0(W(1,10),W(1,17),UV_BMASS_1EPS,AMPL(2,54))
C     Counter-term amplitude(s) for loop diagram number 26
      CALL MP_FFV2_0(W(1,10),W(1,9),W(1,3),R2_SXCW,AMPL(1,55))
C     Counter-term amplitude(s) for loop diagram number 27
      CALL MP_FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQQ_1EPS,AMPL(2,56))
      CALL MP_FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQQ_1EPS,AMPL(2,57))
      CALL MP_FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQQ_1EPS,AMPL(2,58))
      CALL MP_FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQQ_1EPS,AMPL(2,59))
      CALL MP_FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQB,AMPL(1,60))
      CALL MP_FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQQ_1EPS,AMPL(2,61))
      CALL MP_FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQT,AMPL(1,62))
      CALL MP_FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQQ_1EPS,AMPL(2,63))
      CALL MP_FFV1_0(W(1,5),W(1,17),W(1,2),UV_GQQG_1EPS,AMPL(2,64))
      CALL MP_FFV1_0(W(1,5),W(1,17),W(1,2),R2_GQQ,AMPL(1,65))
      CALL MP_FFV1_1(W(1,9),W(1,2),GC_5,MDL_MT,MDL_WT,W(1,18))
C     Counter-term amplitude(s) for loop diagram number 29
      CALL MP_FFV2_0(W(1,5),W(1,18),W(1,3),R2_SXCW,AMPL(1,66))
      CALL MP_FFV2_1(W(1,12),W(1,3),GC_47,MDL_MB,ZERO,W(1,19))
C     Counter-term amplitude(s) for loop diagram number 33
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,11),W(1,19),R2_QQQ,R2_QQB,AMPL(1
     $ ,67))
      CALL MP_R2_QQ_2_0(W(1,11),W(1,19),UV_BMASS,AMPL(1,68))
      CALL MP_R2_QQ_2_0(W(1,11),W(1,19),UV_BMASS_1EPS,AMPL(2,69))
C     Counter-term amplitude(s) for loop diagram number 34
      CALL MP_FFV2_0(W(1,11),W(1,12),W(1,3),R2_SXCW,AMPL(1,70))
C     Counter-term amplitude(s) for loop diagram number 35
      CALL MP_FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQQ_1EPS,AMPL(2,71))
      CALL MP_FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQQ_1EPS,AMPL(2,72))
      CALL MP_FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQQ_1EPS,AMPL(2,73))
      CALL MP_FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQQ_1EPS,AMPL(2,74))
      CALL MP_FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQB,AMPL(1,75))
      CALL MP_FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQQ_1EPS,AMPL(2,76))
      CALL MP_FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQT,AMPL(1,77))
      CALL MP_FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQQ_1EPS,AMPL(2,78))
      CALL MP_FFV1_0(W(1,11),W(1,7),W(1,2),UV_GQQG_1EPS,AMPL(2,79))
      CALL MP_FFV1_0(W(1,11),W(1,7),W(1,2),R2_GQQ,AMPL(1,80))
      CALL MP_FFV1_2(W(1,11),W(1,2),GC_5,MDL_MB,ZERO,W(1,20))
C     Counter-term amplitude(s) for loop diagram number 37
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,20),W(1,7),R2_QQQ,R2_QQB,AMPL(1,81)
     $ )
      CALL MP_R2_QQ_2_0(W(1,20),W(1,7),UV_BMASS,AMPL(1,82))
      CALL MP_R2_QQ_2_0(W(1,20),W(1,7),UV_BMASS_1EPS,AMPL(2,83))
      CALL MP_FFV1_1(W(1,7),W(1,2),GC_5,MDL_MB,ZERO,W(1,21))
C     Counter-term amplitude(s) for loop diagram number 38
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,11),W(1,21),R2_QQQ,R2_QQB,AMPL(1
     $ ,84))
      CALL MP_R2_QQ_2_0(W(1,11),W(1,21),UV_BMASS,AMPL(1,85))
      CALL MP_R2_QQ_2_0(W(1,11),W(1,21),UV_BMASS_1EPS,AMPL(2,86))
C     Counter-term amplitude(s) for loop diagram number 40
      CALL MP_FFV2_0(W(1,20),W(1,4),W(1,3),R2_SXCW,AMPL(1,87))
C     Counter-term amplitude(s) for loop diagram number 43
      CALL MP_FFV1_0(W(1,5),W(1,19),W(1,1),UV_GQQQ_1EPS,AMPL(2,88))
      CALL MP_FFV1_0(W(1,5),W(1,19),W(1,1),UV_GQQQ_1EPS,AMPL(2,89))
      CALL MP_FFV1_0(W(1,5),W(1,19),W(1,1),UV_GQQQ_1EPS,AMPL(2,90))
      CALL MP_FFV1_0(W(1,5),W(1,19),W(1,1),UV_GQQQ_1EPS,AMPL(2,91))
      CALL MP_FFV1_0(W(1,5),W(1,19),W(1,1),UV_GQQB,AMPL(1,92))
      CALL MP_FFV1_0(W(1,5),W(1,19),W(1,1),UV_GQQQ_1EPS,AMPL(2,93))
      CALL MP_FFV1_0(W(1,5),W(1,19),W(1,1),UV_GQQT,AMPL(1,94))
      CALL MP_FFV1_0(W(1,5),W(1,19),W(1,1),UV_GQQQ_1EPS,AMPL(2,95))
      CALL MP_FFV1_0(W(1,5),W(1,19),W(1,1),UV_GQQG_1EPS,AMPL(2,96))
      CALL MP_FFV1_0(W(1,5),W(1,19),W(1,1),R2_GQQ,AMPL(1,97))
C     Counter-term amplitude(s) for loop diagram number 45
      CALL MP_FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQQ_1EPS,AMPL(2,98))
      CALL MP_FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQQ_1EPS,AMPL(2,99))
      CALL MP_FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQQ_1EPS,AMPL(2,100))
      CALL MP_FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQQ_1EPS,AMPL(2,101))
      CALL MP_FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQB,AMPL(1,102))
      CALL MP_FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQQ_1EPS,AMPL(2,103))
      CALL MP_FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQT,AMPL(1,104))
      CALL MP_FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQQ_1EPS,AMPL(2,105))
      CALL MP_FFV1_0(W(1,10),W(1,7),W(1,1),UV_GQQG_1EPS,AMPL(2,106))
      CALL MP_FFV1_0(W(1,10),W(1,7),W(1,1),R2_GQQ,AMPL(1,107))
C     Counter-term amplitude(s) for loop diagram number 48
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),R2_3GQ,AMPL(1,108))
C     Counter-term amplitude(s) for loop diagram number 50
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GQ,AMPL(1,109))
C     Counter-term amplitude(s) for loop diagram number 53
      CALL MP_FFV1_0(W(1,5),W(1,21),W(1,1),UV_GQQQ_1EPS,AMPL(2,110))
      CALL MP_FFV1_0(W(1,5),W(1,21),W(1,1),UV_GQQQ_1EPS,AMPL(2,111))
      CALL MP_FFV1_0(W(1,5),W(1,21),W(1,1),UV_GQQQ_1EPS,AMPL(2,112))
      CALL MP_FFV1_0(W(1,5),W(1,21),W(1,1),UV_GQQQ_1EPS,AMPL(2,113))
      CALL MP_FFV1_0(W(1,5),W(1,21),W(1,1),UV_GQQB,AMPL(1,114))
      CALL MP_FFV1_0(W(1,5),W(1,21),W(1,1),UV_GQQQ_1EPS,AMPL(2,115))
      CALL MP_FFV1_0(W(1,5),W(1,21),W(1,1),UV_GQQT,AMPL(1,116))
      CALL MP_FFV1_0(W(1,5),W(1,21),W(1,1),UV_GQQQ_1EPS,AMPL(2,117))
      CALL MP_FFV1_0(W(1,5),W(1,21),W(1,1),UV_GQQG_1EPS,AMPL(2,118))
      CALL MP_FFV1_0(W(1,5),W(1,21),W(1,1),R2_GQQ,AMPL(1,119))
      CALL MP_FFV1_1(W(1,12),W(1,1),GC_5,MDL_MT,MDL_WT,W(1,22))
C     Counter-term amplitude(s) for loop diagram number 59
      CALL MP_FFV2_0(W(1,5),W(1,22),W(1,3),R2_SXCW,AMPL(1,120))
      CALL MP_FFV1_2(W(1,10),W(1,1),GC_5,MDL_MB,ZERO,W(1,23))
C     Counter-term amplitude(s) for loop diagram number 63
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,23),W(1,7),R2_QQQ,R2_QQB,AMPL(1
     $ ,121))
      CALL MP_R2_QQ_2_0(W(1,23),W(1,7),UV_BMASS,AMPL(1,122))
      CALL MP_R2_QQ_2_0(W(1,23),W(1,7),UV_BMASS_1EPS,AMPL(2,123))
      CALL MP_FFV1_1(W(1,7),W(1,1),GC_5,MDL_MB,ZERO,W(1,24))
C     Counter-term amplitude(s) for loop diagram number 64
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,10),W(1,24),R2_QQQ,R2_QQB,AMPL(1
     $ ,124))
      CALL MP_R2_QQ_2_0(W(1,10),W(1,24),UV_BMASS,AMPL(1,125))
      CALL MP_R2_QQ_2_0(W(1,10),W(1,24),UV_BMASS_1EPS,AMPL(2,126))
C     Counter-term amplitude(s) for loop diagram number 66
      CALL MP_FFV2_0(W(1,23),W(1,4),W(1,3),R2_SXCW,AMPL(1,127))
C     Counter-term amplitude(s) for loop diagram number 69
      CALL MP_FFV1_0(W(1,5),W(1,24),W(1,2),UV_GQQQ_1EPS,AMPL(2,128))
      CALL MP_FFV1_0(W(1,5),W(1,24),W(1,2),UV_GQQQ_1EPS,AMPL(2,129))
      CALL MP_FFV1_0(W(1,5),W(1,24),W(1,2),UV_GQQQ_1EPS,AMPL(2,130))
      CALL MP_FFV1_0(W(1,5),W(1,24),W(1,2),UV_GQQQ_1EPS,AMPL(2,131))
      CALL MP_FFV1_0(W(1,5),W(1,24),W(1,2),UV_GQQB,AMPL(1,132))
      CALL MP_FFV1_0(W(1,5),W(1,24),W(1,2),UV_GQQQ_1EPS,AMPL(2,133))
      CALL MP_FFV1_0(W(1,5),W(1,24),W(1,2),UV_GQQT,AMPL(1,134))
      CALL MP_FFV1_0(W(1,5),W(1,24),W(1,2),UV_GQQQ_1EPS,AMPL(2,135))
      CALL MP_FFV1_0(W(1,5),W(1,24),W(1,2),UV_GQQG_1EPS,AMPL(2,136))
      CALL MP_FFV1_0(W(1,5),W(1,24),W(1,2),R2_GQQ,AMPL(1,137))
C     Counter-term amplitude(s) for loop diagram number 85
      CALL MP_R2_GG_1_R2_GG_3_0(W(1,6),W(1,13),R2_GGQ,R2_GGT,AMPL(1
     $ ,138))
C     Counter-term amplitude(s) for loop diagram number 86
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,8),W(1,16),R2_QQQ,R2_QQT,AMPL(1
     $ ,139))
      CALL MP_R2_QQ_2_0(W(1,8),W(1,16),UV_TMASS,AMPL(1,140))
      CALL MP_R2_QQ_2_0(W(1,8),W(1,16),UV_TMASS_1EPS,AMPL(2,141))
C     Counter-term amplitude(s) for loop diagram number 87
      CALL MP_R2_GG_1_R2_GG_3_0(W(1,6),W(1,14),R2_GGQ,R2_GGT,AMPL(1
     $ ,142))
C     Counter-term amplitude(s) for loop diagram number 88
      CALL MP_FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQQ_1EPS,AMPL(2,143))
      CALL MP_FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQQ_1EPS,AMPL(2,144))
      CALL MP_FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQQ_1EPS,AMPL(2,145))
      CALL MP_FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQQ_1EPS,AMPL(2,146))
      CALL MP_FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQB,AMPL(1,147))
      CALL MP_FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQQ_1EPS,AMPL(2,148))
      CALL MP_FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQT,AMPL(1,149))
      CALL MP_FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQQ_1EPS,AMPL(2,150))
      CALL MP_FFV1_0(W(1,8),W(1,4),W(1,6),UV_GQQG_1EPS,AMPL(2,151))
      CALL MP_FFV1_0(W(1,8),W(1,4),W(1,6),R2_GQQ,AMPL(1,152))
      CALL MP_FFV2_2(W(1,10),W(1,3),GC_47,MDL_MT,MDL_WT,W(1,25))
C     Counter-term amplitude(s) for loop diagram number 90
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,25),W(1,9),R2_QQQ,R2_QQT,AMPL(1
     $ ,153))
      CALL MP_R2_QQ_2_0(W(1,25),W(1,9),UV_TMASS,AMPL(1,154))
      CALL MP_R2_QQ_2_0(W(1,25),W(1,9),UV_TMASS_1EPS,AMPL(2,155))
C     Counter-term amplitude(s) for loop diagram number 91
      CALL MP_FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQQ_1EPS,AMPL(2,156))
      CALL MP_FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQQ_1EPS,AMPL(2,157))
      CALL MP_FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQQ_1EPS,AMPL(2,158))
      CALL MP_FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQQ_1EPS,AMPL(2,159))
      CALL MP_FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQB,AMPL(1,160))
      CALL MP_FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQQ_1EPS,AMPL(2,161))
      CALL MP_FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQT,AMPL(1,162))
      CALL MP_FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQQ_1EPS,AMPL(2,163))
      CALL MP_FFV1_0(W(1,8),W(1,9),W(1,2),UV_GQQG_1EPS,AMPL(2,164))
      CALL MP_FFV1_0(W(1,8),W(1,9),W(1,2),R2_GQQ,AMPL(1,165))
C     Counter-term amplitude(s) for loop diagram number 92
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,8),W(1,18),R2_QQQ,R2_QQT,AMPL(1
     $ ,166))
      CALL MP_R2_QQ_2_0(W(1,8),W(1,18),UV_TMASS,AMPL(1,167))
      CALL MP_R2_QQ_2_0(W(1,8),W(1,18),UV_TMASS_1EPS,AMPL(2,168))
      CALL MP_FFV1_2(W(1,8),W(1,2),GC_5,MDL_MT,MDL_WT,W(1,26))
C     Counter-term amplitude(s) for loop diagram number 93
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,26),W(1,9),R2_QQQ,R2_QQT,AMPL(1
     $ ,169))
      CALL MP_R2_QQ_2_0(W(1,26),W(1,9),UV_TMASS,AMPL(1,170))
      CALL MP_R2_QQ_2_0(W(1,26),W(1,9),UV_TMASS_1EPS,AMPL(2,171))
      CALL MP_FFV2_2(W(1,11),W(1,3),GC_47,MDL_MT,MDL_WT,W(1,27))
C     Counter-term amplitude(s) for loop diagram number 95
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,27),W(1,12),R2_QQQ,R2_QQT,AMPL(1
     $ ,172))
      CALL MP_R2_QQ_2_0(W(1,27),W(1,12),UV_TMASS,AMPL(1,173))
      CALL MP_R2_QQ_2_0(W(1,27),W(1,12),UV_TMASS_1EPS,AMPL(2,174))
C     Counter-term amplitude(s) for loop diagram number 96
      CALL MP_FFV1_0(W(1,27),W(1,4),W(1,2),UV_GQQQ_1EPS,AMPL(2,175))
      CALL MP_FFV1_0(W(1,27),W(1,4),W(1,2),UV_GQQQ_1EPS,AMPL(2,176))
      CALL MP_FFV1_0(W(1,27),W(1,4),W(1,2),UV_GQQQ_1EPS,AMPL(2,177))
      CALL MP_FFV1_0(W(1,27),W(1,4),W(1,2),UV_GQQQ_1EPS,AMPL(2,178))
      CALL MP_FFV1_0(W(1,27),W(1,4),W(1,2),UV_GQQB,AMPL(1,179))
      CALL MP_FFV1_0(W(1,27),W(1,4),W(1,2),UV_GQQQ_1EPS,AMPL(2,180))
      CALL MP_FFV1_0(W(1,27),W(1,4),W(1,2),UV_GQQT,AMPL(1,181))
      CALL MP_FFV1_0(W(1,27),W(1,4),W(1,2),UV_GQQQ_1EPS,AMPL(2,182))
      CALL MP_FFV1_0(W(1,27),W(1,4),W(1,2),UV_GQQG_1EPS,AMPL(2,183))
      CALL MP_FFV1_0(W(1,27),W(1,4),W(1,2),R2_GQQ,AMPL(1,184))
C     Counter-term amplitude(s) for loop diagram number 98
      CALL MP_FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQQ_1EPS,AMPL(2,185))
      CALL MP_FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQQ_1EPS,AMPL(2,186))
      CALL MP_FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQQ_1EPS,AMPL(2,187))
      CALL MP_FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQQ_1EPS,AMPL(2,188))
      CALL MP_FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQB,AMPL(1,189))
      CALL MP_FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQQ_1EPS,AMPL(2,190))
      CALL MP_FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQT,AMPL(1,191))
      CALL MP_FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQQ_1EPS,AMPL(2,192))
      CALL MP_FFV1_0(W(1,8),W(1,12),W(1,1),UV_GQQG_1EPS,AMPL(2,193))
      CALL MP_FFV1_0(W(1,8),W(1,12),W(1,1),R2_GQQ,AMPL(1,194))
C     Counter-term amplitude(s) for loop diagram number 99
      CALL MP_FFV1_0(W(1,25),W(1,4),W(1,1),UV_GQQQ_1EPS,AMPL(2,195))
      CALL MP_FFV1_0(W(1,25),W(1,4),W(1,1),UV_GQQQ_1EPS,AMPL(2,196))
      CALL MP_FFV1_0(W(1,25),W(1,4),W(1,1),UV_GQQQ_1EPS,AMPL(2,197))
      CALL MP_FFV1_0(W(1,25),W(1,4),W(1,1),UV_GQQQ_1EPS,AMPL(2,198))
      CALL MP_FFV1_0(W(1,25),W(1,4),W(1,1),UV_GQQB,AMPL(1,199))
      CALL MP_FFV1_0(W(1,25),W(1,4),W(1,1),UV_GQQQ_1EPS,AMPL(2,200))
      CALL MP_FFV1_0(W(1,25),W(1,4),W(1,1),UV_GQQT,AMPL(1,201))
      CALL MP_FFV1_0(W(1,25),W(1,4),W(1,1),UV_GQQQ_1EPS,AMPL(2,202))
      CALL MP_FFV1_0(W(1,25),W(1,4),W(1,1),UV_GQQG_1EPS,AMPL(2,203))
      CALL MP_FFV1_0(W(1,25),W(1,4),W(1,1),R2_GQQ,AMPL(1,204))
C     Counter-term amplitude(s) for loop diagram number 100
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),R2_3GQ,AMPL(1,205))
C     Counter-term amplitude(s) for loop diagram number 101
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GQ,AMPL(1,206))
C     Counter-term amplitude(s) for loop diagram number 107
      CALL MP_FFV1_0(W(1,26),W(1,4),W(1,1),UV_GQQQ_1EPS,AMPL(2,207))
      CALL MP_FFV1_0(W(1,26),W(1,4),W(1,1),UV_GQQQ_1EPS,AMPL(2,208))
      CALL MP_FFV1_0(W(1,26),W(1,4),W(1,1),UV_GQQQ_1EPS,AMPL(2,209))
      CALL MP_FFV1_0(W(1,26),W(1,4),W(1,1),UV_GQQQ_1EPS,AMPL(2,210))
      CALL MP_FFV1_0(W(1,26),W(1,4),W(1,1),UV_GQQB,AMPL(1,211))
      CALL MP_FFV1_0(W(1,26),W(1,4),W(1,1),UV_GQQQ_1EPS,AMPL(2,212))
      CALL MP_FFV1_0(W(1,26),W(1,4),W(1,1),UV_GQQT,AMPL(1,213))
      CALL MP_FFV1_0(W(1,26),W(1,4),W(1,1),UV_GQQQ_1EPS,AMPL(2,214))
      CALL MP_FFV1_0(W(1,26),W(1,4),W(1,1),UV_GQQG_1EPS,AMPL(2,215))
      CALL MP_FFV1_0(W(1,26),W(1,4),W(1,1),R2_GQQ,AMPL(1,216))
C     Counter-term amplitude(s) for loop diagram number 108
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,8),W(1,22),R2_QQQ,R2_QQT,AMPL(1
     $ ,217))
      CALL MP_R2_QQ_2_0(W(1,8),W(1,22),UV_TMASS,AMPL(1,218))
      CALL MP_R2_QQ_2_0(W(1,8),W(1,22),UV_TMASS_1EPS,AMPL(2,219))
      CALL MP_FFV1_2(W(1,8),W(1,1),GC_5,MDL_MT,MDL_WT,W(1,28))
C     Counter-term amplitude(s) for loop diagram number 109
      CALL MP_R2_QQ_1_R2_QQ_2_0(W(1,28),W(1,12),R2_QQQ,R2_QQT,AMPL(1
     $ ,220))
      CALL MP_R2_QQ_2_0(W(1,28),W(1,12),UV_TMASS,AMPL(1,221))
      CALL MP_R2_QQ_2_0(W(1,28),W(1,12),UV_TMASS_1EPS,AMPL(2,222))
C     Counter-term amplitude(s) for loop diagram number 112
      CALL MP_FFV1_0(W(1,28),W(1,4),W(1,2),UV_GQQQ_1EPS,AMPL(2,223))
      CALL MP_FFV1_0(W(1,28),W(1,4),W(1,2),UV_GQQQ_1EPS,AMPL(2,224))
      CALL MP_FFV1_0(W(1,28),W(1,4),W(1,2),UV_GQQQ_1EPS,AMPL(2,225))
      CALL MP_FFV1_0(W(1,28),W(1,4),W(1,2),UV_GQQQ_1EPS,AMPL(2,226))
      CALL MP_FFV1_0(W(1,28),W(1,4),W(1,2),UV_GQQB,AMPL(1,227))
      CALL MP_FFV1_0(W(1,28),W(1,4),W(1,2),UV_GQQQ_1EPS,AMPL(2,228))
      CALL MP_FFV1_0(W(1,28),W(1,4),W(1,2),UV_GQQT,AMPL(1,229))
      CALL MP_FFV1_0(W(1,28),W(1,4),W(1,2),UV_GQQQ_1EPS,AMPL(2,230))
      CALL MP_FFV1_0(W(1,28),W(1,4),W(1,2),UV_GQQG_1EPS,AMPL(2,231))
      CALL MP_FFV1_0(W(1,28),W(1,4),W(1,2),R2_GQQ,AMPL(1,232))
C     Counter-term amplitude(s) for loop diagram number 119
      CALL MP_R2_GG_1_R2_GG_2_0(W(1,6),W(1,13),R2_GGG_1,R2_GGG_2
     $ ,AMPL(1,233))
C     Counter-term amplitude(s) for loop diagram number 120
      CALL MP_R2_GG_1_R2_GG_2_0(W(1,6),W(1,14),R2_GGG_1,R2_GGG_2
     $ ,AMPL(1,234))
C     Counter-term amplitude(s) for loop diagram number 121
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,13),R2_3GG,AMPL(1,235))
C     Counter-term amplitude(s) for loop diagram number 122
      CALL MP_VVV1_0(W(1,1),W(1,2),W(1,14),R2_3GG,AMPL(1,236))

      GOTO 1001
 2000 CONTINUE
      MP_CT_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

