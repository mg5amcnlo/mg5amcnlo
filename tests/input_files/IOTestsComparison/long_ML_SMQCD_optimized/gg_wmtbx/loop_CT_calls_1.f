      SUBROUTINE ML5_0_LOOP_CT_CALLS_1(P,NHEL,H,IC)
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

      COMPLEX*16 WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE
     $ ,0:NLOOPWAVEFUNCS)
      COMPLEX*16 PL(0:3,0:NLOOPWAVEFUNCS)
      COMMON/ML5_0_WL/WL,PL

      COMPLEX*16 AMPL(3,NCTAMPS)
      COMMON/ML5_0_AMPL/AMPL

C     
C     ----------
C     BEGIN CODE
C     ----------

C     The target squared split order contribution is already reached
C      if true.
      IF (FILTER_SO.AND.CTCALL_REQ_SO_DONE) THEN
        GOTO 1001
      ENDIF

C     CutTools call for loop numbers 1,2,117,118,139,140,133,134,135,13
C     6,137,138
      CALL ML5_0_LOOP_2(6,13,DCMPLX(ZERO),DCMPLX(ZERO),2,I_SO,1)
C     CutTools call for loop numbers 3,4,5,6,119,120,141,142,143,144
      CALL ML5_0_LOOP_3(1,2,13,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,3,I_SO,2)
C     CutTools call for loop numbers 7,29,55
      CALL ML5_0_LOOP_2(7,15,DCMPLX(ZERO),DCMPLX(MDL_MB),1,I_SO,3)
C     CutTools call for loop numbers 8,11
      CALL ML5_0_LOOP_2(6,13,DCMPLX(MDL_MB),DCMPLX(MDL_MB),2,I_SO,4)
C     CutTools call for loop numbers 9
      CALL ML5_0_LOOP_3(5,6,7,DCMPLX(MDL_MB),DCMPLX(MDL_MB)
     $ ,DCMPLX(ZERO),2,I_SO,5)
C     CutTools call for loop numbers 10,69,70,71
      CALL ML5_0_LOOP_3(5,6,7,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(MDL_MB)
     $ ,2,I_SO,6)
C     CutTools call for loop numbers 12,32,58
      CALL ML5_0_LOOP_3(3,4,15,DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MB),2,I_SO,7)
C     CutTools call for loop numbers 13
      CALL ML5_0_LOOP_4(3,4,5,6,DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MB),DCMPLX(MDL_MB),3,I_SO,8)
C     CutTools call for loop numbers 14
      CALL ML5_0_LOOP_4(3,5,4,6,DCMPLX(MDL_MB),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MT),3,I_SO,9)
C     CutTools call for loop numbers 15,21,51
      CALL ML5_0_LOOP_3(3,5,16,DCMPLX(MDL_MB),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),2,I_SO,10)
C     CutTools call for loop numbers 16,78,79,80
      CALL ML5_0_LOOP_4(3,4,6,5,DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MB),3,I_SO,11)
C     CutTools call for loop numbers 17,56
      CALL ML5_0_LOOP_2(10,17,DCMPLX(ZERO),DCMPLX(MDL_MB),1,I_SO,12)
C     CutTools call for loop numbers 18
      CALL ML5_0_LOOP_3(3,9,10,DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MB),2,I_SO,13)
C     CutTools call for loop numbers 19,61
      CALL ML5_0_LOOP_3(2,5,17,DCMPLX(MDL_MB),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MB),2,I_SO,14)
C     CutTools call for loop numbers 20
      CALL ML5_0_LOOP_4(2,3,9,5,DCMPLX(MDL_MB),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MB),3,I_SO,15)
C     CutTools call for loop numbers 22
      CALL ML5_0_LOOP_4(2,5,3,9,DCMPLX(ZERO),DCMPLX(MDL_MB)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),3,I_SO,16)
C     CutTools call for loop numbers 23
      CALL ML5_0_LOOP_4(2,3,5,9,DCMPLX(MDL_MT),DCMPLX(MDL_MB)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MT),3,I_SO,17)
C     CutTools call for loop numbers 24,65
      CALL ML5_0_LOOP_3(2,5,17,DCMPLX(ZERO),DCMPLX(MDL_MB),DCMPLX(ZERO)
     $ ,2,I_SO,18)
C     CutTools call for loop numbers 25,30
      CALL ML5_0_LOOP_2(11,19,DCMPLX(ZERO),DCMPLX(MDL_MB),1,I_SO,19)
C     CutTools call for loop numbers 26
      CALL ML5_0_LOOP_3(3,12,11,DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MB),2,I_SO,20)
C     CutTools call for loop numbers 27
      CALL ML5_0_LOOP_3(2,7,11,DCMPLX(MDL_MB),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MB),2,I_SO,21)
C     CutTools call for loop numbers 28
      CALL ML5_0_LOOP_4(2,3,4,11,DCMPLX(MDL_MB),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MB),3,I_SO,22)
C     CutTools call for loop numbers 31
      CALL ML5_0_LOOP_3(2,7,11,DCMPLX(ZERO),DCMPLX(MDL_MB),DCMPLX(ZERO)
     $ ,2,I_SO,23)
C     CutTools call for loop numbers 33
      CALL ML5_0_LOOP_4(2,3,11,4,DCMPLX(MDL_MT),DCMPLX(MDL_MB)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MT),3,I_SO,24)
C     CutTools call for loop numbers 34
      CALL ML5_0_LOOP_4(2,4,3,11,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MB),DCMPLX(ZERO),3,I_SO,25)
C     CutTools call for loop numbers 35,45
      CALL ML5_0_LOOP_3(1,5,19,DCMPLX(MDL_MB),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MB),2,I_SO,26)
C     CutTools call for loop numbers 36
      CALL ML5_0_LOOP_4(1,3,12,5,DCMPLX(MDL_MB),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MB),3,I_SO,27)
C     CutTools call for loop numbers 37
      CALL ML5_0_LOOP_3(1,7,10,DCMPLX(MDL_MB),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MB),2,I_SO,28)
C     CutTools call for loop numbers 38
      CALL ML5_0_LOOP_4(1,3,4,10,DCMPLX(MDL_MB),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MB),3,I_SO,29)
C     CutTools call for loop numbers 39
      CALL ML5_0_LOOP_5(1,2,3,4,5,DCMPLX(MDL_MB),DCMPLX(MDL_MB)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),DCMPLX(MDL_MB),4,I_SO,30)
C     CutTools call for loop numbers 40,42,43,47
      CALL ML5_0_LOOP_3(1,2,13,DCMPLX(MDL_MB),DCMPLX(MDL_MB)
     $ ,DCMPLX(MDL_MB),3,I_SO,31)
C     CutTools call for loop numbers 41
      CALL ML5_0_LOOP_4(1,2,7,5,DCMPLX(MDL_MB),DCMPLX(MDL_MB)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MB),3,I_SO,32)
C     CutTools call for loop numbers 44
      CALL ML5_0_LOOP_4(1,2,5,7,DCMPLX(MDL_MB),DCMPLX(MDL_MB)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MB),3,I_SO,33)
C     CutTools call for loop numbers 46
      CALL ML5_0_LOOP_4(1,5,2,7,DCMPLX(MDL_MB),DCMPLX(ZERO)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MB),3,I_SO,34)
C     CutTools call for loop numbers 48
      CALL ML5_0_LOOP_5(1,2,5,4,3,DCMPLX(MDL_MB),DCMPLX(MDL_MB)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MT),DCMPLX(MDL_MB),4,I_SO,35)
C     CutTools call for loop numbers 49
      CALL ML5_0_LOOP_5(1,3,2,4,5,DCMPLX(MDL_MB),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),DCMPLX(MDL_MB),4,I_SO,36)
C     CutTools call for loop numbers 50
      CALL ML5_0_LOOP_5(1,3,4,2,5,DCMPLX(MDL_MB),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(MDL_MB),4,I_SO,37)
C     CutTools call for loop numbers 52
      CALL ML5_0_LOOP_4(1,3,5,12,DCMPLX(MDL_MT),DCMPLX(MDL_MB)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MT),3,I_SO,38)
C     CutTools call for loop numbers 53
      CALL ML5_0_LOOP_4(1,5,3,12,DCMPLX(ZERO),DCMPLX(MDL_MB)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),3,I_SO,39)
C     CutTools call for loop numbers 54,66
      CALL ML5_0_LOOP_3(1,5,19,DCMPLX(ZERO),DCMPLX(MDL_MB),DCMPLX(ZERO)
     $ ,2,I_SO,40)
C     CutTools call for loop numbers 57
      CALL ML5_0_LOOP_3(1,7,10,DCMPLX(ZERO),DCMPLX(MDL_MB),DCMPLX(ZERO)
     $ ,2,I_SO,41)
C     CutTools call for loop numbers 59
      CALL ML5_0_LOOP_4(1,3,10,4,DCMPLX(MDL_MT),DCMPLX(MDL_MB)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MT),3,I_SO,42)
C     CutTools call for loop numbers 60
      CALL ML5_0_LOOP_4(1,4,3,10,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MB),DCMPLX(ZERO),3,I_SO,43)
C     CutTools call for loop numbers 62
      CALL ML5_0_LOOP_4(1,5,2,7,DCMPLX(ZERO),DCMPLX(MDL_MB)
     $ ,DCMPLX(MDL_MB),DCMPLX(ZERO),3,I_SO,44)
C     CutTools call for loop numbers 63
      CALL ML5_0_LOOP_5(1,3,2,5,4,DCMPLX(MDL_MT),DCMPLX(MDL_MB)
     $ ,DCMPLX(MDL_MB),DCMPLX(ZERO),DCMPLX(MDL_MT),4,I_SO,45)
C     CutTools call for loop numbers 64
      CALL ML5_0_LOOP_5(1,4,3,2,5,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MB),DCMPLX(MDL_MB),DCMPLX(ZERO),4,I_SO,46)
C     CutTools call for loop numbers 67
      CALL ML5_0_LOOP_4(1,2,7,5,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MB),DCMPLX(ZERO),3,I_SO,47)
C     CutTools call for loop numbers 68
      CALL ML5_0_LOOP_4(1,2,5,7,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MB),DCMPLX(ZERO),3,I_SO,48)
C     CutTools call for loop numbers 72
      CALL ML5_0_LOOP_5(1,3,5,2,4,DCMPLX(MDL_MT),DCMPLX(MDL_MB)
     $ ,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(MDL_MT),4,I_SO,49)
C     CutTools call for loop numbers 73
      CALL ML5_0_LOOP_5(1,2,4,5,3,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MB),DCMPLX(MDL_MT),4,I_SO,50)
C     CutTools call for loop numbers 74
      CALL ML5_0_LOOP_5(1,4,2,3,5,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MB),DCMPLX(ZERO),4,I_SO,51)
C     CutTools call for loop numbers 75
      CALL ML5_0_LOOP_5(1,2,4,3,5,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(MDL_MB),DCMPLX(ZERO),4,I_SO,52)
C     CutTools call for loop numbers 76
      CALL ML5_0_LOOP_5(1,2,3,5,4,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MB),DCMPLX(ZERO),DCMPLX(MDL_MT),4,I_SO,53)
C     CutTools call for loop numbers 77
      CALL ML5_0_LOOP_5(1,2,5,3,4,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MB),DCMPLX(MDL_MT),DCMPLX(ZERO),4,I_SO,54)
C     CutTools call for loop numbers 81,83
      CALL ML5_0_LOOP_2(6,13,DCMPLX(MDL_MT),DCMPLX(MDL_MT),2,I_SO,55)
C     CutTools call for loop numbers 82,88,104
      CALL ML5_0_LOOP_2(8,16,DCMPLX(ZERO),DCMPLX(MDL_MT),1,I_SO,56)
C     CutTools call for loop numbers 84
      CALL ML5_0_LOOP_3(4,6,8,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),2,I_SO,57)
C     CutTools call for loop numbers 85,114,115,116
      CALL ML5_0_LOOP_3(4,6,8,DCMPLX(ZERO),DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,2,I_SO,58)
C     CutTools call for loop numbers 86,89
      CALL ML5_0_LOOP_2(9,25,DCMPLX(ZERO),DCMPLX(MDL_MT),1,I_SO,59)
C     CutTools call for loop numbers 87
      CALL ML5_0_LOOP_3(2,9,8,DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),2,I_SO,60)
C     CutTools call for loop numbers 90
      CALL ML5_0_LOOP_3(2,9,8,DCMPLX(ZERO),DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,2,I_SO,61)
C     CutTools call for loop numbers 91,105
      CALL ML5_0_LOOP_2(12,27,DCMPLX(ZERO),DCMPLX(MDL_MT),1,I_SO,62)
C     CutTools call for loop numbers 92,108
      CALL ML5_0_LOOP_3(2,4,27,DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),2,I_SO,63)
C     CutTools call for loop numbers 93,110
      CALL ML5_0_LOOP_3(2,4,27,DCMPLX(ZERO),DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,2,I_SO,64)
C     CutTools call for loop numbers 94
      CALL ML5_0_LOOP_3(1,12,8,DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),2,I_SO,65)
C     CutTools call for loop numbers 95,103
      CALL ML5_0_LOOP_3(1,4,25,DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),2,I_SO,66)
C     CutTools call for loop numbers 96,97,99,100
      CALL ML5_0_LOOP_3(1,2,13,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),3,I_SO,67)
C     CutTools call for loop numbers 98
      CALL ML5_0_LOOP_4(1,2,4,8,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MT),3,I_SO,68)
C     CutTools call for loop numbers 101
      CALL ML5_0_LOOP_4(1,2,8,4,DCMPLX(MDL_MT),DCMPLX(MDL_MT)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MT),3,I_SO,69)
C     CutTools call for loop numbers 102
      CALL ML5_0_LOOP_4(1,4,2,8,DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,DCMPLX(ZERO),DCMPLX(MDL_MT),3,I_SO,70)
C     CutTools call for loop numbers 106
      CALL ML5_0_LOOP_3(1,12,8,DCMPLX(ZERO),DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,2,I_SO,71)
C     CutTools call for loop numbers 107,111
      CALL ML5_0_LOOP_3(1,4,25,DCMPLX(ZERO),DCMPLX(MDL_MT),DCMPLX(ZERO)
     $ ,2,I_SO,72)
C     CutTools call for loop numbers 109
      CALL ML5_0_LOOP_4(1,4,2,8,DCMPLX(ZERO),DCMPLX(MDL_MT)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),3,I_SO,73)
C     CutTools call for loop numbers 112
      CALL ML5_0_LOOP_4(1,2,8,4,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),3,I_SO,74)
C     CutTools call for loop numbers 113
      CALL ML5_0_LOOP_4(1,2,4,8,DCMPLX(ZERO),DCMPLX(ZERO)
     $ ,DCMPLX(MDL_MT),DCMPLX(ZERO),3,I_SO,75)
C     CutTools call for loop numbers 121,122,123,124,125,126
      CALL ML5_0_LOOP_2_3(1,2,1,13,2,DCMPLX(ZERO),DCMPLX(ZERO),1,I_SO
     $ ,76)
C     CutTools call for loop numbers 127,128,129,130,131,132
      CALL ML5_0_LOOP_2_3(1,2,2,13,1,DCMPLX(ZERO),DCMPLX(ZERO),1,I_SO
     $ ,77)

      GOTO 1001
 5000 CONTINUE
      CTCALL_REQ_SO_DONE=.TRUE.
 1001 CONTINUE
      END

