DOUBLE PRECISION FUNCTION DSIG(PP,WGT,IMODE)
C ****************************************************
C
%(info_lines)s
C
%(process_lines)s
C
C     RETURNS DIFFERENTIAL CROSS SECTION 
c     FOR MULTIPLE PROCESSES IN PROCESS GROUP
C     Input:
C             pp    4 momentum of external particles
C             wgt   weight from Monte Carlo
C             imode 0 run, 1 init, 2 reweight, 3 finalize
C     Output:
C             Amplitude squared and summed
C ****************************************************
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include 'genps.inc'
      include 'nexternal.inc'
      include 'maxamps.inc'
      REAL*8     PI
      PARAMETER (PI=3.1415926d0)
C  
C ARGUMENTS 
C  
      DOUBLE PRECISION PP(0:3,NEXTERNAL), WGT
      INTEGER IMODE
C   
C LOCAL VARIABLES 
C  
      INTEGER I,J,LUN,IDUM,IPROC
      DATA IDUM/0/
      LOGICAL FIRST_TIME
      DATA FIRST_TIME/.TRUE./
      INTEGER NPROC,LIMEVTS
      SAVE FIRST_TIME,NPROC,IDUM
      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
      include 'config_subproc_map.inc'            
      DOUBLE PRECISION SUMPROB,TOTWGT,R
C  
C EXTERNAL FUNCTIONS
C  
      LOGICAL PASSCUTS
      INTEGER NEXTUNOPEN
      REAL XRAN1
      EXTERNAL PASSCUTS,NEXTUNOPEN,XRAN1
%(dsig_def_line)s
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
      include 'run.inc'
C   SUBDIAG is vector of diagram numbers for this config
      INTEGER SUBDIAG(MAXSPROC)
      COMMON/TO_SUB_DIAG/SUBDIAG
C   ICONFIG has this config number
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
C   SELPROC is vector of selection weights for the subprocesses
C   SUMWGT is vector of total weight for the subprocesses
C   NUMEVTS is vector of event calls for the subprocesses
      DOUBLE PRECISION SELPROC(MAXSPROC)
      DOUBLE PRECISION SUMWGT(MAXSPROC)
      INTEGER NUMEVTS(MAXSPROC)
      COMMON/TO_SELPROC/SELPROC,SUMWGT,NUMEVTS
      DATA SELPROC/MAXSPROC*0d0/
      DATA SUMWGT/MAXSPROC*0d0/
      DATA NUMEVTS/MAXSPROC*0/
C ----------
C BEGIN CODE
C ----------
      DSIG=0D0

      IF(FIRST_TIME)THEN
C     Set SUBDIAG for this config
        NPROC=0
        DO I=1,MAXSPROC
          SUBDIAG(I) = CONFSUB(I,ICONFIG)
          IF(SUBDIAG(I).NE.0) NPROC=NPROC+1
        ENDDO
        write(*,*)'Subproc diagrams:',(SUBDIAG(I),I=1,MAXSPROC)
        FIRST_TIME=.FALSE.
      ENDIF

      IF(IMODE.EQ.1)THEN
C     Read in weight file
        LUN=NEXTUNOPEN()
        OPEN(UNIT=LUN,FILE='selproc.dat',STATUS='OLD',ERR=20)        
        READ(LUN,*) (SELPROC(I),I=1,MAXSPROC)
        CLOSE(LUN)
        RETURN
 20     WRITE(*,*)'Error opening file selproc.dat. Set all weights equal.'
        DO I=1,MAXSPROC
          IF(SUBDIAG(I).NE.0) SELPROC(I)=1d0/NPROC
        ENDDO
      ELSE IF(IMODE.EQ.2)THEN
C     Reweight PROCSEL according to the actual weigths
        SUMPROB=0D0
        TOTWGT=0D0
c     Take into account only channels with at least LIMEVTS events
        LIMEVTS=100
        DO I=1,MAXSPROC
          IF(NUMEVTS(I).GE.LIMEVTS)THEN
            TOTWGT=TOTWGT+SUMWGT(I)
            SUMPROB=SUMPROB+SELPROC(I)
          ENDIF
        ENDDO
c     Update SELPROC
        DO I=1,MAXSPROC
          IF(NUMEVTS(I).GE.LIMEVTS)THEN
            SELPROC(I)=SUMWGT(I)/TOTWGT*SUMPROB
          ENDIF
        ENDDO
        WRITE(*,*)'Selection weights after reweight:',(SELPROC(I),I=1,MAXSPROC)
        WRITE(*,*)'Summed weights:',(SUMWGT(I),I=1,MAXSPROC)
        WRITE(*,*)'Events:',(NUMEVTS(I),I=1,MAXSPROC)

      ELSE IF(IMODE.EQ.3)THEN
C     Write out weight file
        LUN=NEXTUNOPEN()
        OPEN(UNIT=LUN,FILE='selproc.dat',STATUS='UNKNOWN')
        WRITE(LUN,*) (SELPROC(I),I=1,MAXSPROC)
        CLOSE(LUN)
        RETURN
      ENDIF

      IF (PASSCUTS(PP)) THEN
C     Select among the subprocesses based on SELPROC

        R=XRAN1(IDUM)
        IPROC=0
        SUMPROB=0D0
        DO I=1,MAXSPROC
          SUMPROB=SUMPROB+SELPROC(I)
          IF(R.LT.SUMPROB)THEN
            IPROC=I
            GOTO 10
          ENDIF
        ENDDO
 10     CONTINUE

        IF(IPROC.EQ.0) RETURN

C     Update weigth w.r.t SELPROC
        WGT=WGT/SELPROC(IPROC)

%(call_dsig_proc_lines)s

        SUMWGT(IPROC)=SUMWGT(IPROC)+DSIG*WGT
        NUMEVTS(IPROC)=NUMEVTS(IPROC)+1
        
      ENDIF
      RETURN
      END
