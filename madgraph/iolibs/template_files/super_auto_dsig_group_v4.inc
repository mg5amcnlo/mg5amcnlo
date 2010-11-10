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
      INTEGER I,J,LUN,IDUM,IPROC,ICONF,JC(NEXTERNAL)
      DATA IDUM/0/
      LOGICAL FIRST_TIME
      DATA FIRST_TIME/.TRUE./
      INTEGER NPROC,LIMEVTS
      SAVE FIRST_TIME,NPROC,IDUM
      INTEGER SYMCONF(0:LMAXCONFIGS)
      SAVE SYMCONF
      DOUBLE PRECISION SUMPROB,TOTWGT,R
      DOUBLE PRECISION P1(0:3,NEXTERNAL)
      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
      INCLUDE 'config_subproc_map.inc'            
      INTEGER PERMS(NEXTERNAL,LMAXCONFIGS)
      INCLUDE 'symperms.inc'
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
      DOUBLE PRECISION SELPROC(MAXSPROC,LMAXCONFIGS)
      DOUBLE PRECISION SUMWGT(MAXSPROC,LMAXCONFIGS)
      INTEGER NUMEVTS(MAXSPROC,LMAXCONFIGS)
      COMMON/TO_SELPROC/SELPROC,SUMWGT,NUMEVTS
      INTEGER LARGEDIM
      PARAMETER (LARGEDIM=MAXSPROC*LMAXCONFIGS)
      DATA SELPROC/LARGEDIM*0d0/
      DATA SUMWGT/LARGEDIM*0d0/
      DATA NUMEVTS/LARGEDIM*0/
C ----------
C BEGIN CODE
C ----------
      DSIG=0D0

      IF(IMODE.EQ.1)THEN
C     Read in symfact file and store used permutations in SYMCONF
        LUN=NEXTUNOPEN()
        OPEN(UNIT=LUN,FILE='../symfact.dat',STATUS='OLD',ERR=10)
        IPROC=1
        SYMCONF(IPROC)=ICONFIG
        DO I=1,MAPCONFIG(0)
          READ(LUN,*) IDUM, ICONF
          IF(ICONF.EQ.-ICONFIG)THEN
            IPROC=IPROC+1
            SYMCONF(IPROC)=I
          ENDIF
        ENDDO
        SYMCONF(0)=IPROC
        CLOSE(LUN)
        WRITE(*,*)'Using configs with permutations:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(I4,a,%(nexternal)dI3,a)') SYMCONF(J),'  (',(PERMS(I,J),I=1,NEXTERNAL),')'
        ENDDO
        GOTO 20
 10     WRITE(*,*)'Error opening symfact.dat. No permutations will be used.'
C     Read in weight file
 20     LUN=NEXTUNOPEN()
        OPEN(UNIT=LUN,FILE='selproc.dat',STATUS='OLD',ERR=30)
        DO J=1,SYMCONF(0)
          READ(LUN,*) (SELPROC(I,J),I=1,MAXSPROC)
        ENDDO
        CLOSE(LUN)
        RETURN
 30     WRITE(*,*)'Error opening file selproc.dat. Set all weights equal.'
C     Find number of contributing diagrams
        NPROC=0
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            IF(CONFSUB(I,SYMCONF(J)).GT.0) NPROC=NPROC+1
          ENDDO
        ENDDO
C     Set SELPROC democratically
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            IF(CONFSUB(I,SYMCONF(J)).NE.0) SELPROC(I,J)=1d0/NPROC
          ENDDO
        ENDDO

      ELSE IF(IMODE.EQ.2)THEN
C     Reweight PROCSEL according to the actual weigths
        SUMPROB=0D0
        TOTWGT=0D0
c     Take into account only channels with at least LIMEVTS events
        LIMEVTS=300
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            IF(NUMEVTS(I,J).GE.LIMEVTS)THEN
              TOTWGT=TOTWGT+SUMWGT(I,J)
              SUMPROB=SUMPROB+SELPROC(I,J)
            ENDIF
          ENDDO
        ENDDO
c     Update SELPROC
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            IF(NUMEVTS(I,J).GE.LIMEVTS)THEN
              SELPROC(I,J)=SUMWGT(I,J)/TOTWGT*SUMPROB
            ENDIF
          ENDDO
        ENDDO
        WRITE(*,*)'Selection weights after reweight:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(100E12.4)')(SELPROC(I,J),I=1,MAXSPROC)
        ENDDO
        WRITE(*,*)'Summed weights:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(100E12.4)')(SUMWGT(I,J),I=1,MAXSPROC)
        ENDDO
        WRITE(*,*)'Events:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(100I12)')(NUMEVTS(I,J),I=1,MAXSPROC)
        ENDDO
        

      ELSE IF(IMODE.EQ.3)THEN
C     Write out weight file
        LUN=NEXTUNOPEN()
        OPEN(UNIT=LUN,FILE='selproc.dat',STATUS='UNKNOWN')
        DO J=1,SYMCONF(0)
          WRITE(LUN,*) (SELPROC(I,J),I=1,MAXSPROC)
        ENDDO
        CLOSE(LUN)
        RETURN
      ENDIF

      IF (PASSCUTS(PP)) THEN
C     Select among the subprocesses based on SELPROC

        R=XRAN1(IDUM)
        ICONF=0
        IPROC=0
        SUMPROB=0D0
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            SUMPROB=SUMPROB+SELPROC(I,J)
            IF(R.LT.SUMPROB)THEN
              IPROC=I
              ICONF=J
              GOTO 40
            ENDIF
          ENDDO
        ENDDO
 40     CONTINUE

        IF(IPROC.EQ.0) RETURN

C     Set SUBDIAG and ICONFIG
        ICONFIG=SYMCONF(ICONF)
        DO I=1,MAXSPROC
          SUBDIAG(I) = CONFSUB(I,SYMCONF(ICONF))
        ENDDO

C     Set momenta according to this permutation
        CALL SWITCHMOM(PP,P1,PERMS(1,ICONFIG),JC,NEXTERNAL)

C     Update weigth w.r.t SELPROC
        WGT=WGT/SELPROC(IPROC,ICONF)

%(call_dsig_proc_lines)s

        SUMWGT(IPROC,ICONF)=SUMWGT(IPROC,ICONF)+DSIG*WGT
        NUMEVTS(IPROC,ICONF)=NUMEVTS(IPROC,ICONF)+1
        
      ENDIF
      RETURN
      END
