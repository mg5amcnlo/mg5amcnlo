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
      INTEGER I,J,K,LUN,IDUM,ICONF,IMIRROR,JC(NEXTERNAL)
      DATA IDUM/0/
      LOGICAL FIRST_TIME
      DATA FIRST_TIME/.TRUE./
      INTEGER NPROC,LIMEVTS
      SAVE FIRST_TIME,NPROC,IDUM
      INTEGER SYMCONF(0:LMAXCONFIGS)
      SAVE SYMCONF
    DOUBLE PRECISION SUMPROB,TOTWGT,R,XDUM
      DOUBLE PRECISION P1(0:3,NEXTERNAL)
      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
      INCLUDE 'config_subproc_map.inc'            
      INTEGER PERMS(NEXTERNAL,LMAXCONFIGS)
      INCLUDE 'symperms.inc'
      LOGICAL MIRRORPROCS(LMAXCONFIGS)
      INCLUDE 'mirrorprocs.inc'
C   SELPROC is vector of selection weights for the subprocesses
C   SUMWGT is vector of total weight for the subprocesses
C   NUMEVTS is vector of event calls for the subprocesses
      DOUBLE PRECISION SELPROC(2, MAXSPROC,LMAXCONFIGS)
      DOUBLE PRECISION SUMWGT(2, MAXSPROC,LMAXCONFIGS)
      INTEGER NUMEVTS(2, MAXSPROC,LMAXCONFIGS)
      INTEGER LARGEDIM
      PARAMETER (LARGEDIM=2*MAXSPROC*LMAXCONFIGS)
      DATA SELPROC/LARGEDIM*0d0/
      DATA SUMWGT/LARGEDIM*0d0/
      DATA NUMEVTS/LARGEDIM*0/
      SAVE SELPROC,SUMWGT,NUMEVTS
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
C   IB gives which beam is which (for mirror processes)
      INTEGER SUBDIAG(MAXSPROC),IB(2)
      COMMON/TO_SUB_DIAG/SUBDIAG,IB
C   ICONFIG has this config number
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
C     IPROC has the present process number
      INTEGER IPROC
      COMMON/TO_IPROC/IPROC
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
          WRITE(*,'(I4,a,%(nexternal)dI3,a)') SYMCONF(J),'  (',(PERMS(I,SYMCONF(J)),I=1,NEXTERNAL),')'
        ENDDO
        GOTO 20
 10     WRITE(*,*)'Error opening symfact.dat. No permutations will be used.'
C     Read in weight file
 20     LUN=NEXTUNOPEN()
        OPEN(UNIT=LUN,FILE='selproc.dat',STATUS='OLD',ERR=30)
        DO J=1,SYMCONF(0)
          READ(LUN,'(%(maxsprocs)dE16.8)') ((SELPROC(K,I,J),K=1,2),I=1,MAXSPROC)
        ENDDO
        CLOSE(LUN)
        GOTO 40
 30     WRITE(*,*)'Error opening file selproc.dat. Set all weights equal.'
C     Find number of contributing diagrams
        NPROC=0
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            IF(CONFSUB(I,SYMCONF(J)).GT.0) THEN 
              NPROC=NPROC+1
              IF(MIRRORPROCS(I)) NPROC=NPROC+1
            ENDIF
          ENDDO
        ENDDO
C     Set SELPROC democratically
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            IF(CONFSUB(I,SYMCONF(J)).NE.0) THEN
              SELPROC(1,I,J)=1d0/NPROC
              IF(MIRRORPROCS(I)) SELPROC(2,I,J)=1d0/NPROC
            ENDIF
          ENDDO
        ENDDO
 40     WRITE(*,*) 'Initial selection weights:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(%(maxsprocs)dE12.4)')((SELPROC(K,I,J),K=1,2),I=1,MAXSPROC)
        ENDDO
        RETURN
      ELSE IF(IMODE.EQ.2)THEN
C     Reweight PROCSEL according to the actual weigths
        SUMPROB=0D0
        TOTWGT=0D0
c     Take into account only channels with at least LIMEVTS events
        LIMEVTS=300
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            DO K=1,2
              IF(NUMEVTS(K,I,J).GE.LIMEVTS)THEN
                TOTWGT=TOTWGT+SUMWGT(K,I,J)
                SUMPROB=SUMPROB+SELPROC(K,I,J)
              ENDIF
            ENDDO
          ENDDO
        ENDDO
c     Update SELPROC
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            DO K=1,2
              IF(NUMEVTS(K,I,J).GE.LIMEVTS)THEN
                SELPROC(K,I,J)=SUMWGT(K,I,J)/TOTWGT*SUMPROB
              ENDIF
            ENDDO
          ENDDO
        ENDDO
        WRITE(*,*)'Selection weights after reweight:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(%(maxsprocs)dE12.4)')((SELPROC(K,I,J),K=1,2),I=1,MAXSPROC)
        ENDDO
        WRITE(*,*)'Summed weights:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(%(maxsprocs)dE12.4)')((SUMWGT(K,I,J),K=1,2),I=1,MAXSPROC)
        ENDDO
        WRITE(*,*)'Events:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(%(maxsprocs)dI12)')((NUMEVTS(K,I,J),K=1,2),I=1,MAXSPROC)
        ENDDO
C       Reset weights and number of events if above LIMEVTS
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            DO K=1,2
              IF(NUMEVTS(K,I,J).GE.LIMEVTS)THEN
                NUMEVTS(K,I,J)=0
                SUMWGT(K,I,J)=0d0
              ENDIF
            ENDDO
          ENDDO
        ENDDO
        RETURN
      ELSE IF(IMODE.EQ.3)THEN
C     Write out weight file
        LUN=NEXTUNOPEN()
        OPEN(UNIT=LUN,FILE='selproc.dat',STATUS='UNKNOWN')
        DO J=1,SYMCONF(0)
          WRITE(LUN,'(%(maxsprocs)dE16.8)') ((SELPROC(K,I,J),K=1,2),I=1,MAXSPROC)
        ENDDO
        CLOSE(LUN)
        RETURN
      ENDIF

C   IMODE.EQ.0, regular run mode

C   Select among the subprocesses based on SELPROC
      IDUM=0
      R=XRAN1(IDUM)
      ICONF=0
      IPROC=0
      SUMPROB=0D0
      DO J=1,SYMCONF(0)
        DO I=1,MAXSPROC
          DO K=1,2
            SUMPROB=SUMPROB+SELPROC(K,I,J)
            IF(R.LT.SUMPROB)THEN
              IPROC=I
              ICONF=J
              IMIRROR=K
              GOTO 50
            ENDIF
          ENDDO
        ENDDO
      ENDDO
 50   CONTINUE

      IF(IPROC.EQ.0) RETURN

C   Set SUBDIAG and ICONFIG
      ICONFIG=SYMCONF(ICONF)
      DO I=1,MAXSPROC
        SUBDIAG(I) = CONFSUB(I,SYMCONF(ICONF))
      ENDDO

C   Set momenta according to this permutation
      CALL SWITCHMOM(PP,P1,PERMS(1,ICONFIG),JC,NEXTERNAL)

      IB(1)=1
      IB(2)=2

      IF(IMIRROR.EQ.2)THEN
C     Flip momenta (rotate around x axis)
        DO I=1,NEXTERNAL
          P1(2,I)=-P1(2,I)
          P1(3,I)=-P1(3,I)
        ENDDO
C     Flip beam identity
        IB(1)=2
        IB(2)=1
C     Flip x values (to get boost right)
        XDUM=XBK(1)
        XBK(1)=XBK(2)
        XBK(2)=XDUM
      ENDIF


      IF (PASSCUTS(P1)) THEN
C     Update weigth w.r.t SELPROC
        WGT=WGT/SELPROC(IMIRROR,IPROC,ICONF)

%(call_dsig_proc_lines)s

C     Update summed weight and number of events
        SUMWGT(IMIRROR,IPROC,ICONF)=SUMWGT(IMIRROR,IPROC,ICONF)+DSIG*WGT
        NUMEVTS(IMIRROR,IPROC,ICONF)=NUMEVTS(IMIRROR,IPROC,ICONF)+1
        

      ENDIF
      RETURN
      END
