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
C             imode 0 run, 1 init, 2 reweight,
C                   3 finalize, 4 only PDFs
C     Output:
C             Amplitude squared and summed
C ****************************************************
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include 'genps.inc'
      include 'maxconfigs.inc'
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
      INTEGER I,J,K,LUN,ICONF,IMIRROR,NPROC
      SAVE NPROC
      INTEGER SYMCONF(0:LMAXCONFIGS)
      SAVE SYMCONF
      DOUBLE PRECISION SUMPROB,TOTWGT,R,XDUM
      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
      INCLUDE 'config_subproc_map.inc'            
      INTEGER PERMS(NEXTERNAL,LMAXCONFIGS)
      INCLUDE 'symperms.inc'
      LOGICAL MIRRORPROCS(MAXSPROC)
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
      INTEGER NEXTUNOPEN
      DOUBLE PRECISION DSIGPROC
      EXTERNAL NEXTUNOPEN,DSIGPROC
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
      include 'run.inc'
C   ICONFIG has this config number
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
C     IPROC has the present process number
      INTEGER IPROC
      COMMON/TO_MIRROR/IMIRROR, IPROC
C     CM_RAP has parton-parton system rapidity
      DOUBLE PRECISION CM_RAP
      LOGICAL SET_CM_RAP
      COMMON/TO_CM_RAP/SET_CM_RAP,CM_RAP
C     Keep track of whether cuts already calculated for this event
      LOGICAL CUTSDONE,CUTSPASSED
      COMMON/TO_CUTSDONE/CUTSDONE,CUTSPASSED
C ----------
C BEGIN CODE
C ----------
      DSIG=0D0

C     Make sure cuts are evaluated for first subprocess
      CUTSDONE=.FALSE.
      CUTSPASSED=.FALSE.

      IF(IMODE.EQ.1)THEN
C       Set up process information from file symfact
        LUN=NEXTUNOPEN()
        IPROC=1
        SYMCONF(IPROC)=ICONFIG
        OPEN(UNIT=LUN,FILE='../symfact.dat',STATUS='OLD',ERR=20)
        DO WHILE(.TRUE.)
          READ(LUN,*,ERR=10,END=10) XDUM, ICONF
          IF(ICONF.EQ.-MAPCONFIG(ICONFIG))THEN
            IPROC=IPROC+1
            SYMCONF(IPROC)=INT(XDUM)
          ENDIF
        ENDDO
 10     SYMCONF(0)=IPROC
        CLOSE(LUN)
        RETURN
 20     SYMCONF(0)=IPROC
        WRITE(*,*)'Error opening symfact.dat. No permutations used.'
        RETURN
      ELSE IF(IMODE.EQ.2)THEN
C     Output weights and number of events
        SUMPROB=0D0
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            DO K=1,2
              SUMPROB=SUMPROB+SUMWGT(K,I,J)
            ENDDO
          ENDDO
        ENDDO
        WRITE(*,*)'Relative summed weights:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(%(nsprocs)dE12.4)')((SUMWGT(K,I,J)/SUMPROB,K=1,2),I=1,MAXSPROC)
        ENDDO
        SUMPROB=0D0
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            DO K=1,2
              SUMPROB=SUMPROB+NUMEVTS(K,I,J)
            ENDDO
          ENDDO
        ENDDO
        WRITE(*,*)'Relative number of events:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(%(nsprocs)dE12.4)')((NUMEVTS(K,I,J)/SUMPROB,K=1,2),I=1,MAXSPROC)
        ENDDO
        WRITE(*,*)'Events:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(%(nsprocs)dI12)')((NUMEVTS(K,I,J),K=1,2),I=1,MAXSPROC)
        ENDDO
C       Reset weights and number of events
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            DO K=1,2
              NUMEVTS(K,I,J)=0
              SUMWGT(K,I,J)=0d0
            ENDDO
          ENDDO
        ENDDO
        RETURN
      ELSE IF(IMODE.EQ.3)THEN
C     No finalize needed
        RETURN
      ENDIF

C   IMODE.EQ.0, regular run mode

C     Select among the subprocesses based on PDF weight
      SUMPROB=0D0
      DO J=1,SYMCONF(0)
         DO IPROC=1,MAXSPROC
            IF(CONFSUB(IPROC,SYMCONF(J)).NE.0) THEN
               DO IMIRROR=1,2
                  IF(IMIRROR.EQ.1.OR.MIRRORPROCS(IPROC))THEN
C                 Calculate PDF weight for all subprocesses
                     SELPROC(IMIRROR,IPROC,J)=DSIGPROC(PP,J,IPROC,IMIRROR,SYMCONF,CONFSUB,1d0,4)
                     SUMPROB=SUMPROB+SELPROC(IMIRROR,IPROC,J)
                     IF(IMIRROR.EQ.2)THEN
C                 Need to flip back x values
                        XDUM=XBK(1)
                        XBK(1)=XBK(2)
                        XBK(2)=XDUM
                        CM_RAP=-CM_RAP
                     ENDIF
                  ENDIF
               ENDDO
            endif
         ENDDO
       ENDDO

C     Perform the selection
      CALL RANMAR(R)
      R=R*SUMPROB
      ICONF=0
      IPROC=0
      TOTWGT=0D0
      DO J=1,SYMCONF(0)
        DO I=1,MAXSPROC
          DO K=1,2
            TOTWGT=TOTWGT+SELPROC(K,I,J)
            IF(R.LT.TOTWGT)THEN
C               Normalize SELPROC to selection probability
                SELPROC(K,I,J)=SELPROC(K,I,J)/SUMPROB
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

C     Redo clustering to ensure consistent with final IPROC
      CUTSDONE=.FALSE.

C     Update weigth w.r.t SELPROC
      WGT=WGT/SELPROC(IMIRROR,IPROC,ICONF)

C     Call DSIGPROC to calculate sigma for process
      DSIG=DSIGPROC(PP,ICONF,IPROC,IMIRROR,SYMCONF,CONFSUB,WGT,IMODE)

      IF(DSIG.GT.0d0)THEN
C     Update summed weight and number of events
        SUMWGT(IMIRROR,IPROC,ICONF)=SUMWGT(IMIRROR,IPROC,ICONF)+DABS(DSIG*WGT)
        NUMEVTS(IMIRROR,IPROC,ICONF)=NUMEVTS(IMIRROR,IPROC,ICONF)+1
      ENDIF

      RETURN
      END

      FUNCTION DSIGPROC(PP,ICONF,IPROC,IMIRROR,SYMCONF,CONFSUB,WGT,IMODE)
C     ****************************************************
C     RETURNS DIFFERENTIAL CROSS SECTION 
C     FOR A PROCESS
C     Input:
C     pp    4 momentum of external particles
C     wgt   weight from Monte Carlo
C     imode 0 run, 1 init, 2 reweight, 3 finalize
C     Output:
C     Amplitude squared and summed
C     ****************************************************

      IMPLICIT NONE
      
      INCLUDE 'genps.inc'
      include 'maxconfigs.inc'
      INCLUDE 'nexternal.inc'
      INCLUDE 'maxamps.inc'
      INCLUDE 'coupl.inc'
      INCLUDE 'run.inc'
C     
C     ARGUMENTS 
C     
      DOUBLE PRECISION DSIGPROC
      DOUBLE PRECISION PP(0:3,NEXTERNAL), WGT
      INTEGER ICONF,IPROC,IMIRROR,IMODE
      INTEGER SYMCONF(0:LMAXCONFIGS)
      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
C     
C     GLOBAL VARIABLES
C     
C     SUBDIAG is vector of diagram numbers for this config
C     IB gives which beam is which (for mirror processes)
      INTEGER SUBDIAG(MAXSPROC),IB(2)
      COMMON/TO_SUB_DIAG/SUBDIAG,IB
C     ICONFIG has this config number
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
C     CM_RAP has parton-parton system rapidity
      DOUBLE PRECISION CM_RAP
      LOGICAL SET_CM_RAP
      COMMON/TO_CM_RAP/SET_CM_RAP,CM_RAP
C  
C EXTERNAL FUNCTIONS
C  
%(dsig_def_line)s
      LOGICAL PASSCUTS
C     
C     LOCAL VARIABLES 
C     
      DOUBLE PRECISION P1(0:3,NEXTERNAL),xdum
      INTEGER I,J,K,JC(NEXTERNAL)
      INTEGER PERMS(NEXTERNAL,LMAXCONFIGS)
      INCLUDE 'symperms.inc'

      ICONFIG=SYMCONF(ICONF)
      DO I=1,MAXSPROC
        SUBDIAG(I) = CONFSUB(I,SYMCONF(ICONF))
      ENDDO

C     Set momenta according to this permutation
      CALL SWITCHMOM(PP,P1,PERMS(1,MAPCONFIG(ICONFIG)),JC,NEXTERNAL)

      IB(1)=1
      IB(2)=2

      IF(IMIRROR.EQ.2)THEN
C       Flip momenta (rotate around x axis)
        DO I=1,NEXTERNAL
          P1(2,I)=-P1(2,I)
          P1(3,I)=-P1(3,I)
        ENDDO
C       Flip beam identity
        IB(1)=2
        IB(2)=1
C       Flip x values (to get boost right)
        XDUM=XBK(1)
        XBK(1)=XBK(2)
        XBK(2)=XDUM
C       Flip CM_RAP (to get rapidity right)
        CM_RAP=-CM_RAP
      ENDIF

      DSIGPROC=0D0

      IF (PASSCUTS(P1)) THEN
%(call_dsig_proc_lines)s
      ENDIF
      RETURN
      END

