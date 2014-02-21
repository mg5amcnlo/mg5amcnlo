DOUBLE PRECISION FUNCTION DSIG(PP,WGT)
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
C     Output:
C             Amplitude squared and summed
C ****************************************************
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include 'nexternal.inc'
      include 'maxamps.inc'
      REAL*8     PI
      PARAMETER (PI=3.1415926d0)
C  
C ARGUMENTS 
C  
      DOUBLE PRECISION PP(0:3,NEXTERNAL), WGT
C   
C LOCAL VARIABLES 
C  
      INTEGER I,J,K,ICONF,IMIRROR,NPROC
      SAVE NPROC
      LOGICAL MIRRORPROCS(MAXSPROC)
      INCLUDE 'mirrorprocs.inc'
      DOUBLE PRECISION XDUM
C  
C EXTERNAL FUNCTIONS
C  
      DOUBLE PRECISION DSIGPROC
      EXTERNAL DSIGPROC
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
      include 'run.inc'
C     IPROC has the present process number
      INTEGER IPROC
      COMMON/TO_MIRROR/IMIRROR, IPROC
C ----------
C BEGIN CODE
C ----------
      DSIG=0D0


C     Select among the subprocesses based on PDF weight
       DO IPROC=1,MAXSPROC
               DO IMIRROR=1,2
                  IF(IMIRROR.EQ.1.OR.MIRRORPROCS(IPROC))THEN
                     DSIG=DSIG+DSIGPROC(PP,IPROC,IMIRROR)
                     IF(IMIRROR.EQ.2)THEN
C                 Need to flip back x values
                        XDUM=XBK(1)
                        XBK(1)=XBK(2)
                        XBK(2)=XDUM
                     ENDIF
                  ENDIF
               ENDDO
       ENDDO


      RETURN
      END

      FUNCTION DSIGPROC(PP,IPROC,IMIRROR)
C     ****************************************************
C     RETURNS DIFFERENTIAL CROSS SECTION 
C     FOR A PROCESS
C     Input:
C     pp    4 momentum of external particles
C     wgt   weight from Monte Carlo
C     Output:
C     Amplitude squared and summed
C     ****************************************************

      IMPLICIT NONE
      
      INCLUDE 'nexternal.inc'
      INCLUDE 'maxamps.inc'
      INCLUDE 'coupl.inc'
      INCLUDE 'run.inc'
C     
C     ARGUMENTS 
C     
      DOUBLE PRECISION DSIGPROC
      DOUBLE PRECISION PP(0:3,NEXTERNAL), WGT
      INTEGER IPROC,IMIRROR
C     
C     GLOBAL VARIABLES
C     
C     SUBDIAG is vector of diagram numbers for this config
C     IB gives which beam is which (for mirror processes)
      INTEGER SUBDIAG(MAXSPROC),IB(2)
      COMMON/TO_SUB_DIAG/SUBDIAG,IB
C  
C EXTERNAL FUNCTIONS
C  
%(dsig_def_line)s
      LOGICAL PASSCUTS
C     
C     LOCAL VARIABLES 
C     
      DOUBLE PRECISION P1(0:3,NEXTERNAL),xdum
      DOUBLE PRECISION PTEMP(0:3)
      INTEGER I,J,K
      INTEGER IMODE

      IMODE=1 ! DUMMY VARIABLE
      DO I=1,NEXTERNAL
        DO J=0,3
          P1(J,I)=PP(J,I)
        ENDDO
      ENDDO

      IB(1)=1
      IB(2)=2

      IF(IMIRROR.EQ.2)THEN
        DO I=0,3
         PTEMP(I)=P1(I,1)
         P1(I,1)=P1(I,2)
         P1(I,2)=PTEMP(I)
        ENDDO
C       Flip x values 
        XDUM=XBK(1)
        XBK(1)=XBK(2)
        XBK(2)=XDUM
      ENDIF

      DSIGPROC=0D0

%(call_dsig_proc_lines)s
      RETURN
      END

