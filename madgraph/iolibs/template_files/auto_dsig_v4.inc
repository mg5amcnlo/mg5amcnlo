DOUBLE PRECISION FUNCTION DSIG%(proc_id)s(PP,WGT,IMODE)
C ****************************************************
C
%(info_lines)s
C
%(process_lines)s
C
C     RETURNS DIFFERENTIAL CROSS SECTION
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
      include 'nexternal.inc'
      include 'maxamps.inc'
      DOUBLE PRECISION       CONV
      PARAMETER (CONV=389379.66*1000)  !CONV TO PICOBARNS
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
      INTEGER I,ITYPE,LP
%(pdf_vars)s      
      DOUBLE PRECISION XPQ(-7:7)
      DOUBLE PRECISION DSIGUU
C  
C EXTERNAL FUNCTIONS
C  
      LOGICAL PASSCUTS
      DOUBLE PRECISION ALPHAS2,REWGT,PDG2PDF
C  
C GLOBAL VARIABLES
C  
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SubProc/ PD, IPROC
%(define_subdiag_lines)s
      include 'coupl.inc'
      include 'run.inc'
C  
C DATA
C  
%(pdf_data)s
C ----------
C BEGIN CODE
C ----------
      DSIG%(proc_id)s=0D0

C   Only run if IMODE is 0
      IF(IMODE.NE.0.AND.IMODE.NE.4) RETURN

%(passcuts_begin)s
%(pdf_lines)s
         IF (IMODE.EQ.4)THEN
            DSIG%(proc_id)s = PD(IPROC)
            RETURN
         ENDIF
         CALL SMATRIX%(proc_id)s(PP,DSIGUU)                                              
         DSIGUU=DSIGUU*REWGT(PP)
         IF (DSIGUU.LT.1D199) THEN
         DSIG%(proc_id)s=%(dsig_line)s
         ELSE
             write(*,*) "Error in matrix element"
             DSIGUU=0D0
             DSIG%(proc_id)s=0D0
         ENDIF
        IF(IMODE.EQ.0.AND.DSIG%(proc_id)s.GT.0D0)THEN
C       Call UNWGT to unweight and store events
           CALL UNWGT(PP,DSIG%(proc_id)s*WGT,%(numproc)d)
        ENDIF
%(passcuts_end)s
      END
