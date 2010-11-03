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
C             imode 0 run, 1 init, 2 finalize
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
      DOUBLE PRECISION u1,ub1,d1,db1,c1,cb1,s1,sb1,b1,bb1
      DOUBLE PRECISION u2,ub2,d2,db2,c2,cb2,s2,sb2,b2,bb2
      DOUBLE PRECISION g1,g2
      DOUBLE PRECISION a1,a2
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
      include 'coupl.inc'
      include 'run.inc'
C  
C DATA
C  
      DATA u1,ub1,d1,db1,c1,cb1,s1,sb1,b1,bb1/10*1d0/
      DATA u2,ub2,d2,db2,c2,cb2,s2,sb2,b2,bb2/10*1d0/
      DATA a1,g1/2*1d0/
      DATA a2,g2/2*1d0/
C ----------
C BEGIN CODE
C ----------
      DSIG%(proc_id)s=0D0

C   Only run if IMODE is 0
      IF(IMODE.NE.0) RETURN

%(passcuts_begin)s
%(pdf_lines)s
         CALL SMATRIX%(proc_id)s(PP,DSIGUU)                                              
         DSIGUU=DSIGUU*REWGT(PP)
         IF (DSIGUU.LT.1D199) THEN
         DSIG%(proc_id)s=%(dsig_line)s
         ELSE
             write(*,*) "Error in matrix element"
             DSIGUU=0D0
             DSIG%(proc_id)s=0D0
         ENDIF
	 CALL UNWGT(PP,%(dsig_line)s*WGT,%(numproc)d)
%(passcuts_end)s
      END
