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
C     Output:
C             Amplitude squared and summed
C ****************************************************
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include 'phasespace.inc'
      include 'nexternal.inc'
      include 'maxamps.inc'
      DOUBLE PRECISION       CONV
      PARAMETER (CONV=1d0)  !NO CONV (GeV^n)
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
      DOUBLE PRECISION u1,ux1,d1,dx1,c1,cx1,s1,sx1,b1,bx1
      DOUBLE PRECISION u2,ux2,d2,dx2,c2,cx2,s2,sx2,b2,bx2
      DOUBLE PRECISION g1,g2
      DOUBLE PRECISION a1,a2
      DOUBLE PRECISION XPQ(-7:7)
      DOUBLE PRECISION DSIGUU
C  
C EXTERNAL FUNCTIONS
C  
      DOUBLE PRECISION PDG2PDF
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
      DATA u1,ux1,d1,dx1,c1,cx1,s1,sx1,b1,bx1/10*1d0/
      DATA u2,ux2,d2,dx2,c2,cx2,s2,sx2,b2,bx2/10*1d0/
      DATA a1,g1/2*1d0/
      DATA a2,g2/2*1d0/
C ----------
C BEGIN CODE
C ----------
      DSIG%(proc_id)s=0D0



%(pdf_lines)s
       CALL SMATRIX%(proc_id)s(PP,DSIGUU)
         IF (DSIGUU.LT.1D199) THEN
         DSIG%(proc_id)s=%(dsig_line)s
         ELSE
             write(*,*) "Error in matrix element"
             DSIGUU=0D0
             DSIG%(proc_id)s=0D0
         ENDIF
      END
