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
C                   3 finalize, 4 only PDFs,
C                   5 squared amplitude only (never
C                     generate events)
C     Output:
C             Amplitude squared and summed
C ****************************************************
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include 'genps.inc'
      include 'nexternal.inc'
      include 'maxconfigs.inc'
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
      INTEGER I,ITYPE,LP,IPROC
%(pdf_vars)s      
      DOUBLE PRECISION XPQ(-7:7),PD(0:MAXPROC)
      DOUBLE PRECISION DSIGUU,R,RCONF
      INTEGER LUN,ICONF,IFACT,NFACT
      DATA NFACT/1/
      SAVE NFACT
C  
C EXTERNAL FUNCTIONS
C  
      LOGICAL PASSCUTS
      DOUBLE PRECISION ALPHAS2,REWGT,PDG2PDF
      INTEGER NEXTUNOPEN
C  
C GLOBAL VARIABLES
C  
      INTEGER          IPSEL
      COMMON /SubProc/ IPSEL
C     MINCFIG has this config number
      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
C     Keep track of whether cuts already calculated for this event
      LOGICAL cutsdone,cutspassed
      COMMON/TO_CUTSDONE/cutsdone,cutspassed
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
      %(cutsdone)s
      IF(IMODE.EQ.1)THEN
C       Set up process information from file symfact
        LUN=NEXTUNOPEN()
        NFACT=1
        OPEN(UNIT=LUN,FILE='../symfact.dat',STATUS='OLD',ERR=20)
        DO WHILE(.TRUE.)
          READ(LUN,*,ERR=10,END=10) RCONF, IFACT
          ICONF=INT(RCONF)
          IF(ICONF.EQ.MAPCONFIG(MINCFIG))THEN
             NFACT=IFACT
          ENDIF
        ENDDO
 10     CLOSE(LUN)
        RETURN
 20     WRITE(*,*)'Error opening symfact.dat. No symmetry factor used.'
        RETURN
      ENDIF
C     Continue only if IMODE is 0, 4 or 5
      IF(IMODE.NE.0.AND.IMODE.NE.4.and.IMODE.NE.5) RETURN

%(passcuts_begin)s
%(pdf_lines)s
         IF (IMODE.EQ.4)THEN
            DSIG%(proc_id)s = PD(0)
            RETURN
         ENDIF
         CALL SMATRIX%(proc_id)s(PP,DSIGUU)
		 IF (IMODE.EQ.5) THEN
            IF (DSIGUU.LT.1D199) THEN		 
		      DSIG%(proc_id)s = DSIGUU*CONV
			else
		      DSIG%(proc_id)s = 0.0d0
			endif
			RETURN
		 ENDIF
C     Select a flavor combination (need to do here for right sign)
         CALL RANMAR(R)
         IPSEL=0
         DO WHILE (R.GE.0D0 .AND. IPSEL.LT.IPROC)
            IPSEL=IPSEL+1
            R=R-DABS(PD(IPSEL))/PD(0)
         ENDDO
         DSIGUU=DSIGUU*REWGT(PP)*NFACT
         IF (DSIGUU.LT.1D199) THEN
c           Set sign of dsig based on sign of PDF and matrix element
            DSIG%(proc_id)s=DSIGN(%(dsig_line)s,DSIGUU*PD(IPSEL))
         ELSE
             write(*,*) "Error in matrix element"
             DSIGUU=0D0
             DSIG%(proc_id)s=0D0
         ENDIF
C       Generate events only if IMODE is 0.
        IF(IMODE.EQ.0.AND.DABS(DSIG%(proc_id)s).GT.0D0)THEN
C       Call UNWGT to unweight and store events
           CALL UNWGT(PP,DSIG%(proc_id)s*WGT,%(numproc)d)
        ENDIF
%(passcuts_end)s
      END
## if(read_write_good_hel) {
C
C     Functionality to handling grid
C
%(read_write_good_hel)s

## }