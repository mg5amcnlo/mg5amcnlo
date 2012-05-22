      SUBROUTINE SLOOPMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C  
C ARGUMENTS 
C  
      %(real_dp_format)s P(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(3)
	  INTEGER HEL, USERHEL
	  common/USERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
	  USERHEL=HEL
      CALL SLOOPMATRIX(P,ANS)
	  END

	  logical function IsZero(toTest, reference_value, ampLn, helN)
      IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NLOOPAMPS
	  PARAMETER (NLOOPAMPS=%(nloopamps)d)
	  %(real_dp_format)s vtiny
	  parameter (vtiny=1.0d-7)
C  
C ARGUMENTS 
C  
	  %(real_dp_format)s toTest, reference_value
	  integer ampLn,heln
C  
C GLOBAL 
C
      %(complex_dp_format)s AMPL(3,NLOOPAMPS)
	  LOGICAL S(NLOOPAMPS)
	  common/AMPL/AMPL,S
C ----------
C BEGIN CODE
C ----------
	  IF(abs(reference_value).eq.0.0d0) then
	    IsZero=.FALSE.
		write(*,*) '##E02 ERRROR Reference value for comparison is zero.'
		STOP
	  else
	    IsZero=((abs(toTest)/abs(reference_value)).lt.vtiny)
	  endif

	  IF((.NOT.ISZERO).AND.(AMPLN.NE.-1).AND.(.NOT.S(AMPLN))) THEN
	    write(*,*) '##W01 WARNING Contribution ',ampLn,' for hel ',helN,' is detected as contributing with CR=',(abs(toTest)/abs(reference_value)),' but is unstable.' 
	  ENDIF

	  end

      SUBROUTINE SLOOPMATRIX(P,ANS)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c and helicities for the point in phase space P(0:3,NEXTERNAL)
c and external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C
      CHARACTER*64 paramFileName
      PARAMETER ( paramFileName='MadLoopParams.dat')
	  %(nbornamps_decl)s
      INTEGER    NLOOPAMPS, NCTAMPS
      PARAMETER (NLOOPAMPS=%(nloopamps)d, NCTAMPS=%(nctamps)d) 
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      %(real_dp_format)s     ZERO
      PARAMETER (ZERO=0D0)
	  %(real_mp_format)s     MP__ZERO
      PARAMETER (MP__ZERO=0E0_16)
      %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C  
C ARGUMENTS 
C  
      %(real_dp_format)s P(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(3)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,K,H,DUMMY
c RESCT STORES THE RESULT OBTAINED FROM ANY OF THE 7 POSSIBLE CT MODES.
	  %(real_dp_format)s RESCT(3,0:6)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  INTEGER NATTEMPTS
	  DATA NATTEMPTS/0/
	  DATA IC/NEXTERNAL*1/
	  %(real_dp_format)s BUFFR(3),TEMP1,TEMP2
      %(complex_dp_format)s CFTOT
	  LOGICAL FOUNDHELFILTER,FOUNDLOOPFILTER
	  DATA FOUNDHELFILTER/.TRUE./
	  DATA FOUNDLOOPFILTER/.TRUE./
	  INTEGER IDEN
      %(den_factor_line)s

C
C FUNCTIONS
C
      LOGICAL ISZERO
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
	  include 'mp_coupl.inc'
	  include 'MadLoopParams.inc'

	  INTEGER NTRY
      DATA NTRY/0/
	  LOGICAL CHECKPHASE
	  DATA CHECKPHASE/.TRUE./
	  %(real_dp_format)s REF
	  DATA REF/0.0d0/
      common/INIT/NTRY,CHECKPHASE,REF

	  LOGICAL MP_DONE
	  DATA MP_DONE/.FALSE./
	  common/MP_DONE/MP_DONE

	  %(real_dp_format)s PS(0:3,NEXTERNAL)
	  common/PSPOINT/PS

	  %(real_dp_format)s LSCALE
	  INTEGER CTMODE	  
      common/CT/LSCALE,CTMODE

	  LOGICAL GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPAMPS,NCOMB)
	  common/Filters/GOODAMP,GOODHEL

	  INTEGER HELPICKED
	  DATA HELPICKED/-1/
	  common/HELCHOICE/HELPICKED
	  INTEGER USERHEL
	  DATA USERHEL/-1/
	  common/USERCHOICE/USERHEL

      %(dp_born_amps_decl)s	  
	  %(complex_dp_format)s W(20,NWAVEFUNCS%(ncomb_helas_objs)s)
	  INTEGER VALIDH
	  common/WFCTS/W  
	  common/VALIDH/VALIDH

      %(complex_dp_format)s AMPL(3,NLOOPAMPS)
	  LOGICAL S(NLOOPAMPS)
	  common/AMPL/AMPL,S

	  INTEGER CF_D(NLOOPAMPS,%(color_matrix_size)s)
	  INTEGER CF_N(NLOOPAMPS,%(color_matrix_size)s)
	  common/CF/CF_D,CF_N

	  INTEGER HELC(NEXTERNAL,NCOMB)
	  common/HELCONFIGS/HELC

C ----------
C BEGIN CODE
C ----------

IF(NTRY.EQ.0) THEN
  CALL MADLOOPPARAMREADER(paramFileName,.TRUE.)
  DO J=1,NCOMB
    DO I=1,NCTAMPS
	  GOODAMP(I,J)=.TRUE.
	ENDDO
  ENDDO
OPEN(1, FILE="LoopFilter.dat", err=100, status='OLD',           action='READ')
  DO J=1,NCOMB
    READ(1,*,END=101) (GOODAMP(I,J),I=NCTAMPS+1,NLOOPAMPS)
  ENDDO
  GOTO 101
100  CONTINUE
  FOUNDLOOPFILTER=.FALSE.
  DO J=1,NCOMB
    DO I=NCTAMPS+1,NLOOPAMPS
	  GOODAMP(I,J)=(.NOT.USELOOPFILTER)
	ENDDO
  ENDDO
101  CONTINUE
CLOSE(1)
OPEN(1, FILE="HelFilter.dat", err=102, status='OLD',           action='READ')
  READ(1,*,END=103) (GOODHEL(I),I=1,NCOMB)
  GOTO 103
102  CONTINUE
  FOUNDHELFILTER=.FALSE.
  DO J=1,NCOMB
	GOODHEL(J)=.TRUE.
  ENDDO
103  CONTINUE
CLOSE(1)
OPEN(1, FILE="ColorNumFactors.dat", err=104, status='OLD',           action='READ')
  DO I=1,NLOOPAMPS
    READ(1,*,END=105) (CF_N(I,J),J=1,%(color_matrix_size)s)
  ENDDO
  GOTO 105
104  CONTINUE
  STOP 'Color factors could not be initialized from file ColorNumFactors.dat. File not found' 
105  CONTINUE
CLOSE(1)
OPEN(1, FILE="ColorDenomFactors.dat", err=106, status='OLD',           action='READ')
  DO I=1,NLOOPAMPS
    READ(1,*,END=107) (CF_D(I,J),J=1,%(color_matrix_size)s)
  ENDDO
  GOTO 107
106  CONTINUE
  STOP 'Color factors could not be initialized from file ColorDenomFactors.dat. File not found' 
107  CONTINUE
CLOSE(1)
OPEN(1, FILE="HelConfigs.dat", err=108, status='OLD',                  action='READ')
  DO H=1,NCOMB
    READ(1,*,END=109) (HELC(I,H),I=1,NEXTERNAL)
  ENDDO
  GOTO 109
108  CONTINUE
  STOP 'Color helictiy configurations could not be initialized from file HelConfigs.dat. File not found' 
109  CONTINUE
CLOSE(1)
ENDIF

LSCALE=DSQRT((P(0,1)+P(0,2))**2-(P(1,1)+P(1,2))**2-(P(2,1)+P(2,2))**2-(P(3,1)+P(3,2))**2)

%(set_reference)s

NTRY=NTRY+1
CHECKPHASE=(NTRY.LE.CHECKCYCLE).AND.(((.NOT.FOUNDLOOPFILTER).AND.USELOOPFILTER).OR.(.NOT.FOUNDHELFILTER))

DO I=1,NEXTERNAL
  DO J=0,3
    PS(J,I)=P(J,I)
  ENDDO
ENDDO

IF ((.NOT. CHECKPHASE).AND.(.NOT.FOUNDHELFILTER)) THEN
OPEN(1, FILE="HelFilter.dat", err=110, status='NEW',       action='WRITE')
  WRITE(1,*) (GOODHEL(I),I=1,NCOMB)
110  CONTINUE
  CLOSE(1)
FOUNDHELFILTER=.TRUE.
ENDIF

IF ((.NOT. CHECKPHASE).AND.(.NOT.FOUNDLOOPFILTER).AND.USELOOPFILTER) THEN
OPEN(1, FILE="LoopFilter.dat", err=111, status='NEW',       action='WRITE')
  DO J=1,NCOMB
    WRITE(1,*) (GOODAMP(I,J),I=NCTAMPS+1,NLOOPAMPS)
  ENDDO
111  CONTINUE
  CLOSE(1)
FOUNDLOOPFILTER=.TRUE.
ENDIF

IF(CHECKPHASE) THEN
  HELPICKED=1
  CTMODE=CTMODEINIT
ELSE
  HELPICKED=USERHEL
  IF (CTMODERUN.NE.-1) THEN
    CTMODE=CTMODERUN
  ELSE
    CTMODE=1
  ENDIF
ENDIF

DO K=1, 3
  BUFFR(K)=0.0d0
  DO I=1,NLOOPAMPS
    AMPL(K,I)=(0.0d0,0.0d0)
  ENDDO
ENDDO

199 CONTINUE

MP_DONE=.FALSE.

200 CONTINUE

DO K=1,3
  ANS(K)=0.0d0
ENDDO

VALIDH=-1
DO H=1,NCOMB
  IF (((HELPICKED.EQ.-1).OR.(HELPICKED.EQ.H)).AND.(CHECKPHASE.OR.GOODHEL(H))) THEN
  IF (VALIDH.EQ.-1) VALIDH=H
  DO I=1,NEXTERNAL
    NHEL(I)=HELC(I,H)
  ENDDO
  %(loop_induced_setup)s
  %(born_ct_helas_calls)s
  %(loop_induced_helas_calls)s
  %(loop_induced_finalize)s
  DO I=1,%(nctamps_or_nloopamps)s
    DO J=1,%(nbornamps_or_nloopamps)s
	  CFTOT=DCMPLX(CF_N(I,J)/DBLE(ABS(CF_D(I,J))),0.0d0)
      IF(CF_D(I,J).LT.0) CFTOT=CFTOT*IMAG1
	  %(squaring)s
    ENDDO
  ENDDO
  ENDIF
ENDDO

%(loop_helas_calls)s

%(actualize_ans)s

IF (CHECKPHASE) THEN
  IF((USERHEL.EQ.-1).OR.(USERHEL.EQ.HELPICKED)) THEN
  BUFFR(1)=BUFFR(1)+ANS(1)
  BUFFR(2)=BUFFR(2)+ANS(2)
  BUFFR(3)=BUFFR(3)+ANS(3)
  ENDIF

C SET THE HELICITY FILTER
  IF(.NOT.FOUNDHELFILTER) THEN
  IF(ISZERO(ABS(ANS(1)+ANS(2)+ANS(3)),REF/DBLE(NCOMB),-1,HELPICKED)) THEN
    IF(NTRY.EQ.1) THEN
	  GOODHEL(HELPICKED)=.FALSE.
	ELSEIF(GOODHEL(HELPICKED)) THEN
	  WRITE(*,*) '##W02 WARNING Inconsistent helicity ',HELPICKED,' filter setting, starting over.'
	  NTRY=0
	ENDIF
  ENDIF
  ENDIF

C SET THE LOOP FILTER
  IF(.NOT.FOUNDLOOPFILTER.AND.USELOOPFILTER) THEN
  	DO I=NCTAMPS+1,NLOOPAMPS
      IF(.NOT.ISZERO(ABS(AMPL(1,I))+ABS(AMPL(2,I))+ABS(AMPL(3,I)),(REF*1.0d-4),I,HELPICKED)) THEN
        IF(NTRY.EQ.1) THEN
	      GOODAMP(I,HELPICKED)=.TRUE.
	    ELSEIF(.NOT.GOODAMP(I,HELPICKED)) THEN
	      WRITE(*,*) '##W02 WARNING Inconsistent loop amp ',I,' for helicity ',HELPICKED,'.'
		  IF(LOOPINITSTARTOVER) THEN
		    WRITE(*,*) '##I01 INFO Initialization starting over because of inconsistency in the loop filter setup.'
	        NTRY=0
		  ELSE
		    GOODAMP(I,HELPICKED)=.TRUE.
		  ENDIF
	    ENDIF
      ENDIF
  	ENDDO
  ENDIF

C GOTO NEXT HELICITY OR FINISH
  IF(HELPICKED.NE.NCOMB) THEN
    HELPICKED=HELPICKED+1
    goto 199
  ELSE
    ANS(1)=BUFFR(1)
	ANS(2)=BUFFR(2)
	ANS(3)=BUFFR(3)
	IF(NTRY.EQ.0) THEN
	  NATTEMPTS=NATTEMPTS+1
	  IF(NATTEMPTS.EQ.MAXATTEMPTS) THEN
	    WRITE(*,*) '##E01 ERROR Could not initialize the filters in ',MAXATTEMPTS,' trials'
		STOP
	  ENDIF
	ENDIF
  ENDIF

ENDIF

ANS(1)=ANS(1)/DBLE(IDEN)
ANS(2)=ANS(2)/DBLE(IDEN)
ANS(3)=ANS(3)/DBLE(IDEN)

IF(.NOT.CHECKPHASE.AND.(CTMODERUN.EQ.-1)) THEN
  RESCT(1,CTMODE)=ANS(1)
  RESCT(2,CTMODE)=ANS(2)
  RESCT(3,CTMODE)=ANS(3)
  IF(CTMODE.EQ.1) THEN
    CTMODE=2
	goto 200
  ELSEIF(CTMODE.EQ.2) THEN
    TEMP1=ABS(RESCT(1,1))+ABS(RESCT(2,1))+ABS(RESCT(3,1))
    TEMP2=ABS(RESCT(1,2))+ABS(RESCT(2,2))+ABS(RESCT(3,2))
	IF((2.0d0*ABS(TEMP1-TEMP2))/(TEMP1+TEMP2).LT.MLSTABTHRES) THEN
	  ANS(1)=0.5d0*(RESCT(1,1)+RESCT(1,2))
	  ANS(2)=0.5d0*(RESCT(2,1)+RESCT(2,2))
	  ANS(3)=0.5d0*(RESCT(3,1)+RESCT(3,2))			  
	ELSE
	  CTMODE=4
	  goto 200
	ENDIF
  ELSEIF(CTMODE.EQ.4) THEN
    CTMODE=5
	goto 200
  ELSEIF(CTMODE.EQ.5) THEN
	ANS(1)=0.5d0*(RESCT(1,4)+RESCT(1,5))
	ANS(2)=0.5d0*(RESCT(2,4)+RESCT(2,5))
	ANS(3)=0.5d0*(RESCT(3,4)+RESCT(3,5))
    TEMP1=ABS(RESCT(1,4))+ABS(RESCT(2,4))+ABS(RESCT(3,4))
    TEMP2=ABS(RESCT(1,5))+ABS(RESCT(2,5))+ABS(RESCT(3,5))
	IF((2.0d0*ABS(TEMP1-TEMP2))/(TEMP1+TEMP2).GT.MLSTABTHRES) THEN
      WRITE(*,*) '##W03 WARNING An unstable PS point was',       ' detected with relative imprecision =',(2.0d0*ABS(TEMP1-TEMP2))/(TEMP1+TEMP2)
	ENDIF 
  ENDIF
ENDIF

      END
