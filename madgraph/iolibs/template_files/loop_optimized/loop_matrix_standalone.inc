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

      SUBROUTINE SLOOPMATRIX(P_IN,ANS)
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
      INTEGER    NLOOPS, NLOOPGROUPS, NCTAMPS
      PARAMETER (NLOOPS=%(nloops)d, NLOOPGROUPS=%(nloop_groups)d, NCTAMPS=%(nctamps)d) 
      INTEGER    NCOLORROWS
	  PARAMETER (NCOLORROWS=%(nloopamps)d)
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS,NLOOPWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d,NLOOPWAVEFUNCS=%(nloopwavefuncs)d)
	  INTEGER MAXLWFSIZE
	  PARAMETER (MAXLWFSIZE=%(max_lwf_size)d)
	  INTEGER LOOPMAXCOEFS, VERTEXMAXCOEFS
	  PARAMETER (LOOPMAXCOEFS=%(loop_max_coefs)d, VERTEXMAXCOEFS=%(vertex_max_coefs)d)
	  INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
      %(real_dp_format)s     ZERO
      PARAMETER (ZERO=0D0)
	  %(real_mp_format)s     MP__ZERO
      PARAMETER (MP__ZERO=0E0_16)
      %(complex_dp_format)s IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C     This parameter is designed for the check timing command of MG5
      LOGICAL SKIPLOOPEVAL
      PARAMETER (SKIPLOOPEVAL=.FALSE.)
C  
C ARGUMENTS 
C  
      %(real_dp_format)s P_IN(0:3,NEXTERNAL)
      %(real_dp_format)s ANS(3)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,K,H,DUMMY
      %(real_dp_format)s P(0:3,NEXTERNAL)	  
c RESCT STORES THE RESULT OBTAINED FROM ANY OF THE 7 POSSIBLE CT MODES.
	  %(real_dp_format)s RESCT(3,0:6)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  INTEGER NATTEMPTS
	  DATA NATTEMPTS/0/
	  DATA IC/NEXTERNAL*1/
	  %(real_dp_format)s HELSAVED(3,NCOMB)	  
	  %(real_dp_format)s BUFFR(3),BUFFR_BIS(3),TEMP1,TEMP2
	  %(complex_dp_format)s COEFS(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
      %(complex_dp_format)s CFTOT
	  LOGICAL FOUNDHELFILTER,FOUNDLOOPFILTER
	  DATA FOUNDHELFILTER/.TRUE./
	  DATA FOUNDLOOPFILTER/.TRUE./
      LOGICAL LOOPFILTERBUFF(NLOOPGROUPS)
	  DATA (LOOPFILTERBUFF(I),I=1,NLOOPGROUPS)/NLOOPGROUPS*.FALSE./

	  INTEGER IDEN
      %(den_factor_line)s
C
C FUNCTIONS
C
      INTEGER ISSAME
      LOGICAL ISZERO
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
	  include 'mp_coupl.inc'
	  include 'MadLoopParams.inc'

	  LOGICAL CHECKPHASE
	  DATA CHECKPHASE/.TRUE./
	  LOGICAL HELDOUBLECHECKED
	  DATA HELDOUBLECHECKED/.FALSE./
      common/INIT/CHECKPHASE, HELDOUBLECHECKED
	  INTEGER NTRY
      DATA NTRY/0/
	  %(real_dp_format)s REF
	  DATA REF/0.0d0/

	  LOGICAL MP_DONE
	  DATA MP_DONE/.FALSE./
	  common/MP_DONE/MP_DONE

	  %(real_dp_format)s PS(0:3,NEXTERNAL)
	  common/PSPOINT/PS

	  %(real_dp_format)s LSCALE
	  INTEGER CTMODE	  
      common/CT/LSCALE,CTMODE

C     The parameter below sets the convention for the helicity filter
C     For a given helicity, the attached integer 'i' means
C     'i' in ]-inf;-HELOFFSET[ -> Helicity is equal, up to a sign, to helicity number abs(i+HELOFFSET)
C     'i' == -HELOFFSET        -> Helicity is analytically zero
C     'i' in ]-HELOFFSET,inf[  -> Helicity is contributing with weight 'i'. If it is zero, it is skipped.
C     Typically, the hel_offset is 10000
	  INTEGER HELOFFSET
	  DATA HELOFFSET/%(hel_offset)d/
	  INTEGER GOODHEL(NCOMB)
	  LOGICAL GOODAMP(NLOOPGROUPS)
	  common/Filters/GOODAMP,GOODHEL,HELOFFSET

	  INTEGER HELPICKED
	  DATA HELPICKED/-1/
	  common/HELCHOICE/HELPICKED
	  INTEGER USERHEL
	  DATA USERHEL/-1/
	  common/USERCHOICE/USERHEL

	  %(dp_born_amps_decl)s	  
	  %(complex_dp_format)s W(20,NWAVEFUNCS)
	  common/W/W  

      %(complex_mp_format)s MPW(20,NWAVEFUNCS)
	  common/MP_W/MPW

	  %(complex_dp_format)s WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,0:NLOOPWAVEFUNCS)
	  %(complex_dp_format)s PL(0:3,0:NLOOPWAVEFUNCS)
	  common/WL/WL,PL

	  %(complex_dp_format)s LOOPCOEFS(0:LOOPMAXCOEFS-1,NLOOPS)
	  common/LCOEFS/LOOPCOEFS

      %(complex_dp_format)s AMPL(3,NCTAMPS)
	  common/AMPL/AMPL

      %(complex_dp_format)s LOOPRES(3,NLOOPGROUPS)
	  LOGICAL S(NLOOPGROUPS)
	  common/LOOPRES/LOOPRES,S

	  INTEGER CF_D(NCOLORROWS,%(color_matrix_size)s)
	  INTEGER CF_N(NCOLORROWS,%(color_matrix_size)s)
	  common/CF/CF_D,CF_N

	  INTEGER HELC(NEXTERNAL,NCOMB)
	  common/HELCONFIGS/HELC

C ----------
C BEGIN CODE
C ----------

IF(NTRY.EQ.0) THEN
  CALL MADLOOPPARAMREADER(paramFileName,.TRUE.)
  HELDOUBLECHECKED=.NOT.DoubleCheckHelicityFilter
OPEN(1, FILE="LoopFilter.dat", err=100, status='OLD',           action='READ')
  DO J=1,NLOOPGROUPS
    READ(1,*,END=101) GOODAMP(J)
  ENDDO
  GOTO 101
100  CONTINUE
  FOUNDLOOPFILTER=.FALSE.
  DO J=1,NLOOPGROUPS
	GOODAMP(J)=(.NOT.USELOOPFILTER)
  ENDDO
101  CONTINUE
CLOSE(1)
OPEN(1, FILE="HelFilter.dat", err=102, status='OLD',           action='READ')
  DO I=1,NCOMB
    READ(1,*,END=103) GOODHEL(I)
  ENDDO
  GOTO 103
102  CONTINUE
  FOUNDHELFILTER=.FALSE.
  DO J=1,NCOMB
	GOODHEL(J)=1
  ENDDO
103  CONTINUE
CLOSE(1)
OPEN(1, FILE="ColorNumFactors.dat", err=104, status='OLD',           action='READ')
  DO I=1,NCOLORROWS
    READ(1,*,END=105) (CF_N(I,J),J=1,%(color_matrix_size)s)
  ENDDO
  GOTO 105
104  CONTINUE
  STOP 'Color factors could not be initialized from file ColorNumFactors.dat. File not found' 
105  CONTINUE
CLOSE(1)
OPEN(1, FILE="ColorDenomFactors.dat", err=106, status='OLD',           action='READ')
  DO I=1,NCOLORROWS
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
C SETUP OF THE COMMON STARTING EXTERNAL LOOP WAVEFUNCTION
C IT IS ALSO PS POINT INDEPENDENT, SO IT CAN BE DONE HERE.
  DO I=0,3
    PL(I,0)=(0.0d0,0.0d0)
  ENDDO
  DO I=1,MAXLWFSIZE
    DO J=0,LOOPMAXCOEFS-1
	  DO K=1,MAXLWFSIZE
	    IF(I.EQ.K.AND.J.EQ.0) then
        	WL(I,J,K,0)=(1.0d0,0.0d0)
		ELSE
        	WL(I,J,K,0)=(0.0d0,0.0d0)
		ENDIF
	  ENDDO
	ENDDO
  ENDDO 
ENDIF

DO I=1,NEXTERNAL
  DO J=0,3
    P(J,I)=P_IN(J,I)
    PS(J,I)=P_IN(J,I)
  ENDDO
ENDDO

CALL IMPROVE_PS_POINT_PRECISION(P)

LSCALE=DSQRT((P(0,1)+P(0,2))**2-(P(1,1)+P(1,2))**2-(P(2,1)+P(2,2))**2-(P(3,1)+P(3,2))**2)

%(set_reference)s

NTRY=NTRY+1
CHECKPHASE=(NTRY.LE.CHECKCYCLE).AND.(((.NOT.FOUNDLOOPFILTER).AND.USELOOPFILTER).OR.(.NOT.FOUNDHELFILTER))

IF ((.NOT. CHECKPHASE).AND.(.NOT.FOUNDHELFILTER)) THEN
OPEN(1, FILE="HelFilter.dat", err=110, status='NEW',       action='WRITE')
  DO I=1,NCOMB
    WRITE(1,*) GOODHEL(I)  
  ENDDO
110  CONTINUE
  CLOSE(1)
FOUNDHELFILTER=.TRUE.
ENDIF

IF ((.NOT. CHECKPHASE).AND.(.NOT.FOUNDLOOPFILTER).AND.USELOOPFILTER) THEN
OPEN(1, FILE="LoopFilter.dat", err=111, status='NEW',       action='WRITE')
  DO J=1,NLOOPGROUPS
    WRITE(1,*) GOODAMP(J)
  ENDDO
111  CONTINUE
  CLOSE(1)
FOUNDLOOPFILTER=.TRUE.
ENDIF

IF(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)) THEN
  HELPICKED=1
  CTMODE=CTMODEINIT
ELSE
  IF (USERHEL.ne.-1.and.GOODHEL(USERHEL).eq.-HELOFFSET) THEN
    ANS(1)=0.0d0
    ANS(2)=0.0d0
    ANS(3)=0.0d0
    goto 9999
  ELSE   
    HELPICKED=USERHEL      
  ENDIF
  IF (CTMODERUN.NE.-1) THEN
    CTMODE=CTMODERUN
  ELSE
    CTMODE=1
  ENDIF
ENDIF

DO K=1, 3
  BUFFR(K)=0.0d0
  DO I=1,NCTAMPS
    AMPL(K,I)=(0.0d0,0.0d0)
  ENDDO
ENDDO

199 CONTINUE

C Check if we directly go to multiple precision
IF (CTMODE.GE.4) THEN
  CALL MP_COMPUTE_LOOP_COEFS(P,BUFFR_BIS)
  MP_DONE=.TRUE.
c Copy the qp wfs to the dp ones as they are used to setup the CT calls.
  DO I=1,NWAVEFUNCS
    DO J=1,MAXLWFSIZE+4
      W(J,I)=MPW(J,I)
    ENDDO
  ENDDO
  goto 300 
ELSE
  MP_DONE=.FALSE.
ENDIF

200 CONTINUE

DO I=1,NLOOPS
  DO J=0,LOOPMAXCOEFS-1
    LOOPCOEFS(J,I)=(0.0d0,0.0d0)
  ENDDO
ENDDO

DO K=1,3
  ANS(K)=0.0d0
ENDDO

DO H=1,NCOMB
  IF ((HELPICKED.EQ.H).OR.((HELPICKED.EQ.-1).AND.(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED).OR.(GOODHEL(H).GT.-HELOFFSET.AND.GOODHEL(H).NE.0)))) THEN
  DO I=1,NEXTERNAL
    NHEL(I)=HELC(I,H)
  ENDDO
  %(born_ct_helas_calls)s
  IF (.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.HELPICKED.EQ.-1) THEN
    DUMMY=GOODHEL(H)
  ELSE
    DUMMY=1
  ENDIF
  DO I=1,%(nctamps_or_nloopamps)s
    DO J=1,%(nbornamps_or_nloopamps)s
	  CFTOT=DCMPLX(CF_N(I,J)/DBLE(ABS(CF_D(I,J))),0.0d0)
      IF(CF_D(I,J).LT.0) CFTOT=CFTOT*IMAG1
	  %(squaring)s
    ENDDO
  ENDDO
  %(coef_construction)s  
  ENDIF
ENDDO

%(coef_merging)s

BUFFR_BIS(1)=ANS(1)
BUFFR_BIS(2)=ANS(2)
BUFFR_BIS(3)=ANS(3)
300 CONTINUE
ANS(1)=BUFFR_BIS(1)
ANS(2)=BUFFR_BIS(2)
ANS(3)=BUFFR_BIS(3)

IF(SKIPLOOPEVAL) THEN
  GOTO 1226
ENDIF

%(loop_CT_calls)s

%(actualize_ans)s

1226 CONTINUE

IF (CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)) THEN
  IF((USERHEL.EQ.-1).OR.(USERHEL.EQ.HELPICKED)) THEN
C   TO KEEP TRACK OF THE FINAL ANSWER TO BE RETURNED DURING CHECK PHASE
    BUFFR(1)=BUFFR(1)+ANS(1)
    BUFFR(2)=BUFFR(2)+ANS(2)
    BUFFR(3)=BUFFR(3)+ANS(3)
  ENDIF
C SAVE RESULT OF EACH INDEPENDENT HELICITY FOR COMPARISON DURING THE HELICITY FILTER SETUP
  HELSAVED(1,HELPICKED)=ANS(1)
  HELSAVED(2,HELPICKED)=ANS(2)
  HELSAVED(3,HELPICKED)=ANS(3)

  IF (CHECKPHASE) THEN
C   SET THE HELICITY FILTER
    IF(.NOT.FOUNDHELFILTER) THEN
      IF(ISZERO(ABS(ANS(1))+ABS(ANS(2))+ABS(ANS(3)),REF/DBLE(NCOMB),-1)) THEN
        IF(NTRY.EQ.1) THEN
	      GOODHEL(HELPICKED)=-HELOFFSET
	    ELSEIF(GOODHEL(HELPICKED).NE.-HELOFFSET) THEN
	      WRITE(*,*) '##W02 WARNING Inconsistent vanishing helicity ',HELPICKED,' filter setting, starting over.'
	      NTRY=0
	    ENDIF
      ELSE
	    DO H=1,HELPICKED-1
		  IF(GOODHEL(H).GT.-HELOFFSET) THEN
		    DUMMY=ISSAME(ANS,HELSAVED(1,H),REF)
		    IF(DUMMY.NE.0) THEN
		      IF(NTRY.EQ.1) THEN
C               Set the matching helicity to be contributing once more
		        GOODHEL(H)=GOODHEL(H)+DUMMY
C               Use an offset to clearly show it is linked to an other one and to avoid overlap
			    GOODHEL(HELPICKED)=-H-HELOFFSET
C             Make sure we have paired this hel config to the same one last PS point
			  ELSEIF(GOODHEL(HELPICKED).NE.(-H-HELOFFSET)) THEN
	      	    WRITE(*,*) '##W02 WARNING Inconsistent matching helicity ',HELPICKED,' filter setting, starting over.'
	      	    NTRY=0
			  ENDIF
			  EXIT
			ENDIF
		  ENDIF
		ENDDO
	  ENDIF
    ENDIF

C   SET THE LOOP FILTER
    IF(.NOT.FOUNDLOOPFILTER.AND.USELOOPFILTER) THEN
  	  DO I=1,NLOOPGROUPS
        IF(.NOT.ISZERO(ABS(LOOPRES(1,I))+ABS(LOOPRES(2,I))+ABS(LOOPRES(3,I)),(REF*1.0d-4),I)) THEN
          IF(NTRY.EQ.1) THEN
	        GOODAMP(I)=.TRUE.
		    LOOPFILTERBUFF(I)=.TRUE.
	      ELSEIF(.NOT.LOOPFILTERBUFF(I)) THEN
	        WRITE(*,*) '##W02 WARNING Inconsistent loop amp ',I,'.'
		    IF(LOOPINITSTARTOVER) THEN
		      WRITE(*,*) '##I01 INFO Initialization starting over because of inconsistency in the loop filter setup.'
	          NTRY=0
		    ELSE
		      GOODAMP(I)=.TRUE.
		    ENDIF
	      ENDIF
        ENDIF
  	  ENDDO
    ENDIF
  ELSEIF (.NOT.HELDOUBLECHECKED)THEN
C   DOUBLE CHECK THE HELICITY FILTER
    IF ((GOODHEL(HELPICKED).EQ.-HELOFFSET).AND.(.NOT.ISZERO(ABS(ANS(1))+ABS(ANS(2))+ABS(ANS(3)),REF/DBLE(NCOMB),-1))) THEN
	  write(*,*) '##E03 Helicity filter could not be successfully double checked.'
	  write(*,*) 'One reason for this is that you have changed sensible parameters which affected what are the zero helicity configurations.'
	  write(*,*) 'Please regenerate the filter by removing HelFilter.dat and rerunning SLOOPMATRIX over a couple of PS points.'
	  STOP
	ELSEIF (GOODHEL(HELPICKED).LT.-HELOFFSET.AND.(ISSAME(ANS,HELSAVED(1,ABS(GOODHEL(HELPICKED)+HELOFFSET)),REF).EQ.0)) THEN
	  write(*,*) '##E03 Helicity filter could not be successfully double checked.'
	  write(*,*) 'One reason for this is that you have changed sensible parameters which affected the dependent helicity relationships.'
	  write(*,*) 'Please regenerate the filter by removing HelFilter.dat and rerunning SLOOPMATRIX over a couple of PS points.'
	  STOP
	ENDIF
C   SET HELDOUBLECHECKED TO .TRUE. WHEN DONE
	IF (HELPICKED.EQ.NCOMB) THEN
	  HELDOUBLECHECKED=.TRUE.
	ENDIF
  ENDIF

C GOTO NEXT HELICITY OR FINISH
  IF(HELPICKED.NE.NCOMB) THEN
    HELPICKED=HELPICKED+1
    goto 199
  ELSE
C   Useful printout
c    do I=1,NCOMB
c      write(*,*) 'HELSAVED(1,',I,')=',HELSAVED(1,I)
c      write(*,*) 'HELSAVED(2,',I,')=',HELSAVED(2,I)
c      write(*,*) 'HELSAVED(3,',I,')=',HELSAVED(3,I)
c      write(*,*) '   GOODHEL(',I,')=',GOODHEL(I)
c    ENDDO
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

IF(.NOT.CHECKPHASE.AND.HELDOUBLECHECKED.AND.(CTMODERUN.EQ.-1)) THEN
  RESCT(1,CTMODE)=ANS(1)
  RESCT(2,CTMODE)=ANS(2)
  RESCT(3,CTMODE)=ANS(3)
  IF(CTMODE.EQ.1) THEN
    CTMODE=2
	goto 300
  ELSEIF(CTMODE.EQ.2) THEN
    TEMP1=ABS(RESCT(1,1))+ABS(RESCT(2,1))+ABS(RESCT(3,1))
    TEMP2=ABS(RESCT(1,2))+ABS(RESCT(2,2))+ABS(RESCT(3,2))
	IF((2.0d0*ABS(TEMP1-TEMP2))/(TEMP1+TEMP2).LT.MLSTABTHRES) THEN
	  ANS(1)=0.5d0*(RESCT(1,1)+RESCT(1,2))
	  ANS(2)=0.5d0*(RESCT(2,1)+RESCT(2,2))
	  ANS(3)=0.5d0*(RESCT(3,1)+RESCT(3,2))			  
	ELSE
	  IF (.NOT.MP_DONE) THEN
        CALL MP_COMPUTE_LOOP_COEFS(P,BUFFR_BIS)
        MP_DONE=.TRUE.	    
	  ENDIF
	  CTMODE=4
	  goto 300
	ENDIF
  ELSEIF(CTMODE.EQ.4) THEN
    CTMODE=5
	goto 300
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
 9999 CONTINUE
      END


	  logical function IsZero(toTest, reference_value, loop)
      IMPLICIT NONE
C  
C CONSTANTS
C
      INTEGER    NLOOPGROUPS
	  PARAMETER (NLOOPGROUPS=%(nloop_groups)d)
C  
C ARGUMENTS 
C  
	  %(real_dp_format)s toTest, reference_value
	  integer loop
C  
C GLOBAL 
C
      INCLUDE 'MadLoopParams.inc'
      %(complex_dp_format)s LOOPRES(3,NLOOPGROUPS)
	  LOGICAL S(NLOOPGROUPS)
      common/LOOPRES/LOOPRES,S
C ----------
C BEGIN CODE
C ----------
	  IF(abs(reference_value).eq.0.0d0) then
	    IsZero=.FALSE.
		write(*,*) '##E02 ERRROR Reference value for comparison is zero.'
		STOP
	  else
	    IsZero=((abs(toTest)/abs(reference_value)).lt.ZEROTHRES)
	  endif

	  IF((.NOT.ISZERO).AND.(loop.NE.-1).AND.(.NOT.S(loop))) THEN
	    write(*,*) '##W01 WARNING Contribution ',loop,' is detected as contributing with CR=',(abs(toTest)/abs(reference_value)),' but is unstable.' 
	  ENDIF

	  end

      integer function ISSAME(resA,resB,ref)
	  IMPLICIT NONE
C     This function compares the result from two different helicity configuration A and B
C     It returns 0 if they are not related and (+/-wgt) if A=(+/-wgt)*B.
C     For now, the only wgt implemented is the integer 1.
C  
C CONSTANTS
C
      integer N_WGT_TO_TRY
	  parameter (N_WGT_TO_TRY=2)
C  
C ARGUMENTS 
C  
	  %(real_dp_format)s resA(3), resB(3)
	  %(real_dp_format)s ref
C  
C LOCAL VARIABLES
C
      LOGICAL ISZERO
      integer I,J
      integer WGT_TO_TRY(N_WGT_TO_TRY)
	  data WGT_TO_TRY/1,-1/
C ----------
C BEGIN CODE
C ----------
      ISSAME=0

	  DO I=1,N_WGT_TO_TRY
	    DO J=1,3
		  IF (IsZero(ABS(resB(J)),ref,-1)) THEN
		    IF(.NOT.IsZero(ABS(resB(J))+ABS(resA(J)),ref,-1)) THEN
		      GOTO 1231
			ENDIF
		  ELSEIF(.NOT.IsZero((resA(J)/resB(J))-DBLE(WGT_TO_TRY(I)),ref,-1)) THEN
		    GOTO 1231		    
		  ENDIF
		ENDDO
		ISSAME = WGT_TO_TRY(I)
		RETURN
 1231   CONTINUE
      ENDDO
	  END

	  SUBROUTINE IMPROVE_PS_POINT_PRECISION(P)
C  
C CONSTANTS 
C 
	  INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
	  %(real_dp_format)s     ZERO
	  PARAMETER (ZERO=0.0d0)
C  
C ARGUMENTS 
C
      %(real_dp_format)s P(0:3,NEXTERNAL)
C
C LOCAL VARIABLES 
C
      INTEGER I,J
	  %(real_dp_format)s BUFF,REF
	  %(real_dp_format)s MASSES(NEXTERNAL-1)	  
C  
C GLOBAL VARIABLES
C 
      include 'coupl.inc'

	  %(masses_def)s

	  DO I=0,3
	    REF=ZERO
		BUFF=ZERO
		DO J=1,NEXTERNAL
		  REF=REF+ABS(P(I,J))
		ENDDO
        DO J=1,%(n_initial)d		
		  BUFF=BUFF-P(I,J)
		ENDDO
        DO J=%(n_initial)d+1,NEXTERNAL	
		  BUFF=BUFF+P(I,J)
		ENDDO
		IF ((BUFF/REF).GT.1.0d-9) THEN
		  STOP 'Four-momentum conservation is not accurate enough.'		  
		ENDIF
	  ENDDO

	  DO I=1,NEXTERNAL-1
	    REF=ABS(P(0,I))+ABS(P(1,I))+ABS(P(2,I))+ABS(P(3,I))
		IF ((SQRT(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2-MASSES(I)**2)/REF).GT.1.0d-4) THEN
		  STOP 'Four-momentum conservation is not accurate enough.'
		ELSE
	      P(3,I)=SIGN(SQRT(P(0,I)**2-P(1,I)**2-P(2,I)**2-MASSES(I)**2),P(3,I))
		ENDIF
	  ENDDO

	  DO I=0,3
	    BUFF=0.0d0 
        DO J=1,%(n_initial)d
		  BUFF=BUFF+P(I,J)
		ENDDO
		DO J=%(n_initial)d+1,NEXTERNAL-1
		  BUFF=BUFF-P(I,J)		
		ENDDO
		P(I,NEXTERNAL)=BUFF
	  ENDDO

	  END
