      SUBROUTINE %(proc_prefix)sSMATRIX(P,ANS_SUMMED)
C
C     Simple routine wrapper to provide the same interface for
C     backward compatibility for usage without split orders.
C
C  
C CONSTANTS
C 
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C 
      REAL*8 P(0:3,NEXTERNAL), ANS_SUMMED
C
C VARIABLES
C
      INTEGER I
	  REAL*8 ANS(0:NSQAMPSO)
C
C BEGIN CODE
C
	  CALL %(proc_prefix)sSMATRIX_SPLITORDERS(P,ANS)
	  ANS_SUMMED=ANS(0)

	  END

      SUBROUTINE %(proc_prefix)sSMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB         
      PARAMETER (             NCOMB=%(ncomb)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
	  INTEGER HEL
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
	  COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
	  CALL %(proc_prefix)sSMATRIX(P,ANS)
	  USERHEL=-1

	  END

      SUBROUTINE %(proc_prefix)sSMATRIX_SPLITORDERS(P,ANS)
C  
%(info_lines)s
C 
C MadGraph StandAlone Version
C 
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB         
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER HELAVGFACTOR
	  PARAMETER (HELAVGFACTOR=%(hel_avg_factor)d)
	  LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
	  DATA CHOSEN_SO_CONFIGS/%(chosen_so_configs)s/
	  COMMON/%(proc_prefix)sCHOSEN_BORN_SQSO/CHOSEN_SO_CONFIGS
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T(NSQAMPSO), BUFF
      INTEGER IHEL,IDEN, I
      INTEGER JC(NEXTERNAL)
      LOGICAL GOODHEL(NCOMB)
      DATA NTRY/0/
      DATA GOODHEL/NCOMB*.FALSE./
%(helicity_lines)s
%(den_factor_line)s
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
	  DATA USERHEL/-1/	  	  
	  COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL

C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
	  DO I=1,NSQAMPSO
      	ANS(I) = 0D0	  
	  ENDDO
C     When spin-2 particles are involved, the Helicity filtering is dangerous for the 2->1 topology.
C     This is because depending on the MC setup the initial PS points have back-to-back initial states
C     for which some of the spin-2 helicity configurations are zero. But they are no longer zero
C     if the point is boosted on the z-axis. Remember that HELAS helicity amplitudes are no longer
C     lorentz invariant with expternal spin-2 particles (only the helicity sum is).
C     For this reason, we simply remove the filterin when there is only three external particles.
      IF (NEXTERNAL.LE.3) THEN
        DO IHEL=1,NCOMB
		  GOODHEL(IHEL)=.TRUE.
		ENDDO
	  ENDIF
      DO IHEL=1,NCOMB
	     IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN
           IF (GOODHEL(IHEL) .OR. NTRY .LT. 2 .OR.USERHEL.NE.-1) THEN
              CALL %(proc_prefix)sMATRIX(P ,NHEL(1,IHEL),JC(1), T)
	 	      BUFF=0D0
			  DO I=1,NSQAMPSO			
                ANS(I)=ANS(I)+T(I)
			    BUFF=BUFF+T(I)
		      ENDDO
              IF (BUFF .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL)) THEN
                GOODHEL(IHEL)=.TRUE.
              ENDIF
            ENDIF
		  ENDIF
       ENDDO
	  ANS(0)=0.0d0
	  DO I=1,NSQAMPSO
        ANS(I)=ANS(I)/DBLE(IDEN)
		IF (CHOSEN_SO_CONFIGS(I)) THEN
		  ANS(0)=ANS(0)+ANS(I)
		ENDIF
	  ENDDO
	  IF(USERHEL.NE.-1) THEN
		ANS(0)=ANS(0)*HELAVGFACTOR
		DO I=1,NSQAMPSO
          ANS(I)=ANS(I)*HELAVGFACTOR
		ENDDO
	  ENDIF
      END

      SUBROUTINE %(proc_prefix)sSMATRIXHEL_SPLITORDERS(P,HEL,ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB         
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
	  INTEGER HEL
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
	  COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
	  CALL %(proc_prefix)sSMATRIX_SPLITORDERS(P,ANS)
	  USERHEL=-1

	  END
 
      SUBROUTINE %(proc_prefix)sMATRIX(P,NHEL,IC,RES)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d) 
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
	  REAL*8 RES(NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS)
	  COMPLEX*16 JAMP(NCOLOR,NAMPSO)
      COMPLEX*16 W(20,NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C
C FUNCTION
C
      INTEGER %(proc_prefix)sSQSOINDEX
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
%(helas_calls)s
%(jamp_lines)s

      RES = 0.D0 
	  DO M = 1, NAMPSO
        DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J,M)
          ENDDO
		  DO N = 1, NAMPSO
            RES(%(proc_prefix)sSQSOINDEX(M,N)) = RES(%(proc_prefix)sSQSOINDEX(M,N)) + ZTEMP*DCONJG(JAMP(I,N))/DENOM(I)
		  ENDDO
        ENDDO
	  ENDDO

      END

      SUBROUTINE %(proc_prefix)sGET_ME(P, ALPHAS, NHEL ,ANS)
      IMPLICIT NONE   
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER NHEL
      DOUBLE PRECISION ALPHAS 
      real*8 pi     
CF2PY INTENT(OUT) :: ANS  
CF2PY INTENT(IN) :: NHEL   
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL) 
CF2PY INTENT(IN) :: ALPHAS
C     ROUTINE FOR F2PY to read the benchmark point.    
C     the include file with the values of the parameters and masses 
      include "coupl.inc"
      
      pi = 3.141592653589793d0
      G = 2* DSQRT(ALPHAS*pi)
      call UPDATE_AS_PARAM()
      if (NHEL.ne.0) then
        CALL %(proc_prefix)sSMATRIXHEL(P, NHEL, ANS)
      else
        CALL %(proc_prefix)sSMATRIX(P, ANS)
      endif
      return 
      end

      SUBROUTINE %(proc_prefix)sINITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.    
      IMPLICIT NONE   
      CHARACTER*180 PATH
CF2PY INTENT(IN) :: PATH 
      call setpara(PATH)  !first call to setup the paramaters    
      return 
      end      
