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
      DO IHEL=1,NCOMB
	     IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN
           IF (GOODHEL(IHEL) .OR. NTRY .LT. 2) THEN
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
      INTEGER CF(NCOLOR*(NCOLOR+1))
      INTEGER CF_INDEX, DENOM
      COMPLEX*16 AMP(NGRAPHS)
      COMPLEX*16 JAMP(NCOLOR,NAMPSO), LNJAMP(NCOLOR,NAMPSO)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP  
      COMPLEX*16 W(18,NWAVEFUNCS)
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
        CF_INDEX = 0
        DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
              CF_INDEX = CF_INDEX +1
              ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J,M)
          ENDDO
		  DO N = 1, NAMPSO
            RES(%(proc_prefix)sSQSOINDEX(M,N)) = RES(%(proc_prefix)sSQSOINDEX(M,N)) + ZTEMP*DCONJG(JAMP(I,N))
		  ENDDO
        ENDDO
		DO N = 1, NAMPSO
            RES(%(proc_prefix)sSQSOINDEX(M,N)) = RES(%(proc_prefix)sSQSOINDEX(M,N))/DENOM
		ENDDO
	  ENDDO

      END

      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      

       SUBROUTINE %(proc_prefix)sBORN(P,NHEL)
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
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N
      COMPLEX*16 ZTEMP
c      REAL*8 CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS)
      COMPLEX*16 JAMP(NCOLOR,NAMPSO), LNJAMP(NCOLOR,NAMPSO)
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP
      COMPLEX*16 W(18,NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/

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
      DO I=1,NEXTERNAL	
	IC(I) = 1
      ENDDO	


%(helas_calls)s
%(jamp_lines)s

      END

      SUBROUTINE %(proc_prefix)sGET_JAMP(njamp, ONEJAMP)

      INTEGER     NCOLOR, NJAMP
      PARAMETER (NCOLOR=%(ncolor)d) 
      INTEGER NAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d)
      COMPLEX*16  JAMP(NCOLOR,NAMPSO), ONEJAMP
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP
	
	ONEJAMP = JAMP(njamp+1,1) ! +1 since njamp start at zero (c convention)
	END

       SUBROUTINE %(proc_prefix)sGET_LNJAMP(njamp, ONEJAMP)

      INTEGER     NCOLOR, NJAMP
      PARAMETER (NCOLOR=%(ncolor)d) 
      INTEGER NAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d)
      COMPLEX*16  JAMP(NCOLOR,NAMPSO), LNJAMP(NCOLOR,NAMPSO), ONEJAMP
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP
	
	ONEJAMP = LNJAMP(njamp+1,1) ! +1 since njamp start at zero (c convention)
	END
	
	
	
	
     SUBROUTINE %(proc_prefix)sGET_NCOLOR(in1, in2, out)

	integer in1, in2, out
	%(color_information)s
     END

     SUBROUTINE %(proc_prefix)sGET_NCOL(ncol)
	integer ncol
	ncol = %(ncolor)d
	return
	end      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
