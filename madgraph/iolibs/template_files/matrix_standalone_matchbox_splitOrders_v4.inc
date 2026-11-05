      SUBROUTINE %(proc_prefix)sSMATRIX(P, FLAV_IDX, ANS_SUMMED)
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
C     FLAV_IDX selects one column of the flavor table, i.e. one of the
C     flavor combinations this (possibly merged) matrix element covers.
C     The canonical flavor is column 1.
      INTEGER FLAV_IDX
C
C VARIABLES
C
      INTEGER I
	  REAL*8 ANS(0:NSQAMPSO)
C
C BEGIN CODE
C
	  CALL %(proc_prefix)sSMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
	  ANS_SUMMED=ANS(0)

	  END

      SUBROUTINE %(proc_prefix)sSMATRIXHEL(P,HEL, FLAV_IDX, ANS)
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
      INTEGER FLAV_IDX
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
	  COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
	  CALL %(proc_prefix)sSMATRIX(P, FLAV_IDX, ANS)
	  USERHEL=-1

	  END

      SUBROUTINE %(proc_prefix)sSMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
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
C     FLAV_IDX (the dummy argument) is declared with NFLAV below.
C
C LOCAL VARIABLES
C
      INTEGER NHEL(NEXTERNAL,NCOMB)
      REAL*8 T(NSQAMPSO), BUFF
      INTEGER IHEL,IDEN, I
      INTEGER JC(NEXTERNAL)
C     One good-helicity table and one warm-up counter per flavor: the
C     helicity configurations that vanish depend on the flavor (masses and
C     the diagrams the flavor mask keeps), so they cannot be shared.
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER NNTRY_FLAV, NGOODHEL_FLAV
      PARAMETER (NNTRY_FLAV=NFLAV)
      PARAMETER (NGOODHEL_FLAV=NCOMB*NFLAV)
      INTEGER FLAV_IDX
      INTEGER NTRY(NFLAV)
      LOGICAL GOODHEL(NCOMB,NFLAV)
      DATA NTRY/NNTRY_FLAV*0/
      DATA GOODHEL/NGOODHEL_FLAV*.FALSE./
%(helicity_lines)s
%(den_factor_line)s
C
C FUNCTIONS
C
      INTEGER %(proc_prefix)sBROKEN_SYM
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
	  DATA USERHEL/-1/
	  COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL

C ----------
C BEGIN CODE
C ----------
C     FLAV_IDX=0 (or out of range): GET_FLAVOR_INDEX could not resolve the
C     requested flavor -- it is not an allowed combination, so its matrix
C     element is identically zero. Short-circuit before touching the
C     1..NFLAV GOODHEL/NTRY arrays.
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) THEN
        DO I=0,NSQAMPSO
          ANS(I) = 0D0
        ENDDO
        RETURN
      ENDIF
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_IDX, FLAVOR)
      NTRY(FLAV_IDX)=NTRY(FLAV_IDX)+1
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
	  DO I=1,NSQAMPSO
      	ANS(I) = 0D0
	  ENDDO
      DO IHEL=1,NCOMB
	     IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN
           IF (GOODHEL(IHEL,FLAV_IDX) .OR. NTRY(FLAV_IDX) .LT. 2) THEN
              CALL %(proc_prefix)sMATRIX(P ,NHEL(1,IHEL),JC(1), FLAV_IDX, T)
	 	      BUFF=0D0
			  DO I=1,NSQAMPSO
                ANS(I)=ANS(I)+T(I)
			    BUFF=BUFF+T(I)
		      ENDDO
              IF (BUFF .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN
                GOODHEL(IHEL,FLAV_IDX)=.TRUE.
              ENDIF
            ENDIF
		  ENDIF
       ENDDO
	  ANS(0)=0.0d0
	  DO I=1,NSQAMPSO
C       IDEN averages the canonical flavor; BROKEN_SYM undoes the identical
C       -particle part of it for the flavors that are not actually identical.
        ANS(I)=ANS(I)/DBLE(IDEN)*%(proc_prefix)sBROKEN_SYM(FLAVOR)
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

      SUBROUTINE %(proc_prefix)sSMATRIXHEL_SPLITORDERS(P,HEL, FLAV_IDX, ANS)
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
      INTEGER FLAV_IDX
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
	  COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
	  CALL %(proc_prefix)sSMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
	  USERHEL=-1

	  END

      SUBROUTINE %(proc_prefix)sMATRIX(P,NHEL,IC,FLAV_IDX,RES)
      use model_object
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
      use aloha_object
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
C     The HELAS calls below take FLAVOR, which the block that follows
C     rebuilds from FLAV_IDX -- one column of the flavor table, i.e. one of
C     the flavor combinations this (possibly merged) matrix element covers.
      INTEGER FLAV_IDX
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
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
      INTEGER FLAVOR(NEXTERNAL)
%(flavor_mask_decl)s
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
%(flavor_mask_setup)s
%(helas_calls)s
%(jamp_lines)s

      RES(:) = 0.D0 
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
	  ENDDO
      RES(:) = RES(:)/DENOM
      END

      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      

       SUBROUTINE %(proc_prefix)sBORN(P,NHEL,FLAVOR)
      use model_object
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
      use aloha_object
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
      INTEGER NHEL(NEXTERNAL)
C     FLAVOR carries the per-leg flavor of the requested combination, in the
C     same convention as the non-split BORN of matrix_standalone_matchbox.inc.
C     It is resolved to a flavor-table column below.
      INTEGER FLAVOR(NEXTERNAL)
C
C LOCAL VARIABLES
C
      INTEGER IC(NEXTERNAL)
      INTEGER I,J,M,N
      COMPLEX*16 ZTEMP
      INTEGER CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER CF_INDEX, DENOM
      COMPLEX*16 AMP(NGRAPHS)
      COMPLEX*16 JAMP(NCOLOR,NAMPSO), LNJAMP(NCOLOR,NAMPSO)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
      INTEGER FLAV_IDX
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX
%(flavor_mask_decl)s

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


      FLAV_IDX = %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR)
C     Unresolved flavor (not an allowed combination): all color amplitudes
C     are zero.
      IF (FLAV_IDX.EQ.0) THEN
        DO M = 1, NAMPSO
          DO I = 1, NCOLOR
            JAMP(I,M) = (0D0, 0D0)
            LNJAMP(I,M) = (0D0, 0D0)
          ENDDO
        ENDDO
        RETURN
      ENDIF
%(flavor_mask_setup)s
%(helas_calls)s
%(jamp_lines)s

      END

      SUBROUTINE %(proc_prefix)sGET_JAMP(njamp, SOINDEX, ONEJAMP)

      INTEGER     NCOLOR, NJAMP, SOINDEX
      PARAMETER (NCOLOR=%(ncolor)d) 
      INTEGER NAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d)
      COMPLEX*16  JAMP(NCOLOR,NAMPSO), LNJAMP(NCOLOR,NAMPSO), ONEJAMP
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP
	
	  ONEJAMP = JAMP(njamp+1, SOINDEX) ! +1 since njamp start at zero (c convention)
	  END

       SUBROUTINE %(proc_prefix)sGET_LNJAMP(njamp, SOINDEX, ONEJAMP)

      INTEGER     NCOLOR, NJAMP, SOINDEX
      PARAMETER (NCOLOR=%(ncolor)d) 
      INTEGER NAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d)
      COMPLEX*16  JAMP(NCOLOR,NAMPSO), LNJAMP(NCOLOR,NAMPSO), ONEJAMP
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP
	
	  ONEJAMP = LNJAMP(njamp+1,SOINDEX) ! +1 since njamp start at zero (c convention)
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
      
    SUBROUTINE %(proc_prefix)sGET_MAXSOINDEX(out)
    
    integer out
    out  = %(nAmpSplitOrders)d
    return 
    end

    subroutine %(proc_prefix)s_get_chosen_so_config(M,N, out)

    integer M, N
    logical out
    PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
    LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
    DATA CHOSEN_SO_CONFIGS/%(chosen_so_configs)s/
    COMMON/%(proc_prefix)sCHOSEN_BORN_SQSO/CHOSEN_SO_CONFIGS

    out  = CHOSEN_SO_CONFIGS(%(proc_prefix)sSQSOINDEX(M,N))
    return
    end


%(broken_sym_function)s


%(flavor_index_function)s


%(flavor_array_function)s
      
