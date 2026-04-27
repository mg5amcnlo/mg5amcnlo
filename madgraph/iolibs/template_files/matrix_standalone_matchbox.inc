      SUBROUTINE %(proc_prefix)sSMATRIX(P,ANS)
C  
%(info_lines)s
C 
C MadGraph5_aMC@NLO StandAlone Version
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
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T
      REAL*8 %(proc_prefix)sMATRIX
      INTEGER IHEL,IDEN, I
      INTEGER JC(NEXTERNAL)
      LOGICAL GOODHEL(NCOMB)
      DATA NTRY/0/
      DATA GOODHEL/NCOMB*.FALSE./
%(helicity_lines)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
      ANS = 0D0
          DO IHEL=1,NCOMB
             IF (GOODHEL(IHEL) .OR. NTRY .LT. 20) THEN
                 T=%(proc_prefix)sMATRIX(P ,NHEL(1,IHEL),JC(1))            
               ANS=ANS+T
               IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL)) THEN
                   GOODHEL(IHEL)=.TRUE.
               ENDIF
             ENDIF
          ENDDO
      ANS=ANS/DBLE(IDEN)
      END
       
       
      REAL*8 FUNCTION %(proc_prefix)sMATRIX(P,NHEL,IC)
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
      INTEGER I,J
      COMPLEX*16 ZTEMP
      INTEGER  CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER DENOM, CF_INDEX
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR), LNJAMP(NCOLOR)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
      COMPLEX*16 W(18,NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
%(global_variable)s
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
%(helas_calls)s
%(jamp_lines)s

      %(proc_prefix)sMATRIX = 0.D0
      CF_INDEX =0
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
              CF_INDEX = CF_INDEX + 1
              ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J)
          ENDDO
          %(proc_prefix)sMATRIX = %(proc_prefix)sMATRIX+ZTEMP*DCONJG(JAMP(I))   
      ENDDO
      %(proc_prefix)sMATRIX = %(proc_prefix)sMATRIX/DENOM
%(amp2_lines)s
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
      INTEGER I,J
      COMPLEX*16 ZTEMP
      INTEGER CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER CF_INDEX, DENOM
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR), LNJAMP(NCOLOR)
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP

      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
      COMPLEX*16 W(18,NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
%(global_variable)s
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
      COMPLEX*16  JAMP(NCOLOR), ONEJAMP
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP
	
	ONEJAMP = JAMP(njamp+1) ! +1 since njamp start at zero (c convention)
	END

       SUBROUTINE %(proc_prefix)sGET_LNJAMP(njamp, ONEJAMP)

      INTEGER     NCOLOR, NJAMP
      PARAMETER (NCOLOR=%(ncolor)d) 
      COMPLEX*16  JAMP(NCOLOR), LNJAMP(NCOLOR), ONEJAMP
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP
	
	ONEJAMP = LNJAMP(njamp+1) ! +1 since njamp start at zero (c convention)
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
