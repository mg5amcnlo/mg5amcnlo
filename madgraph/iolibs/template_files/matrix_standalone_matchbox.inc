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
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMMON/%(proc_prefix)sJAMP/JAMP

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
      COMMON/%(proc_prefix)sJAMP/JAMP
	
	ONEJAMP = JAMP(njamp+1) ! +1 since njamp start at zero (c convention)
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
