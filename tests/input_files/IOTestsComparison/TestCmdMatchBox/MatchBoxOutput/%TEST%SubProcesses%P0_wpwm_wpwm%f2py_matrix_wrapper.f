      SUBROUTINE PY_MG5_0_SMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
      INTEGER                 NCOMB         
      PARAMETER (             NCOMB=81)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)  

C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
	  INTEGER HEL

      CALL MG5_0_SMATRIXHEL(P,HEL,ANS)
	  END

      SUBROUTINE PY_MG5_0_SMATRIX(P,ANS)
C  
C 
C MadGraph5_aMC@NLO StandAlone Version
C 
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C  
C Process: w+ w- > w+ w- WEIGHTED<=4
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
	  INTEGER    NINITIAL 
      PARAMETER (NINITIAL=2)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
      call MG5_0_SMATRIX(P,ANS)
	  END
       
       
      REAL*8 FUNCTION PY_MG5_0_MATRIX(P,NHEL,IC)
C  
C
C Returns amplitude squared -- no average over initial state/symmetry factor
c for the point with external lines W(0:6,NEXTERNAL)
C  
C Process: w+ w- > w+ w- WEIGHTED<=4
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C
C  FUNCTIONS
C
      real*8 MG5_0_MATRIX
      PY_MG5_0_MATRIX = MG5_0_MATRIX(P,NHEL,IC)
      END

      SUBROUTINE PY_MG5_0_GET_value(P, ALPHAS, NHEL ,ANS)
      IMPLICIT NONE   
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER NHEL
      DOUBLE PRECISION ALPHAS 
CF2PY INTENT(OUT) :: ANS  
CF2PY INTENT(IN) :: NHEL   
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL) 
CF2PY INTENT(IN) :: ALPHAS
      call MG5_0_GET_value(P, ALPHAS, NHEL ,ANS)
      return 
      end

      SUBROUTINE PY_MG5_0_INITIALISEMODEL(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.    
      IMPLICIT NONE   
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH 
      call setpara(PATH)  !first call to setup the paramaters    
      return 
      end      

      LOGICAL FUNCTION PY_MG5_0_IS_BORN_HEL_SELECTED(HELID)
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=4)
C
C     ARGUMENTS
C
      INTEGER HELID
      LOGICAL MG5_0_IS_BORN_HEL_SELECTED
      PY_MG5_0_IS_BORN_HEL_SELECTED = MG5_0_IS_BORN_HEL_SELECTED(HELID)
      RETURN
      END
