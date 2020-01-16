      SUBROUTINE SMATRIX_%(N_me)d(P,ANS)
C  
%(info_lines)s
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
      include 'nexternal.inc'
      INTEGER     NCOMB         
      PARAMETER ( NCOMB=%(ncomb)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real
C  
C LOCAL VARIABLES 
C  
      INTEGER IHEL,IDEN,I,T_IDENT(NCOMB)
      REAL*8 MATRIX_%(N_me)d
      REAL*8 T,T_SAVE(NCOMB)
      SAVE T_SAVE,T_IDENT
      INTEGER NHEL(NEXTERNAL,NCOMB)
%(helicity_lines)s
      LOGICAL GOODHEL(NCOMB)
      DATA GOODHEL/NCOMB*.FALSE./
      INTEGER NTRY
      DATA NTRY/0/
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      ANS = 0D0
      DO IHEL=1,NCOMB
        IF (GOODHEL(IHEL) .OR. NTRY .LT. 2) THEN
          IF (NTRY.LT.2) THEN
C for the first ps-point, check for helicities that give
C identical matrix elements
            T=MATRIX_%(N_me)d(P ,NHEL(1,IHEL))            
            T_SAVE(IHEL)=T
            T_IDENT(IHEL)=-1
            DO I=1,IHEL-1
               IF (T.EQ.0D0) EXIT
               IF (T_SAVE(I).EQ.0D0) CYCLE
               IF (ABS(T/T_SAVE(I)-1D0) .LT. 1D-12) THEN
C                  WRITE (*,*) 'FOUND IDENTICAL',T,IHEL,T_SAVE(I),I
                  T_IDENT(IHEL) = I
               ENDIF
            ENDDO
          ELSE
            IF (T_IDENT(IHEL).GT.0) THEN
C if two helicity states are identical, dont recompute
              T=T_SAVE(T_IDENT(IHEL))
              T_SAVE(IHEL)=T
            ELSE
              T=MATRIX_%(N_me)d(P ,NHEL(1,IHEL))            
              T_SAVE(IHEL)=T
            ENDIF
          ENDIF
C add to the sum of helicities
          ANS=ANS+T
          IF (T .NE. 0D0 .AND. .NOT. GOODHEL(IHEL)) THEN
            GOODHEL(IHEL)=.TRUE.
          ENDIF
        ENDIF
      ENDDO
      ANS=ANS/DBLE(IDEN)
      wgt_ME_real=ans
      END
       
       
      REAL*8 FUNCTION MATRIX_%(N_me)d(P,NHEL)
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
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      include 'nexternal.inc'
      include 'coupl.inc'
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      INTEGER IC(NEXTERNAL)
      data ic /nexternal*1/
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR), W(%(wavefunctionsize)d,NWAVEFUNCS)
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
%(helas_calls)s
%(jamp_lines)s
      MATRIX_%(N_me)d = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATRIX_%(N_me)d = MATRIX_%(N_me)d+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
      END
