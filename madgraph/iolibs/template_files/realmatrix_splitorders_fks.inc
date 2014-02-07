      SUBROUTINE SMATRIX%(proc_prefix)s(P,ANS_SUMMED)
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
	  CALL SMATRIX%(proc_prefix)s_SPLITORDERS(P,ANS)
	  ANS_SUMMED=ANS(0)

END



      SUBROUTINE SMATRIX%(proc_prefix)s_SPLITORDERS(P,ANS)
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
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER IHEL,IDEN,I,T_IDENT(NCOMB)
      REAL*8 MATRIX_%(proc_prefix)s
      REAL*8 T,T_SAVE(NCOMB,0:NSQAMPSO)
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
      DO I=0,NSQAMPSO
      	ANS(I) = 0D0	  
      ENDDO
      DO IHEL=1,NCOMB
        IF (GOODHEL(IHEL) .OR. NTRY .LT. 2) THEN
          IF (NTRY.LT.2) THEN
C for the first ps-point, check for helicities that give
C identical matrix elements
            call MATRIX_%(proc_prefix)s(P ,NHEL(1,IHEL),T)            
            DO I=0,NSQAMPSO
                T_SAVE(IHEL,I)=T(I)
            ENDDO
            T_IDENT(IHEL)=-1
	    BUFF=0D0
	    DO I=1,NSQAMPSO			
                ANS(I)=ANS(I)+T(I)
		BUFF=BUFF+T(I)
	    ENDDO
            DO I=1,IHEL-1
               IF (T(0).EQ.0D0) EXIT
               IF (T_SAVE(I,0).EQ.0D0) CYCLE
               IF (ABS(T(0)/T_SAVE(I,0)-1D0) .LT. 1D-12) THEN
C                  WRITE (*,*) 'FOUND IDENTICAL',T,IHEL,T_SAVE(I),I
                  T_IDENT(IHEL) = I
               ENDIF
            ENDDO
          ELSE
            IF (T_IDENT(IHEL).GT.0) THEN
C if two helicity states are identical, dont recompute
            DO I=0,NSQAMPSO
              T(I)=T_SAVE(T_IDENT(IHEL),I)
              T_SAVE(IHEL,I)=T(I)
            ENDDO
            ELSE
              CALL MATRIX_%(proc_prefix)s(P ,NHEL(1,IHEL),T)            
              DO I=0,NSQAMPSO
                T_SAVE(IHEL,I)=T(I)
              ENDDO
            ENDIF
          ENDIF
C add to the sum of helicities
	  DO I=0,NSQAMPSO			
             ANS(I)=ANS(I)+T(I)
	  ENDDO
          IF (T(0) .NE. 0D0 .AND. .NOT. GOODHEL(IHEL)) THEN
            GOODHEL(IHEL)=.TRUE.
          ENDIF
        ENDIF
      ENDDO
      DO I=0,NSQAMPSO			
        ANS(I)=ANS(I)/DBLE(IDEN)
      ENDDO
      END
       

      SUBROUTINE MATRIX_%(proc_prefix)s(P,NHEL,RES)
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
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
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
      REAL*8 RES(NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N
      INTEGER IC(NEXTERNAL)
      data ic /nexternal*1/
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO), W(%(wavefunctionsize)d,NWAVEFUNCS)
C
C FUNCTION
C
      INTEGER SQSOINDEX%(proc_prefix)s
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
%(helas_calls)s
%(jamp_lines)s

      MATRIX_%(proc_prefix)s = 0.D0 
      DO M = 1, NAMPSO
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J,M)
          ENDDO
	  DO N = 1, NAMPSO
            RES(SQSOINDEX%(proc_prefix)s(M,N)) = RES(SQSOINDEX%(proc_prefix)s(M,N)) + ZTEMP*DCONJG(JAMP(I,N))/DENOM(I)
          ENDDO
      ENDDO
      ENDDO

      END

C
C Helper functions to deal with the split orders.
C

      INTEGER FUNCTION SQSOINDEX%(proc_prefix)s(AMPORDERA,AMPORDERB)
C
C This functions plays the role of the interference matrix. It can be hardcoded or 
C made more elegant using hashtables if its execution speed ever becomes a relevant
C factor. From two split order indices of the jamps, it return the corresponding
C index in the squared order canonical ordering.
C
C CONSTANTS
C      
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER AMPORDERA, AMPORDERB
C
C LOCAL VARIABLES
C
      INTEGER I, SQORDERS(NSPLITORDERS)
      INTEGER AMPSPLITORDERS(NAMPSO,NSPLITORDERS)
	  %(ampsplitorders)s
C
C FUNCTION
C
      INTEGER SQSOINDEX_FROM_ORDERS%(proc_prefix)s
C
C BEGIN CODE
C
      DO I=1,NSPLITORDERS
	    SQORDERS(I)=AMPSPLITORDERS(AMPORDERA,I)+AMPSPLITORDERS(AMPORDERB,I)
	  ENDDO
	  SQSOINDEX%(proc_prefix)s=SQSOINDEX_FROM_ORDERS%(proc_prefix)s(SQORDERS)
     END



      INTEGER FUNCTION SQSOINDEX_FROM_ORDERS%(proc_prefix)s(ORDERS)
C
C From a list of values for the split orders, this function returns the
c corresponding index in the squared orders canonical ordering.
C
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER ORDERS(NSPLITORDERS)
C
C LOCAL VARIABLES
C
      INTEGER I,J
      INTEGER SQSPLITORDERS(NSQAMPSO,NSPLITORDERS)
%(sqsplitorders)s
C
C BEGIN CODE
C
      DO I=1,NSQAMPSO
	    DO J=1,NSPLITORDERS
		  IF (ORDERS(J).NE.SQSPLITORDERS(I,J)) GOTO 1009
		ENDDO
		SQSOINDEX_FROM_ORDERS%(proc_prefix)s = I
		RETURN
1009    CONTINUE
	  ENDDO

	  WRITE(*,*) 'ERROR:: Stopping function sqsoindex_from_orders'
	  WRITE(*,*) 'Could not find squared orders ',(ORDERS(I),I=1,NSPLITORDERS)
	  STOP

	  END



      INTEGER FUNCTION GETORDPOWFROMINDEX%(proc_prefix)s(IORDER, INDX)
C
C Return the power of the IORDER-th order appearing at position INDX
C in the split-orders output
C
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER IORDER, INDEX
C
C LOCAL VARIABLES
C
      INTEGER SQSPLITORDERS(NSQAMPSO,NSPLITORDERS)
%(sqsplitorders)s
C
C BEGIN CODE
C
      IF (IORDER.GT.NSPLITORDERS.OR.IORDER.LT.1) THEN
      WRITE(*,*) "INVALID IORDER ", IORDER
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSPLITORDERS
      STOP
      ENDIF

      IF (INDEX.GT.NSQAMPSO.OR.INDX.LT.1) THEN
      WRITE(*,*) "INVALID INDEX ", INDX
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSQAMPSO
      STOP
      ENDIF

      GETORDPOWFROMINDEX%(proc_prefix)s=SPLITORDERS(INDX, IORDER)
      END



      SUBROUTINE GET_NSQSO_REAL%(proc_prefix)s(NSQSO)
C
C     Simple subroutine returning the number of squared split order
C     contributions returned in ANS when calling SMATRIX_SPLITORDERS
C
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSQSO

	  NSQSO=NSQAMPSO

      END
