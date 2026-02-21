      SUBROUTINE SMATRIX%(proc_prefix)s(P,ANS_SUMMED)
C  
%(info_lines)s
C
C
C     Return the sum of the split orders which are required in orders.inc (NLO_ORDERS)
C  
C  
%(process_lines)s
C
C
C CONSTANTS
C 
      implicit none
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
      INTEGER I,J
      REAL*8 ANS(0:NSQAMPSO)
      LOGICAL KEEP_ORDER(NSQAMPSO), FIRSTTIME
      include 'orders.inc'
      data keep_order / NSQAMPSO*.TRUE. /
      data firsttime / .TRUE. /
      integer amp_orders(nsplitorders)
      double precision ans_max, tiny
      parameter (tiny = 1d-12)
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX%(proc_prefix)s
      integer orders_to_amp_split_pos
C
C BEGIN CODE
C

C look for orders which match the nlo order constraint 

if (firsttime) then
 do i = 1, nsqampso
  do j = 1, nsplitorders
   if(GETORDPOWFROMINDEX%(proc_prefix)s(j, i) .gt. nlo_orders(j)) then
    keep_order(i) = .false.
    EXIT 
   endif
  enddo
  if (keep_order(i)) then
   write(*,*) 'REAL %(proc_prefix)s: keeping split order ', i
  else
   write(*,*) 'REAL %(proc_prefix)s: not keeping split order ', i
  endif
 enddo
 firsttime = .false.
endif

CALL SMATRIX%(proc_prefix)s_SPLITORDERS(P,ANS)
ans_summed = 0d0
ans_max = 0d0

C reset the amp_split array
amp_split(1:amp_split_size) = 0d0

do i = 1, nsqampso
 ans_max = max(dabs(ans(i)),ans_max)
enddo

do i = 1, nsqampso
 if (keep_order(i)) then
    ANS_SUMMED = ANS_SUMMED + ANS(I)
C   keep track of the separate pieces correspoinding to different coupling combinations
    do j = 1, nsplitorders
      amp_orders(j) = GETORDPOWFROMINDEX%(proc_prefix)s(j, i)
    enddo
    if (abs(ans(i)).gt.ans_max*tiny) amp_split(orders_to_amp_split_pos(amp_orders)) = ans(i)
 endif
enddo

C avoid fake non-zeros
if (dabs(ans_summed).lt.tiny*ans_max) ans_summed=0d0

wgt_ME_real = ans_summed

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
      INTEGER IHEL,IDEN,I,J,T_IDENT(NCOMB)
      REAL*8 T(0:NSQAMPSO),T_SAVE(NCOMB,0:NSQAMPSO)
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
            DO I=1,IHEL-1
               IF (T(0).EQ.0D0) EXIT
               IF (T_SAVE(I,0).EQ.0D0) CYCLE
                do j = 0, nsqampso
                  IF (ABS(T(j)/T_SAVE(I,j)-1D0) .GT. 1D-12) goto 444
                enddo
                T_IDENT(IHEL) = I
 444            continue
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
	  DO I=1,NSQAMPSO !keep loop from 1!!
             ANS(I)=ANS(I)+T(I)
	  ENDDO
          IF (T(0) .NE. 0D0 .AND. .NOT. GOODHEL(IHEL)) THEN
            GOODHEL(IHEL)=.TRUE.
          ENDIF
        ENDIF
      ENDDO
      DO I=1,NSQAMPSO			
        ANS(I)=ANS(I)/DBLE(IDEN)
        ANS(0)=ANS(0)+ANS(I)
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
      REAL*8 RES(0:NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N
      INTEGER IC(NEXTERNAL)
      data ic /nexternal*1/
      INTEGER CF(NCOLOR*(NCOLOR+1))
      INTEGER CF_INDEX, DENOM
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO), W(%(wavefunctionsize)d,NWAVEFUNCS)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
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
      jamp(:,:) = (0d0,0d0)
%(helas_calls)s
%(jamp_lines)s

      DO I=0,NSQAMPSO
         RES(I)=0d0
      ENDDO
      DO M = 1, NAMPSO
      CF_INDEX = 0
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
              CF_INDEX = CF_INDEX+1
              ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J,M)
          ENDDO
	  DO N = 1, NAMPSO
            RES(SQSOINDEX%(proc_prefix)s(M,N)) = RES(SQSOINDEX%(proc_prefix)s(M,N)) + ZTEMP*DCONJG(JAMP(I,N))
          ENDDO
      ENDDO

	  DO N = 1, NAMPSO
            RES(SQSOINDEX%(proc_prefix)s(M,N)) = RES(SQSOINDEX%(proc_prefix)s(M,N))/DENOM 
          ENDDO
      ENDDO

      DO I=1,NSQAMPSO
         RES(0)=RES(0)+RES(I)
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
      implicit none
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
      implicit none
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
      implicit none
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER IORDER, INDX
C
C LOCAL VARIABLES
C
      integer i
      INTEGER SQSPLITORDERS(NSQAMPSO,NSPLITORDERS)
%(sqsplitorders)s
C
C BEGIN CODE
C
      IF (IORDER.GT.NSPLITORDERS.OR.IORDER.LT.1) THEN
      WRITE(*,*) "INVALID IORDER %(proc_prefix)s", IORDER
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSPLITORDERS
      STOP
      ENDIF

      IF (INDX.GT.NSQAMPSO.OR.INDX.LT.1) THEN
      WRITE(*,*) "INVALID INDX %(proc_prefix)s", INDX
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSQAMPSO
      STOP
      ENDIF

      GETORDPOWFROMINDEX%(proc_prefix)s=SQSPLITORDERS(INDX, IORDER)
      END



      SUBROUTINE GET_NSQSO_REAL%(proc_prefix)s(NSQSO)
C
C     Simple subroutine returning the number of squared split order
C     contributions returned in ANS when calling SMATRIX_SPLITORDERS
C
      implicit none
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSQSO

	  NSQSO=NSQAMPSO

      END
