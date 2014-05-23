C Set of functions to handle the array indices of the split orders


      INTEGER FUNCTION %(proc_prefix)sSQSOINDEX%(proc_id)s(ORDERINDEXA, ORDERINDEXB)
C
C This functions plays the role of the interference matrix. It can be hardcoded or 
C made more elegant using hashtables if its execution speed ever becomes a relevant
C factor. From two split order indices, it return the corresponding index in the squared 
c order canonical ordering.
C
C CONSTANTS
C      

      INTEGER    NSO, NSQUAREDSO, NAMPSO
	  PARAMETER (NSO=%(nSplitOrders)d, NSQUAREDSO=%(nSqAmpSplitOrders)d, NAMPSO=%(nAmpSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER ORDERINDEXA, ORDERINDEXB
C
C LOCAL VARIABLES
C
      INTEGER I, SQORDERS(NSO)
      INTEGER AMPSPLITORDERS(NAMPSO,NSO)
	  %(ampsplitorders)s
C
C FUNCTION
C
      INTEGER %(proc_prefix)sSOINDEX_FOR_SQUARED_ORDERS%(proc_id)s
C
C BEGIN CODE
C
      DO I=1,NSO
	    SQORDERS(I)=AMPSPLITORDERS(ORDERINDEXA,I)+AMPSPLITORDERS(ORDERINDEXB,I)
	  ENDDO
	  %(proc_prefix)sSQSOINDEX%(proc_id)s=%(proc_prefix)sSOINDEX_FOR_SQUARED_ORDERS%(proc_id)s(SQORDERS)
	  END

      INTEGER FUNCTION %(proc_prefix)sSOINDEX_FOR_SQUARED_ORDERS%(proc_id)s(ORDERS)
C
C This functions returns the integer index identifying the squared split orders list passed in argument which corresponds to the values of the following list of couplings (and in this order).
c %(split_order_str_list)s
C
C CONSTANTS
C
      INTEGER    NSO, NSQSO, NAMPSO
	  PARAMETER (NSO=%(nSplitOrders)d, NSQSO=%(nSqAmpSplitOrders)d, NAMPSO=%(nAmpSplitOrders)d)
C
C ARGUMENTS
C
      INTEGER ORDERS(NSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J
	  INTEGER SQSPLITORDERS(NSQSO,NSO)
	  %(sqsplitorders)s
C
C BEGIN CODE
C
      DO I=1,NSQSO
	    DO J=1,NSO
		  IF (ORDERS(J).NE.SQSPLITORDERS(I,J)) GOTO 1009
		ENDDO
		%(proc_prefix)sSOINDEX_FOR_SQUARED_ORDERS%(proc_id)s = I
		RETURN
1009    CONTINUE
	  ENDDO

	  WRITE(*,*) 'ERROR:: Stopping function %(proc_prefix)sSOINDEX_FOR_SQUARED_ORDERS'
	  WRITE(*,*) 'Could not find squared orders ',(ORDERS(I),I=1,NSO)
	  STOP

      END

      SUBROUTINE %(proc_prefix)sGET_NSQSO_BORN%(proc_id)s(NSQSO)
C
C     Simple subroutine returning the number of squared split order
C     contributions returned when calling smatrix_split_orders 
C

      INTEGER    NSQUAREDSO
	  PARAMETER  (NSQUAREDSO=%(nSqAmpSplitOrders)d)

	  INTEGER NSQSO

      NSQSO=NSQUAREDSO

      END
