      SUBROUTINE SBORN%(proc_prefix)s(P,ANS_SUMMED)
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
	  COMPLEX*16 ANS(2,0:NSQAMPSO)
C
C BEGIN CODE
C
	  CALL SBORN%(proc_prefix)s_SPLITORDERS(P,ANS)
	  ANS_SUMMED=ANS(1,0)

END



SUBROUTINE SBORN%(proc_prefix)s_SPLITORDERS(P1,ANS)
C  
%(info_lines)s
C
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P1(0:3,NEXTERNAL-1)
C  
%(process_lines)s
C
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include "nexternal.inc"
      include "born_nhel.inc"
      include "genps.inc"
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  %(ncomb)d )
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*%(nconfs)d)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL-1)
      COMPLEX*16 ANS(2,0:NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER IHEL,IDEN,i,j,jj,glu_ij
      REAL*8 borns(2,0:NSQAMPSO)
      COMPLEX*16 BORNTILDE
      INTEGER NTRY(%(nconfs)d)
      DATA NTRY /%(nconfs)d*0/
      COMPLEX*16 T(2,0:NSQAMPSO)
      INTEGER NHEL(NEXTERNAL-1,NCOMB)
%(helicity_lines)s
%(den_factor_lines)s
%(ij_lines)s
C  
C GLOBAL VARIABLES
C  
      Double Precision amp2(maxamps), jamp2(0:maxamps,0:NSQAMPSO)
      common/to_amps/  amp2,       jamp2
      DATA jamp2(0,0) /   %(ncolor)d/          
      LOGICAL GOODHEL(NCOMB,%(nconfs)d)
      common /c_goodhel/goodhel
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      double precision savemom(nexternal-1,2)
      common/to_savemom/savemom
      double precision hel_fac
      integer get_hel,skip(%(nconfs)d)
      common/cBorn/hel_fac,get_hel,skip
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
C ----------
C BEGIN CODE
C ----------
      iden=iden_values(nfksprocess)
      glu_ij = ij_values(nfksprocess)
      NTRY(nFKSprocess)=NTRY(nFKSprocess)+1
      if (NTRY(nFKSprocess).lt.2) then
         skip(nFKSprocess)=1			 
         do while(nhel(glu_ij ,skip(nFKSprocess)).ne.1)
            skip(nFKSprocess)=skip(nFKSprocess)+1
         enddo
         skip(nFKSprocess)=skip(nFKSprocess)-1
      endif
      DO JJ=1,NGRAPHS
          amp2(jj)=0d0
      ENDDO
      DO I=0,NSQAMPSO
      DO JJ=1,int(jamp2(0,0))
          jamp2(jj,I)=0d0
      ENDDO
      ENDDO
      if (calculatedBorn) then
         do j=1,nexternal-1
            if (savemom(j,1).ne.p1(0,j) .or. savemom(j,2).ne.p1(3,j)) then
               calculatedBorn=.false.
               write (*,*) "momenta not the same in Born"
	       stop
            endif
         enddo
      endif
      if (.not.calculatedBorn) then
         do j=1,nexternal-1
            savemom(j,1)=p1(0,j)
            savemom(j,2)=p1(3,j)
         enddo
         do j=1,max_bhel
            do jj=1,ngraphs
               saveamp(jj,j)=(0d0,0d0)
            enddo
         enddo
      endif
      DO I=0,NSQAMPSO
          ANS(1,I) = 0D0
          ANS(2,I) = 0D0
      ENDDO
      hel_fac=1d0
      DO IHEL=1,NCOMB
        if (nhel(glu_ij,ihel).le.0) then
          IF ((GOODHEL(IHEL,nFKSprocess) .OR. GOODHEL(IHEL+SKIP(nFKSprocess),nFKSprocess) .OR. NTRY(nFKSprocess) .LT. 2) ) THEN
            
            CALL BORN(P1,NHEL(1,IHEL),IHEL,T,borns)
            DO I=0,NSQAMPSO
                ANS(1,I)=ANS(1,I)+T(1,I)
                ANS(2,I)=ANS(2,I)+T(2,I)
            ENDDO
            if ( borns(1,0).ne.0d0 .AND. .NOT. GOODHEL(IHEL,nFKSprocess) ) then
              GOODHEL(IHEL,nFKSprocess)=.TRUE.
            endif
            if ( borns(2,0).ne.0d0 .AND. .NOT. GOODHEL(IHEL+SKIP(nFKSprocess),nFKSprocess) ) then
              GOODHEL(IHEL+SKIP(nFKSprocess),nFKSprocess)=.TRUE.
            endif
	  ENDIF
        ENDIF
      ENDDO
      DO I=0,NSQAMPSO
      ANS(1,I)=ANS(1,I)/DBLE(IDEN)
      ANS(2,I)=ANS(2,I)/DBLE(IDEN)
      ENDDO
      calculatedBorn=.true.
      END
       
       
      SUBROUTINE BORN(P,NHEL,HELL,ANS,borns)
C  
%(info_lines)s
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)

%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   %(ngraphs)d,NEIGEN=  1) 
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      complex*16 imag1
      parameter (imag1 = (0d0,1d0))
      include "nexternal.inc"
      include "born_nhel.inc"
      include "coupl.inc"
      include "genps.inc"
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL-1),borns(2,0:NSQAMPSO)
      INTEGER NHEL(NEXTERNAL-1), HELL
      COMPLEX*16 ANS(2,0:NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N,ihel,back_hel,glu_ij
      INTEGER IC(NEXTERNAL-1),nmo
      parameter (nmo=nexternal-1)
      data ic /nmo*1/
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO), W(%(wavefunctionsize)d,NWAVEFUNCS), jamph(2, ncolor,nampso)
C  
C GLOBAL VARIABLES
C  
      Double Precision amp2(maxamps), jamp2(0:maxamps,NAMPSO)
      common/to_amps/  amp2,       jamp2
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      double precision hel_fac
      integer get_hel,skip(%(nconfs)d)
      common/cBorn/hel_fac,get_hel,skip
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer nfksprocess
      common/c_nfksprocess/nfksprocess

C
C FUNCTION
C
      INTEGER SQSOINDEXB%(proc_prefix)s

%(ij_lines)s
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
      glu_ij = ij_values(nfksprocess)
      DO I = 0, NSQAMPSO
        ANS(1,I)=0D0
        ANS(2,I)=0D0
        borns(1,I)=0d0
        borns(2,I)=0d0
      ENDDO
      back_hel = nhel(glu_ij)
      do ihel=-1,1,2
        if (ihel.eq.-1.or.nhel(glu_ij).ne.0) then
        if (nhel(glu_ij).ne.0) nhel(glu_ij) = ihel
        if (.not. calculatedBorn) then
%(helas_calls)s
        do i=1,ngraphs
          if(ihel.eq.-1)then
            saveamp(i,hell)=amp(i)
          elseif(ihel.eq.1)then
            saveamp(i,hell+skip(nFKSprocess))=amp(i)
          else
            write(*,*) "ERROR #1 in born.f"
            stop
          endif
        enddo
        elseif (calculatedBorn) then
        do i=1,ngraphs
          if(ihel.eq.-1)then
            amp(i)=saveamp(i,hell)
          elseif(ihel.eq.1)then
            amp(i)=saveamp(i,hell+skip(nFKSprocess))
          else
            write(*,*) "ERROR #1 in born.f"
            stop
          endif
        enddo
        endif
%(jamp_lines)s
	DO M = 1, NAMPSO
        DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
            ZTEMP = ZTEMP + CF(J,I)*JAMP(J,M)
          ENDDO
	  DO N = 1, NAMPSO
              BORNS(2-(1-ihel)/2,SQSOINDEXB%(proc_prefix)s(M,N))=BORNS(2-(1-ihel)/2,SQSOINDEXB%(proc_prefix)s(M,N))+ZTEMP*DCONJG(JAMP(I,N))/DENOM(I)   
          ENDDO
        ENDDO
        ENDDO
        Do I = 1, NGRAPHS
          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
        Enddo
        do J = 1,NAMPSO
        Do I = 1, NCOLOR
          Jamp2(i,J)=Jamp2(i,J)+Jamp(i,J)*dconjg(Jamp(i,J))
          Jamph(2-(1-ihel)/2,i,J)=Jamp(i,J)
        Enddo
        ENDDO
      endif
      Enddo
      do I = 1,NSQAMPSO
          ANS(1,I)=BORNS(1,I)+BORNS(2,I)
      ENDDO
      DO M = 1, NAMPSO
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
          ZTEMP = ZTEMP + CF(J,I)*JAMPH(2,J,M)
        ENDDO
	DO N = 1, NAMPSO
        ANS(2,SQSOINDEXB%(proc_prefix)s(M,N))= ANS(2,SQSOINDEXB%(proc_prefix)s(M,N)) + ZTEMP*DCONJG(JAMPH(1,I,N))/DENOM(I)
        ENDDO
      ENDDO
      ENDDO
      nhel(glu_ij) = back_hel
      END
       

      BLOCK DATA GOODHELS
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  %(ncomb)d )
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*%(nconfs)d)
      LOGICAL GOODHEL(NCOMB,%(nconfs)d)
      common /c_goodhel/goodhel
      DATA GOODHEL/THEL*.FALSE./
      end



C
C Helper functions to deal with the split orders.
C

      INTEGER FUNCTION SQSOINDEXB%(proc_prefix)s(AMPORDERA,AMPORDERB)
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
      INTEGER SQSOINDEXB%(proc_prefix)s_FROM_ORDERS
C
C BEGIN CODE
C
      DO I=1,NSPLITORDERS
	    SQORDERS(I)=AMPSPLITORDERS(AMPORDERA,I)+AMPSPLITORDERS(AMPORDERB,I)
	  ENDDO
	  SQSOINDEXB%(proc_prefix)s=SQSOINDEXB%(proc_prefix)s_FROM_ORDERS(SQORDERS)
	  END



      INTEGER FUNCTION SQSOINDEXB%(proc_prefix)s_FROM_ORDERS(ORDERS)
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
		SQSOINDEXB%(proc_prefix)s_FROM_ORDERS = I
		RETURN
1009    CONTINUE
	  ENDDO

	  WRITE(*,*) 'ERROR:: Stopping function sqsoindex_from_orders'
	  WRITE(*,*) 'Could not find squared orders ',(ORDERS(I),I=1,NSPLITORDERS)
	  STOP

	  END



      INTEGER FUNCTION GETORDPOWFROMINDEX_B%(proc_prefix)s(IORDER, INDX)
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
	  INTEGER IORDER, INDX
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

      IF (INDX.GT.NSQAMPSO.OR.INDX.LT.1) THEN
      WRITE(*,*) "INVALID INDX ", INDX
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSQAMPSO
      STOP
      ENDIF

      GETORDPOWFROMINDEX_B%(proc_prefix)s=SPLITORDERS(INDX, IORDER)
      END


      SUBROUTINE GET_NSQSO_BORN%(proc_prefix)s(NSQSO)
C
C     Simple subroutine returning the number of squared split order
C     contributions returned in ANS when calling SMATRIX_SPLITORDERS
C
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSQSO

	  NSQSO=NSQAMPSO

      END


