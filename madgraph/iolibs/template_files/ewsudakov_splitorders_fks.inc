      SUBROUTINE EWSDK_ME_%(ime)d(P,NHEL,ANS_SUMMED)
C  
%(info_lines)s
C
C
C     Return the sum of the split orders which are required in orders.inc 
C     With respect to those at the born, the QED power is increased by 2
C     Note that helicities must be passed as an input argument.
C
C  
%(process_lines)s
C
C  
C CONSTANTS
C 
implicit none
      include 'nexternal.inc'
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C  
C ARGUMENTS 
C 
      REAL*8 P(0:3,NEXTERNAL-1) 
      double complex ANS_SUMMED
      integer nhel(nexternal-1)
C
C VARIABLES
C
      INTEGER I,j
      include 'orders.inc'
      double complex ANS(NSQAMPSO)
      LOGICAL KEEP_ORDER(NSQAMPSO)
      save KEEP_ORDER
      integer amp_orders(nsplitorders)
      logical firsttime
      data firsttime / .TRUE. /
      double precision tiny
      parameter (tiny = 1d-12)
      double precision max_val
      double complex amp_split_ewsud(amp_split_size)
      common /to_amp_split_ewsud/ amp_split_ewsud
      integer iden
      parameter (iden = %(den_factor)d)
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX_B
      integer orders_to_amp_split_pos
C
C BEGIN CODE
C

if (firsttime) then
 do i = 1, nsqampso
  keep_order(i) = .true.
C this is for the orders of the born to integrate
  do j = 1, nsplitorders
   if(GETORDPOWFROMINDEX_B(j, i) .gt. born_orders(j)) then
    keep_order(i) = .false.
    EXIT 
   endif
  enddo

  !if (keep_order(i)) then
  ! write(*,*) 'SDK%(ime)d: keeping split order ', i
  !else
  ! write(*,*) 'SDK%(ime)d: not keeping split order ', i
  !endif
 enddo

 firsttime = .false.
endif

      CALL EWSUDAKOV_ME_%(ime)d_SPLITORDERS(P,NHEL,ANS)
      ans_summed = (0d0,0d0)
      max_val = 0d0

C reset the amp_split_ewsud array
amp_split_ewsud(1:amp_split_size) = dcmplx(0d0,0d0)


do i = 1, nsqampso
 max_val = max(max_val, abs(ans(I)))
enddo

      do i = 1, nsqampso
        if (keep_order(i)) then
         ANS_SUMMED = ans_summed + ANS(I)
         do j = 1, nsplitorders
          amp_orders(j) = GETORDPOWFROMINDEX_B(j, i)
         enddo
         !amp_split_ewsud(orders_to_amp_split_pos(amp_orders)) = ans(I)
         if(abs(ans(i)).gt.max_val*tiny) amp_split_ewsud(orders_to_amp_split_pos(amp_orders)) = ans(i) / iden
        endif
      enddo

C this is to avoid fake non-zero contributions 
if (abs(ans_summed).lt.max_val*tiny) ans_summed=0d0

return
END


       
       
      SUBROUTINE EWSUDAKOV_ME_%(ime)d_SPLITORDERS(p,NHEL,ANS)
C  
%(info_lines)s
C RETURNS THE INTERFERENCE OF THE BORN DIAGRAMS WITH THOSE
C NEEDED FOR THE EVALUATION OF THE EW CORRECTIONS IN THE
C SUDAKOV APPROCIMATION,
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)

%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER     NGRAPHS1, NGRAPHS2 ! graphs for the Born and Sudakov amplitude
      PARAMETER ( NGRAPHS1=%(ngraphs1)d, NGRAPHS2=%(ngraphs2)d) 
      INTEGER NCOLOR1, NCOLOR2
      PARAMETER (NCOLOR1=%(ncolor1)d, NCOLOR2=%(ncolor2)d) 
      INTEGER NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      complex*16 imag1
      parameter (imag1 = (0d0,1d0))
      include "nexternal.inc"
      include "born_nhel.inc"
      INCLUDE "coupl.inc"
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1)
      double complex ANS(NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N
      INTEGER IC(NEXTERNAL-1),NMO
      PARAMETER (NMO=NEXTERNAL-1)
      DATA IC /NMO*1/
      INTEGER DENOM
      INTEGER CF(NCOLOR2,NCOLOR1)
      COMPLEX*16 ZTEMP, AMP1(NGRAPHS1), AMP2(NGRAPHS2), JAMP1(NCOLOR1,NAMPSO), JAMP2(NCOLOR2,NAMPSO), W(%(wavefunctionsize)d,NWAVEFUNCS)
      COMPLEX*16 TMP_JAMP1(%(nb_temp_jamp1)i)
      COMPLEX*16 TMP_JAMP2(%(nb_temp_jamp2)i)
      real*8 p_save(0:3, nexternal-1), p_resh(0:3, nexternal-1)
      logical pass_reshuffle
C CCC THIS IS PROBABLY UNNECESSARY, KEEP COMMENTED FOR THE TIME BEING
C  Imag power takes into account the I factor when a longitudinal Z
C   boson is replaced by its goldstone (Chi). 
C   It is the difference of the number of Chi's in the first and second amplitude which are interferred together
C       integer imag_power
C       data imag_power / %(imag_power)d /

C  
C GLOBAL VARIABLES
C  
C
C FUNCTION
C
      INTEGER SQSOINDEXB
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
C Helas calls and color basis for the base amplitude
%(helas_calls1)s
%(jamp1_lines)s

    DO I = 1, NSQAMPSO
    ANS(I) = (0d0,0d0)
    ENDDO

C Reshuffle the momenta here
p_save(:,:) = p(:,:)
%(reshuffle_calls)s
if (.not.pass_reshuffle) then
C restore the momenta and just quit the function
p(:,:) = p_save(:,:)
return
endif

C Helas calls and color basis for the Sudakov amplitude
%(helas_calls2)s
%(jamp2_lines)s

C Finally interfere the two sets of color-stripped amplitudes
    DO M = 1, NAMPSO
      DO I = 1, NCOLOR1
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR2
              ZTEMP = ZTEMP + CF(j,i)*JAMP2(J,M)
          ENDDO
	  DO N = 1, NAMPSO
              ANS(SQSOINDEXB(M,N))=ANS(SQSOINDEXB(M,N))+ZTEMP*DCONJG(JAMP1(I,N))/DENOM
          ENDDO
      ENDDO
    ENDDO
C C NOT NEEDED
C  include the imaginary factor if needed
C     if (imag_power.ne.0) ans(:) = ans(:) * imag1**imag_power 

    p(:,:) = p_save(:,:)
    END
       
       
