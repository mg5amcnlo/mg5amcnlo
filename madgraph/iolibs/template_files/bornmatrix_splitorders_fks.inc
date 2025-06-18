      SUBROUTINE SBORN(P,ANS_SUMMED)
C  
%(info_lines)s
C
C
C     Return the sum of the split orders which are required in orders.inc (BORN_ORDERS)
C     Also the values needed for the counterterms are stored in the C_BORN_CNT common block
C  
C  
%(process_lines)s
C
C  
C CONSTANTS
C 
      implicit none
      include 'nexternal.inc'
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
C  
C ARGUMENTS 
C 
      REAL*8 P(0:3,NEXTERNAL-1)
      double precision ans_summed
C
C VARIABLES
C
      INTEGER I,J,K
      include 'has_ewsudakov.inc'
      integer nord_integrated_born
      include 'orders.inc'
      COMPLEX*16 ANS(2,0:NSQAMPSO), ANS_CNT(2, NSPLITORDERS)
      LOGICAL KEEP_ORDER(NSQAMPSO), KEEP_ORDER_CNT(NSPLITORDERS, NSQAMPSO), FIRSTTIME
      data keep_order / NSQAMPSO * .TRUE. /
      logical keep_order_born(NSQAMPSO)
      common /c_keep_order_B/ keep_order_born
      common /c_keep_order_cnt/ keep_order_cnt 
      common /c_born_cnt/ ans_cnt
      integer ord_subtract
      data firsttime / .TRUE. /
      integer amp_orders(nsplitorders)
      double precision tiny
      parameter (tiny = 1d-12)
      double precision max_val
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real

      double precision amp2b(%(ngraphs)d), jamp2b(0:%(ncolor)d,0:nampso)
      common/to_amps_born/  amp2b,       jamp2b
      Double Precision amp2(%(ngraphs)d), jamp2(0:%(ncolor)d)
      common/to_amps/  amp2,       jamp2
      logical split_type_used(nsplitorders)
      common/to_split_type_used/split_type_used
      logical force_ijglu_zero
      common /to_force_ijglu/ force_ijglu_zero
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX_B
      integer orders_to_amp_split_pos
C
C BEGIN CODE
C
C look for orders which match the born order constraint 

force_ijglu_zero = .false.

if (firsttime) then
 do i = 1, nsqampso
C this is for the orders of the born to integrate
  do j = 1, nsplitorders
   if(GETORDPOWFROMINDEX_B(j, i) .gt. born_orders(j)) then
    keep_order(i) = .false.
    EXIT 
   endif
  enddo
  keep_order_born = keep_order

C this is for the orders of the counterterms
  do j = 1, nsplitorders
  keep_order_cnt(j,i) = .true.
   do k = 1, nsplitorders
    if (j.eq.k) then
     ord_subtract=2
    else
     ord_subtract=0
    endif
    if(GETORDPOWFROMINDEX_B(k, i) .gt. nlo_orders(k)-ord_subtract) then
     keep_order_cnt(j,i) = .false.
     EXIT 
    endif
   enddo
  enddo

  nord_integrated_born = 0
  if (keep_order(i)) then
   nord_integrated_born = nord_integrated_born + 1
   !write(*,*) 'BORN: keeping split order ', i
  !else
   !write(*,*) 'BORN: not keeping split order ', i
  endif
 enddo

C check that, if one include the sudakov corrections, there is at most one contribution to be integrated at LO
if (has_ewsudakov.and.nord_integrated_born.gt.1) then
 write(*,*) 'ERROR in born.f, cannot compute Sudakov EW corrections with more than one Born coupling combination to be integrated.Found', nord_integrated_born
 stop 1
endif

 do j = 1, nsplitorders
  write(*,*) 'counterterm S.O', j, ordernames(j)
  do i = 1, nsqampso
   if (keep_order_cnt(j,i)) then
    write(*,*) 'BORN: keeping split order', i
   else
    write(*,*) 'BORN: not keeping split order', i
   endif
  enddo
 enddo
 firsttime = .false.
endif

CALL SBORN_SPLITORDERS(P,ANS)

C the born to be integrated
ans_summed = 0d0
max_val = 0d0

C reset the amp_split array
amp_split(1:amp_split_size) = 0d0
amp_split_cnt(1:amp_split_size,1:2,1:nsplitorders) = dcmplx(0d0,0d0)

do i = 1, nsqampso
 max_val = max(max_val, abs(ans(1,I)))
enddo
do i = 1, nsqampso
 if (keep_order(i)) then
    ANS_SUMMED = ANS_SUMMED + ANS(1,I)
C   keep track of the separate pieces correspoinding to different coupling combinations
    do j = 1, nsplitorders
      amp_orders(j) = GETORDPOWFROMINDEX_B(j, i)
    enddo
    if(abs(ans(1,i)).gt.max_val*tiny) amp_split(orders_to_amp_split_pos(amp_orders)) = ans(1,i)
 endif
enddo
C this is to avoid fake non-zero contributions 
if (abs(ans_summed).lt.max_val*tiny) ans_summed=0d0

wgt_me_born=ans_summed

C fill the amp2 and jamp2 arrays
amp2(1:ngraphs)=amp2b(1:ngraphs)! amp2 just needs to be copyed

do i = 0, int(jamp2b(0,0))
jamp2(i)=0d0
do j = 1, nampso
c here sum all, this may be refined later
jamp2(i)=jamp2(i)+jamp2b(i,j)
enddo
enddo

%(skip_amp_cnt)s
C quantities for the counterterms
do j = 1, nsplitorders
 ans_cnt(1:2,j) = (0d0, 0d0)
 if (.not.split_type_used(j)) cycle
 do i = 1, nsqampso
  if (keep_order_cnt(j,i)) then 
   ANS_CNT(1,j) = ANS_CNT(1,J) + ANS(1,I)
   ANS_CNT(2,j) = ANS_CNT(2,J) + ANS(2,I)
C   keep track of the separate pieces also for counterterms
   do k = 1, nsplitorders
      amp_orders(k) = GETORDPOWFROMINDEX_B(k, i)
C take into account the fact that the counterterm for a given split order
C will be multiplied by the corresponding squared coupling
      if (k.eq.j) amp_orders(k) = amp_orders(k) + 2
   enddo
C this is to avoid fake non-zero contributions 
   if (abs(ans(1,I)).gt.max_val*tiny) amp_split_cnt(orders_to_amp_split_pos(amp_orders),1,j) = ans(1,I)
   if (abs(ans(2,I)).gt.max_val*tiny) amp_split_cnt(orders_to_amp_split_pos(amp_orders),2,j) = ans(2,I)
  endif
 enddo
C this is to avoid fake non-zero contributions 
 if (abs(ans_cnt(1,j)).lt.max_val*tiny) ans_cnt(1,j)=(0d0,0d0)
 if (abs(ans_cnt(2,j)).lt.max_val*tiny) ans_cnt(2,j)=(0d0,0d0)
enddo


 999   return
END





      SUBROUTINE SBORN_ONEHEL(P,NHEL,HELL,ANS_SUMMED)
C  
%(info_lines)s
C
C
C     Return the sum of the split orders which are required in orders.inc (BORN_ORDERS)
C     It works for a single set of helicities, passed by NHEL
C     Also the values needed for the counterterms are stored in the C_BORN_CNT common block
C  
C  
%(process_lines)s
C
C  
C CONSTANTS
C 
      implicit none
      include 'nexternal.inc'
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
C  
C ARGUMENTS 
C 
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1)
      integer HELL
      double precision ans_summed
C
C VARIABLES
C
      INTEGER I,J,K
      include 'orders.inc'
      COMPLEX*16 ANS(2,NSQAMPSO)
      REAL*8 borns(2,0:NSQAMPSO)
      LOGICAL KEEP_ORDER(NSQAMPSO)
      common /c_keep_order_B/ keep_order

      LOGICAL KEEP_ORDER_CNT(NSPLITORDERS, NSQAMPSO)
      common /c_keep_order_cnt/ keep_order_cnt 

      integer amp_orders(nsplitorders)
      double precision tiny
      parameter (tiny = 1d-12)
      double precision max_val
      include 'ewsudakov_haslo.inc'
      double complex amp_split_ewsud(amp_split_size)
      common /to_amp_split_ewsud/ amp_split_ewsud
      double complex amp_split_ewsud_LO2(amp_split_size)
      common /to_amp_split_ewsud_LO2/ amp_split_ewsud_LO2
      integer iden
      logical force_ijglu_zero
      common /to_force_ijglu/ force_ijglu_zero
      %(den_factor_lines)s
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX_B
      integer orders_to_amp_split_pos
C
C     stuff for the LO2
      integer lo2_orders(nsplitorders)
      logical orders_equal
      external orders_equal
C
C BEGIN CODE
C
C look for orders which match the born order constraint 

      iden=iden_values(nfksprocess)
force_ijglu_zero = .true.

CALL BORN(P,NHEL,HELL,ANS,borns)

ans_summed = 0d0
max_val = 0d0

C reset the amp_split array
amp_split_ewsud(1:amp_split_size) = dcmplx(0d0,0d0)
amp_split_ewsud_lo2(1:amp_split_size) = dcmplx(0d0,0d0)

do i = 1, nsqampso
 max_val = max(max_val, abs(ans(1,I)))
enddo

call get_lo2_orders(lo2_orders)

do i = 1, nsqampso
    if (keep_order(i)) then
     ANS_SUMMED = ans_summed + ANS(1,I) / iden
     do j = 1, nsplitorders
      amp_orders(j) = GETORDPOWFROMINDEX_B(j, i)
     enddo
     !amp_split_ewsud(orders_to_amp_split_pos(amp_orders)) = ans(1,I)
     if(abs(ans(1,i)).gt.max_val*tiny) amp_split_ewsud(orders_to_amp_split_pos(amp_orders)) = ans(1,i) / iden
    endif
C the following is for the LO2
    if (has_lo2) then
     do j = 1, nsplitorders
      amp_orders(j) = GETORDPOWFROMINDEX_B(j, i)
     enddo
     if (orders_equal(amp_orders, lo2_orders)) then
      if(abs(ans(1,i)).gt.max_val*tiny) amp_split_ewsud_lo2(orders_to_amp_split_pos(amp_orders)) = ans(1,i) / iden
     endif
    endif
enddo

C this is to avoid fake non-zero contributions 
if (abs(ans_summed).lt.max_val*tiny) ans_summed=0d0

  return
END




SUBROUTINE SBORN_SPLITORDERS(P1,ANS)
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
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  %(ncomb)d )
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
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
      COMPLEX*16 T(2,NSQAMPSO)
      INTEGER NHEL(NEXTERNAL-1,NCOMB)
%(helicity_lines)s
%(den_factor_lines)s
%(ij_lines)s
C  
C GLOBAL VARIABLES
C  
      Double Precision amp2(ngraphs), jamp2(0:%(ncolor)d,0:NAMPSO)
      common/to_amps_born/  amp2,       jamp2
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
      logical cond_ij
C ----------
C BEGIN CODE
C ----------
      iden=iden_values(nfksprocess)
      glu_ij = ij_values(nfksprocess)
      NTRY(nFKSprocess)=NTRY(nFKSprocess)+1
      if (NTRY(nFKSprocess).lt.2) then
       if (glu_ij.eq.0) then
         skip(nfksprocess)=0
       else
         skip(nFKSprocess)=1			 
         do while(nhel(glu_ij ,skip(nFKSprocess)).ne.-NHEL(GLU_IJ ,1))
            skip(nFKSprocess)=skip(nFKSprocess)+1
         enddo
         skip(nFKSprocess)=skip(nFKSprocess)-1
       endif
      endif
      DO JJ=1,NGRAPHS
          amp2(jj)=0d0
      ENDDO
      DO I=0,NAMPSO
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
        ! the following lines are to avoid segfaults when glu_ij=0
        cond_ij=skip(nfksprocess).eq.0
        if (.not.cond_ij) cond_ij=cond_ij.or.nhel(glu_ij,ihel).EQ.NHEL(GLU_IJ,1)
        !if (nhel(glu_ij,ihel).EQ.NHEL(GLU_IJ,1).or.skip(nfksprocess).eq.0) then
        if (cond_ij) then
          IF ((GOODHEL(IHEL,nFKSprocess) .OR. GOODHEL(IHEL+SKIP(nFKSprocess),nFKSprocess) .OR. NTRY(nFKSprocess) .LT. 2) ) THEN
            
            CALL BORN(P1,NHEL(1,IHEL),IHEL,T,borns)
            DO I=1,NSQAMPSO
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
      DO I=1,NSQAMPSO
      ANS(1,I)=ANS(1,I)/DBLE(IDEN)
      ANS(2,I)=ANS(2,I)/DBLE(IDEN)
      ANS(1,0)=ANS(1,0)+ANS(1,I)
      ANS(2,0)=ANS(2,0)+ANS(2,I)
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
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL-1),borns(2,0:NSQAMPSO)
      INTEGER NHEL(NEXTERNAL-1), HELL
      COMPLEX*16 ANS(2,NSQAMPSO)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,M,N,ihel,back_hel,glu_ij
      INTEGER IC(NEXTERNAL-1),nmo
      parameter (nmo=nexternal-1)
      data ic /nmo*1/
      INTEGER CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER CF_INDEX, DENOM   
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO), W(%(wavefunctionsize)d,NWAVEFUNCS), jamph(2, ncolor,nampso)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
C  
C GLOBAL VARIABLES
C  
      Double Precision amp2(%(ngraphs)d), jamp2(0:%(ncolor)d,0:nampso)
      common/to_amps_born/  amp2,       jamp2
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      double precision hel_fac
      integer get_hel,skip(%(nconfs)d)
      common/cBorn/hel_fac,get_hel,skip
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
      integer step_hel
      logical cond_ij
      logical force_ijglu_zero
      common /to_force_ijglu/ force_ijglu_zero

C
C FUNCTION
C
      INTEGER SQSOINDEXB

%(ij_lines)s
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
      jamp(:,:) = (0d0,0d0)
      glu_ij = ij_values(nfksprocess)
      if (force_ijglu_zero) glu_ij = 0

      DO I = 1, NSQAMPSO
        ANS(1,I)=0D0
        ANS(2,I)=0D0
        borns(1,I)=0d0
        borns(2,I)=0d0
      ENDDO
      borns(1,0)=0d0
      borns(2,0)=0d0
      if (glu_ij.ne.0) then
        back_hel = nhel(glu_ij)
        if (back_hel.ne.0) then
          step_hel=-2*back_hel
        else
          step_hel=1
        endif
      else
        back_hel=0
        step_hel=1
      endif
      DO IHEL=back_hel,-back_hel,step_hel
        if (glu_ij.ne.0) then
          cond_ij=IHEL.EQ.back_hel.OR.NHEL(GLU_IJ).NE.0
        else
          cond_ij=IHEL.EQ.back_hel 
        endif
        IF (cond_ij) THEN
        if (glu_ij.ne.0) then
          if (nhel(glu_ij).ne.0) nhel(glu_ij) = ihel
        endif
        if (.not. calculatedBorn) then
%(helas_calls)s
        do i=1,ngraphs
          if(ihel.eq.back_hel)then
            saveamp(i,hell)=amp(i)
          elseif(ihel.eq.-back_hel)then
            saveamp(i,hell+skip(nFKSprocess))=amp(i)
          else
            write(*,*) "ERROR #1 in born.f"
            stop
          endif
        enddo
        elseif (calculatedBorn) then
        do i=1,ngraphs
          if(ihel.eq.back_hel)then
            amp(i)=saveamp(i,hell)
          elseif(ihel.eq.-back_hel)then
            amp(i)=saveamp(i,hell+skip(nFKSprocess))
          else
            write(*,*) "ERROR #1 in born.f"
            stop
          endif
        enddo
        endif
%(jamp_lines)s
	DO M = 1, NAMPSO
        CF_INDEX = 0
        DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
            CF_INDEX = CF_INDEX + 1
            ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J,M)
          ENDDO
	  DO N = 1, NAMPSO
              BORNS(2-(1+back_hel*ihel)/2,SQSOINDEXB(M,N))=BORNS(2-(1+back_hel*ihel)/2,SQSOINDEXB(M,N))+ZTEMP*DCONJG(JAMP(I,N))   
          ENDDO
        ENDDO
        DO N = 1, NAMPSO
             BORNS(2-(1+back_hel*ihel)/2,SQSOINDEXB(M,N))=BORNS(2-(1+back_hel*ihel)/2,SQSOINDEXB(M,N))/DENOM
        ENDDO
        ENDDO
        Do I = 1, NGRAPHS
          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
        Enddo
        do J = 1,NAMPSO
        Do I = 1, NCOLOR
          Jamp2(i,J)=Jamp2(i,J)+Jamp(i,J)*dconjg(Jamp(i,J))
          Jamph(2-(1+back_hel*ihel)/2,i,J)=Jamp(i,J)
        Enddo
        ENDDO
      endif
      Enddo
      do i = 1, nsqampso
      borns(1,0)=borns(1,0)+borns(1,i)
      borns(2,0)=borns(2,0)+borns(2,i)
      ans(1,i) = borns(1,i) + borns(2,i) 
      enddo
      DO M = 1, NAMPSO
      CF_INDEX = 0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = I, NCOLOR
          CF_INDEX = CF_INDEX +1
          ZTEMP = ZTEMP + CF(CF_INDEX)*JAMPH(2,J,M)
        ENDDO
	DO N = 1, NAMPSO
        ANS(2,SQSOINDEXB(M,N))= ANS(2,SQSOINDEXB(M,N)) + ZTEMP*DCONJG(JAMPH(1,I,N))
        ENDDO
      ENDDO

      DO N = 1, NAMPSO
      ANS(2,SQSOINDEXB(M,N))= ANS(2,SQSOINDEXB(M,N))/DENOM
      ENDDO
      ENDDO
      if (glu_ij.ne.0) nhel(glu_ij) = back_hel
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

      INTEGER FUNCTION SQSOINDEXB(AMPORDERA,AMPORDERB)
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
      INTEGER SQSOINDEXB_FROM_ORDERS
C
C BEGIN CODE
C
      DO I=1,NSPLITORDERS
	    SQORDERS(I)=AMPSPLITORDERS(AMPORDERA,I)+AMPSPLITORDERS(AMPORDERB,I)
	  ENDDO
	  SQSOINDEXB=SQSOINDEXB_FROM_ORDERS(SQORDERS)
	  END



      INTEGER FUNCTION SQSOINDEXB_FROM_ORDERS(ORDERS)
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
		SQSOINDEXB_FROM_ORDERS = I
		RETURN
1009    CONTINUE
	  ENDDO

	  WRITE(*,*) 'ERROR:: Stopping function sqsoindex_from_orders'
	  WRITE(*,*) 'Could not find squared orders ',(ORDERS(I),I=1,NSPLITORDERS)
	  STOP

	  END



      INTEGER FUNCTION GETORDPOWFROMINDEX_B(IORDER, INDX)
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
      WRITE(*,*) "INVALID IORDER B", IORDER
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSPLITORDERS
      STOP
      ENDIF

      IF (INDX.GT.NSQAMPSO.OR.INDX.LT.1) THEN
      WRITE(*,*) "INVALID INDX B", INDX
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSQAMPSO
      STOP
      ENDIF

      GETORDPOWFROMINDEX_B=SQSPLITORDERS(INDX, IORDER)
      END


      SUBROUTINE GET_NSQSO_B(NSQSO)
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


